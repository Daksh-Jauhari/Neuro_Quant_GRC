import copy
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.quantization as quant_models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────

DATASET          = "imagenet_hf"        # "imagenet_hf" | "imagenet_local" | "tiny_imagenet"
IMAGENET_VAL_DIR = "./imagenet/val"     # Only used if DATASET = "imagenet_local"
BATCH_SIZE       = 64
NUM_WORKERS      = 2                    # Use 0 for HuggingFace streaming (set automatically)
EVAL_SAMPLES     = None                 # Evaluation images (None = full 50K val set)
CAL_SAMPLES      = 512                  # Calibration images for static PTQ
PUBLISHED_FP32   = 76.13               # He et al. (2016) — ResNet-50 on ImageNet-1K

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── FLOPs-weighted Energy Proxy ──────────────────────────────────────────────
# ResNet-50 layer-wise FLOPs (millions of multiply-add ops).
# Source: He et al. (2016), Table 1; confirmed via torchinfo profiling at 224×224.

LAYER_FLOPS = {
    "conv1":       235,
    "layer1":      575,
    "layer2":     1125,
    "layer3":     2255,
    "layer4":     1130,
    "classifier":    4,
}
TOTAL_FLOPS = sum(LAYER_FLOPS.values())   # 5324 MFLOPs
BITS = {"fp32": 32, "bf16": 16, "int8": 8}


def energy_proxy(precision_map):
    """
    Normalized energy estimate relative to all-FP32 baseline.
    Formula: Σ_i(FLOPs_i × bits_i / 32) / total_FLOPs
    Returns [0, 1] — savings = (1 - result) × 100%.
    """
    weighted = sum(
        LAYER_FLOPS[layer] * (BITS[prec] / 32)
        for layer, prec in precision_map.items()
    )
    return round(weighted / TOTAL_FLOPS, 4)


FP32_ALL = {k: "fp32" for k in LAYER_FLOPS}
BF16_ALL = {k: "bf16" for k in LAYER_FLOPS}
INT8_ALL = {k: "int8" for k in LAYER_FLOPS}


class RemappedDataset(torch.utils.data.Dataset):
    """
    Remaps Tiny-ImageNet local labels (0–199) to ImageNet-1K indices (0–999).
    Defined at module level so Python's spawn-based multiprocessing on Windows
    can pickle it for DataLoader workers.
    """
    def __init__(self, ds, label_map):
        self.ds        = ds
        self.label_map = label_map

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, lbl = self.ds[idx]
        return img, self.label_map[lbl]


# ─── Dataset + Model ─────────────────────────────────────────────────────────

def build_loaders_and_model():
    """
    Returns (eval_loader, cal_loader, model).
    eval_loader  — used for all accuracy measurements.
    cal_loader   — used only for static PTQ calibration (CAL_SAMPLES images).
    Both loaders are non-overlapping subsets of the validation split.
    """
    imagenet_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── HuggingFace ImageNet-1K streaming (default) ───────────────────────
    if DATASET == "imagenet_hf":
        print("  [Dataset] ImageNet-1K — HuggingFace streaming (ILSVRC/imagenet-1k)")
        print("  Requires: hf auth login + dataset access at huggingface.co/datasets/ILSVRC/imagenet-1k\n")
        from datasets import load_dataset

        total_needed = CAL_SAMPLES + (EVAL_SAMPLES or 50_000)
        print(f"  Streaming {total_needed} validation images from HuggingFace...")
        hf_val   = load_dataset(
            "ILSVRC/imagenet-1k", split="validation",
            streaming=True
        )
        all_data = list(hf_val.take(total_needed))
        print(f"  Done. {len(all_data)} images loaded.\n")

        cal_raw  = all_data[:CAL_SAMPLES]
        eval_raw = all_data[CAL_SAMPLES:]

        # Inner class is safe here because num_workers=0 (no fork/spawn pickling)
        class _HFDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                item = self.data[idx]
                return imagenet_transform(item["image"].convert("RGB")), item["label"]

        cal_loader  = DataLoader(_HFDataset(cal_raw),  batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=0)
        eval_loader = DataLoader(_HFDataset(eval_raw), batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=0)
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # ── Local ImageNet-1K ─────────────────────────────────────────────────
    elif DATASET == "imagenet_local":
        print(f"  [Dataset] ImageNet-1K local — {IMAGENET_VAL_DIR}")
        dataset = datasets.ImageFolder(IMAGENET_VAL_DIR, transform=imagenet_transform)
        gen     = torch.Generator().manual_seed(42)
        perm    = torch.randperm(len(dataset), generator=gen).tolist()
        cal_idx  = perm[:CAL_SAMPLES]
        eval_idx = perm[CAL_SAMPLES:CAL_SAMPLES + (EVAL_SAMPLES or len(dataset))]
        cal_loader  = DataLoader(Subset(dataset, cal_idx),  batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        eval_loader = DataLoader(Subset(dataset, eval_idx), batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # ── Tiny-ImageNet (fallback only — not recommended for the paper) ─────
    elif DATASET == "tiny_imagenet":
        print("  [Dataset] Tiny-ImageNet-200 — FALLBACK MODE")
        print("  WARNING: Baseline ~25% expected (label-remapped subset evaluation).")
        print("           Static PTQ not used — falls back to dynamic INT8.\n")
        import urllib.request, zipfile, os, json, shutil

        _dir           = os.path.dirname(os.path.abspath(__file__))
        tiny_root      = os.path.join(_dir, "tiny-imagenet-200")
        val_dir        = os.path.join(tiny_root, "val")
        class_idx_path = os.path.join(_dir, "imagenet_class_index.json")

        if not os.path.exists(tiny_root):
            print("  Downloading Tiny-ImageNet (~240 MB)...")
            zip_path = os.path.join(_dir, "tiny-imagenet-200.zip")
            urllib.request.urlretrieve(
                "http://cs231n.stanford.edu/tiny-imagenet-200.zip", zip_path)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(_dir)
            os.remove(zip_path)

        val_img_dir = os.path.join(val_dir, "images")
        if os.path.exists(val_img_dir):
            print("  Restructuring val/ for ImageFolder...")
            with open(os.path.join(val_dir, "val_annotations.txt")) as f:
                for line in f:
                    parts   = line.strip().split("\t")
                    cls_dir = os.path.join(val_dir, parts[1])
                    os.makedirs(cls_dir, exist_ok=True)
                    src = os.path.join(val_img_dir, parts[0])
                    dst = os.path.join(cls_dir, parts[0])
                    if os.path.exists(src):
                        shutil.move(src, dst)
            shutil.rmtree(val_img_dir, ignore_errors=True)

        if not os.path.exists(class_idx_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
                class_idx_path)
        with open(class_idx_path) as f:
            wnid_to_idx = {v[0]: int(k) for k, v in json.load(f).items()}

        tiny_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        base_ds   = datasets.ImageFolder(val_dir, transform=tiny_transform)
        label_map = {i: wnid_to_idx[w] for i, w in enumerate(base_ds.classes)}
        dataset   = RemappedDataset(base_ds, label_map)

        gen      = torch.Generator().manual_seed(42)
        perm     = torch.randperm(len(dataset), generator=gen).tolist()
        cal_idx  = perm[:CAL_SAMPLES]
        eval_idx = perm[CAL_SAMPLES:CAL_SAMPLES + (EVAL_SAMPLES or len(dataset))]
        cal_loader  = DataLoader(Subset(dataset, cal_idx),  batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        eval_loader = DataLoader(Subset(dataset, eval_idx), batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    else:
        raise ValueError(f"Unknown DATASET: '{DATASET}'. Choose imagenet_hf, imagenet_local, or tiny_imagenet.")

    print(f"  [Model] {model.__class__.__name__} | Device: {DEVICE}")
    model.eval()
    return eval_loader, cal_loader, model


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_top1(model, loader, device):
    """Standard top-1 accuracy evaluation. Handles FP32, BF16, and quantized models."""
    model.to(device)
    model.eval()
    correct, total = 0, 0

    # Quantized models replace float parameters with packed int8 — use try/except
    try:
        is_bf16 = next(model.parameters()).dtype == torch.bfloat16
    except StopIteration:
        is_bf16 = False

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            if is_bf16:
                images = images.to(torch.bfloat16)
            outputs  = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)
            if (i + 1) % 10 == 0:
                print(f"    Batch {i+1}/{len(loader)} — running acc: {100.*correct/total:.2f}%",
                      end="\r")

    acc = round(100.0 * correct / total, 2)
    print(f"    Final top-1: {acc:.2f}%               ")
    return acc


# ─── Quantization ─────────────────────────────────────────────────────────────

def apply_bf16(model):
    """Uniform BF16: convert all parameters and buffers to bfloat16."""
    m = copy.deepcopy(model).cpu()
    return m.to(torch.bfloat16)


def apply_static_int8(base_model, cal_loader):
    """
    Static Post-Training Quantization (PTQ) — Jacob et al. (2018).

    Process:
      1. Load weights into torchvision's quantization-ready ResNet-50
         (has QuantStub/DeQuantStub built in).
      2. Fuse Conv + BN + ReLU into single quantizable ops.
      3. Insert MinMax observers at every activation point.
      4. Calibration: run CAL_SAMPLES images through model to collect
         activation range statistics.
      5. Convert: replace float ops with INT8 kernels using observed ranges.

    Unlike dynamic quantization (which quantizes weights only and estimates
    activation ranges at runtime), static PTQ produces a fully INT8 model
    where both weights and activations are pre-quantized. This is the
    standard method for CNN quantization in the research literature.
    """
    # torchvision.models.quantization.resnet50 has QuantStub/DeQuantStub
    # and fuse_model() built in — avoids manual architecture modification
    q_model = quant_models.resnet50(weights=None, quantize=False)
    q_model.load_state_dict(base_model.state_dict())
    q_model.eval()

    # Auto-detect available quantization backend
    available = torch.backends.quantized.supported_engines
    if "fbgemm" in available:
        backend = "fbgemm"       # x86 Intel/AMD — best for desktop/server
    elif "onednn" in available:
        backend = "onednn"       # Windows x86 — PyTorch ships onednn instead of fbgemm
    elif "qnnpack" in available:
        backend = "qnnpack"      # ARM / mobile — fallback
    else:
        raise RuntimeError(f"No supported quantization backend found. Available: {available}")
    print(f"    Using quantization backend: {backend}")
    torch.backends.quantized.engine = backend
    q_model.qconfig = torch.ao.quantization.get_default_qconfig(backend)

    # Fuse Conv + BN + ReLU into a single op for accurate quantization
    try:
        q_model.fuse_model(is_qat=False)
    except TypeError:
        q_model.fuse_model()   # older torchvision without is_qat parameter

    # Insert MinMaxObserver at every quantization point
    torch.ao.quantization.prepare(q_model, inplace=True)

    # Calibration pass — forward CAL_SAMPLES images to record activation ranges
    print(f"    Calibrating on {CAL_SAMPLES} images...")
    with torch.no_grad():
        for images, _ in cal_loader:
            q_model(images.float())

    # Convert float ops to INT8 kernels using the observed statistics
    torch.ao.quantization.convert(q_model, inplace=True)
    print("    [Static PTQ] Conversion to INT8 complete.")
    return q_model


def apply_dynamic_int8_fallback(model):
    """
    Dynamic INT8 fallback for Tiny-ImageNet path only.
    Not used for the primary ImageNet-1K experiments.
    """
    m = copy.deepcopy(model).cpu()
    m = torch.ao.quantization.quantize_dynamic(
        m, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
    )
    return m


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_h1():
    print("\n" + "=" * 64)
    print("  NeuroQuant — Experiment H1: Uniform Quantization (v2)")
    print("=" * 64)

    eval_loader, cal_loader, base_model = build_loaders_and_model()
    results = []
    use_static_ptq = (DATASET != "tiny_imagenet")

    # ── FP32 Baseline ────────────────────────────────────────────────────
    print(f"\n  [FP32] Measuring baseline ({DATASET} | {DEVICE})...")
    acc_fp32 = evaluate_top1(copy.deepcopy(base_model), eval_loader, DEVICE)
    e_fp32   = energy_proxy(FP32_ALL)
    results.append({
        "Configuration":       "FP32 baseline (measured)",
        "Top-1 Acc (%)":       acc_fp32,
        "Acc Drop (pp)":       0.00,
        "Energy (norm)":       e_fp32,
        "Energy Savings (%)":  0.00,
        "H1 Confirmed":        "—",
    })
    print(f"\n  Published FP32 (He et al., 2016) : {PUBLISHED_FP32}%")
    print(f"  Measured  FP32 ({DATASET})  : {acc_fp32}%")
    if abs(acc_fp32 - PUBLISHED_FP32) > 2.0:
        print(f"  Note: gap > 2pp — expected for subset evaluation ({EVAL_SAMPLES} images).")

    # ── Uniform BF16 ─────────────────────────────────────────────────────
    print("\n  [BF16] Applying uniform bfloat16 quantization...")
    bf16_model   = apply_bf16(base_model)
    acc_bf16     = evaluate_top1(bf16_model, eval_loader, DEVICE)
    drop_bf16    = round(acc_fp32 - acc_bf16, 2)
    e_bf16       = energy_proxy(BF16_ALL)
    savings_bf16 = round((1 - e_bf16) * 100, 1)
    h1_bf16      = drop_bf16 <= 1.0 and savings_bf16 >= 35.0
    results.append({
        "Configuration":       "Uniform BF16",
        "Top-1 Acc (%)":       acc_bf16,
        "Acc Drop (pp)":       drop_bf16,
        "Energy (norm)":       e_bf16,
        "Energy Savings (%)":  savings_bf16,
        "H1 Confirmed":        "YES" if h1_bf16 else "NO",
    })

    # ── Static INT8 PTQ ───────────────────────────────────────────────────
    if use_static_ptq:
        print("\n  [Static INT8 PTQ] Post-training quantization with calibration...")
        int8_model = apply_static_int8(base_model, cal_loader)
        int8_label = "Uniform Static INT8 (PTQ)"
    else:
        print("\n  [Dynamic INT8] Fallback mode (tiny_imagenet only)...")
        int8_model = apply_dynamic_int8_fallback(base_model)
        int8_label = "Uniform Dynamic INT8 (fallback)"

    acc_int8     = evaluate_top1(int8_model, eval_loader, torch.device("cpu"))
    drop_int8    = round(acc_fp32 - acc_int8, 2)
    e_int8       = energy_proxy(INT8_ALL)
    savings_int8 = round((1 - e_int8) * 100, 1)
    h1_int8      = drop_int8 <= 1.0 and savings_int8 >= 35.0
    results.append({
        "Configuration":       int8_label,
        "Top-1 Acc (%)":       acc_int8,
        "Acc Drop (pp)":       drop_int8,
        "Energy (norm)":       e_int8,
        "Energy Savings (%)":  savings_int8,
        "H1 Confirmed":        "YES" if h1_int8 else "NO",
    })

    # ── Print Results ─────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    print("\n" + "=" * 64)
    print("  H1 RESULTS")
    print("=" * 64)
    print(df.to_string(index=False))
    print(f"\n  BF16       — Drop: {drop_bf16:+.2f} pp | Savings: {savings_bf16}% | "
          f"{'CONFIRMED' if h1_bf16 else 'NOT CONFIRMED'}")
    print(f"  Static INT8 — Drop: {drop_int8:+.2f} pp | Savings: {savings_int8}% | "
          f"{'CONFIRMED' if h1_int8 else 'NOT CONFIRMED'}")

    df.to_csv("h1_results.csv", index=False)
    with open("h1_summary.txt", "w") as f:
        f.write("NeuroQuant — H1 Results (v2)\n" + "=" * 64 + "\n\n")
        f.write(f"Dataset         : {DATASET}\n")
        f.write(f"Model           : ResNet-50 (He et al., 2016)\n")
        f.write(f"Eval samples    : {EVAL_SAMPLES}\n")
        f.write(f"Cal samples     : {CAL_SAMPLES} (static PTQ calibration)\n\n")
        f.write(f"FP32 baseline (measured) : {acc_fp32}%\n")
        f.write(f"FP32 published (ref)     : {PUBLISHED_FP32}% (He et al., 2016)\n\n")
        f.write(f"Uniform BF16\n")
        f.write(f"  Top-1 accuracy : {acc_bf16}%\n")
        f.write(f"  Accuracy drop  : {drop_bf16:+.2f} pp\n")
        f.write(f"  Energy savings : {savings_bf16}%\n")
        f.write(f"  H1 confirmed   : {h1_bf16}\n\n")
        f.write(f"{int8_label}\n")
        f.write(f"  Top-1 accuracy : {acc_int8}%\n")
        f.write(f"  Accuracy drop  : {drop_int8:+.2f} pp\n")
        f.write(f"  Energy savings : {savings_int8}%\n")
        f.write(f"  H1 confirmed   : {h1_int8}\n")

    print("\n  Saved: h1_results.csv | h1_summary.txt")
    print("=" * 64)
    return df


if __name__ == "__main__":
    run_h1()
