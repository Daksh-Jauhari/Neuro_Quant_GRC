"""
NeuroQuant — Experiment H2: Layer-wise Sensitivity Analysis
============================================================
Tests whether quantization sensitivity increases monotonically with layer
depth in ResNet-50, mirroring the V1->V2->V4->IT precision gradient in the
primate visual cortex.

Hypothesis H2:
    Quantizing earlier layers (conv1, layer1) to INT8 will cause smaller
    accuracy drops than quantizing later layers (layer3, layer4, classifier),
    and this sensitivity will increase monotonically with depth.

Method:
    For each of the six ResNet-50 layer groups, apply INT8 dynamic
    quantization to ONLY that group while all other groups remain at FP32.
    Measure the top-1 accuracy drop vs the FP32 baseline for each group.
    Assess whether the drop values increase monotonically with depth.

Setup:
    pip install -r requirements.txt
    huggingface-cli login

Run:
    python experiment_h2_layer_sensitivity.py

Outputs:
    h2_results.csv   — full layer sensitivity table
    h2_summary.txt   — human-readable summary with monotonicity verdict
"""

import copy
import warnings
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────

DATASET          = "imagenet_hf"   # "imagenet_hf" | "imagenet_local"
IMAGENET_VAL_DIR = "./imagenet/val"
BATCH_SIZE       = 64
EVAL_SAMPLES     = 5000            # Keep identical to H1 for fair comparison
PUBLISHED_FP32   = 76.13           # He et al. (2016) ResNet-50 reference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Layer Definitions ────────────────────────────────────────────────────────
# Maps experiment layer names → ResNet-50 attribute names → cortex analogues.
# Order here is the order of depth — conv1 is earliest, classifier is deepest.

LAYERS = [
    ("conv1",      "conv1",  "V1 — edges & orientations"),
    ("layer1",     "layer1", "V1/V2 — local features"),
    ("layer2",     "layer2", "V2/V4 — shapes & textures"),
    ("layer3",     "layer3", "V4 — complex patterns"),
    ("layer4",     "layer4", "IT cortex — object parts"),
    ("classifier", "fc",     "Decision — object identity"),
]

# ─── FLOPs-weighted Energy Proxy ──────────────────────────────────────────────

LAYER_FLOPS = {
    "conv1":       235,
    "layer1":      575,
    "layer2":     1125,
    "layer3":     2255,
    "layer4":     1130,
    "classifier":    4,
}
TOTAL_FLOPS = sum(LAYER_FLOPS.values())


def energy_proxy_single(layer_name):
    """
    Energy proxy when only one layer is INT8 and all others are FP32.
    Savings = FLOPs_target * (1 - 8/32) / total_FLOPs * 100
    """
    savings = LAYER_FLOPS[layer_name] * (1 - 8 / 32) / TOTAL_FLOPS * 100
    return round(savings, 2)


# ─── Dataset + Model ─────────────────────────────────────────────────────────

def build_loader_and_model():
    """Returns (eval_loader, model). Identical setup to H1 for comparability."""

    imagenet_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if DATASET == "imagenet_hf":
        print("  [Dataset] ImageNet-1K — HuggingFace streaming")
        from datasets import load_dataset

        print(f"  Streaming {EVAL_SAMPLES} validation images...")
        hf_val   = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
        all_data = list(hf_val.take(EVAL_SAMPLES))
        print(f"  Loaded {len(all_data)} images.\n")

        class _HFDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                item = self.data[idx]
                return imagenet_transform(item["image"].convert("RGB")), item["label"]

        loader = DataLoader(_HFDataset(all_data), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    elif DATASET == "imagenet_local":
        import torchvision.datasets as datasets
        from torch.utils.data import Subset
        print(f"  [Dataset] ImageNet-1K local — {IMAGENET_VAL_DIR}")
        dataset = datasets.ImageFolder(IMAGENET_VAL_DIR, transform=imagenet_transform)
        gen = torch.Generator().manual_seed(42)
        idx = torch.randperm(len(dataset), generator=gen)[:EVAL_SAMPLES].tolist()
        loader = DataLoader(Subset(dataset, idx), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)
    else:
        raise ValueError(
            f"Unknown DATASET: {DATASET}. "
            "Use imagenet_hf or imagenet_local. "
            "tiny_imagenet is not supported for H2."
        )

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    print(f"  [Model] ResNet-50 | Device: {DEVICE}")
    model.eval()
    return loader, model


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_top1(model, loader, device, label=""):
    model.to(device)
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs  = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)
            if (i + 1) % 10 == 0:
                print(f"    {label}Batch {i+1}/{len(loader)} "
                      f"— acc: {100.*correct/total:.2f}%", end="\r")

    acc = round(100.0 * correct / total, 2)
    print(f"    {label}Final top-1: {acc:.2f}%               ")
    return acc


# ─── Partial Quantization ─────────────────────────────────────────────────────

def quantize_single_layer(base_model, layer_name, attr_name):
    """
    Returns a copy of base_model where ONLY the specified layer group is
    quantized to INT8 via dynamic quantization. All other layers stay FP32.

    Dynamic quantization is used here (rather than static PTQ) because:
      - It can be applied to any submodule in isolation without calibration data
      - The goal of H2 is relative sensitivity ranking, not absolute accuracy —
        the method is identical across all layers so comparisons are valid
    """
    m = copy.deepcopy(base_model).cpu()
    submodule = getattr(m, attr_name)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress deprecation noise
        quantized = torch.ao.quantization.quantize_dynamic(
            submodule, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
        )

    setattr(m, attr_name, quantized)
    return m


# ─── Monotonicity Check ───────────────────────────────────────────────────────

def check_monotonicity(drops):
    """
    Tests whether the accuracy drops increase monotonically with layer depth.
    Returns (is_monotonic, num_violations, violation_details).
    Violations in ResNet are expected due to skip connections redistributing
    gradient flow — worth discussing if they appear.
    """
    violations = []
    for i in range(1, len(drops)):
        if drops[i] < drops[i - 1]:
            violations.append((i - 1, i, drops[i - 1], drops[i]))
    return len(violations) == 0, len(violations), violations


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_h2():
    print("\n" + "=" * 66)
    print("  NeuroQuant — Experiment H2: Layer-wise Sensitivity Analysis")
    print("=" * 66)

    loader, base_model = build_loader_and_model()
    results = []

    print(f"\n  [FP32] Measuring baseline ({DEVICE})...")
    acc_fp32 = evaluate_top1(copy.deepcopy(base_model), loader, DEVICE, "FP32 ")
    print(f"\n  FP32 baseline : {acc_fp32}%  (published: {PUBLISHED_FP32}%)\n")

    results.append({
        "Layer Group":     "FP32 baseline",
        "Cortex Analog":   "—",
        "Top-1 Acc (%)": acc_fp32,
        "Acc Drop (pp)": 0.00,
        "Energy Saved (%)": 0.00,
    })

    layer_drops = []

    for layer_name, attr_name, cortex in LAYERS:
        print(f"  [{layer_name.upper()}] Quantizing only {layer_name} → INT8  "
              f"(analogue: {cortex})")

        q_model = quantize_single_layer(base_model, layer_name, attr_name)
        acc     = evaluate_top1(q_model, loader, torch.device("cpu"), f"{layer_name} ")
        drop    = round(acc_fp32 - acc, 2)
        saved   = energy_proxy_single(layer_name)

        layer_drops.append(drop)
        results.append({
            "Layer Group":     layer_name,
            "Cortex Analog":   cortex,
            "Top-1 Acc (%)": acc,
            "Acc Drop (pp)": drop,
            "Energy Saved (%)": saved,
        })
        print(f"    Drop: {drop:+.2f} pp | Energy saved (this layer only): {saved}%\n")

    is_mono, n_violations, violation_details = check_monotonicity(layer_drops)
    layer_names = [row[0] for row in LAYERS]

    df = pd.DataFrame(results)
    print("=" * 66)
    print("  H2 RESULTS — Layer-wise Sensitivity Profile")
    print("=" * 66)
    print(df.to_string(index=False))

    print("\n  Sensitivity gradient (accuracy drop by layer depth):")
    for name, drop in zip(layer_names, layer_drops):
        bar = "█" * max(0, int(abs(drop) * 10))
        direction = "↑ loss" if drop > 0 else "↓ gain"
        print(f"    {name:<12} {drop:+.2f} pp  {bar}  {direction}")

    print(f"\n  Monotonic increase with depth: {'YES' if is_mono else 'NO'}")
    if not is_mono:
        print(f"  Violations ({n_violations}):")
        for i, j, d_i, d_j in violation_details:
            print(f"    {layer_names[i]} ({d_i:+.2f}) → {layer_names[j]} ({d_j:+.2f})")
        print("  Note: violations are expected in ResNet due to skip connections.")

    early_drop = sum(layer_drops[:2]) / 2
    late_drop  = sum(layer_drops[-2:]) / 2
    print(f"\n  Mean drop — early layers (conv1, layer1) : {early_drop:+.2f} pp")
    print(f"  Mean drop — late layers  (layer4, fc)    : {late_drop:+.2f} pp")
    confirmed = late_drop > early_drop
    print(f"\n  H2 Confirmed (late > early sensitivity): {'YES' if confirmed else 'NO'}")

    df.to_csv("h2_results.csv", index=False)
    with open("h2_summary.txt", "w") as f:
        f.write("NeuroQuant — H2 Results\n" + "=" * 66 + "\n\n")
        f.write(f"Dataset        : {DATASET}\n")
        f.write(f"Model          : ResNet-50 (He et al., 2016)\n")
        f.write(f"Eval samples   : {EVAL_SAMPLES}\n")
        f.write(f"FP32 baseline  : {acc_fp32}%\n\n")
        f.write("Layer Sensitivity Profile:\n")
        for name, drop in zip(layer_names, layer_drops):
            f.write(f"  {name:<12} drop: {drop:+.2f} pp\n")
        f.write(f"\nMonotonic increase: {'YES' if is_mono else f'NO ({n_violations} violation(s))'}\n")
        f.write(f"Mean early drop : {early_drop:+.2f} pp\n")
        f.write(f"Mean late drop  : {late_drop:+.2f} pp\n")
        f.write(f"H2 confirmed    : {confirmed}\n")

    print("\n  Saved: h2_results.csv | h2_summary.txt")
    print("=" * 66)
    return df


if __name__ == "__main__":
    run_h2()
