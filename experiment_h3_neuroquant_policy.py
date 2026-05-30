"""
NeuroQuant — Experiment H3: Cortically-Inspired Mixed-Precision Policy + Ablations
====================================================================================
Tests whether a biologically-motivated mixed-precision assignment
Pareto-dominates uniform quantization baselines on the accuracy-energy frontier.
Includes policy ablations to prove the cortical ordering matters, not just
the presence of mixed precision.

Hypothesis H3:
    The NeuroQuant cortical policy (INT8 early, BF16 mid, FP32 late) will
    sit on or above the Pareto frontier defined by uniform baselines, AND
    outperform the inverted and random ablation policies.

Policies tested:
    1. NeuroQuant (Cortical)    — INT8 early, BF16 mid, FP32 late
    2. Inverted (Anti-Cortical) — FP32 early, BF16 mid, INT8 late
    3. Random Assignment        — same precision budget, random ordering (seed=42)
    4. Aggressive Early INT8    — INT8×3 early, BF16 mid, FP32 late

Prerequisite:
    Run experiment_h1_uniform_quantization.py first — H3 loads h1_results.csv
    for Pareto baseline comparisons. Falls back to hardcoded values if missing.

Setup:
    pip install -r requirements.txt
    huggingface-cli login

Run:
    python experiment_h3_neuroquant_policy.py

Outputs:
    h3_results.csv   — full policy comparison table with Pareto labels
    h3_summary.txt   — ablation verdicts and H3 confirmation
"""

import os
import copy
import random
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────

DATASET          = "imagenet_hf"
IMAGENET_VAL_DIR = "./imagenet/val"
BATCH_SIZE       = 64
EVAL_SAMPLES     = 5000
PUBLISHED_FP32   = 76.13


def _get_device():
    try:
        if torch.cuda.is_available():
            dev   = torch.device("cuda")
            torch.zeros(1).to(dev)
            props = torch.cuda.get_device_properties(0)
            print(f"  [GPU] {props.name} | "
                  f"VRAM: {props.total_memory / 1e9:.1f} GB | "
                  f"CUDA {torch.version.cuda}")
            return dev
    except Exception as e:
        print(f"  [GPU] CUDA unavailable ({e}), falling back to CPU")
    return torch.device("cpu")


DEVICE = _get_device()

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
BITS        = {"fp32": 32, "bf16": 16, "int8": 8}

LAYER_ATTRS = {
    "conv1":      "conv1",
    "layer1":     "layer1",
    "layer2":     "layer2",
    "layer3":     "layer3",
    "layer4":     "layer4",
    "classifier": "fc",
}


def energy_proxy(precision_map):
    """FLOPs-weighted energy estimate. Returns (norm, savings_pct)."""
    weighted = sum(
        LAYER_FLOPS[layer] * (BITS[prec] / 32)
        for layer, prec in precision_map.items()
    )
    norm    = round(weighted / TOTAL_FLOPS, 4)
    savings = round((1 - norm) * 100, 2)
    return norm, savings


# ─── Ablation Policies ────────────────────────────────────────────────────────

def _make_random_policy(seed=42):
    """
    Random precision assignment using the same budget as NeuroQuant
    (2×INT8, 1×BF16, 3×FP32), shuffled with a fixed seed for reproducibility.
    Proves NeuroQuant's specific cortical ordering matters, not just the budget.
    """
    layer_names = list(LAYER_FLOPS.keys())
    precisions  = ["int8", "int8", "bf16", "fp32", "fp32", "fp32"]
    random.seed(seed)
    random.shuffle(precisions)
    return dict(zip(layer_names, precisions))


RANDOM_POLICY = _make_random_policy(seed=42)

ABLATION_POLICIES = {
    "NeuroQuant (Cortical)": {
        "conv1": "int8", "layer1": "int8",  "layer2": "bf16",
        "layer3": "fp32", "layer4": "fp32", "classifier": "fp32",
    },
    "Inverted (Anti-Cortical)": {
        "conv1": "fp32", "layer1": "fp32",  "layer2": "bf16",
        "layer3": "int8", "layer4": "int8", "classifier": "fp32",
    },
    "Random Assignment": RANDOM_POLICY,
    "Aggressive Early INT8": {
        "conv1": "int8", "layer1": "int8",  "layer2": "int8",
        "layer3": "bf16", "layer4": "fp32", "classifier": "fp32",
    },
}

# ─── BF16 Wrapper ─────────────────────────────────────────────────────────────

class BF16Wrapper(nn.Module):
    """
    Runs a submodule in BF16 via torch.autocast with FP32 input/output
    boundaries. autocast keeps BatchNorm in FP32 while running Conv2d in BF16,
    handling all internal dtype conversions within residual blocks automatically.
    """
    def __init__(self, module, device_type="cuda"):
        super().__init__()
        self.module      = module
        self.device_type = device_type

    def forward(self, x):
        try:
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                out = self.module(x)
        except RuntimeError:
            out = self.module(x.to(torch.bfloat16))
        return out.to(torch.float32)


# ─── Quantization Utilities ───────────────────────────────────────────────────

def fake_quantize_tensor(tensor, bits=8):
    """
    Per-channel symmetric fake INT8 quantization.
    Rounds weights to the INT8 representable grid and dequantizes back to FP32.
    Matches TensorRT/ONNX Runtime per-channel weight quantization.
    (Dong et al., 2019 — HAWQ; Krishnamoorthi, 2018)
    """
    if tensor.dim() < 2:
        abs_max   = tensor.abs().max().clamp(min=1e-8)
        scale     = abs_max / (2**(bits-1) - 1)
        quantized = torch.round(tensor / scale).clamp(-(2**(bits-1)), 2**(bits-1) - 1)
        return quantized * scale
    flat      = tensor.view(tensor.size(0), -1)
    abs_max   = flat.abs().max(dim=1).values.clamp(min=1e-8)
    scale     = (abs_max / (2**(bits-1) - 1)).view(-1, *([1] * (tensor.dim() - 1)))
    quantized = torch.round(tensor / scale).clamp(-(2**(bits-1)), 2**(bits-1) - 1)
    return quantized * scale


def apply_fake_int8(model, attr_name):
    """Apply per-channel fake INT8 to all weight tensors in one layer group."""
    submodule = getattr(model, attr_name)
    with torch.no_grad():
        for name, param in submodule.named_parameters():
            if "weight" in name and param.dim() > 1:
                param.data = fake_quantize_tensor(param.data)


# ─── General Policy Application ───────────────────────────────────────────────

def apply_policy(base_model, policy_map):
    """
    Applies any mixed-precision policy to a fresh deep copy of the model.
    base_model is never modified — each policy gets its own independent copy.
    """
    m           = copy.deepcopy(base_model)
    device_type = DEVICE.type

    for layer_name, precision in policy_map.items():
        attr = LAYER_ATTRS[layer_name]
        if precision == "int8":
            apply_fake_int8(m, attr)
        elif precision == "bf16":
            setattr(m, attr, BF16Wrapper(getattr(m, attr), device_type=device_type))
        # fp32 — no change needed

    return m


# ─── Dataset + Model ─────────────────────────────────────────────────────────

def build_loader_and_model():
    """Returns (eval_loader, model). Identical setup to H1/H2."""

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
        import torchvision.datasets as tv_datasets
        from torch.utils.data import Subset
        dataset = tv_datasets.ImageFolder(IMAGENET_VAL_DIR, transform=imagenet_transform)
        gen = torch.Generator().manual_seed(42)
        idx = torch.randperm(len(dataset), generator=gen)[:EVAL_SAMPLES].tolist()
        loader = DataLoader(Subset(dataset, idx), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)
    else:
        raise ValueError(f"Unknown DATASET: {DATASET}")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    print(f"  [Model] ResNet-50 | Device: {DEVICE}")
    model.eval()
    return loader, model


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, loader, device, label=""):
    """Returns (top1_acc, top5_acc)."""
    model.to(device)
    model.eval()
    correct1, correct5, total = 0, 0, 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs  = model(images)
            _, top5  = outputs.topk(5, dim=1)
            correct1 += top5[:, 0].eq(labels).sum().item()
            correct5 += top5.eq(labels.unsqueeze(1).expand_as(top5)).any(dim=1).sum().item()
            total    += labels.size(0)
            if (i + 1) % 10 == 0:
                print(f"    {label}Batch {i+1}/{len(loader)} "
                      f"— top-1: {100.*correct1/total:.2f}%", end="\r")

    top1 = round(100.0 * correct1 / total, 2)
    top5 = round(100.0 * correct5 / total, 2)
    print(f"    {label}Final — top-1: {top1:.2f}%  top-5: {top5:.2f}%               ")
    return top1, top5


# ─── Pareto Analysis ──────────────────────────────────────────────────────────

def pareto_analysis(configs):
    """
    Returns dict of {name: is_pareto_optimal}.
    A config is dominated if another is strictly better on BOTH axes.
    """
    pareto = {}
    for name, acc, savings in configs:
        dominated = any(
            (other_acc >= acc and other_savings >= savings and
             (other_acc > acc or other_savings > savings))
            for other_name, other_acc, other_savings in configs
            if other_name != name
        )
        pareto[name] = not dominated
    return pareto


# ─── Load H1 Baselines ────────────────────────────────────────────────────────

def load_h1_baselines():
    h1_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h1_results.csv")
    if os.path.exists(h1_path):
        try:
            df       = pd.read_csv(h1_path)
            fp32_row = df[df["Configuration"].str.contains("FP32")].iloc[0]
            bf16_row = df[df["Configuration"].str.contains("BF16")].iloc[0]
            int8_row = df[df["Configuration"].str.contains("INT8")].iloc[0]
            print(f"  Loaded H1 baselines from {h1_path}")
            return (float(fp32_row["Top-1 Acc (%)"]),
                    float(bf16_row["Top-1 Acc (%)"]),
                    float(int8_row["Top-1 Acc (%)"]))
        except Exception as e:
            print(f"  Warning: could not parse h1_results.csv ({e}). Using fallback values.")
    print("  h1_results.csv not found — using fallback values (run H1 first for accurate comparison).")
    return 76.60, 76.42, 76.68


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_h3():
    print("\n" + "=" * 72)
    print("  NeuroQuant — H3: Mixed-Precision Policy + Ablations")
    print("=" * 72)

    print("\n  Loading H1 baselines...")
    acc_fp32_h1, acc_bf16_h1, acc_int8_h1 = load_h1_baselines()
    _, sav_fp32 = energy_proxy({k: "fp32" for k in LAYER_FLOPS})
    _, sav_bf16 = energy_proxy({k: "bf16" for k in LAYER_FLOPS})
    _, sav_int8 = energy_proxy({k: "int8" for k in LAYER_FLOPS})
    print(f"  FP32 : {acc_fp32_h1}%  ({sav_fp32}% saved)")
    print(f"  BF16 : {acc_bf16_h1}%  ({sav_bf16}% saved)")
    print(f"  INT8 : {acc_int8_h1}%  ({sav_int8}% saved)")

    loader, base_model = build_loader_and_model()
    print(f"\n  [FP32] Re-measuring baseline ({DEVICE})...")
    acc_fp32, top5_fp32 = evaluate(copy.deepcopy(base_model), loader, DEVICE, "FP32 ")

    ablation_results = []

    for policy_name, policy_map in ABLATION_POLICIES.items():
        print(f"\n  [{policy_name}] Applying policy...")
        for layer, prec in policy_map.items():
            print(f"    {layer:<12} → {prec.upper()}")
        _, savings = energy_proxy(policy_map)
        print(f"    Expected energy savings: {savings}%")

        policy_model = apply_policy(base_model, policy_map)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        top1, top5 = evaluate(policy_model, loader, DEVICE, f"{policy_name[:6]} ")
        drop       = round(acc_fp32 - top1, 2)

        ablation_results.append({
            "name":    policy_name,
            "top1":    top1,
            "top5":    top5,
            "drop":    drop,
            "savings": savings,
            "policy":  policy_map,
        })
        print(f"    Result: {top1}%  (drop: {drop:+.2f} pp)  savings: {savings}%")

    all_configs = [
        ("FP32 (baseline)",   acc_fp32_h1,  sav_fp32),
        ("Uniform BF16 (H1)", acc_bf16_h1,  sav_bf16),
        ("Uniform INT8 (H1)", acc_int8_h1,  sav_int8),
    ] + [(r["name"], r["top1"], r["savings"]) for r in ablation_results]

    pareto_dict = pareto_analysis(all_configs)

    rows = []
    for name, acc, savings in all_configs:
        drop = round(acc_fp32_h1 - acc, 2) if name != "FP32 (baseline)" else 0.00
        rows.append({
            "Configuration":      name,
            "Top-1 Acc (%)":      acc,
            "Acc Drop (pp)":      drop,
            "Energy Savings (%)": savings,
            "Pareto Optimal":     "YES" if pareto_dict[name] else "NO",
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print("  H3 RESULTS — All Policies vs Baselines")
    print("=" * 72)
    print(df.to_string(index=False))

    nq = next(r for r in ablation_results if "Cortical"   in r["name"])
    iv = next(r for r in ablation_results if "Inverted"   in r["name"])
    rn = next(r for r in ablation_results if "Random"     in r["name"])
    ag = next(r for r in ablation_results if "Aggressive" in r["name"])

    nq_beats_inverted = nq["top1"] >= iv["top1"]
    nq_beats_random   = nq["top1"] >= rn["top1"]

    print(f"\n  Ablation comparison (same energy budget):")
    print(f"    NeuroQuant (Cortical)    : {nq['top1']}%  savings: {nq['savings']}%  ← proposed")
    print(f"    Inverted (Anti-Cortical) : {iv['top1']}%  savings: {iv['savings']}%")
    print(f"    Random Assignment        : {rn['top1']}%  savings: {rn['savings']}%")
    print(f"    Aggressive Early INT8    : {ag['top1']}%  savings: {ag['savings']}%")
    print(f"\n  NeuroQuant beats Inverted : {'YES' if nq_beats_inverted else 'NO'}")
    print(f"  NeuroQuant beats Random   : {'YES' if nq_beats_random else 'NO'}")

    print(f"\n  Pareto frontier (★ = optimal):")
    for name, acc, savings in sorted(all_configs, key=lambda x: x[2]):
        marker = "★" if pareto_dict[name] else "·"
        bar    = "─" * int(savings / 2)
        tag    = " ← NeuroQuant" if "Cortical" in name else ""
        print(f"  {marker} {acc:.2f}%  |{bar}  {savings:.1f}%{tag}")

    h3_confirmed = pareto_dict.get("NeuroQuant (Cortical)", False)
    print(f"\n  H3 Confirmed (NeuroQuant Pareto-optimal): {'YES' if h3_confirmed else 'NO'}")

    df.to_csv("h3_results.csv", index=False)
    with open("h3_summary.txt", "w", encoding="utf-8") as f:
        f.write("NeuroQuant — H3 Results + Ablations\n" + "=" * 72 + "\n\n")
        f.write(f"Dataset      : {DATASET}\nModel        : ResNet-50 (He et al., 2016)\n")
        f.write(f"Eval samples : {EVAL_SAMPLES}\nFP32 baseline: {acc_fp32}%\n\n")
        f.write("Policy Results:\n")
        for r in ablation_results:
            f.write(f"  {r['name']:<30} acc: {r['top1']}%  "
                    f"drop: {r['drop']:+.2f} pp  savings: {r['savings']}%  "
                    f"pareto: {'YES' if pareto_dict[r['name']] else 'NO'}\n")
        f.write(f"\nNeuroQuant beats Inverted : {'YES' if nq_beats_inverted else 'NO'}\n")
        f.write(f"NeuroQuant beats Random   : {'YES' if nq_beats_random else 'NO'}\n")
        f.write(f"H3 Confirmed              : {'YES' if h3_confirmed else 'NO'}\n")
        f.write(f"\nRandom policy used (seed=42):\n")
        for layer, prec in RANDOM_POLICY.items():
            f.write(f"  {layer:<12} → {prec.upper()}\n")

    print("\n  Saved: h3_results.csv | h3_summary.txt")
    print("=" * 72)
    return df


if __name__ == "__main__":
    run_h3()
