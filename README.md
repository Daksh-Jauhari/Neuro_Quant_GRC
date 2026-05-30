# NeuroQuant-GRC

**Biologically-inspired mixed-precision neural network quantization guided by the visual cortex hierarchy.**

NeuroQuant tests whether assigning numerical precision to ResNet-50 layers according to the V1→V2→V4→IT cortical gradient — lower precision early, higher precision late — can Pareto-dominate uniform quantization baselines on the accuracy–energy frontier.

---

## Hypothesis Overview

| ID | Hypothesis | Result |
|----|-----------|--------|
| H1 | Uniform quantization (BF16 / INT8) preserves top-1 accuracy within 1 pp while saving ≥ 35% energy | ✅ Confirmed |
| H2 | Quantization sensitivity increases monotonically with layer depth, mirroring the cortical hierarchy | Run to verify |
| H3 | The cortical mixed-precision policy Pareto-dominates uniform baselines and all ablation policies | Run to verify |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Authenticate with HuggingFace (one-time)

```bash
pip install huggingface_hub
huggingface-cli login
```

> Accept dataset terms at: https://huggingface.co/datasets/ILSVRC/imagenet-1k

### 3. Run experiments in order

```bash
# H1 — Uniform quantization baseline (run this first)
python experiment_h1_uniform_quantization.py

# H2 — Layer-wise sensitivity analysis
python experiment_h2_layer_sensitivity.py

# H3 — Cortical mixed-precision policy + ablations (reads h1_results.csv)
python experiment_h3_neuroquant_policy.py
```

> **Run H1 before H3.** H3 automatically loads `h1_results.csv` to use as uniform baselines for Pareto comparison.

---

## Results

### H1 — Uniform Quantization

| Configuration | Top-1 Acc (%) | Acc Drop (pp) | Energy Savings (%) | H1 Confirmed |
|---|:---:|:---:|:---:|:---:|
| FP32 baseline (measured) | 76.60 | 0.00 | 0.0% | — |
| Uniform BF16 | 76.42 | +0.18 | 50.0% | YES |
| Uniform Static INT8 (PTQ) | 76.68 | −0.08 | 75.0% | YES |

Full data: [`results/h1_results.csv`](results/h1_results.csv)

### H2 — Layer Sensitivity

Run `experiment_h2_layer_sensitivity.py` to populate [`results/h2_results.csv`](results/h2_results.csv).

Expected schema:

| Layer Group | Cortex Analog | Top-1 Acc (%) | Acc Drop (pp) | Energy Saved (%) |
|---|---|:---:|:---:|:---:|
| conv1 | V1 — edges & orientations | — | — | — |
| layer1 | V1/V2 — local features | — | — | — |
| layer2 | V2/V4 — shapes & textures | — | — | — |
| layer3 | V4 — complex patterns | — | — | — |
| layer4 | IT cortex — object parts | — | — | — |
| classifier | Decision — object identity | — | — | — |

### H3 — NeuroQuant Policy vs Ablations

Run `experiment_h3_neuroquant_policy.py` to populate [`results/h3_results.csv`](results/h3_results.csv).

Policies tested:

| Policy | Description |
|--------|-------------|
| NeuroQuant (Cortical) | INT8 early, BF16 mid, FP32 late — the proposed policy |
| Inverted (Anti-Cortical) | FP32 early, BF16 mid, INT8 late — reversal ablation |
| Random Assignment | Same precision budget, random layer order |
| Aggressive Early INT8 | INT8 × 3 early, BF16 mid, FP32 late |

---

## Repository Structure

```
Neuro_Quant_GRC/
├── experiment_h1_uniform_quantization.py   # H1: uniform BF16 + static INT8 PTQ
├── experiment_h2_layer_sensitivity.py      # H2: layer-by-layer INT8 sensitivity
├── experiment_h3_neuroquant_policy.py      # H3: cortical policy + ablations
├── METHODS.md                              # Full experimental methodology
├── requirements.txt                        # Python dependencies
├── .gitignore
└── results/
    ├── h1_results.csv                      # H1 output (pre-run, committed)
    ├── h2_results.csv                      # H2 output schema (run to populate)
    └── h3_results.csv                      # H3 output schema (run to populate)
```

---

## The NeuroQuant Policy (H3)

Inspired by the precision gradient in the primate visual cortex:

| Layer | Precision | Cortex Analogue |
|-------|-----------|----------------|
| conv1 | INT8 | V1 — edges & orientations |
| layer1 | INT8 | V1/V2 — local features |
| layer2 | BF16 | V2/V4 — shapes & textures |
| layer3 | FP32 | V4 — complex patterns |
| layer4 | FP32 | IT cortex — object parts |
| classifier | FP32 | Decision — object identity |

Early layers encode densely redundant, compression-tolerant features. Later layers encode sparse, task-specific representations where precision loss causes disproportionate accuracy degradation.

---

## Dataset Modes

Each script supports three dataset modes — set `DATASET` at the top of each file:

| Mode | Description |
|------|-------------|
| `imagenet_hf` | ImageNet-1K via HuggingFace streaming — **default**, no local download |
| `imagenet_local` | Local ImageNet val directory (set `IMAGENET_VAL_DIR`) |
| `tiny_imagenet` | Tiny-ImageNet-200 fallback — H1 only, ~25% baseline expected |

---

## Methods

See [`METHODS.md`](METHODS.md) for the full experimental protocol, energy proxy definition, quantization procedures, and references.

---

## References

1. He et al. (2016) — Deep Residual Learning for Image Recognition
2. Deng et al. (2009) — ImageNet: A Large-Scale Hierarchical Image Database
3. Yamins & DiCarlo (2016) — Using Goal-Driven Deep Learning Models to Understand Sensory Cortex
4. Micikevicius et al. (2018) — Mixed Precision Training
5. Jacob et al. (2018) — Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
6. Barlow (1961) — Possible Principles Underlying the Transformation of Sensory Messages
