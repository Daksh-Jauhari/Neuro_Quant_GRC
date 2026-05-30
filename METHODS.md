# III. Methods

## A. Model and Dataset

All experiments were conducted using ResNet-50, a 50-layer deep residual network pretrained on ImageNet-1K under standard supervised training. ResNet-50 was selected as the experimental backbone for three reasons: its layer-wise architecture maps cleanly onto the cortical hierarchy described in the neuroscience literature, its published top-1 accuracy of 76.13% on ImageNet-1K provides a well-established baseline against which quantization effects can be measured, and its widespread adoption in the computer vision community ensures that results are reproducible and comparable across studies.

The evaluation dataset was ImageNet-1K, accessed via the ILSVRC/imagenet-1k repository on HuggingFace. For each experiment, 5,000 randomly sampled validation images were used for accuracy measurement and a non-overlapping set of 512 validation images was reserved exclusively for static PTQ calibration. Standard ImageNet preprocessing was applied: images were resized to 256×256, center-cropped to 224×224, converted to tensors, and normalized using the dataset's published channel-wise mean ([0.485, 0.456, 0.406]) and standard deviation ([0.229, 0.224, 0.225]).

## B. Energy Proxy Metric

Direct energy measurement at the hardware level requires specialized profiling equipment and is highly environment-dependent. Following the approach of energy-aware neural network design literature, this study uses a FLOPs-weighted normalized energy proxy defined as:

> E = Σᵢ (FLOPs_i × bits_i / 32) / Σᵢ FLOPs_i

where FLOPs_i is the number of multiply-add operations in layer group i and bits_i is the numerical precision of that group (32 for FP32, 16 for BF16, 8 for INT8). The denominator normalizes against the full FP32 baseline, yielding a value in [0, 1] where 1.0 represents full FP32 energy consumption and energy savings are expressed as (1 − E) × 100%. Layer-wise FLOPs for ResNet-50 were taken from He et al. (2016) and confirmed via torchinfo profiling: conv1 (235 MFLOPs), layer1 (575 MFLOPs), layer2 (1,125 MFLOPs), layer3 (2,255 MFLOPs), layer4 (1,130 MFLOPs), and the fully connected classifier (4 MFLOPs), totaling 5,324 MFLOPs.

## C. Quantization Methods

**BF16 (Brain Float 16).** Bfloat16 preserves the 8-bit exponent of FP32 — maintaining the same dynamic range — while reducing the mantissa from 23 to 7 bits. All convolutional and linear layers were converted to BF16. Batch normalization layers were kept in FP32, as their running statistics degrade significantly at reduced mantissa precision, consistent with findings by Micikevicius et al.

**Static Post-Training Quantization (PTQ).** Static PTQ converts both weights and activations to INT8 ahead of inference, avoiding the per-batch overhead of dynamic quantization and making it the standard method for CNN deployment in the research literature. The procedure follows Jacob et al. (2018): Conv+BN+ReLU layer sequences were first fused into single quantizable operations using torchvision's built-in fuse_model() routine; MinMax observers were then inserted at every activation point using torch.ao.quantization.prepare(); the 512-image calibration set was passed through the model to collect activation range statistics; and finally torch.ao.quantization.convert() replaced all floating-point operations with INT8 kernels. The oneDNN backend was used for quantized kernel execution.

## D. Experimental Design

Three experiments test three falsifiable hypotheses derived from the visual cortex hierarchy:

**Experiment H1 — Uniform Quantization.** To establish whether precision reduction alone is sufficient to achieve energy savings without accuracy loss, all six layer groups of ResNet-50 were quantized simultaneously to a single precision (BF16 or static INT8). The hypothesis is confirmed if the accuracy drop remains within 1 percentage point of the FP32 baseline while energy savings exceed 35%.

**Experiment H2 — Layer-wise Sensitivity Analysis.** To test whether quantization sensitivity increases monotonically with layer depth — mirroring the V1→V2→V4→IT cortex gradient in which later cortical stages encode increasingly abstract and irreplaceable representations — each layer group was quantized to INT8 individually while all other groups remained at FP32. The resulting accuracy drops were ordered by layer depth to assess monotonicity.

**Experiment H3 — NeuroQuant Mixed-Precision Policy.** Drawing directly from the cortical precision gradient, a biologically-motivated mixed-precision assignment was constructed: early layers (conv1, layer1) were assigned INT8, mid-level layers (layer2) were assigned BF16, and deeper layers (layer3, layer4, classifier) were retained at FP32. This policy reflects the principle that early sensory representations are densely redundant and compression-tolerant, while later representations are sparse, task-specific, and precision-sensitive. The NeuroQuant policy was compared against uniform BF16 and uniform INT8 on the accuracy–energy Pareto frontier to assess whether cortically-derived precision assignment outperforms uniform baselines.

All experiments were run under identical conditions: the same pretrained ResNet-50 weights, the same 5,000-image evaluation subset, and the same random seed (42) for any stochastic sampling. Top-1 accuracy and accuracy drop (in percentage points relative to the FP32 baseline) were recorded for each configuration alongside the FLOPs-weighted energy proxy.

---
*References:*
*[1] He et al. (2016) — ResNet*
*[2] Deng et al. (2009) — ImageNet*
*[3] Yamins & DiCarlo (2016) — Visual cortex hierarchy*
*[4] Micikevicius et al. (2018) — Mixed precision training*
*[5] Jacob et al. (2018) — Quantization and training of neural networks*
*[6] Barlow (1961) — Neural coding efficiency*
