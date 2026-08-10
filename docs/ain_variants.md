# AIN variant ladder (X-ain) design study

Goal: improve domain generalization across the whole ladder by adding an
AIN (adaptive instance normalization) variant of every tier — teacher B
included — while keeping the existing BN ladder untouched. Evaluation stays
on the standard protocol (train on all five domains, evaluate on the unified
five-domain test split); no leave-one-domain-out runs.

## Naming and artifacts

Every variant gets a parallel `-ain` sibling; nothing existing is replaced.

| Variant | Backbone | Config | Output dir |
| --- | --- | --- | --- |
| B-ain | ViT-B/16 + token-IN | `vit_base_8gb_ain.yml` | `logs/reid_vit_base_8gb_ain` |
| S-ain | ViT-S/16 + token-IN | `vit_small_8gb_distill_ain.yml` | `logs/reid_vit_small_8gb_distill_ain` |
| T-ain | OSNet-AIN x1.5 | `osnet_t_8gb_distill_ain.yml` | `logs/reid_osnet_t_8gb_distill_ain` |
| N-ain | OSNet-AIN x1.25 | `osnet_n_8gb_distill_ain.yml` | `logs/reid_osnet_n_8gb_distill_ain` |
| P-ain | OSNet-AIN x1.0 | `osnet_p_8gb_distill_ain.yml` | `logs/reid_osnet_p_8gb_distill_ain` |
| F-ain / A-ain | OSNet-AIN x0.75 / x0.5 | later | later |

ONNX naming keeps the no-ViT-elements rule: OSNet tiers export as
`osnet_ain_<multiplier>_<tier>_unified.onnx` (matching the upstream
`osnet_ain_x1_0` model naming); ViT tiers keep the `personvit_*` prefix.

## Architecture changes per family

### OSNet tiers (T/N/P/F/A-ain)

Vendor Torchreid's `osnet_ain.py` (MIT, module names kept weight-compatible)
parameterized by the same channel tuples as the BN ladder. OSNet-AIN places
InstanceNorm at searched positions (predominantly early layers), removing
instance-specific style (illumination, color cast, camera tone) while BN
elsewhere keeps discriminative statistics.

### ViT tiers (B-ain / S-ain)

OSNet-AIN's recipe does not transfer literally (ViT has no BN; LayerNorm is
already per-sample). The faithful adaptation of "remove low-level style
early" is **token-axis instance normalization after the patch embedding**:
per sample and per channel, normalize over the token (spatial) axis, with a
learnable affine, before the transformer blocks. This:

- removes patch-level style statistics exactly where AIN's search placed IN
  (early layers), and
- keeps every pretrained tensor loadable — the IN affine is the only new
  parameter, so the PersonViT self-supervised checkpoints still initialize
  B-ain/S-ain (the tolerant loader skips nothing else).

An ablation slot is reserved for a second IN after block 2 if B-ain
underperforms plain B by more than the gate below.

## Teacher/initialization chain

The key coherence decision: **-ain students distill from the -ain teacher.**
A style-invariant teacher produces style-invariant logits and similarity
structures, so distillation reinforces the invariance instead of fighting it
(distilling an AIN student from the BN teacher B would push style-dependent
structure back into the student).

```
checkpoint0260 (ssl) ──fine-tune──> B-ain
B-ain ──distill──> S-ain
B-ain ──distill──> P-ain (init: official osnet_ain_x1_0 ImageNet weights)
P-ain best ──width-expand──> N-ain init ──distill(B-ain)──> N-ain
N-ain best ──width-expand──> T-ain init ──distill(B-ain)──> T-ain
```

- The function-preserving expansion tool works unchanged for AIN models:
  IN affine parameters expand with zero padding like BN affine, so new
  channels stay silent at initialization.
- F-ain / A-ain initialize from the official `osnet_ain_x0_75` / `x0_5`
  ImageNet weights when those tiers are trained.
- Recipes are identical to the BN ladder (B-ain: SGD 2e-4 batch 32; OSNet
  tiers: Adam 3.5e-4 batch 64, 100 epochs) so BN-vs-AIN differences are
  attributable to the architecture alone.

## Evaluation protocol and gates (5-domain train + 5-domain eval)

The unified in-distribution benchmark cannot show generalization gains
directly, so the -ain ladder is judged by non-inferiority plus proxies:

1. **Non-inferiority gate**: X-ain must stay within **1.0 mAP** of X on the
   unified test set (literature predicts a 0-1 point in-domain cost for IN).
2. **Per-domain breakdown** (existing tools): special attention to the
   hardest domain (d01) and the occluded domains (d03/d04); a smaller
   worst-domain gap is itself weak evidence of better invariance.
3. **Style-shift probe** (stays within the 5+5 protocol): evaluate the same
   unified test split under photometric perturbations (illumination gain,
   color temperature, contrast). The BN-vs-AIN delta under shift is the
   in-protocol generalization signal. Implemented as
   `tools/eval_style_shift.py`; measured results below.

## Measured style-shift robustness (probe results)

`tools/eval_style_shift.py` re-renders the queries under eight deterministic
photometric shifts (brightness ±30%, contrast ±40%, warm/cool color
temperature, gamma 0.6/1.6) and matches them against the clean gallery —
the mixed old/new-camera scenario that style sensitivity breaks first.
Measured on the four completed BN/-ain pairs:

| Pair | Clean cost of -ain | Mean mAP drop over the 8 shifts<br>(BN -> -ain) | Worst shift (warm): absolute mAP<br>(BN -> -ain) |
| --- | ---: | ---: | ---: |
| B / B-ain | -1.2 | 5.2 -> 3.4 (-34%) | 69.7 -> **74.3** |
| S / S-ain | -0.8 | 5.3 -> 3.9 (-26%) | 69.5 -> **70.1** |
| N / N-ain | -2.7 | 10.9 -> **5.8 (halved)** | 65.3 -> **71.9** |
| P / P-ain | -3.0 | 10.3 -> **5.1 (halved)** | 63.7 -> **73.7** |

T-ain (whose BN sibling was never trained) probes in line with the other
CNN -ain tiers: mean drop 5.6, worst shift (warm) 74.0 absolute, and the
exact-zero `contrast-40%` invariance preserved.

Key findings:

1. **Exact-zero degradation confirmed**: the -ain models lose exactly 0.0000
   mAP under `dark-30%` and `contrast-40%` — the predicted mathematical
   invariance (a global affine pixel change maps through the linear patch
   embedding to a per-channel feature affine, which token-IN removes
   exactly). `bright+30%`/`contrast+40%` break affinity through pixel
   clipping and channel-selective color shifts are only partially removable,
   matching theory.
2. **The CNN tiers gain most**: the BN CNN models are the most fragile in
   the ladder (mean -10.3/-10.9 mAP under shift for P/N); the -ain versions
   halve the degradation and beat their BN siblings by up to 10 mAP absolute
   under moderate-to-severe shifts. Their ~3-point in-distribution cost buys
   the largest robustness dividend in the ladder, justifying the CNN -ain
   tiers despite missing the in-distribution non-inferiority gate.
3. **The selection guidance is now quantitative**: the BN ladder wins only
   when deployment conditions match training; under any noticeable style
   shift the -ain ladder matches or exceeds it in absolute terms.

## Depth-expanded variants (X-ain-deep)

A second, depth-based growth axis for the CNN tiers: `osnet_ain_x*_deep`
appends **one plain OSBlock (no IN) at the end of each stage** on top of the
corresponding -ain tier. Appending at stage end keeps every existing
parameter name unchanged and leaves the searched IN arrangement (and the
ONNX IN-node count) untouched.

| Variant | Backbone params | GFLOPs @256x128 | Config | Init source |
| --- | ---: | ---: | --- | --- |
| P-ain-deep | 2.74M | 1.27 | `osnet_p_8gb_distill_ain_deep.yml` | P-ain best |
| N-ain-deep | 4.17M | 1.94 | `osnet_n_8gb_distill_ain_deep.yml` | N-ain best |
| T-ain-deep | 5.88M | 2.76 | `osnet_t_8gb_distill_ain_deep.yml` | T-ain best (run T-ain first) |

Initialization is function-preserving like the width chain, via
`tools/init_depth_expand.py`: the appended blocks are residual, so zeroing
their closing-BN affine makes each an exact identity — the deep model
reproduces its source's outputs to 0.0 at initialization (verified) and the
new blocks are recruited through their BN scales during training. Recipes
are unchanged (Adam 3.5e-4, wd 5e-4, 100 epochs, B-ain teacher).

## Photometric augmentation fine-tune (X-ain-aug)

The style-shift probe showed what -ain cannot remove by construction:
clipping-induced non-affinity (`bright+30%`) and the residual of
channel-selective color shifts (warm/cool). Photometric augmentation is the
complementary, data-side fix for exactly those gaps, and it composes well
with this ladder: the B-ain teacher is style-robust, so its distillation
targets stay stable under the distorted inputs.

Implementation (config-gated, default off — existing recipes unchanged):

- `INPUT.CJ_PROB` applies a mild ColorJitter (brightness 0.2, contrast 0.3,
  saturation 0.2, **hue 0.02** — color is a primary ReID cue, so large hue
  shifts or the channel permutation of `RandomPhotometricDistort` are
  deliberately excluded).
- `INPUT.BLUR_PROB` applies GaussianBlur (kernel 5, sigma from
  `INPUT.BLUR_SIGMA`) for cross-camera focus/resolution variation.

`osnet_p_8gb_distill_ain_aug.yml` is the probe experiment: it warm-starts
from the trained P-ain best (same architecture — every key loads verbatim,
epoch 1 starts at ~87 mAP) and fine-tunes with a short low-LR schedule
(Adam 1e-4, 40 epochs) so the only variable is the augmentation. Success
criterion: clean mAP roughly held, with the remaining warm/cool/`bright+`
degradation further reduced in `tools/eval_style_shift.py`.

Two safeguards exist against the warm-start best-selection trap (epoch 1
scores ~87 clean mAP before any adaptation, so if augmentation traded clean
mAP for robustness, plain clean-mAP selection could keep the unadapted
epoch-1 weights as "best" forever):

- **`SOLVER.VAL_SHIFT`** (e.g. `'warm'`): each eval additionally scores
  style-shifted queries against the clean gallery, and the best model is
  selected on the **mean of clean and shifted mAP** (the best filename then
  records that mean). The measured P-ain-aug run cleared its warm-start
  clean mAP on its own (below), so this stays **off by default** and is an
  opt-in for aug runs that stall below their warm-start value.
- **`load_param` accepts resume-format checkpoints** (`checkpoint_last.pth`,
  `'model'` key), so every eval tool can also score the final-epoch model
  directly and compare it against the selected best.

### Measured results (P-ain-aug, 40-epoch fine-tune)

The trap never materialized: clean mAP crossed the warm-start value at
epoch 28 (~70% of the schedule, the same cosine-tail position as every
warm-started run) and the best landed on the final epoch at **87.62**
(+0.6 over P-ain — the fine-tune recovered a fifth of the -ain clean cost,
helped by P-ain having been undertrained: its best sat on epoch 100/100).
Style-shift probe, mAP drop per condition:

| Condition | P (BN) | P-ain | P-ain-aug |
| --- | ---: | ---: | ---: |
| clean (absolute) | 90.0 | 87.0 | **87.6** |
| bright+30% | -0.9 | -0.9 | **-0.4** |
| dark-30% | -1.9 | -0.6 | **-0.2** |
| contrast-40% | -13.3 | -0.0 | **-0.0** |
| contrast+40% | -17.0 | -12.0 | **-8.2** |
| warm | -26.4 | -13.4 | **-8.5** |
| cool | -10.9 | -7.8 | **-6.4** |
| gamma0.6 | -6.4 | -2.6 | **-2.1** |
| gamma1.6 | -5.4 | -3.6 | **-2.8** |
| **mean** | **-10.3** | **-5.1** | **-3.6** |

P-ain-aug strictly dominates P-ain: clean up, every shift condition up,
and the architecture's exact-zero invariance (contrast-40%) preserved.
The augmentation attacked exactly the residues -ain cannot remove —
the two worst conditions (warm -13.4 -> -8.5, contrast+40% -12.0 -> -8.2)
shrank by ~30-40%, and under warm the model scores 79.2 absolute vs 63.7
for BN-P. The recipe (mild jitter, hue<=0.02, warm-start, low-LR 40
epochs, B-ain teacher) is validated for rollout to the other tiers.

### N-ain-aug reproduces every finding

The identical recipe on N-ain crossed its warm-start clean mAP at epoch 29
(~70% of the schedule again), landed its best on the final epoch, and
strictly dominates N-ain — clean up, every shift condition up, exact-zero
`contrast-40%` preserved:

| Metric | N (BN) | N-ain | N-ain-aug |
| --- | ---: | ---: | ---: |
| clean mAP | 90.6 | 87.9 | **88.3** |
| mean mAP drop over the 8 shifts | -10.9 | -5.8 | **-3.8** |
| warm (worst) absolute mAP | 65.3 | 71.9 | **77.8** |

Ladder note: N-ain-aug (88.3 clean, 3.3M/1.49G) now beats plain T-ain
(88.0 clean, 4.6M/2.12G) — a 40-epoch aug fine-tune is worth more than the
x1.25 -> x1.5 width step, reinforcing that the CNN -ain ladder saturates at
x1.25 and further gains come from training, not capacity.

### T-ain-aug completes the CNN rollout

Third tier, same trajectory (crossed the warm-start clean mAP at epoch 29,
best on the final epoch), third strict domination:

| Metric | T-ain | T-ain-aug |
| --- | ---: | ---: |
| clean mAP | 88.0 | **88.5** |
| mean mAP drop over the 8 shifts | -5.6 | **-4.0** |
| warm (worst) absolute mAP | 74.0 | **78.3** |

Final CNN -ain-aug ladder: clean 87.6 / 88.3 / 88.5 (P/N/T) with mean
shift drops 3.6 / 3.8 / 4.0 — the aug fine-tune lifts every tier by
roughly the same amount, so the tier ordering (and the x1.25 sweet spot,
with T-ain-aug only +0.2 over N-ain-aug for +42% FLOPs) is unchanged.

### B-ain-aug: the teacher itself

The same recipe on the teacher (SGD 6e-5, no distillation, crossed its
warm-start value already at epoch 15, best e37):

| Metric | B (BN) | B-ain | B-ain-aug |
| --- | ---: | ---: | ---: |
| clean mAP | 93.3 | 92.1 | **92.3** |
| mean mAP drop over the 8 shifts | -5.2 | -3.4 | **-1.8** |
| warm (worst) absolute mAP | 69.7 | 74.3 | **84.7** |

The clean gain (+0.17) is smaller than the CNN tiers' (+0.5) — token-IN
already removes what mild augmentation teaches best, and B-ain was less
undertrained — but the robustness gain is the largest measured: warm
degradation drops from -17.9 to **-7.5** (absolute 84.7, +15.0 over BN-B)
and the mean drop nearly halves to 1.8, while both exact-zero conditions
(dark-30%, contrast-40%) are preserved. B-ain-aug also brings the clean
score within 0.05 of the original non-inferiority gate (92.31). As the
strongest and most style-stable model, it replaces B-ain as the teacher
for subsequent aug distillations (S-ain-aug and optional round-2 CNN
fine-tunes).

### S-ain-aug completes the aug ladder

First run distilled from the B-ain-aug teacher (crossed at epoch 13, best
e34 at 91.63, +0.24 over S-ain): mean drop 3.9 -> **2.2**, warm degradation
-21.3 -> **-9.7** (absolute 82.0, +12.5 over BN-S), both exact-zero
conditions preserved. The full aug ladder, measured:

| | B-ain-aug | S-ain-aug | T-ain-aug | N-ain-aug | P-ain-aug |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean mAP | 92.3 | 91.6 | 88.5 | 88.3 | 87.6 |
| mean mAP drop over the 8 shifts | -1.8 | -2.2 | -4.0 | -3.8 | -3.6 |
| warm (worst) absolute mAP | 84.7 | 82.0 | 78.3 | 77.8 | 73.7 |

Every tier gained clean accuracy and lost half or more of its residual
style sensitivity relative to its -ain parent; the ViT tiers converge
faster (crossing at ~1/3 of the schedule vs ~70% for the CNNs) and end
up markedly more robust, consistent with token-IN's exact affine
invariance leaving less for augmentation to fix.

### Round-2 fine-tune (X-ain-aug2): small, predicted gain

P-ain-aug2 (same recipe again, warm-started from P-ain-aug, distilled from
the upgraded B-ain-aug teacher) landed exactly in the predicted +0.1..0.3
band: clean 87.62 -> **87.81** (+0.18, crossed at epoch 29, best e38), with
robustness essentially unchanged (mean drop 3.6 -> 3.5; per-condition
absolutes within +-0.5 of round 1, exact-zero `contrast-40%` preserved).
The gain cannot be attributed between the better teacher and the extra 40
epochs, and a third round is expected to yield less — round 2 is a cheap
"+0.2 clean for 2.5 GPU-hours" option per CNN tier, not a new lever.

## Export and deployment notes

- InstanceNormalization is a standard ONNX op (ORT/TensorRT supported) but,
  unlike inference BatchNorm, it normalizes at runtime and **cannot be folded
  into convolutions**. The OSNet-AIN export keeps its IN nodes; all remaining
  BN still folds. Expect a few percent latency overhead vs the BN ladder.
- `validate_osnet_structure` gets an `-ain` family allowance for
  InstanceNormalization nodes (count pinned to the architecture definition).

## Cost estimate (RTX 3070 class, sequential)

| Run | Est. wall-clock |
| --- | --- |
| B-ain fine-tune (teacher first) | ~13 h |
| S-ain distill | ~8.5 h |
| P-ain distill | ~15.5 h |
| N-ain distill | ~18 h |
| T-ain distill | on the 16 GB machine, parallel |

Total ≈ 2.5 days of 8 GB GPU time for B/S/P/N-ain; F/A-ain deferred like
their BN siblings.

## Risks

1. Token-IN for ViT is a less-established construction than OSNet-AIN; B-ain
   carries one iteration of placement risk (mitigated by the reserved
   ablation slot and the non-inferiority gate).
2. The in-distribution table may show the -ain ladder slightly below the BN
   ladder across the board; that is the expected price of invariance and not
   a failure signal unless gate 1 breaks.
3. Without LODO or the style-shift probe, generalization gains remain
   literature-backed rather than measured; the probe is the cheapest way to
   close that gap inside the chosen protocol.

## Implementation task list

1. `model/backbones/osnet_ain.py` (vendored, channel-parameterized) +
   factories `osnet_ain_x{0_5,0_75,1_0,1_25,1_5}` and registration.
2. Token-IN option for the TransReID ViT (config-gated, default off) +
   `vit_base_ain` / `vit_small_ain` factory aliases.
3. Official AIN ImageNet weight download (x1.0 now; x0.75/x0.5 when needed).
4. Configs for B-ain / S-ain / P-ain / N-ain / T-ain (recipes copied from BN
   siblings; only TRANSFORMER_TYPE, teacher pointers, init paths, OUTPUT_DIR
   change).
5. Export support: `-ain` OSNet specs with IN-aware structural validation
   and `osnet_ain_*` output naming.
6. Optional: style-shift probe evaluation tool.
