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
3. **Optional style-shift probe** (stays within the 5+5 protocol): evaluate
   the same unified test split under photometric perturbations (illumination
   gain, color temperature, contrast). The BN-vs-AIN delta under shift is
   the in-protocol generalization signal. Small eval-only tool; no retraining.

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
