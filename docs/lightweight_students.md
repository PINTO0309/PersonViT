# Lightweight student backbones: T / N / P / F / A design study

Design study for the lightweight student ladder below ViT-S/16, trained by
distillation on the unified `reid` dataset. Tier letters follow metric
prefixes: **T**iny, **N**ano, **P**ico, **F**emto, **A**tto.

## Revision (2026-08-07): the ladder below S is all-OSNet

Measured results overturned the original ViT plan for T and N. The distilled
OSNet x1.0 (2.2M) surpassed both the 192x12 ViT (81.1 mAP, at 2.5x its size)
and the 256x12 ViT (89.1 mAP, at 4.4x its size) on the unified test set, so
the tiers were reassigned:

| Tier | Family | Shape | Params | GFLOPs @256x128 | Embedding dim | Init source |
| --- | --- | --- | ---: | ---: | ---: | --- |
| S (kept) | ViT | 384 / 12 layers / 6 heads | 22.0M | 2.94 | 384 | PersonViT checkpoint0220, distilled from B |
| **T** | CNN | **OSNet x1.5** (96/384/576/768) | 4.6M | 2.12 | 512 | Function-preserving width expansion (tools/init_width_expand.py) |
| **N** | CNN | **OSNet x1.25** (80/320/480/640) | 3.3M | 1.49 | 512 | Function-preserving width expansion |
| **P** | CNN | OSNet x1.0 | 2.2M | 0.98 | 512 | OSNet ImageNet zoo |
| **F** | CNN | OSNet x0.75 | 1.3M | 0.57 | 512 | OSNet ImageNet zoo |
| **A** | CNN | OSNet x0.5 | 0.6M | 0.27 | 512 | OSNet ImageNet zoo |

The former ViT candidates — N as DeiT-Tiny-shaped 192x12, T-a as
layer-dropped 384x6, T-b as width-selected 256x12 — are **retired**; their
sections below are kept as the experimental record that motivated the
switch. Custom multipliers x1.25/x1.5 have no ImageNet zoo weights and are
initialized by zero-padded, function-preserving expansion of a trained
narrower OSNet (ImageNet x1.0 or, preferably, the distilled tier-P model).

## [RETIRED] T as layer-dropped ViT-S (measured record)

`D=384, L=6, H=6` — exactly half of S (10.9M) with the same embedding width.

- **Inheritance**: copy patch embedding, positional embedding, cls token and
  final LayerNorm from S unchanged; copy 6 of the 12 transformer blocks.
  Default block selection `{0, 2, 4, 6, 8, 11}` (uniform coverage while
  always keeping the final block, which feeds the BNNeck); ablate against
  `{0, 2, 4, 6, 8, 10}`. Source checkpoint: the distilled S student
  (`logs/reid_vit_small_8gb_distill/transformer_best_*.pth`) — it is both
  domain-adapted and teacher-aligned. BNNeck and classifier shapes also match
  S, so they can be inherited too.
- **Deployment bonus**: the embedding stays 384-dimensional, so T is a
  drop-in replacement for S in any gallery built with S-compatible pipelines
  (dimensions match; re-extraction is still required since the metric space
  differs).
- **Distillation**: teacher = B by default. Because T shares the 384
  embedding width with S, an S teacher additionally allows
  `DISTILL.EMBED_WEIGHT > 0` (direct cosine embedding transfer) — run both
  and keep the better model.
- Implementation: `vit_t_patch16_224_TransReID` factory (depth=6) plus
  `tools/init_layer_drop.py`, which renames the selected S blocks
  `blocks.{0,2,4,6,8,11} -> blocks.{0..5}` and writes a T-format checkpoint
  loadable via `MODEL.PRETRAIN_CHOICE: 'self'`.

**Measured outcome (B teacher, 60 epochs, batch 64, RTX 3070):** the
depth-12 width-ladder variant wins. Layer-dropped `384x6` (above) reached
88.3 mAP; `256x12` (`vit_t256_patch16_224_TransReID`, width-selection init
via `tools/init_width_select.py`) reached **89.1 mAP** despite its init
transferring almost no function (0.6 mAP zero-shot vs the warm layer-drop
start), crossing the `384x6` curve at epoch 20 — at this tier depth matters
more than width, and more than init quality. The S-teacher + embedding-KD
ablation for `384x6` tracked only ~0.3 mAP above its B-teacher baseline
mid-run and was stopped (resumable state kept in
`logs/reid_vit_t_8gb_distill_s2b/`). Caveats: `256x12` trains ~11% slower
per step than `384x6` at equal batch (deeper stack, smaller GEMMs); batch-1
inference latency is unmeasured; both variants remain below the 1-2 mAP gate
vs distilled S (best: 3.1 mAP behind).

## [RETIRED] N as DeiT-Tiny-shaped ViT (measured record)

`D=192, L=12, H=3` (~5.5M) is exactly DeiT-Tiny, and
`vit_tiny_patch16_224_TransReID` already exists and is registered in
`__factory_T_type`. Init from timm DeiT-Tiny ImageNet weights through the
existing `PRETRAIN_HW_RATIO` position-embedding resize path. Fallback:
width-selection from T.

## P / F / A — OSNet CNN students (now also T and N via custom multipliers)

Rationale for switching families at ≤3M parameters: pure ViTs at this scale
historically trail ReID-specialised CNNs (OSNet x1.0: 2.2M / 0.98 GFLOPs is
the README's own baseline), and OSNet's width-multiplier family lines up
with the P/F/A targets while providing ImageNet-pretrained weights per tier.

- **P = OSNet x1.0** (2.2M), **F = OSNet x0.75** (1.3M), **A = OSNet x0.5**
  (0.6M). GFLOPs from the official Torchreid model zoo.
- **Integration**: a thin adapter class gives OSNet the interface
  `build_transformer` expects — `forward(x, cam_label=None, view_label=None)`
  (camera/view ignored), `in_planes = feature_dim`, and a
  `load_param(path, hw_ratio)` that maps ImageNet zoo weights. Registered in
  `__factory_T_type` as `osnet_x1_0` / `osnet_x0_75` / `osnet_x0_5`, the rest
  of the pipeline (BNNeck head, softmax+triplet, distillation, best/resume)
  works unchanged. ViT-specific config keys (STRIDE_SIZE etc.) are ignored.
- **Vendoring**: implement OSNet as a single self-contained
  `model/backbones/osnet.py` (Torchreid architecture, MIT — keep the license
  header) rather than adding the torchreid dependency to the pinned
  environment.
- **Embedding dim**: OSNet ends in a global-pool + fc producing
  `feature_dim=512` for every multiplier; keep 512 (the fc is cheap and the
  zoo weights include it). If gallery storage matters later, `feature_dim`
  is a constructor argument and can be shrunk per tier at the cost of
  re-initializing that fc.
- **ViT-to-CNN distillation**: relational KD is the primary signal
  (architecture- and dimension-agnostic); logit KD applies unchanged since
  teacher and student share the 7,719-identity space. Expect to lean harder
  on the teacher: sweep `REL_WEIGHT` in {30, 60, 120} for the CNN tiers.

## Distillation strategy — teacher-assistant chain

B (86.5M) to A (0.6M) is a 144x capacity gap; plain KD was expected to
degrade beyond roughly 10x, so a teacher-assistant chain was planned.
**Measured reality: the direct B teacher won every comparison run** — the
S-teacher ablation for the ViT T tier gained only ~0.3 mAP mid-run, and both
the 192x12 ViT (15.7x gap) and OSNet x1.0 (39x gap) trained fine directly
from B. Default: teacher = B for every tier. TA-chain runs (e.g. F from P,
A from F) remain cheap config-only ablations if a small tier stalls.

## Training configuration starting points (single GPU, AMP)

All tiers fit the 8 GB work PC with batch 64 (16 x 4); the teacher forward
dominates step time for small students (teacher outputs cannot be
precomputed because augmentations are sampled per step). Longer schedules
for smaller tiers — distillation is the dominant signal.

| Tier | Batch | Base LR | Epochs | Notes |
| --- | ---: | ---: | ---: | --- |
| T (OSNet x1.5) | 64 | 3.5e-4 | 100 | Adam, cosine, warmup 10, wd 5e-4 — classic BNNeck CNN recipe, as validated on P |
| N (OSNet x1.25) | 64 | 3.5e-4 | 100 | |
| P (OSNet x1.0) | 64 | 3.5e-4 | 100 | |
| F (OSNet x0.75) | 64 | 3.5e-4 | 100 | |
| A (OSNet x0.5) | 64 | 3.5e-4 | 120 | |

On the 96 GB machine, batch 256 / linear-scaled LR applies as for S/B.

## Evaluation gates

Reference points on the unified test set: teacher B = 93.3 mAP; distilled S =
TBD (running). Gates, replacing the earlier OSNet-external comparison (P now
*is* OSNet):

1. Every distilled CNN tier must beat the same architecture trained on the
   unified dataset **without** distillation (one ablation run per tier).
2. T within ~1-2 mAP of distilled S; N within ~3-5. P/F/A are exploratory —
   report accuracy/FLOPs trade-off; a tier that adds no Pareto value over
   its neighbours is dropped from the release set.
3. Report batch-1 latency and model size per tier on the target runtime
   (ONNX) alongside mAP/Rank-1.

## Follow-up implementation tasks

1. `vit_t_patch16_224_TransReID` (depth-6) factory + `__factory_T_type`
   entries; N reuses the existing `vit_tiny` factory.
2. `tools/init_layer_drop.py` — S -> T block inheritance (default
   `{0,2,4,6,8,11}`, selectable).
3. `model/backbones/osnet.py` (vendored, MIT header) + OSNet adapter class +
   factory registration (`osnet_x1_0` / `osnet_x0_75` / `osnet_x0_5`);
   ImageNet zoo weight loader in `load_param`.
4. Configs per tier: `vit_t_8gb_distill.yml`, `vit_n_8gb_distill.yml`,
   `osnet_p_8gb_distill.yml`, `osnet_f_8gb_distill.yml`,
   `osnet_a_8gb_distill.yml` (+96gb variants), teacher pointers per the TA
   chain.
5. Ablations: block-selection variants for T; S-teacher with EMBED_WEIGHT
   for T; direct-from-B teachers (T/N/P); REL_WEIGHT sweep for CNN tiers;
   no-distillation baselines (gate 1).
6. Extend `export_onnx.py` naming for the new tiers once models exist.
