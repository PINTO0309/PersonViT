# Lightweight student backbones: T / N / P / F / A design study

Design study for the lightweight student ladder below ViT-S/16, each tier
roughly halving the parameters of the previous one, trained by distillation
on the unified `reid` dataset. Tier letters follow metric prefixes:
**T**iny, **N**ano, **P**ico, **F**emto, **A**tto. T and N stay Vision
Transformers; **P and below are CNN students** (decision: pure ViTs are not
competitive at ≤3M parameters). Implementation is a follow-up task; this
document fixes architectures, initialization, distillation strategy and
expected footprints.

## Ladder overview

| Tier | Family | Shape | Params | GFLOPs @256x128 | Embedding dim | Init source |
| --- | --- | --- | ---: | ---: | ---: | --- |
| S (existing) | ViT | 384 / 12 layers / 6 heads | 22.0M | 2.94 | 384 | PersonViT checkpoint0220 |
| **T** | ViT | **384 / 6 layers / 6 heads** | 10.9M | ~1.51 | 384 | **direct layer inheritance from S** |
| **N** | ViT | 192 / 12 layers / 3 heads | ~5.5M | ~0.74 | 192 | DeiT-Tiny ImageNet (timm) |
| **P** | CNN | OSNet x1.0 | 2.2M | 0.98 | 512 | OSNet ImageNet zoo |
| **F** | CNN | OSNet x0.75 | 1.3M | 0.57 | 512 | OSNet ImageNet zoo |
| **A** | CNN | OSNet x0.5 | 0.6M | 0.27 | 512 | OSNet ImageNet zoo |

Parameter ratios between neighbours: 22.0 → 10.9 (0.50) → 5.5 (0.50) → 2.2
(0.40) → 1.3 (0.59) → 0.6 (0.46). The CNN tiers use the standard OSNet width
multipliers instead of exact halves so that the published ImageNet weights
and model-zoo baselines remain directly usable.

## T — layer-dropped ViT-S (requirement: direct weight inheritance from S)

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

## N — DeiT-Tiny-shaped ViT (already implemented)

`D=192, L=12, H=3` (~5.5M) is exactly DeiT-Tiny, and
`vit_tiny_patch16_224_TransReID` already exists and is registered in
`__factory_T_type`. Init from timm DeiT-Tiny ImageNet weights through the
existing `PRETRAIN_HW_RATIO` position-embedding resize path. Fallback:
width-selection from T.

## P / F / A — OSNet CNN students

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

B (86.5M) to A (0.6M) is a 144x capacity gap; plain KD degrades beyond
roughly 10x. Recommended chain (every arrow is just a `DISTILL` config
change; any trained student can act as teacher):

```
B (93.3 mAP) ──> S (running) ──> T ──> N ──> P ──> F ──> A
           └─ ablations: T and N directly from B; P directly from N vs T
```

- T: teacher = B (8x, direct) and teacher = S with `EMBED_WEIGHT` enabled —
  keep the better.
- N: teacher = T (2x). P: teacher = N (2.5x). F: teacher = P (1.7x).
  A: teacher = F (2.2x).
- Direct-from-B ablations are cheap and worth one run per tier down to P;
  below P the gap (>39x) makes them unpromising.

## Training configuration starting points (single GPU, AMP)

All tiers fit the 8 GB work PC with batch 64 (16 x 4); the teacher forward
dominates step time for small students (teacher outputs cannot be
precomputed because augmentations are sampled per step). Longer schedules
for smaller tiers — distillation is the dominant signal.

| Tier | Batch | Base LR | Epochs | Notes |
| --- | ---: | ---: | ---: | --- |
| T | 64 | 4e-4 | 60 | SGD, cosine, warmup 10 — same recipe as S |
| N | 64 | 4e-4 | 80 | |
| P | 64 | 3.5e-4 | 100 | OSNet convention (Torchreid) prefers slightly lower LR with AMSGrad/SGD; start SGD 3.5e-4 x cosine, ablate 4e-4 |
| F | 64 | 3.5e-4 | 100 | |
| A | 64 | 3.5e-4 | 120 | |

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
