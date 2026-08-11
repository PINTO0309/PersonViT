# Paper findings and fine-tune ablation plan

Source papers (PDFs under `docs/`), all CVPR 2022:

| PDF | Paper | Core idea |
| --- | --- | --- |
| 2204.09331 | NFormer | Neighbor Transformer aggregating features across the retrieval set; plug-in on a frozen backbone |
| 2112.08740 | FED | Occluded ReID: occlusion-patch augmentation (NPO), mask-supervised part erasing (OEM), feature-level multi-person synthesis (FDM) |
| 2203.14675 | PPLR | Unsupervised ReID, but with transferable parts: camera-aware proxy contrastive loss, part-ensemble self-distillation, adaptive label smoothing |
| 2204.06890 | CAL | Clothes-changing ReID: clothes-adversarial loss over (ID x clothes) classes; late-loss-injection schedule |

Screening criteria: usable as a **warm-start fine-tune component** (the
validated 40-epoch low-LR recipe of the aug rounds), compatible with the
single-image ONNX deployment path, measurable with the fixed unified test
split + per-domain + official-split + style-shift toolchain.

## Candidate components (adopted for ablation)

### C1. Camera-aware proxy contrastive loss (`L_cam`, from PPLR)

Per (ID, camera) pair, keep a momentum feature proxy; each sample pulls
toward same-ID/other-camera proxies and pushes the top-50 hard negative
proxies (InfoNCE, tau=0.07, weight 0.5). Directly shrinks the cross-camera
intra-class variance — the core difficulty of ReID — and our unified set
has the required structure (33 global cameras, ID x camera in every file
name). PPLR reports the largest gains on the many-camera MSMT-like data
(unsupervised baseline mAP +11.7 on MSMT17); the supervised transfer is the
experiment. Watch item: occluded domains have few cameras, so the loss is
dominated by d00-d02.
Cost: ~200-300 lines (proxy bank + loss), negligible train-time overhead.

### C2. NPO occlusion augmentation (from FED)

Paste realistic occluder patches (cropped once from unified-train
backgrounds: vehicles, pillars, umbrellas, ~30 patches) onto a random
corner, sized 1/4-1/2 of the image side. FED reports Occluded-Duke R1
+4.9pt over RandomErasing (ViT-B). Light version first (no mask
supervision); the mask-supervised OEM head is a phase-2 follow-up only if
C2 wins. Config-gated like CJ/BLUR (`INPUT.NPO_PROB`), evaluated primarily
on d03/d04 and the occluded official splits. Interaction to resolve:
replace vs. combine with RandomErasing (FED suggests NPO > RE, keep RE at
p=0.25 in the combined arm).
Cost: transform ~100 lines + one-time patch curation; zero overhead.

### C3. Cosine-classifier CE (normalized weights + temperature, from CAL)

Replace the BNNeck CE with an L2-normalized classifier at tau=1/16
(CAL's degenerate form on single-clothes data). Reported upside on top of
triplet is small (+0.3-0.9 mAP) but the risk and cost are minimal.
Constraint: the logit-KD term compares student logits against the
teacher's unnormalized logits — run this arm with LOGIT_WEIGHT 0 (keep
relational KD, weight 30) to avoid a scale mismatch.
Cost: ~20 lines; zero overhead.

### C4. k-reciprocal re-ranking (eval-time option; NFormer/PPLR both cite it)

`--rerank` flag on the eval tools as a measurement option. No training,
no deployment impact; main README tables stay non-reranked to preserve
comparability — reranked numbers are reported alongside as reference.
Cost: small (established implementation), evaluation time only.

## Deferred / rejected (with reasons)

- **NFormer module** (deferred, exploratory): its aggregation needs the
  whole gallery at query time — incompatible with the single-image ONNX
  path, and our students inherit nothing from it directly. Two indirect
  routes remain worth a note: (a) eval-time aggregation over gallery
  features as a cheap re-ranking alternative, (b) batch-level NFormer on
  the frozen teacher to sharpen relational-KD targets. Revisit after the
  C1-C3 round.
- **FED's FDM** (rejected for now): heaviest implementation, gains
  confined to multi-person-crop scenarios, and FED itself trails TransReID
  on holistic Market mAP.
- **CAL proper** (rejected for now): needs (pseudo) clothes labels; the
  unified set is short-term/same-clothes and our benchmarks cannot measure
  clothes-changing gains. Keep the late-loss-injection schedule as a
  general rule (it matches the warm-start recipe already in use).
- **PGLR/AALS part self-distillation** (deferred): needs part heads and
  epoch-wise k-NN infrastructure; role partially overlaps the existing
  teacher KD. Candidate for a later round if C1 wins and part heads get
  built for OEM anyway.

## Ablation design (teacher first, then students)

**Phase 1 — primary screen on the teacher (B-ain-aug)**. Warm-start
fine-tune of B-ain-aug (92.26), SGD 6e-5, 40 epochs, warmup 5, CJ/BLUR on
— a "B round 2". Screening on the teacher is preferred for two reasons:
(a) students carry a strong relational-KD anchor (weight 30) that can mask
component effects, while the teacher trains without distillation, so the
measurement is unconfounded; (b) a teacher-level win propagates to every
student through re-distillation (proven leverage: the B-ain -> B-ain-aug
teacher upgrade). At ~1.6 GPU-hours per arm (RTX PRO 6000) the teacher is
also the cheaper experimental unit.

| Arm | Config | Change vs control | Adopt if |
| --- | --- | --- | --- |
| A0 control | `vit_base_8gb_ain_aug2_ctrl.yml` | none (plain round 2) | (attribution baseline) |
| A1 `L_cam` | `vit_base_8gb_ain_aug2_cam.yml` | + camera-proxy InfoNCE (w 0.5) | unified mAP > A0 + 0.2 |
| A2 NPO | `vit_base_8gb_ain_aug2_npo.yml` | + `INPUT.NPO_PROB 0.5`, RE 0.25 | d03/d04 + occluded official > A0 + 0.5, clean >= A0 - 0.1 |
| A3 cosine-CE | `vit_base_8gb_ain_aug2_cosce.yml` | normalized classifier, tau=1/16 (no KD conflict on the teacher) | unified mAP > A0 + 0.2 |

All arms additionally run the style-shift probe; any arm that worsens the
mean shift drop by >0.3 is rejected regardless of clean gains.

**Phase 2 — student-side check on P (winners only)**. The already-prepared
P arms (`osnet_p_8gb_distill_ain_aug3_{ctrl,cam,npo,cosce}.yml`, warm-start
from the P-ain-aug2 best 87.81, B-ain-aug teacher) answer a different
question: does the component still help **under the KD anchor**? Run the
ctrl arm plus the Phase-1 winners only. The cosce student arm keeps
LOGIT_WEIGHT 0 (teacher-logit scale mismatch).

**Phase 3 — rollout**. Combine winning components, re-train the teacher
arm as the new B teacher if it won, re-distill/fine-tune S/T/N/P, re-run
the standard evaluation suite, and fold numbers into the README rows.

**Phase 0 (independent, no training)**: implement `--rerank` (C4) and
publish reference reranked numbers for the current flagship models.

## Implementation status

C1-C3 and the four round-3 arm configs are implemented (config-gated, all
defaults off — no existing config changes behavior):

| Piece | Where |
| --- | --- |
| C1 loss | `loss/cam_proxy.py` + `CAMPROXY.*` config group; integrated in `processor.do_train` (logged as `Cam:`); unit-tested (gradient direction, single-camera zero, AMP) |
| C2 aug | `datasets/npo.py` (`INPUT.NPO_PROB` / `INPUT.NPO_PATCH_DIR`); patch set from `tools/build_npo_patches.py` (deterministic border-strip curation, gitignored — regenerate per machine) |
| C3 classifier | `MODEL.COS_CLASSIFIER` / `MODEL.COS_TEMPERATURE` in `build_transformer` (checkpoint keys unchanged) |
| Arm configs (Phase 1, teacher) | `configs/reid/vit_base_8gb_ain_aug2_{ctrl,cam,npo,cosce}.yml`, warm-started from the B-ain-aug best (92.26) |
| Arm configs (Phase 2, student) | `configs/reid/osnet_p_8gb_distill_ain_aug3_{ctrl,cam,npo,cosce}.yml`, warm-started from the P-ain-aug2 best (87.81) |

Launch Phase 1 (from `transreid_pytorch/`; generate patches once per
machine first):

```shell
uv run python tools/build_npo_patches.py
uv run python train.py --config_file configs/reid/vit_base_8gb_ain_aug2_ctrl.yml
uv run python train.py --config_file configs/reid/vit_base_8gb_ain_aug2_cam.yml
uv run python train.py --config_file configs/reid/vit_base_8gb_ain_aug2_npo.yml
uv run python train.py --config_file configs/reid/vit_base_8gb_ain_aug2_cosce.yml
```

Phase 2 runs the corresponding `osnet_p_8gb_distill_ain_aug3_*` configs
for the ctrl arm and the Phase-1 winners.

C4 (`--rerank`) is not yet implemented (phase 0, independent of training).

## Phase 1 results (teacher arms)

| Arm | clean mAP | vs A0 | mean shift drop | warm absolute | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| A0 ctrl | 92.31 | — | -1.9 | 83.7 | baseline (plain round 2 = +0.05 over B-ain-aug, confirming the ~0 prediction) |
| A1 `L_cam` | **93.36** | **+1.05** | **-1.3** | **89.1** | **ADOPTED** — clears the +0.2 gate five-fold and *improves* robustness |
| A2 NPO | 92.57 | +0.26 | -1.7 | 85.7 | **REJECTED** — fails its occlusion gate (see below) |
| A3 cosine-CE | 92.44 | +0.13 | -1.8 | 85.0 | **REJECTED** — positive but under the +0.2 gate |

A1 findings: the gain is monotone from epoch 2 (the component acts
immediately, unlike the LR-tail-driven plain rounds); clean 93.36 now
**exceeds the BN teacher B (93.31)** — L_cam recovered the entire -ain
in-distribution cost; and robustness improved alongside (mean drop
1.9 -> 1.3, warm absolute +5.4, both exact-zero conditions preserved) —
consistent with camera invariance and style invariance being the same
axis viewed from two sides. Cross-camera intra-class variance was
evidently the largest term ID+triplet left on the table. Per-domain
(within mAP, A0 -> A1): the many-camera hardest domain d01 gains most
(+1.32) as hypothesized, but the lift is broad — d03 +1.25, d04 +1.22,
d00 +1.05 — with only the saturated d02 flat (+0.20).

A2 findings: clean nudged up (+0.26) and the probe stayed healthy (mean
drop 1.9 -> 1.7), but the full official-split sweep (all five source
datasets, ctrl vs npo) came back **negative across the board**:

| domain | official split | A0 ctrl | A2 npo | delta | unified within delta |
| --- | --- | ---: | ---: | ---: | ---: |
| d00 | cuhk03np | 98.24 | 98.01 | -0.23 | -0.04 |
| d01 | msmt17 | 91.08 | 90.35 | -0.73 | +0.30 |
| d02 | market | 98.17 | 97.99 | -0.18 | +0.19 |
| d03 | duke_occ | 93.03 | 92.11 | -0.92 | +0.39 |
| d04 | occ_reid | 99.70 | 99.21 | -0.49 | -0.18* |

(*unified d04 -0.03 within; official protocol shown.) Two readings, both
partly supported: (a) **double occlusion** — pasting on the genuinely
occluded d03/d04 train images damages heavily-occluded-view features;
still the best explanation for duke_occ being the worst (-0.92, its
official queries are all-occluded) where the mixed-query unified d03
improved (+0.39). FED avoided this because Occluded-Duke's *train* split
is mostly holistic; ours is not. (b) **distribution familiarity, not
robustness** — the unified test gains flip sign on every official split
(most tellingly holistic msmt17: within +0.30 vs official -0.73), so the
auto-patches (border strips drawn from the training pool itself) taught
the model the augmented unified distribution rather than transferable
occlusion invariance. Lesson recorded: a unified-test gain alone cannot
adopt a component; official splits must agree (L_cam passes both). (c,
minor) halving RandomErasing traded away a proven augmentation. A
domain-conditional variant that would separate (a) from (b) — paste on
d01/d02 only, RandomErasing restored to 0.5 — was implemented
(`INPUT.NPO_EXCLUDE_DOMAINS`, a domain-conditional dataset wrapper in
`datasets/npo.py`, exclusion verified per domain) but **dropped from the
validation queue by decision**: with distribution familiarity (b) as the
dominant reading, even a positive result would not lead to adoption, and
L_cam already lifts the occluded domains more (+1.2) than NPO ever
targeted. The NPO line is closed; the domain-conditional mechanism stays
available (default off) for any future occlusion-augmentation attempt
with properly curated patches.

A3 findings: direction positive everywhere (clean +0.13, probe mean
1.9 -> 1.8, warm absolute +1.3) but below the adoption gate and below the
paper's with-triplet band (+0.3..0.9) — BNNeck + soft triplet evidently
already shape the angular structure cosine-CE targets. No harm, no
adoption; not carried to Phase 2.

**Phase 1 conclusion**: one clear winner. `L_cam` is adopted (+1.05 clean,
official and unified agree, robustness improved); NPO and cosine-CE are
rejected. Phase 2 = the P-tier `cam` arm (does `L_cam` still add under the
relational-KD anchor?) plus its ctrl; Phase 3 = promote
`B-ain-aug2-cam` (93.36) to teacher and re-distill/fine-tune the ladder.

## Measurement checklist per arm

1. unified test (train log best + `eval_official.py`-style final check)
2. `tools/eval_per_domain.py` (d01 for C1, d03/d04 for C2)
3. `tools/eval_official.py` (occluded splits for C2)
4. `tools/eval_style_shift.py` (robustness non-regression gate)
