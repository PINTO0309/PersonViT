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

**Phase 3 rollout configs** (teacher promotion confirmed): each student
warm-starts from its previous-round best and re-distills from
`B-ain-aug2-cam` with **student-side CAMPROXY also enabled** (decision:
combine both levers — the upgraded teacher geometry via distillation and
the direct cross-camera pull — rather than attribute them separately; the
standalone Phase-2 attribution arms remain available but are superseded):

| Run | Config | Warm start |
| --- | --- | --- |
| S-ain-aug2 | `vit_small_8gb_distill_ain_aug2.yml` | S-ain-aug 91.63 |
| P-ain-aug3 | `osnet_p_8gb_distill_ain_aug3.yml` | P-ain-aug2 87.81 |
| N-ain-aug3 | `osnet_n_8gb_distill_ain_aug3.yml` | N-ain-aug2 88.44 |
| T-ain-aug3 | `osnet_t_8gb_distill_ain_aug3.yml` | T-ain-aug2 88.61 |

Note: the initial distillation loss is ~19 (vs ~0.4 when the teacher
matched the students' lineage) — the L_cam teacher's logit/similarity
geometry moved substantially, which is exactly the new signal being
transferred; expect stronger early reshaping than in previous rounds.

**Phase 3 results:**

| Run | clean mAP | vs prev round | probe mean drop | officials |
| --- | ---: | ---: | ---: | --- |
| S-ain-aug2 | **92.68** | **+1.05** | 2.2 -> **1.4** | all five improved (msmt17 +3.0, duke_occ +2.9, market +0.7, cuhk03np +0.9, occ_reid +0.1) |
| P-ain-aug3 (40 ep) | 85.95 | -1.86 | — | reshaping incomplete -> aug4 continuation queued |
| N-ain-aug3 (40 ep) | 85.9 @e40 | -2.5 | — | same (best file is the pre-reshape epoch 1, 86.05) |
| T-ain-aug3 (40 ep) | 85.8 @e40 | -2.8 | — | same (best file is the pre-reshape epoch 1, 86.30) |

S-ain-aug2 findings: the teacher's L_cam gain (+1.05) transferred to the
S student without attenuation, and — unlike the NPO episode — the unified
gain agrees with every official split, with the biggest official jumps on
the many-camera msmt17 (+3.0) and duke_occ (+2.9). Robustness improved in
the same run (warm absolute 82.0 -> 88.2, exact-zero conditions
preserved). Trajectory as predicted: epoch-1 dip to 91.2 (initial distill
loss ~18, reshaping toward the new teacher geometry), then a monotone
climb through epoch 39.

**CNN follow-up arms (P tier, all warm-started from the aug2 best 87.81):**
the aug4 continuation plateaued ~1 mAP below aug2 with terminal
relational-KD loss 0.77 (vs 0.43 under the old teacher; ViT-S reaches
0.49 against the same new teacher), diagnosing a representational
mismatch — the conv student lacks the global pairwise-relation modeling
behind the L_cam teacher's geometry, and capacity is not the axis (P/N/T
2.2-4.6M behaved identically in aug3). Three probes:

- `osnet_p_8gb_distill_ain_relw10.yml` — anchor relaxed to REL_WEIGHT 10.
- `osnet_p_8gb_distill_ain_nokd.yml` — anchor off (with aug4's REL 30
  these form an anchor-strength gradient 30/10/none).
- `osnet_p_8gb_distill_ain_attn{,_nokd}.yml` — architecture axis:
  `osnet_ain_x1_0_attn` adds one bottlenecked residual self-attention
  block after conv5 (512->128->512, 4 heads, +0.20M params, ~+4% MACs;
  zero-gate identity at init, verified 0.0 diff loading the aug2 best).
  Primary metric for the attn+REL30 arm is the terminal relational-KD
  loss (success: well below 0.77, toward the ViT-S reference 0.49); the
  learned gate magnitude is a secondary readout of how much attention the
  model recruits. Adoption for any arm: >= 88.0 clean with official
  splits agreeing and no style-shift regression.

**nokd arm result (P, 100 ep, judged on the final state — the "best" file
is the epoch-1 pre-dip artifact):** dips to 85.6 by epoch 5-10 even with
NO teacher anchor, recovers to **87.8-87.9** ≈ the aug2 level; Cam loss
converges healthily (3.3 -> 0.87). Gate (>= 88.0) missed -> not adopted.
Two corrections to the running interpretation: (a) the dip is a *generic
re-convergence transient* (fresh Adam 1e-4 + strong aug + L_cam on a
converged small CNN), not teacher-driven as first assumed; (b) the anchor
harm still stands, now via endpoints — from the same dip, no-anchor
recovers to 87.8 in 100 ep while the REL-30 new-teacher anchor caps
recovery at ~86.7 after 140 ep (terminal relational-KD 0.77). Also:
student-side L_cam alone adds ~0 clean mAP for the CNN (vs +1.05 on
ViT) — consistent with the missing global-relation modeling that the
attention arms probe.

CNN aug3 findings (all three tiers identical): where the ViT student
crossed its dip in 2 epochs, the small CNNs fell to ~81 mAP by epoch 5-10
while reshaping toward the far-away new-teacher geometry and were still
recovering (+0.5 mAP / 10 epochs) when the 40-epoch schedule ran out —
ending 1.9-2.8 below their aug2 bests (and for N/T the "best" file is the
pre-reshape epoch 1). Not a failure to learn but an incomplete transient:
the aug4 configs continue from the aug3 *final* states
(`checkpoint_last.pth`; smoke-verified initial distill ~0.96 vs aug3's
~19, so the dip does not repeat) with a fresh 100-epoch cosine. Success
bar: beat the aug2 bests (P 87.81 / N 88.44 / T 88.61); otherwise the CNN
tiers keep the aug2 lineage as final.

## Measurement checklist per arm

1. unified test (train log best + `eval_official.py`-style final check)
2. `tools/eval_per_domain.py` (d01 for C1, d03/d04 for C2)
3. `tools/eval_official.py` (occluded splits for C2)
4. `tools/eval_style_shift.py` (robustness non-regression gate)
