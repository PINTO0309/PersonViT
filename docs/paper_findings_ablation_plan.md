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

**attn_nokd v1 result — the block died, not failed:** the run matched the
plain nokd arm exactly (final ~87.9), and the checkpoint shows why:
`gate = -1e-41`, `reduce.weight` norm **0.0**. An exactly-zero gate blocks
every gradient into the block (only the scalar gate itself receives a
gradient, proportional to the near-zero correlation between the random
attention output and the loss gradient), and weight decay (5e-4) eroded
the waiting parameters to zero — a bootstrap deadlock, so the arm says
nothing about attention's value. Design lesson recorded honestly: the
scalar zero-gate (adopted from the ChatGPT-proposed design) is fragile
here; a zero-initialized output projection would have received a
full-rank gradient at init. Fix applied for the v2 reruns: gate init
0.01 (output perturbation ~0.008, warm start effectively preserved) plus
a weight-decay exemption for `.attn.` parameters in make_optimizer;
OUTPUT_DIRs bumped to `..._attn2` / `..._attn_nokd2`. The REL-30 attn arm
must not be run (or re-run) without this fix.

**Attention line closed (pre-registered probe verdict):** with the
bootstrap fix in place (gate alive, internals training — qkv norms grew
to 7-23 across runs), both attention variants failed both decision
signals:

- attn2 (dim 128, REL 30, judged at e21): distill curve identical to the
  attention-free aug3 at every epoch (e10 1.77 vs 1.75, e20 1.04 vs
  1.03); gate self-suppressed 0.035 -> 0.005.
- attn_full (native 512-dim, 8 heads, 30-ep probe, judged at e20):
  distill still above the aug3 reference (e20 1.067 vs 1.034) despite a
  lower-LR advantage from its shorter schedule; gate grew to 0.067 by e5,
  then collapsed to 0.013 — the same self-suppression, delayed.
- attn_nokd2 (no teacher, control): tracked the plain nokd arm exactly
  (e20 86.1 vs 86.2); without a relational loss there is nothing for
  attention to serve.

Conclusion: under relational-KD pressure the model consistently chooses
NOT to mix in attention output, at any width. The CNN's inability to
match the L_cam teacher geometry is **not** missing global-relation
modeling; the remaining suspects are deeper (student embedding width 512
vs teacher 768, conv feature statistics). CNN-tier outcome now rests on
the relw10 arm; if it fails its gate, the CNN tiers finalize on the aug2
lineage and the L_cam benefit remains ViT-only.

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

## Phase 3 final conclusion

The remaining CNN probes (`relw10`, and the aug4 continuations for N/T)
were **rejected by decision** without awaiting completion: after aug4-P
plateaued below aug2, the no-anchor arm topped out at parity, and both
attention probes failed their pre-registered signals, the expected value
of the remaining arms no longer justified their GPU time.

Final state of the ladder:

- **ViT tiers carry the L_cam lineage**: B-ain-aug2-cam (93.36, probe
  mean drop 1.3) as teacher and deployment flagship; S-ain-aug2 (92.68,
  mean drop 1.4). The paper-component campaign's net win is L_cam, worth
  ~+1.05 clean with improved robustness on both ViT tiers.
- **CNN tiers finalize on the aug2 lineage** (the documented fallback):
  P 87.81 / N 88.44 / T 88.61. The L_cam teacher geometry is not
  representable by the OSNet students at any probed anchor strength or
  attention width; their README rows and ONNX artifacts already reflect
  this lineage.

## JPEG-compression exposure (measured before any training)

Following the NPO lesson (build the measuring stick first), the probe
gained two deterministic JPEG round-trip conditions (`jpeg-q40`,
`jpeg-q20`) and the flagships were measured before considering a
training-side JPEG augmentation:

| Model | jpeg-q40 | jpeg-q20 | (worst existing: warm) |
| --- | ---: | ---: | ---: |
| B-ain-aug2-cam | -0.71 | -2.41 | -4.30 |
| S-ain-aug2 | -0.97 | -3.00 | -4.45 |
| P-ain-aug2 | -1.40 | -5.17 | -9.00 |

Verdict: at realistic stream qualities (q40) the exposure is ~1 point,
JPEG is not the worst axis for any tier (color temperature still
dominates), and no ViT-patch/DCT-block interaction appeared. The probe
conditions remain in place (usable as `SOLVER.VAL_SHIFT` too), and the
probe cache invalidates itself when the condition set grows.

**Training-side JPEG ablation (decision: run it).** Five arms, one per
tier, each warm-starting the tier's FINAL-lineage best and changing
exactly one variable — `INPUT.JPEG_PROB 0.5`, quality uniform in
`[20, 90]` (`datasets/jpeg_aug.py`, applied after the photometric
transforms since real pipelines compress after capture):

| Arm | Config | Warm start |
| --- | --- | --- |
| B | `vit_base_8gb_ain_aug2_cam_jpeg.yml` | 93.36 |
| S | `vit_small_8gb_distill_ain_aug2_jpeg.yml` | 92.68 |
| P | `osnet_p_8gb_distill_ain_aug2_jpeg.yml` | 87.81 |
| N | `osnet_n_8gb_distill_ain_aug2_jpeg.yml` | 88.44 |
| T | `osnet_t_8gb_distill_ain_aug2_jpeg.yml` | 88.61 |

The CNN arms keep their aug2 recipe and old teacher (B-ain-aug) — the
lineage decision stands; JPEG is the only delta everywhere. Adoption
gates: jpeg-q20 drop clearly reduced, clean within -0.1, the other 8
probe conditions non-regressed, official splits agreeing.

**B arm result — ADOPTED on all four gates** (baseline B-ain-aug2-cam):
clean 93.36 -> **93.50**; jpeg-q20 drop -2.41 -> **-1.32** (q40 -0.71 ->
-0.43); the eight photometric conditions improved or held (warm
-4.30 -> -3.77, contrast+ -3.36 -> -3.15, mean-8 1.31 -> 1.26, exact
zeros preserved); officials 4 up / 1 saturated-flat (msmt17 **+1.18**,
duke_occ +0.60, market +0.25, cuhk03np +0.11, occ_reid -0.10 at
ceiling). JPEG augmentation acted as a general regularizer, not just
compression hardening. New B flagship:
`logs/reid_vit_base_8gb_ain_aug2_cam_jpeg/transformer_best_e000040_map0.93500.pth`.

**S arm result — ADOPTED on all four gates** (run combined JPEG aug with
the upgraded B-jpeg teacher; baseline S-ain-aug2): clean 92.68 ->
**92.95**; jpeg-q20 -3.00 -> **-1.56** (q40 -0.97 -> -0.60); eight
photometric conditions improved or held (warm -4.45 -> -3.82, mean-8
1.45 -> 1.35, exact zeros preserved); officials 4 up / 1 ceiling-flat
(msmt17 +0.92, duke_occ +0.78, market +0.25, cuhk03np +0.11). New S
flagship:
`logs/reid_vit_small_8gb_distill_ain_aug2_jpeg/transformer_best_e000037_map0.92954.pth`.

**CNN arms — all three ADOPTED. JPEG ablation final: 5/5.** Against
their aug2 baselines:

| Tier | clean | jpeg-q20 | mean-8 | officials |
| --- | --- | --- | --- | --- |
| P | 87.81 -> **88.15** (+0.34) | -5.17 -> **-2.61** | 3.52 -> 3.43 | 4 up, occ_reid -0.13 at ceiling |
| N | 88.44 -> 88.44 (flat) | -4.75 -> **-2.59** | 3.82 -> 3.71 | flat within +-0.2 |
| T | 88.61 -> **88.69** (+0.08) | -4.73 -> **-2.49** | 3.89 -> 3.93 (within tol.) | flat within +-0.2 |

Reading: on the ViT tiers JPEG augmentation acted as a general
regularizer (clean +0.15/+0.27); on the CNNs it is a pure compression
hardening — the jpeg-q20 exposure roughly halves on every tier at zero
cost elsewhere, with P (the most exposed tier) also gaining +0.34 clean.
All five README rows and ONNX artifacts now carry the jpeg lineage.

### No-camproxy JPEG teacher arm (B-ain-aug2-jpeg)

`vit_base_8gb_ain_aug2_jpeg.yml` — same jpeg recipe, CAMPROXY off, same
warm start as the round-2 arms (B-ain-aug 92.26, identical init to A0/A1).
Result: clean **92.30** (e36) — vs A0 ctrl 92.31 the jpeg gain without
L_cam is **0.0** (the +0.15 seen on the cam lineage is not reproduced
without it), and vs cam_jpeg 93.50 the L_cam contribution stays **+1.20**
inside the jpeg recipe: the two components are additive with no overlap.
Probe: jpeg-q20 −1.36 / q40 −0.38 — **compression hardening replicates
in full without L_cam** (cam_jpeg: −1.32/−0.3); photometric mean-8 ~1.7
(warm −6.73, contrast+ −3.25) vs cam_jpeg 1.26 (warm −3.77) — the
photometric-robustness advantage belongs to L_cam, not to jpeg.
Attribution is now fully factored: AIN = architecture-level photometric
robustness, L_cam = clean +1.05..1.20 AND warm/mean photometric gains,
JPEG aug = compression hardening only, all three independent.
Primary purpose: teacher for the `osnet_p_8gb_distill_ain_aug3_jpeg`
student arm — if a no-cam jpeg teacher distills into P at the
old-teacher floor (~0.43) the CNN 0.77 floor is pinned on the L_cam
geometry itself; if 0.77 persists the L_cam-cause hypothesis falls.

**Student result — L_cam-cause CONFIRMED.** P distilled from the no-cam
jpeg teacher (40 ep, same recipe as the P jpeg arm): terminal
relational-KD **0.502** (e1 already 0.556 from the warm start — the
teacher geometry is representable from step one), best clean **88.11**
(e38) vs the P jpeg best 88.15 (old B-ain-aug teacher): parity −0.04.
The 0.77 floor is therefore specific to the L_cam teacher geometry —
not to distillation, capacity, jpeg, or the teacher's training round.
Combined with the ViT-S full transfer (0.49), the picture is: L_cam
adds ~0.27 of unrepresentable structure for a 5-IN OSNet; whether the
INs are the reason is exactly what the running stem_attn arm decides
(its terminal Distill now has a measured CNN-reachable reference of
~0.50 under a representable teacher). Probe: jpeg-q20 −2.45 / q40 −0.71
/ mean-8 3.39 vs the P jpeg best's −2.61 / 3.43 — the jpeg-hardened
teacher adds a marginal q20 gain and parity elsewhere, so the arm's
value is the causal verdict, not a new P lineage (clean −0.04 vs 88.15).

**Tail attention: FINAL closure (fifth environment).** The
`osnet_p_8gb_distill_ain_aug3_jpeg_attn` arm gave the gate its best-case
conditions — a representable teacher (floor 0.50) and a warm start from
the parent's own converged best (88.106), so any Distill improvement had
to flow through the attention block. The gate went from 0.01 straight
through zero to **−0.00067 by e10** (qkv/proj norms healthy at
3.28/1.50 — a learned "don't use it", not weight erosion), Distill
showed no sub-parent movement. KILLED at e10 by decision. Combined with
attn2 (L_cam teacher), attn_nokd2 (no teacher), attn_full (native
width), and stem_attn (IN×1): tail attention is not recruited by OSNet
under any teacher, normalization environment, or width — the 16×8
post-conv5 features hold no residual pairwise-relation information worth
modeling. The remaining attention question is the ENTRANCE placement
(sepattn arms), which is mechanistically distinct and unaffected by
this closure.

## IN-information-loss probe (stem-IN-only arm, running)

Live hypothesis after Phase 3: the OSNet-AIN students' 0.77 relational-KD
floor against the L_cam teacher is caused by the five spatial InstanceNorms
discarding the instance-specific statistics the L_cam geometry is built on
(evidence: IN clean cost is CNN −3.0/−2.7 vs ViT −1.2/−0.8; AIN→AIN
distillation converges fine at 0.39–0.41; the failure is specific to the
L_cam teacher). Probe: `osnet_ain_stem_x1_0_attn` — only the conv1 stem IN
survives (1 IN instead of 5, OSBlockINin → plain BN OSBlock), distilled
from B-ain-aug2-cam-jpeg, warm start P jpeg best 88.15, 140 epochs
(`osnet_p_8gb_distill_ain_stem_attn.yml`). PRIMARY metric: terminal
relational-KD loss (→~0.5 confirms IN as the bottleneck; ~0.77 exonerates).

Interim at e31: mAP crashed to 20.0 at e1 (4 blocks partially re-init)
and recovered to 82.4; Distill 1.08 and still descending with LR high
(8.8e-5) — the floor verdict waits for the cosine tail. The attention
gate self-suppressed AGAIN (0.043 at e7 → 0.0062 at e31): third
environment (IN×5 distilled, IN×5 no-teacher, IN×1 distilled) with the
same signature, so the tail-attention closure is environment-independent
— "IN discarded the information attention needed" is refuted as a
coupling; the IN hypothesis itself remains open pending the terminal
Distill value.

### Entrance separable-attention design (prepared, not started)

Counter-design to the closed tail-attention line: attention placed where
convs can still *use* global context (right after the stem) instead of
after aggregation. Quadratic attention is unaffordable there (2048 tokens,
+55% MACs), so the block is an O(N) separable self-attention
(MobileViTv2-style: 1-ch softmax context scores → one score-weighted key
context vector → `proj(relu(V)·context)`, gated residual, gate init 0.01
with the `.stem_attn.` weight-decay exemption). Cost: +12.5k params
(+0.58%), +25M MACs (+2.6% of P). ONNX graph is NCHW-native: no attention
matrix / head split / transposes; the entrance block's ops delta =
Softmax(axis −1) ×1, ReduceSum ×1, Reshape ×3 (`[1,1,−1]`, `[1,64,−1]`,
`[1,64,1,1]`) — the export contract needs these classes (plus
`attn_qkv_unflatten` from the stem e007 export) added before formal
adoption. The gated tail LiteSelfAttention (128-dim) is KEPT by decision
despite its three self-suppressions: entrance context may change what the
tail sees, and its gate is a free readout of that interaction. Factories:
`osnet_ain_x1_0_sepattn` (standard 5-IN) and `osnet_ain_stem_x1_0_sepattn`
(stem-IN-only), both = entrance separable + tail LiteSelfAttention; config
`osnet_p_8gb_distill_ain_stem_sepattn.yml` mirrors the stem_attn recipe.
Sequencing: start after the stem_attn verdict so the IN-reduction and
entrance-attention variables stay separable; primary readout is the two
gate trajectories (`base.stem_attn.gate` entrance / `base.attn.gate` tail)
vs the thrice-observed tail self-suppression signature.

## P-uplift round (post-closure exploration)

Multi-axis search for further P gains after the causal questions closed.
Evidence-killed axes (not to revisit): capacity (P/N/T transients
identical), plain schedule extension (aug4), direct L_cam on the CNN
(+0), tail attention (5 environments), NPO, cosine-CE.

- **Model soup (measured)**: averaging the two P jpeg bests (aug2_jpeg
  88.145 + aug3_jpeg 88.106; same warm-start basin, solution distance
  0.73%) scores **88.24 / R1 95.0** on the unified split — +0.10 over
  the best member for zero training. Probe + officials gates still
  needed before any README adoption; N-member extension open.
- **Rep overparameterization — GATE PASSED (+0.17)**: best **88.28**
  (e40) vs the 88.106 warm start, terminal rel-KD 0.494 (vs 0.502) —
  new P clean record (previous overall best 88.145). e1 held 88.1
  (function preservation confirmed in vivo); the learned rep branches
  settled at ~28% of the dw3x3 center-tap magnitude before folding.
  Folded exactly (fold_rep --verify 2.7e-6) to
  `logs/.../folded_best.pth`. Probe (folded weights): clean 88.28
  confirmed post-fold; jpeg-q20 −2.37 / q40 −0.70 / mean-8 3.34 — every
  condition at or better than the parent (−2.45 / 3.39), robustness
  gate PASSED. Officials deferred to the ladder end (run once after the
  GeM/embed steps settle the final P). Design details:
  `osnet_ain_x1_0_rep` + `osnet_p_8gb_distill_ain_aug3_jpeg_rep.yml` —
  all 60 LightConv depthwise 3x3s train with two zero-init linear
  branches (dw 1x1 + per-channel identity scale, +11.5k train-time
  params), summed pre-BN, folded EXACTLY into the 3x3 center tap by
  `tools/fold_rep.py` (parity 1.4e-5). Warm start = aug3_jpeg best
  88.106, function-preserving at step one (verified diff 0.0). ONNX:
  train-form graph 696 nodes (BNs unfused behind the branch Adds — the
  rep effect); folded graph 395 nodes, op-identical to the deployed P
  graph (only the release wrapper's L2-normalize tail differs). Judge
  vs 88.106 / rel-KD 0.502; fold before eval/export.
- **GeM pooling (implemented, ready)**: `GeM` module (learnable exponent
  p, init 1 = exact GAP, weight-decay-exempt via `.global_avgpool.p`;
  eps-clamp deviation on normalized features 2.6e-5, mAP-invisible) +
  factories `osnet_ain_x1_0_gem` / `osnet_ain_x1_0_rep_gem` + config
  `osnet_p_8gb_distill_ain_aug3_jpeg_rep_gem.yml`. Ladder step 2: warm
  start = the UNFOLDED rep best (FIXME placeholder until that run
  finishes); if rep fails its gate, switch to `osnet_ain_x1_0_gem` +
  aug3_jpeg best instead — GeM is independent and gets tested either
  way. `fold_rep.py --verify` auto-detects gem checkpoints (parity
  1.3e-5); folded deployment graph = plain P + 2 Pow (export contract
  needs the Pow pair before formal adoption). Watch the learned p
  (retrieval-typical ~3).
- **GeM — REJECTED (+0.007)**: rep_gem best 88.287 vs rep 88.280, noise
  parity. The learned exponent moved to p = 1.416 (the gradient wanted
  > 1) yet yielded nothing — distill-shaped features are already
  GAP-optimal; not worth adding 2 Pow ops to the export contract
  (parsimony, same principle as the cosce rejection). P lineage stays
  the rep best 88.280.
- **Embedding KD (implemented, ready — ladder step 3)**: loss-only
  linear projector student 512 -> teacher 768 (`embed_proj` on the
  model: joins optimizer/checkpoint, absent from forward/export),
  cosine loss gated by DISTILL.EMBED_WEIGHT + new EMBED_PROJ_DIM;
  DistillLoss takes `projector=`, processor passes it through (DDP-safe).
  Config `osnet_p_8gb_distill_ain_aug3_jpeg_rep_embed.yml`: rep arch,
  warm start rep best 88.280 (unfolded), EMBED_WEIGHT 2.0 first probe,
  representable no-cam teacher ONLY (L_cam pairing forbidden — 0.77
  floor pressure). **GATE PASSED (+0.31): best 88.591** (e38) — P
  record again, ladder total +0.45 over the old 88.145. The predicted
  e1 Distill spike (2.11) resolved by e5 (0.665); terminal combined
  Distill 0.533 (includes the 2.0-weighted embed term, not comparable
  to rep's rel+logit-only 0.494). Folded (3.1e-6) to folded_best.pth;
  probe: clean 88.59 post-fold, jpeg-q20 −2.19 / q40 −0.67 / warm
  −8.37 / mean-8 3.22 — every condition at or better than rep (−2.37 /
  3.34), robustness gate PASSED. Reading: the batch-local relational
  loss was NOT carrying each sample's absolute position in teacher
  space — the projector hint adds real signal where GeM (+0.007) found
  nothing. Officials (folded weights): all five splits improved vs the
  old P best — market .9642→.9687, msmt17 .8537→.8655, duke_occ
  .8837→.9002, cuhk03np .9752→.9785, occ_reid .9831→.9848 — the
  official-split agreement gate PASSED; README P-ain-aug rows and both
  P section tables updated to the rep_embed folded best (88.6 / 95.2 /
  97.5 / 98.1). ONNX flagship re-export still pending. Follow-ups:
  EMBED_WEIGHT 5.0 probe, N/T rollout (running).
- **N/T rollout — BOTH GATES PASSED, and S embed +0.08**: N best
  **88.930** (+0.49 vs 88.445), T best **89.000** (+0.31 vs 88.694), S
  embed best **93.029** (+0.08 vs 92.954, small as expected — S already
  tracks the L_cam teacher at 0.49). Probes all non-regressed with
  jpeg-q20 improved on every tier (N −2.59→−2.24, T −2.49→−2.22, S
  −1.53); officials improved across the board (largest: N duke_occ
  .8922→.9015, T msmt17 .8669→.8720, S msmt17 .9282→.9329). README
  rows + officials + style tables updated for S/N/T (T/N style tables
  were previously empty and are now filled); all four flagship ONNX
  pairs re-exported from the new bests (full contract passed, parity
  ~1e-7; exporter and fold_rep now strip the loss-only embed_proj key;
  fold_rep --verify made width-aware). P-uplift round final ladder:
  P 88.145→88.59, N 88.445→88.93, T 88.694→89.00, S 92.954→93.03.
  Rollout design: `osnet_ain_x1_25_rep` / `osnet_ain_x1_5_rep`
  factories + `osnet_{n,t}_8gb_distill_ain_jpeg_rep_embed.yml` —
  deliberately COMBINED arms (teacher swap to no-cam jpeg + rep + embed
  KD in one run each; attribution was established on P: parity −0.04 /
  +0.17 / +0.31). Warm starts = tier jpeg bests (N 88.445 / T 88.694),
  function preservation verified at 0.0 for both. Judge: clean vs the
  tier best, probe non-regression on folded weights.
- **S embed-KD arm (ready)**:
  `vit_small_8gb_distill_ain_aug2_jpeg_embed.yml` — the embed-KD step
  transferred to S (projector 384 -> 768; rep/GeM judged inapplicable
  to the ViT: no foldable dw+BN structure, and pooling swap breaks the
  cls-token warm start). L_cam teacher is fine for S (geometry
  representable at 0.49). Warm start S jpeg best 92.954; judge clean
  vs 92.954 + probe non-regression; no fold step (no rep branches).
- **P embed5+hint arm (ready — ladder step 4, deliberately combined by
  decision)**: `osnet_p_8gb_distill_ain_aug3_jpeg_rep_embed5_hint.yml` —
  EMBED_WEIGHT 2.0→5.0 (dose-response) + new DISTILL.HINT_WEIGHT 2.0
  (FitNets-style: pooled conv4 → loss-only hint_proj 512→768 → cosine
  toward the teacher's FINAL embedding; teacher-side surgery avoided by
  design). Backbone stashes hint_feat when collect_hint is set;
  hint-incapable backbones raise. Warm start rep_embed best 88.591
  (embed_proj reloads, hint_proj new); eval-path parity 0.0. If the
  combined arm wins, credit is shared — split only if warranted. Also
  measured: tail soup (best e38 + last e40) = 88.60, +0.01, dead axis.
  **Upgraded to a SPATIAL hint before any run** (user decision):
  HINT_MODE 'spatial' captures the teacher ViT's last-block patch tokens
  non-destructively (forward hook on blocks[HINT_BLOCK], no vendored
  code/forward/checkpoint change), reshapes them to a [768,16,8] map
  (cls dropped, row-major — order verified against the raw hook
  output), and matches the 1x1-conv-projected student conv4 map per
  position (grids align exactly at stride 16; 128x the signal of the
  pooled variant). First-run arm:
  `osnet_p_8gb_distill_ain_aug3_jpeg_rep_embed5_shint.yml`; the global
  variant is kept as the ablation fallback. Raw block output is the
  target (cosine is scale-invariant; final LN not applied).
  **GATE PASSED (+0.26): best 88.855** (e40) — P record again, ladder
  total +0.71 (88.145 → 88.28 → 88.59 → 88.86), all at zero inference
  cost. e1 held 88.5; the largest-yet Distill spike (2.43) aligned by
  e5. Folded (2.0e-6); probe: clean 88.86 post-fold, mean-8 3.09 (vs
  3.22), warm −7.95 (vs −8.37), jpeg-q40 −0.55, jpeg-q20 −2.26 (delta
  +0.07 vs rep_embed but absolute 86.60 > 86.40) — non-regression
  PASSED, absolutes improved on every condition. Credit stays shared
  between embed 5.0 and the spatial hint (combined by decision); the
  global-hint ablation is optional.
- **N/T shint rollout (ready)**:
  `osnet_{n,t}_8gb_distill_ain_jpeg_rep_embed5_shint.yml` — warm starts
  = tier rep_embed bests (N 88.930 / T 89.000, unfolded; rep branches
  and embed_proj reload, hint_proj new), hint_proj auto-sized per tier
  (N 640→768, T 768→768), builds verified. Judge: clean vs the tier
  best, probe non-regression after fold.
  **N result — GATE PASSED (+0.28): best 89.206** (e40), reproducing
  P's +0.26 almost exactly. Probe (folded): clean 89.21, jpeg-q20
  −2.20, warm −8.61, mean-8 3.30 — every condition at or better than
  N rep_embed (−2.24 / 3.51).
  **T result — REJECTED (flat)**: best 88.996 vs the 89.000 warm start
  (−0.004, noise). The run itself was healthy (same spike/dip/recovery
  shape as P/N); there was simply nothing left to gain. Consistent with
  T's pattern of smallest gains all campaign (jpeg +0.08, rep+embed
  +0.31, shint 0.00): the largest CNN reaches the
  representable-teacher ceiling first. T lineage stays rep_embed
  89.000 — README/ONNX already reflect it, no changes needed. Final
  OSNet ladder: P 88.86 / N 89.21 / T 89.00 (P and N now within 0.15
  and 0.21 of T at 46%/70% of its MACs).
- Queued next per the round plan: ViT-S-as-teacher TA arm (config
  only), dual-teacher partial-L_cam rel-KD (small code), stem-IN + rep
  + embed + L_cam integration arm (gated on the stem_attn verdict).

## SyntheticReID33 integration (d05, round 1)

The 6-domain unified set is live: d05 = SyntheticReID33 (400 train ids,
16,000/400/3,600, cameras c033–c065) → totals 191,560 / 5,144 / 33,542,
8,119 train ids, 66 cameras. Integration checks: layout/protocol
invariants all pass (40 imgs/id, 8 cams/id, 32 cross-camera positives
per query, images 128x256); targeted SHA check: 0 duplicate groups
involve d05 (the 2,503 intra-domain groups are known d01/d03 source
artifacts — the build's all-domain SHA gate was skipped for exactly
this reason and should later split intra- vs cross-domain).

Teacher round 1: `vit_base_8gb_ain_synth_cam_jpeg.yml` (B flagship
recipe + warm start from 93.50; classifier re-inits for 8,119 ids).
**Baselines on the NEW 6-domain val (old flagship zero-shot): overall
93.83** — d00 95.44 / d01 91.60 / d02 97.80 / d03 93.47 / d04 99.77 /
**d05 98.13 zero-shot** — the synthetic domain is EASY for the existing
representation (headroom ~1.9), so the experiment's real question is
whether synthetic diversity helps or dilutes the REAL domains. Gates:
overall > 93.83, d00–d04 non-regression, officials + probe as usual.

**Round 1 result (e40): d05 SOLVED (100.0 mAP/R1), real domains
neutral, no generalization uplift yet.** Per-domain vs zero-shot: d00
+0.38 / d01 −0.28 / d02 −0.22 / d03 d04 flat; legacy val 93.4 (best
still the e1 warm-start eval 93.488). Probe (legacy): clean 93.36
(−0.14), mean-8 1.33 (vs 1.26), jpeg-q20 −1.27 (improved), warm −4.07
(worse). Officials mixed: duke_occ +0.29, msmt17 −0.19, rest ±0.05.
Diagnosis: the 8,119-way head trained from scratch and ended immature
(Acc 0.96 still climbing, terminal loss ~1.65 = CE residue; the
classifier could not warm-start across the id-space change — the ViT
load_param DOES inherit same-shape classifiers, so extension rounds
keep the head). Round 2
(`vit_base_8gb_ain_synth_cam_jpeg2.yml`, warm start = round-1
checkpoint_last incl. classifier, fresh 40ep cosine) decides the
adoption question: judge legacy mAP vs 93.50 + probe/officials with a
mature head; d05 must hold 100.

**Round 2 — ADOPTED as the new B flagship.** Legacy best **93.594**
(e33, first to beat 93.50), 6-domain overall **94.09** (zero-shot
93.83, r1 93.87), d05 held 100.0, per-domain non-regression (d00
+0.42 / d03 +0.13, rest within ±0.16). Probe: clean 93.59, mean-8
1.29 (r1's 1.33 healed back to the 1.26 baseline band), jpeg-q20
−1.22 (improved). Officials: **msmt17 +0.91 (.9526)**, **duke_occ
+0.63 (.9584)**, market +0.23, cuhk03np +0.07, occ_reid −0.15 — a
genuine real-domain generalization gain, satisfying the adoption
criterion outright (both clean AND generalization improved). The
r1→r2 pair also empirically confirms the LP-FT mechanism (Kumar et
al., ICLR 2022): random-head fine-tuning left an OOD signature (clean
neutral, probe worse), and the mature-head round healed it.

**Two remedies compared.** The no-cam teacher took the
class-center-init path (`tools/init_classifier_centers.py`, NCM head
from one forward pass, row norms calibrated to the source classifier):
Acc started at 0.971 (vs 0.000 random / 0.960 after a full maturation
round) and one 40ep round reached legacy parity 92.3 with d05 = 100.0
— the d05 acquisition at zero real-domain cost, in half the compute of
the two-round protocol. Center-init is the standard for all future
id-space changes (README section added); the cam lineage's legacy
uplift (+0.09 and officials gains) is attributable to the synthetic
cameras enriching L_cam, a path the no-cam recipe lacks. Next: student
rounds on the 6-domain set (center-init + single round each), teachers:
cam2 (93.594) for S, no-cam-synth (92.3, e40) for P/N/T.

## Student rounds on the 6-domain set + the CNN ceiling law

Results (center-init + single round each): **S 93.149** (+0.12, cam2
teacher; officials msmt17 +0.68 / duke_occ +0.54 — the teacher's
generalization gain propagates), **P 89.101** (+0.25, probe improved
across the board: mean-8 3.09→2.98), **N flat** (best e1 89.228 ≈ the
89.206 warm start; e40 = 89.08 legacy / d05 99.83 / probe improved
mean-8 3.30→2.98), T pending. d05 solved on every finished tier
(99.8–100). A light-recipe N ablation (rep + embed 2.0, no hint —
T's recipe) tested whether excess teacher pressure causes the −0.13
legacy dip at the ceiling: **flat as well** (best e1 89.216, e40 89.1 =
the shint arm's 89.08) — pressure is NOT the cause; the dip is simply
the cost of learning d05 while at the ceiling, and the recipe does not
matter there (one more confirmation of the ceiling law). N adoption =
the shint e40 (gates already passed). T finished flat too (best e1
89.013; e40 88.91 legacy / d05 99.56 / probe improved warm −9.02→−8.70,
q20 −2.22→−2.15) — T adoption = e40 by the same trade.

**Finding — the heavy-KD recipe works exactly until the
teacher-representable ceiling, which is capacity-independent.** The
same embed5+shint recipe gave: N (old round, from 88.93) +0.28; P (old,
from 88.59) +0.26; P (new, from 88.86) +0.25; but T (old, from 89.00)
0.00 and N (new, from 89.21) flat. The boundary variable is not the
tier or the recipe but the remaining distance to a common ceiling: all
three CNN tiers converge to **89.0–89.2** — in INVERSE capacity order
(P 2.2M → 89.10, N 3.3M → 89.23, T 4.6M → 89.00), re-confirming
"capacity is not the axis". The ceiling is set by teacher quality minus
the ViT→CNN structural gap (~teacher 92.3 − ~3.2, the same gap the 0.50
rel-KD floor measures). Heavier KD (embed 5.0, spatial hint) increases
information transferred per step, which pays exactly while a tier still
has untransferred residual — larger students saturate the teacher in
earlier rounds, so the smallest tier (P) kept benefiting longest
(ladder totals: P +0.96, N +0.78, T +0.31 from their jpeg baselines).
Implication: further CNN gains require raising the ceiling itself —
i.e. a better CNN-representable teacher — not stronger transfer; the
only known path is the stem-IN + L_cam integration arm gated on the
stem_attn verdict (cam2's L_cam geometry stays unrepresentable, 0.77
floor).

## Camera-proxy mimicry branch (staged plan)

Motivation: the L_cam teacher is unrepresentable by the CNN trunk (0.77
floor; forcing it caps recovery), yet external evidence (PINTO0309/soma,
real-video tests of the no-cam OSNet) shows population-statistics
whitening collapses with ≤4 people and stabilizes with more — i.e. the
value of L_cam-style nuisance suppression is real, and a LEARNED,
per-sample version of it would not depend on scene statistics at all.
Idea (user proposal): a dedicated learnable branch mimics the camera
proxy behavior; the cam teacher's signal flows ONLY into that branch
(stop-gradient to the trunk — this removes the established harm
mechanism outright, unlike dual-teacher weighting which still pushes the
trunk); the deployed embedding fuses trunk + branch. Unlike the rejected
tail-attention branches, this branch has its OWN supervision, so gate
starvation cannot kill it — only the fusion has to earn its place.
DECISION: the soma small-N scenario is deliberately NOT added to the
evaluation gates; adoption is judged on the standard gates only.

- **Stage 0 — feasibility probe (`tools/probe_cam_branch.py`) — PASSED,
  with a surprise**: frozen no-cam P student (89.101 folded) + small
  MLP heads per tap, 64k train samples / val = unified query. Results
  (rel-KD ×30 vs cam2 / cos to delta): stem 4.45/0.914, conv2
  3.51/0.922, conv3 2.34/0.931, **conv4 0.36/0.961, embed 0.18/0.971**
  — the gate (< 0.6) is passed decisively at the LATE taps. (Caveat:
  probe numbers are clean-image rel-only; historic floors are
  train-time with augmentation + logit term — the margin absorbs it.)
  Two findings: (1) the early-tap hypothesis is REFUTED — L_cam-relevant
  information is richest AFTER the IN cascade, partially exonerating
  the INs for the 0.77 floor (relevant to the pending stem_attn
  verdict); (2) the floor must be re-read as an OBJECTIVE CONFLICT — a
  single shared embedding cannot be CE/triplet-optimal and
  L_cam-geometric at once — rather than missing information, which is
  precisely the conflict the gradient-isolated branch removes. Stage 1
  therefore taps the FINAL embedding: the branch collapses to a small
  residual MLP head (~0.7M params, ~0.7 MMACs — near-zero cost, far
  below the provisional +2–5% MACs budget).
- **Stage 1 — branch implementation (gated on Stage 0)**: light 1x1-conv
  branch (+2–5% MACs budget) from the best tap; distill target = the
  teacher DELTA (purer than the full cam embedding); trunk protected by
  stop-gradient; fusion = `final = trunk + gamma * branch` with a
  learnable gamma (safety valve; optionally exposed as a runtime input
  so deployments can modulate the built-in whitening — no scene
  statistics needed, hence no small-N failure mode).
- **Stage 2 — P training arm — CEILING BROKEN: best 89.833 (+0.73)**,
  monotone to e40. Both readouts positive: CamBr 16.4 → 0.200 (raw rel
  ≈ 0.006 = the Stage-0 probe optimum) and **gamma 0.01 → 0.513** — the
  first recruited branch of the campaign (vs 5 attention
  self-suppressions; the difference is dedicated supervision carrying
  information the trunk cannot pursue). Gates: 6-domain overall 90.62
  (ALL domains up, d01 +0.99), d05 99.95, probe mean-8 2.98 → **2.43**
  with warm −7.32 → **−5.28** — the L_cam robustness signature
  (warm/cool) transferred to a CNN for the first time. Whitening probe:
  raw sigma-margins lifted at every K (13.7→17.8 at K=2, 4.30→5.27 at
  K=16), external-whitening crossover pushed K≥3 → K≥8, whitening now
  clearly harmful at K=2 (17.8→12.1) — **the branch internalized the
  nuisance suppression; the soma small-N failure mode is answered
  structurally** (per-sample, no scene statistics). Trunk isolation
  held throughout (Distill 0.790, normal regime). Eval/export note:
  folded weights must be evaluated with MODEL.CAM_BRANCH on + plain
  arch (the fused forward is the deployment path); the ONNX contract
  needs the branch ops (Gemm+2, Erf+1) before the README/ONNX update.
- **Stage 3 — rollout + export**: N/T configs; ONNX contract must gain
  the branch's op classes (first non-zero-cost inference addition of
  the campaign — keep the budget explicit in the spec).
- Sequencing: composable with (not blocked by) the stem_attn verdict;
  if stem-IN makes the trunk itself L_cam-representable, the branch and
  the integration arm can be compared or combined.

## Measurement checklist per arm

1. unified test (train log best + `eval_official.py`-style final check)
2. `tools/eval_per_domain.py` (d01 for C1, d03/d04 for C2)
3. `tools/eval_official.py` (occluded splits for C2)
4. `tools/eval_style_shift.py` (robustness non-regression gate)
