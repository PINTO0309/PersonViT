# PersonViT
PersonViT: Large-scale Self-supervised Vision Transformer for Person Re-Identification

## Contributions
## Results
![PersonViT](pics/sota_pic.png)
![PersonViT-tb](pics/sota_table.png)
## Download
You can download pretrained PersonViT models from [ViT-S/16](https://huggingface.co/lakeAGI/PersonViT/tree/main/vits.lup.256x128.wopt.csk.4-8.ar.375.n8) and [ViT-B/16](https://huggingface.co/lakeAGI/PersonViT/tree/main/vitb.lup.256x128.wopt.csk.4-8.ar.375.n8)

You can download person ReID supervised-trained model and log from [reid_ft_model_logs](https://huggingface.co/lakeAGI/PersonViTReID)

### Differences between the released weights

The two model repositories correspond to different stages of the PersonViT training pipeline:

| Repository | Training stage | Intended use |
| --- | --- | --- |
| [PersonViT](https://huggingface.co/lakeAGI/PersonViT) | Self-supervised pre-training on unlabeled person images | Use `checkpoint*.pth` to initialize the ViT backbone before fine-tuning it on a downstream ReID dataset. These checkpoints are not dataset-specific supervised ReID models. |
| [PersonViTReID](https://huggingface.co/lakeAGI/PersonViTReID) | Supervised ReID fine-tuning initialized from the PersonViT checkpoints | Use `transformer_120.pth` to evaluate a model that has already been fine-tuned on the corresponding ReID dataset. The repository provides separate models and training logs for Market1501, MSMT17, DukeMTMC-reID, and Occluded-Duke. |

For example, `e0220` and `e0260` in the fine-tuned model directory names indicate the PersonViT pre-training checkpoint used for initialization, while `transformer_120.pth` is the model obtained after 120 epochs of supervised ReID fine-tuning.

ViT-S/16 and ViT-B/16 use the same 16 x 16 patch size but have different model capacities:

| Architecture | Transformer layers | Embedding dimension | Attention heads | Characteristics |
| --- | ---: | ---: | ---: | --- |
| ViT-S/16 | 12 | 384 | 6 | Smaller, faster, and less memory-intensive |
| ViT-B/16 | 12 | 768 | 12 | Larger capacity, but more computationally expensive |

In short, use a `PersonViT` checkpoint when training on a new ReID dataset, and use the matching `PersonViTReID` checkpoint when reproducing or evaluating the released supervised results. Always select the configuration that matches both the architecture (`small` or `base`) and the target dataset.

### Choosing weights for production

There is no single checkpoint that is optimal for every deployment because ReID accuracy is sensitive to differences in cameras, illumination, viewpoints, clothing, occlusion, detector quality, and image resolution. The recommended production model is therefore a PersonViT checkpoint fine-tuned and validated on data collected from the target environment.

If target-domain fine-tuning is not yet available, the following released models are reasonable starting points:

| Deployment requirement | Suggested starting point |
| --- | --- |
| Accuracy-oriented GPU deployment | `msmt.vitb.lup.256x128.wopt.csk.4-8.ar.375.n8.e0260/transformer_120.pth` |
| Latency- or memory-constrained deployment | `msmt.vits.lup.256x128.wopt.csk.4-8.ar.375.n8.e0220/transformer_120.pth` |
| Deployment dominated by partial occlusion | Also evaluate the corresponding `occ_duke` model |
| Supervised fine-tuning on a new deployment dataset | Initialize from `checkpoint0260.pth` for ViT-B/16 or `checkpoint0220.pth` for ViT-S/16, then fine-tune on the target identities |

The MSMT17 models are useful general baselines because MSMT17 contains more varied cameras, indoor and outdoor scenes, time slots, and lighting conditions than the other released fine-tuning datasets. This does **not** guarantee the best performance in a new domain. Benchmark scores from different datasets are not directly comparable; for example, a higher Market1501 mAP does not prove that the Market1501 model will generalize better to a production camera network.

Within the same dataset, ViT-B/16 provides higher released benchmark accuracy, while ViT-S/16 has a much smaller model and lower inference cost. On MSMT17, the released logs report 80.8% mAP / 92.0% Rank-1 for ViT-B/16 and 74.3% mAP / 88.8% Rank-1 for ViT-S/16. Actual latency and memory usage should be measured on the target hardware with the intended batch size and inference runtime.

The supervised classifier in `transformer_120.pth` predicts identities from its training benchmark and should not be used as a production identity classifier. In production, use the model as an embedding extractor: obtain the 384-dimensional ViT-S/16 or 768-dimensional ViT-B/16 feature, apply L2 normalization, compare embeddings with cosine similarity or Euclidean distance, and calibrate the match threshold using held-out target-domain data. Report false-accept and false-reject rates at the selected threshold in addition to retrieval metrics such as mAP and Rank-1.

### Comparison with OSNet

[OSNet](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_Omni-Scale_Feature_Learning_for_Person_Re-Identification_ICCV_2019_paper.html) is a useful lightweight CNN baseline for evaluating the accuracy/efficiency trade-off of PersonViT. A practical comparison should include OSNet x1.0, PersonViT-S/16, and PersonViT-B/16. For deployment without target-domain adaptation, [OSNet-AIN x1.0](https://arxiv.org/abs/1910.06827) is also relevant because it was designed for improved cross-domain generalization.

| Model | Parameters | GFLOPs at 256 x 128 | Embedding dimension | Typical use |
| --- | ---: | ---: | ---: | --- |
| OSNet x1.0 | 2.2M | 0.98 | 512 | Edge or real-time deployment |
| PersonViT-S/16 | 22.0M | 2.94 | 384 | Balanced accuracy and inference cost |
| PersonViT-B/16 | 86.5M | 11.35 | 768 | Accuracy-oriented GPU deployment |

The OSNet complexity values are from the [official Torchreid model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html). The PersonViT values are calculated from the models in this repository with a 256 x 128 input and a 16 x 16 stride. Runtime latency can differ from FLOPs and must be measured with the target hardware and inference backend.

The following published checkpoint results are shown as **Rank-1 / mAP** and are useful as an off-the-shelf reference:

| Model | Market1501 | DukeMTMC-reID | MSMT17 |
| --- | ---: | ---: | ---: |
| OSNet x1.0 | 94.2 / 82.6 | 87.0 / 70.2 | 74.9 / 43.8 |
| PersonViT-S/16 | 96.8 / 92.9 | 91.9 / 84.7 | 88.8 / 74.3 |
| PersonViT-B/16 | 97.6 / 95.0 | 93.8 / 88.1 | 92.0 / 80.8 |

OSNet results are from the [official model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html), and PersonViT results are from the [released fine-tuning logs](https://huggingface.co/lakeAGI/PersonViTReID). These numbers are **not a controlled architecture-only comparison**: the released models use different pre-training, loss functions, optimizers, data augmentation, and evaluation details. In particular, the OSNet model-zoo baseline uses softmax loss, while PersonViT uses softmax and triplet losses.

For a fair deployment comparison:

1. Use exactly the same person detections, crops, query/gallery split, and 256 x 128 input resolution.
2. Keep each model's expected pixel normalization; OSNet uses ImageNet normalization, while the released PersonViT configuration uses mean and standard deviation `[0.5, 0.5, 0.5]`.
3. L2-normalize both embeddings, use the same distance metric, and disable re-ranking.
4. Measure mAP, Rank-1, false-accept and false-reject rates, batch-1 p50/p95 latency, throughput, peak memory, and model size.
5. Use the same hardware, precision (FP32 or FP16), runtime, batch size, and warm-up procedure.
6. Report results separately for operational conditions such as occlusion, low resolution, nighttime, and individual cameras.

As a starting point, prefer OSNet x1.0 when latency and memory are the primary constraints, PersonViT-S/16 when a moderate compute budget is available, and PersonViT-B/16 when retrieval accuracy is the priority. The final choice should be based on validation data collected from the actual deployment environment.

## ReID Fine-tuning and Evaluating

This repository contains the supervised ReID fine-tuning code
([`transreid_pytorch`](transreid_pytorch)). The self-supervised pre-training
code is not included; fine-tuning starts from a released PersonViT checkpoint.

### 1. Install dependencies

The required packages are listed in
[`transreid_pytorch/setup.py`](transreid_pytorch/setup.py):
`numpy`, `torch`, `torchvision`, `h5py`, `opencv-python`, `yacs`, and `timm`.
The pinned versions (`torch==1.6.0`, `timm==0.3.2`) reflect the original
development environment.

### 2. Download a pre-training checkpoint

Download the self-supervised checkpoints from
[ViT-S/16](https://huggingface.co/lakeAGI/PersonViT/tree/main/vits.lup.256x128.wopt.csk.4-8.ar.375.n8)
or
[ViT-B/16](https://huggingface.co/lakeAGI/PersonViT/tree/main/vitb.lup.256x128.wopt.csk.4-8.ar.375.n8)
and save them under `pretrained/` in the repository root. The recommended
checkpoints are `checkpoint0220.pth` for ViT-S/16 and `checkpoint0260.pth` for
ViT-B/16.

### 3. Prepare the datasets

Place the ReID datasets under `transreid_pytorch/data/`. The directory names
below are fixed by the dataset loaders in
[`transreid_pytorch/datasets`](transreid_pytorch/datasets):

```
transreid_pytorch/data/
├── market1501/
│   ├── bounding_box_train/
│   ├── bounding_box_test/
│   └── query/
├── MSMT17/
│   ├── train/
│   ├── test/
│   ├── list_train.txt
│   ├── list_val.txt
│   ├── list_query.txt
│   └── list_gallery.txt
├── dukemtmcreid/
│   ├── bounding_box_train/
│   ├── bounding_box_test/
│   └── query/
└── Occluded_Duke/
    ├── bounding_box_train/
    ├── bounding_box_test/
    └── query/
```

### 4. Fine-tune on all four datasets

[`run_epochs.sh`](transreid_pytorch/run_epochs.sh) resolves the checkpoint file
from the epoch number and then fine-tunes on MSMT17, Market1501, DukeMTMC-reID,
and Occluded-Duke sequentially (120 epochs each) via
[`run_batch.sh`](transreid_pytorch/run_batch.sh):

```shell
cd transreid_pytorch
sh run_epochs.sh ../pretrained/vits.lup.256x128.wopt.csk.4-8.ar.375.n8/ vits.lup.256x128.wopt.csk.4-8.ar.375.n8 220 0 2 small
```

The positional arguments are:

| # | Argument | Meaning | Example |
| --- | --- | --- | --- |
| 1 | `pretrain_dir` | Directory containing the pre-training checkpoints | `../pretrained/vits.lup.256x128.wopt.csk.4-8.ar.375.n8/` |
| 2 | `output_fix` | Suffix used for the output directories under `logs/` | `vits.lup.256x128.wopt.csk.4-8.ar.375.n8` |
| 3 | `epoch` | Pre-training epoch; selects `checkpoint<epoch>.pth` (default `240`) | `220` for ViT-S, `260` for ViT-B |
| 4 | `device` | GPU ID (default `0`) | `0` |
| 5 | `hw_ratio` | Height/width ratio of the pre-training input (default `2`) | `2` for 256 x 128 |
| 6 | `arch` | Architecture: `small` or `base` (default `small`) | `small` |

Models and logs are written to `transreid_pytorch/logs/<dataset>.<output_fix>.e<epoch>/`.

### 5. Fine-tune on a single dataset (optional)

To train only one dataset, call [`train.py`](transreid_pytorch/train.py)
directly with the matching config from
[`configs/`](transreid_pytorch/configs) (`market`, `msmt17`, `dukemtmc`, or
`occ_duke`, each with `vit_small.yml` and `vit_base.yml`):

```shell
cd transreid_pytorch
python train.py --config_file configs/market/vit_small.yml \
DATASETS.ROOT_DIR ./data/ \
SOLVER.BASE_LR 4e-4 \
MODEL.DEVICE_ID "('0')" \
MODEL.PRETRAIN_PATH ../pretrained/vits.lup.256x128.wopt.csk.4-8.ar.375.n8/checkpoint0220.pth \
MODEL.PRETRAIN_HW_RATIO 2 \
OUTPUT_DIR logs/market.vits.e0220
```

The default solver settings (SGD, base LR `4e-4`, 120 epochs, batch size 64,
input 256 x 128) come from the config files and can be overridden on the
command line in the same `KEY VALUE` style.

### 6. Evaluate a fine-tuned model

Use [`test.py`](transreid_pytorch/test.py) with the same config and the
fine-tuned weight (either your own `logs/.../transformer_120.pth` or a
downloaded [PersonViTReID](https://huggingface.co/lakeAGI/PersonViTReID)
model):

```shell
cd transreid_pytorch
python test.py --config_file configs/market/vit_small.yml \
  DATASETS.ROOT_DIR ./data/ \
  MODEL.DEVICE_ID "('0')" \
  TEST.WEIGHT logs/market.vits.e0220/transformer_120.pth \
  OUTPUT_DIR logs/market.vits.e0220.eval
```

---
---
---

## Unified `reid` dataset

[`transreid_pytorch/tools/build_unified_dataset.py`](transreid_pytorch/tools/build_unified_dataset.py)
fully merges the five source ReID datasets placed under
[`transreid_pytorch/data/`](transreid_pytorch/data) into one training-oriented
dataset. Generated file names are neutral (`p<pid>_d<dom>_c<cam>_<seq>.jpg`)
and keep no trace of the source dataset names; sources are only referred to by
anonymous domain ids `d00`–`d04`:

```
transreid_pytorch/data/reid/
├── train/      175,560 images / 7,719 identities (contiguous classifier labels)
├── query/        4,744 images / 1,362 identities
└── gallery/     29,942 images (includes gallery-only distractors)
```

Split policy, following common ReID conventions:

- The split is identity-disjoint: 15% of the identities of each domain are
  held out for evaluation (`--test-ratio`), stratified per domain so every
  domain appears in both train and test.
- For each test identity, one image per camera with two or more images
  becomes a query and the rest go to the gallery, so every query has a
  cross-camera match. Single-camera identities become gallery distractors.

### Domain-balanced sampling

Two of the five domains are small and heavily occluded, so plain PK sampling
would produce many batches without them. The `domain_balanced_triplet`
sampler ([`datasets/sampler_domain.py`](transreid_pytorch/datasets/sampler_domain.py))
draws the P identities of each batch domain-by-domain with probability
proportional to `(identities per domain) ** DATALOADER.DOMAIN_ALPHA`
(default 0.5). This keeps every batch mixed across domains — the largest
domain stops dominating and the small occluded domains appear in nearly every
batch — without oversampling the small domains as hard as uniform sampling
would. `DOMAIN_ALPHA 1.0` reproduces plain proportional sampling and `0.0`
samples domains uniformly.

### Optional SyntheticReID33 domain (d05)

The gated generator in
[`tools/generate_synth_reid33.py`](transreid_pytorch/tools/generate_synth_reid33.py)
plans 500 fictional adult identities across 33 synthetic cameras. It produces
16,000 train, 400 occluded query, and 3,600 clean gallery images only after a
96-image quality pilot and a 96-image body-rotation pilot pass automatic and
manual QA. Every image request, including identity anchors and camera plates,
is locked to `quality=low`. Batch requests are also locked to
the supported `gpt-image-2` family alias, while the catalog snapshot
`gpt-image-2-2026-04-21` is recorded separately for provenance. There is no
automatic fallback to another model family, and the API key is read only from
`OPENAI_API_KEY`.

Install the optional dependencies and create an offline request preview first:

```shell
uv sync --extra synth
cd transreid_pytorch
python tools/generate_synth_reid33.py pilot --dry-run
```

The dry run writes deterministic identities, camera profiles, all 20,000
sample specifications, and dependency-annotated Batch JSONL without making an
API request. Each of the 33 cameras has a seed-locked mounting height, downward
pitch, focal length, and derived horizon/eye-level vanishing-line position.
The three view families use distinct ranges: high-wide cameras look down by
about 20–28 degrees, diagonal-medium cameras by about 7–12 degrees, and
telephoto-exit cameras by about 1.5–4.5 degrees. Exact values are written to
`state/cameras.jsonl`, generation prompts, plate metadata, and every sample
manifest row. A live pilot is advanced by rerunning the resume command: first
front anchors and 33 empty camera plates, then three edited identity views,
then the 96 Low pilot candidates.

An output root initialized with an older prompt/config is intentionally not
migrated because its existing plates may not encode this geometry, its
reference assets may have been generated above Low quality, or its samples may
have used the previous full-resolution reference profile. Keep that root as an
archive and pass a new `--root` for the Low-only native-quarter pilot.

```shell
export SYNTH_CAMERA_PITCH_ROOT="$PWD/data/SyntheticReID33_pilot_low_camera_pitch_v4"
python tools/generate_synth_reid33.py \
  --root "$SYNTH_CAMERA_PITCH_ROOT" pilot --dry-run
```

```shell
export OPENAI_API_KEY=...       # never written to state or manifests
python tools/generate_synth_reid33.py pilot --resume
# Rerun after each asynchronous Batch stage completes.
python tools/generate_synth_reid33.py pilot --resume
python tools/generate_synth_reid33.py repair-pilot --dry-run
python tools/generate_synth_reid33.py repair-pilot
python tools/generate_synth_reid33.py qa
python tools/generate_synth_reid33.py report
```

`repair-pilot` reprocesses every successful raw candidate with the official
TorchVision SSDLite person detector and Keypoint R-CNN pose model, then chooses
one accepted candidate for each of the 96 Low pilot specifications. This avoids
paying for retries caused by OpenCV HOG false positives. The first invocation
downloads and caches the official model weights. The command refuses to change
the pilot manifest while a Batch is active or completed but not yet collected;
run `pilot --resume` once more in that case. `--dry-run` verifies complete
recoverability without changing final images, manifests, jobs, or attempts.
Person NMS suppresses both ordinary IoU overlaps and lower-score boxes with at
least 85% containment, preventing a single pedestrian's nested upper-body box
from being counted as a second person.
For a clean image with one ankle weakened by a natural walking overlap, the
feet gate accepts the image only when the other ankle remains strongly visible,
the weak ankle logit is at least -2.0, both predicted ankle locations are in
the lower part of the person box and inside the image, and the person box is
clear of the image's bottom edge. Head, shoulder, and occluded-waist thresholds
remain unchanged.
All pilot and full samples share the same framing policy: crop the detected
person bounding box with 5% margin, require at least 85% bbox fill on both
axes, and directly resize that crop to 128x256 like the real ReID inputs.
Full samples also use eight camera-relative body-yaw bins at 45-degree
intervals. Every identity has exactly five images in each yaw bin. The four
occluded queries of each test identity are separated by 90 degrees, while the
existing front/left/right/back anchors are reused to avoid extra reference
generation cost.

Before `qa`, measure the real-data cross-camera positive 5-percentile
calibration for both ONNX models. The command uses every same-PID,
different-camera pair from the unified `query` and `gallery` protocol while
excluding gallery-only distractor PIDs:

```shell
python tools/calibrate_synth_reid33_similarity.py
```

It writes `data/SyntheticReID33/qa/real_similarity_reference.json` with the
measured thresholds, per-domain distributions, model hashes, dataset index
hash, preprocessing contract, and pair counts. Its minimum interface is:

```json
{
  "vit": {"cross_camera_positive_p05": 0.42},
  "osnet": {"cross_camera_positive_p05": 0.37}
}
```

Inspect both `qa/pilot_contact_sheet.jpg` and the uncropped 33-camera geometry
sheet `qa/camera_pitch_contact_sheet.jpg`, copy
`qa/manual_review.template.json` to `qa/manual_review.json`, record the manual
review including `camera_pitch_consistency_rate >= 0.95` and zero
`camera_geometry_failures`, and rerun `report`. The image model is guided by
explicit camera pose but cannot guarantee an exact photogrammetric angle, so
this plate review is a fail-closed manual gate. Full generation remains locked
unless every gate passes. Before approval, run the Low-only body-rotation pilot,
which reuses the existing 81 anchor/plate assets and has a hard USD 2.00 scope
ceiling:

```shell
python tools/generate_synth_reid33.py rotation-pilot --dry-run
python tools/generate_synth_reid33.py rotation-pilot --resume
# Rerun --resume after Batch completion to collect the result; it will not retry.
python tools/generate_synth_reid33.py rotation-pilot --resume
python tools/generate_synth_reid33.py qa --scope rotation-pilot
```

Inspect `qa/rotation_pilot_contact_sheet.jpg`, copy
`rotation_pilot/manual_review.template.json` to
`rotation_pilot/manual_review.json`, record the labeled-yaw and identity review,
then rerun `report`. Approval requires at least 90% yaw accuracy, 95% identity
consistency, all embedding/geometry gates, measured Low usage, a forecast
within the USD ceiling, and unchanged report/config hashes:

```shell
python tools/generate_synth_reid33.py approve --quality low --max-usd 500
python tools/generate_synth_reid33.py full --dry-run
python tools/generate_synth_reid33.py full --resume
# After all 20,000 geometry-valid samples are collected:
python tools/generate_synth_reid33.py qa --scope full
```

Normal approval remains fail-closed. If the user explicitly accepts the risk,
only `pilot.osnet_embedding`, `rotation_pilot.osnet_embedding`, and
`rotation_pilot.manual_review` may be waived with an audit reason. Image count,
decode, geometry, framing, duplicate, ViT, model, configuration, and cost gates
cannot be waived:

```shell
python tools/generate_synth_reid33.py approve \
  --quality low --max-usd 120 \
  --waive-failed-gates \
  --waiver-reason "User explicitly authorized production despite the recorded OSNet and rotation manual-review gate failures."
```

The exact failed gate names, reason, timestamp, report hash, configuration hash,
quality, and hard USD ceiling are preserved in `state/approval.json`.

Each resume call polls existing Batch IDs and submits only newly unblocked
first attempts. Results are matched by `custom_id`; automatic API retries,
prompt-revision resubmissions, and retry cost reserve are all disabled. A
successful API raw response is retained even when local geometry QA fails.
Derived images can be audited and rebuilt from those raw files without an API
request:

```shell
python tools/generate_synth_reid33.py repair-local \
  --scope rotation-pilot --dry-run
python tools/generate_synth_reid33.py repair-local \
  --scope rotation-pilot
python tools/generate_synth_reid33.py report-failures \
  --scope rotation-pilot
```

The same commands accept `--scope pilot` and `--scope full`. Local repair can
restore missing/corrupt final JPEGs, deterministic camera processing, QA, and
manifest rows. It cannot recover an API request that returned no image or fix
wrong identity, anatomy, body yaw, or camera perspective in the raw image. Such
samples remain unresolved and block release rather than triggering paid
generation.

Production is locked to the empirically selected `native_quarter` reference
profile. Anchor and camera-plate generation outputs are retained at 1024x1536
and 576x1152 respectively, then deterministically downsampled with Lanczos to
256x384 and 144x288 JPEGs before their `vision` upload. Only these smaller
copies are used as Image Edit inputs; the generated originals remain available
under `assets/` for audit and local reprocessing. The selected profile reduced
image-input tokens by 75% in the reference-size probe.

The isolated probe below reproduces that selection. It creates seven proxy
pairs from the same existing PID 0 anchor and camera 0 plate, then sends exactly
one synchronous Low Image Edit per pair at a fixed 576x1152 output. It performs
no automatic retry and cannot add its outputs to the training dataset:

```shell
export SYNTH_REF_PROBE_ROOT="$PWD/data/SyntheticReID33_reference_size_probe"
python tools/probe_synth_reid33_reference_sizes.py \
  --root "$SYNTH_REF_PROBE_ROOT" prepare
export OPENAI_API_KEY=...       # required only for run
python tools/probe_synth_reid33_reference_sizes.py \
  --root "$SYNTH_REF_PROBE_ROOT" run --max-usd 0.20
python tools/probe_synth_reid33_reference_sizes.py \
  --root "$SYNTH_REF_PROBE_ROOT" report
```

`report.json` records acceptance, input/output token usage, actual Standard API
cost, Batch-equivalent cost, 20,000-request projection, detector/pose QA, and
ViT/OSNet similarity for every resolution. Review `contact_sheet.jpg` before
auditing the production reference size; one stochastic output per condition is
a cost probe, not a statistically strong quality evaluation.

Full QA runs both ONNX models over all samples, records d05-only Rank-1/mAP as
a label-quality diagnostic, rejects identity/margin/pHash failures without
automatically resubmitting them, and creates five contact sheets containing a PID/occlusion
stratified 5% sample. Copy `qa/full_review.template.json` to
`qa/full_review.json` after reviewing those sheets and every automatic boundary
case. The standalone validator will not release the dataset without both the
20,000-image automatic report and this manual review gate.

After exactly 20,000 accepted images are present, validate and integrate them:

```shell
python tools/validate_synth_reid33.py data/SyntheticReID33
python tools/build_unified_dataset.py --data-root data --force \
  --synthetic-root data/SyntheticReID33
python tools/validate_synth_reid33.py data/SyntheticReID33 \
  --unified-root data/reid
```

The builder preserves d05's explicit split, maps its local cameras to global
IDs 33–65, validates a temporary tree, then atomically switches it into place.
The locked repository baseline becomes 191,560 train, 5,144 query, and 33,542
gallery images across six domains and 66 cameras. Existing loader tuple and
file-name formats are unchanged.
These counts assume camera SIE is disabled, as in the current training setup;
loading a 33-camera SIE checkpoint into the 66-camera model requires a separate
embedding-row migration and is intentionally outside this workflow.

For the final training adoption gate, export each baseline/candidate run as
JSON with `seed`, a shared `recipe_sha256`, `real_domain_mAP` (`d00`–`d04`),
and `occluded_mAP` (`Occluded-Duke`, `Occluded-REID`), then run:

```shell
python tools/evaluate_synth_reid33_adoption.py \
  --baseline results/base-{1,2,3}.json \
  --candidate results/d05-{1,2,3}.json
```

It only returns `adopt` after three matched seeds, no more than a 0.5-point
mean real-domain mAP loss, and at least a 0.5-point mean occluded-set gain.

One allocation constraint is intentionally documented in the config: exact
605/610 images on every camera is mathematically incompatible with always
visiting exactly two of the three cameras at each site. The minimum-deviation
solution used here is 600 images on two cameras, 605 on 22, and 610 on nine;
every identity still has exactly 40 images, eight cameras, and four sites.

### Environment (uv)

The pipeline was updated to run on current PyTorch (torch.amp API,
`torch.load` compatibility, `addmm_` signature). Create the pinned
environment from [`pyproject.toml`](pyproject.toml) with [uv](https://docs.astral.sh/uv/):

```shell
uv sync
source .venv/bin/activate
```

### Training on the unified dataset

Single-GPU configurations are provided per VRAM budget under
[`configs/reid/`](transreid_pytorch/configs/reid). All use AMP, SGD with
cosine schedule, 60 epochs, and K = 4 instances per identity; learning rates
follow linear scaling from the canonical 4e-4 at batch 64:

| Config | Backbone | Batch (P x K) | Base LR | Warmup | Peak GPU memory |
| --- | --- | ---: | ---: | ---: | --- |
| `vit_small_8gb.yml` | ViT-S/16 | 64 (16 x 4) | 4e-4 | 10 | 2.1 GiB (measured) |
| `vit_base_8gb.yml` | ViT-B/16 | 32 (8 x 4) | 2e-4 | 10 | 2.4 GiB (measured) |
| `vit_small_16gb.yml` | ViT-S/16 | 128 (32 x 4) | 8e-4 | 10 | ~3.9 GiB (estimated) |
| `vit_base_16gb.yml` | ViT-B/16 | 64 (16 x 4) | 4e-4 | 10 | ~4.3 GiB (estimated) |
| `vit_small_96gb.yml` | ViT-S/16 | 256 (64 x 4) | 1.6e-3 | 15 | ~7.6 GiB (estimated) |
| `vit_base_96gb.yml` | ViT-B/16 | 256 (64 x 4) | 1.6e-3 | 20 | ~17 GiB (estimated) |

Memory numbers are `torch.cuda.max_memory_allocated()` at 256 x 128 input;
real reserved memory is somewhat higher. Batches beyond 256 are not
configured on purpose: very large PK batches tend to hurt triplet mining
quality, so on a 96 GB GPU the remaining headroom is better spent on higher
input resolution or a larger backbone than on a larger batch.

```shell
cd transreid_pytorch
# sh run_reid.sh <arch: small|base> <vram: 8gb|16gb|96gb> [device] [pretrain]

# ViT-S/16 on an 8 GB GPU
sh run_reid.sh small 8gb 0

# ViT-B/16 on an 8 GB GPU (recommended command)
sh run_reid.sh base 8gb 0
```

The `base 8gb` command resolves to the following explicit invocation:

```shell
python train.py --config_file configs/reid/vit_base_8gb.yml \
MODEL.DEVICE_ID "('0')" \
OUTPUT_DIR logs/reid_vit_base_8gb
```

`vit_base_8gb.yml` keeps ViT-B/16 within a measured 2.4 GiB peak allocation
by using batch 32 (8 identities x 4 instances) with the learning rate scaled
down to 2e-4 accordingly; expect noticeably longer wall-clock time per epoch
than ViT-S/16, since the smaller batch doubles the optimizer steps per epoch
on top of the heavier backbone.

`run_reid.sh` expects the self-supervised checkpoints at
`../pretrained/checkpoint0220.pth` (ViT-S) and `../pretrained/checkpoint0260.pth`
(ViT-B); pass a fourth argument to use another checkpoint.

### Checkpointing and resume

The `reid` configs enable `SOLVER.SAVE_BEST`. Validation runs every
`SOLVER.EVAL_PERIOD` (1) epoch, and whenever the validation mAP improves,
the model is saved as

```
logs/<dir>/transformer_best_e<epoch:06d>_map<mAP:.5f>.pth   # e.g. transformer_best_e000040_map0.75324.pth
```

Only the latest best file is kept (the previous best is deleted), and the
periodic fixed-epoch snapshots (`transformer_<epoch>.pth`) are disabled while
`SAVE_BEST` is on. Use the best file with `test.py` and `TEST.WEIGHT` to
evaluate a finished run.

Independently of best saving, the trainer atomically overwrites
`logs/<dir>/checkpoint_last.pth` at the end of every epoch with a full
training state: model, optimizers, LR scheduler, AMP scaler, epoch counter,
best-model bookkeeping, and all RNG states (Python / NumPy / Torch / CUDA).
Training interrupted for any reason can therefore be resumed exactly, with
the identical LR schedule and data order, from the next epoch:

```shell
python train.py --config_file configs/reid/vit_base_8gb.yml \
  SOLVER.RESUME logs/reid_vit_base_8gb/checkpoint_last.pth \
  MODEL.DEVICE_ID "('0')" \
  OUTPUT_DIR logs/reid_vit_base_8gb
```

A quick pipeline check (dataset statistics, per-batch domain mixture, and a
few real AMP training steps) is available with:

```shell
python tools/smoke_reid.py --config configs/reid/vit_small_8gb.yml
```

### Classifier initialization when the id space changes

`load_param` copies every shape-matching key from `MODEL.PRETRAIN_PATH` —
including the classifier. Whether the head warm-starts therefore depends
only on whether the train-id count changed:

| Warm-start pattern | Classifier | Action needed |
| --- | --- | --- |
| Same dataset build (fine-tune round, recipe ablation, teacher swap) | inherited | none |
| Train-id count changed (new domain integrated, dataset added/removed, split rebuilt with different ids) | **shape mismatch → silently re-initialized** | **run `tools/init_classifier_centers.py` first** |

A randomly re-initialized head is not a cosmetic problem: its early CE
gradients are noise with respect to the true classes and flow into the
warm backbone, so a standard 40-epoch schedule spends most of its budget
maturing the head instead of refining features (observed on the d05
teacher round 1: `Acc` started at 0.000 and the run never beat its own
warm start). The tool removes that phase with one deterministic forward
pass: it computes each class's mean BNNeck feature under the warm
backbone and writes a checkpoint whose classifier starts as a
nearest-class-mean head, norm-calibrated to the source checkpoint
(measured: `Acc` 0.971 from the first training iterations, versus 0.000
without it — equivalent to skipping an entire maturation round):

```shell
python tools/init_classifier_centers.py \
    --config configs/reid/<new-run>.yml \
    --weight "logs/<warm-run>/transformer_best_*.pth" \
    --output logs/<warm-run>/centers_init_<tag>.pth
# then point the new config's MODEL.PRETRAIN_PATH at the --output file
```

How to tell which case you are in: the trainer now prints
`load_param skipped N keys ... ['classifier.weight']` when the head could
not be inherited, and the first `Acc` values of epoch 1 stay at `0.000`.
Either signal means the run should have gone through the tool.

### Knowledge distillation (teacher -> student)

`configs/reid/vit_small_<vram>_distill.yml` trains a ViT-S/16 student under a
frozen fine-tuned ViT-B/16 teacher on the unified dataset:

```shell
cd transreid_pytorch
python train.py --config_file configs/reid/vit_small_8gb_distill.yml
```

The teacher is declared purely in the config: `DISTILL.TEACHER_CONFIG` selects
the teacher architecture (any config file) and `DISTILL.TEACHER_WEIGHT` its
trained checkpoint (glob patterns such as
`logs/reid_vit_base_8gb/transformer_best_*.pth` resolve to the single kept
best file). The teacher runs gradient-free in eval mode — BatchNorm statistics
are never touched — and adds about 0.5 GiB plus one inference pass per step
(measured total: 2.54 GiB for the 8 GB config).

Two backbone-agnostic losses are added to the usual softmax + triplet
objective, so student and teacher embedding dimensions never need to match:

| Loss | Config key | Default | Meaning |
| --- | --- | ---: | --- |
| Logit KD | `DISTILL.LOGIT_WEIGHT` | 1.0 | Temperature-scaled KL on the shared 7,719-way identity logits (`DISTILL.TEMPERATURE`, default 4.0) |
| Relational KD | `DISTILL.REL_WEIGHT` | 30.0 | MSE between the batch cosine-similarity matrices — distills the metric structure retrieval uses |
| Embedding KD | `DISTILL.EMBED_WEIGHT` | 0.0 | Direct cosine loss; enable only when both embedding dims match |

Because the distillation losses hold no trainable parameters, best-model
saving, `checkpoint_last.pth` and `SOLVER.RESUME` work unchanged; on resume
the frozen teacher is simply rebuilt from its config.

To distill into a different (e.g. lighter) student later, register the
backbone in `model/make_model.py` (`__factory_T_type`), point
`MODEL.TRANSFORMER_TYPE` at it, and keep the same `DISTILL` block — the
dimension-agnostic losses require no further changes. Any trained model can
act as the teacher the same way, including a distilled ViT-S teaching an even
smaller student.

### Results on the unified test set

Best checkpoints per variant, evaluated on the unified test split. `-aug` indicates that photometric augmentation fine-tuning was performed.

| Var | Backbone | Params | GFLOPs<br>@256x128 | Emb | mAP | Rank-1 | Rank-5 | Rank-10 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| B | ViT-B/16 | 86.5M | 11.35 | 768 | 93.3 | 97.1 | 98.2 | 98.4 |
| S | ViT-S/16 | 22.0M | 2.94 | 384 | 92.2 | 96.9 | 98.1 | 98.6 |
| B-ain | ViT-B/16<br>+<br>token-IN | 86.5M | 11.35 | 768 | 92.1 | 96.5 | 97.8 | 98.2 |
| S-ain | ViT-S/16<br>+<br>token-IN | 22.0M | 2.94 | 384 | 91.4 | 96.3 | 97.9 | 98.4 |
| T-ain | OSNet-AIN x1.5 | 4.6M | 2.12 | 512 | 88.0 | 94.8 | 97.4 | 98.1 |
| N-ain | OSNet-AIN x1.25 | 3.3M | 1.49 | 512 | 87.9 | 94.9 | 97.3 | 98.0 |
| P-ain | OSNet-AIN x1.0 | 2.2M | 0.98 | 512 | 87.0 | 94.1 | 97.2 | 97.9 |
| B-ain-aug | ViT-B/16<br>+<br>token-IN | 86.5M | 11.35 | 768 | 93.6 | 96.8 | 98.1 | 98.5 |
| S-ain-aug | ViT-S/16<br>+<br>token-IN | 22.0M | 2.94 | 384 | 93.1 | 97.2 | 98.2 | 98.4 |
| T-ain-aug | OSNet-AIN x1.5 | 4.6M | 2.12 | 512 | 88.9 | 95.0 | 97.4 | 98.2 |
| N-ain-aug | OSNet-AIN x1.25<br>+<br>cam-branch | 3.8M | 1.49 | 512 | 90.0 | 95.7 | 97.7 | 98.3 |
| P-ain-aug | OSNet-AIN x1.0<br>+<br>cam-branch | 2.7M | 0.98 | 512 | 89.8 | 95.8 | 97.8 | 98.2 |

- The retired ViT student candidates and their measured results are recorded in [`docs/lightweight_students.md`](docs/lightweight_students.md).
- B-ain is the teacher of the domain-generalization (`-ain`) ladder   ([`docs/ain_variants.md`](docs/ain_variants.md)): token-axis instance normalization after the patch embedding, trained with the B recipe over 75 epochs (the token-IN insertion costs a few adaptation epochs and, at convergence, 1.2 mAP of in-distribution accuracy versus B — the accepted price of style invariance).

#### B-ain-aug -  ViT-B/16 + token-IN - 86.5M

- unified test set eval

  | Var | Backbone | Params | GFLOPs<br>@256x128 | Emb | mAP | Rank-1 | Rank-5 | Rank-10 |
  | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
  | B-ain-aug | ViT-B/16<br>+<br>token-IN | 86.5M | 11.35 | 768 | 93.6 | 96.8 | 98.1 | 98.5 |

- official dataset eval

  | dataset | queries | gallery | mAP | R1 | R5 | R10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | market | 3,368 | 15,913 | 0.9905 | 0.9941 | 0.9982 | 0.9991 |
  | msmt17 | 11,659 | 82,161 | 0.9526 | 0.9738 | 0.9885 | 0.9901 |
  | duke_occ | 2,210 | 17,661 | 0.9584 | 0.9697 | 0.9828 | 0.9860 |
  | cuhk03np | 1,400 | 5,332 | 0.9891 | 0.9893 | 0.9950 | 0.9986 |
  | occ_reid | 1,000 | 1,000 | 0.9961 | 0.9970 | 0.9980 | 1.0000 |

- official dataset style-shift eval - query only shifted

  | condition | mAP | R1 | dmAP | dR1 |
  | --- | ---: | ---: | ---: | ---: |
  | clean | 0.9646 | 0.9791 | — | — |
  | bright+30% | 0.9640 | 0.9786 | -0.0006 | -0.0006 |
  | dark-30% | 0.9646 | 0.9792 | +0.0000 | +0.0001 |
  | contrast-40% | 0.9646 | 0.9791 | +0.0000 | +0.0000 |
  | contrast+40% | 0.9253 | 0.9461 | -0.0393 | -0.0330 |
  | warm | 0.9351 | 0.9546 | -0.0295 | -0.0245 |
  | cool | 0.9515 | 0.9698 | -0.0131 | -0.0094 |
  | gamma0.6 | 0.9585 | 0.9745 | -0.0061 | -0.0046 |
  | gamma1.6 | 0.9561 | 0.9743 | -0.0085 | -0.0048 |
  | jpeg-q40 | 0.9621 | 0.9773 | -0.0025 | -0.0018 |
  | jpeg-q20 | 0.9549 | 0.9731 | -0.0097 | -0.0061 |

#### S-ain-aug -  ViT-S/16 + token-IN - 22.0M

- unified test set eval

  | Var | Backbone | Params | GFLOPs<br>@256x128 | Emb | mAP | Rank-1 | Rank-5 | Rank-10 |
  | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
  | S-ain-aug | ViT-S/16<br>+<br>token-IN | 22.0M | 2.94 | 384 | 93.1 | 97.2 | 98.2 | 98.4 |

- official dataset eval

  | dataset | queries | gallery | mAP | R1 | R5 | R10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | market | 3,368 | 15,913 | 0.9872 | 0.9911 | 0.9976 | 0.9991 |
  | msmt17 | 11,659 | 82,161 | 0.9397 | 0.9697 | 0.9860 | 0.9882 |
  | duke_occ | 2,210 | 17,661 | 0.9527 | 0.9674 | 0.9819 | 0.9855 |
  | cuhk03np | 1,400 | 5,332 | 0.9875 | 0.9900 | 0.9950 | 0.9986 |
  | occ_reid | 1,000 | 1,000 | 0.9955 | 0.9960 | 0.9980 | 0.9990 |

- official dataset style-shift eval - query only shifted

  | condition | mAP | R1 | dmAP | dR1 |
  | --- | ---: | ---: | ---: | ---: |
  | clean | 0.9555 | 0.9759 | — | — |
  | bright+30% | 0.9547 | 0.9753 | -0.0008 | -0.0006 |
  | dark-30% | 0.9555 | 0.9760 | +0.0000 | +0.0001 |
  | contrast-40% | 0.9555 | 0.9760 | +0.0000 | +0.0001 |
  | contrast+40% | 0.9090 | 0.9400 | -0.0465 | -0.0360 |
  | warm | 0.9192 | 0.9477 | -0.0363 | -0.0282 |
  | cool | 0.9411 | 0.9664 | -0.0145 | -0.0095 |
  | gamma0.6 | 0.9481 | 0.9717 | -0.0074 | -0.0042 |
  | gamma1.6 | 0.9448 | 0.9705 | -0.0107 | -0.0054 |
  | jpeg-q40 | 0.9523 | 0.9748 | -0.0032 | -0.0011 |
  | jpeg-q20 | 0.9430 | 0.9693 | -0.0125 | -0.0066 |

#### T-ain-aug - OSNet-AIN x1.5 - 4.6M

- unified test set eval

  | Var | Backbone | Params | GFLOPs<br>@256x128 | Emb | mAP | Rank-1 | Rank-5 | Rank-10 |
  | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
  | T-ain-aug | OSNet-AIN x1.5 | 4.6M | 2.12 | 512 | 88.9 | 95.0 | 97.4 | 98.2 |

- official dataset eval

  | dataset | queries | gallery | mAP | R1 | R5 | R10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | market | 3,368 | 15,913 | 0.9722 | 0.9875 | 0.9958 | 0.9979 |
  | msmt17 | 11,659 | 82,161 | 0.8731 | 0.9476 | 0.9768 | 0.9824 |
  | duke_occ | 2,210 | 17,661 | 0.9115 | 0.9421 | 0.9751 | 0.9828 |
  | cuhk03np | 1,400 | 5,332 | 0.9811 | 0.9836 | 0.9936 | 0.9979 |
  | occ_reid | 1,000 | 1,000 | 0.9842 | 0.9870 | 0.9910 | 0.9970 |

- official dataset style-shift eval - query only shifted

  | condition | mAP | R1 | dmAP | dR1 |
  | --- | ---: | ---: | ---: | ---: |
  | clean | 0.9078 | 0.9584 | — | — |
  | bright+30% | 0.9041 | 0.9560 | -0.0037 | -0.0024 |
  | dark-30% | 0.9066 | 0.9578 | -0.0012 | -0.0006 |
  | contrast-40% | 0.9078 | 0.9582 | -0.0000 | -0.0002 |
  | contrast+40% | 0.8149 | 0.8751 | -0.0929 | -0.0833 |
  | warm | 0.8044 | 0.8712 | -0.1034 | -0.0872 |
  | cool | 0.8247 | 0.8903 | -0.0831 | -0.0681 |
  | gamma0.6 | 0.8833 | 0.9415 | -0.0245 | -0.0169 |
  | gamma1.6 | 0.8722 | 0.9349 | -0.0356 | -0.0235 |
  | jpeg-q40 | 0.9005 | 0.9533 | -0.0073 | -0.0051 |
  | jpeg-q20 | 0.8801 | 0.9376 | -0.0277 | -0.0208 |

#### N-ain-aug - OSNet-AIN x1.25 - 3.3M

- unified test set eval

  | Var | Backbone | Params | GFLOPs<br>@256x128 | Emb | mAP | Rank-1 | Rank-5 | Rank-10 |
  | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
  | N-ain-aug | OSNet-AIN x1.25<br>+<br>cam-branch | 3.8M | 1.49 | 512 | 90.0 | 95.7 | 97.7 | 98.3 |

- official dataset eval

  | dataset | queries | gallery | mAP | R1 | R5 | R10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | market | 3,368 | 15,913 | 0.9731 | 0.9852 | 0.9947 | 0.9982 |
  | msmt17 | 11,659 | 82,161 | 0.8831 | 0.9512 | 0.9781 | 0.9828 |
  | duke_occ | 2,210 | 17,661 | 0.9082 | 0.9425 | 0.9760 | 0.9801 |
  | cuhk03np | 1,400 | 5,332 | 0.9799 | 0.9807 | 0.9936 | 0.9971 |
  | occ_reid | 1,000 | 1,000 | 0.9878 | 0.9910 | 0.9940 | 0.9990 |

- official dataset style-shift eval - query only shifted

  | condition | mAP | R1 | dmAP | dR1 |
  | --- | ---: | ---: | ---: | ---: |
  | clean | 0.9136 | 0.9602 | — | — |
  | bright+30% | 0.9106 | 0.9570 | -0.0030 | -0.0032 |
  | dark-30% | 0.9126 | 0.9592 | -0.0010 | -0.0010 |
  | contrast-40% | 0.9136 | 0.9600 | -0.0000 | -0.0002 |
  | contrast+40% | 0.8302 | 0.8880 | -0.0835 | -0.0722 |
  | warm | 0.8394 | 0.9005 | -0.0742 | -0.0596 |
  | cool | 0.8617 | 0.9214 | -0.0520 | -0.0388 |
  | gamma0.6 | 0.8991 | 0.9507 | -0.0146 | -0.0095 |
  | gamma1.6 | 0.8827 | 0.9399 | -0.0310 | -0.0203 |
  | jpeg-q40 | 0.9072 | 0.9554 | -0.0064 | -0.0048 |
  | jpeg-q20 | 0.8877 | 0.9400 | -0.0259 | -0.0202 |

#### P-ain-aug - OSNet-AIN x1.0 - 2.2M

- unified test set eval

  | Var | Backbone | Params | GFLOPs<br>@256x128 | Emb | mAP | Rank-1 | Rank-5 | Rank-10 |
  | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
  | P-ain-aug | OSNet-AIN x1.0<br>+<br>cam-branch | 2.7M | 0.98 | 512 | 89.8 | 95.8 | 97.8 | 98.2 |

- official dataset eval

  | dataset | queries | gallery | mAP | R1 | R5 | R10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | market | 3,368 | 15,913 | 0.9733 | 0.9881 | 0.9952 | 0.9973 |
  | msmt17 | 11,659 | 82,161 | 0.8832 | 0.9511 | 0.9784 | 0.9827 |
  | duke_occ | 2,210 | 17,661 | 0.9031 | 0.9321 | 0.9724 | 0.9801 |
  | cuhk03np | 1,400 | 5,332 | 0.9804 | 0.9850 | 0.9929 | 0.9979 |
  | occ_reid | 1,000 | 1,000 | 0.9889 | 0.9890 | 0.9940 | 0.9980 |

- official dataset style-shift eval - query only shifted

  | condition | mAP | R1 | dmAP | dR1 |
  | --- | ---: | ---: | ---: | ---: |
  | clean | 0.9132 | 0.9597 | — | — |
  | bright+30% | 0.9102 | 0.9576 | -0.0030 | -0.0021 |
  | dark-30% | 0.9124 | 0.9588 | -0.0007 | -0.0009 |
  | contrast-40% | 0.9132 | 0.9596 | +0.0000 | -0.0001 |
  | contrast+40% | 0.8325 | 0.8887 | -0.0807 | -0.0709 |
  | warm | 0.8458 | 0.9060 | -0.0674 | -0.0536 |
  | cool | 0.8619 | 0.9227 | -0.0513 | -0.0370 |
  | gamma0.6 | 0.8986 | 0.9499 | -0.0146 | -0.0097 |
  | gamma1.6 | 0.8839 | 0.9392 | -0.0293 | -0.0204 |
  | jpeg-q40 | 0.9068 | 0.9548 | -0.0064 | -0.0049 |
  | jpeg-q20 | 0.8873 | 0.9405 | -0.0259 | -0.0192 |

#### osnet_ain_ms_d_c - 2.2M

- official dataset eval

  | dataset | queries | gallery | mAP | R1 | R5 | R10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | market | 3,368 | 15,913 | 0.4580 | 0.7304 | 0.8655 | 0.9047 |
  | msmt17 | 11,659 | 82,161 | 0.4869 | 0.7613 | 0.8662 | 0.8965 |
  | duke_occ | 2,210 | 17,661 | 0.4757 | 0.6167 | 0.7670 | 0.8163 |
  | cuhk03np | 1,400 | 5,332 | 0.5776 | 0.6079 | 0.7779 | 0.8543 |
  | occ_reid | 1,000 | 1,000 | 0.7407 | 0.8040 | 0.8970 | 0.9320 |

- official dataset style-shift eval - query only shifted

  | condition | mAP | R1 | dmAP | dR1 |
  | --- | ---: | ---: | ---: | ---: |
  | clean | 0.5001 | 0.7310 | — | — |
  | bright+30% | 0.4945 | 0.7236 | -0.0055 | -0.0074 |
  | dark-30% | 0.4992 | 0.7307 | -0.0009 | -0.0003 |
  | contrast-40% | 0.4903 | 0.7220 | -0.0097 | -0.0090 |
  | contrast+40% | 0.4101 | 0.6127 | -0.0900 | -0.1183 |
  | warm | 0.4159 | 0.6370 | -0.0841 | -0.0940 |
  | cool | 0.4340 | 0.6668 | -0.0661 | -0.0642 |
  | gamma0.6 | 0.4791 | 0.7068 | -0.0210 | -0.0242 |
  | gamma1.6 | 0.4492 | 0.6788 | -0.0508 | -0.0521 |
  | jpeg-q40 | 0.4819 | 0.7087 | -0.0182 | -0.0223 |
  | jpeg-q20 | 0.4379 | 0.6571 | -0.0621 | -0.0738 |

#### `-ain` variants vs the standard ladder

Every tier exists (or is planned) in two flavors that share the same training recipe, data and evaluation protocol; the only difference is where the network normalizes:

| | Standard ladder (B/S/T/N/P/F/A) | `-ain` ladder (B-ain, S-ain, ...) |
| --- | --- | --- |
| Normalization | BatchNorm (CNN) / LayerNorm (ViT) only | Adds instance normalization at style-sensitive early positions: token-axis IN after the ViT patch embedding; the searched OSNet-AIN placement (IN stem + four IN blocks) for CNN tiers |
| What the IN does | — | Removes each image's own style statistics (illumination, color cast, camera tone) from the features at inference time |
| In-distribution accuracy | Highest on the unified test set | Slightly lower by design |
| Unseen-environment robustness | Sensitive to camera/style shift; BatchNorm also carries training-set statistics into deployment | Style-invariant features and per-sample normalization — measured with the style-shift probe (`tools/eval_style_shift.py`, shifted queries vs clean gallery): mean mAP drop over 8 photometric shifts falls from 5.2 to 3.4 (B pair), 5.3 to 3.9 (S pair), 10.9 to 5.8 (N pair) and 10.3 to 5.1 (P pair), with exact-zero degradation under uniform gain/contrast shifts; under the hardest shift the `-ain` models beat their BN siblings in absolute mAP despite the lower clean score; photometric-augmentation fine-tunes (`*-ain-aug`, `INPUT.CJ_PROB`/`INPUT.BLUR_PROB`) further cut the mean drop (B: 1.3, S: 1.4, P: 3.4, N: 3.7, T: 3.9) while also raising clean mAP (B: 93.5, S: 93.0, P: 88.1, N: 88.4, T: 88.7); all five rows additionally include the JPEG-compression augmentation (`INPUT.JPEG_PROB`, roughly halving the jpeg-q20 probe drop on every tier), and the B and S numbers include the camera-aware proxy contrastive loss (`CAMPROXY`, see [`docs/paper_findings_ablation_plan.md`](docs/paper_findings_ablation_plan.md)) adopted from the paper-component ablation |
| Teacher for distilled tiers | B | B-ain |
| ONNX | BatchNorm folds away entirely | InstanceNormalization nodes remain (runtime normalization; ViT: 1 node, OSNet: 5) with a small latency overhead |

Choose the standard ladder when the deployment cameras resemble the training domains, and the `-ain` ladder when deploying to new environments without target-domain fine-tuning.

### Evaluating on the original datasets' official splits

[`tools/eval_official.py`](transreid_pytorch/tools/eval_official.py) runs any
trained model against the original datasets' own query/gallery protocols
(Market-1501, MSMT17, Occluded-Duke, CUHK03-NP detected, and Occluded-REID
with occluded queries vs whole-body gallery) as a per-dataset performance
reference.

The loaders expect the canonical directory names under
`transreid_pytorch/data/`; symlinks onto the original distributions are
sufficient:

```shell
cd transreid_pytorch/data
ln -sfn Market-1501-v15.09.15 market1501
ln -sfn MSMT17_V1 MSMT17
ln -sfn Occluded-DukeMTMC Occluded_Duke
# CUHK03-NP and Occluded_REID are used under their own names
```

Then evaluate any variant by pairing its config with its best checkpoint:

```shell
cd transreid_pytorch
python tools/eval_official.py \
--config configs/reid/osnet_n_8gb_distill.yml \
--weight "logs/reid_osnet_n_8gb_distill/transformer_best_*.pth"

# selected datasets only: market / msmt17 / duke_occ / cuhk03np / occ_reid
python tools/eval_official.py --config configs/reid/vit_base_8gb.yml \
--weight "logs/reid_vit_base_8gb/transformer_best_*.pth" \
--datasets market occ_reid
```

The model is built once and reused across datasets; reported columns are mAP / Rank-1 / Rank-5 / Rank-10 per dataset. The MSMT17 protocol compares 11,659 queries against 82,161 gallery images and needs roughly 15 GB of host RAM for its distance and ranking matrices. Both tools accept `--markdown` for paste-ready tables.

[`tools/eval_official_onnx.py`](transreid_pytorch/tools/eval_official_onnx.py) runs the same protocols through ONNX Runtime on a deployment artifact — exported models or third-party graphs alike (input/output tensor names are taken from the session, and features are L2-normalized on the evaluation side). The config supplies only the input pipeline; third-party torchreid models expect ImageNet normalization, overridable via the trailing opts:

```shell
# our exported artifact (0.5/0.5 normalization from the config)
python tools/eval_official_onnx.py \
--config configs/reid/osnet_p_8gb_distill_ain_aug2.yml \
--onnx ../onnx/osnet_ain_x1_0_p_unified_aug_n.onnx

# upstream torchreid OSNet-AIN (ImageNet normalization; trained on MS+D+C,
# so only market / occ_reid are genuinely cross-domain references)
python tools/eval_official_onnx.py \
--config configs/reid/osnet_p_8gb_distill_ain.yml \
--onnx ../onnx/osnet_ain_ms_d_c_Nx3x256x128.onnx \
INPUT.PIXEL_MEAN "[0.485,0.456,0.406]" INPUT.PIXEL_STD "[0.229,0.224,0.225]"
```

### Style-shift robustness probe

[`tools/eval_style_shift.py`](transreid_pytorch/tools/eval_style_shift.py) measures how much retrieval quality degrades when the capture style changes, without leaving the unified 5-domain protocol: the queries are re-rendered under eight deterministic photometric shifts (brightness ±30%, contrast ±40%, warm/cool color temperature, gamma 0.6/1.6; defined in [`datasets/style_shift.py`](transreid_pytorch/datasets/style_shift.py)) and matched against the clean gallery — simulating new cameras or lighting joining a deployment. This probe produced the BN-vs-`-ain` robustness numbers in the comparison table above.

```shell
cd transreid_pytorch
python tools/eval_style_shift.py \
--config configs/reid/osnet_p_8gb_distill_ain_aug2.yml \
--weight "logs/reid_osnet_p_8gb_distill_ain_aug2/transformer_best_*.pth"
```

- `--weight` accepts a glob and also resume-format checkpoints (`checkpoint_last.pth`), so both the selected best and the final-epoch model can be probed.
- `--mode all` shifts the gallery too (a fully re-deployed camera network); the default query-only mode is the more discriminative setting.
- `--dataset official` runs the probe over the five source datasets' official splits instead of the unified test split — matching stays within each dataset, and the summary table aggregates all queries (query-count weighted). MSMT17 dominates the runtime of this variant.
- `--markdown` prints a paste-ready table; trailing `KEY VALUE` pairs override the config as usual.

The `dmAP`/`dR1` columns are the drops versus the clean condition; compare models by the mean drop over the eight shifts and the absolute mAP under the worst shift (typically `warm`). `-ain` models are expected to show exactly 0.0000 drop under `dark-30%` and `contrast-40%` — token-IN removes uniform affine pixel changes mathematically. Gallery features are extracted once and reused, so a probe takes roughly 10 minutes for the OSNet tiers and ~30 minutes for ViT-B on an RTX 3070.

### Evaluation result caching

All four evaluation tools (`eval_official.py`, `eval_official_onnx.py`, `eval_per_domain.py`, `eval_style_shift.py`) persist their numeric results to `eval_cache.json` next to the evaluated checkpoint (or ONNX file), keyed by tool, checkpoint identity (name/size/mtime), config, and evaluation parameters. Re-running the same command — for example only to switch between the plain and `--markdown` table formats — reuses the stored numbers and skips feature extraction entirely (a `cache : reused ...` line marks it; `eval_official*` reuses per dataset, so adding datasets recomputes only the missing ones). Pass `--recompute` to force a fresh evaluation. Fresh results are additionally appended with timestamps to `eval_log.txt` in the same directory, building a per-run evaluation history.

## ONNX export

[`export_onnx.py`](export_onnx.py) exports all eight supervised PersonViTReID models (four datasets, each with ViT-S/16 and ViT-B/16) to the [`onnx`](onnx) directory. Missing PyTorch checkpoints are downloaded from the pinned `lakeAGI/PersonViTReID` revision automatically.

Install the export and validation dependencies, then run:

```shell
pip install huggingface_hub onnx onnxruntime onnxsim
python export_onnx.py
```

The fixed-batch models are listed below. Each one is accompanied by a dynamic-
batch model with `_n` before the extension; for example,
`personvit_market_vits16_e0220_n.onnx`.

| Dataset | ViT-S/16 | ViT-B/16 |
| --- | --- | --- |
| Market1501 | `personvit_market_vits16_e0220.onnx` | `personvit_market_vitb16_e0260.onnx` |
| MSMT17 | `personvit_msmt_vits16_e0220.onnx` | `personvit_msmt_vitb16_e0260.onnx` |
| DukeMTMC-reID | `personvit_duke_vits16_e0220.onnx` | `personvit_duke_vitb16_e0260.onnx` |
| Occluded-Duke | `personvit_occ_duke_vits16_e0220.onnx` | `personvit_occ_duke_vitb16_e0260.onnx` |

The models have the following interfaces:

- Fixed model: input `images` has float32 NCHW shape `[1, 3, 256, 128]`; output `embeddings` has shape `[1, 384]` for ViT-S/16 or `[1, 768]` for ViT-B/16.
- `_n.onnx` model: the batch axis is symbolic `N`; the corresponding input and output shapes are `[N, 3, 256, 128]` and `[N, 384]` or `[N, 768]`.
- Preprocessing: resize the RGB person crop to 256 x 128, scale pixels to `[0, 1]`, then normalize each channel with mean `[0.5, 0.5, 0.5]` and standard deviation `[0.5, 0.5, 0.5]`.
- Output embeddings are L2 normalized.

Example ONNX Runtime inference:

```python
import cv2
import numpy as np
import onnxruntime as ort

image = cv2.imread("person.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (128, 256)).astype(np.float32) / 255.0
image = (image - 0.5) / 0.5
images = np.transpose(image, (2, 0, 1))[None]

session = ort.InferenceSession(
    "onnx/personvit_msmt_vits16_e0220.onnx",
    providers=["CPUExecutionProvider"],
)
embeddings = session.run(["embeddings"], {"images": images})[0]
```

To export only selected fixed/dynamic pairs or replace existing files:

```shell
python export_onnx.py --models market-vits16 msmt-vitb16
python export_onnx.py --force
```

### Unified-dataset OSNet tiers

The distilled OSNet tiers trained on the unified `reid` dataset export
through the same fixed-batch-then-rewrite pipeline with a CNN-specific
rewrite path (the single flatten `Reshape` gets an explicit channel count
and `-1` only for the dynamic batch axis; the ViT attention/CLS validation
is replaced by pure-CNN structural checks). The one `BatchNormalization`
that onnxsim leaves behind — the fc `Gemm -> BatchNorm1d` pair — is folded
into the Gemm as an exact per-channel affine transform, so the exported
OSNet graphs contain no BatchNormalization nodes at all (enforced by the
structural validation). The exported graphs contain no
ViT operations — the checkpoint's BNNeck and classifier are dropped and only
the OSNet backbone plus L2 normalization remains — hence the OSNet-first
file naming:

| Tier | Backbone | Fixed model | Embedding |
| --- | --- | --- | ---: |
| T | OSNet x1.5 | `osnet_x1_5_t_unified.onnx` | 512 |
| N | OSNet x1.25 | `osnet_x1_25_n_unified.onnx` | 512 |
| P | OSNet x1.0 | `osnet_x1_0_p_unified.onnx` | 512 |
| F | OSNet x0.75 | `osnet_x0_75_f_unified.onnx` | 512 |
| A | OSNet x0.5 | `osnet_x0_5_a_unified.onnx` | 512 |

Each fixed model is again paired with a dynamic-batch `*_n.onnx`. The
input interface and preprocessing are identical to the released models
(`[N, 3, 256, 128]`, mean/std 0.5); outputs are L2-normalized 512-dim
embeddings. Checkpoints resolve locally from
`transreid_pytorch/logs/reid_osnet_<tier>_8gb_distill/transformer_best_*.pth`:

```shell
python export_onnx.py --models unified      # every tier with a trained checkpoint
python export_onnx.py --models p n          # individual tiers
```

### `-ain-aug` deployment models

The robustness-recommended `-ain-aug` fine-tunes export through the same pipeline. Their InstanceNormalization nodes normalize at runtime and cannot be folded, so the structural validation pins the exact node count per model (1 token-IN for the ViT tiers; 5 for OSNet-AIN x1.0 — the IN stem plus four `OSBlockINin` blocks); every remaining BatchNormalization still folds away:

| Model | Backbone | Fixed model | Embedding | IN nodes |
| --- | --- | --- | ---: | ---: |
| B-ain-aug | ViT-B/16 + token-IN | `personvit_vitb16_ain_unified_aug.onnx` | 768 | 1 |
| S-ain-aug | ViT-S/16 + token-IN | `personvit_vits16_ain_unified_aug.onnx` | 384 | 1 |
| T-ain-aug | OSNet-AIN x1.5 | `osnet_ain_x1_5_t_unified_aug.onnx` | 512 | 5 |
| N-ain-aug | OSNet-AIN x1.25 | `osnet_ain_x1_25_n_unified_aug.onnx` | 512 | 5 |
| P-ain-aug | OSNet-AIN x1.0 | `osnet_ain_x1_0_p_unified_aug.onnx` | 512 | 5 |

Checkpoints resolve locally from the corresponding
`transreid_pytorch/logs/.../transformer_best_*.pth` (the CNN tiers use the
round-2 `reid_osnet_*_8gb_distill_ain_aug2` bests, the weights behind the
README results rows):

```shell
python export_onnx.py --models b-ain-aug s-ain-aug t-ain-aug n-ain-aug p-ain-aug
```

For every checkpoint, the exporter first creates and validates the fixed batch-1
model. It then derives the `_n.onnx` graph from that model. Every `Reshape`
target explicitly specifies all non-batch dimensions; zero-copy dimensions are
not used and `-1` appears only on the leading axis of dynamic targets. onnxsim
(0.7) rewrites each transformer linear layer into a 2-D `Gemm`, so the ViT
graphs carry four validated `Reshape` classes (patch embedding, 36 token
flattens, 12 rank-5 qkv splits, 24 token unflattens). The token flattens fuse
batch and tokens into one leading axis whose extent is `N*129`; fixed models
pin it to `129`, `_n.onnx` models keep `-1` there and annotate the inferred
tensors with the `129N` symbol (plain batch axes use `N`). For example, the
ViT-S/16 patch embedding target is `[1, 384, 128]` in the fixed model and
`[-1, 384, 128]` in the dynamic model.

At `/backbone/Concat`, the symbolic batch size is derived locally from the
adjacent patch embeddings. An all-1.0 tensor with shape `[N, 1, 1]` is
multiplied by the constant CLS token `[1, 1, D]`, and the resulting `[N, 1, D]`
tensor is concatenated with the patch embeddings. No shape-processing branch
is added directly to the public model input.

The 5-D attention transpose remains
`[B, tokens, 3, heads, head_dim] -> [3, B, heads, tokens, head_dim]` with
permutation `[2, 0, 3, 1, 4]`. The exporter explicitly checks all 12 such
attention transposes in every graph, as well as the local CLS broadcast
topology.

Every fixed export is checked with the ONNX checker and simplified with
`onnxsim`. After the N-batch graph rewrite is complete, the resulting
`_n.onnx` model is always passed through `onnxsim` one more time. Structural
validation and the numerical comparison with PyTorch are performed only after
this final simplification.

The final dynamic graph is shape-inferred together with its fixed batch-1
counterpart. Every generated `unk*` dimension that is proven to correspond to
a fixed batch dimension is canonicalized to `N`. Complete tensor type and shape
information is then materialized for every operator input and output, including
initializers, so graph viewers such as Netron display shapes throughout the
model. The exporter rejects the graph if any non-`N` symbolic dimension or
missing operator output information remains.

Fixed models are tested with batch size 1, and `_n.onnx` models are tested with
batch sizes 1 and 2. Only ONNX model files are written; no JSON manifest or
validation report is generated. Use `python export_onnx.py --help` for
local-checkpoint, device, validation, and model-selection options.
