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

Best checkpoints per variant, evaluated on the unified test split (1,362
identities; 4,744 query / 29,942 gallery images across the five domains; no
re-ranking). These numbers are not comparable to the single-benchmark tables
above. All runs: single RTX 3070 (8 GB), AMP, batch 64, the `*_8gb` configs
as committed.

| Var | Backbone | Params | GFLOPs @256x128 | Emb | Trained by | mAP | Rank-1 | Rank-5 | Rank-10 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| B | ViT-B/16 (768 / 12 layers) | 86.5M | 11.35 | 768 | Fine-tuning from `checkpoint0260.pth` (60 epochs) | **93.3** | **97.1** | **98.2** | 98.4 |
| B-ain | ViT-B/16 + token-IN | 86.5M | 11.35 | 768 | Fine-tuning from `checkpoint0260.pth` (75 epochs) | 92.1 | 96.5 | 97.8 | 98.2 |
| S | ViT-S/16 (384 / 12 layers) | 22.0M | 2.94 | 384 | Distillation from B (60 epochs) | 92.2 | 96.9 | 98.1 | **98.6** |
| S-ain | ViT-S/16 + token-IN | 22.0M | 2.94 | 384 | Distillation from B-ain (75 epochs) | 91.4 | 96.3 | 97.9 | 98.4 |
| **N** | OSNet x1.25 | 3.3M | 1.49 | 512 | Distillation from B (100 epochs, Adam 3.5e-4), function-preserving expansion of the distilled P | 90.6 | 96.0 | 97.9 | 98.4 |
| **P** | OSNet x1.0 | 2.2M | 0.98 | 512 | Distillation from B (100 epochs, Adam 3.5e-4), OSNet ImageNet init | 90.0 | 96.1 | 97.8 | 98.4 |

The ladder below S is the OSNet family: **T = OSNet x1.5** (4.6M / 2.12
GFLOPs), **N = OSNet x1.25** (3.3M / 1.49 GFLOPs), P = OSNet x1.0,
F = x0.75, A = x0.5. Results for the remaining tiers are added as their
training completes. The retired ViT student candidates and their measured
results are recorded in
[`docs/lightweight_students.md`](docs/lightweight_students.md).

- Best files: `logs/reid_vit_base_8gb/transformer_best_e000060_map0.93305.pth`,
  `logs/reid_vit_base_8gb_ain/transformer_best_e000055_map0.92092.pth`,
  `logs/reid_vit_small_8gb_distill/transformer_best_e000057_map0.92205.pth`,
  `logs/reid_vit_small_8gb_distill_ain/transformer_best_e000075_map0.91395.pth`,
  `logs/reid_osnet_n_8gb_distill/transformer_best_e000099_map0.90604.pth`
  and `logs/reid_osnet_p_8gb_distill/transformer_best_e000099_map0.90048.pth`.
- Wall-clock on the RTX 3070: ~12.9 h (B, ~750 s/epoch), ~8.5 h (S + teacher
  forward), ~15.5 h (P, 100 epochs), ~18 h (N, 100 epochs); per-epoch
  evaluation included.
- The distilled S keeps within 1.1 mAP / 0.2 Rank-1 of its 4x-larger teacher
  and slightly beats it at Rank-10; the distilled P keeps within 2.2 mAP of
  the 10x-larger S at a quarter of its FLOPs. N starts from a
  function-preserving width expansion of the distilled P (zero-shot identical
  to P), dips during warmup and recovers to +0.6 mAP over P — the expansion
  chain P -> N -> T carries accumulated gains forward.
- B-ain is the teacher of the domain-generalization (`-ain`) ladder
  ([`docs/ain_variants.md`](docs/ain_variants.md)): token-axis instance
  normalization after the patch embedding, trained with the B recipe over 75
  epochs (the token-IN insertion costs a few adaptation epochs and, at
  convergence, 1.2 mAP of in-distribution accuracy versus B — the accepted
  price of style invariance). Trained across two GPUs: epochs 1-21 on the
  RTX 3070, epochs 22-75 resumed on an RTX PRO 6000 (~2.7 min/epoch).

#### `-ain` variants vs the standard ladder

Every tier exists (or is planned) in two flavors that share the same
training recipe, data and evaluation protocol; the only difference is where
the network normalizes:

| | Standard ladder (B/S/T/N/P/F/A) | `-ain` ladder (B-ain, S-ain, ...) |
| --- | --- | --- |
| Normalization | BatchNorm (CNN) / LayerNorm (ViT) only | Adds instance normalization at style-sensitive early positions: token-axis IN after the ViT patch embedding; the searched OSNet-AIN placement (IN stem + four IN blocks) for CNN tiers |
| What the IN does | — | Removes each image's own style statistics (illumination, color cast, camera tone) from the features at inference time |
| In-distribution accuracy | Highest on the unified test set | Slightly lower by design (measured: B-ain 92.1 vs B 93.3 mAP) |
| Unseen-environment robustness | Sensitive to camera/style shift; BatchNorm also carries training-set statistics into deployment | Style-invariant features and per-sample normalization; the intended advantage on cameras and lighting never seen in training (not measurable on this in-distribution benchmark) |
| Teacher for distilled tiers | B | B-ain (so distillation reinforces the invariance instead of fighting it) |
| ONNX | BatchNorm folds away entirely | InstanceNormalization nodes remain (runtime normalization; ViT: 1 node, OSNet: 5) with a small latency overhead |

Choose the standard ladder when the deployment cameras resemble the
training domains, and the `-ain` ladder when deploying to new environments
without target-domain fine-tuning.

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

The model is built once and reused across datasets; reported columns are
mAP / Rank-1 / Rank-5 / Rank-10 per dataset. The MSMT17 protocol compares
11,659 queries against 82,161 gallery images and needs roughly 15 GB of
host RAM for its distance and ranking matrices.

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
