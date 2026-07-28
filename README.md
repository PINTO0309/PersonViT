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

## ReID Fine-tuning and  Evaluating
first download the pretrained models from [ViT-S/16](https://huggingface.co/lakeAGI/PersonViT/tree/main/vits.lup.256x128.wopt.csk.4-8.ar.375.n8) and save it to pretrained
```shell
cd transreid_pytorch
sh run_epochs.sh ../pretrained/vits.lup.256x128.wopt.csk.4-8.ar.375.n8/ vits.lup.256x128.wopt.csk.4-8.ar.375.n8 220 0 2 small
```

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

For every checkpoint, the exporter first creates and validates the fixed batch-1
model. It then derives the `_n.onnx` graph from that model. Every `Reshape`
target explicitly specifies all non-batch dimensions. Fixed models use `1` for
the leading batch dimension; `_n.onnx` models use `-1` only for that leading
dimension. Zero-copy dimensions are not used, and `-1` is never used for a
non-batch dimension. For example, the ViT-S/16 patch embedding target is
`[1, 384, 128]` in the fixed model and `[-1, 384, 128]` in the dynamic model.

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
