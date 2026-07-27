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

## ReID Fine-tuning and  Evaluating
first download the pretrained models from [ViT-S/16](https://huggingface.co/lakeAGI/PersonViT/tree/main/vits.lup.256x128.wopt.csk.4-8.ar.375.n8) and save it to pretrained
```shell
cd transreid_pytorch
sh run_epochs.sh ../pretrained/vits.lup.256x128.wopt.csk.4-8.ar.375.n8/ vits.lup.256x128.wopt.csk.4-8.ar.375.n8 220 0 2 small
```
