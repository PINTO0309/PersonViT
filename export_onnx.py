#!/usr/bin/env python3
"""Export all released supervised PersonViT ReID checkpoints to ONNX.

By default, this script downloads the eight ``transformer_120.pth`` files from
``lakeAGI/PersonViTReID`` and exports L2-normalized ReID embeddings to
``onnx/``.  For each checkpoint, it first writes a model fixed at batch 1 and
then derives an additional ``*_n.onnx`` model whose batch axis is symbolic
``N``.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from onnx import numpy_helper, shape_inference
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent
TRANSREID_ROOT = PROJECT_ROOT / "transreid_pytorch"
if str(TRANSREID_ROOT) not in sys.path:
    sys.path.insert(0, str(TRANSREID_ROOT))

from config.defaults import _C as DEFAULT_CFG  # noqa: E402
from model import make_model  # noqa: E402


HF_REPO_ID = "lakeAGI/PersonViTReID"
HF_REVISION = "20bc52b77acb97e034fcb526b0fa109bb491306d"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 128
PATCH_TOKENS = (IMAGE_HEIGHT // 16) * (IMAGE_WIDTH // 16)
TOKEN_COUNT = PATCH_TOKENS + 1
# The Gemm-form ViT graph fuses batch and tokens ahead of every linear layer;
# tensors along that axis carry N*TOKEN_COUNT elements and are canonicalized
# with this symbol (plain batch axes use "N").
TOKEN_BATCH_SYMBOL = f"{TOKEN_COUNT}N"
PIXEL_MEAN = (0.5, 0.5, 0.5)
PIXEL_STD = (0.5, 0.5, 0.5)

# The `_attn` OSNet variants append one LiteSelfAttention block (bottleneck
# dim 128, 4 heads) over the 16x8 final feature map. Its Gemm-form graph
# carries seven constant Reshapes per block on top of the single OSNet
# flatten, and one fused batch*token axis analogous to the ViT `129N`.
OSNET_MAP_HEIGHT = IMAGE_HEIGHT // 16
OSNET_MAP_WIDTH = IMAGE_WIDTH // 16
OSNET_TOKEN_COUNT = OSNET_MAP_HEIGHT * OSNET_MAP_WIDTH
OSNET_TOKEN_BATCH_SYMBOL = f"{OSNET_TOKEN_COUNT}N"
ATTN_DIM = 128
ATTN_HEADS = 4
ATTN_HEAD_DIM = ATTN_DIM // ATTN_HEADS


@dataclass(frozen=True)
class ReleasedModel:
    key: str
    dataset: str
    architecture: str
    config: str
    checkpoint: str
    output: str
    pretraining_epoch: int
    embedding_dimension: int
    # "vit" uses the ViT-specific graph rewrite; "osnet" uses the CNN rewrite
    # path. Both share the fixed-batch-then-rewrite export flow. Checkpoints
    # come from HF for the released ViT models and from local training logs
    # whenever the checkpoint path is a glob.
    family: str = "vit"
    # -ain variants keep InstanceNormalization at inference (it normalizes at
    # runtime and cannot be folded); the exact node count is pinned here.
    instance_norm_nodes: int = 0
    # LiteSelfAttention blocks in `_attn` OSNet variants (adds Softmax, one
    # extra Gemm, and seven classified Reshapes per block).
    attention_blocks: int = 0


def _released_model(
    dataset_key: str,
    dataset_name: str,
    config_directory: str,
    architecture: str,
) -> ReleasedModel:
    if architecture == "small":
        architecture_token = "vits"
        architecture_name = "ViT-S/16"
        epoch = 220
        embedding_dimension = 384
    elif architecture == "base":
        architecture_token = "vitb"
        architecture_name = "ViT-B/16"
        epoch = 260
        embedding_dimension = 768
    else:  # pragma: no cover - only called by the static model list below
        raise ValueError(f"Unsupported architecture: {architecture}")

    directory = (
        f"{dataset_key}.{architecture_token}.lup.256x128.wopt."
        f"csk.4-8.ar.375.n8.e0{epoch}"
    )
    short_architecture = "vits16" if architecture == "small" else "vitb16"
    return ReleasedModel(
        key=f"{dataset_key}-{short_architecture}",
        dataset=dataset_name,
        architecture=architecture_name,
        config=(
            f"transreid_pytorch/configs/{config_directory}/vit_{architecture}.yml"
        ),
        checkpoint=f"{directory}/transformer_120.pth",
        output=f"personvit_{dataset_key}_{short_architecture}_e0{epoch}.onnx",
        pretraining_epoch=epoch,
        embedding_dimension=embedding_dimension,
    )


RELEASED_MODELS = (
    _released_model("market", "Market1501", "market", "small"),
    _released_model("market", "Market1501", "market", "base"),
    _released_model("msmt", "MSMT17", "msmt17", "small"),
    _released_model("msmt", "MSMT17", "msmt17", "base"),
    _released_model("duke", "DukeMTMC-reID", "dukemtmc", "small"),
    _released_model("duke", "DukeMTMC-reID", "dukemtmc", "base"),
    _released_model("occ_duke", "Occluded-Duke", "occ_duke", "small"),
    _released_model("occ_duke", "Occluded-Duke", "occ_duke", "base"),
)
RELEASED_MODEL_BY_KEY = {model.key: model for model in RELEASED_MODELS}


def _unified_osnet_model(tier: str, multiplier_token: str) -> ReleasedModel:
    """Distilled OSNet tiers trained on the unified `reid` dataset.

    The exported graphs contain no ViT operations (the wrapper exports the
    OSNet backbone plus L2 normalization only), hence the OSNet-first
    file naming ``osnet_<multiplier>_<tier>_unified.onnx``.
    """

    multiplier_name = multiplier_token.replace("_", ".")
    return ReleasedModel(
        key=tier,
        dataset="unified",
        architecture=f"OSNet {multiplier_name}",
        config=f"transreid_pytorch/configs/reid/osnet_{tier}_8gb_distill.yml",
        checkpoint=(
            f"transreid_pytorch/logs/reid_osnet_{tier}_8gb_distill/"
            "transformer_best_*.pth"
        ),
        output=f"osnet_{multiplier_token}_{tier}_unified.onnx",
        pretraining_epoch=0,
        embedding_dimension=512,
        family="osnet",
    )


UNIFIED_OSNET_MODELS = (
    _unified_osnet_model("t", "x1_5"),
    _unified_osnet_model("n", "x1_25"),
    _unified_osnet_model("p", "x1_0"),
    _unified_osnet_model("f", "x0_75"),
    _unified_osnet_model("a", "x0_5"),
)
UNIFIED_OSNET_MODEL_BY_KEY = {model.key: model for model in UNIFIED_OSNET_MODELS}


# The photometric-augmentation fine-tunes of the -ain ladder (the current
# robustness-recommended deployment models). The ViT tiers carry one token-IN
# InstanceNormalization; OSNet-AIN x1.0 carries five (IN stem + 4 OSBlockINin).
AIN_AUG_MODELS = (
    ReleasedModel(
        key="b-ain-aug",
        dataset="unified",
        architecture="ViT-B/16",
        config="transreid_pytorch/configs/reid/vit_base_8gb_ain_aug2_cam_jpeg.yml",
        checkpoint=(
            "transreid_pytorch/logs/reid_vit_base_8gb_ain_synth_cam_jpeg2/"
            "transformer_best_*.pth"
        ),
        output="personvit_vitb16_ain_unified_aug.onnx",
        pretraining_epoch=260,
        embedding_dimension=768,
        instance_norm_nodes=1,
    ),
    ReleasedModel(
        key="s-ain-aug",
        dataset="unified",
        architecture="ViT-S/16",
        config="transreid_pytorch/configs/reid/vit_small_8gb_distill_ain_aug2_jpeg.yml",
        checkpoint=(
            "transreid_pytorch/logs/reid_vit_small_8gb_distill_ain_synth_embed/"
            "transformer_best_*.pth"
        ),
        output="personvit_vits16_ain_unified_aug.onnx",
        pretraining_epoch=220,
        embedding_dimension=384,
        instance_norm_nodes=1,
    ),
    ReleasedModel(
        key="p-ain-aug",
        dataset="unified",
        architecture="OSNet-AIN x1.0",
        config="transreid_pytorch/configs/reid/osnet_p_8gb_distill_ain_aug2_jpeg.yml",
        checkpoint=(
            "transreid_pytorch/logs/reid_osnet_p_8gb_distill_ain_synth_shint/"
            "folded_best.pth"
        ),
        output="osnet_ain_x1_0_p_unified_aug.onnx",
        pretraining_epoch=0,
        embedding_dimension=512,
        family="osnet",
        instance_norm_nodes=5,
    ),
    ReleasedModel(
        key="n-ain-aug",
        dataset="unified",
        architecture="OSNet-AIN x1.25",
        config="transreid_pytorch/configs/reid/osnet_n_8gb_distill_ain_aug2_jpeg.yml",
        checkpoint=(
            "transreid_pytorch/logs/reid_osnet_n_8gb_distill_ain_synth_shint/"
            "folded_e40.pth"
        ),
        output="osnet_ain_x1_25_n_unified_aug.onnx",
        pretraining_epoch=0,
        embedding_dimension=512,
        family="osnet",
        instance_norm_nodes=5,
    ),
    ReleasedModel(
        key="t-ain-aug",
        dataset="unified",
        architecture="OSNet-AIN x1.5",
        config="transreid_pytorch/configs/reid/osnet_t_8gb_distill_ain_aug2_jpeg.yml",
        checkpoint=(
            "transreid_pytorch/logs/reid_osnet_t_8gb_distill_ain_synth_embed/"
            "folded_e40.pth"
        ),
        output="osnet_ain_x1_5_t_unified_aug.onnx",
        pretraining_epoch=0,
        embedding_dimension=512,
        family="osnet",
        instance_norm_nodes=5,
    ),
)
AIN_AUG_MODEL_BY_KEY = {model.key: model for model in AIN_AUG_MODELS}


def _uses_local_checkpoint(spec: ReleasedModel) -> bool:
    return spec.family == "osnet" or "*" in spec.checkpoint


class ReIDExportWrapper(nn.Module):
    """Expose the deployment feature and its evaluation-time normalization."""

    def __init__(self, model: nn.Module, l2_normalize: bool = True):
        super().__init__()
        # The released configs use TEST.NECK_FEAT='before', so the benchmarked
        # inference descriptor is the Transformer backbone output before BNNeck.
        self.backbone = model.base
        self.l2_normalize = bool(l2_normalize)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.backbone(images)
        if self.l2_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-12)
        return embeddings


def unwrap_state_dict(checkpoint: Any) -> OrderedDict[str, torch.Tensor]:
    """Return a normalized state dict without silently discarding parameters."""

    state_dict = checkpoint
    if isinstance(state_dict, Mapping):
        for key in ("state_dict", "model", "module"):
            nested = state_dict.get(key)
            if isinstance(nested, Mapping):
                state_dict = nested
                break
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Checkpoint does not contain a state dict: {type(state_dict)}")

    normalized: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError(
                "The supervised checkpoint must contain only string-to-tensor "
                f"entries, but got {key!r}: {type(value)}"
            )
        normalized[key.removeprefix("module.")] = value
    return normalized


def resolve_checkpoint(
    spec: ReleasedModel,
    checkpoint_root: Path | None,
    cache_dir: Path | None,
    revision: str,
) -> Path:
    if _uses_local_checkpoint(spec):
        base = checkpoint_root if checkpoint_root is not None else PROJECT_ROOT
        matches = sorted(base.glob(spec.checkpoint))
        if not matches:
            raise FileNotFoundError(
                f"No trained checkpoint matches {base / spec.checkpoint}; "
                f"train tier '{spec.key}' first"
            )
        return matches[-1].resolve()

    if checkpoint_root is not None:
        path = checkpoint_root / spec.checkpoint
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path.resolve()

    return Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=spec.checkpoint,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    ).resolve()


def build_inference_model(
    spec: ReleasedModel,
    checkpoint_path: Path,
) -> nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    state_dict = unwrap_state_dict(checkpoint)

    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is None or classifier_weight.ndim != 2:
        raise RuntimeError(
            f"Cannot infer the supervised class count from {checkpoint_path}"
        )
    num_classes, embedding_dimension = classifier_weight.shape
    if embedding_dimension != spec.embedding_dimension:
        raise RuntimeError(
            f"Architecture mismatch for {spec.key}: expected embedding dimension "
            f"{spec.embedding_dimension}, checkpoint has {embedding_dimension}"
        )

    config_path = PROJECT_ROOT / spec.config
    cfg = DEFAULT_CFG.clone()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.PRETRAIN_CHOICE = "finetune"
    cfg.MODEL.PRETRAIN_PATH = ""
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.DEVICE_ID = ""
    cfg.TEST.NECK_FEAT = "before"
    cfg.freeze()

    model = make_model(
        cfg,
        num_class=int(num_classes),
        camera_num=0,
        view_num=0,
    )
    # Training-only auxiliaries are stripped before the strict load: the
    # embedding-KD and intermediate-hint projectors exist purely for the
    # distillation loss and have no inference path.
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith(("embed_proj.", "hint_proj."))}
    # Unlike the repository's permissive inference loader, export requires an
    # exact match so a malformed or mismatched checkpoint cannot go unnoticed.
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def set_model_metadata(
    graph: onnx.ModelProto,
    spec: ReleasedModel,
    checkpoint_path: Path,
    l2_normalize: bool,
    dynamic_batch: bool,
) -> None:
    onnx.helper.set_model_props(
        graph,
        {
            "model_name": spec.key,
            "source_repository": (
                "local unified reid training"
                if _uses_local_checkpoint(spec)
                else HF_REPO_ID
            ),
            "source_checkpoint": spec.checkpoint,
            "checkpoint_file": checkpoint_path.name,
            "dataset": spec.dataset,
            "architecture": spec.architecture,
            "input_name": "images",
            "input_layout": "NCHW",
            "input_dtype": "float32",
            "input_batch": "N" if dynamic_batch else "1",
            "input_height": str(IMAGE_HEIGHT),
            "input_width": str(IMAGE_WIDTH),
            "input_pixel_range": "0.0,1.0",
            "input_mean": ",".join(map(str, PIXEL_MEAN)),
            "input_std": ",".join(map(str, PIXEL_STD)),
            "output_name": "embeddings",
            "embedding_dimension": str(spec.embedding_dimension),
            "embedding_l2_normalized": str(l2_normalize).lower(),
        },
    )


def simplify_graph(
    graph: onnx.ModelProto,
    spec: ReleasedModel,
) -> onnx.ModelProto:
    """Run onnxsim and preserve deployment metadata on the simplified model."""

    metadata = {item.key: item.value for item in graph.metadata_props}
    simplified, succeeded = onnxsim.simplify(graph)
    if not succeeded:
        raise RuntimeError(f"onnxsim validation failed for {spec.key}")
    onnx.helper.set_model_props(simplified, metadata)
    onnx.checker.check_model(simplified)
    return simplified


def _tensor_shapes(model: onnx.ModelProto) -> dict[str, list[str | int | None]]:
    shapes: dict[str, list[str | int | None]] = {}
    for value in (
        *model.graph.input,
        *model.graph.output,
        *model.graph.value_info,
    ):
        dimensions: list[str | int | None] = []
        for dimension in value.type.tensor_type.shape.dim:
            if dimension.dim_param:
                dimensions.append(dimension.dim_param)
            elif dimension.HasField("dim_value"):
                dimensions.append(dimension.dim_value)
            else:
                dimensions.append(None)
        shapes[value.name] = dimensions
    return shapes


def validate_dynamic_value_info(
    model: onnx.ModelProto,
    spec: ReleasedModel,
) -> None:
    """Require complete Netron-visible tensor metadata with batch symbol N."""

    values = {
        value.name: value
        for value in (
            *model.graph.input,
            *model.graph.output,
            *model.graph.value_info,
        )
    }
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    node_outputs = {
        name for node in model.graph.node for name in node.output if name
    }
    missing_outputs = sorted(node_outputs - values.keys())
    if missing_outputs:
        raise RuntimeError(
            f"Missing output ValueInfo for {spec.key}: {missing_outputs[:10]}"
        )
    node_inputs = {
        name for node in model.graph.node for name in node.input if name
    }
    missing_inputs = sorted(node_inputs - values.keys() - initializer_names)
    if missing_inputs:
        raise RuntimeError(
            f"Missing input type/shape information for {spec.key}: "
            f"{missing_inputs[:10]}"
        )

    invalid_values = []
    invalid_symbols = []
    for value in values.values():
        tensor_type = value.type.tensor_type
        if (
            tensor_type.elem_type == onnx.TensorProto.UNDEFINED
            or not tensor_type.HasField("shape")
        ):
            invalid_values.append(value.name)
        for dimension in tensor_type.shape.dim:
            if dimension.dim_param and dimension.dim_param not in (
                "N",
                TOKEN_BATCH_SYMBOL,
                OSNET_TOKEN_BATCH_SYMBOL,
            ):
                invalid_symbols.append((value.name, dimension.dim_param))
    if invalid_values:
        raise RuntimeError(
            f"Missing tensor type/shape metadata for {spec.key}: "
            f"{invalid_values[:10]}"
        )
    if invalid_symbols:
        raise RuntimeError(
            f"Non-canonical dynamic symbols for {spec.key}: {invalid_symbols[:10]}"
        )


def materialize_dynamic_value_info(
    dynamic: onnx.ModelProto,
    fixed: onnx.ModelProto,
    spec: ReleasedModel,
) -> onnx.ModelProto:
    """Replace proven batch-derived unk symbols with N and retain all shapes."""

    fixed = shape_inference.infer_shapes(
        fixed,
        strict_mode=True,
        data_prop=True,
    )
    dynamic = shape_inference.infer_shapes(
        dynamic,
        strict_mode=True,
        data_prop=True,
    )
    fixed_shapes = _tensor_shapes(fixed)
    fixed_shapes.update(
        {
            initializer.name: list(initializer.dims)
            for initializer in fixed.graph.initializer
        }
    )

    replacement_count = 0
    values = (
        *dynamic.graph.input,
        *dynamic.graph.output,
        *dynamic.graph.value_info,
    )
    for value in values:
        dimensions = value.type.tensor_type.shape.dim
        observed_shape = []
        for dimension in dimensions:
            if dimension.dim_param:
                observed_shape.append(dimension.dim_param)
            elif dimension.HasField("dim_value"):
                observed_shape.append(dimension.dim_value)
            else:
                observed_shape.append(None)
        unknown_axes = [
            axis
            for axis, dimension in enumerate(dimensions)
            if dimension.dim_param.startswith("unk")
        ]
        if not unknown_axes:
            continue

        axis_symbols: dict[int, str] = {}
        fixed_shape = fixed_shapes.get(value.name)
        if fixed_shape is not None:
            if len(fixed_shape) != len(observed_shape):
                raise RuntimeError(
                    f"Fixed/dynamic rank mismatch for {value.name}: "
                    f"{fixed_shape} vs {observed_shape}"
                )
            for axis, observed_dimension in enumerate(observed_shape):
                fixed_dimension = fixed_shape[axis]
                if axis in unknown_axes:
                    if fixed_dimension == 1:
                        axis_symbols[axis] = "N"
                    elif fixed_dimension == TOKEN_COUNT:
                        # the Gemm-form token flatten: N*TOKEN_COUNT elements
                        axis_symbols[axis] = TOKEN_BATCH_SYMBOL
                    elif spec.attention_blocks and fixed_dimension == OSNET_TOKEN_COUNT:
                        # the attn token flatten: N*OSNET_TOKEN_COUNT elements
                        axis_symbols[axis] = OSNET_TOKEN_BATCH_SYMBOL
                    else:
                        raise RuntimeError(
                            f"Cannot prove {value.name} axis {axis} is batch: "
                            f"fixed={fixed_dimension}, dynamic={observed_dimension}"
                        )
                elif isinstance(observed_dimension, int):
                    if observed_dimension != fixed_dimension:
                        raise RuntimeError(
                            f"Fixed/dynamic shape mismatch for {value.name} axis "
                            f"{axis}: {fixed_dimension} vs {observed_dimension}"
                        )
        elif value.name == "/backbone/cls_token_batch/ones":
            expected_shape = [observed_shape[0], 1, 1]
            if unknown_axes != [0] or observed_shape != expected_shape:
                raise RuntimeError(
                    f"Unexpected dynamic-only CLS multiplier shape: "
                    f"{observed_shape}"
                )
            axis_symbols[0] = "N"
        else:
            raise RuntimeError(
                f"Cannot prove unknown dimensions are batch-derived for "
                f"{value.name}: {observed_shape}"
            )

        for axis in unknown_axes:
            dimensions[axis].dim_param = axis_symbols[axis]
            replacement_count += 1

    if replacement_count == 0:
        raise RuntimeError(f"No inferred batch symbols were canonicalized for {spec.key}")

    # Initializers already carry type and shape in TensorProto.  Mirroring them
    # into graph ValueInfo makes every operator input self-describing in graph
    # viewers without changing runtime semantics.
    known_value_names = {
        value.name
        for value in (
            *dynamic.graph.input,
            *dynamic.graph.output,
            *dynamic.graph.value_info,
        )
    }
    for initializer in dynamic.graph.initializer:
        if initializer.name in known_value_names:
            continue
        dynamic.graph.value_info.append(
            onnx.helper.make_tensor_value_info(
                initializer.name,
                initializer.data_type,
                list(initializer.dims),
            )
        )
        known_value_names.add(initializer.name)

    metadata = {item.key: item.value for item in dynamic.metadata_props}
    metadata["final_onnxsim"] = "true"
    metadata["intermediate_batch_symbol"] = "N"
    metadata["value_info_complete"] = "true"
    onnx.helper.set_model_props(dynamic, metadata)
    validate_dynamic_value_info(dynamic, spec)
    onnx.checker.check_model(dynamic)
    return dynamic


def _transpose_permutation(node: onnx.NodeProto) -> list[int]:
    for attribute in node.attribute:
        if attribute.name == "perm":
            return list(attribute.ints)
    raise RuntimeError(f"Transpose node has no perm attribute: {node.name}")


def _node_attribute(node: onnx.NodeProto, name: str) -> Any:
    for attribute in node.attribute:
        if attribute.name == name:
            return onnx.helper.get_attribute_value(attribute)
    raise RuntimeError(f"{node.op_type} node has no {name!r} attribute: {node.name}")


def _node_attribute_or_default(node: onnx.NodeProto, name: str, default: Any) -> Any:
    for attribute in node.attribute:
        if attribute.name == name:
            return onnx.helper.get_attribute_value(attribute)
    return default


def validate_attention_transposes(
    model: onnx.ModelProto,
    spec: ReleasedModel,
    batch_dimension: str | int,
) -> None:
    """Validate the rank-5 qkv transpose without rewriting its permutation.

    qkv is reshaped to ``[B, tokens, 3, heads, head_dim]`` and transposed with
    ``[2, 0, 3, 1, 4]``.  Consequently, the batch dimension is axis 1 of the
    transpose output, not axis 0.  Treating this as an ordinary NCHW
    transpose is incorrect.
    """

    inferred = shape_inference.infer_shapes(
        model,
        strict_mode=True,
        data_prop=True,
    )
    shapes = _tensor_shapes(inferred)
    attention_transposes = []
    expected_heads = 6 if spec.architecture == "ViT-S/16" else 12
    for node in inferred.graph.node:
        input_shape = shapes.get(node.input[0]) if node.input else None
        if node.op_type != "Transpose" or input_shape is None or len(input_shape) != 5:
            continue
        attention_transposes.append(node)
        output_shape = shapes.get(node.output[0])
        actual_batch_dimension = batch_dimension
        if batch_dimension == "N":
            actual_batch_dimension = input_shape[0]
            if not isinstance(actual_batch_dimension, str):
                raise RuntimeError(
                    f"Rank-5 Transpose batch axis is not dynamic for {node.name}: "
                    f"{input_shape}"
                )
        expected_input = [
            actual_batch_dimension,
            TOKEN_COUNT,
            3,
            expected_heads,
            64,
        ]
        expected_output = [
            3,
            actual_batch_dimension,
            expected_heads,
            TOKEN_COUNT,
            64,
        ]
        if input_shape != expected_input:
            raise RuntimeError(
                f"Unexpected rank-5 Transpose input for {node.name}: "
                f"{input_shape}, expected {expected_input}"
            )
        if _transpose_permutation(node) != [2, 0, 3, 1, 4]:
            raise RuntimeError(
                f"Unexpected rank-5 Transpose perm for {node.name}: "
                f"{_transpose_permutation(node)}"
            )
        if output_shape != expected_output:
            raise RuntimeError(
                f"Unexpected rank-5 Transpose output for {node.name}: "
                f"{output_shape}, expected {expected_output}"
            )
    if len(attention_transposes) != 12:
        raise RuntimeError(
            f"Expected 12 rank-5 attention Transposes for {spec.key}, "
            f"found {len(attention_transposes)}"
        )


def validate_dynamic_cls_broadcast(
    model: onnx.ModelProto,
    spec: ReleasedModel,
) -> None:
    """Validate the local all-ones Mul used to broadcast CLS over batch N."""

    inferred = shape_inference.infer_shapes(
        model,
        strict_mode=True,
        data_prop=True,
    )
    shapes = _tensor_shapes(inferred)
    producers = {
        output: node for node in inferred.graph.node for output in node.output if output
    }
    initializers = _initializer_map(inferred)
    backbone_concat = next(
        (
            node
            for node in inferred.graph.node
            if node.op_type == "Concat" and node.name == "/backbone/Concat"
        ),
        None,
    )
    if backbone_concat is None:
        raise RuntimeError(f"Cannot find backbone Concat for {spec.key}")

    cls_mul = producers.get(backbone_concat.input[0])
    if cls_mul is None or cls_mul.op_type != "Mul":
        raise RuntimeError(f"CLS input is not produced by Mul for {spec.key}")
    cls_source = initializers.get(cls_mul.input[0])
    if cls_source is None:
        raise RuntimeError(f"CLS Mul has no constant source for {spec.key}")
    if tuple(numpy_helper.to_array(cls_source).shape) != (
        1,
        1,
        spec.embedding_dimension,
    ):
        raise RuntimeError(f"Unexpected CLS source shape for {spec.key}")

    ones_node = producers.get(cls_mul.input[1])
    if ones_node is None or ones_node.op_type != "ConstantOfShape":
        raise RuntimeError(f"CLS Mul has no ConstantOfShape input for {spec.key}")
    ones_value = numpy_helper.to_array(_node_attribute(ones_node, "value"))
    if ones_value.dtype != np.float32 or not np.array_equal(
        ones_value,
        np.ones_like(ones_value),
    ):
        raise RuntimeError(f"CLS multiplier is not all float32 1.0 for {spec.key}")

    shape_concat = producers.get(ones_node.input[0])
    if shape_concat is None or shape_concat.op_type != "Concat":
        raise RuntimeError(f"Cannot find CLS multiplier shape Concat for {spec.key}")
    if _node_attribute(shape_concat, "axis") != 0:
        raise RuntimeError(f"Unexpected CLS multiplier shape axis for {spec.key}")
    shape_node = producers.get(shape_concat.input[0])
    if shape_node is None or shape_node.op_type != "Shape":
        raise RuntimeError(f"Cannot find local CLS batch Shape for {spec.key}")
    if _node_attribute(shape_node, "start") != 0 or _node_attribute(
        shape_node,
        "end",
    ) != 1:
        raise RuntimeError(f"Unexpected local CLS batch Shape slice for {spec.key}")
    patch_embeddings = backbone_concat.input[1]
    if shape_node.input[0] != patch_embeddings:
        raise RuntimeError(
            f"CLS batch Shape is not local to patch embeddings for {spec.key}"
        )
    shape_tail = initializers.get(shape_concat.input[1])
    if shape_tail is None or not np.array_equal(
        numpy_helper.to_array(shape_tail),
        np.asarray([1, 1], dtype=np.int64),
    ):
        raise RuntimeError(f"Unexpected CLS multiplier shape tail for {spec.key}")

    patch_shape = shapes.get(patch_embeddings)
    if patch_shape is None or not isinstance(patch_shape[0], str):
        raise RuntimeError(
            f"Patch embedding batch axis is not dynamic for {spec.key}: {patch_shape}"
        )
    inferred_batch = patch_shape[0]
    expected_shapes = {
        patch_embeddings: [inferred_batch, PATCH_TOKENS, spec.embedding_dimension],
        ones_node.output[0]: [inferred_batch, 1, 1],
        cls_mul.output[0]: [inferred_batch, 1, spec.embedding_dimension],
        backbone_concat.output[0]: [
            inferred_batch,
            TOKEN_COUNT,
            spec.embedding_dimension,
        ],
    }
    for tensor_name, expected_shape in expected_shapes.items():
        if shapes.get(tensor_name) != expected_shape:
            raise RuntimeError(
                f"Unexpected dynamic CLS tensor shape for {tensor_name}: "
                f"{shapes.get(tensor_name)}, expected {expected_shape}"
            )

    direct_image_consumers = [
        node
        for node in inferred.graph.node
        if "images" in node.input
    ]
    if len(direct_image_consumers) != 1 or direct_image_consumers[0].op_type != "Conv":
        raise RuntimeError(
            f"Unexpected operations branched from the model input for {spec.key}: "
            f"{[(node.name, node.op_type) for node in direct_image_consumers]}"
        )


def _initializer_map(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    return {initializer.name: initializer for initializer in model.graph.initializer}


def _replace_initializer(
    initializer: onnx.TensorProto,
    value: np.ndarray,
) -> None:
    initializer.CopyFrom(
        numpy_helper.from_array(np.asarray(value), name=initializer.name)
    )


# onnxsim 0.7 rewrites every transformer linear layer into a 2-D Gemm and
# wraps it with token flattens/unflattens, so the simplified ViT graph holds
# 73 constant-shape Reshapes: the patch embedding, 36 token flattens, the 12
# rank-5 qkv splits and 24 token unflattens (identical for ViT-S and ViT-B).
VIT_EXPECTED_RESHAPE_COUNTS = {
    "patch_embed": 1,
    "token_flatten": 36,
    "qkv": 12,
    "token_unflatten": 24,
}


def _classify_vit_reshape(
    current: np.ndarray,
    spec: ReleasedModel,
    node_name: str,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Classify a constant Reshape target of the Gemm-form ViT graph.

    Returns the class name plus the canonical (fixed, dynamic) targets. The
    token flatten fuses batch and tokens into one leading axis, so its
    dynamic form keeps ``-1`` (the fused ``N*129`` extent has no constant
    representation); every other class reserves ``-1`` for the batch axis.
    """

    heads = 6 if spec.architecture == "ViT-S/16" else 12
    dim = spec.embedding_dimension
    values = current.tolist()
    if (
        len(values) == 3
        and values[0] in (1, -1)
        and values[1] == dim
        and values[2] in (-1, PATCH_TOKENS)
    ):
        return (
            "patch_embed",
            np.asarray([1, dim, PATCH_TOKENS], dtype=np.int64),
            np.asarray([-1, dim, PATCH_TOKENS], dtype=np.int64),
        )
    if len(values) == 2 and values[1] == dim and values[0] in (-1, TOKEN_COUNT):
        return (
            "token_flatten",
            np.asarray([TOKEN_COUNT, dim], dtype=np.int64),
            np.asarray([-1, dim], dtype=np.int64),
        )
    if len(values) == 5 and values[1:] == [TOKEN_COUNT, 3, heads, 64] and values[
        0
    ] in (1, -1):
        return (
            "qkv",
            np.asarray([1, TOKEN_COUNT, 3, heads, 64], dtype=np.int64),
            np.asarray([-1, TOKEN_COUNT, 3, heads, 64], dtype=np.int64),
        )
    if len(values) == 3 and values[1:] == [TOKEN_COUNT, dim] and values[0] in (1, -1):
        return (
            "token_unflatten",
            np.asarray([1, TOKEN_COUNT, dim], dtype=np.int64),
            np.asarray([-1, TOKEN_COUNT, dim], dtype=np.int64),
        )
    raise RuntimeError(
        f"Unclassifiable ViT Reshape target for {spec.key} at {node_name}: {values}"
    )


def set_reshape_shapes(
    model: onnx.ModelProto,
    spec: ReleasedModel,
    batch_dimension: int,
) -> None:
    """Set explicit static dimensions, reserving -1 for dynamic axes only.

    In the fixed batch-1 graph every Reshape target is fully explicit. The
    dynamic graph uses ``-1`` exclusively for the leading axis: the batch for
    the patch/qkv/unflatten targets and the fused batch-token extent for the
    Gemm flattens.
    """

    if batch_dimension not in (1, -1):
        raise ValueError(f"Unsupported Reshape batch dimension: {batch_dimension}")
    initializers = _initializer_map(model)
    reshape_nodes = [node for node in model.graph.node if node.op_type == "Reshape"]
    class_counts: dict[str, int] = dict.fromkeys(VIT_EXPECTED_RESHAPE_COUNTS, 0)
    expected_by_initializer: dict[str, np.ndarray] = {}
    for node in reshape_nodes:
        shape_name = node.input[1]
        initializer = initializers.get(shape_name)
        if initializer is None:
            raise RuntimeError(f"Reshape shape is not constant: {node.name}")
        current = numpy_helper.to_array(initializer)
        if current.ndim != 1:
            raise RuntimeError(f"Unexpected Reshape rank for {node.name}: {current}")
        klass, fixed, dynamic = _classify_vit_reshape(current, spec, node.name)
        class_counts[klass] += 1
        expected = fixed if batch_dimension == 1 else dynamic
        previous = expected_by_initializer.get(shape_name)
        if previous is not None and not np.array_equal(previous, expected):
            raise RuntimeError(
                f"Reshape initializer {shape_name!r} has conflicting uses"
            )
        expected_by_initializer[shape_name] = expected

    if class_counts != VIT_EXPECTED_RESHAPE_COUNTS:
        raise RuntimeError(
            f"Unexpected ViT Reshape census for {spec.key}: {class_counts}, "
            f"expected {VIT_EXPECTED_RESHAPE_COUNTS}"
        )
    for shape_name, expected in expected_by_initializer.items():
        _replace_initializer(initializers[shape_name], expected)


def validate_reshape_shapes(
    model: onnx.ModelProto,
    spec: ReleasedModel,
    dynamic_batch: bool,
) -> None:
    """Reject zero dimensions and non-leading inferred Reshape dimensions."""

    initializers = _initializer_map(model)
    reshape_nodes = [node for node in model.graph.node if node.op_type == "Reshape"]
    class_counts: dict[str, int] = dict.fromkeys(VIT_EXPECTED_RESHAPE_COUNTS, 0)
    for node in reshape_nodes:
        initializer = initializers.get(node.input[1])
        if initializer is None:
            raise RuntimeError(f"Reshape shape is not constant: {node.name}")
        actual = numpy_helper.to_array(initializer)
        klass, fixed, dynamic = _classify_vit_reshape(actual, spec, node.name)
        class_counts[klass] += 1
        expected = dynamic if dynamic_batch else fixed
        if not np.array_equal(actual, expected):
            raise RuntimeError(
                f"Unexpected Reshape shape for {node.name}: "
                f"{actual.tolist()}, expected {expected.tolist()}"
            )
        if np.any(actual == 0):
            raise RuntimeError(f"Reshape shape contains 0 for {node.name}")
        inferred_axes = np.flatnonzero(actual == -1).tolist()
        expected_inferred_axes = [0] if dynamic_batch else []
        if inferred_axes != expected_inferred_axes:
            raise RuntimeError(
                f"Invalid inferred Reshape axes for {node.name}: {inferred_axes}"
            )
    if class_counts != VIT_EXPECTED_RESHAPE_COUNTS:
        raise RuntimeError(
            f"Unexpected ViT Reshape census for {spec.key}: {class_counts}, "
            f"expected {VIT_EXPECTED_RESHAPE_COUNTS}"
        )


OSNET_ATTN_RESHAPE_COUNTS = {
    "attn_tokens_3d": 2,       # spatial flatten + token unflatten [b, 128, 128]
    "attn_head_split": 3,      # q/k/v [b, tokens, heads, head_dim]
    "attn_token_flatten": 1,   # Gemm-form fused batch*token axis [-1, dim]
    "attn_spatial_restore": 1,  # [b, dim, 16, 8]
    "flatten": 1,              # the original GAP flatten [b, channels]
}


def _classify_osnet_reshape(
    values: np.ndarray,
    spec: ReleasedModel,
    node_name: str,
    flatten_channels: int,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Classify a constant Reshape of the (optionally attn-carrying) OSNet
    graph, returning the class plus canonical (fixed, dynamic) targets."""

    entries = values.tolist()
    if len(entries) == 2 and entries[1] == ATTN_DIM and entries[0] in (
            -1, OSNET_TOKEN_COUNT):
        return (
            "attn_token_flatten",
            np.asarray([OSNET_TOKEN_COUNT, ATTN_DIM], dtype=np.int64),
            np.asarray([-1, ATTN_DIM], dtype=np.int64),
        )
    if len(entries) == 2 and entries[0] in (1, -1):
        return (
            "flatten",
            np.asarray([1, flatten_channels], dtype=np.int64),
            np.asarray([-1, flatten_channels], dtype=np.int64),
        )
    if (
        len(entries) == 3
        and entries[0] in (1, -1)
        and entries[1] == ATTN_DIM
        and entries[2] in (-1, OSNET_TOKEN_COUNT)
    ):
        return (
            "attn_tokens_3d",
            np.asarray([1, ATTN_DIM, OSNET_TOKEN_COUNT], dtype=np.int64),
            np.asarray([-1, ATTN_DIM, OSNET_TOKEN_COUNT], dtype=np.int64),
        )
    if len(entries) == 4 and entries[1:] == [
            OSNET_TOKEN_COUNT, ATTN_HEADS, ATTN_HEAD_DIM] and entries[0] in (1, -1):
        return (
            "attn_head_split",
            np.asarray([1, OSNET_TOKEN_COUNT, ATTN_HEADS, ATTN_HEAD_DIM],
                       dtype=np.int64),
            np.asarray([-1, OSNET_TOKEN_COUNT, ATTN_HEADS, ATTN_HEAD_DIM],
                       dtype=np.int64),
        )
    if len(entries) == 4 and entries[1:] == [
            ATTN_DIM, OSNET_MAP_HEIGHT, OSNET_MAP_WIDTH] and entries[0] in (1, -1):
        return (
            "attn_spatial_restore",
            np.asarray([1, ATTN_DIM, OSNET_MAP_HEIGHT, OSNET_MAP_WIDTH],
                       dtype=np.int64),
            np.asarray([-1, ATTN_DIM, OSNET_MAP_HEIGHT, OSNET_MAP_WIDTH],
                       dtype=np.int64),
        )
    raise RuntimeError(
        f"Unclassifiable OSNet Reshape target for {spec.key} at {node_name}: "
        f"{entries}"
    )


def _osnet_flatten_channels(
    model: onnx.ModelProto,
    spec: ReleasedModel,
) -> int:
    """Channel count of the GAP flatten, from the fc Gemm it feeds."""

    initializers = _initializer_map(model)
    gemm_by_input = {
        node.input[0]: node for node in model.graph.node if node.op_type == "Gemm"
    }
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        gemm = gemm_by_input.get(node.output[0])
        if gemm is None:
            continue
        weight = initializers.get(gemm.input[1])
        if weight is None:
            raise RuntimeError(f"fc Gemm weight is not constant for {spec.key}")
        trans_b = _node_attribute_or_default(gemm, "transB", 0)
        channels = int(weight.dims[1] if trans_b else weight.dims[0])
        if channels != ATTN_DIM:  # the attn token flatten also feeds a Gemm
            return channels
    raise RuntimeError(f"Cannot locate the fc Gemm flatten for {spec.key}")


def _expected_osnet_reshape_counts(spec: ReleasedModel) -> dict[str, int]:
    blocks = spec.attention_blocks
    return {
        "attn_tokens_3d": 2 * blocks,
        "attn_head_split": 3 * blocks,
        "attn_token_flatten": blocks,
        "attn_spatial_restore": blocks,
        "flatten": 1,
    }


def set_reshape_shapes_osnet(
    model: onnx.ModelProto,
    spec: ReleasedModel,
    batch_dimension: int,
) -> None:
    """Rewrite every constant OSNet Reshape, reserving -1 for batch only.

    The plain OSNet graph contains exactly one Reshape (the global-average-
    pooled feature map flattened to ``[batch, channels]``); the `_attn`
    variants add seven classified attention Reshapes per block. All non-batch
    dimensions become explicit; ``-1`` remains only on the leading axis of
    dynamic targets (which for the fused batch*token flatten carries
    ``N*OSNET_TOKEN_COUNT`` elements).
    """

    if batch_dimension not in (1, -1):
        raise ValueError(f"Unsupported Reshape batch dimension: {batch_dimension}")
    initializers = _initializer_map(model)
    reshape_nodes = [node for node in model.graph.node if node.op_type == "Reshape"]
    flatten_channels = _osnet_flatten_channels(model, spec)
    counts: dict[str, int] = dict.fromkeys(OSNET_ATTN_RESHAPE_COUNTS, 0)
    for node in reshape_nodes:
        initializer = initializers.get(node.input[1])
        if initializer is None:
            raise RuntimeError(f"Reshape shape is not constant: {node.name}")
        current = numpy_helper.to_array(initializer)
        klass, fixed, dynamic = _classify_osnet_reshape(
            current, spec, node.name, flatten_channels)
        counts[klass] += 1
        _replace_initializer(
            initializer, dynamic if batch_dimension == -1 else fixed)
    expected = _expected_osnet_reshape_counts(spec)
    if counts != expected:
        raise RuntimeError(
            f"Unexpected OSNet Reshape census for {spec.key}: {counts}, "
            f"expected {expected}"
        )


def validate_reshape_shapes_osnet(
    model: onnx.ModelProto,
    spec: ReleasedModel,
    dynamic_batch: bool,
) -> None:
    """Validate every OSNet Reshape against its canonical classified target."""

    initializers = _initializer_map(model)
    reshape_nodes = [node for node in model.graph.node if node.op_type == "Reshape"]
    flatten_channels = _osnet_flatten_channels(model, spec)
    counts: dict[str, int] = dict.fromkeys(OSNET_ATTN_RESHAPE_COUNTS, 0)
    for node in reshape_nodes:
        initializer = initializers.get(node.input[1])
        if initializer is None:
            raise RuntimeError(f"Reshape shape is not constant: {node.name}")
        actual = numpy_helper.to_array(initializer)
        klass, fixed, dynamic = _classify_osnet_reshape(
            actual, spec, node.name, flatten_channels)
        counts[klass] += 1
        expected = dynamic if dynamic_batch else fixed
        if not np.array_equal(actual, expected):
            raise RuntimeError(
                f"Unexpected Reshape shape for {node.name}: "
                f"{actual.tolist()}, expected {expected.tolist()}"
            )
        if np.any(actual == 0):
            raise RuntimeError(f"Reshape shape contains 0 for {node.name}")
        inferred_axes = np.flatnonzero(actual == -1).tolist()
        expected_inferred_axes = [0] if dynamic_batch else []
        if inferred_axes != expected_inferred_axes:
            raise RuntimeError(
                f"Invalid inferred Reshape axes for {node.name}: {inferred_axes}"
            )
    expected_counts = _expected_osnet_reshape_counts(spec)
    if counts != expected_counts:
        raise RuntimeError(
            f"Unexpected OSNet Reshape census for {spec.key}: {counts}, "
            f"expected {expected_counts}"
        )


def fold_osnet_batchnorm(model: onnx.ModelProto, spec: ReleasedModel) -> None:
    """Fold every remaining BatchNormalization into its producing Gemm.

    Inference-time BatchNorm is the exact per-channel affine transform
    ``y = a*x + b`` with ``a = gamma / sqrt(var + eps)`` and
    ``b = beta - a * mean``. onnxsim fuses Conv+BN pairs but leaves the fc
    ``Gemm -> BatchNormalization`` pair; the affine constants fold into the
    Gemm weights (``W' = diag(a) @ W`` for ``transB=1``) and bias
    (``c' = a*c + b``), removing the node without approximation.
    """

    initializers = _initializer_map(model)
    producers = {output: node for node in model.graph.node for output in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    batchnorm_nodes = [
        node for node in model.graph.node if node.op_type == "BatchNormalization"
    ]
    for node in batchnorm_nodes:
        gemm = producers.get(node.input[0])
        if gemm is None or gemm.op_type != "Gemm":
            raise RuntimeError(
                f"BatchNormalization {node.name} is not fed by a Gemm for {spec.key}"
            )
        if len(consumers.get(gemm.output[0], [])) != 1:
            raise RuntimeError(
                f"Gemm output has multiple consumers; cannot fold {node.name}"
            )
        if _node_attribute_or_default(gemm, "alpha", 1.0) != 1.0 or (
            _node_attribute_or_default(gemm, "beta", 1.0) != 1.0
        ) or _node_attribute_or_default(gemm, "transA", 0) != 0:
            raise RuntimeError(f"Unsupported Gemm attributes for {node.name}")

        tensors = []
        for name in (*node.input[1:], gemm.input[1]):
            initializer = initializers.get(name)
            if initializer is None:
                raise RuntimeError(
                    f"Non-constant BatchNormalization/Gemm input for {node.name}"
                )
            tensors.append(numpy_helper.to_array(initializer).astype(np.float64))
        gamma, beta, mean, variance, weight = tensors
        epsilon = _node_attribute_or_default(node, "epsilon", 1e-5)
        scale = gamma / np.sqrt(variance + epsilon)
        shift = beta - scale * mean

        trans_b = _node_attribute_or_default(gemm, "transB", 0)
        fused_weight = (
            weight * scale[:, None] if trans_b else weight * scale[None, :]
        )
        if len(gemm.input) > 2:
            bias_initializer = initializers.get(gemm.input[2])
            if bias_initializer is None:
                raise RuntimeError(f"Non-constant Gemm bias for {node.name}")
            bias = numpy_helper.to_array(bias_initializer).astype(np.float64)
            fused_bias = scale * bias + shift
            _replace_initializer(bias_initializer, fused_bias.astype(np.float32))
        else:
            bias_name = f"{gemm.name}/folded_bn_bias"
            model.graph.initializer.append(
                numpy_helper.from_array(shift.astype(np.float32), name=bias_name)
            )
            gemm.input.append(bias_name)
        _replace_initializer(
            initializers[gemm.input[1]], fused_weight.astype(np.float32)
        )

        gemm.output[0] = node.output[0]
        model.graph.node.remove(node)

    _remove_unused_initializers(model)
    onnx.checker.check_model(model)


def validate_osnet_structure(
    model: onnx.ModelProto,
    spec: ReleasedModel,
) -> None:
    """Validate the pure-CNN topology of an exported OSNet graph.

    No ViT operations may appear: any rank-5 Transpose (the attention qkv
    signature) is rejected. The graph must keep exactly one flatten Reshape
    and one Gemm (the fc projection with its BatchNorm folded), contain the
    channel-gate/global poolings, and branch the public input into a single
    stem Conv.
    """

    for node in model.graph.node:
        if node.op_type == "BatchNormalization":
            raise RuntimeError(
                f"Unfolded BatchNormalization remains for {spec.key}: {node.name}"
            )
        if node.op_type != "Transpose":
            continue
        if len(_transpose_permutation(node)) >= 5:
            raise RuntimeError(
                f"Unexpected rank-5 Transpose in OSNet graph for {spec.key}: "
                f"{node.name}"
            )

    instance_norm_nodes = [
        node for node in model.graph.node if node.op_type == "InstanceNormalization"
    ]
    if len(instance_norm_nodes) != spec.instance_norm_nodes:
        raise RuntimeError(
            f"Expected {spec.instance_norm_nodes} InstanceNormalization nodes "
            f"for {spec.key}, found {len(instance_norm_nodes)}"
        )

    expected_gemms = 1 + spec.attention_blocks
    gemm_nodes = [node for node in model.graph.node if node.op_type == "Gemm"]
    if len(gemm_nodes) != expected_gemms:
        raise RuntimeError(
            f"Expected {expected_gemms} Gemm nodes for {spec.key}, "
            f"found {len(gemm_nodes)}"
        )
    softmax_nodes = [node for node in model.graph.node if node.op_type == "Softmax"]
    if len(softmax_nodes) != spec.attention_blocks:
        raise RuntimeError(
            f"Expected {spec.attention_blocks} Softmax nodes for {spec.key}, "
            f"found {len(softmax_nodes)}"
        )
    pool_nodes = [
        node for node in model.graph.node if node.op_type == "GlobalAveragePool"
    ]
    if not pool_nodes:
        raise RuntimeError(f"Missing GlobalAveragePool nodes for {spec.key}")

    direct_image_consumers = [
        node for node in model.graph.node if "images" in node.input
    ]
    if len(direct_image_consumers) != 1 or direct_image_consumers[0].op_type != "Conv":
        raise RuntimeError(
            f"Unexpected operations branched from the model input for {spec.key}: "
            f"{[(node.name, node.op_type) for node in direct_image_consumers]}"
        )


def _set_symbolic_batch(value: onnx.ValueInfoProto, symbol: str) -> None:
    dimensions = value.type.tensor_type.shape.dim
    if not dimensions:
        raise RuntimeError(f"Tensor has no shape: {value.name}")
    dimensions[0].ClearField("dim_value")
    dimensions[0].dim_param = symbol


def _remove_unused_initializers(model: onnx.ModelProto) -> None:
    used_names = {name for node in model.graph.node for name in node.input if name}
    retained = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name in used_names
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)


def _finish_dynamic_rewrite(model: onnx.ModelProto, spec: ReleasedModel) -> None:
    """Drop the fixed-batch Expand feeding the L2-normalization Div.

    ``[N, D] / [N, 1]`` broadcasts directly; removing the batch-1 Expand
    avoids constructing another runtime batch shape for the normalization.
    """

    producers = {output: node for node in model.graph.node for output in node.output}
    output_producer = producers.get(model.graph.output[0].name)
    if output_producer is None or output_producer.op_type != "Div":
        raise RuntimeError(f"Cannot find output Div for {spec.key}")
    norm_expand = producers.get(output_producer.input[1])
    if norm_expand is None or norm_expand.op_type != "Expand":
        raise RuntimeError(f"Cannot find output normalization Expand for {spec.key}")
    output_producer.input[1] = norm_expand.input[0]
    model.graph.node.remove(norm_expand)


def convert_fixed_batch_to_n(
    fixed_path: Path,
    output_path: Path,
    spec: ReleasedModel,
) -> None:
    """Convert the simplified batch-1 graph to symbolic batch ``N`` safely."""

    model = onnx.load(str(fixed_path))
    onnx.checker.check_model(model)
    if spec.family == "vit":
        validate_reshape_shapes(model, spec, dynamic_batch=False)
        validate_attention_transposes(model, spec, batch_dimension=1)
    else:
        validate_reshape_shapes_osnet(model, spec, dynamic_batch=False)
        validate_osnet_structure(model, spec)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise RuntimeError(f"Expected one ONNX input and output for {spec.key}")
    if model.graph.input[0].name != "images":
        raise RuntimeError(f"Unexpected ONNX input name for {spec.key}")
    if model.graph.output[0].name != "embeddings":
        raise RuntimeError(f"Unexpected ONNX output name for {spec.key}")

    _set_symbolic_batch(model.graph.input[0], "N")
    _set_symbolic_batch(model.graph.output[0], "N")
    initializers = _initializer_map(model)

    # All non-batch dimensions are known.  Reserve -1 exclusively for the
    # leading dynamic batch axis; zero-copy Reshape dimensions are forbidden.
    if spec.family == "osnet":
        set_reshape_shapes_osnet(model, spec, batch_dimension=-1)
        _finish_dynamic_rewrite(model, spec)
        del model.graph.value_info[:]
        metadata = {item.key: item.value for item in model.metadata_props}
        metadata["input_batch"] = "N"
        onnx.helper.set_model_props(model, metadata)
        _remove_unused_initializers(model)
        model = shape_inference.infer_shapes(
            model,
            strict_mode=True,
            data_prop=True,
        )
        validate_reshape_shapes_osnet(model, spec, dynamic_batch=True)
        validate_osnet_structure(model, spec)
        onnx.checker.check_model(model)
        onnx.save(model, str(output_path))
        onnx.checker.check_model(str(output_path))
        return

    set_reshape_shapes(model, spec, batch_dimension=-1)

    # onnxsim folds cls_token.expand(batch, -1, -1) into a batch-1 initializer.
    # Restore batch broadcasting locally at /backbone/Concat.  An all-1.0
    # [N, 1, 1] tensor multiplied by the constant [1, 1, D] CLS token produces
    # [N, 1, D].  Derive N from the adjacent patch embeddings so no shape-only
    # branch is extrapolated from the public model input.
    backbone_concat_index = next(
        (
            index
            for index, node in enumerate(model.graph.node)
            if node.op_type == "Concat" and node.name == "/backbone/Concat"
        ),
        None,
    )
    if backbone_concat_index is None:
        raise RuntimeError(f"Cannot find backbone Concat for {spec.key}")
    backbone_concat = model.graph.node[backbone_concat_index]
    cls_output_name = backbone_concat.input[0]
    cls_initializer = initializers.get(cls_output_name)
    if cls_initializer is None:
        raise RuntimeError(f"Cannot find folded CLS token for {spec.key}")
    cls_value = numpy_helper.to_array(cls_initializer)
    expected_cls_shape = (1, 1, spec.embedding_dimension)
    if cls_value.shape != expected_cls_shape:
        raise RuntimeError(
            f"Unexpected folded CLS token shape for {spec.key}: {cls_value.shape}"
        )
    cls_source_name = f"{cls_output_name}_batch1"
    cls_initializer.name = cls_source_name
    patch_embeddings_name = backbone_concat.input[1]
    dynamic_prefix = "/backbone/cls_token_batch"
    shape_tail_name = f"{dynamic_prefix}/shape_tail"
    model.graph.initializer.append(
        numpy_helper.from_array(
            np.asarray([1, 1], dtype=np.int64),
            name=shape_tail_name,
        )
    )
    dynamic_nodes = [
        onnx.helper.make_node(
            "Shape",
            [patch_embeddings_name],
            [f"{dynamic_prefix}/batch_shape"],
            name=f"{dynamic_prefix}/Shape",
            start=0,
            end=1,
        ),
        onnx.helper.make_node(
            "Concat",
            [f"{dynamic_prefix}/batch_shape", shape_tail_name],
            [f"{dynamic_prefix}/cls_shape"],
            name=f"{dynamic_prefix}/Concat",
            axis=0,
        ),
        onnx.helper.make_node(
            "ConstantOfShape",
            [f"{dynamic_prefix}/cls_shape"],
            [f"{dynamic_prefix}/ones"],
            name=f"{dynamic_prefix}/ConstantOfShape",
            value=numpy_helper.from_array(np.asarray([1.0], dtype=np.float32)),
        ),
        onnx.helper.make_node(
            "Mul",
            [cls_source_name, f"{dynamic_prefix}/ones"],
            [cls_output_name],
            name=f"{dynamic_prefix}/Mul",
        ),
    ]
    for offset, node in enumerate(dynamic_nodes):
        model.graph.node.insert(backbone_concat_index + offset, node)

    _finish_dynamic_rewrite(model, spec)

    del model.graph.value_info[:]
    metadata = {item.key: item.value for item in model.metadata_props}
    metadata["input_batch"] = "N"
    onnx.helper.set_model_props(model, metadata)
    _remove_unused_initializers(model)
    model = shape_inference.infer_shapes(
        model,
        strict_mode=True,
        data_prop=True,
    )
    validate_reshape_shapes(model, spec, dynamic_batch=True)
    validate_dynamic_cls_broadcast(model, spec)
    validate_attention_transposes(model, spec, batch_dimension="N")
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    onnx.checker.check_model(str(output_path))


def validate_graph(
    graph_path: Path,
    spec: ReleasedModel,
    dynamic_batch: bool,
) -> None:
    onnx.checker.check_model(str(graph_path))
    graph = onnx.load(str(graph_path), load_external_data=False)
    nonstandard_domains = sorted(
        {node.domain for node in graph.graph.node if node.domain not in ("", "ai.onnx")}
    )
    if nonstandard_domains:
        raise RuntimeError(
            f"ONNX graph contains non-standard domains: {nonstandard_domains}"
        )

    input_tensor = graph.graph.input[0]
    input_dimensions = input_tensor.type.tensor_type.shape.dim
    if len(input_dimensions) != 4:
        raise RuntimeError(f"Unexpected ONNX input rank for {spec.key}")
    if dynamic_batch:
        if input_dimensions[0].dim_param != "N":
            raise RuntimeError(
                f"The ONNX input batch axis is not N for {spec.key}: "
                f"{input_dimensions[0]}"
            )
    else:
        if input_dimensions[0].dim_value != 1:
            raise RuntimeError(
                f"The ONNX batch axis is not fixed at 1 for {spec.key}: "
                f"{input_dimensions[0]}"
            )
    static_input_shape = [
        input_dimensions[index].dim_value for index in range(1, len(input_dimensions))
    ]
    if static_input_shape != [3, IMAGE_HEIGHT, IMAGE_WIDTH]:
        raise RuntimeError(
            f"Unexpected ONNX input shape for {spec.key}: {static_input_shape}"
        )

    output_tensor = graph.graph.output[0]
    output_dimensions = output_tensor.type.tensor_type.shape.dim
    if len(output_dimensions) != 2:
        raise RuntimeError(f"Unexpected ONNX output rank for {spec.key}")
    if dynamic_batch:
        if output_dimensions[0].dim_param != "N":
            raise RuntimeError(
                f"The ONNX output batch axis is not N for {spec.key}: "
                f"{output_dimensions[0]}"
            )
    elif output_dimensions[0].dim_value != 1:
        raise RuntimeError(
            f"The ONNX output batch axis is not fixed at 1 for {spec.key}"
        )
    if output_dimensions[1].dim_value != spec.embedding_dimension:
        raise RuntimeError(
            f"Unexpected embedding dimension for {spec.key}: "
            f"{output_dimensions[1].dim_value}"
        )
    instance_norm_nodes = [
        node for node in graph.graph.node if node.op_type == "InstanceNormalization"
    ]
    if len(instance_norm_nodes) != spec.instance_norm_nodes:
        raise RuntimeError(
            f"Expected {spec.instance_norm_nodes} InstanceNormalization nodes "
            f"for {spec.key}, found {len(instance_norm_nodes)}"
        )
    if spec.family == "vit":
        validate_reshape_shapes(graph, spec, dynamic_batch=dynamic_batch)
        validate_attention_transposes(
            graph,
            spec,
            batch_dimension="N" if dynamic_batch else 1,
        )
        if dynamic_batch:
            validate_dynamic_value_info(graph, spec)
            validate_dynamic_cls_broadcast(graph, spec)
    else:
        validate_reshape_shapes_osnet(graph, spec, dynamic_batch=dynamic_batch)
        validate_osnet_structure(graph, spec)
        if dynamic_batch:
            validate_dynamic_value_info(graph, spec)


def verify_with_onnxruntime(
    wrapper: nn.Module,
    graph_path: Path,
    device: torch.device,
    seed: int,
    rtol: float,
    atol: float,
    dynamic_batch: bool,
) -> float:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(graph_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    max_absolute_error = 0.0
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_sizes = (1, 2) if dynamic_batch else (1,)
    for batch_size in batch_sizes:
        cpu_input = torch.randn(
            batch_size,
            3,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            generator=generator,
            dtype=torch.float32,
        )
        model_input = cpu_input.to(device)
        with torch.inference_mode():
            reference = wrapper(model_input).cpu().numpy()
        actual = session.run(
            ["embeddings"],
            {"images": cpu_input.numpy()},
        )[0]

        if not np.allclose(reference, actual, rtol=rtol, atol=atol):
            absolute_difference = np.abs(reference - actual)
            raise RuntimeError(
                f"ONNX Runtime output mismatch for batch {batch_size}: "
                f"max_abs={absolute_difference.max():.8g}"
            )
        absolute_difference = np.abs(reference - actual)
        norms = np.linalg.norm(actual, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5):
            raise RuntimeError(
                f"Exported embeddings are not L2 normalized: norms={norms.tolist()}"
            )
        max_absolute_error = max(
            max_absolute_error,
            float(absolute_difference.max()),
        )

    del session
    return max_absolute_error


def export_model(
    spec: ReleasedModel,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (args.output_dir / spec.output).resolve()
    dynamic_output_path = output_path.with_name(
        f"{output_path.stem}_n{output_path.suffix}"
    )
    fixed_needs_export = args.force or not output_path.exists()
    dynamic_needs_export = args.force or not dynamic_output_path.exists()
    if not fixed_needs_export and not dynamic_needs_export:
        print(f"[skip] {output_path.name}, {dynamic_output_path.name}")
        return output_path, dynamic_output_path

    checkpoint_path = resolve_checkpoint(
        spec,
        args.checkpoint_root,
        args.cache_dir,
        args.revision,
    )
    print(f"[load] {spec.key}: {checkpoint_path}")
    model = build_inference_model(spec, checkpoint_path)
    model.to(device)
    wrapper = ReIDExportWrapper(model, l2_normalize=True).to(device).eval()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    images = torch.randn(
        1,
        3,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        generator=generator,
        dtype=torch.float32,
    ).to(device)

    if fixed_needs_export:
        temporary_path = output_path.with_name(f".{output_path.name}.tmp.onnx")
        if temporary_path.exists():
            temporary_path.unlink()
        started = time.perf_counter()
        max_absolute_error: float | None = None
        try:
            with torch.inference_mode():
                torch.onnx.export(
                    wrapper,
                    (images,),
                    str(temporary_path),
                    input_names=["images"],
                    output_names=["embeddings"],
                    dynamic_axes=None,
                    opset_version=args.opset,
                    do_constant_folding=True,
                    external_data=True,
                    dynamo=False,
                )

            onnx.checker.check_model(str(temporary_path))
            graph = simplify_graph(onnx.load(str(temporary_path)), spec)
            if spec.family == "vit":
                set_reshape_shapes(graph, spec, batch_dimension=1)
            else:
                fold_osnet_batchnorm(graph, spec)
                set_reshape_shapes_osnet(graph, spec, batch_dimension=1)
            set_model_metadata(
                graph,
                spec,
                checkpoint_path,
                l2_normalize=True,
                dynamic_batch=False,
            )
            onnx.checker.check_model(graph)
            onnx.save(graph, str(temporary_path))

            validate_graph(temporary_path, spec, dynamic_batch=False)
            if not args.skip_runtime_check:
                max_absolute_error = verify_with_onnxruntime(
                    wrapper,
                    temporary_path,
                    device,
                    args.seed + 1,
                    args.rtol,
                    args.atol,
                    dynamic_batch=False,
                )
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        elapsed_seconds = time.perf_counter() - started
        validation_text = (
            ""
            if max_absolute_error is None
            else f", max_abs_error={max_absolute_error:.3e}"
        )
        print(
            f"[done] {spec.key}: {output_path.name} "
            f"({output_path.stat().st_size / (1024**2):.1f} MiB, "
            f"{elapsed_seconds:.1f}s{validation_text})"
        )
    else:
        print(f"[keep] {output_path.name}")

    if dynamic_needs_export:
        dynamic_temporary_path = dynamic_output_path.with_name(
            f".{dynamic_output_path.name}.tmp.onnx"
        )
        if dynamic_temporary_path.exists():
            dynamic_temporary_path.unlink()
        started = time.perf_counter()
        max_absolute_error = None
        try:
            convert_fixed_batch_to_n(output_path, dynamic_temporary_path, spec)

            # The N-batch rewrite changes Reshape targets and inserts the local
            # CLS broadcast.  Run onnxsim once more on the completed dynamic
            # graph, then perform all structural and numerical validation.
            dynamic_graph = simplify_graph(
                onnx.load(str(dynamic_temporary_path)),
                spec,
            )
            dynamic_graph = materialize_dynamic_value_info(
                dynamic_graph,
                onnx.load(str(output_path)),
                spec,
            )
            onnx.save(dynamic_graph, str(dynamic_temporary_path))
            print(
                f"[onnxsim] {spec.key}: finalized N-batch graph and "
                f"materialized N-shaped ValueInfo"
            )
            validate_graph(dynamic_temporary_path, spec, dynamic_batch=True)
            if not args.skip_runtime_check:
                max_absolute_error = verify_with_onnxruntime(
                    wrapper,
                    dynamic_temporary_path,
                    device,
                    args.seed + 1,
                    args.rtol,
                    args.atol,
                    dynamic_batch=True,
                )
            os.replace(dynamic_temporary_path, dynamic_output_path)
        finally:
            if dynamic_temporary_path.exists():
                dynamic_temporary_path.unlink()

        elapsed_seconds = time.perf_counter() - started
        validation_text = (
            ""
            if max_absolute_error is None
            else f", max_abs_error={max_absolute_error:.3e}"
        )
        print(
            f"[done] {spec.key}: {dynamic_output_path.name} "
            f"({dynamic_output_path.stat().st_size / (1024**2):.1f} MiB, "
            f"{elapsed_seconds:.1f}s{validation_text})"
        )
    else:
        print(f"[keep] {dynamic_output_path.name}")

    del wrapper, model, images
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output_path, dynamic_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "all",
            "unified",
            *RELEASED_MODEL_BY_KEY,
            *UNIFIED_OSNET_MODEL_BY_KEY,
            *AIN_AUG_MODEL_BY_KEY,
        ],
        default=["all"],
        help=(
            "model keys to export; 'all' selects the eight released ViT models, "
            "'unified' selects every unified-dataset OSNet tier with a trained "
            "checkpoint, t/n/p/f/a select individual OSNet tiers, and "
            "b-ain-aug/s-ain-aug/p-ain-aug select the -ain-aug deployment models"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "onnx",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help=(
            "optional local root containing the Hugging Face checkpoint paths; "
            "when omitted, checkpoints are downloaded with huggingface_hub"
        ),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--revision", default=HF_REVISION)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="PyTorch device used during export and reference inference",
    )
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--skip-runtime-check", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "all" in args.models:
        if len(args.models) != 1:
            raise ValueError("'all' cannot be combined with individual model keys")
        selected_models = RELEASED_MODELS
    elif "unified" in args.models:
        if len(args.models) != 1:
            raise ValueError("'unified' cannot be combined with individual model keys")
        available = []
        for spec in UNIFIED_OSNET_MODELS:
            try:
                resolve_checkpoint(spec, args.checkpoint_root, None, args.revision)
            except FileNotFoundError:
                print(f"[skip missing] tier '{spec.key}' has no trained checkpoint")
                continue
            available.append(spec)
        if not available:
            raise FileNotFoundError("No trained unified OSNet checkpoints were found")
        selected_models = tuple(available)
    else:
        combined = {
            **RELEASED_MODEL_BY_KEY,
            **UNIFIED_OSNET_MODEL_BY_KEY,
            **AIN_AUG_MODEL_BY_KEY,
        }
        selected_models = tuple(combined[key] for key in args.models)

    args.output_dir = args.output_dir.resolve()
    if args.checkpoint_root is not None:
        args.checkpoint_root = args.checkpoint_root.resolve()
    if args.cache_dir is not None:
        args.cache_dir = args.cache_dir.resolve()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export was requested, but CUDA is not available")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    output_pairs = [export_model(spec, args, device) for spec in selected_models]
    print(
        f"Prepared {len(output_pairs)} fixed/dynamic model pair(s) "
        f"in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
