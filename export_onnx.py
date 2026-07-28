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
PIXEL_MEAN = (0.5, 0.5, 0.5)
PIXEL_STD = (0.5, 0.5, 0.5)


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
            "source_repository": HF_REPO_ID,
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
            if dimension.dim_param and dimension.dim_param != "N":
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
                    if fixed_dimension != 1:
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
        else:
            raise RuntimeError(
                f"Cannot prove unknown dimensions are batch-derived for "
                f"{value.name}: {observed_shape}"
            )

        for axis in unknown_axes:
            dimensions[axis].dim_param = "N"
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


def _expected_reshape_shape(
    node: onnx.NodeProto,
    spec: ReleasedModel,
    batch_dimension: int,
) -> np.ndarray:
    expected_heads = 6 if spec.architecture == "ViT-S/16" else 12
    if node.name == "/backbone/patch_embed/Reshape":
        shape = [batch_dimension, spec.embedding_dimension, PATCH_TOKENS]
    elif node.name.endswith("/attn/Reshape"):
        shape = [batch_dimension, TOKEN_COUNT, 3, expected_heads, 64]
    elif node.name.endswith("/attn/Reshape_1"):
        shape = [batch_dimension, TOKEN_COUNT, spec.embedding_dimension]
    else:
        raise RuntimeError(f"Unexpected Reshape node for {spec.key}: {node.name}")
    return np.asarray(shape, dtype=np.int64)


def set_reshape_shapes(
    model: onnx.ModelProto,
    spec: ReleasedModel,
    batch_dimension: int,
) -> None:
    """Set explicit static dimensions, reserving -1 for dynamic batch only."""

    if batch_dimension not in (1, -1):
        raise ValueError(f"Unsupported Reshape batch dimension: {batch_dimension}")
    initializers = _initializer_map(model)
    reshape_nodes = [node for node in model.graph.node if node.op_type == "Reshape"]
    if len(reshape_nodes) != 25:
        raise RuntimeError(
            f"Expected 25 Reshape nodes for {spec.key}, found {len(reshape_nodes)}"
        )

    expected_by_initializer: dict[str, np.ndarray] = {}
    for node in reshape_nodes:
        shape_name = node.input[1]
        initializer = initializers.get(shape_name)
        if initializer is None:
            raise RuntimeError(f"Reshape shape is not constant: {node.name}")
        expected = _expected_reshape_shape(node, spec, batch_dimension)
        previous = expected_by_initializer.get(shape_name)
        if previous is not None and not np.array_equal(previous, expected):
            raise RuntimeError(
                f"Reshape initializer {shape_name!r} has conflicting uses"
            )
        current = numpy_helper.to_array(initializer)
        if current.ndim != 1 or current.size != expected.size:
            raise RuntimeError(f"Unexpected Reshape rank for {node.name}: {current}")
        expected_by_initializer[shape_name] = expected

    if len(expected_by_initializer) != 3:
        raise RuntimeError(
            f"Expected 3 shared Reshape shapes for {spec.key}, "
            f"found {len(expected_by_initializer)}"
        )
    for shape_name, expected in expected_by_initializer.items():
        _replace_initializer(initializers[shape_name], expected)


def validate_reshape_shapes(
    model: onnx.ModelProto,
    spec: ReleasedModel,
    dynamic_batch: bool,
) -> None:
    """Reject zero dimensions and non-leading inferred Reshape dimensions."""

    expected_batch = -1 if dynamic_batch else 1
    initializers = _initializer_map(model)
    reshape_nodes = [node for node in model.graph.node if node.op_type == "Reshape"]
    if len(reshape_nodes) != 25:
        raise RuntimeError(
            f"Expected 25 Reshape nodes for {spec.key}, found {len(reshape_nodes)}"
        )
    shape_names = set()
    for node in reshape_nodes:
        shape_name = node.input[1]
        shape_names.add(shape_name)
        initializer = initializers.get(shape_name)
        if initializer is None:
            raise RuntimeError(f"Reshape shape is not constant: {node.name}")
        actual = numpy_helper.to_array(initializer)
        expected = _expected_reshape_shape(node, spec, expected_batch)
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
    if len(shape_names) != 3:
        raise RuntimeError(
            f"Expected 3 shared Reshape shapes for {spec.key}, found {len(shape_names)}"
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


def convert_fixed_batch_to_n(
    fixed_path: Path,
    output_path: Path,
    spec: ReleasedModel,
) -> None:
    """Convert the simplified batch-1 graph to symbolic batch ``N`` safely."""

    model = onnx.load(str(fixed_path))
    onnx.checker.check_model(model)
    validate_reshape_shapes(model, spec, dynamic_batch=False)
    validate_attention_transposes(model, spec, batch_dimension=1)
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

    # [N, D] / [N, 1] broadcasts directly.  Removing the fixed [1, D] Expand
    # avoids constructing another runtime batch shape for L2 normalization.
    producers = {output: node for node in model.graph.node for output in node.output}
    output_producer = producers.get(model.graph.output[0].name)
    if output_producer is None or output_producer.op_type != "Div":
        raise RuntimeError(f"Cannot find output Div for {spec.key}")
    norm_expand = producers.get(output_producer.input[1])
    if norm_expand is None or norm_expand.op_type != "Expand":
        raise RuntimeError(f"Cannot find output normalization Expand for {spec.key}")
    output_producer.input[1] = norm_expand.input[0]
    model.graph.node.remove(norm_expand)

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
    validate_reshape_shapes(graph, spec, dynamic_batch=dynamic_batch)
    validate_attention_transposes(
        graph,
        spec,
        batch_dimension="N" if dynamic_batch else 1,
    )
    if dynamic_batch:
        validate_dynamic_value_info(graph, spec)
        validate_dynamic_cls_broadcast(graph, spec)


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
            set_reshape_shapes(graph, spec, batch_dimension=1)
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
        choices=["all", *RELEASED_MODEL_BY_KEY],
        default=["all"],
        help="released model keys to export; defaults to all eight models",
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
    else:
        selected_models = tuple(RELEASED_MODEL_BY_KEY[key] for key in args.models)

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
