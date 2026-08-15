#!/usr/bin/env python3
"""Measure real-data cross-camera positive similarity thresholds for synthetic QA."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from generate_synth_reid33 import PipelineError, _onnx_embeddings
from synth_reid33_core import atomic_write_json, load_config, sha256_file, utc_now


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "transreid_pytorch" / "configs" / "synth_reid33.yml"
DEFAULT_DATA_ROOT = REPO_ROOT / "transreid_pytorch" / "data" / "reid"
UNIFIED_NAME_RE = re.compile(r"p(?P<pid>\d+)_d(?P<domain>\d+)_c(?P<camera>\d+)_(?P<sequence>\d+)")


def collect_protocol_records(data_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect query identities and their gallery images, excluding gallery-only distractors."""
    records: list[dict[str, Any]] = []
    for split in ("query", "gallery"):
        directory = data_root / split
        if not directory.is_dir():
            raise PipelineError(f"real-data calibration split is missing: {directory}")
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            match = UNIFIED_NAME_RE.fullmatch(path.stem)
            if match is None:
                raise PipelineError(f"unexpected unified ReID filename: {path.name}")
            values = {key: int(value) for key, value in match.groupdict().items()}
            records.append({"path": path, "split": split, **values})

    query_pids = {int(row["pid"]) for row in records if row["split"] == "query"}
    selected = [row for row in records if int(row["pid"]) in query_pids]
    by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_pid[int(row["pid"])].append(row)
    invalid = [pid for pid, rows in by_pid.items() if len({int(row["camera"]) for row in rows}) < 2]
    if invalid:
        raise PipelineError(
            f"{len(invalid)} query identities have no cross-camera positive; first PIDs: {invalid[:10]}"
        )
    gallery_only = {int(row["pid"]) for row in records} - query_pids
    metadata = {
        "all_query_gallery_images": len(records),
        "selected_images": len(selected),
        "query_identities": len(query_pids),
        "excluded_gallery_only_identities": len(gallery_only),
        "excluded_gallery_only_images": len(records) - len(selected),
        "images_by_domain": dict(sorted(Counter(f"d{int(row['domain']):02d}" for row in selected).items())),
    }
    return selected, metadata


def cross_camera_positive_values(
    features: Any, records: Sequence[Mapping[str, Any]]
) -> tuple[Any, dict[str, Any]]:
    """Return every unordered same-PID, different-camera cosine similarity."""
    import numpy as np

    if len(features) != len(records):
        raise ValueError("feature and record counts differ")
    by_pid_camera: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    pid_domain: dict[int, int] = {}
    for index, row in enumerate(records):
        pid = int(row["pid"])
        domain = int(row["domain"])
        if pid in pid_domain and pid_domain[pid] != domain:
            raise ValueError(f"PID {pid} occurs in more than one domain")
        pid_domain[pid] = domain
        by_pid_camera[pid][int(row["camera"])].append(index)

    values: list[Any] = []
    by_domain: dict[str, list[Any]] = defaultdict(list)
    for pid in sorted(by_pid_camera):
        cameras = by_pid_camera[pid]
        camera_ids = sorted(cameras)
        for left_position, left_camera in enumerate(camera_ids):
            left = features[np.asarray(cameras[left_camera], dtype=np.int64)]
            for right_camera in camera_ids[left_position + 1 :]:
                right = features[np.asarray(cameras[right_camera], dtype=np.int64)]
                pair_values = (left @ right.T).reshape(-1).astype(np.float32)
                values.append(pair_values)
                by_domain[f"d{pid_domain[pid]:02d}"].append(pair_values)
    if not values:
        raise PipelineError("real-data calibration produced no cross-camera positive pairs")
    combined = np.concatenate(values)
    domain_values = {domain: np.concatenate(chunks) for domain, chunks in sorted(by_domain.items())}
    return combined, domain_values


def distribution_summary(values: Any) -> dict[str, Any]:
    import numpy as np

    return {
        "pair_count": int(len(values)),
        "p01": round(float(np.quantile(values, 0.01, method="linear")), 7),
        "p05": round(float(np.quantile(values, 0.05, method="linear")), 7),
        "p50": round(float(np.quantile(values, 0.50, method="linear")), 7),
        "p95": round(float(np.quantile(values, 0.95, method="linear")), 7),
        "mean": round(float(np.mean(values)), 7),
        "std": round(float(np.std(values)), 7),
    }


def dataset_index_sha256(data_root: Path, records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in records:
        path = Path(row["path"])
        stat = path.stat()
        relative = path.relative_to(data_root)
        digest.update(f"{relative.as_posix()}\0{stat.st_size}\n".encode())
    return digest.hexdigest()


def calibrate(
    data_root: Path,
    config_path: Path,
    output_path: Path | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    config = load_config(config_path)
    records, dataset_metadata = collect_protocol_records(data_root)
    paths = [Path(row["path"]) for row in records]
    if output_path is None:
        output_path = (config_path.parent / config["qa"]["real_similarity_reference"]).resolve()

    result: dict[str, Any] = {
        "schema_version": "synth-reid33-real-similarity/v1",
        "created_at": utc_now(),
        "dataset": {
            "root": str(data_root),
            "splits": ["query", "gallery"],
            "identity_policy": "PIDs occurring in query; gallery-only distractors excluded",
            "pair_policy": "all unordered same-PID pairs with different global camera IDs",
            "index_sha256": dataset_index_sha256(data_root, records),
            **dataset_metadata,
        },
        "preprocessing": {
            "size": [128, 256],
            "color": "RGB",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "feature_normalization": "L2",
            "cosine_pair_quantile_method": "numpy linear",
        },
    }
    for model_key, config_key in (("vit", "vit_onnx"), ("osnet", "osnet_onnx")):
        model_path = (config_path.parent / config["qa"][config_key]).resolve()
        print(f"{model_key}: embedding {len(paths)} real evaluation images with {model_path.name}")
        features = _onnx_embeddings(
            model_path, paths, batch_size=batch_size, progress_label=f"real calibration {model_key}"
        )
        values, by_domain = cross_camera_positive_values(features, records)
        summary = distribution_summary(values)
        result[model_key] = {
            "cross_camera_positive_p05": summary["p05"],
            "distribution": summary,
            "by_domain": {domain: distribution_summary(domain_values) for domain, domain_values in by_domain.items()},
            "model_path": str(model_path.relative_to(REPO_ROOT)),
            "model_sha256": sha256_file(model_path),
            "embedding_dimension": int(features.shape[1]),
        }
        print(
            f"{model_key}: p05={summary['p05']:.7f}, pairs={summary['pair_count']}",
            flush=True,
        )
    atomic_write_json(output_path, result)
    print(json.dumps({
        "output": str(output_path),
        "images": len(records),
        "pairs": result["vit"]["distribution"]["pair_count"],
        "vit_p05": result["vit"]["cross_camera_positive_p05"],
        "osnet_p05": result["osnet"]["cross_camera_positive_p05"],
    }, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    try:
        calibrate(
            args.data_root.resolve(), args.config.resolve(),
            args.output.resolve() if args.output else None, args.batch_size,
        )
        return 0
    except (PipelineError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
