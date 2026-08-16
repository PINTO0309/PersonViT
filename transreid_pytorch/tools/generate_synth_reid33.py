#!/usr/bin/env python3
"""Build and operate the gated SyntheticReID33 image-generation pipeline.

The command never submits the 20,000-image run until a completed Low-only pilot
report has been approved with a USD ceiling.

Examples (run from ``transreid_pytorch``)::

    python tools/generate_synth_reid33.py pilot --dry-run
    python tools/generate_synth_reid33.py pilot --resume
    python tools/generate_synth_reid33.py report
    python tools/generate_synth_reid33.py rotation-pilot --dry-run
    python tools/generate_synth_reid33.py rotation-pilot --resume
    python tools/generate_synth_reid33.py repair-local --scope rotation-pilot --dry-run
    python tools/generate_synth_reid33.py report-failures --scope rotation-pilot
    python tools/generate_synth_reid33.py qa --scope rotation-pilot
    python tools/generate_synth_reid33.py approve --quality low --max-usd 500
    python tools/generate_synth_reid33.py full --resume
"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

from synth_reid33_core import (
    PROCESSING_VERSION,
    SCHEMA_VERSION,
    SYNTHETIC_RELEASE_FILENAME_VERSION,
    SYNTHETIC_RELEASE_PID_SCOPE,
    aggregate_usage,
    allocate_identity_cameras,
    allocate_pilot_cameras,
    anchor_view_prompt,
    approval_payload,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    classify_batch_item,
    config_sha256,
    identity_prompt,
    load_config,
    make_batch_line,
    make_cameras,
    make_identities,
    make_pilot_samples,
    make_rotation_pilot_samples,
    make_samples,
    plate_prompt,
    process_image,
    perceptual_hash,
    hamming_hex,
    read_jsonl,
    reconcile_batch_output,
    response_usage,
    sample_prompt,
    sha256_bytes,
    sha256_file,
    synthetic_release_filename,
    synthetic_release_sequence,
    usage_cost_usd,
    utc_now,
    validate_pilot,
    validate_plan,
    validate_rotation_pilot,
    verify_approval,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE.parent / "configs" / "synth_reid33.yml"
DEFAULT_ROOT = HERE.parent / "data" / "SyntheticReID33"
ASSET_ORIENTATIONS = ("front", "left", "right", "back")
FORECAST_KEY = "forecast_total_usd_with_configured_retry_reserve"
REFERENCE_PROCESSING_VERSION = "synth-reid33-reference-native-quarter-v1"
WAIVABLE_APPROVAL_GATES = frozenset({
    "pilot.osnet_embedding",
    "rotation_pilot.osnet_embedding",
    "rotation_pilot.manual_review",
})
FULL_LIVE_BATCH_REQUESTS = 100
MAX_ENQUEUED_FULL_REQUESTS = 400
_BATCH_CAPACITY_ERROR_CODES = frozenset({"token_limit_exceeded"})


class PipelineError(RuntimeError):
    pass


def _parse_image_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (AttributeError, TypeError, ValueError) as exc:
        raise PipelineError(f"invalid image size: {value!r}") from exc
    if width <= 0 or height <= 0:
        raise PipelineError(f"invalid image size: {value!r}")
    return width, height


def _reference_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    model = config["model"]
    return {
        "profile": model["reference_profile"],
        "anchor_size": model["reference_anchor_size"],
        "plate_size": model["reference_plate_size"],
        "resampling": "lanczos",
        "jpeg_quality": int(model["reference_jpeg_quality"]),
        "jpeg_subsampling": int(model["reference_jpeg_subsampling"]),
        "processing_version": REFERENCE_PROCESSING_VERSION,
    }


def _write_reference_asset(
    source: Path,
    destination: Path,
    kind: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the exact low-resolution JPEG that will be uploaded as a reference."""
    if kind not in {"anchor", "plate"}:
        raise PipelineError(f"unsupported reference asset kind: {kind}")
    model = config["model"]
    size_key = "reference_anchor_size" if kind == "anchor" else "reference_plate_size"
    expected_source_key = "anchor_size" if kind == "anchor" else "sample_size"
    target_size = _parse_image_size(str(model[size_key]))
    expected_source_size = _parse_image_size(str(model[expected_source_key]))
    try:
        with Image.open(source) as opened:
            opened.load()
            source_size = opened.size
            if source_size != expected_source_size:
                raise PipelineError(
                    f"{kind} source size is {source_size}, expected {expected_source_size}: {source}"
                )
            resized = opened.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            resized.save(
                buffer,
                format="JPEG",
                quality=int(model["reference_jpeg_quality"]),
                subsampling=int(model["reference_jpeg_subsampling"]),
            )
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(f"cannot decode reference source {source}") from exc
    atomic_write_bytes(destination, buffer.getvalue())
    return {
        **_reference_metadata(config),
        "source_size": list(source_size),
        "reference_size": list(target_size),
        "reference_sha256": sha256_file(destination),
    }


def _repair_side_anchor_from_mirror(
    client: "OpenAIBatchClient",
    root: Path,
    config: Mapping[str, Any],
    jobs: Sequence[dict[str, Any]],
    assets: dict[str, dict[str, Any]],
) -> int:
    """Recover an unexecuted left/right anchor from its generated opposite view."""
    repaired = 0
    opposite = {"left": "right", "right": "left"}
    for job in jobs:
        orientation = job.get("orientation")
        if (
            job.get("status") != "needs_revision"
            or job.get("kind") != "anchor"
            or orientation not in opposite
            or job.get("asset_id") in assets
        ):
            continue
        source_id = _asset_id("anchor", int(job["local_pid"]), opposite[str(orientation)])
        source = assets.get(source_id)
        if source is None:
            continue
        source_path = root / source["path"]
        destination = root / "assets" / f"{job['asset_id']}.jpg"
        reference = root / "assets" / "references" / f"{job['asset_id']}.jpg"
        try:
            with Image.open(source_path) as opened:
                opened.load()
                mirrored = opened.convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                buffer = io.BytesIO()
                mirrored.save(
                    buffer,
                    format="JPEG",
                    quality=int(config["model"]["output_compression"]),
                    subsampling=int(config["model"]["reference_jpeg_subsampling"]),
                )
        except Exception as exc:
            raise PipelineError(f"cannot mirror side anchor {source_id}") from exc
        atomic_write_bytes(destination, buffer.getvalue())
        reference_info = _write_reference_asset(destination, reference, "anchor", config)
        file_id = client.upload(reference, purpose="vision")
        assets[str(job["asset_id"])] = {
            "asset_id": job["asset_id"],
            "kind": "anchor",
            "local_pid": job.get("local_pid"),
            "local_camera": None,
            "camera_geometry": None,
            "orientation": orientation,
            "path": str(destination.relative_to(root)),
            "sha256": sha256_file(destination),
            "reference_path": str(reference.relative_to(root)),
            **reference_info,
            "file_id": file_id,
            "model_id": config["model"]["api_id"],
            "catalog_snapshot": config["model"]["catalog_snapshot"],
            "response_model": source.get("response_model"),
            "custom_id": job["custom_id"],
            "completion_mode": "local_horizontal_mirror",
            "source_asset_id": source_id,
            "source_asset_sha256": source["sha256"],
            "source_reference_sha256": source.get("reference_sha256"),
            "repaired_at": utc_now(),
        }
        job["status"] = "succeeded"
        job["completion_mode"] = "local_horizontal_mirror"
        job["source_asset_id"] = source_id
        job["resolved_from_status"] = "needs_revision"
        repaired += 1
    return repaired


class OpenAIBatchClient:
    """Small adapter around the OpenAI SDK, instantiated only for live calls."""

    def __init__(self) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise PipelineError("OPENAI_API_KEY is not set; no API request was made")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise PipelineError("install the 'synth' optional dependencies first") from exc
        self.client = OpenAI()

    def verify_model(self, model_id: str) -> None:
        try:
            model = self.client.models.retrieve(model_id)
        except Exception as exc:
            raise PipelineError(
                f"required Batch model {model_id} is unavailable to this project; "
                "check Organization Verification and model access"
            ) from exc
        if str(getattr(model, "id", "")) != model_id:
            raise PipelineError(f"model lookup did not return the required model ID {model_id}")

    def upload(self, path: Path, purpose: str) -> str:
        with path.open("rb") as handle:
            result = self.client.files.create(file=handle, purpose=purpose)
        return str(result.id)

    def submit(self, input_file_id: str, endpoint: str, metadata: Mapping[str, str]) -> Any:
        return self.client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window="24h",
            metadata=dict(metadata),
        )

    def retrieve(self, batch_id: str) -> Any:
        return self.client.batches.retrieve(batch_id)

    def content(self, file_id: str) -> bytes:
        response = self.client.files.content(file_id)
        if hasattr(response, "read"):
            return response.read()
        if hasattr(response, "content"):
            return bytes(response.content)
        return bytes(response)


def _object_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return json.loads(value.json())


def _terminal_batch_job_status(batch_status: str) -> str:
    """Map a terminal Batch status without retrying invalid input files."""
    if batch_status == "expired":
        return "expired_failed"
    if batch_status == "failed":
        return "batch_validation_failed"
    if batch_status == "cancelled":
        return "cancelled_failed"
    raise ValueError(f"Batch status is not terminal: {batch_status}")


def _is_batch_capacity_failure(batch: Mapping[str, Any]) -> bool:
    if batch.get("status") != "failed":
        return False
    remote = batch.get("remote") or {}
    request_counts = remote.get("request_counts") or {}
    errors = (remote.get("errors") or {}).get("data") or []
    codes = {str(error.get("code", "")) for error in errors}
    return (
        int(request_counts.get("total") or 0) == 0
        and bool(codes)
        and codes <= _BATCH_CAPACITY_ERROR_CODES
    )


def _batch_outstanding_requests(batch: Mapping[str, Any]) -> int:
    """Return requests whose token quota stays reserved until the Batch is terminal."""
    if batch.get("status") not in {"validating", "in_progress", "finalizing"}:
        return 0
    counts = (batch.get("remote") or {}).get("request_counts") or {}
    total = counts.get("total")
    if total is None:
        return len(batch.get("custom_ids") or ())
    return int(total)


def _requeue_capacity_failure(
    batch: dict[str, Any], jobs_by_id: Mapping[str, dict[str, Any]]
) -> bool:
    """Requeue a Batch rejected before execution without creating an API retry attempt."""
    if not _is_batch_capacity_failure(batch) or batch.get("capacity_requeued"):
        return False
    for custom_id in batch["custom_ids"]:
        job = jobs_by_id[custom_id]
        if job.get("batch_id") != batch.get("batch_id"):
            continue
        if job["status"] not in {"submitted", "batch_validation_failed"}:
            continue
        job["status"] = "planned"
        job.pop("batch_id", None)
        job["capacity_requeues"] = int(job.get("capacity_requeues", 0)) + 1
        job["last_capacity_failure_batch_id"] = batch["batch_id"]
    batch["capacity_requeued"] = True
    batch["capacity_requeued_at"] = utc_now()
    return True


def _batch_failure_summaries(batches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for batch in batches:
        if (
            batch.get("status") not in {"failed", "expired", "cancelled"}
            or batch.get("capacity_requeued")
        ):
            continue
        remote = batch.get("remote") or {}
        errors = (remote.get("errors") or {}).get("data") or []
        unique: dict[tuple[str, str, str], int] = Counter()
        for error in errors:
            key = (
                str(error.get("code", "unknown")),
                str(error.get("param", "")),
                str(error.get("message", "")),
            )
            unique[key] += 1
        summaries.append({
            "batch_id": batch.get("batch_id"),
            "status": batch.get("status"),
            "errors": [
                {"code": code, "param": param, "message": message, "count": count}
                for (code, param, message), count in unique.items()
            ],
        })
    return summaries


def _paths(root: Path) -> dict[str, Path]:
    return {
        "state": root / "state",
        "identities": root / "state" / "identities.jsonl",
        "cameras": root / "state" / "cameras.jsonl",
        "samples": root / "state" / "samples.jsonl",
        "pilot_samples": root / "state" / "pilot_samples.jsonl",
        "rotation_pilot_samples": root / "state" / "rotation_pilot_samples.jsonl",
        "plan_report": root / "state" / "plan_validation.json",
        "jobs": root / "state" / "jobs.jsonl",
        "batches": root / "state" / "batches.jsonl",
        "assets": root / "state" / "assets.jsonl",
        "attempts": root / "state" / "attempts.jsonl",
        "approval": root / "state" / "approval.json",
        "report": root / "pilot" / "report.json",
        "pilot_manifest": root / "pilot" / "manifest.jsonl",
        "rotation_pilot_manifest": root / "rotation_pilot" / "manifest.jsonl",
        "rotation_pilot_report": root / "rotation_pilot" / "report.json",
        "manifest": root / "manifest.jsonl",
    }


def initialize(root: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    paths = _paths(root)
    root.mkdir(parents=True, exist_ok=True)
    paths["state"].mkdir(parents=True, exist_ok=True)
    marker = paths["state"] / "config.lock.json"
    digest = config_sha256(config)
    if marker.exists():
        locked = json.loads(marker.read_text(encoding="utf-8"))
        if locked.get("config_sha256") != digest:
            raise PipelineError("config differs from initialized state; use a new output root")
    else:
        atomic_write_json(marker, {
            "schema_version": SCHEMA_VERSION,
            "config_sha256": digest,
            "model_id": config["model"]["api_id"],
            "catalog_snapshot": config["model"]["catalog_snapshot"],
            "created_at": utc_now(),
        })
    if not paths["identities"].exists():
        identities = make_identities(config)
        cameras = make_cameras(config)
        allocation, targets = allocate_identity_cameras(config)
        samples = make_samples(config, allocation)
        pilot_samples = make_pilot_samples(config, allocate_pilot_cameras(config))
        rotation_pilot_samples = make_rotation_pilot_samples(
            config, allocate_pilot_cameras(config)
        )
        validation = validate_plan(config, identities, cameras, samples)
        pilot_validation = validate_pilot(pilot_samples)
        rotation_pilot_validation = validate_rotation_pilot(config, rotation_pilot_samples)
        if not all((
            validation["valid"],
            pilot_validation["valid"],
            rotation_pilot_validation["valid"],
        )):
            raise PipelineError(
                "deterministic plan failed validation: "
                f"{validation} {pilot_validation} {rotation_pilot_validation}"
            )
        atomic_write_jsonl(paths["identities"], identities)
        atomic_write_jsonl(paths["cameras"], cameras)
        atomic_write_jsonl(paths["samples"], samples)
        atomic_write_jsonl(paths["pilot_samples"], pilot_samples)
        atomic_write_jsonl(paths["rotation_pilot_samples"], rotation_pilot_samples)
        atomic_write_json(paths["plan_report"], {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "plan": validation,
            "pilot": pilot_validation,
            "rotation_pilot": rotation_pilot_validation,
            "camera_block_targets": targets,
        })
    return paths


def _asset_id(kind: str, number: int, orientation: str | None = None) -> str:
    if kind == "plate":
        return f"plate-c{number:02d}"
    return f"anchor-p{number:03d}-{orientation}"


def _asset_jobs(
    config: Mapping[str, Any],
    identities: Sequence[Mapping[str, Any]],
    cameras: Sequence[Mapping[str, Any]],
    pids: Iterable[int],
    include_plates: bool,
) -> list[dict[str, Any]]:
    model = config["model"]
    jobs: list[dict[str, Any]] = []
    for pid in pids:
        asset_id = _asset_id("anchor", pid, "front")
        jobs.append({
            "custom_id": f"asset-{asset_id}-a1",
            "scope": "asset",
            "kind": "anchor",
            "asset_id": asset_id,
            "local_pid": pid,
            "orientation": "front",
            "attempt": 1,
            "endpoint": "/v1/images/generations",
            "logical_refs": [],
            "body": {
                "model": model["api_id"],
                "prompt": identity_prompt(identities[pid], config),
                "n": 1,
                "size": model["anchor_size"],
                "quality": model["quality"],
                "output_format": model["output_format"],
                "output_compression": model["output_compression"],
            },
            "status": "planned",
        })
    if include_plates:
        for camera in cameras:
            asset_id = _asset_id("plate", int(camera["local_camera"]))
            jobs.append({
                "custom_id": f"asset-{asset_id}-a1",
                "scope": "asset",
                "kind": "plate",
                "asset_id": asset_id,
                "local_camera": camera["local_camera"],
                "camera_geometry": camera["geometry"],
                "attempt": 1,
                "endpoint": "/v1/images/generations",
                "logical_refs": [],
                "body": {
                    "model": model["api_id"],
                    "prompt": plate_prompt(camera, config),
                    "n": 1,
                    "size": model["sample_size"],
                    "quality": model["quality"],
                    "output_format": model["output_format"],
                    "output_compression": model["output_compression"],
                },
                "status": "planned",
            })
    for pid in pids:
        front = _asset_id("anchor", pid, "front")
        for orientation in ASSET_ORIENTATIONS[1:]:
            asset_id = _asset_id("anchor", pid, orientation)
            jobs.append({
                "custom_id": f"asset-{asset_id}-a1",
                "scope": "asset",
                "kind": "anchor",
                "asset_id": asset_id,
                "local_pid": pid,
                "orientation": orientation,
                "attempt": 1,
                "endpoint": "/v1/images/edits",
                "logical_refs": [front],
                "body": {
                    "model": model["api_id"],
                    "prompt": anchor_view_prompt(orientation, config),
                    "images": [{"file_id": f"ref:{front}"}],
                    "n": 1,
                    "size": model["anchor_size"],
                    "quality": model["quality"],
                    "output_format": model["output_format"],
                    "output_compression": model["output_compression"],
                },
                "status": "blocked_on_refs",
            })
    return jobs


def _sample_jobs(
    config: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    identities: Sequence[Mapping[str, Any]],
    cameras: Sequence[Mapping[str, Any]],
    scope: str,
    quality: str,
) -> list[dict[str, Any]]:
    model = config["model"]
    configured_quality = str(model["quality"])
    if quality != configured_quality:
        raise PipelineError(f"sample quality must be {configured_quality}")
    jobs = []
    for sample in samples:
        if scope != "full" and sample.get("quality") != configured_quality:
            raise PipelineError(f"{scope} sample quality must be {configured_quality}")
        pid = int(sample["local_pid"])
        camera_id = int(sample["local_camera"])
        anchor = _asset_id(
            "anchor", pid, str(sample.get("anchor_orientation", sample["orientation"]))
        )
        plate = _asset_id("plate", camera_id)
        jobs.append({
            "custom_id": f"{scope}-{sample['sample_id']}-a1",
            "scope": scope,
            "kind": "sample",
            "sample_id": sample["sample_id"],
            "attempt": 1,
            "quality": configured_quality,
            "endpoint": "/v1/images/edits",
            "logical_refs": [anchor, plate],
            "body": {
                "model": model["api_id"],
                "prompt": sample_prompt(sample, identities[pid], cameras[camera_id], config),
                "images": [{"file_id": f"ref:{anchor}"}, {"file_id": f"ref:{plate}"}],
                "n": 1,
                "size": model["sample_size"],
                "quality": configured_quality,
                "output_format": model["output_format"],
                "output_compression": model["output_compression"],
            },
            "status": "blocked_on_refs",
        })
    return jobs


def _merge_jobs(path: Path, additions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    by_id = {row["custom_id"]: row for row in rows}
    for row in additions:
        by_id.setdefault(row["custom_id"], dict(row))
    merged = sorted(by_id.values(), key=lambda row: row["custom_id"])
    atomic_write_jsonl(path, merged)
    return merged


def plan_scope(root: Path, config: Mapping[str, Any], scope: str) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    paths = initialize(root, config)
    identities = read_jsonl(paths["identities"])
    cameras = read_jsonl(paths["cameras"])
    if scope == "pilot":
        additions = _asset_jobs(config, identities, cameras, range(12), include_plates=True)
        additions += _sample_jobs(
            config,
            read_jsonl(paths["pilot_samples"]),
            identities,
            cameras,
            "pilot",
            quality=str(config["model"]["quality"]),
        )
    elif scope == "rotation_pilot":
        additions = _sample_jobs(
            config,
            read_jsonl(paths["rotation_pilot_samples"]),
            identities,
            cameras,
            "rotation_pilot",
            quality=str(config["model"]["quality"]),
        )
    elif scope == "full":
        if not paths["approval"].exists():
            raise PipelineError("full generation is locked: run report and approve first")
        approval = json.loads(paths["approval"].read_text(encoding="utf-8"))
        verify_approval(approval, config)
        if not paths["report"].exists():
            raise PipelineError("the report referenced by approval is missing")
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        if approval.get("report_sha256") != sha256_bytes(canonical_json(report)):
            raise PipelineError("pilot report changed after approval; review and approve it again")
        additions = _asset_jobs(config, identities, cameras, range(12, 500), include_plates=False)
        additions += _sample_jobs(
            config, read_jsonl(paths["samples"]), identities, cameras, "full", approval["quality"]
        )
    else:
        raise ValueError(scope)
    return _merge_jobs(paths["jobs"], additions), paths


def _materialize_body(job: Mapping[str, Any], assets: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    refs = job.get("logical_refs") or []
    if any(ref not in assets or not assets[ref].get("file_id") for ref in refs):
        return None
    body = json.loads(json.dumps(job["body"]))
    if refs:
        body["images"] = [{"file_id": assets[ref]["file_id"]} for ref in refs]
    return body


def write_dry_run_requests(
    root: Path, scope: str, jobs: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[Path]:
    request_dir = root / "requests" / "dry-run" / scope
    request_dir.mkdir(parents=True, exist_ok=True)
    for stale in request_dir.glob("*.jsonl"):
        stale.unlink()
    included_scopes = {scope} if scope == "rotation_pilot" else {scope, "asset"}
    selected = [row for row in jobs if row["scope"] in included_scopes]
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        stage = "generation" if not row["logical_refs"] else ("sample" if row["kind"] == "sample" else "views")
        groups[(row["endpoint"], stage)].append(row)
    written = []
    chunk_size = int(config["batch"]["requests_per_sample_batch"])
    for (endpoint, stage), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: row["custom_id"])
        for chunk_index in range(0, len(rows), chunk_size):
            chunk = rows[chunk_index : chunk_index + chunk_size]
            path = request_dir / f"{stage}-{endpoint.rsplit('/', 1)[-1]}-{chunk_index // chunk_size:03d}.jsonl"
            atomic_write_jsonl(path, [make_batch_line(row["custom_id"], endpoint, row["body"]) for row in chunk])
            written.append(path)
    return written


def _refresh_and_collect(
    client: OpenAIBatchClient,
    root: Path,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    batches = read_jsonl(paths["batches"])
    jobs = read_jsonl(paths["jobs"])
    jobs_by_id = {row["custom_id"]: row for row in jobs}
    assets = {row["asset_id"]: row for row in read_jsonl(paths["assets"])}
    attempts = read_jsonl(paths["attempts"])
    pilot_specs = {row["sample_id"]: row for row in read_jsonl(paths["pilot_samples"])}
    rotation_pilot_specs = {
        row["sample_id"]: row for row in read_jsonl(paths["rotation_pilot_samples"])
    }
    full_specs = {row["sample_id"]: row for row in read_jsonl(paths["samples"])}
    specs_by_scope = {
        "pilot": pilot_specs,
        "rotation_pilot": rotation_pilot_specs,
        "full": full_specs,
    }
    cameras = read_jsonl(paths["cameras"])
    changed = False
    for batch in batches:
        if batch["status"] in {"failed", "expired", "cancelled"}:
            if _is_batch_capacity_failure(batch):
                changed = _requeue_capacity_failure(batch, jobs_by_id) or changed
                continue
            job_status = _terminal_batch_job_status(batch["status"])
            for custom_id in batch["custom_ids"]:
                if jobs_by_id[custom_id]["status"] == "submitted":
                    jobs_by_id[custom_id]["status"] = job_status
                    changed = True
            continue
        if batch["status"] == "completed" and batch.get("collected"):
            continue
        remote = _object_dict(client.retrieve(batch["batch_id"]))
        batch["status"] = remote.get("status", batch["status"])
        batch["remote"] = {key: remote.get(key) for key in (
            "request_counts", "output_file_id", "error_file_id", "errors",
            "completed_at", "failed_at", "expired_at", "expires_at",
        )}
        changed = True
        if batch["status"] != "completed":
            if batch["status"] in {"failed", "expired", "cancelled"}:
                if _is_batch_capacity_failure(batch):
                    changed = _requeue_capacity_failure(batch, jobs_by_id) or changed
                else:
                    job_status = _terminal_batch_job_status(batch["status"])
                    for custom_id in batch["custom_ids"]:
                        if jobs_by_id[custom_id]["status"] == "submitted":
                            jobs_by_id[custom_id]["status"] = job_status
            continue
        payload_parts = []
        if remote.get("output_file_id"):
            payload_parts.append(client.content(str(remote["output_file_id"])).rstrip(b"\n"))
        if remote.get("error_file_id"):
            payload_parts.append(client.content(str(remote["error_file_id"])).rstrip(b"\n"))
        payload = b"\n".join(part for part in payload_parts if part) + b"\n"
        result_path = root / "responses" / f"{batch['batch_id']}.jsonl"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        from synth_reid33_core import atomic_write_bytes

        atomic_write_bytes(result_path, payload)
        found, missing = reconcile_batch_output(payload, batch["custom_ids"])
        for custom_id in missing:
            jobs_by_id[custom_id]["status"] = "retryable_failed"
        for custom_id, item in found.items():
            job = jobs_by_id[custom_id]
            classification = classify_batch_item(item)
            usage = response_usage(item)
            attempt_row = {
                "custom_id": custom_id,
                "scope": job["scope"],
                "sample_id": job.get("sample_id"),
                "asset_id": job.get("asset_id"),
                "attempt": job["attempt"],
                "quality": job.get("quality") or job["body"].get("quality"),
                "classification": classification,
                "usage": usage,
                "estimated_cost_usd": usage_cost_usd(usage, config),
                "batch_id": batch["batch_id"],
                "request_id": (item.get("response") or {}).get("request_id"),
                "recorded_at": utc_now(),
                "model_id": config["model"]["api_id"],
                "catalog_snapshot": config["model"]["catalog_snapshot"],
            }
            attempts.append(attempt_row)
            if classification != "succeeded":
                job["status"] = "needs_revision" if classification == "revise_once" else (
                    "retryable_failed" if classification == "retryable" else "terminal_failed"
                )
                continue
            body = (item.get("response") or {}).get("body") or {}
            attempt_row["response_model"] = body.get("model")
            data = body.get("data") or []
            if not data or not data[0].get("b64_json"):
                job["status"] = "planned"
                continue
            image_bytes = base64.b64decode(data[0]["b64_json"], validate=True)
            if job["kind"] in {"anchor", "plate"}:
                raw = root / "assets" / f"{job['asset_id']}.jpg"
                raw.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes(raw, image_bytes)
                reference = root / "assets" / "references" / f"{job['asset_id']}.jpg"
                reference_info = _write_reference_asset(
                    raw, reference, str(job["kind"]), config
                )
                file_id = client.upload(reference, purpose="vision")
                assets[job["asset_id"]] = {
                    "asset_id": job["asset_id"],
                    "kind": job["kind"],
                    "local_pid": job.get("local_pid"),
                    "local_camera": job.get("local_camera"),
                    "camera_geometry": job.get("camera_geometry"),
                    "orientation": job.get("orientation"),
                    "path": str(raw.relative_to(root)),
                    "sha256": sha256_file(raw),
                    "reference_path": str(reference.relative_to(root)),
                    **reference_info,
                    "file_id": file_id,
                    "model_id": config["model"]["api_id"],
                    "catalog_snapshot": config["model"]["catalog_snapshot"],
                    "response_model": body.get("model"),
                    "custom_id": custom_id,
                }
                job["status"] = "succeeded"
            else:
                sample = specs_by_scope[job["scope"]][job["sample_id"]]
                raw = root / "raw" / job["scope"] / f"{job['custom_id']}.jpg"
                raw.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes(raw, image_bytes)
                if job["scope"] == "pilot":
                    final_dir = root / "pilot" / "images"
                    manifest_path = paths["pilot_manifest"]
                    qa_status = "accepted"
                elif job["scope"] == "rotation_pilot":
                    final_dir = root / "rotation_pilot" / "images"
                    manifest_path = paths["rotation_pilot_manifest"]
                    qa_status = "accepted"
                else:
                    final_dir = root / "candidates" / str(sample["split"])
                    manifest_path = paths["manifest"]
                    qa_status = "geometry_passed"
                final = final_dir / f"{sample['sample_id']}.jpg"
                qa = process_image(raw, final, cameras[int(sample["local_camera"])], sample, config)
                attempt_row["qa"] = qa
                if qa["accepted"]:
                    job["status"] = "succeeded"
                    _upsert_manifest(
                        manifest_path,
                        _sample_manifest_row(
                            root, config, sample, job, attempt_row, cameras, assets, qa, final,
                            qa_status=qa_status,
                        ),
                    )
                else:
                    job["status"] = "qa_failed"
        batch["collected"] = True
    if changed:
        atomic_write_jsonl(paths["batches"], batches)
        atomic_write_jsonl(paths["jobs"], sorted(jobs_by_id.values(), key=lambda row: row["custom_id"]))
        atomic_write_jsonl(paths["assets"], sorted(assets.values(), key=lambda row: row["asset_id"]))
        atomic_write_jsonl(paths["attempts"], attempts)


def _upsert_manifest(path: Path, row: Mapping[str, Any]) -> None:
    rows = read_jsonl(path)
    key = (row["sample_id"], row.get("quality")) if row.get("split") == "pilot" else (row["sample_id"], None)
    by_key = {
        ((item["sample_id"], item.get("quality")) if item.get("split") == "pilot" else (item["sample_id"], None)): item
        for item in rows
    }
    by_key[key] = dict(row)
    atomic_write_jsonl(path, sorted(by_key.values(), key=lambda item: (item["sample_id"], item.get("quality", ""))))


def _sample_manifest_row(
    root: Path,
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
    job: Mapping[str, Any],
    attempt: Mapping[str, Any],
    cameras: Sequence[Mapping[str, Any]],
    assets: Mapping[str, Mapping[str, Any]],
    qa: Mapping[str, Any],
    final: Path,
    qa_status: str,
    selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    camera = cameras[int(sample["local_camera"])]
    row = {
        **sample,
        "domain": "d05",
        "site": camera["site"],
        "camera_geometry": camera["geometry"],
        "camera_isp": camera["isp"],
        "camera_processing_version": camera.get("processing_version"),
        "processing_version": qa.get("processing_version"),
        "prompt_version": config["prompt"]["version"],
        "prompt": job["body"]["prompt"],
        "prompt_sha256": sha256_bytes(job["body"]["prompt"].encode()),
        "reference_asset_ids": job["logical_refs"],
        "reference_sha256": [
            assets[ref].get("reference_sha256", assets[ref]["sha256"])
            for ref in job["logical_refs"]
        ],
        "reference_source_sha256": [assets[ref]["sha256"] for ref in job["logical_refs"]],
        "reference_profiles": [
            assets[ref].get("profile", "legacy-original") for ref in job["logical_refs"]
        ],
        "reference_sizes": [
            assets[ref].get("reference_size") for ref in job["logical_refs"]
        ],
        "reference_file_ids": [assets[ref]["file_id"] for ref in job["logical_refs"]],
        "model_id": config["model"]["api_id"],
        "catalog_snapshot": config["model"]["catalog_snapshot"],
        "response_model": attempt.get("response_model"),
        "request_custom_id": job["custom_id"],
        "request_id": attempt.get("request_id"),
        "batch_id": attempt.get("batch_id"),
        "attempt": job["attempt"],
        "quality": job["quality"],
        "usage": attempt.get("usage", {}),
        "estimated_cost_usd": attempt.get("estimated_cost_usd", 0),
        "qa": dict(qa),
        "qa_status": qa_status,
        "final_path": str(final.relative_to(root)),
        "final_sha256": qa["final_sha256"],
    }
    if selection is not None:
        row["selection"] = dict(selection)
    return row


def _prepare_retries(
    jobs: list[dict[str, Any]], config: Mapping[str, Any], identities: Sequence[Mapping[str, Any]],
    cameras: Sequence[Mapping[str, Any]], pilot_specs: Mapping[str, Mapping[str, Any]],
    full_specs: Mapping[str, Mapping[str, Any]],
    rotation_pilot_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return jobs unchanged: all automatic API retry paths are fail-closed."""
    if int(config["batch"]["max_attempts"]) != 1:
        raise PipelineError("automatic retry planner requires batch.max_attempts=1")
    return jobs


def _current_cost(paths: Mapping[str, Path], config: Mapping[str, Any]) -> float:
    return round(sum(float(row.get("estimated_cost_usd", 0)) for row in read_jsonl(paths["attempts"])), 8)


def _scope_cost(paths: Mapping[str, Path], scope: str) -> float:
    return round(sum(
        float(row.get("estimated_cost_usd", 0))
        for row in read_jsonl(paths["attempts"])
        if row.get("scope") == scope
    ), 8)


def enforce_cost_ceiling(current_usd: float, pending_usd: float, next_usd: float, max_usd: float) -> None:
    projected = current_usd + pending_usd + next_usd
    if projected > max_usd:
        raise PipelineError(
            f"cost stop: projected ${projected:.2f} exceeds approved ceiling ${max_usd:.2f}"
        )


def _job_cost_estimates(paths: Mapping[str, Path]) -> tuple[dict[str, float], float]:
    values: dict[str, list[float]] = defaultdict(list)
    all_values = []
    for row in read_jsonl(paths["attempts"]):
        cost = float(row.get("estimated_cost_usd", 0))
        if cost <= 0 or row.get("classification") != "succeeded":
            continue
        key = f"{row.get('scope')}:{row.get('quality')}"
        values[key].append(cost)
        all_values.append(cost)
    estimates = {key: max(costs) for key, costs in values.items()}
    return estimates, max(all_values, default=0.0)


def _live_batch_chunk_size(config: Mapping[str, Any], scope: str) -> int:
    configured = int(config["batch"]["requests_per_sample_batch"])
    return min(configured, FULL_LIVE_BATCH_REQUESTS) if scope == "full" else configured


def _estimated_job_cost(
    job: Mapping[str, Any], cost_estimates: Mapping[str, float], conservative_fallback: float
) -> float:
    if job["kind"] == "sample":
        key = f"pilot:{job.get('quality') or job['body'].get('quality')}"
    else:
        key = f"asset:{job['body'].get('quality')}"
    return float(cost_estimates.get(key, conservative_fallback))


def run_live(root: Path, config: Mapping[str, Any], scope: str) -> None:
    jobs, paths = plan_scope(root, config, scope)
    client = OpenAIBatchClient()
    client.verify_model(str(config["model"]["api_id"]))
    _refresh_and_collect(client, root, config, paths)
    jobs = read_jsonl(paths["jobs"])
    identities = read_jsonl(paths["identities"])
    cameras = read_jsonl(paths["cameras"])
    pilot_specs = {row["sample_id"]: row for row in read_jsonl(paths["pilot_samples"])}
    rotation_pilot_specs = {
        row["sample_id"]: row for row in read_jsonl(paths["rotation_pilot_samples"])
    }
    full_specs = {row["sample_id"]: row for row in read_jsonl(paths["samples"])}
    jobs = _prepare_retries(
        jobs,
        config,
        identities,
        cameras,
        pilot_specs,
        full_specs,
        rotation_pilot_specs,
    )
    assets = {row["asset_id"]: row for row in read_jsonl(paths["assets"])}
    repaired_side_anchors = _repair_side_anchor_from_mirror(
        client, root, config, jobs, assets
    )
    if repaired_side_anchors:
        atomic_write_jsonl(paths["assets"], sorted(assets.values(), key=lambda row: row["asset_id"]))
    for job in jobs:
        if job["status"] == "blocked_on_refs" and _materialize_body(job, assets) is not None:
            job["status"] = "planned"
    eligible = [job for job in jobs if job["status"] == "planned" and job["scope"] in (scope, "asset")]
    if scope == "full":
        approval = json.loads(paths["approval"].read_text(encoding="utf-8"))
        current = _current_cost(paths, config)
        if current >= float(approval["max_usd"]):
            atomic_write_jsonl(paths["jobs"], jobs)
            raise PipelineError(f"cost ceiling reached (${current:.2f}); no batch was submitted")
    elif scope == "rotation_pilot":
        current = _scope_cost(paths, scope)
        maximum = float(config["body_rotation"]["pilot_max_usd"])
        if current >= maximum:
            atomic_write_jsonl(paths["jobs"], jobs)
            raise PipelineError(
                f"body-rotation pilot cost ceiling reached (${current:.2f}); no batch was submitted"
            )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in eligible:
        groups[job["endpoint"]].append(job)
    batches = read_jsonl(paths["batches"])
    chunk_size = _live_batch_chunk_size(config, scope)
    cost_estimates, conservative_fallback = _job_cost_estimates(paths)
    if scope == "rotation_pilot" and "pilot:low" not in cost_estimates:
        raise PipelineError(
            "body-rotation pilot requires measured low-quality pilot usage before submission"
        )
    pending_reserve = sum(
        _estimated_job_cost(job, cost_estimates, conservative_fallback)
        for job in jobs
        if job["status"] == "submitted" and job["scope"] in (scope, "asset")
    )
    outstanding_full_requests = sum(
        _batch_outstanding_requests(batch)
        for batch in batches
        if batch.get("scope") == "full"
    )
    for endpoint, group in sorted(groups.items()):
        group.sort(key=lambda row: row["custom_id"])
        for chunk_no, start in enumerate(range(0, len(group), chunk_size)):
            chunk = group[start : start + chunk_size]
            if (
                scope == "full"
                and outstanding_full_requests + len(chunk) > MAX_ENQUEUED_FULL_REQUESTS
            ):
                break
            request_path = root / "requests" / "live" / f"{scope}-{endpoint.rsplit('/', 1)[-1]}-{utc_now().replace(':', '')}-{chunk_no:03d}.jsonl"
            lines = []
            for job in chunk:
                body = _materialize_body(job, assets)
                if body is None:
                    continue
                lines.append(make_batch_line(job["custom_id"], endpoint, body))
            if not lines:
                continue
            if scope in {"full", "rotation_pilot"}:
                next_estimate = sum(
                    _estimated_job_cost(job, cost_estimates, conservative_fallback)
                    for job in chunk
                )
                maximum = (
                    float(json.loads(paths["approval"].read_text(encoding="utf-8"))["max_usd"])
                    if scope == "full"
                    else float(config["body_rotation"]["pilot_max_usd"])
                )
                current = (
                    _current_cost(paths, config)
                    if scope == "full"
                    else _scope_cost(paths, scope)
                )
                enforce_cost_ceiling(
                    current, pending_reserve, next_estimate, maximum
                )
            atomic_write_jsonl(request_path, lines)
            file_id = client.upload(request_path, purpose="batch")
            remote = _object_dict(client.submit(file_id, endpoint, {
                "pipeline": "SyntheticReID33", "scope": scope, "config": config_sha256(config)[:16]
            }))
            custom_ids = [line["custom_id"] for line in lines]
            batches.append({
                "batch_id": remote["id"],
                "input_file_id": file_id,
                "input_path": str(request_path.relative_to(root)),
                "endpoint": endpoint,
                "scope": scope,
                "custom_ids": custom_ids,
                "status": remote.get("status", "validating"),
                "created_at": utc_now(),
            })
            if scope == "full":
                outstanding_full_requests += len(custom_ids)
            pending_reserve += next_estimate if scope in {"full", "rotation_pilot"} else 0.0
            for custom_id in custom_ids:
                next(job for job in jobs if job["custom_id"] == custom_id)["status"] = "submitted"
                next(job for job in jobs if job["custom_id"] == custom_id)["batch_id"] = remote["id"]
    atomic_write_jsonl(paths["jobs"], sorted(jobs, key=lambda row: row["custom_id"]))
    atomic_write_jsonl(paths["batches"], batches)
    pending = Counter(job["status"] for job in jobs if job["scope"] in (scope, "asset"))
    result = {
        "scope": scope,
        "job_status": pending,
        "current_cost_usd": _current_cost(paths, config),
        "scope_cost_usd": _scope_cost(paths, scope),
    }
    if repaired_side_anchors:
        result["locally_repaired_side_anchors"] = repaired_side_anchors
    failures = _batch_failure_summaries(batches)
    if failures:
        result["batch_failures"] = failures
    print(json.dumps(result, default=dict, indent=2))


def build_report(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    paths = initialize(root, config)
    rows = read_jsonl(paths["pilot_manifest"])
    expected_pilot_images = len(read_jsonl(paths["pilot_samples"]))
    plan_report = json.loads(paths["plan_report"].read_text(encoding="utf-8"))
    all_attempts = read_jsonl(paths["attempts"])
    attempts = [row for row in all_attempts if row.get("scope") in ("pilot", "asset")]
    rotation_attempts = [row for row in all_attempts if row.get("scope") == "rotation_pilot"]
    rotation_report = build_rotation_pilot_report(root, config)
    automatic = {
        "accepted_images": len(rows),
        "expected_images": expected_pilot_images,
        "decode_rate": sum(bool(row.get("qa", {}).get("decode")) for row in rows) / expected_pilot_images,
        "geometry_rate": sum(bool(row.get("qa", {}).get("geometry_pass")) for row in rows) / expected_pilot_images,
        "framing_rate": sum(bool(row.get("qa", {}).get("framing", {}).get("pass")) for row in rows) / expected_pilot_images,
        "sha_unique": len({row.get("final_sha256") for row in rows}) == len(rows),
        "camera_geometry_metadata_valid": bool(
            plan_report.get("plan", {}).get("camera_geometry", {}).get("valid")
        ),
    }
    near_pairs = 0
    comparable_pairs = 0
    by_identity_quality: dict[tuple[int, str], list[str]] = defaultdict(list)
    for row in rows:
        value = row.get("qa", {}).get("phash")
        if value:
            by_identity_quality[(int(row["local_pid"]), str(row["quality"]))].append(value)
    for hashes in by_identity_quality.values():
        for left in range(len(hashes)):
            for right in range(left + 1, len(hashes)):
                comparable_pairs += 1
                near_pairs += hamming_hex(hashes[left], hashes[right]) <= 4
    automatic["phash_near_duplicate_rate"] = near_pairs / comparable_pairs if comparable_pairs else 1.0
    for model_key in ("vit", "osnet"):
        metrics = [row.get("qa", {}).get("embedding", {}).get(model_key, {}) for row in rows]
        valid = [metric for metric in metrics if metric]
        automatic[f"{model_key}_anchor_top1"] = (
            sum(bool(metric.get("anchor_top1")) for metric in valid) / len(valid) if len(valid) == len(rows) and valid else 0.0
        )
        automatic[f"{model_key}_margin_mean"] = (
            sum(float(metric.get("cosine_margin", 0)) for metric in valid) / len(valid) if len(valid) == len(rows) and valid else 0.0
        )
        automatic[f"{model_key}_above_real_p05"] = bool(valid) and len(valid) == len(rows) and all(
            metric.get("above_real_positive_p05") for metric in valid
        )
    manual_path = root / "qa" / "manual_review.json"
    manual = json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.exists() else {
        "complete": False,
        "identity_consistency_rate": 0.0,
        "camera_pitch_consistency_rate": 0.0,
        "camera_geometry_failures": None,
        "major_anatomy_failures": None,
        "text_logo_watermark_failures": None,
        "query_occlusion_valid_rate": 0.0,
    }
    thresholds = config["qa"]
    gates = {
        "image_count": len(rows) == expected_pilot_images,
        "decode": automatic["decode_rate"] >= float(thresholds["decode_rate_min"]),
        "geometry": automatic["geometry_rate"] >= float(thresholds["geometry_rate_min"]),
        "framing": automatic["framing_rate"] >= float(thresholds["geometry_rate_min"]),
        "camera_geometry_metadata": automatic["camera_geometry_metadata_valid"],
        "unique_sha": automatic["sha_unique"],
        "phash_duplicates": automatic["phash_near_duplicate_rate"]
        < float(thresholds["phash_near_duplicate_rate_max"]),
        "vit_embedding": automatic["vit_anchor_top1"] >= float(thresholds["anchor_top1_min"])
        and automatic["vit_margin_mean"] >= float(thresholds["cosine_margin_min"])
        and automatic["vit_above_real_p05"],
        "osnet_embedding": automatic["osnet_anchor_top1"] >= float(thresholds["anchor_top1_min"])
        and automatic["osnet_margin_mean"] >= float(thresholds["cosine_margin_min"])
        and automatic["osnet_above_real_p05"],
        "manual_review": bool(manual.get("complete"))
        and float(manual.get("identity_consistency_rate", 0)) >= float(thresholds["identity_consistency_min"])
        and int(manual.get("major_anatomy_failures") or 0) == 0
        and int(manual.get("text_logo_watermark_failures") or 0) == 0
        and float(manual.get("query_occlusion_valid_rate", 0)) >= 0.98
        and float(manual.get("camera_pitch_consistency_rate", 0))
        >= float(thresholds["camera_pitch_consistency_min"])
        and int(manual.get("camera_geometry_failures") or 0) == 0,
        "body_rotation_pilot": bool(rotation_report.get("rotation_gate_passed")),
    }
    usage = aggregate_usage(row.get("usage", {}) for row in attempts)
    actual_cost = usage_cost_usd(usage, config)
    rotation_usage = aggregate_usage(row.get("usage", {}) for row in rotation_attempts)
    rotation_actual_cost = usage_cost_usd(rotation_usage, config)
    costs_by_quality: dict[str, list[float]] = defaultdict(list)
    for row in attempts:
        if row.get("scope") == "pilot" and row.get("classification") == "succeeded":
            costs_by_quality[str(row.get("quality"))].append(float(row.get("estimated_cost_usd", 0)))
    forecast = {}
    successful_asset_attempts = [
        row for row in attempts
        if row.get("scope") == "asset" and row.get("classification") == "succeeded"
    ]
    asset_costs = [float(row.get("estimated_cost_usd", 0)) for row in successful_asset_attempts]
    successful_assets = {row.get("asset_id") for row in successful_asset_attempts}
    per_asset = sum(asset_costs) / len(asset_costs) if asset_costs else None
    for quality in (str(config["model"]["quality"]),):
        observed = costs_by_quality.get(quality, [])
        if observed and per_asset is not None:
            sample_cost = sum(observed) / len(observed)
            remaining_assets = per_asset * max(0, (500 * 4 + 33) - len(successful_assets))
            full_samples = sample_cost * 20000 * (1 + float(config["batch"]["retry_reserve_fraction"]))
            forecast[quality] = round(
                actual_cost + rotation_actual_cost + remaining_assets + full_samples, 2
            )
        else:
            forecast[quality] = None
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "model_id": config["model"]["api_id"],
        "catalog_snapshot": config["model"]["catalog_snapshot"],
        "reference_inputs": _reference_metadata(config),
        "config_sha256": config_sha256(config),
        "automatic": automatic,
        "manual": manual,
        "gates": gates,
        "pilot_gate_passed": all(gates.values()),
        "usage": usage,
        "pilot_actual_cost_usd": actual_cost,
        "body_rotation_pilot": {
            "gate_passed": rotation_report["rotation_gate_passed"],
            "actual_cost_usd": rotation_actual_cost,
            "report_path": str(paths["rotation_pilot_report"].relative_to(root)),
        },
        FORECAST_KEY: forecast,
        "pricing_snapshot": config["pricing_usd_per_million_tokens"],
    }
    atomic_write_json(paths["report"], report)
    return report


def _onnx_embeddings(
    model_path: Path,
    image_paths: Sequence[Path],
    batch_size: int = 64,
    progress_label: str | None = None,
) -> Any:
    try:
        import cv2
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise PipelineError("opencv, numpy, and onnxruntime are required for embedding QA") from exc
    if not model_path.is_file():
        raise PipelineError(f"embedding QA model is missing: {model_path}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    session = ort.InferenceSession(str(model_path), providers=[p for p in providers if p in available])
    input_name = session.get_inputs()[0].name
    means = np.asarray([0.485, 0.456, 0.406], np.float32)[None, None, :]
    stds = np.asarray([0.229, 0.224, 0.225], np.float32)[None, None, :]
    output = []
    for start in range(0, len(image_paths), batch_size):
        batch = []
        for path in image_paths[start : start + batch_size]:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise PipelineError(f"cannot decode QA image: {path}")
            image = cv2.resize(image, (128, 256), interpolation=cv2.INTER_AREA)[:, :, ::-1] / 255.0
            image = ((image.astype(np.float32) - means) / stds).transpose(2, 0, 1)
            batch.append(image)
        features = session.run(None, {input_name: np.stack(batch)})[0]
        features = features.reshape(features.shape[0], -1).astype(np.float32)
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        output.append(features)
        completed = min(start + batch_size, len(image_paths))
        if progress_label and (completed == len(image_paths) or completed % (batch_size * 20) == 0):
            print(f"{progress_label}: {completed}/{len(image_paths)} images", flush=True)
    return np.concatenate(output, axis=0)


def _write_contact_sheet(
    root: Path, rows: Sequence[Mapping[str, Any]], filename: str = "pilot_contact_sheet.jpg"
) -> Path:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise PipelineError("Pillow is required for the pilot contact sheet") from exc
    thumb_w, thumb_h, label_h = 96, 192, 24
    columns = 8
    ordered = sorted(rows, key=lambda row: (
        row["quality"],
        row["local_pid"],
        row.get("body_yaw_deg", row["local_camera"]),
    ))
    sheet = Image.new("RGB", (columns * thumb_w, ((len(ordered) + columns - 1) // columns) * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(ordered):
        image = Image.open(root / row["final_path"]).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        x = (index % columns) * thumb_w + (thumb_w - image.width) // 2
        y = (index // columns) * (thumb_h + label_h)
        sheet.paste(image, (x, y))
        pitch_down = float(row.get("camera_geometry", {}).get("pitch_down_deg", 0))
        if "body_yaw_deg" in row:
            label = (
                f"p{row['local_pid']:02d} y{int(row['body_yaw_deg']):+04d} "
                f"c{int(row['local_camera']):02d} d{pitch_down:.0f}"
            )
        else:
            label = (
                f"{row['quality'][0]} p{row['local_pid']:02d} "
                f"c{int(row['local_camera']):02d} d{pitch_down:.0f}"
            )
        draw.text((index % columns * thumb_w + 2, y + thumb_h + 3), label, fill="black")
    path = root / "qa" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)
    return path


def _write_camera_plate_contact_sheet(root: Path) -> Path:
    """Render all 33 uncropped plates with their fixed pose metadata for manual QA."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise PipelineError("Pillow is required for the camera plate contact sheet") from exc
    paths = _paths(root)
    assets = {row["asset_id"]: row for row in read_jsonl(paths["assets"])}
    cameras = read_jsonl(paths["cameras"])
    thumb_w, thumb_h, label_h = 192, 384, 34
    columns = 6
    rows = (len(cameras) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, camera in enumerate(cameras):
        camera_id = int(camera["local_camera"])
        asset = assets.get(_asset_id("plate", camera_id))
        if asset is None:
            raise PipelineError(f"missing camera plate asset for camera {camera_id}")
        plate_path = root / str(asset["path"])
        with Image.open(plate_path) as source:
            image = source.convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
        cell_x = (index % columns) * thumb_w
        cell_y = (index // columns) * (thumb_h + label_h)
        x = cell_x + (thumb_w - image.width) // 2
        y = cell_y + (thumb_h - image.height) // 2
        sheet.paste(image, (x, y))
        geometry = camera["geometry"]
        horizon = round(float(geometry["horizon_y_fraction"]) * 100)
        label = (
            f"c{camera_id:02d} down {float(geometry['pitch_down_deg']):.1f}deg "
            f"h {float(geometry['mounting_height_m']):.1f}m y {horizon}%"
        )
        draw.text((cell_x + 3, cell_y + thumb_h + 4), label, fill="black")
    path = root / "qa" / "camera_pitch_contact_sheet.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)
    return path


def _apply_anchor_embedding_qa(
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    rows: list[dict[str, Any]],
    identity_count: int,
) -> list[dict[str, Any]]:
    paths = _paths(root)
    assets = {row["asset_id"]: row for row in read_jsonl(paths["assets"])}
    anchor_paths: list[Path] = []
    anchor_pids: list[int] = []
    for pid in range(identity_count):
        for orientation in ASSET_ORIENTATIONS:
            asset_id = _asset_id("anchor", pid, orientation)
            if asset_id not in assets:
                raise PipelineError(f"missing QA anchor {asset_id}")
            anchor_paths.append(root / assets[asset_id]["path"])
            anchor_pids.append(pid)
    candidate_paths = [root / row["final_path"] for row in rows]
    try:
        import numpy as np
    except ImportError as exc:
        raise PipelineError("numpy is required for embedding QA") from exc
    calibration_path = (config_path.parent / config["qa"]["real_similarity_reference"]).resolve()
    calibration = (
        json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration_path.exists()
        else {}
    )
    anchor_pid_array = np.asarray(anchor_pids)
    for model_key, key in (("vit", "vit_onnx"), ("osnet", "osnet_onnx")):
        model_path = (config_path.parent / config["qa"][key]).resolve()
        anchor_vectors = _onnx_embeddings(model_path, anchor_paths)
        candidate_vectors = _onnx_embeddings(model_path, candidate_paths)
        centroids = []
        for pid in range(identity_count):
            centroid = anchor_vectors[anchor_pid_array == pid].mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            centroids.append(centroid)
        centroids = np.stack(centroids)
        similarities = candidate_vectors @ centroids.T
        p05 = calibration.get(model_key, {}).get("cross_camera_positive_p05")
        for index, row in enumerate(rows):
            pid = int(row["local_pid"])
            own = float(similarities[index, pid])
            impostor = float(np.max(np.delete(similarities[index], pid)))
            row.setdefault("qa", {}).setdefault("embedding", {})[model_key] = {
                "anchor_top1": int(np.argmax(similarities[index])) == pid,
                "same_identity_similarity": round(own, 7),
                "cosine_margin": round(own - impostor, 7),
                "real_positive_p05": p05,
                "above_real_positive_p05": p05 is not None and own >= float(p05),
                "model_sha256": sha256_file(model_path),
            }
    return rows


def run_embedding_qa(root: Path, config: Mapping[str, Any], config_path: Path) -> None:
    paths = initialize(root, config)
    rows = read_jsonl(paths["pilot_manifest"])
    expected = len(read_jsonl(paths["pilot_samples"]))
    if len(rows) != expected:
        raise PipelineError(
            f"embedding QA requires {expected} accepted Low pilot images; found {len(rows)}"
        )
    rows = _apply_anchor_embedding_qa(root, config, config_path, rows, identity_count=12)
    for row in rows:
        row.setdefault("qa", {})["phash"] = row.get("qa", {}).get("phash") or perceptual_hash(root / row["final_path"])
    atomic_write_jsonl(paths["pilot_manifest"], rows)
    contact_sheet = _write_contact_sheet(root, rows)
    camera_sheet = _write_camera_plate_contact_sheet(root)
    template = root / "qa" / "manual_review.template.json"
    if not template.exists():
        atomic_write_json(template, {
            "complete": False,
            "reviewed_contact_sheet": str(contact_sheet.relative_to(root)),
            "reviewed_camera_plate_contact_sheet": str(camera_sheet.relative_to(root)),
            "identity_consistency_rate": 0.0,
            "major_anatomy_failures": 0,
            "text_logo_watermark_failures": 0,
            "query_occlusion_valid_rate": 0.0,
            "camera_pitch_consistency_rate": 0.0,
            "camera_geometry_failures": 0,
            "reviewer_notes": (
                "Copy to manual_review.json only after reviewing both the person sheet and all "
                "33 uncropped camera plates against their labeled pitch, height, and horizon."
            ),
        })
    print(f"embedding QA complete; review {contact_sheet} and {camera_sheet}")


def build_rotation_pilot_report(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    paths = initialize(root, config)
    rows = read_jsonl(paths["rotation_pilot_manifest"])
    expected = int(config["body_rotation"]["pilot_identities"]) * int(
        config["body_rotation"]["pilot_samples_per_identity"]
    )
    plan_validation = validate_rotation_pilot(config, rows)
    automatic = {
        "accepted_images": len(rows),
        "expected_images": expected,
        "decode_rate": sum(bool(row.get("qa", {}).get("decode")) for row in rows) / expected,
        "geometry_rate": sum(
            bool(row.get("qa", {}).get("geometry_pass")) for row in rows
        ) / expected,
        "framing_rate": sum(
            bool(row.get("qa", {}).get("framing", {}).get("pass")) for row in rows
        ) / expected,
        "sha_unique": len({row.get("final_sha256") for row in rows}) == len(rows),
        "planned_yaw_distribution_valid": plan_validation["valid"],
        "body_yaw_counts": plan_validation["body_yaw_counts"],
    }
    comparable_pairs = 0
    near_pairs = 0
    by_pid: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        value = row.get("qa", {}).get("phash")
        if value:
            by_pid[int(row["local_pid"])].append(str(value))
    for hashes in by_pid.values():
        for left in range(len(hashes)):
            for right in range(left + 1, len(hashes)):
                comparable_pairs += 1
                near_pairs += hamming_hex(hashes[left], hashes[right]) <= 4
    automatic["phash_near_duplicate_rate"] = (
        near_pairs / comparable_pairs if comparable_pairs else 1.0
    )
    for model_key in ("vit", "osnet"):
        metrics = [row.get("qa", {}).get("embedding", {}).get(model_key, {}) for row in rows]
        valid = [metric for metric in metrics if metric]
        complete = bool(valid) and len(valid) == len(rows)
        automatic[f"{model_key}_anchor_top1"] = (
            sum(bool(metric.get("anchor_top1")) for metric in valid) / len(valid)
            if complete
            else 0.0
        )
        automatic[f"{model_key}_margin_mean"] = (
            sum(float(metric.get("cosine_margin", 0)) for metric in valid) / len(valid)
            if complete
            else 0.0
        )
        automatic[f"{model_key}_above_real_p05"] = complete and all(
            metric.get("above_real_positive_p05") for metric in valid
        )
    manual_path = root / "rotation_pilot" / "manual_review.json"
    manual = json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.exists() else {
        "complete": False,
        "body_yaw_accuracy_rate": 0.0,
        "identity_consistency_rate": 0.0,
        "camera_pitch_consistency_rate": 0.0,
        "camera_geometry_failures": None,
        "major_anatomy_failures": None,
        "text_logo_watermark_failures": None,
    }
    thresholds = config["qa"]
    gates = {
        "image_count": len(rows) == expected,
        "decode": automatic["decode_rate"] >= float(thresholds["decode_rate_min"]),
        "geometry": automatic["geometry_rate"] >= float(thresholds["geometry_rate_min"]),
        "framing": automatic["framing_rate"] >= float(thresholds["geometry_rate_min"]),
        "yaw_distribution": automatic["planned_yaw_distribution_valid"],
        "unique_sha": automatic["sha_unique"],
        "phash_duplicates": automatic["phash_near_duplicate_rate"]
        < float(thresholds["phash_near_duplicate_rate_max"]),
        "vit_embedding": automatic["vit_anchor_top1"] >= float(thresholds["anchor_top1_min"])
        and automatic["vit_margin_mean"] >= float(thresholds["cosine_margin_min"])
        and automatic["vit_above_real_p05"],
        "osnet_embedding": automatic["osnet_anchor_top1"]
        >= float(thresholds["anchor_top1_min"])
        and automatic["osnet_margin_mean"] >= float(thresholds["cosine_margin_min"])
        and automatic["osnet_above_real_p05"],
        "manual_review": bool(manual.get("complete"))
        and float(manual.get("body_yaw_accuracy_rate", 0))
        >= float(thresholds["body_yaw_accuracy_min"])
        and float(manual.get("identity_consistency_rate", 0))
        >= float(thresholds["identity_consistency_min"])
        and float(manual.get("camera_pitch_consistency_rate", 0))
        >= float(thresholds["camera_pitch_consistency_min"])
        and int(manual.get("camera_geometry_failures") or 0) == 0
        and int(manual.get("major_anatomy_failures") or 0) == 0
        and int(manual.get("text_logo_watermark_failures") or 0) == 0,
    }
    attempts = [
        row for row in read_jsonl(paths["attempts"])
        if row.get("scope") == "rotation_pilot"
    ]
    usage = aggregate_usage(row.get("usage", {}) for row in attempts)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "model_id": config["model"]["api_id"],
        "catalog_snapshot": config["model"]["catalog_snapshot"],
        "reference_inputs": _reference_metadata(config),
        "config_sha256": config_sha256(config),
        "automatic": automatic,
        "manual": manual,
        "gates": gates,
        "rotation_gate_passed": all(gates.values()),
        "usage": usage,
        "actual_cost_usd": usage_cost_usd(usage, config),
    }
    atomic_write_json(paths["rotation_pilot_report"], report)
    return report


def run_rotation_pilot_qa(
    root: Path, config: Mapping[str, Any], config_path: Path
) -> dict[str, Any]:
    paths = initialize(root, config)
    rows = read_jsonl(paths["rotation_pilot_manifest"])
    expected = int(config["body_rotation"]["pilot_identities"]) * int(
        config["body_rotation"]["pilot_samples_per_identity"]
    )
    if len(rows) != expected:
        raise PipelineError(
            f"body-rotation embedding QA requires {expected} accepted images; found {len(rows)}"
        )
    rows = _apply_anchor_embedding_qa(root, config, config_path, rows, identity_count=12)
    for row in rows:
        row.setdefault("qa", {})["phash"] = row.get("qa", {}).get("phash") or perceptual_hash(
            root / row["final_path"]
        )
    atomic_write_jsonl(paths["rotation_pilot_manifest"], rows)
    contact_sheet = _write_contact_sheet(
        root, rows, filename="rotation_pilot_contact_sheet.jpg"
    )
    template = root / "rotation_pilot" / "manual_review.template.json"
    if not template.exists():
        atomic_write_json(template, {
            "complete": False,
            "reviewed_contact_sheet": str(contact_sheet.relative_to(root)),
            "body_yaw_accuracy_rate": 0.0,
            "identity_consistency_rate": 0.0,
            "camera_pitch_consistency_rate": 0.0,
            "camera_geometry_failures": 0,
            "major_anatomy_failures": 0,
            "text_logo_watermark_failures": 0,
            "reviewer_notes": (
                "Copy to rotation_pilot/manual_review.json after checking all 96 labeled yaw "
                "views and their labeled camera pitches."
            ),
        })
    report = build_rotation_pilot_report(root, config)
    print(f"body-rotation QA complete; review {contact_sheet}")
    return report


def _retrieval_metrics(features: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    query_indices = [index for index, row in enumerate(rows) if row["split"] == "query"]
    gallery_indices = [index for index, row in enumerate(rows) if row["split"] == "gallery"]
    aps = []
    rank1 = []
    invalid = 0
    gallery_features = features[gallery_indices]
    for query_index in query_indices:
        query = rows[query_index]
        similarities = gallery_features @ features[query_index]
        keep = np.asarray([
            not (int(rows[index]["local_pid"]) == int(query["local_pid"])
                 and int(rows[index]["local_camera"]) == int(query["local_camera"]))
            for index in gallery_indices
        ])
        order = np.argsort(-similarities[keep])
        kept_indices = np.asarray(gallery_indices)[keep][order]
        matches = np.asarray([
            int(rows[index]["local_pid"]) == int(query["local_pid"]) for index in kept_indices
        ], dtype=np.float32)
        if not matches.any():
            invalid += 1
            continue
        positive_ranks = np.flatnonzero(matches)
        precisions = [float(matches[: rank + 1].sum() / (rank + 1)) for rank in positive_ranks]
        aps.append(sum(precisions) / len(precisions))
        rank1.append(float(matches[0]))
    return {
        "queries": len(query_indices),
        "gallery": len(gallery_indices),
        "valid_queries": len(aps),
        "query_valid_rate": 1.0 if not query_indices else len(aps) / len(query_indices),
        "mAP": sum(aps) / len(aps) if aps else 0.0,
        "rank1": sum(rank1) / len(rank1) if rank1 else 0.0,
        "invalid_queries": invalid,
    }


def _publish_candidate(root: Path, row: dict[str, Any]) -> None:
    source = root / row["final_path"]
    destination = root / "accepted" / str(row["split"]) / synthetic_release_filename(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".publishing")
    temporary.unlink(missing_ok=True)
    row["release_filename_version"] = SYNTHETIC_RELEASE_FILENAME_VERSION
    row["release_pid_scope"] = SYNTHETIC_RELEASE_PID_SCOPE
    row["release_sequence"] = synthetic_release_sequence(row)
    # POSIX rename/replace is permitted to do nothing when source and
    # destination are hard links to the same inode.  On an idempotent QA run,
    # creating another hard link at ``temporary`` and replacing an already
    # published destination would therefore leave ``*.publishing`` behind.
    # Reuse the existing publication directly when it is already the source.
    try:
        already_published = destination.exists() and os.path.samefile(source, destination)
    except OSError:
        already_published = False
    if already_published:
        row["final_path"] = str(destination.relative_to(root))
        return
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    row["final_path"] = str(destination.relative_to(root))


def _full_embedding_acceptance_models(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Keep waived pilot models as advisory metrics during full-dataset QA."""
    approval_path = root / "state" / "approval.json"
    if not approval_path.exists():
        return ("vit", "osnet"), ()
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    waived = set((approval.get("gate_waiver") or {}).get("waived_gates") or ())
    osnet_waived = {
        "pilot.osnet_embedding",
        "rotation_pilot.osnet_embedding",
    }.issubset(waived)
    if osnet_waived:
        return ("vit",), ("osnet",)
    return ("vit", "osnet"), ()


def _full_review_artifacts(
    root: Path,
    accepted: Sequence[Mapping[str, Any]],
    boundary_rows: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[Path], list[Path], Path]:
    review_rows = []
    accepted_by_pid: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in accepted:
        accepted_by_pid[int(row["local_pid"])].append(row)
    for pid in sorted(accepted_by_pid):
        pid_rows = accepted_by_pid[pid]
        for occluded in (False, True):
            candidates = [row for row in pid_rows if bool(row["occluded"]) == occluded]
            if candidates:
                review_rows.append(min(
                    candidates,
                    key=lambda row: sha256_bytes(
                        f"review|{row['local_camera']}|{pid}|{occluded}|{row['sample_id']}".encode()
                    ),
                ))
    review_rows = review_rows[:1000]
    sheet_paths = []
    for chunk_index in range(0, len(review_rows), 200):
        sheet_paths.append(_write_contact_sheet(
            root,
            review_rows[chunk_index : chunk_index + 200],
            filename=f"full_contact_sheet_{chunk_index // 200:02d}.jpg",
        ))
    boundary_sheet_paths = []
    ordered_boundaries = sorted(
        {str(row["sample_id"]): row for row in boundary_rows}.values(),
        key=lambda row: str(row["sample_id"]),
    )
    for chunk_index in range(0, len(ordered_boundaries), 200):
        boundary_sheet_paths.append(_write_contact_sheet(
            root,
            ordered_boundaries[chunk_index : chunk_index + 200],
            filename=f"full_boundary_contact_sheet_{chunk_index // 200:02d}.jpg",
        ))
    manual_template = root / "qa" / "full_review.template.json"
    template = (
        json.loads(manual_template.read_text(encoding="utf-8"))
        if manual_template.exists()
        else {
            "complete": False,
            "sample_fraction": 0.05,
            "reviewed_all_boundary_cases": False,
            "major_anatomy_failures": 0,
            "text_logo_watermark_failures": 0,
            "identity_consistency_rate": 0.0,
            "camera_pitch_consistency_rate": 0.0,
            "camera_geometry_failures": 0,
            "reviewer_notes": (
                "Copy to full_review.json after the stratified 5%, labeled camera pitch, and all "
                "boundary cases are reviewed."
            ),
        }
    )
    template["reviewed_contact_sheets"] = [
        str(path.relative_to(root)) for path in sheet_paths
    ]
    template["reviewed_boundary_contact_sheets"] = [
        str(path.relative_to(root)) for path in boundary_sheet_paths
    ]
    atomic_write_json(manual_template, template)
    return sheet_paths, boundary_sheet_paths, manual_template


def finalize_full_qa_waiver(
    root: Path,
    config: Mapping[str, Any],
    waiver_reason: str,
) -> dict[str, Any]:
    """Publish geometry-passed samples while explicitly recording skipped full QA."""
    reason = waiver_reason.strip()
    if not reason:
        raise PipelineError("a non-empty full QA waiver reason is required")
    paths = initialize(root, config)
    rows = read_jsonl(paths["manifest"])
    if len(rows) != 20_000:
        raise PipelineError(f"full QA waiver requires 20,000 samples; found {len(rows)}")
    invalid = [
        str(row.get("sample_id"))
        for row in rows
        if not row.get("qa", {}).get("accepted")
        or not row.get("qa", {}).get("framing", {}).get("pass")
        or row.get("processing_version") != PROCESSING_VERSION
    ]
    if invalid:
        raise PipelineError(
            f"full QA waiver cannot bypass geometry/processing failures: {invalid[:10]}"
        )
    skipped_checks = [
        "vit_full_embedding",
        "osnet_full_embedding",
        "phash_near_duplicate_comparison",
    ]
    waiver = {
        "authorized": True,
        "authorized_by": "user",
        "recorded_at": utc_now(),
        "reason": reason,
        "skipped_checks": skipped_checks,
        "geometry_qa_required": True,
        "exact_sha_uniqueness_still_required": True,
    }
    for row in rows:
        qa = row.setdefault("qa", {})
        qa["embedding"] = {
            "vit": {"skipped": True, "waived": True},
            "osnet": {"skipped": True, "waived": True},
        }
        qa["embedding_pass"] = None
        qa["phash_duplicate_pass"] = None
        row["qa_status"] = "accepted"
        row["qa_waiver"] = waiver
        _publish_candidate(root, row)
    atomic_write_jsonl(paths["manifest"], sorted(rows, key=lambda row: row["sample_id"]))
    attempts = read_jsonl(paths["attempts"])
    boundary_ids = {
        str(row.get("sample_id"))
        for row in attempts
        if row.get("scope") == "full"
        and row.get("sample_id")
        and (
            row.get("classification") != "succeeded"
            or row.get("qa", {}).get("accepted") is False
        )
    }
    boundary_ids.update(
        str(row["sample_id"])
        for row in rows
        if (row.get("selection") or {}).get("method") == "local_sibling_reprocess"
    )
    boundary_rows = [row for row in rows if str(row["sample_id"]) in boundary_ids]
    sheet_paths, boundary_sheet_paths, manual_template = _full_review_artifacts(
        root, rows, boundary_rows
    )
    report = {
        "created_at": utc_now(),
        "model_id": config["model"]["api_id"],
        "catalog_snapshot": config["model"]["catalog_snapshot"],
        "reference_inputs": _reference_metadata(config),
        "generated": len(rows),
        "accepted": len(rows),
        "rejected": 0,
        "complete": False,
        "automatic_qa_complete": False,
        "release_ready_pending_manual_review": True,
        "phash_near_duplicate_rate": None,
        "retrieval_label_qa": {},
        "embedding_acceptance_models": [],
        "embedding_advisory_models": [],
        "explicit_user_waiver": waiver,
        "review_contact_sheets": [str(path.relative_to(root)) for path in sheet_paths],
        "boundary_case_count": len(boundary_rows),
        "boundary_contact_sheets": [
            str(path.relative_to(root)) for path in boundary_sheet_paths
        ],
        "manual_review_template": str(manual_template.relative_to(root)),
    }
    atomic_write_json(root / "qa" / "full_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def record_full_manual_review_waiver(
    root: Path,
    config: Mapping[str, Any],
    waiver_reason: str,
) -> dict[str, Any]:
    """Record an explicit user waiver without fabricating human review results."""
    reason = waiver_reason.strip()
    if not reason:
        raise PipelineError("a non-empty manual review waiver reason is required")
    paths = initialize(root, config)
    rows = read_jsonl(paths["manifest"])
    if len(rows) != 20_000 or any(row.get("qa_status") != "accepted" for row in rows):
        raise PipelineError("manual review waiver requires exactly 20,000 accepted samples")
    report_path = root / "qa" / "full_report.json"
    if not report_path.exists():
        raise PipelineError("manual review waiver requires full_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if int(report.get("accepted", 0)) != 20_000 or int(report.get("rejected", -1)) != 0:
        raise PipelineError("manual review waiver requires 20,000 release candidates")
    waiver = {
        "authorized": True,
        "authorized_by": "user",
        "recorded_at": utc_now(),
        "reason": reason,
        "skipped_checks": [
            "stratified_5_percent_manual_review",
            "all_boundary_case_manual_review",
        ],
        "review_results_fabricated": False,
    }
    review = {
        "complete": False,
        "skipped": True,
        "sample_fraction": 0.0,
        "reviewed_contact_sheets": [],
        "reviewed_boundary_contact_sheets": [],
        "reviewed_all_boundary_cases": False,
        "identity_consistency_rate": None,
        "camera_pitch_consistency_rate": None,
        "camera_geometry_failures": None,
        "major_anatomy_failures": None,
        "text_logo_watermark_failures": None,
        "reviewer_notes": "Manual review was explicitly skipped; no human result is claimed.",
        "explicit_user_waiver": waiver,
    }
    review_path = root / "qa" / "full_review.json"
    atomic_write_json(review_path, review)
    report["manual_review_complete"] = False
    report["manual_review_skipped"] = True
    report["manual_review_user_waiver"] = waiver
    report["release_ready_for_integration"] = True
    atomic_write_json(report_path, report)
    print(json.dumps(review, ensure_ascii=False, indent=2))
    return review


def run_full_embedding_qa(root: Path, config: Mapping[str, Any], config_path: Path) -> None:
    paths = initialize(root, config)
    rows = read_jsonl(paths["manifest"])
    if len(rows) != 20000:
        raise PipelineError(f"full QA requires all 20,000 generated samples; found {len(rows)}")
    assets = {row["asset_id"]: row for row in read_jsonl(paths["assets"])}
    anchor_paths = []
    anchor_pids = []
    for pid in range(500):
        for orientation in ASSET_ORIENTATIONS:
            asset_id = _asset_id("anchor", pid, orientation)
            if asset_id not in assets:
                raise PipelineError(f"missing QA anchor {asset_id}")
            anchor_paths.append(root / assets[asset_id]["path"])
            anchor_pids.append(pid)
    candidate_paths = [root / row["final_path"] for row in rows]
    try:
        import numpy as np
    except ImportError as exc:
        raise PipelineError("numpy is required for full embedding QA") from exc
    calibration_path = (config_path.parent / config["qa"]["real_similarity_reference"]).resolve()
    if not calibration_path.exists():
        raise PipelineError(f"real-data similarity calibration is missing: {calibration_path}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    retrieval = {}
    for model_key, key in (("vit", "vit_onnx"), ("osnet", "osnet_onnx")):
        p05 = calibration.get(model_key, {}).get("cross_camera_positive_p05")
        if p05 is None:
            raise PipelineError(f"calibration has no {model_key}.cross_camera_positive_p05")
        model_path = (config_path.parent / config["qa"][key]).resolve()
        anchor_vectors = _onnx_embeddings(model_path, anchor_paths)
        candidate_vectors = _onnx_embeddings(model_path, candidate_paths)
        centroids = []
        anchor_pid_array = np.asarray(anchor_pids)
        for pid in range(500):
            centroid = anchor_vectors[anchor_pid_array == pid].mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            centroids.append(centroid)
        centroids = np.stack(centroids)
        similarities = candidate_vectors @ centroids.T
        model_sha256 = sha256_file(model_path)
        for index, row in enumerate(rows):
            pid = int(row["local_pid"])
            own = float(similarities[index, pid])
            impostor = float(np.max(np.delete(similarities[index], pid)))
            row.setdefault("qa", {}).setdefault("embedding", {})[model_key] = {
                "anchor_top1": int(np.argmax(similarities[index])) == pid,
                "same_identity_similarity": round(own, 7),
                "cosine_margin": round(own - impostor, 7),
                "real_positive_p05": float(p05),
                "above_real_positive_p05": own >= float(p05),
                "model_sha256": model_sha256,
            }
        retrieval[model_key] = _retrieval_metrics(candidate_vectors, rows)

    duplicate_ids: set[str] = set()
    by_pid: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        row.setdefault("qa", {})["phash"] = row.get("qa", {}).get("phash") or perceptual_hash(
            root / row["final_path"]
        )
        by_pid[int(row["local_pid"])].append(row)
    comparable_pairs = 0
    near_pairs = 0
    for pid_rows in by_pid.values():
        ordered = sorted(pid_rows, key=lambda row: row["sample_id"])
        for left in range(len(ordered)):
            for right in range(left + 1, len(ordered)):
                comparable_pairs += 1
                if hamming_hex(ordered[left]["qa"]["phash"], ordered[right]["qa"]["phash"]) <= 4:
                    near_pairs += 1
                    duplicate_ids.add(str(ordered[right]["sample_id"]))

    thresholds = config["qa"]
    acceptance_models, advisory_models = _full_embedding_acceptance_models(root)
    rejected = []
    for row in rows:
        embeddings = row["qa"]["embedding"]
        for key in ("vit", "osnet"):
            embeddings[key]["acceptance_required"] = key in acceptance_models
        embedding_pass = all(
            embeddings[key]["anchor_top1"]
            and embeddings[key]["above_real_positive_p05"]
            and float(embeddings[key]["cosine_margin"]) >= float(thresholds["cosine_margin_min"])
            for key in acceptance_models
        )
        duplicate_pass = str(row["sample_id"]) not in duplicate_ids
        row["qa"]["embedding_pass"] = embedding_pass
        row["qa"]["phash_duplicate_pass"] = duplicate_pass
        row["qa_status"] = "accepted" if embedding_pass and duplicate_pass else "rejected_embedding"
        if row["qa_status"] == "accepted":
            _publish_candidate(root, row)
        else:
            published = root / "accepted" / str(row["split"]) / synthetic_release_filename(row)
            published.unlink(missing_ok=True)
            rejected.append(row)

    jobs = read_jsonl(paths["jobs"])
    rejected_custom_ids = {row["request_custom_id"] for row in rejected}
    for job in jobs:
        if job["custom_id"] in rejected_custom_ids and job["status"] == "succeeded":
            job["status"] = "qa_failed"
    atomic_write_jsonl(paths["jobs"], sorted(jobs, key=lambda row: row["custom_id"]))
    atomic_write_jsonl(paths["manifest"], sorted(rows, key=lambda row: row["sample_id"]))

    accepted = [row for row in rows if row["qa_status"] == "accepted"]
    _full_review_artifacts(root, accepted)
    full_report = {
        "created_at": utc_now(),
        "model_id": config["model"]["api_id"],
        "catalog_snapshot": config["model"]["catalog_snapshot"],
        "reference_inputs": _reference_metadata(config),
        "generated": len(rows),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "phash_near_duplicate_rate": near_pairs / comparable_pairs if comparable_pairs else 1.0,
        "retrieval_label_qa": retrieval,
        "embedding_acceptance_models": list(acceptance_models),
        "embedding_advisory_models": list(advisory_models),
        "complete": len(accepted) == 20000 and not rejected,
    }
    atomic_write_json(root / "qa" / "full_report.json", full_report)
    print(json.dumps(full_report, ensure_ascii=False, indent=2))
    if rejected:
        print(
            "rejected samples were marked qa_failed; automatic API retries are disabled. "
            "Run repair-local --scope full, then report-failures --scope full."
        )


def _repair_candidate_rank(job: Mapping[str, Any], qa: Mapping[str, Any]) -> tuple[Any, ...]:
    pose_scores = (qa.get("pose_geometry") or {}).get("region_scores") or {}
    minimum_pose_score = min((float(value) for value in pose_scores.values()), default=-1e9)
    detection_score = float((qa.get("person_detection") or {}).get("principal_score", 0.0))
    return (
        bool(qa.get("accepted")),
        minimum_pose_score,
        detection_score,
        -int(job.get("attempt", 0)),
    )


def _internal_scope(scope: str) -> str:
    aliases = {"pilot": "pilot", "rotation-pilot": "rotation_pilot", "full": "full"}
    try:
        return aliases[scope]
    except KeyError as exc:
        raise ValueError(f"unsupported repair scope: {scope}") from exc


def _scope_specs(paths: Mapping[str, Path], scope: str) -> dict[str, dict[str, Any]]:
    path_key = {
        "pilot": "pilot_samples",
        "rotation_pilot": "rotation_pilot_samples",
        "full": "samples",
    }[scope]
    return {row["sample_id"]: row for row in read_jsonl(paths[path_key])}


def _scope_manifest_path(paths: Mapping[str, Path], scope: str) -> Path:
    return paths[{
        "pilot": "pilot_manifest",
        "rotation_pilot": "rotation_pilot_manifest",
        "full": "manifest",
    }[scope]]


def _local_final_path(root: Path, scope: str, sample: Mapping[str, Any]) -> Path:
    if scope == "pilot":
        directory = root / "pilot" / "images"
    elif scope == "rotation_pilot":
        directory = root / "rotation_pilot" / "images"
    else:
        directory = root / "candidates" / str(sample["split"])
    return directory / f"{sample['sample_id']}.jpg"


def _manifest_row_valid(root: Path, row: Mapping[str, Any] | None, scope: str) -> tuple[bool, str]:
    if row is None:
        return False, "manifest_missing"
    relative = row.get("final_path")
    if not isinstance(relative, str):
        return False, "final_path_missing"
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False, "final_path_escapes_root"
    if not path.is_file():
        return False, "final_image_missing"
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            if image.format != "JPEG" or image.size != (128, 256):
                return False, "final_image_format_or_size_invalid"
    except (ImportError, OSError):
        return False, "final_image_decode_failed"
    try:
        actual_sha = sha256_file(path)
    except OSError:
        return False, "final_image_unreadable"
    if actual_sha != row.get("final_sha256"):
        return False, "final_sha256_mismatch"
    allowed_statuses = {"accepted"} if scope != "full" else {"geometry_passed", "accepted"}
    if row.get("qa_status") not in allowed_statuses:
        return False, "qa_status_not_releasable"
    if not row.get("qa", {}).get("accepted"):
        return False, "local_geometry_not_accepted"
    if row.get("processing_version") != PROCESSING_VERSION:
        return False, "processing_version_stale"
    return True, "valid"


def _successful_raw_candidates(
    root: Path,
    scope: str,
    jobs: Sequence[Mapping[str, Any]],
    attempts_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any], Path]]]:
    candidates: dict[str, list[tuple[dict[str, Any], dict[str, Any], Path]]] = defaultdict(list)
    for job_row in jobs:
        job = dict(job_row)
        if job.get("scope") != scope or job.get("kind") != "sample":
            continue
        attempt_row = attempts_by_id.get(str(job["custom_id"]))
        if attempt_row is None or attempt_row.get("classification") != "succeeded":
            continue
        raw = root / "raw" / scope / f"{job['custom_id']}.jpg"
        if raw.is_file():
            candidates[str(job["sample_id"])].append((job, dict(attempt_row), raw))
    return candidates


def _walking_side(pose: object) -> str | None:
    value = str(pose or "").lower()
    if "walking left foot" in value:
        return "left"
    if "walking right foot" in value:
        return "right"
    return None


def _local_sibling_raw_candidates(
    sample_id: str,
    specs: Mapping[str, Mapping[str, Any]],
    raw_candidates: Mapping[
        str, list[tuple[dict[str, Any], dict[str, Any], Path]]
    ],
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]]:
    """Rank same-PID, same-camera frames that can replace an unusable raw image."""
    target = specs[sample_id]
    target_walking = _walking_side(target.get("pose")) is not None
    ranked = []
    for source_id, rows in raw_candidates.items():
        if source_id == sample_id or source_id not in specs:
            continue
        source = specs[source_id]
        if any(source.get(key) != target.get(key) for key in ("local_pid", "local_camera")):
            continue
        same_split = source.get("split") == target.get("split")
        same_occlusion = source.get("occluded") == target.get("occluded")
        synthetic_occlusion = bool(
            target.get("occluded") and not source.get("occluded")
        )
        if not ((same_split and same_occlusion) or synthetic_occlusion):
            continue
        source_walking = _walking_side(source.get("pose")) is not None
        yaw_delta = abs(
            int(source.get("body_yaw_deg", 0)) - int(target.get("body_yaw_deg", 0))
        )
        yaw_delta = min(yaw_delta, 360 - yaw_delta)
        rank = (
            not same_occlusion,
            not same_split,
            source_walking != target_walking,
            yaw_delta,
            abs(int(source.get("frame", 0)) - int(target.get("frame", 0))),
            source_id,
        )
        for job, attempt, raw in rows:
            ranked.append((rank, dict(source), job, attempt, raw))
    ranked.sort(key=lambda row: row[0])
    return [(source, job, attempt, raw) for _rank, source, job, attempt, raw in ranked]


def _overlay_local_occluder(
    image: Any,
    person_bbox: Sequence[float],
    target_sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Draw a deterministic, text-free foreground occluder over a clean sibling."""
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    x, y, box_width, box_height = (float(value) for value in person_bbox)
    x0 = max(0, min(width - 1, round(x)))
    y0 = max(0, min(height - 1, round(y)))
    x1 = max(x0 + 1, min(width, round(x + box_width)))
    y1 = max(y0 + 1, min(height, round(y + box_height)))
    box_width = x1 - x0
    box_height = y1 - y0
    seed = int(target_sample["generation_seed"])
    patch = image[
        max(0, y1 - max(4, box_height // 10)) : min(height, y1 + max(4, box_height // 10)),
        max(0, x0 - box_width // 2) : min(width, x1 + box_width // 2),
    ]
    base = np.median(patch.reshape(-1, 3), axis=0) if patch.size else np.array([90, 90, 90])
    base_color = tuple(int(max(30, min(185, value * 0.72))) for value in base)
    light_color = tuple(int(min(220, value + 45)) for value in base_color)
    dark_color = tuple(int(max(18, value - 35)) for value in base_color)
    occluder = str(target_sample.get("occluder") or "foreground barrier").lower()
    thickness = max(4, round(box_width * 0.035))

    if "low wall" in occluder:
        left = max(0, x0 - round(box_width * 0.45))
        right = min(width - 1, x1 + round(box_width * 0.45))
        top = min(height - 1, y0 + round(box_height * 0.64))
        bottom = min(height - 1, y1 + round(box_height * 0.08))
        cv2.rectangle(image, (left, top), (right, bottom), base_color, -1)
        cv2.line(image, (left, top), (right, top), light_color, max(3, thickness // 2))
        kind = "low_wall"
    elif "bollard" in occluder:
        center = x0 + round(box_width * (0.46 if seed % 2 else 0.54))
        half_width = max(thickness * 2, round(box_width * 0.16))
        top = y0 + round(box_height * 0.43)
        bottom = min(height - 1, y1 + round(box_height * 0.04))
        cv2.rectangle(
            image,
            (max(0, center - half_width), top),
            (min(width - 1, center + half_width), bottom),
            dark_color,
            -1,
        )
        cv2.ellipse(
            image,
            (center, top),
            (half_width, max(3, half_width // 3)),
            0,
            180,
            360,
            light_color,
            -1,
        )
        kind = "bollard"
    else:
        left = max(0, x0 - round(box_width * 0.18))
        right = min(width - 1, x1 + round(box_width * 0.18))
        verticals = (0.08, 0.42, 0.76) if "cart" in occluder else (0.12, 0.50, 0.88)
        horizontals = (0.58, 0.73, 0.88) if "cart" in occluder else (0.56, 0.72, 0.87)
        for fraction in horizontals:
            row = y0 + round(box_height * fraction)
            cv2.line(image, (left, row), (right, row), light_color, thickness)
            cv2.line(image, (left, row + thickness), (right, row + thickness), dark_color, max(2, thickness // 2))
        for fraction in verticals:
            column = left + round((right - left) * fraction)
            cv2.line(
                image,
                (column, y0 + round(box_height * 0.53)),
                (column, min(height - 1, y1 + round(box_height * 0.04))),
                dark_color,
                thickness,
            )
        if "cart" in occluder:
            base_top = y0 + round(box_height * 0.88)
            cv2.rectangle(
                image,
                (left, base_top),
                (right, min(height - 1, y1 + round(box_height * 0.05))),
                base_color,
                -1,
            )
            kind = "luggage_cart"
        else:
            kind = "railing"
    return {
        "applied": True,
        "kind": kind,
        "source_person_bbox": [round(float(value), 3) for value in person_bbox],
        "target_occluder": target_sample.get("occluder"),
        "target_occlusion_ratio": target_sample.get("target_occlusion_ratio"),
    }


def _render_local_sibling_raw(
    source_raw: Path,
    destination: Path,
    source_sample: Mapping[str, Any],
    target_sample: Mapping[str, Any],
    source_qa: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a deterministic local frame derivative without touching API raw data."""
    import cv2
    import numpy as np

    image = cv2.imread(str(source_raw), cv2.IMREAD_COLOR)
    if image is None:
        raise PipelineError(f"local sibling raw is not decodable: {source_raw}")
    source_side = _walking_side(source_sample.get("pose"))
    target_side = _walking_side(target_sample.get("pose"))
    mirrored = source_side is not None and target_side is not None and source_side != target_side
    if mirrored:
        image = cv2.flip(image, 1)
        transformed_shift = 0
    else:
        height, width = image.shape[:2]
        seed = int(target_sample["generation_seed"])
        shift_x = max(2, round(width * 0.01)) * (-1 if seed % 2 else 1)
        matrix = np.float32([[1.0, 0.0, shift_x], [0.0, 1.0, 0.0]])
        image = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        transformed_shift = shift_x
    synthetic_occluder = {"applied": False}
    if target_sample.get("occluded") and not source_sample.get("occluded"):
        qa = source_qa or {}
        bbox = qa.get("detector_person_bbox") or qa.get("person_bbox")
        if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
            raise PipelineError("synthetic occlusion repair requires a source person bbox")
        bbox = [float(value) for value in bbox]
        if mirrored:
            bbox[0] = image.shape[1] - (bbox[0] + bbox[2])
        else:
            bbox[0] += transformed_shift
        synthetic_occluder = _overlay_local_occluder(image, bbox, target_sample)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 98]):
        raise PipelineError(f"failed to write local sibling derivative: {destination}")
    return {
        "mirrored": mirrored,
        "source_walking_side": source_side,
        "target_walking_side": target_side,
        "non_mirrored_shift_fraction": 0.01,
        "synthetic_occluder": synthetic_occluder,
    }


def report_local_failures(
    root: Path,
    config: Mapping[str, Any],
    requested_scope: str,
    limit: int = 100,
) -> dict[str, Any]:
    """Report unresolved specifications and whether a successful raw response can rebuild them."""
    paths = initialize(root, config)
    scope = _internal_scope(requested_scope)
    specs = _scope_specs(paths, scope)
    manifest_rows = {row["sample_id"]: row for row in read_jsonl(_scope_manifest_path(paths, scope))}
    jobs = read_jsonl(paths["jobs"])
    attempts = read_jsonl(paths["attempts"])
    attempts_by_id = {row["custom_id"]: row for row in attempts}
    raw_candidates = _successful_raw_candidates(root, scope, jobs, attempts_by_id)
    jobs_by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for job in jobs:
        if job.get("scope") == scope and job.get("kind") == "sample":
            jobs_by_sample[str(job["sample_id"])].append(job)
    valid = 0
    recoverable = []
    awaiting_first_attempt = []
    failed_without_raw = []
    for sample_id in sorted(specs):
        is_valid, reason = _manifest_row_valid(root, manifest_rows.get(sample_id), scope)
        if is_valid:
            valid += 1
            continue
        rows = jobs_by_sample.get(sample_id, [])
        status = max(rows, key=lambda row: int(row.get("attempt", 0))).get("status") if rows else "not_planned"
        item = {
            "sample_id": sample_id,
            "reason": reason,
            "job_status": status,
            "successful_raw_candidates": len(raw_candidates.get(sample_id, [])),
        }
        if sample_id in raw_candidates:
            recoverable.append(item)
        elif status in {"not_planned", "planned", "blocked_on_refs", "submitted"}:
            awaiting_first_attempt.append(item)
        else:
            failed_without_raw.append(item)
    active_batches = [
        {"batch_id": row.get("batch_id"), "status": row.get("status")}
        for row in read_jsonl(paths["batches"])
        if row.get("scope") == scope
        and (
            row.get("status") in {"validating", "in_progress", "finalizing", "cancelling"}
            or (row.get("status") == "completed" and not row.get("collected"))
        )
    ]
    return {
        "scope": requested_scope,
        "specifications": len(specs),
        "valid_final_images": valid,
        "unresolved": len(recoverable) + len(awaiting_first_attempt) + len(failed_without_raw),
        "recoverable_from_successful_raw": len(recoverable),
        "awaiting_first_attempt": len(awaiting_first_attempt),
        "failed_without_successful_raw": len(failed_without_raw),
        "automatic_api_retries": int(config["batch"]["max_attempts"]) - 1,
        "active_batches": active_batches,
        "recoverable": recoverable[:limit],
        "awaiting": awaiting_first_attempt[:limit],
        "failed": failed_without_raw[:limit],
        "details_truncated": any(
            len(rows) > limit
            for rows in (recoverable, awaiting_first_attempt, failed_without_raw)
        ),
    }


def repair_local(
    root: Path,
    config: Mapping[str, Any],
    requested_scope: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Rebuild missing or invalid derived images from successful API raw responses only."""
    paths = initialize(root, config)
    scope = _internal_scope(requested_scope)
    active = [
        row for row in read_jsonl(paths["batches"])
        if row.get("scope") == scope
        and (
            row.get("status") in {"validating", "in_progress", "finalizing", "cancelling"}
            or (row.get("status") == "completed" and not row.get("collected"))
        )
    ]
    if active:
        states = ", ".join(f"{row.get('batch_id')}:{row.get('status')}" for row in active)
        raise PipelineError(
            f"local repair requires collected terminal Batch state for {requested_scope}: {states}"
        )
    specs = _scope_specs(paths, scope)
    manifest_path = _scope_manifest_path(paths, scope)
    manifest_rows = {row["sample_id"]: row for row in read_jsonl(manifest_path)}
    jobs = read_jsonl(paths["jobs"])
    jobs_by_id = {row["custom_id"]: row for row in jobs}
    jobs_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in jobs:
        if row.get("scope") == scope and row.get("kind") == "sample":
            jobs_by_sample[str(row["sample_id"])].append(row)
    attempts = read_jsonl(paths["attempts"])
    attempts_by_id = {row["custom_id"]: row for row in attempts}
    cameras = read_jsonl(paths["cameras"])
    assets = {row["asset_id"]: row for row in read_jsonl(paths["assets"])}
    raw_candidates = _successful_raw_candidates(root, scope, jobs, attempts_by_id)
    needs_repair = []
    already_valid = 0
    initial_reasons: dict[str, str] = {}
    for sample_id in sorted(specs):
        valid, reason = _manifest_row_valid(root, manifest_rows.get(sample_id), scope)
        if valid:
            already_valid += 1
        else:
            needs_repair.append(sample_id)
            initial_reasons[sample_id] = reason

    qa_dir = root / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    repaired: dict[
        str,
        tuple[
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            Path,
            int,
            dict[str, Any],
        ],
    ] = {}
    rejected: dict[str, list[dict[str, Any]]] = {}
    missing_raw = []
    sibling_repaired = 0
    with tempfile.TemporaryDirectory(prefix=f".local-repair-{scope}-", dir=qa_dir) as temporary:
        work = Path(temporary)
        processed = 0
        for sample_id in needs_repair:
            sample = specs[sample_id]
            candidates = raw_candidates.get(sample_id, [])
            if not candidates:
                missing_raw.append(sample_id)
            evaluated = []
            for job, attempt, raw in candidates:
                candidate_path = work / f"{job['custom_id']}.jpg"
                qa = process_image(
                    raw,
                    candidate_path,
                    cameras[int(sample["local_camera"])],
                    sample,
                    config,
                )
                evaluated.append((job, attempt, qa, candidate_path))
                processed += 1
            winner = (
                max(evaluated, key=lambda row: _repair_candidate_rank(row[0], row[2]))
                if evaluated
                else None
            )
            target_job = winner[0] if winner else max(
                jobs_by_sample.get(sample_id, []),
                key=lambda row: (int(row.get("attempt", 0)), str(row.get("custom_id", ""))),
                default=None,
            )
            target_attempt = (
                winner[1]
                if winner
                else attempts_by_id.get(str(target_job.get("custom_id")))
                if target_job
                else None
            )
            if winner and winner[2].get("accepted"):
                repaired[sample_id] = (
                    *winner,
                    len(evaluated),
                    {
                        "method": "local_raw_reprocess",
                        "processing_version": winner[2].get("processing_version"),
                        "candidate_count": len(evaluated),
                        "selected_custom_id": winner[0]["custom_id"],
                    },
                )
            else:
                rejection_rows = [
                    {
                        "custom_id": job["custom_id"],
                        "reason": qa.get("reason"),
                        "attempt": job.get("attempt"),
                    }
                    for job, _attempt, qa, _candidate in evaluated
                ]
                if not evaluated:
                    rejection_rows.append({
                        "custom_id": target_job.get("custom_id") if target_job else None,
                        "reason": "successful_raw_missing",
                    })
                if scope == "full" and target_job is not None and target_attempt is not None:
                    sibling_rows = _local_sibling_raw_candidates(
                        sample_id, specs, raw_candidates
                    )
                    sibling_attempts = 0
                    for source, source_job, source_attempt, source_raw in sibling_rows:
                        sibling_attempts += 1
                        local_raw = work / (
                            f"local-sibling-{source['sample_id']}-for-{sample_id}.jpg"
                        )
                        transform = _render_local_sibling_raw(
                            source_raw,
                            local_raw,
                            source,
                            sample,
                            source_qa=(
                                source_attempt.get("qa_reprocessed")
                                or source_attempt.get("qa")
                                or {}
                            ),
                        )
                        candidate_path = work / (
                            f"local-sibling-final-{source['sample_id']}-for-{sample_id}.jpg"
                        )
                        qa = process_image(
                            local_raw,
                            candidate_path,
                            cameras[int(sample["local_camera"])],
                            sample,
                            config,
                        )
                        processed += 1
                        if qa.get("accepted"):
                            qa = dict(qa)
                            qa["local_sibling_repair"] = {
                                "source_sample_id": source["sample_id"],
                                "source_custom_id": source_job["custom_id"],
                                **transform,
                            }
                            repaired[sample_id] = (
                                target_job,
                                target_attempt,
                                qa,
                                candidate_path,
                                len(evaluated) + sibling_attempts,
                                {
                                    "method": "local_sibling_reprocess",
                                    "processing_version": qa.get("processing_version"),
                                    "selected_custom_id": target_job["custom_id"],
                                    "source_sample_id": source["sample_id"],
                                    "source_custom_id": source_job["custom_id"],
                                    **transform,
                                },
                            )
                            sibling_repaired += 1
                            break
                        rejection_rows.append(
                            {
                                "custom_id": target_job["custom_id"],
                                "source_sample_id": source["sample_id"],
                                "source_custom_id": source_job["custom_id"],
                                "reason": qa.get("reason"),
                                "local_sibling": True,
                            }
                        )
                if sample_id not in repaired:
                    rejected[sample_id] = rejection_rows
            if processed and processed % 100 == 0:
                print(f"local repair QA: {processed} raw candidates processed", file=sys.stderr)

        report = {
            "created_at": utc_now(),
            "scope": requested_scope,
            "specifications": len(specs),
            "already_valid": already_valid,
            "needed_repair": len(needs_repair),
            "raw_candidates_processed": processed,
            "locally_repairable": len(repaired),
            "local_sibling_repairable": sibling_repaired,
            "missing_successful_raw_count": len(missing_raw),
            "missing_successful_raw": missing_raw[:100],
            "local_qa_rejected": rejected,
            "initial_reason_counts": dict(Counter(initial_reasons.values())),
            "initial_reasons": dict(list(initial_reasons.items())[:100]),
            "details_truncated": len(missing_raw) > 100 or len(initial_reasons) > 100,
            "automatic_api_retries": int(config["batch"]["max_attempts"]) - 1,
            "dry_run": dry_run,
        }
        if dry_run:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return report

        for sample_id in sorted(repaired):
            job, attempt, qa, candidate_path, candidate_count, selection = repaired[sample_id]
            sample = specs[sample_id]
            final = _local_final_path(root, scope, sample)
            atomic_write_bytes(final, candidate_path.read_bytes())
            qa = dict(qa)
            qa["final_sha256"] = sha256_file(final)
            selection = {**selection, "candidate_count": candidate_count}
            qa_status = "accepted" if scope != "full" else "geometry_passed"
            manifest_rows[sample_id] = _sample_manifest_row(
                root,
                config,
                sample,
                job,
                attempt,
                cameras,
                assets,
                qa,
                final,
                qa_status=qa_status,
                selection=selection,
            )
            attempts_by_id[job["custom_id"]]["qa_reprocessed"] = qa
            attempts_by_id[job["custom_id"]]["selected_by_local_repair"] = True
            jobs_by_id[job["custom_id"]]["status"] = "succeeded"

        atomic_write_jsonl(
            manifest_path, sorted(manifest_rows.values(), key=lambda row: row["sample_id"])
        )
        atomic_write_jsonl(
            paths["attempts"], [attempts_by_id[row["custom_id"]] for row in attempts]
        )
        atomic_write_jsonl(
            paths["jobs"], sorted(jobs_by_id.values(), key=lambda row: row["custom_id"])
        )
        report["repaired"] = len(repaired)
        report["remaining_unresolved"] = len(
            (set(missing_raw) | set(rejected)) - set(repaired)
        )
        report_path = qa_dir / f"local_repair_{scope}_report.json"
        atomic_write_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report


def repair_pilot(root: Path, config: Mapping[str, Any], dry_run: bool = False) -> dict[str, Any]:
    """Re-QA every existing raw pilot attempt and atomically select one per specification."""
    paths = initialize(root, config)
    batches = read_jsonl(paths["batches"])
    active = [
        batch for batch in batches
        if batch.get("status") in {"validating", "in_progress", "finalizing", "cancelling"}
        or (batch.get("status") == "completed" and not batch.get("collected"))
    ]
    if active:
        states = ", ".join(f"{row.get('batch_id')}:{row.get('status')}" for row in active)
        raise PipelineError(
            "pilot repair requires every Batch result to be collected first; "
            f"run pilot --resume after completion ({states})"
        )

    specs = {row["sample_id"]: row for row in read_jsonl(paths["pilot_samples"])}
    expected = 96
    if len(specs) != expected:
        raise PipelineError(
            f"pilot repair expected {expected} Low specifications, found {len(specs)}"
        )
    jobs = read_jsonl(paths["jobs"])
    attempts = read_jsonl(paths["attempts"])
    attempts_by_id = {row["custom_id"]: row for row in attempts}
    cameras = read_jsonl(paths["cameras"])
    assets = {row["asset_id"]: row for row in read_jsonl(paths["assets"])}
    if len(assets) != 81:
        raise PipelineError(f"pilot repair requires all 81 reference assets, found {len(assets)}")

    candidates: dict[str, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]]] = defaultdict(list)
    qa_dir = root / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pilot-repair-", dir=qa_dir) as temporary:
        work = Path(temporary)
        processed = 0
        for job in jobs:
            if job.get("scope") != "pilot" or job.get("kind") != "sample":
                continue
            attempt = attempts_by_id.get(job["custom_id"])
            raw = root / "raw" / "pilot" / f"{job['custom_id']}.jpg"
            if attempt is None or attempt.get("classification") != "succeeded" or not raw.exists():
                continue
            sample = specs[job["sample_id"]]
            candidate_path = work / f"{job['custom_id']}.jpg"
            qa = process_image(raw, candidate_path, cameras[int(sample["local_camera"])], sample, config)
            candidates[job["sample_id"]].append((job, attempt, qa, candidate_path))
            processed += 1
            if processed % 25 == 0:
                print(f"pilot repair QA: {processed} raw candidates processed", file=sys.stderr)
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:  # pragma: no cover - process_image reports this dependency first
                    pass

        missing_raw = sorted(set(specs) - set(candidates))
        selected: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]] = {}
        rejected: dict[str, list[dict[str, Any]]] = {}
        for sample_id, rows in candidates.items():
            winner = max(rows, key=lambda row: _repair_candidate_rank(row[0], row[2]))
            if winner[2].get("accepted"):
                selected[sample_id] = winner
            else:
                rejected[sample_id] = [
                    {
                        "custom_id": job["custom_id"],
                        "attempt": job["attempt"],
                        "quality": job.get("quality"),
                        "reason": qa.get("reason"),
                    }
                    for job, _attempt, qa, _path in rows
                ]

        report = {
            "created_at": utc_now(),
            "processing_version": next(
                (row[2].get("processing_version") for row in selected.values()), None
            ),
            "specifications": len(specs),
            "raw_candidates": sum(len(rows) for rows in candidates.values()),
            "selected": len(selected),
            "missing_raw": missing_raw,
            "rejected": rejected,
            "complete": len(selected) == len(specs) and not missing_raw and not rejected,
            "dry_run": dry_run,
        }
        if not report["complete"]:
            atomic_write_json(qa_dir / "pilot_repair_report.json", report)
            raise PipelineError(
                f"pilot repair could select {len(selected)}/{expected} specifications; "
                "no manifest or final image was changed"
            )
        if dry_run:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return report

        jobs_by_id = {row["custom_id"]: row for row in jobs}
        for rows in candidates.values():
            for candidate_job, _attempt, candidate_qa, _candidate_path in rows:
                jobs_by_id[candidate_job["custom_id"]]["status"] = "superseded"
                attempts_by_id[candidate_job["custom_id"]]["qa_reprocessed"] = candidate_qa
                attempts_by_id[candidate_job["custom_id"]]["selected_by_repair"] = False
        repaired_manifest = []
        for sample_id in sorted(selected):
            job, attempt, qa, candidate_path = selected[sample_id]
            sample = specs[sample_id]
            final = root / "pilot" / "images" / f"{sample_id}.jpg"
            atomic_write_bytes(final, candidate_path.read_bytes())
            qa = dict(qa)
            qa["final_sha256"] = sha256_file(final)
            selection = {
                "method": "best_geometry_candidate",
                "processing_version": qa.get("processing_version"),
                "candidate_count": len(candidates[sample_id]),
                "selected_custom_id": job["custom_id"],
            }
            repaired_manifest.append(_sample_manifest_row(
                root, config, sample, job, attempt, cameras, assets, qa, final,
                qa_status="accepted", selection=selection,
            ))
            attempts_by_id[job["custom_id"]]["selected_by_repair"] = True
            jobs_by_id[job["custom_id"]]["status"] = "succeeded"

        atomic_write_jsonl(paths["pilot_manifest"], repaired_manifest)
        atomic_write_jsonl(paths["attempts"], [attempts_by_id[row["custom_id"]] for row in attempts])
        atomic_write_jsonl(paths["jobs"], sorted(jobs_by_id.values(), key=lambda row: row["custom_id"]))
        atomic_write_json(qa_dir / "pilot_repair_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report


def command_pilot(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    jobs, _ = plan_scope(args.root, config, "pilot")
    if args.dry_run:
        written = write_dry_run_requests(args.root, "pilot", jobs, config)
        print(f"pilot dry-run ready: {len(written)} JSONL files under {written[0].parent if written else args.root}")
        return
    run_live(args.root, config, "pilot")


def command_rotation_pilot(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    jobs, paths = plan_scope(args.root, config, "rotation_pilot")
    if args.dry_run:
        written = write_dry_run_requests(args.root, "rotation_pilot", jobs, config)
        print(
            f"body-rotation pilot dry-run ready: {len(written)} JSONL files; "
            "no API request was made"
        )
        return
    if not args.resume:
        raise PipelineError("live body-rotation pilot execution requires --resume")
    assets = {row["asset_id"]: row for row in read_jsonl(paths["assets"])}
    required = {
        *(_asset_id("plate", camera) for camera in range(33)),
        *(
            _asset_id("anchor", pid, orientation)
            for pid in range(12)
            for orientation in ASSET_ORIENTATIONS
        ),
    }
    missing = sorted(asset_id for asset_id in required if not assets.get(asset_id, {}).get("file_id"))
    if missing:
        raise PipelineError(
            f"body-rotation pilot requires all 81 existing pilot assets; missing {len(missing)}"
        )
    run_live(args.root, config, "rotation_pilot")


def command_repair_pilot(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    repair_pilot(args.root, config, dry_run=args.dry_run)


def command_repair_local(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    repair_local(args.root, config, args.scope, dry_run=args.dry_run)


def command_report_failures(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    if args.limit < 0:
        raise ValueError("--limit must be zero or greater")
    report = report_local_failures(args.root, config, args.scope, limit=args.limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_waive_manual_review(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    record_full_manual_review_waiver(
        args.root, config, str(args.waiver_reason or "")
    )


def command_report(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    report = build_report(args.root, config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_qa(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    if args.skip_embedding_duplicate_qa and args.scope != "full":
        raise PipelineError("--skip-embedding-duplicate-qa is valid only for full scope")
    if args.scope == "pilot":
        run_embedding_qa(args.root, config, args.config.resolve())
        command_report(args, config)
    elif args.scope == "rotation-pilot":
        report = run_rotation_pilot_qa(args.root, config, args.config.resolve())
        print(json.dumps(report, ensure_ascii=False, indent=2))
        command_report(args, config)
    elif args.skip_embedding_duplicate_qa:
        finalize_full_qa_waiver(args.root, config, str(args.waiver_reason or ""))
    else:
        run_full_embedding_qa(args.root, config, args.config.resolve())


def _approval_failed_gates(
    report: Mapping[str, Any], rotation_report: Mapping[str, Any]
) -> list[str]:
    pilot_gates = report.get("gates") or {}
    rotation_gates = rotation_report.get("gates") or {}
    failed = [
        f"pilot.{name}"
        for name, passed in pilot_gates.items()
        if name != "body_rotation_pilot" and not passed
    ]
    rotation_failed = [
        f"rotation_pilot.{name}"
        for name, passed in rotation_gates.items()
        if not passed
    ]
    failed.extend(rotation_failed)
    if not pilot_gates and not report.get("pilot_gate_passed"):
        failed.append("pilot.unreported_gate_state")
    if (
        pilot_gates.get("body_rotation_pilot") is False
        and not rotation_failed
        and not rotation_report.get("rotation_gate_passed")
    ):
        failed.append("pilot.body_rotation_pilot")
    return sorted(set(failed))


def command_approve(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    paths = initialize(args.root, config)
    if not paths["report"].exists():
        raise PipelineError("pilot report does not exist; run report first")
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    selected_forecast = report.get(FORECAST_KEY, {}).get(args.quality)
    if selected_forecast is None:
        raise PipelineError("selected quality has no measured pilot cost forecast")
    if float(selected_forecast) > args.max_usd:
        raise PipelineError(
            f"forecast ${selected_forecast:.2f} exceeds requested ceiling ${args.max_usd:.2f}"
        )
    rotation_path = paths["rotation_pilot_report"]
    rotation_report = (
        json.loads(rotation_path.read_text(encoding="utf-8"))
        if rotation_path.exists()
        else {}
    )
    failed_gates = _approval_failed_gates(report, rotation_report)
    waived_gates: list[str] = []
    waiver_reason = None
    if failed_gates:
        if not args.waive_failed_gates:
            raise PipelineError(
                "pilot approval gates failed: " + ", ".join(failed_gates)
            )
        nonwaivable = sorted(set(failed_gates) - WAIVABLE_APPROVAL_GATES)
        if nonwaivable:
            raise PipelineError(
                "refusing to waive safety-critical gates: " + ", ".join(nonwaivable)
            )
        if not str(args.waiver_reason or "").strip():
            raise PipelineError("--waiver-reason is required with --waive-failed-gates")
        waived_gates = failed_gates
        waiver_reason = str(args.waiver_reason)
    elif args.waive_failed_gates:
        raise PipelineError("no failed gates exist; waiver is unnecessary")
    payload = approval_payload(
        report,
        args.quality,
        args.max_usd,
        config,
        waived_gates=waived_gates,
        waiver_reason=waiver_reason,
    )
    atomic_write_json(paths["approval"], payload)
    suffix = f"; waived {', '.join(waived_gates)}" if waived_gates else ""
    print(f"approved {args.quality} with hard ceiling ${args.max_usd:.2f}{suffix}")


def command_full(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    jobs, paths = plan_scope(args.root, config, "full")
    approval = json.loads(paths["approval"].read_text(encoding="utf-8"))
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    forecast = report[FORECAST_KEY][approval["quality"]]
    if float(forecast) > float(approval["max_usd"]):
        raise PipelineError("approved ceiling is below the current signed forecast")
    if args.dry_run:
        written = write_dry_run_requests(args.root, "full", jobs, config)
        print(f"full dry-run ready: {len(written)} JSONL files; no API request was made")
        return
    if not args.resume:
        raise PipelineError("live full execution requires --resume")
    run_live(args.root, config, "full")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    pilot = sub.add_parser("pilot", help="plan or advance the 12-ID pilot")
    pilot.add_argument("--dry-run", action="store_true")
    pilot.add_argument("--resume", action="store_true", help="poll completed batches and submit newly unblocked stages")
    rotation_pilot = sub.add_parser(
        "rotation-pilot", help="plan or advance the 12-ID, eight-yaw low-quality pilot"
    )
    rotation_pilot.add_argument("--dry-run", action="store_true")
    rotation_pilot.add_argument("--resume", action="store_true")
    repair = sub.add_parser("repair-pilot", help="re-QA raw pilot attempts and select one per specification")
    repair.add_argument("--dry-run", action="store_true")
    local_repair = sub.add_parser(
        "repair-local", help="rebuild derived images from successful raw responses without API calls"
    )
    local_repair.add_argument(
        "--scope", choices=("pilot", "rotation-pilot", "full"), required=True
    )
    local_repair.add_argument("--dry-run", action="store_true")
    failures = sub.add_parser(
        "report-failures", help="show unresolved samples and local raw recoverability"
    )
    failures.add_argument(
        "--scope", choices=("pilot", "rotation-pilot", "full"), required=True
    )
    failures.add_argument("--limit", type=int, default=100)
    manual_waiver = sub.add_parser(
        "waive-manual-review",
        help="record an explicit user waiver for full manual review without fabricating results",
    )
    manual_waiver.add_argument("--waiver-reason", required=True)
    sub.add_parser("report", help="calculate pilot QA gates and measured cost forecast")
    qa = sub.add_parser("qa", help="run ViT/OSNet QA and create manual contact sheets")
    qa.add_argument("--scope", choices=("pilot", "rotation-pilot", "full"), default="pilot")
    qa.add_argument(
        "--skip-embedding-duplicate-qa",
        action="store_true",
        help="for full scope only, record a user-authorized waiver and prepare manual review",
    )
    qa.add_argument(
        "--waiver-reason",
        help="required audit reason with --skip-embedding-duplicate-qa",
    )
    approve = sub.add_parser("approve", help="sign the pilot report and set the full-run ceiling")
    approve.add_argument("--quality", required=True, choices=("low",))
    approve.add_argument("--max-usd", required=True, type=float)
    approve.add_argument(
        "--waive-failed-gates",
        action="store_true",
        help="waive only the explicitly allowlisted failed gates and record the waiver",
    )
    approve.add_argument(
        "--waiver-reason",
        help="required audit reason when --waive-failed-gates is used",
    )
    full = sub.add_parser("full", help="plan or advance the approved 20,000-image run")
    full.add_argument("--dry-run", action="store_true")
    full.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config.resolve())
        args.root = args.root.resolve()
        {
            "pilot": command_pilot,
            "rotation-pilot": command_rotation_pilot,
            "repair-pilot": command_repair_pilot,
            "repair-local": command_repair_local,
            "report-failures": command_report_failures,
            "waive-manual-review": command_waive_manual_review,
            "report": command_report,
            "qa": command_qa,
            "approve": command_approve,
            "full": command_full,
        }[args.command](args, config)
        return 0
    except (PipelineError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
