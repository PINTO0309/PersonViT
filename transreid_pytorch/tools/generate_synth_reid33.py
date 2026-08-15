#!/usr/bin/env python3
"""Build and operate the gated SyntheticReID33 image-generation pipeline.

The command never submits the 20,000-image run until a completed pilot report
has been approved with both an image quality and a USD ceiling.

Examples (run from ``transreid_pytorch``)::

    python tools/generate_synth_reid33.py pilot --dry-run
    python tools/generate_synth_reid33.py pilot --resume
    python tools/generate_synth_reid33.py report
    python tools/generate_synth_reid33.py approve --quality low --max-usd 500
    python tools/generate_synth_reid33.py full --resume
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from synth_reid33_core import (
    SCHEMA_VERSION,
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
    usage_cost_usd,
    utc_now,
    validate_pilot,
    validate_plan,
    verify_approval,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE.parent / "configs" / "synth_reid33.yml"
DEFAULT_ROOT = HERE.parent / "data" / "SyntheticReID33"
ASSET_ORIENTATIONS = ("front", "left", "right", "back")


class PipelineError(RuntimeError):
    pass


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


def _batch_failure_summaries(batches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for batch in batches:
        if batch.get("status") not in {"failed", "expired", "cancelled"}:
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
        "plan_report": root / "state" / "plan_validation.json",
        "jobs": root / "state" / "jobs.jsonl",
        "batches": root / "state" / "batches.jsonl",
        "assets": root / "state" / "assets.jsonl",
        "attempts": root / "state" / "attempts.jsonl",
        "approval": root / "state" / "approval.json",
        "report": root / "pilot" / "report.json",
        "pilot_manifest": root / "pilot" / "manifest.jsonl",
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
        validation = validate_plan(config, identities, cameras, samples)
        pilot_validation = validate_pilot(pilot_samples)
        if not validation["valid"] or not pilot_validation["valid"]:
            raise PipelineError(f"deterministic plan failed validation: {validation} {pilot_validation}")
        atomic_write_jsonl(paths["identities"], identities)
        atomic_write_jsonl(paths["cameras"], cameras)
        atomic_write_jsonl(paths["samples"], samples)
        atomic_write_jsonl(paths["pilot_samples"], pilot_samples)
        atomic_write_json(paths["plan_report"], {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "plan": validation,
            "pilot": pilot_validation,
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
                "quality": model["anchor_quality"],
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
                "attempt": 1,
                "endpoint": "/v1/images/generations",
                "logical_refs": [],
                "body": {
                    "model": model["api_id"],
                    "prompt": plate_prompt(camera, config),
                    "n": 1,
                    "size": model["sample_size"],
                    "quality": model["anchor_quality"],
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
                    "quality": model["anchor_quality"],
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
    jobs = []
    for sample in samples:
        pid = int(sample["local_pid"])
        camera_id = int(sample["local_camera"])
        anchor = _asset_id("anchor", pid, str(sample["orientation"]))
        plate = _asset_id("plate", camera_id)
        jobs.append({
            "custom_id": f"{scope}-{sample['sample_id']}-a1",
            "scope": scope,
            "kind": "sample",
            "sample_id": sample["sample_id"],
            "attempt": 1,
            "quality": quality if scope == "full" else sample["quality"],
            "endpoint": "/v1/images/edits",
            "logical_refs": [anchor, plate],
            "body": {
                "model": model["api_id"],
                "prompt": sample_prompt(sample, identities[pid], cameras[camera_id], config),
                "images": [{"file_id": f"ref:{anchor}"}, {"file_id": f"ref:{plate}"}],
                "n": 1,
                "size": model["sample_size"],
                "quality": quality if scope == "full" else sample["quality"],
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
            config, read_jsonl(paths["pilot_samples"]), identities, cameras, "pilot", quality="low"
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
    selected = [row for row in jobs if row["scope"] in (scope, "asset")]
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
    full_specs = {row["sample_id"]: row for row in read_jsonl(paths["samples"])}
    cameras = read_jsonl(paths["cameras"])
    changed = False
    for batch in batches:
        if batch["status"] in {"failed", "expired", "cancelled"}:
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
                file_id = client.upload(raw, purpose="vision")
                assets[job["asset_id"]] = {
                    "asset_id": job["asset_id"],
                    "kind": job["kind"],
                    "local_pid": job.get("local_pid"),
                    "local_camera": job.get("local_camera"),
                    "orientation": job.get("orientation"),
                    "path": str(raw.relative_to(root)),
                    "sha256": sha256_file(raw),
                    "file_id": file_id,
                    "model_id": config["model"]["api_id"],
                    "catalog_snapshot": config["model"]["catalog_snapshot"],
                    "response_model": body.get("model"),
                    "custom_id": custom_id,
                }
                job["status"] = "succeeded"
            else:
                sample = (pilot_specs if job["scope"] == "pilot" else full_specs)[job["sample_id"]]
                raw = root / "raw" / job["scope"] / f"{job['custom_id']}.jpg"
                raw.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes(raw, image_bytes)
                final = root / ("pilot/images" if job["scope"] == "pilot" else f"candidates/{sample['split']}") / f"{sample['sample_id']}.jpg"
                qa = process_image(raw, final, cameras[int(sample["local_camera"])], sample, config)
                attempt_row["qa"] = qa
                if qa["accepted"]:
                    job["status"] = "succeeded"
                    _upsert_manifest(
                        paths["pilot_manifest"] if job["scope"] == "pilot" else paths["manifest"],
                        _sample_manifest_row(
                            root, config, sample, job, attempt_row, cameras, assets, qa, final,
                            qa_status="accepted" if job["scope"] == "pilot" else "geometry_passed",
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
        "camera_isp": camera["isp"],
        "camera_processing_version": camera.get("processing_version"),
        "processing_version": qa.get("processing_version"),
        "prompt_version": config["prompt"]["version"],
        "prompt": job["body"]["prompt"],
        "prompt_sha256": sha256_bytes(job["body"]["prompt"].encode()),
        "reference_asset_ids": job["logical_refs"],
        "reference_sha256": [assets[ref]["sha256"] for ref in job["logical_refs"]],
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
) -> list[dict[str, Any]]:
    max_attempts = int(config["batch"]["max_attempts"])
    existing_ids = {job["custom_id"] for job in jobs}
    additions = []
    for failed in jobs:
        if failed["status"] not in {"needs_revision", "qa_failed", "retryable_failed", "expired_failed"}:
            continue
        if int(failed["attempt"]) >= max_attempts:
            continue
        # A user-correctable/moderation prompt is revised only once.
        if failed["status"] == "needs_revision" and int(failed["attempt"]) >= 2:
            continue
        attempt = int(failed["attempt"]) + 1
        custom_id = failed["custom_id"].rsplit("-a", 1)[0] + f"-a{attempt}"
        if custom_id in existing_ids:
            continue
        retry = json.loads(json.dumps(failed))
        retry["custom_id"] = custom_id
        retry["attempt"] = attempt
        retry["status"] = "blocked_on_refs" if retry["logical_refs"] else "planned"
        if retry["kind"] == "sample":
            sample = (pilot_specs if retry["scope"] == "pilot" else full_specs)[retry["sample_id"]]
            if failed["status"] == "needs_revision":
                retry["body"]["prompt"] = sample_prompt(
                    sample, identities[int(sample["local_pid"])], cameras[int(sample["local_camera"])], config,
                    neutral_revision=True,
                )
            if failed["body"].get("quality") == "low":
                retry["body"]["quality"] = "medium"
                retry["quality"] = "medium"
        elif failed["status"] == "needs_revision":
            retry["body"]["prompt"] = (
                retry["body"]["prompt"]
                + " Rephrase as a neutral ordinary adult pedestrian dataset reference with no sensitive context."
            )
        additions.append(retry)
        existing_ids.add(custom_id)
    jobs.extend(additions)
    return jobs


def _current_cost(paths: Mapping[str, Path], config: Mapping[str, Any]) -> float:
    return round(sum(float(row.get("estimated_cost_usd", 0)) for row in read_jsonl(paths["attempts"])), 8)


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


def run_live(root: Path, config: Mapping[str, Any], scope: str) -> None:
    jobs, paths = plan_scope(root, config, scope)
    client = OpenAIBatchClient()
    client.verify_model(str(config["model"]["api_id"]))
    _refresh_and_collect(client, root, config, paths)
    jobs = read_jsonl(paths["jobs"])
    identities = read_jsonl(paths["identities"])
    cameras = read_jsonl(paths["cameras"])
    pilot_specs = {row["sample_id"]: row for row in read_jsonl(paths["pilot_samples"])}
    full_specs = {row["sample_id"]: row for row in read_jsonl(paths["samples"])}
    jobs = _prepare_retries(jobs, config, identities, cameras, pilot_specs, full_specs)
    assets = {row["asset_id"]: row for row in read_jsonl(paths["assets"])}
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
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in eligible:
        groups[job["endpoint"]].append(job)
    batches = read_jsonl(paths["batches"])
    chunk_size = int(config["batch"]["requests_per_sample_batch"])
    cost_estimates, conservative_fallback = _job_cost_estimates(paths)
    pending_reserve = 0.0
    for endpoint, group in sorted(groups.items()):
        group.sort(key=lambda row: row["custom_id"])
        for chunk_no, start in enumerate(range(0, len(group), chunk_size)):
            chunk = group[start : start + chunk_size]
            request_path = root / "requests" / "live" / f"{scope}-{endpoint.rsplit('/', 1)[-1]}-{utc_now().replace(':', '')}-{chunk_no:03d}.jsonl"
            lines = []
            for job in chunk:
                body = _materialize_body(job, assets)
                if body is None:
                    continue
                lines.append(make_batch_line(job["custom_id"], endpoint, body))
            if not lines:
                continue
            if scope == "full":
                approval = json.loads(paths["approval"].read_text(encoding="utf-8"))
                next_estimate = 0.0
                for job in chunk:
                    key = f"pilot:{job.get('quality')}" if job["kind"] == "sample" else (
                        f"asset:{job['body'].get('quality')}"
                    )
                    next_estimate += cost_estimates.get(key, conservative_fallback)
                enforce_cost_ceiling(
                    _current_cost(paths, config), pending_reserve, next_estimate, float(approval["max_usd"])
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
            pending_reserve += next_estimate if scope == "full" else 0.0
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
    }
    failures = _batch_failure_summaries(batches)
    if failures:
        result["batch_failures"] = failures
    print(json.dumps(result, default=dict, indent=2))


def build_report(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    paths = initialize(root, config)
    rows = read_jsonl(paths["pilot_manifest"])
    attempts = [row for row in read_jsonl(paths["attempts"]) if row.get("scope") in ("pilot", "asset")]
    automatic = {
        "accepted_images": len(rows),
        "expected_images": 192,
        "decode_rate": sum(bool(row.get("qa", {}).get("decode")) for row in rows) / 192,
        "geometry_rate": sum(bool(row.get("qa", {}).get("geometry_pass")) for row in rows) / 192,
        "framing_rate": sum(bool(row.get("qa", {}).get("framing", {}).get("pass")) for row in rows) / 192,
        "sha_unique": len({row.get("final_sha256") for row in rows}) == len(rows),
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
        "major_anatomy_failures": None,
        "text_logo_watermark_failures": None,
        "query_occlusion_valid_rate": 0.0,
    }
    thresholds = config["qa"]
    gates = {
        "image_count": len(rows) == 192,
        "decode": automatic["decode_rate"] >= float(thresholds["decode_rate_min"]),
        "geometry": automatic["geometry_rate"] >= float(thresholds["geometry_rate_min"]),
        "framing": automatic["framing_rate"] >= float(thresholds["geometry_rate_min"]),
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
        and float(manual.get("query_occlusion_valid_rate", 0)) >= 0.98,
    }
    usage = aggregate_usage(row.get("usage", {}) for row in attempts)
    actual_cost = usage_cost_usd(usage, config)
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
    for quality in config["model"]["pilot_qualities"]:
        observed = costs_by_quality.get(quality, [])
        if observed and per_asset is not None:
            sample_cost = sum(observed) / len(observed)
            remaining_assets = per_asset * max(0, (500 * 4 + 33) - len(successful_assets))
            full_samples = sample_cost * 20000 * (1 + float(config["batch"]["retry_reserve_fraction"]))
            forecast[quality] = round(actual_cost + remaining_assets + full_samples, 2)
        else:
            forecast[quality] = None
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "model_id": config["model"]["api_id"],
        "catalog_snapshot": config["model"]["catalog_snapshot"],
        "config_sha256": config_sha256(config),
        "automatic": automatic,
        "manual": manual,
        "gates": gates,
        "pilot_gate_passed": all(gates.values()),
        "usage": usage,
        "pilot_actual_cost_usd": actual_cost,
        "forecast_total_usd_including_20pct_retry": forecast,
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
    ordered = sorted(rows, key=lambda row: (row["quality"], row["local_pid"], row["local_camera"]))
    sheet = Image.new("RGB", (columns * thumb_w, ((len(ordered) + columns - 1) // columns) * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(ordered):
        image = Image.open(root / row["final_path"]).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        x = (index % columns) * thumb_w + (thumb_w - image.width) // 2
        y = (index // columns) * (thumb_h + label_h)
        sheet.paste(image, (x, y))
        draw.text((index % columns * thumb_w + 2, y + thumb_h + 3),
                  f"{row['quality'][0]} p{row['local_pid']:02d} c{row['local_camera']:02d}", fill="black")
    path = root / "qa" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)
    return path


def run_embedding_qa(root: Path, config: Mapping[str, Any], config_path: Path) -> None:
    paths = initialize(root, config)
    rows = read_jsonl(paths["pilot_manifest"])
    if len(rows) != 192:
        raise PipelineError(f"embedding QA requires 192 accepted pilot images; found {len(rows)}")
    assets = {row["asset_id"]: row for row in read_jsonl(paths["assets"])}
    anchor_paths: list[Path] = []
    anchor_pids: list[int] = []
    for pid in range(12):
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
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else {}
    for model_key, key in (("vit", "vit_onnx"), ("osnet", "osnet_onnx")):
        model_path = (config_path.parent / config["qa"][key]).resolve()
        anchor_vectors = _onnx_embeddings(model_path, anchor_paths)
        candidate_vectors = _onnx_embeddings(model_path, candidate_paths)
        centroids = []
        for pid in range(12):
            centroid = anchor_vectors[np.asarray(anchor_pids) == pid].mean(axis=0)
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
    for row in rows:
        row.setdefault("qa", {})["phash"] = row.get("qa", {}).get("phash") or perceptual_hash(root / row["final_path"])
    atomic_write_jsonl(paths["pilot_manifest"], rows)
    contact_sheet = _write_contact_sheet(root, rows)
    template = root / "qa" / "manual_review.template.json"
    if not template.exists():
        atomic_write_json(template, {
            "complete": False,
            "reviewed_contact_sheet": str(contact_sheet.relative_to(root)),
            "identity_consistency_rate": 0.0,
            "major_anatomy_failures": 0,
            "text_logo_watermark_failures": 0,
            "query_occlusion_valid_rate": 0.0,
            "reviewer_notes": "Copy to manual_review.json only after completing the stratified review.",
        })
    print(f"embedding QA complete; review {contact_sheet}")


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
    destination = root / "accepted" / str(row["split"]) / f"{row['sample_id']}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".publishing")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    row["final_path"] = str(destination.relative_to(root))


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
                "model_sha256": sha256_file(model_path),
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
    rejected = []
    for row in rows:
        embeddings = row["qa"]["embedding"]
        embedding_pass = all(
            embeddings[key]["anchor_top1"]
            and embeddings[key]["above_real_positive_p05"]
            and float(embeddings[key]["cosine_margin"]) >= float(thresholds["cosine_margin_min"])
            for key in ("vit", "osnet")
        )
        duplicate_pass = str(row["sample_id"]) not in duplicate_ids
        row["qa"]["embedding_pass"] = embedding_pass
        row["qa"]["phash_duplicate_pass"] = duplicate_pass
        row["qa_status"] = "accepted" if embedding_pass and duplicate_pass else "rejected_embedding"
        if row["qa_status"] == "accepted":
            _publish_candidate(root, row)
        else:
            published = root / "accepted" / str(row["split"]) / f"{row['sample_id']}.jpg"
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
    manual_template = root / "qa" / "full_review.template.json"
    if not manual_template.exists():
        atomic_write_json(manual_template, {
            "complete": False,
            "sample_fraction": 0.05,
            "reviewed_contact_sheets": [str(path.relative_to(root)) for path in sheet_paths],
            "reviewed_all_boundary_cases": False,
            "major_anatomy_failures": 0,
            "text_logo_watermark_failures": 0,
            "identity_consistency_rate": 0.0,
            "reviewer_notes": "Copy to full_review.json after the stratified 5% and all boundary cases are reviewed.",
        })
    full_report = {
        "created_at": utc_now(),
        "model_id": config["model"]["api_id"],
        "catalog_snapshot": config["model"]["catalog_snapshot"],
        "generated": len(rows),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "phash_near_duplicate_rate": near_pairs / comparable_pairs if comparable_pairs else 1.0,
        "retrieval_label_qa": retrieval,
        "complete": len(accepted) == 20000 and not rejected,
    }
    atomic_write_json(root / "qa" / "full_report.json", full_report)
    print(json.dumps(full_report, ensure_ascii=False, indent=2))
    if rejected:
        print("rejected samples were marked qa_failed; run full --resume for bounded replacements")


def _repair_candidate_rank(job: Mapping[str, Any], qa: Mapping[str, Any]) -> tuple[Any, ...]:
    pose_scores = (qa.get("pose_geometry") or {}).get("region_scores") or {}
    minimum_pose_score = min((float(value) for value in pose_scores.values()), default=-1e9)
    detection_score = float((qa.get("person_detection") or {}).get("principal_score", 0.0))
    return (
        bool(qa.get("accepted")),
        minimum_pose_score,
        detection_score,
        str(job.get("quality")) == "medium",
        -int(job.get("attempt", 0)),
    )


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
    if len(specs) != 192:
        raise PipelineError(f"pilot repair expected 192 specifications, found {len(specs)}")
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
                f"pilot repair could select {len(selected)}/192 specifications; "
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


def command_repair_pilot(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    repair_pilot(args.root, config, dry_run=args.dry_run)


def command_report(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    report = build_report(args.root, config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_qa(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    if args.scope == "pilot":
        run_embedding_qa(args.root, config, args.config.resolve())
        command_report(args, config)
    else:
        run_full_embedding_qa(args.root, config, args.config.resolve())


def command_approve(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    paths = initialize(args.root, config)
    if not paths["report"].exists():
        raise PipelineError("pilot report does not exist; run report first")
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    selected_forecast = report.get("forecast_total_usd_including_20pct_retry", {}).get(args.quality)
    if selected_forecast is None:
        raise PipelineError("selected quality has no measured pilot cost forecast")
    if float(selected_forecast) > args.max_usd:
        raise PipelineError(
            f"forecast ${selected_forecast:.2f} exceeds requested ceiling ${args.max_usd:.2f}"
        )
    payload = approval_payload(report, args.quality, args.max_usd, config)
    atomic_write_json(paths["approval"], payload)
    print(f"approved {args.quality} with hard ceiling ${args.max_usd:.2f}")


def command_full(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    jobs, paths = plan_scope(args.root, config, "full")
    approval = json.loads(paths["approval"].read_text(encoding="utf-8"))
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    forecast = report["forecast_total_usd_including_20pct_retry"][approval["quality"]]
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
    repair = sub.add_parser("repair-pilot", help="re-QA raw pilot attempts and select one per specification")
    repair.add_argument("--dry-run", action="store_true")
    sub.add_parser("report", help="calculate pilot QA gates and measured cost forecast")
    qa = sub.add_parser("qa", help="run ViT/OSNet QA and create manual contact sheets")
    qa.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    approve = sub.add_parser("approve", help="sign the pilot report and set the full-run ceiling")
    approve.add_argument("--quality", required=True, choices=("low", "medium"))
    approve.add_argument("--max-usd", required=True, type=float)
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
            "repair-pilot": command_repair_pilot,
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
