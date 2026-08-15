"""Deterministic planning, validation, and image processing for SyntheticReID33.

This module deliberately contains no API calls.  Network orchestration lives in
``generate_synth_reid33.py`` so allocation and safety gates remain unit-testable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by CLI preflight
    raise RuntimeError("PyYAML is required; install the 'synth' extra") from exc


SCHEMA_VERSION = "synth-reid33/v1"
PROCESSING_VERSION = "synth-reid33-isp-v4-tight-crop-torchvision-qa"
SPLITS = ("train", "query", "gallery")
PERSON_SCORE_MIN = 0.35
PERSON_NMS_IOU = 0.35
POSE_SCORE_MIN = 0.70
POSE_MATCH_IOU_MIN = 0.30
KEYPOINT_VISIBILITY_LOGIT_MIN = 0.0
FINAL_NAME_RE = re.compile(
    r"^p(?P<pid>\d{5})_d(?P<domain>\d{2})_c(?P<camera>\d{3})_(?P<seq>\d{6})\.jpe?g$"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported config schema in {path}")
    if config["model"].get("api_id") != "gpt-image-2":
        raise ValueError("Batch model must be the explicitly approved gpt-image-2 alias")
    if config["model"].get("catalog_snapshot") != "gpt-image-2-2026-04-21":
        raise ValueError("catalog snapshot must be gpt-image-2-2026-04-21")
    if len(config["sites"]) != 11 or len(config["camera_views"]) != 3:
        raise ValueError("configuration must expand to 11 sites x 3 cameras")
    return config


def config_sha256(config: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(config))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n")


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json(row) + b"\n" for row in rows)
    atomic_write_bytes(path, payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            rows.append(item)
    return rows


def stable_rng(seed: int, *parts: object) -> random.Random:
    material = "|".join([str(seed), *(str(part) for part in parts)]).encode()
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def make_identities(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    dataset = config["dataset"]
    attrs = config["identity_attributes"]
    seed = int(config["allocation"]["seed"])
    rows = []
    for pid in range(int(dataset["identities"])):
        rng = stable_rng(seed, "identity", pid)
        low_age, high_age = attrs["age_range"]
        upper_color, lower_color = rng.sample(attrs["colors"], 2)
        row = {
            "local_pid": pid,
            "split_group": "train" if pid < int(dataset["train_identities"]) else "test",
            "fictional": True,
            "adult_age": rng.randint(int(low_age), int(high_age)),
            "presentation": rng.choice(attrs["presentations"]),
            "body_build": rng.choice(attrs["body_builds"]),
            "height": rng.choice(attrs["heights"]),
            "hair_style": rng.choice(attrs["hair_styles"]),
            "hair_color": rng.choice(attrs["hair_colors"]),
            "upper_garment": rng.choice(attrs["upper_garments"]),
            "upper_color": upper_color,
            "lower_garment": rng.choice(attrs["lower_garments"]),
            "lower_color": lower_color,
            "footwear": rng.choice(attrs["footwear"]),
            "footwear_color": rng.choice(["black", "brown", "gray", "off-white"]),
            "carried_item": rng.choice(attrs["carried_items"]),
            "attribute_seed": rng.getrandbits(63),
        }
        row["identity_hash"] = sha256_bytes(canonical_json(row))
        rows.append(row)
    return rows


def make_cameras(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    seed = int(config["allocation"]["seed"])
    offset = int(config["dataset"]["camera_global_offset"])
    cameras = []
    for site_index, site in enumerate(config["sites"]):
        for view_index, view in enumerate(config["camera_views"]):
            local_camera = site_index * 3 + view_index
            rng = stable_rng(seed, "camera", local_camera)
            wb = [round(1.0 + rng.uniform(-0.07, 0.07), 5) for _ in range(3)]
            matrix = []
            for channel in range(3):
                matrix.append([
                    round((1.0 if channel == other else 0.0) + rng.uniform(-0.025, 0.025), 6)
                    for other in range(3)
                ])
            cameras.append({
                "local_camera": local_camera,
                "global_camera": offset + local_camera,
                "site_index": site_index,
                "site": site["key"],
                "site_label": site["label"],
                "environment": site["environment"],
                "lighting": site["lighting"],
                "view": view["key"],
                "view_label": view["label"],
                "yaw_deg": view["yaw_deg"],
                "pitch_deg": view["pitch_deg"],
                "focal_mm": view["focal_mm"],
                "isp": {
                    "white_balance_rgb": wb,
                    "color_matrix_rgb": matrix,
                    "exposure_ev": round(float(view["exposure_ev"]) + rng.uniform(-0.08, 0.08), 5),
                    "gamma": round(float(view["gamma"]) + rng.uniform(-0.025, 0.025), 5),
                    "radial_k1": round(float(view["radial_k1"]) + rng.uniform(-0.006, 0.006), 6),
                    "vignette": round(float(view["vignette"]) + rng.uniform(-0.015, 0.015), 5),
                    "poisson_scale": round(rng.uniform(24.0, 42.0), 4),
                    "gaussian_sigma": round(rng.uniform(1.2, 3.2), 4),
                    "motion_blur_px": round(float(view["motion_blur_px"]) + rng.uniform(0.0, 0.7), 4),
                    "jpeg_quality": int(view["jpeg_quality"]),
                },
                "processing_version": PROCESSING_VERSION,
            })
    return cameras


def _balanced_incidence(
    row_count: int,
    row_degree: int,
    column_degrees: Sequence[int],
    rng: random.Random,
) -> list[list[int]]:
    """Construct a simple bipartite incidence matrix with exact degrees."""
    if sum(column_degrees) != row_count * row_degree:
        raise ValueError("incidence degree sums differ")
    remaining = list(column_degrees)
    rows: list[list[int]] = []
    for row in range(row_count):
        tie = {column: rng.random() for column in range(len(remaining))}
        ordered = sorted(range(len(remaining)), key=lambda c: (-remaining[c], tie[c]))
        chosen = [column for column in ordered if remaining[column] > 0][:row_degree]
        if len(chosen) != row_degree:
            raise RuntimeError(f"allocation became infeasible at row {row}")
        for column in chosen:
            remaining[column] -= 1
        rows.append(sorted(chosen))
    if any(remaining):
        raise RuntimeError(f"allocation left nonzero column degrees: {remaining}")
    return rows


def _camera_targets(config: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    """Return exact site-visit and camera-block targets for the full set.

    With three cameras per site and exactly two selected cameras per visit,
    every site total must be even.  The closest possible distribution to the
    requested 121/122 blocks is 120x2, 121x22, 122x9.
    """
    seed = int(config["allocation"]["seed"])
    rng = stable_rng(seed, "target-layout")
    sites = list(range(11))
    rng.shuffle(sites)
    low_sites = set(sites[:2])
    site_visits = [181 if site in low_sites else 182 for site in range(11)]
    targets: list[int] = []
    for site, visits in enumerate(site_visits):
        values = [121, 121, 120 if visits == 181 else 122]
        stable_rng(seed, "target-site", site).shuffle(values)
        targets.extend(values)
    assert Counter(targets) == Counter({120: 2, 121: 22, 122: 9})
    return site_visits, targets


def allocate_identity_cameras(config: Mapping[str, Any]) -> tuple[dict[int, list[int]], list[int]]:
    dataset = config["dataset"]
    seed = int(config["allocation"]["seed"])
    site_visits, camera_targets = _camera_targets(config)
    pid_sites = _balanced_incidence(
        int(dataset["identities"]),
        int(dataset["sites_per_identity"]),
        site_visits,
        stable_rng(seed, "identity-sites"),
    )
    visits_by_site: dict[int, list[int]] = defaultdict(list)
    for pid, sites in enumerate(pid_sites):
        for site in sites:
            visits_by_site[site].append(pid)
    result: dict[int, list[int]] = defaultdict(list)
    for site in range(11):
        visit_pids = visits_by_site[site]
        rng = stable_rng(seed, "camera-omissions", site)
        rng.shuffle(visit_pids)
        target = camera_targets[site * 3 : site * 3 + 3]
        omissions = [len(visit_pids) - value for value in target]
        omitted_cameras = [camera for camera, count in enumerate(omissions) for _ in range(count)]
        rng.shuffle(omitted_cameras)
        if len(omitted_cameras) != len(visit_pids):
            raise RuntimeError("camera omission degrees are inconsistent")
        for pid, omitted in zip(visit_pids, omitted_cameras):
            result[pid].extend(site * 3 + camera for camera in range(3) if camera != omitted)
    normalized = {pid: sorted(cameras) for pid, cameras in result.items()}
    return normalized, camera_targets


def allocate_pilot_cameras(config: Mapping[str, Any]) -> dict[int, list[int]]:
    """Allocate 12 x 8 pilot views while covering every camera at least twice."""
    seed = int(config["allocation"]["seed"])
    rng = stable_rng(seed, "pilot-sites")
    high_sites = set(rng.sample(range(11), 4))
    site_visits = [5 if site in high_sites else 4 for site in range(11)]
    pid_sites = _balanced_incidence(12, 4, site_visits, rng)
    visits_by_site: dict[int, list[int]] = defaultdict(list)
    for pid, sites in enumerate(pid_sites):
        for site in sites:
            visits_by_site[site].append(pid)
    result: dict[int, list[int]] = defaultdict(list)
    for site in range(11):
        pids = visits_by_site[site]
        site_rng = stable_rng(seed, "pilot-cameras", site)
        site_rng.shuffle(pids)
        targets = [4, 3, 3] if len(pids) == 5 else [2, 3, 3]
        site_rng.shuffle(targets)
        omissions = [len(pids) - count for count in targets]
        omitted = [camera for camera, count in enumerate(omissions) for _ in range(count)]
        site_rng.shuffle(omitted)
        for pid, skip in zip(pids, omitted):
            result[pid].extend(site * 3 + camera for camera in range(3) if camera != skip)
    return {pid: sorted(cameras) for pid, cameras in result.items()}


def _orientation(frame: int, local_camera: int) -> str:
    if frame == 0:
        return "front"
    if frame == 1:
        return "left"
    if frame == 2:
        return "right"
    if frame == 3:
        return "back"
    return "left" if local_camera % 2 else "right"


def make_samples(config: Mapping[str, Any], cameras_by_pid: Mapping[int, Sequence[int]]) -> list[dict[str, Any]]:
    dataset = config["dataset"]
    seed = int(config["allocation"]["seed"])
    train_ids = int(dataset["train_identities"])
    frames_per_camera = int(dataset["frames_per_camera"])
    global_offset = int(dataset["camera_global_offset"])
    samples: list[dict[str, Any]] = []
    seq = 0
    for pid in range(int(dataset["identities"])):
        cameras = list(cameras_by_pid[pid])
        site_samples: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for camera in cameras:
            for frame in range(frames_per_camera):
                site_samples[camera // 3].append((camera, frame))
        occluded: set[tuple[int, int]] = set()
        query: set[tuple[int, int]] = set()
        for site, choices in site_samples.items():
            rng = stable_rng(seed, "occlusion", pid, site)
            if pid < train_ids:
                occluded.update(rng.sample(choices, 3))
            else:
                query.add(rng.choice(choices))
        for camera in cameras:
            for frame in range(frames_per_camera):
                key = (camera, frame)
                is_query = key in query
                is_occluded = key in occluded or is_query
                split = "train" if pid < train_ids else ("query" if is_query else "gallery")
                rng = stable_rng(seed, "sample", pid, camera, frame)
                sample = {
                    "sample_id": f"s{seq:05d}",
                    "local_pid": pid,
                    "global_pid": None,
                    "split": split,
                    "domain_local": int(dataset["domain_local"]),
                    "domain_global": int(dataset["domain_global"]),
                    "local_camera": camera,
                    "global_camera": global_offset + camera,
                    "site_index": camera // 3,
                    "frame": frame,
                    "orientation": _orientation(frame, camera),
                    "occluded": is_occluded,
                    "occluder": rng.choice(["railing", "bollard", "plain luggage cart", "low wall"]) if is_occluded else None,
                    "target_occlusion_ratio": round(rng.uniform(0.20, 0.50), 4) if is_occluded else 0.0,
                    "pose": rng.choice(["walking left foot forward", "walking right foot forward", "standing mid-step"]),
                    "crop_variant": rng.choice(["centered", "slightly left", "slightly right"]),
                    "generation_seed": rng.getrandbits(63),
                }
                samples.append(sample)
                seq += 1
    return samples


def make_pilot_samples(config: Mapping[str, Any], pilot_cameras: Mapping[int, Sequence[int]]) -> list[dict[str, Any]]:
    seed = int(config["allocation"]["seed"])
    offset = int(config["dataset"]["camera_global_offset"])
    rows = []
    for quality in config["model"]["pilot_qualities"]:
        for pid in range(12):
            for index, camera in enumerate(pilot_cameras[pid]):
                rng = stable_rng(seed, "pilot", pid, camera)
                occluded = index % 3 == 0
                rows.append({
                    "sample_id": f"pilot-{quality}-p{pid:03d}-c{camera:02d}",
                    "pilot_spec_id": f"pilot-p{pid:03d}-c{camera:02d}",
                    "quality": quality,
                    "local_pid": pid,
                    "global_pid": None,
                    "split": "pilot",
                    "domain_local": 0,
                    "domain_global": 5,
                    "local_camera": camera,
                    "global_camera": offset + camera,
                    "site_index": camera // 3,
                    "frame": index % 5,
                    "orientation": _orientation(index % 5, camera),
                    "occluded": occluded,
                    "occluder": "railing" if occluded else None,
                    "target_occlusion_ratio": round(rng.uniform(0.20, 0.50), 4) if occluded else 0.0,
                    "pose": rng.choice(["walking left foot forward", "walking right foot forward"]),
                    "crop_variant": rng.choice(["centered", "slightly left", "slightly right"]),
                    "generation_seed": rng.getrandbits(63),
                })
    return rows


def identity_prompt(identity: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    return (
        f"{config['prompt']['invariant_rules']} Create the canonical identity reference for one "
        f"{identity['adult_age']}-year-old fictional adult with {identity['presentation']} presentation, "
        f"{identity['height']} height, {identity['body_build']} build, {identity['hair_color']} "
        f"{identity['hair_style']} hair, a plain {identity['upper_color']} {identity['upper_garment']}, "
        f"plain {identity['lower_color']} {identity['lower_garment']}, {identity['footwear_color']} "
        f"{identity['footwear']}, and {identity['carried_item']}. Neutral front view, head to shoes visible, "
        "plain neutral studio background, natural anatomy."
    )


def anchor_view_prompt(orientation: str, config: Mapping[str, Any]) -> str:
    return (
        f"{config['prompt']['invariant_rules']} Using the first image as the exact identity reference, "
        f"show the same fictional adult in a neutral {orientation} full-body view. Preserve every "
        "identity and wardrobe attribute; plain neutral studio background."
    )


def plate_prompt(camera: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    return (
        f"Fixed empty surveillance-camera background plate of a {camera['site_label']}, "
        f"{camera['view_label']} perspective, {camera['lighting']}, documentary realism. "
        "No people, human reflections, readable text, logos, watermark, vehicles with legible plates, or border."
    )


def sample_prompt(
    sample: Mapping[str, Any],
    identity: Mapping[str, Any],
    camera: Mapping[str, Any],
    config: Mapping[str, Any],
    neutral_revision: bool = False,
) -> str:
    occlusion = ""
    if sample["occluded"]:
        occlusion = (
            f" A foreground {sample['occluder']} naturally covers {sample['target_occlusion_ratio']:.0%} "
            "of the lower or middle body, while the head, shoulders, and enough upper body remain identifiable."
        )
    revision = " Use neutral ordinary pedestrian wording." if neutral_revision else ""
    return (
        f"{config['prompt']['invariant_rules']} {config['prompt']['camera_rules']} "
        f"The first input image is the exact fictional identity and wardrobe reference. The second is the "
        f"empty fixed background for camera {camera['local_camera']}. Place that same adult in the plate at "
        f"{camera['site_label']}, viewed from {camera['view_label']}, {sample['pose']}, facing "
        f"{sample['orientation']}, {sample['crop_variant']} composition. One principal person only."
        f"{occlusion}{revision}"
    )


def validate_plan(
    config: Mapping[str, Any],
    identities: Sequence[Mapping[str, Any]],
    cameras: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = config["dataset"]["expected"]
    errors: list[str] = []
    split_counts = Counter(sample["split"] for sample in samples)
    for split in SPLITS:
        if split_counts[split] != int(expected[split]):
            errors.append(f"{split}: got {split_counts[split]}, expected {expected[split]}")
    if len(samples) != int(expected["total"]):
        errors.append(f"total: got {len(samples)}, expected {expected['total']}")
    if len(identities) != int(config["dataset"]["identities"]):
        errors.append("identity count mismatch")
    if len(cameras) != 33:
        errors.append("camera count mismatch")
    per_pid: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        per_pid[int(sample["local_pid"])].append(sample)
    for pid, rows in per_pid.items():
        if len(rows) != 40 or len({row["local_camera"] for row in rows}) != 8:
            errors.append(f"pid {pid}: expected 40 images and 8 cameras")
        site_counts = Counter(row["site_index"] for row in rows)
        if sorted(site_counts.values()) != [10, 10, 10, 10]:
            errors.append(f"pid {pid}: expected four sites with ten images each")
        if pid < int(config["dataset"]["train_identities"]):
            if sum(bool(row["occluded"]) for row in rows) != 12:
                errors.append(f"pid {pid}: expected 12 train occlusions")
        else:
            queries = [row for row in rows if row["split"] == "query"]
            galleries = [row for row in rows if row["split"] == "gallery"]
            if len(queries) != 4 or len(galleries) != 36 or not all(row["occluded"] for row in queries):
                errors.append(f"pid {pid}: invalid query/gallery allocation")
            for query in queries:
                other_camera_positives = sum(
                    gallery["local_camera"] != query["local_camera"] for gallery in galleries
                )
                if other_camera_positives < 32:
                    errors.append(f"pid {pid}: query lacks 32 cross-camera positives")
    per_camera = Counter(int(sample["local_camera"]) for sample in samples)
    histogram = Counter(per_camera.values())
    expected_hist = Counter({int(k): int(v) for k, v in config["allocation"]["expected_camera_count_histogram"].items()})
    if histogram != expected_hist:
        errors.append(f"camera histogram {dict(histogram)} != {dict(expected_hist)}")
    train_pids = {int(s["local_pid"]) for s in samples if s["split"] == "train"}
    test_pids = {int(s["local_pid"]) for s in samples if s["split"] in ("query", "gallery")}
    if train_pids & test_pids:
        errors.append("train/test PID overlap")
    return {
        "valid": not errors,
        "errors": errors,
        "counts": dict(split_counts),
        "camera_image_counts": dict(sorted(per_camera.items())),
        "camera_count_histogram": dict(sorted(histogram.items())),
    }


def validate_pilot(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    specs = {(s["pilot_spec_id"], s["local_camera"]) for s in samples}
    coverage = Counter(int(s["local_camera"]) for s in samples if s["quality"] == "low")
    errors = []
    if len(samples) != 192 or len(specs) != 96:
        errors.append("pilot must contain 96 specifications at two qualities")
    if set(coverage) != set(range(33)) or min(coverage.values(), default=0) < 2:
        errors.append("pilot must cover every camera at least twice per quality")
    return {"valid": not errors, "errors": errors, "coverage_per_quality": dict(sorted(coverage.items()))}


def make_batch_line(custom_id: str, endpoint: str, body: Mapping[str, Any]) -> dict[str, Any]:
    return {"custom_id": custom_id, "method": "POST", "url": endpoint, "body": dict(body)}


def reconcile_batch_output(
    payload: bytes | str,
    expected_custom_ids: Iterable[str],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    text = payload.decode() if isinstance(payload, bytes) else payload
    expected = set(expected_custom_ids)
    found: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = item.get("custom_id")
        if not isinstance(custom_id, str):
            raise ValueError(f"batch output line {line_number} has no custom_id")
        if custom_id in found:
            raise ValueError(f"duplicate batch output custom_id: {custom_id}")
        if custom_id not in expected:
            raise ValueError(f"unexpected batch output custom_id: {custom_id}")
        found[custom_id] = item
    return found, expected - set(found)


def classify_batch_item(item: Mapping[str, Any]) -> str:
    response = item.get("response") or {}
    if int(response.get("status_code", 0)) == 200:
        return "succeeded"
    error = item.get("error") or response.get("body", {}).get("error") or {}
    code = str(error.get("code", ""))
    kind = str(error.get("type", ""))
    if code == "moderation_blocked" or kind == "image_generation_user_error":
        return "revise_once"
    if code in {"rate_limit_exceeded", "server_error", "timeout"} or int(response.get("status_code", 0)) >= 500:
        return "retryable"
    return "terminal"


def response_usage(item: Mapping[str, Any]) -> dict[str, int]:
    body = (item.get("response") or {}).get("body") or {}
    usage = body.get("usage") or {}
    details = usage.get("input_tokens_details") or usage.get("input_tokens_detail") or {}
    return {
        "text_input_tokens": int(details.get("text_tokens", 0) or 0),
        "cached_text_input_tokens": int(details.get("cached_text_tokens", 0) or 0),
        "image_input_tokens": int(details.get("image_tokens", 0) or 0),
        "cached_image_input_tokens": int(details.get("cached_image_tokens", 0) or 0),
        "input_tokens_unclassified": int(usage.get("input_tokens", 0) or 0)
        - int(details.get("text_tokens", 0) or 0)
        - int(details.get("image_tokens", 0) or 0),
        "image_output_tokens": int(usage.get("output_tokens", 0) or 0),
    }


def usage_cost_usd(usage: Mapping[str, int], config: Mapping[str, Any]) -> float:
    price = config["pricing_usd_per_million_tokens"]["batch"]
    # Unclassified input is conservatively charged at the higher image rate.
    total = (
        int(usage.get("text_input_tokens", 0)) * float(price["text_input"])
        + int(usage.get("cached_text_input_tokens", 0)) * float(price["cached_text_input"])
        + (int(usage.get("image_input_tokens", 0)) + max(0, int(usage.get("input_tokens_unclassified", 0))))
        * float(price["image_input"])
        + int(usage.get("cached_image_input_tokens", 0)) * float(price["cached_image_input"])
        + int(usage.get("image_output_tokens", 0)) * float(price["image_output"])
    ) / 1_000_000.0
    return round(total, 8)


def aggregate_usage(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for row in rows:
        total.update({key: int(value) for key, value in row.items()})
    return dict(total)


def approval_payload(
    report: Mapping[str, Any], quality: str, max_usd: float, config: Mapping[str, Any]
) -> dict[str, Any]:
    if quality not in config["model"]["pilot_qualities"]:
        raise ValueError("quality must be one of the piloted qualities")
    if max_usd <= 0 or not math.isfinite(max_usd):
        raise ValueError("max_usd must be a positive finite number")
    if not report.get("pilot_gate_passed"):
        raise ValueError("pilot report has not passed every automatic and manual gate")
    report_hash = sha256_bytes(canonical_json(report))
    return {
        "schema_version": SCHEMA_VERSION,
        "approved_at": utc_now(),
        "model_id": config["model"]["api_id"],
        "catalog_snapshot": config["model"]["catalog_snapshot"],
        "quality": quality,
        "max_usd": round(float(max_usd), 2),
        "report_sha256": report_hash,
        "config_sha256": config_sha256(config),
    }


def verify_approval(approval: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if approval.get("model_id") != config["model"]["api_id"]:
        raise ValueError("approval model ID differs from config")
    if approval.get("catalog_snapshot") != config["model"]["catalog_snapshot"]:
        raise ValueError("approval catalog snapshot differs from config")
    if approval.get("config_sha256") != config_sha256(config):
        raise ValueError("configuration changed after approval")
    if approval.get("quality") not in config["model"]["pilot_qualities"]:
        raise ValueError("approval quality was not piloted")
    if float(approval.get("max_usd", 0)) <= 0:
        raise ValueError("approval has no valid cost ceiling")


def perceptual_hash(path: Path) -> str:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("opencv-python and numpy are required for image QA") from exc
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot decode {path}")
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    median = float(np.median(dct[1:]))
    return f"{sum((1 << index) for index, value in enumerate(dct.flat) if value > median):016x}"


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


_TORCHVISION_QA_MODELS: dict[str, tuple[Any, Any, Any, str]] = {}


def _xyxy_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    return intersection / max(left_area + right_area - intersection, 1e-9)


def _nms_person_candidates(
    boxes: Sequence[Sequence[float]], scores: Sequence[float], iou_threshold: float = PERSON_NMS_IOU
) -> list[int]:
    """Deterministic NMS for the small set of person detections returned per image."""
    order = sorted(range(len(boxes)), key=lambda index: (-float(scores[index]), index))
    kept: list[int] = []
    for index in order:
        if all(_xyxy_iou(boxes[index], boxes[other]) <= iou_threshold for other in kept):
            kept.append(index)
    return kept


def _load_torchvision_qa_model(kind: str) -> tuple[Any, Any, Any, str]:
    if kind in _TORCHVISION_QA_MODELS:
        return _TORCHVISION_QA_MODELS[kind]
    try:
        import torch
        if kind == "person":
            from torchvision.models.detection import (
                SSDLite320_MobileNet_V3_Large_Weights,
                ssdlite320_mobilenet_v3_large,
            )

            weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            model = ssdlite320_mobilenet_v3_large(weights=weights)
        elif kind == "pose":
            from torchvision.models.detection import (
                KeypointRCNN_ResNet50_FPN_Weights,
                keypointrcnn_resnet50_fpn,
            )

            weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
            model = keypointrcnn_resnet50_fpn(weights=weights)
        else:  # pragma: no cover - internal programming error
            raise ValueError(f"unknown QA model kind: {kind}")
    except ImportError as exc:  # pragma: no cover - dependency preflight
        raise RuntimeError("torch and torchvision are required for robust person QA") from exc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
    value = (model, weights.transforms(), device, weights.name)
    _TORCHVISION_QA_MODELS[kind] = value
    return value


def _torchvision_input(image: Any, transform: Any, device: Any) -> Any:
    from PIL import Image

    rgb = image[:, :, ::-1]
    return transform(Image.fromarray(rgb)).to(device)


def _person_detection(image: Any) -> dict[str, Any]:
    import torch

    model, transform, device, weights_name = _load_torchvision_qa_model("person")
    tensor = _torchvision_input(image, transform, device)
    with torch.inference_mode():
        output = model([tensor])[0]
    mask = (output["labels"] == 1) & (output["scores"] >= PERSON_SCORE_MIN)
    boxes = output["boxes"][mask].detach().cpu().tolist()
    scores = output["scores"][mask].detach().cpu().tolist()
    kept = _nms_person_candidates(boxes, scores)
    boxes = [boxes[index] for index in kept]
    scores = [float(scores[index]) for index in kept]
    bbox = None
    if boxes:
        x0, y0, x1, y1 = boxes[0]
        bbox = tuple(round(value) for value in (x0, y0, x1 - x0, y1 - y0))
    return {
        "bbox": bbox,
        "count": len(boxes),
        "candidate_count_before_nms": int(mask.sum().item()),
        "principal_score": scores[0] if scores else 0.0,
        "backend": "torchvision_ssdlite320_mobilenet_v3_large",
        "weights": weights_name,
        "score_threshold": PERSON_SCORE_MIN,
        "nms_iou_threshold": PERSON_NMS_IOU,
    }


def _pose_geometry(
    image: Any, occluded: bool, bbox: tuple[int, int, int, int] | None
) -> dict[str, Any]:
    """Fail-closed COCO keypoint check for the required visible body regions."""
    if bbox is None:
        return {"available": True, "backend": "torchvision_keypointrcnn_resnet50_fpn",
                "pass": False, "reason": "person_bbox_unavailable"}
    import torch

    model, transform, device, weights_name = _load_torchvision_qa_model("pose")
    tensor = _torchvision_input(image, transform, device)
    with torch.inference_mode():
        output = model([tensor])[0]
    mask = (output["labels"] == 1) & (output["scores"] >= POSE_SCORE_MIN)
    indices = torch.where(mask)[0].detach().cpu().tolist()
    x, y, width, height = bbox
    principal_xyxy = (x, y, x + width, y + height)
    matches = [
        (_xyxy_iou(principal_xyxy, output["boxes"][index].detach().cpu().tolist()),
         float(output["scores"][index].item()), index)
        for index in indices
    ]
    if not matches:
        return {"available": True, "backend": "torchvision_keypointrcnn_resnet50_fpn",
                "weights": weights_name, "pass": False, "reason": "pose_not_detected"}
    match_iou, pose_score, selected = max(matches)
    if match_iou < POSE_MATCH_IOU_MIN:
        return {"available": True, "backend": "torchvision_keypointrcnn_resnet50_fpn",
                "weights": weights_name, "pass": False, "reason": "pose_does_not_match_principal_person",
                "match_iou": match_iou, "pose_score": pose_score}
    if "keypoints_scores" in output:
        scores = output["keypoints_scores"][selected].detach().cpu().tolist()
    else:  # pragma: no cover - compatibility with older torchvision
        scores = output["keypoints"][selected, :, 2].detach().cpu().tolist()

    required = {
        "head": (0,),
        "shoulders": (5, 6),
        ("waist" if occluded else "feet"): ((11, 12) if occluded else (15, 16)),
    }
    region_scores = {name: min(float(scores[index]) for index in members) for name, members in required.items()}
    regions = {name: score >= KEYPOINT_VISIBILITY_LOGIT_MIN for name, score in region_scores.items()}
    passed = all(regions.values())
    return {
        "available": True,
        "backend": "torchvision_keypointrcnn_resnet50_fpn",
        "weights": weights_name,
        "pass": passed,
        "regions": regions,
        "region_scores": region_scores,
        "visibility_logit_threshold": KEYPOINT_VISIBILITY_LOGIT_MIN,
        "match_iou": match_iou,
        "pose_score": pose_score,
        "reason": None if passed else "required_landmarks_not_visible",
    }


def _crop_resize_person(
    image: Any, bbox: tuple[int, int, int, int] | None, margin: float
) -> tuple[Any, dict[str, Any]]:
    """Tightly crop the detected person and resize like standard ReID inputs."""
    import cv2

    height, width = image.shape[:2]
    if bbox is None:
        # The fallback is deterministic and deliberately fails geometry QA.
        crop_width = min(width, round(height * 0.58))
        x0 = max(0, (width - crop_width) // 2)
        x1, y0, y1 = x0 + crop_width, 0, height
        framing = {
            "policy": "tight_bbox_direct_resize",
            "crop_box": [x0, y0, x1, y1],
            "bbox_width_fill": 0.0,
            "bbox_height_fill": 0.0,
            "margin": margin,
        }
    else:
        x, y, w, h = bbox
        x0 = max(0, math.floor(x - w * margin))
        x1 = min(width, math.ceil(x + w * (1 + margin)))
        y0 = max(0, math.floor(y - h * margin))
        y1 = min(height, math.ceil(y + h * (1 + margin)))
        framing = {
            "policy": "tight_bbox_direct_resize",
            "crop_box": [x0, y0, x1, y1],
            "bbox_width_fill": float(w / max(x1 - x0, 1)),
            "bbox_height_fill": float(h / max(y1 - y0, 1)),
            "margin": margin,
        }
    crop = image[y0:y1, x0:x1]
    resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_AREA)
    return resized, framing


def _motion_kernel(length: float, angle_degrees: float) -> Any:
    import cv2
    import numpy as np

    size = max(1, int(round(length)))
    if size <= 1:
        return np.ones((1, 1), np.float32)
    if size % 2 == 0:
        size += 1
    kernel = np.zeros((size, size), np.float32)
    kernel[size // 2, :] = 1.0
    center = (size / 2 - 0.5, size / 2 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (size, size))
    return kernel / max(float(kernel.sum()), 1e-8)


def process_image(
    raw_path: Path,
    final_path: Path,
    camera: Mapping[str, Any],
    sample: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Crop, normalize, and apply a deterministic per-camera imaging model."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("opencv-python and numpy are required for image processing") from exc
    image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if image is None:
        return {"decode": False, "accepted": False, "reason": "decode_failed"}
    raw_height, raw_width = image.shape[:2]
    detection = _person_detection(image)
    bbox = detection["bbox"]
    person_count = int(detection["count"])
    pose_geometry = _pose_geometry(image, bool(sample["occluded"]), bbox)
    rgb = image[:, :, ::-1].astype(np.float32) / 255.0
    isp = camera["isp"]
    rgb *= np.asarray(isp["white_balance_rgb"], dtype=np.float32)[None, None, :]
    rgb = np.einsum("hwc,dc->hwd", rgb, np.asarray(isp["color_matrix_rgb"], dtype=np.float32))
    rgb *= 2.0 ** float(isp["exposure_ev"])
    rgb = np.clip(rgb, 0.0, 1.0) ** (1.0 / float(isp["gamma"]))
    height, width = rgb.shape[:2]
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    radius2 = xx * xx + yy * yy
    rgb *= np.clip(1.0 - float(isp["vignette"]) * radius2, 0.72, 1.0)[:, :, None]
    rng = np.random.default_rng(int(sample["generation_seed"]) & ((1 << 63) - 1))
    poisson_scale = float(isp["poisson_scale"])
    rgb = rng.poisson(np.clip(rgb, 0, 1) * poisson_scale) / poisson_scale
    rgb += rng.normal(0, float(isp["gaussian_sigma"]) / 255.0, rgb.shape)
    bgr = np.clip(rgb[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
    k1 = float(isp["radial_k1"])
    camera_matrix = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], np.float32)
    bgr = cv2.undistort(bgr, camera_matrix, np.array([k1, 0, 0, 0], np.float32))
    angle = (int(sample["generation_seed"]) % 41) - 20
    kernel = _motion_kernel(float(isp["motion_blur_px"]), angle)
    bgr = cv2.filter2D(bgr, -1, kernel)
    bgr, framing = _crop_resize_person(
        bgr, bbox, float(config["dataset"]["bbox_margin"])
    )
    bgr = bgr.astype(np.uint8)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(final_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(isp["jpeg_quality"])])
    if not ok:
        return {"decode": True, "accepted": False, "reason": "jpeg_write_failed"}
    principal_person_detected = bbox is not None and person_count == 1
    framing_min = float(config["dataset"]["framing_fill_min"])
    framing["minimum_bbox_fill"] = framing_min
    framing["pass"] = (
        bbox is not None
        and float(framing["bbox_width_fill"]) >= framing_min
        and float(framing["bbox_height_fill"]) >= framing_min
    )
    geometry = principal_person_detected and bool(pose_geometry["pass"]) and bool(framing["pass"])
    if bbox is None:
        reason = "person_not_detected"
    elif person_count != 1:
        reason = "multiple_people_detected"
    elif not pose_geometry["pass"]:
        reason = str(pose_geometry.get("reason") or "required_landmarks_not_visible")
    elif not framing["pass"]:
        reason = "person_framing_out_of_range"
    else:
        reason = None
    return {
        "decode": True,
        "raw_size": [raw_width, raw_height],
        "final_size": [128, 256],
        "person_bbox": list(bbox) if bbox else None,
        "person_detection_count": person_count,
        "person_detection": detection,
        "principal_person_detected": principal_person_detected,
        "pose_geometry": pose_geometry,
        "framing": framing,
        "geometry_pass": geometry,
        "processing_version": PROCESSING_VERSION,
        "processing_seed": int(sample["generation_seed"]),
        "final_sha256": sha256_file(final_path),
        "phash": perceptual_hash(final_path),
        "accepted": geometry,
        "reason": reason,
    }


def validate_synthetic_manifest(
    root: Path,
    expected: Mapping[str, int] | None = None,
    require_accepted: bool = True,
) -> dict[str, Any]:
    """Validate the standalone accepted dataset before unified integration."""
    manifest_path = root / "manifest.jsonl"
    rows = read_jsonl(manifest_path)
    errors: list[str] = []
    accepted = [row for row in rows if row.get("qa_status") == "accepted"]
    if require_accepted and len(accepted) != len(rows):
        errors.append("manifest contains non-accepted rows")
    selected = accepted if require_accepted else rows
    counts = Counter(str(row.get("split")) for row in selected)
    if expected:
        for split in SPLITS:
            if counts[split] != int(expected[split]):
                errors.append(f"{split}: got {counts[split]}, expected {expected[split]}")
    seen_samples: set[str] = set()
    seen_sha: set[str] = set()
    per_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    resolved: list[tuple[dict[str, Any], Path]] = []
    for index, row in enumerate(selected):
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in seen_samples:
            errors.append(f"row {index}: duplicate or missing sample_id")
        seen_samples.add(sample_id)
        relative = row.get("final_path")
        if not isinstance(relative, str):
            errors.append(f"{sample_id}: missing final_path")
            continue
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            errors.append(f"{sample_id}: final_path escapes dataset root")
            continue
        if not path.is_file():
            errors.append(f"{sample_id}: image is missing")
            continue
        if path.name != f"{sample_id}.jpg":
            errors.append(f"{sample_id}: final filename is not canonical")
        try:
            from PIL import Image

            with Image.open(path) as image:
                if image.size != (128, 256) or image.format != "JPEG":
                    errors.append(
                        f"{sample_id}: expected 128x256 JPEG, got {image.size} {image.format}"
                    )
        except (ImportError, OSError) as exc:
            errors.append(f"{sample_id}: final image decode failed: {exc}")
        actual_sha = sha256_file(path)
        if actual_sha != row.get("final_sha256"):
            errors.append(f"{sample_id}: SHA-256 mismatch")
        if actual_sha in seen_sha:
            errors.append(f"{sample_id}: duplicate image SHA-256")
        seen_sha.add(actual_sha)
        framing = row.get("qa", {}).get("framing", {})
        if not framing.get("pass"):
            errors.append(f"{sample_id}: tight-crop framing QA did not pass")
        if framing.get("policy") != "tight_bbox_direct_resize":
            errors.append(f"{sample_id}: unexpected framing policy")
        if row.get("processing_version") != PROCESSING_VERSION:
            errors.append(f"{sample_id}: processing version is not {PROCESSING_VERSION}")
        pid = int(row.get("local_pid", -1))
        camera = int(row.get("local_camera", -1))
        if not 0 <= pid < 500 or not 0 <= camera < 33:
            errors.append(f"{sample_id}: PID/camera outside SyntheticReID33 range")
        if int(row.get("domain_global", -1)) != 5 or int(row.get("global_camera", -1)) != 33 + camera:
            errors.append(f"{sample_id}: expected d05 and global camera {33 + camera}")
        per_pid[pid].append(row)
        resolved.append((row, path))
    if expected and int(expected.get("total", 0)) == 20000:
        train_pids = {pid for pid, items in per_pid.items() if any(r["split"] == "train" for r in items)}
        test_pids = set(per_pid) - train_pids
        if len(train_pids) != 400 or len(test_pids) != 100 or train_pids & test_pids:
            errors.append("expected 400 identity-disjoint train IDs and 100 test IDs")
        camera_histogram = Counter(
            Counter(int(row["local_camera"]) for row in selected).values()
        )
        if camera_histogram != Counter({600: 2, 605: 22, 610: 9}):
            errors.append(f"unexpected camera image-count histogram: {dict(camera_histogram)}")
        for pid, items in per_pid.items():
            if len(items) != 40 or len({int(r["local_camera"]) for r in items}) != 8:
                errors.append(f"pid {pid}: expected 40 accepted images across 8 cameras")
            train_items = [row for row in items if row["split"] == "train"]
            queries = [row for row in items if row["split"] == "query"]
            galleries = [row for row in items if row["split"] == "gallery"]
            if train_items and sum(bool(row.get("occluded")) for row in train_items) != 12:
                errors.append(f"pid {pid}: expected 12 occluded train images")
            if queries and (len(queries) != 4 or not all(row.get("occluded") for row in queries)):
                errors.append(f"pid {pid}: expected four occluded queries")
            if galleries and (len(galleries) != 36 or any(row.get("occluded") for row in galleries)):
                errors.append(f"pid {pid}: expected 36 clean gallery images")
            for query in (row for row in items if row["split"] == "query"):
                positives = sum(
                    row["split"] == "gallery" and row["local_camera"] != query["local_camera"]
                    for row in items
                )
                if positives < 32:
                    errors.append(f"pid {pid}: query lacks cross-camera positives")
        for row in selected:
            embedding = row.get("qa", {}).get("embedding", {})
            if not all(embedding.get(model) for model in ("vit", "osnet")):
                errors.append(f"{row.get('sample_id')}: missing full embedding QA")
                break
        full_report_path = root / "qa" / "full_report.json"
        if not full_report_path.exists():
            errors.append("full automatic QA report is missing")
        else:
            full_report = json.loads(full_report_path.read_text(encoding="utf-8"))
            if not full_report.get("complete") or int(full_report.get("accepted", 0)) != 20000:
                errors.append("full automatic QA has not accepted exactly 20,000 images")
        review_path = root / "qa" / "full_review.json"
        if not review_path.exists():
            errors.append("stratified 5% full_review.json is missing")
        else:
            review = json.loads(review_path.read_text(encoding="utf-8"))
            review_ok = (
                bool(review.get("complete"))
                and float(review.get("sample_fraction", 0)) >= 0.05
                and bool(review.get("reviewed_all_boundary_cases"))
                and float(review.get("identity_consistency_rate", 0)) >= 0.95
                and int(review.get("major_anatomy_failures", 0)) == 0
                and int(review.get("text_logo_watermark_failures", 0)) == 0
            )
            if not review_ok:
                errors.append("full manual review gate has not passed")
    return {
        "valid": not errors,
        "errors": errors,
        "counts": dict(counts),
        "rows": resolved,
        "unique_pids": len(per_pid),
        "unique_cameras": len({int(row["local_camera"]) for row in selected}),
    }
