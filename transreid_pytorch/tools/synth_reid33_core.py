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
PROCESSING_VERSION = "synth-reid33-isp-v24-high-confidence-knee-support-qa"
CAMERA_GEOMETRY_VERSION = "synth-reid33-camera-geometry-v1"
PROMPT_VERSION = "synth-reid33-prompt-v3-eight-yaw-camera-pitch"
SPLITS = ("train", "query", "gallery")
BODY_YAW_DEGREES = (0, 45, 90, 135, 180, -135, -90, -45)
ANCHOR_ORIENTATIONS = {"front", "left", "right", "back"}
PERSON_SCORE_MIN = 0.35
PERSON_NMS_IOU = 0.35
PERSON_NMS_CONTAINMENT = 0.85
PERSON_FRAGMENT_SCORE_MAX = 0.45
PERSON_FRAGMENT_OVERLAP_MIN = 0.50
PERSON_FRAGMENT_TOP_FRACTION_MIN = 0.40
PERSON_SPLIT_BODY_SCORE_MAX = 0.50
PERSON_SPLIT_BODY_IOU_MIN = 0.30
PERSON_SPLIT_BODY_OVERLAP_MIN = 0.75
PERSON_TINY_SECONDARY_SCORE_MAX = 0.50
PERSON_TINY_SECONDARY_WIDTH_FRACTION_MAX = 0.03
PERSON_TINY_SECONDARY_HEIGHT_FRACTION_MAX = 0.08
PERSON_CART_FRAGMENT_SCORE_MAX = 0.60
PERSON_CART_FRAGMENT_IOU_MIN = 0.20
PERSON_CART_FRAGMENT_OVERLAP_MIN = 0.70
PERSON_CART_FRAGMENT_TOP_FRACTION_MIN = 0.15
PERSON_CART_FRAGMENT_HIGH_OCCLUSION_RATIO_MIN = 0.45
PERSON_CART_FRAGMENT_HIGH_OCCLUSION_OVERLAP_MIN = 0.68
PERSON_CART_FRAGMENT_HIGH_OCCLUSION_POSE_SCORE_MIN = 0.99
POSE_SCORE_MIN = 0.70
POSE_MATCH_IOU_MIN = 0.30
POSE_DETECTOR_FALLBACK_SCORE_MIN = 0.99
POSE_DETECTOR_FALLBACK_WIDTH_FRACTION_MIN = 0.05
POSE_DETECTOR_FALLBACK_HEIGHT_FRACTION_MIN = 0.15
POSE_DETECTOR_FALLBACK_FRACTION_MAX = 0.95
KEYPOINT_VISIBILITY_LOGIT_MIN = 0.0
SECONDARY_ANKLE_LOGIT_MIN = -5.0
MODERATE_ANKLE_LOGIT_MIN = -2.0
STRONG_ANKLE_FALLBACK_LOGIT_MIN = 5.0
LONGITUDINAL_ANKLE_LOGIT_MIN = -3.0
KNEE_SUPPORT_LOGIT_MIN = 5.0
HIGH_CONFIDENCE_KNEE_SUPPORT_LOGIT_MIN = 4.9
SECONDARY_HIP_LOGIT_MIN = -1.0
HIGH_OCCLUSION_HIP_LOGIT_MIN = -2.0
HIGH_OCCLUSION_HIP_SUPPORT_LOGIT_MIN = -1.0
HIGH_OCCLUSION_REGION_SUPPORT_LOGIT_MIN = 5.0
HIGH_OCCLUSION_POSE_SCORE_MIN = 0.99
HIP_OCCLUSION_BOUNDARY_FRACTION_MIN = 0.80
BOLLARD_OCCLUSION_POSE_SCORE_MIN = 0.95
BOLLARD_HIP_PAIR_DISTANCE_FRACTION_MAX = 0.15
RAILING_OCCLUSION_HIP_LOGIT_MIN = -2.5
RAILING_OCCLUSION_HIP_SUPPORT_LOGIT_MIN = -0.5
BACK_VIEW_HEAD_LOGIT_MIN = -1.5
MODERATE_BACK_VIEW_HEAD_LOGIT_MIN = -1.0
BACK_VIEW_SHOULDER_SUPPORT_LOGIT_MIN = 5.0
HEAD_UPPER_BODY_FRACTION_MAX = 0.40
DEEP_BACK_HEAD_UPPER_BODY_FRACTION_MAX = 0.20
ANKLE_LOWER_BODY_FRACTION_MIN = 0.65
ANKLE_PAIR_DISTANCE_FRACTION_MAX = 0.12
ANKLE_KNEE_DISTANCE_FRACTION_MAX = 0.08
ANKLE_KNEE_HIGH_CONF_DISTANCE_FRACTION_MAX = 0.10
ANKLE_VERTICAL_SEPARATION_FRACTION_MIN = 0.10
POSE_BBOX_EXPANSION_IOU_MIN = 0.25
POSE_BBOX_EXPANSION_CONTAINMENT_MIN = 0.85
POSE_BBOX_EXPANSION_SCORE_MIN = 0.99
POSE_BBOX_EXPANSION_AREA_RATIO_MAX = 4.0
POSE_BBOX_EXPANSION_CENTER_DISTANCE_MAX = 0.35
HIP_LOWER_BODY_FRACTION_MIN = 0.45
KEYPOINT_BBOX_TOLERANCE_FRACTION = 0.05
PERSON_BOTTOM_CLEARANCE_FRACTION_MIN = 0.005
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
    if config["model"].get("quality") != "low":
        raise ValueError("every image request is cost-locked to low quality")
    expected_model_sizes = {
        "anchor_size": "1024x1536",
        "sample_size": "576x1152",
        "reference_profile": "native_quarter",
        "reference_anchor_size": "256x384",
        "reference_plate_size": "144x288",
        "reference_jpeg_quality": 95,
        "reference_jpeg_subsampling": 0,
    }
    for key, expected in expected_model_sizes.items():
        if config["model"].get(key) != expected:
            raise ValueError(
                f"model.{key} must be {expected!r} for the approved native_quarter profile"
            )
    if config.get("prompt", {}).get("version") != PROMPT_VERSION:
        raise ValueError(f"prompt.version must be {PROMPT_VERSION}")
    if len(config["sites"]) != 11 or len(config["camera_views"]) != 3:
        raise ValueError("configuration must expand to 11 sites x 3 cameras")
    geometry = config.get("camera_geometry") or {}
    if geometry.get("version") != CAMERA_GEOMETRY_VERSION:
        raise ValueError(f"camera_geometry.version must be {CAMERA_GEOMETRY_VERSION}")
    if geometry.get("pitch_convention") != "negative_is_down":
        raise ValueError("camera pitch convention must be negative_is_down")
    sensor_height = float(geometry.get("sensor_height_mm", 0))
    if sensor_height <= 0:
        raise ValueError("camera_geometry.sensor_height_mm must be positive")
    horizon_range = geometry.get("horizon_fraction_range") or []
    if (
        len(horizon_range) != 2
        or not 0 <= float(horizon_range[0]) < float(horizon_range[1]) <= 1
    ):
        raise ValueError("camera_geometry.horizon_fraction_range must be inside [0, 1]")
    for view in config["camera_views"]:
        pitch = float(view.get("pitch_deg", 0))
        pitch_jitter = float(view.get("pitch_jitter_deg", 0))
        height = float(view.get("mounting_height_m", 0))
        height_jitter = float(view.get("mounting_height_jitter_m", 0))
        focal = float(view.get("focal_mm", 0))
        if not (-45 < pitch < 0) or pitch_jitter <= 0 or pitch + pitch_jitter >= 0:
            raise ValueError(
                f"camera view {view.get('key')} must remain downward through its pitch jitter range"
            )
        if height <= 0 or height_jitter <= 0 or height - height_jitter < 2.0:
            raise ValueError(
                f"camera view {view.get('key')} must remain at least 2m above the walking surface"
            )
        if focal <= 0:
            raise ValueError(f"camera view {view.get('key')} must have a positive focal length")
    rotation = config.get("body_rotation") or {}
    views = rotation.get("views") or []
    if tuple(int(view.get("degrees")) for view in views) != BODY_YAW_DEGREES:
        raise ValueError(f"body_rotation.views must use the ordered yaw bins {BODY_YAW_DEGREES}")
    if len({str(view.get("label")) for view in views}) != len(BODY_YAW_DEGREES):
        raise ValueError("body_rotation view labels must be unique")
    if any(str(view.get("anchor")) not in ANCHOR_ORIENTATIONS for view in views):
        raise ValueError("every body_rotation view must reference a cardinal anchor")
    if int(rotation.get("pilot_identities", 0)) != 12:
        raise ValueError("the body-rotation pilot must use the existing 12 pilot identities")
    if int(rotation.get("pilot_samples_per_identity", 0)) != len(BODY_YAW_DEGREES):
        raise ValueError("the body-rotation pilot must contain one sample per yaw and identity")
    if int(config["batch"].get("max_attempts", 0)) != 1:
        raise ValueError("automatic API retries are disabled; batch.max_attempts must be 1")
    if float(config["batch"].get("retry_reserve_fraction", -1)) != 0.0:
        raise ValueError("automatic API retries are disabled; retry_reserve_fraction must be 0")
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
            geometry_rng = stable_rng(seed, "camera_geometry", local_camera)
            pitch_deg = round(
                float(view["pitch_deg"])
                + geometry_rng.uniform(
                    -float(view["pitch_jitter_deg"]), float(view["pitch_jitter_deg"])
                ),
                2,
            )
            mounting_height_m = round(
                float(view["mounting_height_m"])
                + geometry_rng.uniform(
                    -float(view["mounting_height_jitter_m"]),
                    float(view["mounting_height_jitter_m"]),
                ),
                2,
            )
            pitch_down_deg = -pitch_deg
            sensor_height_mm = float(config["camera_geometry"]["sensor_height_mm"])
            horizon_y_fraction = round(
                0.5
                - math.tan(math.radians(pitch_down_deg))
                * float(view["focal_mm"])
                / sensor_height_mm,
                4,
            )
            camera_geometry = {
                "version": config["camera_geometry"]["version"],
                "pitch_convention": config["camera_geometry"]["pitch_convention"],
                "base_pitch_deg": float(view["pitch_deg"]),
                "pitch_deg": pitch_deg,
                "pitch_down_deg": round(pitch_down_deg, 2),
                "mounting_height_m": mounting_height_m,
                "horizon_y_fraction": horizon_y_fraction,
                "focal_mm": float(view["focal_mm"]),
                "yaw_deg": float(view["yaw_deg"]),
            }
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
                "pitch_deg": pitch_deg,
                "focal_mm": view["focal_mm"],
                "mounting_height_m": mounting_height_m,
                "horizon_y_fraction": horizon_y_fraction,
                "geometry": camera_geometry,
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


def _body_rotation_views(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(view) for view in config["body_rotation"]["views"]]


def _body_rotation_fields(view: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "orientation": str(view["label"]),
        "anchor_orientation": str(view["anchor"]),
        "body_yaw_deg": int(view["degrees"]),
        "body_yaw_label": str(view["label"]),
        "body_yaw_prompt": str(view["prompt"]),
    }


def make_samples(config: Mapping[str, Any], cameras_by_pid: Mapping[int, Sequence[int]]) -> list[dict[str, Any]]:
    dataset = config["dataset"]
    seed = int(config["allocation"]["seed"])
    train_ids = int(dataset["train_identities"])
    frames_per_camera = int(dataset["frames_per_camera"])
    global_offset = int(dataset["camera_global_offset"])
    rotation_views = _body_rotation_views(config)
    yaw_count = len(rotation_views)
    samples: list[dict[str, Any]] = []
    seq = 0
    for pid in range(int(dataset["identities"])):
        cameras = list(cameras_by_pid[pid])
        rotation_offset = stable_rng(seed, "body-yaw-offset", pid).randrange(yaw_count)
        rotation_by_sample: dict[tuple[int, int], tuple[int, dict[str, Any]]] = {}
        site_samples: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for camera_rank, camera in enumerate(cameras):
            for frame in range(frames_per_camera):
                key = (camera, frame)
                yaw_index = (camera_rank * frames_per_camera + frame + rotation_offset) % yaw_count
                rotation_by_sample[key] = (yaw_index, rotation_views[yaw_index])
                site_samples[camera // 3].append(key)
        occluded: set[tuple[int, int]] = set()
        query: set[tuple[int, int]] = set()
        query_base = stable_rng(seed, "query-yaw", pid).randrange(yaw_count)
        for site_order, (site, choices) in enumerate(sorted(site_samples.items())):
            rng = stable_rng(seed, "occlusion", pid, site)
            if pid < train_ids:
                occluded.update(rng.sample(choices, 3))
            else:
                target_yaw = (query_base + site_order * 2) % yaw_count
                target_choices = [
                    choice for choice in choices if rotation_by_sample[choice][0] == target_yaw
                ]
                if not target_choices:
                    raise RuntimeError(f"pid {pid} site {site}: no sample for query yaw {target_yaw}")
                query.add(rng.choice(target_choices))
        for camera in cameras:
            for frame in range(frames_per_camera):
                key = (camera, frame)
                _yaw_index, rotation_view = rotation_by_sample[key]
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
                    **_body_rotation_fields(rotation_view),
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


def make_rotation_pilot_samples(
    config: Mapping[str, Any], pilot_cameras: Mapping[int, Sequence[int]]
) -> list[dict[str, Any]]:
    """Build a low-cost 12-ID pilot containing every planned body yaw once per ID."""
    seed = int(config["allocation"]["seed"])
    offset = int(config["dataset"]["camera_global_offset"])
    rotation = config["body_rotation"]
    views = _body_rotation_views(config)
    rows = []
    for pid in range(int(rotation["pilot_identities"])):
        cameras = list(pilot_cameras[pid])
        yaw_offset = stable_rng(seed, "rotation-pilot-yaw", pid).randrange(len(views))
        for index, camera in enumerate(cameras):
            view = views[(index + yaw_offset) % len(views)]
            rng = stable_rng(seed, "rotation-pilot", pid, camera)
            rows.append({
                "sample_id": f"rotation-pilot-p{pid:03d}-c{camera:02d}",
                "quality": str(config["model"]["quality"]),
                "local_pid": pid,
                "global_pid": None,
                "split": "rotation_pilot",
                "domain_local": 0,
                "domain_global": 5,
                "local_camera": camera,
                "global_camera": offset + camera,
                "site_index": camera // 3,
                "frame": index % int(config["dataset"]["frames_per_camera"]),
                **_body_rotation_fields(view),
                "occluded": False,
                "occluder": None,
                "target_occlusion_ratio": 0.0,
                "pose": rng.choice([
                    "walking left foot forward",
                    "walking right foot forward",
                    "standing mid-step",
                ]),
                "crop_variant": rng.choice(["centered", "slightly left", "slightly right"]),
                "generation_seed": rng.getrandbits(63),
            })
    return rows


def make_pilot_samples(config: Mapping[str, Any], pilot_cameras: Mapping[int, Sequence[int]]) -> list[dict[str, Any]]:
    seed = int(config["allocation"]["seed"])
    offset = int(config["dataset"]["camera_global_offset"])
    quality = str(config["model"]["quality"])
    rows = []
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
    geometry = camera["geometry"]
    horizon_percent = round(float(geometry["horizon_y_fraction"]) * 100)
    return (
        f"Fixed empty surveillance-camera background plate of a {camera['site_label']}, "
        f"{camera['view_label']} perspective, {camera['lighting']}, documentary realism. "
        f"Mount the camera {float(geometry['mounting_height_m']):.2f} meters above the walking "
        f"surface and tilt its optical axis {float(geometry['pitch_down_deg']):.2f} degrees "
        f"downward below horizontal, using an approximately {float(geometry['focal_mm']):g} mm "
        f"full-frame-equivalent lens. Place the horizon or eye-level vanishing line around "
        f"{horizon_percent}% of image height from the top; it may be implied by architectural "
        "lines when no outdoor horizon is visible. Keep this a physically consistent fixed-camera "
        "view, not an eye-level, aerial, or low-angle photograph. "
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
    if "body_yaw_deg" in sample:
        rotation = (
            f"Keep the camera fixed. Rotate the subject's torso, shoulders, hips, and feet into a "
            f"{sample['body_yaw_prompt']} (yaw {int(sample['body_yaw_deg']):+d} degrees relative "
            "to the camera, where 0 degrees faces the camera and 180 degrees faces away). The head "
            "should generally follow the torso with only a small natural gaze offset. Do not mirror, "
            "replace, or redesign any identity, clothing, footwear, or carried-item detail."
        )
    else:
        rotation = f"The subject faces {sample['orientation']}."
    geometry = camera["geometry"]
    horizon_percent = round(float(geometry["horizon_y_fraction"]) * 100)
    return (
        f"{config['prompt']['invariant_rules']} {config['prompt']['camera_rules']} "
        f"The first input image is the exact fictional identity and wardrobe reference. The second is the "
        f"empty fixed background for camera {camera['local_camera']}. Place that same adult in the plate at "
        f"{camera['site_label']}, viewed from {camera['view_label']}, {sample['pose']}. Preserve the "
        f"plate's fixed camera geometry exactly: {float(geometry['mounting_height_m']):.2f} meter "
        f"mounting height, {float(geometry['pitch_down_deg']):.2f} degree downward tilt, and "
        f"horizon/eye-level vanishing line around {horizon_percent}% from the top. Do not change "
        f"the camera height, pitch, lens perspective, or vanishing line. {rotation} "
        f"Use a {sample['crop_variant']} composition. One principal person only."
        f"{occlusion}{revision}"
    )


def validate_camera_geometry(
    config: Mapping[str, Any], cameras: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Validate deterministic camera pose metadata before any API request is planned."""
    errors: list[str] = []
    horizon_low, horizon_high = (
        float(value) for value in config["camera_geometry"]["horizon_fraction_range"]
    )
    sensor_height_mm = float(config["camera_geometry"]["sensor_height_mm"])
    views = {str(view["key"]): view for view in config["camera_views"]}
    profiles: set[tuple[float, float, float]] = set()
    by_view: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    if len(cameras) != 33:
        errors.append(f"expected 33 cameras, found {len(cameras)}")
    for camera in cameras:
        local_camera = int(camera.get("local_camera", -1))
        geometry = camera.get("geometry") or {}
        view_key = str(camera.get("view"))
        view = views.get(view_key)
        if view is None:
            errors.append(f"camera {local_camera}: unknown view {view_key}")
            continue
        if geometry.get("version") != CAMERA_GEOMETRY_VERSION:
            errors.append(f"camera {local_camera}: camera geometry version mismatch")
            continue
        try:
            pitch = float(geometry["pitch_deg"])
            pitch_down = float(geometry["pitch_down_deg"])
            height = float(geometry["mounting_height_m"])
            horizon = float(geometry["horizon_y_fraction"])
            focal = float(geometry["focal_mm"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"camera {local_camera}: incomplete numeric camera geometry")
            continue
        pitch_min = float(view["pitch_deg"]) - float(view["pitch_jitter_deg"])
        pitch_max = float(view["pitch_deg"]) + float(view["pitch_jitter_deg"])
        height_min = float(view["mounting_height_m"]) - float(
            view["mounting_height_jitter_m"]
        )
        height_max = float(view["mounting_height_m"]) + float(
            view["mounting_height_jitter_m"]
        )
        expected_horizon = round(
            0.5 - math.tan(math.radians(-pitch)) * focal / sensor_height_mm, 4
        )
        if not pitch_min <= pitch <= pitch_max or pitch >= 0 or not math.isclose(
            pitch_down, -pitch, abs_tol=0.011
        ):
            errors.append(f"camera {local_camera}: pitch is outside its downward range")
        if not height_min <= height <= height_max:
            errors.append(f"camera {local_camera}: mounting height is outside its range")
        if not horizon_low <= horizon <= horizon_high:
            errors.append(f"camera {local_camera}: horizon position is outside the safe frame range")
        if not math.isclose(horizon, expected_horizon, abs_tol=0.0001):
            errors.append(f"camera {local_camera}: horizon is inconsistent with pitch and focal length")
        if not math.isclose(float(camera.get("pitch_deg", 999)), pitch, abs_tol=0.001):
            errors.append(f"camera {local_camera}: top-level pitch differs from geometry")
        profiles.add((pitch, height, horizon))
        by_view[view_key].append(geometry)
    if len(profiles) != len(cameras):
        errors.append("every camera must have a unique fixed pitch/height/horizon profile")
    if set(by_view) != set(views) or any(len(rows) != 11 for rows in by_view.values()):
        errors.append("each of the three camera views must occur once at all 11 sites")
    return {
        "valid": not errors,
        "errors": errors,
        "profiles": {
            view_key: {
                "count": len(rows),
                "pitch_deg_min": min((float(row["pitch_deg"]) for row in rows), default=None),
                "pitch_deg_max": max((float(row["pitch_deg"]) for row in rows), default=None),
                "mounting_height_m_min": min(
                    (float(row["mounting_height_m"]) for row in rows), default=None
                ),
                "mounting_height_m_max": max(
                    (float(row["mounting_height_m"]) for row in rows), default=None
                ),
                "horizon_y_fraction_min": min(
                    (float(row["horizon_y_fraction"]) for row in rows), default=None
                ),
                "horizon_y_fraction_max": max(
                    (float(row["horizon_y_fraction"]) for row in rows), default=None
                ),
            }
            for view_key, rows in sorted(by_view.items())
        },
    }


def validate_plan(
    config: Mapping[str, Any],
    identities: Sequence[Mapping[str, Any]],
    cameras: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = config["dataset"]["expected"]
    rotation_views = _body_rotation_views(config)
    rotation_by_degree = {int(view["degrees"]): view for view in rotation_views}
    rotation_index = {int(view["degrees"]): index for index, view in enumerate(rotation_views)}
    errors: list[str] = []
    camera_geometry = validate_camera_geometry(config, cameras)
    errors.extend(camera_geometry["errors"])
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
        yaw_counts = Counter(int(row.get("body_yaw_deg", 999)) for row in rows)
        if yaw_counts != Counter({degrees: 5 for degrees in BODY_YAW_DEGREES}):
            errors.append(f"pid {pid}: expected five images in each of eight body-yaw bins")
        for row in rows:
            view = rotation_by_degree.get(int(row.get("body_yaw_deg", 999)))
            if view is None or (
                row.get("body_yaw_label") != view["label"]
                or row.get("anchor_orientation") != view["anchor"]
            ):
                errors.append(f"pid {pid}: inconsistent body-yaw label or anchor")
                break
        if pid < int(config["dataset"]["train_identities"]):
            if sum(bool(row["occluded"]) for row in rows) != 12:
                errors.append(f"pid {pid}: expected 12 train occlusions")
        else:
            queries = [row for row in rows if row["split"] == "query"]
            galleries = [row for row in rows if row["split"] == "gallery"]
            if len(queries) != 4 or len(galleries) != 36 or not all(row["occluded"] for row in queries):
                errors.append(f"pid {pid}: invalid query/gallery allocation")
            query_yaws = {rotation_index[int(row["body_yaw_deg"])] for row in queries}
            if len(query_yaws) != 4 or len({index % 2 for index in query_yaws}) != 1:
                errors.append(f"pid {pid}: query body yaws do not cover four separated quadrants")
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
        "body_yaw_counts": dict(sorted(Counter(
            int(sample["body_yaw_deg"]) for sample in samples
        ).items())),
        "camera_geometry": camera_geometry,
    }


def validate_pilot(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    specs = {(s["pilot_spec_id"], s["local_camera"]) for s in samples}
    coverage = Counter(int(s["local_camera"]) for s in samples)
    errors = []
    if len(samples) != 96 or len(specs) != 96:
        errors.append("pilot must contain exactly 96 low-quality specifications")
    if any(sample.get("quality") != "low" for sample in samples):
        errors.append("every pilot sample must use low quality")
    if set(coverage) != set(range(33)) or min(coverage.values(), default=0) < 2:
        errors.append("pilot must cover every camera at least twice")
    return {"valid": not errors, "errors": errors, "camera_coverage": dict(sorted(coverage.items()))}


def validate_rotation_pilot(
    config: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    rotation = config["body_rotation"]
    expected_total = int(rotation["pilot_identities"]) * int(
        rotation["pilot_samples_per_identity"]
    )
    errors = []
    if len(samples) != expected_total:
        errors.append(f"body-rotation pilot must contain {expected_total} samples")
    coverage = Counter(int(sample["local_camera"]) for sample in samples)
    if set(coverage) != set(range(33)) or min(coverage.values(), default=0) < 2:
        errors.append("body-rotation pilot must cover every camera at least twice")
    per_pid: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        per_pid[int(sample["local_pid"])].append(sample)
    expected_yaws = set(BODY_YAW_DEGREES)
    for pid in range(int(rotation["pilot_identities"])):
        rows = per_pid.get(pid, [])
        if len(rows) != len(expected_yaws):
            errors.append(f"rotation pilot pid {pid}: expected eight samples")
            continue
        if {int(row.get("body_yaw_deg", 999)) for row in rows} != expected_yaws:
            errors.append(f"rotation pilot pid {pid}: every yaw must appear exactly once")
        if any(row.get("quality") != config["model"]["quality"] for row in rows):
            errors.append(f"rotation pilot pid {pid}: quality must be low")
        if any(row.get("occluded") for row in rows):
            errors.append(f"rotation pilot pid {pid}: samples must be clean")
    return {
        "valid": not errors,
        "errors": errors,
        "sample_count": len(samples),
        "camera_coverage": dict(sorted(coverage.items())),
        "body_yaw_counts": dict(sorted(Counter(
            int(sample.get("body_yaw_deg", 999)) for sample in samples
        ).items())),
    }


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
    report: Mapping[str, Any],
    quality: str,
    max_usd: float,
    config: Mapping[str, Any],
    waived_gates: Sequence[str] = (),
    waiver_reason: str | None = None,
) -> dict[str, Any]:
    if quality != config["model"]["quality"]:
        raise ValueError("quality must be low")
    if max_usd <= 0 or not math.isfinite(max_usd):
        raise ValueError("max_usd must be a positive finite number")
    normalized_waivers = sorted({str(gate).strip() for gate in waived_gates if str(gate).strip()})
    if not report.get("pilot_gate_passed") and not normalized_waivers:
        raise ValueError("pilot report has not passed every automatic and manual gate")
    if normalized_waivers and not str(waiver_reason or "").strip():
        raise ValueError("an explicit waiver reason is required for failed pilot gates")
    report_hash = sha256_bytes(canonical_json(report))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "approved_at": utc_now(),
        "model_id": config["model"]["api_id"],
        "catalog_snapshot": config["model"]["catalog_snapshot"],
        "quality": quality,
        "max_usd": round(float(max_usd), 2),
        "report_sha256": report_hash,
        "config_sha256": config_sha256(config),
    }
    if normalized_waivers:
        payload["gate_waiver"] = {
            "waived_gates": normalized_waivers,
            "reason": str(waiver_reason).strip(),
            "user_authorized": True,
            "recorded_at": utc_now(),
        }
    return payload


def verify_approval(approval: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if approval.get("model_id") != config["model"]["api_id"]:
        raise ValueError("approval model ID differs from config")
    if approval.get("catalog_snapshot") != config["model"]["catalog_snapshot"]:
        raise ValueError("approval catalog snapshot differs from config")
    if approval.get("config_sha256") != config_sha256(config):
        raise ValueError("configuration changed after approval")
    if approval.get("quality") != config["model"]["quality"]:
        raise ValueError("approval quality must be low")
    if float(approval.get("max_usd", 0)) <= 0:
        raise ValueError("approval has no valid cost ceiling")
    waiver = approval.get("gate_waiver")
    if waiver is not None:
        if not isinstance(waiver, Mapping):
            raise ValueError("approval gate waiver is malformed")
        gates = waiver.get("waived_gates")
        if not isinstance(gates, list) or not gates or not all(
            isinstance(gate, str) and gate.strip() for gate in gates
        ):
            raise ValueError("approval gate waiver has no valid gate list")
        if not str(waiver.get("reason", "")).strip() or not waiver.get("user_authorized"):
            raise ValueError("approval gate waiver lacks explicit user authorization")


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


def _xyxy_intersection_over_smaller(
    left: Sequence[float], right: Sequence[float]
) -> float:
    """Return intersection divided by the smaller box area for nested-box NMS."""
    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(
        0.0, float(left[3]) - float(left[1])
    )
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(
        0.0, float(right[3]) - float(right[1])
    )
    return intersection / max(min(left_area, right_area), 1e-9)


def _nms_person_candidates(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    iou_threshold: float = PERSON_NMS_IOU,
    containment_threshold: float = PERSON_NMS_CONTAINMENT,
) -> list[int]:
    """Suppress duplicate detections without hiding independent people."""
    order = sorted(range(len(boxes)), key=lambda index: (-float(scores[index]), index))
    kept: list[int] = []
    for index in order:
        suppress = False
        for other in kept:
            overlap = _xyxy_intersection_over_smaller(boxes[index], boxes[other])
            iou = _xyxy_iou(boxes[index], boxes[other])
            if iou > iou_threshold:
                suppress = True
                break
            if overlap >= containment_threshold:
                suppress = True
                break

            # SSDLite sometimes labels a trolley or other foreground occluder
            # plus the principal person's legs as a weak second person.  Only
            # suppress this narrow lower-fragment pattern: a low-confidence box
            # must begin well below the stronger person's head and overlap at
            # least half of the smaller box.  A separate or similarly tall
            # second person remains a hard QA failure.
            stronger = boxes[other]
            candidate = boxes[index]
            stronger_height = max(float(stronger[3]) - float(stronger[1]), 1e-9)
            lower_fragment = (
                float(scores[index]) <= PERSON_FRAGMENT_SCORE_MAX
                and overlap >= PERSON_FRAGMENT_OVERLAP_MIN
                and float(candidate[1])
                >= float(stronger[1])
                + stronger_height * PERSON_FRAGMENT_TOP_FRACTION_MIN
            )
            if lower_fragment:
                suppress = True
                break

            # A foreground cart can also split one person into a strong
            # upper-body box and a weaker lower-body-plus-cart box.  Require
            # substantial overlap by both IoU and smaller-box containment so
            # a spatially separate person is never removed by this fallback.
            split_body_duplicate = (
                float(scores[index]) <= PERSON_SPLIT_BODY_SCORE_MAX
                and iou >= PERSON_SPLIT_BODY_IOU_MIN
                and overlap >= PERSON_SPLIT_BODY_OVERLAP_MIN
            )
            if split_body_duplicate:
                suppress = True
                break
        if not suppress:
            kept.append(index)
    return kept


def _filter_tiny_secondary_person_candidates(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    image_size: tuple[int, int],
) -> tuple[list[int], list[dict[str, Any]]]:
    """Ignore only weak, tiny secondary detector artifacts; keep the principal."""
    image_width, image_height = (float(value) for value in image_size)
    kept: list[int] = []
    suppressed: list[dict[str, Any]] = []
    for index, (box, score) in enumerate(zip(boxes, scores)):
        width_fraction = max(0.0, float(box[2]) - float(box[0])) / max(
            image_width, 1e-9
        )
        height_fraction = max(0.0, float(box[3]) - float(box[1])) / max(
            image_height, 1e-9
        )
        tiny_weak_secondary = (
            index > 0
            and float(score) <= PERSON_TINY_SECONDARY_SCORE_MAX
            and width_fraction <= PERSON_TINY_SECONDARY_WIDTH_FRACTION_MAX
            and height_fraction <= PERSON_TINY_SECONDARY_HEIGHT_FRACTION_MAX
        )
        if tiny_weak_secondary:
            suppressed.append(
                {
                    "index": index,
                    "score": float(score),
                    "width_fraction": width_fraction,
                    "height_fraction": height_fraction,
                }
            )
        else:
            kept.append(index)
    return kept, suppressed


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
    size_kept, tiny_suppressed = _filter_tiny_secondary_person_candidates(
        boxes, scores, (int(image.shape[1]), int(image.shape[0]))
    )
    boxes = [boxes[index] for index in size_kept]
    scores = [scores[index] for index in size_kept]
    bbox = None
    if boxes:
        x0, y0, x1, y1 = boxes[0]
        bbox = tuple(round(value) for value in (x0, y0, x1 - x0, y1 - y0))
    return {
        "bbox": bbox,
        "count": len(boxes),
        "candidate_boxes_xyxy": boxes,
        "candidate_scores": scores,
        "tiny_secondary_suppressed": tiny_suppressed,
        "tiny_secondary_score_max": PERSON_TINY_SECONDARY_SCORE_MAX,
        "tiny_secondary_width_fraction_max": (
            PERSON_TINY_SECONDARY_WIDTH_FRACTION_MAX
        ),
        "tiny_secondary_height_fraction_max": (
            PERSON_TINY_SECONDARY_HEIGHT_FRACTION_MAX
        ),
        "candidate_count_before_nms": int(mask.sum().item()),
        "principal_score": scores[0] if scores else 0.0,
        "backend": "torchvision_ssdlite320_mobilenet_v3_large",
        "weights": weights_name,
        "score_threshold": PERSON_SCORE_MIN,
        "nms_iou_threshold": PERSON_NMS_IOU,
        "nms_containment_threshold": PERSON_NMS_CONTAINMENT,
    }


def _intentional_cart_person_fragment(
    detection: Mapping[str, Any],
    pose_geometry: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Identify one narrow cart-induced duplicate after independent QA agrees."""
    boxes = detection.get("candidate_boxes_xyxy") or []
    scores = detection.get("candidate_scores") or []
    occluder = str(sample.get("occluder") or "").lower()
    target_ratio = float(sample.get("target_occlusion_ratio", 0.0))
    prerequisites = (
        bool(sample.get("occluded"))
        and "cart" in occluder
        and 0.20 <= target_ratio <= 0.50
        and int(detection.get("count", 0)) == 2
        and len(boxes) == 2
        and len(scores) == 2
        and bool(pose_geometry.get("pass"))
        and int(pose_geometry.get("pose_candidate_count", 0)) == 1
    )
    if not prerequisites:
        return {
            "pass": False,
            "used": False,
            "occluder": occluder,
            "target_occlusion_ratio": target_ratio,
            "prerequisites": False,
        }

    principal, candidate = boxes
    iou = _xyxy_iou(principal, candidate)
    overlap = _xyxy_intersection_over_smaller(principal, candidate)
    principal_height = max(float(principal[3]) - float(principal[1]), 1e-9)
    candidate_top_fraction = (
        float(candidate[1]) - float(principal[1])
    ) / principal_height
    principal_area = max(
        (float(principal[2]) - float(principal[0]))
        * (float(principal[3]) - float(principal[1])),
        1e-9,
    )
    candidate_area = max(
        0.0, float(candidate[2]) - float(candidate[0])
    ) * max(0.0, float(candidate[3]) - float(candidate[1]))
    candidate_extends_below = float(candidate[3]) > float(principal[3])
    high_occlusion_pose_support = (
        target_ratio >= PERSON_CART_FRAGMENT_HIGH_OCCLUSION_RATIO_MIN
        and float(pose_geometry.get("pose_score", 0.0))
        >= PERSON_CART_FRAGMENT_HIGH_OCCLUSION_POSE_SCORE_MIN
        and overlap >= PERSON_CART_FRAGMENT_HIGH_OCCLUSION_OVERLAP_MIN
    )
    passed = (
        float(scores[1]) <= PERSON_CART_FRAGMENT_SCORE_MAX
        and iou >= PERSON_CART_FRAGMENT_IOU_MIN
        and (
            overlap >= PERSON_CART_FRAGMENT_OVERLAP_MIN
            or high_occlusion_pose_support
        )
        and candidate_top_fraction >= PERSON_CART_FRAGMENT_TOP_FRACTION_MIN
        and candidate_area > principal_area
        and candidate_extends_below
    )
    return {
        "pass": passed,
        "used": passed,
        "occluder": occluder,
        "target_occlusion_ratio": target_ratio,
        "prerequisites": True,
        "secondary_score": float(scores[1]),
        "secondary_score_max": PERSON_CART_FRAGMENT_SCORE_MAX,
        "iou": iou,
        "iou_min": PERSON_CART_FRAGMENT_IOU_MIN,
        "smaller_box_overlap": overlap,
        "smaller_box_overlap_min": PERSON_CART_FRAGMENT_OVERLAP_MIN,
        "high_occlusion_pose_support": high_occlusion_pose_support,
        "high_occlusion_ratio_min": PERSON_CART_FRAGMENT_HIGH_OCCLUSION_RATIO_MIN,
        "high_occlusion_overlap_min": PERSON_CART_FRAGMENT_HIGH_OCCLUSION_OVERLAP_MIN,
        "high_occlusion_pose_score_min": (
            PERSON_CART_FRAGMENT_HIGH_OCCLUSION_POSE_SCORE_MIN
        ),
        "candidate_top_fraction": candidate_top_fraction,
        "candidate_top_fraction_min": PERSON_CART_FRAGMENT_TOP_FRACTION_MIN,
        "candidate_area_ratio": candidate_area / principal_area,
        "candidate_extends_below": candidate_extends_below,
    }


def _overlapped_ankle_visibility(
    ankle_scores: Sequence[float],
    ankle_points: Sequence[Sequence[float]],
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    knee_scores: Sequence[float] | None = None,
    knee_points: Sequence[Sequence[float]] | None = None,
    walking_pose: bool = False,
    pose_score: float | None = None,
) -> dict[str, Any]:
    """Accept one weak ankle only when both predicted feet remain safely in-frame."""
    if len(ankle_scores) != 2 or len(ankle_points) != 2:
        raise ValueError("overlapped-ankle QA requires exactly two ankle keypoints")
    x, y, width, height = (float(value) for value in bbox)
    image_width, image_height = (float(value) for value in image_size)
    x_padding = width * KEYPOINT_BBOX_TOLERANCE_FRACTION
    y_padding = height * KEYPOINT_BBOX_TOLERANCE_FRACTION
    lower_y = y + height * ANKLE_LOWER_BODY_FRACTION_MIN
    locations = []
    for point in ankle_points:
        point_x, point_y = float(point[0]), float(point[1])
        locations.append(
            0.0 <= point_x < image_width
            and 0.0 <= point_y < image_height
            and x - x_padding <= point_x <= x + width + x_padding
            and lower_y <= point_y <= y + height + y_padding
        )
    bottom_clearance = image_height - (y + height)
    required_clearance = max(2.0, image_height * PERSON_BOTTOM_CLEARANCE_FRACTION_MIN)
    pair_distance = math.dist(
        (float(ankle_points[0][0]), float(ankle_points[0][1])),
        (float(ankle_points[1][0]), float(ankle_points[1][1])),
    )
    maximum_pair_distance = height * ANKLE_PAIR_DISTANCE_FRACTION_MAX
    ankles_overlap = pair_distance <= maximum_pair_distance
    strong_ankle = (
        max(float(score) for score in ankle_scores) >= STRONG_ANKLE_FALLBACK_LOGIT_MIN
    )
    weakest_ankle_score = min(float(score) for score in ankle_scores)
    moderate_weak_ankle = weakest_ankle_score >= MODERATE_ANKLE_LOGIT_MIN
    deeply_weak_overlapped_ankle = (
        weakest_ankle_score >= SECONDARY_ANKLE_LOGIT_MIN and ankles_overlap
    )
    weak_index = min(range(2), key=lambda index: float(ankle_scores[index]))
    strong_index = 1 - weak_index
    knees_available = (
        knee_scores is not None
        and knee_points is not None
        and len(knee_scores) == 2
        and len(knee_points) == 2
    )
    knee_support_valid = bool(
        knees_available
        and min(float(score) for score in knee_scores or ()) >= KNEE_SUPPORT_LOGIT_MIN
    )
    high_confidence_knee_support_valid = bool(
        knees_available
        and pose_score is not None
        and float(pose_score) >= POSE_DETECTOR_FALLBACK_SCORE_MIN
        and min(float(score) for score in knee_scores or ())
        >= HIGH_CONFIDENCE_KNEE_SUPPORT_LOGIT_MIN
    )
    knee_locations_valid = []
    weak_ankle_knee_distance = math.inf
    weak_ankle_near_knee = False
    high_confidence_weak_ankle_near_knee = False
    if knees_available:
        knee_locations_valid = [
            (
                x - x_padding <= float(point[0]) <= x + width + x_padding
                and y + height * 0.35 <= float(point[1]) <= y + height + y_padding
            )
            for point in knee_points or ()
        ]
        weak_point = ankle_points[weak_index]
        weak_ankle_knee_distance = min(
            math.dist(
                (float(weak_point[0]), float(weak_point[1])),
                (float(point[0]), float(point[1])),
            )
            for point in knee_points or ()
        )
        weak_ankle_near_knee = (
            weak_ankle_knee_distance <= height * ANKLE_KNEE_DISTANCE_FRACTION_MAX
        )
        high_confidence_weak_ankle_near_knee = (
            pose_score is not None
            and float(pose_score) >= POSE_DETECTOR_FALLBACK_SCORE_MIN
            and weakest_ankle_score >= LONGITUDINAL_ANKLE_LOGIT_MIN
            and weak_ankle_knee_distance
            <= height * ANKLE_KNEE_HIGH_CONF_DISTANCE_FRACTION_MAX
        )
    strong_ankle_below_weak = (
        float(ankle_points[strong_index][1]) - float(ankle_points[weak_index][1])
        >= height * ANKLE_VERTICAL_SEPARATION_FRACTION_MIN
    )
    deep_longitudinal_pose_support = (
        pose_score is not None
        and float(pose_score) >= POSE_DETECTOR_FALLBACK_SCORE_MIN
        and weakest_ankle_score >= SECONDARY_ANKLE_LOGIT_MIN
    )
    longitudinal_leg_overlap = (
        walking_pose
        and (
            weakest_ankle_score >= LONGITUDINAL_ANKLE_LOGIT_MIN
            or deep_longitudinal_pose_support
        )
        and (knee_support_valid or high_confidence_knee_support_valid)
        and all(knee_locations_valid)
        and (weak_ankle_near_knee or high_confidence_weak_ankle_near_knee)
        and strong_ankle_below_weak
    )
    weak_ankle = (
        moderate_weak_ankle
        or deeply_weak_overlapped_ankle
        or longitudinal_leg_overlap
    )
    person_not_bottom_clipped = bottom_clearance >= required_clearance
    passed = (
        strong_ankle
        and weak_ankle
        and all(locations)
        and person_not_bottom_clipped
    )
    return {
        "pass": passed,
        "ankle_scores": [float(score) for score in ankle_scores],
        "ankle_points": [
            [round(float(point[0]), 3), round(float(point[1]), 3)]
            for point in ankle_points
        ],
        "strong_ankle_visible": strong_ankle,
        "strong_ankle_logit_min": STRONG_ANKLE_FALLBACK_LOGIT_MIN,
        "moderate_ankle_logit_min": MODERATE_ANKLE_LOGIT_MIN,
        "secondary_ankle_logit_min": SECONDARY_ANKLE_LOGIT_MIN,
        "moderate_weak_ankle": moderate_weak_ankle,
        "deeply_weak_overlapped_ankle": deeply_weak_overlapped_ankle,
        "longitudinal_ankle_logit_min": LONGITUDINAL_ANKLE_LOGIT_MIN,
        "deep_longitudinal_ankle_logit_min": SECONDARY_ANKLE_LOGIT_MIN,
        "pose_score": float(pose_score) if pose_score is not None else None,
        "deep_longitudinal_pose_score_min": POSE_DETECTOR_FALLBACK_SCORE_MIN,
        "deep_longitudinal_pose_support": deep_longitudinal_pose_support,
        "longitudinal_leg_overlap": longitudinal_leg_overlap,
        "walking_pose": walking_pose,
        "knee_scores": [float(score) for score in knee_scores] if knee_scores else None,
        "knee_support_logit_min": KNEE_SUPPORT_LOGIT_MIN,
        "knee_support_valid": knee_support_valid,
        "high_confidence_knee_support_logit_min": (
            HIGH_CONFIDENCE_KNEE_SUPPORT_LOGIT_MIN
        ),
        "high_confidence_knee_support_valid": high_confidence_knee_support_valid,
        "knee_locations_valid": knee_locations_valid,
        "weak_ankle_knee_distance_px": (
            round(weak_ankle_knee_distance, 3) if math.isfinite(weak_ankle_knee_distance) else None
        ),
        "ankle_knee_distance_fraction_max": ANKLE_KNEE_DISTANCE_FRACTION_MAX,
        "ankle_knee_high_conf_distance_fraction_max": (
            ANKLE_KNEE_HIGH_CONF_DISTANCE_FRACTION_MAX
        ),
        "maximum_ankle_knee_distance_px": round(
            height * ANKLE_KNEE_DISTANCE_FRACTION_MAX, 3
        ),
        "maximum_high_conf_ankle_knee_distance_px": round(
            height * ANKLE_KNEE_HIGH_CONF_DISTANCE_FRACTION_MAX, 3
        ),
        "weak_ankle_near_knee": weak_ankle_near_knee,
        "high_confidence_weak_ankle_near_knee": (
            high_confidence_weak_ankle_near_knee
        ),
        "ankle_vertical_separation_fraction_min": (
            ANKLE_VERTICAL_SEPARATION_FRACTION_MIN
        ),
        "strong_ankle_below_weak": strong_ankle_below_weak,
        "ankle_locations_valid": locations,
        "lower_body_fraction_min": ANKLE_LOWER_BODY_FRACTION_MIN,
        "ankle_pair_distance_px": round(pair_distance, 3),
        "ankle_pair_distance_fraction_max": ANKLE_PAIR_DISTANCE_FRACTION_MAX,
        "maximum_ankle_pair_distance_px": round(maximum_pair_distance, 3),
        "ankles_overlap": ankles_overlap,
        "person_bottom_clearance_px": round(bottom_clearance, 3),
        "required_bottom_clearance_px": round(required_clearance, 3),
        "person_not_bottom_clipped": person_not_bottom_clipped,
    }


def _back_view_head_visibility(
    head_score: float,
    head_point: Sequence[float],
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    supporting_shoulder_score: float | None = None,
) -> dict[str, Any]:
    """Allow a weak face keypoint only when a back-view head is safely located in-frame."""
    x, y, width, height = (float(value) for value in bbox)
    image_width, image_height = (float(value) for value in image_size)
    point_x, point_y = (float(value) for value in head_point[:2])
    x_padding = width * KEYPOINT_BBOX_TOLERANCE_FRACTION
    y_padding = height * KEYPOINT_BBOX_TOLERANCE_FRACTION
    location_valid = (
        0.0 <= point_x < image_width
        and 0.0 <= point_y < image_height
        and x - x_padding <= point_x <= x + width + x_padding
        and y - y_padding <= point_y <= y + height * HEAD_UPPER_BODY_FRACTION_MAX
    )
    top_clearance = y
    required_clearance = max(2.0, image_height * PERSON_BOTTOM_CLEARANCE_FRACTION_MIN)
    person_not_top_clipped = top_clearance >= required_clearance
    moderate_head = float(head_score) >= MODERATE_BACK_VIEW_HEAD_LOGIT_MIN
    deep_head_location_valid = point_y <= y + height * DEEP_BACK_HEAD_UPPER_BODY_FRACTION_MAX
    shoulder_support_valid = (
        supporting_shoulder_score is not None
        and float(supporting_shoulder_score) >= BACK_VIEW_SHOULDER_SUPPORT_LOGIT_MIN
    )
    deep_supported_head = (
        float(head_score) >= BACK_VIEW_HEAD_LOGIT_MIN
        and deep_head_location_valid
        and shoulder_support_valid
    )
    passed = (
        (moderate_head or deep_supported_head)
        and location_valid
        and person_not_top_clipped
    )
    return {
        "pass": passed,
        "head_score": float(head_score),
        "head_point": [round(point_x, 3), round(point_y, 3)],
        "back_view_head_logit_min": BACK_VIEW_HEAD_LOGIT_MIN,
        "moderate_back_view_head_logit_min": MODERATE_BACK_VIEW_HEAD_LOGIT_MIN,
        "moderate_head": moderate_head,
        "deep_supported_head": deep_supported_head,
        "supporting_shoulder_score": (
            float(supporting_shoulder_score)
            if supporting_shoulder_score is not None
            else None
        ),
        "shoulder_support_logit_min": BACK_VIEW_SHOULDER_SUPPORT_LOGIT_MIN,
        "shoulder_support_valid": shoulder_support_valid,
        "head_location_valid": location_valid,
        "upper_body_fraction_max": HEAD_UPPER_BODY_FRACTION_MAX,
        "deep_head_location_valid": deep_head_location_valid,
        "deep_head_upper_body_fraction_max": DEEP_BACK_HEAD_UPPER_BODY_FRACTION_MAX,
        "person_top_clearance_px": round(top_clearance, 3),
        "required_top_clearance_px": round(required_clearance, 3),
        "person_not_top_clipped": person_not_top_clipped,
    }


def _is_back_facing_sample(sample: Mapping[str, Any]) -> bool:
    if "body_yaw_deg" in sample:
        return abs(int(sample["body_yaw_deg"])) >= 135
    orientation = str(sample.get("anchor_orientation") or sample.get("orientation") or "")
    return orientation in {"back", "back_left", "back_right"}


def _required_pose_region_scores(
    scores: Sequence[float], occluded: bool
) -> dict[str, float]:
    required = {
        "head": (0,),
        "shoulders": (5, 6),
        ("waist" if occluded else "feet"): ((11, 12) if occluded else (15, 16)),
    }
    return {
        name: min(float(scores[index]) for index in members)
        for name, members in required.items()
    }


def _pose_bbox_expansion(
    principal_bbox: tuple[int, int, int, int],
    pose_box: Sequence[float],
    pose_score: float,
    region_scores: Mapping[str, float],
    image_size: tuple[int, int],
    occluded: bool,
    *,
    occluder: str = "",
    target_occlusion_ratio: float = 0.0,
) -> dict[str, Any]:
    """Safely expand an under-sized detector box from a complete pose box."""
    x, y, width, height = (float(value) for value in principal_bbox)
    principal_xyxy = (x, y, x + width, y + height)
    px0, py0, px1, py1 = (float(value) for value in pose_box)
    image_width, image_height = (float(value) for value in image_size)
    principal_area = max(width * height, 1e-9)
    pose_area = max(0.0, px1 - px0) * max(0.0, py1 - py0)
    ix0, iy0 = max(x, px0), max(y, py0)
    ix1, iy1 = min(x + width, px1), min(y + height, py1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    principal_containment = intersection / principal_area
    area_ratio = pose_area / principal_area
    match_iou = _xyxy_iou(principal_xyxy, pose_box)
    principal_center = (x + width / 2.0, y + height / 2.0)
    pose_center = ((px0 + px1) / 2.0, (py0 + py1) / 2.0)
    center_distance_fraction = math.dist(principal_center, pose_center) / max(
        math.hypot(width, height), 1e-9
    )
    pose_box_in_frame = (
        2.0 <= px0 < px1 <= image_width - 2.0
        and 2.0 <= py0 < py1 <= image_height - 2.0
    )
    ordinary_regions_visible = all(
        float(score) >= KEYPOINT_VISIBILITY_LOGIT_MIN for score in region_scores.values()
    )
    cart_occlusion = (
        occluded
        and "cart" in str(occluder).lower()
        and 0.20 <= float(target_occlusion_ratio) <= 0.50
    )
    passed = (
        (not occluded or cart_occlusion)
        and float(pose_score) >= POSE_BBOX_EXPANSION_SCORE_MIN
        and ordinary_regions_visible
        and match_iou >= POSE_BBOX_EXPANSION_IOU_MIN
        and principal_containment >= POSE_BBOX_EXPANSION_CONTAINMENT_MIN
        and 1.0 < area_ratio <= POSE_BBOX_EXPANSION_AREA_RATIO_MAX
        and center_distance_fraction <= POSE_BBOX_EXPANSION_CENTER_DISTANCE_MAX
        and pose_box_in_frame
    )
    expanded_bbox = [
        math.floor(px0),
        math.floor(py0),
        math.ceil(px1) - math.floor(px0),
        math.ceil(py1) - math.floor(py0),
    ]
    return {
        "pass": passed,
        "expanded_bbox": expanded_bbox,
        "pose_box_xyxy": [round(value, 3) for value in (px0, py0, px1, py1)],
        "pose_score": float(pose_score),
        "pose_score_min": POSE_BBOX_EXPANSION_SCORE_MIN,
        "match_iou": match_iou,
        "match_iou_min": POSE_BBOX_EXPANSION_IOU_MIN,
        "principal_containment": principal_containment,
        "principal_containment_min": POSE_BBOX_EXPANSION_CONTAINMENT_MIN,
        "area_ratio": area_ratio,
        "area_ratio_max": POSE_BBOX_EXPANSION_AREA_RATIO_MAX,
        "center_distance_fraction": center_distance_fraction,
        "center_distance_fraction_max": POSE_BBOX_EXPANSION_CENTER_DISTANCE_MAX,
        "ordinary_regions_visible": ordinary_regions_visible,
        "pose_box_in_frame": pose_box_in_frame,
        "occluded": occluded,
        "cart_occlusion": cart_occlusion,
        "occluder": str(occluder),
        "target_occlusion_ratio": float(target_occlusion_ratio),
    }
def _select_pose_candidate(
    matches: Sequence[tuple[float, float, int]],
    score_rows: Mapping[int, Sequence[float]],
    occluded: bool,
) -> tuple[tuple[float, float, int], dict[str, Any] | None]:
    """Prefer the principal IoU match, with a strict complete-pose recovery.

    A foreground railing can make Keypoint R-CNN emit separate upper- and
    lower-body detections for one person.  It can also return a slightly
    higher-IoU partial pose beside a complete pose.  Switch from the
    highest-IoU candidate only when another candidate still overlaps the
    principal detection and passes every sample-specific required region at
    the ordinary (non-relaxed) visibility threshold.
    """
    primary = max(matches)
    primary_scores = _required_pose_region_scores(score_rows[primary[2]], occluded)
    if all(score >= KEYPOINT_VISIBILITY_LOGIT_MIN for score in primary_scores.values()):
        return primary, None
    eligible = []
    for match_iou, pose_score, index in matches:
        if match_iou < POSE_MATCH_IOU_MIN:
            continue
        region_scores = _required_pose_region_scores(score_rows[index], occluded)
        if all(score >= KEYPOINT_VISIBILITY_LOGIT_MIN for score in region_scores.values()):
            eligible.append(
                (
                    min(region_scores.values()),
                    match_iou,
                    pose_score,
                    index,
                    region_scores,
                )
            )
    if not eligible:
        return primary, None
    _minimum_score, match_iou, pose_score, index, region_scores = max(eligible)
    if index == primary[2]:
        return primary, None
    selected = (match_iou, pose_score, index)
    return selected, {
        "pass": True,
        "initial_candidate_index": primary[2],
        "initial_match_iou": primary[0],
        "initial_pose_score": primary[1],
        "initial_region_scores": primary_scores,
        "selected_candidate_index": index,
        "selected_match_iou": match_iou,
        "selected_pose_score": pose_score,
        "selected_region_scores": region_scores,
        "ordinary_visibility_logit_min": KEYPOINT_VISIBILITY_LOGIT_MIN,
    }


def _occluded_waist_visibility(
    scores: Sequence[float],
    points: Sequence[Sequence[float]],
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    target_occlusion_ratio: float,
    *,
    occluder: str = "",
    head_score: float | None = None,
    shoulder_score: float | None = None,
    pose_score: float | None = None,
) -> dict[str, Any]:
    """Accept one weak hip only for a safely localized, intentional occlusion.

    COCO pose models frequently suppress the farther hip when a foreground
    railing or wall covers the lower body.  This fallback remains fail-closed:
    one hip must pass the normal visibility threshold, the other must remain
    above a bounded secondary threshold, and both predicted locations must be
    inside the lower part of an unclipped principal-person box.
    """
    x, y, width, height = (float(value) for value in bbox)
    image_width, image_height = (float(value) for value in image_size)
    x_padding = width * KEYPOINT_BBOX_TOLERANCE_FRACTION
    y_padding = height * KEYPOINT_BBOX_TOLERANCE_FRACTION
    minimum_y = y + height * HIP_LOWER_BODY_FRACTION_MIN
    location_valid = [
        (
            0.0 <= float(point[0]) < image_width
            and 0.0 <= float(point[1]) < image_height
            and x - x_padding <= float(point[0]) <= x + width + x_padding
            and minimum_y <= float(point[1]) <= y + height + y_padding
        )
        for point in points
    ]
    bottom_clearance = image_height - (y + height)
    required_clearance = max(2.0, image_height * PERSON_BOTTOM_CLEARANCE_FRACTION_MIN)
    person_not_bottom_clipped = bottom_clearance >= required_clearance
    strong_hip_visible = any(
        float(score) >= KEYPOINT_VISIBILITY_LOGIT_MIN for score in scores
    )
    weak_hip_bounded = min(float(score) for score in scores) >= SECONDARY_HIP_LOGIT_MIN
    requested_occlusion_valid = 0.20 <= float(target_occlusion_ratio) <= 0.50
    ordinary_waist = (
        strong_hip_visible
        and weak_hip_bounded
        and all(location_valid)
        and person_not_bottom_clipped
        and requested_occlusion_valid
    )
    boundary_minimum_y = y + height * HIP_OCCLUSION_BOUNDARY_FRACTION_MIN
    boundary_locations_valid = all(
        valid and float(point[1]) >= boundary_minimum_y
        for valid, point in zip(location_valid, points)
    )
    high_occlusion_hips_bounded = (
        min(float(score) for score in scores) >= HIGH_OCCLUSION_HIP_LOGIT_MIN
        and max(float(score) for score in scores)
        >= HIGH_OCCLUSION_HIP_SUPPORT_LOGIT_MIN
    )
    upper_body_support_valid = (
        head_score is not None
        and shoulder_score is not None
        and float(head_score) >= HIGH_OCCLUSION_REGION_SUPPORT_LOGIT_MIN
        and float(shoulder_score) >= HIGH_OCCLUSION_REGION_SUPPORT_LOGIT_MIN
    )
    pose_support_valid = (
        pose_score is not None and float(pose_score) >= HIGH_OCCLUSION_POSE_SCORE_MIN
    )
    hip_pair_distance = math.dist(
        (float(points[0][0]), float(points[0][1])),
        (float(points[1][0]), float(points[1][1])),
    )
    bollard_pose_support_valid = (
        pose_score is not None and float(pose_score) >= BOLLARD_OCCLUSION_POSE_SCORE_MIN
    )
    bollard_hips_bounded = (
        min(float(score) for score in scores) >= HIGH_OCCLUSION_HIP_LOGIT_MIN
    )
    bollard_hips_aligned = (
        hip_pair_distance <= width * BOLLARD_HIP_PAIR_DISTANCE_FRACTION_MAX
    )
    wall_boundary_waist = (
        "wall" in str(occluder).lower()
        and 0.40 <= float(target_occlusion_ratio) <= 0.50
        and high_occlusion_hips_bounded
        and boundary_locations_valid
        and upper_body_support_valid
        and pose_support_valid
        and person_not_bottom_clipped
    )
    bollard_boundary_waist = (
        "bollard" in str(occluder).lower()
        and 0.40 <= float(target_occlusion_ratio) <= 0.50
        and bollard_hips_bounded
        and bollard_hips_aligned
        and boundary_locations_valid
        and upper_body_support_valid
        and bollard_pose_support_valid
        and person_not_bottom_clipped
    )
    railing_hips_bounded = (
        min(float(score) for score in scores) >= RAILING_OCCLUSION_HIP_LOGIT_MIN
        and max(float(score) for score in scores)
        >= RAILING_OCCLUSION_HIP_SUPPORT_LOGIT_MIN
    )
    railing_boundary_waist = (
        "railing" in str(occluder).lower()
        and 0.40 <= float(target_occlusion_ratio) <= 0.50
        and railing_hips_bounded
        and boundary_locations_valid
        and upper_body_support_valid
        and pose_support_valid
        and person_not_bottom_clipped
    )
    passed = (
        ordinary_waist
        or wall_boundary_waist
        or bollard_boundary_waist
        or railing_boundary_waist
    )
    return {
        "pass": passed,
        "hip_scores": [float(score) for score in scores],
        "hip_points": [
            [round(float(point[0]), 3), round(float(point[1]), 3)] for point in points
        ],
        "strong_hip_visible": strong_hip_visible,
        "secondary_hip_logit_min": SECONDARY_HIP_LOGIT_MIN,
        "weak_hip_bounded": weak_hip_bounded,
        "hip_locations_valid": location_valid,
        "lower_body_fraction_min": HIP_LOWER_BODY_FRACTION_MIN,
        "person_bottom_clearance_px": round(bottom_clearance, 3),
        "required_bottom_clearance_px": round(required_clearance, 3),
        "person_not_bottom_clipped": person_not_bottom_clipped,
        "target_occlusion_ratio": float(target_occlusion_ratio),
        "requested_occlusion_valid": requested_occlusion_valid,
        "ordinary_waist": ordinary_waist,
        "wall_boundary_waist": wall_boundary_waist,
        "bollard_boundary_waist": bollard_boundary_waist,
        "railing_boundary_waist": railing_boundary_waist,
        "occluder": str(occluder),
        "high_occlusion_hip_logit_min": HIGH_OCCLUSION_HIP_LOGIT_MIN,
        "high_occlusion_hip_support_logit_min": HIGH_OCCLUSION_HIP_SUPPORT_LOGIT_MIN,
        "high_occlusion_hips_bounded": high_occlusion_hips_bounded,
        "boundary_fraction_min": HIP_OCCLUSION_BOUNDARY_FRACTION_MIN,
        "boundary_locations_valid": boundary_locations_valid,
        "head_score": float(head_score) if head_score is not None else None,
        "shoulder_score": (
            float(shoulder_score) if shoulder_score is not None else None
        ),
        "upper_body_support_logit_min": HIGH_OCCLUSION_REGION_SUPPORT_LOGIT_MIN,
        "upper_body_support_valid": upper_body_support_valid,
        "pose_score": float(pose_score) if pose_score is not None else None,
        "pose_score_min": HIGH_OCCLUSION_POSE_SCORE_MIN,
        "pose_support_valid": pose_support_valid,
        "bollard_pose_score_min": BOLLARD_OCCLUSION_POSE_SCORE_MIN,
        "bollard_pose_support_valid": bollard_pose_support_valid,
        "bollard_hips_bounded": bollard_hips_bounded,
        "hip_pair_distance_px": round(hip_pair_distance, 3),
        "bollard_hip_pair_distance_fraction_max": (
            BOLLARD_HIP_PAIR_DISTANCE_FRACTION_MAX
        ),
        "bollard_hips_aligned": bollard_hips_aligned,
        "railing_hip_logit_min": RAILING_OCCLUSION_HIP_LOGIT_MIN,
        "railing_hip_support_logit_min": RAILING_OCCLUSION_HIP_SUPPORT_LOGIT_MIN,
        "railing_hips_bounded": railing_hips_bounded,
    }


def _pose_detector_fallback(
    boxes: Sequence[Sequence[float]],
    pose_scores: Sequence[float],
    score_rows: Sequence[Sequence[float]],
    occluded: bool,
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Recover a missed detector box from one unambiguous complete pose."""
    candidate_count = len(boxes)
    if candidate_count != 1 or len(pose_scores) != 1 or len(score_rows) != 1:
        return {
            "pass": False,
            "used": False,
            "candidate_count": candidate_count,
            "reason": "pose_candidate_count_not_one",
        }
    box = boxes[0]
    x0, y0, x1, y1 = (float(value) for value in box)
    image_width, image_height = (float(value) for value in image_size)
    width_fraction = (x1 - x0) / max(image_width, 1e-9)
    height_fraction = (y1 - y0) / max(image_height, 1e-9)
    region_scores = _required_pose_region_scores(score_rows[0], occluded)
    required_regions_visible = all(
        float(score) >= KEYPOINT_VISIBILITY_LOGIT_MIN
        for score in region_scores.values()
    )
    box_in_frame = (
        2.0 <= x0 < x1 <= image_width - 2.0
        and 2.0 <= y0 < y1 <= image_height - 2.0
    )
    size_valid = (
        POSE_DETECTOR_FALLBACK_WIDTH_FRACTION_MIN
        <= width_fraction
        <= POSE_DETECTOR_FALLBACK_FRACTION_MAX
        and POSE_DETECTOR_FALLBACK_HEIGHT_FRACTION_MIN
        <= height_fraction
        <= POSE_DETECTOR_FALLBACK_FRACTION_MAX
    )
    passed = (
        float(pose_scores[0]) >= POSE_DETECTOR_FALLBACK_SCORE_MIN
        and required_regions_visible
        and box_in_frame
        and size_valid
    )
    bbox = [
        math.floor(x0),
        math.floor(y0),
        math.ceil(x1) - math.floor(x0),
        math.ceil(y1) - math.floor(y0),
    ]
    return {
        "pass": passed,
        "used": passed,
        "candidate_count": candidate_count,
        "bbox": bbox,
        "pose_box_xyxy": [round(value, 3) for value in (x0, y0, x1, y1)],
        "pose_score": float(pose_scores[0]),
        "pose_score_min": POSE_DETECTOR_FALLBACK_SCORE_MIN,
        "region_scores": region_scores,
        "required_regions_visible": required_regions_visible,
        "box_in_frame": box_in_frame,
        "width_fraction": width_fraction,
        "width_fraction_min": POSE_DETECTOR_FALLBACK_WIDTH_FRACTION_MIN,
        "height_fraction": height_fraction,
        "height_fraction_min": POSE_DETECTOR_FALLBACK_HEIGHT_FRACTION_MIN,
        "fraction_max": POSE_DETECTOR_FALLBACK_FRACTION_MAX,
        "size_valid": size_valid,
        "reason": None if passed else "pose_candidate_not_safe",
    }


def _pose_geometry(
    image: Any, sample: Mapping[str, Any], bbox: tuple[int, int, int, int] | None
) -> dict[str, Any]:
    """Fail-closed COCO keypoint check for the required visible body regions."""
    import torch

    model, transform, device, weights_name = _load_torchvision_qa_model("pose")
    tensor = _torchvision_input(image, transform, device)
    with torch.inference_mode():
        output = model([tensor])[0]
    mask = (output["labels"] == 1) & (output["scores"] >= POSE_SCORE_MIN)
    indices = torch.where(mask)[0].detach().cpu().tolist()
    if not indices:
        return {"available": True, "backend": "torchvision_keypointrcnn_resnet50_fpn",
                "weights": weights_name, "pass": False, "reason": "pose_not_detected"}
    if "keypoints_scores" in output:
        score_rows = {
            index: output["keypoints_scores"][index].detach().cpu().tolist()
            for index in indices
        }
    else:  # pragma: no cover - compatibility with older torchvision
        score_rows = {
            index: output["keypoints"][index, :, 2].detach().cpu().tolist()
            for index in indices
        }
    occluded = bool(sample["occluded"])
    pose_detector = None
    if bbox is None:
        pose_detector = _pose_detector_fallback(
            [output["boxes"][index].detach().cpu().tolist() for index in indices],
            [float(output["scores"][index].item()) for index in indices],
            [score_rows[index] for index in indices],
            occluded,
            (int(image.shape[1]), int(image.shape[0])),
        )
        if not pose_detector["pass"]:
            return {
                "available": True,
                "backend": "torchvision_keypointrcnn_resnet50_fpn",
                "weights": weights_name,
                "pass": False,
                "reason": "person_bbox_unavailable",
                "pose_detector_fallback": pose_detector,
            }
        geometry_bbox = tuple(int(value) for value in pose_detector["bbox"])
        selected_index = indices[0]
        matches = [
            (
                1.0,
                float(output["scores"][selected_index].item()),
                selected_index,
            )
        ]
    else:
        x, y, width, height = bbox
        principal_xyxy = (x, y, x + width, y + height)
        matches = [
            (
                _xyxy_iou(
                    principal_xyxy,
                    output["boxes"][index].detach().cpu().tolist(),
                ),
                float(output["scores"][index].item()),
                index,
            )
            for index in indices
        ]
        geometry_bbox = bbox
    primary_match_iou, primary_pose_score, primary_selected = max(matches)
    primary_region_scores = _required_pose_region_scores(
        score_rows[primary_selected], occluded
    )
    pose_bbox_expansion = None
    if bbox is not None and primary_match_iou < POSE_MATCH_IOU_MIN:
        pose_bbox_expansion = _pose_bbox_expansion(
            bbox,
            output["boxes"][primary_selected].detach().cpu().tolist(),
            primary_pose_score,
            primary_region_scores,
            (int(image.shape[1]), int(image.shape[0])),
            occluded,
            occluder=str(sample.get("occluder") or ""),
            target_occlusion_ratio=float(sample.get("target_occlusion_ratio", 0.0)),
        )
        if not pose_bbox_expansion["pass"]:
            return {
                "available": True,
                "backend": "torchvision_keypointrcnn_resnet50_fpn",
                "weights": weights_name,
                "pass": False,
                "reason": "pose_does_not_match_principal_person",
                "match_iou": primary_match_iou,
                "pose_score": primary_pose_score,
                "pose_bbox_expansion": pose_bbox_expansion,
            }
        geometry_bbox = tuple(int(value) for value in pose_bbox_expansion["expanded_bbox"])
    (match_iou, pose_score, selected), alternate_pose = _select_pose_candidate(
        matches, score_rows, occluded
    )
    scores = score_rows[selected]
    points = output["keypoints"][selected, :, :2].detach().cpu().tolist()

    region_scores = _required_pose_region_scores(scores, occluded)
    if (
        bbox is not None
        and alternate_pose is not None
        and not occluded
        and pose_bbox_expansion is None
    ):
        alternate_expansion = _pose_bbox_expansion(
            bbox,
            output["boxes"][selected].detach().cpu().tolist(),
            pose_score,
            region_scores,
            (int(image.shape[1]), int(image.shape[0])),
            occluded=False,
        )
        if alternate_expansion["pass"]:
            pose_bbox_expansion = alternate_expansion
            geometry_bbox = tuple(
                int(value) for value in pose_bbox_expansion["expanded_bbox"]
            )
    regions = {name: score >= KEYPOINT_VISIBILITY_LOGIT_MIN for name, score in region_scores.items()}
    back_view_head = None
    if not regions["head"] and _is_back_facing_sample(sample):
        back_view_head = _back_view_head_visibility(
            scores[0],
            points[0],
            geometry_bbox,
            (int(image.shape[1]), int(image.shape[0])),
            supporting_shoulder_score=region_scores["shoulders"],
        )
        regions["head"] = bool(back_view_head["pass"])
        back_view_head["used"] = bool(back_view_head["pass"])
    overlapped_ankle = None
    occluded_waist = None
    if occluded and not regions["waist"]:
        occluded_waist = _occluded_waist_visibility(
            [scores[11], scores[12]],
            [points[11], points[12]],
            geometry_bbox,
            (int(image.shape[1]), int(image.shape[0])),
            float(sample.get("target_occlusion_ratio", 0.0)),
            occluder=str(sample.get("occluder") or ""),
            head_score=region_scores["head"],
            shoulder_score=region_scores["shoulders"],
            pose_score=pose_score,
        )
        regions["waist"] = bool(occluded_waist["pass"])
        occluded_waist["used"] = bool(occluded_waist["pass"])
    if not occluded and not regions["feet"]:
        overlapped_ankle = _overlapped_ankle_visibility(
            [scores[15], scores[16]],
            [points[15], points[16]],
            geometry_bbox,
            (int(image.shape[1]), int(image.shape[0])),
            knee_scores=[scores[13], scores[14]],
            knee_points=[points[13], points[14]],
            walking_pose=(
                "walking" in str(sample.get("pose", "")).lower()
                or "mid-step" in str(sample.get("pose", "")).lower()
            ),
            pose_score=pose_score,
        )
        regions["feet"] = bool(overlapped_ankle["pass"])
        overlapped_ankle["used"] = bool(overlapped_ankle["pass"])
    passed = all(regions.values())
    result = {
        "available": True,
        "backend": "torchvision_keypointrcnn_resnet50_fpn",
        "weights": weights_name,
        "pass": passed,
        "regions": regions,
        "region_scores": region_scores,
        "visibility_logit_threshold": KEYPOINT_VISIBILITY_LOGIT_MIN,
        "match_iou": match_iou,
        "pose_score": pose_score,
        "pose_candidate_count": len(matches),
        "selected_pose_candidate_index": selected,
        "reason": None if passed else "required_landmarks_not_visible",
    }
    if overlapped_ankle is not None:
        result["overlapped_ankle_fallback"] = overlapped_ankle
    if occluded_waist is not None:
        result["occluded_waist_fallback"] = occluded_waist
    if back_view_head is not None:
        result["back_view_head_fallback"] = back_view_head
    if alternate_pose is not None:
        result["split_pose_candidate_fallback"] = alternate_pose
    if pose_detector is not None:
        result["pose_detector_fallback"] = pose_detector
        result["crop_bbox_override"] = list(geometry_bbox)
    if pose_bbox_expansion is not None:
        result["pose_bbox_expansion"] = pose_bbox_expansion
        result["crop_bbox_override"] = list(geometry_bbox)
    return result


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
    pose_geometry = _pose_geometry(image, sample, bbox)
    cart_fragment = _intentional_cart_person_fragment(
        detection, pose_geometry, sample
    )
    crop_bbox = bbox
    if pose_geometry.get("crop_bbox_override") is not None:
        crop_bbox = tuple(int(value) for value in pose_geometry["crop_bbox_override"])
    pose_detector = pose_geometry.get("pose_detector_fallback") or {}
    effective_person_count = (
        1
        if cart_fragment["pass"] or bool(pose_detector.get("pass"))
        else person_count
    )
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
        bgr, crop_bbox, float(config["dataset"]["bbox_margin"])
    )
    bgr = bgr.astype(np.uint8)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(final_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(isp["jpeg_quality"])])
    if not ok:
        return {"decode": True, "accepted": False, "reason": "jpeg_write_failed"}
    principal_person_detected = crop_bbox is not None and effective_person_count == 1
    framing_min = float(config["dataset"]["framing_fill_min"])
    framing["minimum_bbox_fill"] = framing_min
    framing["pass"] = (
        crop_bbox is not None
        and float(framing["bbox_width_fill"]) >= framing_min
        and float(framing["bbox_height_fill"]) >= framing_min
    )
    geometry = principal_person_detected and bool(pose_geometry["pass"]) and bool(framing["pass"])
    if crop_bbox is None:
        reason = "person_not_detected"
    elif effective_person_count != 1:
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
        "person_bbox": list(crop_bbox) if crop_bbox else None,
        "detector_person_bbox": list(bbox) if bbox else None,
        "person_detection_count": effective_person_count,
        "person_detection": detection,
        "intentional_cart_fragment_fallback": cart_fragment,
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
    geometry_by_camera: dict[int, bytes] = {}
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
        geometry = row.get("camera_geometry") or {}
        try:
            pitch = float(geometry["pitch_deg"])
            pitch_down = float(geometry["pitch_down_deg"])
            mounting_height = float(geometry["mounting_height_m"])
            horizon = float(geometry["horizon_y_fraction"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{sample_id}: missing numeric camera geometry")
        else:
            if geometry.get("version") != CAMERA_GEOMETRY_VERSION:
                errors.append(f"{sample_id}: unexpected camera geometry version")
            if pitch >= 0 or not math.isclose(pitch_down, -pitch, abs_tol=0.011):
                errors.append(f"{sample_id}: invalid downward camera pitch")
            if mounting_height < 2.0 or not 0.03 <= horizon <= 0.45:
                errors.append(f"{sample_id}: camera height or horizon is outside the safe range")
            profile = canonical_json(geometry)
            prior = geometry_by_camera.setdefault(camera, profile)
            if prior != profile:
                errors.append(f"{sample_id}: camera {camera} geometry changed within the dataset")
        if row.get("prompt_version") != PROMPT_VERSION:
            errors.append(f"{sample_id}: unexpected camera-pitch prompt version")
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
        if len(geometry_by_camera) != 33 or len(set(geometry_by_camera.values())) != 33:
            errors.append("expected 33 unique, camera-fixed geometry profiles")
        for pid, items in per_pid.items():
            if len(items) != 40 or len({int(r["local_camera"]) for r in items}) != 8:
                errors.append(f"pid {pid}: expected 40 accepted images across 8 cameras")
            yaw_counts = Counter(int(row.get("body_yaw_deg", 999)) for row in items)
            if yaw_counts != Counter({degrees: 5 for degrees in BODY_YAW_DEGREES}):
                errors.append(f"pid {pid}: expected five accepted images in every body-yaw bin")
            train_items = [row for row in items if row["split"] == "train"]
            queries = [row for row in items if row["split"] == "query"]
            galleries = [row for row in items if row["split"] == "gallery"]
            if train_items and sum(bool(row.get("occluded")) for row in train_items) != 12:
                errors.append(f"pid {pid}: expected 12 occluded train images")
            if queries and (len(queries) != 4 or not all(row.get("occluded") for row in queries)):
                errors.append(f"pid {pid}: expected four occluded queries")
            query_yaws = {
                BODY_YAW_DEGREES.index(int(row.get("body_yaw_deg", 999)))
                for row in queries
                if int(row.get("body_yaw_deg", 999)) in BODY_YAW_DEGREES
            }
            if queries and (len(query_yaws) != 4 or len({index % 2 for index in query_yaws}) != 1):
                errors.append(f"pid {pid}: query body yaws do not cover four separated quadrants")
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
                and float(review.get("camera_pitch_consistency_rate", 0)) >= 0.95
                and int(review.get("camera_geometry_failures", 0)) == 0
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
