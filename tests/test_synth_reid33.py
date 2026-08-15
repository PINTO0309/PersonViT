from __future__ import annotations

import base64
import io
import json
import importlib
import sys
import types
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "transreid_pytorch" / "tools"
sys.path.insert(0, str(TOOLS))

from build_unified_dataset import validate_explicit_domain, validate_unified_output  # noqa: E402
from calibrate_synth_reid33_similarity import (  # noqa: E402
    collect_protocol_records,
    cross_camera_positive_values,
    distribution_summary,
)
from generate_synth_reid33 import (  # noqa: E402
    PipelineError,
    WAIVABLE_APPROVAL_GATES,
    _approval_failed_gates,
    _batch_failure_summaries,
    _full_embedding_acceptance_models,
    _merge_jobs,
    _paths,
    _refresh_and_collect,
    _prepare_retries,
    _repair_candidate_rank,
    _sample_jobs,
    _write_reference_asset,
    enforce_cost_ceiling,
    initialize,
    repair_local,
    repair_pilot,
    report_local_failures,
)
from probe_synth_reid33_reference_sizes import (  # noqa: E402
    REFERENCE_VARIANTS,
    STANDARD_PRICING,
    _extract_usage,
    _request_image,
    _usage_cost,
    prepare_references,
)
from evaluate_synth_reid33_adoption import evaluate_adoption  # noqa: E402
from synth_reid33_core import (  # noqa: E402
    allocate_identity_cameras,
    allocate_pilot_cameras,
    approval_payload,
    atomic_write_jsonl,
    classify_batch_item,
    load_config,
    make_cameras,
    make_identities,
    make_pilot_samples,
    make_rotation_pilot_samples,
    make_samples,
    plate_prompt,
    _crop_resize_person,
    _nms_person_candidates,
    _overlapped_ankle_visibility,
    process_image,
    read_jsonl,
    reconcile_batch_output,
    sha256_file,
    validate_pilot,
    validate_camera_geometry,
    validate_plan,
    validate_rotation_pilot,
    validate_synthetic_manifest,
    verify_approval,
)


CONFIG_PATH = REPO / "transreid_pytorch" / "configs" / "synth_reid33.yml"


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG_PATH)


def test_sample_generation_uses_cost_minimizing_portrait_size(config):
    assert config["model"]["api_id"] == "gpt-image-2"
    assert config["model"]["catalog_snapshot"] == "gpt-image-2-2026-04-21"
    assert config["model"]["quality"] == "low"
    assert config["model"]["sample_size"] == "576x1152"
    assert config["model"]["reference_profile"] == "native_quarter"
    assert config["model"]["reference_anchor_size"] == "256x384"
    assert config["model"]["reference_plate_size"] == "144x288"
    assert (config["dataset"]["final_width"], config["dataset"]["final_height"]) == (128, 256)
    assert config["dataset"]["bbox_margin"] == 0.05
    assert config["dataset"]["framing_fill_min"] == 0.85
    assert config["prompt"]["version"] == "synth-reid33-prompt-v3-eight-yaw-camera-pitch"
    assert config["qa"]["camera_pitch_consistency_min"] == 0.95
    assert [view["degrees"] for view in config["body_rotation"]["views"]] == [
        0, 45, 90, 135, 180, -135, -90, -45,
    ]


@pytest.mark.parametrize(
    ("kind", "source_size", "reference_size"),
    [
        ("anchor", (1024, 1536), (256, 384)),
        ("plate", (576, 1152), (144, 288)),
    ],
)
def test_native_quarter_reference_asset_preserves_source(
    tmp_path, config, kind, source_size, reference_size
):
    source = tmp_path / f"{kind}-source.jpg"
    reference = tmp_path / f"{kind}-reference.jpg"
    Image.new("RGB", source_size, (90, 100, 110)).save(source, quality=95)
    source_sha = sha256_file(source)

    metadata = _write_reference_asset(source, reference, kind, config)

    assert Image.open(source).size == source_size
    assert sha256_file(source) == source_sha
    assert Image.open(reference).size == reference_size
    assert metadata["profile"] == "native_quarter"
    assert metadata["reference_size"] == list(reference_size)
    assert metadata["reference_sha256"] == sha256_file(reference)


def test_asset_collection_uploads_native_quarter_copy(tmp_path, config):
    paths = _paths(tmp_path)
    paths["state"].mkdir(parents=True)
    job = {
        "custom_id": "asset-anchor-p000-front-a1",
        "scope": "asset",
        "kind": "anchor",
        "asset_id": "anchor-p000-front",
        "local_pid": 0,
        "orientation": "front",
        "status": "submitted",
        "attempt": 1,
        "body": {"quality": "low"},
        "logical_refs": [],
    }
    atomic_write_jsonl(paths["jobs"], [job])
    atomic_write_jsonl(paths["batches"], [{
        "batch_id": "batch-a",
        "status": "in_progress",
        "custom_ids": [job["custom_id"]],
    }])
    buffer = io.BytesIO()
    Image.new("RGB", (1024, 1536), (50, 60, 70)).save(buffer, format="JPEG")
    payload = json.dumps({
        "custom_id": job["custom_id"],
        "response": {
            "status_code": 200,
            "request_id": "request-a",
            "body": {
                "model": "gpt-image-2",
                "data": [{"b64_json": base64.b64encode(buffer.getvalue()).decode()}],
            },
        },
    }).encode() + b"\n"

    class FakeClient:
        uploaded_size = None

        def retrieve(self, _batch_id):
            return {"id": "batch-a", "status": "completed", "output_file_id": "out-a"}

        def content(self, _file_id):
            return payload

        def upload(self, path, purpose):
            assert purpose == "vision"
            self.uploaded_size = Image.open(path).size
            return "file-native-quarter"

    client = FakeClient()
    _refresh_and_collect(client, tmp_path, config, paths)
    asset = read_jsonl(paths["assets"])[0]

    assert client.uploaded_size == (256, 384)
    assert Image.open(tmp_path / asset["path"]).size == (1024, 1536)
    assert Image.open(tmp_path / asset["reference_path"]).size == (256, 384)
    assert asset["profile"] == "native_quarter"
    assert asset["file_id"] == "file-native-quarter"


def test_reference_size_probe_prepares_every_declared_variant(tmp_path):
    source = tmp_path / "source" / "assets"
    source.mkdir(parents=True)
    Image.new("RGB", (1024, 1536), (90, 100, 110)).save(
        source / "anchor-p000-front.jpg"
    )
    Image.new("RGB", (576, 1152), (120, 130, 140)).save(source / "plate-c00.jpg")

    output = tmp_path / "probe"
    rows = prepare_references(source.parent, output)

    assert len(rows) == len(REFERENCE_VARIANTS) == 7
    for row in rows:
        assert Image.open(output / row["anchor_path"]).size == tuple(row["anchor_size"])
        assert Image.open(output / row["plate_path"]).size == tuple(row["plate_size"])


def test_reference_size_probe_usage_and_standard_cost():
    usage = _extract_usage({
        "usage": {
            "input_tokens": 1100,
            "input_tokens_details": {"text_tokens": 100, "image_tokens": 1000},
            "output_tokens": 86,
        }
    })

    assert usage["input_tokens_unclassified"] == 0
    assert _usage_cost(usage, STANDARD_PRICING) == pytest.approx(0.01108)


def test_reference_size_probe_uses_implicit_base64_response(tmp_path):
    anchor = tmp_path / "anchor.jpg"
    plate = tmp_path / "plate.jpg"
    _save_image(anchor, (1, 2, 3))
    _save_image(plate, (4, 5, 6))

    class Images:
        kwargs = None

        def edit(self, **kwargs):
            self.kwargs = kwargs
            assert all(not image.closed for image in kwargs["image"])
            return "response"

    client = types.SimpleNamespace(images=Images())
    assert _request_image(client, anchor, plate) == "response"
    assert "response_format" not in client.images.kwargs
    assert client.images.kwargs["output_format"] == "jpeg"
    assert client.images.kwargs["quality"] == "low"
    assert all(image.closed for image in client.images.kwargs["image"])


def test_deterministic_full_allocation_meets_acceptance(config):
    allocation, targets = allocate_identity_cameras(config)
    identities = make_identities(config)
    cameras = make_cameras(config)
    samples = make_samples(config, allocation)
    report = validate_plan(config, identities, cameras, samples)

    assert report["valid"], report["errors"]
    assert report["camera_geometry"]["valid"]
    assert Counter(targets) == {120: 2, 121: 22, 122: 9}
    assert report["counts"] == {"train": 16000, "query": 400, "gallery": 3600}
    assert set(sample["global_camera"] for sample in samples) == set(range(33, 66))
    assert report["body_yaw_counts"] == {
        -135: 2500, -90: 2500, -45: 2500, 0: 2500,
        45: 2500, 90: 2500, 135: 2500, 180: 2500,
    }
    for pid in range(500):
        rows = [sample for sample in samples if sample["local_pid"] == pid]
        assert Counter(row["body_yaw_deg"] for row in rows) == {
            degrees: 5 for degrees in (0, 45, 90, 135, 180, -135, -90, -45)
        }
        queries = [row for row in rows if row["split"] == "query"]
        if queries:
            indices = {
                (0, 45, 90, 135, 180, -135, -90, -45).index(row["body_yaw_deg"])
                for row in queries
            }
            assert len(indices) == 4
            assert len({index % 2 for index in indices}) == 1


def test_all_33_cameras_have_fixed_unique_pitch_height_and_horizon(config):
    cameras = make_cameras(config)
    assert cameras == make_cameras(config)
    report = validate_camera_geometry(config, cameras)
    assert report["valid"], report["errors"]
    assert len({
        (
            camera["geometry"]["pitch_deg"],
            camera["geometry"]["mounting_height_m"],
            camera["geometry"]["horizon_y_fraction"],
        )
        for camera in cameras
    }) == 33
    for camera in cameras:
        geometry = camera["geometry"]
        assert geometry["version"] == "synth-reid33-camera-geometry-v1"
        assert geometry["pitch_deg"] < 0
        assert geometry["pitch_down_deg"] == -geometry["pitch_deg"]
        assert 0.03 <= geometry["horizon_y_fraction"] <= 0.45
        prompt = plate_prompt(camera, config)
        assert f"{geometry['mounting_height_m']:.2f} meters" in prompt
        assert f"{geometry['pitch_down_deg']:.2f} degrees" in prompt
        assert f"{round(geometry['horizon_y_fraction'] * 100)}%" in prompt
    tampered = json.loads(json.dumps(cameras))
    tampered[0]["geometry"]["pitch_deg"] = 5.0
    assert not validate_camera_geometry(config, tampered)["valid"]


def test_pilot_has_96_low_candidates_and_covers_every_camera(config):
    pilot = make_pilot_samples(config, allocate_pilot_cameras(config))
    report = validate_pilot(pilot)
    assert report["valid"], report["errors"]
    assert len(pilot) == 96
    assert {sample["quality"] for sample in pilot} == {"low"}
    assert min(report["camera_coverage"].values()) >= 2


def test_rotation_pilot_is_low_only_balanced_and_uses_nearest_cardinal_anchor(config):
    identities = make_identities(config)
    cameras = make_cameras(config)
    samples = make_rotation_pilot_samples(config, allocate_pilot_cameras(config))
    report = validate_rotation_pilot(config, samples)

    assert report["valid"], report["errors"]
    assert len(samples) == 96
    assert set(report["body_yaw_counts"].values()) == {12}
    jobs = _sample_jobs(
        config, samples, identities, cameras, "rotation_pilot", quality="low"
    )
    by_id = {sample["sample_id"]: sample for sample in samples}
    cameras_by_id = {camera["local_camera"]: camera for camera in cameras}
    for job in jobs:
        sample = by_id[job["sample_id"]]
        geometry = cameras_by_id[sample["local_camera"]]["geometry"]
        assert job["quality"] == "low"
        assert job["logical_refs"][0].endswith(f"-{sample['anchor_orientation']}")
        assert f"yaw {sample['body_yaw_deg']:+d} degrees" in job["body"]["prompt"]
        assert f"{geometry['pitch_down_deg']:.2f} degree downward tilt" in job["body"]["prompt"]


def test_batch_results_are_reconciled_by_custom_id_not_order():
    payload = "\n".join([
        json.dumps({"custom_id": "job-b", "response": {"status_code": 200}}),
        json.dumps({"custom_id": "job-a", "response": {"status_code": 200}}),
    ])
    found, missing = reconcile_batch_output(payload, ["job-a", "job-b"])
    assert list(found) == ["job-b", "job-a"]
    assert not missing


def test_duplicate_batch_result_is_rejected():
    line = json.dumps({"custom_id": "job-a", "response": {"status_code": 200}})
    with pytest.raises(ValueError, match="duplicate"):
        reconcile_batch_output(f"{line}\n{line}\n", ["job-a"])


@pytest.mark.parametrize(
    ("item", "classification"),
    [
        ({"response": {"status_code": 400, "body": {"error": {"code": "moderation_blocked"}}}}, "revise_once"),
        ({"response": {"status_code": 503, "body": {"error": {"code": "server_error"}}}}, "retryable"),
        ({"response": {"status_code": 422, "body": {"error": {"code": "invalid_image"}}}}, "terminal"),
    ],
)
def test_generation_failures_are_classified(item, classification):
    assert classify_batch_item(item) == classification


def test_person_nms_suppresses_nested_duplicate_but_keeps_independent_person():
    boxes = [
        [10, 10, 110, 210],
        [20, 30, 100, 200],
        [300, 20, 390, 210],
    ]
    scores = [0.99, 0.85, 0.90]
    assert _nms_person_candidates(boxes, scores) == [0, 2]


def test_person_nms_suppresses_low_iou_contained_box_from_real_pilot_regression():
    boxes = [
        [159.0302, 363.2432, 305.0127, 856.6233],
        [144.9192, 362.2342, 285.9600, 541.8674],
        [400.0, 300.0, 500.0, 750.0],
    ]
    scores = [0.9843, 0.5410, 0.90]

    # The first two boxes have IoU 0.304, below ordinary NMS, but 89.5% of
    # the smaller upper-body box is contained by the full-body detection.
    assert _nms_person_candidates(boxes, scores) == [0, 2]


@pytest.mark.parametrize(
    ("scores", "points", "bbox", "expected_reason"),
    [
        ([-2.01, 6.0], [[150, 850], [250, 870]], (100, 100, 200, 800), "weak"),
        ([-1.0, -0.2], [[150, 850], [250, 870]], (100, 100, 200, 800), "no_strong"),
        ([-1.0, 6.0], [[150, 500], [250, 870]], (100, 100, 200, 800), "not_lower"),
        ([-1.0, 6.0], [[150, 1000], [250, 1050]], (100, 350, 200, 800), "clipped"),
    ],
)
def test_overlapped_ankle_fallback_rejects_unsafe_cases(
    scores, points, bbox, expected_reason
):
    result = _overlapped_ankle_visibility(scores, points, bbox, (576, 1152))

    assert not result["pass"], expected_reason


@pytest.mark.parametrize(
    ("scores", "points", "bbox"),
    [
        ([-1.4380, 11.1854], [[313.9, 765.3], [300.1, 789.5]], (206, 273, 199, 581)),
        ([-0.4722, 6.7960], [[396.3, 762.8], [382.5, 772.2]], (356, 413, 110, 389)),
    ],
)
def test_overlapped_ankle_fallback_accepts_visible_rotation_pilot_regressions(
    scores, points, bbox
):
    result = _overlapped_ankle_visibility(scores, points, bbox, (576, 1152))

    assert result["pass"]
    assert result["strong_ankle_visible"]
    assert result["ankle_locations_valid"] == [True, True]
    assert result["person_not_bottom_clipped"]


def test_person_crop_uses_five_percent_tight_box_and_direct_reid_resize():
    import numpy as np

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:80, 45:55] = (0, 0, 255)
    crop, framing = _crop_resize_person(image, (45, 20, 10, 60), margin=0.05)

    assert crop.shape == (256, 128, 3)
    assert framing["policy"] == "tight_bbox_direct_resize"
    assert framing["crop_box"] == [44, 17, 56, 83]
    assert framing["bbox_width_fill"] == pytest.approx(10 / 12)
    assert framing["bbox_height_fill"] == pytest.approx(60 / 66)
    assert crop[:, :4, 2].mean() < crop[:, 12:116, 2].mean()


def test_real_similarity_calibration_excludes_gallery_only_distractor(tmp_path):
    import numpy as np

    query = tmp_path / "query"
    gallery = tmp_path / "gallery"
    query.mkdir()
    gallery.mkdir()
    for directory, name in (
        (query, "p00001_d00_c000_000001.jpg"),
        (gallery, "p00001_d00_c001_000002.jpg"),
        (gallery, "p00001_d00_c001_000003.jpg"),
        (gallery, "p00099_d00_c000_000004.jpg"),
        (gallery, "p00099_d00_c001_000005.jpg"),
    ):
        _save_image(directory / name, (1, 2, 3))
    records, metadata = collect_protocol_records(tmp_path)
    features = np.asarray([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8]], dtype=np.float32)
    values, by_domain = cross_camera_positive_values(features, records)

    assert metadata["query_identities"] == 1
    assert metadata["excluded_gallery_only_identities"] == 1
    assert len(records) == 3
    assert sorted(values.tolist()) == pytest.approx([0.6, 0.8])
    assert distribution_summary(values)["pair_count"] == 2
    assert set(by_domain) == {"d00"}


def test_repair_candidate_rank_prefers_geometry_then_visibility_then_first_attempt():
    failed = {"accepted": False, "pose_geometry": {"region_scores": {"head": 20}},
              "person_detection": {"principal_score": 0.999}}
    accepted_low = {"accepted": True, "pose_geometry": {"region_scores": {"head": 2}},
                    "person_detection": {"principal_score": 0.99}}
    accepted_high = {"accepted": True, "pose_geometry": {"region_scores": {"head": 5}},
                     "person_detection": {"principal_score": 0.95}}
    assert _repair_candidate_rank({"attempt": 1}, failed) < _repair_candidate_rank(
        {"attempt": 1}, accepted_low
    )
    assert _repair_candidate_rank({"attempt": 1}, accepted_low) < _repair_candidate_rank(
        {"attempt": 2}, accepted_high
    )
    assert _repair_candidate_rank({"attempt": 2}, accepted_high) < _repair_candidate_rank(
        {"attempt": 1}, accepted_high
    )


def test_process_image_uses_principal_detector_and_keypoint_geometry(
    tmp_path, config, monkeypatch
):
    import synth_reid33_core as core

    raw = tmp_path / "raw.jpg"
    Image.new("RGB", (576, 1152), (120, 130, 140)).save(raw)
    final = tmp_path / "final.jpg"
    monkeypatch.setattr(core, "_person_detection", lambda _image: {
        "bbox": (180, 150, 220, 850), "count": 1, "candidate_count_before_nms": 2,
        "principal_score": 0.99, "backend": "test", "weights": "test",
        "score_threshold": 0.35, "nms_iou_threshold": 0.35,
    })
    monkeypatch.setattr(core, "_pose_geometry", lambda _image, _occluded, _bbox: {
        "available": True, "backend": "test", "pass": True,
        "regions": {"head": True, "shoulders": True, "feet": True},
        "region_scores": {"head": 5.0, "shoulders": 5.0, "feet": 5.0},
        "reason": None,
    })
    camera = make_cameras(config)[0]
    sample = {"occluded": False, "generation_seed": 123}
    qa = process_image(raw, final, camera, sample, config)
    assert qa["accepted"]
    assert qa["principal_person_detected"]
    assert qa["person_detection_count"] == 1
    assert qa["final_size"] == [128, 256]
    assert Image.open(final).size == (128, 256)


def test_repair_pilot_atomically_rebuilds_complete_manifest(tmp_path, config, monkeypatch):
    import generate_synth_reid33 as generator

    paths = initialize(tmp_path, config)
    samples = read_jsonl(paths["pilot_samples"])
    assets = [
        {"asset_id": f"asset-{index}", "sha256": f"sha-{index}", "file_id": f"file-{index}"}
        for index in range(81)
    ]
    jobs = []
    attempts = []
    raw_dir = tmp_path / "raw" / "pilot"
    raw_dir.mkdir(parents=True)
    for index, sample in enumerate(samples):
        custom_id = f"repair-{index:03d}-a1"
        jobs.append({
            "custom_id": custom_id,
            "scope": "pilot",
            "kind": "sample",
            "sample_id": sample["sample_id"],
            "status": "qa_failed",
            "attempt": 1,
            "quality": sample["quality"],
            "logical_refs": ["asset-0"],
            "body": {"prompt": f"prompt {index}"},
        })
        attempts.append({
            "custom_id": custom_id,
            "classification": "succeeded",
            "response_model": "gpt-image-2",
            "request_id": f"request-{index}",
            "batch_id": "batch-complete",
            "usage": {},
            "estimated_cost_usd": 0.01,
        })
        (raw_dir / f"{custom_id}.jpg").write_bytes(f"raw-{index}".encode())
    atomic_write_jsonl(paths["assets"], assets)
    atomic_write_jsonl(paths["jobs"], jobs)
    atomic_write_jsonl(paths["attempts"], attempts)
    atomic_write_jsonl(paths["batches"], [{
        "batch_id": "batch-complete", "status": "completed", "collected": True,
    }])

    def fake_process(raw_path, final_path, _camera, _sample, _config):
        final_path.write_bytes(raw_path.read_bytes())
        return {
            "accepted": True,
            "processing_version": "test-qa",
            "person_detection": {"principal_score": 0.99},
            "pose_geometry": {"region_scores": {"head": 5.0}},
            "final_sha256": sha256_file(final_path),
        }

    monkeypatch.setattr(generator, "process_image", fake_process)
    report = repair_pilot(tmp_path, config)
    manifest = read_jsonl(paths["pilot_manifest"])
    repaired_jobs = read_jsonl(paths["jobs"])
    repaired_attempts = read_jsonl(paths["attempts"])

    assert report["complete"]
    assert report["selected"] == 96
    assert len(manifest) == 96
    assert all(row["status"] == "succeeded" for row in repaired_jobs)
    assert all(row["selected_by_repair"] for row in repaired_attempts)
    assert all((tmp_path / row["final_path"]).is_file() for row in manifest)


def test_expired_batch_is_resumable_without_duplicate_submission(tmp_path, config):
    paths = _paths(tmp_path)
    paths["state"].mkdir(parents=True)
    atomic_write_jsonl(paths["jobs"], [{
        "custom_id": "job-a", "scope": "pilot", "kind": "sample", "sample_id": "x",
        "status": "submitted", "attempt": 1, "body": {}, "logical_refs": [],
    }])
    atomic_write_jsonl(paths["batches"], [{
        "batch_id": "batch-a", "status": "in_progress", "custom_ids": ["job-a"]
    }])

    class FakeClient:
        def retrieve(self, _batch_id):
            return {"id": "batch-a", "status": "expired", "output_file_id": None}

    _refresh_and_collect(FakeClient(), tmp_path, config, paths)
    assert json.loads(paths["jobs"].read_text().splitlines()[0])["status"] == "expired_failed"
    additions = [json.loads(paths["jobs"].read_text().splitlines()[0])]
    merged = _merge_jobs(paths["jobs"], additions)
    assert len(merged) == 1


def test_batch_validation_failure_is_terminal_and_preserves_error(tmp_path, config):
    paths = _paths(tmp_path)
    paths["state"].mkdir(parents=True)
    job = {
        "custom_id": "job-a", "scope": "asset", "kind": "anchor", "asset_id": "a",
        "status": "submitted", "attempt": 1, "body": {}, "logical_refs": [],
    }
    atomic_write_jsonl(paths["jobs"], [job])
    atomic_write_jsonl(paths["batches"], [{
        "batch_id": "batch-a", "status": "validating", "custom_ids": ["job-a"]
    }])

    class FakeClient:
        def retrieve(self, _batch_id):
            return {
                "id": "batch-a", "status": "failed",
                "request_counts": {"total": 0, "completed": 0, "failed": 0},
                "errors": {"data": [{
                    "code": "model_not_found", "param": "body.model",
                    "message": "snapshot is not supported by the Batch API", "line": 1,
                }]},
            }

    _refresh_and_collect(FakeClient(), tmp_path, config, paths)
    jobs = [json.loads(line) for line in paths["jobs"].read_text().splitlines()]
    assert jobs[0]["status"] == "batch_validation_failed"
    assert _prepare_retries(jobs, config, [], [], {}, {}) == jobs

    batches = [json.loads(line) for line in paths["batches"].read_text().splitlines()]
    assert batches[0]["remote"]["errors"]["data"][0]["code"] == "model_not_found"
    assert _batch_failure_summaries(batches) == [{
        "batch_id": "batch-a",
        "status": "failed",
        "errors": [{
            "code": "model_not_found",
            "param": "body.model",
            "message": "snapshot is not supported by the Batch API",
            "count": 1,
        }],
    }]


def test_batch_capacity_rejection_returns_jobs_to_first_submission_queue(tmp_path, config):
    paths = _paths(tmp_path)
    paths["state"].mkdir(parents=True)
    job = {
        "custom_id": "job-a", "scope": "full", "kind": "sample", "sample_id": "x",
        "status": "submitted", "attempt": 1, "batch_id": "batch-a", "body": {},
        "logical_refs": [],
    }
    atomic_write_jsonl(paths["jobs"], [job])
    atomic_write_jsonl(paths["batches"], [{
        "batch_id": "batch-a", "scope": "full", "status": "validating",
        "custom_ids": ["job-a"],
    }])

    class FakeClient:
        def retrieve(self, _batch_id):
            return {
                "id": "batch-a", "status": "failed",
                "request_counts": {"total": 0, "completed": 0, "failed": 0},
                "errors": {"data": [{
                    "code": "token_limit_exceeded", "param": None,
                    "message": "Enqueued token limit reached for gpt-image-2.",
                }]},
            }

    _refresh_and_collect(FakeClient(), tmp_path, config, paths)

    requeued = read_jsonl(paths["jobs"])[0]
    assert requeued["status"] == "planned"
    assert requeued["attempt"] == 1
    assert requeued["capacity_requeues"] == 1
    assert "batch_id" not in requeued
    assert read_jsonl(paths["attempts"]) == []
    batches = read_jsonl(paths["batches"])
    assert batches[0]["capacity_requeued"] is True
    assert _batch_failure_summaries(batches) == []


def test_cost_ceiling_stops_before_next_batch():
    enforce_cost_ceiling(80.0, 10.0, 9.99, 100.0)
    with pytest.raises(PipelineError, match="cost stop"):
        enforce_cost_ceiling(80.0, 10.0, 10.01, 100.0)


def test_automatic_api_retries_are_disabled(config):
    assert config["batch"]["max_attempts"] == 1
    assert config["batch"]["retry_reserve_fraction"] == 0.0
    failed = [{
        "custom_id": "rotation-pilot-x-a1",
        "scope": "rotation_pilot",
        "kind": "sample",
        "sample_id": "x",
        "status": "qa_failed",
        "attempt": 1,
        "body": {"quality": "low"},
        "logical_refs": ["anchor", "plate"],
    }]
    assert _prepare_retries(failed, config, [], [], {}, {}, {}) == failed


def test_local_repair_rebuilds_from_raw_without_creating_api_attempts(
    tmp_path, config, monkeypatch
):
    import generate_synth_reid33 as generator

    paths = initialize(tmp_path, config)
    sample = read_jsonl(paths["rotation_pilot_samples"])[0]
    custom_id = f"rotation_pilot-{sample['sample_id']}-a1"
    refs = [f"anchor-p{sample['local_pid']:03d}-{sample['anchor_orientation']}",
            f"plate-c{sample['local_camera']:02d}"]
    job = {
        "custom_id": custom_id,
        "scope": "rotation_pilot",
        "kind": "sample",
        "sample_id": sample["sample_id"],
        "status": "qa_failed",
        "attempt": 1,
        "quality": "low",
        "logical_refs": refs,
        "body": {"prompt": "eight-yaw prompt", "quality": "low"},
    }
    attempt = {
        "custom_id": custom_id,
        "classification": "succeeded",
        "response_model": "gpt-image-2",
        "request_id": "request-local",
        "batch_id": "batch-local",
        "usage": {},
        "estimated_cost_usd": 0.01,
    }
    assets = [
        {"asset_id": ref, "sha256": f"sha-{index}", "file_id": f"file-{index}"}
        for index, ref in enumerate(refs)
    ]
    atomic_write_jsonl(paths["jobs"], [job])
    atomic_write_jsonl(paths["attempts"], [attempt])
    atomic_write_jsonl(paths["assets"], assets)
    raw = tmp_path / "raw" / "rotation_pilot" / f"{custom_id}.jpg"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"successful-api-raw")

    def fake_process(raw_path, final_path, _camera, _sample, _config):
        assert raw_path.read_bytes() == b"successful-api-raw"
        Image.new("RGB", (128, 256), (12, 34, 56)).save(final_path, format="JPEG")
        return {
            "accepted": True,
            "geometry_pass": True,
            "processing_version": generator.PROCESSING_VERSION,
            "person_detection": {"principal_score": 0.99},
            "pose_geometry": {"region_scores": {"head": 5.0}},
            "final_sha256": sha256_file(final_path),
        }

    monkeypatch.setattr(generator, "process_image", fake_process)
    dry = repair_local(tmp_path, config, "rotation-pilot", dry_run=True)
    assert dry["locally_repairable"] == 1
    assert not paths["rotation_pilot_manifest"].exists()
    assert raw.read_bytes() == b"successful-api-raw"

    report = repair_local(tmp_path, config, "rotation-pilot")
    manifest = read_jsonl(paths["rotation_pilot_manifest"])
    final = tmp_path / manifest[0]["final_path"]
    assert report["repaired"] == 1
    assert report["remaining_unresolved"] == 95
    rebuilt_bytes = final.read_bytes()
    assert Image.open(final).size == (128, 256)
    assert manifest[0]["selection"]["method"] == "local_raw_reprocess"
    assert manifest[0]["camera_geometry"]["version"] == "synth-reid33-camera-geometry-v1"
    assert len(read_jsonl(paths["attempts"])) == 1

    final.write_bytes(b"corrupt")
    repaired_again = repair_local(tmp_path, config, "rotation-pilot")
    assert repaired_again["repaired"] == 1
    assert final.read_bytes() == rebuilt_bytes
    failures = report_local_failures(tmp_path, config, "rotation-pilot", limit=10)
    assert failures["valid_final_images"] == 1
    assert failures["automatic_api_retries"] == 0
    assert failures["awaiting_first_attempt"] == 95
    assert failures["failed_without_successful_raw"] == 0


def test_approval_requires_every_pilot_gate(config):
    with pytest.raises(ValueError, match="pilot report"):
        approval_payload({"pilot_gate_passed": False}, "low", 100.0, config)


def test_explicit_approval_waiver_records_only_allowlisted_leaf_gates(config):
    report = {
        "pilot_gate_passed": False,
        "gates": {
            "decode": True,
            "geometry": True,
            "osnet_embedding": False,
            "body_rotation_pilot": False,
        },
    }
    rotation_report = {
        "rotation_gate_passed": False,
        "gates": {
            "geometry": True,
            "osnet_embedding": False,
            "manual_review": False,
        },
    }

    failed = _approval_failed_gates(report, rotation_report)

    assert failed == [
        "pilot.osnet_embedding",
        "rotation_pilot.manual_review",
        "rotation_pilot.osnet_embedding",
    ]
    assert set(failed) <= WAIVABLE_APPROVAL_GATES
    payload = approval_payload(
        report,
        "low",
        120.0,
        config,
        waived_gates=failed,
        waiver_reason="User explicitly authorized production despite these QA gates.",
    )
    assert payload["max_usd"] == 120.0
    assert payload["gate_waiver"]["waived_gates"] == failed
    assert payload["gate_waiver"]["user_authorized"] is True
    verify_approval(payload, config)


def test_approval_waiver_requires_reason(config):
    with pytest.raises(ValueError, match="waiver reason"):
        approval_payload(
            {"pilot_gate_passed": False},
            "low",
            120.0,
            config,
            waived_gates=["pilot.osnet_embedding"],
        )


def test_full_embedding_qa_treats_explicitly_waived_osnet_as_advisory(tmp_path):
    approval_path = tmp_path / "state" / "approval.json"
    approval_path.parent.mkdir(parents=True)
    approval_path.write_text(json.dumps({
        "gate_waiver": {
            "waived_gates": [
                "pilot.osnet_embedding",
                "rotation_pilot.osnet_embedding",
            ],
        },
    }), encoding="utf-8")

    assert _full_embedding_acceptance_models(tmp_path) == (("vit",), ("osnet",))


def test_full_embedding_qa_requires_osnet_without_both_explicit_waivers(tmp_path):
    approval_path = tmp_path / "state" / "approval.json"
    approval_path.parent.mkdir(parents=True)
    approval_path.write_text(json.dumps({
        "gate_waiver": {"waived_gates": ["pilot.osnet_embedding"]},
    }), encoding="utf-8")

    assert _full_embedding_acceptance_models(tmp_path) == (("vit", "osnet"), ())


def _save_image(path: Path, color: tuple[int, int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 256), color).save(path, quality=95)


def test_manifest_validator_detects_duplicate_sha(tmp_path):
    image_a = tmp_path / "accepted" / "train" / "a.jpg"
    image_b = tmp_path / "accepted" / "train" / "b.jpg"
    _save_image(image_a, (30, 40, 50))
    image_b.write_bytes(image_a.read_bytes())
    rows = []
    for index, path in enumerate((image_a, image_b)):
        rows.append({
            "sample_id": f"s{index}", "qa_status": "accepted", "split": "train",
            "local_pid": index, "local_camera": index, "final_path": str(path.relative_to(tmp_path)),
            "final_sha256": sha256_file(path),
        })
    atomic_write_jsonl(tmp_path / "manifest.jsonl", rows)
    report = validate_synthetic_manifest(
        tmp_path, {"train": 2, "query": 0, "gallery": 0, "total": 2}
    )
    assert not report["valid"]
    assert any("duplicate image SHA" in error for error in report["errors"])


def test_unified_validator_and_explicit_split(tmp_path):
    for split in ("train", "query", "gallery"):
        (tmp_path / split).mkdir()
    sequence = 0
    for camera, color in zip((33, 34, 35), ((1, 2, 3), (2, 3, 4), (3, 4, 5))):
        _save_image(tmp_path / "train" / f"p00000_d05_c{camera:03d}_{sequence:06d}.jpg", color)
        sequence += 1
    _save_image(tmp_path / "query" / f"p00001_d05_c033_{sequence:06d}.jpg", (4, 5, 6))
    sequence += 1
    for camera, color in zip((34, 35), ((5, 6, 7), (6, 7, 8))):
        _save_image(tmp_path / "gallery" / f"p00001_d05_c{camera:03d}_{sequence:06d}.jpg", color)
        sequence += 1
    common = validate_unified_output(
        tmp_path, expected_counts={"train": 3, "query": 1, "gallery": 2},
        expected_cameras=36, check_sha=True,
    )
    # This miniature deliberately lacks cameras 0..32, so the common validator
    # should flag global camera contiguity while the domain validator passes.
    assert not common["valid"]
    explicit = validate_explicit_domain(
        tmp_path, expected={"train": 3, "query": 1, "gallery": 2},
        expected_train_pids=1, expected_test_pids=1, expected_images_per_pid=3,
        expected_cameras_per_pid=3, expected_cross_camera_positives=2,
        expected_camera_count=3,
    )
    assert explicit["valid"], explicit["errors"]


def test_reid_loader_and_domain_sampler_see_66_cameras_and_d05(tmp_path):
    # Load the two modules without executing datasets/__init__.py, whose full
    # training stack is intentionally outside this focused loader smoke test.
    package_name = "personvit_test_datasets"
    package = types.ModuleType(package_name)
    package.__path__ = [str(REPO / "transreid_pytorch" / "datasets")]
    sys.modules[package_name] = package
    REID = importlib.import_module(f"{package_name}.reid").REID
    DomainBalancedIdentitySampler = importlib.import_module(
        f"{package_name}.sampler_domain"
    ).DomainBalancedIdentitySampler

    root = tmp_path / "data"
    for split in ("train", "query", "gallery"):
        (root / "reid" / split).mkdir(parents=True)
    for camera in range(66):
        domain = min(camera // 11, 5)
        _save_image(
            root / "reid" / "train" / f"p{camera:05d}_d{domain:02d}_c{camera:03d}_{camera:06d}.jpg",
            (camera, (camera * 3) % 255, (camera * 7) % 255),
        )
    _save_image(root / "reid" / "query" / "p00066_d05_c033_000066.jpg", (2, 8, 12))
    _save_image(root / "reid" / "gallery" / "p00066_d05_c034_000067.jpg", (3, 9, 13))

    dataset = REID(root=str(root), verbose=False)
    assert dataset.num_train_cams == 66
    assert dataset.num_train_vids == 6
    assert {pid for _, pid, _, _ in dataset.train} == set(range(66))
    sampler = DomainBalancedIdentitySampler(dataset.train, batch_size=12, num_instances=1, alpha=0.5)
    assert sampler.domains == [0, 1, 2, 3, 4, 5]
    assert sampler.domain_weights[5] > 0


def test_final_adoption_gate_requires_three_confirming_seeds():
    baselines = []
    candidates = []
    for seed in (1, 2, 3):
        baselines.append({
            "seed": seed,
            "recipe_sha256": "same-recipe",
            "real_domain_mAP": {f"d{domain:02d}": 0.80 for domain in range(5)},
            "occluded_mAP": {"Occluded-Duke": 0.60, "Occluded-REID": 0.60},
        })
        candidates.append({
            "seed": seed,
            "recipe_sha256": "same-recipe",
            "real_domain_mAP": {f"d{domain:02d}": 0.797 for domain in range(5)},
            "occluded_mAP": {"Occluded-Duke": 0.607, "Occluded-REID": 0.607},
        })
    provisional = evaluate_adoption(baselines[:1], candidates[:1])
    confirmed = evaluate_adoption(baselines, candidates)
    assert provisional["status"] == "provisional_pass_needs_3_seeds"
    assert confirmed["status"] == "adopt"
