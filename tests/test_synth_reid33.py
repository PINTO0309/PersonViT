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
    _batch_outstanding_requests,
    _full_embedding_acceptance_models,
    _live_batch_chunk_size,
    _local_sibling_raw_candidates,
    _merge_jobs,
    _paths,
    _refresh_and_collect,
    _prepare_retries,
    _repair_side_anchor_from_mirror,
    _render_local_sibling_raw,
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
    _back_view_head_visibility,
    _crop_resize_person,
    _filter_tiny_secondary_person_candidates,
    _is_back_facing_sample,
    _intentional_cart_person_fragment,
    _nms_person_candidates,
    _occluded_waist_visibility,
    _overlapped_ankle_visibility,
    _pose_bbox_expansion,
    _pose_detector_fallback,
    _select_pose_candidate,
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


def test_side_anchor_can_be_repaired_locally_from_opposite_view(tmp_path, config):
    source = tmp_path / "assets" / "anchor-p130-right.jpg"
    source.parent.mkdir(parents=True)
    image = Image.new("RGB", (1024, 1536), (20, 30, 40))
    image.paste((200, 40, 20), (0, 0, 512, 1536))
    image.save(source, quality=95)
    assets = {
        "anchor-p130-right": {
            "asset_id": "anchor-p130-right",
            "kind": "anchor",
            "local_pid": 130,
            "orientation": "right",
            "path": str(source.relative_to(tmp_path)),
            "sha256": sha256_file(source),
            "reference_sha256": "source-reference-sha",
            "response_model": "gpt-image-2",
        },
    }
    jobs = [{
        "custom_id": "asset-anchor-p130-left-a1",
        "scope": "asset",
        "kind": "anchor",
        "asset_id": "anchor-p130-left",
        "local_pid": 130,
        "orientation": "left",
        "status": "needs_revision",
    }]

    class FakeClient:
        def upload(self, path, purpose):
            assert Image.open(path).size == (256, 384)
            assert purpose == "vision"
            return "file-local-mirror"

    assert _repair_side_anchor_from_mirror(FakeClient(), tmp_path, config, jobs, assets) == 1
    repaired = assets["anchor-p130-left"]
    assert jobs[0]["status"] == "succeeded"
    assert jobs[0]["completion_mode"] == "local_horizontal_mirror"
    assert repaired["source_asset_id"] == "anchor-p130-right"
    assert repaired["file_id"] == "file-local-mirror"
    repaired_image = Image.open(tmp_path / repaired["path"])
    assert repaired_image.getpixel((100, 768))[0] < repaired_image.getpixel((900, 768))[0]


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


def test_person_nms_suppresses_low_confidence_lower_fragment_from_trolley_regression():
    boxes = [
        [140.9421, 174.2554, 301.4109, 798.6304],
        [96.3614, 464.3799, 334.4975, 985.7607],
    ]
    scores = [0.9245, 0.3974]

    # The weak box begins at the principal person's lower body and covers the
    # trolley plus legs.  It overlaps 53.5% of the smaller box but is neither a
    # conventional IoU duplicate nor fully contained.
    assert _nms_person_candidates(boxes, scores) == [0]


def test_person_nms_suppresses_split_body_trolley_regression():
    boxes = [
        [271.6458, 254.1577, 440.5086, 587.6014],
        [236.8543, 327.8797, 452.9579, 875.6179],
    ]
    scores = [0.8435, 0.4703]

    # The two boxes have IoU 0.335 and 77.9% smaller-box overlap.  The weak
    # lower box covers the person's legs plus the foreground luggage cart.
    assert _nms_person_candidates(boxes, scores) == [0]


@pytest.mark.parametrize(
    ("secondary_box", "secondary_score"),
    [
        ([320, 464, 500, 986], 0.3974),  # spatially independent person
        ([250, 190, 490, 986], 0.3974),  # overlapping full-height second person
        ([96, 464, 334, 986], 0.60),  # confident overlapping second person
        ([237, 328, 453, 876], 0.60),  # confident split-body-shaped candidate
        ([220, 350, 470, 900], 0.47),  # insufficient smaller-box overlap
    ],
)
def test_person_nms_keeps_candidates_outside_lower_fragment_rule(
    secondary_box, secondary_score
):
    boxes = [[141, 174, 301, 799], secondary_box]
    scores = [0.9245, secondary_score]

    assert _nms_person_candidates(boxes, scores) == [0, 1]


def test_intentional_cart_fragment_accepts_real_occlusion_regression():
    detection = {
        "count": 2,
        "candidate_boxes_xyxy": [
            [151.0048, 181.2913, 432.4798, 535.9988],
            [104.7411, 268.3246, 434.3389, 1050.5164],
        ],
        "candidate_scores": [0.7035, 0.5105],
    }
    pose = {"pass": True, "pose_candidate_count": 1}
    sample = {
        "occluded": True,
        "occluder": "plain luggage cart",
        "target_occlusion_ratio": 0.4492,
    }

    result = _intentional_cart_person_fragment(detection, pose, sample)

    assert result["pass"] is True
    assert result["candidate_extends_below"] is True
    assert result["smaller_box_overlap"] >= 0.70


def test_intentional_cart_fragment_accepts_high_occlusion_real_regression():
    detection = {
        "count": 2,
        "candidate_boxes_xyxy": [
            [206.7350, 192.8459, 423.8028, 548.0842],
            [170.7117, 305.1538, 432.2348, 1036.1371],
        ],
        "candidate_scores": [0.8413, 0.3670],
    }
    pose = {"pass": True, "pose_candidate_count": 1, "pose_score": 0.9986659}
    sample = {
        "occluded": True,
        "occluder": "plain luggage cart",
        "target_occlusion_ratio": 0.4666,
    }

    result = _intentional_cart_person_fragment(detection, pose, sample)

    assert result["pass"] is True
    assert result["smaller_box_overlap"] < 0.70
    assert result["high_occlusion_pose_support"] is True


@pytest.mark.parametrize(
    ("ratio", "pose_score"),
    [(0.4499, 0.999), (0.4666, 0.9899)],
)
def test_intentional_cart_fragment_rejects_weak_high_occlusion_support(
    ratio, pose_score
):
    detection = {
        "count": 2,
        "candidate_boxes_xyxy": [
            [206.7350, 192.8459, 423.8028, 548.0842],
            [170.7117, 305.1538, 432.2348, 1036.1371],
        ],
        "candidate_scores": [0.8413, 0.3670],
    }
    pose = {"pass": True, "pose_candidate_count": 1, "pose_score": pose_score}
    sample = {
        "occluded": True,
        "occluder": "plain luggage cart",
        "target_occlusion_ratio": ratio,
    }

    assert _intentional_cart_person_fragment(detection, pose, sample)["pass"] is False


def test_tiny_secondary_person_filter_suppresses_real_edge_artifact():
    boxes = [
        [66.329, 330.612, 205.135, 741.614],
        [543.973, 173.618, 556.363, 237.258],
    ]
    scores = [0.9929, 0.3735]

    kept, suppressed = _filter_tiny_secondary_person_candidates(
        boxes, scores, (576, 1152)
    )

    assert kept == [0]
    assert suppressed[0]["index"] == 1


@pytest.mark.parametrize(
    ("secondary_box", "secondary_score"),
    [
        ([500, 170, 530, 240], 0.37),  # wide enough to remain relevant
        ([544, 174, 556, 300], 0.37),  # tall enough to remain relevant
        ([544, 174, 556, 237], 0.80),  # strongly supported tiny person
    ],
)
def test_tiny_secondary_person_filter_keeps_non_artifacts(
    secondary_box, secondary_score
):
    kept, suppressed = _filter_tiny_secondary_person_candidates(
        [[66, 331, 205, 742], secondary_box],
        [0.99, secondary_score],
        (576, 1152),
    )

    assert kept == [0, 1]
    assert suppressed == []


@pytest.mark.parametrize(
    ("sample_update", "pose_update", "score", "reason"),
    [
        ({"occluded": False}, {}, 0.5105, "not occluded"),
        ({"occluder": "plain railing"}, {}, 0.5105, "not a cart"),
        ({}, {"pose_candidate_count": 2}, 0.5105, "ambiguous pose"),
        ({}, {}, 0.61, "confident second person"),
    ],
)
def test_intentional_cart_fragment_rejects_unsafe_cases(
    sample_update, pose_update, score, reason
):
    detection = {
        "count": 2,
        "candidate_boxes_xyxy": [
            [151, 181, 432, 536],
            [105, 268, 434, 1051],
        ],
        "candidate_scores": [0.7035, score],
    }
    pose = {"pass": True, "pose_candidate_count": 1, **pose_update}
    sample = {
        "occluded": True,
        "occluder": "plain luggage cart",
        "target_occlusion_ratio": 0.4492,
        **sample_update,
    }

    assert _intentional_cart_person_fragment(detection, pose, sample)["pass"] is False, reason


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


def test_overlapped_ankle_fallback_accepts_real_crossed_leg_regression():
    result = _overlapped_ankle_visibility(
        [-4.812676, 11.165815],
        [[367.642, 822.249], [395.281, 853.320]],
        (287, 276, 210, 652),
        (576, 1152),
    )

    assert result["pass"] is True
    assert result["ankles_overlap"] is True
    assert result["strong_ankle_visible"] is True


@pytest.mark.parametrize(
    ("scores", "points", "bbox"),
    [
        ([-1.299072, 9.422829], [[268.826, 732.555], [254.161, 812.843]], (199, 329, 145, 517)),
        ([-1.522069, 8.900734], [[215.769, 697.131], [246.866, 776.592]], (160, 313, 139, 519)),
    ],
)
def test_overlapped_ankle_fallback_accepts_moderately_weak_separated_feet(
    scores, points, bbox
):
    result = _overlapped_ankle_visibility(scores, points, bbox, (576, 1152))

    assert result["pass"] is True
    assert result["moderate_weak_ankle"] is True
    assert result["ankles_overlap"] is False


def test_overlapped_ankle_fallback_accepts_longitudinal_self_occlusion():
    result = _overlapped_ankle_visibility(
        [-2.554703, 9.315398],
        [[146.475, 757.395], [134.369, 861.934]],
        (62, 438, 165, 456),
        (576, 1152),
        knee_scores=[5.679, 9.347],
        knee_points=[[143.0, 748.8], [148.2, 764.3]],
        walking_pose=True,
    )

    assert result["pass"] is True
    assert result["longitudinal_leg_overlap"] is True
    assert result["weak_ankle_near_knee"] is True
    assert result["strong_ankle_below_weak"] is True


def test_overlapped_ankle_fallback_accepts_deep_high_confidence_longitudinal_leg():
    result = _overlapped_ankle_visibility(
        [11.4089, -4.1573],
        [[385.607, 859.738], [415.875, 746.587]],
        (303, 360, 197, 575),
        (576, 1152),
        knee_scores=[14.2049, 6.3966],
        knee_points=[[390, 720], [408, 750]],
        walking_pose=True,
        pose_score=0.999877,
    )

    assert result["pass"] is True
    assert result["longitudinal_leg_overlap"] is True
    assert result["deep_longitudinal_pose_support"] is True


def test_deep_longitudinal_leg_requires_high_pose_confidence():
    result = _overlapped_ankle_visibility(
        [11.4089, -4.1573],
        [[385.607, 859.738], [415.875, 746.587]],
        (303, 360, 197, 575),
        (576, 1152),
        knee_scores=[14.2049, 6.3966],
        knee_points=[[390, 720], [408, 750]],
        walking_pose=True,
        pose_score=0.98,
    )

    assert result["pass"] is False
    assert result["deep_longitudinal_pose_support"] is False


def test_high_confidence_longitudinal_leg_allows_ten_percent_knee_proximity():
    result = _overlapped_ankle_visibility(
        [-2.2490, 8.8488],
        [[152.881, 754.011], [145.976, 829.107]],
        (70, 340, 158, 512),
        (576, 1152),
        knee_scores=[6.9438, 9.8717],
        knee_points=[[142.523, 706.537], [161.511, 709.126]],
        walking_pose=True,
        pose_score=0.999736,
    )

    assert result["pass"] is True
    assert result["weak_ankle_near_knee"] is False
    assert result["high_confidence_weak_ankle_near_knee"] is True


def test_high_confidence_longitudinal_leg_allows_real_knee_support_regression():
    result = _overlapped_ankle_visibility(
        [11.7202, -2.3827],
        [[318.318, 823.419], [351.976, 705.115]],
        (215, 213, 234, 676),
        (576, 1152),
        knee_scores=[14.3765, 4.9308],
        knee_points=[[321.0, 730.0], [352.0, 727.7]],
        walking_pose=True,
        pose_score=0.999881,
    )

    assert result["pass"] is True
    assert result["knee_support_valid"] is False
    assert result["high_confidence_knee_support_valid"] is True


@pytest.mark.parametrize(
    ("pose_score", "weak_knee_score"),
    [(0.9899, 4.9308), (0.9999, 4.8999)],
)
def test_relaxed_knee_support_requires_high_confidence_and_bounded_knee(
    pose_score, weak_knee_score
):
    result = _overlapped_ankle_visibility(
        [11.7202, -2.3827],
        [[318.318, 823.419], [351.976, 705.115]],
        (215, 213, 234, 676),
        (576, 1152),
        knee_scores=[14.3765, weak_knee_score],
        knee_points=[[321.0, 730.0], [352.0, 727.7]],
        walking_pose=True,
        pose_score=pose_score,
    )

    assert result["pass"] is False
    assert result["high_confidence_knee_support_valid"] is False


@pytest.mark.parametrize("pose_score", [None, 0.989])
def test_extended_knee_proximity_requires_high_pose_confidence(pose_score):
    result = _overlapped_ankle_visibility(
        [-2.2490, 8.8488],
        [[152.881, 754.011], [145.976, 829.107]],
        (70, 340, 158, 512),
        (576, 1152),
        knee_scores=[6.9438, 9.8717],
        knee_points=[[142.523, 706.537], [161.511, 709.126]],
        walking_pose=True,
        pose_score=pose_score,
    )

    assert result["pass"] is False
    assert result["high_confidence_weak_ankle_near_knee"] is False


@pytest.mark.parametrize(
    ("ankle_scores", "knee_scores", "knee_points", "walking_pose", "reason"),
    [
        ([-3.01, 9.3], [5.7, 9.3], [[143, 749], [148, 764]], True, "too_weak"),
        ([-2.55, 9.3], [4.9, 9.3], [[143, 749], [148, 764]], True, "weak_knee"),
        ([-2.55, 9.3], [5.7, 9.3], [[100, 600], [110, 620]], True, "far_knee"),
        ([-2.55, 9.3], [5.7, 9.3], [[143, 749], [148, 764]], False, "not_walking"),
    ],
)
def test_longitudinal_self_occlusion_rejects_unsafe_cases(
    ankle_scores, knee_scores, knee_points, walking_pose, reason
):
    result = _overlapped_ankle_visibility(
        ankle_scores,
        [[146.475, 757.395], [134.369, 861.934]],
        (62, 438, 165, 456),
        (576, 1152),
        knee_scores=knee_scores,
        knee_points=knee_points,
        walking_pose=walking_pose,
    )

    assert result["pass"] is False, reason


def test_pose_bbox_expansion_accepts_complete_small_person_regression():
    result = _pose_bbox_expansion(
        (269, 374, 77, 243),
        [210.82, 364.07, 377.62, 758.65],
        0.9998467,
        {"head": 14.649, "shoulders": 7.069, "feet": 5.612},
        (576, 1152),
        occluded=False,
    )

    assert result["pass"] is True
    assert result["principal_containment"] >= 0.85
    assert result["match_iou"] >= 0.25
    assert result["expanded_bbox"] == [210, 364, 168, 395]


def test_pose_bbox_expansion_accepts_real_cart_occlusion_regression():
    result = _pose_bbox_expansion(
        (272, 203, 99, 236),
        [253.992, 176.827, 427.828, 638.420],
        0.99982196,
        {"head": 5.0, "shoulders": 5.0, "waist": 5.0},
        (576, 1152),
        occluded=True,
        occluder="plain luggage cart",
        target_occlusion_ratio=0.2672,
    )

    assert result["pass"] is True
    assert result["cart_occlusion"] is True
    assert result["expanded_bbox"] == [253, 176, 175, 463]


@pytest.mark.parametrize(
    ("occluder", "ratio"),
    [
        ("railing", 0.2672),
        ("plain luggage cart", 0.19),
        ("plain luggage cart", 0.51),
    ],
)
def test_pose_bbox_expansion_rejects_nonqualifying_occluded_expansion(
    occluder, ratio
):
    result = _pose_bbox_expansion(
        (272, 203, 99, 236),
        [253.992, 176.827, 427.828, 638.420],
        0.99982196,
        {"head": 5.0, "shoulders": 5.0, "waist": 5.0},
        (576, 1152),
        occluded=True,
        occluder=occluder,
        target_occlusion_ratio=ratio,
    )

    assert result["pass"] is False


@pytest.mark.parametrize(
    ("pose_score", "regions", "pose_box", "occluded"),
    [
        (0.98, {"head": 5, "shoulders": 5, "feet": 5}, [210, 364, 378, 759], False),
        (0.999, {"head": 5, "shoulders": -0.1, "feet": 5}, [210, 364, 378, 759], False),
        (0.999, {"head": 5, "shoulders": 5, "feet": 5}, [100, 200, 500, 1000], False),
        (0.999, {"head": 5, "shoulders": 5, "waist": 5}, [210, 364, 378, 759], True),
    ],
)
def test_pose_bbox_expansion_rejects_unsafe_cases(
    pose_score, regions, pose_box, occluded
):
    result = _pose_bbox_expansion(
        (269, 374, 77, 243),
        pose_box,
        pose_score,
        regions,
        (576, 1152),
        occluded,
    )

    assert result["pass"] is False


def test_pose_detector_fallback_accepts_single_complete_real_regression():
    scores = [0.0] * 17
    scores[0] = 5.371
    scores[5] = 8.680
    scores[6] = 9.0
    scores[11] = 7.881
    scores[12] = 8.0
    result = _pose_detector_fallback(
        [[249.285, 252.820, 348.362, 503.539]],
        [0.999076],
        [scores],
        occluded=True,
        image_size=(576, 1152),
    )

    assert result["pass"] is True
    assert result["bbox"] == [249, 252, 100, 252]
    assert result["required_regions_visible"] is True


@pytest.mark.parametrize(
    ("boxes", "pose_scores", "score_update", "reason"),
    [
        ([], [], {}, "no candidate"),
        (
            [[249, 253, 348, 504], [50, 100, 200, 500]],
            [0.999, 0.998],
            {},
            "ambiguous candidates",
        ),
        ([[249, 253, 348, 504]], [0.989], {}, "pose score too low"),
        ([[249, 253, 348, 504]], [0.999], {11: -0.1}, "waist missing"),
        ([[249, 253, 260, 504]], [0.999], {}, "box too narrow"),
        ([[249, 1, 348, 504]], [0.999], {}, "box at image edge"),
    ],
)
def test_pose_detector_fallback_rejects_unsafe_cases(
    boxes, pose_scores, score_update, reason
):
    base_scores = [0.0] * 17
    for index, value in {0: 5.4, 5: 8.7, 6: 9.0, 11: 7.9, 12: 8.0}.items():
        base_scores[index] = value
    for index, value in score_update.items():
        base_scores[index] = value
    rows = [list(base_scores) for _ in boxes]

    result = _pose_detector_fallback(
        boxes, pose_scores, rows, occluded=True, image_size=(576, 1152)
    )

    assert result["pass"] is False, reason


def test_split_pose_candidate_fallback_selects_complete_occluded_upper_body():
    matches = [(0.521, 0.999, 0), (0.643, 0.959, 1)]
    score_rows = {
        0: [15.271, 0, 0, 0, 0, 8.993, 9.161, 0, 0, 0, 0, 8.946, 9.863, 0, 0, -3.404, -3.416],
        1: [-3.440, 0, 0, 0, 0, -7.684, -6.311, 0, 0, 0, 0, 9.009, 7.964, 0, 0, 6.697, 2.694],
    }

    selected, fallback = _select_pose_candidate(matches, score_rows, occluded=True)

    assert selected == matches[0]
    assert fallback is not None
    assert fallback["pass"] is True
    assert fallback["initial_candidate_index"] == 1
    assert fallback["selected_candidate_index"] == 0


def test_split_pose_candidate_fallback_does_not_merge_incomplete_candidates():
    matches = [(0.51, 0.99, 0), (0.64, 0.96, 1)]
    score_rows = {
        0: [8, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, -2, 8, 0, 0, 0, 0],
        1: [-2, 0, 0, 0, 0, -2, -2, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0],
    }

    selected, fallback = _select_pose_candidate(matches, score_rows, occluded=True)

    assert selected == matches[1]
    assert fallback is None


def test_complete_pose_candidate_fallback_selects_visible_non_occluded_feet():
    matches = [(0.63998, 0.99789, 0), (0.68294, 0.86319, 1)]
    score_rows = {
        0: [14.85, 0, 0, 0, 0, 7.49, 8.0, 0, 0, 0, 0, 8, 8, 10.12, 6.93, 6.62, 4.02],
        1: [13.44, 0, 0, 0, 0, 6.94, 8.0, 0, 0, 0, 0, 8, 8, 3.70, 5.19, -5.08, -5.33],
    }

    selected, fallback = _select_pose_candidate(matches, score_rows, occluded=False)

    assert selected == matches[0]
    assert fallback is not None
    assert fallback["initial_candidate_index"] == 1
    assert fallback["selected_candidate_index"] == 0
    assert fallback["selected_region_scores"]["feet"] == pytest.approx(4.02)


def test_back_view_head_fallback_accepts_weak_face_keypoint_only_in_safe_head_region():
    accepted = _back_view_head_visibility(
        -0.75,
        (400.0, 360.0),
        (342, 322, 126, 311),
        (576, 1152),
    )
    misplaced = _back_view_head_visibility(
        -0.75,
        (400.0, 600.0),
        (342, 322, 126, 311),
        (576, 1152),
    )
    too_weak = _back_view_head_visibility(
        -1.01,
        (400.0, 360.0),
        (342, 322, 126, 311),
        (576, 1152),
    )

    assert accepted["pass"] is True
    assert accepted["head_location_valid"] is True
    assert misplaced["pass"] is False
    assert too_weak["pass"] is False


def test_back_view_head_fallback_accepts_deep_score_with_strong_shoulder_support():
    accepted = _back_view_head_visibility(
        -1.216256,
        (352.402, 304.957),
        (316, 278, 144, 475),
        (576, 1152),
        supporting_shoulder_score=8.087672,
    )
    weak_shoulders = _back_view_head_visibility(
        -1.216256,
        (352.402, 304.957),
        (316, 278, 144, 475),
        (576, 1152),
        supporting_shoulder_score=4.99,
    )
    misplaced = _back_view_head_visibility(
        -1.216256,
        (352.402, 400.0),
        (316, 278, 144, 475),
        (576, 1152),
        supporting_shoulder_score=8.087672,
    )

    assert accepted["pass"] is True
    assert accepted["deep_supported_head"] is True
    assert weak_shoulders["pass"] is False
    assert misplaced["pass"] is False


def test_occluded_waist_fallback_accepts_real_low_wall_regression():
    result = _occluded_waist_visibility(
        [-0.777958, 0.369189],
        [[312.984, 492.299], [311.258, 492.299]],
        (303, 298, 125, 203),
        (576, 1152),
        0.498,
    )

    assert result["pass"] is True
    assert result["strong_hip_visible"] is True
    assert result["weak_hip_bounded"] is True
    assert result["hip_locations_valid"] == [True, True]
    assert result["person_not_bottom_clipped"] is True
    assert result["requested_occlusion_valid"] is True


@pytest.mark.parametrize(
    ("scores", "points", "bbox", "ratio", "expected_reason"),
    [
        ([-1.01, 5.0], [[150, 700], [250, 710]], (100, 100, 200, 700), 0.4, "weak"),
        ([-0.5, -0.2], [[150, 700], [250, 710]], (100, 100, 200, 700), 0.4, "no_strong"),
        ([-0.5, 5.0], [[150, 300], [250, 710]], (100, 100, 200, 700), 0.4, "not_lower"),
        ([-0.5, 5.0], [[150, 1050], [250, 1060]], (100, 400, 200, 750), 0.4, "clipped"),
        ([-0.5, 5.0], [[150, 700], [250, 710]], (100, 100, 200, 700), 0.0, "not_requested"),
    ],
)
def test_occluded_waist_fallback_rejects_unsafe_cases(
    scores, points, bbox, ratio, expected_reason
):
    result = _occluded_waist_visibility(scores, points, bbox, (576, 1152), ratio)

    assert result["pass"] is False, expected_reason


def test_occluded_waist_fallback_accepts_high_occlusion_wall_boundary():
    result = _occluded_waist_visibility(
        [-1.79399, -0.92716],
        [[406.396, 498.784], [350.234, 498.784]],
        (308, 198, 161, 322),
        (576, 1152),
        0.4918,
        occluder="low wall",
        head_score=12.6249,
        shoulder_score=8.0469,
        pose_score=0.9973,
    )

    assert result["pass"] is True
    assert result["ordinary_waist"] is False
    assert result["wall_boundary_waist"] is True
    assert result["boundary_locations_valid"] is True


def test_occluded_waist_fallback_accepts_high_occlusion_bollard_boundary():
    result = _occluded_waist_visibility(
        [-1.2122, -1.8092],
        [[327.932, 413.287], [321.014, 413.287]],
        (261, 217, 97, 200),
        (576, 1152),
        0.4824,
        occluder="bollard",
        head_score=12.1756,
        shoulder_score=8.0509,
        pose_score=0.97465,
    )

    assert result["pass"] is True
    assert result["ordinary_waist"] is False
    assert result["bollard_boundary_waist"] is True
    assert result["bollard_hips_aligned"] is True


@pytest.mark.parametrize(
    ("points", "pose_score", "reason"),
    [
        ([[328, 413], [300, 413]], 0.975, "hips not aligned"),
        ([[328, 413], [321, 413]], 0.949, "weak pose"),
    ],
)
def test_bollard_boundary_waist_rejects_unsafe_cases(points, pose_score, reason):
    result = _occluded_waist_visibility(
        [-1.21, -1.81],
        points,
        (261, 217, 97, 200),
        (576, 1152),
        0.4824,
        occluder="bollard",
        head_score=12.2,
        shoulder_score=8.1,
        pose_score=pose_score,
    )

    assert result["pass"] is False, reason


def test_occluded_waist_fallback_accepts_high_occlusion_railing_boundary():
    result = _occluded_waist_visibility(
        [-2.2071, -0.2478],
        [[279.926, 491.665], [211.889, 491.665]],
        (185, 274, 125, 217),
        (576, 1152),
        0.4631,
        occluder="railing",
        head_score=12.7017,
        shoulder_score=7.7005,
        pose_score=0.9923,
    )

    assert result["pass"] is True
    assert result["ordinary_waist"] is False
    assert result["railing_boundary_waist"] is True
    assert result["railing_hips_bounded"] is True


@pytest.mark.parametrize(
    ("scores", "pose_score", "reason"),
    [
        ([-2.51, -0.24], 0.992, "occluded hip too weak"),
        ([-2.20, -0.51], 0.992, "supporting hip too weak"),
        ([-2.20, -0.24], 0.989, "pose too weak"),
    ],
)
def test_railing_boundary_waist_rejects_unsafe_cases(scores, pose_score, reason):
    result = _occluded_waist_visibility(
        scores,
        [[280, 492], [212, 492]],
        (185, 274, 125, 217),
        (576, 1152),
        0.4631,
        occluder="railing",
        head_score=12.7,
        shoulder_score=7.7,
        pose_score=pose_score,
    )

    assert result["pass"] is False, reason


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"occluder": "plain railing"}, "not a wall"),
        ({"target_occlusion_ratio": 0.39}, "not high occlusion"),
        ({"scores": [-2.01, -0.9]}, "hip too weak"),
        ({"shoulder_score": 4.9}, "weak upper-body support"),
        ({"pose_score": 0.98}, "weak pose"),
        ({"points": [[406, 400], [350, 400]]}, "hips not at boundary"),
    ],
)
def test_wall_boundary_waist_rejects_unsafe_cases(kwargs, reason):
    values = {
        "scores": [-1.79, -0.93],
        "points": [[406, 499], [350, 499]],
        "target_occlusion_ratio": 0.4918,
        "occluder": "low wall",
        "head_score": 12.6,
        "shoulder_score": 8.0,
        "pose_score": 0.997,
        **kwargs,
    }
    result = _occluded_waist_visibility(
        values["scores"],
        values["points"],
        (308, 198, 161, 322),
        (576, 1152),
        values["target_occlusion_ratio"],
        occluder=values["occluder"],
        head_score=values["head_score"],
        shoulder_score=values["shoulder_score"],
        pose_score=values["pose_score"],
    )

    assert result["pass"] is False, reason


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ({"body_yaw_deg": 135}, True),
        ({"body_yaw_deg": -135}, True),
        ({"body_yaw_deg": 90}, False),
        ({"anchor_orientation": "back"}, True),
        ({"orientation": "front"}, False),
    ],
)
def test_back_facing_sample_detection_is_restricted_to_rear_yaws(sample, expected):
    assert _is_back_facing_sample(sample) is expected


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


@pytest.mark.parametrize(
    ("batch", "expected"),
    [
        ({"status": "completed", "custom_ids": list(range(400))}, 0),
        ({"status": "validating", "custom_ids": list(range(400))}, 400),
        ({
            "status": "in_progress",
            "custom_ids": list(range(400)),
            "remote": {"request_counts": {"total": 400, "completed": 219, "failed": 1}},
        }, 400),
        ({
            "status": "finalizing",
            "custom_ids": list(range(88)),
            "remote": {"request_counts": {"total": 88, "completed": 87, "failed": 0}},
        }, 88),
    ],
)
def test_batch_outstanding_request_count_tracks_active_quota_reservation(batch, expected):
    assert _batch_outstanding_requests(batch) == expected


def test_full_live_batches_use_small_chunks_without_changing_pilot_batches(config):
    assert config["batch"]["requests_per_sample_batch"] == 400
    assert _live_batch_chunk_size(config, "full") == 100
    assert _live_batch_chunk_size(config, "pilot") == 400
    assert _live_batch_chunk_size(config, "rotation_pilot") == 400


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


def test_local_sibling_repair_selects_same_identity_camera_and_mirrors_walking_side(
    tmp_path,
):
    source_raw = tmp_path / "source.jpg"
    image = Image.new("RGB", (100, 200), (20, 30, 40))
    image.paste((220, 30, 20), (0, 0, 50, 200))
    image.save(source_raw, quality=100)
    target = {
        "sample_id": "target",
        "local_pid": 189,
        "local_camera": 15,
        "split": "train",
        "occluded": False,
        "body_yaw_deg": 0,
        "frame": 3,
        "pose": "walking right foot forward",
        "generation_seed": 123,
    }
    source = {
        **target,
        "sample_id": "source",
        "body_yaw_deg": -45,
        "frame": 2,
        "pose": "walking left foot forward",
    }
    job = {"custom_id": "full-source-a1"}
    attempt = {"classification": "succeeded"}
    candidates = _local_sibling_raw_candidates(
        "target",
        {"target": target, "source": source},
        {"source": [(job, attempt, source_raw)]},
    )

    assert candidates[0][0]["sample_id"] == "source"
    destination = tmp_path / "derived.jpg"
    transform = _render_local_sibling_raw(
        source_raw, destination, source, target
    )
    derived = Image.open(destination).convert("RGB")
    assert transform["mirrored"] is True
    assert derived.getpixel((10, 100))[0] < derived.getpixel((90, 100))[0]


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
