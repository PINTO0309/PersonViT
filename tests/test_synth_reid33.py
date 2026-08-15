from __future__ import annotations

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
from generate_synth_reid33 import (  # noqa: E402
    PipelineError,
    _batch_failure_summaries,
    _merge_jobs,
    _paths,
    _refresh_and_collect,
    _prepare_retries,
    enforce_cost_ceiling,
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
    make_samples,
    reconcile_batch_output,
    sha256_file,
    validate_pilot,
    validate_plan,
    validate_synthetic_manifest,
)


CONFIG_PATH = REPO / "transreid_pytorch" / "configs" / "synth_reid33.yml"


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG_PATH)


def test_sample_generation_uses_cost_minimizing_portrait_size(config):
    assert config["model"]["api_id"] == "gpt-image-2"
    assert config["model"]["catalog_snapshot"] == "gpt-image-2-2026-04-21"
    assert config["model"]["sample_size"] == "576x1152"
    assert (config["dataset"]["final_width"], config["dataset"]["final_height"]) == (128, 256)


def test_deterministic_full_allocation_meets_acceptance(config):
    allocation, targets = allocate_identity_cameras(config)
    identities = make_identities(config)
    cameras = make_cameras(config)
    samples = make_samples(config, allocation)
    report = validate_plan(config, identities, cameras, samples)

    assert report["valid"], report["errors"]
    assert Counter(targets) == {120: 2, 121: 22, 122: 9}
    assert report["counts"] == {"train": 16000, "query": 400, "gallery": 3600}
    assert set(sample["global_camera"] for sample in samples) == set(range(33, 66))


def test_pilot_has_192_candidates_and_covers_every_camera(config):
    pilot = make_pilot_samples(config, allocate_pilot_cameras(config))
    report = validate_pilot(pilot)
    assert report["valid"], report["errors"]
    assert min(report["coverage_per_quality"].values()) >= 2


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


def test_cost_ceiling_stops_before_next_batch():
    enforce_cost_ceiling(80.0, 10.0, 9.99, 100.0)
    with pytest.raises(PipelineError, match="cost stop"):
        enforce_cost_ceiling(80.0, 10.0, 10.01, 100.0)


def test_approval_requires_every_pilot_gate(config):
    with pytest.raises(ValueError, match="pilot report"):
        approval_payload({"pilot_gate_passed": False}, "low", 100.0, config)


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
