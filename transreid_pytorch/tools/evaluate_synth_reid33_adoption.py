#!/usr/bin/env python3
"""Apply the real-domain non-regression and occlusion gain adoption gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


REAL_DOMAINS = ("d00", "d01", "d02", "d03", "d04")
OCCLUDED_SETS = ("Occluded-Duke", "Occluded-REID")


def load_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"seed", "recipe_sha256", "real_domain_mAP", "occluded_mAP"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    return value


def evaluate_adoption(
    baselines: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    baseline_by_seed = {int(row["seed"]): row for row in baselines}
    candidate_by_seed = {int(row["seed"]): row for row in candidates}
    if len(baseline_by_seed) != len(baselines) or len(candidate_by_seed) != len(candidates):
        raise ValueError("duplicate seed in results")
    if set(baseline_by_seed) != set(candidate_by_seed):
        raise ValueError("baseline and candidate seed sets differ")
    per_seed = []
    for seed in sorted(baseline_by_seed):
        baseline = baseline_by_seed[seed]
        candidate = candidate_by_seed[seed]
        if baseline["recipe_sha256"] != candidate["recipe_sha256"]:
            raise ValueError(f"seed {seed}: training recipes differ")
        real_baseline = mean(float(baseline["real_domain_mAP"][key]) for key in REAL_DOMAINS)
        real_candidate = mean(float(candidate["real_domain_mAP"][key]) for key in REAL_DOMAINS)
        occ_baseline = mean(float(baseline["occluded_mAP"][key]) for key in OCCLUDED_SETS)
        occ_candidate = mean(float(candidate["occluded_mAP"][key]) for key in OCCLUDED_SETS)
        per_seed.append({
            "seed": seed,
            "real_mean_mAP_baseline": real_baseline,
            "real_mean_mAP_candidate": real_candidate,
            "real_mAP_change_points": (real_candidate - real_baseline) * 100.0,
            "occluded_mean_mAP_baseline": occ_baseline,
            "occluded_mean_mAP_candidate": occ_candidate,
            "occluded_mAP_change_points": (occ_candidate - occ_baseline) * 100.0,
        })
    real_change = mean(row["real_mAP_change_points"] for row in per_seed)
    occluded_change = mean(row["occluded_mAP_change_points"] for row in per_seed)
    metric_pass = real_change >= -0.5 and occluded_change >= 0.5
    three_seed_confirmation = len(per_seed) >= 3
    return {
        "seeds": len(per_seed),
        "per_seed": per_seed,
        "mean_real_mAP_change_points": real_change,
        "mean_occluded_mAP_change_points": occluded_change,
        "real_non_regression_pass": real_change >= -0.5,
        "occlusion_gain_pass": occluded_change >= 0.5,
        "three_seed_confirmation": three_seed_confirmation,
        "status": "adopt" if metric_pass and three_seed_confirmation else (
            "provisional_pass_needs_3_seeds" if metric_pass else "reject"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = evaluate_adoption(
            [load_result(path) for path in args.baseline],
            [load_result(path) for path in args.candidate],
        )
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["status"] != "reject" else 2


if __name__ == "__main__":
    raise SystemExit(main())
