#!/usr/bin/env python3
"""Validate SyntheticReID33 alone and, optionally, its unified d05 projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_unified_dataset import (
    SYNTHETIC_EXPECTED,
    UNIFIED_WITH_SYNTH_EXPECTED,
    validate_explicit_domain,
    validate_unified_output,
)
from synth_reid33_core import validate_synthetic_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("synthetic_root", type=Path)
    parser.add_argument("--unified-root", type=Path)
    parser.add_argument("--skip-sha-validation", action="store_true")
    args = parser.parse_args()

    standalone = validate_synthetic_manifest(args.synthetic_root.resolve(), SYNTHETIC_EXPECTED)
    report = {
        "standalone": {key: value for key, value in standalone.items() if key != "rows"},
    }
    valid = standalone["valid"]
    if args.unified_root:
        unified = validate_unified_output(
            args.unified_root.resolve(),
            expected_counts=UNIFIED_WITH_SYNTH_EXPECTED,
            expected_cameras=66,
            check_sha=not args.skip_sha_validation,
        )
        d05 = validate_explicit_domain(args.unified_root.resolve(), domain=5, camera_start=33)
        report["unified"] = unified
        report["d05"] = d05
        valid = valid and unified["valid"] and d05["valid"]
    report["valid"] = valid
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
