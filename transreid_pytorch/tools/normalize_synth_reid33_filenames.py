#!/usr/bin/env python3
"""Normalize SyntheticReID33 release JPEGs to the unified-ReID filename schema."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from synth_reid33_core import (
    SYNTHETIC_RELEASE_FILENAME_VERSION,
    SYNTHETIC_RELEASE_PID_SCOPE,
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    sha256_file,
    synthetic_release_filename,
    synthetic_release_sequence,
    utc_now,
    validate_synthetic_manifest,
)


EXPECTED = {"train": 16_000, "query": 400, "gallery": 3_600, "total": 20_000}


def normalize_release_filenames(root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Publish canonical hardlinks, atomically switch manifest paths, then clean legacy names."""
    root = root.resolve()
    manifest_path = root / "manifest.jsonl"
    accepted_root = root / "accepted"
    rows = read_jsonl(manifest_path)
    if len(rows) != EXPECTED["total"]:
        raise ValueError(f"expected 20,000 manifest rows, found {len(rows)}")

    before = validate_synthetic_manifest(
        root,
        EXPECTED,
        require_release_filenames=False,
    )
    if not before["valid"]:
        detail = "\n  - ".join(before["errors"][:30])
        raise ValueError(f"pre-migration dataset is invalid:\n  - {detail}")

    plans: list[tuple[dict[str, Any], Path, Path]] = []
    destinations: set[Path] = set()
    legacy_paths: set[Path] = set()
    for row in rows:
        source = (root / str(row["final_path"])).resolve()
        destination = (
            accepted_root / str(row["split"]) / synthetic_release_filename(row)
        ).resolve()
        try:
            destination.relative_to(accepted_root)
        except ValueError as exc:
            raise ValueError(f"destination escapes accepted root: {destination}") from exc
        if destination in destinations:
            raise ValueError(f"duplicate destination filename: {destination}")
        destinations.add(destination)
        legacy_paths.add(
            (accepted_root / str(row["split"]) / f"{row['sample_id']}.jpg").resolve()
        )
        if not source.is_file() and not destination.is_file():
            raise FileNotFoundError(f"neither source nor canonical destination exists: {row['sample_id']}")
        plans.append((row, source, destination))

    current_jpegs = {path.resolve() for path in accepted_root.glob("*/*.jpg")}
    allowed = destinations | legacy_paths | {
        source for _, source, _ in plans if source.is_relative_to(accepted_root)
    }
    unexpected = current_jpegs - allowed
    if unexpected:
        raise ValueError(f"accepted contains unexpected JPEGs: {sorted(unexpected)[:10]}")

    report = {
        "dataset_name": "SyntheticReID33",
        "created_at": utc_now(),
        "dry_run": dry_run,
        "filename_version": SYNTHETIC_RELEASE_FILENAME_VERSION,
        "pid_scope": SYNTHETIC_RELEASE_PID_SCOPE,
        "pattern": "p{pid:05d}_d{domain:02d}_c{camera:03d}_{sequence:06d}.jpg",
        "images": len(plans),
        "renamed": sum(source != destination for _, source, destination in plans),
        "unchanged": sum(source == destination for _, source, destination in plans),
        "domain": 5,
        "global_camera_range": [33, 65],
        "integration_performed": False,
    }
    if dry_run:
        return report

    created: set[Path] = set()
    backup = root / ".manifest.filename-migration-backup.jsonl"
    if backup.exists():
        raise RuntimeError(f"stale manifest backup blocks migration: {backup}")
    manifest_switched = False
    committed = False
    try:
        for row, source, destination in plans:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if sha256_file(destination) != row.get("final_sha256"):
                    raise ValueError(f"canonical destination SHA mismatch: {destination}")
                continue
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
            created.add(destination)

        shutil.copy2(manifest_path, backup)
        for row, _, destination in plans:
            row["final_path"] = destination.relative_to(root).as_posix()
            row["release_filename_version"] = SYNTHETIC_RELEASE_FILENAME_VERSION
            row["release_pid_scope"] = SYNTHETIC_RELEASE_PID_SCOPE
            row["release_sequence"] = synthetic_release_sequence(row)
        atomic_write_jsonl(manifest_path, sorted(rows, key=lambda item: item["sample_id"]))
        manifest_switched = True

        after = validate_synthetic_manifest(root, EXPECTED)
        if not after["valid"]:
            detail = "\n  - ".join(after["errors"][:30])
            raise ValueError(f"post-migration dataset is invalid:\n  - {detail}")
        committed = True

        obsolete = {
            path
            for path in legacy_paths
            | {source for _, source, _ in plans if source.is_relative_to(accepted_root)}
            if path not in destinations
        }
        for path in obsolete:
            path.unlink(missing_ok=True)
        remaining = {path.resolve() for path in accepted_root.glob("*/*.jpg")}
        if remaining != destinations:
            raise RuntimeError(
                f"canonical publication count mismatch: {len(remaining)} vs {len(destinations)}"
            )
        backup.unlink(missing_ok=True)
        atomic_write_json(root / "state" / "filename_migration.json", report)
        return report
    except BaseException:
        if not committed and manifest_switched and backup.exists():
            os.replace(backup, manifest_path)
        if not committed:
            for path in created:
                path.unlink(missing_ok=True)
        raise
    finally:
        backup.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = normalize_release_filenames(args.root, dry_run=args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
