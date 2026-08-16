#!/usr/bin/env python3
"""Durable supervisor for the approved SyntheticReID33 production run.

The supervisor never logs the API key. It removes the fixed tmpfs secret as soon
as all remote Batch work has reached a terminal state, then continues with local
repair, full QA, and validation. Unified-dataset construction remains available
unless ``--no-integrate`` is selected. Review waivers are recorded rather than
fabricating review results.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent / "data" / "SyntheticReID33"
SECRET_PATH = Path("/dev/shm/personvit_openai_api_key")
PENDING_STATUSES = frozenset({"planned", "blocked_on_refs", "submitted"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Supervisor:
    def __init__(self, root: Path, poll_seconds: int, no_integrate: bool = False) -> None:
        self.root = root.resolve()
        self.poll_seconds = poll_seconds
        self.no_integrate = no_integrate
        self.state_dir = self.root / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.state_dir / "production_supervisor.json"
        self.log_path = self.state_dir / "production_supervisor.log"
        self.repo_root = HERE.parents[1]
        self.generator = HERE / "generate_synth_reid33.py"
        self.validator = HERE / "validate_synth_reid33.py"
        self.builder = HERE / "build_unified_dataset.py"

    def status(self, stage: str, **values: Any) -> None:
        payload = {"updated_at": _utc_now(), "stage": stage, **values}
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.status_path)

    def log(self, message: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{_utc_now()}] {message.rstrip()}\n")

    def run(self, command: Sequence[str], *, api_key: str | None = None) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        if api_key is not None:
            environment["OPENAI_API_KEY"] = api_key
        result = subprocess.run(
            list(command),
            cwd=self.repo_root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        rendered = " ".join(command)
        self.log(f"command exit={result.returncode}: {rendered}")
        if result.stdout:
            self.log(result.stdout)
        if result.stderr:
            self.log(f"stderr: {result.stderr}")
        return result

    def jobs(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.state_dir / "jobs.jsonl")

    def manifest_count(self) -> int:
        return len(_read_jsonl(self.root / "manifest.jsonl"))

    def job_counts(self) -> dict[str, int]:
        return dict(Counter(
            row.get("status", "unknown")
            for row in self.jobs()
            if row.get("scope") in {"full", "asset"}
        ))

    def pending_count(self) -> int:
        return sum(
            row.get("status") in PENDING_STATUSES
            for row in self.jobs()
            if row.get("scope") in {"full", "asset"}
        )

    def delete_secret(self) -> None:
        if SECRET_PATH.exists():
            SECRET_PATH.unlink()
            self.log(f"deleted temporary API-key file: {SECRET_PATH}")

    def advance_remote_batches(self) -> bool:
        while self.pending_count() or not self.jobs():
            if not SECRET_PATH.is_file():
                self.status(
                    "blocked_missing_api_key",
                    job_status=self.job_counts(),
                    manifest_images=self.manifest_count(),
                )
                return False
            api_key = SECRET_PATH.read_text(encoding="utf-8").strip()
            if not api_key:
                self.status("blocked_empty_api_key", job_status=self.job_counts())
                return False
            result = self.run([
                sys.executable,
                str(self.generator),
                "--root",
                str(self.root),
                "full",
                "--resume",
            ], api_key=api_key)
            del api_key
            counts = self.job_counts()
            self.status(
                "generating",
                job_status=counts,
                manifest_images=self.manifest_count(),
                last_command_exit=result.returncode,
            )
            if result.returncode != 0:
                self.status(
                    "blocked_generation_command",
                    job_status=counts,
                    manifest_images=self.manifest_count(),
                    last_command_exit=result.returncode,
                )
                return False
            if self.pending_count():
                time.sleep(self.poll_seconds)
        return True

    def repair_if_needed(self) -> bool:
        if self.manifest_count() == 20_000:
            return True
        # repair-local evaluates every stale processing version in a temporary
        # directory before committing files and the manifest. Running a dry-run
        # first therefore duplicates the dominant full-dataset QA work without
        # adding an extra atomicity guarantee.
        result = self.run([
            sys.executable,
            str(self.generator),
            "--root",
            str(self.root),
            "repair-local",
            "--scope",
            "full",
        ])
        count = self.manifest_count()
        if result.returncode != 0 or count != 20_000:
            self.status(
                "blocked_local_repair",
                job_status=self.job_counts(),
                manifest_images=count,
                last_command_exit=result.returncode,
            )
            return False
        return True

    def run_local_qa(self) -> bool:
        report_path = self.root / "qa" / "full_report.json"
        if report_path.exists():
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            waiver = existing.get("explicit_user_waiver") or {}
            if (
                waiver.get("authorized")
                and waiver.get("authorized_by") == "user"
                and existing.get("release_ready_pending_manual_review")
                and int(existing.get("accepted", 0)) == 20_000
            ):
                self.log("full embedding/near-duplicate QA skipped under explicit user waiver")
                return True
        self.status("running_full_qa", manifest_images=self.manifest_count())
        result = self.run([
            sys.executable,
            str(self.generator),
            "--root",
            str(self.root),
            "qa",
            "--scope",
            "full",
        ])
        report: Mapping[str, Any] = {}
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        if result.returncode != 0 or not report.get("complete"):
            self.status(
                "blocked_full_qa",
                accepted=report.get("accepted"),
                rejected=report.get("rejected"),
                last_command_exit=result.returncode,
            )
            return False
        return True

    def validate_and_integrate(self) -> bool:
        validation = self.run([sys.executable, str(self.validator), str(self.root)])
        if validation.returncode != 0:
            combined = f"{validation.stdout}\n{validation.stderr}"
            if (
                "full manual review gate has not passed" in combined
                or "stratified 5% full_review.json is missing" in combined
            ):
                self.status(
                    "awaiting_manual_review",
                    review_template=str(self.root / "qa" / "full_review.template.json"),
                )
            else:
                self.status("blocked_validation", last_command_exit=validation.returncode)
            return False
        data_root = self.root.parent
        unified_root = data_root / "reid"
        build = self.run([
            sys.executable,
            str(self.builder),
            "--data-root",
            str(data_root),
            "--synthetic-root",
            str(self.root),
            "--force",
        ])
        if build.returncode != 0:
            self.status("blocked_unified_build", last_command_exit=build.returncode)
            return False
        final_validation = self.run([
            sys.executable,
            str(self.validator),
            str(self.root),
            "--unified-root",
            str(unified_root),
        ])
        if final_validation.returncode != 0:
            self.status("blocked_unified_validation", last_command_exit=final_validation.returncode)
            return False
        self.status("complete", manifest_images=20_000, unified_root=str(unified_root))
        return True

    def validate_standalone_only(self) -> bool:
        validation = self.run([sys.executable, str(self.validator), str(self.root)])
        if validation.returncode != 0:
            self.status(
                "blocked_standalone_validation",
                last_command_exit=validation.returncode,
            )
            return False
        self.status(
            "standalone_complete_not_integrated",
            manifest_images=20_000,
            integration_performed=False,
        )
        return True

    def execute(self) -> int:
        self.status("starting", job_status=self.job_counts(), manifest_images=self.manifest_count())
        if not self.advance_remote_batches():
            return 2
        self.delete_secret()
        self.status(
            "remote_generation_complete",
            api_key_deleted=not SECRET_PATH.exists(),
            job_status=self.job_counts(),
            manifest_images=self.manifest_count(),
        )
        if not self.repair_if_needed():
            return 3
        if not self.run_local_qa():
            return 4
        if self.no_integrate:
            return 0 if self.validate_standalone_only() else 5
        if not self.validate_and_integrate():
            return 5
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--no-integrate",
        action="store_true",
        help="validate the standalone dataset and stop without modifying data/reid",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.poll_seconds < 15:
        raise SystemExit("--poll-seconds must be at least 15")
    supervisor = Supervisor(args.root, args.poll_seconds, no_integrate=args.no_integrate)
    lock_path = supervisor.state_dir / "production_supervisor.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another production supervisor is already running") from None
        return supervisor.execute()


if __name__ == "__main__":
    raise SystemExit(main())
