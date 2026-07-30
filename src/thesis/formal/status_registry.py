"""Atomic central status registry for formal multi-job orchestration."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


TERMINAL_COMPLETE = "COMPLETE"
TERMINAL_FAILED = "FAILED_WITH_REASON"
TERMINAL_INTERRUPTED = "INTERRUPTED_RESUMABLE"
STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"

ALLOWED_TERMINAL = {TERMINAL_COMPLETE, TERMINAL_FAILED, TERMINAL_INTERRUPTED}


class FormalStatusRegistry:
    """Process-safe-ish atomic JSON registry (replace via temp + os.replace)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"jobs": {}, "replace_failed_seeds": False})

    def _read(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: dict[str, Any]) -> None:
        data = dict(data)
        data["replace_failed_seeds"] = False
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.path.name + ".", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    def get(self, job_id: str) -> dict[str, Any] | None:
        return self._read().get("jobs", {}).get(job_id)

    def upsert(self, job_id: str, record: dict[str, Any]) -> None:
        data = self._read()
        jobs = dict(data.get("jobs") or {})
        jobs[job_id] = dict(record)
        data["jobs"] = jobs
        self._write(data)

    def is_complete(self, job_id: str) -> bool:
        rec = self.get(job_id)
        return bool(rec and rec.get("status") == TERMINAL_COMPLETE)

    def should_skip(self, job_id: str) -> bool:
        return self.is_complete(job_id)

    def should_resume(self, job_id: str) -> bool:
        rec = self.get(job_id)
        return bool(rec and rec.get("status") == TERMINAL_INTERRUPTED)

    def account_all(self, expected_job_ids: list[str]) -> dict[str, int]:
        data = self._read()
        jobs = data.get("jobs") or {}
        counts = {
            "expected": len(expected_job_ids),
            "complete": 0,
            "failed": 0,
            "interrupted": 0,
            "pending_or_missing": 0,
            "running": 0,
        }
        for jid in expected_job_ids:
            rec = jobs.get(jid)
            if rec is None:
                counts["pending_or_missing"] += 1
                continue
            st = rec.get("status")
            if st == TERMINAL_COMPLETE:
                counts["complete"] += 1
            elif st == TERMINAL_FAILED:
                counts["failed"] += 1
            elif st == TERMINAL_INTERRUPTED:
                counts["interrupted"] += 1
            elif st == STATUS_RUNNING:
                counts["running"] += 1
            else:
                counts["pending_or_missing"] += 1
        return counts


__all__ = [
    "ALLOWED_TERMINAL",
    "FormalStatusRegistry",
    "STATUS_PENDING",
    "STATUS_RUNNING",
    "TERMINAL_COMPLETE",
    "TERMINAL_FAILED",
    "TERMINAL_INTERRUPTED",
]
