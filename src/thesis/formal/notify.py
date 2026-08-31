"""Completion notification support for formal 100K training."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_notification_payload(
    *,
    status: str,
    run_id: str,
    completed_jobs: int,
    failed_jobs: int,
    results_branch: str,
    results_commit: str,
    github_repository: str,
    report_path: str,
) -> dict[str, Any]:
    ts = datetime.now(timezone.utc).isoformat()
    title = (
        "[Formal 100K] Training COMPLETE"
        if status.upper() == "COMPLETE"
        else "[Formal 100K] Training FAILED"
    )
    body = (
        f"{title}\n"
        f"run_id={run_id}\n"
        f"completed={completed_jobs} failed={failed_jobs}\n"
        f"branch={results_branch} commit={results_commit}\n"
        f"repo={github_repository}\n"
        f"report={report_path}\n"
        f"ts={ts}"
    )
    return {
        "status": status,
        "run_id": run_id,
        "completed_jobs": int(completed_jobs),
        "failed_jobs": int(failed_jobs),
        "results_branch": results_branch,
        "results_commit": results_commit,
        "GitHub_repository": github_repository,
        "github_repository": github_repository,
        "report_path": report_path,
        "completion_timestamp": ts,
        # Compatible webhook fields
        "content": body,
        "text": body,
        "message": body,
        "title": title,
    }


def write_local_notification(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def post_webhook(url: str, payload: dict[str, Any], *, timeout: float = 15.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {
                "ok": True,
                "status_code": getattr(resp, "status", 200),
                "error": None,
            }
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        return {"ok": False, "status_code": None, "error": str(exc)}


def maybe_notify(
    *,
    record_path: Path,
    payload: dict[str, Any],
    webhook_env_var: str = "FORMAL_NOTIFY_WEBHOOK_URL",
) -> dict[str, Any]:
    """Always write local record. Optional webhook from env (never stored in Git)."""
    write_local_notification(record_path, payload)
    url = os.environ.get(webhook_env_var, "").strip()
    result = {
        "local_record": str(record_path),
        "webhook_attempted": bool(url),
        "webhook": None,
    }
    if url:
        result["webhook"] = post_webhook(url, payload)
    return result


def github_issue_title(status: str) -> str:
    if status.upper() == "COMPLETE":
        return "[Formal 100K] Training COMPLETE"
    return "[Formal 100K] Training FAILED"


__all__ = [
    "build_notification_payload",
    "github_issue_title",
    "maybe_notify",
    "post_webhook",
    "write_local_notification",
]
