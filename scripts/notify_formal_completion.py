#!/usr/bin/env python3
"""Create local formal completion notification; optional webhook / gh issue."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--completed-jobs", type=int, required=True)
    parser.add_argument("--failed-jobs", type=int, required=True)
    parser.add_argument("--results-branch", required=True)
    parser.add_argument("--results-commit", required=True)
    parser.add_argument(
        "--github-repository", default="https://github.com/Li2043/Graduation-Thesis"
    )
    parser.add_argument("--report-path", required=True)
    parser.add_argument(
        "--record-path",
        type=Path,
        default=None,
        help="Local notification JSON path",
    )
    parser.add_argument(
        "--create-github-issue",
        action="store_true",
        help="Create GitHub issue via gh after verified push",
    )
    parser.add_argument("--assignee", default="Li2043")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from thesis.formal.notify import (
        build_notification_payload,
        github_issue_title,
        maybe_notify,
    )

    payload = build_notification_payload(
        status=args.status,
        run_id=args.run_id,
        completed_jobs=args.completed_jobs,
        failed_jobs=args.failed_jobs,
        results_branch=args.results_branch,
        results_commit=args.results_commit,
        github_repository=args.github_repository,
        report_path=args.report_path,
    )
    record = Path(args.record_path) if args.record_path else (
        REPO_ROOT / "experiments" / "formal" / "stage6a_formal_training" / "notifications" /
        f"{args.run_id}_notification.json"
    )
    result = maybe_notify(record_path=record, payload=payload)
    print(json.dumps({"payload": payload, "notify_result": result}, indent=2))

    if args.create_github_issue:
        title = github_issue_title(args.status)
        body = payload["message"]
        cmd = [
            "gh",
            "issue",
            "create",
            "--repo",
            "Li2043/Graduation-Thesis",
            "--title",
            title,
            "--body",
            body,
            "--assignee",
            args.assignee,
        ]
        try:
            out = subprocess.check_output(cmd, text=True)
            print(out)
        except Exception as exc:  # noqa: BLE001
            # Issue failure must not alter scientific results
            fail_path = record.with_name(record.stem + "_issue_error.json")
            fail_path.write_text(
                json.dumps({"error": str(exc)}, indent=2), encoding="utf-8"
            )
            print(f"github issue failed (recorded): {exc}", file=sys.stderr)
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
