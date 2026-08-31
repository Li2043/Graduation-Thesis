"""Completion notification payload tests."""

from thesis.formal.notify import build_notification_payload, github_issue_title, maybe_notify


def test_notification_payload_and_local_record(tmp_path):
    payload = build_notification_payload(
        status="COMPLETE",
        run_id="r1",
        completed_jobs=30,
        failed_jobs=0,
        results_branch="formal/runner-100k",
        results_commit="abc",
        github_repository="https://github.com/Li2043/Graduation-Thesis",
        report_path="reports/x.md",
    )
    for key in ("content", "text", "message", "title", "status", "run_id"):
        assert key in payload
    assert github_issue_title("COMPLETE").startswith("[Formal 100K]")
    record = tmp_path / "note.json"
    result = maybe_notify(record_path=record, payload=payload)
    assert record.is_file()
    assert result["webhook_attempted"] is False
