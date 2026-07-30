#!/usr/bin/env python3
"""Paper / formal artifact integrity snapshot for Stage 7A-1."""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PILOT_ROOT = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[4]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def collect_paper(repo: Path) -> list[Path]:
    out: list[Path] = []
    patterns = ["*.tex", "*.bib", "chapter*.md", "thesis*.md", "dissertation*.md", "*.docx"]
    for pat in patterns:
        out.extend(repo.rglob(pat))
    # exclude venv / releases noise
    cleaned = []
    for p in out:
        s = str(p).replace("\\", "/")
        if any(x in s for x in (".venv", "node_modules", "/releases/", "__pycache__")):
            continue
        cleaned.append(p)
    return sorted(set(cleaned))


def collect_formal(repo: Path) -> list[Path]:
    roots = [
        repo / "experiments" / "formal" / "stage6a_formal_training",
        repo / "experiments" / "formal" / "stage6b_h1",
    ]
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            s = str(p).replace("\\", "/")
            if any(x in s for x in ("/checkpoints/", ".pt", "/output.zip", "__pycache__")):
                # still hash small manifests/scripts/reports; skip huge binaries
                if p.suffix in {".pt", ".zip", ".parquet"} or "/checkpoints/" in s:
                    continue
            out.append(p)
    return sorted(set(out))


def write_csv(path: Path, files: list[Path], label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in files:
        try:
            rel = p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            rel = p.as_posix()
        rows.append(
            {
                "path": rel,
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
                "mtime_ns": getattr(p.stat(), "st_mtime_ns", int(p.stat().st_mtime * 1e9)),
                "label": label,
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["path", "sha256", "size_bytes", "mtime_ns", "label", "captured_at_utc"],
        )
        w.writeheader()
        w.writerows(rows)


def compare(before: Path, after: Path) -> list[str]:
    def load(p: Path) -> dict[str, str]:
        with p.open(encoding="utf-8", newline="") as f:
            return {r["path"]: r["sha256"] for r in csv.DictReader(f)}

    b, a = load(before), load(after)
    changed = []
    for k, hv in b.items():
        if k in a and a[k] != hv:
            changed.append(k)
    return changed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["before", "after", "compare"], required=True)
    args = parser.parse_args(argv)
    man = PILOT_ROOT / "manifests"
    if args.phase in {"before", "after"}:
        write_csv(man / f"paper_file_integrity_{args.phase}.csv", collect_paper(REPO_ROOT), "paper")
        write_csv(
            man / f"formal_artifact_integrity_{args.phase}.csv",
            collect_formal(REPO_ROOT),
            "formal",
        )
        print(f"wrote {args.phase} integrity snapshots")
        return 0
    paper_ch = compare(
        man / "paper_file_integrity_before.csv",
        man / "paper_file_integrity_after.csv",
    )
    formal_ch = compare(
        man / "formal_artifact_integrity_before.csv",
        man / "formal_artifact_integrity_after.csv",
    )
    report = {
        "changed_paper_files": paper_ch,
        "changed_formal_artifacts": formal_ch,
        "ok": len(paper_ch) == 0 and len(formal_ch) == 0,
    }
    import json

    (man / "integrity_compare.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
