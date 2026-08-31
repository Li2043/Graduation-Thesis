#!/usr/bin/env python3
"""Watch 07_EVALUATE terminal file; print AGENT_EVAL_DONE when finished."""
from __future__ import annotations

import sys
import time
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit("usage: _watch_eval_done.py <terminal_log_path>")
TERM = Path(sys.argv[1])
print("EVAL_WATCHER_ARMED", flush=True)
while True:
    text = TERM.read_text(encoding="utf-8", errors="ignore") if TERM.exists() else ""
    if "exit_code:" in text and "ended_at:" in text:
        print("AGENT_EVAL_DONE", flush=True)
        print(text[-3000:], flush=True)
        raise SystemExit(0)
    time.sleep(30)
