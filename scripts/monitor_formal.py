#!/usr/bin/env python3
"""Poll status.py on an interval and print a live summary. Read-only --
never launches or kills anything. Ctrl+C to stop."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import BUNDLE_ROOT, python_exe  # noqa: E402


def main() -> int:
    interval = 60
    if len(sys.argv) > 1:
        interval = int(sys.argv[1])
    print(f"[monitor_formal] polling every {interval}s. Ctrl+C to stop.")
    try:
        while True:
            subprocess.run([python_exe(), str(Path(__file__).resolve().parent / "status.py")],
                            cwd=str(BUNDLE_ROOT))
            print(f"--- next check in {interval}s ---")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[monitor_formal] stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
