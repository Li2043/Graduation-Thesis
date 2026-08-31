# START HERE

**Status (2026-08-19): the original mission below (steps 1-4) is
DONE — all 18 formal runs completed, evaluated, and audited.** There is
now a SECOND, separate mission in progress: an independent-seed
replication study. **Read `README.md` Section 14 and
`NEEDS_USER_DECISION.md` first** — there is a blocking hardware
decision pending before any new training starts. The steps below are
kept for historical/onboarding reference only; do not re-run them.

1. Copy this entire folder to a local SSD (do not train from the USB drive).训练结束之后需要把所有结果放入这个文件夹中，不要遗漏
2. Open the local copy's root folder in Cursor.
3. Read **README.md** completely -- it is written as an instruction
   document for Cursor and covers everything: environment setup, GPU
   benchmarking, finishing the in-progress curriculum, freezing the
   formal manifest, launching the 18 formal runs, monitoring, crash
   recovery, and evaluation.
4. Then run, in order: `00_SETUP.bat` -> `01_PREFLIGHT.bat` ->
   `02_CONTINUE_CURRICULUM.bat` (repeat until both new seeds reach
   C64_R50) -> `03_FREEZE_FORMAL.bat` -> `04_START_FORMAL.bat`.

This is a frozen thesis experiment. Do not change any scientific
parameter (observation, reward, welfare, DQN hyperparameters, seed
list, curriculum, held-out banks). README.md Section 9 lists exactly
what you may and may not do autonomously — Section 14 and
`protocol/new_protocol.md` §43 add the equivalent rules for the new
replication mission.
