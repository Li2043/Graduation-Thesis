# Path changes — F:\dense reward

## Method

Full-tree case-insensitive search across the copied `project\`, `scripts\`, `configs\` for `F:\`, `D:\`,
`C:\Users`, and the literal string `正式训练` (see `provenance/environment_setup_report.md` for the exact
grep commands used). This is a search of the actual copied source text, not a guess based on filenames.

## Hardcoded paths found

| File | Old hardcoded value | Kind | Action |
|---|---|---|---|
| `study_b_fairness_mappo\scripts\evaluate_formal_welfare.py` | `Path(r"F:\正式训练")` | write-path (CKPT_ROOT/BANK_ROOT/OUT_ROOT base) + sys.path | **Fixed** → `Path(__file__).resolve().parents[5]` |
| `study_b_fairness_mappo\scripts\evaluate_formal_behavioral.py` | `Path(r"F:\正式训练")` | same | **Fixed**, same pattern |
| `study_b_fairness_mappo\scripts\evaluate_high_burden_diagnostic.py` | `Path(r"F:\正式训练")` | same | **Fixed**, same pattern |
| `study_b_fairness_mappo\scripts\launch_wsc_formal_batch.py` | `PY_EXE`/`SCENARIO_BANK` → `F:\正式训练\...`; `MATRIX_CSV`/`LOGS_DIR`/`REGISTRY_CSV`/`COMMANDS_TXT` → `F:\正式训练_seed_replication_v1\...` | write-path into the OLD live bundle | **Fixed** → all relative to `Path(__file__).resolve().parents[5]` |
| `study_b_fairness_mappo\scripts\launch_wsc_formal_batch_v2.py` | same shape as above, `..._v2` variants | write-path into the OLD live bundle | **Fixed**, same pattern |
| `scripts\_watch_eval_done.py` | `r"C:\Users\SamChui\.cursor\projects\d\terminals\88775.txt"` | unrelated personal read path, dead code (not imported by anything in the traced call chain) | **Left unmodified** — intentionally inert, not a write path into the old env, fixing it changes no reachable behavior |
| `project\tests\**` (4 files) | various `C:\Users\HP\Desktop\...` literals | test fixture strings passed to path-format-checkers, never opened | **Left unmodified** — not real I/O paths |
| `scripts\_common.py` (docstring only) | mentions `F:\\正式训练`, `D:\\正式训练`, `C:\\thesis_formal_training` as illustrative examples of "any drive/folder this bundle could live at" | comment text | **Left unmodified** — not a path value, just prose explaining the file's own portability design (which is itself correct/relative: `BUNDLE_ROOT = Path(__file__).resolve().parent.parent`) |

## Intentional / read-only external paths (correctly NOT changed)

- Every orchestration script's own path resolution (`_common.py`'s `BUNDLE_ROOT`, `PROJECT_ROOT`,
  `SB_SCRIPTS`, etc., and `replication_common.py`'s constants derived from it) was already
  `Path(__file__).resolve().parent(...)`-relative in the source bundle — no absolute-path hardcoding to fix;
  copying the file to `F:\dense reward\` makes it resolve to the new root automatically.
- The wheelhouse used for the offline `pip install` intentionally still points at
  `F:\正式训练_seed_replication_v1\wheelhouse\cpu` (read-only, not copied — see
  `copied_files_manifest.txt`). This is a deliberate, documented external reference, not an oversight.
- `provenance/source_environment.md` and this file both reference old-bundle paths in prose — documentation,
  not executable path values.

## New write-path convention in the copy

Every script that resolves `BUNDLE_ROOT`/`PROJECT_ROOT` via `Path(__file__).resolve().parents[N]` (rather
than a hardcoded string) now transparently writes into `F:\dense reward\{checkpoints,outputs,logs,
verification}\...` instead of the old bundle's equivalents, with zero further code changes required — this
was the existing, already-portable design for the majority of the codebase (`_common.py` and everything
built on it); the 5 files listed above were the only exceptions found.
