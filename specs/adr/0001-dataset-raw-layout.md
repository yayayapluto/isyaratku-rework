# ADR 0001 — Complete `data/raw/wl-bisindo/` rather than widen the extract source

Status: proposed (needs user confirmation — SCOPE open question 1)
Date: 2026-10-16
Phase: 1 DISCOVER

## Context

The documented extraction command is:

```bash
python -m training.extract          # DEFAULT_SOURCE_DIR = data/raw/wl-bisindo
```

Verified state on disk (2026-10-16):

| Location | Count |
| --- | --- |
| `data/raw/wl-bisindo.zip` entries | 1600 |
| `data/raw/wl-bisindo/` `.mp4` | 1155 |
| `data/raw/pilih75/` `.mp4` | 446 |
| zip entries missing from `wl-bisindo/` | 445 |
| of those, found in `pilih75/` | 445 |
| unaccounted | 0 |
| `data/extracted/*.npz` | 1600 |
| `data/self/` | empty |

So the 1600-video dataset was extracted in **two** runs from **two** directories.
The packaged `.npz` set is complete. The documented command, run against the
documented layout, would produce 1155.

Nothing is lost: `training/extract.py:179` skips existing `.npz`, so the split was
harmless in practice. The defect is reproducibility, not data loss.

## Decision

**Preferred:** restore `data/raw/wl-bisindo/` to its full 1600 files, so the
documented command is true again.

Options considered:

1. **Extract the 445 missing files from `wl-bisindo.zip` into `wl-bisindo/`.**
   Keeps the documented command verbatim (`extract.py:49` unchanged), keeps
   `pilih75` as what its name implies (a curated 446-file selection), and makes
   the raw layout match the docs. Cost: one 2.13 GB zip read; ~445 file writes.
2. **Change `DEFAULT_SOURCE_DIR` or `iter_videos` to walk two directories.**
   Smaller diff, but it hides a broken layout behind code: the next person still
   cannot tell how many videos exist or where. Rejected as the primary fix.
3. **Delete `wl-bisindo/` and always extract from the zip.** Removes the
   two-directory ambiguity entirely, but adds a zip-reading step to every extract
   run and changes a working workflow. Rejected as larger than the problem.

Option 1 is chosen: the code is already correct; the filesystem drifted.

## Consequences

- `python -m training.extract` becomes reproducible from a clean checkout that has
  `data/raw/` populated.
- `pilih75/` keeps its meaning as a selection, not as an accidental second source.
- `wl-bisindo.zip` can stay as the 2.13 GB archive of record, or be removed once
  `wl-bisindo/` is complete — user's call; it is git-ignored either way.
- The npz count must be re-verified after the fix by running `extract` (should
  report 1600 skipped / 0 processed on an already-complete tree).

## Verification

```bash
python -c "import pathlib;print(len(list(pathlib.Path('data/raw/wl-bisindo').glob('*.mp4'))))"   # expect 1600
python -m training.dataset                                                                       # expect 1600 npz
```
