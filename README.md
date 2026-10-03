# isyaratku

Real-time BISINDO sign language translator for everyday conversation. The app reads sign language from a webcam, shows the result as text on screen, then speaks it through a virtual audio cable (VB-Cable) so meeting participants can hear it. Target problem: two-way communication between deaf speakers and hearing speakers.

What runs today: camera → landmark → prediction → smoothing → overlay + virtual camera, and label → TTS → cable. Prediction accuracy is **still low**, and the pipeline has never been tested end to end with Zoom or Meet — see the status section.

## What this app gives a user

1. **Translates word signs into on-screen text** across 32 BISINDO glosses, with a subtitle overlay over video.
2. **Speaks that text out loud** through offline piper TTS to VB-Cable, so a meeting hears it without internet access.
3. **Acts as a virtual camera** for meetings: participants see the app output, not the raw webcam image.
4. **Per-frame letter and number mode**, a separate path from the word model, with its own measured accuracy.

## Honest status

### Working and verified

- Full pipeline: camera → MediaPipe landmarks → windowing → prediction → smoothing → overlay → OBS Virtual Camera.
- Audio: offline piper TTS plays stable labels to the audio cable; measured cable RMS 0.1047 and audio duration 18.61 s for 8 labels (`docs/environment.md:110-124`).
- Two PySide6 interface modes: ready to use (Start/Stop) and debug (dashboard with FPS, drop, window, landmark, and word log panels).
- Start no longer freezes: `Signal(list)` → `Signal(object)` in `src/ui/check_task.py` keeps `CheckResults` intact across the signal boundary, so the pre-check no longer opens a second camera on the GUI thread.
- Word predictor p50 latency 0.066 ms (`docs/tech-decisions.md:176-182`); the headless path passes (`Headless smoke: LOLOS`).
- Silent failures now log: every `print(..., file=sys.stderr)` site that never reached the log now uses `logger.warning`, so predictor, TTS, and config failures leave traces in `logs/isyaratku-<date>.log`.
- **Letter and number static mode is available when enabled** (`static.enabled`, default `false`), with measured validation accuracy 0.9576 (letters, 26 classes) and 0.9868 (numbers, 11 classes) on the file's train/val split — not cross-signer generalization.

### Not done, and known limits

- **Word model accuracy is low.** Whole-window accuracy is 0.5745, but accuracy on hands-only windows is only 0.0856 on test signer3 (`docs/tech-decisions.md:45-52`). The model recognizes the signer who took part in the recording, not the signs in general: leave-one-signer-out is 0.0806–0.6058, while a random split inside a single signer reaches 0.9799.
- **Never tested with a meeting app.** Zoom and Meet have never been run together with this feed: whether the device appears in the meeting camera list, and whether participants see moving images, is unproven (`docs/tech-decisions.md`, section "Status slice 1: yang belum terbukti"). Only a local two-process round trip is proven.
- **The static datasets carry no signer metadata**, so letter and number accuracy is in-distribution, not cross-signer generalization. Letters also have 332 of 1,910 validation rows identical to training rows (17.38%), so both the dedup metric (0.9576) and the as-is metric (0.9534) are reported.
- **The 11th number class is named `?`** and its BISINDO meaning is not verified from the dataset source.
- **The `angka-dinamis` model is trained but not wired** into the app pipeline.
- **CPU load:** MediaPipe landmark extraction is the heaviest part of the pipeline; this machine's MSMF camera backend measures 28.56 fps, so output video fps depends on machine load during the demo.

## Installation

Prerequisites: Python 3.14 (developed and tested on 3.14.6), VB-Cable, and OBS Studio (for the virtual camera).

```bash
git clone https://github.com/yayayapluto/isyaratku-rework.git
cd isyaratku-rework
```

There is no virtualenv on the development machine; dependencies are installed directly on the system interpreter. Make sure `python` on PATH points at the Python 3.14 interpreter that holds those dependencies.

Most artifact paths are relative to CWD (`configs/app.toml`, `models/baseline.npz`, `models/mediapipe/*.task`), so all commands must run from the repository root.

## Main commands

| Command | Purpose | Notes |
| --- | --- | --- |
| `python -m src.ui.app` | Run the app in ready-to-use mode | Start/Stop buttons; pre-flight check of camera, virtual camera, and cable before Start |
| `python -m src.ui.app --mode debug` | Single-window debug dashboard | Panels for raw and overlaid video, output FPS, frames sent and dropped, voting and cooldown status, log of spoken words |
| `python -m src.ui.app --headless --seconds 5` | Full smoke test without a GUI | Uses `FakeCameraSource` and `FakeVirtualCameraSink` (`src/ui/app.py:117`); no hardware; last output line must be `Headless smoke: LOLOS` |
| `python -m training.setup_voice` | Download the piper voice (~62 MB) and warm the cache | Writes `id_ID-news_tts-medium.onnx` and `.onnx.json` to `models/tts/`; needs network on first use |
| `python -m training.setup_voice --check` | Check whether the voice is already present | No network, downloads nothing |
| `python -m training.setup_voice --warm-cache` | Warm the per-label WAV cache only | Pre-synthesizes every label before the demo so Start does not wait on the first synthesis |
| `python -m pytest -q -p no:cacheprovider` | Run the whole test suite | 276 passed, 0 skipped; add `QT_QPA_PLATFORM=offscreen` when there is no display |

## Letter and number mode

Static mode is a separate path from the word model, not extra labels on the word model. Reason: the word model uses a time window and a motion gate (`IDLE_MOTION_FLOOR` 0.05) that discards motionless signs, while letters and numbers are held still. Static mode therefore gets its own `Smoother` instance with `motion_floor = 0,0`, its gate is hand presence (`MIN_HAND_PRESENCE` 0.5), and its feature space is 126 columns (2 hands × 21 × 3) instead of the word model's 456 columns.

Measured on the source files' train/val split:

| Model | Classes | Validation accuracy | Macro F1 | Artifact |
| --- | --- | --- | --- | --- |
| Letters A–Z | 26 | 0.9576 dedup / 0.9534 as-is | 0.9563 / 0.9530 | `models/huruf.npz` |
| Static digits | 10 digits `0`–`9` plus `?` (11 classes, 11th named `?`) | 0.9868 | 0.9857 | `models/angka.npz` |
| Dynamic numbers | 10 | 0.8769 | 0.8730 | `models/angka-dinamis.npz` (not wired into the pipeline) |

Measured latency: 121 µs per frame for letters alone, 280 µs for the letters-plus-numbers hybrid, and only 1.26% of the 33 ms frame budget at 30 fps when combined with the word path. Inference is not the bottleneck.

To turn static mode on: uncomment the `[static]` section in `configs/app.toml`, set `enabled = true`, then run `python -m src.ui.app --mode debug`. Recognized letters assemble into words; a single 1.5 s pause closes a word. That pause threshold reuses `smoothing_cooldown_seconds`, so a repeated letter inside one name can split into two words, and this has not been measured on real recordings.

## Retraining

1. Prepare a dataset, then run `python -m training.extract` to extract landmarks into `data/extracted/` (file pattern `signer{N}_label{M}_sample{K}.npz`). The default `--source` is `data/raw/wl-bisindo/` (`training/extract.py:49`), so putting another dataset folder in `data/raw/` does NOT make it visible: pass `--source` and `--out` explicitly. The `data/` directory is not in git, so datasets must exist on the local disk.
2. Run `python -m training.train --stem <name>` to produce a new artifact under that name. To include self-recorded data as extra training data: `python -m training.train --stem <name> --signer-tambahan-train 99`, which goes into train only.
3. Convert to the runtime format: `python -m training.export_numpy --model models/<name>.joblib --out models/<name>.npz`. This script reads the `.joblib` as its source (`training/export_numpy.py:54`) and writes the `.npz` beside it; `--out` overrides that location.
4. **Runtime path warning:** `models/baseline.npz` is the default target (`src/adapters/predictor.py:33`) and there is no config key to replace it, so a new artifact must be placed as `models/baseline.npz`.
5. Static models are trained with a separate script: `python -m training.train_static --target huruf` (also `angka` and `angka-dinamis`), with `--model mlp|logreg` and `--dedup`. That script writes `.joblib`, `.json`, and `.npz` for static models.

`--stem baseline` overwrites the old artifacts. Use a different name when the baseline must be kept.

Honest dataset note: `data/extracted/` currently holds only 1,600 `.npz` files from signer0 to signer4 (32 word glosses). The word model still depends on one signer.

## Tests and logs

```bash
python -m pytest -q -p no:cacheprovider
```

The `-p no:cacheprovider` flag is not optional: an earlier run spent long enough to trigger a harness timeout. Tests use `QT_QPA_PLATFORM=offscreen`, so no display is needed. The suite sometimes hangs in teardown; when that happens, read the last output line to see which test ran last.

Every run writes a log to `logs/isyaratku-YYYY-MM-DD.log`, one file per day, local time with offset `+0700` (`docs/AGENTS.md:76-79`). Each line has the form `time offset LEVEL module.name message`; the last line shows the stage currently running. The `logs/` directory is not in git.

## Architecture

Dependency direction: `ui` → `core` ← `adapters`. The `src/core/` package does not import `src/ui/`, `src/adapters/`, or GUI code, contains no magic numbers, and every adapter has a fake for tests. The full rules are in `AGENTS.md`.

```
camera ─┐
        ├─ MediaPipe landmark ─ windowing (30 frames, stride 5) ─ normalisation
camera ─┘                                                       │
                                                                ▼
                                         LogReg prediction ─ smoothing (vote 3, 1.5 s cooldown)
                                                                │
                            ┌───────────────────────────────────┴────────────────┐
                            ▼                                                    ▼
                    frame overlay                                        standing label
                            │                                                    │
                            ▼                                                    ▼
              OBS Virtual Camera (message to meeting)              offline piper-tts ─ VB-Cable
```

The static path runs after the word predictor, not inside it: `_run_static_path` is called after `_run_predictor` in the capture loop, composed through the `on_static_word` hook. The word path therefore has no letter-specific branch.

## Configuration

All tuning numbers are centralized in `_CONTRACT` in `src/core/config.py`: 26 `section.key` keys with their defaults. `configs/app.toml` deliberately contains only comments, documenting defaults and measured results, with no active keys. Individual values can be overridden through an environment variable:

```bash
set ISYARATKU_CONFIG=<full-path-to>\app.toml
python -m src.ui.app
```

The TOML file holds only the keys you want to override, for example `camera.device_index = 2`. Keys outside the contract are rejected by `load_config()`, and `tests/test_config.py` keeps the contract in sync with the `AppConfig` fields. The three new keys for static mode are `static.enabled` (default `false`), `static.model_path_huruf`, and `static.model_path_angka`.

## Model honesty

Every number below is measured, not a claim:

| Metric | Value |
| --- | --- |
| Whole-window accuracy (test signer3) | 0.5745 |
| Hands-only window accuracy (test signer3) | **0.0856** |
| Glosses whose argmax is "Sore" | 12 of 32 |
| Glosses with ≤10 hands-only test windows | 10 of 32 |
| Word predictor p50 | 0.066 ms |
| Static predictor p50 (letters) | 121 µs |

Whole-window accuracy is 7× higher than reality because 924 of the 1,718 test windows are labeled "tidak ada isyarat", and that class dominates (`docs/tech-decisions.md:47`). The error pattern is one-directional: "Sore" absorbs 251 windows and "Bagaimana" 196 windows, while the reverse is nearly zero — the mark of signer bias, not of similar words.

Practical impact for the demo: this model still depends heavily on one signer. Use is only reasonable for signer0–3 until per-user calibration is done.

## CWD limits in the built EXE

These limits differ per path, so they must not be merged into one statement:

- `models/baseline.npz` (word path), the landmark `.task` files, and `configs/app.toml` are read directly with `Path(...)` and no resolver (`src/adapters/predictor.py:33`, `src/adapters/landmark.py:68-76`), as is `DEFAULT_CONFIG_PATH`, which is CWD-relative (`src/core/config.py:20`). For those, the EXE **only runs with CWD = repository root**; from any other folder it fails with `ModuleNotFoundError: No module named 'src'` (measured).
- `models/huruf.npz` and `models/angka.npz` (static path) have a resolver, `_resolusi_artifact`, that falls back to `sys._MEIPASS` when frozen (`src/adapters/static_predictor.py:51-67`). The bundled specs `isyaratku-ready.spec:7` and `isyaratku-debug.spec:7` do bundle both files, so the code path for them is present. Running the built EXE from a non-repository CWD has not been exercised yet, so that launch stays unverified.

## Building the EXE

Three separate EXEs are bundled with PyInstaller 6.21.0 from entry points in `tools/` (not `-m`, because PyInstaller needs a script file name): `tools/exe_ready.py` for ready-to-use mode, `tools/exe_debug.py` for debug mode, and `tools/exe_train.py` for self-training. Commands and measured numbers are in `build-exe.md`.

Target A core command, run from the repository root, with `<temp-build-dir>` outside the repo so `dist/` and `build/` never touch git:

```bash
python -m PyInstaller --noconfirm --clean --onedir --name isyaratku-ready \
    --distpath <temp-build-dir> --workpath <temp-build-dir> \
    --collect-submodules mediapipe --collect-binaries mediapipe --collect-data mediapipe \
    --collect-all PySide6 --collect-all sounddevice \
    --hidden-import piper --hidden-import cv2 --hidden-import numpy \
    --hidden-import pyvirtualcam --collect-submodules src \
    --exclude-module torch --exclude-module torchvision --exclude-module tensorboard \
    --exclude-module scipy --exclude-module sklearn \
    --add-data "configs/app.toml;configs" \
    --add-data "models/baseline.npz;models" \
    --add-data "models/mediapipe/hand_landmarker.task;models/mediapipe" \
    --add-data "models/mediapipe/pose_landmarker_lite.task;models/mediapipe" \
    tools/exe_ready.py
```

The differences between targets must be read in `build-exe.md` and not generalized: `isyaratku-ready.spec:7` and `isyaratku-debug.spec:7` bundle `models/huruf.npz` and `models/angka.npz` in addition to `models/baseline.npz`, while `isyaratku-train.spec` does not bundle either, because the Target C path does not use them. Real-GUI, real-TTS, `--onefile`, and cross-machine runs stay recorded as untested.

## Screenshots

These three files do not exist yet; the table below is a marker, not a claim that screenshots are available.

| Area | File | What it shows |
| --- | --- | --- |
| Ready-to-use mode | `docs/images/ui-siap-pakai.png` | Minimal window with Start/Stop buttons and a running status indicator |
| Debug dashboard | `docs/images/ui-debug.png` | Panels for raw and overlaid video, output FPS, frames sent and dropped, voting and cooldown status, word log |
| In a meeting app | `docs/images/meeting.png` | Translation output as an OBS Virtual Camera subtitle, with VB-Cable audio heard by participants |

## Running the demo

1. Open the meeting app (Zoom, Google Meet, and so on).
2. In the meeting camera settings, choose **OBS Virtual Camera** as the camera device, so the speaker's face image is replaced by the app pipeline output.
3. Run `python -m src.ui.app`, press **Start**, and wait for the status to read "berjalan".
4. For participants to **hear** you: choose **CABLE In 16 Ch (VB-Audio Virtual Cable)** as the microphone in the meeting app. The app routes TTS output to the cable player, and the meeting app captures the cable capture endpoint as a microphone.

VB-Cable and OBS Virtual Camera must already be installed; the automatic check at Start verifies the camera, the virtual camera, and the cable, then reports a specific error when one of them is not ready.

## Other documentation and attribution

| File | Contents |
| --- | --- |
| `build-exe.md` | Building the three PyInstaller EXEs: commands, required options, measured numbers |
| `docs/architecture.md` | Pipeline flow, config contract, module boundaries |
| `docs/tech-decisions.md` | Measured decisions, rejected thresholds, latency |
| `docs/implementation-plan.md` | Per-slice completion criteria, including unchecked ones |
| `docs/dataset-notes.md` | Candidate datasets and their licenses |
| `docs/environment.md` | Verified environment facts: Python, packages, VB-Cable, OBS |

Attribution, read from the repo and docs; software component licenses need rechecking before publication:

- Word dataset `glennleonali/wl-bisindo` (Kaggle), recorded as **CC BY-NC 4.0** in `docs/dataset-notes.md:30`: non-commercial use per its license, and attribution needs checking for competition use.
- MediaPipe Tasks (hand and pose landmarker), `google/mediapipe`: license needs checking.
- piper-tts and the Indonesian voice `rhasspy/piper-voices` (`id_ID-news_tts-medium.onnx`): license needs checking.
- BISINDO as a community sign language; sign dictionary sources need checking.
- Letter and number datasets come from `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks` (CC BY 4.0), `achmadnoer/alfabet-bisindo` (CC0), `agungmrf/indonesian-sign-language-bisindo`, and `sifaqeinstein/bisindo`, used to verify the 0–25 to A–Z label mapping.

## Roadmap

1. **Slice 6 — completing the ready-to-use mode and the debug dashboard** (in progress): stabilizing both modes, the automatic check at Start, actionable error messages, status indicators.
2. **Slice 7 — numbers and letters**: datasets are downloaded, the label mapping is verified, and models are trained and wired as a static path with a default-off switch. Remaining work is measuring the repeated-letter split threshold, verifying the meaning of the 11th number class, and testing on a real webcam.
3. **Once numbers and letters are truly stable**: per-user calibration (several repetitions per gloss before a demo), whose minimum prerequisites are already documented in `docs/tech-decisions.md:45-52`.
