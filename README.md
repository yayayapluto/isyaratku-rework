# isyaratku

Real-time BISINDO (Indonesian Sign Language) word translator for everyday conversations. The app reads signs from a webcam, renders them as text over the video, speaks them through offline TTS into a virtual audio cable, and feeds the result to a video meeting as a virtual camera. Target problem: two-way communication between deaf and hearing speakers without an interpreter.

## Contents

- [What works today](#what-works-today)
- [Current accuracy — read before trusting a demo](#current-accuracy--read-before-trusting-a-demo)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Development](#development)
- [Building a standalone executable](#building-a-standalone-executable)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [TODO](#todo)

## What works today

- Full pipeline: camera → MediaPipe landmarks → windowing → prediction → smoothing → overlay → OBS Virtual Camera.
- Offline TTS: piper synthesizes stable labels to the audio cable; no network needed at demo time.
- Two interfaces: a ready-to-use mode (Start/Stop) and a debug dashboard with FPS, frames sent/dropped, voting status, and a word log.
- A separate per-frame letter and number path (`static.*` config keys), off by default.
- A headless smoke run with fake camera and fake virtual camera: `python -m src.ui.app --headless --seconds 3` prints `Headless smoke: LOLOS`.
- Run the automated suite with `python -m pytest -q -p no:cacheprovider` (read the current count and duration from its final line; no count is asserted here because the suite keeps growing).

## Current accuracy — read before trusting a demo

| Metric | Value | Source |
| --- | --- | --- |
| Whole-window accuracy, test signer3 | 0.5745 | `models/baseline.json` |
| Accuracy on hands-only windows, test signer3 | **0.0856** | `models/baseline.json` |
| Test windows labeled "tidak ada isyarat" (no sign) | 924 of 1,718 | `models/baseline.json` → `jumlah_window` |
| Classes trained | 33 sign glosses | `models/baseline.json` |
| Letters A–Z, validation accuracy | 0.9576 dedup / 0.9534 as-is | `models/huruf.json` |
| Static digits `0`–`9` plus `?` (11 classes), validation accuracy | 0.9868 | `models/angka.json` |
| Dynamic numbers (10 classes), validation accuracy | 0.8769 | `models/angka-dinamis.json` |

Three limits sit behind these numbers:

1. **The word model is signer-dependent, not sign-general.** All glosses were recorded by a fixed set of signers, so cross-signer accuracy is far below the tables above. Measured on this machine: hold-one-signer-out 0.0806–0.6058 versus 0.9799 on a random split inside a single signer.
2. **Static (letter/number) datasets carry no signer metadata.** Their accuracy is in-distribution only. Letters also have 332 of 1,910 validation rows byte-identical to training rows (17.38%), which is why both the deduplicated and as-is numbers are reported.
3. **Never tested with a real meeting app.** Zoom and Meet have not been run against this feed, so whether the virtual camera device shows up and renders correctly there is unproven. Only a local two-process round trip is proven.

Consequence for a demo: today the word path is reliable mainly for the signers in the recording set, and the meeting integration is unverified. The letter/number path needs the on-camera signer to match the recorded population.

## Prerequisites

| Requirement | Version | Purpose |
| --- | --- | --- |
| Python | 3.14 developed and tested on 3.14.6 | runtime for all commands below |
| VB-Cable | latest | virtual audio device so meeting participants hear the TTS output |
| OBS Studio | with the virtual camera built in | appears as `OBS Virtual Camera` in meeting apps |
| Webcam | any UVC device | sign capture; the development machine uses 640×480 at 30 fps |
| Optional: a GPU | not required | only speeds up MediaPipe; CPU runs the pipeline |

Webcam backend note: `src/adapters/camera.py` uses `cv2.CAP_MSMF` because it measured 28.56 fps versus 8.80 fps for `CAP_DSHOW` on the development webcam. On a different machine the ranking may differ; change `CAMERA_BACKEND` there if capture drops below 25 fps.

## Installation

1. Clone the repository and enter it:

```bash
git clone https://github.com/yayayapluto/isyaratku-rework.git
cd isyaratku-rework
```

2. Install the pinned dependencies onto the active Python interpreter. The project has no virtualenv on the development machine; dependencies live directly on the system interpreter.

```bash
python -m pip install -r requirements.txt
```

On a fresh install pip may resolve `opencv-contrib-python` for the same 5.0.0.93 version, because mediapipe depends on it; the commands here were verified against the environment already present on the machine.

3. Download the piper TTS voice (~62 MB, one time, needs network):

```bash
python -m training.setup_voice
```

4. Verify the install without a GUI, a webcam, or VB-Cable:

```bash
QT_QPA_PLATFORM=offscreen python -m src.ui.app --headless --seconds 3
```

Expected last line: `Headless smoke: LOLOS`.

Most artifact paths are relative to the current working directory (`configs/app.toml`, `models/baseline.npz`, `models/mediapipe/*.task`), so run every command from the repository root.

## Usage

### Minimal working example

```bash
python -m src.ui.app
```

Opens the ready-to-use window. Press **Start**; the automated check verifies the camera, the virtual camera, and the audio cable, then the pipeline begins. To be heard in a meeting, select **CABLE In 16 Ch (VB-Audio Virtual Cable)** as the microphone in that meeting app.

### Full setup

```bash
# window 1 — debug dashboard: panels for FPS, dropped frames, voting, word log
python -m src.ui.app --mode debug

# window 2 — headless run with fake adapters, 5 seconds, for a quick regression check
QT_QPA_PLATFORM=offscreen python -m src.ui.app --headless --seconds 5
```

Meetings: pick **OBS Virtual Camera** as the camera device so participants see the app output instead of the raw webcam, and pick **CABLE In 16 Ch (VB-Audio Virtual Cable)** as the microphone so participants hear the spoken labels.

### Turning on the letter and number path

Uncomment the `[static]` block in `configs/app.toml` and set `enabled = true`, then run `python -m src.ui.app --mode debug`. Each recognized letter appends to a word; a 1.5 s pause closes the word. That pause threshold reuses `smoothing.cooldown_seconds`, so a repeated letter in one name can split into two words. This has not been measured on real recordings.

## Configuration

All tuning numbers live in a single contract: `_CONTRACT` in `src/core/config.py`, currently 26 `section.key` entries. `configs/app.toml` deliberately holds only comments and no active keys, so out of the box every value comes from `_CONTRACT`. To override values, write a TOML file with just the keys you want and point the app at it:

```bat
set ISYARATKU_CONFIG=%CD%\override.toml
python -m src.ui.app
```

`ISYARATKU_CONFIG` wins over the default path. `override.toml` may contain only keys present in `_CONTRACT`; unknown keys are rejected. `tests/test_config.py` keeps the contract and the `AppConfig` fields in sync.

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `camera.device_index` | int | `0` | webcam index passed to OpenCV |
| `camera.fps` | int | `30` | requested capture frame rate |
| `camera.width` / `camera.height` | int | `640` / `480` | requested capture resolution |
| `landmark.hand_model_path` | str | `models/mediapipe/hand_landmarker.task` | hand landmarker bundle |
| `landmark.pose_model_path` | str | `models/mediapipe/pose_landmarker_lite.task` | pose landmarker bundle |
| `landmark.max_num_hands` | int | `2` | hands extracted per frame |
| `landmark.model_complexity` | int | `0` | MediaPipe model complexity |
| `window.frame_count` | int | `30` | frames per prediction window |
| `window.stride` | int | `5` | frames between consecutive windows |
| `smoothing.confidence_threshold` | float | `0.7` | minimum probability to vote |
| `smoothing.vote_count` | int | `3` | consecutive votes required to emit a label |
| `smoothing.cooldown_seconds` | float | `1.5` | minimum delay between two emitted labels |
| `tts.enabled` | bool | `true` | speak labels through piper TTS |
| `tts.device_name` | str | `CABLE Output` | audio endpoint name the app matches |
| `tts.rate` | int | `160` | speech rate in words per minute |
| `tts.speak_cooldown_seconds` | float | `2.5` | minimum delay between two spoken labels |
| `pipeline.stats_window` | int | `240` | frames used for the FPS average on the dashboard |
| `pipeline.read_failure_poll_seconds` | float | `0.05` | interval between retries after a frame read failure |
| `pipeline.read_failure_timeout_seconds` | float | `5.0` | total time tolerated for repeated read failures |
| `pipeline.stop_timeout_seconds` | float | `2.0` | time allowed for the capture thread to stop |
| `queue.max_size` | int | `4` | bounded queue between capture and processing |
| `virtual_camera.backend` | str | `obs` | virtual camera backend selected at runtime |
| `static.enabled` | bool | `false` | enable the per-frame letter/number path |
| `static.model_path_huruf` | str | `models/huruf.npz` | letter model for the static path |
| `static.model_path_angka` | str | `models/angka.npz` | digit model for the static path |

There are no other environment variables. No secrets, tokens, or credentials are required.

Note on `tts.device_name`: the value `CABLE Output` names the cable family, not a literal device string. `match_cable_device` in `src/adapters/tts.py` matches the substring `cable` on endpoints whose `max_output_channels > 0`, because the endpoint literally named "CABLE Output" is the capture side (0 output channels) and would send audio nowhere.

## Project structure

```
isyaratku-rework/
├── src/
│   ├── core/          pure logic: landmark normalisation, windowing, smoothing, pipeline, static path
│   ├── adapters/      everything that touches the outside: camera, MediaPipe, TTS, virtual camera, model artifacts
│   └── ui/            PySide6 views (ready + debug), overlay rendering, headless entry point
├── training/          extraction, training, export — not imported by the runtime
├── tests/             pytest suite using fake adapters
├── models/            trained artifacts (tracked in git)
│   ├── baseline.{joblib,json,npz}       33-class word model
│   ├── huruf.{joblib,json,npz}          26-class letter model
│   ├── angka.{joblib,json,npz}          11-class digit model
│   ├── angka-dinamis.{...}              10-class dynamic number model, not wired into the pipeline
│   └── mediapipe/*.task                 MediaPipe hand and pose landmarker bundles
├── configs/app.toml    tunable numbers, documented, no active keys
├── data/              raw and extracted datasets — gitignored, must exist locally
├── docs/              project contracts: decisions, architecture, plan, dataset notes
├── tools/             PyInstaller entry points for the ready, debug, and train executables
└── logs/              daily runtime logs — gitignored
```

The dependency direction is `ui → core ← adapters`. `src/core/` imports neither `src/ui/` nor `src/adapters/` and contains no GUI code. Full rules: `docs/AGENTS.md`.

## Development

Run the whole suite:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q -p no:cacheprovider
```

`-p no:cacheprovider` is not optional: without it an earlier run stalled long enough to exceed the harness timeout. Read the pass count from the final line; the suite occasionally hangs during teardown, and the printed count is still valid. Offscreen mode is needed on a headless machine, though the tests never open a window.

Quickest check without any hardware:

```bash
QT_QPA_PLATFORM=offscreen python -m src.ui.app --headless --seconds 3
```

### Measured latency on this machine (30 fps, 33.33 ms budget)

| Path | p50 | Share of frame budget |
| --- | --- | --- |
| `StaticPath.feed` with fake predictor | 80.2 µs | 0.24% |
| `StaticPath.feed` with real hybrid predictor | 220.0 µs | 0.66% |
| Feature bridge `normalise` → 126 columns | 26.9 µs | 0.08% |
| `StaticTrainedPredictor.predict` (letters) | 64.5 µs | 0.19% |
| `StaticHybridPredictor.predict` (letters + digits) | 135.3 µs | 0.41% |

Inference is not the bottleneck of the pipeline; landmark extraction is.

### Retraining

```bash
# 1. extract landmarks from the dataset videos into data/extracted/
python -m training.extract --source data/raw/<dataset> --out data/extracted

# 2. train and save a new artifact (baseline, or any --stem name)
python -m training.train --stem <name>

# 3. convert to the numpy-only runtime format (no sklearn in the frozen app)
python -m training.export_numpy --model models/<name>.joblib --out models/<name>.npz

# 4. train the static letter / digit models separately
python -m training.train_static --target huruf
python -m training.train_static --target angka
```

Three gotchas:

- `python -m training.train --stem baseline` overwrites the tracked default. Use `--stem <name>` to keep the baseline, and note that `src/adapters/predictor.py` hardcodes `models/baseline.npz` as the runtime target — a different stem is not reachable from config.
- `training/extract.py` defaults to `data/raw/wl-bisindo/`. Dropping another dataset folder into `data/raw/` does not make it visible; pass `--source` and `--out` explicitly.
- `data/` is gitignored, so the datasets must exist on the local disk before extraction can run.

## Building a standalone executable

Full commands, the required PyInstaller options, and measured build numbers live in `build-exe.md`. In short:

```bash
python -m PyInstaller --noconfirm --clean --onedir --name isyaratku-ready --collect-all PySide6 tools/exe_ready.py
```

The `.spec` files in the repository root (`isyaratku-ready.spec`, `isyaratku-debug.spec`, `isyaratku-train.spec`) are the maintained versions; the deployed executables have not been tested from a non-repository working directory, with real GUI hardware, with real TTS, as `--onefile`, or on another machine.

## Documentation

| File | Contents |
| --- | --- |
| `docs/AGENTS.md` | mandatory entry point: goals, directory map, engineering rules |
| `docs/architecture.md` | pipeline flow, config contract, threading, fake adapters |
| `docs/tech-decisions.md` | measured decisions, rejected options, honest metrics |
| `docs/implementation-plan.md` | per-slice acceptance criteria, including unchecked ones |
| `docs/dataset-notes.md` | candidate datasets and their licenses |
| `docs/environment.md` | verified environment: Python, packages, VB-Cable, OBS |

### Attribution

- Word dataset `glennleonali/wl-bisindo` (Kaggle), recorded as **CC BY-NC 4.0** in `docs/dataset-notes.md`: non-commercial only; recheck the license before any commercial use.
- Letter datasets: `suryaadji/bisindo-alphabet-mediapipe-hand-landmarks` (CC BY 4.0), `achmadnoer/alfabet-bisindo` (CC0), `agungmrf/indonesian-sign-language-bisindo`, `sifaqeinstein/bisindo`.
- [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/vision) — hand and pose landmarker bundles; license see upstream.
- [piper-tts](https://github.com/rhasspy/piper) and the Indonesian voice `rhasspy/piper-voices` (`id_ID-news_tts-medium.onnx`); licenses see upstream.
- BISINDO is a community sign language; dictionary sources still need checking before publication.

## Contributing

There is no `CONTRIBUTING.md`. Read `docs/AGENTS.md` first — it defines the dependency direction, the ban on GUI imports inside `src/core/`, the no-magic-numbers rule, the requirement of a fake for every adapter, and the commit-per-change discipline. Commits use Conventional Commits in Indonesian subjects, at most 72 characters.

## License

No `LICENSE` file exists in the repository yet. All datasets remain under their own terms (see Attribution). Add one before any distribution.

## TODO

Those items are missing from the repository and need a human decision:

- `LICENSE` file — none present.
- `CONTRIBUTING.md` — workflow is documented only inside `docs/AGENTS.md`.
- Real-meeting verification — Zoom and Meet have never been tested against this feed.
- End-to-end test with a real TTS device and real virtual camera.
- Screenshot assets — `docs/images/` is empty, so no screenshots are embedded above.
- The 11th digit class is named `?`; its meaning is unverified from the dataset source.
- Signer metadata for the letter and number datasets, to make their numbers auditable.
- This project has no CI workflow, so no badges are shown.
