# isyaratku

Real-time BISINDO sign language translator for everyday conversation. The app reads sign language from a webcam, shows the result as text on screen, and speaks it as audio through a virtual speaker (VB-Cabel) so meeting participants can hear it. Targeted problem: two-way communication between deaf speakers and hearing speakers — the hearing speaker uses the app's camera as their face image in a meeting, and offline TTS turns sign labels into audible words.

What actually runs today: the camera → landmark → prediction → smoothing → overlay + virtual camera pipeline, and label → TTS → cable. Prediction accuracy is **still low** and it has never been tested end-to-end with Zoom/Meet — see the status section.

## Honest status

**Working and verified:**

- Full pipeline: camera → MediaPipe landmarks → windowing → prediction → smoothing → overlay → OBS Virtual Camera.
- Audio: offline piper TTS plays stable labels to the audio cable; measured cable RMS 0.1047 and audio duration 18.61 s for 8 labels (`docs/environment.md:110-124`).
- Two PySide6 interface modes: ready to use (Start/Stop) and debug (dashboard with FPS, drop, window, landmark, and word log panels).
- Start no longer freezes: `Signal(list)` → `Signal(object)` in `src/ui/check_task.py` keeps `CheckResults` intact across the signal boundary, so the pre-check no longer opens a second camera on the GUI thread (a second camera acquire used to take ~27.5 s and block the GUI thread; before the `warm_up` order fix).
- Predictor p50 latency 0.066 ms (`docs/tech-decisions.md:178-181`); the headless path passes (`Headless smoke: LOLOS`).
- Silent failures now log: every `print(..., file=sys.stderr)` site in the app never reached the log, because the logging setup in `src/core/logging.py` installs only a file handler, so a failing predictor, a failing TTS, or a config warning was completely silent. All those sites now log through `logger.warning` in `src/core/pipeline.py`, `src/adapters/tts.py` (`SpeechSink._record`), and `src/core/config.py` (`_warn`), so both failure and success leave traces in `logs/isyaratku-<date>.log`; the single intentional residue is `src/core/logging.py:130`, which writes to `sys.stderr` because that OSError fallback runs when the log file itself cannot be created, so stderr is the only available channel.

**Not done / limitations:**

- **Low accuracy.** Whole-window accuracy is 0.5745, but accuracy on hands-only windows is only 0.0856 on test signer3 (`docs/tech-decisions.md:47`). The baseline model recognizes the signer who took part in the recording, not the signs in general — leave-one-signer-out is only 0.0806–0.6058, while a random split within a single signer reaches 0.9799 (`docs/tech-decisions.md:50`). Full threshold numbers are in the model honesty section.
- **Never tested with a meeting app.** Zoom/Meet has never been run together with this feed: whether the device appears in the meeting camera list, and whether other participants see moving images, is still unproven (`docs/tech-decisions.md` section "Status slice 1: yang belum terbukti"). Only a local two-process round trip is proven.
- **Words only.** The ANGKA and HURUF datasets have not been downloaded; slice 7 (numbers, then letters) is still open.
- **CPU load:** MediaPipe landmark extraction is the heaviest part of the pipeline; on the developer machine the MSMF camera backend measures 28.56 fps capture (`configs/app.toml:43`), so output video fps depends on machine load during the demo.

## How to run

Prerequisites: Python 3.14 (this repo is developed and tested on Python 3.14.6). The project has no virtualenv on the developer machine — run directly with the system interpreter.

```bash
git clone https://github.com/yayayapluto/isyaratku-rework.git
cd isyaratku-rework

python -m training.setup_voice   # unduh voice piper (~62 MB) + pra-sintesis cache label
python -m training.train         # bangun artefak model di models/baseline.npz
python -m src.ui.app             # mode siap pakai
```

Options:

```bash
python -m src.ui.app --mode debug      # dasbor debug satu jendela
python -m src.ui.app --headless --seconds 5   # smoke test tanpa GUI (camera+sink fake)
```

`python -m training.train` needs the KATA dataset in `data/raw/`; `data/` is not tracked by git. `python -m training.setup_voice` can also be used on its own: `--warm-cache` to fill the WAV cache before a demo, `--check` to confirm a voice is present without network access.

Tests:

```bash
python -m pytest -q -p no:cacheprovider
```

`-p no:cacheprovider` is not optional: a previous run consumed enough time to cause harness timeouts. The test path uses `QT_QPA_PLATFORM=offscreen`, so no display is needed.

## How to run the demo

1. Open the meeting app (Zoom, Google Meet, and so on).
2. In your meeting camera settings, choose **OBS Virtual Camera** as the camera device. The speaker's face image is replaced by the app pipeline output.
3. Run `python -m src.ui.app`, press **Start**, and wait for the status to read "berjalan".
4. For meeting participants to **hear** you: choose the microphone **CABLE In 16 Ch (VB-Audio Virtual Cable)** in the meeting app. The app routes TTS output to the cable player, and the meeting app captures the cable capture endpoint as a microphone.

VB-Cabel and OBS Virtual Camera must already be installed on the system; the automatic check at Start verifies the camera, the virtual camera, and the cable, then reports a specific error if one of them is not ready.

Pressing **Start** used to take ~25-33 seconds: `cv2.VideoCapture`'s width/height/fps `set()` calls ran before the first frame read, and with the MSMF backend each property forced a full stream re-init (~6-7 s each). The camera now reads its first frame first and only sets a property when the stream does not already match the config. Measured on this machine: the pre-check's first camera acquire costs 6.8-7.9 s (the `VideoCapture` open alone is 6.1-8.2 s and the first read ~0.5 s), and the step that used to open a SECOND camera now reuses the pre-check's live handle for 0.113 s, so pressing Start no longer pays a second acquisition at all (`src/adapters/camera.py`, `warm_up`, `FRAME_WARMUP_MAX`). MSMF remains the default backend (28.56 fps).

## Architecture

Data flow:

```
kamera ─┐
        ├─ landmark MediaPipe ─ windowing (30 frame, stride 5) ─ normalisasi
kamera ─┘                                                        │
                                                                 ▼
                                          prediksi LogReg ─ smoothing (vote 3, cooldown 1,5 s)
                                                                 │
                            ┌────────────────────────────────────┴───────────────┐
                            ▼                                                    ▼
                 overlay di frame                                        label berdiri
                            │                                                    │
                            ▼                                                    ▼
              OBS Virtual Camera (pesan ke meeting)                    piper-tts offline ─ VB-Cabel
```

Dependency rule: `ui -> core <- adapters`. `src/core/` must not import GUI, hardware, or model libraries; it holds only the pipeline, configuration, and data types. All hardware access lives in `src/adapters/`, and every adapter has a fake counterpart (`FakeCameraSource`, `FakeLandmarkExtractor`, `FakeVirtualCameraSink`, `FakeTTS`) used by tests and by the headless path without real devices.

## Configuration

All tuning numbers are centralized in `_CONTRACT` in `src/core/config.py` (23 `section.key` keys, defaults included there). `configs/app.toml` deliberately contains only comments: documentation of defaults and measured results, with no active keys. Override individual values through an environment variable:

```bash
set ISYARATKU_CONFIG=D:\jalur\ke\app.toml    # Windows
python -m src.ui.app
```

The TOML file holds only the keys you want to override, for example `camera.device_index = 2`. Keys other than those in the contract are rejected by `load_config()`, and `tests/test_config.py` keeps the contract in sync with the `AppConfig` fields.

## Testing

- 217 tests, `python -m pytest -q -p no:cacheprovider`, all passing.
- Tests run offscreen (`QT_QPA_PLATFORM=offscreen`) using fake adapters, so CI or a laptop without a webcam can still run them.
- One real-device path that cannot be tested automatically (Zoom/Meet end-to-end) still has to be proven manually.

## Model honesty

These are my numbers, all measured, not claims:

| Metric | Value |
| --- | --- |
| Whole-window accuracy (test signer3) | 0.5745 |
| Hands-only window accuracy (test signer3) | **0.0856** |
| Gloss with argmax "Sore" | 12 of 32 |
| Gloss with ≤10 hands-only test windows | 10 of 32 |
| Predictor p50 | 0.066 ms |

Overall accuracy is 7x higher than reality because 924 of 1718 test windows are labeled "tidak ada isyarat"; that class is what dominates. The error pattern is one-directional: "Sore" absorbs 251 windows and "Bagaimana" 196 windows, while the reverse is nearly zero — the mark of signer bias, not of similar words.

For full per-class numbers:

- `docs/confusion-baseline.csv` — confusion matrix per gloss (test signer3)
- `docs/tech-decisions.md` — benchmarks, rejected thresholds, and the accuracy diagnosis
- `docs/implementation-plan.md` — per-slice completion criteria, including unchecked ones

Practical impact for the demo: this model still depends heavily on one signer. Use is only reasonable for signer0–3 until per-user calibration is done.

## Other documentation and attribution

Supporting documents in `docs/`:

| File | Contents |
| --- | --- |
| `docs/architecture.md` | Pipeline flow, config contract, module boundaries |
| `docs/tech-decisions.md` | Measured decisions, rejected thresholds, latency |
| `docs/implementation-plan.md` | Per-slice completion criteria, including unchecked ones |
| `docs/dataset-notes.md` | Candidate datasets and their licenses |
| `docs/environment.md` | Verified environment facts (Python, packages, VB-Cabel, OBS) |
| `docs/project-overview.md` | Project overview |

Attribution (names and licenses read from the repo/docs; software component licenses need to be rechecked before publication):

- KATA dataset: `glennleonali/wl-bisindo` (Kaggle), recorded as **CC BY-NC 4.0** in `docs/dataset-notes.md:30` — non-commercial use per its license, attribution needs checking for competition use.
- MediaPipe Tasks (hand + pose landmarker), `google/mediapipe` — license needs checking.
- piper-tts and the Indonesian voice `rhasspy/piper-voices` (`id_ID-news_tts-medium.onnx`) — license needs checking.
- BISINDO as a community sign language; sign dictionary sources need checking.

## Roadmap

1. **Slice 6 — completing the ready-to-use mode and the debug dashboard** (in progress): stabilizing both modes, the automatic check at Start, actionable error messages, status indicators.
2. **Slice 7 — numbers**, then **letters** if there is enough time, on the condition that slice 6 is stable and the same pipeline does not branch specially per sign type.
3. **Once numbers are truly stable:** per-user calibration (several repetitions per gloss before a demo), whose minimum prerequisites are already documented in `docs/tech-decisions.md:50`.
