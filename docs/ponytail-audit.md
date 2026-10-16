# Ponytail Audit — Whole Codebase

Report-only. Applies nothing. Generated 2026-10-16.

Scope: `src/` (6315 L), `tests/` (8273 L), `training/` (3674 L), `tools/` (106 L), `requirements.txt`, root scripts.
Build output (`dist/`, `build/`, `__pycache__/`, `.pytest_cache/`) excluded — artifacts, not source.

Structure: over-engineering and complexity only. Correctness, security, and performance are out of scope.

Tags:
- `delete` — dead code, zero callers
- `dupe` — duplicated logic
- `yagni` — speculative flexibility, one-implementation abstraction
- `stdlib` — hand-rolled thing Python ships
- `shrink` — same logic, fewer lines

Format: `<tag> <what to cut>. <replacement>. [path]`. Ranked biggest cut first.

---

## Ranked findings

**1. `delete` `pyttsx3==2.99` from requirements.** Imported nowhere; `src/adapters/tts.py:21` explicitly refuses it ("Kenapa TIDAK ada fallback pyttsx3"). `piper`, which *is* imported at tts.py:48, is **not pinned**. [requirements.txt:6]
`-1 dep`; add `piper==1.8.0` (docs/tech-decisions.md:218).

**2. `yagni` 13 test files each re-declare `NO_FILE` + an identical `def config(**overrides)`.** 3 distinct hash-identical bodies; `fe10325f` appears 5×, `53ea8a68` 3×. Fixture, not per-file boilerplate. [tests/test_{features,idle_gating,landmarks,pipeline,pipeline_latency,pipeline_predictor,predictor,smoothing,static_path,views}.py]
`conftest.py` fixture `config` + module-scope `NO_FILE`. `-~60 L`.

**3. `delete` `DummyPredictor` entirely (~111 L incl. `_ranked`, `sign_labels`, 6 constants).** Zero production callers — grep `DummyPredictor` in `src/`+`tools/` matches only its own definition. `check_task.py:217` injects `TrainedPredictor`. Only tests subclass it. [src/core/predictor.py:89-200]
Move to test support, or delete. `-111 L`.

**4. `yagni` `_release_camera` + pixmap-scale block + `closeEvent` + `_stop_timer_on_error` duplicated in both views.** `_release_camera` bodies byte-identical (24 L); scale tail identical 18 L with comments. [debug_view.py:218-241, 514-522, 574-602 ↔ ready_view.py:191-214, 366-372]
Hoist `stop_pipeline(view)` + `_set_scaled(label, img)` into `check_task.py`. `-~70 L` net.

**5. `yagni` shared view utilities live in `ready_view.py`, imported *from* by `debug_view.py`** (10 names, debug_view.py:26-43, re-exported as class attrs). View-as-utility-module; makes #4 unfixable in place. [src/ui/ready_view.py]
Move `pixmap_bgr`/`_on_gui_thread`/`STATUS_*`/`_set_mode`/`_message_warning` to `check_task.py`. Enables #4. `-0 L` direct, unlocks ~70.

**6. `delete` `_stage_fps_label` — created, styled, laid out, never written.** Zero `.setText` callers; renders `FPS per stage: -` forever. [debug_view.py:125,136]
`-2 L` + 6 comment lines.

**7. `yagni` `PredictionProbe` (75 L file) wraps one implementation for one consumer.** `labels` pure delegate, `predict` = inner + one tuple assign. `_Predicts` Protocol has one implementer, never `isinstance`d. `format_ranked(count=…)` never overridden. [src/ui/prediction_probe.py]
Read `prediction.ranked` at build site; keep `format_ranked` as plain fn. `-~55 L`.

**8. `dupe` softmax + argsort top-k block verbatim in both predictors.** Same `shifted/exp/exp.sum()` → `argsort[::-1][:K]` → ranked comprehension. [adapters/predictor.py:99-106 ↔ adapters/static_predictor.py:189-216]
One `_softmax_top_k` in `core/predictor.py` (both already import it). `-14 L`.

**9. `dupe` `wrist_normalise_126` (features.py:93-119) ≡ `StaticTrainedPredictor._pra_proses` (static_predictor.py:265-280).** Byte-identical algorithm; features.py docstring admits "Duplikat sadar". Static path runs both per frame. [both]
Have adapter call `features.wrist_normalise_126`; delete one copy. `-27 L`.

**10. `delete` `StaticTrainedPredictor.predict_label` + `StaticHybridPredictor.predict_label` (~40 L).** Zero external callers; `static_path.py:144` calls `.predict`. Docstring claims it's the Smoother injection point — stale. [static_predictor.py:194-209, 309-320]
The pair is also finding #8's dupe, so deleting it subsumes part of it. `-40 L`.

**11. `yagni` `StaticPath` — 7 passthrough properties.** `last_letter`/`letters_emitted`/`words_completed`/`gap_seconds` read only by tests; `letters`/`word` read only internally; only `labels` has a real caller (check_task.py:241). Ctor param `gap_seconds` never passed in production. [static_path.py:44-57, 67-105]
Keep `labels` + 2 counters; rest plain attrs. `-16 L`.

**12. `delete` `incomplete_percentage()` — zero callers.** `IncompleteTracker` is what production uses (debug_view.py:17,74,179,394). [src/core/landmarks.py:90-98]
`-9 L`.

**13. `delete` `Smoother.now()` — zero callers.** Timestamps arrive as parameters by design (smoothing.py:7-8). Also drops `import time`. [smoothing.py:147-149]
`-3 L` + import.

**14. `stdlib` `_percentile` hand-rolled nearest-rank + a second `sorted()`.** `math` imported only for this; `label_latency` already sorts. [pipeline.py:33-41, 821-830]
Three index reads on the sorted list. `-14 L`, drops `import math`.

**15. `delete` `ModeSwitch.mode` property, `Smoother.now`, `hand_present` (1 caller), `_DELTA_HAND_OFFSET`, `_has_prediction_stage`, `_as_int`, `_identity`, `_smoother_status_snapshot`.** Each is a one-line wrapper or alias with 0-2 callers; `_identity` is a dataclass default production never exercises. [mode_switch.py:120-123, smoothing.py:50,73-76, pipeline.py:50-52,55-57,107-108,647-655]
Inline all. `-~28 L`.

**16. `delete` `_default_local_speaker`, `_match_local_speaker`, `_Jam`, `FAKE_VOICE`, `FAKE_SPEED`, `FAKE_ALL_MISSING_PHASE`, `DEFAULT_MODEL` alias, `audio_device`, `local_device` props, `class_count`.** Each grep-confirmed single-hit (its own definition). [tts.py:69-70,133-147,299-302,354-357,555-558; camera.py:130; landmark.py:138; predictor.py:33; static_predictor.py:121-124]
`FAKE_SPEED` is shadowed by the `speed=4.0` literal at camera.py:144. `-~40 L`.

**17. `yagni` `OpenCvCameraSource(backend=…)` never passed; `_SharedCameraSource` subclass exists only to skip the parent ctor.** Header already fixes MSMF. [camera.py:94-98; checks.py:58-71]
Keyword on parent replaces 15-L subclass. `-~17 L`.

**18. `yagni` `make_renderer`'s `raw_sink`/`mirror`/`landmarks` flags are constant per view.** Ready = `(None, True, False)`, Debug = `(store, True, True)`. Config object for two fixed configs. [check_task.py:508-578]
Two factories. `-12 L`.

**19. `dupe` `_pack_hand`/`_pack_pose` (constants swap) and `_synthetic_hand`/`_synthetic_pose` (count swap).** [landmark.py:176-211]
Two helpers. `-16 L`.

**20. `dupe` WAV-read + sounddevice-query boilerplate written 2× and 3×.** [tts.py:149-155,166-173,229-240,456-461,484-491]
`_baca_wav()` + `_query_devices()`. `-17 L`.

**21. `yagni` `FakePredictor(repeat=…)` / `FakeTTS.play(at=…)`+`_cooldown` / `model_path` props / `voice_available` passthrough.** Dead flexibility: `repeat=False` has one assertion, `at=` has no caller (guard moved to `SpeechSink.feed`), `voice_available` is a one-line delegate, static `model_path` has no reader. [predictor.py:216-240; tts.py:522-548,771-773; static_predictor.py:111-114]
`-~34 L`.

**22. `yagni` `_Langkah` class times 3 stages of a single 0.162 s op already bracketed by `t0`.** [check_task.py:41-47 + 8 call sites]
`-17 L` net.

**23. `shrink` `check_task._run_headless` pattern: 6 function-body imports, `_take_shared_camera`/`_close_quietly` one-caller getattr helpers, `app.py` unused `start_checks` bool + single-call `_run_gui`/`_run_headless`/`_parse_args` split.** [app.py:53-61; check_task.py]
`-~30 L`.

---

## Not findings (checked, keep)

- `cv2`/`mediapipe`/`numpy`/`pyvirtualcam`/`PySide6`/`sounddevice` — all load-bearing.
- `training/features_angles.py`, `diagnose_demo_v1.py`, `diagnose_urutan.py`, `record_self.py` — documented experiment/measurement scripts with cited results in `docs/tech-decisions.md`. Research artifacts, not debt.
- `train.py` vs `train_static.py` vs `train_demo_kata.py` — genuinely different model families (logreg-ringkas / MLP / demo-word). Shared `latih_logreg` already imported across (train_demo_kata.py:46).
- `_check_obs_virtual_camera` winreg walk — no stdlib enumerator for DirectShow CLSID instances.
- `tools/exe_*.py` (106 L total) — 3 tiny PyInstaller entry shims; the `sys.path` walk is duplicated 3× but needed for frozen bundles. ~10 L, low value.
- `virtual_camera.py` — clean; its two `ponytail:` comments mark deliberate ceilings.
- `configs/app.toml` — 65 lines, mostly measured numbers and rationale. Serves the "why default" contract.

---

## Net

```
net: -~600 lines, -1 dep (pyttsx3), +1 dep pin (piper)
```

Biggest single cut: **#3 `DummyPredictor` (-111 L, zero prod callers)**. Highest leverage for structural cleanup: **#5** — moving view utils out of `ready_view.py` turns #4 from judgement calls into mechanical deletions.

---

## Application caveats

- `_SharedCameraSource` (#17) and `FakeTTS(config)` (#21) are load-bearing for `test_views.py:416` and `check_task.py:599,603` — call sites must land in the same commit.
- #9's static-path normalisation runs twice per frame today; collapsing it changes a hot path, so verify against `tests/test_static_predictor.py` + `test_static_path.py` after.
