from __future__ import annotations

import textwrap

import pytest

from src.core import config as config_module
from src.core.config import CONTRACT_KEYS, AppConfig, ConfigError, load_config
from dataclasses import fields

DEFAULT_CONFIG_PATH = "configs/app.toml"


# -- defaults ----------------------------------------------------------------
def test_defaults_load_without_file(tmp_path) -> None:
    config = load_config(str(tmp_path / "tidak-ada.toml"))
    assert config.camera_device_index == 0
    assert config.camera_width == 640
    assert config.camera_height == 480
    assert config.camera_fps == 30
    assert config.landmark_max_num_hands == 2
    assert config.landmark_model_complexity == 0
    assert config.landmark_hand_model_path == "models/mediapipe/hand_landmarker.task"
    assert config.landmark_pose_model_path == (
        "models/mediapipe/pose_landmarker_lite.task"
    )
    assert config.window_frame_count == 30
    assert config.window_stride == 5
    assert config.smoothing_confidence_threshold == 0.7
    assert config.smoothing_vote_count == 3
    assert config.smoothing_cooldown_seconds == 1.5
    assert config.queue_max_size == 4
    assert config.tts_device_name == "CABLE Output"
    assert config.tts_rate == 160
    assert config.virtual_camera_backend == "obs"
    assert config.pipeline_read_failure_poll_seconds == 0.05


def test_default_path_exists_in_repo() -> None:
    assert config_module.DEFAULT_CONFIG_PATH == DEFAULT_CONFIG_PATH


# -- override ----------------------------------------------------------------
def test_file_overrides_one_key(tmp_path) -> None:
    path = tmp_path / "override.toml"
    path.write_text("[camera]\ndevice_index = 2\n", encoding="utf-8")
    config = load_config(str(path))
    assert config.camera_device_index == 2
    assert config.camera_width == 640


def test_int_accepted_for_float_key(tmp_path) -> None:
    path = tmp_path / "float.toml"
    path.write_text("[smoothing]\nconfidence_threshold = 1\n", encoding="utf-8")
    config = load_config(str(path))
    assert isinstance(config.smoothing_confidence_threshold, float)
    assert config.smoothing_confidence_threshold == 1.0


# -- fail fast ---------------------------------------------------------------
def test_wrong_type_raises_naming_the_key(tmp_path) -> None:
    path = tmp_path / "salah.toml"
    path.write_text("[camera]\ndevice_index = 'nol'\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="camera.device_index"):
        load_config(str(path))


def test_zero_or_negative_raises_naming_the_key(tmp_path) -> None:
    path = tmp_path / "nol.toml"
    path.write_text("[queue]\nmax_size = 0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="queue.max_size"):
        load_config(str(path))


def test_non_positive_stop_timeout_raises_naming_the_key(tmp_path) -> None:
    path = tmp_path / "nonsense.toml"
    path.write_text("[pipeline]\nstop_timeout_seconds = -1.0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="pipeline.stop_timeout_seconds"):
        load_config(str(path))


# -- unknown keys ------------------------------------------------------------
def test_unknown_key_warns_and_does_not_raise(tmp_path, caplog) -> None:
    path = tmp_path / "asing.toml"
    path.write_text(
        "[kamera]\ntidak_dikenal = 3\n\n[camera]\nngawur = 'apa'\n",
        encoding="utf-8",
    )
    config = load_config(str(path))
    assert config.camera_width == 640
    assert "key config tak dikenal" in caplog.text
    assert "camera.ngawur" in caplog.text
    assert "kamera" in caplog.text
    assert [r.levelname for r in caplog.records] == ["WARNING", "WARNING"]


# -- env var -----------------------------------------------------------------
def test_env_var_honoured(tmp_path, monkeypatch) -> None:
    path = tmp_path / "dari-env.toml"
    path.write_text("[window]\nstride = 9\n", encoding="utf-8")
    monkeypatch.setenv(config_module.ENV_CONFIG_PATH, str(path))
    assert load_config().window_stride == 9


def test_env_var_beats_default_path(tmp_path, monkeypatch) -> None:
    repo_default = tmp_path / "repo.toml"
    repo_default.write_text("[window]\nstride = 1\n", encoding="utf-8")
    other = tmp_path / "lain.toml"
    other.write_text("[window]\nstride = 42\n", encoding="utf-8")
    monkeypatch.setenv(config_module.ENV_CONFIG_PATH, str(other))
    monkeypatch.chdir(tmp_path)
    assert load_config().window_stride == 42


def test_missing_env_var_falls_back_to_default(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(config_module.ENV_CONFIG_PATH, raising=False)
    monkeypatch.chdir(tmp_path)
    assert load_config().window_frame_count == 30


# -- kontrak -----------------------------------------------------------------
def test_dataclass_fields_match_contract_keys() -> None:
    field_names = {f.name for f in fields(AppConfig)}
    expected = {key.replace(".", "_") for key in CONTRACT_KEYS}
    assert field_names == expected


def test_contract_key_names_are_documented_exactly() -> None:
    assert CONTRACT_KEYS == (
        "camera.device_index",
        "camera.width",
        "camera.height",
        "camera.fps",
        "landmark.max_num_hands",
        "landmark.model_complexity",
        "landmark.hand_model_path",
        "landmark.pose_model_path",
        "window.frame_count",
        "window.stride",
        "smoothing.confidence_threshold",
        "smoothing.vote_count",
        "smoothing.cooldown_seconds",
        "queue.max_size",
        "pipeline.stats_window",
        "pipeline.stop_timeout_seconds",
        "pipeline.read_failure_timeout_seconds",
        "pipeline.read_failure_poll_seconds",
        "tts.device_name",
        "tts.rate",
        "tts.enabled",
        "tts.speak_cooldown_seconds",
        "virtual_camera.backend",
    )


def test_committed_config_file_loads_cleanly() -> None:
    from pathlib import Path

    if not Path(DEFAULT_CONFIG_PATH).is_file():
        pytest.skip("berkas config default tidak ada")
    config = load_config(DEFAULT_CONFIG_PATH)
    assert config.queue_max_size == 4
