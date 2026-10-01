"""Loader konfigurasi IsyaratKu Cam.

Kontrak key (lihat docs/architecture.md) dicatat di ``_CONTRACT`` dan dipakai
untuk: nama field dataclass, default, dan validasi tipe. Menambah key berarti
menambah satu baris di tabel itu saja.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENV_CONFIG_PATH = "ISYARATKU_CONFIG"
DEFAULT_CONFIG_PATH = "configs/app.toml"


class ConfigError(ValueError):
    """Config tidak valid: tipe salah, nilai di luar batas, atau key hilang."""


# (section, key, tipe, default)
_CONTRACT: tuple[tuple[str, str, str, Any], ...] = (
    ("camera", "device_index", "int", 0),
    ("camera", "width", "int", 640),
    ("camera", "height", "int", 480),
    ("camera", "fps", "int", 30),
    ("landmark", "max_num_hands", "int", 2),
    ("landmark", "model_complexity", "int", 0),
    ("landmark", "hand_model_path", "str", "models/mediapipe/hand_landmarker.task"),
    ("landmark", "pose_model_path", "str", "models/mediapipe/pose_landmarker_lite.task"),
    ("window", "frame_count", "int", 30),
    ("window", "stride", "int", 5),
    ("smoothing", "confidence_threshold", "float", 0.7),
    ("smoothing", "vote_count", "int", 3),
    ("smoothing", "cooldown_seconds", "float", 1.5),
    ("queue", "max_size", "int", 4),
    ("pipeline", "stats_window", "int", 240),
    ("pipeline", "stop_timeout_seconds", "float", 2.0),
    ("tts", "device_name", "str", "CABLE Output"),
    ("tts", "rate", "int", 160),
    ("virtual_camera", "backend", "str", "obs"),
)

#: Nama key persis seperti di kontrak, urut sesuai tabel.
CONTRACT_KEYS: tuple[str, ...] = tuple(f"{sec}.{key}" for sec, key, _, _ in _CONTRACT)

_KINDS = {f"{sec}.{key}": kind for sec, key, kind, _ in _CONTRACT}
_DEFAULTS = {f"{sec}_{key}": default for sec, key, _, default in _CONTRACT}
_SECTIONS = {sec for sec, _, _, _ in _CONTRACT}
_SECTION_KEYS = {sec: {key for s, key, _, _ in _CONTRACT if s == sec} for sec in _SECTIONS}

# Nilai yang harus > 0; yang lain hanya wajib ada.
_POSITIVE = frozenset({
    "camera.width",
    "camera.height",
    "camera.fps",
    "landmark.max_num_hands",
    "window.frame_count",
    "window.stride",
    "queue.max_size",
    "pipeline.stats_window",
    "pipeline.stop_timeout_seconds",
    "tts.rate",
})
_NON_NEGATIVE = frozenset({"camera.device_index"})


@dataclass(frozen=True)
class AppConfig:
    """Snapshot konfigurasi bertipe. Tidak dimutasi setelah load."""

    camera_device_index: int
    camera_width: int
    camera_height: int
    camera_fps: int
    landmark_max_num_hands: int
    landmark_model_complexity: int
    landmark_hand_model_path: str
    landmark_pose_model_path: str
    window_frame_count: int
    window_stride: int
    smoothing_confidence_threshold: float
    smoothing_vote_count: int
    smoothing_cooldown_seconds: float
    queue_max_size: int
    pipeline_stats_window: int
    pipeline_stop_timeout_seconds: float
    tts_device_name: str
    tts_rate: int
    virtual_camera_backend: str


def resolve_config_path() -> str:
    """Path config: env ``ISYARATKU_CONFIG`` menang atas lokasi default."""
    return os.environ.get(ENV_CONFIG_PATH) or DEFAULT_CONFIG_PATH


def load_config(path: str | None = None) -> AppConfig:
    """Muat config dari berkas TOML; berkas tak ada berarti semua default.

    Fungsi murni dari ``path`` (dan env untuk resolusi default): tidak menyimpan
    state global, tidak mengubah berkas.
    """
    file_path = Path(path) if path is not None else Path(resolve_config_path())
    values = dict(_DEFAULTS)
    if file_path.is_file():
        with file_path.open("rb") as handle:
            raw = tomllib.load(handle)
        _merge(raw, values)
    _check_present(values)
    return AppConfig(**values)


def _merge(raw: dict[str, Any], values: dict[str, Any]) -> None:
    for section, section_data in raw.items():
        if not isinstance(section_data, dict) or section not in _SECTIONS:
            _warn(section)
            continue
        known = _SECTION_KEYS[section]
        for key, value in section_data.items():
            if key not in known:
                _warn(f"{section}.{key}")
                continue
            dotted = f"{section}.{key}"
            values[f"{section}_{key}"] = _check_value(dotted, _KINDS[dotted], value)


def _check_value(dotted: str, kind: str, value: Any) -> Any:
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"Config '{dotted}' harus int, dapat {type(value).__name__}: {value!r}")
        _check_range(dotted, value)
        return value
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"Config '{dotted}' harus float, dapat {type(value).__name__}: {value!r}")
        _check_range(dotted, float(value))
        return float(value)
    if not isinstance(value, str):
        raise ConfigError(f"Config '{dotted}' harus str, dapat {type(value).__name__}: {value!r}")
    return value


def _check_range(dotted: str, value: float) -> None:
    if dotted in _POSITIVE and value <= 0:
        raise ConfigError(f"Config '{dotted}' harus lebih besar dari 0, dapat {value}")
    if dotted in _NON_NEGATIVE and value < 0:
        raise ConfigError(f"Config '{dotted}' tidak boleh negatif, dapat {value}")


def _check_present(values: dict[str, Any]) -> None:
    for field, value in values.items():
        if value is None:
            dotted = _dotted(field)
            raise ConfigError(f"Config '{dotted}' wajib ada tapi tidak memiliki nilai")


def _dotted(field: str) -> str:
    for dotted in CONTRACT_KEYS:
        if dotted.replace(".", "_") == field:
            return dotted
    return field


def _warn(key: str) -> None:
    print(f"Peringatan: key config tak dikenal '{key}' diabaikan.", file=sys.stderr)
