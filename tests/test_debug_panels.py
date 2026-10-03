"""Tes panel debug: urutan tiga prediksi teratas dan passthrough probe.

Deterministik, tanpa MediaPipe, tanpa model asli: predictor disuntik sebagai
objek tiruan yang mengembalikan `Prediction` tetap.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from src.core.config import AppConfig, load_config
from src.core.pipeline import Frame
from src.core.predictor import Prediction
from src.ui.debug_view import DebugView
from src.ui.prediction_probe import (
    RANKED_DISPLAY_COUNT,
    PredictionProbe,
    format_ranked,
)

NO_FILE = "berkas-yang-tidak-ada.toml"


def config() -> AppConfig:
    return load_config(NO_FILE)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app

class PredictorTiruan:
    """Predictor tetap: ranked yang sama untuk setiap panggilan predict()."""

    def __init__(self, ranked) -> None:
        self.ranked = tuple(ranked)
        self.labels = tuple(label for label, _ in self.ranked)

    def predict(self, features) -> Prediction:
        label, confidence = self.ranked[0]
        return Prediction(label=label, confidence=confidence, ranked=self.ranked)



def _frame(text: str = "") -> Frame:
    return Frame(np.zeros((4, 4, 3), dtype=np.uint8), 0.0, 0, text=text)


def test_format_ranked_shows_three_rows_with_confidence() -> None:
    ranked = (
        ("Sore", 0.42),
        ("Bagaimana", 0.21),
        ("Saya", 0.11),
        ("Air", 0.05),
    )
    text = format_ranked(ranked)
    assert text == "1. Sore 0.42 | 2. Bagaimana 0.21 | 3. Saya 0.11", text
    assert len(text.split(" | ")) == RANKED_DISPLAY_COUNT


def test_format_ranked_empty_is_a_dash() -> None:
    assert format_ranked(()) == "-"
    assert format_ranked([]) == "-"


def test_format_ranked_survives_long_label_and_missing_confidence() -> None:
    panjang = "A" * 300
    ranked = ((panjang, 0.9), ("Saya", None), ("", 0.1))
    text = format_ranked(ranked)
    rows = text.split(" | ")
    assert rows[0].startswith("1. " + panjang), "label panjang tetap utuh"
    assert rows[1] == "2. Saya", f"confidence None tampil tanpa angka: {rows}"
    assert len(rows) == 2, "label kosong dilewati, bukan jadi baris hantu"


def test_probe_is_a_transparent_passthrough() -> None:
    ranked = (("Sore", 0.42), ("Bagaimana", 0.21), ("Saya", 0.11))
    inner = PredictorTiruan(ranked)
    probe = PredictionProbe(inner)

    features = np.zeros((30, 456), dtype=np.float32)
    result = probe.predict(features)

    assert isinstance(result, Prediction), "pipeline harus tetap menerima Prediction"
    assert result.label == "Sore" and result.confidence == 0.42
    assert result.ranked == ranked, "ranked diteruskan apa adanya"
    assert probe.labels == inner.labels, "labels tidak diteruskan"
    assert probe.read_ranked() == ranked


def test_probe_propagates_predictor_errors_unchanged() -> None:
    class PredictorGagal:
        labels = ("Sore",)

        def predict(self, features):
            raise ValueError("model rusak")

    probe = PredictionProbe(PredictorGagal())
    with pytest.raises(ValueError, match="model rusak"):
        probe.predict(np.zeros((3, 3), dtype=np.float32))
    assert probe.read_ranked() == (), "gagal tidak boleh meninggalkan ranked lama"


def test_probe_keeps_only_the_last_ranked() -> None:
    class PredictorUrut:
        labels = ("Sore", "Saya")

        def __init__(self) -> None:
            self.antrian = (
                (("Sore", 0.42), ("Bagaimana", 0.21)),
                (("Saya", 0.99), ("Air", 0.5)),
            )
            self.panggilan = 0

        def predict(self, features):
            ranked = self.antrian[min(self.panggilan, 1)]
            self.panggilan += 1
            return Prediction(ranked[0][0], ranked[0][1], ranked)

    probe = PredictionProbe(PredictorUrut())
    probe.predict(None)
    probe.predict(None)
    assert probe.read_ranked() == (("Saya", 0.99), ("Air", 0.5)), (
        "probe harus menyimpan ranked TERAKHIR, bukan riwayat"
    )


def test_debug_ranked_panel_reads_probe_from_pipeline(qapp) -> None:
    debug = DebugView(config())
    assert debug._ranked.text() == "Tiga prediksi teratas: -", (
        "sebelum ada prediksi panel tidak boleh mengklaim angka"
    )

    ranked = (("Sore", 0.42), ("Bagaimana", 0.21), ("Saya", 0.11))
    inner = PredictorTiruan(ranked)

    class PipelinePalsu:

        def stop(self) -> None:
            return None

        def __init__(self, predictor) -> None:
            self.predictor = predictor

    probe = PredictionProbe(inner)
    probe.predict(None)
    debug._pipeline = PipelinePalsu(probe)
    debug._on_frame(_frame(""))
    assert debug._ranked.text() == (
        "Tiga prediksi teratas: "
        "1. Sore 0.42 | 2. Bagaimana 0.21 | 3. Saya 0.11"
    ), debug._ranked.text()

    # Pipeline tanpa probe (mis. DummyPredictor langsung) tidak boleh memecah panel.
    debug._pipeline = PipelinePalsu(inner)
    debug._on_frame(_frame(""))
    assert debug._ranked.text() == "Tiga prediksi teratas: -"
    debug.close()


def test_debug_ranked_panel_keeps_text_panel_working(qapp) -> None:
    """Panel teks lama tetap jalan: ranked tidak menimpa label frame."""
    debug = DebugView(config())
    debug._on_frame(_frame("MAKAN"))
    assert debug._predictions.text() == "Prediksi teratas: MAKAN"
    debug.close()
