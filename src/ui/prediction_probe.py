"""Pembaca hasil predictor mode debug: kunci urutan ranked terbaru.

``Pipeline._run_predictor`` hanya menyimpan label hasil smoothing, jadi
``Prediction.ranked`` yang dihitung predictor tidak pernah sampai ke UI.
Karena pipeline tidak boleh diubah untuk panel debug, predictor dibungkus di
sini: panel debug membaca ranked lewat bungkus ini, bukan lewat pipeline.
"""

from __future__ import annotations

from typing import Protocol

from ..core.predictor import Prediction

#: Jumlah pasangan label/keyakinan yang ditampilkan panel "tiga prediksi".
RANKED_DISPLAY_COUNT = 3


class _Predicts(Protocol):
    """Kontrak minimum predictor yang bisa dibungkus di sini."""

    @property
    def labels(self) -> tuple[str, ...]: ...

    def predict(self, features) -> Prediction: ...


class PredictionProbe:
    """Bungkus predictor: panel debug bisa membaca urutan ranked terbaru.

    ``predict()`` memanggil predictor asli dan mengembalikan hasilnya apa
    adanya — pipeline menerima objek ``Prediction`` yang sama, termasuk galat
    yang sama. Ranked dari panggilan terakhir dikunci di sini (satu slot,
    bukan riwayat), lalu dibaca panel lewat ``read_ranked``.
    """

    def __init__(self, inner: _Predicts) -> None:
        self._inner = inner
        self._ranked: tuple[tuple[str, float], ...] = ()

    @property
    def labels(self) -> tuple[str, ...]:
        """Daftar label predictor asli; dipakai warm_up TTS."""
        return self._inner.labels

    def predict(self, features) -> Prediction:
        """Prediksi apa adanya, plus kunci ranked untuk panel debug."""
        prediction = self._inner.predict(features)
        self._ranked = tuple(prediction.ranked)
        return prediction

    def read_ranked(self) -> tuple[tuple[str, float], ...]:
        """Ranked dari ``predict()`` terakhir; ``()`` bila belum ada."""
        return self._ranked


def format_ranked(ranked, count: int = RANKED_DISPLAY_COUNT) -> str:
    """Satu baris ``1. Label 0.42 | 2. Label 0.21``; ``-`` bila kosong.

    Confidence yang bukan angka (None, teks) tetap tampil sebagai label
    tanpa angka: panel debug tidak boleh mati karena data cacat.
    """
    rows = []
    for label, confidence in list(ranked)[:count]:
        if not label:
            continue
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            rows.append(f"{label} {float(confidence):.2f}")
        else:
            rows.append(f"{label}")
    if not rows:
        return "-"
    return " | ".join(
        f"{index}. {text}" for index, text in enumerate(rows, start=1)
    )
