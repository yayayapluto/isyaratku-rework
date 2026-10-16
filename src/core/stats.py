"""Perhitungan statistik pipeline, tanpa threading dan tanpa queue.

Isinya murni angka: persentil nearest-rank dan peringkasan latensi. Dulu
keduanya tinggal di ``core/pipeline.py`` bercampur dengan thread, lock, dan
lifecycle, sehingga menguji rumusnya menuntut membangun pipeline penuh.
Sekarang bisa diuji langsung — ``__main__`` di bawah sudah melakukannya.

``Pipeline`` tetap pemilik deque-nya sendiri (butuh ``maxlen`` dan berada di
bawah lock yang sama dengan penghitung lain); modul ini hanya menyediakan
rumusnya.
"""

from __future__ import annotations

import math
import time

#: Ambang penskalaan ke milidetik untuk pelaporan latensi.
_MS_PER_SECOND = 1000.0


def clock() -> float:
    """Cap waktu sistem; hanya untuk jalur runtime, tes menyuntiknya."""
    return time.monotonic()


def percentile(ordered: list[float], fraction: float) -> float:
    """Persentil nearest-rank atas daftar yang SUDAH terurut.

    Sengaja bukan pustaka statistik: satu indeks bulat naik, tanpa
    interpolasi, jadi nilai yang diharapkan bisa dihitung tangan di tes
    dan tidak ada dependensi baru untuk dua angka. ``fraction`` 0.5 =
    p50, 0.95 = p95.
    """
    if not ordered:
        raise ValueError("percentile butuh daftar tidak kosong.")
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def latency_summary(samples: list[float]) -> dict[str, float]:
    """Ringkas sampel latensi detik menjadi p50/p95/max milidetik.

    Daftar kosong berarti belum ada sampel — kembaliannya ``count`` 0 saja,
    bukan angka nol palsu yang bisa disalahartikan sebagai latensi terukur.
    Daftar yang datang tidak harus terurut; fungsi ini yang mengurutkan.
    """
    if not samples:
        return {"count": 0}
    ordered = sorted(samples)
    return {
        "count": len(ordered),
        "p50_ms": percentile(ordered, 0.50) * _MS_PER_SECOND,
        "p95_ms": percentile(ordered, 0.95) * _MS_PER_SECOND,
        "max_ms": ordered[-1] * _MS_PER_SECOND,
    }


def fps_from_timestamps(sent_at: list[float]) -> tuple[float, float]:
    """FPS dan rentang waktu dari cap waktu frame yang sudah terkirim.

    FPS dihitung dari jumlah SELANG, bukan jumlah frame: n cap waktu punya
    n-1 selang, dan memakai n akan melebihkan nilainya. Kurang dari dua
    cap waktu berarti belum ada selang yang bisa diukur — kembaliannya
    (0,0), bukan angka dari pembagian nol.
    """
    if len(sent_at) < 2:
        return 0.0, 0.0
    span = sent_at[-1] - sent_at[0]
    if span <= 0:
        return 0.0, span
    return (len(sent_at) - 1) / span, span


if __name__ == "__main__":  # self-check: python src/core/stats.py
    # nearest-rank tanpa interpolasi: elemen ke-1, ke-2, ke-2 untuk 1..4
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.50) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
    assert percentile([7.0], 0.95) == 7.0
    try:
        percentile([], 0.5)
    except ValueError:
        pass
    else:  # pragma: no cover - jalur kegagalan
        raise AssertionError("percentile([]) harus melempar ValueError")

    assert latency_summary([]) == {"count": 0}
    ringkas = latency_summary([0.001, 0.002, 0.003, 0.004])
    assert ringkas["count"] == 4
    assert ringkas["p50_ms"] == 2.0
    assert ringkas["max_ms"] == 4.0

    # fps memakai n-1 selang: 5 cap waktu dalam 2 detik -> 2.0 fps
    fps, span = fps_from_timestamps([0.0, 0.5, 1.0, 1.5, 2.0])
    assert fps == 2.0, fps
    assert span == 2.0
    assert fps_from_timestamps([1.0]) == (0.0, 0.0)
    assert fps_from_timestamps([]) == (0.0, 0.0)
    assert fps_from_timestamps([1.0, 1.0]) == (0.0, 0.0)

    print("stats self-check OK")
