"""Loop pengiriman frame: ambil dari queue, render, kirim ke sink.

Dulu loop ini tinggal di ``core/pipeline.py`` sebagai metode yang menulis
belasan field ``Pipeline`` langsung (``_sent``, ``_sent_at``,
``_record_label_latency``, ``_fail``, ``self.renderer``, ``self.sink``,
tiga callback). Akibatnya aturan yang sebenarnya sederhana — "kuras queue
sampai stop DAN kosong, teks placeholder hanya bila tidak ada label, satu
kegagalan kirim menghentikan semuanya" — hanya bisa diuji lewat pipeline
penuh dengan kamera dan thread.

``OutputLoop`` memegang aturannya; ``Pipeline`` tetap memegang penghitung,
lock, dan siklus hidup thread. Penghitungan dilaporkan lewat callback
``on_sent`` supaya urutan penguncian di pipeline tidak berubah: loop ini
TIDAK mengambil lock apa pun, pemanggil yang menguncinya.

Kenapa "kuras sampai kosong" penting: kalau loop berhenti begitu ``_stop``
diset, frame yang sudah ada di queue ikut hilang dan ``stop()`` akan join
thread yang tidak pernah mengirim sisa terakhirnya. Kondisi berhenti karena
itu gabungan, bukan salah satu.
"""

from __future__ import annotations

import logging
import queue
from collections.abc import Callable

from .data import Frame
from .stats import clock

logger = logging.getLogger(__name__)

#: Batas tunggu pengambilan frame. Sama konsepnya dengan jeda polling sisi
#: capture (20 Hz) tapi tidak dijadikan tunable: ``queue.get`` memblokir
#: sehingga loop tidak memutar CPU selama menunggu, jadi angka ini hanya
#: perlu cukup kecil agar ``stop()`` responsif.
OUTPUT_POLL_SECONDS = 0.05


class OutputLoop:
    """Kuras queue frame, render, kirim ke sink; lapor lewat callback.

    Pemanggil menyediakan ``should_stop`` (biasanya ``threading.Event.is_set``)
    dan ``is_empty`` (biasanya ``queue.empty``); keduanya dipisah dari objek
    ini supaya loop tidak perlu tahu apa pun soal thread selain kapan berhenti.
    """

    def __init__(
        self,
        frames: queue.Queue[Frame],
        *,
        renderer: Callable[[Frame], Frame],
        send: Callable[[Frame], None],
        placeholder_text: Callable[[], str],
        should_stop: Callable[[], bool],
        on_sent: Callable[[Frame], None],
        on_frame: Callable[[Frame], None] | None = None,
        on_stats: Callable[[], None] | None = None,
        on_landmarks: Callable[[object | None], None] | None = None,
        poll_seconds: float = OUTPUT_POLL_SECONDS,
    ) -> None:
        self._frames = frames
        self._renderer = renderer
        self._send = send
        # Teks placeholder dibaca dari callable, bukan disalin saat init:
        # ``Pipeline.text`` bisa diubah pemanggil setelah pipeline jalan,
        # dan salinan akan membekukan nilai lama.
        self._placeholder_text = placeholder_text
        self._should_stop = should_stop
        self._on_sent = on_sent
        self._on_frame = on_frame
        self._on_stats = on_stats
        self._on_landmarks = on_landmarks
        self._poll = poll_seconds

    def run(self, on_error: Callable[[Exception], None]) -> None:
        """Jalankan sampai stop DAN queue kosong, atau sampai satu kirim gagal.

        ``on_error`` dipanggil sekali lalu loop berhenti; pipeline yang
        memutuskan artinya (mencatat galat, menyetop thread lain).
        """
        while not (self._should_stop() and self._frames.empty()):
            try:
                frame = self._frames.get(timeout=self._poll)
            except queue.Empty:
                continue
            try:
                # Teks placeholder hanya berlaku bila predictor tidak
                # menghasilkan label; label predictor menang supaya overlay
                # memakai teks hasil inferensi.
                if not frame.text:
                    frame.text = self._placeholder_text()
                frame = self._renderer(frame)
                self._send(frame)
            except Exception as exc:
                on_error(exc)
                return
            self._on_sent(frame)
            if self._on_frame is not None:
                self._on_frame(frame)
            if self._on_stats is not None:
                self._on_stats()
            if self._on_landmarks is not None:
                self._on_landmarks(frame.landmarks)


def label_latency_sample(frame: Frame) -> float | None:
    """Selisih detik dari label stabil terbit sampai frame ini dikirim.

    ``None`` bila frame tidak membawa label — pemanggil yang memutuskan
    apakah sampel itu dicatat, sehingga loop kirim tidak perlu tahu soal
    statistik latensi.
    """
    if frame.label_emitted_at is None:
        return None
    return clock() - frame.label_emitted_at


if __name__ == "__main__":  # self-check: python -m src.core.output

    # Dijalankan sebagai modul (``-m``) supaya impor relatif di atas bekerja;
    # sebagai skrip lewat path, ``src/core/logging.py`` juga akan menutupi
    # ``logging`` milik Python.
    q: queue.Queue[Frame] = queue.Queue()
    terkirim: list[Frame] = []
    diterima: list[Frame] = []
    kirim_ke: list[int] = [0]

    def _kirim(frame: Frame) -> None:
        kirim_ke[0] += 1
        if kirim_ke[0] == 3:
            raise RuntimeError("sink mati")

    loop = OutputLoop(
        q,
        renderer=lambda f: f,
        send=_kirim,
        placeholder_text=lambda: "teks-placeholder",
        should_stop=lambda: True,  # stop sudah diset: tetap harus menguras
        on_sent=terkirim.append,
        on_frame=diterima.append,
    )

    for i in range(5):
        q.put(Frame(image=None, timestamp=float(i) * 0.1, index=i))

    galat: list[Exception] = []
    loop.run(galat.append)

    # Tiga frame pertama sampai ke sink; frame ke-3 melempar sehingga
    # frame ke-4 dan ke-5 TIDAK dikirim, tapi tetap keluar dari queue.
    assert len(terkirim) == 2, terkirim
    assert len(galat) == 1 and isinstance(galat[0], RuntimeError)
    assert len(diterima) == 2, "on_frame hanya untuk frame yang terkirim"
    assert all(f.text == "teks-placeholder" for f in terkirim), (
        "teks placeholder harus terisi saat predictor tidak memberi label"
    )
    print("output self-check OK")
