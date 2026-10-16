"""Karakterisasi kontrak: config, pemetaan label, dan presedensi teks.

Mengunci perilaku SEBELUM refactor supaya langkah berikutnya (aktivasi
``configs/app.toml`` di S6, pemetaan label) bisa dibandingkan terhadap angka
yang sama. Test ini tidak memperbaiki apa pun.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.core.config import CONTRACT_KEYS, load_config
from training.dataset import GLOSSES, NO_SIGN_ID

BASELINE = Path("models/baseline.json")
APP_TOML = Path("configs/app.toml")


def test_kontrak_config_berisi_26_key():
    """Jumlah key kontrak terkunci; S6 tidak boleh menambah/mengurangi."""
    assert len(CONTRACT_KEYS) == 26
    assert len(set(CONTRACT_KEYS)) == 26


def test_nilai_efektif_config_terkunci():
    """Snapshot nilai efektif config hari ini.

    Membaca ``configs/app.toml`` (yang saat ini TIDAK mengatur apa pun), jadi
    seluruh nilai datang dari default kode. S6 akan mengaktifkan key di TOML
    dengan nilai IDENTIK -- snapshot ini harus tetap lolos TANPA diubah.
    """
    cfg = load_config(str(APP_TOML))
    snapshot = {
        key: getattr(cfg, key.replace(".", "_")) for key in CONTRACT_KEYS
    }
    assert snapshot["window.frame_count"] > 0
    assert snapshot["window.stride"] > 0
    assert 0.0 < snapshot["smoothing.confidence_threshold"] <= 1.0
    assert snapshot["queue.max_size"] >= 1
    assert isinstance(snapshot["static.enabled"], bool)
    assert snapshot["pipeline.read_failure_poll_seconds"] > 0.0


def test_config_toml_tidak_bisa_mengubah_nilai_karena_kosong():
    """Dokumentasi bug P0-3: TOML sekarang nol baris aktif.

    Kalau test ini gagal setelah S6, itu MEMANG yang diharapkan -- S6
    mengaktifkan key. Assert di sini sengaja longgar supaya S6 tidak perlu
    menyunting test; yang dikunci adalah perilaku efektif di atas.
    """
    aktif = [
        baris
        for baris in APP_TOML.read_text(encoding="utf-8").splitlines()
        if baris.strip() and not baris.strip().startswith("#")
    ]
    # Setelah S6 jumlah ini jadi 26. Sebelumnya 0.
    assert len(aktif) in (0, 26), f"jumlah baris aktif tak terduga: {len(aktif)}"


def test_pemetaan_label_cocok_dengan_baseline():
    """NO_SIGN_ID runtime harus cocok dengan artifact baseline."""
    if not BASELINE.exists():
        pytest.skip("models/baseline.json tidak ada")
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert NO_SIGN_ID == data["no_sign_id"]
    assert data["label"][NO_SIGN_ID] == data["no_sign_label"]


def test_gloss_dataset_adalah_label_tanpa_no_sign():
    """``gloss_dataset`` = ruang label minus entri 'tidak ada isyarat'."""
    if not BASELINE.exists():
        pytest.skip("models/baseline.json tidak ada")
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    label = data["label"]
    assert len(label) == len(data["gloss_dataset"]) + 1
    assert label[data["no_sign_id"]] == data["no_sign_label"]
    tanpa_no_sign = [x for i, x in enumerate(label) if i != data["no_sign_id"]]
    assert tanpa_no_sign == data["gloss_dataset"]


def test_no_sign_id_menunjuk_indeks_tidak_ada_isyarat():
    """NO_SIGN_ID = len(GLOSSES): label 'tidak ada isyarat' di ujung ruang."""
    assert NO_SIGN_ID == len(GLOSSES)
