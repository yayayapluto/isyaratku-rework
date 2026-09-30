"""Fixtures bersama. GUI jalan offscreen supaya konstruksi view tidak butuh layar."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _offscreen_qt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
