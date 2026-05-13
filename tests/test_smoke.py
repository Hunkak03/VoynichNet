"""Smoke tests: settings, tokenizer, CLI flags (no GPU training)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_settings_roundtrip(tmp_path, monkeypatch):
    import config as cfg

    monkeypatch.setattr(cfg, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    s = cfg.Settings()
    s.theme = "void"
    s.training_seed = None
    s.save()
    s2 = cfg.Settings.load()
    assert s2.theme == "void"
    assert s2.training_seed is None


def test_tokenizer_fit_roundtrip():
    from core.tokenizer import VoynichTokenizer

    t = VoynichTokenizer(mode="char")
    raw = "hello world"
    clean = t.limpiar_texto(raw)
    t.ajustar(clean)
    ids = t.codificar(clean)
    assert ids
    out = t.decodificar(ids)
    assert "hello" in out or "helloworld" in out.replace(" ", "")


def test_cli_main_no_args_returns_none(monkeypatch):
    import main as m

    monkeypatch.setattr(sys, "argv", ["main.py"])
    assert m._cli_main() is None


def test_cli_version(monkeypatch, capsys):
    import main as m

    monkeypatch.setattr(sys, "argv", ["main.py", "--version"])
    assert m._cli_main() == 0
    assert m.VOYNICH_VERSION in capsys.readouterr().out


def test_subprocess_smoke():
    r = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--smoke"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr + r.stdout


def test_print_settings_json(monkeypatch, capsys):
    import main as m

    monkeypatch.setattr(sys, "argv", ["main.py", "--print-settings"])
    assert m._cli_main() == 0
    data = json.loads(capsys.readouterr().out)
    assert "theme" in data
    assert "training_seed" in data
