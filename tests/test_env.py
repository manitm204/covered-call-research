"""Env loading: precedence, placeholders, and no-crash behavior."""

import os

from xsp_research.env import load_dotenv, thetadata_api_key, thetadata_base_url


def _clean(monkeypatch):
    for k in ("THETADATA_API_KEY", "THETADATA_BASE_URL"):
        monkeypatch.delenv(k, raising=False)


def test_missing_file_is_fine(tmp_path, monkeypatch):
    _clean(monkeypatch)
    assert load_dotenv(tmp_path / "nope.env") == {}
    assert thetadata_api_key(tmp_path / "nope.env") is None


def test_loads_values_and_strips_quotes(tmp_path, monkeypatch):
    _clean(monkeypatch)
    env = tmp_path / ".env"
    env.write_text('# comment\nTHETADATA_API_KEY="abc123"\nTHETADATA_BASE_URL=http://x:1/\n')
    assert thetadata_api_key(env) == "abc123"
    assert thetadata_base_url(env) == "http://x:1"


def test_real_environment_wins(tmp_path, monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("THETADATA_API_KEY", "from-real-env")
    env = tmp_path / ".env"
    env.write_text("THETADATA_API_KEY=from-file\n")
    assert thetadata_api_key(env) == "from-real-env"


def test_placeholder_treated_as_unset(tmp_path, monkeypatch):
    _clean(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("THETADATA_API_KEY=put-your-key-here\n")
    assert thetadata_api_key(env) is None


def test_never_writes_key_material_anywhere(tmp_path, monkeypatch):
    """Loader only touches os.environ; nothing is echoed or persisted."""
    _clean(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("THETADATA_API_KEY=secret\n")
    loaded = load_dotenv(env)
    assert loaded == {"THETADATA_API_KEY": "secret"}
    assert os.environ["THETADATA_API_KEY"] == "secret"
