"""Minimal .env loading — no third-party dependency, no key material in code.

Precedence: real environment variables always win; .env only fills gaps. Keys
are never logged or echoed. The .env file itself is git-ignored; a committed
.env.example documents the expected fields.
"""

from __future__ import annotations

import os
from pathlib import Path

_PLACEHOLDERS = {"", "put-your-key-here"}


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Parse KEY=VALUE lines from `path` into os.environ (without overriding
    existing variables). Returns the keys it set. Missing file is fine."""
    p = Path(path)
    loaded: dict[str, str] = {}
    if not p.exists():
        return loaded
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def thetadata_api_key(dotenv_path: str | Path = ".env") -> str | None:
    """API key from the environment (loading .env if needed); None when unset
    or still the placeholder. Callers must produce a clear error, not crash."""
    load_dotenv(dotenv_path)
    key = os.environ.get("THETADATA_API_KEY", "").strip()
    return key if key not in _PLACEHOLDERS else None


def thetadata_base_url(dotenv_path: str | Path = ".env") -> str:
    load_dotenv(dotenv_path)
    return os.environ.get("THETADATA_BASE_URL", "http://127.0.0.1:25503").rstrip("/")
