"""Logging setup with secret redaction. API keys never reach the log file."""
from __future__ import annotations

import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

_KEY_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)(\s*[=:]\s*)([^\s,'\"]+)"),
]
_SECRET_ENV = ("ANTHROPIC_API_KEY", "BRAVE_API_KEY", "PICOVOICE_ACCESS_KEY")


def redact(text: str) -> str:
    for env in _SECRET_ENV:
        val = os.environ.get(env)
        if val and len(val) > 6:
            text = text.replace(val, "[REDACTED]")
    text = _KEY_PATTERNS[0].sub("[REDACTED]", text)
    text = _KEY_PATTERNS[1].sub("[REDACTED]", text)
    text = _KEY_PATTERNS[2].sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        red = redact(msg)
        if red != msg:
            record.msg, record.args = red, ()
        return True


def setup_logging(log_file: Path | None, level: str = "INFO", console: bool = False) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    filt = RedactingFilter()
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.addFilter(filt)
        root.addHandler(fh)
    if console:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.addFilter(filt)
        sh.setLevel(logging.WARNING)
        root.addHandler(sh)
    for noisy in ("httpx", "httpcore", "anthropic", "urllib3", "faster_whisper", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
