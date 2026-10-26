"""Configuration loading (config.yaml + .env) and OS detection."""
from __future__ import annotations

import copy
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class OSInfo:
    system: str          # "windows" | "macos" | "linux" | other
    release: str
    machine: str
    python: str

    @property
    def is_windows(self) -> bool:
        return self.system == "windows"

    @property
    def is_macos(self) -> bool:
        return self.system == "macos"

    @property
    def is_linux(self) -> bool:
        return self.system == "linux"

    @property
    def audio_backend(self) -> str:
        return {"windows": "WASAPI/MME (PortAudio)", "macos": "CoreAudio (PortAudio)",
                "linux": "ALSA/PulseAudio/PipeWire (PortAudio)"}.get(self.system, "PortAudio")

    @property
    def system_tts(self) -> str:
        return {"windows": "SAPI5", "macos": "NSSpeechSynthesizer / say", "linux": "espeak-ng"}.get(
            self.system, "unknown")


def detect_os() -> OSInfo:
    raw = platform.system().lower()
    system = {"darwin": "macos", "windows": "windows", "linux": "linux"}.get(raw, raw)
    return OSInfo(system=system, release=platform.release(), machine=platform.machine(),
                  python=sys.version.split()[0])


class Config:
    """Thin wrapper around the YAML dict with dotted-path access."""

    def __init__(self, data: dict, path: Path | None = None):
        self._data = data
        self.path = path
        self.root = path.parent if path else PROJECT_ROOT
        self.os = detect_os()

    # dotted access: cfg.get("llm.anthropic.chat_model")
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def section(self, dotted: str) -> dict:
        val = self.get(dotted, {})
        return val if isinstance(val, dict) else {}

    def path_of(self, dotted: str, default: str) -> Path:
        p = Path(os.path.expanduser(str(self.get(dotted, default))))
        return p if p.is_absolute() else (self.root / p)

    @property
    def data(self) -> dict:
        return self._data

    def copy(self) -> "Config":
        c = Config(copy.deepcopy(self._data), self.path)
        return c


def load_env(root: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    env = root / ".env"
    if env.exists():
        load_dotenv(env, override=False)


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Config:
    p = Path(path) if path else Path(os.environ.get("BI_CONFIG", DEFAULT_CONFIG))
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}. Copy config.yaml from the project root.")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"config.yaml is not valid YAML: {e}") from e
    cfg = Config(data, p.resolve())
    load_env(cfg.root)
    for k, v in (overrides or {}).items():
        cfg.set(k, v)
    return cfg
