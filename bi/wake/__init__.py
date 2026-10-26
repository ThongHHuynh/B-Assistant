"""Wake word detection.

openWakeWord ships no "Hey Bi" model, so the default is the built-in "hey_jarvis" placeholder.
Point wake.model at a custom models/hey_bi.onnx once trained - see docs/WAKEWORD.md.
Picovoice Porcupine is supported as an alternative engine (wake.engine: porcupine).
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class WakeError(RuntimeError):
    pass


class OpenWakeWordDetector:
    def __init__(self, model: str, threshold: float = 0.5, cooldown_s: float = 1.5, root: Path | None = None):
        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError as e:
            raise WakeError("openwakeword is not installed. Run: pip install -r requirements.txt") from e
        p = Path(os.path.expanduser(model))
        if not p.is_absolute() and root is not None and (root / p).exists():
            p = root / p
        is_path = p.suffix in (".onnx", ".tflite")
        if is_path and not p.exists():
            raise WakeError(f"Wake word model not found: {p}. See docs/WAKEWORD.md.")
        res = Path(openwakeword.__file__).parent / "resources" / "models"
        needed = ["melspectrogram.onnx", "embedding_model.onnx"]
        if not all((res / n).exists() for n in needed) or (not is_path and not list(res.glob(f"{model}*.onnx"))):
            try:
                from openwakeword.utils import download_models
                download_models([] if is_path else [model])
            except Exception as e:  # noqa: BLE001
                raise WakeError(f"openWakeWord models are missing and couldn't be downloaded ({e}). "
                                "Connect to the internet and run: python -m bi.doctor --download") from e
        self.name = p.stem if is_path else model
        self.model = Model(wakeword_models=[str(p) if is_path else model], inference_framework="onnx")
        self.threshold = threshold
        self.cooldown = cooldown_s
        self._last = 0.0
        self.last_score = 0.0

    def process(self, frame: np.ndarray) -> bool:
        scores = self.model.predict(frame)
        self.last_score = max(scores.values()) if scores else 0.0
        now = time.monotonic()
        if self.last_score >= self.threshold and now - self._last > self.cooldown:
            self._last = now
            self.model.reset()
            return True
        return False

    def reset(self) -> None:
        self.model.reset()


class PorcupineDetector:
    def __init__(self, keyword_path: str = "", sensitivity: float = 0.5, cooldown_s: float = 1.5):
        try:
            import pvporcupine
        except ImportError as e:
            raise WakeError("Porcupine engine selected but pvporcupine isn't installed: "
                            "pip install -r requirements-optional.txt") from e
        key = os.environ.get("PICOVOICE_ACCESS_KEY")
        if not key:
            raise WakeError("PICOVOICE_ACCESS_KEY is not set in .env (free key at console.picovoice.ai).")
        if keyword_path:
            self.pp = pvporcupine.create(access_key=key, keyword_paths=[keyword_path], sensitivities=[sensitivity])
            self.name = Path(keyword_path).stem
        else:
            self.pp = pvporcupine.create(access_key=key, keywords=["jarvis"], sensitivities=[sensitivity])
            self.name = "jarvis"
        self.buf = np.zeros(0, dtype=np.int16)
        self.cooldown = cooldown_s
        self._last = 0.0
        self.last_score = 0.0

    def process(self, frame: np.ndarray) -> bool:
        self.buf = np.concatenate([self.buf, frame])
        n = self.pp.frame_length
        hit = False
        while len(self.buf) >= n:
            chunk, self.buf = self.buf[:n], self.buf[n:]
            if self.pp.process(chunk.tolist()) >= 0 and time.monotonic() - self._last > self.cooldown:
                self._last = time.monotonic()
                hit = True
        self.last_score = 1.0 if hit else 0.0
        return hit

    def reset(self) -> None:
        self.buf = np.zeros(0, dtype=np.int16)


def make_wake(cfg):
    w = cfg.section("wake")
    engine = (w.get("engine") or "openwakeword").lower()
    if engine == "porcupine":
        return PorcupineDetector(w.get("porcupine_keyword_path", ""), float(w.get("threshold", 0.5)),
                                 float(w.get("cooldown_s", 1.5)))
    return OpenWakeWordDetector(str(w.get("model", "hey_jarvis")), float(w.get("threshold", 0.5)),
                                float(w.get("cooldown_s", 1.5)), cfg.root)
