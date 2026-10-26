"""Speech-to-text with faster-whisper, model kept loaded (int8 on CPU by default)."""
from __future__ import annotations

import logging
import os
import time

import numpy as np

log = logging.getLogger(__name__)

# Whisper's favourite hallucinations on near-silence
_HALLUCINATIONS = {"you", "thank you", "thank you.", "thanks for watching!", "thanks for watching", ".", "bye.",
                   "subtitles by the amara.org community", "you.", "i'm sorry."}


class STTError(RuntimeError):
    pass


class STT:
    def __init__(self, cfg):
        s = cfg.section("stt")
        self.model_name = s.get("model", "base.en")
        self.language = s.get("language", "en") or None
        self.beam_size = int(s.get("beam_size", 1))
        self.initial_prompt = s.get("initial_prompt") or None
        root = cfg.path_of("paths.models_dir", "models") / "whisper"
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise STTError("faster-whisper is not installed. Run: pip install -r requirements.txt") from e
        t0 = time.perf_counter()
        try:
            self.model = WhisperModel(self.model_name, device=s.get("device", "cpu"),
                                      compute_type=s.get("compute_type", "int8"), download_root=str(root),
                                      cpu_threads=min(4, os.cpu_count() or 4))
        except Exception as e:  # noqa: BLE001
            raise STTError(f"Whisper model '{self.model_name}' is missing and couldn't be downloaded ({e}). "
                           "Connect to the internet and run: python -m bi.doctor --download") from e
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.transcribe(np.zeros(16000, dtype=np.int16))       # warm-up
        log.info("whisper %s loaded in %.0f ms", self.model_name, self.load_ms)

    def transcribe(self, pcm: np.ndarray, sr: int = 16000) -> str:
        if len(pcm) < sr * 0.2:
            return ""
        audio = pcm.astype(np.float32) / 32768.0
        segments, _info = self.model.transcribe(audio, language=self.language, beam_size=self.beam_size,
                                                vad_filter=False, without_timestamps=True,
                                                condition_on_previous_text=False, initial_prompt=self.initial_prompt)
        parts = []
        for seg in segments:
            if seg.no_speech_prob > 0.6 and seg.avg_logprob < -0.8:
                continue
            parts.append(seg.text)
        text = " ".join(p.strip() for p in parts).strip()
        if text.lower() in _HALLUCINATIONS:
            return ""
        return text
