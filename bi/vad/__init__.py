"""Voice activity detection: end-of-speech endpointing and barge-in detection (webrtcvad)."""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass

import numpy as np

from ..audio import rms


class FrameVAD:
    """Classifies one 80 ms frame (16 kHz int16) as speech: webrtcvad on 20 ms sub-frames + energy gate."""

    SUB_MS = 20

    def __init__(self, aggressiveness: int = 2, speech_ratio: float = 0.5, min_rms: float = 120.0,
                 sample_rate: int = 16000):
        import webrtcvad
        self.vad = webrtcvad.Vad(int(aggressiveness))
        self.ratio = speech_ratio
        self.min_rms = min_rms
        self.sr = sample_rate
        self.sub = sample_rate * self.SUB_MS // 1000

    def is_speech(self, frame: np.ndarray, level: float | None = None) -> bool:
        level = rms(frame) if level is None else level
        if level < self.min_rms:
            return False
        n = len(frame) // self.sub
        if n == 0:
            return False
        voiced = sum(self.vad.is_speech(frame[i * self.sub:(i + 1) * self.sub].tobytes(), self.sr) for i in range(n))
        return voiced / n >= self.ratio


@dataclass
class EndpointConfig:
    frame_ms: int = 80
    onset_ms: int = 160
    trailing_silence_ms: int = 900
    max_utterance_s: float = 15.0
    listen_timeout_s: float = 5.0
    pre_roll_ms: int = 300

    @classmethod
    def from_cfg(cls, cfg) -> "EndpointConfig":
        v = cfg.section("vad")
        return cls(frame_ms=int(cfg.get("audio.frame_ms", 80)), onset_ms=int(v.get("onset_ms", 160)),
                   trailing_silence_ms=int(v.get("trailing_silence_ms", 900)),
                   max_utterance_s=float(v.get("max_utterance_s", 15)),
                   listen_timeout_s=float(v.get("listen_timeout_s", 5)), pre_roll_ms=int(v.get("pre_roll_ms", 300)))


class Endpointer:
    """Feed frames; returns events:
        "start"    speech onset confirmed (onset_ms of sustained speech)
        "end"      trailing_silence_ms of silence after speech  -> utterance complete
        "max"      max_utterance_s reached                      -> utterance complete
        "timeout"  no speech within listen_timeout_s
        None       keep going
    audio() returns the utterance including pre-roll.
    """

    def __init__(self, vad: FrameVAD, ec: EndpointConfig, listen_timeout_s: float | None = None):
        self.vad = vad
        self.ec = ec
        self.f_onset = max(1, math.ceil(ec.onset_ms / ec.frame_ms))
        self.f_trail = max(1, math.ceil(ec.trailing_silence_ms / ec.frame_ms))
        self.f_max = max(1, int(ec.max_utterance_s * 1000 / ec.frame_ms))
        timeout = ec.listen_timeout_s if listen_timeout_s is None else listen_timeout_s
        self.f_timeout = max(1, int(timeout * 1000 / ec.frame_ms))
        self.pre = collections.deque(maxlen=max(1, math.ceil(ec.pre_roll_ms / ec.frame_ms)) + self.f_onset)
        self.reset()

    def reset(self) -> None:
        self.speaking = False
        self.done = False
        self.frames: list[np.ndarray] = []
        self.n_seen = 0
        self.run_voiced = 0
        self.run_silence = 0
        self.pre.clear()
        self.speech_frames = 0

    def prime(self, frames: list[np.ndarray]) -> None:
        """Start already in speech (barge-in: the onset frames were captured while Bi was talking)."""
        self.reset()
        self.speaking = True
        self.frames = list(frames)
        self.speech_frames = len(frames)

    def feed(self, frame: np.ndarray) -> str | None:
        if self.done:
            return None
        self.n_seen += 1
        voiced = self.vad.is_speech(frame)
        if not self.speaking:
            self.pre.append(frame)
            self.run_voiced = self.run_voiced + 1 if voiced else 0
            if self.run_voiced >= self.f_onset:
                self.speaking = True
                self.frames = list(self.pre)
                self.speech_frames = self.run_voiced
                self.run_silence = 0
                return "start"
            if self.n_seen >= self.f_timeout:
                self.done = True
                return "timeout"
            return None
        self.frames.append(frame)
        if voiced:
            self.run_silence = 0
            self.speech_frames += 1
        else:
            self.run_silence += 1
        if self.run_silence >= self.f_trail:
            self.done = True
            return "end"
        if len(self.frames) >= self.f_max:
            self.done = True
            return "max"
        return None

    def audio(self, trim_trailing: bool = True) -> np.ndarray:
        frames = self.frames
        if trim_trailing and self.run_silence > 2:
            frames = frames[: len(frames) - (self.run_silence - 2)]
        return np.concatenate(frames) if frames else np.zeros(0, np.int16)


class BargeInDetector:
    """Detects the user talking over Bi.

    Laptop speakers leak into the mic, so in speaker mode:
      * nothing counts during the first echo_guard_ms of playback; those frames seed the echo level;
      * the echo level keeps adapting (EMA) on frames that aren't part of a candidate interruption;
      * an interruption needs barge_in_min_ms of consecutive VAD speech whose MEAN level exceeds
        echo_level x echo_factor (speech energy swings ~6x frame to frame, so per-frame tests are brittle).
    headphones_mode drops the guard and the echo test.
    """

    def __init__(self, vad: FrameVAD, frame_ms: int = 80, min_ms: int = 240, echo_guard_ms: int = 300,
                 echo_factor: float = 2.5, headphones: bool = False):
        self.vad = vad
        self.frame_ms = frame_ms
        self.f_min = max(1, math.ceil(min_ms / frame_ms))
        self.guard_ms = 0 if headphones else echo_guard_ms
        self.factor = echo_factor
        self.headphones = headphones
        self.reset()

    def reset(self) -> None:
        self.guard_levels: list[float] = []
        self.echo: float | None = None
        self.captured: list[np.ndarray] = []
        self.levels: list[float] = []

    @property
    def run(self) -> int:
        return len(self.captured)

    def feed(self, frame: np.ndarray, playback_ms: float) -> bool:
        level = rms(frame)
        if playback_ms < self.guard_ms:
            self.guard_levels.append(level)
            return False
        if self.echo is None and not self.headphones:
            self.echo = float(np.mean(self.guard_levels)) if self.guard_levels else 0.0
        if self.vad.is_speech(frame, level):
            self.captured.append(frame)
            self.levels.append(level)
        else:
            self._absorb()
            self._ema(level)
            return False
        if len(self.captured) < self.f_min:
            return False
        if self.headphones:
            return True
        window = self.levels[-self.f_min:]
        if float(np.mean(window)) > max(self.echo * self.factor, self.vad.min_rms * 2):
            return True
        # speech-like but not loud enough: it's Bi's own echo; let it raise the echo estimate
        self._ema(self.levels[-self.f_min])
        self.captured, self.levels = self.captured[-(self.f_min - 1):], self.levels[-(self.f_min - 1):]
        return False

    def _ema(self, level: float, alpha: float = 0.1) -> None:
        if self.echo is not None:
            self.echo = (1 - alpha) * self.echo + alpha * level

    def _absorb(self) -> None:
        for lv in self.levels:
            self._ema(lv)
        self.captured, self.levels = [], []
