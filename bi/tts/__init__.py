"""Text-to-speech: Piper (local neural) with automatic fallback to pyttsx3, then the OS voice.

Speaker pipelines synthesis and playback: sentence N+1 is synthesized while sentence N plays,
and stop() cuts both instantly for barge-in.
"""
from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from ..audio import Player, make_chime, read_wav, to_int16

log = logging.getLogger(__name__)


class TTSError(RuntimeError):
    pass


class PiperTTS:
    name = "piper"

    def __init__(self, voice_path: Path, length_scale: float = 1.0):
        if not voice_path.exists():
            raise TTSError(f"Piper voice not found at {voice_path}. Run: python -m bi.doctor --download")
        from piper import PiperVoice
        from piper.config import SynthesisConfig
        self.voice = PiperVoice.load(str(voice_path))
        self.syn = SynthesisConfig(length_scale=length_scale)

    def synth(self, text: str) -> Iterator[tuple[np.ndarray, int]]:
        for chunk in self.voice.synthesize(text, syn_config=self.syn):
            yield chunk.audio_int16_array, chunk.sample_rate


def _read_any(path: str) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="int16", always_2d=True)
        return data.mean(axis=1).astype(np.int16), sr
    except Exception:  # noqa: BLE001
        return read_wav(path)


class Pyttsx3TTS:
    """pyttsx3 bound to one worker thread (its drivers are not thread-safe); renders to a temp file
    so playback goes through the same interruptible Player as Piper."""

    name = "pyttsx3"

    def __init__(self, rate: int = 185):
        import pyttsx3
        self._jobs: "queue.Queue" = queue.Queue()
        ready, err = threading.Event(), []

        def worker():
            try:
                eng = pyttsx3.init()
                eng.setProperty("rate", rate)
                try:    # the espeak driver prints "Audio saved to ..." on every file; keep the console clean
                    import pyttsx3.drivers.espeak as _esp
                    _esp.print = lambda *a, **k: None
                except Exception:  # noqa: BLE001
                    pass
            except Exception as e:  # noqa: BLE001
                err.append(e)
                ready.set()
                return
            ready.set()
            while True:
                job = self._jobs.get()
                if job is None:
                    break
                text, path, done = job
                try:
                    eng.save_to_file(text, path)
                    eng.runAndWait()
                except Exception as e:  # noqa: BLE001
                    log.warning("pyttsx3 failed: %s", e)
                done.set()

        threading.Thread(target=worker, daemon=True, name="pyttsx3").start()
        ready.wait(10)
        if err:
            raise TTSError(f"pyttsx3 could not start: {err[0]}")

    def synth(self, text: str) -> Iterator[tuple[np.ndarray, int]]:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        done = threading.Event()
        self._jobs.put((text, path, done))
        done.wait(30)
        try:
            if os.path.getsize(path) > 44:
                yield _read_any(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


class SystemTTS:
    """OS voice via its command-line tool: say (macOS), espeak-ng (Linux), SAPI via PowerShell (Windows)."""

    name = "system"

    def __init__(self, system: str):
        self.system = system
        if system == "macos" and shutil.which("say"):
            self.kind = "say"
        elif system == "windows" and shutil.which("powershell"):
            self.kind = "sapi"
        elif shutil.which("espeak-ng") or shutil.which("espeak"):
            self.kind = "espeak"
            self.exe = shutil.which("espeak-ng") or shutil.which("espeak")
        else:
            raise TTSError("No system TTS found (say / SAPI / espeak-ng).")

    def synth(self, text: str) -> Iterator[tuple[np.ndarray, int]]:
        suffix = ".aiff" if self.kind == "say" else ".wav"
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            if self.kind == "say":
                cmd = ["say", "-o", path, text]
            elif self.kind == "sapi":
                safe = text.replace("'", "''")
                cmd = ["powershell", "-NoProfile", "-Command",
                       "Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                       f"$s.SetOutputToWaveFile('{path}'); $s.Speak('{safe}'); $s.Dispose()"]
            else:
                cmd = [self.exe, "-w", path, text]
            subprocess.run(cmd, check=True, timeout=30, capture_output=True)
            yield _read_any(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


def piper_voice_path(cfg) -> Path:
    return cfg.path_of("paths.models_dir", "models") / "piper" / f"{cfg.get('tts.piper_voice', 'en_US-lessac-medium')}.onnx"


def make_tts(cfg):
    first = (cfg.get("tts.engine") or "piper").lower()
    order = [first] + [e for e in ("piper", "pyttsx3", "system") if e != first]
    errors = []
    for eng in order:
        try:
            if eng == "piper":
                return PiperTTS(piper_voice_path(cfg), float(cfg.get("tts.piper_length_scale", 1.0)))
            if eng == "pyttsx3":
                return Pyttsx3TTS(int(cfg.get("tts.pyttsx3_rate", 185)))
            if eng == "system":
                return SystemTTS(cfg.os.system)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{eng}: {e}")
            log.warning("TTS engine %s unavailable: %s", eng, e)
    raise TTSError("No TTS engine available: " + "; ".join(errors))


class Speaker:
    """Sentence queue -> synthesis thread -> Player. Thread-safe; stop() is instant."""

    def __init__(self, tts, player: Player, chime_volume: float = 0.25):
        self.tts = tts
        self.player = player
        self.chime_volume = chime_volume
        self._q: "queue.Queue[tuple[str, int] | None]" = queue.Queue()
        self._gen = 0
        self._lock = threading.Lock()            # makes "still current? -> enqueue" atomic with stop()
        self._synthesizing = threading.Event()
        self.first_audio_at: float | None = None
        self._utt_start: float | None = None
        self._t = threading.Thread(target=self._run, daemon=True, name="tts")
        self._t.start()

    @property
    def spoken(self) -> list[str]:
        """Sentences that actually started playing in the current utterance (what the user heard)."""
        return list(self.player.heard)

    def begin_utterance(self) -> None:
        self.first_audio_at = None
        self._utt_start = time.perf_counter()
        self.player.begin_utterance()

    def say(self, text: str) -> None:
        if self._utt_start is None:
            self.begin_utterance()
        self._q.put((text, self._gen))

    def chime(self, kind: str = "wake") -> None:
        self.player.play(make_chime(kind, 24000, self.chime_volume), 24000)

    def stop(self) -> None:
        with self._lock:
            self._gen += 1
            self.player.stop()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._utt_start = None

    @property
    def busy(self) -> bool:
        return not self._q.empty() or self._synthesizing.is_set() or self.player.busy

    def wait_idle(self, timeout: float | None = None) -> bool:
        end = None if timeout is None else time.monotonic() + timeout
        while self.busy:
            if end is not None and time.monotonic() > end:
                return False
            time.sleep(0.01)
        self._utt_start = None
        return True

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            text, gen = item
            if gen != self._gen:
                continue
            self._synthesizing.set()
            try:
                first = True
                for pcm, sr in self.tts.synth(text):
                    if gen != self._gen:
                        break
                    with self._lock:
                        if gen != self._gen:
                            break
                        if first and self.first_audio_at is None:
                            self.first_audio_at = time.perf_counter()
                        self.player.play(to_int16(pcm), sr, tag=text if first else None)
                    first = False
            except Exception as e:  # noqa: BLE001
                log.warning("TTS failed for %r: %s", text[:40], e)
            finally:
                self._synthesizing.clear()

    def close(self) -> None:
        self.stop()
        self._q.put(None)
        self.player.close()


def make_speaker(cfg, null_audio: bool = False) -> Speaker:
    player = Player(cfg.get("audio.output_device"), null=null_audio)
    return Speaker(make_tts(cfg), player, float(cfg.get("audio.chime_volume", 0.25)))
