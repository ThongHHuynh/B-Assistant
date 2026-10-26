"""Audio I/O: microphone capture (16 kHz mono, 80 ms frames) and interruptible playback."""
from __future__ import annotations

import logging
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)


class AudioError(RuntimeError):
    pass


def _sd():
    try:
        import sounddevice as sd
        return sd
    except OSError as e:   # PortAudio library missing
        raise AudioError("PortAudio is not installed. Linux: sudo apt install libportaudio2. "
                         "macOS: brew install portaudio. Windows: reinstall sounddevice.") from e


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear-interpolation resampler; good enough for speech and chimes."""
    if sr_in == sr_out or len(x) == 0:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    t_in = np.arange(len(x)) / sr_in
    t_out = np.arange(n_out) / sr_out
    y = np.interp(t_out, t_in, x.astype(np.float32))
    return y.astype(x.dtype) if x.dtype != np.int16 else np.clip(y, -32768, 32767).astype(np.int16)


def to_int16(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.int16:
        return x
    return np.clip(np.asarray(x, dtype=np.float32) * 32767, -32768, 32767).astype(np.int16)


def rms(frame: np.ndarray) -> float:
    if len(frame) == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return data, sr


def write_wav(path: str | Path, pcm: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(to_int16(pcm).tobytes())


def _pick_device(sd, spec, kind: str):
    if spec is None or spec == "":
        return None
    if isinstance(spec, int) or str(spec).isdigit():
        return int(spec)
    for i, d in enumerate(sd.query_devices()):
        ch = d["max_input_channels"] if kind == "input" else d["max_output_channels"]
        if ch > 0 and str(spec).lower() in d["name"].lower():
            return i
    raise AudioError(f"No {kind} device matching '{spec}'. Run python -m bi.doctor to list devices.")


# ------------------------------------------------------------------ capture --
class MicStream:
    """Non-blocking microphone capture. Frames (int16, 16 kHz, frame_ms) arrive on self.frames."""

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 80, device=None):
        self.sr = sample_rate
        self.frame_len = sample_rate * frame_ms // 1000
        self.frame_ms = frame_ms
        self.device = device
        self.frames: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
        self._stream = None
        self._native_sr = sample_rate
        self.muted = False
        self.overflows = 0

    def start(self) -> None:
        sd = _sd()
        dev = _pick_device(sd, self.device, "input")
        try:
            info = sd.query_devices(dev, "input")
        except Exception as e:  # noqa: BLE001
            raise AudioError("No microphone found. Check that a mic is connected and that this app has "
                             "microphone permission (macOS: System Settings > Privacy > Microphone).") from e
        for rate in (self.sr, int(info["default_samplerate"])):
            try:
                block = rate * self.frame_ms // 1000
                self._stream = sd.InputStream(samplerate=rate, channels=1, dtype="int16", blocksize=block,
                                              device=dev, callback=self._callback, latency="low")
                self._native_sr = rate
                break
            except Exception as e:  # noqa: BLE001
                log.info("mic at %d Hz unavailable (%s)", rate, e)
        if self._stream is None:
            raise AudioError(f"Couldn't open the microphone '{info['name']}'.")
        self._stream.start()
        log.info("mic '%s' open at %d Hz (resampled to %d)", info["name"], self._native_sr, self.sr)

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.overflows += 1
        if self.muted:
            return
        x = indata[:, 0].copy()
        if self._native_sr != self.sr:
            x = resample(x, self._native_sr, self.sr)
            if len(x) != self.frame_len:
                x = np.pad(x, (0, max(0, self.frame_len - len(x))))[: self.frame_len]
        try:
            self.frames.put_nowait(x)
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put_nowait(x)

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self) -> None:
        while not self.frames.empty():
            try:
                self.frames.get_nowait()
            except queue.Empty:
                break

    def set_muted(self, muted: bool) -> None:
        """Muting stops the input stream so the OS mic indicator turns off."""
        self.muted = muted
        if self._stream is None:
            return
        if muted:
            self._stream.stop()
            self.clear()
        else:
            self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class WavFrameSource:
    """Feeds a WAV file as mic frames (tests/bench). realtime=True paces frames like a real mic."""

    def __init__(self, path: str | Path, frame_ms: int = 80, realtime: bool = False, tail_silence_s: float = 0.0):
        pcm, sr = read_wav(path)
        self.sr = 16000
        pcm = resample(pcm, sr, self.sr)
        pcm = np.concatenate([pcm, np.zeros(int(tail_silence_s * self.sr), np.int16)])
        self.frame_len = self.sr * frame_ms // 1000
        self.frame_ms = frame_ms
        n = len(pcm) // self.frame_len
        self._frames = [pcm[i * self.frame_len:(i + 1) * self.frame_len] for i in range(n)]
        self.realtime = realtime
        self.muted = False
        self._i = 0

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        if self._i >= len(self._frames):
            return None
        if self.realtime:
            due = self._t0 + (self._i + 1) * self.frame_ms / 1000
            time.sleep(max(0.0, due - time.perf_counter()))
        f = self._frames[self._i]
        self._i += 1
        return f

    def clear(self) -> None:
        pass

    def set_muted(self, muted: bool) -> None:
        self.muted = muted

    def stop(self) -> None:
        pass


# ----------------------------------------------------------------- playback --
class Player:
    """Plays queued PCM on a background thread. stop() silences output within ~30 ms (barge-in)."""

    BLOCK_MS = 30

    def __init__(self, device=None, null: bool = False):
        self._q: "queue.Queue[tuple[np.ndarray, int, int, str | None]]" = queue.Queue()
        self.heard: list[str] = []               # tags of items that started playing this utterance
        self._stop = threading.Event()
        self._busy = threading.Event()
        self._closed = False
        self._gen = 0
        self.started_at: float | None = None     # when the current utterance began playing
        self.on_start: Callable[[], None] | None = None
        self.device = device
        self.null = null
        self.out_sr = 24000
        self._stream = None
        if not null:
            try:
                sd = _sd()
                dev = _pick_device(sd, device, "output")
                info = sd.query_devices(dev, "output")
                self.out_sr = int(info["default_samplerate"])
                self._stream = sd.OutputStream(samplerate=self.out_sr, channels=1, dtype="int16", device=dev,
                                               latency="low")
                self._stream.start()
            except Exception as e:  # noqa: BLE001
                log.warning("no audio output device (%s); playback will be silent", e)
                self.null = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="player")
        self._thread.start()

    def play(self, pcm: np.ndarray, sr: int, tag: str | None = None) -> None:
        self._busy.set()
        self._q.put((to_int16(pcm), sr, self._gen, tag))

    def begin_utterance(self) -> None:
        self.started_at = None
        self.heard = []

    def stop(self) -> None:
        self._gen += 1
        self._stop.set()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self.started_at = None

    @property
    def busy(self) -> bool:
        return self._busy.is_set() or not self._q.empty()

    def wait_idle(self, timeout: float | None = None) -> bool:
        end = None if timeout is None else time.monotonic() + timeout
        while self.busy:
            if end is not None and time.monotonic() > end:
                return False
            time.sleep(0.01)
        return True

    def _run(self) -> None:
        while not self._closed:
            try:
                pcm, sr, gen, tag = self._q.get(timeout=0.1)
            except queue.Empty:
                self._busy.clear()
                continue
            if gen != self._gen:
                continue
            self._stop.clear()
            self._busy.set()
            pcm = resample(pcm, sr, self.out_sr)
            if tag:
                self.heard.append(tag)
            if self.started_at is None:
                self.started_at = time.perf_counter()
                if self.on_start:
                    try:
                        self.on_start()
                    except Exception:  # noqa: BLE001
                        log.exception("on_start callback failed")
            block = self.out_sr * self.BLOCK_MS // 1000
            t0 = time.perf_counter()
            for i in range(0, len(pcm), block):
                if self._stop.is_set() or gen != self._gen:
                    break
                chunk = pcm[i:i + block]
                if self.null:
                    due = t0 + (i + len(chunk)) / self.out_sr
                    time.sleep(max(0.0, due - time.perf_counter()))
                else:
                    try:
                        self._stream.write(chunk.reshape(-1, 1))
                    except Exception as e:  # noqa: BLE001
                        log.warning("audio write failed: %s", e)
                        break
            if self._q.empty():
                self._busy.clear()

    def close(self) -> None:
        self.stop()
        self._closed = True
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass


def make_chime(kind: str = "wake", sr: int = 24000, volume: float = 0.25) -> np.ndarray:
    """Two soft sine notes. 'wake' rises, 'sleep' falls, 'error' is low."""
    notes = {"wake": (880.0, 1318.5), "sleep": (1318.5, 880.0), "error": (330.0, 262.0), "ok": (1046.5, 1046.5)}[kind]
    out = []
    for f in notes:
        t = np.arange(int(0.09 * sr)) / sr
        env = np.minimum(1, t / 0.008) * np.exp(-t * 18)
        tone = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)
        out.append(tone * env)
    x = np.concatenate(out) / 1.25 * volume
    return to_int16(x)
