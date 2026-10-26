"""The hands-free voice loop.

Main thread: reads 80 ms mic frames and drives the state machine.
Worker thread: runs the brain for one turn and feeds sentences to the Speaker as they stream in.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..brain.agent import Action, Say
from ..vad import BargeInDetector, EndpointConfig, Endpointer, FrameVAD
from .statemachine import State, StateMachine
from .ui import KeyReader, StatusLine

log = logging.getLogger(__name__)


@dataclass
class TurnMetrics:
    speech_end_at: float = 0.0          # endpointer decided the user finished
    stt_ms: float = 0.0
    retrieval_ms: float = 0.0
    llm_first_token_ms: float | None = None
    first_audio_ms: float | None = None  # end of user speech -> first audio out
    transcript: str = ""
    extra: dict = field(default_factory=dict)


class VoiceAssistant:
    def __init__(self, cfg, brain, mic, speaker, stt, wake=None, ui: StatusLine | None = None,
                 keys: KeyReader | None = None, push_to_talk: bool = False):
        self.cfg = cfg
        self.brain = brain
        self.mic = mic
        self.speaker = speaker
        self.stt = stt
        self.wake = wake
        self.ptt = push_to_talk
        self.ui = ui or StatusLine(enabled=False)
        self.keys = keys
        a, v = cfg.section("audio"), cfg.section("vad")
        self.frame_ms = int(a.get("frame_ms", 80))
        self.vad = FrameVAD(int(v.get("aggressiveness", 2)), float(v.get("speech_ratio", 0.5)),
                            float(v.get("min_rms", 120)))
        self.ec = EndpointConfig.from_cfg(cfg)
        self.follow_up_s = float(v.get("follow_up_s", 6))
        self.headphones = bool(a.get("headphones_mode", False))
        self.barge_in_enabled = bool(a.get("barge_in", True))
        self.chime_on = bool(a.get("chime", True))
        self.barge = BargeInDetector(self.vad, self.frame_ms, int(a.get("barge_in_min_ms", 240)),
                                     int(a.get("echo_guard_ms", 300)), float(a.get("barge_in_echo_factor", 2.5)),
                                     self.headphones)
        self.sm = StateMachine(on_change=self._on_state)
        self.ep: Endpointer | None = None
        self.cancel = threading.Event()
        self.worker: threading.Thread | None = None
        self.worker_done = threading.Event()
        self.worker_done.set()
        self.actions: "queue.Queue[str]" = queue.Queue()
        self.alerts: "queue.Queue[str]" = queue.Queue()
        self.pending_action: str | None = None
        self.metrics = TurnMetrics()
        self.history: list[TurnMetrics] = []
        self.running = True
        self.ignore_until = 0.0                 # skip mic frames while our own chime plays
        self._reset_barge = threading.Event()
        self.speaker.player.on_start = self._on_first_audio

    # ------------------------------------------------------------- helpers --
    def _on_state(self, old: State, new: State, event: str) -> None:
        self.ui.set(new.value)
        if new == State.IDLE and old != State.MUTED:
            self.brain.conversation_paused()

    def _on_first_audio(self) -> None:
        self._reset_barge.set()                 # echo guard starts when playback actually starts
        if self.metrics.first_audio_ms is None and self.metrics.speech_end_at:
            self.metrics.first_audio_ms = (time.perf_counter() - self.metrics.speech_end_at) * 1000
        if self.sm.state == State.THINKING:
            self.sm.fire("first_audio")

    def _listen(self, timeout: float | None = None) -> None:
        self.ep = Endpointer(self.vad, self.ec, listen_timeout_s=timeout)

    def alert(self, message: str) -> None:
        """Reminders/timers from other threads. Spoken when Bi isn't mid-answer."""
        self.alerts.put(message)

    # ---------------------------------------------------------------- turn --
    def _start_turn(self, audio: np.ndarray) -> None:
        self.metrics = TurnMetrics(speech_end_at=time.perf_counter())
        self.brain.prefetch()                       # retrieval prefetch overlaps with STT
        t = time.perf_counter()
        text = self.stt.transcribe(audio)
        self.metrics.stt_ms = (time.perf_counter() - t) * 1000
        self.metrics.transcript = text
        if not text:
            log.info("no words recognized (%.0f ms of audio)", len(audio) / 16)
            self.sm.fire("empty")
            self._listen(self.follow_up_s)
            return
        self.ui.print(f"You: {text}")
        self.cancel = threading.Event()
        self.worker_done.clear()
        self.speaker.begin_utterance()
        self.barge.reset()
        self.worker = threading.Thread(target=self._run_brain, args=(text, self.cancel), daemon=True, name="brain")
        self.worker.start()

    def _run_brain(self, text: str, cancel: threading.Event) -> None:
        said = []
        try:
            for ev in self.brain.respond(text, cancel=cancel):
                if cancel.is_set():
                    break
                if isinstance(ev, Say):
                    said.append(ev.text)
                    self.speaker.say(ev.text)
                elif isinstance(ev, Action):
                    self.actions.put(ev.kind)
        except Exception as e:  # noqa: BLE001
            log.exception("brain failed")
            self.speaker.say("Sorry, something went wrong on my side.")
            said.append(f"(error: {e})")
        finally:
            s = self.brain.last_stats
            self.metrics.retrieval_ms = s.retrieval_ms
            self.metrics.llm_first_token_ms = s.first_token_ms
            if said and not cancel.is_set():
                self.ui.print(f"{self.brain.name}: {' '.join(said)}")
            self.worker_done.set()

    def _barge_in(self, captured: list[np.ndarray]) -> None:
        self.cancel.set()
        heard = self.speaker.spoken
        self.speaker.stop()
        self.brain.note_interrupted(heard)
        self.ui.print(f"{self.brain.name}: {' '.join(heard)} …(interrupted)")
        self.sm.fire("barge_in")
        self._listen()
        self.ep.prime(captured)

    # ---------------------------------------------------------------- step --
    def step(self, frame: np.ndarray | None) -> None:
        """Process one mic frame (or None when no audio arrived) plus any pending events."""
        self._handle_keys()
        st = self.sm.state
        if st == State.MUTED:
            return
        if st in (State.THINKING, State.SPEAKING):
            self._step_responding(frame)
            return
        if st in (State.IDLE, State.FOLLOW_UP) and not self.alerts.empty():
            self._speak_alert(self.alerts.get())
            return
        if frame is None:
            return
        if time.monotonic() < self.ignore_until:
            return
        if st == State.IDLE:
            if not self.ptt and self.wake is not None and self.wake.process(frame):
                self._on_wake()
        elif st == State.LISTENING:
            ev = self.ep.feed(frame)
            if ev in ("end", "max"):
                self.sm.fire("speech_end")
                self._start_turn(self.ep.audio())
            elif ev == "timeout":
                self.sm.fire("timeout")
        elif st == State.FOLLOW_UP:
            if not self.ptt and self.wake is not None and self.wake.process(frame):
                self._on_wake()
                return
            ev = self.ep.feed(frame)
            if ev == "start":
                self.sm.fire("speech_start")       # same endpointer keeps collecting
            elif ev == "timeout":
                self.sm.fire("timeout")
                if self.chime_on:
                    self.speaker.chime("sleep")

    def _chime_and_listen(self, event: str) -> None:
        self.speaker.stop()
        if self.chime_on:
            self.speaker.chime("wake")
            self.ignore_until = time.monotonic() + 0.25
        self.mic.clear()
        self.sm.fire(event)
        self._listen()

    def _on_wake(self) -> None:
        log.info("wake word (score %.2f)", getattr(self.wake, "last_score", 0))
        self._chime_and_listen("wake")

    def _step_responding(self, frame: np.ndarray | None) -> None:
        while not self.actions.empty():
            self._handle_action(self.actions.get())
            if self.sm.state not in (State.THINKING, State.SPEAKING):
                return
        if self._reset_barge.is_set():
            self._reset_barge.clear()
            self.barge.reset()
        if frame is not None and self.barge_in_enabled:
            started = self.speaker.player.started_at
            playback_ms = (time.perf_counter() - started) * 1000 if started else 10_000
            if self.barge.feed(frame, playback_ms):
                self._barge_in(list(self.barge.captured))
                return
        if self.worker_done.is_set() and not self.speaker.busy:
            action, self.pending_action = self.pending_action, None
            self.history.append(self.metrics)
            log.info("turn: stt %.0f ms, retrieval %.0f ms, first token %s ms, first audio %s ms",
                     self.metrics.stt_ms, self.metrics.retrieval_ms,
                     _r(self.metrics.llm_first_token_ms), _r(self.metrics.first_audio_ms))
            if action == "sleep":
                self.sm.fire("sleep")
            elif action == "mute":
                self._set_muted(True)
            else:
                self.sm.fire("done")
                self._listen(self.follow_up_s)

    def _handle_action(self, kind: str) -> None:
        if kind == "stop":
            self.speaker.stop()
            self.sm.fire("stop")
            self._listen(self.follow_up_s)
        elif kind == "cancel":
            self.speaker.stop()
            self.sm.fire("cancel")
        elif kind in ("sleep", "mute"):
            self.pending_action = kind          # let the short acknowledgement finish first

    def _speak_alert(self, message: str) -> None:
        self.ui.print(f"{self.brain.name}: ⏰ {message}")
        if self.chime_on:
            self.speaker.chime("ok")
        self.speaker.begin_utterance()
        self.speaker.say(message)
        self.speaker.wait_idle(30)
        self.mic.clear()

    def _set_muted(self, muted: bool) -> None:
        self.mic.set_muted(muted)
        if muted:
            self.cancel.set()
            self.speaker.stop()
            self.sm.fire("mute")
        else:
            self.sm.fire("unmute")
        self.ui.print("Microphone " + ("muted. Press m to unmute." if muted else "on."))

    def _handle_keys(self) -> None:
        if self.keys is None:
            return
        while (k := self.keys.get()) is not None:
            if k == "q":
                self.running = False
            elif k == "m":
                self._set_muted(self.sm.state != State.MUTED)
            elif k in ("\n", "\r", " ") and self.ptt:
                if self.sm.state in (State.THINKING, State.SPEAKING):
                    self.cancel.set()
                    self.brain.note_interrupted(self.speaker.spoken)
                    self.speaker.stop()
                    self.sm.fire("barge_in")
                    self._listen()
                elif self.sm.state == State.LISTENING and self.ep and self.ep.speaking:
                    self.sm.fire("speech_end")      # second Enter: done talking
                    self._start_turn(self.ep.audio(trim_trailing=False))
                elif self.sm.state in (State.IDLE, State.FOLLOW_UP):
                    self._chime_and_listen("ptt")

    # ----------------------------------------------------------------- run --
    def run(self, start_muted: bool = False) -> None:
        self.mic.start()
        self.ui.set("idle")
        if start_muted:
            self._set_muted(True)
        while self.running:
            frame = self.mic.read(timeout=0.1)
            self.step(frame)
            if self.ptt and self.sm.state == State.FOLLOW_UP:
                self.sm.fire("timeout")            # push-to-talk has no follow-up window


def _r(v):
    return None if v is None else round(v)


def run_voice(app, push_to_talk: bool = False, start_muted: bool = False) -> int:
    """Build the real audio components with clear error messages, then run until 'q' or Ctrl+C."""
    from ..audio import AudioError, MicStream
    from ..memory.consolidate import NightlyConsolidator
    from ..stt import STT, STTError
    from ..tts import TTSError, make_speaker
    from ..wake import WakeError, make_wake

    cfg = app.cfg
    print(f"Starting {app.brain.name} on {cfg.os.system} ({cfg.os.audio_backend})…", flush=True)
    if app.llm_error:
        print(f"  ! {app.llm_error}\n  ! Only memory commands will work until this is fixed.")
    try:
        speaker = make_speaker(cfg)
        print(f"  TTS: {speaker.tts.name}" + ("" if speaker.tts.name == "piper" else " (Piper voice missing - fallback)"))
        stt = STT(cfg)
        print(f"  STT: faster-whisper {stt.model_name} (loaded in {stt.load_ms:.0f} ms)")
        wake = None
        if not push_to_talk:
            wake = make_wake(cfg)
            print(f"  Wake word: '{wake.name}' (placeholder for 'Hey Bi' - see docs/WAKEWORD.md)"
                  if "jarvis" in wake.name else f"  Wake word: '{wake.name}'")
        mic = MicStream(int(cfg.get("audio.sample_rate", 16000)), int(cfg.get("audio.frame_ms", 80)),
                        cfg.get("audio.input_device"))
    except (AudioError, STTError, TTSError, WakeError) as e:
        print(f"\nCan't start voice mode: {e}\nTip: python -m bi.doctor checks everything; "
              "python main.py --text works without audio.")
        return 1
    if not cfg.get("audio.headphones_mode"):
        print("  Speaker mode: echo guard on. Headphones give the most reliable barge-in (--headphones).")
    ui = StatusLine(app.brain.name, bool(cfg.get("ui.status_line", True)), push_to_talk)
    keys = KeyReader() if cfg.get("ui.hotkeys", True) else None
    va = VoiceAssistant(cfg, app.brain, mic, speaker, stt, wake, ui, keys, push_to_talk)
    if app.tools is not None and hasattr(app.tools, "set_alert"):
        app.tools.set_alert(va.alert)
    nightly = NightlyConsolidator(app.store, app.llm, cfg)
    nightly.start()
    if keys:
        keys.start()
    try:
        va.run(start_muted=start_muted)
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001
        log.exception("voice loop crashed")
        print(f"\nVoice loop error: {e}")
        return 1
    finally:
        if keys:
            keys.restore()
        nightly.stop()
        mic.stop()
        speaker.close()
        print()
    return 0
