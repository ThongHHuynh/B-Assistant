import time

import numpy as np
import pytest

from bi.audio import Player
from bi.brain.agent import Brain
from bi.tts import Speaker
from bi.voice.loop import VoiceAssistant
from bi.voice.statemachine import State, StateMachine
from tests.conftest import ScriptedLLM
from tests.test_vad import frames

S = State


# ------------------------------------------------------------ pure transitions --
def test_happy_path_transitions():
    seen = []
    sm = StateMachine(on_change=lambda o, n, e: seen.append(n))
    for ev in ["wake", "speech_end", "first_audio", "done", "speech_start", "speech_end", "first_audio", "done",
               "timeout"]:
        assert sm.fire(ev), ev
    assert seen == [S.LISTENING, S.THINKING, S.SPEAKING, S.FOLLOW_UP, S.LISTENING, S.THINKING, S.SPEAKING,
                    S.FOLLOW_UP, S.IDLE]


def test_invalid_events_are_ignored():
    sm = StateMachine()
    assert not sm.fire("speech_end") and sm.state == S.IDLE
    assert not sm.fire("first_audio")
    sm.fire("wake")
    assert not sm.fire("speech_start") and sm.state == S.LISTENING


def test_barge_in_stop_sleep_mute():
    sm = StateMachine()
    sm.fire("wake"); sm.fire("speech_end"); sm.fire("first_audio")
    assert sm.fire("barge_in") and sm.state == S.LISTENING
    sm.fire("speech_end"); sm.fire("first_audio")
    assert sm.fire("stop") and sm.state == S.FOLLOW_UP
    assert sm.fire("sleep") and sm.state == S.IDLE
    assert sm.fire("mute") and sm.state == S.MUTED
    assert not sm.fire("wake") and not sm.fire("sleep")      # muted ignores everything but unmute
    assert sm.fire("unmute") and sm.state == S.IDLE
    sm.fire("wake")
    assert sm.fire("timeout") and sm.state == S.IDLE         # no speech after wake word


# ---------------------------------------------------------- voice loop (fakes) --
class FakeTTS:
    name = "fake"

    def synth(self, text):
        yield np.full(int(16000 * 0.4), 0, np.int16), 16000      # 0.4 s per sentence


class FakeSTT:
    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe(self, pcm, sr=16000):
        return self.texts.pop(0) if self.texts else ""


class FakeWake:
    name = "fake"
    last_score = 1.0

    def __init__(self):
        self.armed = False

    def process(self, frame):
        hit, self.armed = self.armed, False
        return hit


class FakeMic:
    def start(self): pass
    def clear(self): pass
    def set_muted(self, m): self.muted = m
    def stop(self): pass


SPEECH = frames("utterance.wav")
SILENCE = frames("silence.wav")
LOUD = [f for f in SPEECH if np.sqrt(np.mean(f.astype(np.float32) ** 2)) > 800]


@pytest.fixture
def make_va(cfg, store):
    made = []

    def _make(replies, texts, **cfg_over):
        cfg.set("vad.follow_up_s", 1.0)
        cfg.set("audio.chime", False)
        for k, v in cfg_over.items():
            cfg.set(k, v)
        brain = Brain(cfg, store, ScriptedLLM(replies))
        speaker = Speaker(FakeTTS(), Player(null=True))
        va = VoiceAssistant(cfg, brain, FakeMic(), speaker, FakeSTT(texts), FakeWake())
        made.append(speaker)
        return va

    yield _make
    for sp in made:
        sp.close()


def feed(va, frames_, until=None, timeout=10.0):
    """Feed frames (cycling silence afterwards) until `until(va)` or timeout."""
    end = time.monotonic() + timeout
    it = iter(frames_)
    while time.monotonic() < end:
        f = next(it, None)
        va.step(f if f is not None else SILENCE[0])
        if f is None:
            time.sleep(0.02)
        if until and until(va):
            return True
    return until is None


def states(va):
    return [n for _, _, n in va.sm.history]


def test_full_conversation_with_follow_up(make_va):
    va = make_va(["It's sunny. Around twenty degrees.", "You're welcome."], ["what's the weather", "thanks a lot"])
    va.wake.armed = True
    assert feed(va, SILENCE[:2] + SPEECH, until=lambda v: v.sm.state == S.FOLLOW_UP)
    assert states(va)[:4] == [S.LISTENING, S.THINKING, S.SPEAKING, S.FOLLOW_UP]
    m = va.history[-1]
    assert m.transcript == "what's the weather" and m.first_audio_ms is not None
    # follow-up: just talk, no wake word
    assert feed(va, SPEECH, until=lambda v: len(v.history) == 2 and v.sm.state == S.FOLLOW_UP)
    assert S.LISTENING in states(va)[4:]
    # silence through the follow-up window -> IDLE
    assert feed(va, [], until=lambda v: v.sm.state == S.IDLE, timeout=5)
    turns = va.brain.store.session_turns(va.brain.session_id)
    assert [t.text for t in turns if t.role == "user"] == ["what's the weather", "thanks a lot"]


def test_barge_in_stops_playback_and_listens(make_va):
    long = " ".join(f"This is sentence number {i}." for i in range(1, 9))
    va = make_va([long], ["tell me something long"], **{"audio.headphones_mode": True})
    va.wake.armed = True
    assert feed(va, SILENCE[:2] + SPEECH, until=lambda v: v.sm.state == S.SPEAKING)
    time.sleep(0.5)
    assert feed(va, LOUD * 3, until=lambda v: v.sm.state == S.LISTENING, timeout=5)
    assert va.cancel.is_set()
    assert va.speaker.player.wait_idle(0.1)               # silent within 100 ms
    heard = va.brain.history[-1]["content"]
    assert "interrupted" in heard and "sentence number 8" not in heard
    assert va.ep.speaking                                  # the interrupting speech is already being captured


def test_speaker_mode_echo_does_not_barge_in(make_va):
    va = make_va(["One. Two. Three."], ["count"])
    va.wake.armed = True
    assert feed(va, SILENCE[:2] + SPEECH, until=lambda v: v.sm.state == S.SPEAKING)
    echo = [(f * 0.3).astype(np.int16) for f in LOUD]     # Bi's own voice leaking into the mic
    assert feed(va, echo * 2, until=lambda v: v.sm.state == S.FOLLOW_UP, timeout=8)
    assert S.LISTENING not in states(va)[3:]


def test_stop_command_and_sleep(make_va):
    va = make_va([], ["stop", "go to sleep"])
    va.wake.armed = True
    assert feed(va, SILENCE[:2] + SPEECH, until=lambda v: v.sm.state == S.FOLLOW_UP)
    assert feed(va, SPEECH, until=lambda v: v.sm.state == S.IDLE)
    assert states(va)[-1] == S.IDLE and va.brain.llm.calls == []


def test_listen_timeout_returns_to_idle(make_va):
    va = make_va([], [])
    va.wake.armed = True
    va.ec.listen_timeout_s = 1.0
    assert feed(va, SILENCE * 2, until=lambda v: v.sm.state == S.IDLE and len(v.sm.history) >= 2)
    assert states(va) == [S.LISTENING, S.IDLE]


def test_alert_spoken_when_idle(make_va):
    va = make_va([], [])
    va.alert("Reminder: call mom.")
    t = time.monotonic()
    va.step(SILENCE[0])
    assert time.monotonic() - t >= 0.3                     # spoke it (0.4 s fake audio)


def test_push_to_talk_keys(make_va):
    va = make_va(["Hi."], ["hello"])
    va.ptt = True

    class Keys:
        def __init__(self): self.q = ["\n"]
        def get(self): return self.q.pop(0) if self.q else None

    va.keys = Keys()
    assert feed(va, SILENCE[:1] + SPEECH, until=lambda v: len(v.history) == 1)
    assert states(va)[:3] == [S.LISTENING, S.THINKING, S.SPEAKING]
