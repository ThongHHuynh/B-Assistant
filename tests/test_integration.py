"""Hands-free loop with the real wake word model, real VAD and real Piper voice (skipped if missing).
STT and the LLM are stand-ins, so this checks the wiring and timing, not recognition quality."""
import time

import pytest

from bi.audio import Player
from bi.brain.agent import Brain
from bi.models import piper_path, wake_ready
from bi.voice.loop import VoiceAssistant
from bi.voice.statemachine import State
from tests.conftest import ScriptedLLM
from tests.test_statemachine import FakeMic, FakeSTT, SILENCE, feed_rt, states
from tests.test_vad import frames


def test_wake_to_spoken_answer_with_real_models(cfg, store):
    if not (wake_ready(cfg)[0] and piper_path(cfg).exists()):
        pytest.skip("wake word model / Piper voice not downloaded")
    from bi.tts import PiperTTS, Speaker
    from bi.wake import make_wake
    cfg.set("vad.follow_up_s", 1.0)
    speaker = Speaker(PiperTTS(piper_path(cfg)), Player(null=True))
    brain = Brain(cfg, store, ScriptedLLM(["Your dentist appointment is on Thursday at nine. Want a reminder?"]))
    va = VoiceAssistant(cfg, brain, FakeMic(), speaker, FakeSTT(["what time is my dentist appointment"]),
                        make_wake(cfg))
    try:
        feed_rt(va, frames("wake_then_question.wav"), until=lambda v: v.history, timeout=20)
        assert states(va)[:4] == [State.LISTENING, State.THINKING, State.SPEAKING, State.FOLLOW_UP]
        m = va.history[0]
        assert m.speculative_hit                       # transcript was ready when the endpoint fired
        assert m.first_audio_ms is not None and m.first_audio_ms < 500, m
        assert speaker.spoken == ["Your dentist appointment is on Thursday at nine.", "Want a reminder?"]
    finally:
        speaker.close()
