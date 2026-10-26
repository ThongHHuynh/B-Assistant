"""End-of-speech logic on recorded fixtures (tests/fixtures/*.wav, 16 kHz mono)."""
import numpy as np
import pytest

from bi.audio import WavFrameSource, rms
from bi.vad import BargeInDetector, EndpointConfig, Endpointer, FrameVAD
from tests.conftest import FIXTURES

def frames(name):
    src = WavFrameSource(FIXTURES / name)
    out = []
    while (f := src.read()) is not None:
        out.append(f)
    return out


EC = EndpointConfig(frame_ms=80, onset_ms=160, trailing_silence_ms=900, max_utterance_s=15, listen_timeout_s=5)


def run_endpointer(name, ec=EC, timeout=None):
    ep = Endpointer(FrameVAD(aggressiveness=2, speech_ratio=0.5, min_rms=120), ec, listen_timeout_s=timeout)
    events = []
    for i, f in enumerate(frames(name)):
        ev = ep.feed(f)
        if ev:
            events.append((ev, (i + 1) * 80))       # time at the end of the frame that fired
        if ep.done:
            break
    return events, ep


def speech_bounds(name):
    """Ground truth from the fixture: first/last 80 ms frame well above the noise floor."""
    levels = [rms(f) for f in frames(name)]
    loud = [i for i, l in enumerate(levels) if l > 400]
    return loud[0] * 80, (loud[-1] + 1) * 80


def test_utterance_ends_after_trailing_silence():
    events, ep = run_endpointer("utterance.wav")
    kinds = [e for e, _ in events]
    assert kinds == ["start", "end"]
    start_gt, end_gt = speech_bounds("utterance.wav")
    start_ms, end_ms = events[0][1], events[1][1]
    assert abs(start_ms - start_gt) <= 240                   # onset within 3 frames
    # end fires ~900 ms after speech stops (quantized to 80 ms frames, +/- VAD hangover)
    assert 800 <= end_ms - end_gt <= 1300, (end_ms, end_gt)
    audio = ep.audio()
    assert len(audio) / 16000 >= (end_gt - start_gt) / 1000  # whole utterance captured incl. pre-roll


def test_short_pause_does_not_end_utterance():
    events, ep = run_endpointer("pause_mid_sentence.wav")
    assert [e for e, _ in events] == ["start", "end"]
    _, end_gt = speech_bounds("pause_mid_sentence.wav")
    assert events[-1][1] > end_gt                            # ended after the second phrase, not in the gap


def test_silence_and_noise_time_out():
    for name in ("silence.wav", "fan_noise.wav"):
        events, _ = run_endpointer(name, timeout=5)
        assert [e for e, _ in events] == ["timeout"], name
        assert events[0][1] == pytest.approx(5000, abs=100)


def test_max_utterance_length():
    ec = EndpointConfig(frame_ms=80, onset_ms=160, trailing_silence_ms=900, max_utterance_s=6, listen_timeout_s=5)
    events, ep = run_endpointer("long_speech.wav", ec)
    assert [e for e, _ in events] == ["start", "max"]
    assert len(ep.audio(trim_trailing=False)) / 16000 == pytest.approx(6, abs=0.1)


def test_barge_in_requires_sustained_speech_above_echo():
    vad = FrameVAD(min_rms=120)
    speech = frames("utterance.wav")
    loud = [f for f in speech if rms(f) > 800]
    quiet_echo = [(f * 0.3).astype(np.int16) for f in loud]   # Bi's own voice leaking back, attenuated
    det = BargeInDetector(vad, min_ms=240, echo_guard_ms=300, echo_factor=2.5)
    t = 0
    for f in quiet_echo[:4]:                                  # guard window: measures echo, never triggers
        assert not det.feed(f, t)
        t += 80
    assert not any(det.feed(f, t + i * 80) for i, f in enumerate(quiet_echo[4:12]))
    hits = [det.feed(f, 1000 + i * 80) for i, f in enumerate(loud[:4])]  # the user talks over it, louder
    assert hits[-1] and len(det.captured) >= 3
    hp = BargeInDetector(vad, min_ms=240, headphones=True)   # headphones: no guard, no echo test
    assert any(hp.feed(f, i * 80) for i, f in enumerate(loud[:4]))


def test_wake_word_then_endpoint_on_fixture(cfg):
    """Real openWakeWord placeholder model ('hey jarvis'); skipped if the model isn't downloaded."""
    from bi.models import wake_ready
    if not wake_ready(cfg)[0]:
        pytest.skip("wake word model not downloaded (python -m bi.doctor --download)")
    from bi.wake import make_wake
    w = make_wake(cfg)
    fr = frames("wake_then_question.wav")
    hit = next((i for i, f in enumerate(fr) if w.process(f)), None)
    assert hit is not None and hit * 80 < 2500                 # fires during/just after "hey jarvis"
    ep = Endpointer(FrameVAD(), EC)
    end = next(i for i, f in enumerate(fr[hit + 1:], hit + 1) if ep.feed(f) in ("end", "max"))
    _, speech_end = speech_bounds("wake_then_question.wav")
    assert 800 <= (end + 1) * 80 - speech_end <= 1300
    # unrelated speech never wakes it
    w2 = make_wake(cfg)
    assert not any(w2.process(f) for f in frames("utterance.wav") + frames("long_speech.wav"))
