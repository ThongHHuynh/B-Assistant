"""Generate VAD/STT test fixtures (16 kHz mono WAV) from synthesized speech.

Real recordings are better: drop your own 16 kHz mono WAVs into tests/fixtures/ with the same names.
Requires espeak-ng (Linux: apt install espeak-ng, macOS: brew install espeak-ng).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bi.audio import read_wav, resample, write_wav  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
SR = 16000
rng = np.random.default_rng(7)


def speech(text: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        subprocess.run(["espeak-ng", "-s", "165", "-a", "120", "-w", f.name, text], check=True)
        pcm, sr = read_wav(f.name)
    return resample(pcm, sr, SR)


def noise(seconds: float, level: float = 60) -> np.ndarray:
    return (rng.normal(0, level, int(seconds * SR))).astype(np.int16)


def silence(seconds: float) -> np.ndarray:
    return noise(seconds, 25)          # a quiet room is never digital zero


def mix(*parts) -> np.ndarray:
    x = np.concatenate(parts).astype(np.float32)
    x += rng.normal(0, 25, len(x))     # constant room noise floor
    return np.clip(x, -32768, 32767).astype(np.int16)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    q = speech("What's the weather going to be like in Lisbon tomorrow?")
    write_wav(OUT / "utterance.wav", mix(silence(0.6), q, silence(2.0)), SR)
    # two phrases with a 500 ms pause in between: must NOT end early (pause < 900 ms)
    a, b = speech("Set a timer"), speech("for ten minutes please")
    write_wav(OUT / "pause_mid_sentence.wav", mix(silence(0.5), a, silence(0.5), b, silence(1.8)), SR)
    write_wav(OUT / "silence.wav", mix(silence(6.0)), SR)
    write_wav(OUT / "fan_noise.wav", mix(noise(6.0, 90)), SR)
    long = speech("This is a very long monologue that keeps going and going without any real pause. " * 3)
    write_wav(OUT / "long_speech.wav", mix(silence(0.3), long, silence(1.0)), SR)
    write_wav(OUT / "stop.wav", mix(silence(0.3), speech("stop"), silence(1.5)), SR)
    # wake word placeholder ("hey jarvis") followed by a question, as a user would say it
    write_wav(OUT / "wake_then_question.wav",
              mix(silence(0.8), speech("hey jarvis"), silence(0.4), speech("what time is my dentist appointment"),
                  silence(2.0)), SR)
    for p in sorted(OUT.glob("*.wav")):
        pcm, _ = read_wav(p)
        print(f"{p.name:24} {len(pcm) / SR:5.2f} s")


if __name__ == "__main__":
    main()
