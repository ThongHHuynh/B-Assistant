"""Health check: python -m bi.doctor [--download] [--online] [--mic]

Checks Python/OS, packages, audio devices, models, API keys and the database, and says how to fix
each problem. --download fetches missing models, --online verifies the API key / Ollama with a free call,
--mic records 3 seconds and reports the input level.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

OK, WARN, FAIL = "✔", "⚠", "✘"


class Report:
    def __init__(self):
        self.fails = 0
        self.warns = 0

    def line(self, status: str, what: str, detail: str = "", fix: str = "") -> None:
        if status == FAIL:
            self.fails += 1
        elif status == WARN:
            self.warns += 1
        print(f" {status} {what}" + (f": {detail}" if detail else ""))
        if fix and status != OK:
            print(f"     → {fix}")


def check_python(r: Report, cfg) -> None:
    v = sys.version_info
    os_ = cfg.os
    r.line(OK, "OS", f"{os_.system} {os_.release} ({os_.machine}); audio via {os_.audio_backend}; system voice {os_.system_tts}")
    if v < (3, 11):
        r.line(FAIL, "Python", sys.version.split()[0], "Python 3.11 or newer is required.")
    elif os_.is_linux and v >= (3, 13):
        r.line(WARN, "Python", f"{sys.version.split()[0]} - openwakeword needs tflite-runtime on Linux, which has no "
               "wheels for 3.13+", "Use a Python 3.11 venv on Linux.")
    else:
        r.line(OK, "Python", sys.version.split()[0] + (" (venv)" if sys.prefix != sys.base_prefix else " (not in a venv)"))


PKGS = [("anthropic", "LLM (Anthropic)"), ("ollama", "LLM (Ollama)"), ("yaml", "config"), ("numpy", "audio math"),
        ("sounddevice", "audio I/O"), ("soundfile", "audio files"), ("webrtcvad", "VAD"),
        ("faster_whisper", "speech-to-text"), ("openwakeword", "wake word"), ("onnxruntime", "ONNX models"),
        ("piper", "Piper TTS"), ("pyttsx3", "fallback TTS"), ("tokenizers", "embeddings"), ("sqlite_vec", "vector index"),
        ("httpx", "web/downloads"), ("pypdf", "PDF search"), ("docx", "DOCX search")]


def check_packages(r: Report) -> None:
    missing = []
    for mod, why in PKGS:
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{mod} ({why}): {type(e).__name__}")
    if missing:
        r.line(FAIL, "Packages", "; ".join(missing), "pip install -r requirements.txt (inside the venv)")
    else:
        r.line(OK, "Packages", f"all {len(PKGS)} importable")


def check_audio(r: Report, cfg, mic_test: bool) -> None:
    try:
        import sounddevice as sd
    except OSError:
        fix = {"linux": "sudo apt install libportaudio2", "macos": "brew install portaudio"}.get(cfg.os.system,
                                                                                               "reinstall sounddevice")
        r.line(FAIL, "PortAudio", "library not found", fix)
        return
    try:
        devs = sd.query_devices()
    except Exception as e:  # noqa: BLE001
        r.line(FAIL, "Audio devices", str(e))
        return
    ins = [(i, d) for i, d in enumerate(devs) if d["max_input_channels"] > 0]
    outs = [(i, d) for i, d in enumerate(devs) if d["max_output_channels"] > 0]
    if not ins:
        fix = ("System Settings > Privacy & Security > Microphone: allow your terminal" if cfg.os.is_macos else
               "Settings > Privacy > Microphone: allow desktop apps" if cfg.os.is_windows else
               "Check `arecord -l`; make sure PipeWire/PulseAudio is running")
        r.line(FAIL, "Microphone", "no input device found", fix)
    else:
        try:
            d = sd.query_devices(kind="input")
            r.line(OK, "Microphone", f"default '{d['name']}' ({int(d['default_samplerate'])} Hz); {len(ins)} input device(s)")
        except Exception:  # noqa: BLE001
            r.line(WARN, "Microphone", "no default input device", "Set audio.input_device in config.yaml")
        for i, dd in ins[:8]:
            print(f"       [{i}] {dd['name']}")
    if not outs:
        r.line(FAIL, "Speakers", "no output device found")
    else:
        try:
            d = sd.query_devices(kind="output")
            r.line(OK, "Speakers", f"default '{d['name']}'")
        except Exception:  # noqa: BLE001
            r.line(WARN, "Speakers", "no default output device", "Set audio.output_device in config.yaml")
    hp = cfg.get("audio.headphones_mode")
    r.line(OK if hp else WARN, "Echo handling", "headphones_mode on" if hp else
           "speaker mode (echo guard on)", "Headphones make barge-in much more reliable: audio.headphones_mode: true")
    if mic_test and ins:
        import numpy as np
        print("     recording 3 s - say something…")
        rec = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype="int16")
        sd.wait()
        lvl = float(np.sqrt(np.mean(rec.astype(np.float32) ** 2)))
        peak = int(np.abs(rec).max())
        st = OK if peak > 1500 else WARN
        r.line(st, "Mic level", f"rms {lvl:.0f}, peak {peak}", "Very quiet: raise the input volume in your OS settings")


def check_models(r: Report, cfg) -> None:
    from . import models
    r.line(OK if models.embedder_ready(cfg) else WARN, "Embedding model", "all-MiniLM-L6-v2 (ONNX)",
           "python -m bi.doctor --download (falls back to the lexical hash embedder until then)")
    r.line(OK if models.whisper_cached(cfg) else FAIL, "Whisper STT model", cfg.get("stt.model", "base.en"),
           "python -m bi.doctor --download")
    pp = models.piper_path(cfg)
    r.line(OK if pp.exists() else WARN, "Piper voice", str(pp.name),
           "python -m bi.doctor --download (pyttsx3/system voice is used until then)")
    ok, where = models.wake_ready(cfg)
    placeholder = "hey_jarvis" in str(cfg.get("wake.model"))
    r.line(OK if ok else FAIL, "Wake word model", where + ("  [placeholder: say 'Hey Jarvis']" if placeholder else ""),
           "python -m bi.doctor --download; custom 'Hey Bi': docs/WAKEWORD.md")


def check_llm(r: Report, cfg, online: bool) -> None:
    backend = cfg.get("llm.backend", "anthropic")
    if backend == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            r.line(FAIL, "Anthropic API key", "ANTHROPIC_API_KEY not set",
                   "cp .env.example .env and paste your key (console.anthropic.com), or use llm.backend: ollama")
        else:
            r.line(OK, "Anthropic API key", f"set ({len(key)} characters, never printed)")
            if online:
                try:
                    import anthropic
                    c = anthropic.Anthropic(timeout=10)
                    for role in ("chat_model", "think_model", "extractor_model"):
                        m = cfg.get(f"llm.anthropic.{role}")
                        c.models.retrieve(m)
                    r.line(OK, "Anthropic API", "key accepted; configured models available")
                except Exception as e:  # noqa: BLE001
                    r.line(FAIL, "Anthropic API", f"{type(e).__name__}: {str(e)[:120]}", "Check the key and model names")
    host = cfg.get("llm.ollama.host", "http://localhost:11434")
    if backend == "ollama" or online:
        try:
            import httpx
            tags = httpx.get(f"{host}/api/tags", timeout=3).json().get("models", [])
            names = {m["name"] for m in tags}
            want = {cfg.get("llm.ollama.chat_model"), cfg.get("llm.ollama.extractor_model")}
            missing = [w for w in want if w and w not in names and f"{w}:latest" not in names]
            r.line(WARN if missing else OK, "Ollama", f"running at {host}; {len(names)} model(s)" +
                   (f"; missing {missing}" if missing else ""), f"ollama pull {' '.join(missing)}" if missing else "")
        except Exception:  # noqa: BLE001
            r.line(FAIL if backend == "ollama" else WARN, "Ollama", f"not reachable at {host}",
                   "Install from ollama.com, then: ollama serve")


def check_storage(r: Report, cfg) -> None:
    db = cfg.path_of("paths.db", "data/bi.db")
    try:
        db.parent.mkdir(parents=True, exist_ok=True)
        import sqlite3
        import sqlite_vec
        c = sqlite3.connect(db)
        c.enable_load_extension(True)
        sqlite_vec.load(c)
        v = c.execute("select vec_version()").fetchone()[0]
        n = c.execute("select count(*) from sqlite_master where name='facts'").fetchone()[0]
        facts = c.execute("select count(*) from facts where superseded_by is null and archived=0").fetchone()[0] if n else 0
        c.close()
        r.line(OK, "Memory database", f"{db} (sqlite-vec {v}; {facts} active facts)")
    except Exception as e:  # noqa: BLE001
        r.line(WARN, "Memory database", f"{db}: {e}", "Check folder permissions; numpy search is used if sqlite-vec fails")


def check_tts(r: Report, cfg) -> None:
    from .tts import make_tts
    try:
        t = make_tts(cfg)
        t0 = time.perf_counter()
        list(t.synth("Hello."))
        r.line(OK if t.name == "piper" else WARN, "TTS", f"{t.name} works ({(time.perf_counter() - t0) * 1000:.0f} ms)",
               "Download the Piper voice for natural speech: python -m bi.doctor --download")
    except Exception as e:  # noqa: BLE001
        r.line(FAIL, "TTS", str(e), "Linux: sudo apt install espeak-ng; others: python -m bi.doctor --download")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m bi.doctor", description="Check Bi's setup")
    p.add_argument("--config")
    p.add_argument("--download", action="store_true", help="download missing models")
    p.add_argument("--online", action="store_true", help="verify API key / Ollama with a free request")
    p.add_argument("--mic", action="store_true", help="record 3 s and report the mic level")
    a = p.parse_args(argv)
    from .config import ConfigError, load_config
    from .logs import setup_logging
    try:
        cfg = load_config(a.config)
    except ConfigError as e:
        print(f" {FAIL} config.yaml: {e}")
        return 1
    setup_logging(cfg.path_of("paths.log_file", "data/logs/bi.log"))
    print(f"Bi doctor — config: {cfg.path}\n")
    if a.download:
        from .models import download_all
        print("Downloading models (one-time, needs internet):")
        download_all(cfg)
        print()
    r = Report()
    check_python(r, cfg)
    check_packages(r)
    check_audio(r, cfg, a.mic)
    check_models(r, cfg)
    check_tts(r, cfg)
    check_llm(r, cfg, a.online)
    check_storage(r, cfg)
    print(f"\n{r.fails} problem(s), {r.warns} warning(s).")
    if r.fails == 0:
        print("Ready: python main.py   (or python main.py --text to try without audio)")
    return 1 if r.fails else 0


if __name__ == "__main__":
    sys.exit(main())
