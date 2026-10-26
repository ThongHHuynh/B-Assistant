"""Model locations and one-shot downloads (python -m bi.doctor --download)."""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

PIPER_GITHUB = "https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-{slug}.tar.gz"
PIPER_URL = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/{family}/{code}/{name}/{quality}/"
             "{code}-{name}-{quality}{ext}?download=true")


def whisper_cached(cfg) -> bool:
    root = cfg.path_of("paths.models_dir", "models") / "whisper"
    name = cfg.get("stt.model", "base.en")
    return any(root.glob(f"models--*faster-whisper-{name}*/snapshots/*/model.bin"))


def piper_path(cfg) -> Path:
    from .tts import piper_voice_path
    return piper_voice_path(cfg)


def wake_ready(cfg) -> tuple[bool, str]:
    model = str(cfg.get("wake.model", "hey_jarvis"))
    if cfg.get("wake.engine", "openwakeword") == "porcupine":
        return True, "porcupine (keys checked separately)"
    try:
        import openwakeword
    except ImportError:
        return False, "openwakeword not installed"
    res = Path(openwakeword.__file__).parent / "resources" / "models"
    base_ok = (res / "melspectrogram.onnx").exists() and (res / "embedding_model.onnx").exists()
    if model.endswith((".onnx", ".tflite")):
        p = Path(model) if Path(model).is_absolute() else cfg.root / model
        return base_ok and p.exists(), str(p)
    return base_ok and bool(list(res.glob(f"{model}*.onnx"))), f"{model} in {res}"


def embedder_ready(cfg) -> bool:
    return (cfg.path_of("paths.models_dir", "models") / "all-MiniLM-L6-v2-onnx" / "model.onnx").exists()


def download_piper(cfg) -> Path:
    """HuggingFace (rhasspy/piper-voices) first; the older GitHub release tarball as a fallback."""
    voice = cfg.get("tts.piper_voice", "en_US-lessac-medium")
    dest = piper_path(cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        return _piper_from_hf(voice, dest)
    except Exception as e:  # noqa: BLE001
        log.warning("Piper voice from HuggingFace failed (%s); trying GitHub release", e)
        return _piper_from_github(voice, dest)


def _piper_from_github(voice: str, dest: Path) -> Path:
    import io
    import tarfile

    import httpx
    url = PIPER_GITHUB.format(slug=voice.replace("_", "-").lower())
    r = httpx.get(url, follow_redirects=True, timeout=300)
    r.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(r.content)) as tf:
        for m in tf.getmembers():
            src = tf.extractfile(m) if m.isfile() else None
            if src is None:
                continue
            if m.name.endswith(".onnx.json"):
                dest.with_suffix(".onnx.json").write_bytes(src.read())
            elif m.name.endswith(".onnx"):
                dest.write_bytes(src.read())
    if not dest.exists():
        raise RuntimeError(f"no .onnx voice in {url}")
    return dest


def _piper_from_hf(voice: str, dest: Path) -> Path:
    import httpx
    code, name, quality = voice.split("-", 2)
    for ext in (".onnx.json", ".onnx"):
        url = PIPER_URL.format(family=code.split("_")[0], code=code, name=name, quality=quality, ext=ext)
        target = dest.with_suffix(".onnx.json") if ext == ".onnx.json" else dest
        if target.exists():
            continue
        tmp = target.with_suffix(target.suffix + ".part")
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        tmp.replace(target)
    return dest


def download_all(cfg, report=print) -> bool:
    ok = True
    steps = [
        ("Embeddings (all-MiniLM-L6-v2, ONNX)", lambda: __import__("bi.memory.embed", fromlist=["x"]).ensure_onnx_minilm(
            cfg.get("memory.embeddings.onnx_url"), cfg.path_of("paths.models_dir", "models"))),
        (f"Whisper {cfg.get('stt.model', 'base.en')}", lambda: __import__("bi.stt", fromlist=["STT"]).STT(cfg)),
        (f"Piper voice {cfg.get('tts.piper_voice')}", lambda: download_piper(cfg)),
        (f"Wake word {cfg.get('wake.model')}", lambda: __import__("bi.wake", fromlist=["make_wake"]).make_wake(cfg)),
    ]
    for label, fn in steps:
        report(f"  downloading {label} …")
        try:
            fn()
            report(f"  ✔ {label}")
        except Exception as e:  # noqa: BLE001
            ok = False
            report(f"  ✘ {label}: {e}")
    return ok
