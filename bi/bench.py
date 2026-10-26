"""Latency benchmark: python -m bi.bench [--trials 3] [--no-llm]

Measures each stage of a voice turn with the real components installed on this machine:
  end of speech -> endpoint   (VAD on a recorded utterance, real time base)
  wake word inference          (per 80 ms frame)
  STT                          (faster-whisper on the same utterance; with/without speculation)
  retrieval                    (2,000 facts + 5,000 snippets, real embedder)
  LLM first token / sentence   (configured backend; costs a few tokens)
  TTS first audio              (configured engine)
Stages whose models/keys are missing are reported as skipped, never estimated.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "utterance.wav"


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


class Results:
    def __init__(self):
        self.rows: list[tuple[str, str]] = []
        self.values: dict[str, float] = {}

    def add(self, name: str, text: str, value: float | None = None, key: str | None = None) -> None:
        self.rows.append((name, text))
        if value is not None and key:
            self.values[key] = value
        print(f"  {name:<34} {text}", flush=True)


def bench_endpoint(cfg, r: Results):
    from .audio import WavFrameSource, rms
    from .vad import EndpointConfig, Endpointer, FrameVAD
    v = cfg.section("vad")
    vad = FrameVAD(int(v.get("aggressiveness", 2)), float(v.get("speech_ratio", 0.5)), float(v.get("min_rms", 120)))
    ep = Endpointer(vad, EndpointConfig.from_cfg(cfg))
    src = WavFrameSource(FIXTURE)
    frames = []
    while (f := src.read()) is not None:
        frames.append(f)
    loud = [i for i, f in enumerate(frames) if rms(f) > 400]
    speech_end_ms = (loud[-1] + 1) * 80
    t0 = time.perf_counter()
    end_ms = None
    for i, f in enumerate(frames):
        if ep.feed(f) in ("end", "max"):
            end_ms = (i + 1) * 80
            break
    per_frame = (time.perf_counter() - t0) * 1000 / (i + 1)
    delay = end_ms - speech_end_ms
    r.add("end of speech -> endpoint", f"{delay:.0f} ms  (trailing_silence_ms={v.get('trailing_silence_ms')}, "
          f"VAD {per_frame:.2f} ms/frame)", delay, "endpoint")
    return ep.audio()


def bench_wake(cfg, r: Results):
    try:
        from .wake import make_wake
        w = make_wake(cfg)
    except Exception as e:  # noqa: BLE001
        r.add("wake word inference", f"skipped ({str(e)[:70]})")
        return
    frame = np.zeros(1280, np.int16)
    for _ in range(5):
        w.process(frame)
    t = time.perf_counter()
    for _ in range(50):
        w.process(frame)
    r.add("wake word inference", f"{(time.perf_counter() - t) * 1000 / 50:.1f} ms per 80 ms frame ({w.name})")
    # "Hey Jarvis, what time is my dentist appointment?" -> when does Bi wake, and when does it know you're done?
    from .audio import WavFrameSource, rms
    from .vad import EndpointConfig, Endpointer, FrameVAD
    fixture = FIXTURE.with_name("wake_then_question.wav")
    if not fixture.exists():
        return
    src = WavFrameSource(fixture)
    frames = []
    while (f := src.read()) is not None:
        frames.append(f)
    w.reset()
    hit = next((i for i, f in enumerate(frames) if w.process(f)), None)
    if hit is None:
        r.add("wake -> end of speech", "wake word not detected in fixture")
        return
    v = cfg.section("vad")
    ep = Endpointer(FrameVAD(int(v.get("aggressiveness", 2)), float(v.get("speech_ratio", 0.5)),
                             float(v.get("min_rms", 120))), EndpointConfig.from_cfg(cfg))
    end = next((i for i, f in enumerate(frames[hit + 1:], hit + 1) if ep.feed(f) in ("end", "max")), None)
    loud = [i for i, f in enumerate(frames) if rms(f) > 400]
    if end is not None:
        r.add("wake -> end of speech detected", f"wake fired at {(hit + 1) * 80} ms; endpoint {(end + 1) * 80} ms, "
              f"{(end - loud[-1]) * 80} ms after you stopped talking")


def bench_stt(cfg, r: Results, audio: np.ndarray):
    try:
        from .stt import STT
        stt = STT(cfg)
    except Exception as e:  # noqa: BLE001
        r.add("STT", f"skipped ({str(e)[:90]})")
        return
    times, text = [], ""
    for _ in range(3):
        t = time.perf_counter()
        text = stt.transcribe(audio)
        times.append((time.perf_counter() - t) * 1000)
    med = statistics.median(times)
    r.add(f"STT ({stt.model_name}, {len(audio) / 16000:.1f} s audio)", f"{med:.0f} ms  \"{text[:50]}\"", med, "stt")
    spec_ms = int(cfg.get("stt.speculative_after_ms", 300))
    trail = int(cfg.get("vad.trailing_silence_ms", 900))
    hidden = max(0.0, med - max(0, trail - spec_ms))
    on_path = hidden if cfg.get("stt.speculative", True) else med
    r.add("  STT on critical path", f"{on_path:.0f} ms (speculative={'on' if cfg.get('stt.speculative', True) else 'off'};"
          f" starts {spec_ms} ms into the {trail} ms silence)", on_path, "stt_path")


def bench_retrieval(cfg, r: Results):
    from .memory.embed import make_embedder
    from .memory.retrieval import Retriever
    from .memory.store import MemoryStore
    emb = make_embedder(cfg)
    rng = np.random.default_rng(0)
    subjects = ["coffee", "running", "sister", "python", "guitar", "Lisbon", "dentist", "budget", "garden", "chess",
                "sushi", "meeting", "flight", "novel", "yoga", "piano", "cat", "mortgage", "camera", "hiking"]
    verbs = ["likes", "is learning", "talked about", "plans", "worries about", "prefers", "bought", "visited"]
    with tempfile.TemporaryDirectory() as d:
        store = MemoryStore(Path(d) / "bench.db", emb)
        facts = [f"User {rng.choice(verbs)} {rng.choice(subjects)} number {i}" for i in range(2000)]
        chunks = [f"User: tell me about {rng.choice(subjects)} {i}\nBi: here is something about {rng.choice(subjects)}"
                  for i in range(5000)]
        t = time.perf_counter()
        for start in range(0, len(facts), 256):
            batch = facts[start:start + 256]
            vecs = emb.embed(batch)
            with store.lock:
                for text, vec in zip(batch, vecs):
                    cur = store._x("INSERT INTO facts(text, category, created_at, updated_at, confidence) "
                                   "VALUES (?, 'other', ?, ?, 0.7)", (text, time.time(), time.time()))
                    store.index.upsert("fact", cur.lastrowid, vec)
        for start in range(0, len(chunks), 256):
            batch = chunks[start:start + 256]
            for text, vec in zip(batch, emb.embed(batch)):
                store.add_chunk("turns", "bench", text, vec=vec)
        build = time.perf_counter() - t
        ret = Retriever(store, cfg)
        queries = ["what coffee do I like", "when is my dentist appointment", "how is my running going",
                   "tell me about my sister", "what should I practice on the piano"]
        times = []
        for i in range(40):
            ret.prefetch(None, True)
            times.append(ret.retrieve(queries[i % len(queries)], None).elapsed_ms)
        store.close()
    p50, p95 = pct(times, 50), pct(times, 95)
    r.add("retrieval (2k facts, 5k snippets)", f"p50 {p50:.1f} ms, p95 {p95:.1f} ms  ({emb.name}, "
          f"{store.index.backend}; index built in {build:.1f} s)", p50, "retrieval")


def bench_llm(cfg, r: Results, trials: int):
    from .brain.llm import LLMError, TextDelta, make_llm
    from .brain.prompt import build_system
    from .brain.sentences import SentenceSplitter
    import datetime as dt
    try:
        llm = make_llm(cfg)
    except LLMError as e:
        r.add("LLM first token", f"skipped ({str(e)[:90]})")
        return
    system = build_system(cfg.get("assistant.name", "Bi"), dt.datetime.now().astimezone())
    firsts, sentences = [], []
    for i in range(trials):
        sp = SentenceSplitter(int(cfg.get("tts.first_chunk_min_chars", 24)))
        t = time.perf_counter()
        ft = fs = None
        try:
            for ev in llm.stream([{"role": "user", "content": ["Give me one tip for sleeping better.",
                                                               "What's a good name for a goldfish?",
                                                               "Why is the sky blue?"][i % 3]}], None, system):
                if isinstance(ev, TextDelta):
                    now = (time.perf_counter() - t) * 1000
                    ft = ft or now
                    if fs is None and sp.push(ev.text):
                        fs = now
        except LLMError as e:
            r.add("LLM first token", f"failed ({str(e)[:90]})")
            return
        firsts.append(ft or 0)
        sentences.append(fs or ft or 0)
    model = llm.model_for("chat")
    r.add(f"LLM first token ({model})", f"median {statistics.median(firsts):.0f} ms over {trials}",
          statistics.median(firsts), "llm_token")
    r.add("LLM first speakable chunk", f"median {statistics.median(sentences):.0f} ms", statistics.median(sentences),
          "llm_sentence")


def bench_tts(cfg, r: Results):
    from .tts import make_tts
    try:
        tts = make_tts(cfg)
    except Exception as e:  # noqa: BLE001
        r.add("TTS first audio", f"skipped ({e})")
        return
    list(tts.synth("Warm up."))
    times = []
    for s in ["Sure, here's one tip.", "It's about twenty degrees outside.", "Your timer is set."]:
        t = time.perf_counter()
        next(iter(tts.synth(s)))
        times.append((time.perf_counter() - t) * 1000)
    med = statistics.median(times)
    r.add(f"TTS first audio ({tts.name})", f"median {med:.0f} ms", med, "tts")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m bi.bench", description=__doc__.split("\n\n")[0])
    p.add_argument("--config")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--no-llm", action="store_true", help="skip the LLM stage (no tokens spent)")
    a = p.parse_args(argv)
    from .config import load_config
    from .logs import setup_logging
    cfg = load_config(a.config)
    setup_logging(cfg.path_of("paths.log_file", "data/logs/bi.log"))
    print(f"Bi latency benchmark on {cfg.os.system} ({cfg.os.machine}, Python {cfg.os.python})\n")
    r = Results()
    audio = bench_endpoint(cfg, r)
    bench_wake(cfg, r)
    bench_stt(cfg, r, audio)
    bench_retrieval(cfg, r)
    if not a.no_llm:
        bench_llm(cfg, r, a.trials)
    bench_tts(cfg, r)
    v = r.values
    print()
    after_endpoint = ["stt_path", "retrieval", "llm_sentence", "tts"]
    missing = [k for k in after_endpoint if k not in v]
    total = sum(v.get(k, 0) for k in after_endpoint)
    if missing:
        print(f"  end of speech -> first audio:  incomplete (not measured here: {', '.join(missing)})")
        print(f"    measured part: endpoint {v.get('endpoint', 0):.0f} ms + {total:.0f} ms")
    else:
        e2e = v["endpoint"] + total
        print(f"  end of speech -> first audio:  {e2e:.0f} ms  (endpoint {v['endpoint']:.0f} + STT {v['stt_path']:.0f}"
              f" + retrieval {v['retrieval']:.0f} + LLM first chunk {v['llm_sentence']:.0f} + TTS {v['tts']:.0f})")
        stages = {"endpoint (trailing silence)": v["endpoint"], "STT": v["stt_path"], "retrieval": v["retrieval"],
                  "LLM": v["llm_sentence"], "TTS": v["tts"]}
        worst = max(stages, key=stages.get)
        print(f"  biggest stage: {worst} ({stages[worst]:.0f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
