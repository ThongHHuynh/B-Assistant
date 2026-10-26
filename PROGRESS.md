# Bi – build progress

Environment used for building: Linux cloud container (Ubuntu 24.04, Python 3.11 venv, 4 CPU, no mic/speakers).
Network policy there blocked huggingface.co, github.com and ollama.com, so Whisper / Piper / wake-word model
downloads and a live Ollama could not be exercised; those paths are written to work on your laptop and
flagged below as **not yet verified live**.

## Stage 1 – skeleton, config, `--text`, LLM interface, raw log ✅
- `config.yaml` holds every model name, threshold and path; `.env` holds keys (`.env.example` provided).
- `bi/brain/llm.py`: one interface `stream(messages, tools, system, role) -> TextDelta | ToolCall | Done`
  with `AnthropicLLM` (default) and `OllamaLLM`. Static persona is the first system block with
  `cache_control`; time + retrieved memory follow it. "think harder" routes a turn to `think_model`
  (with server-side refusal fallback enabled).
- History design: completed turns are replayed as text only; the provider's raw content (thinking
  blocks included) is replayed unchanged only inside one turn's tool loop. This keeps every request
  compatible with the API's preserved-thinking check even though the system prompt changes each turn.
- SQLite store `data/bi.db`: sessions, turns (raw log), facts, chunks, embeddings, notes, reminders, tool log.
- `python main.py --text` works; without a key it says exactly what to do and memory commands still work.
- Tests: brain (10) + LLM backends (5, real Anthropic SDK against a local mock SSE server).
- Measured: warm start of `--text` 1.7 s (was 8.5 s before switching the default embedder to ONNX, see below).

**Not verified live:** a real Anthropic call (no API key in the build container) and Ollama (blocked).

## Decisions
- Embeddings default to the ONNX build of all-MiniLM-L6-v2 (same weights as sentence-transformers):
  0.35 s load, ~2 ms per query embedding, no torch. `sentence-transformers` remains a selectable backend
  (`requirements-optional.txt`). A hash embedder is the last-resort fallback.
- Vector index: sqlite-vec (works on Win/macOS/Linux wheels); exact numpy search over the same SQLite
  table is the fallback instead of Chroma (no extra service, fine at personal scale).

## Stage 2 – memory: embeddings, retrieval, extractor, commands, import ✅
- Semantic index: MiniLM embeddings of facts and conversation snippets in sqlite-vec (numpy fallback).
- Retrieval per turn under a 100 ms budget, prefetching recent turns + core facts while STT finishes.
  Measured: ~4 ms per retrieval on a small DB (query embedding ~2 ms).
- Thresholds calibrated on real MiniLM scores: related query/fact pairs 0.18–0.65, unrelated ≤ 0.12,
  so `min_fact_score` 0.15 / `min_snippet_score` 0.20 / forget 0.25 (the first guess of 0.30 dropped half the hits).
- Background fact extractor (Haiku, JSON-schema output) every 6 turns and when a conversation pauses/ends;
  contradictions supersede old facts; secrets and sensitive categories are filtered unless you said "remember".
- Voice/text commands: "remember that…", "forget…" (confirms, then deletes the fact, its older versions,
  the source turn + Bi's reply, and every log line quoting it), "what do you know about me".
- `python -m bi.memory list|search|add|edit|delete|export|consolidate|stats|reindex|turns`.
- Consolidation (nightly in-process + on demand): merge duplicates, summarize old sessions, decay/archive.
- `python -m bi.import_history <file>`: ChatGPT / Claude JSON exports, generic JSON, markdown/text.
- Tests: 35 passing (memory insert/supersede/forget, ranking, both vector backends, extractor,
  privacy filters, consolidation, importers, semantic retrieval with the real MiniLM model).

**Not verified live:** extractor and session summaries against the real model (no API key here).

## Stage 3 – push-to-talk voice: STT + streaming sentence-by-sentence TTS ✅
## Stage 4 – hands-free: wake word, VAD, state machine, follow-up, chime, barge-in ✅
(Built together: push-to-talk is the same loop with Enter instead of the wake word.)
- Mic: sounddevice, 16 kHz mono, 80 ms frames, callback -> queue; opens at the device rate and resamples
  if 16 kHz isn't supported (common on Windows/macOS). Mute stops the stream (OS mic light goes off).
- VAD: webrtcvad on 20 ms sub-frames + energy gate; 160 ms onset, 900 ms trailing silence, 15 s max.
- STT: faster-whisper `base.en`, int8, loaded once and warmed up; filters Whisper's silence hallucinations.
- TTS: Piper -> pyttsx3 -> OS voice (say / SAPI / espeak-ng), automatic fallback. Synthesis of sentence
  N+1 overlaps playback of sentence N; the first chunk is cut at a comma once ≥24 chars.
- State machine IDLE → LISTENING → THINKING → SPEAKING → FOLLOW_UP (6 s, no wake word) → IDLE, plus MUTED.
- Barge-in: 240 ms of sustained speech stops playback within ~30 ms and the interrupting audio becomes
  the next utterance. Speaker mode: 300 ms echo guard + adaptive echo level (mean level over the window
  must exceed 2.5× echo). `headphones_mode: true` removes both. Bi's history records what you actually heard.
- Bypass commands: stop (→ follow-up), cancel / never mind (→ idle), go to sleep (→ idle), mute.
- Status line + hotkeys (m mute, q quit, Enter push-to-talk); reminders are spoken when idle.
- Tests: VAD on fixtures (onset, 900 ms end, mid-sentence pause, silence/fan-noise timeout, max length,
  barge-in vs echo), state machine transitions, and the full loop with fake mic/STT/TTS
  (conversation + follow-up, barge-in, echo not triggering, stop/sleep, timeout, alerts, push-to-talk).
- Fixtures are synthesized with espeak-ng plus a room-noise floor (`scripts/make_fixtures.py`); replace them
  with real laptop-mic recordings of the same names for a stronger test.
- Measured here: espeak-ng first audio 12 ms, pyttsx3 174 ms (first call). Fixed during testing:
  barge-in echo guard was never armed (baseline computed before playback started); the wake chime
  could trigger speech onset; a stale TTS chunk could play after a stop (race) – all fixed + tested.

**Not verified live:** real microphone/speakers (container has none), Whisper STT, Piper voice and the
openWakeWord model (downloads blocked). `python main.py --ptt` here reports the missing Whisper model clearly.

## Stage 5 – tools and reminders ✅
- Registry with JSON schemas, input validation (model output is untrusted), deterministic order (cacheable).
- Tools: get_time, take_note / read_notes, set_timer, set_reminder, list_reminders, cancel_reminder,
  web_search (DuckDuckGo HTML, no key; Brave with BRAVE_API_KEY; or none), file_search (read-only,
  name + content incl. PDF/DOCX, time-budgeted, configured folders only).
- Enforcement: a tool that changes state or uses the network is disabled unless listed in
  `tools.side_effects` / `tools.network`; every such call is logged to the `tool_log` table.
- Spoken confirmation for `tools.require_confirmation` ("Want me to set a pasta timer for 5 minutes?"):
  yes → the paused turn resumes with the identical system prompt and the tool result; no → "Okay, I won't";
  anything else → the pending action is dropped and the new request handled normally.
- Reminders/timers persist in SQLite, fire from a scheduler thread, are spoken (chime + voice) when Bi is
  idle, and missed ones are announced after a restart. No destructive tools exist.
- Tests: 10 new (validation, config enforcement, confirm yes/no/ignored, read-only tool flow,
  tool errors, reminder + timer alerts, notes, DuckDuckGo parser). Total 60 passing.

**Not verified live:** web_search against DuckDuckGo/Brave (blocked here; it returns a graceful message).

## Stage 6 – consolidation, bench, doctor, tests, README, run-on-login ✅
- `python -m bi.doctor [--download] [--online] [--mic]`: OS, Python, packages, PortAudio + devices, echo mode,
  models, TTS, API key (never printed), Ollama, database; each failure says how to fix it.
- `python -m bi.bench`: per-stage latency with the real installed components; missing stages are skipped,
  never estimated.
- `python -m bi.service install|uninstall|status|print`: systemd user unit / LaunchAgent / Startup-folder launcher.
- README with per-OS setup, privacy table, commands, memory CLI, encryption-at-rest, troubleshooting.
- Tests: 76 passing (+ sentence splitter, speculative STT, wake word on a fixture, service files, integration).

### Benchmark (this build machine: 4-core x86_64 cloud VM, no GPU)
| Stage | Result |
|---|---|
| end of speech → endpoint detected | **960 ms** (900 ms configured trailing silence + 80 ms frame quantization; VAD 0.02 ms/frame) |
| wake word inference (hey_jarvis) | 1.8 ms per 80 ms frame; fires on the fixture, 0 false wakes on the other fixtures |
| STT (faster-whisper base.en) | **not measured** – model download blocked here; run `python -m bi.bench` on your laptop |
| retrieval | 5 ms typical; p50 14.6 / p95 22.7 ms with 2,000 facts + 5,000 snippets (budget 100 ms) |
| LLM first token / first sentence | **not measured** – no API key / Ollama here |
| TTS first audio (Piper lessac-medium) | 57 ms median for a full first sentence (pyttsx3 fallback: 11 ms) |
| endpoint → first audio, instant LLM | 130–190 ms (retrieval + Piper + threads), measured with the real wake model, VAD and Piper |

**Biggest bottleneck and fix.** With the required 0.9 s trailing silence, the variable cost after the
endpoint was STT: Whisper ran only once the silence ended, typically 0.3–0.8 s for base.en on a laptop CPU.
Now Bi transcribes **speculatively** 300 ms into the silence. If you don't resume, the transcript is ready
when the endpoint fires (tested: critical-path STT went from 500 ms to <100 ms with a 500 ms STT); if you do resume,
the guess is discarded. The first sentence is also cut early at a comma, so Piper starts sooner.
Expected end of speech → first audio on a laptop ≈ 960 + ~170 + LLM time to first chunk. The ~1.5 s target
therefore holds when the model's first sentence arrives within ~400 ms. See "next improvements" for the
remaining lever (speculative LLM start / shorter trailing silence).

### What is placeholder / not verified live
- **Wake word**: `hey_jarvis` placeholder (say "Hey Jarvis"); custom `hey_bi.onnx` per docs/WAKEWORD.md.
- Real Anthropic and Ollama calls (no key / blocked); request shapes verified with the real SDK against a mock server.
- Whisper STT (HuggingFace blocked). Real mic/speakers (none in the container).
- VAD fixtures are synthesized speech + noise, not real laptop-mic recordings.
