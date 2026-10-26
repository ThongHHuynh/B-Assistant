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
