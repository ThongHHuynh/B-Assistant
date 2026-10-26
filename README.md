# Bi – a hands-free, voice-first personal assistant with permanent memory

Say the wake word, talk naturally, and keep talking during a short follow-up window. Bi answers in short
spoken sentences while the model is still writing, remembers you across sessions, and can take notes,
set timers and reminders, search the web and search your files. You can interrupt it at any time.

```
"Hey Jarvis"* ─► chime ─► you talk ─► 0.9 s of silence ─► Bi starts speaking ─► 6 s follow-up window ─► sleep
                                       (Whisper already transcribed during the silence)
```
\* **Placeholder wake word.** There is no pretrained "Hey Bi" model yet. Bi ships with openWakeWord's
built-in `hey_jarvis`; [docs/WAKEWORD.md](docs/WAKEWORD.md) explains how to train and swap in `hey_bi.onnx`
(or use Picovoice Porcupine).

## Privacy: what stays on your laptop

| Runs locally | Sent to the cloud (only with `llm.backend: anthropic`) |
|---|---|
| Wake word (openWakeWord), voice activity detection (webrtcvad) | The text of your current turn |
| Speech-to-text (faster-whisper), text-to-speech (Piper / OS voice) | The retrieved memory for that turn (a few facts and snippets) |
| Embeddings (all-MiniLM-L6-v2) and the whole memory database | The last few turns of the current conversation |
| Notes, reminders, file search | Background fact extraction / summaries (same model provider) |

Audio never leaves the machine. With `llm.backend: ollama`, nothing leaves it at all (web search excepted).
Secrets (passwords, API keys, card/ID numbers) are redacted from the conversation log and never stored as
facts; sensitive topics (health, religion, politics, …) are not memorized unless you say "remember that…".
API keys live in `.env` and are never written to logs.

## Install

You need Python **3.11** (3.12 works on Windows/macOS; on Linux use 3.11 because openWakeWord needs
`tflite-runtime`), a microphone and speakers. The first run downloads about 250 MB of local models.

### Windows 10/11 (PowerShell)
```powershell
winget install Python.Python.3.11        # or python.org installer; tick "Add to PATH"
cd path\to\bi
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1                # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
copy .env.example .env                    # then put your ANTHROPIC_API_KEY in .env
python -m bi.doctor --download            # checks everything and downloads the models
python main.py
```
Settings > Privacy & security > Microphone: allow desktop apps to use the microphone.

### macOS (Apple Silicon or Intel)
```bash
brew install python@3.11 portaudio espeak-ng
cd path/to/bi
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                      # add ANTHROPIC_API_KEY
python -m bi.doctor --download
python main.py
```
The first launch asks for microphone access for your terminal app (System Settings > Privacy & Security >
Microphone).

### Linux (Ubuntu/Debian; Fedora: use dnf equivalents)
```bash
sudo apt install python3.11 python3.11-venv libportaudio2 espeak-ng
cd path/to/bi
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                      # add ANTHROPIC_API_KEY
python -m bi.doctor --download
python main.py
```

### Local-only (no cloud): Ollama
Install from <https://ollama.com>, then `ollama pull llama3.2:3b`, and set `llm.backend: ollama` in
`config.yaml` (or run `python main.py --backend ollama`).

## Run

| Command | What it does |
|---|---|
| `python main.py` | Hands-free: wake word → conversation → follow-up window |
| `python main.py --ptt` | Push-to-talk: press **Enter** to talk (Enter again to finish early, or to interrupt) |
| `python main.py --text` | Keyboard chat with the full brain, memory and tools (no audio). Add `--speak` to hear replies |
| `python main.py --mute` | Start with the microphone off |
| `python main.py --headphones` | Headphones mode: no echo guard, most reliable barge-in |

While running: **m** mutes or unmutes the mic, **q** quits. The status line shows `IDLE / LISTENING / THINKING /
SPEAKING / FOLLOW-UP / MUTED`.

### Things to say
- Anything, really. Answers are 1–3 spoken sentences; say "tell me more" or "in detail" for longer ones.
- "**Remember that** my sister's name is Anna." · "**What do you know about me?**" · "**Forget** my sister's name."
  (Bi reads back what it will delete and waits for "yes".)
- "**Think harder:** should I refinance my mortgage?" routes one turn to the larger model.
- "Set a pasta timer for 8 minutes." · "Remind me at 5 to call mom." · "Take a note: buy batteries."
  · "What's in my notes?" · "Find my tax documents." · "Search the web for tonight's game score."
  Actions that change something ask "Want me to…?" first. Answer yes or no.
- "**Stop**" (stop talking, stay attentive) · "**Cancel**" / "**Never mind**" (drop it, go idle) ·
  "**Go to sleep**" (idle) · "**Mute the mic**".
- Interrupt by just talking over Bi. Playback stops and your words become the next turn.

## Memory

Everything lives in one SQLite file, `data/bi.db`:
1. **Raw log**: every turn with timestamps and session ids. Never deleted automatically.
2. **Facts**: short statements ("User prefers tea over coffee.") with category, confidence, source turn
   and history. A background extractor (Claude Haiku) pulls durable facts after each conversation and every
   6 turns. When something changes, the old fact is marked *superseded*, not deleted.
3. **Semantic index**: MiniLM embeddings of facts, conversation snippets and session summaries
   (sqlite-vec; numpy fallback).

Each turn, Bi retrieves the most relevant facts and snippets within a 100 ms budget (measured: about 5 ms
on a small database, 15 ms p50 with 2,000 facts and 5,000 snippets). It also includes a few core facts
and the end of your previous conversation, and puts them in the system prompt. It never dumps the whole
memory into the prompt.

```bash
python -m bi.memory list [--all] [--category person]   # what Bi knows (* = you said "remember")
python -m bi.memory search "coffee"                    # facts + past conversation snippets
python -m bi.memory add "User is vegetarian." --category preference
python -m bi.memory edit 12 "User prefers green tea."
python -m bi.memory delete 12 [--scrub]                # --scrub also redacts it from the raw log
python -m bi.memory export --format md --out memory.md # or --format json
python -m bi.memory consolidate                        # merge duplicates, summarize, decay (also runs nightly)
python -m bi.memory stats | turns | reindex
python -m bi.import_history conversations.json         # ChatGPT / Claude export, JSON or markdown transcript
```
Consolidation also runs nightly at `memory.consolidation.nightly_at` while Bi is running (it catches up
if the laptop was asleep). It merges near-duplicate facts, summarizes sessions older than 2 days, and
decays and then archives low-confidence facts that haven't been used in 30 days. Facts you explicitly
asked Bi to remember never decay.

### Encrypt the memory at rest (optional)
`data/` holds your conversations. Put it on an encrypted volume and point `paths.data_dir` / `paths.db` at it:
- **macOS**: FileVault (System Settings > Privacy & Security) encrypts the whole disk; or create an
  encrypted APFS disk image in Disk Utility (File > New Image > Blank Image, Encryption: AES-256) and set
  `paths.db: /Volumes/BiData/bi.db`.
- **Windows**: BitLocker (Pro/Enterprise) or Device Encryption (Home) for the whole drive; or a VeraCrypt
  container mounted as a drive letter.
- **Linux**: full-disk LUKS (installer option), or an encrypted directory with `gocryptfs`:
  `gocryptfs -init ~/.bi-enc && gocryptfs ~/.bi-enc ~/bi-data`, then set `paths.db: ~/bi-data/bi.db`.

## Configuration

All model names, thresholds and paths are in [`config.yaml`](config.yaml) (commented). The ones you
are most likely to touch:

| Setting | Default | Notes |
|---|---|---|
| `llm.backend` | `anthropic` | or `ollama` |
| `llm.anthropic.chat_model` / `think_model` / `extractor_model` | `claude-haiku-5-5` / `claude-opus-5-5` / `claude-haiku-5-5` | fast model for normal turns, larger one for "think harder" |
| `wake.model` / `wake.threshold` | `hey_jarvis` / `0.5` | see docs/WAKEWORD.md |
| `vad.trailing_silence_ms` | `900` | how long a pause ends your turn; lower = faster but may cut you off |
| `vad.follow_up_s` | `6` | follow-up window without the wake word |
| `stt.model` | `base.en` | `tiny.en` faster, `small.en` more accurate |
| `audio.headphones_mode` | `false` | `true` disables the echo guard |
| `tts.engine` / `tts.piper_voice` | `piper` / `en_US-lessac-medium` | falls back to pyttsx3 then the OS voice |
| `tools.require_confirmation` | state-changing tools | remove a tool to skip its "Want me to…?" |
| `tools.file_search.folders` | `~/Documents`, `~/Desktop` | read-only |
| `tools.web_search.provider` | `duckduckgo` | `brave` (needs `BRAVE_API_KEY`) or `none` |

## Run at login
```bash
python -m bi.service install      # Linux: systemd user service · macOS: LaunchAgent · Windows: Startup folder
python -m bi.service status
python -m bi.service uninstall
```
Linux logs: `journalctl --user -u bi -f`. macOS logs: `data/logs/launchd.*.log`. The app log is always
`data/logs/bi.log`. When Bi runs this way there's no terminal, so use the voice commands and "mute the mic".

## Health check and benchmark
```bash
python -m bi.doctor [--download] [--online] [--mic]   # devices, models, keys, database; says how to fix each
python -m bi.bench [--no-llm]                          # per-stage latency on this machine
python -m pytest -q                                    # 76 tests, no mic or API key needed
```

## How it fits together
```
mic (16 kHz, 80 ms frames) ─► wake word ─► VAD endpointer ──► faster-whisper ──┐   (speculative: starts
                                  ▲            │ barge-in                       │    300 ms into the pause)
                                  │            ▼                                ▼
 speaker ◄── Player ◄── Piper ◄── sentence splitter ◄── LLM stream ◄── brain: commands · memory retrieval
         (interruptible)  (N+1 synthesizing while N plays)    (Anthropic | Ollama)   · tools · confirmations
                                                                                      │
                                       SQLite: raw log · facts · snippets · vectors ◄─┘ background extractor
```
- `bi/brain`: LLM interface (`stream(messages, tools, system) -> TextDelta | ToolCall | Done`), prompt
  builder (static persona first so it can be prompt-cached), sentence splitter, voice commands, agent loop
- `bi/memory`: store, embeddings, vector index, retrieval, extractor, consolidation, privacy filters, CLI
- `bi/audio`, `bi/wake`, `bi/vad`, `bi/stt`, `bi/tts`, `bi/voice` (state machine, loop, status line)
- `bi/tools`: registry + notes, timers/reminders, web search, file search, time

## Troubleshooting
| Problem | Fix |
|---|---|
| "No microphone found" | `python -m bi.doctor`; grant mic permission; set `audio.input_device` to a name from the list |
| Bi wakes by itself / never wakes | adjust `wake.threshold` (docs/WAKEWORD.md) |
| Bi cuts you off mid-sentence | raise `vad.trailing_silence_ms` to 1100 |
| Bi interrupts itself (hears its own voice) | use headphones + `--headphones`, or raise `audio.barge_in_echo_factor` |
| Robotic voice | Piper voice missing: `python -m bi.doctor --download` |
| "ANTHROPIC_API_KEY is not set" | `cp .env.example .env` and add the key, or use Ollama |
| Slow answers | `python -m bi.bench`; try `stt.model: tiny.en`, keep `chat_effort: low` |
