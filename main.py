"""Bi - hands-free, voice-first personal assistant.

    python main.py              hands-free: wake word -> conversation -> follow-up window
    python main.py --ptt        push-to-talk: press Enter to talk
    python main.py --text       keyboard chat (full brain + memory, no audio)
    python main.py --mute       start hands-free with the microphone muted
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

from bi.config import ConfigError, load_config


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Bi voice assistant")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--text", action="store_true", help="keyboard chat mode (no audio)")
    mode.add_argument("--ptt", action="store_true", help="push-to-talk: press Enter to talk")
    p.add_argument("--speak", action="store_true", help="in --text mode, also speak replies")
    p.add_argument("--mute", action="store_true", help="start with the microphone muted")
    p.add_argument("--config", help="path to config.yaml")
    p.add_argument("--backend", choices=["anthropic", "ollama"], help="override llm.backend")
    p.add_argument("--headphones", action="store_true", help="override audio.headphones_mode=true")
    return p.parse_args(argv)


def run_text(app, speak: bool = False) -> None:
    speaker = None
    if speak:
        from bi.tts import make_speaker
        speaker = make_speaker(app.cfg)
    name = app.brain.name
    print(f"{name} text mode. Type a message. Commands: /new  /facts  /stats  /quit")
    if app.llm_error:
        print(f"  ! {app.llm_error}\n  ! Memory commands still work (remember / forget / what do you know).")
    while True:
        try:
            line = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/new":
            app.brain.end_session()
            print("(new session)")
            continue
        if line == "/stats":
            print(app.store.stats())
            s = app.brain.last_stats
            print(f"last turn: retrieval {s.retrieval_ms:.0f} ms, first token {s.first_token_ms or 0:.0f} ms, "
                  f"first sentence {s.first_sentence_ms or 0:.0f} ms, total {s.total_ms:.0f} ms, model {s.model}")
            continue
        if line == "/facts":
            for f in app.store.list_facts():
                print(f"  [{f.id}] ({f.category}, {f.confidence:.2f}) {f.text}")
            continue
        print(f"{name}: ", end="", flush=True)
        cancel = threading.Event()
        try:
            for ev in app.brain.respond(line, cancel=cancel):
                if hasattr(ev, "text"):
                    print(ev.text, end=" ", flush=True)
                    if speaker:
                        speaker.say(ev.text)
                elif ev.kind == "sleep":
                    print("(sleeping - in text mode that just ends the conversation)", end="")
                    app.brain.conversation_paused()
                elif ev.kind in ("stop", "cancel"):
                    if speaker:
                        speaker.stop()
                    print("(ok)", end="")
        except KeyboardInterrupt:
            cancel.set()
            if speaker:
                speaker.stop()
            print(" (interrupted)", end="")
        print()
        if speaker:
            speaker.wait_idle()
    if speaker:
        speaker.close()


def main(argv=None) -> int:
    args = parse_args(argv)
    overrides = {}
    if args.backend:
        overrides["llm.backend"] = args.backend
    if args.headphones:
        overrides["audio.headphones_mode"] = True
    try:
        cfg = load_config(args.config, overrides)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2

    from bi.app import build_app

    alerts: list[str] = []
    app = build_app(cfg, on_alert=lambda msg: (alerts.append(msg), print(f"\n[{cfg.get('assistant.name', 'Bi')}] {msg}")),
                    quiet=not args.text)
    try:
        if args.text:
            run_text(app, speak=args.speak)
        else:
            from bi.voice.loop import run_voice
            return run_voice(app, push_to_talk=args.ptt, start_muted=args.mute)
    finally:
        print("Saving memories…", flush=True)
        t = time.time()
        app.close()
        if time.time() - t > 1:
            print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
