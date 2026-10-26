"""Inspect and edit Bi's memory.

    python -m bi.memory list [--all] [--category preference]
    python -m bi.memory search "coffee"
    python -m bi.memory add "User prefers tea." [--category preference]
    python -m bi.memory edit 12 "User prefers green tea."
    python -m bi.memory delete 12 [--scrub]
    python -m bi.memory export [--format json|md] [--out file]
    python -m bi.memory consolidate | stats | reindex | turns [--session ID] [--limit N]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from ..config import load_config
from ..logs import setup_logging
from .store import CATEGORIES, open_store


def _fmt_ts(ts) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "-"


def _print_fact(f) -> None:
    state = "" if f.active else (f"  [superseded by {f.superseded_by}]" if f.superseded_by else "  [archived]")
    star = "*" if f.explicit else " "
    print(f"{f.id:>5}{star} {f.category:<10} {f.confidence:4.2f}  {_fmt_ts(f.updated_at)}  {f.text}{state}")


def to_markdown(data: dict) -> str:
    lines = ["# Bi memory export", "", f"Exported {_fmt_ts(data['exported_at'])}", "", "## Facts", ""]
    for cat in CATEGORIES:
        facts = [f for f in data["facts"] if f["category"] == cat and f["superseded_by"] is None and not f["archived"]]
        if facts:
            lines.append(f"### {cat.title()}")
            lines += [f"- {f['text']} _(id {f['id']}, confidence {f['confidence']:.2f})_" for f in facts]
            lines.append("")
    old = [f for f in data["facts"] if f["superseded_by"] is not None or f["archived"]]
    if old:
        lines += ["## Superseded / archived", ""] + [f"- ~~{f['text']}~~ (id {f['id']})" for f in old] + [""]
    lines += ["## Conversations", ""]
    by_session: dict[str, list] = {}
    for t in data["turns"]:
        by_session.setdefault(t["session_id"], []).append(t)
    for s in data["sessions"]:
        turns = by_session.get(s["id"], [])
        if not turns:
            continue
        lines.append(f"### {_fmt_ts(s['started_at'])} ({s['id']})")
        if s.get("summary"):
            lines.append(f"_Summary: {s['summary']}_")
        lines += [f"- **{'You' if t['role'] == 'user' else 'Bi'}:** {t['text']}" for t in turns] + [""]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m bi.memory", description="Inspect and edit Bi's memory")
    p.add_argument("--config")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list"); s.add_argument("--all", action="store_true"); s.add_argument("--category", choices=CATEGORIES)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=8)
    s = sub.add_parser("add"); s.add_argument("text"); s.add_argument("--category", choices=CATEGORIES, default="other")
    s = sub.add_parser("edit"); s.add_argument("id", type=int); s.add_argument("text", nargs="?")
    s.add_argument("--category", choices=CATEGORIES); s.add_argument("--confidence", type=float)
    s = sub.add_parser("delete"); s.add_argument("id", type=int, nargs="+")
    s.add_argument("--scrub", action="store_true", help="also redact the source turns in the raw log")
    s.add_argument("-y", "--yes", action="store_true")
    s = sub.add_parser("export"); s.add_argument("--format", choices=["json", "md"], default="json"); s.add_argument("--out")
    sub.add_parser("consolidate")
    sub.add_parser("stats")
    sub.add_parser("reindex")
    s = sub.add_parser("turns"); s.add_argument("--session"); s.add_argument("--limit", type=int, default=30)
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    setup_logging(cfg.path_of("paths.log_file", "data/logs/bi.log"))
    store = open_store(cfg)
    try:
        if a.cmd == "list":
            facts = store.list_facts(include_inactive=a.all, category=a.category)
            print(f"{'id':>5}  {'category':<10} conf  updated           text   (* = you said 'remember')")
            for f in facts:
                _print_fact(f)
            print(f"\n{len(facts)} facts")
        elif a.cmd == "search":
            print("facts:")
            for f, sc in store.search_facts(a.query, k=a.k):
                print(f"  {sc:5.2f} ", end="")
                _print_fact(f)
            print("conversation snippets:")
            for c, sc in store.search_chunks(a.query, k=a.k):
                print(f"  {sc:5.2f}  [{_fmt_ts(c['ts'])}] {c['text'][:140].replace(chr(10), ' | ')}")
        elif a.cmd == "add":
            fid, created = store.add_fact(a.text, a.category, 0.95, explicit=True)
            print(("added" if created else "already known as") + f" fact {fid}")
        elif a.cmd == "edit":
            if not store.update_fact(a.id, a.text, a.category, a.confidence):
                print(f"no fact with id {a.id}", file=sys.stderr)
                return 1
            _print_fact(store.get_fact(a.id))
        elif a.cmd == "delete":
            for fid in a.id:
                f = store.get_fact(fid)
                if not f:
                    print(f"no fact with id {fid}", file=sys.stderr)
                    continue
                if not a.yes and input(f"Delete [{fid}] {f.text!r} and its older versions? [y/N] ").lower() != "y":
                    continue
                gone = store.delete_fact(fid, cascade=True, scrub_sources=a.scrub)
                print(f"deleted facts {gone}")
        elif a.cmd == "export":
            data = store.export()
            out = json.dumps(data, indent=2, ensure_ascii=False) if a.format == "json" else to_markdown(data)
            if a.out:
                with open(a.out, "w", encoding="utf-8") as fh:
                    fh.write(out)
                print(f"wrote {a.out}")
            else:
                print(out)
        elif a.cmd == "consolidate":
            from ..brain.llm import LLMConfigError, make_llm
            from .consolidate import consolidate
            try:
                llm = make_llm(cfg)
            except LLMConfigError as e:
                print(f"note: {e}\n(session summaries skipped; duplicate merge and decay still run)")
                llm = None
            print(consolidate(store, llm, cfg))
        elif a.cmd == "stats":
            for k, v in store.stats().items():
                print(f"{k:>18}: {v}")
        elif a.cmd == "reindex":
            print(f"re-embedded {store.reindex()} items with {store.embedder.name}")
        elif a.cmd == "turns":
            rows = store.session_turns(a.session) if a.session else store.recent_turns(a.limit)
            for t in rows[-a.limit:]:
                print(f"{t.id:>6} {_fmt_ts(t.ts)} {t.session_id} {t.role:>9}: {t.text}")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
