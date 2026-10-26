"""Import a chat export into Bi's raw log and extract facts from it.

    python -m bi.import_history conversations.json            # ChatGPT or Claude export
    python -m bi.import_history notes.md --no-extract          # markdown / plain text transcript
    python -m bi.import_history export.json --limit 50         # only the 50 most recent conversations

Supported: ChatGPT conversations.json (mapping tree), Claude conversations.json (chat_messages),
generic JSON ([{role, content}] or [{"messages": [...]}]), and markdown/text with "User:"/"Assistant:"
lines or "## User" / "## Assistant" headings.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

USER_ROLES = {"user", "human", "you", "me", "q"}
ASSISTANT_ROLES = {"assistant", "ai", "bot", "chatgpt", "gpt", "claude", "bi", "model", "a"}


@dataclass
class Conversation:
    title: str
    started_at: float | None
    turns: list[tuple[str, str, float | None]] = field(default_factory=list)   # (role, text, ts)


def _ts(v) -> float | None:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v / 1000 if v > 1e12 else v)
    try:
        return dt.datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _role(r: str) -> str | None:
    r = (r or "").strip().lower()
    if r in USER_ROLES:
        return "user"
    if r in ASSISTANT_ROLES:
        return "assistant"
    return None


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "parts" in content:
            return "\n".join(p for p in content["parts"] if isinstance(p, str))
        return content.get("text", "") or ""
    if isinstance(content, list):
        return "\n".join(_text(c) for c in content if isinstance(c, (str, dict)))
    return ""


def parse_chatgpt(conv: dict) -> Conversation:
    mapping = conv.get("mapping", {})
    node_id = conv.get("current_node")
    chain = []
    while node_id and node_id in mapping:
        chain.append(mapping[node_id])
        node_id = mapping[node_id].get("parent")
    if not chain:   # no current_node: fall back to all nodes by time
        chain = sorted(mapping.values(), key=lambda n: (n.get("message") or {}).get("create_time") or 0, reverse=True)
    c = Conversation(conv.get("title") or "ChatGPT conversation", _ts(conv.get("create_time")))
    for node in reversed(chain):
        m = node.get("message") or {}
        role = _role((m.get("author") or {}).get("role", ""))
        text = _text(m.get("content")).strip()
        if role and text:
            c.turns.append((role, text, _ts(m.get("create_time"))))
    return c


def parse_claude(conv: dict) -> Conversation:
    c = Conversation(conv.get("name") or "Claude conversation", _ts(conv.get("created_at")))
    for m in conv.get("chat_messages", []):
        role = _role(m.get("sender", ""))
        text = (m.get("text") or _text(m.get("content"))).strip()
        if role and text:
            c.turns.append((role, text, _ts(m.get("created_at"))))
    return c


def parse_generic(msgs: list, title: str = "Imported conversation") -> Conversation:
    c = Conversation(title, None)
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = _role(m.get("role") or m.get("sender") or m.get("author") or "")
        text = _text(m.get("content", m.get("text", ""))).strip()
        if role and text:
            ts = _ts(m.get("timestamp") or m.get("created_at") or m.get("time"))
            c.turns.append((role, text, ts))
            c.started_at = c.started_at or ts
    return c


def parse_json(data) -> list[Conversation]:
    if isinstance(data, dict):
        if "conversations" in data:
            data = data["conversations"]
        elif "messages" in data:
            return [parse_generic(data["messages"], data.get("title", "Imported conversation"))]
        else:
            data = [data]
    convs = []
    if data and all(isinstance(x, dict) and ("role" in x or "sender" in x) and "mapping" not in x for x in data):
        return [parse_generic(data)]
    for item in data:
        if not isinstance(item, dict):
            continue
        if "mapping" in item:
            convs.append(parse_chatgpt(item))
        elif "chat_messages" in item:
            convs.append(parse_claude(item))
        elif "messages" in item:
            convs.append(parse_generic(item["messages"], item.get("title") or item.get("name") or "Imported conversation"))
    return convs


_LINE = re.compile(r"^\s*(?:\*\*|__)?([A-Za-z]+)(?:\*\*|__)?\s*[:：]\s*(?:\*\*|__)?\s*(.*)$")
_HEAD = re.compile(r"^\s*#{1,6}\s*([A-Za-z]+)\s*:?\s*$")


def parse_text(text: str, title: str) -> list[Conversation]:
    convs: list[Conversation] = []
    cur = Conversation(title, None)
    role, buf = None, []

    def flush():
        nonlocal buf
        if role and "".join(buf).strip():
            cur.turns.append((role, "\n".join(buf).strip(), None))
        buf = []

    for line in text.splitlines():
        if re.match(r"^\s*(---+|===+)\s*$", line) or (re.match(r"^#\s+\S", line) and not _HEAD.match(line)):
            flush()
            role = None
            if cur.turns:
                convs.append(cur)
            cur = Conversation(line.strip("#-= ").strip() or title, None)
            continue
        h = _HEAD.match(line)
        m = _LINE.match(line)
        if h and _role(h.group(1)):
            flush()
            role = _role(h.group(1))
            continue
        if m and _role(m.group(1)):
            flush()
            role = _role(m.group(1))
            buf = [m.group(2)]
            continue
        buf.append(line)
    flush()
    if cur.turns:
        convs.append(cur)
    return convs


def load_conversations(path: Path) -> list[Conversation]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json" or raw.lstrip()[:1] in "[{":
        try:
            return parse_json(json.loads(raw))
        except json.JSONDecodeError:
            pass
    return parse_text(raw, path.stem)


def import_conversations(store, convs: list[Conversation], extractor=None, assistant_name: str = "Bi",
                         progress=print) -> dict:
    stats = {"conversations": 0, "turns": 0, "chunks": 0, "facts": 0}
    for i, c in enumerate(convs, 1):
        if not c.turns:
            continue
        start = c.started_at or next((t for _, _, t in c.turns if t), None)
        sid = store.start_session("import", started_at=start, title=c.title[:200])
        ids: list[tuple[int, str, str]] = []
        for role, text, ts in c.turns:
            ids.append((store.add_turn(sid, role, text, ts=ts or start), role, text))
        store.end_session(sid)
        # snippets: user turn + (trimmed) assistant reply
        for j, (tid, role, text) in enumerate(ids):
            if role != "user":
                continue
            reply = next(((t, x) for t, r, x in ids[j + 1:j + 2] if r == "assistant"), None)
            chunk = f"User: {text[:600]}" + (f"\nAssistant: {reply[1][:400]}" if reply else "")
            store.add_chunk("turns", sid, chunk, tid, reply[0] if reply else tid, ts=start)
            stats["chunks"] += 1
        stats["conversations"] += 1
        stats["turns"] += len(ids)
        if extractor is not None:
            turns = store.session_turns(sid)
            window, size = [], 0
            for t in turns:
                window.append(t)
                size += len(t.text) if t.role == "user" else min(len(t.text), 300)
                if size > 9000:
                    stats["facts"] += len(extractor.extract_from_turns(sid, _trim(window)))
                    window, size = [], 0
            if any(t.role == "user" for t in window):
                stats["facts"] += len(extractor.extract_from_turns(sid, _trim(window)))
        progress(f"  [{i}/{len(convs)}] {c.title[:60]!r}: {len(c.turns)} turns")
    return stats


def _trim(turns):
    """Shorten assistant turns for extraction: facts come from the user's words."""
    from .memory.store import Turn
    return [t if t.role == "user" else Turn(t.id, t.session_id, t.role, t.text[:300], t.ts) for t in turns]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m bi.import_history", description=__doc__.split("\n\n")[0])
    p.add_argument("file")
    p.add_argument("--config")
    p.add_argument("--no-extract", action="store_true", help="import the log only; skip LLM fact extraction")
    p.add_argument("--limit", type=int, default=0, help="only the N most recent conversations")
    p.add_argument("-y", "--yes", action="store_true", help="don't ask before running extraction")
    a = p.parse_args(argv)

    from .config import load_config
    from .logs import setup_logging
    from .memory.store import open_store

    path = Path(a.file).expanduser()
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1
    convs = [c for c in load_conversations(path) if c.turns]
    if not convs:
        print("No conversations found. Supported: ChatGPT/Claude JSON exports, generic JSON, or User:/Assistant: text.")
        return 1
    convs.sort(key=lambda c: c.started_at or 0)
    if a.limit:
        convs = convs[-a.limit:]
    n_turns = sum(len(c.turns) for c in convs)
    print(f"Found {len(convs)} conversations, {n_turns} turns in {path.name}.")

    cfg = load_config(a.config)
    setup_logging(cfg.path_of("paths.log_file", "data/logs/bi.log"))
    store = open_store(cfg)
    extractor = None
    if not a.no_extract:
        from .brain.llm import LLMConfigError, make_llm
        from .memory.extractor import FactExtractor
        try:
            llm = make_llm(cfg)
            user_chars = sum(len(t[1]) for c in convs for t in c.turns if t[0] == "user")
            calls = len(convs) + user_chars // 9000
            print(f"Fact extraction will make roughly {calls} calls to {llm.model_for('extractor')}.")
            if a.yes or input("Run extraction? [Y/n] ").strip().lower() in ("", "y", "yes"):
                extractor = FactExtractor(store, llm)
        except LLMConfigError as e:
            print(f"Skipping fact extraction: {e}")
    stats = import_conversations(store, convs, extractor)
    print(f"Imported {stats['conversations']} conversations, {stats['turns']} turns, {stats['chunks']} snippets; "
          f"{stats['facts']} new facts.")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
