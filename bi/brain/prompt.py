"""System prompt builder: static persona first (prompt-cached), dynamic context after."""
from __future__ import annotations

import datetime as dt

from .llm import SystemPrompt

PERSONA = """You are {name}, a voice-first personal assistant running on the user's laptop. Everything you \
write is converted to speech.

How to talk:
- Answer in 1 to 3 short, natural spoken sentences unless the user asks for more detail.
- Never use markdown, bullet points, numbered lists, headings, tables, code blocks, emoji or URLs. If several \
items matter, say them in one flowing sentence.
- Write numbers, times and units the way a person would say them.
- Lead with the answer. No preambles like "Great question" and no sign-offs.
- If you didn't catch something or a request is ambiguous, ask one short question.

Memory:
- You have a long-term memory of the user. Relevant facts and past conversation snippets appear below under \
"What you remember". Use them naturally, the way a friend would, without announcing that you are reading \
from memory. Never invent memories. If the facts conflict, prefer the newer one.
- The user can say "remember that ...", "forget ...", or "what do you know about me"; those are handled for \
you, so you don't need tools for them.

Tools:
- Use a tool when it gives a better answer than guessing: current time or date, notes, timers and reminders, \
web search for current events, prices or facts you're unsure of, and searching the user's files.
- Some tools ask the user for a spoken confirmation before they run. Call them normally; the system handles \
the confirmation.
- After a tool returns, summarize the result in speech, briefly."""


def build_static(name: str) -> str:
    return PERSONA.format(name=name)


def _ago(ts: float, now: float) -> str:
    d = now - ts
    if d < 3600:
        return "earlier today" if d > 600 else "a few minutes ago"
    if d < 86400:
        return "earlier today" if dt.datetime.fromtimestamp(ts).date() == dt.datetime.fromtimestamp(now).date() else "yesterday"
    days = int(d // 86400)
    return f"{days} days ago" if days < 30 else dt.datetime.fromtimestamp(ts).strftime("%B %Y")


def build_dynamic(now: dt.datetime, facts: list, snippets: list, recent: list, user_name: str = "",
                  tz_name: str = "") -> str:
    lines = [f"Current date and time: {now.strftime('%A, %B %d, %Y, %I:%M %p').replace(' 0', ' ')}"
             + (f" ({tz_name})" if tz_name else "") + "."]
    if user_name:
        lines.append(f"The user's name is {user_name}.")
    ts_now = now.timestamp()
    if facts or snippets or recent:
        lines.append("\nWhat you remember (retrieved for this turn; may be partial):")
    if facts:
        lines.append("Facts about the user:")
        for f in facts:
            lines.append(f"- {f.text}")
    if snippets:
        lines.append("Related past conversations:")
        for c in snippets:
            lines.append(f"- ({_ago(c['ts'], ts_now)}) {c['text'][:400]}")
    if recent:
        lines.append("End of your previous conversation:")
        for t in recent:
            who = "User" if t.role == "user" else "You"
            lines.append(f"- {who}: {t.text[:240]}")
    return "\n".join(lines)


def build_system(name: str, now: dt.datetime, facts=(), snippets=(), recent=(), user_name: str = "",
                 tz_name: str = "") -> SystemPrompt:
    return SystemPrompt(static=build_static(name),
                        dynamic=build_dynamic(now, list(facts), list(snippets), list(recent), user_name, tz_name))
