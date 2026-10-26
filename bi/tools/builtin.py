"""Built-in tools: time/date, notes, timers & reminders, web search, file search."""
from __future__ import annotations

import datetime as dt
import logging
import os
import re
import threading
import time
from pathlib import Path

from . import Tool, ToolInputError, ToolRegistry

log = logging.getLogger(__name__)


def human_duration(seconds: int) -> str:
    seconds = int(seconds)
    parts = []
    for unit, n in (("hour", 3600), ("minute", 60), ("second", 1)):
        q, seconds = divmod(seconds, n)
        if q:
            parts.append(f"{q} {unit}{'s' if q != 1 else ''}")
    return " and ".join(parts[:2]) or "0 seconds"


def human_time(ts: float, now: float | None = None) -> str:
    d = dt.datetime.fromtimestamp(ts)
    n = dt.datetime.fromtimestamp(now or time.time())
    t = d.strftime("%I:%M %p").lstrip("0")
    if d.date() == n.date():
        return f"at {t}"
    if d.date() == (n + dt.timedelta(days=1)).date():
        return f"tomorrow at {t}"
    return d.strftime("on %A, %B %d at ") + t


# ---------------------------------------------------------------- time --
def get_time(args: dict) -> str:
    tzname = args.get("timezone")
    if tzname:
        try:
            from zoneinfo import ZoneInfo
            now = dt.datetime.now(ZoneInfo(tzname))
        except Exception:  # noqa: BLE001
            raise ToolInputError(f"unknown time zone '{tzname}' (use an IANA name like Europe/Paris)")
    else:
        now = dt.datetime.now().astimezone()
    return now.strftime("%A, %B %d, %Y, %I:%M %p ") + (now.tzname() or "")


# ----------------------------------------------------------- reminders --
class ReminderScheduler(threading.Thread):
    """Fires due reminders/timers through registry.alert(); survives restarts (stored in SQLite)."""

    def __init__(self, store, registry: ToolRegistry, poll_s: float = 0.5):
        super().__init__(daemon=True, name="reminders")
        self.store, self.reg, self.poll = store, registry, poll_s
        self.stop_event = threading.Event()
        self.wake = threading.Event()

    def run(self) -> None:
        first = True
        while not self.stop_event.is_set():
            now = time.time()
            for r in self.store.pending_reminders():
                if r["due_at"] > now:
                    break
                late = now - r["due_at"]
                if r["kind"] == "timer":
                    msg = f"Your {r['text']} timer is done." if r["text"] else "Your timer is done."
                elif first and late > 120:
                    msg = f"You missed a reminder {human_time(r['due_at'], now)}: {r['text']}."
                else:
                    msg = f"Reminder: {r['text']}."
                self.store.mark_reminder_fired(r["id"])
                self.reg.alert(msg)
            first = False
            self.wake.wait(self.poll)
            self.wake.clear()

    def stop(self) -> None:
        self.stop_event.set()
        self.wake.set()


def _parse_when(args: dict, now: float) -> float:
    if args.get("in_minutes") is not None:
        return now + float(args["in_minutes"]) * 60
    at = str(args.get("at", "")).strip()
    if not at:
        raise ToolInputError("give either 'at' (ISO local time) or 'in_minutes'")
    try:
        d = dt.datetime.fromisoformat(at.replace("Z", "+00:00"))
    except ValueError:
        m = re.fullmatch(r"(\d{1,2}):(\d{2})", at)
        if not m:
            raise ToolInputError(f"can't parse time '{at}'; use ISO 8601 like 2026-05-01T17:30")
        base = dt.datetime.fromtimestamp(now)
        d = base.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
        if d.timestamp() <= now:
            d += dt.timedelta(days=1)
    return d.timestamp()


def register_builtin(reg: ToolRegistry, cfg, store) -> None:
    tcfg = cfg.section("tools")

    reg.register(Tool(
        "get_time", "Get the current local date and time, optionally in another IANA time zone. Call this when "
        "the user asks for the time or date somewhere else; the local time is already in your context.",
        {"type": "object", "properties": {"timezone": {"type": "string", "description": "IANA zone, e.g. Asia/Tokyo"}}},
        get_time))

    # ------------------------------------------------------------ notes --
    def take_note(a):
        nid = store.add_note(a["text"].strip())
        return f"Saved note #{nid}."

    def read_notes(a):
        notes = store.notes(200)
        q = (a.get("query") or "").lower().strip()
        if q:
            words = [w for w in re.findall(r"\w+", q) if len(w) > 2]
            notes = [n for n in notes if any(w in n["text"].lower() for w in words)] if words else notes
        notes = notes[: int(a.get("limit") or 5)]
        if not notes:
            return "No matching notes." if q else "There are no notes yet."
        return "\n".join(f"#{n['id']} ({dt.datetime.fromtimestamp(n['created_at']):%b %d}): {n['text']}" for n in notes)

    reg.register(Tool(
        "take_note", "Save a note for the user. Call when they ask you to note, jot down or write something down.",
        {"type": "object", "properties": {"text": {"type": "string", "description": "The note, in the user's words"}},
         "required": ["text"]},
        take_note, side_effect=True, describe=lambda a: f"take a note saying: {a['text']}"))
    reg.register(Tool(
        "read_notes", "Read back the user's saved notes, newest first, optionally filtered by words.",
        {"type": "object", "properties": {"query": {"type": "string"},
                                          "limit": {"type": "integer", "minimum": 1}}},
        read_notes))

    # ------------------------------------------------- timers/reminders --
    def set_timer(a):
        secs = int(a["seconds"])
        if secs <= 0 or secs > 86400:
            raise ToolInputError("seconds must be between 1 and 86400")
        due = time.time() + secs
        rid = store.add_reminder(a.get("label", "").strip(), due, kind="timer")
        sched.wake.set()
        return f"Timer #{rid} set for {human_duration(secs)}, ending {human_time(due)}."

    def set_reminder(a):
        now = time.time()
        due = _parse_when(a, now)
        if due <= now:
            raise ToolInputError("that time is in the past")
        rid = store.add_reminder(a["text"].strip(), due)
        sched.wake.set()
        return f"Reminder #{rid} set {human_time(due, now)}: {a['text'].strip()}."

    def list_reminders(_a):
        items = store.pending_reminders()
        if not items:
            return "No pending reminders or timers."
        now = time.time()
        return "\n".join(f"#{r['id']} {r['kind']} {human_time(r['due_at'], now)}: {r['text'] or '(no label)'}"
                         for r in items)

    def cancel_reminder(a):
        return f"Cancelled #{a['id']}." if store.cancel_reminder(int(a["id"])) else f"No pending reminder #{a['id']}."

    def _describe_reminder(a):
        try:
            when = human_time(_parse_when(a, time.time()))
        except ToolInputError:
            when = "at the time you said"
        return f"remind you {when} to {a['text']}"

    def _describe_cancel(a):
        match = [r for r in store.pending_reminders() if r["id"] == int(a["id"])]
        return f"cancel the {match[0]['kind']} '{match[0]['text']}'" if match else f"cancel reminder {a['id']}"

    reg.register(Tool(
        "set_timer", "Start a countdown timer. Bi speaks up when it ends.",
        {"type": "object", "properties": {"seconds": {"type": "integer", "minimum": 1},
                                          "label": {"type": "string", "description": "e.g. pasta"}},
         "required": ["seconds"]},
        set_timer, side_effect=True,
        describe=lambda a: f"set a {(a.get('label') or '').strip() + ' ' if a.get('label') else ''}timer for "
                           f"{human_duration(int(a['seconds']))}"))
    reg.register(Tool(
        "set_reminder", "Remind the user about something at a specific time; Bi says it out loud then. "
        "Give 'at' as a local ISO 8601 date-time computed from the current time in your context, or 'in_minutes'.",
        {"type": "object", "properties": {"text": {"type": "string", "description": "What to remind about"},
                                          "at": {"type": "string", "description": "Local time, e.g. 2026-05-01T17:30"},
                                          "in_minutes": {"type": "number", "minimum": 0}},
         "required": ["text"]},
        set_reminder, side_effect=True, describe=_describe_reminder))
    reg.register(Tool("list_reminders", "List pending reminders and timers.", {"type": "object", "properties": {}},
                      list_reminders))
    reg.register(Tool(
        "cancel_reminder", "Cancel a pending reminder or timer by its number (see list_reminders).",
        {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
        cancel_reminder, side_effect=True, describe=_describe_cancel))

    # --------------------------------------------------------- web search --
    from .web import make_search
    search = make_search(tcfg.get("web_search", {}))
    if search is not None:
        reg.register(Tool(
            "web_search", "Search the web. Call for current events, news, prices, opening hours, weather, sports "
            "results, or facts you are not sure about. Returns titles, snippets and URLs.",
            {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            lambda a: search(a["query"]), network=True))

    # -------------------------------------------------------- file search --
    from .files import FileSearcher
    fs = FileSearcher(tcfg.get("file_search", {}))
    reg.register(Tool(
        "file_search", "Search the user's documents (read-only) by file name and content. Call when they ask "
        "about something in their files, notes on disk, or where a document is.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        lambda a: fs.search(a["query"])))

    sched = ReminderScheduler(store, reg)
    if any(n in reg.tools for n in ("set_timer", "set_reminder")):
        sched.start()
        reg.add_closer(sched.stop)
