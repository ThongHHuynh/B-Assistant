"""The brain: one user utterance in, a stream of speakable sentences out.

    for ev in brain.respond("what's on my calendar", cancel=event):
        Say(text)      -> speak / print it
        Action(kind)   -> stop | cancel | sleep | mute   (handled by the voice loop)
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterator

from ..memory.extractor import FactExtractor
from ..memory.privacy import redact_secrets
from ..memory.retrieval import Retrieved, Retriever
from ..memory.store import MemoryStore
from .commands import parse_command, parse_yes_no
from .llm import LLMConfigError, LLMError, SystemPrompt, TextDelta, ToolCall, Done
from .prompt import build_system
from .sentences import SentenceSplitter, clean_for_speech

log = logging.getLogger(__name__)
MAX_TOOL_ROUNDS = 5


@dataclass
class Say:
    text: str


@dataclass
class Action:
    kind: str


@dataclass
class PendingForget:
    fact_ids: list[int]
    user_text: str
    created: float = field(default_factory=time.time)


@dataclass
class PendingTool:
    messages: list[dict]          # turn messages up to and including the assistant tool_use message
    system: SystemPrompt
    role: str
    calls: list[ToolCall]
    user_text: str
    spoken: list[str]
    created: float = field(default_factory=time.time)


@dataclass
class TurnStats:
    retrieval_ms: float = 0.0
    first_token_ms: float | None = None
    first_sentence_ms: float | None = None
    total_ms: float = 0.0
    model: str = ""
    tools: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- helpers --
_PRON_3P = [(r"\bI am\b", "User is"), (r"\bI'm\b", "User is"), (r"\bI've\b", "User has"), (r"\bI'd\b", "User would"),
            (r"\bI'll\b", "User will"), (r"\bI was\b", "User was"), (r"\bI\b", "User"), (r"\bmy\b", "User's"),
            (r"\bMy\b", "User's"), (r"\bmine\b", "User's"), (r"\bme\b", "User"), (r"\bmyself\b", "User")]
_PRON_2P = [(r"\bUser is\b", "you are"), (r"\bUser has\b", "you have"), (r"\bUser was\b", "you were"),
            (r"\bUser's\b", "your"), (r"\bthe user's\b", "your"), (r"\bUser\b", "you"), (r"\bthe user\b", "you")]


def to_third_person(text: str) -> str:
    t = text.strip()
    for pat, rep in _PRON_3P:
        t = re.sub(pat, rep, t)
    return t[0].upper() + t[1:] if t else t


def to_second_person(text: str) -> str:
    t = text.strip().rstrip(".")
    for pat, rep in _PRON_2P:
        t = re.sub(pat, rep, t, flags=re.I)
    # third-person verbs right after "you": "you prefers" -> "you prefer"
    t = re.sub(r"\byou (\w+?)(?<!s)s\b", lambda m: "you " + m.group(1) if m.group(1) not in ("i", "ha", "wa") else m.group(0), t)
    return t


def guess_category(text: str) -> str:
    low = text.lower()
    if re.search(r"\b(prefer|like|love|hate|dislike|favou?rite|enjoy|can't stand|allergic)\b", low):
        return "preference"
    if re.search(r"\b(every|usually|always|often|each (day|morning|week)|routine|habit|tend to)\b", low):
        return "habit"
    if re.search(r"\b(project|working on|building|deadline|launch|thesis|startup)\b", low):
        return "project"
    if re.search(r"\b(name is|named|wife|husband|partner|sister|brother|mom|mother|dad|father|son|daughter|"
                 r"friend|boss|colleague|cowork|girlfriend|boyfriend|kid|dog|cat)\b", low):
        return "person"
    return "other"


# ------------------------------------------------------------------ brain --
class Brain:
    def __init__(self, cfg, store: MemoryStore, llm=None, tools=None):
        self.cfg = cfg
        self.store = store
        self.llm = llm
        self.tools = tools
        self.name = cfg.get("assistant.name", "Bi")
        self.user_name = cfg.get("assistant.user_name", "") or ""
        self.retriever = Retriever(store, cfg)
        self.extractor = FactExtractor(store, llm, bool(cfg.get("memory.extractor.enabled", True)))
        self.bg = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory-bg")
        self.extract_every = int(cfg.get("memory.extractor.every_n_turns", 6))
        self.session_timeout = float(cfg.get("memory.session_timeout_s", 300))
        self.max_history = int(cfg.get("llm.max_history_turns", 8))
        self.first_chunk = int(cfg.get("tts.first_chunk_min_chars", 24))
        self.session_id: str | None = None
        self.history: list[dict] = []
        self.turns_since_extract = 0
        self.last_activity = 0.0
        self.pending: PendingForget | PendingTool | None = None
        self.think_next = False
        self.last_stats = TurnStats()
        self._lock = threading.RLock()

    # ------------------------------------------------------------ sessions --
    def ensure_session(self, source: str = "live") -> str:
        now = time.time()
        if self.session_id and now - self.last_activity > self.session_timeout:
            self.end_session()
        if not self.session_id:
            self.session_id = self.store.start_session(source)
            self.history = []
            self.turns_since_extract = 0
            log.info("session %s started", self.session_id)
        self.last_activity = now
        return self.session_id

    def end_session(self, wait: bool = False) -> None:
        sid = self.session_id
        if not sid:
            return
        self.store.end_session(sid)
        fut = self.extractor.submit(sid)
        if wait and fut is not None:
            try:
                fut.result(timeout=30)
            except Exception as e:  # noqa: BLE001
                log.warning("final extraction did not finish: %s", e)
        self.session_id = None
        self.history = []
        self.pending = None

    def conversation_paused(self) -> None:
        """The voice loop went back to IDLE: extract facts from what was said so far."""
        if self.session_id and self.turns_since_extract:
            self.extractor.submit(self.session_id)
            self.turns_since_extract = 0

    def close(self) -> None:
        self.end_session(wait=True)
        self.bg.shutdown(wait=True)

    def prefetch(self) -> None:
        """Call as soon as the user stops speaking, while STT runs."""
        sid = self.session_id if (self.session_id and time.time() - self.last_activity <= self.session_timeout) else None
        self.retriever.prefetch(sid, first_turn=not self.history)

    # ---------------------------------------------------------------- turn --
    def respond(self, text: str, cancel: threading.Event | None = None) -> Iterator[Say | Action]:
        with self._lock:
            yield from self._respond(text.strip(), cancel)

    def _respond(self, text: str, cancel) -> Iterator[Say | Action]:
        t0 = time.perf_counter()
        self.last_stats = TurnStats()
        if not text:
            return
        sid = self.ensure_session()

        # 1. a pending confirmation gets first claim on the utterance
        if self.pending is not None:
            handled = yield from self._handle_pending(text, cancel)
            if handled:
                return

        # 2. commands that bypass the LLM
        cmd = parse_command(text)
        if cmd is not None and cmd.kind != "think":
            yield from self._command(cmd, text)
            return
        role = "chat"
        if cmd is not None and cmd.kind == "think":
            if not cmd.arg:
                self.think_next = True
                yield from self._say_and_log(text, "Okay, I'll think harder. What's the question?")
                return
            text, role = cmd.arg, "think"
        elif self.think_next:
            role, self.think_next = "think", False

        if self.llm is None:
            yield from self._say_and_log(text, "I don't have a language model configured, so I can only handle "
                                               "memory commands right now.")
            return

        # 3. memory retrieval (time-budgeted) -> system prompt
        ret = self.retriever.retrieve(text, sid, first_turn=not self.history)
        self.last_stats.retrieval_ms = ret.elapsed_ms
        system = self._system(ret)
        messages = self.history + [{"role": "user", "content": text}]
        yield from self._run_llm(messages, system, role, text, [], cancel, t0)

    def _system(self, ret: Retrieved) -> SystemPrompt:
        tzname = self.cfg.get("assistant.timezone") or ""
        now = dt.datetime.now().astimezone()
        return build_system(self.name, now, ret.facts, ret.snippets, ret.recent, self.user_name,
                            tzname or now.tzname() or "")

    # ------------------------------------------------------------ LLM loop --
    def _run_llm(self, messages: list[dict], system: SystemPrompt, role: str, user_text: str,
                 spoken: list[str], cancel, t0: float) -> Iterator[Say | Action]:
        stats = self.last_stats
        interrupted = False
        for _round in range(MAX_TOOL_ROUNDS):
            splitter = SentenceSplitter(self.first_chunk if not spoken else 10**6)
            done: Done | None = None
            try:
                for ev in self.llm.stream(messages, self.tools.specs() if self.tools else None, system,
                                          role=role, cancel=cancel):
                    if isinstance(ev, TextDelta):
                        if stats.first_token_ms is None:
                            stats.first_token_ms = (time.perf_counter() - t0) * 1000
                        for s in splitter.push(ev.text):
                            if stats.first_sentence_ms is None:
                                stats.first_sentence_ms = (time.perf_counter() - t0) * 1000
                            spoken.append(s)
                            yield Say(s)
                    elif isinstance(ev, Done):
                        done = ev
            except LLMConfigError as e:
                log.error("LLM config error: %s", e)
                msg = str(e) if len(str(e)) < 160 else "My language model isn't configured correctly. Run python -m bi.doctor."
                spoken.append(msg)
                yield Say(msg)
                break
            except LLMError as e:
                log.error("LLM error: %s", e)
                spoken.append(str(e))
                yield Say(str(e))
                break
            if done is None or done.stop_reason == "cancelled":
                interrupted = True
                break
            for s in splitter.flush():
                if stats.first_sentence_ms is None:
                    stats.first_sentence_ms = (time.perf_counter() - t0) * 1000
                spoken.append(s)
                yield Say(s)
            stats.model = done.model
            if done.stop_reason == "refusal":
                msg = "Sorry, I can't help with that one."
                spoken.append(msg)
                yield Say(msg)
                break
            if not done.tool_calls or self.tools is None:
                if done.stop_reason == "max_tokens" and not done.text:
                    spoken.append("Sorry, that answer got cut off.")
                    yield Say(spoken[-1])
                break
            messages = messages + [{"role": "assistant", "content": done.text, "raw": done.raw,
                                    "tool_calls": [{"id": c.id, "name": c.name, "input": c.input}
                                                   for c in done.tool_calls]}]
            confirm = [c for c in done.tool_calls if self.tools.requires_confirmation(c.name)]
            if confirm:
                question = self.tools.confirmation_prompt(done.tool_calls)
                self.pending = PendingTool(messages, system, role, done.tool_calls, user_text, spoken)
                spoken.append(question)
                yield Say(question)
                self._finish_turn(user_text, spoken, interrupted=False, t0=t0, keep_pending=True)
                return
            messages = messages + self._run_tools(done.tool_calls)
        self._finish_turn(user_text, spoken, interrupted, t0)

    def _run_tools(self, calls: list[ToolCall]) -> list[dict]:
        out = []
        for c in calls:
            content, is_error = self.tools.execute(c.name, c.input)
            self.last_stats.tools.append(c.name)
            out.append({"role": "tool", "tool_call_id": c.id, "name": c.name, "content": content, "is_error": is_error})
        return out

    def _handle_pending(self, text: str, cancel) -> Iterator[Say | Action]:
        p, self.pending = self.pending, None
        if time.time() - p.created > 120:
            return False
        yn = parse_yes_no(text)
        if isinstance(p, PendingForget):
            if yn == "yes":
                for fid in p.fact_ids:
                    f = self.store.get_fact(fid)
                    phrases = [f.text.rstrip("."), to_second_person(f.text)] if f else []
                    self.store.delete_fact(fid, cascade=True, scrub_sources=True, scrub_phrases=phrases)
                yield from self._say_and_log(text, "Done. I've forgotten that.")
                return True
            if yn == "no":
                yield from self._say_and_log(text, "Okay, I'll keep it.")
                return True
            return False
        # pending tool call(s)
        if yn == "yes":
            t0 = time.perf_counter()
            messages = p.messages + self._run_tools(p.calls)
            yield from self._run_llm(messages, p.system, p.role, text, [], cancel, t0)
            return True
        if yn == "no":
            yield from self._say_and_log(text, "Okay, I won't.")
            return True
        return False          # anything else: drop the pending action and treat as a new request

    # ------------------------------------------------------------ commands --
    def _command(self, cmd, text: str) -> Iterator[Say | Action]:
        k = cmd.kind
        if k == "stop":
            self._log_only(text)
            yield Action("stop")
        elif k == "cancel":
            self.pending = None
            self._log_only(text)
            yield Action("cancel")
        elif k == "sleep":
            yield from self._say_and_log(text, "Okay.")
            yield Action("sleep")
        elif k == "mute":
            yield from self._say_and_log(text, "Microphone off. Press M to turn it back on.")
            yield Action("mute")
        elif k == "remember":
            yield from self._remember(cmd.arg, text)
        elif k == "forget":
            yield from self._forget(cmd.arg, text)
        elif k == "recall":
            yield from self._recall(text)

    def _remember(self, what: str, text: str) -> Iterator[Say]:
        what = what.strip().rstrip(".!?")
        if len(what) < 3:
            yield from self._say_and_log(text, "What should I remember?")
            return
        fact = to_third_person(what)    # explicit requests bypass the secret/sensitive filter by design
        turn_id = self.store.add_turn(self.ensure_session(), "user", text, meta={"explicit_remember": True})
        fid, created = self.store.add_fact(fact, guess_category(fact), 0.95, source_turn_id=turn_id, explicit=True)
        reply = f"Got it. I'll remember that {to_second_person(fact)}." if created else "I already knew that, but thanks for confirming."
        self.store.add_turn(self.session_id, "assistant", reply)
        self._remember_exchange(text, reply)
        yield Say(reply)

    def _forget(self, what: str, text: str) -> Iterator[Say]:
        min_score = float(self.cfg.get("memory.retrieval.forget_min_score", 0.25))
        hits = [(f, s) for f, s in self.store.search_facts(what, k=5) if s >= min_score]
        if not hits:
            yield from self._say_and_log(text, "I couldn't find anything like that in my memory.")
            return
        top = hits[0][1]
        chosen = [f for f, s in hits if s >= top - 0.05][:3]
        said = " and ".join(to_second_person(f.text) for f in chosen)
        self.pending = PendingForget([f.id for f in chosen], text)
        n = "that" if len(chosen) == 1 else f"these {len(chosen)} things"
        yield from self._say_and_log(text, f"I'll forget {n}: {said}. Should I go ahead?")

    def _recall(self, text: str) -> Iterator[Say]:
        facts = self.store.list_facts()
        if not facts:
            yield from self._say_and_log(text, "I don't know much about you yet. Tell me about yourself, or say "
                                               "remember that, and I'll keep it in mind.")
            return
        facts.sort(key=lambda f: (-f.explicit, -f.confidence, -f.updated_at))
        top = facts[:25]
        reply = ""
        if self.llm is not None and len(top) > 3:
            try:
                reply = self.llm.complete(
                    "Facts:\n" + "\n".join(f"- {f.text}" for f in top) +
                    f"\n\nThere are {len(facts)} facts in total. In two or three spoken sentences, addressed to the "
                    "user as 'you', summarize what you know about them. Plain sentences, no lists or markdown.",
                    system="You are a voice assistant summarizing your memory of the user.", role="extractor",
                    max_tokens=400)
                reply = clean_for_speech(reply)
            except LLMError as e:
                log.warning("recall summary failed: %s", e)
        if not reply:
            parts = [to_second_person(f.text) for f in top[:4]]
            reply = f"I know {len(facts)} things about you. For example, " + "; ".join(parts) + "."
        yield from self._say_and_log(text, reply)

    # ----------------------------------------------------------- bookkeeping --
    def _log_only(self, user_text: str) -> None:
        self.store.add_turn(self.ensure_session(), "user", redact_secrets(user_text))

    def _say_and_log(self, user_text: str, reply: str) -> Iterator[Say]:
        sid = self.ensure_session()
        self.store.add_turn(sid, "user", redact_secrets(user_text))
        self.store.add_turn(sid, "assistant", reply)
        self._remember_exchange(user_text, reply)
        yield Say(reply)

    def _remember_exchange(self, user_text: str, reply: str) -> None:
        self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]
        self.history = self.history[-2 * self.max_history:]
        self.last_activity = time.time()

    def _finish_turn(self, user_text: str, spoken: list[str], interrupted: bool, t0: float,
                     keep_pending: bool = False) -> None:
        sid = self.ensure_session()
        reply = " ".join(spoken).strip()
        if interrupted:
            reply = (reply + " …[interrupted]").strip()
        user_id = self.store.add_turn(sid, "user", redact_secrets(user_text))
        asst_id = self.store.add_turn(sid, "assistant", reply or "(no reply)",
                                      meta={"model": self.last_stats.model, "tools": self.last_stats.tools} or None)
        self._remember_exchange(user_text, reply or "…")
        self.last_stats.total_ms = (time.perf_counter() - t0) * 1000
        chunk = f"User: {redact_secrets(user_text)}\n{self.name}: {reply}"
        self.bg.submit(self._index_chunk, sid, chunk, user_id, asst_id)
        self.turns_since_extract += 1
        if self.turns_since_extract >= self.extract_every:
            self.turns_since_extract = 0
            self.extractor.submit(sid)

    def _index_chunk(self, sid, text, first, last) -> None:
        try:
            self.store.add_chunk("turns", sid, text, first, last)
        except Exception as e:  # noqa: BLE001
            log.warning("chunk indexing failed: %s", e)

    def note_interrupted(self, heard: list[str]) -> None:
        """Barge-in: the user only heard part of the last reply; reflect that in the session history."""
        if self.history and self.history[-1]["role"] == "assistant":
            self.history[-1] = {"role": "assistant",
                                "content": (" ".join(heard) + " …[the user interrupted here]").strip()}
