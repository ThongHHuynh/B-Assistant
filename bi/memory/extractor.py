"""Background fact extraction from conversation transcripts (cheap/fast model)."""
from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor

from .privacy import allowed_to_store
from .store import CATEGORIES, MemoryStore, Turn

log = logging.getLogger(__name__)

SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "confidence": {"type": "number"},
                    "supersedes": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "category", "confidence", "supersedes"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}

SYSTEM = """You maintain a personal assistant's long-term memory about its user. From a conversation \
transcript, extract durable facts worth remembering for future conversations: preferences, people in the \
user's life, ongoing projects, habits and routines, and stable personal details (name, where they live, job).

Rules:
- Each fact is one short, standalone sentence in third person starting with "User", e.g. "User's sister is \
named Anna." or "User prefers tea over coffee."
- Only facts stated or clearly implied by the USER. Ignore what the assistant said, one-off requests, \
small talk and anything transient (today's weather, a timer they set).
- Never extract secrets (passwords, PINs, API keys, card, bank, passport or ID numbers) or sensitive \
details (health, religion, sexuality, politics, ethnicity, finances, legal status) unless the user explicitly \
asked to have it remembered.
- You get the existing memory as "[id] text". Skip facts already known. If a new fact updates or contradicts \
an existing one (moved city, changed preference, new job), output the new fact and list the old ids in \
"supersedes".
- confidence: 0.9 for explicit statements, 0.6 for reasonable inferences, lower for guesses.
- Return {"facts": []} when there is nothing durable. Quality over quantity."""


def _parse(text: str) -> list[dict]:
    text = text.strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    facts = data.get("facts", []) if isinstance(data, dict) else []
    out = []
    for f in facts:
        if not isinstance(f, dict) or not isinstance(f.get("text"), str) or not f["text"].strip():
            continue
        try:
            conf = float(f.get("confidence", 0.7))
        except (TypeError, ValueError):
            conf = 0.7
        sup = [int(s) for s in f.get("supersedes", []) or [] if isinstance(s, (int, float, str)) and str(s).isdigit()]
        out.append({"text": f["text"].strip(), "category": f.get("category", "other"),
                    "confidence": max(0.0, min(1.0, conf)), "supersedes": sup})
    return out


class FactExtractor:
    def __init__(self, store: MemoryStore, llm, enabled: bool = True):
        self.store, self.llm, self.enabled = store, llm, enabled
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="extractor")
        self._lock = threading.Lock()

    def submit(self, session_id: str) -> Future | None:
        if not self.enabled or self.llm is None:
            return None
        return self.pool.submit(self._safe_run, session_id)

    def _safe_run(self, session_id: str) -> list[int]:
        try:
            return self.run(session_id)
        except Exception as e:  # noqa: BLE001 - background job must not crash the assistant
            log.warning("fact extraction failed for %s: %s", session_id, e)
            return []

    def existing_context(self, turns: list[Turn], limit: int = 30) -> list:
        facts = self.store.list_facts()
        if len(facts) <= limit:
            return facts
        seen, picked = set(), []
        user_texts = [t.text for t in turns if t.role == "user"][-12:]
        for vec in self.store.embed(user_texts) if user_texts else []:
            for f, _ in self.store.search_facts(k=5, qvec=vec):
                if f.id not in seen:
                    seen.add(f.id)
                    picked.append(f)
        return picked[:limit]

    def run(self, session_id: str) -> list[int]:
        with self._lock:
            sess = self.store.session(session_id)
            if not sess:
                return []
            turns = self.store.session_turns(session_id, after_id=sess["extracted_upto"] or 0)
            if not any(t.role == "user" for t in turns):
                return []
            return self.extract_from_turns(session_id, turns)

    def extract_from_turns(self, session_id: str, turns: list[Turn]) -> list[int]:
        existing = self.existing_context(turns)
        known = "\n".join(f"[{f.id}] {f.text}" for f in existing) or "(empty)"
        transcript = "\n".join(f"{'User' if t.role == 'user' else 'Assistant'}: {t.text}" for t in turns)
        prompt = f"Existing memory:\n{known}\n\nTranscript:\n{transcript[-12000:]}\n\nExtract the facts as JSON."
        raw = self.llm.complete(prompt, system=SYSTEM, role="extractor", json_schema=SCHEMA, max_tokens=1500)
        facts = _parse(raw)
        valid_ids = {f.id for f in existing}
        user_turns = [t for t in turns if t.role == "user"]
        added = []
        for f in facts:
            ok, why = allowed_to_store(f["text"], explicit=False)
            if not ok:
                log.info("extractor skipped a fact (%s)", why)
                continue
            source = self._best_source(f["text"], user_turns)
            fid, created = self.store.add_fact(f["text"], f["category"], f["confidence"], source_turn_id=source,
                                               supersedes=[s for s in f["supersedes"] if s in valid_ids])
            if created:
                added.append(fid)
        if turns:
            self.store.set_extracted_upto(session_id, max(t.id for t in turns))
        log.info("extractor: %d new facts from %d turns", len(added), len(turns))
        return added

    def _best_source(self, fact: str, user_turns: list[Turn]) -> int | None:
        if not user_turns:
            return None
        if len(user_turns) == 1:
            return user_turns[0].id
        vecs = self.store.embed([fact] + [t.text for t in user_turns])
        sims = vecs[1:] @ vecs[0]
        return user_turns[int(sims.argmax())].id
