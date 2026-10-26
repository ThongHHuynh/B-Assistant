"""Memory consolidation: merge duplicate facts, summarize old sessions, decay unused facts.

Runs nightly in-process (memory.consolidation.nightly_at) and on demand:
    python -m bi.memory consolidate
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from .store import MemoryStore

log = logging.getLogger(__name__)


@dataclass
class ConsolidationReport:
    merged: int = 0
    summarized: int = 0
    decayed: int = 0
    archived: int = 0

    def __str__(self) -> str:
        return (f"merged {self.merged} duplicate facts, summarized {self.summarized} sessions, "
                f"decayed {self.decayed}, archived {self.archived}")


SUMMARY_SYSTEM = ("You summarize a past conversation between a user and their voice assistant for the "
                  "assistant's long-term memory. Write 1-3 plain sentences in third person ('User asked...'), "
                  "keeping topics, decisions, names and anything the user may refer back to. No lists.")


def merge_duplicates(store: MemoryStore, threshold: float) -> int:
    facts = store.list_facts()
    if len(facts) < 2:
        return 0
    vecs = []
    keep = []
    for f in facts:
        v = store.index.get("fact", f.id)
        if v is not None and len(v) == store.embedder.dim:
            vecs.append(v)
            keep.append(f)
    if len(keep) < 2:
        return 0
    m = np.stack(vecs)
    sims = m @ m.T
    merged, dead = 0, set()
    # prefer explicit, then higher confidence, then most recent as the survivor
    order = sorted(range(len(keep)), key=lambda i: (-keep[i].explicit, -keep[i].confidence, -keep[i].updated_at))
    for a_pos, i in enumerate(order):
        if keep[i].id in dead:
            continue
        for j in order[a_pos + 1:]:
            if keep[j].id in dead or keep[i].category != keep[j].category:
                continue
            if sims[i, j] >= threshold:
                store.supersede(keep[j].id, keep[i].id)
                store.update_fact(keep[i].id, confidence=min(1.0, keep[i].confidence + 0.05))
                dead.add(keep[j].id)
                merged += 1
    return merged


def summarize_sessions(store: MemoryStore, llm, older_than_days: float, limit: int = 20) -> int:
    if llm is None:
        return 0
    cutoff = time.time() - older_than_days * 86400
    done = 0
    for s in store.sessions(unsummarized_before=cutoff)[:limit]:
        turns = store.session_turns(s["id"])
        if len(turns) < 2:
            store.set_session_summary(s["id"], "")
            continue
        transcript = "\n".join(f"{'User' if t.role == 'user' else 'Assistant'}: {t.text}" for t in turns)[-10000:]
        try:
            summary = llm.complete(transcript, system=SUMMARY_SYSTEM, role="extractor", max_tokens=300).strip()
        except Exception as e:  # noqa: BLE001
            log.warning("summary failed for %s: %s", s["id"], e)
            break
        if summary:
            day = dt.datetime.fromtimestamp(s["started_at"]).strftime("%b %d, %Y")
            store.set_session_summary(s["id"], summary)
            store.add_chunk("summary", s["id"], f"[{day}] {summary}", turns[0].id, turns[-1].id, ts=s["started_at"])
            done += 1
    return done


def decay(store: MemoryStore, after_days: float, factor: float, archive_below: float) -> tuple[int, int]:
    now = time.time()
    decayed = archived = 0
    for f in store.list_facts():
        if f.explicit or f.confidence >= 0.8:
            continue
        last = f.last_used_at or f.updated_at
        if now - last < after_days * 86400:
            continue
        new_conf = f.confidence * factor
        if new_conf < archive_below:
            store.archive_fact(f.id)
            archived += 1
        else:
            store.update_fact(f.id, confidence=new_conf)
            decayed += 1
    return decayed, archived


def consolidate(store: MemoryStore, llm, cfg) -> ConsolidationReport:
    c = cfg.section("memory.consolidation")
    r = ConsolidationReport()
    r.merged = merge_duplicates(store, float(c.get("duplicate_threshold", 0.90)))
    r.summarized = summarize_sessions(store, llm, float(c.get("summarize_after_days", 2)))
    r.decayed, r.archived = decay(store, float(c.get("decay_after_days", 30)), float(c.get("decay_factor", 0.85)),
                                  float(c.get("archive_below", 0.2)))
    store.meta_set("last_consolidation", str(time.time()))
    log.info("consolidation: %s", r)
    return r


class NightlyConsolidator(threading.Thread):
    """Runs consolidate() once a day at nightly_at (local time) while Bi is running."""

    def __init__(self, store: MemoryStore, llm, cfg):
        super().__init__(daemon=True, name="consolidator")
        self.store, self.llm, self.cfg = store, llm, cfg
        self.stop_event = threading.Event()
        hh, mm = (str(cfg.get("memory.consolidation.nightly_at", "03:30")) + ":0").split(":")[:2]
        self.at = dt.time(int(hh), int(mm))

    def seconds_until_next(self, now: dt.datetime | None = None) -> float:
        now = now or dt.datetime.now()
        target = now.replace(hour=self.at.hour, minute=self.at.minute, second=0, microsecond=0)
        if target <= now:
            target += dt.timedelta(days=1)
        return (target - now).total_seconds()

    def run(self) -> None:
        last = float(self.store.meta_get("last_consolidation", "0") or 0)
        if time.time() - last > 36 * 3600:          # missed a night (laptop was asleep): catch up soon
            if self.stop_event.wait(120):
                return
            self._run_once()
        while not self.stop_event.wait(self.seconds_until_next()):
            self._run_once()

    def _run_once(self) -> None:
        try:
            consolidate(self.store, self.llm, self.cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("nightly consolidation failed: %s", e)

    def stop(self) -> None:
        self.stop_event.set()
