"""Per-turn memory retrieval under a strict time budget.

prefetch() runs while STT is still finishing (recent turns from earlier sessions, embedder warm);
retrieve() embeds the final transcript and ranks facts and snippets, returning whatever is ready
when the budget runs out instead of blocking the turn.
"""
from __future__ import annotations

import logging
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout
from dataclasses import dataclass, field

from .store import Fact, MemoryStore, Turn

log = logging.getLogger(__name__)


@dataclass
class Retrieved:
    facts: list[Fact] = field(default_factory=list)
    snippets: list[dict] = field(default_factory=list)
    recent: list[Turn] = field(default_factory=list)
    elapsed_ms: float = 0.0
    timed_out: bool = False


def rank_facts(cands: list[tuple[Fact, float]], now: float, min_score: float, k: int) -> list[tuple[Fact, float]]:
    """Blend semantic similarity with confidence, recency and explicitness."""
    scored = []
    for f, sim in cands:
        if not f.active or sim < min_score:
            continue
        age_days = max(0.0, (now - f.updated_at) / 86400)
        score = sim * (0.8 + 0.2 * f.confidence) + 0.04 * math.exp(-age_days / 30) + (0.03 if f.explicit else 0.0)
        scored.append((f, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


def rank_snippets(cands: list[tuple[dict, float]], now: float, min_score: float, k: int) -> list[tuple[dict, float]]:
    scored = []
    for c, sim in cands:
        if sim < min_score:
            continue
        age_days = max(0.0, (now - c["ts"]) / 86400)
        bonus = 0.03 if c["kind"] == "summary" else 0.0
        scored.append((c, sim + 0.03 * math.exp(-age_days / 14) + bonus))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


class Retriever:
    def __init__(self, store: MemoryStore, cfg):
        r = cfg.section("memory.retrieval")
        self.store = store
        self.budget_ms = float(r.get("budget_ms", 100))
        self.k_facts = int(r.get("top_k_facts", 6))
        self.k_snips = int(r.get("top_k_snippets", 3))
        self.min_fact = float(r.get("min_fact_score", 0.30))
        self.min_snip = float(r.get("min_snippet_score", 0.35))
        self.n_recent = int(r.get("recent_turns", 6))
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="retrieval")
        self._prefetch: Future | None = None
        self._prefetch_sid: str | None = None

    def prefetch(self, session_id: str | None, first_turn: bool) -> None:
        """Start loading context that doesn't depend on the transcript. Call when the user stops talking."""
        def work() -> list[Turn]:
            self.store.embed(["warm"])           # keeps the embedder hot (first-call latency)
            return self.store.recent_turns(self.n_recent, exclude_session=session_id) if first_turn else []
        self._prefetch_sid = session_id
        self._prefetch = self.pool.submit(work)

    def retrieve(self, query: str, session_id: str | None, first_turn: bool = True,
                 budget_ms: float | None = None) -> Retrieved:
        t0 = time.perf_counter()
        budget = (budget_ms if budget_ms is not None else self.budget_ms) / 1000
        if self._prefetch is None or self._prefetch_sid != session_id:
            self.prefetch(session_id, first_turn)
        pre = self._prefetch
        self._prefetch = None

        def search() -> tuple[list, list]:
            qvec = self.store.embed([query])[0]
            facts = self.store.search_facts(k=self.k_facts * 2, qvec=qvec)
            snips = self.store.search_chunks(k=self.k_snips * 2, qvec=qvec, exclude_session=session_id)
            return facts, snips

        fut = self.pool.submit(search)
        out = Retrieved()
        now = time.time()
        try:
            facts, snips = fut.result(timeout=max(0.0, budget - (time.perf_counter() - t0)))
            out.facts = [f for f, _ in rank_facts(facts, now, self.min_fact, self.k_facts)]
            out.snippets = [c for c, _ in rank_snippets(snips, now, self.min_snip, self.k_snips)]
        except FutTimeout:
            out.timed_out = True
            log.warning("retrieval exceeded %.0f ms budget; continuing without semantic results", budget * 1000)
        try:
            out.recent = pre.result(timeout=max(0.005, budget - (time.perf_counter() - t0))) if pre else []
        except FutTimeout:
            out.timed_out = True
        out.elapsed_ms = (time.perf_counter() - t0) * 1000
        if out.facts:
            self.pool.submit(self.store.touch_facts, [f.id for f in out.facts])
        return out
