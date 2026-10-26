"""SQLite memory store: raw log, facts, chunks, notes, reminders, tool log.

One database file (data/bi.db). Layers:
  1. turns     - every turn (role, text, ts, session_id); never deleted automatically
  2. facts     - durable facts/preferences; contradictions are superseded, not deleted
  3. chunks    - conversation snippets and session summaries, embedded for semantic search
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .embed import Embedder
from .vector import VectorIndex

log = logging.getLogger(__name__)

CATEGORIES = ("preference", "person", "project", "habit", "other")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, started_at REAL NOT NULL, ended_at REAL, source TEXT DEFAULT 'live',
    title TEXT, summary TEXT, extracted_upto INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL,
    text TEXT NOT NULL, ts REAL NOT NULL, meta TEXT);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, id);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'other', source_turn_id INTEGER,
    created_at REAL NOT NULL, updated_at REAL NOT NULL, confidence REAL NOT NULL DEFAULT 0.7,
    superseded_by INTEGER, explicit INTEGER NOT NULL DEFAULT 0, archived INTEGER NOT NULL DEFAULT 0,
    last_used_at REAL, use_count INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS facts_active ON facts(superseded_by, archived);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, session_id TEXT, text TEXT NOT NULL,
    first_turn_id INTEGER, last_turn_id INTEGER, ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS embeddings (
    kind TEXT NOT NULL, ref_id INTEGER NOT NULL, active INTEGER NOT NULL DEFAULT 1, vec BLOB NOT NULL,
    PRIMARY KEY (kind, ref_id));
CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, due_at REAL NOT NULL, created_at REAL NOT NULL,
    fired_at REAL, cancelled INTEGER NOT NULL DEFAULT 0, kind TEXT DEFAULT 'reminder');
CREATE TABLE IF NOT EXISTS tool_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, tool TEXT NOT NULL, args TEXT, result TEXT,
    status TEXT, side_effect INTEGER, network INTEGER);
"""


@dataclass
class Fact:
    id: int
    text: str
    category: str
    source_turn_id: int | None
    created_at: float
    updated_at: float
    confidence: float
    superseded_by: int | None
    explicit: bool
    archived: bool
    last_used_at: float | None
    use_count: int

    @property
    def active(self) -> bool:
        return self.superseded_by is None and not self.archived

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Turn:
    id: int
    session_id: str
    role: str
    text: str
    ts: float


_FACT_COLS = ("id, text, category, source_turn_id, created_at, updated_at, confidence, superseded_by, "
              "explicit, archived, last_used_at, use_count")


def _fact(row) -> Fact:
    return Fact(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], bool(row[8]), bool(row[9]),
                row[10], row[11])


class MemoryStore:
    def __init__(self, db_path: Path | str, embedder: Embedder, index_backend: str = "auto",
                 dup_threshold: float = 0.92):
        self.path = Path(db_path)
        if str(db_path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.embedder = embedder
        self.dup_threshold = dup_threshold
        self.index = VectorIndex(self.conn, self.lock, embedder.dim, index_backend)
        self._check_embedder()

    # ------------------------------------------------------------ internals --
    def _q(self, sql: str, args: Iterable[Any] = ()) -> list:
        with self.lock:
            return self.conn.execute(sql, tuple(args)).fetchall()

    def _x(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self.lock:
            return self.conn.execute(sql, tuple(args))

    def meta_get(self, key: str, default: str | None = None) -> str | None:
        r = self._q("SELECT value FROM meta WHERE key=?", (key,))
        return r[0][0] if r else default

    def meta_set(self, key: str, value: str) -> None:
        self._x("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    def _check_embedder(self) -> None:
        sig = f"{self.embedder.name}:{self.embedder.dim}"
        old = self.meta_get("embedder")
        if old != sig:
            n = self._q("SELECT COUNT(*) FROM embeddings")[0][0]
            if old and n:
                log.warning("embedder changed (%s -> %s); re-indexing %d vectors", old, sig, n)
                self.reindex()
            self.meta_set("embedder", sig)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.embedder.embed(texts)

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    # ------------------------------------------------------------- sessions --
    def start_session(self, source: str = "live", started_at: float | None = None, title: str | None = None) -> str:
        sid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self._x("INSERT INTO sessions(id, started_at, source, title) VALUES (?,?,?,?)",
                (sid, started_at or time.time(), source, title))
        return sid

    def end_session(self, sid: str) -> None:
        self._x("UPDATE sessions SET ended_at=? WHERE id=? AND ended_at IS NULL", (time.time(), sid))

    def session(self, sid: str) -> dict | None:
        r = self._q("SELECT id, started_at, ended_at, source, title, summary, extracted_upto FROM sessions WHERE id=?",
                    (sid,))
        if not r:
            return None
        keys = ("id", "started_at", "ended_at", "source", "title", "summary", "extracted_upto")
        return dict(zip(keys, r[0]))

    def set_session_summary(self, sid: str, summary: str) -> None:
        self._x("UPDATE sessions SET summary=? WHERE id=?", (summary, sid))

    def set_extracted_upto(self, sid: str, turn_id: int) -> None:
        self._x("UPDATE sessions SET extracted_upto=MAX(extracted_upto, ?) WHERE id=?", (turn_id, sid))

    def sessions(self, unsummarized_before: float | None = None) -> list[dict]:
        sql = "SELECT id, started_at, ended_at, summary FROM sessions"
        args: list = []
        if unsummarized_before is not None:
            sql += " WHERE summary IS NULL AND started_at < ?"
            args.append(unsummarized_before)
        sql += " ORDER BY started_at"
        return [dict(zip(("id", "started_at", "ended_at", "summary"), r)) for r in self._q(sql, args)]

    # ---------------------------------------------------------------- turns --
    def add_turn(self, sid: str, role: str, text: str, ts: float | None = None, meta: dict | None = None) -> int:
        cur = self._x("INSERT INTO turns(session_id, role, text, ts, meta) VALUES (?,?,?,?,?)",
                      (sid, role, text, ts or time.time(), json.dumps(meta) if meta else None))
        return int(cur.lastrowid)

    def session_turns(self, sid: str, after_id: int = 0) -> list[Turn]:
        return [Turn(*r) for r in self._q(
            "SELECT id, session_id, role, text, ts FROM turns WHERE session_id=? AND id>? ORDER BY id", (sid, after_id))]

    def recent_turns(self, n: int, exclude_session: str | None = None) -> list[Turn]:
        rows = self._q("SELECT id, session_id, role, text, ts FROM turns WHERE session_id != ? ORDER BY id DESC LIMIT ?",
                       (exclude_session or "", n))
        return [Turn(*r) for r in reversed(rows)]

    def turn(self, turn_id: int) -> Turn | None:
        r = self._q("SELECT id, session_id, role, text, ts FROM turns WHERE id=?", (turn_id,))
        return Turn(*r[0]) if r else None

    def redact_turn(self, turn_id: int, replacement: str = "[forgotten at user's request]") -> None:
        self._x("UPDATE turns SET text=? WHERE id=?", (replacement, turn_id))

    # ---------------------------------------------------------------- facts --
    def add_fact(self, text: str, category: str = "other", confidence: float = 0.7,
                 source_turn_id: int | None = None, explicit: bool = False,
                 supersedes: Iterable[int] = (), now: float | None = None) -> tuple[int, bool]:
        """Insert a fact (or reinforce a near-duplicate). Returns (fact_id, created)."""
        text = text.strip().rstrip(".") + "."
        category = category if category in CATEGORIES else "other"
        now = now or time.time()
        vec = self.embed([text])[0]
        supersedes = [s for s in supersedes if s]
        if not supersedes:
            for fid, sim in self.index.search("fact", vec, 3):
                if sim >= self.dup_threshold:
                    f = self.get_fact(fid)
                    if f and f.active:
                        self._x("UPDATE facts SET updated_at=?, confidence=MIN(1.0, MAX(confidence, ?) + 0.05), "
                                "explicit=MAX(explicit, ?) WHERE id=?", (now, confidence, int(explicit), fid))
                        return fid, False
        with self.lock:
            cur = self._x("INSERT INTO facts(text, category, source_turn_id, created_at, updated_at, confidence, explicit) "
                          "VALUES (?,?,?,?,?,?,?)", (text, category, source_turn_id, now, now,
                                                     max(0.0, min(1.0, confidence)), int(explicit)))
            fid = int(cur.lastrowid)
            self.index.upsert("fact", fid, vec)
            for old in supersedes:
                self.supersede(old, fid)
        return fid, True

    def supersede(self, old_id: int, new_id: int) -> None:
        if old_id == new_id:
            return
        with self.lock:
            self._x("UPDATE facts SET superseded_by=?, updated_at=? WHERE id=? AND superseded_by IS NULL",
                    (new_id, time.time(), old_id))
            self.index.set_active("fact", old_id, False)

    def get_fact(self, fid: int) -> Fact | None:
        r = self._q(f"SELECT {_FACT_COLS} FROM facts WHERE id=?", (fid,))
        return _fact(r[0]) if r else None

    def list_facts(self, include_inactive: bool = False, category: str | None = None) -> list[Fact]:
        sql = f"SELECT {_FACT_COLS} FROM facts"
        conds, args = [], []
        if not include_inactive:
            conds.append("superseded_by IS NULL AND archived=0")
        if category:
            conds.append("category=?")
            args.append(category)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY category, updated_at DESC"
        return [_fact(r) for r in self._q(sql, args)]

    def update_fact(self, fid: int, text: str | None = None, category: str | None = None,
                    confidence: float | None = None) -> bool:
        f = self.get_fact(fid)
        if not f:
            return False
        with self.lock:
            if text is not None:
                text = text.strip().rstrip(".") + "."
                self._x("UPDATE facts SET text=? WHERE id=?", (text, fid))
                self.index.upsert("fact", fid, self.embed([text])[0])
                if not f.active:
                    self.index.set_active("fact", fid, False)
            if category is not None:
                self._x("UPDATE facts SET category=? WHERE id=?", (category if category in CATEGORIES else "other", fid))
            if confidence is not None:
                self._x("UPDATE facts SET confidence=? WHERE id=?", (max(0.0, min(1.0, confidence)), fid))
            self._x("UPDATE facts SET updated_at=? WHERE id=?", (time.time(), fid))
        return True

    def archive_fact(self, fid: int) -> None:
        with self.lock:
            self._x("UPDATE facts SET archived=1, updated_at=? WHERE id=?", (time.time(), fid))
            self.index.set_active("fact", fid, False)

    def delete_fact(self, fid: int, cascade: bool = True, scrub_sources: bool = False) -> list[int]:
        """Delete a fact completely: the row, its vector, and (cascade) the older versions it superseded.
        scrub_sources also redacts the raw-log turns the facts came from and drops their snippets."""
        with self.lock:
            ids, frontier = [fid], [fid]
            while cascade and frontier:
                nxt = [r[0] for r in self._q(
                    f"SELECT id FROM facts WHERE superseded_by IN ({','.join('?' * len(frontier))})", frontier)]
                nxt = [i for i in nxt if i not in ids]
                ids += nxt
                frontier = nxt
            sources = [r[0] for r in self._q(
                f"SELECT source_turn_id FROM facts WHERE id IN ({','.join('?' * len(ids))}) AND source_turn_id IS NOT NULL",
                ids)]
            for i in ids:
                self.index.delete("fact", i)
                self._x("DELETE FROM facts WHERE id=?", (i,))
            # facts that pointed at a deleted fact become active again only if nothing else replaced them
            self._x(f"UPDATE facts SET superseded_by=NULL WHERE superseded_by IN ({','.join('?' * len(ids))})", ids)
            if scrub_sources:
                for tid in set(sources):
                    self.redact_turn(tid)
                    for (cid,) in self._q("SELECT id FROM chunks WHERE ? BETWEEN first_turn_id AND last_turn_id", (tid,)):
                        self.delete_chunk(cid)
        return ids

    def touch_facts(self, ids: Iterable[int]) -> None:
        ids = list(ids)
        if ids:
            self._x(f"UPDATE facts SET last_used_at=?, use_count=use_count+1 WHERE id IN ({','.join('?' * len(ids))})",
                    [time.time(), *ids])

    def search_facts(self, query: str | None = None, k: int = 5, qvec: np.ndarray | None = None
                     ) -> list[tuple[Fact, float]]:
        if qvec is None:
            qvec = self.embed([query or ""])[0]
        out = []
        for fid, sim in self.index.search("fact", qvec, k * 3):
            f = self.get_fact(fid)
            if f and f.active:
                out.append((f, sim))
        return out[:k]

    # --------------------------------------------------------------- chunks --
    def add_chunk(self, kind: str, sid: str | None, text: str, first_turn_id: int | None = None,
                  last_turn_id: int | None = None, ts: float | None = None, vec: np.ndarray | None = None) -> int:
        with self.lock:
            cur = self._x("INSERT INTO chunks(kind, session_id, text, first_turn_id, last_turn_id, ts) VALUES (?,?,?,?,?,?)",
                          (kind, sid, text, first_turn_id, last_turn_id, ts or time.time()))
            cid = int(cur.lastrowid)
            self.index.upsert("chunk", cid, self.embed([text])[0] if vec is None else vec)
        return cid

    def delete_chunk(self, cid: int) -> None:
        with self.lock:
            self.index.delete("chunk", cid)
            self._x("DELETE FROM chunks WHERE id=?", (cid,))

    def get_chunk(self, cid: int) -> dict | None:
        r = self._q("SELECT id, kind, session_id, text, first_turn_id, last_turn_id, ts FROM chunks WHERE id=?", (cid,))
        keys = ("id", "kind", "session_id", "text", "first_turn_id", "last_turn_id", "ts")
        return dict(zip(keys, r[0])) if r else None

    def search_chunks(self, query: str | None = None, k: int = 3, qvec: np.ndarray | None = None,
                      exclude_session: str | None = None) -> list[tuple[dict, float]]:
        if qvec is None:
            qvec = self.embed([query or ""])[0]
        out = []
        for cid, sim in self.index.search("chunk", qvec, k * 3):
            c = self.get_chunk(cid)
            if c and (exclude_session is None or c["session_id"] != exclude_session):
                out.append((c, sim))
        return out[:k]

    # --------------------------------------------------------- notes/reminders --
    def add_note(self, text: str) -> int:
        return int(self._x("INSERT INTO notes(text, created_at) VALUES (?,?)", (text, time.time())).lastrowid)

    def notes(self, limit: int = 20) -> list[dict]:
        return [{"id": r[0], "text": r[1], "created_at": r[2]}
                for r in self._q("SELECT id, text, created_at FROM notes ORDER BY id DESC LIMIT ?", (limit,))]

    def add_reminder(self, text: str, due_at: float, kind: str = "reminder") -> int:
        return int(self._x("INSERT INTO reminders(text, due_at, created_at, kind) VALUES (?,?,?,?)",
                           (text, due_at, time.time(), kind)).lastrowid)

    def pending_reminders(self) -> list[dict]:
        return [{"id": r[0], "text": r[1], "due_at": r[2], "kind": r[3]} for r in self._q(
            "SELECT id, text, due_at, kind FROM reminders WHERE fired_at IS NULL AND cancelled=0 ORDER BY due_at")]

    def mark_reminder_fired(self, rid: int) -> None:
        self._x("UPDATE reminders SET fired_at=? WHERE id=?", (time.time(), rid))

    def cancel_reminder(self, rid: int) -> bool:
        return self._x("UPDATE reminders SET cancelled=1 WHERE id=? AND fired_at IS NULL AND cancelled=0",
                       (rid,)).rowcount > 0

    def log_tool(self, tool: str, args: dict, result: str, status: str, side_effect: bool, network: bool) -> None:
        self._x("INSERT INTO tool_log(ts, tool, args, result, status, side_effect, network) VALUES (?,?,?,?,?,?,?)",
                (time.time(), tool, json.dumps(args)[:2000], (result or "")[:2000], status, int(side_effect), int(network)))

    # ---------------------------------------------------------------- admin --
    def reindex(self) -> int:
        with self.lock:
            self.index.clear()
            facts = self._q("SELECT id, text, superseded_by, archived FROM facts")
            chunks = self._q("SELECT id, text FROM chunks")
            n = 0
            for batch_start in range(0, len(facts), 64):
                batch = facts[batch_start:batch_start + 64]
                vecs = self.embed([r[1] for r in batch])
                for (fid, _, sup, arch), v in zip(batch, vecs):
                    self.index.upsert("fact", fid, v)
                    if sup is not None or arch:
                        self.index.set_active("fact", fid, False)
                    n += 1
            for batch_start in range(0, len(chunks), 64):
                batch = chunks[batch_start:batch_start + 64]
                for (cid, _), v in zip(batch, self.embed([r[1] for r in batch])):
                    self.index.upsert("chunk", cid, v)
                    n += 1
            self.meta_set("embedder", f"{self.embedder.name}:{self.embedder.dim}")
        return n

    def stats(self) -> dict:
        q = lambda s: self._q(s)[0][0]  # noqa: E731
        return {"sessions": q("SELECT COUNT(*) FROM sessions"), "turns": q("SELECT COUNT(*) FROM turns"),
                "facts_active": q("SELECT COUNT(*) FROM facts WHERE superseded_by IS NULL AND archived=0"),
                "facts_total": q("SELECT COUNT(*) FROM facts"), "chunks": q("SELECT COUNT(*) FROM chunks"),
                "notes": q("SELECT COUNT(*) FROM notes"),
                "reminders_pending": q("SELECT COUNT(*) FROM reminders WHERE fired_at IS NULL AND cancelled=0"),
                "vector_index": self.index.backend, "embedder": self.embedder.name}

    def export(self) -> dict:
        return {
            "exported_at": time.time(),
            "facts": [f.to_dict() for f in self.list_facts(include_inactive=True)],
            "sessions": self.sessions(),
            "turns": [dict(zip(("id", "session_id", "role", "text", "ts"), r))
                      for r in self._q("SELECT id, session_id, role, text, ts FROM turns ORDER BY id")],
            "notes": self.notes(10_000),
            "reminders": [dict(zip(("id", "text", "due_at", "created_at", "fired_at", "cancelled"), r))
                          for r in self._q("SELECT id, text, due_at, created_at, fired_at, cancelled FROM reminders")],
        }


def open_store(cfg, embedder: Embedder | None = None) -> MemoryStore:
    from .embed import make_embedder
    emb = embedder or make_embedder(cfg)
    return MemoryStore(cfg.path_of("paths.db", "data/bi.db"), emb,
                       cfg.get("memory.vector_index", "auto"),
                       float(cfg.get("memory.consolidation.duplicate_threshold", 0.92)))
