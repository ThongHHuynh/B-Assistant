"""Vector index over the `embeddings` table.

The `embeddings` table (kind, ref_id, active, vec BLOB) is the source of truth.
sqlite-vec mirrors active rows into vec0 virtual tables for fast KNN; if the extension
can't load, an in-memory numpy matrix does exact cosine search (fine for personal scale).
"""
from __future__ import annotations

import logging
import sqlite3
import threading

import numpy as np

log = logging.getLogger(__name__)
KINDS = ("fact", "chunk")


class VectorIndex:
    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock, dim: int, backend: str = "auto"):
        self.conn, self.lock, self.dim = conn, lock, dim
        self.backend = "numpy"
        self._cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = {k: None for k in KINDS}
        if backend in ("auto", "sqlite-vec"):
            try:
                import sqlite_vec
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                self.backend = "sqlite-vec"
            except Exception as e:  # noqa: BLE001
                if backend == "sqlite-vec":
                    raise
                log.warning("sqlite-vec unavailable (%s); using numpy search", e)
        if self.backend == "sqlite-vec":
            self._ensure_vec_tables()

    def _ensure_vec_tables(self) -> None:
        with self.lock:
            for k in KINDS:
                row = self.conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (f"vec_{k}",)).fetchone()
                if row and f"float[{self.dim}]" not in row[0]:
                    self.conn.execute(f"DROP TABLE vec_{k}")
                    row = None
                if not row:
                    self.conn.execute(f"CREATE VIRTUAL TABLE vec_{k} USING vec0("
                                      f"embedding float[{self.dim}] distance_metric=cosine)")
                    self._mirror(k)
            self.conn.commit()

    def _mirror(self, kind: str) -> None:
        rows = self.conn.execute("SELECT ref_id, vec FROM embeddings WHERE kind=? AND active=1", (kind,)).fetchall()
        for ref_id, blob in rows:
            if len(blob) == self.dim * 4:
                self.conn.execute(f"INSERT OR REPLACE INTO vec_{kind}(rowid, embedding) VALUES (?, ?)", (ref_id, blob))

    def upsert(self, kind: str, ref_id: int, vec: np.ndarray) -> None:
        blob = np.asarray(vec, dtype=np.float32).tobytes()
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO embeddings(kind, ref_id, active, vec) VALUES (?,?,1,?)",
                              (kind, ref_id, blob))
            if self.backend == "sqlite-vec":
                self.conn.execute(f"DELETE FROM vec_{kind} WHERE rowid=?", (ref_id,))
                self.conn.execute(f"INSERT INTO vec_{kind}(rowid, embedding) VALUES (?, ?)", (ref_id, blob))
            self._cache[kind] = None

    def set_active(self, kind: str, ref_id: int, active: bool) -> None:
        with self.lock:
            self.conn.execute("UPDATE embeddings SET active=? WHERE kind=? AND ref_id=?", (int(active), kind, ref_id))
            if self.backend == "sqlite-vec":
                self.conn.execute(f"DELETE FROM vec_{kind} WHERE rowid=?", (ref_id,))
                if active:
                    row = self.conn.execute("SELECT vec FROM embeddings WHERE kind=? AND ref_id=?",
                                            (kind, ref_id)).fetchone()
                    if row:
                        self.conn.execute(f"INSERT INTO vec_{kind}(rowid, embedding) VALUES (?, ?)", (ref_id, row[0]))
            self._cache[kind] = None

    def delete(self, kind: str, ref_id: int) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM embeddings WHERE kind=? AND ref_id=?", (kind, ref_id))
            if self.backend == "sqlite-vec":
                self.conn.execute(f"DELETE FROM vec_{kind} WHERE rowid=?", (ref_id,))
            self._cache[kind] = None

    def get(self, kind: str, ref_id: int) -> np.ndarray | None:
        with self.lock:
            row = self.conn.execute("SELECT vec FROM embeddings WHERE kind=? AND ref_id=?", (kind, ref_id)).fetchone()
        return np.frombuffer(row[0], dtype=np.float32) if row else None

    def clear(self) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM embeddings")
            if self.backend == "sqlite-vec":
                for k in KINDS:
                    self.conn.execute(f"DELETE FROM vec_{k}")
            self._cache = {k: None for k in KINDS}

    def search(self, kind: str, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return [(ref_id, cosine_similarity)] for active rows, best first."""
        q = np.asarray(qvec, dtype=np.float32)
        if self.backend == "sqlite-vec":
            with self.lock:
                rows = self.conn.execute(
                    f"SELECT rowid, distance FROM vec_{kind} WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                    (q.tobytes(), int(k))).fetchall()
            return [(int(r), 1.0 - float(d)) for r, d in rows]
        ids, mat = self._matrix(kind)
        if len(ids) == 0:
            return []
        sims = mat @ q
        top = np.argsort(-sims)[:k]
        return [(int(ids[i]), float(sims[i])) for i in top]

    def _matrix(self, kind: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cache.get(kind)
        if cached is not None:
            return cached
        with self.lock:
            rows = self.conn.execute("SELECT ref_id, vec FROM embeddings WHERE kind=? AND active=1", (kind,)).fetchall()
        rows = [(r, b) for r, b in rows if len(b) == self.dim * 4]
        ids = np.array([r for r, _ in rows], dtype=np.int64)
        mat = (np.stack([np.frombuffer(b, dtype=np.float32) for _, b in rows]) if rows
               else np.zeros((0, self.dim), np.float32))
        self._cache[kind] = (ids, mat)
        return ids, mat
