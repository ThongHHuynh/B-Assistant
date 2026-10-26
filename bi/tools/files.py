"""Read-only file search over a configured list of folders (name + content match)."""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

log = logging.getLogger(__name__)
TEXT_EXT = {".txt", ".md", ".csv", ".json", ".py", ".yaml", ".yml", ".log", ".rst", ".html", ".tex", ".org"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "Library", "AppData", ".cache", ".Trash"}


def read_text(path: Path, max_chars: int = 200_000) -> str:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join((p.extract_text() or "") for p in reader.pages[:20])[:max_chars]
        if ext == ".docx":
            import docx
            return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)[:max_chars]
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars)
    except Exception as e:  # noqa: BLE001
        log.debug("can't read %s: %s", path, e)
        return ""


class FileSearcher:
    def __init__(self, c: dict):
        self.folders = [Path(os.path.expanduser(str(f))) for f in c.get("folders", [])]
        self.exts = {e.lower() for e in c.get("extensions", sorted(TEXT_EXT | {".pdf", ".docx"}))}
        self.max_results = int(c.get("max_results", 5))
        self.max_bytes = float(c.get("max_file_mb", 5)) * 1_000_000
        self.budget_s = float(c.get("time_budget_s", 4))
        self.max_files = int(c.get("max_files", 5000))

    def search(self, query: str) -> str:
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2] or query.lower().split()
        if not words:
            return "Empty query."
        existing = [f for f in self.folders if f.is_dir()]
        if not existing:
            return "No searchable folders are configured (tools.file_search.folders in config.yaml)."
        t0, scanned, hits = time.monotonic(), 0, []
        for root in existing:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
                for fn in filenames:
                    p = Path(dirpath) / fn
                    if p.suffix.lower() not in self.exts:
                        continue
                    scanned += 1
                    try:
                        if p.stat().st_size > self.max_bytes:
                            continue
                    except OSError:
                        continue
                    name = fn.lower()
                    name_score = sum(3 for w in words if w in name)
                    text = read_text(p)
                    low = text.lower()
                    content_score = sum(min(low.count(w), 5) for w in words)
                    covered = sum(1 for w in words if w in name or w in low)
                    if covered == 0:
                        continue
                    score = name_score + content_score + 4 * covered
                    hits.append((score, p, self._snippet(text, words)))
                    if time.monotonic() - t0 > self.budget_s or scanned >= self.max_files:
                        break
                if time.monotonic() - t0 > self.budget_s or scanned >= self.max_files:
                    break
        if not hits:
            return f"No files matched '{query}' (searched {scanned} files)."
        hits.sort(key=lambda h: -h[0])
        lines = [f"{p.name} in {p.parent}: {snip}" for _, p, snip in hits[: self.max_results]]
        return f"Top matches (searched {scanned} files):\n" + "\n".join(lines)

    @staticmethod
    def _snippet(text: str, words: list[str], width: int = 160) -> str:
        low = text.lower()
        pos = min((low.find(w) for w in words if w in low), default=-1)
        if pos < 0:
            return re.sub(r"\s+", " ", text[:width]).strip()
        start = max(0, pos - width // 3)
        return "…" + re.sub(r"\s+", " ", text[start:start + width]).strip() + "…"
