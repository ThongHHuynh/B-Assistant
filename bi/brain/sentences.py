"""Streaming sentence splitter for sentence-by-sentence TTS.

Feed token deltas with push(); complete sentences come out as soon as they end. The very
first chunk may be cut early at a clause boundary (comma, colon, dash) once it is long
enough, so speech can start before the first full sentence has streamed in.
Markdown that would sound odd when read aloud is stripped.
"""
from __future__ import annotations

import re

_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "approx", "no",
           "mt", "ft", "inc", "ltd", "co", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
           "oct", "nov", "dec", "u.s", "a.m", "p.m"}
_END = re.compile(r"([.!?…]+)([\"')\]]*)(\s+|$)")
_CLAUSE = re.compile(r"([,;:—–]|\s-)\s+")


def clean_for_speech(text: str) -> str:
    t = re.sub(r"```.*?```", " ", text, flags=re.S)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"(\*\*|__|\*|_{1,2})(\S(?:.*?\S)?)\1", r"\2", t)
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)
    t = re.sub(r"(?m)^\s*(?:[-*+•]|\d+[.)])\s+", "", t)
    t = re.sub(r"(?m)^\s*>\s?", "", t)
    t = t.replace("&", " and ").replace("|", " ")
    return re.sub(r"\s+", " ", t).strip()


class SentenceSplitter:
    def __init__(self, first_chunk_min_chars: int = 24, max_chars: int = 220):
        self.buf = ""
        self.first_min = first_chunk_min_chars
        self.max_chars = max_chars
        self.emitted = 0

    def push(self, delta: str) -> list[str]:
        self.buf += delta
        out: list[str] = []
        while True:
            s = self._take()
            if s is None:
                break
            s = clean_for_speech(s)
            if s:
                out.append(s)
                self.emitted += 1
        return out

    def flush(self) -> list[str]:
        rest, self.buf = clean_for_speech(self.buf), ""
        if rest:
            self.emitted += 1
            return [rest]
        return []

    def _take(self) -> str | None:
        buf = self.buf
        # a paragraph break always ends a chunk
        nl = buf.find("\n\n")
        for m in _END.finditer(buf):
            end = m.end(2)
            if nl != -1 and nl < m.start():
                break
            if m.group(3) == "" :
                return None          # punctuation at buffer end: wait for more (could be "3.5" or "e.g.")
            word = re.findall(r"([A-Za-z.]+)$", buf[:m.start()])
            if m.group(1) == "." and word and word[0].lower().rstrip(".") in _ABBREV:
                continue
            if m.group(1) == "." and re.search(r"\b[A-Z]$", buf[:m.start()]):
                continue             # initials like "J. Smith"
            if len(buf[:end].strip()) < 2:
                continue
            self.buf = buf[m.end():]
            return buf[:end]
        if nl != -1:
            self.buf = buf[nl + 2:]
            return buf[:nl]
        if self.emitted == 0 and len(buf) >= self.first_min:
            for m in _CLAUSE.finditer(buf):
                if m.start() >= self.first_min * 0.6 and len(buf) - m.end() >= 1:
                    self.buf = buf[m.end():]
                    return buf[:m.start() + 1].rstrip()
        if len(buf) > self.max_chars:
            cut = buf.rfind(" ", 0, self.max_chars)
            cut = cut if cut > 40 else self.max_chars
            self.buf = buf[cut:].lstrip()
            return buf[:cut]
        return None


def split_sentences(text: str, first_chunk_min_chars: int = 10**6) -> list[str]:
    sp = SentenceSplitter(first_chunk_min_chars)
    return sp.push(text) + sp.flush()
