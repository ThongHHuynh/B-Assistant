"""Secret and sensitive-content detection for the memory layer.

Facts that contain secrets (passwords, API keys, card/ID numbers) or fall in sensitive
categories (health, religion, sexuality, politics, ...) are never stored by the automatic
extractor. An explicit "remember that ..." overrides the sensitive-category rule and the
secret rule (the user asked), but secrets are still redacted from the raw conversation log.
"""
from __future__ import annotations

import re

_SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("api key", re.compile(r"\b(sk-[A-Za-z0-9_\-]{16,}|sk-ant-[A-Za-z0-9_\-]{8,}|AKIA[0-9A-Z]{16}|"
                           r"gh[pousr]_[A-Za-z0-9]{20,}|xox[abpr]-[A-Za-z0-9\-]{10,}|AIza[0-9A-Za-z_\-]{30,})\b")),
    ("password", re.compile(r"(?i)\b(password|passcode|passphrase|pin(?: code)?|pass word)\b"
                            r"(?:\s+(?:is|was|:|=)|\s*[:=])\s*\S+")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\b")),
    ("id number", re.compile(r"(?i)\b(passport|social security|ssn|driver'?s? licen[cs]e|national id|"
                             r"id|tax id|routing|account|card)\s+(?:number|no\.?|#)\s*(?:is|:)?\s*[A-Z0-9\-]{5,}")),
    ("token", re.compile(r"\b(?=[A-Za-z0-9_\-]*\d)(?=[A-Za-z0-9_\-]*[A-Za-z])[A-Za-z0-9_\-]{32,}\b")),
]
_CARD = re.compile(r"\b(?:\d[ \-]?){13,19}\b")

_SENSITIVE: dict[str, list[str]] = {
    "health": ["diagnos", "disease", "disorder", "depress", "anxiety", "medication", "prescription", "therapy",
               "therapist", "cancer", "hiv", "diabetes", "pregnan", "mental health", "surgery", "illness",
               "adhd", "autism", "bipolar", "schizophren", "rehab", "addiction"],
    "religion": ["religion", "religious", "christian", "muslim", "islam", "jewish", "judaism", "hindu",
                 "buddhis", "atheist", "church", "mosque", "synagogue"],
    "sexuality": ["gay", "lesbian", "bisexual", "transgender", "sexual orientation", "queer", "sex life"],
    "politics": ["democrat", "republican", "political party", "voted for", "left-wing", "right-wing",
                 "conservative party", "labour party", "political view"],
    "ethnicity": ["ethnicity", "my race", "racial"],
    "legal": ["criminal record", "arrested", "convicted", "immigration status", "undocumented", "visa status"],
    "finance": ["salary", "bank account", "credit score", "debt", "net worth"],
}


def luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def find_secrets(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for label, pat in _SECRET_PATTERNS:
        for m in pat.finditer(text):
            found.append((label, m.group(0)))
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            found.append(("card number", m.group(0)))
    return found


def contains_secret(text: str) -> bool:
    return bool(find_secrets(text))


def redact_secrets(text: str) -> str:
    for label, value in sorted(find_secrets(text), key=lambda x: -len(x[1])):
        text = text.replace(value, f"[redacted {label}]")
    return text


def sensitive_category(text: str) -> str | None:
    low = text.lower()
    for cat, words in _SENSITIVE.items():
        for w in words:
            if re.search(r"\b" + re.escape(w), low):
                return cat
    return None


def allowed_to_store(text: str, explicit: bool) -> tuple[bool, str]:
    """(ok, reason). Explicit 'remember that' requests are always allowed."""
    if explicit:
        return True, ""
    if contains_secret(text):
        return False, "contains a secret"
    cat = sensitive_category(text)
    if cat:
        return False, f"sensitive ({cat})"
    return True, ""
