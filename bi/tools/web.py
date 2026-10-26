"""Pluggable web search providers: duckduckgo (no key), brave (BRAVE_API_KEY), none."""
from __future__ import annotations

import html
import logging
import os
import re
from typing import Callable
from urllib.parse import parse_qs, unquote, urlparse

import httpx

log = logging.getLogger(__name__)
UA = "Mozilla/5.0 (compatible; Bi-assistant/0.1; +https://github.com/)"


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s))).strip()


def _format(results: list[dict], n: int) -> str:
    if not results:
        return "No results found."
    return "\n".join(f"{i}. {r['title']} - {r['snippet'][:300]} ({r['url']})" for i, r in enumerate(results[:n], 1))


def duckduckgo(query: str, n: int = 3, timeout: float = 6.0) -> str:
    r = httpx.post("https://html.duckduckgo.com/html/", data={"q": query}, headers={"User-Agent": UA},
                   timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return _format(parse_duckduckgo(r.text), n)


def parse_duckduckgo(page: str) -> list[dict]:
    out = []
    blocks = re.split(r'<div class="result results_links', page)[1:]
    for b in blocks:
        a = re.search(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', b, re.S)
        sn = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', b, re.S)
        if not a:
            continue
        url = html.unescape(a.group(1))
        if "duckduckgo.com/l/" in url:
            url = unquote(parse_qs(urlparse(url).query).get("uddg", [url])[0])
        if "duckduckgo.com/y.js" in url:      # ads
            continue
        out.append({"title": _clean(a.group(2)), "snippet": _clean(sn.group(1)) if sn else "", "url": url})
    return out


def brave(query: str, n: int = 3, timeout: float = 6.0) -> str:
    key = os.environ.get("BRAVE_API_KEY")
    if not key:
        return "Web search isn't configured: set BRAVE_API_KEY in .env."
    r = httpx.get("https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": n},
                  headers={"X-Subscription-Token": key, "Accept": "application/json"}, timeout=timeout)
    r.raise_for_status()
    items = r.json().get("web", {}).get("results", [])
    return _format([{"title": _clean(i.get("title", "")), "snippet": _clean(i.get("description", "")),
                     "url": i.get("url", "")} for i in items], n)


PROVIDERS: dict[str, Callable[..., str]] = {"duckduckgo": duckduckgo, "brave": brave}


def make_search(c: dict) -> Callable[[str], str] | None:
    name = (c.get("provider") or "duckduckgo").lower()
    if name == "none":
        return None
    fn = PROVIDERS.get(name)
    if fn is None:
        log.warning("unknown web_search provider %s; web search disabled", name)
        return None
    n, timeout = int(c.get("max_results", 3)), float(c.get("timeout_s", 6))

    def search(query: str) -> str:
        try:
            return fn(query, n, timeout)
        except httpx.HTTPError as e:
            return f"Web search failed ({type(e).__name__}). I may be offline."
    return search
