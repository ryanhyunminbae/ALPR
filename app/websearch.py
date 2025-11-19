"""
Lightweight web search helper for demo purposes.

The functions in this module perform a simple DuckDuckGo HTML query to gather
publicly available links that mention the recognized license plate text. The
search is disabled by default for privacy/safety reasons and can be enabled by
setting the environment variable ``PLATE_WEB_SEARCH_ENABLED=1``.

This feature is intended strictly for educational demonstrations. Results are
scraped from third-party sites and may change at any time.
"""

from __future__ import annotations

import html
import os
import re
from typing import List, Dict
from urllib.parse import parse_qs, unquote, urlparse

import requests

DUCKDUCKGO_HTML_ENDPOINT = "https://duckduckgo.com/html/"
RESULT_PATTERN = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
    r'<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL,
)


def _is_enabled() -> bool:
    flag = os.getenv("PLATE_WEB_SEARCH_ENABLED", os.getenv("ENABLE_WEB_SEARCH", "0"))
    return flag.lower() in {"1", "true", "yes", "on"}


def _clean_url(url: str) -> str:
    """
    DuckDuckGo wraps outgoing links in redirect URLs. Attempt to unwrap them.
    """
    parsed = urlparse(url)
    if (not parsed.netloc or parsed.netloc.endswith("duckduckgo.com")) and parsed.path.startswith("/l/"):
        qs = parse_qs(parsed.query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    return url


def search_public_info(plate_text: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Search the web for publicly available mentions of a license plate.

    Parameters
    ----------
    plate_text:
        Normalized license plate text.
    max_results:
        Maximum number of search results to return.
    """
    if not plate_text or not _is_enabled():
        return []

    query = f'"license plate" "{plate_text}"'

    try:
        response = requests.get(
            DUCKDUCKGO_HTML_ENDPOINT,
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (ALPR Demo)"},
            timeout=8,
        )
        response.raise_for_status()
    except Exception:
        return []

    matches = RESULT_PATTERN.finditer(response.text)
    results: List[Dict[str, str]] = []

    for match in matches:
        url = _clean_url(match.group("url"))
        title = html.unescape(re.sub(r"<.*?>", "", match.group("title")))
        snippet = html.unescape(re.sub(r"<.*?>", "", match.group("snippet")))
        results.append({"title": title.strip(), "url": url.strip(), "snippet": snippet.strip()})
        if len(results) >= max_results:
            break

    return results

