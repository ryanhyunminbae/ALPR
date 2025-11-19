import types

import pytest

from app import websearch


def test_search_disabled_by_default(monkeypatch):
    monkeypatch.delenv("PLATE_WEB_SEARCH_ENABLED", raising=False)
    monkeypatch.delenv("ENABLE_WEB_SEARCH", raising=False)
    results = websearch.search_public_info("ABC123")
    assert results == []


def test_search_returns_parsed_results(monkeypatch):
    monkeypatch.setenv("PLATE_WEB_SEARCH_ENABLED", "1")

    def fake_get(url, params, headers, timeout):
        text = """
        <div class="result">
            <a rel="nofollow" class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fplate">Plate report</a>
            <a class="result__snippet">Sample snippet for the plate.</a>
        </div>
        """
        response = types.SimpleNamespace(
            text=text,
            status_code=200,
            raise_for_status=lambda: None,
        )
        return response

    monkeypatch.setattr(websearch.requests, "get", fake_get)
    results = websearch.search_public_info("ABC123", max_results=1)
    assert results == [
        {
            "title": "Plate report",
            "url": "https://example.com/plate",
            "snippet": "Sample snippet for the plate.",
        }
    ]

