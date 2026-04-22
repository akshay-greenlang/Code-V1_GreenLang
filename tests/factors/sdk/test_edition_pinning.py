# -*- coding: utf-8 -*-
"""Tests for edition-pinning ergonomics in the Factors Python SDK.

Covers
------
* :meth:`FactorsClient.pin_edition` returns a NEW client (immutable).
* The pinned client sends ``X-GreenLang-Edition`` on every request.
* When the server returns a mismatched edition, the client raises
  :class:`EditionMismatchError` — it does not silently accept the drift.
* The context manager (``with client.edition(...)``) restores the
  previous state (the outer client's pin is not mutated).
* Pin coverage spans resolve, resolve-explain, factor detail, search,
  audit bundle, bulk export / diff, quality — every route that carries
  the header.
* Async mirror behaves the same.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest

from greenlang.factors.sdk.python import (
    AsyncFactorsClient,
    EditionMismatchError,
    FactorsClient,
)

# --------------------------------------------------------------------------- #
# Mock transport helper.                                                       #
# --------------------------------------------------------------------------- #


def _build_mock(
    response_edition: Optional[str],
    *,
    path_contains: Optional[str] = None,
    body: Optional[Dict[str, Any]] = None,
) -> "MockCapture":
    """Build an httpx.MockTransport that records every request."""

    calls: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(
            {
                "method": request.method,
                "url": str(request.url),
                "headers": {k.lower(): v for k, v in request.headers.items()},
            }
        )
        if path_contains is not None:
            assert path_contains in str(request.url), (
                f"expected path containing {path_contains!r}, got {request.url}"
            )
        hdrs: Dict[str, str] = {}
        if response_edition is not None:
            hdrs["X-GreenLang-Edition"] = response_edition
        payload = body if body is not None else {"factors": [], "count": 0}
        return httpx.Response(200, json=payload, headers=hdrs)

    return MockCapture(transport=httpx.MockTransport(handler), calls=calls)


class MockCapture:
    def __init__(self, *, transport: httpx.MockTransport, calls: List[Dict[str, Any]]) -> None:
        self.transport = transport
        self.calls = calls

    def last(self) -> Dict[str, Any]:
        return self.calls[-1]

    def edition_headers(self) -> List[str]:
        return [
            c["headers"].get("x-greenlang-edition", "") for c in self.calls
        ]


def _make_client(
    mock: MockCapture,
    *,
    pinned_edition: Optional[str] = None,
) -> FactorsClient:
    return FactorsClient(
        base_url="https://api.greenlang.io",
        api_key="gl_test",
        verify_greenlang_cert=False,
        transport=mock.transport,
        pinned_edition=pinned_edition,
    )


# --------------------------------------------------------------------------- #
# pin_edition returns a NEW client, not a mutated one.                         #
# --------------------------------------------------------------------------- #


class TestPinEditionImmutability:
    def test_pin_edition_returns_new_client(self) -> None:
        mock = _build_mock("2027.Q1-electricity")
        base = _make_client(mock)
        pinned = base.pin_edition("2027.Q1-electricity")
        assert pinned is not base
        assert base.pinned_edition is None
        assert pinned.pinned_edition == "2027.Q1-electricity"

    def test_pin_edition_rejects_empty_string(self) -> None:
        mock = _build_mock("x")
        base = _make_client(mock)
        with pytest.raises(ValueError):
            base.pin_edition("")

    def test_pin_edition_rejects_non_string(self) -> None:
        mock = _build_mock("x")
        base = _make_client(mock)
        with pytest.raises(ValueError):
            base.pin_edition(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Header injection on every route.                                             #
# --------------------------------------------------------------------------- #


class TestPinnedHeaderInjection:
    EDITION = "2027.Q1-electricity"

    @pytest.mark.parametrize(
        "method_name, kwargs, path_fragment",
        [
            ("search", {"query": "diesel"}, "/factors/search"),
            ("search_v2", {"query": "diesel"}, "/factors/search/v2"),
            ("list_factors", {}, "/factors"),
            ("get_factor", {"factor_id": "ef_abc"}, "/factors/ef_abc"),
            (
                "resolve_explain",
                {"factor_id": "ef_abc"},
                "/factors/ef_abc/explain",
            ),
            (
                "audit_bundle",
                {"factor_id": "ef_abc"},
                "/factors/ef_abc/audit-bundle",
            ),
            (
                "diff",
                {
                    "factor_id": "ef_abc",
                    "left_edition": "2026.Q4",
                    "right_edition": "2027.Q1",
                },
                "/factors/ef_abc/diff",
            ),
            ("coverage", {}, "/factors/coverage"),
            ("list_editions", {}, "/editions"),
            ("list_sources", {}, "/factors/source-registry"),
        ],
    )
    def test_get_routes_carry_edition_header(
        self,
        method_name: str,
        kwargs: Dict[str, Any],
        path_fragment: str,
    ) -> None:
        body: Dict[str, Any]
        if method_name == "get_factor":
            body = {"factor_id": "ef_abc", "co2e_per_unit": 1.0, "unit": "kg"}
        elif method_name == "resolve_explain":
            body = {"factor_id": "ef_abc"}
        elif method_name == "diff":
            body = {
                "factor_id": "ef_abc",
                "left": {},
                "right": {},
                "changes": [],
            }
        elif method_name == "audit_bundle":
            body = {"factor_id": "ef_abc", "evidence": []}
        elif method_name == "coverage":
            body = {"total_factors": 0}
        elif method_name == "list_editions":
            body = {"editions": []}
        elif method_name == "list_sources":
            body = {"sources": []}
        else:
            body = {"factors": [], "count": 0}

        mock = _build_mock(self.EDITION, body=body, path_contains=path_fragment)
        client = _make_client(mock).pin_edition(self.EDITION)

        method: Callable[..., Any] = getattr(client, method_name)
        method(**kwargs)

        assert mock.calls, "request was not sent"
        assert mock.last()["headers"]["x-greenlang-edition"] == self.EDITION

    def test_post_routes_carry_edition_header(self) -> None:
        # resolve (POST) and match (POST) and resolve_batch (POST).
        mock = _build_mock(
            self.EDITION,
            body={"factor_id": "ef_abc"},
        )
        client = _make_client(mock).pin_edition(self.EDITION)
        client.resolve({"activity": "diesel"})
        assert mock.last()["headers"]["x-greenlang-edition"] == self.EDITION

        mock2 = _build_mock(self.EDITION, body={"candidates": []})
        client2 = _make_client(mock2).pin_edition(self.EDITION)
        client2.match("diesel")
        assert mock2.last()["headers"]["x-greenlang-edition"] == self.EDITION

        mock3 = _build_mock(
            self.EDITION, body={"job_id": "job_123", "status": "queued"}
        )
        client3 = _make_client(mock3).pin_edition(self.EDITION)
        client3.resolve_batch([{"activity": "diesel"}])
        assert mock3.last()["headers"]["x-greenlang-edition"] == self.EDITION

    def test_unpinned_client_does_not_send_header(self) -> None:
        mock = _build_mock(self.EDITION)
        client = _make_client(mock)
        client.search("diesel")
        # The transport-level X-Factors-Edition is untouched; our
        # X-GreenLang-Edition header is only sent by pinned clients.
        assert "x-greenlang-edition" not in mock.last()["headers"]


# --------------------------------------------------------------------------- #
# EditionMismatchError on drift.                                               #
# --------------------------------------------------------------------------- #


class TestEditionMismatch:
    def test_mismatched_response_raises(self) -> None:
        mock = _build_mock("2027.Q2-electricity")  # server drift
        client = _make_client(mock).pin_edition("2027.Q1-electricity")
        with pytest.raises(EditionMismatchError) as excinfo:
            client.search("diesel")
        err = excinfo.value
        assert err.pinned_edition == "2027.Q1-electricity"
        assert err.returned_edition == "2027.Q2-electricity"
        assert "2027.Q1-electricity" in str(err)

    def test_matching_edition_does_not_raise(self) -> None:
        mock = _build_mock("2027.Q1-electricity")
        client = _make_client(mock).pin_edition("2027.Q1-electricity")
        # No exception expected.
        client.search("diesel")
        assert mock.calls

    def test_missing_response_header_is_tolerated(self) -> None:
        # When the server omits the edition header, we cannot prove drift
        # and must allow the request through — pinning only fails on
        # explicit mismatch.
        mock = _build_mock(None)
        client = _make_client(mock).pin_edition("2027.Q1-electricity")
        client.search("diesel")  # no raise

    def test_error_context_preserves_path(self) -> None:
        mock = _build_mock("bad-edition")
        client = _make_client(mock).pin_edition("good-edition")
        with pytest.raises(EditionMismatchError) as excinfo:
            client.get_factor("ef_abc")
        ctx = excinfo.value.context
        assert "path" in ctx
        assert ctx["path"].endswith("/factors/ef_abc")


# --------------------------------------------------------------------------- #
# Context-manager restores previous pin.                                       #
# --------------------------------------------------------------------------- #


class TestEditionContextManager:
    def test_parent_pin_unchanged_after_context(self) -> None:
        mock = _build_mock("2027.Q1")
        parent = _make_client(mock).pin_edition("base-edition")
        assert parent.pinned_edition == "base-edition"
        with parent.edition("override-edition") as scoped:
            assert scoped.pinned_edition == "override-edition"
            assert parent.pinned_edition == "base-edition"
        # After the block, parent's pin must be untouched.
        assert parent.pinned_edition == "base-edition"

    def test_unpinned_parent_stays_unpinned(self) -> None:
        mock = _build_mock("scoped-edition")
        parent = _make_client(mock)
        assert parent.pinned_edition is None
        with parent.edition("scoped-edition") as scoped:
            assert scoped.pinned_edition == "scoped-edition"
        assert parent.pinned_edition is None

    def test_context_sends_scoped_header(self) -> None:
        mock = _build_mock("scoped-edition")
        parent = _make_client(mock)
        with parent.edition("scoped-edition") as scoped:
            scoped.search("diesel")
        assert mock.last()["headers"]["x-greenlang-edition"] == "scoped-edition"

    def test_context_raises_on_drift(self) -> None:
        mock = _build_mock("drift-edition")
        parent = _make_client(mock)
        with pytest.raises(EditionMismatchError):
            with parent.edition("scoped-edition") as scoped:
                scoped.search("diesel")


# --------------------------------------------------------------------------- #
# Async mirror.                                                                #
# --------------------------------------------------------------------------- #


def _build_async_mock(
    response_edition: Optional[str],
    *,
    body: Optional[Dict[str, Any]] = None,
) -> MockCapture:
    calls: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(
            {
                "method": request.method,
                "url": str(request.url),
                "headers": {k.lower(): v for k, v in request.headers.items()},
            }
        )
        hdrs: Dict[str, str] = {}
        if response_edition is not None:
            hdrs["X-GreenLang-Edition"] = response_edition
        payload = body if body is not None else {"factors": [], "count": 0}
        return httpx.Response(200, json=payload, headers=hdrs)

    return MockCapture(transport=httpx.MockTransport(handler), calls=calls)


class TestAsyncEditionPinning:
    def test_async_pin_header_sent(self) -> None:
        mock = _build_async_mock("2027.Q1")

        async def _run() -> None:
            client = AsyncFactorsClient(
                base_url="https://api.greenlang.io",
                api_key="gl_test",
                verify_greenlang_cert=False,
                transport=mock.transport,
            )
            try:
                pinned = client.pin_edition("2027.Q1")
                await pinned.search("diesel")
            finally:
                await client.aclose()

        asyncio.run(_run())
        assert mock.last()["headers"]["x-greenlang-edition"] == "2027.Q1"

    def test_async_mismatch_raises(self) -> None:
        mock = _build_async_mock("drift")

        async def _run() -> None:
            client = AsyncFactorsClient(
                base_url="https://api.greenlang.io",
                api_key="gl_test",
                verify_greenlang_cert=False,
                transport=mock.transport,
            )
            try:
                pinned = client.pin_edition("good")
                with pytest.raises(EditionMismatchError):
                    await pinned.search("diesel")
            finally:
                await client.aclose()

        asyncio.run(_run())

    def test_async_context_manager_restores_pin(self) -> None:
        mock = _build_async_mock("scoped-edition")

        async def _run() -> None:
            client = AsyncFactorsClient(
                base_url="https://api.greenlang.io",
                api_key="gl_test",
                verify_greenlang_cert=False,
                transport=mock.transport,
            )
            try:
                assert client.pinned_edition is None
                async with client.edition("scoped-edition") as scoped:
                    assert scoped.pinned_edition == "scoped-edition"
                    await scoped.search("diesel")
                assert client.pinned_edition is None
            finally:
                await client.aclose()

        asyncio.run(_run())
        assert mock.last()["headers"]["x-greenlang-edition"] == "scoped-edition"
