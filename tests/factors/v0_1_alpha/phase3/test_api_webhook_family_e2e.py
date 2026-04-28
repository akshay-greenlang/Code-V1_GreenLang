# -*- coding: utf-8 -*-
"""Phase 3 / Wave 2.5 — api_webhook family end-to-end test.

Covers:

  * **PACT API path** — :class:`ApiFetcher` driven against an injected
    transport returns a synthetic PACT response; the parser produces 5
    factor records that match the committed golden snapshot.
  * **Webhook path** — POST to the webhook router (via
    :class:`fastapi.testclient.TestClient`) stores the artifact + run;
    the manual replay endpoint replays the artifact through the
    pipeline successfully.
  * **HMAC failure** — POST with an invalid signature returns 401 and
    persists nothing.
  * **Idempotency** — the same event id POSTed twice returns
    ``duplicate=true`` on the second call without creating a second
    run.
  * **Pagination** — :class:`ApiFetcher` follows ``X-Next-Cursor``
    across two mock pages and merges the bodies.

The suite is hermetic: zero network calls, no external dependencies.

References
----------
- ``docs/factors/PHASE_3_PLAN.md`` §"Fetcher / parser families"
  (Block 3, api_webhook).
- ``greenlang/factors/ingestion/webhook.py`` — the receiver under test.
- ``greenlang/factors/ingestion/fetchers.py`` — the ApiFetcher /
  WebhookReplayFetcher pair under test.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.factors.ingestion.fetchers import (
    ApiFetcher,
    RetryPolicy,
    WebhookReplayFetcher,
)
from greenlang.factors.ingestion.parsers._phase3_api_adapters import (
    PHASE3_PACT_PARSER_VERSION,
    PHASE3_PRIVATE_PACK_PARSER_VERSION,
    Phase3PactApiParser,
    Phase3PrivatePackParser,
)
from greenlang.factors.ingestion.webhook import (
    HEADER_API_VERSION,
    HEADER_APPROVER,
    HEADER_EVENT_ID,
    HEADER_SIGNATURE,
    WebhookArtifactStore,
    WebhookIdempotencyCache,
    WebhookRunRecorder,
    build_webhook_router,
    verify_signature,
)
from tests.factors.v0_1_alpha.phase3.parser_snapshots._helper import (
    compare_to_snapshot,
    regenerate_if_env,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_FIXTURE_DIR: Path = Path(__file__).resolve().parent / "fixtures"
_PACT_FIXTURE: Path = _FIXTURE_DIR / "pact_api_response_mini.json"
_PRIVATE_PACK_FIXTURE: Path = _FIXTURE_DIR / "private_pack_response_mini.json"
_WEBHOOK_EVENT_FIXTURE: Path = _FIXTURE_DIR / "webhook_event_mini.json"

_VOLATILE_TOP = {"published_at"}
_VOLATILE_EXTRACTION = {"ingested_at"}
_VOLATILE_REVIEW = {"reviewed_at", "approved_at"}


def _scrub(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop volatile timestamp fields so the snapshot is deterministic."""
    out: List[Dict[str, Any]] = []
    for r in records:
        c = dict(r)
        for k in _VOLATILE_TOP:
            c.pop(k, None)
        if isinstance(c.get("extraction"), dict):
            ext = dict(c["extraction"])
            for k in _VOLATILE_EXTRACTION:
                ext.pop(k, None)
            c["extraction"] = ext
        if isinstance(c.get("review"), dict):
            rev = dict(c["review"])
            for k in _VOLATILE_REVIEW:
                rev.pop(k, None)
            c["review"] = rev
        out.append(c)
    return out


def _make_pact_transport(
    body: bytes,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Return a single-page transport callable + a request log list."""
    log: List[Dict[str, Any]] = []

    def _transport(*, method: str, url: str, headers: Dict[str, str],
                   params: Dict[str, str]) -> Tuple[int, Dict[str, str], bytes]:
        log.append({"method": method, "url": url, "params": dict(params)})
        return 200, {"content-type": "application/json"}, body

    return _transport, log


# ---------------------------------------------------------------------------
# 1. PACT API path — ApiFetcher + parser + snapshot.
# ---------------------------------------------------------------------------


def test_pact_api_fetch_parse_matches_snapshot() -> None:
    """ApiFetcher drives synthetic PACT JSON; parser output matches golden."""
    pact_body = _PACT_FIXTURE.read_bytes()
    transport, _log = _make_pact_transport(pact_body)
    fetcher = ApiFetcher(
        base_url="https://api.partner.example/pact/footprints",
        auth_method="bearer",
        credentials_resolver=lambda: {"token": "fake-token"},
        transport=transport,
        retry_policy=RetryPolicy(max_attempts=1, backoff_s=0.0),
    )
    result = fetcher.fetch_paginated()
    assert result.pages_followed == 1
    assert result.sha256 == hashlib.sha256(pact_body).hexdigest()
    assert result.body_bytes == pact_body

    parser = Phase3PactApiParser()
    records = parser.parse_bytes(
        result.body_bytes,
        artifact_uri="file://pact_api.json",
        artifact_sha256="0" * 64,
    )
    assert len(records) == 5
    assert records[0]["urn"].startswith("urn:gl:factor:phase3-alpha:pact:")
    assert records[0]["value"] == 8.45

    scrubbed = _scrub(records)
    regenerate_if_env("pact_api", PHASE3_PACT_PARSER_VERSION, scrubbed)
    compare_to_snapshot("pact_api", PHASE3_PACT_PARSER_VERSION, scrubbed)


def test_private_pack_parser_matches_snapshot() -> None:
    """Private-pack parser shape against committed golden snapshot."""
    body = _PRIVATE_PACK_FIXTURE.read_bytes()
    parser = Phase3PrivatePackParser()
    records = parser.parse_bytes(
        body,
        artifact_uri="file://private_pack.json",
        artifact_sha256="0" * 64,
    )
    assert len(records) == 5
    assert records[0]["urn"].startswith("urn:gl:factor:phase3-alpha:private-pack:")
    assert records[0]["value"] == 2.685

    scrubbed = _scrub(records)
    regenerate_if_env("private_pack", PHASE3_PRIVATE_PACK_PARSER_VERSION, scrubbed)
    compare_to_snapshot("private_pack", PHASE3_PRIVATE_PACK_PARSER_VERSION, scrubbed)


# ---------------------------------------------------------------------------
# 2. Pagination — ApiFetcher follows X-Next-Cursor across two pages.
# ---------------------------------------------------------------------------


def test_api_fetcher_follows_pagination_cursor() -> None:
    """ApiFetcher follows ``X-Next-Cursor`` header across 2 mock pages."""
    page1 = json.dumps({
        "pagination": {"next_cursor": "cur-page-2"},
        "data": [{"id": "p1-r1", "value": 1.1}],
    }).encode("utf-8")
    page2 = json.dumps({
        "pagination": {"next_cursor": None},
        "data": [{"id": "p2-r1", "value": 2.2}],
    }).encode("utf-8")
    pages = [page1, page2]
    log: List[Dict[str, Any]] = []

    def _transport(*, method: str, url: str, headers: Dict[str, str],
                   params: Dict[str, str]) -> Tuple[int, Dict[str, str], bytes]:
        idx = len(log)
        log.append({"params": dict(params)})
        body = pages[idx]
        return 200, {"content-type": "application/json"}, body

    fetcher = ApiFetcher(
        base_url="https://api.partner.example/pact/footprints",
        transport=_transport,
        retry_policy=RetryPolicy(max_attempts=1, backoff_s=0.0),
    )
    result = fetcher.fetch_paginated()
    assert result.pages_followed == 2
    # First call has no cursor; second carries the cursor from page 1.
    assert log[0]["params"].get("cursor") in (None, "")
    assert log[1]["params"]["cursor"] == "cur-page-2"
    # Aggregated body contains BOTH page payloads.
    assert b"p1-r1" in result.body_bytes
    assert b"p2-r1" in result.body_bytes


# ---------------------------------------------------------------------------
# 3. Webhook path — TestClient round-trip + replay.
# ---------------------------------------------------------------------------


@pytest.fixture()
def hermetic_webhook_app(monkeypatch: pytest.MonkeyPatch) -> Tuple[FastAPI, TestClient,
                                                                   WebhookArtifactStore,
                                                                   WebhookRunRecorder,
                                                                   str]:
    """Return (app, client, artifact_store, run_recorder, secret) for tests.

    The router uses a fresh per-test artifact store + idempotency cache
    so test ordering does not leak. The shared HMAC secret is set via
    monkeypatch so :func:`_resolve_secret` finds it at request time.
    """
    artifact_store = WebhookArtifactStore()
    idempotency = WebhookIdempotencyCache()
    run_recorder = WebhookRunRecorder()
    source_id = "acme-private-pack-2024"
    secret = "wave2-5-test-secret"
    monkeypatch.setenv(
        "GL_FACTORS_WEBHOOK_SECRET_ACME_PRIVATE_PACK_2024", secret,
    )
    router = build_webhook_router(
        artifact_store=artifact_store,
        idempotency=idempotency,
        run_recorder=run_recorder,
    )
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    return app, client, artifact_store, run_recorder, secret


def _sign(body: bytes, secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"), msg=body, digestmod=hashlib.sha256,
    ).hexdigest()


def test_webhook_post_creates_artifact_and_run(hermetic_webhook_app) -> None:
    """Successful POST stores artifact + creates run row."""
    _app, client, artifact_store, run_recorder, secret = hermetic_webhook_app
    body = _WEBHOOK_EVENT_FIXTURE.read_bytes()
    sig = _sign(body, secret)
    resp = client.post(
        "/v1/factors/ingest/webhook/acme-private-pack-2024",
        content=body,
        headers={
            HEADER_SIGNATURE: sig,
            HEADER_EVENT_ID: "evt-wave25-001",
            HEADER_API_VERSION: "2024-04",
            "content-type": "application/json",
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["duplicate"] is False
    assert payload["run_id"]
    assert payload["artifact_id"]
    assert payload["sha256"] == hashlib.sha256(body).hexdigest()

    # Side-effect assertions: artifact + run actually persisted.
    assert len(artifact_store) == 1
    artifact = artifact_store.get(payload["artifact_id"])
    assert artifact is not None
    assert artifact.event_id == "evt-wave25-001"
    assert artifact.api_version == "2024-04"
    assert len(run_recorder) == 1
    run_row = run_recorder.get(payload["run_id"])
    assert run_row is not None
    assert run_row["status"] == "created"
    assert run_row["source_id"] == "acme-private-pack-2024"


def test_webhook_replay_drives_pipeline(hermetic_webhook_app) -> None:
    """Replay endpoint finds the stored artifact and creates a new run row."""
    _app, client, artifact_store, run_recorder, secret = hermetic_webhook_app
    body = _WEBHOOK_EVENT_FIXTURE.read_bytes()
    sig = _sign(body, secret)
    resp1 = client.post(
        "/v1/factors/ingest/webhook/acme-private-pack-2024",
        content=body,
        headers={
            HEADER_SIGNATURE: sig,
            HEADER_EVENT_ID: "evt-wave25-002",
            HEADER_API_VERSION: "2024-04",
        },
    )
    assert resp1.status_code == 200
    artifact_id = resp1.json()["artifact_id"]

    # Replay against the stored artifact.
    resp2 = client.post(
        "/v1/factors/ingest/webhook/acme-private-pack-2024/replay",
        json={"artifact_id": artifact_id},
        headers={HEADER_APPROVER: "human:approver@greenlang.io"},
    )
    assert resp2.status_code == 200, resp2.text
    payload = resp2.json()
    assert payload["artifact_id"] == artifact_id
    assert payload["approver"] == "human:approver@greenlang.io"
    assert payload["run_id"] != payload["original_run_id"]
    # Two run rows now: original + replay.
    assert len(run_recorder) == 2

    # The WebhookReplayFetcher must read back the same bytes the parser
    # would consume, end-to-end, without going through the HTTP layer.
    replay_fetcher = WebhookReplayFetcher(artifact_store)
    body_back = replay_fetcher.fetch(artifact_id)
    assert body_back == body
    parser = Phase3PrivatePackParser()
    records = parser.parse_bytes(
        body_back,
        artifact_uri="webhook://%s" % artifact_id,
        artifact_sha256=hashlib.sha256(body).hexdigest(),
    )
    assert len(records) == 1
    assert records[0]["urn"] == "urn:gl:factor:phase3-alpha:private-pack:ac-fuel-001:v1"


# ---------------------------------------------------------------------------
# 4. HMAC failure — wrong signature -> 401, no side effects.
# ---------------------------------------------------------------------------


def test_webhook_post_rejects_invalid_signature(hermetic_webhook_app) -> None:
    """Invalid HMAC -> 401 and zero side effects."""
    _app, client, artifact_store, run_recorder, _secret = hermetic_webhook_app
    body = _WEBHOOK_EVENT_FIXTURE.read_bytes()
    resp = client.post(
        "/v1/factors/ingest/webhook/acme-private-pack-2024",
        content=body,
        headers={
            HEADER_SIGNATURE: "0" * 64,  # wrong digest
            HEADER_EVENT_ID: "evt-wave25-bad",
        },
    )
    assert resp.status_code == 401
    # Artifact + run stores remain empty.
    assert len(artifact_store) == 0
    assert len(run_recorder) == 0


# ---------------------------------------------------------------------------
# 5. Idempotency — same event id twice returns duplicate=true on second call.
# ---------------------------------------------------------------------------


def test_webhook_idempotent_on_duplicate_event_id(hermetic_webhook_app) -> None:
    """Re-POSTing the same event id within the window returns duplicate=true."""
    _app, client, artifact_store, run_recorder, secret = hermetic_webhook_app
    body = _WEBHOOK_EVENT_FIXTURE.read_bytes()
    sig = _sign(body, secret)
    headers = {
        HEADER_SIGNATURE: sig,
        HEADER_EVENT_ID: "evt-wave25-dup",
    }
    resp1 = client.post(
        "/v1/factors/ingest/webhook/acme-private-pack-2024",
        content=body, headers=headers,
    )
    assert resp1.status_code == 200
    assert resp1.json()["duplicate"] is False
    first_run_id = resp1.json()["run_id"]

    resp2 = client.post(
        "/v1/factors/ingest/webhook/acme-private-pack-2024",
        content=body, headers=headers,
    )
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["duplicate"] is True
    assert body2["run_id"] == first_run_id
    # Only one artifact + one run despite two POSTs.
    assert len(artifact_store) == 1
    assert len(run_recorder) == 1


# ---------------------------------------------------------------------------
# 6. Smoke — verify_signature is constant-time and matches /rejects.
# ---------------------------------------------------------------------------


def test_verify_signature_constant_time_match_and_reject() -> None:
    secret = "abc123"
    body = b"hello-wave-2-5"
    good = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert verify_signature(body=body, signature=good, secret=secret) is True
    assert verify_signature(body=body, signature="0" * 64, secret=secret) is False
    assert verify_signature(body=body, signature="", secret=secret) is False


# ---------------------------------------------------------------------------
# 7. Router shape — at least 2 routes (receive + replay).
# ---------------------------------------------------------------------------


def test_webhook_router_exposes_at_least_two_routes() -> None:
    """The acceptance check ``len(webhook_router.routes) >= 2`` holds."""
    from greenlang.factors.ingestion.webhook import webhook_router

    assert len([r for r in webhook_router.routes]) >= 2
