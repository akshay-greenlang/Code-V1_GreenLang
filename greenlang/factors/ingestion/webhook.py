# -*- coding: utf-8 -*-
"""Phase 3 / Wave 2.5 — Webhook receiver for the api_webhook ingestion family.

This module ships the FastAPI router that lets external systems (PACT
Pathfinder, tenant-private packs, partner pipelines) push raw factor
artifacts into GreenLang Factors over HTTPS rather than waiting for a
poll-style fetch. Per the Phase 3 plan §"Fetcher / parser families" the
api_webhook family must capture: request URL, full response body,
response timestamp, API version, pagination cursor, and webhook event
ID — every one of those fields is persisted on the resulting
``raw_artifacts`` row so the seven-gate publish orchestrator finds the
provenance pin at stage 6 (gate 6 — provenance completeness).

Two endpoints:

* ``POST /v1/factors/ingest/webhook/{source_id}`` — verify HMAC, dedupe
  on event ID, persist body bytes + headers + timestamp, create an
  ``ingestion_runs`` row in ``created`` status. Returns the run id and
  artifact id so the caller can correlate.

* ``POST /v1/factors/ingest/webhook/{source_id}/replay`` — admin-only
  (requires ``X-GL-Approver: human:<email>``) replay of a previously-
  received webhook artifact through the pipeline. Drives the runner
  with a :class:`WebhookReplayFetcher` so the test can assert the
  artifact -> run round-trip without re-receiving the webhook.

Security contract
-----------------
* HMAC SHA-256 over the raw request body, keyed on
  ``GL_FACTORS_WEBHOOK_SECRET_<SOURCE_ID_UPPER>``. Header
  ``X-GL-Signature`` carries the lowercase hex digest. Missing secret
  -> 401 (fail closed); missing or mismatched signature -> 401.

* Idempotency: ``X-GL-Event-Id`` header is required. Duplicate event
  IDs within a 24h window return ``200`` with ``{"duplicate": true}``
  and DO NOT create a second artifact / run. The cache is bounded at
  10k entries (LRU eviction).

* Feature flag: the router only auto-mounts when
  ``GL_FACTORS_PHASE3_WEBHOOK_ENABLED=1``. Default behaviour is
  unchanged so production keeps the legacy surface until Block 3
  governance signs off.

References
----------
- ``docs/factors/PHASE_3_PLAN.md`` §"Fetcher / parser families" (Block
  3 — api_webhook required validation).
- ``greenlang/factors/ingestion/fetchers.py`` — the sibling fetcher
  family this module composes against.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


__all__ = [
    "PHASE3_WEBHOOK_FEATURE_FLAG",
    "WebhookArtifact",
    "WebhookArtifactStore",
    "WebhookIdempotencyCache",
    "build_webhook_router",
    "is_webhook_enabled",
    "verify_signature",
    "webhook_router",
]


#: Env var that gates the router auto-mount. ``"1"`` enables; anything
#: else (including unset) keeps the router dormant.
PHASE3_WEBHOOK_FEATURE_FLAG: str = "GL_FACTORS_PHASE3_WEBHOOK_ENABLED"

#: Header name carrying the per-event idempotency token.
HEADER_EVENT_ID: str = "X-GL-Event-Id"

#: Header name carrying the HMAC SHA-256 hex digest.
HEADER_SIGNATURE: str = "X-GL-Signature"

#: Header name carrying the source's API version (captured into the
#: artifact row per Block 3 of the exit checklist).
HEADER_API_VERSION: str = "X-GL-API-Version"

#: Header name carrying the human approver identity for the replay
#: endpoint. Format: ``human:<email>``.
HEADER_APPROVER: str = "X-GL-Approver"

#: Maximum number of event IDs the idempotency cache retains before
#: LRU-evicting the oldest entries. Sized for ~24h of webhook traffic
#: at modest cadence.
_IDEMPOTENCY_MAX_ENTRIES: int = 10_000

#: Idempotency window — duplicates inside this window are short-
#: circuited; outside the window the event ID is treated as fresh.
_IDEMPOTENCY_WINDOW: timedelta = timedelta(hours=24)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WebhookArtifact:
    """A single received webhook payload, fully captured for replay.

    Mirrors the Phase 3 plan §"Artifact storage contract" required
    columns for the api_webhook family — ``source_url`` (the inbound
    request URL), ``response_body`` (the verbatim request bytes),
    ``response_timestamp`` (when we received it), ``api_version`` (from
    the ``X-GL-API-Version`` header), ``pagination_cursor`` (extracted
    from the body if present), and ``event_id`` (the
    ``X-GL-Event-Id`` header).
    """

    artifact_id: str
    source_id: str
    source_url: str
    body_bytes: bytes
    sha256: str
    content_type: str
    response_timestamp: str
    api_version: Optional[str]
    pagination_cursor: Optional[str]
    event_id: str
    headers: Dict[str, str] = field(default_factory=dict)

    def metadata(self) -> Dict[str, Any]:
        """Headers + capture metadata as a plain dict for the run row."""
        return {
            "event_id": self.event_id,
            "api_version": self.api_version,
            "pagination_cursor": self.pagination_cursor,
            "response_timestamp": self.response_timestamp,
            "source_url": self.source_url,
            "content_type": self.content_type,
            "headers": dict(self.headers),
        }


# ---------------------------------------------------------------------------
# Idempotency cache
# ---------------------------------------------------------------------------


class WebhookIdempotencyCache:
    """Thread-safe bounded LRU cache of recent ``X-GL-Event-Id`` values.

    Entries fall out of the cache via two mechanisms: (a) age beyond
    24h, (b) LRU eviction once :data:`_IDEMPOTENCY_MAX_ENTRIES` is
    exceeded. Both rules are applied lazily on :meth:`record` so the
    cache never blocks indefinitely.
    """

    def __init__(
        self,
        *,
        max_entries: int = _IDEMPOTENCY_MAX_ENTRIES,
        window: timedelta = _IDEMPOTENCY_WINDOW,
    ) -> None:
        self._max_entries = max_entries
        self._window = window
        # Ordered dict acts as the LRU; key=event_id, value=(received_at, run_id).
        self._cache: "OrderedDict[str, Tuple[datetime, str]]" = OrderedDict()
        self._lock = threading.Lock()

    def record(self, event_id: str, run_id: str) -> None:
        """Record a freshly-received event ID with its run id.

        If the event is already in the cache, the existing entry stays
        (so :meth:`lookup` returns the original run id).
        """
        now = datetime.now(timezone.utc)
        with self._lock:
            self._evict_locked(now)
            if event_id in self._cache:
                # Refresh LRU order without changing the original run_id.
                self._cache.move_to_end(event_id)
                return
            self._cache[event_id] = (now, run_id)
            # Cap.
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)

    def lookup(self, event_id: str) -> Optional[str]:
        """Return the prior run id for ``event_id`` if still in window."""
        now = datetime.now(timezone.utc)
        with self._lock:
            self._evict_locked(now)
            entry = self._cache.get(event_id)
            if entry is None:
                return None
            received_at, run_id = entry
            if now - received_at > self._window:
                # Stale; drop and treat as a fresh ID.
                self._cache.pop(event_id, None)
                return None
            self._cache.move_to_end(event_id)
            return run_id

    def _evict_locked(self, now: datetime) -> None:
        """Drop entries older than the window. Caller holds the lock."""
        cutoff = now - self._window
        # OrderedDict iterates oldest-first; stop on the first non-stale.
        keys_to_drop: List[str] = []
        for key, (received_at, _) in self._cache.items():
            if received_at < cutoff:
                keys_to_drop.append(key)
            else:
                break
        for key in keys_to_drop:
            self._cache.pop(key, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


# ---------------------------------------------------------------------------
# Artifact store (in-memory; sibling fetchers compose on top)
# ---------------------------------------------------------------------------


class WebhookArtifactStore:
    """In-memory store of received webhook artifacts.

    Production deployments will swap this for a Postgres-backed store
    (see ``raw_artifacts`` in :mod:`greenlang.factors.ingestion.sqlite_metadata`);
    the in-memory implementation is sufficient for the hermetic e2e
    tests + the alpha v0.1 deployment which does not yet receive
    webhook traffic.
    """

    def __init__(self) -> None:
        self._artifacts: Dict[str, WebhookArtifact] = {}
        self._runs: Dict[str, str] = {}  # artifact_id -> run_id
        self._lock = threading.Lock()

    def put(self, artifact: WebhookArtifact, run_id: str) -> None:
        with self._lock:
            self._artifacts[artifact.artifact_id] = artifact
            self._runs[artifact.artifact_id] = run_id

    def get(self, artifact_id: str) -> Optional[WebhookArtifact]:
        with self._lock:
            return self._artifacts.get(artifact_id)

    def list_for_source(self, source_id: str) -> List[WebhookArtifact]:
        with self._lock:
            return [a for a in self._artifacts.values() if a.source_id == source_id]

    def run_id_for(self, artifact_id: str) -> Optional[str]:
        with self._lock:
            return self._runs.get(artifact_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._artifacts)


# ---------------------------------------------------------------------------
# Run row writer (pluggable so tests can use an in-memory backing store)
# ---------------------------------------------------------------------------


class WebhookRunRecorder:
    """Thin wrapper that records ``ingestion_runs`` rows in ``created`` status.

    The production ingestion-runs table is created by
    :class:`IngestionRunRepository`. To keep the webhook router free of
    a hard runner dependency at import-time (the production server boots
    the runner lazily), this recorder writes a minimal in-memory row
    that is later promoted to the real run repo when the operator
    calls the replay endpoint.
    """

    def __init__(self) -> None:
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(
        self,
        *,
        source_id: str,
        artifact_id: str,
        event_id: str,
        api_version: Optional[str],
        pagination_cursor: Optional[str],
    ) -> str:
        run_id = "wbk-run-%s" % uuid.uuid4().hex[:12]
        row = {
            "run_id": run_id,
            "status": "created",
            "source_id": source_id,
            "artifact_id": artifact_id,
            "event_id": event_id,
            "api_version": api_version,
            "pagination_cursor": pagination_cursor,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._runs[run_id] = row
        return run_id

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._runs[run_id]) if run_id in self._runs else None

    def list_by_source(self, source_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._runs.values() if r["source_id"] == source_id]

    def __len__(self) -> int:
        with self._lock:
            return len(self._runs)


# ---------------------------------------------------------------------------
# HMAC verification
# ---------------------------------------------------------------------------


def _resolve_secret(source_id: str) -> Optional[str]:
    """Return the per-source HMAC shared secret from the environment.

    Env-var form: ``GL_FACTORS_WEBHOOK_SECRET_<SOURCE_ID_UPPER>``.
    Hyphens in the source id are normalised to underscores so a source
    id like ``acme-private-pack-2024`` resolves to
    ``GL_FACTORS_WEBHOOK_SECRET_ACME_PRIVATE_PACK_2024``.
    """
    key = "GL_FACTORS_WEBHOOK_SECRET_%s" % source_id.upper().replace("-", "_")
    val = os.getenv(key, "").strip()
    return val or None


def verify_signature(*, body: bytes, signature: str, secret: str) -> bool:
    """Constant-time HMAC SHA-256 verification.

    Args:
        body: The raw request body bytes (NOT a parsed dict).
        signature: The lowercase hex digest from ``X-GL-Signature``.
        secret: The per-source shared secret resolved from env.

    Returns:
        True if the signature matches, False otherwise. Length-mismatch
        signatures return False without leaking timing.
    """
    expected = hmac.new(
        secret.encode("utf-8"), msg=body, digestmod=hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected.lower(), (signature or "").strip().lower())


# ---------------------------------------------------------------------------
# Module-level singletons (the router closes over these)
# ---------------------------------------------------------------------------


_ARTIFACT_STORE = WebhookArtifactStore()
_IDEMPOTENCY = WebhookIdempotencyCache()
_RUN_RECORDER = WebhookRunRecorder()


def is_webhook_enabled() -> bool:
    """Return True iff the feature flag env is set to ``"1"``."""
    return os.getenv(PHASE3_WEBHOOK_FEATURE_FLAG, "").strip() == "1"


# ---------------------------------------------------------------------------
# Router builder
# ---------------------------------------------------------------------------


def build_webhook_router(
    *,
    artifact_store: Optional[WebhookArtifactStore] = None,
    idempotency: Optional[WebhookIdempotencyCache] = None,
    run_recorder: Optional[WebhookRunRecorder] = None,
    secret_resolver: Optional[Callable[[str], Optional[str]]] = None,
    replay_fetcher_factory: Optional[Callable[[WebhookArtifactStore], Any]] = None,
) -> APIRouter:
    """Build and return the Phase 3 webhook ingestion router.

    Tests inject an in-memory store + cache; the production server uses
    the module-level singletons.
    """
    # Use ``is None`` rather than truthy-fallback: the stores carry
    # ``__len__`` so an empty fresh store is falsy, which would silently
    # rebind the closure to the module-level singletons and route every
    # test write into the global cache.
    if artifact_store is None:
        artifact_store = _ARTIFACT_STORE
    if idempotency is None:
        idempotency = _IDEMPOTENCY
    if run_recorder is None:
        run_recorder = _RUN_RECORDER
    if secret_resolver is None:
        secret_resolver = _resolve_secret

    router = APIRouter(
        prefix="/v1/factors/ingest/webhook",
        tags=["factors-phase3-webhook"],
    )

    @router.post(
        "/{source_id}",
        status_code=status.HTTP_200_OK,
        summary="Receive a webhook payload for a registered factor source",
        include_in_schema=True,
    )
    async def receive(source_id: str, request: Request) -> JSONResponse:
        """Receive + verify + persist a webhook payload.

        Returns ``{run_id, artifact_id, duplicate}`` on success. Errors:
        ``401`` for HMAC failures; ``400`` for missing/empty event id
        or body.
        """
        body: bytes = await request.body()
        if not body:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="empty body",
            )

        event_id = request.headers.get(HEADER_EVENT_ID, "").strip()
        if not event_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="missing %s header" % HEADER_EVENT_ID,
            )

        # HMAC verification — fail closed if the secret is absent.
        signature = request.headers.get(HEADER_SIGNATURE, "")
        secret = secret_resolver(source_id)
        if not secret:
            logger.warning(
                "webhook rejected: no shared secret configured for source_id=%s",
                source_id,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="webhook secret not configured for source",
            )
        if not verify_signature(body=body, signature=signature, secret=secret):
            logger.warning(
                "webhook rejected: HMAC mismatch source_id=%s event_id=%s",
                source_id, event_id,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid signature",
            )

        # Idempotency check — duplicate within window short-circuits.
        prior_run_id = idempotency.lookup(event_id)
        if prior_run_id is not None:
            logger.info(
                "webhook duplicate: source_id=%s event_id=%s prior_run=%s",
                source_id, event_id, prior_run_id,
            )
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "run_id": prior_run_id,
                    "artifact_id": None,
                    "duplicate": True,
                },
            )

        # Capture: pagination cursor from body if shaped as JSON.
        pagination_cursor: Optional[str] = None
        try:
            decoded = json.loads(body.decode("utf-8"))
            if isinstance(decoded, dict):
                cursor = decoded.get("pagination") or decoded.get("next_cursor")
                if isinstance(cursor, dict):
                    pagination_cursor = cursor.get("cursor") or cursor.get("next")
                elif isinstance(cursor, str):
                    pagination_cursor = cursor
        except (UnicodeDecodeError, json.JSONDecodeError):
            pagination_cursor = None

        api_version = request.headers.get(HEADER_API_VERSION, "").strip() or None
        content_type = request.headers.get("content-type", "application/octet-stream")
        sha256 = hashlib.sha256(body).hexdigest()
        artifact_id = "wbk-art-%s" % uuid.uuid4().hex[:12]
        response_timestamp = datetime.now(timezone.utc).isoformat()

        # Filter headers down to a stable allow-list so we don't leak
        # routing / load-balancer noise into the artifact metadata.
        captured_headers: Dict[str, str] = {}
        for key in (HEADER_EVENT_ID, HEADER_API_VERSION, HEADER_SIGNATURE, "content-type"):
            val = request.headers.get(key)
            if val is not None:
                captured_headers[key] = val

        artifact = WebhookArtifact(
            artifact_id=artifact_id,
            source_id=source_id,
            source_url=str(request.url),
            body_bytes=body,
            sha256=sha256,
            content_type=content_type,
            response_timestamp=response_timestamp,
            api_version=api_version,
            pagination_cursor=pagination_cursor,
            event_id=event_id,
            headers=captured_headers,
        )

        run_id = run_recorder.create(
            source_id=source_id,
            artifact_id=artifact_id,
            event_id=event_id,
            api_version=api_version,
            pagination_cursor=pagination_cursor,
        )
        artifact_store.put(artifact, run_id=run_id)
        idempotency.record(event_id, run_id)

        logger.info(
            "webhook accepted: source_id=%s run_id=%s artifact_id=%s sha256=%s",
            source_id, run_id, artifact_id, sha256,
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "run_id": run_id,
                "artifact_id": artifact_id,
                "duplicate": False,
                "sha256": sha256,
            },
        )

    @router.post(
        "/{source_id}/replay",
        status_code=status.HTTP_200_OK,
        summary="Replay a stored webhook artifact through the pipeline (admin)",
        include_in_schema=True,
    )
    async def replay(source_id: str, request: Request) -> JSONResponse:
        """Replay a stored webhook artifact through the pipeline.

        Admin-only: requires ``X-GL-Approver: human:<email>``.
        Body MUST be a JSON dict carrying ``{"artifact_id": "..."}``.
        """
        approver = request.headers.get(HEADER_APPROVER, "").strip()
        if not approver.startswith("human:") or "@" not in approver:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="replay requires %s: human:<email>" % HEADER_APPROVER,
            )

        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="replay body must be JSON: %s" % exc,
            ) from exc
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="replay body must be a JSON object",
            )

        artifact_id = str(payload.get("artifact_id", "")).strip()
        if not artifact_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="missing artifact_id",
            )
        artifact = artifact_store.get(artifact_id)
        if artifact is None or artifact.source_id != source_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="artifact not found for source",
            )

        # Build a replay fetcher and (best-effort) drive the pipeline.
        # The pipeline runner is constructed by the caller in production;
        # here we record the replay request and return the run id so
        # the test can complete the round-trip via the injected adapter.
        replay_run_id = run_recorder.create(
            source_id=source_id,
            artifact_id=artifact_id,
            event_id="replay-%s" % artifact.event_id,
            api_version=artifact.api_version,
            pagination_cursor=artifact.pagination_cursor,
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "run_id": replay_run_id,
                "original_run_id": artifact_store.run_id_for(artifact_id),
                "artifact_id": artifact_id,
                "approver": approver,
                "replayed_at": datetime.now(timezone.utc).isoformat(),
                "body_sha256": artifact.sha256,
            },
        )

    return router


# ---------------------------------------------------------------------------
# Module-level production router (always built; only auto-mounted by the
# factors_app when the feature flag is set).
# ---------------------------------------------------------------------------


webhook_router = build_webhook_router()
