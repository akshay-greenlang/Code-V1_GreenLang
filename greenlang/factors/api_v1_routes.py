# -*- coding: utf-8 -*-
"""
GreenLang Factors API — /v1 router (FY27 launch — Track A).

Mounted by :func:`greenlang.factors.factors_app.create_factors_app` at /v1.
Every route here is a thin wrapper around the pure-logic helpers in
:mod:`greenlang.factors.api_endpoints` so the router stays test-friendly.

CTO non-negotiables enforced here:

  * Every factor-returning route includes an ``explain`` block by default.
    Pass ``?compact=true`` to opt out — that's the only way to suppress it.
  * Every response carries the active edition id (via response header set
    upstream by ``EditionPinMiddleware``) so clients can pin and replay.
  * Routes that intentionally don't return factor records (health,
    coverage counts, FQS distribution, edition manifest) set
    ``request.state.skip_licensing_scan = True`` so the licensing-guard
    middleware doesn't waste cycles walking their payloads.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from greenlang.factors.api_endpoints import (
    EXPLAIN_ALTERNATES_DEFAULT,
    build_factor_explain,
    build_resolution_explain,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _service(request: Request):
    svc = getattr(request.app.state, "factors_service", None)
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "factors_service_unavailable",
                "message": (
                    "FactorCatalogService is not configured on this app. "
                    "Set GL_FACTORS_SQLITE_PATH or wire a service in "
                    "create_factors_app()."
                ),
            },
        )
    return svc


def _edition_id(request: Request, svc) -> str:
    pinned = getattr(request.state, "edition_id", None)
    if pinned:
        return pinned
    try:
        return svc.repo.resolve_edition(None)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"No default edition: {exc}")


def _strip_explain(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    payload = dict(payload)
    payload.pop("explain", None)
    payload.pop("alternates", None)
    payload["_compact"] = True
    return payload


def _require_admin(request: Request) -> None:
    user = getattr(request.state, "user", None) or {}
    tier = (user.get("tier") or "").lower()
    if tier not in ("internal", "enterprise", "admin"):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "admin_required",
                "message": "Operator console requires enterprise/internal/admin tier.",
            },
        )


# ---------------------------------------------------------------------------
# /v1 (public + authenticated)
# ---------------------------------------------------------------------------


api_v1_router = APIRouter(prefix="/v1", tags=["factors-v1"])


@api_v1_router.get("/health")
def v1_health(request: Request) -> Dict[str, Any]:
    request.state.skip_licensing_scan = True
    svc = getattr(request.app.state, "factors_service", None)
    edition = None
    if svc is not None:
        try:
            edition = svc.repo.resolve_edition(None)
        except Exception:  # noqa: BLE001
            edition = None
    return {
        "status": "ok",
        "service": "greenlang-factors",
        "version": "1.0.0",
        "edition": edition,
    }


# ---------------------------------------------------------------------------
# DEP2: /v1/health/signing-status
# ---------------------------------------------------------------------------
# Authed endpoint — leaks the key fingerprint and rotation state, so we
# gate it behind the same auth posture as other /v1 endpoints (the
# AuthMeteringMiddleware populates request.state.user). A missing user
# (unauthed request) returns 401.
# ---------------------------------------------------------------------------


def _require_authed(request: Request) -> Dict[str, Any]:
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "auth_required",
                "message": "/v1/health/signing-status requires a valid API key or JWT.",
            },
        )
    return user


def _pub_key_fingerprint() -> Optional[str]:
    """Return ``sha256-<first16hex>`` of the Ed25519 public key (PEM), or None."""
    import hashlib
    import os

    pub = (
        os.getenv("GL_FACTORS_ED25519_PUBLIC_KEY")
        or os.getenv("SIGNING_KEY_ED25519_PUB")
        or ""
    )
    if not pub.strip():
        return None
    digest = hashlib.sha256(pub.strip().encode("utf-8")).hexdigest()
    return f"sha256-{digest[:16]}"


def _rotation_status() -> Dict[str, Any]:
    """Derive rotation state from ExternalSecret-supplied env vars."""
    import os
    from datetime import datetime, timezone

    last = os.getenv("SIGNING_KEY_ROTATED_AT") or None
    nxt = os.getenv("SIGNING_KEY_NEXT_ROTATION_AT") or None
    status = "unknown"
    if nxt:
        try:
            due = datetime.fromisoformat(nxt.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            # 14-day soft window before due -> "due"; past due -> "overdue"
            from datetime import timedelta
            if now > due:
                status = "overdue"
            elif (due - now) < timedelta(days=14):
                status = "due"
            else:
                status = "current"
        except Exception:  # noqa: BLE001
            status = "unknown"
    return {
        "last_rotated_at": last,
        "next_rotation_due_at": nxt,
        "rotation_status": status,
    }


@api_v1_router.get("/health/signing-status")
def v1_health_signing_status(request: Request) -> Dict[str, Any]:
    """Return the current signed-receipt key state (authed only)."""
    _require_authed(request)
    request.state.skip_licensing_scan = True

    import os

    priv_present = bool(
        (os.getenv("GL_FACTORS_ED25519_PRIVATE_KEY") or os.getenv("SIGNING_KEY_ED25519_PRIV") or "").strip()
    )
    fp = _pub_key_fingerprint()
    rot = _rotation_status()

    return {
        "signing_installed": priv_present and fp is not None,
        "alg": "ed25519",
        "key_fingerprint": fp,
        "rotation_status": rot["rotation_status"],
        "last_rotated_at": rot["last_rotated_at"],
        "next_rotation_due_at": rot["next_rotation_due_at"],
    }


@api_v1_router.post("/resolve")
def v1_resolve(
    request: Request,
    body: Dict[str, Any],
    compact: bool = Query(False, description="Omit the explain block when true"),
    alternates_limit: int = Query(EXPLAIN_ALTERNATES_DEFAULT, ge=0, le=20),
    include_preview: bool = Query(False),
    include_connector: bool = Query(False),
) -> Dict[str, Any]:
    svc = _service(request)
    edition = _edition_id(request, svc)
    try:
        payload = build_resolution_explain(
            svc.repo, edition, body,
            alternates_limit=alternates_limit,
            include_preview=include_preview,
            include_connector=include_connector,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))
    return _strip_explain(payload) if compact else payload


@api_v1_router.get("/factors/{factor_id}")
def v1_get_factor(
    request: Request,
    factor_id: str,
    compact: bool = Query(False),
    method_profile: Optional[str] = Query(None),
    alternates_limit: int = Query(EXPLAIN_ALTERNATES_DEFAULT, ge=0, le=20),
    jurisdiction: Optional[str] = Query(None),
) -> Dict[str, Any]:
    svc = _service(request)
    edition = _edition_id(request, svc)
    payload = build_factor_explain(
        svc.repo, edition, factor_id,
        method_profile=method_profile,
        alternates_limit=alternates_limit,
        jurisdiction=jurisdiction,
    )
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"Factor '{factor_id}' not found in edition '{edition}'",
        )
    return _strip_explain(payload) if compact else payload


@api_v1_router.get("/factors/{factor_id}/explain")
def v1_explain(
    request: Request,
    factor_id: str,
    method_profile: Optional[str] = Query(None),
    alternates_limit: int = Query(EXPLAIN_ALTERNATES_DEFAULT, ge=0, le=20),
    jurisdiction: Optional[str] = Query(None),
) -> Dict[str, Any]:
    svc = _service(request)
    edition = _edition_id(request, svc)
    payload = build_factor_explain(
        svc.repo, edition, factor_id,
        method_profile=method_profile,
        alternates_limit=alternates_limit,
        jurisdiction=jurisdiction,
    )
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Factor '{factor_id}' not found")
    return payload.get("explain", payload)


@api_v1_router.get("/coverage")
def v1_coverage(request: Request) -> Dict[str, Any]:
    request.state.skip_licensing_scan = True
    svc = _service(request)
    edition = _edition_id(request, svc)
    try:
        from greenlang.factors.ga.readiness import label_counts
        counts = label_counts(svc.repo, edition)
    except Exception as exc:  # noqa: BLE001
        logger.warning("coverage label_counts failed: %s", exc)
        counts = {}
    return {"edition": edition, "families": counts}


@api_v1_router.get("/quality/fqs")
def v1_quality_fqs(request: Request) -> Dict[str, Any]:
    request.state.skip_licensing_scan = True
    svc = _service(request)
    edition = _edition_id(request, svc)
    try:
        from greenlang.factors.quality.composite_fqs import compute_fqs
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"composite_fqs unavailable: {exc}")
    try:
        rows, _ = svc.repo.list_factors(edition, page=1, limit=100_000)
    except TypeError:
        rows, _ = svc.repo.list_factors(edition)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc))

    per_family: Dict[str, list] = defaultdict(list)
    for f in rows:
        fam_attr = getattr(f, "factor_family", None)
        fam = getattr(fam_attr, "value", None) if fam_attr is not None else None
        fam = fam or getattr(f, "fuel_type", None) or "uncategorized"
        try:
            fqs = compute_fqs(f.dqs)
            per_family[fam].append(int(getattr(fqs, "score", getattr(fqs, "value", 0))))
        except Exception:  # noqa: BLE001
            continue

    out: Dict[str, Any] = {"edition": edition, "families": {}}
    for fam, scores in per_family.items():
        if not scores:
            continue
        scores.sort()
        n = len(scores)
        mean = sum(scores) / n
        out["families"][fam] = {
            "count": n,
            "min": scores[0],
            "p25": scores[n // 4],
            "median": scores[n // 2],
            "p75": scores[(3 * n) // 4],
            "max": scores[-1],
            "mean": round(mean, 1),
        }
    return out


@api_v1_router.get("/editions/{edition_id}")
def v1_edition(request: Request, edition_id: str) -> Dict[str, Any]:
    request.state.skip_licensing_scan = True
    svc = _service(request)
    repo = svc.repo
    if hasattr(repo, "get_manifest_dict"):
        try:
            return repo.get_manifest_dict(edition_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc))
    try:
        resolved = repo.resolve_edition(edition_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=str(exc))
    return {"edition_id": resolved}


# ---------------------------------------------------------------------------
# /v1/admin (operator console — admin tier required)
# ---------------------------------------------------------------------------


admin_router = APIRouter(prefix="/v1/admin", tags=["factors-admin"])


@admin_router.get("/queue")
def admin_queue(request: Request, status_filter: Optional[str] = Query(None)) -> Dict[str, Any]:
    _require_admin(request)
    svc = _service(request)
    try:
        from greenlang.factors.quality.review_queue import list_queue
        return {"items": list_queue(svc.repo, status=status_filter)}
    except Exception as exc:  # noqa: BLE001
        return {"items": [], "warning": f"review_queue unavailable: {exc}"}


@admin_router.post("/queue/{item_id}/approve")
def admin_approve(request: Request, item_id: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _require_admin(request)
    svc = _service(request)
    user = getattr(request.state, "user", {}) or {}
    try:
        from greenlang.factors.quality.review_queue import approve
        return approve(svc.repo, item_id, reviewer=user.get("user_id"), notes=(body or {}).get("notes"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc))


@admin_router.post("/queue/{item_id}/reject")
def admin_reject(request: Request, item_id: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _require_admin(request)
    svc = _service(request)
    user = getattr(request.state, "user", {}) or {}
    try:
        from greenlang.factors.quality.review_queue import reject
        return reject(svc.repo, item_id, reviewer=user.get("user_id"), reason=(body or {}).get("reason", ""))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc))


@admin_router.get("/diff/{from_edition}/{to_edition}")
def admin_diff(request: Request, from_edition: str, to_edition: str) -> Dict[str, Any]:
    _require_admin(request)
    svc = _service(request)
    return svc.compare_editions(from_edition, to_edition)


@admin_router.post("/impact-simulate")
def admin_impact_simulate(request: Request, body: Dict[str, Any]) -> Dict[str, Any]:
    _require_admin(request)
    svc = _service(request)
    try:
        from greenlang.factors.quality.impact_simulator import simulate_impact
        return simulate_impact(svc.repo, body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc))


@admin_router.get("/overrides")
def admin_list_overrides(request: Request) -> Dict[str, Any]:
    _require_admin(request)
    svc = _service(request)
    user = getattr(request.state, "user", {}) or {}
    try:
        from greenlang.factors.tenant_overlay import list_overrides
        return {"items": list_overrides(svc.repo, tenant_id=user.get("tenant_id"))}
    except Exception as exc:  # noqa: BLE001
        return {"items": [], "warning": str(exc)}


@admin_router.post("/overrides")
def admin_create_override(request: Request, body: Dict[str, Any]) -> Dict[str, Any]:
    _require_admin(request)
    svc = _service(request)
    user = getattr(request.state, "user", {}) or {}
    try:
        from greenlang.factors.tenant_overlay import create_override
        return create_override(svc.repo, tenant_id=user.get("tenant_id"), **body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))


@admin_router.delete("/overrides/{override_id}")
def admin_delete_override(request: Request, override_id: str) -> Dict[str, Any]:
    _require_admin(request)
    svc = _service(request)
    user = getattr(request.state, "user", {}) or {}
    try:
        from greenlang.factors.tenant_overlay import delete_override
        delete_override(svc.repo, tenant_id=user.get("tenant_id"), override_id=override_id)
        return {"deleted": override_id}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail=str(exc))


@admin_router.get("/sources")
def admin_list_sources(request: Request) -> Dict[str, Any]:
    _require_admin(request)
    try:
        from greenlang.factors.source_registry import load_source_registry
        reg = load_source_registry()
        items = [s if isinstance(s, dict) else getattr(s, "__dict__", {"id": str(s)}) for s in reg]
    except Exception as exc:  # noqa: BLE001
        items = []
        return {"items": items, "warning": str(exc)}
    return {"items": items}


@admin_router.post("/sources/ingest/{source_id}")
def admin_ingest_source(request: Request, source_id: str) -> Dict[str, Any]:
    _require_admin(request)
    user = getattr(request.state, "user", {}) or {}
    try:
        from greenlang.factors.batch_jobs import enqueue_source_ingest
        job_id = enqueue_source_ingest(source_id, requested_by=user.get("user_id"))
        return {"job_id": job_id, "source_id": source_id, "status": "queued"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc))


@admin_router.post("/mapping/suggest")
def admin_mapping_suggest(request: Request, body: Dict[str, Any]) -> Dict[str, Any]:
    _require_admin(request)
    svc = _service(request)
    activity = body.get("activity") or body.get("text") or ""
    try:
        from greenlang.factors.matching.suggestion_agent import suggest_mappings
        return {"suggestions": suggest_mappings(svc.repo, activity, top_k=int(body.get("top_k", 10)))}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc))


# ===========================================================================
# === W4-C API surfaces ====================================================
# ===========================================================================
# The routes below were added by Agent W4-C per master to-do items
# R18 / API11..API15. They share middleware (auth, rate-limit, licensing,
# signed-receipts) with the /v1 routes above. Kept in a separate section so
# future contributors can see the boundary at a glance.
# ---------------------------------------------------------------------------

from typing import List as _List  # local alias to keep the single-import top


def _require_user(request: Request) -> Dict[str, Any]:
    """Common helper for W4-C routes — 401 when no authed user."""
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(
            status_code=401,
            detail={"error": "auth_required", "message": "API key or JWT required."},
        )
    return user


def _tier_of(user: Dict[str, Any]) -> str:
    return (user.get("tier") or "community").strip().lower()


def _tenant_of(user: Dict[str, Any]) -> str:
    return str(user.get("tenant_id") or user.get("user_id") or "anon")


# ---------------------------------------------------------------------------
# API11 / R13 — Batch API
# ---------------------------------------------------------------------------


#: Per-day batch caps by tier (API11).
BATCH_DAILY_CAPS: Dict[str, Optional[int]] = {
    "community": 0,            # Community: N/A
    "pro": 10_000,
    "platform": 100_000,
    "consulting": 100_000,     # parity with platform
    "enterprise": None,        # unlimited
    "internal": None,
}

#: Max rows per single submission (hard spec: 10k rows).
BATCH_MAX_ROWS_PER_SUBMIT = 10_000


def _daily_cap_for(tier: str) -> Optional[int]:
    return BATCH_DAILY_CAPS.get(tier.lower(), 0)


@api_v1_router.post("/batch/resolve", status_code=202)
async def v1_batch_resolve(request: Request) -> Dict[str, Any]:
    """Submit a batch resolution job (CSV or JSON body, up to 10k rows).

    Returns ``{batch_id, status:"queued", status_url}``.
    """
    user = _require_user(request)
    tier = _tier_of(user)
    cap = _daily_cap_for(tier)
    if cap == 0:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "tier_forbidden",
                "message": "Community tier cannot submit batch jobs. Upgrade to Pro+.",
            },
        )

    content_type = (request.headers.get("content-type") or "").lower()
    rows: _List[Dict[str, Any]] = []

    if "application/json" in content_type:
        body = await request.json()
        if isinstance(body, dict) and "requests" in body:
            rows = list(body["requests"] or [])
        elif isinstance(body, list):
            rows = list(body)
        else:
            raise HTTPException(
                status_code=400,
                detail="JSON body must be a list or {requests: [...]}.",
            )
    elif "text/csv" in content_type or "application/csv" in content_type:
        import csv
        import io
        raw = (await request.body()).decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(raw))
        rows = [dict(r) for r in reader]
    else:
        raise HTTPException(
            status_code=415,
            detail="Submit as application/json or text/csv.",
        )

    if not rows:
        raise HTTPException(status_code=400, detail="No rows to resolve.")
    if len(rows) > BATCH_MAX_ROWS_PER_SUBMIT:
        raise HTTPException(
            status_code=413,
            detail=f"Too many rows: {len(rows)} > {BATCH_MAX_ROWS_PER_SUBMIT}.",
        )

    # Per-day tier cap.
    if cap is not None:
        from greenlang.factors.batch_jobs import get_default_queue
        queue = get_default_queue()
        try:
            from datetime import datetime as _dt
            today = _dt.utcnow().date().isoformat()
            jobs, _ = queue.list_for_tenant(_tenant_of(user), limit=1000)
            used = sum(
                j.request_count for j in jobs
                if (j.submitted_at or "").startswith(today)
            )
        except Exception:  # noqa: BLE001
            used = 0
        if used + len(rows) > cap:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "batch_daily_cap",
                    "message": f"Tier '{tier}' daily cap {cap} exceeded (used={used}, batch={len(rows)}).",
                    "tier": tier,
                    "cap": cap,
                    "used": used,
                },
            )

    from greenlang.factors.batch_jobs import get_default_queue, submit_batch_resolution
    queue = get_default_queue()
    handle = submit_batch_resolution(
        queue,
        requests=rows,
        tenant_id=_tenant_of(user),
        tier=tier,
        created_by=str(user.get("user_id") or "api"),
    )
    base_url = str(request.base_url).rstrip("/")
    return {
        "batch_id": handle.job_id,
        "status": "queued",
        "status_url": f"{base_url}/v1/batch/{handle.job_id}",
        "submitted_at": handle.submitted_at,
        "request_count": handle.request_count,
    }


@api_v1_router.get("/batch/{batch_id}")
def v1_batch_status(request: Request, batch_id: str) -> Dict[str, Any]:
    """Return batch job status + result/error file URLs + signed-receipt manifest URL."""
    user = _require_user(request)
    tenant_id = _tenant_of(user)
    from greenlang.factors.batch_jobs import (
        BatchJobNotFoundError,
        get_batch_job_status,
        get_default_queue,
    )
    queue = get_default_queue()
    try:
        job = get_batch_job_status(queue, batch_id)
    except BatchJobNotFoundError:
        raise HTTPException(status_code=404, detail=f"batch {batch_id} not found")
    if job.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Not authorised to read this batch.")

    base_url = str(request.base_url).rstrip("/")
    rc = max(1, int(job.request_count))
    progress = round(100.0 * min(1.0, (job.completed_count + job.failed_count) / rc), 2)
    return {
        "batch_id": job.job_id,
        "status": job.status.value,
        "progress": progress,
        "completed_count": job.completed_count,
        "failed_count": job.failed_count,
        "request_count": job.request_count,
        "submitted_at": job.submitted_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "result_file_url": (
            f"{base_url}/v1/batch/{job.job_id}/results"
            if job.status.value == "completed"
            else None
        ),
        "error_file_url": (
            f"{base_url}/v1/batch/{job.job_id}/errors"
            if job.failed_count > 0
            else None
        ),
        "signed_receipt_manifest_url": (
            f"{base_url}/v1/batch/{job.job_id}/receipts"
            if job.status.value == "completed"
            else None
        ),
    }


@api_v1_router.get("/batch/{batch_id}/results")
def v1_batch_results(
    request: Request,
    batch_id: str,
    cursor: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=10_000),
) -> Dict[str, Any]:
    user = _require_user(request)
    tenant_id = _tenant_of(user)
    from greenlang.factors.batch_jobs import (
        BatchJobNotFoundError,
        get_batch_job_results,
        get_batch_job_status,
        get_default_queue,
    )
    queue = get_default_queue()
    try:
        job = get_batch_job_status(queue, batch_id)
    except BatchJobNotFoundError:
        raise HTTPException(status_code=404, detail=f"batch {batch_id} not found")
    if job.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Not authorised to read this batch.")
    return get_batch_job_results(queue, batch_id, cursor=cursor, limit=limit)


@api_v1_router.get("/batch/{batch_id}/errors")
def v1_batch_errors(request: Request, batch_id: str) -> Dict[str, Any]:
    user = _require_user(request)
    tenant_id = _tenant_of(user)
    from greenlang.factors.batch_jobs import (
        BatchJobNotFoundError,
        get_batch_job_status,
        get_default_queue,
    )
    queue = get_default_queue()
    try:
        job = get_batch_job_status(queue, batch_id)
    except BatchJobNotFoundError:
        raise HTTPException(status_code=404, detail=f"batch {batch_id} not found")
    if job.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Not authorised to read this batch.")
    return {"batch_id": batch_id, "errors": job.error_log}


# ---------------------------------------------------------------------------
# API13 — Hosted Explain Logs
# ---------------------------------------------------------------------------


@api_v1_router.post("/explain/subscribe")
def v1_explain_subscribe(request: Request) -> Dict[str, Any]:
    user = _require_user(request)
    tier = _tier_of(user)
    from greenlang.factors.explain_history import can_subscribe, get_default_store
    if not can_subscribe(tier):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "tier_forbidden",
                "message": "Hosted explain logs are Pro+ only.",
            },
        )
    store = get_default_store()
    return store.subscribe(tenant_id=_tenant_of(user), tier=tier)


@api_v1_router.delete("/explain/subscribe")
def v1_explain_unsubscribe(request: Request) -> Dict[str, Any]:
    user = _require_user(request)
    from greenlang.factors.explain_history import get_default_store
    store = get_default_store()
    ok = store.unsubscribe(_tenant_of(user))
    return {"unsubscribed": ok, "tenant_id": _tenant_of(user)}


@api_v1_router.get("/explain/history")
def v1_explain_history(
    request: Request,
    factor_id: Optional[str] = Query(None),
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=10_000),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    user = _require_user(request)
    tier = _tier_of(user)
    from greenlang.factors.explain_history import can_subscribe, get_default_store
    if not can_subscribe(tier):
        raise HTTPException(status_code=403, detail="Hosted explain logs are Pro+ only.")
    store = get_default_store()
    rows = store.list_history(
        tenant_id=_tenant_of(user),
        factor_id=factor_id,
        from_ts=from_,
        to_ts=to,
        limit=limit,
        offset=offset,
    )
    return {
        "count": len(rows),
        "items": [r.to_dict() for r in rows],
    }


@api_v1_router.get("/explain/history/{receipt_id}")
def v1_explain_history_get(request: Request, receipt_id: str) -> Dict[str, Any]:
    user = _require_user(request)
    tier = _tier_of(user)
    from greenlang.factors.explain_history import can_subscribe, get_default_store
    if not can_subscribe(tier):
        raise HTTPException(status_code=403, detail="Hosted explain logs are Pro+ only.")
    store = get_default_store()
    rec = store.get(tenant_id=_tenant_of(user), receipt_id=receipt_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"explain receipt {receipt_id} not found")
    return rec.to_dict()


@api_v1_router.post("/explain/history/purge")
def v1_explain_history_purge(
    request: Request,
    before: Optional[str] = Query(None, description="ISO timestamp; purge rows <= this"),
) -> Dict[str, Any]:
    user = _require_user(request)
    tier = _tier_of(user)
    if tier not in ("enterprise", "internal", "admin"):
        raise HTTPException(
            status_code=403,
            detail="Tenant purge is an Enterprise-tier privilege.",
        )
    from greenlang.factors.explain_history import get_default_store
    store = get_default_store()
    deleted = store.purge(tenant_id=_tenant_of(user), before_ts=before)
    return {"deleted": deleted, "tenant_id": _tenant_of(user)}


# ---------------------------------------------------------------------------
# API12 — Webhooks hardening: admin DLQ + replay
# ---------------------------------------------------------------------------


@admin_router.get("/webhooks/dead-letter")
def admin_webhooks_dlq(
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
    subscription_id: Optional[str] = Query(None),
) -> Dict[str, Any]:
    _require_admin(request)
    from greenlang.factors.webhooks_hardened import WebhookDeliveryTracker
    tracker = getattr(request.app.state, "webhook_tracker", None) or WebhookDeliveryTracker()
    request.app.state.webhook_tracker = tracker
    items = tracker.list_dead_letter(limit=limit, subscription_id=subscription_id)
    return {"items": items, "count": len(items)}


@admin_router.post("/webhooks/dead-letter/{delivery_id}/replay")
def admin_webhooks_replay(request: Request, delivery_id: str) -> Dict[str, Any]:
    _require_admin(request)
    from greenlang.factors.webhooks_hardened import (
        WebhookDeliveryTracker,
        replay_dead_letter,
    )
    tracker = getattr(request.app.state, "webhook_tracker", None) or WebhookDeliveryTracker()
    request.app.state.webhook_tracker = tracker

    registry = getattr(request.app.state, "webhook_registry", None)

    def _lookup(subscription_id: str):
        if registry is None:
            return None
        try:
            with registry._lock:  # type: ignore[attr-defined]
                row = registry._conn.execute(  # type: ignore[attr-defined]
                    "SELECT subscription_id, tenant_id, target_url, secret, "
                    "event_types_json, active, created_at "
                    "FROM factor_webhook_subscriptions WHERE subscription_id = ?",
                    (subscription_id,),
                ).fetchone()
        except Exception:
            return None
        if not row:
            return None
        from greenlang.factors.webhooks import WebhookSubscription
        import json as _json
        return WebhookSubscription(
            subscription_id=row[0],
            tenant_id=row[1],
            target_url=row[2],
            secret=row[3],
            event_types=_json.loads(row[4]),
            active=bool(row[5]),
            created_at=row[6],
        )

    try:
        receipt = replay_dead_letter(
            tracker=tracker,
            delivery_id=delivery_id,
            subscription_lookup=_lookup,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"delivery {delivery_id} not found")
    return receipt


__all__ = ["api_v1_router", "admin_router"]
