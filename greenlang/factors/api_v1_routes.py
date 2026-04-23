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


__all__ = ["api_v1_router", "admin_router"]
