# -*- coding: utf-8 -*-
"""
Additional API endpoint logic for Factors (F032-F036).

Pure-logic functions that power the API endpoints. Keeps the FastAPI router
thin and this module fully testable without HTTP.

F032: Audit bundle export
F033: Bulk export (streaming)
F034: Factor diff (field-by-field)
F035: Search v2 POST body with sort + pagination
F036: ETag / Cache-Control helpers
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

from greenlang.data.emission_factor_record import EmissionFactorRecord
from greenlang.factors.catalog_repository import FactorCatalogRepository

logger = logging.getLogger(__name__)


# ==================== F032: Audit bundle export ====================


def build_audit_bundle(
    repo: FactorCatalogRepository,
    edition_id: str,
    factor_id: str,
    *,
    raw_artifact_uri: Optional[str] = None,
    parser_log: Optional[str] = None,
    qa_errors: Optional[List[str]] = None,
    reviewer_decision: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build a full audit bundle for a factor, including provenance chain
    and SHA-256 verification.

    Returns None if factor not found.
    """
    factor = repo.get_factor(edition_id, factor_id)
    if not factor:
        return None

    factor_dict = factor.to_dict()
    # Build verification chain
    payload_json = json.dumps(factor_dict, sort_keys=True, default=str)
    payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

    bundle = {
        "factor_id": factor.factor_id,
        "edition_id": edition_id,
        "content_hash": factor.content_hash,
        "payload_sha256": payload_hash,
        "normalized_record": factor_dict,
        "provenance": {
            "source_org": factor.provenance.source_org,
            "source_publication": factor.provenance.source_publication,
            "source_year": factor.provenance.source_year,
            "methodology": factor.provenance.methodology.value,
            "version": factor.provenance.version,
            "citation": factor.provenance.citation,
            "source_url": factor.provenance.source_url,
        },
        "license_info": {
            "license": factor.license_info.license,
            "redistribution_allowed": factor.license_info.redistribution_allowed,
            "commercial_use_allowed": factor.license_info.commercial_use_allowed,
            "attribution_required": factor.license_info.attribution_required,
        },
        "quality": {
            "dqs_overall": factor.dqs.overall_score,
            "dqs_rating": factor.dqs.rating.value,
            "uncertainty_95ci": factor.uncertainty_95ci,
        },
        "raw_artifact_uri": raw_artifact_uri,
        "parser_log": parser_log,
        "qa_errors": qa_errors or [],
        "reviewer_decision": reviewer_decision,
        "verification_chain": {
            "content_hash": factor.content_hash,
            "payload_sha256": payload_hash,
            "algorithm": "SHA-256",
        },
    }
    logger.info(
        "Audit bundle built: factor=%s edition=%s hash=%s",
        factor_id, edition_id, payload_hash[:12],
    )
    return bundle


# ==================== F033: Bulk export ====================


def bulk_export_factors(
    repo: FactorCatalogRepository,
    edition_id: str,
    *,
    status_filter: Optional[str] = None,
    geography: Optional[str] = None,
    fuel_type: Optional[str] = None,
    scope: Optional[str] = None,
    source_id: Optional[str] = None,
    include_preview: bool = False,
    include_connector: bool = False,
    max_rows: int = 0,
) -> List[Dict[str, Any]]:
    """
    Export factors as list of dicts (JSON Lines ready).

    Enforces license: excludes connector_only unless include_connector is True.
    Respects max_rows limit (0 = unlimited).
    """
    factors, total = repo.list_factors(
        edition_id,
        fuel_type=fuel_type,
        geography=geography,
        scope=scope,
        page=1,
        limit=max_rows if max_rows > 0 else 100_000,
        include_preview=include_preview,
        include_connector=include_connector,
    )

    rows: List[Dict[str, Any]] = []
    for f in factors:
        if status_filter:
            st = getattr(f, "factor_status", "certified") or "certified"
            if st != status_filter:
                continue
        if source_id:
            sid = getattr(f, "source_id", None)
            if sid != source_id:
                continue
        # License enforcement: skip connector_only if redistribution not allowed
        lic = f.license_info
        if not lic.redistribution_allowed:
            st = getattr(f, "factor_status", "certified") or "certified"
            if st == "connector_only":
                continue

        rows.append(f.to_dict())
        if max_rows > 0 and len(rows) >= max_rows:
            break

    logger.info(
        "Bulk export: edition=%s rows=%d total_available=%d",
        edition_id, len(rows), total,
    )
    return rows


def bulk_export_manifest(
    repo: FactorCatalogRepository,
    edition_id: str,
    row_count: int,
) -> Dict[str, Any]:
    """Build export manifest header for bulk export response."""
    manifest = repo.get_manifest_dict(edition_id)
    return {
        "edition_id": edition_id,
        "manifest_hash": manifest.get("manifest_fingerprint", ""),
        "exported_rows": row_count,
        "format": "json_lines",
    }


# ==================== F034: Factor diff ====================


def diff_factor_between_editions(
    repo: FactorCatalogRepository,
    factor_id: str,
    left_edition: str,
    right_edition: str,
) -> Dict[str, Any]:
    """
    Compare a specific factor between two editions, field by field.

    Returns a diff dict with changed, added, and removed fields.
    """
    left_factor = repo.get_factor(left_edition, factor_id)
    right_factor = repo.get_factor(right_edition, factor_id)

    result: Dict[str, Any] = {
        "factor_id": factor_id,
        "left_edition": left_edition,
        "right_edition": right_edition,
        "left_exists": left_factor is not None,
        "right_exists": right_factor is not None,
        "changes": [],
    }

    if not left_factor and not right_factor:
        result["status"] = "not_found"
        return result
    if not left_factor:
        result["status"] = "added"
        return result
    if not right_factor:
        result["status"] = "removed"
        return result

    left_dict = left_factor.to_dict()
    right_dict = right_factor.to_dict()

    if left_factor.content_hash == right_factor.content_hash:
        result["status"] = "unchanged"
        return result

    result["status"] = "changed"
    changes = _diff_dicts(left_dict, right_dict)
    result["changes"] = changes
    result["left_content_hash"] = left_factor.content_hash
    result["right_content_hash"] = right_factor.content_hash
    return result


def _diff_dicts(
    left: Dict[str, Any],
    right: Dict[str, Any],
    prefix: str = "",
) -> List[Dict[str, Any]]:
    """Recursively diff two dicts, returning list of field changes."""
    changes = []
    all_keys = set(left.keys()) | set(right.keys())

    for key in sorted(all_keys):
        path = f"{prefix}.{key}" if prefix else key
        lv = left.get(key)
        rv = right.get(key)

        if key not in left:
            changes.append({"field": path, "type": "added", "new_value": rv})
        elif key not in right:
            changes.append({"field": path, "type": "removed", "old_value": lv})
        elif isinstance(lv, dict) and isinstance(rv, dict):
            changes.extend(_diff_dicts(lv, rv, prefix=path))
        elif lv != rv:
            changes.append({
                "field": path,
                "type": "changed",
                "old_value": lv,
                "new_value": rv,
            })

    return changes


# ==================== F035: Search v2 helpers ====================

VALID_SORT_FIELDS = {
    "relevance",
    "dqs_score",
    "co2e_total",
    "source_year",
    "factor_id",
}


@dataclass
class SearchV2Request:
    """POST body for /api/v1/factors/search/v2."""

    query: str
    geography: Optional[str] = None
    fuel_type: Optional[str] = None
    scope: Optional[str] = None
    source_id: Optional[str] = None
    factor_status: Optional[str] = None
    license_class: Optional[str] = None
    dqs_min: Optional[float] = None
    valid_on_date: Optional[str] = None
    sector_tags: Optional[List[str]] = None
    activity_tags: Optional[List[str]] = None
    sort_by: str = "relevance"
    sort_order: str = "desc"
    offset: int = 0
    limit: int = 20


@dataclass
class SearchV2Result:
    """Result container for search v2."""

    factors: List[Dict[str, Any]]
    total_count: int
    offset: int
    limit: int
    query: str
    sort_by: str
    sort_order: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factors": self.factors,
            "total_count": self.total_count,
            "offset": self.offset,
            "limit": self.limit,
            "query": self.query,
            "sort_by": self.sort_by,
            "sort_order": self.sort_order,
        }


def search_v2(
    repo: FactorCatalogRepository,
    edition_id: str,
    req: SearchV2Request,
    *,
    include_preview: bool = False,
    include_connector: bool = False,
) -> SearchV2Result:
    """
    Enhanced search with post-filtering, sorting, and offset-based pagination.
    """
    # Wide fetch for post-filtering
    wide_limit = max(500, req.limit * 15)
    raw = repo.search_factors(
        edition_id,
        query=req.query,
        geography=req.geography,
        limit=wide_limit,
        include_preview=include_preview,
        include_connector=include_connector,
        factor_status=req.factor_status,
        source_id=req.source_id,
    )

    # Post-filter
    filtered = []
    for f in raw:
        if req.fuel_type and f.fuel_type.lower() != req.fuel_type.lower():
            continue
        if req.scope and f.scope.value != req.scope:
            continue
        if req.license_class:
            lc = getattr(f, "license_class", None) or ""
            if lc.lower() != req.license_class.lower():
                continue
        if req.dqs_min is not None and f.dqs.overall_score < req.dqs_min:
            continue
        if req.valid_on_date:
            try:
                check = date.fromisoformat(req.valid_on_date)
                if f.valid_from > check:
                    continue
                if f.valid_to and f.valid_to < check:
                    continue
            except ValueError:
                pass
        if req.sector_tags:
            ftags = set(getattr(f, "sector_tags", []) or [])
            if not ftags.intersection(req.sector_tags):
                continue
        if req.activity_tags:
            atags = set(getattr(f, "activity_tags", []) or [])
            if not atags.intersection(req.activity_tags):
                continue
        filtered.append(f)

    total_count = len(filtered)

    # Sort
    sort_by = req.sort_by if req.sort_by in VALID_SORT_FIELDS else "relevance"
    reverse = req.sort_order.lower() != "asc"

    if sort_by == "dqs_score":
        filtered.sort(key=lambda f: f.dqs.overall_score, reverse=reverse)
    elif sort_by == "co2e_total":
        filtered.sort(key=lambda f: f.gwp_100yr.co2e_total, reverse=reverse)
    elif sort_by == "source_year":
        filtered.sort(key=lambda f: f.provenance.source_year, reverse=reverse)
    elif sort_by == "factor_id":
        filtered.sort(key=lambda f: f.factor_id, reverse=reverse)
    # relevance: keep original order (already ranked by search)

    # Pagination
    offset = max(0, req.offset)
    page = filtered[offset: offset + req.limit]

    # Convert to dicts with summary info
    factor_dicts = []
    for f in page:
        factor_dicts.append({
            "factor_id": f.factor_id,
            "fuel_type": f.fuel_type,
            "geography": f.geography,
            "scope": f.scope.value,
            "boundary": f.boundary.value,
            "co2e_per_unit": f.gwp_100yr.co2e_total,
            "unit": f.unit,
            "source": f.provenance.source_org,
            "source_year": f.provenance.source_year,
            "dqs_score": f.dqs.overall_score,
            "factor_status": getattr(f, "factor_status", "certified") or "certified",
            "source_id": getattr(f, "source_id", None),
        })

    return SearchV2Result(
        factors=factor_dicts,
        total_count=total_count,
        offset=offset,
        limit=req.limit,
        query=req.query,
        sort_by=sort_by,
        sort_order=req.sort_order,
    )


# ==================== F036: ETag / Cache-Control helpers ====================


def compute_etag(factor: EmissionFactorRecord) -> str:
    """Compute ETag from content_hash for a factor record."""
    return f'"{factor.content_hash}"'


def compute_etag_from_dict(data: Dict[str, Any]) -> str:
    """Compute ETag from arbitrary dict (for list/search responses)."""
    payload = json.dumps(data, sort_keys=True, default=str)
    return f'"{hashlib.sha256(payload.encode("utf-8")).hexdigest()}"'


def cache_control_for_status(factor_status: str) -> str:
    """Return Cache-Control header value based on factor status."""
    st = (factor_status or "certified").lower()
    if st == "certified":
        return "public, max-age=3600"
    if st == "preview":
        return "public, max-age=600"
    if st == "connector_only":
        return "private, max-age=600"
    # deprecated or unknown
    return "no-cache"


def check_etag_match(
    if_none_match: Optional[str],
    current_etag: str,
) -> bool:
    """Check if client's If-None-Match matches current ETag (304 scenario)."""
    if not if_none_match:
        return False
    # Handle weak/strong comparison
    client = if_none_match.strip().strip('"').lstrip("W/").strip('"')
    server = current_etag.strip('"')
    return client == server


def compute_list_etag(factors: List[Any], edition_id: str) -> str:
    """
    Compute ETag for list endpoint responses.

    Combines content hashes of all factors in the page with the edition_id
    to produce a deterministic ETag. Suitable for paginated list responses.

    Args:
        factors: List of EmissionFactorRecord or summary dicts from the page.
        edition_id: Current edition identifier.

    Returns:
        Quoted ETag string (e.g. '"abc123..."').
    """
    parts = [edition_id]
    for f in factors:
        if hasattr(f, "content_hash"):
            parts.append(f.content_hash)
        elif isinstance(f, dict) and "content_hash" in f:
            parts.append(f["content_hash"])
        elif hasattr(f, "factor_id"):
            parts.append(f.factor_id)
        elif isinstance(f, dict) and "factor_id" in f:
            parts.append(f["factor_id"])
    combined = "|".join(parts)
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return f'"{digest}"'


def compute_search_etag(query: str, results: List[Any], edition_id: str) -> str:
    """
    Compute ETag for search endpoint responses.

    Incorporates the query string alongside result hashes so that
    different queries produce distinct ETags even if they return the
    same factors.

    Args:
        query: The search query string.
        results: List of EmissionFactorRecord or summary dicts.
        edition_id: Current edition identifier.

    Returns:
        Quoted ETag string (e.g. '"def456..."').
    """
    parts = [edition_id, query]
    for r in results:
        if hasattr(r, "content_hash"):
            parts.append(r.content_hash)
        elif isinstance(r, dict) and "content_hash" in r:
            parts.append(r["content_hash"])
        elif hasattr(r, "factor_id"):
            parts.append(r.factor_id)
        elif isinstance(r, dict) and "factor_id" in r:
            parts.append(r["factor_id"])
    combined = "|".join(parts)
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return f'"{digest}"'


def cache_control_for_list() -> str:
    """Return Cache-Control header value for list/search aggregate endpoints.

    List and search responses aggregate multiple factors, so they use a
    shorter max-age (5 minutes) than individual factor lookups.
    """
    return "public, max-age=300"


# ==================== GAP-2: Explain endpoint helpers ====================


#: Maximum alternates per explain response (CTO cap).
EXPLAIN_ALTERNATES_MAX: int = 20
#: Default alternates per explain response (CTO default).
EXPLAIN_ALTERNATES_DEFAULT: int = 5


def clamp_alternates_limit(limit: Optional[int]) -> int:
    """Clamp caller-supplied alternates limit into the allowed window.

    Args:
        limit: Caller-supplied `?limit=N`.  ``None`` falls back to the default.

    Returns:
        Integer in the range ``[1, EXPLAIN_ALTERNATES_MAX]``.
    """
    if limit is None:
        return EXPLAIN_ALTERNATES_DEFAULT
    try:
        n = int(limit)
    except (TypeError, ValueError):
        return EXPLAIN_ALTERNATES_DEFAULT
    if n < 1:
        return 1
    if n > EXPLAIN_ALTERNATES_MAX:
        return EXPLAIN_ALTERNATES_MAX
    return n


def cache_control_for_explain() -> str:
    """Return Cache-Control for /explain and /alternates endpoints.

    The explain payload depends on the resolution cascade which in turn
    depends on the factor catalog + the selected method profile.  Both
    are stable per-edition, so we allow a 5-minute cache.
    """
    return "private, max-age=300"


def default_method_profile_for_factor(factor: Any) -> str:
    """Pick a sensible default ``method_profile`` for a GET /explain call.

    Priority order:
        1. ``factor.method_profile`` (v2 canonical field)
        2. Derived from ``factor.scope`` (1→scope1, 2→scope2_location, 3→scope3)
        3. Fallback: ``"corporate_scope1"``
    """
    mp = getattr(factor, "method_profile", None)
    if mp:
        return str(mp)

    scope = getattr(factor, "scope", None)
    scope_val = getattr(scope, "value", None) if scope is not None else None
    if scope_val == "1":
        return "corporate_scope1"
    if scope_val == "2":
        return "corporate_scope2_location_based"
    if scope_val == "3":
        return "corporate_scope3"
    return "corporate_scope1"


def _build_single_factor_engine(factor: Any):
    """Wrap a single factor record as a fully-loaded ResolutionEngine.

    The engine treats the factor as a step-5 ``country_or_sector_average``
    candidate.  This is used by GET /explain when the caller does not
    supply a full ``ResolutionRequest`` — we still want to show *why*
    this factor would win in a default context.
    """
    from greenlang.factors.resolution.engine import ResolutionEngine

    def _source(_req, label):
        if label == "country_or_sector_average":
            return [factor]
        return []

    return ResolutionEngine(candidate_source=_source)


def _build_repo_engine(
    repo: FactorCatalogRepository,
    edition_id: str,
    *,
    include_preview: bool = False,
    include_connector: bool = False,
):
    """Build a ResolutionEngine whose candidate_source is backed by the repo.

    The candidate source serves records from the repo at each cascade step,
    relying on the request context (supplier_id, facility_id, …) and the
    engine's own tie-break logic to select winners.  For simplicity and
    determinism we return the same filtered pool at every non-customer
    step and let the method-pack selection rule filter further.
    """
    from greenlang.factors.resolution.engine import ResolutionEngine

    def _source(req, label):
        # The customer_override step is handled by tenant_overlay_reader,
        # so we never serve records here for that label.
        if label == "customer_override":
            return []
        try:
            rows, _ = repo.list_factors(
                edition_id,
                geography=req.jurisdiction,
                page=1,
                limit=500,
                include_preview=include_preview,
                include_connector=include_connector,
            )
        except TypeError:
            # Older repo signatures without kw-only filters.
            rows, _ = repo.list_factors(edition_id, page=1, limit=500)
        return list(rows)

    return ResolutionEngine(candidate_source=_source)


def build_factor_explain(
    repo: FactorCatalogRepository,
    edition_id: str,
    factor_id: str,
    *,
    method_profile: Optional[str] = None,
    alternates_limit: int = EXPLAIN_ALTERNATES_DEFAULT,
    jurisdiction: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build an explain payload for a specific ``factor_id``.

    Returns ``None`` if the factor does not exist in the edition.  The
    payload is the same shape as ``ResolvedFactor.model_dump()`` plus a
    compact ``explain()`` view, so clients get both the structured object
    and the human-oriented derivation block.
    """
    from greenlang.data.canonical_v2 import MethodProfile
    from greenlang.factors.resolution.request import ResolutionRequest

    factor = repo.get_factor(edition_id, factor_id)
    if factor is None:
        return None

    profile = method_profile or default_method_profile_for_factor(factor)
    try:
        profile_enum = MethodProfile(profile)
    except ValueError:
        profile_enum = MethodProfile.CORPORATE_SCOPE1

    engine = _build_single_factor_engine(factor)
    req = ResolutionRequest(
        activity=getattr(factor, "fuel_type", None) or factor_id,
        method_profile=profile_enum,
        jurisdiction=jurisdiction or getattr(factor, "geography", None),
    )
    resolved = engine.resolve(req)

    # Clamp alternates server-side.
    alts = list(resolved.alternates)[:clamp_alternates_limit(alternates_limit)]
    payload = resolved.model_dump(mode="json")
    payload["alternates"] = [a.model_dump(mode="json") for a in alts]
    payload["explain"] = resolved.explain()
    payload["explain"]["alternates"] = payload["alternates"]
    return payload


def build_resolution_explain(
    repo: FactorCatalogRepository,
    edition_id: str,
    request_dict: Dict[str, Any],
    *,
    alternates_limit: int = EXPLAIN_ALTERNATES_DEFAULT,
    include_preview: bool = False,
    include_connector: bool = False,
) -> Dict[str, Any]:
    """Run the full 7-step cascade against a user-supplied ResolutionRequest.

    The input dict is validated against :class:`ResolutionRequest` — on
    validation failure, Pydantic will raise ``ValidationError`` which the
    caller must translate into an HTTP 400.
    """
    from greenlang.factors.resolution.request import ResolutionRequest

    req = ResolutionRequest(**request_dict)
    engine = _build_repo_engine(
        repo,
        edition_id,
        include_preview=include_preview,
        include_connector=include_connector,
    )
    resolved = engine.resolve(req)

    alts = list(resolved.alternates)[:clamp_alternates_limit(alternates_limit)]
    payload = resolved.model_dump(mode="json")
    payload["alternates"] = [a.model_dump(mode="json") for a in alts]
    payload["explain"] = resolved.explain()
    payload["explain"]["alternates"] = payload["alternates"]
    payload["method_profile"] = req.method_profile.value
    return payload


def build_factor_alternates(
    repo: FactorCatalogRepository,
    edition_id: str,
    factor_id: str,
    *,
    method_profile: Optional[str] = None,
    alternates_limit: int = EXPLAIN_ALTERNATES_DEFAULT,
    include_preview: bool = False,
    include_connector: bool = False,
) -> Optional[Dict[str, Any]]:
    """List alternative factors that could resolve for the same activity.

    Returns ``None`` if the anchor factor is not found.  The payload
    contains the chosen factor's ``factor_id`` (for grounding) plus the
    top-N alternates scored by the tie-break engine.
    """
    from greenlang.data.canonical_v2 import MethodProfile
    from greenlang.factors.resolution.request import ResolutionRequest

    anchor = repo.get_factor(edition_id, factor_id)
    if anchor is None:
        return None

    profile = method_profile or default_method_profile_for_factor(anchor)
    try:
        profile_enum = MethodProfile(profile)
    except ValueError:
        profile_enum = MethodProfile.CORPORATE_SCOPE1

    engine = _build_repo_engine(
        repo,
        edition_id,
        include_preview=include_preview,
        include_connector=include_connector,
    )
    req = ResolutionRequest(
        activity=getattr(anchor, "fuel_type", None) or factor_id,
        method_profile=profile_enum,
        jurisdiction=getattr(anchor, "geography", None),
    )
    try:
        resolved = engine.resolve(req)
    except Exception:
        # Cascade exhausted — return just the anchor as-is.
        return {
            "factor_id": factor_id,
            "edition_id": edition_id,
            "chosen_factor_id": factor_id,
            "method_profile": profile_enum.value,
            "alternates": [],
        }

    alts = list(resolved.alternates)[:clamp_alternates_limit(alternates_limit)]
    return {
        "factor_id": factor_id,
        "edition_id": edition_id,
        "chosen_factor_id": resolved.chosen_factor_id,
        "method_profile": profile_enum.value,
        "alternates": [a.model_dump(mode="json") for a in alts],
    }


def compute_explain_etag(payload: Dict[str, Any], edition_id: str) -> str:
    """Compute a stable ETag for an explain payload.

    The ETag covers the chosen factor id + version + method profile +
    edition so cache invalidation happens on any of those axes.
    """
    chosen = payload.get("chosen_factor_id") or payload.get("factor_id") or ""
    version = payload.get("factor_version") or payload.get("method_pack_version") or ""
    profile = payload.get("method_profile") or ""
    combined = f"{edition_id}|{chosen}|{version}|{profile}|{len(payload.get('alternates', []))}"
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return f'"{digest}"'


# ==================== GAP-6: Impact Simulator REST adapters ====================


def build_impact_simulation(
    repo: FactorCatalogRepository,
    *,
    factor_id: str,
    hypothetical_value: Any = None,
    tenant_scope: Optional[List[str]] = None,
    edition_id: Optional[str] = None,
    ledger_entries: Optional[List[Dict[str, Any]]] = None,
    evidence_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the impact simulator for a single factor change.

    Wraps the existing :class:`ImpactSimulator` (which is NOT rewritten
    here — we only adapt it for REST).

    Args:
        repo: The catalog repo (used to read the current numeric value
            of the factor for delta calculation).
        factor_id: The factor being changed.
        hypothetical_value: One of:
            * ``None`` -> no value delta, just list affected computations
            * ``"deprecation"`` -> mark factor deprecated (value unchanged
              but every consumer is flagged)
            * a ``dict`` with ``{"co2e_total": float, ...}`` -> explicit
              new value for delta calc
            * a ``float`` -> shorthand for ``{"co2e_total": <float>}``
        tenant_scope: Optional list of tenant IDs to restrict the scan.
        edition_id: Edition to scope the factor lookup (for value delta).
        ledger_entries: Pre-loaded ledger rows (tests / dev).
        evidence_records: Pre-loaded evidence rows (tests / dev).

    Returns:
        The ``ImpactReport.to_dict()`` payload augmented with the REST
        adapter's metadata (``simulation_mode``, ``tenant_scope``).
    """
    from greenlang.factors.quality.impact_simulator import ImpactSimulator

    # Load ledger + evidence.  Production callers pass them in; dev/test
    # runs accept empty lists (useful for sanity-checking the adapter).
    ledger = list(ledger_entries or [])
    evidence = list(evidence_records or [])

    # Optionally filter ledger/evidence by tenant_scope before passing
    # into the simulator.  This keeps the simulator itself ignorant of
    # multi-tenancy concerns.
    if tenant_scope:
        allowed = set(tenant_scope)
        ledger = [e for e in ledger if (e.get("tenant_id") in allowed)]
        evidence = [r for r in evidence if (r.get("tenant_id") in allowed)]

    simulator = ImpactSimulator(
        ledger_entries=ledger,
        evidence_records=evidence,
    )

    # Build the value_map.  We read the current value from the repo and
    # compute the hypothetical new value based on ``hypothetical_value``.
    value_map: Optional[Dict[str, Dict[str, float]]] = None
    simulation_mode = "listing_only"
    if hypothetical_value is not None and edition_id is not None:
        current = repo.get_factor(edition_id, factor_id)
        if current is not None:
            try:
                old_value = float(current.gwp_100yr.co2e_total)
            except (AttributeError, TypeError, ValueError):
                old_value = 0.0

            if hypothetical_value == "deprecation":
                simulation_mode = "deprecation"
                value_map = {
                    factor_id: {"old": old_value, "new": old_value}
                }
            elif isinstance(hypothetical_value, (int, float)):
                simulation_mode = "value_override"
                value_map = {
                    factor_id: {
                        "old": old_value,
                        "new": float(hypothetical_value),
                    }
                }
            elif isinstance(hypothetical_value, dict):
                simulation_mode = "value_override"
                new_val = hypothetical_value.get("co2e_total", old_value)
                try:
                    new_val_f = float(new_val)
                except (TypeError, ValueError):
                    new_val_f = old_value
                value_map = {
                    factor_id: {"old": old_value, "new": new_val_f}
                }

    report = simulator.simulate_replacement(
        replaced_factor_ids=[factor_id],
        value_map=value_map,
    )
    payload = report.to_dict()
    payload["simulation_mode"] = simulation_mode
    payload["tenant_scope"] = list(tenant_scope) if tenant_scope else None
    payload["edition_id"] = edition_id
    payload["factor_id"] = factor_id
    logger.info(
        "Impact simulation built: factor=%s mode=%s computations=%d tenants=%d",
        factor_id, simulation_mode, len(report.computations), len(report.tenants),
    )
    return payload


def build_impact_simulation_batch(
    repo: FactorCatalogRepository,
    *,
    items: List[Dict[str, Any]],
    edition_id: Optional[str] = None,
    ledger_entries: Optional[List[Dict[str, Any]]] = None,
    evidence_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run :func:`build_impact_simulation` for a list of factor changes.

    Each item is a dict with ``factor_id`` (required), ``hypothetical_value``
    (optional), and ``tenant_scope`` (optional).  Returns a combined
    report with per-factor subreports + aggregated totals.
    """
    reports = []
    total_computations = 0
    tenant_set: set = set()
    for item in items:
        fid = item.get("factor_id")
        if not fid:
            continue
        sub = build_impact_simulation(
            repo,
            factor_id=fid,
            hypothetical_value=item.get("hypothetical_value"),
            tenant_scope=item.get("tenant_scope"),
            edition_id=edition_id,
            ledger_entries=ledger_entries,
            evidence_records=evidence_records,
        )
        reports.append(sub)
        total_computations += int(
            sub.get("summary", {}).get("affected_computations", 0)
        )
        tenant_set.update(sub.get("tenants") or [])
    return {
        "edition_id": edition_id,
        "item_count": len(reports),
        "aggregated_summary": {
            "affected_computations": total_computations,
            "affected_tenants": len(tenant_set),
        },
        "reports": reports,
    }


def list_dependent_computations(
    *,
    factor_id: str,
    ledger_entries: List[Dict[str, Any]],
    tenant_scope: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return the list of ledger rows that consumed ``factor_id``.

    Thin read helper used by the GET /dependent-computations endpoint.
    Filters the ledger entries by factor_id in metadata and (optionally)
    tenant.  Results are sorted newest-first for the UI.
    """
    allowed_tenants = set(tenant_scope) if tenant_scope else None
    dependents: List[Dict[str, Any]] = []
    for entry in ledger_entries:
        meta = entry.get("metadata") or {}
        entry_fid = meta.get("factor_id") or entry.get("factor_id")
        if entry_fid != factor_id:
            continue
        tenant = entry.get("tenant_id") or meta.get("tenant_id")
        if allowed_tenants is not None and tenant not in allowed_tenants:
            continue
        dependents.append(
            {
                "computation_id": str(
                    entry.get("entity_id") or entry.get("id") or ""
                ),
                "computation_hash": str(
                    entry.get("chain_hash") or entry.get("content_hash") or ""
                ),
                "tenant_id": tenant,
                "factor_id": entry_fid,
                "factor_version": meta.get("new_factor_version")
                or meta.get("factor_version"),
                "recorded_at": entry.get("recorded_at"),
            }
        )
    dependents.sort(
        key=lambda r: str(r.get("recorded_at") or ""),
        reverse=True,
    )
    return dependents


def compute_impact_etag(payload: Dict[str, Any]) -> str:
    """ETag for impact-sim + dependent-computations responses."""
    body = json.dumps(payload, sort_keys=True, default=str)
    return f'"{hashlib.sha256(body.encode("utf-8")).hexdigest()}"'


def cache_control_for_impact() -> str:
    """Cache-Control for impact simulation responses (always private)."""
    return "private, max-age=60"
