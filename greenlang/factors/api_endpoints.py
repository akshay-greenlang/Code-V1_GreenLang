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
