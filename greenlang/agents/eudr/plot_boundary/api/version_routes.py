# -*- coding: utf-8 -*-
"""
Version History Routes - AGENT-EUDR-006 Plot Boundary Manager API

Endpoints for retrieving immutable boundary version history, querying
boundary state at a specific date, computing version diffs, and
tracing the complete provenance lineage chain.

Endpoints:
    GET /versions/{plot_id}          - Get version history
    GET /versions/{plot_id}/at       - Get boundary at a specific date
    GET /versions/{plot_id}/diff     - Compute version diff
    GET /versions/{plot_id}/lineage  - Get complete version lineage

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager Agent (GL-EUDR-PBM-006)
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status

from greenlang.agents.eudr.plot_boundary.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_boundary_versioner,
    rate_limit_standard,
    require_permission,
    validate_date_param,
    validate_plot_id,
)
from greenlang.agents.eudr.plot_boundary.api.schemas import (
    BoundaryResponseSchema,
    VersionDiffResponseSchema,
    VersionHistoryResponseSchema,
    VersionLineageResponseSchema,
    VersionSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Version History"])


# ---------------------------------------------------------------------------
# In-memory version store (replaced by database in production)
# ---------------------------------------------------------------------------

_version_store: Dict[str, List[Dict[str, Any]]] = {}


def _get_version_store() -> Dict[str, List[Dict[str, Any]]]:
    """Return the version store. Replaceable for testing."""
    return _version_store


def _build_version_schema(record: Dict[str, Any]) -> VersionSchema:
    """Build a VersionSchema from an internal record.

    Args:
        record: Internal version storage record.

    Returns:
        VersionSchema for API serialization.
    """
    return VersionSchema(
        plot_id=record.get("plot_id", ""),
        version_number=record.get("version_number", 1),
        change_reason=record.get("change_reason", "initial_registration"),
        changed_by=record.get("changed_by", "system"),
        changed_at=record.get(
            "changed_at",
            datetime.now(timezone.utc).replace(microsecond=0),
        ),
        geometry=record.get("geometry"),
        area_m2=record.get("area_m2", 0.0),
        area_hectares=record.get("area_hectares", 0.0),
        area_diff_m2=record.get("area_diff_m2", 0.0),
        area_diff_pct=record.get("area_diff_pct", 0.0),
        vertex_count=record.get("vertex_count", 0),
        provenance_hash=record.get("provenance_hash", ""),
        parent_provenance_hash=record.get("parent_provenance_hash"),
    )


# ---------------------------------------------------------------------------
# GET /versions/{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/versions/{plot_id}",
    response_model=VersionHistoryResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Get version history",
    description=(
        "Retrieve the complete version history for a plot boundary. "
        "Returns all versions ordered by version number (newest first), "
        "including area changes, change reasons, and provenance hashes. "
        "Versions are retained per EUDR Article 31 retention requirements."
    ),
    responses={
        200: {"description": "Version history"},
        400: {"model": ErrorResponse, "description": "Invalid plot_id"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"description": "No version history found"},
    },
)
async def get_version_history(
    plot_id: str = Path(..., description="Plot identifier"),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:versions:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> VersionHistoryResponseSchema:
    """Retrieve complete version history for a plot boundary.

    Returns all versions ordered by version number (newest first),
    with area changes, provenance hashes, and change metadata.

    Args:
        plot_id: Plot identifier from URL path.
        user: Authenticated user with versions:read permission.

    Returns:
        VersionHistoryResponseSchema with all version records.

    Raises:
        HTTPException: 400 if plot_id invalid, 404 if no history found.
    """
    validated_id = validate_plot_id(plot_id)

    logger.info(
        "Get version history: user=%s plot_id=%s",
        user.user_id,
        validated_id,
    )

    try:
        versioner = get_boundary_versioner()

        # Try to use real engine if available
        if hasattr(versioner, "get_history"):
            result = versioner.get_history(plot_id=validated_id)
            return result

        # Check in-memory store
        store = _get_version_store()
        records = store.get(validated_id, [])

        if not records:
            # Return empty history (plot may exist without explicit versions)
            return VersionHistoryResponseSchema(
                plot_id=validated_id,
                total_versions=0,
                current_version=1,
                versions=[],
            )

        versions = [_build_version_schema(r) for r in records]
        # Sort newest first
        versions.sort(key=lambda v: v.version_number, reverse=True)
        current = versions[0].version_number if versions else 1

        return VersionHistoryResponseSchema(
            plot_id=validated_id,
            total_versions=len(versions),
            current_version=current,
            versions=versions,
        )

    except ValueError as exc:
        logger.warning(
            "Get version history error: user=%s plot_id=%s error=%s",
            user.user_id,
            validated_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Get version history failed: user=%s plot_id=%s error=%s",
            user.user_id,
            validated_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Version history retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /versions/{plot_id}/at
# ---------------------------------------------------------------------------


@router.get(
    "/versions/{plot_id}/at",
    response_model=VersionSchema,
    status_code=status.HTTP_200_OK,
    summary="Get boundary at a specific date",
    description=(
        "Retrieve the boundary version that was active at a specific "
        "date. Returns the most recent version created on or before "
        "the specified date. Useful for historical compliance queries."
    ),
    responses={
        200: {"description": "Version at the specified date"},
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"description": "No version found at the specified date"},
    },
)
async def get_version_at_date(
    plot_id: str = Path(..., description="Plot identifier"),
    date: str = Query(
        ...,
        description="ISO 8601 date string (e.g. 2024-12-31 or 2024-12-31T00:00:00Z)",
    ),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:versions:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> VersionSchema:
    """Retrieve the boundary version active at a specific date.

    Returns the most recent version created on or before the
    specified date. This supports historical boundary queries
    for compliance auditing.

    Args:
        plot_id: Plot identifier from URL path.
        date: ISO 8601 date string query parameter.
        user: Authenticated user with versions:read permission.

    Returns:
        VersionSchema for the version active at the specified date.

    Raises:
        HTTPException: 400 if parameters invalid, 404 if no version found.
    """
    validated_id = validate_plot_id(plot_id)
    target_date = validate_date_param(date)

    logger.info(
        "Get version at date: user=%s plot_id=%s date=%s",
        user.user_id,
        validated_id,
        date,
    )

    try:
        versioner = get_boundary_versioner()

        # Try to use real engine if available
        if hasattr(versioner, "get_at_date"):
            result = versioner.get_at_date(
                plot_id=validated_id,
                target_date=target_date,
            )
            return result

        # Check in-memory store
        store = _get_version_store()
        records = store.get(validated_id, [])

        if not records:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"No version history found for plot {validated_id} "
                    f"at date {date}"
                ),
            )

        # Find the most recent version on or before the target date
        candidates = [
            r for r in records
            if r.get("changed_at", datetime.min) <= target_date
        ]

        if not candidates:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"No version found for plot {validated_id} "
                    f"at or before {date}"
                ),
            )

        candidates.sort(
            key=lambda r: r.get("changed_at", datetime.min),
            reverse=True,
        )
        return _build_version_schema(candidates[0])

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Get version at date failed: user=%s plot_id=%s date=%s error=%s",
            user.user_id,
            validated_id,
            date,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Version at date query failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /versions/{plot_id}/diff
# ---------------------------------------------------------------------------


@router.get(
    "/versions/{plot_id}/diff",
    response_model=VersionDiffResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Compute version diff",
    description=(
        "Compute the geometric difference between two versions of a "
        "plot boundary. Returns added, removed, and unchanged areas "
        "along with the symmetric difference geometry."
    ),
    responses={
        200: {"description": "Version diff result"},
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"description": "Version not found"},
    },
)
async def get_version_diff(
    plot_id: str = Path(..., description="Plot identifier"),
    version_a: int = Query(
        ...,
        ge=1,
        description="First version number for comparison",
    ),
    version_b: int = Query(
        ...,
        ge=1,
        description="Second version number for comparison",
    ),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:versions:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> VersionDiffResponseSchema:
    """Compute the geometric diff between two boundary versions.

    Calculates the added, removed, and unchanged areas between
    version_a and version_b using geometric set operations.

    Args:
        plot_id: Plot identifier from URL path.
        version_a: First version number.
        version_b: Second version number.
        user: Authenticated user with versions:read permission.

    Returns:
        VersionDiffResponseSchema with area changes.

    Raises:
        HTTPException: 400 if versions same, 404 if version not found.
    """
    validated_id = validate_plot_id(plot_id)

    if version_a == version_b:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"version_a and version_b must differ, both are {version_a}",
        )

    logger.info(
        "Get version diff: user=%s plot_id=%s version_a=%d version_b=%d",
        user.user_id,
        validated_id,
        version_a,
        version_b,
    )

    try:
        versioner = get_boundary_versioner()

        # Try to use real engine if available
        if hasattr(versioner, "compute_diff"):
            result = versioner.compute_diff(
                plot_id=validated_id,
                version_a=version_a,
                version_b=version_b,
            )
            return result

        # Stub response for development
        return VersionDiffResponseSchema(
            plot_id=validated_id,
            version_a=version_a,
            version_b=version_b,
            added_area_m2=0.0,
            removed_area_m2=0.0,
            unchanged_area_m2=0.0,
            net_area_change_m2=0.0,
            net_area_change_pct=0.0,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Get version diff failed: user=%s plot_id=%s error=%s",
            user.user_id,
            validated_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Version diff computation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /versions/{plot_id}/lineage
# ---------------------------------------------------------------------------


@router.get(
    "/versions/{plot_id}/lineage",
    response_model=VersionLineageResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Get version lineage",
    description=(
        "Retrieve the complete version lineage for a plot boundary, "
        "ordered oldest to newest. Includes provenance chain validation "
        "to verify the integrity of the SHA-256 hash chain across all "
        "versions."
    ),
    responses={
        200: {"description": "Version lineage"},
        400: {"model": ErrorResponse, "description": "Invalid plot_id"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"description": "No lineage found"},
    },
)
async def get_version_lineage(
    plot_id: str = Path(..., description="Plot identifier"),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:versions:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> VersionLineageResponseSchema:
    """Retrieve complete version lineage with provenance chain validation.

    Returns all versions ordered oldest to newest with a flag
    indicating whether the SHA-256 provenance hash chain is intact.

    Args:
        plot_id: Plot identifier from URL path.
        user: Authenticated user with versions:read permission.

    Returns:
        VersionLineageResponseSchema with ordered lineage.

    Raises:
        HTTPException: 400 if plot_id invalid, 404 if no lineage.
    """
    validated_id = validate_plot_id(plot_id)

    logger.info(
        "Get version lineage: user=%s plot_id=%s",
        user.user_id,
        validated_id,
    )

    try:
        versioner = get_boundary_versioner()

        # Try to use real engine if available
        if hasattr(versioner, "get_lineage"):
            result = versioner.get_lineage(plot_id=validated_id)
            return result

        # Check in-memory store
        store = _get_version_store()
        records = store.get(validated_id, [])

        if not records:
            return VersionLineageResponseSchema(
                plot_id=validated_id,
                total_versions=0,
                lineage=[],
                provenance_chain_valid=True,
            )

        versions = [_build_version_schema(r) for r in records]
        # Sort oldest first for lineage
        versions.sort(key=lambda v: v.version_number)

        # Validate provenance chain
        chain_valid = True
        for i in range(1, len(versions)):
            expected_parent = versions[i - 1].provenance_hash
            actual_parent = versions[i].parent_provenance_hash
            if actual_parent and actual_parent != expected_parent:
                chain_valid = False
                logger.warning(
                    "Provenance chain broken: plot_id=%s version=%d "
                    "expected=%s actual=%s",
                    validated_id,
                    versions[i].version_number,
                    expected_parent,
                    actual_parent,
                )
                break

        return VersionLineageResponseSchema(
            plot_id=validated_id,
            total_versions=len(versions),
            lineage=versions,
            provenance_chain_valid=chain_valid,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Get version lineage failed: user=%s plot_id=%s error=%s",
            user.user_id,
            validated_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Version lineage retrieval failed due to an internal error",
        )
