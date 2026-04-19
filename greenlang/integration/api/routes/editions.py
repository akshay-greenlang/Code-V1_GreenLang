# -*- coding: utf-8 -*-
"""
GreenLang API - Edition Routes

REST API endpoints for factor catalog edition management.

Routes:
- GET  /api/v1/editions              - List all editions
- GET  /api/v1/editions/compare      - Compare two editions
- GET  /api/v1/editions/{edition_id}/changelog - Get edition changelog
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from greenlang.integration.api.models import (
    EditionChangelogResponse,
    EditionCompareResponse,
    EditionListResponse,
    EditionSummary,
    ErrorResponse,
)
from greenlang.integration.api.dependencies import (
    get_current_user,
    get_factor_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/editions", tags=["Editions"])


@router.get(
    "",
    response_model=EditionListResponse,
    summary="List all catalog editions",
    description="Returns all published and pending editions visible to the caller",
    responses={200: {"description": "Edition list"}},
)
async def list_editions(
    request: Request,
    include_pending: bool = Query(True, description="Include pending editions"),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> EditionListResponse:
    """List all catalog editions."""
    rows = svc.repo.list_editions(include_pending=include_pending)
    default = svc.repo.get_default_edition_id()

    return EditionListResponse(
        editions=[
            EditionSummary(
                edition_id=r.edition_id,
                status=r.status,
                label=r.label,
                manifest_hash=r.manifest_hash,
            )
            for r in rows
        ],
        default_edition_id=default,
    )


@router.get(
    "/compare",
    response_model=EditionCompareResponse,
    summary="Compare two editions",
    description="Diff factor IDs and content hashes between two editions",
    responses={
        200: {"description": "Edition comparison"},
        400: {"model": ErrorResponse, "description": "Invalid edition IDs"},
    },
)
async def compare_editions(
    request: Request,
    left: str = Query(..., description="Left (older) edition ID"),
    right: str = Query(..., description="Right (newer) edition ID"),
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> EditionCompareResponse:
    """Compare two editions to see added, removed, and changed factors."""
    try:
        result = svc.compare_editions(left, right)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return EditionCompareResponse(**result)


@router.get(
    "/{edition_id}/changelog",
    response_model=EditionChangelogResponse,
    summary="Get edition changelog",
    description="Human and machine-readable changelog lines for an edition",
    responses={
        200: {"description": "Changelog entries"},
        404: {"model": ErrorResponse, "description": "Edition not found"},
    },
)
async def get_changelog(
    request: Request,
    edition_id: str,
    current_user: dict = Depends(get_current_user),
    svc=Depends(get_factor_service),
) -> EditionChangelogResponse:
    """Get changelog entries for a specific edition."""
    try:
        svc.repo.resolve_edition(edition_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    lines = svc.repo.get_changelog(edition_id)
    return EditionChangelogResponse(edition_id=edition_id, changelog=lines)


__all__ = ["router"]
