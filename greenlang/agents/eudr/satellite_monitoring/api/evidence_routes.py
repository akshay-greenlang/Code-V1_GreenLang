# -*- coding: utf-8 -*-
"""
Evidence Routes - AGENT-EUDR-003 Satellite Monitoring API

Endpoints for generating, retrieving, and downloading EUDR evidence
packages that compile satellite monitoring data for compliance reporting.

Endpoints:
    POST /package              - Generate evidence package
    GET  /{package_id}         - Retrieve evidence package
    GET  /{package_id}/download - Download evidence in requested format

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-003 Satellite Monitoring Agent (GL-EUDR-SAT-003)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from greenlang.agents.eudr.satellite_monitoring.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_satellite_service,
    rate_limit_export,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.satellite_monitoring.api.schemas import (
    EvidencePackageResponse,
    GenerateEvidenceApiRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Evidence Packages"])


# ---------------------------------------------------------------------------
# In-memory evidence store (replaced by database in production)
# ---------------------------------------------------------------------------

_evidence_store: Dict[str, Dict[str, Any]] = {}


def _get_evidence_store() -> Dict[str, Dict[str, Any]]:
    """Return the evidence store. Replaceable for testing."""
    return _evidence_store


# ---------------------------------------------------------------------------
# POST /package
# ---------------------------------------------------------------------------


@router.post(
    "/package",
    response_model=EvidencePackageResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate evidence package",
    description=(
        "Generate a comprehensive satellite monitoring evidence package "
        "for a production plot. Compiles baseline snapshots, change "
        "detection history, NDVI time series, satellite imagery "
        "references, and compliance assessment into a single package. "
        "Supports JSON, CSV, and PDF output formats."
    ),
    responses={
        202: {"description": "Evidence package generation initiated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def generate_evidence_package(
    body: GenerateEvidenceApiRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:evidence:write")
    ),
    _rate: None = Depends(rate_limit_export),
) -> EvidencePackageResponse:
    """Generate an evidence package for a production plot.

    Compiles all satellite monitoring data into a comprehensive
    evidence package suitable for EUDR compliance submissions.

    Args:
        body: Evidence generation request with plot and format details.
        user: Authenticated user with evidence:write permission.

    Returns:
        EvidencePackageResponse with compiled evidence data.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    package_id = f"evd-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Evidence package generation: user=%s plot_id=%s operator=%s "
        "format=%s include_ts=%s",
        user.user_id,
        body.plot_id,
        body.operator_id,
        body.format,
        body.include_time_series,
    )

    # Authorization: ensure user can only generate for their own operator
    operator_id = user.operator_id or user.user_id
    if body.operator_id != operator_id and "admin" not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to generate evidence for this operator",
        )

    try:
        service = get_satellite_service()

        result = service.generate_evidence_package(
            plot_id=body.plot_id,
            operator_id=body.operator_id,
            output_format=body.format,
            include_time_series=body.include_time_series,
            include_imagery_refs=body.include_imagery_refs,
        )

        elapsed = time.monotonic() - start

        # Build response
        download_url = None
        if body.format != "json":
            download_url = getattr(result, "download_url", None)

        response = EvidencePackageResponse(
            package_id=getattr(result, "package_id", package_id),
            plot_id=body.plot_id,
            operator_id=body.operator_id,
            format=body.format,
            status="generated",
            baseline_snapshot=getattr(result, "baseline_snapshot", None),
            change_detections=getattr(result, "change_detections", []),
            time_series=getattr(result, "time_series", None) if body.include_time_series else None,
            imagery_references=getattr(result, "imagery_references", None) if body.include_imagery_refs else None,
            alerts=getattr(result, "alerts", []),
            compliance_assessment=getattr(result, "compliance_assessment", {}),
            total_monitoring_events=getattr(result, "total_monitoring_events", 0),
            date_range_start=getattr(result, "date_range_start", None),
            date_range_end=getattr(result, "date_range_end", None),
            download_url=download_url,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

        # Store for retrieval
        store = _get_evidence_store()
        store[package_id] = {
            "package_id": package_id,
            "plot_id": body.plot_id,
            "operator_id": body.operator_id,
            "format": body.format,
            "status": "generated",
            "response_data": response.model_dump(mode="json"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": user.user_id,
        }

        logger.info(
            "Evidence package generated: package_id=%s plot_id=%s "
            "operator=%s elapsed_ms=%.1f",
            package_id,
            body.plot_id,
            body.operator_id,
            elapsed * 1000,
        )

        return response

    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning(
            "Evidence generation error: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Evidence generation failed: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evidence package generation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /{package_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{package_id}",
    response_model=EvidencePackageResponse,
    summary="Retrieve evidence package",
    description="Retrieve a previously generated evidence package by its ID.",
    responses={
        200: {"description": "Evidence package"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Package not found"},
    },
)
async def get_evidence_package(
    package_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:evidence:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> EvidencePackageResponse:
    """Retrieve a previously generated evidence package.

    Args:
        package_id: Evidence package identifier.
        user: Authenticated user with evidence:read permission.

    Returns:
        EvidencePackageResponse with the stored evidence data.

    Raises:
        HTTPException: 404 if package not found, 403 if unauthorized.
    """
    logger.info(
        "Evidence retrieval: user=%s package_id=%s",
        user.user_id,
        package_id,
    )

    store = _get_evidence_store()
    package = store.get(package_id)

    if package is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence package {package_id} not found",
        )

    # Authorization check
    operator_id = user.operator_id or user.user_id
    package_operator = package.get("operator_id", "")
    if package_operator != operator_id and "admin" not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this evidence package",
        )

    response_data = package.get("response_data", {})
    return EvidencePackageResponse(**response_data)


# ---------------------------------------------------------------------------
# GET /{package_id}/download
# ---------------------------------------------------------------------------


@router.get(
    "/{package_id}/download",
    summary="Download evidence package",
    description=(
        "Download the evidence package in the format specified during "
        "generation. For JSON format, returns the data directly. For "
        "CSV/PDF, returns the formatted content."
    ),
    responses={
        200: {"description": "Evidence package download"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Package not found"},
    },
)
async def download_evidence_package(
    package_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:evidence:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> JSONResponse:
    """Download evidence package in the requested format.

    Args:
        package_id: Evidence package identifier.
        user: Authenticated user with evidence:read permission.

    Returns:
        JSON response with evidence content or download metadata.

    Raises:
        HTTPException: 404 if package not found, 403 if unauthorized.
    """
    logger.info(
        "Evidence download: user=%s package_id=%s",
        user.user_id,
        package_id,
    )

    store = _get_evidence_store()
    package = store.get(package_id)

    if package is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence package {package_id} not found",
        )

    # Authorization check
    operator_id = user.operator_id or user.user_id
    package_operator = package.get("operator_id", "")
    if package_operator != operator_id and "admin" not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to download this evidence package",
        )

    output_format = package.get("format", "json")
    response_data = package.get("response_data", {})

    if output_format == "json":
        return JSONResponse(
            content=response_data,
            headers={
                "Content-Disposition": f"attachment; filename=evidence_{package_id}.json",
            },
        )
    elif output_format == "csv":
        # Generate CSV content from the evidence data
        csv_content = _generate_csv_content(response_data)
        return JSONResponse(
            content={
                "package_id": package_id,
                "format": "csv",
                "content": csv_content,
                "filename": f"evidence_{package_id}.csv",
            },
            headers={
                "Content-Disposition": f"attachment; filename=evidence_{package_id}.csv",
            },
        )
    else:
        # PDF-ready data structure
        return JSONResponse(
            content={
                "package_id": package_id,
                "format": "pdf",
                "pdf_data": response_data,
                "filename": f"evidence_{package_id}.pdf",
                "note": "Use a PDF rendering service to generate the final PDF",
            },
            headers={
                "Content-Disposition": f"attachment; filename=evidence_{package_id}.pdf",
            },
        )


def _generate_csv_content(data: Dict[str, Any]) -> str:
    """Generate CSV string from evidence package data.

    Args:
        data: Evidence package response data dictionary.

    Returns:
        CSV-formatted string with key evidence fields.
    """
    lines = [
        "plot_id,detection_date,deforestation_detected,change_classification,"
        "ndvi_delta,confidence,forest_loss_ha",
    ]

    for detection in data.get("change_detections", []):
        lines.append(
            f"{data.get('plot_id', '')},"
            f"{detection.get('analysis_date', '')},"
            f"{detection.get('deforestation_detected', '')},"
            f"{detection.get('change_classification', '')},"
            f"{detection.get('ndvi_delta', '')},"
            f"{detection.get('confidence', '')},"
            f"{detection.get('forest_loss_ha', '')}"
        )

    return "\n".join(lines)
