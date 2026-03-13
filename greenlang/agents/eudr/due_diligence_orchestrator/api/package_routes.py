# -*- coding: utf-8 -*-
"""
DDS Package Routes - AGENT-EUDR-026 Due Diligence Orchestrator API

Endpoints for generating, retrieving, downloading, and validating
Due Diligence Statement (DDS) packages per EUDR Article 12. Packages
compile all 25 agent outputs into a structured, audit-ready evidence
bundle with SHA-256 integrity hashes and provenance chain.

Endpoints (4):
    POST /workflows/{id}/package    - Generate a DDS package
    GET  /packages/{package_id}     - Get package details
    GET  /packages/{package_id}/download - Download package in specified format
    POST /packages/validate         - Validate a package against DDS schema

RBAC Permissions:
    eudr-ddo:packages:generate  - Generate DD packages
    eudr-ddo:packages:read      - View DD packages
    eudr-ddo:packages:download  - Download packages (PDF, ZIP, JSON)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from greenlang.agents.eudr.due_diligence_orchestrator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_ddo_service,
    rate_limit_package,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    GeneratePackageRequest,
    PackageGenerationResponse,
    WorkflowStatus,
    _utcnow,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["DDS Package Management"])


# ---------------------------------------------------------------------------
# POST /workflows/{workflow_id}/package -- Generate DDS package
# ---------------------------------------------------------------------------


@router.post(
    "/workflows/{workflow_id}/package",
    response_model=PackageGenerationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a due diligence package",
    description=(
        "Generate a complete Due Diligence Statement (DDS) package from "
        "the workflow outputs per EUDR Article 12(2). Compiles all 25 "
        "agent outputs into a structured evidence bundle with DDS-compatible "
        "JSON, human-readable PDF, and SHA-256 integrity hashes."
    ),
    responses={
        201: {"description": "Package generated successfully"},
        400: {"model": ErrorResponse, "description": "Workflow not ready for package generation"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Workflow not found"},
        409: {"model": ErrorResponse, "description": "Quality gates not passed"},
    },
)
async def generate_package(
    request: Request,
    workflow_id: str,
    formats: Optional[str] = Query(
        default="json,pdf",
        description="Comma-separated output formats: json, pdf, html, zip",
    ),
    language: str = Query(
        default="en",
        description="Report language: en, fr, de, es, pt",
    ),
    include_executive_summary: bool = Query(
        default=True,
        description="Whether to include executive summary",
    ),
    include_evidence_annexes: bool = Query(
        default=True,
        description="Whether to include evidence annexes",
    ),
    user: AuthUser = Depends(require_permission("eudr-ddo:packages:generate")),
    _rate: AuthUser = Depends(rate_limit_package),
) -> PackageGenerationResponse:
    """Generate a due diligence evidence package.

    Compiles all agent outputs from the completed workflow into a
    DDS-compatible package per EUDR Article 12(2). The package includes
    sections for product identification, origin, geolocation, deforestation
    verification, legal compliance, risk assessment, mitigation, supply
    chain traceability, and audit trail.

    Args:
        request: FastAPI request object.
        workflow_id: Unique workflow identifier.
        formats: Comma-separated output formats.
        language: Report language (ISO 639-1).
        include_executive_summary: Whether to include summary.
        include_evidence_annexes: Whether to include annexes.
        user: Authenticated and authorized user.

    Returns:
        PackageGenerationResponse with the generated package.

    Raises:
        HTTPException: 409 if quality gates not passed.
    """
    logger.info(
        "generate_package: user=%s workflow_id=%s formats=%s language=%s",
        user.user_id,
        workflow_id,
        formats,
        language,
    )

    # Validate language
    valid_languages = {"en", "fr", "de", "es", "pt"}
    if language not in valid_languages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid language: {language}. Valid: {', '.join(sorted(valid_languages))}",
        )

    service = get_ddo_service()

    # Verify workflow exists and is in appropriate state
    state = service._state_manager.get_state(workflow_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )

    # Package generation is allowed for running (post QG-3), completing, or completed workflows
    allowed_states = {
        WorkflowStatus.RUNNING,
        WorkflowStatus.COMPLETING,
        WorkflowStatus.COMPLETED,
    }
    if state.status not in allowed_states:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot generate package for workflow in {state.status.value} status. "
                f"Workflow must have passed all quality gates."
            ),
        )

    format_list = [f.strip() for f in (formats or "json,pdf").split(",")]
    valid_formats = {"json", "pdf", "html", "zip"}
    invalid = set(format_list) - valid_formats
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid formats: {', '.join(invalid)}. Valid: {', '.join(sorted(valid_formats))}",
        )

    package_request = GeneratePackageRequest(
        workflow_id=workflow_id,
        formats=format_list,
        language=language,
        include_executive_summary=include_executive_summary,
        include_evidence_annexes=include_evidence_annexes,
    )

    try:
        return service.generate_package(package_request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# GET /packages/{package_id} -- Get package details
# ---------------------------------------------------------------------------


@router.get(
    "/packages/{package_id}",
    status_code=status.HTTP_200_OK,
    summary="Get DDS package details",
    description=(
        "Get the details of a generated DDS package including all "
        "sections, quality gate results, risk profile, integrity hash, "
        "and available download URLs."
    ),
    responses={
        200: {"description": "Package retrieved successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Package not found"},
    },
)
async def get_package(
    request: Request,
    package_id: str,
    user: AuthUser = Depends(require_permission("eudr-ddo:packages:read")),
    _rate: AuthUser = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """Get DDS package details.

    Returns the full package metadata including all sections, quality
    gate summaries, risk profile, integrity hash, and download URLs
    for available formats.

    Args:
        request: FastAPI request object.
        package_id: Unique package identifier.
        user: Authenticated and authorized user.

    Returns:
        Dictionary with package details and section summaries.

    Raises:
        HTTPException: 404 if package not found.
    """
    logger.info(
        "get_package: user=%s package_id=%s",
        user.user_id,
        package_id,
    )

    service = get_ddo_service()
    package = service._package_generator.get_package(package_id)
    if package is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Package {package_id} not found",
        )

    sections_summary = []
    for section in package.sections:
        sections_summary.append({
            "section_number": section.section_number,
            "title": section.title,
            "fields_count": len(section.fields),
            "completeness_pct": str(section.completeness_pct),
        })

    return {
        "package_id": package.package_id,
        "workflow_id": package.workflow_id,
        "dds_schema_version": package.dds_schema_version,
        "operator_id": package.operator_id,
        "operator_name": package.operator_name,
        "commodity": package.commodity.value if package.commodity and hasattr(package.commodity, "value") else package.commodity,
        "workflow_type": package.workflow_type.value if hasattr(package.workflow_type, "value") else str(package.workflow_type),
        "sections": sections_summary,
        "total_agents_executed": package.total_agents_executed,
        "total_duration_ms": str(package.total_duration_ms) if package.total_duration_ms else None,
        "language": package.language,
        "integrity_hash": package.integrity_hash,
        "download_urls": package.download_urls,
        "generated_at": package.generated_at.isoformat() if package.generated_at else None,
        "generated_by": package.generated_by,
        "provenance_hash": package.provenance_hash,
    }


# ---------------------------------------------------------------------------
# GET /packages/{package_id}/download -- Download package
# ---------------------------------------------------------------------------


@router.get(
    "/packages/{package_id}/download",
    status_code=status.HTTP_200_OK,
    summary="Download DDS package",
    description=(
        "Download a generated DDS package in the specified format. "
        "Available formats: json (DDS-compatible JSON), pdf (human-readable "
        "report), html (web format), zip (complete evidence bundle)."
    ),
    responses={
        200: {"description": "Download URL or content returned"},
        400: {"model": ErrorResponse, "description": "Invalid format"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Package not found"},
    },
)
async def download_package(
    request: Request,
    package_id: str,
    format: str = Query(
        default="json",
        description="Download format: json, pdf, html, zip",
    ),
    user: AuthUser = Depends(require_permission("eudr-ddo:packages:download")),
    _rate: AuthUser = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """Download a DDS package.

    Returns a download URL or the package content in the requested
    format. Large packages (PDF, ZIP) return a pre-signed S3 URL.
    JSON format can be returned inline.

    Args:
        request: FastAPI request object.
        package_id: Unique package identifier.
        format: Download format (json, pdf, html, zip).
        user: Authenticated and authorized user.

    Returns:
        Dictionary with download URL or inline content.

    Raises:
        HTTPException: 400 if invalid format, 404 if package not found.
    """
    logger.info(
        "download_package: user=%s package_id=%s format=%s",
        user.user_id,
        package_id,
        format,
    )

    valid_formats = {"json", "pdf", "html", "zip"}
    if format not in valid_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid format: {format}. Valid: {', '.join(sorted(valid_formats))}",
        )

    service = get_ddo_service()
    package = service._package_generator.get_package(package_id)
    if package is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Package {package_id} not found",
        )

    # Check if download URL exists for the requested format
    download_url = package.download_urls.get(format)

    if download_url:
        return {
            "package_id": package_id,
            "format": format,
            "download_url": download_url,
            "integrity_hash": package.integrity_hash,
            "expires_in_seconds": 3600,
        }

    # For JSON format, return inline if no URL
    if format == "json":
        return {
            "package_id": package_id,
            "format": "json",
            "content": package.model_dump(mode="json") if hasattr(package, "model_dump") else {},
            "integrity_hash": package.integrity_hash,
        }

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Format '{format}' not available for package {package_id}. Generate the package with this format first.",
    )


# ---------------------------------------------------------------------------
# POST /packages/validate -- Validate package against DDS schema
# ---------------------------------------------------------------------------


@router.post(
    "/packages/validate",
    status_code=status.HTTP_200_OK,
    summary="Validate package against DDS schema",
    description=(
        "Validate a DDS package against the EU Information System schema. "
        "Checks all mandatory Article 12(2) fields, data types, value "
        "ranges, and cross-field consistency rules."
    ),
    responses={
        200: {"description": "Validation result returned"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Package not found"},
    },
)
async def validate_package(
    request: Request,
    package_id: str = Query(
        ...,
        description="Package ID to validate",
    ),
    user: AuthUser = Depends(require_permission("eudr-ddo:packages:read")),
    _rate: AuthUser = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """Validate a DDS package against the EU Information System schema.

    Runs the full DDS schema validation suite checking all mandatory
    Article 12(2)(a-j) fields, data types, value ranges, and
    cross-field consistency rules.

    Args:
        request: FastAPI request object.
        package_id: Package identifier to validate.
        user: Authenticated and authorized user.

    Returns:
        Dictionary with validation result, errors, and field coverage.

    Raises:
        HTTPException: 404 if package not found.
    """
    logger.info(
        "validate_package: user=%s package_id=%s",
        user.user_id,
        package_id,
    )

    service = get_ddo_service()
    package = service._package_generator.get_package(package_id)
    if package is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Package {package_id} not found",
        )

    # Run DDS schema validation
    validation_result = service._package_generator.validate_dds_schema(package)

    is_valid = validation_result.get("valid", False) if isinstance(validation_result, dict) else bool(validation_result)
    errors = validation_result.get("errors", []) if isinstance(validation_result, dict) else []
    warnings = validation_result.get("warnings", []) if isinstance(validation_result, dict) else []

    # Calculate field coverage
    total_fields = 0
    populated_fields = 0
    for section in package.sections:
        for field in section.fields:
            total_fields += 1
            if field.value is not None:
                populated_fields += 1

    field_coverage_pct = (
        Decimal(str(populated_fields)) / Decimal(str(total_fields)) * Decimal("100")
    ).quantize(Decimal("0.01")) if total_fields > 0 else Decimal("0")

    return {
        "package_id": package_id,
        "valid": is_valid,
        "dds_schema_version": package.dds_schema_version,
        "errors": errors,
        "warnings": warnings,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "field_coverage": {
            "total_fields": total_fields,
            "populated_fields": populated_fields,
            "coverage_pct": str(field_coverage_pct),
        },
        "integrity_hash": package.integrity_hash,
        "validated_at": _utcnow().isoformat(),
        "validated_by": user.user_id,
    }
