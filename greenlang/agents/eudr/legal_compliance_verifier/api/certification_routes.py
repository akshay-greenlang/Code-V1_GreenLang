# -*- coding: utf-8 -*-
"""
Certification Routes - AGENT-EUDR-023 Legal Compliance Verifier API

Endpoints for certification validation, listing, detail retrieval, and
EUDR equivalence assessment for third-party sustainability certifications
(FSC, PEFC, RSPO, Rainforest Alliance, etc.) per EUDR Articles 4, 10, 12.

Endpoints:
    POST /certifications/validate          - Validate a certification
    GET  /certifications                   - List certifications (paginated)
    GET  /certifications/{cert_id}         - Get certification details
    POST /certifications/eudr-equivalence  - Check EUDR equivalence

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023, CertificationValidationEngine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.legal_compliance_verifier.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_certification_engine,
    get_pagination,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.legal_compliance_verifier.api.schemas import (
    CertDetailResponse,
    CertificationEntry,
    CertificationTypeEnum,
    CertListResponse,
    CertStatusEnum,
    CertValidateRequest,
    CertValidateResponse,
    EUDRCommodityEnum,
    EUDREquivalenceRequest,
    EUDREquivalenceResponse,
    EquivalenceGapEntry,
    EquivalenceResultEnum,
    ErrorResponse,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/certifications", tags=["Certifications"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /certifications/validate
# ---------------------------------------------------------------------------


@router.post(
    "/validate",
    response_model=CertValidateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Validate a certification",
    description=(
        "Validate a third-party sustainability certification (FSC, PEFC, "
        "RSPO, etc.) for authenticity, validity, and scope coverage. "
        "Checks certificate number against issuing body databases when "
        "available and assesses EUDR equivalence."
    ),
    responses={
        201: {"description": "Certification validated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def validate_certification(
    request: Request,
    body: CertValidateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:certification:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> CertValidateResponse:
    """Validate a sustainability certification.

    Args:
        body: Certification validation request.
        user: Authenticated user with certification:create permission.

    Returns:
        CertValidateResponse with validation results.
    """
    start = time.monotonic()

    try:
        engine = get_certification_engine()
        result = engine.validate(
            certification_type=body.certification_type.value,
            certificate_number=body.certificate_number,
            holder_name=body.holder_name,
            holder_country=body.holder_country,
            issue_date=body.issue_date,
            expiry_date=body.expiry_date,
            scope=body.scope,
            commodities=[c.value for c in body.commodities] if body.commodities else None,
            supplier_id=body.supplier_id,
            verification_url=body.verification_url,
            validated_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Certification validation failed: invalid data",
            )

        certification = CertificationEntry(
            cert_id=result.get("cert_id", ""),
            certification_type=CertificationTypeEnum(
                result.get("certification_type", body.certification_type.value)
            ),
            certificate_number=result.get("certificate_number", body.certificate_number),
            status=CertStatusEnum(result.get("status", "valid")),
            holder_name=result.get("holder_name", body.holder_name),
            holder_country=result.get("holder_country", body.holder_country),
            issue_date=result.get("issue_date", body.issue_date),
            expiry_date=result.get("expiry_date", body.expiry_date),
            scope=result.get("scope", body.scope),
            commodities=[EUDRCommodityEnum(c) for c in result.get("commodities", [])]
            if result.get("commodities") else [],
            eudr_equivalence=EquivalenceResultEnum(result["eudr_equivalence"])
            if result.get("eudr_equivalence") else None,
            validation_score=Decimal(str(result["validation_score"]))
            if result.get("validation_score") is not None else None,
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"cert_validate:{body.certificate_number}",
            certification.cert_id,
        )

        logger.info(
            "Certification validated: id=%s type=%s number=%s status=%s user=%s",
            certification.cert_id,
            body.certification_type.value,
            body.certificate_number,
            certification.status.value,
            user.user_id,
        )

        return CertValidateResponse(
            certification=certification,
            validation_details=result.get("validation_details", {}),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["CertificationValidationEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Certification validation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Certification validation failed",
        )


# ---------------------------------------------------------------------------
# GET /certifications
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=CertListResponse,
    summary="List certifications",
    description=(
        "Retrieve a paginated list of validated certifications with "
        "optional filtering by type, status, commodity, and supplier."
    ),
    responses={
        200: {"description": "Certifications retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_certifications(
    request: Request,
    certification_type: Optional[CertificationTypeEnum] = Query(
        None, description="Filter by certification type"
    ),
    cert_status: Optional[CertStatusEnum] = Query(
        None, alias="status", description="Filter by validation status"
    ),
    commodity: Optional[EUDRCommodityEnum] = Query(
        None, description="Filter by commodity"
    ),
    supplier_id: Optional[str] = Query(
        None, description="Filter by supplier ID"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-lcv:certification:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CertListResponse:
    """List validated certifications with pagination.

    Args:
        certification_type: Optional type filter.
        cert_status: Optional status filter.
        commodity: Optional commodity filter.
        supplier_id: Optional supplier filter.
        pagination: Pagination parameters.
        user: Authenticated user with certification:read permission.

    Returns:
        CertListResponse with paginated certifications.
    """
    start = time.monotonic()

    try:
        engine = get_certification_engine()
        result = engine.list_certifications(
            certification_type=certification_type.value if certification_type else None,
            status=cert_status.value if cert_status else None,
            commodity=commodity.value if commodity else None,
            supplier_id=supplier_id,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        certifications = []
        for c in result.get("certifications", []):
            certifications.append(
                CertificationEntry(
                    cert_id=c.get("cert_id", ""),
                    certification_type=CertificationTypeEnum(
                        c.get("certification_type", "other")
                    ),
                    certificate_number=c.get("certificate_number", ""),
                    status=CertStatusEnum(c.get("status", "valid")),
                    holder_name=c.get("holder_name"),
                    holder_country=c.get("holder_country"),
                    issue_date=c.get("issue_date"),
                    expiry_date=c.get("expiry_date"),
                    scope=c.get("scope"),
                    commodities=[EUDRCommodityEnum(x) for x in c.get("commodities", [])],
                    eudr_equivalence=EquivalenceResultEnum(c["eudr_equivalence"])
                    if c.get("eudr_equivalence") else None,
                    validation_score=Decimal(str(c["validation_score"]))
                    if c.get("validation_score") is not None else None,
                )
            )

        total = result.get("total", len(certifications))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"cert_list:{certification_type}:{cert_status}",
            str(total),
        )

        logger.info(
            "Certifications listed: total=%d user=%s",
            total,
            user.user_id,
        )

        return CertListResponse(
            certifications=certifications,
            total_certifications=total,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["CertificationValidationEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Certification listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Certification listing failed",
        )


# ---------------------------------------------------------------------------
# GET /certifications/{cert_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{cert_id}",
    response_model=CertDetailResponse,
    summary="Get certification details",
    description=(
        "Retrieve full details of a validated certification including "
        "validation details, verification URL, and audit trail."
    ),
    responses={
        200: {"description": "Certification details retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Certification not found"},
    },
)
async def get_certification_detail(
    cert_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:certification:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CertDetailResponse:
    """Get detailed information about a certification.

    Args:
        cert_id: Unique certification record identifier.
        user: Authenticated user with certification:read permission.

    Returns:
        CertDetailResponse with full certification details.
    """
    start = time.monotonic()

    try:
        engine = get_certification_engine()
        result = engine.get_detail(cert_id=cert_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Certification not found: {cert_id}",
            )

        certification = CertificationEntry(
            cert_id=result.get("cert_id", cert_id),
            certification_type=CertificationTypeEnum(
                result.get("certification_type", "other")
            ),
            certificate_number=result.get("certificate_number", ""),
            status=CertStatusEnum(result.get("status", "valid")),
            holder_name=result.get("holder_name"),
            holder_country=result.get("holder_country"),
            issue_date=result.get("issue_date"),
            expiry_date=result.get("expiry_date"),
            scope=result.get("scope"),
            commodities=[EUDRCommodityEnum(c) for c in result.get("commodities", [])],
            eudr_equivalence=EquivalenceResultEnum(result["eudr_equivalence"])
            if result.get("eudr_equivalence") else None,
            validation_score=Decimal(str(result["validation_score"]))
            if result.get("validation_score") is not None else None,
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"cert_detail:{cert_id}",
            certification.certificate_number,
        )

        logger.info(
            "Certification detail retrieved: id=%s user=%s",
            cert_id,
            user.user_id,
        )

        return CertDetailResponse(
            certification=certification,
            validation_details=result.get("validation_details", {}),
            verification_url=result.get("verification_url"),
            supplier_id=result.get("supplier_id"),
            related_documents=result.get("related_documents", []),
            audit_trail=result.get("audit_trail", []),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["CertificationValidationEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Certification detail retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Certification detail retrieval failed",
        )


# ---------------------------------------------------------------------------
# POST /certifications/eudr-equivalence
# ---------------------------------------------------------------------------


@router.post(
    "/eudr-equivalence",
    response_model=EUDREquivalenceResponse,
    summary="Check EUDR equivalence of certification",
    description=(
        "Assess whether a certification scheme meets EUDR requirements for "
        "specific commodities. Performs gap analysis against EUDR Articles "
        "4, 10, 12 and provides remediation guidance for identified gaps."
    ),
    responses={
        200: {"description": "Equivalence assessment completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def check_eudr_equivalence(
    request: Request,
    body: EUDREquivalenceRequest,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:certification:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> EUDREquivalenceResponse:
    """Check EUDR equivalence of a certification scheme.

    Args:
        body: Equivalence check request with certification type and commodities.
        user: Authenticated user with certification:read permission.

    Returns:
        EUDREquivalenceResponse with equivalence assessment and gap analysis.
    """
    start = time.monotonic()

    try:
        engine = get_certification_engine()
        result = engine.check_eudr_equivalence(
            certification_type=body.certification_type.value,
            certificate_number=body.certificate_number,
            commodities=[c.value for c in body.commodities],
            country_code=body.country_code,
            include_gap_analysis=body.include_gap_analysis,
        )

        gaps = []
        for g in result.get("gaps", []):
            gaps.append(
                EquivalenceGapEntry(
                    requirement=g.get("requirement", ""),
                    eudr_article=g.get("eudr_article", ""),
                    gap_severity=g.get("gap_severity", "medium"),
                    description=g.get("description", ""),
                    remediation_suggestion=g.get("remediation_suggestion"),
                )
            )

        equivalence_result = EquivalenceResultEnum(
            result.get("equivalence_result", "under_review")
        )
        equivalence_score = Decimal(str(result.get("equivalence_score", 0)))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"eudr_equivalence:{body.certification_type.value}:{body.commodities}",
            str(equivalence_result.value),
        )

        logger.info(
            "EUDR equivalence: type=%s result=%s score=%s gaps=%d user=%s",
            body.certification_type.value,
            equivalence_result.value,
            equivalence_score,
            len(gaps),
            user.user_id,
        )

        return EUDREquivalenceResponse(
            certification_type=body.certification_type,
            equivalence_result=equivalence_result,
            equivalence_score=equivalence_score,
            commodities_assessed=body.commodities,
            requirements_met=result.get("requirements_met", 0),
            requirements_total=result.get("requirements_total", 0),
            gaps=gaps,
            regulatory_reference=result.get(
                "regulatory_reference",
                "EU 2023/1115 Article 4, Article 10",
            ),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=[
                    "CertificationValidationEngine",
                    "EUDR Requirements Database",
                ],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("EUDR equivalence check failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="EUDR equivalence check failed",
        )
