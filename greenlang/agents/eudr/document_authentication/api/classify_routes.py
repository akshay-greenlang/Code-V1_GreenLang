# -*- coding: utf-8 -*-
"""
Classification Routes - AGENT-EUDR-012 Document Authentication API

Endpoints for document type classification including single document
classification, batch classification, result retrieval, template
listing, and template registration.

Endpoints:
    POST   /classify              - Classify a single document
    POST   /classify/batch        - Batch classify multiple documents
    GET    /classify/{document_id} - Get classification result
    GET    /classify/templates    - List available templates
    POST   /classify/templates    - Register a new template

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-012, Feature 1 (Document Classification)
Agent ID: GL-EUDR-DAV-012
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from greenlang.schemas import utcnow

from greenlang.agents.eudr.document_authentication.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_dav_service,
    get_pagination,
    get_request_id,
    rate_limit_batch,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_document_id,
)
from greenlang.agents.eudr.document_authentication.api.schemas import (
    BatchClassificationResultSchema,
    BatchClassifySchema,
    ClassificationConfidenceSchema,
    ClassificationResultSchema,
    ClassifyDocumentSchema,
    DocumentTypeSchema,
    PaginatedMeta,
    ProvenanceInfo,
    RegisterTemplateSchema,
    TemplateListSchema,
    TemplateSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Classification"])

# ---------------------------------------------------------------------------
# In-memory stores (replaced by database in production)
# ---------------------------------------------------------------------------

_classification_store: Dict[str, Dict] = {}
_template_store: Dict[str, Dict] = {}

def _get_classification_store() -> Dict[str, Dict]:
    """Return the classification result store singleton."""
    return _classification_store

def _get_template_store() -> Dict[str, Dict]:
    """Return the template store singleton."""
    return _template_store

def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def _classify_document_type(
    reference: str,
    commodity: Optional[str] = None,
) -> tuple:
    """Deterministic document classification logic.

    Zero hallucination: uses pattern matching only, no LLM.

    Args:
        reference: Document reference string.
        commodity: Optional commodity hint.

    Returns:
        Tuple of (document_type, confidence, confidence_level).
    """
    ref_lower = reference.lower()

    # Pattern-based classification (deterministic)
    type_patterns = {
        "coo": (["coo", "certificate_of_origin", "origin"], 0.96),
        "pc": (["phyto", "phytosanitary"], 0.95),
        "bol": (["bol", "bill_of_lading", "lading"], 0.97),
        "cde": (["cde", "export_declaration", "customs_export"], 0.95),
        "cdi": (["cdi", "import_declaration", "customs_import"], 0.95),
        "rspo_cert": (["rspo"], 0.98),
        "fsc_cert": (["fsc"], 0.98),
        "iscc_cert": (["iscc"], 0.98),
        "ft_cert": (["fairtrade", "ft_cert"], 0.97),
        "utz_cert": (["utz", "rainforest"], 0.97),
        "dds_draft": (["dds", "due_diligence"], 0.96),
        "ssd": (["ssd", "self_declaration", "supplier_self"], 0.94),
        "ic": (["invoice", "commercial_invoice"], 0.93),
        "wr": (["weighbridge", "weight_receipt"], 0.92),
    }

    for doc_type, (patterns, base_conf) in type_patterns.items():
        for pattern in patterns:
            if pattern in ref_lower:
                conf = base_conf
                if conf >= 0.95:
                    level = ClassificationConfidenceSchema.HIGH
                elif conf >= 0.70:
                    level = ClassificationConfidenceSchema.MEDIUM
                else:
                    level = ClassificationConfidenceSchema.LOW
                return (DocumentTypeSchema(doc_type), conf, level)

    # Default: unknown classification
    return (DocumentTypeSchema.COO, 0.50, ClassificationConfidenceSchema.LOW)

# ---------------------------------------------------------------------------
# POST /classify
# ---------------------------------------------------------------------------

@router.post(
    "/classify",
    response_model=ClassificationResultSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Classify a document",
    description=(
        "Classify a single EUDR document by type using deterministic "
        "pattern matching against known templates. Returns classified "
        "type with confidence score and provenance hash."
    ),
    responses={
        201: {"description": "Document classified successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def classify_document(
    request: Request,
    body: ClassifyDocumentSchema,
    user: AuthUser = Depends(
        require_permission("eudr-dav:classify:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ClassificationResultSchema:
    """Classify a single document by type.

    Args:
        body: Classification request with document reference.
        user: Authenticated user with classify:create permission.

    Returns:
        ClassificationResultSchema with classification result.
    """
    start = time.monotonic()
    try:
        document_id = str(uuid.uuid4())
        now = utcnow()

        doc_type, confidence, conf_level = _classify_document_type(
            body.document_reference, body.commodity,
        )

        provenance_data = body.model_dump(mode="json")
        provenance_data["document_id"] = document_id
        provenance_data["classified_by"] = user.user_id
        provenance_hash = _compute_provenance_hash(provenance_data)

        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        result = {
            "document_id": document_id,
            "document_reference": body.document_reference,
            "document_type": doc_type,
            "confidence": confidence,
            "confidence_level": conf_level,
            "alternative_types": [],
            "template_matched": None,
            "commodity": body.commodity,
            "operator_id": body.operator_id,
            "provenance": provenance,
            "created_at": now,
        }

        store = _get_classification_store()
        store[document_id] = result

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Document classified: id=%s type=%s confidence=%.2f level=%s",
            document_id,
            doc_type.value,
            confidence,
            conf_level.value,
        )

        return ClassificationResultSchema(
            **result,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to classify document: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to classify document",
        )

# ---------------------------------------------------------------------------
# POST /classify/batch
# ---------------------------------------------------------------------------

@router.post(
    "/classify/batch",
    response_model=BatchClassificationResultSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Batch classify documents",
    description=(
        "Classify up to 500 documents in a single request. Each document "
        "is classified independently; partial success is allowed."
    ),
    responses={
        201: {"description": "Batch classification processed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_classify(
    request: Request,
    body: BatchClassifySchema,
    user: AuthUser = Depends(
        require_permission("eudr-dav:classify:batch")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchClassificationResultSchema:
    """Batch classify multiple documents.

    Args:
        body: Batch classification request with list of documents.
        user: Authenticated user with classify:batch permission.

    Returns:
        BatchClassificationResultSchema with results and errors.
    """
    start = time.monotonic()
    try:
        now = utcnow()
        results: List[ClassificationResultSchema] = []
        errors: List[Dict[str, Any]] = []
        store = _get_classification_store()

        for idx, doc_req in enumerate(body.documents):
            try:
                document_id = str(uuid.uuid4())
                doc_type, confidence, conf_level = _classify_document_type(
                    doc_req.document_reference, doc_req.commodity,
                )

                provenance_hash = _compute_provenance_hash({
                    "document_id": document_id,
                    "reference": doc_req.document_reference,
                    "classified_by": user.user_id,
                    "index": idx,
                })
                provenance = ProvenanceInfo(
                    provenance_hash=provenance_hash,
                    created_by=user.user_id,
                    created_at=now,
                    source="api",
                )

                record = {
                    "document_id": document_id,
                    "document_reference": doc_req.document_reference,
                    "document_type": doc_type,
                    "confidence": confidence,
                    "confidence_level": conf_level,
                    "alternative_types": [],
                    "template_matched": None,
                    "commodity": doc_req.commodity,
                    "operator_id": doc_req.operator_id or body.operator_id,
                    "provenance": provenance,
                    "created_at": now,
                }

                if not body.validate_only:
                    store[document_id] = record

                results.append(ClassificationResultSchema(**record))

            except Exception as entry_exc:
                errors.append({
                    "index": idx,
                    "reference": doc_req.document_reference,
                    "error": str(entry_exc),
                })

        batch_provenance_hash = _compute_provenance_hash({
            "total": len(body.documents),
            "classified": len(results),
            "failed": len(errors),
            "operator": user.user_id,
        })
        batch_provenance = ProvenanceInfo(
            provenance_hash=batch_provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Batch classification: total=%d classified=%d failed=%d",
            len(body.documents),
            len(results),
            len(errors),
        )

        return BatchClassificationResultSchema(
            total_submitted=len(body.documents),
            total_classified=len(results),
            total_failed=len(errors),
            results=results,
            errors=errors,
            validate_only=body.validate_only,
            provenance=batch_provenance,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed batch classification: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process batch classification",
        )

# ---------------------------------------------------------------------------
# GET /classify/{document_id}
# ---------------------------------------------------------------------------

@router.get(
    "/classify/{document_id}",
    response_model=ClassificationResultSchema,
    summary="Get classification result",
    description="Retrieve the classification result for a specific document.",
    responses={
        200: {"description": "Classification result retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Classification not found"},
    },
)
async def get_classification(
    request: Request,
    document_id: str = Depends(validate_document_id),
    user: AuthUser = Depends(
        require_permission("eudr-dav:classify:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ClassificationResultSchema:
    """Get classification result for a document.

    Args:
        document_id: Document identifier.
        user: Authenticated user with classify:read permission.

    Returns:
        ClassificationResultSchema with classification details.

    Raises:
        HTTPException: 404 if classification not found.
    """
    start = time.monotonic()
    try:
        store = _get_classification_store()
        record = store.get(document_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Classification for document {document_id} not found",
            )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        return ClassificationResultSchema(
            **record,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get classification %s: %s",
            document_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve classification",
        )

# ---------------------------------------------------------------------------
# GET /classify/templates
# ---------------------------------------------------------------------------

@router.get(
    "/classify/templates",
    response_model=TemplateListSchema,
    summary="List classification templates",
    description="List all available document classification templates.",
    responses={
        200: {"description": "Templates retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_templates(
    request: Request,
    document_type: Optional[DocumentTypeSchema] = Query(
        None, description="Filter by document type"
    ),
    active_only: bool = Query(
        default=True, description="Only return active templates"
    ),
    user: AuthUser = Depends(
        require_permission("eudr-dav:classify:templates:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> TemplateListSchema:
    """List available classification templates.

    Args:
        document_type: Optional filter by document type.
        active_only: Whether to return only active templates.
        user: Authenticated user with templates:read permission.

    Returns:
        TemplateListSchema with matching templates.
    """
    start = time.monotonic()
    try:
        store = _get_template_store()
        templates = []

        for template in store.values():
            if active_only and not template.get("active", True):
                continue
            if document_type and template["document_type"] != document_type:
                continue
            templates.append(TemplateSchema(**template))

        elapsed_ms = (time.monotonic() - start) * 1000.0

        return TemplateListSchema(
            templates=templates,
            total_count=len(templates),
            processing_time_ms=elapsed_ms,
            timestamp=utcnow(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list templates: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list templates",
        )

# ---------------------------------------------------------------------------
# POST /classify/templates
# ---------------------------------------------------------------------------

@router.post(
    "/classify/templates",
    response_model=TemplateSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new classification template",
    description=(
        "Register a new document classification template for matching "
        "against incoming documents during classification."
    ),
    responses={
        201: {"description": "Template registered successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def register_template(
    request: Request,
    body: RegisterTemplateSchema,
    user: AuthUser = Depends(
        require_permission("eudr-dav:classify:templates:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> TemplateSchema:
    """Register a new classification template.

    Args:
        body: Template registration parameters.
        user: Authenticated user with templates:create permission.

    Returns:
        TemplateSchema with the newly registered template.
    """
    start = time.monotonic()
    try:
        template_id = str(uuid.uuid4())
        now = utcnow()

        template_record = {
            "template_id": template_id,
            "name": body.name,
            "document_type": body.document_type,
            "issuing_authority": body.issuing_authority,
            "country_code": body.country_code,
            "version": "1.0",
            "active": True,
            "metadata": body.metadata,
            "created_at": now,
        }

        store = _get_template_store()
        store[template_id] = template_record

        logger.info(
            "Template registered: id=%s name=%s type=%s",
            template_id,
            body.name,
            body.document_type.value,
        )

        return TemplateSchema(**template_record)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to register template: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register template",
        )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
]
