# -*- coding: utf-8 -*-
"""
PDF & Invoice Extractor REST API Router - AGENT-DATA-001: PDF Extractor

FastAPI router providing 20 endpoints for document ingestion, field
extraction, invoice/manifest/utility-bill processing, document
classification, template management, validation, job tracking,
statistics, and health monitoring.

All endpoints are mounted under ``/api/v1/pdf-extractor``.

Endpoints:
    1.  POST   /v1/documents/ingest            - Ingest single document
    2.  POST   /v1/documents/batch             - Batch ingest documents
    3.  GET    /v1/documents                   - List documents
    4.  GET    /v1/documents/{document_id}     - Get document details
    5.  GET    /v1/documents/{document_id}/fields  - Get extracted fields
    6.  GET    /v1/documents/{document_id}/pages   - Get document pages
    7.  POST   /v1/documents/{document_id}/reprocess - Reprocess document
    8.  POST   /v1/documents/classify          - Classify document type
    9.  POST   /v1/invoices/extract            - Extract invoice data
    10. GET    /v1/invoices/{document_id}      - Get invoice result
    11. POST   /v1/manifests/extract           - Extract manifest data
    12. GET    /v1/manifests/{document_id}     - Get manifest result
    13. POST   /v1/utility-bills/extract       - Extract utility bill
    14. GET    /v1/utility-bills/{document_id} - Get utility bill result
    15. POST   /v1/templates                   - Create extraction template
    16. GET    /v1/templates                   - List templates
    17. POST   /v1/validate/{document_id}      - Run validation
    18. GET    /v1/jobs                        - List extraction jobs
    19. GET    /v1/statistics                  - Get statistics
    20. GET    /health                         - Health check

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-001 PDF & Invoice Extractor
Status: Production Ready
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import (no `from __future__ import annotations` here)
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[assignment, misc]
    logger.warning("FastAPI not available; PDF extractor router is None")


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class IngestDocumentBody(BaseModel):
        """Request body for ingesting a single document."""
        file_path: Optional[str] = Field(
            None, description="Server-side file path to ingest",
        )
        file_content: Optional[str] = Field(
            None, description="Raw text content of the document",
        )
        file_base64: Optional[str] = Field(
            None, description="Base64-encoded file content",
        )
        document_type: Optional[str] = Field(
            None, description="Document type hint (invoice, manifest, utility_bill, receipt, other)",
        )
        ocr_engine: Optional[str] = Field(
            None, description="OCR engine override (tesseract, textract, azure, google)",
        )
        confidence_threshold: Optional[float] = Field(
            None, ge=0.0, le=1.0, description="Minimum confidence threshold",
        )
        tenant_id: str = Field(
            default="default", description="Tenant identifier",
        )

    class BatchDocumentItem(BaseModel):
        """Single item within a batch ingest request."""
        file_path: Optional[str] = Field(None, description="Server-side path")
        file_content: Optional[str] = Field(None, description="Raw text content")
        file_base64: Optional[str] = Field(None, description="Base64-encoded content")
        document_type: Optional[str] = Field(None, description="Document type hint")

    class BatchIngestBody(BaseModel):
        """Request body for batch document ingestion."""
        documents: List[BatchDocumentItem] = Field(
            ..., description="List of documents to ingest",
        )
        ocr_engine: Optional[str] = Field(
            None, description="OCR engine override for the batch",
        )
        confidence_threshold: Optional[float] = Field(
            None, ge=0.0, le=1.0, description="Minimum confidence threshold",
        )
        tenant_id: str = Field(
            default="default", description="Tenant identifier",
        )

    class ClassifyDocumentBody(BaseModel):
        """Request body for document classification."""
        text: Optional[str] = Field(
            None, description="Extracted text for classification",
        )
        file_name: Optional[str] = Field(
            None, description="Original filename for heuristic hints",
        )

    class ExtractInvoiceBody(BaseModel):
        """Request body for invoice extraction."""
        document_id: Optional[str] = Field(
            None, description="Existing document ID to extract from",
        )
        text: Optional[str] = Field(
            None, description="Raw text for direct extraction",
        )
        confidence_threshold: Optional[float] = Field(
            None, ge=0.0, le=1.0, description="Minimum confidence threshold",
        )
        template: Optional[str] = Field(
            None, description="Extraction template name to apply",
        )

    class ExtractManifestBody(BaseModel):
        """Request body for manifest extraction."""
        document_id: Optional[str] = Field(
            None, description="Existing document ID to extract from",
        )
        text: Optional[str] = Field(
            None, description="Raw text for direct extraction",
        )
        confidence_threshold: Optional[float] = Field(
            None, ge=0.0, le=1.0, description="Minimum confidence threshold",
        )

    class ExtractUtilityBillBody(BaseModel):
        """Request body for utility bill extraction."""
        document_id: Optional[str] = Field(
            None, description="Existing document ID to extract from",
        )
        text: Optional[str] = Field(
            None, description="Raw text for direct extraction",
        )
        confidence_threshold: Optional[float] = Field(
            None, ge=0.0, le=1.0, description="Minimum confidence threshold",
        )

    class CreateTemplateBody(BaseModel):
        """Request body for creating an extraction template."""
        name: str = Field(..., description="Template name")
        template_type: str = Field(
            ..., description="Template type (invoice, manifest, utility_bill, generic)",
        )
        field_patterns: Dict[str, Any] = Field(
            ..., description="Field name to regex/pattern mapping",
        )
        validation_rules: Dict[str, Any] = Field(
            default_factory=dict, description="Cross-field validation rules",
        )
        description: str = Field(default="", description="Template description")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/pdf-extractor",
        tags=["pdf-extractor"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Request) -> Any:
    """Extract PDFExtractorService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        PDFExtractorService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    service = getattr(request.app.state, "pdf_extractor_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="PDF extractor service not configured",
        )
    return service


if FASTAPI_AVAILABLE:

    # ------------------------------------------------------------------
    # 1. Ingest single document
    # ------------------------------------------------------------------
    @router.post("/v1/documents/ingest")
    async def ingest_document(
        body: IngestDocumentBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Ingest a single document for extraction."""
        service = _get_service(request)
        try:
            record = service.ingest_document(
                file_path=body.file_path,
                file_content=body.file_content,
                file_base64=body.file_base64,
                document_type=body.document_type,
                ocr_engine=body.ocr_engine,
                confidence_threshold=body.confidence_threshold,
                tenant_id=body.tenant_id,
            )
            return record.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 2. Batch ingest documents
    # ------------------------------------------------------------------
    @router.post("/v1/documents/batch")
    async def batch_ingest(
        body: BatchIngestBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Batch ingest multiple documents."""
        service = _get_service(request)
        try:
            documents = [d.model_dump() for d in body.documents]
            batch_job = service.ingest_batch(
                documents=documents,
                ocr_engine=body.ocr_engine,
                confidence_threshold=body.confidence_threshold,
                tenant_id=body.tenant_id,
            )
            return batch_job.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 3. List documents
    # ------------------------------------------------------------------
    @router.get("/v1/documents")
    async def list_documents(
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        tenant_id: Optional[str] = Query(None),
        document_type: Optional[str] = Query(None),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List ingested documents with optional filters."""
        service = _get_service(request)
        filters: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if tenant_id is not None:
            filters["tenant_id"] = tenant_id
        if document_type is not None:
            filters["document_type"] = document_type

        documents = service.list_documents(filters=filters)
        return {
            "documents": [d.model_dump(mode="json") for d in documents],
            "count": len(documents),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 4. Get document details
    # ------------------------------------------------------------------
    @router.get("/v1/documents/{document_id}")
    async def get_document(
        document_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get document details by ID."""
        service = _get_service(request)
        record = service.get_document(document_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found",
            )
        return record.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 5. Get extracted fields
    # ------------------------------------------------------------------
    @router.get("/v1/documents/{document_id}/fields")
    async def get_document_fields(
        document_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get extracted fields for a document."""
        service = _get_service(request)
        record = service.get_document(document_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found",
            )
        return {
            "document_id": document_id,
            "fields": record.extracted_fields,
            "count": len(record.extracted_fields),
        }

    # ------------------------------------------------------------------
    # 6. Get document pages
    # ------------------------------------------------------------------
    @router.get("/v1/documents/{document_id}/pages")
    async def get_document_pages(
        document_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get page-level information for a document."""
        service = _get_service(request)
        record = service.get_document(document_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found",
            )
        return {
            "document_id": document_id,
            "pages": record.pages,
            "page_count": record.page_count,
        }

    # ------------------------------------------------------------------
    # 7. Reprocess document
    # ------------------------------------------------------------------
    @router.post("/v1/documents/{document_id}/reprocess")
    async def reprocess_document(
        document_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Reprocess an already-ingested document."""
        service = _get_service(request)
        try:
            job = service.reprocess_document(document_id)
            return job.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # ------------------------------------------------------------------
    # 8. Classify document type
    # ------------------------------------------------------------------
    @router.post("/v1/documents/classify")
    async def classify_document(
        body: ClassifyDocumentBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Classify a document's type from text or filename."""
        service = _get_service(request)
        try:
            doc_type, confidence = service.classify_document(
                text=body.text,
                file_name=body.file_name,
            )
            return {
                "document_type": doc_type,
                "confidence": confidence,
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 9. Extract invoice data
    # ------------------------------------------------------------------
    @router.post("/v1/invoices/extract")
    async def extract_invoice(
        body: ExtractInvoiceBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Extract structured invoice data from a document."""
        service = _get_service(request)
        try:
            result = service.extract_invoice(
                document_id_or_text=body.document_id or body.text,
                confidence_threshold=body.confidence_threshold,
                template=body.template,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 10. Get invoice result
    # ------------------------------------------------------------------
    @router.get("/v1/invoices/{document_id}")
    async def get_invoice(
        document_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get invoice extraction result by document ID."""
        service = _get_service(request)
        record = service.get_document(document_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Invoice document {document_id} not found",
            )
        return {
            "document_id": document_id,
            "document_type": record.document_type,
            "invoice_data": record.extraction_result,
            "confidence": record.confidence,
            "provenance_hash": record.provenance_hash,
        }

    # ------------------------------------------------------------------
    # 11. Extract manifest data
    # ------------------------------------------------------------------
    @router.post("/v1/manifests/extract")
    async def extract_manifest(
        body: ExtractManifestBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Extract structured manifest data from a document."""
        service = _get_service(request)
        try:
            result = service.extract_manifest(
                document_id_or_text=body.document_id or body.text,
                confidence_threshold=body.confidence_threshold,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 12. Get manifest result
    # ------------------------------------------------------------------
    @router.get("/v1/manifests/{document_id}")
    async def get_manifest(
        document_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get manifest extraction result by document ID."""
        service = _get_service(request)
        record = service.get_document(document_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Manifest document {document_id} not found",
            )
        return {
            "document_id": document_id,
            "document_type": record.document_type,
            "manifest_data": record.extraction_result,
            "confidence": record.confidence,
            "provenance_hash": record.provenance_hash,
        }

    # ------------------------------------------------------------------
    # 13. Extract utility bill
    # ------------------------------------------------------------------
    @router.post("/v1/utility-bills/extract")
    async def extract_utility_bill(
        body: ExtractUtilityBillBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Extract structured utility bill data from a document."""
        service = _get_service(request)
        try:
            result = service.extract_utility_bill(
                document_id_or_text=body.document_id or body.text,
                confidence_threshold=body.confidence_threshold,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 14. Get utility bill result
    # ------------------------------------------------------------------
    @router.get("/v1/utility-bills/{document_id}")
    async def get_utility_bill(
        document_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get utility bill extraction result by document ID."""
        service = _get_service(request)
        record = service.get_document(document_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Utility bill document {document_id} not found",
            )
        return {
            "document_id": document_id,
            "document_type": record.document_type,
            "utility_bill_data": record.extraction_result,
            "confidence": record.confidence,
            "provenance_hash": record.provenance_hash,
        }

    # ------------------------------------------------------------------
    # 15. Create extraction template
    # ------------------------------------------------------------------
    @router.post("/v1/templates")
    async def create_template(
        body: CreateTemplateBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new extraction template."""
        service = _get_service(request)
        try:
            template = service.create_template(
                name=body.name,
                template_type=body.template_type,
                field_patterns=body.field_patterns,
                validation_rules=body.validation_rules,
            )
            return template.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 16. List templates
    # ------------------------------------------------------------------
    @router.get("/v1/templates")
    async def list_templates(
        request: Request,
    ) -> Dict[str, Any]:
        """List all extraction templates."""
        service = _get_service(request)
        templates = service.list_templates()
        return {
            "templates": [t.model_dump(mode="json") for t in templates],
            "count": len(templates),
        }

    # ------------------------------------------------------------------
    # 17. Validate document
    # ------------------------------------------------------------------
    @router.post("/v1/validate/{document_id}")
    async def validate_document(
        document_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Run validation rules on an extracted document."""
        service = _get_service(request)
        try:
            results = service.validate_document(document_id)
            return {
                "document_id": document_id,
                "validation_results": [
                    r.model_dump(mode="json") for r in results
                ],
                "count": len(results),
                "all_passed": all(r.passed for r in results),
            }
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # ------------------------------------------------------------------
    # 18. List extraction jobs
    # ------------------------------------------------------------------
    @router.get("/v1/jobs")
    async def list_jobs(
        status: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=200),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List extraction jobs with optional status filter."""
        service = _get_service(request)
        jobs = service.list_jobs(status=status, limit=limit)
        return {
            "jobs": [j.model_dump(mode="json") for j in jobs],
            "count": len(jobs),
        }

    # ------------------------------------------------------------------
    # 19. Get statistics
    # ------------------------------------------------------------------
    @router.get("/v1/statistics")
    async def get_statistics(
        request: Request,
    ) -> Dict[str, Any]:
        """Get PDF extractor service statistics."""
        service = _get_service(request)
        stats = service.get_statistics()
        return stats.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 20. Health check
    # ------------------------------------------------------------------
    @router.get("/health")
    async def health() -> Dict[str, str]:
        """PDF extractor service health check endpoint."""
        return {"status": "healthy", "service": "pdf-extractor"}


__all__ = [
    "router",
]
