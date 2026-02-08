# -*- coding: utf-8 -*-
"""
Excel & CSV Normalizer REST API Router - AGENT-DATA-002: Excel Normalizer

FastAPI router providing 20 endpoints for file upload, column mapping,
data type detection, schema validation, data normalization, quality
scoring, transform pipelines, template management, job tracking,
statistics, and health monitoring.

All endpoints are mounted under ``/api/v1/excel-normalizer``.

Endpoints:
    1.  POST   /v1/files/upload              - Upload single file
    2.  POST   /v1/files/batch               - Batch upload files
    3.  GET    /v1/files                     - List files
    4.  GET    /v1/files/{file_id}           - Get file details
    5.  GET    /v1/files/{file_id}/sheets    - Get file sheets
    6.  GET    /v1/files/{file_id}/preview   - Preview file data
    7.  POST   /v1/files/{file_id}/reprocess - Reprocess file
    8.  POST   /v1/normalize                 - Normalize inline data
    9.  POST   /v1/columns/map               - Map columns to canonical schema
    10. GET    /v1/columns/canonical          - List canonical fields
    11. POST   /v1/columns/detect-types      - Detect column data types
    12. POST   /v1/validate                  - Validate data against schema
    13. POST   /v1/transform                 - Apply transform pipeline
    14. GET    /v1/quality/{file_id}          - Get quality score
    15. POST   /v1/templates                 - Create mapping template
    16. GET    /v1/templates                 - List templates
    17. GET    /v1/templates/{template_id}   - Get template details
    18. GET    /v1/jobs                      - List normalization jobs
    19. GET    /v1/statistics                - Get statistics
    20. GET    /health                       - Health check

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-002 Excel & CSV Normalizer
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
    logger.warning("FastAPI not available; Excel normalizer router is None")


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class UploadFileBody(BaseModel):
        """Request body for uploading a single file."""
        file_name: str = Field(
            ..., description="Original filename including extension",
        )
        file_content_base64: str = Field(
            ..., description="Base64-encoded file content",
        )
        file_format: Optional[str] = Field(
            None, description="File format hint (xlsx, xls, csv, tsv); auto-detected if omitted",
        )
        template_id: Optional[str] = Field(
            None, description="Mapping template ID to apply during normalization",
        )
        tenant_id: str = Field(
            default="default", description="Tenant identifier",
        )

    class BatchFileItem(BaseModel):
        """Single item within a batch upload request."""
        file_name: str = Field(..., description="Original filename")
        file_content_base64: str = Field(
            ..., description="Base64-encoded content",
        )
        file_format: Optional[str] = Field(
            None, description="File format hint",
        )
        template_id: Optional[str] = Field(
            None, description="Mapping template ID",
        )

    class BatchUploadBody(BaseModel):
        """Request body for batch file upload."""
        files: List[BatchFileItem] = Field(
            ..., description="List of files to upload and normalize",
        )
        parallel: bool = Field(
            default=True, description="Whether to process files in parallel",
        )
        tenant_id: str = Field(
            default="default", description="Tenant identifier",
        )

    class ReprocessBody(BaseModel):
        """Request body for reprocessing a file."""
        template_id: Optional[str] = Field(
            None, description="New mapping template ID to apply",
        )
        config: Optional[Dict[str, Any]] = Field(
            None, description="Override configuration for reprocessing",
        )

    class NormalizeBody(BaseModel):
        """Request body for inline data normalization."""
        data: List[Dict[str, Any]] = Field(
            ..., description="List of row dictionaries to normalize",
        )
        column_mappings: Dict[str, str] = Field(
            ..., description="Mapping of source column names to canonical names",
        )
        tenant_id: str = Field(
            default="default", description="Tenant identifier",
        )

    class MapColumnsBody(BaseModel):
        """Request body for column mapping."""
        headers: List[str] = Field(
            ..., description="Source column headers to map",
        )
        strategy: Optional[str] = Field(
            None, description="Mapping strategy (exact, synonym, fuzzy, pattern, manual)",
        )
        template_id: Optional[str] = Field(
            None, description="Template ID for pre-defined mappings",
        )

    class DetectTypesBody(BaseModel):
        """Request body for column data type detection."""
        values: List[List[Any]] = Field(
            ..., description="Column-oriented sample values for type detection",
        )
        headers: Optional[List[str]] = Field(
            None, description="Column headers for labeling results",
        )

    class ValidateBody(BaseModel):
        """Request body for data validation."""
        data: List[Dict[str, Any]] = Field(
            ..., description="Row dictionaries to validate",
        )
        schema_name: str = Field(
            ..., description="Schema name to validate against",
        )

    class TransformBody(BaseModel):
        """Request body for applying transform operations."""
        file_id: str = Field(
            ..., description="File ID whose data will be transformed",
        )
        operations: List[Dict[str, Any]] = Field(
            ..., description="Ordered list of transform operations to apply",
        )

    class CreateTemplateBody(BaseModel):
        """Request body for creating a mapping template."""
        template_name: str = Field(
            ..., description="Human-readable template name",
        )
        description: str = Field(
            default="", description="Template description",
        )
        source_type: str = Field(
            ..., description="Source file type (xlsx, csv, generic)",
        )
        column_mappings: Dict[str, str] = Field(
            ..., description="Mapping of source column names to canonical names",
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/excel-normalizer",
        tags=["excel-normalizer"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Request) -> Any:
    """Extract ExcelNormalizerService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        ExcelNormalizerService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    service = getattr(request.app.state, "excel_normalizer_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Excel normalizer service not configured",
        )
    return service


if FASTAPI_AVAILABLE:

    # ------------------------------------------------------------------
    # 1. Upload single file
    # ------------------------------------------------------------------
    @router.post("/v1/files/upload")
    async def upload_file(
        body: UploadFileBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Upload and normalize a single Excel/CSV file."""
        service = _get_service(request)
        try:
            record = service.upload_file(
                file_name=body.file_name,
                file_content=body.file_content_base64,
                file_format=body.file_format or "auto",
                template_id=body.template_id,
                tenant_id=body.tenant_id,
            )
            return record.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 2. Batch upload files
    # ------------------------------------------------------------------
    @router.post("/v1/files/batch")
    async def batch_upload(
        body: BatchUploadBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Batch upload and normalize multiple files."""
        service = _get_service(request)
        try:
            files = [f.model_dump() for f in body.files]
            batch_job = service.upload_batch(
                files=files,
                parallel=body.parallel,
                tenant_id=body.tenant_id,
            )
            return batch_job.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 3. List files
    # ------------------------------------------------------------------
    @router.get("/v1/files")
    async def list_files(
        tenant_id: Optional[str] = Query(None),
        file_format: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List uploaded files with optional filters."""
        service = _get_service(request)
        files = service.list_files(
            tenant_id=tenant_id or "default",
            limit=limit,
            offset=offset,
        )
        return {
            "files": [f.model_dump(mode="json") for f in files],
            "count": len(files),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 4. Get file details
    # ------------------------------------------------------------------
    @router.get("/v1/files/{file_id}")
    async def get_file(
        file_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get file details by ID."""
        service = _get_service(request)
        record = service.get_file(file_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"File {file_id} not found",
            )
        return record.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 5. Get file sheets
    # ------------------------------------------------------------------
    @router.get("/v1/files/{file_id}/sheets")
    async def get_file_sheets(
        file_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get sheet-level information for an uploaded file."""
        service = _get_service(request)
        record = service.get_file(file_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"File {file_id} not found",
            )
        return {
            "file_id": file_id,
            "sheets": record.sheets,
            "sheet_count": len(record.sheets),
        }

    # ------------------------------------------------------------------
    # 6. Preview file data
    # ------------------------------------------------------------------
    @router.get("/v1/files/{file_id}/preview")
    async def preview_file(
        file_id: str,
        sheet_index: int = Query(0, ge=0),
        max_rows: int = Query(20, ge=1, le=200),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """Preview rows from a specific sheet of an uploaded file."""
        service = _get_service(request)
        record = service.get_file(file_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"File {file_id} not found",
            )
        # Return first max_rows from the requested sheet
        rows = record.normalized_data[:max_rows] if record.normalized_data else []
        return {
            "file_id": file_id,
            "sheet_index": sheet_index,
            "rows": rows,
            "row_count": len(rows),
            "total_rows": record.row_count,
        }

    # ------------------------------------------------------------------
    # 7. Reprocess file
    # ------------------------------------------------------------------
    @router.post("/v1/files/{file_id}/reprocess")
    async def reprocess_file(
        file_id: str,
        body: ReprocessBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Reprocess an already-uploaded file with new settings."""
        service = _get_service(request)
        try:
            record = service.get_file(file_id)
            if record is None:
                raise ValueError(f"File {file_id} not found")
            # Re-run normalization pipeline with updated template
            result = service.upload_file(
                file_name=record.file_name,
                file_content=record.raw_content_base64,
                file_format=record.file_format,
                template_id=body.template_id,
                tenant_id=record.tenant_id,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # ------------------------------------------------------------------
    # 8. Normalize inline data
    # ------------------------------------------------------------------
    @router.post("/v1/normalize")
    async def normalize_data(
        body: NormalizeBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Normalize inline data with explicit column mappings."""
        service = _get_service(request)
        try:
            result = service.normalize_data(
                data=body.data,
                column_mappings=body.column_mappings,
                tenant_id=body.tenant_id,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 9. Map columns to canonical schema
    # ------------------------------------------------------------------
    @router.post("/v1/columns/map")
    async def map_columns(
        body: MapColumnsBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Map source column headers to GreenLang canonical schema."""
        service = _get_service(request)
        try:
            result = service.map_columns(
                headers=body.headers,
                strategy=body.strategy or "fuzzy",
                template_id=body.template_id,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 10. List canonical fields
    # ------------------------------------------------------------------
    @router.get("/v1/columns/canonical")
    async def get_canonical_fields(
        category: Optional[str] = Query(None),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List GreenLang canonical field definitions."""
        service = _get_service(request)
        result = service.get_canonical_fields(category=category)
        return result.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 11. Detect column data types
    # ------------------------------------------------------------------
    @router.post("/v1/columns/detect-types")
    async def detect_types(
        body: DetectTypesBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Detect data types from column sample values."""
        service = _get_service(request)
        try:
            result = service.detect_types(
                values=body.values,
                headers=body.headers,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 12. Validate data against schema
    # ------------------------------------------------------------------
    @router.post("/v1/validate")
    async def validate_data(
        body: ValidateBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Validate row data against a named schema."""
        service = _get_service(request)
        try:
            result = service.validate_data(
                data=body.data,
                schema_name=body.schema_name,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 13. Apply transform pipeline
    # ------------------------------------------------------------------
    @router.post("/v1/transform")
    async def apply_transforms(
        body: TransformBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Apply a sequence of transform operations to file data."""
        service = _get_service(request)
        try:
            result = service.apply_transforms(
                data=None,
                operations=body.operations,
                file_id=body.file_id,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 14. Get quality score
    # ------------------------------------------------------------------
    @router.get("/v1/quality/{file_id}")
    async def get_quality(
        file_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get data quality score for an uploaded file."""
        service = _get_service(request)
        record = service.get_file(file_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"File {file_id} not found",
            )
        return {
            "file_id": file_id,
            "quality_score": record.quality_score,
            "completeness": record.completeness_score,
            "accuracy": record.accuracy_score,
            "consistency": record.consistency_score,
            "provenance_hash": record.provenance_hash,
        }

    # ------------------------------------------------------------------
    # 15. Create mapping template
    # ------------------------------------------------------------------
    @router.post("/v1/templates")
    async def create_template(
        body: CreateTemplateBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new column mapping template."""
        service = _get_service(request)
        try:
            template = service.create_template(
                name=body.template_name,
                description=body.description,
                source_type=body.source_type,
                mappings=body.column_mappings,
            )
            return template.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 16. List templates
    # ------------------------------------------------------------------
    @router.get("/v1/templates")
    async def list_templates(
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List all mapping templates."""
        service = _get_service(request)
        templates = service.list_templates()
        paginated = templates[offset:offset + limit]
        return {
            "templates": [t.model_dump(mode="json") for t in paginated],
            "count": len(paginated),
            "total": len(templates),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 17. Get template details
    # ------------------------------------------------------------------
    @router.get("/v1/templates/{template_id}")
    async def get_template(
        template_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get mapping template details by ID."""
        service = _get_service(request)
        template = service.get_template(template_id)
        if template is None:
            raise HTTPException(
                status_code=404,
                detail=f"Template {template_id} not found",
            )
        return template.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 18. List normalization jobs
    # ------------------------------------------------------------------
    @router.get("/v1/jobs")
    async def list_jobs(
        status: Optional[str] = Query(None),
        tenant_id: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List normalization jobs with optional filters."""
        service = _get_service(request)
        jobs = service.list_jobs(
            status=status,
            tenant_id=tenant_id,
            limit=limit,
            offset=offset,
        )
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
        """Get Excel normalizer service statistics."""
        service = _get_service(request)
        stats = service.get_statistics()
        return stats.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 20. Health check
    # ------------------------------------------------------------------
    @router.get("/health")
    async def health() -> Dict[str, str]:
        """Excel normalizer service health check endpoint."""
        return {"status": "healthy", "service": "excel-normalizer"}


__all__ = [
    "router",
]
