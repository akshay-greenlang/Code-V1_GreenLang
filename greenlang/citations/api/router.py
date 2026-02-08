# -*- coding: utf-8 -*-
"""
Citations & Evidence REST API Router - AGENT-FOUND-005: Citations & Evidence

FastAPI router providing 20 endpoints for citation management,
evidence packaging, verification, export/import, and provenance.

All endpoints are mounted under ``/api/v1/citations``.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-005 Citations & Evidence
Status: Production Ready
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import  (no `from __future__ import annotations` here)
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[assignment, misc]
    logger.warning("FastAPI not available; citations router is None")


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class CreateCitationRequest(BaseModel):
        """Request body for creating a citation."""
        citation_type: str = Field(..., description="Citation type")
        source_authority: str = Field(..., description="Source authority")
        metadata: Dict[str, Any] = Field(..., description="Citation metadata")
        effective_date: str = Field(..., description="Effective date (YYYY-MM-DD)")
        user_id: str = Field("system", description="User creating the citation")
        change_reason: str = Field("Initial creation", description="Reason")
        citation_id: Optional[str] = Field(None, description="Optional pre-assigned ID")
        expiration_date: Optional[str] = Field(None, description="Expiration date")
        supersedes: Optional[str] = Field(None, description="ID this supersedes")
        regulatory_frameworks: Optional[List[str]] = Field(None, description="Frameworks")
        abstract: Optional[str] = Field(None, description="Abstract text")
        key_values: Optional[Dict[str, Any]] = Field(None, description="Key values")
        notes: Optional[str] = Field(None, description="Notes")

    class UpdateCitationRequest(BaseModel):
        """Request body for updating a citation."""
        user_id: str = Field("system", description="User making the change")
        reason: str = Field("Citation update", description="Reason for change")
        metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
        expiration_date: Optional[str] = Field(None, description="New expiration date")
        abstract: Optional[str] = Field(None, description="New abstract")
        key_values: Optional[Dict[str, Any]] = Field(None, description="New key values")
        notes: Optional[str] = Field(None, description="New notes")
        regulatory_frameworks: Optional[List[str]] = Field(None, description="New frameworks")

    class CreatePackageRequest(BaseModel):
        """Request body for creating an evidence package."""
        name: str = Field(..., description="Package name")
        description: str = Field("", description="Package description")
        user_id: str = Field("system", description="User creating the package")
        calculation_context: Optional[Dict[str, Any]] = Field(None, description="Context")
        calculation_result: Optional[Dict[str, Any]] = Field(None, description="Result")
        regulatory_frameworks: Optional[List[str]] = Field(None, description="Frameworks")
        compliance_notes: Optional[str] = Field(None, description="Compliance notes")

    class AddEvidenceItemRequest(BaseModel):
        """Request body for adding an evidence item to a package."""
        evidence_type: str = Field(..., description="Evidence type")
        description: str = Field(..., description="Evidence description")
        data: Dict[str, Any] = Field(default_factory=dict, description="Evidence data")
        citation_ids: Optional[List[str]] = Field(None, description="Citation IDs")
        source_system: Optional[str] = Field(None, description="Source system")
        source_agent: Optional[str] = Field(None, description="Source agent")

    class AddCitationToPackageRequest(BaseModel):
        """Request body for adding a citation to a package."""
        citation_id: str = Field(..., description="Citation ID to add")

    class VerifyBatchRequest(BaseModel):
        """Request body for batch verification."""
        citation_ids: List[str] = Field(..., description="Citation IDs to verify")
        user_id: str = Field("system", description="User performing verification")

    class ExportRequest(BaseModel):
        """Request body for export."""
        citation_ids: Optional[List[str]] = Field(None, description="IDs to export")
        format: str = Field("json", description="Export format: bibtex, json, csl")

    class ImportBibtexRequest(BaseModel):
        """Request body for BibTeX import."""
        content: str = Field(..., description="BibTeX content string")
        user_id: str = Field("system", description="User performing import")

    class ImportJsonRequest(BaseModel):
        """Request body for JSON import."""
        content: str = Field(..., description="JSON content string")
        user_id: str = Field("system", description="User performing import")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/citations",
        tags=["citations"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Request) -> Any:
    """Extract CitationsService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        CitationsService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    service = getattr(request.app.state, "citations_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Citations service not configured",
        )
    return service


if FASTAPI_AVAILABLE:

    # 1. Health check
    @router.get("/health")
    async def health() -> Dict[str, str]:
        """Citations & Evidence service health check endpoint."""
        return {"status": "healthy", "service": "citations-evidence"}

    # 2. Metrics summary
    @router.get("/metrics")
    async def metrics_endpoint(request: Request) -> Dict[str, Any]:
        """Get citations & evidence metrics summary."""
        service = _get_service(request)
        return service.get_metrics()

    # 3. Create citation
    @router.post("/")
    async def create_citation(
        body: CreateCitationRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new citation."""
        service = _get_service(request)
        try:
            citation = service.registry.create(
                citation_type=body.citation_type,
                source_authority=body.source_authority,
                metadata=body.metadata,
                effective_date=body.effective_date,
                user_id=body.user_id,
                change_reason=body.change_reason,
                citation_id=body.citation_id,
                expiration_date=body.expiration_date,
                supersedes=body.supersedes,
                regulatory_frameworks=body.regulatory_frameworks,
                abstract=body.abstract,
                key_values=body.key_values,
                notes=body.notes,
            )
            return citation.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 4. List citations
    @router.get("/")
    async def list_citations(
        citation_type: Optional[str] = Query(None),
        source_authority: Optional[str] = Query(None),
        verification_status: Optional[str] = Query(None),
        search: Optional[str] = Query(None),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List citations with optional filtering."""
        service = _get_service(request)
        citations = service.registry.list(
            citation_type=citation_type,
            source_authority=source_authority,
            verification_status=verification_status,
            search=search,
        )
        return {
            "citations": [c.model_dump(mode="json") for c in citations],
            "count": len(citations),
        }

    # 5. Get citation
    @router.get("/{citation_id}")
    async def get_citation(
        citation_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a specific citation by ID."""
        service = _get_service(request)
        citation = service.registry.get(citation_id)
        if citation is None:
            raise HTTPException(
                status_code=404,
                detail=f"Citation {citation_id} not found",
            )
        return citation.model_dump(mode="json")

    # 6. Update citation
    @router.put("/{citation_id}")
    async def update_citation(
        citation_id: str,
        body: UpdateCitationRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Update a citation."""
        service = _get_service(request)
        try:
            citation = service.registry.update(
                citation_id=citation_id,
                user_id=body.user_id,
                reason=body.reason,
                metadata=body.metadata,
                expiration_date=body.expiration_date,
                abstract=body.abstract,
                key_values=body.key_values,
                notes=body.notes,
                regulatory_frameworks=body.regulatory_frameworks,
            )
            return citation.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 7. Delete citation
    @router.delete("/{citation_id}")
    async def delete_citation(
        citation_id: str,
        user_id: str = Query("system"),
        reason: str = Query("Deletion"),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """Delete a citation."""
        service = _get_service(request)
        try:
            deleted = service.registry.delete(
                citation_id=citation_id,
                user_id=user_id,
                reason=reason,
            )
            if not deleted:
                raise HTTPException(
                    status_code=404,
                    detail=f"Citation {citation_id} not found",
                )
            return {"citation_id": citation_id, "deleted": True}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 8. Get citation versions
    @router.get("/{citation_id}/versions")
    async def get_versions(
        citation_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get version history for a citation."""
        service = _get_service(request)
        try:
            versions = service.registry.get_versions(citation_id)
            return {
                "citation_id": citation_id,
                "versions": [v.model_dump(mode="json") for v in versions],
                "count": len(versions),
            }
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # 9. Verify citation
    @router.post("/{citation_id}/verify")
    async def verify_citation(
        citation_id: str,
        user_id: str = Query("system"),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """Verify a single citation."""
        service = _get_service(request)
        try:
            record = service.verification_engine.verify_citation(
                citation_id, user_id,
            )
            return record.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # 10. Batch verify citations
    @router.post("/verify/batch")
    async def verify_batch(
        body: VerifyBatchRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Verify multiple citations."""
        service = _get_service(request)
        results = service.verification_engine.verify_batch(
            body.citation_ids, body.user_id,
        )
        return {
            "results": {
                cid: rec.model_dump(mode="json")
                for cid, rec in results.items()
            },
            "count": len(results),
        }

    # 11. Get verification history
    @router.get("/{citation_id}/verification-history")
    async def get_verification_history(
        citation_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get verification history for a citation."""
        service = _get_service(request)
        records = service.verification_engine.get_verification_history(
            citation_id,
        )
        return {
            "citation_id": citation_id,
            "records": [r.model_dump(mode="json") for r in records],
            "count": len(records),
        }

    # 12. Create evidence package
    @router.post("/packages")
    async def create_package(
        body: CreatePackageRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new evidence package."""
        service = _get_service(request)
        try:
            package = service.evidence_manager.create_package(
                name=body.name,
                description=body.description,
                user_id=body.user_id,
                calculation_context=body.calculation_context,
                calculation_result=body.calculation_result,
                regulatory_frameworks=body.regulatory_frameworks,
                compliance_notes=body.compliance_notes,
            )
            return package.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 13. List evidence packages
    @router.get("/packages/list")
    async def list_packages(
        created_by: Optional[str] = Query(None),
        finalized_only: bool = Query(False),
        search: Optional[str] = Query(None),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List evidence packages with optional filtering."""
        service = _get_service(request)
        packages = service.evidence_manager.list_packages(
            created_by=created_by,
            finalized_only=finalized_only,
            search=search,
        )
        return {
            "packages": [p.model_dump(mode="json") for p in packages],
            "count": len(packages),
        }

    # 14. Get evidence package
    @router.get("/packages/{package_id}")
    async def get_package(
        package_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a specific evidence package."""
        service = _get_service(request)
        package = service.evidence_manager.get_package(package_id)
        if package is None:
            raise HTTPException(
                status_code=404,
                detail=f"Package {package_id} not found",
            )
        return package.model_dump(mode="json")

    # 15. Add evidence item to package
    @router.post("/packages/{package_id}/items")
    async def add_evidence_item(
        package_id: str,
        body: AddEvidenceItemRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Add an evidence item to a package."""
        from greenlang.citations.models import EvidenceItem, EvidenceType

        service = _get_service(request)
        try:
            item = EvidenceItem(
                evidence_type=EvidenceType(body.evidence_type),
                description=body.description,
                data=body.data,
                citation_ids=body.citation_ids or [],
                source_system=body.source_system,
                source_agent=body.source_agent,
            )
            package = service.evidence_manager.add_item(package_id, item)
            return package.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 16. Add citation to package
    @router.post("/packages/{package_id}/citations")
    async def add_citation_to_package(
        package_id: str,
        body: AddCitationToPackageRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Add a citation reference to a package."""
        service = _get_service(request)
        try:
            package = service.evidence_manager.add_citation(
                package_id, body.citation_id,
            )
            return package.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 17. Finalize package
    @router.post("/packages/{package_id}/finalize")
    async def finalize_package(
        package_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Finalize an evidence package."""
        service = _get_service(request)
        try:
            package_hash = service.evidence_manager.finalize_package(
                package_id,
            )
            return {
                "package_id": package_id,
                "package_hash": package_hash,
                "finalized": True,
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 18. Delete package
    @router.delete("/packages/{package_id}")
    async def delete_package(
        package_id: str,
        user_id: str = Query("system"),
        reason: str = Query("Deletion"),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """Delete an evidence package."""
        service = _get_service(request)
        try:
            deleted = service.evidence_manager.delete_package(
                package_id, user_id, reason,
            )
            if not deleted:
                raise HTTPException(
                    status_code=404,
                    detail=f"Package {package_id} not found",
                )
            return {"package_id": package_id, "deleted": True}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 19. Export citations
    @router.post("/export")
    async def export_citations(
        body: ExportRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Export citations to specified format."""
        service = _get_service(request)
        fmt = body.format.lower()
        try:
            if fmt == "bibtex":
                content = service.export_import.export_bibtex(body.citation_ids)
            elif fmt == "csl":
                content = service.export_import.export_csl(body.citation_ids)
            else:
                content = service.export_import.export_json(body.citation_ids)

            return {
                "format": fmt,
                "content": content,
                "count": len(body.citation_ids) if body.citation_ids else service.registry.count,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 20. Import citations
    @router.post("/import")
    async def import_citations(
        body: ImportJsonRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Import citations from JSON format."""
        service = _get_service(request)
        try:
            citations = service.export_import.import_json(
                body.content, body.user_id,
            )
            return {
                "imported_count": len(citations),
                "citation_ids": [c.citation_id for c in citations],
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))


__all__ = [
    "router",
]
