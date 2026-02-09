# -*- coding: utf-8 -*-
"""
Spend Data Categorizer REST API Router - AGENT-DATA-009

FastAPI router providing 20 endpoints for spend record ingestion,
taxonomy classification, Scope 3 mapping, emission calculation,
rule management, analytics, reporting, statistics, and health monitoring.

All endpoints are mounted under ``/api/v1/spend-categorizer``.

Endpoints:
    1.  POST   /v1/ingest                              - Ingest spend records
    2.  POST   /v1/ingest/file                          - Ingest from file
    3.  GET    /v1/records                              - List spend records
    4.  GET    /v1/records/{record_id}                  - Get single record
    5.  POST   /v1/classify                             - Classify spend record
    6.  POST   /v1/classify/batch                       - Batch classification
    7.  POST   /v1/map-scope3                           - Map to Scope 3
    8.  POST   /v1/map-scope3/batch                     - Batch Scope 3 mapping
    9.  POST   /v1/calculate-emissions                  - Calculate emissions
    10. POST   /v1/calculate-emissions/batch             - Batch emission calculation
    11. GET    /v1/emission-factors                      - List emission factors
    12. GET    /v1/emission-factors/{taxonomy_code}      - Get factor
    13. POST   /v1/rules                                - Create rule
    14. GET    /v1/rules                                - List rules
    15. PUT    /v1/rules/{rule_id}                      - Update rule
    16. DELETE /v1/rules/{rule_id}                      - Delete rule
    17. GET    /v1/analytics                            - Get analytics
    18. GET    /v1/analytics/hotspots                   - Get hotspots
    19. POST   /v1/reports                              - Generate report
    20. GET    /health                                  - Health check

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-009 Spend Data Categorizer (GL-DATA-SUP-002)
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
    logger.warning(
        "FastAPI not available; spend categorizer router is None"
    )


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class IngestRecordsBody(BaseModel):
        """Request body for ingesting spend records."""
        records: List[Dict[str, Any]] = Field(
            ..., description="List of spend record dicts with vendor_name, amount, description, etc.",
        )
        source: str = Field(
            default="manual",
            description="Data source identifier (csv, excel, api, erp, manual)",
        )

    class IngestFileBody(BaseModel):
        """Request body for ingesting from a file path."""
        file_path: str = Field(
            ..., description="Path to CSV or Excel file",
        )
        file_type: str = Field(
            default="csv",
            description="File type (csv, excel)",
        )
        sheet_name: Optional[str] = Field(
            None, description="Excel sheet name (uses first sheet if omitted)",
        )
        source: str = Field(
            default="file",
            description="Data source label",
        )

    class ClassifyRecordBody(BaseModel):
        """Request body for classifying a spend record."""
        record_id: str = Field(
            ..., description="Spend record identifier to classify",
        )
        taxonomy_system: Optional[str] = Field(
            None,
            description="Taxonomy system override (unspsc, naics, nace, custom)",
        )

    class ClassifyBatchBody(BaseModel):
        """Request body for batch classification."""
        record_ids: List[str] = Field(
            ..., description="List of record identifiers to classify",
        )
        taxonomy_system: Optional[str] = Field(
            None,
            description="Taxonomy system override",
        )

    class MapScope3Body(BaseModel):
        """Request body for mapping a record to Scope 3."""
        record_id: str = Field(
            ..., description="Spend record identifier to map",
        )

    class MapScope3BatchBody(BaseModel):
        """Request body for batch Scope 3 mapping."""
        record_ids: List[str] = Field(
            ..., description="List of record identifiers to map",
        )

    class CalculateEmissionsBody(BaseModel):
        """Request body for calculating emissions."""
        record_id: str = Field(
            ..., description="Spend record identifier",
        )
        factor_source: Optional[str] = Field(
            None,
            description="Emission factor source override (eeio, exiobase, defra)",
        )

    class CalculateEmissionsBatchBody(BaseModel):
        """Request body for batch emission calculation."""
        record_ids: List[str] = Field(
            ..., description="List of record identifiers",
        )
        factor_source: Optional[str] = Field(
            None,
            description="Emission factor source override",
        )

    class CreateRuleBody(BaseModel):
        """Request body for creating a classification rule."""
        name: str = Field(
            ..., description="Rule display name",
        )
        taxonomy_code: str = Field(
            ..., description="Target taxonomy code to assign",
        )
        conditions: Dict[str, Any] = Field(
            ..., description="Rule conditions (vendor_pattern, keywords, gl_codes, cost_centers)",
        )
        taxonomy_system: str = Field(
            default="unspsc",
            description="Taxonomy system (unspsc, naics, nace)",
        )
        scope3_category: Optional[int] = Field(
            None, description="Optional Scope 3 category override (1-15)",
        )
        description: str = Field(
            default="", description="Rule description",
        )
        priority: int = Field(
            default=100, description="Rule priority (lower = higher priority)",
        )

    class UpdateRuleBody(BaseModel):
        """Request body for updating a classification rule."""
        name: Optional[str] = Field(
            None, description="New rule name",
        )
        description: Optional[str] = Field(
            None, description="New description",
        )
        taxonomy_code: Optional[str] = Field(
            None, description="New taxonomy code",
        )
        conditions: Optional[Dict[str, Any]] = Field(
            None, description="New conditions",
        )
        priority: Optional[int] = Field(
            None, description="New priority",
        )
        is_active: Optional[bool] = Field(
            None, description="New active status",
        )

    class GenerateReportBody(BaseModel):
        """Request body for generating a report."""
        report_type: str = Field(
            default="summary",
            description="Report type (summary, detailed, emissions, scope3)",
        )
        format: str = Field(
            default="json",
            description="Report format (json, csv, excel, pdf)",
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/spend-categorizer",
        tags=["Spend Categorizer"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Request) -> Any:
    """Extract SpendCategorizerService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        SpendCategorizerService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    service = getattr(
        request.app.state, "spend_categorizer_service", None,
    )
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Spend categorizer service not configured",
        )
    return service


if FASTAPI_AVAILABLE:

    # ------------------------------------------------------------------
    # 1. Ingest spend records
    # ------------------------------------------------------------------
    @router.post("/v1/ingest")
    async def ingest_records(
        body: IngestRecordsBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Ingest a batch of spend records."""
        service = _get_service(request)
        try:
            records = service.ingest_records(
                records=body.records,
                source=body.source,
            )
            return {
                "records": [r.model_dump(mode="json") for r in records],
                "count": len(records),
                "source": body.source,
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 2. Ingest from file
    # ------------------------------------------------------------------
    @router.post("/v1/ingest/file")
    async def ingest_file(
        body: IngestFileBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Ingest spend records from a CSV or Excel file."""
        service = _get_service(request)
        try:
            if body.file_type == "excel":
                records = service.ingest_excel(
                    file_path=body.file_path,
                    sheet_name=body.sheet_name,
                    source=body.source,
                )
            else:
                records = service.ingest_csv(
                    file_path=body.file_path,
                    source=body.source,
                )
            return {
                "records": [r.model_dump(mode="json") for r in records],
                "count": len(records),
                "file_type": body.file_type,
                "source": body.source,
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 3. List spend records
    # ------------------------------------------------------------------
    @router.get("/v1/records")
    async def list_records(
        source: Optional[str] = Query(None),
        status: Optional[str] = Query(None),
        vendor_name: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List spend records with optional filters."""
        service = _get_service(request)
        records = service.list_records(
            source=source,
            status=status,
            vendor_name=vendor_name,
            limit=limit,
            offset=offset,
        )
        return {
            "records": [r.model_dump(mode="json") for r in records],
            "count": len(records),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 4. Get single record
    # ------------------------------------------------------------------
    @router.get("/v1/records/{record_id}")
    async def get_record(
        record_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a spend record by ID."""
        service = _get_service(request)
        record = service.get_record(record_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Record {record_id} not found",
            )
        return record.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 5. Classify spend record
    # ------------------------------------------------------------------
    @router.post("/v1/classify")
    async def classify_record(
        body: ClassifyRecordBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Classify a spend record into a taxonomy category."""
        service = _get_service(request)
        try:
            result = service.classify_record(
                record_id=body.record_id,
                taxonomy_system=body.taxonomy_system,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 6. Batch classification
    # ------------------------------------------------------------------
    @router.post("/v1/classify/batch")
    async def classify_batch(
        body: ClassifyBatchBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Classify multiple spend records in batch."""
        service = _get_service(request)
        results = service.classify_batch(
            record_ids=body.record_ids,
            taxonomy_system=body.taxonomy_system,
        )
        return {
            "classifications": [r.model_dump(mode="json") for r in results],
            "count": len(results),
            "requested": len(body.record_ids),
        }

    # ------------------------------------------------------------------
    # 7. Map to Scope 3
    # ------------------------------------------------------------------
    @router.post("/v1/map-scope3")
    async def map_scope3(
        body: MapScope3Body,
        request: Request,
    ) -> Dict[str, Any]:
        """Map a spend record to a GHG Protocol Scope 3 category."""
        service = _get_service(request)
        try:
            result = service.map_scope3(
                record_id=body.record_id,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 8. Batch Scope 3 mapping
    # ------------------------------------------------------------------
    @router.post("/v1/map-scope3/batch")
    async def map_scope3_batch(
        body: MapScope3BatchBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Map multiple records to Scope 3 categories in batch."""
        service = _get_service(request)
        results = service.map_scope3_batch(
            record_ids=body.record_ids,
        )
        return {
            "assignments": [r.model_dump(mode="json") for r in results],
            "count": len(results),
            "requested": len(body.record_ids),
        }

    # ------------------------------------------------------------------
    # 9. Calculate emissions
    # ------------------------------------------------------------------
    @router.post("/v1/calculate-emissions")
    async def calculate_emissions(
        body: CalculateEmissionsBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Calculate emissions for a spend record."""
        service = _get_service(request)
        try:
            result = service.calculate_emissions(
                record_id=body.record_id,
                factor_source=body.factor_source,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 10. Batch emission calculation
    # ------------------------------------------------------------------
    @router.post("/v1/calculate-emissions/batch")
    async def calculate_emissions_batch(
        body: CalculateEmissionsBatchBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Calculate emissions for multiple records in batch."""
        service = _get_service(request)
        results = service.calculate_emissions_batch(
            record_ids=body.record_ids,
            factor_source=body.factor_source,
        )
        return {
            "calculations": [r.model_dump(mode="json") for r in results],
            "count": len(results),
            "requested": len(body.record_ids),
        }

    # ------------------------------------------------------------------
    # 11. List emission factors
    # ------------------------------------------------------------------
    @router.get("/v1/emission-factors")
    async def list_emission_factors(
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List available emission factors."""
        service = _get_service(request)
        factors = service.list_emission_factors(
            limit=limit,
            offset=offset,
        )
        return {
            "emission_factors": factors,
            "count": len(factors),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 12. Get emission factor by taxonomy code
    # ------------------------------------------------------------------
    @router.get("/v1/emission-factors/{taxonomy_code}")
    async def get_emission_factor(
        taxonomy_code: str,
        source: str = Query("eeio"),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """Get the emission factor for a specific taxonomy code."""
        service = _get_service(request)
        factor = service.get_emission_factor(
            taxonomy_code=taxonomy_code,
            source=source,
        )
        return {
            "taxonomy_code": taxonomy_code,
            "source": source,
            **factor,
        }

    # ------------------------------------------------------------------
    # 13. Create rule
    # ------------------------------------------------------------------
    @router.post("/v1/rules")
    async def create_rule(
        body: CreateRuleBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new classification rule."""
        service = _get_service(request)
        try:
            rule = service.create_rule(
                name=body.name,
                taxonomy_code=body.taxonomy_code,
                conditions=body.conditions,
                taxonomy_system=body.taxonomy_system,
                scope3_category=body.scope3_category,
                description=body.description,
                priority=body.priority,
            )
            return rule.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 14. List rules
    # ------------------------------------------------------------------
    @router.get("/v1/rules")
    async def list_rules(
        is_active: Optional[bool] = Query(None),
        taxonomy_system: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List classification rules with optional filters."""
        service = _get_service(request)
        rules = service.list_rules(
            is_active=is_active,
            taxonomy_system=taxonomy_system,
            limit=limit,
            offset=offset,
        )
        return {
            "rules": [r.model_dump(mode="json") for r in rules],
            "count": len(rules),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 15. Update rule
    # ------------------------------------------------------------------
    @router.put("/v1/rules/{rule_id}")
    async def update_rule(
        rule_id: str,
        body: UpdateRuleBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Update an existing classification rule."""
        service = _get_service(request)
        try:
            rule = service.update_rule(
                rule_id=rule_id,
                name=body.name,
                description=body.description,
                taxonomy_code=body.taxonomy_code,
                conditions=body.conditions,
                priority=body.priority,
                is_active=body.is_active,
            )
            return rule.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # ------------------------------------------------------------------
    # 16. Delete rule
    # ------------------------------------------------------------------
    @router.delete("/v1/rules/{rule_id}")
    async def delete_rule(
        rule_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Delete a classification rule."""
        service = _get_service(request)
        deleted = service.delete_rule(rule_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Rule {rule_id} not found",
            )
        return {
            "deleted": True,
            "rule_id": rule_id,
        }

    # ------------------------------------------------------------------
    # 17. Get analytics
    # ------------------------------------------------------------------
    @router.get("/v1/analytics")
    async def get_analytics(
        request: Request,
    ) -> Dict[str, Any]:
        """Get spend categorization analytics summary."""
        service = _get_service(request)
        analytics = service.get_analytics()
        return analytics.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 18. Get hotspots
    # ------------------------------------------------------------------
    @router.get("/v1/analytics/hotspots")
    async def get_hotspots(
        top_n: int = Query(10, ge=1, le=100),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """Get emission hotspots (highest emitting categories)."""
        service = _get_service(request)
        hotspots = service.get_hotspots(top_n=top_n)
        return {
            "hotspots": hotspots,
            "count": len(hotspots),
        }

    # ------------------------------------------------------------------
    # 19. Generate report
    # ------------------------------------------------------------------
    @router.post("/v1/reports")
    async def generate_report(
        body: GenerateReportBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Generate a spend categorization report."""
        service = _get_service(request)
        report = service.generate_report(
            report_type=body.report_type,
            report_format=body.format,
        )
        return report.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 20. Health check
    # ------------------------------------------------------------------
    @router.get("/health")
    async def health(
        request: Request,
    ) -> Dict[str, Any]:
        """Spend categorizer service health check endpoint."""
        service = _get_service(request)
        return service.health_check()


__all__ = [
    "router",
]
