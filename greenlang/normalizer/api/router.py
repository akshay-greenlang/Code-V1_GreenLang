# -*- coding: utf-8 -*-
"""
Normalizer REST API Router - AGENT-FOUND-003: Unit & Reference Normalizer

FastAPI router providing 15 endpoints for unit conversion, entity
resolution, dimensional analysis, GWP lookup, and provenance.

All endpoints are mounted under ``/api/v1/normalizer``.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-003 Unit & Reference Normalizer
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[assignment, misc]
    logger.warning("FastAPI not available; normalizer router is None")


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class ConvertRequest(BaseModel):
        """Request body for a single unit conversion."""
        value: float = Field(..., description="Numeric value to convert")
        from_unit: str = Field(..., description="Source unit")
        to_unit: str = Field(..., description="Target unit")
        precision: Optional[int] = Field(None, ge=0, le=30, description="Decimal precision")

    class BatchConvertRequest(BaseModel):
        """Request body for batch unit conversion."""
        items: List[Dict[str, Any]] = Field(
            ..., description="List of {value, from_unit, to_unit} dicts",
        )

    class GHGConvertRequest(BaseModel):
        """Request body for GHG conversion."""
        value: float = Field(..., description="Mass value")
        from_gas: str = Field(..., description="Source gas (CO2, CH4, N2O, etc.)")
        to_gas: str = Field("CO2e", description="Target gas (default CO2e)")
        gwp_version: Optional[str] = Field(None, description="AR5 or AR6")
        gwp_timeframe: Optional[int] = Field(None, description="20 or 100 years")

    class ResolveEntityRequest(BaseModel):
        """Request body for entity resolution."""
        name: str = Field(..., description="Entity name to resolve")

    class BatchResolveRequest(BaseModel):
        """Request body for batch entity resolution."""
        items: List[str] = Field(..., description="Entity names to resolve")
        entity_type: str = Field(..., description="Entity type (fuel/material/process)")

    class SearchVocabularyRequest(BaseModel):
        """Request body for vocabulary search."""
        query: str = Field(..., description="Search query")
        entity_type: str = Field(..., description="Entity type (fuel/material/process)")
        limit: int = Field(10, ge=1, le=100, description="Max results")

    class CompatibilityCheckRequest(BaseModel):
        """Request body for compatibility check."""
        from_unit: str = Field(..., description="Source unit")
        to_unit: str = Field(..., description="Target unit")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/normalizer",
        tags=["normalizer"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Request) -> Any:
    """Extract NormalizerService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        NormalizerService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    service = getattr(request.app.state, "normalizer_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Normalizer service not configured",
        )
    return service


if FASTAPI_AVAILABLE:

    # 1. Health check
    @router.get("/health")
    async def health() -> Dict[str, str]:
        """Normalizer health check endpoint."""
        return {"status": "healthy", "service": "normalizer"}

    # 2. Convert single unit
    @router.post("/convert")
    async def convert(body: ConvertRequest, request: Request) -> Dict[str, Any]:
        """Convert a value from one unit to another."""
        service = _get_service(request)
        try:
            result = service.convert(
                body.value, body.from_unit, body.to_unit, body.precision,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 3. Batch convert
    @router.post("/convert/batch")
    async def batch_convert(body: BatchConvertRequest, request: Request) -> Dict[str, Any]:
        """Convert a batch of values."""
        service = _get_service(request)
        try:
            result = service.batch_convert(body.items)
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 4. GHG conversion
    @router.post("/convert/ghg")
    async def convert_ghg(body: GHGConvertRequest, request: Request) -> Dict[str, Any]:
        """Convert GHG emissions using GWP factors."""
        service = _get_service(request)
        try:
            result = service.get_converter().convert_ghg(
                body.value, body.from_gas, body.to_gas,
                body.gwp_version, body.gwp_timeframe,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 5. Get conversion factor
    @router.get("/factor")
    async def get_factor(
        from_unit: str = Query(...), to_unit: str = Query(...),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """Get the conversion factor between two units."""
        service = _get_service(request)
        try:
            factor = service.get_converter().get_conversion_factor(from_unit, to_unit)
            return {"from_unit": from_unit, "to_unit": to_unit, "factor": str(factor)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 6. Resolve fuel
    @router.post("/resolve/fuel")
    async def resolve_fuel(body: ResolveEntityRequest, request: Request) -> Dict[str, Any]:
        """Resolve a fuel name to canonical form."""
        service = _get_service(request)
        result = service.resolve_fuel(body.name)
        return result.model_dump(mode="json")

    # 7. Resolve material
    @router.post("/resolve/material")
    async def resolve_material(body: ResolveEntityRequest, request: Request) -> Dict[str, Any]:
        """Resolve a material name to canonical form."""
        service = _get_service(request)
        result = service.resolve_material(body.name)
        return result.model_dump(mode="json")

    # 8. Resolve process
    @router.post("/resolve/process")
    async def resolve_process(body: ResolveEntityRequest, request: Request) -> Dict[str, Any]:
        """Resolve a process name to canonical form."""
        service = _get_service(request)
        result = service.resolve_process(body.name)
        return result.model_dump(mode="json")

    # 9. Batch resolve
    @router.post("/resolve/batch")
    async def batch_resolve(body: BatchResolveRequest, request: Request) -> Dict[str, Any]:
        """Resolve a batch of entity names."""
        service = _get_service(request)
        try:
            result = service.get_resolver().batch_resolve(body.items, body.entity_type)
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 10. Search vocabulary
    @router.post("/vocabulary/search")
    async def search_vocabulary(body: SearchVocabularyRequest, request: Request) -> Dict[str, Any]:
        """Search entity vocabulary."""
        service = _get_service(request)
        try:
            results = service.get_resolver().search_vocabulary(
                body.query, body.entity_type, body.limit,
            )
            return {"results": [r.model_dump(mode="json") for r in results]}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 11. Check compatibility
    @router.post("/dimensions/check")
    async def check_compatibility(body: CompatibilityCheckRequest, request: Request) -> Dict[str, Any]:
        """Check if two units are dimensionally compatible."""
        service = _get_service(request)
        from greenlang.normalizer.dimensional import DimensionalAnalyzer
        analyzer = DimensionalAnalyzer()
        compatible = analyzer.check_compatibility(body.from_unit, body.to_unit)
        from_dim = analyzer.get_dimension(body.from_unit)
        to_dim = analyzer.get_dimension(body.to_unit)
        return {
            "from_unit": body.from_unit,
            "to_unit": body.to_unit,
            "compatible": compatible,
            "from_dimension": from_dim.value if from_dim else None,
            "to_dimension": to_dim.value if to_dim else None,
        }

    # 12. List dimensions
    @router.get("/dimensions")
    async def list_dimensions(request: Request) -> Dict[str, Any]:
        """List all supported dimensions."""
        service = _get_service(request)
        dims = service.get_converter().supported_dimensions()
        return {"dimensions": [d.model_dump(mode="json") for d in dims]}

    # 13. List units (optionally filtered)
    @router.get("/units")
    async def list_units(
        dimension: Optional[str] = Query(None),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List supported units, optionally filtered by dimension."""
        service = _get_service(request)
        from greenlang.normalizer.models import UnitDimension
        dim_filter = None
        if dimension:
            try:
                dim_filter = UnitDimension(dimension.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Unknown dimension: {dimension}",
                )
        units = service.get_converter().supported_units(dim_filter)
        return {"units": [u.model_dump(mode="json") for u in units]}

    # 14. GWP table lookup
    @router.get("/gwp")
    async def get_gwp_table(
        version: str = Query("AR6"),
        timeframe: int = Query(100),
    ) -> Dict[str, Any]:
        """Get GWP values for a specific IPCC version and timeframe."""
        from greenlang.normalizer.converter import GWP_TABLES
        table_key = f"{version.upper()}_{timeframe}"
        table = GWP_TABLES.get(table_key)
        if table is None:
            raise HTTPException(
                status_code=400,
                detail=f"No GWP table for {table_key}. Available: {list(GWP_TABLES.keys())}",
            )
        return {
            "version": version.upper(),
            "timeframe": timeframe,
            "values": {k: str(v) for k, v in table.items()},
        }

    # 15. Normalize unit name
    @router.get("/normalize")
    async def normalize_unit(
        unit: str = Query(...),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """Normalize a unit name to canonical form."""
        service = _get_service(request)
        normalized = service.get_converter().normalize_unit_name(unit)
        from greenlang.normalizer.dimensional import DimensionalAnalyzer
        analyzer = DimensionalAnalyzer()
        dimension = analyzer.get_dimension(unit)
        return {
            "original": unit,
            "normalized": normalized,
            "dimension": dimension.value if dimension else None,
            "is_known": dimension is not None,
        }


__all__ = [
    "router",
]
