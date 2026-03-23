# -*- coding: utf-8 -*-
"""
ERP/Finance Connector REST API Router - AGENT-DATA-003: ERP Connector

FastAPI router providing 20 endpoints for ERP connection management,
spend data sync, purchase order sync, inventory sync, vendor/material
mapping, Scope 3 emissions calculation, statistics, and health monitoring.

All endpoints are mounted under ``/api/v1/erp-connector``.

Endpoints:
    1.  POST   /v1/connections                    - Register ERP connection
    2.  GET    /v1/connections                    - List connections
    3.  GET    /v1/connections/{connection_id}    - Get connection details
    4.  POST   /v1/connections/{connection_id}/test - Test connectivity
    5.  DELETE /v1/connections/{connection_id}    - Remove connection
    6.  POST   /v1/spend/sync                    - Sync spend data
    7.  GET    /v1/spend                         - Query spend records
    8.  GET    /v1/spend/summary                 - Get spend summary
    9.  POST   /v1/purchase-orders/sync          - Sync purchase orders
    10. GET    /v1/purchase-orders               - Query purchase orders
    11. GET    /v1/purchase-orders/{po_number}   - Get single purchase order
    12. POST   /v1/inventory/sync                - Sync inventory
    13. GET    /v1/inventory                     - Query inventory
    14. POST   /v1/mappings/vendors              - Map vendor to Scope 3 category
    15. GET    /v1/mappings/vendors              - List vendor mappings
    16. POST   /v1/mappings/materials            - Map material to Scope 3 category
    17. POST   /v1/emissions/calculate           - Calculate Scope 3 emissions
    18. GET    /v1/emissions/summary             - Get emissions summary
    19. GET    /v1/statistics                    - Get service statistics
    20. GET    /health                           - Health check

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-003 ERP/Finance Connector
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
    logger.warning("FastAPI not available; ERP connector router is None")


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class RegisterConnectionBody(BaseModel):
        """Request body for registering an ERP connection."""
        erp_system: str = Field(
            ..., description="ERP system type (sap, oracle, netsuite, dynamics, simulated)",
        )
        host: str = Field(
            ..., description="ERP host address or URL",
        )
        port: int = Field(
            default=443, ge=1, le=65535, description="ERP connection port",
        )
        username: str = Field(
            ..., description="Authentication username",
        )
        password: Optional[str] = Field(
            None, description="Authentication password (stored encrypted)",
        )
        tenant_id: str = Field(
            default="default", description="Tenant identifier",
        )
        database_name: Optional[str] = Field(
            None, description="ERP database or instance name",
        )
        connection_params: Dict[str, Any] = Field(
            default_factory=dict, description="Additional connection parameters",
        )

    class SyncSpendBody(BaseModel):
        """Request body for syncing spend data."""
        connection_id: str = Field(
            ..., description="ERP connection ID to sync from",
        )
        start_date: str = Field(
            ..., description="Sync period start date (ISO 8601)",
        )
        end_date: str = Field(
            ..., description="Sync period end date (ISO 8601)",
        )
        vendor_ids: List[str] = Field(
            default_factory=list, description="Optional vendor ID filter",
        )
        spend_categories: List[str] = Field(
            default_factory=list, description="Optional spend category filter",
        )
        incremental: bool = Field(
            default=True, description="Whether to use incremental sync",
        )

    class SyncPOBody(BaseModel):
        """Request body for syncing purchase orders."""
        connection_id: str = Field(
            ..., description="ERP connection ID to sync from",
        )
        start_date: str = Field(
            ..., description="Sync period start date (ISO 8601)",
        )
        end_date: str = Field(
            ..., description="Sync period end date (ISO 8601)",
        )
        statuses: List[str] = Field(
            default_factory=list, description="Optional PO status filter",
        )
        incremental: bool = Field(
            default=True, description="Whether to use incremental sync",
        )

    class SyncInventoryBody(BaseModel):
        """Request body for syncing inventory data."""
        connection_id: str = Field(
            ..., description="ERP connection ID to sync from",
        )
        warehouse_ids: List[str] = Field(
            default_factory=list, description="Optional warehouse ID filter",
        )
        material_groups: List[str] = Field(
            default_factory=list, description="Optional material group filter",
        )

    class MapVendorBody(BaseModel):
        """Request body for mapping a vendor to a Scope 3 category."""
        vendor_id: str = Field(
            ..., description="Vendor identifier from ERP",
        )
        vendor_name: str = Field(
            ..., description="Vendor display name",
        )
        category: str = Field(
            ..., description="Scope 3 category (e.g. cat1_purchased_goods)",
        )
        spend_category: str = Field(
            default="general", description="Spend classification category",
        )
        emission_factor: Optional[float] = Field(
            None, ge=0.0, description="Custom emission factor (kgCO2e per unit currency)",
        )

    class MapMaterialBody(BaseModel):
        """Request body for mapping a material to a Scope 3 category."""
        material_id: str = Field(
            ..., description="Material identifier from ERP",
        )
        material_name: str = Field(
            ..., description="Material display name",
        )
        category: str = Field(
            ..., description="Scope 3 category (e.g. cat1_purchased_goods)",
        )
        spend_category: str = Field(
            default="general", description="Spend classification category",
        )
        emission_factor: Optional[float] = Field(
            None, ge=0.0, description="Custom emission factor (kgCO2e per unit)",
        )

    class CalculateEmissionsBody(BaseModel):
        """Request body for calculating Scope 3 emissions."""
        connection_id: str = Field(
            ..., description="ERP connection ID with synced data",
        )
        start_date: str = Field(
            ..., description="Calculation period start date (ISO 8601)",
        )
        end_date: str = Field(
            ..., description="Calculation period end date (ISO 8601)",
        )
        methodology: str = Field(
            default="eeio",
            description="Emission calculation methodology (eeio, spend_based, hybrid, supplier_specific)",
        )
        scope3_categories: List[str] = Field(
            default_factory=list, description="Optional Scope 3 category filter",
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/erp-connector",
        tags=["erp-connector"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Request) -> Any:
    """Extract ERPConnectorService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        ERPConnectorService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    service = getattr(request.app.state, "erp_connector_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="ERP connector service not configured",
        )
    return service


if FASTAPI_AVAILABLE:

    # ------------------------------------------------------------------
    # 1. Register ERP connection
    # ------------------------------------------------------------------
    @router.post("/v1/connections")
    async def register_connection(
        body: RegisterConnectionBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Register a new ERP system connection."""
        service = _get_service(request)
        try:
            record = service.register_connection(
                erp_system=body.erp_system,
                host=body.host,
                port=body.port,
                username=body.username,
                password=body.password,
                tenant_id=body.tenant_id,
                database_name=body.database_name,
                connection_params=body.connection_params,
            )
            return record.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 2. List connections
    # ------------------------------------------------------------------
    @router.get("/v1/connections")
    async def list_connections(
        tenant_id: Optional[str] = Query(None),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List registered ERP connections with optional tenant filter."""
        service = _get_service(request)
        connections = service.list_connections(tenant_id=tenant_id)
        return {
            "connections": [c.model_dump(mode="json") for c in connections],
            "count": len(connections),
        }

    # ------------------------------------------------------------------
    # 3. Get connection details
    # ------------------------------------------------------------------
    @router.get("/v1/connections/{connection_id}")
    async def get_connection(
        connection_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get ERP connection details by ID."""
        service = _get_service(request)
        record = service.get_connection(connection_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Connection {connection_id} not found",
            )
        return record.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 4. Test connection
    # ------------------------------------------------------------------
    @router.post("/v1/connections/{connection_id}/test")
    async def test_connection(
        connection_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Test an ERP connection's connectivity."""
        service = _get_service(request)
        try:
            result = service.test_connection(connection_id)
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # ------------------------------------------------------------------
    # 5. Remove connection
    # ------------------------------------------------------------------
    @router.delete("/v1/connections/{connection_id}")
    async def remove_connection(
        connection_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Remove an ERP connection."""
        service = _get_service(request)
        try:
            result = service.remove_connection(connection_id)
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # ------------------------------------------------------------------
    # 6. Sync spend data
    # ------------------------------------------------------------------
    @router.post("/v1/spend/sync")
    async def sync_spend(
        body: SyncSpendBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Sync spend data from an ERP connection."""
        service = _get_service(request)
        try:
            result = service.sync_spend(
                connection_id=body.connection_id,
                start_date=body.start_date,
                end_date=body.end_date,
                vendor_ids=body.vendor_ids,
                spend_categories=body.spend_categories,
                incremental=body.incremental,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 7. Query spend records
    # ------------------------------------------------------------------
    @router.get("/v1/spend")
    async def get_spend(
        connection_id: str = Query(...),
        start_date: Optional[str] = Query(None),
        end_date: Optional[str] = Query(None),
        vendor_ids: Optional[str] = Query(None, description="Comma-separated vendor IDs"),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """Query synced spend records with filters."""
        service = _get_service(request)
        vendor_id_list = vendor_ids.split(",") if vendor_ids else None
        records = service.get_spend(
            connection_id=connection_id,
            start_date=start_date,
            end_date=end_date,
            vendor_ids=vendor_id_list,
            limit=limit,
            offset=offset,
        )
        return {
            "spend_records": [r.model_dump(mode="json") for r in records],
            "count": len(records),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 8. Get spend summary
    # ------------------------------------------------------------------
    @router.get("/v1/spend/summary")
    async def get_spend_summary(
        connection_id: str = Query(...),
        start_date: Optional[str] = Query(None),
        end_date: Optional[str] = Query(None),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """Get aggregated spend summary for a connection."""
        service = _get_service(request)
        summary = service.get_spend_summary(
            connection_id=connection_id,
            start_date=start_date,
            end_date=end_date,
        )
        return summary.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 9. Sync purchase orders
    # ------------------------------------------------------------------
    @router.post("/v1/purchase-orders/sync")
    async def sync_purchase_orders(
        body: SyncPOBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Sync purchase orders from an ERP connection."""
        service = _get_service(request)
        try:
            result = service.sync_purchase_orders(
                connection_id=body.connection_id,
                start_date=body.start_date,
                end_date=body.end_date,
                statuses=body.statuses,
                incremental=body.incremental,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 10. Query purchase orders
    # ------------------------------------------------------------------
    @router.get("/v1/purchase-orders")
    async def get_purchase_orders(
        connection_id: str = Query(...),
        start_date: Optional[str] = Query(None),
        end_date: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """Query synced purchase orders with filters."""
        service = _get_service(request)
        orders = service.get_purchase_orders(
            connection_id=connection_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )
        return {
            "purchase_orders": [po.model_dump(mode="json") for po in orders],
            "count": len(orders),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 11. Get single purchase order
    # ------------------------------------------------------------------
    @router.get("/v1/purchase-orders/{po_number}")
    async def get_purchase_order(
        po_number: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a single purchase order by PO number."""
        service = _get_service(request)
        record = service.get_purchase_order(po_number)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Purchase order {po_number} not found",
            )
        return record.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 12. Sync inventory
    # ------------------------------------------------------------------
    @router.post("/v1/inventory/sync")
    async def sync_inventory(
        body: SyncInventoryBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Sync inventory data from an ERP connection."""
        service = _get_service(request)
        try:
            result = service.sync_inventory(
                connection_id=body.connection_id,
                warehouse_ids=body.warehouse_ids,
                material_groups=body.material_groups,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 13. Query inventory
    # ------------------------------------------------------------------
    @router.get("/v1/inventory")
    async def get_inventory(
        connection_id: str = Query(...),
        warehouse_id: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """Query synced inventory items with filters."""
        service = _get_service(request)
        items = service.get_inventory(
            connection_id=connection_id,
            warehouse_id=warehouse_id,
            limit=limit,
            offset=offset,
        )
        return {
            "inventory_items": [it.model_dump(mode="json") for it in items],
            "count": len(items),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 14. Map vendor to Scope 3 category
    # ------------------------------------------------------------------
    @router.post("/v1/mappings/vendors")
    async def map_vendor(
        body: MapVendorBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Map a vendor to a Scope 3 category with optional emission factor."""
        service = _get_service(request)
        try:
            mapping = service.map_vendor(
                vendor_id=body.vendor_id,
                vendor_name=body.vendor_name,
                category=body.category,
                spend_category=body.spend_category,
                emission_factor=body.emission_factor,
            )
            return mapping.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 15. List vendor mappings
    # ------------------------------------------------------------------
    @router.get("/v1/mappings/vendors")
    async def list_vendor_mappings(
        request: Request,
    ) -> Dict[str, Any]:
        """List all vendor-to-Scope-3 mappings."""
        service = _get_service(request)
        mappings = service.list_vendor_mappings()
        return {
            "vendor_mappings": [m.model_dump(mode="json") for m in mappings],
            "count": len(mappings),
        }

    # ------------------------------------------------------------------
    # 16. Map material to Scope 3 category
    # ------------------------------------------------------------------
    @router.post("/v1/mappings/materials")
    async def map_material(
        body: MapMaterialBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Map a material to a Scope 3 category with optional emission factor."""
        service = _get_service(request)
        try:
            mapping = service.map_material(
                material_id=body.material_id,
                material_name=body.material_name,
                category=body.category,
                spend_category=body.spend_category,
                emission_factor=body.emission_factor,
            )
            return mapping.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 17. Calculate Scope 3 emissions
    # ------------------------------------------------------------------
    @router.post("/v1/emissions/calculate")
    async def calculate_emissions(
        body: CalculateEmissionsBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Calculate Scope 3 emissions from synced ERP data."""
        service = _get_service(request)
        try:
            result = service.calculate_emissions(
                connection_id=body.connection_id,
                start_date=body.start_date,
                end_date=body.end_date,
                methodology=body.methodology,
                scope3_categories=body.scope3_categories,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 18. Get emissions summary
    # ------------------------------------------------------------------
    @router.get("/v1/emissions/summary")
    async def get_emissions_summary(
        connection_id: str = Query(...),
        start_date: Optional[str] = Query(None),
        end_date: Optional[str] = Query(None),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """Get aggregated emissions summary for a connection."""
        service = _get_service(request)
        summary = service.get_emissions_summary(
            connection_id=connection_id,
            start_date=start_date,
            end_date=end_date,
        )
        return summary.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 19. Get statistics
    # ------------------------------------------------------------------
    @router.get("/v1/statistics")
    async def get_statistics(
        request: Request,
    ) -> Dict[str, Any]:
        """Get ERP connector service statistics."""
        service = _get_service(request)
        stats = service.get_statistics()
        return stats.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 20. Health check
    # ------------------------------------------------------------------
    @router.get("/health")
    async def health() -> Dict[str, str]:
        """ERP connector service health check endpoint."""
        return {"status": "healthy", "service": "erp-connector"}


__all__ = [
    "router",
]
