# -*- coding: utf-8 -*-
"""
Refrigerants & F-Gas Agent REST API Router - AGENT-MRV-002

FastAPI router providing 20 REST API endpoints for the Refrigerants
& F-Gas service at ``/api/v1/refrigerants-fgas``.

Endpoints:
     1. POST   /calculate              - Single calculation
     2. POST   /calculate/batch        - Batch calculation
     3. GET    /calculations           - List calculations (paginated)
     4. GET    /calculations/{calc_id} - Get calculation details
     5. POST   /refrigerants           - Register custom refrigerant
     6. GET    /refrigerants           - List refrigerants (with category filter)
     7. GET    /refrigerants/{ref_id}  - Get refrigerant properties
     8. POST   /equipment              - Register equipment
     9. GET    /equipment              - List equipment (with filters)
    10. GET    /equipment/{equip_id}   - Get equipment details
    11. POST   /service-events         - Log service event
    12. GET    /service-events         - List service events
    13. POST   /leak-rates             - Register custom leak rate
    14. GET    /leak-rates             - List leak rates
    15. POST   /compliance/check       - Check compliance
    16. GET    /compliance             - List compliance records
    17. POST   /uncertainty            - Run uncertainty analysis
    18. GET    /audit/{calc_id}        - Get audit trail
    19. GET    /health                 - Health check
    20. GET    /stats                  - Service statistics

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-002 Refrigerants & F-Gas (GL-MRV-SCOPE1-002)
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, HTTPException, Query
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[assignment, misc]

router: Optional[Any] = None

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/refrigerants-fgas",
        tags=["refrigerants-fgas"],
    )

    # ------------------------------------------------------------------
    # Lazy service accessor
    # ------------------------------------------------------------------
    def _get_service():
        """Get the RefrigerantsFGasService singleton.

        Raises HTTPException 503 if the service has not been initialized.
        """
        from greenlang.refrigerants_fgas.setup import get_service
        svc = get_service()
        if svc is None:
            raise HTTPException(
                status_code=503,
                detail="Refrigerants & F-Gas service not initialized",
            )
        return svc

    # ==================================================================
    # 1. POST /calculate - Single calculation
    # ==================================================================

    @router.post("/calculate")
    async def calculate_emissions(
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate GHG emissions for a single refrigerant record.

        Request body must include refrigerant_type and charge_kg.
        Optional: method, gwp_source, equipment_type, equipment_id,
        facility_id, custom_leak_rate_pct, mass_balance_data.
        """
        svc = _get_service()
        refrigerant_type = body.get("refrigerant_type")
        charge_kg = body.get("charge_kg")

        if not refrigerant_type or charge_kg is None:
            raise HTTPException(
                status_code=400,
                detail="refrigerant_type and charge_kg are required",
            )

        try:
            return svc.calculate(
                refrigerant_type=refrigerant_type,
                charge_kg=float(charge_kg),
                method=body.get("method", "equipment_based"),
                gwp_source=body.get("gwp_source", "AR6"),
                equipment_type=body.get("equipment_type", ""),
                equipment_id=body.get("equipment_id", ""),
                facility_id=body.get("facility_id", ""),
                custom_leak_rate_pct=body.get("custom_leak_rate_pct"),
                mass_balance_data=body.get("mass_balance_data"),
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=str(exc),
            )

    # ==================================================================
    # 2. POST /calculate/batch - Batch calculation
    # ==================================================================

    @router.post("/calculate/batch")
    async def calculate_batch(
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Batch calculate GHG emissions for multiple refrigerant records.

        Request body must include 'inputs' (list of records) with each
        record containing refrigerant_type and charge_kg.
        """
        svc = _get_service()
        inputs = body.get("inputs")
        if not inputs or not isinstance(inputs, list):
            raise HTTPException(
                status_code=400,
                detail="'inputs' must be a non-empty list",
            )

        try:
            return svc.calculate_batch(inputs=inputs)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=str(exc),
            )

    # ==================================================================
    # 3. GET /calculations - List calculations (paginated)
    # ==================================================================

    @router.get("/calculations")
    async def list_calculations(
        refrigerant_type: Optional[str] = None,
        facility_id: Optional[str] = None,
        method: Optional[str] = None,
        status: Optional[str] = None,
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
    ) -> List[Dict[str, Any]]:
        """List stored calculation results with optional filters.

        Supports pagination via skip and limit query parameters.
        """
        svc = _get_service()
        all_calcs = list(svc._calculations.values())

        # Apply filters
        if refrigerant_type is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("refrigerant_type") == refrigerant_type
            ]
        if facility_id is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("facility_id") == facility_id
            ]
        if method is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("method") == method
            ]
        if status is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("status") == status
            ]

        return all_calcs[skip:skip + limit]

    # ==================================================================
    # 4. GET /calculations/{calc_id} - Get calculation details
    # ==================================================================

    @router.get("/calculations/{calc_id}")
    async def get_calculation(calc_id: str) -> Dict[str, Any]:
        """Get calculation details and audit trail for a specific calculation."""
        svc = _get_service()
        result = svc._calculations.get(calc_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Calculation {calc_id} not found",
            )

        # Attach audit trail
        audit = svc.get_audit_trail(calc_id)
        result_with_audit = dict(result)
        result_with_audit["audit_trail"] = audit
        return result_with_audit

    # ==================================================================
    # 5. POST /refrigerants - Register custom refrigerant
    # ==================================================================

    @router.post("/refrigerants")
    async def register_refrigerant(
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Register a custom refrigerant type with its properties.

        Request body must include refrigerant_type and category.
        Optional: display_name, formula, gwp_ar4, gwp_ar5, gwp_ar6,
        gwp_ar6_20yr, is_blend, components, ozone_depletion_potential.
        """
        svc = _get_service()
        ref_type = body.get("refrigerant_type", "")
        if not ref_type:
            raise HTTPException(
                status_code=400,
                detail="refrigerant_type is required",
            )

        if svc._refrigerant_database_engine is not None:
            try:
                result = svc._refrigerant_database_engine.register(
                    **body,
                )
                if isinstance(result, dict):
                    svc._refrigerants[ref_type] = result
                    return result
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump(mode="json")
                    svc._refrigerants[ref_type] = result_dict
                    return result_dict
            except (AttributeError, TypeError):
                pass

        # Fallback: in-memory registration
        from greenlang.refrigerants_fgas.setup import _compute_hash
        body.setdefault("status", "registered")
        body["provenance_hash"] = _compute_hash(body)
        svc._refrigerants[ref_type] = body
        return body

    # ==================================================================
    # 6. GET /refrigerants - List refrigerants (with category filter)
    # ==================================================================

    @router.get("/refrigerants")
    async def list_refrigerants(
        category: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List all available refrigerant types with optional category filter."""
        svc = _get_service()
        refs = svc.list_refrigerants(category=category)
        return refs[offset:offset + limit]

    # ==================================================================
    # 7. GET /refrigerants/{ref_id} - Get refrigerant properties
    # ==================================================================

    @router.get("/refrigerants/{ref_id}")
    async def get_refrigerant(ref_id: str) -> Dict[str, Any]:
        """Get properties and GWP values for a refrigerant type."""
        svc = _get_service()
        result = svc.get_refrigerant(ref_id)

        if result.get("error"):
            raise HTTPException(
                status_code=404,
                detail=f"Refrigerant type {ref_id} not found",
            )

        return result

    # ==================================================================
    # 8. POST /equipment - Register equipment
    # ==================================================================

    @router.post("/equipment")
    async def register_equipment(
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Register a refrigeration or HVAC equipment profile.

        Optional fields: equipment_id, equipment_type, name, facility_id,
        refrigerant_type, charge_kg, capacity_kw, age_years.
        """
        svc = _get_service()
        return svc.register_equipment(**body)

    # ==================================================================
    # 9. GET /equipment - List equipment (with filters)
    # ==================================================================

    @router.get("/equipment")
    async def list_equipment(
        equipment_type: Optional[str] = None,
        facility_id: Optional[str] = None,
        refrigerant_type: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List registered equipment profiles with optional filters."""
        svc = _get_service()
        profiles = list(svc._equipment_profiles.values())

        if equipment_type is not None:
            profiles = [
                p for p in profiles
                if p.get("equipment_type") == equipment_type
            ]
        if facility_id is not None:
            profiles = [
                p for p in profiles
                if p.get("facility_id") == facility_id
            ]
        if refrigerant_type is not None:
            profiles = [
                p for p in profiles
                if p.get("refrigerant_type") == refrigerant_type
            ]

        return profiles[offset:offset + limit]

    # ==================================================================
    # 10. GET /equipment/{equip_id} - Get equipment details
    # ==================================================================

    @router.get("/equipment/{equip_id}")
    async def get_equipment(equip_id: str) -> Dict[str, Any]:
        """Get equipment profile details by equipment ID."""
        svc = _get_service()
        result = svc._equipment_profiles.get(equip_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Equipment {equip_id} not found",
            )
        return result

    # ==================================================================
    # 11. POST /service-events - Log service event
    # ==================================================================

    @router.post("/service-events")
    async def log_service_event(
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Log a service event for an equipment item.

        Request body must include equipment_id and event_type.
        Optional: refrigerant_type, quantity_kg, date, technician, notes.
        """
        svc = _get_service()
        equipment_id = body.get("equipment_id", "")
        event_type = body.get("event_type", "")

        if not equipment_id or not event_type:
            raise HTTPException(
                status_code=400,
                detail="equipment_id and event_type are required",
            )

        return svc.log_service_event(
            equipment_id=equipment_id,
            event_type=event_type,
            refrigerant_type=body.get("refrigerant_type", ""),
            quantity_kg=float(body.get("quantity_kg", 0.0)),
            date=body.get("date"),
            technician=body.get("technician", ""),
            notes=body.get("notes", ""),
        )

    # ==================================================================
    # 12. GET /service-events - List service events
    # ==================================================================

    @router.get("/service-events")
    async def list_service_events(
        equipment_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List service events with optional equipment and type filters."""
        svc = _get_service()
        events = list(svc._service_events.values())

        if equipment_id is not None:
            events = [
                e for e in events
                if e.get("equipment_id") == equipment_id
            ]
        if event_type is not None:
            events = [
                e for e in events
                if e.get("event_type") == event_type
            ]

        return events[offset:offset + limit]

    # ==================================================================
    # 13. POST /leak-rates - Register custom leak rate
    # ==================================================================

    @router.post("/leak-rates")
    async def register_leak_rate(
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Register a custom leak rate for an equipment type.

        Request body must include equipment_type and base_rate_pct.
        Optional: age_factor, climate_factor, ldar_adjustment, source.
        """
        svc = _get_service()
        equipment_type = body.get("equipment_type", "")
        if not equipment_type:
            raise HTTPException(
                status_code=400,
                detail="equipment_type is required",
            )

        from greenlang.refrigerants_fgas.setup import (
            LeakRateResponse,
            _compute_hash,
            _new_uuid,
        )

        base_rate = float(body.get("base_rate_pct", 5.0))
        age_factor = float(body.get("age_factor", 1.0))
        climate_factor = float(body.get("climate_factor", 1.0))
        ldar_adjustment = float(body.get("ldar_adjustment", 1.0))
        effective_rate = base_rate * age_factor * climate_factor * ldar_adjustment
        effective_rate = min(effective_rate, 100.0)

        lr = LeakRateResponse(
            leak_rate_id=body.get("leak_rate_id", _new_uuid()),
            equipment_type=equipment_type,
            base_rate_pct=base_rate,
            age_factor=age_factor,
            climate_factor=climate_factor,
            ldar_adjustment=ldar_adjustment,
            effective_rate_pct=round(effective_rate, 4),
            source=body.get("source", "custom"),
        )
        lr.provenance_hash = _compute_hash(lr)
        result_dict = lr.model_dump()
        svc._leak_rates[lr.leak_rate_id] = result_dict

        logger.info(
            "Registered leak rate %s: type=%s rate=%.2f%%",
            lr.leak_rate_id, equipment_type, effective_rate,
        )
        return result_dict

    # ==================================================================
    # 14. GET /leak-rates - List leak rates
    # ==================================================================

    @router.get("/leak-rates")
    async def list_leak_rates(
        equipment_type: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List registered leak rates with optional equipment type filter."""
        svc = _get_service()
        rates = list(svc._leak_rates.values())

        if equipment_type is not None:
            rates = [
                r for r in rates
                if r.get("equipment_type") == equipment_type
            ]

        return rates[offset:offset + limit]

    # ==================================================================
    # 15. POST /compliance/check - Check compliance
    # ==================================================================

    @router.post("/compliance/check")
    async def check_compliance(
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check regulatory compliance for a calculation or set of frameworks.

        Request body may include:
        - calculation_id: Optional calculation to check.
        - frameworks: Optional list of framework names.
        """
        svc = _get_service()
        calculation_id = body.get("calculation_id")
        frameworks = body.get("frameworks")

        result = svc.check_compliance(
            calculation_id=calculation_id,
            frameworks=frameworks,
        )

        # Cache compliance records
        if calculation_id and isinstance(result, dict):
            svc._compliance_records[calculation_id] = result

        return result

    # ==================================================================
    # 16. GET /compliance - List compliance records
    # ==================================================================

    @router.get("/compliance")
    async def list_compliance(
        framework: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List stored compliance check results with optional framework filter."""
        svc = _get_service()
        all_records = list(svc._compliance_records.values())

        if framework is not None:
            filtered = []
            for rec in all_records:
                records = rec.get("records", [])
                matching = [
                    r for r in records
                    if r.get("framework") == framework
                ]
                if matching:
                    filtered.append({
                        "records": matching,
                        "framework_filter": framework,
                    })
            all_records = filtered

        return all_records[offset:offset + limit]

    # ==================================================================
    # 17. POST /uncertainty - Run uncertainty analysis
    # ==================================================================

    @router.post("/uncertainty")
    async def run_uncertainty(
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run uncertainty analysis on a calculation result.

        Request body must include calculation_id. Optional: iterations.
        """
        svc = _get_service()
        calc_id = body.get("calculation_id", "")

        if not calc_id:
            raise HTTPException(
                status_code=400,
                detail="calculation_id is required",
            )

        calc_result = svc._calculations.get(calc_id)
        if calc_result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Calculation {calc_id} not found",
            )

        return svc.run_uncertainty(
            calculation_id=calc_id,
            iterations=body.get("iterations"),
        )

    # ==================================================================
    # 18. GET /audit/{calc_id} - Get audit trail
    # ==================================================================

    @router.get("/audit/{calc_id}")
    async def get_audit_trail(calc_id: str) -> Dict[str, Any]:
        """Get the audit trail for a specific calculation."""
        svc = _get_service()

        # Verify calculation exists
        calc_result = svc._calculations.get(calc_id)
        if calc_result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Calculation {calc_id} not found",
            )

        entries = svc.get_audit_trail(calc_id)

        # Compute chain hash
        from greenlang.refrigerants_fgas.setup import _compute_hash
        chain_hash = _compute_hash(entries) if entries else ""

        return {
            "calculation_id": calc_id,
            "entries": entries,
            "total_entries": len(entries),
            "chain_hash": chain_hash,
        }

    # ==================================================================
    # 19. GET /health - Health check
    # ==================================================================

    @router.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check for the refrigerants & F-gas service."""
        svc = _get_service()
        return svc.get_health()

    # ==================================================================
    # 20. GET /stats - Service statistics
    # ==================================================================

    @router.get("/stats")
    async def get_stats() -> Dict[str, Any]:
        """Get aggregate statistics for the refrigerants & F-gas service."""
        svc = _get_service()
        return svc.get_stats()


__all__ = ["router"]
