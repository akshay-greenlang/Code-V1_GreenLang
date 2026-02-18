# -*- coding: utf-8 -*-
"""
Stationary Combustion Agent REST API Router - AGENT-MRV-001

FastAPI router providing 20 REST API endpoints for the Stationary
Combustion service at ``/api/v1/stationary-combustion``.

Endpoints:
     1. POST   /calculate              - Calculate emissions for single record
     2. POST   /calculate/batch        - Batch calculate for multiple records
     3. GET    /calculations           - List calculation results (pagination)
     4. GET    /calculations/{calc_id} - Get calculation details + audit trail
     5. POST   /fuels                  - Register a custom fuel type
     6. GET    /fuels                  - List all fuel types
     7. GET    /fuels/{fuel_id}        - Get fuel properties
     8. POST   /factors                - Register a custom emission factor
     9. GET    /factors                - List emission factors (filterable)
    10. GET    /factors/{factor_id}    - Get emission factor details
    11. POST   /equipment              - Register equipment profile
    12. GET    /equipment              - List equipment profiles
    13. GET    /equipment/{equip_id}   - Get equipment profile details
    14. POST   /aggregate              - Aggregate calculations for facility
    15. GET    /aggregations           - List facility aggregations
    16. POST   /uncertainty            - Run uncertainty analysis
    17. GET    /audit/{calc_id}        - Get audit trail for calculation
    18. POST   /validate               - Validate input data without calculating
    19. GET    /health                 - Health check
    20. GET    /stats                  - Service statistics

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-001 Stationary Combustion (GL-MRV-SCOPE1-001)
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
        prefix="/api/v1/stationary-combustion",
        tags=["Stationary Combustion"],
    )

    # ------------------------------------------------------------------
    # Lazy service accessor
    # ------------------------------------------------------------------
    def _get_service():
        """Get the StationaryCombustionService singleton.

        Raises HTTPException 503 if the service has not been initialized.
        """
        from greenlang.stationary_combustion.setup import get_service
        svc = get_service()
        if svc is None:
            raise HTTPException(
                status_code=503,
                detail="Stationary Combustion service not initialized",
            )
        return svc

    # ==================================================================
    # 1. POST /calculate - Calculate emissions for single record
    # ==================================================================

    @router.post("/calculate")
    async def calculate_emissions(body: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate GHG emissions for a single stationary combustion record.

        Request body must include fuel_type, quantity, and unit.
        Optional: gwp_source, ef_source, tier, heating_value_basis,
        facility_id, equipment_id, include_biogenic.
        """
        svc = _get_service()
        fuel_type = body.get("fuel_type")
        quantity = body.get("quantity")
        unit = body.get("unit")

        if not fuel_type or quantity is None or not unit:
            raise HTTPException(
                status_code=400,
                detail="fuel_type, quantity, and unit are required",
            )

        return svc.calculate(
            fuel_type=fuel_type,
            quantity=float(quantity),
            unit=unit,
            gwp_source=body.get("gwp_source", "AR6"),
            ef_source=body.get("ef_source", "EPA"),
            tier=body.get("tier"),
            heating_value_basis=body.get("heating_value_basis", "HHV"),
            include_biogenic=body.get("include_biogenic", False),
            facility_id=body.get("facility_id"),
            equipment_id=body.get("equipment_id"),
        )

    # ==================================================================
    # 2. POST /calculate/batch - Batch calculate for multiple records
    # ==================================================================

    @router.post("/calculate/batch")
    async def calculate_batch(body: Dict[str, Any]) -> Dict[str, Any]:
        """Batch calculate GHG emissions for multiple combustion records.

        Request body must include 'inputs' (list of records) with each
        record containing fuel_type, quantity, and unit.
        Optional: gwp_source, include_biogenic.
        """
        svc = _get_service()
        inputs = body.get("inputs")
        if not inputs or not isinstance(inputs, list):
            raise HTTPException(
                status_code=400,
                detail="'inputs' must be a non-empty list",
            )

        return svc.calculate_batch(
            inputs=inputs,
            gwp_source=body.get("gwp_source", "AR6"),
            include_biogenic=body.get("include_biogenic", False),
        )

    # ==================================================================
    # 3. GET /calculations - List calculation results
    # ==================================================================

    @router.get("/calculations")
    async def list_calculations(
        fuel_type: Optional[str] = None,
        facility_id: Optional[str] = None,
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
        if fuel_type is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("fuel_type") == fuel_type
            ]
        if facility_id is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("facility_id") == facility_id
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
    # 5. POST /fuels - Register a custom fuel type
    # ==================================================================

    @router.post("/fuels")
    async def register_fuel(body: Dict[str, Any]) -> Dict[str, Any]:
        """Register a custom fuel type with its physical properties.

        Request body must include fuel_type, category, display_name,
        hhv, and ncv.
        """
        svc = _get_service()
        fuel_type = body.get("fuel_type", "")
        if not fuel_type:
            raise HTTPException(
                status_code=400,
                detail="fuel_type is required",
            )

        if svc._fuel_database_engine is not None:
            try:
                result = svc._fuel_database_engine.register_fuel(**body)
                if isinstance(result, dict):
                    svc._fuel_types[fuel_type] = result
                    return result
            except (AttributeError, TypeError):
                pass

        # Fallback: in-memory registration
        svc._fuel_types[fuel_type] = body
        body.setdefault("status", "registered")
        return body

    # ==================================================================
    # 6. GET /fuels - List all fuel types
    # ==================================================================

    @router.get("/fuels")
    async def list_fuels(
        category: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List all available fuel types with optional category filter."""
        svc = _get_service()
        fuels = svc.list_fuel_types()

        if category is not None:
            fuels = [
                f for f in fuels
                if f.get("category") == category
            ]

        return fuels[offset:offset + limit]

    # ==================================================================
    # 7. GET /fuels/{fuel_id} - Get fuel properties
    # ==================================================================

    @router.get("/fuels/{fuel_id}")
    async def get_fuel(fuel_id: str) -> Dict[str, Any]:
        """Get physical and regulatory properties for a fuel type."""
        svc = _get_service()
        result = svc.get_fuel_properties(fuel_id)

        if result.get("error"):
            raise HTTPException(
                status_code=404,
                detail=f"Fuel type {fuel_id} not found",
            )

        return result

    # ==================================================================
    # 8. POST /factors - Register a custom emission factor
    # ==================================================================

    @router.post("/factors")
    async def register_factor(body: Dict[str, Any]) -> Dict[str, Any]:
        """Register a custom emission factor.

        Request body must include fuel_type, gas, value, unit, and source.
        """
        svc = _get_service()
        fuel_type = body.get("fuel_type", "")
        gas = body.get("gas", "")
        source = body.get("source", "CUSTOM")

        if not fuel_type or not gas:
            raise HTTPException(
                status_code=400,
                detail="fuel_type and gas are required",
            )

        factor_key = f"{fuel_type}:{gas}:{source}"

        if svc._factor_selector_engine is not None:
            try:
                result = svc._factor_selector_engine.register_factor(**body)
                if isinstance(result, dict):
                    svc._emission_factors[factor_key] = result
                    return result
            except (AttributeError, TypeError):
                pass

        # Fallback: in-memory registration
        from greenlang.stationary_combustion.setup import _new_uuid, _compute_hash
        body.setdefault("factor_id", _new_uuid())
        body["provenance_hash"] = _compute_hash(body)
        svc._emission_factors[factor_key] = body
        return body

    # ==================================================================
    # 9. GET /factors - List emission factors
    # ==================================================================

    @router.get("/factors")
    async def list_factors(
        fuel_type: Optional[str] = None,
        source: Optional[str] = None,
        gas: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List emission factors with optional filters."""
        svc = _get_service()
        factors = list(svc._emission_factors.values())

        if fuel_type is not None:
            factors = [
                f for f in factors
                if f.get("fuel_type") == fuel_type
            ]
        if source is not None:
            factors = [
                f for f in factors
                if f.get("source") == source
            ]
        if gas is not None:
            factors = [
                f for f in factors
                if f.get("gas") == gas
            ]

        return factors[offset:offset + limit]

    # ==================================================================
    # 10. GET /factors/{factor_id} - Get emission factor details
    # ==================================================================

    @router.get("/factors/{factor_id}")
    async def get_factor(factor_id: str) -> Dict[str, Any]:
        """Get emission factor details by factor ID."""
        svc = _get_service()

        # Search by factor_id across all stored factors
        for factor in svc._emission_factors.values():
            if factor.get("factor_id") == factor_id:
                return factor

        raise HTTPException(
            status_code=404,
            detail=f"Emission factor {factor_id} not found",
        )

    # ==================================================================
    # 11. POST /equipment - Register equipment profile
    # ==================================================================

    @router.post("/equipment")
    async def register_equipment(body: Dict[str, Any]) -> Dict[str, Any]:
        """Register a stationary combustion equipment profile.

        Optional fields: equipment_id, equipment_type, name, facility_id,
        rated_capacity_mw, efficiency, load_factor, age_years.
        """
        svc = _get_service()
        return svc.register_equipment(**body)

    # ==================================================================
    # 12. GET /equipment - List equipment profiles
    # ==================================================================

    @router.get("/equipment")
    async def list_equipment(
        equipment_type: Optional[str] = None,
        facility_id: Optional[str] = None,
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

        return profiles[offset:offset + limit]

    # ==================================================================
    # 13. GET /equipment/{equip_id} - Get equipment profile details
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
    # 14. POST /aggregate - Aggregate calculations for facility
    # ==================================================================

    @router.post("/aggregate")
    async def aggregate_calculations(body: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate calculation results by facility.

        Request body may include:
        - calculation_ids: List of calculation IDs to aggregate.
        - control_approach: OPERATIONAL, FINANCIAL, or EQUITY_SHARE.
        - facility_id: Optional facility filter.
        """
        svc = _get_service()
        calc_ids = body.get("calculation_ids", [])
        control_approach = body.get("control_approach", "OPERATIONAL")
        facility_id = body.get("facility_id")

        # Gather calculation results
        calcs = []
        if calc_ids:
            for cid in calc_ids:
                c = svc._calculations.get(cid)
                if c is not None:
                    calcs.append(c)
        else:
            calcs = list(svc._calculations.values())

        if facility_id is not None:
            calcs = [
                c for c in calcs
                if c.get("facility_id") == facility_id
            ]

        if not calcs:
            return {
                "aggregations": [],
                "total_facilities": 0,
                "control_approach": control_approach,
            }

        # Delegate to pipeline engine if available
        if svc._pipeline_engine is not None:
            from greenlang.stationary_combustion.models import CalculationResult
            try:
                # Attempt to convert dicts to CalculationResult
                calc_results = []
                for c in calcs:
                    try:
                        calc_results.append(CalculationResult(**c))
                    except (TypeError, ValueError):
                        pass

                if calc_results:
                    aggregations = svc._pipeline_engine.aggregate_by_facility(
                        calc_results, control_approach,
                    )
                    agg_dicts = [
                        a.model_dump(mode="json") for a in aggregations
                    ]
                    # Cache aggregations
                    for ad in agg_dicts:
                        svc._aggregations[ad["facility_id"]] = ad
                    return {
                        "aggregations": agg_dicts,
                        "total_facilities": len(agg_dicts),
                        "control_approach": control_approach,
                    }
            except (TypeError, ValueError, AttributeError):
                pass

        # Fallback: simple aggregation
        from collections import defaultdict
        facility_map = defaultdict(lambda: {
            "total_co2e_tonnes": 0.0,
            "calculation_count": 0,
        })
        for c in calcs:
            fid = c.get("facility_id", "UNASSIGNED")
            facility_map[fid]["total_co2e_tonnes"] += c.get(
                "total_co2e_tonnes", 0.0,
            )
            facility_map[fid]["calculation_count"] += 1

        agg_dicts = [
            {
                "facility_id": fid,
                "total_co2e_tonnes": data["total_co2e_tonnes"],
                "calculation_count": data["calculation_count"],
                "control_approach": control_approach,
            }
            for fid, data in facility_map.items()
        ]

        return {
            "aggregations": agg_dicts,
            "total_facilities": len(agg_dicts),
            "control_approach": control_approach,
        }

    # ==================================================================
    # 15. GET /aggregations - List facility aggregations
    # ==================================================================

    @router.get("/aggregations")
    async def list_aggregations(
        facility_id: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List facility-level emission aggregations."""
        svc = _get_service()
        aggs = list(svc._aggregations.values())

        if facility_id is not None:
            aggs = [
                a for a in aggs
                if a.get("facility_id") == facility_id
            ]

        return aggs[offset:offset + limit]

    # ==================================================================
    # 16. POST /uncertainty - Run uncertainty analysis
    # ==================================================================

    @router.post("/uncertainty")
    async def run_uncertainty(body: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo uncertainty analysis on a calculation result.

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

        return svc.get_uncertainty(
            calculation_result=calc_result,
            iterations=body.get("iterations"),
        )

    # ==================================================================
    # 17. GET /audit/{calc_id} - Get audit trail for calculation
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
        return {
            "calculation_id": calc_id,
            "entries": entries,
            "total_entries": len(entries),
        }

    # ==================================================================
    # 18. POST /validate - Validate input data without calculating
    # ==================================================================

    @router.post("/validate")
    async def validate_inputs(body: Dict[str, Any]) -> Dict[str, Any]:
        """Validate combustion input data without performing calculations.

        Request body must include 'inputs' (list of records).
        """
        svc = _get_service()
        inputs = body.get("inputs")
        if not inputs or not isinstance(inputs, list):
            raise HTTPException(
                status_code=400,
                detail="'inputs' must be a non-empty list",
            )

        if svc._pipeline_engine is not None:
            from greenlang.stationary_combustion.models import (
                CombustionInput,
            )
            from decimal import Decimal

            from greenlang.stationary_combustion.setup import _new_uuid

            combustion_inputs = []
            parse_errors = []
            for idx, inp in enumerate(inputs):
                try:
                    ci = CombustionInput(
                        calculation_id=inp.get(
                            "calculation_id", _new_uuid(),
                        ),
                        fuel_type=inp["fuel_type"],
                        quantity=Decimal(str(inp["quantity"])),
                        unit=inp["unit"],
                        heating_value_basis=inp.get(
                            "heating_value_basis", "HHV",
                        ),
                        ef_source=inp.get("ef_source", "EPA"),
                        tier=inp.get("tier"),
                        facility_id=inp.get("facility_id"),
                        equipment_id=inp.get("equipment_id"),
                    )
                    combustion_inputs.append(ci)
                except (KeyError, ValueError, TypeError) as exc:
                    parse_errors.append(
                        f"Input [{idx}]: parse error - {exc}"
                    )

            if parse_errors:
                return {
                    "valid": False,
                    "errors": parse_errors,
                    "warnings": [],
                    "validated_count": len(combustion_inputs),
                    "total_count": len(inputs),
                }

            return svc._pipeline_engine.validate_inputs(combustion_inputs)

        # Fallback: basic validation
        errors = []
        for idx, inp in enumerate(inputs):
            if "fuel_type" not in inp:
                errors.append(f"Input [{idx}]: fuel_type is required")
            if "quantity" not in inp:
                errors.append(f"Input [{idx}]: quantity is required")
            if "unit" not in inp:
                errors.append(f"Input [{idx}]: unit is required")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": [],
            "validated_count": len(inputs) - len(errors),
            "total_count": len(inputs),
        }

    # ==================================================================
    # 19. GET /health - Health check
    # ==================================================================

    @router.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check for the stationary combustion service."""
        svc = _get_service()
        return svc.get_health()

    # ==================================================================
    # 20. GET /stats - Service statistics
    # ==================================================================

    @router.get("/stats")
    async def get_stats() -> Dict[str, Any]:
        """Get aggregate statistics for the stationary combustion service."""
        svc = _get_service()
        return svc.get_statistics()


__all__ = ["router"]
