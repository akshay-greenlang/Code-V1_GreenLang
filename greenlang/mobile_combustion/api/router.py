# -*- coding: utf-8 -*-
"""
Mobile Combustion Agent REST API Router - AGENT-MRV-003

FastAPI router providing 20 REST API endpoints for the Mobile Combustion
service at ``/api/v1/mobile-combustion``.

Endpoints:
     1. POST   /calculate              - Calculate mobile combustion emissions
     2. POST   /calculate/batch        - Batch calculate
     3. GET    /calculations           - List calculations (paginated)
     4. GET    /calculations/{calc_id} - Get calculation detail
     5. POST   /vehicles               - Register vehicle
     6. GET    /vehicles               - List vehicles (paginated, filterable)
     7. GET    /vehicles/{vehicle_id}  - Get vehicle detail
     8. POST   /trips                  - Log a trip
     9. GET    /trips                  - List trips (paginated, filterable)
    10. GET    /trips/{trip_id}        - Get trip detail
    11. POST   /fuels                  - Register custom fuel
    12. GET    /fuels                  - List fuel types
    13. POST   /factors                - Register custom emission factor
    14. GET    /factors                - List emission factors
    15. POST   /aggregate              - Aggregate fleet emissions
    16. GET    /aggregations           - List aggregations
    17. POST   /uncertainty            - Run uncertainty analysis
    18. POST   /compliance/check       - Run compliance check
    19. GET    /health                 - Health check
    20. GET    /stats                  - Service statistics

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-003 Mobile Combustion (GL-MRV-SCOPE1-003)
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
        prefix="/api/v1/mobile-combustion",
        tags=["mobile-combustion"],
    )

    # ------------------------------------------------------------------
    # Lazy service accessor
    # ------------------------------------------------------------------
    def _get_service():
        """Get the MobileCombustionService singleton.

        Raises HTTPException 503 if the service has not been initialized.
        """
        from greenlang.mobile_combustion.setup import get_service
        svc = get_service()
        if svc is None:
            raise HTTPException(
                status_code=503,
                detail="Mobile Combustion service not initialized",
            )
        return svc

    # ==================================================================
    # 1. POST /calculate - Calculate mobile combustion emissions
    # ==================================================================

    @router.post("/calculate")
    async def calculate_emissions(
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate GHG emissions for a single mobile combustion record.

        Request body must include calculation_method (FUEL_BASED,
        DISTANCE_BASED, or SPEND_BASED) and the corresponding required
        fields. For FUEL_BASED: fuel_quantity and fuel_unit. For
        DISTANCE_BASED: distance and distance_unit. For SPEND_BASED:
        spend_amount. Optional: vehicle_type, fuel_type, gwp_source,
        vehicle_id, facility_id, tier.
        """
        svc = _get_service()

        method = body.get("calculation_method", "FUEL_BASED").upper()
        if method == "FUEL_BASED" and body.get("fuel_quantity") is None:
            raise HTTPException(
                status_code=400,
                detail="fuel_quantity is required for FUEL_BASED method",
            )
        if method == "DISTANCE_BASED" and body.get("distance") is None:
            raise HTTPException(
                status_code=400,
                detail="distance is required for DISTANCE_BASED method",
            )
        if method == "SPEND_BASED" and body.get("spend_amount") is None:
            raise HTTPException(
                status_code=400,
                detail="spend_amount is required for SPEND_BASED method",
            )

        try:
            return svc.calculate(input_data=body)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=str(exc),
            )

    # ==================================================================
    # 2. POST /calculate/batch - Batch calculate
    # ==================================================================

    @router.post("/calculate/batch")
    async def calculate_batch(
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Batch calculate GHG emissions for multiple records.

        Request body must include 'inputs' (list of records) with each
        record containing the required fields for its calculation method.
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
        vehicle_type: Optional[str] = None,
        fuel_type: Optional[str] = None,
        calculation_method: Optional[str] = None,
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

        if vehicle_type is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("vehicle_type") == vehicle_type
            ]
        if fuel_type is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("fuel_type") == fuel_type
            ]
        if calculation_method is not None:
            all_calcs = [
                c for c in all_calcs
                if c.get("calculation_method") == calculation_method
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
    # 4. GET /calculations/{calc_id} - Get calculation detail
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

        result_with_audit = dict(result)
        audit = result.get("audit_entries", [])
        result_with_audit["audit_trail"] = audit
        return result_with_audit

    # ==================================================================
    # 5. POST /vehicles - Register vehicle
    # ==================================================================

    @router.post("/vehicles", status_code=201)
    async def register_vehicle(
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Register a vehicle in the fleet.

        Optional fields: vehicle_id, vehicle_type, fuel_type, name,
        facility_id, fleet_id, make, model, year, fuel_economy,
        fuel_economy_unit, odometer_km.
        """
        svc = _get_service()
        vehicle_id = svc.register_vehicle(registration=body)
        vehicle = svc._vehicles.get(vehicle_id, {"vehicle_id": vehicle_id})
        return vehicle

    # ==================================================================
    # 6. GET /vehicles - List vehicles (paginated, filterable)
    # ==================================================================

    @router.get("/vehicles")
    async def list_vehicles(
        vehicle_type: Optional[str] = None,
        fuel_type: Optional[str] = None,
        facility_id: Optional[str] = None,
        fleet_id: Optional[str] = None,
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
    ) -> List[Dict[str, Any]]:
        """List registered vehicles with optional filters."""
        svc = _get_service()
        filters = {}
        if vehicle_type is not None:
            filters["vehicle_type"] = vehicle_type
        if fuel_type is not None:
            filters["fuel_type"] = fuel_type
        if facility_id is not None:
            filters["facility_id"] = facility_id
        if fleet_id is not None:
            filters["fleet_id"] = fleet_id

        vehicles = svc.list_vehicles(filters=filters)
        return vehicles[skip:skip + limit]

    # ==================================================================
    # 7. GET /vehicles/{vehicle_id} - Get vehicle detail
    # ==================================================================

    @router.get("/vehicles/{vehicle_id}")
    async def get_vehicle(vehicle_id: str) -> Dict[str, Any]:
        """Get vehicle details by vehicle ID."""
        svc = _get_service()
        result = svc.get_vehicle(vehicle_id)

        if result.get("error"):
            raise HTTPException(
                status_code=404,
                detail=f"Vehicle {vehicle_id} not found",
            )

        return result

    # ==================================================================
    # 8. POST /trips - Log a trip
    # ==================================================================

    @router.post("/trips", status_code=201)
    async def log_trip(
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Log a trip for a vehicle.

        Request body must include vehicle_id. Optional: distance_km,
        distance_unit, fuel_quantity, fuel_unit, start_date, end_date,
        origin, destination, purpose.
        """
        svc = _get_service()
        vehicle_id = body.get("vehicle_id", "")

        if not vehicle_id:
            raise HTTPException(
                status_code=400,
                detail="vehicle_id is required",
            )

        trip_id = svc.log_trip(trip=body)
        trip = svc._trips.get(trip_id, {"trip_id": trip_id})
        return trip

    # ==================================================================
    # 9. GET /trips - List trips (paginated, filterable)
    # ==================================================================

    @router.get("/trips")
    async def list_trips(
        vehicle_id: Optional[str] = None,
        purpose: Optional[str] = None,
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
    ) -> List[Dict[str, Any]]:
        """List trip records with optional vehicle and purpose filters."""
        svc = _get_service()
        trips = list(svc._trips.values())

        if vehicle_id is not None:
            trips = [
                t for t in trips
                if t.get("vehicle_id") == vehicle_id
            ]
        if purpose is not None:
            trips = [
                t for t in trips
                if t.get("purpose") == purpose
            ]

        return trips[skip:skip + limit]

    # ==================================================================
    # 10. GET /trips/{trip_id} - Get trip detail
    # ==================================================================

    @router.get("/trips/{trip_id}")
    async def get_trip(trip_id: str) -> Dict[str, Any]:
        """Get trip details by trip ID."""
        svc = _get_service()
        result = svc._trips.get(trip_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Trip {trip_id} not found",
            )
        return result

    # ==================================================================
    # 11. POST /fuels - Register custom fuel
    # ==================================================================

    @router.post("/fuels", status_code=201)
    async def register_fuel(
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Register a custom fuel type.

        Request body must include fuel_type. Optional: display_name,
        category, co2_kg_per_gallon, fossil_fraction.
        """
        svc = _get_service()
        fuel_type = body.get("fuel_type", "")
        if not fuel_type:
            raise HTTPException(
                status_code=400,
                detail="fuel_type is required",
            )

        from greenlang.mobile_combustion.setup import _compute_hash

        body.setdefault("display_name", fuel_type.replace("_", " ").title())
        body.setdefault("category", "CUSTOM")
        body["provenance_hash"] = _compute_hash(body)
        svc._fuels[fuel_type] = body

        logger.info("Registered custom fuel type: %s", fuel_type)
        return body

    # ==================================================================
    # 12. GET /fuels - List fuel types
    # ==================================================================

    @router.get("/fuels")
    async def list_fuels(
        category: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List all available fuel types with optional category filter."""
        svc = _get_service()
        fuels = svc.get_fuel_types()

        # Include custom fuels
        custom_fuels = list(svc._fuels.values())
        all_fuels = fuels + custom_fuels

        if category is not None:
            all_fuels = [
                f for f in all_fuels
                if f.get("category") == category
            ]

        return all_fuels[offset:offset + limit]

    # ==================================================================
    # 13. POST /factors - Register custom emission factor
    # ==================================================================

    @router.post("/factors", status_code=201)
    async def register_factor(
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Register a custom emission factor.

        Request body must include fuel_type, gas, and value. Optional:
        unit, source, geography, year, reference.
        """
        svc = _get_service()
        fuel_type = body.get("fuel_type", "")
        gas = body.get("gas", "")

        if not fuel_type or not gas:
            raise HTTPException(
                status_code=400,
                detail="fuel_type and gas are required",
            )

        if body.get("value") is None:
            raise HTTPException(
                status_code=400,
                detail="value is required",
            )

        from greenlang.mobile_combustion.setup import (
            _compute_hash,
            _new_uuid,
        )

        factor_id = body.get("factor_id", _new_uuid())
        body["factor_id"] = factor_id
        body.setdefault("unit", "kg/gallon")
        body.setdefault("source", "CUSTOM")
        body.setdefault("geography", "GLOBAL")
        body.setdefault("year", 2025)
        body["provenance_hash"] = _compute_hash(body)
        svc._emission_factors[factor_id] = body

        logger.info(
            "Registered custom emission factor %s: %s %s = %s",
            factor_id, fuel_type, gas, body.get("value"),
        )
        return body

    # ==================================================================
    # 14. GET /factors - List emission factors
    # ==================================================================

    @router.get("/factors")
    async def list_factors(
        fuel_type: Optional[str] = None,
        vehicle_type: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List emission factors with optional filters."""
        svc = _get_service()
        filters = {}
        if fuel_type is not None:
            filters["fuel_type"] = fuel_type
        if vehicle_type is not None:
            filters["vehicle_type"] = vehicle_type
        if source is not None:
            filters["source"] = source

        factors = svc.get_emission_factors(filters=filters)

        # Include custom factors
        custom_factors = list(svc._emission_factors.values())
        if fuel_type:
            custom_factors = [
                f for f in custom_factors
                if f.get("fuel_type") == fuel_type
            ]
        if source:
            custom_factors = [
                f for f in custom_factors
                if f.get("source") == source
            ]

        all_factors = factors + custom_factors
        return all_factors[offset:offset + limit]

    # ==================================================================
    # 15. POST /aggregate - Aggregate fleet emissions
    # ==================================================================

    @router.post("/aggregate")
    async def aggregate_fleet(
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregate fleet emissions for a reporting period.

        Request body may include: period, facility_id, fleet_id,
        vehicle_type, fuel_type.
        """
        svc = _get_service()
        period = body.get("period", "")
        filters = {
            k: v for k, v in body.items()
            if k in ("facility_id", "fleet_id", "vehicle_type", "fuel_type")
            and v is not None
        }

        return svc.aggregate_fleet(period=period, filters=filters)

    # ==================================================================
    # 16. GET /aggregations - List aggregations
    # ==================================================================

    @router.get("/aggregations")
    async def list_aggregations(
        period: Optional[str] = None,
        facility_id: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> List[Dict[str, Any]]:
        """List stored fleet aggregation results."""
        svc = _get_service()
        aggregations = list(svc._aggregations.values())

        if period is not None:
            aggregations = [
                a for a in aggregations
                if a.get("period") == period
            ]
        if facility_id is not None:
            aggregations = [
                a for a in aggregations
                if facility_id in a.get("by_facility", {})
            ]

        return aggregations[offset:offset + limit]

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

        return svc.run_uncertainty(input_data=body)

    # ==================================================================
    # 18. POST /compliance/check - Run compliance check
    # ==================================================================

    @router.post("/compliance/check")
    async def check_compliance(
        body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check regulatory compliance for calculations.

        Request body may include:
        - calculation_ids: Optional list of calculation IDs.
        - framework: Regulatory framework name (default: GHG_PROTOCOL).
        """
        svc = _get_service()
        framework = body.get("framework", "GHG_PROTOCOL")
        calculation_ids = body.get("calculation_ids")

        results = None
        if calculation_ids and isinstance(calculation_ids, list):
            results = [
                svc._calculations[cid]
                for cid in calculation_ids
                if cid in svc._calculations
            ]

        comp_result = svc.check_compliance(
            results=results,
            framework=framework,
        )

        # Cache compliance result
        comp_id = body.get("compliance_id", "")
        if comp_id:
            svc._compliance_records[comp_id] = comp_result

        return comp_result

    # ==================================================================
    # 19. GET /health - Health check
    # ==================================================================

    @router.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check for the mobile combustion service."""
        svc = _get_service()
        return svc.health_check()

    # ==================================================================
    # 20. GET /stats - Service statistics
    # ==================================================================

    @router.get("/stats")
    async def get_stats() -> Dict[str, Any]:
        """Get aggregate statistics for the mobile combustion service."""
        svc = _get_service()
        return svc.get_stats()


__all__ = ["router"]
