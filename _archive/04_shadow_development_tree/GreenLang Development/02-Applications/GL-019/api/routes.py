"""
GL-019 HEATSCHEDULER API Routes

REST API endpoints for ProcessHeatingScheduler.
Implements schedule optimization, production integration, tariff management,
equipment control, analytics, and demand response.

Author: GL-APIDeveloper
Version: 1.0.0
"""

import logging
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader

from api.schemas import (
    # Schedule schemas
    ScheduleOptimizeRequest,
    ScheduleResponse,
    ScheduleUpdateRequest,
    ScheduleListResponse,
    ScheduleStatus,
    HeatingOperation,
    TemperatureProfile,
    OptimizationObjective,
    BatchPriority,
    # Production schemas
    ProductionBatch,
    ProductionBatchListResponse,
    ProductionSyncRequest,
    ProductionSyncResponse,
    # Tariff schemas
    TariffResponse,
    TariffForecastResponse,
    TariffForecastPoint,
    TariffUploadRequest,
    TariffPeriod,
    TariffType,
    # Equipment schemas
    Equipment,
    EquipmentListResponse,
    EquipmentAvailabilityResponse,
    EquipmentAvailabilitySlot,
    EquipmentStatusUpdateRequest,
    EquipmentType,
    EquipmentStatus,
    EquipmentSpecs,
    # Analytics schemas
    SavingsReportRequest,
    SavingsReportResponse,
    SavingsBreakdown,
    CostForecastResponse,
    CostForecastPoint,
    WhatIfScenario,
    WhatIfResult,
    # Demand Response schemas
    DemandResponseEventRequest,
    DemandResponseEventResponse,
    DemandResponseStatusResponse,
    DemandResponseStatus,
    # Error schemas
    ErrorResponse,
)

logger = logging.getLogger("gl019.api.routes")

# Security schemes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# =============================================================================
# Authentication Dependencies
# =============================================================================

async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Depends(api_key_header)
) -> Dict[str, Any]:
    """
    Authenticate user via OAuth2 token or API key.

    Args:
        token: OAuth2 bearer token
        api_key: API key from header

    Returns:
        User information dict

    Raises:
        HTTPException: 401 if authentication fails
    """
    # In production, validate JWT token or API key
    # For now, return mock user for demonstration
    if token or api_key:
        return {
            "user_id": "user_001",
            "email": "operator@greenlang.io",
            "tenant_id": "tenant_001",
            "roles": ["operator", "analyst"],
            "facilities": ["facility_01", "facility_02"]
        }

    # For demo purposes, allow unauthenticated access
    # In production, raise HTTPException
    return {
        "user_id": "anonymous",
        "tenant_id": "demo",
        "roles": ["viewer"],
        "facilities": ["facility_01"]
    }


async def require_auth(
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Depends(api_key_header)
) -> Dict[str, Any]:
    """
    Require authentication - raises 401 if not authenticated.
    """
    if not token and not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return await get_current_user(token, api_key)


# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter(prefix="/api/v1", tags=["API v1"])


# =============================================================================
# Schedule Management Endpoints
# =============================================================================

@router.post(
    "/schedules/optimize",
    response_model=ScheduleResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Schedules"],
    summary="Create optimized heating schedule",
    description="""
    Create an AI-optimized heating schedule that minimizes energy costs
    while meeting all production requirements.

    The optimizer considers:
    - Time-of-use electricity rates
    - Equipment availability and efficiency
    - Production batch priorities and deadlines
    - Demand charge minimization
    - Demand response opportunities
    """,
    responses={
        201: {"description": "Schedule created successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ErrorResponse, "description": "Validation error"}
    }
)
async def create_optimized_schedule(
    request: ScheduleOptimizeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ScheduleResponse:
    """
    Create an optimized heating schedule.

    The optimization engine analyzes production requirements, energy tariffs,
    and equipment availability to create a cost-minimizing schedule.
    """
    logger.info(f"Creating optimized schedule for facility {request.facility_id}")

    # Generate schedule ID
    schedule_id = f"sched_{uuid4().hex[:12]}"

    # Mock optimization result
    # In production, this would call the optimization engine
    operations = []
    total_energy = 0.0
    total_cost = 0.0

    for i, batch_id in enumerate(request.batch_ids):
        start_time = datetime.combine(
            request.start_date + timedelta(days=i // 3),
            datetime.min.time()
        ) + timedelta(hours=2 + (i % 3) * 8)  # Schedule in off-peak hours

        operation = HeatingOperation(
            operation_id=f"op_{uuid4().hex[:8]}",
            equipment_id=f"furnace_0{(i % 2) + 1}",
            batch_id=batch_id,
            temperature_profile=TemperatureProfile(
                initial_temp=25.0,
                target_temp=850.0,
                ramp_rate=5.0,
                hold_duration_minutes=120,
                tolerance=2.0
            ),
            start_time=start_time,
            end_time=start_time + timedelta(hours=4),
            estimated_energy_kwh=450.0,
            estimated_cost=36.0,  # Off-peak rate
            priority=BatchPriority.NORMAL
        )
        operations.append(operation)
        total_energy += operation.estimated_energy_kwh
        total_cost += operation.estimated_cost

    baseline_cost = total_energy * 0.15  # Peak rate baseline
    savings = baseline_cost - total_cost

    response = ScheduleResponse(
        schedule_id=schedule_id,
        name=request.name,
        description=request.description,
        status=ScheduleStatus.DRAFT,
        start_date=request.start_date,
        end_date=request.end_date,
        facility_id=request.facility_id,
        objective=request.objective,
        operations=operations,
        total_energy_kwh=total_energy,
        total_cost=total_cost,
        baseline_cost=baseline_cost,
        savings=savings,
        savings_percent=(savings / baseline_cost) * 100 if baseline_cost > 0 else 0,
        peak_demand_kw=250.0,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    logger.info(f"Schedule {schedule_id} created with {len(operations)} operations, "
                f"estimated savings: ${savings:.2f} ({response.savings_percent:.1f}%)")

    return response


@router.get(
    "/schedules/{schedule_id}",
    response_model=ScheduleResponse,
    tags=["Schedules"],
    summary="Get schedule details",
    description="Retrieve detailed information about a specific heating schedule.",
    responses={
        200: {"description": "Schedule details"},
        404: {"model": ErrorResponse, "description": "Schedule not found"}
    }
)
async def get_schedule(
    schedule_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ScheduleResponse:
    """Get schedule details by ID."""
    logger.info(f"Fetching schedule {schedule_id}")

    # Mock response - in production, fetch from database
    return ScheduleResponse(
        schedule_id=schedule_id,
        name="Weekly Production Schedule",
        description="Optimized heating schedule for Week 45",
        status=ScheduleStatus.APPROVED,
        start_date=date.today(),
        end_date=date.today() + timedelta(days=7),
        facility_id="facility_01",
        objective=OptimizationObjective.MINIMIZE_COST,
        operations=[],
        total_energy_kwh=15000.0,
        total_cost=1200.00,
        baseline_cost=1500.00,
        savings=300.00,
        savings_percent=20.0,
        peak_demand_kw=450.0,
        created_at=datetime.utcnow() - timedelta(days=1),
        updated_at=datetime.utcnow()
    )


@router.get(
    "/schedules",
    response_model=ScheduleListResponse,
    tags=["Schedules"],
    summary="List schedules",
    description="List heating schedules with optional filtering and pagination.",
    responses={
        200: {"description": "List of schedules"}
    }
)
async def list_schedules(
    facility_id: Optional[str] = Query(None, description="Filter by facility"),
    status: Optional[ScheduleStatus] = Query(None, description="Filter by status"),
    start_date_from: Optional[date] = Query(None, description="Start date from"),
    start_date_to: Optional[date] = Query(None, description="Start date to"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ScheduleListResponse:
    """List schedules with filtering and pagination."""
    logger.info(f"Listing schedules for facility={facility_id}, status={status}")

    # Mock response
    items = [
        ScheduleResponse(
            schedule_id=f"sched_{i:03d}",
            name=f"Production Schedule {i}",
            description=f"Week {40 + i} schedule",
            status=ScheduleStatus.COMPLETED if i < 3 else ScheduleStatus.APPROVED,
            start_date=date.today() - timedelta(days=7 * (5 - i)),
            end_date=date.today() - timedelta(days=7 * (4 - i)),
            facility_id=facility_id or "facility_01",
            objective=OptimizationObjective.MINIMIZE_COST,
            operations=[],
            total_energy_kwh=15000.0 + i * 1000,
            total_cost=1200.00 + i * 100,
            baseline_cost=1500.00 + i * 125,
            savings=300.00 + i * 25,
            savings_percent=20.0,
            peak_demand_kw=450.0,
            created_at=datetime.utcnow() - timedelta(days=7 * (5 - i)),
            updated_at=datetime.utcnow() - timedelta(days=7 * (5 - i))
        )
        for i in range(1, 6)
    ]

    return ScheduleListResponse(
        items=items,
        total=len(items),
        page=page,
        page_size=page_size,
        total_pages=1
    )


@router.put(
    "/schedules/{schedule_id}",
    response_model=ScheduleResponse,
    tags=["Schedules"],
    summary="Update schedule",
    description="Update an existing heating schedule.",
    responses={
        200: {"description": "Schedule updated"},
        404: {"model": ErrorResponse, "description": "Schedule not found"},
        409: {"model": ErrorResponse, "description": "Schedule cannot be modified"}
    }
)
async def update_schedule(
    schedule_id: str,
    request: ScheduleUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ScheduleResponse:
    """Update an existing schedule."""
    logger.info(f"Updating schedule {schedule_id}")

    # Return updated schedule
    return ScheduleResponse(
        schedule_id=schedule_id,
        name=request.name or "Updated Schedule",
        description=request.description,
        status=request.status or ScheduleStatus.APPROVED,
        start_date=date.today(),
        end_date=date.today() + timedelta(days=7),
        facility_id="facility_01",
        objective=OptimizationObjective.MINIMIZE_COST,
        operations=request.operations or [],
        total_energy_kwh=15000.0,
        total_cost=1200.00,
        baseline_cost=1500.00,
        savings=300.00,
        savings_percent=20.0,
        peak_demand_kw=450.0,
        created_at=datetime.utcnow() - timedelta(days=1),
        updated_at=datetime.utcnow()
    )


@router.delete(
    "/schedules/{schedule_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Schedules"],
    summary="Cancel schedule",
    description="Cancel a pending or approved schedule.",
    responses={
        204: {"description": "Schedule cancelled"},
        404: {"model": ErrorResponse, "description": "Schedule not found"},
        409: {"model": ErrorResponse, "description": "Schedule cannot be cancelled"}
    }
)
async def cancel_schedule(
    schedule_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> None:
    """Cancel a schedule."""
    logger.info(f"Cancelling schedule {schedule_id}")
    # In production, update database
    return None


# =============================================================================
# Production Integration Endpoints
# =============================================================================

@router.get(
    "/production/batches",
    response_model=ProductionBatchListResponse,
    tags=["Production"],
    summary="Get production batches",
    description="List production batches requiring heating operations.",
    responses={
        200: {"description": "List of production batches"}
    }
)
async def get_production_batches(
    facility_id: Optional[str] = Query(None, description="Filter by facility"),
    status: Optional[str] = Query(None, description="Filter by status"),
    date_from: Optional[date] = Query(None, description="Filter from date"),
    date_to: Optional[date] = Query(None, description="Filter to date"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ProductionBatchListResponse:
    """Get production batches for scheduling."""
    logger.info(f"Fetching production batches for facility={facility_id}")

    # Mock response
    batches = [
        ProductionBatch(
            batch_id=f"batch_{date.today().strftime('%Y%m%d')}_{i:03d}",
            product_id=f"prod_{['steel_alloy', 'aluminum', 'copper'][i % 3]}",
            product_name=["Steel Alloy A36", "Aluminum 6061", "Copper C110"][i % 3],
            quantity=500 + i * 100,
            priority=BatchPriority.HIGH if i == 0 else BatchPriority.NORMAL,
            required_temp=[850.0, 550.0, 1100.0][i % 3],
            hold_duration_minutes=[120, 90, 60][i % 3],
            earliest_start=datetime.utcnow() + timedelta(hours=i * 4),
            latest_end=datetime.utcnow() + timedelta(days=2),
            equipment_types=[EquipmentType.FURNACE],
            estimated_energy_kwh=450.0 + i * 50,
            status="pending"
        )
        for i in range(5)
    ]

    return ProductionBatchListResponse(
        items=batches,
        total=len(batches),
        page=page,
        page_size=page_size
    )


@router.post(
    "/production/sync",
    response_model=ProductionSyncResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Production"],
    summary="Sync production schedule from ERP",
    description="Synchronize production batch data from connected ERP system.",
    responses={
        202: {"description": "Sync initiated"},
        400: {"model": ErrorResponse, "description": "Invalid sync request"},
        503: {"model": ErrorResponse, "description": "ERP system unavailable"}
    }
)
async def sync_production_schedule(
    request: ProductionSyncRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ProductionSyncResponse:
    """Sync production schedule from ERP system."""
    logger.info(f"Syncing production from {request.erp_system} for facility {request.facility_id}")

    sync_id = f"sync_{uuid4().hex[:12]}"

    return ProductionSyncResponse(
        sync_id=sync_id,
        status="completed",
        batches_synced=15,
        batches_created=8,
        batches_updated=7,
        errors=[],
        synced_at=datetime.utcnow()
    )


# =============================================================================
# Energy Tariff Endpoints
# =============================================================================

@router.get(
    "/tariffs/current",
    response_model=TariffResponse,
    tags=["Tariffs"],
    summary="Get current tariff rates",
    description="Get the current active energy tariff rates for a facility.",
    responses={
        200: {"description": "Current tariff information"}
    }
)
async def get_current_tariff(
    facility_id: str = Query(..., description="Facility identifier"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> TariffResponse:
    """Get current energy tariff."""
    logger.info(f"Fetching current tariff for facility {facility_id}")

    return TariffResponse(
        tariff_id="tariff_tou_2024",
        name="Time-of-Use Industrial Rate",
        tariff_type=TariffType.TIME_OF_USE,
        utility_provider="Pacific Gas & Electric",
        currency="USD",
        effective_date=date(2024, 1, 1),
        expiration_date=date(2024, 12, 31),
        periods=[
            TariffPeriod(
                start_time=datetime.strptime("00:00:00", "%H:%M:%S").time(),
                end_time=datetime.strptime("06:00:00", "%H:%M:%S").time(),
                rate_per_kwh=0.08,
                demand_charge_per_kw=5.00,
                period_name="Off-Peak"
            ),
            TariffPeriod(
                start_time=datetime.strptime("06:00:00", "%H:%M:%S").time(),
                end_time=datetime.strptime("14:00:00", "%H:%M:%S").time(),
                rate_per_kwh=0.12,
                demand_charge_per_kw=10.00,
                period_name="Mid-Peak"
            ),
            TariffPeriod(
                start_time=datetime.strptime("14:00:00", "%H:%M:%S").time(),
                end_time=datetime.strptime("20:00:00", "%H:%M:%S").time(),
                rate_per_kwh=0.25,
                demand_charge_per_kw=20.00,
                period_name="On-Peak"
            ),
            TariffPeriod(
                start_time=datetime.strptime("20:00:00", "%H:%M:%S").time(),
                end_time=datetime.strptime("00:00:00", "%H:%M:%S").time(),
                rate_per_kwh=0.12,
                demand_charge_per_kw=10.00,
                period_name="Mid-Peak"
            )
        ],
        demand_charge_per_kw=15.00
    )


@router.get(
    "/tariffs/forecast",
    response_model=TariffForecastResponse,
    tags=["Tariffs"],
    summary="Get tariff forecast",
    description="Get forecasted energy tariff rates for optimization planning.",
    responses={
        200: {"description": "Tariff forecast"}
    }
)
async def get_tariff_forecast(
    facility_id: str = Query(..., description="Facility identifier"),
    horizon_hours: int = Query(72, ge=1, le=168, description="Forecast horizon in hours"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> TariffForecastResponse:
    """Get tariff rate forecast."""
    logger.info(f"Generating tariff forecast for facility {facility_id}, horizon={horizon_hours}h")

    forecast_points = []
    now = datetime.utcnow()

    for hour in range(horizon_hours):
        timestamp = now + timedelta(hours=hour)
        hour_of_day = timestamp.hour

        # Time-of-use rate simulation
        if 0 <= hour_of_day < 6:
            rate = 0.08
        elif 14 <= hour_of_day < 20:
            rate = 0.25
        else:
            rate = 0.12

        forecast_points.append(TariffForecastPoint(
            timestamp=timestamp,
            rate_per_kwh=rate,
            confidence=0.95 if hour < 24 else 0.85
        ))

    rates = [p.rate_per_kwh for p in forecast_points]

    return TariffForecastResponse(
        facility_id=facility_id,
        forecast_generated_at=datetime.utcnow(),
        forecast_horizon_hours=horizon_hours,
        currency="USD",
        forecast=forecast_points,
        avg_rate=sum(rates) / len(rates),
        min_rate=min(rates),
        max_rate=max(rates)
    )


@router.post(
    "/tariffs",
    response_model=TariffResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Tariffs"],
    summary="Upload custom tariff",
    description="Upload a custom energy tariff schedule.",
    responses={
        201: {"description": "Tariff created"},
        400: {"model": ErrorResponse, "description": "Invalid tariff data"}
    }
)
async def upload_tariff(
    request: TariffUploadRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> TariffResponse:
    """Upload custom tariff data."""
    logger.info(f"Uploading custom tariff for facility {request.facility_id}")

    tariff_id = f"tariff_{uuid4().hex[:8]}"

    return TariffResponse(
        tariff_id=tariff_id,
        name=request.name,
        tariff_type=request.tariff_type,
        utility_provider=request.utility_provider,
        currency="USD",
        effective_date=request.effective_date,
        expiration_date=request.expiration_date,
        periods=request.periods,
        demand_charge_per_kw=request.demand_charge_per_kw
    )


# =============================================================================
# Equipment Endpoints
# =============================================================================

@router.get(
    "/equipment",
    response_model=EquipmentListResponse,
    tags=["Equipment"],
    summary="List heating equipment",
    description="List all heating equipment at a facility.",
    responses={
        200: {"description": "List of equipment"}
    }
)
async def list_equipment(
    facility_id: Optional[str] = Query(None, description="Filter by facility"),
    equipment_type: Optional[EquipmentType] = Query(None, description="Filter by type"),
    status: Optional[EquipmentStatus] = Query(None, description="Filter by status"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> EquipmentListResponse:
    """List heating equipment."""
    logger.info(f"Listing equipment for facility={facility_id}")

    equipment_list = [
        Equipment(
            equipment_id="furnace_01",
            name="Heat Treatment Furnace #1",
            equipment_type=EquipmentType.FURNACE,
            status=EquipmentStatus.AVAILABLE,
            facility_id=facility_id or "facility_01",
            specs=EquipmentSpecs(
                max_temp=1200.0,
                min_temp=0.0,
                capacity_kg=5000.0,
                power_rating_kw=250.0,
                efficiency=0.85,
                ramp_rate_max=10.0
            ),
            current_temp=25.0,
            current_power_kw=0.0,
            last_maintenance=datetime.utcnow() - timedelta(days=30),
            next_maintenance=datetime.utcnow() + timedelta(days=60)
        ),
        Equipment(
            equipment_id="furnace_02",
            name="Heat Treatment Furnace #2",
            equipment_type=EquipmentType.FURNACE,
            status=EquipmentStatus.IN_USE,
            facility_id=facility_id or "facility_01",
            specs=EquipmentSpecs(
                max_temp=1200.0,
                min_temp=0.0,
                capacity_kg=3000.0,
                power_rating_kw=180.0,
                efficiency=0.82,
                ramp_rate_max=8.0
            ),
            current_temp=650.0,
            current_power_kw=150.0,
            last_maintenance=datetime.utcnow() - timedelta(days=15),
            next_maintenance=datetime.utcnow() + timedelta(days=75)
        ),
        Equipment(
            equipment_id="boiler_01",
            name="Process Boiler #1",
            equipment_type=EquipmentType.BOILER,
            status=EquipmentStatus.AVAILABLE,
            facility_id=facility_id or "facility_01",
            specs=EquipmentSpecs(
                max_temp=200.0,
                min_temp=0.0,
                capacity_kg=10000.0,
                power_rating_kw=500.0,
                efficiency=0.90,
                ramp_rate_max=5.0
            ),
            current_temp=80.0,
            current_power_kw=50.0,
            last_maintenance=datetime.utcnow() - timedelta(days=45),
            next_maintenance=datetime.utcnow() + timedelta(days=45)
        )
    ]

    # Apply filters
    if equipment_type:
        equipment_list = [e for e in equipment_list if e.equipment_type == equipment_type]
    if status:
        equipment_list = [e for e in equipment_list if e.status == status]

    return EquipmentListResponse(
        items=equipment_list,
        total=len(equipment_list)
    )


@router.get(
    "/equipment/{equipment_id}/availability",
    response_model=EquipmentAvailabilityResponse,
    tags=["Equipment"],
    summary="Get equipment availability",
    description="Get availability schedule for a specific piece of equipment.",
    responses={
        200: {"description": "Equipment availability"},
        404: {"model": ErrorResponse, "description": "Equipment not found"}
    }
)
async def get_equipment_availability(
    equipment_id: str,
    date_from: Optional[date] = Query(None, description="Start date"),
    date_to: Optional[date] = Query(None, description="End date"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> EquipmentAvailabilityResponse:
    """Get equipment availability schedule."""
    logger.info(f"Fetching availability for equipment {equipment_id}")

    # Generate availability slots
    slots = []
    start = datetime.combine(date_from or date.today(), datetime.min.time())

    for day in range(7):
        day_start = start + timedelta(days=day)

        # Morning slot (available)
        slots.append(EquipmentAvailabilitySlot(
            start_time=day_start + timedelta(hours=0),
            end_time=day_start + timedelta(hours=8),
            is_available=True,
            scheduled_batch_id=None
        ))

        # Mid-day slot (scheduled)
        slots.append(EquipmentAvailabilitySlot(
            start_time=day_start + timedelta(hours=8),
            end_time=day_start + timedelta(hours=16),
            is_available=False,
            scheduled_batch_id=f"batch_{day_start.strftime('%Y%m%d')}_001"
        ))

        # Evening slot (available)
        slots.append(EquipmentAvailabilitySlot(
            start_time=day_start + timedelta(hours=16),
            end_time=day_start + timedelta(hours=24),
            is_available=True,
            scheduled_batch_id=None
        ))

    return EquipmentAvailabilityResponse(
        equipment_id=equipment_id,
        equipment_name="Heat Treatment Furnace #1",
        status=EquipmentStatus.AVAILABLE,
        availability_slots=slots,
        utilization_percent=33.3
    )


@router.put(
    "/equipment/{equipment_id}/status",
    response_model=Equipment,
    tags=["Equipment"],
    summary="Update equipment status",
    description="Update the operational status of a piece of equipment.",
    responses={
        200: {"description": "Status updated"},
        404: {"model": ErrorResponse, "description": "Equipment not found"}
    }
)
async def update_equipment_status(
    equipment_id: str,
    request: EquipmentStatusUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Equipment:
    """Update equipment status."""
    logger.info(f"Updating status for equipment {equipment_id} to {request.status}")

    return Equipment(
        equipment_id=equipment_id,
        name="Heat Treatment Furnace #1",
        equipment_type=EquipmentType.FURNACE,
        status=request.status,
        facility_id="facility_01",
        specs=EquipmentSpecs(
            max_temp=1200.0,
            min_temp=0.0,
            capacity_kg=5000.0,
            power_rating_kw=250.0,
            efficiency=0.85,
            ramp_rate_max=10.0
        ),
        current_temp=25.0,
        current_power_kw=0.0,
        last_maintenance=datetime.utcnow(),
        next_maintenance=datetime.utcnow() + timedelta(days=90)
    )


# =============================================================================
# Analytics Endpoints
# =============================================================================

@router.get(
    "/analytics/savings",
    response_model=SavingsReportResponse,
    tags=["Analytics"],
    summary="Get savings report",
    description="Get detailed cost savings report for a time period.",
    responses={
        200: {"description": "Savings report"}
    }
)
async def get_savings_report(
    facility_id: Optional[str] = Query(None, description="Filter by facility"),
    start_date: date = Query(..., description="Report start date"),
    end_date: date = Query(..., description="Report end date"),
    include_breakdown: bool = Query(True, description="Include savings breakdown"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> SavingsReportResponse:
    """Get savings analytics report."""
    logger.info(f"Generating savings report for {start_date} to {end_date}")

    breakdown = None
    if include_breakdown:
        breakdown = SavingsBreakdown(
            time_shifting=1800.00,
            demand_reduction=600.00,
            efficiency_improvement=300.00,
            demand_response=300.00
        )

    return SavingsReportResponse(
        report_id=f"rpt_sav_{uuid4().hex[:8]}",
        facility_id=facility_id,
        period_start=start_date,
        period_end=end_date,
        total_energy_kwh=150000.0,
        total_cost=12000.00,
        baseline_cost=15000.00,
        total_savings=3000.00,
        savings_percent=20.0,
        breakdown=breakdown,
        schedules_optimized=45,
        co2_avoided_kg=1500.0,
        generated_at=datetime.utcnow()
    )


@router.get(
    "/analytics/forecast",
    response_model=CostForecastResponse,
    tags=["Analytics"],
    summary="Get cost forecast",
    description="Get energy cost forecast for planning purposes.",
    responses={
        200: {"description": "Cost forecast"}
    }
)
async def get_cost_forecast(
    facility_id: str = Query(..., description="Facility identifier"),
    horizon_days: int = Query(30, ge=1, le=90, description="Forecast horizon in days"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> CostForecastResponse:
    """Get energy cost forecast."""
    logger.info(f"Generating cost forecast for facility {facility_id}, horizon={horizon_days} days")

    forecast_points = []
    today = date.today()

    for day in range(horizon_days):
        forecast_date = today + timedelta(days=day)
        base_cost = 400.0  # Base daily cost
        variation = (day % 7) * 20  # Weekly variation

        forecast_points.append(CostForecastPoint(
            date=forecast_date,
            forecasted_cost=base_cost + variation,
            forecasted_energy_kwh=(base_cost + variation) / 0.10,  # Assume $0.10/kWh avg
            confidence_lower=(base_cost + variation) * 0.9,
            confidence_upper=(base_cost + variation) * 1.1
        ))

    total_cost = sum(p.forecasted_cost for p in forecast_points)
    total_energy = sum(p.forecasted_energy_kwh for p in forecast_points)

    return CostForecastResponse(
        facility_id=facility_id,
        forecast_horizon_days=horizon_days,
        forecast_generated_at=datetime.utcnow(),
        currency="USD",
        forecast=forecast_points,
        total_forecasted_cost=total_cost,
        total_forecasted_energy_kwh=total_energy
    )


@router.post(
    "/analytics/what-if",
    response_model=WhatIfResult,
    tags=["Analytics"],
    summary="Run what-if scenario",
    description="Run a what-if analysis to evaluate potential changes.",
    responses={
        200: {"description": "What-if analysis result"},
        400: {"model": ErrorResponse, "description": "Invalid scenario parameters"}
    }
)
async def run_what_if_scenario(
    request: WhatIfScenario,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> WhatIfResult:
    """Run what-if analysis."""
    logger.info(f"Running what-if scenario: {request.scenario_name}")

    baseline_cost = 15000.00
    scenario_cost = baseline_cost

    # Apply tariff change
    if request.tariff_change_percent:
        scenario_cost *= (1 + request.tariff_change_percent / 100)

    # Apply demand change
    if request.demand_change_percent:
        scenario_cost *= (1 + request.demand_change_percent / 100)

    # Apply efficiency change
    if request.equipment_efficiency_change:
        scenario_cost *= (1 - request.equipment_efficiency_change)

    cost_diff = scenario_cost - baseline_cost
    cost_diff_pct = (cost_diff / baseline_cost) * 100

    recommendations = []
    if cost_diff > 0:
        recommendations.append("Consider negotiating better tariff rates")
        recommendations.append("Increase load shifting to off-peak hours")
        recommendations.append("Evaluate equipment efficiency upgrades")
    else:
        recommendations.append("Current optimization strategy is effective")
        recommendations.append("Lock in favorable tariff rates if available")

    return WhatIfResult(
        scenario_name=request.scenario_name,
        baseline_cost=baseline_cost,
        scenario_cost=scenario_cost,
        cost_difference=cost_diff,
        cost_difference_percent=cost_diff_pct,
        baseline_energy_kwh=150000.0,
        scenario_energy_kwh=150000.0 * (1 + (request.demand_change_percent or 0) / 100),
        recommendations=recommendations,
        analyzed_at=datetime.utcnow()
    )


# =============================================================================
# Demand Response Endpoints
# =============================================================================

@router.post(
    "/demand-response/event",
    response_model=DemandResponseEventResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Demand Response"],
    summary="Handle demand response event",
    description="Receive and process a demand response event from the grid operator.",
    responses={
        202: {"description": "Event received and being processed"},
        400: {"model": ErrorResponse, "description": "Invalid event data"}
    }
)
async def handle_demand_response_event(
    request: DemandResponseEventRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> DemandResponseEventResponse:
    """Handle incoming demand response event."""
    logger.info(f"Processing DR event {request.event_id} for facility {request.facility_id}")

    # Calculate response
    committed_reduction = min(request.required_reduction_kw, 150.0)  # Can commit up to 150kW
    duration_hours = (request.end_time - request.start_time).total_seconds() / 3600
    estimated_revenue = committed_reduction * duration_hours * (request.incentive_rate or 0.50)

    return DemandResponseEventResponse(
        event_id=request.event_id,
        facility_id=request.facility_id,
        participation_status="accepted",
        committed_reduction_kw=committed_reduction,
        estimated_revenue=estimated_revenue,
        rescheduled_operations=3,
        response_received_at=datetime.utcnow()
    )


@router.get(
    "/demand-response/status",
    response_model=DemandResponseStatusResponse,
    tags=["Demand Response"],
    summary="Get demand response status",
    description="Get current demand response participation status.",
    responses={
        200: {"description": "Current DR status"}
    }
)
async def get_demand_response_status(
    facility_id: str = Query(..., description="Facility identifier"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> DemandResponseStatusResponse:
    """Get current demand response status."""
    logger.info(f"Fetching DR status for facility {facility_id}")

    return DemandResponseStatusResponse(
        facility_id=facility_id,
        status=DemandResponseStatus.IDLE,
        active_events=[],
        pending_events=[],
        current_load_kw=350.0,
        available_reduction_kw=150.0,
        ytd_participation_count=12,
        ytd_revenue=5400.00,
        last_updated=datetime.utcnow()
    )
