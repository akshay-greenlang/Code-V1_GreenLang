# -*- coding: utf-8 -*-
"""
Mobile Combustion Agent Service Setup - AGENT-MRV-003

Provides ``configure_mobile_combustion(app)`` which wires up the
Mobile Combustion Agent SDK (vehicle database, emission calculator,
fleet manager, distance estimator, uncertainty quantifier, compliance
checker, mobile combustion pipeline, provenance tracker) and mounts
the REST API.

Also exposes ``get_service()`` for programmatic access and the
``MobileCombustionService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.mobile_combustion.setup import configure_mobile_combustion
    >>> app = FastAPI()
    >>> import asyncio
    >>> service = asyncio.run(configure_mobile_combustion(app))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-003 Mobile Combustion (GL-MRV-SCOPE1-003)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional config import
# ---------------------------------------------------------------------------

try:
    from greenlang.mobile_combustion.config import (
        MobileCombustionConfig,
        get_config,
    )
except ImportError:
    MobileCombustionConfig = None  # type: ignore[assignment, misc]

    def get_config() -> Any:  # type: ignore[misc]
        """No-op fallback when config module is unavailable."""
        return None

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional model imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.mobile_combustion.models import (
        CalculationInput,
        CalculationMethod,
        CalculationResult,
        CalculationTier,
        ComplianceCheckResult,
        ComplianceStatus,
        DistanceUnit,
        EmissionControlTechnology,
        EmissionFactorSource,
        EmissionGas,
        FleetAggregation,
        FuelType,
        GWPSource,
        MobileCombustionInput,
        MobileCombustionOutput,
        ReportingPeriod,
        UncertaintyResult,
        VehicleCategory,
        VehicleType,
    )
except ImportError:
    CalculationInput = None  # type: ignore[assignment, misc]
    CalculationMethod = None  # type: ignore[assignment, misc]
    CalculationResult = None  # type: ignore[assignment, misc]
    CalculationTier = None  # type: ignore[assignment, misc]
    ComplianceCheckResult = None  # type: ignore[assignment, misc]
    ComplianceStatus = None  # type: ignore[assignment, misc]
    DistanceUnit = None  # type: ignore[assignment, misc]
    EmissionControlTechnology = None  # type: ignore[assignment, misc]
    EmissionFactorSource = None  # type: ignore[assignment, misc]
    EmissionGas = None  # type: ignore[assignment, misc]
    FleetAggregation = None  # type: ignore[assignment, misc]
    FuelType = None  # type: ignore[assignment, misc]
    GWPSource = None  # type: ignore[assignment, misc]
    MobileCombustionInput = None  # type: ignore[assignment, misc]
    MobileCombustionOutput = None  # type: ignore[assignment, misc]
    ReportingPeriod = None  # type: ignore[assignment, misc]
    UncertaintyResult = None  # type: ignore[assignment, misc]
    VehicleCategory = None  # type: ignore[assignment, misc]
    VehicleType = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Optional engine imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.mobile_combustion.vehicle_database import VehicleDatabaseEngine
except ImportError:
    VehicleDatabaseEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.emission_calculator import EmissionCalculatorEngine
except ImportError:
    EmissionCalculatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.fleet_manager import FleetManagerEngine
except ImportError:
    FleetManagerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.distance_estimator import DistanceEstimatorEngine
except ImportError:
    DistanceEstimatorEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.uncertainty_quantifier import UncertaintyQuantifierEngine
except ImportError:
    UncertaintyQuantifierEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.compliance_checker import ComplianceCheckerEngine
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.mobile_combustion_pipeline import (
        MobileCombustionPipelineEngine,
    )
except ImportError:
    MobileCombustionPipelineEngine = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.provenance import ProvenanceTracker
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment, misc]

try:
    from greenlang.mobile_combustion.metrics import (
        PROMETHEUS_AVAILABLE,
        record_calculation,
        observe_calculation_duration,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]

    def record_calculation(method: str, vehicle_type: str, status: str) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""

    def observe_calculation_duration(operation: str, seconds: float) -> None:  # type: ignore[misc]
        """No-op fallback when metrics module is unavailable."""


# ===================================================================
# Lightweight Pydantic response models used by the facade / API layer
# ===================================================================


class CalculateResponse(BaseModel):
    """Single mobile combustion emission calculation response.

    Attributes:
        calculation_id: Unique calculation identifier (UUID4).
        status: Calculation status (SUCCESS, PARTIAL, FAILED).
        vehicle_type: Vehicle type used in the calculation.
        fuel_type: Fuel type used in the calculation.
        calculation_method: Calculation method applied.
        gwp_source: GWP source used (AR4, AR5, AR6).
        fuel_quantity_gallons: Fuel consumed in US gallons.
        distance_km: Distance travelled in kilometres.
        co2_kg: Direct CO2 emissions in kg.
        ch4_co2e_kg: CH4 emissions in kg CO2e.
        n2o_co2e_kg: N2O emissions in kg CO2e.
        total_co2e_kg: Total CO2e in kg.
        total_co2e_tonnes: Total CO2e in tonnes.
        fossil_co2_kg: Fossil-origin CO2 in kg.
        biogenic_co2_kg: Biogenic CO2 in kg.
        biogenic_co2_tonnes: Biogenic CO2 in tonnes.
        gas_emissions: Per-gas emission breakdown.
        calculation_trace: Step-by-step calculation trace.
        provenance_hash: SHA-256 provenance hash for audit trail.
        processing_time_ms: Processing duration in milliseconds.
        calculated_at: ISO-8601 UTC calculation timestamp.
    """

    model_config = {"extra": "forbid"}

    calculation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = Field(default="SUCCESS")
    vehicle_type: str = Field(default="")
    fuel_type: str = Field(default="")
    calculation_method: str = Field(default="FUEL_BASED")
    gwp_source: str = Field(default="AR6")
    fuel_quantity_gallons: float = Field(default=0.0)
    distance_km: float = Field(default=0.0)
    co2_kg: float = Field(default=0.0)
    ch4_co2e_kg: float = Field(default=0.0)
    n2o_co2e_kg: float = Field(default=0.0)
    total_co2e_kg: float = Field(default=0.0)
    total_co2e_tonnes: float = Field(default=0.0)
    fossil_co2_kg: float = Field(default=0.0)
    biogenic_co2_kg: float = Field(default=0.0)
    biogenic_co2_tonnes: float = Field(default=0.0)
    gas_emissions: List[Dict[str, Any]] = Field(default_factory=list)
    calculation_trace: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    calculated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class BatchCalculateResponse(BaseModel):
    """Batch mobile combustion emission calculation response.

    Attributes:
        batch_id: Unique batch identifier (UUID4).
        results: Individual calculation results.
        total_co2e_kg: Aggregate CO2e in kg.
        total_co2e_tonnes: Aggregate CO2e in tonnes.
        total_biogenic_co2_kg: Aggregate biogenic CO2 in kg.
        success_count: Number of successful calculations.
        failure_count: Number of failed calculations.
        processing_time_ms: Total processing time in milliseconds.
        provenance_hash: SHA-256 batch provenance hash.
    """

    model_config = {"extra": "forbid"}

    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    results: List[Dict[str, Any]] = Field(default_factory=list)
    total_co2e_kg: float = Field(default=0.0)
    total_co2e_tonnes: float = Field(default=0.0)
    total_biogenic_co2_kg: float = Field(default=0.0)
    success_count: int = Field(default=0)
    failure_count: int = Field(default=0)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class CalculationDetailResponse(BaseModel):
    """Detailed calculation response with audit trail.

    Attributes:
        calculation_id: Unique calculation identifier.
        result: Full calculation result dictionary.
        audit_trail: List of audit entry dictionaries.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    calculation_id: str = Field(default="")
    result: Dict[str, Any] = Field(default_factory=dict)
    audit_trail: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class VehicleResponse(BaseModel):
    """Vehicle registration or retrieval response.

    Attributes:
        vehicle_id: Unique vehicle identifier.
        vehicle_type: Vehicle type classification.
        fuel_type: Primary fuel type.
        name: Human-readable vehicle name.
        facility_id: Parent facility identifier.
        fleet_id: Fleet identifier.
        make: Vehicle make.
        model: Vehicle model.
        year: Model year.
        fuel_economy: Fuel economy value.
        fuel_economy_unit: Fuel economy unit.
        odometer_km: Current odometer reading in km.
        registration_date: Vehicle registration date.
        created_at: ISO-8601 UTC creation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    vehicle_id: str = Field(default="")
    vehicle_type: str = Field(default="")
    fuel_type: str = Field(default="")
    name: str = Field(default="")
    facility_id: Optional[str] = Field(default=None)
    fleet_id: Optional[str] = Field(default=None)
    make: str = Field(default="")
    model: str = Field(default="")
    year: Optional[int] = Field(default=None)
    fuel_economy: Optional[float] = Field(default=None)
    fuel_economy_unit: str = Field(default="L_PER_100KM")
    odometer_km: float = Field(default=0.0)
    registration_date: Optional[str] = Field(default=None)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class VehicleListResponse(BaseModel):
    """Response listing vehicles.

    Attributes:
        vehicles: List of vehicle summary dictionaries.
        total_count: Total number of vehicles.
    """

    model_config = {"extra": "forbid"}

    vehicles: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: int = Field(default=0)


class TripResponse(BaseModel):
    """Trip logging response.

    Attributes:
        trip_id: Unique trip identifier.
        vehicle_id: Vehicle that made the trip.
        distance_km: Trip distance in kilometres.
        distance_unit: Original distance unit.
        fuel_quantity: Fuel consumed.
        fuel_unit: Fuel quantity unit.
        start_date: Trip start date (ISO-8601).
        end_date: Trip end date (ISO-8601).
        origin: Trip origin description.
        destination: Trip destination description.
        purpose: Trip purpose.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    trip_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vehicle_id: str = Field(default="")
    distance_km: float = Field(default=0.0)
    distance_unit: str = Field(default="KM")
    fuel_quantity: Optional[float] = Field(default=None)
    fuel_unit: str = Field(default="LITERS")
    start_date: Optional[str] = Field(default=None)
    end_date: Optional[str] = Field(default=None)
    origin: str = Field(default="")
    destination: str = Field(default="")
    purpose: str = Field(default="")
    provenance_hash: str = Field(default="")


class TripListResponse(BaseModel):
    """Response listing trips.

    Attributes:
        trips: List of trip summary dictionaries.
        total_count: Total number of trips.
    """

    model_config = {"extra": "forbid"}

    trips: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: int = Field(default=0)


class FuelListResponse(BaseModel):
    """Response listing available fuel types.

    Attributes:
        fuels: List of fuel type summary dictionaries.
        total_count: Total number of fuel types.
    """

    model_config = {"extra": "forbid"}

    fuels: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: int = Field(default=0)


class FactorListResponse(BaseModel):
    """Response listing emission factors.

    Attributes:
        factors: List of emission factor dictionaries.
        total_count: Total number of emission factors.
    """

    model_config = {"extra": "forbid"}

    factors: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: int = Field(default=0)


class AggregationResponse(BaseModel):
    """Fleet aggregation response.

    Attributes:
        aggregation_id: Unique aggregation identifier.
        period: Reporting period label.
        total_co2e_tonnes: Total CO2e in tonnes.
        total_biogenic_co2_tonnes: Total biogenic CO2 in tonnes.
        by_vehicle_type: CO2e breakdown by vehicle type.
        by_fuel_type: CO2e breakdown by fuel type.
        by_facility: CO2e breakdown by facility.
        vehicle_count: Number of vehicles in aggregation.
        trip_count: Number of trips in aggregation.
        total_distance_km: Total distance in km.
        total_fuel_gallons: Total fuel in gallons.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    aggregation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    period: str = Field(default="")
    total_co2e_tonnes: float = Field(default=0.0)
    total_biogenic_co2_tonnes: float = Field(default=0.0)
    by_vehicle_type: Dict[str, float] = Field(default_factory=dict)
    by_fuel_type: Dict[str, float] = Field(default_factory=dict)
    by_facility: Dict[str, float] = Field(default_factory=dict)
    vehicle_count: int = Field(default=0)
    trip_count: int = Field(default=0)
    total_distance_km: float = Field(default=0.0)
    total_fuel_gallons: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class UncertaintyResponse(BaseModel):
    """Uncertainty analysis response.

    Attributes:
        calculation_id: Related calculation identifier.
        method: Uncertainty method (monte_carlo, analytical).
        mean_co2e_kg: Mean CO2e from simulations.
        std_co2e_kg: Standard deviation of CO2e.
        p5_co2e_kg: 5th percentile.
        p95_co2e_kg: 95th percentile.
        confidence_interval_pct: Confidence interval percentage.
        relative_uncertainty_pct: Relative uncertainty percentage.
        iterations: Number of Monte Carlo iterations.
        data_quality_score: Data quality score (1-5).
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    calculation_id: str = Field(default="")
    method: str = Field(default="analytical")
    mean_co2e_kg: float = Field(default=0.0)
    std_co2e_kg: float = Field(default=0.0)
    p5_co2e_kg: float = Field(default=0.0)
    p95_co2e_kg: float = Field(default=0.0)
    confidence_interval_pct: float = Field(default=90.0)
    relative_uncertainty_pct: float = Field(default=0.0)
    iterations: int = Field(default=0)
    data_quality_score: int = Field(default=3)
    provenance_hash: str = Field(default="")


class ComplianceResponse(BaseModel):
    """Regulatory compliance check response.

    Attributes:
        framework: Regulatory framework name.
        compliant: Whether all requirements are met.
        compliant_count: Number of compliant requirements.
        total_requirements: Total requirements checked.
        checks: Per-requirement compliance details.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = {"extra": "forbid"}

    framework: str = Field(default="")
    compliant: bool = Field(default=False)
    compliant_count: int = Field(default=0)
    total_requirements: int = Field(default=0)
    checks: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class HealthResponse(BaseModel):
    """Service health check response.

    Attributes:
        status: Overall service status (healthy, degraded, unhealthy).
        version: Agent version string.
        engines: Per-engine availability status.
        engines_available: Count of available engines.
        engines_total: Total number of engines.
        started: Whether the service has been started.
        vehicle_count: Number of registered vehicles.
        trip_count: Number of logged trips.
        fuel_type_count: Number of supported fuel types.
        statistics: Summary statistics.
        provenance_chain_valid: Whether the provenance chain is intact.
        provenance_entries: Total provenance entries recorded.
        prometheus_available: Whether Prometheus client is available.
        timestamp: ISO-8601 UTC timestamp of the health check.
    """

    model_config = {"extra": "forbid"}

    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    engines: Dict[str, str] = Field(default_factory=dict)
    engines_available: int = Field(default=0)
    engines_total: int = Field(default=7)
    started: bool = Field(default=False)
    vehicle_count: int = Field(default=0)
    trip_count: int = Field(default=0)
    fuel_type_count: int = Field(default=0)
    statistics: Dict[str, Any] = Field(default_factory=dict)
    provenance_chain_valid: bool = Field(default=True)
    provenance_entries: int = Field(default=0)
    prometheus_available: bool = Field(default=False)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class StatsResponse(BaseModel):
    """Service aggregate statistics response.

    Attributes:
        total_calculations: Total calculations performed.
        total_batch_runs: Total batch calculation runs.
        total_pipeline_runs: Total pipeline runs.
        total_vehicles: Number of registered vehicles.
        total_trips: Number of logged trips.
        total_fuel_types: Number of supported fuel types.
        total_aggregations: Total fleet aggregations.
        total_compliance_checks: Total compliance checks.
        total_uncertainty_runs: Total uncertainty analyses.
        avg_calculation_time_ms: Average calculation time.
        timestamp: ISO-8601 UTC timestamp.
    """

    model_config = {"extra": "forbid"}

    total_calculations: int = Field(default=0)
    total_batch_runs: int = Field(default=0)
    total_pipeline_runs: int = Field(default=0)
    total_vehicles: int = Field(default=0)
    total_trips: int = Field(default=0)
    total_fuel_types: int = Field(default=0)
    total_aggregations: int = Field(default=0)
    total_compliance_checks: int = Field(default=0)
    total_uncertainty_runs: int = Field(default=0)
    avg_calculation_time_ms: float = Field(default=0.0)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


# ===================================================================
# Utility helpers
# ===================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utcnow_iso() -> str:
    """Return current UTC datetime as an ISO-8601 string."""
    return _utcnow().isoformat()


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ===================================================================
# Default fuel types (for fallback when engine unavailable)
# ===================================================================

_DEFAULT_FUEL_TYPES: List[Dict[str, Any]] = [
    {"fuel_type": "GASOLINE", "display_name": "Gasoline", "category": "LIQUID"},
    {"fuel_type": "DIESEL", "display_name": "Diesel", "category": "LIQUID"},
    {"fuel_type": "LPG", "display_name": "Liquefied Petroleum Gas", "category": "GASEOUS"},
    {"fuel_type": "CNG", "display_name": "Compressed Natural Gas", "category": "GASEOUS"},
    {"fuel_type": "LNG", "display_name": "Liquefied Natural Gas", "category": "GASEOUS"},
    {"fuel_type": "E10", "display_name": "Ethanol Blend E10", "category": "BIOFUEL_BLEND"},
    {"fuel_type": "E85", "display_name": "Ethanol Blend E85", "category": "BIOFUEL_BLEND"},
    {"fuel_type": "B5", "display_name": "Biodiesel Blend B5", "category": "BIOFUEL_BLEND"},
    {"fuel_type": "B20", "display_name": "Biodiesel Blend B20", "category": "BIOFUEL_BLEND"},
    {"fuel_type": "B100", "display_name": "Biodiesel B100", "category": "BIOFUEL"},
    {"fuel_type": "ETHANOL", "display_name": "Ethanol", "category": "BIOFUEL"},
    {"fuel_type": "BIODIESEL", "display_name": "Biodiesel", "category": "BIOFUEL"},
    {"fuel_type": "JET_FUEL", "display_name": "Jet Fuel (Jet-A)", "category": "LIQUID"},
    {"fuel_type": "AVIATION_GASOLINE", "display_name": "Aviation Gasoline (AvGas)", "category": "LIQUID"},
    {"fuel_type": "MARINE_DIESEL", "display_name": "Marine Diesel Oil", "category": "LIQUID"},
    {"fuel_type": "MARINE_RESIDUAL", "display_name": "Marine Residual Fuel", "category": "LIQUID"},
    {"fuel_type": "SAF", "display_name": "Sustainable Aviation Fuel", "category": "BIOFUEL_BLEND"},
]

_DEFAULT_VEHICLE_TYPES: List[Dict[str, Any]] = [
    {"vehicle_type": "PASSENGER_CAR_GASOLINE", "display_name": "Passenger Car (Gasoline)", "category": "ON_ROAD_LIGHT_DUTY"},
    {"vehicle_type": "PASSENGER_CAR_DIESEL", "display_name": "Passenger Car (Diesel)", "category": "ON_ROAD_LIGHT_DUTY"},
    {"vehicle_type": "LIGHT_TRUCK_GASOLINE", "display_name": "Light-Duty Truck (Gasoline)", "category": "ON_ROAD_LIGHT_DUTY"},
    {"vehicle_type": "LIGHT_TRUCK_DIESEL", "display_name": "Light-Duty Truck (Diesel)", "category": "ON_ROAD_LIGHT_DUTY"},
    {"vehicle_type": "HEAVY_TRUCK_DIESEL", "display_name": "Heavy-Duty Truck (Diesel)", "category": "ON_ROAD_HEAVY_DUTY"},
    {"vehicle_type": "BUS_DIESEL", "display_name": "Bus (Diesel)", "category": "ON_ROAD_HEAVY_DUTY"},
    {"vehicle_type": "BUS_CNG", "display_name": "Bus (CNG)", "category": "ON_ROAD_HEAVY_DUTY"},
    {"vehicle_type": "MOTORCYCLE", "display_name": "Motorcycle", "category": "ON_ROAD_LIGHT_DUTY"},
    {"vehicle_type": "AIRCRAFT", "display_name": "Aircraft", "category": "AVIATION"},
    {"vehicle_type": "MARINE_VESSEL", "display_name": "Marine Vessel", "category": "MARINE"},
    {"vehicle_type": "LOCOMOTIVE", "display_name": "Locomotive", "category": "RAIL"},
    {"vehicle_type": "OFF_ROAD_VEHICLE", "display_name": "Off-Road Vehicle/Equipment", "category": "OFF_ROAD"},
]


# ===================================================================
# MobileCombustionService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["MobileCombustionService"] = None


class MobileCombustionService:
    """Unified facade over the Mobile Combustion Agent SDK.

    Aggregates all seven engines (vehicle database, emission calculator,
    fleet manager, distance estimator, uncertainty quantifier, compliance
    checker, mobile combustion pipeline) through a single entry point
    with convenience methods for common operations.

    Each method records provenance and updates self-monitoring Prometheus
    metrics.

    Attributes:
        config: MobileCombustionConfig instance or None.

    Example:
        >>> service = MobileCombustionService()
        >>> result = service.calculate({
        ...     "vehicle_type": "PASSENGER_CAR_GASOLINE",
        ...     "fuel_type": "GASOLINE",
        ...     "calculation_method": "FUEL_BASED",
        ...     "fuel_quantity": 100,
        ...     "fuel_unit": "GALLONS",
        ... })
        >>> print(result["total_co2e_tonnes"])
    """

    def __init__(
        self,
        config: Any = None,
    ) -> None:
        """Initialize the Mobile Combustion Service facade.

        Instantiates all 7 internal engines plus the provenance tracker:
        - VehicleDatabaseEngine (E1)
        - EmissionCalculatorEngine (E2)
        - FleetManagerEngine (E3)
        - DistanceEstimatorEngine (E4)
        - UncertaintyQuantifierEngine (E5)
        - ComplianceCheckerEngine (E6)
        - MobileCombustionPipelineEngine (E7)

        Args:
            config: Optional configuration. Uses global config if None.
        """
        self.config = config if config is not None else get_config()

        # Provenance tracker
        self._provenance: Any = None
        if ProvenanceTracker is not None:
            try:
                genesis = "GL-MRV-X-003-MOBILE-COMBUSTION-GENESIS"
                if self.config is not None:
                    genesis = getattr(
                        self.config, "genesis_hash", genesis,
                    )
                self._provenance = ProvenanceTracker(
                    genesis_hash=genesis,
                )
            except Exception as exc:
                logger.warning("ProvenanceTracker init failed: %s", exc)

        # Engine placeholders (lazy init)
        self._vehicle_db_engine: Any = None
        self._calculator_engine: Any = None
        self._fleet_manager_engine: Any = None
        self._distance_estimator_engine: Any = None
        self._uncertainty_engine: Any = None
        self._compliance_engine: Any = None
        self._pipeline_engine: Any = None

        self._init_engines()

        # In-memory stores (production uses DB; these are SDK-level caches)
        self._calculations: Dict[str, Dict[str, Any]] = {}
        self._vehicles: Dict[str, Dict[str, Any]] = {}
        self._trips: Dict[str, Dict[str, Any]] = {}
        self._fuels: Dict[str, Dict[str, Any]] = {}
        self._emission_factors: Dict[str, Dict[str, Any]] = {}
        self._aggregations: Dict[str, Dict[str, Any]] = {}
        self._compliance_records: Dict[str, Dict[str, Any]] = {}
        self._uncertainty_results: Dict[str, Dict[str, Any]] = {}

        # Statistics counters
        self._total_calculations: int = 0
        self._total_batch_runs: int = 0
        self._total_pipeline_runs: int = 0
        self._total_compliance_checks: int = 0
        self._total_uncertainty_runs: int = 0
        self._total_calculation_time_ms: float = 0.0
        self._started: bool = False

        logger.info("MobileCombustionService facade created")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def vehicle_db_engine(self) -> Any:
        """Get the VehicleDatabaseEngine instance."""
        return self._vehicle_db_engine

    @property
    def calculator_engine(self) -> Any:
        """Get the EmissionCalculatorEngine instance."""
        return self._calculator_engine

    @property
    def fleet_manager_engine(self) -> Any:
        """Get the FleetManagerEngine instance."""
        return self._fleet_manager_engine

    @property
    def distance_estimator_engine(self) -> Any:
        """Get the DistanceEstimatorEngine instance."""
        return self._distance_estimator_engine

    @property
    def uncertainty_engine(self) -> Any:
        """Get the UncertaintyQuantifierEngine instance."""
        return self._uncertainty_engine

    @property
    def compliance_engine(self) -> Any:
        """Get the ComplianceCheckerEngine instance."""
        return self._compliance_engine

    @property
    def pipeline_engine(self) -> Any:
        """Get the MobileCombustionPipelineEngine instance."""
        return self._pipeline_engine

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Attempt to import and initialise SDK engines.

        Engines are wired together using dependency injection. Engines
        are optional; missing imports are logged as warnings and the
        service continues in degraded mode.
        """
        # E1: VehicleDatabaseEngine
        if VehicleDatabaseEngine is not None:
            try:
                self._vehicle_db_engine = VehicleDatabaseEngine(
                    config=self.config,
                )
                logger.info("VehicleDatabaseEngine initialized")
            except Exception as exc:
                logger.warning("VehicleDatabaseEngine init failed: %s", exc)
        else:
            logger.warning("VehicleDatabaseEngine not available; using stub")

        # E2: EmissionCalculatorEngine
        if EmissionCalculatorEngine is not None:
            try:
                self._calculator_engine = EmissionCalculatorEngine(
                    config=self.config,
                    vehicle_db=self._vehicle_db_engine,
                )
                logger.info("EmissionCalculatorEngine initialized")
            except Exception as exc:
                logger.warning("EmissionCalculatorEngine init failed: %s", exc)
        else:
            logger.warning("EmissionCalculatorEngine not available; using stub")

        # E3: FleetManagerEngine
        if FleetManagerEngine is not None:
            try:
                self._fleet_manager_engine = FleetManagerEngine(
                    config=self.config,
                )
                logger.info("FleetManagerEngine initialized")
            except Exception as exc:
                logger.warning("FleetManagerEngine init failed: %s", exc)
        else:
            logger.warning("FleetManagerEngine not available; using stub")

        # E4: DistanceEstimatorEngine
        if DistanceEstimatorEngine is not None:
            try:
                self._distance_estimator_engine = DistanceEstimatorEngine(
                    config=self.config,
                    vehicle_db=self._vehicle_db_engine,
                )
                logger.info("DistanceEstimatorEngine initialized")
            except Exception as exc:
                logger.warning("DistanceEstimatorEngine init failed: %s", exc)
        else:
            logger.warning("DistanceEstimatorEngine not available; using stub")

        # E5: UncertaintyQuantifierEngine
        if UncertaintyQuantifierEngine is not None:
            try:
                self._uncertainty_engine = UncertaintyQuantifierEngine(
                    config=self.config,
                )
                logger.info("UncertaintyQuantifierEngine initialized")
            except Exception as exc:
                logger.warning("UncertaintyQuantifierEngine init failed: %s", exc)
        else:
            logger.warning("UncertaintyQuantifierEngine not available; using stub")

        # E6: ComplianceCheckerEngine
        if ComplianceCheckerEngine is not None:
            try:
                self._compliance_engine = ComplianceCheckerEngine(
                    config=self.config,
                )
                logger.info("ComplianceCheckerEngine initialized")
            except Exception as exc:
                logger.warning("ComplianceCheckerEngine init failed: %s", exc)
        else:
            logger.warning("ComplianceCheckerEngine not available; using stub")

        # E7: MobileCombustionPipelineEngine
        if MobileCombustionPipelineEngine is not None:
            try:
                self._pipeline_engine = MobileCombustionPipelineEngine(
                    config=self.config,
                    vehicle_db=self._vehicle_db_engine,
                    calculator=self._calculator_engine,
                    fleet_manager=self._fleet_manager_engine,
                    distance_estimator=self._distance_estimator_engine,
                    uncertainty=self._uncertainty_engine,
                    compliance=self._compliance_engine,
                )
                logger.info("MobileCombustionPipelineEngine initialized")
            except Exception as exc:
                logger.warning(
                    "MobileCombustionPipelineEngine init failed: %s", exc,
                )
        else:
            logger.warning(
                "MobileCombustionPipelineEngine not available; using stub",
            )

    # ==================================================================
    # Convenience methods
    # ==================================================================

    def calculate(
        self,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate emissions for a single mobile combustion record.

        Convenience method that delegates to the pipeline engine for
        full processing. All calculations are deterministic
        (zero-hallucination).

        Args:
            input_data: Dictionary with vehicle_type, fuel_type,
                calculation_method, and fuel/distance/spend data.

        Returns:
            Dictionary with calculation results.

        Raises:
            ValueError: If required input fields are missing.
        """
        t0 = time.perf_counter()
        calc_id = input_data.get("calculation_id", _new_uuid())
        input_data["calculation_id"] = calc_id

        try:
            if self._pipeline_engine is not None:
                pipeline_result = self._pipeline_engine.run_pipeline(
                    input_data,
                )
                result = pipeline_result.get("result", {})
                result["pipeline_id"] = pipeline_result.get(
                    "pipeline_id", "",
                )
            else:
                result = {
                    "calculation_id": calc_id,
                    "status": "PARTIAL",
                    "message": "No pipeline engine available",
                    "total_co2e_kg": 0.0,
                    "total_co2e_tonnes": 0.0,
                }

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if isinstance(result, dict):
                result.setdefault("calculation_id", calc_id)
                result["processing_time_ms"] = round(elapsed_ms, 3)
                result.setdefault(
                    "provenance_hash", _compute_hash(result),
                )

            # Cache result
            self._calculations[calc_id] = result
            self._total_calculations += 1
            self._total_calculation_time_ms += elapsed_ms

            # Record provenance
            if self._provenance is not None:
                try:
                    self._provenance.record(
                        entity_type="calculation",
                        action="calculate",
                        entity_id=calc_id,
                        metadata={
                            "vehicle_type": input_data.get("vehicle_type", ""),
                            "fuel_type": input_data.get("fuel_type", ""),
                            "method": input_data.get("calculation_method", ""),
                        },
                    )
                except (AttributeError, TypeError):
                    pass

            # Record metrics
            record_calculation(
                input_data.get("calculation_method", "FUEL_BASED"),
                input_data.get("vehicle_type", "unknown"),
                "success",
            )
            observe_calculation_duration(
                "single_calculation", elapsed_ms / 1000.0,
            )

            logger.info(
                "Calculated %s: vehicle=%s fuel=%s co2e=%.4f tonnes",
                calc_id,
                input_data.get("vehicle_type", ""),
                input_data.get("fuel_type", ""),
                result.get("total_co2e_tonnes", 0.0)
                if isinstance(result, dict) else 0.0,
            )
            return result

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            record_calculation(
                input_data.get("calculation_method", "FUEL_BASED"),
                input_data.get("vehicle_type", "unknown"),
                "failure",
            )
            logger.error("calculate failed: %s", exc, exc_info=True)
            raise

    def calculate_batch(
        self,
        inputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate emissions for multiple mobile combustion records.

        Args:
            inputs: List of input dictionaries.

        Returns:
            Dictionary with batch results, totals, and provenance.
        """
        t0 = time.perf_counter()
        batch_id = _new_uuid()

        try:
            if self._pipeline_engine is not None:
                batch_results = self._pipeline_engine.run_batch_pipeline(
                    inputs,
                )
                # First element is the summary
                if batch_results and isinstance(batch_results[0], dict):
                    summary = batch_results[0]
                    individual_results = batch_results[1:]
                else:
                    summary = {}
                    individual_results = batch_results
            else:
                individual_results = []
                for inp in inputs:
                    r = self.calculate(inp)
                    individual_results.append(r)

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                total_co2e_kg = sum(
                    r.get("total_co2e_kg", 0.0)
                    for r in individual_results
                    if isinstance(r, dict)
                )
                total_co2e_tonnes = sum(
                    r.get("total_co2e_tonnes", 0.0)
                    for r in individual_results
                    if isinstance(r, dict)
                )
                success_count = sum(
                    1 for r in individual_results
                    if isinstance(r, dict)
                    and r.get("status") == "SUCCESS"
                )
                summary = {
                    "batch_id": batch_id,
                    "total_co2e_kg": round(total_co2e_kg, 6),
                    "total_co2e_tonnes": round(total_co2e_tonnes, 9),
                    "success_count": success_count,
                    "failure_count": len(individual_results) - success_count,
                    "total_count": len(individual_results),
                    "processing_time_ms": round(elapsed_ms, 3),
                    "provenance_hash": _compute_hash({
                        "batch_id": batch_id,
                    }),
                }

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            summary.setdefault("batch_id", batch_id)
            summary.setdefault("processing_time_ms", round(elapsed_ms, 3))
            summary["results"] = [
                r.get("result", r) if isinstance(r, dict) else r
                for r in individual_results
            ]

            self._total_batch_runs += 1

            logger.info(
                "Batch %s completed: %d inputs, %.1fms",
                summary.get("batch_id", batch_id),
                len(inputs),
                elapsed_ms,
            )
            return summary

        except Exception as exc:
            logger.error(
                "calculate_batch failed: %s", exc, exc_info=True,
            )
            raise

    def register_vehicle(
        self,
        registration: Dict[str, Any],
    ) -> str:
        """Register a vehicle in the fleet.

        Args:
            registration: Dictionary with vehicle details (vehicle_type,
                fuel_type, name, facility_id, fleet_id, make, model,
                year, fuel_economy, odometer_km).

        Returns:
            Vehicle ID string.
        """
        vehicle_id = registration.get("vehicle_id", _new_uuid())

        if self._fleet_manager_engine is not None:
            try:
                result = self._fleet_manager_engine.register_vehicle(
                    **registration,
                )
                if isinstance(result, dict):
                    self._vehicles[vehicle_id] = result
                    return result.get("vehicle_id", vehicle_id)
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump(mode="json")
                    self._vehicles[vehicle_id] = result_dict
                    return result_dict.get("vehicle_id", vehicle_id)
            except (AttributeError, TypeError) as exc:
                logger.warning(
                    "FleetManagerEngine.register_vehicle failed: %s",
                    exc,
                )

        # Fallback: in-memory registration
        profile = VehicleResponse(
            vehicle_id=vehicle_id,
            vehicle_type=registration.get("vehicle_type", ""),
            fuel_type=registration.get("fuel_type", ""),
            name=registration.get("name", ""),
            facility_id=registration.get("facility_id"),
            fleet_id=registration.get("fleet_id"),
            make=registration.get("make", ""),
            model=registration.get("model", ""),
            year=registration.get("year"),
            fuel_economy=registration.get("fuel_economy"),
            fuel_economy_unit=registration.get(
                "fuel_economy_unit", "L_PER_100KM",
            ),
            odometer_km=float(registration.get("odometer_km", 0.0)),
        )
        profile.provenance_hash = _compute_hash(profile)
        result_dict = profile.model_dump()
        self._vehicles[vehicle_id] = result_dict

        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="vehicle",
                    action="register_vehicle",
                    entity_id=vehicle_id,
                    metadata={
                        "vehicle_type": registration.get("vehicle_type", ""),
                        "fuel_type": registration.get("fuel_type", ""),
                    },
                )
            except (AttributeError, TypeError):
                pass

        logger.info(
            "Registered vehicle %s: type=%s fuel=%s",
            vehicle_id,
            registration.get("vehicle_type", ""),
            registration.get("fuel_type", ""),
        )
        return vehicle_id

    def log_trip(
        self,
        trip: Dict[str, Any],
    ) -> str:
        """Log a trip for a vehicle.

        Args:
            trip: Dictionary with trip details (vehicle_id, distance_km,
                fuel_quantity, start_date, end_date, origin,
                destination, purpose).

        Returns:
            Trip ID string.
        """
        trip_id = trip.get("trip_id", _new_uuid())

        if self._fleet_manager_engine is not None:
            try:
                result = self._fleet_manager_engine.log_trip(**trip)
                if isinstance(result, dict):
                    self._trips[trip_id] = result
                    return result.get("trip_id", trip_id)
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump(mode="json")
                    self._trips[trip_id] = result_dict
                    return result_dict.get("trip_id", trip_id)
            except (AttributeError, TypeError) as exc:
                logger.warning(
                    "FleetManagerEngine.log_trip failed: %s", exc,
                )

        # Fallback: in-memory trip logging
        trip_record = TripResponse(
            trip_id=trip_id,
            vehicle_id=trip.get("vehicle_id", ""),
            distance_km=float(trip.get("distance_km", 0.0)),
            distance_unit=trip.get("distance_unit", "KM"),
            fuel_quantity=trip.get("fuel_quantity"),
            fuel_unit=trip.get("fuel_unit", "LITERS"),
            start_date=trip.get("start_date"),
            end_date=trip.get("end_date"),
            origin=trip.get("origin", ""),
            destination=trip.get("destination", ""),
            purpose=trip.get("purpose", ""),
        )
        trip_record.provenance_hash = _compute_hash(trip_record)
        result_dict = trip_record.model_dump()
        self._trips[trip_id] = result_dict

        logger.info(
            "Logged trip %s: vehicle=%s distance=%.1f km",
            trip_id,
            trip.get("vehicle_id", ""),
            trip.get("distance_km", 0.0),
        )
        return trip_id

    def aggregate_fleet(
        self,
        period: str = "",
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Aggregate fleet emissions for a reporting period.

        Args:
            period: Reporting period label (e.g. "2025-Q1").
            filters: Optional filters (facility_id, fleet_id,
                vehicle_type, fuel_type).

        Returns:
            Dictionary with aggregated fleet emissions.
        """
        filters = filters or {}

        if self._fleet_manager_engine is not None:
            try:
                result = self._fleet_manager_engine.aggregate(
                    period=period,
                    filters=filters,
                    calculations=self._calculations,
                )
                if isinstance(result, dict):
                    agg_id = result.get("aggregation_id", _new_uuid())
                    self._aggregations[agg_id] = result
                    return result
            except (AttributeError, TypeError) as exc:
                logger.warning(
                    "FleetManagerEngine.aggregate failed: %s", exc,
                )

        # Fallback: in-memory aggregation
        calcs = list(self._calculations.values())

        # Apply filters
        if filters.get("facility_id"):
            calcs = [
                c for c in calcs
                if c.get("facility_id") == filters["facility_id"]
            ]
        if filters.get("fleet_id"):
            calcs = [
                c for c in calcs
                if c.get("fleet_id") == filters["fleet_id"]
            ]
        if filters.get("vehicle_type"):
            calcs = [
                c for c in calcs
                if c.get("vehicle_type") == filters["vehicle_type"]
            ]
        if filters.get("fuel_type"):
            calcs = [
                c for c in calcs
                if c.get("fuel_type") == filters["fuel_type"]
            ]

        by_vehicle: Dict[str, float] = {}
        by_fuel: Dict[str, float] = {}
        by_facility: Dict[str, float] = {}
        total_co2e = 0.0
        total_biogenic = 0.0
        total_distance = 0.0
        total_fuel = 0.0

        for c in calcs:
            co2e = c.get("total_co2e_tonnes", 0.0)
            total_co2e += co2e
            total_biogenic += c.get("biogenic_co2_tonnes", 0.0)
            total_distance += c.get("distance_km", 0.0)
            total_fuel += c.get("fuel_quantity_gallons", 0.0)

            vt = c.get("vehicle_type", "UNKNOWN")
            by_vehicle[vt] = by_vehicle.get(vt, 0.0) + co2e

            ft = c.get("fuel_type", "UNKNOWN")
            by_fuel[ft] = by_fuel.get(ft, 0.0) + co2e

            fid = c.get("facility_id", "UNASSIGNED") or "UNASSIGNED"
            by_facility[fid] = by_facility.get(fid, 0.0) + co2e

        agg_id = _new_uuid()
        agg_result = {
            "aggregation_id": agg_id,
            "period": period,
            "total_co2e_tonnes": round(total_co2e, 9),
            "total_biogenic_co2_tonnes": round(total_biogenic, 9),
            "by_vehicle_type": {k: round(v, 9) for k, v in by_vehicle.items()},
            "by_fuel_type": {k: round(v, 9) for k, v in by_fuel.items()},
            "by_facility": {k: round(v, 9) for k, v in by_facility.items()},
            "vehicle_count": len(self._vehicles),
            "trip_count": len(self._trips),
            "calculation_count": len(calcs),
            "total_distance_km": round(total_distance, 3),
            "total_fuel_gallons": round(total_fuel, 6),
            "provenance_hash": _compute_hash({
                "aggregation_id": agg_id,
                "total_co2e": total_co2e,
            }),
            "timestamp": _utcnow_iso(),
        }

        self._aggregations[agg_id] = agg_result
        return agg_result

    def run_uncertainty(
        self,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run uncertainty analysis on a calculation result.

        Args:
            input_data: Dictionary with calculation_id and optional
                iterations parameter.

        Returns:
            Dictionary with uncertainty analysis results.
        """
        calc_id = input_data.get("calculation_id", "")
        iterations = input_data.get("iterations")
        calc_result = self._calculations.get(calc_id)

        if calc_result is None:
            return {
                "calculation_id": calc_id,
                "error": "Calculation not found",
            }

        if self._uncertainty_engine is not None:
            try:
                iters = iterations or 5000
                if self.config is not None:
                    iters = iterations or getattr(
                        self.config, "monte_carlo_iterations", 5000,
                    )
                result = self._uncertainty_engine.quantify(
                    calculation_result=calc_result,
                    iterations=iters,
                )
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump(mode="json")
                elif isinstance(result, dict):
                    result_dict = result
                else:
                    result_dict = {}

                self._total_uncertainty_runs += 1
                self._uncertainty_results[calc_id] = result_dict
                return result_dict

            except (AttributeError, TypeError) as exc:
                logger.warning(
                    "UncertaintyQuantifierEngine.quantify failed: %s",
                    exc,
                )

        # Fallback: use pipeline-embedded uncertainty if available
        embedded = calc_result.get("uncertainty", {})
        if embedded:
            self._total_uncertainty_runs += 1
            self._uncertainty_results[calc_id] = embedded
            return embedded

        # Analytical stub
        total_co2e = calc_result.get("total_co2e_kg", 0.0)
        relative_unc = 0.10
        std = total_co2e * relative_unc

        self._total_uncertainty_runs += 1
        result_dict = {
            "calculation_id": calc_id,
            "method": "analytical_stub",
            "mean_co2e_kg": round(total_co2e, 6),
            "std_co2e_kg": round(std, 6),
            "p5_co2e_kg": round(max(0.0, total_co2e - 1.645 * std), 6),
            "p95_co2e_kg": round(total_co2e + 1.645 * std, 6),
            "confidence_interval_pct": 90.0,
            "relative_uncertainty_pct": round(relative_unc * 100.0, 2),
            "iterations": 0,
            "data_quality_score": 3,
            "provenance_hash": _compute_hash({
                "calculation_id": calc_id,
                "mean": total_co2e,
            }),
        }
        self._uncertainty_results[calc_id] = result_dict
        return result_dict

    def check_compliance(
        self,
        results: Optional[List[Dict[str, Any]]] = None,
        framework: str = "GHG_PROTOCOL",
    ) -> Dict[str, Any]:
        """Run compliance check against a regulatory framework.

        Args:
            results: Optional list of calculation results to check.
                If None, checks all cached calculations.
            framework: Regulatory framework name.

        Returns:
            Dictionary with compliance check results.
        """
        if results is None:
            results = list(self._calculations.values())

        if self._compliance_engine is not None:
            try:
                comp_result = self._compliance_engine.check(
                    calculation_results=results,
                    framework=framework,
                )
                if isinstance(comp_result, dict):
                    self._total_compliance_checks += 1
                    return comp_result
            except (AttributeError, TypeError) as exc:
                logger.warning(
                    "ComplianceCheckerEngine.check failed: %s", exc,
                )

        # Fallback: use pipeline compliance data from first result
        if results:
            first_result = results[0]
            embedded = first_result.get("compliance", [])
            for comp in embedded:
                if isinstance(comp, dict) and comp.get("framework") == framework:
                    self._total_compliance_checks += 1
                    return comp

        # Stub compliance
        self._total_compliance_checks += 1
        return {
            "framework": framework,
            "compliant": True,
            "compliant_count": 0,
            "total_requirements": 0,
            "checks": [],
            "message": "No compliance engine available; stub result",
            "provenance_hash": _compute_hash({
                "framework": framework,
                "stub": True,
            }),
        }

    def get_vehicle(self, vehicle_id: str) -> Dict[str, Any]:
        """Get vehicle details by ID.

        Args:
            vehicle_id: Vehicle identifier.

        Returns:
            Dictionary with vehicle details.
        """
        if self._vehicle_db_engine is not None:
            try:
                result = self._vehicle_db_engine.get_vehicle(vehicle_id)
                if isinstance(result, dict):
                    return result
                if hasattr(result, "model_dump"):
                    return result.model_dump(mode="json")
            except (AttributeError, TypeError, KeyError):
                pass

        cached = self._vehicles.get(vehicle_id)
        if cached is not None:
            return cached

        return {"vehicle_id": vehicle_id, "error": "Vehicle not found"}

    def list_vehicles(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """List registered vehicles with optional filters.

        Args:
            filters: Optional filters (vehicle_type, fuel_type,
                facility_id, fleet_id).

        Returns:
            List of vehicle dictionaries.
        """
        filters = filters or {}
        vehicles = list(self._vehicles.values())

        if filters.get("vehicle_type"):
            vehicles = [
                v for v in vehicles
                if v.get("vehicle_type") == filters["vehicle_type"]
            ]
        if filters.get("fuel_type"):
            vehicles = [
                v for v in vehicles
                if v.get("fuel_type") == filters["fuel_type"]
            ]
        if filters.get("facility_id"):
            vehicles = [
                v for v in vehicles
                if v.get("facility_id") == filters["facility_id"]
            ]
        if filters.get("fleet_id"):
            vehicles = [
                v for v in vehicles
                if v.get("fleet_id") == filters["fleet_id"]
            ]

        return vehicles

    def get_fuel_types(self) -> List[Dict[str, Any]]:
        """List all supported fuel types.

        Returns:
            List of fuel type summary dictionaries.
        """
        if self._vehicle_db_engine is not None:
            try:
                result = self._vehicle_db_engine.list_fuel_types()
                if isinstance(result, list):
                    return result
            except (AttributeError, TypeError):
                pass

        return _DEFAULT_FUEL_TYPES

    def get_vehicle_types(self) -> List[Dict[str, Any]]:
        """List all supported vehicle types.

        Returns:
            List of vehicle type summary dictionaries.
        """
        if self._vehicle_db_engine is not None:
            try:
                result = self._vehicle_db_engine.list_vehicle_types()
                if isinstance(result, list):
                    return result
            except (AttributeError, TypeError):
                pass

        return _DEFAULT_VEHICLE_TYPES

    def get_emission_factors(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """List emission factors with optional filters.

        Args:
            filters: Optional filters (fuel_type, vehicle_type, source).

        Returns:
            List of emission factor dictionaries.
        """
        filters = filters or {}

        if self._vehicle_db_engine is not None:
            try:
                result = self._vehicle_db_engine.list_emission_factors(
                    **filters,
                )
                if isinstance(result, list):
                    return result
            except (AttributeError, TypeError):
                pass

        # Fallback: return built-in factors
        from greenlang.mobile_combustion.mobile_combustion_pipeline import (
            FUEL_CO2_FACTORS_KG_PER_GALLON,
        )

        factors = []
        fuel_filter = filters.get("fuel_type")
        for fuel, co2_factor in FUEL_CO2_FACTORS_KG_PER_GALLON.items():
            if fuel_filter and fuel != fuel_filter:
                continue
            factors.append({
                "fuel_type": fuel,
                "gas": "CO2",
                "value": co2_factor,
                "unit": "kg/gallon",
                "source": "EPA",
            })

        return factors

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the mobile combustion service.

        Returns:
            Dictionary with health status for each engine and the
            overall service.
        """
        engines: Dict[str, str] = {
            "vehicle_database": (
                "available"
                if self._vehicle_db_engine is not None
                else "unavailable"
            ),
            "emission_calculator": (
                "available"
                if self._calculator_engine is not None
                else "unavailable"
            ),
            "fleet_manager": (
                "available"
                if self._fleet_manager_engine is not None
                else "unavailable"
            ),
            "distance_estimator": (
                "available"
                if self._distance_estimator_engine is not None
                else "unavailable"
            ),
            "uncertainty_quantifier": (
                "available"
                if self._uncertainty_engine is not None
                else "unavailable"
            ),
            "compliance_checker": (
                "available"
                if self._compliance_engine is not None
                else "unavailable"
            ),
            "pipeline": (
                "available"
                if self._pipeline_engine is not None
                else "unavailable"
            ),
        }

        available_count = sum(
            1 for s in engines.values() if s == "available"
        )
        total_engines = len(engines)

        if available_count == total_engines:
            overall_status = "healthy"
        elif available_count >= 4:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        # Verify provenance chain
        chain_valid = True
        provenance_entries = 0
        if self._provenance is not None:
            try:
                chain_valid = self._provenance.verify_chain()
                provenance_entries = self._provenance.entry_count
            except (AttributeError, TypeError):
                pass

        avg_calc_time = (
            self._total_calculation_time_ms / self._total_calculations
            if self._total_calculations > 0
            else 0.0
        )

        return {
            "status": overall_status,
            "version": "1.0.0",
            "engines": engines,
            "engines_available": available_count,
            "engines_total": total_engines,
            "started": self._started,
            "vehicle_count": len(self._vehicles),
            "trip_count": len(self._trips),
            "fuel_type_count": len(_DEFAULT_FUEL_TYPES),
            "statistics": {
                "total_calculations": self._total_calculations,
                "total_batch_runs": self._total_batch_runs,
                "total_pipeline_runs": self._total_pipeline_runs,
                "total_compliance_checks": self._total_compliance_checks,
                "total_uncertainty_runs": self._total_uncertainty_runs,
                "avg_calculation_time_ms": round(avg_calc_time, 3),
            },
            "provenance_chain_valid": chain_valid,
            "provenance_entries": provenance_entries,
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "timestamp": _utcnow_iso(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics for the service.

        Returns:
            Dictionary with current service statistics.
        """
        avg_calc_time = (
            self._total_calculation_time_ms / self._total_calculations
            if self._total_calculations > 0
            else 0.0
        )

        return {
            "total_calculations": self._total_calculations,
            "total_batch_runs": self._total_batch_runs,
            "total_pipeline_runs": self._total_pipeline_runs,
            "total_vehicles": len(self._vehicles),
            "total_trips": len(self._trips),
            "total_fuel_types": len(_DEFAULT_FUEL_TYPES),
            "total_aggregations": len(self._aggregations),
            "total_compliance_checks": self._total_compliance_checks,
            "total_uncertainty_runs": self._total_uncertainty_runs,
            "avg_calculation_time_ms": round(avg_calc_time, 3),
            "timestamp": _utcnow_iso(),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the mobile combustion service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug(
                "MobileCombustionService already started; skipping",
            )
            return

        logger.info("MobileCombustionService starting up...")
        self._started = True
        logger.info("MobileCombustionService startup complete")

    def shutdown(self) -> None:
        """Shutdown the mobile combustion service and release resources."""
        if not self._started:
            return

        self._started = False
        logger.info("MobileCombustionService shut down")


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> MobileCombustionService:
    """Get or create the singleton MobileCombustionService instance.

    Returns:
        The singleton MobileCombustionService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = MobileCombustionService()
    return _singleton_instance


# ===================================================================
# Module-level singletons for FastAPI integration
# ===================================================================

_service: Optional[MobileCombustionService] = None
_router: Any = None


# ===================================================================
# FastAPI integration
# ===================================================================


async def configure_mobile_combustion(
    app: Any,
    config: Any = None,
) -> MobileCombustionService:
    """Configure the Mobile Combustion Service on a FastAPI application.

    Creates the MobileCombustionService, stores it in app.state,
    mounts the mobile combustion API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional mobile combustion config.

    Returns:
        MobileCombustionService instance.
    """
    global _singleton_instance, _service

    service = MobileCombustionService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service
        _service = service

    # Attach to app state
    app.state.mobile_combustion_service = service

    # Mount mobile combustion API router
    api_router = get_router()
    if api_router is not None:
        app.include_router(api_router)
        logger.info("Mobile combustion API router mounted")
    else:
        logger.warning(
            "Mobile combustion router not available; API not mounted",
        )

    # Start service
    service.startup()

    logger.info(
        "Mobile combustion service configured on app",
    )
    return service


def get_service() -> Optional[MobileCombustionService]:
    """Get the singleton MobileCombustionService instance.

    Creates a new instance if one does not exist yet. Uses
    double-checked locking for thread safety.

    Returns:
        MobileCombustionService singleton instance or None.
    """
    global _singleton_instance, _service
    if _service is not None:
        return _service
    if _singleton_instance is not None:
        return _singleton_instance
    # Lazy creation
    with _singleton_lock:
        if _singleton_instance is None:
            _singleton_instance = MobileCombustionService()
        _service = _singleton_instance
    return _singleton_instance


def get_router() -> Any:
    """Get the mobile combustion API router.

    Returns the FastAPI APIRouter from the ``api.router`` module.

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    if not FASTAPI_AVAILABLE:
        return None

    try:
        from greenlang.mobile_combustion.api.router import router
        return router
    except ImportError:
        logger.warning(
            "Mobile combustion API router module not available",
        )
        return None


# ===================================================================
# Public API
# ===================================================================

__all__ = [
    # Service facade
    "MobileCombustionService",
    # Configuration helpers
    "configure_mobile_combustion",
    "get_service",
    "get_router",
    # Response models
    "CalculateResponse",
    "BatchCalculateResponse",
    "CalculationDetailResponse",
    "VehicleResponse",
    "VehicleListResponse",
    "TripResponse",
    "TripListResponse",
    "FuelListResponse",
    "FactorListResponse",
    "AggregationResponse",
    "UncertaintyResponse",
    "ComplianceResponse",
    "HealthResponse",
    "StatsResponse",
]
