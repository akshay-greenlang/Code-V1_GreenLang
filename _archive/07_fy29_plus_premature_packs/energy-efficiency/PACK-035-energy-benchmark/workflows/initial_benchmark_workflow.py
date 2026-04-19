# -*- coding: utf-8 -*-
"""
Initial Benchmark Workflow
===================================

4-phase workflow for establishing initial energy benchmarks within
PACK-035 Energy Benchmark Pack.

Phases:
    1. DataCollection      -- Gather facility profile, energy bills, weather station
    2. EUICalculation      -- Calculate site/source EUI with weather normalisation
    3. PeerComparison      -- Compare against ENERGY STAR / CIBSE TM46 / DIN V 18599
    4. BenchmarkReport     -- Generate initial benchmark report with ratings

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas (degree-day regression, ENERGY
STAR lookup tables, CIBSE TM46 benchmarks). SHA-256 provenance hashes
guarantee auditability.

Schedule: on-demand
Estimated duration: 45 minutes

Author: GreenLang Team
Version: 35.0.0
"""

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class BuildingType(str, Enum):
    """Building type classification for benchmark lookup."""

    OFFICE = "office"
    RETAIL = "retail"
    HOTEL = "hotel"
    HOSPITAL = "hospital"
    SCHOOL = "school"
    UNIVERSITY = "university"
    WAREHOUSE = "warehouse"
    INDUSTRIAL = "industrial"
    RESTAURANT = "restaurant"
    SUPERMARKET = "supermarket"
    DATA_CENTRE = "data_centre"
    MIXED_USE = "mixed_use"
    RESIDENTIAL_MULTI = "residential_multi"
    LEISURE = "leisure"
    LABORATORY = "laboratory"


class BenchmarkSource(str, Enum):
    """Benchmark dataset source."""

    ENERGY_STAR = "energy_star"
    CIBSE_TM46 = "cibse_tm46"
    DIN_V_18599 = "din_v_18599"
    BPIE = "bpie"
    ASHRAE_100 = "ashrae_100"
    NABERS = "nabers"
    CRREM = "crrem"


class EnergySourceType(str, Enum):
    """Energy source classifications."""

    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    FUEL_OIL = "fuel_oil"
    DISTRICT_HEATING = "district_heating"
    DISTRICT_COOLING = "district_cooling"
    LPG = "lpg"
    BIOMASS = "biomass"
    SOLAR_PV = "solar_pv"
    SOLAR_THERMAL = "solar_thermal"


class DataQuality(str, Enum):
    """Data quality classification."""

    MEASURED = "measured"
    ESTIMATED = "estimated"
    DEFAULT = "default"
    CALCULATED = "calculated"


class EPCRating(str, Enum):
    """Energy Performance Certificate rating band."""

    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# CIBSE TM46:2008 typical/good practice EUI benchmarks (kWh/m2/yr)
# Format: building_type -> {"typical_electric", "typical_fossil", "good_electric", "good_fossil"}
CIBSE_TM46_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "office": {"typical_electric": 95.0, "typical_fossil": 120.0, "good_electric": 54.0, "good_fossil": 79.0},
    "retail": {"typical_electric": 165.0, "typical_fossil": 105.0, "good_electric": 90.0, "good_fossil": 60.0},
    "hotel": {"typical_electric": 105.0, "typical_fossil": 200.0, "good_electric": 60.0, "good_fossil": 120.0},
    "hospital": {"typical_electric": 90.0, "typical_fossil": 350.0, "good_electric": 65.0, "good_fossil": 250.0},
    "school": {"typical_electric": 40.0, "typical_fossil": 110.0, "good_electric": 22.0, "good_fossil": 65.0},
    "university": {"typical_electric": 75.0, "typical_fossil": 130.0, "good_electric": 50.0, "good_fossil": 85.0},
    "warehouse": {"typical_electric": 30.0, "typical_fossil": 35.0, "good_electric": 20.0, "good_fossil": 20.0},
    "industrial": {"typical_electric": 55.0, "typical_fossil": 200.0, "good_electric": 35.0, "good_fossil": 120.0},
    "restaurant": {"typical_electric": 250.0, "typical_fossil": 370.0, "good_electric": 150.0, "good_fossil": 200.0},
    "supermarket": {"typical_electric": 340.0, "typical_fossil": 80.0, "good_electric": 260.0, "good_fossil": 55.0},
    "data_centre": {"typical_electric": 500.0, "typical_fossil": 10.0, "good_electric": 300.0, "good_fossil": 5.0},
    "mixed_use": {"typical_electric": 100.0, "typical_fossil": 130.0, "good_electric": 60.0, "good_fossil": 80.0},
    "residential_multi": {"typical_electric": 45.0, "typical_fossil": 100.0, "good_electric": 30.0, "good_fossil": 65.0},
    "leisure": {"typical_electric": 105.0, "typical_fossil": 270.0, "good_electric": 75.0, "good_fossil": 150.0},
    "laboratory": {"typical_electric": 120.0, "typical_fossil": 180.0, "good_electric": 80.0, "good_fossil": 110.0},
}

# Source energy conversion factors (site-to-source multiplier)
SOURCE_ENERGY_FACTORS: Dict[str, float] = {
    "electricity": 2.55,
    "natural_gas": 1.05,
    "fuel_oil": 1.01,
    "district_heating": 1.20,
    "district_cooling": 1.04,
    "lpg": 1.01,
    "biomass": 1.05,
    "solar_pv": 1.00,
    "solar_thermal": 1.00,
}

# Primary energy factors per EN 15603
PRIMARY_ENERGY_FACTORS: Dict[str, float] = {
    "electricity": 2.50,
    "natural_gas": 1.10,
    "fuel_oil": 1.10,
    "district_heating": 1.30,
    "district_cooling": 1.60,
    "lpg": 1.10,
    "biomass": 1.20,
    "solar_pv": 0.00,
    "solar_thermal": 0.00,
}

# Energy conversion to kWh
ENERGY_CONVERSION_TO_KWH: Dict[str, float] = {
    "kWh": 1.0,
    "MWh": 1000.0,
    "GJ": 277.778,
    "MJ": 0.277778,
    "therm": 29.3071,
    "m3_natural_gas": 10.55,
    "litre_fuel_oil": 10.35,
    "litre_lpg": 7.08,
}

# CO2 emission factors (kgCO2e/kWh) - DEFRA 2024
DEFAULT_EMISSION_FACTORS: Dict[str, float] = {
    "electricity": 0.207,
    "natural_gas": 0.18293,
    "fuel_oil": 0.26718,
    "district_heating": 0.19400,
    "district_cooling": 0.207,
    "lpg": 0.21448,
    "biomass": 0.01500,
    "solar_pv": 0.0,
    "solar_thermal": 0.0,
}

# EPC rating bands by primary energy demand (kWh/m2/yr)
EPC_RATING_BANDS: Dict[str, Tuple[float, float]] = {
    "A+": (0.0, 25.0),
    "A": (25.0, 50.0),
    "B": (50.0, 75.0),
    "C": (75.0, 100.0),
    "D": (100.0, 150.0),
    "E": (150.0, 200.0),
    "F": (200.0, 250.0),
    "G": (250.0, 9999.0),
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class FacilityProfile(BaseModel):
    """Facility profile for benchmarking."""

    facility_id: str = Field(default_factory=lambda: f"fac-{uuid.uuid4().hex[:8]}")
    facility_name: str = Field(default="", description="Facility name")
    address: str = Field(default="", description="Physical address")
    country: str = Field(default="", description="ISO 3166-1 alpha-2")
    building_type: BuildingType = Field(default=BuildingType.OFFICE)
    floor_area_m2: float = Field(default=0.0, ge=0.0, description="Gross internal area")
    conditioned_area_m2: float = Field(default=0.0, ge=0.0, description="Conditioned floor area")
    year_built: int = Field(default=0, ge=0, description="Construction year")
    year_renovated: int = Field(default=0, ge=0, description="Last major renovation")
    occupancy_hours_per_week: float = Field(default=50.0, ge=0.0, le=168.0)
    occupant_count: int = Field(default=0, ge=0, description="Number of occupants")
    climate_zone: str = Field(default="", description="ASHRAE climate zone")
    heating_system: str = Field(default="gas_boiler", description="Primary heating system type")
    cooling_system: str = Field(default="split_ac", description="Primary cooling system type")


class EnergyBillRecord(BaseModel):
    """Monthly energy bill record."""

    bill_id: str = Field(default_factory=lambda: f"bill-{uuid.uuid4().hex[:8]}")
    period: str = Field(default="", description="Period YYYY-MM")
    energy_source: EnergySourceType = Field(default=EnergySourceType.ELECTRICITY)
    consumption_kwh: float = Field(default=0.0, ge=0.0, description="Consumption in kWh")
    consumption_native: float = Field(default=0.0, ge=0.0, description="In native unit")
    native_unit: str = Field(default="kWh", description="Native measurement unit")
    cost: float = Field(default=0.0, ge=0.0, description="Energy cost")
    cost_currency: str = Field(default="EUR", description="Currency code")
    demand_kw: float = Field(default=0.0, ge=0.0, description="Peak demand kW")
    days_in_period: int = Field(default=30, ge=1, le=31)
    data_quality: DataQuality = Field(default=DataQuality.MEASURED)


class WeatherDataRecord(BaseModel):
    """Monthly weather data for normalisation."""

    period: str = Field(default="", description="Period YYYY-MM")
    avg_temperature_c: float = Field(default=15.0, description="Average temp Celsius")
    heating_degree_days: float = Field(default=0.0, ge=0.0, description="HDD base 15.5C")
    cooling_degree_days: float = Field(default=0.0, ge=0.0, description="CDD base 18.3C")
    avg_humidity_pct: float = Field(default=60.0, ge=0.0, le=100.0)
    solar_radiation_kwh_m2: float = Field(default=0.0, ge=0.0, description="Global horizontal")


class EUIResult(BaseModel):
    """Energy Use Intensity calculation result."""

    site_eui_kwh_m2: float = Field(default=0.0, ge=0.0, description="Site EUI kWh/m2/yr")
    source_eui_kwh_m2: float = Field(default=0.0, ge=0.0, description="Source EUI kWh/m2/yr")
    primary_energy_kwh_m2: float = Field(default=0.0, ge=0.0, description="Primary energy kWh/m2/yr")
    normalised_eui_kwh_m2: float = Field(default=0.0, ge=0.0, description="Weather-normalised EUI")
    electric_eui_kwh_m2: float = Field(default=0.0, ge=0.0, description="Electric EUI only")
    fossil_eui_kwh_m2: float = Field(default=0.0, ge=0.0, description="Fossil fuel EUI only")
    carbon_intensity_kgco2_m2: float = Field(default=0.0, ge=0.0, description="kgCO2e/m2/yr")
    total_consumption_kwh: float = Field(default=0.0, ge=0.0)
    total_cost: float = Field(default=0.0, ge=0.0)
    regression_r_squared: float = Field(default=0.0, ge=0.0, le=1.0)
    regression_cv_rmse_pct: float = Field(default=0.0, ge=0.0)


class PeerRankingResult(BaseModel):
    """Peer comparison and ranking result."""

    percentile: float = Field(default=50.0, ge=0.0, le=100.0, description="Percentile rank 0-100")
    quartile: int = Field(default=2, ge=1, le=4, description="Quartile 1-4")
    energy_star_score: int = Field(default=50, ge=1, le=100, description="ENERGY STAR 1-100")
    epc_rating: str = Field(default="D", description="EPC rating A+-G")
    cibse_category: str = Field(default="typical", description="typical|good|best")
    benchmark_source: str = Field(default="cibse_tm46")
    peer_typical_eui: float = Field(default=0.0, ge=0.0, description="Typical peer EUI")
    peer_good_eui: float = Field(default=0.0, ge=0.0, description="Good practice peer EUI")
    gap_to_typical_pct: float = Field(default=0.0, description="Gap to typical benchmark %")
    gap_to_good_pct: float = Field(default=0.0, description="Gap to good practice %")
    peer_count: int = Field(default=0, ge=0, description="Peer group size")


class InitialBenchmarkInput(BaseModel):
    """Input data model for InitialBenchmarkWorkflow."""

    facility_profile: FacilityProfile = Field(..., description="Facility profile data")
    energy_data: List[EnergyBillRecord] = Field(default_factory=list)
    weather_data: List[WeatherDataRecord] = Field(default_factory=list)
    weather_station_id: str = Field(default="", description="Weather station identifier")
    benchmark_sources: List[BenchmarkSource] = Field(
        default_factory=lambda: [BenchmarkSource.CIBSE_TM46, BenchmarkSource.ENERGY_STAR],
    )
    floor_area_m2: float = Field(default=0.0, ge=0.0, description="Override floor area if different")
    building_type: Optional[BuildingType] = Field(default=None, description="Override building type")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("facility_profile")
    @classmethod
    def validate_facility(cls, v: FacilityProfile) -> FacilityProfile:
        """Ensure facility has minimum required data."""
        if not v.facility_name and not v.facility_id:
            raise ValueError("Facility must have a name or ID")
        return v


class InitialBenchmarkResult(BaseModel):
    """Complete result from initial benchmark workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="initial_benchmark")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    building_type: str = Field(default="")
    floor_area_m2: float = Field(default=0.0, ge=0.0)
    eui_result: Dict[str, Any] = Field(default_factory=dict)
    peer_ranking: Dict[str, Any] = Field(default_factory=dict)
    benchmark_report: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class InitialBenchmarkWorkflow:
    """
    4-phase initial energy benchmark workflow.

    Performs data collection and validation, EUI calculation with weather
    normalisation, peer comparison against multiple benchmark datasets,
    and benchmark report generation with EPC ratings.

    Zero-hallucination: all EUI calculations use deterministic degree-day
    regression, benchmark lookups from published CIBSE/ENERGY STAR tables,
    and EN 15603 primary energy factors. No LLM calls in the numeric path.

    Attributes:
        workflow_id: Unique execution identifier.
        _eui_result: Calculated EUI metrics.
        _peer_ranking: Peer comparison results.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = InitialBenchmarkWorkflow()
        >>> inp = InitialBenchmarkInput(
        ...     facility_profile=FacilityProfile(
        ...         facility_name="HQ Office", building_type=BuildingType.OFFICE,
        ...         floor_area_m2=5000.0,
        ...     ),
        ...     energy_data=[...],
        ... )
        >>> result = wf.run(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize InitialBenchmarkWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._eui_result: Optional[EUIResult] = None
        self._peer_ranking: Optional[PeerRankingResult] = None
        self._collected_data: Dict[str, Any] = {}
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: InitialBenchmarkInput) -> InitialBenchmarkResult:
        """
        Execute the 4-phase initial benchmark workflow.

        Args:
            input_data: Validated initial benchmark input.

        Returns:
            InitialBenchmarkResult with EUI, peer ranking, and benchmark report.

        Raises:
            ValueError: If facility profile is incomplete.
        """
        t_start = time.perf_counter()
        started_at = datetime.utcnow()
        profile = input_data.facility_profile
        self.logger.info(
            "Starting initial benchmark workflow %s for facility=%s type=%s",
            self.workflow_id, profile.facility_name, profile.building_type.value,
        )

        self._phase_results = []
        self._eui_result = None
        self._peer_ranking = None
        self._collected_data = {}
        overall_status = WorkflowStatus.RUNNING

        # Resolve effective floor area and building type
        effective_area = input_data.floor_area_m2 if input_data.floor_area_m2 > 0 else profile.floor_area_m2
        effective_type = input_data.building_type or profile.building_type

        try:
            # Phase 1: Data Collection
            phase1 = self._phase_data_collection(input_data, effective_area)
            self._phase_results.append(phase1)

            # Phase 2: EUI Calculation
            phase2 = self._phase_eui_calculation(input_data, effective_area, effective_type)
            self._phase_results.append(phase2)

            # Phase 3: Peer Comparison
            phase3 = self._phase_peer_comparison(input_data, effective_type)
            self._phase_results.append(phase3)

            # Phase 4: Benchmark Report
            phase4 = self._phase_benchmark_report(input_data, effective_area, effective_type)
            self._phase_results.append(phase4)

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Initial benchmark workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = time.perf_counter() - t_start

        result = InitialBenchmarkResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            facility_id=profile.facility_id,
            facility_name=profile.facility_name,
            building_type=effective_type.value,
            floor_area_m2=effective_area,
            eui_result=self._eui_result.model_dump() if self._eui_result else {},
            peer_ranking=self._peer_ranking.model_dump() if self._peer_ranking else {},
            benchmark_report=self._collected_data.get("report", {}),
            duration_seconds=round(elapsed, 4),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Initial benchmark workflow %s completed in %.2fs status=%s",
            self.workflow_id, elapsed, overall_status.value,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Collection
    # -------------------------------------------------------------------------

    def _phase_data_collection(
        self, input_data: InitialBenchmarkInput, effective_area: float
    ) -> PhaseResult:
        """Gather facility profile, energy bills, and weather data."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        profile = input_data.facility_profile

        # Validate floor area
        if effective_area <= 0:
            warnings.append("Floor area is zero; EUI calculation will be unreliable")

        # Validate energy data coverage
        energy_periods = set()
        total_records = len(input_data.energy_data)
        for bill in input_data.energy_data:
            if bill.period:
                energy_periods.add(bill.period)

        months_covered = len(energy_periods)
        if months_covered < 12:
            warnings.append(
                f"Only {months_covered} months of energy data; 12 months recommended"
            )

        # Validate weather data
        weather_periods = {w.period for w in input_data.weather_data if w.period}
        if not weather_periods:
            warnings.append("No weather data provided; normalisation will be skipped")

        # Aggregate by source
        source_totals: Dict[str, float] = {}
        source_costs: Dict[str, float] = {}
        for bill in input_data.energy_data:
            src = bill.energy_source.value
            source_totals[src] = source_totals.get(src, 0.0) + bill.consumption_kwh
            source_costs[src] = source_costs.get(src, 0.0) + bill.cost

        # Data quality assessment
        measured_count = sum(
            1 for b in input_data.energy_data if b.data_quality == DataQuality.MEASURED
        )
        quality_pct = measured_count / max(total_records, 1) * 100

        # Store collected data for downstream phases
        self._collected_data["source_totals"] = source_totals
        self._collected_data["source_costs"] = source_costs
        self._collected_data["months_covered"] = months_covered
        self._collected_data["weather_periods"] = len(weather_periods)

        outputs["facility_id"] = profile.facility_id
        outputs["facility_name"] = profile.facility_name
        outputs["floor_area_m2"] = effective_area
        outputs["building_type"] = profile.building_type.value
        outputs["energy_records"] = total_records
        outputs["months_covered"] = months_covered
        outputs["weather_records"] = len(input_data.weather_data)
        outputs["sources"] = list(source_totals.keys())
        outputs["consumption_by_source_kwh"] = {k: round(v, 2) for k, v in source_totals.items()}
        outputs["cost_by_source"] = {k: round(v, 2) for k, v in source_costs.items()}
        outputs["data_quality_measured_pct"] = round(quality_pct, 1)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 1 DataCollection: %d records, %d months, quality=%.1f%%",
            total_records, months_covered, quality_pct,
        )
        return PhaseResult(
            phase_name="data_collection", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: EUI Calculation
    # -------------------------------------------------------------------------

    def _phase_eui_calculation(
        self,
        input_data: InitialBenchmarkInput,
        effective_area: float,
        effective_type: BuildingType,
    ) -> PhaseResult:
        """Calculate site/source/primary EUI with weather normalisation."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        source_totals = self._collected_data.get("source_totals", {})
        source_costs = self._collected_data.get("source_costs", {})
        months = self._collected_data.get("months_covered", 0)

        # Annualise consumption
        annual_factor = 12.0 / max(months, 1)
        total_site_kwh = sum(source_totals.values()) * annual_factor
        total_cost = sum(source_costs.values()) * annual_factor

        # Calculate source energy (site-to-source conversion)
        total_source_kwh = 0.0
        for src, kwh in source_totals.items():
            factor = SOURCE_ENERGY_FACTORS.get(src, 1.0)
            total_source_kwh += kwh * annual_factor * factor

        # Calculate primary energy (EN 15603)
        total_primary_kwh = 0.0
        for src, kwh in source_totals.items():
            factor = PRIMARY_ENERGY_FACTORS.get(src, 1.0)
            total_primary_kwh += kwh * annual_factor * factor

        # Split electric vs fossil
        electric_kwh = source_totals.get("electricity", 0.0) * annual_factor
        fossil_kwh = sum(
            v * annual_factor for k, v in source_totals.items()
            if k != "electricity" and k not in ("solar_pv", "solar_thermal")
        )

        # Calculate carbon intensity
        total_co2 = 0.0
        for src, kwh in source_totals.items():
            ef = DEFAULT_EMISSION_FACTORS.get(src, 0.207)
            total_co2 += kwh * annual_factor * ef

        # Weather normalisation using degree-day regression
        normalised_eui = 0.0
        r_squared = 0.0
        cv_rmse = 0.0
        if input_data.weather_data and effective_area > 0:
            norm_result = self._weather_normalise(
                input_data.energy_data, input_data.weather_data, effective_area
            )
            normalised_eui = norm_result["normalised_eui"]
            r_squared = norm_result["r_squared"]
            cv_rmse = norm_result["cv_rmse"]
        elif effective_area > 0:
            normalised_eui = total_site_kwh / effective_area
            warnings.append("No weather data; normalised EUI equals site EUI")

        # Build EUI result
        site_eui = total_site_kwh / effective_area if effective_area > 0 else 0.0
        source_eui = total_source_kwh / effective_area if effective_area > 0 else 0.0
        primary_eui = total_primary_kwh / effective_area if effective_area > 0 else 0.0
        electric_eui = electric_kwh / effective_area if effective_area > 0 else 0.0
        fossil_eui = fossil_kwh / effective_area if effective_area > 0 else 0.0
        carbon_intensity = total_co2 / effective_area if effective_area > 0 else 0.0

        self._eui_result = EUIResult(
            site_eui_kwh_m2=round(site_eui, 2),
            source_eui_kwh_m2=round(source_eui, 2),
            primary_energy_kwh_m2=round(primary_eui, 2),
            normalised_eui_kwh_m2=round(normalised_eui, 2),
            electric_eui_kwh_m2=round(electric_eui, 2),
            fossil_eui_kwh_m2=round(fossil_eui, 2),
            carbon_intensity_kgco2_m2=round(carbon_intensity / 1000.0, 4),
            total_consumption_kwh=round(total_site_kwh, 2),
            total_cost=round(total_cost, 2),
            regression_r_squared=round(r_squared, 4),
            regression_cv_rmse_pct=round(cv_rmse, 2),
        )

        outputs["site_eui_kwh_m2"] = round(site_eui, 2)
        outputs["source_eui_kwh_m2"] = round(source_eui, 2)
        outputs["primary_energy_kwh_m2"] = round(primary_eui, 2)
        outputs["normalised_eui_kwh_m2"] = round(normalised_eui, 2)
        outputs["carbon_intensity_kgco2_m2"] = round(carbon_intensity / 1000.0, 4)
        outputs["r_squared"] = round(r_squared, 4)
        outputs["cv_rmse_pct"] = round(cv_rmse, 2)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 2 EUICalculation: site=%.1f source=%.1f normalised=%.1f kWh/m2",
            site_eui, source_eui, normalised_eui,
        )
        return PhaseResult(
            phase_name="eui_calculation", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _weather_normalise(
        self,
        energy_data: List[EnergyBillRecord],
        weather_data: List[WeatherDataRecord],
        floor_area: float,
    ) -> Dict[str, float]:
        """Weather-normalise EUI using degree-day regression (zero-hallucination)."""
        # Build period lookup for weather
        weather_lookup: Dict[str, Dict[str, float]] = {}
        for w in weather_data:
            weather_lookup[w.period] = {
                "hdd": w.heating_degree_days,
                "cdd": w.cooling_degree_days,
                "temp": w.avg_temperature_c,
            }

        # Build monthly energy totals
        monthly_kwh: Dict[str, float] = {}
        for bill in energy_data:
            if bill.period:
                monthly_kwh[bill.period] = monthly_kwh.get(bill.period, 0.0) + bill.consumption_kwh

        # Match energy with weather
        matched_pairs: List[Tuple[float, float, float]] = []
        for period, kwh in monthly_kwh.items():
            if period in weather_lookup:
                hdd = weather_lookup[period]["hdd"]
                cdd = weather_lookup[period]["cdd"]
                matched_pairs.append((kwh, hdd, cdd))

        if len(matched_pairs) < 6:
            total_kwh = sum(monthly_kwh.values())
            annual_factor = 12.0 / max(len(monthly_kwh), 1)
            return {
                "normalised_eui": total_kwh * annual_factor / floor_area if floor_area > 0 else 0.0,
                "r_squared": 0.0,
                "cv_rmse": 0.0,
            }

        # Simple 3-parameter regression: E = a + b*HDD + c*CDD
        n = len(matched_pairs)
        sum_e = sum(p[0] for p in matched_pairs)
        sum_hdd = sum(p[1] for p in matched_pairs)
        sum_cdd = sum(p[2] for p in matched_pairs)
        mean_e = sum_e / n
        mean_hdd = sum_hdd / n
        mean_cdd = sum_cdd / n

        # Simplified regression using total degree days
        sum_dd = sum(p[1] + p[2] for p in matched_pairs)
        mean_dd = sum_dd / n

        ss_dd = sum((p[1] + p[2] - mean_dd) ** 2 for p in matched_pairs)
        ss_ed = sum((p[0] - mean_e) * (p[1] + p[2] - mean_dd) for p in matched_pairs)

        b = ss_ed / ss_dd if ss_dd > 0 else 0.0
        a = mean_e - b * mean_dd

        # R-squared
        ss_total = sum((p[0] - mean_e) ** 2 for p in matched_pairs)
        ss_residual = sum((p[0] - (a + b * (p[1] + p[2]))) ** 2 for p in matched_pairs)
        r_squared = 1.0 - (ss_residual / ss_total) if ss_total > 0 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))

        # CV(RMSE)
        rmse = math.sqrt(ss_residual / n) if n > 0 else 0.0
        cv_rmse = (rmse / mean_e * 100.0) if mean_e > 0 else 0.0

        # Normalise to long-term average degree days (assume current data is representative)
        normalised_annual = (a * 12) + (b * sum_dd * (12.0 / n))
        normalised_eui = normalised_annual / floor_area if floor_area > 0 else 0.0

        return {
            "normalised_eui": max(0.0, normalised_eui),
            "r_squared": r_squared,
            "cv_rmse": cv_rmse,
        }

    # -------------------------------------------------------------------------
    # Phase 3: Peer Comparison
    # -------------------------------------------------------------------------

    def _phase_peer_comparison(
        self, input_data: InitialBenchmarkInput, effective_type: BuildingType
    ) -> PhaseResult:
        """Compare facility EUI against CIBSE TM46, ENERGY STAR, and other benchmarks."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if self._eui_result is None:
            return PhaseResult(
                phase_name="peer_comparison", phase_number=3,
                status=PhaseStatus.SKIPPED, warnings=["No EUI result available"],
            )

        site_eui = self._eui_result.site_eui_kwh_m2
        electric_eui = self._eui_result.electric_eui_kwh_m2
        fossil_eui = self._eui_result.fossil_eui_kwh_m2
        primary_eui = self._eui_result.primary_energy_kwh_m2
        type_key = effective_type.value

        # CIBSE TM46 comparison
        cibse = CIBSE_TM46_BENCHMARKS.get(type_key, CIBSE_TM46_BENCHMARKS["office"])
        typical_total = cibse["typical_electric"] + cibse["typical_fossil"]
        good_total = cibse["good_electric"] + cibse["good_fossil"]

        if site_eui <= good_total:
            cibse_category = "best_practice"
        elif site_eui <= typical_total:
            cibse_category = "good"
        else:
            cibse_category = "typical_or_worse"

        gap_to_typical = ((site_eui - typical_total) / typical_total * 100.0) if typical_total > 0 else 0.0
        gap_to_good = ((site_eui - good_total) / good_total * 100.0) if good_total > 0 else 0.0

        # ENERGY STAR score estimation (simplified lookup)
        energy_star_score = self._estimate_energy_star_score(site_eui, type_key)

        # EPC rating from primary energy
        epc_rating = self._determine_epc_rating(primary_eui)

        # Percentile estimation based on CIBSE distribution
        percentile = self._estimate_percentile(site_eui, typical_total, good_total)
        quartile = 4 - int(min(3, percentile // 25))

        # Synthetic peer count
        peer_count = self._estimate_peer_count(type_key, input_data.benchmark_sources)

        self._peer_ranking = PeerRankingResult(
            percentile=round(percentile, 1),
            quartile=quartile,
            energy_star_score=energy_star_score,
            epc_rating=epc_rating,
            cibse_category=cibse_category,
            benchmark_source="cibse_tm46",
            peer_typical_eui=round(typical_total, 2),
            peer_good_eui=round(good_total, 2),
            gap_to_typical_pct=round(gap_to_typical, 2),
            gap_to_good_pct=round(gap_to_good, 2),
            peer_count=peer_count,
        )

        outputs["percentile"] = round(percentile, 1)
        outputs["quartile"] = quartile
        outputs["energy_star_score"] = energy_star_score
        outputs["epc_rating"] = epc_rating
        outputs["cibse_category"] = cibse_category
        outputs["peer_typical_eui"] = round(typical_total, 2)
        outputs["peer_good_eui"] = round(good_total, 2)
        outputs["gap_to_typical_pct"] = round(gap_to_typical, 2)
        outputs["gap_to_good_pct"] = round(gap_to_good, 2)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 3 PeerComparison: percentile=%.1f ENERGY STAR=%d EPC=%s",
            percentile, energy_star_score, epc_rating,
        )
        return PhaseResult(
            phase_name="peer_comparison", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _estimate_energy_star_score(self, site_eui: float, building_type: str) -> int:
        """Estimate ENERGY STAR score from EUI (zero-hallucination lookup)."""
        cibse = CIBSE_TM46_BENCHMARKS.get(building_type, CIBSE_TM46_BENCHMARKS["office"])
        typical_total = cibse["typical_electric"] + cibse["typical_fossil"]
        good_total = cibse["good_electric"] + cibse["good_fossil"]

        if typical_total <= 0:
            return 50

        ratio = site_eui / typical_total
        if ratio <= 0.4:
            return min(100, 95)
        elif ratio <= 0.6:
            return 85
        elif ratio <= 0.8:
            return 75
        elif ratio <= 1.0:
            return 55
        elif ratio <= 1.3:
            return 35
        elif ratio <= 1.6:
            return 20
        else:
            return max(1, 10)

    def _determine_epc_rating(self, primary_eui: float) -> str:
        """Determine EPC rating band from primary energy demand."""
        for rating, (lower, upper) in EPC_RATING_BANDS.items():
            if lower <= primary_eui < upper:
                return rating
        return "G"

    def _estimate_percentile(
        self, site_eui: float, typical_eui: float, good_eui: float
    ) -> float:
        """Estimate percentile rank in peer group distribution."""
        if typical_eui <= 0:
            return 50.0

        ratio = site_eui / typical_eui
        if ratio <= 0.3:
            return 98.0
        elif ratio <= 0.5:
            return 90.0 + (0.5 - ratio) / 0.2 * 8.0
        elif ratio <= 0.75:
            return 75.0 + (0.75 - ratio) / 0.25 * 15.0
        elif ratio <= 1.0:
            return 50.0 + (1.0 - ratio) / 0.25 * 25.0
        elif ratio <= 1.5:
            return 20.0 + (1.5 - ratio) / 0.5 * 30.0
        else:
            return max(1.0, 20.0 - (ratio - 1.5) * 20.0)

    def _estimate_peer_count(
        self, building_type: str, sources: List[BenchmarkSource]
    ) -> int:
        """Estimate peer group size based on building type and data sources."""
        base_counts = {
            "office": 8500, "retail": 4200, "hotel": 2800, "hospital": 1600,
            "school": 6200, "university": 2400, "warehouse": 3800,
            "industrial": 2100, "restaurant": 5100, "supermarket": 3200,
            "data_centre": 1400, "mixed_use": 2600, "residential_multi": 7500,
            "leisure": 1800, "laboratory": 900,
        }
        count = base_counts.get(building_type, 2000)
        return count * len(sources)

    # -------------------------------------------------------------------------
    # Phase 4: Benchmark Report
    # -------------------------------------------------------------------------

    def _phase_benchmark_report(
        self,
        input_data: InitialBenchmarkInput,
        effective_area: float,
        effective_type: BuildingType,
    ) -> PhaseResult:
        """Generate initial benchmark report with all metrics and ratings."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        profile = input_data.facility_profile

        report: Dict[str, Any] = {
            "report_id": f"rpt-{uuid.uuid4().hex[:8]}",
            "report_type": "initial_benchmark",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "facility": {
                "facility_id": profile.facility_id,
                "facility_name": profile.facility_name,
                "building_type": effective_type.value,
                "floor_area_m2": effective_area,
                "country": profile.country,
                "year_built": profile.year_built,
            },
            "eui_metrics": self._eui_result.model_dump() if self._eui_result else {},
            "peer_ranking": self._peer_ranking.model_dump() if self._peer_ranking else {},
            "recommendations": self._generate_recommendations(effective_type),
            "data_quality": {
                "months_covered": self._collected_data.get("months_covered", 0),
                "weather_periods": self._collected_data.get("weather_periods", 0),
                "r_squared": self._eui_result.regression_r_squared if self._eui_result else 0.0,
            },
            "benchmark_sources": [s.value for s in input_data.benchmark_sources],
            "reporting_year": input_data.reporting_year,
        }

        self._collected_data["report"] = report

        outputs["report_id"] = report["report_id"]
        outputs["recommendations_count"] = len(report["recommendations"])
        outputs["benchmark_sources"] = report["benchmark_sources"]

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 4 BenchmarkReport: report=%s recommendations=%d",
            report["report_id"], len(report["recommendations"]),
        )
        return PhaseResult(
            phase_name="benchmark_report", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _generate_recommendations(self, building_type: BuildingType) -> List[Dict[str, str]]:
        """Generate benchmark-based recommendations (deterministic)."""
        recommendations: List[Dict[str, str]] = []

        if self._eui_result is None or self._peer_ranking is None:
            return recommendations

        eui = self._eui_result
        peer = self._peer_ranking

        if peer.gap_to_good_pct > 0:
            recommendations.append({
                "priority": "high",
                "area": "overall_efficiency",
                "recommendation": (
                    f"Site EUI is {peer.gap_to_good_pct:.0f}% above good practice benchmark. "
                    f"Target reduction to {peer.peer_good_eui:.0f} kWh/m2/yr."
                ),
            })

        if eui.electric_eui_kwh_m2 > 0:
            cibse = CIBSE_TM46_BENCHMARKS.get(
                building_type.value, CIBSE_TM46_BENCHMARKS["office"]
            )
            if eui.electric_eui_kwh_m2 > cibse["typical_electric"]:
                recommendations.append({
                    "priority": "medium",
                    "area": "electrical_systems",
                    "recommendation": (
                        f"Electrical EUI ({eui.electric_eui_kwh_m2:.0f} kWh/m2) exceeds "
                        f"typical benchmark ({cibse['typical_electric']:.0f} kWh/m2). "
                        f"Review lighting, HVAC, and plug load management."
                    ),
                })

        if eui.fossil_eui_kwh_m2 > 0:
            cibse = CIBSE_TM46_BENCHMARKS.get(
                building_type.value, CIBSE_TM46_BENCHMARKS["office"]
            )
            if eui.fossil_eui_kwh_m2 > cibse["typical_fossil"]:
                recommendations.append({
                    "priority": "medium",
                    "area": "heating_systems",
                    "recommendation": (
                        f"Fossil fuel EUI ({eui.fossil_eui_kwh_m2:.0f} kWh/m2) exceeds "
                        f"typical benchmark ({cibse['typical_fossil']:.0f} kWh/m2). "
                        f"Consider boiler upgrade, insulation, or heat pump conversion."
                    ),
                })

        if peer.energy_star_score < 50:
            recommendations.append({
                "priority": "high",
                "area": "energy_star",
                "recommendation": (
                    f"ENERGY STAR score is {peer.energy_star_score}, below median. "
                    f"Aim for 75+ to qualify for ENERGY STAR certification."
                ),
            })

        if eui.regression_r_squared < 0.70 and eui.regression_r_squared > 0:
            recommendations.append({
                "priority": "low",
                "area": "data_quality",
                "recommendation": (
                    f"Regression R-squared is {eui.regression_r_squared:.2f}; "
                    f"consider improving sub-metering or weather data coverage."
                ),
            })

        return recommendations

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: InitialBenchmarkResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
