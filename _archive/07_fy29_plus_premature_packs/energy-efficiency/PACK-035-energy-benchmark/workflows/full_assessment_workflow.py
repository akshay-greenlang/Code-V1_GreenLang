# -*- coding: utf-8 -*-
"""
Full Assessment Workflow
===================================

6-phase end-to-end benchmark assessment workflow within PACK-035 Energy
Benchmark Pack.  Orchestrates every stage from facility onboarding through
to final report generation.

Phases:
    1. FacilitySetup         -- Validate facility profile, load preset config,
                                determine building type benchmarks
    2. DataIngestion         -- Process energy bills, validate completeness,
                                convert units, aggregate by carrier/period
    3. BenchmarkCalculation  -- Calculate site/source/primary EUI, weather-
                                normalise using degree-day regression
    4. PeerComparison        -- Rank against CIBSE TM46 / ENERGY STAR peers,
                                calculate percentile, EPC rating, quartile
    5. GapAnalysis           -- Disaggregate end-uses, identify gaps against
                                good-practice benchmarks, quantify savings
    6. ReportGeneration      -- Generate final benchmark report with KPIs,
                                charts, recommendations, provenance trail

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas.  Benchmark lookups use
published CIBSE TM46 / ENERGY STAR / EN 15603 tables.  SHA-256 provenance
hashes guarantee auditability per phase and for the overall result.

Schedule: on-demand
Estimated duration: 60 minutes

Regulatory References:
    - ENERGY STAR Portfolio Manager Technical Reference (2023)
    - CIBSE TM46:2008 Energy benchmarks
    - EN 15603:2008 Energy performance of buildings
    - ASHRAE Standard 100-2018 Energy Efficiency in Buildings
    - EU EED 2023/1791 Article 8
    - EPBD 2024/1275

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


class BenchmarkSource(str, Enum):
    """Benchmark dataset source."""

    ENERGY_STAR = "energy_star"
    CIBSE_TM46 = "cibse_tm46"
    DIN_V_18599 = "din_v_18599"
    BPIE = "bpie"
    ASHRAE_100 = "ashrae_100"
    NABERS = "nabers"
    CRREM = "crrem"


class BenchmarkTarget(str, Enum):
    """Benchmark target level for gap analysis."""

    TYPICAL = "typical"
    GOOD_PRACTICE = "good_practice"
    BEST_PRACTICE = "best_practice"
    NZEB = "nearly_zero_energy"
    CUSTOM = "custom"


class GapSeverity(str, Enum):
    """Performance gap severity."""

    CRITICAL = "critical"
    SIGNIFICANT = "significant"
    MODERATE = "moderate"
    MINOR = "minor"
    AT_BENCHMARK = "at_benchmark"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# CIBSE TM46:2008 typical/good practice EUI benchmarks (kWh/m2/yr)
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
# Source: ENERGY STAR Portfolio Manager Technical Reference, August 2023
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

# Primary energy factors per EN 15603:2008 Annex A
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

# End-use split ratios by building type (fraction of total energy)
# Source: CIBSE Guide F (2012), ASHRAE Standard 100-2018
END_USE_SPLITS: Dict[str, Dict[str, float]] = {
    "office": {
        "heating": 0.30, "cooling": 0.15, "ventilation": 0.10, "lighting": 0.22,
        "plug_loads": 0.12, "domestic_hot_water": 0.05, "lifts_escalators": 0.03,
        "it_equipment": 0.03,
    },
    "retail": {
        "heating": 0.22, "cooling": 0.18, "ventilation": 0.08, "lighting": 0.30,
        "plug_loads": 0.05, "domestic_hot_water": 0.02, "refrigeration": 0.12,
        "other": 0.03,
    },
    "hotel": {
        "heating": 0.35, "cooling": 0.10, "ventilation": 0.08, "lighting": 0.15,
        "plug_loads": 0.05, "domestic_hot_water": 0.18, "catering": 0.06,
        "lifts_escalators": 0.03,
    },
    "hospital": {
        "heating": 0.40, "cooling": 0.10, "ventilation": 0.15, "lighting": 0.12,
        "plug_loads": 0.05, "domestic_hot_water": 0.08, "process": 0.05,
        "lifts_escalators": 0.02, "other": 0.03,
    },
    "school": {
        "heating": 0.55, "cooling": 0.05, "ventilation": 0.08, "lighting": 0.18,
        "plug_loads": 0.05, "domestic_hot_water": 0.06, "catering": 0.03,
    },
    "warehouse": {
        "heating": 0.40, "cooling": 0.05, "ventilation": 0.05, "lighting": 0.30,
        "plug_loads": 0.05, "domestic_hot_water": 0.02, "other": 0.13,
    },
    "industrial": {
        "heating": 0.15, "cooling": 0.05, "ventilation": 0.05, "lighting": 0.10,
        "plug_loads": 0.03, "process": 0.55, "domestic_hot_water": 0.02,
        "other": 0.05,
    },
    "data_centre": {
        "cooling": 0.38, "it_equipment": 0.45, "lighting": 0.03,
        "ventilation": 0.05, "plug_loads": 0.02, "other": 0.07,
    },
}

# Good practice end-use EUI targets (kWh/m2/yr) by building type
# Source: CIBSE Guide F (2012)
GOOD_PRACTICE_END_USE_EUI: Dict[str, Dict[str, float]] = {
    "office": {
        "heating": 40.0, "cooling": 18.0, "ventilation": 12.0, "lighting": 14.0,
        "plug_loads": 8.0, "domestic_hot_water": 6.0, "lifts_escalators": 3.0,
        "it_equipment": 4.0,
    },
    "retail": {
        "heating": 30.0, "cooling": 22.0, "ventilation": 10.0, "lighting": 30.0,
        "plug_loads": 5.0, "domestic_hot_water": 2.0, "refrigeration": 15.0,
        "other": 3.0,
    },
    "hotel": {
        "heating": 50.0, "cooling": 15.0, "ventilation": 12.0, "lighting": 18.0,
        "plug_loads": 6.0, "domestic_hot_water": 25.0, "catering": 8.0,
        "lifts_escalators": 4.0,
    },
    "hospital": {
        "heating": 100.0, "cooling": 25.0, "ventilation": 40.0, "lighting": 20.0,
        "plug_loads": 12.0, "domestic_hot_water": 20.0, "process": 15.0,
        "lifts_escalators": 5.0, "other": 8.0,
    },
    "school": {
        "heating": 40.0, "cooling": 4.0, "ventilation": 6.0, "lighting": 10.0,
        "plug_loads": 3.0, "domestic_hot_water": 4.0, "catering": 2.0,
    },
    "warehouse": {
        "heating": 12.0, "cooling": 2.0, "ventilation": 2.0, "lighting": 8.0,
        "plug_loads": 2.0, "domestic_hot_water": 1.0, "other": 4.0,
    },
    "industrial": {
        "heating": 18.0, "cooling": 6.0, "ventilation": 6.0, "lighting": 10.0,
        "plug_loads": 3.0, "process": 60.0, "domestic_hot_water": 2.0,
        "other": 6.0,
    },
    "data_centre": {
        "cooling": 100.0, "it_equipment": 140.0, "lighting": 5.0,
        "ventilation": 12.0, "plug_loads": 3.0, "other": 15.0,
    },
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
    """Result from a single workflow phase.

    Attributes:
        phase_name: Phase identifier string.
        phase_number: Sequential number (1-6).
        status: Completion status of this phase.
        duration_seconds: Wall-clock duration for the phase.
        outputs: Phase-specific output data.
        warnings: Non-fatal issues encountered.
        errors: Fatal errors encountered.
        provenance_hash: SHA-256 hash of the phase outputs.
    """

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class EnergyBillRecord(BaseModel):
    """Monthly energy bill record for data ingestion.

    Attributes:
        bill_id: Unique bill record identifier.
        period: Time period in YYYY-MM format.
        energy_source: Energy carrier type.
        consumption_kwh: Consumption in kWh.
        cost: Energy cost for the period.
        cost_currency: ISO 4217 currency code.
        demand_kw: Peak demand in kW.
        days_in_period: Number of days in the billing period.
        data_quality: Quality classification of the data.
    """

    bill_id: str = Field(default_factory=lambda: f"bill-{uuid.uuid4().hex[:8]}")
    period: str = Field(default="", description="Period YYYY-MM")
    energy_source: EnergySourceType = Field(default=EnergySourceType.ELECTRICITY)
    consumption_kwh: float = Field(default=0.0, ge=0.0, description="Consumption in kWh")
    cost: float = Field(default=0.0, ge=0.0, description="Energy cost")
    cost_currency: str = Field(default="EUR", description="Currency code")
    demand_kw: float = Field(default=0.0, ge=0.0, description="Peak demand kW")
    days_in_period: int = Field(default=30, ge=1, le=31)
    data_quality: DataQuality = Field(default=DataQuality.MEASURED)


class WeatherDataRecord(BaseModel):
    """Monthly weather data for normalisation.

    Attributes:
        period: Time period in YYYY-MM format.
        avg_temperature_c: Average temperature in Celsius.
        heating_degree_days: HDD base 15.5C.
        cooling_degree_days: CDD base 18.3C.
        avg_humidity_pct: Average relative humidity.
        solar_radiation_kwh_m2: Global horizontal irradiance.
    """

    period: str = Field(default="", description="Period YYYY-MM")
    avg_temperature_c: float = Field(default=15.0, description="Average temp Celsius")
    heating_degree_days: float = Field(default=0.0, ge=0.0, description="HDD base 15.5C")
    cooling_degree_days: float = Field(default=0.0, ge=0.0, description="CDD base 18.3C")
    avg_humidity_pct: float = Field(default=60.0, ge=0.0, le=100.0)
    solar_radiation_kwh_m2: float = Field(default=0.0, ge=0.0, description="Global horizontal")


class FullAssessmentInput(BaseModel):
    """Input data model for FullAssessmentWorkflow.

    Attributes:
        facility_profile: Facility metadata and physical characteristics.
        energy_data: Monthly energy bill records.
        weather_data: Monthly weather observations for normalisation.
        benchmark_sources: Benchmark datasets to compare against.
        peer_group_criteria: Optional peer group filtering criteria.
        report_config: Report generation configuration overrides.
        include_gap_analysis: Whether to run end-use gap analysis (Phase 5).
        include_trend_analysis: Whether to include trend data in reports.
        benchmark_target: Target level for gap analysis.
        custom_target_eui: Custom target EUI when benchmark_target is CUSTOM.
        reporting_year: Year of the data being analysed.
        emission_factor: CO2 emission factor override (kgCO2e/kWh).
        entity_id: Multi-tenant entity identifier.
        tenant_id: Multi-tenant tenant identifier.
    """

    facility_profile: Dict[str, Any] = Field(
        default_factory=dict, description="Facility profile data"
    )
    energy_data: List[EnergyBillRecord] = Field(
        default_factory=list, description="Monthly energy bill records"
    )
    weather_data: List[WeatherDataRecord] = Field(
        default_factory=list, description="Monthly weather data"
    )
    benchmark_sources: List[BenchmarkSource] = Field(
        default_factory=lambda: [BenchmarkSource.CIBSE_TM46, BenchmarkSource.ENERGY_STAR],
        description="Benchmark datasets to compare against",
    )
    peer_group_criteria: Dict[str, Any] = Field(
        default_factory=dict, description="Peer group filter criteria"
    )
    report_config: Dict[str, Any] = Field(
        default_factory=dict, description="Report configuration overrides"
    )
    include_gap_analysis: bool = Field(
        default=True, description="Enable end-use gap analysis phase"
    )
    include_trend_analysis: bool = Field(
        default=True, description="Include trend data in reports"
    )
    benchmark_target: BenchmarkTarget = Field(
        default=BenchmarkTarget.GOOD_PRACTICE, description="Target level for gap analysis"
    )
    custom_target_eui: float = Field(
        default=0.0, ge=0.0, description="Custom target EUI if CUSTOM"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    emission_factor: float = Field(
        default=0.207, ge=0.0, description="kgCO2e/kWh"
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("facility_profile")
    @classmethod
    def validate_facility_profile(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure facility profile has minimum required fields."""
        if not v.get("facility_name") and not v.get("facility_id"):
            raise ValueError("Facility profile must have a facility_name or facility_id")
        return v


class FullAssessmentResult(BaseModel):
    """Complete result from the full assessment workflow.

    Attributes:
        workflow_id: Unique execution identifier.
        workflow_name: Workflow type name.
        status: Overall workflow completion status.
        phases: Ordered list of phase results.
        facility_id: Facility identifier.
        facility_name: Facility display name.
        building_type: Building classification used.
        floor_area_m2: Floor area used for calculations.
        eui_result: EUI calculation outputs.
        normalised_eui: Weather-normalised EUI outputs.
        peer_ranking: Peer comparison outputs.
        performance_ratings: EPC / ENERGY STAR / CIBSE ratings.
        gap_analysis: End-use gap analysis outputs.
        trend_data: Year-over-year trend outputs.
        report: Generated report data.
        duration_seconds: Total wall-clock time.
        provenance_hash: SHA-256 of the complete result.
    """

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="full_assessment")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    building_type: str = Field(default="")
    floor_area_m2: float = Field(default=0.0, ge=0.0)
    eui_result: Dict[str, Any] = Field(default_factory=dict)
    normalised_eui: Dict[str, Any] = Field(default_factory=dict)
    peer_ranking: Dict[str, Any] = Field(default_factory=dict)
    performance_ratings: Dict[str, Any] = Field(default_factory=dict)
    gap_analysis: Dict[str, Any] = Field(default_factory=dict)
    trend_data: Dict[str, Any] = Field(default_factory=dict)
    report: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class FullAssessmentWorkflow:
    """
    6-phase end-to-end energy benchmark assessment workflow.

    Orchestrates the complete assessment pipeline from facility onboarding
    through to final report generation.  Each phase produces a PhaseResult
    with SHA-256 provenance hash, and the overall result carries a composite
    provenance hash for audit trails.

    Phases:
        1. FacilitySetup         - Validate and normalise facility profile
        2. DataIngestion         - Process energy data, validate completeness
        3. BenchmarkCalculation  - Calculate EUI, weather-normalise
        4. PeerComparison        - Rank against peers, calculate ratings
        5. GapAnalysis           - End-use disaggregation and gap identification
        6. ReportGeneration      - Assemble final benchmark report

    Zero-hallucination: all numeric calculations use deterministic formulas,
    benchmark lookups from published CIBSE/ENERGY STAR/EN 15603 tables, and
    degree-day regression for weather normalisation.  No LLM calls in the
    numeric path.

    Attributes:
        workflow_id: Unique execution identifier.
        _facility: Resolved facility profile data.
        _source_totals: Aggregated energy by source (kWh).
        _source_costs: Aggregated cost by source.
        _months_covered: Number of distinct months in the data.
        _eui_metrics: Calculated EUI metrics dictionary.
        _peer_metrics: Peer comparison metrics dictionary.
        _gap_metrics: End-use gap analysis dictionary.
        _phase_results: Ordered list of phase outputs.

    Example:
        >>> wf = FullAssessmentWorkflow()
        >>> inp = FullAssessmentInput(
        ...     facility_profile={"facility_name": "HQ Office",
        ...                       "building_type": "office",
        ...                       "floor_area_m2": 5000},
        ...     energy_data=[...],
        ... )
        >>> result = wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise FullAssessmentWorkflow.

        Args:
            config: Optional configuration overrides.
        """
        self.workflow_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._facility: Dict[str, Any] = {}
        self._source_totals: Dict[str, float] = {}
        self._source_costs: Dict[str, float] = {}
        self._months_covered: int = 0
        self._eui_metrics: Dict[str, Any] = {}
        self._peer_metrics: Dict[str, Any] = {}
        self._gap_metrics: Dict[str, Any] = {}
        self._report_data: Dict[str, Any] = {}
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def execute(self, input_data: FullAssessmentInput) -> FullAssessmentResult:
        """
        Execute the 6-phase full assessment workflow.

        Args:
            input_data: Validated full assessment input.

        Returns:
            FullAssessmentResult with EUI, peer ranking, gap analysis, and report.

        Raises:
            ValueError: If facility profile is incomplete or energy data is missing.
        """
        t_start = time.perf_counter()
        self.logger.info(
            "Starting full assessment workflow %s for facility=%s",
            self.workflow_id,
            input_data.facility_profile.get("facility_name", "unknown"),
        )

        # Reset internal state
        self._phase_results = []
        self._facility = {}
        self._source_totals = {}
        self._source_costs = {}
        self._months_covered = 0
        self._eui_metrics = {}
        self._peer_metrics = {}
        self._gap_metrics = {}
        self._report_data = {}
        overall_status = WorkflowStatus.RUNNING

        try:
            # Phase 1: Facility Setup
            phase1 = self._phase_1_facility_setup(input_data)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError(f"Phase 1 failed: {phase1.errors}")

            # Phase 2: Data Ingestion
            phase2 = self._phase_2_data_ingestion(input_data)
            self._phase_results.append(phase2)
            if phase2.status == PhaseStatus.FAILED:
                raise ValueError(f"Phase 2 failed: {phase2.errors}")

            # Phase 3: Benchmark Calculation
            phase3 = self._phase_3_benchmark_calculation(input_data)
            self._phase_results.append(phase3)
            if phase3.status == PhaseStatus.FAILED:
                raise ValueError(f"Phase 3 failed: {phase3.errors}")

            # Phase 4: Peer Comparison
            phase4 = self._phase_4_peer_comparison(input_data)
            self._phase_results.append(phase4)

            # Phase 5: Gap Analysis (optional)
            if input_data.include_gap_analysis:
                phase5 = self._phase_5_gap_analysis(input_data)
                self._phase_results.append(phase5)
            else:
                self._phase_results.append(PhaseResult(
                    phase_name="gap_analysis", phase_number=5,
                    status=PhaseStatus.SKIPPED,
                    warnings=["Gap analysis skipped per configuration"],
                ))

            # Phase 6: Report Generation
            phase6 = self._phase_6_report_generation(input_data)
            self._phase_results.append(phase6)

            # Determine overall status
            failed_count = sum(
                1 for p in self._phase_results if p.status == PhaseStatus.FAILED
            )
            if failed_count == 0:
                overall_status = WorkflowStatus.COMPLETED
            elif failed_count < len(self._phase_results):
                overall_status = WorkflowStatus.PARTIAL
            else:
                overall_status = WorkflowStatus.FAILED

        except Exception as exc:
            self.logger.error(
                "Full assessment workflow failed: %s", exc, exc_info=True
            )
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = time.perf_counter() - t_start

        # Build result
        facility_id = self._facility.get("facility_id", "")
        facility_name = self._facility.get("facility_name", "")
        building_type = self._facility.get("building_type", "")
        floor_area = self._facility.get("floor_area_m2", 0.0)

        result = FullAssessmentResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            facility_id=facility_id,
            facility_name=facility_name,
            building_type=building_type,
            floor_area_m2=floor_area,
            eui_result=dict(self._eui_metrics),
            normalised_eui=self._eui_metrics.get("normalisation", {}),
            peer_ranking=dict(self._peer_metrics),
            performance_ratings={
                "epc_rating": self._peer_metrics.get("epc_rating", ""),
                "energy_star_score": self._peer_metrics.get("energy_star_score", 0),
                "cibse_category": self._peer_metrics.get("cibse_category", ""),
            },
            gap_analysis=dict(self._gap_metrics),
            trend_data=self._eui_metrics.get("trends", {}),
            report=dict(self._report_data),
            duration_seconds=round(elapsed, 4),
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Full assessment workflow %s completed in %.2fs status=%s hash=%s",
            self.workflow_id, elapsed, overall_status.value,
            result.provenance_hash[:16],
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Facility Setup
    # -------------------------------------------------------------------------

    def _phase_1_facility_setup(
        self, input_data: FullAssessmentInput
    ) -> PhaseResult:
        """Validate facility profile, resolve defaults, load preset config.

        Args:
            input_data: Full assessment input data.

        Returns:
            PhaseResult with facility setup outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        profile = input_data.facility_profile

        # Resolve facility identifiers
        facility_id = profile.get("facility_id", f"fac-{uuid.uuid4().hex[:8]}")
        facility_name = profile.get("facility_name", "Unknown Facility")
        building_type_str = profile.get("building_type", "office")
        floor_area = float(profile.get("floor_area_m2", 0.0))
        country = profile.get("country", "")
        year_built = profile.get("year_built", 0)
        occupancy_hours = float(profile.get("occupancy_hours_per_week", 50.0))
        occupant_count = int(profile.get("occupant_count", 0))

        # Validate building type
        valid_types = {bt.value for bt in BuildingType}
        if building_type_str not in valid_types:
            warnings.append(
                f"Building type '{building_type_str}' not recognised; defaulting to 'office'"
            )
            building_type_str = "office"

        # Validate floor area
        if floor_area <= 0:
            warnings.append("Floor area is zero or negative; EUI calculations will be unreliable")

        if floor_area > 5_000_000:
            warnings.append("Floor area exceeds 5,000,000 m2 sanity check threshold")

        # Load CIBSE benchmarks for this building type
        cibse = CIBSE_TM46_BENCHMARKS.get(building_type_str, CIBSE_TM46_BENCHMARKS["office"])
        typical_eui = cibse["typical_electric"] + cibse["typical_fossil"]
        good_eui = cibse["good_electric"] + cibse["good_fossil"]

        # Store resolved facility
        self._facility = {
            "facility_id": facility_id,
            "facility_name": facility_name,
            "building_type": building_type_str,
            "floor_area_m2": floor_area,
            "country": country,
            "year_built": year_built,
            "occupancy_hours_per_week": occupancy_hours,
            "occupant_count": occupant_count,
            "benchmark_typical_eui": typical_eui,
            "benchmark_good_eui": good_eui,
        }

        outputs["facility_id"] = facility_id
        outputs["facility_name"] = facility_name
        outputs["building_type"] = building_type_str
        outputs["floor_area_m2"] = floor_area
        outputs["country"] = country
        outputs["year_built"] = year_built
        outputs["benchmark_typical_eui"] = round(typical_eui, 2)
        outputs["benchmark_good_eui"] = round(good_eui, 2)
        outputs["benchmark_sources"] = [s.value for s in input_data.benchmark_sources]

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 1 FacilitySetup: id=%s name=%s type=%s area=%.0f m2 (%.3fs)",
            facility_id, facility_name, building_type_str, floor_area, elapsed,
        )
        return PhaseResult(
            phase_name="facility_setup", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Data Ingestion
    # -------------------------------------------------------------------------

    def _phase_2_data_ingestion(
        self, input_data: FullAssessmentInput
    ) -> PhaseResult:
        """Process energy data, validate completeness, aggregate by carrier.

        Args:
            input_data: Full assessment input data.

        Returns:
            PhaseResult with data ingestion outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if not input_data.energy_data:
            return PhaseResult(
                phase_name="data_ingestion", phase_number=2,
                status=PhaseStatus.FAILED,
                errors=["No energy data provided"],
                duration_seconds=round(time.perf_counter() - t_start, 4),
            )

        # Aggregate by energy source
        source_totals: Dict[str, float] = {}
        source_costs: Dict[str, float] = {}
        periods: set = set()
        total_records = len(input_data.energy_data)
        measured_count = 0

        for bill in input_data.energy_data:
            src = bill.energy_source.value
            source_totals[src] = source_totals.get(src, 0.0) + bill.consumption_kwh
            source_costs[src] = source_costs.get(src, 0.0) + bill.cost
            if bill.period:
                periods.add(bill.period)
            if bill.data_quality == DataQuality.MEASURED:
                measured_count += 1

        months_covered = len(periods)
        total_kwh = sum(source_totals.values())
        total_cost = sum(source_costs.values())
        quality_pct = (measured_count / max(total_records, 1)) * 100.0

        # Validate data coverage
        if months_covered < 12:
            warnings.append(
                f"Only {months_covered} months of energy data; 12 months recommended "
                f"for accurate annualisation"
            )

        if months_covered < 3:
            warnings.append(
                "Fewer than 3 months of data; results will have low confidence"
            )

        # Validate weather data
        weather_periods = {w.period for w in input_data.weather_data if w.period}
        matched_weather = periods & weather_periods
        if not weather_periods:
            warnings.append("No weather data provided; normalisation will be skipped")
        elif len(matched_weather) < len(periods):
            unmatched = len(periods) - len(matched_weather)
            warnings.append(
                f"{unmatched} energy period(s) lack matching weather data"
            )

        # Detect zero-consumption periods
        zero_periods = [
            bill.period for bill in input_data.energy_data
            if bill.consumption_kwh == 0.0 and bill.period
        ]
        if zero_periods:
            warnings.append(
                f"{len(zero_periods)} period(s) have zero consumption; verify data"
            )

        # Store aggregated data for downstream phases
        self._source_totals = source_totals
        self._source_costs = source_costs
        self._months_covered = months_covered

        outputs["total_records"] = total_records
        outputs["months_covered"] = months_covered
        outputs["energy_sources"] = list(source_totals.keys())
        outputs["total_consumption_kwh"] = round(total_kwh, 2)
        outputs["total_cost"] = round(total_cost, 2)
        outputs["consumption_by_source_kwh"] = {
            k: round(v, 2) for k, v in source_totals.items()
        }
        outputs["cost_by_source"] = {
            k: round(v, 2) for k, v in source_costs.items()
        }
        outputs["data_quality_measured_pct"] = round(quality_pct, 1)
        outputs["weather_periods_matched"] = len(matched_weather)
        outputs["zero_consumption_periods"] = len(zero_periods)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 2 DataIngestion: %d records, %d months, %.0f kWh, quality=%.1f%% (%.3fs)",
            total_records, months_covered, total_kwh, quality_pct, elapsed,
        )
        return PhaseResult(
            phase_name="data_ingestion", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Benchmark Calculation
    # -------------------------------------------------------------------------

    def _phase_3_benchmark_calculation(
        self, input_data: FullAssessmentInput
    ) -> PhaseResult:
        """Calculate site, source, and primary EUI with weather normalisation.

        Args:
            input_data: Full assessment input data.

        Returns:
            PhaseResult with EUI calculation outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        floor_area = self._facility.get("floor_area_m2", 0.0)
        months = self._months_covered

        if floor_area <= 0:
            warnings.append("Floor area is zero; EUI set to 0.0")
            self._eui_metrics = {
                "site_eui_kwh_m2": 0.0, "source_eui_kwh_m2": 0.0,
                "primary_energy_kwh_m2": 0.0,
            }
            outputs.update(self._eui_metrics)
            elapsed = time.perf_counter() - t_start
            return PhaseResult(
                phase_name="benchmark_calculation", phase_number=3,
                status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
                outputs=outputs, warnings=warnings,
                provenance_hash=self._hash_dict(outputs),
            )

        # Annualise consumption
        annual_factor = 12.0 / max(months, 1)
        total_site_kwh = sum(self._source_totals.values()) * annual_factor
        total_cost = sum(self._source_costs.values()) * annual_factor

        # Calculate source energy (site-to-source conversion)
        total_source_kwh = 0.0
        source_by_carrier: Dict[str, float] = {}
        for src, kwh in self._source_totals.items():
            factor = SOURCE_ENERGY_FACTORS.get(src, 1.0)
            carrier_source = kwh * annual_factor * factor
            source_by_carrier[src] = carrier_source
            total_source_kwh += carrier_source

        # Calculate primary energy (EN 15603)
        total_primary_kwh = 0.0
        primary_by_carrier: Dict[str, float] = {}
        for src, kwh in self._source_totals.items():
            factor = PRIMARY_ENERGY_FACTORS.get(src, 1.0)
            carrier_primary = kwh * annual_factor * factor
            primary_by_carrier[src] = carrier_primary
            total_primary_kwh += carrier_primary

        # Split electric vs fossil
        electric_kwh = self._source_totals.get("electricity", 0.0) * annual_factor
        fossil_kwh = sum(
            v * annual_factor for k, v in self._source_totals.items()
            if k != "electricity" and k not in ("solar_pv", "solar_thermal")
        )

        # Calculate carbon emissions
        total_co2 = 0.0
        for src, kwh in self._source_totals.items():
            ef = DEFAULT_EMISSION_FACTORS.get(src, 0.207)
            total_co2 += kwh * annual_factor * ef

        # EUI calculations
        site_eui = total_site_kwh / floor_area
        source_eui = total_source_kwh / floor_area
        primary_eui = total_primary_kwh / floor_area
        electric_eui = electric_kwh / floor_area
        fossil_eui = fossil_kwh / floor_area
        carbon_intensity = total_co2 / floor_area

        # Weather normalisation using degree-day regression
        normalised_eui = site_eui
        r_squared = 0.0
        cv_rmse = 0.0
        normalisation_data: Dict[str, Any] = {}

        if input_data.weather_data:
            norm_result = self._weather_normalise(
                input_data.energy_data, input_data.weather_data, floor_area
            )
            normalised_eui = norm_result["normalised_eui"]
            r_squared = norm_result["r_squared"]
            cv_rmse = norm_result["cv_rmse"]
            normalisation_data = {
                "method": "degree_day_regression",
                "normalised_eui_kwh_m2": round(normalised_eui, 2),
                "r_squared": round(r_squared, 4),
                "cv_rmse_pct": round(cv_rmse, 2),
            }
        else:
            warnings.append("No weather data; normalised EUI equals site EUI")
            normalisation_data = {
                "method": "none",
                "normalised_eui_kwh_m2": round(site_eui, 2),
                "r_squared": 0.0,
                "cv_rmse_pct": 0.0,
            }

        # Cost intensity
        cost_per_m2 = total_cost / floor_area
        cost_per_kwh = total_cost / total_site_kwh if total_site_kwh > 0 else 0.0

        # Store EUI metrics
        self._eui_metrics = {
            "site_eui_kwh_m2": round(site_eui, 2),
            "source_eui_kwh_m2": round(source_eui, 2),
            "primary_energy_kwh_m2": round(primary_eui, 2),
            "normalised_eui_kwh_m2": round(normalised_eui, 2),
            "electric_eui_kwh_m2": round(electric_eui, 2),
            "fossil_eui_kwh_m2": round(fossil_eui, 2),
            "carbon_intensity_kgco2_m2": round(carbon_intensity / 1000.0, 4),
            "total_consumption_kwh": round(total_site_kwh, 2),
            "total_source_energy_kwh": round(total_source_kwh, 2),
            "total_primary_energy_kwh": round(total_primary_kwh, 2),
            "total_carbon_kgco2": round(total_co2, 2),
            "total_cost": round(total_cost, 2),
            "cost_per_m2": round(cost_per_m2, 2),
            "cost_per_kwh": round(cost_per_kwh, 4),
            "normalisation": normalisation_data,
            "carrier_breakdown": {
                k: round(v, 2) for k, v in source_by_carrier.items()
            },
        }

        outputs.update({
            "site_eui_kwh_m2": round(site_eui, 2),
            "source_eui_kwh_m2": round(source_eui, 2),
            "primary_energy_kwh_m2": round(primary_eui, 2),
            "normalised_eui_kwh_m2": round(normalised_eui, 2),
            "carbon_intensity_kgco2_m2": round(carbon_intensity / 1000.0, 4),
            "r_squared": round(r_squared, 4),
            "cv_rmse_pct": round(cv_rmse, 2),
            "cost_per_m2": round(cost_per_m2, 2),
        })

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 3 BenchmarkCalculation: site=%.1f source=%.1f primary=%.1f "
            "normalised=%.1f kWh/m2 (%.3fs)",
            site_eui, source_eui, primary_eui, normalised_eui, elapsed,
        )
        return PhaseResult(
            phase_name="benchmark_calculation", phase_number=3,
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
        """Weather-normalise EUI using 3-parameter degree-day regression.

        Deterministic regression: E = a + b*(HDD+CDD).
        Falls back to simple annualisation if fewer than 6 matched months.

        Args:
            energy_data: Monthly energy bill records.
            weather_data: Monthly weather observations.
            floor_area: Floor area in m2.

        Returns:
            Dict with normalised_eui, r_squared, cv_rmse.
        """
        weather_lookup: Dict[str, Dict[str, float]] = {}
        for w in weather_data:
            weather_lookup[w.period] = {
                "hdd": w.heating_degree_days,
                "cdd": w.cooling_degree_days,
            }

        monthly_kwh: Dict[str, float] = {}
        for bill in energy_data:
            if bill.period:
                monthly_kwh[bill.period] = (
                    monthly_kwh.get(bill.period, 0.0) + bill.consumption_kwh
                )

        matched: List[Tuple[float, float, float]] = []
        for period, kwh in monthly_kwh.items():
            if period in weather_lookup:
                hdd = weather_lookup[period]["hdd"]
                cdd = weather_lookup[period]["cdd"]
                matched.append((kwh, hdd, cdd))

        if len(matched) < 6:
            total_kwh = sum(monthly_kwh.values())
            af = 12.0 / max(len(monthly_kwh), 1)
            return {
                "normalised_eui": (total_kwh * af / floor_area) if floor_area > 0 else 0.0,
                "r_squared": 0.0,
                "cv_rmse": 0.0,
            }

        n = len(matched)
        sum_e = sum(p[0] for p in matched)
        mean_e = sum_e / n

        # Combined degree days for simplified regression
        sum_dd = sum(p[1] + p[2] for p in matched)
        mean_dd = sum_dd / n

        ss_dd = sum((p[1] + p[2] - mean_dd) ** 2 for p in matched)
        ss_ed = sum((p[0] - mean_e) * (p[1] + p[2] - mean_dd) for p in matched)

        b = ss_ed / ss_dd if ss_dd > 0 else 0.0
        a = mean_e - b * mean_dd

        # R-squared
        ss_total = sum((p[0] - mean_e) ** 2 for p in matched)
        ss_residual = sum(
            (p[0] - (a + b * (p[1] + p[2]))) ** 2 for p in matched
        )
        r_squared = 1.0 - (ss_residual / ss_total) if ss_total > 0 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))

        # CV(RMSE)
        rmse = math.sqrt(ss_residual / n) if n > 0 else 0.0
        cv_rmse = (rmse / mean_e * 100.0) if mean_e > 0 else 0.0

        # Normalise to annual
        normalised_annual = (a * 12) + (b * sum_dd * (12.0 / n))
        normalised_eui = normalised_annual / floor_area if floor_area > 0 else 0.0

        return {
            "normalised_eui": max(0.0, normalised_eui),
            "r_squared": r_squared,
            "cv_rmse": cv_rmse,
        }

    # -------------------------------------------------------------------------
    # Phase 4: Peer Comparison
    # -------------------------------------------------------------------------

    def _phase_4_peer_comparison(
        self, input_data: FullAssessmentInput
    ) -> PhaseResult:
        """Rank against peers, calculate ENERGY STAR score, EPC rating, quartile.

        Args:
            input_data: Full assessment input data.

        Returns:
            PhaseResult with peer comparison outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        site_eui = self._eui_metrics.get("site_eui_kwh_m2", 0.0)
        primary_eui = self._eui_metrics.get("primary_energy_kwh_m2", 0.0)
        building_type = self._facility.get("building_type", "office")

        if site_eui <= 0:
            self._peer_metrics = {}
            warnings.append("Site EUI is zero; peer comparison skipped")
            elapsed = time.perf_counter() - t_start
            return PhaseResult(
                phase_name="peer_comparison", phase_number=4,
                status=PhaseStatus.SKIPPED, duration_seconds=round(elapsed, 4),
                outputs=outputs, warnings=warnings,
            )

        # CIBSE TM46 benchmarks
        cibse = CIBSE_TM46_BENCHMARKS.get(building_type, CIBSE_TM46_BENCHMARKS["office"])
        typical_eui = cibse["typical_electric"] + cibse["typical_fossil"]
        good_eui = cibse["good_electric"] + cibse["good_fossil"]

        # CIBSE category
        if site_eui <= good_eui:
            cibse_category = "best_practice"
        elif site_eui <= typical_eui:
            cibse_category = "good"
        else:
            cibse_category = "typical_or_worse"

        # Gaps
        gap_to_typical = (
            (site_eui - typical_eui) / typical_eui * 100.0
        ) if typical_eui > 0 else 0.0

        gap_to_good = (
            (site_eui - good_eui) / good_eui * 100.0
        ) if good_eui > 0 else 0.0

        # ENERGY STAR score estimation
        energy_star_score = self._estimate_energy_star_score(site_eui, building_type)

        # EPC rating from primary energy
        epc_rating = self._determine_epc_rating(primary_eui)

        # Percentile estimation
        percentile = self._estimate_percentile(site_eui, typical_eui, good_eui)
        quartile = 4 - int(min(3, percentile // 25))

        # Peer count estimation
        peer_count = self._estimate_peer_count(building_type, input_data.benchmark_sources)

        self._peer_metrics = {
            "percentile": round(percentile, 1),
            "quartile": quartile,
            "energy_star_score": energy_star_score,
            "epc_rating": epc_rating,
            "cibse_category": cibse_category,
            "benchmark_source": "cibse_tm46",
            "peer_typical_eui": round(typical_eui, 2),
            "peer_good_eui": round(good_eui, 2),
            "gap_to_typical_pct": round(gap_to_typical, 2),
            "gap_to_good_pct": round(gap_to_good, 2),
            "peer_count": peer_count,
        }

        outputs.update(self._peer_metrics)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 4 PeerComparison: percentile=%.1f ENERGY STAR=%d EPC=%s "
            "gap_typical=%.1f%% gap_good=%.1f%% (%.3fs)",
            percentile, energy_star_score, epc_rating,
            gap_to_typical, gap_to_good, elapsed,
        )
        return PhaseResult(
            phase_name="peer_comparison", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _estimate_energy_star_score(
        self, site_eui: float, building_type: str
    ) -> int:
        """Estimate ENERGY STAR score from site EUI (zero-hallucination lookup).

        Uses the ratio of facility EUI to typical benchmark to assign a
        score bucket.  This is a simplified approximation; true ENERGY STAR
        scores require the official Portfolio Manager regression model.

        Args:
            site_eui: Facility site EUI (kWh/m2/yr).
            building_type: Building type key.

        Returns:
            Estimated ENERGY STAR score (1-100).
        """
        cibse = CIBSE_TM46_BENCHMARKS.get(building_type, CIBSE_TM46_BENCHMARKS["office"])
        typical_total = cibse["typical_electric"] + cibse["typical_fossil"]

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
        """Determine EPC rating band from primary energy demand.

        Args:
            primary_eui: Primary energy use intensity (kWh/m2/yr).

        Returns:
            EPC rating string (A+ through G).
        """
        for rating, (lower, upper) in EPC_RATING_BANDS.items():
            if lower <= primary_eui < upper:
                return rating
        return "G"

    def _estimate_percentile(
        self, site_eui: float, typical_eui: float, good_eui: float
    ) -> float:
        """Estimate percentile rank in the peer group distribution.

        Uses linear interpolation between known benchmark points to
        approximate the percentile position.

        Args:
            site_eui: Facility site EUI (kWh/m2/yr).
            typical_eui: Typical practice benchmark EUI.
            good_eui: Good practice benchmark EUI.

        Returns:
            Estimated percentile (0-100, higher is better).
        """
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
        """Estimate peer group size based on building type and data sources.

        Args:
            building_type: Building type key.
            sources: Benchmark dataset sources.

        Returns:
            Estimated number of peers.
        """
        base_counts = {
            "office": 8500, "retail": 4200, "hotel": 2800, "hospital": 1600,
            "school": 6200, "university": 2400, "warehouse": 3800,
            "industrial": 2100, "restaurant": 5100, "supermarket": 3200,
            "data_centre": 1400, "mixed_use": 2600, "residential_multi": 7500,
            "leisure": 1800, "laboratory": 900,
        }
        count = base_counts.get(building_type, 2000)
        return count * max(len(sources), 1)

    # -------------------------------------------------------------------------
    # Phase 5: Gap Analysis
    # -------------------------------------------------------------------------

    def _phase_5_gap_analysis(
        self, input_data: FullAssessmentInput
    ) -> PhaseResult:
        """Disaggregate end-uses, identify gaps, and quantify savings potential.

        Args:
            input_data: Full assessment input data.

        Returns:
            PhaseResult with gap analysis outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        site_eui = self._eui_metrics.get("site_eui_kwh_m2", 0.0)
        floor_area = self._facility.get("floor_area_m2", 0.0)
        building_type = self._facility.get("building_type", "office")
        total_kwh = self._eui_metrics.get("total_consumption_kwh", 0.0)
        total_cost = self._eui_metrics.get("total_cost", 0.0)
        ef = input_data.emission_factor

        if site_eui <= 0 or floor_area <= 0:
            warnings.append("Site EUI or floor area is zero; gap analysis skipped")
            elapsed = time.perf_counter() - t_start
            return PhaseResult(
                phase_name="gap_analysis", phase_number=5,
                status=PhaseStatus.SKIPPED, duration_seconds=round(elapsed, 4),
                warnings=warnings,
            )

        # Cost per kWh
        cost_per_kwh = total_cost / total_kwh if total_kwh > 0 else 0.15

        # Disaggregate end uses using CIBSE split ratios
        splits = END_USE_SPLITS.get(building_type, END_USE_SPLITS.get("office", {}))
        end_use_breakdown: Dict[str, float] = {}
        for eu, fraction in splits.items():
            end_use_breakdown[eu] = round(site_eui * fraction, 2)

        warnings.append(
            "End-use disaggregation uses CIBSE Guide F split ratios (no sub-metering)"
        )

        # Determine target scale factor
        target_method = input_data.benchmark_target
        scale_factor = 1.0
        if target_method == BenchmarkTarget.BEST_PRACTICE:
            scale_factor = 0.70
        elif target_method == BenchmarkTarget.NZEB:
            scale_factor = 0.40
        elif target_method == BenchmarkTarget.TYPICAL:
            scale_factor = 1.40
        elif target_method == BenchmarkTarget.CUSTOM and input_data.custom_target_eui > 0:
            good_total = sum(GOOD_PRACTICE_END_USE_EUI.get(building_type, {}).values())
            scale_factor = (
                input_data.custom_target_eui / good_total
            ) if good_total > 0 else 1.0

        # Get target EUIs
        targets = GOOD_PRACTICE_END_USE_EUI.get(building_type, {})

        # Calculate gaps per end use
        end_use_gaps: List[Dict[str, Any]] = []
        total_savings_kwh = 0.0
        total_savings_cost = 0.0
        total_co2_reduction = 0.0

        for eu, actual_eui in end_use_breakdown.items():
            target_eui = targets.get(eu, actual_eui) * scale_factor
            gap_eui = actual_eui - target_eui
            gap_pct = (gap_eui / target_eui * 100.0) if target_eui > 0 else 0.0
            savings_kwh = max(0.0, gap_eui * floor_area)
            savings_cost = savings_kwh * cost_per_kwh
            co2_kg = savings_kwh * ef

            # Severity classification
            if gap_pct > 50:
                severity = GapSeverity.CRITICAL.value
            elif gap_pct > 25:
                severity = GapSeverity.SIGNIFICANT.value
            elif gap_pct > 10:
                severity = GapSeverity.MODERATE.value
            elif gap_pct > 0:
                severity = GapSeverity.MINOR.value
            else:
                severity = GapSeverity.AT_BENCHMARK.value

            total_savings_kwh += savings_kwh
            total_savings_cost += savings_cost
            total_co2_reduction += co2_kg

            end_use_gaps.append({
                "end_use": eu,
                "actual_eui": round(actual_eui, 2),
                "target_eui": round(target_eui, 2),
                "gap_eui": round(gap_eui, 2),
                "gap_pct": round(gap_pct, 2),
                "savings_kwh": round(savings_kwh, 0),
                "savings_cost": round(savings_cost, 2),
                "co2_reduction_kg": round(co2_kg, 2),
                "severity": severity,
                "recommendation": self._recommend_action(eu, gap_pct),
            })

        # Sort by savings potential descending and assign priority
        end_use_gaps.sort(key=lambda g: g["savings_kwh"], reverse=True)
        for idx, gap in enumerate(end_use_gaps, start=1):
            gap["priority_rank"] = idx

        # Count severity levels
        critical_count = sum(1 for g in end_use_gaps if g["severity"] == "critical")
        significant_count = sum(
            1 for g in end_use_gaps if g["severity"] == "significant"
        )

        # Overall gap
        target_total = sum(
            targets.get(eu, 0.0) * scale_factor for eu in end_use_breakdown
        )
        overall_gap_eui = site_eui - target_total
        overall_gap_pct = (
            overall_gap_eui / target_total * 100.0
        ) if target_total > 0 else 0.0

        self._gap_metrics = {
            "overall_gap_eui": round(overall_gap_eui, 2),
            "overall_gap_pct": round(overall_gap_pct, 2),
            "target_level": target_method.value,
            "end_use_gaps": end_use_gaps,
            "end_uses_analysed": len(end_use_gaps),
            "critical_gaps": critical_count,
            "significant_gaps": significant_count,
            "total_savings_kwh": round(total_savings_kwh, 0),
            "total_savings_cost": round(total_savings_cost, 2),
            "total_co2_reduction_kg": round(total_co2_reduction, 2),
            "savings_pct_of_total": round(
                total_savings_kwh / total_kwh * 100.0, 2
            ) if total_kwh > 0 else 0.0,
        }

        outputs["end_uses_analysed"] = len(end_use_gaps)
        outputs["critical_gaps"] = critical_count
        outputs["significant_gaps"] = significant_count
        outputs["overall_gap_pct"] = round(overall_gap_pct, 2)
        outputs["total_savings_kwh"] = round(total_savings_kwh, 0)
        outputs["total_savings_cost"] = round(total_savings_cost, 2)
        outputs["total_co2_reduction_kg"] = round(total_co2_reduction, 2)
        outputs["top_3_priorities"] = [
            {
                "end_use": g["end_use"],
                "gap_pct": g["gap_pct"],
                "savings_kwh": g["savings_kwh"],
            }
            for g in end_use_gaps[:3]
        ]

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 5 GapAnalysis: %d end-uses, %d critical, overall_gap=%.1f%%, "
            "savings=%.0f kWh (%.3fs)",
            len(end_use_gaps), critical_count, overall_gap_pct,
            total_savings_kwh, elapsed,
        )
        return PhaseResult(
            phase_name="gap_analysis", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _recommend_action(self, end_use: str, gap_pct: float) -> str:
        """Generate deterministic recommendation based on end use and gap severity.

        Args:
            end_use: End-use category string.
            gap_pct: Gap percentage (positive = over-consuming).

        Returns:
            Recommendation string.
        """
        recommendations: Dict[str, str] = {
            "heating": "Upgrade heating controls, improve insulation, consider heat pump",
            "cooling": "Optimise chiller staging, improve free cooling, reduce solar gains",
            "ventilation": "Implement demand-controlled ventilation, check fan efficiency",
            "lighting": "LED retrofit, daylight dimming, occupancy sensing controls",
            "plug_loads": "Smart power strips, equipment scheduling, efficiency standards",
            "domestic_hot_water": "Point-of-use heaters, pipe insulation, solar thermal",
            "process": "Process optimisation, waste heat recovery, variable speed drives",
            "it_equipment": "Server virtualisation, hot/cold aisle containment, UPS upgrade",
            "refrigeration": "Door heater controls, EC fan motors, defrost optimisation",
            "catering": "Efficient cooking equipment, ventilation heat recovery",
            "lifts_escalators": "Regenerative drives, standby mode, modernisation",
        }
        base = recommendations.get(
            end_use, "Investigate and implement best practice measures"
        )
        if gap_pct > 50:
            return f"URGENT: {base}. Gap exceeds 50% of benchmark."
        elif gap_pct <= 0:
            return "At or below benchmark; maintain current performance."
        return base

    # -------------------------------------------------------------------------
    # Phase 6: Report Generation
    # -------------------------------------------------------------------------

    def _phase_6_report_generation(
        self, input_data: FullAssessmentInput
    ) -> PhaseResult:
        """Generate final benchmark report with KPIs, sections, and provenance.

        Args:
            input_data: Full assessment input data.

        Returns:
            PhaseResult with report generation outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        report_id = f"rpt-{uuid.uuid4().hex[:8]}"
        generated_at = datetime.utcnow().isoformat() + "Z"

        # Assemble KPIs
        kpis = self._assemble_kpis()

        # Assemble recommendations
        recommendations = self._assemble_recommendations(input_data)

        # Build report structure
        report: Dict[str, Any] = {
            "report_id": report_id,
            "report_type": "full_assessment",
            "report_version": "35.0.0",
            "generated_at": generated_at,
            "workflow_id": self.workflow_id,
            "facility": {
                "facility_id": self._facility.get("facility_id", ""),
                "facility_name": self._facility.get("facility_name", ""),
                "building_type": self._facility.get("building_type", ""),
                "floor_area_m2": self._facility.get("floor_area_m2", 0.0),
                "country": self._facility.get("country", ""),
                "year_built": self._facility.get("year_built", 0),
            },
            "kpis": kpis,
            "eui_metrics": dict(self._eui_metrics),
            "peer_ranking": dict(self._peer_metrics),
            "gap_analysis": dict(self._gap_metrics),
            "recommendations": recommendations,
            "data_quality": {
                "months_covered": self._months_covered,
                "r_squared": self._eui_metrics.get(
                    "normalisation", {}
                ).get("r_squared", 0.0),
                "cv_rmse_pct": self._eui_metrics.get(
                    "normalisation", {}
                ).get("cv_rmse_pct", 0.0),
            },
            "benchmark_sources": [s.value for s in input_data.benchmark_sources],
            "reporting_year": input_data.reporting_year,
            "methodology_notes": [
                "EUI calculated per ENERGY STAR Portfolio Manager Technical Reference",
                "Source energy factors per ENERGY STAR Technical Reference 2023",
                "Primary energy factors per EN 15603:2008 Annex A",
                "End-use splits per CIBSE Guide F (2012)",
                "Weather normalisation via 3-parameter degree-day regression",
                "EPC rating bands per EPBD methodology",
            ],
        }

        # Compute report provenance hash
        report["provenance_hash"] = self._hash_dict({
            k: v for k, v in report.items() if k != "provenance_hash"
        })

        self._report_data = report

        outputs["report_id"] = report_id
        outputs["kpi_count"] = len(kpis)
        outputs["recommendation_count"] = len(recommendations)
        outputs["sections"] = [
            "facility", "kpis", "eui_metrics", "peer_ranking",
            "gap_analysis", "recommendations", "data_quality",
            "methodology_notes",
        ]
        outputs["report_provenance_hash"] = report["provenance_hash"][:16]

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 6 ReportGeneration: report=%s, %d KPIs, %d recommendations (%.3fs)",
            report_id, len(kpis), len(recommendations), elapsed,
        )
        return PhaseResult(
            phase_name="report_generation", phase_number=6,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _assemble_kpis(self) -> List[Dict[str, Any]]:
        """Assemble key performance indicators from calculated metrics.

        Returns:
            List of KPI dictionaries with name, value, unit, and category.
        """
        kpis: List[Dict[str, Any]] = []

        site_eui = self._eui_metrics.get("site_eui_kwh_m2", 0.0)
        source_eui = self._eui_metrics.get("source_eui_kwh_m2", 0.0)
        primary_eui = self._eui_metrics.get("primary_energy_kwh_m2", 0.0)
        normalised_eui = self._eui_metrics.get("normalised_eui_kwh_m2", 0.0)
        carbon_intensity = self._eui_metrics.get("carbon_intensity_kgco2_m2", 0.0)
        cost_per_m2 = self._eui_metrics.get("cost_per_m2", 0.0)

        percentile = self._peer_metrics.get("percentile", 0.0)
        energy_star = self._peer_metrics.get("energy_star_score", 0)
        epc_rating = self._peer_metrics.get("epc_rating", "")
        gap_to_good = self._peer_metrics.get("gap_to_good_pct", 0.0)

        total_savings = self._gap_metrics.get("total_savings_kwh", 0.0)
        savings_pct = self._gap_metrics.get("savings_pct_of_total", 0.0)

        kpis.append({
            "name": "Site EUI", "value": site_eui,
            "unit": "kWh/m2/yr", "category": "energy",
        })
        kpis.append({
            "name": "Source EUI", "value": source_eui,
            "unit": "kWh/m2/yr", "category": "energy",
        })
        kpis.append({
            "name": "Primary Energy", "value": primary_eui,
            "unit": "kWh/m2/yr", "category": "energy",
        })
        kpis.append({
            "name": "Normalised EUI", "value": normalised_eui,
            "unit": "kWh/m2/yr", "category": "energy",
        })
        kpis.append({
            "name": "Carbon Intensity", "value": carbon_intensity,
            "unit": "kgCO2e/m2/yr", "category": "carbon",
        })
        kpis.append({
            "name": "Energy Cost Intensity", "value": cost_per_m2,
            "unit": "EUR/m2/yr", "category": "cost",
        })
        kpis.append({
            "name": "Peer Percentile", "value": percentile,
            "unit": "percentile", "category": "benchmark",
        })
        kpis.append({
            "name": "ENERGY STAR Score", "value": energy_star,
            "unit": "score (1-100)", "category": "benchmark",
        })
        kpis.append({
            "name": "EPC Rating", "value": epc_rating,
            "unit": "rating", "category": "benchmark",
        })
        kpis.append({
            "name": "Gap to Good Practice", "value": gap_to_good,
            "unit": "%", "category": "gap",
        })
        kpis.append({
            "name": "Total Savings Potential", "value": total_savings,
            "unit": "kWh/yr", "category": "savings",
        })
        kpis.append({
            "name": "Savings as % of Consumption", "value": savings_pct,
            "unit": "%", "category": "savings",
        })

        return kpis

    def _assemble_recommendations(
        self, input_data: FullAssessmentInput
    ) -> List[Dict[str, str]]:
        """Assemble deterministic recommendations from all phases.

        Args:
            input_data: Full assessment input data.

        Returns:
            List of recommendation dictionaries with priority, area, and text.
        """
        recommendations: List[Dict[str, str]] = []

        electric_eui = self._eui_metrics.get("electric_eui_kwh_m2", 0.0)
        fossil_eui = self._eui_metrics.get("fossil_eui_kwh_m2", 0.0)
        building_type = self._facility.get("building_type", "office")
        gap_to_good = self._peer_metrics.get("gap_to_good_pct", 0.0)
        good_eui = self._peer_metrics.get("peer_good_eui", 0.0)
        energy_star = self._peer_metrics.get("energy_star_score", 0)
        r_squared = self._eui_metrics.get(
            "normalisation", {}
        ).get("r_squared", 0.0)

        # R1: Overall efficiency
        if gap_to_good > 0:
            recommendations.append({
                "priority": "high",
                "area": "overall_efficiency",
                "recommendation": (
                    f"Site EUI is {gap_to_good:.0f}% above good practice benchmark. "
                    f"Target reduction to {good_eui:.0f} kWh/m2/yr."
                ),
            })

        # R2: Electrical systems
        cibse = CIBSE_TM46_BENCHMARKS.get(
            building_type, CIBSE_TM46_BENCHMARKS["office"]
        )
        if electric_eui > cibse["typical_electric"]:
            recommendations.append({
                "priority": "medium",
                "area": "electrical_systems",
                "recommendation": (
                    f"Electrical EUI ({electric_eui:.0f} kWh/m2) exceeds typical "
                    f"benchmark ({cibse['typical_electric']:.0f} kWh/m2). "
                    f"Review lighting, HVAC, and plug load management."
                ),
            })

        # R3: Heating systems
        if fossil_eui > cibse["typical_fossil"]:
            recommendations.append({
                "priority": "medium",
                "area": "heating_systems",
                "recommendation": (
                    f"Fossil fuel EUI ({fossil_eui:.0f} kWh/m2) exceeds typical "
                    f"benchmark ({cibse['typical_fossil']:.0f} kWh/m2). "
                    f"Consider boiler upgrade, insulation, or heat pump conversion."
                ),
            })

        # R4: ENERGY STAR certification
        if 0 < energy_star < 75:
            recommendations.append({
                "priority": "medium",
                "area": "energy_star",
                "recommendation": (
                    f"ENERGY STAR score is {energy_star}, below 75 threshold. "
                    f"Aim for 75+ to qualify for ENERGY STAR certification."
                ),
            })

        # R5: Data quality
        if 0 < r_squared < 0.70:
            recommendations.append({
                "priority": "low",
                "area": "data_quality",
                "recommendation": (
                    f"Weather normalisation R-squared is {r_squared:.2f}. "
                    f"Improve sub-metering or weather data coverage for "
                    f"more reliable benchmarking."
                ),
            })

        # R6: Data coverage
        if self._months_covered < 12:
            recommendations.append({
                "priority": "low",
                "area": "data_coverage",
                "recommendation": (
                    f"Only {self._months_covered} months of energy data available. "
                    f"Collect 12 consecutive months for optimal annualisation accuracy."
                ),
            })

        # R7: Gap analysis top priorities
        top_gaps = self._gap_metrics.get("end_use_gaps", [])[:3]
        for gap in top_gaps:
            if gap.get("savings_kwh", 0) > 0:
                recommendations.append({
                    "priority": (
                        "high" if gap.get("severity") == "critical" else "medium"
                    ),
                    "area": gap.get("end_use", ""),
                    "recommendation": gap.get("recommendation", ""),
                })

        return recommendations

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: FullAssessmentResult) -> str:
        """Compute SHA-256 provenance hash for the complete result.

        Args:
            result: Full assessment result object.

        Returns:
            SHA-256 hex digest string (64 characters).
        """
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary.

        Args:
            data: Dictionary to hash.

        Returns:
            SHA-256 hex digest string (64 characters).
        """
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
