# -*- coding: utf-8 -*-
"""
Store Emissions Workflow
===========================

4-phase workflow for store-level GHG emissions calculation within
PACK-014 CSRD Retail and Consumer Goods Pack.

Phases:
    1. DataCollection      -- Gather energy, refrigerant, and fleet data per store
    2. Scope1Calculation   -- Heating, refrigerant leakage, fleet emissions
    3. Scope2Calculation   -- Location-based and market-based electricity
    4. Consolidation       -- Multi-store rollup, KPIs (emissions/sqm, per employee)

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated emission factors.
SHA-256 provenance hashes guarantee auditability.

Author: GreenLang Team
Version: 14.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

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


class FuelType(str, Enum):
    """Store heating fuel types."""

    NATURAL_GAS = "natural_gas"
    LPG = "lpg"
    HEATING_OIL = "heating_oil"
    DIESEL = "diesel"
    PROPANE = "propane"


class RefrigerantType(str, Enum):
    """Common retail refrigerants."""

    R404A = "R-404A"
    R134A = "R-134a"
    R407C = "R-407C"
    R410A = "R-410A"
    R290 = "R-290"
    R744 = "R-744"
    R449A = "R-449A"
    R448A = "R-448A"


class VehicleFuelType(str, Enum):
    """Fleet vehicle fuel types."""

    DIESEL = "diesel"
    PETROL = "petrol"
    CNG = "cng"
    ELECTRIC = "electric"
    HYBRID = "hybrid"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class EnergyRecord(BaseModel):
    """Energy consumption record for a store."""

    fuel_type: str = Field(..., description="Fuel or energy type")
    consumption_kwh: float = Field(default=0.0, ge=0.0, description="Consumption in kWh")
    consumption_litres: float = Field(default=0.0, ge=0.0, description="Consumption in litres")
    consumption_m3: float = Field(default=0.0, ge=0.0, description="Consumption in cubic metres")
    unit: str = Field(default="kWh", description="Measurement unit")
    period_start: str = Field(default="", description="Period start YYYY-MM-DD")
    period_end: str = Field(default="", description="Period end YYYY-MM-DD")
    data_quality: str = Field(default="measured", description="measured|estimated|default")


class RefrigerantRecord(BaseModel):
    """Refrigerant charge and leakage data for a store."""

    refrigerant_type: str = Field(..., description="Refrigerant designation")
    charge_kg: float = Field(default=0.0, ge=0.0, description="Total charge in kg")
    leakage_kg: float = Field(default=0.0, ge=0.0, description="Annual leakage in kg")
    leakage_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    gwp: float = Field(default=0.0, ge=0.0, description="Global Warming Potential")
    equipment_type: str = Field(default="display_case", description="Equipment type")


class FleetRecord(BaseModel):
    """Fleet vehicle emissions data."""

    vehicle_id: str = Field(default="", description="Vehicle identifier")
    fuel_type: str = Field(default="diesel", description="Vehicle fuel type")
    distance_km: float = Field(default=0.0, ge=0.0, description="Distance driven in km")
    fuel_consumed_litres: float = Field(default=0.0, ge=0.0, description="Fuel consumed")
    vehicle_type: str = Field(default="van", description="Vehicle type")


class ElectricityRecord(BaseModel):
    """Electricity consumption record."""

    consumption_kwh: float = Field(default=0.0, ge=0.0, description="kWh consumed")
    grid_region: str = Field(default="EU-AVG", description="Grid emission factor region")
    renewable_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Renewable pct")
    supplier: str = Field(default="", description="Electricity supplier")
    has_guarantee_of_origin: bool = Field(default=False, description="Has GoO certificate")
    market_ef_kgco2_kwh: Optional[float] = Field(None, ge=0.0, description="Supplier EF")
    period_start: str = Field(default="", description="Period start")
    period_end: str = Field(default="", description="Period end")


class StoreData(BaseModel):
    """All environmental data for a single retail store."""

    store_id: str = Field(..., description="Unique store identifier")
    store_name: str = Field(default="", description="Human-readable store name")
    country: str = Field(default="", description="ISO 3166-1 alpha-2")
    region: str = Field(default="", description="Region or state")
    floor_area_sqm: float = Field(default=0.0, ge=0.0, description="Floor area in sqm")
    employee_count: int = Field(default=0, ge=0, description="Number of employees")
    store_type: str = Field(default="supermarket", description="Store type")
    energy_records: List[EnergyRecord] = Field(default_factory=list)
    refrigerant_records: List[RefrigerantRecord] = Field(default_factory=list)
    fleet_records: List[FleetRecord] = Field(default_factory=list)
    electricity_records: List[ElectricityRecord] = Field(default_factory=list)


class StoreEmissionsConfig(BaseModel):
    """Configuration for store emissions workflow."""

    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    include_fleet: bool = Field(default=True)
    include_refrigerants: bool = Field(default=True)
    consolidation_method: str = Field(default="operational_control")
    location_ef_source: str = Field(default="IEA_2024")
    market_ef_source: str = Field(default="AIB_residual_2024")
    gwp_assessment_report: str = Field(default="AR6", description="AR5 or AR6")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("consolidation_method")
    @classmethod
    def validate_consolidation(cls, v: str) -> str:
        """Validate consolidation approach."""
        allowed = {"operational_control", "financial_control", "equity_share"}
        if v not in allowed:
            raise ValueError(f"consolidation_method must be one of {allowed}")
        return v


class StoreEmissionBreakdown(BaseModel):
    """Emission breakdown for a single store."""

    store_id: str = Field(..., description="Store identifier")
    store_name: str = Field(default="")
    scope1_heating_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_refrigerant_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_fleet_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_total_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    total_tco2e: float = Field(default=0.0, ge=0.0)
    emissions_per_sqm: float = Field(default=0.0, ge=0.0)
    emissions_per_employee: float = Field(default=0.0, ge=0.0)
    floor_area_sqm: float = Field(default=0.0, ge=0.0)
    employee_count: int = Field(default=0, ge=0)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)


class ConsolidatedKPIs(BaseModel):
    """Consolidated KPIs across all stores."""

    total_scope1_tco2e: float = Field(default=0.0, ge=0.0)
    total_scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    total_scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    total_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    avg_emissions_per_sqm: float = Field(default=0.0, ge=0.0)
    avg_emissions_per_employee: float = Field(default=0.0, ge=0.0)
    total_floor_area_sqm: float = Field(default=0.0, ge=0.0)
    total_employees: int = Field(default=0, ge=0)
    store_count: int = Field(default=0, ge=0)
    highest_emitting_store: str = Field(default="")
    lowest_emitting_store: str = Field(default="")
    refrigerant_share_pct: float = Field(default=0.0)
    fleet_share_pct: float = Field(default=0.0)
    heating_share_pct: float = Field(default=0.0)
    renewable_electricity_pct: float = Field(default=0.0)


class StoreEmissionsInput(BaseModel):
    """Input data model for StoreEmissionsWorkflow."""

    stores: List[StoreData] = Field(..., min_length=1, description="List of stores")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    config: StoreEmissionsConfig = Field(default_factory=StoreEmissionsConfig)

    @field_validator("stores")
    @classmethod
    def validate_stores(cls, v: List[StoreData]) -> List[StoreData]:
        """Ensure store IDs are unique."""
        ids = [s.store_id for s in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate store IDs found")
        return v


class StoreEmissionsResult(BaseModel):
    """Complete result from store emissions workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="store_emissions")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    per_store_results: List[StoreEmissionBreakdown] = Field(default_factory=list)
    consolidated: ConsolidatedKPIs = Field(default_factory=ConsolidatedKPIs)
    scope1_total_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_total_tco2e: float = Field(default=0.0, ge=0.0)
    kpis: Dict[str, Any] = Field(default_factory=dict)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="", description="SHA-256 of complete output")


# =============================================================================
# EMISSION FACTOR CONSTANTS (Zero-Hallucination)
# =============================================================================

FUEL_EMISSION_FACTORS: Dict[str, float] = {
    "natural_gas": 0.18293,
    "lpg": 0.21448,
    "heating_oil": 0.24685,
    "diesel": 0.25301,
    "propane": 0.21448,
}

FUEL_LITRE_TO_KWH: Dict[str, float] = {
    "natural_gas": 10.55,
    "lpg": 7.08,
    "heating_oil": 10.35,
    "diesel": 10.0,
    "propane": 7.08,
}

REFRIGERANT_GWP_AR6: Dict[str, float] = {
    "R-404A": 4728.0, "R-134a": 1530.0, "R-407C": 1774.0, "R-410A": 2088.0,
    "R-290": 0.02, "R-744": 1.0, "R-449A": 1282.0, "R-448A": 1273.0,
}

REFRIGERANT_GWP_AR5: Dict[str, float] = {
    "R-404A": 3922.0, "R-134a": 1430.0, "R-407C": 1774.0, "R-410A": 2088.0,
    "R-290": 3.3, "R-744": 1.0, "R-449A": 1397.0, "R-448A": 1387.0,
}

VEHICLE_FUEL_EF: Dict[str, float] = {
    "diesel": 2.70494, "petrol": 2.31440, "cng": 2.53990,
    "electric": 0.0, "hybrid": 1.73160,
}

GRID_LOCATION_EF: Dict[str, float] = {
    "EU-AVG": 0.2556, "DE": 0.3850, "FR": 0.0520, "ES": 0.1480,
    "IT": 0.2580, "NL": 0.3280, "PL": 0.6340, "SE": 0.0130,
    "AT": 0.0890, "BE": 0.1340, "UK": 0.2070, "PT": 0.1740,
    "CZ": 0.3880, "RO": 0.2560, "HU": 0.2100, "DK": 0.0990,
    "FI": 0.0640, "IE": 0.2960, "BG": 0.3740, "GR": 0.3240,
}

GRID_MARKET_EF: Dict[str, float] = {
    "EU-AVG": 0.4200, "DE": 0.5810, "FR": 0.0560, "ES": 0.2680,
    "IT": 0.4370, "NL": 0.4750, "PL": 0.7020, "SE": 0.0210,
    "AT": 0.2410, "BE": 0.2050, "UK": 0.3420,
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class StoreEmissionsWorkflow:
    """
    4-phase store-level emissions calculation workflow.

    Computes Scope 1 (heating, refrigerant leakage, fleet) and Scope 2
    (location-based, market-based) emissions for each store in a retail
    portfolio, then consolidates into multi-store KPIs.

    Zero-hallucination: all calculations use deterministic emission factors
    from DEFRA 2024, IEA 2024, and AIB residual mix. No LLM calls in the
    numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.
        _store_breakdowns: Per-store emission results.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = StoreEmissionsWorkflow()
        >>> inp = StoreEmissionsInput(stores=[store1], reporting_year=2025)
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize StoreEmissionsWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._store_breakdowns: List[StoreEmissionBreakdown] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[StoreEmissionsInput] = None,
        stores: Optional[List[StoreData]] = None,
        reporting_year: int = 2025,
        config: Optional[StoreEmissionsConfig] = None,
    ) -> StoreEmissionsResult:
        """
        Execute the 4-phase store emissions workflow.

        Args:
            input_data: Full input model (preferred).
            stores: List of store data (fallback).
            reporting_year: Reporting year (fallback).
            config: Workflow configuration (fallback).

        Returns:
            StoreEmissionsResult with per-store and consolidated data.

        Raises:
            ValueError: If no store data is provided.
        """
        if input_data is None:
            if stores is None:
                raise ValueError("Either input_data or stores must be provided")
            input_data = StoreEmissionsInput(
                stores=stores,
                reporting_year=reporting_year,
                config=config or StoreEmissionsConfig(),
            )

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting store emissions workflow %s for %d stores, year=%d",
            self.workflow_id, len(input_data.stores), input_data.reporting_year,
        )

        self._phase_results = []
        self._store_breakdowns = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_data_collection(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_scope1_calculation(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_scope2_calculation(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_consolidation(input_data)
            self._phase_results.append(phase4)

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Store emissions workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        consolidated = self._build_consolidated_kpis()

        result = StoreEmissionsResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            per_store_results=self._store_breakdowns,
            consolidated=consolidated,
            scope1_total_tco2e=consolidated.total_scope1_tco2e,
            scope2_total_tco2e=consolidated.total_scope2_location_tco2e,
            kpis={
                "emissions_per_sqm": consolidated.avg_emissions_per_sqm,
                "emissions_per_employee": consolidated.avg_emissions_per_employee,
                "refrigerant_share_pct": consolidated.refrigerant_share_pct,
                "fleet_share_pct": consolidated.fleet_share_pct,
                "heating_share_pct": consolidated.heating_share_pct,
                "renewable_pct": consolidated.renewable_electricity_pct,
                "store_count": consolidated.store_count,
            },
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Store emissions workflow %s completed in %.2fs status=%s",
            self.workflow_id, elapsed, overall_status.value,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Collection
    # -------------------------------------------------------------------------

    async def _phase_data_collection(
        self, input_data: StoreEmissionsInput
    ) -> PhaseResult:
        """Gather and validate energy, refrigerant, and fleet data per store."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for store in input_data.stores:
            store_warnings = self._validate_store_data(store, input_data.config)
            warnings.extend(store_warnings)

        outputs["stores_validated"] = len(input_data.stores)
        outputs["total_energy_records"] = sum(len(s.energy_records) for s in input_data.stores)
        outputs["total_refrigerant_records"] = sum(len(s.refrigerant_records) for s in input_data.stores)
        outputs["total_fleet_records"] = sum(len(s.fleet_records) for s in input_data.stores)
        outputs["total_electricity_records"] = sum(len(s.electricity_records) for s in input_data.stores)

        elapsed = (datetime.utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="data_collection", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _validate_store_data(self, store: StoreData, config: StoreEmissionsConfig) -> List[str]:
        """Validate a single store's data completeness."""
        warnings: List[str] = []
        if store.floor_area_sqm <= 0:
            warnings.append(f"Store {store.store_id}: floor_area_sqm is zero")
        if store.employee_count <= 0:
            warnings.append(f"Store {store.store_id}: employee_count is zero")
        if not store.energy_records and not store.electricity_records:
            warnings.append(f"Store {store.store_id}: no energy or electricity records")
        if config.include_refrigerants and not store.refrigerant_records:
            warnings.append(f"Store {store.store_id}: no refrigerant data")
        if config.include_fleet and not store.fleet_records:
            warnings.append(f"Store {store.store_id}: no fleet records")
        return warnings

    # -------------------------------------------------------------------------
    # Phase 2: Scope 1 Calculation
    # -------------------------------------------------------------------------

    async def _phase_scope1_calculation(self, input_data: StoreEmissionsInput) -> PhaseResult:
        """Calculate Scope 1 emissions: heating, refrigerant, fleet."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        config = input_data.config
        gwp_table = REFRIGERANT_GWP_AR6 if config.gwp_assessment_report == "AR6" else REFRIGERANT_GWP_AR5

        for store in input_data.stores:
            heating = self._calc_heating_emissions(store)
            refrigerant = self._calc_refrigerant_emissions(store, gwp_table) if config.include_refrigerants else 0.0
            fleet = self._calc_fleet_emissions(store) if config.include_fleet else 0.0

            breakdown = self._get_or_create_breakdown(store)
            breakdown.scope1_heating_tco2e = round(heating, 6)
            breakdown.scope1_refrigerant_tco2e = round(refrigerant, 6)
            breakdown.scope1_fleet_tco2e = round(fleet, 6)
            breakdown.scope1_total_tco2e = round(heating + refrigerant + fleet, 6)

        outputs["stores_calculated"] = len(input_data.stores)
        outputs["total_scope1_tco2e"] = round(sum(b.scope1_total_tco2e for b in self._store_breakdowns), 6)
        elapsed = (datetime.utcnow() - started).total_seconds()

        self.logger.info("Phase 2 Scope1Calculation: total=%.4f tCO2e", outputs["total_scope1_tco2e"])
        return PhaseResult(
            phase_name="scope1_calculation", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _calc_heating_emissions(self, store: StoreData) -> float:
        """Calculate Scope 1 heating emissions from fuel combustion (tCO2e)."""
        total = 0.0
        for rec in store.energy_records:
            ef = FUEL_EMISSION_FACTORS.get(rec.fuel_type, 0.0)
            if rec.consumption_kwh > 0:
                total += (rec.consumption_kwh * ef) / 1000.0
            elif rec.consumption_litres > 0:
                kwh = rec.consumption_litres * FUEL_LITRE_TO_KWH.get(rec.fuel_type, 10.0)
                total += (kwh * ef) / 1000.0
            elif rec.consumption_m3 > 0:
                kwh = rec.consumption_m3 * 10.55
                total += (kwh * ef) / 1000.0
        return total

    def _calc_refrigerant_emissions(self, store: StoreData, gwp_table: Dict[str, float]) -> float:
        """Calculate Scope 1 refrigerant leakage emissions (tCO2e)."""
        total = 0.0
        for rec in store.refrigerant_records:
            gwp = rec.gwp if rec.gwp > 0 else gwp_table.get(rec.refrigerant_type, 0.0)
            leakage_kg = rec.leakage_kg
            if leakage_kg <= 0 and rec.leakage_rate_pct > 0 and rec.charge_kg > 0:
                leakage_kg = rec.charge_kg * (rec.leakage_rate_pct / 100.0)
            total += (leakage_kg * gwp) / 1000.0
        return total

    def _calc_fleet_emissions(self, store: StoreData) -> float:
        """Calculate Scope 1 fleet vehicle emissions (tCO2e)."""
        total = 0.0
        for rec in store.fleet_records:
            ef = VEHICLE_FUEL_EF.get(rec.fuel_type, 0.0)
            total += (rec.fuel_consumed_litres * ef) / 1000.0
        return total

    # -------------------------------------------------------------------------
    # Phase 3: Scope 2 Calculation
    # -------------------------------------------------------------------------

    async def _phase_scope2_calculation(self, input_data: StoreEmissionsInput) -> PhaseResult:
        """Calculate Scope 2 location-based and market-based electricity emissions."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for store in input_data.stores:
            loc_based = self._calc_location_based(store)
            mkt_based = self._calc_market_based(store)

            breakdown = self._get_or_create_breakdown(store)
            breakdown.scope2_location_tco2e = round(loc_based, 6)
            breakdown.scope2_market_tco2e = round(mkt_based, 6)
            breakdown.total_tco2e = round(breakdown.scope1_total_tco2e + loc_based, 6)

            if store.floor_area_sqm > 0:
                breakdown.emissions_per_sqm = round(breakdown.total_tco2e / store.floor_area_sqm, 6)
            if store.employee_count > 0:
                breakdown.emissions_per_employee = round(breakdown.total_tco2e / store.employee_count, 6)
            breakdown.floor_area_sqm = store.floor_area_sqm
            breakdown.employee_count = store.employee_count

        outputs["total_scope2_location_tco2e"] = round(sum(b.scope2_location_tco2e for b in self._store_breakdowns), 6)
        outputs["total_scope2_market_tco2e"] = round(sum(b.scope2_market_tco2e for b in self._store_breakdowns), 6)
        elapsed = (datetime.utcnow() - started).total_seconds()

        self.logger.info("Phase 3 Scope2: location=%.4f, market=%.4f", outputs["total_scope2_location_tco2e"], outputs["total_scope2_market_tco2e"])
        return PhaseResult(
            phase_name="scope2_calculation", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _calc_location_based(self, store: StoreData) -> float:
        """Calculate location-based Scope 2 emissions (tCO2e)."""
        total = 0.0
        for rec in store.electricity_records:
            region = rec.grid_region if rec.grid_region else "EU-AVG"
            ef = GRID_LOCATION_EF.get(region, GRID_LOCATION_EF["EU-AVG"])
            total += (rec.consumption_kwh * ef) / 1000.0
        return total

    def _calc_market_based(self, store: StoreData) -> float:
        """Calculate market-based Scope 2 emissions (tCO2e)."""
        total = 0.0
        for rec in store.electricity_records:
            if rec.has_guarantee_of_origin:
                continue
            if rec.market_ef_kgco2_kwh is not None and rec.market_ef_kgco2_kwh >= 0:
                ef = rec.market_ef_kgco2_kwh
            else:
                region = rec.grid_region if rec.grid_region else "EU-AVG"
                ef = GRID_MARKET_EF.get(region, GRID_MARKET_EF.get("EU-AVG", 0.42))
            total += (rec.consumption_kwh * ef) / 1000.0
        return total

    # -------------------------------------------------------------------------
    # Phase 4: Consolidation
    # -------------------------------------------------------------------------

    async def _phase_consolidation(self, input_data: StoreEmissionsInput) -> PhaseResult:
        """Consolidate multi-store results into portfolio-level KPIs."""
        started = datetime.utcnow()
        outputs: Dict[str, Any] = {}

        consolidated = self._build_consolidated_kpis()
        outputs["total_scope1"] = consolidated.total_scope1_tco2e
        outputs["total_scope2_location"] = consolidated.total_scope2_location_tco2e
        outputs["total_scope2_market"] = consolidated.total_scope2_market_tco2e
        outputs["total_emissions"] = consolidated.total_emissions_tco2e
        outputs["avg_emissions_per_sqm"] = consolidated.avg_emissions_per_sqm
        outputs["avg_emissions_per_employee"] = consolidated.avg_emissions_per_employee
        outputs["store_count"] = consolidated.store_count

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info("Phase 4 Consolidation: total=%.4f tCO2e, %d stores", consolidated.total_emissions_tco2e, consolidated.store_count)
        return PhaseResult(
            phase_name="consolidation", status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs,
            provenance_hash=self._hash_dict(outputs),
        )

    def _build_consolidated_kpis(self) -> ConsolidatedKPIs:
        """Build consolidated KPIs from per-store breakdowns."""
        if not self._store_breakdowns:
            return ConsolidatedKPIs()

        total_s1 = sum(b.scope1_total_tco2e for b in self._store_breakdowns)
        total_s2_loc = sum(b.scope2_location_tco2e for b in self._store_breakdowns)
        total_s2_mkt = sum(b.scope2_market_tco2e for b in self._store_breakdowns)
        total_all = total_s1 + total_s2_loc
        total_area = sum(b.floor_area_sqm for b in self._store_breakdowns)
        total_emp = sum(b.employee_count for b in self._store_breakdowns)
        count = len(self._store_breakdowns)

        sorted_stores = sorted(self._store_breakdowns, key=lambda b: b.total_tco2e, reverse=True)
        highest = sorted_stores[0].store_id if sorted_stores else ""
        lowest = sorted_stores[-1].store_id if sorted_stores else ""

        total_heating = sum(b.scope1_heating_tco2e for b in self._store_breakdowns)
        total_refrig = sum(b.scope1_refrigerant_tco2e for b in self._store_breakdowns)
        total_fleet = sum(b.scope1_fleet_tco2e for b in self._store_breakdowns)

        return ConsolidatedKPIs(
            total_scope1_tco2e=round(total_s1, 4),
            total_scope2_location_tco2e=round(total_s2_loc, 4),
            total_scope2_market_tco2e=round(total_s2_mkt, 4),
            total_emissions_tco2e=round(total_all, 4),
            avg_emissions_per_sqm=round(total_all / total_area, 4) if total_area > 0 else 0.0,
            avg_emissions_per_employee=round(total_all / total_emp, 4) if total_emp > 0 else 0.0,
            total_floor_area_sqm=round(total_area, 2),
            total_employees=total_emp,
            store_count=count,
            highest_emitting_store=highest,
            lowest_emitting_store=lowest,
            refrigerant_share_pct=round((total_refrig / total_s1 * 100) if total_s1 > 0 else 0.0, 2),
            fleet_share_pct=round((total_fleet / total_s1 * 100) if total_s1 > 0 else 0.0, 2),
            heating_share_pct=round((total_heating / total_s1 * 100) if total_s1 > 0 else 0.0, 2),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_or_create_breakdown(self, store: StoreData) -> StoreEmissionBreakdown:
        """Get existing or create new breakdown for a store."""
        for bd in self._store_breakdowns:
            if bd.store_id == store.store_id:
                return bd
        new_bd = StoreEmissionBreakdown(
            store_id=store.store_id, store_name=store.store_name,
            floor_area_sqm=store.floor_area_sqm, employee_count=store.employee_count,
        )
        self._store_breakdowns.append(new_bd)
        return new_bd

    def _compute_provenance(self, result: StoreEmissionsResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
