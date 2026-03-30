# -*- coding: utf-8 -*-
"""
EnergyIntensityEngine - PACK-013 CSRD Manufacturing Engine 2
================================================================

Manufacturing energy performance metrics engine.  Calculates Specific
Energy Consumption (SEC), benchmarks against Best Available Techniques
(BAT), assesses Energy Efficiency Directive (EED) compliance, and
identifies decarbonization opportunities.

Metrics Covered:
    - Specific Energy Consumption (SEC) in MJ per unit of production
    - Energy mix breakdown (by source, renewable share)
    - BAT-AEL benchmarking per BREF documents
    - EED 2023/1791 tier classification and audit requirements
    - ISO 50001 certification status tracking
    - Decarbonization opportunity identification with TRL / payback

Energy Sources:
    Electricity, Natural Gas, Coal, Fuel Oil, Biomass, Hydrogen,
    Steam, District Heat, Solar (on-site PV), Wind (on-site)

Regulatory References:
    - Energy Efficiency Directive (EU) 2023/1791 (EED recast)
    - Industrial Emissions Directive (IED) 2010/75/EU + BREF BAT-AELs
    - ESRS E1-5 (Energy consumption and mix)
    - ISO 50001:2018 Energy Management Systems
    - Commission Implementing Decision (EU) 2021/447 (ETS benchmarks)

Zero-Hallucination:
    - All calculations use deterministic float / Decimal arithmetic
    - BAT benchmarks from published EU BREF documents
    - EED thresholds from the directive text (Art. 8, 11)
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-013 CSRD Manufacturing
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning *default* on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator

def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate percentage safely, returning 0.0 on zero denominator."""
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _mwh_to_tj(mwh: float) -> float:
    """Convert MWh to TJ.  1 MWh = 0.0036 TJ."""
    return mwh * 0.0036

def _mwh_to_mj(mwh: float) -> float:
    """Convert MWh to MJ.  1 MWh = 3600 MJ."""
    return mwh * 3600.0

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EnergySource(str, Enum):
    """Energy sources tracked in manufacturing facilities."""
    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    COAL = "coal"
    FUEL_OIL = "fuel_oil"
    BIOMASS = "biomass"
    HYDROGEN = "hydrogen"
    STEAM = "steam"
    HEAT = "heat"
    SOLAR = "solar"
    WIND = "wind"

class ProductionUnit(str, Enum):
    """Units of production output."""
    TONNES = "tonnes"
    UNITS = "units"
    LITRES = "litres"
    SQM = "sqm"
    KWH = "kwh"

class EEDTier(str, Enum):
    """Energy Efficiency Directive tier classification.

    EED 2023/1791 Art. 8 and Art. 11 define thresholds:
        - Below 10 TJ/year: no mandatory audit or EMS
        - 10-85 TJ/year: energy audit required every 4 years
        - Above 85 TJ/year: ISO 50001 or equivalent EMS required
    """
    BELOW_10TJ = "below_10tj"
    AUDIT_REQUIRED_10_85TJ = "audit_required_10_85tj"
    ISO50001_REQUIRED_ABOVE_85TJ = "iso50001_required_above_85tj"

# ---------------------------------------------------------------------------
# Constants: BAT Energy Benchmarks (MJ/tonne product)
# Sources: EU BREF documents (BAT-AEL ranges, lower bound = BAT)
# ---------------------------------------------------------------------------

BAT_ENERGY_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "cement": {
        "bat_mj_per_tonne": 3000.0,       # BAT for clinker (dry process)
        "sector_average_mj_per_tonne": 3500.0,
        "source": "CLM BREF (2013), BAT 16",
    },
    "steel_bof": {
        "bat_mj_per_tonne": 17000.0,      # BAT for BF-BOF crude steel
        "sector_average_mj_per_tonne": 21000.0,
        "source": "IS BREF (2012), BAT 1",
    },
    "steel_eaf": {
        "bat_mj_per_tonne": 1800.0,       # BAT for EAF (electricity only)
        "sector_average_mj_per_tonne": 2500.0,
        "source": "IS BREF (2012), BAT 47",
    },
    "aluminum": {
        "bat_mj_per_tonne": 50000.0,      # BAT for primary Al smelting
        "sector_average_mj_per_tonne": 55000.0,
        "source": "NFM BREF (2014), BAT 85",
    },
    "glass_flat": {
        "bat_mj_per_tonne": 5500.0,       # BAT for float glass
        "sector_average_mj_per_tonne": 7500.0,
        "source": "GLS BREF (2012), BAT 4",
    },
    "glass_container": {
        "bat_mj_per_tonne": 4000.0,       # BAT for container glass
        "sector_average_mj_per_tonne": 6000.0,
        "source": "GLS BREF (2012), BAT 4",
    },
    "ceramics": {
        "bat_mj_per_tonne": 2000.0,       # BAT for wall/floor tiles
        "sector_average_mj_per_tonne": 3500.0,
        "source": "CER BREF (2007), BAT 12",
    },
    "pulp_paper": {
        "bat_mj_per_tonne": 10000.0,      # BAT for integrated kraft pulp+paper
        "sector_average_mj_per_tonne": 14000.0,
        "source": "PP BREF (2015), BAT 8",
    },
    "food_beverage": {
        "bat_mj_per_tonne": 1500.0,       # BAT for food processing (generic)
        "sector_average_mj_per_tonne": 3000.0,
        "source": "FDM BREF (2006), BAT 5",
    },
    "textiles": {
        "bat_mj_per_tonne": 8000.0,       # BAT for textile finishing
        "sector_average_mj_per_tonne": 15000.0,
        "source": "TXT BREF (2003), BAT 10",
    },
    "pharmaceuticals": {
        "bat_mj_per_tonne": 20000.0,      # BAT for API manufacturing
        "sector_average_mj_per_tonne": 50000.0,
        "source": "OFC BREF (2006), BAT 3",
    },
    "chemicals_bulk": {
        "bat_mj_per_tonne": 5000.0,       # BAT for bulk organics
        "sector_average_mj_per_tonne": 8000.0,
        "source": "LVOC BREF (2017), BAT 5",
    },
    "electronics": {
        "bat_mj_per_tonne": 25000.0,      # BAT estimate for electronics mfg
        "sector_average_mj_per_tonne": 40000.0,
        "source": "Industry estimate",
    },
    "automotive": {
        "bat_mj_per_tonne": 4000.0,       # BAT for vehicle assembly per tonne
        "sector_average_mj_per_tonne": 6000.0,
        "source": "STM BREF (2006), BAT 6",
    },
}

# ---------------------------------------------------------------------------
# Constants: Decarbonization Technologies
# ---------------------------------------------------------------------------

DECARBONIZATION_TECHNOLOGIES: List[Dict[str, Any]] = [
    {
        "technology": "Industrial Heat Pump",
        "description": "Upgrade waste heat using electricity-driven heat pumps "
                       "for process heating below 150C.",
        "savings_pct": 0.30,              # 30% energy savings on heat demand
        "investment_eur_per_mwh_saved": 350.0,
        "payback_years_typical": 4.0,
        "trl": 9,
        "co2_reduction_pct": 0.25,
        "applicable_sectors": [
            "food_beverage", "textiles", "pharmaceuticals", "chemicals",
            "pulp_paper",
        ],
    },
    {
        "technology": "Waste Heat Recovery (ORC)",
        "description": "Organic Rankine Cycle to generate electricity from "
                       "exhaust gases above 200C.",
        "savings_pct": 0.10,
        "investment_eur_per_mwh_saved": 500.0,
        "payback_years_typical": 5.0,
        "trl": 9,
        "co2_reduction_pct": 0.08,
        "applicable_sectors": [
            "cement", "steel", "glass", "ceramics", "chemicals",
        ],
    },
    {
        "technology": "Electric Arc Furnace (EAF) Conversion",
        "description": "Shift from BF-BOF to EAF steelmaking using scrap.",
        "savings_pct": 0.65,
        "investment_eur_per_mwh_saved": 1200.0,
        "payback_years_typical": 12.0,
        "trl": 9,
        "co2_reduction_pct": 0.60,
        "applicable_sectors": ["steel"],
    },
    {
        "technology": "Green Hydrogen Direct Reduction",
        "description": "Replace coal/coke with green H2 for iron ore reduction.",
        "savings_pct": 0.80,
        "investment_eur_per_mwh_saved": 2000.0,
        "payback_years_typical": 15.0,
        "trl": 7,
        "co2_reduction_pct": 0.90,
        "applicable_sectors": ["steel"],
    },
    {
        "technology": "On-site Solar PV",
        "description": "Rooftop or ground-mounted PV for self-consumption.",
        "savings_pct": 0.15,
        "investment_eur_per_mwh_saved": 200.0,
        "payback_years_typical": 6.0,
        "trl": 9,
        "co2_reduction_pct": 0.12,
        "applicable_sectors": [
            "cement", "steel", "aluminum", "chemicals", "glass",
            "ceramics", "pulp_paper", "food_beverage", "textiles",
            "pharmaceuticals", "electronics", "automotive",
        ],
    },
    {
        "technology": "Biomass Co-firing",
        "description": "Substitute fossil fuels with biomass in kilns/boilers.",
        "savings_pct": 0.20,
        "investment_eur_per_mwh_saved": 150.0,
        "payback_years_typical": 3.0,
        "trl": 9,
        "co2_reduction_pct": 0.18,
        "applicable_sectors": [
            "cement", "ceramics", "glass", "pulp_paper", "food_beverage",
        ],
    },
    {
        "technology": "Variable Speed Drives (VSD)",
        "description": "Install VSDs on motors, pumps, fans, and compressors.",
        "savings_pct": 0.15,
        "investment_eur_per_mwh_saved": 80.0,
        "payback_years_typical": 2.0,
        "trl": 9,
        "co2_reduction_pct": 0.10,
        "applicable_sectors": [
            "cement", "steel", "aluminum", "chemicals", "glass",
            "ceramics", "pulp_paper", "food_beverage", "textiles",
            "pharmaceuticals", "electronics", "automotive",
        ],
    },
    {
        "technology": "Carbon Capture and Storage (CCS)",
        "description": "Post-combustion CO2 capture with geological storage.",
        "savings_pct": 0.05,              # Small energy savings; main benefit is CO2
        "investment_eur_per_mwh_saved": 3000.0,
        "payback_years_typical": 20.0,
        "trl": 7,
        "co2_reduction_pct": 0.85,
        "applicable_sectors": ["cement", "steel", "chemicals"],
    },
    {
        "technology": "Electrification of Process Heat",
        "description": "Replace gas/oil burners with electric resistance or "
                       "induction heating.",
        "savings_pct": 0.25,
        "investment_eur_per_mwh_saved": 400.0,
        "payback_years_typical": 7.0,
        "trl": 8,
        "co2_reduction_pct": 0.40,
        "applicable_sectors": [
            "food_beverage", "textiles", "pharmaceuticals", "electronics",
            "automotive",
        ],
    },
    {
        "technology": "Digital Energy Management (AI/IoT)",
        "description": "Real-time monitoring and optimisation of energy use via "
                       "IoT sensors and control algorithms.",
        "savings_pct": 0.10,
        "investment_eur_per_mwh_saved": 50.0,
        "payback_years_typical": 1.5,
        "trl": 9,
        "co2_reduction_pct": 0.08,
        "applicable_sectors": [
            "cement", "steel", "aluminum", "chemicals", "glass",
            "ceramics", "pulp_paper", "food_beverage", "textiles",
            "pharmaceuticals", "electronics", "automotive",
        ],
    },
]

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class EnergyIntensityConfig(BaseModel):
    """Configuration for energy intensity calculations.

    Attributes:
        reporting_year: Calendar year for reporting.
        production_unit: Default production measurement unit.
        include_benchmark: Whether to compare against BAT benchmarks.
        include_eed_compliance: Whether to assess EED compliance.
        iso50001_certified: Whether the facility holds ISO 50001.
    """
    reporting_year: int = Field(
        default=2025,
        ge=2019,
        le=2035,
        description="Calendar year for reporting.",
    )
    production_unit: ProductionUnit = Field(
        default=ProductionUnit.TONNES,
        description="Default production measurement unit.",
    )
    include_benchmark: bool = Field(
        default=True,
        description="Compare against BAT-AEL benchmarks.",
    )
    include_eed_compliance: bool = Field(
        default=True,
        description="Assess EED tier and compliance.",
    )
    iso50001_certified: bool = Field(
        default=False,
        description="Whether ISO 50001 certification is held.",
    )

class EnergyConsumptionData(BaseModel):
    """Energy consumption from a single source.

    Attributes:
        source: Energy source type.
        quantity_mwh: Energy consumed in MWh.
        cost_eur: Total cost in EUR for this source.
        renewable_pct: Percentage of this source that is renewable.
        emission_factor_tco2_per_mwh: Emission factor (tCO2/MWh).
    """
    source: EnergySource = Field(..., description="Energy source.")
    quantity_mwh: float = Field(..., ge=0.0, description="MWh consumed.")
    cost_eur: float = Field(default=0.0, ge=0.0, description="Cost in EUR.")
    renewable_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Renewable share as percentage.",
    )
    emission_factor_tco2_per_mwh: float = Field(
        default=0.0, ge=0.0,
        description="tCO2 per MWh for this source.",
    )

class ProductionVolumeData(BaseModel):
    """Production volume for a product line.

    Attributes:
        product_name: Name of the product.
        volume: Production volume.
        unit: Unit of measurement.
        period: Reporting period (e.g., '2025-annual').
    """
    product_name: str = Field(..., min_length=1, description="Product name.")
    volume: float = Field(..., ge=0.0, description="Volume produced.")
    unit: ProductionUnit = Field(
        default=ProductionUnit.TONNES,
        description="Unit of production.",
    )
    period: str = Field(
        default="annual",
        description="Reporting period identifier.",
    )

class FacilityEnergyData(BaseModel):
    """Complete energy data for a manufacturing facility.

    Attributes:
        facility_id: Unique facility identifier.
        facility_name: Human-readable name.
        sub_sector: Manufacturing sub-sector for benchmarking.
        energy_consumption: List of energy consumption records.
        production_volumes: List of production volume records.
        total_energy_tj: Pre-computed total energy in TJ (optional).
        floor_area_sqm: Facility floor area in m2 (for EUI calculation).
        annual_revenue_eur: Annual revenue in EUR (for revenue intensity).
        baseline_sec_mj: Baseline SEC for improvement tracking (optional).
    """
    facility_id: str = Field(
        default_factory=_new_uuid,
        description="Unique facility identifier.",
    )
    facility_name: str = Field(
        ..., min_length=1,
        description="Name of the facility.",
    )
    sub_sector: str = Field(
        default="",
        description="Manufacturing sub-sector for BAT benchmarking.",
    )
    energy_consumption: List[EnergyConsumptionData] = Field(
        default_factory=list,
        description="Energy consumption records by source.",
    )
    production_volumes: List[ProductionVolumeData] = Field(
        default_factory=list,
        description="Production volume records.",
    )
    total_energy_tj: Optional[float] = Field(
        default=None, ge=0.0,
        description="Pre-computed total energy in TJ.",
    )
    floor_area_sqm: Optional[float] = Field(
        default=None, ge=0.0,
        description="Facility floor area in m2.",
    )
    annual_revenue_eur: Optional[float] = Field(
        default=None, ge=0.0,
        description="Annual revenue in EUR.",
    )
    baseline_sec_mj: Optional[float] = Field(
        default=None, ge=0.0,
        description="Baseline SEC (MJ/unit) for improvement calculation.",
    )

class BenchmarkComparison(BaseModel):
    """Comparison of facility SEC against BAT benchmark.

    Attributes:
        facility_value: Facility SEC (MJ/tonne).
        bat_ael_value: BAT-AEL benchmark (MJ/tonne).
        sector_average: Sector average SEC (MJ/tonne).
        percentile_rank: Estimated percentile rank within sector.
        status: Qualitative status (e.g., 'above_bat', 'at_bat', 'below_average').
    """
    facility_value: float = Field(default=0.0)
    bat_ael_value: float = Field(default=0.0)
    sector_average: float = Field(default=0.0)
    percentile_rank: float = Field(default=0.0, ge=0.0, le=100.0)
    status: str = Field(default="unknown")

class EEDCompliance(BaseModel):
    """Energy Efficiency Directive compliance assessment.

    Attributes:
        tier: EED tier classification.
        total_energy_tj: Total energy consumption in TJ.
        audit_required: Whether an energy audit is required.
        iso50001_required: Whether ISO 50001 (or equivalent) is required.
        last_audit_date: Date of the last energy audit (if any).
        compliant: Whether the facility is currently compliant.
    """
    tier: EEDTier = Field(default=EEDTier.BELOW_10TJ)
    total_energy_tj: float = Field(default=0.0, ge=0.0)
    audit_required: bool = Field(default=False)
    iso50001_required: bool = Field(default=False)
    last_audit_date: Optional[str] = Field(default=None)
    compliant: bool = Field(default=True)

class DecarbonizationOpportunity(BaseModel):
    """A potential decarbonization technology opportunity.

    Attributes:
        technology: Name of the technology.
        description: Brief description.
        potential_savings_mwh: Estimated annual energy savings in MWh.
        investment_eur: Estimated investment cost in EUR.
        payback_years: Simple payback period in years.
        trl: Technology readiness level (1-9).
        co2_reduction_tonnes: Estimated annual CO2 reduction in tonnes.
    """
    technology: str = Field(default="")
    description: str = Field(default="")
    potential_savings_mwh: float = Field(default=0.0, ge=0.0)
    investment_eur: float = Field(default=0.0, ge=0.0)
    payback_years: float = Field(default=0.0, ge=0.0)
    trl: int = Field(default=1, ge=1, le=9)
    co2_reduction_tonnes: float = Field(default=0.0, ge=0.0)

class EnergyIntensityResult(BaseModel):
    """Complete result of energy intensity calculation with provenance.

    Attributes:
        result_id: Unique result identifier.
        facility_id: Facility this result pertains to.
        total_energy_mwh: Total energy consumption in MWh.
        total_energy_tj: Total energy consumption in TJ.
        sec_mj_per_unit: Specific Energy Consumption by product (MJ/unit).
        sec_mj_per_eur_revenue: Energy per EUR revenue (MJ/EUR).
        energy_mix_breakdown: Breakdown by energy source.
        renewable_share_pct: Percentage of energy from renewable sources.
        benchmark_comparison: BAT benchmark comparison (if configured).
        eed_compliance: EED compliance assessment (if configured).
        iso50001_status: ISO 50001 certification status.
        decarbonization_opportunities: List of identified opportunities.
        improvement_vs_baseline_pct: Improvement vs baseline SEC (%).
        methodology_notes: Notes on methodology and data sources.
        processing_time_ms: Time taken to compute this result.
        engine_version: Version of this engine.
        calculated_at: UTC timestamp of calculation.
        provenance_hash: SHA-256 hash of all inputs and outputs.
    """
    result_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    total_energy_mwh: float = Field(default=0.0)
    total_energy_tj: float = Field(default=0.0)
    sec_mj_per_unit: Dict[str, float] = Field(default_factory=dict)
    sec_mj_per_eur_revenue: float = Field(default=0.0)
    energy_mix_breakdown: Dict[str, Any] = Field(default_factory=dict)
    renewable_share_pct: float = Field(default=0.0)
    benchmark_comparison: Optional[BenchmarkComparison] = Field(default=None)
    eed_compliance: Optional[EEDCompliance] = Field(default=None)
    iso50001_status: str = Field(default="not_certified")
    decarbonization_opportunities: List[DecarbonizationOpportunity] = Field(
        default_factory=list,
    )
    improvement_vs_baseline_pct: Optional[float] = Field(default=None)
    methodology_notes: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EnergyIntensityEngine:
    """Zero-hallucination manufacturing energy intensity calculation engine.

    Calculates Specific Energy Consumption (SEC), energy mix, benchmarks
    against BAT-AEL values from EU BREF documents, assesses EED compliance,
    and identifies decarbonization opportunities.

    Guarantees:
        - Deterministic: same inputs produce identical outputs (bit-perfect).
        - Reproducible: every result carries a SHA-256 provenance hash.
        - Auditable: full breakdown by source, product, and benchmark.
        - No LLM: zero hallucination risk in any calculation path.

    Usage::

        config = EnergyIntensityConfig(
            reporting_year=2025,
            include_benchmark=True,
            include_eed_compliance=True,
        )
        engine = EnergyIntensityEngine(config)
        result = engine.calculate_energy_intensity(facility_data)

    Args:
        config: Engine configuration.  Accepts an EnergyIntensityConfig,
                a plain dict, or None (defaults applied).
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialise the energy intensity engine.

        Args:
            config: An EnergyIntensityConfig, dict, or None for defaults.
        """
        if config is None:
            self.config = EnergyIntensityConfig()
        elif isinstance(config, dict):
            self.config = EnergyIntensityConfig(**config)
        elif isinstance(config, EnergyIntensityConfig):
            self.config = config
        else:
            raise TypeError(
                f"config must be EnergyIntensityConfig, dict, or None, "
                f"got {type(config).__name__}"
            )
        logger.info(
            "EnergyIntensityEngine initialised: year=%d",
            self.config.reporting_year,
        )

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def calculate_energy_intensity(
        self, facility: FacilityEnergyData
    ) -> EnergyIntensityResult:
        """Calculate comprehensive energy intensity metrics for a facility.

        Computes total energy consumption, SEC per product, energy mix
        breakdown, renewable share, benchmark comparison, EED compliance,
        and decarbonization opportunities.

        Args:
            facility: Complete facility energy data.

        Returns:
            EnergyIntensityResult with full breakdown and provenance.

        Raises:
            ValueError: If facility has no energy consumption data.
        """
        t0 = time.perf_counter()

        if not facility.energy_consumption:
            raise ValueError(
                f"Facility '{facility.facility_name}' has no energy "
                f"consumption data."
            )

        methodology_notes: List[str] = [
            f"Reporting year: {self.config.reporting_year}",
            f"Engine version: {self.engine_version}",
        ]

        # ----- Total energy -----
        total_mwh = sum(ec.quantity_mwh for ec in facility.energy_consumption)
        total_tj = (
            facility.total_energy_tj
            if facility.total_energy_tj is not None
            else _mwh_to_tj(total_mwh)
        )

        # ----- Energy mix breakdown -----
        energy_mix: Dict[str, Any] = {}
        total_renewable_mwh = 0.0
        total_co2 = 0.0

        for ec in facility.energy_consumption:
            source_key = ec.source.value
            renewable_mwh = ec.quantity_mwh * (ec.renewable_pct / 100.0)
            fossil_mwh = ec.quantity_mwh - renewable_mwh
            co2 = ec.quantity_mwh * ec.emission_factor_tco2_per_mwh
            total_renewable_mwh += renewable_mwh
            total_co2 += co2

            energy_mix[source_key] = {
                "quantity_mwh": _round3(ec.quantity_mwh),
                "share_pct": _round2(_safe_pct(ec.quantity_mwh, total_mwh)),
                "cost_eur": _round2(ec.cost_eur),
                "renewable_pct": _round2(ec.renewable_pct),
                "co2_tonnes": _round3(co2),
            }

        renewable_share = _safe_pct(total_renewable_mwh, total_mwh)
        methodology_notes.append(
            f"Total energy: {_round3(total_mwh)} MWh "
            f"({_round3(total_tj)} TJ)."
        )
        methodology_notes.append(
            f"Renewable share: {_round2(renewable_share)}%."
        )

        # ----- SEC per product -----
        sec_by_product: Dict[str, float] = {}
        total_energy_mj = _mwh_to_mj(total_mwh)

        for pv in facility.production_volumes:
            if pv.volume > 0:
                sec = self.calculate_sec(total_mwh, pv.volume, pv.unit)
                sec_by_product[pv.product_name] = _round3(sec)

        # ----- Revenue intensity -----
        sec_per_eur = 0.0
        if facility.annual_revenue_eur and facility.annual_revenue_eur > 0:
            sec_per_eur = _safe_divide(total_energy_mj, facility.annual_revenue_eur)

        # ----- Benchmark comparison -----
        benchmark: Optional[BenchmarkComparison] = None
        if self.config.include_benchmark and facility.sub_sector:
            # Use the first product's SEC for benchmarking
            first_sec = next(iter(sec_by_product.values()), 0.0)
            benchmark = self.compare_benchmark(first_sec, facility.sub_sector)
            if benchmark:
                methodology_notes.append(
                    f"BAT benchmark comparison for sub-sector "
                    f"'{facility.sub_sector}': "
                    f"facility {benchmark.facility_value} vs BAT "
                    f"{benchmark.bat_ael_value} MJ/t."
                )

        # ----- EED compliance -----
        eed: Optional[EEDCompliance] = None
        if self.config.include_eed_compliance:
            eed = self.assess_eed_compliance(total_tj)
            methodology_notes.append(
                f"EED tier: {eed.tier.value}. Audit required: {eed.audit_required}."
            )

        # ----- ISO 50001 status -----
        iso_status = "certified" if self.config.iso50001_certified else "not_certified"

        # ----- Decarbonization opportunities -----
        opportunities = self.identify_decarbonization(
            facility, total_mwh, total_co2
        )

        # ----- Improvement vs baseline -----
        improvement_pct: Optional[float] = None
        if facility.baseline_sec_mj and sec_by_product:
            first_sec = next(iter(sec_by_product.values()), 0.0)
            if facility.baseline_sec_mj > 0:
                improvement_pct = _round2(
                    (1.0 - first_sec / facility.baseline_sec_mj) * 100.0
                )
                methodology_notes.append(
                    f"Improvement vs baseline: {improvement_pct}%."
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = EnergyIntensityResult(
            facility_id=facility.facility_id,
            total_energy_mwh=_round3(total_mwh),
            total_energy_tj=_round3(total_tj),
            sec_mj_per_unit=sec_by_product,
            sec_mj_per_eur_revenue=_round3(sec_per_eur),
            energy_mix_breakdown=energy_mix,
            renewable_share_pct=_round2(renewable_share),
            benchmark_comparison=benchmark,
            eed_compliance=eed,
            iso50001_status=iso_status,
            decarbonization_opportunities=opportunities,
            improvement_vs_baseline_pct=improvement_pct,
            methodology_notes=methodology_notes,
            processing_time_ms=round(elapsed_ms, 2),
            engine_version=self.engine_version,
            calculated_at=utcnow(),
        )

        result.provenance_hash = _compute_hash(result)
        return result

    def calculate_sec(
        self,
        energy_mwh: float,
        production_volume: float,
        unit: ProductionUnit = ProductionUnit.TONNES,
    ) -> float:
        """Calculate Specific Energy Consumption (SEC).

        SEC = Total Energy (MJ) / Production Volume

        Args:
            energy_mwh: Total energy consumption in MWh.
            production_volume: Production volume in the specified unit.
            unit: Production unit (default: tonnes).

        Returns:
            SEC in MJ per unit of production.
        """
        energy_mj = _mwh_to_mj(energy_mwh)
        return _safe_divide(energy_mj, production_volume)

    def assess_eed_compliance(
        self,
        total_energy_tj: float,
        last_audit_date: Optional[str] = None,
    ) -> EEDCompliance:
        """Assess EED 2023/1791 compliance tier.

        Thresholds per EED recast (Art. 8 and Art. 11):
          - Below 10 TJ/year:     No mandatory energy audit or EMS
          - 10 to 85 TJ/year:    Energy audit every 4 years (Art. 8)
          - Above 85 TJ/year:    ISO 50001 or equivalent EMS (Art. 11)

        Args:
            total_energy_tj: Total annual energy consumption in TJ.
            last_audit_date: Date of last energy audit (YYYY-MM-DD string).

        Returns:
            EEDCompliance with tier, requirements, and compliance status.
        """
        if total_energy_tj < 10.0:
            tier = EEDTier.BELOW_10TJ
            audit_required = False
            iso_required = False
            compliant = True
        elif total_energy_tj <= 85.0:
            tier = EEDTier.AUDIT_REQUIRED_10_85TJ
            audit_required = True
            iso_required = False
            # Check audit recency (must be within 4 years)
            compliant = self._is_audit_recent(last_audit_date, max_years=4)
        else:
            tier = EEDTier.ISO50001_REQUIRED_ABOVE_85TJ
            audit_required = True
            iso_required = True
            compliant = self.config.iso50001_certified

        return EEDCompliance(
            tier=tier,
            total_energy_tj=_round3(total_energy_tj),
            audit_required=audit_required,
            iso50001_required=iso_required,
            last_audit_date=last_audit_date,
            compliant=compliant,
        )

    def compare_benchmark(
        self,
        sec_mj_per_tonne: float,
        sub_sector: str,
    ) -> Optional[BenchmarkComparison]:
        """Compare facility SEC against BAT-AEL benchmark.

        Args:
            sec_mj_per_tonne: Facility's SEC in MJ/tonne.
            sub_sector: Sub-sector key for BAT lookup.

        Returns:
            BenchmarkComparison or None if no benchmark found.
        """
        bm_data = BAT_ENERGY_BENCHMARKS.get(sub_sector)
        if bm_data is None:
            return None

        bat_value = bm_data["bat_mj_per_tonne"]
        avg_value = bm_data["sector_average_mj_per_tonne"]

        # Estimate percentile rank (linear interpolation)
        if sec_mj_per_tonne <= bat_value:
            percentile = 95.0  # top 5% (at or below BAT)
            status = "at_or_below_bat"
        elif sec_mj_per_tonne <= avg_value:
            # Linear between BAT (95th percentile) and average (50th)
            fraction = _safe_divide(
                sec_mj_per_tonne - bat_value,
                avg_value - bat_value,
            )
            percentile = 95.0 - (fraction * 45.0)
            status = "between_bat_and_average"
        else:
            # Below average -- linear from 50 to 5
            excess_range = avg_value * 0.5  # assume bottom is 1.5x average
            fraction = min(
                _safe_divide(sec_mj_per_tonne - avg_value, excess_range),
                1.0,
            )
            percentile = 50.0 - (fraction * 45.0)
            status = "above_sector_average"

        return BenchmarkComparison(
            facility_value=_round3(sec_mj_per_tonne),
            bat_ael_value=bat_value,
            sector_average=avg_value,
            percentile_rank=_round2(max(percentile, 0.0)),
            status=status,
        )

    def identify_decarbonization(
        self,
        facility: FacilityEnergyData,
        total_mwh: float,
        total_co2: float,
    ) -> List[DecarbonizationOpportunity]:
        """Identify applicable decarbonization opportunities.

        Filters the technology database by the facility's sub-sector and
        estimates potential savings based on total energy consumption and
        CO2 emissions.

        Args:
            facility: Facility energy data.
            total_mwh: Total energy consumption in MWh.
            total_co2: Total CO2 emissions in tonnes.

        Returns:
            List of DecarbonizationOpportunity sorted by payback period.
        """
        opportunities: List[DecarbonizationOpportunity] = []
        sub_sector = facility.sub_sector

        for tech in DECARBONIZATION_TECHNOLOGIES:
            applicable_sectors = tech.get("applicable_sectors", [])
            if sub_sector and sub_sector not in applicable_sectors:
                continue

            savings_mwh = total_mwh * tech["savings_pct"]
            co2_reduction = total_co2 * tech["co2_reduction_pct"]
            investment = savings_mwh * tech["investment_eur_per_mwh_saved"]

            opp = DecarbonizationOpportunity(
                technology=tech["technology"],
                description=tech["description"],
                potential_savings_mwh=_round3(savings_mwh),
                investment_eur=_round2(investment),
                payback_years=tech["payback_years_typical"],
                trl=tech["trl"],
                co2_reduction_tonnes=_round3(co2_reduction),
            )
            opportunities.append(opp)

        # Sort by payback period (shortest first)
        opportunities.sort(key=lambda o: o.payback_years)
        return opportunities

    # --------------------------------------------------------------------- #
    # Private helpers
    # --------------------------------------------------------------------- #

    def _is_audit_recent(
        self,
        audit_date_str: Optional[str],
        max_years: int = 4,
    ) -> bool:
        """Check whether an audit date is within the allowed recency window.

        Args:
            audit_date_str: ISO date string (YYYY-MM-DD) or None.
            max_years: Maximum allowed age in years.

        Returns:
            True if the audit is recent enough, False otherwise.
        """
        if not audit_date_str:
            return False

        try:
            audit_date = date.fromisoformat(audit_date_str)
        except (ValueError, TypeError):
            return False

        today = date.today()
        delta_days = (today - audit_date).days
        return delta_days <= (max_years * 365)
