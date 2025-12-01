# -*- coding: utf-8 -*-
"""
Economic Impact Calculator for GL-015 INSULSCAN

Comprehensive economic analysis for insulation assessment and repair decisions.
Zero-hallucination guarantee through deterministic calculations with complete
provenance tracking.

Analysis Capabilities:
1. Repair Cost Estimation (materials, labor, access, permits)
2. Energy Savings Calculation (heat loss reduction, fuel escalation)
3. Payback Analysis (simple, discounted, break-even)
4. NPV and IRR Calculation (discounted cash flow)
5. Life Cycle Cost Analysis (TCO, maintenance, replacement)
6. Economic Thickness Optimization (ASTM C680)
7. Budget Impact Analysis (CapEx, OpEx, cash flow)
8. Carbon Economics (credits, taxes, ESG benefits)

Standards References:
- ASTM C680: Standard Practice for Economic Thickness of Thermal Insulation
- 3E Plus: NAIMA Economic Thickness for Industrial Insulation
- CINI Manual: Insulation Cost Estimating Handbook
- EPA GHG Reporting: Carbon Economics Framework

Author: GreenLang AI Agent Factory
Created: 2025-12-01
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from pydantic import BaseModel, Field, validator
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from dataclasses import dataclass, field
import hashlib
import json
import math
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS AND CONSTANTS
# =============================================================================

class InsulationType(str, Enum):
    """Insulation material types with thermal properties."""
    MINERAL_WOOL = "mineral_wool"
    CALCIUM_SILICATE = "calcium_silicate"
    CELLULAR_GLASS = "cellular_glass"
    PERLITE = "perlite"
    FIBERGLASS = "fiberglass"
    AEROGEL = "aerogel"
    POLYURETHANE_FOAM = "polyurethane_foam"
    PHENOLIC_FOAM = "phenolic_foam"
    CERAMIC_FIBER = "ceramic_fiber"
    MICROPOROUS = "microporous"


class RepairComplexity(str, Enum):
    """Repair complexity levels affecting labor costs."""
    SIMPLE = "simple"           # Minor patching, accessible location
    MODERATE = "moderate"       # Sectional replacement, standard access
    COMPLEX = "complex"         # Full replacement, difficult access
    CRITICAL = "critical"       # High-temperature, confined space, permits


class AccessRequirement(str, Enum):
    """Access equipment requirements."""
    GROUND_LEVEL = "ground_level"
    LADDER = "ladder"
    SCAFFOLDING = "scaffolding"
    LIFT_PLATFORM = "lift_platform"
    ROPE_ACCESS = "rope_access"
    CONFINED_SPACE = "confined_space"


class EquipmentType(str, Enum):
    """Industrial equipment types for insulation."""
    PIPE = "pipe"
    VESSEL = "vessel"
    TANK = "tank"
    BOILER = "boiler"
    HEAT_EXCHANGER = "heat_exchanger"
    DUCT = "duct"
    TURBINE = "turbine"
    VALVE = "valve"
    FLANGE = "flange"


class FuelType(str, Enum):
    """Fuel types for energy cost calculations."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL = "fuel_oil"
    COAL = "coal"
    ELECTRICITY = "electricity"
    STEAM = "steam"
    LPG = "lpg"
    BIOMASS = "biomass"


class DepreciationMethod(str, Enum):
    """Depreciation methods for financial analysis."""
    STRAIGHT_LINE = "straight_line"
    MACRS_5_YEAR = "macrs_5"
    MACRS_7_YEAR = "macrs_7"
    MACRS_15_YEAR = "macrs_15"
    DOUBLE_DECLINING = "double_declining"


class CarbonPricingScheme(str, Enum):
    """Carbon pricing mechanisms."""
    EU_ETS = "eu_ets"           # EU Emissions Trading System
    CBAM = "cbam"               # Carbon Border Adjustment Mechanism
    US_RGGI = "us_rggi"         # Regional Greenhouse Gas Initiative
    CALIFORNIA_CAP = "california_cap"
    CARBON_TAX = "carbon_tax"   # Direct carbon tax
    VOLUNTARY = "voluntary"     # Voluntary carbon market


# =============================================================================
# COST FACTOR DATABASES
# =============================================================================

@dataclass
class InsulationMaterialCosts:
    """Material cost database per insulation type (USD/m3 installed)."""
    costs: Dict[InsulationType, Tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.costs:
            self.costs = {
                InsulationType.MINERAL_WOOL: (150.0, 350.0),
                InsulationType.CALCIUM_SILICATE: (400.0, 800.0),
                InsulationType.CELLULAR_GLASS: (500.0, 1000.0),
                InsulationType.PERLITE: (300.0, 600.0),
                InsulationType.FIBERGLASS: (120.0, 280.0),
                InsulationType.AEROGEL: (2000.0, 5000.0),
                InsulationType.POLYURETHANE_FOAM: (250.0, 500.0),
                InsulationType.PHENOLIC_FOAM: (350.0, 700.0),
                InsulationType.CERAMIC_FIBER: (600.0, 1200.0),
                InsulationType.MICROPOROUS: (1500.0, 3500.0),
            }

    def get_cost(self, insulation_type: InsulationType, quality_factor: float = 0.5) -> float:
        """Get interpolated cost based on quality factor (0-1)."""
        min_cost, max_cost = self.costs[insulation_type]
        return min_cost + (max_cost - min_cost) * quality_factor


@dataclass
class LaborRates:
    """Labor rate database by complexity and region (USD/hour)."""
    base_rates: Dict[RepairComplexity, float] = field(default_factory=dict)
    regional_multipliers: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.base_rates:
            self.base_rates = {
                RepairComplexity.SIMPLE: 45.0,
                RepairComplexity.MODERATE: 65.0,
                RepairComplexity.COMPLEX: 95.0,
                RepairComplexity.CRITICAL: 135.0,
            }
        if not self.regional_multipliers:
            self.regional_multipliers = {
                "us_northeast": 1.25,
                "us_southeast": 0.95,
                "us_midwest": 1.00,
                "us_west": 1.30,
                "us_gulf": 1.10,
                "europe_west": 1.40,
                "europe_east": 0.70,
                "asia_pacific": 0.60,
                "middle_east": 0.85,
                "default": 1.00,
            }


@dataclass
class AccessCosts:
    """Access equipment costs (USD/day rental + setup)."""
    daily_rental: Dict[AccessRequirement, float] = field(default_factory=dict)
    setup_cost: Dict[AccessRequirement, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.daily_rental:
            self.daily_rental = {
                AccessRequirement.GROUND_LEVEL: 0.0,
                AccessRequirement.LADDER: 15.0,
                AccessRequirement.SCAFFOLDING: 250.0,
                AccessRequirement.LIFT_PLATFORM: 350.0,
                AccessRequirement.ROPE_ACCESS: 500.0,
                AccessRequirement.CONFINED_SPACE: 400.0,
            }
        if not self.setup_cost:
            self.setup_cost = {
                AccessRequirement.GROUND_LEVEL: 0.0,
                AccessRequirement.LADDER: 50.0,
                AccessRequirement.SCAFFOLDING: 1500.0,
                AccessRequirement.LIFT_PLATFORM: 500.0,
                AccessRequirement.ROPE_ACCESS: 800.0,
                AccessRequirement.CONFINED_SPACE: 1200.0,
            }


@dataclass
class ThermalConductivity:
    """Thermal conductivity database (W/m-K at mean temperature)."""
    values: Dict[InsulationType, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.values:
            # k-values at different mean temperatures (50C, 150C, 300C)
            self.values = {
                InsulationType.MINERAL_WOOL: {"50C": 0.035, "150C": 0.045, "300C": 0.065},
                InsulationType.CALCIUM_SILICATE: {"50C": 0.055, "150C": 0.070, "300C": 0.095},
                InsulationType.CELLULAR_GLASS: {"50C": 0.040, "150C": 0.052, "300C": 0.075},
                InsulationType.PERLITE: {"50C": 0.050, "150C": 0.062, "300C": 0.085},
                InsulationType.FIBERGLASS: {"50C": 0.032, "150C": 0.042, "300C": 0.060},
                InsulationType.AEROGEL: {"50C": 0.015, "150C": 0.020, "300C": 0.028},
                InsulationType.POLYURETHANE_FOAM: {"50C": 0.025, "150C": 0.035, "300C": None},
                InsulationType.PHENOLIC_FOAM: {"50C": 0.022, "150C": 0.030, "300C": None},
                InsulationType.CERAMIC_FIBER: {"50C": 0.045, "150C": 0.065, "300C": 0.095},
                InsulationType.MICROPOROUS: {"50C": 0.020, "150C": 0.025, "300C": 0.032},
            }

    def get_k_value(self, insulation_type: InsulationType, mean_temp_c: float) -> float:
        """Interpolate k-value for given mean temperature."""
        k_data = self.values[insulation_type]

        if mean_temp_c <= 50:
            return k_data["50C"]
        elif mean_temp_c <= 150:
            # Linear interpolation between 50C and 150C
            t_ratio = (mean_temp_c - 50) / 100
            return k_data["50C"] + t_ratio * (k_data["150C"] - k_data["50C"])
        elif k_data.get("300C") is not None:
            # Linear interpolation between 150C and 300C
            t_ratio = (mean_temp_c - 150) / 150
            return k_data["150C"] + t_ratio * (k_data["300C"] - k_data["150C"])
        else:
            return k_data["150C"]  # Max temp for foam insulations


@dataclass
class FuelProperties:
    """Fuel properties database."""
    heating_values: Dict[FuelType, float] = field(default_factory=dict)  # MJ/unit
    unit_costs: Dict[FuelType, float] = field(default_factory=dict)      # USD/unit
    co2_factors: Dict[FuelType, float] = field(default_factory=dict)     # kg CO2/unit

    def __post_init__(self):
        if not self.heating_values:
            self.heating_values = {
                FuelType.NATURAL_GAS: 38.3,      # MJ/m3
                FuelType.FUEL_OIL: 38.6,         # MJ/liter
                FuelType.COAL: 24.0,             # MJ/kg
                FuelType.ELECTRICITY: 3.6,       # MJ/kWh
                FuelType.STEAM: 2.7,             # MJ/kg steam
                FuelType.LPG: 46.4,              # MJ/kg
                FuelType.BIOMASS: 18.0,          # MJ/kg
            }
        if not self.unit_costs:
            self.unit_costs = {
                FuelType.NATURAL_GAS: 0.35,      # USD/m3
                FuelType.FUEL_OIL: 0.75,         # USD/liter
                FuelType.COAL: 0.08,             # USD/kg
                FuelType.ELECTRICITY: 0.12,      # USD/kWh
                FuelType.STEAM: 0.035,           # USD/kg
                FuelType.LPG: 0.90,              # USD/kg
                FuelType.BIOMASS: 0.05,          # USD/kg
            }
        if not self.co2_factors:
            self.co2_factors = {
                FuelType.NATURAL_GAS: 1.89,      # kg CO2/m3
                FuelType.FUEL_OIL: 2.68,         # kg CO2/liter
                FuelType.COAL: 2.42,             # kg CO2/kg
                FuelType.ELECTRICITY: 0.45,      # kg CO2/kWh (grid average)
                FuelType.STEAM: 0.20,            # kg CO2/kg (boiler efficiency adjusted)
                FuelType.LPG: 2.98,              # kg CO2/kg
                FuelType.BIOMASS: 0.10,          # kg CO2/kg (biogenic)
            }


# =============================================================================
# INPUT MODELS
# =============================================================================

class RepairCostInput(BaseModel):
    """Input parameters for repair cost estimation."""
    equipment_type: EquipmentType = Field(..., description="Type of equipment")
    insulation_type: InsulationType = Field(..., description="Insulation material")
    repair_area_m2: float = Field(..., gt=0, description="Area requiring repair (m2)")
    thickness_mm: float = Field(..., gt=0, le=500, description="Insulation thickness (mm)")
    complexity: RepairComplexity = Field(RepairComplexity.MODERATE, description="Repair complexity")
    access_requirement: AccessRequirement = Field(
        AccessRequirement.GROUND_LEVEL, description="Access equipment needed"
    )
    region: str = Field("default", description="Geographic region for labor rates")
    estimated_duration_days: float = Field(1.0, gt=0, description="Estimated repair duration")
    requires_permit: bool = Field(False, description="Permit required")
    requires_hot_work: bool = Field(False, description="Hot work permit required")
    asbestos_abatement: bool = Field(False, description="Asbestos removal needed")
    quality_factor: float = Field(0.5, ge=0, le=1, description="Material quality (0=economy, 1=premium)")
    contingency_percent: float = Field(15.0, ge=0, le=50, description="Contingency percentage")

    @validator('thickness_mm')
    def validate_thickness(cls, v, values):
        """Validate thickness is reasonable for equipment type."""
        if v > 300 and values.get('equipment_type') == EquipmentType.PIPE:
            logger.warning(f"Unusually thick insulation for pipe: {v}mm")
        return v


class EnergySavingsInput(BaseModel):
    """Input parameters for energy savings calculation."""
    # Current state (damaged/missing insulation)
    current_heat_loss_w_per_m2: float = Field(..., gt=0, description="Current heat loss (W/m2)")
    surface_area_m2: float = Field(..., gt=0, description="Surface area (m2)")

    # Operating conditions
    process_temp_c: float = Field(..., description="Process temperature (C)")
    ambient_temp_c: float = Field(25.0, description="Ambient temperature (C)")
    operating_hours_per_year: float = Field(8000, ge=0, le=8760, description="Operating hours/year")

    # Post-repair state
    target_heat_loss_w_per_m2: Optional[float] = Field(
        None, gt=0, description="Target heat loss after repair (W/m2)"
    )
    new_insulation_type: Optional[InsulationType] = Field(None, description="New insulation type")
    new_thickness_mm: Optional[float] = Field(None, gt=0, description="New insulation thickness (mm)")

    # Fuel information
    fuel_type: FuelType = Field(FuelType.NATURAL_GAS, description="Fuel type for heating")
    fuel_cost_per_unit: Optional[float] = Field(None, gt=0, description="Custom fuel cost")
    boiler_efficiency: float = Field(0.85, ge=0.5, le=1.0, description="Boiler/heater efficiency")

    # Escalation
    fuel_price_escalation_percent: float = Field(3.0, ge=0, le=15, description="Annual fuel escalation")


class FinancialParameters(BaseModel):
    """Financial analysis parameters."""
    discount_rate_percent: float = Field(10.0, ge=0, le=30, description="Discount rate (%)")
    analysis_period_years: int = Field(20, ge=1, le=50, description="Analysis period (years)")
    inflation_rate_percent: float = Field(2.5, ge=0, le=15, description="General inflation (%)")
    tax_rate_percent: float = Field(25.0, ge=0, le=50, description="Corporate tax rate (%)")
    depreciation_method: DepreciationMethod = Field(
        DepreciationMethod.MACRS_7_YEAR, description="Depreciation method"
    )
    residual_value_percent: float = Field(0.0, ge=0, le=30, description="Residual value (%)")
    financing_rate_percent: Optional[float] = Field(None, ge=0, le=20, description="Financing rate (%)")
    financing_term_years: Optional[int] = Field(None, ge=1, le=20, description="Financing term (years)")


class LifecycleInput(BaseModel):
    """Input parameters for life cycle cost analysis."""
    equipment_life_years: int = Field(25, ge=5, le=50, description="Equipment remaining life (years)")
    insulation_life_years: int = Field(20, ge=5, le=40, description="Expected insulation life (years)")
    annual_inspection_cost: float = Field(0.0, ge=0, description="Annual inspection cost (USD)")
    maintenance_percent_of_capital: float = Field(2.0, ge=0, le=10, description="Annual maintenance (%)")
    replacement_cost_escalation_percent: float = Field(2.5, ge=0, le=10, description="Replacement cost escalation (%)")
    decommissioning_cost_percent: float = Field(5.0, ge=0, le=20, description="End-of-life cost (%)")


class CarbonEconomicsInput(BaseModel):
    """Input parameters for carbon economics calculation."""
    pricing_scheme: CarbonPricingScheme = Field(
        CarbonPricingScheme.VOLUNTARY, description="Carbon pricing scheme"
    )
    carbon_price_per_tonne: Optional[float] = Field(None, ge=0, description="Carbon price (USD/tonne)")
    carbon_price_escalation_percent: float = Field(5.0, ge=0, le=20, description="Carbon price escalation (%)")
    include_scope_2: bool = Field(True, description="Include Scope 2 (electricity) emissions")
    esg_reporting_value: float = Field(0.0, ge=0, description="ESG reporting benefit (USD/year)")


class EconomicThicknessInput(BaseModel):
    """Input parameters for ASTM C680 economic thickness calculation."""
    equipment_type: EquipmentType = Field(..., description="Equipment type")
    pipe_diameter_mm: Optional[float] = Field(None, gt=0, description="Pipe outer diameter (mm)")
    process_temp_c: float = Field(..., description="Process temperature (C)")
    ambient_temp_c: float = Field(25.0, description="Ambient temperature (C)")
    wind_speed_m_s: float = Field(0.0, ge=0, le=20, description="Wind speed (m/s)")
    insulation_type: InsulationType = Field(..., description="Insulation material")
    fuel_type: FuelType = Field(FuelType.NATURAL_GAS, description="Fuel type")
    fuel_cost_per_unit: Optional[float] = Field(None, gt=0, description="Custom fuel cost")
    operating_hours_per_year: float = Field(8000, ge=0, le=8760, description="Operating hours/year")
    analysis_period_years: int = Field(20, ge=5, le=50, description="Analysis period (years)")
    discount_rate_percent: float = Field(10.0, ge=0, le=30, description="Discount rate (%)")


# =============================================================================
# RESULT MODELS
# =============================================================================

class RepairCostResult(BaseModel):
    """Detailed repair cost breakdown."""
    # Material costs
    insulation_material_cost: float = Field(..., description="Insulation material cost (USD)")
    jacketing_cost: float = Field(..., description="Jacketing/cladding cost (USD)")
    accessories_cost: float = Field(..., description="Bands, fasteners, sealants (USD)")
    total_material_cost: float = Field(..., description="Total material cost (USD)")

    # Labor costs
    labor_hours: float = Field(..., description="Estimated labor hours")
    labor_rate: float = Field(..., description="Labor rate (USD/hour)")
    base_labor_cost: float = Field(..., description="Base labor cost (USD)")
    supervision_cost: float = Field(..., description="Supervision cost (USD)")
    total_labor_cost: float = Field(..., description="Total labor cost (USD)")

    # Access and equipment
    access_setup_cost: float = Field(..., description="Access setup cost (USD)")
    access_rental_cost: float = Field(..., description="Access rental cost (USD)")
    total_access_cost: float = Field(..., description="Total access cost (USD)")

    # Permits and safety
    permit_cost: float = Field(..., description="Permit costs (USD)")
    safety_equipment_cost: float = Field(..., description="Safety equipment cost (USD)")
    asbestos_abatement_cost: float = Field(..., description="Asbestos abatement cost (USD)")
    total_permit_safety_cost: float = Field(..., description="Total permit/safety cost (USD)")

    # Totals
    subtotal: float = Field(..., description="Subtotal before contingency (USD)")
    contingency: float = Field(..., description="Contingency amount (USD)")
    total_repair_cost: float = Field(..., description="Total repair cost (USD)")
    cost_per_m2: float = Field(..., description="Cost per square meter (USD/m2)")

    # Provenance
    calculation_hash: str = Field(..., description="SHA-256 calculation hash")


class EnergySavingsResult(BaseModel):
    """Energy savings calculation results."""
    # Heat loss values
    current_heat_loss_kw: float = Field(..., description="Current heat loss (kW)")
    post_repair_heat_loss_kw: float = Field(..., description="Post-repair heat loss (kW)")
    heat_loss_reduction_kw: float = Field(..., description="Heat loss reduction (kW)")
    heat_loss_reduction_percent: float = Field(..., description="Heat loss reduction (%)")

    # Annual energy
    annual_energy_loss_current_mj: float = Field(..., description="Current annual energy loss (MJ)")
    annual_energy_loss_repaired_mj: float = Field(..., description="Post-repair annual energy loss (MJ)")
    annual_energy_savings_mj: float = Field(..., description="Annual energy savings (MJ)")

    # Annual costs
    annual_fuel_cost_current: float = Field(..., description="Current annual fuel cost (USD)")
    annual_fuel_cost_repaired: float = Field(..., description="Post-repair annual fuel cost (USD)")
    first_year_savings: float = Field(..., description="First year savings (USD)")

    # Escalated savings (NPV basis)
    lifetime_savings_nominal: float = Field(..., description="Lifetime savings nominal (USD)")
    lifetime_savings_present_value: float = Field(..., description="Lifetime savings NPV (USD)")

    # Fuel consumption
    annual_fuel_reduction_units: float = Field(..., description="Annual fuel reduction (units)")
    fuel_unit: str = Field(..., description="Fuel unit of measure")

    # Provenance
    calculation_hash: str = Field(..., description="SHA-256 calculation hash")


class PaybackResult(BaseModel):
    """Payback analysis results."""
    # Simple payback
    simple_payback_years: float = Field(..., description="Simple payback period (years)")
    simple_payback_months: float = Field(..., description="Simple payback period (months)")

    # Discounted payback
    discounted_payback_years: float = Field(..., description="Discounted payback period (years)")

    # Break-even analysis
    break_even_fuel_price: float = Field(..., description="Break-even fuel price (USD/unit)")
    break_even_operating_hours: float = Field(..., description="Break-even operating hours/year")

    # Risk metrics
    payback_within_insulation_life: bool = Field(..., description="Payback within insulation life")
    payback_within_equipment_life: bool = Field(..., description="Payback within equipment life")

    # Provenance
    calculation_hash: str = Field(..., description="SHA-256 calculation hash")


class NPVIRRResult(BaseModel):
    """NPV and IRR calculation results."""
    # NPV metrics
    npv_usd: float = Field(..., description="Net Present Value (USD)")
    npv_per_m2: float = Field(..., description="NPV per square meter (USD/m2)")

    # IRR metrics
    irr_percent: float = Field(..., description="Internal Rate of Return (%)")
    modified_irr_percent: float = Field(..., description="Modified IRR (%)")

    # Profitability metrics
    profitability_index: float = Field(..., description="Profitability Index (PI)")
    benefit_cost_ratio: float = Field(..., description="Benefit-Cost Ratio (BCR)")

    # Cash flow analysis
    cumulative_cash_flow: List[float] = Field(..., description="Cumulative cash flow by year")
    annual_cash_flows: List[float] = Field(..., description="Annual cash flows")

    # Decision support
    is_economically_viable: bool = Field(..., description="Project is economically viable")
    recommendation: str = Field(..., description="Investment recommendation")

    # Provenance
    calculation_hash: str = Field(..., description="SHA-256 calculation hash")


class LifecycleCostResult(BaseModel):
    """Life cycle cost analysis results."""
    # Initial costs
    initial_capital_cost: float = Field(..., description="Initial capital cost (USD)")

    # Operating costs (NPV)
    total_energy_cost_npv: float = Field(..., description="Total energy cost NPV (USD)")
    total_maintenance_cost_npv: float = Field(..., description="Total maintenance cost NPV (USD)")
    total_inspection_cost_npv: float = Field(..., description="Total inspection cost NPV (USD)")

    # Replacement costs (NPV)
    replacement_cost_npv: float = Field(..., description="Replacement cost NPV (USD)")
    number_of_replacements: int = Field(..., description="Number of replacements")

    # End-of-life
    decommissioning_cost_npv: float = Field(..., description="Decommissioning cost NPV (USD)")
    residual_value_npv: float = Field(..., description="Residual value NPV (USD)")

    # Total cost of ownership
    total_cost_of_ownership: float = Field(..., description="Total cost of ownership (USD)")
    annualized_cost: float = Field(..., description="Annualized cost (USD/year)")
    cost_per_m2_per_year: float = Field(..., description="Cost per m2 per year (USD/m2/year)")

    # Comparison metrics
    do_nothing_tco: float = Field(..., description="Do-nothing TCO (USD)")
    repair_tco: float = Field(..., description="Repair option TCO (USD)")
    savings_vs_do_nothing: float = Field(..., description="Savings vs do-nothing (USD)")

    # Year-by-year breakdown
    annual_cost_breakdown: List[Dict[str, float]] = Field(..., description="Annual cost breakdown")

    # Provenance
    calculation_hash: str = Field(..., description="SHA-256 calculation hash")


class EconomicThicknessResult(BaseModel):
    """Economic thickness optimization results per ASTM C680."""
    # Optimal thickness
    economic_thickness_mm: float = Field(..., description="Economic thickness (mm)")
    recommended_thickness_mm: float = Field(..., description="Recommended standard thickness (mm)")

    # Cost analysis at economic thickness
    installed_cost_at_economic: float = Field(..., description="Installed cost at economic thickness (USD/m)")
    annual_energy_cost_at_economic: float = Field(..., description="Annual energy cost (USD/m/year)")
    total_annual_cost_at_economic: float = Field(..., description="Total annual cost (USD/m/year)")

    # Thickness sensitivity
    thickness_analysis: List[Dict[str, float]] = Field(..., description="Cost vs thickness analysis")

    # Diminishing returns
    marginal_savings_per_mm: float = Field(..., description="Marginal savings per additional mm")
    diminishing_returns_thickness_mm: float = Field(..., description="Thickness at diminishing returns")

    # Upgrade vs repair decision
    current_thickness_mm: Optional[float] = Field(None, description="Current thickness (mm)")
    upgrade_recommended: bool = Field(..., description="Upgrade to thicker insulation recommended")
    upgrade_roi_percent: Optional[float] = Field(None, description="Upgrade ROI (%)")

    # Surface temperature
    surface_temp_at_economic_c: float = Field(..., description="Surface temperature at economic thickness (C)")

    # Provenance
    calculation_hash: str = Field(..., description="SHA-256 calculation hash")


class BudgetImpactResult(BaseModel):
    """Budget impact analysis results."""
    # Capital expenditure
    total_capex: float = Field(..., description="Total capital expenditure (USD)")
    capex_by_category: Dict[str, float] = Field(..., description="CapEx breakdown by category")

    # Operating expense impact
    annual_opex_reduction: float = Field(..., description="Annual OpEx reduction (USD)")
    opex_impact_by_year: List[float] = Field(..., description="OpEx impact by year")

    # Cash flow projection
    monthly_cash_flow: List[Dict[str, float]] = Field(..., description="Monthly cash flow (first year)")
    annual_cash_flow: List[Dict[str, float]] = Field(..., description="Annual cash flow projection")

    # Financing analysis
    financing_required: bool = Field(..., description="Financing required")
    monthly_payment: Optional[float] = Field(None, description="Monthly financing payment (USD)")
    total_interest_cost: Optional[float] = Field(None, description="Total interest cost (USD)")

    # Budget summary
    year_1_net_cash_impact: float = Field(..., description="Year 1 net cash impact (USD)")
    break_even_year: int = Field(..., description="Break-even year")
    cumulative_benefit_year_5: float = Field(..., description="Cumulative benefit at Year 5 (USD)")

    # Funding sources
    recommended_funding_source: str = Field(..., description="Recommended funding source")

    # Provenance
    calculation_hash: str = Field(..., description="SHA-256 calculation hash")


class CarbonEconomicsResult(BaseModel):
    """Carbon economics calculation results."""
    # Emissions reduction
    annual_co2_reduction_tonnes: float = Field(..., description="Annual CO2 reduction (tonnes)")
    lifetime_co2_reduction_tonnes: float = Field(..., description="Lifetime CO2 reduction (tonnes)")

    # Carbon value
    carbon_credit_value_annual: float = Field(..., description="Annual carbon credit value (USD)")
    carbon_credit_value_lifetime: float = Field(..., description="Lifetime carbon credit value (USD)")
    carbon_credit_value_npv: float = Field(..., description="Carbon credit NPV (USD)")

    # Carbon liability avoided
    carbon_tax_avoided_annual: float = Field(..., description="Annual carbon tax avoided (USD)")
    carbon_tax_avoided_lifetime: float = Field(..., description="Lifetime carbon tax avoided (USD)")

    # Marginal abatement cost
    marginal_abatement_cost: float = Field(..., description="Marginal abatement cost (USD/tonne CO2)")

    # ESG benefits
    esg_reporting_benefit: float = Field(..., description="ESG reporting benefit (USD)")
    sustainability_score_improvement: float = Field(..., description="Sustainability score improvement (%)")

    # Total carbon economics value
    total_carbon_value_annual: float = Field(..., description="Total carbon value annual (USD)")
    total_carbon_value_npv: float = Field(..., description="Total carbon value NPV (USD)")

    # Provenance
    calculation_hash: str = Field(..., description="SHA-256 calculation hash")


# =============================================================================
# MAIN CALCULATOR CLASS
# =============================================================================

class EconomicCalculator:
    """
    Zero-hallucination Economic Impact Calculator for Insulation Assessment.

    All calculations are deterministic and physics/economics-based.
    Complete provenance tracking with SHA-256 hashes.
    Performance target: <5ms per calculation.

    Standards Compliance:
    - ASTM C680 (Economic Thickness of Thermal Insulation)
    - 3E Plus (NAIMA Economic Thickness)
    - CINI Manual (Cost Estimating)
    """

    # MACRS depreciation schedules
    MACRS_SCHEDULES = {
        DepreciationMethod.MACRS_5_YEAR: [0.20, 0.32, 0.192, 0.1152, 0.1152, 0.0576],
        DepreciationMethod.MACRS_7_YEAR: [0.1429, 0.2449, 0.1749, 0.1249, 0.0893, 0.0892, 0.0893, 0.0446],
        DepreciationMethod.MACRS_15_YEAR: [0.05, 0.095, 0.0855, 0.077, 0.0693, 0.0623,
                                           0.059, 0.059, 0.0591, 0.059, 0.0591, 0.059,
                                           0.0591, 0.059, 0.0591, 0.0295],
    }

    # Carbon prices by scheme (USD/tonne CO2)
    CARBON_PRICES = {
        CarbonPricingScheme.EU_ETS: 85.0,
        CarbonPricingScheme.CBAM: 90.0,
        CarbonPricingScheme.US_RGGI: 15.0,
        CarbonPricingScheme.CALIFORNIA_CAP: 35.0,
        CarbonPricingScheme.CARBON_TAX: 50.0,
        CarbonPricingScheme.VOLUNTARY: 25.0,
    }

    # Standard insulation thicknesses (mm)
    STANDARD_THICKNESSES = [25, 38, 50, 65, 75, 100, 125, 150, 175, 200, 250, 300]

    def __init__(self):
        """Initialize economic calculator with cost databases."""
        self.material_costs = InsulationMaterialCosts()
        self.labor_rates = LaborRates()
        self.access_costs = AccessCosts()
        self.thermal_conductivity = ThermalConductivity()
        self.fuel_properties = FuelProperties()
        self.logger = logging.getLogger(__name__)

    # =========================================================================
    # 1. REPAIR COST ESTIMATION
    # =========================================================================

    def estimate_repair_cost(self, inputs: RepairCostInput) -> RepairCostResult:
        """
        Estimate comprehensive repair costs including materials, labor, access, and permits.

        Args:
            inputs: Repair cost input parameters

        Returns:
            Detailed repair cost breakdown with provenance hash
        """
        self.logger.info(f"Estimating repair cost for {inputs.repair_area_m2} m2 {inputs.insulation_type.value}")

        # Calculate volume for material costs
        volume_m3 = inputs.repair_area_m2 * (inputs.thickness_mm / 1000)

        # Material costs
        material_cost_per_m3 = self.material_costs.get_cost(
            inputs.insulation_type, inputs.quality_factor
        )
        insulation_material_cost = Decimal(str(volume_m3 * material_cost_per_m3))

        # Jacketing cost (typically 30-50% of insulation cost for metal jacketing)
        jacketing_factor = 0.40
        jacketing_cost = insulation_material_cost * Decimal(str(jacketing_factor))

        # Accessories (bands, fasteners, sealants, vapor barriers) - 15% of materials
        accessories_cost = (insulation_material_cost + jacketing_cost) * Decimal("0.15")

        total_material_cost = insulation_material_cost + jacketing_cost + accessories_cost

        # Labor costs
        base_labor_rate = self.labor_rates.base_rates[inputs.complexity]
        regional_multiplier = self.labor_rates.regional_multipliers.get(
            inputs.region, self.labor_rates.regional_multipliers["default"]
        )
        labor_rate = Decimal(str(base_labor_rate * regional_multiplier))

        # Labor hours based on complexity and area
        labor_productivity = {
            RepairComplexity.SIMPLE: 0.5,      # m2/hour
            RepairComplexity.MODERATE: 0.35,
            RepairComplexity.COMPLEX: 0.25,
            RepairComplexity.CRITICAL: 0.15,
        }
        productivity = labor_productivity[inputs.complexity]
        labor_hours = Decimal(str(inputs.repair_area_m2 / productivity))

        base_labor_cost = labor_hours * labor_rate

        # Supervision (15% of labor for complex, 10% otherwise)
        supervision_factor = Decimal("0.15") if inputs.complexity in [
            RepairComplexity.COMPLEX, RepairComplexity.CRITICAL
        ] else Decimal("0.10")
        supervision_cost = base_labor_cost * supervision_factor

        total_labor_cost = base_labor_cost + supervision_cost

        # Access costs
        access_setup = Decimal(str(self.access_costs.setup_cost[inputs.access_requirement]))
        access_daily = Decimal(str(self.access_costs.daily_rental[inputs.access_requirement]))
        access_rental = access_daily * Decimal(str(inputs.estimated_duration_days))
        total_access_cost = access_setup + access_rental

        # Permits and safety
        permit_cost = Decimal("0")
        if inputs.requires_permit:
            permit_cost += Decimal("500")  # Base permit fee
        if inputs.requires_hot_work:
            permit_cost += Decimal("750")  # Hot work permit and fire watch

        # Safety equipment
        safety_cost = Decimal("200") if inputs.complexity in [
            RepairComplexity.COMPLEX, RepairComplexity.CRITICAL
        ] else Decimal("50")

        # Asbestos abatement (significant cost if required)
        asbestos_cost = Decimal("0")
        if inputs.asbestos_abatement:
            asbestos_cost = Decimal(str(inputs.repair_area_m2 * 150))  # ~$150/m2 for abatement

        total_permit_safety = permit_cost + safety_cost + asbestos_cost

        # Subtotal
        subtotal = total_material_cost + total_labor_cost + total_access_cost + total_permit_safety

        # Contingency
        contingency = subtotal * Decimal(str(inputs.contingency_percent / 100))

        # Total
        total_repair_cost = subtotal + contingency
        cost_per_m2 = total_repair_cost / Decimal(str(inputs.repair_area_m2))

        # Calculate provenance hash
        calc_hash = self._calculate_hash({
            "inputs": inputs.dict(),
            "total": str(total_repair_cost),
        })

        return RepairCostResult(
            insulation_material_cost=float(insulation_material_cost.quantize(Decimal("0.01"))),
            jacketing_cost=float(jacketing_cost.quantize(Decimal("0.01"))),
            accessories_cost=float(accessories_cost.quantize(Decimal("0.01"))),
            total_material_cost=float(total_material_cost.quantize(Decimal("0.01"))),
            labor_hours=float(labor_hours.quantize(Decimal("0.1"))),
            labor_rate=float(labor_rate.quantize(Decimal("0.01"))),
            base_labor_cost=float(base_labor_cost.quantize(Decimal("0.01"))),
            supervision_cost=float(supervision_cost.quantize(Decimal("0.01"))),
            total_labor_cost=float(total_labor_cost.quantize(Decimal("0.01"))),
            access_setup_cost=float(access_setup.quantize(Decimal("0.01"))),
            access_rental_cost=float(access_rental.quantize(Decimal("0.01"))),
            total_access_cost=float(total_access_cost.quantize(Decimal("0.01"))),
            permit_cost=float(permit_cost.quantize(Decimal("0.01"))),
            safety_equipment_cost=float(safety_cost.quantize(Decimal("0.01"))),
            asbestos_abatement_cost=float(asbestos_cost.quantize(Decimal("0.01"))),
            total_permit_safety_cost=float(total_permit_safety.quantize(Decimal("0.01"))),
            subtotal=float(subtotal.quantize(Decimal("0.01"))),
            contingency=float(contingency.quantize(Decimal("0.01"))),
            total_repair_cost=float(total_repair_cost.quantize(Decimal("0.01"))),
            cost_per_m2=float(cost_per_m2.quantize(Decimal("0.01"))),
            calculation_hash=calc_hash,
        )

    # =========================================================================
    # 2. ENERGY SAVINGS CALCULATION
    # =========================================================================

    def calculate_energy_savings(
        self,
        inputs: EnergySavingsInput,
        financial_params: FinancialParameters
    ) -> EnergySavingsResult:
        """
        Calculate energy savings from insulation repair/upgrade.

        Args:
            inputs: Energy savings input parameters
            financial_params: Financial analysis parameters

        Returns:
            Energy savings calculation results with provenance
        """
        self.logger.info("Calculating energy savings")

        # Current heat loss
        current_heat_loss_w = inputs.current_heat_loss_w_per_m2 * inputs.surface_area_m2
        current_heat_loss_kw = current_heat_loss_w / 1000

        # Post-repair heat loss
        if inputs.target_heat_loss_w_per_m2:
            post_repair_heat_loss_w = inputs.target_heat_loss_w_per_m2 * inputs.surface_area_m2
        elif inputs.new_insulation_type and inputs.new_thickness_mm:
            # Calculate from insulation properties
            mean_temp = (inputs.process_temp_c + inputs.ambient_temp_c) / 2
            k_value = self.thermal_conductivity.get_k_value(inputs.new_insulation_type, mean_temp)

            # Simplified flat surface heat loss: Q = k * A * dT / t
            delta_t = inputs.process_temp_c - inputs.ambient_temp_c
            thickness_m = inputs.new_thickness_mm / 1000
            post_repair_heat_loss_w = (k_value * inputs.surface_area_m2 * delta_t) / thickness_m * 1000
        else:
            # Default: assume 90% reduction in heat loss
            post_repair_heat_loss_w = current_heat_loss_w * 0.10

        post_repair_heat_loss_kw = post_repair_heat_loss_w / 1000

        # Heat loss reduction
        heat_loss_reduction_kw = current_heat_loss_kw - post_repair_heat_loss_kw
        heat_loss_reduction_percent = (heat_loss_reduction_kw / current_heat_loss_kw * 100) if current_heat_loss_kw > 0 else 0

        # Annual energy loss (MJ)
        hours_to_seconds = 3600
        annual_energy_loss_current_mj = current_heat_loss_kw * inputs.operating_hours_per_year * 3.6  # kWh to MJ
        annual_energy_loss_repaired_mj = post_repair_heat_loss_kw * inputs.operating_hours_per_year * 3.6
        annual_energy_savings_mj = annual_energy_loss_current_mj - annual_energy_loss_repaired_mj

        # Fuel consumption and costs
        fuel_props = self.fuel_properties
        heating_value = fuel_props.heating_values[inputs.fuel_type]
        fuel_cost = inputs.fuel_cost_per_unit or fuel_props.unit_costs[inputs.fuel_type]

        # Account for boiler efficiency
        annual_fuel_current = annual_energy_loss_current_mj / (heating_value * inputs.boiler_efficiency)
        annual_fuel_repaired = annual_energy_loss_repaired_mj / (heating_value * inputs.boiler_efficiency)
        annual_fuel_reduction = annual_fuel_current - annual_fuel_repaired

        # Fuel unit
        fuel_units = {
            FuelType.NATURAL_GAS: "m3",
            FuelType.FUEL_OIL: "liters",
            FuelType.COAL: "kg",
            FuelType.ELECTRICITY: "kWh",
            FuelType.STEAM: "kg",
            FuelType.LPG: "kg",
            FuelType.BIOMASS: "kg",
        }

        # Annual costs
        annual_fuel_cost_current = annual_fuel_current * fuel_cost
        annual_fuel_cost_repaired = annual_fuel_repaired * fuel_cost
        first_year_savings = annual_fuel_cost_current - annual_fuel_cost_repaired

        # Lifetime savings with escalation
        discount_rate = financial_params.discount_rate_percent / 100
        escalation_rate = inputs.fuel_price_escalation_percent / 100

        lifetime_savings_nominal = Decimal("0")
        lifetime_savings_pv = Decimal("0")

        for year in range(1, financial_params.analysis_period_years + 1):
            # Escalated savings
            escalated_savings = Decimal(str(first_year_savings)) * Decimal(str((1 + escalation_rate) ** (year - 1)))
            lifetime_savings_nominal += escalated_savings

            # Discounted savings
            discount_factor = Decimal(str((1 + discount_rate) ** year))
            lifetime_savings_pv += escalated_savings / discount_factor

        # Calculate provenance hash
        calc_hash = self._calculate_hash({
            "inputs": inputs.dict(),
            "first_year_savings": str(first_year_savings),
            "lifetime_pv": str(lifetime_savings_pv),
        })

        return EnergySavingsResult(
            current_heat_loss_kw=round(current_heat_loss_kw, 3),
            post_repair_heat_loss_kw=round(post_repair_heat_loss_kw, 3),
            heat_loss_reduction_kw=round(heat_loss_reduction_kw, 3),
            heat_loss_reduction_percent=round(heat_loss_reduction_percent, 1),
            annual_energy_loss_current_mj=round(annual_energy_loss_current_mj, 1),
            annual_energy_loss_repaired_mj=round(annual_energy_loss_repaired_mj, 1),
            annual_energy_savings_mj=round(annual_energy_savings_mj, 1),
            annual_fuel_cost_current=round(annual_fuel_cost_current, 2),
            annual_fuel_cost_repaired=round(annual_fuel_cost_repaired, 2),
            first_year_savings=round(first_year_savings, 2),
            lifetime_savings_nominal=float(lifetime_savings_nominal.quantize(Decimal("0.01"))),
            lifetime_savings_present_value=float(lifetime_savings_pv.quantize(Decimal("0.01"))),
            annual_fuel_reduction_units=round(annual_fuel_reduction, 2),
            fuel_unit=fuel_units[inputs.fuel_type],
            calculation_hash=calc_hash,
        )

    # =========================================================================
    # 3. PAYBACK ANALYSIS
    # =========================================================================

    def calculate_payback_period(
        self,
        repair_cost: RepairCostResult,
        energy_savings: EnergySavingsResult,
        financial_params: FinancialParameters,
        lifecycle_inputs: LifecycleInput
    ) -> PaybackResult:
        """
        Calculate simple and discounted payback periods with break-even analysis.

        Args:
            repair_cost: Repair cost calculation result
            energy_savings: Energy savings calculation result
            financial_params: Financial parameters
            lifecycle_inputs: Lifecycle analysis inputs

        Returns:
            Payback analysis results with provenance
        """
        self.logger.info("Calculating payback period")

        total_investment = Decimal(str(repair_cost.total_repair_cost))
        annual_savings = Decimal(str(energy_savings.first_year_savings))

        # Simple payback
        if annual_savings > 0:
            simple_payback_years = float(total_investment / annual_savings)
            simple_payback_months = simple_payback_years * 12
        else:
            simple_payback_years = 999.0
            simple_payback_months = 999.0 * 12

        # Discounted payback
        discount_rate = Decimal(str(financial_params.discount_rate_percent / 100))
        escalation_rate = Decimal("0.03")  # Assume 3% fuel escalation

        cumulative_pv = Decimal("0")
        discounted_payback_years = 999.0

        for year in range(1, 51):  # Max 50 years
            escalated_savings = annual_savings * ((1 + escalation_rate) ** (year - 1))
            pv = escalated_savings / ((1 + discount_rate) ** year)
            cumulative_pv += pv

            if cumulative_pv >= total_investment:
                # Interpolate for fractional year
                previous_cumulative = cumulative_pv - pv
                fraction = float((total_investment - previous_cumulative) / pv)
                discounted_payback_years = year - 1 + fraction
                break

        # Break-even fuel price
        # At break-even, annual savings (at new fuel price) = annualized investment cost
        annualized_investment = float(total_investment) / financial_params.analysis_period_years
        fuel_reduction = energy_savings.annual_fuel_reduction_units

        if fuel_reduction > 0:
            break_even_fuel_price = annualized_investment / fuel_reduction
        else:
            break_even_fuel_price = 999.0

        # Break-even operating hours
        # Savings are proportional to operating hours
        if energy_savings.first_year_savings > 0:
            operating_hours_ratio = annualized_investment / energy_savings.first_year_savings
            break_even_hours = operating_hours_ratio * 8000  # Assuming 8000 base hours
        else:
            break_even_hours = 999.0

        # Risk metrics
        payback_within_insulation = simple_payback_years <= lifecycle_inputs.insulation_life_years
        payback_within_equipment = simple_payback_years <= lifecycle_inputs.equipment_life_years

        # Calculate provenance hash
        calc_hash = self._calculate_hash({
            "repair_cost": repair_cost.total_repair_cost,
            "annual_savings": energy_savings.first_year_savings,
            "simple_payback": simple_payback_years,
            "discounted_payback": discounted_payback_years,
        })

        return PaybackResult(
            simple_payback_years=round(simple_payback_years, 2),
            simple_payback_months=round(simple_payback_months, 1),
            discounted_payback_years=round(discounted_payback_years, 2),
            break_even_fuel_price=round(break_even_fuel_price, 4),
            break_even_operating_hours=round(break_even_hours, 0),
            payback_within_insulation_life=payback_within_insulation,
            payback_within_equipment_life=payback_within_equipment,
            calculation_hash=calc_hash,
        )

    # =========================================================================
    # 4. NPV AND IRR CALCULATION
    # =========================================================================

    def calculate_npv_irr(
        self,
        repair_cost: RepairCostResult,
        energy_savings: EnergySavingsResult,
        financial_params: FinancialParameters,
        surface_area_m2: float
    ) -> NPVIRRResult:
        """
        Calculate Net Present Value, Internal Rate of Return, and related metrics.

        Args:
            repair_cost: Repair cost calculation result
            energy_savings: Energy savings calculation result
            financial_params: Financial parameters
            surface_area_m2: Surface area for per-m2 calculations

        Returns:
            NPV/IRR calculation results with provenance
        """
        self.logger.info("Calculating NPV and IRR")

        initial_investment = Decimal(str(repair_cost.total_repair_cost))
        annual_savings_base = Decimal(str(energy_savings.first_year_savings))

        discount_rate = Decimal(str(financial_params.discount_rate_percent / 100))
        escalation_rate = Decimal("0.03")  # 3% fuel price escalation

        # Build cash flow series
        annual_cash_flows = [-float(initial_investment)]
        cumulative_cash_flow = [-float(initial_investment)]

        npv = -initial_investment

        for year in range(1, financial_params.analysis_period_years + 1):
            # Escalated savings
            escalated_savings = annual_savings_base * ((1 + escalation_rate) ** (year - 1))

            # Maintenance costs (2% of capital)
            maintenance = initial_investment * Decimal("0.02")

            # Net cash flow for year
            net_cf = escalated_savings - maintenance
            annual_cash_flows.append(float(net_cf))

            # Cumulative
            cumulative = Decimal(str(cumulative_cash_flow[-1])) + net_cf
            cumulative_cash_flow.append(float(cumulative))

            # NPV contribution
            pv = net_cf / ((1 + discount_rate) ** year)
            npv += pv

        # Add residual value in final year
        residual_value = initial_investment * Decimal(str(financial_params.residual_value_percent / 100))
        residual_pv = residual_value / ((1 + discount_rate) ** financial_params.analysis_period_years)
        npv += residual_pv

        # Calculate IRR using Newton-Raphson method
        irr = self._calculate_irr(annual_cash_flows)

        # Modified IRR (MIRR)
        mirr = self._calculate_mirr(
            annual_cash_flows,
            float(discount_rate),
            float(discount_rate) + 0.02  # Reinvestment rate
        )

        # Profitability Index
        total_pv_benefits = npv + initial_investment
        profitability_index = float(total_pv_benefits / initial_investment) if initial_investment > 0 else 0

        # Benefit-Cost Ratio
        bcr = float(total_pv_benefits / initial_investment) if initial_investment > 0 else 0

        # NPV per m2
        npv_per_m2 = float(npv) / surface_area_m2 if surface_area_m2 > 0 else 0

        # Economic viability check
        is_viable = float(npv) > 0 and irr > float(discount_rate) * 100

        # Recommendation
        if float(npv) > 0 and irr > 15:
            recommendation = "HIGHLY RECOMMENDED - Strong positive economics"
        elif float(npv) > 0 and irr > 10:
            recommendation = "RECOMMENDED - Positive economics"
        elif float(npv) > 0:
            recommendation = "MARGINALLY RECOMMENDED - Modest positive economics"
        else:
            recommendation = "NOT RECOMMENDED - Negative NPV"

        # Calculate provenance hash
        calc_hash = self._calculate_hash({
            "initial_investment": str(initial_investment),
            "npv": str(npv),
            "irr": irr,
        })

        return NPVIRRResult(
            npv_usd=float(npv.quantize(Decimal("0.01"))),
            npv_per_m2=round(npv_per_m2, 2),
            irr_percent=round(irr, 2),
            modified_irr_percent=round(mirr, 2),
            profitability_index=round(profitability_index, 3),
            benefit_cost_ratio=round(bcr, 3),
            cumulative_cash_flow=cumulative_cash_flow,
            annual_cash_flows=annual_cash_flows,
            is_economically_viable=is_viable,
            recommendation=recommendation,
            calculation_hash=calc_hash,
        )

    def _calculate_irr(self, cash_flows: List[float], max_iterations: int = 100) -> float:
        """Calculate IRR using Newton-Raphson method."""
        if len(cash_flows) < 2:
            return 0.0

        # Initial guess
        irr = 0.10

        for _ in range(max_iterations):
            npv = 0.0
            npv_derivative = 0.0

            for year, cf in enumerate(cash_flows):
                discount_factor = (1 + irr) ** year
                if discount_factor != 0:
                    npv += cf / discount_factor
                    if year > 0:
                        npv_derivative -= year * cf / ((1 + irr) ** (year + 1))

            # Check convergence
            if abs(npv) < 0.01:
                break

            # Newton-Raphson update
            if npv_derivative != 0:
                irr = irr - npv / npv_derivative
            else:
                break

            # Bounds check
            irr = max(-0.99, min(irr, 10.0))

        return irr * 100  # Convert to percentage

    def _calculate_mirr(
        self,
        cash_flows: List[float],
        finance_rate: float,
        reinvest_rate: float
    ) -> float:
        """Calculate Modified Internal Rate of Return."""
        n = len(cash_flows) - 1
        if n <= 0:
            return 0.0

        # Separate positive and negative cash flows
        positive_cfs = [max(0, cf) for cf in cash_flows]
        negative_cfs = [min(0, cf) for cf in cash_flows]

        # Future value of positive cash flows (reinvestment)
        fv_positive = sum(
            cf * ((1 + reinvest_rate) ** (n - t))
            for t, cf in enumerate(positive_cfs)
        )

        # Present value of negative cash flows (financing)
        pv_negative = sum(
            cf / ((1 + finance_rate) ** t)
            for t, cf in enumerate(negative_cfs)
        )

        if pv_negative == 0 or fv_positive <= 0:
            return 0.0

        # MIRR formula
        mirr = ((fv_positive / abs(pv_negative)) ** (1 / n)) - 1

        return mirr * 100  # Convert to percentage

    # =========================================================================
    # 5. LIFE CYCLE COST ANALYSIS
    # =========================================================================

    def perform_lifecycle_cost_analysis(
        self,
        repair_cost: RepairCostResult,
        energy_savings: EnergySavingsResult,
        financial_params: FinancialParameters,
        lifecycle_inputs: LifecycleInput,
        surface_area_m2: float
    ) -> LifecycleCostResult:
        """
        Perform comprehensive life cycle cost analysis (LCCA).

        Args:
            repair_cost: Repair cost result
            energy_savings: Energy savings result
            financial_params: Financial parameters
            lifecycle_inputs: Lifecycle inputs
            surface_area_m2: Surface area

        Returns:
            Life cycle cost analysis results with provenance
        """
        self.logger.info("Performing life cycle cost analysis")

        analysis_years = min(
            financial_params.analysis_period_years,
            lifecycle_inputs.equipment_life_years
        )

        discount_rate = financial_params.discount_rate_percent / 100
        inflation_rate = financial_params.inflation_rate_percent / 100
        fuel_escalation = 0.03  # 3%

        initial_capital = Decimal(str(repair_cost.total_repair_cost))
        annual_energy_cost_base = Decimal(str(energy_savings.annual_fuel_cost_repaired))
        maintenance_rate = Decimal(str(lifecycle_inputs.maintenance_percent_of_capital / 100))
        inspection_cost = Decimal(str(lifecycle_inputs.annual_inspection_cost))

        # Calculate NPV of operating costs
        total_energy_cost_npv = Decimal("0")
        total_maintenance_cost_npv = Decimal("0")
        total_inspection_cost_npv = Decimal("0")

        annual_cost_breakdown = []

        for year in range(1, analysis_years + 1):
            # Escalated costs
            energy_cost = annual_energy_cost_base * Decimal(str((1 + fuel_escalation) ** (year - 1)))
            maintenance_cost = initial_capital * maintenance_rate * Decimal(str((1 + inflation_rate) ** (year - 1)))
            inspection = inspection_cost * Decimal(str((1 + inflation_rate) ** (year - 1)))

            # Discount factor
            df = Decimal(str((1 + discount_rate) ** year))

            # NPV contributions
            total_energy_cost_npv += energy_cost / df
            total_maintenance_cost_npv += maintenance_cost / df
            total_inspection_cost_npv += inspection / df

            annual_cost_breakdown.append({
                "year": year,
                "energy_cost": float(energy_cost),
                "maintenance_cost": float(maintenance_cost),
                "inspection_cost": float(inspection),
                "total_cost": float(energy_cost + maintenance_cost + inspection),
            })

        # Replacement costs
        num_replacements = (analysis_years - 1) // lifecycle_inputs.insulation_life_years
        replacement_cost_npv = Decimal("0")

        for i in range(1, num_replacements + 1):
            replacement_year = i * lifecycle_inputs.insulation_life_years
            if replacement_year < analysis_years:
                escalated_cost = initial_capital * Decimal(str(
                    (1 + lifecycle_inputs.replacement_cost_escalation_percent / 100) ** replacement_year
                ))
                df = Decimal(str((1 + discount_rate) ** replacement_year))
                replacement_cost_npv += escalated_cost / df

        # Decommissioning cost
        decommissioning_cost = initial_capital * Decimal(str(lifecycle_inputs.decommissioning_cost_percent / 100))
        decommissioning_df = Decimal(str((1 + discount_rate) ** analysis_years))
        decommissioning_cost_npv = decommissioning_cost / decommissioning_df

        # Residual value
        residual_value = initial_capital * Decimal(str(financial_params.residual_value_percent / 100))
        residual_value_npv = residual_value / decommissioning_df

        # Total cost of ownership
        total_tco = (
            initial_capital +
            total_energy_cost_npv +
            total_maintenance_cost_npv +
            total_inspection_cost_npv +
            replacement_cost_npv +
            decommissioning_cost_npv -
            residual_value_npv
        )

        # Annualized cost
        # Using capital recovery factor: CRF = r(1+r)^n / ((1+r)^n - 1)
        r = Decimal(str(discount_rate))
        n = analysis_years
        crf = (r * (1 + r) ** n) / ((1 + r) ** n - 1)
        annualized_cost = total_tco * crf

        cost_per_m2_per_year = annualized_cost / Decimal(str(surface_area_m2))

        # Do-nothing scenario TCO
        do_nothing_energy_npv = Decimal("0")
        for year in range(1, analysis_years + 1):
            energy_cost = Decimal(str(energy_savings.annual_fuel_cost_current)) * Decimal(str((1 + fuel_escalation) ** (year - 1)))
            df = Decimal(str((1 + discount_rate) ** year))
            do_nothing_energy_npv += energy_cost / df

        do_nothing_tco = do_nothing_energy_npv  # Simplified: just energy costs
        repair_tco = total_tco
        savings_vs_do_nothing = do_nothing_tco - repair_tco

        # Calculate provenance hash
        calc_hash = self._calculate_hash({
            "initial_capital": str(initial_capital),
            "total_tco": str(total_tco),
            "savings": str(savings_vs_do_nothing),
        })

        return LifecycleCostResult(
            initial_capital_cost=float(initial_capital.quantize(Decimal("0.01"))),
            total_energy_cost_npv=float(total_energy_cost_npv.quantize(Decimal("0.01"))),
            total_maintenance_cost_npv=float(total_maintenance_cost_npv.quantize(Decimal("0.01"))),
            total_inspection_cost_npv=float(total_inspection_cost_npv.quantize(Decimal("0.01"))),
            replacement_cost_npv=float(replacement_cost_npv.quantize(Decimal("0.01"))),
            number_of_replacements=num_replacements,
            decommissioning_cost_npv=float(decommissioning_cost_npv.quantize(Decimal("0.01"))),
            residual_value_npv=float(residual_value_npv.quantize(Decimal("0.01"))),
            total_cost_of_ownership=float(total_tco.quantize(Decimal("0.01"))),
            annualized_cost=float(annualized_cost.quantize(Decimal("0.01"))),
            cost_per_m2_per_year=float(cost_per_m2_per_year.quantize(Decimal("0.01"))),
            do_nothing_tco=float(do_nothing_tco.quantize(Decimal("0.01"))),
            repair_tco=float(repair_tco.quantize(Decimal("0.01"))),
            savings_vs_do_nothing=float(savings_vs_do_nothing.quantize(Decimal("0.01"))),
            annual_cost_breakdown=annual_cost_breakdown,
            calculation_hash=calc_hash,
        )

    # =========================================================================
    # 6. ECONOMIC THICKNESS OPTIMIZATION (ASTM C680)
    # =========================================================================

    def calculate_economic_thickness(
        self,
        inputs: EconomicThicknessInput,
        current_thickness_mm: Optional[float] = None
    ) -> EconomicThicknessResult:
        """
        Calculate economic thickness per ASTM C680 methodology.

        The economic thickness is where total annual cost (insulation + heat loss)
        is minimized.

        Args:
            inputs: Economic thickness input parameters
            current_thickness_mm: Current insulation thickness (if upgrading)

        Returns:
            Economic thickness optimization results with provenance
        """
        self.logger.info("Calculating economic thickness per ASTM C680")

        # Get thermal conductivity at mean temperature
        mean_temp = (inputs.process_temp_c + inputs.ambient_temp_c) / 2
        k_value = self.thermal_conductivity.get_k_value(inputs.insulation_type, mean_temp)

        # Fuel properties
        heating_value = self.fuel_properties.heating_values[inputs.fuel_type]
        fuel_cost = inputs.fuel_cost_per_unit or self.fuel_properties.unit_costs[inputs.fuel_type]

        # Discount rate for annualization
        discount_rate = inputs.discount_rate_percent / 100
        n = inputs.analysis_period_years
        crf = (discount_rate * (1 + discount_rate) ** n) / ((1 + discount_rate) ** n - 1)

        # Analyze costs at different thicknesses
        thickness_analysis = []
        min_total_cost = float('inf')
        economic_thickness = 0

        # Temperature difference
        delta_t = inputs.process_temp_c - inputs.ambient_temp_c

        # Per linear meter analysis (for pipes) or per m2 (flat surfaces)
        is_pipe = inputs.equipment_type == EquipmentType.PIPE and inputs.pipe_diameter_mm

        for thickness_mm in range(25, 351, 5):  # 25mm to 350mm in 5mm steps
            thickness_m = thickness_mm / 1000

            # Insulation cost (material + installation)
            if is_pipe:
                # Cylindrical calculation - volume per linear meter
                r_inner = inputs.pipe_diameter_mm / 2000  # m
                r_outer = r_inner + thickness_m
                volume_per_m = math.pi * (r_outer ** 2 - r_inner ** 2)
                surface_area_per_m = 2 * math.pi * r_outer  # Outer surface
            else:
                # Flat surface - per m2
                volume_per_m = thickness_m  # m3 per m2
                surface_area_per_m = 1.0  # m2 per m2

            material_cost = self.material_costs.get_cost(inputs.insulation_type, 0.5)
            installed_cost = volume_per_m * material_cost * 1.5  # 50% installation factor
            annualized_insulation_cost = installed_cost * crf

            # Heat loss calculation
            if is_pipe:
                # Cylindrical heat loss: Q = 2*pi*k*L*dT / ln(r_outer/r_inner)
                r_inner = inputs.pipe_diameter_mm / 2000
                r_outer = r_inner + thickness_m
                heat_loss_per_m = (2 * math.pi * k_value * delta_t) / math.log(r_outer / r_inner)
            else:
                # Flat surface: Q = k*A*dT/t
                heat_loss_per_m = (k_value * delta_t) / thickness_m * 1000  # W/m2

            # Annual energy cost
            annual_hours = inputs.operating_hours_per_year
            annual_energy_loss_mj = heat_loss_per_m * annual_hours * 3.6 / 1000  # MJ per m or m2
            annual_fuel_consumption = annual_energy_loss_mj / (heating_value * 0.85)  # 85% efficiency
            annual_energy_cost = annual_fuel_consumption * fuel_cost

            # Total annual cost
            total_annual_cost = annualized_insulation_cost + annual_energy_cost

            thickness_analysis.append({
                "thickness_mm": thickness_mm,
                "installed_cost": round(installed_cost, 2),
                "annualized_insulation_cost": round(annualized_insulation_cost, 2),
                "heat_loss_w": round(heat_loss_per_m, 2),
                "annual_energy_cost": round(annual_energy_cost, 2),
                "total_annual_cost": round(total_annual_cost, 2),
            })

            if total_annual_cost < min_total_cost:
                min_total_cost = total_annual_cost
                economic_thickness = thickness_mm

        # Find recommended standard thickness
        recommended_thickness = min(
            t for t in self.STANDARD_THICKNESSES if t >= economic_thickness
        )

        # Get costs at economic thickness
        economic_data = next(
            d for d in thickness_analysis if d["thickness_mm"] == economic_thickness
        )

        # Marginal savings per additional mm (at economic thickness)
        if economic_thickness > 25:
            prev_data = next(
                d for d in thickness_analysis if d["thickness_mm"] == economic_thickness - 5
            )
            marginal_savings = (prev_data["total_annual_cost"] - economic_data["total_annual_cost"]) / 5
        else:
            marginal_savings = 0

        # Diminishing returns threshold (where marginal savings < $0.10/mm)
        diminishing_thickness = economic_thickness
        for i, data in enumerate(thickness_analysis):
            if i > 0:
                prev = thickness_analysis[i - 1]
                marginal = (prev["total_annual_cost"] - data["total_annual_cost"]) / 5
                if marginal < 0.10:
                    diminishing_thickness = data["thickness_mm"]
                    break

        # Upgrade analysis (if current thickness provided)
        upgrade_recommended = False
        upgrade_roi = None

        if current_thickness_mm and current_thickness_mm < economic_thickness:
            current_data = next(
                (d for d in thickness_analysis if d["thickness_mm"] >= current_thickness_mm),
                thickness_analysis[0]
            )
            annual_savings = current_data["total_annual_cost"] - economic_data["total_annual_cost"]
            upgrade_cost = economic_data["installed_cost"] - current_data["installed_cost"]

            if upgrade_cost > 0:
                upgrade_roi = (annual_savings / upgrade_cost) * 100
                upgrade_recommended = upgrade_roi > 10  # Recommend if >10% ROI

        # Surface temperature at economic thickness
        if is_pipe:
            r_inner = inputs.pipe_diameter_mm / 2000
            r_outer = r_inner + (economic_thickness / 1000)
            # Simplified surface temperature calculation
            h_surface = 10  # W/m2K ambient convection
            q = economic_data["heat_loss_w"]
            surface_temp = inputs.ambient_temp_c + q / (h_surface * 2 * math.pi * r_outer)
        else:
            h_surface = 10
            q = economic_data["heat_loss_w"]
            surface_temp = inputs.ambient_temp_c + q / h_surface

        # Calculate provenance hash
        calc_hash = self._calculate_hash({
            "inputs": inputs.dict(),
            "economic_thickness": economic_thickness,
            "min_total_cost": min_total_cost,
        })

        return EconomicThicknessResult(
            economic_thickness_mm=economic_thickness,
            recommended_thickness_mm=recommended_thickness,
            installed_cost_at_economic=economic_data["installed_cost"],
            annual_energy_cost_at_economic=economic_data["annual_energy_cost"],
            total_annual_cost_at_economic=economic_data["total_annual_cost"],
            thickness_analysis=thickness_analysis[:20],  # First 20 for brevity
            marginal_savings_per_mm=round(marginal_savings, 4),
            diminishing_returns_thickness_mm=diminishing_thickness,
            current_thickness_mm=current_thickness_mm,
            upgrade_recommended=upgrade_recommended,
            upgrade_roi_percent=round(upgrade_roi, 2) if upgrade_roi else None,
            surface_temp_at_economic_c=round(surface_temp, 1),
            calculation_hash=calc_hash,
        )

    # =========================================================================
    # 7. BUDGET IMPACT ANALYSIS
    # =========================================================================

    def analyze_budget_impact(
        self,
        repair_cost: RepairCostResult,
        energy_savings: EnergySavingsResult,
        financial_params: FinancialParameters
    ) -> BudgetImpactResult:
        """
        Analyze budget impact including CapEx, OpEx, and cash flow projections.

        Args:
            repair_cost: Repair cost result
            energy_savings: Energy savings result
            financial_params: Financial parameters

        Returns:
            Budget impact analysis results with provenance
        """
        self.logger.info("Analyzing budget impact")

        total_capex = repair_cost.total_repair_cost
        annual_opex_reduction = energy_savings.first_year_savings

        # CapEx breakdown
        capex_by_category = {
            "materials": repair_cost.total_material_cost,
            "labor": repair_cost.total_labor_cost,
            "access_equipment": repair_cost.total_access_cost,
            "permits_safety": repair_cost.total_permit_safety_cost,
            "contingency": repair_cost.contingency,
        }

        # OpEx impact by year (with escalation)
        escalation_rate = 0.03
        opex_impact_by_year = []
        for year in range(1, financial_params.analysis_period_years + 1):
            escalated_savings = annual_opex_reduction * ((1 + escalation_rate) ** (year - 1))
            opex_impact_by_year.append(round(escalated_savings, 2))

        # Monthly cash flow (first year)
        monthly_savings = annual_opex_reduction / 12
        monthly_cash_flow = []

        # Assume CapEx spent in first month
        for month in range(1, 13):
            if month == 1:
                net_cf = -total_capex + monthly_savings
            else:
                net_cf = monthly_savings

            monthly_cash_flow.append({
                "month": month,
                "capex": -total_capex if month == 1 else 0,
                "opex_savings": round(monthly_savings, 2),
                "net_cash_flow": round(net_cf, 2),
                "cumulative": round(
                    sum(m["net_cash_flow"] for m in monthly_cash_flow) + net_cf
                    if monthly_cash_flow else net_cf, 2
                ),
            })

        # Annual cash flow projection
        annual_cash_flow = []
        cumulative = -total_capex

        for year in range(1, financial_params.analysis_period_years + 1):
            savings = opex_impact_by_year[year - 1]
            maintenance = total_capex * 0.02  # 2% maintenance
            net_cf = savings - maintenance if year > 1 else savings - maintenance - total_capex
            cumulative += (savings - maintenance) if year > 1 else net_cf + total_capex

            annual_cash_flow.append({
                "year": year,
                "capex": -total_capex if year == 1 else 0,
                "opex_savings": round(savings, 2),
                "maintenance": round(-maintenance, 2),
                "net_cash_flow": round(net_cf if year > 1 else -total_capex + savings - maintenance, 2),
                "cumulative": round(cumulative, 2),
            })

        # Financing analysis
        financing_required = total_capex > 50000  # Threshold for financing
        monthly_payment = None
        total_interest = None

        if financial_params.financing_rate_percent and financial_params.financing_term_years:
            # Calculate loan payment
            r = financial_params.financing_rate_percent / 100 / 12
            n = financial_params.financing_term_years * 12
            if r > 0:
                monthly_payment = total_capex * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
                total_interest = monthly_payment * n - total_capex

        # Year 1 net cash impact
        year_1_net = annual_cash_flow[0]["net_cash_flow"]

        # Break-even year
        break_even_year = financial_params.analysis_period_years
        for acf in annual_cash_flow:
            if acf["cumulative"] >= 0:
                break_even_year = acf["year"]
                break

        # Cumulative benefit at Year 5
        year_5_cumulative = annual_cash_flow[min(4, len(annual_cash_flow) - 1)]["cumulative"]

        # Recommended funding source
        if total_capex < 25000:
            funding_source = "Operating Budget - Minor Capital"
        elif total_capex < 100000:
            funding_source = "Capital Budget - Energy Efficiency"
        elif total_capex < 500000:
            funding_source = "Energy Performance Contract"
        else:
            funding_source = "Project Financing or Green Bond"

        # Calculate provenance hash
        calc_hash = self._calculate_hash({
            "total_capex": total_capex,
            "annual_opex_reduction": annual_opex_reduction,
            "break_even_year": break_even_year,
        })

        return BudgetImpactResult(
            total_capex=round(total_capex, 2),
            capex_by_category=capex_by_category,
            annual_opex_reduction=round(annual_opex_reduction, 2),
            opex_impact_by_year=opex_impact_by_year,
            monthly_cash_flow=monthly_cash_flow,
            annual_cash_flow=annual_cash_flow,
            financing_required=financing_required,
            monthly_payment=round(monthly_payment, 2) if monthly_payment else None,
            total_interest_cost=round(total_interest, 2) if total_interest else None,
            year_1_net_cash_impact=round(year_1_net, 2),
            break_even_year=break_even_year,
            cumulative_benefit_year_5=round(year_5_cumulative, 2),
            recommended_funding_source=funding_source,
            calculation_hash=calc_hash,
        )

    # =========================================================================
    # 8. CARBON ECONOMICS
    # =========================================================================

    def calculate_carbon_economics(
        self,
        energy_savings: EnergySavingsResult,
        repair_cost: RepairCostResult,
        carbon_inputs: CarbonEconomicsInput,
        financial_params: FinancialParameters,
        fuel_type: FuelType
    ) -> CarbonEconomicsResult:
        """
        Calculate carbon economics including credits, taxes, and ESG benefits.

        Args:
            energy_savings: Energy savings result
            repair_cost: Repair cost result
            carbon_inputs: Carbon economics inputs
            financial_params: Financial parameters
            fuel_type: Fuel type for emissions calculation

        Returns:
            Carbon economics results with provenance
        """
        self.logger.info("Calculating carbon economics")

        # CO2 reduction from fuel savings
        co2_factor = self.fuel_properties.co2_factors[fuel_type]
        annual_fuel_reduction = energy_savings.annual_fuel_reduction_units

        annual_co2_reduction = annual_fuel_reduction * co2_factor / 1000  # tonnes
        lifetime_co2_reduction = annual_co2_reduction * financial_params.analysis_period_years

        # Carbon price
        carbon_price = carbon_inputs.carbon_price_per_tonne or self.CARBON_PRICES.get(
            carbon_inputs.pricing_scheme, 25.0
        )

        # Annual carbon value
        carbon_credit_annual = annual_co2_reduction * carbon_price

        # Lifetime and NPV of carbon value
        discount_rate = financial_params.discount_rate_percent / 100
        escalation_rate = carbon_inputs.carbon_price_escalation_percent / 100

        carbon_credit_lifetime = Decimal("0")
        carbon_credit_npv = Decimal("0")

        for year in range(1, financial_params.analysis_period_years + 1):
            escalated_price = Decimal(str(carbon_price)) * Decimal(str((1 + escalation_rate) ** (year - 1)))
            annual_value = Decimal(str(annual_co2_reduction)) * escalated_price
            carbon_credit_lifetime += annual_value

            df = Decimal(str((1 + discount_rate) ** year))
            carbon_credit_npv += annual_value / df

        # Carbon tax avoided (same calculation, different framing)
        carbon_tax_annual = carbon_credit_annual
        carbon_tax_lifetime = float(carbon_credit_lifetime)

        # Marginal abatement cost
        if lifetime_co2_reduction > 0:
            mac = repair_cost.total_repair_cost / lifetime_co2_reduction
        else:
            mac = 0

        # ESG benefits
        esg_benefit = carbon_inputs.esg_reporting_value

        # Sustainability score improvement (simplified metric)
        # Based on CO2 reduction relative to typical industrial baseline
        baseline_tonnes_per_year = 100  # Assumed baseline
        sustainability_improvement = (annual_co2_reduction / baseline_tonnes_per_year) * 10

        # Total carbon value
        total_carbon_annual = carbon_credit_annual + esg_benefit
        total_carbon_npv = float(carbon_credit_npv) + (esg_benefit * financial_params.analysis_period_years)

        # Calculate provenance hash
        calc_hash = self._calculate_hash({
            "annual_co2_reduction": annual_co2_reduction,
            "carbon_price": carbon_price,
            "npv": str(carbon_credit_npv),
        })

        return CarbonEconomicsResult(
            annual_co2_reduction_tonnes=round(annual_co2_reduction, 3),
            lifetime_co2_reduction_tonnes=round(lifetime_co2_reduction, 3),
            carbon_credit_value_annual=round(carbon_credit_annual, 2),
            carbon_credit_value_lifetime=float(carbon_credit_lifetime.quantize(Decimal("0.01"))),
            carbon_credit_value_npv=float(carbon_credit_npv.quantize(Decimal("0.01"))),
            carbon_tax_avoided_annual=round(carbon_tax_annual, 2),
            carbon_tax_avoided_lifetime=round(carbon_tax_lifetime, 2),
            marginal_abatement_cost=round(mac, 2),
            esg_reporting_benefit=round(esg_benefit, 2),
            sustainability_score_improvement=round(sustainability_improvement, 2),
            total_carbon_value_annual=round(total_carbon_annual, 2),
            total_carbon_value_npv=round(total_carbon_npv, 2),
            calculation_hash=calc_hash,
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        hash_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def run_comprehensive_analysis(
        self,
        repair_inputs: RepairCostInput,
        energy_inputs: EnergySavingsInput,
        financial_params: FinancialParameters,
        lifecycle_inputs: LifecycleInput,
        carbon_inputs: CarbonEconomicsInput
    ) -> Dict[str, Any]:
        """
        Run all economic analyses and return comprehensive results.

        Args:
            repair_inputs: Repair cost inputs
            energy_inputs: Energy savings inputs
            financial_params: Financial parameters
            lifecycle_inputs: Lifecycle inputs
            carbon_inputs: Carbon economics inputs

        Returns:
            Dictionary containing all analysis results
        """
        self.logger.info("Running comprehensive economic analysis")

        # 1. Repair cost estimation
        repair_cost = self.estimate_repair_cost(repair_inputs)

        # 2. Energy savings calculation
        energy_savings = self.calculate_energy_savings(energy_inputs, financial_params)

        # 3. Payback analysis
        payback = self.calculate_payback_period(
            repair_cost, energy_savings, financial_params, lifecycle_inputs
        )

        # 4. NPV/IRR calculation
        npv_irr = self.calculate_npv_irr(
            repair_cost, energy_savings, financial_params, repair_inputs.repair_area_m2
        )

        # 5. Life cycle cost analysis
        lifecycle = self.perform_lifecycle_cost_analysis(
            repair_cost, energy_savings, financial_params,
            lifecycle_inputs, repair_inputs.repair_area_m2
        )

        # 6. Budget impact analysis
        budget = self.analyze_budget_impact(repair_cost, energy_savings, financial_params)

        # 7. Carbon economics
        carbon = self.calculate_carbon_economics(
            energy_savings, repair_cost, carbon_inputs,
            financial_params, energy_inputs.fuel_type
        )

        # Generate master provenance hash
        master_hash = self._calculate_hash({
            "repair_hash": repair_cost.calculation_hash,
            "energy_hash": energy_savings.calculation_hash,
            "payback_hash": payback.calculation_hash,
            "npv_hash": npv_irr.calculation_hash,
            "lifecycle_hash": lifecycle.calculation_hash,
            "budget_hash": budget.calculation_hash,
            "carbon_hash": carbon.calculation_hash,
        })

        return {
            "repair_cost": repair_cost,
            "energy_savings": energy_savings,
            "payback": payback,
            "npv_irr": npv_irr,
            "lifecycle": lifecycle,
            "budget_impact": budget,
            "carbon_economics": carbon,
            "master_provenance_hash": master_hash,
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize calculator
    calculator = EconomicCalculator()

    # Example: Industrial pipe insulation repair analysis
    repair_inputs = RepairCostInput(
        equipment_type=EquipmentType.PIPE,
        insulation_type=InsulationType.MINERAL_WOOL,
        repair_area_m2=50.0,
        thickness_mm=75.0,
        complexity=RepairComplexity.MODERATE,
        access_requirement=AccessRequirement.SCAFFOLDING,
        region="us_gulf",
        estimated_duration_days=3.0,
        requires_permit=True,
        requires_hot_work=False,
        asbestos_abatement=False,
        quality_factor=0.5,
        contingency_percent=15.0,
    )

    energy_inputs = EnergySavingsInput(
        current_heat_loss_w_per_m2=500.0,
        surface_area_m2=50.0,
        process_temp_c=250.0,
        ambient_temp_c=25.0,
        operating_hours_per_year=8000,
        target_heat_loss_w_per_m2=50.0,
        fuel_type=FuelType.NATURAL_GAS,
        boiler_efficiency=0.85,
        fuel_price_escalation_percent=3.0,
    )

    financial_params = FinancialParameters(
        discount_rate_percent=10.0,
        analysis_period_years=20,
        inflation_rate_percent=2.5,
        tax_rate_percent=25.0,
        depreciation_method=DepreciationMethod.MACRS_7_YEAR,
        residual_value_percent=5.0,
    )

    lifecycle_inputs = LifecycleInput(
        equipment_life_years=25,
        insulation_life_years=20,
        annual_inspection_cost=500.0,
        maintenance_percent_of_capital=2.0,
        replacement_cost_escalation_percent=2.5,
        decommissioning_cost_percent=5.0,
    )

    carbon_inputs = CarbonEconomicsInput(
        pricing_scheme=CarbonPricingScheme.VOLUNTARY,
        carbon_price_per_tonne=30.0,
        carbon_price_escalation_percent=5.0,
        include_scope_2=True,
        esg_reporting_value=1000.0,
    )

    # Run comprehensive analysis
    results = calculator.run_comprehensive_analysis(
        repair_inputs,
        energy_inputs,
        financial_params,
        lifecycle_inputs,
        carbon_inputs,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("ECONOMIC IMPACT ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\n1. REPAIR COST:")
    print(f"   Total Repair Cost: ${results['repair_cost'].total_repair_cost:,.2f}")
    print(f"   Cost per m2: ${results['repair_cost'].cost_per_m2:,.2f}")

    print(f"\n2. ENERGY SAVINGS:")
    print(f"   Heat Loss Reduction: {results['energy_savings'].heat_loss_reduction_percent:.1f}%")
    print(f"   First Year Savings: ${results['energy_savings'].first_year_savings:,.2f}")
    print(f"   Lifetime Savings (NPV): ${results['energy_savings'].lifetime_savings_present_value:,.2f}")

    print(f"\n3. PAYBACK ANALYSIS:")
    print(f"   Simple Payback: {results['payback'].simple_payback_years:.1f} years")
    print(f"   Discounted Payback: {results['payback'].discounted_payback_years:.1f} years")

    print(f"\n4. NPV/IRR:")
    print(f"   NPV: ${results['npv_irr'].npv_usd:,.2f}")
    print(f"   IRR: {results['npv_irr'].irr_percent:.1f}%")
    print(f"   Recommendation: {results['npv_irr'].recommendation}")

    print(f"\n5. LIFE CYCLE COST:")
    print(f"   Total Cost of Ownership: ${results['lifecycle'].total_cost_of_ownership:,.2f}")
    print(f"   Savings vs Do-Nothing: ${results['lifecycle'].savings_vs_do_nothing:,.2f}")

    print(f"\n6. CARBON ECONOMICS:")
    print(f"   Annual CO2 Reduction: {results['carbon_economics'].annual_co2_reduction_tonnes:.1f} tonnes")
    print(f"   Carbon Value (NPV): ${results['carbon_economics'].carbon_credit_value_npv:,.2f}")
    print(f"   Marginal Abatement Cost: ${results['carbon_economics'].marginal_abatement_cost:.2f}/tonne")

    print(f"\n7. BUDGET IMPACT:")
    print(f"   Total CapEx: ${results['budget_impact'].total_capex:,.2f}")
    print(f"   Break-even Year: {results['budget_impact'].break_even_year}")
    print(f"   Recommended Funding: {results['budget_impact'].recommended_funding_source}")

    print(f"\n" + "=" * 60)
    print(f"Master Provenance Hash: {results['master_provenance_hash'][:32]}...")
    print(f"Analysis Timestamp: {results['analysis_timestamp']}")
    print("=" * 60)
