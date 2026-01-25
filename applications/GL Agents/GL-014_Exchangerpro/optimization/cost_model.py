# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro - Cleaning Cost Model

Comprehensive 5-component cost model for heat exchanger cleaning optimization:
1. EnergyLossCost - Utility cost due to reduced UA/effectiveness
2. ProductionLossCost - Off-spec product or throughput constraints
3. CleaningCost - Labor, chemicals, contractors
4. DowntimeCost - Lost margin during offline cleaning
5. RiskPenalty - Probability-weighted constraint violation costs

Zero-Hallucination Principle:
    All cost calculations use explicit deterministic formulas.
    Energy losses are computed from thermal performance degradation.
    No LLM-generated cost estimates or optimization decisions.

Author: GreenLang AI Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math

from pydantic import BaseModel, Field, validator, root_validator

logger = logging.getLogger(__name__)


class CleaningMethodType(str, Enum):
    """Types of cleaning methods available."""
    CHEMICAL_ONLINE = "chemical_online"  # Online chemical cleaning
    CHEMICAL_OFFLINE = "chemical_offline"  # Offline chemical cleaning
    MECHANICAL_HYDROBLAST = "mechanical_hydroblast"  # High-pressure water
    MECHANICAL_PIGGING = "mechanical_pigging"  # Pig cleaning
    MECHANICAL_BRUSH = "mechanical_brush"  # Mechanical brushing
    COMBINED = "combined"  # Chemical + mechanical


class CostCategory(str, Enum):
    """Categories of costs in the model."""
    ENERGY_LOSS = "energy_loss"
    PRODUCTION_LOSS = "production_loss"
    CLEANING = "cleaning"
    DOWNTIME = "downtime"
    RISK_PENALTY = "risk_penalty"


class ConstraintType(str, Enum):
    """Types of operational constraints."""
    DELTA_P_MAX = "delta_p_max"  # Maximum pressure drop
    T_OUTLET_MAX = "t_outlet_max"  # Maximum outlet temperature
    T_OUTLET_MIN = "t_outlet_min"  # Minimum outlet temperature
    UA_MIN = "ua_min"  # Minimum heat transfer coefficient
    EFFECTIVENESS_MIN = "effectiveness_min"  # Minimum effectiveness


@dataclass
class CostModelConfig:
    """
    Configuration for the cleaning cost model.

    All costs are in USD unless otherwise specified.
    """
    # Energy costs
    fuel_cost_usd_per_gj: float = 4.0  # Natural gas ~$4/GJ
    electricity_cost_usd_per_kwh: float = 0.08
    steam_cost_usd_per_tonne: float = 25.0
    cooling_water_cost_usd_per_m3: float = 0.05

    # Production parameters
    production_margin_usd_per_hour: float = 5000.0  # Gross margin per hour
    throughput_loss_fraction_per_ua_pct: float = 0.005  # 0.5% throughput loss per 1% UA loss
    off_spec_cost_usd_per_tonne: float = 50.0  # Cost of off-spec product

    # Cleaning costs
    chemical_cleaning_cost_usd: float = 15000.0
    mechanical_cleaning_cost_usd: float = 25000.0
    combined_cleaning_cost_usd: float = 35000.0
    labor_rate_usd_per_hour: float = 75.0
    contractor_markup: float = 1.5

    # Downtime parameters
    typical_cleaning_duration_hours: float = 24.0
    mechanical_cleaning_duration_hours: float = 48.0
    startup_loss_hours: float = 4.0  # Transition losses

    # Risk parameters
    delta_p_violation_cost_usd: float = 100000.0  # Pump trip/damage
    temperature_violation_cost_usd: float = 50000.0  # Off-spec product
    unplanned_shutdown_cost_usd: float = 250000.0  # Emergency shutdown

    # Operating hours
    operating_hours_per_year: float = 8000.0

    # Discount rate for NPV calculations
    discount_rate: float = 0.10  # 10% annual


class EnergyLossCost(BaseModel):
    """
    Energy loss cost due to reduced UA/effectiveness.

    Calculates additional utility consumption when heat exchanger
    performance degrades due to fouling.
    """
    exchanger_id: str = Field(..., description="Heat exchanger identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Performance metrics
    current_ua_kw_k: float = Field(..., gt=0, description="Current UA value (kW/K)")
    clean_ua_kw_k: float = Field(..., gt=0, description="Clean (design) UA value (kW/K)")
    ua_degradation_fraction: float = Field(..., ge=0, le=1, description="UA loss fraction")

    # Energy loss calculation
    heat_duty_kw: float = Field(..., ge=0, description="Current heat duty (kW)")
    additional_utility_kw: float = Field(..., ge=0, description="Extra utility needed (kW)")
    utility_type: str = Field("steam", description="Type of utility (steam, fuel, electricity)")
    utility_unit_cost: float = Field(..., gt=0, description="Cost per unit of utility")

    # Costs
    hourly_energy_loss_usd: float = Field(..., ge=0, description="Hourly energy loss cost")
    daily_energy_loss_usd: float = Field(..., ge=0, description="Daily energy loss cost")
    annual_energy_loss_usd: float = Field(..., ge=0, description="Annualized energy loss")

    # Provenance
    calculation_method: str = Field("ua_degradation", description="Calculation method used")
    provenance_hash: str = Field("", description="SHA-256 hash for audit")

    @validator('ua_degradation_fraction', pre=True, always=True)
    def compute_degradation(cls, v, values):
        """Compute UA degradation if not provided."""
        if v is None or v == 0:
            current = values.get('current_ua_kw_k', 0)
            clean = values.get('clean_ua_kw_k', 1)
            if clean > 0:
                return 1.0 - (current / clean)
        return v


class ProductionLossCost(BaseModel):
    """
    Production loss cost due to fouling-induced constraints.

    Includes throughput reduction and off-spec product costs.
    """
    exchanger_id: str = Field(..., description="Heat exchanger identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Performance metrics
    current_effectiveness: float = Field(..., ge=0, le=1)
    design_effectiveness: float = Field(..., ge=0, le=1)
    effectiveness_degradation: float = Field(..., ge=0, le=1)

    # Throughput impact
    design_throughput_tph: float = Field(..., ge=0, description="Design throughput (t/h)")
    current_throughput_tph: float = Field(..., ge=0, description="Current throughput (t/h)")
    throughput_loss_tph: float = Field(..., ge=0, description="Throughput loss (t/h)")
    throughput_loss_fraction: float = Field(..., ge=0, le=1)

    # Off-spec product
    off_spec_fraction: float = Field(0.0, ge=0, le=1, description="Fraction off-spec")
    off_spec_cost_rate_usd_h: float = Field(0.0, ge=0, description="Hourly off-spec cost")

    # Costs
    hourly_throughput_loss_usd: float = Field(..., ge=0)
    daily_production_loss_usd: float = Field(..., ge=0)
    annual_production_loss_usd: float = Field(..., ge=0)

    # Provenance
    calculation_method: str = Field("effectiveness_linear")
    provenance_hash: str = Field("")


class CleaningCost(BaseModel):
    """
    Direct cost of cleaning intervention.

    Includes labor, chemicals, equipment, and contractor costs.
    """
    exchanger_id: str = Field(..., description="Heat exchanger identifier")
    cleaning_method: CleaningMethodType = Field(...)

    # Labor costs
    labor_hours: float = Field(..., ge=0, description="Total labor hours")
    labor_rate_usd_h: float = Field(..., gt=0, description="Labor rate ($/h)")
    labor_cost_usd: float = Field(..., ge=0, description="Total labor cost")

    # Materials costs
    chemical_cost_usd: float = Field(0.0, ge=0, description="Chemical costs")
    equipment_cost_usd: float = Field(0.0, ge=0, description="Equipment rental/usage")

    # Contractor costs
    contractor_cost_usd: float = Field(0.0, ge=0, description="External contractor cost")
    contractor_markup: float = Field(1.0, ge=1.0, description="Contractor markup factor")

    # Disposal and environmental
    waste_disposal_cost_usd: float = Field(0.0, ge=0, description="Waste disposal cost")
    environmental_fees_usd: float = Field(0.0, ge=0, description="Environmental compliance")

    # Total
    total_cleaning_cost_usd: float = Field(..., ge=0, description="Total cleaning cost")

    # Effectiveness
    expected_ua_recovery: float = Field(..., ge=0, le=1, description="Expected UA recovery")
    expected_cleaning_duration_hours: float = Field(..., gt=0)

    # Provenance
    cost_estimate_source: str = Field("historical_average")
    provenance_hash: str = Field("")

    @root_validator
    def compute_total(cls, values):
        """Compute total cleaning cost."""
        labor = values.get('labor_cost_usd', 0)
        chemical = values.get('chemical_cost_usd', 0)
        equipment = values.get('equipment_cost_usd', 0)
        contractor = values.get('contractor_cost_usd', 0)
        markup = values.get('contractor_markup', 1.0)
        disposal = values.get('waste_disposal_cost_usd', 0)
        environmental = values.get('environmental_fees_usd', 0)

        total = labor + chemical + equipment + (contractor * markup) + disposal + environmental
        values['total_cleaning_cost_usd'] = total
        return values


class DowntimeCost(BaseModel):
    """
    Cost of lost production during cleaning downtime.

    Includes direct margin loss and startup transition costs.
    """
    exchanger_id: str = Field(..., description="Heat exchanger identifier")

    # Downtime parameters
    cleaning_duration_hours: float = Field(..., gt=0, description="Cleaning duration (h)")
    startup_duration_hours: float = Field(4.0, ge=0, description="Startup duration (h)")
    total_outage_hours: float = Field(..., gt=0, description="Total outage duration")

    # Production impact
    production_rate_tph: float = Field(..., ge=0, description="Normal production rate")
    product_margin_usd_per_tonne: float = Field(..., ge=0, description="Margin per tonne")
    hourly_margin_usd: float = Field(..., ge=0, description="Hourly gross margin")

    # Costs
    cleaning_period_loss_usd: float = Field(..., ge=0)
    startup_loss_usd: float = Field(..., ge=0)
    total_downtime_cost_usd: float = Field(..., ge=0)

    # Partial operation
    partial_operation_fraction: float = Field(
        0.0, ge=0, le=1, description="Fraction of capacity during cleaning"
    )
    partial_operation_savings_usd: float = Field(0.0, ge=0)

    # Provenance
    calculation_method: str = Field("margin_loss")
    provenance_hash: str = Field("")

    @root_validator
    def compute_totals(cls, values):
        """Compute downtime costs."""
        cleaning_hours = values.get('cleaning_duration_hours', 0)
        startup_hours = values.get('startup_duration_hours', 0)
        hourly_margin = values.get('hourly_margin_usd', 0)
        partial_fraction = values.get('partial_operation_fraction', 0)

        total_outage = cleaning_hours + startup_hours
        values['total_outage_hours'] = total_outage

        # Full loss during cleaning
        cleaning_loss = cleaning_hours * hourly_margin * (1 - partial_fraction)
        values['cleaning_period_loss_usd'] = cleaning_loss

        # Partial loss during startup
        startup_loss = startup_hours * hourly_margin * 0.5  # 50% efficiency during startup
        values['startup_loss_usd'] = startup_loss

        values['total_downtime_cost_usd'] = cleaning_loss + startup_loss
        values['partial_operation_savings_usd'] = cleaning_hours * hourly_margin * partial_fraction

        return values


class RiskPenalty(BaseModel):
    """
    Probability-weighted cost of constraint violations.

    Models the expected cost of operational limit exceedances.
    """
    exchanger_id: str = Field(..., description="Heat exchanger identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    evaluation_horizon_days: int = Field(30, ge=1, description="Risk evaluation period")

    # Delta-P constraint
    current_delta_p_kpa: float = Field(..., ge=0, description="Current pressure drop")
    delta_p_limit_kpa: float = Field(..., gt=0, description="Pressure drop limit")
    delta_p_margin_fraction: float = Field(..., description="Margin to limit (negative = violated)")
    delta_p_violation_probability: float = Field(..., ge=0, le=1)
    delta_p_violation_cost_usd: float = Field(..., ge=0, description="Cost if violated")
    delta_p_expected_cost_usd: float = Field(..., ge=0, description="Probability-weighted cost")

    # Temperature constraint
    current_t_outlet_c: float = Field(..., description="Current outlet temperature")
    t_outlet_limit_c: float = Field(..., description="Outlet temperature limit")
    t_violation_probability: float = Field(..., ge=0, le=1)
    t_violation_cost_usd: float = Field(..., ge=0)
    t_expected_cost_usd: float = Field(..., ge=0)

    # Unplanned shutdown risk
    shutdown_probability: float = Field(..., ge=0, le=1, description="Unplanned shutdown probability")
    shutdown_cost_usd: float = Field(..., ge=0, description="Unplanned shutdown cost")
    shutdown_expected_cost_usd: float = Field(..., ge=0)

    # Total risk penalty
    total_risk_penalty_usd: float = Field(..., ge=0, description="Total expected risk cost")

    # Provenance
    risk_model: str = Field("logistic_regression")
    provenance_hash: str = Field("")

    @root_validator
    def compute_expected_costs(cls, values):
        """Compute probability-weighted expected costs."""
        # Delta-P expected cost
        dp_prob = values.get('delta_p_violation_probability', 0)
        dp_cost = values.get('delta_p_violation_cost_usd', 0)
        values['delta_p_expected_cost_usd'] = dp_prob * dp_cost

        # Temperature expected cost
        t_prob = values.get('t_violation_probability', 0)
        t_cost = values.get('t_violation_cost_usd', 0)
        values['t_expected_cost_usd'] = t_prob * t_cost

        # Shutdown expected cost
        sd_prob = values.get('shutdown_probability', 0)
        sd_cost = values.get('shutdown_cost_usd', 0)
        values['shutdown_expected_cost_usd'] = sd_prob * sd_cost

        # Total
        values['total_risk_penalty_usd'] = (
            values['delta_p_expected_cost_usd'] +
            values['t_expected_cost_usd'] +
            values['shutdown_expected_cost_usd']
        )

        return values


class TotalCostBreakdown(BaseModel):
    """
    Complete cost breakdown for a cleaning decision.

    Aggregates all 5 cost components with full traceability.
    """
    exchanger_id: str = Field(..., description="Heat exchanger identifier")
    evaluation_date: datetime = Field(default_factory=datetime.utcnow)
    horizon_days: int = Field(..., ge=1, description="Planning horizon in days")

    # Individual cost components
    energy_loss_cost: EnergyLossCost = Field(..., description="Energy loss component")
    production_loss_cost: ProductionLossCost = Field(..., description="Production loss component")
    cleaning_cost: Optional[CleaningCost] = Field(None, description="Cleaning cost (if cleaning)")
    downtime_cost: Optional[DowntimeCost] = Field(None, description="Downtime cost (if cleaning)")
    risk_penalty: RiskPenalty = Field(..., description="Risk penalty component")

    # Aggregated costs (over horizon)
    total_energy_loss_usd: float = Field(..., ge=0)
    total_production_loss_usd: float = Field(..., ge=0)
    total_cleaning_cost_usd: float = Field(0.0, ge=0)
    total_downtime_cost_usd: float = Field(0.0, ge=0)
    total_risk_penalty_usd: float = Field(..., ge=0)

    # Grand total
    total_cost_usd: float = Field(..., ge=0, description="Total cost over horizon")
    daily_average_cost_usd: float = Field(..., ge=0, description="Daily average cost")

    # Cost shares
    cost_breakdown_pct: Dict[str, float] = Field(
        default_factory=dict, description="Percentage breakdown by category"
    )

    # Assumptions
    assumptions: Dict[str, Any] = Field(default_factory=dict, description="Key assumptions")

    # Provenance
    provenance_hash: str = Field("", description="SHA-256 hash for audit trail")
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = Field("1.0.0")

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for complete audit trail."""
        content = {
            "exchanger_id": self.exchanger_id,
            "evaluation_date": self.evaluation_date.isoformat(),
            "horizon_days": self.horizon_days,
            "total_cost_usd": self.total_cost_usd,
            "cost_breakdown": self.cost_breakdown_pct,
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


class CostCurve(BaseModel):
    """
    Cost trajectory over time for a given operating scenario.

    Used for visualizing cost evolution and finding optimal cleaning times.
    """
    exchanger_id: str = Field(...)
    start_date: datetime = Field(...)
    end_date: datetime = Field(...)

    # Time series data
    dates: List[datetime] = Field(..., description="Evaluation dates")
    cumulative_energy_loss_usd: List[float] = Field(..., description="Cumulative energy loss")
    cumulative_production_loss_usd: List[float] = Field(..., description="Cumulative production loss")
    cumulative_risk_penalty_usd: List[float] = Field(..., description="Cumulative risk penalty")
    total_cost_without_cleaning_usd: List[float] = Field(..., description="Total if no cleaning")

    # Optimal cleaning point
    optimal_cleaning_day: Optional[int] = Field(None, description="Optimal day to clean")
    optimal_total_cost_usd: Optional[float] = Field(None, description="Cost with optimal cleaning")

    # Provenance
    provenance_hash: str = Field("")
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)


class CostProjection(BaseModel):
    """
    Forward-looking cost projection with uncertainty bounds.
    """
    exchanger_id: str = Field(...)
    projection_start: datetime = Field(...)
    projection_end: datetime = Field(...)
    horizon_days: int = Field(..., ge=1)

    # Fouling trajectory inputs
    current_ua_kw_k: float = Field(..., gt=0)
    projected_ua_trajectory: List[float] = Field(..., description="Projected UA over time")
    fouling_rate_per_day: float = Field(..., description="Daily fouling rate")

    # Cost projections
    dates: List[datetime] = Field(...)
    daily_operating_cost_usd: List[float] = Field(...)
    cumulative_cost_usd: List[float] = Field(...)

    # Uncertainty bounds (95% CI)
    cumulative_cost_lower_usd: List[float] = Field(...)
    cumulative_cost_upper_usd: List[float] = Field(...)

    # Key milestones
    constraint_violation_date: Optional[datetime] = Field(None)
    recommended_cleaning_date: Optional[datetime] = Field(None)
    breakeven_days: Optional[int] = Field(None, description="Days until cleaning pays off")

    # Provenance
    projection_method: str = Field("linear_extrapolation")
    confidence_level: float = Field(0.95)
    provenance_hash: str = Field("")


class CleaningCostModel:
    """
    Comprehensive 5-component cost model for heat exchanger cleaning optimization.

    This class implements deterministic cost calculations for:
    1. Energy losses from reduced heat transfer
    2. Production losses from throughput/quality constraints
    3. Direct cleaning intervention costs
    4. Downtime costs during cleaning
    5. Risk penalties for constraint violations

    All calculations follow the zero-hallucination principle: costs are computed
    from explicit engineering formulas, not LLM estimates.

    Example:
        >>> config = CostModelConfig(fuel_cost_usd_per_gj=4.5)
        >>> model = CleaningCostModel(config)
        >>> energy_loss = model.calculate_energy_loss(
        ...     exchanger_id="HX-001",
        ...     current_ua=450.0,
        ...     clean_ua=500.0,
        ...     heat_duty=1000.0
        ... )
        >>> print(f"Daily energy loss: ${energy_loss.daily_energy_loss_usd:,.0f}")
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        config: Optional[CostModelConfig] = None,
        seed: int = 42,
    ) -> None:
        """
        Initialize the cleaning cost model.

        Args:
            config: Cost model configuration parameters
            seed: Random seed for reproducibility in probabilistic calculations
        """
        self.config = config or CostModelConfig()
        self.seed = seed
        self._rng_state = seed

        logger.info(
            f"CleaningCostModel initialized: fuel=${self.config.fuel_cost_usd_per_gj}/GJ, "
            f"margin=${self.config.production_margin_usd_per_hour}/h"
        )

    def calculate_energy_loss(
        self,
        exchanger_id: str,
        current_ua_kw_k: float,
        clean_ua_kw_k: float,
        heat_duty_kw: float,
        utility_type: str = "steam",
    ) -> EnergyLossCost:
        """
        Calculate energy loss cost due to UA degradation.

        The energy penalty is computed as the additional utility consumption
        required to maintain the same heat duty with reduced UA.

        Formula:
            Q = UA * LMTD
            With reduced UA, either Q decreases or LMTD must increase,
            requiring additional utility consumption.

        Args:
            exchanger_id: Heat exchanger identifier
            current_ua_kw_k: Current overall heat transfer coefficient (kW/K)
            clean_ua_kw_k: Clean/design UA value (kW/K)
            heat_duty_kw: Current heat duty (kW)
            utility_type: Type of utility used

        Returns:
            EnergyLossCost with calculated costs
        """
        # Calculate UA degradation
        ua_degradation = 1.0 - (current_ua_kw_k / clean_ua_kw_k) if clean_ua_kw_k > 0 else 0

        # Additional utility needed to compensate for UA loss
        # Simplified model: additional utility proportional to UA degradation
        # More rigorous: solve LMTD equations with temperature constraints
        additional_utility_kw = heat_duty_kw * ua_degradation * 0.3  # 30% efficiency factor

        # Get utility cost
        if utility_type == "steam":
            # Steam at ~2.7 GJ/tonne, config has $/tonne
            utility_unit_cost = self.config.steam_cost_usd_per_tonne / 2.7  # $/GJ
            hourly_cost = additional_utility_kw * 0.0036 * utility_unit_cost  # kW to GJ/h
        elif utility_type == "fuel":
            utility_unit_cost = self.config.fuel_cost_usd_per_gj
            hourly_cost = additional_utility_kw * 0.0036 * utility_unit_cost
        else:  # electricity
            utility_unit_cost = self.config.electricity_cost_usd_per_kwh
            hourly_cost = additional_utility_kw * utility_unit_cost

        daily_cost = hourly_cost * 24
        annual_cost = hourly_cost * self.config.operating_hours_per_year

        result = EnergyLossCost(
            exchanger_id=exchanger_id,
            current_ua_kw_k=current_ua_kw_k,
            clean_ua_kw_k=clean_ua_kw_k,
            ua_degradation_fraction=ua_degradation,
            heat_duty_kw=heat_duty_kw,
            additional_utility_kw=additional_utility_kw,
            utility_type=utility_type,
            utility_unit_cost=utility_unit_cost,
            hourly_energy_loss_usd=round(hourly_cost, 2),
            daily_energy_loss_usd=round(daily_cost, 2),
            annual_energy_loss_usd=round(annual_cost, 2),
            calculation_method="ua_degradation_linear",
        )

        # Compute provenance hash
        result.provenance_hash = self._compute_hash({
            "exchanger_id": exchanger_id,
            "current_ua": current_ua_kw_k,
            "clean_ua": clean_ua_kw_k,
            "daily_cost": daily_cost,
        })

        return result

    def calculate_production_loss(
        self,
        exchanger_id: str,
        current_effectiveness: float,
        design_effectiveness: float,
        design_throughput_tph: float,
        product_margin_usd_per_tonne: float,
    ) -> ProductionLossCost:
        """
        Calculate production loss cost due to effectiveness degradation.

        Throughput reduction is modeled as proportional to effectiveness loss,
        with a configurable sensitivity factor.

        Args:
            exchanger_id: Heat exchanger identifier
            current_effectiveness: Current thermal effectiveness (0-1)
            design_effectiveness: Design thermal effectiveness (0-1)
            design_throughput_tph: Design throughput in tonnes per hour
            product_margin_usd_per_tonne: Product margin in $/tonne

        Returns:
            ProductionLossCost with calculated costs
        """
        # Calculate effectiveness degradation
        effectiveness_degradation = 1.0 - (current_effectiveness / design_effectiveness)
        effectiveness_degradation = max(0, min(1, effectiveness_degradation))

        # Throughput loss proportional to effectiveness degradation
        throughput_loss_fraction = (
            effectiveness_degradation *
            self.config.throughput_loss_fraction_per_ua_pct * 100
        )
        throughput_loss_fraction = min(throughput_loss_fraction, 0.30)  # Cap at 30%

        current_throughput = design_throughput_tph * (1 - throughput_loss_fraction)
        throughput_loss = design_throughput_tph - current_throughput

        # Calculate costs
        hourly_loss = throughput_loss * product_margin_usd_per_tonne
        daily_loss = hourly_loss * 24
        annual_loss = hourly_loss * self.config.operating_hours_per_year

        result = ProductionLossCost(
            exchanger_id=exchanger_id,
            current_effectiveness=current_effectiveness,
            design_effectiveness=design_effectiveness,
            effectiveness_degradation=effectiveness_degradation,
            design_throughput_tph=design_throughput_tph,
            current_throughput_tph=current_throughput,
            throughput_loss_tph=throughput_loss,
            throughput_loss_fraction=throughput_loss_fraction,
            off_spec_fraction=0.0,
            off_spec_cost_rate_usd_h=0.0,
            hourly_throughput_loss_usd=round(hourly_loss, 2),
            daily_production_loss_usd=round(daily_loss, 2),
            annual_production_loss_usd=round(annual_loss, 2),
        )

        result.provenance_hash = self._compute_hash({
            "exchanger_id": exchanger_id,
            "effectiveness_degradation": effectiveness_degradation,
            "throughput_loss": throughput_loss,
            "daily_loss": daily_loss,
        })

        return result

    def calculate_cleaning_cost(
        self,
        exchanger_id: str,
        cleaning_method: CleaningMethodType,
        labor_hours: Optional[float] = None,
        expected_ua_recovery: float = 0.95,
    ) -> CleaningCost:
        """
        Calculate direct cost of cleaning intervention.

        Costs are based on cleaning method with typical values from
        industrial experience and can be customized.

        Args:
            exchanger_id: Heat exchanger identifier
            cleaning_method: Type of cleaning method
            labor_hours: Override labor hours (uses default if None)
            expected_ua_recovery: Expected UA recovery fraction (0-1)

        Returns:
            CleaningCost with calculated costs
        """
        # Default parameters by cleaning method
        method_params = {
            CleaningMethodType.CHEMICAL_ONLINE: {
                "labor_hours": 8,
                "chemical_cost": 5000,
                "equipment_cost": 1000,
                "duration_hours": 12,
                "ua_recovery": 0.85,
            },
            CleaningMethodType.CHEMICAL_OFFLINE: {
                "labor_hours": 16,
                "chemical_cost": 8000,
                "equipment_cost": 2000,
                "duration_hours": 24,
                "ua_recovery": 0.92,
            },
            CleaningMethodType.MECHANICAL_HYDROBLAST: {
                "labor_hours": 24,
                "chemical_cost": 0,
                "equipment_cost": 5000,
                "duration_hours": 36,
                "ua_recovery": 0.98,
            },
            CleaningMethodType.MECHANICAL_PIGGING: {
                "labor_hours": 12,
                "chemical_cost": 0,
                "equipment_cost": 3000,
                "duration_hours": 18,
                "ua_recovery": 0.95,
            },
            CleaningMethodType.MECHANICAL_BRUSH: {
                "labor_hours": 20,
                "chemical_cost": 0,
                "equipment_cost": 2000,
                "duration_hours": 30,
                "ua_recovery": 0.90,
            },
            CleaningMethodType.COMBINED: {
                "labor_hours": 32,
                "chemical_cost": 8000,
                "equipment_cost": 5000,
                "duration_hours": 48,
                "ua_recovery": 0.98,
            },
        }

        params = method_params.get(cleaning_method, method_params[CleaningMethodType.CHEMICAL_OFFLINE])

        actual_labor_hours = labor_hours if labor_hours is not None else params["labor_hours"]
        labor_cost = actual_labor_hours * self.config.labor_rate_usd_per_hour

        result = CleaningCost(
            exchanger_id=exchanger_id,
            cleaning_method=cleaning_method,
            labor_hours=actual_labor_hours,
            labor_rate_usd_h=self.config.labor_rate_usd_per_hour,
            labor_cost_usd=labor_cost,
            chemical_cost_usd=params["chemical_cost"],
            equipment_cost_usd=params["equipment_cost"],
            contractor_cost_usd=0.0,
            contractor_markup=self.config.contractor_markup,
            waste_disposal_cost_usd=500.0,  # Fixed estimate
            environmental_fees_usd=200.0,  # Fixed estimate
            total_cleaning_cost_usd=0.0,  # Computed by validator
            expected_ua_recovery=expected_ua_recovery if expected_ua_recovery else params["ua_recovery"],
            expected_cleaning_duration_hours=params["duration_hours"],
        )

        # Recompute total
        result.total_cleaning_cost_usd = (
            result.labor_cost_usd +
            result.chemical_cost_usd +
            result.equipment_cost_usd +
            result.waste_disposal_cost_usd +
            result.environmental_fees_usd
        )

        result.provenance_hash = self._compute_hash({
            "exchanger_id": exchanger_id,
            "method": cleaning_method.value,
            "total_cost": result.total_cleaning_cost_usd,
        })

        return result

    def calculate_downtime_cost(
        self,
        exchanger_id: str,
        cleaning_duration_hours: float,
        production_rate_tph: float,
        product_margin_usd_per_tonne: float,
        partial_operation_fraction: float = 0.0,
    ) -> DowntimeCost:
        """
        Calculate cost of lost production during cleaning downtime.

        Args:
            exchanger_id: Heat exchanger identifier
            cleaning_duration_hours: Duration of cleaning (hours)
            production_rate_tph: Normal production rate (t/h)
            product_margin_usd_per_tonne: Product margin ($/tonne)
            partial_operation_fraction: Fraction of capacity during cleaning

        Returns:
            DowntimeCost with calculated costs
        """
        hourly_margin = production_rate_tph * product_margin_usd_per_tonne

        result = DowntimeCost(
            exchanger_id=exchanger_id,
            cleaning_duration_hours=cleaning_duration_hours,
            startup_duration_hours=self.config.startup_loss_hours,
            total_outage_hours=cleaning_duration_hours + self.config.startup_loss_hours,
            production_rate_tph=production_rate_tph,
            product_margin_usd_per_tonne=product_margin_usd_per_tonne,
            hourly_margin_usd=hourly_margin,
            cleaning_period_loss_usd=0.0,  # Computed by validator
            startup_loss_usd=0.0,  # Computed by validator
            total_downtime_cost_usd=0.0,  # Computed by validator
            partial_operation_fraction=partial_operation_fraction,
        )

        result.provenance_hash = self._compute_hash({
            "exchanger_id": exchanger_id,
            "duration_hours": cleaning_duration_hours,
            "total_cost": result.total_downtime_cost_usd,
        })

        return result

    def calculate_risk_penalty(
        self,
        exchanger_id: str,
        current_delta_p_kpa: float,
        delta_p_limit_kpa: float,
        current_t_outlet_c: float,
        t_outlet_limit_c: float,
        days_since_cleaning: int,
        fouling_rate_per_day: float,
        horizon_days: int = 30,
    ) -> RiskPenalty:
        """
        Calculate probability-weighted risk penalty for constraint violations.

        Uses logistic regression-style probability model based on margin to limits
        and historical fouling trajectory.

        Args:
            exchanger_id: Heat exchanger identifier
            current_delta_p_kpa: Current pressure drop (kPa)
            delta_p_limit_kpa: Maximum allowable pressure drop (kPa)
            current_t_outlet_c: Current outlet temperature (C)
            t_outlet_limit_c: Temperature limit (C)
            days_since_cleaning: Days since last cleaning
            fouling_rate_per_day: Daily fouling rate (fractional)
            horizon_days: Risk evaluation horizon (days)

        Returns:
            RiskPenalty with probability-weighted costs
        """
        # Delta-P violation probability
        dp_margin = (delta_p_limit_kpa - current_delta_p_kpa) / delta_p_limit_kpa
        dp_trend_factor = 1 + (days_since_cleaning * fouling_rate_per_day * horizon_days)

        # Logistic probability model
        dp_violation_prob = self._logistic_probability(-dp_margin * 5 + dp_trend_factor - 1)

        # Temperature violation probability
        t_margin = (t_outlet_limit_c - current_t_outlet_c) / max(abs(t_outlet_limit_c), 1)
        t_violation_prob = self._logistic_probability(-t_margin * 5 + dp_trend_factor - 1)

        # Unplanned shutdown probability (combination of severe violations)
        shutdown_prob = dp_violation_prob * 0.3 + t_violation_prob * 0.2

        result = RiskPenalty(
            exchanger_id=exchanger_id,
            evaluation_horizon_days=horizon_days,
            current_delta_p_kpa=current_delta_p_kpa,
            delta_p_limit_kpa=delta_p_limit_kpa,
            delta_p_margin_fraction=dp_margin,
            delta_p_violation_probability=round(dp_violation_prob, 4),
            delta_p_violation_cost_usd=self.config.delta_p_violation_cost_usd,
            delta_p_expected_cost_usd=0.0,  # Computed by validator
            current_t_outlet_c=current_t_outlet_c,
            t_outlet_limit_c=t_outlet_limit_c,
            t_violation_probability=round(t_violation_prob, 4),
            t_violation_cost_usd=self.config.temperature_violation_cost_usd,
            t_expected_cost_usd=0.0,  # Computed by validator
            shutdown_probability=round(shutdown_prob, 4),
            shutdown_cost_usd=self.config.unplanned_shutdown_cost_usd,
            shutdown_expected_cost_usd=0.0,  # Computed by validator
            total_risk_penalty_usd=0.0,  # Computed by validator
        )

        result.provenance_hash = self._compute_hash({
            "exchanger_id": exchanger_id,
            "dp_prob": dp_violation_prob,
            "t_prob": t_violation_prob,
            "total_risk": result.total_risk_penalty_usd,
        })

        return result

    def calculate_total_cost(
        self,
        exchanger_id: str,
        horizon_days: int,
        current_ua_kw_k: float,
        clean_ua_kw_k: float,
        heat_duty_kw: float,
        current_effectiveness: float,
        design_effectiveness: float,
        design_throughput_tph: float,
        product_margin_usd_per_tonne: float,
        current_delta_p_kpa: float,
        delta_p_limit_kpa: float,
        current_t_outlet_c: float,
        t_outlet_limit_c: float,
        days_since_cleaning: int,
        fouling_rate_per_day: float,
        cleaning_method: Optional[CleaningMethodType] = None,
        include_cleaning: bool = False,
    ) -> TotalCostBreakdown:
        """
        Calculate complete cost breakdown over planning horizon.

        Aggregates all 5 cost components into a comprehensive breakdown.

        Args:
            exchanger_id: Heat exchanger identifier
            horizon_days: Planning horizon in days
            current_ua_kw_k: Current UA value (kW/K)
            clean_ua_kw_k: Clean UA value (kW/K)
            heat_duty_kw: Heat duty (kW)
            current_effectiveness: Current thermal effectiveness
            design_effectiveness: Design thermal effectiveness
            design_throughput_tph: Design throughput (t/h)
            product_margin_usd_per_tonne: Product margin ($/t)
            current_delta_p_kpa: Current pressure drop (kPa)
            delta_p_limit_kpa: Pressure drop limit (kPa)
            current_t_outlet_c: Current outlet temperature (C)
            t_outlet_limit_c: Outlet temperature limit (C)
            days_since_cleaning: Days since last cleaning
            fouling_rate_per_day: Daily fouling rate
            cleaning_method: Cleaning method (if include_cleaning=True)
            include_cleaning: Whether to include cleaning in costs

        Returns:
            TotalCostBreakdown with all cost components
        """
        # Calculate individual components
        energy_loss = self.calculate_energy_loss(
            exchanger_id=exchanger_id,
            current_ua_kw_k=current_ua_kw_k,
            clean_ua_kw_k=clean_ua_kw_k,
            heat_duty_kw=heat_duty_kw,
        )

        production_loss = self.calculate_production_loss(
            exchanger_id=exchanger_id,
            current_effectiveness=current_effectiveness,
            design_effectiveness=design_effectiveness,
            design_throughput_tph=design_throughput_tph,
            product_margin_usd_per_tonne=product_margin_usd_per_tonne,
        )

        risk_penalty = self.calculate_risk_penalty(
            exchanger_id=exchanger_id,
            current_delta_p_kpa=current_delta_p_kpa,
            delta_p_limit_kpa=delta_p_limit_kpa,
            current_t_outlet_c=current_t_outlet_c,
            t_outlet_limit_c=t_outlet_limit_c,
            days_since_cleaning=days_since_cleaning,
            fouling_rate_per_day=fouling_rate_per_day,
            horizon_days=horizon_days,
        )

        # Calculate horizon costs
        total_energy_loss = energy_loss.daily_energy_loss_usd * horizon_days
        total_production_loss = production_loss.daily_production_loss_usd * horizon_days
        total_risk_penalty = risk_penalty.total_risk_penalty_usd

        # Cleaning costs (optional)
        cleaning_cost = None
        downtime_cost = None
        total_cleaning = 0.0
        total_downtime = 0.0

        if include_cleaning and cleaning_method:
            cleaning_cost = self.calculate_cleaning_cost(
                exchanger_id=exchanger_id,
                cleaning_method=cleaning_method,
            )
            total_cleaning = cleaning_cost.total_cleaning_cost_usd

            production_rate = design_throughput_tph
            downtime_cost = self.calculate_downtime_cost(
                exchanger_id=exchanger_id,
                cleaning_duration_hours=cleaning_cost.expected_cleaning_duration_hours,
                production_rate_tph=production_rate,
                product_margin_usd_per_tonne=product_margin_usd_per_tonne,
            )
            total_downtime = downtime_cost.total_downtime_cost_usd

        # Grand total
        total_cost = (
            total_energy_loss +
            total_production_loss +
            total_cleaning +
            total_downtime +
            total_risk_penalty
        )

        # Percentage breakdown
        breakdown_pct = {}
        if total_cost > 0:
            breakdown_pct = {
                CostCategory.ENERGY_LOSS.value: round(100 * total_energy_loss / total_cost, 1),
                CostCategory.PRODUCTION_LOSS.value: round(100 * total_production_loss / total_cost, 1),
                CostCategory.CLEANING.value: round(100 * total_cleaning / total_cost, 1),
                CostCategory.DOWNTIME.value: round(100 * total_downtime / total_cost, 1),
                CostCategory.RISK_PENALTY.value: round(100 * total_risk_penalty / total_cost, 1),
            }

        result = TotalCostBreakdown(
            exchanger_id=exchanger_id,
            horizon_days=horizon_days,
            energy_loss_cost=energy_loss,
            production_loss_cost=production_loss,
            cleaning_cost=cleaning_cost,
            downtime_cost=downtime_cost,
            risk_penalty=risk_penalty,
            total_energy_loss_usd=round(total_energy_loss, 2),
            total_production_loss_usd=round(total_production_loss, 2),
            total_cleaning_cost_usd=round(total_cleaning, 2),
            total_downtime_cost_usd=round(total_downtime, 2),
            total_risk_penalty_usd=round(total_risk_penalty, 2),
            total_cost_usd=round(total_cost, 2),
            daily_average_cost_usd=round(total_cost / horizon_days, 2) if horizon_days > 0 else 0,
            cost_breakdown_pct=breakdown_pct,
            assumptions={
                "fuel_cost_usd_per_gj": self.config.fuel_cost_usd_per_gj,
                "production_margin_usd_per_hour": self.config.production_margin_usd_per_hour,
                "operating_hours_per_year": self.config.operating_hours_per_year,
            },
        )

        result.provenance_hash = result.compute_provenance_hash()

        logger.debug(
            f"Total cost for {exchanger_id} over {horizon_days} days: "
            f"${total_cost:,.0f} (cleaning={include_cleaning})"
        )

        return result

    def project_costs(
        self,
        exchanger_id: str,
        current_ua_kw_k: float,
        clean_ua_kw_k: float,
        heat_duty_kw: float,
        fouling_rate_per_day: float,
        horizon_days: int = 90,
    ) -> CostProjection:
        """
        Project costs forward over time assuming continued fouling.

        Args:
            exchanger_id: Heat exchanger identifier
            current_ua_kw_k: Current UA value (kW/K)
            clean_ua_kw_k: Clean UA value (kW/K)
            heat_duty_kw: Heat duty (kW)
            fouling_rate_per_day: Daily fouling rate (fractional)
            horizon_days: Projection horizon (days)

        Returns:
            CostProjection with daily and cumulative costs
        """
        start_date = datetime.utcnow()
        dates = []
        ua_trajectory = []
        daily_costs = []
        cumulative_costs = []
        cumulative_lower = []
        cumulative_upper = []

        running_total = 0.0

        for day in range(horizon_days):
            current_date = start_date + timedelta(days=day)
            dates.append(current_date)

            # Project UA degradation
            ua_factor = max(0.5, 1.0 - fouling_rate_per_day * day)
            projected_ua = clean_ua_kw_k * ua_factor
            ua_trajectory.append(projected_ua)

            # Calculate daily cost
            energy_loss = self.calculate_energy_loss(
                exchanger_id=exchanger_id,
                current_ua_kw_k=projected_ua,
                clean_ua_kw_k=clean_ua_kw_k,
                heat_duty_kw=heat_duty_kw,
            )

            daily_cost = energy_loss.daily_energy_loss_usd
            daily_costs.append(daily_cost)

            running_total += daily_cost
            cumulative_costs.append(running_total)

            # Uncertainty bounds (simplified: +/- 20%)
            cumulative_lower.append(running_total * 0.8)
            cumulative_upper.append(running_total * 1.2)

        result = CostProjection(
            exchanger_id=exchanger_id,
            projection_start=start_date,
            projection_end=start_date + timedelta(days=horizon_days),
            horizon_days=horizon_days,
            current_ua_kw_k=current_ua_kw_k,
            projected_ua_trajectory=ua_trajectory,
            fouling_rate_per_day=fouling_rate_per_day,
            dates=dates,
            daily_operating_cost_usd=daily_costs,
            cumulative_cost_usd=cumulative_costs,
            cumulative_cost_lower_usd=cumulative_lower,
            cumulative_cost_upper_usd=cumulative_upper,
        )

        result.provenance_hash = self._compute_hash({
            "exchanger_id": exchanger_id,
            "horizon_days": horizon_days,
            "total_cost": running_total,
        })

        return result

    def _logistic_probability(self, x: float) -> float:
        """
        Compute logistic probability.

        Args:
            x: Input value

        Returns:
            Probability between 0 and 1
        """
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """
        Compute SHA-256 hash for provenance tracking.

        Args:
            data: Dictionary of values to hash

        Returns:
            SHA-256 hash string (first 16 characters)
        """
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
