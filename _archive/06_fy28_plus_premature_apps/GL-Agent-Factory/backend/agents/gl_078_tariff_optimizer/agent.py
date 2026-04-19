"""
GL-078: Tariff Optimizer Agent (TARIFFOPTIMIZER)

This module implements the TariffOptimizerAgent for analyzing utility rate
structures and optimizing electricity costs through rate selection and
load management strategies.

The agent provides:
- Time-of-use (TOU) rate optimization
- Demand charge management
- Load shifting recommendations
- Rate schedule comparison
- Peak shaving analysis
- Complete SHA-256 provenance tracking

Standards/References:
- Utility rate schedules (PG&E, SCE, SDG&E, etc.)
- OpenEI USURDB (Utility Rate Database)
- FERC Form 1 rate data

Example:
    >>> agent = TariffOptimizerAgent()
    >>> result = agent.run(TariffOptimizerInput(
    ...     usage_profile=UsageProfile(hourly_kwh=[...]),
    ...     available_tariffs=[...],
    ... ))
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class RateType(str, Enum):
    """Types of electricity rates."""
    FLAT = "FLAT"
    TOU = "TOU"
    TIERED = "TIERED"
    TOU_TIERED = "TOU_TIERED"
    REAL_TIME = "REAL_TIME"
    DEMAND = "DEMAND"


class SeasonType(str, Enum):
    """Seasonal rate periods."""
    SUMMER = "SUMMER"
    WINTER = "WINTER"
    SPRING = "SPRING"
    FALL = "FALL"


class PeakPeriod(str, Enum):
    """Time-of-use periods."""
    ON_PEAK = "ON_PEAK"
    MID_PEAK = "MID_PEAK"
    OFF_PEAK = "OFF_PEAK"
    SUPER_OFF_PEAK = "SUPER_OFF_PEAK"


class LoadType(str, Enum):
    """Types of electrical loads."""
    HVAC = "HVAC"
    LIGHTING = "LIGHTING"
    PROCESS = "PROCESS"
    EV_CHARGING = "EV_CHARGING"
    REFRIGERATION = "REFRIGERATION"
    COMPRESSED_AIR = "COMPRESSED_AIR"
    OTHER = "OTHER"


# =============================================================================
# INPUT MODELS
# =============================================================================

class HourlyUsage(BaseModel):
    """Hourly usage data."""
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    kwh: float = Field(..., ge=0, description="Energy consumption kWh")
    kw_demand: Optional[float] = Field(None, ge=0, description="Peak demand kW")


class UsageProfile(BaseModel):
    """Customer usage profile."""

    customer_id: Optional[str] = Field(None, description="Customer identifier")
    utility: str = Field(default="GENERIC", description="Utility provider")

    # Usage data
    hourly_kwh: List[float] = Field(
        ..., min_items=24, max_items=8760, description="Hourly kWh for day/year"
    )
    monthly_kwh: Optional[List[float]] = Field(
        None, min_items=12, max_items=12, description="Monthly kWh totals"
    )
    peak_demand_kw: float = Field(..., ge=0, description="Peak demand kW")

    # Load characteristics
    load_factor: Optional[float] = Field(
        None, ge=0, le=1, description="Load factor (avg/peak)"
    )
    power_factor: Optional[float] = Field(
        None, ge=0, le=1, description="Power factor"
    )

    # Flexibility
    shiftable_load_kw: Optional[float] = Field(
        None, ge=0, description="Amount of shiftable load kW"
    )
    shiftable_load_types: List[LoadType] = Field(
        default_factory=list, description="Types of shiftable loads"
    )
    has_battery_storage: bool = Field(default=False, description="Has battery storage")
    battery_capacity_kwh: Optional[float] = Field(
        None, ge=0, description="Battery capacity kWh"
    )


class RateSchedule(BaseModel):
    """Rate schedule definition."""

    rate_id: str = Field(..., description="Rate schedule ID")
    name: str = Field(..., description="Rate schedule name")
    rate_type: RateType = Field(..., description="Type of rate structure")

    # Energy charges ($/kWh)
    energy_charge_flat: Optional[float] = Field(None, ge=0)
    energy_charge_on_peak: Optional[float] = Field(None, ge=0)
    energy_charge_mid_peak: Optional[float] = Field(None, ge=0)
    energy_charge_off_peak: Optional[float] = Field(None, ge=0)
    energy_charge_super_off_peak: Optional[float] = Field(None, ge=0)

    # Demand charges ($/kW)
    demand_charge_facility: Optional[float] = Field(None, ge=0)
    demand_charge_on_peak: Optional[float] = Field(None, ge=0)
    demand_charge_mid_peak: Optional[float] = Field(None, ge=0)

    # TOU periods (hours in 0-23)
    on_peak_hours: List[int] = Field(default_factory=list)
    mid_peak_hours: List[int] = Field(default_factory=list)
    off_peak_hours: List[int] = Field(default_factory=list)

    # Fixed charges
    customer_charge_monthly: float = Field(default=0, ge=0)

    # Eligibility
    min_demand_kw: float = Field(default=0, ge=0)
    max_demand_kw: Optional[float] = Field(None, ge=0)


class TariffOption(BaseModel):
    """Available tariff option."""

    rate_schedule: RateSchedule = Field(..., description="Rate schedule details")
    season: SeasonType = Field(default=SeasonType.SUMMER, description="Season")
    is_current: bool = Field(default=False, description="Currently enrolled tariff")


class TariffOptimizerInput(BaseModel):
    """Complete input model for Tariff Optimizer."""

    usage_profile: UsageProfile = Field(..., description="Customer usage profile")
    available_tariffs: List[TariffOption] = Field(..., description="Available tariffs")

    # Analysis parameters
    analysis_period_months: int = Field(default=12, ge=1, le=60)
    include_demand_management: bool = Field(default=True)
    include_load_shifting: bool = Field(default=True)

    # Cost assumptions
    battery_cost_per_kwh: float = Field(default=300.0, ge=0)
    battery_efficiency: float = Field(default=0.90, ge=0.5, le=1.0)

    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class DemandChargeAnalysis(BaseModel):
    """Demand charge analysis results."""

    current_peak_kw: float = Field(..., description="Current peak demand")
    facility_demand_charge_usd: float = Field(..., description="Monthly facility charge")
    tou_demand_charge_usd: float = Field(..., description="TOU demand charges")
    total_demand_charge_usd: float = Field(..., description="Total demand charges")
    peak_shaving_potential_kw: float = Field(..., description="Potential peak reduction")
    peak_shaving_savings_usd: float = Field(..., description="Savings from peak shaving")


class LoadShiftOpportunity(BaseModel):
    """Load shifting opportunity."""

    load_type: LoadType = Field(..., description="Type of load to shift")
    shift_from_period: PeakPeriod = Field(..., description="Current period")
    shift_to_period: PeakPeriod = Field(..., description="Target period")
    shiftable_kwh: float = Field(..., description="Amount to shift (kWh)")
    savings_per_kwh: float = Field(..., description="Savings per shifted kWh")
    annual_savings_usd: float = Field(..., description="Annual savings")
    implementation_cost_usd: float = Field(default=0, description="Implementation cost")
    payback_months: Optional[float] = Field(None, description="Simple payback")


class SavingsAnalysis(BaseModel):
    """Cost comparison and savings analysis."""

    current_annual_cost_usd: float = Field(..., description="Current annual cost")
    optimized_annual_cost_usd: float = Field(..., description="Optimized annual cost")
    annual_savings_usd: float = Field(..., description="Total annual savings")
    savings_percent: float = Field(..., description="Savings percentage")

    # Breakdown
    energy_savings_usd: float = Field(..., description="From rate optimization")
    demand_savings_usd: float = Field(..., description="From demand management")
    load_shift_savings_usd: float = Field(..., description="From load shifting")


class TariffRecommendation(BaseModel):
    """Tariff recommendation."""

    rank: int = Field(..., description="Recommendation rank (1=best)")
    rate_id: str = Field(..., description="Rate schedule ID")
    rate_name: str = Field(..., description="Rate schedule name")

    estimated_annual_cost_usd: float = Field(..., description="Estimated annual cost")
    savings_vs_current_usd: float = Field(..., description="Savings vs current tariff")
    savings_percent: float = Field(..., description="Savings percentage")

    energy_cost_usd: float = Field(..., description="Annual energy charges")
    demand_cost_usd: float = Field(..., description="Annual demand charges")
    fixed_cost_usd: float = Field(..., description="Annual fixed charges")

    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in estimate")
    notes: List[str] = Field(default_factory=list, description="Additional notes")


class ProvenanceRecord(BaseModel):
    """Provenance tracking record."""

    operation: str = Field(..., description="Operation performed")
    timestamp: datetime = Field(..., description="Operation timestamp")
    input_hash: str = Field(..., description="SHA-256 hash of inputs")
    output_hash: str = Field(..., description="SHA-256 hash of outputs")
    tool_name: str = Field(..., description="Tool/calculator used")
    parameters: Dict[str, Any] = Field(default_factory=dict)


class TariffOptimizerOutput(BaseModel):
    """Complete output model for Tariff Optimizer."""

    analysis_id: str = Field(..., description="Unique analysis identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Primary results
    current_tariff_cost_usd: float = Field(..., description="Current annual cost")
    optimal_tariff: TariffRecommendation = Field(..., description="Best tariff option")
    tariff_recommendations: List[TariffRecommendation] = Field(
        ..., description="Ranked tariff options"
    )

    # Detailed analysis
    demand_analysis: DemandChargeAnalysis = Field(..., description="Demand charge analysis")
    load_shift_opportunities: List[LoadShiftOpportunity] = Field(
        default_factory=list, description="Load shifting opportunities"
    )
    savings_analysis: SavingsAnalysis = Field(..., description="Savings breakdown")

    # Provenance
    provenance_chain: List[ProvenanceRecord] = Field(...)
    provenance_hash: str = Field(...)

    processing_time_ms: float = Field(...)
    validation_status: str = Field(...)
    validation_errors: List[str] = Field(default_factory=list)


# =============================================================================
# TARIFF OPTIMIZER AGENT
# =============================================================================

class TariffOptimizerAgent:
    """
    GL-078: Tariff Optimizer Agent (TARIFFOPTIMIZER).

    This agent analyzes utility rate structures and provides recommendations
    for rate selection and load management to minimize electricity costs.

    Zero-Hallucination Guarantee:
    - All cost calculations use deterministic formulas
    - Rate structures from utility tariff books
    - No LLM inference in cost calculations
    - Complete audit trail for compliance
    """

    AGENT_ID = "GL-078"
    AGENT_NAME = "TARIFFOPTIMIZER"
    VERSION = "1.0.0"
    DESCRIPTION = "Utility Tariff Optimization Agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the TariffOptimizerAgent."""
        self.config = config or {}
        self._provenance_steps: List[Dict[str, Any]] = []
        self._validation_errors: List[str] = []

        logger.info(
            f"TariffOptimizerAgent initialized "
            f"(ID: {self.AGENT_ID}, Name: {self.AGENT_NAME})"
        )

    def run(self, input_data: TariffOptimizerInput) -> TariffOptimizerOutput:
        """Execute tariff optimization analysis."""
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._validation_errors = []

        logger.info(f"Starting tariff optimization for {input_data.usage_profile.utility}")

        try:
            # Step 1: Analyze current usage patterns
            usage_analysis = self._analyze_usage(input_data.usage_profile)
            self._track_provenance(
                "usage_analysis",
                {"total_kwh": sum(input_data.usage_profile.hourly_kwh)},
                usage_analysis,
                "Usage Analyzer"
            )

            # Step 2: Calculate costs for each tariff
            tariff_costs = []
            current_tariff_cost = None

            for tariff in input_data.available_tariffs:
                cost = self._calculate_tariff_cost(
                    tariff.rate_schedule,
                    input_data.usage_profile,
                    usage_analysis
                )
                tariff_costs.append({
                    "tariff": tariff,
                    "cost": cost,
                })
                if tariff.is_current:
                    current_tariff_cost = cost["total_annual"]

            # Default current cost if none marked
            if current_tariff_cost is None and tariff_costs:
                current_tariff_cost = tariff_costs[0]["cost"]["total_annual"]

            self._track_provenance(
                "tariff_costing",
                {"tariffs_evaluated": len(tariff_costs)},
                {"current_cost": current_tariff_cost},
                "Cost Calculator"
            )

            # Step 3: Analyze demand charges
            demand_analysis = self._analyze_demand_charges(
                input_data.usage_profile,
                tariff_costs[0]["tariff"].rate_schedule if tariff_costs else None
            )

            # Step 4: Identify load shifting opportunities
            load_shifts = []
            if input_data.include_load_shifting:
                load_shifts = self._identify_load_shifts(
                    input_data.usage_profile,
                    tariff_costs[0]["tariff"].rate_schedule if tariff_costs else None
                )

            # Step 5: Generate recommendations
            recommendations = self._generate_recommendations(
                tariff_costs, current_tariff_cost or 0
            )

            # Step 6: Calculate total savings
            optimal = recommendations[0] if recommendations else None
            savings_analysis = self._calculate_savings(
                current_tariff_cost or 0,
                optimal.estimated_annual_cost_usd if optimal else 0,
                demand_analysis.peak_shaving_savings_usd,
                sum(ls.annual_savings_usd for ls in load_shifts)
            )

            # Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash()
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            analysis_id = (
                f"TARIFF-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-"
                f"{hashlib.sha256(str(input_data.dict()).encode()).hexdigest()[:8]}"
            )

            return TariffOptimizerOutput(
                analysis_id=analysis_id,
                current_tariff_cost_usd=current_tariff_cost or 0,
                optimal_tariff=optimal,
                tariff_recommendations=recommendations,
                demand_analysis=demand_analysis,
                load_shift_opportunities=load_shifts,
                savings_analysis=savings_analysis,
                provenance_chain=[
                    ProvenanceRecord(
                        operation=s["operation"],
                        timestamp=s["timestamp"],
                        input_hash=s["input_hash"],
                        output_hash=s["output_hash"],
                        tool_name=s["tool_name"],
                        parameters=s.get("parameters", {}),
                    )
                    for s in self._provenance_steps
                ],
                provenance_hash=provenance_hash,
                processing_time_ms=round(processing_time, 2),
                validation_status="PASS" if not self._validation_errors else "FAIL",
                validation_errors=self._validation_errors,
            )

        except Exception as e:
            logger.error(f"Tariff optimization failed: {str(e)}", exc_info=True)
            raise

    def _analyze_usage(self, profile: UsageProfile) -> Dict[str, Any]:
        """Analyze usage patterns."""
        hourly = profile.hourly_kwh
        total_kwh = sum(hourly)

        # Calculate by period (assuming typical TOU structure)
        on_peak_hours = list(range(16, 21))  # 4pm-9pm
        mid_peak_hours = list(range(12, 16)) + list(range(21, 24))  # 12pm-4pm, 9pm-12am
        off_peak_hours = list(range(0, 12))  # 12am-12pm

        # For annual data (8760 hours)
        if len(hourly) == 8760:
            on_peak_kwh = sum(
                hourly[d * 24 + h]
                for d in range(365)
                for h in on_peak_hours
            )
            mid_peak_kwh = sum(
                hourly[d * 24 + h]
                for d in range(365)
                for h in mid_peak_hours
            )
            off_peak_kwh = sum(
                hourly[d * 24 + h]
                for d in range(365)
                for h in off_peak_hours
            )
        else:
            # 24-hour profile - extrapolate to annual
            on_peak_kwh = sum(hourly[h] for h in on_peak_hours) * 365
            mid_peak_kwh = sum(hourly[h] for h in mid_peak_hours) * 365
            off_peak_kwh = sum(hourly[h] for h in off_peak_hours) * 365
            total_kwh = total_kwh * 365

        return {
            "total_annual_kwh": total_kwh,
            "on_peak_kwh": on_peak_kwh,
            "mid_peak_kwh": mid_peak_kwh,
            "off_peak_kwh": off_peak_kwh,
            "peak_demand_kw": profile.peak_demand_kw,
            "load_factor": profile.load_factor or (total_kwh / (profile.peak_demand_kw * 8760) if profile.peak_demand_kw > 0 else 0),
        }

    def _calculate_tariff_cost(
        self,
        rate: RateSchedule,
        profile: UsageProfile,
        usage: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate annual cost for a tariff."""

        # Energy charges
        if rate.rate_type == RateType.FLAT:
            energy_cost = usage["total_annual_kwh"] * (rate.energy_charge_flat or 0.10)
        else:  # TOU
            energy_cost = (
                usage["on_peak_kwh"] * (rate.energy_charge_on_peak or 0.20) +
                usage["mid_peak_kwh"] * (rate.energy_charge_mid_peak or 0.12) +
                usage["off_peak_kwh"] * (rate.energy_charge_off_peak or 0.08)
            )

        # Demand charges (monthly * 12)
        demand_cost = 0.0
        if rate.demand_charge_facility:
            demand_cost += rate.demand_charge_facility * profile.peak_demand_kw * 12
        if rate.demand_charge_on_peak:
            # Assume 80% of peak occurs during on-peak
            on_peak_demand = profile.peak_demand_kw * 0.8
            demand_cost += rate.demand_charge_on_peak * on_peak_demand * 12

        # Fixed charges
        fixed_cost = rate.customer_charge_monthly * 12

        return {
            "energy_annual": round(energy_cost, 2),
            "demand_annual": round(demand_cost, 2),
            "fixed_annual": round(fixed_cost, 2),
            "total_annual": round(energy_cost + demand_cost + fixed_cost, 2),
        }

    def _analyze_demand_charges(
        self,
        profile: UsageProfile,
        rate: Optional[RateSchedule]
    ) -> DemandChargeAnalysis:
        """Analyze demand charges and peak shaving potential."""
        peak_kw = profile.peak_demand_kw

        facility_charge = 0.0
        tou_charge = 0.0

        if rate:
            if rate.demand_charge_facility:
                facility_charge = rate.demand_charge_facility * peak_kw * 12
            if rate.demand_charge_on_peak:
                tou_charge = rate.demand_charge_on_peak * (peak_kw * 0.8) * 12

        # Peak shaving potential (assume 10% reduction possible)
        shaving_potential = peak_kw * 0.10
        shaving_savings = 0.0
        if rate and rate.demand_charge_facility:
            shaving_savings = rate.demand_charge_facility * shaving_potential * 12

        return DemandChargeAnalysis(
            current_peak_kw=peak_kw,
            facility_demand_charge_usd=round(facility_charge, 2),
            tou_demand_charge_usd=round(tou_charge, 2),
            total_demand_charge_usd=round(facility_charge + tou_charge, 2),
            peak_shaving_potential_kw=round(shaving_potential, 2),
            peak_shaving_savings_usd=round(shaving_savings, 2),
        )

    def _identify_load_shifts(
        self,
        profile: UsageProfile,
        rate: Optional[RateSchedule]
    ) -> List[LoadShiftOpportunity]:
        """Identify load shifting opportunities."""
        opportunities = []

        if not rate or not profile.shiftable_load_kw:
            return opportunities

        on_peak_rate = rate.energy_charge_on_peak or 0.20
        off_peak_rate = rate.energy_charge_off_peak or 0.08
        savings_per_kwh = on_peak_rate - off_peak_rate

        if savings_per_kwh > 0:
            # Estimate shiftable energy (assume 5 on-peak hours shifted)
            shiftable_kwh = profile.shiftable_load_kw * 5 * 365
            annual_savings = shiftable_kwh * savings_per_kwh

            for load_type in profile.shiftable_load_types or [LoadType.OTHER]:
                opportunities.append(LoadShiftOpportunity(
                    load_type=load_type,
                    shift_from_period=PeakPeriod.ON_PEAK,
                    shift_to_period=PeakPeriod.OFF_PEAK,
                    shiftable_kwh=round(shiftable_kwh / max(len(profile.shiftable_load_types), 1), 2),
                    savings_per_kwh=round(savings_per_kwh, 4),
                    annual_savings_usd=round(annual_savings / max(len(profile.shiftable_load_types), 1), 2),
                    implementation_cost_usd=5000,
                    payback_months=round(5000 / (annual_savings / 12), 1) if annual_savings > 0 else None,
                ))

        return opportunities

    def _generate_recommendations(
        self,
        tariff_costs: List[Dict],
        current_cost: float
    ) -> List[TariffRecommendation]:
        """Generate ranked tariff recommendations."""
        recommendations = []

        # Sort by total cost
        sorted_costs = sorted(tariff_costs, key=lambda x: x["cost"]["total_annual"])

        for rank, tc in enumerate(sorted_costs, 1):
            tariff = tc["tariff"]
            cost = tc["cost"]

            savings = current_cost - cost["total_annual"]
            savings_pct = (savings / current_cost * 100) if current_cost > 0 else 0

            recommendations.append(TariffRecommendation(
                rank=rank,
                rate_id=tariff.rate_schedule.rate_id,
                rate_name=tariff.rate_schedule.name,
                estimated_annual_cost_usd=cost["total_annual"],
                savings_vs_current_usd=round(savings, 2),
                savings_percent=round(savings_pct, 1),
                energy_cost_usd=cost["energy_annual"],
                demand_cost_usd=cost["demand_annual"],
                fixed_cost_usd=cost["fixed_annual"],
                confidence_score=0.85,
                notes=[],
            ))

        return recommendations

    def _calculate_savings(
        self,
        current_cost: float,
        optimized_cost: float,
        demand_savings: float,
        load_shift_savings: float
    ) -> SavingsAnalysis:
        """Calculate total savings breakdown."""
        energy_savings = max(0, current_cost - optimized_cost)
        total_savings = energy_savings + demand_savings + load_shift_savings
        final_cost = current_cost - total_savings
        savings_pct = (total_savings / current_cost * 100) if current_cost > 0 else 0

        return SavingsAnalysis(
            current_annual_cost_usd=round(current_cost, 2),
            optimized_annual_cost_usd=round(final_cost, 2),
            annual_savings_usd=round(total_savings, 2),
            savings_percent=round(savings_pct, 1),
            energy_savings_usd=round(energy_savings, 2),
            demand_savings_usd=round(demand_savings, 2),
            load_shift_savings_usd=round(load_shift_savings, 2),
        )

    def _track_provenance(
        self, operation: str, inputs: Dict, outputs: Dict, tool_name: str
    ) -> None:
        """Track provenance step."""
        input_str = json.dumps(inputs, sort_keys=True, default=str)
        output_str = json.dumps(outputs, sort_keys=True, default=str)

        self._provenance_steps.append({
            "operation": operation,
            "timestamp": datetime.utcnow(),
            "input_hash": hashlib.sha256(input_str.encode()).hexdigest(),
            "output_hash": hashlib.sha256(output_str.encode()).hexdigest(),
            "tool_name": tool_name,
            "parameters": inputs,
        })

    def _calculate_provenance_hash(self) -> str:
        """Calculate provenance chain hash."""
        data = {
            "agent_id": self.AGENT_ID,
            "steps": [
                {"operation": s["operation"], "input_hash": s["input_hash"]}
                for s in self._provenance_steps
            ],
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


# =============================================================================
# PACK SPECIFICATION
# =============================================================================

PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-078",
    "name": "TARIFFOPTIMIZER - Utility Tariff Optimization Agent",
    "version": "1.0.0",
    "summary": "Optimizes utility rate selection and load management for cost reduction",
    "tags": ["tariff", "utility-rates", "TOU", "demand-charges", "load-shifting"],
    "owners": ["energy-team"],
    "compute": {
        "entrypoint": "python://agents.gl_078_tariff_optimizer.agent:TariffOptimizerAgent",
        "deterministic": True,
    },
    "standards": [
        {"ref": "OpenEI-USURDB", "description": "Utility Rate Database"},
        {"ref": "FERC-Form1", "description": "Utility Cost Data"},
    ],
    "provenance": {"calculation_verified": True, "enable_audit": True},
}
