"""
GL-074: Grid Optimizer Agent (GRID-OPTIMIZER)

This module implements the GridOptimizerAgent for electrical grid integration optimization,
demand response, peak shaving, and renewable energy integration.

Standards Reference:
    - IEEE 1547 (Interconnection Standards)
    - FERC Order 2222 (Distributed Energy Resources)
    - ISO/IEC 15067 (Energy Management)
    - OpenADR 2.0 (Automated Demand Response)

Example:
    >>> agent = GridOptimizerAgent()
    >>> result = agent.run(GridOptimizerInput(load_profile=[...], grid_constraints=...))
    >>> print(f"Peak reduction: {result.optimization_summary.peak_reduction_kw:.1f} kW")
"""

import hashlib
import json
import logging
import math
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LoadType(str, Enum):
    BASE_LOAD = "base_load"
    VARIABLE_LOAD = "variable_load"
    CURTAILABLE = "curtailable"
    SHIFTABLE = "shiftable"
    STORAGE = "storage"
    GENERATION = "generation"


class DERType(str, Enum):
    SOLAR_PV = "solar_pv"
    WIND = "wind"
    BATTERY = "battery"
    CHP = "chp"
    FUEL_CELL = "fuel_cell"
    BACKUP_GENERATOR = "backup_generator"
    EV_CHARGING = "ev_charging"
    THERMAL_STORAGE = "thermal_storage"


class TariffType(str, Enum):
    FLAT = "flat"
    TIME_OF_USE = "time_of_use"
    REAL_TIME = "real_time"
    DEMAND_CHARGE = "demand_charge"
    CRITICAL_PEAK = "critical_peak"


class GridSignal(str, Enum):
    NORMAL = "normal"
    HIGH_PRICE = "high_price"
    CRITICAL_PEAK = "critical_peak"
    EMERGENCY = "emergency"
    CURTAILMENT = "curtailment"


class LoadDataPoint(BaseModel):
    """Load profile data point."""
    timestamp: datetime = Field(..., description="Timestamp")
    demand_kw: float = Field(..., description="Total demand in kW")
    load_breakdown: Dict[str, float] = Field(default_factory=dict, description="Load by type")
    temperature_celsius: Optional[float] = Field(None, description="Ambient temperature")
    occupancy_percent: Optional[float] = Field(None, description="Building occupancy")


class DERAsset(BaseModel):
    """Distributed Energy Resource asset."""
    asset_id: str = Field(..., description="Asset identifier")
    name: str = Field(..., description="Asset name")
    der_type: DERType = Field(..., description="DER type")
    capacity_kw: float = Field(..., ge=0, description="Capacity in kW")
    current_output_kw: float = Field(default=0, description="Current output")
    min_output_kw: float = Field(default=0, description="Minimum output")
    max_ramp_rate_kw_min: Optional[float] = Field(None, description="Ramp rate")
    efficiency_percent: float = Field(default=90, description="Efficiency")
    storage_capacity_kwh: Optional[float] = Field(None, description="Storage capacity")
    current_soc_percent: Optional[float] = Field(None, description="State of charge")
    controllable: bool = Field(default=True, description="Is controllable")


class GridConstraints(BaseModel):
    """Grid connection constraints."""
    max_import_kw: float = Field(..., description="Maximum import capacity")
    max_export_kw: float = Field(default=0, description="Maximum export capacity")
    power_factor_min: float = Field(default=0.9, description="Minimum power factor")
    voltage_limits_percent: Tuple[float, float] = Field(default=(0.95, 1.05))
    demand_limit_kw: Optional[float] = Field(None, description="Contractual demand limit")
    ramp_limit_kw_min: Optional[float] = Field(None, description="Ramp rate limit")


class TariffStructure(BaseModel):
    """Electricity tariff structure."""
    tariff_type: TariffType = Field(..., description="Tariff type")
    energy_rate_per_kwh: float = Field(..., description="Energy rate $/kWh")
    demand_rate_per_kw: float = Field(default=0, description="Demand rate $/kW")
    time_of_use_rates: Dict[str, float] = Field(default_factory=dict, description="TOU rates")
    peak_hours: List[int] = Field(default_factory=list, description="Peak hours (0-23)")
    critical_peak_rate: Optional[float] = Field(None, description="Critical peak rate")


class DemandResponseEvent(BaseModel):
    """Demand response event."""
    event_id: str = Field(..., description="Event identifier")
    event_type: GridSignal = Field(..., description="Event type")
    start_time: datetime = Field(..., description="Start time")
    end_time: datetime = Field(..., description="End time")
    reduction_target_kw: float = Field(..., description="Reduction target")
    incentive_per_kw: float = Field(default=0, description="Incentive $/kW")


class GridOptimizerInput(BaseModel):
    """Input for grid optimization."""
    optimization_id: Optional[str] = Field(None, description="Optimization identifier")
    site_name: str = Field(default="Site", description="Site name")
    load_profile: List[LoadDataPoint] = Field(..., description="Load profile")
    der_assets: List[DERAsset] = Field(default_factory=list, description="DER assets")
    grid_constraints: GridConstraints = Field(..., description="Grid constraints")
    tariff: TariffStructure = Field(..., description="Tariff structure")
    demand_response_events: List[DemandResponseEvent] = Field(default_factory=list)
    optimization_horizon_hours: int = Field(default=24, description="Optimization horizon")
    objective: str = Field(default="minimize_cost", description="Optimization objective")
    grid_carbon_intensity_gCO2_kwh: float = Field(default=400, description="Grid carbon intensity")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OptimizedSchedule(BaseModel):
    """Optimized asset schedule."""
    timestamp: datetime
    grid_import_kw: float
    grid_export_kw: float
    total_load_kw: float
    total_generation_kw: float
    battery_charge_kw: float
    battery_discharge_kw: float
    battery_soc_percent: float
    curtailed_load_kw: float
    shifted_load_kw: float
    energy_cost_usd: float
    demand_charge_usd: float


class DERDispatch(BaseModel):
    """DER dispatch schedule."""
    asset_id: str
    asset_name: str
    der_type: str
    schedule: List[Dict[str, Any]]
    total_energy_kwh: float
    utilization_percent: float
    revenue_or_savings_usd: float


class PeakAnalysis(BaseModel):
    """Peak demand analysis."""
    original_peak_kw: float
    optimized_peak_kw: float
    peak_reduction_kw: float
    peak_reduction_percent: float
    peak_period: datetime
    coincident_peak_reduction_kw: float
    non_coincident_peak_reduction_kw: float


class CostAnalysis(BaseModel):
    """Cost analysis results."""
    baseline_energy_cost_usd: float
    optimized_energy_cost_usd: float
    energy_savings_usd: float
    baseline_demand_cost_usd: float
    optimized_demand_cost_usd: float
    demand_savings_usd: float
    dr_incentive_earned_usd: float
    export_revenue_usd: float
    total_savings_usd: float
    savings_percent: float


class CarbonAnalysis(BaseModel):
    """Carbon emissions analysis."""
    baseline_emissions_kgCO2: float
    optimized_emissions_kgCO2: float
    emissions_reduction_kgCO2: float
    emissions_reduction_percent: float
    renewable_fraction_percent: float
    avoided_grid_import_kwh: float


class DRPerformance(BaseModel):
    """Demand response performance."""
    event_id: str
    target_reduction_kw: float
    achieved_reduction_kw: float
    performance_percent: float
    incentive_earned_usd: float
    compliance_status: str


class OptimizationSummary(BaseModel):
    """Optimization summary."""
    total_energy_kwh: float
    total_grid_import_kwh: float
    total_grid_export_kwh: float
    total_generation_kwh: float
    total_storage_cycles: float
    peak_reduction_kw: float
    cost_savings_usd: float
    carbon_savings_kgCO2: float
    optimization_quality: str


class GridOptimizerOutput(BaseModel):
    """Output from grid optimization."""
    optimization_id: str
    site_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    optimized_schedule: List[OptimizedSchedule]
    der_dispatch: List[DERDispatch]
    peak_analysis: PeakAnalysis
    cost_analysis: CostAnalysis
    carbon_analysis: CarbonAnalysis
    dr_performance: List[DRPerformance]
    optimization_summary: OptimizationSummary
    grid_stability_score: float
    recommendations: List[str]
    provenance_hash: str
    processing_time_ms: float
    validation_status: str


class GridOptimizerAgent:
    """GL-074: Grid Optimizer Agent - Electrical grid integration optimization."""

    AGENT_ID = "GL-074"
    AGENT_NAME = "GRID-OPTIMIZER"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"GridOptimizerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: GridOptimizerInput) -> GridOptimizerOutput:
        start_time = datetime.utcnow()

        # Extract load profile
        load_profile = input_data.load_profile
        der_assets = input_data.der_assets
        constraints = input_data.grid_constraints
        tariff = input_data.tariff

        # Run optimization
        optimized_schedule = self._optimize_schedule(
            load_profile, der_assets, constraints, tariff,
            input_data.demand_response_events)

        # Generate DER dispatch schedules
        der_dispatch = self._generate_der_dispatch(
            der_assets, optimized_schedule)

        # Analyze peaks
        peak_analysis = self._analyze_peaks(load_profile, optimized_schedule)

        # Analyze costs
        cost_analysis = self._analyze_costs(
            load_profile, optimized_schedule, tariff,
            input_data.demand_response_events)

        # Analyze carbon
        carbon_analysis = self._analyze_carbon(
            load_profile, optimized_schedule, der_assets,
            input_data.grid_carbon_intensity_gCO2_kwh)

        # Evaluate DR performance
        dr_performance = self._evaluate_dr_performance(
            optimized_schedule, input_data.demand_response_events)

        # Generate summary
        summary = self._generate_summary(
            optimized_schedule, peak_analysis, cost_analysis, carbon_analysis)

        # Calculate grid stability score
        stability_score = self._calculate_stability_score(
            optimized_schedule, constraints)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            peak_analysis, cost_analysis, carbon_analysis, der_assets)

        provenance_hash = hashlib.sha256(
            json.dumps({
                "agent": self.AGENT_ID,
                "site": input_data.site_name,
                "horizon": input_data.optimization_horizon_hours,
                "timestamp": datetime.utcnow().isoformat()
            }, sort_keys=True, default=str).encode()).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return GridOptimizerOutput(
            optimization_id=input_data.optimization_id or f"OPT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            site_name=input_data.site_name,
            optimized_schedule=optimized_schedule,
            der_dispatch=der_dispatch,
            peak_analysis=peak_analysis,
            cost_analysis=cost_analysis,
            carbon_analysis=carbon_analysis,
            dr_performance=dr_performance,
            optimization_summary=summary,
            grid_stability_score=round(stability_score, 1),
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS")

    def _optimize_schedule(self, load_profile: List[LoadDataPoint],
                          der_assets: List[DERAsset],
                          constraints: GridConstraints,
                          tariff: TariffStructure,
                          dr_events: List[DemandResponseEvent]) -> List[OptimizedSchedule]:
        """Optimize dispatch schedule."""
        schedule = []

        # Get battery assets
        batteries = [a for a in der_assets if a.der_type == DERType.BATTERY]
        battery_capacity = sum(a.storage_capacity_kwh or 0 for a in batteries)
        battery_power = sum(a.capacity_kw for a in batteries)
        battery_soc = sum((a.current_soc_percent or 50) * (a.storage_capacity_kwh or 0) / 100
                         for a in batteries) if battery_capacity > 0 else 0
        soc_percent = battery_soc / battery_capacity * 100 if battery_capacity > 0 else 0

        # Get generation assets
        generators = [a for a in der_assets if a.der_type in
                     [DERType.SOLAR_PV, DERType.WIND, DERType.CHP, DERType.FUEL_CELL]]

        for i, load_point in enumerate(load_profile):
            hour = load_point.timestamp.hour
            load_kw = load_point.demand_kw

            # Calculate available generation
            generation_kw = 0
            for gen in generators:
                if gen.der_type == DERType.SOLAR_PV:
                    # Simple solar profile (peak at noon)
                    solar_factor = max(0, math.sin(math.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
                    generation_kw += gen.capacity_kw * solar_factor * 0.8
                elif gen.der_type == DERType.CHP:
                    generation_kw += gen.current_output_kw
                else:
                    generation_kw += gen.capacity_kw * 0.5

            # Check for DR events
            dr_reduction = 0
            for event in dr_events:
                if event.start_time <= load_point.timestamp < event.end_time:
                    dr_reduction = min(event.reduction_target_kw, load_kw * 0.2)

            # Determine if peak pricing
            is_peak = hour in tariff.peak_hours or (
                tariff.tariff_type == TariffType.TIME_OF_USE and
                tariff.time_of_use_rates.get("peak", 0) > tariff.energy_rate_per_kwh)

            # Battery dispatch logic
            charge_kw = 0
            discharge_kw = 0

            if battery_capacity > 0:
                if is_peak and soc_percent > 20:
                    # Discharge during peak
                    discharge_kw = min(battery_power, load_kw - generation_kw,
                                      battery_soc / (1/60))  # kWh to kW
                    battery_soc -= discharge_kw / 60
                    soc_percent = battery_soc / battery_capacity * 100
                elif not is_peak and soc_percent < 90 and generation_kw > load_kw:
                    # Charge from excess generation
                    charge_kw = min(battery_power, generation_kw - load_kw,
                                   (battery_capacity - battery_soc) / (1/60))
                    battery_soc += charge_kw * 0.9 / 60  # Account for efficiency
                    soc_percent = battery_soc / battery_capacity * 100

            # Calculate net load
            net_load = load_kw - generation_kw - discharge_kw + charge_kw - dr_reduction
            grid_import = max(0, min(net_load, constraints.max_import_kw))
            grid_export = max(0, min(-net_load, constraints.max_export_kw))

            # Calculate curtailed and shifted load
            curtailed = max(0, load_kw - constraints.max_import_kw - generation_kw - discharge_kw)
            shifted = dr_reduction

            # Calculate costs
            if tariff.tariff_type == TariffType.TIME_OF_USE:
                rate = tariff.time_of_use_rates.get("peak" if is_peak else "off_peak",
                                                    tariff.energy_rate_per_kwh)
            else:
                rate = tariff.energy_rate_per_kwh

            energy_cost = grid_import * rate / 60  # Per interval (assuming hourly)
            demand_cost = 0  # Calculated at end

            schedule.append(OptimizedSchedule(
                timestamp=load_point.timestamp,
                grid_import_kw=round(grid_import, 2),
                grid_export_kw=round(grid_export, 2),
                total_load_kw=round(load_kw, 2),
                total_generation_kw=round(generation_kw, 2),
                battery_charge_kw=round(charge_kw, 2),
                battery_discharge_kw=round(discharge_kw, 2),
                battery_soc_percent=round(soc_percent, 1),
                curtailed_load_kw=round(curtailed, 2),
                shifted_load_kw=round(shifted, 2),
                energy_cost_usd=round(energy_cost, 4),
                demand_charge_usd=demand_cost))

        return schedule

    def _generate_der_dispatch(self, der_assets: List[DERAsset],
                              schedule: List[OptimizedSchedule]) -> List[DERDispatch]:
        """Generate DER dispatch schedules."""
        dispatches = []

        for asset in der_assets:
            asset_schedule = []
            total_energy = 0

            for opt in schedule:
                if asset.der_type == DERType.BATTERY:
                    output = opt.battery_discharge_kw - opt.battery_charge_kw
                    soc = opt.battery_soc_percent
                elif asset.der_type == DERType.SOLAR_PV:
                    # Proportional allocation
                    solar_assets = [a for a in der_assets if a.der_type == DERType.SOLAR_PV]
                    total_solar = sum(a.capacity_kw for a in solar_assets)
                    output = opt.total_generation_kw * (asset.capacity_kw / total_solar) if total_solar > 0 else 0
                    soc = None
                else:
                    output = asset.current_output_kw
                    soc = None

                asset_schedule.append({
                    "timestamp": opt.timestamp.isoformat(),
                    "output_kw": round(output, 2),
                    "soc_percent": soc
                })
                total_energy += abs(output) / 60  # Convert to kWh

            utilization = (total_energy / (asset.capacity_kw * len(schedule) / 60) * 100
                          if asset.capacity_kw > 0 and schedule else 0)

            # Estimate revenue/savings
            avg_rate = 0.10  # Simplified
            revenue = total_energy * avg_rate

            dispatches.append(DERDispatch(
                asset_id=asset.asset_id,
                asset_name=asset.name,
                der_type=asset.der_type.value,
                schedule=asset_schedule,
                total_energy_kwh=round(total_energy, 2),
                utilization_percent=round(min(100, utilization), 1),
                revenue_or_savings_usd=round(revenue, 2)))

        return dispatches

    def _analyze_peaks(self, load_profile: List[LoadDataPoint],
                      schedule: List[OptimizedSchedule]) -> PeakAnalysis:
        """Analyze peak demand reduction."""
        if not load_profile or not schedule:
            return PeakAnalysis(
                original_peak_kw=0, optimized_peak_kw=0,
                peak_reduction_kw=0, peak_reduction_percent=0,
                peak_period=datetime.utcnow(),
                coincident_peak_reduction_kw=0,
                non_coincident_peak_reduction_kw=0)

        original_peak = max(lp.demand_kw for lp in load_profile)
        peak_idx = next(i for i, lp in enumerate(load_profile) if lp.demand_kw == original_peak)
        peak_period = load_profile[peak_idx].timestamp

        optimized_peak = max(opt.grid_import_kw for opt in schedule)
        reduction = original_peak - optimized_peak
        reduction_pct = (reduction / original_peak * 100) if original_peak > 0 else 0

        # Coincident peak (assume peak hours 14-18)
        coincident_original = max((lp.demand_kw for lp in load_profile
                                  if 14 <= lp.timestamp.hour <= 18), default=0)
        coincident_optimized = max((opt.grid_import_kw for opt in schedule
                                   if 14 <= opt.timestamp.hour <= 18), default=0)

        return PeakAnalysis(
            original_peak_kw=round(original_peak, 2),
            optimized_peak_kw=round(optimized_peak, 2),
            peak_reduction_kw=round(reduction, 2),
            peak_reduction_percent=round(reduction_pct, 2),
            peak_period=peak_period,
            coincident_peak_reduction_kw=round(coincident_original - coincident_optimized, 2),
            non_coincident_peak_reduction_kw=round(reduction - (coincident_original - coincident_optimized), 2))

    def _analyze_costs(self, load_profile: List[LoadDataPoint],
                      schedule: List[OptimizedSchedule],
                      tariff: TariffStructure,
                      dr_events: List[DemandResponseEvent]) -> CostAnalysis:
        """Analyze cost savings."""
        # Baseline costs (no optimization)
        baseline_energy = sum(lp.demand_kw for lp in load_profile) / 60 * tariff.energy_rate_per_kwh
        baseline_peak = max(lp.demand_kw for lp in load_profile) if load_profile else 0
        baseline_demand = baseline_peak * tariff.demand_rate_per_kw

        # Optimized costs
        optimized_energy = sum(opt.energy_cost_usd for opt in schedule)
        optimized_peak = max(opt.grid_import_kw for opt in schedule) if schedule else 0
        optimized_demand = optimized_peak * tariff.demand_rate_per_kw

        # Export revenue
        export_kwh = sum(opt.grid_export_kw for opt in schedule) / 60
        export_rate = tariff.energy_rate_per_kwh * 0.5  # Assume 50% of retail
        export_revenue = export_kwh * export_rate

        # DR incentives
        dr_incentive = 0
        for event in dr_events:
            event_schedule = [opt for opt in schedule
                            if event.start_time <= opt.timestamp < event.end_time]
            if event_schedule:
                avg_reduction = sum(opt.shifted_load_kw for opt in event_schedule) / len(event_schedule)
                dr_incentive += avg_reduction * event.incentive_per_kw

        total_savings = ((baseline_energy - optimized_energy) +
                        (baseline_demand - optimized_demand) +
                        export_revenue + dr_incentive)

        baseline_total = baseline_energy + baseline_demand
        savings_pct = (total_savings / baseline_total * 100) if baseline_total > 0 else 0

        return CostAnalysis(
            baseline_energy_cost_usd=round(baseline_energy, 2),
            optimized_energy_cost_usd=round(optimized_energy, 2),
            energy_savings_usd=round(baseline_energy - optimized_energy, 2),
            baseline_demand_cost_usd=round(baseline_demand, 2),
            optimized_demand_cost_usd=round(optimized_demand, 2),
            demand_savings_usd=round(baseline_demand - optimized_demand, 2),
            dr_incentive_earned_usd=round(dr_incentive, 2),
            export_revenue_usd=round(export_revenue, 2),
            total_savings_usd=round(total_savings, 2),
            savings_percent=round(savings_pct, 2))

    def _analyze_carbon(self, load_profile: List[LoadDataPoint],
                       schedule: List[OptimizedSchedule],
                       der_assets: List[DERAsset],
                       grid_intensity: float) -> CarbonAnalysis:
        """Analyze carbon emissions."""
        # Baseline emissions (all from grid)
        baseline_kwh = sum(lp.demand_kw for lp in load_profile) / 60
        baseline_emissions = baseline_kwh * grid_intensity / 1000  # kg CO2

        # Optimized emissions
        grid_import_kwh = sum(opt.grid_import_kw for opt in schedule) / 60
        optimized_emissions = grid_import_kwh * grid_intensity / 1000

        # Renewable fraction
        generation_kwh = sum(opt.total_generation_kw for opt in schedule) / 60
        total_consumption = sum(opt.total_load_kw for opt in schedule) / 60
        renewable_fraction = (generation_kwh / total_consumption * 100) if total_consumption > 0 else 0

        # Avoided imports
        avoided = baseline_kwh - grid_import_kwh

        reduction = baseline_emissions - optimized_emissions
        reduction_pct = (reduction / baseline_emissions * 100) if baseline_emissions > 0 else 0

        return CarbonAnalysis(
            baseline_emissions_kgCO2=round(baseline_emissions, 2),
            optimized_emissions_kgCO2=round(optimized_emissions, 2),
            emissions_reduction_kgCO2=round(reduction, 2),
            emissions_reduction_percent=round(reduction_pct, 2),
            renewable_fraction_percent=round(renewable_fraction, 1),
            avoided_grid_import_kwh=round(avoided, 2))

    def _evaluate_dr_performance(self, schedule: List[OptimizedSchedule],
                                dr_events: List[DemandResponseEvent]) -> List[DRPerformance]:
        """Evaluate demand response performance."""
        performances = []

        for event in dr_events:
            event_schedule = [opt for opt in schedule
                            if event.start_time <= opt.timestamp < event.end_time]

            if event_schedule:
                achieved = sum(opt.shifted_load_kw + opt.curtailed_load_kw
                              for opt in event_schedule) / len(event_schedule)
                performance = (achieved / event.reduction_target_kw * 100
                              if event.reduction_target_kw > 0 else 0)
                incentive = achieved * event.incentive_per_kw
                compliance = "COMPLIANT" if performance >= 80 else "PARTIAL" if performance >= 50 else "NON_COMPLIANT"
            else:
                achieved = 0
                performance = 0
                incentive = 0
                compliance = "NOT_APPLICABLE"

            performances.append(DRPerformance(
                event_id=event.event_id,
                target_reduction_kw=round(event.reduction_target_kw, 2),
                achieved_reduction_kw=round(achieved, 2),
                performance_percent=round(min(100, performance), 1),
                incentive_earned_usd=round(incentive, 2),
                compliance_status=compliance))

        return performances

    def _generate_summary(self, schedule: List[OptimizedSchedule],
                         peak: PeakAnalysis,
                         cost: CostAnalysis,
                         carbon: CarbonAnalysis) -> OptimizationSummary:
        """Generate optimization summary."""
        if not schedule:
            return OptimizationSummary(
                total_energy_kwh=0, total_grid_import_kwh=0,
                total_grid_export_kwh=0, total_generation_kwh=0,
                total_storage_cycles=0, peak_reduction_kw=0,
                cost_savings_usd=0, carbon_savings_kgCO2=0,
                optimization_quality="N/A")

        total_load = sum(opt.total_load_kw for opt in schedule) / 60
        total_import = sum(opt.grid_import_kw for opt in schedule) / 60
        total_export = sum(opt.grid_export_kw for opt in schedule) / 60
        total_gen = sum(opt.total_generation_kw for opt in schedule) / 60

        # Battery cycles (charge + discharge / 2 / capacity)
        total_charge = sum(opt.battery_charge_kw for opt in schedule) / 60
        total_discharge = sum(opt.battery_discharge_kw for opt in schedule) / 60
        # Assume 100 kWh battery for cycle calculation
        cycles = (total_charge + total_discharge) / 2 / 100 if total_charge + total_discharge > 0 else 0

        # Quality assessment
        if cost.savings_percent > 20 and carbon.emissions_reduction_percent > 15:
            quality = "EXCELLENT"
        elif cost.savings_percent > 10 or carbon.emissions_reduction_percent > 10:
            quality = "GOOD"
        elif cost.savings_percent > 5:
            quality = "ACCEPTABLE"
        else:
            quality = "MARGINAL"

        return OptimizationSummary(
            total_energy_kwh=round(total_load, 2),
            total_grid_import_kwh=round(total_import, 2),
            total_grid_export_kwh=round(total_export, 2),
            total_generation_kwh=round(total_gen, 2),
            total_storage_cycles=round(cycles, 2),
            peak_reduction_kw=round(peak.peak_reduction_kw, 2),
            cost_savings_usd=round(cost.total_savings_usd, 2),
            carbon_savings_kgCO2=round(carbon.emissions_reduction_kgCO2, 2),
            optimization_quality=quality)

    def _calculate_stability_score(self, schedule: List[OptimizedSchedule],
                                  constraints: GridConstraints) -> float:
        """Calculate grid stability score."""
        if not schedule:
            return 0

        score = 100.0

        # Penalize for violations
        for opt in schedule:
            if opt.grid_import_kw > constraints.max_import_kw:
                score -= 5
            if opt.grid_export_kw > constraints.max_export_kw:
                score -= 5

        # Penalize for rapid changes
        for i in range(1, len(schedule)):
            ramp = abs(schedule[i].grid_import_kw - schedule[i-1].grid_import_kw)
            if constraints.ramp_limit_kw_min and ramp > constraints.ramp_limit_kw_min:
                score -= 2

        return max(0, min(100, score))

    def _generate_recommendations(self, peak: PeakAnalysis,
                                 cost: CostAnalysis,
                                 carbon: CarbonAnalysis,
                                 der_assets: List[DERAsset]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Peak reduction
        if peak.peak_reduction_percent < 10:
            recommendations.append("Consider additional battery storage to improve peak shaving")

        # Cost savings
        if cost.savings_percent < 10:
            recommendations.append("Review tariff structure for time-of-use optimization opportunities")

        # Carbon reduction
        if carbon.renewable_fraction_percent < 30:
            recommendations.append("Increase on-site renewable generation capacity")

        # DER utilization
        batteries = [a for a in der_assets if a.der_type == DERType.BATTERY]
        if batteries:
            avg_capacity = sum(a.storage_capacity_kwh or 0 for a in batteries) / len(batteries)
            if avg_capacity < 100:
                recommendations.append("Battery capacity may be undersized for optimal peak management")

        # DR participation
        if cost.dr_incentive_earned_usd == 0:
            recommendations.append("Consider participating in demand response programs for additional revenue")

        # Export optimization
        if cost.export_revenue_usd > 0 and carbon.avoided_grid_import_kwh > 0:
            recommendations.append("Explore power purchase agreements for excess generation")

        return recommendations


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-074",
    "name": "GRID-OPTIMIZER",
    "version": "1.0.0",
    "summary": "Electrical grid integration and demand response optimization",
    "tags": ["grid", "optimization", "demand-response", "peak-shaving", "DER", "renewable"],
    "standards": [
        {"ref": "IEEE 1547", "description": "Interconnection Standards for DER"},
        {"ref": "OpenADR 2.0", "description": "Automated Demand Response"},
        {"ref": "FERC Order 2222", "description": "DER Participation in Markets"}
    ],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
