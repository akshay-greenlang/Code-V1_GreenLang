"""
GreenLang Grid Integration Tools
=================================

Shared tools for grid and utility analysis across all agents.

Tools:
- GridIntegrationTool: Analyze grid capacity, demand charges, TOU rates, DR programs

This tool eliminates duplicate grid analysis code across Phase 3 v3 agents
and provides standardized grid integration metrics.

Author: GreenLang Framework Team
Date: October 2025
Status: Production Ready
"""

from typing import Any, Dict, List, Optional
import numpy as np
from datetime import datetime

from .base import BaseTool, ToolDef, ToolResult, ToolSafety
from greenlang.agents.citations import CalculationCitation


# ==============================================================================
# Grid Integration Tool
# ==============================================================================

class GridIntegrationTool(BaseTool):
    """
    Analyze grid integration and utility cost optimization.

    This tool provides:
    - Grid capacity vs demand analysis
    - Demand charge optimization
    - Time-of-use (TOU) cost analysis
    - Demand response (DR) program benefits
    - Peak shaving opportunities
    - Energy storage integration analysis

    Idempotent (queries grid data) and safe for all agents.
    Eliminates duplicate grid analysis code across Phase 3 v3 agents.

    Example:
        >>> tool = GridIntegrationTool()
        >>> result = tool.execute(
        ...     peak_demand_kw=500,
        ...     load_profile=[450, 420, 400, ...],  # 24 hours
        ...     grid_capacity_kw=600,
        ...     demand_charge_per_kw=15.0,
        ...     energy_rate_per_kwh=0.12,
        ...     tou_rates={"peak": 0.18, "off_peak": 0.08}
        ... )
        >>> print(result.data["monthly_demand_charge"])
        7500.0
    """

    def __init__(self):
        super().__init__(
            name="analyze_grid_integration",
            description="Analyze grid capacity, demand charges, TOU rates, and demand response opportunities for optimal utility cost management",
            safety=ToolSafety.IDEMPOTENT
        )

    def execute(
        self,
        peak_demand_kw: float,
        load_profile: List[float],
        grid_capacity_kw: float,
        demand_charge_per_kw: float,
        energy_rate_per_kwh: float,
        tou_rates: Optional[Dict[str, float]] = None,
        tou_schedule: Optional[Dict[str, List[int]]] = None,
        dr_program_available: bool = False,
        dr_incentive_per_kwh: float = 0.0,
        dr_hours: Optional[List[int]] = None,
        storage_capacity_kwh: float = 0.0,
        storage_power_kw: float = 0.0,
        grid_region: str = "US",
        billing_period_days: int = 30,
    ) -> ToolResult:
        """
        Execute comprehensive grid integration analysis.

        Args:
            peak_demand_kw: Peak demand in kW
            load_profile: Hourly load profile in kW (24 hours or 8760 hours)
            grid_capacity_kw: Available grid capacity in kW
            demand_charge_per_kw: Demand charge in $/kW/month
            energy_rate_per_kwh: Standard energy rate in $/kWh
            tou_rates: Time-of-use rates dict, e.g., {"peak": 0.18, "off_peak": 0.08}
            tou_schedule: TOU schedule dict, e.g., {"peak": [12,13,14,15,16,17,18]}
            dr_program_available: Whether demand response program is available
            dr_incentive_per_kwh: DR incentive in $/kWh for load reduction
            dr_hours: Hours when DR events typically occur (0-23)
            storage_capacity_kwh: Energy storage capacity in kWh
            storage_power_kw: Energy storage power rating in kW
            grid_region: Grid region identifier
            billing_period_days: Billing period in days (default 30)

        Returns:
            ToolResult with grid integration metrics:
                - capacity_utilization: Peak demand / grid capacity (%)
                - capacity_headroom_kw: Available capacity (kW)
                - monthly_demand_charge: Monthly demand charge ($)
                - monthly_energy_cost: Monthly energy cost ($)
                - total_monthly_cost: Total monthly utility cost ($)
                - peak_shaving_opportunity_kw: Potential peak reduction (kW)
                - peak_shaving_savings: Potential savings from peak shaving ($)
                - dr_potential_savings: Potential DR program savings ($)
                - tou_cost_breakdown: Cost by TOU period
                - storage_optimization: Storage dispatch recommendations
        """
        try:
            # Input validation
            if peak_demand_kw < 0:
                return ToolResult(
                    success=False,
                    error="peak_demand_kw must be non-negative"
                )

            if grid_capacity_kw <= 0:
                return ToolResult(
                    success=False,
                    error="grid_capacity_kw must be positive"
                )

            if not load_profile or len(load_profile) == 0:
                return ToolResult(
                    success=False,
                    error="load_profile cannot be empty"
                )

            # Validate load profile length (24 hours or 8760 hours)
            if len(load_profile) not in [24, 8760]:
                return ToolResult(
                    success=False,
                    error="load_profile must be 24 hours or 8760 hours (full year)"
                )

            # Calculate capacity metrics
            capacity_utilization = (peak_demand_kw / grid_capacity_kw) * 100
            capacity_headroom_kw = grid_capacity_kw - peak_demand_kw

            # Determine if at capacity risk
            at_capacity_risk = capacity_utilization > 90

            # Calculate demand charge (monthly)
            monthly_demand_charge = peak_demand_kw * demand_charge_per_kw

            # Calculate energy consumption
            if len(load_profile) == 24:
                # Daily profile - scale to monthly
                daily_energy_kwh = sum(load_profile)
                monthly_energy_kwh = daily_energy_kwh * billing_period_days
            else:
                # Annual profile - scale to monthly
                annual_energy_kwh = sum(load_profile)
                monthly_energy_kwh = annual_energy_kwh / 12

            # Calculate energy costs (with TOU if provided)
            if tou_rates and tou_schedule:
                tou_cost_breakdown = self._calculate_tou_costs(
                    load_profile, tou_rates, tou_schedule, billing_period_days
                )
                monthly_energy_cost = sum(tou_cost_breakdown.values())
            else:
                # Standard flat rate
                monthly_energy_cost = monthly_energy_kwh * energy_rate_per_kwh
                tou_cost_breakdown = {"standard": monthly_energy_cost}

            # Total monthly cost
            total_monthly_cost = monthly_demand_charge + monthly_energy_cost

            # Calculate peak shaving opportunity
            peak_shaving_analysis = self._analyze_peak_shaving(
                load_profile, demand_charge_per_kw, billing_period_days
            )

            # Calculate DR program potential
            dr_analysis = None
            if dr_program_available:
                dr_analysis = self._analyze_demand_response(
                    load_profile,
                    dr_incentive_per_kwh,
                    dr_hours or [14, 15, 16, 17, 18],  # Default peak hours
                    billing_period_days
                )

            # Calculate storage optimization
            storage_analysis = None
            if storage_capacity_kwh > 0 and storage_power_kw > 0:
                storage_analysis = self._analyze_storage_optimization(
                    load_profile,
                    storage_capacity_kwh,
                    storage_power_kw,
                    tou_rates or {},
                    tou_schedule or {},
                    demand_charge_per_kw,
                    billing_period_days
                )

            # Create calculation citations
            citations = [
                CalculationCitation(
                    step_name="calculate_capacity_utilization",
                    formula="capacity_utilization = (peak_demand_kw / grid_capacity_kw) × 100",
                    inputs={
                        "peak_demand_kw": peak_demand_kw,
                        "grid_capacity_kw": grid_capacity_kw,
                    },
                    output={
                        "capacity_utilization": capacity_utilization,
                        "unit": "percent"
                    }
                ),
                CalculationCitation(
                    step_name="calculate_demand_charge",
                    formula="monthly_demand_charge = peak_demand_kw × demand_charge_per_kw",
                    inputs={
                        "peak_demand_kw": peak_demand_kw,
                        "demand_charge_per_kw": demand_charge_per_kw,
                    },
                    output={
                        "monthly_demand_charge": monthly_demand_charge,
                        "unit": "USD"
                    }
                )
            ]

            return ToolResult(
                success=True,
                data={
                    # Capacity metrics
                    "capacity_utilization_percent": round(capacity_utilization, 2),
                    "capacity_headroom_kw": round(capacity_headroom_kw, 2),
                    "at_capacity_risk": at_capacity_risk,
                    "peak_demand_kw": peak_demand_kw,
                    "grid_capacity_kw": grid_capacity_kw,

                    # Cost metrics
                    "monthly_demand_charge": round(monthly_demand_charge, 2),
                    "monthly_energy_cost": round(monthly_energy_cost, 2),
                    "total_monthly_cost": round(total_monthly_cost, 2),
                    "annual_utility_cost": round(total_monthly_cost * 12, 2),

                    # Energy consumption
                    "monthly_energy_kwh": round(monthly_energy_kwh, 2),
                    "average_load_kw": round(sum(load_profile) / len(load_profile), 2),

                    # TOU analysis
                    "tou_cost_breakdown": {
                        period: round(cost, 2)
                        for period, cost in tou_cost_breakdown.items()
                    },

                    # Peak shaving
                    "peak_shaving_opportunity_kw": round(
                        peak_shaving_analysis["opportunity_kw"], 2
                    ),
                    "peak_shaving_potential_savings": round(
                        peak_shaving_analysis["potential_savings"], 2
                    ),
                    "peak_shaving_target_kw": round(
                        peak_shaving_analysis["target_kw"], 2
                    ),

                    # Demand response
                    "dr_available": dr_program_available,
                    "dr_potential_savings": round(
                        dr_analysis["potential_savings"], 2
                    ) if dr_analysis else 0.0,
                    "dr_average_load_reduction_kw": round(
                        dr_analysis["avg_load_reduction_kw"], 2
                    ) if dr_analysis else 0.0,

                    # Storage optimization
                    "storage_enabled": storage_capacity_kwh > 0,
                    "storage_peak_reduction_kw": round(
                        storage_analysis["peak_reduction_kw"], 2
                    ) if storage_analysis else 0.0,
                    "storage_annual_savings": round(
                        storage_analysis["annual_savings"], 2
                    ) if storage_analysis else 0.0,
                    "storage_arbitrage_value": round(
                        storage_analysis["arbitrage_value"], 2
                    ) if storage_analysis else 0.0,
                },
                citations=citations,
                metadata={
                    "grid_region": grid_region,
                    "billing_period_days": billing_period_days,
                    "load_profile_hours": len(load_profile),
                    "tou_enabled": tou_rates is not None,
                    "dr_enabled": dr_program_available,
                    "storage_enabled": storage_capacity_kwh > 0,
                    "summary": (
                        f"Capacity: {capacity_utilization:.1f}%, "
                        f"Monthly Cost: ${total_monthly_cost:,.2f}, "
                        f"Peak Shaving Savings: ${peak_shaving_analysis['potential_savings']:,.2f}"
                    )
                }
            )

        except Exception as e:
            self.logger.error(f"Grid integration analysis failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                error=f"Grid integration analysis failed: {str(e)}"
            )

    def _calculate_tou_costs(
        self,
        load_profile: List[float],
        tou_rates: Dict[str, float],
        tou_schedule: Dict[str, List[int]],
        billing_period_days: int
    ) -> Dict[str, float]:
        """
        Calculate energy costs by TOU period.

        Args:
            load_profile: Hourly load profile
            tou_rates: TOU rates by period
            tou_schedule: TOU schedule (hour assignments)
            billing_period_days: Billing period in days

        Returns:
            Dictionary of costs by TOU period
        """
        costs = {period: 0.0 for period in tou_rates.keys()}

        # Determine if daily or annual profile
        is_daily = len(load_profile) == 24

        if is_daily:
            # Process daily profile
            for hour, load_kw in enumerate(load_profile):
                # Determine TOU period for this hour
                period = self._get_tou_period(hour, tou_schedule)

                # Calculate energy and cost
                energy_kwh = load_kw  # 1 hour
                rate = tou_rates.get(period, tou_rates.get("standard", 0.12))
                costs[period] += energy_kwh * rate

            # Scale to billing period
            for period in costs:
                costs[period] *= billing_period_days
        else:
            # Process annual profile
            for hour, load_kw in enumerate(load_profile):
                hour_of_day = hour % 24

                # Determine TOU period
                period = self._get_tou_period(hour_of_day, tou_schedule)

                # Calculate energy and cost
                energy_kwh = load_kw  # 1 hour
                rate = tou_rates.get(period, tou_rates.get("standard", 0.12))
                costs[period] += energy_kwh * rate

            # Scale to monthly (annual / 12)
            for period in costs:
                costs[period] /= 12

        return costs

    def _get_tou_period(self, hour: int, tou_schedule: Dict[str, List[int]]) -> str:
        """
        Determine TOU period for a given hour.

        Args:
            hour: Hour of day (0-23)
            tou_schedule: TOU schedule dict

        Returns:
            TOU period name
        """
        for period, hours in tou_schedule.items():
            if hour in hours:
                return period

        # Default to off_peak or standard
        return "off_peak" if "off_peak" in tou_schedule else "standard"

    def _analyze_peak_shaving(
        self,
        load_profile: List[float],
        demand_charge_per_kw: float,
        billing_period_days: int
    ) -> Dict[str, float]:
        """
        Analyze peak shaving opportunities.

        Args:
            load_profile: Hourly load profile
            demand_charge_per_kw: Demand charge rate
            billing_period_days: Billing period

        Returns:
            Dictionary with peak shaving analysis
        """
        # Sort loads to find top peaks
        sorted_loads = sorted(load_profile, reverse=True)

        # Current peak
        current_peak = max(load_profile)

        # Target peak (average of top 5% of loads)
        top_5_percent = int(len(sorted_loads) * 0.05)
        if top_5_percent == 0:
            top_5_percent = 1

        target_peak = np.mean(sorted_loads[top_5_percent:top_5_percent*2])

        # Peak shaving opportunity
        opportunity_kw = current_peak - target_peak

        # Potential monthly savings
        potential_savings = opportunity_kw * demand_charge_per_kw

        return {
            "opportunity_kw": max(0, opportunity_kw),
            "potential_savings": max(0, potential_savings),
            "target_kw": target_peak,
            "current_peak_kw": current_peak,
        }

    def _analyze_demand_response(
        self,
        load_profile: List[float],
        dr_incentive_per_kwh: float,
        dr_hours: List[int],
        billing_period_days: int
    ) -> Dict[str, float]:
        """
        Analyze demand response program benefits.

        Args:
            load_profile: Hourly load profile
            dr_incentive_per_kwh: DR incentive rate
            dr_hours: Hours when DR events occur
            billing_period_days: Billing period

        Returns:
            Dictionary with DR analysis
        """
        # Determine if daily or annual profile
        is_daily = len(load_profile) == 24

        # Calculate average load during DR hours
        dr_loads = []

        if is_daily:
            for hour in dr_hours:
                if hour < len(load_profile):
                    dr_loads.append(load_profile[hour])
        else:
            for hour, load_kw in enumerate(load_profile):
                hour_of_day = hour % 24
                if hour_of_day in dr_hours:
                    dr_loads.append(load_kw)

        if not dr_loads:
            return {
                "potential_savings": 0.0,
                "avg_load_reduction_kw": 0.0,
            }

        avg_dr_load = np.mean(dr_loads)

        # Assume 20% load reduction during DR events
        load_reduction_kw = avg_dr_load * 0.20

        # Calculate DR energy reduction
        if is_daily:
            # Assume 10 DR events per month, 2 hours each
            monthly_dr_hours = 10 * 2
            monthly_energy_reduction = load_reduction_kw * monthly_dr_hours
        else:
            # Annual profile - assume 100 DR events per year, 2 hours each
            annual_dr_hours = 100 * 2
            annual_energy_reduction = load_reduction_kw * annual_dr_hours
            monthly_energy_reduction = annual_energy_reduction / 12

        # Calculate incentive value
        potential_savings = monthly_energy_reduction * dr_incentive_per_kwh

        return {
            "potential_savings": potential_savings,
            "avg_load_reduction_kw": load_reduction_kw,
            "monthly_energy_reduction_kwh": monthly_energy_reduction,
        }

    def _analyze_storage_optimization(
        self,
        load_profile: List[float],
        storage_capacity_kwh: float,
        storage_power_kw: float,
        tou_rates: Dict[str, float],
        tou_schedule: Dict[str, List[int]],
        demand_charge_per_kw: float,
        billing_period_days: int
    ) -> Dict[str, float]:
        """
        Analyze energy storage optimization benefits.

        Args:
            load_profile: Hourly load profile
            storage_capacity_kwh: Storage capacity
            storage_power_kw: Storage power rating
            tou_rates: TOU rates
            tou_schedule: TOU schedule
            demand_charge_per_kw: Demand charge rate
            billing_period_days: Billing period

        Returns:
            Dictionary with storage optimization analysis
        """
        # Simplified storage optimization
        # In production, would use more sophisticated optimization (LP, DP)

        current_peak = max(load_profile)

        # Calculate potential peak reduction
        # Storage can reduce peak by up to storage_power_kw
        peak_reduction_kw = min(storage_power_kw, current_peak * 0.15)

        # Demand charge savings
        monthly_demand_savings = peak_reduction_kw * demand_charge_per_kw

        # Arbitrage value (if TOU rates available)
        arbitrage_value = 0.0
        if tou_rates and "peak" in tou_rates and "off_peak" in tou_rates:
            # Calculate arbitrage opportunity
            rate_spread = tou_rates["peak"] - tou_rates["off_peak"]

            # Assume 1 charge/discharge cycle per day
            if len(load_profile) == 24:
                daily_arbitrage = storage_capacity_kwh * rate_spread * 0.85  # 85% round-trip efficiency
                monthly_arbitrage = daily_arbitrage * billing_period_days
            else:
                annual_arbitrage = storage_capacity_kwh * rate_spread * 0.85 * 365
                monthly_arbitrage = annual_arbitrage / 12

            arbitrage_value = monthly_arbitrage

        # Total annual savings
        annual_savings = (monthly_demand_savings + arbitrage_value) * 12

        return {
            "peak_reduction_kw": peak_reduction_kw,
            "monthly_demand_savings": monthly_demand_savings,
            "arbitrage_value": arbitrage_value,
            "annual_savings": annual_savings,
            "payback_years_estimate": None,  # Would require capital cost
        }

    def get_tool_def(self) -> ToolDef:
        """Get tool definition for ChatSession."""
        return ToolDef(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "required": [
                    "peak_demand_kw",
                    "load_profile",
                    "grid_capacity_kw",
                    "demand_charge_per_kw",
                    "energy_rate_per_kwh"
                ],
                "properties": {
                    "peak_demand_kw": {
                        "type": "number",
                        "description": "Peak demand in kW",
                        "minimum": 0
                    },
                    "load_profile": {
                        "type": "array",
                        "description": "Hourly load profile in kW (24 hours or 8760 hours)",
                        "items": {"type": "number", "minimum": 0},
                        "minItems": 24
                    },
                    "grid_capacity_kw": {
                        "type": "number",
                        "description": "Available grid capacity in kW",
                        "minimum": 0
                    },
                    "demand_charge_per_kw": {
                        "type": "number",
                        "description": "Demand charge in $/kW/month",
                        "minimum": 0
                    },
                    "energy_rate_per_kwh": {
                        "type": "number",
                        "description": "Standard energy rate in $/kWh",
                        "minimum": 0
                    },
                    "tou_rates": {
                        "type": "object",
                        "description": "Time-of-use rates by period (e.g., {'peak': 0.18, 'off_peak': 0.08})"
                    },
                    "tou_schedule": {
                        "type": "object",
                        "description": "TOU schedule with hour lists (e.g., {'peak': [12,13,14,15,16,17,18]})"
                    },
                    "dr_program_available": {
                        "type": "boolean",
                        "description": "Whether demand response program is available",
                        "default": False
                    },
                    "dr_incentive_per_kwh": {
                        "type": "number",
                        "description": "DR incentive in $/kWh for load reduction",
                        "default": 0.0,
                        "minimum": 0
                    },
                    "dr_hours": {
                        "type": "array",
                        "description": "Hours when DR events typically occur (0-23)",
                        "items": {"type": "integer", "minimum": 0, "maximum": 23}
                    },
                    "storage_capacity_kwh": {
                        "type": "number",
                        "description": "Energy storage capacity in kWh",
                        "default": 0.0,
                        "minimum": 0
                    },
                    "storage_power_kw": {
                        "type": "number",
                        "description": "Energy storage power rating in kW",
                        "default": 0.0,
                        "minimum": 0
                    },
                    "grid_region": {
                        "type": "string",
                        "description": "Grid region identifier (e.g., 'US-WECC', 'US-ERCOT')",
                        "default": "US"
                    },
                    "billing_period_days": {
                        "type": "integer",
                        "description": "Billing period in days",
                        "default": 30,
                        "minimum": 1,
                        "maximum": 365
                    }
                }
            },
            safety=self.safety
        )
