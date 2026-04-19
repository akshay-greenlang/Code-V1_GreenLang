# -*- coding: utf-8 -*-
"""
GL-DECARB-X-002: Marginal Abatement Cost Curve (MACC) Generator Agent
======================================================================

Builds Marginal Abatement Cost (MAC) curves from abatement options data.
Provides visualization-ready data and analysis of decarbonization opportunities.

Capabilities:
    - Generate MAC curves sorted by cost per tCO2e
    - Calculate cumulative abatement potential
    - Identify cost-negative opportunities (left side of curve)
    - Apply filters (sector, TRL, timeline, budget)
    - Support regional cost variations
    - Calculate weighted average cost of abatement
    - Identify breakeven points for target emissions
    - Generate curve data for visualization

Zero-Hallucination Principle:
    All MAC curve data is derived from audited abatement options library.
    Calculations are fully deterministic with complete provenance tracking.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.base_agents import DeterministicAgent, AuditEntry
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism import (
    DeterministicClock,
    content_hash,
    deterministic_id,
)
from greenlang.agents.decarbonization.planning.abatement_options_library import (
    AbatementOption,
    AbatementCategory,
    SectorApplicability,
    TechnologyReadinessLevel,
    CostRange,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class MACCOutputFormat(str, Enum):
    """Output format for MACC data."""
    JSON = "json"
    CSV = "csv"
    CHART_DATA = "chart_data"


class CostMetric(str, Enum):
    """Which cost metric to use for MAC curve."""
    MID = "mid"
    LOW = "low"
    HIGH = "high"
    WEIGHTED = "weighted"


# =============================================================================
# Pydantic Models
# =============================================================================

class MACCDataPoint(BaseModel):
    """Single data point on the MAC curve."""
    option_id: str = Field(..., description="Abatement option ID")
    name: str = Field(..., description="Option name")
    category: str = Field(..., description="Abatement category")

    # Cost data
    cost_per_tco2e: float = Field(..., description="Abatement cost ($/tCO2e)")
    cost_low: float = Field(..., description="Low cost estimate")
    cost_high: float = Field(..., description="High cost estimate")

    # Potential data
    abatement_potential_tco2e: float = Field(..., ge=0, description="Annual abatement potential (tCO2e)")
    cumulative_potential_tco2e: float = Field(..., ge=0, description="Cumulative potential up to this option")

    # For chart rendering
    bar_width: float = Field(..., ge=0, description="Width of bar in chart (= abatement potential)")
    bar_start: float = Field(..., ge=0, description="Starting x position of bar")
    bar_end: float = Field(..., ge=0, description="Ending x position of bar")

    # Metadata
    trl: int = Field(..., ge=1, le=9, description="Technology Readiness Level")
    implementation_months: int = Field(..., ge=0, description="Implementation timeline (months)")
    is_cost_negative: bool = Field(..., description="Whether option has negative cost")


class MACCCurve(BaseModel):
    """Complete MAC curve with data points and statistics."""
    curve_id: str = Field(..., description="Unique curve identifier")
    generated_at: datetime = Field(default_factory=DeterministicClock.now)

    # Data points (sorted by cost ascending)
    data_points: List[MACCDataPoint] = Field(default_factory=list)

    # Summary statistics
    total_options: int = Field(default=0, description="Total number of options")
    total_potential_tco2e: float = Field(default=0.0, ge=0, description="Total abatement potential")
    cost_negative_potential_tco2e: float = Field(default=0.0, ge=0, description="Cost-negative abatement potential")
    weighted_avg_cost: float = Field(default=0.0, description="Weighted average cost ($/tCO2e)")
    median_cost: float = Field(default=0.0, description="Median cost ($/tCO2e)")
    min_cost: float = Field(default=0.0, description="Minimum cost")
    max_cost: float = Field(default=0.0, description="Maximum cost")

    # Breakeven analysis
    breakeven_points: Dict[float, float] = Field(
        default_factory=dict,
        description="Breakeven points: carbon_price -> cumulative_potential"
    )

    # Filters applied
    filters_applied: Dict[str, Any] = Field(default_factory=dict)

    # Provenance
    source_options_hash: str = Field(default="", description="Hash of source options")
    provenance_hash: str = Field(default="", description="Hash of complete curve")


class MACCGeneratorInput(BaseModel):
    """Input model for MACCGeneratorAgent."""
    operation: str = Field(
        default="generate",
        description="Operation: 'generate', 'get_breakeven', 'get_budget_constrained'"
    )

    # Abatement options (required for generate)
    abatement_options: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of abatement options (from AbatementOptionsLibraryAgent)"
    )

    # Filters
    sector: Optional[str] = Field(None, description="Filter by sector")
    category: Optional[str] = Field(None, description="Filter by category")
    min_trl: Optional[int] = Field(None, ge=1, le=9, description="Minimum TRL filter")
    max_implementation_months: Optional[int] = Field(None, ge=0, description="Maximum implementation time")
    region: Optional[str] = Field(None, description="Region for regional costs")

    # Cost metric to use
    cost_metric: CostMetric = Field(default=CostMetric.MID, description="Which cost metric to use")

    # For budget-constrained analysis
    budget_usd: Optional[float] = Field(None, ge=0, description="Budget constraint (USD)")

    # For breakeven analysis
    carbon_prices: List[float] = Field(
        default_factory=lambda: [25, 50, 75, 100, 150, 200],
        description="Carbon prices for breakeven analysis"
    )

    # For target-based analysis
    target_reduction_tco2e: Optional[float] = Field(None, ge=0, description="Target reduction for cost analysis")

    # Output format
    output_format: MACCOutputFormat = Field(default=MACCOutputFormat.JSON)


class MACCGeneratorOutput(BaseModel):
    """Output model for MACCGeneratorAgent."""
    operation: str = Field(..., description="Operation performed")
    success: bool = Field(..., description="Whether operation succeeded")

    # Main result
    macc_curve: Optional[MACCCurve] = Field(None, description="Generated MAC curve")

    # For budget-constrained analysis
    selected_options: List[MACCDataPoint] = Field(
        default_factory=list,
        description="Options selected within budget"
    )
    total_investment_usd: float = Field(default=0.0, description="Total investment required")
    total_abatement_tco2e: float = Field(default=0.0, description="Total abatement achieved")
    average_cost_per_tco2e: float = Field(default=0.0, description="Average cost of selected options")

    # For target analysis
    cost_to_reach_target: Optional[float] = Field(None, description="Total cost to reach target reduction")
    options_needed_for_target: List[str] = Field(
        default_factory=list,
        description="Option IDs needed to reach target"
    )

    # Metadata
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    error_message: Optional[str] = Field(None)


# =============================================================================
# Agent Implementation
# =============================================================================

class MACCGeneratorAgent(DeterministicAgent):
    """
    GL-DECARB-X-002: Marginal Abatement Cost Curve (MACC) Generator Agent

    Builds MAC curves from abatement options data, providing visualization-ready
    data and analysis for decarbonization planning.

    Zero-Hallucination Implementation:
        - All calculations are deterministic with full audit trail
        - No costs or potentials are generated without source data
        - Complete provenance tracking for all curve data
        - Reproducible results for identical inputs

    Attributes:
        config: Agent configuration

    Example:
        >>> agent = MACCGeneratorAgent()
        >>> # Get options from AbatementOptionsLibraryAgent first
        >>> result = agent.run({
        ...     "operation": "generate",
        ...     "abatement_options": options_list,
        ...     "sector": "buildings_commercial"
        ... })
        >>> print(f"Total potential: {result.data['macc_curve']['total_potential_tco2e']} tCO2e")
    """

    AGENT_ID = "GL-DECARB-X-002"
    AGENT_NAME = "MACC Generator Agent"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="MACCGeneratorAgent",
        category=AgentCategory.CRITICAL,
        description="Builds Marginal Abatement Cost Curves from options data"
    )

    def __init__(self, config: Optional[AgentConfig] = None, enable_audit_trail: bool = True):
        """
        Initialize the MACCGeneratorAgent.

        Args:
            config: Optional agent configuration
            enable_audit_trail: Whether to enable audit trail
        """
        super().__init__(enable_audit_trail=enable_audit_trail)

        self.config = config or AgentConfig(
            name=self.AGENT_NAME,
            description="Builds Marginal Abatement Cost Curves",
            version=self.VERSION
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute MACC generation operation.

        Args:
            inputs: Input dictionary with operation and parameters

        Returns:
            Dictionary with MACC results
        """
        start_time = time.time()
        calculation_trace = []

        try:
            # Parse input
            macc_input = MACCGeneratorInput(**inputs)
            calculation_trace.append(f"Operation: {macc_input.operation}")

            # Route to appropriate handler
            if macc_input.operation == "generate":
                result = self._generate_macc(macc_input, calculation_trace)
            elif macc_input.operation == "get_breakeven":
                result = self._get_breakeven_analysis(macc_input, calculation_trace)
            elif macc_input.operation == "get_budget_constrained":
                result = self._get_budget_constrained(macc_input, calculation_trace)
            elif macc_input.operation == "get_target_analysis":
                result = self._get_target_analysis(macc_input, calculation_trace)
            else:
                raise ValueError(f"Unknown operation: {macc_input.operation}")

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            result["processing_time_ms"] = processing_time

            # Capture audit entry
            self._capture_audit_entry(
                operation=macc_input.operation,
                inputs={"operation": macc_input.operation, "options_count": len(macc_input.abatement_options)},
                outputs={"success": result["success"]},
                calculation_trace=calculation_trace
            )

            return result

        except Exception as e:
            self.logger.error(f"MACC generation failed: {str(e)}", exc_info=True)
            processing_time = (time.time() - start_time) * 1000

            return {
                "operation": inputs.get("operation", "unknown"),
                "success": False,
                "error_message": str(e),
                "processing_time_ms": processing_time,
                "timestamp": DeterministicClock.now().isoformat()
            }

    def _generate_macc(
        self,
        macc_input: MACCGeneratorInput,
        calculation_trace: List[str]
    ) -> Dict[str, Any]:
        """Generate MAC curve from abatement options."""
        # Parse options
        options = self._parse_options(macc_input.abatement_options)
        calculation_trace.append(f"Parsed {len(options)} options")

        # Apply filters
        filtered_options = self._apply_filters(options, macc_input, calculation_trace)
        calculation_trace.append(f"After filtering: {len(filtered_options)} options")

        if not filtered_options:
            return {
                "operation": "generate",
                "success": True,
                "macc_curve": MACCCurve(
                    curve_id=deterministic_id({"empty": True}, "macc_"),
                    data_points=[],
                    total_options=0,
                    total_potential_tco2e=0,
                    filters_applied=self._get_filters_dict(macc_input)
                ).model_dump(),
                "timestamp": DeterministicClock.now().isoformat()
            }

        # Sort by cost (ascending)
        def get_cost(opt: Dict[str, Any]) -> float:
            cost_range = opt.get("cost_range", {})
            if macc_input.region and macc_input.region in opt.get("regional_variations", {}):
                cost_range = opt["regional_variations"][macc_input.region]

            if macc_input.cost_metric == CostMetric.LOW:
                return cost_range.get("low", 0)
            elif macc_input.cost_metric == CostMetric.HIGH:
                return cost_range.get("high", 0)
            else:
                return cost_range.get("mid", 0)

        filtered_options.sort(key=get_cost)
        calculation_trace.append("Sorted options by cost (ascending)")

        # Build data points
        data_points = []
        cumulative_potential = 0.0
        cost_negative_potential = 0.0

        for opt in filtered_options:
            cost_range = opt.get("cost_range", {})
            if macc_input.region and macc_input.region in opt.get("regional_variations", {}):
                cost_range = opt["regional_variations"][macc_input.region]

            cost = cost_range.get("mid", 0)
            cost_low = cost_range.get("low", cost)
            cost_high = cost_range.get("high", cost)

            potential = opt.get("reduction_potential", {}).get("reduction_tco2e_per_year", 0)
            bar_start = cumulative_potential
            cumulative_potential += potential
            bar_end = cumulative_potential

            if cost < 0:
                cost_negative_potential += potential

            impl_timeline = opt.get("implementation_timeline", {})
            impl_months = impl_timeline.get("total_months", 0)

            data_point = MACCDataPoint(
                option_id=opt.get("option_id", ""),
                name=opt.get("name", ""),
                category=opt.get("category", ""),
                cost_per_tco2e=cost,
                cost_low=cost_low,
                cost_high=cost_high,
                abatement_potential_tco2e=potential,
                cumulative_potential_tco2e=cumulative_potential,
                bar_width=potential,
                bar_start=bar_start,
                bar_end=bar_end,
                trl=opt.get("trl", 5),
                implementation_months=impl_months,
                is_cost_negative=cost < 0
            )
            data_points.append(data_point)

        calculation_trace.append(f"Built {len(data_points)} data points")
        calculation_trace.append(f"Total potential: {cumulative_potential:.0f} tCO2e")
        calculation_trace.append(f"Cost-negative potential: {cost_negative_potential:.0f} tCO2e")

        # Calculate statistics
        costs = [dp.cost_per_tco2e for dp in data_points]
        potentials = [dp.abatement_potential_tco2e for dp in data_points]

        weighted_avg_cost = (
            sum(c * p for c, p in zip(costs, potentials)) / sum(potentials)
            if sum(potentials) > 0 else 0
        )

        sorted_costs = sorted(costs)
        median_cost = sorted_costs[len(sorted_costs) // 2] if sorted_costs else 0

        # Calculate breakeven points
        breakeven_points = {}
        for carbon_price in macc_input.carbon_prices:
            cumulative = 0.0
            for dp in data_points:
                if dp.cost_per_tco2e <= carbon_price:
                    cumulative = dp.cumulative_potential_tco2e
            breakeven_points[carbon_price] = cumulative

        calculation_trace.append(f"Calculated breakeven points for {len(macc_input.carbon_prices)} prices")

        # Calculate hashes
        source_hash = content_hash({
            "options": [o.get("option_id") for o in filtered_options]
        })

        curve = MACCCurve(
            curve_id=deterministic_id({"source_hash": source_hash}, "macc_"),
            data_points=data_points,
            total_options=len(data_points),
            total_potential_tco2e=cumulative_potential,
            cost_negative_potential_tco2e=cost_negative_potential,
            weighted_avg_cost=weighted_avg_cost,
            median_cost=median_cost,
            min_cost=min(costs) if costs else 0,
            max_cost=max(costs) if costs else 0,
            breakeven_points=breakeven_points,
            filters_applied=self._get_filters_dict(macc_input),
            source_options_hash=source_hash
        )

        curve.provenance_hash = content_hash(curve.model_dump(exclude={"provenance_hash"}))

        return {
            "operation": "generate",
            "success": True,
            "macc_curve": curve.model_dump(),
            "timestamp": DeterministicClock.now().isoformat()
        }

    def _get_breakeven_analysis(
        self,
        macc_input: MACCGeneratorInput,
        calculation_trace: List[str]
    ) -> Dict[str, Any]:
        """Get detailed breakeven analysis at different carbon prices."""
        # First generate the MACC
        macc_result = self._generate_macc(macc_input, calculation_trace)

        if not macc_result["success"]:
            return macc_result

        curve = macc_result["macc_curve"]
        breakeven_points = curve.get("breakeven_points", {})

        # Build detailed analysis
        analysis = []
        for price in sorted(macc_input.carbon_prices):
            potential = breakeven_points.get(price, 0)
            viable_options = [
                dp for dp in curve.get("data_points", [])
                if dp.get("cost_per_tco2e", 0) <= price
            ]

            analysis.append({
                "carbon_price_usd": price,
                "viable_potential_tco2e": potential,
                "viable_options_count": len(viable_options),
                "viable_option_ids": [dp.get("option_id") for dp in viable_options],
                "percentage_of_total": (
                    (potential / curve["total_potential_tco2e"] * 100)
                    if curve["total_potential_tco2e"] > 0 else 0
                )
            })

        calculation_trace.append(f"Completed breakeven analysis for {len(analysis)} price points")

        return {
            "operation": "get_breakeven",
            "success": True,
            "macc_curve": curve,
            "breakeven_analysis": analysis,
            "timestamp": DeterministicClock.now().isoformat()
        }

    def _get_budget_constrained(
        self,
        macc_input: MACCGeneratorInput,
        calculation_trace: List[str]
    ) -> Dict[str, Any]:
        """Select options within budget constraint, maximizing abatement."""
        if macc_input.budget_usd is None:
            raise ValueError("budget_usd required for budget-constrained analysis")

        # First generate the MACC
        macc_result = self._generate_macc(macc_input, calculation_trace)

        if not macc_result["success"]:
            return macc_result

        curve = macc_result["macc_curve"]
        data_points = [MACCDataPoint(**dp) for dp in curve.get("data_points", [])]

        # Select options within budget (greedy by cost efficiency)
        # Sort by cost (already sorted in MACC)
        selected = []
        total_investment = 0.0
        total_abatement = 0.0
        remaining_budget = macc_input.budget_usd

        for dp in data_points:
            # Calculate investment for this option
            # Investment = abatement * cost (if cost is positive)
            # For negative cost options, they generate savings
            option_investment = dp.abatement_potential_tco2e * max(0, dp.cost_per_tco2e)

            if option_investment <= remaining_budget:
                selected.append(dp)
                total_investment += option_investment
                total_abatement += dp.abatement_potential_tco2e
                remaining_budget -= option_investment

        calculation_trace.append(f"Selected {len(selected)} options within ${macc_input.budget_usd:,.0f} budget")
        calculation_trace.append(f"Total investment: ${total_investment:,.0f}")
        calculation_trace.append(f"Total abatement: {total_abatement:,.0f} tCO2e")

        avg_cost = total_investment / total_abatement if total_abatement > 0 else 0

        return {
            "operation": "get_budget_constrained",
            "success": True,
            "macc_curve": curve,
            "selected_options": [dp.model_dump() for dp in selected],
            "total_investment_usd": total_investment,
            "total_abatement_tco2e": total_abatement,
            "average_cost_per_tco2e": avg_cost,
            "budget_utilization": (total_investment / macc_input.budget_usd * 100) if macc_input.budget_usd > 0 else 0,
            "remaining_budget_usd": remaining_budget,
            "timestamp": DeterministicClock.now().isoformat()
        }

    def _get_target_analysis(
        self,
        macc_input: MACCGeneratorInput,
        calculation_trace: List[str]
    ) -> Dict[str, Any]:
        """Analyze cost to reach a specific abatement target."""
        if macc_input.target_reduction_tco2e is None:
            raise ValueError("target_reduction_tco2e required for target analysis")

        # First generate the MACC
        macc_result = self._generate_macc(macc_input, calculation_trace)

        if not macc_result["success"]:
            return macc_result

        curve = macc_result["macc_curve"]
        data_points = [MACCDataPoint(**dp) for dp in curve.get("data_points", [])]

        # Find options needed to reach target
        selected = []
        total_cost = 0.0
        cumulative_abatement = 0.0
        target = macc_input.target_reduction_tco2e

        for dp in data_points:
            if cumulative_abatement >= target:
                break

            selected.append(dp.option_id)

            # Calculate proportional cost for this option
            if cumulative_abatement + dp.abatement_potential_tco2e <= target:
                # Take full option
                option_cost = dp.abatement_potential_tco2e * dp.cost_per_tco2e
                cumulative_abatement += dp.abatement_potential_tco2e
            else:
                # Take partial option (only what's needed)
                needed = target - cumulative_abatement
                option_cost = needed * dp.cost_per_tco2e
                cumulative_abatement = target

            total_cost += option_cost

        calculation_trace.append(f"Target: {target:,.0f} tCO2e")
        calculation_trace.append(f"Options needed: {len(selected)}")
        calculation_trace.append(f"Total cost: ${total_cost:,.0f}")

        achievable = cumulative_abatement >= target * 0.99  # Allow 1% tolerance

        return {
            "operation": "get_target_analysis",
            "success": True,
            "macc_curve": curve,
            "target_reduction_tco2e": target,
            "target_achievable": achievable,
            "achieved_reduction_tco2e": cumulative_abatement,
            "cost_to_reach_target": total_cost,
            "options_needed_for_target": selected,
            "average_cost_per_tco2e": total_cost / cumulative_abatement if cumulative_abatement > 0 else 0,
            "timestamp": DeterministicClock.now().isoformat()
        }

    def _parse_options(self, options_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse options data into consistent format."""
        parsed = []
        for opt in options_data:
            # Handle both dict and AbatementOption formats
            if isinstance(opt, dict):
                parsed.append(opt)
            else:
                # Assume it's a Pydantic model
                parsed.append(opt.model_dump() if hasattr(opt, 'model_dump') else dict(opt))
        return parsed

    def _apply_filters(
        self,
        options: List[Dict[str, Any]],
        macc_input: MACCGeneratorInput,
        calculation_trace: List[str]
    ) -> List[Dict[str, Any]]:
        """Apply filters to options list."""
        filtered = options.copy()

        # Sector filter
        if macc_input.sector:
            filtered = [
                o for o in filtered
                if macc_input.sector in o.get("sectors", [])
            ]
            calculation_trace.append(f"Filtered by sector: {macc_input.sector}")

        # Category filter
        if macc_input.category:
            filtered = [
                o for o in filtered
                if o.get("category") == macc_input.category
            ]
            calculation_trace.append(f"Filtered by category: {macc_input.category}")

        # TRL filter
        if macc_input.min_trl:
            filtered = [
                o for o in filtered
                if o.get("trl", 0) >= macc_input.min_trl
            ]
            calculation_trace.append(f"Filtered by min TRL: {macc_input.min_trl}")

        # Implementation time filter
        if macc_input.max_implementation_months:
            filtered = [
                o for o in filtered
                if o.get("implementation_timeline", {}).get("total_months", 999) <= macc_input.max_implementation_months
            ]
            calculation_trace.append(f"Filtered by max implementation: {macc_input.max_implementation_months} months")

        return filtered

    def _get_filters_dict(self, macc_input: MACCGeneratorInput) -> Dict[str, Any]:
        """Get dictionary of applied filters."""
        filters = {}
        if macc_input.sector:
            filters["sector"] = macc_input.sector
        if macc_input.category:
            filters["category"] = macc_input.category
        if macc_input.min_trl:
            filters["min_trl"] = macc_input.min_trl
        if macc_input.max_implementation_months:
            filters["max_implementation_months"] = macc_input.max_implementation_months
        if macc_input.region:
            filters["region"] = macc_input.region
        filters["cost_metric"] = macc_input.cost_metric.value
        return filters

    # =========================================================================
    # Public API Methods
    # =========================================================================

    def generate_curve(
        self,
        options: List[Dict[str, Any]],
        sector: Optional[str] = None,
        category: Optional[str] = None,
        min_trl: Optional[int] = None,
        region: Optional[str] = None
    ) -> MACCCurve:
        """
        Generate a MAC curve from options.

        Args:
            options: List of abatement options (from AbatementOptionsLibraryAgent)
            sector: Optional sector filter
            category: Optional category filter
            min_trl: Optional minimum TRL filter
            region: Optional region for regional costs

        Returns:
            MACCCurve with data points and statistics
        """
        result = self.execute({
            "operation": "generate",
            "abatement_options": options,
            "sector": sector,
            "category": category,
            "min_trl": min_trl,
            "region": region
        })

        if result["success"]:
            return MACCCurve(**result["macc_curve"])
        else:
            raise ValueError(result.get("error_message", "MACC generation failed"))

    def get_viable_options_at_price(
        self,
        options: List[Dict[str, Any]],
        carbon_price: float
    ) -> Tuple[List[MACCDataPoint], float]:
        """
        Get options viable at a given carbon price.

        Args:
            options: List of abatement options
            carbon_price: Carbon price threshold ($/tCO2e)

        Returns:
            Tuple of (list of viable options, total potential)
        """
        result = self.execute({
            "operation": "generate",
            "abatement_options": options,
            "carbon_prices": [carbon_price]
        })

        if not result["success"]:
            raise ValueError(result.get("error_message", "MACC generation failed"))

        curve = result["macc_curve"]
        data_points = [MACCDataPoint(**dp) for dp in curve.get("data_points", [])]

        viable = [dp for dp in data_points if dp.cost_per_tco2e <= carbon_price]
        total_potential = sum(dp.abatement_potential_tco2e for dp in viable)

        return viable, total_potential
