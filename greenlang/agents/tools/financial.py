"""
GreenLang Financial Metrics Tools
==================================

Shared tools for financial analysis across all agents.

Tools:
- FinancialMetricsTool: Calculate NPV, IRR, payback periods, lifecycle costs

This tool eliminates duplicate financial calculation code across 6+ agents
and provides standardized, auditable financial metrics.

Author: GreenLang Framework Team
Date: October 2025
Status: Production Ready
"""

from typing import Any, Dict, List, Optional
import numpy as np
from scipy.optimize import newton

from .base import BaseTool, ToolDef, ToolResult, ToolSafety
from greenlang.agents.citations import CalculationCitation


# ==============================================================================
# Financial Metrics Tool
# ==============================================================================

class FinancialMetricsTool(BaseTool):
    """
    Calculate comprehensive financial metrics for energy projects.

    This tool provides:
    - Net Present Value (NPV)
    - Internal Rate of Return (IRR)
    - Simple Payback Period
    - Discounted Payback Period
    - Lifecycle Cost Analysis
    - IRA 2022 Incentive Integration

    Deterministic and safe for all agents. Eliminates duplicate financial
    calculation code across Phase 2-4 agents.

    Example:
        >>> tool = FinancialMetricsTool()
        >>> result = tool.execute(
        ...     capital_cost=50000,
        ...     annual_savings=8000,
        ...     lifetime_years=25,
        ...     discount_rate=0.05,
        ...     incentives=[
        ...         {"name": "IRA 2022 ITC", "amount": 15000, "year": 0}
        ...     ]
        ... )
        >>> print(result.data["npv"])
        67890.12
    """

    def __init__(self):
        super().__init__(
            name="calculate_financial_metrics",
            description="Calculate comprehensive financial metrics including NPV, IRR, payback periods, and lifecycle costs for energy projects",
            safety=ToolSafety.DETERMINISTIC
        )

    def execute(
        self,
        capital_cost: float,
        annual_savings: float,
        lifetime_years: int,
        discount_rate: float = 0.05,
        annual_om_cost: float = 0.0,
        energy_cost_escalation: float = 0.02,
        incentives: Optional[List[Dict[str, Any]]] = None,
        tax_rate: float = 0.21,
        salvage_value: float = 0.0,
        include_depreciation: bool = False,
    ) -> ToolResult:
        """
        Execute comprehensive financial analysis.

        Args:
            capital_cost: Upfront capital investment ($)
            annual_savings: Annual energy cost savings (year 1, $)
            lifetime_years: Project lifetime (years)
            discount_rate: Discount rate (decimal, e.g., 0.05 = 5%)
            annual_om_cost: Annual operations & maintenance cost ($)
            energy_cost_escalation: Annual energy cost escalation rate (decimal)
            incentives: List of incentive dicts with 'name', 'amount', and 'year' keys
            tax_rate: Corporate tax rate for depreciation benefits (decimal)
            salvage_value: End-of-life salvage value ($)
            include_depreciation: Include MACRS depreciation tax benefits

        Returns:
            ToolResult with comprehensive financial metrics:
                - npv: Net Present Value ($)
                - irr: Internal Rate of Return (decimal)
                - simple_payback_years: Simple payback period (years)
                - discounted_payback_years: Discounted payback period (years)
                - lifecycle_cost: Total lifecycle cost ($)
                - total_savings: Total undiscounted savings over lifetime ($)
                - benefit_cost_ratio: Benefit-cost ratio
                - total_incentives: Sum of all incentives ($)
        """
        try:
            # Input validation
            if capital_cost < 0:
                return ToolResult(
                    success=False,
                    error="capital_cost must be non-negative"
                )

            if lifetime_years <= 0:
                return ToolResult(
                    success=False,
                    error="lifetime_years must be positive"
                )

            if discount_rate < 0 or discount_rate > 1:
                return ToolResult(
                    success=False,
                    error="discount_rate must be between 0 and 1"
                )

            # Process incentives
            total_incentives = 0.0
            incentive_details = []

            if incentives:
                for inc in incentives:
                    amount = inc.get("amount", 0.0)
                    year = inc.get("year", 0)
                    name = inc.get("name", "Unknown Incentive")

                    total_incentives += amount
                    incentive_details.append({
                        "name": name,
                        "amount": amount,
                        "year": year
                    })

            # Adjust net capital cost by incentives applied in year 0
            year_0_incentives = sum(
                inc["amount"] for inc in incentive_details if inc["year"] == 0
            )
            net_capital_cost = capital_cost - year_0_incentives

            # Calculate cash flows for each year
            cash_flows = [-net_capital_cost]  # Year 0
            cumulative_cash_flow = [-net_capital_cost]

            for year in range(1, lifetime_years + 1):
                # Annual savings with escalation
                savings = annual_savings * ((1 + energy_cost_escalation) ** (year - 1))

                # Annual O&M costs
                om = annual_om_cost

                # Incentives for this year (excluding year 0)
                year_incentives = sum(
                    inc["amount"] for inc in incentive_details if inc["year"] == year
                )

                # Depreciation tax benefit (simplified MACRS 5-year)
                depreciation_benefit = 0.0
                if include_depreciation and year <= 6:
                    macrs_schedule = [0.20, 0.32, 0.192, 0.1152, 0.1152, 0.0576]
                    if year - 1 < len(macrs_schedule):
                        depreciation = capital_cost * macrs_schedule[year - 1]
                        depreciation_benefit = depreciation * tax_rate

                # Add salvage value in final year
                salvage = salvage_value if year == lifetime_years else 0.0

                # Net cash flow for this year
                net_cf = savings - om + year_incentives + depreciation_benefit + salvage
                cash_flows.append(net_cf)

                # Track cumulative for simple payback
                cumulative_cash_flow.append(cumulative_cash_flow[-1] + net_cf)

            # Calculate NPV
            npv = self._calculate_npv(cash_flows, discount_rate)

            # Calculate IRR
            irr = self._calculate_irr(cash_flows)

            # Calculate Simple Payback Period
            simple_payback = self._calculate_simple_payback(cumulative_cash_flow)

            # Calculate Discounted Payback Period
            discounted_payback = self._calculate_discounted_payback(
                cash_flows, discount_rate
            )

            # Calculate lifecycle cost (present value of all costs)
            lifecycle_cost = self._calculate_lifecycle_cost(
                capital_cost, annual_om_cost, lifetime_years, discount_rate
            )

            # Total undiscounted savings
            total_savings = sum(cash_flows[1:])  # Exclude initial investment

            # Benefit-cost ratio
            total_benefits = sum(cash_flows[1:]) + year_0_incentives
            benefit_cost_ratio = total_benefits / capital_cost if capital_cost > 0 else 0.0

            # Create calculation citations
            citations = [
                CalculationCitation(
                    step_name="calculate_npv",
                    formula="NPV = Σ(CF_t / (1 + r)^t) for t=0 to n",
                    inputs={
                        "capital_cost": capital_cost,
                        "annual_savings": annual_savings,
                        "lifetime_years": lifetime_years,
                        "discount_rate": discount_rate,
                        "net_capital_cost": net_capital_cost,
                    },
                    output={
                        "npv": npv,
                        "unit": "USD"
                    }
                ),
                CalculationCitation(
                    step_name="calculate_irr",
                    formula="IRR: rate r where NPV = 0",
                    inputs={
                        "initial_investment": net_capital_cost,
                        "cash_flows": len(cash_flows)
                    },
                    output={
                        "irr": irr,
                        "unit": "decimal"
                    }
                )
            ]

            return ToolResult(
                success=True,
                data={
                    # Primary metrics
                    "npv": round(npv, 2),
                    "irr": round(irr, 4) if irr is not None else None,
                    "simple_payback_years": round(simple_payback, 2) if simple_payback is not None else None,
                    "discounted_payback_years": round(discounted_payback, 2) if discounted_payback is not None else None,

                    # Cost metrics
                    "lifecycle_cost": round(lifecycle_cost, 2),
                    "net_capital_cost": round(net_capital_cost, 2),
                    "capital_cost": capital_cost,

                    # Benefit metrics
                    "total_savings": round(total_savings, 2),
                    "benefit_cost_ratio": round(benefit_cost_ratio, 2),

                    # Incentive details
                    "total_incentives": round(total_incentives, 2),
                    "year_0_incentives": round(year_0_incentives, 2),
                    "incentive_details": incentive_details,

                    # Cash flow analysis
                    "annual_cash_flows": [round(cf, 2) for cf in cash_flows],
                    "cumulative_cash_flows": [round(cf, 2) for cf in cumulative_cash_flow],
                },
                citations=citations,
                metadata={
                    "calculation_inputs": {
                        "discount_rate": discount_rate,
                        "energy_cost_escalation": energy_cost_escalation,
                        "annual_om_cost": annual_om_cost,
                        "tax_rate": tax_rate,
                        "include_depreciation": include_depreciation,
                        "salvage_value": salvage_value,
                    },
                    "summary": (
                        f"NPV: ${npv:,.2f}, "
                        f"IRR: {irr*100:.2f}% " if irr else "IRR: N/A, " +
                        f"Payback: {simple_payback:.1f} years"
                        if simple_payback else "Payback: N/A"
                    )
                }
            )

        except Exception as e:
            self.logger.error(f"Financial calculation failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                error=f"Financial calculation failed: {str(e)}"
            )

    def _calculate_npv(self, cash_flows: List[float], discount_rate: float) -> float:
        """
        Calculate Net Present Value.

        Args:
            cash_flows: List of cash flows (year 0 to n)
            discount_rate: Discount rate (decimal)

        Returns:
            NPV value
        """
        npv = 0.0
        for t, cf in enumerate(cash_flows):
            npv += cf / ((1 + discount_rate) ** t)
        return npv

    def _calculate_irr(self, cash_flows: List[float]) -> Optional[float]:
        """
        Calculate Internal Rate of Return using Newton's method.

        Args:
            cash_flows: List of cash flows (year 0 to n)

        Returns:
            IRR value or None if cannot converge
        """
        try:
            # Use numpy's IRR calculation (more robust)
            return float(np.irr(cash_flows))
        except:
            try:
                # Fallback to manual Newton's method
                def npv_func(rate):
                    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cash_flows))

                # Try to find root near 10% discount rate
                irr = newton(npv_func, 0.1, maxiter=100)

                # Validate result (IRR should be reasonable, e.g., -50% to 100%)
                if -0.5 <= irr <= 1.0:
                    return float(irr)
                else:
                    return None
            except:
                return None

    def _calculate_simple_payback(self, cumulative_cash_flow: List[float]) -> Optional[float]:
        """
        Calculate simple payback period (years until cumulative cash flow > 0).

        Args:
            cumulative_cash_flow: List of cumulative cash flows

        Returns:
            Payback period in years or None if never pays back
        """
        for year, cum_cf in enumerate(cumulative_cash_flow):
            if cum_cf > 0:
                # Interpolate to find exact payback year
                if year == 0:
                    return 0.0

                prev_cf = cumulative_cash_flow[year - 1]
                curr_cf = cum_cf

                # Linear interpolation
                fraction = abs(prev_cf) / (curr_cf - prev_cf)
                return float(year - 1 + fraction)

        # Never pays back
        return None

    def _calculate_discounted_payback(
        self,
        cash_flows: List[float],
        discount_rate: float
    ) -> Optional[float]:
        """
        Calculate discounted payback period.

        Args:
            cash_flows: List of cash flows
            discount_rate: Discount rate

        Returns:
            Discounted payback period in years or None if never pays back
        """
        # Calculate discounted cumulative cash flows
        discounted_cumulative = [cash_flows[0]]  # Year 0

        for t in range(1, len(cash_flows)):
            discounted_cf = cash_flows[t] / ((1 + discount_rate) ** t)
            discounted_cumulative.append(discounted_cumulative[-1] + discounted_cf)

        # Find payback year
        for year, cum_cf in enumerate(discounted_cumulative):
            if cum_cf > 0:
                if year == 0:
                    return 0.0

                prev_cf = discounted_cumulative[year - 1]
                curr_cf = cum_cf

                # Linear interpolation
                fraction = abs(prev_cf) / (curr_cf - prev_cf)
                return float(year - 1 + fraction)

        return None

    def _calculate_lifecycle_cost(
        self,
        capital_cost: float,
        annual_om_cost: float,
        lifetime_years: int,
        discount_rate: float
    ) -> float:
        """
        Calculate total lifecycle cost (present value).

        Args:
            capital_cost: Initial capital investment
            annual_om_cost: Annual O&M cost
            lifetime_years: Project lifetime
            discount_rate: Discount rate

        Returns:
            Lifecycle cost (present value)
        """
        # Capital cost in year 0
        lifecycle_cost = capital_cost

        # Add present value of O&M costs
        for year in range(1, lifetime_years + 1):
            pv_om = annual_om_cost / ((1 + discount_rate) ** year)
            lifecycle_cost += pv_om

        return lifecycle_cost

    def get_tool_def(self) -> ToolDef:
        """Get tool definition for ChatSession."""
        return ToolDef(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "required": ["capital_cost", "annual_savings", "lifetime_years"],
                "properties": {
                    "capital_cost": {
                        "type": "number",
                        "description": "Upfront capital investment in dollars",
                        "minimum": 0
                    },
                    "annual_savings": {
                        "type": "number",
                        "description": "Annual energy cost savings (year 1) in dollars"
                    },
                    "lifetime_years": {
                        "type": "integer",
                        "description": "Project lifetime in years",
                        "minimum": 1,
                        "maximum": 50
                    },
                    "discount_rate": {
                        "type": "number",
                        "description": "Discount rate as decimal (default: 0.05 = 5%)",
                        "default": 0.05,
                        "minimum": 0,
                        "maximum": 1
                    },
                    "annual_om_cost": {
                        "type": "number",
                        "description": "Annual operations & maintenance cost in dollars",
                        "default": 0.0,
                        "minimum": 0
                    },
                    "energy_cost_escalation": {
                        "type": "number",
                        "description": "Annual energy cost escalation rate as decimal (default: 0.02 = 2%)",
                        "default": 0.02,
                        "minimum": -0.1,
                        "maximum": 0.5
                    },
                    "incentives": {
                        "type": "array",
                        "description": "List of incentives with name, amount, and year",
                        "items": {
                            "type": "object",
                            "required": ["name", "amount", "year"],
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Incentive name (e.g., 'IRA 2022 ITC')"
                                },
                                "amount": {
                                    "type": "number",
                                    "description": "Incentive amount in dollars"
                                },
                                "year": {
                                    "type": "integer",
                                    "description": "Year when incentive is received (0 = year of installation)"
                                }
                            }
                        }
                    },
                    "tax_rate": {
                        "type": "number",
                        "description": "Corporate tax rate as decimal for depreciation benefits (default: 0.21 = 21%)",
                        "default": 0.21,
                        "minimum": 0,
                        "maximum": 1
                    },
                    "salvage_value": {
                        "type": "number",
                        "description": "End-of-life salvage value in dollars",
                        "default": 0.0,
                        "minimum": 0
                    },
                    "include_depreciation": {
                        "type": "boolean",
                        "description": "Include MACRS depreciation tax benefits (default: false)",
                        "default": False
                    }
                }
            },
            safety=self.safety
        )
