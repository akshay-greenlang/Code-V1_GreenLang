"""
GL-006 WasteHeatRecovery Agent - Economic Optimizer Module

This module provides comprehensive economic analysis and optimization for
waste heat recovery projects. It implements industry-standard financial
metrics and multi-objective optimization for investment decisions.

Features:
    - Net Present Value (NPV) calculation with sensitivity analysis
    - Internal Rate of Return (IRR) calculation (Newton-Raphson)
    - Modified IRR (MIRR) for reinvestment rate adjustment
    - Payback period (simple and discounted)
    - Levelized Cost of Energy (LCOE)
    - Monte Carlo risk analysis
    - Multi-objective optimization (cost vs. energy)
    - Portfolio optimization for multiple projects
    - Carbon credit valuation
    - Total Cost of Ownership (TCO)

Standards Reference:
    - ASHRAE Handbook: HVAC Applications, Chapter 37 (Owning/Operating Costs)
    - NIST Handbook 135: Life-Cycle Cost Analysis
    - DOE Federal Energy Management Program (FEMP) Guidelines
    - EPRI Technical Assessment Guide (TAG)

Example:
    >>> optimizer = EconomicOptimizer(discount_rate=0.08, project_life_years=20)
    >>> project = WasteHeatProject(
    ...     name="Economizer Install",
    ...     capital_cost=150000,
    ...     annual_savings=45000,
    ... )
    >>> result = optimizer.analyze_project(project)
    >>> print(f"NPV: ${result.npv:,.0f}")
    >>> print(f"IRR: {result.irr_pct:.1f}%")
    >>> print(f"Payback: {result.simple_payback_years:.1f} years")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import logging
import math
import random
import uuid

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Economic defaults
DEFAULT_DISCOUNT_RATE = 0.08  # 8% real discount rate
DEFAULT_INFLATION_RATE = 0.025  # 2.5% general inflation
DEFAULT_ENERGY_ESCALATION_RATE = 0.03  # 3% energy cost escalation
DEFAULT_PROJECT_LIFE_YEARS = 20
DEFAULT_TAX_RATE = 0.21  # 21% federal corporate tax
DEFAULT_DEPRECIATION_YEARS = 7  # MACRS 7-year

# Carbon pricing
DEFAULT_CARBON_PRICE_USD_PER_TON = 51.0  # EPA social cost of carbon 2025
CARBON_PRICE_ESCALATION = 0.02  # 2% annual increase

# Monte Carlo defaults
DEFAULT_MONTE_CARLO_ITERATIONS = 10000


# =============================================================================
# ENUMS
# =============================================================================

class ProjectPhase(Enum):
    """Project implementation phases."""
    CONCEPT = "concept"
    FEASIBILITY = "feasibility"
    ENGINEERING = "engineering"
    PROCUREMENT = "procurement"
    CONSTRUCTION = "construction"
    COMMISSIONING = "commissioning"
    OPERATION = "operation"


class RiskCategory(Enum):
    """Project risk categories."""
    TECHNICAL = "technical"
    FINANCIAL = "financial"
    REGULATORY = "regulatory"
    OPERATIONAL = "operational"
    MARKET = "market"


class DepreciationMethod(Enum):
    """Depreciation calculation methods."""
    STRAIGHT_LINE = "straight_line"
    MACRS_5 = "macrs_5"
    MACRS_7 = "macrs_7"
    MACRS_15 = "macrs_15"


# =============================================================================
# DATA MODELS
# =============================================================================

class EnergyMetrics(BaseModel):
    """Energy savings and emissions metrics."""

    annual_energy_savings_mmbtu: float = Field(
        ...,
        ge=0,
        description="Annual energy savings (MMBTU)"
    )
    annual_electricity_savings_kwh: float = Field(
        default=0.0,
        ge=0,
        description="Annual electricity savings (kWh)"
    )
    annual_fuel_savings_mmbtu: float = Field(
        default=0.0,
        ge=0,
        description="Annual fuel savings (MMBTU)"
    )
    fuel_type: str = Field(
        default="natural_gas",
        description="Primary fuel type displaced"
    )
    co2_reduction_tons_yr: float = Field(
        default=0.0,
        ge=0,
        description="Annual CO2 reduction (metric tons)"
    )
    emission_factor_kg_co2_per_mmbtu: float = Field(
        default=53.06,
        description="Emission factor for displaced fuel"
    )


class CostBreakdown(BaseModel):
    """Detailed cost breakdown."""

    # Capital costs
    equipment_cost_usd: float = Field(default=0.0, ge=0)
    installation_cost_usd: float = Field(default=0.0, ge=0)
    engineering_cost_usd: float = Field(default=0.0, ge=0)
    permitting_cost_usd: float = Field(default=0.0, ge=0)
    contingency_pct: float = Field(default=0.15, ge=0, le=0.5)

    # Operating costs
    annual_maintenance_usd: float = Field(default=0.0, ge=0)
    annual_insurance_usd: float = Field(default=0.0, ge=0)
    annual_labor_usd: float = Field(default=0.0, ge=0)
    annual_consumables_usd: float = Field(default=0.0, ge=0)

    # End of life
    salvage_value_usd: float = Field(default=0.0, ge=0)
    decommissioning_cost_usd: float = Field(default=0.0, ge=0)

    @property
    def total_capital_cost(self) -> float:
        """Calculate total capital cost including contingency."""
        base_capital = (
            self.equipment_cost_usd +
            self.installation_cost_usd +
            self.engineering_cost_usd +
            self.permitting_cost_usd
        )
        return base_capital * (1 + self.contingency_pct)

    @property
    def total_annual_operating_cost(self) -> float:
        """Calculate total annual operating cost."""
        return (
            self.annual_maintenance_usd +
            self.annual_insurance_usd +
            self.annual_labor_usd +
            self.annual_consumables_usd
        )


class IncentivePackage(BaseModel):
    """Financial incentives and tax benefits."""

    federal_tax_credit_pct: float = Field(
        default=0.0,
        ge=0,
        le=0.5,
        description="Federal ITC percentage"
    )
    state_rebate_usd: float = Field(
        default=0.0,
        ge=0,
        description="State/utility rebate"
    )
    utility_incentive_per_mmbtu: float = Field(
        default=0.0,
        ge=0,
        description="Utility incentive per MMBTU saved"
    )
    accelerated_depreciation: bool = Field(
        default=True,
        description="Use accelerated (MACRS) depreciation"
    )
    carbon_credit_eligible: bool = Field(
        default=False,
        description="Eligible for carbon credits"
    )
    carbon_price_usd_per_ton: float = Field(
        default=DEFAULT_CARBON_PRICE_USD_PER_TON,
        ge=0,
        description="Carbon credit price"
    )


class WasteHeatProject(BaseModel):
    """Waste heat recovery project definition."""

    project_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Project identifier"
    )
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(default=None)

    # Capital and operating costs
    capital_cost_usd: float = Field(
        ...,
        ge=0,
        description="Total capital cost"
    )
    annual_operating_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Annual O&M cost"
    )
    cost_breakdown: Optional[CostBreakdown] = Field(
        default=None,
        description="Detailed cost breakdown"
    )

    # Revenue/savings
    annual_energy_savings_usd: float = Field(
        ...,
        ge=0,
        description="Annual energy cost savings"
    )
    annual_other_revenue_usd: float = Field(
        default=0.0,
        ge=0,
        description="Other annual revenue"
    )

    # Energy metrics
    energy_metrics: Optional[EnergyMetrics] = Field(default=None)

    # Incentives
    incentives: Optional[IncentivePackage] = Field(default=None)

    # Timeline
    construction_months: int = Field(
        default=6,
        ge=1,
        le=48,
        description="Construction period"
    )
    project_life_years: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Project economic life"
    )

    # Risk
    risk_factors: Optional[Dict[str, float]] = Field(
        default=None,
        description="Risk factors for sensitivity"
    )

    @property
    def annual_net_savings(self) -> float:
        """Calculate annual net savings."""
        return (
            self.annual_energy_savings_usd +
            self.annual_other_revenue_usd -
            self.annual_operating_cost_usd
        )


class YearlyProjection(BaseModel):
    """Year-by-year financial projection."""

    year: int = Field(...)
    gross_savings_usd: float = Field(...)
    operating_cost_usd: float = Field(...)
    net_cash_flow_usd: float = Field(...)
    depreciation_usd: float = Field(default=0.0)
    tax_benefit_usd: float = Field(default=0.0)
    after_tax_cash_flow_usd: float = Field(default=0.0)
    cumulative_cash_flow_usd: float = Field(...)
    discounted_cash_flow_usd: float = Field(...)
    cumulative_npv_usd: float = Field(...)


class EconomicAnalysisResult(BaseModel):
    """Complete economic analysis result."""

    analysis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    project_name: str = Field(...)

    # Key metrics
    npv_usd: float = Field(..., description="Net Present Value")
    irr_pct: float = Field(..., description="Internal Rate of Return (%)")
    mirr_pct: Optional[float] = Field(default=None, description="Modified IRR (%)")
    simple_payback_years: float = Field(..., description="Simple payback period")
    discounted_payback_years: Optional[float] = Field(
        default=None,
        description="Discounted payback period"
    )
    roi_pct: float = Field(..., description="Return on Investment (%)")
    savings_investment_ratio: float = Field(
        ...,
        description="Savings-to-Investment Ratio"
    )

    # Levelized metrics
    lcoe_usd_per_mmbtu: Optional[float] = Field(
        default=None,
        description="Levelized Cost of Energy"
    )

    # Carbon
    carbon_benefit_usd: float = Field(
        default=0.0,
        description="Total carbon credit value"
    )
    carbon_cost_effectiveness_usd_per_ton: Optional[float] = Field(
        default=None,
        description="Cost per ton CO2 avoided"
    )

    # Total value
    total_lifetime_savings_usd: float = Field(...)
    total_lifetime_costs_usd: float = Field(...)
    net_benefit_usd: float = Field(...)

    # Projections
    yearly_projections: List[YearlyProjection] = Field(default_factory=list)

    # Sensitivity analysis
    sensitivity_results: Optional[Dict[str, Any]] = Field(default=None)

    # Monte Carlo
    monte_carlo_results: Optional[Dict[str, Any]] = Field(default=None)

    # Ranking
    economic_ranking: str = Field(
        default="",
        description="Project economic ranking"
    )
    recommendation: str = Field(
        default="",
        description="Implementation recommendation"
    )

    # Analysis parameters
    discount_rate_used: float = Field(...)
    inflation_rate_used: float = Field(...)
    energy_escalation_used: float = Field(...)
    project_life_years: int = Field(...)

    # Provenance
    provenance_hash: str = Field(default="")


class PortfolioAnalysisResult(BaseModel):
    """Portfolio optimization result for multiple projects."""

    analysis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    # Portfolio metrics
    total_investment_usd: float = Field(...)
    portfolio_npv_usd: float = Field(...)
    portfolio_irr_pct: Optional[float] = Field(default=None)
    portfolio_payback_years: float = Field(...)

    # Project rankings
    project_rankings: List[Dict[str, Any]] = Field(default_factory=list)

    # Budget-constrained selection
    selected_projects: List[str] = Field(default_factory=list)
    budget_utilization_pct: float = Field(default=0.0)

    # Implementation phases
    implementation_schedule: List[Dict[str, Any]] = Field(default_factory=list)

    # Aggregate benefits
    total_energy_savings_mmbtu_yr: float = Field(default=0.0)
    total_co2_reduction_tons_yr: float = Field(default=0.0)
    total_annual_savings_usd: float = Field(default=0.0)


# =============================================================================
# ECONOMIC OPTIMIZER CLASS
# =============================================================================

class EconomicOptimizer:
    """
    Comprehensive Economic Optimizer for Waste Heat Recovery Projects.

    Provides industry-standard financial analysis including NPV, IRR,
    payback, sensitivity analysis, and portfolio optimization.

    Attributes:
        discount_rate: Real discount rate (nominal - inflation)
        inflation_rate: General inflation rate
        energy_escalation: Energy cost escalation rate
        tax_rate: Corporate tax rate
        project_life: Default project life in years

    Example:
        >>> optimizer = EconomicOptimizer(discount_rate=0.08)
        >>> result = optimizer.analyze_project(project)
        >>> print(f"NPV: ${result.npv_usd:,.0f}")
    """

    def __init__(
        self,
        discount_rate: float = DEFAULT_DISCOUNT_RATE,
        inflation_rate: float = DEFAULT_INFLATION_RATE,
        energy_escalation_rate: float = DEFAULT_ENERGY_ESCALATION_RATE,
        tax_rate: float = DEFAULT_TAX_RATE,
        project_life_years: int = DEFAULT_PROJECT_LIFE_YEARS,
    ) -> None:
        """
        Initialize the Economic Optimizer.

        Args:
            discount_rate: Real discount rate (default 8%)
            inflation_rate: General inflation rate (default 2.5%)
            energy_escalation_rate: Energy cost escalation (default 3%)
            tax_rate: Corporate tax rate (default 21%)
            project_life_years: Default project life (default 20 years)
        """
        self.discount_rate = discount_rate
        self.inflation_rate = inflation_rate
        self.energy_escalation = energy_escalation_rate
        self.tax_rate = tax_rate
        self.project_life = project_life_years

        logger.info(
            f"EconomicOptimizer initialized: r={discount_rate*100:.1f}%, "
            f"life={project_life_years} years"
        )

    def analyze_project(
        self,
        project: WasteHeatProject,
        include_sensitivity: bool = True,
        include_monte_carlo: bool = False,
        monte_carlo_iterations: int = DEFAULT_MONTE_CARLO_ITERATIONS,
    ) -> EconomicAnalysisResult:
        """
        Perform comprehensive economic analysis of a waste heat project.

        Args:
            project: Project definition
            include_sensitivity: Include sensitivity analysis
            include_monte_carlo: Include Monte Carlo risk analysis
            monte_carlo_iterations: Number of Monte Carlo iterations

        Returns:
            Complete economic analysis results
        """
        logger.info(f"Analyzing project: {project.name}")

        project_life = project.project_life_years or self.project_life

        # Calculate base case cash flows
        yearly_projections = self._calculate_yearly_projections(
            project, project_life
        )

        # Calculate key metrics
        npv = self._calculate_npv(yearly_projections, project.capital_cost_usd)
        irr = self._calculate_irr(
            project.capital_cost_usd,
            [p.net_cash_flow_usd for p in yearly_projections],
        )
        mirr = self._calculate_mirr(
            project.capital_cost_usd,
            [p.net_cash_flow_usd for p in yearly_projections],
        )

        simple_payback = self._calculate_simple_payback(
            project.capital_cost_usd,
            project.annual_net_savings,
        )
        discounted_payback = self._calculate_discounted_payback(
            yearly_projections, project.capital_cost_usd
        )

        # ROI and SIR
        total_savings = sum(p.gross_savings_usd for p in yearly_projections)
        total_costs = project.capital_cost_usd + sum(
            p.operating_cost_usd for p in yearly_projections
        )
        roi = ((total_savings - total_costs) / project.capital_cost_usd) * 100
        sir = total_savings / project.capital_cost_usd

        # LCOE calculation
        lcoe = None
        if project.energy_metrics:
            lcoe = self._calculate_lcoe(
                project.capital_cost_usd,
                [p.operating_cost_usd for p in yearly_projections],
                project.energy_metrics.annual_energy_savings_mmbtu,
                project_life,
            )

        # Carbon economics
        carbon_benefit = 0.0
        carbon_cost_effectiveness = None
        if project.energy_metrics and project.energy_metrics.co2_reduction_tons_yr > 0:
            carbon_price = (
                project.incentives.carbon_price_usd_per_ton
                if project.incentives else DEFAULT_CARBON_PRICE_USD_PER_TON
            )
            carbon_benefit = self._calculate_carbon_benefit(
                project.energy_metrics.co2_reduction_tons_yr,
                carbon_price,
                project_life,
            )
            carbon_cost_effectiveness = (
                project.capital_cost_usd /
                (project.energy_metrics.co2_reduction_tons_yr * project_life)
            )

        # Sensitivity analysis
        sensitivity_results = None
        if include_sensitivity:
            sensitivity_results = self._sensitivity_analysis(project)

        # Monte Carlo
        monte_carlo_results = None
        if include_monte_carlo:
            monte_carlo_results = self._monte_carlo_analysis(
                project, monte_carlo_iterations
            )

        # Generate ranking and recommendation
        ranking, recommendation = self._generate_recommendation(
            npv, irr, simple_payback, sir
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance(project)

        result = EconomicAnalysisResult(
            project_name=project.name,
            npv_usd=round(npv, 0),
            irr_pct=round(irr * 100, 2),
            mirr_pct=round(mirr * 100, 2) if mirr else None,
            simple_payback_years=round(simple_payback, 2),
            discounted_payback_years=round(discounted_payback, 2) if discounted_payback else None,
            roi_pct=round(roi, 2),
            savings_investment_ratio=round(sir, 3),
            lcoe_usd_per_mmbtu=round(lcoe, 2) if lcoe else None,
            carbon_benefit_usd=round(carbon_benefit, 0),
            carbon_cost_effectiveness_usd_per_ton=round(carbon_cost_effectiveness, 2) if carbon_cost_effectiveness else None,
            total_lifetime_savings_usd=round(total_savings, 0),
            total_lifetime_costs_usd=round(total_costs, 0),
            net_benefit_usd=round(total_savings - total_costs, 0),
            yearly_projections=yearly_projections,
            sensitivity_results=sensitivity_results,
            monte_carlo_results=monte_carlo_results,
            economic_ranking=ranking,
            recommendation=recommendation,
            discount_rate_used=self.discount_rate,
            inflation_rate_used=self.inflation_rate,
            energy_escalation_used=self.energy_escalation,
            project_life_years=project_life,
            provenance_hash=provenance_hash,
        )

        logger.info(
            f"Analysis complete: NPV=${npv:,.0f}, IRR={irr*100:.1f}%, "
            f"Payback={simple_payback:.1f} years"
        )

        return result

    def _calculate_yearly_projections(
        self,
        project: WasteHeatProject,
        project_life: int,
    ) -> List[YearlyProjection]:
        """Calculate year-by-year financial projections."""
        projections = []
        cumulative_cash_flow = -project.capital_cost_usd
        cumulative_npv = -project.capital_cost_usd

        # Calculate incentive value
        incentive_credit = 0.0
        if project.incentives:
            incentive_credit += project.capital_cost_usd * project.incentives.federal_tax_credit_pct
            incentive_credit += project.incentives.state_rebate_usd

        # Get depreciation schedule
        depreciation_schedule = self._get_depreciation_schedule(
            project.capital_cost_usd - incentive_credit,
            project_life,
        )

        for year in range(1, project_life + 1):
            # Escalate savings and costs
            savings_escalation = (1 + self.energy_escalation) ** (year - 1)
            cost_escalation = (1 + self.inflation_rate) ** (year - 1)

            gross_savings = project.annual_energy_savings_usd * savings_escalation
            gross_savings += project.annual_other_revenue_usd * cost_escalation

            # Add carbon credits if applicable
            if project.incentives and project.incentives.carbon_credit_eligible:
                if project.energy_metrics:
                    carbon_escalation = (1 + CARBON_PRICE_ESCALATION) ** (year - 1)
                    carbon_revenue = (
                        project.energy_metrics.co2_reduction_tons_yr *
                        project.incentives.carbon_price_usd_per_ton *
                        carbon_escalation
                    )
                    gross_savings += carbon_revenue

            operating_cost = project.annual_operating_cost_usd * cost_escalation

            net_cash_flow = gross_savings - operating_cost

            # Year 1: add incentive credit
            if year == 1:
                net_cash_flow += incentive_credit

            # Depreciation and tax effects
            depreciation = depreciation_schedule.get(year, 0)
            taxable_income = net_cash_flow - depreciation
            tax_benefit = 0.0
            if taxable_income < 0:  # Tax shield
                tax_benefit = abs(taxable_income) * self.tax_rate

            after_tax_cf = net_cash_flow + tax_benefit

            # Cumulative and discounted
            cumulative_cash_flow += net_cash_flow
            discount_factor = (1 + self.discount_rate) ** year
            discounted_cf = net_cash_flow / discount_factor
            cumulative_npv += discounted_cf

            projections.append(YearlyProjection(
                year=year,
                gross_savings_usd=round(gross_savings, 0),
                operating_cost_usd=round(operating_cost, 0),
                net_cash_flow_usd=round(net_cash_flow, 0),
                depreciation_usd=round(depreciation, 0),
                tax_benefit_usd=round(tax_benefit, 0),
                after_tax_cash_flow_usd=round(after_tax_cf, 0),
                cumulative_cash_flow_usd=round(cumulative_cash_flow, 0),
                discounted_cash_flow_usd=round(discounted_cf, 0),
                cumulative_npv_usd=round(cumulative_npv, 0),
            ))

        return projections

    def _get_depreciation_schedule(
        self,
        depreciable_basis: float,
        project_life: int,
        method: DepreciationMethod = DepreciationMethod.MACRS_7,
    ) -> Dict[int, float]:
        """Get depreciation schedule."""
        schedule = {}

        if method == DepreciationMethod.STRAIGHT_LINE:
            annual_depreciation = depreciable_basis / project_life
            for year in range(1, project_life + 1):
                schedule[year] = annual_depreciation

        elif method == DepreciationMethod.MACRS_7:
            # MACRS 7-year percentages
            macrs_rates = [0.1429, 0.2449, 0.1749, 0.1249, 0.0893, 0.0892, 0.0893, 0.0446]
            for i, rate in enumerate(macrs_rates):
                schedule[i + 1] = depreciable_basis * rate

        elif method == DepreciationMethod.MACRS_5:
            macrs_rates = [0.20, 0.32, 0.192, 0.1152, 0.1152, 0.0576]
            for i, rate in enumerate(macrs_rates):
                schedule[i + 1] = depreciable_basis * rate

        elif method == DepreciationMethod.MACRS_15:
            macrs_rates = [0.05, 0.095, 0.0855, 0.077, 0.0693, 0.0623,
                           0.0590, 0.0590, 0.0591, 0.0590, 0.0591, 0.0590,
                           0.0591, 0.0590, 0.0591, 0.0295]
            for i, rate in enumerate(macrs_rates):
                schedule[i + 1] = depreciable_basis * rate

        return schedule

    def _calculate_npv(
        self,
        projections: List[YearlyProjection],
        initial_investment: float,
    ) -> float:
        """Calculate Net Present Value."""
        npv = -initial_investment
        for proj in projections:
            npv += proj.discounted_cash_flow_usd
        return npv

    def _calculate_irr(
        self,
        initial_investment: float,
        annual_cash_flows: List[float],
        max_iterations: int = 100,
        tolerance: float = 0.0001,
    ) -> float:
        """
        Calculate Internal Rate of Return using Newton-Raphson method.

        IRR is the discount rate that makes NPV = 0.
        """
        # Initial guess
        irr = 0.10

        for iteration in range(max_iterations):
            # Calculate NPV and derivative at current IRR
            npv = -initial_investment
            d_npv = 0.0

            for t, cf in enumerate(annual_cash_flows, 1):
                factor = (1 + irr) ** t
                npv += cf / factor
                d_npv -= t * cf / (factor * (1 + irr))

            # Check convergence
            if abs(npv) < tolerance:
                return irr

            # Newton-Raphson update
            if abs(d_npv) > 1e-10:
                irr = irr - npv / d_npv
            else:
                break

            # Bound IRR to reasonable range
            irr = max(-0.99, min(10.0, irr))

        # If no convergence, estimate from payback
        logger.warning("IRR did not converge, using approximation")
        avg_cf = sum(annual_cash_flows) / len(annual_cash_flows)
        if avg_cf > 0:
            return (avg_cf / initial_investment) - 0.05
        return 0.0

    def _calculate_mirr(
        self,
        initial_investment: float,
        annual_cash_flows: List[float],
        reinvestment_rate: Optional[float] = None,
        finance_rate: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate Modified Internal Rate of Return.

        MIRR assumes reinvestment at a specified rate rather than IRR.
        """
        if reinvestment_rate is None:
            reinvestment_rate = self.discount_rate
        if finance_rate is None:
            finance_rate = self.discount_rate

        n = len(annual_cash_flows)

        # Future value of positive cash flows (reinvested)
        fv_positive = 0.0
        for t, cf in enumerate(annual_cash_flows):
            if cf > 0:
                fv_positive += cf * ((1 + reinvestment_rate) ** (n - t))

        # Present value of negative cash flows (financed)
        pv_negative = initial_investment
        for t, cf in enumerate(annual_cash_flows, 1):
            if cf < 0:
                pv_negative += abs(cf) / ((1 + finance_rate) ** t)

        if pv_negative <= 0 or fv_positive <= 0:
            return None

        # MIRR = (FV_positive / PV_negative)^(1/n) - 1
        mirr = (fv_positive / pv_negative) ** (1 / n) - 1
        return mirr

    def _calculate_simple_payback(
        self,
        initial_investment: float,
        annual_savings: float,
    ) -> float:
        """Calculate simple payback period in years."""
        if annual_savings <= 0:
            return float('inf')
        return initial_investment / annual_savings

    def _calculate_discounted_payback(
        self,
        projections: List[YearlyProjection],
        initial_investment: float,
    ) -> Optional[float]:
        """Calculate discounted payback period."""
        cumulative = -initial_investment

        for proj in projections:
            cumulative += proj.discounted_cash_flow_usd
            if cumulative >= 0:
                # Interpolate for fractional year
                if proj.discounted_cash_flow_usd > 0:
                    fraction = (
                        (cumulative - proj.discounted_cash_flow_usd + initial_investment) /
                        proj.discounted_cash_flow_usd
                    )
                    return proj.year - 1 + (1 - fraction)
                return float(proj.year)

        return None  # Never pays back

    def _calculate_lcoe(
        self,
        capital_cost: float,
        operating_costs: List[float],
        annual_energy_mmbtu: float,
        project_life: int,
    ) -> float:
        """
        Calculate Levelized Cost of Energy.

        LCOE = Sum(Costs_t / (1+r)^t) / Sum(Energy_t / (1+r)^t)
        """
        pv_costs = capital_cost
        pv_energy = 0.0

        for t, oc in enumerate(operating_costs, 1):
            discount_factor = (1 + self.discount_rate) ** t
            pv_costs += oc / discount_factor
            pv_energy += annual_energy_mmbtu / discount_factor

        if pv_energy <= 0:
            return 0.0

        return pv_costs / pv_energy

    def _calculate_carbon_benefit(
        self,
        annual_co2_reduction: float,
        carbon_price: float,
        project_life: int,
    ) -> float:
        """Calculate total carbon credit value over project life."""
        total_benefit = 0.0

        for year in range(1, project_life + 1):
            escalated_price = carbon_price * ((1 + CARBON_PRICE_ESCALATION) ** (year - 1))
            discount_factor = (1 + self.discount_rate) ** year
            total_benefit += (annual_co2_reduction * escalated_price) / discount_factor

        return total_benefit

    def _sensitivity_analysis(
        self,
        project: WasteHeatProject,
    ) -> Dict[str, Any]:
        """Perform sensitivity analysis on key parameters."""
        base_npv = self._quick_npv(project)

        sensitivities = {}

        # Parameters to test: +/- 20%
        test_range = [-0.20, -0.10, 0.0, 0.10, 0.20]

        # Capital cost sensitivity
        capital_sensitivity = []
        for delta in test_range:
            modified = project.copy(deep=True)
            modified.capital_cost_usd *= (1 + delta)
            npv = self._quick_npv(modified)
            capital_sensitivity.append({
                "delta_pct": delta * 100,
                "npv": round(npv, 0),
                "npv_change_pct": round((npv - base_npv) / abs(base_npv) * 100, 1) if base_npv != 0 else 0,
            })
        sensitivities["capital_cost"] = capital_sensitivity

        # Energy savings sensitivity
        savings_sensitivity = []
        for delta in test_range:
            modified = project.copy(deep=True)
            modified.annual_energy_savings_usd *= (1 + delta)
            npv = self._quick_npv(modified)
            savings_sensitivity.append({
                "delta_pct": delta * 100,
                "npv": round(npv, 0),
                "npv_change_pct": round((npv - base_npv) / abs(base_npv) * 100, 1) if base_npv != 0 else 0,
            })
        sensitivities["energy_savings"] = savings_sensitivity

        # Discount rate sensitivity
        rate_sensitivity = []
        original_rate = self.discount_rate
        for delta in test_range:
            self.discount_rate = original_rate * (1 + delta)
            npv = self._quick_npv(project)
            rate_sensitivity.append({
                "discount_rate_pct": round(self.discount_rate * 100, 1),
                "npv": round(npv, 0),
                "npv_change_pct": round((npv - base_npv) / abs(base_npv) * 100, 1) if base_npv != 0 else 0,
            })
        self.discount_rate = original_rate
        sensitivities["discount_rate"] = rate_sensitivity

        # Find most sensitive parameter
        capital_impact = abs(capital_sensitivity[-1]["npv"] - capital_sensitivity[0]["npv"])
        savings_impact = abs(savings_sensitivity[-1]["npv"] - savings_sensitivity[0]["npv"])
        rate_impact = abs(rate_sensitivity[-1]["npv"] - rate_sensitivity[0]["npv"])

        if capital_impact >= savings_impact and capital_impact >= rate_impact:
            most_sensitive = "capital_cost"
        elif savings_impact >= rate_impact:
            most_sensitive = "energy_savings"
        else:
            most_sensitive = "discount_rate"

        sensitivities["most_sensitive_parameter"] = most_sensitive

        return sensitivities

    def _monte_carlo_analysis(
        self,
        project: WasteHeatProject,
        iterations: int,
    ) -> Dict[str, Any]:
        """Perform Monte Carlo risk analysis."""
        npv_results = []
        irr_results = []
        payback_results = []

        # Define uncertainty ranges
        capital_std = 0.15  # 15% standard deviation
        savings_std = 0.20  # 20% standard deviation
        opex_std = 0.10     # 10% standard deviation

        for i in range(iterations):
            # Sample from distributions
            capital_factor = max(0.5, random.gauss(1.0, capital_std))
            savings_factor = max(0.3, random.gauss(1.0, savings_std))
            opex_factor = max(0.5, random.gauss(1.0, opex_std))

            # Create modified project
            modified = project.copy(deep=True)
            modified.capital_cost_usd *= capital_factor
            modified.annual_energy_savings_usd *= savings_factor
            modified.annual_operating_cost_usd *= opex_factor

            # Calculate metrics
            npv = self._quick_npv(modified)
            npv_results.append(npv)

            payback = self._calculate_simple_payback(
                modified.capital_cost_usd,
                modified.annual_net_savings,
            )
            payback_results.append(payback)

        # Statistical summary
        npv_results.sort()
        payback_results.sort()

        def percentile(data, pct):
            idx = int(len(data) * pct / 100)
            return data[min(idx, len(data) - 1)]

        return {
            "iterations": iterations,
            "npv_statistics": {
                "mean": round(sum(npv_results) / len(npv_results), 0),
                "std_dev": round(self._std_dev(npv_results), 0),
                "min": round(min(npv_results), 0),
                "max": round(max(npv_results), 0),
                "p10": round(percentile(npv_results, 10), 0),
                "p50": round(percentile(npv_results, 50), 0),
                "p90": round(percentile(npv_results, 90), 0),
            },
            "payback_statistics": {
                "mean": round(sum(payback_results) / len(payback_results), 2),
                "p10": round(percentile(payback_results, 10), 2),
                "p90": round(percentile(payback_results, 90), 2),
            },
            "probability_npv_positive": round(
                len([n for n in npv_results if n > 0]) / len(npv_results) * 100, 1
            ),
            "probability_payback_under_3_years": round(
                len([p for p in payback_results if p < 3]) / len(payback_results) * 100, 1
            ),
        }

    def _std_dev(self, data: List[float]) -> float:
        """Calculate standard deviation."""
        if len(data) < 2:
            return 0.0
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        return math.sqrt(variance)

    def _quick_npv(self, project: WasteHeatProject) -> float:
        """Quick NPV calculation without full projections."""
        npv = -project.capital_cost_usd
        project_life = project.project_life_years or self.project_life

        for year in range(1, project_life + 1):
            savings_factor = (1 + self.energy_escalation) ** (year - 1)
            cost_factor = (1 + self.inflation_rate) ** (year - 1)

            net_cf = (
                project.annual_energy_savings_usd * savings_factor -
                project.annual_operating_cost_usd * cost_factor
            )
            npv += net_cf / ((1 + self.discount_rate) ** year)

        return npv

    def _generate_recommendation(
        self,
        npv: float,
        irr: float,
        payback: float,
        sir: float,
    ) -> Tuple[str, str]:
        """Generate project ranking and recommendation."""
        # Scoring
        score = 0

        # NPV scoring
        if npv > 500000:
            score += 4
        elif npv > 100000:
            score += 3
        elif npv > 0:
            score += 2
        elif npv > -50000:
            score += 1

        # IRR scoring
        if irr > 0.30:
            score += 4
        elif irr > 0.20:
            score += 3
        elif irr > 0.10:
            score += 2
        elif irr > 0.05:
            score += 1

        # Payback scoring
        if payback < 2:
            score += 4
        elif payback < 3:
            score += 3
        elif payback < 5:
            score += 2
        elif payback < 7:
            score += 1

        # SIR scoring
        if sir > 2.0:
            score += 4
        elif sir > 1.5:
            score += 3
        elif sir > 1.0:
            score += 2

        # Determine ranking
        if score >= 14:
            ranking = "Excellent"
            recommendation = (
                "Strongly recommended for immediate implementation. "
                "Outstanding financial returns with minimal risk."
            )
        elif score >= 10:
            ranking = "Very Good"
            recommendation = (
                "Recommended for near-term implementation. "
                "Strong financial returns justify priority status."
            )
        elif score >= 7:
            ranking = "Good"
            recommendation = (
                "Recommended for inclusion in capital budget. "
                "Solid returns warrant implementation within 1-2 years."
            )
        elif score >= 4:
            ranking = "Marginal"
            recommendation = (
                "Consider if budget allows. "
                "Returns are acceptable but not compelling. "
                "May improve with incentives or design optimization."
            )
        else:
            ranking = "Poor"
            recommendation = (
                "Not recommended without significant changes. "
                "Project does not meet minimum financial hurdles. "
                "Revisit scope or pursue alternative technologies."
            )

        return ranking, recommendation

    def optimize_portfolio(
        self,
        projects: List[WasteHeatProject],
        budget_constraint_usd: Optional[float] = None,
        max_projects: Optional[int] = None,
    ) -> PortfolioAnalysisResult:
        """
        Optimize a portfolio of waste heat recovery projects.

        Uses profitability index (PI = NPV / Investment) for ranking
        when budget-constrained.

        Args:
            projects: List of candidate projects
            budget_constraint_usd: Maximum total investment
            max_projects: Maximum number of projects to select

        Returns:
            Portfolio analysis result with optimal selection
        """
        logger.info(f"Optimizing portfolio of {len(projects)} projects")

        # Analyze each project
        project_analyses = []
        for project in projects:
            analysis = self.analyze_project(
                project,
                include_sensitivity=False,
                include_monte_carlo=False,
            )
            pi = analysis.npv_usd / project.capital_cost_usd if project.capital_cost_usd > 0 else 0
            project_analyses.append({
                "project_id": project.project_id,
                "name": project.name,
                "capital_cost": project.capital_cost_usd,
                "npv": analysis.npv_usd,
                "irr_pct": analysis.irr_pct,
                "payback_years": analysis.simple_payback_years,
                "profitability_index": round(pi, 3),
                "annual_savings": project.annual_net_savings,
                "energy_savings_mmbtu": project.energy_metrics.annual_energy_savings_mmbtu if project.energy_metrics else 0,
                "co2_reduction_tons": project.energy_metrics.co2_reduction_tons_yr if project.energy_metrics else 0,
            })

        # Rank by profitability index
        project_analyses.sort(key=lambda x: x["profitability_index"], reverse=True)

        # Add rankings
        for i, proj in enumerate(project_analyses):
            proj["rank"] = i + 1

        # Select projects under budget constraint
        selected = []
        remaining_budget = budget_constraint_usd or float('inf')
        remaining_slots = max_projects or len(projects)

        for proj in project_analyses:
            if (proj["capital_cost"] <= remaining_budget and
                    remaining_slots > 0 and
                    proj["npv"] > 0):
                selected.append(proj["name"])
                remaining_budget -= proj["capital_cost"]
                remaining_slots -= 1

        # Calculate portfolio metrics
        selected_projects = [p for p in project_analyses if p["name"] in selected]
        total_investment = sum(p["capital_cost"] for p in selected_projects)
        portfolio_npv = sum(p["npv"] for p in selected_projects)
        total_annual_savings = sum(p["annual_savings"] for p in selected_projects)

        portfolio_payback = (
            total_investment / total_annual_savings
            if total_annual_savings > 0 else float('inf')
        )

        # Budget utilization
        budget_utilization = (
            (total_investment / budget_constraint_usd * 100)
            if budget_constraint_usd else 0
        )

        # Implementation schedule (phased by rank)
        schedule = []
        for proj in selected_projects:
            phase = (
                "Phase 1 (Year 1)" if proj["rank"] <= 3 else
                "Phase 2 (Year 2)" if proj["rank"] <= 6 else
                "Phase 3 (Year 3)"
            )
            schedule.append({
                "project_name": proj["name"],
                "implementation_phase": phase,
                "capital_cost": proj["capital_cost"],
                "expected_npv": proj["npv"],
            })

        # Aggregate energy/emissions
        total_energy = sum(p.get("energy_savings_mmbtu", 0) for p in selected_projects)
        total_co2 = sum(p.get("co2_reduction_tons", 0) for p in selected_projects)

        return PortfolioAnalysisResult(
            total_investment_usd=round(total_investment, 0),
            portfolio_npv_usd=round(portfolio_npv, 0),
            portfolio_payback_years=round(portfolio_payback, 2),
            project_rankings=project_analyses,
            selected_projects=selected,
            budget_utilization_pct=round(budget_utilization, 1),
            implementation_schedule=schedule,
            total_energy_savings_mmbtu_yr=round(total_energy, 0),
            total_co2_reduction_tons_yr=round(total_co2, 0),
            total_annual_savings_usd=round(total_annual_savings, 0),
        )

    def _calculate_provenance(self, project: WasteHeatProject) -> str:
        """Calculate SHA-256 provenance hash."""
        import json

        provenance_data = {
            "project": project.dict(),
            "discount_rate": self.discount_rate,
            "energy_escalation": self.energy_escalation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        provenance_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_capital_recovery_factor(
    discount_rate: float,
    years: int,
) -> float:
    """
    Calculate Capital Recovery Factor (CRF).

    CRF = r(1+r)^n / ((1+r)^n - 1)

    Used to convert present value to uniform annual payments.

    Args:
        discount_rate: Discount rate
        years: Number of years

    Returns:
        Capital Recovery Factor
    """
    if discount_rate <= 0:
        return 1.0 / years

    factor = (1 + discount_rate) ** years
    return (discount_rate * factor) / (factor - 1)


def calculate_present_worth_factor(
    discount_rate: float,
    years: int,
) -> float:
    """
    Calculate Present Worth Factor (PWF).

    PWF = ((1+r)^n - 1) / (r(1+r)^n)

    Used to convert uniform annual payments to present value.

    Args:
        discount_rate: Discount rate
        years: Number of years

    Returns:
        Present Worth Factor
    """
    if discount_rate <= 0:
        return float(years)

    factor = (1 + discount_rate) ** years
    return (factor - 1) / (discount_rate * factor)


def estimate_installation_cost(
    equipment_cost_usd: float,
    installation_type: str = "standard",
) -> Dict[str, float]:
    """
    Estimate installation cost based on equipment cost.

    Uses industry-standard installation factors.

    Args:
        equipment_cost_usd: Equipment purchase cost
        installation_type: standard, retrofit, or greenfield

    Returns:
        Cost breakdown dictionary
    """
    installation_factors = {
        "standard": 0.50,
        "retrofit": 0.75,
        "greenfield": 0.35,
    }

    factor = installation_factors.get(installation_type, 0.50)
    installation_cost = equipment_cost_usd * factor

    # Engineering and permitting
    engineering_cost = equipment_cost_usd * 0.10
    permitting_cost = equipment_cost_usd * 0.02

    # Contingency
    subtotal = equipment_cost_usd + installation_cost + engineering_cost + permitting_cost
    contingency = subtotal * 0.15

    return {
        "equipment_cost_usd": equipment_cost_usd,
        "installation_cost_usd": round(installation_cost, 0),
        "engineering_cost_usd": round(engineering_cost, 0),
        "permitting_cost_usd": round(permitting_cost, 0),
        "contingency_usd": round(contingency, 0),
        "total_capital_cost_usd": round(subtotal + contingency, 0),
    }
