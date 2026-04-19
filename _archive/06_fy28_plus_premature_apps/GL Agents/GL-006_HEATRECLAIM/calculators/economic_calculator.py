"""
GL-006 HEATRECLAIM - Economic Calculator

Implements deterministic economic analysis for heat recovery projects
including capital cost estimation, operating cost, NPV, IRR, and payback.

All calculations are reproducible with SHA-256 provenance tracking.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math

from ..core.schemas import (
    HeatExchanger,
    EconomicAnalysisResult,
)
from ..core.config import EconomicParameters, ExchangerType

logger = logging.getLogger(__name__)


@dataclass
class CapitalCostBreakdown:
    """Breakdown of capital costs."""

    equipment_cost_usd: float
    installation_cost_usd: float
    piping_cost_usd: float
    instrumentation_cost_usd: float
    engineering_cost_usd: float
    contingency_usd: float
    total_capital_usd: float


@dataclass
class OperatingCostBreakdown:
    """Breakdown of annual operating costs."""

    utility_cost_usd_yr: float
    maintenance_cost_usd_yr: float
    labor_cost_usd_yr: float
    insurance_cost_usd_yr: float
    total_operating_usd_yr: float


class EconomicCalculator:
    """
    Economic analysis calculator for heat recovery systems.

    Provides deterministic calculations for:
    - Heat exchanger capital cost estimation
    - Operating cost estimation
    - Annual utility savings
    - NPV, IRR, and payback period
    - Total annual cost (TAC)

    Example:
        >>> calc = EconomicCalculator()
        >>> result = calc.calculate(
        ...     exchangers=exchangers,
        ...     utility_savings_kW=1000,
        ...     utility_cost_usd_gj=15.0
        ... )
        >>> print(f"Payback: {result.payback_period_years:.1f} years")
    """

    VERSION = "1.0.0"

    # Cost correlation coefficients (USD, 2024 basis)
    # Capital cost = a * Area^b
    COST_CORRELATIONS = {
        ExchangerType.SHELL_AND_TUBE: {"a": 10000, "b": 0.68, "material_factor": 1.0},
        ExchangerType.PLATE: {"a": 8000, "b": 0.72, "material_factor": 1.2},
        ExchangerType.PLATE_FIN: {"a": 12000, "b": 0.65, "material_factor": 1.5},
        ExchangerType.SPIRAL: {"a": 15000, "b": 0.60, "material_factor": 1.3},
        ExchangerType.AIR_COOLED: {"a": 25000, "b": 0.55, "material_factor": 1.0},
        ExchangerType.DOUBLE_PIPE: {"a": 5000, "b": 0.75, "material_factor": 1.0},
        ExchangerType.ECONOMIZER: {"a": 8000, "b": 0.70, "material_factor": 1.0},
        ExchangerType.RECUPERATOR: {"a": 12000, "b": 0.65, "material_factor": 1.2},
    }

    def __init__(
        self,
        params: Optional[EconomicParameters] = None,
        cost_year: int = 2024,
    ) -> None:
        """
        Initialize economic calculator.

        Args:
            params: Economic parameters for calculations
            cost_year: Base year for cost estimates
        """
        self.params = params or EconomicParameters()
        self.cost_year = cost_year

    def estimate_exchanger_capital_cost(
        self,
        exchanger: HeatExchanger,
        area_m2: Optional[float] = None,
    ) -> float:
        """
        Estimate capital cost for a heat exchanger.

        Uses correlation: Cost = a * Area^b * material_factor * installation_factor

        Args:
            exchanger: Heat exchanger specification
            area_m2: Override area (if not in exchanger object)

        Returns:
            Estimated capital cost (USD)
        """
        area = area_m2 if area_m2 is not None else exchanger.area_m2

        if area <= 0:
            return 0.0

        # Get cost correlation for exchanger type
        corr = self.COST_CORRELATIONS.get(
            exchanger.exchanger_type,
            self.COST_CORRELATIONS[ExchangerType.SHELL_AND_TUBE]
        )

        # Base equipment cost
        equipment_cost = corr["a"] * (area ** corr["b"])

        # Apply material factor
        equipment_cost *= corr["material_factor"]

        # Apply installation factor
        installed_cost = equipment_cost * self.params.installation_factor

        # Add piping and instrumentation
        piping = equipment_cost * self.params.piping_factor
        instrumentation = equipment_cost * self.params.instrumentation_factor

        total_cost = installed_cost + piping + instrumentation

        return round(total_cost, 2)

    def calculate_capital_costs(
        self,
        exchangers: List[HeatExchanger],
        include_contingency: bool = True,
        contingency_percent: float = 15.0,
    ) -> CapitalCostBreakdown:
        """
        Calculate total capital cost for a set of exchangers.

        Args:
            exchangers: List of heat exchangers
            include_contingency: Add contingency to estimate
            contingency_percent: Contingency percentage

        Returns:
            CapitalCostBreakdown with itemized costs
        """
        equipment_cost = 0.0
        installation_cost = 0.0
        piping_cost = 0.0
        instrumentation_cost = 0.0

        for hx in exchangers:
            if hx.is_existing and not hx.is_reused:
                continue  # Skip existing exchangers not being modified

            area = hx.area_m2
            if area <= 0:
                continue

            # Get correlation
            corr = self.COST_CORRELATIONS.get(
                hx.exchanger_type,
                self.COST_CORRELATIONS[ExchangerType.SHELL_AND_TUBE]
            )

            # Base equipment cost
            eq_cost = corr["a"] * (area ** corr["b"]) * corr["material_factor"]
            equipment_cost += eq_cost

            # Installation
            installation_cost += eq_cost * (self.params.installation_factor - 1)

            # Piping and instrumentation
            piping_cost += eq_cost * self.params.piping_factor
            instrumentation_cost += eq_cost * self.params.instrumentation_factor

        # Engineering (typically 10% of equipment)
        engineering_cost = equipment_cost * 0.10

        # Subtotal
        subtotal = (
            equipment_cost +
            installation_cost +
            piping_cost +
            instrumentation_cost +
            engineering_cost
        )

        # Contingency
        if include_contingency:
            contingency = subtotal * contingency_percent / 100
        else:
            contingency = 0.0

        total = subtotal + contingency

        return CapitalCostBreakdown(
            equipment_cost_usd=round(equipment_cost, 2),
            installation_cost_usd=round(installation_cost, 2),
            piping_cost_usd=round(piping_cost, 2),
            instrumentation_cost_usd=round(instrumentation_cost, 2),
            engineering_cost_usd=round(engineering_cost, 2),
            contingency_usd=round(contingency, 2),
            total_capital_usd=round(total, 2),
        )

    def calculate_utility_savings(
        self,
        hot_utility_reduction_kW: float,
        cold_utility_reduction_kW: float,
        hot_utility_cost_usd_gj: Optional[float] = None,
        cold_utility_cost_usd_gj: Optional[float] = None,
        operating_hours_per_year: Optional[int] = None,
    ) -> float:
        """
        Calculate annual utility cost savings.

        Args:
            hot_utility_reduction_kW: Reduction in hot utility (kW)
            cold_utility_reduction_kW: Reduction in cold utility (kW)
            hot_utility_cost_usd_gj: Hot utility cost ($/GJ)
            cold_utility_cost_usd_gj: Cold utility cost ($/GJ)
            operating_hours_per_year: Annual operating hours

        Returns:
            Annual savings (USD/year)
        """
        hot_cost = hot_utility_cost_usd_gj or self.params.steam_cost_usd_gj
        cold_cost = cold_utility_cost_usd_gj or self.params.cooling_water_cost_usd_gj
        hours = operating_hours_per_year or self.params.operating_hours_per_year

        # Convert kW to GJ/year
        # 1 kW * 1 hour = 3.6 MJ = 0.0036 GJ
        gj_per_kw_hr = 0.0036

        hot_gj_yr = hot_utility_reduction_kW * hours * gj_per_kw_hr
        cold_gj_yr = cold_utility_reduction_kW * hours * gj_per_kw_hr

        hot_savings = hot_gj_yr * hot_cost
        cold_savings = cold_gj_yr * cold_cost

        return round(hot_savings + cold_savings, 2)

    def calculate_operating_costs(
        self,
        capital_cost_usd: float,
        utility_cost_usd_yr: float = 0.0,
    ) -> OperatingCostBreakdown:
        """
        Calculate annual operating costs.

        Args:
            capital_cost_usd: Total capital investment
            utility_cost_usd_yr: Annual utility cost

        Returns:
            OperatingCostBreakdown with itemized costs
        """
        # Maintenance as fraction of capital
        maintenance = capital_cost_usd * self.params.maintenance_cost_fraction

        # Labor (minimal for heat exchangers)
        labor = capital_cost_usd * 0.01  # 1% of capital

        # Insurance
        insurance = capital_cost_usd * 0.005  # 0.5% of capital

        total = utility_cost_usd_yr + maintenance + labor + insurance

        return OperatingCostBreakdown(
            utility_cost_usd_yr=round(utility_cost_usd_yr, 2),
            maintenance_cost_usd_yr=round(maintenance, 2),
            labor_cost_usd_yr=round(labor, 2),
            insurance_cost_usd_yr=round(insurance, 2),
            total_operating_usd_yr=round(total, 2),
        )

    def calculate_npv(
        self,
        capital_cost_usd: float,
        annual_savings_usd: float,
        annual_operating_cost_usd: float = 0.0,
        discount_rate: Optional[float] = None,
        project_lifetime_years: Optional[int] = None,
    ) -> float:
        """
        Calculate Net Present Value.

        NPV = -Capex + Σ (Savings - OpCost) / (1 + r)^t

        Args:
            capital_cost_usd: Initial capital investment
            annual_savings_usd: Annual savings from project
            annual_operating_cost_usd: Annual operating cost
            discount_rate: Discount rate (decimal)
            project_lifetime_years: Project lifetime

        Returns:
            NPV in USD
        """
        r = discount_rate or self.params.discount_rate
        n = project_lifetime_years or self.params.project_lifetime_years

        net_annual_benefit = annual_savings_usd - annual_operating_cost_usd

        # Present value of annuity
        if r > 0:
            pv_factor = (1 - (1 + r) ** (-n)) / r
        else:
            pv_factor = n

        npv = -capital_cost_usd + net_annual_benefit * pv_factor

        return round(npv, 2)

    def calculate_irr(
        self,
        capital_cost_usd: float,
        annual_savings_usd: float,
        annual_operating_cost_usd: float = 0.0,
        project_lifetime_years: Optional[int] = None,
        max_iterations: int = 100,
        tolerance: float = 0.0001,
    ) -> Optional[float]:
        """
        Calculate Internal Rate of Return using Newton-Raphson method.

        Args:
            capital_cost_usd: Initial capital investment
            annual_savings_usd: Annual savings
            annual_operating_cost_usd: Annual operating cost
            project_lifetime_years: Project lifetime
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance

        Returns:
            IRR as decimal (e.g., 0.15 for 15%), or None if not found
        """
        n = project_lifetime_years or self.params.project_lifetime_years
        net_annual = annual_savings_usd - annual_operating_cost_usd

        if net_annual <= 0:
            return None

        # Initial guess
        irr = 0.10  # Start at 10%

        for _ in range(max_iterations):
            # Calculate NPV at current IRR
            if irr <= -1:
                return None

            pv_factor = (1 - (1 + irr) ** (-n)) / irr if irr != 0 else n
            npv = -capital_cost_usd + net_annual * pv_factor

            # Derivative of NPV with respect to r
            if irr != 0:
                d_pv = (
                    n * (1 + irr) ** (-n - 1) / irr -
                    (1 - (1 + irr) ** (-n)) / (irr ** 2)
                )
                d_npv = net_annual * d_pv
            else:
                d_npv = -net_annual * n * (n + 1) / 2

            if abs(d_npv) < 1e-10:
                break

            # Newton-Raphson update
            irr_new = irr - npv / d_npv

            if abs(irr_new - irr) < tolerance:
                return round(irr_new, 4)

            irr = irr_new

        # If didn't converge, try simple search
        for test_irr in [i / 100 for i in range(1, 100)]:
            pv_factor = (1 - (1 + test_irr) ** (-n)) / test_irr
            npv = -capital_cost_usd + net_annual * pv_factor
            if abs(npv) < capital_cost_usd * 0.01:
                return round(test_irr, 4)

        return None

    def calculate_payback(
        self,
        capital_cost_usd: float,
        annual_savings_usd: float,
        annual_operating_cost_usd: float = 0.0,
    ) -> float:
        """
        Calculate simple payback period.

        Args:
            capital_cost_usd: Initial capital investment
            annual_savings_usd: Annual savings
            annual_operating_cost_usd: Annual operating cost

        Returns:
            Payback period in years
        """
        net_annual = annual_savings_usd - annual_operating_cost_usd

        if net_annual <= 0:
            return float('inf')

        return round(capital_cost_usd / net_annual, 2)

    def calculate_total_annual_cost(
        self,
        capital_cost_usd: float,
        annual_operating_cost_usd: float,
        discount_rate: Optional[float] = None,
        project_lifetime_years: Optional[int] = None,
    ) -> float:
        """
        Calculate Total Annual Cost (TAC).

        TAC = Annualized Capital + Operating Cost

        Args:
            capital_cost_usd: Total capital investment
            annual_operating_cost_usd: Annual operating cost
            discount_rate: Discount rate
            project_lifetime_years: Project lifetime

        Returns:
            TAC in USD/year
        """
        r = discount_rate or self.params.discount_rate
        n = project_lifetime_years or self.params.project_lifetime_years

        # Capital recovery factor
        if r > 0:
            crf = r * (1 + r) ** n / ((1 + r) ** n - 1)
        else:
            crf = 1 / n

        annualized_capital = capital_cost_usd * crf
        tac = annualized_capital + annual_operating_cost_usd

        return round(tac, 2)

    def calculate_full_analysis(
        self,
        exchangers: List[HeatExchanger],
        hot_utility_reduction_kW: float,
        cold_utility_reduction_kW: float,
        baseline_utility_cost_usd_yr: float = 0.0,
    ) -> EconomicAnalysisResult:
        """
        Perform complete economic analysis.

        Args:
            exchangers: List of heat exchangers in design
            hot_utility_reduction_kW: Hot utility saved
            cold_utility_reduction_kW: Cold utility saved
            baseline_utility_cost_usd_yr: Current annual utility cost

        Returns:
            EconomicAnalysisResult with all metrics
        """
        # Capital costs
        capital = self.calculate_capital_costs(exchangers)

        # Utility savings
        utility_savings = self.calculate_utility_savings(
            hot_utility_reduction_kW,
            cold_utility_reduction_kW,
        )

        # Operating costs (for new equipment)
        new_utility_cost = baseline_utility_cost_usd_yr - utility_savings
        operating = self.calculate_operating_costs(
            capital.total_capital_usd,
            max(0, new_utility_cost),
        )

        # Investment metrics
        npv = self.calculate_npv(
            capital.total_capital_usd,
            utility_savings,
            operating.maintenance_cost_usd_yr,
        )

        irr = self.calculate_irr(
            capital.total_capital_usd,
            utility_savings,
            operating.maintenance_cost_usd_yr,
        )

        payback = self.calculate_payback(
            capital.total_capital_usd,
            utility_savings,
            operating.maintenance_cost_usd_yr,
        )

        tac = self.calculate_total_annual_cost(
            capital.total_capital_usd,
            operating.total_operating_usd_yr,
        )

        # ROI
        net_annual = utility_savings - operating.maintenance_cost_usd_yr
        roi = (net_annual / capital.total_capital_usd * 100
               if capital.total_capital_usd > 0 else 0.0)

        # CO2 reduction estimate
        # Assume 0.2 kg CO2/kWh for steam, 0.4 kg CO2/kWh for electricity
        co2_factor = 0.2  # kg CO2/kWh for steam
        hours = self.params.operating_hours_per_year
        co2_tonnes = (hot_utility_reduction_kW * hours * co2_factor) / 1000

        # Carbon credit value (assume $50/tonne)
        carbon_value = co2_tonnes * 50

        # Build result
        input_hash = self._compute_hash({
            "exchangers": [hx.exchanger_id for hx in exchangers],
            "hot_utility_reduction_kW": hot_utility_reduction_kW,
            "cold_utility_reduction_kW": cold_utility_reduction_kW,
        })

        result = EconomicAnalysisResult(
            total_capital_cost_usd=capital.total_capital_usd,
            equipment_cost_usd=capital.equipment_cost_usd,
            installation_cost_usd=capital.installation_cost_usd,
            piping_cost_usd=capital.piping_cost_usd,
            instrumentation_cost_usd=capital.instrumentation_cost_usd,
            annual_operating_cost_usd=operating.total_operating_usd_yr,
            utility_cost_usd_yr=max(0, new_utility_cost),
            maintenance_cost_usd_yr=operating.maintenance_cost_usd_yr,
            annual_utility_savings_usd=utility_savings,
            annual_net_savings_usd=round(net_annual, 2),
            total_annual_cost_usd=tac,
            payback_period_years=payback,
            npv_usd=npv,
            irr_percent=round(irr * 100, 2) if irr else None,
            roi_percent=round(roi, 2),
            discount_rate=self.params.discount_rate,
            project_lifetime_years=self.params.project_lifetime_years,
            operating_hours_per_year=self.params.operating_hours_per_year,
            co2_reduction_tonnes_yr=round(co2_tonnes, 2),
            carbon_credit_value_usd_yr=round(carbon_value, 2),
            input_hash=input_hash,
        )

        result.output_hash = self._compute_hash({
            "total_capital_cost_usd": result.total_capital_cost_usd,
            "npv_usd": result.npv_usd,
            "payback_period_years": result.payback_period_years,
        })

        return result

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
