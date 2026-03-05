"""
Financial Impact Quantification Engine -- Three-Statement Climate Impact

Implements financial impact quantification across income statement, balance
sheet, and cash flow statement per TCFD/IFRS S2 requirements.

Provides:
  - Income statement impact: revenue, COGS, opex, carbon costs, insurance
  - Balance sheet impact: asset impairment, provisions, deferred costs
  - Cash flow impact: operating, investing, financing climate cash flows
  - Combined three-statement impact analysis
  - NPV and IRR calculations for climate investments
  - Marginal Abatement Cost Curve (MACC) generation
  - Carbon price sensitivity analysis
  - Monte Carlo financial simulation
  - Climate Value at Risk (CVaR)
  - Year-by-year financial projections under scenarios
  - Adaptation ROI analysis

All financial calculations are deterministic (zero-hallucination).
No LLM calls for numeric computations.

Reference:
    - TCFD Annex: Financial Impact Quantification (June 2017)
    - IFRS S2 Paragraphs 13-14 (Financial Position, Performance, Cash Flows)
    - NGFS Financial Impact Assessment Framework

Example:
    >>> engine = FinancialImpactEngine(config)
    >>> is_impact = await engine.calculate_income_statement_impact("org-1", scenario)
    >>> total = await engine.calculate_total_financial_impact("org-1", [scenario])
"""

from __future__ import annotations

import logging
import math
import random
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    FinancialImpactCategory,
    FinancialStatementType,
    SCENARIO_CARBON_PRICES,
    SCENARIO_LIBRARY,
    ScenarioType,
    SectorType,
    TCFDAppConfig,
    TimeHorizon,
)
from .models import (
    FinancialImpact,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Financial Impact Templates by Sector
# ---------------------------------------------------------------------------

FINANCIAL_IMPACT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "energy": {
        "revenue_risk_pct": Decimal("15"),
        "opex_increase_pct": Decimal("20"),
        "capex_transition_pct": Decimal("30"),
        "asset_impairment_pct": Decimal("25"),
        "insurance_increase_pct": Decimal("8"),
    },
    "transportation": {
        "revenue_risk_pct": Decimal("8"),
        "opex_increase_pct": Decimal("15"),
        "capex_transition_pct": Decimal("25"),
        "asset_impairment_pct": Decimal("12"),
        "insurance_increase_pct": Decimal("10"),
    },
    "materials_buildings": {
        "revenue_risk_pct": Decimal("10"),
        "opex_increase_pct": Decimal("18"),
        "capex_transition_pct": Decimal("20"),
        "asset_impairment_pct": Decimal("15"),
        "insurance_increase_pct": Decimal("12"),
    },
    "banking": {
        "revenue_risk_pct": Decimal("5"),
        "opex_increase_pct": Decimal("3"),
        "capex_transition_pct": Decimal("5"),
        "asset_impairment_pct": Decimal("10"),
        "insurance_increase_pct": Decimal("2"),
    },
    "technology_media": {
        "revenue_risk_pct": Decimal("3"),
        "opex_increase_pct": Decimal("8"),
        "capex_transition_pct": Decimal("10"),
        "asset_impairment_pct": Decimal("3"),
        "insurance_increase_pct": Decimal("3"),
    },
    "agriculture_food_forest": {
        "revenue_risk_pct": Decimal("18"),
        "opex_increase_pct": Decimal("12"),
        "capex_transition_pct": Decimal("15"),
        "asset_impairment_pct": Decimal("10"),
        "insurance_increase_pct": Decimal("15"),
    },
    "consumer_goods": {
        "revenue_risk_pct": Decimal("6"),
        "opex_increase_pct": Decimal("10"),
        "capex_transition_pct": Decimal("8"),
        "asset_impairment_pct": Decimal("5"),
        "insurance_increase_pct": Decimal("5"),
    },
    "default": {
        "revenue_risk_pct": Decimal("8"),
        "opex_increase_pct": Decimal("10"),
        "capex_transition_pct": Decimal("12"),
        "asset_impairment_pct": Decimal("8"),
        "insurance_increase_pct": Decimal("5"),
    },
}


class FinancialImpactEngine:
    """
    Climate financial impact quantification across three statements.

    Calculates income statement, balance sheet, and cash flow impacts
    from climate scenarios with NPV/IRR analysis and Monte Carlo simulation.

    Attributes:
        config: Application configuration.
        _impacts: In-memory store of financial impacts by org_id.
        _macc_curves: Cached MACC curves by org_id.
    """

    def __init__(self, config: Optional[TCFDAppConfig] = None) -> None:
        """Initialize FinancialImpactEngine."""
        self.config = config or TCFDAppConfig()
        self._impacts: Dict[str, List[FinancialImpact]] = {}
        self._macc_curves: Dict[str, List[Dict[str, Any]]] = {}
        self._projections: Dict[str, Dict[str, Any]] = {}
        logger.info("FinancialImpactEngine initialized")

    # ------------------------------------------------------------------
    # Income Statement Impact
    # ------------------------------------------------------------------

    async def calculate_income_statement_impact(
        self,
        org_id: str,
        scenario_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate climate impact on income statement.

        Evaluates revenue risk, carbon cost, operating cost increases,
        insurance premium changes, and opportunity revenue.

        Args:
            org_id: Organization ID.
            scenario_result: Dict with financial data and scenario outputs:
                - revenue (float): Current annual revenue.
                - cogs (float): Current cost of goods sold.
                - operating_expenses (float): Current opex.
                - emissions_tco2e (float): Annual emissions.
                - carbon_price (float): Carbon price under scenario.
                - sector (str): Sector key.
                - revenue_at_risk_pct (float): Percentage of revenue at risk.
                - opportunity_revenue (float): Climate opportunity revenue.

        Returns:
            Dict with line-item impacts and totals.
        """
        start = _now()

        revenue = Decimal(str(scenario_result.get("revenue", 0)))
        cogs = Decimal(str(scenario_result.get("cogs", 0)))
        opex = Decimal(str(scenario_result.get("operating_expenses", 0)))
        emissions = Decimal(str(scenario_result.get("emissions_tco2e", 0)))
        carbon_price = Decimal(str(scenario_result.get("carbon_price", 0)))
        sector = scenario_result.get("sector", "default")
        revenue_at_risk_pct = Decimal(str(scenario_result.get("revenue_at_risk_pct", 0)))
        opp_revenue = Decimal(str(scenario_result.get("opportunity_revenue", 0)))

        template = FINANCIAL_IMPACT_TEMPLATES.get(sector, FINANCIAL_IMPACT_TEMPLATES["default"])

        # Revenue impact
        if revenue_at_risk_pct > 0:
            revenue_loss = (revenue * revenue_at_risk_pct / Decimal("100")).quantize(Decimal("0.01"))
        else:
            revenue_loss = (revenue * template["revenue_risk_pct"] / Decimal("100")).quantize(Decimal("0.01"))

        # Carbon cost
        carbon_cost = (emissions * carbon_price).quantize(Decimal("0.01"))

        # Operating cost increase
        opex_increase = (opex * template["opex_increase_pct"] / Decimal("100")).quantize(Decimal("0.01"))

        # Insurance cost increase
        insurance_increase = (
            (revenue * Decimal("0.01")) * template["insurance_increase_pct"] / Decimal("100")
        ).quantize(Decimal("0.01"))

        # Net income impact
        net_impact = (
            -revenue_loss + opp_revenue - carbon_cost - opex_increase - insurance_increase
        ).quantize(Decimal("0.01"))

        # Store impacts
        impact_items = [
            ("Revenue at Risk", "revenue", -revenue_loss),
            ("Opportunity Revenue", "revenue", opp_revenue),
            ("Carbon Cost", "operating_cost", -carbon_cost),
            ("Operating Cost Increase", "operating_cost", -opex_increase),
            ("Insurance Premium Increase", "insurance_cost", -insurance_increase),
        ]

        for name, category, amount in impact_items:
            impact = FinancialImpact(
                org_id=org_id,
                statement_area="income_statement",
                line_item=name,
                current_value=revenue if "Revenue" in name else opex,
                projected_value=(revenue if "Revenue" in name else opex) + amount,
                impact_amount=amount,
            )
            self._store_impact(org_id, impact)

        result = {
            "org_id": org_id,
            "statement": "income_statement",
            "revenue_loss": str(revenue_loss),
            "opportunity_revenue": str(opp_revenue),
            "carbon_cost": str(carbon_cost),
            "opex_increase": str(opex_increase),
            "insurance_increase": str(insurance_increase),
            "net_income_impact": str(net_impact),
            "net_margin_impact_pct": str(
                (net_impact / max(revenue, Decimal("1")) * Decimal("100")).quantize(Decimal("0.1"))
            ),
            "provenance_hash": _sha256(f"{org_id}:is:{net_impact}"),
        }

        elapsed_ms = (_now() - start).total_seconds() * 1000
        logger.info(
            "Income statement impact for org=%s: net=$%.0f in %.1f ms",
            org_id, net_impact, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Balance Sheet Impact
    # ------------------------------------------------------------------

    async def calculate_balance_sheet_impact(
        self,
        org_id: str,
        scenario_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate climate impact on balance sheet.

        Evaluates asset impairment, stranded asset write-downs, adaptation
        investments, and climate-related provisions.

        Args:
            org_id: Organization ID.
            scenario_result: Dict with:
                - total_assets (float): Total asset value.
                - ppe_value (float): Property, plant, and equipment.
                - stranded_asset_pct (float): % assets at stranding risk.
                - adaptation_investment (float): Planned adaptation capex.
                - sector (str): Sector key.

        Returns:
            Dict with balance sheet impact breakdown.
        """
        start = _now()

        total_assets = Decimal(str(scenario_result.get("total_assets", 0)))
        ppe = Decimal(str(scenario_result.get("ppe_value", 0)))
        stranded_pct = Decimal(str(scenario_result.get("stranded_asset_pct", 0)))
        adaptation_inv = Decimal(str(scenario_result.get("adaptation_investment", 0)))
        sector = scenario_result.get("sector", "default")

        template = FINANCIAL_IMPACT_TEMPLATES.get(sector, FINANCIAL_IMPACT_TEMPLATES["default"])

        # Asset impairment
        impairment = (ppe * template["asset_impairment_pct"] / Decimal("100")).quantize(Decimal("0.01"))

        # Stranded assets
        stranded_value = (ppe * stranded_pct / Decimal("100")).quantize(Decimal("0.01"))

        # Climate provisions (estimated future liabilities)
        provisions = (total_assets * Decimal("0.02")).quantize(Decimal("0.01"))

        # Net asset impact
        net_impact = (-impairment - stranded_value - provisions + adaptation_inv).quantize(Decimal("0.01"))

        result = {
            "org_id": org_id,
            "statement": "balance_sheet",
            "asset_impairment": str(impairment),
            "stranded_asset_writedown": str(stranded_value),
            "climate_provisions": str(provisions),
            "adaptation_investment_capitalized": str(adaptation_inv),
            "net_asset_impact": str(net_impact),
            "net_asset_impact_pct": str(
                (net_impact / max(total_assets, Decimal("1")) * Decimal("100")).quantize(Decimal("0.1"))
            ),
            "provenance_hash": _sha256(f"{org_id}:bs:{net_impact}"),
        }

        elapsed_ms = (_now() - start).total_seconds() * 1000
        logger.info(
            "Balance sheet impact for org=%s: net=$%.0f in %.1f ms",
            org_id, net_impact, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Cash Flow Impact
    # ------------------------------------------------------------------

    async def calculate_cash_flow_impact(
        self,
        org_id: str,
        scenario_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate climate impact on cash flow statement.

        Args:
            org_id: Organization ID.
            scenario_result: Dict with:
                - operating_cash_flow (float): Current operating CF.
                - carbon_cost (float): Annual carbon costs.
                - adaptation_capex (float): Adaptation capital expenditure.
                - transition_capex (float): Transition capital expenditure.
                - green_financing (float): Green financing proceeds.
                - insurance_premium_change (float): Change in insurance costs.

        Returns:
            Dict with cash flow impact breakdown.
        """
        start = _now()

        ocf = Decimal(str(scenario_result.get("operating_cash_flow", 0)))
        carbon_cost = Decimal(str(scenario_result.get("carbon_cost", 0)))
        adapt_capex = Decimal(str(scenario_result.get("adaptation_capex", 0)))
        trans_capex = Decimal(str(scenario_result.get("transition_capex", 0)))
        green_fin = Decimal(str(scenario_result.get("green_financing", 0)))
        insurance_change = Decimal(str(scenario_result.get("insurance_premium_change", 0)))

        # Operating cash flow impact
        operating_impact = (-carbon_cost - insurance_change).quantize(Decimal("0.01"))

        # Investing cash flow impact
        investing_impact = (-adapt_capex - trans_capex).quantize(Decimal("0.01"))

        # Financing cash flow impact
        financing_impact = green_fin.quantize(Decimal("0.01"))

        # Net cash flow impact
        net_impact = (operating_impact + investing_impact + financing_impact).quantize(Decimal("0.01"))

        result = {
            "org_id": org_id,
            "statement": "cash_flow",
            "operating_cf_impact": str(operating_impact),
            "carbon_cost_outflow": str(carbon_cost),
            "insurance_change": str(insurance_change),
            "investing_cf_impact": str(investing_impact),
            "adaptation_capex": str(adapt_capex),
            "transition_capex": str(trans_capex),
            "financing_cf_impact": str(financing_impact),
            "green_financing_inflow": str(green_fin),
            "net_cash_flow_impact": str(net_impact),
            "provenance_hash": _sha256(f"{org_id}:cf:{net_impact}"),
        }

        elapsed_ms = (_now() - start).total_seconds() * 1000
        logger.info(
            "Cash flow impact for org=%s: net=$%.0f in %.1f ms",
            org_id, net_impact, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Total Three-Statement Impact
    # ------------------------------------------------------------------

    async def calculate_total_financial_impact(
        self,
        org_id: str,
        scenario_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate total climate financial impact across all three statements.

        Args:
            org_id: Organization ID.
            scenario_results: List of scenario result dicts.

        Returns:
            Dict with combined three-statement impact summary.
        """
        start = _now()
        all_is: List[Dict[str, Any]] = []
        all_bs: List[Dict[str, Any]] = []
        all_cf: List[Dict[str, Any]] = []

        for scenario in scenario_results:
            is_impact = await self.calculate_income_statement_impact(org_id, scenario)
            bs_impact = await self.calculate_balance_sheet_impact(org_id, scenario)
            cf_impact = await self.calculate_cash_flow_impact(org_id, scenario)
            all_is.append(is_impact)
            all_bs.append(bs_impact)
            all_cf.append(cf_impact)

        # Aggregate across scenarios (worst case)
        worst_is = min(
            (Decimal(r["net_income_impact"]) for r in all_is), default=Decimal("0"),
        )
        worst_bs = min(
            (Decimal(r["net_asset_impact"]) for r in all_bs), default=Decimal("0"),
        )
        worst_cf = min(
            (Decimal(r["net_cash_flow_impact"]) for r in all_cf), default=Decimal("0"),
        )

        result = {
            "org_id": org_id,
            "scenarios_analyzed": len(scenario_results),
            "income_statement_impacts": all_is,
            "balance_sheet_impacts": all_bs,
            "cash_flow_impacts": all_cf,
            "worst_case_income_impact": str(worst_is),
            "worst_case_asset_impact": str(worst_bs),
            "worst_case_cash_flow_impact": str(worst_cf),
            "total_worst_case_exposure": str((worst_is + worst_bs + worst_cf).quantize(Decimal("0.01"))),
            "provenance_hash": _sha256(f"{org_id}:total:{worst_is}:{worst_bs}:{worst_cf}"),
        }

        elapsed_ms = (_now() - start).total_seconds() * 1000
        logger.info(
            "Total financial impact for org=%s: %d scenarios analyzed in %.1f ms",
            org_id, len(scenario_results), elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # NPV and IRR
    # ------------------------------------------------------------------

    async def calculate_npv(
        self,
        cash_flows: List[Decimal],
        discount_rate: Decimal,
    ) -> Dict[str, Any]:
        """
        Calculate Net Present Value of climate-related cash flows.

        Args:
            cash_flows: List of annual cash flows. Index 0 = year 0 (initial investment).
            discount_rate: Discount rate as decimal (e.g. 0.08 for 8%).

        Returns:
            Dict with NPV, payback period, and profitability index.
        """
        rate = float(discount_rate)
        npv = float(cash_flows[0]) if cash_flows else 0.0

        for t in range(1, len(cash_flows)):
            npv += float(cash_flows[t]) / ((1 + rate) ** t)

        # Payback period
        cumulative = float(cash_flows[0]) if cash_flows else 0.0
        payback = len(cash_flows)
        for t in range(1, len(cash_flows)):
            cumulative += float(cash_flows[t])
            if cumulative >= 0:
                payback = t
                break

        # Profitability index
        pv_benefits = sum(
            float(cash_flows[t]) / ((1 + rate) ** t)
            for t in range(1, len(cash_flows))
        ) if len(cash_flows) > 1 else 0.0
        initial = abs(float(cash_flows[0])) if cash_flows else 1.0
        pi = pv_benefits / max(initial, 0.01)

        return {
            "npv": str(Decimal(str(round(npv, 2))),),
            "discount_rate": str(discount_rate),
            "payback_period_years": payback,
            "profitability_index": str(Decimal(str(round(pi, 3)))),
            "years": len(cash_flows),
            "recommendation": "Invest" if npv > 0 else "Do not invest",
        }

    async def calculate_irr(
        self,
        cash_flows: List[Decimal],
    ) -> float:
        """
        Calculate Internal Rate of Return using Newton-Raphson method.

        Args:
            cash_flows: List of annual cash flows.

        Returns:
            IRR as float (e.g. 0.15 for 15%).
        """
        if not cash_flows or len(cash_flows) < 2:
            return 0.0

        flows = [float(cf) for cf in cash_flows]

        # Newton-Raphson iteration
        rate = 0.10  # Initial guess
        for _ in range(200):
            npv = sum(f / ((1 + rate) ** t) for t, f in enumerate(flows))
            dnpv = sum(-t * f / ((1 + rate) ** (t + 1)) for t, f in enumerate(flows))
            if abs(dnpv) < 1e-12:
                break
            rate -= npv / dnpv
            if abs(npv) < 0.01:
                break

        return round(rate, 4)

    # ------------------------------------------------------------------
    # MACC (Marginal Abatement Cost Curve)
    # ------------------------------------------------------------------

    async def generate_macc(
        self,
        org_id: str,
        abatement_options: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate a Marginal Abatement Cost Curve.

        Each option dict contains:
          - name (str): Abatement option name.
          - abatement_tco2e (float): Annual tCO2e abated.
          - cost_per_tco2e (float): Cost per tCO2e abated (can be negative).
          - capex_usd (float): Capital expenditure required.
          - implementation_years (int): Years to implement.

        Options are sorted by cost_per_tco2e (cheapest first = MACC order).

        Args:
            org_id: Organization ID.
            abatement_options: List of abatement option dicts.

        Returns:
            Sorted list of MACC entries with cumulative abatement.
        """
        start = _now()

        entries: List[Dict[str, Any]] = []
        for opt in abatement_options:
            entries.append({
                "name": opt.get("name", "Unknown"),
                "abatement_tco2e": Decimal(str(opt.get("abatement_tco2e", 0))),
                "cost_per_tco2e": Decimal(str(opt.get("cost_per_tco2e", 0))),
                "capex_usd": Decimal(str(opt.get("capex_usd", 0))),
                "implementation_years": opt.get("implementation_years", 1),
            })

        # Sort by marginal cost (cheapest first = left side of MACC)
        entries.sort(key=lambda e: e["cost_per_tco2e"])

        # Calculate cumulative abatement
        cumulative = Decimal("0")
        for entry in entries:
            cumulative += entry["abatement_tco2e"]
            entry["cumulative_abatement_tco2e"] = cumulative
            entry["total_annual_cost"] = (
                entry["abatement_tco2e"] * entry["cost_per_tco2e"]
            ).quantize(Decimal("0.01"))
            # Convert Decimals to strings for serialization
            for key in ["abatement_tco2e", "cost_per_tco2e", "capex_usd",
                        "cumulative_abatement_tco2e", "total_annual_cost"]:
                entry[key] = str(entry[key])

        self._macc_curves[org_id] = entries

        elapsed_ms = (_now() - start).total_seconds() * 1000
        logger.info(
            "MACC generated for org=%s: %d options, cumulative=%s tCO2e in %.1f ms",
            org_id, len(entries),
            entries[-1]["cumulative_abatement_tco2e"] if entries else "0",
            elapsed_ms,
        )
        return entries

    # ------------------------------------------------------------------
    # Carbon Price Sensitivity
    # ------------------------------------------------------------------

    async def run_carbon_price_sensitivity(
        self,
        org_id: str,
        price_range: List[Decimal],
        emissions_tco2e: Decimal,
    ) -> List[Dict[str, Any]]:
        """
        Run sensitivity analysis on carbon price impact.

        Args:
            org_id: Organization ID.
            price_range: List of carbon prices to evaluate (USD/tCO2e).
            emissions_tco2e: Annual emissions in tCO2e.

        Returns:
            List of sensitivity results (one per price point).
        """
        start = _now()
        results: List[Dict[str, Any]] = []
        emissions = Decimal(str(emissions_tco2e))

        for price in price_range:
            p = Decimal(str(price))
            annual_cost = (emissions * p).quantize(Decimal("0.01"))
            results.append({
                "carbon_price_usd": str(p),
                "annual_cost_usd": str(annual_cost),
                "monthly_cost_usd": str((annual_cost / Decimal("12")).quantize(Decimal("0.01"))),
                "cost_per_revenue_pct": "N/A",  # Requires revenue input
            })

        elapsed_ms = (_now() - start).total_seconds() * 1000
        logger.info(
            "Carbon price sensitivity for org=%s: %d price points in %.1f ms",
            org_id, len(results), elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------
    # Monte Carlo Financial Simulation
    # ------------------------------------------------------------------

    async def run_monte_carlo_financial(
        self,
        org_id: str,
        scenario: Dict[str, Any],
        iterations: int = 10000,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation on financial impact.

        Varies key parameters (carbon price, demand shift, physical damage)
        within uncertainty ranges to produce a distribution of outcomes.

        Args:
            org_id: Organization ID.
            scenario: Base scenario parameters with uncertainty ranges.
            iterations: Number of simulation iterations.

        Returns:
            Dict with distribution statistics (mean, median, p5, p95, std).
        """
        start = _now()
        rng = random.Random(42)  # Deterministic seed for reproducibility

        base_revenue = float(scenario.get("revenue", 1000000))
        base_emissions = float(scenario.get("emissions_tco2e", 10000))
        carbon_price_mean = float(scenario.get("carbon_price", 100))
        carbon_price_std = float(scenario.get("carbon_price_std", 30))
        demand_shift_mean = float(scenario.get("demand_shift_pct", 5))
        demand_shift_std = float(scenario.get("demand_shift_std", 3))

        outcomes: List[float] = []
        for _ in range(iterations):
            cp = max(0, rng.gauss(carbon_price_mean, carbon_price_std))
            ds = rng.gauss(demand_shift_mean, demand_shift_std)

            carbon_cost = base_emissions * cp
            revenue_loss = base_revenue * ds / 100
            net_impact = -(carbon_cost + revenue_loss)
            outcomes.append(net_impact)

        outcomes.sort()
        n = len(outcomes)
        mean_val = sum(outcomes) / n
        median_val = outcomes[n // 2]
        p5 = outcomes[int(n * 0.05)]
        p95 = outcomes[int(n * 0.95)]
        variance = sum((x - mean_val) ** 2 for x in outcomes) / n
        std_dev = variance ** 0.5

        result = {
            "org_id": org_id,
            "iterations": iterations,
            "mean_impact": str(Decimal(str(round(mean_val, 2)))),
            "median_impact": str(Decimal(str(round(median_val, 2)))),
            "p5_impact": str(Decimal(str(round(p5, 2)))),
            "p95_impact": str(Decimal(str(round(p95, 2)))),
            "std_dev": str(Decimal(str(round(std_dev, 2)))),
            "max_loss": str(Decimal(str(round(min(outcomes), 2)))),
            "max_gain": str(Decimal(str(round(max(outcomes), 2)))),
            "probability_of_loss_pct": str(
                Decimal(str(round(sum(1 for o in outcomes if o < 0) / n * 100, 1)))
            ),
            "provenance_hash": _sha256(f"{org_id}:mc:{iterations}:{mean_val:.2f}"),
        }

        elapsed_ms = (_now() - start).total_seconds() * 1000
        logger.info(
            "Monte Carlo for org=%s: %d iterations, mean=$%.0f, p5=$%.0f, p95=$%.0f in %.1f ms",
            org_id, iterations, mean_val, p5, p95, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Climate Value at Risk
    # ------------------------------------------------------------------

    async def calculate_climate_var(
        self,
        org_id: str,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Calculate Climate Value at Risk (CVaR).

        Uses stored financial impacts and Monte Carlo results to compute
        the maximum expected loss at a given confidence level.

        Args:
            org_id: Organization ID.
            confidence_level: Confidence level (e.g. 0.95 for 95%).

        Returns:
            Dict with CVaR, confidence level, and contributing factors.
        """
        start = _now()
        impacts = self._impacts.get(org_id, [])

        if not impacts:
            return {
                "org_id": org_id,
                "climate_var": "0",
                "confidence_level": confidence_level,
                "message": "No financial impacts recorded",
            }

        # Sum all negative impacts
        total_negative = sum(
            i.impact_amount for i in impacts if i.impact_amount < Decimal("0")
        )
        total_positive = sum(
            i.impact_amount for i in impacts if i.impact_amount > Decimal("0")
        )

        # Apply confidence level scaling (higher confidence = larger VaR)
        confidence_factor = Decimal(str(1 + (confidence_level - 0.5) * 2))
        climate_var = (abs(total_negative) * confidence_factor).quantize(Decimal("0.01"))

        result = {
            "org_id": org_id,
            "climate_var": str(climate_var),
            "confidence_level": confidence_level,
            "total_negative_impacts": str(total_negative),
            "total_positive_impacts": str(total_positive),
            "net_exposure": str((total_negative + total_positive).quantize(Decimal("0.01"))),
            "impact_count": len(impacts),
            "provenance_hash": _sha256(f"{org_id}:cvar:{climate_var}:{confidence_level}"),
        }

        elapsed_ms = (_now() - start).total_seconds() * 1000
        logger.info(
            "Climate VaR for org=%s: $%.0f at %.0f%% confidence in %.1f ms",
            org_id, climate_var, confidence_level * 100, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Year-by-Year Projections
    # ------------------------------------------------------------------

    async def project_scenario_financials(
        self,
        org_id: str,
        scenario_result: Dict[str, Any],
        years: int = 10,
    ) -> Dict[str, Any]:
        """
        Project year-by-year financial impact under a scenario.

        Args:
            org_id: Organization ID.
            scenario_result: Base scenario with financial data.
            years: Number of years to project.

        Returns:
            Dict with year-by-year projections for each financial metric.
        """
        start = _now()

        base_revenue = Decimal(str(scenario_result.get("revenue", 0)))
        emissions = Decimal(str(scenario_result.get("emissions_tco2e", 0)))
        sector = scenario_result.get("sector", "default")
        scenario_type = scenario_result.get("scenario_type", "iea_aps")

        # Get carbon price trajectory
        try:
            s_type = ScenarioType(scenario_type)
        except ValueError:
            s_type = ScenarioType.IEA_APS

        carbon_prices = SCENARIO_CARBON_PRICES.get(s_type, {})
        template = FINANCIAL_IMPACT_TEMPLATES.get(sector, FINANCIAL_IMPACT_TEMPLATES["default"])

        projections: List[Dict[str, Any]] = []
        current_year = 2025

        for y in range(years):
            proj_year = current_year + y

            # Interpolate carbon price
            cp = self._interpolate_carbon_price(carbon_prices, proj_year)

            # Assume gradual emissions reduction (2% per year)
            year_emissions = (emissions * (Decimal("1") - Decimal("0.02") * Decimal(str(y)))).quantize(Decimal("0.01"))
            year_emissions = max(Decimal("0"), year_emissions)

            carbon_cost = (year_emissions * Decimal(str(cp))).quantize(Decimal("0.01"))

            # Revenue grows at 3% minus risk
            risk_factor = template["revenue_risk_pct"] / Decimal("100") * Decimal(str(y)) / Decimal(str(max(years, 1)))
            year_revenue = (
                base_revenue * (Decimal("1") + Decimal("0.03") * Decimal(str(y))) * (Decimal("1") - risk_factor)
            ).quantize(Decimal("0.01"))

            projections.append({
                "year": proj_year,
                "revenue": str(year_revenue),
                "carbon_cost": str(carbon_cost),
                "emissions_tco2e": str(year_emissions),
                "carbon_price": str(Decimal(str(cp))),
                "net_climate_impact": str((-carbon_cost).quantize(Decimal("0.01"))),
            })

        result = {
            "org_id": org_id,
            "scenario_type": scenario_type,
            "projection_years": years,
            "projections": projections,
            "provenance_hash": _sha256(f"{org_id}:proj:{years}:{scenario_type}"),
        }

        self._projections[org_id] = result

        elapsed_ms = (_now() - start).total_seconds() * 1000
        logger.info(
            "Financial projections for org=%s: %d years under %s in %.1f ms",
            org_id, years, scenario_type, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Adaptation ROI
    # ------------------------------------------------------------------

    async def calculate_adaptation_roi(
        self,
        investments: List[Dict[str, Any]],
        avoided_damages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate ROI for climate adaptation investments.

        Args:
            investments: List of dicts with 'name', 'cost_usd', 'year'.
            avoided_damages: List of dicts with 'name', 'avoided_usd', 'year'.

        Returns:
            Dict with total investment, total avoided damages, ROI, BCR.
        """
        start = _now()

        total_invest = sum(Decimal(str(i.get("cost_usd", 0))) for i in investments)
        total_avoided = sum(Decimal(str(d.get("avoided_usd", 0))) for d in avoided_damages)

        roi = Decimal("0")
        if total_invest > 0:
            roi = ((total_avoided - total_invest) / total_invest * Decimal("100")).quantize(Decimal("0.1"))

        bcr = Decimal("0")
        if total_invest > 0:
            bcr = (total_avoided / total_invest).quantize(Decimal("0.01"))

        result = {
            "total_investment": str(total_invest),
            "total_avoided_damages": str(total_avoided),
            "net_benefit": str((total_avoided - total_invest).quantize(Decimal("0.01"))),
            "roi_pct": str(roi),
            "benefit_cost_ratio": str(bcr),
            "investment_count": len(investments),
            "recommendation": "Strong investment case" if bcr > Decimal("1.5") else (
                "Marginal investment case" if bcr > Decimal("1") else "Investment case needs strengthening"
            ),
        }

        elapsed_ms = (_now() - start).total_seconds() * 1000
        logger.info(
            "Adaptation ROI: invest=$%.0f, avoided=$%.0f, ROI=%.1f%% in %.1f ms",
            total_invest, total_avoided, roi, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _store_impact(self, org_id: str, impact: FinancialImpact) -> None:
        """Store a financial impact in memory."""
        if org_id not in self._impacts:
            self._impacts[org_id] = []
        self._impacts[org_id].append(impact)

    @staticmethod
    def _interpolate_carbon_price(
        trajectory: Dict[int, int],
        year: int,
    ) -> float:
        """Linearly interpolate carbon price for a given year."""
        if not trajectory:
            return 0.0

        years = sorted(trajectory.keys())
        if year <= years[0]:
            return float(trajectory[years[0]])
        if year >= years[-1]:
            return float(trajectory[years[-1]])

        # Find bounding years
        for i in range(len(years) - 1):
            if years[i] <= year <= years[i + 1]:
                y1, y2 = years[i], years[i + 1]
                p1, p2 = float(trajectory[y1]), float(trajectory[y2])
                fraction = (year - y1) / (y2 - y1)
                return p1 + (p2 - p1) * fraction

        return float(trajectory[years[-1]])
