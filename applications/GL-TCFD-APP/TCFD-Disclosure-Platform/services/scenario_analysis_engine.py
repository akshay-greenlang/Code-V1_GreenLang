"""
Scenario Analysis Engine -- TCFD Strategy (c): Climate Scenario Analysis

Implements quantitative climate scenario analysis per TCFD Strategy (c)
and IFRS S2 paragraph 22 (climate resilience assessment).

Provides:
  - 8 pre-built climate scenarios (IEA NZE/APS/STEPS, NGFS x4, Custom)
  - Multi-scenario comparison across temperature pathways
  - Revenue, cost, asset impairment, and capex impact calculations
  - Sensitivity analysis on key parameters (tornado chart support)
  - Monte Carlo simulation for probability distributions
  - NPV and IRR calculations for climate investments
  - Marginal Abatement Cost Curve (MACC) generation
  - Climate resilience assessment per IFRS S2 para 22
  - Carbon price and energy mix trajectory projections
  - Year-by-year financial projection with discounting
  - Energy transition pathway analysis
  - Scenario-specific disclosure generation

All financial calculations are deterministic (zero-hallucination).
No LLM calls for numeric computations.

Reference:
    - TCFD Technical Supplement: Scenario Analysis (October 2020)
    - IFRS S2 Paragraph 22 (Climate Resilience)
    - IEA World Energy Outlook 2023
    - NGFS Climate Scenarios v4 (September 2022)

Example:
    >>> engine = ScenarioAnalysisEngine(config)
    >>> scenarios = await engine.get_prebuilt_scenarios()
    >>> result = await engine.run_scenario_analysis("org-1", scenarios[0].id, params)
    >>> comparison = await engine.run_multi_scenario_comparison("org-1", [s.id for s in scenarios], params)
"""

from __future__ import annotations

import logging
import random
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    SCENARIO_CARBON_PRICES,
    SCENARIO_LIBRARY,
    SCENARIO_TEMPERATURE,
    SECTOR_TRANSITION_PROFILES,
    ScenarioType,
    SectorType,
    TCFDAppConfig,
    TemperatureOutcome,
    TimeHorizon,
)
from .models import (
    ScenarioDefinition,
    ScenarioParameters,
    ScenarioResult,
    ScenarioComparison,
    SensitivityResult,
    RunScenarioAnalysisRequest,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector-Specific Revenue Sensitivity to Climate Scenarios
# ---------------------------------------------------------------------------

_SECTOR_REVENUE_SENSITIVITY: Dict[SectorType, Dict[str, Decimal]] = {
    SectorType.ENERGY: {
        "below_1_5c": Decimal("-15"),
        "around_2c": Decimal("-10"),
        "above_2_5c": Decimal("-5"),
        "above_3c": Decimal("-3"),
    },
    SectorType.TRANSPORTATION: {
        "below_1_5c": Decimal("-8"),
        "around_2c": Decimal("-6"),
        "above_2_5c": Decimal("-4"),
        "above_3c": Decimal("-2"),
    },
    SectorType.MATERIALS_BUILDINGS: {
        "below_1_5c": Decimal("-6"),
        "around_2c": Decimal("-5"),
        "above_2_5c": Decimal("-4"),
        "above_3c": Decimal("-3"),
    },
    SectorType.AGRICULTURE_FOOD_FOREST: {
        "below_1_5c": Decimal("-3"),
        "around_2c": Decimal("-5"),
        "above_2_5c": Decimal("-10"),
        "above_3c": Decimal("-18"),
    },
    SectorType.BANKING: {
        "below_1_5c": Decimal("-4"),
        "around_2c": Decimal("-5"),
        "above_2_5c": Decimal("-8"),
        "above_3c": Decimal("-12"),
    },
    SectorType.INSURANCE: {
        "below_1_5c": Decimal("-3"),
        "around_2c": Decimal("-6"),
        "above_2_5c": Decimal("-12"),
        "above_3c": Decimal("-20"),
    },
}

# ---------------------------------------------------------------------------
# Energy Transition Milestones by Scenario
# ---------------------------------------------------------------------------

_ENERGY_TRANSITION_MILESTONES: Dict[ScenarioType, List[Dict[str, Any]]] = {
    ScenarioType.IEA_NZE: [
        {"year": 2025, "milestone": "No new unabated coal plants approved"},
        {"year": 2030, "milestone": "60% of global car sales are EVs"},
        {"year": 2035, "milestone": "All new buildings zero-carbon-ready"},
        {"year": 2040, "milestone": "50% of heavy truck sales are electric/hydrogen"},
        {"year": 2050, "milestone": "Net zero CO2 emissions globally"},
    ],
    ScenarioType.IEA_APS: [
        {"year": 2030, "milestone": "35% of car sales are EVs"},
        {"year": 2035, "milestone": "Coal phase-down in advanced economies"},
        {"year": 2040, "milestone": "Renewables dominate power generation"},
        {"year": 2050, "milestone": "Advanced economies achieve net zero"},
    ],
    ScenarioType.IEA_STEPS: [
        {"year": 2030, "milestone": "Renewables reach 32% of energy mix"},
        {"year": 2040, "milestone": "Coal declines modestly in OECD"},
        {"year": 2050, "milestone": "Fossil fuels still 42% of energy mix"},
    ],
    ScenarioType.NGFS_DELAYED_TRANSITION: [
        {"year": 2030, "milestone": "Emissions plateau; no significant decline"},
        {"year": 2035, "milestone": "Abrupt policy shock with carbon price surge"},
        {"year": 2040, "milestone": "Rapid and disorderly transition begins"},
        {"year": 2050, "milestone": "Net zero achieved but at very high cost"},
    ],
}


class ScenarioAnalysisEngine:
    """
    Climate scenario analysis engine implementing TCFD Strategy (c) and
    IFRS S2 paragraph 22 climate resilience assessment.

    Manages pre-built and custom scenarios, runs quantitative impact
    analysis, sensitivity analysis, Monte Carlo simulation, and
    generates NPV/IRR and MACC outputs.

    Attributes:
        config: Application configuration.
        _scenarios: In-memory scenario definitions.
        _results: In-memory results keyed by org_id.
    """

    def __init__(self, config: Optional[TCFDAppConfig] = None) -> None:
        self.config = config or TCFDAppConfig()
        self._scenarios: Dict[str, ScenarioDefinition] = {}
        self._results: Dict[str, List[ScenarioResult]] = {}
        self._init_prebuilt_scenarios()
        logger.info("ScenarioAnalysisEngine initialized with %d pre-built scenarios", len(self._scenarios))

    def _init_prebuilt_scenarios(self) -> None:
        """Initialize pre-built IEA and NGFS scenarios from the library."""
        for scenario_type, lib_data in SCENARIO_LIBRARY.items():
            if scenario_type == ScenarioType.CUSTOM:
                continue
            params = ScenarioParameters(
                carbon_price_trajectory=lib_data.get("carbon_price_trajectory", {}),
                energy_mix_trajectory=lib_data.get("energy_mix_trajectory", {}),
                temperature_pathway=lib_data.get("temperature_projection", {}),
            )
            scenario = ScenarioDefinition(
                tenant_id="system",
                name=lib_data["name"],
                scenario_type=scenario_type,
                temperature_outcome=lib_data["temperature_outcome"],
                description=lib_data["description"],
                parameters=params,
                source="IEA WEO 2023 / NGFS Phase IV",
                is_custom=False,
            )
            self._scenarios[scenario.id] = scenario

    # ------------------------------------------------------------------
    # Scenario CRUD
    # ------------------------------------------------------------------

    async def get_prebuilt_scenarios(self) -> List[ScenarioDefinition]:
        """Return all pre-built scenario definitions."""
        return [s for s in self._scenarios.values() if not s.is_custom]

    async def get_scenario(self, scenario_id: str) -> ScenarioDefinition:
        """Retrieve a scenario by ID."""
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            raise ValueError(f"Scenario {scenario_id} not found")
        return scenario

    async def create_custom_scenario(
        self, tenant_id: str, name: str,
        carbon_prices: Dict[int, Decimal],
        energy_mix: Dict[int, Dict[str, int]],
        temperature: Dict[int, Decimal],
        description: str = "",
    ) -> ScenarioDefinition:
        """Create a user-defined custom scenario."""
        params = ScenarioParameters(
            carbon_price_trajectory=carbon_prices,
            energy_mix_trajectory=energy_mix,
            temperature_pathway=temperature,
        )
        scenario = ScenarioDefinition(
            tenant_id=tenant_id,
            name=name,
            scenario_type=ScenarioType.CUSTOM,
            temperature_outcome=TemperatureOutcome.AROUND_2C,
            description=description,
            parameters=params,
            is_custom=True,
        )
        self._scenarios[scenario.id] = scenario
        logger.info("Created custom scenario '%s' (id=%s)", name, scenario.id)
        return scenario

    async def list_all_scenarios(self) -> List[ScenarioDefinition]:
        """Return all scenarios (pre-built and custom)."""
        return list(self._scenarios.values())

    async def delete_custom_scenario(self, scenario_id: str) -> bool:
        """
        Delete a custom scenario.

        Args:
            scenario_id: Scenario to delete.

        Returns:
            True if deleted, False if not found or not custom.
        """
        scenario = self._scenarios.get(scenario_id)
        if scenario is None or not scenario.is_custom:
            return False
        del self._scenarios[scenario_id]
        logger.info("Deleted custom scenario %s", scenario_id)
        return True

    # ------------------------------------------------------------------
    # Core Scenario Analysis
    # ------------------------------------------------------------------

    async def run_scenario_analysis(
        self, org_id: str, scenario_id: str,
        params: RunScenarioAnalysisRequest,
    ) -> ScenarioResult:
        """
        Run a full scenario analysis for an organization.

        Calculates revenue impact, cost impact, asset impairment,
        capex requirements, carbon costs, and NPV under the given scenario.

        Args:
            org_id: Organization identifier.
            scenario_id: Scenario definition ID.
            params: Analysis parameters (emissions, revenue, costs, etc.).

        Returns:
            ScenarioResult with full quantitative impact assessment.
        """
        start = datetime.utcnow()
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            raise ValueError(f"Scenario {scenario_id} not found")

        lib = scenario.parameters
        carbon_prices = lib.carbon_price_trajectory if lib else {}
        energy_mix = lib.energy_mix_trajectory if lib else {}

        carbon_price_2030 = carbon_prices.get(2030, Decimal("50"))
        carbon_price_2050 = carbon_prices.get(2050, Decimal("100"))

        total_emissions = params.emissions_scope1_tco2e + params.emissions_scope2_tco2e
        annual_carbon_cost = total_emissions * carbon_price_2030

        revenue_impact_pct = self._calc_revenue_impact(scenario, params)
        cost_impact_pct = self._calc_cost_impact(scenario, params, annual_carbon_cost)
        asset_impairment_pct = self._calc_asset_impairment(scenario)
        capex_required = self._calc_capex_required(scenario, params)

        stranded_value = params.total_assets_usd * (asset_impairment_pct / Decimal("100"))

        discount_rate = params.discount_rate or self.config.default_discount_rate
        npv = self._calc_npv(
            revenue_impact_pct, cost_impact_pct, params.revenue_base_usd,
            params.cost_base_usd, capex_required, discount_rate,
        )

        ci_lower = npv * Decimal("0.7")
        ci_upper = npv * Decimal("1.3")

        if params.enable_monte_carlo and self.config.enable_monte_carlo:
            ci_lower, ci_upper = self._run_monte_carlo_quick(
                npv, revenue_impact_pct, cost_impact_pct,
            )

        assumptions = self._build_assumptions(scenario, params)

        narrative = self._build_narrative(
            scenario, revenue_impact_pct, cost_impact_pct,
            annual_carbon_cost, stranded_value,
        )

        result = ScenarioResult(
            tenant_id="default",
            scenario_id=scenario_id,
            org_id=org_id,
            analysis_date=date.today(),
            revenue_impact_pct=revenue_impact_pct,
            cost_impact_pct=cost_impact_pct,
            asset_impairment_pct=asset_impairment_pct,
            capex_required_usd=capex_required,
            npv_usd=npv,
            carbon_cost_annual_usd=annual_carbon_cost,
            stranded_asset_value_usd=stranded_value,
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            key_assumptions=assumptions,
            narrative=narrative,
        )

        if org_id not in self._results:
            self._results[org_id] = []
        self._results[org_id].append(result)

        elapsed = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Scenario analysis for org %s (%s): NPV=$%.0f, rev=%.1f%%, cost=%.1f%% in %.1f ms",
            org_id, scenario.name, npv, revenue_impact_pct, cost_impact_pct, elapsed,
        )
        return result

    async def run_multi_scenario_comparison(
        self, org_id: str, scenario_ids: List[str],
        params: RunScenarioAnalysisRequest,
    ) -> ScenarioComparison:
        """Run the same analysis across multiple scenarios and compare."""
        results: List[ScenarioResult] = []
        for sid in scenario_ids:
            result = await self.run_scenario_analysis(org_id, sid, params)
            results.append(result)

        best = max(results, key=lambda r: r.npv_usd) if results else None
        worst = min(results, key=lambda r: r.npv_usd) if results else None

        summary_parts: List[str] = []
        for r in results:
            scenario = self._scenarios.get(r.scenario_id)
            name = scenario.name if scenario else r.scenario_id
            summary_parts.append(f"{name}: NPV=${r.npv_usd:,.0f}")

        comparison = ScenarioComparison(
            tenant_id="default",
            org_id=org_id,
            scenario_results=results,
            most_resilient_scenario=best.scenario_id if best else None,
            highest_risk_scenario=worst.scenario_id if worst else None,
            summary="; ".join(summary_parts),
        )

        logger.info(
            "Multi-scenario comparison for org %s: %d scenarios compared",
            org_id, len(results),
        )
        return comparison

    # ------------------------------------------------------------------
    # Year-by-Year Financial Projection
    # ------------------------------------------------------------------

    async def run_yearly_projection(
        self, org_id: str, scenario_id: str,
        params: RunScenarioAnalysisRequest,
        start_year: int = 2025,
        end_year: int = 2050,
    ) -> Dict[str, Any]:
        """
        Generate year-by-year financial projection under a scenario.

        Interpolates carbon prices, applies progressive revenue and cost
        impacts, and calculates annual NPV contributions.

        Args:
            org_id: Organization identifier.
            scenario_id: Scenario definition ID.
            params: Analysis parameters.
            start_year: Projection start year.
            end_year: Projection end year.

        Returns:
            Dict with year-by-year projections and cumulative metrics.
        """
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            raise ValueError(f"Scenario {scenario_id} not found")

        lib = scenario.parameters
        carbon_prices = lib.carbon_price_trajectory if lib else {}
        discount_rate = params.discount_rate or self.config.default_discount_rate

        total_emissions = params.emissions_scope1_tco2e + params.emissions_scope2_tco2e
        base_revenue = params.revenue_base_usd
        base_cost = params.cost_base_usd

        # Calculate full scenario impacts as a reference
        base_revenue_impact = self._calc_revenue_impact(scenario, params)
        base_cost_impact = self._calc_cost_impact(
            scenario, params, total_emissions * carbon_prices.get(2030, Decimal("50")),
        )

        yearly_data: List[Dict[str, Any]] = []
        cumulative_npv = Decimal("0")
        cumulative_carbon_cost = Decimal("0")

        total_years = end_year - start_year
        if total_years <= 0:
            total_years = 1

        for year in range(start_year, end_year + 1):
            t = year - start_year
            progress = Decimal(str(t)) / Decimal(str(total_years))

            # Interpolate carbon price for this year
            carbon_price = self._interpolate_value(year, carbon_prices)
            annual_carbon_cost = total_emissions * carbon_price

            # Progressive impact (ramps linearly from 0 to full impact)
            year_revenue_impact = base_revenue_impact * progress
            year_cost_impact = base_cost_impact * progress

            revenue_change = base_revenue * year_revenue_impact / Decimal("100")
            cost_change = base_cost * year_cost_impact / Decimal("100")
            net_annual_impact = revenue_change - cost_change - annual_carbon_cost

            # Discount
            discount_factor = (Decimal("1") + discount_rate) ** t
            pv_impact = Decimal("0")
            if discount_factor != 0:
                pv_impact = (net_annual_impact / discount_factor).quantize(Decimal("0.01"))
            cumulative_npv += pv_impact
            cumulative_carbon_cost += annual_carbon_cost

            # Emissions trajectory (assumes linear reduction under ambitious scenarios)
            temp = SCENARIO_TEMPERATURE.get(scenario.scenario_type, 2.0)
            if temp <= 1.5:
                emissions_factor = max(Decimal("1") - progress * Decimal("0.7"), Decimal("0.3"))
            elif temp <= 2.0:
                emissions_factor = max(Decimal("1") - progress * Decimal("0.5"), Decimal("0.5"))
            else:
                emissions_factor = max(Decimal("1") - progress * Decimal("0.2"), Decimal("0.8"))

            projected_emissions = (total_emissions * emissions_factor).quantize(Decimal("0.01"))

            yearly_data.append({
                "year": year,
                "carbon_price_usd": str(carbon_price),
                "annual_carbon_cost": str(annual_carbon_cost.quantize(Decimal("0.01"))),
                "revenue_impact_pct": str(year_revenue_impact.quantize(Decimal("0.1"))),
                "cost_impact_pct": str(year_cost_impact.quantize(Decimal("0.1"))),
                "net_annual_impact": str(net_annual_impact.quantize(Decimal("0.01"))),
                "pv_impact": str(pv_impact),
                "cumulative_npv": str(cumulative_npv.quantize(Decimal("0.01"))),
                "projected_emissions_tco2e": str(projected_emissions),
            })

        return {
            "org_id": org_id,
            "scenario_id": scenario_id,
            "scenario_name": scenario.name,
            "start_year": start_year,
            "end_year": end_year,
            "discount_rate": str(discount_rate),
            "projection_years": len(yearly_data),
            "cumulative_npv": str(cumulative_npv.quantize(Decimal("0.01"))),
            "cumulative_carbon_cost": str(cumulative_carbon_cost.quantize(Decimal("0.01"))),
            "yearly_data": yearly_data,
        }

    # ------------------------------------------------------------------
    # Sensitivity Analysis
    # ------------------------------------------------------------------

    async def run_sensitivity_analysis(
        self, org_id: str, scenario_id: str,
        params: RunScenarioAnalysisRequest,
        variable: str = "carbon_price",
        variations: Optional[List[Decimal]] = None,
    ) -> List[SensitivityResult]:
        """Run sensitivity analysis on a specific variable."""
        if variations is None:
            variations = [Decimal("-50"), Decimal("-25"), Decimal("0"), Decimal("25"), Decimal("50")]

        base_result = await self.run_scenario_analysis(org_id, scenario_id, params)
        results: List[SensitivityResult] = []

        for pct_change in variations:
            factor = Decimal("1") + pct_change / Decimal("100")
            adjusted_npv = base_result.npv_usd * factor
            adjusted_revenue = base_result.revenue_impact_pct * factor
            adjusted_cost = base_result.cost_impact_pct * factor

            sr = SensitivityResult(
                tenant_id="default",
                scenario_id=scenario_id,
                variable_name=variable,
                base_value=base_result.npv_usd,
                tested_value=adjusted_npv,
                change_pct=pct_change,
                npv_impact_usd=adjusted_npv - base_result.npv_usd,
                revenue_impact_pct=adjusted_revenue,
                cost_impact_pct=adjusted_cost,
            )
            results.append(sr)

        logger.info(
            "Sensitivity analysis for org %s on %s: %d variations",
            org_id, variable, len(results),
        )
        return results

    async def run_multi_variable_sensitivity(
        self, org_id: str, scenario_id: str,
        params: RunScenarioAnalysisRequest,
        variables: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run sensitivity analysis on multiple variables for tornado chart.

        Args:
            org_id: Organization identifier.
            scenario_id: Scenario ID.
            params: Analysis parameters.
            variables: List of variables to test. Defaults to
                [carbon_price, emissions, revenue, discount_rate].

        Returns:
            Dict with per-variable sensitivity ranges for tornado chart.
        """
        if variables is None:
            variables = ["carbon_price", "emissions", "revenue", "discount_rate"]

        base_result = await self.run_scenario_analysis(org_id, scenario_id, params)
        base_npv = base_result.npv_usd

        tornado_data: List[Dict[str, Any]] = []
        test_range = [Decimal("-30"), Decimal("30")]

        for variable in variables:
            low_factor = Decimal("1") + test_range[0] / Decimal("100")
            high_factor = Decimal("1") + test_range[1] / Decimal("100")

            low_npv = base_npv * low_factor
            high_npv = base_npv * high_factor

            # For some variables, the direction is inverted
            if variable in ("emissions", "discount_rate"):
                low_npv, high_npv = high_npv, low_npv

            tornado_data.append({
                "variable": variable,
                "low_case_npv": str(low_npv.quantize(Decimal("0.01"))),
                "base_case_npv": str(base_npv.quantize(Decimal("0.01"))),
                "high_case_npv": str(high_npv.quantize(Decimal("0.01"))),
                "range": str((high_npv - low_npv).quantize(Decimal("0.01"))),
                "low_pct_change": str(test_range[0]),
                "high_pct_change": str(test_range[1]),
            })

        # Sort by range (widest first)
        tornado_data.sort(key=lambda x: abs(Decimal(x["range"])), reverse=True)

        return {
            "org_id": org_id,
            "scenario_id": scenario_id,
            "base_npv": str(base_npv.quantize(Decimal("0.01"))),
            "variables_tested": len(tornado_data),
            "tornado_data": tornado_data,
            "most_sensitive_variable": tornado_data[0]["variable"] if tornado_data else None,
        }

    # ------------------------------------------------------------------
    # Monte Carlo Simulation
    # ------------------------------------------------------------------

    async def run_monte_carlo(
        self, org_id: str, scenario_id: str,
        params: RunScenarioAnalysisRequest,
        iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for uncertainty quantification."""
        n = iterations or self.config.monte_carlo_iterations
        base_result = await self.run_scenario_analysis(org_id, scenario_id, params)
        base_npv = float(base_result.npv_usd)

        random.seed(42)
        samples: List[float] = []
        for _ in range(n):
            revenue_var = random.gauss(1.0, 0.15)
            cost_var = random.gauss(1.0, 0.10)
            carbon_var = random.gauss(1.0, 0.20)
            sample_npv = base_npv * revenue_var * cost_var * carbon_var
            samples.append(sample_npv)

        samples.sort()
        p5 = samples[int(n * 0.05)]
        p10 = samples[int(n * 0.10)]
        p25 = samples[int(n * 0.25)]
        p50 = samples[int(n * 0.50)]
        p75 = samples[int(n * 0.75)]
        p90 = samples[int(n * 0.90)]
        p95 = samples[int(n * 0.95)]
        mean_val = sum(samples) / n

        # Standard deviation
        variance = sum((s - mean_val) ** 2 for s in samples) / n
        std_dev = variance ** 0.5

        logger.info(
            "Monte Carlo for org %s: %d iterations, mean=$%.0f, p5=$%.0f, p95=$%.0f",
            org_id, n, mean_val, p5, p95,
        )

        return {
            "org_id": org_id,
            "scenario_id": scenario_id,
            "iterations": n,
            "mean_npv": str(Decimal(str(round(mean_val, 2)))),
            "std_dev": str(Decimal(str(round(std_dev, 2)))),
            "percentiles": {
                "p5": str(Decimal(str(round(p5, 2)))),
                "p10": str(Decimal(str(round(p10, 2)))),
                "p25": str(Decimal(str(round(p25, 2)))),
                "p50": str(Decimal(str(round(p50, 2)))),
                "p75": str(Decimal(str(round(p75, 2)))),
                "p90": str(Decimal(str(round(p90, 2)))),
                "p95": str(Decimal(str(round(p95, 2)))),
            },
            "probability_of_loss": str(
                Decimal(str(round(sum(1 for s in samples if s < 0) / n * 100, 1)))
            ),
            "value_at_risk_5pct": str(Decimal(str(round(abs(p5), 2)))),
        }

    # ------------------------------------------------------------------
    # NPV, IRR, and MACC
    # ------------------------------------------------------------------

    async def calculate_npv(
        self, cash_flows: Dict[int, Decimal],
        discount_rate: Optional[Decimal] = None,
    ) -> Decimal:
        """Calculate NPV from a series of cash flows."""
        rate = discount_rate or self.config.default_discount_rate
        npv = Decimal("0")
        base_year = min(cash_flows.keys()) if cash_flows else 2025
        for year, cf in sorted(cash_flows.items()):
            t = year - base_year
            if t < 0:
                continue
            discount_factor = (Decimal("1") + rate) ** t
            npv += cf / discount_factor
        return npv.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    async def calculate_irr(
        self, cash_flows: Dict[int, Decimal],
    ) -> Optional[Decimal]:
        """Calculate approximate IRR using bisection method."""
        sorted_flows = [cf for _, cf in sorted(cash_flows.items())]
        if not sorted_flows or all(cf >= 0 for cf in sorted_flows):
            return None

        low, high = Decimal("-0.5"), Decimal("2.0")
        for _ in range(100):
            mid = (low + high) / 2
            npv = Decimal("0")
            for t, cf in enumerate(sorted_flows):
                denom = (Decimal("1") + mid) ** t
                if denom != 0:
                    npv += cf / denom
            if abs(npv) < Decimal("0.01"):
                return (mid * 100).quantize(Decimal("0.1"))
            if npv > 0:
                low = mid
            else:
                high = mid

        return ((low + high) / 2 * 100).quantize(Decimal("0.1"))

    async def generate_macc(
        self, org_id: str,
        abatement_measures: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate Marginal Abatement Cost Curve data."""
        macc: List[Dict[str, Any]] = []
        for measure in abatement_measures:
            abatement = Decimal(str(measure.get("abatement_tco2e", 0)))
            cost = Decimal(str(measure.get("cost_usd", 0)))
            cost_per_tonne = Decimal("0")
            if abatement > 0:
                cost_per_tonne = (cost / abatement).quantize(Decimal("0.01"))

            macc.append({
                "measure": measure.get("name", "Unknown"),
                "abatement_tco2e": str(abatement),
                "cost_per_tco2e": str(cost_per_tonne),
                "total_cost_usd": str(cost),
                "category": measure.get("category", "other"),
            })

        macc.sort(key=lambda x: Decimal(x["cost_per_tco2e"]))

        cumulative = Decimal("0")
        for entry in macc:
            cumulative += Decimal(entry["abatement_tco2e"])
            entry["cumulative_abatement_tco2e"] = str(cumulative)

        logger.info("MACC generated for org %s: %d measures", org_id, len(macc))
        return macc

    async def generate_macc_with_scenario_context(
        self, org_id: str,
        abatement_measures: List[Dict[str, Any]],
        scenario_id: str,
    ) -> Dict[str, Any]:
        """
        Generate MACC with scenario-specific carbon price overlay.

        Shows which abatement measures are cost-effective under the
        selected scenario's carbon price trajectory.

        Args:
            org_id: Organization identifier.
            abatement_measures: List of abatement measures.
            scenario_id: Scenario for carbon price reference.

        Returns:
            Dict with MACC data plus cost-effectiveness analysis.
        """
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            raise ValueError(f"Scenario {scenario_id} not found")

        macc = await self.generate_macc(org_id, abatement_measures)

        lib = scenario.parameters
        carbon_prices = lib.carbon_price_trajectory if lib else {}
        price_2030 = carbon_prices.get(2030, Decimal("50"))
        price_2050 = carbon_prices.get(2050, Decimal("100"))

        cost_effective_2030: List[Dict[str, Any]] = []
        cost_effective_2050: List[Dict[str, Any]] = []
        total_abatement_below_2030 = Decimal("0")
        total_abatement_below_2050 = Decimal("0")

        for entry in macc:
            cost_per_tonne = Decimal(entry["cost_per_tco2e"])
            abatement = Decimal(entry["abatement_tco2e"])

            if cost_per_tonne <= price_2030:
                cost_effective_2030.append(entry)
                total_abatement_below_2030 += abatement

            if cost_per_tonne <= price_2050:
                cost_effective_2050.append(entry)
                total_abatement_below_2050 += abatement

        return {
            "org_id": org_id,
            "scenario_id": scenario_id,
            "scenario_name": scenario.name,
            "carbon_price_2030": str(price_2030),
            "carbon_price_2050": str(price_2050),
            "total_measures": len(macc),
            "macc": macc,
            "cost_effective_at_2030_price": {
                "measure_count": len(cost_effective_2030),
                "total_abatement_tco2e": str(total_abatement_below_2030),
            },
            "cost_effective_at_2050_price": {
                "measure_count": len(cost_effective_2050),
                "total_abatement_tco2e": str(total_abatement_below_2050),
            },
        }

    # ------------------------------------------------------------------
    # Climate Resilience Assessment (IFRS S2 Para 22)
    # ------------------------------------------------------------------

    async def assess_climate_resilience(
        self, org_id: str,
    ) -> Dict[str, Any]:
        """
        Assess climate resilience per IFRS S2 paragraph 22.

        Evaluates the organization's strategic resilience across
        all analyzed scenarios using NPV spread, positive outcome
        ratio, and scenario diversity.
        """
        results = self._results.get(org_id, [])
        if not results:
            return {
                "org_id": org_id,
                "resilience_assessment": "No scenarios analyzed yet",
                "scenarios_assessed": 0,
            }

        npvs = [float(r.npv_usd) for r in results]
        avg_npv = sum(npvs) / len(npvs)
        worst_npv = min(npvs)
        best_npv = max(npvs)
        spread = best_npv - worst_npv

        positive_count = sum(1 for n in npvs if n >= 0)
        resilience_ratio = positive_count / len(npvs) if npvs else 0

        if resilience_ratio >= 0.8:
            rating = "high"
        elif resilience_ratio >= 0.5:
            rating = "moderate"
        else:
            rating = "low"

        # Additional resilience metrics
        scenario_types_covered = set()
        for r in results:
            scenario = self._scenarios.get(r.scenario_id)
            if scenario:
                scenario_types_covered.add(scenario.scenario_type.value)

        # Temperature diversity (how many different temperature outcomes?)
        temp_outcomes = set()
        for r in results:
            scenario = self._scenarios.get(r.scenario_id)
            if scenario:
                temp_outcomes.add(scenario.temperature_outcome.value)

        # Scenario diversity score (0-100)
        diversity_score = min(len(scenario_types_covered) * 15, 100)

        logger.info(
            "Climate resilience for org %s: rating=%s, ratio=%.1f%%",
            org_id, rating, resilience_ratio * 100,
        )

        return {
            "org_id": org_id,
            "scenarios_assessed": len(results),
            "resilience_rating": rating,
            "resilience_ratio": round(resilience_ratio, 2),
            "average_npv": str(Decimal(str(round(avg_npv, 2)))),
            "worst_case_npv": str(Decimal(str(round(worst_npv, 2)))),
            "best_case_npv": str(Decimal(str(round(best_npv, 2)))),
            "npv_spread": str(Decimal(str(round(spread, 2)))),
            "positive_outcome_scenarios": positive_count,
            "scenario_types_covered": list(scenario_types_covered),
            "temperature_outcomes_covered": list(temp_outcomes),
            "scenario_diversity_score": diversity_score,
            "ifrs_s2_para_22": (
                f"Under {len(results)} climate scenario(s), the organization demonstrates "
                f"'{rating}' resilience with {positive_count}/{len(results)} scenarios "
                f"yielding positive NPV outcomes. "
                f"Scenarios span {len(temp_outcomes)} temperature outcome(s) and "
                f"{len(scenario_types_covered)} scenario type(s)."
            ),
        }

    # ------------------------------------------------------------------
    # Trajectory Lookups
    # ------------------------------------------------------------------

    async def get_carbon_price_trajectory(
        self, scenario_id: str,
    ) -> Dict[int, str]:
        """Get carbon price trajectory for a scenario."""
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            raise ValueError(f"Scenario {scenario_id} not found")
        prices = scenario.parameters.carbon_price_trajectory if scenario.parameters else {}
        return {yr: str(price) for yr, price in sorted(prices.items())}

    async def get_energy_mix_trajectory(
        self, scenario_id: str,
    ) -> Dict[int, Dict[str, int]]:
        """Get energy mix trajectory for a scenario."""
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            raise ValueError(f"Scenario {scenario_id} not found")
        return scenario.parameters.energy_mix_trajectory if scenario.parameters else {}

    async def get_temperature_pathway(
        self, scenario_id: str,
    ) -> Dict[int, str]:
        """Get temperature pathway for a scenario."""
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            raise ValueError(f"Scenario {scenario_id} not found")
        pathway = scenario.parameters.temperature_pathway if scenario.parameters else {}
        return {yr: str(temp) for yr, temp in sorted(pathway.items())}

    async def get_energy_transition_milestones(
        self, scenario_type: ScenarioType,
    ) -> List[Dict[str, Any]]:
        """
        Get energy transition milestones for a scenario type.

        Args:
            scenario_type: Scenario type to look up milestones for.

        Returns:
            List of milestone dicts with year and description.
        """
        return _ENERGY_TRANSITION_MILESTONES.get(scenario_type, [])

    async def compare_trajectories(
        self, scenario_ids: List[str],
        metric: str = "carbon_price",
    ) -> Dict[str, Any]:
        """
        Compare a specific trajectory metric across multiple scenarios.

        Args:
            scenario_ids: List of scenario IDs to compare.
            metric: One of "carbon_price", "energy_mix", "temperature".

        Returns:
            Dict with side-by-side trajectory comparison.
        """
        comparison: Dict[str, Dict[int, str]] = {}

        for sid in scenario_ids:
            scenario = self._scenarios.get(sid)
            if scenario is None:
                continue

            name = scenario.name
            lib = scenario.parameters
            if lib is None:
                continue

            if metric == "carbon_price":
                data = {yr: str(val) for yr, val in sorted(lib.carbon_price_trajectory.items())}
            elif metric == "temperature":
                data = {yr: str(val) for yr, val in sorted(lib.temperature_pathway.items())}
            elif metric == "energy_mix":
                data = {}
                for yr, mix in sorted(lib.energy_mix_trajectory.items()):
                    renewable_pct = mix.get("renewable_pct", 0)
                    data[yr] = str(renewable_pct)
            else:
                data = {}

            comparison[name] = data

        return {
            "metric": metric,
            "scenarios_compared": len(comparison),
            "trajectories": comparison,
        }

    # ------------------------------------------------------------------
    # Sector-Specific Scenario Analysis
    # ------------------------------------------------------------------

    async def run_sector_scenario_analysis(
        self, org_id: str, scenario_id: str,
        params: RunScenarioAnalysisRequest,
        sector: SectorType,
    ) -> Dict[str, Any]:
        """
        Run scenario analysis with sector-specific adjustments.

        Applies sector transition profiles and revenue sensitivity
        factors to produce more granular sector-level projections.

        Args:
            org_id: Organization identifier.
            scenario_id: Scenario definition ID.
            params: Analysis parameters.
            sector: TCFD sector classification.

        Returns:
            Dict with sector-adjusted scenario results.
        """
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            raise ValueError(f"Scenario {scenario_id} not found")

        # Base analysis
        base_result = await self.run_scenario_analysis(org_id, scenario_id, params)

        # Get sector profile
        profile = SECTOR_TRANSITION_PROFILES.get(sector, {})
        exposure_level = profile.get("transition_exposure", "medium")
        stranding_risk = profile.get("stranding_risk", "medium")

        # Sector-specific revenue sensitivity
        sector_sensitivity = _SECTOR_REVENUE_SENSITIVITY.get(sector, {})
        temp_key = scenario.temperature_outcome.value
        sector_revenue_adj = sector_sensitivity.get(temp_key, Decimal("0"))

        # Apply sector adjustment
        adjusted_revenue_impact = base_result.revenue_impact_pct + sector_revenue_adj

        # Stranding risk multiplier
        stranding_multiplier = {
            "low": Decimal("0.5"),
            "medium": Decimal("1.0"),
            "high": Decimal("1.5"),
            "very_high": Decimal("2.0"),
        }
        asset_impairment_adj = base_result.asset_impairment_pct * stranding_multiplier.get(
            stranding_risk, Decimal("1.0"),
        )

        return {
            "org_id": org_id,
            "scenario_id": scenario_id,
            "scenario_name": scenario.name,
            "sector": sector.value,
            "base_revenue_impact_pct": str(base_result.revenue_impact_pct),
            "sector_adjusted_revenue_impact_pct": str(adjusted_revenue_impact),
            "sector_revenue_adjustment": str(sector_revenue_adj),
            "base_asset_impairment_pct": str(base_result.asset_impairment_pct),
            "sector_adjusted_asset_impairment_pct": str(asset_impairment_adj.quantize(Decimal("0.1"))),
            "sector_transition_exposure": exposure_level,
            "sector_stranding_risk": stranding_risk,
            "sector_decarbonization_pathway": profile.get("decarbonization_pathway", ""),
            "npv_usd": str(base_result.npv_usd),
            "carbon_cost_annual": str(base_result.carbon_cost_annual_usd),
        }

    # ------------------------------------------------------------------
    # Scenario Disclosure Generation
    # ------------------------------------------------------------------

    async def generate_scenario_disclosure(
        self, org_id: str,
    ) -> Dict[str, Any]:
        """
        Generate TCFD Strategy (c) scenario analysis disclosure.

        Summarizes all scenarios analyzed, key findings, and resilience
        assessment for TCFD and IFRS S2 reporting.

        Args:
            org_id: Organization identifier.

        Returns:
            Dict with Strategy (c) disclosure content and compliance score.
        """
        results = self._results.get(org_id, [])
        if not results:
            return {
                "org_id": org_id,
                "content": "No scenario analyses have been conducted.",
                "compliance_score": 0,
            }

        # Gather scenario details
        scenario_summaries: List[str] = []
        for r in results:
            scenario = self._scenarios.get(r.scenario_id)
            name = scenario.name if scenario else "Custom"
            temp = scenario.temperature_outcome.value if scenario else "unknown"
            scenario_summaries.append(
                f"{name} ({temp}): NPV impact ${r.npv_usd:,.0f}, "
                f"revenue impact {r.revenue_impact_pct}%, "
                f"annual carbon cost ${r.carbon_cost_annual_usd:,.0f}"
            )

        # Resilience assessment
        resilience = await self.assess_climate_resilience(org_id)

        # Build narrative
        narrative_parts = [
            f"The organization has conducted climate scenario analysis using "
            f"{len(results)} scenario(s). ",
        ]
        for summary in scenario_summaries:
            narrative_parts.append(f"  - {summary}. ")

        narrative_parts.append(
            f"Overall climate resilience is rated as "
            f"'{resilience.get('resilience_rating', 'unknown')}' "
            f"with {resilience.get('positive_outcome_scenarios', 0)}/{len(results)} "
            f"scenarios yielding positive NPV outcomes."
        )

        # Compliance scoring
        compliance_score = self._score_scenario_disclosure(results, resilience)

        return {
            "org_id": org_id,
            "ref": "Strategy (c)",
            "title": "Scenario Analysis and Climate Resilience",
            "content": "".join(narrative_parts),
            "scenario_count": len(results),
            "scenario_summaries": scenario_summaries,
            "resilience_rating": resilience.get("resilience_rating"),
            "compliance_score": compliance_score,
        }

    # ------------------------------------------------------------------
    # Private Calculation Methods (Zero-Hallucination)
    # ------------------------------------------------------------------

    def _calc_revenue_impact(
        self, scenario: ScenarioDefinition,
        params: RunScenarioAnalysisRequest,
    ) -> Decimal:
        """Calculate revenue impact percentage under scenario."""
        temp = SCENARIO_TEMPERATURE.get(scenario.scenario_type, 2.0)
        if temp <= 1.5:
            base_impact = Decimal("-5")
        elif temp <= 2.0:
            base_impact = Decimal("-8")
        elif temp <= 2.5:
            base_impact = Decimal("-12")
        else:
            base_impact = Decimal("-18")

        if params.emissions_scope1_tco2e > 0 and params.revenue_base_usd > 0:
            intensity = params.emissions_scope1_tco2e / (params.revenue_base_usd / Decimal("1000000"))
            if intensity > Decimal("100"):
                base_impact *= Decimal("1.3")
            elif intensity > Decimal("50"):
                base_impact *= Decimal("1.1")

        return base_impact.quantize(Decimal("0.1"))

    def _calc_cost_impact(
        self, scenario: ScenarioDefinition,
        params: RunScenarioAnalysisRequest,
        annual_carbon_cost: Decimal,
    ) -> Decimal:
        """Calculate cost impact percentage under scenario."""
        if params.cost_base_usd <= 0:
            return Decimal("0")

        carbon_pct = (annual_carbon_cost / params.cost_base_usd * 100).quantize(Decimal("0.1"))

        temp = SCENARIO_TEMPERATURE.get(scenario.scenario_type, 2.0)
        if temp <= 1.5:
            adaptation_pct = Decimal("2")
        elif temp <= 2.0:
            adaptation_pct = Decimal("4")
        else:
            adaptation_pct = Decimal("8")

        return (carbon_pct + adaptation_pct).quantize(Decimal("0.1"))

    def _calc_asset_impairment(self, scenario: ScenarioDefinition) -> Decimal:
        """Calculate asset impairment percentage."""
        temp = SCENARIO_TEMPERATURE.get(scenario.scenario_type, 2.0)
        if temp <= 1.5:
            return Decimal("3.0")
        elif temp <= 2.0:
            return Decimal("6.0")
        elif temp <= 2.5:
            return Decimal("10.0")
        else:
            return Decimal("18.0")

    def _calc_capex_required(
        self, scenario: ScenarioDefinition,
        params: RunScenarioAnalysisRequest,
    ) -> Decimal:
        """Calculate capex required for climate adaptation."""
        temp = SCENARIO_TEMPERATURE.get(scenario.scenario_type, 2.0)
        if temp <= 1.5:
            capex_pct = Decimal("0.05")
        elif temp <= 2.0:
            capex_pct = Decimal("0.03")
        else:
            capex_pct = Decimal("0.02")

        base = params.total_assets_usd if params.total_assets_usd > 0 else params.revenue_base_usd
        return (base * capex_pct).quantize(Decimal("0.01"))

    def _calc_npv(
        self, revenue_pct: Decimal, cost_pct: Decimal,
        revenue_base: Decimal, cost_base: Decimal,
        capex: Decimal, discount_rate: Decimal,
    ) -> Decimal:
        """Calculate NPV of climate impacts over projection horizon."""
        years = self.config.financial_projection_years
        annual_revenue_impact = revenue_base * revenue_pct / Decimal("100")
        annual_cost_impact = cost_base * cost_pct / Decimal("100")
        annual_net = annual_revenue_impact - annual_cost_impact

        npv = -capex
        for t in range(1, years + 1):
            discount_factor = (Decimal("1") + discount_rate) ** t
            npv += annual_net / discount_factor

        return npv.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _run_monte_carlo_quick(
        self, base_npv: Decimal,
        revenue_pct: Decimal, cost_pct: Decimal,
    ) -> Tuple[Decimal, Decimal]:
        """Quick Monte Carlo for confidence interval estimation."""
        n = min(self.config.monte_carlo_iterations, 5000)
        random.seed(42)
        base = float(base_npv)
        samples: List[float] = []

        for _ in range(n):
            factor = random.gauss(1.0, 0.15)
            samples.append(base * factor)

        samples.sort()
        p5 = Decimal(str(round(samples[int(n * 0.05)], 2)))
        p95 = Decimal(str(round(samples[int(n * 0.95)], 2)))
        return p5, p95

    @staticmethod
    def _build_assumptions(
        scenario: ScenarioDefinition,
        params: RunScenarioAnalysisRequest,
    ) -> List[str]:
        """Build list of key assumptions for the analysis."""
        assumptions = [
            f"Scenario: {scenario.name}",
            f"Temperature outcome: {scenario.temperature_outcome.value}",
            f"Revenue base: ${params.revenue_base_usd:,.0f}",
            f"Cost base: ${params.cost_base_usd:,.0f}",
            f"Total assets: ${params.total_assets_usd:,.0f}",
            f"Scope 1 emissions: {params.emissions_scope1_tco2e:,.0f} tCO2e",
            f"Scope 2 emissions: {params.emissions_scope2_tco2e:,.0f} tCO2e",
        ]
        return assumptions

    @staticmethod
    def _build_narrative(
        scenario: ScenarioDefinition,
        revenue_pct: Decimal, cost_pct: Decimal,
        carbon_cost: Decimal, stranded_value: Decimal,
    ) -> str:
        """Build scenario narrative summary."""
        return (
            f"Under the {scenario.name} scenario "
            f"(temperature outcome: {scenario.temperature_outcome.value}), "
            f"the organization faces a {abs(revenue_pct):.1f}% revenue impact "
            f"and {cost_pct:.1f}% cost increase. "
            f"Annual carbon costs are estimated at ${carbon_cost:,.0f}. "
            f"Stranded asset exposure is ${stranded_value:,.0f}."
        )

    @staticmethod
    def _interpolate_value(
        year: int, trajectory: Dict[int, Decimal],
    ) -> Decimal:
        """
        Linearly interpolate a value for a given year from a trajectory.

        If the year falls between two known data points, a linear
        interpolation is used. If before the first point or after
        the last, the nearest endpoint value is returned.
        """
        if not trajectory:
            return Decimal("0")

        sorted_points = sorted(trajectory.items())

        # Before first point
        if year <= sorted_points[0][0]:
            return sorted_points[0][1]
        # After last point
        if year >= sorted_points[-1][0]:
            return sorted_points[-1][1]

        # Find bounding points and interpolate
        for i in range(len(sorted_points) - 1):
            y0, v0 = sorted_points[i]
            y1, v1 = sorted_points[i + 1]
            if y0 <= year <= y1:
                span = y1 - y0
                if span == 0:
                    return v0
                fraction = Decimal(str(year - y0)) / Decimal(str(span))
                return (v0 + (v1 - v0) * fraction).quantize(Decimal("0.01"))

        return sorted_points[-1][1]

    @staticmethod
    def _score_scenario_disclosure(
        results: List[ScenarioResult],
        resilience: Dict[str, Any],
    ) -> int:
        """
        Score Strategy (c) disclosure completeness (0-100).

        Criteria:
        - Scenario analysis conducted: 20 pts
        - Multiple scenarios (2+): 15 pts
        - Both orderly and disorderly scenarios: 10 pts
        - Quantitative NPV reported: 15 pts
        - Confidence intervals provided: 10 pts
        - Resilience assessment completed: 15 pts
        - Key assumptions documented: 15 pts
        """
        score = 20  # Base for having results

        # Multiple scenarios
        if len(results) >= 2:
            score += 15
        elif len(results) == 1:
            score += 5

        # Scenario diversity
        scenario_types = resilience.get("scenario_types_covered", [])
        if len(scenario_types) >= 3:
            score += 10
        elif len(scenario_types) >= 2:
            score += 5

        # Quantitative NPV
        if all(r.npv_usd != 0 for r in results):
            score += 15

        # Confidence intervals
        if all(r.confidence_interval_lower != 0 for r in results):
            score += 10

        # Resilience assessment
        if resilience.get("resilience_rating"):
            score += 15

        # Key assumptions
        if all(len(r.key_assumptions) >= 3 for r in results):
            score += 15
        elif any(r.key_assumptions for r in results):
            score += 8

        return min(score, 100)
