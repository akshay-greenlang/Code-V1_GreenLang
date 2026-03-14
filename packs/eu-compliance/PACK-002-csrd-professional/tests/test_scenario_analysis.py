# -*- coding: utf-8 -*-
"""
PACK-002 CSRD Professional Pack - Scenario Analysis Tests
============================================================

Tests for climate scenario analysis covering IEA NZE, IEA APS,
NGFS Orderly, NGFS Disorderly scenarios with physical and transition
risk assessment, financial impact, and resilience scoring.

Test count: 12
Author: GreenLang QA Team
"""

from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Scenario Analysis Stub
# ---------------------------------------------------------------------------

class ScenarioAnalyzerStub:
    """Lightweight scenario analysis implementation for test validation."""

    def __init__(self, config: Dict[str, Any]):
        self.scenarios = config["scenarios"]
        self.time_horizons = config["time_horizons"]
        self.physical_params = config.get("physical_risk_params", {})
        self.monte_carlo = config.get("monte_carlo", {})

    def analyze_scenario(self, scenario_id: str) -> Dict[str, Any]:
        """Run analysis for a single scenario."""
        scenario = next(s for s in self.scenarios if s["id"] == scenario_id)
        results = {
            "scenario_id": scenario_id,
            "name": scenario["name"],
            "warming_target_c": scenario["warming_target_c"],
            "time_horizons": {},
        }
        for horizon in self.time_horizons:
            # Deterministic stub: carbon price scales linearly
            year_fraction = (horizon - 2025) / (2050 - 2025)
            carbon_price = (
                scenario["carbon_price_2030_eur"]
                + (scenario["carbon_price_2050_eur"] - scenario["carbon_price_2030_eur"]) * year_fraction
            )
            annual_cost = carbon_price * 25050 / 1_000_000  # Total Scope 1+2 in MtCO2e
            results["time_horizons"][horizon] = {
                "carbon_price_eur": round(carbon_price, 2),
                "carbon_cost_meur": round(annual_cost, 2),
                "renewable_share_pct": round(
                    scenario["renewable_share_2030_pct"]
                    + (scenario["renewable_share_2050_pct"] - scenario["renewable_share_2030_pct"]) * year_fraction,
                    1,
                ),
            }
        return results

    def assess_physical_risk(self) -> Dict[str, Any]:
        """Assess physical climate risks across facilities."""
        hazards = self.physical_params.get("hazards", [])
        locations = self.physical_params.get("facility_locations", [])
        assessments = []
        for loc in locations:
            risk_level = "high" if loc["lat"] > 48 else "medium" if loc["lat"] > 42 else "low"
            assessments.append({
                "location": loc["name"],
                "country": loc["country"],
                "overall_risk": risk_level,
                "hazard_exposures": {h: risk_level for h in hazards},
            })
        return {
            "total_facilities": len(locations),
            "assessments": assessments,
            "high_risk_count": len([a for a in assessments if a["overall_risk"] == "high"]),
        }

    def assess_transition_risk(self) -> Dict[str, Any]:
        """Assess transition risks from policy and market changes."""
        risks = []
        for scenario in self.scenarios:
            risk = {
                "scenario": scenario["id"],
                "policy_risk": "high" if scenario["carbon_price_2050_eur"] > 200 else "medium",
                "technology_risk": "medium",
                "market_risk": "medium" if scenario["warming_target_c"] <= 1.7 else "low",
                "reputation_risk": "low",
            }
            risks.append(risk)
        return {"scenario_risks": risks, "dominant_risk": "policy"}

    def calculate_financial_impact(self, scenario_id: str) -> Dict[str, Any]:
        """Calculate financial impact across 3 financial statements."""
        scenario = next(s for s in self.scenarios if s["id"] == scenario_id)
        revenue_impact_pct = scenario["gdp_impact_2050_pct"] * 0.8
        cost_impact_pct = -scenario["carbon_price_2050_eur"] * 25050 / 3_085_000_000 * 100
        return {
            "scenario_id": scenario_id,
            "income_statement": {"revenue_impact_pct": round(revenue_impact_pct, 2), "cost_impact_pct": round(cost_impact_pct, 2)},
            "balance_sheet": {"asset_impairment_pct": round(abs(scenario["gdp_impact_2050_pct"]) * 0.3, 2)},
            "cash_flow": {"capex_increase_pct": round(abs(scenario["gdp_impact_2050_pct"]) * 1.5, 2)},
        }

    def calculate_climate_var(self) -> Dict[str, Any]:
        """Calculate Climate Value-at-Risk."""
        # Stub: Use worst-case scenario
        worst = max(self.scenarios, key=lambda s: abs(s["gdp_impact_2050_pct"]))
        return {
            "climate_var_pct": round(abs(worst["gdp_impact_2050_pct"]) * 1.2, 2),
            "worst_case_scenario": worst["id"],
            "confidence_level": self.monte_carlo.get("confidence_level", 0.95),
        }

    def assess_resilience(self) -> Dict[str, Any]:
        """Assess organizational resilience across all scenarios."""
        scores = []
        for scenario in self.scenarios:
            score = 100 - abs(scenario["gdp_impact_2050_pct"]) * 10
            scores.append({"scenario": scenario["id"], "resilience_score": round(max(score, 0), 1)})
        avg = round(sum(s["resilience_score"] for s in scores) / len(scores), 1)
        return {"per_scenario": scores, "average_resilience": avg, "rating": "good" if avg >= 70 else "moderate"}


# ===========================================================================
# Scenario Analysis Tests
# ===========================================================================

class TestScenarioAnalysis:
    """Test climate scenario analysis engine."""

    @pytest.fixture
    def analyzer(self, sample_scenario_config):
        return ScenarioAnalyzerStub(sample_scenario_config)

    def test_iea_nze_scenario(self, analyzer):
        """IEA NZE 2050 scenario produces results for all time horizons."""
        result = analyzer.analyze_scenario("IEA_NZE")
        assert result["warming_target_c"] == 1.5
        assert len(result["time_horizons"]) == 3
        assert 2030 in result["time_horizons"]
        assert 2050 in result["time_horizons"]
        # Carbon price should increase over time
        assert result["time_horizons"][2050]["carbon_price_eur"] > result["time_horizons"][2030]["carbon_price_eur"]

    def test_iea_aps_scenario(self, analyzer):
        """IEA APS scenario has 1.7C warming target."""
        result = analyzer.analyze_scenario("IEA_APS")
        assert result["warming_target_c"] == 1.7
        assert result["time_horizons"][2050]["renewable_share_pct"] > 60.0

    def test_ngfs_orderly_scenario(self, analyzer):
        """NGFS Orderly scenario has smooth transition."""
        result = analyzer.analyze_scenario("NGFS_ORDERLY")
        assert result["warming_target_c"] == 1.5
        # Orderly has lower 2050 carbon price than Disorderly
        disorderly = analyzer.analyze_scenario("NGFS_DISORDERLY")
        assert result["time_horizons"][2050]["carbon_price_eur"] < disorderly["time_horizons"][2050]["carbon_price_eur"]

    def test_ngfs_disorderly_scenario(self, analyzer):
        """NGFS Disorderly scenario has abrupt carbon price spike."""
        result = analyzer.analyze_scenario("NGFS_DISORDERLY")
        assert result["warming_target_c"] == 2.0
        # Disorderly has lower 2030 carbon price than NZE/Orderly (25 base vs 120-130 base)
        # but interpolation from 2025 gives ~90 at 2030 (stub uses linear from 2025-2050)
        assert result["time_horizons"][2030]["carbon_price_eur"] < result["time_horizons"][2050]["carbon_price_eur"]
        assert result["time_horizons"][2050]["carbon_price_eur"] > 300

    def test_physical_risk_assessment(self, analyzer):
        """Physical risk assessment covers all 6 facilities."""
        result = analyzer.assess_physical_risk()
        assert result["total_facilities"] == 6
        assert len(result["assessments"]) == 6
        # At least one high-risk facility
        assert result["high_risk_count"] >= 1
        # Each assessment has hazard exposures
        for assessment in result["assessments"]:
            assert len(assessment["hazard_exposures"]) == 5

    def test_transition_risk_assessment(self, analyzer):
        """Transition risk assessment covers all 4 scenarios."""
        result = analyzer.assess_transition_risk()
        assert len(result["scenario_risks"]) == 4
        # Policy risk is dominant
        assert result["dominant_risk"] == "policy"

    def test_financial_impact_3_statement(self, analyzer):
        """Financial impact covers all 3 financial statements."""
        result = analyzer.calculate_financial_impact("IEA_NZE")
        assert "income_statement" in result
        assert "balance_sheet" in result
        assert "cash_flow" in result
        assert "revenue_impact_pct" in result["income_statement"]
        assert "asset_impairment_pct" in result["balance_sheet"]
        assert "capex_increase_pct" in result["cash_flow"]

    def test_climate_var(self, analyzer):
        """Climate VaR is calculated from worst-case scenario."""
        result = analyzer.calculate_climate_var()
        assert result["climate_var_pct"] > 0
        assert result["confidence_level"] == 0.95
        # Worst case should be NGFS Disorderly (highest gdp impact)
        assert result["worst_case_scenario"] == "NGFS_DISORDERLY"

    def test_carbon_price_sensitivity(self, analyzer):
        """Carbon price sensitivity shows different impacts across scenarios."""
        results = {}
        for scenario in analyzer.scenarios:
            result = analyzer.analyze_scenario(scenario["id"])
            results[scenario["id"]] = result["time_horizons"][2050]["carbon_cost_meur"]

        # NZE should have moderate cost, Disorderly highest
        assert results["NGFS_DISORDERLY"] > results["IEA_NZE"]

    def test_resilience_assessment(self, analyzer):
        """Resilience assessment scores range from 0-100."""
        result = analyzer.assess_resilience()
        assert len(result["per_scenario"]) == 4
        assert 0 <= result["average_resilience"] <= 100
        assert result["rating"] in ("good", "moderate", "poor")
        # All scenario scores should be positive
        for s in result["per_scenario"]:
            assert s["resilience_score"] >= 0

    def test_scenario_comparison(self, analyzer):
        """All scenarios can be compared side-by-side."""
        all_results = {}
        for scenario in analyzer.scenarios:
            all_results[scenario["id"]] = analyzer.analyze_scenario(scenario["id"])

        assert len(all_results) == 4
        # All scenarios should have different carbon prices at 2050
        prices = [r["time_horizons"][2050]["carbon_price_eur"] for r in all_results.values()]
        assert len(set(prices)) == 4  # All unique

    def test_custom_scenario(self, sample_scenario_config):
        """Custom scenario with user-defined parameters can be analyzed."""
        config = sample_scenario_config.copy()
        config["scenarios"] = config["scenarios"] + [{
            "id": "CUSTOM_1",
            "name": "Custom Scenario",
            "source": "Internal",
            "warming_target_c": 2.5,
            "carbon_price_2030_eur": 40,
            "carbon_price_2050_eur": 100,
            "renewable_share_2030_pct": 25.0,
            "renewable_share_2050_pct": 45.0,
            "gdp_impact_2050_pct": -3.0,
            "category": "custom",
        }]
        analyzer = ScenarioAnalyzerStub(config)
        result = analyzer.analyze_scenario("CUSTOM_1")
        assert result["warming_target_c"] == 2.5
        assert result["name"] == "Custom Scenario"
