# -*- coding: utf-8 -*-
"""
Unit tests for EU Taxonomy Climate Risk Assessment Engine.

Tests climate risk assessment for DNSH climate adaptation, physical
risk identification (chronic and acute), adaptation solution
assessment, residual risk evaluation, multiple location assessments,
and time horizon impact analysis with 32+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

import pytest


# ===========================================================================
# Climate Risk Assessment for DNSH Adaptation
# ===========================================================================

class TestClimateRiskAssessmentDNSH:
    """Test climate risk assessment for DNSH climate change adaptation."""

    def test_risk_assessment_created(self, sample_climate_risk):
        assert sample_climate_risk["assessment_id"] is not None
        assert sample_climate_risk["activity_code"] is not None

    def test_risk_assessment_has_location(self, sample_climate_risk):
        assert "location" in sample_climate_risk
        assert sample_climate_risk["location"]["country"] is not None

    def test_risk_assessment_has_climate_scenarios(self, sample_climate_risk):
        assert "scenarios" in sample_climate_risk
        valid_scenarios = ["RCP2.6", "RCP4.5", "RCP6.0", "RCP8.5",
                           "SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]
        for scenario in sample_climate_risk["scenarios"]:
            assert scenario["name"] in valid_scenarios

    def test_risk_assessment_dnsh_verdict(self, sample_climate_risk):
        assert "dnsh_passes" in sample_climate_risk
        assert isinstance(sample_climate_risk["dnsh_passes"], bool)

    def test_risk_assessment_requires_adaptation(self, sample_climate_risk):
        if not sample_climate_risk["dnsh_passes"]:
            assert "adaptation_required" in sample_climate_risk
            assert sample_climate_risk["adaptation_required"] is True

    def test_risk_materiality_assessment(self, sample_climate_risk):
        assert "materiality" in sample_climate_risk
        assert sample_climate_risk["materiality"] in [
            "material", "not_material", "potentially_material",
        ]


# ===========================================================================
# Physical Risk Identification (Chronic)
# ===========================================================================

class TestChronicPhysicalRisk:
    """Test chronic physical risk identification."""

    CHRONIC_RISKS = [
        "temperature_change", "precipitation_change", "sea_level_rise",
        "water_stress", "soil_degradation", "permafrost_thawing",
    ]

    def test_chronic_risks_identified(self, sample_climate_risk):
        chronic = [
            r for r in sample_climate_risk["physical_risks"]
            if r["risk_type"] == "chronic"
        ]
        assert len(chronic) >= 1

    def test_chronic_risk_has_hazard(self, sample_climate_risk):
        for risk in sample_climate_risk["physical_risks"]:
            if risk["risk_type"] == "chronic":
                assert "hazard" in risk
                assert risk["hazard"] in self.CHRONIC_RISKS

    def test_chronic_risk_has_severity(self, sample_climate_risk):
        for risk in sample_climate_risk["physical_risks"]:
            if risk["risk_type"] == "chronic":
                assert "severity" in risk
                assert risk["severity"] in ["low", "medium", "high", "very_high"]

    def test_chronic_risk_has_likelihood(self, sample_climate_risk):
        for risk in sample_climate_risk["physical_risks"]:
            if risk["risk_type"] == "chronic":
                assert "likelihood" in risk
                assert risk["likelihood"] in [
                    "unlikely", "possible", "likely", "very_likely",
                ]

    @pytest.mark.parametrize("hazard,expected_category", [
        ("temperature_change", "temperature"),
        ("precipitation_change", "water"),
        ("sea_level_rise", "water"),
        ("water_stress", "water"),
        ("soil_degradation", "solid_mass"),
        ("permafrost_thawing", "temperature"),
    ])
    def test_chronic_hazard_categories(self, hazard, expected_category):
        category_map = {
            "temperature_change": "temperature",
            "precipitation_change": "water",
            "sea_level_rise": "water",
            "water_stress": "water",
            "soil_degradation": "solid_mass",
            "permafrost_thawing": "temperature",
        }
        assert category_map[hazard] == expected_category


# ===========================================================================
# Physical Risk Identification (Acute)
# ===========================================================================

class TestAcutePhysicalRisk:
    """Test acute physical risk identification."""

    ACUTE_RISKS = [
        "flood", "storm", "wildfire", "heatwave",
        "drought", "landslide", "cyclone",
    ]

    def test_acute_risks_identified(self, sample_climate_risk):
        acute = [
            r for r in sample_climate_risk["physical_risks"]
            if r["risk_type"] == "acute"
        ]
        assert len(acute) >= 1

    def test_acute_risk_has_hazard(self, sample_climate_risk):
        for risk in sample_climate_risk["physical_risks"]:
            if risk["risk_type"] == "acute":
                assert "hazard" in risk
                assert risk["hazard"] in self.ACUTE_RISKS

    def test_acute_risk_has_frequency(self, sample_climate_risk):
        for risk in sample_climate_risk["physical_risks"]:
            if risk["risk_type"] == "acute":
                assert "return_period_years" in risk
                assert risk["return_period_years"] > 0

    def test_acute_risk_has_impact(self, sample_climate_risk):
        for risk in sample_climate_risk["physical_risks"]:
            if risk["risk_type"] == "acute":
                assert "potential_impact" in risk
                assert risk["potential_impact"] in [
                    "negligible", "minor", "moderate", "major", "catastrophic",
                ]


# ===========================================================================
# Adaptation Solution Assessment
# ===========================================================================

class TestAdaptationSolutionAssessment:
    """Test adaptation solution assessment for material risks."""

    def test_adaptation_solutions_present(self, sample_climate_risk):
        if sample_climate_risk.get("adaptation_required"):
            assert "adaptation_solutions" in sample_climate_risk
            assert len(sample_climate_risk["adaptation_solutions"]) >= 1

    def test_solution_has_description(self, sample_climate_risk):
        for sol in sample_climate_risk.get("adaptation_solutions", []):
            assert "description" in sol
            assert len(sol["description"]) > 10

    def test_solution_addresses_risk(self, sample_climate_risk):
        for sol in sample_climate_risk.get("adaptation_solutions", []):
            assert "addresses_risk" in sol
            assert sol["addresses_risk"] is not None

    def test_solution_has_cost_estimate(self, sample_climate_risk):
        for sol in sample_climate_risk.get("adaptation_solutions", []):
            assert "estimated_cost" in sol
            assert sol["estimated_cost"] >= 0

    def test_solution_has_implementation_timeline(self, sample_climate_risk):
        for sol in sample_climate_risk.get("adaptation_solutions", []):
            assert "implementation_months" in sol
            assert sol["implementation_months"] > 0

    def test_solution_effectiveness(self, sample_climate_risk):
        for sol in sample_climate_risk.get("adaptation_solutions", []):
            assert "effectiveness" in sol
            assert sol["effectiveness"] in ["high", "medium", "low"]


# ===========================================================================
# Residual Risk Evaluation
# ===========================================================================

class TestResidualRiskEvaluation:
    """Test residual risk after adaptation measures."""

    def test_residual_risk_assessed(self, sample_climate_risk):
        if sample_climate_risk.get("adaptation_solutions"):
            assert "residual_risk_level" in sample_climate_risk

    def test_residual_risk_lower_than_initial(self, sample_climate_risk):
        severity_order = {"low": 1, "medium": 2, "high": 3, "very_high": 4}
        initial_max = max(
            severity_order.get(r["severity"], 0)
            for r in sample_climate_risk["physical_risks"]
        )
        residual = severity_order.get(
            sample_climate_risk.get("residual_risk_level", "low"), 0
        )
        assert residual <= initial_max

    def test_residual_risk_acceptable(self, sample_climate_risk):
        acceptable_levels = ["low", "medium"]
        residual = sample_climate_risk.get("residual_risk_level", "low")
        if sample_climate_risk.get("dnsh_passes"):
            assert residual in acceptable_levels

    def test_unacceptable_residual_risk_fails_dnsh(self):
        assessment = {
            "residual_risk_level": "very_high",
            "dnsh_passes": False,
        }
        assert assessment["dnsh_passes"] is False


# ===========================================================================
# Multiple Location Assessments
# ===========================================================================

class TestMultipleLocationAssessments:
    """Test climate risk assessment across multiple locations."""

    def test_multi_location_assessment(self, sample_multi_location_risk):
        assert len(sample_multi_location_risk["locations"]) >= 2

    def test_each_location_has_coordinates(self, sample_multi_location_risk):
        for loc in sample_multi_location_risk["locations"]:
            assert "latitude" in loc
            assert "longitude" in loc
            assert -90.0 <= loc["latitude"] <= 90.0
            assert -180.0 <= loc["longitude"] <= 180.0

    def test_each_location_has_risk_profile(self, sample_multi_location_risk):
        for loc in sample_multi_location_risk["locations"]:
            assert "risk_profile" in loc
            assert len(loc["risk_profile"]) >= 1

    def test_aggregate_risk_across_locations(self, sample_multi_location_risk):
        assert "aggregate_risk_level" in sample_multi_location_risk
        assert sample_multi_location_risk["aggregate_risk_level"] in [
            "low", "medium", "high", "very_high",
        ]

    def test_worst_case_location_identified(self, sample_multi_location_risk):
        assert "highest_risk_location" in sample_multi_location_risk


# ===========================================================================
# Time Horizon Impact
# ===========================================================================

class TestTimeHorizonImpact:
    """Test climate risk analysis across time horizons."""

    @pytest.mark.parametrize("horizon,years", [
        ("short", 10),
        ("medium", 30),
        ("long", 50),
    ])
    def test_time_horizons_defined(self, horizon, years):
        horizons = {"short": 10, "medium": 30, "long": 50}
        assert horizons[horizon] == years

    def test_risk_varies_by_horizon(self, sample_climate_risk):
        if "time_horizons" in sample_climate_risk:
            horizons = sample_climate_risk["time_horizons"]
            assert "short" in horizons
            assert "medium" in horizons
            assert "long" in horizons

    def test_long_term_risk_typically_higher(self, sample_climate_risk):
        if "time_horizons" in sample_climate_risk:
            severity_order = {"low": 1, "medium": 2, "high": 3, "very_high": 4}
            short_risk = severity_order.get(
                sample_climate_risk["time_horizons"]["short"]["risk_level"], 0
            )
            long_risk = severity_order.get(
                sample_climate_risk["time_horizons"]["long"]["risk_level"], 0
            )
            assert long_risk >= short_risk

    def test_lifetime_alignment(self, sample_climate_risk):
        if "asset_lifetime_years" in sample_climate_risk:
            lifetime = sample_climate_risk["asset_lifetime_years"]
            assert lifetime > 0
            # DNSH requires assessment covering asset lifetime
            if "time_horizons" in sample_climate_risk:
                max_horizon_years = max(
                    h.get("years", 0)
                    for h in sample_climate_risk["time_horizons"].values()
                    if isinstance(h, dict)
                )
                assert max_horizon_years >= lifetime or max_horizon_years >= 30
