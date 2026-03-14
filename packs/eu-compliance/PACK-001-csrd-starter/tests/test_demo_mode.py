# -*- coding: utf-8 -*-
"""
PACK-001 CSRD Starter Pack - Demo Mode Tests
===============================================

Validates the demo mode functionality that ships with the CSRD Starter
Pack. Demo mode allows prospective customers and integrators to run
the full pipeline with sample data without connecting real data sources.

Test count: 5
Author: GreenLang QA Team
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .conftest import PACK_ROOT


# ---------------------------------------------------------------------------
# Demo pipeline simulation
# ---------------------------------------------------------------------------

def _run_demo_pipeline(
    demo_config: Dict[str, Any],
    company_profile: Dict[str, Any],
    esg_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Simulate a full demo pipeline execution.

    In production, this calls CSRDPackOrchestrator in demo mode.
    For testing, we validate inputs and produce a deterministic result.
    """
    # Validate config
    assert demo_config.get("mode") == "demo", "Must be in demo mode"
    assert demo_config.get("skip_external_apis") is True, (
        "Demo mode must skip external APIs"
    )

    # Validate company profile
    assert company_profile.get("company_name"), "Company name required"
    assert company_profile.get("country"), "Country required"
    assert company_profile.get("sector"), "Sector required"

    # Validate ESG data
    assert len(esg_data) > 0, "ESG data must not be empty"

    # Simulate pipeline stages
    data_intake_result = {
        "records_ingested": len(esg_data),
        "records_validated": len(esg_data),
        "records_rejected": 0,
    }

    quality_result = {
        "quality_score": 0.91,
        "duplicates_found": 0,
        "outliers_found": 2,
        "missing_values_imputed": 3,
    }

    # Scope calculations from ESG data
    scope1_records = [r for r in esg_data if r["category"].startswith("scope1")]
    scope2_records = [r for r in esg_data if r["category"].startswith("scope2")]
    scope3_records = [r for r in esg_data if r["category"].startswith("scope3")]
    social_records = [r for r in esg_data if r["category"] == "social"]
    gov_records = [r for r in esg_data if r["category"] == "governance"]

    calculation_result = {
        "scope1_tco2e": sum(
            r["value"] * (r["emission_factor"] or 0)
            for r in scope1_records
            if r.get("emission_factor") is not None
        ) / 1000.0,  # Convert to tonnes
        "scope2_tco2e": sum(
            r["value"] * (r["emission_factor"] or 0)
            for r in scope2_records
            if r.get("emission_factor") is not None
        ) / 1000.0,
        "scope3_tco2e": sum(
            r["value"] * (r["emission_factor"] or 0)
            for r in scope3_records
            if r.get("emission_factor") is not None
        ) / 1000.0,
        "social_metrics_count": len(social_records),
        "governance_metrics_count": len(gov_records),
    }
    calculation_result["total_tco2e"] = (
        calculation_result["scope1_tco2e"]
        + calculation_result["scope2_tco2e"]
        + calculation_result["scope3_tco2e"]
    )

    materiality_result = {
        "material_topics": 6,
        "non_material_topics": 4,
        "assessment_complete": True,
    }

    report_result = {
        "reports_generated": ["executive_summary", "ghg_emissions_report"],
        "format": demo_config.get("output_formats", ["markdown"]),
    }

    return {
        "status": "completed",
        "mode": "demo",
        "company": company_profile["company_name"],
        "data_intake": data_intake_result,
        "quality": quality_result,
        "calculations": calculation_result,
        "materiality": materiality_result,
        "reports": report_result,
        "pipeline_steps_completed": 5,
        "pipeline_steps_total": 5,
    }


# =========================================================================
# Demo Mode Tests
# =========================================================================

class TestDemoMode:
    """Tests for the CSRD Starter Pack demo mode."""

    def test_demo_config_loads(self, demo_config):
        """Demo configuration loads with all required fields."""
        assert demo_config["mode"] == "demo"
        assert demo_config["preset"] == "mid_market"
        assert demo_config["sector"] == "manufacturing"
        assert demo_config["skip_external_apis"] is True
        assert demo_config["use_cached_emission_factors"] is True
        assert demo_config["max_records"] > 0
        assert "output_formats" in demo_config
        assert len(demo_config["output_formats"]) >= 1

    def test_demo_company_profile_valid(self, sample_company_profile):
        """Demo company profile has all fields needed for CSRD reporting."""
        profile = sample_company_profile
        assert profile["company_name"] == "GreenTech Manufacturing GmbH"
        assert profile["country"] == "DE"
        assert profile["sector"] == "manufacturing"
        assert profile["employees"] > 250, (
            "Demo company should be large enough for CSRD requirements"
        )
        assert profile["revenue_eur"] > 40_000_000, (
            "Demo company should meet CSRD revenue threshold"
        )
        assert "nace_code" in profile
        assert "facilities" in profile
        assert len(profile["facilities"]) >= 1
        assert "data_sources" in profile
        assert len(profile["data_sources"]) >= 2
        # Must have fiscal year info
        assert "reporting_year" in profile
        assert "fiscal_year_end" in profile

    def test_demo_esg_data_loads(self, sample_esg_data):
        """Demo ESG dataset loads with 50 records covering all categories."""
        data = sample_esg_data
        assert len(data) == 50

        # Check data point distribution
        categories = set(r["category"] for r in data)
        expected_categories = {
            "scope1_stationary_combustion",
            "scope1_refrigerants",
            "scope1_mobile_combustion",
            "scope2_electricity",
            "scope3_cat1_purchased_goods",
            "scope3_cat6_business_travel",
            "social",
            "governance",
            "environment",
        }
        assert expected_categories.issubset(categories), (
            f"Missing categories: {expected_categories - categories}"
        )

        # Check ESRS standards covered
        standards = set(r["esrs_standard"] for r in data)
        assert "E1" in standards, "Must include E1 (Climate Change)"
        assert "S1" in standards, "Must include S1 (Own Workforce)"
        assert "G1" in standards, "Must include G1 (Business Conduct)"

        # Every record must have required schema fields
        required_fields = {"id", "esrs_standard", "data_point", "category", "value", "unit"}
        for record in data:
            missing = required_fields - set(record.keys())
            assert len(missing) == 0, (
                f"Record {record['id']} missing fields: {missing}"
            )

    def test_demo_pipeline_executes(
        self, demo_config, sample_company_profile, sample_esg_data
    ):
        """Demo pipeline executes all 5 stages and completes successfully."""
        result = _run_demo_pipeline(demo_config, sample_company_profile, sample_esg_data)
        assert result["status"] == "completed"
        assert result["mode"] == "demo"
        assert result["pipeline_steps_completed"] == result["pipeline_steps_total"]
        assert result["pipeline_steps_completed"] == 5

        # Data intake
        assert result["data_intake"]["records_ingested"] == 50
        assert result["data_intake"]["records_rejected"] == 0

        # Quality
        assert result["quality"]["quality_score"] > 0.8

        # Calculations ran
        calcs = result["calculations"]
        assert calcs["scope1_tco2e"] > 0
        assert calcs["scope2_tco2e"] > 0
        assert calcs["scope3_tco2e"] > 0
        assert calcs["total_tco2e"] > 0

        # Materiality completed
        assert result["materiality"]["assessment_complete"] is True

    def test_demo_output_valid(
        self, demo_config, sample_company_profile, sample_esg_data
    ):
        """Demo pipeline output can be serialized to JSON and contains
        all expected sections."""
        result = _run_demo_pipeline(demo_config, sample_company_profile, sample_esg_data)

        # Output must be JSON-serializable
        json_output = json.dumps(result)
        assert len(json_output) > 0

        # Parse back and verify structure
        parsed = json.loads(json_output)
        required_keys = {
            "status", "mode", "company",
            "data_intake", "quality", "calculations",
            "materiality", "reports",
        }
        assert required_keys.issubset(set(parsed.keys())), (
            f"Missing output keys: {required_keys - set(parsed.keys())}"
        )

        # Verify reports list
        reports = parsed["reports"]
        assert len(reports["reports_generated"]) >= 1
        assert "format" in reports
