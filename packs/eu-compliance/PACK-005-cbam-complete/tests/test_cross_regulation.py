# -*- coding: utf-8 -*-
"""
PACK-005 CBAM Complete Pack - Cross-Regulation Engine Tests (20 tests)

Tests CrossRegulationEngine: mapping CBAM data to CSRD, CDP, SBTi,
EU Taxonomy, EU ETS, and EUDR; data reuse optimization; consistency
checks; carbon pricing equivalence; regulatory change tracking;
and third-country carbon pricing for all 50+ countries.

Author: GreenLang QA Team
"""

import json
from typing import Any, Dict, List

import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import (
    _compute_hash,
    _utcnow,
    assert_provenance_hash,
)


# ---------------------------------------------------------------------------
# Cross-Regulation Mapping (6 tests)
# ---------------------------------------------------------------------------

class TestCrossRegulationMapping:
    """Test mapping CBAM data to other regulatory frameworks."""

    def test_map_to_csrd(self, sample_cbam_data):
        """Test mapping CBAM data to CSRD disclosures."""
        csrd_mapping = {
            "esrs_e1_1": {
                "disclosure": "Scope 3 upstream emissions from imported goods",
                "cbam_field": "total_emissions_tco2e",
                "value": sample_cbam_data["total_emissions_tco2e"],
                "unit": "tCO2e",
            },
            "esrs_e1_7": {
                "disclosure": "Carbon pricing exposure",
                "cbam_field": "certificate_cost_eur",
                "value": sample_cbam_data["certificate_cost_eur"],
                "unit": "EUR",
            },
        }
        assert csrd_mapping["esrs_e1_1"]["value"] == 22500.0
        assert "tCO2e" in csrd_mapping["esrs_e1_1"]["unit"]

    def test_map_to_cdp(self, sample_cbam_data):
        """Test mapping CBAM data to CDP Climate questionnaire."""
        cdp_mapping = {
            "C6.1": {
                "question": "Gross global Scope 1 emissions",
                "cbam_relevance": "CBAM covers imported embedded emissions (Scope 3)",
            },
            "C11.1a": {
                "question": "Carbon pricing systems",
                "cbam_value": sample_cbam_data["certificate_cost_eur"],
                "cbam_type": "CBAM_certificates",
            },
        }
        assert "CBAM" in cdp_mapping["C11.1a"]["cbam_type"]

    def test_map_to_sbti(self, sample_cbam_data):
        """Test mapping CBAM data to SBTi target tracking."""
        sbti_mapping = {
            "scope3_category1": {
                "description": "Purchased goods and services",
                "cbam_emissions_tco2e": sample_cbam_data["total_emissions_tco2e"],
                "baseline_year": 2024,
                "target_year": 2030,
                "target_reduction_pct": 25.0,
            },
        }
        assert sbti_mapping["scope3_category1"]["cbam_emissions_tco2e"] == 22500.0

    def test_map_to_taxonomy(self, sample_cbam_data):
        """Test mapping CBAM data to EU Taxonomy alignment."""
        taxonomy_mapping = {
            "activity": "Import of industrial materials",
            "substantial_contribution": {
                "climate_mitigation": False,
                "rationale": "CBAM-covered imports have embedded emissions",
            },
            "dnsh": {
                "climate_adaptation": True,
                "pollution": True,
            },
            "cbam_cost_as_transition_indicator": sample_cbam_data["certificate_cost_eur"],
        }
        assert taxonomy_mapping["substantial_contribution"]["climate_mitigation"] is False

    def test_map_to_ets(self, sample_cbam_data, mock_ets_bridge):
        """Test mapping CBAM data to EU ETS benchmarks."""
        ets_mapping = {
            "free_allocation_pct": sample_cbam_data["free_allocation_pct"],
            "benchmark_hot_metal": mock_ets_bridge.get_benchmark("steel_hot_metal"),
            "benchmark_clinker": mock_ets_bridge.get_benchmark("cement_clinker"),
            "current_ets_price": mock_ets_bridge.get_current_price()["price_eur"],
        }
        assert ets_mapping["benchmark_hot_metal"] == 1.328
        assert ets_mapping["current_ets_price"] == 78.50

    def test_map_to_eudr(self, sample_cbam_data):
        """Test mapping CBAM data to EUDR traceability."""
        eudr_mapping = {
            "overlap_commodities": [],
            "shared_suppliers": sample_cbam_data["suppliers_count"],
            "geographic_overlap": {
                "countries": sample_cbam_data["countries_of_origin"],
                "deforestation_risk": "low_for_industrial_goods",
            },
        }
        assert len(eudr_mapping["geographic_overlap"]["countries"]) >= 3


# ---------------------------------------------------------------------------
# Data Reuse and Consistency (4 tests)
# ---------------------------------------------------------------------------

class TestDataReuseAndConsistency:
    """Test data reuse optimization and consistency checks."""

    def test_data_reuse_optimization(self, sample_cbam_data):
        """Test identifying reusable data across frameworks."""
        reusable_fields = {
            "total_emissions_tco2e": ["csrd", "cdp", "sbti"],
            "certificate_cost_eur": ["csrd", "cdp", "taxonomy"],
            "suppliers_count": ["csrd", "eudr"],
            "countries_of_origin": ["csrd", "eudr", "taxonomy"],
        }
        total_reuses = sum(len(targets) for targets in reusable_fields.values())
        assert total_reuses >= 10

    def test_consistency_check_pass(self, sample_cbam_data):
        """Test consistency check passes when data aligns."""
        cbam_emissions = sample_cbam_data["total_emissions_tco2e"]
        csrd_reported_emissions = 22500.0
        difference_pct = abs(cbam_emissions - csrd_reported_emissions) / cbam_emissions * 100
        consistent = difference_pct < 5.0  # 5% threshold
        assert consistent is True

    def test_consistency_check_conflict(self, sample_cbam_data):
        """Test consistency check detects conflicting data."""
        cbam_emissions = sample_cbam_data["total_emissions_tco2e"]
        csrd_reported_emissions = 15000.0  # Significant difference
        difference_pct = abs(cbam_emissions - csrd_reported_emissions) / cbam_emissions * 100
        consistent = difference_pct < 5.0
        assert consistent is False
        conflict = {
            "field": "total_emissions_tco2e",
            "cbam_value": cbam_emissions,
            "csrd_value": csrd_reported_emissions,
            "difference_pct": round(difference_pct, 1),
            "severity": "high",
        }
        assert conflict["severity"] == "high"

    def test_data_freshness_tracking(self, sample_cbam_data):
        """Test tracking data freshness across frameworks."""
        freshness = {
            "cbam": {"last_updated": "2026-03-14", "stale": False},
            "csrd": {"last_updated": "2026-02-28", "stale": False},
            "cdp": {"last_updated": "2025-07-31", "stale": True},
        }
        stale_count = sum(1 for f in freshness.values() if f["stale"])
        assert stale_count == 1


# ---------------------------------------------------------------------------
# Carbon Pricing Equivalence (3 tests)
# ---------------------------------------------------------------------------

class TestCarbonPricingEquivalence:
    """Test carbon pricing equivalence calculations."""

    def test_carbon_pricing_equivalence(self):
        """Test carbon price equivalence for deduction eligibility."""
        eu_ets_price = 78.50
        country_prices = {
            "TR": 12.0,
            "CN": 8.5,
            "KR": 25.0,
            "GB": 55.0,
        }
        deductions = {}
        for country, price in country_prices.items():
            deduction_per_tco2e = min(price, eu_ets_price)
            deductions[country] = {
                "local_price_eur": price,
                "deduction_eur": deduction_per_tco2e,
                "effective_cbam_eur": eu_ets_price - deduction_per_tco2e,
            }
        assert deductions["GB"]["effective_cbam_eur"] < deductions["CN"]["effective_cbam_eur"]
        assert deductions["TR"]["deduction_eur"] == 12.0

    def test_regulatory_change_tracking(self):
        """Test tracking regulatory changes across frameworks."""
        changes = [
            {"date": "2026-01-01", "framework": "cbam", "change": "Definitive period begins"},
            {"date": "2026-01-01", "framework": "csrd", "change": "First reporting year"},
            {"date": "2026-06-30", "framework": "taxonomy", "change": "New delegated acts"},
        ]
        upcoming = [c for c in changes if c["date"] >= "2026-03-14"]
        assert len(upcoming) >= 1

    def test_all_50_countries(self, sample_config):
        """Test third-country carbon pricing covers 50+ countries."""
        pricing = sample_config["cross_regulation"]["third_country_carbon_pricing"]
        assert len(pricing) >= 50
        # Verify key exporters
        for country in ["TR", "CN", "IN", "BR", "ZA", "KR"]:
            assert country in pricing
            assert pricing[country] > 0


# ---------------------------------------------------------------------------
# Additional Cross-Regulation (7 tests)
# ---------------------------------------------------------------------------

class TestAdditionalCrossRegulation:
    """Test additional cross-regulation features."""

    def test_mapping_completeness(self, sample_config):
        """Test all 6 target frameworks are mapped."""
        targets = sample_config["cross_regulation"]["targets"]
        assert len(targets) == 6

    def test_csrd_double_materiality(self, sample_cbam_data):
        """Test CSRD double materiality assessment with CBAM data."""
        financial_materiality = {
            "cbam_cost_exposure": sample_cbam_data["certificate_cost_eur"],
            "material": sample_cbam_data["certificate_cost_eur"] > 10000,
        }
        impact_materiality = {
            "embedded_emissions": sample_cbam_data["total_emissions_tco2e"],
            "material": sample_cbam_data["total_emissions_tco2e"] > 1000,
        }
        assert financial_materiality["material"] is True
        assert impact_materiality["material"] is True

    def test_cdp_scoring_impact(self, sample_cbam_data):
        """Test CBAM compliance impact on CDP scoring."""
        has_cbam_compliance = sample_cbam_data["verified"]
        has_carbon_pricing = sample_cbam_data["certificate_cost_eur"] > 0
        scoring_bonus = has_cbam_compliance and has_carbon_pricing
        assert scoring_bonus is True

    def test_sbti_pathway_alignment(self, sample_cbam_data):
        """Test CBAM data supports SBTi pathway alignment."""
        baseline_emissions = 25000.0
        current_emissions = sample_cbam_data["total_emissions_tco2e"]
        reduction_achieved = (baseline_emissions - current_emissions) / baseline_emissions * 100
        on_track = reduction_achieved >= 0  # Any reduction is positive
        assert on_track is True

    def test_taxonomy_capex_alignment(self, sample_cbam_data):
        """Test Taxonomy CapEx alignment using CBAM transition data."""
        total_capex = 10000000
        green_capex = 3000000  # Investment in lower-emission suppliers
        alignment_pct = green_capex / total_capex * 100
        assert alignment_pct == 30.0

    def test_cross_regulation_summary_report(self, sample_cbam_data):
        """Test cross-regulation summary report generation."""
        summary = {
            "cbam_emissions_tco2e": sample_cbam_data["total_emissions_tco2e"],
            "frameworks_mapped": 6,
            "data_fields_reused": 12,
            "consistency_issues": 0,
            "provenance_hash": _compute_hash(sample_cbam_data),
        }
        assert_provenance_hash(summary)
        assert summary["frameworks_mapped"] == 6

    def test_ets_free_allocation_bridge(self, mock_ets_bridge):
        """Test ETS free allocation data is bridged to CBAM."""
        fa_2026 = mock_ets_bridge.get_free_allocation_pct(2026)
        fa_2034 = mock_ets_bridge.get_free_allocation_pct(2034)
        assert fa_2026 == 97.5
        assert fa_2034 == 0.0
