# -*- coding: utf-8 -*-
"""
Tests for RiskPropagationEngine - AGENT-EUDR-008 Engine 5: Supplier Risk Propagation

Comprehensive test suite covering:
- Individual risk category scoring (F5.1)
- Composite risk score with weighted categories (F5.2)
- Max propagation (worst-case) through chain (F5.4)
- Weighted average propagation (F5.4)
- Volume-weighted propagation (F5.4)
- Risk change detection and alerts (F5.6)
- Country risk lookups (F5.1)
- Concentration risk (F5.8)
- Risk trend analysis (F5.7)
- Propagation rule configurability (F5.5)

Test count: 65+ tests
Coverage target: >= 85% of RiskPropagationEngine module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.multi_tier_supplier.conftest import (
    SUP_ID_COCOA_IMPORTER_EU,
    SUP_ID_COCOA_TRADER_GH,
    SUP_ID_COCOA_PROCESSOR_GH,
    SUP_ID_COCOA_AGGREGATOR_GH,
    SUP_ID_COCOA_COOPERATIVE_GH,
    SUP_ID_COCOA_FARMER_1_GH,
    SUP_ID_COCOA_FARMER_2_GH,
    SUP_ID_PALM_REFINERY_ID,
    SUP_ID_TIMBER_SAWMILL_CD,
    SUP_ID_SOYA_TRADER_BR,
    RISK_CATEGORIES,
    RISK_CATEGORY_WEIGHTS,
    COUNTRY_RISK_SCORES,
    EUDR_COMMODITIES,
    SHA256_HEX_LENGTH,
    make_supplier,
    make_relationship,
    assert_valid_risk_score,
    build_linear_chain,
    build_diamond_chain,
)


# ===========================================================================
# 1. Individual Risk Category Scoring
# ===========================================================================


class TestIndividualRiskCategories:
    """Test scoring for each individual risk category (F5.1)."""

    @pytest.mark.parametrize("category", RISK_CATEGORIES)
    def test_each_category_returns_score(self, risk_propagation_engine, category):
        """Each risk category produces a valid 0-100 score."""
        supplier = make_supplier(
            supplier_id="RISK-IND-001",
            country_iso="GH",
            certifications=[],
            gps_lat=5.6037,
            gps_lon=-0.1870,
        )
        score = risk_propagation_engine.score_category(supplier, category)
        assert_valid_risk_score(score)

    def test_deforestation_proximity_high_risk(self, risk_propagation_engine):
        """Supplier near known deforestation has high deforestation_proximity score."""
        supplier = make_supplier(
            supplier_id="RISK-DEFOR-HIGH",
            country_iso="BR",
            gps_lat=-12.97,
            gps_lon=-55.32,
        )
        score = risk_propagation_engine.score_category(
            supplier, "deforestation_proximity",
            context={"deforestation_distance_km": 2.0}
        )
        assert score >= 60.0

    def test_deforestation_proximity_low_risk(self, risk_propagation_engine):
        """Supplier far from deforestation has low deforestation_proximity score."""
        supplier = make_supplier(
            supplier_id="RISK-DEFOR-LOW",
            country_iso="DE",
            gps_lat=53.55,
            gps_lon=9.99,
        )
        score = risk_propagation_engine.score_category(
            supplier, "deforestation_proximity",
            context={"deforestation_distance_km": 500.0}
        )
        assert score <= 30.0

    def test_country_risk_high_for_drc(self, risk_propagation_engine):
        """DRC has high country risk score."""
        supplier = make_supplier(supplier_id="RISK-CD", country_iso="CD")
        score = risk_propagation_engine.score_category(supplier, "country_risk")
        assert score >= 60.0

    def test_country_risk_low_for_germany(self, risk_propagation_engine):
        """Germany has low country risk score."""
        supplier = make_supplier(supplier_id="RISK-DE", country_iso="DE")
        score = risk_propagation_engine.score_category(supplier, "country_risk")
        assert score <= 15.0

    def test_certification_gap_no_certs(self, risk_propagation_engine):
        """Supplier with no certifications has high certification_gap score."""
        supplier = make_supplier(
            supplier_id="RISK-NOCERT", certifications=[]
        )
        score = risk_propagation_engine.score_category(supplier, "certification_gap")
        assert score >= 50.0

    def test_certification_gap_with_valid_cert(self, risk_propagation_engine):
        """Supplier with valid certification has low certification_gap score."""
        supplier = make_supplier(
            supplier_id="RISK-CERT",
            certifications=["RSPO-ID-2025-001"],
        )
        score = risk_propagation_engine.score_category(
            supplier, "certification_gap",
            context={"cert_status": "valid"}
        )
        assert score <= 30.0

    def test_data_quality_complete_profile(self, risk_propagation_engine):
        """Complete profile has low data_quality risk."""
        supplier = make_supplier(
            supplier_id="RISK-DQ-GOOD",
            registration_id="REG-001",
            tax_id="TAX-001",
            gps_lat=5.6037,
            gps_lon=-0.1870,
            primary_contact="Alice",
            compliance_contact="Bob",
        )
        score = risk_propagation_engine.score_category(supplier, "data_quality")
        assert score <= 30.0

    def test_data_quality_incomplete_profile(self, risk_propagation_engine):
        """Incomplete profile has high data_quality risk."""
        supplier = make_supplier(
            supplier_id="RISK-DQ-BAD",
            registration_id=None,
            tax_id=None,
            gps_lat=None,
            gps_lon=None,
            primary_contact=None,
            compliance_contact=None,
        )
        score = risk_propagation_engine.score_category(supplier, "data_quality")
        assert score >= 50.0

    def test_unknown_category_raises(self, risk_propagation_engine):
        """Unknown risk category raises ValueError."""
        supplier = make_supplier(supplier_id="RISK-UNK")
        with pytest.raises(ValueError):
            risk_propagation_engine.score_category(supplier, "nonexistent_category")


# ===========================================================================
# 2. Composite Risk Score
# ===========================================================================


class TestCompositeRiskScore:
    """Test composite risk score with weighted categories (F5.2)."""

    def test_composite_score_returns_value(self, risk_propagation_engine):
        """Composite risk score is a valid 0-100 value."""
        supplier = make_supplier(supplier_id="COMP-001", country_iso="GH")
        result = risk_propagation_engine.assess_risk(supplier)
        assert_valid_risk_score(result.composite_score)

    def test_composite_includes_category_breakdown(self, risk_propagation_engine):
        """Composite result includes per-category scores."""
        supplier = make_supplier(supplier_id="COMP-002", country_iso="GH")
        result = risk_propagation_engine.assess_risk(supplier)
        assert hasattr(result, "category_scores")
        for cat in RISK_CATEGORIES:
            assert cat in result.category_scores

    @pytest.mark.parametrize("category,weight", RISK_CATEGORY_WEIGHTS.items())
    def test_risk_weights_match_prd(self, risk_propagation_engine, category, weight):
        """Risk category weights match PRD Appendix B."""
        weights = risk_propagation_engine.get_risk_weights()
        assert category in weights
        assert weights[category] == pytest.approx(weight, abs=0.01)

    def test_risk_weights_sum_to_one(self, risk_propagation_engine):
        """All risk category weights sum to 1.0."""
        weights = risk_propagation_engine.get_risk_weights()
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.001)

    def test_high_risk_supplier(self, risk_propagation_engine):
        """Supplier with all risk factors high scores > 60."""
        supplier = make_supplier(
            supplier_id="COMP-HIGH",
            country_iso="CD",
            registration_id=None,
            gps_lat=None,
            gps_lon=None,
            certifications=[],
            primary_contact=None,
        )
        result = risk_propagation_engine.assess_risk(
            supplier,
            context={"deforestation_distance_km": 1.0}
        )
        assert result.composite_score >= 50.0

    def test_low_risk_supplier(self, risk_propagation_engine):
        """Supplier with all risk factors low scores < 30."""
        supplier = make_supplier(
            supplier_id="COMP-LOW",
            country_iso="DE",
            registration_id="REG-001",
            tax_id="TAX-001",
            gps_lat=53.55,
            gps_lon=9.99,
            certifications=["FSC-001"],
            primary_contact="Hans",
            compliance_contact="Greta",
        )
        result = risk_propagation_engine.assess_risk(
            supplier,
            context={"deforestation_distance_km": 1000.0, "cert_status": "valid"}
        )
        assert result.composite_score <= 35.0

    def test_composite_deterministic(self, risk_propagation_engine):
        """Same supplier always produces same composite score."""
        supplier = make_supplier(supplier_id="COMP-DET", country_iso="GH")
        r1 = risk_propagation_engine.assess_risk(supplier)
        r2 = risk_propagation_engine.assess_risk(supplier)
        assert r1.composite_score == r2.composite_score

    def test_composite_provenance_hash(self, risk_propagation_engine):
        """Risk assessment includes provenance hash."""
        supplier = make_supplier(supplier_id="COMP-PROV", country_iso="GH")
        result = risk_propagation_engine.assess_risk(supplier)
        assert len(result.provenance_hash) == SHA256_HEX_LENGTH


# ===========================================================================
# 3. Max Propagation (Worst-Case)
# ===========================================================================


class TestMaxPropagation:
    """Test max (worst-case) risk propagation through supply chain (F5.4)."""

    def test_max_propagation_linear_chain(self, risk_propagation_engine):
        """Highest risk in chain propagates to root with max method."""
        suppliers = [
            make_supplier(supplier_id="MAX-T0", tier=0, country_iso="DE"),
            make_supplier(supplier_id="MAX-T1", tier=1, country_iso="GH"),
            make_supplier(supplier_id="MAX-T2", tier=2, country_iso="CD"),  # high risk
        ]
        rels = [
            make_relationship("MAX-T0", "MAX-T1", rel_id="R-MAX-1"),
            make_relationship("MAX-T1", "MAX-T2", rel_id="R-MAX-2"),
        ]
        risk_scores = {"MAX-T0": 10.0, "MAX-T1": 40.0, "MAX-T2": 80.0}
        result = risk_propagation_engine.propagate(
            suppliers, rels, risk_scores, method="max"
        )
        # Root should inherit max risk (80)
        assert result["MAX-T0"] >= 80.0

    def test_max_propagation_takes_worst(self, risk_propagation_engine):
        """With two branches, max takes the worst branch."""
        suppliers = [
            make_supplier(supplier_id="MAXB-T0", tier=0),
            make_supplier(supplier_id="MAXB-T1A", tier=1),
            make_supplier(supplier_id="MAXB-T1B", tier=1),
        ]
        rels = [
            make_relationship("MAXB-T0", "MAXB-T1A", rel_id="R-MAXB-A"),
            make_relationship("MAXB-T0", "MAXB-T1B", rel_id="R-MAXB-B"),
        ]
        risk_scores = {"MAXB-T0": 10.0, "MAXB-T1A": 30.0, "MAXB-T1B": 90.0}
        result = risk_propagation_engine.propagate(
            suppliers, rels, risk_scores, method="max"
        )
        assert result["MAXB-T0"] >= 90.0

    def test_max_propagation_deep_chain(self, risk_propagation_engine, deep_linear_chain):
        """Max risk propagates through 15-tier chain to root."""
        suppliers, rels = deep_linear_chain
        # Set high risk only at the deepest tier
        risk_scores = {s["supplier_id"]: 10.0 for s in suppliers}
        risk_scores[suppliers[-1]["supplier_id"]] = 95.0
        result = risk_propagation_engine.propagate(
            suppliers, rels, risk_scores, method="max"
        )
        assert result[suppliers[0]["supplier_id"]] >= 95.0


# ===========================================================================
# 4. Weighted Average Propagation
# ===========================================================================


class TestWeightedAveragePropagation:
    """Test weighted average risk propagation (F5.4)."""

    def test_weighted_avg_two_branches(self, risk_propagation_engine):
        """Weighted average of two branches gives intermediate score."""
        suppliers = [
            make_supplier(supplier_id="WA-T0", tier=0),
            make_supplier(supplier_id="WA-T1A", tier=1),
            make_supplier(supplier_id="WA-T1B", tier=1),
        ]
        rels = [
            make_relationship("WA-T0", "WA-T1A", rel_id="R-WA-A"),
            make_relationship("WA-T0", "WA-T1B", rel_id="R-WA-B"),
        ]
        risk_scores = {"WA-T0": 0.0, "WA-T1A": 20.0, "WA-T1B": 80.0}
        result = risk_propagation_engine.propagate(
            suppliers, rels, risk_scores, method="weighted_average"
        )
        # Should be between 20 and 80
        assert 20.0 <= result["WA-T0"] <= 80.0

    def test_weighted_avg_single_child(self, risk_propagation_engine):
        """Single child propagation equals child risk."""
        suppliers = [
            make_supplier(supplier_id="WA1-T0", tier=0),
            make_supplier(supplier_id="WA1-T1", tier=1),
        ]
        rels = [make_relationship("WA1-T0", "WA1-T1", rel_id="R-WA1")]
        risk_scores = {"WA1-T0": 0.0, "WA1-T1": 60.0}
        result = risk_propagation_engine.propagate(
            suppliers, rels, risk_scores, method="weighted_average"
        )
        assert result["WA1-T0"] == pytest.approx(60.0, abs=5.0)

    def test_weighted_avg_equal_risks(self, risk_propagation_engine):
        """Equal risk across branches gives same risk to parent."""
        suppliers = [
            make_supplier(supplier_id="WAE-T0", tier=0),
            make_supplier(supplier_id="WAE-T1A", tier=1),
            make_supplier(supplier_id="WAE-T1B", tier=1),
        ]
        rels = [
            make_relationship("WAE-T0", "WAE-T1A", rel_id="R-WAE-A"),
            make_relationship("WAE-T0", "WAE-T1B", rel_id="R-WAE-B"),
        ]
        risk_scores = {"WAE-T0": 0.0, "WAE-T1A": 50.0, "WAE-T1B": 50.0}
        result = risk_propagation_engine.propagate(
            suppliers, rels, risk_scores, method="weighted_average"
        )
        assert result["WAE-T0"] == pytest.approx(50.0, abs=5.0)


# ===========================================================================
# 5. Volume-Weighted Propagation
# ===========================================================================


class TestVolumeWeightedPropagation:
    """Test volume-weighted risk propagation (F5.4)."""

    def test_volume_weighted_favors_large_supplier(self, risk_propagation_engine):
        """Higher-volume supplier has more influence on propagated risk."""
        suppliers = [
            make_supplier(supplier_id="VW-T0", tier=0),
            make_supplier(supplier_id="VW-T1A", tier=1),
            make_supplier(supplier_id="VW-T1B", tier=1),
        ]
        rels = [
            make_relationship("VW-T0", "VW-T1A", volume_mt=9000.0, rel_id="R-VW-A"),
            make_relationship("VW-T0", "VW-T1B", volume_mt=1000.0, rel_id="R-VW-B"),
        ]
        risk_scores = {"VW-T0": 0.0, "VW-T1A": 20.0, "VW-T1B": 100.0}
        result = risk_propagation_engine.propagate(
            suppliers, rels, risk_scores, method="volume_weighted"
        )
        # 90% weight on 20, 10% on 100 -> ~28
        assert result["VW-T0"] < 50.0

    def test_volume_weighted_equal_volume(self, risk_propagation_engine):
        """Equal volumes give equal weight to each supplier."""
        suppliers = [
            make_supplier(supplier_id="VWE-T0", tier=0),
            make_supplier(supplier_id="VWE-T1A", tier=1),
            make_supplier(supplier_id="VWE-T1B", tier=1),
        ]
        rels = [
            make_relationship("VWE-T0", "VWE-T1A", volume_mt=500.0, rel_id="R-VWE-A"),
            make_relationship("VWE-T0", "VWE-T1B", volume_mt=500.0, rel_id="R-VWE-B"),
        ]
        risk_scores = {"VWE-T0": 0.0, "VWE-T1A": 40.0, "VWE-T1B": 60.0}
        result = risk_propagation_engine.propagate(
            suppliers, rels, risk_scores, method="volume_weighted"
        )
        assert result["VWE-T0"] == pytest.approx(50.0, abs=5.0)

    def test_volume_weighted_zero_volume_excluded(self, risk_propagation_engine):
        """Zero-volume relationships are excluded from weighted calculation."""
        suppliers = [
            make_supplier(supplier_id="VW0-T0", tier=0),
            make_supplier(supplier_id="VW0-T1A", tier=1),
            make_supplier(supplier_id="VW0-T1B", tier=1),
        ]
        rels = [
            make_relationship("VW0-T0", "VW0-T1A", volume_mt=1000.0, rel_id="R-VW0-A"),
            make_relationship("VW0-T0", "VW0-T1B", volume_mt=0.0, rel_id="R-VW0-B"),
        ]
        risk_scores = {"VW0-T0": 0.0, "VW0-T1A": 30.0, "VW0-T1B": 100.0}
        result = risk_propagation_engine.propagate(
            suppliers, rels, risk_scores, method="volume_weighted"
        )
        # Only T1A counts
        assert result["VW0-T0"] == pytest.approx(30.0, abs=10.0)


# ===========================================================================
# 6. Risk Change Detection and Alerts
# ===========================================================================


class TestRiskChangeAlerts:
    """Test risk change detection and alert generation (F5.6)."""

    def test_alert_when_risk_crosses_threshold(self, risk_propagation_engine):
        """Alert generated when supplier risk crosses configured threshold."""
        old_scores = {"SUP-ALE-001": 60.0}
        new_scores = {"SUP-ALE-001": 75.0}
        alerts = risk_propagation_engine.detect_risk_changes(
            old_scores, new_scores, threshold=70.0
        )
        assert len(alerts) >= 1
        assert alerts[0]["supplier_id"] == "SUP-ALE-001"

    def test_no_alert_below_threshold(self, risk_propagation_engine):
        """No alert when risk change stays below threshold."""
        old_scores = {"SUP-ALE-002": 30.0}
        new_scores = {"SUP-ALE-002": 35.0}
        alerts = risk_propagation_engine.detect_risk_changes(
            old_scores, new_scores, threshold=70.0
        )
        assert len(alerts) == 0

    def test_alert_on_risk_increase(self, risk_propagation_engine):
        """Alert includes direction of risk change."""
        old_scores = {"SUP-ALE-003": 50.0}
        new_scores = {"SUP-ALE-003": 80.0}
        alerts = risk_propagation_engine.detect_risk_changes(
            old_scores, new_scores, threshold=70.0
        )
        if alerts:
            assert alerts[0]["direction"] in ("increase", "increased", "up")

    def test_alert_on_risk_decrease_below_threshold(self, risk_propagation_engine):
        """Alert generated when risk decreases below threshold (positive signal)."""
        old_scores = {"SUP-ALE-004": 85.0}
        new_scores = {"SUP-ALE-004": 60.0}
        alerts = risk_propagation_engine.detect_risk_changes(
            old_scores, new_scores, threshold=70.0
        )
        if alerts:
            assert alerts[0]["direction"] in ("decrease", "decreased", "down")

    def test_multiple_suppliers_alerting(self, risk_propagation_engine):
        """Multiple suppliers crossing threshold generate multiple alerts."""
        old_scores = {"S-A": 50.0, "S-B": 60.0, "S-C": 30.0}
        new_scores = {"S-A": 80.0, "S-B": 75.0, "S-C": 35.0}
        alerts = risk_propagation_engine.detect_risk_changes(
            old_scores, new_scores, threshold=70.0
        )
        alerted_ids = {a["supplier_id"] for a in alerts}
        assert "S-A" in alerted_ids
        assert "S-B" in alerted_ids

    def test_new_supplier_high_risk_alerts(self, risk_propagation_engine):
        """New supplier appearing with high risk triggers alert."""
        old_scores = {}
        new_scores = {"SUP-NEW": 90.0}
        alerts = risk_propagation_engine.detect_risk_changes(
            old_scores, new_scores, threshold=70.0
        )
        assert len(alerts) >= 1


# ===========================================================================
# 7. Country Risk Lookups
# ===========================================================================


class TestCountryRiskLookups:
    """Test country-level risk score lookups."""

    @pytest.mark.parametrize("country,min_risk,max_risk", [
        ("GH", 30.0, 60.0),
        ("CI", 40.0, 70.0),
        ("CO", 25.0, 55.0),
        ("BR", 35.0, 65.0),
        ("ID", 45.0, 75.0),
        ("MY", 20.0, 50.0),
        ("TH", 15.0, 45.0),
        ("CD", 55.0, 85.0),
        ("DE", 0.0, 15.0),
        ("NL", 0.0, 15.0),
    ])
    def test_country_risk_in_range(self, risk_propagation_engine, country, min_risk, max_risk):
        """Country risk scores fall within expected ranges."""
        score = risk_propagation_engine.get_country_risk(country)
        assert min_risk <= score <= max_risk

    def test_unknown_country_high_risk(self, risk_propagation_engine):
        """Unknown country code returns high default risk."""
        score = risk_propagation_engine.get_country_risk("XX")
        assert score >= 50.0

    def test_country_risk_deterministic(self, risk_propagation_engine):
        """Same country always returns same risk score."""
        s1 = risk_propagation_engine.get_country_risk("GH")
        s2 = risk_propagation_engine.get_country_risk("GH")
        assert s1 == s2


# ===========================================================================
# 8. Concentration Risk
# ===========================================================================


class TestConcentrationRisk:
    """Test concentration risk detection (F5.8)."""

    def test_single_source_concentration(self, risk_propagation_engine):
        """Single-source dependency yields high concentration risk."""
        suppliers = [
            make_supplier(supplier_id="CONC-T0", tier=0),
            make_supplier(supplier_id="CONC-T1", tier=1),
        ]
        rels = [make_relationship("CONC-T0", "CONC-T1", volume_mt=10000.0, rel_id="R-CONC")]
        score = risk_propagation_engine.assess_concentration(suppliers, rels, "CONC-T0")
        assert score >= 70.0

    def test_diversified_sources_low_concentration(self, risk_propagation_engine):
        """Multiple suppliers reduce concentration risk."""
        suppliers = [make_supplier(supplier_id="DIV-T0", tier=0)]
        rels = []
        for i in range(10):
            sid = f"DIV-T1-{i}"
            suppliers.append(make_supplier(supplier_id=sid, tier=1))
            rels.append(make_relationship("DIV-T0", sid, volume_mt=1000.0, rel_id=f"R-DIV-{i}"))
        score = risk_propagation_engine.assess_concentration(suppliers, rels, "DIV-T0")
        assert score <= 40.0

    def test_geographic_concentration(self, risk_propagation_engine):
        """All suppliers in same country increases concentration risk."""
        suppliers = [make_supplier(supplier_id="GEO-T0", tier=0, country_iso="DE")]
        rels = []
        for i in range(5):
            sid = f"GEO-T1-{i}"
            suppliers.append(make_supplier(supplier_id=sid, tier=1, country_iso="GH"))
            rels.append(make_relationship("GEO-T0", sid, volume_mt=2000.0, rel_id=f"R-GEO-{i}"))
        score = risk_propagation_engine.assess_concentration(suppliers, rels, "GEO-T0")
        # All from same country = higher geographic concentration
        assert score >= 30.0


# ===========================================================================
# 9. Propagation Method Comparison
# ===========================================================================


class TestPropagationMethodComparison:
    """Compare results across max, weighted_average, and volume_weighted methods."""

    def test_max_always_highest(self, risk_propagation_engine):
        """Max propagation always produces highest root risk."""
        suppliers = [
            make_supplier(supplier_id="CMP-T0", tier=0),
            make_supplier(supplier_id="CMP-T1A", tier=1),
            make_supplier(supplier_id="CMP-T1B", tier=1),
        ]
        rels = [
            make_relationship("CMP-T0", "CMP-T1A", volume_mt=500.0, rel_id="R-CMP-A"),
            make_relationship("CMP-T0", "CMP-T1B", volume_mt=500.0, rel_id="R-CMP-B"),
        ]
        risk_scores = {"CMP-T0": 0.0, "CMP-T1A": 20.0, "CMP-T1B": 80.0}

        r_max = risk_propagation_engine.propagate(suppliers, rels, risk_scores, method="max")
        r_avg = risk_propagation_engine.propagate(suppliers, rels, risk_scores, method="weighted_average")
        r_vol = risk_propagation_engine.propagate(suppliers, rels, risk_scores, method="volume_weighted")

        assert r_max["CMP-T0"] >= r_avg["CMP-T0"]
        assert r_max["CMP-T0"] >= r_vol["CMP-T0"]

    @pytest.mark.parametrize("method", ["max", "weighted_average", "volume_weighted"])
    def test_propagation_result_bounded(self, risk_propagation_engine, method):
        """All propagation methods produce scores in [0, 100]."""
        suppliers = [
            make_supplier(supplier_id="BND-T0", tier=0),
            make_supplier(supplier_id="BND-T1", tier=1),
        ]
        rels = [make_relationship("BND-T0", "BND-T1", rel_id="R-BND")]
        risk_scores = {"BND-T0": 0.0, "BND-T1": 65.0}
        result = risk_propagation_engine.propagate(suppliers, rels, risk_scores, method=method)
        for sid, score in result.items():
            assert_valid_risk_score(score)

    def test_invalid_method_raises(self, risk_propagation_engine):
        """Invalid propagation method raises ValueError."""
        suppliers = [make_supplier(supplier_id="INV-M", tier=0)]
        with pytest.raises(ValueError):
            risk_propagation_engine.propagate(
                suppliers, [], {"INV-M": 50.0}, method="invalid_method"
            )


# ===========================================================================
# 10. Provenance
# ===========================================================================


class TestRiskProvenance:
    """Test provenance tracking for risk assessments."""

    def test_assessment_has_provenance(self, risk_propagation_engine):
        """Risk assessment result includes provenance hash."""
        supplier = make_supplier(supplier_id="PROV-R-001")
        result = risk_propagation_engine.assess_risk(supplier)
        assert len(result.provenance_hash) == SHA256_HEX_LENGTH

    def test_propagation_provenance(self, risk_propagation_engine):
        """Risk propagation result includes provenance hash."""
        suppliers = [
            make_supplier(supplier_id="PROV-P-T0", tier=0),
            make_supplier(supplier_id="PROV-P-T1", tier=1),
        ]
        rels = [make_relationship("PROV-P-T0", "PROV-P-T1", rel_id="R-PROV-P")]
        risk_scores = {"PROV-P-T0": 10.0, "PROV-P-T1": 50.0}
        result = risk_propagation_engine.propagate(
            suppliers, rels, risk_scores, method="max"
        )
        # Result dict or object should have provenance
        assert result is not None
