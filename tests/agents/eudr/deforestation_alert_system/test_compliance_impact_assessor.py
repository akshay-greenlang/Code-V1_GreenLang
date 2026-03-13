# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceImpactAssessor - AGENT-EUDR-020 Engine 8

Tests EUDR compliance impact assessment including compliance outcome
determination, supply chain tracing, market restriction assessment,
financial impact calculation, remediation plan generation, DDS risk
evaluation, and provenance tracking.

Compliance Decision Logic:
    POST_CUTOFF + CRITICAL -> NON_COMPLIANT + market restriction
    POST_CUTOFF + HIGH     -> NON_COMPLIANT + market restriction
    POST_CUTOFF + MEDIUM   -> UNDER_REVIEW
    PRE_CUTOFF + any       -> COMPLIANT
    UNCERTAIN + any        -> REMEDIATION_REQUIRED
    ONGOING + CRITICAL     -> NON_COMPLIANT

Coverage targets: 85%+ across all ComplianceImpactAssessor methods.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-020
Agent ID: GL-EUDR-DAS-020
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.agents.eudr.deforestation_alert_system.engines.compliance_impact_assessor import (
    ALERT_SUPPLIER_MAP,
    AffectedProduct,
    AffectedSupplier,
    COMMODITY_PRICES_EUR_PER_TON,
    COMMODITY_YIELD_PER_HA,
    COMPLIANCE_RULES,
    ComplianceImpact,
    ComplianceImpactAssessor,
    ComplianceOutcome,
    DDS_RISK_MAP,
    ImpactSeverity,
    MARKET_RESTRICTION_LOSS_MULTIPLIER,
    MARKET_RESTRICTION_THRESHOLD,
    MAX_BATCH_SIZE,
    OUTCOME_REMEDIATION_MAP,
    REFERENCE_PRODUCTS,
    REFERENCE_SUPPLIERS,
    REGULATORY_FINE_CEILING_PCT,
    REMEDIATION_COST_ESTIMATES,
    REMEDIATION_TIMELINE_DAYS,
    RemediationAction,
    RemediationPlan,
    RemediationPriority,
    SEVERITY_ORDER,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def assessor() -> ComplianceImpactAssessor:
    """Create a default ComplianceImpactAssessor instance."""
    return ComplianceImpactAssessor()


@pytest.fixture
def assessor_with_custom_data() -> ComplianceImpactAssessor:
    """Create an assessor with custom supplier/product data."""
    a = ComplianceImpactAssessor()
    a.load_supplier_data("SUP-CUSTOM-1", {
        "name": "Custom Soya Farm",
        "country": "BR",
        "commodity": "soya",
        "plots": ["PLOT-C1", "PLOT-C2"],
        "annual_volume_tons": Decimal("2000"),
        "products": ["PROD-C1"],
    })
    a.load_product_data("PROD-C1", {
        "name": "Custom Soybean Meal",
        "hs_code": "2304",
        "commodity": "soya",
        "annual_value_eur": Decimal("300000"),
    })
    a.load_alert_supplier_mapping("alert-custom", ["SUP-CUSTOM-1"])
    return a


@pytest.fixture
def post_cutoff_critical_assessment(
    assessor: ComplianceImpactAssessor,
) -> Dict[str, Any]:
    """Assess impact for POST_CUTOFF CRITICAL scenario."""
    return assessor.assess_impact(
        "sample_post_cutoff",
        supply_chain_context={
            "cutoff_result": "POST_CUTOFF",
            "severity": "CRITICAL",
            "area_ha": "50",
        },
    )


# ---------------------------------------------------------------------------
# TestImpactAssessment
# ---------------------------------------------------------------------------


class TestImpactAssessment:
    """Tests for assess_impact with various scenarios."""

    def test_post_cutoff_critical_is_non_compliant(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Post-cutoff CRITICAL deforestation produces NON_COMPLIANT + market restriction."""
        result = assessor.assess_impact(
            "sample_post_cutoff",
            {"cutoff_result": "POST_CUTOFF", "severity": "CRITICAL"},
        )
        assert result["compliance_outcome"] == ComplianceOutcome.NON_COMPLIANT.value
        assert result["market_restriction"] is True
        assert result["provenance_hash"] != ""
        assert len(result["provenance_hash"]) == 64

    def test_pre_cutoff_is_compliant(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Pre-cutoff deforestation produces COMPLIANT (monitoring may still be needed)."""
        result = assessor.assess_impact(
            "sample_pre_cutoff",
            {"cutoff_result": "PRE_CUTOFF", "severity": "CRITICAL"},
        )
        assert result["compliance_outcome"] == ComplianceOutcome.COMPLIANT.value
        assert result["market_restriction"] is False

    def test_uncertain_is_remediation_required(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Uncertain timing produces REMEDIATION_REQUIRED."""
        result = assessor.assess_impact(
            "sample_uncertain",
            {"cutoff_result": "UNCERTAIN", "severity": "HIGH"},
        )
        assert result["compliance_outcome"] == ComplianceOutcome.REMEDIATION_REQUIRED.value

    def test_ongoing_critical_is_non_compliant(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Ongoing CRITICAL produces NON_COMPLIANT."""
        result = assessor.assess_impact(
            "sample_ongoing",
            {"cutoff_result": "ONGOING", "severity": "CRITICAL"},
        )
        assert result["compliance_outcome"] == ComplianceOutcome.NON_COMPLIANT.value

    def test_post_cutoff_medium_is_under_review(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Post-cutoff MEDIUM severity produces UNDER_REVIEW."""
        result = assessor.assess_impact(
            "alert-medium",
            {"cutoff_result": "POST_CUTOFF", "severity": "MEDIUM"},
        )
        assert result["compliance_outcome"] == ComplianceOutcome.UNDER_REVIEW.value

    def test_assess_impact_empty_id_raises(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Empty alert_id raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            assessor.assess_impact("")

    def test_assess_impact_no_context(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Assess without context defaults to UNCERTAIN/MEDIUM."""
        result = assessor.assess_impact("alert-no-ctx")
        assert "compliance_outcome" in result

    def test_assess_impact_result_fields(
        self, post_cutoff_critical_assessment: Dict[str, Any]
    ) -> None:
        """Impact assessment result includes all required fields."""
        result = post_cutoff_critical_assessment
        assert "impact_id" in result
        assert "compliance_outcome" in result
        assert "market_restriction" in result
        assert "affected_suppliers" in result
        assert "affected_products" in result
        assert "remediation_actions" in result
        assert "estimated_financial_impact_eur" in result
        assert "risk_to_dds" in result
        assert "regulatory_references" in result
        assert "provenance_hash" in result
        assert "processing_time_ms" in result

    def test_assess_impact_caches_result(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Assessment result is cached."""
        assessor.assess_impact(
            "alert-cache",
            {"cutoff_result": "POST_CUTOFF", "severity": "HIGH"},
        )
        assert "alert-cache" in assessor._assessment_cache


# ---------------------------------------------------------------------------
# TestComplianceOutcomeMatrix
# ---------------------------------------------------------------------------


class TestComplianceOutcomeMatrix:
    """Tests for _determine_compliance_outcome across the full matrix."""

    @pytest.mark.parametrize(
        "cutoff_result,severity,expected_outcome",
        [
            ("POST_CUTOFF", "CRITICAL", "NON_COMPLIANT"),
            ("POST_CUTOFF", "HIGH", "NON_COMPLIANT"),
            ("POST_CUTOFF", "MEDIUM", "UNDER_REVIEW"),
            ("POST_CUTOFF", "LOW", "UNDER_REVIEW"),
            ("PRE_CUTOFF", "CRITICAL", "COMPLIANT"),
            ("PRE_CUTOFF", "HIGH", "COMPLIANT"),
            ("PRE_CUTOFF", "MEDIUM", "COMPLIANT"),
            ("PRE_CUTOFF", "LOW", "COMPLIANT"),
            ("ONGOING", "CRITICAL", "NON_COMPLIANT"),
            ("ONGOING", "HIGH", "REMEDIATION_REQUIRED"),
            ("ONGOING", "MEDIUM", "UNDER_REVIEW"),
            ("ONGOING", "LOW", "UNDER_REVIEW"),
            ("UNCERTAIN", "CRITICAL", "REMEDIATION_REQUIRED"),
            ("UNCERTAIN", "HIGH", "REMEDIATION_REQUIRED"),
            ("UNCERTAIN", "MEDIUM", "REMEDIATION_REQUIRED"),
            ("UNCERTAIN", "LOW", "UNDER_REVIEW"),
        ],
    )
    def test_compliance_outcome_matrix(
        self,
        assessor: ComplianceImpactAssessor,
        cutoff_result: str,
        severity: str,
        expected_outcome: str,
    ) -> None:
        """Parametrized test for the complete compliance outcome matrix."""
        outcome = assessor._determine_compliance_outcome(
            cutoff_result, severity
        )
        assert outcome == expected_outcome


# ---------------------------------------------------------------------------
# TestAffectedProducts
# ---------------------------------------------------------------------------


class TestAffectedProducts:
    """Tests for get_affected_products tracing through supply chain."""

    def test_get_affected_products_with_known_alert(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Get affected products for a known alert."""
        result = assessor.get_affected_products("sample_post_cutoff")
        assert "affected_products" in result
        assert result["total_products"] >= 1
        assert result["provenance_hash"] != ""
        assert len(result["provenance_hash"]) == 64

    def test_get_affected_products_empty_id_raises(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Empty alert_id raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            assessor.get_affected_products("")

    def test_get_affected_products_unknown_alert(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Unknown alert returns empty products list."""
        result = assessor.get_affected_products("unknown-alert-xyz")
        assert result["total_products"] == 0

    def test_get_affected_products_custom_data(
        self, assessor_with_custom_data: ComplianceImpactAssessor
    ) -> None:
        """Custom supplier/product data is traced correctly."""
        result = assessor_with_custom_data.get_affected_products("alert-custom")
        assert result["total_products"] >= 1
        products = result["affected_products"]
        assert any(p["product_name"] == "Custom Soybean Meal" for p in products)


# ---------------------------------------------------------------------------
# TestRecommendations
# ---------------------------------------------------------------------------


class TestRecommendations:
    """Tests for get_recommendations for various risk levels."""

    def test_recommendations_for_non_compliant(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """NON_COMPLIANT assessment generates remediation recommendations."""
        assessor.assess_impact(
            "alert-rec",
            {"cutoff_result": "POST_CUTOFF", "severity": "CRITICAL"},
        )
        result = assessor.get_recommendations("alert-rec")
        assert result["total_recommendations"] >= 1
        assert result["compliance_outcome"] == ComplianceOutcome.NON_COMPLIANT.value
        assert result["provenance_hash"] != ""

    def test_recommendations_for_compliant(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """COMPLIANT assessment generates no remediation."""
        assessor.assess_impact(
            "alert-comp-rec",
            {"cutoff_result": "PRE_CUTOFF", "severity": "LOW"},
        )
        result = assessor.get_recommendations("alert-comp-rec")
        assert result["total_recommendations"] == 0

    def test_recommendations_empty_id_raises(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Empty alert_id raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            assessor.get_recommendations("")

    def test_recommendations_without_prior_assessment(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Recommendations without prior assessment default to UNDER_REVIEW."""
        result = assessor.get_recommendations("alert-no-prior")
        assert result["compliance_outcome"] == ComplianceOutcome.UNDER_REVIEW.value


# ---------------------------------------------------------------------------
# TestRemediationPlan
# ---------------------------------------------------------------------------


class TestRemediationPlan:
    """Tests for create_remediation_plan with actions."""

    def test_create_remediation_plan(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Create remediation plan from assessment."""
        assessment = assessor.assess_impact(
            "alert-plan",
            {"cutoff_result": "POST_CUTOFF", "severity": "CRITICAL"},
        )
        result = assessor.create_remediation_plan(
            impact_id=assessment["impact_id"],
            alert_id="alert-plan",
        )
        assert "plan_id" in result
        assert len(result["actions"]) >= 1
        assert Decimal(result["total_estimated_cost_eur"]) > Decimal("0")
        assert result["total_timeline_days"] > 0
        assert result["priority"] == RemediationPriority.IMMEDIATE.value
        assert result["provenance_hash"] != ""
        assert len(result["provenance_hash"]) == 64

    def test_create_plan_empty_impact_id_raises(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Empty impact_id raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            assessor.create_remediation_plan(impact_id="")

    def test_create_plan_with_custom_actions(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Create plan with specific actions override."""
        result = assessor.create_remediation_plan(
            impact_id="ci-custom",
            alert_id="alert-custom-plan",
            actions=[
                RemediationAction.SUPPLIER_AUDIT.value,
                RemediationAction.ENHANCED_MONITORING.value,
            ],
        )
        action_types = [a["action"] for a in result["actions"]]
        assert RemediationAction.SUPPLIER_AUDIT.value in action_types
        assert RemediationAction.ENHANCED_MONITORING.value in action_types

    def test_create_plan_with_responsible_parties(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Create plan with custom responsible parties."""
        result = assessor.create_remediation_plan(
            impact_id="ci-parties",
            responsible_parties=["compliance_officer", "supply_chain_manager"],
        )
        assert "compliance_officer" in result["responsible_parties"]
        assert "supply_chain_manager" in result["responsible_parties"]

    def test_non_compliant_plan_priority_immediate(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """NON_COMPLIANT plans have IMMEDIATE priority."""
        assessment = assessor.assess_impact(
            "alert-prio",
            {"cutoff_result": "POST_CUTOFF", "severity": "CRITICAL"},
        )
        plan = assessor.create_remediation_plan(
            impact_id=assessment["impact_id"],
        )
        assert plan["priority"] == RemediationPriority.IMMEDIATE.value


# ---------------------------------------------------------------------------
# TestSupplyChainTracing
# ---------------------------------------------------------------------------


class TestSupplyChainTracing:
    """Tests for _trace_supply_chain_impact upstream/downstream."""

    def test_trace_affected_suppliers_known_alert(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Known alert traces to reference suppliers."""
        suppliers = assessor._trace_affected_suppliers("sample_post_cutoff")
        assert len(suppliers) >= 1
        for s in suppliers:
            assert isinstance(s, AffectedSupplier)
            assert s.supplier_id != ""

    def test_trace_affected_suppliers_unknown(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Unknown alert returns empty supplier list."""
        suppliers = assessor._trace_affected_suppliers("unknown-xyz")
        assert len(suppliers) == 0

    def test_trace_affected_suppliers_custom(
        self, assessor_with_custom_data: ComplianceImpactAssessor
    ) -> None:
        """Custom alert mapping traces to custom suppliers."""
        suppliers = assessor_with_custom_data._trace_affected_suppliers(
            "alert-custom"
        )
        assert len(suppliers) >= 1
        assert suppliers[0].supplier_name == "Custom Soya Farm"


# ---------------------------------------------------------------------------
# TestMarketRestriction
# ---------------------------------------------------------------------------


class TestMarketRestriction:
    """Tests for _assess_market_restriction."""

    @pytest.mark.parametrize(
        "cutoff_result,severity,expected_restriction",
        [
            ("POST_CUTOFF", "CRITICAL", True),
            ("POST_CUTOFF", "HIGH", True),
            ("POST_CUTOFF", "MEDIUM", False),
            ("POST_CUTOFF", "LOW", False),
            ("PRE_CUTOFF", "CRITICAL", False),
            ("PRE_CUTOFF", "HIGH", False),
            ("ONGOING", "CRITICAL", True),
            ("ONGOING", "HIGH", True),
            ("ONGOING", "MEDIUM", False),
            ("UNCERTAIN", "CRITICAL", False),
        ],
    )
    def test_market_restriction_matrix(
        self,
        assessor: ComplianceImpactAssessor,
        cutoff_result: str,
        severity: str,
        expected_restriction: bool,
    ) -> None:
        """Parametrized test for market restriction decisions."""
        result = assessor._assess_market_restriction(cutoff_result, severity)
        assert result is expected_restriction


# ---------------------------------------------------------------------------
# TestFinancialImpact
# ---------------------------------------------------------------------------


class TestFinancialImpact:
    """Tests for _calculate_financial_impact."""

    def test_financial_impact_with_restriction(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Market restriction increases financial impact."""
        products = [
            AffectedProduct(
                product_id="P1",
                annual_value_eur=Decimal("100000"),
                market_restriction=True,
            ),
        ]
        financial = assessor._calculate_financial_impact(
            products, True, Decimal("10")
        )
        assert financial["total"] > Decimal("100000")
        assert financial["product_value"] == Decimal("100000.00")

    def test_financial_impact_without_restriction(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """No market restriction produces lower financial impact."""
        products = [
            AffectedProduct(
                product_id="P2",
                annual_value_eur=Decimal("100000"),
                market_restriction=False,
            ),
        ]
        financial_restricted = assessor._calculate_financial_impact(
            products, True, Decimal("10")
        )
        financial_unrestricted = assessor._calculate_financial_impact(
            products, False, Decimal("10")
        )
        assert financial_restricted["total"] > financial_unrestricted["total"]

    def test_financial_impact_no_products(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """No products produces minimal financial impact."""
        financial = assessor._calculate_financial_impact([], False, Decimal("10"))
        assert financial["product_value"] == Decimal("0.00")


# ---------------------------------------------------------------------------
# TestRemediationActions
# ---------------------------------------------------------------------------


class TestRemediationActions:
    """Tests for _generate_remediation_actions based on impact severity."""

    def test_non_compliant_generates_multiple_actions(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """NON_COMPLIANT outcome generates multiple remediation actions."""
        actions = assessor._generate_remediation_actions(
            ComplianceOutcome.NON_COMPLIANT.value, "CRITICAL"
        )
        assert len(actions) >= 3
        action_types = [a["action"] for a in actions]
        assert RemediationAction.PRODUCT_WITHDRAWAL.value in action_types

    def test_compliant_generates_no_actions(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """COMPLIANT outcome generates no remediation actions."""
        actions = assessor._generate_remediation_actions(
            ComplianceOutcome.COMPLIANT.value, "LOW"
        )
        assert len(actions) == 0

    def test_under_review_generates_monitoring(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """UNDER_REVIEW generates enhanced monitoring."""
        actions = assessor._generate_remediation_actions(
            ComplianceOutcome.UNDER_REVIEW.value, "MEDIUM"
        )
        action_types = [a["action"] for a in actions]
        assert RemediationAction.ENHANCED_MONITORING.value in action_types


# ---------------------------------------------------------------------------
# TestDDSRisk
# ---------------------------------------------------------------------------


class TestDDSRisk:
    """Tests for _assess_dds_risk for Due Diligence Statement implications."""

    def test_non_compliant_affects_dds(self) -> None:
        """NON_COMPLIANT affects DDS."""
        assert DDS_RISK_MAP[ComplianceOutcome.NON_COMPLIANT.value] is True

    def test_compliant_does_not_affect_dds(self) -> None:
        """COMPLIANT does not affect DDS."""
        assert DDS_RISK_MAP[ComplianceOutcome.COMPLIANT.value] is False

    def test_remediation_required_affects_dds(self) -> None:
        """REMEDIATION_REQUIRED affects DDS."""
        assert DDS_RISK_MAP[ComplianceOutcome.REMEDIATION_REQUIRED.value] is True

    def test_under_review_affects_dds(self) -> None:
        """UNDER_REVIEW affects DDS."""
        assert DDS_RISK_MAP[ComplianceOutcome.UNDER_REVIEW.value] is True


# ---------------------------------------------------------------------------
# TestArticle4Compliance
# ---------------------------------------------------------------------------


class TestArticle4Compliance:
    """Tests for EUDR Article 4: Products from post-cutoff deforested areas
    cannot be placed on the EU market."""

    def test_article4_post_cutoff_critical(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Post-cutoff CRITICAL deforestation triggers market ban per Article 3."""
        result = assessor.assess_impact(
            "alert-art4",
            {"cutoff_result": "POST_CUTOFF", "severity": "CRITICAL"},
        )
        assert result["compliance_outcome"] == ComplianceOutcome.NON_COMPLIANT.value
        assert result["market_restriction"] is True
        assert any(
            "Article 3" in ref or "Article 4" in ref or "EUDR" in ref
            for ref in result.get("regulatory_references", [])
        )

    def test_article4_pre_cutoff_no_restriction(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Pre-cutoff products are not restricted per EUDR."""
        result = assessor.assess_impact(
            "alert-art4-pre",
            {"cutoff_result": "PRE_CUTOFF", "severity": "CRITICAL"},
        )
        assert result["market_restriction"] is False


# ---------------------------------------------------------------------------
# TestProvenance
# ---------------------------------------------------------------------------


class TestProvenance:
    """Tests for provenance hash generation and determinism."""

    def test_impact_provenance(
        self, post_cutoff_critical_assessment: Dict[str, Any]
    ) -> None:
        """Impact assessment has provenance hash."""
        assert len(post_cutoff_critical_assessment["provenance_hash"]) == 64

    def test_affected_products_provenance(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Affected products result has provenance hash."""
        result = assessor.get_affected_products("sample_post_cutoff")
        assert len(result["provenance_hash"]) == 64

    def test_recommendations_provenance(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Recommendations result has provenance hash."""
        result = assessor.get_recommendations("sample_post_cutoff")
        assert len(result["provenance_hash"]) == 64

    def test_remediation_plan_provenance(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Remediation plan has provenance hash."""
        result = assessor.create_remediation_plan(
            impact_id="ci-prov-test",
            alert_id="alert-prov",
        )
        assert len(result["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestDataClasses
# ---------------------------------------------------------------------------


class TestDataClasses:
    """Tests for data class serialization."""

    def test_affected_supplier_to_dict(self) -> None:
        """AffectedSupplier serialization works."""
        s = AffectedSupplier(
            supplier_id="SUP-1",
            supplier_name="Test Farm",
            commodity="soya",
            annual_volume_tons=Decimal("5000"),
        )
        d = s.to_dict()
        assert d["supplier_name"] == "Test Farm"
        assert d["annual_volume_tons"] == "5000"

    def test_affected_product_to_dict(self) -> None:
        """AffectedProduct serialization works."""
        p = AffectedProduct(
            product_id="PROD-1",
            product_name="Soybean Meal",
            hs_code="2304",
            commodity="soya",
            annual_value_eur=Decimal("500000"),
            market_restriction=True,
            withdrawal_required=True,
        )
        d = p.to_dict()
        assert d["market_restriction"] is True
        assert d["hs_code"] == "2304"

    def test_compliance_impact_to_dict(self) -> None:
        """ComplianceImpact serialization includes all fields."""
        ci = ComplianceImpact(
            impact_id="ci-1",
            compliance_outcome=ComplianceOutcome.NON_COMPLIANT.value,
            market_restriction=True,
            estimated_financial_impact_eur=Decimal("1000000"),
        )
        d = ci.to_dict()
        assert d["compliance_outcome"] == "NON_COMPLIANT"
        assert d["estimated_financial_impact_eur"] == "1000000"

    def test_remediation_plan_to_dict(self) -> None:
        """RemediationPlan serialization works."""
        rp = RemediationPlan(
            plan_id="rp-1",
            compliance_outcome=ComplianceOutcome.NON_COMPLIANT.value,
            total_estimated_cost_eur=Decimal("75000"),
            total_timeline_days=30,
            priority=RemediationPriority.IMMEDIATE.value,
        )
        d = rp.to_dict()
        assert d["priority"] == "IMMEDIATE"
        assert d["total_timeline_days"] == 30


# ---------------------------------------------------------------------------
# TestDataLoading
# ---------------------------------------------------------------------------


class TestDataLoading:
    """Tests for data loading methods."""

    def test_load_supplier_data(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Load custom supplier data."""
        assessor.load_supplier_data("SUP-TEST", {"name": "Test", "commodity": "cocoa"})
        assert "SUP-TEST" in assessor._custom_suppliers

    def test_load_supplier_empty_id_raises(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Empty supplier_id raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            assessor.load_supplier_data("", {"name": "Test"})

    def test_load_supplier_empty_data_raises(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Empty data raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            assessor.load_supplier_data("SUP-X", {})

    def test_load_product_data(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Load custom product data."""
        assessor.load_product_data("PROD-TEST", {"name": "Test Product"})
        assert "PROD-TEST" in assessor._custom_products

    def test_load_product_empty_id_raises(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Empty product_id raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            assessor.load_product_data("", {"name": "Test"})

    def test_load_alert_mapping(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Load custom alert-to-supplier mapping."""
        assessor.load_alert_supplier_mapping("alert-map", ["SUP-1"])
        assert "alert-map" in assessor._custom_alert_mapping

    def test_load_alert_mapping_empty_id_raises(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Empty alert_id raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            assessor.load_alert_supplier_mapping("", ["SUP-1"])

    def test_load_alert_mapping_empty_suppliers_raises(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Empty supplier_ids raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            assessor.load_alert_supplier_mapping("alert-empty", [])


# ---------------------------------------------------------------------------
# TestGetStatistics
# ---------------------------------------------------------------------------


class TestGetStatistics:
    """Tests for get_statistics."""

    def test_statistics_includes_config(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Statistics include engine configuration."""
        stats = assessor.get_statistics()
        assert stats["engine"] == "ComplianceImpactAssessor"
        assert stats["reference_suppliers"] == len(REFERENCE_SUPPLIERS)
        assert stats["reference_products"] == len(REFERENCE_PRODUCTS)
        assert stats["compliance_rules"] == len(COMPLIANCE_RULES)

    def test_statistics_reflects_cached_assessments(
        self, assessor: ComplianceImpactAssessor
    ) -> None:
        """Statistics reflect cached assessment count."""
        assessor.assess_impact(
            "alert-stat",
            {"cutoff_result": "POST_CUTOFF", "severity": "HIGH"},
        )
        stats = assessor.get_statistics()
        assert stats["assessments_cached"] >= 1


# ---------------------------------------------------------------------------
# TestConstants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_compliance_rules_coverage(self) -> None:
        """All major cutoff_result/severity combinations are covered."""
        for cutoff in ("POST_CUTOFF", "PRE_CUTOFF", "ONGOING", "UNCERTAIN"):
            for severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
                assert (cutoff, severity) in COMPLIANCE_RULES

    def test_severity_ordering(self) -> None:
        """CRITICAL > HIGH > MEDIUM > LOW."""
        assert SEVERITY_ORDER["CRITICAL"] > SEVERITY_ORDER["HIGH"]
        assert SEVERITY_ORDER["HIGH"] > SEVERITY_ORDER["MEDIUM"]
        assert SEVERITY_ORDER["MEDIUM"] > SEVERITY_ORDER["LOW"]

    def test_commodity_prices_exist(self) -> None:
        """All EUDR commodities have reference prices."""
        for commodity in ("cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood"):
            assert commodity in COMMODITY_PRICES_EUR_PER_TON
            assert COMMODITY_PRICES_EUR_PER_TON[commodity] > Decimal("0")

    def test_remediation_costs_exist(self) -> None:
        """All remediation actions have cost estimates."""
        for action in RemediationAction:
            assert action.value in REMEDIATION_COST_ESTIMATES

    def test_remediation_timelines_exist(self) -> None:
        """All remediation actions have timeline estimates."""
        for action in RemediationAction:
            assert action.value in REMEDIATION_TIMELINE_DAYS
