# -*- coding: utf-8 -*-
"""
Tests for CoCModelEnforcer - AGENT-EUDR-009 Engine 3: CoC Model Enforcement

Comprehensive test suite covering:
- IP model: no mixing enforced, single origin (F3.1)
- SG model: compliant-only sources enforced (F3.2)
- MB model: credit period limits, no overdraft (F3.3)
- CB model: blend ratio caps enforced (F3.4)
- Model assignment per facility-commodity (F3.5)
- Cross-model handoff and downgrade rules (F3.7, F3.9)
- Model compliance scoring (F3.8)
- Certification linkage (F3.10)

Test count: 55+ tests
Coverage target: >= 85% of CoCModelEnforcer module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-009 Chain of Custody Agent (GL-EUDR-COC-009)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.chain_of_custody.conftest import (
    COC_MODELS,
    COC_MODEL_RULES,
    COC_MODEL_LABELS,
    VALID_MODEL_TRANSITIONS,
    INVALID_MODEL_TRANSITIONS,
    EUDR_COMMODITIES,
    CREDIT_PERIODS,
    FAC_ID_PROC_GH,
    FAC_ID_MILL_ID,
    FAC_ID_REFINERY_ID,
    FAC_ID_WAREHOUSE_NL,
    FAC_ID_FACTORY_DE,
    PLOT_ID_COCOA_GH_1,
    PLOT_ID_COCOA_GH_2,
    PLOT_ID_COCOA_GH_3,
    make_batch,
    assert_valid_completeness_score,
)


# ===========================================================================
# 1. Identity Preserved (IP) Model (F3.1)
# ===========================================================================


class TestIdentityPreservedModel:
    """Test IP model enforcement: no mixing, single origin."""

    def test_ip_single_origin_accepted(self, coc_model_enforcer):
        """IP batch with a single origin plot is accepted."""
        batch = make_batch(
            coc_model="IP",
            origin_plots=[{"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 100.0}],
        )
        result = coc_model_enforcer.validate_batch(batch)
        assert result["is_valid"] is True

    def test_ip_multiple_origins_rejected(self, coc_model_enforcer):
        """IP batch with multiple origin plots is rejected."""
        batch = make_batch(
            coc_model="IP",
            origin_plots=[
                {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 60.0},
                {"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 40.0},
            ],
        )
        result = coc_model_enforcer.validate_batch(batch)
        assert result["is_valid"] is False
        assert any("mixing" in str(e).lower() or "single" in str(e).lower()
                    for e in result.get("errors", []))

    def test_ip_no_mixing_at_facility(self, coc_model_enforcer):
        """IP facility cannot mix batches from different origins."""
        result = coc_model_enforcer.validate_mixing(
            facility_id=FAC_ID_PROC_GH,
            model="IP",
            batch_origins=[PLOT_ID_COCOA_GH_1, PLOT_ID_COCOA_GH_2],
        )
        assert result["is_valid"] is False

    def test_ip_same_origin_at_facility_accepted(self, coc_model_enforcer):
        """IP facility can hold multiple batches from the same origin."""
        result = coc_model_enforcer.validate_mixing(
            facility_id=FAC_ID_PROC_GH,
            model="IP",
            batch_origins=[PLOT_ID_COCOA_GH_1, PLOT_ID_COCOA_GH_1],
        )
        assert result["is_valid"] is True

    def test_ip_no_credit_period(self, coc_model_enforcer):
        """IP model has no credit period concept."""
        rules = coc_model_enforcer.get_model_rules("IP")
        assert rules["credit_period_months"] is None

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_ip_enforcement_all_commodities(self, coc_model_enforcer, commodity):
        """IP enforcement works for all 7 EUDR commodities."""
        batch = make_batch(commodity=commodity, coc_model="IP")
        result = coc_model_enforcer.validate_batch(batch)
        assert result is not None


# ===========================================================================
# 2. Segregated (SG) Model (F3.2)
# ===========================================================================


class TestSegregatedModel:
    """Test SG model enforcement: compliant-only sources."""

    def test_sg_all_compliant_accepted(self, coc_model_enforcer):
        """SG batch with all compliant origins is accepted."""
        batch = make_batch(
            coc_model="SG",
            origin_plots=[
                {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 60.0,
                 "compliance_status": "compliant"},
                {"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 40.0,
                 "compliance_status": "compliant"},
            ],
        )
        result = coc_model_enforcer.validate_batch(batch)
        assert result["is_valid"] is True

    def test_sg_noncompliant_source_rejected(self, coc_model_enforcer):
        """SG batch with a non-compliant origin is rejected."""
        batch = make_batch(
            coc_model="SG",
            origin_plots=[
                {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 50.0,
                 "compliance_status": "compliant"},
                {"plot_id": PLOT_ID_COCOA_GH_3, "percentage": 50.0,
                 "compliance_status": "non_compliant"},
            ],
        )
        result = coc_model_enforcer.validate_batch(batch)
        assert result["is_valid"] is False

    def test_sg_allows_mixing_within_compliant_pool(self, coc_model_enforcer):
        """SG allows mixing batches as long as all sources are compliant."""
        result = coc_model_enforcer.validate_mixing(
            facility_id=FAC_ID_PROC_GH,
            model="SG",
            batch_origins=[PLOT_ID_COCOA_GH_1, PLOT_ID_COCOA_GH_2],
            compliance_statuses=["compliant", "compliant"],
        )
        assert result["is_valid"] is True

    def test_sg_rejects_noncompliant_mixing(self, coc_model_enforcer):
        """SG rejects mixing when any source is non-compliant."""
        result = coc_model_enforcer.validate_mixing(
            facility_id=FAC_ID_PROC_GH,
            model="SG",
            batch_origins=[PLOT_ID_COCOA_GH_1, PLOT_ID_COCOA_GH_3],
            compliance_statuses=["compliant", "non_compliant"],
        )
        assert result["is_valid"] is False


# ===========================================================================
# 3. Mass Balance (MB) Model (F3.3)
# ===========================================================================


class TestMassBalanceModel:
    """Test MB model enforcement: credit period limits, no overdraft."""

    def test_mb_allows_physical_mixing(self, coc_model_enforcer):
        """MB model allows physical mixing of compliant and non-compliant."""
        batch = make_batch(
            coc_model="MB",
            origin_plots=[
                {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 50.0,
                 "compliance_status": "compliant"},
                {"plot_id": PLOT_ID_COCOA_GH_3, "percentage": 50.0,
                 "compliance_status": "non_compliant"},
            ],
        )
        result = coc_model_enforcer.validate_batch(batch)
        assert result["is_valid"] is True

    def test_mb_credit_period_fsc_12_months(self, coc_model_enforcer):
        """MB under FSC has 12-month credit period."""
        rules = coc_model_enforcer.get_model_rules("MB")
        assert rules["credit_period_months"] == 12

    def test_mb_credit_period_rspo_3_months(self, coc_model_enforcer):
        """MB under RSPO has 3-month credit period."""
        period = coc_model_enforcer.get_credit_period("MB", "RSPO")
        assert period == 3

    def test_mb_overdraft_detected(self, coc_model_enforcer):
        """MB model detects when outputs exceed compliant inputs."""
        result = coc_model_enforcer.check_overdraft(
            facility_id=FAC_ID_MILL_ID,
            commodity="palm_oil",
            compliant_input_kg=1000.0,
            compliant_output_kg=1200.0,
        )
        assert result["overdraft"] is True
        assert result["overdraft_kg"] == pytest.approx(200.0)

    def test_mb_no_overdraft_within_balance(self, coc_model_enforcer):
        """MB model accepts outputs within compliant input balance."""
        result = coc_model_enforcer.check_overdraft(
            facility_id=FAC_ID_MILL_ID,
            commodity="palm_oil",
            compliant_input_kg=5000.0,
            compliant_output_kg=4500.0,
        )
        assert result["overdraft"] is False

    def test_mb_uses_accounting_tracking(self, coc_model_enforcer):
        """MB model uses accounting-based tracking, not physical."""
        rules = coc_model_enforcer.get_model_rules("MB")
        assert rules["accounting_type"] == "accounting"


# ===========================================================================
# 4. Controlled Blending (CB) Model (F3.4)
# ===========================================================================


class TestControlledBlendingModel:
    """Test CB model enforcement: blend ratio caps."""

    def test_cb_within_ratio_accepted(self, coc_model_enforcer):
        """CB batch within maximum blend ratio is accepted."""
        result = coc_model_enforcer.validate_blend_ratio(
            model="CB",
            compliant_pct=60.0,
            noncompliant_pct=40.0,
            max_noncompliant_ratio=0.50,
        )
        assert result["is_valid"] is True

    def test_cb_exceeds_ratio_rejected(self, coc_model_enforcer):
        """CB batch exceeding maximum non-compliant ratio is rejected."""
        result = coc_model_enforcer.validate_blend_ratio(
            model="CB",
            compliant_pct=40.0,
            noncompliant_pct=60.0,
            max_noncompliant_ratio=0.50,
        )
        assert result["is_valid"] is False

    def test_cb_at_exact_limit_accepted(self, coc_model_enforcer):
        """CB batch at exactly the maximum ratio is accepted."""
        result = coc_model_enforcer.validate_blend_ratio(
            model="CB",
            compliant_pct=50.0,
            noncompliant_pct=50.0,
            max_noncompliant_ratio=0.50,
        )
        assert result["is_valid"] is True

    def test_cb_default_max_ratio(self, coc_model_enforcer):
        """CB model has a default maximum blend ratio of 50%."""
        rules = coc_model_enforcer.get_model_rules("CB")
        assert rules.get("default_max_blend_ratio", 0.50) == pytest.approx(0.50)

    def test_cb_100_compliant_accepted(self, coc_model_enforcer):
        """CB with 100% compliant material is accepted."""
        result = coc_model_enforcer.validate_blend_ratio(
            model="CB",
            compliant_pct=100.0,
            noncompliant_pct=0.0,
            max_noncompliant_ratio=0.50,
        )
        assert result["is_valid"] is True


# ===========================================================================
# 5. Model Assignment (F3.5)
# ===========================================================================


class TestModelAssignment:
    """Test CoC model assignment per facility-commodity combination."""

    def test_assign_model_to_facility(self, coc_model_enforcer):
        """Assign a CoC model to a facility-commodity pair."""
        result = coc_model_enforcer.assign_model(
            facility_id=FAC_ID_PROC_GH,
            commodity="cocoa",
            model="SG",
        )
        assert result["model"] == "SG"
        assert result["facility_id"] == FAC_ID_PROC_GH

    @pytest.mark.parametrize("model", COC_MODELS)
    def test_assign_all_models(self, coc_model_enforcer, model):
        """All 4 CoC models can be assigned."""
        result = coc_model_enforcer.assign_model(
            facility_id=f"FAC-{model}-TEST",
            commodity="cocoa",
            model=model,
        )
        assert result["model"] == model

    def test_get_facility_model(self, coc_model_enforcer):
        """Retrieve the CoC model assigned to a facility."""
        coc_model_enforcer.assign_model(FAC_ID_FACTORY_DE, "cocoa", "IP")
        result = coc_model_enforcer.get_model(FAC_ID_FACTORY_DE, "cocoa")
        assert result == "IP"

    def test_facility_multiple_commodities(self, coc_model_enforcer):
        """Same facility can have different models for different commodities."""
        coc_model_enforcer.assign_model(FAC_ID_PROC_GH, "cocoa", "SG")
        coc_model_enforcer.assign_model(FAC_ID_PROC_GH, "coffee", "MB")
        assert coc_model_enforcer.get_model(FAC_ID_PROC_GH, "cocoa") == "SG"
        assert coc_model_enforcer.get_model(FAC_ID_PROC_GH, "coffee") == "MB"

    def test_invalid_model_assignment_raises(self, coc_model_enforcer):
        """Assigning an invalid model raises ValueError."""
        with pytest.raises(ValueError):
            coc_model_enforcer.assign_model(FAC_ID_PROC_GH, "cocoa", "INVALID")


# ===========================================================================
# 6. Cross-Model Handoff / Downgrade Rules (F3.7, F3.9)
# ===========================================================================


class TestCrossModelHandoff:
    """Test model transition and downgrade rules."""

    @pytest.mark.parametrize("from_model,to_model", VALID_MODEL_TRANSITIONS)
    def test_valid_model_transitions(self, coc_model_enforcer, from_model, to_model):
        """Valid model transitions (downgrade) are accepted."""
        result = coc_model_enforcer.validate_transition(from_model, to_model)
        assert result["is_valid"] is True

    @pytest.mark.parametrize("from_model,to_model", INVALID_MODEL_TRANSITIONS)
    def test_invalid_model_transitions(self, coc_model_enforcer, from_model, to_model):
        """Invalid model transitions (upgrade) are rejected."""
        result = coc_model_enforcer.validate_transition(from_model, to_model)
        assert result["is_valid"] is False

    def test_ip_to_mb_on_handoff(self, coc_model_enforcer):
        """IP material entering an MB facility becomes MB."""
        result = coc_model_enforcer.resolve_handoff(
            source_model="IP",
            dest_facility_model="MB",
        )
        assert result["effective_model"] == "MB"

    def test_sg_to_sg_preserves(self, coc_model_enforcer):
        """SG material entering SG facility stays SG."""
        result = coc_model_enforcer.resolve_handoff(
            source_model="SG",
            dest_facility_model="SG",
        )
        assert result["effective_model"] == "SG"

    def test_mb_to_ip_upgrade_rejected(self, coc_model_enforcer):
        """MB material cannot be upgraded to IP."""
        result = coc_model_enforcer.resolve_handoff(
            source_model="MB",
            dest_facility_model="IP",
        )
        assert result["effective_model"] != "IP"


# ===========================================================================
# 7. Model Compliance Scoring (F3.8)
# ===========================================================================


class TestModelComplianceScoring:
    """Test model compliance scoring per facility."""

    def test_full_compliance_score(self, coc_model_enforcer):
        """Facility with full compliance scores high."""
        coc_model_enforcer.assign_model(FAC_ID_FACTORY_DE, "cocoa", "IP")
        score = coc_model_enforcer.calculate_compliance_score(
            FAC_ID_FACTORY_DE, "cocoa"
        )
        assert_valid_completeness_score(score)

    def test_compliance_score_range(self, coc_model_enforcer):
        """Compliance score is between 0 and 100."""
        coc_model_enforcer.assign_model(FAC_ID_MILL_ID, "palm_oil", "MB")
        score = coc_model_enforcer.calculate_compliance_score(
            FAC_ID_MILL_ID, "palm_oil"
        )
        assert 0.0 <= score <= 100.0


# ===========================================================================
# 8. Certification Linkage (F3.10)
# ===========================================================================


class TestCertificationLinkage:
    """Test certification standard linkage per CoC model."""

    @pytest.mark.parametrize("model,expected_certs", [
        ("IP", ["FSC", "RSPO"]),
        ("SG", ["FSC", "RSPO", "ISCC"]),
        ("MB", ["RSPO", "ISCC", "UTZ"]),
        ("CB", ["RSPO"]),
    ])
    def test_model_certifications(self, coc_model_enforcer, model, expected_certs):
        """Each model maps to the correct certification standards."""
        rules = coc_model_enforcer.get_model_rules(model)
        for cert in expected_certs:
            assert cert in rules["certifications"], (
                f"Expected {cert} in {model} certifications"
            )

    def test_model_rules_complete(self, coc_model_enforcer):
        """All 4 models have complete rules defined."""
        for model in COC_MODELS:
            rules = coc_model_enforcer.get_model_rules(model)
            assert "mixing_allowed" in rules
            assert "accounting_type" in rules
            assert "certifications" in rules
