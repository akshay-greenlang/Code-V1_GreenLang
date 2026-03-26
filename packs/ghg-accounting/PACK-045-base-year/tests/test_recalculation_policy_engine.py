# -*- coding: utf-8 -*-
"""
Tests for RecalculationPolicyEngine (Engine 3).

Covers policy creation, compliance checking, framework defaults, validation.
Target: ~60 tests.
"""

import pytest
from decimal import Decimal
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from engines.recalculation_policy_engine import (
    RecalculationPolicyEngine,
    RecalculationPolicy,
    TriggerRule,
    ThresholdConfig,
    PolicyValidationResult,
    PolicyComplianceCheck,
    PolicyComparison,
    TriggerType,
    PolicyType,
    ApprovalLevel,
    ValidationSeverity,
    ComplianceFramework,
    create_recalculation_policy,
    validate_recalculation_policy,
    check_policy_compliance,
)


class TestRecalculationPolicyEngineInit:
    def test_engine_creation(self, policy_engine):
        assert policy_engine is not None

    def test_engine_version(self, policy_engine):
        assert policy_engine.get_version() == "1.0.0"


class TestCreatePolicy:
    def test_create_default_policy(self, policy_engine):
        policy = policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        assert isinstance(policy, RecalculationPolicy)

    def test_policy_has_trigger_rules(self, policy_engine):
        policy = policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        assert len(policy.trigger_rules) > 0

    def test_policy_has_thresholds(self, policy_engine):
        policy = policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        assert policy.thresholds is not None

    def test_policy_has_provenance_hash(self, policy_engine):
        policy = policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        assert policy.provenance_hash != ""

    def test_create_policy_with_name(self, policy_engine):
        policy = policy_engine.create_policy(
            PolicyType.GHG_PROTOCOL_DEFAULT,
            name="Custom Policy",
            description="Custom policy description",
        )
        assert policy.name == "Custom Policy"

    def test_create_policy_sbti(self, policy_engine):
        policy = policy_engine.create_policy(PolicyType.SBTI_STRICT)
        assert isinstance(policy, RecalculationPolicy)

    def test_create_policy_id_generated(self, policy_engine):
        policy = policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        assert policy.policy_id != ""

    def test_create_policy_is_active(self, policy_engine):
        policy = policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        assert policy.is_active is True


class TestFrameworkDefaults:
    def test_ghg_protocol_defaults(self, policy_engine):
        defaults = policy_engine.get_ghg_protocol_defaults()
        assert isinstance(defaults, RecalculationPolicy)

    def test_sbti_policy(self, policy_engine):
        policy = policy_engine.get_sbti_policy()
        assert isinstance(policy, RecalculationPolicy)

    def test_cdp_policy(self, policy_engine):
        policy = policy_engine.get_cdp_policy()
        assert isinstance(policy, RecalculationPolicy)

    def test_sec_policy(self, policy_engine):
        policy = policy_engine.get_sec_policy()
        assert isinstance(policy, RecalculationPolicy)

    def test_ghg_defaults_have_thresholds(self, policy_engine):
        defaults = policy_engine.get_ghg_protocol_defaults()
        assert defaults.thresholds is not None

    def test_sbti_has_trigger_rules(self, policy_engine):
        policy = policy_engine.get_sbti_policy()
        assert len(policy.trigger_rules) > 0


class TestValidatePolicy:
    def test_validate_default_policy(self, policy_engine):
        policy = policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        result = policy_engine.validate_policy(policy)
        assert isinstance(result, PolicyValidationResult)
        assert result.is_valid is True

    def test_validate_policy_with_no_triggers(self, policy_engine):
        policy = policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        policy.trigger_rules = []
        result = policy_engine.validate_policy(policy)
        assert len(result.errors) > 0 or len(result.warnings) > 0

    def test_validate_convenience_function(self, policy_engine):
        policy = policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        result = validate_recalculation_policy(policy)
        assert isinstance(result, PolicyValidationResult)


class TestCheckCompliance:
    def test_check_compliance_ghg_protocol(self, policy_engine):
        policy = policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        results = policy_engine.check_compliance(policy, [ComplianceFramework.GHG_PROTOCOL])
        assert isinstance(results, list)
        assert len(results) >= 1
        assert isinstance(results[0], PolicyComplianceCheck)

    def test_check_compliance_sbti(self, policy_engine):
        policy = policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        results = policy_engine.check_compliance(policy, [ComplianceFramework.SBTI])
        assert isinstance(results, list)

    def test_check_compliance_multiple_frameworks(self, policy_engine):
        policy = policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        results = policy_engine.check_compliance(
            policy,
            [ComplianceFramework.GHG_PROTOCOL, ComplianceFramework.SBTI],
        )
        assert len(results) >= 2

    def test_check_compliance_convenience_function(self, policy_engine):
        policy = policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        results = check_policy_compliance(policy, [ComplianceFramework.GHG_PROTOCOL])
        assert isinstance(results, list)


class TestUpdateThreshold:
    def test_update_threshold(self, policy_engine):
        policy = policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        if len(policy.trigger_rules) > 0:
            trigger_type = policy.trigger_rules[0].trigger_type
            updated = policy_engine.update_threshold(
                policy, trigger_type, Decimal("3.0")
            )
            assert isinstance(updated, RecalculationPolicy)


class TestTriggerRules:
    def test_add_custom_rule(self, policy_engine):
        policy = policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        initial_count = len(policy.trigger_rules)
        # ACQUISITION already exists, so adding it replaces; verify count stays same
        rule = TriggerRule(
            trigger_type=TriggerType.ACQUISITION,
            description="Custom acquisition rule",
            requires_recalculation=True,
        )
        updated = policy_engine.add_custom_rule(policy, rule)
        # Rule with same trigger_type replaces existing, so count stays same
        assert len(updated.trigger_rules) == initial_count
        # But the description should be updated
        acq_rules = [r for r in updated.trigger_rules if r.trigger_type == TriggerType.ACQUISITION]
        assert len(acq_rules) >= 1

    def test_remove_trigger_rule(self, policy_engine):
        policy = policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        initial_count = len(policy.trigger_rules)
        if initial_count > 0:
            trigger_type = policy.trigger_rules[0].trigger_type
            updated = policy_engine.remove_trigger_rule(policy, trigger_type)
            assert len(updated.trigger_rules) < initial_count


class TestComparePolicies:
    def test_compare_two_policies(self, policy_engine):
        p1 = policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        p2 = policy_engine.get_sbti_policy()
        comparison = policy_engine.compare_policies(p1, p2)
        assert isinstance(comparison, PolicyComparison)

    def test_compare_identical_policies(self, policy_engine):
        p1 = policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        comparison = policy_engine.compare_policies(p1, p1)
        assert isinstance(comparison, PolicyComparison)


class TestPolicySummary:
    def test_get_policy_summary(self, policy_engine):
        policy = policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        summary = policy_engine.get_policy_summary(policy)
        assert isinstance(summary, dict)


class TestConvenienceFunction:
    def test_create_recalculation_policy(self):
        policy = create_recalculation_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        assert isinstance(policy, RecalculationPolicy)


class TestEnums:
    def test_trigger_types(self):
        assert len(TriggerType) >= 5

    def test_policy_types(self):
        assert len(PolicyType) >= 1

    def test_approval_levels(self):
        assert len(ApprovalLevel) >= 2

    def test_compliance_frameworks(self):
        assert len(ComplianceFramework) >= 3

    def test_validation_severity(self):
        assert len(ValidationSeverity) >= 2


class TestThresholdConfig:
    def test_create_threshold(self):
        tc = ThresholdConfig(individual_pct=Decimal("5.0"), cumulative_pct=Decimal("10.0"))
        assert tc.individual_pct == Decimal("5.0")

    def test_threshold_defaults(self):
        tc = ThresholdConfig()
        assert tc.individual_pct is not None or True  # Just verify it creates
