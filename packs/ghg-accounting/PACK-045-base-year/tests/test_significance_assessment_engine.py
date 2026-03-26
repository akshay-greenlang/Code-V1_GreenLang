# -*- coding: utf-8 -*-
"""
Tests for SignificanceAssessmentEngine (Engine 5).

Covers individual assessment, cumulative assessment, sensitivity analysis,
evidence packaging, and recommendation logic.
Target: ~60 tests.
"""

import pytest
from decimal import Decimal
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from engines.significance_assessment_engine import (
    SignificanceAssessmentEngine,
    TriggerInput,
    AssessmentPolicy,
    SignificanceResult,
    TriggerAssessment,
    CumulativeAssessment,
    SensitivityResult,
    SensitivityScenario,
    EvidencePackage,
    EvidenceItem,
    TriggerType,
    AssessmentOutcome,
    SignificanceMethod,
    EvidenceCategory,
)


# ============================================================================
# Engine Init
# ============================================================================

class TestSignificanceAssessmentEngineInit:
    def test_engine_creation(self, significance_engine):
        assert significance_engine is not None

    def test_engine_is_instance(self, significance_engine):
        assert isinstance(significance_engine, SignificanceAssessmentEngine)


# ============================================================================
# Assess Individual
# ============================================================================

class TestAssessIndividual:
    def test_significant_change(self, significance_engine):
        trigger = TriggerInput(
            trigger_id="T-001",
            trigger_type=TriggerType.ACQUISITION,
            emission_impact_tco2e=Decimal("6000"),
            description="Major acquisition",
        )
        result = significance_engine.assess_individual(
            trigger, base_year_total=Decimal("100000"), threshold_pct=Decimal("5.0")
        )
        assert isinstance(result, TriggerAssessment)
        assert result.outcome == AssessmentOutcome.SIGNIFICANT

    def test_non_significant_change(self, significance_engine):
        trigger = TriggerInput(
            trigger_id="T-002",
            trigger_type=TriggerType.ERROR_CORRECTION,
            emission_impact_tco2e=Decimal("1000"),
            description="Minor error fix",
        )
        result = significance_engine.assess_individual(
            trigger, base_year_total=Decimal("100000"), threshold_pct=Decimal("5.0")
        )
        assert result.outcome == AssessmentOutcome.NOT_SIGNIFICANT

    def test_boundary_case_exactly_at_threshold(self, significance_engine):
        trigger = TriggerInput(
            trigger_id="T-003",
            trigger_type=TriggerType.ACQUISITION,
            emission_impact_tco2e=Decimal("5000"),
            description="Exactly at threshold",
        )
        result = significance_engine.assess_individual(
            trigger, base_year_total=Decimal("100000"), threshold_pct=Decimal("5.0")
        )
        assert isinstance(result, TriggerAssessment)
        # 5% exactly at threshold -> SIGNIFICANT or BORDERLINE
        assert result.outcome in (AssessmentOutcome.SIGNIFICANT, AssessmentOutcome.BORDERLINE)

    def test_borderline_case(self, significance_engine):
        """Impact near threshold within margin -> borderline."""
        trigger = TriggerInput(
            trigger_id="T-004",
            trigger_type=TriggerType.METHODOLOGY_CHANGE,
            emission_impact_tco2e=Decimal("4500"),
            description="Near threshold",
        )
        result = significance_engine.assess_individual(
            trigger, base_year_total=Decimal("100000"),
            threshold_pct=Decimal("5.0"),
            borderline_margin_pct=Decimal("1.0"),
        )
        assert isinstance(result, TriggerAssessment)
        # 4.5% is within 1% margin of 5% threshold -> borderline
        assert result.outcome == AssessmentOutcome.BORDERLINE

    def test_zero_impact(self, significance_engine):
        trigger = TriggerInput(
            trigger_id="T-005",
            trigger_type=TriggerType.METHODOLOGY_CHANGE,
            emission_impact_tco2e=Decimal("0"),
            description="No impact change",
        )
        result = significance_engine.assess_individual(
            trigger, base_year_total=Decimal("100000")
        )
        assert result.outcome == AssessmentOutcome.NOT_SIGNIFICANT

    def test_merger_always_significant(self, significance_engine):
        trigger = TriggerInput(
            trigger_id="T-006",
            trigger_type=TriggerType.MERGER,
            emission_impact_tco2e=Decimal("100"),  # Small but merger
            description="Merger",
        )
        result = significance_engine.assess_individual(
            trigger, base_year_total=Decimal("100000"),
            merger_always_significant=True,
        )
        assert result.outcome == AssessmentOutcome.SIGNIFICANT

    def test_significance_pct_calculation(self, significance_engine):
        trigger = TriggerInput(
            trigger_id="T-007",
            trigger_type=TriggerType.ACQUISITION,
            emission_impact_tco2e=Decimal("8000"),
        )
        result = significance_engine.assess_individual(
            trigger, base_year_total=Decimal("100000")
        )
        assert result.significance_pct >= Decimal("7.0")
        assert result.significance_pct <= Decimal("9.0")

    def test_custom_threshold(self, significance_engine):
        trigger = TriggerInput(
            trigger_id="T-008",
            trigger_type=TriggerType.ACQUISITION,
            emission_impact_tco2e=Decimal("3000"),
        )
        # With 2% threshold, 3% impact should be significant
        result = significance_engine.assess_individual(
            trigger, base_year_total=Decimal("100000"),
            threshold_pct=Decimal("2.0"),
        )
        assert result.outcome == AssessmentOutcome.SIGNIFICANT


# ============================================================================
# Assess Cumulative
# ============================================================================

class TestAssessCumulative:
    def test_cumulative_significant(self, significance_engine):
        triggers = [
            TriggerInput(trigger_id="T-C1", trigger_type=TriggerType.ACQUISITION,
                         emission_impact_tco2e=Decimal("4000")),
            TriggerInput(trigger_id="T-C2", trigger_type=TriggerType.ACQUISITION,
                         emission_impact_tco2e=Decimal("4000")),
        ]
        result = significance_engine.assess_cumulative(
            triggers, base_year_total=Decimal("100000"),
            cumulative_threshold_pct=Decimal("5.0"),
        )
        assert isinstance(result, CumulativeAssessment)
        # 8% > 5% -> significant
        assert result.cumulative_outcome == AssessmentOutcome.SIGNIFICANT

    def test_cumulative_not_significant(self, significance_engine):
        triggers = [
            TriggerInput(trigger_id="T-C3", trigger_type=TriggerType.ERROR_CORRECTION,
                         emission_impact_tco2e=Decimal("500")),
        ]
        result = significance_engine.assess_cumulative(
            triggers, base_year_total=Decimal("100000"),
            cumulative_threshold_pct=Decimal("5.0"),
        )
        assert result.cumulative_outcome == AssessmentOutcome.NOT_SIGNIFICANT

    def test_cumulative_empty_triggers(self, significance_engine):
        result = significance_engine.assess_cumulative(
            [], base_year_total=Decimal("100000")
        )
        assert result.cumulative_outcome == AssessmentOutcome.NOT_SIGNIFICANT

    def test_cumulative_trigger_count(self, significance_engine):
        triggers = [
            TriggerInput(trigger_id=f"T-{i}", trigger_type=TriggerType.ACQUISITION,
                         emission_impact_tco2e=Decimal("1000"))
            for i in range(5)
        ]
        result = significance_engine.assess_cumulative(
            triggers, base_year_total=Decimal("100000")
        )
        assert result.trigger_count == 5

    def test_cumulative_impact_calculation(self, significance_engine):
        triggers = [
            TriggerInput(trigger_id="T-X1", trigger_type=TriggerType.ACQUISITION,
                         emission_impact_tco2e=Decimal("3000")),
            TriggerInput(trigger_id="T-X2", trigger_type=TriggerType.DIVESTITURE,
                         emission_impact_tco2e=Decimal("2000")),
        ]
        result = significance_engine.assess_cumulative(
            triggers, base_year_total=Decimal("100000")
        )
        assert result.cumulative_impact_tco2e >= Decimal("0")


# ============================================================================
# Assess Significance (combined)
# ============================================================================

class TestAssessSignificance:
    def test_combined_assessment(self, significance_engine):
        triggers = [
            TriggerInput(trigger_id="T-S1", trigger_type=TriggerType.ACQUISITION,
                         emission_impact_tco2e=Decimal("6000")),
        ]
        result = significance_engine.assess_significance(
            triggers, base_year_total_tco2e=Decimal("100000")
        )
        assert isinstance(result, SignificanceResult)
        assert result.provenance_hash != ""

    def test_combined_with_policy(self, significance_engine):
        triggers = [
            TriggerInput(trigger_id="T-S2", trigger_type=TriggerType.ACQUISITION,
                         emission_impact_tco2e=Decimal("6000")),
        ]
        policy = AssessmentPolicy(
            individual_threshold_pct=Decimal("5.0"),
            cumulative_threshold_pct=Decimal("10.0"),
            assessment_method=SignificanceMethod.COMBINED,
        )
        result = significance_engine.assess_significance(
            triggers, base_year_total_tco2e=Decimal("100000"), policy=policy
        )
        assert isinstance(result, SignificanceResult)

    def test_individual_only_method(self, significance_engine):
        triggers = [
            TriggerInput(trigger_id="T-S3", trigger_type=TriggerType.ACQUISITION,
                         emission_impact_tco2e=Decimal("6000")),
        ]
        policy = AssessmentPolicy(assessment_method=SignificanceMethod.INDIVIDUAL)
        result = significance_engine.assess_significance(
            triggers, base_year_total_tco2e=Decimal("100000"), policy=policy
        )
        assert isinstance(result, SignificanceResult)

    def test_cumulative_only_method(self, significance_engine):
        triggers = [
            TriggerInput(trigger_id="T-S4", trigger_type=TriggerType.ACQUISITION,
                         emission_impact_tco2e=Decimal("6000")),
        ]
        policy = AssessmentPolicy(assessment_method=SignificanceMethod.CUMULATIVE)
        result = significance_engine.assess_significance(
            triggers, base_year_total_tco2e=Decimal("100000"), policy=policy
        )
        assert isinstance(result, SignificanceResult)

    def test_result_has_individual_assessments(self, significance_engine):
        triggers = [
            TriggerInput(trigger_id="T-S5", trigger_type=TriggerType.ACQUISITION,
                         emission_impact_tco2e=Decimal("6000")),
        ]
        result = significance_engine.assess_significance(
            triggers, base_year_total_tco2e=Decimal("100000")
        )
        assert len(result.individual_assessments) >= 1

    def test_result_has_base_year_total(self, significance_engine):
        triggers = [
            TriggerInput(trigger_id="T-S6", trigger_type=TriggerType.ACQUISITION,
                         emission_impact_tco2e=Decimal("6000")),
        ]
        result = significance_engine.assess_significance(
            triggers, base_year_total_tco2e=Decimal("100000")
        )
        assert result.base_year_total_tco2e == Decimal("100000")


# ============================================================================
# Sensitivity Analysis
# ============================================================================

class TestSensitivity:
    def test_run_sensitivity(self, significance_engine):
        triggers = [
            TriggerInput(trigger_id="T-SN1", trigger_type=TriggerType.ACQUISITION,
                         emission_impact_tco2e=Decimal("4500")),
        ]
        results = significance_engine.run_sensitivity(
            triggers, base_year_total=Decimal("100000")
        )
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_sensitivity_scenarios(self, significance_engine):
        triggers = [
            TriggerInput(trigger_id="T-SN2", trigger_type=TriggerType.ACQUISITION,
                         emission_impact_tco2e=Decimal("5000")),
        ]
        results = significance_engine.run_sensitivity(
            triggers, base_year_total=Decimal("100000")
        )
        scenarios = [r.scenario for r in results]
        assert SensitivityScenario.BASE_CASE in scenarios

    def test_sensitivity_with_custom_range(self, significance_engine):
        triggers = [
            TriggerInput(trigger_id="T-SN3", trigger_type=TriggerType.ACQUISITION,
                         emission_impact_tco2e=Decimal("4500")),
        ]
        results = significance_engine.run_sensitivity(
            triggers, base_year_total=Decimal("100000"),
            range_pct=Decimal("0.30"),
        )
        assert len(results) >= 1


# ============================================================================
# Evidence Package
# ============================================================================

class TestEvidencePackage:
    def test_generate_evidence_package(self, significance_engine):
        triggers = [
            TriggerInput(trigger_id="T-EP1", trigger_type=TriggerType.ACQUISITION,
                         emission_impact_tco2e=Decimal("6000")),
        ]
        sig_result = significance_engine.assess_significance(
            triggers, base_year_total_tco2e=Decimal("100000")
        )
        package = significance_engine.generate_evidence_package(sig_result)
        assert isinstance(package, EvidencePackage)

    def test_evidence_has_items(self, significance_engine):
        triggers = [
            TriggerInput(trigger_id="T-EP2", trigger_type=TriggerType.ACQUISITION,
                         emission_impact_tco2e=Decimal("6000")),
        ]
        sig_result = significance_engine.assess_significance(
            triggers, base_year_total_tco2e=Decimal("100000")
        )
        package = significance_engine.generate_evidence_package(sig_result)
        assert len(package.items) >= 1


# ============================================================================
# Recommend Action
# ============================================================================

class TestRecommendAction:
    def test_recommend_action(self, significance_engine):
        triggers = [
            TriggerInput(trigger_id="T-RA1", trigger_type=TriggerType.ACQUISITION,
                         emission_impact_tco2e=Decimal("8000")),
        ]
        sig_result = significance_engine.assess_significance(
            triggers, base_year_total_tco2e=Decimal("100000")
        )
        policy = AssessmentPolicy()
        recommendation = significance_engine.recommend_action(
            individual_assessments=sig_result.individual_assessments,
            cumulative_assessment=sig_result.cumulative_assessment,
            sensitivity_results=sig_result.sensitivity_results,
            recalculation_required=sig_result.recalculation_required,
            policy=policy,
        )
        assert isinstance(recommendation, str)
        assert len(recommendation) > 0


# ============================================================================
# Enums
# ============================================================================

class TestAssessmentEnums:
    def test_assessment_outcome(self):
        assert AssessmentOutcome.SIGNIFICANT is not None
        assert AssessmentOutcome.NOT_SIGNIFICANT is not None
        assert AssessmentOutcome.BORDERLINE is not None
        assert len(AssessmentOutcome) == 3

    def test_significance_method(self):
        assert SignificanceMethod.INDIVIDUAL is not None
        assert SignificanceMethod.CUMULATIVE is not None
        assert SignificanceMethod.COMBINED is not None
        assert len(SignificanceMethod) == 3

    def test_trigger_type(self):
        assert TriggerType.ACQUISITION is not None
        assert len(TriggerType) == 7

    def test_evidence_category(self):
        assert len(EvidenceCategory) >= 1

    def test_sensitivity_scenario(self):
        assert SensitivityScenario.BASE_CASE is not None
        assert SensitivityScenario.LOW_IMPACT is not None
        assert SensitivityScenario.HIGH_IMPACT is not None


# ============================================================================
# Assessment Policy Model
# ============================================================================

class TestAssessmentPolicy:
    def test_create_policy_defaults(self):
        policy = AssessmentPolicy()
        assert policy.individual_threshold_pct == Decimal("5.0")
        assert policy.cumulative_threshold_pct == Decimal("5.0")
        assert policy.borderline_margin_pct == Decimal("1.0")
        assert policy.assessment_method == SignificanceMethod.COMBINED

    def test_create_policy_custom(self):
        policy = AssessmentPolicy(
            individual_threshold_pct=Decimal("3.0"),
            cumulative_threshold_pct=Decimal("8.0"),
            sbti_mode=False,
        )
        assert policy.individual_threshold_pct == Decimal("3.0")
        assert policy.cumulative_threshold_pct == Decimal("8.0")

    def test_sbti_mode_overrides_thresholds(self):
        """When sbti_mode=True, thresholds are overridden to SBTi defaults."""
        policy = AssessmentPolicy(
            individual_threshold_pct=Decimal("3.0"),
            sbti_mode=True,
        )
        # SBTi mode overrides to 5%
        assert policy.individual_threshold_pct == Decimal("5.0")
        assert policy.sbti_mode is True

    def test_policy_merger_flag(self):
        policy = AssessmentPolicy(merger_always_significant=False)
        assert policy.merger_always_significant is False


# ============================================================================
# TriggerInput Model
# ============================================================================

class TestTriggerInput:
    def test_create_trigger_input(self):
        t = TriggerInput(
            trigger_id="T-MI1",
            trigger_type=TriggerType.ACQUISITION,
            emission_impact_tco2e=Decimal("5000"),
        )
        assert t.trigger_type == TriggerType.ACQUISITION

    def test_trigger_input_with_description(self):
        t = TriggerInput(
            trigger_id="T-MI2",
            trigger_type=TriggerType.MERGER,
            emission_impact_tco2e=Decimal("10000"),
            description="Major merger",
        )
        assert t.description == "Major merger"

    def test_trigger_input_scope(self):
        t = TriggerInput(
            trigger_id="T-MI3",
            trigger_type=TriggerType.ACQUISITION,
            emission_impact_tco2e=Decimal("5000"),
            scope="scope_1",
        )
        assert t.scope == "scope_1"
