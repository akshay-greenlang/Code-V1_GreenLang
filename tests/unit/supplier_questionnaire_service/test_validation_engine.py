# -*- coding: utf-8 -*-
"""
Unit Tests for ValidationEngine - AGENT-DATA-008
=================================================

Tests all methods of ValidationEngine with 85%+ coverage.
Validates structural, completeness, consistency, framework-specific,
and data quality validation layers, plus batch validation and fix
suggestions.

Test count target: ~80 tests
Author: GreenLang Platform Team / GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pytest

from greenlang.supplier_questionnaire.validation_engine import ValidationEngine
from greenlang.supplier_questionnaire.models import (
    Answer,
    Framework,
    QuestionnaireResponse,
    QuestionnaireTemplate,
    QuestionType,
    TemplateQuestion,
    TemplateSection,
    ValidationCheck,
    ValidationSeverity,
    ValidationSummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _q(
    qid: str = "q1",
    code: str = "C6.1",
    qtype: QuestionType = QuestionType.NUMERIC,
    required: bool = True,
    choices: Optional[List[str]] = None,
    weight: float = 1.0,
) -> TemplateQuestion:
    return TemplateQuestion(
        question_id=qid, code=code, text=f"Question {code}",
        question_type=qtype, required=required, choices=choices,
        weight=weight,
    )


def _a(
    qid: str = "q1",
    value: Any = 100.0,
    evidence: Optional[List[str]] = None,
    confidence: float = 0.8,
) -> Answer:
    return Answer(
        question_id=qid, value=value,
        evidence_refs=evidence or [], confidence=confidence,
    )


def _tpl(
    sections: Optional[List[TemplateSection]] = None,
    framework: Framework = Framework.CUSTOM,
    template_id: str = "tpl-001",
) -> QuestionnaireTemplate:
    return QuestionnaireTemplate(
        template_id=template_id,
        name="Test Template",
        framework=framework,
        sections=sections or [],
    )


def _resp(
    answers: Optional[List[Answer]] = None,
    template_id: str = "tpl-001",
    response_id: str = "resp-001",
) -> QuestionnaireResponse:
    return QuestionnaireResponse(
        response_id=response_id,
        distribution_id="dist-001",
        template_id=template_id,
        supplier_id="sup-001",
        answers=answers or [],
    )


def _sec(name: str = "Section", questions: Optional[List[TemplateQuestion]] = None, weight: float = 1.0) -> TemplateSection:
    return TemplateSection(
        name=name, questions=questions or [], weight=weight,
    )


def _cdp_template() -> QuestionnaireTemplate:
    """Build a CDP template with all required question codes."""
    boundary_qs = [_q(f"b{i}", code) for i, code in enumerate(["C0.1", "C0.2", "C0.3"])]
    methodology_qs = [_q(f"m{i}", code, QuestionType.TEXT) for i, code in enumerate(["C5.1", "C5.2"])]
    target_qs = [_q(f"t{i}", code, QuestionType.TEXT) for i, code in enumerate(["C4.1", "C4.1a", "C4.2"])]
    verification_qs = [_q(f"v{i}", code, QuestionType.TEXT) for i, code in enumerate(["C10.1", "C10.2"])]
    emissions_qs = [_q(f"e{i}", code, QuestionType.NUMERIC) for i, code in enumerate(["C6.1", "C6.3", "C6.5"])]
    governance_qs = [_q(f"g{i}", code, QuestionType.TEXT) for i, code in enumerate(["C1.1", "C1.1a", "C1.2"])]
    return _tpl(
        template_id="cdp-tpl",
        framework=Framework.CDP_CLIMATE,
        sections=[
            _sec("Boundary", boundary_qs),
            _sec("Methodology", methodology_qs),
            _sec("Targets", target_qs),
            _sec("Verification", verification_qs),
            _sec("Emissions", emissions_qs),
            _sec("Governance", governance_qs),
        ],
    )


def _ecovadis_template() -> QuestionnaireTemplate:
    policy_qs = [_q(f"p{i}", code, QuestionType.TEXT) for i, code in enumerate(["ENV.1", "LAB.1", "ETH.1", "SUP.1"])]
    action_qs = [_q(f"a{i}", code, QuestionType.TEXT) for i, code in enumerate(["ENV.2", "LAB.2", "ETH.2", "SUP.3"])]
    cert_qs = [_q("c0", "ENV.4", QuestionType.TEXT)]
    metric_qs = [_q(f"mt{i}", code, QuestionType.NUMERIC) for i, code in enumerate(["ENV.3", "LAB.3", "SUP.2"])]
    return _tpl(
        template_id="ev-tpl", framework=Framework.ECOVADIS,
        sections=[
            _sec("Policies", policy_qs),
            _sec("Actions", action_qs),
            _sec("Certifications", cert_qs),
            _sec("Metrics", metric_qs),
        ],
    )


def _djsi_template() -> QuestionnaireTemplate:
    eco_qs = [_q(f"eco{i}", code, QuestionType.NUMERIC) for i, code in enumerate(["ECO.1", "ECO.2", "ECO.3", "ECO.4"])]
    env_qs = [_q(f"env{i}", code, QuestionType.NUMERIC) for i, code in enumerate(["ENV.1", "ENV.2", "ENV.3", "ENV.4"])]
    soc_qs = [_q(f"soc{i}", code, QuestionType.TEXT) for i, code in enumerate(["SOC.1", "SOC.2", "SOC.3"])]
    return _tpl(
        template_id="djsi-tpl", framework=Framework.DJSI,
        sections=[_sec("Economic", eco_qs), _sec("Environmental", env_qs), _sec("Social", soc_qs)],
    )


def _full_cdp_response(template: QuestionnaireTemplate) -> QuestionnaireResponse:
    answers = []
    for section in template.sections:
        for q in section.questions:
            if q.question_type == QuestionType.NUMERIC:
                answers.append(_a(q.question_id, 1000.0, ["doc.pdf"], 0.9))
            else:
                answers.append(_a(q.question_id, "Provided", ["doc.pdf"], 0.9))
    return _resp(answers=answers, template_id=template.template_id)


# ============================================================================
# TEST CLASS: Initialization
# ============================================================================


class TestValidationEngineInit:

    def test_init_default_config(self):
        engine = ValidationEngine()
        assert engine._strict is False
        assert engine._yoy_threshold == 50.0
        assert engine._min_data_quality == 60.0

    def test_init_custom_config(self):
        engine = ValidationEngine({"strict_mode": True, "yoy_threshold_pct": 30.0, "min_data_quality": 80.0})
        assert engine._strict is True
        assert engine._yoy_threshold == 30.0
        assert engine._min_data_quality == 80.0

    def test_init_stats_zeroed(self):
        engine = ValidationEngine()
        stats = engine.get_statistics()
        assert stats["validations_run"] == 0
        assert stats["batch_validations"] == 0

    def test_init_empty_results_cache(self):
        engine = ValidationEngine()
        assert len(engine._validation_results) == 0


# ============================================================================
# TEST CLASS: validate_structural
# ============================================================================


class TestValidateStructural:

    def test_structural_numeric_valid(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C6.1", QuestionType.NUMERIC)])])
        resp = _resp(answers=[_a("q1", 42.5)])
        checks = engine.validate_structural(resp, tpl)
        assert len(checks) == 1
        assert checks[0].passed is True

    def test_structural_numeric_invalid_string(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C6.1", QuestionType.NUMERIC)])])
        resp = _resp(answers=[_a("q1", "not_a_number")])
        checks = engine.validate_structural(resp, tpl)
        assert checks[0].passed is False
        assert checks[0].severity == ValidationSeverity.ERROR

    def test_structural_yes_no_valid(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C1.1", QuestionType.YES_NO)])])
        resp = _resp(answers=[_a("q1", True)])
        checks = engine.validate_structural(resp, tpl)
        assert checks[0].passed is True

    def test_structural_yes_no_invalid(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C1.1", QuestionType.YES_NO)])])
        resp = _resp(answers=[_a("q1", "yes")])
        checks = engine.validate_structural(resp, tpl)
        assert checks[0].passed is False

    def test_structural_single_choice_valid(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C1.1", QuestionType.SINGLE_CHOICE, choices=["A", "B"])])])
        resp = _resp(answers=[_a("q1", "A")])
        checks = engine.validate_structural(resp, tpl)
        assert checks[0].passed is True

    def test_structural_single_choice_invalid(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C1.1", QuestionType.SINGLE_CHOICE, choices=["A", "B"])])])
        resp = _resp(answers=[_a("q1", "D")])
        checks = engine.validate_structural(resp, tpl)
        assert checks[0].passed is False

    def test_structural_multi_choice_valid(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C1.1", QuestionType.MULTI_CHOICE, choices=["X", "Y", "Z"])])])
        resp = _resp(answers=[_a("q1", ["X", "Z"])])
        checks = engine.validate_structural(resp, tpl)
        assert checks[0].passed is True

    def test_structural_multi_choice_invalid(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C1.1", QuestionType.MULTI_CHOICE, choices=["X", "Y"])])])
        resp = _resp(answers=[_a("q1", ["X", "W"])])
        checks = engine.validate_structural(resp, tpl)
        assert checks[0].passed is False

    def test_structural_percentage_valid(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "P1", QuestionType.PERCENTAGE)])])
        resp = _resp(answers=[_a("q1", 75.0)])
        checks = engine.validate_structural(resp, tpl)
        assert checks[0].passed is True

    def test_structural_percentage_out_of_range(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "P1", QuestionType.PERCENTAGE)])])
        resp = _resp(answers=[_a("q1", 150.0)])
        checks = engine.validate_structural(resp, tpl)
        assert checks[0].passed is False

    def test_structural_percentage_negative(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "P1", QuestionType.PERCENTAGE)])])
        resp = _resp(answers=[_a("q1", -5.0)])
        checks = engine.validate_structural(resp, tpl)
        assert checks[0].passed is False

    def test_structural_text_present(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C0.1", QuestionType.TEXT)])])
        resp = _resp(answers=[_a("q1", "Some text")])
        checks = engine.validate_structural(resp, tpl)
        assert checks[0].passed is True

    def test_structural_text_empty(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C0.1", QuestionType.TEXT)])])
        resp = _resp(answers=[_a("q1", "")])
        checks = engine.validate_structural(resp, tpl)
        assert checks[0].passed is False

    def test_structural_unknown_question(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [])])
        resp = _resp(answers=[_a("q_unknown", "val")])
        checks = engine.validate_structural(resp, tpl)
        assert len(checks) == 1
        assert checks[0].passed is False
        assert checks[0].severity == ValidationSeverity.WARNING

    def test_structural_no_answers(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C0.1")])])
        resp = _resp(answers=[])
        checks = engine.validate_structural(resp, tpl)
        assert len(checks) == 0

    def test_structural_percentage_boundary_zero(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "P1", QuestionType.PERCENTAGE)])])
        resp = _resp(answers=[_a("q1", 0.0)])
        checks = engine.validate_structural(resp, tpl)
        assert checks[0].passed is True

    def test_structural_percentage_boundary_hundred(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "P1", QuestionType.PERCENTAGE)])])
        resp = _resp(answers=[_a("q1", 100.0)])
        checks = engine.validate_structural(resp, tpl)
        assert checks[0].passed is True


# ============================================================================
# TEST CLASS: validate_completeness
# ============================================================================


class TestValidateCompleteness:

    def test_completeness_all_required_answered(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("Sec", [_q("q1", "C0.1", required=True), _q("q2", "C0.2", required=True)])])
        resp = _resp(answers=[_a("q1", "v"), _a("q2", "v")])
        checks = engine.validate_completeness(resp, tpl)
        section_check = [c for c in checks if "100" in c.message]
        assert len(section_check) > 0
        assert section_check[0].passed is True

    def test_completeness_missing_required(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("Sec", [_q("q1", "C0.1", required=True), _q("q2", "C0.2", required=True)])])
        resp = _resp(answers=[_a("q1", "v")])
        checks = engine.validate_completeness(resp, tpl)
        error_checks = [c for c in checks if c.severity == ValidationSeverity.ERROR]
        assert len(error_checks) >= 1

    def test_completeness_optional_not_flagged(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("Sec", [_q("q1", "C0.1", required=False)])])
        resp = _resp(answers=[])
        checks = engine.validate_completeness(resp, tpl)
        errors = [c for c in checks if c.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0

    def test_completeness_section_percentage(self):
        engine = ValidationEngine()
        qs = [_q(f"q{i}", f"C{i}", required=True) for i in range(4)]
        tpl = _tpl(sections=[_sec("Sec", qs)])
        resp = _resp(answers=[_a("q0", "v"), _a("q1", "v")])
        checks = engine.validate_completeness(resp, tpl)
        section_checks = [c for c in checks if "50" in c.message]
        assert len(section_checks) >= 1

    def test_completeness_multiple_sections(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[
            _sec("S1", [_q("q1", "C1", required=True)]),
            _sec("S2", [_q("q2", "C2", required=True)]),
        ])
        resp = _resp(answers=[_a("q1", "v")])
        checks = engine.validate_completeness(resp, tpl)
        failed = [c for c in checks if not c.passed]
        assert len(failed) >= 1


# ============================================================================
# TEST CLASS: validate_consistency
# ============================================================================


class TestValidateConsistency:

    def _emissions_tpl(self):
        return _tpl(sections=[_sec("Em", [
            _q("e1", "C6.1", QuestionType.NUMERIC),
            _q("e3", "C6.3", QuestionType.NUMERIC),
            _q("e5", "C6.5", QuestionType.NUMERIC),
        ])])

    def test_consistency_scope_sum(self):
        engine = ValidationEngine()
        resp = _resp(answers=[_a("e1", 1000.0), _a("e3", 2000.0), _a("e5", 7000.0)])
        checks = engine.validate_consistency(resp, self._emissions_tpl())
        sum_check = [c for c in checks if "10,000" in c.message or "10000" in c.message]
        assert len(sum_check) >= 1

    def test_consistency_scope3_small_warning(self):
        engine = ValidationEngine()
        resp = _resp(answers=[_a("e1", 5000.0), _a("e3", 5000.0), _a("e5", 100.0)])
        checks = engine.validate_consistency(resp, self._emissions_tpl())
        warnings = [c for c in checks if c.severity == ValidationSeverity.WARNING]
        assert len(warnings) >= 1

    def test_consistency_scope3_normal_passes(self):
        engine = ValidationEngine()
        resp = _resp(answers=[_a("e1", 1000.0), _a("e3", 2000.0), _a("e5", 10000.0)])
        checks = engine.validate_consistency(resp, self._emissions_tpl())
        ratio_checks = [c for c in checks if "ratio" in c.message.lower() and c.passed]
        assert len(ratio_checks) >= 1

    def test_consistency_negative_emissions_error(self):
        engine = ValidationEngine()
        resp = _resp(answers=[_a("e1", -500.0), _a("e3", 2000.0), _a("e5", 3000.0)])
        checks = engine.validate_consistency(resp, self._emissions_tpl())
        errors = [c for c in checks if c.severity == ValidationSeverity.ERROR]
        assert len(errors) >= 1

    def test_consistency_negative_energy_error(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("E", [_q("en", "C8.1", QuestionType.NUMERIC)])])
        resp = _resp(answers=[_a("en", -100.0)])
        checks = engine.validate_consistency(resp, tpl)
        errors = [c for c in checks if c.severity == ValidationSeverity.ERROR]
        assert len(errors) >= 1

    def test_consistency_percentage_out_of_range(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("L", [_q("pq", "LAB.3", QuestionType.NUMERIC)])])
        resp = _resp(answers=[_a("pq", 120.0)])
        checks = engine.validate_consistency(resp, tpl)
        errors = [c for c in checks if c.severity == ValidationSeverity.ERROR]
        assert len(errors) >= 1

    def test_consistency_no_applicable(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("M", [_q("q1", "MISC.1", QuestionType.TEXT)])])
        resp = _resp(answers=[_a("q1", "text")])
        checks = engine.validate_consistency(resp, tpl)
        assert len(checks) >= 1

    def test_consistency_string_numeric_value(self):
        engine = ValidationEngine()
        resp = _resp(answers=[_a("e1", "1,000"), _a("e3", "2,000"), _a("e5", "7,000")])
        checks = engine.validate_consistency(resp, self._emissions_tpl())
        sum_checks = [c for c in checks if "10,000" in c.message]
        assert len(sum_checks) >= 1


# ============================================================================
# TEST CLASS: validate_framework CDP
# ============================================================================


class TestValidateFrameworkCDP:

    def test_cdp_full_response_all_pass(self):
        engine = ValidationEngine()
        tpl = _cdp_template()
        resp = _full_cdp_response(tpl)
        checks = engine.validate_framework(resp, tpl)
        assert len(checks) >= 6
        passed = [c for c in checks if c.passed]
        assert len(passed) >= 6

    def test_cdp_missing_boundary_error(self):
        engine = ValidationEngine()
        tpl = _cdp_template()
        checks = engine.validate_framework(_resp(answers=[]), tpl)
        boundary_check = [c for c in checks if "boundary" in c.message.lower()]
        assert len(boundary_check) >= 1
        assert boundary_check[0].passed is False

    def test_cdp_missing_methodology_error(self):
        engine = ValidationEngine()
        checks = engine.validate_framework(_resp(answers=[]), _cdp_template())
        meth = [c for c in checks if "methodology" in c.message.lower()]
        assert len(meth) >= 1 and meth[0].passed is False

    def test_cdp_missing_targets_warning(self):
        engine = ValidationEngine()
        checks = engine.validate_framework(_resp(answers=[]), _cdp_template())
        target = [c for c in checks if "target" in c.message.lower()]
        assert len(target) >= 1 and target[0].severity == ValidationSeverity.WARNING

    def test_cdp_missing_verification_warning(self):
        engine = ValidationEngine()
        checks = engine.validate_framework(_resp(answers=[]), _cdp_template())
        verif = [c for c in checks if "verification" in c.message.lower()]
        assert len(verif) >= 1 and verif[0].severity == ValidationSeverity.WARNING

    def test_cdp_missing_emissions_error(self):
        engine = ValidationEngine()
        checks = engine.validate_framework(_resp(answers=[]), _cdp_template())
        emiss = [c for c in checks if "emissions" in c.message.lower()]
        assert len(emiss) >= 1

    def test_cdp_missing_governance(self):
        engine = ValidationEngine()
        checks = engine.validate_framework(_resp(answers=[]), _cdp_template())
        gov = [c for c in checks if "governance" in c.message.lower()]
        assert len(gov) >= 1

    def test_cdp_partial_boundary(self):
        engine = ValidationEngine()
        tpl = _cdp_template()
        resp = _resp(answers=[_a("b0", "val")])
        checks = engine.validate_framework(resp, tpl)
        boundary = [c for c in checks if "boundary" in c.message.lower()]
        assert len(boundary) >= 1 and boundary[0].passed is False


# ============================================================================
# TEST CLASS: validate_framework EcoVadis
# ============================================================================


class TestValidateFrameworkEcoVadis:

    def test_ecovadis_full_pass(self):
        engine = ValidationEngine()
        tpl = _ecovadis_template()
        answers = [_a(q.question_id, "resp", ["ev.pdf"], 0.9) for s in tpl.sections for q in s.questions]
        checks = engine.validate_framework(_resp(answers=answers), tpl)
        assert len(checks) >= 5

    def test_ecovadis_no_policies_error(self):
        engine = ValidationEngine()
        checks = engine.validate_framework(_resp(answers=[]), _ecovadis_template())
        policy = [c for c in checks if "polic" in c.message.lower()]
        assert len(policy) >= 1 and policy[0].severity == ValidationSeverity.ERROR

    def test_ecovadis_no_actions_warning(self):
        engine = ValidationEngine()
        checks = engine.validate_framework(_resp(answers=[]), _ecovadis_template())
        action = [c for c in checks if "action" in c.message.lower()]
        assert len(action) >= 1 and action[0].severity == ValidationSeverity.WARNING

    def test_ecovadis_no_evidence(self):
        engine = ValidationEngine()
        tpl = _ecovadis_template()
        answers = [_a(q.question_id, "val", [], 0.5) for s in tpl.sections for q in s.questions]
        checks = engine.validate_framework(_resp(answers=answers), tpl)
        ev = [c for c in checks if "evidence" in c.message.lower()]
        assert len(ev) >= 1

    def test_ecovadis_certifications(self):
        engine = ValidationEngine()
        checks = engine.validate_framework(_resp(answers=[]), _ecovadis_template())
        cert = [c for c in checks if "certif" in c.message.lower()]
        assert len(cert) >= 1


# ============================================================================
# TEST CLASS: validate_framework DJSI
# ============================================================================


class TestValidateFrameworkDJSI:

    def test_djsi_full_coverage(self):
        engine = ValidationEngine()
        tpl = _djsi_template()
        answers = [_a(q.question_id, 75.0) for s in tpl.sections for q in s.questions]
        checks = engine.validate_framework(_resp(answers=answers), tpl)
        assert len(checks) == 3 and all(c.passed for c in checks)

    def test_djsi_no_coverage_warnings(self):
        engine = ValidationEngine()
        checks = engine.validate_framework(_resp(answers=[]), _djsi_template())
        assert len(checks) == 3 and all(not c.passed for c in checks)

    def test_djsi_partial_economic(self):
        engine = ValidationEngine()
        tpl = _djsi_template()
        resp = _resp(answers=[_a("eco0", 50.0), _a("eco1", 60.0)])
        checks = engine.validate_framework(resp, tpl)
        eco = [c for c in checks if "Economic" in c.message]
        assert len(eco) >= 1 and eco[0].passed is True

    def test_djsi_below_50pct(self):
        engine = ValidationEngine()
        resp = _resp(answers=[_a("eco0", 50.0)])
        checks = engine.validate_framework(resp, _djsi_template())
        eco = [c for c in checks if "Economic" in c.message]
        assert len(eco) >= 1 and eco[0].passed is False


# ============================================================================
# TEST CLASS: validate_framework Other
# ============================================================================


class TestValidateFrameworkOther:

    def test_custom_framework_info(self):
        engine = ValidationEngine()
        checks = engine.validate_framework(_resp(), _tpl(framework=Framework.CUSTOM))
        assert len(checks) == 1 and checks[0].passed is True and checks[0].severity == ValidationSeverity.INFO


# ============================================================================
# TEST CLASS: validate_data_quality
# ============================================================================


class TestValidateDataQuality:

    def test_quality_high_score(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C6.1", QuestionType.NUMERIC, required=True)])])
        resp = _resp(answers=[_a("q1", 100.0, ["ev.pdf"], 1.0)])
        score = engine.validate_data_quality(resp, tpl)
        assert score >= 90.0 and score <= 100.0

    def test_quality_zero_answers(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C6.1", QuestionType.NUMERIC, required=True)])])
        score = engine.validate_data_quality(_resp(answers=[]), tpl)
        assert score == 0.0

    def test_quality_no_evidence(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C6.1", QuestionType.NUMERIC, required=True)])])
        score = engine.validate_data_quality(_resp(answers=[_a("q1", 50.0, [], 0.5)]), tpl)
        assert 0.0 <= score <= 100.0

    def test_quality_bounds(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C6.1", QuestionType.NUMERIC, required=True)])])
        score = engine.validate_data_quality(_resp(answers=[_a("q1", 50.0, [], 0.0)]), tpl)
        assert 0.0 <= score <= 100.0


# ============================================================================
# TEST CLASS: validate_response (full pipeline)
# ============================================================================


class TestValidateResponse:

    def test_returns_summary(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C6.1", QuestionType.NUMERIC, required=True)])])
        summary = engine.validate_response("r1", tpl, _resp(answers=[_a("q1", 100.0)]))
        assert isinstance(summary, ValidationSummary)
        assert summary.response_id == "r1"

    def test_valid_no_errors(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C6.1", QuestionType.NUMERIC, required=True)])])
        summary = engine.validate_response("r1", tpl, _resp(answers=[_a("q1", 100.0)]))
        assert summary.is_valid is True and summary.error_count == 0

    def test_with_errors(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C6.1", QuestionType.NUMERIC, required=True)])])
        summary = engine.validate_response("r1", tpl, _resp(answers=[_a("q1", "bad")]))
        assert summary.is_valid is False and summary.error_count >= 1

    def test_provenance_hash(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C6.1", QuestionType.NUMERIC, required=True)])])
        summary = engine.validate_response("r1", tpl, _resp(answers=[_a("q1", 50.0)]))
        assert len(summary.provenance_hash) == 64

    def test_updates_stats(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C0.1", QuestionType.TEXT, required=True)])])
        engine.validate_response("r1", tpl, _resp(answers=[_a("q1", "val")]))
        assert engine.get_statistics()["validations_run"] == 1

    def test_caches_result(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C0.1", QuestionType.TEXT, required=True)])])
        engine.validate_response("r1", tpl, _resp(answers=[_a("q1", "val")]))
        assert "r1" in engine._validation_results

    def test_cdp_full_pipeline(self):
        engine = ValidationEngine()
        tpl = _cdp_template()
        summary = engine.validate_response("r1", tpl, _full_cdp_response(tpl))
        assert summary.is_valid is True and summary.total_checks > 10

    def test_all_five_layers(self):
        engine = ValidationEngine()
        tpl = _cdp_template()
        summary = engine.validate_response("r1", tpl, _full_cdp_response(tpl))
        check_types = {c.check_type for c in summary.checks}
        assert "structural" in check_types and "completeness" in check_types
        assert "consistency" in check_types and "framework" in check_types


# ============================================================================
# TEST CLASS: batch_validate
# ============================================================================


class TestBatchValidate:

    def test_batch_multiple(self):
        engine = ValidationEngine()
        q = _q("q1", "C0.1", QuestionType.TEXT, required=True)
        tpl = _tpl(template_id="t1", sections=[_sec("S", [q])])
        r1 = _resp(answers=[_a("q1", "a")], template_id="t1", response_id="r1")
        r2 = _resp(answers=[_a("q1", "b")], template_id="t1", response_id="r2")
        results = engine.batch_validate(["r1", "r2"], {"t1": tpl}, {"r1": r1, "r2": r2})
        assert len(results) == 2

    def test_batch_missing_response(self):
        engine = ValidationEngine()
        assert len(engine.batch_validate(["missing"], {}, {})) == 0

    def test_batch_missing_template(self):
        engine = ValidationEngine()
        r1 = _resp(template_id="tpl-missing", response_id="r1")
        assert len(engine.batch_validate(["r1"], {}, {"r1": r1})) == 0

    def test_batch_updates_stats(self):
        engine = ValidationEngine()
        q = _q("q1", "C0.1", QuestionType.TEXT, required=True)
        tpl = _tpl(template_id="t1", sections=[_sec("S", [q])])
        r1 = _resp(answers=[_a("q1", "v")], template_id="t1", response_id="r1")
        engine.batch_validate(["r1"], {"t1": tpl}, {"r1": r1})
        assert engine.get_statistics()["batch_validations"] == 1

    def test_batch_empty_list(self):
        engine = ValidationEngine()
        assert len(engine.batch_validate([], {}, {})) == 0


# ============================================================================
# TEST CLASS: suggest_fixes
# ============================================================================


class TestSuggestFixes:

    def test_passed_check_empty(self):
        engine = ValidationEngine()
        check = ValidationCheck(check_type="structural", passed=True, message="OK")
        assert len(engine.suggest_fixes(check)) == 0

    def test_structural_fix(self):
        engine = ValidationEngine()
        check = ValidationCheck(check_type="structural", passed=False, message="expected numeric value", suggestion="Fix it")
        suggestions = engine.suggest_fixes(check)
        assert len(suggestions) >= 2 and "Fix it" in suggestions

    def test_completeness_fix(self):
        engine = ValidationEngine()
        check = ValidationCheck(check_type="completeness", passed=False, message="Missing")
        suggestions = engine.suggest_fixes(check)
        assert any("required" in s.lower() for s in suggestions)

    def test_consistency_fix(self):
        engine = ValidationEngine()
        check = ValidationCheck(check_type="consistency", passed=False, message="Negative")
        suggestions = engine.suggest_fixes(check)
        assert any("consistency" in s.lower() or "related" in s.lower() for s in suggestions)

    def test_framework_fix(self):
        engine = ValidationEngine()
        check = ValidationCheck(check_type="framework", passed=False, message="Missing boundary")
        suggestions = engine.suggest_fixes(check)
        assert any("framework" in s.lower() for s in suggestions)

    def test_numeric_keyword(self):
        engine = ValidationEngine()
        check = ValidationCheck(check_type="structural", passed=False, message="expected numeric, got str")
        suggestions = engine.suggest_fixes(check)
        assert any("numeric" in s.lower() or "number" in s.lower() for s in suggestions)

    def test_choice_keyword(self):
        engine = ValidationEngine()
        check = ValidationCheck(check_type="structural", passed=False, message="invalid choice selected")
        suggestions = engine.suggest_fixes(check)
        assert any("select" in s.lower() or "option" in s.lower() for s in suggestions)


# ============================================================================
# TEST CLASS: Provenance
# ============================================================================


class TestProvenance:

    def test_sha256_format(self):
        engine = ValidationEngine()
        h = engine._compute_provenance("test", "data")
        assert len(h) == 64 and re.match(r"^[0-9a-f]{64}$", h)

    def test_deterministic_within_same_second(self):
        engine = ValidationEngine()
        h1 = engine._compute_provenance("a", "b")
        h2 = engine._compute_provenance("a", "b")
        assert h1 == h2


# ============================================================================
# TEST CLASS: Edge cases
# ============================================================================


class TestEdgeCases:

    def test_empty_template(self):
        engine = ValidationEngine()
        summary = engine.validate_response("r1", _tpl(), _resp())
        assert summary.is_valid is True

    def test_numeric_none_value(self):
        engine = ValidationEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "C6.1", QuestionType.NUMERIC)])])
        checks = engine.validate_structural(_resp(answers=[_a("q1", None)]), tpl)
        assert checks[0].passed is False

    def test_large_batch(self):
        engine = ValidationEngine()
        q = _q("q1", "C0.1", QuestionType.TEXT, required=True)
        tpl = _tpl(template_id="t1", sections=[_sec("S", [q])])
        templates = {"t1": tpl}
        responses = {f"r{i}": _resp(answers=[_a("q1", f"v{i}")], template_id="t1", response_id=f"r{i}") for i in range(50)}
        results = engine.batch_validate(list(responses.keys()), templates, responses)
        assert len(results) == 50
