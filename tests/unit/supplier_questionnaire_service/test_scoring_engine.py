# -*- coding: utf-8 -*-
"""
Unit Tests for ScoringEngine - AGENT-DATA-008
===============================================

Tests all methods of ScoringEngine with 85%+ coverage.
Validates CDP (A-D- grading), EcoVadis (0-100), DJSI (0-100),
custom weighted scoring, benchmarking, trend analysis, tier
assignment, grade assignment, normalisation, and section-level scoring.

Test count target: ~80 tests
Author: GreenLang Platform Team / GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest

from greenlang.supplier_questionnaire.scoring_engine import ScoringEngine
from greenlang.supplier_questionnaire.models import (
    Answer,
    CDPGrade,
    Framework,
    PerformanceTier,
    QuestionnaireResponse,
    QuestionnaireScore,
    QuestionnaireTemplate,
    QuestionType,
    TemplateQuestion,
    TemplateSection,
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


def _sec(
    name: str = "Section",
    questions: Optional[List[TemplateQuestion]] = None,
    weight: float = 1.0,
) -> TemplateSection:
    return TemplateSection(
        name=name, questions=questions or [], weight=weight,
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
    supplier_id: str = "sup-001",
) -> QuestionnaireResponse:
    return QuestionnaireResponse(
        response_id=response_id,
        distribution_id="dist-001",
        template_id=template_id,
        supplier_id=supplier_id,
        answers=answers or [],
    )


def _cdp_template() -> QuestionnaireTemplate:
    """A CDP template with 3 sections of required questions."""
    s1 = _sec("Boundary", [
        _q("b1", "C0.1", QuestionType.TEXT),
        _q("b2", "C0.2", QuestionType.TEXT),
    ])
    s2 = _sec("Emissions", [
        _q("e1", "C6.1", QuestionType.NUMERIC),
        _q("e3", "C6.3", QuestionType.NUMERIC),
    ])
    s3 = _sec("Governance", [
        _q("g1", "C1.1", QuestionType.TEXT),
        _q("g2", "C1.2", QuestionType.TEXT),
    ])
    return _tpl(template_id="cdp-tpl", framework=Framework.CDP_CLIMATE, sections=[s1, s2, s3])


def _ecovadis_template() -> QuestionnaireTemplate:
    """An EcoVadis template with 4 theme sections (25% each)."""
    env = _sec("Environment", [_q("env1", "ENV.1"), _q("env2", "ENV.2")])
    lab = _sec("Labor & Human Rights", [_q("lab1", "LAB.1"), _q("lab2", "LAB.2")])
    eth = _sec("Ethics", [_q("eth1", "ETH.1"), _q("eth2", "ETH.2")])
    sup = _sec("Sustainable Procurement", [_q("sup1", "SUP.1"), _q("sup2", "SUP.2")])
    return _tpl(template_id="ev-tpl", framework=Framework.ECOVADIS, sections=[env, lab, eth, sup])


def _djsi_template() -> QuestionnaireTemplate:
    """A DJSI template with 3 dimension sections."""
    eco = _sec("Economic Dimension", [_q("eco1", "ECO.1"), _q("eco2", "ECO.2")])
    env = _sec("Environmental Dimension", [_q("env1", "ENV.1"), _q("env2", "ENV.2")])
    soc = _sec("Social Dimension", [_q("soc1", "SOC.1"), _q("soc2", "SOC.2")])
    return _tpl(template_id="djsi-tpl", framework=Framework.DJSI, sections=[eco, env, soc])


def _full_answers(template: QuestionnaireTemplate, value: Any = 100.0) -> List[Answer]:
    """Build answers for all questions in a template."""
    answers = []
    for sec in template.sections:
        for q in sec.questions:
            if q.question_type == QuestionType.NUMERIC:
                answers.append(_a(q.question_id, value, ["doc.pdf"], 0.9))
            else:
                answers.append(_a(q.question_id, "Detailed " * 30, ["doc.pdf"], 0.9))
    return answers


# ============================================================================
# TEST CLASS: Initialization
# ============================================================================


class TestScoringEngineInit:

    def test_init_default(self):
        engine = ScoringEngine()
        assert engine._default_weight == 1.0

    def test_init_custom_weight(self):
        engine = ScoringEngine({"default_weight": 2.5})
        assert engine._default_weight == 2.5

    def test_init_stats_zeroed(self):
        engine = ScoringEngine()
        stats = engine.get_statistics()
        assert stats["responses_scored"] == 0
        assert stats["cdp_scores"] == 0
        assert stats["ecovadis_scores"] == 0
        assert stats["djsi_scores"] == 0
        assert stats["custom_scores"] == 0

    def test_init_empty_scores(self):
        engine = ScoringEngine()
        assert engine.get_statistics()["active_scores"] == 0

    def test_init_empty_suppliers(self):
        engine = ScoringEngine()
        assert engine.get_statistics()["suppliers_scored"] == 0


# ============================================================================
# TEST CLASS: score_response routing
# ============================================================================


class TestScoreResponse:

    def test_routes_cdp(self):
        engine = ScoringEngine()
        tpl = _cdp_template()
        resp = _resp(answers=_full_answers(tpl), template_id=tpl.template_id)
        score = engine.score_response("r1", tpl, resp, "cdp_climate")
        assert score.framework == Framework.CDP_CLIMATE

    def test_routes_ecovadis(self):
        engine = ScoringEngine()
        tpl = _ecovadis_template()
        resp = _resp(answers=_full_answers(tpl), template_id=tpl.template_id)
        score = engine.score_response("r1", tpl, resp, "ecovadis")
        assert score.framework == Framework.ECOVADIS

    def test_routes_djsi(self):
        engine = ScoringEngine()
        tpl = _djsi_template()
        resp = _resp(answers=_full_answers(tpl), template_id=tpl.template_id)
        score = engine.score_response("r1", tpl, resp, "djsi")
        assert score.framework == Framework.DJSI

    def test_routes_custom_fallback(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "X.1")])])
        resp = _resp(answers=[_a("q1", 50.0)])
        score = engine.score_response("r1", tpl, resp, "unknown_fw")
        assert score.framework == Framework.CUSTOM

    def test_score_stored(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "X.1")])])
        resp = _resp(answers=[_a("q1", 50.0)])
        score = engine.score_response("r1", tpl, resp)
        retrieved = engine.get_score(score.score_id)
        assert retrieved.score_id == score.score_id

    def test_score_response_updates_stats(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "X.1")])])
        resp = _resp(answers=[_a("q1", 50.0)])
        engine.score_response("r1", tpl, resp)
        assert engine.get_statistics()["responses_scored"] == 1

    def test_score_response_sets_identifiers(self):
        engine = ScoringEngine()
        tpl = _tpl(template_id="tpl-99", sections=[_sec("S", [_q("q1", "X.1")])])
        resp = _resp(answers=[_a("q1", 50.0)], template_id="tpl-99", supplier_id="sup-99")
        score = engine.score_response("r-99", tpl, resp)
        assert score.response_id == "r-99"
        assert score.template_id == "tpl-99"
        assert score.supplier_id == "sup-99"

    def test_score_response_provenance(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "X.1")])])
        score = engine.score_response("r1", tpl, _resp(answers=[_a("q1", 10.0)]))
        assert len(score.provenance_hash) == 64
        assert re.match(r"^[0-9a-f]{64}$", score.provenance_hash)


# ============================================================================
# TEST CLASS: score_cdp
# ============================================================================


class TestScoreCDP:

    def test_cdp_full_answers_high_score(self):
        engine = ScoringEngine()
        tpl = _cdp_template()
        resp = _resp(answers=_full_answers(tpl), template_id=tpl.template_id)
        score = engine.score_cdp(resp, tpl)
        assert score.normalized_score >= 70.0
        assert score.cdp_grade is not None

    def test_cdp_empty_response_zero(self):
        engine = ScoringEngine()
        tpl = _cdp_template()
        score = engine.score_cdp(_resp(answers=[], template_id=tpl.template_id), tpl)
        assert score.normalized_score == 0.0 or score.normalized_score < 10.0
        assert score.cdp_grade in (CDPGrade.F, CDPGrade.D_MINUS)

    def test_cdp_section_scores_populated(self):
        engine = ScoringEngine()
        tpl = _cdp_template()
        resp = _resp(answers=_full_answers(tpl), template_id=tpl.template_id)
        score = engine.score_cdp(resp, tpl)
        assert len(score.section_scores) >= 3
        assert "Boundary" in score.section_scores
        assert "Emissions" in score.section_scores
        assert "Governance" in score.section_scores

    def test_cdp_methodology_string(self):
        engine = ScoringEngine()
        tpl = _cdp_template()
        score = engine.score_cdp(_resp(answers=_full_answers(tpl), template_id=tpl.template_id), tpl)
        assert "CDP" in score.methodology

    def test_cdp_partial_answers(self):
        engine = ScoringEngine()
        tpl = _cdp_template()
        answers = _full_answers(tpl)[:3]
        score = engine.score_cdp(_resp(answers=answers, template_id=tpl.template_id), tpl)
        assert 0.0 <= score.normalized_score <= 100.0

    def test_cdp_increments_stats(self):
        engine = ScoringEngine()
        tpl = _cdp_template()
        engine.score_cdp(_resp(answers=_full_answers(tpl), template_id=tpl.template_id), tpl)
        assert engine.get_statistics()["cdp_scores"] == 1

    def test_cdp_answer_quality_with_evidence(self):
        engine = ScoringEngine()
        tpl = _cdp_template()
        answers = [_a(q.question_id, 100.0, ["ev.pdf"], 1.0) for sec in tpl.sections for q in sec.questions]
        score = engine.score_cdp(_resp(answers=answers, template_id=tpl.template_id), tpl)
        assert score.normalized_score >= 60.0

    def test_cdp_answer_quality_no_evidence(self):
        engine = ScoringEngine()
        tpl = _cdp_template()
        answers = [_a(q.question_id, 100.0, [], 0.5) for sec in tpl.sections for q in sec.questions]
        score_no_ev = engine.score_cdp(_resp(answers=answers, template_id=tpl.template_id), tpl)
        answers_ev = [_a(q.question_id, 100.0, ["ev.pdf"], 0.9) for sec in tpl.sections for q in sec.questions]
        score_ev = engine.score_cdp(_resp(answers=answers_ev, template_id=tpl.template_id, response_id="r2"), tpl)
        assert score_ev.normalized_score >= score_no_ev.normalized_score

    def test_cdp_empty_sections_zero(self):
        engine = ScoringEngine()
        tpl = _tpl(template_id="empty-cdp", framework=Framework.CDP_CLIMATE, sections=[])
        score = engine.score_cdp(_resp(answers=[], template_id="empty-cdp"), tpl)
        assert score.normalized_score == 0.0


# ============================================================================
# TEST CLASS: score_ecovadis
# ============================================================================


class TestScoreEcoVadis:

    def test_ecovadis_full_answers(self):
        engine = ScoringEngine()
        tpl = _ecovadis_template()
        resp = _resp(answers=_full_answers(tpl), template_id=tpl.template_id)
        score = engine.score_ecovadis(resp, tpl)
        assert 0.0 <= score.normalized_score <= 100.0
        assert score.cdp_grade is None

    def test_ecovadis_empty_zero(self):
        engine = ScoringEngine()
        tpl = _ecovadis_template()
        score = engine.score_ecovadis(_resp(answers=[], template_id=tpl.template_id), tpl)
        assert score.normalized_score == 0.0

    def test_ecovadis_section_scores(self):
        engine = ScoringEngine()
        tpl = _ecovadis_template()
        resp = _resp(answers=_full_answers(tpl), template_id=tpl.template_id)
        score = engine.score_ecovadis(resp, tpl)
        assert "Environment" in score.section_scores
        assert "Labor & Human Rights" in score.section_scores
        assert "Ethics" in score.section_scores
        assert "Sustainable Procurement" in score.section_scores

    def test_ecovadis_methodology_string(self):
        engine = ScoringEngine()
        tpl = _ecovadis_template()
        score = engine.score_ecovadis(_resp(answers=_full_answers(tpl), template_id=tpl.template_id), tpl)
        assert "EcoVadis" in score.methodology

    def test_ecovadis_increments_stats(self):
        engine = ScoringEngine()
        tpl = _ecovadis_template()
        engine.score_ecovadis(_resp(answers=_full_answers(tpl), template_id=tpl.template_id), tpl)
        assert engine.get_statistics()["ecovadis_scores"] == 1

    def test_ecovadis_partial_themes(self):
        engine = ScoringEngine()
        tpl = _ecovadis_template()
        answers = [_a("env1", 80.0, ["doc.pdf"], 0.9), _a("env2", 90.0, ["doc.pdf"], 0.9)]
        score = engine.score_ecovadis(_resp(answers=answers, template_id=tpl.template_id), tpl)
        assert score.normalized_score > 0.0
        assert score.section_scores.get("Environment", 0.0) > 0.0

    def test_ecovadis_performance_tier(self):
        engine = ScoringEngine()
        tpl = _ecovadis_template()
        score = engine.score_ecovadis(_resp(answers=_full_answers(tpl), template_id=tpl.template_id), tpl)
        assert score.performance_tier in [t for t in PerformanceTier]

    def test_ecovadis_provenance(self):
        engine = ScoringEngine()
        tpl = _ecovadis_template()
        score = engine.score_ecovadis(_resp(answers=_full_answers(tpl), template_id=tpl.template_id), tpl)
        assert len(score.provenance_hash) == 64


# ============================================================================
# TEST CLASS: score_djsi
# ============================================================================


class TestScoreDJSI:

    def test_djsi_full_answers(self):
        engine = ScoringEngine()
        tpl = _djsi_template()
        resp = _resp(answers=_full_answers(tpl), template_id=tpl.template_id)
        score = engine.score_djsi(resp, tpl)
        assert 0.0 <= score.normalized_score <= 100.0

    def test_djsi_empty_zero(self):
        engine = ScoringEngine()
        tpl = _djsi_template()
        score = engine.score_djsi(_resp(answers=[], template_id=tpl.template_id), tpl)
        assert score.normalized_score == 0.0

    def test_djsi_section_scores(self):
        engine = ScoringEngine()
        tpl = _djsi_template()
        score = engine.score_djsi(_resp(answers=_full_answers(tpl), template_id=tpl.template_id), tpl)
        assert "Economic Dimension" in score.section_scores
        assert "Environmental Dimension" in score.section_scores
        assert "Social Dimension" in score.section_scores

    def test_djsi_methodology_string(self):
        engine = ScoringEngine()
        tpl = _djsi_template()
        score = engine.score_djsi(_resp(answers=_full_answers(tpl), template_id=tpl.template_id), tpl)
        assert "DJSI" in score.methodology

    def test_djsi_increments_stats(self):
        engine = ScoringEngine()
        tpl = _djsi_template()
        engine.score_djsi(_resp(answers=_full_answers(tpl), template_id=tpl.template_id), tpl)
        assert engine.get_statistics()["djsi_scores"] == 1

    def test_djsi_provenance(self):
        engine = ScoringEngine()
        tpl = _djsi_template()
        score = engine.score_djsi(_resp(answers=_full_answers(tpl), template_id=tpl.template_id), tpl)
        assert len(score.provenance_hash) == 64

    def test_djsi_no_cdp_grade(self):
        engine = ScoringEngine()
        tpl = _djsi_template()
        score = engine.score_djsi(_resp(answers=_full_answers(tpl), template_id=tpl.template_id), tpl)
        assert score.cdp_grade is None


# ============================================================================
# TEST CLASS: score_custom
# ============================================================================


class TestScoreCustom:

    def test_custom_basic(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("S1", [_q("q1", "X.1")], weight=1.0)])
        resp = _resp(answers=[_a("q1", 80.0, ["doc.pdf"], 0.9)])
        score = engine.score_custom(resp, tpl)
        assert 0.0 <= score.normalized_score <= 100.0
        assert score.framework == Framework.CUSTOM

    def test_custom_with_weight_overrides(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[
            _sec("S1", [_q("q1", "X.1")], weight=1.0),
            _sec("S2", [_q("q2", "X.2")], weight=1.0),
        ])
        resp = _resp(answers=[_a("q1", 90.0, ["doc.pdf"], 0.9), _a("q2", 10.0, [], 0.3)])
        score1 = engine.score_custom(resp, tpl)
        score2 = engine.score_custom(resp, tpl, weights={"S1": 10.0, "S2": 0.1})
        assert score2.normalized_score >= score1.normalized_score - 5.0

    def test_custom_empty_response(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("S1", [_q("q1", "X.1")])])
        score = engine.score_custom(_resp(answers=[]), tpl)
        assert score.normalized_score == 0.0

    def test_custom_increments_stats(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("S1", [_q("q1", "X.1")])])
        engine.score_custom(_resp(answers=[_a("q1", 50.0)]), tpl)
        assert engine.get_statistics()["custom_scores"] == 1

    def test_custom_section_scores(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("Alpha", [_q("q1", "X.1")]), _sec("Beta", [_q("q2", "X.2")])])
        resp = _resp(answers=[_a("q1", 50.0), _a("q2", 70.0)])
        score = engine.score_custom(resp, tpl)
        assert "Alpha" in score.section_scores
        assert "Beta" in score.section_scores

    def test_custom_methodology(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("S1", [_q("q1", "X.1")])])
        score = engine.score_custom(_resp(answers=[_a("q1", 50.0)]), tpl)
        assert "custom" in score.methodology.lower() or "Custom" in score.methodology


# ============================================================================
# TEST CLASS: get_score
# ============================================================================


class TestGetScore:

    def test_get_existing_score(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "X.1")])])
        score = engine.score_response("r1", tpl, _resp(answers=[_a("q1", 50.0)]))
        assert engine.get_score(score.score_id).score_id == score.score_id

    def test_get_unknown_score_raises(self):
        engine = ScoringEngine()
        with pytest.raises(ValueError, match="Unknown score"):
            engine.get_score("nonexistent-id")


# ============================================================================
# TEST CLASS: get_supplier_scores
# ============================================================================


class TestGetSupplierScores:

    def test_get_supplier_scores_empty(self):
        engine = ScoringEngine()
        assert engine.get_supplier_scores("sup-999") == []

    def test_get_supplier_scores_multiple(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "X.1")])])
        engine.score_response("r1", tpl, _resp(answers=[_a("q1", 50.0)], supplier_id="sup-A"))
        engine.score_response("r2", tpl, _resp(answers=[_a("q1", 60.0)], supplier_id="sup-A", response_id="resp-002"))
        engine.score_response("r3", tpl, _resp(answers=[_a("q1", 70.0)], supplier_id="sup-B", response_id="resp-003"))
        scores = engine.get_supplier_scores("sup-A")
        assert len(scores) == 2
        assert all(s.supplier_id == "sup-A" for s in scores)


# ============================================================================
# TEST CLASS: benchmark_supplier
# ============================================================================


class TestBenchmarkSupplier:

    def test_benchmark_no_scores(self):
        engine = ScoringEngine()
        result = engine.benchmark_supplier("sup-A", "cdp_climate", "energy")
        assert result["supplier_score"] is None
        assert result["message"] == "No scores found for this supplier/framework"

    def test_benchmark_with_scores(self):
        engine = ScoringEngine()
        tpl = _cdp_template()
        resp = _resp(answers=_full_answers(tpl), template_id=tpl.template_id, supplier_id="sup-A")
        engine.score_response("r1", tpl, resp, "cdp_climate")
        result = engine.benchmark_supplier("sup-A", "cdp_climate", "energy")
        assert result["supplier_score"] is not None
        assert result["industry_avg"] == 55.0
        assert 1 <= result["percentile"] <= 99

    def test_benchmark_default_industry(self):
        engine = ScoringEngine()
        tpl = _cdp_template()
        resp = _resp(answers=_full_answers(tpl), template_id=tpl.template_id, supplier_id="sup-A")
        engine.score_response("r1", tpl, resp, "cdp_climate")
        result = engine.benchmark_supplier("sup-A", "cdp_climate")
        assert result["industry"] == "default"
        assert result["industry_avg"] == 45.0

    def test_benchmark_updates_stats(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "X.1")])])
        engine.score_response("r1", tpl, _resp(answers=[_a("q1", 50.0)], supplier_id="sup-A"))
        engine.benchmark_supplier("sup-A", "custom")
        assert engine.get_statistics()["benchmarks_generated"] == 1

    def test_benchmark_provenance(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "X.1")])])
        engine.score_response("r1", tpl, _resp(answers=[_a("q1", 50.0)], supplier_id="sup-A"))
        result = engine.benchmark_supplier("sup-A", "custom")
        assert len(result["provenance_hash"]) == 64

    def test_benchmark_unknown_industry(self):
        engine = ScoringEngine()
        tpl = _cdp_template()
        resp = _resp(answers=_full_answers(tpl), template_id=tpl.template_id, supplier_id="sup-A")
        engine.score_response("r1", tpl, resp, "cdp_climate")
        result = engine.benchmark_supplier("sup-A", "cdp_climate", "alien_sector")
        assert result["industry_avg"] is not None


# ============================================================================
# TEST CLASS: get_trend
# ============================================================================


class TestGetTrend:

    def test_trend_no_scores(self):
        engine = ScoringEngine()
        result = engine.get_trend("sup-X", "cdp_climate")
        assert result["total_scores"] == 0
        assert result["trend_direction"] == "stable"

    def test_trend_single_score(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "X.1")])])
        engine.score_response("r1", tpl, _resp(answers=[_a("q1", 50.0)], supplier_id="sup-A"))
        result = engine.get_trend("sup-A", "custom")
        assert result["total_scores"] == 1
        assert result["trend_direction"] == "stable"
        assert result["data_points"][0]["change"] == 0.0

    def test_trend_improving(self):
        engine = ScoringEngine()
        tpl = _cdp_template()
        resp1 = _resp(answers=_full_answers(tpl), template_id=tpl.template_id, supplier_id="sup-A")
        s1 = engine.score_response("r1", tpl, resp1, "cdp_climate")
        s1.normalized_score = 40.0
        s1.scored_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        resp2 = _resp(answers=_full_answers(tpl), template_id=tpl.template_id, supplier_id="sup-A", response_id="resp-002")
        s2 = engine.score_response("r2", tpl, resp2, "cdp_climate")
        s2.normalized_score = 70.0
        s2.scored_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = engine.get_trend("sup-A", "cdp_climate")
        assert result["total_scores"] == 2
        assert result["trend_direction"] == "improving"

    def test_trend_declining(self):
        engine = ScoringEngine()
        tpl = _cdp_template()
        resp1 = _resp(answers=_full_answers(tpl), template_id=tpl.template_id, supplier_id="sup-A")
        s1 = engine.score_response("r1", tpl, resp1, "cdp_climate")
        s1.normalized_score = 80.0
        s1.scored_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        resp2 = _resp(answers=_full_answers(tpl), template_id=tpl.template_id, supplier_id="sup-A", response_id="resp-002")
        s2 = engine.score_response("r2", tpl, resp2, "cdp_climate")
        s2.normalized_score = 30.0
        s2.scored_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = engine.get_trend("sup-A", "cdp_climate")
        assert result["trend_direction"] == "declining"

    def test_trend_updates_stats(self):
        engine = ScoringEngine()
        engine.get_trend("sup-X", "custom")
        assert engine.get_statistics()["trends_generated"] == 1

    def test_trend_data_points_have_change(self):
        engine = ScoringEngine()
        tpl = _cdp_template()
        resp = _resp(answers=_full_answers(tpl), template_id=tpl.template_id, supplier_id="sup-A")
        s1 = engine.score_response("r1", tpl, resp, "cdp_climate")
        s1.scored_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        resp2 = _resp(answers=_full_answers(tpl), template_id=tpl.template_id, supplier_id="sup-A", response_id="r2")
        s2 = engine.score_response("r2", tpl, resp2, "cdp_climate")
        s2.scored_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = engine.get_trend("sup-A", "cdp_climate")
        assert len(result["data_points"]) == 2
        assert "change" in result["data_points"][0]


# ============================================================================
# TEST CLASS: assign_performance_tier
# ============================================================================


class TestAssignPerformanceTier:

    def test_leader(self):
        engine = ScoringEngine()
        assert engine.assign_performance_tier(90.0) == PerformanceTier.LEADER

    def test_leader_boundary(self):
        engine = ScoringEngine()
        assert engine.assign_performance_tier(80.0) == PerformanceTier.LEADER

    def test_advanced(self):
        engine = ScoringEngine()
        assert engine.assign_performance_tier(79.9) == PerformanceTier.ADVANCED

    def test_advanced_boundary(self):
        engine = ScoringEngine()
        assert engine.assign_performance_tier(60.0) == PerformanceTier.ADVANCED

    def test_intermediate(self):
        engine = ScoringEngine()
        assert engine.assign_performance_tier(50.0) == PerformanceTier.INTERMEDIATE

    def test_intermediate_boundary(self):
        engine = ScoringEngine()
        assert engine.assign_performance_tier(40.0) == PerformanceTier.INTERMEDIATE

    def test_beginner(self):
        engine = ScoringEngine()
        assert engine.assign_performance_tier(30.0) == PerformanceTier.BEGINNER

    def test_beginner_boundary(self):
        engine = ScoringEngine()
        assert engine.assign_performance_tier(20.0) == PerformanceTier.BEGINNER

    def test_laggard(self):
        engine = ScoringEngine()
        assert engine.assign_performance_tier(10.0) == PerformanceTier.LAGGARD

    def test_laggard_zero(self):
        engine = ScoringEngine()
        assert engine.assign_performance_tier(0.0) == PerformanceTier.LAGGARD

    def test_100_is_leader(self):
        engine = ScoringEngine()
        assert engine.assign_performance_tier(100.0) == PerformanceTier.LEADER


# ============================================================================
# TEST CLASS: assign_cdp_grade
# ============================================================================


class TestAssignCDPGrade:

    def test_grade_A(self):
        engine = ScoringEngine()
        assert engine.assign_cdp_grade(95.0) == CDPGrade.A

    def test_grade_A_boundary(self):
        engine = ScoringEngine()
        assert engine.assign_cdp_grade(90.0) == CDPGrade.A

    def test_grade_A_minus(self):
        engine = ScoringEngine()
        assert engine.assign_cdp_grade(85.0) == CDPGrade.A_MINUS

    def test_grade_B(self):
        engine = ScoringEngine()
        assert engine.assign_cdp_grade(75.0) == CDPGrade.B

    def test_grade_B_minus(self):
        engine = ScoringEngine()
        assert engine.assign_cdp_grade(65.0) == CDPGrade.B_MINUS

    def test_grade_C(self):
        engine = ScoringEngine()
        assert engine.assign_cdp_grade(55.0) == CDPGrade.C

    def test_grade_C_minus(self):
        engine = ScoringEngine()
        assert engine.assign_cdp_grade(45.0) == CDPGrade.C_MINUS

    def test_grade_D(self):
        engine = ScoringEngine()
        assert engine.assign_cdp_grade(35.0) == CDPGrade.D

    def test_grade_D_minus(self):
        engine = ScoringEngine()
        assert engine.assign_cdp_grade(25.0) == CDPGrade.D_MINUS

    def test_grade_F(self):
        engine = ScoringEngine()
        assert engine.assign_cdp_grade(10.0) == CDPGrade.F

    def test_grade_F_zero(self):
        engine = ScoringEngine()
        assert engine.assign_cdp_grade(0.0) == CDPGrade.F


# ============================================================================
# TEST CLASS: normalize_score
# ============================================================================


class TestNormalizeScore:

    def test_in_range(self):
        engine = ScoringEngine()
        assert engine.normalize_score(55.5, "custom") == 55.5

    def test_clamp_low(self):
        engine = ScoringEngine()
        assert engine.normalize_score(-10.0, "custom") == 0.0

    def test_clamp_high(self):
        engine = ScoringEngine()
        assert engine.normalize_score(150.0, "custom") == 100.0

    def test_zero(self):
        engine = ScoringEngine()
        assert engine.normalize_score(0.0, "custom") == 0.0

    def test_hundred(self):
        engine = ScoringEngine()
        assert engine.normalize_score(100.0, "custom") == 100.0

    def test_rounding(self):
        engine = ScoringEngine()
        assert engine.normalize_score(55.55, "custom") == 55.6


# ============================================================================
# TEST CLASS: get_statistics
# ============================================================================


class TestGetStatistics:

    def test_statistics_timestamp(self):
        engine = ScoringEngine()
        stats = engine.get_statistics()
        assert "timestamp" in stats

    def test_statistics_after_scoring(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "X.1")])])
        engine.score_response("r1", tpl, _resp(answers=[_a("q1", 50.0)]))
        stats = engine.get_statistics()
        assert stats["responses_scored"] == 1
        assert stats["active_scores"] == 1
        assert stats["suppliers_scored"] == 1


# ============================================================================
# TEST CLASS: Provenance
# ============================================================================


class TestProvenance:

    def test_sha256_format(self):
        engine = ScoringEngine()
        h = engine._compute_provenance("test", "data")
        assert len(h) == 64
        assert re.match(r"^[0-9a-f]{64}$", h)

    def test_deterministic_within_same_second(self):
        engine = ScoringEngine()
        h1 = engine._compute_provenance("a", "b")
        h2 = engine._compute_provenance("a", "b")
        assert h1 == h2


# ============================================================================
# TEST CLASS: Edge cases
# ============================================================================


class TestEdgeCases:

    def test_template_no_sections(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[])
        score = engine.score_custom(_resp(answers=[]), tpl)
        assert score.normalized_score == 0.0

    def test_template_empty_section(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("Empty", [])])
        score = engine.score_custom(_resp(answers=[]), tpl)
        assert score.normalized_score == 0.0

    def test_zero_weight_questions(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "X.1", weight=0.0)])])
        score = engine.score_custom(_resp(answers=[_a("q1", 50.0)]), tpl)
        assert score.normalized_score == 0.0

    def test_boolean_answer_depth(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "X.1", QuestionType.YES_NO)])])
        resp = _resp(answers=[_a("q1", True, ["doc.pdf"], 0.9)])
        score = engine.score_custom(resp, tpl)
        assert 0.0 <= score.normalized_score <= 100.0

    def test_list_answer_depth(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("S", [_q("q1", "X.1", QuestionType.MULTI_CHOICE, choices=["A", "B", "C"])])])
        resp = _resp(answers=[_a("q1", ["A", "B", "C"], ["doc.pdf"], 0.9)])
        score = engine.score_custom(resp, tpl)
        assert 0.0 <= score.normalized_score <= 100.0

    def test_evidence_higher_quality(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("S1", [_q("q1", "Q1", QuestionType.NUMERIC)])])
        resp_ev = _resp(answers=[_a("q1", 100.0, ["ev.pdf"], 0.9)])
        score_ev = engine.score_custom(resp_ev, tpl)
        resp_no = _resp(answers=[_a("q1", 100.0, [], 0.3)], response_id="resp-002")
        score_no = engine.score_custom(resp_no, tpl)
        assert score_ev.normalized_score >= score_no.normalized_score

    def test_long_text_higher_depth(self):
        engine = ScoringEngine()
        tpl = _tpl(sections=[_sec("S1", [_q("q1", "Q1", QuestionType.TEXT)])])
        resp_long = _resp(answers=[_a("q1", "A" * 300, [], 0.8)])
        resp_short = _resp(answers=[_a("q1", "OK", [], 0.8)], response_id="resp-002")
        score_long = engine.score_custom(resp_long, tpl)
        score_short = engine.score_custom(resp_short, tpl)
        assert score_long.normalized_score >= score_short.normalized_score

    def test_multiple_scores_same_supplier(self):
        engine = ScoringEngine()
        tpl = _cdp_template()
        resp = _resp(answers=_full_answers(tpl), template_id=tpl.template_id, supplier_id="s1")
        engine.score_response("r1", tpl, resp, "cdp_climate")
        engine.score_response("r2", tpl, _resp(answers=_full_answers(tpl), template_id=tpl.template_id, supplier_id="s1", response_id="r2"), "cdp_climate")
        engine.score_response("r3", tpl, _resp(answers=_full_answers(tpl), template_id=tpl.template_id, supplier_id="s1", response_id="r3"), "cdp_climate")
        scores = engine.get_supplier_scores("s1")
        assert len(scores) == 3
