# -*- coding: utf-8 -*-
"""
Unit Tests for AnalyticsEngine - AGENT-DATA-008
=================================================

Tests all methods of AnalyticsEngine with 85%+ coverage.
Validates campaign analytics, response rate, supplier benchmarking,
compliance gaps, score distribution, trend analysis, data quality
distribution, top/bottom performers, report generation (4 formats),
geographic summary, framework coverage, and SHA-256 provenance.

Test count target: ~70 tests
Author: GreenLang Platform Team / GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from greenlang.supplier_questionnaire.analytics import AnalyticsEngine
from greenlang.supplier_questionnaire.models import (
    CampaignAnalytics,
    Distribution,
    DistributionStatus,
    Framework,
    PerformanceTier,
    QuestionnaireResponse,
    QuestionnaireScore,
    ValidationSummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _dist(
    dist_id: str = "dist-001",
    supplier_id: str = "sup-001",
    campaign_id: str = "camp-001",
    status: DistributionStatus = DistributionStatus.SENT,
    email: str = "user@example.com",
) -> Distribution:
    return Distribution(
        distribution_id=dist_id,
        template_id="tpl-001",
        supplier_id=supplier_id,
        supplier_name=f"Supplier {supplier_id}",
        supplier_email=email,
        campaign_id=campaign_id,
        status=status,
    )


def _score(
    supplier_id: str = "sup-001",
    framework: Framework = Framework.CDP_CLIMATE,
    normalized_score: float = 75.0,
    section_scores: Optional[Dict[str, float]] = None,
    scored_at: Optional[datetime] = None,
) -> QuestionnaireScore:
    return QuestionnaireScore(
        response_id=f"resp-{supplier_id}",
        template_id="tpl-001",
        supplier_id=supplier_id,
        framework=framework,
        raw_score=normalized_score,
        normalized_score=normalized_score,
        performance_tier=PerformanceTier.ADVANCED if normalized_score >= 60 else PerformanceTier.BEGINNER,
        section_scores=section_scores or {"Sec1": normalized_score},
        scored_at=scored_at or _utcnow(),
        provenance_hash="a" * 64,
    )


def _resp(
    supplier_id: str = "sup-001",
    response_id: str = "resp-001",
    completion_pct: float = 80.0,
) -> QuestionnaireResponse:
    return QuestionnaireResponse(
        response_id=response_id,
        distribution_id="dist-001",
        template_id="tpl-001",
        supplier_id=supplier_id,
        answers=[],
        completion_pct=completion_pct,
    )


def _validation(
    response_id: str = "resp-001",
    data_quality_score: float = 75.0,
) -> ValidationSummary:
    return ValidationSummary(
        response_id=response_id,
        template_id="tpl-001",
        checks=[],
        total_checks=10,
        passed_checks=8,
        failed_checks=2,
        warning_count=1,
        error_count=1,
        is_valid=True,
        data_quality_score=data_quality_score,
        provenance_hash="b" * 64,
    )


def _sample_dists(count: int = 5, campaign_id: str = "camp-001") -> List[Distribution]:
    """Create a set of distributions with varying statuses."""
    statuses = [
        DistributionStatus.SUBMITTED,
        DistributionStatus.SUBMITTED,
        DistributionStatus.SENT,
        DistributionStatus.OPENED,
        DistributionStatus.PENDING,
    ]
    return [
        _dist(
            dist_id=f"dist-{i:03d}",
            supplier_id=f"sup-{i:03d}",
            campaign_id=campaign_id,
            status=statuses[i % len(statuses)],
            email=f"user{i}@example.com",
        )
        for i in range(count)
    ]


def _sample_scores(count: int = 5, framework: Framework = Framework.CDP_CLIMATE) -> List[QuestionnaireScore]:
    """Create a set of scores with varying values."""
    values = [90.0, 75.0, 60.0, 45.0, 30.0]
    return [
        _score(
            supplier_id=f"sup-{i:03d}",
            framework=framework,
            normalized_score=values[i % len(values)],
            section_scores={"Sec1": values[i % len(values)], "Sec2": values[i % len(values)] - 10},
        )
        for i in range(count)
    ]


# ============================================================================
# TEST CLASS: Initialization
# ============================================================================


class TestAnalyticsEngineInit:

    def test_init_defaults(self):
        engine = AnalyticsEngine()
        assert engine._histogram_bins == 10
        assert engine._top_n_default == 10

    def test_init_custom_config(self):
        engine = AnalyticsEngine({"histogram_bins": 5, "top_n_default": 3})
        assert engine._histogram_bins == 5
        assert engine._top_n_default == 3

    def test_init_stats_zeroed(self):
        engine = AnalyticsEngine()
        stats = engine.get_statistics()
        assert stats["campaign_analytics"] == 0
        assert stats["benchmarks_generated"] == 0
        assert stats["reports_generated"] == 0

    def test_init_empty_cache(self):
        engine = AnalyticsEngine()
        assert engine.get_statistics()["cached_analytics"] == 0


# ============================================================================
# TEST CLASS: get_campaign_analytics
# ============================================================================


class TestGetCampaignAnalytics:

    def test_basic_analytics(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(5)
        analytics = engine.get_campaign_analytics("camp-001", dists)
        assert isinstance(analytics, CampaignAnalytics)
        assert analytics.campaign_id == "camp-001"
        assert analytics.total_distributions == 5

    def test_response_rate(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(5)
        analytics = engine.get_campaign_analytics("camp-001", dists)
        assert analytics.response_rate == 40.0

    def test_status_breakdown(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(5)
        analytics = engine.get_campaign_analytics("camp-001", dists)
        assert "submitted" in analytics.status_breakdown
        assert analytics.status_breakdown["submitted"] == 2

    def test_with_scores(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(5)
        scores = _sample_scores(5)
        analytics = engine.get_campaign_analytics("camp-001", dists, scores=scores)
        assert analytics.avg_score > 0.0

    def test_with_responses(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        responses = [_resp(f"sup-{i:03d}", f"resp-{i:03d}", 70.0 + i * 5) for i in range(3)]
        analytics = engine.get_campaign_analytics("camp-001", dists, responses=responses)
        assert analytics.avg_completion_pct > 0.0

    def test_with_validations(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        responses = [_resp(f"sup-{i:03d}", f"resp-{i:03d}") for i in range(3)]
        validations = [_validation(f"resp-{i:03d}", 60.0 + i * 10) for i in range(3)]
        analytics = engine.get_campaign_analytics(
            "camp-001", dists, responses=responses, validations=validations,
        )
        assert analytics.avg_data_quality > 0.0

    def test_section_avg_scores(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        scores = _sample_scores(3)
        analytics = engine.get_campaign_analytics("camp-001", dists, scores=scores)
        assert "Sec1" in analytics.section_avg_scores

    def test_provenance_hash(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        analytics = engine.get_campaign_analytics("camp-001", dists)
        assert len(analytics.provenance_hash) == 64

    def test_updates_stats(self):
        engine = AnalyticsEngine()
        engine.get_campaign_analytics("camp-001", _sample_dists(3))
        assert engine.get_statistics()["campaign_analytics"] == 1

    def test_caches_result(self):
        engine = AnalyticsEngine()
        engine.get_campaign_analytics("camp-001", _sample_dists(3))
        assert engine.get_statistics()["cached_analytics"] == 1

    def test_empty_distributions(self):
        engine = AnalyticsEngine()
        analytics = engine.get_campaign_analytics("camp-001", [])
        assert analytics.total_distributions == 0
        assert analytics.response_rate == 0.0

    def test_no_matching_campaign(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3, campaign_id="camp-other")
        analytics = engine.get_campaign_analytics("camp-001", dists)
        assert analytics.total_distributions == 0


# ============================================================================
# TEST CLASS: get_response_rate
# ============================================================================


class TestGetResponseRate:

    def test_basic_rate(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(5)
        rate = engine.get_response_rate("camp-001", dists)
        assert rate == 40.0

    def test_all_submitted(self):
        engine = AnalyticsEngine()
        dists = [_dist(f"d{i}", f"s{i}", "camp-001", DistributionStatus.SUBMITTED) for i in range(3)]
        rate = engine.get_response_rate("camp-001", dists)
        assert rate == 100.0

    def test_none_submitted(self):
        engine = AnalyticsEngine()
        dists = [_dist(f"d{i}", f"s{i}", "camp-001", DistributionStatus.SENT) for i in range(3)]
        rate = engine.get_response_rate("camp-001", dists)
        assert rate == 0.0

    def test_empty_distributions(self):
        engine = AnalyticsEngine()
        rate = engine.get_response_rate("camp-001", [])
        assert rate == 0.0

    def test_wrong_campaign(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3, campaign_id="camp-other")
        rate = engine.get_response_rate("camp-001", dists)
        assert rate == 0.0


# ============================================================================
# TEST CLASS: get_supplier_benchmark
# ============================================================================


class TestGetSupplierBenchmark:

    def test_benchmark_basic(self):
        engine = AnalyticsEngine()
        scores = _sample_scores(5)
        result = engine.get_supplier_benchmark("sup-000", "cdp_climate", "energy", scores)
        assert result["supplier_score"] == 90.0
        assert result["percentile"] is not None
        assert result["population_size"] == 5

    def test_benchmark_no_scores(self):
        engine = AnalyticsEngine()
        result = engine.get_supplier_benchmark("sup-X", "cdp_climate", "energy", [])
        assert "No scores available" in result["message"]

    def test_benchmark_supplier_not_found(self):
        engine = AnalyticsEngine()
        scores = _sample_scores(3)
        result = engine.get_supplier_benchmark("sup-unknown", "cdp_climate", "energy", scores)
        assert "No scores found for this supplier" in result["message"]

    def test_benchmark_statistics(self):
        engine = AnalyticsEngine()
        scores = _sample_scores(5)
        result = engine.get_supplier_benchmark("sup-000", "cdp_climate", "energy", scores)
        assert "population_avg" in result
        assert "population_median" in result
        assert "population_min" in result
        assert "population_max" in result

    def test_benchmark_updates_stats(self):
        engine = AnalyticsEngine()
        scores = _sample_scores(3)
        engine.get_supplier_benchmark("sup-000", "cdp_climate", "energy", scores)
        assert engine.get_statistics()["benchmarks_generated"] == 1

    def test_benchmark_provenance(self):
        engine = AnalyticsEngine()
        scores = _sample_scores(3)
        result = engine.get_supplier_benchmark("sup-000", "cdp_climate", "energy", scores)
        assert len(result["provenance_hash"]) == 64


# ============================================================================
# TEST CLASS: get_compliance_gaps
# ============================================================================


class TestGetComplianceGaps:

    def test_gaps_basic(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        scores = _sample_scores(3)
        result = engine.get_compliance_gaps("camp-001", "cdp_climate", scores, dists)
        assert "gaps" in result
        assert len(result["gaps"]) >= 1

    def test_gaps_sorted_ascending(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        scores = _sample_scores(3)
        result = engine.get_compliance_gaps("camp-001", "cdp_climate", scores, dists)
        gaps = result["gaps"]
        if len(gaps) >= 2:
            assert gaps[0]["avg_score"] <= gaps[-1]["avg_score"]

    def test_gaps_weakest_strongest(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        scores = _sample_scores(3)
        result = engine.get_compliance_gaps("camp-001", "cdp_climate", scores, dists)
        assert result["weakest_section"] is not None
        assert result["strongest_section"] is not None
        assert result["weakest_section"]["avg_score"] <= result["strongest_section"]["avg_score"]

    def test_gaps_no_scores(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        result = engine.get_compliance_gaps("camp-001", "cdp_climate", [], dists)
        assert "No scores available" in result["message"]

    def test_gaps_updates_stats(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        scores = _sample_scores(3)
        engine.get_compliance_gaps("camp-001", "cdp_climate", scores, dists)
        assert engine.get_statistics()["gap_analyses"] == 1


# ============================================================================
# TEST CLASS: get_score_distribution
# ============================================================================


class TestGetScoreDistribution:

    def test_distribution_basic(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(5)
        scores = _sample_scores(5)
        result = engine.get_score_distribution("camp-001", scores, dists)
        assert result["total_scores"] == 5
        assert "histogram" in result

    def test_distribution_statistics(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(5)
        scores = _sample_scores(5)
        result = engine.get_score_distribution("camp-001", scores, dists)
        assert result["mean"] > 0.0
        assert result["min"] <= result["max"]

    def test_distribution_empty(self):
        engine = AnalyticsEngine()
        result = engine.get_score_distribution("camp-001", [], [])
        assert result["total_scores"] == 0

    def test_distribution_provenance(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        scores = _sample_scores(3)
        result = engine.get_score_distribution("camp-001", scores, dists)
        assert len(result["provenance_hash"]) == 64


# ============================================================================
# TEST CLASS: get_trend_analysis
# ============================================================================


class TestGetTrendAnalysis:

    def test_trend_basic(self):
        engine = AnalyticsEngine()
        base_time = _utcnow()
        scores = [
            _score("sup-001", Framework.CDP_CLIMATE, 50.0, scored_at=base_time - timedelta(days=60)),
            _score("sup-001", Framework.CDP_CLIMATE, 70.0, scored_at=base_time),
        ]
        result = engine.get_trend_analysis("sup-001", "cdp_climate", scores)
        assert result["total_periods"] == 2

    def test_trend_improving(self):
        engine = AnalyticsEngine()
        base_time = _utcnow()
        scores = [
            _score("sup-001", Framework.CDP_CLIMATE, 40.0, scored_at=base_time - timedelta(days=60)),
            _score("sup-001", Framework.CDP_CLIMATE, 80.0, scored_at=base_time),
        ]
        result = engine.get_trend_analysis("sup-001", "cdp_climate", scores)
        assert result["trend"] == "improving"

    def test_trend_declining(self):
        engine = AnalyticsEngine()
        base_time = _utcnow()
        scores = [
            _score("sup-001", Framework.CDP_CLIMATE, 80.0, scored_at=base_time - timedelta(days=60)),
            _score("sup-001", Framework.CDP_CLIMATE, 40.0, scored_at=base_time),
        ]
        result = engine.get_trend_analysis("sup-001", "cdp_climate", scores)
        assert result["trend"] == "declining"

    def test_trend_stable(self):
        engine = AnalyticsEngine()
        base_time = _utcnow()
        scores = [
            _score("sup-001", Framework.CDP_CLIMATE, 50.0, scored_at=base_time - timedelta(days=60)),
            _score("sup-001", Framework.CDP_CLIMATE, 51.0, scored_at=base_time),
        ]
        result = engine.get_trend_analysis("sup-001", "cdp_climate", scores)
        assert result["trend"] == "stable"

    def test_trend_single_point(self):
        engine = AnalyticsEngine()
        scores = [_score("sup-001", Framework.CDP_CLIMATE, 50.0)]
        result = engine.get_trend_analysis("sup-001", "cdp_climate", scores)
        assert result["total_periods"] == 1
        assert result["trend"] == "stable"

    def test_trend_updates_stats(self):
        engine = AnalyticsEngine()
        engine.get_trend_analysis("sup-001", "cdp_climate", [])
        assert engine.get_statistics()["trend_analyses"] == 1

    def test_trend_data_points_have_change(self):
        engine = AnalyticsEngine()
        base_time = _utcnow()
        scores = [
            _score("sup-001", Framework.CDP_CLIMATE, 50.0, scored_at=base_time - timedelta(days=60)),
            _score("sup-001", Framework.CDP_CLIMATE, 70.0, scored_at=base_time),
        ]
        result = engine.get_trend_analysis("sup-001", "cdp_climate", scores)
        assert result["data_points"][0]["change"] == 0.0
        assert result["data_points"][1]["change"] == 20.0


# ============================================================================
# TEST CLASS: get_data_quality_distribution
# ============================================================================


class TestGetDataQualityDistribution:

    def test_quality_basic(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        validations = [_validation(f"resp-{i}", 50.0 + i * 15) for i in range(3)]
        result = engine.get_data_quality_distribution("camp-001", validations, dists)
        assert result["total_validated"] >= 1
        assert "histogram" in result

    def test_quality_above_below_threshold(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        validations = [
            _validation("r1", 80.0),
            _validation("r2", 50.0),
            _validation("r3", 70.0),
        ]
        result = engine.get_data_quality_distribution("camp-001", validations, dists)
        assert result["above_threshold"] >= 0
        assert result["below_threshold"] >= 0

    def test_quality_statistics(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        validations = [_validation(f"r{i}", 60.0 + i * 10) for i in range(3)]
        result = engine.get_data_quality_distribution("camp-001", validations, dists)
        assert result["mean"] > 0.0
        assert result["min"] <= result["max"]


# ============================================================================
# TEST CLASS: identify_top_performers and identify_bottom_performers
# ============================================================================


class TestIdentifyPerformers:

    def test_top_performers(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(5)
        scores = _sample_scores(5)
        result = engine.identify_top_performers("camp-001", scores, dists, n=3)
        assert len(result) <= 3
        if len(result) >= 2:
            assert result[0]["normalized_score"] >= result[1]["normalized_score"]

    def test_bottom_performers(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(5)
        scores = _sample_scores(5)
        result = engine.identify_bottom_performers("camp-001", scores, dists, n=3)
        assert len(result) <= 3
        if len(result) >= 2:
            assert result[0]["normalized_score"] <= result[1]["normalized_score"]

    def test_performers_ranked(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        scores = _sample_scores(3)
        result = engine.identify_top_performers("camp-001", scores, dists)
        for i, r in enumerate(result, start=1):
            assert r["rank"] == i

    def test_performers_empty(self):
        engine = AnalyticsEngine()
        result = engine.identify_top_performers("camp-001", [], [])
        assert len(result) == 0

    def test_top_performers_n_limit(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(10, "camp-001")
        scores = _sample_scores(10)
        result = engine.identify_top_performers("camp-001", scores, dists, n=3)
        assert len(result) <= 3


# ============================================================================
# TEST CLASS: generate_report - TEXT
# ============================================================================


class TestGenerateReportText:

    def test_text_report_basic(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(5)
        report = engine.generate_report("camp-001", "text", dists)
        assert "CAMPAIGN REPORT" in report
        assert "camp-001" in report

    def test_text_report_summary(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(5)
        report = engine.generate_report("camp-001", "text", dists)
        assert "Total Distributions" in report
        assert "Response Rate" in report

    def test_text_report_provenance(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        report = engine.generate_report("camp-001", "text", dists)
        assert "Provenance" in report

    def test_text_report_status_breakdown(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(5)
        report = engine.generate_report("camp-001", "text", dists)
        assert "Status Breakdown" in report


# ============================================================================
# TEST CLASS: generate_report - JSON
# ============================================================================


class TestGenerateReportJSON:

    def test_json_report_valid(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        report = engine.generate_report("camp-001", "json", dists)
        data = json.loads(report)
        assert data["campaign_id"] == "camp-001"

    def test_json_report_contains_fields(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        report = engine.generate_report("camp-001", "json", dists)
        data = json.loads(report)
        assert "total_distributions" in data
        assert "response_rate" in data
        assert "provenance_hash" in data


# ============================================================================
# TEST CLASS: generate_report - MARKDOWN
# ============================================================================


class TestGenerateReportMarkdown:

    def test_markdown_report_headers(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        report = engine.generate_report("camp-001", "markdown", dists)
        assert "# Supplier Questionnaire Campaign Report" in report
        assert "## Summary" in report

    def test_markdown_report_table(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        report = engine.generate_report("camp-001", "markdown", dists)
        assert "| Metric | Value |" in report

    def test_markdown_provenance(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        report = engine.generate_report("camp-001", "markdown", dists)
        assert "Provenance" in report


# ============================================================================
# TEST CLASS: generate_report - HTML
# ============================================================================


class TestGenerateReportHTML:

    def test_html_report_structure(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        report = engine.generate_report("camp-001", "html", dists)
        assert "<!DOCTYPE html>" in report
        assert "<h1>" in report
        assert "</html>" in report

    def test_html_report_campaign_id(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        report = engine.generate_report("camp-001", "html", dists)
        assert "camp-001" in report

    def test_html_provenance(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        report = engine.generate_report("camp-001", "html", dists)
        assert "Provenance" in report

    def test_html_has_css(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        report = engine.generate_report("camp-001", "html", dists)
        assert "<style>" in report


# ============================================================================
# TEST CLASS: generate_report - general
# ============================================================================


class TestGenerateReportGeneral:

    def test_unknown_format_defaults_text(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        report = engine.generate_report("camp-001", "unknown_fmt", dists)
        assert "CAMPAIGN REPORT" in report

    def test_updates_stats(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        engine.generate_report("camp-001", "text", dists)
        assert engine.get_statistics()["reports_generated"] == 1


# ============================================================================
# TEST CLASS: get_geographic_summary
# ============================================================================


class TestGetGeographicSummary:

    def test_geographic_basic(self):
        engine = AnalyticsEngine()
        dists = [
            _dist("d1", "s1", "camp-001", email="user@example.de"),
            _dist("d2", "s2", "camp-001", email="user@example.fr"),
            _dist("d3", "s3", "camp-001", email="user@example.com"),
        ]
        result = engine.get_geographic_summary("camp-001", dists)
        assert result["total_countries"] >= 1
        assert "countries" in result

    def test_geographic_country_mapping(self):
        engine = AnalyticsEngine()
        dists = [_dist("d1", "s1", "camp-001", email="user@company.de")]
        result = engine.get_geographic_summary("camp-001", dists)
        assert "germany" in result["countries"]

    def test_geographic_global_com(self):
        engine = AnalyticsEngine()
        dists = [_dist("d1", "s1", "camp-001", email="user@company.com")]
        result = engine.get_geographic_summary("camp-001", dists)
        assert "global" in result["countries"]

    def test_geographic_response_rate(self):
        engine = AnalyticsEngine()
        dists = [
            _dist("d1", "s1", "camp-001", DistributionStatus.SUBMITTED, email="user@co.de"),
            _dist("d2", "s2", "camp-001", DistributionStatus.SENT, email="user@co.de"),
        ]
        result = engine.get_geographic_summary("camp-001", dists)
        assert result["countries"]["germany"]["response_rate"] == 50.0

    def test_geographic_with_scores(self):
        engine = AnalyticsEngine()
        dists = [_dist("d1", "s1", "camp-001", email="user@co.de")]
        scores = [_score("s1", normalized_score=80.0)]
        result = engine.get_geographic_summary("camp-001", dists, scores=scores)
        assert result["countries"]["germany"]["avg_score"] == 80.0

    def test_geographic_empty_email(self):
        engine = AnalyticsEngine()
        dists = [_dist("d1", "s1", "camp-001", email="")]
        result = engine.get_geographic_summary("camp-001", dists)
        assert "unknown" in result["countries"]

    def test_geographic_empty_campaign(self):
        engine = AnalyticsEngine()
        result = engine.get_geographic_summary("camp-001", [])
        assert result["total_countries"] == 0


# ============================================================================
# TEST CLASS: get_framework_coverage
# ============================================================================


class TestGetFrameworkCoverage:

    def test_coverage_basic(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        scores = _sample_scores(3)
        result = engine.get_framework_coverage("camp-001", scores, dists)
        assert result["total_frameworks"] >= 1
        assert "frameworks" in result

    def test_coverage_section_details(self):
        engine = AnalyticsEngine()
        dists = _sample_dists(3)
        scores = _sample_scores(3)
        result = engine.get_framework_coverage("camp-001", scores, dists)
        cdp_coverage = result["frameworks"].get("cdp_climate")
        if cdp_coverage:
            assert "sections" in cdp_coverage
            assert cdp_coverage["total_scores"] >= 1

    def test_coverage_empty(self):
        engine = AnalyticsEngine()
        result = engine.get_framework_coverage("camp-001", [], [])
        assert result["total_frameworks"] == 0

    def test_coverage_multiple_frameworks(self):
        engine = AnalyticsEngine()
        dists = [
            _dist("d1", "s1", "camp-001"),
            _dist("d2", "s2", "camp-001"),
        ]
        scores = [
            _score("s1", Framework.CDP_CLIMATE, 60.0, {"A": 60.0}),
            _score("s2", Framework.ECOVADIS, 70.0, {"B": 70.0}),
        ]
        result = engine.get_framework_coverage("camp-001", scores, dists)
        assert result["total_frameworks"] == 2

    def test_coverage_section_stats(self):
        engine = AnalyticsEngine()
        dists = [
            _dist("d1", "s1", "camp-001"),
            _dist("d2", "s2", "camp-001"),
        ]
        scores = [
            _score("s1", Framework.CDP_CLIMATE, 40.0, {"Emissions": 40.0}),
            _score("s2", Framework.CDP_CLIMATE, 80.0, {"Emissions": 80.0}),
        ]
        result = engine.get_framework_coverage("camp-001", scores, dists)
        sec = result["frameworks"]["cdp_climate"]["sections"]["Emissions"]
        assert sec["avg_score"] == 60.0
        assert sec["min_score"] == 40.0
        assert sec["max_score"] == 80.0


# ============================================================================
# TEST CLASS: get_statistics
# ============================================================================


class TestGetStatistics:

    def test_statistics_keys(self):
        engine = AnalyticsEngine()
        stats = engine.get_statistics()
        expected = {
            "campaign_analytics", "benchmarks_generated", "gap_analyses",
            "reports_generated", "trend_analyses", "errors",
            "cached_analytics", "timestamp",
        }
        assert expected.issubset(set(stats.keys()))

    def test_statistics_timestamp(self):
        engine = AnalyticsEngine()
        stats = engine.get_statistics()
        assert "timestamp" in stats


# ============================================================================
# TEST CLASS: Provenance
# ============================================================================


class TestProvenance:

    def test_sha256_format(self):
        engine = AnalyticsEngine()
        h = engine._compute_provenance("test", "data")
        assert len(h) == 64
        assert re.match(r"^[0-9a-f]{64}$", h)

    def test_deterministic_within_same_second(self):
        engine = AnalyticsEngine()
        h1 = engine._compute_provenance("a", "b")
        h2 = engine._compute_provenance("a", "b")
        assert h1 == h2


# ============================================================================
# TEST CLASS: Edge cases
# ============================================================================


class TestEdgeCases:

    def test_single_distribution_campaign(self):
        engine = AnalyticsEngine()
        dists = [_dist("d1", "s1", "camp-001", DistributionStatus.SUBMITTED)]
        analytics = engine.get_campaign_analytics("camp-001", dists)
        assert analytics.total_distributions == 1
        assert analytics.response_rate == 100.0

    def test_all_bounced(self):
        engine = AnalyticsEngine()
        dists = [_dist(f"d{i}", f"s{i}", "camp-001", DistributionStatus.BOUNCED) for i in range(3)]
        analytics = engine.get_campaign_analytics("camp-001", dists)
        assert analytics.response_rate == 0.0
        assert analytics.status_breakdown.get("bounced") == 3
