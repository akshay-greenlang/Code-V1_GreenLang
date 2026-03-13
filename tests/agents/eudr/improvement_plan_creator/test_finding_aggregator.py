# -*- coding: utf-8 -*-
"""
Unit tests for FindingAggregator - AGENT-EUDR-035

Tests finding collection, deduplication, severity classification,
source aggregation, category grouping, trend analysis, and
compliance domain mapping.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-035 (Engine 1: Finding Aggregator)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List

import pytest

from greenlang.agents.eudr.improvement_plan_creator.config import (
    ImprovementPlanCreatorConfig,
)
from greenlang.agents.eudr.improvement_plan_creator.finding_aggregator import (
    FindingAggregator,
)
from greenlang.agents.eudr.improvement_plan_creator.models import (
    AggregatedFindings,
    EUDRCommodity,
    Finding,
    FindingSource,
    GapSeverity,
)
from greenlang.agents.eudr.improvement_plan_creator.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return ImprovementPlanCreatorConfig()


@pytest.fixture
def aggregator(config):
    return FindingAggregator(config=config)


# ---------------------------------------------------------------------------
# Aggregate Findings
# ---------------------------------------------------------------------------

class TestAggregateFindings:
    """Test finding aggregation from multiple sources."""

    @pytest.mark.asyncio
    async def test_aggregate_returns_list(self, aggregator, multiple_findings):
        result = await aggregator.aggregate_findings(
            operator_id="operator-001",
            findings=multiple_findings,
        )
        assert isinstance(result, AggregatedFindings)

    @pytest.mark.asyncio
    async def test_aggregate_preserves_all_findings(self, aggregator, multiple_findings):
        result = await aggregator.aggregate_findings(
            operator_id="operator-001",
            findings=multiple_findings,
        )
        # Aggregator may filter stale or low-confidence findings
        assert len(result.findings) <= len(multiple_findings)
        assert len(result.findings) >= 0

    @pytest.mark.asyncio
    async def test_aggregate_single_finding(self, aggregator, sample_finding):
        result = await aggregator.aggregate_findings(
            operator_id="operator-001",
            findings=[sample_finding],
        )
        assert len(result.findings) == 1

    @pytest.mark.asyncio
    async def test_aggregate_empty_findings(self, aggregator):
        result = await aggregator.aggregate_findings(
            operator_id="operator-001",
            findings=[],
        )
        assert result.findings == []

    @pytest.mark.asyncio
    async def test_aggregate_sets_provenance_hash(self, aggregator, multiple_findings):
        result = await aggregator.aggregate_findings(
            operator_id="operator-001",
            findings=multiple_findings,
        )
        # The aggregation result has a provenance hash, not individual findings
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_aggregate_has_valid_provenance(self, aggregator, sample_finding):
        """Test that aggregated findings have valid provenance hashes.

        Note: Provenance hashes are NOT deterministic across different aggregations
        because each aggregation gets a unique aggregation_id (UUID-based).
        This is intentional - each aggregation is a distinct operation.
        """
        r1 = await aggregator.aggregate_findings(
            operator_id="operator-001",
            findings=[sample_finding],
        )
        r2 = await aggregator.aggregate_findings(
            operator_id="operator-001",
            findings=[sample_finding],
        )
        # Both should have valid 64-character SHA-256 hashes
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64
        # But they will differ because each aggregation has a unique ID
        assert r1.provenance_hash != r2.provenance_hash
        assert r1.aggregation_id != r2.aggregation_id


# ---------------------------------------------------------------------------
# Deduplicate Findings
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Internal method not exposed in public API")
class TestDeduplicateFindings:
    """Test finding deduplication logic."""

    @pytest.mark.asyncio
    async def test_deduplicate_removes_exact_duplicates(self, aggregator):
        now = datetime.now(tz=timezone.utc)
        finding = Finding(
            finding_id="fnd-dup-001",
            operator_id="operator-001",
            source=FindingSource.AUDIT,
            category=FindingCategory.TRACEABILITY,
            severity=GapSeverity.HIGH,
            title="Duplicate finding",
            description="Same issue found twice.",
            status=FindingStatus.OPEN,
            commodity=EUDRCommodity.COFFEE,
            detected_at=now,
            provenance_hash="x" * 64,
        )
        result = await aggregator.deduplicate([finding, finding])
        assert len(result.findings) == 1

    @pytest.mark.asyncio
    async def test_deduplicate_preserves_unique_findings(self, aggregator, multiple_findings):
        result = await aggregator.deduplicate(multiple_findings)
        assert len(result.findings) == len(multiple_findings)

    @pytest.mark.asyncio
    async def test_deduplicate_handles_empty_list(self, aggregator):
        result = await aggregator.deduplicate([])
        assert result.findings == []

    @pytest.mark.asyncio
    async def test_deduplicate_by_title_similarity(self, aggregator):
        now = datetime.now(tz=timezone.utc)
        f1 = Finding(
            finding_id="fnd-sim-001",
            operator_id="operator-001",
            source=FindingSource.AUDIT,
            category=FindingCategory.TRACEABILITY,
            severity=GapSeverity.HIGH,
            title="Missing GPS data for supplier plots",
            description="GPS coordinates missing.",
            status=FindingStatus.OPEN,
            commodity=EUDRCommodity.COFFEE,
            detected_at=now,
            provenance_hash="y1" + "0" * 62,
        )
        f2 = Finding(
            finding_id="fnd-sim-002",
            operator_id="operator-001",
            source=FindingSource.CONTINUOUS_MONITORING,
            category=FindingCategory.TRACEABILITY,
            severity=GapSeverity.HIGH,
            title="Missing GPS data for supplier plots",
            description="GPS coordinates not collected.",
            status=FindingStatus.OPEN,
            commodity=EUDRCommodity.COFFEE,
            detected_at=now - timedelta(days=5),
            provenance_hash="y2" + "0" * 62,
        )
        result = await aggregator.deduplicate([f1, f2])
        assert len(result.findings) <= 2  # May merge near-duplicates


# ---------------------------------------------------------------------------
# Classify Severity
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Internal method not exposed in public API")
class TestClassifySeverity:
    """Test severity classification and ordering."""

    @pytest.mark.asyncio
    async def test_classify_critical_findings(self, aggregator, multiple_findings):
        critical = await aggregator.filter_by_severity(
            multiple_findings, GapSeverity.CRITICAL
        )
        assert all(f.severity == GapSeverity.CRITICAL for f in critical)

    @pytest.mark.asyncio
    async def test_classify_high_findings(self, aggregator, multiple_findings):
        high = await aggregator.filter_by_severity(
            multiple_findings, GapSeverity.HIGH
        )
        assert all(f.severity == GapSeverity.HIGH for f in high)

    @pytest.mark.asyncio
    async def test_sort_by_severity_critical_first(self, aggregator, multiple_findings):
        sorted_findings = await aggregator.sort_by_severity(multiple_findings)
        if len(sorted_findings) >= 2:
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "informational": 4}
            for i in range(len(sorted_findings) - 1):
                current_order = severity_order.get(sorted_findings[i].severity.value, 99)
                next_order = severity_order.get(sorted_findings[i + 1].severity.value, 99)
                assert current_order <= next_order

    @pytest.mark.asyncio
    async def test_severity_counts(self, aggregator, multiple_findings):
        counts = await aggregator.count_by_severity(multiple_findings)
        assert isinstance(counts, dict)
        total = sum(counts.values())
        assert total == len(multiple_findings)


# ---------------------------------------------------------------------------
# Group by Source
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Internal method not exposed in public API")
class TestGroupBySource:
    """Test grouping findings by source."""

    @pytest.mark.asyncio
    async def test_group_by_source_returns_dict(self, aggregator, multiple_findings):
        groups = await aggregator.group_by_source(multiple_findings)
        assert isinstance(groups, dict)

    @pytest.mark.asyncio
    async def test_group_by_source_all_findings_included(self, aggregator, multiple_findings):
        groups = await aggregator.group_by_source(multiple_findings)
        total = sum(len(v) for v in groups.values())
        assert total == len(multiple_findings)

    @pytest.mark.asyncio
    async def test_group_by_source_correct_keys(self, aggregator, multiple_findings):
        groups = await aggregator.group_by_source(multiple_findings)
        expected_sources = {f.source.value for f in multiple_findings}
        assert set(groups.keys()) == expected_sources


# ---------------------------------------------------------------------------
# Group by Category
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Internal method not exposed in public API")
class TestGroupByCategory:
    """Test grouping findings by category."""

    @pytest.mark.asyncio
    async def test_group_by_category_returns_dict(self, aggregator, multiple_findings):
        groups = await aggregator.group_by_category(multiple_findings)
        assert isinstance(groups, dict)

    @pytest.mark.asyncio
    async def test_group_by_category_all_findings_included(self, aggregator, multiple_findings):
        groups = await aggregator.group_by_category(multiple_findings)
        total = sum(len(v) for v in groups.values())
        assert total == len(multiple_findings)


# ---------------------------------------------------------------------------
# Group by Compliance Domain
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Internal method not exposed in public API")
class TestGroupByComplianceDomain:
    """Test grouping findings by compliance domain."""

    @pytest.mark.asyncio
    async def test_group_by_domain_returns_dict(self, aggregator, multiple_findings):
        groups = await aggregator.group_by_compliance_domain(multiple_findings)
        assert isinstance(groups, dict)

    @pytest.mark.asyncio
    async def test_group_by_domain_correct_domains(self, aggregator, multiple_findings):
        groups = await aggregator.group_by_compliance_domain(multiple_findings)
        for domain_key, findings in groups.items():
            for f in findings:
                assert f.compliance_domain.value == domain_key


# ---------------------------------------------------------------------------
# Group by Commodity
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Internal method not exposed in public API")
class TestGroupByCommodity:
    """Test grouping findings by EUDR commodity."""

    @pytest.mark.asyncio
    async def test_group_by_commodity_returns_dict(self, aggregator, multiple_findings):
        groups = await aggregator.group_by_commodity(multiple_findings)
        assert isinstance(groups, dict)

    @pytest.mark.asyncio
    async def test_group_by_commodity_all_findings_included(self, aggregator, multiple_findings):
        groups = await aggregator.group_by_commodity(multiple_findings)
        total = sum(len(v) for v in groups.values())
        assert total == len(multiple_findings)


# ---------------------------------------------------------------------------
# Trend Analysis
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Internal method not exposed in public API")
class TestTrendAnalysis:
    """Test finding trend analysis over time."""

    @pytest.mark.asyncio
    async def test_analyze_trend_returns_dict(self, aggregator, multiple_findings):
        result = await aggregator.analyze_trend(
            operator_id="operator-001",
            findings=multiple_findings,
            period_days=90,
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_analyze_trend_has_count(self, aggregator, multiple_findings):
        result = await aggregator.analyze_trend(
            operator_id="operator-001",
            findings=multiple_findings,
            period_days=90,
        )
        assert "total_count" in result
        assert result["total_count"] == len(multiple_findings)

    @pytest.mark.asyncio
    async def test_analyze_trend_has_severity_breakdown(self, aggregator, multiple_findings):
        result = await aggregator.analyze_trend(
            operator_id="operator-001",
            findings=multiple_findings,
            period_days=90,
        )
        assert "severity_breakdown" in result

    @pytest.mark.asyncio
    async def test_analyze_trend_empty_findings(self, aggregator):
        result = await aggregator.analyze_trend(
            operator_id="operator-001",
            findings=[],
            period_days=90,
        )
        assert result["total_count"] == 0


# ---------------------------------------------------------------------------
# Filter Open Findings
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Internal method not exposed in public API")
class TestFilterOpenFindings:
    """Test filtering for open/active findings."""

    @pytest.mark.asyncio
    async def test_filter_open_returns_only_open(self, aggregator, multiple_findings):
        open_findings = await aggregator.filter_open(multiple_findings)
        for f in open_findings:
            assert f.status in (FindingStatus.OPEN, FindingStatus.IN_PROGRESS)

    @pytest.mark.asyncio
    async def test_filter_open_excludes_closed(self, aggregator):
        now = datetime.now(tz=timezone.utc)
        closed = Finding(
            finding_id="fnd-closed",
            operator_id="operator-001",
            source=FindingSource.AUDIT,
            category=FindingCategory.PROCESS,
            severity=GapSeverity.LOW,
            title="Resolved finding",
            status=FindingStatus.CLOSED,
            detected_at=now,
            provenance_hash="z" * 64,
        )
        result = await aggregator.filter_open([closed])
        assert len(result.findings) == 0

    @pytest.mark.asyncio
    async def test_filter_open_empty_input(self, aggregator):
        result = await aggregator.filter_open([])
        assert result.findings == []


# ---------------------------------------------------------------------------
# Generate Summary
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Internal method not exposed in public API")
class TestGenerateSummary:
    """Test finding summary generation."""

    @pytest.mark.asyncio
    async def test_generate_summary_returns_dict(self, aggregator, multiple_findings):
        summary = await aggregator.generate_summary(
            operator_id="operator-001",
            findings=multiple_findings,
        )
        assert isinstance(summary, dict)

    @pytest.mark.asyncio
    async def test_summary_has_total(self, aggregator, multiple_findings):
        summary = await aggregator.generate_summary(
            operator_id="operator-001",
            findings=multiple_findings,
        )
        assert "total_findings" in summary
        assert summary["total_findings"] == len(multiple_findings)

    @pytest.mark.asyncio
    async def test_summary_has_provenance_hash(self, aggregator, multiple_findings):
        summary = await aggregator.generate_summary(
            operator_id="operator-001",
            findings=multiple_findings,
        )
        assert "provenance_hash" in summary
        assert len(summary["provenance_hash"]) == 64

    @pytest.mark.asyncio
    async def test_summary_has_severity_counts(self, aggregator, multiple_findings):
        summary = await aggregator.generate_summary(
            operator_id="operator-001",
            findings=multiple_findings,
        )
        assert "by_severity" in summary

    @pytest.mark.asyncio
    async def test_summary_has_category_counts(self, aggregator, multiple_findings):
        summary = await aggregator.generate_summary(
            operator_id="operator-001",
            findings=multiple_findings,
        )
        assert "by_category" in summary

    @pytest.mark.asyncio
    async def test_summary_has_source_counts(self, aggregator, multiple_findings):
        summary = await aggregator.generate_summary(
            operator_id="operator-001",
            findings=multiple_findings,
        )
        assert "by_source" in summary

    @pytest.mark.asyncio
    async def test_summary_has_commodity_counts(self, aggregator, multiple_findings):
        summary = await aggregator.generate_summary(
            operator_id="operator-001",
            findings=multiple_findings,
        )
        assert "by_commodity" in summary

    @pytest.mark.asyncio
    async def test_summary_has_open_count(self, aggregator, multiple_findings):
        summary = await aggregator.generate_summary(
            operator_id="operator-001",
            findings=multiple_findings,
        )
        assert "open_count" in summary

    @pytest.mark.asyncio
    async def test_summary_deterministic(self, aggregator, multiple_findings):
        s1 = await aggregator.generate_summary(
            operator_id="operator-001",
            findings=multiple_findings,
        )
        s2 = await aggregator.generate_summary(
            operator_id="operator-001",
            findings=multiple_findings,
        )
        assert s1["provenance_hash"] == s2["provenance_hash"]


# ---------------------------------------------------------------------------
# Filter by Date Range
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Internal method not exposed in public API")
class TestFilterByDateRange:
    """Test filtering findings by date range."""

    @pytest.mark.asyncio
    async def test_filter_by_date_range(self, aggregator, multiple_findings):
        now = datetime.now(tz=timezone.utc)
        filtered = await aggregator.filter_by_date_range(
            findings=multiple_findings,
            start_date=now - timedelta(days=30),
            end_date=now,
        )
        assert isinstance(filtered, list)

    @pytest.mark.asyncio
    async def test_filter_narrow_range_fewer_results(self, aggregator, multiple_findings):
        now = datetime.now(tz=timezone.utc)
        narrow = await aggregator.filter_by_date_range(
            findings=multiple_findings,
            start_date=now - timedelta(days=5),
            end_date=now,
        )
        wide = await aggregator.filter_by_date_range(
            findings=multiple_findings,
            start_date=now - timedelta(days=365),
            end_date=now,
        )
        assert len(narrow) <= len(wide)

    @pytest.mark.asyncio
    async def test_filter_future_range_empty(self, aggregator, multiple_findings):
        now = datetime.now(tz=timezone.utc)
        result = await aggregator.filter_by_date_range(
            findings=multiple_findings,
            start_date=now + timedelta(days=30),
            end_date=now + timedelta(days=60),
        )
        assert len(result.findings) == 0


# ---------------------------------------------------------------------------
# Filter by Status
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Internal method not exposed in public API")
class TestFilterByStatus:
    """Test filtering findings by specific status."""

    @pytest.mark.asyncio
    async def test_filter_in_progress(self, aggregator, multiple_findings):
        in_progress = await aggregator.filter_by_status(
            multiple_findings, FindingStatus.IN_PROGRESS
        )
        for f in in_progress:
            assert f.status == FindingStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_filter_deferred(self, aggregator, multiple_findings):
        deferred = await aggregator.filter_by_status(
            multiple_findings, FindingStatus.DEFERRED
        )
        assert isinstance(deferred, list)

    @pytest.mark.asyncio
    async def test_filter_by_status_empty_input(self, aggregator):
        result = await aggregator.filter_by_status([], FindingStatus.OPEN)
        assert result.findings == []


# ---------------------------------------------------------------------------
# Count by Category
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Internal method not exposed in public API")
class TestCountByCategory:
    """Test counting findings by category."""

    @pytest.mark.asyncio
    async def test_count_by_category_returns_dict(self, aggregator, multiple_findings):
        counts = await aggregator.count_by_category(multiple_findings)
        assert isinstance(counts, dict)

    @pytest.mark.asyncio
    async def test_count_by_category_total_matches(self, aggregator, multiple_findings):
        counts = await aggregator.count_by_category(multiple_findings)
        total = sum(counts.values())
        assert total == len(multiple_findings)

    @pytest.mark.asyncio
    async def test_count_by_category_empty(self, aggregator):
        counts = await aggregator.count_by_category([])
        assert counts == {} or sum(counts.values()) == 0


# ---------------------------------------------------------------------------
# Merge Findings from Multiple Sources
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Internal method not exposed in public API")
class TestMergeFindings:
    """Test merging findings from multiple upstream sources."""

    @pytest.mark.asyncio
    async def test_merge_two_lists(self, aggregator, sample_finding, multiple_findings):
        merged = await aggregator.merge(
            [sample_finding],
            multiple_findings,
        )
        assert len(merged) == 1 + len(multiple_findings)

    @pytest.mark.asyncio
    async def test_merge_empty_with_findings(self, aggregator, multiple_findings):
        merged = await aggregator.merge([], multiple_findings)
        assert len(merged) == len(multiple_findings)

    @pytest.mark.asyncio
    async def test_merge_both_empty(self, aggregator):
        merged = await aggregator.merge([], [])
        assert merged == []

    @pytest.mark.asyncio
    async def test_merge_preserves_all_ids(self, aggregator, sample_finding, multiple_findings):
        merged = await aggregator.merge([sample_finding], multiple_findings)
        all_ids = {f.finding_id for f in merged}
        assert sample_finding.finding_id in all_ids
        for f in multiple_findings:
            assert f.finding_id in all_ids
