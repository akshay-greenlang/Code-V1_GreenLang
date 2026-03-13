# -*- coding: utf-8 -*-
"""
Unit tests for RootCauseAnalyzer - AGENT-EUDR-032

Tests four analysis methods (five_whys, fishbone, fault_tree, correlation),
causal chain building, confidence scoring, record retrieval, listing, and
health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.grievance_mechanism_manager.config import (
    GrievanceMechanismManagerConfig,
)
from greenlang.agents.eudr.grievance_mechanism_manager.root_cause_analyzer import (
    RootCauseAnalyzer,
)
from greenlang.agents.eudr.grievance_mechanism_manager.models import (
    AnalysisMethod,
    CausalChainStep,
    RootCauseRecord,
)


@pytest.fixture
def config():
    return GrievanceMechanismManagerConfig()


@pytest.fixture
def analyzer(config):
    return RootCauseAnalyzer(config=config)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_analyzer_created(self, analyzer):
        assert analyzer is not None

    def test_default_config(self):
        a = RootCauseAnalyzer()
        assert a.config is not None

    def test_records_empty(self, analyzer):
        assert len(analyzer._records) == 0


# ---------------------------------------------------------------------------
# Five-Whys Analysis
# ---------------------------------------------------------------------------


class TestFiveWhysAnalysis:
    def test_pollution_keywords(self, analyzer):
        result = analyzer._five_whys_analysis(
            "Water pollution from factory site", "environmental",
        )
        assert result["primary_cause"] is not None
        assert "environmental" in result["primary_cause"].lower()
        assert result["depth"] == 5
        assert result["confidence"] == 72

    def test_contamination_keywords(self, analyzer):
        result = analyzer._five_whys_analysis(
            "Soil contamination near plantation", "environmental",
        )
        assert result["confidence"] == 72
        assert len(result["causal_chain"]) == 5

    def test_rights_keywords(self, analyzer):
        result = analyzer._five_whys_analysis(
            "Indigenous rights violation", "human_rights",
        )
        assert "FPIC" in result["primary_cause"] or "due diligence" in result["primary_cause"].lower()
        assert result["depth"] == 4
        assert result["confidence"] == 68

    def test_labor_keywords(self, analyzer):
        result = analyzer._five_whys_analysis(
            "Labor wages dispute", "labor",
        )
        assert result["depth"] == 4
        assert result["confidence"] == 65

    def test_generic_description(self, analyzer):
        result = analyzer._five_whys_analysis(
            "General process complaint", "process",
        )
        assert result["depth"] == 3
        assert result["confidence"] == 55

    def test_causal_chain_has_step_types(self, analyzer):
        result = analyzer._five_whys_analysis(
            "Water pollution event", "environmental",
        )
        types = {s.step_type for s in result["causal_chain"]}
        assert "proximate" in types
        assert "root" in types

    def test_recommendations_generated(self, analyzer):
        result = analyzer._five_whys_analysis("Pollution observed", "environmental")
        assert len(result["recommendations"]) >= 2

    def test_contributing_factors_present(self, analyzer):
        result = analyzer._five_whys_analysis("Pollution observed", "environmental")
        assert len(result["contributing_factors"]) > 0

    def test_contributing_factors_have_weights(self, analyzer):
        result = analyzer._five_whys_analysis("Pollution observed", "environmental")
        for f in result["contributing_factors"]:
            assert "weight" in f
            assert f["weight"] > 0


# ---------------------------------------------------------------------------
# Fishbone Analysis
# ---------------------------------------------------------------------------


class TestFishboneAnalysis:
    def test_returns_six_categories(self, analyzer):
        result = analyzer._fishbone_analysis("Test", "process")
        assert len(result["contributing_factors"]) == 6

    def test_categories_present(self, analyzer):
        result = analyzer._fishbone_analysis("Test", "process")
        cats = {f["category"] for f in result["contributing_factors"]}
        assert "people" in cats
        assert "process" in cats
        assert "policy" in cats

    def test_causal_chain_matches_categories(self, analyzer):
        result = analyzer._fishbone_analysis("Test", "process")
        assert len(result["causal_chain"]) == 6

    def test_confidence_score(self, analyzer):
        result = analyzer._fishbone_analysis("Test", "process")
        assert result["confidence"] == 60

    def test_depth_equals_category_count(self, analyzer):
        result = analyzer._fishbone_analysis("Test", "process")
        assert result["depth"] == 6

    def test_primary_cause_set(self, analyzer):
        result = analyzer._fishbone_analysis("Test", "process")
        assert "systematic" in result["primary_cause"].lower() or "systemic" in result["primary_cause"].lower()


# ---------------------------------------------------------------------------
# Fault Tree Analysis
# ---------------------------------------------------------------------------


class TestFaultTreeAnalysis:
    def test_fault_tree_chain_length(self, analyzer):
        result = analyzer._fault_tree_analysis("Test", "process")
        assert len(result["causal_chain"]) == 3

    def test_fault_tree_confidence(self, analyzer):
        result = analyzer._fault_tree_analysis("Test", "process")
        assert result["confidence"] == 62

    def test_fault_tree_factors(self, analyzer):
        result = analyzer._fault_tree_analysis("Test", "process")
        assert len(result["contributing_factors"]) == 2
        weights = [f["weight"] for f in result["contributing_factors"]]
        assert sum(weights) == 1.0

    def test_fault_tree_primary_cause(self, analyzer):
        result = analyzer._fault_tree_analysis("Test", "process")
        assert "control" in result["primary_cause"].lower()


# ---------------------------------------------------------------------------
# Correlation Analysis
# ---------------------------------------------------------------------------


class TestCorrelationAnalysis:
    def test_correlation_chain_length(self, analyzer):
        result = analyzer._correlation_analysis("Test", "process")
        assert len(result["causal_chain"]) == 2

    def test_correlation_confidence(self, analyzer):
        result = analyzer._correlation_analysis("Test", "process")
        assert result["confidence"] == 50

    def test_correlation_factors(self, analyzer):
        result = analyzer._correlation_analysis("Test", "process")
        assert len(result["contributing_factors"]) == 2

    def test_correlation_primary_cause(self, analyzer):
        result = analyzer._correlation_analysis("Test", "process")
        assert "correlation" in result["primary_cause"].lower() or "correlated" in result["primary_cause"].lower()


# ---------------------------------------------------------------------------
# Analyze (async integration)
# ---------------------------------------------------------------------------


class TestAnalyze:
    @pytest.mark.asyncio
    async def test_analyze_returns_record(self, analyzer, sample_root_cause_record):
        record = await analyzer.analyze(
            "g-001", "OP-001", sample_root_cause_record,
        )
        assert isinstance(record, RootCauseRecord)

    @pytest.mark.asyncio
    async def test_default_method_five_whys(self, analyzer, sample_root_cause_record):
        record = await analyzer.analyze(
            "g-001", "OP-001", sample_root_cause_record,
        )
        assert record.analysis_method == AnalysisMethod.FIVE_WHYS

    @pytest.mark.asyncio
    async def test_override_method_fishbone(self, analyzer, sample_root_cause_record):
        record = await analyzer.analyze(
            "g-001", "OP-001", sample_root_cause_record,
            method="fishbone",
        )
        assert record.analysis_method == AnalysisMethod.FISHBONE

    @pytest.mark.asyncio
    async def test_override_method_fault_tree(self, analyzer, sample_root_cause_record):
        record = await analyzer.analyze(
            "g-001", "OP-001", sample_root_cause_record,
            method="fault_tree",
        )
        assert record.analysis_method == AnalysisMethod.FAULT_TREE

    @pytest.mark.asyncio
    async def test_override_method_correlation(self, analyzer, sample_root_cause_record):
        record = await analyzer.analyze(
            "g-001", "OP-001", sample_root_cause_record,
            method="correlation",
        )
        assert record.analysis_method == AnalysisMethod.CORRELATION

    @pytest.mark.asyncio
    async def test_invalid_method_falls_back(self, analyzer, sample_root_cause_record):
        record = await analyzer.analyze(
            "g-001", "OP-001", sample_root_cause_record,
            method="invalid_method",
        )
        assert record.analysis_method == AnalysisMethod.FIVE_WHYS

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, analyzer, sample_root_cause_record):
        record = await analyzer.analyze(
            "g-001", "OP-001", sample_root_cause_record,
        )
        assert len(record.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_confidence_score_positive(self, analyzer, sample_root_cause_record):
        record = await analyzer.analyze(
            "g-001", "OP-001", sample_root_cause_record,
        )
        assert record.confidence_score > 0

    @pytest.mark.asyncio
    async def test_primary_cause_not_empty(self, analyzer, sample_root_cause_record):
        record = await analyzer.analyze(
            "g-001", "OP-001", sample_root_cause_record,
        )
        assert len(record.primary_cause) > 0

    @pytest.mark.asyncio
    async def test_causal_chain_not_empty(self, analyzer, sample_root_cause_record):
        record = await analyzer.analyze(
            "g-001", "OP-001", sample_root_cause_record,
        )
        assert len(record.causal_chain) > 0

    @pytest.mark.asyncio
    async def test_record_stored(self, analyzer, sample_root_cause_record):
        record = await analyzer.analyze(
            "g-001", "OP-001", sample_root_cause_record,
        )
        assert record.root_cause_id in analyzer._records

    @pytest.mark.asyncio
    async def test_recommendations_present(self, analyzer, sample_root_cause_record):
        record = await analyzer.analyze(
            "g-001", "OP-001", sample_root_cause_record,
        )
        assert len(record.recommendations) >= 1

    @pytest.mark.asyncio
    async def test_operator_id_set(self, analyzer, sample_root_cause_record):
        record = await analyzer.analyze(
            "g-001", "OP-001", sample_root_cause_record,
        )
        assert record.operator_id == "OP-001"

    @pytest.mark.asyncio
    async def test_grievance_id_set(self, analyzer, sample_root_cause_record):
        record = await analyzer.analyze(
            "g-001", "OP-001", sample_root_cause_record,
        )
        assert record.grievance_id == "g-001"


# ---------------------------------------------------------------------------
# Retrieval and Listing
# ---------------------------------------------------------------------------


class TestRetrievalAndListing:
    @pytest.mark.asyncio
    async def test_get_root_cause(self, analyzer, sample_root_cause_record):
        record = await analyzer.analyze("g-001", "OP-001", sample_root_cause_record)
        retrieved = await analyzer.get_root_cause(record.root_cause_id)
        assert retrieved is not None
        assert retrieved.root_cause_id == record.root_cause_id

    @pytest.mark.asyncio
    async def test_get_root_cause_not_found(self, analyzer):
        result = await analyzer.get_root_cause("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_all(self, analyzer, sample_root_cause_record):
        await analyzer.analyze("g-001", "OP-001", sample_root_cause_record)
        await analyzer.analyze("g-002", "OP-002", sample_root_cause_record)
        results = await analyzer.list_root_causes()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_list_filter_grievance(self, analyzer, sample_root_cause_record):
        await analyzer.analyze("g-001", "OP-001", sample_root_cause_record)
        await analyzer.analyze("g-002", "OP-001", sample_root_cause_record)
        results = await analyzer.list_root_causes(grievance_id="g-001")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_filter_operator(self, analyzer, sample_root_cause_record):
        await analyzer.analyze("g-001", "OP-001", sample_root_cause_record)
        await analyzer.analyze("g-002", "OP-002", sample_root_cause_record)
        results = await analyzer.list_root_causes(operator_id="OP-001")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_filter_method(self, analyzer, sample_root_cause_record):
        await analyzer.analyze("g-001", "OP-001", sample_root_cause_record, method="fishbone")
        await analyzer.analyze("g-002", "OP-001", sample_root_cause_record, method="five_whys")
        results = await analyzer.list_root_causes(method="fishbone")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_empty(self, analyzer):
        results = await analyzer.list_root_causes()
        assert results == []


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check(self, analyzer):
        health = await analyzer.health_check()
        assert health["status"] == "healthy"
        assert health["engine"] == "RootCauseAnalyzer"

    @pytest.mark.asyncio
    async def test_health_check_record_count(self, analyzer, sample_root_cause_record):
        await analyzer.analyze("g-001", "OP-001", sample_root_cause_record)
        health = await analyzer.health_check()
        assert health["record_count"] == 1
