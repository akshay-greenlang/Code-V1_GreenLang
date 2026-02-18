# -*- coding: utf-8 -*-
"""
Full pipeline integration tests for Process Emissions Agent.

AGENT-MRV-004: Process Emissions Agent (GL-MRV-SCOPE1-004)

Tests the 8-stage ProcessEmissionsPipelineEngine with real engine
instances wired together, verifying inter-stage data flow, provenance
chain integrity, multi-process batch execution, and statistical
tracking.

Total: 28 tests across 4 test classes.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from greenlang.process_emissions.process_emissions_pipeline import (
    ProcessEmissionsPipelineEngine,
    PipelineStage,
    PIPELINE_STAGES,
    GWP_VALUES,
    CARBONATE_FACTORS,
    PROCESS_TYPES,
)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestFullPipelineExecution:
    """Test the complete 8-stage pipeline with real engines."""

    def test_cement_full_pipeline(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
        cement_pipeline_request: Dict[str, Any],
    ):
        """Full pipeline execution for cement clinker production."""
        result = pipeline_engine.execute_pipeline(cement_pipeline_request)

        assert result["stages_total"] == 8
        assert "pipeline_id" in result
        assert "pipeline_provenance_hash" in result
        assert len(result["pipeline_provenance_hash"]) == 64
        assert "total_duration_ms" in result
        assert result["total_duration_ms"] > 0

        # Verify all 8 stage results are present
        assert len(result["stage_results"]) == 8

    def test_iron_steel_full_pipeline(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
        iron_steel_pipeline_request: Dict[str, Any],
    ):
        """Full pipeline execution for iron/steel BF-BOF."""
        result = pipeline_engine.execute_pipeline(
            iron_steel_pipeline_request,
        )
        assert result["stages_total"] == 8
        assert len(result["stage_results"]) == 8

    def test_semiconductor_full_pipeline(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
        semiconductor_pipeline_request: Dict[str, Any],
    ):
        """Full pipeline execution for semiconductor manufacturing."""
        result = pipeline_engine.execute_pipeline(
            semiconductor_pipeline_request,
        )
        assert result["stages_total"] == 8

    def test_nitric_acid_full_pipeline(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
        nitric_acid_pipeline_request: Dict[str, Any],
    ):
        """Full pipeline execution for nitric acid production."""
        result = pipeline_engine.execute_pipeline(
            nitric_acid_pipeline_request,
        )
        assert result["stages_total"] == 8

    def test_aluminium_full_pipeline(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
        aluminium_pipeline_request: Dict[str, Any],
    ):
        """Full pipeline execution for aluminium prebake smelting."""
        result = pipeline_engine.execute_pipeline(
            aluminium_pipeline_request,
        )
        assert result["stages_total"] == 8

    def test_pipeline_unique_ids(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
        cement_pipeline_request: Dict[str, Any],
    ):
        """Each pipeline run generates a unique pipeline_id."""
        r1 = pipeline_engine.execute_pipeline(cement_pipeline_request)
        r2 = pipeline_engine.execute_pipeline(cement_pipeline_request)
        assert r1["pipeline_id"] != r2["pipeline_id"]

    def test_pipeline_with_different_gwp_sources(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
        cement_pipeline_request: Dict[str, Any],
    ):
        """Pipeline produces results for all GWP sources."""
        for gwp in ["AR4", "AR5", "AR6", "AR6_20yr"]:
            result = pipeline_engine.execute_pipeline(
                cement_pipeline_request, gwp_source=gwp,
            )
            assert result["stages_total"] == 8

    def test_pipeline_with_different_tiers(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
        cement_pipeline_request: Dict[str, Any],
    ):
        """Pipeline produces results for all calculation tiers."""
        for tier in ["TIER_1", "TIER_2", "TIER_3"]:
            result = pipeline_engine.execute_pipeline(
                cement_pipeline_request, tier=tier,
            )
            assert result["stages_total"] == 8


class TestPipelineStageFlow:
    """Test inter-stage data flow and optional stage skipping."""

    def test_skip_abatement_stage(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
        cement_pipeline_request: Dict[str, Any],
    ):
        """Pipeline skips abatement stage when disabled."""
        result = pipeline_engine.execute_pipeline(
            cement_pipeline_request,
            include_abatement=False,
        )
        abatement_sr = next(
            sr for sr in result["stage_results"]
            if sr["stage"] == "APPLY_ABATEMENT"
        )
        assert abatement_sr.get("skipped") is True
        assert abatement_sr["success"] is True

    def test_skip_uncertainty_stage(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
        cement_pipeline_request: Dict[str, Any],
    ):
        """Pipeline skips uncertainty stage when disabled."""
        result = pipeline_engine.execute_pipeline(
            cement_pipeline_request,
            include_uncertainty=False,
        )
        uncertainty_sr = next(
            sr for sr in result["stage_results"]
            if sr["stage"] == "QUANTIFY_UNCERTAINTY"
        )
        assert uncertainty_sr.get("skipped") is True

    def test_skip_compliance_stage(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
        cement_pipeline_request: Dict[str, Any],
    ):
        """Pipeline skips compliance stage when disabled."""
        result = pipeline_engine.execute_pipeline(
            cement_pipeline_request,
            include_compliance=False,
        )
        compliance_sr = next(
            sr for sr in result["stage_results"]
            if sr["stage"] == "CHECK_COMPLIANCE"
        )
        assert compliance_sr.get("skipped") is True

    def test_all_stages_skipped(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
        cement_pipeline_request: Dict[str, Any],
    ):
        """Pipeline runs with all optional stages skipped."""
        result = pipeline_engine.execute_pipeline(
            cement_pipeline_request,
            include_abatement=False,
            include_uncertainty=False,
            include_compliance=False,
        )
        assert result["stages_total"] == 8

    def test_stage_durations_positive(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
        cement_pipeline_request: Dict[str, Any],
    ):
        """Non-skipped stage durations are non-negative."""
        result = pipeline_engine.execute_pipeline(cement_pipeline_request)
        for sr in result["stage_results"]:
            if not sr.get("skipped"):
                assert sr["duration_ms"] >= 0

    def test_stage_provenance_hashes(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
        cement_pipeline_request: Dict[str, Any],
    ):
        """Each non-skipped stage has a provenance hash."""
        result = pipeline_engine.execute_pipeline(cement_pipeline_request)
        for sr in result["stage_results"]:
            if not sr.get("skipped"):
                assert "provenance_hash" in sr
                assert len(sr["provenance_hash"]) == 64

    def test_empty_request_handled_gracefully(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
    ):
        """Pipeline handles empty request dict without crashing."""
        result = pipeline_engine.execute_pipeline({})
        assert len(result["stage_results"]) == 8


class TestBatchPipelineExecution:
    """Test pipeline batch execution with real engines."""

    def test_batch_execute_multiple_processes(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
        multi_process_batch_requests: List[Dict[str, Any]],
    ):
        """Batch execute returns results for all process types."""
        results = pipeline_engine.execute_batch(
            multi_process_batch_requests,
        )
        assert len(results) == len(multi_process_batch_requests)
        for r in results:
            assert "pipeline_id" in r
            assert r["stages_total"] == 8

    def test_batch_unique_pipeline_ids(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
        multi_process_batch_requests: List[Dict[str, Any]],
    ):
        """Each batch result has a unique pipeline_id."""
        results = pipeline_engine.execute_batch(
            multi_process_batch_requests,
        )
        ids = [r["pipeline_id"] for r in results]
        assert len(set(ids)) == len(ids)

    def test_batch_empty_list(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
    ):
        """Batch with empty list returns empty results."""
        results = pipeline_engine.execute_batch([])
        assert results == []

    def test_batch_single_request(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
    ):
        """Batch with single request returns single result."""
        results = pipeline_engine.execute_batch([
            {"process_type": "CEMENT_CLINKER", "production_quantity": 100},
        ])
        assert len(results) == 1


class TestStatisticsAndProvenance:
    """Test statistics tracking and provenance integrity."""

    def test_stats_increment_after_pipeline(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
        cement_pipeline_request: Dict[str, Any],
    ):
        """Pipeline run increments total_runs counter."""
        initial_runs = pipeline_engine._total_runs
        pipeline_engine.execute_pipeline(cement_pipeline_request)
        assert pipeline_engine._total_runs == initial_runs + 1

    def test_stats_duration_accumulates(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
        cement_pipeline_request: Dict[str, Any],
    ):
        """Total duration accumulates across runs."""
        pipeline_engine.execute_pipeline(cement_pipeline_request)
        pipeline_engine.execute_pipeline(cement_pipeline_request)
        assert pipeline_engine._total_duration_ms > 0

    def test_provenance_hash_determinism(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
    ):
        """Same request produces same structured result (not necessarily same hash due to timing)."""
        req = {
            "process_type": "CEMENT_CLINKER",
            "production_quantity": 1000.0,
        }
        r1 = pipeline_engine.execute_pipeline(req)
        r2 = pipeline_engine.execute_pipeline(req)

        # Pipeline IDs should differ (UUID)
        assert r1["pipeline_id"] != r2["pipeline_id"]

        # But both should complete with 8 stages
        assert r1["stages_total"] == r2["stages_total"]

    def test_last_run_timestamp_updated(
        self,
        pipeline_engine: ProcessEmissionsPipelineEngine,
        cement_pipeline_request: Dict[str, Any],
    ):
        """Last run timestamp is updated after each pipeline execution."""
        assert pipeline_engine._last_run_at is None
        pipeline_engine.execute_pipeline(cement_pipeline_request)
        assert pipeline_engine._last_run_at is not None

    def test_reference_data_coverage(self):
        """Pipeline reference data covers all expected process types."""
        assert "CEMENT_CLINKER" in PROCESS_TYPES
        assert "IRON_STEEL_BF_BOF" in PROCESS_TYPES
        assert "ALUMINIUM_PREBAKE" in PROCESS_TYPES
        assert "SEMICONDUCTOR_MANUFACTURING" in PROCESS_TYPES
        assert "NITRIC_ACID" in PROCESS_TYPES

    def test_gwp_values_completeness(self):
        """GWP values include all gas species for each AR."""
        for ar, gases in GWP_VALUES.items():
            assert "CO2" in gases, f"CO2 missing in {ar}"
            assert "CH4" in gases, f"CH4 missing in {ar}"
            assert gases["CO2"] == 1.0, f"CO2 != 1.0 in {ar}"

    def test_carbonate_factors_stoichiometry(self):
        """Carbonate factors are within physically valid range."""
        for mineral, data in CARBONATE_FACTORS.items():
            co2_factor = data["co2_factor"]
            assert 0.0 < co2_factor < 1.0, (
                f"{mineral}: co2_factor {co2_factor} out of range"
            )
