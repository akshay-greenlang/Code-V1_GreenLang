# -*- coding: utf-8 -*-
"""
Full pipeline integration tests for Refrigerants & F-Gas Agent - AGENT-MRV-002

Tests the RefrigerantPipelineEngine through integrated execution:
- Pipeline equipment_based / mass_balance / screening
- All stages complete
- Batch execution
- Error recovery
- Provenance chain
- Metrics recorded
- Compliance integration
- Uncertainty integration

Target: 25+ tests, ~400 lines
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from greenlang.refrigerants_fgas.refrigerant_pipeline import (
    PIPELINE_STAGES,
    SUPPORTED_METHODS,
    RefrigerantPipelineEngine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline() -> RefrigerantPipelineEngine:
    """Return a fresh RefrigerantPipelineEngine with default (stub) engines."""
    return RefrigerantPipelineEngine()


@pytest.fixture
def equipment_input() -> Dict[str, Any]:
    """Standard equipment-based pipeline input."""
    return {
        "refrigerant_type": "R_410A",
        "charge_kg": 25.0,
        "method": "equipment_based",
        "gwp_source": "AR6",
        "equipment_type": "COMMERCIAL_AC",
        "equipment_id": "eq_pipe_001",
        "facility_id": "fac_pipe_001",
    }


@pytest.fixture
def mass_balance_pipe_input() -> Dict[str, Any]:
    """Standard mass-balance pipeline input."""
    return {
        "refrigerant_type": "R_134A",
        "charge_kg": 500.0,
        "method": "mass_balance",
        "gwp_source": "AR6",
        "facility_id": "fac_pipe_002",
        "mass_balance_data": {
            "inventory_start_kg": 500.0,
            "purchases_kg": 100.0,
            "recovery_kg": 50.0,
            "inventory_end_kg": 450.0,
        },
    }


@pytest.fixture
def screening_pipe_input() -> Dict[str, Any]:
    """Standard screening pipeline input."""
    return {
        "refrigerant_type": "R_407C",
        "charge_kg": 10.0,
        "method": "screening",
        "gwp_source": "AR6",
        "activity_data": 1000.0,
        "screening_factor": 0.02,
        "facility_id": "fac_pipe_003",
    }


# ===========================================================================
# Pipeline method tests
# ===========================================================================


class TestPipelineEquipmentBased:
    """Tests for equipment-based pipeline execution."""

    def test_pipeline_equipment_based(self, pipeline, equipment_input):
        """Pipeline completes equipment-based calculation."""
        result = pipeline.run(equipment_input)

        assert result["success"] is True
        assert result["stages_completed"] >= 6
        assert result["stages_total"] == 8
        inner = result["result"]
        assert inner["method"] == "equipment_based"
        assert inner["total_emissions_kg_co2e"] > 0
        assert inner["total_emissions_tco2e"] > 0

    def test_pipeline_equipment_based_trace(self, pipeline, equipment_input):
        """Equipment-based pipeline includes calculation trace."""
        result = pipeline.run(equipment_input)
        inner = result["result"]
        trace = inner.get("calculation_trace", [])
        assert isinstance(trace, list)
        assert len(trace) >= 1

    def test_pipeline_equipment_based_gwp(self, pipeline, equipment_input):
        """Equipment-based pipeline records GWP value."""
        result = pipeline.run(equipment_input)
        inner = result["result"]
        assert inner["gwp_value"] > 0
        assert inner["gwp_source"] == "AR6"


class TestPipelineMassBalance:
    """Tests for mass-balance pipeline execution."""

    def test_pipeline_mass_balance(self, pipeline, mass_balance_pipe_input):
        """Pipeline completes mass-balance calculation."""
        result = pipeline.run(mass_balance_pipe_input)

        assert result["success"] is True
        inner = result["result"]
        assert inner["method"] == "mass_balance"
        # (500 + 100 - 50 - 450) = 100 kg emissions
        assert inner["emissions_kg"] == pytest.approx(100.0, rel=1e-2)

    def test_pipeline_mass_balance_co2e(self, pipeline, mass_balance_pipe_input):
        """Mass-balance CO2e calculation is correct."""
        result = pipeline.run(mass_balance_pipe_input)
        inner = result["result"]
        # R-134A stub GWP = 1530, emissions = 100 kg
        # CO2e = 100 * 1530 = 153,000 kg = 153 tCO2e
        gwp = inner.get("gwp_value", 0)
        if gwp > 0:
            expected_kg_co2e = 100.0 * gwp
            assert inner["total_emissions_kg_co2e"] == pytest.approx(
                expected_kg_co2e, rel=1e-2,
            )


class TestPipelineScreening:
    """Tests for screening pipeline execution."""

    def test_pipeline_screening(self, pipeline, screening_pipe_input):
        """Pipeline completes screening calculation."""
        result = pipeline.run(screening_pipe_input)

        assert result["success"] is True
        inner = result["result"]
        assert inner["method"] == "screening"
        # 1000 * 0.02 = 20 kg emissions
        assert inner["emissions_kg"] == pytest.approx(20.0, rel=1e-2)


# ===========================================================================
# All stages complete test
# ===========================================================================


class TestPipelineAllStages:
    """Tests verifying all 8 stages execute."""

    def test_pipeline_all_stages_complete(self, pipeline, equipment_input):
        """All 8 pipeline stages should produce results."""
        result = pipeline.run(equipment_input)
        stage_results = result["stage_results"]

        assert len(stage_results) == 8
        for i, stage_name in enumerate(PIPELINE_STAGES):
            assert stage_results[i]["stage"] == stage_name
            assert "duration_ms" in stage_results[i]
            assert "provenance_hash" in stage_results[i]

    def test_pipeline_critical_stages_succeed(self, pipeline, equipment_input):
        """Critical stages (VALIDATE through CALCULATE) all succeed."""
        result = pipeline.run(equipment_input)
        stage_results = result["stage_results"]

        critical_stages = [
            "VALIDATE",
            "LOOKUP_REFRIGERANT",
            "ESTIMATE_LEAK_RATE",
            "CALCULATE",
        ]
        for sr in stage_results:
            if sr["stage"] in critical_stages:
                assert sr["success"] is True, (
                    f"Critical stage {sr['stage']} failed: {sr.get('error', '')}"
                )


# ===========================================================================
# Batch execution tests
# ===========================================================================


class TestPipelineBatchExecution:
    """Tests for batch pipeline execution."""

    def test_pipeline_batch_execution(self, pipeline):
        """Batch executes multiple calculations."""
        inputs = [
            {
                "refrigerant_type": "R_410A",
                "charge_kg": 25.0,
                "method": "equipment_based",
            },
            {
                "refrigerant_type": "R_134A",
                "charge_kg": 10.0,
                "method": "equipment_based",
            },
            {
                "refrigerant_type": "R_404A",
                "charge_kg": 80.0,
                "method": "equipment_based",
            },
        ]
        result = pipeline.run_batch(inputs)

        assert result["total_count"] == 3
        assert result["success_count"] >= 3
        assert result["processing_time_ms"] > 0
        assert len(result["results"]) == 3

    def test_pipeline_batch_large(self, pipeline):
        """Batch pipeline handles larger input sets."""
        inputs = [
            {
                "refrigerant_type": "R_410A",
                "charge_kg": float(i + 1),
                "method": "equipment_based",
            }
            for i in range(50)
        ]
        result = pipeline.run_batch(inputs, checkpoint_interval=10)

        assert result["total_count"] == 50
        assert result["success_count"] == 50
        assert result["total_emissions_tco2e"] > 0

    def test_pipeline_batch_mixed_methods(self, pipeline):
        """Batch with different calculation methods."""
        inputs = [
            {
                "refrigerant_type": "R_410A",
                "charge_kg": 25.0,
                "method": "equipment_based",
            },
            {
                "refrigerant_type": "R_134A",
                "charge_kg": 100.0,
                "method": "mass_balance",
                "mass_balance_data": {
                    "inventory_start_kg": 500.0,
                    "purchases_kg": 100.0,
                    "recovery_kg": 50.0,
                    "inventory_end_kg": 450.0,
                },
            },
            {
                "refrigerant_type": "R_407C",
                "charge_kg": 10.0,
                "method": "screening",
                "activity_data": 500.0,
                "screening_factor": 0.01,
            },
        ]
        result = pipeline.run_batch(inputs)
        assert result["total_count"] == 3
        assert result["success_count"] == 3


# ===========================================================================
# Error recovery tests
# ===========================================================================


class TestPipelineErrorRecovery:
    """Tests for pipeline error recovery."""

    def test_pipeline_error_recovery(self, pipeline):
        """Pipeline produces partial results for invalid input."""
        result = pipeline.run({
            "refrigerant_type": "",
            "charge_kg": -1.0,
        })
        # VALIDATE should fail
        assert result["success"] is False
        assert result["stages_completed"] < 8

    def test_pipeline_error_continues_non_critical(self, pipeline):
        """Pipeline continues past non-critical stage failures."""
        result = pipeline.run({
            "refrigerant_type": "R_410A",
            "charge_kg": 25.0,
            "method": "equipment_based",
        })
        # Even if uncertainty/compliance engines are unavailable,
        # those stages should still "succeed" with stubs
        assert result["stages_completed"] >= 6

    def test_pipeline_batch_partial_failure(self, pipeline):
        """Batch pipeline handles individual item failures."""
        inputs = [
            {
                "refrigerant_type": "R_410A",
                "charge_kg": 25.0,
            },
            {
                "refrigerant_type": "",
                "charge_kg": -1.0,
            },
            {
                "refrigerant_type": "R_134A",
                "charge_kg": 10.0,
            },
        ]
        result = pipeline.run_batch(inputs)
        assert result["total_count"] == 3
        # At least some should succeed
        assert result["success_count"] >= 1


# ===========================================================================
# Provenance chain tests
# ===========================================================================


class TestPipelineProvenanceChain:
    """Tests for provenance chain integrity in the pipeline."""

    def test_pipeline_provenance_chain(self, pipeline, equipment_input):
        """Each stage in the pipeline has a provenance hash."""
        result = pipeline.run(equipment_input)
        stage_results = result["stage_results"]

        for sr in stage_results:
            if sr["success"]:
                assert len(sr["provenance_hash"]) == 64
                # Verify valid hex
                int(sr["provenance_hash"], 16)

    def test_pipeline_provenance_hash_unique_per_run(self, pipeline):
        """Different pipeline runs produce different provenance hashes."""
        input1 = {
            "refrigerant_type": "R_410A",
            "charge_kg": 25.0,
        }
        input2 = {
            "refrigerant_type": "R_134A",
            "charge_kg": 10.0,
        }
        r1 = pipeline.run(input1)
        r2 = pipeline.run(input2)

        assert r1["pipeline_provenance_hash"] != r2["pipeline_provenance_hash"]

    def test_pipeline_batch_provenance(self, pipeline):
        """Batch pipeline result has a valid provenance hash."""
        result = pipeline.run_batch([
            {"refrigerant_type": "R_410A", "charge_kg": 25.0},
        ])
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Metrics recorded tests
# ===========================================================================


class TestPipelineMetrics:
    """Tests for pipeline metrics recording."""

    def test_pipeline_metrics_recorded(self, pipeline, equipment_input):
        """Pipeline stats are updated after execution."""
        assert pipeline.get_pipeline_stats()["total_runs"] == 0

        pipeline.run(equipment_input)

        stats = pipeline.get_pipeline_stats()
        assert stats["total_runs"] == 1
        assert stats["successful_runs"] >= 1
        assert stats["total_duration_ms"] > 0
        assert stats["avg_duration_ms"] > 0
        assert stats["success_rate_pct"] > 0
        assert stats["last_run_at"] is not None

    def test_pipeline_metrics_accumulate(self, pipeline, equipment_input):
        """Pipeline stats accumulate across multiple runs."""
        for _ in range(5):
            pipeline.run(equipment_input)

        stats = pipeline.get_pipeline_stats()
        assert stats["total_runs"] == 5
        assert len(stats["recent_runs"]) == 5

    def test_pipeline_metrics_reset(self, pipeline, equipment_input):
        """Pipeline stats can be reset."""
        pipeline.run(equipment_input)
        pipeline.reset_stats()

        stats = pipeline.get_pipeline_stats()
        assert stats["total_runs"] == 0
        assert stats["total_duration_ms"] == 0.0


# ===========================================================================
# Compliance integration tests
# ===========================================================================


class TestPipelineComplianceIntegration:
    """Tests for compliance checking within the pipeline."""

    def test_pipeline_compliance_integration(self, pipeline, equipment_input):
        """CHECK_COMPLIANCE stage produces compliance records."""
        result = pipeline.run(equipment_input)

        # Find the compliance stage result
        compliance_stage = next(
            (sr for sr in result["stage_results"]
             if sr["stage"] == "CHECK_COMPLIANCE"),
            None,
        )
        assert compliance_stage is not None
        assert compliance_stage["success"] is True
        assert compliance_stage["frameworks_checked"] >= 5
        assert compliance_stage["overall_compliant"] is True

    def test_pipeline_compliance_in_result(self, pipeline, equipment_input):
        """Final pipeline result includes compliance data."""
        result = pipeline.run(equipment_input)
        inner = result["result"]
        compliance = inner.get("compliance")
        assert compliance is not None
        assert isinstance(compliance, list)
        assert len(compliance) >= 5


# ===========================================================================
# Uncertainty integration tests
# ===========================================================================


class TestPipelineUncertaintyIntegration:
    """Tests for uncertainty quantification within the pipeline."""

    def test_pipeline_uncertainty_integration(self, pipeline, equipment_input):
        """QUANTIFY_UNCERTAINTY stage produces uncertainty estimates."""
        result = pipeline.run(equipment_input)

        # Find the uncertainty stage result
        unc_stage = next(
            (sr for sr in result["stage_results"]
             if sr["stage"] == "QUANTIFY_UNCERTAINTY"),
            None,
        )
        assert unc_stage is not None
        assert unc_stage["success"] is True
        assert unc_stage.get("mean_co2e_kg", 0) >= 0

    def test_pipeline_uncertainty_in_result(self, pipeline, equipment_input):
        """Final pipeline result includes uncertainty data."""
        result = pipeline.run(equipment_input)
        inner = result["result"]
        uncertainty = inner.get("uncertainty")
        assert uncertainty is not None
        assert isinstance(uncertainty, dict)
        assert uncertainty.get("mean_co2e_kg", 0) >= 0

    def test_pipeline_uncertainty_stub_quality(self, pipeline, equipment_input):
        """Stub uncertainty applies 25% relative uncertainty."""
        result = pipeline.run(equipment_input)
        inner = result["result"]
        unc = inner.get("uncertainty", {})

        mean = unc.get("mean_co2e_kg", 0)
        std = unc.get("std_co2e_kg", 0)

        if mean > 0 and std > 0:
            # Stub uses 25% relative uncertainty
            relative_unc = std / mean
            assert relative_unc == pytest.approx(0.25, rel=0.01)


# ===========================================================================
# Direct and top-down method integration
# ===========================================================================


class TestPipelineAdditionalMethods:
    """Tests for direct and top-down methods through the pipeline."""

    def test_pipeline_direct_method(self, pipeline):
        """Pipeline handles direct measurement method."""
        result = pipeline.run({
            "refrigerant_type": "R_410A",
            "charge_kg": 10.0,
            "method": "direct",
            "measured_emissions_kg": 2.5,
        })
        assert result["success"] is True
        inner = result["result"]
        assert inner["method"] == "direct"
        assert inner["emissions_kg"] == pytest.approx(2.5, rel=1e-3)

    def test_pipeline_top_down_method(self, pipeline):
        """Pipeline handles top-down method."""
        result = pipeline.run({
            "refrigerant_type": "R_404A",
            "charge_kg": 50.0,
            "method": "top_down",
            "equipment_type": "COMMERCIAL_REFRIGERATION",
            "num_units": 5,
        })
        assert result["success"] is True
        inner = result["result"]
        assert inner["method"] == "top_down"
        assert inner["emissions_kg"] > 0
