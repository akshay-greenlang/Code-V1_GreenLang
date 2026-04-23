"""
GL-006 HEATRECLAIM - Orchestrator Tests

Integration tests for the main orchestrator.
"""

import pytest
import asyncio
from typing import List

from ..core.orchestrator import HeatReclaimOrchestrator
from ..core.config import (
    HeatReclaimConfig,
    OptimizationObjective,
    OptimizationMode,
)
from ..core.schemas import (
    HeatStream,
    OptimizationResult,
    OptimizationStatus,
)


class TestHeatReclaimOrchestrator:
    """Tests for HeatReclaimOrchestrator."""

    def test_init_default(self):
        """Test default initialization."""
        orchestrator = HeatReclaimOrchestrator()
        assert orchestrator.config is not None
        assert orchestrator.config.delta_t_min_C == 10.0

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = HeatReclaimConfig(delta_t_min_C=15.0)
        orchestrator = HeatReclaimOrchestrator(config=config)
        assert orchestrator.config.delta_t_min_C == 15.0

    def test_optimize_simple_problem(
        self,
        simple_hot_streams,
        simple_cold_streams,
    ):
        """Test optimization on simple problem."""
        orchestrator = HeatReclaimOrchestrator()

        result = orchestrator.optimize(
            hot_streams=simple_hot_streams,
            cold_streams=simple_cold_streams,
            delta_t_min=10.0,
        )

        assert isinstance(result, OptimizationResult)
        assert result.status == OptimizationStatus.COMPLETED
        assert result.pinch_analysis is not None
        assert result.recommended_design is not None

    def test_optimize_with_exergy(
        self,
        simple_hot_streams,
        simple_cold_streams,
    ):
        """Test optimization with exergy analysis."""
        orchestrator = HeatReclaimOrchestrator()

        result = orchestrator.optimize(
            hot_streams=simple_hot_streams,
            cold_streams=simple_cold_streams,
            delta_t_min=10.0,
            include_exergy=True,
        )

        assert result.recommended_design.exergy_analysis is not None

    def test_optimize_with_economics(
        self,
        simple_hot_streams,
        simple_cold_streams,
    ):
        """Test optimization with economic analysis."""
        orchestrator = HeatReclaimOrchestrator()

        result = orchestrator.optimize(
            hot_streams=simple_hot_streams,
            cold_streams=simple_cold_streams,
            delta_t_min=10.0,
        )

        assert result.recommended_design.economic_analysis is not None

    def test_run_pinch_analysis(
        self,
        simple_hot_streams,
        simple_cold_streams,
    ):
        """Test pinch analysis only."""
        orchestrator = HeatReclaimOrchestrator()

        result = orchestrator.run_pinch_analysis(
            hot_streams=simple_hot_streams,
            cold_streams=simple_cold_streams,
            delta_t_min=10.0,
        )

        assert result.pinch_temperature_C > 0
        assert result.minimum_hot_utility_kW >= 0
        assert result.minimum_cold_utility_kW >= 0

    def test_different_objectives(
        self,
        simple_hot_streams,
        simple_cold_streams,
    ):
        """Test different optimization objectives."""
        orchestrator = HeatReclaimOrchestrator()

        objectives = [
            OptimizationObjective.MINIMIZE_COST,
            OptimizationObjective.MINIMIZE_UTILITY,
            OptimizationObjective.MINIMIZE_EXCHANGERS,
        ]

        results = {}
        for obj in objectives:
            result = orchestrator.optimize(
                hot_streams=simple_hot_streams,
                cold_streams=simple_cold_streams,
                objective=obj,
            )
            results[obj] = result

        # All should complete successfully
        assert all(r.status == OptimizationStatus.COMPLETED for r in results.values())

        # MINIMIZE_UTILITY should have lowest utility
        assert (
            results[OptimizationObjective.MINIMIZE_UTILITY].recommended_design.hot_utility_required_kW
            <= results[OptimizationObjective.MINIMIZE_COST].recommended_design.hot_utility_required_kW
        )

    @pytest.mark.asyncio
    async def test_optimize_async(
        self,
        simple_hot_streams,
        simple_cold_streams,
    ):
        """Test async optimization."""
        orchestrator = HeatReclaimOrchestrator()

        result = await orchestrator.optimize_async(
            hot_streams=simple_hot_streams,
            cold_streams=simple_cold_streams,
            delta_t_min=10.0,
        )

        assert result.status == OptimizationStatus.COMPLETED

    def test_health_check(self):
        """Test health check endpoint."""
        orchestrator = HeatReclaimOrchestrator()
        health = orchestrator.health_check()

        assert health["status"] == "healthy"
        assert "components" in health

    def test_get_status(self):
        """Test agent status."""
        orchestrator = HeatReclaimOrchestrator()
        status = orchestrator.get_status()

        assert status.agent_id == "GL-006"
        assert status.is_ready is True

    def test_invalid_streams_rejected(self):
        """Test that invalid streams are rejected."""
        orchestrator = HeatReclaimOrchestrator()

        # Empty streams
        with pytest.raises(ValueError):
            orchestrator.optimize(
                hot_streams=[],
                cold_streams=[],
            )

    def test_result_provenance(
        self,
        simple_hot_streams,
        simple_cold_streams,
    ):
        """Test result has provenance information."""
        orchestrator = HeatReclaimOrchestrator()

        result = orchestrator.optimize(
            hot_streams=simple_hot_streams,
            cold_streams=simple_cold_streams,
        )

        assert result.request_id is not None
        assert result.pinch_analysis.computation_hash is not None

    def test_explainability_generated(
        self,
        simple_hot_streams,
        simple_cold_streams,
    ):
        """Test explainability report is generated."""
        orchestrator = HeatReclaimOrchestrator()

        result = orchestrator.optimize(
            hot_streams=simple_hot_streams,
            cold_streams=simple_cold_streams,
        )

        assert result.explanation_summary is not None
        assert len(result.explanation_summary) > 0
        assert result.key_drivers is not None


class TestWorkflowIntegration:
    """Integration tests for complete workflow."""

    def test_complete_workflow(
        self,
        textbook_hot_streams,
        textbook_cold_streams,
    ):
        """Test complete optimization workflow."""
        orchestrator = HeatReclaimOrchestrator()

        result = orchestrator.optimize(
            hot_streams=textbook_hot_streams,
            cold_streams=textbook_cold_streams,
            delta_t_min=10.0,
            include_exergy=True,
            include_uncertainty=False,
        )

        # Verify all components present
        assert result.pinch_analysis is not None
        assert result.recommended_design is not None
        assert result.recommended_design.exchangers is not None
        assert len(result.recommended_design.exchangers) > 0

        # Verify energy balance
        total_hot = sum(s.duty_kW for s in textbook_hot_streams)
        total_cold = sum(s.duty_kW for s in textbook_cold_streams)
        heat_recovered = result.recommended_design.total_heat_recovered_kW
        hot_utility = result.recommended_design.hot_utility_required_kW
        cold_utility = result.recommended_design.cold_utility_required_kW

        # Hot balance: total_hot = heat_recovered + cold_utility
        hot_balance_error = abs(total_hot - heat_recovered - cold_utility)
        assert hot_balance_error < 10.0  # Within 10 kW

        # Cold balance: total_cold = heat_recovered + hot_utility
        cold_balance_error = abs(total_cold - heat_recovered - hot_utility)
        assert cold_balance_error < 10.0

    def test_industrial_scale_problem(
        self,
        industrial_hot_streams,
        industrial_cold_streams,
    ):
        """Test industrial-scale optimization."""
        orchestrator = HeatReclaimOrchestrator()

        result = orchestrator.optimize(
            hot_streams=industrial_hot_streams,
            cold_streams=industrial_cold_streams,
            delta_t_min=20.0,
        )

        assert result.status == OptimizationStatus.COMPLETED
        assert result.optimization_time_seconds < 60.0  # Should complete in < 1 minute

    def test_deterministic_results(
        self,
        simple_hot_streams,
        simple_cold_streams,
    ):
        """Test results are deterministic."""
        orchestrator = HeatReclaimOrchestrator()

        result1 = orchestrator.optimize(
            hot_streams=simple_hot_streams,
            cold_streams=simple_cold_streams,
            delta_t_min=10.0,
        )

        result2 = orchestrator.optimize(
            hot_streams=simple_hot_streams,
            cold_streams=simple_cold_streams,
            delta_t_min=10.0,
        )

        # Results should be identical
        assert result1.pinch_analysis.pinch_temperature_C == result2.pinch_analysis.pinch_temperature_C
        assert result1.pinch_analysis.maximum_heat_recovery_kW == result2.pinch_analysis.maximum_heat_recovery_kW
        assert result1.recommended_design.exchanger_count == result2.recommended_design.exchanger_count


class TestErrorHandling:
    """Tests for error handling."""

    def test_infeasible_problem_handling(self):
        """Test handling of infeasible problems."""
        orchestrator = HeatReclaimOrchestrator()

        # Create impossible problem: cold target > hot supply
        from ..core.config import StreamType, Phase

        hot = [
            HeatStream(
                stream_id="H1",
                stream_name="Hot 1",
                stream_type=StreamType.HOT,
                fluid_name="Water",
                phase=Phase.LIQUID,
                T_supply_C=50.0,  # Lower than cold target
                T_target_C=30.0,
                m_dot_kg_s=1.0,
                Cp_kJ_kgK=4.18,
            ),
        ]
        cold = [
            HeatStream(
                stream_id="C1",
                stream_name="Cold 1",
                stream_type=StreamType.COLD,
                fluid_name="Water",
                phase=Phase.LIQUID,
                T_supply_C=20.0,
                T_target_C=100.0,  # Higher than hot supply
                m_dot_kg_s=1.0,
                Cp_kJ_kgK=4.18,
            ),
        ]

        # Should still return result (with all utility)
        result = orchestrator.optimize(hot_streams=hot, cold_streams=cold)
        assert result is not None

    def test_timeout_handling(
        self,
        simple_hot_streams,
        simple_cold_streams,
    ):
        """Test timeout is respected."""
        orchestrator = HeatReclaimOrchestrator()

        # Very short timeout
        result = orchestrator.optimize(
            hot_streams=simple_hot_streams,
            cold_streams=simple_cold_streams,
            max_time_seconds=0.001,
        )

        # Should still return something (possibly suboptimal)
        assert result is not None
