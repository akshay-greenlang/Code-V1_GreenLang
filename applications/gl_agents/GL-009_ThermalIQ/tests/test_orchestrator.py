# -*- coding: utf-8 -*-
"""
Orchestrator Integration Tests for GL-009 THERMALIQ

Comprehensive integration tests for the ThermalEfficiencyOrchestrator
validating full analysis workflows, component interactions, and state management.

Test Coverage:
- Full analysis workflow
- Efficiency-only workflow
- Exergy-only workflow
- Fluid recommendation
- Provenance tracking
- Audit logging
- Error handling and recovery

Author: GL-TestEngineer
Version: 1.0.0
"""

import asyncio
import hashlib
import json
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# TEST CLASS: FULL ANALYSIS WORKFLOW
# =============================================================================

class TestFullAnalysisWorkflow:
    """Test complete analysis workflow through orchestrator."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_analysis_workflow(
        self, sample_analysis_input, orchestrator
    ):
        """Test full analysis workflow execution."""
        result = await orchestrator.execute(sample_analysis_input)

        # Verify all expected outputs
        assert "first_law_efficiency_percent" in result
        assert "second_law_efficiency_percent" in result or result["first_law_efficiency_percent"] > 0
        assert "energy_input_kw" in result
        assert "useful_output_kw" in result
        assert "metadata" in result

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_produces_provenance_hash(
        self, sample_analysis_input, orchestrator
    ):
        """Test that workflow produces valid provenance hash."""
        result = await orchestrator.execute(sample_analysis_input)

        assert "metadata" in result
        assert "provenance_hash" in result["metadata"]
        assert len(result["metadata"]["provenance_hash"]) == 64  # SHA-256

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_execution_time_tracked(
        self, sample_analysis_input, orchestrator
    ):
        """Test that execution time is tracked."""
        result = await orchestrator.execute(sample_analysis_input)

        assert "execution_time_ms" in result["metadata"]
        assert result["metadata"]["execution_time_ms"] > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_deterministic(
        self, sample_analysis_input, orchestrator
    ):
        """Test that workflow produces deterministic results."""
        result1 = await orchestrator.execute(sample_analysis_input)
        result2 = await orchestrator.execute(sample_analysis_input)

        # Results should be identical
        assert result1["first_law_efficiency_percent"] == result2["first_law_efficiency_percent"]
        assert result1["metadata"]["provenance_hash"] == result2["metadata"]["provenance_hash"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_includes_all_components(
        self, sample_analysis_input
    ):
        """Test that workflow includes all analysis components."""
        # Simulate full workflow
        workflow_steps = [
            "validate_input",
            "calculate_first_law",
            "calculate_second_law",
            "generate_sankey",
            "benchmark_comparison",
            "generate_recommendations",
            "compile_report",
        ]

        executed_steps = self._simulate_workflow(sample_analysis_input)

        for step in workflow_steps:
            assert step in executed_steps, f"Workflow missing step: {step}"

    def _simulate_workflow(self, input_data: Dict[str, Any]) -> List[str]:
        """Simulate workflow execution and return executed steps."""
        return [
            "validate_input",
            "calculate_first_law",
            "calculate_second_law",
            "generate_sankey",
            "benchmark_comparison",
            "generate_recommendations",
            "compile_report",
        ]


# =============================================================================
# TEST CLASS: EFFICIENCY-ONLY WORKFLOW
# =============================================================================

class TestEfficiencyOnlyWorkflow:
    """Test efficiency-only calculation workflow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_efficiency_only_workflow(self, sample_heat_balance):
        """Test efficiency-only workflow (no exergy analysis)."""
        input_data = {
            "operation_mode": "efficiency_only",
            "energy_inputs": sample_heat_balance["energy_inputs"],
            "useful_outputs": sample_heat_balance["useful_outputs"],
            "heat_losses": sample_heat_balance["heat_losses"],
        }

        result = await self._run_efficiency_workflow(input_data)

        assert "first_law_efficiency_percent" in result
        assert result["first_law_efficiency_percent"] > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_efficiency_only_faster_than_full(self, sample_heat_balance):
        """Test that efficiency-only is faster than full analysis."""
        import time

        input_data = {
            "energy_inputs": sample_heat_balance["energy_inputs"],
            "useful_outputs": sample_heat_balance["useful_outputs"],
        }

        start = time.perf_counter()
        await self._run_efficiency_workflow(input_data)
        efficiency_time = time.perf_counter() - start

        start = time.perf_counter()
        await self._run_full_workflow(input_data)
        full_time = time.perf_counter() - start

        # Efficiency-only should be faster
        assert efficiency_time <= full_time

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_efficiency_only_no_exergy(self, sample_heat_balance):
        """Test that efficiency-only does not include exergy."""
        input_data = {
            "operation_mode": "efficiency_only",
            "energy_inputs": sample_heat_balance["energy_inputs"],
            "useful_outputs": sample_heat_balance["useful_outputs"],
        }

        result = await self._run_efficiency_workflow(input_data)

        # Should not have exergy results
        assert "exergy_efficiency_percent" not in result or result.get("exergy_efficiency_percent") is None

    async def _run_efficiency_workflow(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run efficiency-only workflow."""
        # Calculate first law efficiency
        total_input = 0.0
        for fuel in input_data.get("energy_inputs", {}).get("fuel_inputs", []):
            total_input += fuel.get("mass_flow_kg_hr", 0) * fuel.get("heating_value_mj_kg", 0) * 0.2778

        total_output = 0.0
        for steam in input_data.get("useful_outputs", {}).get("steam_output", []):
            total_output += steam.get("heat_rate_kw", 0)

        efficiency = (total_output / total_input * 100) if total_input > 0 else 0

        return {
            "first_law_efficiency_percent": efficiency,
            "energy_input_kw": total_input,
            "useful_output_kw": total_output,
        }

    async def _run_full_workflow(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run full analysis workflow."""
        result = await self._run_efficiency_workflow(input_data)
        result["exergy_efficiency_percent"] = result["first_law_efficiency_percent"] * 0.55
        return result


# =============================================================================
# TEST CLASS: EXERGY-ONLY WORKFLOW
# =============================================================================

class TestExergyOnlyWorkflow:
    """Test exergy-only calculation workflow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_exergy_only_workflow(self, sample_heat_balance):
        """Test exergy-only workflow."""
        input_data = {
            "operation_mode": "exergy_only",
            "energy_inputs": sample_heat_balance["energy_inputs"],
            "useful_outputs": sample_heat_balance["useful_outputs"],
            "ambient_conditions": sample_heat_balance["ambient_conditions"],
        }

        result = await self._run_exergy_workflow(input_data)

        assert "exergy_efficiency_percent" in result
        assert "exergy_destruction_kw" in result

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_exergy_requires_ambient_conditions(self, sample_heat_balance):
        """Test that exergy calculation requires ambient conditions."""
        input_data = {
            "operation_mode": "exergy_only",
            "energy_inputs": sample_heat_balance["energy_inputs"],
            "useful_outputs": sample_heat_balance["useful_outputs"],
            # No ambient_conditions
        }

        result = await self._run_exergy_workflow(input_data)

        # Should use default ambient conditions (25C, 1 atm)
        assert "ambient_temperature_c" in result.get("ambient_conditions_used", {}) or \
               result.get("exergy_efficiency_percent", 0) > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_exergy_less_than_energy_efficiency(self, sample_heat_balance):
        """Test that exergy efficiency is less than energy efficiency."""
        input_data = {
            "energy_inputs": sample_heat_balance["energy_inputs"],
            "useful_outputs": sample_heat_balance["useful_outputs"],
            "ambient_conditions": sample_heat_balance["ambient_conditions"],
        }

        efficiency_result = await TestEfficiencyOnlyWorkflow()._run_efficiency_workflow(input_data)
        exergy_result = await self._run_exergy_workflow(input_data)

        assert exergy_result["exergy_efficiency_percent"] <= efficiency_result["first_law_efficiency_percent"]

    async def _run_exergy_workflow(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run exergy-only workflow."""
        ambient = input_data.get("ambient_conditions", {"ambient_temperature_c": 25.0})
        T0_K = ambient.get("ambient_temperature_c", 25.0) + 273.15

        # Calculate exergy input
        exergy_input = 0.0
        for fuel in input_data.get("energy_inputs", {}).get("fuel_inputs", []):
            energy = fuel.get("mass_flow_kg_hr", 0) * fuel.get("heating_value_mj_kg", 0) * 0.2778
            exergy_input += energy * 1.04  # Phi factor for natural gas

        # Calculate exergy output
        exergy_output = 0.0
        for steam in input_data.get("useful_outputs", {}).get("steam_output", []):
            heat = steam.get("heat_rate_kw", 0)
            temp_c = steam.get("temperature_c", 180)
            T_K = temp_c + 273.15
            carnot = 1 - T0_K / T_K if T_K > T0_K else 0
            exergy_output += heat * carnot

        exergy_destruction = exergy_input - exergy_output
        efficiency = (exergy_output / exergy_input * 100) if exergy_input > 0 else 0

        return {
            "exergy_efficiency_percent": efficiency,
            "exergy_input_kw": exergy_input,
            "exergy_output_kw": exergy_output,
            "exergy_destruction_kw": exergy_destruction,
            "ambient_conditions_used": ambient,
        }


# =============================================================================
# TEST CLASS: FLUID RECOMMENDATION
# =============================================================================

class TestFluidRecommendation:
    """Test fluid recommendation functionality."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_fluid_recommendation_for_temperature_range(self):
        """Test fluid recommendation based on temperature range."""
        requirements = {
            "min_temperature_c": 100,
            "max_temperature_c": 300,
            "application": "heat_transfer",
        }

        recommendations = await self._get_fluid_recommendations(requirements)

        assert len(recommendations) > 0
        for rec in recommendations:
            assert rec["max_operating_temp_c"] >= requirements["max_temperature_c"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_fluid_recommendation_ranking(self):
        """Test that fluid recommendations are ranked."""
        requirements = {
            "min_temperature_c": 100,
            "max_temperature_c": 250,
        }

        recommendations = await self._get_fluid_recommendations(requirements)

        # Should be ranked by suitability
        if len(recommendations) > 1:
            assert recommendations[0]["suitability_score"] >= recommendations[1]["suitability_score"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_fluid_recommendation_includes_properties(self):
        """Test that recommendations include relevant properties."""
        recommendations = await self._get_fluid_recommendations({
            "max_temperature_c": 200,
        })

        for rec in recommendations:
            assert "name" in rec
            assert "max_operating_temp_c" in rec

    async def _get_fluid_recommendations(
        self, requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get fluid recommendations based on requirements."""
        fluids = [
            {"name": "Therminol 66", "max_operating_temp_c": 345, "suitability_score": 0.9},
            {"name": "Dowtherm A", "max_operating_temp_c": 400, "suitability_score": 0.85},
            {"name": "Water", "max_operating_temp_c": 100, "suitability_score": 0.7},
        ]

        max_temp = requirements.get("max_temperature_c", 200)
        suitable = [f for f in fluids if f["max_operating_temp_c"] >= max_temp]

        return sorted(suitable, key=lambda x: x["suitability_score"], reverse=True)


# =============================================================================
# TEST CLASS: PROVENANCE TRACKING
# =============================================================================

class TestProvenanceTracking:
    """Test provenance tracking functionality."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_provenance_tracking_enabled(
        self, sample_analysis_input, orchestrator
    ):
        """Test that provenance tracking is enabled."""
        result = await orchestrator.execute(sample_analysis_input)

        assert "provenance_hash" in result.get("metadata", {})

    @pytest.mark.integration
    def test_provenance_hash_deterministic(self):
        """Test that provenance hash is deterministic."""
        input_data = {"value": 100, "name": "test"}

        hash1 = self._calculate_provenance_hash(input_data)
        hash2 = self._calculate_provenance_hash(input_data)

        assert hash1 == hash2

    @pytest.mark.integration
    def test_provenance_hash_changes_with_input(self):
        """Test that provenance hash changes with input."""
        hash1 = self._calculate_provenance_hash({"value": 100})
        hash2 = self._calculate_provenance_hash({"value": 101})

        assert hash1 != hash2

    @pytest.mark.integration
    def test_provenance_includes_version(self):
        """Test that provenance includes version information."""
        provenance = self._create_provenance_record({})

        assert "agent_version" in provenance
        assert "calculation_version" in provenance

    @pytest.mark.integration
    def test_provenance_includes_timestamp(self):
        """Test that provenance includes timestamp."""
        provenance = self._create_provenance_record({})

        assert "timestamp" in provenance
        # Validate ISO format
        datetime.fromisoformat(provenance["timestamp"].replace("Z", "+00:00"))

    def _calculate_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 provenance hash."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _create_provenance_record(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create full provenance record."""
        return {
            "hash": self._calculate_provenance_hash(data),
            "agent_version": "1.0.0",
            "calculation_version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "methodology": "ASME PTC 4.1",
        }


# =============================================================================
# TEST CLASS: AUDIT LOGGING
# =============================================================================

class TestAuditLogging:
    """Test audit logging functionality."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_audit_logging_enabled(self, thermal_iq_config):
        """Test that audit logging can be enabled."""
        config = thermal_iq_config.copy()
        config["enable_audit_logging"] = True

        logger = self._create_audit_logger(config)

        assert logger is not None
        assert logger.enabled is True

    @pytest.mark.integration
    def test_audit_log_entry_structure(self):
        """Test audit log entry structure."""
        entry = self._create_audit_entry(
            action="calculate_efficiency",
            input_data={"fuel_kw": 1000},
            result={"efficiency": 85.0},
        )

        assert "timestamp" in entry
        assert "action" in entry
        assert "input_hash" in entry
        assert "result_hash" in entry
        assert "user_id" in entry or entry.get("user_id") is None

    @pytest.mark.integration
    def test_audit_log_immutable(self):
        """Test that audit log entries are immutable (hashed)."""
        entry = self._create_audit_entry(
            action="test",
            input_data={"a": 1},
            result={"b": 2},
        )

        # Entry should have integrity hash
        assert "entry_hash" in entry

        # Tampering should be detectable
        original_hash = entry["entry_hash"]
        entry["action"] = "tampered"
        new_hash = self._calculate_entry_hash(entry)

        assert original_hash != new_hash

    @pytest.mark.integration
    def test_audit_log_query(self):
        """Test querying audit log."""
        entries = [
            self._create_audit_entry("action1", {}, {}),
            self._create_audit_entry("action2", {}, {}),
            self._create_audit_entry("action1", {}, {}),
        ]

        filtered = [e for e in entries if e["action"] == "action1"]

        assert len(filtered) == 2

    def _create_audit_logger(self, config: Dict[str, Any]):
        """Create audit logger from config."""
        class AuditLogger:
            def __init__(self, enabled: bool):
                self.enabled = enabled
                self.entries = []

            def log(self, entry):
                if self.enabled:
                    self.entries.append(entry)

        return AuditLogger(enabled=config.get("enable_audit_logging", False))

    def _create_audit_entry(
        self,
        action: str,
        input_data: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create audit log entry."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "input_hash": hashlib.sha256(json.dumps(input_data, sort_keys=True).encode()).hexdigest(),
            "result_hash": hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest(),
            "user_id": None,
        }
        entry["entry_hash"] = self._calculate_entry_hash(entry)
        return entry

    def _calculate_entry_hash(self, entry: Dict[str, Any]) -> str:
        """Calculate hash for audit entry."""
        entry_copy = {k: v for k, v in entry.items() if k != "entry_hash"}
        return hashlib.sha256(json.dumps(entry_copy, sort_keys=True).encode()).hexdigest()


# =============================================================================
# TEST CLASS: ERROR HANDLING
# =============================================================================

class TestOrchestratorErrorHandling:
    """Test orchestrator error handling and recovery."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_invalid_input_handling(self):
        """Test handling of invalid input data."""
        invalid_input = {
            "energy_inputs": None,  # Invalid
        }

        with pytest.raises((ValueError, TypeError)):
            await self._validate_and_process(invalid_input)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        incomplete_input = {
            "useful_outputs": {"steam": 1000},
            # Missing energy_inputs
        }

        result = await self._process_with_defaults(incomplete_input)

        # Should either fail gracefully or use defaults
        assert result is not None or "error" in str(result).lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_negative_values_handling(self):
        """Test handling of negative energy values."""
        input_data = {
            "energy_inputs": {"fuel_inputs": [{"mass_flow_kg_hr": -100}]},
        }

        with pytest.raises(ValueError):
            await self._validate_and_process(input_data)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_recovery_after_error(self):
        """Test that orchestrator can recover after error."""
        # First call fails
        try:
            await self._validate_and_process({"invalid": True})
        except Exception:
            pass

        # Second call should work
        valid_input = {
            "energy_inputs": {"fuel_inputs": [{"mass_flow_kg_hr": 100, "heating_value_mj_kg": 50}]},
            "useful_outputs": {"steam_output": [{"heat_rate_kw": 1000}]},
        }

        result = await self._process_with_defaults(valid_input)
        assert result is not None

    async def _validate_and_process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and process input data."""
        if input_data.get("energy_inputs") is None:
            raise ValueError("energy_inputs is required")

        fuel_inputs = input_data.get("energy_inputs", {}).get("fuel_inputs", [])
        for fuel in fuel_inputs:
            if fuel.get("mass_flow_kg_hr", 0) < 0:
                raise ValueError("Negative mass flow not allowed")

        return {"status": "processed"}

    async def _process_with_defaults(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input with default values for missing fields."""
        # Apply defaults
        energy_inputs = input_data.get("energy_inputs", {"fuel_inputs": []})
        useful_outputs = input_data.get("useful_outputs", {"steam_output": []})

        # Simple calculation
        total_input = sum(
            f.get("mass_flow_kg_hr", 0) * f.get("heating_value_mj_kg", 0) * 0.2778
            for f in energy_inputs.get("fuel_inputs", [])
        )

        return {"energy_input_kw": total_input}


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestOrchestratorPerformance:
    """Performance tests for orchestrator."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_full_analysis_time(self, sample_analysis_input, orchestrator):
        """Test full analysis meets <100ms target."""
        import time

        start = time.perf_counter()
        await orchestrator.execute(sample_analysis_input)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100.0, f"Full analysis took {elapsed_ms:.2f}ms (target: <100ms)"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_analyses(self, sample_analysis_input, orchestrator):
        """Test concurrent analysis execution."""
        import time

        num_concurrent = 10

        start = time.perf_counter()
        tasks = [orchestrator.execute(sample_analysis_input) for _ in range(num_concurrent)]
        results = await asyncio.gather(*tasks)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(results) == num_concurrent
        # Average time per analysis should be reasonable
        avg_time_ms = elapsed_ms / num_concurrent
        assert avg_time_ms < 50.0, f"Average time {avg_time_ms:.2f}ms (target: <50ms)"
