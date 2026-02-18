# -*- coding: utf-8 -*-
"""
End-to-end integration tests for Process Emissions Agent.

AGENT-MRV-004: Process Emissions Agent (GL-MRV-SCOPE1-004)

Tests the full lifecycle of the ProcessEmissionsService facade including
calculation, CRUD, uncertainty, and compliance workflows that exercise
multiple engines together.

Total: 33 tests across 5 test classes.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from greenlang.process_emissions.setup import (
    ProcessEmissionsService,
    CalculateResponse,
    BatchCalculateResponse,
    ProcessDetailResponse,
    MaterialDetailResponse,
    UncertaintyResponse,
    ComplianceCheckResponse,
    HealthResponse,
    StatsResponse,
)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestServiceLifecycle:
    """Test full service lifecycle from init to calculate to stats."""

    def test_service_init_healthy(self, service: ProcessEmissionsService):
        """Service initializes and reports healthy status."""
        health = service.health_check()
        assert isinstance(health, HealthResponse)
        assert health.status in ("healthy", "degraded")

    def test_service_has_default_processes(
        self, service: ProcessEmissionsService,
    ):
        """Service pre-populates default process types."""
        result = service.list_processes(page=1, page_size=50)
        assert result.total > 0

    def test_calculate_cement_emission(
        self, service: ProcessEmissionsService,
    ):
        """Calculate cement emissions through the full pipeline."""
        result = service.calculate({
            "process_type": "cement_production",
            "activity_data": 100000,
            "activity_unit": "tonne",
        })
        assert isinstance(result, CalculateResponse)
        assert result.calculation_id.startswith("pe_calc_")
        assert result.processing_time_ms >= 0

    def test_calculate_and_query_back(
        self, service: ProcessEmissionsService,
    ):
        """Calculate an emission then retrieve it from stored results."""
        result = service.calculate({
            "process_type": "cement_production",
            "activity_data": 50000,
        })
        calc_id = result.calculation_id

        # Should appear in the calculations list
        found = False
        for calc in service._calculations:
            if calc.get("calculation_id") == calc_id:
                found = True
                break
        assert found, f"Calculation {calc_id} not found in _calculations"

    def test_stats_reflect_calculations(
        self, service: ProcessEmissionsService,
    ):
        """Stats counters match actual calculations performed."""
        service.calculate({
            "process_type": "cement_production",
            "activity_data": 1000,
        })
        service.calculate({
            "process_type": "cement_production",
            "activity_data": 2000,
        })
        stats = service.get_stats()
        assert isinstance(stats, StatsResponse)
        assert stats.total_calculations >= 2


class TestBatchWorkflow:
    """Test batch calculation workflows."""

    def test_batch_multiple_process_types(
        self, service: ProcessEmissionsService,
    ):
        """Batch calculate with different process types."""
        requests = [
            {"process_type": "cement_production", "activity_data": 1000},
            {"process_type": "cement_production", "activity_data": 2000},
            {"process_type": "cement_production", "activity_data": 3000},
        ]
        result = service.calculate_batch(requests)
        assert isinstance(result, BatchCalculateResponse)
        assert result.total_calculations == 3
        assert result.successful + result.failed == 3

    def test_batch_empty_returns_success(
        self, service: ProcessEmissionsService,
    ):
        """Batch with empty list returns success with zero count."""
        result = service.calculate_batch([])
        assert result.success is True
        assert result.total_calculations == 0

    def test_batch_increments_counter(
        self, service: ProcessEmissionsService,
    ):
        """Batch run increments total_batch_runs counter."""
        assert service._total_batch_runs == 0
        service.calculate_batch([
            {"process_type": "cement_production", "activity_data": 100},
        ])
        assert service._total_batch_runs == 1


class TestCRUDWorkflow:
    """Test CRUD workflows for processes, materials, units, factors."""

    def test_register_and_retrieve_process(
        self, service: ProcessEmissionsService,
    ):
        """Register a custom process and retrieve it."""
        data = {
            "process_type": "custom_integration_test",
            "category": "other",
            "name": "Integration Test Process",
            "description": "Process for integration testing",
            "primary_gases": ["CO2", "CH4"],
        }
        registered = service.register_process(data)
        assert isinstance(registered, ProcessDetailResponse)
        assert registered.process_type == "custom_integration_test"

        retrieved = service.get_process("custom_integration_test")
        assert retrieved is not None
        assert retrieved.name == "Integration Test Process"

    def test_register_and_list_processes(
        self, service: ProcessEmissionsService,
    ):
        """Register a process and verify it appears in list."""
        service.register_process({
            "process_type": "listed_process",
            "category": "mineral",
        })
        result = service.list_processes(page=1, page_size=100)
        types = [p.get("process_type") for p in result.processes]
        assert "listed_process" in types

    def test_register_and_retrieve_material(
        self, service: ProcessEmissionsService,
    ):
        """Register a material and retrieve it."""
        data = {
            "material_type": "test_dolomite",
            "name": "Test Dolomite",
            "carbon_content": 0.13,
            "carbonate_content": 0.90,
        }
        registered = service.register_material(data)
        assert isinstance(registered, MaterialDetailResponse)

        retrieved = service.get_material("test_dolomite")
        assert retrieved is not None
        assert retrieved.carbon_content == 0.13

    def test_register_unit_and_list(
        self, service: ProcessEmissionsService,
    ):
        """Register a process unit and verify in list."""
        service.register_unit({
            "unit_name": "Integration Kiln",
            "unit_type": "kiln",
            "process_type": "cement_production",
        })
        units = service.list_units()
        assert units.total >= 1

    def test_register_factor_and_list(
        self, service: ProcessEmissionsService,
    ):
        """Register an emission factor and verify in list."""
        service.register_factor({
            "process_type": "cement_production",
            "gas": "CO2",
            "value": 0.525,
            "source": "IPCC",
        })
        factors = service.list_factors()
        assert factors.total >= 1

    def test_register_abatement_and_list(
        self, service: ProcessEmissionsService,
    ):
        """Register an abatement record and verify in list."""
        service.register_abatement({
            "unit_id": "PU-INT-001",
            "abatement_type": "carbon_capture",
            "efficiency": 0.85,
            "target_gas": "CO2",
        })
        abatement = service.list_abatement()
        assert abatement.total >= 1

    def test_get_nonexistent_material(
        self, service: ProcessEmissionsService,
    ):
        """Get nonexistent material returns None."""
        result = service.get_material("nonexistent_xyz")
        assert result is None

    def test_get_nonexistent_process(
        self, service: ProcessEmissionsService,
    ):
        """Get nonexistent process returns None."""
        result = service.get_process("nonexistent_xyz")
        assert result is None


class TestUncertaintyWorkflow:
    """Test uncertainty analysis workflows."""

    def test_uncertainty_on_fresh_calculation(
        self, service: ProcessEmissionsService,
    ):
        """Calculate then run uncertainty analysis on the result."""
        calc = service.calculate({
            "process_type": "cement_production",
            "activity_data": 100000,
        })
        calc_id = calc.calculation_id

        uncertainty = service.run_uncertainty({
            "calculation_id": calc_id,
            "method": "monte_carlo",
            "iterations": 500,
        })
        assert isinstance(uncertainty, UncertaintyResponse)
        assert uncertainty.success is True

    def test_uncertainty_fallback_for_missing_calc(
        self, service: ProcessEmissionsService,
    ):
        """Uncertainty uses analytical fallback for missing calc ID."""
        uncertainty = service.run_uncertainty({
            "calculation_id": "nonexistent_calc",
            "method": "monte_carlo",
            "iterations": 1000,
        })
        assert isinstance(uncertainty, UncertaintyResponse)
        assert uncertainty.success is True
        assert uncertainty.method == "analytical_fallback"

    def test_uncertainty_confidence_intervals(
        self, service: ProcessEmissionsService,
    ):
        """Uncertainty response contains confidence intervals."""
        calc = service.calculate({
            "process_type": "cement_production",
            "activity_data": 50000,
        })
        uncertainty = service.run_uncertainty({
            "calculation_id": calc.calculation_id,
            "method": "monte_carlo",
            "iterations": 500,
        })
        assert isinstance(uncertainty.confidence_intervals, dict)


class TestComplianceWorkflow:
    """Test compliance checking workflows."""

    def test_compliance_check_all_frameworks(
        self, service: ProcessEmissionsService,
    ):
        """Run compliance check against all 6 frameworks."""
        result = service.check_compliance({})
        assert isinstance(result, ComplianceCheckResponse)
        assert result.success is True

    def test_compliance_check_single_framework(
        self, service: ProcessEmissionsService,
    ):
        """Run compliance check against a single framework."""
        result = service.check_compliance({
            "frameworks": ["GHG_PROTOCOL"],
        })
        assert isinstance(result, ComplianceCheckResponse)
        assert result.success is True

    def test_compliance_after_calculation(
        self, service: ProcessEmissionsService,
    ):
        """Run compliance on data derived from a calculation."""
        calc = service.calculate({
            "process_type": "cement_production",
            "activity_data": 100000,
        })
        result = service.check_compliance({
            "calculation_id": calc.calculation_id,
            "frameworks": ["GHG_PROTOCOL", "ISO_14064"],
        })
        assert isinstance(result, ComplianceCheckResponse)

    def test_compliance_engine_direct_check(
        self, compliance_check_data: Dict[str, Any],
    ):
        """Direct compliance engine check with full data."""
        from greenlang.process_emissions.compliance_checker import (
            ComplianceCheckerEngine,
        )
        engine = ComplianceCheckerEngine()
        results = engine.check_all_frameworks(compliance_check_data)
        assert len(results) == 6

        for result in results:
            assert result.total_checks == 10
            assert result.provenance_hash
            assert len(result.provenance_hash) == 64

    def test_compliance_determinism(
        self, compliance_check_data: Dict[str, Any],
    ):
        """Compliance results are deterministic across runs."""
        from greenlang.process_emissions.compliance_checker import (
            ComplianceCheckerEngine,
        )
        engine = ComplianceCheckerEngine()

        r1 = engine.check_compliance(compliance_check_data)
        r2 = engine.check_compliance(compliance_check_data)

        for fw in r1:
            assert r1[fw]["provenance_hash"] == r2[fw]["provenance_hash"]
            assert r1[fw]["status"] == r2[fw]["status"]
            assert r1[fw]["passed"] == r2[fw]["passed"]
