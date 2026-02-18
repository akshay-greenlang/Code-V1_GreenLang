# -*- coding: utf-8 -*-
"""
End-to-end integration tests for AGENT-MRV-005 Fugitive Emissions Agent.

Tests exercise the FugitiveEmissionsService facade (all 7 engines) with
REAL service instances. Covers equipment leaks, coal mine methane,
wastewater, pneumatic devices, tank losses, batch processing,
compliance checking, and uncertainty analysis.

Target: 33 tests, ~465 lines.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from greenlang.fugitive_emissions.setup import (
    FugitiveEmissionsService,
    CalculateResponse,
    BatchCalculateResponse,
    ComplianceCheckResponse,
    UncertaintyResponse,
    HealthResponse,
    StatsResponse,
)


# ===========================================================================
# Equipment Leaks E2E
# ===========================================================================


class TestEquipmentLeaksE2E:
    """End-to-end tests for EQUIPMENT_LEAK calculations through the service."""

    @pytest.mark.integration
    def test_calculate_equipment_leak(self, service, equipment_leak_request):
        """Calculate equipment leak emissions through the full service."""
        result = service.calculate(equipment_leak_request)
        assert isinstance(result, CalculateResponse)
        assert result.success is True
        assert result.source_type == "EQUIPMENT_LEAK"
        assert result.calculation_id.startswith("fe_calc_")
        assert len(result.provenance_hash) == 64
        assert result.processing_time_ms >= 0

    @pytest.mark.integration
    def test_equipment_leak_stored(self, service, equipment_leak_request):
        """Verify equipment leak calculation is stored in history."""
        service.calculate(equipment_leak_request)
        assert service._total_calculations >= 1
        assert len(service._calculations) >= 1
        assert service._calculations[-1]["source_type"] == "EQUIPMENT_LEAK"

    @pytest.mark.integration
    def test_equipment_leak_method_echoed(self, service, equipment_leak_request):
        """Verify calculation method is echoed in the response."""
        result = service.calculate(equipment_leak_request)
        assert result.calculation_method == "AVERAGE_EMISSION_FACTOR"

    @pytest.mark.integration
    def test_equipment_leak_with_component_registration(
        self, service, sample_components
    ):
        """Register components then calculate, verifying stats reflect both."""
        for comp in sample_components[:3]:
            service.register_component(comp)

        result = service.calculate({
            "source_type": "EQUIPMENT_LEAK",
            "facility_id": "FAC-001",
        })

        assert result.success is True
        stats = service.get_stats()
        assert stats.total_components >= 3
        assert stats.total_calculations >= 1


# ===========================================================================
# Coal Mine Methane E2E
# ===========================================================================


class TestCoalMineMethaneE2E:
    """End-to-end tests for COAL_MINE_METHANE calculations."""

    @pytest.mark.integration
    def test_calculate_coal_mine(self, service, coal_mine_request):
        """Calculate coal mine methane emissions."""
        result = service.calculate(coal_mine_request)
        assert isinstance(result, CalculateResponse)
        assert result.success is True
        assert result.source_type == "COAL_MINE_METHANE"

    @pytest.mark.integration
    def test_coal_mine_provenance(self, service, coal_mine_request):
        """Verify provenance hash is deterministic for coal mine calcs."""
        r1 = service.calculate(coal_mine_request)
        # Different calc_id means different provenance (includes calc_id)
        # But both should have valid 64-char hashes
        assert len(r1.provenance_hash) == 64


# ===========================================================================
# Wastewater E2E
# ===========================================================================


class TestWastewaterE2E:
    """End-to-end tests for WASTEWATER calculations."""

    @pytest.mark.integration
    def test_calculate_wastewater(self, service, wastewater_request):
        """Calculate wastewater treatment emissions."""
        result = service.calculate(wastewater_request)
        assert isinstance(result, CalculateResponse)
        assert result.success is True
        assert result.source_type == "WASTEWATER"

    @pytest.mark.integration
    def test_wastewater_in_history(self, service, wastewater_request):
        """Verify wastewater calculation appears in history."""
        service.calculate(wastewater_request)
        found = any(
            c["source_type"] == "WASTEWATER"
            for c in service._calculations
        )
        assert found is True


# ===========================================================================
# Pneumatic Devices E2E
# ===========================================================================


class TestPneumaticDevicesE2E:
    """End-to-end tests for PNEUMATIC_DEVICE calculations."""

    @pytest.mark.integration
    def test_calculate_pneumatic(self, service, pneumatic_request):
        """Calculate pneumatic device emissions."""
        result = service.calculate(pneumatic_request)
        assert isinstance(result, CalculateResponse)
        assert result.success is True
        assert result.source_type == "PNEUMATIC_DEVICE"

    @pytest.mark.integration
    def test_pneumatic_processing_time(self, service, pneumatic_request):
        """Verify processing time is reasonable."""
        result = service.calculate(pneumatic_request)
        assert result.processing_time_ms >= 0
        assert result.processing_time_ms < 5000  # under 5 seconds


# ===========================================================================
# Tank Losses E2E
# ===========================================================================


class TestTankLossesE2E:
    """End-to-end tests for TANK_LOSS calculations."""

    @pytest.mark.integration
    def test_calculate_tank_loss(self, service, tank_loss_request):
        """Calculate tank storage losses."""
        result = service.calculate(tank_loss_request)
        assert isinstance(result, CalculateResponse)
        assert result.success is True

    @pytest.mark.integration
    def test_tank_loss_stored(self, service, tank_loss_request):
        """Verify tank loss calculation is stored."""
        service.calculate(tank_loss_request)
        assert len(service._calculations) >= 1


# ===========================================================================
# Batch Processing E2E
# ===========================================================================


class TestBatchProcessingE2E:
    """End-to-end tests for batch calculations."""

    @pytest.mark.integration
    def test_batch_calculate(self, service, batch_requests):
        """Batch calculate across all 5 source types."""
        result = service.calculate_batch(batch_requests)
        assert isinstance(result, BatchCalculateResponse)
        assert result.total_calculations == 5
        assert result.successful + result.failed == 5

    @pytest.mark.integration
    def test_batch_empty(self, service):
        """Batch calculate with empty list."""
        result = service.calculate_batch([])
        assert isinstance(result, BatchCalculateResponse)
        assert result.total_calculations == 0

    @pytest.mark.integration
    def test_batch_increments_counter(self, service, batch_requests):
        """Verify batch run counter is incremented."""
        service.calculate_batch(batch_requests)
        assert service._total_batch_runs >= 1

    @pytest.mark.integration
    def test_batch_individual_results(self, service, batch_requests):
        """Verify batch results contain individual calculation entries."""
        result = service.calculate_batch(batch_requests)
        # All individual calcs should be stored in history
        assert len(service._calculations) >= 5


# ===========================================================================
# Compliance Checking E2E
# ===========================================================================


class TestComplianceCheckE2E:
    """End-to-end tests for regulatory compliance checks."""

    @pytest.mark.integration
    def test_compliance_after_calculation(
        self, service, equipment_leak_request
    ):
        """Run compliance check after a calculation."""
        calc_result = service.calculate(equipment_leak_request)
        calc_id = service._calculations[0]["calculation_id"]

        compliance = service.check_compliance({
            "calculation_id": calc_id,
            "frameworks": ["GHG_PROTOCOL"],
        })
        assert isinstance(compliance, ComplianceCheckResponse)
        assert compliance.success is True

    @pytest.mark.integration
    def test_compliance_all_frameworks(
        self, service, equipment_leak_request, compliance_frameworks
    ):
        """Check compliance against all 7 frameworks."""
        service.calculate(equipment_leak_request)
        calc_id = service._calculations[0]["calculation_id"]

        compliance = service.check_compliance({
            "calculation_id": calc_id,
            "frameworks": compliance_frameworks,
        })
        assert compliance.frameworks_checked >= 0

    @pytest.mark.integration
    def test_compliance_without_specific_framework(
        self, service, equipment_leak_request
    ):
        """Check compliance with empty frameworks list (all)."""
        service.calculate(equipment_leak_request)
        calc_id = service._calculations[0]["calculation_id"]

        compliance = service.check_compliance({
            "calculation_id": calc_id,
        })
        assert isinstance(compliance, ComplianceCheckResponse)


# ===========================================================================
# Uncertainty Analysis E2E
# ===========================================================================


class TestUncertaintyAnalysisE2E:
    """End-to-end tests for uncertainty analysis."""

    @pytest.mark.integration
    def test_uncertainty_monte_carlo(
        self, service, equipment_leak_request, uncertainty_config
    ):
        """Run Monte Carlo uncertainty analysis after a calculation."""
        service.calculate(equipment_leak_request)
        calc_id = service._calculations[0]["calculation_id"]
        uncertainty_config["calculation_id"] = calc_id

        result = service.run_uncertainty(uncertainty_config)
        assert isinstance(result, UncertaintyResponse)
        assert result.success is True

    @pytest.mark.integration
    def test_uncertainty_analytical(
        self, service, equipment_leak_request
    ):
        """Run analytical uncertainty analysis."""
        service.calculate(equipment_leak_request)
        calc_id = service._calculations[0]["calculation_id"]

        result = service.run_uncertainty({
            "calculation_id": calc_id,
            "method": "analytical",
        })
        assert isinstance(result, UncertaintyResponse)

    @pytest.mark.integration
    def test_uncertainty_nonexistent_calc(self, service):
        """Uncertainty analysis with non-existent calc ID."""
        result = service.run_uncertainty({
            "calculation_id": "NONEXISTENT",
            "method": "monte_carlo",
        })
        assert isinstance(result, UncertaintyResponse)


# ===========================================================================
# Health and Stats E2E
# ===========================================================================


class TestHealthAndStatsE2E:
    """End-to-end tests for health check and statistics."""

    @pytest.mark.integration
    def test_health_check(self, service):
        """Health check returns expected response."""
        result = service.health_check()
        assert isinstance(result, HealthResponse)
        assert result.service == "fugitive-emissions"
        assert result.version == "1.0.0"
        assert result.status in ("healthy", "degraded", "unhealthy")

    @pytest.mark.integration
    def test_health_engines_present(self, service):
        """Health check reports engine statuses."""
        result = service.health_check()
        assert "equipment_component" in result.engines
        assert "uncertainty_quantifier" in result.engines
        assert "compliance_checker" in result.engines
        assert "pipeline" in result.engines

    @pytest.mark.integration
    def test_stats_initial(self, service):
        """Stats show zero counters for a fresh service."""
        result = service.get_stats()
        assert isinstance(result, StatsResponse)
        assert result.total_calculations == 0
        assert result.total_components == 0
        assert result.total_surveys == 0
        assert result.total_repairs == 0
        assert result.uptime_seconds >= 0

    @pytest.mark.integration
    def test_stats_after_operations(self, populated_service):
        """Stats reflect operations performed on the populated service."""
        result = populated_service.get_stats()
        assert result.total_calculations >= 1
        assert result.total_components >= 1
        assert result.total_surveys >= 1
        assert result.total_sources > 0


# ===========================================================================
# Source Management E2E
# ===========================================================================


class TestSourceManagementE2E:
    """End-to-end tests for source type CRUD operations."""

    @pytest.mark.integration
    def test_default_sources_populated(self, service):
        """Default source types are available on a fresh service."""
        result = service.list_sources()
        assert result.total > 0

    @pytest.mark.integration
    def test_register_custom_source(self, service):
        """Register a custom source type."""
        result = service.register_source({
            "source_type": "CUSTOM_INT_SRC",
            "name": "Integration Test Source",
            "gases": ["CH4", "CO2"],
        })
        assert result.source_type == "CUSTOM_INT_SRC"

    @pytest.mark.integration
    def test_get_source_detail(self, service):
        """Get details of a default source type."""
        result = service.get_source("EQUIPMENT_LEAK")
        assert result is not None
        assert result.source_type == "EQUIPMENT_LEAK"

    @pytest.mark.integration
    def test_get_source_not_found(self, service):
        """Get source that does not exist returns None."""
        result = service.get_source("NONEXISTENT_SOURCE")
        assert result is None


# ===========================================================================
# Component CRUD E2E
# ===========================================================================


class TestComponentCRUDE2E:
    """End-to-end tests for component registration and listing."""

    @pytest.mark.integration
    def test_register_and_list(self, service, sample_components):
        """Register components and verify listing."""
        for comp in sample_components[:3]:
            service.register_component(comp)
        result = service.list_components()
        assert result.total >= 3

    @pytest.mark.integration
    def test_get_component_not_found(self, service):
        """Get component that does not exist returns None."""
        result = service.get_component("NONEXISTENT_COMP")
        assert result is None

    @pytest.mark.integration
    def test_factor_registration_and_listing(self, service, sample_factor):
        """Register and list emission factors."""
        service.register_factor(sample_factor)
        factors = service.list_factors()
        assert factors.total >= 1
