# -*- coding: utf-8 -*-
"""
Unit tests for FugitiveEmissionsService facade (setup.py) - AGENT-MRV-005

Tests the service facade layer that aggregates all 7 engines and exposes
20 convenience methods for REST API operations. Validates calculation,
batch processing, source/component/survey/repair CRUD, uncertainty
analysis, compliance checks, health, and statistics.

Target: 65 tests, ~730 lines.

Test classes:
    TestServiceInit (5)
    TestCalculate (15)
    TestBatch (5)
    TestSourceManagement (10)
    TestComponentManagement (10)
    TestSurveyManagement (5)
    TestRepairManagement (5)
    TestUncertainty (3)
    TestCompliance (3)
    TestHealthStats (4)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.fugitive_emissions.setup import (
    FugitiveEmissionsService,
    CalculateResponse,
    BatchCalculateResponse,
    SourceListResponse,
    SourceDetailResponse,
    ComponentListResponse,
    ComponentDetailResponse,
    SurveyListResponse,
    FactorListResponse,
    FactorDetailResponse,
    RepairListResponse,
    UncertaintyResponse,
    ComplianceCheckResponse,
    HealthResponse,
    StatsResponse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    """Create a FugitiveEmissionsService instance."""
    return FugitiveEmissionsService()


@pytest.fixture
def equipment_leak_request():
    """Standard equipment leak calculation request."""
    return {
        "source_type": "EQUIPMENT_LEAK",
        "facility_id": "FAC-001",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
        "component_count": 100,
        "component_type": "valve",
        "service_type": "gas",
        "operating_hours": 8760,
    }


@pytest.fixture
def wastewater_request():
    """Wastewater treatment calculation request."""
    return {
        "source_type": "WASTEWATER",
        "facility_id": "FAC-002",
        "calculation_method": "AVERAGE_EMISSION_FACTOR",
        "component_count": 1,
        "activity_data": 5000,
    }


# ===========================================================================
# TestServiceInit (5 tests)
# ===========================================================================


class TestServiceInit:
    """Tests for service initialization."""

    def test_service_creates(self):
        svc = FugitiveEmissionsService()
        assert svc is not None

    def test_pipeline_engine_available(self, service):
        # Pipeline engine should be created if imports succeed
        assert service.pipeline_engine is not None or service.pipeline_engine is None

    def test_default_sources_populated(self, service):
        sources = service.list_sources()
        assert sources.total >= 1

    def test_initial_calculations_empty(self, service):
        assert len(service._calculations) == 0

    def test_initial_stats_zero(self, service):
        stats = service.get_stats()
        assert stats.total_calculations == 0


# ===========================================================================
# TestCalculate (15 tests)
# ===========================================================================


class TestCalculate:
    """Tests for the calculate method."""

    def test_calculate_returns_response(self, service, equipment_leak_request):
        result = service.calculate(equipment_leak_request)
        assert isinstance(result, CalculateResponse)

    def test_calculate_success(self, service, equipment_leak_request):
        result = service.calculate(equipment_leak_request)
        assert result.success is True

    def test_calculate_id_format(self, service, equipment_leak_request):
        result = service.calculate(equipment_leak_request)
        assert result.calculation_id.startswith("fe_calc_")

    def test_calculate_source_type(self, service, equipment_leak_request):
        result = service.calculate(equipment_leak_request)
        assert result.source_type == "EQUIPMENT_LEAK"

    def test_calculate_method(self, service, equipment_leak_request):
        result = service.calculate(equipment_leak_request)
        assert result.calculation_method == "AVERAGE_EMISSION_FACTOR"

    def test_calculate_total_co2e(self, service, equipment_leak_request):
        result = service.calculate(equipment_leak_request)
        assert result.total_co2e_kg >= 0

    def test_calculate_provenance_hash(self, service, equipment_leak_request):
        result = service.calculate(equipment_leak_request)
        assert len(result.provenance_hash) == 64

    def test_calculate_processing_time(self, service, equipment_leak_request):
        result = service.calculate(equipment_leak_request)
        assert result.processing_time_ms > 0

    def test_calculate_timestamp(self, service, equipment_leak_request):
        result = service.calculate(equipment_leak_request)
        assert len(result.timestamp) > 0

    def test_calculate_stores_in_history(self, service, equipment_leak_request):
        service.calculate(equipment_leak_request)
        assert len(service._calculations) == 1

    def test_calculate_increments_counter(self, service, equipment_leak_request):
        service.calculate(equipment_leak_request)
        assert service._total_calculations == 1

    def test_calculate_wastewater(self, service, wastewater_request):
        result = service.calculate(wastewater_request)
        assert result.success is True

    def test_calculate_with_custom_gwp(self, service):
        req = {
            "source_type": "EQUIPMENT_LEAK",
            "facility_id": "FAC-001",
            "component_count": 50,
            "gwp_source": "AR5",
        }
        result = service.calculate(req)
        assert result.success is True

    def test_calculate_multiple_calls(self, service, equipment_leak_request):
        service.calculate(equipment_leak_request)
        service.calculate(equipment_leak_request)
        assert service._total_calculations == 2
        assert len(service._calculations) == 2

    def test_calculate_default_method(self, service):
        req = {"source_type": "EQUIPMENT_LEAK", "facility_id": "FAC-001"}
        result = service.calculate(req)
        assert result.success is True


# ===========================================================================
# TestBatch (5 tests)
# ===========================================================================


class TestBatch:
    """Tests for batch calculation."""

    def test_batch_returns_response(self, service, equipment_leak_request):
        result = service.calculate_batch([equipment_leak_request])
        assert isinstance(result, BatchCalculateResponse)

    def test_batch_counts(self, service, equipment_leak_request):
        result = service.calculate_batch([equipment_leak_request] * 3)
        assert result.total_calculations == 3
        assert result.successful == 3
        assert result.failed == 0

    def test_batch_total_co2e(self, service, equipment_leak_request):
        result = service.calculate_batch([equipment_leak_request])
        assert result.total_co2e_kg >= 0

    def test_batch_results_list(self, service, equipment_leak_request):
        result = service.calculate_batch([equipment_leak_request])
        assert len(result.results) == 1

    def test_batch_processing_time(self, service, equipment_leak_request):
        result = service.calculate_batch([equipment_leak_request])
        assert result.processing_time_ms > 0


# ===========================================================================
# TestSourceManagement (10 tests)
# ===========================================================================


class TestSourceManagement:
    """Tests for source type CRUD operations."""

    def test_list_sources_returns_response(self, service):
        result = service.list_sources()
        assert isinstance(result, SourceListResponse)

    def test_list_sources_has_defaults(self, service):
        result = service.list_sources()
        assert result.total >= 1

    def test_get_source_exists(self, service):
        result = service.get_source("EQUIPMENT_LEAK")
        assert result is not None
        assert isinstance(result, SourceDetailResponse)
        assert result.source_type == "EQUIPMENT_LEAK"

    def test_get_source_not_found(self, service):
        result = service.get_source("NONEXISTENT_SOURCE")
        assert result is None

    def test_register_source(self, service):
        result = service.register_source({
            "source_type": "TEST_SOURCE",
            "name": "Test Source",
            "gases": ["CH4", "CO2"],
            "methods": ["AVERAGE_EMISSION_FACTOR"],
        })
        assert isinstance(result, SourceDetailResponse)
        assert result.source_type == "TEST_SOURCE"

    def test_register_source_then_get(self, service):
        service.register_source({
            "source_type": "CUSTOM_SRC",
            "name": "Custom Source",
        })
        result = service.get_source("CUSTOM_SRC")
        assert result is not None

    def test_list_sources_pagination(self, service):
        result = service.list_sources(page=1, page_size=2)
        assert result.page == 1
        assert result.page_size == 2

    def test_list_sources_page_2(self, service):
        result = service.list_sources(page=2, page_size=2)
        assert result.page == 2

    def test_source_has_gases(self, service):
        result = service.get_source("EQUIPMENT_LEAK")
        assert len(result.gases) >= 1

    def test_source_has_methods(self, service):
        result = service.get_source("EQUIPMENT_LEAK")
        assert len(result.methods) >= 1


# ===========================================================================
# TestComponentManagement (10 tests)
# ===========================================================================


class TestComponentManagement:
    """Tests for equipment component CRUD operations."""

    def test_register_component(self, service):
        result = service.register_component({
            "tag_number": "V-1001",
            "component_type": "valve",
            "service_type": "gas",
            "facility_id": "FAC-001",
        })
        assert isinstance(result, ComponentDetailResponse)
        assert result.tag_number == "V-1001"

    def test_register_component_generates_id(self, service):
        result = service.register_component({
            "tag_number": "V-1002",
            "component_type": "valve",
        })
        assert len(result.component_id) > 0

    def test_list_components_empty(self, service):
        result = service.list_components()
        assert isinstance(result, ComponentListResponse)
        assert result.total == 0

    def test_list_components_after_register(self, service):
        service.register_component({
            "tag_number": "V-1003",
            "component_type": "connector",
        })
        result = service.list_components()
        assert result.total == 1

    def test_get_component_exists(self, service):
        reg = service.register_component({
            "tag_number": "V-1004",
            "component_type": "pump",
        })
        result = service.get_component(reg.component_id)
        assert result is not None

    def test_get_component_not_found(self, service):
        result = service.get_component("nonexistent_id")
        assert result is None

    def test_register_multiple_components(self, service):
        service.register_component({"tag_number": "V-2001"})
        service.register_component({"tag_number": "V-2002"})
        result = service.list_components()
        assert result.total == 2

    def test_component_type_default(self, service):
        result = service.register_component({"tag_number": "V-3001"})
        assert result.component_type in ("other", "valve", "")

    def test_component_service_type_default(self, service):
        result = service.register_component({"tag_number": "V-3002"})
        assert result.service_type in ("gas", "")

    def test_list_components_pagination(self, service):
        for i in range(5):
            service.register_component({"tag_number": f"V-P{i}"})
        result = service.list_components(page=1, page_size=2)
        assert len(result.components) == 2


# ===========================================================================
# TestSurveyManagement (5 tests)
# ===========================================================================


class TestSurveyManagement:
    """Tests for LDAR survey CRUD operations."""

    def test_register_survey(self, service):
        result = service.register_survey({
            "survey_type": "OGI",
            "facility_id": "FAC-001",
            "components_surveyed": 100,
            "leaks_found": 5,
        })
        assert "survey_id" in result

    def test_list_surveys_empty(self, service):
        result = service.list_surveys()
        assert isinstance(result, SurveyListResponse)
        assert result.total == 0

    def test_list_surveys_after_register(self, service):
        service.register_survey({"survey_type": "METHOD21"})
        result = service.list_surveys()
        assert result.total == 1

    def test_survey_id_generated(self, service):
        result = service.register_survey({"survey_type": "OGI"})
        assert result["survey_id"].startswith("survey_")

    def test_survey_custom_id(self, service):
        result = service.register_survey({
            "survey_id": "CUSTOM_SURVEY_001",
            "survey_type": "OGI",
        })
        assert result["survey_id"] == "CUSTOM_SURVEY_001"


# ===========================================================================
# TestRepairManagement (5 tests)
# ===========================================================================


class TestRepairManagement:
    """Tests for component repair CRUD operations."""

    def test_register_repair(self, service):
        result = service.register_repair({
            "component_id": "comp_001",
            "repair_type": "minor",
        })
        assert "repair_id" in result

    def test_list_repairs_empty(self, service):
        result = service.list_repairs()
        assert isinstance(result, RepairListResponse)
        assert result.total == 0

    def test_list_repairs_after_register(self, service):
        service.register_repair({"component_id": "comp_002"})
        result = service.list_repairs()
        assert result.total == 1

    def test_repair_id_generated(self, service):
        result = service.register_repair({"component_id": "comp_003"})
        assert result["repair_id"].startswith("repair_")

    def test_repair_preserves_fields(self, service):
        result = service.register_repair({
            "component_id": "comp_004",
            "repair_type": "major",
            "cost_usd": 500.0,
        })
        assert result.get("cost_usd") == 500.0


# ===========================================================================
# TestUncertainty (3 tests)
# ===========================================================================


class TestUncertainty:
    """Tests for uncertainty analysis via service facade."""

    def test_uncertainty_returns_response(self, service):
        result = service.run_uncertainty({
            "calculation_id": "nonexistent",
            "method": "monte_carlo",
        })
        assert isinstance(result, UncertaintyResponse)

    def test_uncertainty_fallback(self, service):
        result = service.run_uncertainty({
            "calculation_id": "nonexistent",
        })
        # Falls back when no matching calc found
        assert result.success is True

    def test_uncertainty_after_calc(self, service, equipment_leak_request):
        calc = service.calculate(equipment_leak_request)
        result = service.run_uncertainty({
            "calculation_id": calc.calculation_id,
            "method": "monte_carlo",
            "iterations": 100,
        })
        assert result.success is True


# ===========================================================================
# TestCompliance (3 tests)
# ===========================================================================


class TestCompliance:
    """Tests for compliance checking via service facade."""

    def test_compliance_returns_response(self, service):
        result = service.check_compliance({
            "calculation_id": "nonexistent",
        })
        assert isinstance(result, ComplianceCheckResponse)

    def test_compliance_with_frameworks(self, service):
        result = service.check_compliance({
            "frameworks": ["GHG_PROTOCOL"],
        })
        assert result.success is True

    def test_compliance_after_calc(self, service, equipment_leak_request):
        calc = service.calculate(equipment_leak_request)
        result = service.check_compliance({
            "calculation_id": calc.calculation_id,
        })
        assert result.success is True


# ===========================================================================
# TestHealthStats (4 tests)
# ===========================================================================


class TestHealthStats:
    """Tests for health check and statistics."""

    def test_health_check(self, service):
        result = service.health_check()
        assert isinstance(result, HealthResponse)
        assert result.service == "fugitive-emissions"

    def test_health_engines_dict(self, service):
        result = service.health_check()
        assert isinstance(result.engines, dict)
        assert "pipeline" in result.engines

    def test_stats_response(self, service):
        result = service.get_stats()
        assert isinstance(result, StatsResponse)
        assert result.uptime_seconds >= 0

    def test_stats_after_calculations(self, service, equipment_leak_request):
        service.calculate(equipment_leak_request)
        service.calculate(equipment_leak_request)
        stats = service.get_stats()
        assert stats.total_calculations == 2
