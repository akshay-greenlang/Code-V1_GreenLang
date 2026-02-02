# -*- coding: utf-8 -*-
"""
Scope3ReportingAgent Tests
GL-VCCI Scope 3 Platform

Comprehensive test suite for the reporting agent.

Version: 1.0.0
Phase: 3 (Weeks 16-18)
Date: 2025-10-30
"""

import pytest
import json
from pathlib import Path
from datetime import datetime

from services.agents.reporting import (
    Scope3ReportingAgent,
    CompanyInfo,
    EmissionsData,
    EnergyData,
    IntensityMetrics,
    TransportData,
    ReportStandard,
    ExportFormat,
    ValidationLevel,
)
from services.agents.reporting.exceptions import ReportingError, ValidationError


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_company_info():
    """Sample company information."""
    return CompanyInfo(
        name="Test Corp",
        reporting_year=2024,
        headquarters="New York, USA",
        number_of_employees=1000,
        annual_revenue_usd=100_000_000,
        industry_sector="Technology",
    )


@pytest.fixture
def sample_emissions_data():
    """Sample emissions data."""
    return EmissionsData(
        scope1_tco2e=1234.5,
        scope2_location_tco2e=2345.6,
        scope2_market_tco2e=1890.3,
        scope3_tco2e=20000.0,
        scope3_categories={1: 15000.0, 4: 3000.0, 6: 2000.0},
        avg_dqi_score=85.5,
        data_quality_by_scope={"Scope 1": 92.0, "Scope 2": 95.0, "Scope 3": 80.0},
        reporting_period_start=datetime(2024, 1, 1),
        reporting_period_end=datetime(2024, 12, 31),
        prior_year_emissions={"scope1_tco2e": 1300.0, "scope2_tco2e": 2500.0, "scope3_tco2e": 19000.0, "total_tco2e": 22800.0},
        yoy_change_pct=3.4,
    )


@pytest.fixture
def sample_energy_data():
    """Sample energy data."""
    return EnergyData(
        total_energy_mwh=10000.0,
        renewable_energy_mwh=3000.0,
        non_renewable_energy_mwh=7000.0,
        renewable_pct=30.0,
    )


@pytest.fixture
def sample_intensity_metrics():
    """Sample intensity metrics."""
    return IntensityMetrics(
        tco2e_per_million_usd=235.79,
        tco2e_per_fte=23.58,
    )


@pytest.fixture
def sample_transport_data():
    """Sample transport data."""
    return TransportData(
        transport_by_mode={
            "road": {"emissions_tco2e": 1500.0, "tonne_km": 50000},
            "sea": {"emissions_tco2e": 1200.0, "tonne_km": 80000},
            "air": {"emissions_tco2e": 300.0, "tonne_km": 5000},
        },
        total_tonne_km=135000,
        total_emissions_tco2e=3000.0,
        emission_factors_used=[
            {"mode": "road", "factor": 0.030, "source": "DEFRA 2024"},
            {"mode": "sea", "factor": 0.015, "source": "GLEC 2024"},
        ],
        data_quality_score=88.0,
        methodology="ISO 14083:2023",
    )


@pytest.fixture
def reporting_agent():
    """Create reporting agent instance."""
    return Scope3ReportingAgent()


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

def test_agent_initialization():
    """Test agent initialization."""
    agent = Scope3ReportingAgent()
    assert agent is not None
    assert agent.validator is not None
    assert agent.chart_generator is not None
    assert agent.esrs_generator is not None


def test_agent_custom_config():
    """Test agent with custom configuration."""
    config = {"validation_level": ValidationLevel.STRICT}
    agent = Scope3ReportingAgent(config=config)
    assert agent.validation_level == ValidationLevel.STRICT


# ============================================================================
# ESRS E1 TESTS
# ============================================================================

def test_generate_esrs_e1_report_json(reporting_agent, sample_company_info, sample_emissions_data):
    """Test ESRS E1 report generation (JSON)."""
    result = reporting_agent.generate_esrs_e1_report(
        emissions_data=sample_emissions_data,
        company_info=sample_company_info,
        export_format="json",
        output_path="test_esrs_e1.json",
    )

    assert result.success is True
    assert result.metadata.standard == ReportStandard.ESRS_E1
    assert result.metadata.export_format == ExportFormat.JSON
    assert result.file_path is not None
    assert len(result.sections_generated) > 0

    # Cleanup
    Path(result.file_path).unlink(missing_ok=True)


def test_generate_esrs_e1_with_energy(reporting_agent, sample_company_info, sample_emissions_data, sample_energy_data):
    """Test ESRS E1 report with energy data."""
    result = reporting_agent.generate_esrs_e1_report(
        emissions_data=sample_emissions_data,
        company_info=sample_company_info,
        energy_data=sample_energy_data,
        export_format="json",
    )

    assert result.success is True
    assert result.tables_count >= 2  # GHG + Energy tables

    # Cleanup
    Path(result.file_path).unlink(missing_ok=True)


def test_generate_esrs_e1_with_intensity(reporting_agent, sample_company_info, sample_emissions_data, sample_intensity_metrics):
    """Test ESRS E1 report with intensity metrics."""
    result = reporting_agent.generate_esrs_e1_report(
        emissions_data=sample_emissions_data,
        company_info=sample_company_info,
        intensity_metrics=sample_intensity_metrics,
        export_format="json",
    )

    assert result.success is True
    assert "intensity_metrics" in result.content

    # Cleanup
    Path(result.file_path).unlink(missing_ok=True)


# ============================================================================
# CDP TESTS
# ============================================================================

def test_generate_cdp_report(reporting_agent, sample_company_info, sample_emissions_data):
    """Test CDP questionnaire generation."""
    result = reporting_agent.generate_cdp_report(
        emissions_data=sample_emissions_data,
        company_info=sample_company_info,
        export_format="json",
    )

    assert result.success is True
    assert result.metadata.standard == ReportStandard.CDP
    assert "C6" in result.content
    assert result.content.get("auto_population_rate", 0) > 0.7

    # Cleanup
    Path(result.file_path).unlink(missing_ok=True)


# ============================================================================
# IFRS S2 TESTS
# ============================================================================

def test_generate_ifrs_s2_report(reporting_agent, sample_company_info, sample_emissions_data):
    """Test IFRS S2 report generation."""
    result = reporting_agent.generate_ifrs_s2_report(
        emissions_data=sample_emissions_data,
        company_info=sample_company_info,
        export_format="json",
    )

    assert result.success is True
    assert result.metadata.standard == ReportStandard.IFRS_S2
    assert "pillars" in result.content

    # Cleanup
    Path(result.file_path).unlink(missing_ok=True)


# ============================================================================
# ISO 14083 TESTS
# ============================================================================

def test_generate_iso_14083_certificate(reporting_agent, sample_transport_data):
    """Test ISO 14083 certificate generation."""
    result = reporting_agent.generate_iso_14083_certificate(
        transport_data=sample_transport_data,
    )

    assert result.success is True
    assert result.metadata.standard == ReportStandard.ISO_14083
    assert "certificate_id" in result.content
    assert result.content["standard"] == "ISO 14083:2023"

    # Cleanup
    Path(result.file_path).unlink(missing_ok=True)


# ============================================================================
# VALIDATION TESTS
# ============================================================================

def test_validate_readiness_esrs(reporting_agent, sample_emissions_data, sample_company_info):
    """Test validation for ESRS E1."""
    validation_result = reporting_agent.validate_readiness(
        emissions_data=sample_emissions_data,
        standard="esrs_e1",
        company_info=sample_company_info,
    )

    assert validation_result is not None
    assert validation_result.standard == ReportStandard.ESRS_E1
    assert validation_result.passed_checks > 0


def test_validate_readiness_cdp(reporting_agent, sample_emissions_data):
    """Test validation for CDP."""
    validation_result = reporting_agent.validate_readiness(
        emissions_data=sample_emissions_data,
        standard="cdp",
    )

    assert validation_result is not None
    assert validation_result.standard == ReportStandard.CDP


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_missing_required_data():
    """Test error when required data is missing."""
    agent = Scope3ReportingAgent(config={"validation_level": ValidationLevel.STRICT})

    # Incomplete emissions data
    incomplete_data = EmissionsData(
        scope1_tco2e=100.0,
        scope2_location_tco2e=0.0,  # Missing
        scope2_market_tco2e=0.0,
        scope3_tco2e=0.0,  # Missing
        scope3_categories={},
        avg_dqi_score=50.0,
        reporting_period_start=datetime(2024, 1, 1),
        reporting_period_end=datetime(2024, 12, 31),
    )

    company_info = CompanyInfo(name="Test", reporting_year=2024)

    with pytest.raises((ValidationError, ReportingError)):
        agent.generate_esrs_e1_report(
            emissions_data=incomplete_data,
            company_info=company_info,
        )


# ============================================================================
# CHART GENERATION TESTS
# ============================================================================

def test_chart_generation(reporting_agent, sample_emissions_data):
    """Test that charts are generated."""
    company_info = CompanyInfo(name="Test", reporting_year=2024)

    result = reporting_agent.generate_esrs_e1_report(
        emissions_data=sample_emissions_data,
        company_info=company_info,
        export_format="json",
    )

    assert result.charts_count > 0
    assert len(result.charts) > 0

    # Cleanup
    Path(result.file_path).unlink(missing_ok=True)
    for chart in result.charts:
        if chart.image_path:
            Path(chart.image_path).unlink(missing_ok=True)


# ============================================================================
# EXPORT FORMAT TESTS
# ============================================================================

@pytest.mark.parametrize("export_format", ["json", "excel"])
def test_export_formats(reporting_agent, sample_company_info, sample_emissions_data, export_format):
    """Test different export formats."""
    result = reporting_agent.generate_esrs_e1_report(
        emissions_data=sample_emissions_data,
        company_info=sample_company_info,
        export_format=export_format,
    )

    assert result.success is True
    assert Path(result.file_path).exists()

    # Cleanup
    Path(result.file_path).unlink(missing_ok=True)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_full_workflow(reporting_agent, sample_company_info, sample_emissions_data, sample_energy_data, sample_intensity_metrics):
    """Test full workflow: validate, generate, export."""
    # Step 1: Validate
    validation = reporting_agent.validate_readiness(
        emissions_data=sample_emissions_data,
        standard="esrs_e1",
        company_info=sample_company_info,
        energy_data=sample_energy_data,
    )

    assert validation.is_valid is True

    # Step 2: Generate report
    result = reporting_agent.generate_esrs_e1_report(
        emissions_data=sample_emissions_data,
        company_info=sample_company_info,
        energy_data=sample_energy_data,
        intensity_metrics=sample_intensity_metrics,
        export_format="json",
    )

    assert result.success is True
    assert result.validation_result is not None
    assert Path(result.file_path).exists()

    # Cleanup
    Path(result.file_path).unlink(missing_ok=True)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

def test_report_generation_performance(reporting_agent, sample_company_info, sample_emissions_data):
    """Test report generation performance."""
    import time

    start = time.time()

    result = reporting_agent.generate_esrs_e1_report(
        emissions_data=sample_emissions_data,
        company_info=sample_company_info,
        export_format="json",
    )

    duration = time.time() - start

    assert result.success is True
    assert duration < 10.0  # Should complete in under 10 seconds

    # Cleanup
    Path(result.file_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
