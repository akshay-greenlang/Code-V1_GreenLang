# -*- coding: utf-8 -*-
"""
Unit tests for ClimateIntelligenceReporter.

Tests emissions calculations, GHG Protocol compliance, and uncertainty.

Author: GL-TestEngineer
Date: December 2025
"""

import pytest
from datetime import datetime, timezone
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reporting.climate_reporter import (
    ClimateIntelligenceReporter,
    ReporterConfig,
    FuelType,
    ScopeClassification,
    ReportingPeriod,
    ComplianceFramework,
)


@dataclass
class MockDiagnostic:
    """Mock diagnostic result for testing."""
    trap_id: str
    condition: str
    energy_loss_kw: float
    location: str = ""
    system: str = ""


class TestClimateIntelligenceReporter:
    """Tests for ClimateIntelligenceReporter class."""

    @pytest.fixture
    def reporter(self):
        """Create default reporter."""
        return ClimateIntelligenceReporter()

    @pytest.fixture
    def natural_gas_config(self):
        """Create natural gas configuration."""
        return ReporterConfig(
            fuel_type=FuelType.NATURAL_GAS,
            boiler_efficiency=0.85,
            operating_hours_per_year=8760.0,
            carbon_price_usd_per_tonne=85.0,
        )

    @pytest.fixture
    def mock_diagnostics(self):
        """Create mock diagnostic results."""
        return [
            MockDiagnostic("ST-001", "failed", 50.0, "Building A"),
            MockDiagnostic("ST-002", "leaking", 25.0, "Building A"),
            MockDiagnostic("ST-003", "healthy", 0.0, "Building B"),
            MockDiagnostic("ST-004", "failed", 75.0, "Building C"),
        ]

    def test_reporter_initialization(self, reporter):
        """Test reporter initializes correctly."""
        assert reporter is not None
        assert reporter.config is not None

    def test_calculate_emissions_basic(self, reporter):
        """Test basic emissions calculation."""
        annual_mwh, annual_co2e_kg, annual_co2e_tonnes = reporter.calculate_emissions(
            energy_loss_kw=100.0
        )

        assert annual_mwh > 0
        assert annual_co2e_kg > 0
        assert annual_co2e_tonnes > 0
        assert annual_co2e_tonnes == annual_co2e_kg / 1000.0

    def test_calculate_emissions_zero_loss(self, reporter):
        """Test emissions with zero energy loss."""
        annual_mwh, annual_co2e_kg, annual_co2e_tonnes = reporter.calculate_emissions(
            energy_loss_kw=0.0
        )

        assert annual_mwh == 0.0
        assert annual_co2e_kg == 0.0
        assert annual_co2e_tonnes == 0.0

    def test_emissions_proportional_to_loss(self, reporter):
        """Test emissions scale proportionally with energy loss."""
        _, _, co2e_50kw = reporter.calculate_emissions(50.0)
        _, _, co2e_100kw = reporter.calculate_emissions(100.0)

        assert abs(co2e_100kw - 2 * co2e_50kw) < 0.001

    def test_uncertainty_bounds(self, reporter):
        """Test uncertainty calculation."""
        lower, upper, uncertainty_kg = reporter.calculate_uncertainty(100.0)

        assert lower < 100.0
        assert upper > 100.0
        assert lower >= 0
        assert uncertainty_kg > 0

    def test_uncertainty_disabled(self):
        """Test uncertainty when disabled."""
        config = ReporterConfig(include_uncertainty=False)
        reporter = ClimateIntelligenceReporter(config)

        lower, upper, uncertainty_kg = reporter.calculate_uncertainty(100.0)

        assert lower == 100.0
        assert upper == 100.0
        assert uncertainty_kg == 0.0

    def test_carbon_cost_calculation(self, reporter):
        """Test carbon cost calculation."""
        cost = reporter.calculate_carbon_cost(1.0)  # 1 tonne

        assert cost == reporter.config.carbon_price_usd_per_tonne

    def test_scope_classification_natural_gas(self, reporter):
        """Test scope classification for natural gas."""
        scope = reporter.classify_scope(FuelType.NATURAL_GAS)

        assert scope == ScopeClassification.SCOPE_1

    def test_scope_classification_electricity(self):
        """Test scope classification for electricity."""
        config = ReporterConfig(fuel_type=FuelType.ELECTRICITY)
        reporter = ClimateIntelligenceReporter(config)
        scope = reporter.classify_scope()

        assert scope == ScopeClassification.SCOPE_2

    def test_generate_report(self, reporter, mock_diagnostics):
        """Test report generation."""
        report = reporter.generate_report(mock_diagnostics)

        assert report is not None
        assert report.report_id is not None
        assert report.fleet_metrics is not None
        assert len(report.trap_records) == len(mock_diagnostics)

    def test_report_metrics(self, reporter, mock_diagnostics):
        """Test fleet metrics in report."""
        report = reporter.generate_report(mock_diagnostics)
        metrics = report.fleet_metrics

        assert metrics.total_traps == 4
        assert metrics.failed_traps == 2
        assert metrics.leaking_traps == 1
        assert metrics.total_energy_loss_kw == 150.0  # 50 + 25 + 0 + 75
        assert metrics.total_co2e_tonnes > 0

    def test_report_compliance(self, reporter, mock_diagnostics):
        """Test compliance status in report."""
        report = reporter.generate_report(mock_diagnostics)

        assert report.compliance_status is not None
        assert "ghg_protocol" in report.compliance_status

    def test_report_provenance_hash(self, reporter, mock_diagnostics):
        """Test provenance hash in report."""
        report = reporter.generate_report(mock_diagnostics)

        assert report.provenance_hash is not None
        assert len(report.provenance_hash) == 16

    def test_report_to_dict(self, reporter, mock_diagnostics):
        """Test report serialization."""
        report = reporter.generate_report(mock_diagnostics)
        report_dict = report.to_dict()

        assert "report_id" in report_dict
        assert "fleet_metrics" in report_dict
        assert "trap_records" in report_dict
        assert "compliance_status" in report_dict

    def test_report_to_json(self, reporter, mock_diagnostics):
        """Test JSON serialization."""
        report = reporter.generate_report(mock_diagnostics)
        json_str = report.to_json()

        assert isinstance(json_str, str)
        assert "report_id" in json_str

    def test_executive_summary(self, reporter, mock_diagnostics):
        """Test executive summary generation."""
        report = reporter.generate_report(mock_diagnostics)
        summary = reporter.generate_executive_summary(report)

        assert "EXECUTIVE SUMMARY" in summary
        assert "CO2e EMISSIONS" in summary
        assert "FINANCIAL IMPACT" in summary

    def test_reduction_opportunities(self, reporter, mock_diagnostics):
        """Test reduction opportunity identification."""
        report = reporter.generate_report(mock_diagnostics)
        opportunities = reporter.get_reduction_opportunities(report, top_n=5)

        assert len(opportunities) <= 5
        if opportunities:
            assert "trap_id" in opportunities[0]
            assert "annual_co2e_tonnes" in opportunities[0]

    def test_deterministic_report(self, reporter, mock_diagnostics):
        """Test that same input produces same output."""
        report1 = reporter.generate_report(mock_diagnostics)
        report2 = reporter.generate_report(mock_diagnostics)

        assert report1.fleet_metrics.total_co2e_tonnes == report2.fleet_metrics.total_co2e_tonnes
        assert report1.provenance_hash == report2.provenance_hash


class TestReporterConfig:
    """Tests for ReporterConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = ReporterConfig()

        assert config.fuel_type == FuelType.NATURAL_GAS
        assert config.operating_hours_per_year == 8760.0
        assert config.carbon_price_usd_per_tonne == 85.0

    def test_get_emission_factor(self):
        """Test emission factor retrieval."""
        config = ReporterConfig(fuel_type=FuelType.NATURAL_GAS)
        factor = config.get_emission_factor()

        assert factor > 0
        assert factor < 1.0  # kg CO2e per kWh

    def test_emission_factor_override(self):
        """Test emission factor override."""
        config = ReporterConfig(emission_factor_override=0.5)
        factor = config.get_emission_factor()

        assert factor == 0.5

    def test_boiler_efficiency_override(self):
        """Test boiler efficiency override."""
        config = ReporterConfig(boiler_efficiency=0.90)
        efficiency = config.get_boiler_efficiency()

        assert efficiency == 0.90
