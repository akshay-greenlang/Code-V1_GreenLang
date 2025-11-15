"""
Unit tests for GL-002 Boiler Efficiency Tools

Tests all tool modules including optimization tools, diagnostic tools,
reporting tools, and utility functions. Validates input/output correctness
and provenance tracking.

Target: 30+ tests for comprehensive tool coverage
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Tuple
import json
import hashlib
from decimal import Decimal

# Import tool modules (adjust based on actual implementation)
from greenlang_boiler_efficiency.tools import (
    EfficiencyOptimizer,
    DiagnosticAnalyzer,
    ReportGenerator,
    ProvenanceTracker,
    DataValidator,
    TrendAnalyzer,
    AnomalyDetector,
    PerformancePredictor,
    MaintenanceScheduler,
    CostCalculator,
)
from greenlang_core.exceptions import (
    ValidationError,
    OptimizationError,
    AnalysisError,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def efficiency_optimizer():
    """Create EfficiencyOptimizer instance."""
    config = {
        "optimization_method": "gradient_descent",
        "max_iterations": 100,
        "tolerance": 1e-6,
        "constraints": {
            "min_excess_air": 1.05,
            "max_excess_air": 1.25,
            "min_efficiency": 0.80,
        },
    }
    return EfficiencyOptimizer(config)


@pytest.fixture
def diagnostic_analyzer():
    """Create DiagnosticAnalyzer instance."""
    return DiagnosticAnalyzer()


@pytest.fixture
def provenance_tracker():
    """Create ProvenanceTracker instance."""
    return ProvenanceTracker()


@pytest.fixture
def sample_operational_data():
    """Create sample operational data for testing."""
    return {
        "timestamp": datetime.now(),
        "fuel_flow": 100.0,
        "steam_flow": 1500.0,
        "efficiency": 0.85,
        "excess_air": 1.15,
        "stack_temperature": 150.0,
        "o2_percentage": 3.0,
        "co_ppm": 50,
    }


@pytest.fixture
def historical_data():
    """Create historical data for trend analysis."""
    base_time = datetime.now() - timedelta(days=30)
    data = []

    for i in range(720):  # 30 days of hourly data
        data.append({
            "timestamp": base_time + timedelta(hours=i),
            "efficiency": 0.85 + np.random.normal(0, 0.02),
            "fuel_flow": 100 + np.random.normal(0, 5),
            "steam_flow": 1500 + np.random.normal(0, 50),
        })

    return data


# ============================================================================
# TEST EFFICIENCY OPTIMIZER
# ============================================================================

class TestEfficiencyOptimizer:
    """Test efficiency optimization tool."""

    def test_optimize_excess_air(self, efficiency_optimizer, sample_operational_data):
        """Test excess air optimization."""
        result = efficiency_optimizer.optimize_excess_air(
            current_excess_air=sample_operational_data["excess_air"],
            o2_percentage=sample_operational_data["o2_percentage"],
            co_ppm=sample_operational_data["co_ppm"],
        )

        assert result["optimal_excess_air"] is not None
        assert 1.05 <= result["optimal_excess_air"] <= 1.25
        assert result["expected_efficiency_gain"] >= 0
        assert "recommendations" in result

    def test_multi_parameter_optimization(self, efficiency_optimizer):
        """Test optimization of multiple parameters simultaneously."""
        parameters = {
            "excess_air": 1.20,
            "feedwater_temperature": 80,
            "blowdown_rate": 0.04,
            "steam_pressure": 10.0,
        }

        result = efficiency_optimizer.optimize_multiple(parameters)

        assert "optimal_values" in result
        assert all(param in result["optimal_values"] for param in parameters)
        assert result["total_efficiency_gain"] >= 0
        assert result["convergence_achieved"] is True

    def test_constraint_satisfaction(self, efficiency_optimizer):
        """Test that optimization respects constraints."""
        constraints = {
            "min_efficiency": 0.82,
            "max_emissions": 100,  # kg/h CO2
            "min_steam_quality": 0.99,
        }

        result = efficiency_optimizer.optimize_with_constraints(
            initial_params={"excess_air": 1.30},  # Start outside optimal range
            constraints=constraints,
        )

        # Verify all constraints are satisfied
        assert result["final_efficiency"] >= constraints["min_efficiency"]
        assert result["emissions"] <= constraints["max_emissions"]
        assert result["steam_quality"] >= constraints["min_steam_quality"]

    def test_optimization_convergence(self, efficiency_optimizer):
        """Test that optimization converges properly."""
        result = efficiency_optimizer.optimize_excess_air(
            current_excess_air=1.50,  # Far from optimal
            o2_percentage=7.0,
            co_ppm=200,
        )

        assert result["iterations"] < 100  # Should converge before max iterations
        assert result["convergence_error"] < 1e-6
        assert result["optimization_successful"] is True

    def test_optimization_failure_handling(self, efficiency_optimizer):
        """Test handling of optimization failures."""
        with pytest.raises(OptimizationError):
            # Impossible constraints
            efficiency_optimizer.optimize_with_constraints(
                initial_params={"excess_air": 1.15},
                constraints={
                    "min_efficiency": 0.99,  # Unrealistic
                    "max_fuel_consumption": 10,  # Too low
                },
            )


# ============================================================================
# TEST DIAGNOSTIC ANALYZER
# ============================================================================

class TestDiagnosticAnalyzer:
    """Test diagnostic analysis tool."""

    def test_diagnose_efficiency_drop(self, diagnostic_analyzer, sample_operational_data):
        """Test diagnosis of efficiency drop."""
        current_data = sample_operational_data.copy()
        current_data["efficiency"] = 0.75  # Lower than normal

        baseline_data = sample_operational_data.copy()
        baseline_data["efficiency"] = 0.85

        diagnosis = diagnostic_analyzer.diagnose_efficiency_drop(
            current=current_data, baseline=baseline_data
        )

        assert "root_causes" in diagnosis
        assert len(diagnosis["root_causes"]) > 0
        assert "severity" in diagnosis
        assert diagnosis["severity"] in ["low", "medium", "high", "critical"]
        assert "corrective_actions" in diagnosis

    def test_combustion_analysis(self, diagnostic_analyzer):
        """Test combustion diagnostics."""
        combustion_data = {
            "o2_percentage": 5.0,  # High
            "co_ppm": 500,  # High
            "nox_ppm": 150,
            "smoke_number": 2,
        }

        analysis = diagnostic_analyzer.analyze_combustion(combustion_data)

        assert analysis["combustion_quality"] in ["poor", "fair", "good", "excellent"]
        assert "issues" in analysis
        assert "incomplete_combustion" in analysis["issues"]  # High CO indicates this
        assert "recommendations" in analysis

    def test_heat_loss_diagnostics(self, diagnostic_analyzer):
        """Test heat loss diagnostic analysis."""
        heat_loss_data = {
            "stack_loss": 15.0,  # %
            "radiation_loss": 3.0,  # %
            "convection_loss": 2.0,  # %
            "blowdown_loss": 4.0,  # %
        }

        diagnosis = diagnostic_analyzer.diagnose_heat_losses(heat_loss_data)

        assert "total_loss" in diagnosis
        assert diagnosis["total_loss"] == 24.0
        assert "primary_loss_source" in diagnosis
        assert diagnosis["primary_loss_source"] == "stack_loss"
        assert "improvement_potential" in diagnosis

    def test_component_health_assessment(self, diagnostic_analyzer):
        """Test component health assessment."""
        component_data = {
            "burner_pressure_drop": 150,  # mbar
            "air_preheater_effectiveness": 0.65,
            "economizer_effectiveness": 0.70,
            "feedwater_pump_efficiency": 0.75,
        }

        health = diagnostic_analyzer.assess_component_health(component_data)

        assert "overall_health_score" in health
        assert 0 <= health["overall_health_score"] <= 100
        assert "component_scores" in health
        assert "maintenance_recommendations" in health

    def test_performance_degradation_analysis(self, diagnostic_analyzer, historical_data):
        """Test analysis of performance degradation over time."""
        analysis = diagnostic_analyzer.analyze_degradation(historical_data)

        assert "degradation_rate" in analysis  # % per month
        assert "projected_maintenance_date" in analysis
        assert "confidence_interval" in analysis
        assert "contributing_factors" in analysis


# ============================================================================
# TEST REPORT GENERATOR
# ============================================================================

class TestReportGenerator:
    """Test report generation tool."""

    @pytest.fixture
    def report_generator(self):
        """Create ReportGenerator instance."""
        return ReportGenerator()

    def test_efficiency_report_generation(self, report_generator, sample_operational_data):
        """Test efficiency report generation."""
        report = report_generator.generate_efficiency_report(
            data=sample_operational_data,
            period="daily",
            format="json",
        )

        assert "summary" in report
        assert "efficiency" in report["summary"]
        assert "fuel_consumption" in report["summary"]
        assert "steam_production" in report["summary"]
        assert "recommendations" in report

    def test_compliance_report(self, report_generator):
        """Test compliance report generation."""
        compliance_data = {
            "emissions": {
                "nox": 45,  # ppm
                "co": 80,  # ppm
                "so2": 5,  # ppm
            },
            "limits": {
                "nox": 50,
                "co": 100,
                "so2": 10,
            },
            "standard": "EPA_NSPS",
        }

        report = report_generator.generate_compliance_report(compliance_data)

        assert "compliance_status" in report
        assert report["compliance_status"] == "COMPLIANT"
        assert "margin_to_limits" in report
        assert report["margin_to_limits"]["nox"] == 5  # 50-45

    def test_cost_analysis_report(self, report_generator):
        """Test cost analysis report generation."""
        cost_data = {
            "fuel_consumption": 2400,  # kg/day
            "fuel_cost": 0.35,  # $/kg
            "steam_production": 36000,  # kg/day
            "steam_value": 0.08,  # $/kg
        }

        report = report_generator.generate_cost_report(cost_data)

        assert "daily_fuel_cost" in report
        assert report["daily_fuel_cost"] == 840  # 2400 * 0.35
        assert "daily_steam_value" in report
        assert report["daily_steam_value"] == 2880  # 36000 * 0.08
        assert "cost_per_unit_steam" in report

    def test_multi_format_report_generation(self, report_generator, sample_operational_data):
        """Test report generation in multiple formats."""
        formats = ["json", "xml", "csv", "html", "pdf"]

        for fmt in formats:
            report = report_generator.generate_efficiency_report(
                data=sample_operational_data, format=fmt
            )

            if fmt == "json":
                assert isinstance(report, dict)
            elif fmt == "csv":
                assert "," in report  # CSV should have commas
            elif fmt == "html":
                assert "<html>" in report or "<table>" in report
            elif fmt == "pdf":
                assert report.startswith(b"%PDF")  # PDF magic number

    def test_custom_report_template(self, report_generator):
        """Test custom report template usage."""
        template = {
            "title": "Custom Boiler Report",
            "sections": ["efficiency", "emissions", "costs"],
            "charts": ["trend_line", "pie_chart"],
        }

        report = report_generator.generate_custom_report(
            data=sample_operational_data, template=template
        )

        assert report["title"] == "Custom Boiler Report"
        assert all(section in report for section in template["sections"])
        assert "charts" in report


# ============================================================================
# TEST PROVENANCE TRACKER
# ============================================================================

class TestProvenanceTracker:
    """Test provenance tracking tool."""

    def test_provenance_hash_generation(self, provenance_tracker):
        """Test generation of provenance hash."""
        data = {
            "input": {"fuel_flow": 100, "steam_flow": 1500},
            "calculation": "efficiency",
            "result": 0.85,
            "timestamp": datetime.now().isoformat(),
        }

        hash1 = provenance_tracker.generate_hash(data)
        hash2 = provenance_tracker.generate_hash(data)

        assert hash1 == hash2  # Same input -> same hash
        assert len(hash1) == 64  # SHA-256 hash
        assert all(c in "0123456789abcdef" for c in hash1)

    def test_provenance_chain_creation(self, provenance_tracker):
        """Test creation of provenance chain."""
        steps = [
            {"step": "input", "data": {"fuel_flow": 100}},
            {"step": "calculation", "method": "direct"},
            {"step": "output", "result": 0.85},
        ]

        chain = provenance_tracker.create_chain(steps)

        assert len(chain) == 3
        assert all("hash" in step for step in chain)
        assert all("parent_hash" in step for step in chain[1:])

        # Verify chain integrity
        for i in range(len(chain) - 1):
            assert chain[i + 1]["parent_hash"] == chain[i]["hash"]

    def test_provenance_verification(self, provenance_tracker):
        """Test provenance verification."""
        original_data = {"fuel_flow": 100, "efficiency": 0.85}
        hash_value = provenance_tracker.generate_hash(original_data)

        # Verify with same data
        is_valid = provenance_tracker.verify(original_data, hash_value)
        assert is_valid is True

        # Verify with modified data
        modified_data = {"fuel_flow": 101, "efficiency": 0.85}
        is_valid = provenance_tracker.verify(modified_data, hash_value)
        assert is_valid is False

    def test_audit_trail_generation(self, provenance_tracker):
        """Test audit trail generation."""
        operations = [
            {"operation": "read", "source": "SCADA", "timestamp": datetime.now()},
            {"operation": "calculate", "method": "efficiency", "timestamp": datetime.now()},
            {"operation": "write", "destination": "ERP", "timestamp": datetime.now()},
        ]

        trail = provenance_tracker.create_audit_trail(operations)

        assert "trail_id" in trail
        assert "operations" in trail
        assert len(trail["operations"]) == 3
        assert "signature" in trail  # Digital signature of trail

    def test_provenance_metadata_tracking(self, provenance_tracker):
        """Test tracking of provenance metadata."""
        metadata = {
            "agent_version": "2.0.0",
            "calculation_method": "indirect",
            "data_sources": ["SCADA", "DCS"],
            "user": "system",
        }

        provenance = provenance_tracker.create_provenance(
            data={"efficiency": 0.85}, metadata=metadata
        )

        assert "hash" in provenance
        assert "metadata" in provenance
        assert provenance["metadata"]["agent_version"] == "2.0.0"
        assert len(provenance["metadata"]["data_sources"]) == 2


# ============================================================================
# TEST DATA VALIDATOR
# ============================================================================

class TestDataValidator:
    """Test data validation tool."""

    @pytest.fixture
    def data_validator(self):
        """Create DataValidator instance."""
        return DataValidator()

    def test_range_validation(self, data_validator):
        """Test value range validation."""
        rules = {
            "fuel_flow": {"min": 0, "max": 1000},
            "efficiency": {"min": 0, "max": 1},
            "temperature": {"min": -50, "max": 500},
        }

        valid_data = {"fuel_flow": 100, "efficiency": 0.85, "temperature": 180}

        is_valid, errors = data_validator.validate_ranges(valid_data, rules)
        assert is_valid is True
        assert len(errors) == 0

        invalid_data = {"fuel_flow": -10, "efficiency": 1.5, "temperature": 600}

        is_valid, errors = data_validator.validate_ranges(invalid_data, rules)
        assert is_valid is False
        assert len(errors) == 3

    def test_type_validation(self, data_validator):
        """Test data type validation."""
        schema = {
            "fuel_flow": float,
            "boiler_id": str,
            "is_operational": bool,
            "sensor_count": int,
        }

        valid_data = {
            "fuel_flow": 100.5,
            "boiler_id": "BOILER01",
            "is_operational": True,
            "sensor_count": 25,
        }

        is_valid = data_validator.validate_types(valid_data, schema)
        assert is_valid is True

    def test_consistency_validation(self, data_validator):
        """Test data consistency validation."""
        data = {
            "fuel_energy_in": 5000,  # kW
            "steam_energy_out": 4000,  # kW
            "losses": 1000,  # kW
            "efficiency": 0.80,
        }

        # Check energy balance
        is_consistent = data_validator.validate_consistency(data)
        assert is_consistent is True

        # Inconsistent data
        data["losses"] = 500  # Doesn't balance
        is_consistent = data_validator.validate_consistency(data)
        assert is_consistent is False

    def test_temporal_validation(self, data_validator):
        """Test temporal data validation."""
        time_series = [
            {"timestamp": datetime.now() - timedelta(hours=2), "value": 100},
            {"timestamp": datetime.now() - timedelta(hours=1), "value": 105},
            {"timestamp": datetime.now(), "value": 110},
        ]

        # Check chronological order
        is_valid = data_validator.validate_temporal_order(time_series)
        assert is_valid is True

        # Out of order
        time_series[0], time_series[2] = time_series[2], time_series[0]
        is_valid = data_validator.validate_temporal_order(time_series)
        assert is_valid is False


# ============================================================================
# TEST TREND ANALYZER
# ============================================================================

class TestTrendAnalyzer:
    """Test trend analysis tool."""

    @pytest.fixture
    def trend_analyzer(self):
        """Create TrendAnalyzer instance."""
        return TrendAnalyzer()

    def test_trend_detection(self, trend_analyzer, historical_data):
        """Test trend detection in time series data."""
        efficiency_data = [d["efficiency"] for d in historical_data]

        trend = trend_analyzer.detect_trend(
            data=efficiency_data, window_size=24  # 24-hour window
        )

        assert "direction" in trend
        assert trend["direction"] in ["increasing", "decreasing", "stable"]
        assert "slope" in trend
        assert "confidence" in trend
        assert 0 <= trend["confidence"] <= 1

    def test_seasonality_detection(self, trend_analyzer):
        """Test seasonality detection."""
        # Create data with daily seasonality
        hours = 168  # One week
        data = []
        for i in range(hours):
            value = 100 + 10 * np.sin(2 * np.pi * i / 24)  # Daily pattern
            data.append(value)

        seasonality = trend_analyzer.detect_seasonality(data)

        assert "has_seasonality" in seasonality
        assert seasonality["has_seasonality"] is True
        assert "period" in seasonality
        assert seasonality["period"] == 24  # Daily

    def test_anomaly_detection_in_trends(self, trend_analyzer):
        """Test anomaly detection within trends."""
        normal_data = [100 + np.random.normal(0, 5) for _ in range(100)]
        normal_data[50] = 150  # Inject anomaly

        anomalies = trend_analyzer.detect_anomalies(
            data=normal_data, method="isolation_forest", threshold=0.95
        )

        assert len(anomalies) > 0
        assert 50 in [a["index"] for a in anomalies]

    def test_forecast_generation(self, trend_analyzer, historical_data):
        """Test forecasting based on trends."""
        efficiency_data = [d["efficiency"] for d in historical_data]

        forecast = trend_analyzer.forecast(
            historical_data=efficiency_data,
            periods=24,  # Forecast 24 hours
            method="arima",
        )

        assert len(forecast["predictions"]) == 24
        assert "confidence_intervals" in forecast
        assert "model_metrics" in forecast
        assert forecast["model_metrics"]["rmse"] is not None


# ============================================================================
# TEST PERFORMANCE PREDICTOR
# ============================================================================

class TestPerformancePredictor:
    """Test performance prediction tool."""

    @pytest.fixture
    def performance_predictor(self):
        """Create PerformancePredictor instance."""
        return PerformancePredictor()

    def test_efficiency_prediction(self, performance_predictor):
        """Test efficiency prediction."""
        input_conditions = {
            "fuel_type": "natural_gas",
            "fuel_flow": 105,
            "excess_air": 1.18,
            "ambient_temperature": 28,
            "load_factor": 0.8,
        }

        prediction = performance_predictor.predict_efficiency(input_conditions)

        assert "predicted_efficiency" in prediction
        assert 0 <= prediction["predicted_efficiency"] <= 1
        assert "confidence_interval" in prediction
        assert "model_used" in prediction

    def test_maintenance_impact_prediction(self, performance_predictor):
        """Test prediction of maintenance impact on performance."""
        maintenance_action = {
            "type": "burner_tuning",
            "current_efficiency": 0.82,
            "component_age_days": 365,
        }

        impact = performance_predictor.predict_maintenance_impact(maintenance_action)

        assert "efficiency_improvement" in impact
        assert impact["efficiency_improvement"] >= 0
        assert "duration_days" in impact
        assert "roi_days" in impact  # Return on investment

    def test_degradation_prediction(self, performance_predictor, historical_data):
        """Test performance degradation prediction."""
        prediction = performance_predictor.predict_degradation(
            historical_data=historical_data, forecast_days=90
        )

        assert "degradation_curve" in prediction
        assert len(prediction["degradation_curve"]) == 90
        assert "critical_date" in prediction  # When efficiency drops below threshold
        assert "uncertainty_bounds" in prediction


# ============================================================================
# TEST COST CALCULATOR
# ============================================================================

class TestCostCalculator:
    """Test cost calculation tool."""

    @pytest.fixture
    def cost_calculator(self):
        """Create CostCalculator instance."""
        config = {
            "fuel_costs": {
                "natural_gas": 0.35,  # $/kg
                "coal": 0.15,  # $/kg
                "oil": 0.45,  # $/kg
            },
            "emission_costs": {
                "co2": 25,  # $/tonne
                "nox": 500,  # $/tonne
            },
        }
        return CostCalculator(config)

    def test_operating_cost_calculation(self, cost_calculator):
        """Test operating cost calculation."""
        operating_data = {
            "fuel_type": "natural_gas",
            "fuel_consumption": 2400,  # kg/day
            "maintenance_hours": 8,
            "labor_rate": 50,  # $/hour
        }

        costs = cost_calculator.calculate_operating_costs(operating_data)

        assert "fuel_cost" in costs
        assert costs["fuel_cost"] == 840  # 2400 * 0.35
        assert "labor_cost" in costs
        assert costs["labor_cost"] == 400  # 8 * 50
        assert "total_daily_cost" in costs

    def test_efficiency_improvement_savings(self, cost_calculator):
        """Test calculation of savings from efficiency improvement."""
        scenario = {
            "current_efficiency": 0.82,
            "improved_efficiency": 0.85,
            "daily_fuel_consumption": 2400,  # kg
            "fuel_type": "natural_gas",
            "operating_days": 365,
        }

        savings = cost_calculator.calculate_improvement_savings(scenario)

        assert "daily_savings" in savings
        assert "annual_savings" in savings
        assert savings["annual_savings"] > 0
        assert "payback_period" in savings

    def test_emission_cost_calculation(self, cost_calculator):
        """Test emission cost calculation."""
        emissions = {
            "co2": 5.5,  # tonnes/day
            "nox": 0.05,  # tonnes/day
        }

        costs = cost_calculator.calculate_emission_costs(emissions)

        assert "co2_cost" in costs
        assert costs["co2_cost"] == 137.5  # 5.5 * 25
        assert "nox_cost" in costs
        assert costs["nox_cost"] == 25  # 0.05 * 500
        assert "total_emission_cost" in costs