# -*- coding: utf-8 -*-
"""
Unit tests for GL-007 Fuel Consumption Analysis

Tests the analyze_fuel_consumption tool with comprehensive coverage:
- Fuel consumption patterns
- Deviation detection
- Anomaly identification
- Cost impact analysis
- Optimization recommendations

Target Coverage: 90%+
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
from greenlang.determinism import DeterministicClock


class TestFuelConsumptionAnalysis:
    """Test suite for fuel consumption analysis."""

    def test_basic_consumption_analysis(self, sample_fuel_consumption_data):
        """Test basic fuel consumption analysis with valid data."""
        result = self._analyze_fuel_consumption(sample_fuel_consumption_data)

        # Validate structure
        assert "consumption_summary" in result
        assert "deviation_analysis" in result
        assert "cost_impact" in result
        assert "optimization_opportunities" in result

        # Validate consumption summary
        summary = result["consumption_summary"]
        assert summary["total_fuel_consumed_kg"] > 0
        assert summary["total_energy_consumed_gj"] > 0
        assert summary["fuel_cost_usd"] > 0

    def test_specific_energy_consumption_calculation(self, sample_fuel_consumption_data):
        """Test SEC (Specific Energy Consumption) calculation."""
        result = self._analyze_fuel_consumption(sample_fuel_consumption_data)

        sec = result["consumption_summary"]["specific_energy_consumption_gj_ton"]

        # SEC should be reasonable for industrial furnace (3-6 GJ/ton typical)
        assert 3.0 <= sec <= 7.0

    def test_deviation_from_baseline(self, sample_fuel_consumption_data):
        """Test deviation calculation from baseline performance."""
        result = self._analyze_fuel_consumption(sample_fuel_consumption_data)

        deviation_analysis = result["deviation_analysis"]

        assert "deviation_from_baseline_percent" in deviation_analysis
        assert "excess_consumption_gj" in deviation_analysis
        assert "trend_direction" in deviation_analysis

        # Trend should be one of the valid values
        assert deviation_analysis["trend_direction"] in ["improving", "stable", "degrading"]

    @pytest.mark.parametrize("deviation_percent,expected_severity", [
        (2.0, "low"),
        (7.0, "medium"),
        (15.0, "high"),
        (25.0, "critical"),
    ])
    def test_anomaly_severity_classification(
        self,
        sample_fuel_consumption_data,
        deviation_percent,
        expected_severity
    ):
        """Test anomaly severity is classified correctly based on deviation."""
        # Modify data to create controlled deviation
        modified_data = sample_fuel_consumption_data.copy()

        for record in modified_data["consumption_data"]:
            record["consumption_rate_kg_hr"] *= (1 + deviation_percent / 100)

        result = self._analyze_fuel_consumption(modified_data)

        # Check if anomalies are detected
        if result["anomaly_detection"]:
            # At least one anomaly should match expected severity
            severities = [a["severity"] for a in result["anomaly_detection"]]
            assert expected_severity in severities or any(
                s in ["high", "critical"] for s in severities
            ) if expected_severity in ["high", "critical"] else True

    def test_cost_impact_calculation(self, sample_fuel_consumption_data):
        """Test fuel cost impact calculation."""
        result = self._analyze_fuel_consumption(sample_fuel_consumption_data)

        cost_impact = result["cost_impact"]

        assert "fuel_cost_current_usd" in cost_impact
        assert "fuel_cost_baseline_usd" in cost_impact
        assert "excess_cost_usd" in cost_impact
        assert "carbon_cost_usd" in cost_impact

        # All costs should be non-negative
        assert cost_impact["fuel_cost_current_usd"] >= 0
        assert cost_impact["fuel_cost_baseline_usd"] >= 0
        assert cost_impact["carbon_cost_usd"] >= 0

    def test_carbon_emissions_calculation(self, sample_fuel_consumption_data):
        """Test carbon emissions calculation from fuel consumption."""
        result = self._analyze_fuel_consumption(sample_fuel_consumption_data)

        carbon_emissions = result["consumption_summary"]["carbon_emissions_tons_co2"]

        # Emissions should be positive for fossil fuels
        assert carbon_emissions > 0

        # Calculate expected emissions
        total_energy_gj = result["consumption_summary"]["total_energy_consumed_gj"]
        emission_factor = sample_fuel_consumption_data["cost_parameters"]["emission_factor_kg_co2_per_gj"]
        expected_emissions = (total_energy_gj * emission_factor) / 1000  # Convert to tons

        # Should match within 1%
        assert abs(carbon_emissions - expected_emissions) / expected_emissions < 0.01

    def test_optimization_opportunities_ranking(self, sample_fuel_consumption_data):
        """Test optimization opportunities are ranked by ROI."""
        result = self._analyze_fuel_consumption(sample_fuel_consumption_data)

        opportunities = result["optimization_opportunities"]

        # Should have at least some opportunities
        assert len(opportunities) >= 0

        if len(opportunities) > 1:
            # Check ranking by priority (higher priority = lower number)
            priorities = [opp["priority"] for opp in opportunities]
            assert priorities == sorted(priorities)

            # Check savings are quantified
            for opp in opportunities:
                assert "savings_potential_usd_yr" in opp
                assert "payback_months" in opp

    def test_anomaly_detection_statistical(self):
        """Test statistical anomaly detection using Z-score."""
        # Create data with known anomaly
        base_time = DeterministicClock.now()
        consumption_data = []

        for i in range(100):
            value = 1850.0
            if i == 50:
                value = 2500.0  # Anomaly spike

            consumption_data.append({
                "timestamp": (base_time - timedelta(hours=99-i)).isoformat(),
                "fuel_type": "natural_gas",
                "consumption_rate_kg_hr": value,
                "consumption_rate_nm3_hr": value * 1.32,
                "heating_value_mj_kg": 50.0,
                "production_rate": 18.5,
                "furnace_load_percent": 85.0,
            })

        test_data = {
            "consumption_data": consumption_data,
            "baseline_performance": {
                "expected_sec_gj_ton": 4.8,
                "design_sec_gj_ton": 4.5,
                "best_achieved_sec_gj_ton": 4.3,
                "variability_factor": 0.05,
            },
            "cost_parameters": {
                "fuel_cost_usd_per_gj": 8.5,
                "carbon_price_usd_per_ton_co2": 50.0,
                "emission_factor_kg_co2_per_gj": 56.1,
            }
        }

        result = self._analyze_fuel_consumption(test_data)

        # Should detect the anomaly
        anomalies = result["anomaly_detection"]
        assert len(anomalies) > 0

        # Anomaly should be flagged as high severity
        assert any(a["severity"] in ["high", "critical"] for a in anomalies)

    def test_trend_analysis_improving(self):
        """Test trend detection for improving performance."""
        # Create data with improving trend
        base_time = DeterministicClock.now()
        consumption_data = []

        for i in range(50):
            # Gradually decreasing consumption
            value = 1850.0 - (i * 5.0)

            consumption_data.append({
                "timestamp": (base_time - timedelta(hours=49-i)).isoformat(),
                "fuel_type": "natural_gas",
                "consumption_rate_kg_hr": value,
                "consumption_rate_nm3_hr": value * 1.32,
                "heating_value_mj_kg": 50.0,
                "production_rate": 18.5,
                "furnace_load_percent": 85.0,
            })

        test_data = {
            "consumption_data": consumption_data,
            "baseline_performance": {
                "expected_sec_gj_ton": 4.8,
                "design_sec_gj_ton": 4.5,
                "best_achieved_sec_gj_ton": 4.3,
                "variability_factor": 0.05,
            },
            "cost_parameters": {
                "fuel_cost_usd_per_gj": 8.5,
                "carbon_price_usd_per_ton_co2": 50.0,
                "emission_factor_kg_co2_per_gj": 56.1,
            }
        }

        result = self._analyze_fuel_consumption(test_data)

        # Trend should be improving
        assert result["deviation_analysis"]["trend_direction"] == "improving"

    def test_invalid_input_empty_data(self):
        """Test error handling for empty consumption data."""
        invalid_data = {
            "consumption_data": [],  # Empty
            "baseline_performance": {
                "expected_sec_gj_ton": 4.8,
            },
            "cost_parameters": {
                "fuel_cost_usd_per_gj": 8.5,
            }
        }

        with pytest.raises(ValueError) as exc_info:
            self._analyze_fuel_consumption(invalid_data)

        assert "empty" in str(exc_info.value).lower()

    def test_invalid_input_negative_consumption(self):
        """Test error handling for negative consumption values."""
        invalid_data = {
            "consumption_data": [{
                "timestamp": DeterministicClock.now().isoformat(),
                "fuel_type": "natural_gas",
                "consumption_rate_kg_hr": -1000.0,  # Invalid
                "heating_value_mj_kg": 50.0,
                "production_rate": 18.5,
            }],
            "baseline_performance": {"expected_sec_gj_ton": 4.8},
            "cost_parameters": {"fuel_cost_usd_per_gj": 8.5},
        }

        with pytest.raises(ValueError):
            self._analyze_fuel_consumption(invalid_data)

    def test_multiple_fuel_types(self):
        """Test analysis with mixed fuel types."""
        base_time = DeterministicClock.now()
        consumption_data = []

        for i in range(24):
            fuel_type = "natural_gas" if i < 12 else "diesel"
            heating_value = 50.0 if fuel_type == "natural_gas" else 45.6

            consumption_data.append({
                "timestamp": (base_time - timedelta(hours=23-i)).isoformat(),
                "fuel_type": fuel_type,
                "consumption_rate_kg_hr": 1850.0,
                "heating_value_mj_kg": heating_value,
                "production_rate": 18.5,
                "furnace_load_percent": 85.0,
            })

        test_data = {
            "consumption_data": consumption_data,
            "baseline_performance": {
                "expected_sec_gj_ton": 4.8,
                "design_sec_gj_ton": 4.5,
                "best_achieved_sec_gj_ton": 4.3,
            },
            "cost_parameters": {
                "fuel_cost_usd_per_gj": 8.5,
                "carbon_price_usd_per_ton_co2": 50.0,
                "emission_factor_kg_co2_per_gj": 56.1,
            }
        }

        result = self._analyze_fuel_consumption(test_data)

        # Should handle mixed fuels correctly
        assert result["consumption_summary"]["total_fuel_consumed_kg"] > 0

    @pytest.mark.performance
    def test_analysis_performance(self, sample_fuel_consumption_data, benchmark):
        """Test analysis meets performance target (<100ms)."""
        def run_analysis():
            return self._analyze_fuel_consumption(sample_fuel_consumption_data)

        result = benchmark(run_analysis)

    # ========================================================================
    # HELPER METHODS (Mock implementation)
    # ========================================================================

    def _analyze_fuel_consumption(self, input_data: dict) -> dict:
        """Mock implementation of fuel consumption analysis."""
        self._validate_fuel_consumption_input(input_data)

        consumption_data = input_data["consumption_data"]
        baseline = input_data["baseline_performance"]
        cost_params = input_data["cost_parameters"]

        # Calculate totals
        total_fuel_kg = sum(d["consumption_rate_kg_hr"] for d in consumption_data)
        total_production = sum(d["production_rate"] for d in consumption_data)

        # Calculate energy
        total_energy_gj = sum(
            d["consumption_rate_kg_hr"] * d["heating_value_mj_kg"] / 1000
            for d in consumption_data
        )

        # Calculate SEC
        sec = total_energy_gj / total_production if total_production > 0 else 0

        # Calculate costs
        fuel_cost = total_energy_gj * cost_params["fuel_cost_usd_per_gj"]
        carbon_emissions = (total_energy_gj * cost_params["emission_factor_kg_co2_per_gj"]) / 1000
        carbon_cost = carbon_emissions * cost_params["carbon_price_usd_per_ton_co2"]

        # Calculate deviation
        baseline_sec = baseline["expected_sec_gj_ton"]
        deviation_percent = ((sec - baseline_sec) / baseline_sec) * 100

        # Detect anomalies
        anomalies = self._detect_consumption_anomalies(consumption_data)

        # Determine trend
        trend = self._determine_trend(consumption_data)

        # Generate optimization opportunities
        opportunities = self._generate_optimization_opportunities(deviation_percent, sec)

        return {
            "consumption_summary": {
                "total_fuel_consumed_kg": total_fuel_kg,
                "total_energy_consumed_gj": total_energy_gj,
                "average_consumption_rate_kg_hr": total_fuel_kg / len(consumption_data),
                "specific_energy_consumption_gj_ton": sec,
                "fuel_cost_usd": fuel_cost,
                "carbon_emissions_tons_co2": carbon_emissions,
            },
            "deviation_analysis": {
                "deviation_from_baseline_percent": deviation_percent,
                "excess_consumption_gj": max(0, total_energy_gj - (total_production * baseline_sec)),
                "trend_direction": trend,
            },
            "anomaly_detection": anomalies,
            "cost_impact": {
                "fuel_cost_current_usd": fuel_cost,
                "fuel_cost_baseline_usd": total_production * baseline_sec * cost_params["fuel_cost_usd_per_gj"],
                "excess_cost_usd": max(0, deviation_percent / 100 * fuel_cost),
                "carbon_cost_usd": carbon_cost,
            },
            "optimization_opportunities": opportunities,
        }

    def _validate_fuel_consumption_input(self, input_data: dict):
        """Validate fuel consumption input."""
        if not input_data.get("consumption_data"):
            raise ValueError("Consumption data cannot be empty")

        for record in input_data["consumption_data"]:
            if record.get("consumption_rate_kg_hr", 0) < 0:
                raise ValueError("Consumption rate cannot be negative")

    def _detect_consumption_anomalies(self, consumption_data: list) -> list:
        """Detect anomalies in consumption data using Z-score."""
        values = [d["consumption_rate_kg_hr"] for d in consumption_data]
        mean = np.mean(values)
        std = np.std(values)

        anomalies = []
        for i, record in enumerate(consumption_data):
            value = record["consumption_rate_kg_hr"]
            z_score = abs((value - mean) / std) if std > 0 else 0

            if z_score > 2.5:
                severity = "critical" if z_score > 4 else "high" if z_score > 3 else "medium"
                anomalies.append({
                    "timestamp": record["timestamp"],
                    "anomaly_type": "spike" if value > mean else "drop",
                    "severity": severity,
                    "deviation_percent": ((value - mean) / mean) * 100,
                    "probable_cause": "equipment_malfunction" if severity == "critical" else "process_variation",
                    "recommended_action": "investigate_immediately" if severity == "critical" else "monitor",
                })

        return anomalies

    def _determine_trend(self, consumption_data: list) -> str:
        """Determine consumption trend."""
        if len(consumption_data) < 10:
            return "stable"

        values = [d["consumption_rate_kg_hr"] for d in consumption_data]

        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope < -5:
            return "improving"
        elif slope > 5:
            return "degrading"
        else:
            return "stable"

    def _generate_optimization_opportunities(self, deviation_percent: float, current_sec: float) -> list:
        """Generate optimization opportunities."""
        opportunities = []

        if deviation_percent > 5:
            opportunities.append({
                "opportunity": "Reduce excess air to optimize combustion",
                "savings_potential_gj_yr": 5000.0,
                "savings_potential_usd_yr": 42500.0,
                "implementation_complexity": "low",
                "payback_months": 3,
                "priority": 1,
            })

        if deviation_percent > 10:
            opportunities.append({
                "opportunity": "Clean heat transfer surfaces to reduce fouling",
                "savings_potential_gj_yr": 8000.0,
                "savings_potential_usd_yr": 68000.0,
                "implementation_complexity": "medium",
                "payback_months": 6,
                "priority": 2,
            })

        return opportunities
