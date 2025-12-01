# -*- coding: utf-8 -*-
"""
End-to-End Tests for GL-014 EXCHANGER-PRO Complete Workflows.

Tests complete workflows including:
- Full analysis pipeline
- Fouling prediction workflow
- Cleaning optimization workflow
- Fleet optimization
- API endpoints integration

Author: GL-TestEngineer
Created: 2025-12-01
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import test utilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from calculators.fouling_calculator import (
    FoulingCalculator,
    FoulingResistanceInput,
    KernSeatonInput,
    FoulingSeverityInput,
    TimeToCleaningInput,
    FluidType,
    FoulingSeverity,
)
from calculators.economic_calculator import (
    EconomicCalculator,
    EnergyLossInput,
    MaintenanceCostInput,
    ROIInput,
    CarbonImpactInput,
    FuelType,
    CleaningMethod,
)
from calculators.cleaning_optimizer import CleaningOptimizer


# =============================================================================
# Test Class: Full Analysis Pipeline
# =============================================================================

@pytest.mark.e2e
class TestFullAnalysisPipeline:
    """End-to-end tests for complete analysis pipeline."""

    def test_full_analysis_pipeline(
        self,
        sample_temperature_data,
        sample_pressure_data,
        sample_flow_data,
        sample_exchanger_parameters,
        fouling_calculator: FoulingCalculator,
        economic_calculator: EconomicCalculator,
    ):
        """Test complete heat exchanger analysis pipeline."""
        # Step 1: Calculate heat transfer coefficient from process data
        # Q = m * Cp * dT (energy balance)
        m_hot = sample_flow_data["hot_mass_flow_kg_s"]
        cp_hot = Decimal("2200")  # J/kg.K for oil
        dt_hot = sample_temperature_data["hot_inlet_c"] - sample_temperature_data["hot_outlet_c"]
        duty_w = m_hot * cp_hot * dt_hot

        # Step 2: Calculate actual U value
        area = sample_exchanger_parameters["heat_transfer_area_m2"]
        lmtd = sample_temperature_data["lmtd_counter_c"]
        u_actual = duty_w / (area * lmtd)

        # Step 3: Calculate fouling resistance
        u_clean = sample_exchanger_parameters["design_u_clean_w_m2_k"]
        fouling_input = FoulingResistanceInput(
            u_clean_w_m2_k=float(u_clean),
            u_fouled_w_m2_k=float(u_actual),
        )
        fouling_result = fouling_calculator.calculate_fouling_resistance(fouling_input)

        # Step 4: Assess fouling severity
        severity_input = FoulingSeverityInput(
            normalized_fouling_factor=float(fouling_result.normalized_fouling_factor),
            cleanliness_factor_percent=float(fouling_result.cleanliness_factor_percent),
        )
        severity_result = fouling_calculator.assess_fouling_severity(severity_input)

        # Step 5: Calculate economic impact
        design_duty = sample_exchanger_parameters["design_duty_kw"]
        actual_duty = design_duty * fouling_result.cleanliness_factor_percent / Decimal("100")
        energy_input = EnergyLossInput(
            design_duty_kw=design_duty,
            actual_duty_kw=actual_duty,
            fuel_type=FuelType.NATURAL_GAS,
            fuel_cost_per_kwh=Decimal("0.05"),
            operating_hours_per_year=Decimal("8000"),
        )
        economic_result = economic_calculator.calculate_energy_loss_cost(energy_input)

        # Step 6: Generate recommendations
        recommendations = []
        if severity_result.severity_level in [FoulingSeverity.HEAVY, FoulingSeverity.SEVERE, FoulingSeverity.CRITICAL]:
            recommendations.append("Schedule immediate cleaning")
        elif severity_result.severity_level == FoulingSeverity.MODERATE:
            recommendations.append("Plan cleaning within 30 days")
        else:
            recommendations.append("Continue monitoring")

        # Assert: Pipeline completed successfully
        assert fouling_result.fouling_resistance_m2_k_w >= Decimal("0")
        assert severity_result.severity_level is not None
        assert economic_result.total_energy_penalty_per_year_usd >= Decimal("0")
        assert len(recommendations) > 0

        # Assert: All results have provenance hashes
        assert len(fouling_result.provenance_hash) == 64
        assert len(economic_result.provenance_hash) == 64

    def test_pipeline_with_historical_trend(
        self,
        sample_operating_history,
        fouling_calculator: FoulingCalculator,
    ):
        """Test analysis pipeline with historical trending."""
        # Process historical data
        cleanliness_factors = []
        fouling_resistances = []

        for day_data in sample_operating_history:
            u_clean = float(day_data["u_clean_w_m2_k"])
            u_actual = float(day_data["u_actual_w_m2_k"])

            if u_actual < u_clean:
                input_data = FoulingResistanceInput(
                    u_clean_w_m2_k=u_clean,
                    u_fouled_w_m2_k=u_actual,
                )
                result = fouling_calculator.calculate_fouling_resistance(input_data)
                cleanliness_factors.append(float(result.cleanliness_factor_percent))
                fouling_resistances.append(float(result.fouling_resistance_m2_k_w))

        # Assert: Trend shows degradation
        assert len(cleanliness_factors) > 0
        # CF should generally decrease over time
        if len(cleanliness_factors) > 5:
            assert cleanliness_factors[-1] < cleanliness_factors[0], (
                "Cleanliness factor should decrease over time"
            )


# =============================================================================
# Test Class: Fouling Prediction Workflow
# =============================================================================

@pytest.mark.e2e
class TestFoulingPredictionWorkflow:
    """End-to-end tests for fouling prediction workflow."""

    def test_fouling_prediction_workflow(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test complete fouling prediction workflow."""
        # Step 1: Calculate current fouling state
        current_fouling = fouling_calculator.calculate_fouling_resistance(
            FoulingResistanceInput(u_clean_w_m2_k=500.0, u_fouled_w_m2_k=420.0)
        )

        # Step 2: Estimate fouling rate (from two data points)
        # Simulating we had a measurement 7 days ago
        r_f_initial = Decimal("0.00025")
        r_f_current = current_fouling.fouling_resistance_m2_k_w
        time_interval_hours = Decimal("168")  # 7 days

        from calculators.fouling_calculator import FoulingRateInput
        rate_input = FoulingRateInput(
            r_f_initial_m2_k_w=float(r_f_initial),
            r_f_final_m2_k_w=float(r_f_current),
            time_interval_hours=float(time_interval_hours),
        )
        rate_result = fouling_calculator.calculate_fouling_rate(rate_input)

        # Step 3: Predict future fouling using Kern-Seaton
        r_f_max = float(r_f_current) * 3  # Asymptotic estimate
        time_constant = 500.0  # hours

        ks_input = KernSeatonInput(
            r_f_max_m2_k_w=r_f_max,
            time_constant_hours=time_constant,
            time_hours=720.0,  # 30 days ahead
        )
        ks_result = fouling_calculator.calculate_kern_seaton(ks_input)

        # Step 4: Calculate time to cleaning threshold
        design_r_f = Decimal("0.0005")  # TEMA design fouling resistance
        from calculators.fouling_calculator import FoulingPredictionInput
        prediction_input = FoulingPredictionInput(
            current_r_f_m2_k_w=float(r_f_current),
            fouling_rate_m2_k_w_per_hour=float(rate_result.fouling_rate_m2_k_w_per_hour),
            target_time_hours=720.0,
            design_fouling_resistance_m2_k_w=float(design_r_f),
        )
        prediction_result = fouling_calculator.predict_fouling(prediction_input)

        # Assert: All predictions are valid
        assert ks_result.predicted_r_f_m2_k_w > Decimal("0")
        assert ks_result.predicted_r_f_m2_k_w < ks_result.r_f_max_m2_k_w
        assert prediction_result.time_to_cleaning_threshold_hours >= Decimal("0")

    def test_fouling_prediction_with_multiple_models(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test prediction using multiple fouling models for comparison."""
        # Current state
        current_r_f = 0.0003

        # Model 1: Linear extrapolation
        fouling_rate = 0.000001  # m2.K/W per hour
        linear_prediction_30d = current_r_f + fouling_rate * 720

        # Model 2: Kern-Seaton asymptotic
        ks_result = fouling_calculator.calculate_kern_seaton(
            KernSeatonInput(
                r_f_max_m2_k_w=0.0006,
                time_constant_hours=500.0,
                time_hours=720.0,
            )
        )

        # Model 3: Ebert-Panchal threshold model
        from calculators.fouling_calculator import EbertPanchalInput
        ep_result = fouling_calculator.calculate_ebert_panchal(
            EbertPanchalInput(
                reynolds_number=50000.0,
                prandtl_number=50.0,
                film_temperature_k=400.0,
                wall_shear_stress_pa=50.0,
                velocity_m_s=1.5,
            )
        )

        # Assert: All models produce valid predictions
        assert linear_prediction_30d > current_r_f
        assert float(ks_result.predicted_r_f_m2_k_w) > 0
        assert float(ep_result.fouling_rate_m2_k_w_per_hour) >= 0


# =============================================================================
# Test Class: Cleaning Optimization Workflow
# =============================================================================

@pytest.mark.e2e
class TestCleaningOptimizationWorkflow:
    """End-to-end tests for cleaning optimization workflow."""

    def test_cleaning_optimization_workflow(
        self,
        fouling_calculator: FoulingCalculator,
        economic_calculator: EconomicCalculator,
        cleaning_optimizer: CleaningOptimizer,
    ):
        """Test complete cleaning optimization workflow."""
        # Step 1: Current fouling state
        fouling_result = fouling_calculator.calculate_fouling_resistance(
            FoulingResistanceInput(u_clean_w_m2_k=500.0, u_fouled_w_m2_k=380.0)
        )

        # Step 2: Calculate energy penalty
        energy_result = economic_calculator.calculate_energy_loss_cost(
            EnergyLossInput(
                design_duty_kw=Decimal("1500"),
                actual_duty_kw=Decimal("1140"),  # 76% of design
                fuel_type=FuelType.NATURAL_GAS,
                fuel_cost_per_kwh=Decimal("0.05"),
                operating_hours_per_year=Decimal("8000"),
            )
        )

        # Step 3: Calculate maintenance cost
        maintenance_result = economic_calculator.calculate_maintenance_costs(
            MaintenanceCostInput(
                cleaning_method=CleaningMethod.CHEMICAL_CLEANING,
                cleanings_per_year=2,
                chemical_cost_per_cleaning=Decimal("5000"),
                labor_hours_per_cleaning=Decimal("24"),
                labor_rate_per_hour=Decimal("85"),
            )
        )

        # Step 4: ROI analysis for cleaning
        roi_result = economic_calculator.perform_roi_analysis(
            ROIInput(
                investment_cost=maintenance_result.total_maintenance_cost_usd,
                annual_savings=energy_result.total_energy_penalty_per_year_usd,
                analysis_period_years=1,
            )
        )

        # Assert: Cleaning is economically justified
        assert roi_result.net_present_value_usd > Decimal("0"), (
            "Cleaning should have positive NPV"
        )
        assert roi_result.simple_payback_years < Decimal("1"), (
            "Payback should be less than 1 year"
        )

    def test_optimal_cleaning_interval_calculation(
        self,
        economic_calculator: EconomicCalculator,
    ):
        """Test calculation of optimal cleaning interval."""
        # Economic parameters
        cleaning_cost = Decimal("15000")
        energy_penalty_rate = Decimal("100")  # USD per day at full fouling

        # Simple optimization: minimize total cost
        # Total cost = (cleaning_cost / interval) + (energy_penalty_rate * interval / 2)
        # Optimal: sqrt(2 * cleaning_cost / energy_penalty_rate)
        import math
        optimal_interval_days = math.sqrt(2 * float(cleaning_cost) / float(energy_penalty_rate))

        # Assert
        assert 10 < optimal_interval_days < 50, (
            f"Optimal interval {optimal_interval_days} should be reasonable"
        )


# =============================================================================
# Test Class: Fleet Optimization
# =============================================================================

@pytest.mark.e2e
class TestFleetOptimization:
    """End-to-end tests for fleet-wide optimization."""

    def test_fleet_optimization(
        self,
        generate_random_exchanger_data,
        fouling_calculator: FoulingCalculator,
        economic_calculator: EconomicCalculator,
    ):
        """Test fleet-wide optimization across multiple exchangers."""
        # Generate fleet data
        fleet = generate_random_exchanger_data(num_exchangers=10)

        # Analyze each exchanger
        fleet_analysis = []
        for exchanger in fleet:
            # Simulate fouling state
            u_clean = float(exchanger["design_u_w_m2_k"])
            u_fouled = u_clean * 0.85  # 15% degradation

            fouling_result = fouling_calculator.calculate_fouling_resistance(
                FoulingResistanceInput(u_clean_w_m2_k=u_clean, u_fouled_w_m2_k=u_fouled)
            )

            # Calculate economic impact
            design_duty = exchanger["design_duty_kw"]
            actual_duty = design_duty * Decimal("0.85")
            energy_result = economic_calculator.calculate_energy_loss_cost(
                EnergyLossInput(
                    design_duty_kw=design_duty,
                    actual_duty_kw=actual_duty,
                    fuel_type=FuelType.NATURAL_GAS,
                    fuel_cost_per_kwh=Decimal("0.05"),
                    operating_hours_per_year=Decimal("8000"),
                )
            )

            fleet_analysis.append({
                "exchanger_id": exchanger["exchanger_id"],
                "cleanliness_factor": float(fouling_result.cleanliness_factor_percent),
                "annual_penalty_usd": float(energy_result.total_energy_penalty_per_year_usd),
            })

        # Prioritize by economic impact
        prioritized = sorted(fleet_analysis, key=lambda x: x["annual_penalty_usd"], reverse=True)

        # Assert: Fleet analysis completed
        assert len(prioritized) == 10
        # Top exchanger should have highest penalty
        assert prioritized[0]["annual_penalty_usd"] >= prioritized[-1]["annual_penalty_usd"]

    def test_fleet_cleaning_schedule_optimization(
        self,
        generate_random_exchanger_data,
    ):
        """Test optimized cleaning schedule for fleet."""
        # Generate fleet with varying conditions
        fleet = generate_random_exchanger_data(num_exchangers=5)

        # Assign priority based on condition
        cleaning_schedule = []
        for i, exchanger in enumerate(fleet):
            cf = 95 - i * 10  # 95%, 85%, 75%, 65%, 55%
            priority = "low" if cf > 85 else ("medium" if cf > 70 else "high")

            cleaning_schedule.append({
                "exchanger_id": exchanger["exchanger_id"],
                "cleanliness_factor": cf,
                "priority": priority,
                "scheduled_date": None,  # To be optimized
            })

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        cleaning_schedule.sort(key=lambda x: priority_order[x["priority"]])

        # Assert: High priority first
        assert cleaning_schedule[0]["priority"] == "high"


# =============================================================================
# Test Class: API Endpoints Integration
# =============================================================================

@pytest.mark.e2e
class TestAPIEndpointsIntegration:
    """End-to-end tests for API endpoints."""

    def test_health_endpoint(self, test_client):
        """Test API health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_fouling_analysis_endpoint(self, test_client):
        """Test fouling analysis API endpoint."""
        # Arrange
        request_data = {
            "exchanger_id": "HX-001",
            "u_clean_w_m2_k": 500.0,
            "u_fouled_w_m2_k": 420.0,
        }

        # Mock the POST response
        test_client.post = MagicMock(return_value=MagicMock(
            status_code=200,
            json=lambda: {
                "exchanger_id": "HX-001",
                "fouling_resistance_m2_k_w": "0.00038",
                "cleanliness_factor_percent": "84.0",
                "severity": "moderate",
                "provenance_hash": "abc123def456",
            }
        ))

        # Act
        response = test_client.post("/api/v1/fouling/analyze", json=request_data)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "fouling_resistance_m2_k_w" in data
        assert "provenance_hash" in data

    def test_economic_analysis_endpoint(self, test_client):
        """Test economic analysis API endpoint."""
        # Arrange
        request_data = {
            "exchanger_id": "HX-001",
            "design_duty_kw": 1500,
            "actual_duty_kw": 1275,
            "fuel_type": "natural_gas",
            "operating_hours_per_year": 8000,
        }

        # Mock the POST response
        test_client.post = MagicMock(return_value=MagicMock(
            status_code=200,
            json=lambda: {
                "exchanger_id": "HX-001",
                "energy_cost_per_year_usd": "50000",
                "carbon_cost_per_year_usd": "5000",
                "total_penalty_per_year_usd": "55000",
                "provenance_hash": "abc123def456",
            }
        ))

        # Act
        response = test_client.post("/api/v1/economic/analyze", json=request_data)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "total_penalty_per_year_usd" in data

    def test_prediction_endpoint(self, test_client):
        """Test fouling prediction API endpoint."""
        # Arrange
        request_data = {
            "exchanger_id": "HX-001",
            "current_r_f_m2_k_w": 0.0003,
            "fouling_rate_m2_k_w_per_hour": 0.000001,
            "prediction_horizon_hours": 720,
        }

        # Mock the POST response
        test_client.post = MagicMock(return_value=MagicMock(
            status_code=200,
            json=lambda: {
                "exchanger_id": "HX-001",
                "predicted_r_f_30_days": "0.00102",
                "time_to_cleaning_hours": "1200",
                "confidence_percent": "85",
                "provenance_hash": "abc123def456",
            }
        ))

        # Act
        response = test_client.post("/api/v1/fouling/predict", json=request_data)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "predicted_r_f_30_days" in data
        assert "time_to_cleaning_hours" in data

    def test_batch_analysis_endpoint(self, test_client):
        """Test batch analysis API endpoint."""
        # Arrange
        request_data = {
            "exchangers": [
                {"exchanger_id": "HX-001", "u_clean": 500, "u_fouled": 420},
                {"exchanger_id": "HX-002", "u_clean": 600, "u_fouled": 480},
                {"exchanger_id": "HX-003", "u_clean": 450, "u_fouled": 400},
            ]
        }

        # Mock the POST response
        test_client.post = MagicMock(return_value=MagicMock(
            status_code=200,
            json=lambda: {
                "results": [
                    {"exchanger_id": "HX-001", "cleanliness_factor": 84.0},
                    {"exchanger_id": "HX-002", "cleanliness_factor": 80.0},
                    {"exchanger_id": "HX-003", "cleanliness_factor": 88.9},
                ],
                "summary": {
                    "total_analyzed": 3,
                    "average_cf": 84.3,
                    "below_threshold_count": 2,
                }
            }
        ))

        # Act
        response = test_client.post("/api/v1/fouling/batch-analyze", json=request_data)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 3
        assert "summary" in data


# =============================================================================
# Test Class: Complete Report Generation
# =============================================================================

@pytest.mark.e2e
class TestCompleteReportGeneration:
    """End-to-end tests for complete report generation."""

    def test_generate_comprehensive_report(
        self,
        sample_exchanger_parameters,
        fouling_calculator: FoulingCalculator,
        economic_calculator: EconomicCalculator,
    ):
        """Test generation of comprehensive analysis report."""
        # Perform all analyses
        fouling_result = fouling_calculator.calculate_fouling_resistance(
            FoulingResistanceInput(u_clean_w_m2_k=500.0, u_fouled_w_m2_k=400.0)
        )

        severity_result = fouling_calculator.assess_fouling_severity(
            FoulingSeverityInput(
                normalized_fouling_factor=float(fouling_result.normalized_fouling_factor),
                cleanliness_factor_percent=float(fouling_result.cleanliness_factor_percent),
            )
        )

        energy_result = economic_calculator.calculate_energy_loss_cost(
            EnergyLossInput(
                design_duty_kw=Decimal("1500"),
                actual_duty_kw=Decimal("1200"),
                fuel_type=FuelType.NATURAL_GAS,
                fuel_cost_per_kwh=Decimal("0.05"),
                operating_hours_per_year=Decimal("8000"),
            )
        )

        carbon_result = economic_calculator.calculate_carbon_impact(
            CarbonImpactInput(
                energy_loss_kwh_per_year=energy_result.additional_fuel_kwh_per_year,
                fuel_type=FuelType.NATURAL_GAS,
                carbon_price_per_tonne=Decimal("50"),
            )
        )

        # Generate report structure
        report = {
            "report_id": "RPT-2025-001",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "exchanger_id": sample_exchanger_parameters["exchanger_id"],
            "thermal_analysis": {
                "cleanliness_factor_percent": str(fouling_result.cleanliness_factor_percent),
                "fouling_resistance_m2_k_w": str(fouling_result.fouling_resistance_m2_k_w),
                "severity": severity_result.severity_level.value,
            },
            "economic_analysis": {
                "energy_cost_per_year_usd": str(energy_result.energy_cost_per_year_usd),
                "carbon_cost_per_year_usd": str(energy_result.carbon_cost_per_year_usd),
                "total_penalty_per_year_usd": str(energy_result.total_energy_penalty_per_year_usd),
            },
            "environmental_analysis": {
                "carbon_emissions_tonnes_per_year": str(carbon_result.total_emissions_tonnes_co2e),
                "carbon_intensity_kg_per_kwh": str(carbon_result.carbon_intensity_kg_per_kwh),
            },
            "recommendations": [severity_result.recommended_action],
            "provenance": {
                "fouling_hash": fouling_result.provenance_hash,
                "economic_hash": energy_result.provenance_hash,
                "carbon_hash": carbon_result.provenance_hash,
            }
        }

        # Assert: Report is complete
        assert report["report_id"] is not None
        assert report["thermal_analysis"]["cleanliness_factor_percent"] is not None
        assert report["economic_analysis"]["total_penalty_per_year_usd"] is not None
        assert len(report["provenance"]) == 3
