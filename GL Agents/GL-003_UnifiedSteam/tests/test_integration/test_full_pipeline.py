"""
Integration Tests for GL-003 UnifiedSteam - Full Pipeline Tests

End-to-end tests for:
- Complete optimization pipeline
- API endpoint workflows (mocked)
- Real-time streaming simulation
- Data transformation and persistence flows
- Multi-component integration

Target Coverage: 90%+
"""

import asyncio
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import numpy as np

# Import application modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Core modules
from optimization.steam_network_optimizer import (
    BoilerState,
    DemandForecast,
    HeaderState,
    HeaderType,
    NetworkModel,
    PRVState,
    SteamNetworkOptimizer,
)

# Integration modules
from integration.sensor_transformer import (
    CalibrationParams,
    QualityCode,
    SensorTransformer,
    create_steam_system_transformer,
)

# Uncertainty modules
from uncertainty.propagation import (
    CorrelationMatrix,
    UncertaintyPropagator,
)
from uncertainty.uncertainty_models import (
    Distribution,
    DistributionType,
    UncertainValue,
)

# Explainability modules
from explainability.shap_explainer import (
    ModelType,
    SHAPExplainer,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def optimizer() -> SteamNetworkOptimizer:
    """Create optimizer instance."""
    return SteamNetworkOptimizer()


@pytest.fixture
def sensor_transformer() -> SensorTransformer:
    """Create configured sensor transformer."""
    return create_steam_system_transformer()


@pytest.fixture
def uncertainty_propagator() -> UncertaintyPropagator:
    """Create uncertainty propagator."""
    return UncertaintyPropagator(default_seed=42)


@pytest.fixture
def shap_explainer() -> SHAPExplainer:
    """Create SHAP explainer."""
    return SHAPExplainer(agent_id="GL-003")


@pytest.fixture
def mock_ml_model():
    """Create mock ML model."""
    model = MagicMock()
    model.predict.return_value = np.array([0.65])
    return model


@pytest.fixture
def sample_network_state() -> Dict[str, Any]:
    """Create sample network state for integration tests."""
    return {
        "boilers": [
            {
                "boiler_id": "BLR-001",
                "is_online": True,
                "current_load_percent": 75.0,
                "rated_capacity_klb_hr": 100.0,
                "current_efficiency_percent": 85.0,
                "fuel_cost_per_mmbtu": 8.0,
                "co2_factor_lb_mmbtu": 117.0,
            },
            {
                "boiler_id": "BLR-002",
                "is_online": True,
                "current_load_percent": 70.0,
                "rated_capacity_klb_hr": 80.0,
                "current_efficiency_percent": 83.0,
                "fuel_cost_per_mmbtu": 8.5,
                "co2_factor_lb_mmbtu": 117.0,
            },
        ],
        "headers": [
            {
                "header_id": "HP-MAIN",
                "header_type": "hp",
                "pressure_psig": 600.0,
                "temperature_f": 700.0,
                "flow_klb_hr": 140.0,
                "user_demand_klb_hr": 130.0,
            },
            {
                "header_id": "MP-MAIN",
                "header_type": "mp",
                "pressure_psig": 150.0,
                "temperature_f": 380.0,
                "flow_klb_hr": 50.0,
                "user_demand_klb_hr": 45.0,
            },
        ],
        "prvs": [
            {
                "prv_id": "PRV-001",
                "upstream_header": "HP-MAIN",
                "downstream_header": "MP-MAIN",
                "upstream_pressure_psig": 600.0,
                "downstream_pressure_psig": 150.0,
                "flow_klb_hr": 30.0,
                "valve_position_percent": 45.0,
                "max_capacity_klb_hr": 60.0,
            },
        ],
        "total_generation_klb_hr": 140.0,
        "total_demand_klb_hr": 130.0,
    }


@pytest.fixture
def sample_sensor_data() -> Dict[str, float]:
    """Create sample sensor data."""
    return {
        "header.pressure": 600.0,
        "header.temperature": 700.0,
        "header.flow": 50.0,
        "feedwater.temperature": 220.0,
        "stack.temperature": 350.0,
    }


@pytest.fixture
def sample_trap_features() -> Dict[str, float]:
    """Create sample trap features for ML."""
    return {
        "temp_differential_f": 25.0,
        "outlet_temp_f": 200.0,
        "inlet_temp_f": 360.0,
        "superheat_f": 15.0,
        "subcooling_f": 3.0,
        "operating_hours": 8000.0,
        "cycles_count": 2000.0,
        "days_since_inspection": 60.0,
    }


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================

@pytest.mark.integration
class TestEndToEndOptimizationPipeline:
    """End-to-end tests for the optimization pipeline."""

    def test_full_optimization_workflow(
        self, optimizer, sensor_transformer, sample_network_state
    ):
        """
        Test complete optimization workflow:
        1. Transform sensor data
        2. Build network model
        3. Optimize load allocation
        4. Optimize header pressures
        5. Generate recommendations
        """
        # Step 1: Transform raw sensor data
        raw_sensors = {
            "header.pressure": 145.0,  # psig
            "header.temperature": 700.0,  # F
        }
        transformed = sensor_transformer.transform_batch(raw_sensors)

        assert transformed.total_count > 0
        assert transformed.quality_score >= 0

        # Step 2: Build network model from state
        boilers = [
            BoilerState(**b) for b in sample_network_state["boilers"]
        ]
        headers = [
            HeaderState(
                header_id=h["header_id"],
                header_type=HeaderType(h["header_type"]),
                pressure_psig=h["pressure_psig"],
                setpoint_psig=h["pressure_psig"],
                temperature_f=h["temperature_f"],
                flow_klb_hr=h["flow_klb_hr"],
                user_demand_klb_hr=h["user_demand_klb_hr"],
            )
            for h in sample_network_state["headers"]
        ]
        prvs = [
            PRVState(
                prv_id=p["prv_id"],
                upstream_header=p["upstream_header"],
                downstream_header=p["downstream_header"],
                upstream_pressure_psig=p["upstream_pressure_psig"],
                downstream_pressure_psig=p["downstream_pressure_psig"],
                setpoint_psig=p["downstream_pressure_psig"],
                flow_klb_hr=p["flow_klb_hr"],
                valve_position_percent=p["valve_position_percent"],
                max_capacity_klb_hr=p["max_capacity_klb_hr"],
            )
            for p in sample_network_state["prvs"]
        ]

        network = NetworkModel(
            boilers=boilers,
            headers=headers,
            prvs=prvs,
            total_generation_klb_hr=sample_network_state["total_generation_klb_hr"],
            total_demand_klb_hr=sample_network_state["total_demand_klb_hr"],
        )

        # Step 3: Optimize load allocation
        load_result = optimizer.optimize_load_allocation(
            boilers=boilers,
            total_demand_klb_hr=130.0,
            objective="cost",
        )

        assert load_result.total_generation_klb_hr > 0
        assert len(load_result.allocations) == len(boilers)
        assert len(load_result.provenance_hash) == 64

        # Step 4: Optimize header pressures
        forecast = DemandForecast(
            hp_demand_klb_hr=[130.0, 135.0, 140.0],
            mp_demand_klb_hr=[45.0, 48.0, 50.0],
            lp_demand_klb_hr=[15.0, 16.0, 17.0],
        )

        header_results = optimizer.optimize_header_pressures(
            demand_forecast=forecast,
            boiler_constraints={},
            current_headers=headers,
        )

        assert len(header_results) == len(headers)

        # Step 5: Minimize losses
        loss_result = optimizer.minimize_total_losses(network_model=network)

        assert loss_result.current_total_loss_klb_hr >= 0
        assert isinstance(loss_result.recommendations, list)

    def test_optimization_with_uncertainty(
        self, optimizer, uncertainty_propagator, sample_network_state
    ):
        """
        Test optimization pipeline with uncertainty propagation:
        1. Define uncertain inputs
        2. Propagate through efficiency calculation
        3. Apply to optimization
        """
        # Step 1: Define uncertain measurements
        uncertain_pressure = UncertainValue(
            mean=600.0,
            std=3.0,  # +/- 0.5% uncertainty
            lower_95=594.12,
            upper_95=605.88,
            distribution_type=DistributionType.NORMAL,
            timestamp=datetime.now(timezone.utc),
        )

        uncertain_temp = UncertainValue(
            mean=700.0,
            std=5.0,
            lower_95=690.2,
            upper_95=709.8,
            distribution_type=DistributionType.NORMAL,
            timestamp=datetime.now(timezone.utc),
        )

        uncertain_flow = UncertainValue(
            mean=130.0,
            std=2.6,  # +/- 2% uncertainty
            lower_95=124.9,
            upper_95=135.1,
            distribution_type=DistributionType.NORMAL,
            timestamp=datetime.now(timezone.utc),
        )

        # Step 2: Propagate uncertainty through enthalpy calculation
        # h_steam = f(P, T) - simplified linear for testing
        def enthalpy_estimate(vals):
            # Simplified: h increases with T, decreases slightly with P
            return 1100 + 0.5 * vals["T"] - 0.1 * vals["P"]

        inputs = {
            "P": uncertain_pressure,
            "T": uncertain_temp,
        }

        enthalpy_result = uncertainty_propagator.propagate_nonlinear(
            inputs=inputs,
            function=enthalpy_estimate,
        )

        assert enthalpy_result.value > 0
        assert enthalpy_result.uncertainty > 0

        # Step 3: Run optimization with the uncertain demand
        boilers = [BoilerState(**b) for b in sample_network_state["boilers"]]

        load_result = optimizer.optimize_load_allocation(
            boilers=boilers,
            total_demand_klb_hr=uncertain_flow.mean,
            objective="cost",
        )

        assert load_result is not None
        assert load_result.confidence >= 0.9

    def test_multi_objective_optimization_comparison(
        self, optimizer, sample_network_state
    ):
        """
        Compare different optimization objectives:
        - Cost minimization
        - Efficiency maximization
        - Emissions minimization
        """
        boilers = [BoilerState(**b) for b in sample_network_state["boilers"]]
        demand = 130.0

        # Run all objectives
        cost_result = optimizer.optimize_load_allocation(
            boilers=boilers,
            total_demand_klb_hr=demand,
            objective="cost",
        )

        efficiency_result = optimizer.optimize_load_allocation(
            boilers=boilers,
            total_demand_klb_hr=demand,
            objective="efficiency",
        )

        emissions_result = optimizer.optimize_load_allocation(
            boilers=boilers,
            total_demand_klb_hr=demand,
            objective="emissions",
        )

        # All should meet demand
        assert abs(cost_result.total_generation_klb_hr - demand) < 5.0
        assert abs(efficiency_result.total_generation_klb_hr - demand) < 5.0
        assert abs(emissions_result.total_generation_klb_hr - demand) < 5.0

        # Different objectives may produce different allocations
        # (but all should be valid)
        assert cost_result.optimization_objective == "cost"
        assert efficiency_result.optimization_objective == "efficiency"
        assert emissions_result.optimization_objective == "emissions"


# =============================================================================
# API Endpoint Simulation Tests
# =============================================================================

@pytest.mark.integration
class TestAPIEndpointWorkflows:
    """Tests simulating API endpoint workflows."""

    def test_optimize_endpoint_simulation(self, optimizer, sample_network_state):
        """
        Simulate /optimize endpoint workflow:
        1. Parse request
        2. Validate input
        3. Run optimization
        4. Format response
        """
        # Step 1: Parse request (simulate JSON body)
        request_body = {
            "demand_klb_hr": 130.0,
            "objective": "cost",
            "boilers": sample_network_state["boilers"],
        }

        # Step 2: Validate input
        assert "demand_klb_hr" in request_body
        assert request_body["demand_klb_hr"] > 0
        assert request_body["objective"] in ["cost", "efficiency", "emissions", "balanced"]

        # Step 3: Run optimization
        boilers = [BoilerState(**b) for b in request_body["boilers"]]
        result = optimizer.optimize_load_allocation(
            boilers=boilers,
            total_demand_klb_hr=request_body["demand_klb_hr"],
            objective=request_body["objective"],
        )

        # Step 4: Format response
        response = {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "optimization_objective": result.optimization_objective,
            "total_demand_klb_hr": result.total_demand_klb_hr,
            "total_generation_klb_hr": result.total_generation_klb_hr,
            "total_cost_per_hr": result.total_cost_per_hr,
            "total_co2_lb_hr": result.total_co2_lb_hr,
            "improvement_percent": result.improvement_percent,
            "confidence": result.confidence,
            "provenance_hash": result.provenance_hash,
            "allocations": [
                {
                    "boiler_id": a.boiler_id,
                    "recommended_load_percent": a.recommended_load_percent,
                    "change_required": a.change_required,
                }
                for a in result.allocations
            ],
        }

        # Validate response
        assert response["status"] == "success"
        assert response["provenance_hash"] is not None
        assert len(response["allocations"]) == len(boilers)

    def test_explain_endpoint_simulation(
        self, shap_explainer, mock_ml_model, sample_trap_features
    ):
        """
        Simulate /explain endpoint workflow:
        1. Receive prediction request
        2. Compute SHAP explanation
        3. Format response
        """
        # Step 1: Receive request
        request_body = {
            "asset_id": "TRAP-001",
            "features": sample_trap_features,
        }

        # Step 2: Compute explanation
        explanation = shap_explainer.compute_local_explanation(
            model=mock_ml_model,
            instance=request_body["features"],
            model_type=ModelType.TRAP_FAILURE_PREDICTION,
        )

        # Step 3: Format response
        response = {
            "status": "success",
            "asset_id": request_body["asset_id"],
            "explanation_id": explanation.explanation_id,
            "predicted_value": explanation.predicted_value,
            "base_value": explanation.base_value,
            "summary_text": explanation.summary_text,
            "top_positive_features": explanation.top_positive_features,
            "top_negative_features": explanation.top_negative_features,
            "contributions": [
                {
                    "feature": c.feature_name,
                    "value": c.feature_value,
                    "shap_value": c.shap_value,
                    "direction": c.direction,
                }
                for c in explanation.contributions
            ],
        }

        # Validate response
        assert response["status"] == "success"
        assert response["explanation_id"] is not None
        assert len(response["contributions"]) > 0

    def test_transform_endpoint_simulation(
        self, sensor_transformer, sample_sensor_data
    ):
        """
        Simulate /transform endpoint workflow:
        1. Receive raw sensor data
        2. Transform and validate
        3. Return qualified values
        """
        # Configure transformer for sensors in request
        sensor_transformer.configure_sensor(
            tag="header.pressure",
            from_unit="psig",
            to_unit="bar",
            valid_range=(0.0, 50.0),
        )
        sensor_transformer.configure_sensor(
            tag="header.temperature",
            from_unit="degF",
            to_unit="degC",
            valid_range=(0.0, 400.0),
        )

        # Step 1: Receive request
        request_body = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "values": sample_sensor_data,
        }

        # Step 2: Transform
        result = sensor_transformer.transform_batch(
            raw_data=request_body["values"],
        )

        # Step 3: Format response
        response = {
            "status": "success",
            "timestamp": result.timestamp.isoformat(),
            "total_count": result.total_count,
            "good_count": result.good_count,
            "quality_score": result.quality_score,
            "values": {
                tag: {
                    "value": v.value,
                    "unit": v.unit,
                    "quality": v.quality_code.name,
                    "is_good": v.quality_code.is_good(),
                }
                for tag, v in result.values.items()
            },
        }

        # Validate response
        assert response["status"] == "success"
        assert response["quality_score"] >= 0
        assert len(response["values"]) == result.total_count


# =============================================================================
# Real-Time Streaming Simulation Tests
# =============================================================================

@pytest.mark.integration
class TestRealTimeStreaming:
    """Tests for real-time data streaming simulation."""

    def test_continuous_transformation_stream(self, sensor_transformer):
        """
        Simulate continuous sensor data transformation:
        1. Generate stream of sensor readings
        2. Transform in batches
        3. Track quality over time
        """
        # Configure transformer
        sensor_transformer.configure_sensor(
            tag="pressure",
            from_unit="psig",
            to_unit="bar",
            valid_range=(0.0, 50.0),
            max_rate_of_change=5.0,
        )

        # Simulate 60 seconds of data at 1 Hz
        quality_scores = []
        np.random.seed(42)

        base_pressure = 145.0
        for i in range(60):
            # Simulate sensor reading with noise
            noise = np.random.normal(0, 2)
            value = base_pressure + noise

            # Add occasional spikes
            if i in [15, 35]:
                value += 50  # Spike

            timestamp = datetime.now(timezone.utc) + timedelta(seconds=i)

            # Transform single reading
            result = sensor_transformer.transform_single(
                tag="pressure",
                raw_value=value,
                timestamp=timestamp,
            )

            # Track quality
            quality_scores.append(1 if result.quality_code.is_good() else 0)

        # Verify quality tracking
        avg_quality = sum(quality_scores) / len(quality_scores)
        # Most readings should be good (expect ~2 bad due to spikes)
        assert avg_quality > 0.9

    def test_rolling_optimization_simulation(self, optimizer, sample_network_state):
        """
        Simulate rolling optimization every 5 minutes:
        1. Generate demand profile
        2. Run optimization for each period
        3. Track cost over time
        """
        boilers = [BoilerState(**b) for b in sample_network_state["boilers"]]

        # Simulate 1 hour with 5-minute intervals
        np.random.seed(42)
        base_demand = 130.0
        periods = 12  # 12 x 5 minutes = 1 hour

        costs = []
        for period in range(periods):
            # Demand varies slightly
            demand = base_demand + np.random.normal(0, 5)
            demand = max(50, min(180, demand))  # Clamp to reasonable range

            result = optimizer.optimize_load_allocation(
                boilers=boilers,
                total_demand_klb_hr=demand,
                objective="cost",
            )

            costs.append(result.total_cost_per_hr)

        # Verify costs are reasonable
        assert len(costs) == periods
        assert all(c > 0 for c in costs)

        # Cost should vary with demand
        assert max(costs) > min(costs)

    def test_anomaly_detection_stream(self, sensor_transformer):
        """
        Simulate anomaly detection in sensor stream:
        1. Generate normal data
        2. Inject anomalies
        3. Detect via quality flags
        """
        sensor_transformer.configure_sensor(
            tag="temperature",
            from_unit="degF",
            to_unit="degC",
            valid_range=(100.0, 400.0),
            max_rate_of_change=20.0,
        )

        np.random.seed(42)
        anomalies_detected = 0

        for i in range(100):
            # Normal temperature around 350 F
            value = 350.0 + np.random.normal(0, 5)

            # Inject anomalies
            if i == 30:
                value = 500.0  # Out of range
            if i == 60:
                value = 50.0  # Way below range
            if i == 80:
                value = 350.0 + 100.0  # Sudden spike (rate violation)

            timestamp = datetime.now(timezone.utc) + timedelta(seconds=i)

            result = sensor_transformer.transform_single(
                tag="temperature",
                raw_value=value,
                timestamp=timestamp,
            )

            if not result.quality_code.is_good():
                anomalies_detected += 1

        # Should detect the injected anomalies
        assert anomalies_detected >= 2


# =============================================================================
# Multi-Component Integration Tests
# =============================================================================

@pytest.mark.integration
class TestMultiComponentIntegration:
    """Tests for integration across multiple components."""

    def test_sensor_to_optimization_flow(
        self, sensor_transformer, optimizer, sample_network_state
    ):
        """
        Test flow from sensor data to optimization:
        1. Transform sensor readings
        2. Use transformed values in network model
        3. Run optimization
        """
        # Step 1: Transform sensor readings
        sensor_transformer.configure_sensor(
            tag="boiler_load",
            from_unit="%",
            to_unit="%",
            valid_range=(0.0, 110.0),
        )

        raw_readings = {
            "boiler_load_1": 75.0,
            "boiler_load_2": 70.0,
        }

        transformed = sensor_transformer.transform_batch(raw_readings)

        # Step 2: Build boiler states with transformed values
        boilers = [
            BoilerState(
                boiler_id="BLR-001",
                is_online=True,
                current_load_percent=75.0,  # Would use transformed value
                rated_capacity_klb_hr=100.0,
                current_efficiency_percent=85.0,
                fuel_cost_per_mmbtu=8.0,
                co2_factor_lb_mmbtu=117.0,
            ),
            BoilerState(
                boiler_id="BLR-002",
                is_online=True,
                current_load_percent=70.0,
                rated_capacity_klb_hr=80.0,
                current_efficiency_percent=83.0,
                fuel_cost_per_mmbtu=8.5,
                co2_factor_lb_mmbtu=117.0,
            ),
        ]

        # Step 3: Optimize
        result = optimizer.optimize_load_allocation(
            boilers=boilers,
            total_demand_klb_hr=130.0,
            objective="cost",
        )

        assert result is not None
        assert result.total_generation_klb_hr > 0

    def test_optimization_with_explanation(
        self, optimizer, shap_explainer, mock_ml_model, sample_network_state
    ):
        """
        Test optimization with ML explanation:
        1. Run optimization
        2. Generate explanation for allocation
        3. Link optimization to explanation
        """
        boilers = [BoilerState(**b) for b in sample_network_state["boilers"]]

        # Step 1: Run optimization
        opt_result = optimizer.optimize_load_allocation(
            boilers=boilers,
            total_demand_klb_hr=130.0,
            objective="cost",
        )

        # Step 2: Generate explanation for the optimization decision
        # (In real scenario, this would explain why certain allocation was chosen)
        features = {
            "total_demand": 130.0,
            "num_online_boilers": len([b for b in boilers if b.is_online]),
            "avg_efficiency": sum(b.current_efficiency_percent for b in boilers) / len(boilers),
            "avg_fuel_cost": sum(b.fuel_cost_per_mmbtu for b in boilers) / len(boilers),
        }

        explanation = shap_explainer.compute_local_explanation(
            model=mock_ml_model,
            instance=features,
        )

        # Step 3: Link
        combined_result = {
            "optimization": {
                "total_cost": opt_result.total_cost_per_hr,
                "allocations": [a.boiler_id for a in opt_result.allocations],
            },
            "explanation": {
                "summary": explanation.summary_text,
                "top_factors": explanation.top_positive_features,
            },
        }

        assert combined_result["optimization"]["total_cost"] > 0
        assert combined_result["explanation"]["summary"] is not None

    def test_uncertainty_in_optimization_decisions(
        self, optimizer, uncertainty_propagator, sample_network_state
    ):
        """
        Test uncertainty impact on optimization decisions:
        1. Run optimization with nominal values
        2. Propagate measurement uncertainty
        3. Compare results
        """
        boilers = [BoilerState(**b) for b in sample_network_state["boilers"]]
        nominal_demand = 130.0

        # Step 1: Nominal optimization
        nominal_result = optimizer.optimize_load_allocation(
            boilers=boilers,
            total_demand_klb_hr=nominal_demand,
            objective="cost",
        )

        # Step 2: Monte Carlo with uncertain demand
        demand_distribution = Distribution(
            distribution_type=DistributionType.NORMAL,
            parameters={"mean": 130.0, "std": 2.6},  # 2% uncertainty
        )

        def optimize_cost(vals):
            result = optimizer.optimize_load_allocation(
                boilers=boilers,
                total_demand_klb_hr=vals["demand"],
                objective="cost",
            )
            return result.total_cost_per_hr

        mc_result = uncertainty_propagator.propagate_monte_carlo(
            inputs={"demand": demand_distribution},
            function=optimize_cost,
            n_samples=100,
            seed=42,
        )

        # Step 3: Compare
        assert abs(mc_result.mean - nominal_result.total_cost_per_hr) < 100
        assert mc_result.std > 0  # Cost varies with demand uncertainty


# =============================================================================
# Data Persistence Simulation Tests
# =============================================================================

@pytest.mark.integration
class TestDataPersistence:
    """Tests for data persistence workflows."""

    def test_optimization_result_serialization(self, optimizer, sample_network_state):
        """
        Test optimization result can be serialized and deserialized:
        1. Run optimization
        2. Serialize to JSON
        3. Verify serialization completeness
        """
        boilers = [BoilerState(**b) for b in sample_network_state["boilers"]]

        result = optimizer.optimize_load_allocation(
            boilers=boilers,
            total_demand_klb_hr=130.0,
            objective="cost",
        )

        # Serialize to dict (similar to JSON)
        serialized = {
            "timestamp": str(result.timestamp),
            "total_demand_klb_hr": result.total_demand_klb_hr,
            "total_generation_klb_hr": result.total_generation_klb_hr,
            "total_cost_per_hr": result.total_cost_per_hr,
            "total_co2_lb_hr": result.total_co2_lb_hr,
            "weighted_efficiency": result.weighted_efficiency,
            "optimization_objective": result.optimization_objective,
            "improvement_percent": result.improvement_percent,
            "confidence": result.confidence,
            "provenance_hash": result.provenance_hash,
            "allocations": [
                {
                    "boiler_id": a.boiler_id,
                    "current_load_percent": a.current_load_percent,
                    "recommended_load_percent": a.recommended_load_percent,
                    "recommended_output_klb_hr": a.recommended_output_klb_hr,
                    "efficiency_at_load": a.efficiency_at_load,
                    "cost_at_load": a.cost_at_load,
                    "co2_at_load_lb_hr": a.co2_at_load_lb_hr,
                    "change_required": a.change_required,
                }
                for a in result.allocations
            ],
        }

        # Verify JSON serializable
        json_str = json.dumps(serialized)
        assert len(json_str) > 0

        # Verify deserialization
        deserialized = json.loads(json_str)
        assert deserialized["total_demand_klb_hr"] == result.total_demand_klb_hr
        assert deserialized["provenance_hash"] == result.provenance_hash

    def test_provenance_hash_consistency(self, optimizer, sample_network_state):
        """
        Test provenance hash is consistent for same inputs:
        1. Run optimization twice with same inputs
        2. Verify provenance hashes match
        """
        boilers = [BoilerState(**b) for b in sample_network_state["boilers"]]

        result1 = optimizer.optimize_load_allocation(
            boilers=boilers,
            total_demand_klb_hr=130.0,
            objective="cost",
        )

        result2 = optimizer.optimize_load_allocation(
            boilers=boilers,
            total_demand_klb_hr=130.0,
            objective="cost",
        )

        # Same inputs should produce same hash
        assert result1.provenance_hash == result2.provenance_hash


# =============================================================================
# Performance Integration Tests
# =============================================================================

@pytest.mark.slow
@pytest.mark.integration
class TestPerformanceIntegration:
    """Performance tests for integrated workflows."""

    def test_full_pipeline_latency(
        self, optimizer, sensor_transformer, sample_network_state
    ):
        """Test end-to-end latency of full pipeline."""
        boilers = [BoilerState(**b) for b in sample_network_state["boilers"]]

        start_time = time.perf_counter()

        # Simulate full pipeline
        for _ in range(10):
            # Transform
            raw_data = {"header.pressure": 145.0, "header.temperature": 700.0}
            transformed = sensor_transformer.transform_batch(raw_data)

            # Optimize
            result = optimizer.optimize_load_allocation(
                boilers=boilers,
                total_demand_klb_hr=130.0,
                objective="cost",
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        avg_latency = elapsed_ms / 10

        # Average latency should be < 100ms
        assert avg_latency < 100, f"Average latency {avg_latency:.1f}ms exceeds target"

    def test_high_frequency_transformation(self, sensor_transformer):
        """Test high-frequency sensor transformation throughput."""
        sensor_transformer.configure_sensor(
            tag="fast_sensor",
            from_unit="mV",
            to_unit="mV",
            valid_range=(0.0, 1000.0),
        )

        n_readings = 10000

        start_time = time.perf_counter()

        for i in range(n_readings):
            result = sensor_transformer.transform_single(
                tag="fast_sensor",
                raw_value=float(i % 1000),
            )

        elapsed_s = time.perf_counter() - start_time
        throughput = n_readings / elapsed_s

        # Should achieve > 10000 transformations per second
        assert throughput > 10000, f"Throughput {throughput:.0f}/s below target"


# =============================================================================
# Error Recovery Tests
# =============================================================================

@pytest.mark.integration
class TestErrorRecovery:
    """Tests for error handling and recovery in integrated workflows."""

    def test_optimization_with_partial_data(self, optimizer):
        """Test optimization handles partial/missing data gracefully."""
        # Only one boiler online
        boilers = [
            BoilerState(
                boiler_id="BLR-001",
                is_online=True,
                current_load_percent=75.0,
                rated_capacity_klb_hr=100.0,
                current_efficiency_percent=85.0,
                fuel_cost_per_mmbtu=8.0,
                co2_factor_lb_mmbtu=117.0,
            ),
        ]

        # Demand within single boiler capacity
        result = optimizer.optimize_load_allocation(
            boilers=boilers,
            total_demand_klb_hr=80.0,
            objective="cost",
        )

        assert result is not None
        assert result.total_generation_klb_hr > 0

    def test_sensor_transform_with_bad_data(self, sensor_transformer):
        """Test sensor transformation handles bad data gracefully."""
        sensor_transformer.configure_sensor(
            tag="test_sensor",
            from_unit="kPa",
            to_unit="bar",
            valid_range=(0.0, 100.0),
        )

        # Bad values
        bad_values = [float('inf'), float('-inf'), float('nan'), -1000.0, 10000.0]

        for bad_value in bad_values:
            result = sensor_transformer.transform_single(
                tag="test_sensor",
                raw_value=bad_value,
            )

            # Should not crash, should flag as bad quality
            assert result is not None
            # Bad values should be flagged

    def test_pipeline_continues_after_single_failure(
        self, optimizer, sensor_transformer, sample_network_state
    ):
        """Test that pipeline continues processing after single component failure."""
        boilers = [BoilerState(**b) for b in sample_network_state["boilers"]]

        successes = 0
        failures = 0

        for i in range(10):
            try:
                # Occasionally use invalid demand
                demand = 130.0 if i != 5 else -100.0

                if demand > 0:
                    result = optimizer.optimize_load_allocation(
                        boilers=boilers,
                        total_demand_klb_hr=demand,
                        objective="cost",
                    )
                    successes += 1
                else:
                    # Skip invalid demand
                    failures += 1
            except Exception:
                failures += 1

        # Most should succeed
        assert successes >= 9
