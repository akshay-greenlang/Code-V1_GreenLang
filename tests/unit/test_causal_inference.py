"""
Unit tests for causal inference module.

Tests cover:
- CausalInference core functionality
- ProcessHeatCausalModels factory methods
- Causal graph construction
- Effect estimation and robustness
- Counterfactual predictions
- Confounding detection
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

from greenlang.ml.explainability.causal_inference import (
    CausalInference,
    CausalInferenceConfig,
    CausalEffectResult,
    CounterfactualResult,
    ProcessHeatCausalModels,
    IdentificationMethod,
    EstimationMethod,
    RefutationMethod
)


class TestCausalInferenceConfig:
    """Tests for CausalInferenceConfig."""

    def test_config_creation_minimal(self):
        """Test creating config with minimal required fields."""
        config = CausalInferenceConfig(
            treatment="treatment",
            outcome="outcome"
        )

        assert config.treatment == "treatment"
        assert config.outcome == "outcome"
        assert config.identification_method == IdentificationMethod.BACKDOOR
        assert config.confidence_level == 0.95

    def test_config_creation_full(self):
        """Test creating config with all fields."""
        config = CausalInferenceConfig(
            treatment="T",
            outcome="Y",
            common_causes=["X1", "X2"],
            instruments=["Z"],
            effect_modifiers=["M"],
            identification_method=IdentificationMethod.IV,
            estimation_method=EstimationMethod.PROPENSITY_SCORE_MATCHING,
            confidence_level=0.99,
            n_bootstrap=500,
            random_state=123
        )

        assert config.treatment == "T"
        assert config.outcome == "Y"
        assert len(config.common_causes) == 2
        assert config.instruments == ["Z"]
        assert config.confidence_level == 0.99
        assert config.n_bootstrap == 500

    def test_config_invalid_confidence_level(self):
        """Test that invalid confidence level raises error."""
        with pytest.raises(ValueError):
            CausalInferenceConfig(
                treatment="T",
                outcome="Y",
                confidence_level=0.5  # Too low
            )

        with pytest.raises(ValueError):
            CausalInferenceConfig(
                treatment="T",
                outcome="Y",
                confidence_level=1.5  # Too high
            )


class TestCausalInferenceInit:
    """Tests for CausalInference initialization."""

    @staticmethod
    def create_test_data():
        """Create simple test dataset."""
        return pd.DataFrame({
            "treatment": [0, 1, 0, 1, 0, 1] * 10,
            "outcome": [1, 2, 1.5, 2.5, 1, 2] * 10,
            "confounder1": [1, 1, 2, 2, 3, 3] * 10,
            "confounder2": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 10
        })

    def test_init_valid_data(self):
        """Test initialization with valid data."""
        data = self.create_test_data()
        config = CausalInferenceConfig(
            treatment="treatment",
            outcome="outcome",
            common_causes=["confounder1", "confounder2"]
        )

        ci = CausalInference(data, config)

        assert ci.config.treatment == "treatment"
        assert ci.config.outcome == "outcome"
        assert len(ci.data) == 60
        assert ci._model is None

    def test_init_missing_treatment_column(self):
        """Test that missing treatment column raises error."""
        data = pd.DataFrame({
            "outcome": [1, 2, 3],
            "confounder": [1, 1, 2]
        })

        config = CausalInferenceConfig(
            treatment="missing_treatment",
            outcome="outcome"
        )

        with pytest.raises(ValueError, match="Missing columns"):
            CausalInference(data, config)

    def test_init_missing_outcome_column(self):
        """Test that missing outcome column raises error."""
        data = pd.DataFrame({
            "treatment": [0, 1, 0],
            "confounder": [1, 1, 2]
        })

        config = CausalInferenceConfig(
            treatment="treatment",
            outcome="missing_outcome"
        )

        with pytest.raises(ValueError, match="Missing columns"):
            CausalInference(data, config)

    def test_init_missing_confounder_column(self):
        """Test that missing confounder raises error."""
        data = pd.DataFrame({
            "treatment": [0, 1, 0],
            "outcome": [1, 2, 1]
        })

        config = CausalInferenceConfig(
            treatment="treatment",
            outcome="outcome",
            common_causes=["missing_confounder"]
        )

        with pytest.raises(ValueError, match="Missing columns"):
            CausalInference(data, config)


class TestCausalGraphConstruction:
    """Tests for causal graph construction."""

    @staticmethod
    def create_test_data():
        """Create test data."""
        return pd.DataFrame({
            "T": [0, 1, 0, 1],
            "Y": [1, 2, 1, 2],
            "X": [1, 2, 1, 2],
            "Z": [0.5, 0.6, 0.7, 0.8]
        })

    def test_graph_backdoor_confounding(self):
        """Test causal graph with backdoor confounding."""
        data = self.create_test_data()
        config = CausalInferenceConfig(
            treatment="T",
            outcome="Y",
            common_causes=["X"]
        )

        ci = CausalInference(data, config)
        graph = ci._build_causal_graph()

        # Check essential edges
        assert '"T" -> "Y"' in graph
        assert '"X" -> "T"' in graph
        assert '"X" -> "Y"' in graph

    def test_graph_instrumental_variable(self):
        """Test causal graph with instrumental variable."""
        data = self.create_test_data()
        config = CausalInferenceConfig(
            treatment="T",
            outcome="Y",
            instruments=["Z"]
        )

        ci = CausalInference(data, config)
        graph = ci._build_causal_graph()

        # Check instrument edge
        assert '"Z" -> "T"' in graph
        assert '"T" -> "Y"' in graph

    def test_graph_effect_modifier(self):
        """Test causal graph with effect modifier."""
        data = self.create_test_data()
        config = CausalInferenceConfig(
            treatment="T",
            outcome="Y",
            effect_modifiers=["X"]
        )

        ci = CausalInference(data, config)
        graph = ci._build_causal_graph()

        # Effect modifier affects outcome
        assert '"X" -> "Y"' in graph

    def test_graph_get_causal_graph(self):
        """Test getting causal graph."""
        data = self.create_test_data()
        config = CausalInferenceConfig(
            treatment="T",
            outcome="Y",
            common_causes=["X"]
        )

        ci = CausalInference(data, config)
        graph = ci.get_causal_graph()

        assert isinstance(graph, str)
        assert "digraph" in graph
        assert '"T" -> "Y"' in graph


class TestProvenanceTracking:
    """Tests for provenance hash tracking."""

    @staticmethod
    def create_test_data():
        """Create test data."""
        return pd.DataFrame({
            "T": [0, 1, 0, 1],
            "Y": [1, 2, 1, 2]
        })

    def test_provenance_hash_deterministic(self):
        """Test that provenance hash is deterministic."""
        data = self.create_test_data()
        config = CausalInferenceConfig(treatment="T", outcome="Y")

        ci = CausalInference(data, config)

        hash1 = ci._calculate_provenance(0.5, "linear_regression")
        hash2 = ci._calculate_provenance(0.5, "linear_regression")

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length

    def test_provenance_hash_different_estimates(self):
        """Test that different estimates produce different hashes."""
        data = self.create_test_data()
        config = CausalInferenceConfig(treatment="T", outcome="Y")

        ci = CausalInference(data, config)

        hash1 = ci._calculate_provenance(0.5, "linear_regression")
        hash2 = ci._calculate_provenance(0.6, "linear_regression")

        assert hash1 != hash2

    def test_provenance_hash_different_methods(self):
        """Test that different methods produce different hashes."""
        data = self.create_test_data()
        config = CausalInferenceConfig(treatment="T", outcome="Y")

        ci = CausalInference(data, config)

        hash1 = ci._calculate_provenance(0.5, "linear_regression")
        hash2 = ci._calculate_provenance(0.5, "propensity_score")

        assert hash1 != hash2


class TestBootstrapConfidenceInterval:
    """Tests for bootstrap confidence interval calculation."""

    @staticmethod
    def create_test_data():
        """Create test data with clear signal."""
        np.random.seed(42)
        n = 100
        treatment = np.repeat([0, 1], n // 2)
        outcome = treatment * 2 + np.random.normal(0, 0.5, n)

        return pd.DataFrame({
            "treatment": treatment,
            "outcome": outcome
        })

    def test_bootstrap_ci_reasonable_bounds(self):
        """Test that bootstrap CI produces reasonable bounds."""
        data = self.create_test_data()
        config = CausalInferenceConfig(
            treatment="treatment",
            outcome="outcome",
            n_bootstrap=100,
            confidence_level=0.95
        )

        ci = CausalInference(data, config)
        lower, upper = ci._bootstrap_confidence_interval(2.0)

        # CI should contain the true effect (2.0)
        assert lower < 2.0 < upper
        # CI should be symmetric-ish around point estimate
        assert (2.0 - lower) > 0
        assert (upper - 2.0) > 0

    def test_bootstrap_ci_width_decreases_with_samples(self):
        """Test that CI width decreases with more bootstrap samples."""
        data = self.create_test_data()

        config_100 = CausalInferenceConfig(
            treatment="treatment",
            outcome="outcome",
            n_bootstrap=100
        )
        ci_100 = CausalInference(data, config_100)
        lower_100, upper_100 = ci_100._bootstrap_confidence_interval(2.0)

        config_500 = CausalInferenceConfig(
            treatment="treatment",
            outcome="outcome",
            n_bootstrap=500
        )
        ci_500 = CausalInference(data, config_500)
        lower_500, upper_500 = ci_500._bootstrap_confidence_interval(2.0)

        # More samples should give tighter CI
        width_100 = upper_100 - lower_100
        width_500 = upper_500 - lower_500

        assert width_500 <= width_100


class TestCounterfactualAnalysis:
    """Tests for counterfactual prediction."""

    @staticmethod
    def create_test_data():
        """Create test data with known relationship."""
        np.random.seed(42)
        n = 50
        treatment = np.random.uniform(0, 1, n)
        outcome = 1 + 2 * treatment + np.random.normal(0, 0.1, n)

        return pd.DataFrame({
            "treatment": treatment,
            "outcome": outcome
        })

    def test_counterfactual_result_structure(self):
        """Test that counterfactual result has expected structure."""
        data = self.create_test_data()
        config = CausalInferenceConfig(treatment="treatment", outcome="outcome")

        ci = CausalInference(data, config)

        instance = {"treatment": 0.5, "outcome": 2.0}
        result = ci.estimate_counterfactual(instance, treatment_value=0.7)

        assert isinstance(result, CounterfactualResult)
        assert result.original_outcome == 2.0
        assert result.treatment_value == 0.7
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] <= result.confidence_interval[1]
        assert len(result.provenance_hash) == 64

    def test_counterfactual_outcome_changes_with_treatment(self):
        """Test that counterfactual outcome changes with treatment value."""
        data = self.create_test_data()
        config = CausalInferenceConfig(treatment="treatment", outcome="outcome")

        ci = CausalInference(data, config)

        instance = {"treatment": 0.5, "outcome": 2.0}

        # Lower treatment value
        result_low = ci.estimate_counterfactual(instance, treatment_value=0.3)

        # Higher treatment value
        result_high = ci.estimate_counterfactual(instance, treatment_value=0.7)

        # With positive ATE, higher treatment should give higher outcome
        assert result_high.counterfactual_outcome >= result_low.counterfactual_outcome


class TestProcessHeatExcessAirModel:
    """Tests for process heat excess air model."""

    @staticmethod
    def create_process_heat_data():
        """Create realistic process heat data."""
        np.random.seed(42)
        n = 100

        excess_air = np.random.uniform(0.15, 0.35, n)
        burner_age = np.random.uniform(1, 15, n)
        fuel_type = np.random.choice(["gas", "oil"], n)
        ambient_temp = np.random.uniform(10, 30, n)

        # Efficiency decreases with excess air
        efficiency = (
            0.90 -
            (excess_air - 0.15) * 0.3 -
            burner_age * 0.01 +
            (fuel_type == "gas").astype(float) * 0.02 +
            np.random.normal(0, 0.02, n)
        )

        return pd.DataFrame({
            "excess_air_ratio": excess_air,
            "efficiency": efficiency,
            "fuel_type": fuel_type,
            "burner_age": burner_age,
            "ambient_temp": ambient_temp
        })

    def test_excess_air_model_creation(self):
        """Test creating excess air efficiency model."""
        data = self.create_process_heat_data()

        model = ProcessHeatCausalModels.excess_air_efficiency_model(data)

        assert model.config.treatment == "excess_air_ratio"
        assert model.config.outcome == "efficiency"
        assert "fuel_type" in model.config.common_causes

    def test_excess_air_model_causal_graph(self):
        """Test excess air model causal graph structure."""
        data = self.create_process_heat_data()

        model = ProcessHeatCausalModels.excess_air_efficiency_model(data)
        graph = model.get_causal_graph()

        # Check treatment -> outcome
        assert '"excess_air_ratio" -> "efficiency"' in graph


class TestProcessHeatMaintenanceModel:
    """Tests for process heat maintenance model."""

    @staticmethod
    def create_maintenance_data():
        """Create maintenance data."""
        np.random.seed(42)
        n = 100

        maintenance_freq = np.random.uniform(1, 8, n)
        equipment_age = np.random.uniform(1, 20, n)
        utilization = np.random.uniform(0.5, 0.95, n)

        # Failure decreases with maintenance
        failure_prob = (
            0.15 +
            equipment_age * 0.003 +
            (utilization - 0.5) * 0.05 -
            maintenance_freq * 0.01 +
            np.random.normal(0, 0.01, n)
        )

        return pd.DataFrame({
            "maintenance_frequency": maintenance_freq,
            "failure_probability": np.clip(failure_prob, 0, 1),
            "equipment_age": equipment_age,
            "utilization": utilization,
            "maintenance_cost": maintenance_freq * 2000
        })

    def test_maintenance_model_creation(self):
        """Test creating maintenance model."""
        data = self.create_maintenance_data()

        model = ProcessHeatCausalModels.maintenance_failure_model(data)

        assert model.config.treatment == "maintenance_frequency"
        assert model.config.outcome == "failure_probability"

    def test_maintenance_model_confounders(self):
        """Test maintenance model identifies correct confounders."""
        data = self.create_maintenance_data()

        model = ProcessHeatCausalModels.maintenance_failure_model(data)
        confounders = model.get_confounders()

        assert "equipment_age" in confounders
        assert "utilization" in confounders


class TestProcessHeatLoadModel:
    """Tests for process heat load model."""

    @staticmethod
    def create_load_data():
        """Create load-emissions data."""
        np.random.seed(42)
        n = 100

        load_change = np.random.uniform(-30, 30, n)
        demand = np.random.choice(["peak", "off-peak"], n)
        weather = np.random.uniform(10, 35, n)
        fuel = np.random.choice(["gas", "oil"], n)
        efficiency = np.random.uniform(0.78, 0.88, n)

        # Emissions increase with load
        emissions = (
            100 +
            load_change * 1.5 +
            (demand == "peak").astype(float) * 20 -
            efficiency * 50 +
            np.random.normal(0, 5, n)
        )

        return pd.DataFrame({
            "steam_load_change": load_change,
            "co2_emissions": np.clip(emissions, 50, 300),
            "demand_pattern": demand,
            "weather_temp": weather,
            "fuel_type": fuel,
            "boiler_efficiency": efficiency
        })

    def test_load_model_creation(self):
        """Test creating load-emissions model."""
        data = self.create_load_data()

        model = ProcessHeatCausalModels.load_changes_emissions_model(data)

        assert model.config.treatment == "steam_load_change"
        assert model.config.outcome == "co2_emissions"


class TestProcessHeatFoulingModel:
    """Tests for process heat fouling model."""

    @staticmethod
    def create_fouling_data():
        """Create fouling data."""
        np.random.seed(42)
        n = 100

        fouling = np.random.uniform(0.5, 3.5, n)
        water_qual = np.random.choice(["good", "fair", "poor"], n)
        op_hours = np.random.uniform(1000, 50000, n)
        temp = np.random.uniform(75, 100, n)
        velocity = np.random.uniform(1.0, 2.5, n)

        # Heat transfer decreases with fouling
        u = (
            1800 -
            fouling * 150 -
            (op_hours / 10000) * 40 +
            (water_qual == "good").astype(float) * 80 +
            temp * 1.5 +
            velocity * 80 +
            np.random.normal(0, 25, n)
        )

        return pd.DataFrame({
            "fouling_mm": fouling,
            "heat_transfer_coef": np.clip(u, 700, 1800),
            "water_quality": water_qual,
            "operating_hours": op_hours,
            "temperature": temp,
            "fluid_velocity": velocity
        })

    def test_fouling_model_creation(self):
        """Test creating fouling model."""
        data = self.create_fouling_data()

        model = ProcessHeatCausalModels.fouling_heat_transfer_model(data)

        assert model.config.treatment == "fouling_mm"
        assert model.config.outcome == "heat_transfer_coef"

    def test_fouling_model_confounders(self):
        """Test fouling model confounders."""
        data = self.create_fouling_data()

        model = ProcessHeatCausalModels.fouling_heat_transfer_model(data)
        confounders = model.get_confounders()

        assert "water_quality" in confounders
        assert "operating_hours" in confounders
        assert "temperature" in confounders


class TestCausalEffectResult:
    """Tests for CausalEffectResult."""

    def test_result_creation(self):
        """Test creating causal effect result."""
        result = CausalEffectResult(
            average_treatment_effect=1.5,
            confidence_interval=(1.2, 1.8),
            standard_error=0.15,
            p_value=0.001,
            identification_method="backdoor",
            estimation_method="linear_regression",
            refutation_results={},
            is_robust=True,
            provenance_hash="abc123",
            processing_time_ms=100.5,
            n_samples=100
        )

        assert result.average_treatment_effect == 1.5
        assert result.confidence_interval == (1.2, 1.8)
        assert result.is_robust is True

    def test_result_timestamp_auto_generated(self):
        """Test that timestamp is auto-generated."""
        result = CausalEffectResult(
            average_treatment_effect=1.0,
            confidence_interval=(0.8, 1.2),
            standard_error=0.1,
            identification_method="backdoor",
            estimation_method="linear_regression",
            refutation_results={},
            is_robust=True,
            provenance_hash="abc",
            processing_time_ms=50,
            n_samples=100
        )

        assert isinstance(result.timestamp, datetime)


class TestCounterfactualResult:
    """Tests for CounterfactualResult."""

    def test_counterfactual_result_creation(self):
        """Test creating counterfactual result."""
        result = CounterfactualResult(
            original_outcome=2.0,
            counterfactual_outcome=2.5,
            individual_treatment_effect=0.5,
            treatment_value=0.7,
            confidence_interval=(2.3, 2.7),
            provenance_hash="def456"
        )

        assert result.original_outcome == 2.0
        assert result.counterfactual_outcome == 2.5
        assert result.individual_treatment_effect == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
