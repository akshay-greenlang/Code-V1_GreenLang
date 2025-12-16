# -*- coding: utf-8 -*-
"""
GL-021 BURNERSENTRY Configuration Tests

Tests for configuration validation and defaults.
"""

import pytest
from pydantic import ValidationError


class TestBurnerMaintenanceConfig:
    """Test configuration classes."""

    def test_config_import(self):
        """Test configuration module imports."""
        from greenlang.agents.process_heat.gl_021_burner_maintenance.config import (
            GL021Config,
        )
        assert GL021Config is not None

    def test_default_config_creation(self):
        """Test default configuration creation."""
        from greenlang.agents.process_heat.gl_021_burner_maintenance.config import (
            GL021Config,
        )
        config = GL021Config()
        assert config is not None
        assert config.agent_id == "GL-021"
        assert config.agent_name == "BURNERSENTRY"

    def test_config_with_custom_values(self):
        """Test configuration with custom values."""
        from greenlang.agents.process_heat.gl_021_burner_maintenance.config import (
            GL021Config,
        )
        config = GL021Config(
            enable_shap=True,
            enable_lime=True,
            weibull_analysis_enabled=True,
            cox_hazards_enabled=True,
        )
        assert config.enable_shap is True
        assert config.enable_lime is True

    def test_config_safety_settings(self):
        """Test safety-related configuration."""
        from greenlang.agents.process_heat.gl_021_burner_maintenance.config import (
            GL021Config,
        )
        config = GL021Config()
        # NFPA 85/86 compliance defaults
        assert hasattr(config, 'nfpa_85_compliance')
        assert hasattr(config, 'nfpa_86_compliance')

    def test_config_thresholds(self):
        """Test threshold configurations."""
        from greenlang.agents.process_heat.gl_021_burner_maintenance.config import (
            GL021Config,
        )
        config = GL021Config()
        assert hasattr(config, 'health_score_warning_threshold')
        assert hasattr(config, 'health_score_critical_threshold')
        assert config.health_score_warning_threshold >= config.health_score_critical_threshold

    def test_config_serialization(self):
        """Test configuration JSON serialization."""
        from greenlang.agents.process_heat.gl_021_burner_maintenance.config import (
            GL021Config,
        )
        config = GL021Config()
        json_str = config.json()
        assert "agent_id" in json_str
        assert "GL-021" in json_str

    def test_config_validation_invalid_threshold(self):
        """Test configuration validation for invalid thresholds."""
        from greenlang.agents.process_heat.gl_021_burner_maintenance.config import (
            GL021Config,
        )
        # Should handle or reject invalid thresholds
        with pytest.raises((ValidationError, ValueError)):
            GL021Config(
                health_score_warning_threshold=-10.0,  # Invalid negative
            )


class TestComponentWeightConfig:
    """Test component weight configurations."""

    def test_component_weights_sum_to_one(self):
        """Test that default component weights sum to 1.0."""
        from greenlang.agents.process_heat.gl_021_burner_maintenance.explainability import (
            HealthScoreExplainer,
        )
        explainer = HealthScoreExplainer()
        total = sum(explainer.component_weights.values())
        assert abs(total - 1.0) < 0.001

    def test_custom_component_weights(self):
        """Test custom component weight configuration."""
        from greenlang.agents.process_heat.gl_021_burner_maintenance.explainability import (
            HealthScoreExplainer,
            BurnerComponent,
        )
        custom_weights = {
            BurnerComponent.FLAME_SCANNER: 0.20,
            BurnerComponent.IGNITOR: 0.15,
            BurnerComponent.FUEL_VALVE: 0.20,
            BurnerComponent.AIR_DAMPER: 0.10,
            BurnerComponent.PILOT_ASSEMBLY: 0.10,
            BurnerComponent.MAIN_BURNER: 0.10,
            BurnerComponent.COMBUSTION_AIR_FAN: 0.05,
            BurnerComponent.GAS_TRAIN: 0.05,
            BurnerComponent.FLAME_STABILITY: 0.025,
            BurnerComponent.EMISSION_QUALITY: 0.025,
        }
        explainer = HealthScoreExplainer(component_weights=custom_weights)
        assert explainer.component_weights[BurnerComponent.FLAME_SCANNER] == 0.20

    def test_invalid_component_weights_rejected(self):
        """Test that invalid component weights are rejected."""
        from greenlang.agents.process_heat.gl_021_burner_maintenance.explainability import (
            HealthScoreExplainer,
            BurnerComponent,
        )
        invalid_weights = {
            BurnerComponent.FLAME_SCANNER: 0.50,  # Only one component
        }
        # Weights don't sum to 1.0
        with pytest.raises(ValueError):
            HealthScoreExplainer(component_weights=invalid_weights)


class TestDegradationRateConfig:
    """Test degradation rate configurations."""

    def test_degradation_rates_defined(self):
        """Test that degradation rates are defined for all components."""
        from greenlang.agents.process_heat.gl_021_burner_maintenance.explainability import (
            HealthScoreExplainer,
            BurnerComponent,
        )
        for component in BurnerComponent:
            assert component in HealthScoreExplainer.DEGRADATION_RATES

    def test_degradation_rates_positive(self):
        """Test that all degradation rates are positive."""
        from greenlang.agents.process_heat.gl_021_burner_maintenance.explainability import (
            HealthScoreExplainer,
        )
        for rate in HealthScoreExplainer.DEGRADATION_RATES.values():
            assert rate > 0
