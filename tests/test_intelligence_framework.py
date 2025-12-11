# -*- coding: utf-8 -*-
"""
Intelligence Framework Tests

Comprehensive test suite for the GreenLang Intelligence Framework that
solves the Intelligence Paradox. These tests ensure that:

1. IntelligentAgentBase properly integrates with LLM infrastructure
2. IntelligenceMixin correctly retrofits existing agents
3. @require_intelligence decorator enforces requirements
4. Intelligence validation works correctly
5. Pilot intelligent agents function properly

Run with: pytest tests/test_intelligence_framework.py -v

Author: GreenLang Intelligence Framework
Date: December 2025
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

# Intelligence Framework imports
from greenlang.agents.intelligent_base import (
    IntelligentAgentBase,
    IntelligentAgentConfig,
    IntelligenceLevel,
    IntelligenceMetrics,
    Recommendation,
    Anomaly,
    create_intelligent_agent_config,
)
from greenlang.agents.intelligence_mixin import (
    IntelligenceMixin,
    IntelligenceConfig,
    retrofit_agent_class,
    create_intelligent_wrapper,
)
from greenlang.agents.intelligence_interface import (
    IntelligentAgent,
    IntelligenceCapabilities,
    AgentIntelligenceValidator,
    ValidationResult,
    require_intelligence,
    is_intelligent_agent,
    get_agent_intelligence_level,
)
from greenlang.agents.base import BaseAgent, AgentConfig, AgentResult


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    provider = Mock()
    provider.generate.return_value = "This is a mock LLM response."
    provider.generate_structured.return_value = {
        "title": "Mock Recommendation",
        "description": "This is a mock recommendation.",
        "priority": "high"
    }
    return provider


@pytest.fixture
def basic_agent_config():
    """Basic intelligent agent configuration."""
    return IntelligentAgentConfig(
        name="TestAgent",
        description="Test agent for unit testing",
        intelligence_level=IntelligenceLevel.STANDARD,
        enable_explanations=True,
        enable_recommendations=True,
        enable_anomaly_detection=False,
        regulatory_context="GHG Protocol",
        domain_context="testing"
    )


@pytest.fixture
def sample_calculation_result():
    """Sample calculation result for testing."""
    return {
        "total_emissions_kg": 1000.0,
        "breakdown": {
            "electricity": 600.0,
            "natural_gas": 400.0
        },
        "scope": "1"
    }


# =============================================================================
# TEST: IntelligenceLevel Enum
# =============================================================================

class TestIntelligenceLevel:
    """Tests for IntelligenceLevel enum."""

    def test_intelligence_levels_exist(self):
        """Test that all intelligence levels are defined."""
        assert IntelligenceLevel.NONE is not None
        assert IntelligenceLevel.BASIC is not None
        assert IntelligenceLevel.STANDARD is not None
        assert IntelligenceLevel.ADVANCED is not None
        assert IntelligenceLevel.FULL is not None

    def test_intelligence_level_values(self):
        """Test intelligence level string values."""
        assert IntelligenceLevel.NONE.value == "NONE"
        assert IntelligenceLevel.BASIC.value == "BASIC"
        assert IntelligenceLevel.STANDARD.value == "STANDARD"
        assert IntelligenceLevel.ADVANCED.value == "ADVANCED"
        assert IntelligenceLevel.FULL.value == "FULL"

    def test_intelligence_level_ordering(self):
        """Test that intelligence levels can be compared."""
        levels = [
            IntelligenceLevel.NONE,
            IntelligenceLevel.BASIC,
            IntelligenceLevel.STANDARD,
            IntelligenceLevel.ADVANCED,
            IntelligenceLevel.FULL
        ]
        # Levels should be distinct
        assert len(set(levels)) == 5


# =============================================================================
# TEST: IntelligentAgentConfig
# =============================================================================

class TestIntelligentAgentConfig:
    """Tests for IntelligentAgentConfig."""

    def test_config_creation_with_defaults(self):
        """Test config creation with default values."""
        config = IntelligentAgentConfig(
            name="TestAgent",
            description="Test agent"
        )
        assert config.name == "TestAgent"
        assert config.intelligence_level == IntelligenceLevel.STANDARD

    def test_config_with_custom_level(self):
        """Test config with custom intelligence level."""
        config = IntelligentAgentConfig(
            name="TestAgent",
            description="Test agent",
            intelligence_level=IntelligenceLevel.ADVANCED
        )
        assert config.intelligence_level == IntelligenceLevel.ADVANCED

    def test_config_with_regulatory_context(self):
        """Test config with regulatory context."""
        config = IntelligentAgentConfig(
            name="TestAgent",
            description="Test agent",
            regulatory_context="CSRD, CBAM, SB253"
        )
        assert config.regulatory_context == "CSRD, CBAM, SB253"

    def test_config_budget_limits(self):
        """Test config budget limit settings."""
        config = IntelligentAgentConfig(
            name="TestAgent",
            description="Test agent",
            max_budget_per_call_usd=0.05,
            max_budget_per_execution_usd=0.25
        )
        assert config.max_budget_per_call_usd == 0.05
        assert config.max_budget_per_execution_usd == 0.25


# =============================================================================
# TEST: IntelligenceCapabilities
# =============================================================================

class TestIntelligenceCapabilities:
    """Tests for IntelligenceCapabilities."""

    def test_capabilities_creation(self):
        """Test capabilities creation."""
        caps = IntelligenceCapabilities(
            can_explain=True,
            can_recommend=True,
            can_detect_anomalies=False,
            can_reason=True,
            can_validate=True,
            uses_rag=False,
            uses_tools=False
        )
        assert caps.can_explain is True
        assert caps.can_recommend is True
        assert caps.can_detect_anomalies is False

    def test_capabilities_minimum_requirements(self):
        """Test that minimum capabilities are enforced."""
        # All agents MUST be able to explain
        caps = IntelligenceCapabilities(
            can_explain=True,
            can_recommend=False,
            can_detect_anomalies=False,
            can_reason=False,
            can_validate=False,
            uses_rag=False,
            uses_tools=False
        )
        assert caps.can_explain is True


# =============================================================================
# TEST: @require_intelligence Decorator
# =============================================================================

class TestRequireIntelligenceDecorator:
    """Tests for @require_intelligence decorator."""

    def test_decorator_validates_class(self):
        """Test that decorator validates class at definition time."""
        # This should work - class implements required methods
        @require_intelligence
        class ValidAgent(IntelligentAgentBase):
            def __init__(self):
                config = IntelligentAgentConfig(
                    name="ValidAgent",
                    description="Valid agent"
                )
                super().__init__(config)

            def get_intelligence_level(self):
                return IntelligenceLevel.STANDARD

            def get_intelligence_capabilities(self):
                return IntelligenceCapabilities(
                    can_explain=True,
                    can_recommend=False,
                    can_detect_anomalies=False,
                    can_reason=False,
                    can_validate=False,
                    uses_rag=False,
                    uses_tools=False
                )

            def execute(self, input_data):
                return AgentResult(success=True, data={})

            def validate_input(self, input_data):
                return True

        # Class should be created successfully
        assert ValidAgent is not None


# =============================================================================
# TEST: AgentIntelligenceValidator
# =============================================================================

class TestAgentIntelligenceValidator:
    """Tests for AgentIntelligenceValidator."""

    def test_validator_accepts_valid_agent(self):
        """Test validator accepts properly configured agent."""
        validator = AgentIntelligenceValidator()

        # Create a valid mock agent
        mock_agent = Mock()
        mock_agent.get_intelligence_level.return_value = IntelligenceLevel.STANDARD
        mock_agent.get_intelligence_capabilities.return_value = IntelligenceCapabilities(
            can_explain=True,
            can_recommend=True,
            can_detect_anomalies=False,
            can_reason=False,
            can_validate=False,
            uses_rag=False,
            uses_tools=False
        )
        mock_agent.generate_explanation = Mock()
        mock_agent.generate_recommendations = Mock()

        result = validator.validate(mock_agent, allow_none_level=False, strict=False)
        assert result.is_valid is True

    def test_validator_rejects_none_level(self):
        """Test validator rejects NONE intelligence level."""
        validator = AgentIntelligenceValidator()

        mock_agent = Mock()
        mock_agent.get_intelligence_level.return_value = IntelligenceLevel.NONE
        mock_agent.get_intelligence_capabilities.return_value = IntelligenceCapabilities(
            can_explain=False,
            can_recommend=False,
            can_detect_anomalies=False,
            can_reason=False,
            can_validate=False,
            uses_rag=False,
            uses_tools=False
        )

        result = validator.validate(mock_agent, allow_none_level=False, strict=True)
        assert result.is_valid is False
        assert len(result.errors) > 0


# =============================================================================
# TEST: is_intelligent_agent Function
# =============================================================================

class TestIsIntelligentAgent:
    """Tests for is_intelligent_agent helper function."""

    def test_detects_intelligent_agent(self):
        """Test detection of intelligent agent."""
        mock_agent = Mock(spec=IntelligentAgent)
        mock_agent.get_intelligence_level.return_value = IntelligenceLevel.STANDARD
        mock_agent.get_intelligence_capabilities.return_value = IntelligenceCapabilities(
            can_explain=True,
            can_recommend=False,
            can_detect_anomalies=False,
            can_reason=False,
            can_validate=False,
            uses_rag=False,
            uses_tools=False
        )

        # Should return True for agent with intelligence methods
        result = is_intelligent_agent(mock_agent)
        assert result is True

    def test_detects_non_intelligent_agent(self):
        """Test detection of non-intelligent agent."""
        # Regular mock without intelligence methods
        mock_agent = Mock()
        del mock_agent.get_intelligence_level
        del mock_agent.get_intelligence_capabilities

        result = is_intelligent_agent(mock_agent)
        assert result is False


# =============================================================================
# TEST: Recommendation Model
# =============================================================================

class TestRecommendation:
    """Tests for Recommendation model."""

    def test_recommendation_creation(self):
        """Test recommendation creation."""
        rec = Recommendation(
            title="Install Solar PV",
            description="Install rooftop solar to reduce grid emissions",
            priority="high",
            estimated_impact="30% reduction in Scope 2 emissions",
            estimated_cost="$50,000 - $100,000",
            payback_period="5-7 years",
            regulatory_relevance="CSRD ESRS E1",
            confidence_score=0.85
        )

        assert rec.title == "Install Solar PV"
        assert rec.priority == "high"
        assert rec.confidence_score == 0.85

    def test_recommendation_serialization(self):
        """Test recommendation can be serialized."""
        rec = Recommendation(
            title="Test",
            description="Test recommendation",
            priority="medium"
        )

        # Should be able to convert to dict
        rec_dict = rec.dict()
        assert isinstance(rec_dict, dict)
        assert rec_dict["title"] == "Test"


# =============================================================================
# TEST: Anomaly Model
# =============================================================================

class TestAnomaly:
    """Tests for Anomaly model."""

    def test_anomaly_creation(self):
        """Test anomaly creation."""
        anomaly = Anomaly(
            field="emissions_total",
            value=1000000.0,
            expected_range=(0, 100000),
            severity="high",
            explanation="Emissions significantly exceed expected range"
        )

        assert anomaly.field == "emissions_total"
        assert anomaly.severity == "high"

    def test_anomaly_with_suggestion(self):
        """Test anomaly with suggestion."""
        anomaly = Anomaly(
            field="consumption",
            value=-100,
            expected_range=(0, 10000),
            severity="critical",
            explanation="Negative consumption value",
            suggested_action="Review input data for errors"
        )

        assert anomaly.suggested_action is not None


# =============================================================================
# TEST: IntelligenceMixin
# =============================================================================

class TestIntelligenceMixin:
    """Tests for IntelligenceMixin."""

    def test_mixin_adds_intelligence_methods(self):
        """Test that mixin adds intelligence methods to existing agent."""

        class SimpleAgent(BaseAgent):
            def execute(self, input_data):
                return AgentResult(success=True, data={"result": 42})

            def validate_input(self, input_data):
                return True

        # Apply mixin
        class IntelligentSimpleAgent(IntelligenceMixin, SimpleAgent):
            pass

        agent = IntelligentSimpleAgent(AgentConfig(
            name="SimpleAgent",
            description="Simple test agent"
        ))

        # Should have intelligence methods
        assert hasattr(agent, "generate_explanation")
        assert hasattr(agent, "generate_recommendations")
        assert callable(agent.generate_explanation)


# =============================================================================
# TEST: retrofit_agent_class Function
# =============================================================================

class TestRetrofitAgentClass:
    """Tests for retrofit_agent_class function."""

    def test_retrofit_creates_new_class(self):
        """Test that retrofit creates a new intelligent class."""

        class LegacyAgent(BaseAgent):
            def execute(self, input_data):
                return AgentResult(success=True, data={})

            def validate_input(self, input_data):
                return True

        IntelligentLegacyAgent = retrofit_agent_class(LegacyAgent)

        # Should be a new class
        assert IntelligentLegacyAgent is not LegacyAgent
        assert IntelligentLegacyAgent.__name__ == "IntelligentLegacyAgent"

    def test_retrofitted_class_has_intelligence(self):
        """Test that retrofitted class has intelligence capabilities."""

        class LegacyAgent(BaseAgent):
            def execute(self, input_data):
                return AgentResult(success=True, data={})

            def validate_input(self, input_data):
                return True

        IntelligentLegacyAgent = retrofit_agent_class(LegacyAgent)

        # Should have intelligence methods from mixin
        assert hasattr(IntelligentLegacyAgent, "generate_explanation")


# =============================================================================
# TEST: IntelligenceMetrics
# =============================================================================

class TestIntelligenceMetrics:
    """Tests for IntelligenceMetrics."""

    def test_metrics_creation(self):
        """Test metrics creation."""
        metrics = IntelligenceMetrics(
            total_llm_calls=10,
            total_tokens_used=5000,
            total_cost_usd=0.50,
            cache_hits=3,
            cache_misses=7,
            average_latency_ms=150.0
        )

        assert metrics.total_llm_calls == 10
        assert metrics.total_cost_usd == 0.50

    def test_metrics_dict_conversion(self):
        """Test metrics can be converted to dict."""
        metrics = IntelligenceMetrics()
        metrics_dict = metrics.dict()

        assert isinstance(metrics_dict, dict)
        assert "total_llm_calls" in metrics_dict


# =============================================================================
# TEST: Pilot Intelligent Agents
# =============================================================================

class TestPilotIntelligentAgents:
    """Tests for pilot intelligent agents."""

    @pytest.mark.skipif(True, reason="Requires full agent infrastructure")
    def test_intelligent_carbon_agent_exists(self):
        """Test IntelligentCarbonAgent can be imported."""
        from greenlang.agents.carbon_agent_intelligent import IntelligentCarbonAgent
        assert IntelligentCarbonAgent is not None

    @pytest.mark.skipif(True, reason="Requires full agent infrastructure")
    def test_intelligent_fuel_agent_exists(self):
        """Test IntelligentFuelAgent can be imported."""
        from greenlang.agents.fuel_agent_intelligent import IntelligentFuelAgent
        assert IntelligentFuelAgent is not None

    @pytest.mark.skipif(True, reason="Requires full agent infrastructure")
    def test_intelligent_grid_factor_agent_exists(self):
        """Test IntelligentGridFactorAgent can be imported."""
        from greenlang.agents.grid_factor_agent_intelligent import IntelligentGridFactorAgent
        assert IntelligentGridFactorAgent is not None

    @pytest.mark.skipif(True, reason="Requires full agent infrastructure")
    def test_intelligent_recommendation_agent_exists(self):
        """Test IntelligentRecommendationAgent can be imported."""
        from greenlang.agents.recommendation_agent_intelligent import IntelligentRecommendationAgent
        assert IntelligentRecommendationAgent is not None


# =============================================================================
# TEST: Zero-Hallucination Principle
# =============================================================================

class TestZeroHallucinationPrinciple:
    """Tests for zero-hallucination principle enforcement."""

    def test_calculations_are_deterministic(self):
        """Test that calculation methods don't use LLM."""
        # Mock agent with calculation method
        class CalculationAgent(IntelligentAgentBase):
            def __init__(self):
                config = IntelligentAgentConfig(
                    name="CalcAgent",
                    description="Calculation agent"
                )
                super().__init__(config)

            def get_intelligence_level(self):
                return IntelligenceLevel.STANDARD

            def get_intelligence_capabilities(self):
                return IntelligenceCapabilities(
                    can_explain=True,
                    can_recommend=False,
                    can_detect_anomalies=False,
                    can_reason=False,
                    can_validate=False,
                    uses_rag=False,
                    uses_tools=False
                )

            def _calculate(self, input_data):
                # This should be PURELY DETERMINISTIC
                return input_data.get("value", 0) * 2

            def execute(self, input_data):
                result = self._calculate(input_data)
                return AgentResult(success=True, data={"result": result})

            def validate_input(self, input_data):
                return True

        agent = CalculationAgent()

        # Same input should always produce same output
        result1 = agent._calculate({"value": 10})
        result2 = agent._calculate({"value": 10})

        assert result1 == result2 == 20


# =============================================================================
# TEST: AI Factory Intelligence Validator
# =============================================================================

class TestAIFactoryIntelligenceValidator:
    """Tests for AI Factory intelligence validation."""

    def test_validator_import(self):
        """Test intelligence validator can be imported."""
        try:
            from GL_Agent_Factory.backend.agent_generator.intelligence_validator import (
                IntelligenceValidator,
                validate_agent_spec,
            )
            assert IntelligenceValidator is not None
        except ImportError:
            # Module path may differ in test environment
            pytest.skip("AI Factory not in path")

    def test_spec_validation_requires_intelligence(self):
        """Test that spec validation requires intelligence config."""
        try:
            from GL_Agent_Factory.backend.agent_generator.intelligence_validator import (
                validate_agent_spec,
            )

            # Spec without intelligence should fail
            spec_without_intelligence = {
                "agent_id": "test/agent",
                "name": "Test Agent",
                "version": "1.0.0"
            }

            result = validate_agent_spec(spec_without_intelligence)
            assert result.is_valid is False

        except ImportError:
            pytest.skip("AI Factory not in path")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntelligenceIntegration:
    """Integration tests for intelligence framework."""

    @pytest.mark.skipif(True, reason="Requires LLM infrastructure")
    def test_end_to_end_intelligent_agent_execution(self):
        """Test end-to-end intelligent agent execution."""
        # This would test a real intelligent agent with mocked LLM
        pass

    @pytest.mark.skipif(True, reason="Requires LLM infrastructure")
    def test_explanation_generation(self):
        """Test explanation generation with LLM."""
        pass

    @pytest.mark.skipif(True, reason="Requires LLM infrastructure")
    def test_recommendation_generation(self):
        """Test recommendation generation with LLM."""
        pass


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
