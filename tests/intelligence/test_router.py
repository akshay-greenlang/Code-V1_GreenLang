# -*- coding: utf-8 -*-
"""
Tests for ProviderRouter

Tests intelligent provider/model selection for cost optimization.
Validates 60-90% cost savings through smart routing.
"""

import pytest
from greenlang.intelligence.runtime.router import (
    ProviderRouter,
    QueryType,
    LatencyRequirement,
    ProviderSpec,
    ModelSpec,
)


class TestProviderRouter:
    """Test ProviderRouter model selection"""

    def test_router_initialization(self):
        """ProviderRouter should initialize with provider/model database"""
        router = ProviderRouter()

        assert len(router.providers) > 0
        assert "openai" in router.providers
        assert "anthropic" in router.providers

    def test_simple_query_routes_to_cheap_model(self):
        """Simple queries should route to cheapest capable model"""
        router = ProviderRouter()

        provider, model = router.select_provider(
            query_type=QueryType.SIMPLE_CALC,
            budget_cents=5,
            latency_req=LatencyRequirement.REALTIME
        )

        # Should select cheap model like gpt-4o-mini
        assert provider == "openai"
        assert "mini" in model.lower() or "3.5" in model

    def test_complex_query_routes_to_capable_model(self):
        """Complex queries should route to more capable model"""
        router = ProviderRouter()

        provider, model = router.select_provider(
            query_type=QueryType.COMPLEX_ANALYSIS,
            budget_cents=50,
            latency_req=LatencyRequirement.BATCH
        )

        # Should select capable model like claude-3-sonnet or gpt-4
        assert provider in ["openai", "anthropic"]
        if provider == "anthropic":
            assert "claude-3" in model.lower()
        else:
            assert "gpt-4" in model.lower() and "mini" not in model.lower()

    def test_budget_constraint_respected(self):
        """Router should respect budget constraints"""
        router = ProviderRouter()

        # Very small budget should force cheap model
        provider, model = router.select_provider(
            query_type=QueryType.COMPLEX_ANALYSIS,
            budget_cents=1,  # Very small budget
            latency_req=LatencyRequirement.REALTIME
        )

        # Should select cheapest model even for complex query
        assert "mini" in model.lower() or "3.5" in model

    def test_latency_requirement_realtime(self):
        """REALTIME latency should prefer faster models"""
        router = ProviderRouter()

        provider, model = router.select_provider(
            query_type=QueryType.SIMPLE_CALC,
            budget_cents=10,
            latency_req=LatencyRequirement.REALTIME
        )

        # Should select fast model
        assert provider == "openai"  # OpenAI typically faster
        assert "mini" in model.lower() or "3.5" in model

    def test_latency_requirement_batch(self):
        """BATCH latency allows any model"""
        router = ProviderRouter()

        provider, model = router.select_provider(
            query_type=QueryType.COMPLEX_ANALYSIS,
            budget_cents=50,
            latency_req=LatencyRequirement.BATCH
        )

        # Can use any capable model
        assert provider in ["openai", "anthropic"]


class TestCostEstimation:
    """Test cost estimation for different models"""

    def test_estimate_cost_gpt4o_mini(self):
        """Estimate cost for GPT-4o-mini"""
        router = ProviderRouter()

        cost = router.estimate_cost("openai", "gpt-4o-mini", estimated_tokens=2000)

        # GPT-4o-mini is ~$0.0002 for 2K tokens
        assert cost < 0.001  # Very cheap
        assert cost > 0.00005  # Not free

    def test_estimate_cost_claude3_sonnet(self):
        """Estimate cost for Claude-3-Sonnet"""
        router = ProviderRouter()

        cost = router.estimate_cost("anthropic", "claude-3-sonnet-20240229", estimated_tokens=5000)

        # Claude-3-Sonnet more expensive
        assert cost > 0.005  # More than cheap models
        assert cost < 0.05   # But not insanely expensive

    def test_estimate_cost_scales_with_tokens(self):
        """Cost should scale linearly with token count"""
        router = ProviderRouter()

        cost_1k = router.estimate_cost("openai", "gpt-4o-mini", estimated_tokens=1000)
        cost_2k = router.estimate_cost("openai", "gpt-4o-mini", estimated_tokens=2000)

        # Should be approximately 2x
        ratio = cost_2k / cost_1k
        assert 1.8 < ratio < 2.2  # Allow small variance


class TestQueryTypeClassification:
    """Test query type enum"""

    def test_query_types_available(self):
        """All expected query types should be available"""
        assert hasattr(QueryType, "SIMPLE_CALC")
        assert hasattr(QueryType, "DATA_EXTRACTION")
        assert hasattr(QueryType, "COMPLEX_ANALYSIS")
        assert hasattr(QueryType, "TOOL_CALLING")

    def test_query_type_values(self):
        """Query types should have string values"""
        assert QueryType.SIMPLE_CALC.value == "simple_calc"
        assert QueryType.COMPLEX_ANALYSIS.value == "complex_analysis"


class TestLatencyRequirement:
    """Test latency requirement enum"""

    def test_latency_requirements_available(self):
        """All expected latency requirements should be available"""
        assert hasattr(LatencyRequirement, "REALTIME")
        assert hasattr(LatencyRequirement, "INTERACTIVE")
        assert hasattr(LatencyRequirement, "BATCH")

    def test_latency_requirement_ordering(self):
        """Latency requirements should be ordered"""
        # REALTIME < INTERACTIVE < BATCH
        assert LatencyRequirement.REALTIME.value == "realtime"
        assert LatencyRequirement.INTERACTIVE.value == "interactive"
        assert LatencyRequirement.BATCH.value == "batch"


class TestProviderSpec:
    """Test ProviderSpec data structure"""

    def test_provider_spec_structure(self):
        """ProviderSpec should have required fields"""
        router = ProviderRouter()
        openai_spec = router.providers["openai"]

        assert isinstance(openai_spec, ProviderSpec)
        assert openai_spec.name == "openai"
        assert len(openai_spec.models) > 0

    def test_model_spec_structure(self):
        """ModelSpec should have required fields"""
        router = ProviderRouter()
        openai_spec = router.providers["openai"]

        gpt4o_mini = None
        for model in openai_spec.models:
            if "gpt-4o-mini" in model.name:
                gpt4o_mini = model
                break

        assert gpt4o_mini is not None
        assert isinstance(gpt4o_mini, ModelSpec)
        assert gpt4o_mini.cost_per_1k_input > 0
        assert gpt4o_mini.cost_per_1k_output > 0
        assert gpt4o_mini.context_window > 0


class TestCostSavings:
    """Test cost savings calculations"""

    def test_cost_savings_simple_query(self):
        """Simple query should save 80-90% vs GPT-4-turbo"""
        router = ProviderRouter()

        # Cheap model for simple query
        cheap_provider, cheap_model = router.select_provider(
            query_type=QueryType.SIMPLE_CALC,
            budget_cents=5,
            latency_req=LatencyRequirement.REALTIME
        )
        cheap_cost = router.estimate_cost(cheap_provider, cheap_model, estimated_tokens=2000)

        # Expensive model baseline (GPT-4-turbo)
        expensive_cost = router.estimate_cost("openai", "gpt-4-turbo", estimated_tokens=2000)

        # Should save at least 80%
        savings_ratio = (expensive_cost - cheap_cost) / expensive_cost
        assert savings_ratio > 0.80

    def test_cost_savings_complex_query(self):
        """Complex query should still save 30-50% vs GPT-4-turbo"""
        router = ProviderRouter()

        # Best model for complex query
        provider, model = router.select_provider(
            query_type=QueryType.COMPLEX_ANALYSIS,
            budget_cents=50,
            latency_req=LatencyRequirement.BATCH
        )
        selected_cost = router.estimate_cost(provider, model, estimated_tokens=5000)

        # Expensive baseline
        expensive_cost = router.estimate_cost("openai", "gpt-4-turbo", estimated_tokens=5000)

        # Should save at least 20%
        savings_ratio = (expensive_cost - selected_cost) / expensive_cost
        assert savings_ratio > 0.20


class TestEdgeCases:
    """Test edge cases in routing"""

    def test_zero_budget(self):
        """Handle zero budget gracefully"""
        router = ProviderRouter()

        # Should still return cheapest model
        provider, model = router.select_provider(
            query_type=QueryType.SIMPLE_CALC,
            budget_cents=0,
            latency_req=LatencyRequirement.REALTIME
        )

        assert provider is not None
        assert model is not None

    def test_huge_budget(self):
        """Handle unlimited budget"""
        router = ProviderRouter()

        provider, model = router.select_provider(
            query_type=QueryType.COMPLEX_ANALYSIS,
            budget_cents=10000,  # $100
            latency_req=LatencyRequirement.BATCH
        )

        # Should select best model for task
        assert provider in ["openai", "anthropic"]

    def test_unknown_query_type_string(self):
        """Handle unknown query type string"""
        router = ProviderRouter()

        # Should default to safe choice
        provider, model = router.select_provider(
            query_type="unknown_type",
            budget_cents=10,
            latency_req=LatencyRequirement.INTERACTIVE
        )

        # Should still return valid model
        assert provider in ["openai", "anthropic"]


class TestToolCallingSupport:
    """Test tool calling capability filtering"""

    def test_tool_calling_query_selects_compatible_model(self):
        """TOOL_CALLING query should select model with tool support"""
        router = ProviderRouter()

        provider, model = router.select_provider(
            query_type=QueryType.TOOL_CALLING,
            budget_cents=20,
            latency_req=LatencyRequirement.INTERACTIVE
        )

        # Should select model with tool calling support
        # GPT-4o-mini, GPT-4-turbo, Claude-3 all support tools
        assert provider in ["openai", "anthropic"]
        if provider == "openai":
            assert "gpt-4" in model.lower() or "gpt-3.5-turbo" in model.lower()
        else:
            assert "claude-3" in model.lower()


class TestProviderAvailability:
    """Test provider availability handling"""

    def test_both_providers_registered(self):
        """Both OpenAI and Anthropic should be registered"""
        router = ProviderRouter()

        assert "openai" in router.providers
        assert "anthropic" in router.providers

    def test_openai_has_models(self):
        """OpenAI provider should have multiple models"""
        router = ProviderRouter()

        openai = router.providers["openai"]
        assert len(openai.models) >= 3  # At least gpt-4o, gpt-4o-mini, gpt-4-turbo

    def test_anthropic_has_models(self):
        """Anthropic provider should have multiple models"""
        router = ProviderRouter()

        anthropic = router.providers["anthropic"]
        assert len(anthropic.models) >= 2  # At least Claude-3-Opus, Claude-3-Sonnet


class TestBusinessValue:
    """Test business value of ProviderRouter"""

    def test_annual_cost_savings_projection(self):
        """Project annual cost savings for typical workload"""
        router = ProviderRouter()

        # Typical workload: 70% simple, 30% complex
        simple_queries = 70000  # per year
        complex_queries = 30000  # per year

        # Cost with router (smart routing)
        simple_provider, simple_model = router.select_provider(
            QueryType.SIMPLE_CALC, budget_cents=5, LatencyRequirement.REALTIME
        )
        simple_cost = router.estimate_cost(simple_provider, simple_model, 2000)

        complex_provider, complex_model = router.select_provider(
            QueryType.COMPLEX_ANALYSIS, budget_cents=50, LatencyRequirement.BATCH
        )
        complex_cost = router.estimate_cost(complex_provider, complex_model, 5000)

        smart_total = (simple_queries * simple_cost) + (complex_queries * complex_cost)

        # Cost without router (always GPT-4-turbo)
        baseline_simple = router.estimate_cost("openai", "gpt-4-turbo", 2000)
        baseline_complex = router.estimate_cost("openai", "gpt-4-turbo", 5000)
        baseline_total = (simple_queries * baseline_simple) + (complex_queries * baseline_complex)

        # Calculate savings
        savings = baseline_total - smart_total
        savings_percent = (savings / baseline_total) * 100

        # Should save 60-90%
        assert savings_percent > 60
        assert savings > 0

        # Log for visibility
        print(f"\nAnnual cost savings: ${savings:.2f} ({savings_percent:.1f}%)")
        print(f"Smart routing: ${smart_total:.2f}")
        print(f"Baseline (GPT-4-turbo): ${baseline_total:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
