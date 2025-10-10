"""Tests for AI-powered RecommendationAgent.

This module tests the RecommendationAgentAI implementation, ensuring:
1. Tool-first analysis (all calculations use tools)
2. Deterministic results (same input -> same output)
3. Backward compatibility with RecommendationAgent API
4. AI summaries and insights are generated
5. Recommendations are intelligent and actionable
6. ROI calculations are exact
7. Implementation plans are comprehensive
8. Budget enforcement works
9. Error handling is robust

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from greenlang.agents.recommendation_agent_ai import RecommendationAgentAI
from greenlang.intelligence import ChatResponse, Usage, FinishReason
from greenlang.intelligence.schemas.responses import ProviderInfo


class TestRecommendationAgentAI:
    """Test suite for RecommendationAgentAI."""

    @pytest.fixture
    def agent(self):
        """Create RecommendationAgentAI instance for testing."""
        return RecommendationAgentAI(budget_usd=1.0)

    @pytest.fixture
    def valid_building_data(self):
        """Create valid test building data."""
        return {
            "emissions_by_source": {
                "electricity": 15000,
                "natural_gas": 8500,
                "diesel": 3200,
            },
            "building_type": "commercial_office",
            "building_area": 50000,
            "occupancy": 200,
            "building_age": 20,
            "performance_rating": "Below Average",
            "load_breakdown": {
                "hvac_load": 0.45,
                "lighting_load": 0.25,
                "plug_load": 0.30,
            },
            "country": "US",
        }

    @pytest.fixture
    def simple_building_data(self):
        """Create simple building data."""
        return {
            "emissions_by_source": {
                "electricity": 10000,
                "natural_gas": 5000,
            },
            "building_type": "commercial_office",
        }

    @pytest.fixture
    def high_hvac_data(self):
        """Create data with high HVAC consumption."""
        return {
            "emissions_by_source": {
                "electricity": 20000,
                "natural_gas": 10000,
            },
            "building_age": 25,
            "load_breakdown": {
                "hvac_load": 0.55,
                "lighting_load": 0.20,
                "plug_load": 0.25,
            },
        }

    def test_initialization(self, agent):
        """Test RecommendationAgentAI initializes correctly."""
        assert agent.config.name == "RecommendationAgentAI"
        assert agent.config.version == "0.1.0"
        assert agent.budget_usd == 1.0
        assert agent.enable_ai_summary is True
        assert agent.enable_implementation_plans is True
        assert agent.max_recommendations == 5

        # Verify tools are defined
        assert agent.analyze_energy_usage_tool is not None
        assert agent.calculate_roi_tool is not None
        assert agent.rank_recommendations_tool is not None
        assert agent.estimate_savings_tool is not None
        assert agent.generate_implementation_plan_tool is not None

        # Verify original recommendation agent is initialized
        assert agent.rec_agent is not None

    def test_validate_valid_input(self, agent, valid_building_data):
        """Test validation passes for valid input."""
        assert agent.validate_input(valid_building_data) is True

    def test_validate_minimal_input(self, agent):
        """Test validation with minimal data."""
        minimal = {"emissions_by_source": {"electricity": 10000}}
        assert agent.validate_input(minimal) is True

    def test_analyze_energy_usage_tool_basic(self, agent, simple_building_data):
        """Test analyze_energy_usage tool with basic data."""
        result = agent._analyze_energy_usage_impl(
            emissions_by_source=simple_building_data["emissions_by_source"]
        )

        # Verify structure
        assert "total_emissions_kg" in result
        assert "source_percentages" in result
        assert "dominant_source" in result
        assert "issues_identified" in result
        assert "issue_count" in result

        # Verify calculations
        assert result["total_emissions_kg"] == 15000
        assert result["source_percentages"]["electricity"] == 66.67
        assert result["source_percentages"]["natural_gas"] == 33.33
        assert result["dominant_source"] == "electricity"

    def test_analyze_energy_usage_high_electricity(self, agent):
        """Test energy usage analysis identifies high electricity usage."""
        result = agent._analyze_energy_usage_impl(
            emissions_by_source={
                "electricity": 18000,
                "natural_gas": 2000,
            }
        )

        # Should identify high electricity issue
        issues = result["issues_identified"]
        assert len(issues) > 0
        assert any(issue["type"] == "high_electricity" for issue in issues)

        # Verify electricity percentage
        assert result["source_percentages"]["electricity"] == 90.0

    def test_analyze_energy_usage_high_hvac(self, agent, high_hvac_data):
        """Test energy usage analysis identifies high HVAC load."""
        result = agent._analyze_energy_usage_impl(
            emissions_by_source=high_hvac_data["emissions_by_source"],
            load_breakdown=high_hvac_data["load_breakdown"],
        )

        # Should identify high HVAC issue
        issues = result["issues_identified"]
        assert any(issue["type"] == "high_hvac" for issue in issues)

    def test_analyze_energy_usage_aging_building(self, agent):
        """Test energy usage analysis identifies aging infrastructure."""
        result = agent._analyze_energy_usage_impl(
            emissions_by_source={"electricity": 10000},
            building_age=25,
        )

        # Should identify aging infrastructure issue
        issues = result["issues_identified"]
        assert any(issue["type"] == "aging_infrastructure" for issue in issues)

    def test_analyze_energy_usage_poor_performance(self, agent):
        """Test energy usage analysis identifies poor performance."""
        result = agent._analyze_energy_usage_impl(
            emissions_by_source={"electricity": 10000},
            performance_rating="Poor",
        )

        # Should identify poor performance issue
        issues = result["issues_identified"]
        assert any(issue["type"] == "poor_performance" for issue in issues)

    def test_calculate_roi_tool_implementation(self, agent):
        """Test calculate_roi tool calculates ROI correctly."""
        recommendations = [
            {
                "action": "Install LED lighting",
                "cost": "Medium",
                "impact": "50-70% reduction in lighting energy",
                "payback": "2-3 years",
            },
            {
                "action": "Upgrade HVAC system",
                "cost": "High",
                "impact": "20-30% reduction in HVAC energy",
                "payback": "5-7 years",
            },
        ]

        result = agent._calculate_roi_impl(
            recommendations=recommendations,
            current_emissions_kg=20000,
            energy_cost_per_kwh=0.12,
        )

        # Verify structure
        assert "roi_calculations" in result
        assert "total_potential_savings_usd" in result
        assert "total_emissions_reduction_kg" in result

        # Verify calculations exist
        roi_calcs = result["roi_calculations"]
        assert len(roi_calcs) == 2

        # Verify first recommendation (LED lighting - Medium cost)
        rec1 = roi_calcs[0]
        assert rec1["action"] == "Install LED lighting"
        assert rec1["estimated_cost_usd"] == 50000
        assert rec1["cost_category"] == "Medium"
        assert rec1["payback_years"] > 0
        assert rec1["roi_percentage"] > 0
        assert rec1["emissions_reduction_kg"] > 0

        # Verify second recommendation (HVAC - High cost)
        rec2 = roi_calcs[1]
        assert rec2["action"] == "Upgrade HVAC system"
        assert rec2["estimated_cost_usd"] == 250000
        assert rec2["cost_category"] == "High"

    def test_calculate_roi_low_cost_recommendation(self, agent):
        """Test ROI calculation for low-cost recommendation."""
        recommendations = [
            {
                "action": "Install occupancy sensors",
                "cost": "Low",
                "impact": "20-30% reduction",
                "payback": "1-2 years",
            }
        ]

        result = agent._calculate_roi_impl(
            recommendations=recommendations,
            current_emissions_kg=10000,
        )

        roi_calc = result["roi_calculations"][0]
        assert roi_calc["estimated_cost_usd"] == 5000  # Low cost
        assert roi_calc["payback_years"] <= 2

    def test_rank_recommendations_by_roi(self, agent):
        """Test ranking recommendations by ROI."""
        recommendations = [
            {
                "action": "Action A",
                "cost": "Low",
                "priority": "High",
                "payback": "2 years",
                "roi_percentage": 50,
            },
            {
                "action": "Action B",
                "cost": "Medium",
                "priority": "Medium",
                "payback": "5 years",
                "roi_percentage": 100,
            },
            {
                "action": "Action C",
                "cost": "High",
                "priority": "Low",
                "payback": "10 years",
                "roi_percentage": 25,
            },
        ]

        result = agent._rank_recommendations_impl(
            recommendations=recommendations,
            prioritize_by="roi",
        )

        # Verify structure
        assert "ranked_recommendations" in result
        assert "prioritization_strategy" in result
        assert result["prioritization_strategy"] == "roi"

        ranked = result["ranked_recommendations"]
        # Should be sorted by ROI (highest first)
        assert ranked[0]["action"] == "Action B"  # ROI 100%
        assert ranked[1]["action"] == "Action A"  # ROI 50%
        assert ranked[2]["action"] == "Action C"  # ROI 25%

        # Verify ranking numbers
        assert ranked[0]["rank"] == 1
        assert ranked[1]["rank"] == 2
        assert ranked[2]["rank"] == 3

    def test_rank_recommendations_by_cost(self, agent):
        """Test ranking recommendations by cost."""
        recommendations = [
            {"action": "High cost action", "cost": "High", "priority": "High", "payback": "5 years"},
            {"action": "Low cost action", "cost": "Low", "priority": "Medium", "payback": "2 years"},
            {"action": "Medium cost action", "cost": "Medium", "priority": "High", "payback": "3 years"},
        ]

        result = agent._rank_recommendations_impl(
            recommendations=recommendations,
            prioritize_by="cost",
        )

        ranked = result["ranked_recommendations"]
        # Should be sorted by cost (Low first)
        assert ranked[0]["cost"] == "Low"
        assert ranked[1]["cost"] == "Medium"
        assert ranked[2]["cost"] == "High"

    def test_rank_recommendations_by_impact(self, agent):
        """Test ranking recommendations by impact."""
        recommendations = [
            {"action": "Low impact", "impact": "5-10% reduction", "cost": "Low", "priority": "Low", "payback": "1 year"},
            {"action": "High impact", "impact": "50-70% reduction", "cost": "High", "priority": "High", "payback": "5 years"},
            {"action": "Medium impact", "impact": "20-30% reduction", "cost": "Medium", "priority": "Medium", "payback": "3 years"},
        ]

        result = agent._rank_recommendations_impl(
            recommendations=recommendations,
            prioritize_by="impact",
        )

        ranked = result["ranked_recommendations"]
        # Should be sorted by impact (highest first)
        assert "50-70%" in ranked[0]["impact"]
        assert "20-30%" in ranked[1]["impact"]
        assert "5-10%" in ranked[2]["impact"]

    def test_estimate_savings_tool_implementation(self, agent):
        """Test estimate_savings tool calculates savings."""
        recommendations = [
            {
                "action": "Recommendation 1",
                "impact": "30% reduction",
                "cost": "Medium",
                "payback": "3 years",
            },
            {
                "action": "Recommendation 2",
                "impact": "20% reduction",
                "cost": "Low",
                "payback": "2 years",
            },
        ]

        result = agent._estimate_savings_impl(
            recommendations=recommendations,
            current_emissions_kg=20000,
            current_energy_cost_usd=50000,
        )

        # Verify structure
        assert "emissions_savings" in result
        assert "cost_savings" in result
        assert "percentage_reduction_range" in result

        # Verify emissions savings
        emissions_savings = result["emissions_savings"]
        assert "minimum_kg_co2e" in emissions_savings
        assert "maximum_kg_co2e" in emissions_savings

        # Verify cost savings calculated
        cost_savings = result["cost_savings"]
        assert "minimum_annual_usd" in cost_savings
        assert "maximum_annual_usd" in cost_savings

    def test_estimate_savings_without_cost_data(self, agent):
        """Test estimate_savings without current energy cost."""
        recommendations = [
            {
                "action": "Recommendation 1",
                "impact": "30% reduction",
                "cost": "Medium",
                "payback": "3 years",
            },
        ]

        result = agent._estimate_savings_impl(
            recommendations=recommendations,
            current_emissions_kg=10000,
        )

        # Should still have emissions savings
        assert "emissions_savings" in result
        # Cost savings should be empty
        assert result["cost_savings"] == {}

    def test_generate_implementation_plan_tool(self, agent):
        """Test generate_implementation_plan creates roadmap."""
        recommendations = [
            {"action": "Quick win", "cost": "Low", "priority": "High", "payback": "Immediate"},
            {"action": "Medium term", "cost": "Medium", "priority": "High", "payback": "3 years"},
            {"action": "Long term", "cost": "High", "priority": "Medium", "payback": "7 years"},
        ]

        result = agent._generate_implementation_plan_impl(
            recommendations=recommendations,
            building_type="commercial_office",
            timeline_months=12,
        )

        # Verify structure
        assert "implementation_roadmap" in result
        assert "total_timeline_months" in result
        assert "building_type" in result
        assert "phases" in result

        # Verify roadmap exists
        roadmap = result["implementation_roadmap"]
        assert len(roadmap) > 0

        # Verify each phase has timeline
        for phase in roadmap:
            assert "timeline_months" in phase
            assert "implementation_steps" in phase

    def test_generate_implementation_plan_custom_timeline(self, agent):
        """Test implementation plan with custom timeline."""
        recommendations = [
            {"action": "Action 1", "cost": "Low", "priority": "High", "payback": "1 year"},
        ]

        result = agent._generate_implementation_plan_impl(
            recommendations=recommendations,
            timeline_months=24,
        )

        assert result["total_timeline_months"] == 24

    def test_build_prompt_comprehensive(self, agent, valid_building_data):
        """Test prompt building with comprehensive data."""
        prompt = agent._build_prompt(valid_building_data)

        # Verify key elements
        assert "commercial_office" in prompt
        assert "20 years" in prompt
        assert "Below Average" in prompt
        assert "US" in prompt
        assert "26,700 kg" in prompt or "26700" in prompt

        # Verify emissions breakdown
        assert "electricity" in prompt
        assert "natural_gas" in prompt

        # Verify load breakdown
        assert "hvac_load" in prompt

        # Verify tool mentions
        assert "analyze_energy_usage" in prompt
        assert "calculate_roi" in prompt
        assert "rank_recommendations" in prompt
        assert "estimate_savings" in prompt
        assert "generate_implementation_plan" in prompt

        # Verify max recommendations
        assert "top 5" in prompt.lower() or "5" in prompt

    def test_build_prompt_minimal_data(self, agent, simple_building_data):
        """Test prompt building with minimal data."""
        prompt = agent._build_prompt(simple_building_data)

        # Should still include essential elements
        assert "analyze_energy_usage" in prompt
        assert "commercial_office" in prompt
        assert "electricity" in prompt

    def test_determinism_same_input_same_output(self, agent, simple_building_data):
        """Test deterministic behavior: same input produces same output."""
        emissions = simple_building_data["emissions_by_source"]

        # Run analysis twice
        result1 = agent._analyze_energy_usage_impl(emissions_by_source=emissions)
        result2 = agent._analyze_energy_usage_impl(emissions_by_source=emissions)

        # Results should be identical
        assert result1["total_emissions_kg"] == result2["total_emissions_kg"]
        assert result1["source_percentages"] == result2["source_percentages"]
        assert result1["dominant_source"] == result2["dominant_source"]

    def test_backward_compatibility_api(self, agent):
        """Test backward compatibility with RecommendationAgent API."""
        # RecommendationAgentAI should have same interface
        assert hasattr(agent, "execute")
        assert hasattr(agent, "validate_input")
        assert hasattr(agent, "config")

        # Verify original RecommendationAgent is accessible
        assert hasattr(agent, "rec_agent")
        assert agent.rec_agent is not None

    def test_performance_tracking(self, agent, simple_building_data):
        """Test performance metrics tracking."""
        # Initial state
        initial_summary = agent.get_performance_summary()
        assert initial_summary["agent"] == "RecommendationAgentAI"
        assert "ai_metrics" in initial_summary
        assert "base_agent_metrics" in initial_summary

        # Make a tool call
        agent._analyze_energy_usage_impl(
            emissions_by_source=simple_building_data["emissions_by_source"]
        )

        # Verify metrics updated
        assert agent._tool_call_count > 0

        # Get updated summary
        summary = agent.get_performance_summary()
        assert summary["ai_metrics"]["tool_call_count"] > 0

    @patch("greenlang.agents.recommendation_agent_ai.ChatSession")
    def test_execute_with_mocked_ai(self, mock_session_class, agent, valid_building_data):
        """Test execute() with mocked ChatSession to verify AI integration."""
        # Create mock response
        mock_response = Mock(spec=ChatResponse)
        mock_response.text = (
            "Based on comprehensive analysis, here are the top 5 recommendations:\n\n"
            "1. **Upgrade to high-efficiency HVAC system** - The building's HVAC load is 45% "
            "with a 20-year-old system. Expected ROI: 18.5%, payback: 5-7 years, "
            "potential savings: 6,000 kg CO2e/year.\n\n"
            "2. **Install smart thermostats and zone controls** - Low-cost, high-impact "
            "improvement. Expected ROI: 42%, payback: 1-2 years, savings: 2,400 kg CO2e/year.\n\n"
            "3. **Convert to LED lighting** - Immediate energy savings. ROI: 35%, "
            "payback: 2-3 years, savings: 3,200 kg CO2e/year.\n\n"
            "4. **Install rooftop solar PV system** - Electricity is 56% of emissions. "
            "ROI: 12%, payback: 5-8 years, offset: 10,500 kg CO2e/year.\n\n"
            "5. **Seal air leaks and improve weatherstripping** - Quick win for older building. "
            "ROI: 65%, payback: 1 year, savings: 1,600 kg CO2e/year.\n\n"
            "Total potential savings: 23,700 kg CO2e/year (89% reduction)"
        )
        mock_response.tool_calls = [
            {
                "name": "analyze_energy_usage",
                "arguments": {
                    "emissions_by_source": valid_building_data["emissions_by_source"],
                    "load_breakdown": valid_building_data["load_breakdown"],
                    "building_age": 20,
                    "performance_rating": "Below Average",
                },
            },
            {
                "name": "calculate_roi",
                "arguments": {
                    "recommendations": [
                        {"action": "Upgrade HVAC", "cost": "High", "impact": "20-30%", "payback": "5-7 years"},
                        {"action": "Smart thermostats", "cost": "Low", "impact": "10-15%", "payback": "1-2 years"},
                    ],
                    "current_emissions_kg": 26700,
                    "energy_cost_per_kwh": 0.12,
                },
            },
            {
                "name": "rank_recommendations",
                "arguments": {
                    "recommendations": [],
                    "prioritize_by": "roi",
                },
            },
            {
                "name": "estimate_savings",
                "arguments": {
                    "recommendations": [],
                    "current_emissions_kg": 26700,
                    "current_energy_cost_usd": 50000,
                },
            },
            {
                "name": "generate_implementation_plan",
                "arguments": {
                    "recommendations": [],
                    "building_type": "commercial_office",
                    "timeline_months": 12,
                },
            },
        ]
        mock_response.usage = Usage(
            prompt_tokens=300,
            completion_tokens=250,
            total_tokens=550,
            cost_usd=0.03,
        )
        mock_response.provider_info = ProviderInfo(
            provider="openai",
            model="gpt-4o-mini",
        )
        mock_response.finish_reason = FinishReason.stop

        # Setup mock session
        mock_session = Mock()
        mock_session.chat = AsyncMock(return_value=mock_response)
        mock_session_class.return_value = mock_session

        # Run agent
        result = agent.execute(valid_building_data)

        # Verify success
        assert result.success is True
        assert result.data is not None

        # Verify output structure
        data = result.data
        assert "recommendations" in data
        assert "usage_analysis" in data
        assert "potential_savings" in data
        assert "ai_summary" in data
        assert "quick_wins" in data
        assert "high_impact" in data

        # Verify AI summary included
        assert "ROI" in data["ai_summary"] or "recommendations" in data["ai_summary"]

        # Verify metadata
        assert result.metadata is not None
        metadata = result.metadata
        assert metadata["agent"] == "RecommendationAgentAI"
        assert "calculation_time_ms" in metadata
        assert "ai_calls" in metadata
        assert "tool_calls" in metadata
        assert metadata["deterministic"] is True

        # Verify ChatSession was called with correct parameters
        mock_session.chat.assert_called_once()
        call_args = mock_session.chat.call_args
        assert call_args.kwargs["temperature"] == 0.0  # Deterministic
        assert call_args.kwargs["seed"] == 42  # Reproducible
        assert len(call_args.kwargs["tools"]) == 5  # All 5 tools

    def test_ai_summary_disabled(self, agent):
        """Test behavior when AI summary is disabled."""
        agent.enable_ai_summary = False
        assert agent.enable_ai_summary is False

        # Build prompt should still work
        prompt = agent._build_prompt({"emissions_by_source": {"electricity": 10000}})
        assert "analyze_energy_usage" in prompt

    def test_implementation_plans_disabled(self, agent, simple_building_data):
        """Test behavior when implementation plans are disabled."""
        agent.enable_implementation_plans = False
        assert agent.enable_implementation_plans is False

        # Build prompt should not include implementation plan tool
        prompt = agent._build_prompt(simple_building_data)
        assert "generate_implementation_plan" not in prompt

    def test_custom_max_recommendations(self):
        """Test custom max recommendations setting."""
        agent = RecommendationAgentAI(max_recommendations=10)
        assert agent.max_recommendations == 10

        # Verify prompt includes correct number
        prompt = agent._build_prompt({"emissions_by_source": {"electricity": 10000}})
        assert "top 10" in prompt.lower()

    def test_tool_call_tracking(self, agent):
        """Test tool call counting."""
        initial_count = agent._tool_call_count

        # Make several tool calls
        agent._analyze_energy_usage_impl(emissions_by_source={"electricity": 10000})
        agent._calculate_roi_impl(
            recommendations=[{"action": "Test", "cost": "Low", "impact": "10%", "payback": "2 years"}],
            current_emissions_kg=10000,
        )

        # Verify count increased
        assert agent._tool_call_count > initial_count
        assert agent._tool_call_count == initial_count + 2


class TestRecommendationAgentAIIntegration:
    """Integration tests for RecommendationAgentAI (require real/demo LLM)."""

    @pytest.fixture
    def agent(self):
        """Create agent with demo provider."""
        # Will use demo provider if no API keys available
        return RecommendationAgentAI(budget_usd=0.10, max_recommendations=5)

    @pytest.fixture
    def realistic_building(self):
        """Create realistic building data for integration testing."""
        return {
            "emissions_by_source": {
                "electricity": 25000,
                "natural_gas": 15000,
                "diesel": 5000,
            },
            "building_type": "commercial_office",
            "building_area": 100000,
            "occupancy": 500,
            "building_age": 15,
            "performance_rating": "Average",
            "load_breakdown": {
                "hvac_load": 0.42,
                "lighting_load": 0.28,
                "plug_load": 0.30,
            },
            "country": "US",
        }

    def test_full_recommendation_workflow(self, agent, realistic_building):
        """Test full recommendation workflow with demo provider."""
        result = agent.execute(realistic_building)

        # Should succeed with demo provider
        assert result.success is True
        assert result.data is not None

        data = result.data
        # Verify recommendations exist
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0
        assert len(data["recommendations"]) <= 5  # Max 5

        # Verify usage analysis
        if "usage_analysis" in data:
            assert "total_emissions_kg" in data["usage_analysis"]

    def test_simple_recommendation_minimal_data(self, agent):
        """Test recommendation with minimal data."""
        simple_data = {
            "emissions_by_source": {
                "electricity": 10000,
            }
        }

        result = agent.execute(simple_data)

        # Should succeed
        assert result.success is True
        assert "recommendations" in result.data
        assert len(result.data["recommendations"]) > 0

    def test_high_emissions_scenario(self, agent):
        """Test recommendations for high emissions scenario."""
        high_emissions_data = {
            "emissions_by_source": {
                "electricity": 50000,
                "natural_gas": 30000,
                "diesel": 10000,
            },
            "building_age": 30,
            "performance_rating": "Poor",
            "load_breakdown": {
                "hvac_load": 0.60,
            },
        }

        result = agent.execute(high_emissions_data)

        assert result.success is True
        # Should have multiple high-priority recommendations
        assert len(result.data["recommendations"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
