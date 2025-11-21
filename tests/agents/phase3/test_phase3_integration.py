# -*- coding: utf-8 -*-
"""
Phase 3 Integration Tests

Comprehensive integration tests for all 4 transformed Phase 3 agents:
- Decarbonization Roadmap Agent V3
- Boiler Replacement Agent V3
- Industrial Heat Pump Agent V3
- Waste Heat Recovery Agent V3

Tests cover:
1. RAG integration
2. Multi-step reasoning
3. Tool orchestration
4. Temperature 0.7 behavior
5. Result structure validation
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from greenlang.agents.decarbonization_roadmap_agent_ai_v3 import DecarbonizationRoadmapAgentAI_V3
from greenlang.agents.boiler_replacement_agent_ai_v3 import BoilerReplacementAgentAI_V3
from greenlang.agents.industrial_heat_pump_agent_ai_v3 import IndustrialHeatPumpAgentAI_V3
from greenlang.agents.waste_heat_recovery_agent_ai_v3 import WasteHeatRecoveryAgentAI_V3


class TestPhase3AgentArchitecture:
    """Test common architecture patterns across all Phase 3 agents."""

    def test_all_agents_inherit_from_reasoning_agent(self):
        """All Phase 3 agents should inherit from ReasoningAgent."""
        from greenlang.agents.base_agents import ReasoningAgent

        assert issubclass(DecarbonizationRoadmapAgentAI_V3, ReasoningAgent)
        assert issubclass(BoilerReplacementAgentAI_V3, ReasoningAgent)
        assert issubclass(IndustrialHeatPumpAgentAI_V3, ReasoningAgent)
        assert issubclass(WasteHeatRecoveryAgentAI_V3, ReasoningAgent)

    def test_all_agents_have_category_recommendation(self):
        """All Phase 3 agents should have RECOMMENDATION category."""
        from greenlang.agents.categories import AgentCategory

        agents = [
            DecarbonizationRoadmapAgentAI_V3,
            BoilerReplacementAgentAI_V3,
            IndustrialHeatPumpAgentAI_V3,
            WasteHeatRecoveryAgentAI_V3
        ]

        for agent_class in agents:
            assert agent_class.category == AgentCategory.RECOMMENDATION

    def test_all_agents_have_metadata(self):
        """All Phase 3 agents should have complete metadata."""
        agents = [
            DecarbonizationRoadmapAgentAI_V3,
            BoilerReplacementAgentAI_V3,
            IndustrialHeatPumpAgentAI_V3,
            WasteHeatRecoveryAgentAI_V3
        ]

        for agent_class in agents:
            metadata = agent_class.metadata
            assert metadata is not None
            assert metadata.uses_chat_session == True
            assert metadata.uses_rag == True
            assert metadata.uses_tools == True
            assert "v3" in metadata.name.lower() or "v2" in metadata.name.lower()


@pytest.mark.asyncio
class TestDecarbonizationRoadmapAgentV3:
    """Integration tests for Decarbonization Roadmap Agent V3."""

    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = DecarbonizationRoadmapAgentAI_V3()
        assert agent is not None
        assert agent.category.value == "RECOMMENDATION"

    async def test_agent_reason_with_mocks(
        self,
        mock_rag_engine,
        mock_chat_session,
        sample_facility_context,
        assert_reasoning_agent_result
    ):
        """Test agent.reason() method with mocked dependencies."""
        agent = DecarbonizationRoadmapAgentAI_V3()

        result = await agent.reason(
            context=sample_facility_context,
            session=mock_chat_session,
            rag_engine=mock_rag_engine
        )

        # Assert result structure
        assert_reasoning_agent_result(result)

        # Assert RAG was called
        mock_rag_engine.query.assert_called_once()

        # Assert ChatSession was called
        assert mock_chat_session.chat.call_count >= 1

        # Assert temperature 0.7
        call_args = mock_chat_session.chat.call_args
        assert call_args.kwargs.get("temperature") == 0.7

    async def test_rag_collections(
        self,
        mock_rag_engine,
        mock_chat_session,
        sample_facility_context
    ):
        """Test that agent queries correct RAG collections."""
        agent = DecarbonizationRoadmapAgentAI_V3()

        await agent.reason(
            context=sample_facility_context,
            session=mock_chat_session,
            rag_engine=mock_rag_engine
        )

        # Check RAG collections
        call_args = mock_rag_engine.query.call_args
        collections = call_args.kwargs.get("collections", [])

        expected_collections = [
            "decarbonization_case_studies",
            "industrial_best_practices",
            "technology_database",
            "financial_models",
            "regulatory_compliance",
            "site_feasibility"
        ]

        assert len(collections) == len(expected_collections)
        for expected in expected_collections:
            assert expected in collections


@pytest.mark.asyncio
class TestBoilerReplacementAgentV3:
    """Integration tests for Boiler Replacement Agent V3."""

    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = BoilerReplacementAgentAI_V3()
        assert agent is not None

    async def test_agent_reason_with_mocks(
        self,
        mock_rag_engine,
        mock_chat_session,
        sample_boiler_context,
        assert_reasoning_agent_result
    ):
        """Test agent.reason() method."""
        agent = BoilerReplacementAgentAI_V3()

        result = await agent.reason(
            context=sample_boiler_context,
            session=mock_chat_session,
            rag_engine=mock_rag_engine
        )

        assert_reasoning_agent_result(result)

        # Boiler-specific assertions
        assert "recommended_option" in result or "recommended_technology" in result

    async def test_rag_collections(
        self,
        mock_rag_engine,
        mock_chat_session,
        sample_boiler_context
    ):
        """Test RAG collections for boiler agent."""
        agent = BoilerReplacementAgentAI_V3()

        await agent.reason(
            context=sample_boiler_context,
            session=mock_chat_session,
            rag_engine=mock_rag_engine
        )

        call_args = mock_rag_engine.query.call_args
        collections = call_args.kwargs.get("collections", [])

        expected_collections = [
            "boiler_specifications",
            "boiler_case_studies",
            "vendor_catalogs",
            "maintenance_best_practices",
            "asme_standards"
        ]

        assert len(collections) == len(expected_collections)


@pytest.mark.asyncio
class TestIndustrialHeatPumpAgentV3:
    """Integration tests for Industrial Heat Pump Agent V3."""

    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = IndustrialHeatPumpAgentAI_V3()
        assert agent is not None

    async def test_agent_reason_with_mocks(
        self,
        mock_rag_engine,
        mock_chat_session,
        sample_heat_pump_context,
        assert_reasoning_agent_result
    ):
        """Test agent.reason() method."""
        agent = IndustrialHeatPumpAgentAI_V3()

        result = await agent.reason(
            context=sample_heat_pump_context,
            session=mock_chat_session,
            rag_engine=mock_rag_engine
        )

        assert_reasoning_agent_result(result)

    async def test_rag_collections(
        self,
        mock_rag_engine,
        mock_chat_session,
        sample_heat_pump_context
    ):
        """Test RAG collections for heat pump agent."""
        agent = IndustrialHeatPumpAgentAI_V3()

        await agent.reason(
            context=sample_heat_pump_context,
            session=mock_chat_session,
            rag_engine=mock_rag_engine
        )

        call_args = mock_rag_engine.query.call_args
        collections = call_args.kwargs.get("collections", [])

        expected_collections = [
            "heat_pump_specifications",
            "carnot_efficiency_models",
            "case_studies_heat_pumps",
            "cop_performance_data"
        ]

        assert len(collections) == len(expected_collections)


@pytest.mark.asyncio
class TestWasteHeatRecoveryAgentV3:
    """Integration tests for Waste Heat Recovery Agent V3."""

    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = WasteHeatRecoveryAgentAI_V3()
        assert agent is not None

    async def test_agent_reason_with_mocks(
        self,
        mock_rag_engine,
        mock_chat_session,
        sample_whr_context,
        assert_reasoning_agent_result
    ):
        """Test agent.reason() method."""
        agent = WasteHeatRecoveryAgentAI_V3()

        result = await agent.reason(
            context=sample_whr_context,
            session=mock_chat_session,
            rag_engine=mock_rag_engine
        )

        assert_reasoning_agent_result(result)

    async def test_rag_collections(
        self,
        mock_rag_engine,
        mock_chat_session,
        sample_whr_context
    ):
        """Test RAG collections for WHR agent."""
        agent = WasteHeatRecoveryAgentAI_V3()

        await agent.reason(
            context=sample_whr_context,
            session=mock_chat_session,
            rag_engine=mock_rag_engine
        )

        call_args = mock_rag_engine.query.call_args
        collections = call_args.kwargs.get("collections", [])

        expected_collections = [
            "whr_technologies",
            "heat_exchanger_specs",
            "pinch_analysis_data",
            "case_studies_whr"
        ]

        assert len(collections) == len(expected_collections)


@pytest.mark.asyncio
class TestMultiStepReasoning:
    """Test multi-step reasoning capabilities."""

    async def test_tool_orchestration_loop(
        self,
        mock_rag_engine,
        mock_chat_session,
        sample_facility_context
    ):
        """Test that agents support multi-turn tool orchestration."""
        agent = DecarbonizationRoadmapAgentAI_V3()

        # Mock tool calls in response
        mock_response_with_tools = Mock()
        mock_response_with_tools.text = "Analyzing..."
        mock_response_with_tools.tool_calls = [
            {
                "id": "call_1",
                "name": "aggregate_ghg_inventory",
                "arguments": '{"fuel_consumption": {"natural_gas": 50000}, "electricity_kwh": 15000000, "grid_region": "CAISO"}'
            }
        ]
        mock_response_with_tools.provider_info = {"model": "gpt-4"}
        mock_response_with_tools.usage = {"total_tokens": 1000, "total_cost": 0.05}

        # Final response without tools
        mock_final_response = Mock()
        mock_final_response.text = "Final recommendation..."
        mock_final_response.tool_calls = []
        mock_final_response.provider_info = {"model": "gpt-4"}
        mock_final_response.usage = {"total_tokens": 2000, "total_cost": 0.10}

        # Configure mock to return different responses
        mock_chat_session.chat.side_effect = [
            mock_response_with_tools,
            mock_final_response
        ]

        result = await agent.reason(
            context=sample_facility_context,
            session=mock_chat_session,
            rag_engine=mock_rag_engine
        )

        # Should have made multiple chat calls
        assert mock_chat_session.chat.call_count >= 2

        # Should have tool execution trace
        assert "reasoning_trace" in result
        assert "tool_execution" in result["reasoning_trace"]
        assert "orchestration_iterations" in result["reasoning_trace"]

    async def test_temperature_consistency(
        self,
        mock_rag_engine,
        mock_chat_session,
        sample_facility_context
    ):
        """Test that temperature 0.7 is used consistently."""
        agent = DecarbonizationRoadmapAgentAI_V3()

        await agent.reason(
            context=sample_facility_context,
            session=mock_chat_session,
            rag_engine=mock_rag_engine
        )

        # Check all chat calls used temperature 0.7
        for call in mock_chat_session.chat.call_args_list:
            assert call.kwargs.get("temperature") == 0.7


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling and resilience."""

    async def test_agent_handles_rag_failure(
        self,
        mock_chat_session,
        sample_facility_context
    ):
        """Test agent handles RAG engine failure gracefully."""
        agent = DecarbonizationRoadmapAgentAI_V3()

        # Mock RAG engine that fails
        failing_rag = Mock()
        failing_rag.query = AsyncMock(side_effect=Exception("RAG connection failed"))

        result = await agent.reason(
            context=sample_facility_context,
            session=mock_chat_session,
            rag_engine=failing_rag
        )

        # Should return error result
        assert "success" in result
        assert result["success"] == False
        assert "error" in result

    async def test_agent_handles_chat_session_failure(
        self,
        mock_rag_engine,
        sample_facility_context
    ):
        """Test agent handles ChatSession failure gracefully."""
        agent = DecarbonizationRoadmapAgentAI_V3()

        # Mock ChatSession that fails
        failing_session = Mock()
        failing_session.chat = AsyncMock(side_effect=Exception("API timeout"))

        result = await agent.reason(
            context=sample_facility_context,
            session=failing_session,
            rag_engine=mock_rag_engine
        )

        # Should return error result
        assert "success" in result
        assert result["success"] == False
        assert "error" in result


@pytest.mark.asyncio
class TestToolImplementations:
    """Test tool implementations."""

    def test_decarbonization_tools_count(self):
        """Test Decarbonization agent has 11 tools."""
        agent = DecarbonizationRoadmapAgentAI_V3()
        tools = agent._get_all_tools()
        assert len(tools) == 11

    def test_boiler_tools_count(self):
        """Test Boiler agent has 11 tools."""
        agent = BoilerReplacementAgentAI_V3()
        tools = agent._get_all_tools()
        assert len(tools) == 11

    def test_heat_pump_tools_count(self):
        """Test Heat Pump agent has 11 tools."""
        agent = IndustrialHeatPumpAgentAI_V3()
        tools = agent._get_all_tools()
        assert len(tools) == 11

    def test_whr_tools_count(self):
        """Test WHR agent has 11 tools."""
        agent = WasteHeatRecoveryAgentAI_V3()
        tools = agent._get_all_tools()
        assert len(tools) == 11

    def test_phase3_tools_present(self):
        """Test that Phase 3 tools are present."""
        # Decarbonization
        agent1 = DecarbonizationRoadmapAgentAI_V3()
        tools1 = [t.name for t in agent1._get_all_tools()]
        assert "technology_database_tool" in tools1
        assert "financial_analysis_tool" in tools1
        assert "spatial_constraints_tool" in tools1

        # Boiler
        agent2 = BoilerReplacementAgentAI_V3()
        tools2 = [t.name for t in agent2._get_all_tools()]
        assert "boiler_database_tool" in tools2
        assert "cost_estimation_tool" in tools2
        assert "sizing_tool" in tools2

        # Heat Pump
        agent3 = IndustrialHeatPumpAgentAI_V3()
        tools3 = [t.name for t in agent3._get_all_tools()]
        assert "heat_pump_database_tool" in tools3
        assert "cop_calculator_tool" in tools3
        assert "grid_integration_tool" in tools3

        # WHR
        agent4 = WasteHeatRecoveryAgentAI_V3()
        tools4 = [t.name for t in agent4._get_all_tools()]
        assert "whr_database_tool" in tools4
        assert "heat_cascade_tool" in tools4
        assert "payback_calculator_tool" in tools4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
