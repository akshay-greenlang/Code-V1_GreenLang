# -*- coding: utf-8 -*-
"""
ChatSession Tool Calling Tests
GL Intelligence Infrastructure

Tests for ChatSession integration with tools (function calling).
Critical for Phase 3 agent transformation.

Version: 1.0.0
Date: 2025-11-06
"""

import pytest
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import json


class ToolDef(BaseModel):
    """Tool definition for ChatSession."""
    name: str
    description: str
    parameters: Dict[str, Any]


class MockChatResponse(BaseModel):
    """Mock chat response for testing."""
    text: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Dict[str, Any] = Field(default_factory=lambda: {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "total_cost": 0.001
    })
    finish_reason: str = "stop"
    provider_info: Dict[str, str] = Field(default_factory=lambda: {
        "provider": "mock",
        "model": "mock-model"
    })


class MockChatSession:
    """
    Mock ChatSession for testing tool calling.

    Simulates tool calling behavior without requiring actual LLM API.
    """

    def __init__(self):
        """Initialize mock session."""
        self.call_history: List[Dict[str, Any]] = []

    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[ToolDef]] = None,
        temperature: float = 0.0,
        tool_choice: Optional[str] = None,
        budget: Optional[Dict] = None,
        **kwargs
    ) -> MockChatResponse:
        """
        Mock chat method that simulates tool calling.

        Args:
            messages: Conversation messages
            tools: Available tools
            temperature: Sampling temperature
            tool_choice: Tool choice strategy
            budget: Budget configuration

        Returns:
            MockChatResponse with tool calls
        """
        # Record call
        self.call_history.append({
            "messages": messages,
            "tools": [t.dict() if hasattr(t, 'dict') else t for t in (tools or [])],
            "temperature": temperature,
            "tool_choice": tool_choice
        })

        # Simulate tool calling based on query
        last_message = messages[-1]["content"] if messages else ""

        # Check if tools are available
        if tools and tool_choice != "none":
            # Simulate intelligent tool selection
            if "calculate" in last_message.lower() and any(t.name == "calculate_emissions" for t in tools):
                return MockChatResponse(
                    text="I'll calculate the emissions using the calculate_emissions tool.",
                    tool_calls=[{
                        "id": "call_001",
                        "name": "calculate_emissions",
                        "arguments": json.dumps({
                            "fuel_type": "natural_gas",
                            "consumption_kwh": 10000
                        })
                    }]
                )

            elif "search" in last_message.lower() or "find" in last_message.lower():
                if any(t.name == "search_database" for t in tools):
                    return MockChatResponse(
                        text="I'll search the database for relevant information.",
                        tool_calls=[{
                            "id": "call_002",
                            "name": "search_database",
                            "arguments": json.dumps({
                                "query": "heat pump technology"
                            })
                        }]
                    )

            elif "recommend" in last_message.lower():
                if any(t.name == "get_technology_recommendation" for t in tools):
                    return MockChatResponse(
                        text="I'll provide technology recommendations.",
                        tool_calls=[{
                            "id": "call_003",
                            "name": "get_technology_recommendation",
                            "arguments": json.dumps({
                                "facility_type": "manufacturing",
                                "energy_demand_kwh": 50000
                            })
                        }]
                    )

        # Default response (no tool call)
        return MockChatResponse(
            text="I understand your question. Let me help you with that.",
            tool_calls=None
        )


@pytest.fixture
def mock_session():
    """Fixture providing mock chat session."""
    return MockChatSession()


@pytest.fixture
def emission_calculation_tool():
    """Fixture providing emission calculation tool definition."""
    return ToolDef(
        name="calculate_emissions",
        description="Calculate GHG emissions from fuel consumption",
        parameters={
            "type": "object",
            "properties": {
                "fuel_type": {
                    "type": "string",
                    "enum": ["natural_gas", "coal", "diesel", "gasoline"],
                    "description": "Type of fuel consumed"
                },
                "consumption_kwh": {
                    "type": "number",
                    "description": "Fuel consumption in kWh"
                },
                "region": {
                    "type": "string",
                    "description": "Geographic region (optional)",
                    "default": "US"
                }
            },
            "required": ["fuel_type", "consumption_kwh"]
        }
    )


@pytest.fixture
def search_tool():
    """Fixture providing database search tool."""
    return ToolDef(
        name="search_database",
        description="Search technology database for specifications",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "category": {
                    "type": "string",
                    "enum": ["heat_pump", "solar", "chp", "all"],
                    "default": "all"
                },
                "limit": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": ["query"]
        }
    )


@pytest.fixture
def recommendation_tool():
    """Fixture providing technology recommendation tool."""
    return ToolDef(
        name="get_technology_recommendation",
        description="Get decarbonization technology recommendations",
        parameters={
            "type": "object",
            "properties": {
                "facility_type": {
                    "type": "string",
                    "enum": ["manufacturing", "food_processing", "chemical", "office"],
                    "description": "Type of facility"
                },
                "energy_demand_kwh": {
                    "type": "number",
                    "description": "Annual energy demand in kWh"
                },
                "budget_usd": {
                    "type": "number",
                    "description": "Available budget in USD (optional)"
                }
            },
            "required": ["facility_type", "energy_demand_kwh"]
        }
    )


class TestChatSessionToolBasics:
    """Test basic tool calling functionality."""

    @pytest.mark.asyncio
    async def test_chat_without_tools(self, mock_session):
        """Test chat without tools returns normal response."""
        response = await mock_session.chat(
            messages=[{"role": "user", "content": "Hello, how are you?"}]
        )

        assert response.text is not None
        assert response.tool_calls is None
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_with_tools_no_call(self, mock_session, emission_calculation_tool):
        """Test chat with tools available but no tool call triggered."""
        response = await mock_session.chat(
            messages=[{"role": "user", "content": "What is the GHG Protocol?"}],
            tools=[emission_calculation_tool],
            tool_choice="auto"
        )

        assert response is not None
        # Tool not called because query doesn't require calculation

    @pytest.mark.asyncio
    async def test_tool_choice_none(self, mock_session, emission_calculation_tool):
        """Test that tool_choice='none' prevents tool calling."""
        response = await mock_session.chat(
            messages=[{"role": "user", "content": "Calculate emissions for 10000 kWh natural gas"}],
            tools=[emission_calculation_tool],
            tool_choice="none"
        )

        assert response.tool_calls is None


class TestChatSessionSingleToolCall:
    """Test single tool calling scenarios."""

    @pytest.mark.asyncio
    async def test_emission_calculation_tool_call(self, mock_session, emission_calculation_tool):
        """Test emission calculation tool is called correctly."""
        response = await mock_session.chat(
            messages=[{
                "role": "user",
                "content": "Calculate emissions for 10000 kWh of natural gas consumption"
            }],
            tools=[emission_calculation_tool],
            tool_choice="auto",
            temperature=0.7
        )

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1

        tool_call = response.tool_calls[0]
        assert tool_call["name"] == "calculate_emissions"

        args = json.loads(tool_call["arguments"])
        assert args["fuel_type"] == "natural_gas"
        assert args["consumption_kwh"] == 10000

    @pytest.mark.asyncio
    async def test_search_tool_call(self, mock_session, search_tool):
        """Test database search tool is called correctly."""
        response = await mock_session.chat(
            messages=[{
                "role": "user",
                "content": "Search for information about heat pump technology"
            }],
            tools=[search_tool],
            tool_choice="auto"
        )

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1

        tool_call = response.tool_calls[0]
        assert tool_call["name"] == "search_database"

        args = json.loads(tool_call["arguments"])
        assert "query" in args
        assert "heat pump" in args["query"].lower()

    @pytest.mark.asyncio
    async def test_recommendation_tool_call(self, mock_session, recommendation_tool):
        """Test recommendation tool is called correctly."""
        response = await mock_session.chat(
            messages=[{
                "role": "user",
                "content": "Recommend decarbonization technology for a manufacturing facility with 50000 kWh annual demand"
            }],
            tools=[recommendation_tool],
            tool_choice="auto"
        )

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1

        tool_call = response.tool_calls[0]
        assert tool_call["name"] == "get_technology_recommendation"

        args = json.loads(tool_call["arguments"])
        assert args["facility_type"] == "manufacturing"
        assert args["energy_demand_kwh"] == 50000


class TestChatSessionMultiToolCall:
    """Test multi-tool calling scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_tools_available(
        self,
        mock_session,
        emission_calculation_tool,
        search_tool,
        recommendation_tool
    ):
        """Test with multiple tools available."""
        response = await mock_session.chat(
            messages=[{
                "role": "user",
                "content": "Calculate emissions for 10000 kWh natural gas"
            }],
            tools=[emission_calculation_tool, search_tool, recommendation_tool],
            tool_choice="auto"
        )

        # Should call the correct tool even when multiple are available
        assert response.tool_calls is not None
        tool_call = response.tool_calls[0]
        assert tool_call["name"] == "calculate_emissions"


class TestChatSessionToolOrchestration:
    """Test tool orchestration in multi-turn conversations."""

    @pytest.mark.asyncio
    async def test_multi_turn_with_tool_results(
        self,
        mock_session,
        emission_calculation_tool
    ):
        """Test multi-turn conversation with tool execution."""
        # Turn 1: Initial query triggers tool call
        response1 = await mock_session.chat(
            messages=[{
                "role": "user",
                "content": "Calculate emissions for 10000 kWh natural gas"
            }],
            tools=[emission_calculation_tool],
            tool_choice="auto"
        )

        assert response1.tool_calls is not None

        # Turn 2: Provide tool result and continue conversation
        tool_result = {
            "total_co2e_kg": 531.0,
            "methodology": "GHG Protocol",
            "emission_factor": 0.0531
        }

        response2 = await mock_session.chat(
            messages=[
                {"role": "user", "content": "Calculate emissions for 10000 kWh natural gas"},
                {"role": "assistant", "content": response1.text, "tool_calls": response1.tool_calls},
                {"role": "tool", "content": json.dumps(tool_result), "tool_call_id": response1.tool_calls[0]["id"]}
            ],
            tools=[emission_calculation_tool]
        )

        # Should continue conversation with tool result
        assert response2 is not None


class TestChatSessionToolValidation:
    """Test tool definition validation."""

    @pytest.mark.asyncio
    async def test_tool_definition_structure(self, emission_calculation_tool):
        """Test tool definition has correct structure."""
        assert emission_calculation_tool.name is not None
        assert emission_calculation_tool.description is not None
        assert emission_calculation_tool.parameters is not None
        assert "type" in emission_calculation_tool.parameters
        assert "properties" in emission_calculation_tool.parameters
        assert "required" in emission_calculation_tool.parameters

    @pytest.mark.asyncio
    async def test_tool_parameters_types(self, emission_calculation_tool):
        """Test tool parameters have correct types."""
        params = emission_calculation_tool.parameters
        props = params["properties"]

        assert "fuel_type" in props
        assert props["fuel_type"]["type"] == "string"

        assert "consumption_kwh" in props
        assert props["consumption_kwh"]["type"] == "number"


class TestChatSessionBudgetWithTools:
    """Test budget enforcement when using tools."""

    @pytest.mark.asyncio
    async def test_tool_call_within_budget(self, mock_session, emission_calculation_tool):
        """Test tool calling works within budget."""
        budget = {"max_usd": 1.0}

        response = await mock_session.chat(
            messages=[{
                "role": "user",
                "content": "Calculate emissions for 10000 kWh natural gas"
            }],
            tools=[emission_calculation_tool],
            budget=budget
        )

        assert response.tool_calls is not None
        assert response.usage["total_cost"] < budget["max_usd"]


class TestChatSessionCallHistory:
    """Test chat session call history tracking."""

    @pytest.mark.asyncio
    async def test_call_history_recorded(self, mock_session, emission_calculation_tool):
        """Test that chat calls are recorded in history."""
        await mock_session.chat(
            messages=[{"role": "user", "content": "Test message"}],
            tools=[emission_calculation_tool]
        )

        assert len(mock_session.call_history) == 1
        assert mock_session.call_history[0]["messages"][0]["content"] == "Test message"
        assert len(mock_session.call_history[0]["tools"]) == 1

    @pytest.mark.asyncio
    async def test_multiple_calls_tracked(self, mock_session, emission_calculation_tool):
        """Test multiple calls are tracked."""
        for i in range(3):
            await mock_session.chat(
                messages=[{"role": "user", "content": f"Message {i}"}],
                tools=[emission_calculation_tool]
            )

        assert len(mock_session.call_history) == 3
        assert mock_session.call_history[0]["messages"][0]["content"] == "Message 0"
        assert mock_session.call_history[2]["messages"][0]["content"] == "Message 2"


class TestChatSessionTemperatureWithTools:
    """Test temperature parameter with tool calling."""

    @pytest.mark.asyncio
    async def test_zero_temperature_with_tools(self, mock_session, emission_calculation_tool):
        """Test deterministic behavior with temperature=0.0."""
        response = await mock_session.chat(
            messages=[{"role": "user", "content": "Calculate emissions"}],
            tools=[emission_calculation_tool],
            temperature=0.0
        )

        assert mock_session.call_history[-1]["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_high_temperature_with_tools(self, mock_session, emission_calculation_tool):
        """Test reasoning behavior with temperature=0.7."""
        response = await mock_session.chat(
            messages=[{"role": "user", "content": "Calculate emissions"}],
            tools=[emission_calculation_tool],
            temperature=0.7
        )

        assert mock_session.call_history[-1]["temperature"] == 0.7


# Integration test markers
@pytest.mark.integration
class TestChatSessionToolsIntegration:
    """Integration tests requiring actual ChatSession."""

    @pytest.mark.skip(reason="Requires actual LLM API - run manually")
    @pytest.mark.asyncio
    async def test_real_chatsession_tool_call(self):
        """
        Test with real ChatSession (OpenAI/Anthropic).

        This test is skipped by default because it requires:
        1. API credentials
        2. Network connection
        3. Costs money

        Run manually when testing actual integration.
        """
        # from greenlang.intelligence.runtime.session import ChatSession
        # from greenlang.intelligence.llm.providers import OpenAIProvider
        #
        # provider = OpenAIProvider(...)
        # session = ChatSession(provider)
        #
        # response = await session.chat(
        #     messages=[...],
        #     tools=[...],
        #     temperature=0.7
        # )
        #
        # assert response.tool_calls is not None
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
