"""
Unit tests for LLMProvider interface conformance

Tests that FakeProvider correctly implements the LLMProvider interface
and validates proper budget enforcement, telemetry emission, and error handling.
"""

import pytest
from greenlang.intelligence.providers.base import LLMProvider, LLMCapabilities
from greenlang.intelligence.schemas.messages import ChatMessage, Role
from greenlang.intelligence.schemas.tools import ToolDef
from greenlang.intelligence.schemas.responses import FinishReason
from greenlang.intelligence.runtime.budget import Budget, BudgetExceeded
from greenlang.intelligence.runtime.session import ChatSession
from greenlang.intelligence.runtime.telemetry import IntelligenceTelemetry, ConsoleEmitter
from greenlang.intelligence.schemas.jsonschema import JSONSchema

from tests.intelligence.fakes import (
    FakeProvider,
    make_text_response,
    make_tool_call_response,
    make_json_response,
)


class TestFakeProviderInterface:
    """Test FakeProvider implements all required LLMProvider methods"""

    @pytest.mark.asyncio
    async def test_fake_provider_is_llm_provider(self):
        """FakeProvider should be an instance of LLMProvider"""
        fake = FakeProvider()
        assert isinstance(fake, LLMProvider)

    @pytest.mark.asyncio
    async def test_fake_provider_has_capabilities(self):
        """FakeProvider should expose capabilities property"""
        fake = FakeProvider()
        assert hasattr(fake, "capabilities")
        assert isinstance(fake.capabilities, LLMCapabilities)

        # Default capabilities should support all features
        assert fake.capabilities.function_calling is True
        assert fake.capabilities.json_schema_mode is True
        assert fake.capabilities.max_output_tokens > 0
        assert fake.capabilities.context_window_tokens > 0

    @pytest.mark.asyncio
    async def test_fake_provider_custom_capabilities(self):
        """FakeProvider should accept custom capabilities"""
        custom_caps = LLMCapabilities(
            function_calling=False,
            json_schema_mode=False,
            max_output_tokens=2048,
            context_window_tokens=4096,
        )
        fake = FakeProvider(capabilities=custom_caps)

        assert fake.capabilities.function_calling is False
        assert fake.capabilities.json_schema_mode is False
        assert fake.capabilities.max_output_tokens == 2048

    @pytest.mark.asyncio
    async def test_fake_provider_chat_returns_response(self):
        """FakeProvider.chat() should return ChatResponse"""
        fake = FakeProvider([make_text_response("Test response")])
        budget = Budget(max_usd=0.50)

        response = await fake.chat(
            messages=[ChatMessage(role=Role.user, content="Hello")],
            budget=budget,
        )

        assert response.text == "Test response"
        assert response.usage is not None
        assert response.finish_reason is not None

    @pytest.mark.asyncio
    async def test_fake_provider_accepts_all_parameters(self):
        """FakeProvider.chat() should accept all standard parameters"""
        fake = FakeProvider([make_text_response("Response")])
        budget = Budget(max_usd=0.50)

        tools = [
            ToolDef(
                name="test_tool",
                description="Test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]

        json_schema: JSONSchema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
        }

        # Should not raise
        response = await fake.chat(
            messages=[ChatMessage(role=Role.user, content="Test")],
            tools=tools,
            json_schema=json_schema,
            budget=budget,
            temperature=0.7,
            top_p=0.9,
            seed=42,
            tool_choice="auto",
            metadata={"session_id": "test_123"},
        )

        assert response is not None


class TestChatSessionOrchestration:
    """Test ChatSession orchestration with FakeProvider"""

    @pytest.mark.asyncio
    async def test_chat_session_with_fake_provider(self):
        """ChatSession should work with FakeProvider"""
        fake = FakeProvider([make_text_response("Test response", cost_usd=0.05)])
        session = ChatSession(fake)
        budget = Budget(max_usd=0.50)

        response = await session.chat(
            messages=[ChatMessage(role=Role.user, content="Hello")],
            budget=budget,
        )

        assert response.text == "Test response"
        assert budget.spent_usd == 0.05

    @pytest.mark.asyncio
    async def test_chat_session_validates_empty_messages(self):
        """ChatSession should raise ValueError for empty messages"""
        fake = FakeProvider([make_text_response("Response")])
        session = ChatSession(fake)
        budget = Budget(max_usd=0.50)

        with pytest.raises(ValueError, match="messages cannot be empty"):
            await session.chat(messages=[], budget=budget)

    @pytest.mark.asyncio
    async def test_chat_session_checks_exhausted_budget(self):
        """ChatSession should raise BudgetExceeded if budget already exhausted"""
        fake = FakeProvider([make_text_response("Response", cost_usd=0.10)])
        session = ChatSession(fake)
        budget = Budget(max_usd=0.10)
        budget.spent_usd = 0.10  # Already exhausted

        with pytest.raises(BudgetExceeded, match="already exhausted"):
            await session.chat(
                messages=[ChatMessage(role=Role.user, content="Test")],
                budget=budget,
            )


class TestBudgetEnforcement:
    """Test budget enforcement with FakeProvider"""

    @pytest.mark.asyncio
    async def test_budget_check_before_add(self):
        """Provider should check budget before adding usage"""
        fake = FakeProvider([make_text_response("Response", cost_usd=0.60)])
        budget = Budget(max_usd=0.50)

        with pytest.raises(BudgetExceeded):
            await fake.chat(
                messages=[ChatMessage(role=Role.user, content="Test")],
                budget=budget,
            )

    @pytest.mark.asyncio
    async def test_budget_add_after_success(self):
        """Provider should add usage to budget after successful call"""
        fake = FakeProvider([make_text_response("Response", cost_usd=0.05, tokens=100)])
        budget = Budget(max_usd=0.50)

        await fake.chat(
            messages=[ChatMessage(role=Role.user, content="Test")],
            budget=budget,
        )

        assert budget.spent_usd == 0.05
        assert budget.spent_tokens == 100

    @pytest.mark.asyncio
    async def test_budget_token_cap_enforcement(self):
        """Provider should enforce token cap"""
        fake = FakeProvider([make_text_response("Response", cost_usd=0.01, tokens=5000)])
        budget = Budget(max_usd=0.50, max_tokens=4000)

        with pytest.raises(BudgetExceeded, match="Token cap exceeded"):
            await fake.chat(
                messages=[ChatMessage(role=Role.user, content="Test")],
                budget=budget,
            )

    @pytest.mark.asyncio
    async def test_budget_dollar_cap_enforcement(self):
        """Provider should enforce dollar cap"""
        fake = FakeProvider([make_text_response("Response", cost_usd=0.60, tokens=100)])
        budget = Budget(max_usd=0.50, max_tokens=10000)

        with pytest.raises(BudgetExceeded, match="Dollar cap exceeded"):
            await fake.chat(
                messages=[ChatMessage(role=Role.user, content="Test")],
                budget=budget,
            )

    @pytest.mark.asyncio
    async def test_budget_remaining_calculations(self):
        """Budget should track remaining budget correctly"""
        budget = Budget(max_usd=1.00, max_tokens=10000)

        assert budget.remaining_usd == 1.00
        assert budget.remaining_tokens == 10000

        budget.add(add_usd=0.30, add_tokens=3000)

        assert budget.remaining_usd == 0.70
        assert budget.remaining_tokens == 7000


class TestTelemetryEmission:
    """Test telemetry emission during provider calls"""

    @pytest.mark.asyncio
    async def test_telemetry_emits_on_success(self):
        """ChatSession should emit telemetry event on successful call"""
        fake = FakeProvider([make_text_response("Response", cost_usd=0.05, tokens=100)])

        events = []
        def capture_event(event_type, metadata):
            events.append({"type": event_type, "meta": metadata})

        session = ChatSession(fake, telemetry=capture_event)
        budget = Budget(max_usd=0.50)

        await session.chat(
            messages=[ChatMessage(role=Role.user, content="Test")],
            budget=budget,
        )

        # Should have emitted event
        assert len(events) == 1
        assert events[0]["type"] == "llm.chat"
        assert events[0]["meta"]["cost_usd"] == 0.05
        assert events[0]["meta"]["total_tokens"] == 100

    @pytest.mark.asyncio
    async def test_telemetry_with_intelligence_telemetry_object(self):
        """ChatSession should work with IntelligenceTelemetry object"""
        fake = FakeProvider([make_text_response("Response", cost_usd=0.05)])

        # Use IntelligenceTelemetry with ConsoleEmitter (won't actually print in tests)
        telemetry = IntelligenceTelemetry(emitter=ConsoleEmitter())
        session = ChatSession(fake, telemetry=telemetry)
        budget = Budget(max_usd=0.50)

        # Should not raise
        response = await session.chat(
            messages=[ChatMessage(role=Role.user, content="Test")],
            budget=budget,
        )

        assert response.text == "Response"

    @pytest.mark.asyncio
    async def test_telemetry_includes_tool_calls(self):
        """Telemetry should include tool call information"""
        fake = FakeProvider([
            make_tool_call_response([
                {"name": "test_tool", "arguments": {"x": 10}}
            ])
        ])

        events = []
        def capture_event(event_type, metadata):
            events.append({"type": event_type, "meta": metadata})

        session = ChatSession(fake, telemetry=capture_event)
        budget = Budget(max_usd=0.50)

        await session.chat(
            messages=[ChatMessage(role=Role.user, content="Test")],
            tools=[ToolDef(name="test_tool", description="", parameters={})],
            budget=budget,
        )

        # Should include tool call info
        assert len(events) == 1
        assert "test_tool" in events[0]["meta"]["tool_calls"]


class TestErrorPropagation:
    """Test error handling and propagation"""

    @pytest.mark.asyncio
    async def test_budget_exceeded_propagates(self):
        """BudgetExceeded should propagate through ChatSession"""
        fake = FakeProvider([make_text_response("Response", cost_usd=1.00)])
        session = ChatSession(fake)
        budget = Budget(max_usd=0.50)

        with pytest.raises(BudgetExceeded):
            await session.chat(
                messages=[ChatMessage(role=Role.user, content="Test")],
                budget=budget,
            )

    @pytest.mark.asyncio
    async def test_provider_error_propagates(self):
        """Provider errors should propagate through ChatSession"""
        from greenlang.intelligence.providers.errors import ProviderRateLimit

        fake = FakeProvider(
            error_to_raise=ProviderRateLimit("Rate limit", "fake", retry_after=60)
        )
        session = ChatSession(fake)
        budget = Budget(max_usd=0.50)

        with pytest.raises(ProviderRateLimit) as exc_info:
            await session.chat(
                messages=[ChatMessage(role=Role.user, content="Test")],
                budget=budget,
            )

        assert exc_info.value.retry_after == 60

    @pytest.mark.asyncio
    async def test_capability_validation_error(self):
        """Should raise error if requesting unsupported feature"""
        fake = FakeProvider(
            capabilities=LLMCapabilities(function_calling=False)
        )
        budget = Budget(max_usd=0.50)

        tools = [ToolDef(name="tool", description="", parameters={})]

        with pytest.raises(ValueError, match="does not support function calling"):
            await fake.chat(
                messages=[ChatMessage(role=Role.user, content="Test")],
                tools=tools,
                budget=budget,
            )


class TestAsyncPatterns:
    """Test async execution patterns"""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_calls(self):
        """Should support concurrent provider calls"""
        import asyncio

        fake = FakeProvider([make_text_response("Response", cost_usd=0.05)])

        async def make_call():
            budget = Budget(max_usd=0.50)
            return await fake.chat(
                messages=[ChatMessage(role=Role.user, content="Test")],
                budget=budget,
            )

        # Make 5 concurrent calls
        responses = await asyncio.gather(*[make_call() for _ in range(5)])

        assert len(responses) == 5
        assert all(r.text == "Response" for r in responses)

    @pytest.mark.asyncio
    async def test_async_context_preservation(self):
        """Async context should be preserved across calls"""
        fake = FakeProvider([
            make_text_response("First"),
            make_text_response("Second"),
        ])
        budget = Budget(max_usd=0.50)

        # First call
        response1 = await fake.chat(
            messages=[ChatMessage(role=Role.user, content="Test 1")],
            budget=budget,
        )

        # Second call
        response2 = await fake.chat(
            messages=[ChatMessage(role=Role.user, content="Test 2")],
            budget=budget,
        )

        # Should cycle through responses
        assert response1.text == "First"
        assert response2.text == "Second"


class TestCallHistoryTracking:
    """Test FakeProvider call history tracking"""

    @pytest.mark.asyncio
    async def test_records_all_parameters(self):
        """FakeProvider should record all call parameters"""
        fake = FakeProvider([make_text_response("Response")])
        budget = Budget(max_usd=0.50)

        messages = [
            ChatMessage(role=Role.system, content="System prompt"),
            ChatMessage(role=Role.user, content="User message"),
        ]

        tools = [ToolDef(name="tool1", description="", parameters={})]

        await fake.chat(
            messages=messages,
            tools=tools,
            budget=budget,
            temperature=0.8,
            seed=42,
        )

        assert fake.get_call_count() == 1

        last_call = fake.get_last_call()
        assert len(last_call["messages"]) == 2
        assert last_call["messages"][0].content == "System prompt"
        assert len(last_call["tools"]) == 1
        assert last_call["temperature"] == 0.8
        assert last_call["seed"] == 42

    @pytest.mark.asyncio
    async def test_reset_clears_history(self):
        """FakeProvider.reset() should clear call history"""
        fake = FakeProvider([make_text_response("Response")])
        budget = Budget(max_usd=0.50)

        await fake.chat(
            messages=[ChatMessage(role=Role.user, content="Test")],
            budget=budget,
        )

        assert fake.get_call_count() == 1

        fake.reset()

        assert fake.get_call_count() == 0
        assert fake.get_last_call() is None

    @pytest.mark.asyncio
    async def test_multiple_calls_recorded(self):
        """FakeProvider should record multiple calls"""
        fake = FakeProvider([make_text_response("Response")])
        budget = Budget(max_usd=0.50)

        for i in range(3):
            await fake.chat(
                messages=[ChatMessage(role=Role.user, content=f"Message {i}")],
                budget=budget,
            )

        assert fake.get_call_count() == 3

        # Verify messages from each call
        assert fake.call_history[0]["messages"][0].content == "Message 0"
        assert fake.call_history[1]["messages"][0].content == "Message 1"
        assert fake.call_history[2]["messages"][0].content == "Message 2"
