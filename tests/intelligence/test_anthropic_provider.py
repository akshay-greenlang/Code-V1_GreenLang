"""
Mocked tests for AnthropicProvider

Tests CTO specification compliance:
- Tool-call emission path (returns tool_calls, no message_json)
- JSON-mode success on first try (cost calls = 1)
- JSON-mode success after repairs (2 fails + 1 success → cost calls = 3)
- JSON-mode failure after >3 retries → GLJsonParseError (cost calls = 4)
- Budget cap exceeded → GLBudgetExceededError with partial cost
- 429 and 5xx error mapping
- Usage parsing
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import json
import os

# Set environment variable for all tests
os.environ['ANTHROPIC_API_KEY'] = 'test-key-123'

from greenlang.intelligence.providers.anthropic import AnthropicProvider
from greenlang.intelligence.providers.base import LLMProviderConfig
from greenlang.intelligence.schemas.messages import ChatMessage, Role
from greenlang.intelligence.schemas.tools import ToolDef
from greenlang.intelligence.runtime.budget import Budget, BudgetExceeded
from greenlang.intelligence.runtime.json_validator import GLJsonParseError
from greenlang.intelligence.providers.errors import (
    ProviderRateLimit,
    ProviderServerError,
    ProviderTimeout,
    ProviderAuthError,
)


@pytest.fixture
def mock_anthropic_client():
    """Mock AsyncAnthropic client"""
    return AsyncMock()


@pytest.fixture
def provider_config():
    """Provider configuration for testing"""
    return LLMProviderConfig(
        model="claude-3-sonnet-20240229",
        api_key_env="ANTHROPIC_API_KEY",
        timeout_s=60.0,
        max_retries=3
    )


@pytest.fixture
def provider(provider_config, mock_anthropic_client):
    """AnthropicProvider with mocked client"""
    with patch('greenlang.intelligence.providers.anthropic.AsyncAnthropic', return_value=mock_anthropic_client):
        provider = AnthropicProvider(provider_config)
        provider.client = mock_anthropic_client
        return provider


def make_anthropic_response(
    text=None,
    tool_uses=None,
    stop_reason="end_turn",
    input_tokens=100,
    output_tokens=50,
    model="claude-3-sonnet-20240229"
):
    """Create mock Anthropic API response"""
    response = Mock()
    response.id = "msg_test123"
    response.model = model
    response.stop_reason = stop_reason

    # Usage
    response.usage = Mock()
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens

    # Content blocks
    content = []
    if text:
        text_block = Mock()
        text_block.type = "text"
        text_block.text = text
        content.append(text_block)

    if tool_uses:
        for tool_use in tool_uses:
            tool_block = Mock()
            tool_block.type = "tool_use"
            tool_block.id = tool_use.get("id", "toolu_123")
            tool_block.name = tool_use["name"]
            tool_block.input = tool_use["input"]
            content.append(tool_block)

    response.content = content

    return response


class TestAnthropicProviderInitialization:
    """Test provider initialization"""

    def test_provider_requires_api_key(self):
        """Provider should raise if API key not found"""
        config = LLMProviderConfig(
            model="claude-3-sonnet-20240229",
            api_key_env="NONEXISTENT_KEY",
            timeout_s=60.0
        )

        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="API key not found"):
                AnthropicProvider(config)

    def test_provider_initializes_with_valid_key(self, provider_config):
        """Provider should initialize with valid API key"""
        with patch('greenlang.intelligence.providers.anthropic.AsyncAnthropic'):
            provider = AnthropicProvider(provider_config)
            assert provider.config.model == "claude-3-sonnet-20240229"

    def test_provider_has_capabilities(self, provider):
        """Provider should expose capabilities"""
        assert provider.capabilities.function_calling is True
        assert provider.capabilities.max_output_tokens > 0


class TestToolCallEmissionPath:
    """Test tool-call emission (CTO Spec: returns tool_calls, no message_json)"""

    @pytest.mark.asyncio
    async def test_tool_call_emission_no_text(self, provider, mock_anthropic_client):
        """Tool calls should be returned without text"""
        tool_use = {
            "id": "toolu_abc123",
            "name": "calculate_emissions",
            "input": {"fuel_type": "diesel", "amount": 100}
        }

        mock_response = make_anthropic_response(
            text=None,
            tool_uses=[tool_use],
            stop_reason="tool_use"
        )

        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        messages = [ChatMessage(role=Role.user, content="Calculate emissions")]
        tools = [ToolDef(
            name="calculate_emissions",
            description="Calculate emissions",
            parameters={"type": "object", "properties": {}}
        )]
        budget = Budget(max_usd=0.50)

        response = await provider.chat(messages=messages, tools=tools, budget=budget)

        # Verify tool calls returned
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "calculate_emissions"
        assert response.tool_calls[0]["arguments"]["fuel_type"] == "diesel"
        assert response.text is None

    @pytest.mark.asyncio
    async def test_tool_call_neutral_format(self, provider, mock_anthropic_client):
        """Tool calls should be in neutral format"""
        tool_use = {
            "id": "toolu_xyz789",
            "name": "get_weather",
            "input": {"location": "SF", "unit": "celsius"}
        }

        mock_response = make_anthropic_response(
            tool_uses=[tool_use],
            stop_reason="tool_use"
        )

        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        messages = [ChatMessage(role=Role.user, content="Test")]
        tools = [ToolDef(name="get_weather", description="Test", parameters={})]
        budget = Budget(max_usd=0.50)

        response = await provider.chat(messages=messages, tools=tools, budget=budget)

        # Verify neutral format: [{"id": str, "name": str, "arguments": dict}]
        assert "id" in response.tool_calls[0]
        assert "name" in response.tool_calls[0]
        assert "arguments" in response.tool_calls[0]
        assert response.tool_calls[0]["id"] == "toolu_xyz789"
        assert isinstance(response.tool_calls[0]["arguments"], dict)


class TestJSONModeFirstTrySuccess:
    """Test JSON-mode success on first try (CTO Spec: cost calls = 1)"""

    @pytest.mark.asyncio
    async def test_json_mode_success_first_attempt(self, provider, mock_anthropic_client):
        """Valid JSON on first attempt should succeed with 1 API call"""
        valid_json = '{"result": 42, "status": "ok"}'
        mock_response = make_anthropic_response(
            text=valid_json,
            stop_reason="end_turn",
            input_tokens=100,
            output_tokens=20
        )

        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        messages = [ChatMessage(role=Role.user, content="Return JSON")]
        json_schema = {
            "type": "object",
            "properties": {
                "result": {"type": "number"},
                "status": {"type": "string"}
            },
            "required": ["result", "status"]
        }
        budget = Budget(max_usd=0.50)

        response = await provider.chat(
            messages=messages,
            json_schema=json_schema,
            budget=budget
        )

        # Verify success
        assert response.text == valid_json
        assert json.loads(response.text)["result"] == 42

        # Verify only 1 API call
        assert mock_anthropic_client.messages.create.call_count == 1

        # Verify cost metered
        assert budget.spent_usd > 0
        assert response.usage.total_tokens == 120


class TestJSONModeRepairSuccess:
    """Test JSON-mode success after repairs (CTO Spec: 2 fails + 1 success → cost calls = 3)"""

    @pytest.mark.asyncio
    async def test_json_repair_after_two_failures(self, provider, mock_anthropic_client):
        """Should succeed on 3rd attempt after 2 repair prompts"""
        invalid_json_1 = '{result: 42}'
        invalid_json_2 = '{"result": "not_a_number"}'
        valid_json = '{"result": 42, "status": "ok"}'

        responses = [
            make_anthropic_response(text=invalid_json_1, output_tokens=10),
            make_anthropic_response(text=invalid_json_2, output_tokens=15),
            make_anthropic_response(text=valid_json, output_tokens=20)
        ]

        mock_anthropic_client.messages.create = AsyncMock(side_effect=responses)

        messages = [ChatMessage(role=Role.user, content="Return JSON")]
        json_schema = {
            "type": "object",
            "properties": {
                "result": {"type": "number"},
                "status": {"type": "string"}
            },
            "required": ["result", "status"]
        }
        budget = Budget(max_usd=0.50)

        response = await provider.chat(
            messages=messages,
            json_schema=json_schema,
            budget=budget,
            metadata={"request_id": "test_repair"}
        )

        # Verify success on 3rd attempt
        assert response.text == valid_json

        # Verify 3 API calls (CTO Spec: cost calls = 3)
        assert mock_anthropic_client.messages.create.call_count == 3

        # Verify cost metered on ALL 3 attempts
        assert budget.spent_tokens == 345  # 100*3 + 10 + 15 + 20


class TestJSONModeFailureAfterRetries:
    """Test JSON-mode failure after >3 retries (CTO Spec: cost calls = 4, raise GLJsonParseError)"""

    @pytest.mark.asyncio
    async def test_json_failure_after_four_attempts(self, provider, mock_anthropic_client):
        """Should fail with GLJsonParseError after 4 failed attempts"""
        invalid_responses = [
            make_anthropic_response(text='{invalid1}', output_tokens=10),
            make_anthropic_response(text='{invalid2}', output_tokens=10),
            make_anthropic_response(text='{invalid3}', output_tokens=10),
            make_anthropic_response(text='{invalid4}', output_tokens=10)
        ]

        mock_anthropic_client.messages.create = AsyncMock(side_effect=invalid_responses)

        messages = [ChatMessage(role=Role.user, content="Return JSON")]
        json_schema = {
            "type": "object",
            "properties": {"result": {"type": "number"}},
            "required": ["result"]
        }
        budget = Budget(max_usd=0.50)

        # Should raise GLJsonParseError after >3 attempts
        with pytest.raises(GLJsonParseError) as exc_info:
            await provider.chat(
                messages=messages,
                json_schema=json_schema,
                budget=budget,
                metadata={"request_id": "test_fail"}
            )

        # Verify error details
        assert exc_info.value.attempts == 4
        assert exc_info.value.request_id == "test_fail"

        # Verify 4 API calls
        assert mock_anthropic_client.messages.create.call_count == 4

        # Verify cost metered on ALL 4 attempts
        assert budget.spent_usd > 0
        assert budget.spent_tokens == 440


class TestBudgetCapEnforcement:
    """Test budget cap exceeded → BudgetExceeded"""

    @pytest.mark.asyncio
    async def test_budget_exceeded_before_call(self, provider, mock_anthropic_client):
        """Should raise BudgetExceeded if estimated cost exceeds cap"""
        messages = [ChatMessage(role=Role.user, content="Write a very long essay")]
        budget = Budget(max_usd=0.0001)

        with pytest.raises(BudgetExceeded):
            await provider.chat(messages=messages, budget=budget)

        # Should fail before API call
        assert mock_anthropic_client.messages.create.call_count == 0

    @pytest.mark.asyncio
    async def test_budget_partial_tracking(self, provider, mock_anthropic_client):
        """Budget should track partial cost"""
        # Use large token counts to ensure cost exceeds budget
        mock_response = make_anthropic_response(
            text="Response",
            input_tokens=1_000_000,  # Large token count
            output_tokens=1_000_000
        )

        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        messages = [ChatMessage(role=Role.user, content="Test")]
        # claude-3-sonnet: $3/1M input, $15/1M output
        # 1M + 1M tokens = $18, so budget of $20 fits one call but not two
        budget = Budget(max_usd=20.0)

        # First call succeeds and uses most of budget
        await provider.chat(messages=messages, budget=budget)
        first_cost = budget.spent_usd
        assert first_cost > 17.0  # First call costs ~$18

        # Second call should exceed (would need another ~$18)
        with pytest.raises(BudgetExceeded):
            await provider.chat(messages=messages, budget=budget)

        # Budget shows only first call cost
        assert budget.spent_usd == first_cost


class TestErrorMapping:
    """Test error mapping to exception taxonomy"""

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, provider, mock_anthropic_client):
        """Should map rate limit to ProviderRateLimit"""
        try:
            from anthropic import APIStatusError
        except ImportError:
            pytest.skip("anthropic package not installed")

        # Create proper APIStatusError with all required parameters
        mock_response = Mock(status_code=429)
        error = APIStatusError(
            message="Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}}
        )
        error.status_code = 429

        mock_anthropic_client.messages.create = AsyncMock(side_effect=error)

        messages = [ChatMessage(role=Role.user, content="Test")]
        budget = Budget(max_usd=0.50)

        with pytest.raises(ProviderRateLimit):
            await provider.chat(messages=messages, budget=budget)

    @pytest.mark.asyncio
    async def test_server_error_5xx(self, provider, mock_anthropic_client):
        """Should map 5xx to ProviderServerError"""
        try:
            from anthropic import APIStatusError
        except ImportError:
            pytest.skip("anthropic package not installed")

        # Create proper APIStatusError with all required parameters
        mock_response = Mock(status_code=500)
        error = APIStatusError(
            message="Internal server error",
            response=mock_response,
            body={"error": {"message": "Internal server error"}}
        )
        error.status_code = 500

        mock_anthropic_client.messages.create = AsyncMock(side_effect=error)

        messages = [ChatMessage(role=Role.user, content="Test")]
        budget = Budget(max_usd=0.50)

        with pytest.raises(ProviderServerError):
            await provider.chat(messages=messages, budget=budget)


class TestUsageParsing:
    """Test usage parsing from Anthropic responses"""

    @pytest.mark.asyncio
    async def test_usage_parsed_from_response(self, provider, mock_anthropic_client):
        """Should parse usage from API response"""
        mock_response = make_anthropic_response(
            text="Test response",
            input_tokens=250,
            output_tokens=100
        )

        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        messages = [ChatMessage(role=Role.user, content="Test")]
        budget = Budget(max_usd=0.50)

        response = await provider.chat(messages=messages, budget=budget)

        # Verify parsed usage
        assert response.usage.prompt_tokens == 250
        assert response.usage.completion_tokens == 100
        assert response.usage.total_tokens == 350
        assert response.usage.cost_usd > 0

    @pytest.mark.asyncio
    async def test_cost_calculation_accuracy(self, provider, mock_anthropic_client):
        """Cost should be calculated using model pricing"""
        mock_response = make_anthropic_response(
            text="Test",
            input_tokens=1_000_000,
            output_tokens=1_000_000
        )

        mock_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

        messages = [ChatMessage(role=Role.user, content="Test")]
        budget = Budget(max_usd=50.0)

        response = await provider.chat(messages=messages, budget=budget)

        # claude-3-sonnet: $3/1M input, $15/1M output
        expected_cost = 3.0 + 15.0  # $18.00

        assert abs(response.usage.cost_usd - expected_cost) < 0.01


class TestRetryLogic:
    """Test exponential backoff retry"""

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self, provider, mock_anthropic_client):
        """Should retry with exponential backoff"""
        try:
            from anthropic import APIStatusError
        except ImportError:
            pytest.skip("anthropic package not installed")

        # Create proper APIStatusError with all required parameters
        mock_response_error = Mock(status_code=429)
        error = APIStatusError(
            message="Rate limit",
            response=mock_response_error,
            body={"error": {"message": "Rate limit"}}
        )
        error.status_code = 429

        mock_response = make_anthropic_response(text="Success")

        mock_anthropic_client.messages.create = AsyncMock(
            side_effect=[error, error, mock_response]
        )

        messages = [ChatMessage(role=Role.user, content="Test")]
        budget = Budget(max_usd=0.50)

        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            response = await provider.chat(messages=messages, budget=budget)

            # Should have slept twice
            assert mock_sleep.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
