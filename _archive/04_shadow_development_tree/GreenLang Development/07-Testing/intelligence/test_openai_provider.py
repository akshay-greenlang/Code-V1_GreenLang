# -*- coding: utf-8 -*-
"""
Mocked tests for OpenAIProvider

Tests CTO specification compliance:
- Tool-call emission path (returns tool_calls, no message_json)
- JSON-mode success on first try (cost calls = 1)
- JSON-mode success after repairs (2 fails + 1 success → cost calls = 3)
- JSON-mode failure after >3 retries → GLJsonParseError (cost calls = 4)
- Budget cap exceeded → GLBudgetExceededError with partial cost
- 429 and 5xx error mapping
- Usage parsing vs estimation paths
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json
import os

# Set environment variable for all tests
os.environ['OPENAI_API_KEY'] = 'test-key-123'

from greenlang.intelligence.providers.openai import OpenAIProvider
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
def mock_openai_client():
    """Mock AsyncOpenAI client"""
    return AsyncMock()


@pytest.fixture
def provider_config():
    """Provider configuration for testing"""
    return LLMProviderConfig(
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        timeout_s=30.0,
        max_retries=3
    )


@pytest.fixture
def provider(provider_config, mock_openai_client):
    """OpenAIProvider with mocked client"""
    with patch('greenlang.intelligence.providers.openai.AsyncOpenAI', return_value=mock_openai_client):
        provider = OpenAIProvider(provider_config)
        provider.client = mock_openai_client
        return provider


def make_openai_response(
    text=None,
    tool_calls=None,
    finish_reason="stop",
    prompt_tokens=100,
    completion_tokens=50,
    model="gpt-4o-mini"
):
    """Create mock OpenAI API response"""
    response = Mock()
    response.id = "chatcmpl-test123"
    response.model = model

    # Usage
    response.usage = Mock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = prompt_tokens + completion_tokens

    # Choices
    choice = Mock()
    choice.finish_reason = finish_reason
    choice.message = Mock()
    choice.message.content = text
    choice.message.tool_calls = tool_calls

    response.choices = [choice]

    return response


def make_tool_call(name="test_tool", arguments=None, tool_id="call_123"):
    """Create mock OpenAI tool call"""
    tool_call = Mock()
    tool_call.id = tool_id
    tool_call.type = "function"
    tool_call.function = Mock()
    tool_call.function.name = name
    tool_call.function.arguments = json.dumps(arguments or {"param": "value"})
    return tool_call


class TestOpenAIProviderInitialization:
    """Test provider initialization"""

    def test_provider_requires_api_key(self):
        """Provider should raise if API key not found"""
        config = LLMProviderConfig(
            model="gpt-4o-mini",
            api_key_env="NONEXISTENT_KEY",
            timeout_s=30.0
        )

        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="API key not found"):
                OpenAIProvider(config)

    def test_provider_initializes_with_valid_key(self, provider_config):
        """Provider should initialize with valid API key"""
        with patch('greenlang.intelligence.providers.openai.AsyncOpenAI'):
            provider = OpenAIProvider(provider_config)
            assert provider.config.model == "gpt-4o-mini"
            assert provider.model_base == "gpt-4o-mini"

    def test_provider_has_capabilities(self, provider):
        """Provider should expose capabilities"""
        assert provider.capabilities.function_calling is True
        assert provider.capabilities.json_schema_mode is True
        assert provider.capabilities.max_output_tokens > 0


class TestToolCallEmissionPath:
    """Test tool-call emission (CTO Spec: returns tool_calls, no message_json)"""

    @pytest.mark.asyncio
    async def test_tool_call_emission_no_text_response(self, provider, mock_openai_client):
        """Tool calls should be returned without text response"""
        # Mock response with tool call
        tool_call = make_tool_call(
            name="calculate_emissions",
            arguments={"fuel_type": "diesel", "amount": 100}
        )
        mock_response = make_openai_response(
            text=None,  # No text when tool call
            tool_calls=[tool_call],
            finish_reason="tool_calls"
        )

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Call provider
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
        assert response.text is None  # No text when tool call

    @pytest.mark.asyncio
    async def test_tool_call_neutral_format(self, provider, mock_openai_client):
        """Tool calls should be in neutral format [{"id": str, "name": str, "arguments": dict}]"""
        tool_call = make_tool_call(
            name="get_weather",
            arguments={"location": "SF", "unit": "celsius"},
            tool_id="call_abc123"
        )
        mock_response = make_openai_response(
            tool_calls=[tool_call],
            finish_reason="tool_calls"
        )

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [ChatMessage(role=Role.user, content="Test")]
        tools = [ToolDef(name="get_weather", description="Test", parameters={})]
        budget = Budget(max_usd=0.50)

        response = await provider.chat(messages=messages, tools=tools, budget=budget)

        # Verify neutral format
        assert "id" in response.tool_calls[0]
        assert "name" in response.tool_calls[0]
        assert "arguments" in response.tool_calls[0]
        assert response.tool_calls[0]["id"] == "call_abc123"
        assert isinstance(response.tool_calls[0]["arguments"], dict)

    @pytest.mark.asyncio
    async def test_cost_metered_on_tool_call(self, provider, mock_openai_client):
        """Cost should be metered even for tool calls"""
        tool_call = make_tool_call()
        mock_response = make_openai_response(
            tool_calls=[tool_call],
            finish_reason="tool_calls",
            prompt_tokens=200,
            completion_tokens=50
        )

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [ChatMessage(role=Role.user, content="Test")]
        tools = [ToolDef(name="test_tool", description="Test", parameters={})]
        budget = Budget(max_usd=0.50)

        response = await provider.chat(messages=messages, tools=tools, budget=budget)

        # Verify cost metered
        assert response.usage.total_tokens == 250
        assert response.usage.cost_usd > 0
        assert budget.spent_usd > 0


class TestJSONModeFirstTrySuccess:
    """Test JSON-mode success on first try (CTO Spec: cost calls = 1)"""

    @pytest.mark.asyncio
    async def test_json_mode_success_first_attempt(self, provider, mock_openai_client):
        """Valid JSON on first attempt should succeed with 1 API call"""
        valid_json = '{"result": 42, "status": "ok"}'
        mock_response = make_openai_response(
            text=valid_json,
            finish_reason="stop",
            prompt_tokens=100,
            completion_tokens=20
        )

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

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
        assert mock_openai_client.chat.completions.create.call_count == 1

        # Verify cost metered once
        assert budget.spent_usd > 0
        assert response.usage.total_tokens == 120


class TestJSONModeRepairSuccess:
    """Test JSON-mode success after repairs (CTO Spec: 2 fails + 1 success → cost calls = 3)"""

    @pytest.mark.asyncio
    async def test_json_repair_after_two_failures(self, provider, mock_openai_client):
        """Should succeed on 3rd attempt after 2 repair prompts"""
        # Mock 3 responses: 2 invalid, 1 valid
        invalid_json_1 = '{result: 42}'  # Missing quotes
        invalid_json_2 = '{"result": "not_a_number"}'  # Wrong type
        valid_json = '{"result": 42, "status": "ok"}'

        responses = [
            make_openai_response(text=invalid_json_1, completion_tokens=10),
            make_openai_response(text=invalid_json_2, completion_tokens=15),
            make_openai_response(text=valid_json, completion_tokens=20)
        ]

        mock_openai_client.chat.completions.create = AsyncMock(side_effect=responses)

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
        assert json.loads(response.text)["result"] == 42

        # Verify 3 API calls (CTO Spec: cost calls = 3)
        assert mock_openai_client.chat.completions.create.call_count == 3

        # Verify cost metered on ALL 3 attempts
        assert budget.spent_tokens > 0
        # Total tokens = 100*3 (prompt) + 10 + 15 + 20 (completions) = 345
        expected_tokens = 300 + 45
        assert budget.spent_tokens == expected_tokens


class TestJSONModeFailureAfterRetries:
    """Test JSON-mode failure after >3 retries (CTO Spec: cost calls = 4, raise GLJsonParseError)"""

    @pytest.mark.asyncio
    async def test_json_failure_after_four_attempts(self, provider, mock_openai_client):
        """Should fail with GLJsonParseError after 4 failed attempts"""
        # Mock 4 invalid responses
        invalid_responses = [
            make_openai_response(text='{invalid1}', completion_tokens=10),
            make_openai_response(text='{invalid2}', completion_tokens=10),
            make_openai_response(text='{invalid3}', completion_tokens=10),
            make_openai_response(text='{invalid4}', completion_tokens=10)
        ]

        mock_openai_client.chat.completions.create = AsyncMock(side_effect=invalid_responses)

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

        # Verify 4 API calls (CTO Spec: cost calls = 4)
        assert mock_openai_client.chat.completions.create.call_count == 4

        # Verify cost metered on ALL 4 attempts (even failed ones)
        assert budget.spent_usd > 0
        assert budget.spent_tokens == 440  # 100*4 + 10*4


class TestBudgetCapEnforcement:
    """Test budget cap exceeded → GLBudgetExceededError with partial cost"""

    @pytest.mark.asyncio
    async def test_budget_exceeded_before_call(self, provider, mock_openai_client):
        """Should raise BudgetExceeded if estimated cost would exceed cap"""
        messages = [ChatMessage(role=Role.user, content="Write a very long essay")]
        budget = Budget(max_usd=0.0001)  # Very low budget

        with pytest.raises(BudgetExceeded) as exc_info:
            await provider.chat(messages=messages, budget=budget)

        # Should fail before API call
        assert mock_openai_client.chat.completions.create.call_count == 0

        # Budget should show estimated cost
        assert exc_info.value.spent_usd > 0
        assert exc_info.value.max_usd == 0.0001

    @pytest.mark.asyncio
    async def test_budget_partial_tracking_on_failure(self, provider, mock_openai_client):
        """Budget should track partial cost even if request fails"""
        # Use large token counts to ensure cost exceeds tiny budget
        mock_response_1 = make_openai_response(
            text="Response 1",
            prompt_tokens=500_000,  # Large token count
            completion_tokens=500_000
        )

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response_1)

        messages = [ChatMessage(role=Role.user, content="Test")]
        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        # 500k + 500k tokens = ~$0.375, so budget of $0.40 fits one call but not two
        budget = Budget(max_usd=0.40)

        # First call succeeds and uses most of budget
        response_1 = await provider.chat(messages=messages, budget=budget)
        first_cost = budget.spent_usd
        assert first_cost > 0.35  # First call costs ~$0.375

        # Second call should exceed budget (would need another ~$0.375)
        with pytest.raises(BudgetExceeded):
            await provider.chat(messages=messages, budget=budget)

        # Budget should still show only first call cost
        assert budget.spent_usd == first_cost


class TestErrorMapping429And5xx:
    """Test 429 and 5xx error mapping to correct exceptions"""

    @pytest.mark.asyncio
    async def test_rate_limit_error_429(self, provider, mock_openai_client):
        """429 should map to ProviderRateLimit"""
        from openai import RateLimitError

        # Create proper RateLimitError with required parameters
        mock_response = Mock(status_code=429)
        error = RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}}
        )
        mock_openai_client.chat.completions.create = AsyncMock(side_effect=error)

        messages = [ChatMessage(role=Role.user, content="Test")]
        budget = Budget(max_usd=0.50)

        with pytest.raises(ProviderRateLimit):
            await provider.chat(messages=messages, budget=budget)

    @pytest.mark.asyncio
    async def test_server_error_5xx(self, provider, mock_openai_client):
        """5xx should map to ProviderServerError"""
        from openai import InternalServerError

        # Create proper InternalServerError with required parameters
        mock_response = Mock(status_code=500)
        error = InternalServerError(
            message="Internal server error",
            response=mock_response,
            body={"error": {"message": "Internal server error"}}
        )
        mock_openai_client.chat.completions.create = AsyncMock(side_effect=error)

        messages = [ChatMessage(role=Role.user, content="Test")]
        budget = Budget(max_usd=0.50)

        with pytest.raises(ProviderServerError):
            await provider.chat(messages=messages, budget=budget)

    @pytest.mark.asyncio
    async def test_timeout_error(self, provider, mock_openai_client):
        """Timeout should map to ProviderTimeout"""
        from openai import APITimeoutError

        # Create proper APITimeoutError
        error = APITimeoutError(request=Mock())
        mock_openai_client.chat.completions.create = AsyncMock(side_effect=error)

        messages = [ChatMessage(role=Role.user, content="Test")]
        budget = Budget(max_usd=0.50)

        with pytest.raises(ProviderTimeout):
            await provider.chat(messages=messages, budget=budget)

    @pytest.mark.asyncio
    async def test_auth_error_401(self, provider, mock_openai_client):
        """401 should map to ProviderAuthError"""
        from openai import AuthenticationError

        # Create proper AuthenticationError with required parameters
        mock_response = Mock(status_code=401)
        error = AuthenticationError(
            message="Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}}
        )
        mock_openai_client.chat.completions.create = AsyncMock(side_effect=error)

        messages = [ChatMessage(role=Role.user, content="Test")]
        budget = Budget(max_usd=0.50)

        with pytest.raises(ProviderAuthError):
            await provider.chat(messages=messages, budget=budget)


class TestUsageParsingVsEstimation:
    """Test usage parsing from response vs estimation"""

    @pytest.mark.asyncio
    async def test_usage_parsed_from_response(self, provider, mock_openai_client):
        """Should parse usage from API response when available"""
        mock_response = make_openai_response(
            text="Test response",
            prompt_tokens=250,
            completion_tokens=100
        )

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [ChatMessage(role=Role.user, content="Test")]
        budget = Budget(max_usd=0.50)

        response = await provider.chat(messages=messages, budget=budget)

        # Verify parsed usage (not estimated)
        assert response.usage.prompt_tokens == 250
        assert response.usage.completion_tokens == 100
        assert response.usage.total_tokens == 350
        assert response.usage.cost_usd > 0

    @pytest.mark.asyncio
    async def test_cost_calculation_accuracy(self, provider, mock_openai_client):
        """Cost should be calculated using model pricing"""
        mock_response = make_openai_response(
            text="Test",
            prompt_tokens=1000,
            completion_tokens=1000,
            model="gpt-4o-mini"
        )

        mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [ChatMessage(role=Role.user, content="Test")]
        budget = Budget(max_usd=0.50)

        response = await provider.chat(messages=messages, budget=budget)

        # gpt-4o-mini: $0.15/1M input, $0.60/1M output
        expected_cost = (1000 / 1_000_000 * 0.15) + (1000 / 1_000_000 * 0.60)
        expected_cost = expected_cost  # $0.00075

        assert abs(response.usage.cost_usd - expected_cost) < 0.000001


class TestRetryLogic:
    """Test exponential backoff retry logic"""

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self, provider, mock_openai_client):
        """Should retry with exponential backoff on retryable errors"""
        from openai import RateLimitError

        # Create proper RateLimitError for retry tests
        mock_response_error = Mock(status_code=429)
        error = RateLimitError(
            message="Rate limit",
            response=mock_response_error,
            body={"error": {"message": "Rate limit"}}
        )
        mock_response = make_openai_response(text="Success")

        mock_openai_client.chat.completions.create = AsyncMock(
            side_effect=[error, error, mock_response]
        )

        messages = [ChatMessage(role=Role.user, content="Test")]
        budget = Budget(max_usd=0.50)

        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            response = await provider.chat(messages=messages, budget=budget)

            # Should have slept twice (exponential backoff)
            assert mock_sleep.call_count == 2
            # First retry: 1s, second retry: 2s
            assert mock_sleep.call_args_list[0][0][0] == 1.0
            assert mock_sleep.call_args_list[1][0][0] == 2.0

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, provider, mock_openai_client):
        """Should fail after max_retries exceeded"""
        from openai import RateLimitError

        # Create proper RateLimitError
        mock_response_error = Mock(status_code=429)
        error = RateLimitError(
            message="Rate limit",
            response=mock_response_error,
            body={"error": {"message": "Rate limit"}}
        )
        mock_openai_client.chat.completions.create = AsyncMock(side_effect=error)

        messages = [ChatMessage(role=Role.user, content="Test")]
        budget = Budget(max_usd=0.50)

        with pytest.raises(ProviderRateLimit):
            await provider.chat(messages=messages, budget=budget)

        # Should have tried max_retries + 1 times (initial + 3 retries = 4)
        assert mock_openai_client.chat.completions.create.call_count == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
