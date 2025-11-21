# -*- coding: utf-8 -*-
"""
Chat Session Orchestration

ChatSession coordinates LLM interactions with:
- Provider delegation (calls LLMProvider.chat())
- Budget enforcement (prevents runaway costs)
- Telemetry emission (audit trail for compliance)
- Error handling (classify and log provider errors)

Key responsibilities:
1. Validate inputs before provider call
2. Enforce budget caps (raise BudgetExceeded if would exceed)
3. Delegate to provider with error handling
4. Emit telemetry events (LLM call, usage, errors)
5. Update budget tracker with actual usage

Type-safe, async-first design for production workloads.
"""

from __future__ import annotations
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

from greenlang.intelligence.providers.base import LLMProvider
from greenlang.intelligence.schemas.messages import ChatMessage
from greenlang.intelligence.schemas.tools import ToolDef
from greenlang.intelligence.schemas.responses import ChatResponse
from greenlang.intelligence.schemas.jsonschema import JSONSchema
from greenlang.intelligence.runtime.budget import Budget, BudgetExceeded
from greenlang.intelligence.providers.errors import (
    ProviderError,
    classify_provider_error,
)


class ChatSession:
    """
    Orchestrates LLM calls through a provider with budget and telemetry

    ChatSession is the main entry point for LLM interactions in GreenLang.
    It wraps an LLMProvider with:
    - Budget enforcement (cost/token caps)
    - Telemetry emission (audit logs for compliance)
    - Error handling (classify provider errors)
    - Input validation (tools, json_schema, messages)

    Usage:
        # Create session with provider
        provider = OpenAIProvider(config)
        session = ChatSession(provider)

        # Simple chat
        response = await session.chat(
            messages=[
                ChatMessage(role=Role.user, content="Calculate emissions for 100 gallons diesel")
            ],
            budget=Budget(max_usd=0.50)
        )
        print(response.text)

        # With function calling
        response = await session.chat(
            messages=[...],
            tools=[
                ToolDef(
                    name="calculate_fuel_emissions",
                    description="Calculate CO2e emissions from fuel combustion",
                    parameters={...}
                )
            ],
            budget=Budget(max_usd=0.50)
        )
        if response.tool_calls:
            print(f"LLM requested tool: {response.tool_calls[0]['name']}")

        # With telemetry
        from greenlang.intelligence.runtime.telemetry import IntelligenceTelemetry, FileEmitter

        telemetry = IntelligenceTelemetry(
            emitter=FileEmitter("logs/intelligence.jsonl")
        )
        session = ChatSession(provider, telemetry=telemetry)

        # All LLM calls will be logged to file
        response = await session.chat(
            messages=[...],
            budget=Budget(max_usd=0.50)
        )

    Architecture:
        ChatSession (orchestration) -> LLMProvider (vendor API) -> OpenAI/Anthropic/etc.

        ChatSession adds:
        - Budget checks BEFORE and AFTER provider call
        - Telemetry emission (what was called, how much it cost, what it returned)
        - Error classification (normalize provider errors)
        - Input validation (messages not empty, budget set, etc.)
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        telemetry: Optional[Any] = None,
    ):
        """
        Initialize chat session

        Args:
            provider: LLM provider (OpenAI, Anthropic, etc.)
            telemetry: Optional telemetry emitter or callback
                - If IntelligenceTelemetry: Uses log_llm_call()
                - If callable: Calls with (event_type, metadata)
                - If None: No telemetry (default)

        Example:
            # Without telemetry
            session = ChatSession(provider)

            # With IntelligenceTelemetry
            from greenlang.intelligence.runtime.telemetry import IntelligenceTelemetry
            telemetry = IntelligenceTelemetry()
            session = ChatSession(provider, telemetry=telemetry)

            # With callback
            def log_event(event_type: str, metadata: dict):
                print(f"{event_type}: {metadata}")
            session = ChatSession(provider, telemetry=log_event)
        """
        self.provider = provider
        self.telemetry = telemetry

    async def chat(
        self,
        messages: List[ChatMessage],
        *,
        tools: Optional[List[ToolDef]] = None,
        json_schema: Optional[JSONSchema] = None,
        budget: Budget,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        tool_choice: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> ChatResponse:
        """
        Execute chat completion with budget enforcement and telemetry

        This is the main entry point for LLM interactions. It:
        1. Validates inputs (messages not empty, budget set)
        2. Delegates to provider.chat() with error handling
        3. Updates budget with actual usage
        4. Emits telemetry event with usage/cost/latency
        5. Returns normalized ChatResponse

        Args:
            messages: Conversation history (must be non-empty)
            tools: Available tools for function calling (None = no tools)
            json_schema: JSON schema for response validation (None = no schema)
            budget: Budget tracker (enforces cost/token caps)
            temperature: Sampling temperature (0.0 = deterministic, 2.0 = creative)
            top_p: Nucleus sampling threshold (1.0 = disabled)
            seed: Random seed for reproducibility (None = random)
            tool_choice: Tool selection strategy ("auto"/"none"/"required"/tool_name)
            metadata: Provider-specific metadata (e.g., user ID for tracking)
            **kwargs: Additional provider-specific parameters

        Returns:
            ChatResponse with text, tool calls, usage, and finish reason

        Raises:
            ValueError: If messages empty or invalid
            BudgetExceeded: If request would exceed budget cap
            ProviderError: If provider call fails (ProviderAuthError, ProviderRateLimit, etc.)

        Example:
            # Simple text completion
            response = await session.chat(
                messages=[
                    ChatMessage(role=Role.system, content="You are a climate analyst"),
                    ChatMessage(role=Role.user, content="Explain carbon offsets")
                ],
                budget=Budget(max_usd=0.10)
            )
            print(f"Response: {response.text}")
            print(f"Cost: ${response.usage.cost_usd:.4f}")

            # With function calling
            response = await session.chat(
                messages=[
                    ChatMessage(role=Role.user, content="Calculate emissions for 50 gallons gasoline")
                ],
                tools=[
                    ToolDef(
                        name="calculate_fuel_emissions",
                        description="Calculate CO2e emissions from fuel combustion",
                        parameters={
                            "type": "object",
                            "properties": {
                                "fuel_type": {"type": "string", "enum": ["gasoline", "diesel"]},
                                "amount": {"type": "number", "minimum": 0}
                            },
                            "required": ["fuel_type", "amount"]
                        }
                    )
                ],
                tool_choice="auto",
                budget=Budget(max_usd=0.20)
            )
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                print(f"LLM wants to call: {tool_call['name']}({tool_call['arguments']})")

            # With JSON schema validation
            response = await session.chat(
                messages=[
                    ChatMessage(role=Role.user, content="Return the grid intensity for CA as JSON")
                ],
                json_schema={
                    "type": "object",
                    "properties": {
                        "region": {"type": "string"},
                        "intensity": {"type": "number"},
                        "unit": {"type": "string"}
                    },
                    "required": ["region", "intensity", "unit"]
                },
                budget=Budget(max_usd=0.10)
            )
            import json
            data = json.loads(response.text)
            print(f"Region: {data['region']}, Intensity: {data['intensity']} {data['unit']}")

            # Error handling
            try:
                response = await session.chat(
                    messages=[...],
                    budget=Budget(max_usd=0.01)  # Very tight budget
                )
            except BudgetExceeded as e:
                print(f"Budget exceeded: {e}")
            except ProviderAuthError as e:
                print(f"Auth error: {e}")
            except ProviderRateLimit as e:
                print(f"Rate limited: {e}, retry after {e.retry_after}s")
        """
        # 1. Validate inputs
        if not messages:
            raise ValueError("messages cannot be empty")

        # 2. Check if budget is already exhausted
        if budget.is_exhausted:
            raise BudgetExceeded(
                message="Budget already exhausted before call",
                spent_usd=budget.spent_usd,
                max_usd=budget.max_usd,
                spent_tokens=budget.spent_tokens,
                max_tokens=budget.max_tokens,
            )

        # 3. Record start time for latency tracking
        start_time_ms = int(time.time() * 1000)

        # 4. Call provider with error handling
        try:
            response = await self.provider.chat(
                messages=messages,
                tools=tools,
                json_schema=json_schema,
                budget=budget,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                tool_choice=tool_choice,
                metadata=metadata,
                **kwargs,
            )
        except BudgetExceeded as e:
            # Budget exceeded during provider call, log and re-raise
            self._emit_error_telemetry(
                error_type="budget_exceeded",
                error_message=str(e),
                latency_ms=int(time.time() * 1000) - start_time_ms,
            )
            raise
        except ProviderError as e:
            # Provider error (already classified), log and re-raise
            self._emit_error_telemetry(
                error_type=e.__class__.__name__,
                error_message=str(e),
                latency_ms=int(time.time() * 1000) - start_time_ms,
                provider=e.provider,
                status_code=e.status_code,
            )
            raise
        except Exception as e:
            # Unknown error, classify and wrap
            classified_error = classify_provider_error(
                error=e,
                provider=(
                    self.provider.config.model.split("-")[0]
                    if hasattr(self.provider, "config")
                    else "unknown"
                ),
            )
            self._emit_error_telemetry(
                error_type=classified_error.__class__.__name__,
                error_message=str(classified_error),
                latency_ms=int(time.time() * 1000) - start_time_ms,
            )
            raise classified_error from e

        # 5. Calculate latency
        latency_ms = int(time.time() * 1000) - start_time_ms

        # 6. Emit telemetry (success case)
        self._emit_success_telemetry(
            messages=messages,
            response=response,
            latency_ms=latency_ms,
        )

        # 7. Return response (budget already updated by provider)
        return response

    def _emit_success_telemetry(
        self,
        messages: List[ChatMessage],
        response: ChatResponse,
        latency_ms: int,
    ) -> None:
        """
        Emit telemetry event for successful LLM call

        Args:
            messages: Input messages
            response: LLM response
            latency_ms: Call latency in milliseconds
        """
        if self.telemetry is None:
            return

        try:
            # Construct prompt text from messages
            prompt_text = "\n".join(
                f"{msg.role}: {msg.content or '(tool call)'}" for msg in messages
            )
            response_text = response.text or "(tool calls)"

            # Extract tool call names
            tool_call_names = [tc.get("name", "unknown") for tc in response.tool_calls]

            # Check if telemetry is IntelligenceTelemetry or callable
            if hasattr(self.telemetry, "log_llm_call"):
                # IntelligenceTelemetry object
                self.telemetry.log_llm_call(
                    provider=response.provider_info.provider,
                    model=response.provider_info.model,
                    prompt=prompt_text,
                    response=response_text,
                    usage=response.usage,
                    tool_calls=tool_call_names,
                    latency_ms=latency_ms,
                    finish_reason=response.finish_reason,
                )
            elif callable(self.telemetry):
                # Callback function
                self.telemetry(
                    "llm.chat",
                    {
                        "provider": response.provider_info.provider,
                        "model": response.provider_info.model,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                        "cost_usd": response.usage.cost_usd,
                        "latency_ms": latency_ms,
                        "tool_calls": tool_call_names,
                        "finish_reason": response.finish_reason,
                    },
                )
        except Exception:
            # Never fail on telemetry errors
            pass

    def _emit_error_telemetry(
        self,
        error_type: str,
        error_message: str,
        latency_ms: int,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        """
        Emit telemetry event for failed LLM call

        Args:
            error_type: Error class name (e.g., "ProviderRateLimit")
            error_message: Error message
            latency_ms: Call latency in milliseconds
            provider: Provider name (if available)
            status_code: HTTP status code (if available)
        """
        if self.telemetry is None:
            return

        try:
            if callable(self.telemetry):
                # Callback function
                self.telemetry(
                    "llm.error",
                    {
                        "error_type": error_type,
                        "error_message": error_message,
                        "latency_ms": latency_ms,
                        "provider": provider,
                        "status_code": status_code,
                    },
                )
        except Exception:
            # Never fail on telemetry errors
            pass
