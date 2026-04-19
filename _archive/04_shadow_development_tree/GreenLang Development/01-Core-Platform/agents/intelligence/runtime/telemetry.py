# -*- coding: utf-8 -*-
"""
Telemetry and Audit Logging

Immutable audit trail for:
- LLM inference events (model, tokens, cost, latency)
- Tool executions (which tools, arguments, results)
- Budget tracking (spending per agent/workflow)
- Security events (prompt injection attempts, hallucinations detected)

Compliance features:
- GDPR: Prompt hashes (not full prompts) to avoid PII logging
- SOC 2: Immutable append-only audit log
- Financial audit: Cost attribution with timestamps
"""

from __future__ import annotations
import hashlib
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Callable
from pydantic import BaseModel, Field
from greenlang.utilities.determinism import DeterministicClock


class TelemetryEvent(BaseModel):
    """
    Single telemetry event

    Base class for all telemetry events (LLM calls, tool executions, etc.)
    """

    event_type: str = Field(description="Event type (llm.chat, tool.execute, etc.)")
    timestamp: str = Field(
        default_factory=lambda: DeterministicClock.utcnow().isoformat(),
        description="ISO 8601 timestamp (UTC)",
    )
    run_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("GL_RUN_ID"),
        description="GreenLang run ID (for correlation)",
    )
    agent_id: Optional[str] = Field(
        default_factory=lambda: os.getenv("GL_AGENT_ID"),
        description="Agent ID (for cost attribution)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Event-specific metadata"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "event_type": "llm.chat",
                    "timestamp": "2025-10-01T12:34:56.789Z",
                    "run_id": "run_abc123",
                    "agent_id": "fuel_agent",
                    "metadata": {
                        "provider": "openai",
                        "model": "gpt-4-0613",
                        "tokens": 1650,
                        "cost_usd": 0.0234,
                    },
                }
            ]
        }


class LLMEvent(TelemetryEvent):
    """
    LLM inference event

    Records:
    - Which model was used
    - Prompt hash (not full prompt, for GDPR compliance)
    - Token usage and cost
    - Tool calls requested
    - Latency
    """

    event_type: str = "llm.chat"
    provider: str = Field(description="LLM provider (openai, anthropic)")
    model: str = Field(description="Model name (gpt-4-0613, claude-2)")
    prompt_hash: str = Field(description="SHA-256 hash of prompt (not full prompt)")
    response_hash: str = Field(description="SHA-256 hash of response")
    prompt_tokens: int = Field(description="Input tokens")
    completion_tokens: int = Field(description="Output tokens")
    total_tokens: int = Field(description="Total tokens")
    cost_usd: float = Field(description="Cost in USD")
    latency_ms: int = Field(description="Latency in milliseconds")
    tool_calls: List[str] = Field(
        default_factory=list, description="Tool names requested"
    )
    finish_reason: str = Field(description="Why generation stopped")

    @staticmethod
    def hash_text(text: str) -> str:
        """Generate SHA-256 hash of text (for GDPR-compliant logging)"""
        return hashlib.sha256(text.encode()).hexdigest()


class ToolEvent(TelemetryEvent):
    """
    Tool execution event

    Records:
    - Which tool was executed
    - Argument hash (not full arguments, may contain PII)
    - Result hash
    - Success/failure
    - Latency
    """

    event_type: str = "tool.execute"
    tool_name: str = Field(description="Tool name")
    arguments_hash: str = Field(description="SHA-256 hash of arguments")
    result_hash: str = Field(description="SHA-256 hash of result")
    success: bool = Field(description="Whether tool execution succeeded")
    latency_ms: int = Field(description="Execution time in milliseconds")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class BudgetEvent(TelemetryEvent):
    """Budget tracking event"""

    event_type: str = "budget.update"
    spent_usd: float = Field(description="USD spent so far")
    max_usd: float = Field(description="Budget cap")
    spent_tokens: int = Field(description="Tokens consumed")
    remaining_usd: float = Field(description="Remaining budget")


class SecurityEvent(TelemetryEvent):
    """
    Security event (prompt injection, hallucination, etc.)

    Critical for audit trail in regulated environments.
    """

    event_type: str = "security.alert"
    alert_type: str = Field(
        description="Alert type (prompt_injection, hallucination, etc.)"
    )
    severity: str = Field(description="Severity (low, medium, high, critical)")
    details: str = Field(description="Alert details")
    blocked: bool = Field(description="Whether action was blocked")


class TelemetryEmitter(Protocol):
    """
    Interface for telemetry emitters

    Implementations:
    - NoOpEmitter: Does nothing (default, for testing)
    - ConsoleEmitter: Prints to stdout
    - FileEmitter: Appends to JSON Lines file
    - CloudEmitter: Sends to monitoring service (Datadog, CloudWatch, etc.)
    """

    def emit(self, event: TelemetryEvent) -> None:
        """Emit a telemetry event"""
        ...


class NoOpEmitter:
    """No-op emitter (does nothing)"""

    def emit(self, event: TelemetryEvent) -> None:
        pass


class ConsoleEmitter:
    """Emits events to stdout (for development)"""

    def emit(self, event: TelemetryEvent) -> None:
        print(f"[TELEMETRY] {event.event_type}: {event.model_dump_json()}")


class FileEmitter:
    """
    Appends events to JSON Lines file

    Format: One JSON object per line (JSON Lines / NDJSON)
    Enables:
    - Grep-friendly logs
    - Streaming log processing
    - Immutable audit trail
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def emit(self, event: TelemetryEvent) -> None:
        try:
            with open(self.file_path, "a") as f:
                f.write(event.model_dump_json() + "\n")
        except Exception as e:
            # Never fail on telemetry errors
            print(f"[TELEMETRY ERROR] Failed to write event: {e}")


class IntelligenceTelemetry:
    """
    Main telemetry interface for Intelligence Layer

    Usage:
        telemetry = IntelligenceTelemetry(
            emitter=FileEmitter("logs/intelligence_audit.jsonl")
        )

        # Log LLM call
        telemetry.log_llm_call(
            provider="openai",
            model="gpt-4-0613",
            prompt="Calculate emissions...",
            response="Based on the tool...",
            usage=Usage(...),
            tool_calls=["calculate_fuel_emissions"],
            latency_ms=3450
        )

        # Log tool execution
        telemetry.log_tool_execution(
            tool_name="calculate_fuel_emissions",
            arguments={"fuel": "diesel", "amount": 100},
            result={"emissions": 1021},
            success=True,
            latency_ms=120
        )

        # Log security alert
        telemetry.log_security_alert(
            alert_type="prompt_injection",
            severity="high",
            details="Detected 'ignore previous instructions'",
            blocked=True
        )
    """

    def __init__(self, emitter: Optional[TelemetryEmitter] = None):
        self.emitter = emitter or NoOpEmitter()

    def log_llm_call(
        self,
        provider: str,
        model: str,
        prompt: str,
        response: str,
        usage: Any,  # Usage object
        tool_calls: List[str],
        latency_ms: int,
        finish_reason: str = "stop",
    ) -> None:
        """Log LLM inference event"""
        event = LLMEvent(
            provider=provider,
            model=model,
            prompt_hash=LLMEvent.hash_text(prompt),
            response_hash=LLMEvent.hash_text(response),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=usage.cost_usd,
            latency_ms=latency_ms,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )
        self.emitter.emit(event)

    def log_tool_execution(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
        latency_ms: int,
        error: Optional[str] = None,
    ) -> None:
        """Log tool execution event"""
        import json

        event = ToolEvent(
            tool_name=tool_name,
            arguments_hash=LLMEvent.hash_text(json.dumps(arguments, sort_keys=True)),
            result_hash=LLMEvent.hash_text(str(result)),
            success=success,
            latency_ms=latency_ms,
            error=error,
        )
        self.emitter.emit(event)

    def log_budget_update(
        self, spent_usd: float, max_usd: float, spent_tokens: int
    ) -> None:
        """Log budget tracking event"""
        event = BudgetEvent(
            spent_usd=spent_usd,
            max_usd=max_usd,
            spent_tokens=spent_tokens,
            remaining_usd=max(0.0, max_usd - spent_usd),
        )
        self.emitter.emit(event)

    def log_security_alert(
        self, alert_type: str, severity: str, details: str, blocked: bool
    ) -> None:
        """Log security event"""
        event = SecurityEvent(
            alert_type=alert_type, severity=severity, details=details, blocked=blocked
        )
        self.emitter.emit(event)

    def create_callback(self) -> Callable[[str, Dict], None]:
        """
        Create a simple callback for legacy code

        Returns:
            Callback function: (event_type: str, metadata: dict) -> None

        Usage:
            telemetry = IntelligenceTelemetry()
            callback = telemetry.create_callback()

            # Can be passed to ChatSession
            session = ChatSession(provider, telemetry=callback)
        """

        def callback(event_type: str, metadata: Dict[str, Any]) -> None:
            try:
                event = TelemetryEvent(event_type=event_type, metadata=metadata)
                self.emitter.emit(event)
            except Exception:
                # Never fail on telemetry errors
                pass

        return callback
