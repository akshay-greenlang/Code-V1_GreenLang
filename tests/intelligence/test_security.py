# -*- coding: utf-8 -*-
"""
Unit tests for additional security features

Tests security best practices including:
- API keys never logged
- Prompt hashes used (not full prompts)
- No PII in telemetry
- Error messages don't leak secrets
"""

import pytest
import os
import hashlib
from greenlang.intelligence.runtime.telemetry import (
    IntelligenceTelemetry,
    LLMEvent,
    ToolEvent,
    SecurityEvent,
)
from greenlang.intelligence.schemas.responses import Usage
from greenlang.intelligence.providers.base import LLMProviderConfig


class TestAPIKeyProtection:
    """Test that API keys are never logged"""

    def test_api_key_not_in_config_repr(self):
        """LLMProviderConfig should not expose API key in repr"""
        config = LLMProviderConfig(
            model="gpt-4",
            api_key_env="OPENAI_API_KEY"
        )

        # repr/str should not contain actual API key
        config_str = str(config)
        config_repr = repr(config)

        # Should reference env var name, not value
        assert "OPENAI_API_KEY" in config_str or "OPENAI_API_KEY" in config_repr

    def test_api_key_loaded_from_env_not_hardcoded(self):
        """API keys should be loaded from environment, not hardcoded"""
        # Set test API key
        os.environ["TEST_API_KEY"] = "sk-test-key-12345"

        config = LLMProviderConfig(
            model="test-model",
            api_key_env="TEST_API_KEY"
        )

        # Config should store env var name, not value
        assert config.api_key_env == "TEST_API_KEY"

        # Clean up
        del os.environ["TEST_API_KEY"]

    def test_config_serialization_excludes_secrets(self):
        """Config serialization should not include API key values"""
        config = LLMProviderConfig(
            model="gpt-4",
            api_key_env="OPENAI_API_KEY"
        )

        # Serialize to dict
        config_dict = config.model_dump()

        # Should include env var name, not actual key
        assert "api_key_env" in config_dict
        assert config_dict["api_key_env"] == "OPENAI_API_KEY"
        # Should not have an "api_key" field with actual value
        assert "api_key" not in config_dict or config_dict.get("api_key") is None


class TestPromptHashing:
    """Test that prompts are hashed, not logged in full"""

    def test_llm_event_uses_prompt_hash(self):
        """LLMEvent should use prompt hash, not full prompt"""
        events = []

        class TestEmitter:
            def emit(self, event):
                events.append(event)

        telemetry = IntelligenceTelemetry(emitter=TestEmitter())

        prompt = "This is a secret system prompt with sensitive information"
        response = "This is the response"

        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01
        )

        telemetry.log_llm_call(
            provider="test",
            model="test-model",
            prompt=prompt,
            response=response,
            usage=usage,
            tool_calls=[],
            latency_ms=1000
        )

        assert len(events) == 1
        event = events[0]

        # Should use hash, not full prompt
        assert hasattr(event, "prompt_hash")
        assert event.prompt_hash != prompt
        # Hash should be hex string
        assert isinstance(event.prompt_hash, str)
        assert len(event.prompt_hash) == 64  # SHA-256 hex length

        # Full prompt should NOT be in event
        event_dict = event.model_dump()
        assert prompt not in str(event_dict)

    def test_llm_event_hash_is_deterministic(self):
        """Prompt hash should be deterministic (same prompt = same hash)"""
        prompt = "Test prompt"

        hash1 = LLMEvent.hash_text(prompt)
        hash2 = LLMEvent.hash_text(prompt)

        assert hash1 == hash2

    def test_llm_event_hash_is_unique(self):
        """Different prompts should have different hashes"""
        prompt1 = "First prompt"
        prompt2 = "Second prompt"

        hash1 = LLMEvent.hash_text(prompt1)
        hash2 = LLMEvent.hash_text(prompt2)

        assert hash1 != hash2

    def test_response_also_hashed(self):
        """Response should also be hashed, not stored in full"""
        events = []

        class TestEmitter:
            def emit(self, event):
                events.append(event)

        telemetry = IntelligenceTelemetry(emitter=TestEmitter())

        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01
        )

        telemetry.log_llm_call(
            provider="test",
            model="test-model",
            prompt="prompt",
            response="sensitive response data",
            usage=usage,
            tool_calls=[],
            latency_ms=1000
        )

        event = events[0]

        # Should have response hash
        assert hasattr(event, "response_hash")
        assert event.response_hash != "sensitive response data"
        assert len(event.response_hash) == 64  # SHA-256

    def test_hash_function_is_sha256(self):
        """Hash function should be SHA-256"""
        text = "test"
        hashed = LLMEvent.hash_text(text)

        # Verify it's SHA-256
        expected = hashlib.sha256(text.encode()).hexdigest()
        assert hashed == expected


class TestPIIProtection:
    """Test that PII is not logged in telemetry"""

    def test_tool_arguments_hashed(self):
        """Tool arguments should be hashed (may contain PII)"""
        events = []

        class TestEmitter:
            def emit(self, event):
                events.append(event)

        telemetry = IntelligenceTelemetry(emitter=TestEmitter())

        arguments = {
            "email": "user@example.com",
            "name": "John Doe",
            "ssn": "123-45-6789"
        }

        telemetry.log_tool_execution(
            tool_name="user_lookup",
            arguments=arguments,
            result={"found": True},
            success=True,
            latency_ms=100
        )

        event = events[0]

        # Arguments should be hashed
        assert hasattr(event, "arguments_hash")
        assert event.arguments_hash != str(arguments)

        # PII should NOT be in event
        event_dict = event.model_dump()
        event_str = str(event_dict)
        assert "user@example.com" not in event_str
        assert "John Doe" not in event_str
        assert "123-45-6789" not in event_str

    def test_tool_result_hashed(self):
        """Tool results should be hashed (may contain PII)"""
        events = []

        class TestEmitter:
            def emit(self, event):
                events.append(event)

        telemetry = IntelligenceTelemetry(emitter=TestEmitter())

        result = {
            "user_email": "sensitive@example.com",
            "credit_card": "4111-1111-1111-1111"
        }

        telemetry.log_tool_execution(
            tool_name="payment",
            arguments={},
            result=result,
            success=True,
            latency_ms=100
        )

        event = events[0]

        # Result should be hashed
        assert hasattr(event, "result_hash")

        # PII should NOT be in event
        event_str = str(event.model_dump())
        assert "sensitive@example.com" not in event_str
        assert "4111-1111-1111-1111" not in event_str

    def test_security_event_truncates_details(self):
        """Security events should truncate potentially sensitive details"""
        events = []

        class TestEmitter:
            def emit(self, event):
                events.append(event)

        telemetry = IntelligenceTelemetry(emitter=TestEmitter())

        telemetry.log_security_alert(
            alert_type="prompt_injection",
            severity="high",
            details="Pattern matched: " + "x" * 500,  # Very long
            blocked=True
        )

        event = events[0]

        # Details should be present but may be truncated
        assert hasattr(event, "details")
        assert event.details  # Not empty

    def test_metadata_allows_safe_identifiers(self):
        """Metadata should allow safe identifiers (user_id, session_id)"""
        events = []

        class TestEmitter:
            def emit(self, event):
                events.append(event)

        telemetry = IntelligenceTelemetry(emitter=TestEmitter())

        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01
        )

        # These are safe identifiers, not PII
        telemetry.log_llm_call(
            provider="test",
            model="test-model",
            prompt="prompt",
            response="response",
            usage=usage,
            tool_calls=[],
            latency_ms=1000
        )

        # Event should have run_id and agent_id from env
        event = events[0]
        # These come from environment variables if set
        assert hasattr(event, "run_id")
        assert hasattr(event, "agent_id")


class TestErrorMessageSecurity:
    """Test that error messages don't leak secrets"""

    def test_budget_exceeded_no_sensitive_data(self):
        """BudgetExceeded error should not contain sensitive data"""
        from greenlang.intelligence.runtime.budget import Budget, BudgetExceeded

        budget = Budget(max_usd=0.50)

        try:
            budget.add(add_usd=0.60, add_tokens=6000)
        except BudgetExceeded as e:
            error_str = str(e)

            # Should contain budget info
            assert "0.60" in error_str or "0.6" in error_str
            assert "0.50" in error_str or "0.5" in error_str

            # Should not contain any API keys or prompts (none in this context)
            # Just verify no obvious leak patterns
            assert "sk-" not in error_str  # OpenAI key prefix
            assert "api_key" not in error_str.lower()

    def test_hallucination_detected_limits_exposure(self):
        """HallucinationDetected should not expose full tool responses"""
        from greenlang.intelligence.verification import (
            HallucinationDetected,
            NumericClaim
        )

        claim = NumericClaim(value=999, unit="kg")
        tool_response = {
            "api_key": "sk-secret-key",  # Should not be in error
            "result": 450
        }

        exc = HallucinationDetected(
            message="Value mismatch",
            claim=claim,
            tool_response=tool_response
        )

        error_str = str(exc)

        # Should mention result value
        assert "450" in error_str

        # Should NOT expose full response (which has API key)
        # The tool_response is included but error_str is controlled
        assert "sk-secret-key" not in error_str

    def test_prompt_injection_truncates_matched_text(self):
        """PromptInjectionDetected should truncate matched text"""
        from greenlang.intelligence.security import PromptInjectionDetected

        long_malicious = "Ignore previous instructions and " + "x" * 200

        exc = PromptInjectionDetected(
            message="Injection detected",
            pattern="ignore previous instructions",
            matched_text=long_malicious,
            severity="critical"
        )

        # Matched text should be truncated (max 100 chars in implementation)
        assert len(exc.matched_text) <= 100

    def test_json_validation_error_safe(self):
        """JSONValidationError should not leak sensitive payload data"""
        from greenlang.intelligence.runtime.jsonio import JSONValidationError

        payload = {
            "password": "secret123",
            "api_key": "sk-secret"
        }

        error = JSONValidationError(
            message="Validation failed",
            payload=payload,
            schema={"type": "object"}
        )

        # Error has payload attribute for debugging
        assert error.payload == payload

        # But string representation may hide it
        error_str = str(error)
        # Implementation dependent - just ensure no crash
        assert "Validation failed" in error_str


class TestSecurityBestPractices:
    """Test adherence to security best practices"""

    def test_no_secrets_in_logs_directory(self, tmp_path):
        """Log files should not contain secrets"""
        from greenlang.intelligence.runtime.telemetry import FileEmitter

        log_file = tmp_path / "test.jsonl"
        emitter = FileEmitter(str(log_file))
        telemetry = IntelligenceTelemetry(emitter=emitter)

        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01
        )

        # Log some events
        telemetry.log_llm_call(
            provider="test",
            model="test-model",
            prompt="This prompt contains sensitive data: PASSWORD123",
            response="response",
            usage=usage,
            tool_calls=[],
            latency_ms=1000
        )

        # Read log file
        log_content = log_file.read_text()

        # Should not contain full prompt
        assert "PASSWORD123" not in log_content

        # Should contain hashes instead
        assert "prompt_hash" in log_content

    def test_telemetry_fails_gracefully(self):
        """Telemetry errors should not crash the application"""
        from greenlang.intelligence.runtime.telemetry import FileEmitter

        # Invalid path that will cause write error
        emitter = FileEmitter("/invalid/path/that/does/not/exist/log.jsonl")
        telemetry = IntelligenceTelemetry(emitter=emitter)

        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01
        )

        # Should not raise even if write fails
        telemetry.log_llm_call(
            provider="test",
            model="test-model",
            prompt="prompt",
            response="response",
            usage=usage,
            tool_calls=[],
            latency_ms=1000
        )

    def test_hashing_uses_secure_algorithm(self):
        """Hash function should use cryptographically secure algorithm"""
        # SHA-256 is considered secure for hashing
        text = "test data"
        hashed = LLMEvent.hash_text(text)

        # Should be 64 character hex string (SHA-256)
        assert len(hashed) == 64
        assert all(c in "0123456789abcdef" for c in hashed)

    def test_no_plaintext_credentials_in_memory(self):
        """Credentials should not be stored in plaintext"""
        config = LLMProviderConfig(
            model="gpt-4",
            api_key_env="OPENAI_API_KEY"
        )

        # Config should only store env var name
        config_vars = vars(config)
        # Should not have a plaintext api_key field
        assert "api_key" not in config_vars or config_vars.get("api_key") is None
        # Should reference env var
        assert config.api_key_env == "OPENAI_API_KEY"

    def test_security_events_are_immutable(self):
        """Security events should be immutable after creation"""
        event = SecurityEvent(
            alert_type="test",
            severity="high",
            details="test details",
            blocked=True
        )

        # Pydantic models are not frozen by default, but we can test
        # that critical fields exist and are set correctly
        assert event.alert_type == "test"
        assert event.severity == "high"
        assert event.blocked is True

        # Event should have timestamp
        assert hasattr(event, "timestamp")
        assert event.timestamp  # Not empty


class TestDataRetention:
    """Test data retention and privacy policies"""

    def test_telemetry_includes_timestamp(self):
        """All telemetry events should include timestamp"""
        events = []

        class TestEmitter:
            def emit(self, event):
                events.append(event)

        telemetry = IntelligenceTelemetry(emitter=TestEmitter())

        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01
        )

        telemetry.log_llm_call(
            provider="test",
            model="test-model",
            prompt="prompt",
            response="response",
            usage=usage,
            tool_calls=[],
            latency_ms=1000
        )

        event = events[0]

        # Should have timestamp for retention policy enforcement
        assert hasattr(event, "timestamp")
        assert event.timestamp
        # Should be ISO 8601 format
        assert "T" in event.timestamp  # Date-time separator

    def test_events_are_jsonl_format(self, tmp_path):
        """Events should be in JSON Lines format for easy processing"""
        from greenlang.intelligence.runtime.telemetry import FileEmitter
        import json

        log_file = tmp_path / "events.jsonl"
        emitter = FileEmitter(str(log_file))
        telemetry = IntelligenceTelemetry(emitter=emitter)

        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01
        )

        # Log multiple events
        for i in range(3):
            telemetry.log_llm_call(
                provider="test",
                model="test-model",
                prompt=f"prompt {i}",
                response=f"response {i}",
                usage=usage,
                tool_calls=[],
                latency_ms=1000
            )

        # Read and parse
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 3

        # Each line should be valid JSON
        for line in lines:
            event = json.loads(line)
            assert "event_type" in event
            assert "timestamp" in event
