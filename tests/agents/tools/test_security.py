"""
GreenLang Tool Security Test Suite
===================================

Comprehensive security testing for tool infrastructure.

Tests:
- Input validation (all validator types)
- Rate limiting (token bucket, per-tool, per-user)
- Audit logging (privacy, rotation, queries)
- Security configuration (presets, overrides)
- Integration tests (end-to-end security)

Author: GreenLang Framework Team
Date: November 2025
Status: Production Ready
"""

import pytest
import time
import tempfile
from pathlib import Path
from typing import Dict, Any

from greenlang.agents.tools.base import BaseTool, ToolDef, ToolResult, ToolSafety
from greenlang.agents.tools.validation import (
    ValidationRule,
    ValidationResult,
    RangeValidator,
    TypeValidator,
    EnumValidator,
    RegexValidator,
    CustomValidator,
    CompositeValidator,
)
from greenlang.agents.tools.rate_limiting import (
    RateLimiter,
    RateLimitExceeded,
    TokenBucket,
)
from greenlang.agents.tools.audit import (
    AuditLogger,
    AuditLogEntry,
)
from greenlang.agents.tools.security_config import (
    SecurityConfig,
    development_config,
    testing_config,
    production_config,
    high_security_config,
    configure_security,
    SecurityContext,
)


# ==============================================================================
# Test Input Validation
# ==============================================================================

class TestRangeValidator:
    """Test range validation."""

    def test_valid_range(self):
        """Test value within range."""
        validator = RangeValidator(min_value=0, max_value=100)
        result = validator.validate(50)

        assert result.valid
        assert result.sanitized_value == 50
        assert not result.errors

    def test_below_minimum(self):
        """Test value below minimum."""
        validator = RangeValidator(min_value=0, max_value=100)
        result = validator.validate(-10)

        assert not result.valid
        assert "below minimum" in result.errors[0]

    def test_above_maximum(self):
        """Test value above maximum."""
        validator = RangeValidator(min_value=0, max_value=100)
        result = validator.validate(150)

        assert not result.valid
        assert "exceeds maximum" in result.errors[0]

    def test_exclusive_bounds(self):
        """Test exclusive min/max bounds."""
        validator = RangeValidator(
            min_value=0,
            max_value=100,
            min_inclusive=False,
            max_inclusive=False
        )

        # Boundaries should fail
        assert not validator.validate(0).valid
        assert not validator.validate(100).valid

        # Inside boundaries should pass
        assert validator.validate(50).valid

    def test_non_numeric_value(self):
        """Test non-numeric value."""
        validator = RangeValidator(min_value=0, max_value=100)
        result = validator.validate("invalid")

        assert not result.valid
        assert "must be numeric" in result.errors[0]


class TestTypeValidator:
    """Test type validation."""

    def test_correct_type(self):
        """Test value with correct type."""
        validator = TypeValidator(expected_type=int)
        result = validator.validate(42)

        assert result.valid
        assert result.sanitized_value == 42

    def test_wrong_type(self):
        """Test value with wrong type."""
        validator = TypeValidator(expected_type=int)
        result = validator.validate("not an int")

        assert not result.valid
        assert "Expected type int" in result.errors[0]

    def test_type_coercion(self):
        """Test automatic type coercion."""
        validator = TypeValidator(expected_type=int, coerce=True)
        result = validator.validate("42")

        assert result.valid
        assert result.sanitized_value == 42
        assert len(result.warnings) > 0

    def test_multiple_types(self):
        """Test validation with multiple allowed types."""
        validator = TypeValidator(expected_type=(int, float))

        assert validator.validate(42).valid
        assert validator.validate(42.5).valid
        assert not validator.validate("string").valid


class TestEnumValidator:
    """Test enum validation."""

    def test_valid_value(self):
        """Test value in allowed set."""
        validator = EnumValidator(allowed_values=["red", "green", "blue"])
        result = validator.validate("red")

        assert result.valid
        assert result.sanitized_value == "red"

    def test_invalid_value(self):
        """Test value not in allowed set."""
        validator = EnumValidator(allowed_values=["red", "green", "blue"])
        result = validator.validate("yellow")

        assert not result.valid
        assert "not in allowed values" in result.errors[0]

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        validator = EnumValidator(
            allowed_values=["red", "green", "blue"],
            case_sensitive=False
        )
        result = validator.validate("RED")

        assert result.valid
        assert result.sanitized_value == "red"  # Normalized
        assert len(result.warnings) > 0


class TestRegexValidator:
    """Test regex validation."""

    def test_matching_pattern(self):
        """Test value matching pattern."""
        validator = RegexValidator(pattern=r"^[A-Z]{2}[0-9]{4}$")
        result = validator.validate("AB1234")

        assert result.valid

    def test_non_matching_pattern(self):
        """Test value not matching pattern."""
        validator = RegexValidator(pattern=r"^[A-Z]{2}[0-9]{4}$")
        result = validator.validate("invalid")

        assert not result.valid
        assert "does not match pattern" in result.errors[0]

    def test_non_string_value(self):
        """Test non-string value."""
        validator = RegexValidator(pattern=r"^\d+$")
        result = validator.validate(123)

        assert not result.valid
        assert "must be a string" in result.errors[0]


class TestCustomValidator:
    """Test custom validation."""

    def test_custom_function_bool(self):
        """Test custom function returning bool."""
        def is_even(value, context):
            return value % 2 == 0

        validator = CustomValidator(validation_fn=is_even)

        assert validator.validate(4).valid
        assert not validator.validate(5).valid

    def test_custom_function_tuple(self):
        """Test custom function returning tuple."""
        def is_positive(value, context):
            if value > 0:
                return True, "Valid"
            return False, "Value must be positive"

        validator = CustomValidator(validation_fn=is_positive)

        result = validator.validate(-5)
        assert not result.valid
        assert "must be positive" in result.errors[0]


class TestCompositeValidator:
    """Test composite validation."""

    def test_all_mode_success(self):
        """Test all validators pass (AND logic)."""
        validators = [
            TypeValidator(int),
            RangeValidator(min_value=0, max_value=100)
        ]
        validator = CompositeValidator(validators, mode="all")

        result = validator.validate(50)
        assert result.valid

    def test_all_mode_failure(self):
        """Test one validator fails (AND logic)."""
        validators = [
            TypeValidator(int),
            RangeValidator(min_value=0, max_value=100)
        ]
        validator = CompositeValidator(validators, mode="all")

        result = validator.validate(150)
        assert not result.valid

    def test_any_mode_success(self):
        """Test at least one validator passes (OR logic)."""
        validators = [
            TypeValidator(int),
            TypeValidator(str)
        ]
        validator = CompositeValidator(validators, mode="any")

        assert validator.validate(42).valid
        assert validator.validate("hello").valid

    def test_any_mode_failure(self):
        """Test all validators fail (OR logic)."""
        validators = [
            RangeValidator(min_value=0, max_value=10),
            RangeValidator(min_value=90, max_value=100)
        ]
        validator = CompositeValidator(validators, mode="any")

        result = validator.validate(50)
        assert not result.valid


# ==============================================================================
# Test Rate Limiting
# ==============================================================================

class TestTokenBucket:
    """Test token bucket implementation."""

    def test_initial_state(self):
        """Test bucket starts full."""
        bucket = TokenBucket.create(rate=10, capacity=20)

        assert bucket.tokens == 20
        assert bucket.capacity == 20
        assert bucket.rate == 10

    def test_consume_tokens(self):
        """Test consuming tokens."""
        bucket = TokenBucket.create(rate=10, capacity=20)

        assert bucket.consume(5)
        assert bucket.tokens == 15

        assert bucket.consume(15)
        assert bucket.tokens == 0

    def test_insufficient_tokens(self):
        """Test consuming more tokens than available."""
        bucket = TokenBucket.create(rate=10, capacity=20)

        bucket.consume(20)
        assert not bucket.consume(1)

    def test_token_refill(self):
        """Test automatic token refill."""
        bucket = TokenBucket.create(rate=10, capacity=20)

        # Consume all tokens
        bucket.consume(20)
        assert bucket.tokens == 0

        # Wait for refill
        time.sleep(0.5)
        bucket.refill()

        # Should have ~5 tokens (0.5s * 10 tokens/s)
        assert bucket.tokens >= 4 and bucket.tokens <= 6

    def test_wait_time_calculation(self):
        """Test wait time calculation."""
        bucket = TokenBucket.create(rate=10, capacity=20)

        bucket.consume(20)  # Empty bucket

        wait_time = bucket.get_wait_time(10)
        assert wait_time >= 0.9 and wait_time <= 1.1  # ~1 second


class TestRateLimiter:
    """Test rate limiter."""

    def test_basic_rate_limit(self):
        """Test basic rate limiting."""
        limiter = RateLimiter(rate=10, burst=10, per_tool=True)

        # Should allow up to burst size
        for i in range(10):
            assert limiter.check_limit("test_tool")
            limiter.consume("test_tool")

        # Should deny after burst exhausted
        assert not limiter.check_limit("test_tool")

    def test_per_tool_limits(self):
        """Test per-tool rate limiting."""
        limiter = RateLimiter(
            rate=10,
            burst=10,
            per_tool=True,
            per_tool_limits={
                "fast_tool": (100, 200),
                "slow_tool": (1, 2)
            }
        )

        # Fast tool should have high limit
        for i in range(200):
            assert limiter.check_limit("fast_tool")
            limiter.consume("fast_tool")

        # Slow tool should have low limit
        limiter.consume("slow_tool")
        limiter.consume("slow_tool")
        assert not limiter.check_limit("slow_tool")

    def test_rate_limit_exception(self):
        """Test rate limit exception."""
        limiter = RateLimiter(rate=1, burst=1, per_tool=True)

        limiter.consume("test_tool")

        with pytest.raises(RateLimitExceeded) as exc_info:
            limiter.check_and_consume("test_tool")

        assert exc_info.value.tool_name == "test_tool"
        assert exc_info.value.retry_after > 0

    def test_rate_limit_recovery(self):
        """Test rate limit recovery over time."""
        limiter = RateLimiter(rate=10, burst=5, per_tool=True)

        # Exhaust tokens
        for i in range(5):
            limiter.consume("test_tool")

        assert not limiter.check_limit("test_tool")

        # Wait for recovery
        time.sleep(0.6)

        # Should have recovered ~6 tokens
        assert limiter.check_limit("test_tool")

    def test_concurrent_requests(self):
        """Test thread-safe concurrent requests."""
        import threading

        limiter = RateLimiter(rate=100, burst=100, per_tool=True)
        success_count = [0]
        fail_count = [0]

        def make_request():
            if limiter.check_limit("concurrent_tool"):
                limiter.consume("concurrent_tool")
                success_count[0] += 1
            else:
                fail_count[0] += 1

        # Make concurrent requests
        threads = []
        for i in range(150):
            t = threading.Thread(target=make_request)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should allow exactly burst size
        assert success_count[0] <= 100
        assert fail_count[0] >= 50


# ==============================================================================
# Test Audit Logging
# ==============================================================================

class TestAuditLogger:
    """Test audit logging."""

    def test_log_execution(self):
        """Test logging tool execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.jsonl"
            logger = AuditLogger(log_file=log_file)

            # Log execution
            result = ToolResult(success=True, data={"result": 42})
            logger.log_execution(
                tool_name="test_tool",
                inputs={"param": "value"},
                result=result,
                execution_time_ms=10.5
            )

            # Verify log written
            assert log_file.exists()

            # Verify stats
            stats = logger.get_stats()
            assert stats["total_logged"] == 1
            assert stats["success_count"] == 1

    def test_privacy_hashing(self):
        """Test that inputs/outputs are hashed for privacy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.jsonl"
            logger = AuditLogger(log_file=log_file)

            # Log with sensitive data
            result = ToolResult(success=True, data={"api_key": "secret123"})
            logger.log_execution(
                tool_name="test_tool",
                inputs={"password": "secret456"},
                result=result,
                execution_time_ms=10.5
            )

            # Read log file and verify no raw secrets
            with open(log_file, 'r') as f:
                log_content = f.read()

            assert "secret123" not in log_content
            assert "secret456" not in log_content
            assert "input_hash" in log_content
            assert "output_hash" in log_content

    def test_query_logs(self):
        """Test querying audit logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.jsonl"
            logger = AuditLogger(log_file=log_file)

            # Log multiple executions
            for i in range(10):
                result = ToolResult(success=(i % 2 == 0), data={"count": i})
                logger.log_execution(
                    tool_name=f"tool_{i % 3}",
                    inputs={"index": i},
                    result=result,
                    execution_time_ms=10.0 + i
                )

            # Query all logs
            all_logs = logger.query_logs()
            assert len(all_logs) >= 10

            # Query by tool name
            tool_0_logs = logger.query_logs(tool_name="tool_0")
            assert len(tool_0_logs) >= 3

            # Query successes only
            success_logs = logger.query_logs(success_only=True)
            assert all(log.success for log in success_logs)

            # Query failures only
            failure_logs = logger.query_logs(failure_only=True)
            assert all(not log.success for log in failure_logs)

    def test_log_rotation(self):
        """Test log rotation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.jsonl"
            logger = AuditLogger(
                log_file=log_file,
                max_log_size_mb=0.001  # Very small for testing
            )

            # Write enough logs to trigger rotation
            for i in range(100):
                result = ToolResult(success=True, data={"count": i})
                logger.log_execution(
                    tool_name="test_tool",
                    inputs={"index": i},
                    result=result,
                    execution_time_ms=10.0
                )

            # Should have created rotated files
            rotated_files = list(Path(tmpdir).glob("audit_*.jsonl"))
            assert len(rotated_files) > 0


# ==============================================================================
# Test Security Configuration
# ==============================================================================

class TestSecurityConfig:
    """Test security configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = SecurityConfig()

        assert config.enable_validation
        assert config.enable_rate_limiting
        assert config.enable_audit_logging

    def test_development_preset(self):
        """Test development preset."""
        config = development_config()

        assert config.enable_validation
        assert not config.enable_rate_limiting  # Disabled for dev
        assert config.debug_mode

    def test_production_preset(self):
        """Test production preset."""
        config = production_config()

        assert config.enable_validation
        assert config.strict_validation
        assert config.enable_rate_limiting
        assert config.enable_audit_logging
        assert not config.debug_mode

    def test_high_security_preset(self):
        """Test high security preset."""
        config = high_security_config()

        assert config.strict_validation
        assert config.default_rate_per_second == 5  # Lower rate
        assert config.max_retries == 1

    def test_tool_whitelist(self):
        """Test tool whitelist."""
        config = SecurityConfig(tool_whitelist={"allowed_tool"})

        assert config.is_tool_allowed("allowed_tool")
        assert not config.is_tool_allowed("forbidden_tool")

    def test_tool_blacklist(self):
        """Test tool blacklist."""
        config = SecurityConfig(tool_blacklist={"forbidden_tool"})

        assert config.is_tool_allowed("allowed_tool")
        assert not config.is_tool_allowed("forbidden_tool")

    def test_security_context(self):
        """Test security context manager."""
        from greenlang.agents.tools.security_config import get_security_config

        configure_security(preset="production")

        original_debug = get_security_config().debug_mode

        # Temporarily change config
        with SecurityContext(debug_mode=True):
            assert get_security_config().debug_mode

        # Config restored
        assert get_security_config().debug_mode == original_debug


# ==============================================================================
# Test Integration
# ==============================================================================

class TestSecurityIntegration:
    """Test end-to-end security integration."""

    def test_tool_with_validation(self):
        """Test tool with input validation."""

        class ValidatedTool(BaseTool):
            def __init__(self):
                super().__init__(
                    name="validated_tool",
                    description="Tool with validation",
                    validation_rules={
                        "amount": [
                            TypeValidator(float, coerce=True),
                            RangeValidator(min_value=0, max_value=1000)
                        ]
                    }
                )

            def execute(self, amount: float) -> ToolResult:
                return ToolResult(success=True, data={"amount": amount})

            def get_tool_def(self) -> ToolDef:
                return ToolDef(
                    name=self.name,
                    description=self.description,
                    parameters={"type": "object", "properties": {}},
                    safety=self.safety
                )

        tool = ValidatedTool()

        # Valid input
        result = tool(amount=50.0)
        assert result.success

        # Invalid input (out of range)
        result = tool(amount=2000.0)
        assert not result.success
        assert "exceeds maximum" in result.error

    def test_tool_with_rate_limiting(self):
        """Test tool with rate limiting."""
        configure_security(
            preset="production",
            per_tool_limits={"limited_tool": (2, 2)}
        )

        class LimitedTool(BaseTool):
            def __init__(self):
                super().__init__(
                    name="limited_tool",
                    description="Tool with rate limiting"
                )

            def execute(self) -> ToolResult:
                return ToolResult(success=True, data={})

            def get_tool_def(self) -> ToolDef:
                return ToolDef(
                    name=self.name,
                    description=self.description,
                    parameters={"type": "object", "properties": {}},
                    safety=self.safety
                )

        tool = LimitedTool()

        # First 2 calls should succeed
        assert tool().success
        assert tool().success

        # Third call should be rate limited
        result = tool()
        assert not result.success
        assert "rate limit" in result.error.lower()

    def test_tool_with_audit_logging(self):
        """Test tool with audit logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.jsonl"

            from greenlang.agents.tools.audit import configure_audit_logger
            configure_audit_logger(log_file=log_file)

            class AuditedTool(BaseTool):
                def __init__(self):
                    super().__init__(
                        name="audited_tool",
                        description="Tool with audit logging"
                    )

                def execute(self, value: int) -> ToolResult:
                    return ToolResult(success=True, data={"value": value})

                def get_tool_def(self) -> ToolDef:
                    return ToolDef(
                        name=self.name,
                        description=self.description,
                        parameters={"type": "object", "properties": {}},
                        safety=self.safety
                    )

            tool = AuditedTool()
            tool.set_context(user_id="user123", session_id="session456")

            # Execute tool
            result = tool(value=42)
            assert result.success

            # Verify audit log
            assert log_file.exists()

            from greenlang.agents.tools.audit import get_audit_logger
            logger = get_audit_logger()
            logs = logger.query_logs(tool_name="audited_tool")

            assert len(logs) > 0
            assert logs[0].user_id == "user123"
            assert logs[0].session_id == "session456"


# ==============================================================================
# Performance Tests
# ==============================================================================

class TestSecurityPerformance:
    """Test security overhead is minimal."""

    def test_validation_overhead(self):
        """Test validation adds <1ms overhead."""
        validator = RangeValidator(min_value=0, max_value=100)

        start = time.perf_counter()
        for i in range(1000):
            validator.validate(50)
        elapsed = (time.perf_counter() - start) * 1000

        # Should take <1000ms for 1000 validations
        assert elapsed < 1000

    def test_rate_limiting_overhead(self):
        """Test rate limiting adds <1ms overhead."""
        limiter = RateLimiter(rate=1000, burst=1000)

        start = time.perf_counter()
        for i in range(1000):
            limiter.check_limit("test_tool")
        elapsed = (time.perf_counter() - start) * 1000

        # Should take <1000ms for 1000 checks
        assert elapsed < 1000
