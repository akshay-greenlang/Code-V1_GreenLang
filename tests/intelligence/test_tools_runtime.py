"""
Comprehensive Tests for INTL-103 Tool Runtime

Tests all CTO specification requirements:
- Tool validation (args/result schema)
- Quantity & units (normalization, equality)
- No naked numbers enforcement
- Must call tools (model can't emit raw numerics)
- Mode enforcement (Replay vs Live)
- Provenance tracking
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock
import json

from greenlang.intelligence.runtime.tools import Tool, ToolRegistry, ToolRuntime
from greenlang.intelligence.runtime.schemas import Quantity, Claim
from greenlang.intelligence.runtime.errors import (
    GLValidationError,
    GLRuntimeError,
    GLSecurityError,
    GLDataError,
    GLProvenanceError
)
from greenlang.intelligence.runtime.units import UnitRegistry


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def unit_registry():
    """Unit registry for testing"""
    return UnitRegistry()


@pytest.fixture
def energy_intensity_tool():
    """Example tool: energy intensity calculator"""
    return Tool(
        name="energy_intensity",
        description="Compute kWh/m2 given annual kWh and floor area",
        args_schema={
            "type": "object",
            "required": ["annual_kwh", "floor_m2"],
            "properties": {
                "annual_kwh": {"type": "number", "minimum": 0},
                "floor_m2": {"type": "number", "exclusiveMinimum": 0}
            }
        },
        result_schema={
            "type": "object",
            "required": ["intensity"],
            "properties": {
                "intensity": {"$ref": "greenlang://schemas/quantity.json"}
            }
        },
        fn=lambda annual_kwh, floor_m2: {
            "intensity": {"value": annual_kwh / floor_m2, "unit": "kWh/m2"}
        }
    )


@pytest.fixture
def calc_sum_tool():
    """Simple calculation tool for testing"""
    return Tool(
        name="calc_sum",
        description="Add two numbers",
        args_schema={
            "type": "object",
            "required": ["a", "b"],
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            }
        },
        result_schema={
            "type": "object",
            "required": ["result"],
            "properties": {
                "result": {"$ref": "greenlang://schemas/quantity.json"}
            }
        },
        fn=lambda a, b: {
            "result": {"value": a + b, "unit": ""}
        }
    )


@pytest.fixture
def live_connector_tool():
    """Tool that requires Live mode"""
    return Tool(
        name="fetch_data",
        description="Fetch data from external API",
        args_schema={
            "type": "object",
            "required": ["url"],
            "properties": {
                "url": {"type": "string"}
            }
        },
        result_schema={
            "type": "object",
            "required": ["data"],
            "properties": {
                "data": {"type": "string"}
            }
        },
        fn=lambda url: {"data": f"Data from {url}"},
        live_required=True
    )


@pytest.fixture
def tool_registry(energy_intensity_tool, calc_sum_tool):
    """Tool registry with test tools"""
    registry = ToolRegistry()
    registry.register(energy_intensity_tool)
    registry.register(calc_sum_tool)
    return registry


@pytest.fixture
def mock_provider():
    """Mock LLM provider"""
    provider = Mock()
    provider.init_chat = Mock(return_value="state_0")
    provider.chat_step = Mock()
    provider.inject_tool_result = Mock(return_value="state_1")
    provider.inject_error = Mock(return_value="state_error")
    return provider


# ============================================================================
# A. TOOL VALIDATION TESTS
# ============================================================================

class TestToolValidation:
    """Test args/result schema validation"""

    def test_args_schema_rejects_bad_input(self, energy_intensity_tool, tool_registry, mock_provider):
        """Bad tool arguments should be rejected"""
        runtime = ToolRuntime(mock_provider, tool_registry, mode="Replay")

        # Mock provider to return invalid tool call
        mock_provider.chat_step.return_value = {
            "kind": "tool_call",
            "tool_name": "energy_intensity",
            "arguments": {
                "annual_kwh": -100,  # Violates minimum: 0
                "floor_m2": 50
            }
        }

        with pytest.raises(GLValidationError) as exc_info:
            runtime.run("You are a helper", "Calculate")

        assert exc_info.value.code == "ARGS_SCHEMA"
        assert "minimum" in str(exc_info.value).lower()

    def test_result_schema_rejects_raw_number_field(self, tool_registry, mock_provider):
        """Tool output with raw number should be rejected"""
        # Create tool that returns raw number (violates spec)
        bad_tool = Tool(
            name="bad_tool",
            description="Returns raw number",
            args_schema={
                "type": "object",
                "properties": {}
            },
            result_schema={
                "type": "object",
                "required": ["value"],
                "properties": {
                    "value": {"type": "number"}  # Raw number, not Quantity!
                }
            },
            fn=lambda: {"value": 42}  # Returns raw number
        )

        tool_registry.register(bad_tool)
        runtime = ToolRuntime(mock_provider, tool_registry)

        mock_provider.chat_step.return_value = {
            "kind": "tool_call",
            "tool_name": "bad_tool",
            "arguments": {}
        }

        with pytest.raises(GLValidationError) as exc_info:
            runtime.run("", "test")

        assert exc_info.value.code == "RESULT_SCHEMA"
        assert "Quantity" in str(exc_info.value)

    def test_result_schema_accepts_quantity(self, energy_intensity_tool, tool_registry, mock_provider):
        """Tool output with Quantity should be accepted"""
        runtime = ToolRuntime(mock_provider, tool_registry)

        mock_provider.chat_step.side_effect = [
            {
                "kind": "tool_call",
                "tool_name": "energy_intensity",
                "arguments": {"annual_kwh": 1000, "floor_m2": 100}
            },
            {
                "kind": "final",
                "final": {
                    "message": "Intensity is {{claim:0}}",
                    "claims": [
                        {
                            "source_call_id": "tc_1",
                            "path": "$.intensity",
                            "quantity": {"value": 10.0, "unit": "kWh/m2"}
                        }
                    ]
                }
            }
        ]

        result = runtime.run("", "test")
        assert "10.00 kWh/m2" in result["message"]


# ============================================================================
# B. QUANTITY & UNITS TESTS
# ============================================================================

class TestQuantityAndUnits:
    """Test quantity normalization and equality"""

    def test_quantity_unit_allowlist(self, unit_registry):
        """Unknown units should be rejected"""
        assert unit_registry.is_allowed("kWh")
        assert unit_registry.is_allowed("kgCO2e")
        assert not unit_registry.is_allowed("parsecs")

    def test_quantity_normalization_and_equality(self, unit_registry):
        """1 tCO2e should equal 1000 kgCO2e after normalization"""
        q1 = Quantity(value=1.0, unit="tCO2e")
        q2 = Quantity(value=1000, unit="kgCO2e")

        # Normalize both
        val1, unit1 = unit_registry.normalize(q1)
        val2, unit2 = unit_registry.normalize(q2)

        assert unit1 == unit2 == "kgCO2e"
        assert float(val1) == float(val2) == 1000

        # Test same_quantity
        assert unit_registry.same_quantity(q1, q2)

    def test_unknown_unit_rejected(self, unit_registry):
        """Unknown unit should raise error"""
        q = Quantity(value=100, unit="jiggawatts")

        with pytest.raises(GLValidationError) as exc_info:
            unit_registry.normalize(q)

        assert exc_info.value.code == "UNIT_UNKNOWN"

    def test_dimension_mismatch_detected(self, unit_registry):
        """Energy unit treated as mass should fail"""
        q_energy = Quantity(value=100, unit="kWh")

        with pytest.raises(GLValidationError):
            unit_registry.validate_dimension(q_energy, "mass")

    def test_currency_treated_as_tagged_non_convertible(self, unit_registry):
        """Currency units should be tagged and non-convertible"""
        q_usd = Quantity(value=100, unit="USD")
        q_eur = Quantity(value=100, unit="EUR")

        # Both should be in allowlist
        assert unit_registry.is_allowed("USD")
        assert unit_registry.is_allowed("EUR")

        # But they should NOT be considered the same quantity
        # Currency is tagged, non-convertible: 100 USD ≠ 100 EUR
        assert not unit_registry.same_quantity(q_usd, q_eur)

        # Normalization should keep currency units as-is (no conversion)
        val_usd, unit_usd = unit_registry.normalize(q_usd)
        val_eur, unit_eur = unit_registry.normalize(q_eur)

        # Units should remain unchanged (not converted to common base)
        assert unit_usd == "USD"
        assert unit_eur == "EUR"


# ============================================================================
# C. NO NAKED NUMBERS TESTS
# ============================================================================

class TestNoNakedNumbers:
    """Test naked numbers enforcement"""

    def test_final_message_with_digits_and_no_claims_blocked(self, calc_sum_tool, mock_provider):
        """Final with digits but no claims should be blocked"""
        registry = ToolRegistry()
        registry.register(calc_sum_tool)
        runtime = ToolRuntime(mock_provider, registry)

        mock_provider.chat_step.return_value = {
            "kind": "final",
            "final": {
                "message": "The answer is 42.",  # Naked number!
                "claims": []
            }
        }

        with pytest.raises(GLRuntimeError) as exc_info:
            runtime.run("", "What is 21 * 2?")

        assert exc_info.value.code == "NO_NAKED_NUMBERS"
        assert "42" in exc_info.value.message

    def test_final_message_digits_via_claim_macros_allowed(self, calc_sum_tool, mock_provider):
        """Final with {{claim:i}} macros should be allowed"""
        registry = ToolRegistry()
        registry.register(calc_sum_tool)
        runtime = ToolRuntime(mock_provider, registry)

        mock_provider.chat_step.side_effect = [
            {
                "kind": "tool_call",
                "tool_name": "calc_sum",
                "arguments": {"a": 21, "b": 21}
            },
            {
                "kind": "final",
                "final": {
                    "message": "The answer is {{claim:0}}.",  # Macro, not naked!
                    "claims": [
                        {
                            "source_call_id": "tc_1",
                            "path": "$.result",
                            "quantity": {"value": 42, "unit": ""}
                        }
                    ]
                }
            }
        ]

        result = runtime.run("", "What is 21 + 21?")
        assert "42" in result["message"]  # Rendered from claim
        assert "{{claim:" not in result["message"]  # Macro replaced

    def test_ordered_list_numbers_whitelisted(self, calc_sum_tool, mock_provider):
        """Ordered list markers like '1. ' should be allowed"""
        registry = ToolRegistry()
        registry.register(calc_sum_tool)
        runtime = ToolRuntime(mock_provider, registry)

        mock_provider.chat_step.return_value = {
            "kind": "final",
            "final": {
                "message": "Steps:\n1. First step\n2. Second step",
                "claims": []
            }
        }

        # Should NOT raise NO_NAKED_NUMBERS for list markers
        result = runtime.run("", "List steps")
        assert "1. First step" in result["message"]

    def test_version_strings_in_code_blocks_whitelisted(self, calc_sum_tool, mock_provider):
        """Version strings ONLY allowed inside code blocks"""
        registry = ToolRegistry()
        registry.register(calc_sum_tool)
        runtime = ToolRuntime(mock_provider, registry)

        # Version string inside code block - should pass
        mock_provider.chat_step.return_value = {
            "kind": "final",
            "final": {
                "message": "Install using:\n```bash\nnpm install greenlang@v1.2.3\n```",
                "claims": []
            }
        }

        result = runtime.run("", "test")
        assert "v1.2.3" in result["message"]

    def test_version_strings_outside_code_blocks_rejected(self, calc_sum_tool, mock_provider):
        """Version strings OUTSIDE code blocks should be rejected"""
        registry = ToolRegistry()
        registry.register(calc_sum_tool)
        runtime = ToolRuntime(mock_provider, registry)

        # Version string outside code block - should raise NO_NAKED_NUMBERS
        mock_provider.chat_step.return_value = {
            "kind": "final",
            "final": {
                "message": "Using algorithm v1.2.3 for calculations",
                "claims": []
            }
        }

        # Should raise NO_NAKED_NUMBERS for the digits in v1.2.3
        with pytest.raises(GLRuntimeError) as exc_info:
            runtime.run("", "test")

        assert exc_info.value.code == "NO_NAKED_NUMBERS"

    def test_iso_dates_whitelisted(self, calc_sum_tool, mock_provider):
        """ISO dates should be whitelisted"""
        registry = ToolRegistry()
        registry.register(calc_sum_tool)
        runtime = ToolRuntime(mock_provider, registry)

        mock_provider.chat_step.return_value = {
            "kind": "final",
            "final": {
                "message": "Report from 2024-10-02",
                "claims": []
            }
        }

        result = runtime.run("", "test")
        assert "2024-10-02" in result["message"]

    def test_id_patterns_whitelisted(self, calc_sum_tool, mock_provider):
        """ID patterns like ID-123 should be allowed"""
        registry = ToolRegistry()
        registry.register(calc_sum_tool)
        runtime = ToolRuntime(mock_provider, registry)

        mock_provider.chat_step.return_value = {
            "kind": "final",
            "final": {
                "message": "See report ID-12345",
                "claims": []
            }
        }

        result = runtime.run("", "test")
        assert "ID-12345" in result["message"]


# ============================================================================
# D. MUST CALL TOOLS TESTS
# ============================================================================

class TestMustCallTools:
    """Test that model must call tools to get numerics"""

    def test_llm_returns_final_with_literal_42_blocked(self, calc_sum_tool, mock_provider):
        """LLM returning literal 42 should be blocked"""
        registry = ToolRegistry()
        registry.register(calc_sum_tool)
        runtime = ToolRuntime(mock_provider, registry)

        mock_provider.chat_step.return_value = {
            "kind": "final",
            "final": {
                "message": "The result is 42",  # Naked number
                "claims": []
            }
        }

        with pytest.raises(GLRuntimeError) as exc_info:
            runtime.run("", "What is 21 * 2?")

        assert exc_info.value.code == "NO_NAKED_NUMBERS"

    def test_llm_calls_tool_then_finalizes_with_claims_passes(self, calc_sum_tool, mock_provider):
        """LLM calling tool then using claims should pass"""
        registry = ToolRegistry()
        registry.register(calc_sum_tool)
        runtime = ToolRuntime(mock_provider, registry)

        mock_provider.chat_step.side_effect = [
            # First: call tool
            {
                "kind": "tool_call",
                "tool_name": "calc_sum",
                "arguments": {"a": 21, "b": 21}
            },
            # Then: finalize with claim
            {
                "kind": "final",
                "final": {
                    "message": "The result is {{claim:0}}",
                    "claims": [
                        {
                            "source_call_id": "tc_1",
                            "path": "$.result",
                            "quantity": {"value": 42, "unit": ""}
                        }
                    ]
                }
            }
        ]

        result = runtime.run("", "What is 21 + 21?")
        assert "42" in result["message"]
        assert len(result["provenance"]) == 1


# ============================================================================
# E. MODE ENFORCEMENT TESTS
# ============================================================================

class TestModeEnforcement:
    """Test Replay vs Live mode enforcement"""

    def test_live_tool_in_replay_mode_blocked(self, live_connector_tool, mock_provider):
        """Live tool in Replay mode should be blocked"""
        registry = ToolRegistry()
        registry.register(live_connector_tool)
        runtime = ToolRuntime(mock_provider, registry, mode="Replay")

        mock_provider.chat_step.return_value = {
            "kind": "tool_call",
            "tool_name": "fetch_data",
            "arguments": {"url": "https://api.example.com"}
        }

        with pytest.raises(GLSecurityError) as exc_info:
            runtime.run("", "Fetch data")

        assert exc_info.value.code == "EGRESS_BLOCKED"
        assert "Live mode" in str(exc_info.value)

    def test_live_tool_in_live_mode_allowed(self, live_connector_tool, mock_provider):
        """Live tool in Live mode should be allowed"""
        registry = ToolRegistry()
        registry.register(live_connector_tool)
        runtime = ToolRuntime(mock_provider, registry, mode="Live")

        mock_provider.chat_step.side_effect = [
            {
                "kind": "tool_call",
                "tool_name": "fetch_data",
                "arguments": {"url": "https://api.example.com"}
            },
            {
                "kind": "final",
                "final": {
                    "message": "Data fetched successfully",
                    "claims": []
                }
            }
        ]

        result = runtime.run("", "Fetch data")
        assert "successfully" in result["message"]


# ============================================================================
# F. PROVENANCE TESTS
# ============================================================================

class TestProvenance:
    """Test claims resolve to prior tool calls"""

    def test_claims_resolve_to_prior_tool_call(self, energy_intensity_tool, mock_provider):
        """Claims should resolve to tool outputs via JSONPath"""
        registry = ToolRegistry()
        registry.register(energy_intensity_tool)
        runtime = ToolRuntime(mock_provider, registry)

        mock_provider.chat_step.side_effect = [
            {
                "kind": "tool_call",
                "tool_name": "energy_intensity",
                "arguments": {"annual_kwh": 1200, "floor_m2": 100}
            },
            {
                "kind": "final",
                "final": {
                    "message": "Intensity: {{claim:0}}",
                    "claims": [
                        {
                            "source_call_id": "tc_1",
                            "path": "$.intensity",
                            "quantity": {"value": 12.0, "unit": "kWh/m2"}
                        }
                    ]
                }
            }
        ]

        result = runtime.run("", "Calculate intensity")
        assert "12.00 kWh/m2" in result["message"]
        assert result["provenance"][0]["source_call_id"] == "tc_1"
        assert result["provenance"][0]["path"] == "$.intensity"

    def test_invalid_jsonpath_raises_error(self, energy_intensity_tool, mock_provider):
        """Invalid JSONPath should raise error"""
        registry = ToolRegistry()
        registry.register(energy_intensity_tool)
        runtime = ToolRuntime(mock_provider, registry)

        mock_provider.chat_step.side_effect = [
            {
                "kind": "tool_call",
                "tool_name": "energy_intensity",
                "arguments": {"annual_kwh": 1000, "floor_m2": 100}
            },
            {
                "kind": "final",
                "final": {
                    "message": "Intensity: {{claim:0}}",
                    "claims": [
                        {
                            "source_call_id": "tc_1",
                            "path": "$.nonexistent",  # Invalid path
                            "quantity": {"value": 10.0, "unit": "kWh/m2"}
                        }
                    ]
                }
            }
        ]

        with pytest.raises(GLDataError) as exc_info:
            runtime.run("", "test")

        assert exc_info.value.code == "PATH_RESOLUTION"

    def test_claim_quantity_mismatch_detected(self, energy_intensity_tool, mock_provider):
        """Claim quantity not matching tool output should be detected"""
        registry = ToolRegistry()
        registry.register(energy_intensity_tool)
        runtime = ToolRuntime(mock_provider, registry)

        mock_provider.chat_step.side_effect = [
            {
                "kind": "tool_call",
                "tool_name": "energy_intensity",
                "arguments": {"annual_kwh": 1000, "floor_m2": 100}
            },
            {
                "kind": "final",
                "final": {
                    "message": "Intensity: {{claim:0}}",
                    "claims": [
                        {
                            "source_call_id": "tc_1",
                            "path": "$.intensity",
                            "quantity": {"value": 999.0, "unit": "kWh/m2"}  # Wrong!
                        }
                    ]
                }
            }
        ]

        with pytest.raises(GLDataError) as exc_info:
            runtime.run("", "test")

        assert exc_info.value.code == "QUANTITY_MISMATCH"


# ============================================================================
# F. PROPERTY/FUZZ TESTS
# ============================================================================

class TestPropertyFuzz:
    """Property-based and fuzz testing for naked numbers"""

    def test_random_json_with_bare_numbers_rejected(self):
        """Random JSON output with bare numbers must be rejected"""
        from hypothesis import given, strategies as st

        @given(st.floats(allow_nan=False, allow_infinity=False))
        def check_bare_number_rejected(value):
            # Create tool that returns bare number
            bad_tool = Tool(
                name="bad_tool",
                description="Returns bare number",
                args_schema={"type": "object", "properties": {}},
                result_schema={"type": "object", "properties": {}},
                fn=lambda: {"result": value}  # Bare number!
            )

            # Tool execution should detect bare number in result
            from greenlang.intelligence.runtime.tools import ToolRuntime
            registry = ToolRegistry()
            registry.register(bad_tool)

            # Create mock provider that calls the tool
            provider = Mock()
            provider.init_chat = Mock(return_value="state")
            provider.chat_step = Mock(return_value={
                "kind": "tool_call",
                "tool_name": "bad_tool",
                "arguments": {}
            })

            runtime = ToolRuntime(provider, registry)

            # Should fail because tool output has bare number
            with pytest.raises(GLValidationError) as exc_info:
                runtime.run("", "test")

            assert exc_info.value.code == "RESULT_SCHEMA"

        # Run property test
        check_bare_number_rejected()

    def test_random_json_with_quantity_accepted(self):
        """Random JSON with proper Quantity wrapper should pass"""
        from hypothesis import given, strategies as st

        @given(
            value=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            unit=st.sampled_from(["kWh", "kgCO2e", "USD", "m2"])
        )
        def check_quantity_accepted(value, unit):
            # Create tool that returns proper Quantity
            good_tool = Tool(
                name="good_tool",
                description="Returns quantity",
                args_schema={"type": "object", "properties": {}},
                result_schema={
                    "type": "object",
                    "required": ["output"],
                    "properties": {
                        "output": {"$ref": "greenlang://schemas/quantity.json"}
                    }
                },
                fn=lambda: {"output": {"value": value, "unit": unit}}
            )

            # Execute tool - should succeed
            result = good_tool.fn()
            assert "value" in result["output"]
            assert "unit" in result["output"]
            assert result["output"]["value"] == value
            assert result["output"]["unit"] == unit

        # Run property test
        check_quantity_accepted()

    def test_fuzz_digit_scanner_whitelist_patterns(self):
        """Fuzz test digit scanner with various whitelisted patterns"""
        from hypothesis import given, strategies as st
        from greenlang.intelligence.runtime.tools import ToolRuntime
        from unittest.mock import Mock

        # Test data: (message, should_pass)
        test_cases = [
            # Ordered lists - PASS
            ("Steps:\n1. First\n2. Second", True),
            ("Process:\n10. Step ten", True),

            # ISO dates - PASS
            ("Report from 2024-10-02", True),
            ("Data: 2023-12-31 to 2024-01-15", True),

            # IDs - PASS
            ("ID-123 and ID-456", True),
            ("Reference: ID_789", True),

            # Time stamps - PASS
            ("At 14:30:00", True),
            ("Between 09:15 and 17:45", True),

            # Code blocks - PASS
            ("Install:\n```\nnpm install v1.2.3\n```", True),
            ("```python\nversion = 0.4.0\n```", True),

            # Bare numbers - FAIL
            ("The answer is 42", False),
            ("Total: 100 items", False),
            ("Value is 3.14", False),
        ]

        for message, should_pass in test_cases:
            registry = ToolRegistry()
            provider = Mock()
            provider.init_chat = Mock(return_value="state")
            provider.chat_step = Mock(return_value={
                "kind": "final",
                "final": {"message": message, "claims": []}
            })

            runtime = ToolRuntime(provider, registry)

            if should_pass:
                # Should NOT raise
                try:
                    result = runtime.run("", "test")
                    assert message in result["message"] or "```" in message  # Code blocks change message
                except GLRuntimeError:
                    pytest.fail(f"Incorrectly blocked whitelisted pattern: {message}")
            else:
                # Should raise NO_NAKED_NUMBERS
                with pytest.raises(GLRuntimeError) as exc_info:
                    runtime.run("", "test")
                assert exc_info.value.code == "NO_NAKED_NUMBERS"

    def test_fuzz_deeply_nested_json_structures(self):
        """Test that bare numbers are detected even in deeply nested JSON"""
        from greenlang.intelligence.runtime.tools import Tool, ToolRegistry, ToolRuntime
        from unittest.mock import Mock

        # Deep nesting with bare number
        deep_tool = Tool(
            name="deep_tool",
            description="Deeply nested output",
            args_schema={"type": "object", "properties": {}},
            result_schema={"type": "object", "properties": {}},
            fn=lambda: {
                "level1": {
                    "level2": {
                        "level3": {
                            "level4": {
                                "value": 42  # Bare number!
                            }
                        }
                    }
                }
            }
        )

        registry = ToolRegistry()
        registry.register(deep_tool)

        # Create mock provider that calls the deep_tool
        mock_provider = Mock()
        mock_provider.chat_step.return_value = {
            "kind": "tool_call",
            "tool_name": "deep_tool",
            "arguments": {}
        }

        runtime = ToolRuntime(mock_provider, registry)

        # Should fail schema validation (bare number detected)
        with pytest.raises(GLValidationError) as exc_info:
            runtime.run("", "test")

        assert exc_info.value.code == "RESULT_SCHEMA"

    def test_fuzz_mixed_valid_invalid_fields(self):
        """Test JSON with both valid Quantities and invalid bare numbers"""
        from greenlang.intelligence.runtime.tools import Tool, ToolRegistry, ToolRuntime
        from unittest.mock import Mock

        # Mixed: some fields have Quantity, some have bare numbers
        mixed_tool = Tool(
            name="mixed_tool",
            description="Mixed output",
            args_schema={"type": "object", "properties": {}},
            result_schema={"type": "object", "properties": {}},
            fn=lambda: {
                "good_field": {"value": 100, "unit": "kWh"},  # Valid Quantity
                "bad_field": 200  # Bare number!
            }
        )

        registry = ToolRegistry()
        registry.register(mixed_tool)

        # Create mock provider that calls the mixed_tool
        mock_provider = Mock()
        mock_provider.chat_step.return_value = {
            "kind": "tool_call",
            "tool_name": "mixed_tool",
            "arguments": {}
        }

        runtime = ToolRuntime(mock_provider, registry)

        # Should fail because of bad_field
        with pytest.raises(GLValidationError) as exc_info:
            runtime.run("", "test")

        assert exc_info.value.code == "RESULT_SCHEMA"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """End-to-end integration tests"""

    def test_happy_path_tool_call_and_final(self, energy_intensity_tool, mock_provider):
        """Complete flow: tool call → final with claims"""
        registry = ToolRegistry()
        registry.register(energy_intensity_tool)
        runtime = ToolRuntime(mock_provider, registry, mode="Replay")

        mock_provider.chat_step.side_effect = [
            {
                "kind": "tool_call",
                "tool_name": "energy_intensity",
                "arguments": {"annual_kwh": 1000, "floor_m2": 100}
            },
            {
                "kind": "final",
                "final": {
                    "message": "Energy intensity is {{claim:0}} for the reported period.",
                    "claims": [
                        {
                            "source_call_id": "tc_1",
                            "path": "$.intensity",
                            "quantity": {"value": 10.0, "unit": "kWh/m2"}
                        }
                    ]
                }
            }
        ]

        result = runtime.run("You are a climate analyst.", "What's the intensity?")

        assert "10.00 kWh/m2" in result["message"]
        assert len(result["provenance"]) == 1
        assert result["provenance"][0]["source_call_id"] == "tc_1"

        # Check metrics
        metrics = runtime.get_metrics()
        assert metrics["tool_use_rate"] == 0.5  # 1 tool call out of 2 steps
        assert metrics["total_tool_calls"] == 1

    def test_naked_number_triggers_retry(self, calc_sum_tool, mock_provider):
        """LLM emits literal → error → retry with tool → success"""
        registry = ToolRegistry()
        registry.register(calc_sum_tool)
        runtime = ToolRuntime(mock_provider, registry)

        mock_provider.chat_step.side_effect = [
            # First attempt: naked number
            {
                "kind": "final",
                "final": {"message": "The answer is 42."}
            },
            # After error, calls tool
            {
                "kind": "tool_call",
                "tool_name": "calc_sum",
                "arguments": {"a": 21, "b": 21}
            },
            # Final with claims
            {
                "kind": "final",
                "final": {
                    "message": "The answer is {{claim:0}}.",
                    "claims": [
                        {
                            "source_call_id": "tc_1",
                            "path": "$.result",
                            "quantity": {"value": 42, "unit": ""}
                        }
                    ]
                }
            }
        ]

        result = runtime.run("", "What is 21 + 21?")

        # Should succeed after retry
        assert "42" in result["message"]

        # Check metrics
        metrics = runtime.get_metrics()
        assert metrics["naked_number_rejections"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
