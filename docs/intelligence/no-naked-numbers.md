# No Naked Numbers: Tool Runtime Guide

## Overview

The **"No Naked Numbers"** policy enforces that **all numeric values shown to users must come from validated tool outputs with explicit units**. This ensures data integrity, traceability, and reproducibility in climate intelligence applications.

## Core Principle

> **No number reaches the user unless it came from a validated tool output, carries a recognized unit, and is tied to explicit provenance.**

---

## Quick Start

### 1. Define a Tool with Quantity Output

All tool outputs containing numeric values must wrap them in `Quantity` objects:

```python
from greenlang.intelligence.runtime.tools import Tool

energy_tool = Tool(
    name="calculate_intensity",
    description="Calculate energy intensity in kWh/m2",
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
            "intensity": {
                "$ref": "greenlang://schemas/quantity.json"
            }
        }
    },
    fn=lambda annual_kwh, floor_m2: {
        "intensity": {
            "value": annual_kwh / floor_m2,
            "unit": "kWh/m2"
        }
    }
)
```

**Key Requirements:**
- ✅ `result_schema` references `greenlang://schemas/quantity.json`
- ✅ Function returns `{"value": <number>, "unit": <string>}`
- ❌ Function CANNOT return raw numbers like `{"intensity": 12.5}`

---

### 2. Register and Run

```python
from greenlang.intelligence.runtime.tools import ToolRegistry, ToolRuntime

# Register tool
registry = ToolRegistry()
registry.register(energy_tool)

# Create runtime
runtime = ToolRuntime(provider, registry, mode="Replay")

# Run
result = runtime.run(
    system_prompt="You are a climate advisor.",
    user_msg="What's the energy intensity?"
)

print(result["message"])      # "The intensity is 12.00 kWh/m2"
print(result["provenance"])    # [{source_call_id: "tc_1", path: "$.intensity", ...}]
```

---

## The {{claim:i}} Macro System

### How It Works

1. **LLM calls a tool** → Runtime executes and validates output
2. **LLM provides final answer** using `{{claim:i}}` macros:
   ```json
   {
     "kind": "final",
     "final": {
       "message": "The intensity is {{claim:0}} and the cost is {{claim:1}}.",
       "claims": [
         {
           "source_call_id": "tc_1",
           "path": "$.intensity",
           "quantity": {"value": 12.0, "unit": "kWh/m2"}
         },
         {
           "source_call_id": "tc_2",
           "path": "$.cost",
           "quantity": {"value": 500, "unit": "USD"}
         }
       ]
     }
   }
   ```

3. **Runtime resolves claims**:
   - Looks up tool call `tc_1`
   - Uses JSONPath `$.intensity` to extract quantity
   - Compares claimed quantity vs actual (after unit normalization)
   - If match → renders `{{claim:0}}` as `"12.00 kWh/m2"`
   - If mismatch → raises `GLDataError.QUANTITY_MISMATCH`

4. **Scan for naked numbers**:
   - After rendering, scans message for any remaining digits
   - Excludes resolved claim values
   - Whitelists specific contexts (see below)
   - Any other digit → `GLRuntimeError.NO_NAKED_NUMBERS`

---

## Allowed Contexts (Whitelist)

These patterns are allowed in final messages **without** using {{claim:i}}:

| Context | Example | Pattern |
|---------|---------|---------|
| **Ordered lists** | `1. First step\n2. Second step` | `(?:^|\n)\d+\.\s` |
| **ISO dates** | `2024-10-02` | `\b\d{4}-\d{2}-\d{2}\b` |
| **Version strings** | `v0.4.0` | `\bv\d+\.\d+(\.\d+)?\b` |
| **ID patterns** | `ID-123`, `ID_456` | `\bID[-_]?\d+\b` |
| **Time stamps** | `14:30:00` | `\b\d{2}:\d{2}(:\d{2})?\b` |

**Everything else with digits is BLOCKED.**

---

## Claims Schema Reference

### Claim Structure

```typescript
{
  source_call_id: string,     // Tool call ID (e.g., "tc_1")
  path: string,               // JSONPath to quantity (e.g., "$.intensity")
  quantity: {                 // The claimed value
    value: number,
    unit: string
  }
}
```

### JSONPath Format

- **Simple field:** `$.intensity`
- **Nested field:** `$.emissions.total`
- **Array element:** Not currently supported

**Must point to a Quantity object** in the tool output.

---

## Quantity Schema

The **ONLY** legal way to carry numeric values:

```json
{
  "value": 12.5,    // The numeric value
  "unit": "kWh/m2"  // The unit (must be in allowlist)
}
```

### Allowed Units

| Category | Units |
|----------|-------|
| **Energy** | Wh, kWh, MWh, GWh, MJ, GJ |
| **Power** | W, kW, MW |
| **Emissions** | gCO2e, kgCO2e, tCO2e |
| **Dimensionless** | %, percent |
| **Currency** | USD, EUR, GBP, INR, CNY, JPY (non-convertible) |
| **Volume** | m3, L, gal |
| **Mass** | g, kg, t, ton |
| **Length** | m, km, mi, ft |
| **Temperature** | K, C, degC, F, degF |
| **Area** | m2, km2, ft2 |
| **Intensity** | kWh/m2, kgCO2e/m2, kWh/m2/year |

**Unknown units** → `GLValidationError.UNIT_UNKNOWN`

---

## Common Errors & Fixes

### ❌ Error: NO_NAKED_NUMBERS

**Message:**
```
[NO_NAKED_NUMBERS] Naked number '42' detected at position 23
```

**Cause:** LLM tried to include a numeric value directly in the message.

**Fix:**
1. Call a tool to get the value
2. Use `{{claim:i}}` macro in final message
3. Provide matching claim in `claims[]`

**Example:**
```json
// ❌ WRONG
{
  "kind": "final",
  "final": {
    "message": "The answer is 42.",
    "claims": []
  }
}

// ✅ CORRECT
{
  "kind": "final",
  "final": {
    "message": "The answer is {{claim:0}}.",
    "claims": [
      {
        "source_call_id": "tc_1",
        "path": "$.result",
        "quantity": {"value": 42, "unit": "kWh"}
      }
    ]
  }
}
```

---

### ❌ Error: RESULT_SCHEMA

**Message:**
```
[RESULT_SCHEMA] Tool output validation failed: Additional properties are not allowed ('intensity' was unexpected)
```

**Cause:** Tool returned a raw number instead of Quantity.

**Fix:** Wrap ALL numerics in Quantity:

```python
# ❌ WRONG
fn=lambda kwh, m2: {"intensity": kwh / m2}

# ✅ CORRECT
fn=lambda kwh, m2: {
    "intensity": {
        "value": kwh / m2,
        "unit": "kWh/m2"
    }
}
```

---

### ❌ Error: QUANTITY_MISMATCH

**Message:**
```
[QUANTITY_MISMATCH] Claim 0 mismatch: tool returned value=10.0 unit='kWh/m2',
but claimed value=12.0 unit='kWh/m2'
```

**Cause:** Claim doesn't match actual tool output.

**Fix:** Ensure claimed quantity exactly matches tool output (after normalization):

```python
# Tool output
{"intensity": {"value": 10.0, "unit": "kWh/m2"}}

# ✅ CORRECT claim
{
  "source_call_id": "tc_1",
  "path": "$.intensity",
  "quantity": {"value": 10.0, "unit": "kWh/m2"}  # Must match!
}
```

---

### ❌ Error: PATH_RESOLUTION

**Message:**
```
[PATH_RESOLUTION] Path '$.wrong_field' not found in output
```

**Cause:** JSONPath doesn't exist in tool output.

**Fix:** Verify path matches tool output structure:

```python
# Tool output
{"emissions": {"total": {"value": 100, "unit": "kgCO2e"}}}

# ❌ WRONG path
"$.total"  # Missing 'emissions'

# ✅ CORRECT path
"$.emissions.total"
```

---

### ❌ Error: EGRESS_BLOCKED

**Message:**
```
[EGRESS_BLOCKED] Tool 'fetch_weather' requires Live mode but runtime is in Replay
```

**Cause:** Tried to call a network-requiring tool in Replay mode.

**Fix:**
1. Switch to Live mode: `ToolRuntime(provider, registry, mode="Live")`
2. OR use snapshot/cached data in Replay mode

---

## Replay vs Live Mode

### Replay Mode (Default)
- **Purpose:** Deterministic, reproducible runs
- **Restrictions:** Blocks tools with `live_required=True`
- **Use case:** Testing, snapshots, audits

### Live Mode
- **Purpose:** Real-time data access
- **Allows:** Network calls, API requests, database queries
- **Use case:** Production inference, real-time monitoring

```python
# Replay (deterministic)
runtime = ToolRuntime(provider, registry, mode="Replay")

# Live (network access)
runtime = ToolRuntime(provider, registry, mode="Live")
```

---

## Metrics & Observability

After each run, access metrics via:

```python
metrics = runtime.get_metrics()

print(metrics["tool_use_rate"])              # % of steps that were tool calls
print(metrics["total_tool_calls"])           # Count of tool executions
print(metrics["naked_number_rejections"])    # Count of blocked finals
print(metrics["total_steps"])                # Total LLM interactions
```

---

## Best Practices

### ✅ DO

- Define result schemas with `greenlang://schemas/quantity.json` references
- Return Quantity objects for ALL numeric tool outputs
- Use {{claim:i}} macros for all user-facing numbers
- Provide claims[] with exact matches to tool outputs
- Test tools independently before integration

### ❌ DON'T

- Return raw numbers from tools
- Hard-code numeric values in final messages
- Mix claimed units with tool output units
- Skip result_schema validation
- Use unknown/custom units without adding to allowlist

---

## Complete Working Example

```python
from greenlang.intelligence.runtime.tools import Tool, ToolRegistry, ToolRuntime
from unittest.mock import Mock

# 1. Define tool with Quantity output
carbon_tool = Tool(
    name="calculate_emissions",
    description="Calculate CO2e emissions from fuel",
    args_schema={
        "type": "object",
        "required": ["fuel_kg", "emission_factor"],
        "properties": {
            "fuel_kg": {"type": "number"},
            "emission_factor": {"type": "number"}
        }
    },
    result_schema={
        "type": "object",
        "required": ["emissions"],
        "properties": {
            "emissions": {"$ref": "greenlang://schemas/quantity.json"}
        }
    },
    fn=lambda fuel_kg, emission_factor: {
        "emissions": {
            "value": fuel_kg * emission_factor,
            "unit": "kgCO2e"
        }
    }
)

# 2. Register tool
registry = ToolRegistry()
registry.register(carbon_tool)

# 3. Create mock provider
provider = Mock()
provider.init_chat = Mock(return_value="state_0")
provider.chat_step = Mock(side_effect=[
    # Step 1: Call tool
    {
        "kind": "tool_call",
        "tool_name": "calculate_emissions",
        "arguments": {"fuel_kg": 100, "emission_factor": 2.68}
    },
    # Step 2: Final with claim
    {
        "kind": "final",
        "final": {
            "message": "Burning 100 kg of fuel produces {{claim:0}} of CO2e.",
            "claims": [
                {
                    "source_call_id": "tc_1",
                    "path": "$.emissions",
                    "quantity": {"value": 268.0, "unit": "kgCO2e"}
                }
            ]
        }
    }
])
provider.inject_tool_result = Mock(return_value="state_1")

# 4. Run runtime
runtime = ToolRuntime(provider, registry, mode="Replay")
result = runtime.run("You are a climate advisor.", "Calculate emissions")

# 5. Results
print(result["message"])
# "Burning 100 kg of fuel produces 268.00 kgCO2e of CO2e."

print(result["provenance"])
# [{"source_call_id": "tc_1", "path": "$.emissions", "quantity": {...}}]
```

---

## Testing Your Tools

```python
def test_tool_no_naked_numbers():
    """Verify tool returns Quantity, not raw number"""
    registry = ToolRegistry()
    registry.register(my_tool)

    # Execute tool
    result = registry.invoke("my_tool", {"arg": 100})

    # Verify output has Quantity
    assert "value" in result["output_field"]
    assert "unit" in result["output_field"]
    assert result["output_field"]["unit"] in ["kWh", "kgCO2e", ...]
```

---

## FAQ

**Q: Can I show version numbers like "v0.4.0" in messages?**
A: Yes, version strings are whitelisted and don't require {{claim:i}}.

**Q: What about ordered lists like "1. First step"?**
A: Allowed at line start. Use `\n1. ` format.

**Q: Can I use custom units?**
A: Add them to `UnitRegistry.allowlist` in `units.py`. Must be pint-compatible.

**Q: What if my tool needs network access?**
A: Set `live_required=True` and use `mode="Live"` in runtime.

**Q: How do I handle currency conversions?**
A: Currency is treated as "tagged, non-convertible". 100 USD ≠ 100 EUR.

---

## References

- **Code:** `greenlang/intelligence/runtime/tools.py`
- **Tests:** `tests/intelligence/test_tools_runtime.py`
- **Example:** `examples/runtime_no_naked_numbers_demo.py`
- **Schemas:** `greenlang/intelligence/runtime/schemas.py`
- **Units:** `greenlang/intelligence/runtime/units.py`

---

**Last Updated:** October 2, 2025
**Version:** INTL-103 v1.0
