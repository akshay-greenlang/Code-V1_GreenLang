# DOC-601: Using Tools, Not Guessing

**Document ID:** DOC-601
**Title:** Using Tools, Not Guessing - Tool-First Architecture for Climate Intelligence
**Version:** 1.0
**Date:** October 22, 2025
**Status:** Production Ready
**Owner:** GreenLang Core Team

---

## Executive Summary

This document defines the **"Using Tools, Not Guessing"** architecture pattern that ensures all AI-generated outputs are grounded in validated, traceable tool executions. This pattern eliminates hallucinated numbers, ensures unit consistency, and provides complete provenance tracking for all numeric claims.

**Core Mandate:** No number reaches the user unless it came from a validated tool output, carries a recognized unit, and is tied to explicit provenance.

---

## Table of Contents

1. [Philosophy: Tools vs Guessing](#philosophy-tools-vs-guessing)
2. [The No Naked Numbers Policy](#the-no-naked-numbers-policy)
3. [Tool-First Architecture](#tool-first-architecture)
4. [Implementation Guide](#implementation-guide)
5. [The {{claim:i}} Macro System](#the-claimi-macro-system)
6. [Quantity Schema & Units](#quantity-schema--units)
7. [Replay vs Live Mode](#replay-vs-live-mode)
8. [Common Patterns & Anti-Patterns](#common-patterns--anti-patterns)
9. [Error Handling & Debugging](#error-handling--debugging)
10. [Metrics & Observability](#metrics--observability)
11. [Testing Guidelines](#testing-guidelines)
12. [References](#references)

---

## Philosophy: Tools vs Guessing

### The Problem: AI Hallucination

Traditional LLM-based systems allow the model to generate numeric outputs directly, leading to:
- **Hallucinated numbers** with no factual basis
- **Unit inconsistencies** (mixing kWh with MWh)
- **Lack of traceability** (where did this number come from?)
- **Non-reproducibility** (different runs give different answers)

### The Solution: Tool-Grounded Outputs

GreenLang enforces a **tool-first architecture** where:
1. **All calculations happen in validated tools** (Python functions with schemas)
2. **All numeric outputs are Quantity objects** with value + unit
3. **All user-facing numbers reference tool outputs** via {{claim:i}} macros
4. **All claims are verified** before rendering to user

### Benefits

| Traditional LLM | Tool-First (GreenLang) |
|----------------|------------------------|
| "The building uses about 500 kWh" | "The building uses {{claim:0}}" → verified against `calculate_energy` tool |
| Units may be wrong/missing | All Quantities have validated units |
| No provenance | Full provenance chain: call_id → tool → output → claim |
| Non-reproducible | Deterministic with Replay mode + seed |
| Hallucination risk | Impossible to hallucinate - must come from tool |

---

## The No Naked Numbers Policy

### Definition

**"Naked Number"**: Any digit sequence in the final message that is NOT:
1. Resolved from a {{claim:i}} macro backed by a tool call
2. Whitelisted as a non-numeric context (dates, versions, list numbers)

### Enforcement

The `ToolRuntime` performs post-resolution scanning:
1. LLM provides final message with {{claim:i}} macros
2. Runtime resolves claims and replaces macros with formatted values
3. Runtime scans the rendered message for any remaining digits
4. If found (and not whitelisted) → **raises `GLRuntimeError.NO_NAKED_NUMBERS`**

### Whitelisted Contexts

These patterns are allowed without {{claim:i}}:

| Context | Example | Regex Pattern |
|---------|---------|---------------|
| Ordered lists | `1. First step\n2. Second step` | `(?:^|\n)\d+\.\s` |
| ISO dates | `2024-10-02` | `\b\d{4}-\d{2}-\d{2}\b` |
| Version strings | `v0.4.0` | `\bv\d+\.\d+(\.\d+)?\b` |
| ID patterns | `ID-123`, `ID_456` | `\bID[-_]?\d+\b` |
| Time stamps | `14:30:00` | `\b\d{2}:\d{2}(:\d{2})?\b` |

**Everything else with digits is BLOCKED.**

---

## Tool-First Architecture

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│  User Query: "What's the carbon intensity?"             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  LLM (Claude/GPT) decides to call tool                  │
│  → Step 1: {"kind": "tool_call", "tool_name": "calc"}  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  ToolRuntime validates & executes tool                  │
│  → Validates args against args_schema                   │
│  → Executes fn(args)                                    │
│  → Validates result against result_schema               │
│  → Stores in tool_call_history[call_id]                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Tool returns: {"intensity": {"value": 12, "unit": ...}}│
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  LLM provides final answer with claim                   │
│  → Step 2: {"kind": "final",                           │
│             "message": "The intensity is {{claim:0}}",  │
│             "claims": [{"source_call_id": "tc_1", ...}] │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  ToolRuntime verifies & resolves claims                 │
│  → Lookup tool_call_history["tc_1"]                     │
│  → Extract quantity via JSONPath $.intensity            │
│  → Compare claimed vs actual (after unit normalization) │
│  → If match → replace {{claim:0}} with "12.00 kWh/m2"  │
│  → If mismatch → raise QUANTITY_MISMATCH                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Scan for naked numbers in rendered message             │
│  → If found → raise NO_NAKED_NUMBERS                    │
│  → If clean → return to user with provenance            │
└─────────────────────────────────────────────────────────┘
```

### Key Components

1. **Tool** - Function with schemas defining args/results
2. **ToolRegistry** - Central repository of available tools
3. **ToolRuntime** - Orchestrates LLM + tool execution + validation
4. **Quantity** - Value + unit wrapper for all numeric data
5. **Claim** - Reference to tool output via call_id + JSONPath
6. **Provenance** - Traceable chain from user question → tool → claim

---

## Implementation Guide

### 1. Define a Tool with Quantity Output

All numeric outputs MUST be wrapped in Quantity objects:

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

### 2. Register and Create Runtime

```python
from greenlang.intelligence.runtime.tools import ToolRegistry, ToolRuntime

# Register tool
registry = ToolRegistry()
registry.register(energy_tool)

# Create runtime (Replay mode for determinism)
runtime = ToolRuntime(provider, registry, mode="Replay")
```

### 3. Run with Provenance Tracking

```python
result = runtime.run(
    system_prompt="You are a climate advisor.",
    user_msg="What's the energy intensity for 10000 kWh and 800 m2?"
)

print(result["message"])      # "The intensity is 12.50 kWh/m2"
print(result["provenance"])   # Full provenance chain
print(result["metrics"])      # Tool usage metrics
```

---

## The {{claim:i}} Macro System

### How It Works

1. **LLM calls tool** → Runtime executes and validates output
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
   - Looks up tool call `tc_1` in history
   - Uses JSONPath `$.intensity` to extract quantity
   - Compares claimed quantity vs actual (after unit normalization)
   - If match → renders `{{claim:0}}` as `"12.00 kWh/m2"`
   - If mismatch → raises `GLDataError.QUANTITY_MISMATCH`

4. **Scan for naked numbers**:
   - After rendering, scans message for any remaining digits
   - Excludes resolved claim values
   - Whitelists specific contexts (dates, versions, etc.)
   - Any other digit → `GLRuntimeError.NO_NAKED_NUMBERS`

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

## Quantity Schema & Units

### Quantity Structure

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

### Unit Normalization

GreenLang uses **pint** for unit handling:
- Conversion: `1000 kWh` = `1 MWh`
- Dimensionality checking: `kWh/m2` is valid intensity
- Non-convertible: `100 USD` ≠ `100 EUR` (treated as separate dimensions)

**Unknown units** → `GLValidationError.UNIT_UNKNOWN`

---

## Replay vs Live Mode

### Replay Mode (Default)

**Purpose:** Deterministic, reproducible runs

```python
runtime = ToolRuntime(provider, registry, mode="Replay")
```

**Characteristics:**
- Blocks tools with `live_required=True`
- Enforces determinism (temperature=0.0, seed=42)
- No network access
- Ideal for: Testing, snapshots, audits, regression tests

### Live Mode

**Purpose:** Real-time data access

```python
runtime = ToolRuntime(provider, registry, mode="Live")
```

**Characteristics:**
- Allows tools with `live_required=True`
- Network calls permitted
- External API access
- Ideal for: Production inference, real-time monitoring

### Choosing the Right Mode

| Scenario | Mode |
|----------|------|
| Running tests | Replay |
| Creating golden snapshots | Replay |
| Production inference with cached data | Replay |
| Production inference with live API calls | Live |
| Development with mocked data | Replay |
| Fetching current weather/grid data | Live |

---

## Common Patterns & Anti-Patterns

### ✅ Pattern 1: Tool-First Calculation

**GOOD:**
```python
# Define tool
calc_tool = Tool(
    name="calculate_savings",
    description="Calculate cost savings from efficiency upgrade",
    args_schema={...},
    result_schema={
        "type": "object",
        "properties": {
            "savings": {"$ref": "greenlang://schemas/quantity.json"}
        }
    },
    fn=lambda baseline, improved, rate: {
        "savings": {
            "value": (baseline - improved) * rate,
            "unit": "USD"
        }
    }
)

# LLM calls tool, then provides final with claim
{
  "kind": "final",
  "message": "You will save {{claim:0}} per year.",
  "claims": [{"source_call_id": "tc_1", "path": "$.savings", ...}]
}
```

**BAD:**
```python
# LLM guesses/calculates directly
{
  "kind": "final",
  "message": "You will save approximately $2,500 per year.",  # ❌ BLOCKED
  "claims": []
}
```

### ✅ Pattern 2: Multi-Step Calculation

**GOOD:**
```python
# Step 1: Calculate baseline
tool_call_1 = {"tool_name": "calculate_baseline", "arguments": {...}}
# Result: {"baseline": {"value": 1000, "unit": "kWh"}}

# Step 2: Calculate savings using baseline
tool_call_2 = {"tool_name": "calculate_savings", "arguments": {"baseline_kwh": 1000, ...}}
# Result: {"savings": {"value": 2500, "unit": "USD"}}

# Step 3: Final with both claims
{
  "message": "Your baseline is {{claim:0}}, so you can save {{claim:1}}.",
  "claims": [
    {"source_call_id": "tc_1", "path": "$.baseline", ...},
    {"source_call_id": "tc_2", "path": "$.savings", ...}
  ]
}
```

**BAD:**
```python
# LLM does intermediate calculation in its "head"
tool_call_1 = {"tool_name": "calculate_baseline", ...}
# Result: 1000 kWh

# LLM then says: "Since you use 1000 kWh and rate is $0.15, you'll save $150"
# ❌ BLOCKED - the $150 is not from a tool
```

### ❌ Anti-Pattern 1: Raw Number Returns

**BAD:**
```python
energy_tool = Tool(
    name="calculate_energy",
    result_schema={
        "type": "object",
        "properties": {
            "energy": {"type": "number"}  # ❌ NO! Must be Quantity
        }
    },
    fn=lambda: {"energy": 1000}  # ❌ Raw number
)
```

**GOOD:**
```python
energy_tool = Tool(
    name="calculate_energy",
    result_schema={
        "type": "object",
        "properties": {
            "energy": {"$ref": "greenlang://schemas/quantity.json"}
        }
    },
    fn=lambda: {"energy": {"value": 1000, "unit": "kWh"}}  # ✅ Quantity
)
```

### ❌ Anti-Pattern 2: Hard-Coded "Facts"

**BAD:**
```python
{
  "kind": "final",
  "message": "The average home uses 10,000 kWh per year.",  # ❌ Where did this come from?
  "claims": []
}
```

**GOOD:**
```python
# Create tool that returns reference data
reference_tool = Tool(
    name="get_average_consumption",
    description="Get average residential consumption",
    result_schema={...},
    fn=lambda: {"average": {"value": 10000, "unit": "kWh/year"}}
)

# LLM calls tool, then cites it
{
  "message": "The average home uses {{claim:0}} per year.",
  "claims": [{"source_call_id": "tc_1", "path": "$.average", ...}]
}
```

### ❌ Anti-Pattern 3: Mixing Units

**BAD:**
```python
{
  "message": "You use {{claim:0}} of energy, which is {{claim:1}}.",
  "claims": [
    {"path": "$.energy", "quantity": {"value": 1000, "unit": "kWh"}},
    {"path": "$.energy_mwh", "quantity": {"value": 1, "unit": "MWh"}}  # ❌ Don't claim same thing twice
  ]
}
```

**GOOD:**
```python
# Let the tool do unit conversion if needed
{
  "message": "You use {{claim:0}} of energy.",
  "claims": [
    {"path": "$.energy", "quantity": {"value": 1000, "unit": "kWh"}}
  ]
}
# User sees: "You use 1000.00 kWh of energy."
```

---

## Error Handling & Debugging

### Error: NO_NAKED_NUMBERS

**Message:**
```
[NO_NAKED_NUMBERS] Naked number '42' detected at position 23
```

**Cause:** LLM included a digit in the message without a {{claim:i}} macro.

**Fix:**
1. Call a tool to get the value
2. Use {{claim:i}} in final message
3. Provide matching claim in claims[]

**Example:**
```python
# ❌ WRONG
{
  "kind": "final",
  "message": "The answer is 42.",
  "claims": []
}

# ✅ CORRECT
{
  "kind": "final",
  "message": "The answer is {{claim:0}}.",
  "claims": [
    {
      "source_call_id": "tc_1",
      "path": "$.result",
      "quantity": {"value": 42, "unit": "kWh"}
    }
  ]
}
```

### Error: RESULT_SCHEMA

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

### Error: QUANTITY_MISMATCH

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

### Error: PATH_RESOLUTION

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

### Error: EGRESS_BLOCKED

**Message:**
```
[EGRESS_BLOCKED] Tool 'fetch_weather' requires Live mode but runtime is in Replay
```

**Cause:** Tried to call a network-requiring tool in Replay mode.

**Fix:**
1. Switch to Live mode: `ToolRuntime(provider, registry, mode="Live")`
2. OR use snapshot/cached data in Replay mode

---

## Metrics & Observability

After each run, access metrics via:

```python
metrics = runtime.get_metrics()

# Tool usage tracking
print(metrics["tool_use_rate"])              # % of steps that were tool calls
print(metrics["total_tool_calls"])           # Count of tool executions
print(metrics["unique_tools_used"])          # Number of distinct tools

# Policy enforcement
print(metrics["naked_number_rejections"])    # Count of blocked finals
print(metrics["quantity_mismatches"])        # Count of claim mismatches

# Performance
print(metrics["total_steps"])                # Total LLM interactions
print(metrics["avg_tool_latency_ms"])        # Average tool execution time
```

### Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("greenlang.intelligence.runtime")
```

Log output includes:
- Tool call invocations with arguments
- Tool execution results
- Claim verification (match/mismatch)
- Naked number scan results

---

## Testing Guidelines

### Unit Test: Tool Output

```python
def test_tool_returns_quantity():
    """Verify tool returns Quantity, not raw number"""
    registry = ToolRegistry()
    registry.register(energy_tool)

    # Execute tool
    result = registry.invoke("calculate_intensity", {
        "annual_kwh": 10000,
        "floor_m2": 800
    })

    # Verify output is Quantity
    assert "value" in result["intensity"]
    assert "unit" in result["intensity"]
    assert result["intensity"]["unit"] == "kWh/m2"
    assert result["intensity"]["value"] == 12.5
```

### Integration Test: Full Runtime

```python
def test_runtime_no_naked_numbers():
    """Verify runtime enforces no-naked-numbers policy"""
    provider = create_mock_provider([
        {"kind": "tool_call", "tool_name": "calc", ...},
        {
            "kind": "final",
            "message": "Result is {{claim:0}}",
            "claims": [{"source_call_id": "tc_1", ...}]
        }
    ])

    runtime = ToolRuntime(provider, registry, mode="Replay")
    result = runtime.run("system", "user query")

    # Verify no naked numbers in output
    assert "{{claim:" not in result["message"]  # Macros resolved
    assert result["provenance"]  # Has provenance
```

### Snapshot Test: Golden Files

```python
@pytest.mark.snapshot
def test_agent_golden(snapshot):
    """Verify agent output matches golden snapshot"""
    runtime = ToolRuntime(provider, registry, mode="Replay", seed=42)
    result = runtime.run("system", "Calculate intensity")

    # Compare against golden file
    snapshot.assert_match(result["message"], "golden_message.txt")
    snapshot.assert_match(result["provenance"], "golden_provenance.json")
```

---

## Best Practices Summary

### ✅ DO

- Define result schemas with `greenlang://schemas/quantity.json` references
- Return Quantity objects for ALL numeric tool outputs
- Use {{claim:i}} macros for all user-facing numbers
- Provide claims[] with exact matches to tool outputs
- Test tools independently before integration
- Use Replay mode for deterministic testing
- Log provenance for auditing
- Document tool purpose and units in description

### ❌ DON'T

- Return raw numbers from tools
- Hard-code numeric values in final messages
- Mix claimed units with tool output units
- Skip result_schema validation
- Use unknown/custom units without adding to allowlist
- Call network tools in Replay mode without `live_required` flag
- Trust LLM calculations - always use tools
- Guess or approximate - calculate precisely

---

## Complete Working Example

```python
from greenlang.intelligence.runtime.tools import Tool, ToolRegistry, ToolRuntime
from unittest.mock import Mock

# 1. Define tool with Quantity output
carbon_tool = Tool(
    name="calculate_emissions",
    description="Calculate CO2e emissions from fuel combustion",
    args_schema={
        "type": "object",
        "required": ["fuel_kg", "emission_factor"],
        "properties": {
            "fuel_kg": {"type": "number", "minimum": 0},
            "emission_factor": {"type": "number", "minimum": 0}
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
            "message": "Burning 100 kg of fuel produces {{claim:0}} of CO2e emissions.",
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
result = runtime.run(
    system_prompt="You are a climate advisor.",
    user_msg="Calculate emissions for 100 kg of natural gas"
)

# 5. Verify results
print(result["message"])
# Output: "Burning 100 kg of fuel produces 268.00 kgCO2e of CO2e emissions."

print(result["provenance"])
# Output: [{"source_call_id": "tc_1", "path": "$.emissions", "quantity": {...}}]

print(result["metrics"]["tool_use_rate"])
# Output: 0.5  (1 tool call out of 2 steps)
```

---

## Migration Guide: Converting Existing Agents

### Step 1: Audit Current Outputs

Identify all numeric outputs currently returned by your agent:

```python
# Old agent (guessing)
def analyze_building(data):
    return f"Your building uses approximately 10,000 kWh per year"  # ❌
```

### Step 2: Create Tools for Each Calculation

```python
# New approach (tool-grounded)
energy_tool = Tool(
    name="calculate_annual_energy",
    description="Calculate annual energy consumption",
    args_schema={...},
    result_schema={
        "type": "object",
        "properties": {
            "annual_energy": {"$ref": "greenlang://schemas/quantity.json"}
        }
    },
    fn=lambda monthly_kwh: {
        "annual_energy": {
            "value": monthly_kwh * 12,
            "unit": "kWh/year"
        }
    }
)
```

### Step 3: Update Agent to Use ToolRuntime

```python
from greenlang.intelligence.runtime.tools import ToolRuntime

# Register tools
registry = ToolRegistry()
registry.register(energy_tool)

# Create runtime
runtime = ToolRuntime(provider, registry, mode="Replay")

# Run
result = runtime.run(
    system_prompt="You are a building energy advisor.",
    user_msg="Analyze my building"
)
```

### Step 4: Add Tests

```python
def test_agent_no_naked_numbers():
    result = runtime.run("system", "Calculate energy")
    # Verify no raw numbers in output
    assert not re.search(r'\d+(?:\.\d+)?', result["message"], re.IGNORECASE)
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
A: Set `live_required=True` in Tool definition and use `mode="Live"` in runtime.

**Q: How do I handle currency conversions?**
A: Currency is "tagged, non-convertible". 100 USD ≠ 100 EUR. Use separate tools for currency conversion if needed.

**Q: Can I have tools that return arrays of Quantities?**
A: Not directly supported yet. Return a single summary Quantity or make multiple tool calls.

**Q: What if the LLM refuses to use tools?**
A: Tune system prompt to emphasize tool usage. Include examples in few-shot prompts.

**Q: How do I debug claim mismatches?**
A: Enable DEBUG logging to see tool outputs and claim verification details.

---

## References

### Code
- **Runtime:** `greenlang/intelligence/runtime/tools.py`
- **Schemas:** `greenlang/intelligence/runtime/schemas.py`
- **Units:** `greenlang/intelligence/runtime/units.py`
- **Errors:** `greenlang/intelligence/runtime/errors.py`

### Tests
- **Unit Tests:** `tests/intelligence/test_tools_runtime.py`
- **Integration Tests:** `tests/agents/test_*_agent_ai.py`
- **Snapshot Tests:** `artifacts/W1/provenance_samples/`

### Examples
- **Demo:** `examples/runtime_no_naked_numbers_demo.py`
- **Agent Template:** `templates/agent_deployment_pack.yaml`

### Documentation
- **No Naked Numbers Deep Dive:** `docs/intelligence/no-naked-numbers.md`
- **Tool Development:** `docs/intelligence/tool_development.md`
- **Agent Deployment:** `templates/README_DEPLOYMENT_TEMPLATE.md`

---

**Last Updated:** October 22, 2025
**Document Version:** 1.0
**Related:** INTL-103, DEVX-501, AGT-701/702/703
