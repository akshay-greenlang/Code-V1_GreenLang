# Tool Authoring Guide for GreenLang Agents

**Version:** 1.0
**Last Updated:** 2025-10-01
**Status:** Production Ready

## Overview

This guide shows how to expose GreenLang agent methods as LLM-callable tools using the `@tool` decorator. Tools enable LLMs (OpenAI, Anthropic) to invoke agent functionality via function calling.

## Quick Start

### 1. Import the Decorator

```python
from greenlang.intelligence.runtime.tools import tool
```

### 2. Decorate Your Method

```python
@tool(
    name="calculate_carbon_footprint",
    description="Calculate total carbon footprint from emission sources",
    parameters_schema={
        "type": "object",
        "properties": {
            "emissions": {
                "type": "array",
                "description": "List of emission sources",
                "items": {
                    "type": "object",
                    "properties": {
                        "fuel_type": {"type": "string"},
                        "co2e_emissions_kg": {"type": "number"}
                    }
                }
            }
        },
        "required": ["emissions"]
    },
    returns_schema={
        "type": "object",
        "properties": {
            "total_co2e": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                    "source": {"type": "string"}
                }
            }
        }
    },
    timeout_s=10.0
)
def calculate_carbon_footprint(self, emissions: List[Dict]) -> Dict:
    # Your implementation here
    pass
```

### 3. Register with ToolRegistry

```python
from greenlang.intelligence.runtime.tools import ToolRegistry

agent = CarbonAgent()
registry = ToolRegistry()
registry.register_from_agent(agent)

# Use in LLM calls
tool_defs = registry.get_tool_defs()
response = await provider.chat(messages=[...], tools=tool_defs)
```

## Complete Example: CarbonAgent

```python
from typing import List, Dict, Any
from greenlang.agents.base import BaseAgent, AgentResult
from greenlang.intelligence.runtime.tools import tool


class CarbonAgent(BaseAgent):
    """Agent for calculating carbon footprint"""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Original agent execution method"""
        emissions_list = input_data["emissions"]
        total_co2e = sum(e.get("co2e_emissions_kg", 0) for e in emissions_list)

        return AgentResult(
            success=True,
            data={
                "total_co2e_kg": total_co2e,
                "total_co2e_tons": total_co2e / 1000
            }
        )

    @tool(
        name="calculate_carbon_footprint",
        description=(
            "Calculate total carbon footprint from a list of emission sources. "
            "Aggregates CO2e emissions and provides breakdown by source. "
            "Returns total emissions with units and sources (No Naked Numbers)."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "emissions": {
                    "type": "array",
                    "description": "List of emission sources with CO2e values",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fuel_type": {
                                "type": "string",
                                "description": "Type of fuel or emission source"
                            },
                            "co2e_emissions_kg": {
                                "type": "number",
                                "minimum": 0,
                                "description": "CO2e emissions in kilograms"
                            }
                        },
                        "required": ["fuel_type", "co2e_emissions_kg"]
                    }
                }
            },
            "required": ["emissions"]
        },
        returns_schema={
            "type": "object",
            "properties": {
                "total_co2e": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "unit": {"type": "string", "enum": ["kg_CO2e"]},
                        "source": {"type": "string"}
                    },
                    "required": ["value", "unit", "source"]
                },
                "total_co2e_tons": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "unit": {"type": "string", "enum": ["tonnes_CO2e"]},
                        "source": {"type": "string"}
                    },
                    "required": ["value", "unit", "source"]
                }
            },
            "required": ["total_co2e", "total_co2e_tons"]
        },
        timeout_s=10.0
    )
    def calculate_carbon_footprint(
        self,
        emissions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        LLM-callable tool wrapper for execute()

        Transforms input/output to match LLM function calling requirements
        and enforces "No Naked Numbers" rule.
        """
        # Call existing execute method
        result = self.execute({"emissions": emissions})

        # Transform to "No Naked Numbers" format
        return {
            "total_co2e": {
                "value": result.data["total_co2e_kg"],
                "unit": "kg_CO2e",
                "source": "CarbonAgent aggregation"
            },
            "total_co2e_tons": {
                "value": result.data["total_co2e_tons"],
                "unit": "tonnes_CO2e",
                "source": "CarbonAgent aggregation"
            }
        }
```

## Critical Rules

### 1. No Naked Numbers

**ALL numeric values MUST include:**
- `value`: The numeric value
- `unit`: The unit of measurement
- `source`: Where the value came from

❌ **BAD:**
```python
return {"total_co2e": 504.4}
```

✅ **GOOD:**
```python
return {
    "total_co2e": {
        "value": 504.4,
        "unit": "kg_CO2e",
        "source": "CarbonAgent aggregation"
    }
}
```

### 2. Wrap Existing Methods

**DO NOT rewrite agent logic.** The `@tool` method should wrap existing `execute()` or `run()` methods:

```python
@tool(...)
def tool_method(self, arg1, arg2):
    # Build input for existing method
    input_data = {"arg1": arg1, "arg2": arg2}

    # Call existing method
    result = self.execute(input_data)

    # Transform output to match returns_schema
    return transform_to_no_naked_numbers(result.data)
```

### 3. JSON Schema Validation

Both `parameters_schema` and `returns_schema` are **validated automatically**:
- Invalid arguments → `ToolInvocationError`
- Invalid returns → `ToolInvocationError`

Use JSON Schema Draft 7 syntax:
- `type`: "object", "array", "string", "number", "integer", "boolean"
- `properties`: Object field definitions
- `items`: Array item schema
- `required`: Required field list
- `minimum`, `maximum`: Numeric constraints
- `enum`: Allowed string values
- `description`: Field documentation

### 4. Timeout Configuration

Set `timeout_s` based on expected execution time:
- Simple lookups: `5.0s`
- Calculations: `10.0s`
- Simulations: `30.0s`
- Complex analysis: `60.0s`

If timeout is exceeded, `ToolInvocationError` is raised.

## Common Patterns

### Pattern 1: Simple Lookup

```python
@tool(
    name="get_emission_factor",
    description="Retrieve emission factor for country/fuel",
    parameters_schema={
        "type": "object",
        "properties": {
            "country": {"type": "string"},
            "fuel_type": {"type": "string"}
        },
        "required": ["country", "fuel_type"]
    },
    returns_schema={
        "type": "object",
        "properties": {
            "emission_factor": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                    "source": {"type": "string"}
                }
            }
        }
    },
    timeout_s=5.0
)
def get_emission_factor(self, country: str, fuel_type: str) -> Dict:
    result = self.run({"country": country, "fuel_type": fuel_type})
    data = result["data"]
    return {
        "emission_factor": {
            "value": data["emission_factor"],
            "unit": data["unit"],
            "source": data["source"]
        }
    }
```

### Pattern 2: Complex Calculation

```python
@tool(
    name="simulate_energy_balance",
    description="Simulate hourly solar energy balance",
    parameters_schema={
        "type": "object",
        "properties": {
            "solar_data": {"type": "string"},  # JSON DataFrame
            "load_data": {"type": "string"},   # JSON DataFrame
            "area_m2": {"type": "number", "minimum": 0}
        },
        "required": ["solar_data", "load_data", "area_m2"]
    },
    returns_schema={
        "type": "object",
        "properties": {
            "solar_fraction": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "unit": {"type": "string", "enum": ["fraction"]},
                    "source": {"type": "string"}
                }
            }
        }
    },
    timeout_s=30.0
)
def simulate_energy_balance(
    self,
    solar_data: str,
    load_data: str,
    area_m2: float
) -> Dict:
    result = self.execute({
        "solar_data": solar_data,
        "load_data": load_data,
        "area_m2": area_m2
    })

    return {
        "solar_fraction": {
            "value": result.data["solar_fraction"],
            "unit": "fraction",
            "source": "EnergyBalanceAgent simulation"
        }
    }
```

### Pattern 3: Array Returns

```python
@tool(
    name="get_emissions_breakdown",
    description="Get emissions breakdown by source",
    parameters_schema={...},
    returns_schema={
        "type": "object",
        "properties": {
            "breakdown": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "emissions": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "number"},
                                "unit": {"type": "string"},
                                "source": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
    },
    timeout_s=10.0
)
def get_emissions_breakdown(self, emissions_list: List) -> Dict:
    # Return array of emissions, each with units/sources
    breakdown = []
    for emission in emissions_list:
        breakdown.append({
            "source": emission["fuel_type"],
            "emissions": {
                "value": emission["co2e_emissions_kg"],
                "unit": "kg_CO2e",
                "source": "User input"
            }
        })
    return {"breakdown": breakdown}
```

## Testing Your Tool

### Unit Test Template

```python
import pytest
from greenlang.intelligence.runtime.tools import ToolRegistry


def test_my_agent_tool():
    """Test MyAgent tool registration and invocation"""
    agent = MyAgent()
    registry = ToolRegistry()

    # 1. Test registration
    count = registry.register_from_agent(agent)
    assert count == 1
    assert "my_tool" in registry.get_tool_names()

    # 2. Test invocation with valid args
    result = registry.invoke("my_tool", {"arg1": "value1"})
    assert "output" in result

    # 3. Test "No Naked Numbers" compliance
    assert "value" in result["output"]
    assert "unit" in result["output"]
    assert "source" in result["output"]

    # 4. Test invalid args
    with pytest.raises(ToolInvocationError):
        registry.invoke("my_tool", {})  # Missing required arg
```

### Integration Test with LLM

```python
import asyncio
from greenlang.intelligence.providers.openai import OpenAIProvider
from greenlang.intelligence.runtime.budget import Budget


async def test_llm_tool_integration():
    """Test LLM calling agent tool"""
    # Setup
    agent = CarbonAgent()
    registry = ToolRegistry()
    registry.register_from_agent(agent)

    provider = OpenAIProvider(config)

    # LLM call with tools
    messages = [
        {"role": "system", "content": "You are a climate analyst."},
        {"role": "user", "content": "Calculate emissions for 100 kg diesel"}
    ]

    response = await provider.chat(
        messages=messages,
        tools=registry.get_tool_defs(),
        budget=Budget(max_usd=0.10)
    )

    # Verify LLM requested tool
    assert response.tool_calls
    assert response.tool_calls[0]["name"] == "calculate_carbon_footprint"

    # Execute tool
    result = registry.invoke(
        response.tool_calls[0]["name"],
        response.tool_calls[0]["arguments"]
    )

    # Verify result
    assert "total_co2e" in result
    assert result["total_co2e"]["unit"] == "kg_CO2e"
```

## Troubleshooting

### Error: "Tool validation failed"

**Cause:** Arguments don't match `parameters_schema`

**Solution:** Check JSON Schema carefully. Common issues:
- Wrong type (string vs number)
- Missing required field
- Extra fields not in schema

### Error: "Return validation failed"

**Cause:** Return value doesn't match `returns_schema`

**Solution:** Ensure all numeric values have `value`, `unit`, `source` structure.

### Error: "Tool not found"

**Cause:** Tool not registered or wrong name

**Solution:**
1. Check `registry.get_tool_names()` to see registered tools
2. Verify `@tool(name="...")` matches invoke name
3. Ensure `register_from_agent()` was called

### Error: "Tool timeout"

**Cause:** Execution exceeded `timeout_s`

**Solution:**
1. Increase timeout for long-running operations
2. Optimize agent implementation
3. Split into multiple smaller tools

## Best Practices

### 1. One Tool Per Primary Operation

Each agent should have **one primary tool** that wraps its main functionality:
- CarbonAgent → `calculate_carbon_footprint`
- GridFactorAgent → `get_emission_factor`
- EnergyBalanceAgent → `simulate_solar_energy_balance`

### 2. Clear, Actionable Descriptions

LLMs use descriptions to decide when to call tools. Be specific:

❌ **BAD:** "Calculate emissions"
✅ **GOOD:** "Calculate total carbon footprint from a list of emission sources. Aggregates CO2e emissions and provides breakdown by source."

### 3. Validate Early, Return Fast

Check arguments early and fail fast:

```python
@tool(...)
def my_tool(self, arg1: str) -> Dict:
    # Validate first
    if not arg1:
        raise ValueError("arg1 cannot be empty")

    # Then execute
    result = self.execute({"arg1": arg1})
    return transform(result)
```

### 4. Document with Examples

Add docstrings with usage examples:

```python
@tool(...)
def calculate_carbon_footprint(self, emissions: List) -> Dict:
    """
    Calculate total carbon footprint

    Example:
        >>> result = agent.calculate_carbon_footprint(
        ...     emissions=[
        ...         {"fuel_type": "diesel", "co2e_emissions_kg": 268.5}
        ...     ]
        ... )
        >>> print(result["total_co2e"]["value"])
        268.5
    """
```

### 5. Use Enums for Known Values

Restrict string values with `enum`:

```python
"unit": {"type": "string", "enum": ["kg_CO2e", "tonnes_CO2e", "lbs_CO2e"]}
```

## Migration Checklist

Converting existing agents to tools:

- [ ] Import `@tool` decorator
- [ ] Add `@tool` to primary method
- [ ] Define `parameters_schema` (JSON Schema)
- [ ] Define `returns_schema` (JSON Schema with No Naked Numbers)
- [ ] Set appropriate `timeout_s`
- [ ] Wrap existing `execute()` / `run()` method
- [ ] Transform returns to include units/sources
- [ ] Write unit test for registration
- [ ] Write unit test for invocation
- [ ] Write integration test with LLM
- [ ] Update agent documentation

## References

- **ToolRegistry Implementation:** `greenlang/intelligence/runtime/tools.py`
- **Example Agents:**
  - `greenlang/agents/carbon_agent.py`
  - `greenlang/agents/grid_factor_agent.py`
  - `greenlang/agents/energy_balance_agent.py`
- **Tests:** `tests/intelligence/test_tools.py`
- **Complete Demo:** `examples/intelligence/complete_demo.py`
- **JSON Schema Spec:** https://json-schema.org/draft-07/schema

## Support

Questions? Open an issue on GitHub or contact the GreenLang Intelligence team.

---

**Last Updated:** 2025-10-01
**Maintainer:** GreenLang Intelligence Team
**Version:** 1.0
