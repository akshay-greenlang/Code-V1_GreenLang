# GL-{ID}: {AGENT_NAME}

> {Short one-line description of the agent}

## Overview

{2-3 paragraph description of what this agent does, its purpose, and when to use it}

## Category Information

| Property | Value |
|----------|-------|
| **Agent ID** | GL-{ID} |
| **Agent Name** | {AGENT_NAME} |
| **Category** | {Category} |
| **Type** | {Calculator/Optimizer/Monitor/etc.} |
| **Complexity** | {Low/Medium/High} |
| **Priority** | {P0/P1/P2/P3} |
| **Status** | {Implemented/In Progress/Planned} |

## Standards Compliance

This agent implements calculations according to:

- {Standard 1, e.g., "GHG Protocol Corporate Standard"}
- {Standard 2, e.g., "ISO 14064-1:2018"}
- {Standard 3}

## Input Schema

### Required Inputs

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `{field_name}` | `{type}` | {unit} | {Description} |

### Optional Inputs

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `{field_name}` | `{type}` | `{default}` | {Description} |

### Example Input

```json
{
  "field_1": "value",
  "field_2": 100,
  "config": {
    "option": true
  }
}
```

## Output Schema

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `{field_name}` | `{type}` | {unit} | {Description} |

### Example Output

```json
{
  "result": 42.5,
  "unit": "tCO2e",
  "methodology": "GHG Protocol",
  "provenance": {
    "agent_id": "GL-{ID}",
    "version": "1.0.0",
    "emission_factors_used": []
  }
}
```

## Methodology

### Calculation Formula

{Describe the core formula or methodology}

```
Result = Input_A × Emission_Factor × Conversion_Factor
```

### Data Sources

| Source | Version | Coverage | Link |
|--------|---------|----------|------|
| {EPA} | {2024} | {US} | [Link](url) |
| {DEFRA} | {2024} | {UK} | [Link](url) |

### Assumptions

1. {Assumption 1}
2. {Assumption 2}
3. {Assumption 3}

### Limitations

- {Limitation 1}
- {Limitation 2}

## Usage Examples

### Python SDK

```python
from gl_agent_factory import AgentRegistry

registry = AgentRegistry()
agent = registry.get_agent("GL-{ID}")

result = agent.calculate({
    "input_field": 100,
    "region": "US"
})

print(f"Result: {result.value} {result.unit}")
```

### API

```bash
curl -X POST "https://api.greenlang.io/v1/agents/GL-{ID}/execute" \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "input_field": 100,
      "region": "US"
    }
  }'
```

## Testing

### Run Agent Tests

```bash
pytest backend/agents/gl_{id}_{name}/tests/ -v
```

### Test Coverage

Current coverage: {XX}%

## Related Agents

- [GL-{RELATED_ID_1}](../gl_{related_id_1}/README.md) - {Brief description}
- [GL-{RELATED_ID_2}](../gl_{related_id_2}/README.md) - {Brief description}

## Changelog

### Version 1.0.0 (YYYY-MM-DD)

- Initial implementation
- {Feature 1}
- {Feature 2}

## References

1. {Reference 1 with link}
2. {Reference 2 with link}

---

*Last updated: YYYY-MM-DD*
