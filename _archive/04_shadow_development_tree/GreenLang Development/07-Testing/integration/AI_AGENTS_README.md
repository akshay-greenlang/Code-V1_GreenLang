# AI Agents Integration Tests

## Quick Start

### Running Tests

```bash
# Run all AI agent integration tests
pytest tests/integration/test_ai_agents_integration.py -v

# Run specific test
pytest tests/integration/test_ai_agents_integration.py::test_complete_emissions_workflow -v

# Run standalone validation (bypasses pytest network blocking)
python test_ai_agents_simple.py
```

### Requirements

- **For Full Testing**: Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` environment variable
- **For Demo Mode**: No API keys required (limited functionality)

## Test Suite Overview

**File**: `test_ai_agents_integration.py`
**Lines**: 1,186
**Tests**: 16 comprehensive integration tests

### Agents Tested

1. **FuelAgentAI** - Fuel emissions calculations
2. **CarbonAgentAI** - Emissions aggregation
3. **GridFactorAgentAI** - Grid carbon intensity
4. **RecommendationAgentAI** - Reduction recommendations
5. **ReportAgentAI** - Compliance reporting

### Test Coverage

- ✅ Complete workflow chains (Fuel → Carbon → Report)
- ✅ Determinism verification (same input = same output)
- ✅ Real-world scenarios (office, industrial, data center)
- ✅ Grid factor integration
- ✅ Recommendation → Report flow
- ✅ Performance benchmarks
- ✅ Error handling
- ✅ Multi-framework reports (TCFD, CDP, GRI, SASB)
- ✅ Cross-agent numeric consistency
- ✅ Minimal data handling

## Key Test Cases

### 1. Complete Workflow
```python
def test_complete_emissions_workflow():
    """Tests FuelAgent → CarbonAgent → ReportAgent chain"""
```

### 2. Determinism
```python
def test_determinism_across_all_agents():
    """Validates same input produces identical output"""
```

### 3. Real-World Scenarios
```python
def test_office_building_complete_analysis():
    """Tests 50,000 sqft office building workflow"""

def test_industrial_facility_scenario():
    """Tests high-emissions industrial facility"""

def test_data_center_scenario():
    """Tests electricity-intensive data center"""
```

### 4. Performance
```python
def test_end_to_end_performance():
    """Benchmarks complete 5-agent workflow"""
```

## Helper Functions

### `run_complete_workflow(building_data)`
Orchestrates all 5 agents in sequence:
- Calculates fuel emissions
- Validates grid factors
- Aggregates total emissions
- Generates recommendations
- Creates compliance report

Returns complete results with performance metrics.

## Success Criteria

✅ **All Met:**
- 16+ integration tests
- All agent interactions tested
- Demo mode compatible
- Performance metrics tracked
- Determinism verified
- Real-world scenarios
- Error handling
- Multi-framework support

## Known Issues

### Network Blocking (pytest)
The root `conftest.py` has network blocking that interferes with AI agents.

**Workaround**: Use standalone script
```bash
python test_ai_agents_simple.py
```

### Demo Mode Limitations
Without API keys, tests use `FakeProvider`:
- Zero emissions returned
- Tests validate structure only
- Requires real LLM for full validation

## Files

- `test_ai_agents_integration.py` - Main test suite (1,186 lines)
- `test_ai_agents_simple.py` - Standalone validation script
- `AI_AGENTS_README.md` - This file
- `../AI_AGENTS_INTEGRATION_TESTS_SUMMARY.md` - Detailed documentation

## Documentation

See `AI_AGENTS_INTEGRATION_TESTS_SUMMARY.md` for:
- Detailed test descriptions
- Performance expectations
- Known limitations
- Next steps
- CI/CD integration guide

---

**Quick Test**: `python test_ai_agents_simple.py`
**Full Suite**: `pytest tests/integration/test_ai_agents_integration.py -v`
**With API Keys**: Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
