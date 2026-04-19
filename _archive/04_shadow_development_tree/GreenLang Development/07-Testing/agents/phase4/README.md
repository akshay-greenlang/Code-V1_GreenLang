# Phase 4 Agent Test Suite

Comprehensive test infrastructure for Phase 4 InsightAgent pattern agents.

## Overview

This test suite validates the **InsightAgent pattern** implementation across all 4 Phase 4 agents:
1. **Anomaly Investigation Agent** - Hybrid anomaly detection + AI root cause analysis
2. **Forecast Explanation Agent** - Deterministic SARIMA forecasting + AI narrative generation
3. **Benchmark Insight Agent** - Deterministic benchmarking + AI competitive insights
4. **Report Narrative Agent V2** - Deterministic report data + AI narrative generation

## Test Suite Structure

```
tests/agents/phase4/
├── __init__.py                    # Package initialization
├── conftest.py                    # Pytest fixtures and mocks (331 lines)
├── test_phase4_integration.py     # Integration tests (864 lines, 31 tests)
└── README.md                      # This file
```

**Total: 1,204 lines of test infrastructure**

## Test Coverage

### 1. InsightAgent Pattern Compliance Tests (6 tests)
Validates that all agents follow the InsightAgent pattern:
- ✓ Inherit from `InsightAgent` base class
- ✓ Have `AgentCategory.INSIGHT` category
- ✓ Complete metadata with `uses_chat_session=True`, `uses_rag=True`
- ✓ Implement `calculate()` method (deterministic)
- ✓ Implement `explain()` method (AI-powered)
- ✓ Support audit trail capture

### 2. Agent-Specific Tests (16 tests)

#### Anomaly Investigation Agent (4 tests)
- ✓ Deterministic anomaly detection using Isolation Forest
- ✓ AI root cause analysis with RAG retrieval
- ✓ Tool integration (maintenance_log_tool, sensor_diagnostic_tool, weather_data_tool)
- ✓ Investigation report generation

#### Forecast Explanation Agent (4 tests)
- ✓ Deterministic SARIMA forecasting
- ✓ AI narrative generation with trend/seasonality analysis
- ✓ Tool integration (historical_trend_tool, seasonality_tool, event_correlation_tool)
- ✓ Stakeholder-appropriate explanations

#### Benchmark Insight Agent (4 tests)
- ✓ Deterministic carbon intensity calculations
- ✓ Rating and percentile assignment
- ✓ AI competitive insights with peer comparison
- ✓ Benchmark threshold validation

#### Report Narrative Agent V2 (4 tests)
- ✓ Deterministic report data aggregation (6 tools)
- ✓ Framework-compliant structure (TCFD, CDP, GRI, SASB, SEC, ISO14064)
- ✓ AI narrative generation with RAG-enhanced best practices
- ✓ Tool integration (data_visualization_tool, stakeholder_preference_tool)

### 3. RAG Integration Tests (3 tests)
- ✓ RAG query structure validation
- ✓ Collection specification correctness
- ✓ Knowledge retrieval and formatting
- ✓ Agent-specific collection usage:
  - **Anomaly**: anomaly_patterns, root_cause_database, sensor_specifications, maintenance_procedures
  - **Forecast**: forecasting_patterns, seasonality_library, event_database, forecast_narratives
  - **Benchmark**: industry_benchmarks, best_practices, competitive_analysis, building_performance
  - **Report**: narrative_templates, compliance_guidance, industry_reporting, esg_best_practices

### 4. Temperature and Budget Tests (1 test)
- ✓ All agents use `temperature=0.6` for analytical consistency
- ✓ Budget enforcement (separate for calculate vs explain)
- ✓ Cost tracking per operation

### 5. Error Handling Tests (3 tests)
- ✓ Graceful handling of RAG engine failures
- ✓ Graceful handling of ChatSession failures
- ✓ Input validation and error messages
- ✓ Degraded mode operation

### 6. Reproducibility Tests (2 tests)
- ✓ `calculate()` produces identical results for same inputs
- ✓ Audit trail capture and export
- ✓ Calculation trace completeness
- ✓ Input/output hash verification

### 7. Tool Definition Tests (1 test)
- ✓ Tool schemas follow OpenAI function calling format
- ✓ Required parameters specified
- ✓ Parameter types and descriptions complete
- ✓ Tool execution mocks

## Fixtures in conftest.py

### Mock Infrastructure
- `mock_rag_engine` - RAG engine with configurable query results
- `mock_chat_session` - ChatSession with AI response mocking
- `mock_tool_responses` - Pre-configured tool execution results

### Sample Data
- `sample_anomaly_data` - 500 time series samples with injected anomalies
- `sample_forecast_data` - 48 months of data with trend + seasonality
- `sample_benchmark_context` - Building emissions and characteristics
- `sample_report_context` - Complete report generation inputs

### Assertion Helpers
- `assert_insight_agent_result` - Validates InsightAgent output structure
- `assert_temperature_compliance` - Checks temperature=0.6 usage
- `assert_rag_collections` - Verifies correct RAG collections queried
- `assert_deterministic_calculation` - Tests reproducibility

## Running the Tests

### Run All Phase 4 Tests
```bash
pytest tests/agents/phase4/ -v
```

### Run Specific Test Class
```bash
pytest tests/agents/phase4/test_phase4_integration.py::TestAnomalyInvestigationAgent -v
```

### Run with Coverage
```bash
pytest tests/agents/phase4/ --cov=greenlang.agents --cov-report=html
```

### Run Async Tests Only
```bash
pytest tests/agents/phase4/ -k "asyncio" -v
```

## Test Patterns

### Pattern 1: Deterministic Calculate + AI Explain
```python
# Step 1: Deterministic calculation
result = agent.calculate(inputs)
assert "calculation_trace" in result

# Step 2: AI explanation
explanation = await agent.explain(
    calculation_result=result,
    context=context,
    session=mock_chat_session,
    rag_engine=mock_rag_engine,
    temperature=0.6
)

# Verify
assert mock_rag_engine.query.called
assert mock_chat_session.chat.called
```

### Pattern 2: Reproducibility Verification
```python
result1 = agent.calculate(inputs)
result2 = agent.calculate(inputs)

# Compare numeric fields
for key in result1.keys():
    if isinstance(result1[key], (int, float)):
        assert result1[key] == result2[key]
```

### Pattern 3: Tool Integration Testing
```python
tools = agent._get_investigation_tools()

# Verify tool schema
for tool in tools:
    assert "name" in tool
    assert "description" in tool
    assert "parameters" in tool
    assert tool["parameters"]["type"] == "object"
```

## Key Differences from Phase 3

| Aspect | Phase 3 (ReasoningAgent) | Phase 4 (InsightAgent) |
|--------|-------------------------|------------------------|
| Base Class | `ReasoningAgent` | `InsightAgent` |
| Category | `RECOMMENDATION` | `INSIGHT` |
| Temperature | 0.7 (creative reasoning) | 0.6 (analytical consistency) |
| Methods | `reason()` (full AI) | `calculate()` + `explain()` (hybrid) |
| Calculations | AI-driven with tools | Deterministic + AI narratives |
| Audit Trail | Optional | Supported for calculations |
| Reproducibility | Not guaranteed | Guaranteed for calculate() |

## Architecture Compliance

### InsightAgent Requirements
All Phase 4 agents must:
1. ✓ Inherit from `InsightAgent` base class
2. ✓ Implement deterministic `calculate()` method
3. ✓ Implement async `explain()` method with RAG
4. ✓ Use temperature ≤ 0.7 for consistency
5. ✓ Category = `AgentCategory.INSIGHT`
6. ✓ Support audit trail for calculations
7. ✓ Clear separation: numbers (calculate) vs narratives (explain)

### Tool Architecture
- **Calculate Phase**: Deterministic tools (database queries, calculations)
- **Explain Phase**: AI enhancement tools (visualization recommendations, stakeholder tailoring)

### RAG Collections
Each agent has 4+ specialized collections:
- Knowledge grounding for AI insights
- Historical patterns and case studies
- Best practices and benchmarks
- Framework-specific guidance

## Test Execution Requirements

### Dependencies
```python
pytest>=7.0.0
pytest-asyncio>=0.21.0
pandas>=1.5.0
numpy>=1.24.0
```

### Environment
- Python 3.10+
- No external APIs required (all mocked)
- No database connections required
- Fully isolated unit tests

## Success Criteria

All tests must pass with:
- ✓ 100% deterministic calculate() behavior
- ✓ Temperature 0.6 enforcement
- ✓ Proper RAG collection usage
- ✓ Tool integration validation
- ✓ Error handling coverage
- ✓ Audit trail completeness

## Future Enhancements

1. **Performance Benchmarks**: Add timing tests for calculate() speed
2. **Integration Tests**: Test with real RAG engine and ChatSession
3. **Load Tests**: Verify performance under concurrent requests
4. **Compliance Tests**: CSRD/CBAM regulatory compliance validation
5. **E2E Tests**: Full workflow tests with real data

## Related Documentation

- **Phase 3 Tests**: `tests/agents/phase3/` (ReasoningAgent pattern)
- **Base Agents**: `greenlang/agents/base_agents.py` (InsightAgent implementation)
- **Agent Categories**: `greenlang/agents/categories.py` (AgentCategory enum)
- **Phase 4 Agents**:
  - `greenlang/agents/anomaly_investigation_agent.py`
  - `greenlang/agents/forecast_explanation_agent.py`
  - `greenlang/agents/benchmark_agent_ai.py`
  - `greenlang/agents/report_narrative_agent_ai_v2.py`

## Contact

For questions or issues with Phase 4 tests, refer to:
- **Architecture**: InsightAgent pattern documentation
- **Test Patterns**: Phase 3 test suite (similar structure)
- **Agent Implementation**: Individual agent docstrings

---

**Test Suite Status**: ✅ Complete (31 tests, 1,204 lines)
**Coverage**: Architecture compliance, tool integration, RAG usage, error handling, reproducibility
**Pattern**: InsightAgent (deterministic + AI hybrid)
**Last Updated**: 2025-11-06
