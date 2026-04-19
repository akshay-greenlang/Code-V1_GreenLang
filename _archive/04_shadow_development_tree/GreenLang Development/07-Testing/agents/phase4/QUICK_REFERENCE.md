# Phase 4 Test Suite Quick Reference

## Test Files Created

```
tests/agents/phase4/
├── __init__.py                    # 9 lines - Package initialization
├── conftest.py                    # 331 lines - Fixtures and mocks
├── test_phase4_integration.py     # 864 lines - 31 comprehensive tests
├── README.md                      # 9.6KB - Complete documentation
└── QUICK_REFERENCE.md            # This file
```

**Total: 1,204 lines of test infrastructure**

## Agents Under Test

| Agent | Pattern | Temperature | Calculate | Explain |
|-------|---------|-------------|-----------|---------|
| Anomaly Investigation | InsightAgent | 0.6 | Isolation Forest | Root cause analysis |
| Forecast Explanation | InsightAgent | 0.6 | SARIMA forecast | Narrative generation |
| Benchmark Insight | InsightAgent | 0.6 | Carbon intensity | Competitive insights |
| Report Narrative V2 | InsightAgent | 0.6 | Report data (6 tools) | Framework narrative |

## Test Categories (31 Tests)

1. **Architecture Compliance** (6 tests)
   - InsightAgent inheritance
   - Category validation
   - Metadata completeness
   - Method implementation

2. **Agent-Specific** (16 tests, 4 per agent)
   - Deterministic calculations
   - AI explanation generation
   - Tool integration
   - Result validation

3. **RAG Integration** (3 tests)
   - Query structure
   - Collection usage
   - Knowledge retrieval

4. **Temperature/Budget** (1 test)
   - Temperature 0.6 enforcement
   - Cost tracking

5. **Error Handling** (3 tests)
   - RAG failures
   - ChatSession failures
   - Input validation

6. **Reproducibility** (2 tests)
   - Deterministic behavior
   - Audit trail capture

## Quick Run Commands

```bash
# Run all Phase 4 tests
pytest tests/agents/phase4/ -v

# Run specific agent tests
pytest tests/agents/phase4/ -k "Anomaly" -v
pytest tests/agents/phase4/ -k "Forecast" -v
pytest tests/agents/phase4/ -k "Benchmark" -v
pytest tests/agents/phase4/ -k "Report" -v

# Run async tests only
pytest tests/agents/phase4/ -k "asyncio" -v

# Run with coverage
pytest tests/agents/phase4/ --cov=greenlang.agents --cov-report=html

# Run specific test class
pytest tests/agents/phase4/test_phase4_integration.py::TestAnomalyInvestigationAgent -v
```

## Key Fixtures (conftest.py)

### Mock Infrastructure
- `mock_rag_engine` - Mocked RAG with 4 sample chunks
- `mock_chat_session` - Mocked ChatSession with AI responses
- `mock_tool_responses` - Pre-configured tool results

### Sample Data
- `sample_anomaly_data` - 500 samples, 4 injected anomalies
- `sample_forecast_data` - 48 months, trend + seasonality
- `sample_benchmark_context` - Building emissions data
- `sample_report_context` - TCFD report inputs

### Assertion Helpers
- `assert_insight_agent_result(calc_result, explanation)`
- `assert_temperature_compliance(mock_session, 0.6)`
- `assert_rag_collections(mock_rag, expected_collections)`
- `assert_deterministic_calculation(agent, inputs)`

## Test Pattern Examples

### Basic Test Structure
```python
@pytest.mark.asyncio
async def test_agent_workflow(
    mock_rag_engine,
    mock_chat_session,
    sample_data
):
    agent = AnomalyInvestigationAgent()

    # Step 1: Deterministic calculation
    result = agent.calculate({"data": sample_data})
    assert "calculation_trace" in result

    # Step 2: AI explanation
    explanation = await agent.explain(
        calculation_result=result,
        context={"system_type": "HVAC"},
        session=mock_chat_session,
        rag_engine=mock_rag_engine,
        temperature=0.6
    )

    # Verify
    assert isinstance(explanation, str)
    assert len(explanation) > 100
```

### Reproducibility Test
```python
def test_deterministic(sample_data):
    agent = BenchmarkAgentAI()

    result1 = agent.calculate(sample_data)
    result2 = agent.calculate(sample_data)

    assert result1["carbon_intensity"] == result2["carbon_intensity"]
    assert result1["rating"] == result2["rating"]
```

### RAG Collection Test
```python
async def test_rag_collections(
    mock_rag_engine,
    mock_chat_session,
    assert_rag_collections
):
    agent = ForecastExplanationAgent()

    calc_result = agent.calculate({...})
    await agent.explain(calc_result, {...}, mock_chat_session, mock_rag_engine)

    assert_rag_collections(mock_rag_engine, [
        "forecasting_patterns",
        "seasonality_library",
        "event_database",
        "forecast_narratives"
    ])
```

## RAG Collections by Agent

| Agent | Collections |
|-------|-------------|
| **Anomaly Investigation** | anomaly_patterns, root_cause_database, sensor_specifications, maintenance_procedures |
| **Forecast Explanation** | forecasting_patterns, seasonality_library, event_database, forecast_narratives |
| **Benchmark Insight** | industry_benchmarks, best_practices, competitive_analysis, building_performance |
| **Report Narrative** | narrative_templates, compliance_guidance, industry_reporting, esg_best_practices |

## Tools by Agent

| Agent | Calculate Tools | Explain Tools |
|-------|----------------|---------------|
| **Anomaly** | isolation_forest | maintenance_log_tool, sensor_diagnostic_tool, weather_data_tool |
| **Forecast** | sarima_model | historical_trend_tool, seasonality_tool, event_correlation_tool |
| **Benchmark** | carbon_intensity_calc | (RAG-only, no tools) |
| **Report** | 6 data tools (fetch, trends, charts, format, compliance, summary) | data_visualization_tool, stakeholder_preference_tool |

## Expected Test Results

All 31 tests should **PASS** with:
- ✅ InsightAgent pattern compliance verified
- ✅ Temperature 0.6 enforced across all agents
- ✅ Deterministic calculate() behavior confirmed
- ✅ RAG integration validated
- ✅ Tool schemas verified
- ✅ Error handling tested
- ✅ Audit trails captured

## Common Issues & Solutions

### Issue: RAG mock not returning chunks
**Solution**: Verify `mock_rag_engine.query` is AsyncMock with chunks attribute

### Issue: Temperature not 0.6
**Solution**: Check agent default temperature parameter in explain() call

### Issue: Calculate not deterministic
**Solution**: Ensure no random state, timestamps, or network calls in calculate()

### Issue: Tool schema validation fails
**Solution**: Verify all tools have name, description, parameters with type="object"

## Success Metrics

- **31/31 tests passing** ✅
- **Code coverage**: Aim for 80%+ on Phase 4 agents
- **No warnings** during test execution
- **All async tests** complete without hanging
- **Deterministic tests** pass on repeated runs

## Next Steps After Tests Pass

1. ✅ Verify test discovery: `pytest --collect-only tests/agents/phase4/`
2. ✅ Run full suite: `pytest tests/agents/phase4/ -v`
3. ✅ Check coverage: `pytest tests/agents/phase4/ --cov`
4. ✅ Run with different seeds: Verify reproducibility
5. ✅ Integration test: Test with real ChatSession/RAG (optional)

## File Sizes

- `conftest.py`: 11KB (331 lines of fixtures)
- `test_phase4_integration.py`: 29KB (864 lines, 31 tests)
- `README.md`: 9.6KB (comprehensive documentation)
- **Total**: ~50KB of test infrastructure

---

**Quick Status Check**:
```bash
# Count tests
grep -c "def test_" tests/agents/phase4/test_phase4_integration.py
# Expected: 31

# Count async tests
grep -c "async def test_" tests/agents/phase4/test_phase4_integration.py
# Expected: 13

# Count total lines
wc -l tests/agents/phase4/*.py
# Expected: 1,204 total
```

**Last Updated**: 2025-11-06
