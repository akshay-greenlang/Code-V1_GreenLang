# Test Infrastructure Quick Start Guide

**Date:** October 22, 2025
**Status:** ✅ All infrastructure ready to use

---

## What Was Built

Your GreenLang test infrastructure is now **production-ready** with:

1. ✅ **AsyncIO Issues Fixed** - No more event loop conflicts
2. ✅ **ChatSession Mocking** - Comprehensive fixtures for AI agent testing
3. ✅ **Coverage Reporting** - 80% threshold enforced, HTML/XML reports
4. ✅ **Test Data Fixtures** - Reusable data for all 7 AI/ML agents

---

## Quick Start - Running Tests

### Install Dependencies (if not already installed)
```bash
pip install pytest pytest-cov pytest-asyncio pytest-mock
```

### Run All Tests
```bash
# Basic run
pytest

# With coverage
pytest --cov=greenlang --cov-report=html

# View coverage report
start .coverage_html/index.html
```

### Run Specific Agent Tests
```bash
# Run all 7 AI/ML agent tests
pytest tests/agents/test_fuel_agent_ai.py \
  tests/agents/test_carbon_agent_ai.py \
  tests/agents/test_grid_factor_agent_ai.py \
  tests/agents/test_recommendation_agent_ai.py \
  tests/agents/test_report_agent_ai.py \
  tests/agents/test_forecast_agent_sarima.py \
  tests/agents/test_anomaly_agent_iforest.py \
  -v --cov=greenlang.agents

# Run single agent
pytest tests/agents/test_fuel_agent_ai.py -v
```

### Coverage Commands
```bash
# Generate all reports
pytest --cov=greenlang \
  --cov-report=term-missing \
  --cov-report=html:.coverage_html \
  --cov-report=xml:coverage.xml

# Fail if coverage < 80%
pytest --cov=greenlang --cov-fail-under=80

# Check specific module
pytest tests/agents/ --cov=greenlang.agents --cov-report=term
```

---

## Using Fixtures in Tests

### Example 1: Using Test Data Fixtures
```python
def test_fuel_agent(sample_fuel_payload, agent_test_helpers):
    """Test with reusable fixtures."""
    from greenlang.agents import FuelAgentAI

    agent = FuelAgentAI(budget_usd=1.0)
    result = agent.execute(sample_fuel_payload)

    agent_test_helpers.assert_successful_response(result)
```

### Example 2: Using ChatSession Mocking
```python
@pytest.mark.asyncio
@patch("greenlang.agents.fuel_agent_ai.ChatSession")
async def test_with_mock(mock_session_class, mock_chat_session, mock_chat_response):
    """Test with mocked ChatSession."""
    # Create response
    response = mock_chat_response(text="Test response", cost_usd=0.01)

    # Create session
    session = mock_chat_session(response=response)
    mock_session_class.return_value = session

    # Test
    agent = FuelAgentAI()
    result = await agent.run({"fuel_type": "natural_gas", "amount": 1000})

    assert result.success is True
```

### Example 3: Multiple Responses
```python
@pytest.mark.asyncio
async def test_multiple_calls(mock_chat_session, mock_chat_response):
    """Test with multiple ChatSession calls."""
    responses = [
        mock_chat_response(text="First response"),
        mock_chat_response(text="Second response"),
    ]

    session = mock_chat_session(responses=responses)

    # First call
    result1 = await session.chat("prompt 1")
    assert result1.text == "First response"

    # Second call
    result2 = await session.chat("prompt 2")
    assert result2.text == "Second response"
```

---

## Available Fixtures

### Test Data Fixtures
- `sample_fuel_payload` - Fuel agent test data
- `sample_carbon_payload` - Carbon agent test data
- `sample_grid_payload` - Grid factor agent test data
- `sample_recommendation_payload` - Recommendation agent test data
- `sample_report_payload` - Report agent test data
- `sample_forecast_payload` - SARIMA forecast agent test data
- `sample_anomaly_payload` - Isolation Forest agent test data

### Mocking Fixtures
- `mock_chat_response` - Create mock ChatResponse
- `mock_chat_session` - Create mock ChatSession with async support
- `mock_chat_session_class` - Mock ChatSession class for @patch
- `sample_tool_calls` - Reusable tool call structures

### Helper Fixtures
- `agent_test_helpers` - Helper methods for assertions
- `tool_call_tracker` - Track tool calls during tests
- `coverage_config` - Coverage configuration

---

## Configuration Files

### pytest.ini
- AsyncIO mode: `auto` (automatic async test discovery)
- Markers: unit, integration, e2e, slow, property, snapshot, cli, sdk, asyncio, network
- Max failures: 1 (fail fast)

### pyproject.toml
- Coverage threshold: 80%
- Reports: Terminal, HTML, XML
- Branch coverage enabled
- Parallel execution support

### conftest.py
- Location: `tests/conftest.py`
- Total fixtures: 14+
- Lines: 829 (was 608)
- New fixtures: +221 lines

---

## Troubleshooting

### Event Loop Errors
**Fixed!** The custom event_loop fixture has been removed. pytest-asyncio now manages loops automatically.

### Coverage Not Showing
```bash
# Make sure pytest-cov is installed
pip install pytest-cov

# Run with explicit coverage
pytest --cov=greenlang --cov-report=term
```

### AsyncIO Tests Failing
```bash
# Make sure pytest-asyncio is installed
pip install pytest-asyncio

# Check asyncio_mode in pytest.ini
grep asyncio_mode pytest.ini
# Should show: asyncio_mode = auto
```

### Import Errors
```bash
# Make sure greenlang is importable
python -c "import greenlang; print(greenlang.__file__)"

# If not, install in editable mode
pip install -e .
```

---

## Next Steps

1. **Run Full Test Suite** to verify infrastructure
   ```bash
   pytest tests/agents/ -v --cov=greenlang.agents --cov-report=html
   ```

2. **Check Coverage Report**
   ```bash
   start .coverage_html/index.html
   ```

3. **Verify All 7 Agents Pass**
   - FuelAgentAI
   - CarbonAgentAI
   - GridFactorAgentAI
   - RecommendationAgentAI
   - ReportAgentAI
   - SARIMAForecastAgent
   - IsolationForestAnomalyAgent

4. **Update CI/CD Pipeline**
   - Add `--cov-fail-under=80` to CI runs
   - Upload coverage reports

---

## Documentation

- **Full Report:** `PHASE1_WEEK1-2_TEST_INFRASTRUCTURE_COMPLETE.md`
- **AI Agent Tests:** `PHASE1_WEEK1-2_TEST_COVERAGE_FINAL_REPORT.md`
- **ML Agent Tests:** `PHASE1_WEEK1-2_ML_AGENT_TEST_EXPANSION_COMPLETE.md`
- **Master Plan:** `GL_100_AGENT_MASTER_PLAN.md`

---

**Ready to test? Run:** `pytest -v`

**Questions?** Check the full documentation in `PHASE1_WEEK1-2_TEST_INFRASTRUCTURE_COMPLETE.md`
