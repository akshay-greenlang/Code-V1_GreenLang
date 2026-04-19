# Agent Tests

This directory contains tests for GreenLang agents.

## Test Files

### test_fuel_agent_ai.py
Tests for the AI-powered FuelAgent (`FuelAgentAI`).

**Test Coverage:**
- Initialization and configuration
- Input validation
- Tool implementations (calculate_emissions, lookup_emission_factor, generate_recommendations)
- Determinism (same input → same output)
- Backward compatibility with original FuelAgent
- Error handling
- Performance tracking
- Renewable offset calculations
- Efficiency adjustments
- Prompt building
- Mocked AI integration

**Running Tests:**
```bash
# Run all FuelAgentAI tests
pytest tests/agents/test_fuel_agent_ai.py -v

# Run specific test
pytest tests/agents/test_fuel_agent_ai.py::TestFuelAgentAI::test_initialization -v

# Run with coverage
pytest tests/agents/test_fuel_agent_ai.py --cov=greenlang.agents.fuel_agent_ai
```

## Test Classes

### TestFuelAgentAI
Unit tests for FuelAgentAI components (17 tests)

### TestFuelAgentAIIntegration
Integration tests with real/demo LLM (2 tests)

## Notes

- Tests work in demo mode (no API keys required)
- All tool calculations are deterministic
- Backward compatibility verified against FuelAgent
- Uses pytest fixtures for reusable test data
