# Agent Coordination Integration Tests

Comprehensive integration test suite for inter-agent coordination in GreenLang industrial automation system.

## Overview

This test suite validates coordination and data flow between related GreenLang agents, ensuring seamless integration, message format compatibility, error handling, and performance under concurrent operations.

## Test Coverage

### 1. GL-001 THERMOSYNC ↔ GL-002 FLAMEGUARD Coordination

**File**: `test_gl001_gl002_coordination.py`

**Scenarios**:
- GL-001 orchestrates GL-002 for boiler optimization
- GL-001 receives heat demand requirements
- GL-001 calls GL-002 to optimize boiler settings
- GL-002 returns optimal combustion parameters
- GL-001 validates and applies recommendations

**Test Count**: 15+ tests covering:
- Boiler optimization requests
- Data flow compatibility
- Valid combustion parameters
- Recommendation validation
- Error handling (GL-002 failure, graceful recovery)
- Concurrent coordination (multiple boilers)
- Latency measurement
- Message format compatibility
- Data integrity across boundaries
- Provenance tracking
- Performance under load
- Timeout handling

### 2. GL-001 THERMOSYNC ↔ GL-006 HEATRECLAIM Coordination

**File**: `test_gl001_gl006_coordination.py`

**Scenarios**:
- GL-001 identifies waste heat streams
- GL-006 analyzes recovery opportunities
- GL-006 returns prioritized opportunities
- GL-001 updates heat distribution strategy

**Test Count**: 15+ tests covering:
- Waste heat stream identification
- Recovery opportunity analysis
- Opportunity prioritization
- Heat distribution updates
- End-to-end recovery workflow
- Economic analysis validation
- Technology selection
- Concurrent stream analysis
- Constraint enforcement
- Data flow compatibility
- Real-time coordination
- Performance with large stream sets

### 3. GL-003 STEAMWISE ↔ GL-008 TRAPCATCHER Coordination

**File**: `test_gl003_gl008_coordination.py`

**Scenarios**:
- GL-003 monitors steam system
- GL-003 detects pressure anomalies
- GL-003 calls GL-008 for steam trap inspection
- GL-008 returns failed trap locations
- GL-003 updates system efficiency calculations

**Test Count**: 15+ tests covering:
- Pressure anomaly detection
- Trap inspection triggering
- Failed trap location reporting
- Efficiency impact calculations
- End-to-end coordination workflow
- Failure mode identification
- Maintenance prioritization
- Steam loss calculation
- Inspection method selection
- Concurrent inspections
- False positive handling
- Real-time monitoring latency
- Data integrity
- Provenance tracking

### 4. GL-002 FLAMEGUARD ↔ GL-010 EMISSIONWATCH Coordination

**File**: `test_gl002_gl010_coordination.py`

**Scenarios**:
- GL-002 optimizes boiler for efficiency
- GL-010 monitors emissions compliance
- GL-002 requests emission constraints from GL-010
- GL-010 provides NOx/SOx limits
- GL-002 optimizes within constraints
- GL-010 validates emissions stay compliant

**Test Count**: 15+ tests covering:
- Emission constraint requests
- Regulatory limit provisioning
- Constrained optimization
- Compliance validation (compliant/non-compliant)
- Violation detection
- End-to-end constrained optimization
- Multi-objective optimization
- Constraint violation handling
- Real-time emissions monitoring
- Dynamic constraint updates
- Concurrent compliance checks
- Violation severity classification
- Efficiency vs emissions tradeoff
- Data format compatibility
- Continuous monitoring performance

### 5. GL-001 THERMOSYNC ↔ GL-009 THERMALIQ Coordination

**File**: `test_gl001_gl009_coordination.py`

**Scenarios**:
- GL-001 requests efficiency analysis
- GL-009 calculates first/second law efficiency
- GL-009 performs exergy analysis
- GL-001 uses efficiency data for optimization

**Test Count**: 15+ tests covering:
- Efficiency analysis requests
- First law (energy) efficiency calculation
- Second law (exergy) efficiency calculation
- Exergy analysis (input, output, destruction)
- Efficiency data for optimization
- End-to-end workflow
- Comprehensive efficiency metrics
- Thermodynamic validation
- Temperature impact on exergy
- Optimization recommendations
- Concurrent analyses
- Performance tracking integration
- Optimization feedback loop
- Real-time analysis latency
- Data format compatibility
- Provenance tracking

## Running Tests

### Run All Agent Coordination Tests

```bash
pytest tests/integration/agent_coordination/ -v
```

### Run Specific Test Suite

```bash
# GL-001 ↔ GL-002 coordination
pytest tests/integration/agent_coordination/test_gl001_gl002_coordination.py -v

# GL-001 ↔ GL-006 coordination
pytest tests/integration/agent_coordination/test_gl001_gl006_coordination.py -v

# GL-003 ↔ GL-008 coordination
pytest tests/integration/agent_coordination/test_gl003_gl008_coordination.py -v

# GL-002 ↔ GL-010 coordination
pytest tests/integration/agent_coordination/test_gl002_gl010_coordination.py -v

# GL-001 ↔ GL-009 coordination
pytest tests/integration/agent_coordination/test_gl001_gl009_coordination.py -v
```

### Run with Coverage

```bash
pytest tests/integration/agent_coordination/ --cov=greenlang.agents --cov-report=html --cov-report=term
```

### Run Performance Tests Only

```bash
pytest tests/integration/agent_coordination/ -m performance -v
```

### Run with Async Support

```bash
pytest tests/integration/agent_coordination/ -v --asyncio-mode=auto
```

## Test Markers

Tests use pytest markers for categorization:

- `@pytest.mark.asyncio` - Async tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.slow` - Slow-running tests

## Fixtures

Shared fixtures are defined in `conftest.py`:

### Agent Fixtures
- `mock_gl001_orchestrator` - Mock GL-001 ProcessHeatOrchestrator
- `mock_gl002_optimizer` - Mock GL-002 BoilerEfficiencyOptimizer
- `mock_gl003_orchestrator` - Mock GL-003 SteamSystemOrchestrator
- `mock_gl006_optimizer` - Mock GL-006 HeatRecoveryOptimizer
- `mock_gl008_monitor` - Mock GL-008 SteamTrapMonitor
- `mock_gl009_analyzer` - Mock GL-009 ThermalEfficiencyAnalyzer
- `mock_gl010_monitor` - Mock GL-010 EmissionsMonitor

### Data Fixtures
- `sample_data_generator` - Generate sample sensor/thermal/emissions data
- `thermal_system_data` - Sample thermal system data
- `waste_heat_streams_payload` - Sample waste heat streams
- `steam_system_data` - Sample steam system monitoring data
- `boiler_operation_data` - Sample boiler operation data

### Helper Fixtures
- `coordination_helpers` - Coordination test helper functions
- `validation_helpers` - Validation helper functions
- `performance_thresholds` - Performance threshold values
- `mock_message_bus` - Mock message bus for agent communication

## Test Structure

Each test file follows this structure:

```python
# Test Fixtures
@pytest.fixture
def agent_config():
    """Agent configuration."""
    pass

@pytest.fixture
def mock_agent(agent_config):
    """Mock agent instance."""
    pass

# Test Class
class TestAgentCoordination:
    """Test suite for agent coordination."""

    @pytest.mark.asyncio
    async def test_coordination_scenario(self, mock_agent1, mock_agent2):
        """Test specific coordination scenario."""
        # Arrange
        # Act
        # Assert
        pass
```

## Performance Requirements

All coordination tests must meet these performance requirements:

| Metric | Requirement |
|--------|-------------|
| Max coordination latency | < 200ms |
| Min throughput | > 10 requests/second |
| Success rate | ≥ 95% |
| Memory increase | < 100MB |

## Data Flow Validation

Tests validate:

1. **Message Format Compatibility**: Agent outputs match expected input formats
2. **Data Integrity**: No data corruption across agent boundaries
3. **Provenance Tracking**: All operations tracked with provenance hashes
4. **Error Propagation**: Errors handled gracefully with proper messaging

## Error Handling Tests

Each test suite includes:

- Agent failure scenarios
- Timeout handling
- Invalid data format handling
- Partial response handling
- Concurrent request failures

## Continuous Integration

Tests run automatically on:

- Pull requests
- Main branch commits
- Nightly builds

CI Configuration:
```yaml
# .github/workflows/agent-coordination-tests.yml
name: Agent Coordination Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run coordination tests
        run: pytest tests/integration/agent_coordination/ -v
```

## Test Data

Test data generators create realistic scenarios:

- **SampleDataGenerator**: Generate sensor/thermal/emissions data
- **MockAgentFactory**: Create mock agents with standard interfaces
- **MockMessageBus**: Simulate inter-agent messaging

## Troubleshooting

### Common Issues

**Issue**: Async test timeout
```
Solution: Increase timeout with @pytest.mark.asyncio(timeout=10)
```

**Issue**: Mock agent not responding
```
Solution: Verify async mock methods use AsyncMock
```

**Issue**: Data format mismatch
```
Solution: Check fixture data structure matches agent expectations
```

## Contributing

When adding new coordination tests:

1. Create descriptive test names: `test_<agent1>_<action>_<agent2>`
2. Use fixtures for reusable test data
3. Include docstrings explaining test scenario
4. Add performance assertions
5. Test both success and failure paths
6. Validate data integrity and provenance
7. Test concurrent operations

## Metrics

Test suite metrics:

- **Total Tests**: 75+ tests across 5 coordination pairs
- **Coverage Target**: 85%+ for coordination logic
- **Execution Time**: < 2 minutes for full suite
- **Pass Rate**: 100% required for merge

## References

- [GreenLang Agent Architecture](../../../docs/agent_architecture.md)
- [Agent Coordination Patterns](../../../docs/coordination_patterns.md)
- [Testing Strategy](../../../docs/testing_strategy.md)
- [Performance Requirements](../../../docs/performance_requirements.md)

## License

Copyright (c) 2025 GreenLang Project
