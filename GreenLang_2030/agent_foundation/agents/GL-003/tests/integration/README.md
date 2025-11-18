# GL-003 SteamSystemAnalyzer Integration Tests

Comprehensive integration test suite for GL-003 SteamSystemAnalyzer agent. These tests validate complete system integration including SCADA connectivity, steam meters, pressure sensors, end-to-end workflows, and parent agent coordination.

## Table of Contents

- [Overview](#overview)
- [Test Infrastructure](#test-infrastructure)
- [Test Coverage](#test-coverage)
- [Quick Start](#quick-start)
- [Running Tests](#running-tests)
- [Test Structure](#test-structure)
- [Mock Services](#mock-services)
- [Configuration](#configuration)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Overview

The GL-003 integration test suite validates:

- **SCADA/DCS Integration**: OPC UA, Modbus TCP/RTU connectivity
- **Steam Meter Integration**: Flow measurement, totalizers, energy calculation
- **Pressure Sensor Integration**: Multi-point monitoring, high-frequency sampling
- **End-to-End Workflows**: Complete analysis workflows, leak detection, trap analysis
- **Parent Coordination**: Message bus communication, task delegation, result aggregation

**Total Test Scenarios**: 150+
**Total Lines of Code**: ~8,500+
**Coverage Target**: 85%+

---

## Test Infrastructure

The test infrastructure uses Docker Compose to orchestrate:

### Core Services

1. **Mock OPC UA Server** (Port 4840)
   - Simulates steam system SCADA tags
   - Provides real-time data subscriptions
   - Supports historical data retrieval

2. **Mock Modbus Server** (Port 502)
   - Simulates field device registers
   - Steam meters, pressure sensors, temperature sensors
   - Supports Modbus TCP protocol

3. **Mock Steam Meters** (Ports 5020-5021)
   - Simulate flow measurement devices
   - Volumetric and mass flow
   - Energy calculation and totalizers

4. **Mock Pressure Sensors** (Ports 5030-5031)
   - Simulate 4-20mA analog sensors
   - High-frequency sampling
   - Absolute, gauge, differential pressure

5. **Mock Temperature Sensors** (Port 5040)
   - PT100/K-Type thermocouples
   - Temperature monitoring

6. **MQTT Broker - Mosquitto** (Port 1883)
   - Message bus for parent coordination
   - Real-time event streaming
   - QoS level support

### Database Services

7. **PostgreSQL** (Port 5432)
   - Test data persistence
   - Historical data storage
   - Query testing

8. **Redis** (Port 6379)
   - Cache testing
   - State management
   - Real-time data buffering

9. **TimescaleDB** (Port 5433)
   - Time-series data storage
   - Performance testing
   - Historical trend analysis

---

## Test Coverage

### 1. SCADA Integration (`test_scada_integration.py`)

**Test Scenarios: 40+**

- Connection Management
  - OPC UA connection establishment
  - Modbus TCP connection
  - Redundant connection setup
  - Connection retry logic
  - Failover to backup server
  - Health check mechanism

- OPC UA Operations
  - Node browsing
  - Single/multiple node reading
  - Node value writing
  - Subscriptions and callbacks
  - Historical data retrieval
  - Data type handling

- Modbus Operations
  - Holding register read/write
  - Input register reading
  - Register scaling and conversion
  - Unit ID addressing
  - Exception handling

- Real-Time Data
  - Continuous data streaming
  - High-frequency sampling
  - Data quality monitoring
  - Deadband filtering
  - Timestamp synchronization
  - Batch read optimization

- Data Buffering
  - Circular buffer operation
  - Buffer statistics
  - Time range queries

- Alarm Management
  - Alarm configuration
  - Alarm activation/deactivation
  - Callback notifications
  - Priority sorting

- Error Handling
  - Read/write error handling
  - Connection loss recovery
  - Timeout handling
  - Concurrent operations

### 2. Steam Meter Integration (`test_steam_meter_integration.py`)

**Test Scenarios: 35+**

- Connection Management
  - Modbus meter connection
  - HART protocol connection
  - Health check
  - Retry logic

- Flow Measurement
  - Volumetric flow reading
  - Mass flow reading
  - Complete measurement
  - Accuracy validation
  - Range validation
  - Zero flow detection
  - Reverse flow detection

- Totalizer Operations
  - Totalizer reading
  - Totalizer increment
  - Totalizer reset
  - Multiple totalizers

- Pressure/Temperature Compensation
  - Pressure compensation
  - Temperature compensation
  - Density calculation
  - Saturated steam compensation

- Energy Measurement
  - Energy flow calculation
  - Unit conversion
  - Energy totalizer
  - Calculation accuracy

- Steam Quality
  - Quality reading
  - Wet steam detection
  - Superheated steam detection

- Diagnostics
  - Meter diagnostics
  - Health status
  - Sensor health
  - Error detection
  - Signal strength

- Calibration
  - Calibration status
  - Calibration due check
  - Drift monitoring
  - Zero point calibration

- Multi-Meter Operations
  - Multiple meter connection
  - Parallel reading
  - Flow aggregation

### 3. Pressure Sensor Integration (`test_pressure_sensor_integration.py`)

**Test Scenarios: 30+**

- Connection Management
- Pressure Reading (gauge, absolute, differential)
- High-Frequency Sampling
- Pressure Drop Analysis
- Leak Detection Support
- Sensor Diagnostics
- Calibration Validation
- 4-20mA Signal Processing
- Continuous Monitoring
- Alarm Threshold Monitoring

### 4. End-to-End Workflow (`test_e2e_workflow.py`)

**Test Scenarios: 25+**

- Complete Analysis Workflow
  - Full system analysis
  - Real data analysis
  - Historical analysis
  - Multi-component integration

- Real-Time Monitoring
  - Continuous monitoring
  - Alarm generation
  - Dashboard updates

- Leak Detection Workflow
  - Leak detection analysis
  - Notification workflow
  - Investigation reports

- Steam Trap Analysis
  - Trap analysis
  - Failed trap identification
  - Maintenance scheduling

- Efficiency Optimization
  - Efficiency calculation
  - Optimization recommendations
  - Savings potential

- Report Generation
  - Daily reports
  - Monthly summaries
  - Custom reports

- Long-Running Workflows
  - 24-hour monitoring
  - Weekly trend analysis

- Error Recovery
  - Sensor failure recovery
  - Communication loss recovery

### 5. Parent Coordination (`test_parent_coordination.py`)

**Test Scenarios: 30+**

- Message Bus Communication
  - Connection
  - Publish/subscribe
  - Priority handling
  - QoS levels

- Task Delegation
  - Receive requests
  - Execute tasks
  - Timeout handling
  - Parallel execution

- Result Aggregation
  - Send results to parent
  - Format validation
  - Incremental streaming
  - Error reporting

- State Synchronization
  - Heartbeat mechanism
  - State snapshots
  - Configuration updates

- Error Propagation
  - Sensor error propagation
  - Analysis failure propagation
  - Critical error escalation

- Load Balancing
  - Task queue management
  - Resource monitoring
  - Backpressure handling

---

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Git

### 1. Clone Repository

```bash
cd GreenLang_2030/agent_foundation/agents/GL-003
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r tests/integration/requirements-test.txt
```

### 3. Run Tests with Docker

```bash
cd tests/integration
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### 4. View Results

Test results will be in `test-results/`:
- `integration-results.xml` - JUnit XML results
- `coverage.xml` - Coverage report
- `htmlcov/` - HTML coverage report

---

## Running Tests

### Option 1: Docker Compose (Recommended)

Run all tests with complete infrastructure:

```bash
cd tests/integration
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### Option 2: Local Execution

Start mock servers:

```bash
python mock_servers.py
```

In another terminal, run tests:

```bash
pytest tests/integration/ -v --tb=short
```

### Option 3: Specific Test Files

Run specific test file:

```bash
pytest tests/integration/test_scada_integration.py -v
```

Run specific test class:

```bash
pytest tests/integration/test_scada_integration.py::TestSCADAConnectionManagement -v
```

Run specific test:

```bash
pytest tests/integration/test_scada_integration.py::TestSCADAConnectionManagement::test_opc_ua_connection_establishment -v
```

### Option 4: With Coverage

```bash
pytest tests/integration/ --cov=integrations --cov=calculators --cov=steam_system_orchestrator --cov-report=html
```

### Option 5: Parallel Execution

```bash
pytest tests/integration/ -n 4  # Run with 4 workers
```

### Option 6: Markers

Run only SCADA tests:

```bash
pytest tests/integration/ -m scada
```

Run only slow tests:

```bash
pytest tests/integration/ -m slow
```

Skip slow tests:

```bash
pytest tests/integration/ -m "not slow"
```

---

## Test Structure

```
tests/integration/
├── __init__.py                          # Package initialization
├── conftest.py                          # Shared fixtures (655 lines)
├── mock_servers.py                      # Mock server implementations (711 lines)
├── docker-compose.test.yml              # Docker infrastructure
├── mosquitto.conf                       # MQTT broker config
├── requirements-test.txt                # Test dependencies
├── test_scada_integration.py            # SCADA tests (1058 lines)
├── test_steam_meter_integration.py      # Steam meter tests (876 lines)
├── test_pressure_sensor_integration.py  # Pressure sensor tests (400+ lines)
├── test_e2e_workflow.py                 # E2E workflow tests (600+ lines)
├── test_parent_coordination.py          # Coordination tests (600+ lines)
├── README.md                            # This file
└── test-results/                        # Generated test results
```

---

## Mock Services

### MockOPCUAServer

Simulates OPC UA SCADA server with steam system nodes:

```python
# Nodes available:
ns=2;s=STEAM.HEADER.PRESSURE
ns=2;s=STEAM.HEADER.TEMPERATURE
ns=2;s=STEAM.HEADER.FLOW
ns=2;s=STEAM.HEADER.QUALITY
ns=2;s=CONDENSATE.RETURN.FLOW
ns=2;s=TRAP.ST001.STATUS
... (and more)
```

### MockModbusServer

Simulates Modbus TCP server with register map:

```python
# Register addresses:
40001 - Steam header pressure (bar * 10)
40002 - Steam header temperature (°C * 10)
40003 - Flow rate (t/hr * 10)
40004 - Steam quality (% * 100)
40021-40030 - Pressure sensors
40041-40050 - Temperature sensors
40061-40100 - Steam meters
40101-40120 - Steam trap status
```

### MockSteamMeterServer

Simulates steam flow meter:

- Volumetric flow (m3/hr)
- Mass flow (kg/hr)
- Totalizer (m3)
- Energy flow (kW)
- Pressure and temperature

### MockPressureSensorServer

Simulates pressure sensor:

- Gauge/absolute/differential pressure
- High-frequency sampling (up to 100 Hz)
- 4-20mA signal simulation
- Diagnostics

---

## Configuration

### Environment Variables

```bash
# SCADA
SCADA_HOST=localhost
SCADA_PORT=4840

# Modbus
MODBUS_HOST=localhost
MODBUS_PORT=502

# Steam Meters
STEAM_METER_1_HOST=localhost
STEAM_METER_1_PORT=5020

# Pressure Sensors
PRESSURE_SENSOR_1_HOST=localhost
PRESSURE_SENSOR_1_PORT=5030

# MQTT
MQTT_HOST=localhost
MQTT_PORT=1883

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=gl003_test
POSTGRES_USER=gl003_user
POSTGRES_PASSWORD=test_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Test Configuration
TEST_TIMEOUT=300
LOG_LEVEL=INFO
```

### Pytest Configuration

`pytest.ini` settings:

```ini
[pytest]
asyncio_mode = auto
testpaths = tests/integration
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    integration: Integration tests
    e2e: End-to-end tests
    scada: SCADA integration tests
    steam_meter: Steam meter tests
    pressure_sensor: Pressure sensor tests
    slow: Slow running tests
```

---

## CI/CD Integration

### GitHub Actions

```yaml
name: GL-003 Integration Tests

on:
  push:
    paths:
      - 'GreenLang_2030/agent_foundation/agents/GL-003/**'
  pull_request:

jobs:
  integration-tests:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Run Integration Tests
        run: |
          cd GreenLang_2030/agent_foundation/agents/GL-003/tests/integration
          docker-compose -f docker-compose.test.yml up --abort-on-container-exit

      - name: Upload Test Results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: test-results
          path: test-results/

      - name: Upload Coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./test-results/coverage.xml
```

### GitLab CI

```yaml
integration_tests:
  stage: test
  image: docker:latest
  services:
    - docker:dind
  script:
    - cd GreenLang_2030/agent_foundation/agents/GL-003/tests/integration
    - docker-compose -f docker-compose.test.yml up --abort-on-container-exit
  artifacts:
    reports:
      junit: test-results/integration-results.xml
      coverage_report:
        coverage_format: cobertura
        path: test-results/coverage.xml
```

---

## Troubleshooting

### Common Issues

#### 1. Docker Services Not Starting

**Problem**: Mock servers fail to start

**Solution**:
```bash
# Check Docker logs
docker-compose -f docker-compose.test.yml logs

# Restart services
docker-compose -f docker-compose.test.yml down
docker-compose -f docker-compose.test.yml up --force-recreate
```

#### 2. Port Conflicts

**Problem**: Port already in use

**Solution**:
```bash
# Check port usage
netstat -tulpn | grep <PORT>

# Kill process or change port in docker-compose.test.yml
```

#### 3. Test Timeouts

**Problem**: Tests timing out

**Solution**:
```bash
# Increase timeout in pytest.ini or environment
export TEST_TIMEOUT=600

# Or run with longer timeout
pytest tests/integration/ --timeout=600
```

#### 4. Connection Failures

**Problem**: Cannot connect to mock servers

**Solution**:
```bash
# Verify services are healthy
docker-compose -f docker-compose.test.yml ps

# Check network connectivity
docker network inspect gl003-test-network
```

#### 5. Database Connection Issues

**Problem**: Cannot connect to PostgreSQL

**Solution**:
```bash
# Check database is ready
docker-compose -f docker-compose.test.yml exec postgres pg_isready -U gl003_user

# Recreate database
docker-compose -f docker-compose.test.yml down -v
docker-compose -f docker-compose.test.yml up postgres
```

---

## Best Practices

### 1. Writing Integration Tests

- **Use fixtures**: Leverage conftest.py fixtures
- **Test isolation**: Each test should be independent
- **Cleanup**: Always cleanup resources (use yield in fixtures)
- **Async/await**: Use async properly for concurrent operations
- **Assertions**: Use descriptive assertions with messages

### 2. Mock Server Usage

- **Realistic data**: Mock servers should simulate realistic behavior
- **Variation**: Add random variation to simulated values
- **Errors**: Include error scenarios in mocks

### 3. Performance

- **Parallel execution**: Use pytest-xdist for parallel tests
- **Mark slow tests**: Use `@pytest.mark.slow` for long tests
- **Optimize fixtures**: Use appropriate fixture scopes

### 4. Debugging

- **Verbose output**: Run with `-v` or `-vv`
- **Print debugging**: Use `-s` to see print statements
- **PDB**: Use `--pdb` to drop into debugger on failure
- **Log level**: Set `LOG_LEVEL=DEBUG` for detailed logs

### 5. Coverage

- **Target 85%+**: Aim for high coverage
- **Cover edge cases**: Test error paths and boundaries
- **Integration focus**: Test component interactions

---

## Maintenance

### Updating Mock Servers

When adding new features:

1. Update `mock_servers.py` with new simulated behaviors
2. Add corresponding test cases
3. Update fixtures in `conftest.py` if needed
4. Update this README

### Adding New Test Files

1. Create test file: `test_<feature>_integration.py`
2. Import fixtures from `conftest.py`
3. Add markers: `@pytest.mark.integration`
4. Update docker-compose if new services needed
5. Update README with new test coverage

### Keeping Tests Current

- Review tests quarterly
- Update for API changes
- Refactor duplicated code
- Remove obsolete tests
- Update documentation

---

## Support

For issues or questions:

- **Documentation**: See `ARCHITECTURE.md`
- **Issues**: GitHub Issues
- **Team**: GreenLang Test Engineering Team

---

## License

Copyright (c) 2025 GreenLang. All rights reserved.

---

**Last Updated**: 2025-01-17
**Version**: 1.0.0
**Maintainer**: GreenLang Test Engineering Team
