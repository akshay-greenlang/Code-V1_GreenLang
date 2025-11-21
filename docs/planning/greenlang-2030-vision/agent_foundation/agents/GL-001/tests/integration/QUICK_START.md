# GL-001 Integration Tests - Quick Start Guide

**5-Minute Setup Guide for GL-001 ProcessHeatOrchestrator Integration Tests**

---

## Prerequisites Check

```bash
# 1. Check Docker
docker --version
# Required: Docker 20.10+

# 2. Check Python
python --version
# Required: Python 3.10+

# 3. Check current directory
cd GreenLang_2030/agent_foundation/agents/GL-001
pwd
```

---

## Installation (2 minutes)

### Step 1: Install Test Dependencies

```bash
# Install all test requirements
pip install -r tests/integration/requirements-test.txt

# Or install core dependencies only
pip install pytest pytest-asyncio pytest-cov aiohttp psycopg redis
```

### Step 2: Start Docker Infrastructure

```bash
# Start all test services (PostgreSQL, Redis, MQTT, Mock servers)
docker-compose -f tests/integration/docker-compose.test.yml up -d

# Wait for services to be ready (30 seconds)
sleep 30

# Check service health
docker-compose -f tests/integration/docker-compose.test.yml ps
```

**Expected Output**:
```
NAME                    STATUS
gl001-test-postgres     Up (healthy)
gl001-test-redis        Up (healthy)
gl001-test-mosquitto    Up (healthy)
gl001-mock-scada-plant1 Up (healthy)
gl001-mock-sap          Up (healthy)
gl001-mock-gl002        Up (healthy)
...
```

---

## Running Tests (1 minute)

### Option 1: Run All Integration Tests

```bash
# Run all integration tests with verbose output
pytest tests/integration/ -v

# Expected: 40+ tests, ~2-3 minutes execution
```

### Option 2: Run Specific Test Categories

```bash
# E2E workflow tests only (12 tests, ~45s)
pytest tests/integration/test_e2e_workflow.py -v

# SCADA integration tests (18 tests, ~90s)
pytest tests/integration/test_scada_integration.py -v

# Agent coordination tests (14 tests, ~40s)
pytest tests/integration/test_agent_coordination.py -v
```

### Option 3: Run with Coverage

```bash
# Run with coverage report
pytest tests/integration/ -v --cov=. --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Option 4: Run Parallel (Faster)

```bash
# Run tests in parallel (uses all CPU cores)
pytest tests/integration/ -v -n auto

# Expected: 2x-4x faster execution
```

---

## Verify Success

### Check Test Output

Look for:
```
============================= test session starts =============================
collected 44 items

tests/integration/test_e2e_workflow.py::test_full_plant_heat_optimization_workflow PASSED [ 2%]
tests/integration/test_e2e_workflow.py::test_multi_agent_coordination_workflow PASSED [ 4%]
...

============================= 44 passed in 180.23s =============================
```

### Check Coverage

```bash
# View coverage summary
pytest tests/integration/ --cov=. --cov-report=term

# Expected output:
# Name                              Stmts   Miss  Cover
# -----------------------------------------------------
# process_heat_orchestrator.py        320     42    87%
# tools.py                            245     28    89%
# config.py                            85      8    91%
# -----------------------------------------------------
# TOTAL                               650     78    88%
```

---

## Cleanup (30 seconds)

```bash
# Stop all Docker services
docker-compose -f tests/integration/docker-compose.test.yml down

# Remove volumes (optional - resets databases)
docker-compose -f tests/integration/docker-compose.test.yml down -v

# Clean up Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null
```

---

## Common Commands

### Test Execution

```bash
# Run specific test
pytest tests/integration/test_e2e_workflow.py::test_full_plant_heat_optimization_workflow -v

# Run tests matching pattern
pytest tests/integration/ -v -k "scada"

# Run with markers
pytest tests/integration/ -v -m e2e
pytest tests/integration/ -v -m performance

# Stop on first failure
pytest tests/integration/ -v -x

# Show print statements
pytest tests/integration/ -v -s

# Debug mode (drop into pdb on failure)
pytest tests/integration/ -v --pdb
```

### Docker Management

```bash
# View logs
docker-compose -f tests/integration/docker-compose.test.yml logs -f

# Restart specific service
docker-compose -f tests/integration/docker-compose.test.yml restart postgres

# Check service status
docker-compose -f tests/integration/docker-compose.test.yml ps

# Execute command in container
docker exec -it gl001-test-postgres psql -U postgres
```

---

## Troubleshooting

### Issue: Tests Fail with Connection Error

**Solution**: Ensure Docker services are running
```bash
docker-compose -f tests/integration/docker-compose.test.yml ps
docker-compose -f tests/integration/docker-compose.test.yml restart
```

### Issue: Port Already in Use

**Solution**: Stop conflicting services or change ports
```bash
# Find process using port
netstat -an | grep 5432

# Stop conflicting service
# Or edit docker-compose.test.yml to use different ports
```

### Issue: Slow Test Execution

**Solution**: Run in parallel
```bash
pytest tests/integration/ -v -n auto
```

### Issue: Import Errors

**Solution**: Install dependencies
```bash
pip install -r tests/integration/requirements-test.txt
```

---

## Test Markers

Use markers to run specific test categories:

```bash
# E2E workflow tests
pytest tests/integration/ -v -m e2e

# SCADA integration tests
pytest tests/integration/ -v -m scada

# ERP integration tests
pytest tests/integration/ -v -m erp

# Agent coordination tests
pytest tests/integration/ -v -m coordination

# Multi-plant tests
pytest tests/integration/ -v -m multi_plant

# Performance tests
pytest tests/integration/ -v -m performance

# Slow-running tests
pytest tests/integration/ -v -m slow

# Docker-required tests
pytest tests/integration/ -v --docker
```

---

## Environment Variables

Customize test configuration:

```bash
# Database
export TEST_POSTGRES_HOST=localhost
export TEST_POSTGRES_PORT=5432
export TEST_REDIS_HOST=localhost
export TEST_REDIS_PORT=6379

# SCADA
export TEST_SCADA_OPC_PORT=4840
export TEST_SCADA_MODBUS_PORT=502

# Multi-plant
export TEST_MULTI_PLANT_COUNT=3

# Run tests
pytest tests/integration/ -v
```

---

## CI/CD Integration

### GitHub Actions (Automated)

Tests run automatically on:
- Pull requests to `main`
- Nightly builds (2 AM UTC)
- Manual workflow dispatch

### Manual CI Run

```bash
# Simulate CI environment
docker-compose -f tests/integration/docker-compose.test.yml up -d
pytest tests/integration/ -v --cov=. --cov-report=xml
docker-compose -f tests/integration/docker-compose.test.yml down -v
```

---

## Performance Benchmarks

Expected performance (reference hardware: 4-core CPU, 16GB RAM):

| Test Category | Tests | Duration | Avg/Test |
|---------------|-------|----------|----------|
| E2E Workflow | 12 | 45s | 3.75s |
| SCADA Integration | 18 | 90s | 5.0s |
| Agent Coordination | 14 | 40s | 2.86s |
| Performance | 8 | 120s | 15.0s |
| **Total** | **52+** | **~5min** | **~6s** |

---

## Next Steps

After successful test run:

1. **Review Coverage Report**
   ```bash
   pytest tests/integration/ --cov=. --cov-report=html
   open htmlcov/index.html
   ```

2. **Check Test Metrics**
   - Coverage: Should be >85%
   - Pass Rate: Should be 100%
   - Performance: Should meet targets

3. **Add Custom Tests**
   - Use test templates in README.md
   - Follow GL-TestEngineer patterns
   - Add to appropriate test file

4. **Configure CI/CD**
   - Review `.github/workflows/gl-001-integration-tests.yml`
   - Customize for your environment
   - Enable automated testing

---

## Resources

- **Full Documentation**: `tests/integration/README.md`
- **Test Summary**: `tests/integration/INTEGRATION_TEST_SUMMARY.md`
- **Mock Servers**: `tests/integration/mock_servers.py`
- **Fixtures**: `tests/integration/conftest.py`

---

## Support

For issues or questions:

1. Check test logs: `pytest tests/integration/ -v -s`
2. Check Docker logs: `docker-compose logs`
3. Review documentation: `tests/integration/README.md`
4. Contact: GreenLang Test Engineering Team

---

**Quick Start Complete!** 🎉

You now have a fully functional GL-001 integration test suite running.

**Next Command**:
```bash
pytest tests/integration/ -v
```

**Expected Result**: All tests passing ✅
