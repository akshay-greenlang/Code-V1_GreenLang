# Phase 7: Testing & Documentation Gap Analysis

## GreenLang Process Heat Agents - Testing Engineering Assessment

**Document ID:** GL-PHASE7-GAP-001
**Version:** 1.0.0
**Created:** December 5, 2025
**Author:** GL-TestEngineer
**Status:** Phase 7 is 40% Complete (8/20 tasks)

---

## Executive Summary

This document provides a comprehensive gap analysis for Phase 7 (Integration Testing & Documentation) of the GreenLang Process Heat Agents engineering initiative. The analysis covers 12 remaining tasks across contract testing, chaos engineering, security testing, compliance validation, and documentation deliverables.

### Current Status

| Category | Completed | Remaining | Completion % |
|----------|-----------|-----------|--------------|
| Integration Testing (7.1) | 4 | 4 | 50% |
| Documentation (7.2) | 4 | 8 | 33% |
| **Total Phase 7** | **8** | **12** | **40%** |

### Priority Ranking (Critical to Launch)

| Priority | Task | Impact | Effort | Recommendation |
|----------|------|--------|--------|----------------|
| P0 - Critical | TASK-248: Compliance Validation Tests | Regulatory | High | Implement first |
| P0 - Critical | TASK-252: API Reference Documentation | Developer adoption | Medium | Parallel with P0 |
| P1 - High | TASK-247: Security Penetration Tests | Security audit | High | Week 14 |
| P1 - High | TASK-253: Operator Manuals | Production ops | Medium | Week 14-15 |
| P2 - Medium | TASK-243: Contract Testing | API stability | Medium | Week 15 |
| P2 - Medium | TASK-246: Chaos Testing | Resilience | High | Week 15 |
| P3 - Normal | TASK-254-260: Remaining docs | User experience | Medium | Week 15-16 |

---

## Section 1: Existing Test Infrastructure Analysis

### 1.1 Current Test Coverage

The GreenLang codebase has substantial existing test infrastructure:

```
tests/
├── agents/                    # Agent-specific tests (43+ test files)
│   ├── phase3/               # Phase 3 integration tests
│   ├── phase4/               # Phase 4 integration tests
│   ├── phase5/               # Critical path compliance tests
│   └── tools/                # Tool-specific tests
├── e2e/                      # End-to-end workflow tests
│   └── test_complete_workflows.py  # 25 E2E test cases
├── integration/              # Integration test suite
│   ├── agents/               # Agent integration tests
│   ├── api/                  # API integration tests
│   ├── protocols/            # Protocol integration tests (OPC-UA, MQTT, Kafka, Modbus)
│   └── test_agent_pipeline_comprehensive.py  # 50 integration tests
├── performance/              # Performance test suite
│   ├── test_load_comprehensive.py        # 15 load tests
│   └── test_performance_benchmarks.py    # 40 performance tests
├── unit/                     # Unit test suite
│   ├── test_gl001_carbon_agent.py       # 50 unit tests
│   ├── test_gl002_cbam_agent.py
│   └── ... (7 agent test files)
├── golden/                   # Golden file tests (regulatory accuracy)
│   ├── test_eudr_golden.py
│   ├── test_cbam_golden.py
│   └── ... (8 golden test files)
├── compliance/               # [EMPTY - GAP]
├── fixtures/                 # Test fixtures and generators
└── mocks/                    # Mock services
```

### 1.2 Test Categories Summary

| Test Category | Files | Test Cases | Coverage Target | Actual |
|--------------|-------|------------|-----------------|--------|
| Unit Tests | 15+ | 400+ | 85% | ~75% |
| Integration Tests | 10+ | 150+ | 80% | ~70% |
| E2E Tests | 1 | 25 | 100% workflows | 4/4 |
| Performance Tests | 2 | 55 | <10ms p95 | Passing |
| Golden Tests | 8 | 50+ | 100% accuracy | Passing |
| Compliance Tests | 0 | 0 | 100% | **GAP** |
| Contract Tests | 0 | 0 | 90% | **GAP** |
| Chaos Tests | 0 | 0 | All scenarios | **GAP** |
| Security Tests | 0 | 0 | OWASP Top 10 | **GAP** |

### 1.3 Strengths of Current Test Infrastructure

1. **Determinism Testing** - Comprehensive critical path compliance tests in `tests/agents/phase5/`
2. **Performance Benchmarks** - Well-defined latency/throughput targets with automated validation
3. **Golden File Testing** - Regulatory calculation accuracy validation
4. **Protocol Integration** - OPC-UA, MQTT, Kafka, Modbus protocol tests
5. **Provenance Tracking** - SHA-256 hash verification for audit trails
6. **Async Testing** - pytest-asyncio support for async agent testing

---

## Section 2: Gap Analysis - Integration Testing (7.1)

### 2.1 TASK-243: Contract Testing Implementation

**Status:** NOT IMPLEMENTED
**Priority:** P2 - Medium
**Estimated Effort:** 3-4 days

#### Gap Description

Contract testing verifies that API consumers and providers adhere to agreed-upon contracts. This is critical for:
- Agent-to-Agent communication contracts
- ERP connector API contracts
- External API integration contracts (EU CBAM, EUDR systems)

#### Implementation Plan

```python
# tests/contract/test_agent_contracts.py
"""
Contract Testing with Pact Python

Validates provider-consumer contracts for GreenLang agents.
Uses Pact for consumer-driven contract testing.
"""

import pytest
from pact import Consumer, Provider, Term, Like, EachLike
from datetime import datetime

# Consumer: Orchestrator Agent
# Provider: Calculation Agents (GL-001 to GL-020)

class TestAgentContracts:
    """Contract tests for agent communication."""

    @pytest.fixture(scope="module")
    def pact(self):
        """Create Pact contract."""
        pact = Consumer('OrchestratorAgent').has_pact_with(
            Provider('CarbonEmissionsAgent'),
            host_name='localhost',
            port=1234,
            pact_dir='./pacts'
        )
        pact.start_service()
        yield pact
        pact.stop_service()

    def test_carbon_emissions_calculation_contract(self, pact):
        """Contract: Carbon emissions calculation request/response."""

        # Expected request format
        expected_request = {
            "fuel_type": "natural_gas",
            "quantity": Like(1000.0),
            "unit": Term(r"^(MJ|kWh|m3|L|therms)$", "MJ"),
            "region": Term(r"^[A-Z]{2}$", "US"),
            "scope": Term(r"^(SCOPE_1|SCOPE_2|SCOPE_3)$", "SCOPE_1")
        }

        # Expected response format
        expected_response = {
            "success": True,
            "emissions_kgco2e": Like(56.1),
            "emission_factor": Like(0.0561),
            "emission_factor_source": Like("EPA"),
            "emission_factor_unit": Like("kgCO2e/MJ"),
            "provenance_hash": Term(r"^[a-f0-9]{64}$", "abc123..."),
            "calculated_at": Like(datetime.now().isoformat())
        }

        # Define contract
        (pact
         .given('Carbon emissions agent is available')
         .upon_receiving('A request to calculate emissions')
         .with_request('POST', '/v1/agents/carbon/calculate')
         .will_respond_with(200, body=expected_response))

        # Verify contract
        with pact:
            # Make actual API call
            response = requests.post(
                f"{pact.uri}/v1/agents/carbon/calculate",
                json=expected_request
            )
            assert response.status_code == 200
```

#### Contract Categories to Implement

| Contract | Consumer | Provider | Priority |
|----------|----------|----------|----------|
| Carbon Calculation | Orchestrator | GL-001 | P1 |
| CBAM Compliance | Orchestrator | GL-002 | P1 |
| Steam Tables | GL-003 | Thermodynamic Library | P1 |
| ERP Data Fetch | All Agents | ERP Connector | P0 |
| Event Publishing | All Agents | Kafka/Event Bus | P1 |
| Protocol Messages | All Agents | OPC-UA Server | P2 |

#### Technology Selection

**Recommended: Pact Python**
- Consumer-driven contract testing
- Contract broker for versioning
- CI/CD integration
- Python native support

```yaml
# .github/workflows/contract-tests.yml
name: Contract Tests
on: [push, pull_request]

jobs:
  contract-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Pact
        run: pip install pact-python
      - name: Run Contract Tests
        run: pytest tests/contract/ -v
      - name: Publish Contracts
        run: |
          pact-broker publish ./pacts \
            --broker-base-url ${{ secrets.PACT_BROKER_URL }} \
            --consumer-app-version ${{ github.sha }}
```

---

### 2.2 TASK-246: Chaos Testing Framework

**Status:** NOT IMPLEMENTED
**Priority:** P2 - Medium
**Estimated Effort:** 5-7 days

#### Gap Description

Chaos testing validates system resilience under failure conditions. Required for:
- Kubernetes pod failures
- Network partitions
- Database connection drops
- Message queue failures
- Downstream service unavailability

#### Implementation Plan

```yaml
# infrastructure/chaos/litmus-experiments/agent-pod-kill.yaml
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosExperiment
metadata:
  name: gl-agent-pod-kill
  labels:
    app: greenlang
    component: chaos
spec:
  definition:
    scope: Namespaced
    permissions:
      - apiGroups: [""]
        resources: ["pods"]
        verbs: ["create", "delete", "get", "list", "patch", "update"]
    image: litmuschaos/go-runner:latest
    imagePullPolicy: Always
    args:
      - -c
      - ./experiments -name pod-delete
    command:
      - /bin/bash
    env:
      - name: TOTAL_CHAOS_DURATION
        value: "30"
      - name: CHAOS_INTERVAL
        value: "10"
      - name: PODS_AFFECTED_PERC
        value: "50"
    labels:
      name: pod-delete
      app.kubernetes.io/part-of: litmus
---
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosEngine
metadata:
  name: gl-agent-chaos
spec:
  appinfo:
    appns: greenlang
    applabel: "app=gl-orchestrator"
    appkind: deployment
  chaosServiceAccount: litmus-admin
  experiments:
    - name: gl-agent-pod-kill
      spec:
        components:
          env:
            - name: TARGET_CONTAINER
              value: "gl-orchestrator"
            - name: TOTAL_CHAOS_DURATION
              value: "60"
```

#### Chaos Test Scenarios

| Scenario | Tool | Target | Expected Behavior |
|----------|------|--------|-------------------|
| Pod Kill | Litmus | Agent pods | Auto-restart, no data loss |
| Network Partition | Chaos Mesh | Agent-DB | Circuit breaker activation |
| CPU Stress | Litmus | Compute pods | Graceful degradation |
| Memory Pressure | Litmus | All pods | OOM handling, restart |
| Kafka Failure | Custom | Message bus | Message retry, DLQ |
| Database Timeout | Toxiproxy | PostgreSQL | Retry with backoff |
| API Latency | Chaos Mesh | External APIs | Timeout handling |

#### Chaos Testing Framework Structure

```
tests/chaos/
├── __init__.py
├── conftest.py                    # Chaos test fixtures
├── experiments/
│   ├── pod_failure.py            # Pod kill experiments
│   ├── network_partition.py      # Network chaos
│   ├── resource_stress.py        # CPU/Memory stress
│   └── dependency_failure.py     # Database/Queue failures
├── scenarios/
│   ├── test_agent_resilience.py  # Agent failure scenarios
│   ├── test_pipeline_recovery.py # Pipeline chaos
│   └── test_data_consistency.py  # Data integrity under chaos
└── reports/
    └── chaos_report_template.md
```

#### Chaos Test Implementation

```python
# tests/chaos/scenarios/test_agent_resilience.py
"""
Chaos Engineering Tests for GreenLang Agents

Validates system resilience under various failure conditions.
Requires Kubernetes cluster with Litmus Chaos installed.
"""

import pytest
import asyncio
import kubernetes
from datetime import datetime, timedelta
from typing import Dict, Any

class TestAgentResilience:
    """Chaos tests for agent resilience."""

    @pytest.fixture
    def k8s_client(self):
        """Initialize Kubernetes client."""
        kubernetes.config.load_incluster_config()
        return kubernetes.client.CoreV1Api()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_agent_pod_kill_recovery(self, k8s_client):
        """CHAOS-001: Test agent recovers from pod kill."""
        namespace = "greenlang"
        deployment = "gl-orchestrator"

        # Get current pod count
        pods_before = k8s_client.list_namespaced_pod(
            namespace,
            label_selector=f"app={deployment}"
        )
        initial_count = len(pods_before.items)

        # Kill a pod
        pod_to_kill = pods_before.items[0].metadata.name
        k8s_client.delete_namespaced_pod(pod_to_kill, namespace)

        # Wait for recovery
        await asyncio.sleep(30)

        # Verify recovery
        pods_after = k8s_client.list_namespaced_pod(
            namespace,
            label_selector=f"app={deployment}"
        )

        assert len(pods_after.items) == initial_count
        assert all(p.status.phase == "Running" for p in pods_after.items)

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_calculation_during_chaos(self, k8s_client, mock_agent):
        """CHAOS-002: Test calculations continue during chaos."""
        # Start continuous calculations
        calculation_results = []
        errors = []

        async def continuous_calculation():
            for i in range(100):
                try:
                    result = await mock_agent.process({"quantity": 1000})
                    calculation_results.append(result)
                except Exception as e:
                    errors.append(str(e))
                await asyncio.sleep(0.1)

        # Run calculations while inducing chaos
        calc_task = asyncio.create_task(continuous_calculation())

        # Induce chaos after 2 seconds
        await asyncio.sleep(2)
        # (Chaos injection would happen here via Litmus)

        await calc_task

        # Verify high success rate (>95%)
        success_rate = len(calculation_results) / (len(calculation_results) + len(errors))
        assert success_rate > 0.95, f"Success rate {success_rate} below 95%"

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self):
        """CHAOS-003: Test circuit breaker activates under load."""
        from greenlang.infrastructure.resilience.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            half_open_requests=3
        )

        # Simulate failures
        for _ in range(10):
            try:
                async with breaker:
                    raise ConnectionError("Simulated failure")
            except ConnectionError:
                pass

        # Circuit should be open
        assert breaker.state == "OPEN"

        # Requests should be rejected
        with pytest.raises(Exception) as exc_info:
            async with breaker:
                pass
        assert "Circuit breaker is open" in str(exc_info.value)
```

---

### 2.3 TASK-247: Security Penetration Tests

**Status:** NOT IMPLEMENTED
**Priority:** P1 - High
**Estimated Effort:** 5-7 days

#### Gap Description

Security penetration testing is required for:
- OWASP Top 10 vulnerability assessment
- API security testing
- Authentication/Authorization bypass attempts
- Data leakage prevention validation
- Input validation and injection testing

#### OWASP Top 10 Test Coverage

| OWASP Category | Test Focus | Priority |
|----------------|------------|----------|
| A01: Broken Access Control | RBAC bypass, privilege escalation | P0 |
| A02: Cryptographic Failures | TLS, encryption at rest | P0 |
| A03: Injection | SQL, NoSQL, Command injection | P0 |
| A04: Insecure Design | Business logic flaws | P1 |
| A05: Security Misconfiguration | Default credentials, headers | P1 |
| A06: Vulnerable Components | Dependency scanning | P1 |
| A07: Authentication Failures | Session hijacking, brute force | P0 |
| A08: Software Integrity | Supply chain attacks | P2 |
| A09: Logging Failures | Insufficient logging | P2 |
| A10: SSRF | Server-side request forgery | P1 |

#### Security Test Implementation

```python
# tests/security/test_owasp_top10.py
"""
OWASP Top 10 Security Tests for GreenLang API

Validates API security against OWASP Top 10 vulnerabilities.
Run with: pytest tests/security/ -v --tb=short -m security
"""

import pytest
import requests
import jwt
from datetime import datetime, timedelta

class TestBrokenAccessControl:
    """A01:2021 - Broken Access Control Tests."""

    @pytest.mark.security
    def test_horizontal_privilege_escalation(self, api_client, user_token, other_user_id):
        """SEC-A01-001: Test horizontal privilege escalation prevention."""
        # Try to access another user's emissions data
        response = api_client.get(
            f"/v1/users/{other_user_id}/emissions",
            headers={"Authorization": f"Bearer {user_token}"}
        )

        # Should be forbidden
        assert response.status_code == 403

    @pytest.mark.security
    def test_vertical_privilege_escalation(self, api_client, user_token):
        """SEC-A01-002: Test vertical privilege escalation prevention."""
        # Try to access admin endpoint with regular user token
        response = api_client.post(
            "/v1/admin/users/create",
            headers={"Authorization": f"Bearer {user_token}"},
            json={"username": "hacker", "role": "admin"}
        )

        # Should be forbidden
        assert response.status_code == 403

    @pytest.mark.security
    def test_direct_object_reference(self, api_client, user_token):
        """SEC-A01-003: Test insecure direct object references."""
        # Try to access sequential IDs
        for i in range(1, 100):
            response = api_client.get(
                f"/v1/calculations/{i}",
                headers={"Authorization": f"Bearer {user_token}"}
            )
            # Should only return user's own calculations or 404
            assert response.status_code in [200, 403, 404]
            if response.status_code == 200:
                data = response.json()
                assert data.get("user_id") == "current_user_id"


class TestCryptographicFailures:
    """A02:2021 - Cryptographic Failures Tests."""

    @pytest.mark.security
    def test_tls_version(self, api_url):
        """SEC-A02-001: Test TLS 1.3 enforcement."""
        import ssl
        import socket

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_3

        with socket.create_connection((api_url, 443)) as sock:
            with context.wrap_socket(sock, server_hostname=api_url) as ssock:
                assert ssock.version() == "TLSv1.3"

    @pytest.mark.security
    def test_sensitive_data_not_in_url(self, api_client, user_token):
        """SEC-A02-002: Test sensitive data not in URL parameters."""
        # API keys, passwords, tokens should not be in URLs
        response = api_client.post(
            "/v1/auth/login",
            json={"username": "test", "password": "test123"}
        )

        # Check response doesn't leak credentials
        assert "password" not in response.text
        assert "api_key" not in response.url


class TestInjection:
    """A03:2021 - Injection Tests."""

    @pytest.mark.security
    @pytest.mark.parametrize("payload", [
        "'; DROP TABLE emissions; --",
        "\" OR \"1\"=\"1",
        "1 OR 1=1",
        "'; EXEC xp_cmdshell('dir'); --",
        "${7*7}",
        "{{7*7}}",
    ])
    def test_sql_injection(self, api_client, user_token, payload):
        """SEC-A03-001: Test SQL injection prevention."""
        response = api_client.get(
            f"/v1/emissions?facility_id={payload}",
            headers={"Authorization": f"Bearer {user_token}"}
        )

        # Should not return database errors
        assert "SQL" not in response.text
        assert "syntax" not in response.text.lower()
        assert response.status_code in [200, 400, 422]

    @pytest.mark.security
    @pytest.mark.parametrize("payload", [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "javascript:alert('XSS')",
        "<svg onload=alert('XSS')>",
    ])
    def test_xss_prevention(self, api_client, user_token, payload):
        """SEC-A03-002: Test XSS prevention."""
        response = api_client.post(
            "/v1/facilities",
            headers={"Authorization": f"Bearer {user_token}"},
            json={"name": payload, "location": "Test"}
        )

        # Payload should be escaped in response
        if response.status_code == 201:
            data = response.json()
            assert "<script>" not in data.get("name", "")


class TestAuthenticationFailures:
    """A07:2021 - Identification and Authentication Failures."""

    @pytest.mark.security
    def test_brute_force_protection(self, api_client):
        """SEC-A07-001: Test brute force protection."""
        # Attempt multiple failed logins
        for i in range(20):
            response = api_client.post(
                "/v1/auth/login",
                json={"username": "test", "password": f"wrong{i}"}
            )

        # Should be rate limited
        assert response.status_code == 429  # Too Many Requests

    @pytest.mark.security
    def test_jwt_algorithm_confusion(self, api_client):
        """SEC-A07-002: Test JWT algorithm confusion prevention."""
        # Create token with 'none' algorithm
        malicious_token = jwt.encode(
            {"sub": "admin", "role": "admin"},
            key="",
            algorithm="none"
        )

        response = api_client.get(
            "/v1/admin/dashboard",
            headers={"Authorization": f"Bearer {malicious_token}"}
        )

        # Should reject 'none' algorithm
        assert response.status_code == 401

    @pytest.mark.security
    def test_session_fixation(self, api_client):
        """SEC-A07-003: Test session fixation prevention."""
        # Get initial session
        response1 = api_client.post(
            "/v1/auth/login",
            json={"username": "test", "password": "test123"}
        )
        token1 = response1.json().get("access_token")

        # Login again
        response2 = api_client.post(
            "/v1/auth/login",
            json={"username": "test", "password": "test123"}
        )
        token2 = response2.json().get("access_token")

        # Tokens should be different (new session on login)
        assert token1 != token2
```

#### Security Testing Tools

| Tool | Purpose | Integration |
|------|---------|-------------|
| OWASP ZAP | Dynamic application security testing | CI/CD |
| Bandit | Python static analysis | Pre-commit |
| Safety | Dependency vulnerability scanning | CI/CD |
| Trivy | Container scanning | Docker build |
| SonarQube | Code quality and security | PR checks |

---

### 2.4 TASK-248: Compliance Validation Tests

**Status:** NOT IMPLEMENTED
**Priority:** P0 - Critical
**Estimated Effort:** 7-10 days

#### Gap Description

Compliance validation tests ensure GreenLang calculations meet regulatory requirements:
- EPA emission factor accuracy
- GHG Protocol methodology compliance
- ISO 14064-1 calculation standards
- EU ETS reporting requirements
- CBAM calculation accuracy
- EUDR geolocation precision

#### Compliance Test Suite Structure

```
tests/compliance/
├── __init__.py
├── conftest.py                           # Compliance test fixtures
├── epa/
│   ├── test_epa_emission_factors.py     # EPA 2024 emission factors
│   ├── test_epa_part75_cems.py          # EPA Part 75 CEMS compliance
│   └── test_epa_part98_ghg.py           # EPA Part 98 GHG reporting
├── ghg_protocol/
│   ├── test_scope1_calculations.py      # Scope 1 methodology
│   ├── test_scope2_calculations.py      # Scope 2 location/market
│   └── test_scope3_calculations.py      # Scope 3 categories
├── iso/
│   ├── test_iso14064_1.py               # GHG accounting
│   ├── test_iso14064_2.py               # GHG projects
│   └── test_iso50001.py                 # Energy management
├── eu/
│   ├── test_eu_ets.py                   # EU ETS compliance
│   ├── test_cbam_calculations.py        # CBAM carbon intensity
│   ├── test_eudr_compliance.py          # EUDR deforestation
│   └── test_csrd_reporting.py           # CSRD sustainability
├── regulatory_data/
│   ├── epa_emission_factors_2024.json   # EPA reference data
│   ├── defra_factors_2024.json          # DEFRA UK factors
│   ├── cbam_benchmarks_2024.json        # CBAM product benchmarks
│   └── egrid_factors_2023.json          # eGRID regional factors
└── reports/
    └── compliance_report_template.md
```

#### Compliance Test Implementation

```python
# tests/compliance/epa/test_epa_emission_factors.py
"""
EPA Emission Factor Compliance Tests

Validates GreenLang emission factors match EPA published values.
Reference: EPA Emission Factors Hub 2024
"""

import pytest
import json
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

# Load EPA reference data
EPA_FACTORS_PATH = Path(__file__).parent.parent / "regulatory_data" / "epa_emission_factors_2024.json"

@pytest.fixture
def epa_reference_factors():
    """Load EPA 2024 emission factors."""
    with open(EPA_FACTORS_PATH) as f:
        return json.load(f)


class TestEPAStationary Combustion:
    """Test EPA Stationary Combustion Emission Factors."""

    @pytest.mark.compliance
    @pytest.mark.parametrize("fuel_type,unit,expected_co2,expected_ch4,expected_n2o", [
        ("natural_gas", "scf", 0.05444, 0.001, 0.0001),  # kg CO2/scf
        ("natural_gas", "MJ", 0.05306, 0.001, 0.0001),   # kg CO2/MJ
        ("diesel", "gallon", 10.21, 0.0004, 0.00004),    # kg CO2/gallon
        ("coal_bituminous", "short_ton", 2327.4, 0.011, 0.0016),
        ("fuel_oil_no2", "gallon", 10.19, 0.0004, 0.00004),
        ("propane", "gallon", 5.74, 0.0003, 0.00004),
    ])
    def test_stationary_combustion_factors(
        self,
        agent,
        fuel_type,
        unit,
        expected_co2,
        expected_ch4,
        expected_n2o
    ):
        """COMP-EPA-001: Validate stationary combustion factors match EPA."""
        factor = agent._get_emission_factor(fuel_type, "US", "stationary_combustion")

        assert factor is not None, f"Missing factor for {fuel_type}"

        # CO2 factor must match within 0.1%
        assert factor.co2 == pytest.approx(expected_co2, rel=0.001), \
            f"CO2 factor mismatch for {fuel_type}: expected {expected_co2}, got {factor.co2}"

        # CH4 and N2O within 1%
        assert factor.ch4 == pytest.approx(expected_ch4, rel=0.01)
        assert factor.n2o == pytest.approx(expected_n2o, rel=0.01)

    @pytest.mark.compliance
    def test_co2e_calculation_matches_epa_methodology(self, agent):
        """COMP-EPA-002: Validate CO2e calculation uses EPA GWP values."""
        # EPA AR5 GWP values
        GWP_CO2 = 1
        GWP_CH4 = 28  # AR5 value
        GWP_N2O = 265  # AR5 value

        input_data = {
            "fuel_type": "natural_gas",
            "quantity": 1000,
            "unit": "MJ",
            "region": "US"
        }

        result = agent.run(input_data)

        # Manually calculate expected CO2e
        factor = agent._get_emission_factor("natural_gas", "US")
        expected_co2e = (
            (factor.co2 * GWP_CO2) +
            (factor.ch4 * GWP_CH4) +
            (factor.n2o * GWP_N2O)
        ) * 1000

        assert result.emissions_kgco2e == pytest.approx(expected_co2e, rel=0.001)


class TestEPAeGRIDFactors:
    """Test EPA eGRID electricity emission factors."""

    @pytest.mark.compliance
    @pytest.mark.parametrize("subregion,expected_factor", [
        ("CAMX", 0.231),   # California
        ("ERCT", 0.389),   # Texas
        ("FRCC", 0.394),   # Florida
        ("MROE", 0.563),   # Midwest
        ("NEWE", 0.234),   # New England
        ("NWPP", 0.302),   # Northwest
        ("RFCE", 0.299),   # RFC East
        ("RMPA", 0.528),   # Rocky Mountains
        ("SPNO", 0.407),   # SPP North
        ("SRSO", 0.404),   # SERC South
    ])
    def test_egrid_subregion_factors(self, agent, subregion, expected_factor):
        """COMP-EPA-003: Validate eGRID subregion factors."""
        factor = agent._get_electricity_factor(subregion)

        assert factor is not None
        assert factor == pytest.approx(expected_factor, rel=0.05)  # 5% tolerance


class TestCBAMCompliance:
    """CBAM calculation compliance tests."""

    @pytest.mark.compliance
    @pytest.mark.parametrize("product_type,benchmark_tco2e_per_tonne", [
        ("steel_hot_rolled_coil", 1.85),
        ("steel_cold_rolled_coil", 2.10),
        ("cement_clinker", 0.766),
        ("cement_portland", 0.670),
        ("aluminum_unwrought", 8.60),
        ("fertilizer_ammonia", 2.40),
        ("fertilizer_urea", 1.80),
        ("fertilizer_nitric_acid", 2.93),
        ("hydrogen", 8.85),
        ("electricity", 0.000),  # Product-specific
    ])
    def test_cbam_benchmark_values(self, cbam_agent, product_type, benchmark_tco2e_per_tonne):
        """COMP-CBAM-001: Validate CBAM product benchmarks."""
        benchmark = cbam_agent._get_benchmark(product_type)

        assert benchmark is not None
        assert benchmark.value == pytest.approx(benchmark_tco2e_per_tonne, rel=0.001)

    @pytest.mark.compliance
    def test_cbam_carbon_intensity_calculation(self, cbam_agent):
        """COMP-CBAM-002: Validate CBAM carbon intensity formula."""
        input_data = {
            "product_type": "steel_hot_rolled_coil",
            "quantity_tonnes": 1000,
            "direct_emissions_tco2e": 1500,
            "indirect_emissions_tco2e": 500
        }

        result = cbam_agent.process(input_data)

        # Carbon intensity = (direct + indirect) / quantity
        expected_ci = (1500 + 500) / 1000  # = 2.0 tCO2e/t

        assert result.carbon_intensity_tco2e_per_tonne == pytest.approx(expected_ci, rel=0.001)

        # Surplus = (CI - benchmark) * quantity
        benchmark = 1.85
        expected_surplus = (expected_ci - benchmark) * 1000  # = 150 tCO2e

        assert result.surplus_emissions_tco2e == pytest.approx(expected_surplus, rel=0.001)


class TestEUDRCompliance:
    """EUDR compliance tests."""

    @pytest.mark.compliance
    def test_eudr_geolocation_precision_requirement(self, eudr_agent):
        """COMP-EUDR-001: Validate EUDR geolocation precision (<10m)."""
        # EUDR requires polygon precision < 10 meters
        input_data = {
            "commodity_type": "coffee",
            "geolocation": {
                "type": "Point",
                "coordinates": [-47.9292, -15.7801],
                "precision_meters": 5
            }
        }

        result = eudr_agent.validate_geolocation(input_data["geolocation"])

        assert result.valid is True
        assert result.precision_adequate is True

    @pytest.mark.compliance
    def test_eudr_deforestation_cutoff_date(self, eudr_agent):
        """COMP-EUDR-002: Validate December 31, 2020 cutoff date."""
        # Production after cutoff should pass
        input_after = {
            "commodity_type": "soy",
            "production_date": "2021-01-01",
            "geolocation": {"coordinates": [-50.0, -10.0]}
        }
        result_after = eudr_agent.check_deforestation(input_after)
        assert result_after.deforestation_free is True

        # Production before cutoff needs verification
        input_before = {
            "commodity_type": "soy",
            "production_date": "2020-06-15",
            "geolocation": {"coordinates": [-50.0, -10.0]}
        }
        result_before = eudr_agent.check_deforestation(input_before)
        # Should require satellite verification
        assert result_before.requires_satellite_verification is True

    @pytest.mark.compliance
    @pytest.mark.parametrize("commodity", [
        "cattle", "cocoa", "coffee", "palm_oil", "rubber", "soy", "wood"
    ])
    def test_eudr_regulated_commodities(self, eudr_agent, commodity):
        """COMP-EUDR-003: Validate all 7 EUDR commodities covered."""
        result = eudr_agent.is_regulated(commodity)
        assert result is True, f"{commodity} should be EUDR regulated"
```

---

## Section 3: Gap Analysis - Documentation (7.2)

### 3.1 Documentation Structure Recommendation

```
docs/
├── api/                          # API Reference (TASK-252)
│   ├── README.md                 # API overview
│   ├── authentication.md         # Auth guide
│   ├── agents/                   # Per-agent API docs
│   │   ├── gl-001-carbon.md
│   │   ├── gl-002-cbam.md
│   │   └── ...
│   ├── endpoints/                # REST endpoint reference
│   └── schemas/                  # Request/response schemas
│
├── operator/                     # Operator Manuals (TASK-253)
│   ├── README.md
│   ├── deployment/               # Deployment procedures
│   ├── monitoring/               # Monitoring setup
│   ├── alerting/                 # Alert configuration
│   └── runbooks/                 # Operational runbooks
│
├── admin/                        # Administrator Guides (TASK-254)
│   ├── README.md
│   ├── installation/             # Installation guides
│   ├── configuration/            # Configuration reference
│   ├── security/                 # Security setup
│   └── maintenance/              # Maintenance procedures
│
├── troubleshooting/              # Troubleshooting Guides (TASK-255)
│   ├── README.md
│   ├── common-issues.md          # Common issues
│   ├── error-codes.md            # Error code reference
│   ├── performance.md            # Performance issues
│   └── debugging.md              # Debug procedures
│
├── training/                     # Training Materials (TASK-256)
│   ├── README.md
│   ├── fundamentals/             # GreenLang basics
│   ├── agents/                   # Agent training
│   ├── integration/              # Integration training
│   └── exercises/                # Hands-on exercises
│
├── quickstart/                   # Quick Start Guides (TASK-257)
│   ├── README.md
│   ├── 5-minute-guide.md         # Fastest path
│   ├── first-calculation.md      # First calculation
│   └── deployment-quickstart.md  # Quick deployment
│
├── tutorials/                    # Video Tutorials (TASK-258)
│   ├── README.md
│   └── video-index.md            # Video links
│
├── faq/                          # FAQ Documentation (TASK-259)
│   ├── README.md
│   ├── general.md                # General FAQ
│   ├── technical.md              # Technical FAQ
│   └── regulatory.md             # Regulatory FAQ
│
└── releases/                     # Release Notes (TASK-260)
    ├── RELEASE_NOTES_TEMPLATE.md
    ├── CHANGELOG.md
    └── versions/
        ├── v1.0.0.md
        └── ...
```

### 3.2 TASK-252: API Reference Documentation

**Priority:** P0 - Critical

#### Template: Agent API Documentation

```markdown
# GL-001: Carbon Emissions Calculator Agent

## Overview

The Carbon Emissions Calculator Agent (GL-001) calculates greenhouse gas emissions
from fuel consumption data using validated emission factors from EPA, DEFRA, and IEA.

## API Endpoint

```
POST /v1/agents/carbon/calculate
```

## Authentication

```http
Authorization: Bearer <access_token>
```

## Request Schema

```json
{
  "fuel_type": "natural_gas",     // Required: FuelType enum
  "quantity": 1000.0,             // Required: Numeric > 0
  "unit": "MJ",                   // Required: Unit enum
  "region": "US",                 // Required: ISO 3166-1 alpha-2
  "scope": "SCOPE_1",             // Optional: Default SCOPE_1
  "calculation_method": "location" // Optional: location|market
}
```

### FuelType Enum

| Value | Description | Default Unit |
|-------|-------------|--------------|
| `natural_gas` | Natural gas | m3, MJ, therms |
| `diesel` | Diesel fuel | L, gallon |
| `gasoline` | Motor gasoline | L, gallon |
| `coal` | Coal (bituminous) | kg, short_ton |
| `electricity_grid` | Grid electricity | kWh |
| `fuel_oil` | Fuel oil #2/#4 | L, gallon |
| `propane` | LPG/Propane | L, gallon |

## Response Schema

```json
{
  "success": true,
  "emissions_kgco2e": 56.1,
  "emission_factor": 0.0561,
  "emission_factor_unit": "kgCO2e/MJ",
  "emission_factor_source": "EPA",
  "emission_factor_year": 2024,
  "scope": 1,
  "calculation_method": "location",
  "provenance_hash": "abc123...",
  "calculated_at": "2025-01-15T10:30:00Z"
}
```

## Error Responses

| Status | Code | Description |
|--------|------|-------------|
| 400 | INVALID_INPUT | Invalid request parameters |
| 401 | UNAUTHORIZED | Missing or invalid token |
| 404 | FACTOR_NOT_FOUND | Emission factor unavailable |
| 429 | RATE_LIMITED | Too many requests |
| 500 | CALCULATION_ERROR | Internal calculation error |

## Example

```bash
curl -X POST https://api.greenlang.io/v1/agents/carbon/calculate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "fuel_type": "natural_gas",
    "quantity": 1000,
    "unit": "MJ",
    "region": "US",
    "scope": "SCOPE_1"
  }'
```

## Emission Factor Sources

| Source | Region | Update Frequency |
|--------|--------|------------------|
| EPA | US | Annual |
| DEFRA | UK | Annual |
| IEA | Global | Annual |
| eGRID | US (regional) | Biennial |

## Related Agents

- [GL-002: CBAM Compliance Agent](./gl-002-cbam.md)
- [GL-010: Emissions Guardian Agent](./gl-010-emissions.md)
```

### 3.3 TASK-260: Release Notes Template

```markdown
# Release Notes v{VERSION}

**Release Date:** {DATE}
**Release Type:** {Major|Minor|Patch|Security}

## Highlights

Brief summary of the most important changes in this release.

## New Features

### Feature 1: {Feature Name}
- Description of the feature
- How to use it
- Related documentation links

### Feature 2: {Feature Name}
- Description
- Usage

## Improvements

- {Improvement 1}
- {Improvement 2}
- {Improvement 3}

## Bug Fixes

- **{BUG-ID}**: {Brief description of the bug and fix}
- **{BUG-ID}**: {Brief description}

## Breaking Changes

**IMPORTANT:** Review these changes before upgrading.

### {Breaking Change 1}

**Before:**
```python
# Old API
agent.calculate(data)
```

**After:**
```python
# New API
agent.run(data)
```

**Migration Guide:** [Link to migration guide]

## Deprecations

The following features are deprecated and will be removed in version X.Y.Z:

| Feature | Deprecated | Removal Version | Replacement |
|---------|------------|-----------------|-------------|
| `FuelAgentAI` | v1.5.0 | v2.0.0 | `FuelAgent` |

## Security Updates

- **{CVE-ID}**: {Description of security fix}
- Dependency updates addressing security vulnerabilities

## Performance Improvements

| Component | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Carbon Agent | 15ms | 5ms | 3x faster |

## Dependency Updates

| Package | Previous | New | Notes |
|---------|----------|-----|-------|
| pydantic | 2.4.0 | 2.5.0 | |
| numpy | 1.25.0 | 1.26.0 | |

## Known Issues

- {Known issue 1}
- {Known issue 2}

## Upgrade Instructions

```bash
pip install --upgrade greenlang==X.Y.Z
```

### Configuration Changes

Add the following to your configuration:

```yaml
greenlang:
  new_feature:
    enabled: true
```

## Contributors

Thanks to all contributors for this release:
- @contributor1
- @contributor2

## Full Changelog

[Compare v{PREVIOUS} to v{VERSION}](https://github.com/greenlang/greenlang/compare/v{PREVIOUS}...v{VERSION})
```

---

## Section 4: Test Coverage Targets

### 4.1 Coverage Requirements by Component

| Component | Unit | Integration | E2E | Performance | Compliance | Target |
|-----------|------|-------------|-----|-------------|------------|--------|
| GL-001 Carbon | 85% | 80% | 100% | Pass | 100% | **85%+** |
| GL-002 CBAM | 85% | 80% | 100% | Pass | 100% | **85%+** |
| GL-003 Steam | 85% | 80% | 100% | Pass | 100% | **85%+** |
| GL-004-020 | 80% | 75% | 100% | Pass | 90% | **80%+** |
| API Layer | 90% | 85% | 100% | Pass | N/A | **85%+** |
| Infrastructure | 75% | 80% | 100% | Pass | N/A | **75%+** |
| **Overall** | **85%** | **80%** | **100%** | **Pass** | **95%** | **85%+** |

### 4.2 pytest-cov Configuration

```ini
# pytest.ini
[pytest]
addopts =
    --cov=greenlang
    --cov-report=html:coverage_html
    --cov-report=xml:coverage.xml
    --cov-report=term-missing
    --cov-fail-under=85
    --cov-branch

testpaths = tests

markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance tests
    compliance: Compliance validation tests
    security: Security tests
    chaos: Chaos engineering tests
    contract: Contract tests
    critical_path: Critical path compliance tests

filterwarnings =
    error
    ignore::DeprecationWarning
```

### 4.3 Coverage Reporting

```yaml
# .github/workflows/coverage.yml
name: Test Coverage

on: [push, pull_request]

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install pytest-cov codecov

      - name: Run tests with coverage
        run: |
          pytest tests/ \
            --cov=greenlang \
            --cov-report=xml \
            --cov-fail-under=85

      - name: Upload to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          fail_ci_if_error: true
          verbose: true
```

---

## Section 5: Implementation Roadmap

### 5.1 Week-by-Week Plan

```
Week 14:
├── TASK-248: Compliance Validation Tests (P0)
│   ├── EPA emission factor tests
│   ├── CBAM calculation tests
│   └── EUDR compliance tests
├── TASK-252: API Reference Documentation (P0)
│   ├── GL-001 to GL-005 API docs
│   └── Authentication guide
└── TASK-247: Security Penetration Tests (P1)
    ├── OWASP Top 10 test suite
    └── Security scanning setup

Week 15:
├── TASK-253: Operator Manuals (P1)
│   ├── Deployment procedures
│   ├── Monitoring setup
│   └── Runbooks
├── TASK-243: Contract Testing (P2)
│   ├── Pact integration
│   └── Core agent contracts
├── TASK-246: Chaos Testing (P2)
│   ├── Litmus setup
│   └── Initial chaos experiments
└── TASK-254: Administrator Guides (P3)

Week 16:
├── TASK-255: Troubleshooting Guides (P3)
├── TASK-256: Training Materials (P3)
├── TASK-257: Quick Start Guides (P3)
├── TASK-258: Video Tutorials (P3)
├── TASK-259: FAQ Documentation (P3)
└── TASK-260: Release Notes Template (P3)
```

### 5.2 Resource Requirements

| Task | Estimated Effort | Skills Required |
|------|------------------|-----------------|
| TASK-243 | 3-4 days | Python, Pact, API design |
| TASK-246 | 5-7 days | Kubernetes, Litmus, SRE |
| TASK-247 | 5-7 days | Security, OWASP, Penetration testing |
| TASK-248 | 7-10 days | Regulatory knowledge, Python |
| TASK-252 | 5-7 days | Technical writing, OpenAPI |
| TASK-253 | 3-4 days | Operations, Technical writing |
| TASK-254 | 2-3 days | System administration |
| TASK-255 | 2-3 days | Support experience |
| TASK-256 | 5-7 days | Training design |
| TASK-257 | 2 days | Technical writing |
| TASK-258 | 3-5 days | Video production |
| TASK-259 | 2 days | Product knowledge |
| TASK-260 | 1 day | Release management |

**Total Estimated Effort:** 40-55 person-days

---

## Section 6: Success Criteria

### 6.1 Testing Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Overall Test Coverage | >= 85% | pytest-cov |
| Critical Path Coverage | 100% | Phase 5 tests |
| Compliance Test Pass Rate | 100% | CI/CD pipeline |
| Performance Test Pass Rate | 100% | Benchmark tests |
| Contract Test Coverage | >= 90% | Pact verification |
| Security Test Pass Rate | 100% | OWASP ZAP |
| Chaos Test Recovery | >= 99.5% | Litmus experiments |

### 6.2 Documentation Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| API Documentation Coverage | 100% agents | Manual review |
| Quick Start Time to First Calculation | < 5 minutes | User testing |
| Documentation Accuracy | Zero errors | Review process |
| FAQ Coverage of Support Tickets | >= 80% | Support analysis |

---

## Appendix A: File Locations

### Existing Test Files (Completed)

| File | Location | Tests |
|------|----------|-------|
| E2E Tests | `tests/e2e/test_complete_workflows.py` | 25 |
| Integration Pipeline | `tests/integration/test_agent_pipeline_comprehensive.py` | 50 |
| Load Tests | `tests/performance/test_load_comprehensive.py` | 15 |
| Performance Benchmarks | `tests/performance/test_performance_benchmarks.py` | 40 |
| Unit Tests | `tests/unit/test_gl001_carbon_agent.py` | 50 |
| Critical Path | `tests/agents/phase5/test_critical_path_compliance.py` | 50+ |

### New Test Files (To Create)

| File | Location | Description |
|------|----------|-------------|
| Contract Tests | `tests/contract/test_agent_contracts.py` | Pact contracts |
| Chaos Tests | `tests/chaos/scenarios/test_agent_resilience.py` | Chaos experiments |
| Security Tests | `tests/security/test_owasp_top10.py` | OWASP validation |
| Compliance Tests | `tests/compliance/epa/test_epa_emission_factors.py` | EPA validation |
| CBAM Compliance | `tests/compliance/eu/test_cbam_calculations.py` | CBAM validation |
| EUDR Compliance | `tests/compliance/eu/test_eudr_compliance.py` | EUDR validation |

---

## Appendix B: Regulatory Reference Data

### EPA Emission Factors 2024

```json
{
  "stationary_combustion": {
    "natural_gas": {
      "scf": {"co2": 0.05444, "ch4": 0.001, "n2o": 0.0001},
      "MJ": {"co2": 0.05306, "ch4": 0.001, "n2o": 0.0001}
    },
    "diesel": {
      "gallon": {"co2": 10.21, "ch4": 0.0004, "n2o": 0.00004}
    }
  },
  "mobile_combustion": {
    "gasoline_passenger": {
      "mile": {"co2": 0.404, "ch4": 0.0000076, "n2o": 0.0000076}
    }
  }
}
```

### CBAM Product Benchmarks 2024

```json
{
  "steel_hot_rolled_coil": 1.85,
  "steel_cold_rolled_coil": 2.10,
  "cement_clinker": 0.766,
  "cement_portland": 0.670,
  "aluminum_unwrought": 8.60,
  "fertilizer_ammonia": 2.40,
  "fertilizer_urea": 1.80,
  "fertilizer_nitric_acid": 2.93,
  "hydrogen": 8.85
}
```

---

**Document Status:** Ready for Implementation
**Next Review:** Week 14 Sprint Planning
**Owner:** GL-TestEngineer
