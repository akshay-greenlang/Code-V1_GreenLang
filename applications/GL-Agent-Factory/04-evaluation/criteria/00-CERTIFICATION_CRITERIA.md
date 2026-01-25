# Agent Certification Criteria

**Version:** 1.0.0
**Date:** 2025-12-03
**Status:** Active
**Owner:** GreenLang Quality Engineering Team

---

## Executive Summary

This document defines the **12-Dimension Certification Criteria** that every GreenLang agent must pass to achieve **Certified** status and production deployment authorization. These criteria ensure agents meet rigorous standards for accuracy, compliance, climate science validity, performance, and operational excellence.

**Certification Requirement:** Agents must pass **ALL 12 dimensions** to be certified. A failure in any single dimension results in rejection and requires remediation before re-application.

---

## 12-Dimension Certification Framework

| Dimension | Weight | Pass Threshold | Sign-Off Required |
|-----------|--------|---------------|-------------------|
| 1. Specification Completeness | 5% | 100% | Product Manager |
| 2. Code Implementation | 10% | 100% | Lead Engineer |
| 3. Test Coverage | 15% | >85% coverage | QA Engineer |
| 4. Deterministic AI Guarantees | 10% | 100% reproducibility | QA Engineer |
| 5. Documentation Completeness | 5% | 100% | Technical Writer |
| 6. Compliance & Security | 20% | 100% | Legal + InfoSec |
| 7. Deployment Readiness | 5% | 100% | DevOps Engineer |
| 8. Exit Bar Criteria | 10% | 100% | QA Engineer |
| 9. Integration & Coordination | 5% | 100% | Integration Engineer |
| 10. Business Impact & Metrics | 5% | 100% | Product Manager |
| 11. Operational Excellence | 5% | 100% | SRE |
| 12. Continuous Improvement | 5% | 100% | Engineering Manager |

**Total Weight:** 100%

**Minimum Passing Score:** 100% (all dimensions must pass)

---

## Dimension 1: Specification Completeness

### Objective

Validate that the agent specification is complete, unambiguous, and provides sufficient detail for implementation and certification.

### Pass Criteria

**MUST HAVE (100% Required):**

1. **Agent Metadata:**
   - Agent name (e.g., "BoilerEfficiencyOptimizer")
   - Agent ID (e.g., "GL-002")
   - Version (semantic versioning: major.minor.patch)
   - Domain (e.g., "Industrial Decarbonization")
   - Priority (P0, P1, P2, P3)

2. **Agent Purpose:**
   - Clear problem statement (what problem does this agent solve?)
   - Target audience (who uses this agent?)
   - Value proposition (why is this agent valuable?)
   - Success criteria (how do we measure success?)

3. **Tools Definition (for each tool):**
   - Tool name (snake_case)
   - Tool description (what does this tool do?)
   - Input parameters (name, type, description, required/optional, default value, validation rules)
   - Output schema (name, type, description, units)
   - Calculation methodology (formulas, algorithms, standards referenced)
   - Example usage (realistic example with inputs and expected outputs)

4. **Standards & Regulations:**
   - List all industry standards referenced (e.g., ASHRAE, ISO, ASTM)
   - List all regulatory requirements (e.g., EPA, CBAM, CSRD)
   - Citation format: "Standard Name (Year), Section X.Y"

5. **Dependencies:**
   - Other agents required (e.g., "Requires FuelAgent for emission factors")
   - External data sources (e.g., "Requires weather data from NOAA")
   - Third-party APIs (e.g., "Requires EIA API for fuel prices")

6. **Limitations & Assumptions:**
   - Clearly state all assumptions (e.g., "Assumes steady-state operation")
   - Clearly state limitations (e.g., "Not valid for boilers >100 MMBtu/hr")
   - Edge cases not covered (e.g., "Does not handle dual-fuel boilers")

### Evaluation Methodology

**Automated Checks:**
- YAML schema validation (specification file must parse)
- Required fields present (metadata, purpose, tools, standards)
- Tool schema validation (all tools have inputs, outputs, methodology)

**Manual Review:**
- Product Manager reviews for completeness and clarity
- Subject matter expert reviews technical accuracy
- Documentation team reviews for readability

### Example Pass Criteria

```yaml
# Example: Boiler Efficiency Optimizer Specification

metadata:
  agent_id: GL-002
  agent_name: BoilerEfficiencyOptimizer
  version: 1.0.0
  domain: industrial_decarbonization
  priority: P0

purpose:
  problem: "Industrial boilers operate at 60-85% efficiency, wasting 15-40% of fuel energy"
  audience: "Facility managers, energy engineers, sustainability teams"
  value: "Optimize boiler efficiency to reduce fuel consumption, costs, and emissions"
  success: "Achieve >80% efficiency for 95% of boilers analyzed"

tools:
  - name: calculate_boiler_efficiency
    description: "Calculate boiler efficiency using ASME PTC 4 methodology"
    inputs:
      - name: fuel_type
        type: string
        required: true
        description: "Fuel type (natural_gas, coal, oil, biomass)"
        validation: "Must be one of: [natural_gas, coal, oil, biomass]"
      - name: firing_rate_mmbtu_hr
        type: float
        required: true
        description: "Boiler firing rate in MMBtu/hr"
        validation: "Must be > 0 and < 100"
    outputs:
      - name: efficiency_percent
        type: float
        description: "Boiler efficiency (%)"
        units: "percent"
      - name: fuel_savings_potential_mmbtu_yr
        type: float
        description: "Annual fuel savings potential"
        units: "MMBtu/year"
    methodology: |
      Efficiency calculated using ASME PTC 4 heat loss method:
      Efficiency = 100% - (Stack Loss + Radiation Loss + Blowdown Loss + ...)

      Stack Loss: Q_stack = m_flue × cp × (T_flue - T_ambient)
      Radiation Loss: 0.5% for insulated boilers, 2% for uninsulated
      Blowdown Loss: Q_blowdown = m_blowdown × h_fg

      Standards: ASME PTC 4-2013 (Fired Steam Generators)
    example:
      input:
        fuel_type: natural_gas
        firing_rate_mmbtu_hr: 15.0
        flue_gas_temp_f: 350
        ambient_temp_f: 70
      output:
        efficiency_percent: 82.45
        fuel_savings_potential_mmbtu_yr: 1250

standards:
  - ASME PTC 4-2013 (Fired Steam Generators)
  - ISO 50001:2018 (Energy Management Systems)
  - EPA 40 CFR Part 60 Subpart Db (Industrial Boilers)

dependencies:
  agents:
    - FuelAgent (emission factors)
    - GridFactorAgent (electricity emissions)
  data_sources:
    - Fuel price database (EIA)
    - Weather data (NOAA)

limitations:
  - Valid for boilers 1-100 MMBtu/hr firing rate
  - Assumes steady-state operation (not valid during startup/shutdown)
  - Does not handle dual-fuel or multi-burner configurations
  - Requires manual input of flue gas composition (no real-time sensors)
```

### Sign-Off

**Product Manager:** _________________ Date: _______

---

## Dimension 2: Code Implementation

### Objective

Validate that the agent implementation is production-grade, follows best practices, and correctly implements all specified tools.

### Pass Criteria

**CODE QUALITY (100% Required):**

1. **All Tools Implemented:**
   - Every tool in specification has corresponding implementation
   - Tool signatures match specification (inputs, outputs)
   - Tool logic correctly implements specified methodology

2. **Error Handling:**
   - All functions have try/except blocks for error handling
   - Input validation on all parameters (type, range, format)
   - Graceful degradation when dependencies unavailable
   - Clear, actionable error messages

3. **Provenance Tracking:**
   - Every tool returns provenance metadata
   - Provenance includes: tool name, version, timestamp, inputs, methodology, standards
   - Provenance hash is deterministic (same inputs → same hash)

4. **Code Structure:**
   - Clear separation of concerns (calculation tools vs orchestration)
   - Functions are single-purpose (<100 lines)
   - No hardcoded values (use configuration)
   - No secrets in code (use environment variables)

5. **Type Annotations:**
   - All functions have type hints (parameters and return values)
   - Type hints are accurate and complete
   - Mypy passes with no errors

6. **Documentation:**
   - Module-level docstring (purpose, architecture, usage)
   - Function-level docstrings (Args, Returns, Raises, Example)
   - Inline comments for complex logic
   - Calculation formulas documented with units

### Code Quality Checklist

```python
# PASS Example: Production-grade implementation

from typing import Dict, Any
import logging
from datetime import datetime
from pydantic import BaseModel, validator

logger = logging.getLogger(__name__)


class BoilerEfficiencyInput(BaseModel):
    """Input parameters for boiler efficiency calculation."""
    fuel_type: str
    firing_rate_mmbtu_hr: float
    flue_gas_temp_f: float
    ambient_temp_f: float

    @validator('fuel_type')
    def validate_fuel_type(cls, v):
        allowed = ['natural_gas', 'coal', 'oil', 'biomass']
        if v not in allowed:
            raise ValueError(f"fuel_type must be one of {allowed}")
        return v

    @validator('firing_rate_mmbtu_hr')
    def validate_firing_rate(cls, v):
        if v <= 0 or v > 100:
            raise ValueError("firing_rate must be 0 < x <= 100 MMBtu/hr")
        return v


def calculate_boiler_efficiency(
    fuel_type: str,
    firing_rate_mmbtu_hr: float,
    flue_gas_temp_f: float,
    ambient_temp_f: float
) -> Dict[str, Any]:
    """
    Calculate boiler efficiency using ASME PTC 4 heat loss method.

    Args:
        fuel_type: Fuel type (natural_gas, coal, oil, biomass)
        firing_rate_mmbtu_hr: Boiler firing rate in MMBtu/hr
        flue_gas_temp_f: Flue gas temperature in °F
        ambient_temp_f: Ambient temperature in °F

    Returns:
        Dict containing:
            - efficiency_percent: Boiler efficiency (%)
            - stack_loss_percent: Stack heat loss (%)
            - radiation_loss_percent: Radiation heat loss (%)
            - provenance: Calculation metadata

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If calculation fails

    Example:
        >>> result = calculate_boiler_efficiency(
        ...     fuel_type="natural_gas",
        ...     firing_rate_mmbtu_hr=15.0,
        ...     flue_gas_temp_f=350,
        ...     ambient_temp_f=70
        ... )
        >>> print(result['efficiency_percent'])
        82.45
    """
    try:
        # Validate inputs using Pydantic
        inputs = BoilerEfficiencyInput(
            fuel_type=fuel_type,
            firing_rate_mmbtu_hr=firing_rate_mmbtu_hr,
            flue_gas_temp_f=flue_gas_temp_f,
            ambient_temp_f=ambient_temp_f
        )

        # Calculate stack loss
        # Q_stack = m_flue × cp × ΔT / Q_input
        delta_t = flue_gas_temp_f - ambient_temp_f
        stack_loss_percent = 0.01 * delta_t  # Simplified for example

        # Calculate radiation loss (0.5% for insulated boilers)
        radiation_loss_percent = 0.5

        # Calculate efficiency
        efficiency_percent = 100.0 - stack_loss_percent - radiation_loss_percent

        # Build provenance
        provenance = {
            "tool": "calculate_boiler_efficiency",
            "version": "1.0.0",
            "methodology": "ASME PTC 4-2013 heat loss method",
            "timestamp": datetime.utcnow().isoformat(),
            "inputs": {
                "fuel_type": fuel_type,
                "firing_rate_mmbtu_hr": firing_rate_mmbtu_hr,
                "flue_gas_temp_f": flue_gas_temp_f,
                "ambient_temp_f": ambient_temp_f
            },
            "standards": ["ASME PTC 4-2013"],
            "deterministic": True
        }

        logger.info(f"Boiler efficiency calculated: {efficiency_percent:.2f}%")

        return {
            "efficiency_percent": efficiency_percent,
            "stack_loss_percent": stack_loss_percent,
            "radiation_loss_percent": radiation_loss_percent,
            "provenance": provenance
        }

    except ValueError as e:
        logger.error(f"Input validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        raise RuntimeError(f"Boiler efficiency calculation failed: {e}")
```

### Evaluation Methodology

**Automated Checks:**
- Linting (flake8, pylint): Must score >8.0/10
- Type checking (mypy): Must pass with no errors
- Security scan (bandit): No P0/P1 vulnerabilities
- Complexity analysis (radon): Cyclomatic complexity <10

**Manual Review:**
- Code review by 2+ senior engineers
- Architecture review for scalability
- Performance review for optimization opportunities

### Sign-Off

**Lead Engineer:** _________________ Date: _______

---

## Dimension 3: Test Coverage

### Objective

Validate that the agent has comprehensive test coverage (>85%) with unit tests, integration tests, golden tests, and edge case tests.

### Pass Criteria

**TEST COVERAGE (>85% Required):**

1. **Unit Tests:**
   - Every tool has dedicated unit tests
   - Test happy path (valid inputs → expected outputs)
   - Test edge cases (boundary values, extreme inputs)
   - Test error handling (invalid inputs → ValueError)
   - Target: >90% coverage of tool code

2. **Integration Tests:**
   - Test agent orchestration (multi-tool workflows)
   - Test dependencies (agent-to-agent integration)
   - Test external API integration (mocked)
   - Target: >80% coverage of orchestration code

3. **Golden Tests:**
   - 25+ golden test scenarios with known correct answers
   - Cover all major calculation paths
   - Test determinism (same inputs → same outputs across runs)
   - Test cross-platform reproducibility (Windows, Linux, macOS)
   - Target: 100% of golden tests pass

4. **Performance Tests:**
   - Latency test (P95 <4 seconds)
   - Cost test (<$0.15 per analysis)
   - Throughput test (>100 req/s sustained)
   - Target: 100% of performance tests pass

5. **Security Tests:**
   - Input validation tests (SQL injection, XSS, command injection)
   - Authentication tests (valid/invalid tokens)
   - Authorization tests (RBAC enforcement)
   - Target: 100% of security tests pass

### Test Coverage Breakdown

```
Overall Coverage: >85%
├── Unit Tests: >90% of tool code
├── Integration Tests: >80% of orchestration code
├── Golden Tests: 25+ scenarios, 100% pass
├── Performance Tests: 100% pass
└── Security Tests: 100% pass
```

### Example Test Suite Structure

```python
# tests/agents/test_boiler_efficiency_optimizer.py

import pytest
from greenlang.agents.boiler_efficiency_optimizer import BoilerEfficiencyOptimizer


class TestBoilerEfficiencyCalculation:
    """Unit tests for boiler efficiency calculation tool."""

    def test_calculate_efficiency_natural_gas(self):
        """Test efficiency calculation for natural gas boiler."""
        agent = BoilerEfficiencyOptimizer()
        result = agent.calculate_boiler_efficiency(
            fuel_type="natural_gas",
            firing_rate_mmbtu_hr=15.0,
            flue_gas_temp_f=350,
            ambient_temp_f=70
        )

        assert result['efficiency_percent'] == pytest.approx(82.45, rel=0.01)
        assert result['stack_loss_percent'] < 10.0
        assert result['provenance']['deterministic'] is True

    def test_calculate_efficiency_invalid_fuel(self):
        """Test error handling for invalid fuel type."""
        agent = BoilerEfficiencyOptimizer()

        with pytest.raises(ValueError, match="fuel_type must be one of"):
            agent.calculate_boiler_efficiency(
                fuel_type="nuclear",  # Invalid
                firing_rate_mmbtu_hr=15.0,
                flue_gas_temp_f=350,
                ambient_temp_f=70
            )

    @pytest.mark.parametrize("firing_rate,expected_efficiency", [
        (1.0, 78.5),
        (10.0, 82.0),
        (50.0, 85.2),
        (100.0, 87.1),
    ])
    def test_calculate_efficiency_firing_rate_range(
        self, firing_rate, expected_efficiency
    ):
        """Test efficiency calculation across firing rate range."""
        agent = BoilerEfficiencyOptimizer()
        result = agent.calculate_boiler_efficiency(
            fuel_type="natural_gas",
            firing_rate_mmbtu_hr=firing_rate,
            flue_gas_temp_f=350,
            ambient_temp_f=70
        )

        assert result['efficiency_percent'] == pytest.approx(
            expected_efficiency, rel=0.05
        )


class TestGoldenScenarios:
    """Golden tests with known correct answers."""

    def test_golden_scenario_001_natural_gas_boiler(self):
        """
        Golden test: Natural gas boiler efficiency
        Known correct answer: 82.45678901234567%
        """
        agent = BoilerEfficiencyOptimizer()
        result = agent.calculate_boiler_efficiency(
            fuel_type="natural_gas",
            firing_rate_mmbtu_hr=15.0,
            flue_gas_temp_f=350.0,
            ambient_temp_f=70.0
        )

        # Bit-perfect reproducibility (12 decimal places)
        assert result['efficiency_percent'] == pytest.approx(
            82.45678901234567, rel=1e-12
        )

        # Provenance hash must match known value
        assert result['provenance']['hash'] == "a1b2c3d4e5f6g7h8..."

    def test_determinism_across_runs(self):
        """Test that same inputs produce same outputs across runs."""
        agent = BoilerEfficiencyOptimizer()

        results = [
            agent.calculate_boiler_efficiency(
                fuel_type="natural_gas",
                firing_rate_mmbtu_hr=15.0,
                flue_gas_temp_f=350.0,
                ambient_temp_f=70.0
            )
            for _ in range(10)
        ]

        # All results must be identical
        first_hash = results[0]['provenance']['hash']
        for result in results[1:]:
            assert result['provenance']['hash'] == first_hash


class TestPerformance:
    """Performance tests (latency, cost, throughput)."""

    def test_latency_under_4_seconds(self, benchmark):
        """Test that P95 latency is <4 seconds."""
        agent = BoilerEfficiencyOptimizer()

        result = benchmark(
            agent.calculate_boiler_efficiency,
            fuel_type="natural_gas",
            firing_rate_mmbtu_hr=15.0,
            flue_gas_temp_f=350,
            ambient_temp_f=70
        )

        # Benchmark automatically calculates P95
        assert benchmark.stats.stats.max < 4.0  # P95 < 4s

    def test_cost_under_15_cents(self):
        """Test that cost per analysis is <$0.15."""
        agent = BoilerEfficiencyOptimizer()

        result = agent.calculate_boiler_efficiency(
            fuel_type="natural_gas",
            firing_rate_mmbtu_hr=15.0,
            flue_gas_temp_f=350,
            ambient_temp_f=70
        )

        assert result['cost_usd'] < 0.15
```

### Evaluation Methodology

**Automated Checks:**
- pytest with pytest-cov: Calculate coverage percentage
- Coverage report: Must be >85% overall
- Golden tests: Must have 25+ scenarios, 100% passing
- Performance tests: Must meet latency, cost, throughput targets

**Manual Review:**
- QA Engineer reviews test quality and coverage
- Identifies missing test cases
- Reviews edge case coverage

### Sign-Off

**QA Engineer:** _________________ Date: _______

---

## Dimension 4: Deterministic AI Guarantees

### Objective

Validate that the agent produces bit-perfect reproducible results (same inputs → same outputs) across different environments, platforms, and execution times.

### Pass Criteria

**DETERMINISM (100% Reproducibility Required):**

1. **Configuration:**
   - LLM temperature = 0.0 (no randomness)
   - Random seed = 42 (if using random number generation)
   - Deterministic flag = True (in all tool outputs)

2. **Tool Implementation:**
   - All calculations use deterministic math (no random numbers)
   - Floating-point arithmetic uses consistent precision (Decimal for high-precision)
   - No time-based randomness (use fixed timestamps for testing)
   - No external API calls with non-deterministic responses (mock for testing)

3. **Cross-Run Reproducibility:**
   - Same inputs produce same outputs across 10 consecutive runs
   - Provenance hash is identical across runs
   - No variation in numerical outputs (bit-perfect match)

4. **Cross-Platform Reproducibility:**
   - Same inputs produce same outputs on Windows, Linux, macOS
   - Same inputs produce same outputs on x86_64 and ARM architectures
   - Same inputs produce same outputs in local, Docker, and CI/CD environments

5. **Cross-Version Reproducibility:**
   - Same inputs produce same outputs across Python 3.8, 3.9, 3.10, 3.11, 3.12
   - Breaking changes require major version bump (1.0.0 → 2.0.0)

### Determinism Test Example

```python
def test_determinism_cross_run():
    """Test that same inputs produce same outputs across 10 runs."""
    agent = BoilerEfficiencyOptimizer(temperature=0.0, seed=42)

    results = [
        agent.calculate_boiler_efficiency(
            fuel_type="natural_gas",
            firing_rate_mmbtu_hr=15.0,
            flue_gas_temp_f=350.0,
            ambient_temp_f=70.0
        )
        for _ in range(10)
    ]

    # Extract provenance hashes
    hashes = [r['provenance']['hash'] for r in results]

    # All hashes must be identical
    assert len(set(hashes)) == 1, "Provenance hash varies across runs"

    # All numerical outputs must be identical (bit-perfect)
    efficiencies = [r['efficiency_percent'] for r in results]
    assert len(set(efficiencies)) == 1, "Efficiency varies across runs"


@pytest.mark.parametrize("platform", ["windows", "linux", "macos"])
def test_determinism_cross_platform(platform):
    """Test that same inputs produce same outputs across platforms."""
    # This test runs in CI/CD on multiple platforms
    agent = BoilerEfficiencyOptimizer(temperature=0.0, seed=42)

    result = agent.calculate_boiler_efficiency(
        fuel_type="natural_gas",
        firing_rate_mmbtu_hr=15.0,
        flue_gas_temp_f=350.0,
        ambient_temp_f=70.0
    )

    # Known correct answer (from golden test)
    expected_hash = "a1b2c3d4e5f6g7h8..."

    assert result['provenance']['hash'] == expected_hash
    assert result['efficiency_percent'] == pytest.approx(
        82.45678901234567, rel=1e-12
    )
```

### Evaluation Methodology

**Automated Checks:**
- Run determinism tests in CI/CD (10 runs per test)
- Run cross-platform tests on Windows, Linux, macOS
- Run cross-version tests on Python 3.8-3.12
- Verify all tests pass (100% pass rate required)

**Manual Review:**
- QA Engineer reviews configuration for deterministic settings
- QA Engineer reviews code for sources of non-determinism

### Sign-Off

**QA Engineer:** _________________ Date: _______

---

## Dimension 5: Documentation Completeness

### Objective

Validate that the agent has comprehensive, clear, and accurate documentation for developers, users, and operators.

### Pass Criteria

**DOCUMENTATION (100% Required):**

1. **Specification (YAML):**
   - Complete agent metadata (name, ID, version, domain, priority)
   - Complete tool definitions (inputs, outputs, methodology, examples)
   - Standards and regulations referenced
   - Dependencies documented
   - Limitations and assumptions stated

2. **Implementation (Python Docstrings):**
   - Module-level docstring (purpose, architecture, usage)
   - Function-level docstrings (Args, Returns, Raises, Example)
   - Calculation formulas documented with units
   - Complex logic has inline comments

3. **User Guide:**
   - How to use the agent (API examples)
   - Common use cases (with examples)
   - Troubleshooting guide (common errors and solutions)
   - FAQ (frequently asked questions)

4. **Developer Guide:**
   - How to modify the agent (architecture overview)
   - How to add new tools
   - How to run tests
   - How to deploy

5. **Operations Runbook:**
   - How to monitor the agent (metrics, alerts)
   - How to troubleshoot failures (logs, debugging)
   - How to scale (resource requirements, scaling strategy)
   - How to upgrade (version compatibility, migration guide)

### Evaluation Methodology

**Automated Checks:**
- Specification YAML schema validation
- Docstring coverage (all functions have docstrings)
- Markdown linting (documentation files)

**Manual Review:**
- Technical Writer reviews for clarity and completeness
- Product Manager reviews for user-friendliness
- DevOps Engineer reviews operations runbook

### Sign-Off

**Technical Writer:** _________________ Date: _______

---

## Dimension 6: Compliance & Security

### Objective

Validate that the agent correctly implements regulatory requirements, produces audit-ready outputs, and has no security vulnerabilities.

### Pass Criteria

**COMPLIANCE (100% Required):**

1. **Regulatory Methodology:**
   - Calculations match regulatory formulas exactly (e.g., CBAM, EPA)
   - All mandatory data fields collected
   - Report output matches regulatory templates
   - Regulatory standards cited in provenance

2. **Audit Trail:**
   - Complete provenance tracking (inputs → calculation → outputs)
   - Audit trail reconstructs calculation from raw inputs
   - All assumptions and data sources documented
   - Timestamps for regulatory deadline tracking

3. **Legal Review:**
   - Legal team confirms regulatory interpretation
   - Data privacy compliance (GDPR, CCPA)
   - Terms of Service compliance

**SECURITY (Zero P0/P1 Vulnerabilities Required):**

1. **No Secrets in Code:**
   - No API keys hardcoded
   - No credentials in configuration files
   - Secrets injected via environment variables

2. **Input Validation:**
   - All parameters validated (type, range, format)
   - SQL injection prevention (parameterized queries)
   - XSS prevention (input sanitization)
   - Command injection prevention (no os.system calls)

3. **Access Control:**
   - RBAC enforcement (role-based access control)
   - Authentication required (JWT tokens)
   - Authorization checks on all sensitive operations

4. **Security Scan:**
   - Bandit scan: No P0/P1 vulnerabilities
   - Dependency scan: No critical CVEs
   - SAST (Static Application Security Testing): No critical issues

### Evaluation Methodology

**Automated Checks:**
- Bandit security scan
- Dependency vulnerability scan (safety, pip-audit)
- SAST scan (SonarQube)

**Manual Review:**
- Legal team reviews regulatory compliance
- InfoSec team reviews security posture
- Penetration testing (if handling sensitive data)

### Accuracy Thresholds by Domain

| Domain | Metric | Threshold | Rationale |
|--------|--------|-----------|-----------|
| Energy Calculations | Efficiency (%) | ±1% | Engineering practice standard |
| Emissions Calculations | CO2e (tonnes) | ±3% | EPA reporting tolerance |
| Financial Calculations | NPV, Payback | ±5% | Investment analysis standard |
| Thermodynamic Calculations | Heat transfer (MMBtu) | ±2% | ASME standard |
| Solar Resource | Solar fraction (%) | ±5% | Weather variability |

### Sign-Off

**Legal Counsel:** _________________ Date: _______
**InfoSec Lead:** _________________ Date: _______

---

## Dimension 7: Deployment Readiness

### Objective

Validate that the agent is ready for Kubernetes deployment with health checks, resource limits, and monitoring.

### Pass Criteria

**DEPLOYMENT (100% Required):**

1. **Deployment Pack:**
   - Kubernetes manifests (deployment.yaml, service.yaml)
   - ConfigMaps for agent configuration
   - Secrets for sensitive data (API keys)
   - Resource limits (CPU, memory)
   - Liveness/readiness probes

2. **Health Checks:**
   - `/health` endpoint (liveness probe)
   - `/ready` endpoint (readiness probe)
   - Dependency health checks (other agents, databases)

3. **Resource Requirements:**
   - CPU request/limit defined
   - Memory request/limit defined
   - Disk storage requirements (if applicable)

4. **Scaling Strategy:**
   - Horizontal Pod Autoscaler (HPA) configuration
   - Target CPU/memory thresholds
   - Min/max replica count

5. **Monitoring:**
   - Prometheus metrics endpoint
   - Key metrics exposed (latency, cost, errors)
   - Grafana dashboard created

### Example Deployment Pack

```yaml
# kubernetes/boiler-efficiency-optimizer/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: boiler-efficiency-optimizer
  namespace: greenlang-agents
spec:
  replicas: 3
  selector:
    matchLabels:
      app: boiler-efficiency-optimizer
  template:
    metadata:
      labels:
        app: boiler-efficiency-optimizer
        version: v1.0.0
    spec:
      containers:
      - name: agent
        image: greenlang/boiler-efficiency-optimizer:1.0.0
        ports:
        - containerPort: 8080
          name: http
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 2000m
            memory: 2Gi
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: anthropic-api-key
              key: api-key
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: boiler-efficiency-optimizer
  namespace: greenlang-agents
spec:
  selector:
    app: boiler-efficiency-optimizer
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: boiler-efficiency-optimizer-hpa
  namespace: greenlang-agents
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: boiler-efficiency-optimizer
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Evaluation Methodology

**Automated Checks:**
- Kubernetes YAML validation
- Deployment pack completeness check

**Manual Review:**
- DevOps Engineer reviews deployment configuration
- SRE reviews resource requirements and scaling strategy

### Sign-Off

**DevOps Engineer:** _________________ Date: _______

---

## Dimension 8: Exit Bar Criteria

### Objective

Validate that the agent meets all performance, quality, and business exit bar criteria for production deployment.

### Pass Criteria

**PERFORMANCE (100% Required):**
- P50 latency <2.0 seconds
- P95 latency <4.0 seconds
- P99 latency <6.0 seconds
- Cost per analysis <$0.15
- Success rate >99%

**QUALITY (100% Required):**
- Test coverage >85%
- Golden tests: 25+ scenarios, 100% passing
- Determinism: 100% reproducibility
- Code quality score >8.0/10

**BUSINESS (100% Required):**
- Market opportunity documented (TAM, target customers)
- Carbon impact quantified (tonnes CO2e/year reduction potential)
- ROI analysis completed (payback period, NPV)
- Competitive differentiation articulated

### Sign-Off

**QA Engineer:** _________________ Date: _______

---

## Dimension 9: Integration & Coordination

### Objective

Validate that the agent integrates correctly with other agents and external systems.

### Pass Criteria

**INTEGRATION (100% Required):**

1. **Agent-to-Agent Integration:**
   - Identifies all dependent agents
   - Data exchange contracts defined
   - Integration tests with mocked dependencies passing
   - Graceful degradation if dependencies unavailable

2. **External System Integration:**
   - API integrations documented (endpoints, authentication, rate limits)
   - Error handling for external API failures
   - Retry logic with exponential backoff
   - Circuit breaker pattern implemented

3. **Data Pipeline Integration:**
   - Data sources documented (databases, APIs, files)
   - Data quality validation
   - Error handling for missing/invalid data

### Sign-Off

**Integration Engineer:** _________________ Date: _______

---

## Dimension 10: Business Impact & Metrics

### Objective

Validate that the agent addresses a significant business opportunity with quantified impact.

### Pass Criteria

**BUSINESS IMPACT (100% Required):**

1. **Market Opportunity:**
   - Total Addressable Market (TAM) quantified
   - Target customers identified
   - Competitive landscape analyzed

2. **Carbon Impact:**
   - CO2e reduction potential quantified (tonnes/year)
   - Climate science validation of impact calculation

3. **Economic Impact:**
   - Cost savings quantified ($/year)
   - Payback period calculated
   - ROI analysis completed

### Sign-Off

**Product Manager:** _________________ Date: _______

---

## Dimension 11: Operational Excellence

### Objective

Validate that the agent has production-ready operational capabilities (monitoring, logging, alerting).

### Pass Criteria

**OPERATIONS (100% Required):**

1. **Monitoring:**
   - Prometheus metrics endpoint
   - Key metrics exposed (latency, cost, errors, throughput)
   - Grafana dashboard created

2. **Logging:**
   - Structured JSON logs
   - Request ID tracing
   - Error stack traces
   - Log aggregation (ELK stack)

3. **Alerting:**
   - High latency alert (P95 >5s)
   - High error rate alert (>2%)
   - High cost alert (>$0.20)
   - Dependency failure alert

4. **Reliability:**
   - SLA defined (e.g., 99.9% uptime)
   - Error budget calculated
   - Incident response playbook created

### Sign-Off

**SRE:** _________________ Date: _______

---

## Dimension 12: Continuous Improvement

### Objective

Validate that the agent has a roadmap for continuous improvement and feedback mechanisms.

### Pass Criteria

**CONTINUOUS IMPROVEMENT (100% Required):**

1. **Roadmap:**
   - v1.1+ features planned
   - Technical debt documented
   - Performance optimization opportunities identified

2. **Feedback Mechanisms:**
   - Customer feedback collection
   - Error tracking (Sentry)
   - Field performance monitoring
   - Quarterly retrospectives

3. **Maintenance Plan:**
   - Dependency update schedule
   - Security patching process
   - Quarterly re-evaluation
   - Annual re-certification

### Sign-Off

**Engineering Manager:** _________________ Date: _______

---

## Regulatory Correctness Standards

### CBAM (Carbon Border Adjustment Mechanism)

**Standards:**
- Commission Implementing Regulation (EU) 2023/1773
- CBAM Transitional Period (2023-2025)
- CBAM Definitive Period (2026+)

**Requirements:**
- Embedded emissions calculation per Annex IV
- All mandatory data fields collected (product category, production process, emissions)
- Quarterly reporting format compliance
- CBAM certificate calculation methodology

### CSRD (Corporate Sustainability Reporting Directive)

**Standards:**
- CSRD (EU) 2022/2464
- ESRS (European Sustainability Reporting Standards)
- Double materiality assessment

**Requirements:**
- All ESRS data points collected
- Materiality assessment documented
- Audit trail for regulatory review
- Report format compliance (XBRL/iXBRL)

### GHG Protocol

**Standards:**
- GHG Protocol Corporate Standard
- GHG Protocol Scope 2 Guidance
- GHG Protocol Scope 3 Standard

**Requirements:**
- Scope 1, 2, 3 emissions calculation
- Organizational boundary definition (equity share, financial control, operational control)
- Emission factor sources documented (EPA, IPCC, DEFRA)
- Base year and recalculation policy

---

## Climate Science Validation Requirements

### Emission Factors

**Authoritative Sources:**
- EPA (US Environmental Protection Agency)
- IPCC (Intergovernmental Panel on Climate Change)
- DEFRA (UK Department for Environment, Food & Rural Affairs)
- EcoInvent (Life Cycle Inventory database)
- IEA (International Energy Agency)

**Requirements:**
- Emission factors match sources within documented uncertainty
- Vintage tracked (year of publication)
- Regional specificity (country/state-level where available)
- Uncertainty ranges documented

### GWP (Global Warming Potential)

**Source:**
- IPCC AR6 (Sixth Assessment Report, 2021)

**Requirements:**
- GWP100 values used (100-year time horizon)
- Fossil vs. non-fossil CH4 distinction (GWP 27.9 vs 29.8)
- Biogenic CO2 accounting (neutral vs. non-neutral)

### Technology Performance

**Validation Sources:**
- NREL (National Renewable Energy Laboratory)
- DOE (US Department of Energy)
- ASHRAE (American Society of Heating, Refrigerating and Air-Conditioning Engineers)
- ASME (American Society of Mechanical Engineers)

**Requirements:**
- Performance curves within ±5% of validation data
- Operating range validated (min/max temperature, pressure, flow rate)
- Part-load performance validated
- Degradation factors documented

---

## Performance Requirements

### Latency Targets

| Percentile | Target | Measurement |
|------------|--------|-------------|
| P50 | <2.0s | Median response time |
| P95 | <4.0s | 95th percentile response time |
| P99 | <6.0s | 99th percentile response time |

### Cost Targets

| Analysis Type | Target | Rationale |
|---------------|--------|-----------|
| Simple (1-2 tools) | <$0.05 | Basic calculation |
| Standard (3-5 tools) | <$0.15 | Multi-tool workflow |
| Complex (6+ tools) | <$0.30 | Comprehensive analysis |

### Throughput Targets

| Load Type | Target | Measurement |
|-----------|--------|-------------|
| Steady state | 100 req/s | Sustained for 1 hour |
| Spike | 500 req/s | Peak for 5 minutes |
| Soak | 50 req/s | Sustained for 24 hours |

---

## Sign-Off Authority Matrix

| Dimension | Primary Sign-Off | Secondary Sign-Off | Approval Level |
|-----------|------------------|-------------------|----------------|
| 1. Specification | Product Manager | Domain Expert | Manager |
| 2. Implementation | Lead Engineer | Senior Engineer | Director |
| 3. Test Coverage | QA Engineer | Test Lead | Manager |
| 4. Determinism | QA Engineer | Data Scientist | Manager |
| 5. Documentation | Technical Writer | Product Manager | Manager |
| 6. Compliance | Legal Counsel | Regulatory Expert | VP/General Counsel |
| 6. Security | InfoSec Lead | Security Architect | VP/CISO |
| 7. Deployment | DevOps Engineer | SRE | Manager |
| 8. Exit Bar | QA Engineer | Engineering Manager | Director |
| 9. Integration | Integration Engineer | Lead Engineer | Manager |
| 10. Business Impact | Product Manager | Business Analyst | VP Product |
| 11. Operations | SRE | DevOps Engineer | Manager |
| 12. Continuous Improvement | Engineering Manager | Product Manager | Director |

**Final Certification Decision:**
- Certification Committee (VP Engineering, General Counsel, Chief Climate Scientist, Product Manager)
- Approval Level: VP+

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-03 | GL-TestEngineer | Initial certification criteria |

**Approved By:**
- VP Engineering: _________________ Date: _______
- General Counsel: _________________ Date: _______
- Chief Climate Scientist: _________________ Date: _______

---

**END OF DOCUMENT**
