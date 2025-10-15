# Deterministic Agents Remediation Plan
## Comprehensive 12-Dimension Compliance Initiative

**Plan Owner:** Head of AI & Climate Intelligence
**Date:** October 13, 2025
**Scope:** 15 deterministic agents requiring full compliance
**Target:** Bring all agents to 12/12 dimension compliance
**Timeline:** 4-6 weeks (parallel execution)

---

## Executive Summary

### Current State
- **15 deterministic agents** in production codebase
- **Average compliance:** 3.2/12 dimensions (27%)
- **Critical gaps:** No specs (14/15), no tests (15/15), low coverage (5-33%)
- **Business impact:** These agents underpin all AI-powered agents

### Target State
- **100% compliance** across all 12 dimensions
- **≥80% test coverage** for all agents
- **Complete specifications** (AgentSpec V2.0)
- **Production-ready** with monitoring and operations

### Strategy
- **Phase 1:** Fix 5 production-ready agents (P0 - Critical)
- **Phase 2:** Complete 10 incomplete agents (P1-P2)
- **Approach:** Parallel sub-agent execution for maximum velocity

---

## Agent Prioritization

### Priority P0 (Critical - Week 1-2)
**5 Production-Ready Agents** - Currently used by AI agents, immediate impact

| Agent | Current Coverage | Dependencies | Priority Reason |
|-------|-----------------|--------------|-----------------|
| **FuelAgent** | 13.79% | FuelAgentAI, industrial agents | Core emissions calculation |
| **CarbonAgent** | 11.94% | CarbonAgentAI, all reporting | GHG Protocol compliance |
| **GridFactorAgent** | 20.24% | GridFactorAgentAI, electricity | Grid emissions critical |
| **RecommendationAgent** | 9.88% | RecommendationAgentAI | Actionable insights |
| **ReportAgent** | 5.17% | ReportAgentAI | Executive reporting |

**Business Impact:** These 5 agents are called by 7 AI agents in production. Fixing them improves the entire ecosystem.

### Priority P1 (High - Week 3-4)
**5 Core Infrastructure Agents**

| Agent | Current Coverage | Status | Priority Reason |
|-------|-----------------|--------|-----------------|
| **BoilerAgent** | 10.13% | Has spec | Spec exists, needs tests |
| **IntensityAgent** | 9.43% | Incomplete | Carbon intensity calculations |
| **ValidatorAgent** | 7.63% | Incomplete | Input validation critical |
| **BuildingProfileAgent** | 13.10% | Incomplete | Building characteristics |
| **BenchmarkAgent** | 9.47% | Incomplete | Performance benchmarking |

### Priority P2 (Medium - Week 5-6)
**5 Specialized Agents**

| Agent | Current Coverage | Status | Priority Reason |
|-------|-----------------|--------|-----------------|
| **SolarResourceAgent** | 28.57% | Incomplete | Solar analysis |
| **LoadProfileAgent** | 33.33% | Incomplete | Energy load modeling |
| **SiteInputAgent** | 33.33% | Incomplete | Site data collection |
| **FieldLayoutAgent** | 24.00% | Incomplete | Solar field design |
| **EnergyBalanceAgent** | 19.57% | Incomplete | Energy balance calcs |

---

## 12-Dimension Compliance Matrix

### Current Status for All 15 Agents

| Dimension | P0 (5 agents) | P1 (5 agents) | P2 (5 agents) | Total Pass |
|-----------|---------------|---------------|---------------|------------|
| **D1: Specification** | 0/5 ❌ | 1/5 ⚠️ | 0/5 ❌ | **1/15 (7%)** |
| **D2: Implementation** | 5/5 ✅ | 5/5 ✅ | 5/5 ✅ | **15/15 (100%)** |
| **D3: Test Coverage** | 0/5 ❌ | 0/5 ❌ | 0/5 ❌ | **0/15 (0%)** |
| **D4: Deterministic** | 5/5 ✅ | 5/5 ✅ | 5/5 ✅ | **15/15 (100%)** |
| **D5: Documentation** | 3/5 ⚠️ | 2/5 ⚠️ | 1/5 ⚠️ | **6/15 (40%)** |
| **D6: Compliance** | 2/5 ⚠️ | 1/5 ⚠️ | 0/5 ❌ | **3/15 (20%)** |
| **D7: Deployment** | 0/5 ❌ | 0/5 ❌ | 0/5 ❌ | **0/15 (0%)** |
| **D8: Exit Bar** | 0/5 ❌ | 0/5 ❌ | 0/5 ❌ | **0/15 (0%)** |
| **D9: Integration** | 5/5 ✅ | 3/5 ⚠️ | 2/5 ⚠️ | **10/15 (67%)** |
| **D10: Business Impact** | 3/5 ⚠️ | 2/5 ⚠️ | 1/5 ⚠️ | **6/15 (40%)** |
| **D11: Operations** | 0/5 ❌ | 0/5 ❌ | 0/5 ❌ | **0/15 (0%)** |
| **D12: Improvement** | 0/5 ❌ | 0/5 ❌ | 0/5 ❌ | **0/15 (0%)** |

**Overall:** 48/180 dimensions passed (27%)

---

## Remediation Strategy

### Phase 1: P0 Agents (Week 1-2)
**Goal:** Bring 5 production-ready agents to 10/12 compliance (83%)

#### Week 1: Specifications + Tests
**Deliverables:**
1. **5 AgentSpec V2.0 YAML files** (D1)
   - FuelAgent.yaml
   - CarbonAgent.yaml
   - GridFactorAgent.yaml
   - RecommendationAgent.yaml
   - ReportAgent.yaml

2. **5 Test suites** (D3)
   - test_fuel_agent.py (20+ tests)
   - test_carbon_agent.py (15+ tests)
   - test_grid_factor_agent.py (18+ tests)
   - test_recommendation_agent.py (25+ tests)
   - test_report_agent.py (30+ tests)

3. **Documentation enhancement** (D5)
   - README for each agent
   - API documentation
   - Usage examples

**Effort:** 80 developer-hours (2 developers × 40 hours)

#### Week 2: Deployment + Operations
**Deliverables:**
1. **Deployment configs** (D7)
   - Docker configs
   - Environment variables
   - Health checks

2. **Operations setup** (D11)
   - Monitoring dashboards
   - Alerting rules
   - Performance tracking

3. **Exit bar validation** (D8)
   - All quality gates passed
   - Security scans clean
   - Performance benchmarks met

**Effort:** 40 developer-hours

**Expected Outcome:** 5 agents at 10/12 dimensions (83%)

---

### Phase 2: P1 Agents (Week 3-4)
**Goal:** Complete 5 infrastructure agents to 9/12 compliance (75%)

#### Week 3: Specifications + Tests
**Deliverables:**
1. **4 AgentSpec V2.0 YAML files** (BoilerAgent already has spec)
2. **5 comprehensive test suites**
3. **Documentation for all agents**

**Effort:** 60 developer-hours

#### Week 4: Integration + Operations
**Deliverables:**
1. **Integration testing** with dependent agents
2. **Deployment configs**
3. **Basic operations setup**

**Effort:** 30 developer-hours

**Expected Outcome:** 5 agents at 9/12 dimensions (75%)

---

### Phase 3: P2 Agents (Week 5-6)
**Goal:** Complete 5 specialized agents to 8/12 compliance (67%)

#### Week 5-6: Full Development Cycle
**Deliverables:**
1. **5 AgentSpec V2.0 YAML files**
2. **5 test suites**
3. **Documentation**
4. **Basic deployment configs**

**Effort:** 60 developer-hours

**Expected Outcome:** 5 agents at 8/12 dimensions (67%)

---

## Detailed Work Breakdown

### Dimension 1: Specification Completeness

**Task:** Create 14 AgentSpec V2.0 YAML files (BoilerAgent already exists)

**Template Pattern:**
```yaml
agent_metadata:
  agent_id: "core/fuel_agent"
  agent_name: "FuelAgent"
  version: "1.0.0"
  domain: "Core"
  category: "Emissions_Calculation"
  complexity: "Medium"
  priority: "P0_Critical"
  base_agent: "BaseAgent"
  status: "Production"

description:
  purpose: |
    Calculates CO2e emissions from fuel consumption using EPA/IPCC emission factors.
    Supports 18 fuel types with regional factors and uncertainty quantification.

  key_capabilities:
    - "Calculate Scope 1 direct emissions from fuel combustion"
    - "Support 18 fuel types (natural gas, diesel, coal, etc.)"
    - "Apply regional emission factors (EPA, IPCC, UK DEFRA)"
    - "Quantify uncertainty and data quality"

tools:
  tool_count: 3
  tools_list:
    - tool_id: "calculate_fuel_emissions"
      name: "calculate_fuel_emissions"
      description: "Calculate CO2e emissions from fuel consumption"
      category: "calculation"
      deterministic: true

      parameters:
        type: "object"
        properties:
          fuel_type:
            type: "string"
            enum: ["natural_gas", "diesel", "gasoline", "coal", "propane"]
          consumption:
            type: "number"
            description: "Fuel consumption quantity"
          units:
            type: "string"
            enum: ["therms", "gallons", "kg", "tons"]
        required: ["fuel_type", "consumption", "units"]

      returns:
        type: "object"
        properties:
          co2e_kg:
            type: "number"
            description: "Total CO2e emissions in kg"
          breakdown:
            type: "object"
            description: "CO2, CH4, N2O breakdown"

      implementation:
        physics_formula: "CO2e = consumption × emission_factor × GWP"
        data_source: "EPA Emission Factors Hub"
        standards: ["EPA 40 CFR Part 98", "ISO 14064-1"]

testing:
  test_coverage_target: 0.85
  test_categories:
    - category: "unit_tests"
      description: "Test emission calculations for all fuel types"
      count: 20
    - category: "integration_tests"
      description: "Test with real emission factor data"
      count: 10
    - category: "determinism_tests"
      description: "Verify reproducibility"
      count: 5
    - category: "boundary_tests"
      description: "Test edge cases (zero, negative, huge values)"
      count: 10
```

**Effort per spec:** 4-6 hours
**Total effort:** 56-84 hours (14 specs)

**Sub-Agent Strategy:** Deploy spec generation agent to create all 14 specs in parallel (estimated 12 hours with AI assistance)

---

### Dimension 3: Test Coverage

**Task:** Create 15 comprehensive test suites

**Test Structure:**
```python
# tests/agents/test_fuel_agent.py

import pytest
from greenlang.agents.fuel_agent import FuelAgent

class TestFuelAgentUnitTests:
    """Unit tests for individual calculations."""

    def test_natural_gas_calculation_exact(self):
        """Test natural gas emission calculation with known values."""
        agent = FuelAgent()
        result = agent.calculate_emissions(
            fuel_type="natural_gas",
            consumption=100,  # therms
            units="therms"
        )

        # EPA factor: 5.3 kg CO2e/therm
        assert result["co2e_kg"] == pytest.approx(530.0, rel=0.01)
        assert result["co2_kg"] == pytest.approx(528.0, rel=0.01)
        assert result["ch4_kg"] == pytest.approx(1.5, rel=0.01)

    def test_all_fuel_types(self, fuel_types):
        """Test all 18 supported fuel types."""
        agent = FuelAgent()
        for fuel_type in fuel_types:
            result = agent.calculate_emissions(
                fuel_type=fuel_type,
                consumption=100,
                units="gallons"
            )
            assert result["co2e_kg"] > 0
            assert "breakdown" in result

class TestFuelAgentIntegration:
    """Integration tests with real data."""

    def test_commercial_building_scenario(self):
        """Test realistic commercial building fuel usage."""
        agent = FuelAgent()

        # Typical office building: 10,000 therms/year natural gas
        result = agent.calculate_emissions(
            fuel_type="natural_gas",
            consumption=10000,
            units="therms"
        )

        # Expected: ~53 metric tons CO2e/year
        assert 50000 < result["co2e_kg"] < 56000

class TestFuelAgentDeterminism:
    """Determinism tests - same input = same output."""

    def test_reproducibility_10_runs(self):
        """Run same calculation 10 times, verify identical results."""
        agent = FuelAgent()

        results = [
            agent.calculate_emissions(
                fuel_type="diesel",
                consumption=500,
                units="gallons"
            )
            for _ in range(10)
        ]

        # All results must be byte-identical
        assert all(r["co2e_kg"] == results[0]["co2e_kg"] for r in results)

class TestFuelAgentBoundary:
    """Boundary and edge case tests."""

    def test_zero_consumption(self):
        """Test zero fuel consumption."""
        agent = FuelAgent()
        result = agent.calculate_emissions(
            fuel_type="natural_gas",
            consumption=0,
            units="therms"
        )
        assert result["co2e_kg"] == 0

    def test_negative_consumption_error(self):
        """Test negative consumption raises error."""
        agent = FuelAgent()
        with pytest.raises(ValueError, match="Consumption must be non-negative"):
            agent.calculate_emissions(
                fuel_type="natural_gas",
                consumption=-100,
                units="therms"
            )

    def test_invalid_fuel_type(self):
        """Test invalid fuel type raises error."""
        agent = FuelAgent()
        with pytest.raises(ValueError, match="Unsupported fuel type"):
            agent.calculate_emissions(
                fuel_type="invalid_fuel",
                consumption=100,
                units="therms"
            )
```

**Test Targets:**
- **P0 agents:** 20-30 tests each, 85%+ coverage
- **P1 agents:** 15-25 tests each, 80%+ coverage
- **P2 agents:** 10-20 tests each, 75%+ coverage

**Effort per agent:** 6-10 hours
**Total effort:** 90-150 hours (15 agents)

**Sub-Agent Strategy:** Deploy test generation agent to create test suites in parallel (estimated 24 hours with AI assistance)

---

### Dimension 5: Documentation

**Task:** Create comprehensive documentation for all agents

**Documentation Structure:**

#### 1. Agent README (per agent)
```markdown
# FuelAgent - Fuel Combustion Emissions Calculator

## Overview
Calculates CO2e emissions from fuel combustion using authoritative emission factors from EPA, IPCC, and UK DEFRA.

## Features
- ✅ 18 fuel types supported
- ✅ Regional emission factors
- ✅ GHG Protocol compliant
- ✅ Uncertainty quantification
- ✅ Data quality indicators

## Quick Start
\```python
from greenlang.agents import FuelAgent

agent = FuelAgent()
result = agent.calculate_emissions(
    fuel_type="natural_gas",
    consumption=1000,  # therms
    units="therms"
)

print(f"CO2e: {result['co2e_kg']} kg")
print(f"Breakdown: {result['breakdown']}")
\```

## API Reference

### calculate_emissions()
Calculate CO2e emissions from fuel consumption.

**Parameters:**
- `fuel_type` (str): Fuel type (see supported fuels below)
- `consumption` (float): Fuel consumption quantity
- `units` (str): Units (therms, gallons, kg, tons)
- `emission_factor_source` (str, optional): "EPA" | "IPCC" | "UK_DEFRA"

**Returns:**
- `co2e_kg` (float): Total CO2e emissions in kg
- `co2_kg` (float): CO2 emissions in kg
- `ch4_kg` (float): CH4 emissions in kg
- `n2o_kg` (float): N2O emissions in kg
- `breakdown` (dict): Detailed breakdown by gas
- `data_quality` (str): Data quality indicator (Tier 1-3)

## Supported Fuels
1. Natural Gas
2. Diesel
3. Gasoline
4. Coal (bituminous, anthracite, lignite)
5. Propane
6. Fuel Oil (#2, #4, #6)
[... 18 total]

## Standards Compliance
- ✅ GHG Protocol Corporate Standard
- ✅ ISO 14064-1:2018
- ✅ EPA 40 CFR Part 98 (Mandatory Reporting)
- ✅ IPCC 2019 Refinement

## Data Sources
- EPA Emission Factors Hub (2024)
- IPCC AR6 GWP values
- UK DEFRA 2024 conversion factors
```

**Effort per agent:** 2-3 hours
**Total effort:** 30-45 hours

---

### Dimension 7: Deployment Readiness

**Task:** Create deployment configurations for all agents

**Deliverables:**

#### 1. Docker Configuration
```dockerfile
# Dockerfile.fuel_agent
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY greenlang/ greenlang/
COPY data/ data/

ENV PYTHONPATH=/app
ENV AGENT_NAME=FuelAgent

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "from greenlang.agents import FuelAgent; FuelAgent().health_check()"

CMD ["python", "-m", "greenlang.agents.fuel_agent"]
```

#### 2. Environment Configuration
```yaml
# config/fuel_agent.yaml
agent:
  name: "FuelAgent"
  version: "1.0.0"
  log_level: "INFO"

data:
  emission_factors_path: "data/emission_factors_registry.yaml"
  cache_enabled: true
  cache_ttl: 3600

performance:
  max_latency_ms: 1000
  timeout_seconds: 5

monitoring:
  metrics_enabled: true
  metrics_port: 9090
  health_check_path: "/health"
```

#### 3. Health Check Implementation
```python
def health_check(self) -> Dict[str, Any]:
    """Agent health status."""
    try:
        # Test emission factor loading
        test_factor = self._get_emission_factor("natural_gas", "EPA")

        # Test calculation
        test_result = self.calculate_emissions(
            fuel_type="natural_gas",
            consumption=1,
            units="therms"
        )

        return {
            "status": "healthy",
            "version": "1.0.0",
            "emission_factors_loaded": len(self._factors),
            "last_calculation": datetime.now().isoformat(),
            "test_passed": test_result["co2e_kg"] > 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
```

**Effort per agent:** 1-2 hours
**Total effort:** 15-30 hours

---

### Dimension 11: Operations

**Task:** Set up monitoring and operations for all agents

**Deliverables:**

#### 1. Monitoring Dashboard (Grafana)
```json
{
  "dashboard": {
    "title": "FuelAgent Metrics",
    "panels": [
      {
        "title": "Calculation Rate",
        "targets": [
          "rate(fuel_agent_calculations_total[5m])"
        ]
      },
      {
        "title": "Latency (p95)",
        "targets": [
          "histogram_quantile(0.95, fuel_agent_latency_seconds)"
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          "rate(fuel_agent_errors_total[5m])"
        ]
      }
    ]
  }
}
```

#### 2. Alerting Rules
```yaml
groups:
  - name: fuel_agent_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(fuel_agent_errors_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "FuelAgent error rate above 5%"

      - alert: HighLatency
        expr: histogram_quantile(0.95, fuel_agent_latency_seconds) > 2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "FuelAgent p95 latency above 2s"
```

#### 3. Performance Tracking
```python
class FuelAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self._metrics = {
            "calculations_total": 0,
            "calculations_success": 0,
            "calculations_error": 0,
            "total_latency_ms": 0,
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get operational metrics."""
        return {
            "agent": "FuelAgent",
            "calculations": {
                "total": self._metrics["calculations_total"],
                "success_rate": (
                    self._metrics["calculations_success"] /
                    max(self._metrics["calculations_total"], 1)
                ),
                "error_rate": (
                    self._metrics["calculations_error"] /
                    max(self._metrics["calculations_total"], 1)
                ),
            },
            "performance": {
                "avg_latency_ms": (
                    self._metrics["total_latency_ms"] /
                    max(self._metrics["calculations_total"], 1)
                ),
            }
        }
```

**Effort:** 20 hours (shared across all agents)

---

## Resource Requirements

### Personnel
- **2 Senior Python Developers** (Specs, Tests, Documentation)
- **1 DevOps Engineer** (Deployment, Monitoring)
- **1 QA Engineer** (Test validation, Coverage)
- **1 Technical Writer** (Documentation polish)

### Timeline
- **Phase 1 (P0):** 2 weeks (5 agents)
- **Phase 2 (P1):** 2 weeks (5 agents)
- **Phase 3 (P2):** 2 weeks (5 agents)
- **Total:** 6 weeks

### Effort Breakdown
| Task | Effort (hours) | % of Total |
|------|---------------|------------|
| Specifications | 56-84 | 20% |
| Test Development | 90-150 | 35% |
| Documentation | 30-45 | 12% |
| Deployment Configs | 15-30 | 7% |
| Operations Setup | 20 | 5% |
| Integration Testing | 40 | 10% |
| Code Review & QA | 30 | 8% |
| **Total** | **281-399** | **100%** |

**Average:** 340 hours = 8.5 weeks @ 1 FTE

With **2 developers in parallel:** 4.25 weeks = **1 month**

---

## Success Criteria

### Phase 1 Completion (P0 Agents)
- ✅ 5 agents with complete AgentSpec V2.0 YAML
- ✅ 5 agents with ≥85% test coverage
- ✅ All tests passing (0 failures)
- ✅ Documentation complete (README, API docs)
- ✅ Deployment configs ready
- ✅ Monitoring dashboards live
- ✅ **Target:** 10/12 dimensions passed (83%)

### Phase 2 Completion (P1 Agents)
- ✅ 5 additional agents compliant
- ✅ Integration testing with dependent agents
- ✅ **Target:** 9/12 dimensions passed (75%)

### Phase 3 Completion (P2 Agents)
- ✅ All 15 agents compliant
- ✅ Overall test coverage: 80%+
- ✅ **Target:** 8/12 dimensions passed (67%)

### Final State
- **15/15 agents** with specifications ✅
- **15/15 agents** with comprehensive tests ✅
- **Overall coverage:** 80%+ ✅
- **All agents** deployed with monitoring ✅
- **Average compliance:** 9.5/12 dimensions (79%)

---

## Risk Mitigation

### Risk 1: Test Coverage Target Not Met
**Probability:** Medium
**Impact:** High
**Mitigation:**
- Start with most critical agents (P0)
- Use test generation tools to accelerate
- Focus on unit tests first (easier to achieve coverage)

### Risk 2: Specification Quality Issues
**Probability:** Low
**Impact:** Medium
**Mitigation:**
- Use Agent 1-12 specs as templates
- Run validation script on all specs
- Peer review before implementation

### Risk 3: Timeline Slippage
**Probability:** Medium
**Impact:** Medium
**Mitigation:**
- Parallel execution with 2 developers
- Use AI-assisted code/test generation
- Prioritize P0 agents, defer P2 if needed

---

## Next Actions

### Immediate (This Week)
1. **Approve remediation plan** (this document)
2. **Assign resources** (2 developers, 1 DevOps, 1 QA)
3. **Deploy spec generation agent** for P0 agents
4. **Deploy test generation agent** for P0 agents
5. **Set up project tracking** (GitHub issues/project board)

### Week 1
1. Generate 5 specifications (P0 agents)
2. Create 5 test suites
3. Achieve 85%+ coverage for FuelAgent (validate approach)

### Week 2
1. Complete testing for all P0 agents
2. Create deployment configs
3. Set up monitoring dashboards
4. Validate 10/12 compliance for P0 agents

---

## Tracking & Reporting

### Daily Standup Questions
1. Which agents progressed yesterday?
2. What dimension(s) were completed?
3. Any blockers?

### Weekly Status Report
- Agents completed this week
- Dimensions passed (breakdown)
- Test coverage progress
- Blockers and resolutions

### Metrics Dashboard
- **Compliance Score:** X/180 dimensions (15 agents × 12 dimensions)
- **Test Coverage:** X% overall
- **Agents Complete:** X/15
- **On Track for Timeline:** Yes/No

---

## Conclusion

This remediation plan provides a **systematic, measurable approach** to bringing all 15 deterministic agents to full compliance with the 12-dimension standard. By using **parallel execution** and **AI-assisted development**, we can complete this work in **4-6 weeks** instead of 15 weeks sequential.

**Expected Outcome:**
- **15 fully compliant agents**
- **80%+ test coverage**
- **Production-ready with monitoring**
- **Foundation for 84-agent ecosystem**

**Business Impact:**
- Improved reliability for 7 AI agents
- Faster development of new agents
- Compliance-ready for enterprise customers
- Reduced technical debt

---

**Plan Status:** READY FOR EXECUTION
**Approval Required:** Engineering Lead, Head of Product
**Start Date:** Week 11 (immediately)
**Completion Date:** Week 16 (6 weeks)

---

**Document Version:** 1.0
**Last Updated:** 2025-10-13
**Owner:** Head of AI & Climate Intelligence
