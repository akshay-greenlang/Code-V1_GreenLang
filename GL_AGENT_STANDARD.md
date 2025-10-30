# GreenLang Agent Development Standard (GL-ADS)

**Version:** 1.0.0
**Status:** Official Standard
**Date:** October 23, 2025
**Authority:** Head of Industrial Agents, AI & Climate Intelligence

---

## TABLE OF CONTENTS

1. [Overview](#1-overview)
2. [12-Dimension Production Readiness Framework](#2-12-dimension-production-readiness-framework)
3. [Specification Standards](#3-specification-standards)
4. [Implementation Standards](#4-implementation-standards)
5. [Test Suite Standards](#5-test-suite-standards)
6. [Documentation Standards](#6-documentation-standards)
7. [Deployment Standards](#7-deployment-standards)
8. [Quality Metrics & Exit Criteria](#8-quality-metrics--exit-criteria)
9. [Deliverables Checklist](#9-deliverables-checklist)
10. [Review & Approval Process](#10-review--approval-process)
11. [Templates](#11-templates)
12. [Appendix: Phase 2A Reference Agents](#12-appendix-phase-2a-reference-agents)

---

## 1. OVERVIEW

### 1.1 Purpose

This document defines the **official standards, patterns, and requirements** for developing GreenLang AI agents. All agents must comply with these standards to ensure:

- **Consistency:** Uniform architecture and quality across all agents
- **Determinism:** Reproducible, auditable AI outputs
- **Production Readiness:** Enterprise-grade reliability and performance
- **Maintainability:** Clear patterns for long-term evolution
- **Compliance:** Security, privacy, and regulatory requirements

### 1.2 Scope

This standard applies to **all GreenLang AI agents** including:
- Domain-specific agents (Industrial, Buildings, Transportation, Agriculture, Energy)
- Master coordinators and orchestrators
- Specialized technology agents
- Supporting utility agents

### 1.3 Authority

Compliance with GL-ADS is **mandatory** for all agent development. Exceptions require written approval from the Head of Industrial Agents.

### 1.4 Document Conventions

- **MUST**: Absolute requirement
- **SHOULD**: Strong recommendation
- **MAY**: Optional feature
- ✅ = Pass/Complete
- ⚠️ = Warning/Attention Required
- 🔴 = Fail/Incomplete

---

## 2. 12-DIMENSION PRODUCTION READINESS FRAMEWORK

Every GreenLang agent **MUST** pass all 12 dimensions before production deployment. This framework is derived from Phase 2A agents (#1, #2, #3, #4, #12) that collectively represent 45,195 lines of validated production code.

### Dimension 1: Specification Completeness

**Status Required:** PASS ✅

**Requirements:**
- **File Location:** `specs/domain<N>_<domain_name>/<category>/agent_<NNN>_<agent_name>.yaml`
- **Format:** YAML (GreenLang Agent Specification v2.0)
- **Size Target:** 800-2,900 lines (median: 1,400 lines)
- **Sections:** 11/11 mandatory sections complete
- **Tools:** 7-10 comprehensive tools defined
- **AI Configuration:**
  - `temperature: 0.0` (deterministic)
  - `seed: 42` (reproducible)
  - `budget_usd: $0.10 - $2.00` (based on complexity)

**Mandatory Sections:**
1. `agent_metadata` - ID, name, version, domain, category, priority
2. `description` - Purpose, strategic context, key capabilities, dependencies
3. `tools` - Complete tool definitions with parameters and returns
4. `ai_configuration` - Deterministic settings, budget, model selection
5. `input_schema` - Agent input validation
6. `output_schema` - Agent output structure
7. `business_impact` - Market size, carbon impact, ROI metrics
8. `technical_standards` - Industry standards (ISO, ASME, ASHRAE, etc.)
9. `compliance` - Security, privacy, regulatory requirements
10. `deployment` - Resource requirements, dependencies
11. `version_history` - Changelog and evolution

**Tool Requirements per Tool:**
- Unique `tool_id` and descriptive `name`
- `deterministic: true` flag
- Complete `parameters` schema with types, units, ranges
- Complete `returns` schema with expected outputs
- `implementation` section with physics formulas and calculation methods
- `standards` references (e.g., AHRI 540, ASME PTC 4.1)
- `example` with input/output
- `validation` criteria and accuracy targets

**Quality Standards:**
- **Industry Standards:** 4-7 recognized standards referenced
- **Dependencies:** All agent dependencies explicitly declared
- **Market Analysis:** Clear market size and carbon impact
- **Technical Depth:** Engineering-grade specifications with formulas

**Reference Achievements:**
- Agent #12: 2,848 lines (LARGEST - master orchestrator)
- Agent #2: 1,427 lines (LARGEST specification among single-tech agents)
- Agent #3: 1,419 lines (comprehensive thermodynamics)
- Agent #4: 1,394 lines (comprehensive heat transfer)
- Agent #1: 856 lines (focused solar thermal)

---

### Dimension 2: Code Implementation

**Status Required:** PASS ✅

**Requirements:**
- **File Location:** `greenlang/agents/<agent_name>_agent_ai.py`
- **Size Target:** 1,300-2,200 lines (median: 1,700 lines)
- **Architecture:** Tool-first design with ChatSession orchestration
- **Code Quality:** Production-grade with comprehensive error handling

**Mandatory Architecture:**

```python
"""
<Agent Name> - <One-line description>

Comprehensive module docstring (40-150 lines) covering:
- Thermodynamic/domain theory
- Key equations and calculation methods
- Standards compliance
- Usage examples
"""

from greenlang.core.chat_session import ChatSession
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import numpy as np  # For numerical calculations

# Constants and databases
class <Domain>Properties:
    """Domain-specific property databases"""
    # Temperature-dependent properties
    # Material characteristics
    # Technology performance curves

class <Agent>Config(BaseModel):
    """Configuration schema with validation"""
    param1: float = Field(..., ge=0, description="Parameter description")
    # All configurable parameters

class <Agent>AI:
    """
    <Agent Name> - Detailed class docstring

    Attributes:
        session (ChatSession): AI orchestration
        config (<Agent>Config): Configuration

    Methods:
        tool_1(...) -> Dict: Description
        tool_2(...) -> Dict: Description
        # All tool methods
    """

    def __init__(self, config: <Agent>Config):
        self.config = config
        self.session = ChatSession(
            agent_id="agent_<NNN>_<name>",
            temperature=0.0,
            seed=42,
            deterministic=True
        )

    def tool_1(self, **kwargs) -> Dict[str, Any]:
        """
        Tool 1 description

        Args:
            param1 (type): Description with units
            param2 (type): Description with units

        Returns:
            Dict containing:
                - result1 (type): Description
                - result2 (type): Description

        Raises:
            ValueError: If inputs invalid

        Example:
            >>> result = agent.tool_1(param1=100, param2=50)
            >>> result['result1']
            75.5
        """
        # Input validation
        if param1 <= 0:
            raise ValueError("param1 must be positive")

        # Deterministic calculation
        result1 = <calculation>

        # Return structured output
        return {
            "result1": result1,
            "result2": result2,
            "deterministic": True,
            "provenance": {
                "tool": "tool_1",
                "method": "calculation_method",
                "standard": "STANDARD_NAME"
            }
        }

    # Additional tools...

    def analyze(self, user_query: str) -> Dict[str, Any]:
        """Main agent entry point with AI orchestration"""
        prompt = f"""You are {self.config.agent_name}.

        User query: {user_query}

        Available tools: {self.list_tools()}

        Analyze the query and use tools to provide comprehensive analysis.
        """

        response = self.session.run(
            prompt=prompt,
            tools=self.get_tools(),
            budget_usd=self.config.budget_usd
        )

        return response
```

**Implementation Requirements:**

1. **Deterministic Tools:** All calculations in tool methods (zero LLM math)
2. **Input Validation:** Pydantic models or explicit checks
3. **Error Handling:** Comprehensive try/except with informative messages
4. **Type Hints:** All parameters and returns fully typed
5. **Docstrings:** Module, class, and method docstrings with examples
6. **Constants:** Domain-specific databases and property tables
7. **Unit Consistency:** Clear units in docstrings (SI or Imperial with conversion)
8. **Provenance:** All tool outputs include metadata (method, standard, deterministic flag)
9. **Standards Compliance:** Implement industry-standard calculation methods
10. **Numerical Stability:** Handle edge cases (divide by zero, log of negative, etc.)

**Code Quality Standards:**
- **Modularity:** Clear separation of concerns (tools, databases, orchestration)
- **Readability:** Self-documenting code with clear variable names
- **Performance:** Efficient algorithms (avoid nested loops where possible)
- **Security:** No hardcoded credentials, API keys, or secrets
- **Maintainability:** Code organized for easy updates and extensions

**Reference Achievements:**
- Agent #12: 2,178 lines (LARGEST - master orchestrator)
- Agent #3: 1,872 lines (LARGEST single-tech - comprehensive thermodynamics)
- Agent #4: 1,831 lines (comprehensive heat transfer modeling)
- Agent #2: 1,610 lines (retrofit-focused boiler analysis)
- Agent #1: 1,373 lines (focused solar thermal)

---

### Dimension 3: Test Coverage

**Status Required:** PASS ✅

**Requirements:**
- **File Location:** `tests/agents/test_<agent_name>_agent_ai.py`
- **Size Target:** 900-1,600 lines (median: 1,400 lines)
- **Test Count:** 40+ test methods (50+ recommended)
- **Coverage:** **≥85%** (80% minimum, 85%+ achieved by Phase 2A agents)
- **Test Categories:** 6 required categories

**Mandatory Test Categories:**

#### 1. Unit Tests (25-30 tests)
Test individual tool implementations in isolation.

**Requirements:**
- 2-4 tests per tool (8 tools × 3 = 24 tests minimum)
- Cover typical use cases
- Test calculation accuracy
- Verify output structure

**Example:**
```python
def test_tool_1_typical_case(self):
    """Test Tool 1 with typical industrial parameters"""
    agent = AgentAI(config=default_config)
    result = agent.tool_1(param1=100, param2=50)

    assert result["result1"] > 0
    assert result["deterministic"] == True
    assert "provenance" in result
```

#### 2. Integration Tests (6-10 tests)
Test full agent execution with AI orchestration.

**Requirements:**
- Full analyze() method execution
- Multi-tool workflows
- Error handling (invalid inputs, missing data)
- Health check endpoint

**Example:**
```python
def test_full_agent_analysis(self):
    """Test complete agent workflow"""
    agent = AgentAI(config=default_config)
    result = agent.analyze("Analyze heat pump for 160°F process")

    assert "recommendations" in result
    assert result["status"] == "success"
```

#### 3. Determinism Tests (3-5 tests)
Verify reproducibility across multiple runs.

**Requirements:**
- Run same input 3 times
- Verify byte-identical outputs (or floating point within tolerance)
- Test across different tools
- Verify seed effectiveness

**Example:**
```python
def test_determinism_tool_1(self):
    """Verify Tool 1 produces identical results across runs"""
    agent = AgentAI(config=default_config)

    results = [agent.tool_1(param1=100, param2=50) for _ in range(3)]

    # All results should be identical
    assert results[0] == results[1] == results[2]
```

#### 4. Boundary Tests (5-10 tests)
Test edge cases and error conditions.

**Requirements:**
- Zero/negative inputs
- Extreme values (very high/low temperatures, pressures)
- Invalid combinations
- Missing required parameters
- Out-of-range values

**Example:**
```python
def test_tool_1_zero_input(self):
    """Test Tool 1 handles zero input gracefully"""
    agent = AgentAI(config=default_config)

    with pytest.raises(ValueError, match="param1 must be positive"):
        agent.tool_1(param1=0, param2=50)
```

#### 5. Domain-Specific Validation Tests (4-8 tests)
Test physics, thermodynamics, or domain-specific correctness.

**Requirements:**
- Energy balance (Q_in = Q_out)
- Thermodynamic laws (Exergy < Energy, Carnot limits)
- Efficiency bounds (0 < η < 1)
- Material property consistency
- Financial metric cross-validation (NPV vs IRR)

**Example:**
```python
def test_energy_balance(self):
    """Verify energy balance: Q_hot = Q_cold"""
    agent = AgentAI(config=default_config)
    result = agent.calculate_heat_transfer(...)

    q_hot = result["heat_released"]
    q_cold = result["heat_absorbed"]

    assert abs(q_hot - q_cold) / q_hot < 0.01  # Within 1%
```

#### 6. Performance Tests (2-4 tests)
Test latency and cost requirements.

**Requirements:**
- Latency: <3-4s per analysis (varies by complexity)
- Cost: <$0.10-$2.00 per analysis (varies by agent)
- Memory: <512MB RAM
- CPU: <1 core

**Example:**
```python
def test_latency_requirement(self):
    """Verify agent meets <3s latency requirement"""
    agent = AgentAI(config=default_config)

    import time
    start = time.time()
    result = agent.analyze("Quick analysis query")
    duration = time.time() - start

    assert duration < 3.0, f"Analysis took {duration:.2f}s (max 3.0s)"
```

**Testing Framework:**
```python
import pytest
import unittest
from greenlang.agents.<agent_name>_agent_ai import AgentAI, AgentConfig

class TestAgentAI(unittest.TestCase):
    """Comprehensive test suite for Agent AI"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = AgentConfig(
            agent_name="TestAgent",
            budget_usd=0.10
        )
        self.agent = AgentAI(config=self.config)

    # Unit tests
    def test_tool_1_typical(self): ...
    def test_tool_2_typical(self): ...
    # ... all unit tests

    # Integration tests
    def test_full_analysis(self): ...
    def test_error_handling(self): ...
    # ... all integration tests

    # Determinism tests
    def test_determinism_tool_1(self): ...
    # ... all determinism tests

    # Boundary tests
    def test_zero_input(self): ...
    def test_negative_input(self): ...
    # ... all boundary tests

    # Domain validation tests
    def test_energy_balance(self): ...
    def test_efficiency_bounds(self): ...
    # ... all validation tests

    # Performance tests
    def test_latency(self): ...
    def test_cost(self): ...
```

**Coverage Measurement:**
```bash
# Run tests with coverage
pytest tests/agents/test_<agent>_agent_ai.py --cov=greenlang/agents/<agent>_agent_ai --cov-report=html

# Verify coverage ≥85%
coverage report
```

**Reference Achievements:**
- Agent #1: 1,538 lines (LARGEST test suite), 45+ tests, 85%+ coverage
- Agent #3: 1,531 lines, 54+ tests (MOST tests), 85%+ coverage
- Agent #2: 1,431 lines, 50+ tests, 85%+ coverage
- Agent #4: 1,142 lines, 50+ tests, 85%+ coverage
- Agent #12: 925 lines, 40+ tests, 80%+ coverage

---

### Dimension 4: Deterministic AI Guarantees

**Status Required:** PASS ✅

**Critical Requirement:** ALL agents MUST be 100% deterministic and reproducible.

**Configuration Requirements:**
```yaml
ai_configuration:
  temperature: 0.0  # MANDATORY - No randomness
  seed: 42          # MANDATORY - Reproducible initialization
  deterministic: true
```

**Implementation Requirements:**

1. **Zero LLM Math:** All calculations in deterministic tool functions
   - ❌ WRONG: Asking LLM "What is 2+2?"
   - ✅ CORRECT: Python function `def add(a, b): return a + b`

2. **Tool-First Architecture:** All domain logic in tools, not prompts
   - ❌ WRONG: "Calculate the COP using Carnot method"
   - ✅ CORRECT: `def calculate_cop(T_hot, T_cold): return T_hot / (T_hot - T_cold)`

3. **Provenance Tracking:** All outputs include deterministic flag
```python
return {
    "result": calculated_value,
    "deterministic": True,
    "provenance": {
        "tool": "calculate_cop",
        "method": "Carnot efficiency",
        "standard": "AHRI 540",
        "timestamp": "2025-10-23T10:30:00Z"
    }
}
```

4. **Determinism Verification:** Test suite MUST include determinism tests
```python
def test_determinism(self):
    """Verify identical outputs across 3 runs"""
    results = [agent.analyze(query) for _ in range(3)]
    assert results[0] == results[1] == results[2]
```

**Why Determinism Matters:**
- **Auditability:** Regulatory compliance requires reproducible decisions
- **Debugging:** Reproduce issues in production
- **Trust:** Users trust consistent recommendations
- **Testing:** Reliable test assertions
- **Compliance:** Financial and safety decisions must be repeatable

**Reference:** All Phase 2A agents (1, 2, 3, 4, 12) achieve 100% determinism.

---

### Dimension 5: Documentation Completeness

**Status Required:** PASS ✅

**Requirements:**

#### 1. Module Docstring (40-150 lines)
Comprehensive overview at top of implementation file.

**Required Content:**
- Agent purpose and capabilities
- Thermodynamic/domain theory foundations
- Key equations and calculation methods
- Standards compliance (ASME, ISO, ASHRAE, etc.)
- Usage examples
- Dependencies

**Example:**
```python
"""
IndustrialHeatPumpAgent_AI - Industrial Heat Pump Analysis and Optimization

This agent performs comprehensive thermodynamic analysis of industrial heat pump
systems for process heat applications. It covers air-source, water-source,
ground-source, and waste heat recovery heat pumps.

THERMODYNAMIC FOUNDATION:
-----------------------
Heat pumps transfer thermal energy from low to high temperature using mechanical
work (electricity). The Coefficient of Performance (COP) quantifies efficiency:

    COP = Q_delivered / W_compressor

The theoretical maximum COP is the Carnot COP:

    COP_carnot = T_hot / (T_hot - T_cold)  [in absolute temperature]

Real heat pumps achieve 40-55% of Carnot efficiency (Carnot efficiency).

CALCULATION METHODS:
------------------
1. COP Calculation: Carnot method with empirical corrections per AHRI 540
2. Capacity: Q = m_dot × (h_discharge - h_suction)
3. Part-Load: Performance degradation at reduced capacity
4. Economics: LCOH = (CAPEX × CRF + OPEX) / Annual_Heat_Delivered

STANDARDS COMPLIANCE:
-------------------
- AHRI 540: Performance rating of positive displacement refrigerant compressors
- ISO 13612: Heating and cooling systems in buildings
- ASHRAE Handbook: HVAC Systems and Equipment

USAGE EXAMPLE:
------------
    >>> agent = IndustrialHeatPumpAgent_AI(config)
    >>> result = agent.calculate_cop(
    ...     source_temp=50,  # °F
    ...     sink_temp=160,   # °F
    ...     compressor_type="screw"
    ... )
    >>> print(f"COP: {result['cop']:.2f}")
    COP: 3.45
"""
```

#### 2. Class Docstring (20-50 lines)
Detailed class documentation.

**Required Content:**
- Class purpose
- Key attributes
- Available methods (tool summary)
- Configuration requirements
- Example instantiation

#### 3. Method/Tool Docstrings (10-30 lines each)
Every tool method MUST have comprehensive docstring.

**Required Content:**
- Purpose and description
- Parameters with types, units, ranges
- Returns with types and structure
- Raises (exceptions)
- Example usage
- Physics formulas (if applicable)

**Example:**
```python
def calculate_heat_pump_cop(
    self,
    source_temperature_f: float,
    sink_temperature_f: float,
    compressor_type: str,
    refrigerant: str,
    part_load_ratio: float = 1.0
) -> Dict[str, Any]:
    """
    Calculate heat pump Coefficient of Performance using Carnot efficiency method.

    Uses thermodynamic principles to determine heating COP based on temperature
    lift and compressor technology. Implements AHRI 540 methodology with empirical
    corrections for real-world performance.

    Args:
        source_temperature_f (float): Heat source temperature in °F.
            Range: -20 to 150°F (air, water, ground, or waste heat)
        sink_temperature_f (float): Heat delivery temperature in °F.
            Range: 80 to 250°F (process requirement)
        compressor_type (str): Compressor technology.
            Options: "scroll", "screw", "centrifugal", "reciprocating"
        refrigerant (str): Refrigerant type.
            Options: "R134a", "R410A", "R1234yf", "R744_CO2", "ammonia_R717"
        part_load_ratio (float, optional): Operating load fraction. Default: 1.0
            Range: 0.2 to 1.0

    Returns:
        Dict[str, Any]: Heat pump performance metrics
            - cop_heating (float): Coefficient of Performance
            - carnot_cop (float): Theoretical maximum COP
            - carnot_efficiency (float): Actual COP / Carnot COP (0.40-0.55)
            - temperature_lift_f (float): Temperature difference (°F)
            - compressor_power_kw (float): Electrical power consumption (kW)
            - heat_output_btu_hr (float): Heat delivery rate (Btu/hr)
            - deterministic (bool): True
            - provenance (Dict): Calculation metadata

    Raises:
        ValueError: If source_temperature >= sink_temperature (thermodynamically impossible)
        ValueError: If temperatures outside valid ranges
        ValueError: If invalid compressor_type or refrigerant

    Example:
        >>> agent = IndustrialHeatPumpAgent_AI(config)
        >>> result = agent.calculate_heat_pump_cop(
        ...     source_temperature_f=50,
        ...     sink_temperature_f=160,
        ...     compressor_type="screw",
        ...     refrigerant="R134a",
        ...     part_load_ratio=0.80
        ... )
        >>> print(f"COP: {result['cop_heating']:.2f}")
        COP: 3.28
        >>> print(f"Carnot Efficiency: {result['carnot_efficiency']:.1%}")
        Carnot Efficiency: 52.0%

    Physics:
        Carnot COP = T_sink_R / (T_sink_R - T_source_R)
        where T_R = T_F + 459.67 (Rankine absolute temperature)

        Actual COP = Carnot COP × Carnot Efficiency × Part_Load_Factor

    Standards:
        - AHRI 540: Performance Rating of Positive Displacement Refrigerant Compressors
        - ISO 13612: Heating and cooling systems in buildings
    """
```

#### 4. Inline Comments
Strategic comments for complex logic.

**Guidelines:**
- Comment WHY, not WHAT (code shows what)
- Explain non-obvious physics or business logic
- Reference standards or papers for complex calculations
- Use TODO/FIXME for known issues

**Example:**
```python
# Apply Carnot efficiency based on compressor type (AHRI 540 empirical data)
# Screw compressors achieve 50-55% of Carnot due to lower internal losses
if compressor_type == "screw":
    carnot_efficiency = 0.52
elif compressor_type == "scroll":
    carnot_efficiency = 0.47  # Lower due to capacity limitations
```

**Reference Achievements:**
- Agent #4: 152-line module docstring (comprehensive heat transfer theory)
- Agent #3: 47-line module docstring (thermodynamic foundation)
- All agents: Complete tool docstrings with examples

---

### Dimension 6: Compliance & Security

**Status Required:** PASS ✅

**Security Requirements:**

#### 1. Zero Secrets (MANDATORY)
- ❌ NO hardcoded API keys, passwords, tokens, credentials
- ❌ NO database connection strings in code
- ❌ NO private keys or certificates
- ✅ Use environment variables or secret management
- ✅ Configuration injection at runtime

**Verification:**
```bash
# Scan for secrets
git secrets --scan
trufflehog --regex --entropy=True .

# Search for common patterns
grep -r "api_key\|password\|secret\|token" greenlang/agents/
```

#### 2. SBOM Required
Every agent MUST have Software Bill of Materials.

**Requirements:**
- List all dependencies with versions
- Include transitive dependencies
- Document CVE scanning results
- Update quarterly

**Example SBOM:**
```yaml
sbom:
  format: "CycloneDX"
  version: "1.4"

  components:
    - name: "pydantic"
      version: "2.4.2"
      type: "library"
      cve_scan_date: "2025-10-23"
      vulnerabilities: []

    - name: "numpy"
      version: "1.26.0"
      type: "library"
      cve_scan_date: "2025-10-23"
      vulnerabilities: []
```

#### 3. Industry Standards Compliance
Agents MUST comply with relevant industry standards.

**Minimum:** 4 industry standards
**Target:** 6+ standards (Agent #3 and #4 achieve 6-7 standards)

**Common Standards:**
- **Thermodynamics:** ASME PTC 4.1, ISO 9806, ASHRAE 93
- **Heat Pumps:** AHRI 540, ISO 13612
- **Heat Exchangers:** TEMA, ASME Section VIII
- **Emissions:** GHG Protocol, ISO 14064, EPA eGRID
- **Energy:** ISO 50001, ASHRAE 90.1
- **Financial:** FASB, GAAP
- **Safety:** NFPA, OSHA regulations

**Documentation:**
```python
STANDARDS_COMPLIANCE = {
    "AHRI_540": "Performance rating of compressors - used in COP calculation",
    "ISO_13612": "Heat pump systems - sizing and performance",
    "ASHRAE_Handbook": "HVAC Systems and Equipment - reference data",
    "GHG_Protocol": "Greenhouse gas accounting and reporting",
    "ISO_14064": "GHG emissions quantification and verification",
    "EPA_eGRID": "Electricity grid emission factors"
}
```

#### 4. Certifications (Optional but Recommended)
```python
CERTIFICATIONS = {
    "ENERGY_STAR": "Energy efficiency certification",
    "AHRI_Certified": "Performance ratings verified by AHRI",
    "ISO_50001": "Energy management system compliance"
}
```

#### 5. Input Validation
All user inputs MUST be validated.

**Requirements:**
- Type checking (Pydantic or explicit)
- Range validation (min/max)
- Unit validation
- Enum validation (fixed choices)
- Cross-parameter validation

**Example:**
```python
from pydantic import BaseModel, Field, validator

class HeatPumpInput(BaseModel):
    source_temp: float = Field(..., ge=-20, le=150, description="Source temperature °F")
    sink_temp: float = Field(..., ge=80, le=250, description="Sink temperature °F")
    compressor: str = Field(..., regex="^(scroll|screw|centrifugal|reciprocating)$")

    @validator("sink_temp")
    def validate_temperature_lift(cls, v, values):
        if "source_temp" in values and v <= values["source_temp"]:
            raise ValueError("Sink temp must exceed source temp (2nd law thermodynamics)")
        return v
```

#### 6. Authentication
Production agents MUST implement authentication.

**Methods:**
- Bearer token authentication (preferred)
- API key validation
- OAuth 2.0 (for user-facing agents)
- mTLS (for inter-agent communication)

**Example:**
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if not validate_token(token):  # Implement token validation
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return token
```

**Reference:** All Phase 2A agents achieve zero secrets and 4-7 standards compliance.

---

### Dimension 7: Deployment Readiness

**Status Required:** PASS ✅

**Requirements:**

#### 1. Deployment Pack Structure
Every agent MUST have a complete Kubernetes deployment pack.

**File Location:** `packs/<agent_name>_ai/deployment_pack.yaml`

**Required Components:**
1. **Deployment:** Pod specification with resource limits
2. **Service:** ClusterIP service for internal access
3. **Ingress:** HTTPS ingress with TLS (optional for external agents)
4. **ConfigMap:** Non-sensitive configuration
5. **Secret:** Sensitive configuration (managed externally)
6. **HorizontalPodAutoscaler:** Auto-scaling configuration
7. **NetworkPolicy:** Security restrictions
8. **ServiceMonitor:** Prometheus metrics (optional)

**Deployment Template:**
```yaml
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: industrial-heat-pump-agent
  namespace: greenlang-agents
  labels:
    app: industrial-heat-pump-agent
    domain: industrial
    category: process-heat
    version: v1.0.0
spec:
  replicas: 3  # High availability
  selector:
    matchLabels:
      app: industrial-heat-pump-agent
  template:
    metadata:
      labels:
        app: industrial-heat-pump-agent
    spec:
      containers:
      - name: agent
        image: greenlang/industrial-heat-pump-agent:v1.0.0
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: AGENT_ID
          value: "industrial/heat_pump_agent"
        - name: LOG_LEVEL
          value: "INFO"
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: anthropic-credentials
              key: api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "500m"
          limits:
            memory: "512Mi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: industrial-heat-pump-agent
  namespace: greenlang-agents
spec:
  selector:
    app: industrial-heat-pump-agent
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
  type: ClusterIP

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: industrial-heat-pump-agent-hpa
  namespace: greenlang-agents
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: industrial-heat-pump-agent
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

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: industrial-heat-pump-agent-netpol
  namespace: greenlang-agents
spec:
  podSelector:
    matchLabels:
      app: industrial-heat-pump-agent
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: greenlang-api-gateway
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: anthropic-api
    ports:
    - protocol: TCP
      port: 443
```

#### 2. Health Checks
Agents MUST implement health endpoints.

**Requirements:**
- `/health` - Liveness probe (is agent alive?)
- `/ready` - Readiness probe (is agent ready to serve traffic?)
- `/metrics` - Prometheus metrics (optional but recommended)

**Example:**
```python
from fastapi import FastAPI, status

app = FastAPI()

@app.get("/health")
def health_check():
    """Liveness probe - basic health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/ready")
def readiness_check():
    """Readiness probe - check dependencies"""
    try:
        # Check Anthropic API connectivity
        test_session = ChatSession(agent_id="health_check")
        # Check database connectivity if applicable
        # Check other critical dependencies
        return {"status": "ready", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "error": str(e)}
        )

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return {
        "agent_requests_total": REQUESTS_COUNTER,
        "agent_request_duration_seconds": DURATION_HISTOGRAM,
        "agent_errors_total": ERRORS_COUNTER,
        "agent_cost_usd_total": COST_COUNTER
    }
```

#### 3. Resource Requirements
Document and enforce resource limits.

**Standard Limits:**
- **Memory:** 256MB request, 512MB limit
- **CPU:** 0.5 core request, 1.0 core limit
- **Replicas:** 3 minimum (HA), 10 maximum (auto-scale)
- **Storage:** Ephemeral (no persistent volumes for stateless agents)

**Adjust based on agent complexity:**
- Simple agents (Agent #1): 256MB/512MB
- Complex agents (Agent #12): 512MB/1GB

#### 4. Dependencies
Declare all dependencies with versions.

**Example:**
```yaml
dependencies:
  python: "3.11+"

  packages:
    - pydantic==2.4.2
    - numpy==1.26.0
    - scipy==1.11.3
    - pandas==2.1.1
    - fastapi==0.104.1
    - uvicorn==0.24.0

  greenlang_core:
    - chat_session>=2.0.0
    - tool_registry>=1.5.0

  external_agents:
    - agent_id: "agents/grid_factor_agent_ai"
      version: ">=1.0.0"
      relationship: "receives_data_from"
    - agent_id: "agents/fuel_agent_ai"
      version: ">=1.0.0"
      relationship: "receives_data_from"
```

#### 5. Container Image
Build and publish container image.

**Dockerfile Example:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY greenlang/ greenlang/
COPY specs/ specs/

# Non-root user
RUN useradd -m -u 1000 greenlang
USER greenlang

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
  CMD curl -f http://localhost:8080/health || exit 1

# Run agent
CMD ["uvicorn", "greenlang.agents.industrial_heat_pump_agent_ai:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Reference:** All Phase 2A agents have deployment pack templates (850-900 lines).

---

### Dimension 8: Exit Bar Criteria

**Status Required:** PASS ✅

**Quality Metrics - ALL MUST PASS:**

#### 1. Test Coverage ≥85%
- **Minimum:** 80%
- **Target:** 85%+
- **Achieved:** All Phase 2A agents meet 85%+

**Measurement:**
```bash
pytest tests/agents/test_<agent>_agent_ai.py \
  --cov=greenlang/agents/<agent>_agent_ai \
  --cov-report=html \
  --cov-fail-under=85
```

#### 2. Latency <3-4s
Performance requirement varies by agent complexity.

**Targets:**
- **Simple agents:** <3s (Agents #1, #2, #3, #4)
- **Complex agents:** <4s (Master orchestrators)
- **Orchestrators:** <10s (Agent #12)

**Measurement:**
```python
import time

start = time.time()
result = agent.analyze(query)
duration = time.time() - start

assert duration < 3.0, f"Latency {duration:.2f}s exceeds 3.0s target"
```

#### 3. Cost <$0.10-$2.00 per analysis
Cost target varies by agent complexity.

**Targets:**
- **Simple agents:** <$0.10 (Agents #1, #3, #4)
- **Medium agents:** <$0.15 (Agent #2)
- **Complex agents:** <$2.00 (Agent #12 - calls multiple agents)

**Measurement:**
```python
result = agent.analyze(query)
cost_usd = result["provenance"]["cost_usd"]

assert cost_usd < 0.10, f"Cost ${cost_usd:.2f} exceeds $0.10 target"
```

#### 4. Security Scan: Zero Critical/High CVEs
Scan for vulnerabilities before deployment.

**Tools:**
```bash
# Dependency vulnerabilities
safety check

# Container image scanning
trivy image greenlang/agent:v1.0.0

# Code security
bandit -r greenlang/agents/<agent>_agent_ai.py
```

**Acceptance:** Zero critical or high severity vulnerabilities.

#### 5. Code Quality Checks
Automated code quality verification.

**Tools:**
```bash
# Linting
flake8 greenlang/agents/<agent>_agent_ai.py --max-line-length=100

# Type checking
mypy greenlang/agents/<agent>_agent_ai.py --strict

# Formatting
black greenlang/agents/<agent>_agent_ai.py --check
```

#### 6. Documentation Complete
All documentation deliverables present:
- ✅ Validation Summary
- ✅ Final Status Report
- ✅ 3 Demo Scripts
- ✅ Deployment Pack

#### 7. Determinism Verified
Determinism tests MUST pass.

```python
def test_determinism():
    results = [agent.analyze(query) for _ in range(3)]
    assert results[0] == results[1] == results[2]
```

**Reference Achievements:**
- **All Phase 2A agents:** Meet all 7 exit bar criteria
- **Coverage:** 85%+ (exceeds 80% minimum)
- **Latency:** <3s average (target <3-4s)
- **Cost:** <$0.12 average (target <$0.15)
- **Security:** Zero critical CVEs
- **Determinism:** 100% reproducible

---

### Dimension 9: Integration & Coordination

**Status Required:** PASS ✅

**Requirements:**

#### 1. Agent Dependencies Declaration
All agent dependencies MUST be explicitly declared in specification.

**Format:**
```yaml
dependencies:
  - agent_id: "agents/grid_factor_agent_ai"
    relationship: "receives_data_from"
    data: "Grid carbon intensity (kg CO2e/kWh) by region"
    required: true

  - agent_id: "agents/fuel_agent_ai"
    relationship: "receives_data_from"
    data: "Fuel prices ($/MMBtu) and emission factors"
    required: true

  - agent_id: "industrial/process_heat_agent"
    relationship: "coordinates_with"
    data: "Process heat requirements and temperature profiles"
    required: false

  - agent_id: "agents/project_finance_agent_ai"
    relationship: "provides_data_to"
    data: "CAPEX/OPEX for financial analysis"
    required: false
```

**Relationship Types:**
- `receives_data_from` - This agent consumes data from dependency
- `provides_data_to` - This agent provides data to dependency
- `coordinates_with` - Bi-directional collaboration
- `orchestrates` - This agent calls dependency (master orchestrator only)

#### 2. Integration Tests
Test multi-agent workflows.

**Requirements:**
- Test calls to each dependency (mocked or real)
- Test error handling when dependency unavailable
- Test data format compatibility
- Test graceful degradation

**Example:**
```python
def test_integration_with_grid_factor_agent(self):
    """Test integration with Grid Factor Agent"""
    # Mock Grid Factor Agent response
    mock_grid_data = {
        "region": "California",
        "emission_factor_kg_per_kwh": 0.227,
        "timestamp": "2025-10-23T10:00:00Z"
    }

    with patch("greenlang.agents.grid_factor_agent_ai.get_grid_factor") as mock:
        mock.return_value = mock_grid_data

        result = agent.analyze("Heat pump electricity emissions in California")

        assert "grid_emissions" in result
        assert result["grid_emissions"]["factor"] == 0.227
```

#### 3. Graceful Degradation
Agents SHOULD handle missing dependencies gracefully.

**Strategies:**
- Use cached/default values when dependency unavailable
- Provide degraded functionality with warning
- Clear error messages guiding user to resolve dependency

**Example:**
```python
def get_grid_factor(region: str) -> float:
    """Get grid emission factor with fallback"""
    try:
        # Try to call Grid Factor Agent
        return grid_factor_agent.get_factor(region)
    except AgentUnavailableError:
        logger.warning("Grid Factor Agent unavailable, using default US average")
        return 0.417  # US average kg CO2e/kWh
    except Exception as e:
        logger.error(f"Error calling Grid Factor Agent: {e}")
        raise
```

#### 4. Master Orchestrators
Agents that orchestrate multiple agents (like Agent #12) have special requirements.

**Additional Requirements:**
- Declare ALL coordinated agents (Agent #12 coordinates 11 agents)
- Implement dependency graph and sequencing logic
- Handle parallel agent calls where possible
- Aggregate results from multiple agents
- Implement timeout and retry logic

**Example - Agent #12:**
```python
class DecarbonizationRoadmapAgent:
    """Master orchestrator coordinating all 11 industrial agents"""

    COORDINATED_AGENTS = [
        "industrial/process_heat_agent",
        "industrial/boiler_replacement_agent",
        "industrial/heat_pump_agent",
        "industrial/waste_heat_recovery_agent",
        # ... 7 more agents
    ]

    def generate_roadmap(self, facility_data: Dict) -> Dict:
        """Orchestrate all agents to create comprehensive roadmap"""

        # Step 1: Identify opportunities (parallel calls)
        opportunities = await asyncio.gather(
            self.call_agent("process_heat_agent", facility_data),
            self.call_agent("boiler_replacement_agent", facility_data),
            self.call_agent("heat_pump_agent", facility_data),
            # ...
        )

        # Step 2: Rank by MAC (Marginal Abatement Cost)
        ranked = self.calculate_mac_curve(opportunities)

        # Step 3: Sequence implementation
        roadmap = self.design_implementation_sequence(ranked)

        return roadmap
```

**Reference:**
- **Agent #12:** Coordinates 11 agents (most complex orchestration)
- **Agents #1-4:** Coordinate with 3-5 agents each
- **All agents:** Pass integration tests

---

### Dimension 10: Business Impact & Metrics

**Status Required:** PASS ✅

**Requirements:**

Every agent MUST document clear business impact and define success metrics.

#### 1. Market Size Analysis
Quantify addressable market.

**Requirements:**
- Total Addressable Market (TAM) in USD
- Geographic scope (global, regional, country)
- Market segment breakdown
- Growth rate (CAGR)
- Market maturity

**Example:**
```yaml
business_impact:
  market_analysis:
    total_addressable_market_usd: 18_000_000_000  # $18B
    geographic_scope: "Global"
    market_segments:
      - segment: "Food & Beverage Processing"
        market_share: 35%
        size_usd: 6_300_000_000
      - segment: "Chemical Manufacturing"
        market_share: 25%
        size_usd: 4_500_000_000
      - segment: "Textile & Paper"
        market_share: 20%
        size_usd: 3_600_000_000
      - segment: "Other Industrial"
        market_share: 20%
        size_usd: 3_600_000_000

    growth_rate_cagr: 15%
    market_maturity: "Growth stage - rapid adoption post-2020"
```

#### 2. Carbon Impact Assessment
Quantify decarbonization potential.

**Requirements:**
- Total carbon impact (Gt CO2e/year addressable)
- Percentage of global/sector emissions
- Typical project carbon reduction
- Time horizon

**Example:**
```yaml
carbon_impact:
  total_addressable_gt_co2e_per_year: 1.2  # 1.2 Gt CO2e/year
  percentage_global_emissions: 2.5%  # 2.5% of 48 Gt global emissions
  percentage_industrial_emissions: 10%  # 10% of 12 Gt industrial emissions

  typical_project:
    annual_reduction_tonnes_co2e: 500  # 500 tonnes per facility
    reduction_percentage: 30%  # 30% reduction in heat-related emissions
    time_horizon_years: 20  # 20-year project lifetime
```

#### 3. Financial Metrics
Define expected ROI and payback.

**Requirements:**
- Typical CAPEX range
- Annual savings range
- Payback period range
- IRR (Internal Rate of Return) range
- NPV analysis

**Example:**
```yaml
financial_metrics:
  typical_project:
    capex_usd:
      minimum: 500_000
      median: 1_200_000
      maximum: 3_000_000

    annual_savings_usd:
      minimum: 100_000
      median: 250_000
      maximum: 600_000

    simple_payback_years:
      best_case: 3.0
      typical: 5.0
      worst_case: 8.0

    irr_percentage:
      minimum: 12%
      median: 20%
      maximum: 30%

    npv_20yr_usd:
      discount_rate: 8%
      median: 1_500_000
```

#### 4. Success Metrics
Define measurable KPIs for agent success.

**Categories:**

**Usage Metrics:**
- Number of analyses performed
- Unique customers/facilities analyzed
- Geographic distribution
- Industry sector distribution

**Quality Metrics:**
- Analysis accuracy (vs actual implementations)
- Prediction error (COP, savings, payback)
- User satisfaction score (NPS, CSAT)
- Time to recommendation

**Business Metrics:**
- Projects implemented (from recommendations)
- Total CAPEX influenced
- Total carbon reduction achieved (tonnes CO2e)
- Revenue generated

**Example:**
```yaml
success_metrics:
  usage:
    target_analyses_per_month: 500
    target_unique_customers: 100
    target_geographies: 15

  quality:
    target_cop_prediction_error: "<5%"
    target_payback_prediction_error: "<10%"
    target_nps_score: ">50"
    target_time_to_recommendation: "<3 minutes"

  business:
    target_implementation_rate: "20%"  # 20% of recommendations implemented
    target_capex_influenced_usd: 50_000_000
    target_carbon_reduction_tonnes: 10_000
```

#### 5. Competitive Advantage
Articulate why GreenLang agent is superior.

**Example:**
```yaml
competitive_advantage:
  - "Only AI agent with comprehensive thermodynamic modeling per AHRI 540"
  - "Integration with grid carbon intensity for true climate impact"
  - "Deterministic recommendations (100% reproducible for compliance)"
  - "Multi-technology comparison (air/water/ground source in single analysis)"
  - "Real-time equipment database with 500+ heat pump models"
```

**Reference Achievements - Phase 2A:**
- **Agent #1:** $120B market, 0.8 Gt CO2e/yr, 3-7 yr payback
- **Agent #2:** $45B market, 0.9 Gt CO2e/yr, 2-5 yr payback
- **Agent #3:** $18B market, 1.2 Gt CO2e/yr, 3-8 yr payback
- **Agent #4:** $75B market, 1.4 Gt CO2e/yr, 0.5-3 yr payback (BEST)
- **Agent #12:** $120B market, 5+ Gt CO2e/yr aggregated, N/A (orchestrator)
- **Total:** $378B market, 9+ Gt CO2e/yr (20% of global emissions)

---

### Dimension 11: Operational Excellence

**Status Required:** PASS ✅

**Requirements:**

#### 1. Health & Monitoring
Production-ready health checks and monitoring.

**Required Endpoints:**
```python
@app.get("/health")
def health():
    """Liveness probe"""
    return {"status": "healthy"}

@app.get("/ready")
def ready():
    """Readiness probe - check dependencies"""
    return {"status": "ready", "dependencies": check_dependencies()}

@app.get("/metrics")
def metrics():
    """Prometheus metrics"""
    return {
        "requests_total": COUNTER,
        "request_duration_seconds": HISTOGRAM,
        "errors_total": COUNTER,
        "cost_usd_total": COUNTER
    }
```

**Prometheus Metrics:**
- `greenlang_agent_requests_total{agent="heat_pump", status="success|error"}`
- `greenlang_agent_request_duration_seconds{agent="heat_pump"}` (histogram)
- `greenlang_agent_errors_total{agent="heat_pump", error_type="..."}`
- `greenlang_agent_cost_usd_total{agent="heat_pump"}`
- `greenlang_agent_tool_calls_total{agent="heat_pump", tool="..."}`

#### 2. Logging
Comprehensive structured logging.

**Requirements:**
- JSON-formatted logs
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Correlation IDs for request tracing
- Sensitive data masking

**Example:**
```python
import logging
import json

logger = logging.getLogger("greenlang.agents.heat_pump")

def analyze(self, query: str, correlation_id: str = None):
    """Analyze with structured logging"""
    correlation_id = correlation_id or str(uuid.uuid4())

    logger.info(json.dumps({
        "event": "analysis_start",
        "correlation_id": correlation_id,
        "agent": "industrial_heat_pump_agent",
        "timestamp": datetime.now().isoformat()
    }))

    try:
        result = self.session.run(...)

        logger.info(json.dumps({
            "event": "analysis_complete",
            "correlation_id": correlation_id,
            "duration_ms": duration,
            "cost_usd": cost,
            "tools_used": [...]
        }))

        return result

    except Exception as e:
        logger.error(json.dumps({
            "event": "analysis_error",
            "correlation_id": correlation_id,
            "error": str(e),
            "error_type": type(e).__name__
        }))
        raise
```

#### 3. Error Handling
Graceful error handling with actionable messages.

**Strategies:**
- Validate inputs early (fail fast)
- Specific exception types
- User-friendly error messages
- Retry logic for transient errors
- Circuit breaker for dependencies

**Example:**
```python
class HeatPumpAnalysisError(Exception):
    """Base exception for heat pump agent"""
    pass

class InvalidTemperatureError(HeatPumpAnalysisError):
    """Temperature inputs violate thermodynamic constraints"""
    pass

class DependencyUnavailableError(HeatPumpAnalysisError):
    """Required dependency agent unavailable"""
    pass

def calculate_cop(self, source_temp: float, sink_temp: float):
    """Calculate COP with comprehensive error handling"""

    # Validate inputs
    if source_temp >= sink_temp:
        raise InvalidTemperatureError(
            f"Source temperature ({source_temp}°F) must be less than "
            f"sink temperature ({sink_temp}°F). This violates the second law "
            f"of thermodynamics - heat cannot spontaneously flow from cold to hot."
        )

    # Call with retry logic
    try:
        return self._calculate_cop_internal(source_temp, sink_temp)
    except Exception as e:
        logger.error(f"COP calculation failed: {e}")
        raise HeatPumpAnalysisError(
            f"Unable to calculate COP. Please verify input temperatures and try again. "
            f"Error: {str(e)}"
        )
```

#### 4. Performance Optimization
Optimize for latency and cost.

**Techniques:**
- Cache frequently used data (material properties, emission factors)
- Vectorize calculations (NumPy arrays vs loops)
- Avoid unnecessary LLM calls
- Batch processing where applicable
- Lazy loading of heavy dependencies

**Example:**
```python
from functools import lru_cache

class HeatPumpAgent:

    @lru_cache(maxsize=128)
    def get_refrigerant_properties(self, refrigerant: str, temp: float):
        """Cache refrigerant properties (expensive lookup)"""
        return self.refrigerant_db.lookup(refrigerant, temp)

    def calculate_cop_batch(self, scenarios: List[Dict]) -> List[Dict]:
        """Batch processing for multiple scenarios"""
        # Vectorized calculation
        source_temps = np.array([s["source_temp"] for s in scenarios])
        sink_temps = np.array([s["sink_temp"] for s in scenarios])

        # Single vectorized calculation
        cops = self._vectorized_cop_calculation(source_temps, sink_temps)

        return [{"cop": cop} for cop in cops]
```

#### 5. Alerting
Define alerts for production issues.

**Critical Alerts:**
- Agent unavailable (health check failing)
- Error rate >1%
- Latency >10s (p99)
- Cost spike (>2x baseline)
- Dependency failures

**Example Prometheus Alerts:**
```yaml
alerts:
  - alert: HighErrorRate
    expr: |
      rate(greenlang_agent_errors_total{agent="heat_pump"}[5m]) > 0.01
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Heat Pump Agent error rate >1%"

  - alert: HighLatency
    expr: |
      histogram_quantile(0.99, greenlang_agent_request_duration_seconds{agent="heat_pump"}) > 10
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Heat Pump Agent p99 latency >10s"
```

**Reference:** All Phase 2A agents implement health checks, structured logging, and comprehensive error handling.

---

### Dimension 12: Continuous Improvement

**Status Required:** PASS ✅

**Requirements:**

#### 1. Version Control
Complete version history and changelog.

**Format:**
```yaml
version_history:
  current_version: "1.0.0"

  releases:
    - version: "1.0.0"
      date: "2025-10-23"
      status: "Production"
      changes:
        - "Initial production release"
        - "8 tools: COP calculation, technology selection, sizing, economics"
        - "85% test coverage, 54 tests"
        - "AHRI 540 and ISO 13612 compliance"

    - version: "0.9.0"
      date: "2025-10-15"
      status: "Beta"
      changes:
        - "Beta release for field validation"
        - "Added part-load performance modeling"
        - "Expanded refrigerant database to 6 types"

    - version: "0.5.0"
      date: "2025-09-01"
      status: "Alpha"
      changes:
        - "Alpha release for internal testing"
        - "Core thermodynamic calculations"
        - "Basic COP calculation"
```

#### 2. Review & Approval
Document review process and sign-offs.

**Required Reviews:**
- Technical Lead (code quality, architecture)
- Domain Expert (thermodynamics, industry standards)
- Security Review (secrets, vulnerabilities)
- QA Sign-off (testing complete, coverage met)

**Format:**
```yaml
reviews:
  - reviewer: "Dr. Jane Smith"
    role: "Lead Thermodynamics Engineer"
    date: "2025-10-20"
    status: "Approved"
    comments: "Thermodynamic calculations validated against AHRI data. COP predictions within 3% of certified performance."

  - reviewer: "John Doe"
    role: "AI Technical Lead"
    date: "2025-10-21"
    status: "Approved"
    comments: "Code quality excellent. Determinism verified. Test coverage 85%. Ready for production."

  - reviewer: "Security Team"
    role: "Security Review"
    date: "2025-10-22"
    status: "Approved"
    comments: "No secrets detected. SBOM complete. Zero critical CVEs."
```

#### 3. Feedback Mechanism
Enable continuous improvement through user feedback.

**Mechanisms:**
- Provenance tracking (which tools used, parameters)
- User satisfaction surveys
- A/B testing support (provenance enables comparison)
- Field validation studies (predicted vs actual performance)

**Example:**
```python
def analyze(self, query: str) -> Dict:
    """Analyze with provenance for feedback loop"""
    result = self.session.run(...)

    # Attach detailed provenance
    result["provenance"] = {
        "agent_version": "1.0.0",
        "tools_used": ["calculate_cop", "select_technology"],
        "parameters": {...},
        "timestamp": "2025-10-23T10:30:00Z",
        "session_id": "abc123",  # For A/B testing
        "deterministic": True
    }

    return result

# Field validation
def submit_feedback(session_id: str, actual_cop: float, predicted_cop: float):
    """Submit field data for model improvement"""
    error_pct = abs(actual_cop - predicted_cop) / actual_cop * 100

    logger.info(f"Field validation: Session {session_id}, Error {error_pct:.1f}%")

    # Store for analysis
    feedback_db.insert({
        "session_id": session_id,
        "predicted_cop": predicted_cop,
        "actual_cop": actual_cop,
        "error_pct": error_pct,
        "timestamp": datetime.now()
    })
```

#### 4. Roadmap
Define future enhancements (v1.1, v1.2, etc.)

**Format:**
```yaml
roadmap:
  v1.1:
    target_date: "Q1 2026"
    features:
      - "Add cascade heat pump systems (multiple stages)"
      - "Integrate with thermal storage optimization"
      - "Expand temperature range to 250°C (CO2 heat pumps)"
      - "Real-time electricity price integration"

  v1.2:
    target_date: "Q2 2026"
    features:
      - "Hybrid heat pump + solar thermal systems"
      - "Demand response optimization"
      - "Predictive maintenance algorithms"
      - "Integration with Building Energy Management Systems (BEMS)"
```

#### 5. Field Validation Plan
Plan to validate agent predictions in real-world deployments.

**Components:**
- Beta customer program (10-20 facilities)
- Measurement & Verification (M&V) protocol
- Quarterly performance reviews
- Model refinement based on field data

**Example:**
```yaml
field_validation:
  beta_program:
    target_participants: 15
    industries:
      - "Food processing: 5 facilities"
      - "Chemical manufacturing: 5 facilities"
      - "Textile/paper: 5 facilities"

  mv_protocol:
    measurements:
      - "COP at full load (quarterly)"
      - "COP at part load (quarterly)"
      - "Energy consumption (monthly)"
      - "Cost savings (monthly)"

    accuracy_targets:
      - "COP prediction: ±5%"
      - "Energy savings: ±10%"
      - "Cost savings: ±15%"

  review_schedule:
    - "Month 3: Initial field data review"
    - "Month 6: Mid-term performance assessment"
    - "Month 12: Annual validation study"
```

**Reference:** All Phase 2A agents include version history, review approvals, and v1.1 roadmaps.

---

## 3. SPECIFICATION STANDARDS

### 3.1 File Structure

**Location:** `specs/domain<N>_<domain>/<category>/agent_<NNN>_<name>.yaml`

**Naming Convention:**
- `domain<N>`: Domain number (1=Industrial, 2=Buildings, 3=Transportation, etc.)
- `<domain>`: Domain name (industrial, buildings, transportation, etc.)
- `<category>`: Category (industrial_process, building_systems, vehicle_fleet, etc.)
- `agent_<NNN>`: Three-digit agent number (001, 002, etc.)
- `<name>`: Descriptive agent name (heat_pump, boiler_replacement, etc.)

**Examples:**
- `specs/domain1_industrial/industrial_process/agent_003_industrial_heat_pump.yaml`
- `specs/domain1_industrial/industrial_process/agent_012_decarbonization_roadmap.yaml`

### 3.2 Complete Specification Template

See Section 2, Dimension 1 for the complete 11-section YAML template.

**Key Requirements:**
- 800-2,900 lines (median: 1,400 lines)
- 11/11 mandatory sections complete
- 7-10 tools with full schemas
- temperature=0.0, seed=42
- 4+ industry standards

---

## 4. IMPLEMENTATION STANDARDS

See Section 2, Dimension 2 for complete implementation standards including:
- File structure and naming
- Required imports
- Configuration classes
- Database/constants classes
- Main agent class template
- Tool implementation patterns

**Key Requirements:**
- 1,300-2,200 lines (median: 1,700 lines)
- Tool-first architecture
- All calculations deterministic
- Comprehensive docstrings
- Full error handling

---

## 5. TEST SUITE STANDARDS

See Section 2, Dimension 3 for complete test suite standards including:
- Test class structure
- 6 test categories (Unit, Integration, Determinism, Boundary, Domain, Performance)
- Test templates and examples

**Key Requirements:**
- 900-1,600 lines (median: 1,400 lines)
- 40+ tests (50+ recommended)
- Coverage ≥85%
- All 6 test categories

---

## 6. DOCUMENTATION STANDARDS

See Section 2, Dimension 5 for complete documentation standards.

**Required Documents:**
1. **Validation Summary** (~400-550 lines)
   - 12-dimension assessment
   - All dimensions PASS

2. **Final Status Report** (~400-550 lines)
   - 100% completion status
   - Code statistics
   - Production approval

3. **Demo Scripts** (3 required, 250-650 lines each)
   - Different industries
   - Runnable standalone
   - Complete with expected results

**Total Documentation:** ~2,500 lines + ~1,200 lines demos

---

## 7. DEPLOYMENT STANDARDS

See Section 2, Dimension 7 for complete deployment standards.

**Required:**
- Deployment pack (~850-900 lines)
- Kubernetes manifests
- Health checks
- Container image

---

## 8. QUALITY METRICS & EXIT CRITERIA

See Section 2, Dimension 8 for complete exit criteria.

**All Must Pass:**
- Test Coverage ≥85%
- Test Count ≥40
- Latency <3-4s
- Cost <$0.10-$2.00
- Zero critical CVEs
- 100% determinism
- All docs complete

---

## 9. DELIVERABLES CHECKLIST

### Complete Deliverables (9 Files)

**Core Development (3 files):**
1. ✅ Specification (800-2,900 lines)
2. ✅ Implementation (1,300-2,200 lines)
3. ✅ Test Suite (900-1,600 lines, 40+ tests, 85%+ coverage)

**Documentation (5 files):**
4. ✅ Validation Summary (~400-550 lines)
5. ✅ Final Status Report (~400-550 lines)
6. ✅ Demo Script #1 (250-650 lines)
7. ✅ Demo Script #2 (250-650 lines)
8. ✅ Demo Script #3 (250-650 lines)

**Deployment (1 file):**
9. ✅ Deployment Pack (~850-900 lines)

**Total Target:** 9 files, ~7,000-9,000 lines

### Progress Tracker

```
Core Development:     [ ] [ ] [ ]  (0/3)
Documentation:        [ ] [ ] [ ] [ ] [ ]  (0/5)
Deployment:           [ ]  (0/1)
Reviews:              [ ] [ ] [ ] [ ]  (0/4)

Overall: 0% Complete
```

---

## 10. REVIEW & APPROVAL PROCESS

### Four Required Reviews

**Stage 1: Technical Review**
- Reviewer: Technical Lead
- Focus: Code quality, architecture, tests, performance
- Approval Required: YES

**Stage 2: Domain Expert Review**
- Reviewer: Domain Expert (Thermodynamics, Energy, etc.)
- Focus: Domain correctness, standards compliance, accuracy
- Approval Required: YES

**Stage 3: Security Review**
- Reviewer: Security Team
- Focus: Secrets, vulnerabilities, SBOM, authentication
- Approval Required: YES

**Stage 4: QA Sign-Off**
- Reviewer: QA Lead
- Focus: All deliverables, exit criteria, production readiness
- Approval Required: YES

**Final Approval:** Head of Industrial Agents

---

## 11. TEMPLATES

### 11.1 Quick Start Checklist

```markdown
# Agent #<NNN> Development Progress

## Week 1: Specification
- [ ] Create spec file (800-2,900 lines)
- [ ] Define 7-10 tools
- [ ] Document 4+ standards
- [ ] Declare dependencies

## Week 2-3: Implementation
- [ ] Create implementation (1,300-2,200 lines)
- [ ] Implement all tools (deterministic)
- [ ] Add comprehensive docstrings
- [ ] Implement error handling

## Week 4: Testing
- [ ] Create test suite (900-1,600 lines)
- [ ] Write 40+ tests (6 categories)
- [ ] Achieve 85%+ coverage
- [ ] Verify determinism

## Week 5: Documentation
- [ ] Validation summary
- [ ] Final status report
- [ ] 3 demo scripts

## Week 6: Deployment
- [ ] Deployment pack
- [ ] Dockerfile
- [ ] Security scan
- [ ] Deploy to staging

## Week 7: Review & Approval
- [ ] Technical review
- [ ] Domain expert review
- [ ] Security review
- [ ] QA sign-off
- [ ] Production approval

Status: ___% Complete
```

### 11.2 File Naming Reference

```
specs/domain<N>_<domain>/<category>/agent_<NNN>_<name>.yaml
greenlang/agents/<name>_agent_ai.py
tests/agents/test_<name>_agent_ai.py
AGENT_<NNN>_VALIDATION_SUMMARY.md
AGENT_<NNN>_FINAL_STATUS.md
demos/<type>/demo_<NNN>_<description>.py
packs/<name>_ai/deployment_pack.yaml
docker/<name>/Dockerfile
```

---

## 12. APPENDIX: PHASE 2A REFERENCE AGENTS

### 12.1 Achievement Summary

Phase 2A established the gold standard with 5 production-ready agents:

| Agent | Lines | Tests | Coverage | Market | Carbon | Payback | Status |
|-------|-------|-------|----------|--------|--------|---------|--------|
| **#1** | 7,387 | 45+ | 85%+ | $120B | 0.8 Gt/yr | 3-7 yrs | ✅ 100% |
| **#2** | 7,368 | 50+ | 85%+ | $45B | 0.9 Gt/yr | 2-5 yrs | ✅ 100% |
| **#3** | 7,645 | 54+ | 85%+ | $18B | 1.2 Gt/yr | 3-8 yrs | ✅ 100% |
| **#4** | 7,217 | 50+ | 85%+ | $75B | 1.4 Gt/yr | 0.5-3 yrs | ✅ 100% |
| **#12** | 9,100 | 40+ | 80%+ | $120B | 5+ Gt/yr | N/A | ✅ 100% |
| **TOTAL** | **45,195** | **239+** | **85%+** | **$378B** | **9+ Gt/yr** | **Varies** | **✅ 100%** |

### 12.2 Key Learnings

**What Worked Well:**
1. Tool-first architecture ensures determinism
2. 12-dimension framework catches all quality issues
3. 85%+ coverage standard prevents bugs
4. 3 demo scripts prove real-world value
5. Systematic completion prevents shortcuts

**Best Practices:**
1. Follow Agent #4 pattern (best payback)
2. Don't settle for 80% coverage (aim for 85%+)
3. Complete all 12 dimensions (no shortcuts)
4. Create 3 diverse demos (prove versatility)
5. Design for orchestration from start

### 12.3 Agent Highlights

**Agent #1 (IndustrialProcessHeatAgent_AI):**
- LARGEST test suite (1,538 lines)
- Master coordinator role
- $120B market (tied largest)

**Agent #2 (BoilerReplacementAgent_AI):**
- LARGEST specification (1,427 lines)
- Retrofit specialization
- Universal applicability

**Agent #3 (IndustrialHeatPumpAgent_AI):**
- LARGEST implementation (1,872 lines)
- MOST tests (54+)
- Comprehensive thermodynamics

**Agent #4 (WasteHeatRecoveryAgent_AI):**
- BEST payback (0.5-3 years)
- Highest adoption potential
- Excellent business case

**Agent #12 (DecarbonizationRoadmapAgent_AI):**
- LARGEST overall (9,100 lines)
- Master orchestrator
- Coordinates all 11 agents

### 12.4 Use as Templates

**For Simple Agents:** Use Agent #1 as template
**For Medium Agents:** Use Agent #2 or #4 as template
**For Complex Agents:** Use Agent #3 as template
**For Orchestrators:** Use Agent #12 as template

---

## CONCLUSION

The **GreenLang Agent Development Standard (GL-ADS)** provides a comprehensive, battle-tested framework for developing world-class AI agents.

By following this standard, all agents will achieve:

✅ **Consistency** - Uniform architecture across all agents
✅ **Determinism** - 100% reproducible outputs
✅ **Quality** - 85%+ test coverage, comprehensive validation
✅ **Production Readiness** - Enterprise-grade reliability
✅ **Business Impact** - Clear market and carbon metrics
✅ **Compliance** - Security, privacy, and regulatory requirements

**This standard is MANDATORY for all agent development.**

**Key Success Factors:**
1. Follow the 12-dimension framework rigorously
2. Achieve 85%+ test coverage (not just 80%)
3. Complete all 9 deliverables before declaring "done"
4. Get all 4 reviews/approvals
5. Use Phase 2A agents as reference templates

**Phase 2A Proof:**
- 5 agents, 45,195 lines of code
- 239+ tests, 85%+ average coverage
- $378B market, 9+ Gt CO2e/year impact
- 100% production-ready

**The standard works. Follow it.**

---

**Document:** GL_AGENT_STANDARD.md
**Version:** 1.0.0
**Status:** Official Standard
**Authority:** Head of Industrial Agents, AI & Climate Intelligence
**Date:** October 23, 2025
**Reference:** Phase 2A Agents (#1, #2, #3, #4, #12)

---

**END OF DOCUMENT**