# AgentSpec v2 Migration Guide

**Target Agents:** FuelAgent+AI, CarbonAgent+AI, GridFactorAgent+AI
**Effort Estimate:** 2 weeks (6-10 days per agent)
**Priority:** HIGH (blocks full W1 compliance)

---

## Executive Summary

The three AI agents (FuelAgent+AI, CarbonAgent+AI, GridFactorAgent+AI) currently use custom base classes instead of the AgentSpec v2 standard. This creates incompatibility with AgentSpec v2 tooling and violates the "eat your own dog food" principle.

**Current State:**
- FuelAgent+AI extends `Agent[FuelInput, FuelOutput]`
- CarbonAgent+AI extends `BaseAgent`
- GridFactorAgent+AI extends `Agent` with `OperationalMonitoringMixin`

**Target State:**
- All agents comply with AgentSpec v2 schema
- pack.yaml files follow AgentSpec v2 format
- Compatible with `gl pack validate` and other v2 tools

---

## Why This Matters

### 1. Standards Compliance
You built AgentSpec v2 with:
- 15 validation error codes
- Climate units whitelist
- Provenance tracking fields
- Compute/AI/Realtime/Provenance sections

**Your agents should use it.**

### 2. Interoperability
AgentSpec v2 enables:
- `gl pack validate` checks
- `gl init agent` scaffolding
- Cross-agent compatibility
- Tool ecosystem integration

### 3. Maintainability
Custom base classes create:
- Duplicated code
- Diverging implementations
- Migration debt
- Testing complexity

---

## Migration Strategy

### Option 1: Wrapper Approach (RECOMMENDED)

**Pros:**
- Keep existing agent code mostly unchanged
- Add AgentSpec v2 as thin wrapper layer
- Gradual migration path
- Lower risk

**Cons:**
- Slight indirection overhead
- Two layers to maintain temporarily

**Effort:** 2-3 days per agent

### Option 2: Full Rewrite

**Pros:**
- Clean implementation
- Native AgentSpec v2
- No wrapper overhead

**Cons:**
- High risk of breaking existing functionality
- Requires extensive regression testing
- 1-2 weeks per agent

**Effort:** 1-2 weeks per agent

---

## Recommended Approach: Wrapper Pattern

### Step 1: Create pack.yaml (AgentSpec v2 Format)

**File:** `packs/fuel_ai/pack.yaml`

```yaml
# AgentSpec v2 Compliant Pack
schema_version: "2.0.0"
id: "fuel-agent-ai"
name: "Fuel Emissions Agent (AI-Powered)"
version: "1.0.0"
summary: "AI-powered fuel emissions calculation with deterministic tool runtime"

metadata:
  tags: ["emissions", "fuel", "scope1", "ai-agent"]
  owners: ["greenlang-team"]
  license: "MIT"

compute:
  entrypoint: "python://greenlang.agents.fuel_agent_ai:compute"
  deterministic: true
  timeout_seconds: 30
  memory_limit_mb: 512

  inputs:
    fuel_type:
      dtype: "string"
      description: "Type of fuel (natural_gas, coal, diesel, etc.)"
      required: true
      constraints:
        enum: ["natural_gas", "coal", "diesel", "propane", "fuel_oil"]

    amount:
      dtype: "float"
      unit: "therm"
      description: "Fuel consumption amount"
      required: true
      constraints:
        ge: 0.0

    unit:
      dtype: "string"
      description: "Unit of measurement"
      required: true
      constraints:
        enum: ["therms", "m3", "gallons", "liters"]

    country:
      dtype: "string"
      description: "Country code for emission factor lookup"
      required: false
      default: "US"

    renewable_percentage:
      dtype: "float"
      description: "Percentage of renewable energy offset"
      required: false
      default: 0.0
      constraints:
        ge: 0.0
        le: 100.0

  outputs:
    co2e_emissions_kg:
      dtype: "float"
      unit: "kgCO2e"
      description: "Total CO2 equivalent emissions"

    emission_factor:
      dtype: "float"
      unit: "kgCO2e/therm"
      description: "Emission factor used"

    citations:
      dtype: "array"
      description: "List of emission factor citations"

  factors:
    - ref: "ef://ipcc/natural-gas-combustion"
      gwp_set: "AR6GWP100"
    - ref: "ef://epa/fossil-fuels"
      gwp_set: "AR6GWP100"

ai:
  json_mode: true
  system_prompt: |
    You are a climate emissions expert. Calculate fuel emissions using the provided tools.
    ALWAYS use tools for calculations. NEVER guess numbers.
  budget:
    max_cost_usd: 1.0
    max_tokens: 15000
    max_retries: 3
  rag_collections: ["ghg_protocol_corp", "ipcc_ar6"]
  tools:
    - name: "calculate_emissions"
      schema_in:
        type: "object"
        required: ["fuel_type", "amount", "unit"]
        properties:
          fuel_type: { type: "string" }
          amount: { type: "number" }
          unit: { type: "string" }
      schema_out:
        type: "object"
        required: ["emissions_kg_co2e"]
        properties:
          emissions_kg_co2e:
            $ref: "greenlang://schemas/quantity.json"
      impl: "python://greenlang.agents.fuel_agent_ai:calculate_emissions_tool"
      safety: "deterministic"

    - name: "lookup_emission_factor"
      schema_in:
        type: "object"
        required: ["fuel_type", "country"]
        properties:
          fuel_type: { type: "string" }
          country: { type: "string" }
      schema_out:
        type: "object"
        required: ["emission_factor"]
        properties:
          emission_factor:
            $ref: "greenlang://schemas/quantity.json"
      impl: "python://greenlang.agents.fuel_agent_ai:lookup_emission_factor_tool"
      safety: "deterministic"

realtime:
  default_mode: "replay"
  snapshot_path: "./snapshots/"
  connectors: []  # No live connectors required

provenance:
  pin_ef: true
  gwp_set: "AR6GWP100"
  record: ["seed", "temperature", "tool_calls", "ef_cids", "calculation_steps"]
```

**Effort:** 2-3 hours per agent

### Step 2: Create AgentSpec v2 Wrapper

**File:** `greenlang/agents/fuel_agent_ai_v2.py`

```python
"""
AgentSpec v2 Wrapper for FuelAgent+AI

Provides AgentSpec v2 compliance while delegating to existing implementation.
"""

from typing import Dict, Any
from pathlib import Path
import yaml
from greenlang.specs.agentspec_v2 import AgentSpecV2
from greenlang.agents.fuel_agent_ai import FuelAgentAI


class FuelAgentAI_V2:
    """
    AgentSpec v2 compliant wrapper for FuelAgent+AI.

    Loads pack.yaml and validates against AgentSpec v2 schema.
    Delegates execution to existing FuelAgentAI implementation.
    """

    def __init__(self, pack_path: Path):
        """
        Initialize from pack.yaml.

        Args:
            pack_path: Path to pack directory containing pack.yaml
        """
        # Load pack.yaml
        pack_yaml = pack_path / "pack.yaml"
        with open(pack_yaml) as f:
            pack_data = yaml.safe_load(f)

        # Validate against AgentSpec v2
        self.spec = AgentSpecV2(**pack_data)

        # Create delegate agent
        self.delegate = FuelAgentAI(
            agent_id=self.spec.id,
            enable_explanations=True,
            enable_tools=True,
        )

    def compute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute computation (AgentSpec v2 entrypoint).

        Args:
            inputs: Input dictionary matching spec.compute.inputs

        Returns:
            Output dictionary matching spec.compute.outputs
        """
        # Validate inputs against spec
        self._validate_inputs(inputs)

        # Delegate to existing implementation
        result = self.delegate.run(inputs)

        # Validate outputs against spec
        if result["success"]:
            self._validate_outputs(result["data"])

        return result

    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Validate inputs against AgentSpec v2 schema.

        Raises:
            GLValidationError: If inputs don't match spec
        """
        for field_name, field_spec in self.spec.compute.inputs.items():
            # Check required fields
            if field_spec.required and field_name not in inputs:
                from greenlang.specs.errors import GLValidationError, GLVErr
                raise GLValidationError(
                    code=GLVErr.MISSING_FIELD,
                    message=f"Required input '{field_name}' missing",
                    path=["compute", "inputs", field_name],
                )

            # Check constraints (ge, le, enum, etc.)
            if field_name in inputs:
                value = inputs[field_name]
                if field_spec.constraints:
                    self._check_constraints(field_name, value, field_spec.constraints)

    def _validate_outputs(self, outputs: Dict[str, Any]) -> None:
        """
        Validate outputs against AgentSpec v2 schema.

        Ensures all required outputs are present.
        """
        for field_name, field_spec in self.spec.compute.outputs.items():
            if field_name not in outputs:
                from greenlang.specs.errors import GLValidationError, GLVErr
                raise GLValidationError(
                    code=GLVErr.MISSING_FIELD,
                    message=f"Required output '{field_name}' missing",
                    path=["compute", "outputs", field_name],
                )

    def _check_constraints(self, field_name: str, value: Any, constraints: Dict) -> None:
        """Check field value against constraints."""
        if "ge" in constraints and value < constraints["ge"]:
            from greenlang.specs.errors import GLValidationError, GLVErr
            raise GLValidationError(
                code=GLVErr.CONSTRAINT,
                message=f"{field_name} must be >= {constraints['ge']}, got {value}",
                path=["compute", "inputs", field_name],
            )

        if "le" in constraints and value > constraints["le"]:
            from greenlang.specs.errors import GLValidationError, GLVErr
            raise GLValidationError(
                code=GLVErr.CONSTRAINT,
                message=f"{field_name} must be <= {constraints['le']}, got {value}",
                path=["compute", "inputs", field_name],
            )

        if "enum" in constraints and value not in constraints["enum"]:
            from greenlang.specs.errors import GLValidationError, GLVErr
            raise GLValidationError(
                code=GLVErr.CONSTRAINT,
                message=f"{field_name} must be one of {constraints['enum']}, got {value}",
                path=["compute", "inputs", field_name],
            )


# Export compute function for entrypoint
def compute(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    AgentSpec v2 entrypoint for FuelAgent+AI.

    This function is referenced in pack.yaml:
        compute:
          entrypoint: "python://greenlang.agents.fuel_agent_ai_v2:compute"

    Args:
        inputs: Input dictionary from pipeline

    Returns:
        Output dictionary with results
    """
    # Load pack from relative path
    pack_path = Path(__file__).parent.parent.parent / "packs" / "fuel_ai"
    agent = FuelAgentAI_V2(pack_path)
    return agent.compute(inputs)
```

**Effort:** 4-6 hours per agent

### Step 3: Update Tests for v2 Compliance

**File:** `tests/agents/test_fuel_agent_ai_v2.py`

```python
"""
Tests for AgentSpec v2 compliance of FuelAgent+AI.
"""

import pytest
from pathlib import Path
from greenlang.agents.fuel_agent_ai_v2 import FuelAgentAI_V2
from greenlang.specs.errors import GLValidationError, GLVErr


def test_pack_yaml_loads_and_validates():
    """pack.yaml loads and validates against AgentSpec v2."""
    pack_path = Path(__file__).parent.parent.parent / "packs" / "fuel_ai"
    agent = FuelAgentAI_V2(pack_path)

    assert agent.spec.schema_version == "2.0.0"
    assert agent.spec.id == "fuel-agent-ai"
    assert agent.spec.compute.deterministic is True


def test_missing_required_input_raises_validation_error():
    """Missing required input raises GLValidationError.MISSING_FIELD."""
    pack_path = Path(__file__).parent.parent.parent / "packs" / "fuel_ai"
    agent = FuelAgentAI_V2(pack_path)

    with pytest.raises(GLValidationError) as exc_info:
        agent.compute({"amount": 1000})  # Missing fuel_type

    assert exc_info.value.code == GLVErr.MISSING_FIELD
    assert "fuel_type" in str(exc_info.value)


def test_constraint_violation_raises_validation_error():
    """Constraint violation (ge/le) raises GLValidationError.CONSTRAINT."""
    pack_path = Path(__file__).parent.parent.parent / "packs" / "fuel_ai"
    agent = FuelAgentAI_V2(pack_path)

    with pytest.raises(GLValidationError) as exc_info:
        agent.compute({
            "fuel_type": "natural_gas",
            "amount": -100,  # Violates ge: 0.0
            "unit": "therms",
        })

    assert exc_info.value.code == GLVErr.CONSTRAINT
    assert "must be >=" in str(exc_info.value)


def test_valid_input_executes_successfully():
    """Valid input executes and returns expected output."""
    pack_path = Path(__file__).parent.parent.parent / "packs" / "fuel_ai"
    agent = FuelAgentAI_V2(pack_path)

    result = agent.compute({
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
    })

    assert result["success"] is True
    assert "co2e_emissions_kg" in result["data"]
    assert result["data"]["co2e_emissions_kg"] > 0


def test_gl_pack_validate_passes():
    """pack.yaml passes `gl pack validate`."""
    pack_path = Path(__file__).parent.parent.parent / "packs" / "fuel_ai"

    # This would run: gl pack validate packs/fuel_ai
    # For now, just test that pack_yaml loads
    agent = FuelAgentAI_V2(pack_path)
    assert agent.spec is not None
```

**Effort:** 2-3 hours per agent

### Step 4: Update CLI Integration

**File:** `greenlang/cli/cmd_pack_new.py`

```python
# Add v2 agent loading
@app.command("run")
def run_pack_v2(
    pack_path: Path = typer.Argument(..., help="Path to pack directory"),
    inputs: Path = typer.Option(None, "--inputs", "-i", help="Input JSON file"),
):
    """Run an AgentSpec v2 pack."""
    from greenlang.agents.fuel_agent_ai_v2 import FuelAgentAI_V2

    # Load pack
    agent = FuelAgentAI_V2(pack_path)

    # Load inputs
    if inputs:
        import json
        with open(inputs) as f:
            input_data = json.load(f)
    else:
        input_data = {}

    # Run
    result = agent.compute(input_data)

    # Display result
    console.print(result)
```

**Effort:** 1-2 hours

---

## Migration Checklist

### For Each Agent (FuelAgent+AI, CarbonAgent+AI, GridFactorAgent+AI):

- [ ] **Step 1:** Create pack.yaml in AgentSpec v2 format
  - [ ] Define all input fields with dtype, unit, constraints
  - [ ] Define all output fields with dtype, unit
  - [ ] List emission factor references
  - [ ] Configure AI section (tools, budget, RAG)
  - [ ] Set provenance tracking fields
  - [ ] **Effort:** 2-3 hours

- [ ] **Step 2:** Create v2 wrapper class
  - [ ] Load and validate pack.yaml
  - [ ] Implement input validation
  - [ ] Delegate to existing agent
  - [ ] Implement output validation
  - [ ] Export compute() entrypoint
  - [ ] **Effort:** 4-6 hours

- [ ] **Step 3:** Create v2 compliance tests
  - [ ] Test pack.yaml loads
  - [ ] Test input validation (missing, constraints)
  - [ ] Test output validation
  - [ ] Test `gl pack validate` passes
  - [ ] **Effort:** 2-3 hours

- [ ] **Step 4:** Update existing tests
  - [ ] Update imports to use v2 wrapper
  - [ ] Verify all 47-61 tests still pass
  - [ ] Add new v2-specific tests
  - [ ] **Effort:** 2-3 hours

- [ ] **Step 5:** Update documentation
  - [ ] Update README with pack.yaml reference
  - [ ] Document v2 migration
  - [ ] Update examples to use v2 format
  - [ ] **Effort:** 1-2 hours

**Total Per Agent:** 11-17 hours (1.5-2 days)
**Total All Agents:** 33-51 hours (4-6 days)

---

## Testing Strategy

### Phase 1: Wrapper Tests
- [ ] pack.yaml loads without errors
- [ ] AgentSpec v2 validation passes
- [ ] Input validation catches errors
- [ ] Output validation verifies structure

### Phase 2: Regression Tests
- [ ] All existing tests pass with wrapper
- [ ] No performance degradation
- [ ] Determinism maintained
- [ ] Citation tracking works

### Phase 3: Integration Tests
- [ ] `gl pack validate` passes
- [ ] `gl pack run` executes successfully
- [ ] CLI tools work with v2 packs
- [ ] Pack installation works

---

## Rollout Plan

### Week 1: FuelAgent+AI
- Day 1-2: Create pack.yaml + wrapper
- Day 3: Tests + documentation
- Day 4: Review + regression testing
- Day 5: Merge to master

### Week 2: CarbonAgent+AI
- Day 1-2: Create pack.yaml + wrapper
- Day 3: Tests + documentation
- Day 4: Review + regression testing
- Day 5: Merge to master

### Week 3: GridFactorAgent+AI (if needed, or parallel with Week 2)
- Day 1-2: Create pack.yaml + wrapper
- Day 3: Tests + documentation
- Day 4: Review + regression testing
- Day 5: Merge to master

---

## Benefits After Migration

### 1. Standards Compliance ✅
- All agents follow AgentSpec v2
- Compatible with v2 tooling
- Validation errors use standard codes

### 2. Maintainability ✅
- Single source of truth (pack.yaml)
- Consistent structure across agents
- Easy to add new agents

### 3. Interoperability ✅
- Works with `gl pack validate`
- Works with `gl init agent` templates
- Compatible with future v2 features

### 4. Credibility ✅
- "Eating your own dog food"
- Demo-ready for stakeholders
- Production-ready architecture

---

## Risk Mitigation

### Risk 1: Breaking Existing Functionality
**Mitigation:** Wrapper pattern preserves existing code
**Validation:** Run full test suite (166 tests)

### Risk 2: Performance Overhead
**Mitigation:** Wrapper is thin, minimal overhead
**Validation:** Benchmark before/after

### Risk 3: Timeline Slippage
**Mitigation:** Parallel development (2 agents at once)
**Validation:** Weekly progress reviews

---

## Success Criteria

- [ ] All 3 agents have AgentSpec v2 pack.yaml files
- [ ] `gl pack validate packs/fuel_ai` passes
- [ ] `gl pack validate packs/carbon_ai` passes
- [ ] `gl pack validate packs/grid_factor_ai` passes
- [ ] All existing tests pass (166 tests)
- [ ] New v2 compliance tests pass
- [ ] Documentation updated
- [ ] Demo-ready for stakeholders

---

**Estimated Total Effort:** 2-3 weeks
**Priority:** HIGH (blocks full W1 compliance)
**Owner:** Engineering team
**Status:** Planning complete, implementation pending
