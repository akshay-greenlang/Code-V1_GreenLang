# AgentSpec to Code Mapping

**Version**: 1.0.0
**Status**: Design
**Last Updated**: 2025-12-03
**Owner**: GL Backend Developer

---

## Executive Summary

This document defines the precise mapping from AgentSpec v2 YAML sections to generated Python code, templates, and configurations. It serves as the canonical reference for how each AgentSpec field translates into production code.

**Key Principles**:
- **One-to-One Mapping**: Every AgentSpec section maps to specific code artifacts
- **Zero Ambiguity**: Clear, deterministic transformation rules
- **Type Safety**: AgentSpec types map to Python type hints
- **Validation Preservation**: Constraints become Pydantic validators
- **Provenance Tracking**: All spec metadata embedded in generated code

---

## 1. Metadata Section Mapping

### 1.1 AgentSpec Metadata

```yaml
schema_version: "2.0.0"
id: "emissions/fuel_agent_v1"
name: "Fuel Emissions Agent"
version: "1.0.0"
summary: "Calculate fuel combustion emissions"

metadata:
  tags: ["emissions", "scope1", "fuel"]
  owners: ["greenlang-team"]
  license: "MIT"
```

### 1.2 Generated Code

#### Module Docstring

```python
"""
Fuel Emissions Agent

Calculate fuel combustion emissions

Generated from AgentSpec v2
Agent ID: emissions/fuel_agent_v1
Version: 1.0.0
Schema Version: 2.0.0
License: MIT
Owners: greenlang-team
Tags: emissions, scope1, fuel

Auto-generated on: {timestamp}
Generator Version: 1.0.0
"""
```

#### Agent Class Attributes

```python
class FuelAgentAI(BaseAgent):
    """Fuel Emissions Agent."""

    # AgentSpec metadata
    AGENT_ID = "emissions/fuel_agent_v1"
    AGENT_VERSION = "1.0.0"
    SCHEMA_VERSION = "2.0.0"
    TAGS = ["emissions", "scope1", "fuel"]
    OWNERS = ["greenlang-team"]
    LICENSE = "MIT"
```

#### pack.yaml (copied verbatim)

```yaml
# Original AgentSpec v2 copied to pack.yaml
schema_version: "2.0.0"
id: "emissions/fuel_agent_v1"
# ... rest of spec
```

---

## 2. Compute Section Mapping

### 2.1 Compute Configuration

```yaml
compute:
  entrypoint: "python://greenlang.agents.fuel_agent_ai:compute"
  deterministic: true
  timeout_seconds: 30
  memory_limit_mb: 512
```

### 2.2 Generated Code

#### Agent Configuration

```python
from pydantic import BaseModel, Field

class FuelAgentConfig(BaseModel):
    """Configuration for Fuel Agent."""

    deterministic: bool = Field(
        default=True,
        description="Deterministic execution mode"
    )
    timeout_seconds: int = Field(
        default=30,
        description="Maximum execution time"
    )
    memory_limit_mb: int = Field(
        default=512,
        description="Memory limit in MB"
    )
    seed: Optional[int] = Field(
        default=42 if deterministic else None,
        description="Random seed for determinism"
    )
    temperature: float = Field(
        default=0.0 if deterministic else 0.7,
        description="LLM temperature"
    )
```

#### Entrypoint Function

```python
def compute(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entrypoint for Fuel Agent.

    Defined in AgentSpec: python://greenlang.agents.fuel_agent_ai:compute

    Args:
        inputs: Input dictionary matching InputModel schema

    Returns:
        Output dictionary matching OutputModel schema
    """
    config = FuelAgentConfig()
    agent = FuelAgentAI(config)
    result = agent.process(FuelInput(**inputs))
    return result.dict()
```

---

## 3. Input/Output Schema Mapping

### 3.1 Input Schema

```yaml
compute:
  inputs:
    fuel_type:
      dtype: "string"
      description: "Type of fuel"
      required: true
      constraints:
        enum: ["natural_gas", "coal", "diesel"]

    amount:
      dtype: "float64"
      unit: "therm"
      description: "Fuel consumption amount"
      required: true
      constraints:
        ge: 0.0

    country:
      dtype: "string"
      description: "Country code"
      required: false
      default: "US"

    renewable_percentage:
      dtype: "float64"
      description: "Renewable energy percentage"
      required: false
      default: 0.0
      constraints:
        ge: 0.0
        le: 100.0
```

### 3.2 Generated Pydantic Models

```python
from typing import Optional, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum

class FuelType(str, Enum):
    """Fuel type enumeration."""
    NATURAL_GAS = "natural_gas"
    COAL = "coal"
    DIESEL = "diesel"


class FuelInput(BaseModel):
    """Input data model for Fuel Agent."""

    fuel_type: FuelType = Field(
        ...,
        description="Type of fuel"
    )

    amount: float = Field(
        ...,
        description="Fuel consumption amount",
        ge=0.0  # Constraint: greater than or equal to 0.0
    )

    country: str = Field(
        default="US",
        description="Country code"
    )

    renewable_percentage: float = Field(
        default=0.0,
        description="Renewable energy percentage",
        ge=0.0,
        le=100.0
    )

    @validator('amount')
    def validate_amount_positive(cls, v):
        """Validate amount is positive."""
        if v < 0:
            raise ValueError("amount must be non-negative")
        return v

    @validator('renewable_percentage')
    def validate_percentage_range(cls, v):
        """Validate percentage is in valid range."""
        if not 0.0 <= v <= 100.0:
            raise ValueError("renewable_percentage must be between 0 and 100")
        return v

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"  # No extra fields allowed
```

### 3.3 Type Mapping Table

| AgentSpec dtype | Python Type | Pydantic Field Type |
|-----------------|-------------|---------------------|
| `string` | `str` | `str` |
| `float64` | `float` | `float` |
| `float32` | `float` | `float` |
| `int64` | `int` | `int` |
| `int32` | `int` | `int` |
| `bool` | `bool` | `bool` |
| `array` | `List[Any]` | `List` |
| `object` | `Dict[str, Any]` | `Dict` |
| `datetime` | `datetime` | `datetime` |

### 3.4 Constraint Mapping

| AgentSpec Constraint | Pydantic Implementation |
|---------------------|-------------------------|
| `required: true` | `Field(...)` (required) |
| `required: false` | `Field(default=X)` (optional) |
| `default: value` | `Field(default=value)` |
| `ge: N` | `Field(..., ge=N)` |
| `le: N` | `Field(..., le=N)` |
| `gt: N` | `Field(..., gt=N)` |
| `lt: N` | `Field(..., lt=N)` |
| `enum: [a, b, c]` | `Literal["a", "b", "c"]` or `Enum` |
| `min_length: N` | `Field(..., min_length=N)` |
| `max_length: N` | `Field(..., max_length=N)` |
| `pattern: "regex"` | `Field(..., regex="regex")` |

### 3.5 Output Schema Mapping

```yaml
compute:
  outputs:
    co2e_emissions_kg:
      dtype: "float64"
      unit: "kgCO2e"
      description: "Total CO2 equivalent emissions"

    emission_factor:
      dtype: "float64"
      unit: "kgCO2e/therm"
      description: "Emission factor used"

    citations:
      dtype: "array"
      description: "List of emission factor citations"
```

Generated Output Model:

```python
from typing import List, Dict, Any

class Citation(BaseModel):
    """Citation model."""
    source: str
    ef_cid: str
    value: float
    unit: str


class FuelOutput(BaseModel):
    """Output data model for Fuel Agent."""

    co2e_emissions_kg: float = Field(
        ...,
        description="Total CO2 equivalent emissions"
    )

    emission_factor: float = Field(
        ...,
        description="Emission factor used"
    )

    citations: List[Citation] = Field(
        default_factory=list,
        description="List of emission factor citations"
    )

    # Auto-added provenance fields
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for audit trail"
    )

    processing_time_ms: float = Field(
        ...,
        description="Processing duration in milliseconds"
    )

    agent_id: str = Field(
        default="emissions/fuel_agent_v1",
        description="Agent identifier"
    )

    agent_version: str = Field(
        default="1.0.0",
        description="Agent version"
    )
```

---

## 4. AI Section Mapping

### 4.1 AI Configuration

```yaml
ai:
  json_mode: true
  system_prompt: "You are a climate emissions expert. Calculate fuel emissions using the provided tools."
  budget:
    max_cost_usd: 1.0
    max_tokens: 15000
    max_retries: 3
  rag_collections: ["ghg_protocol_corp", "ipcc_ar6"]
  tools:
    - name: "calculate_emissions"
      schema_in:
        type: "object"
        required: ["fuel_type", "amount"]
      schema_out:
        type: "object"
        required: ["emissions_kg_co2e"]
      impl: "python://greenlang.calculators.fuel:calculate_fuel_emissions"
      safety: "deterministic"
```

### 4.2 Generated Code

#### Prompt Template

```python
# File: prompts.py

SYSTEM_PROMPT = """
You are a climate emissions expert. Calculate fuel emissions using the provided tools.

IMPORTANT RULES:
1. ALWAYS use tools for calculations - NEVER guess numbers
2. Use deterministic tools for all numeric calculations
3. Provide complete citations for all emission factors
4. Track provenance for audit trail

Available Tools:
- calculate_emissions: Calculate fuel combustion emissions

Your response MUST be valid JSON matching the output schema.
""".strip()


def get_system_prompt(context: Optional[Dict[str, Any]] = None) -> str:
    """Get system prompt with optional context."""
    prompt = SYSTEM_PROMPT
    if context:
        prompt += f"\n\nAdditional Context:\n{json.dumps(context, indent=2)}"
    return prompt
```

#### LLM Configuration

```python
from anthropic import Anthropic
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    """LLM configuration from AgentSpec."""

    json_mode: bool = Field(
        default=True,
        description="Enable JSON mode"
    )

    max_cost_usd: float = Field(
        default=1.0,
        description="Maximum cost per request in USD"
    )

    max_tokens: int = Field(
        default=15000,
        description="Maximum tokens per request"
    )

    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts"
    )

    temperature: float = Field(
        default=0.0,  # Deterministic
        description="Sampling temperature"
    )

    model: str = Field(
        default="claude-3-opus-20240229",
        description="LLM model identifier"
    )


class FuelAgentAI(BaseAgent):
    """Fuel Agent with LLM integration."""

    def __init__(self, config: FuelAgentConfig):
        super().__init__(config)
        self.llm_config = LLMConfig()
        self.client = Anthropic()

    def _call_llm(self, messages: List[Dict], tools: List[Dict]) -> Dict:
        """Call LLM with budget and retry logic."""
        for attempt in range(self.llm_config.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.llm_config.model,
                    max_tokens=self.llm_config.max_tokens,
                    temperature=self.llm_config.temperature,
                    messages=messages,
                    tools=tools
                )
                return response
            except Exception as e:
                if attempt == self.llm_config.max_retries - 1:
                    raise
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
```

#### Tool Integration

```python
# File: tools.py

from greenlang.calculators.fuel import calculate_fuel_emissions

def calculate_emissions_tool(fuel_type: str, amount: float, unit: str) -> Dict[str, Any]:
    """
    Calculate fuel emissions tool.

    DETERMINISTIC: This tool uses zero-hallucination calculator.

    Tool Definition from AgentSpec:
    - Name: calculate_emissions
    - Implementation: greenlang.calculators.fuel:calculate_fuel_emissions
    - Safety: deterministic

    Args:
        fuel_type: Type of fuel
        amount: Fuel consumption amount
        unit: Unit of measurement

    Returns:
        Emissions calculation result with citations
    """
    # Call GreenLang calculator (zero-hallucination)
    result = calculate_fuel_emissions(
        fuel_type=fuel_type,
        amount=amount,
        unit=unit
    )

    return {
        "emissions_kg_co2e": result.emissions_kg_co2e,
        "emission_factor": result.emission_factor,
        "citations": result.citations,
        "calculation_hash": result.calculation_hash
    }


# Tool schema for LLM
CALCULATE_EMISSIONS_TOOL_SCHEMA = {
    "name": "calculate_emissions",
    "description": "Calculate fuel combustion emissions using deterministic calculator",
    "input_schema": {
        "type": "object",
        "required": ["fuel_type", "amount", "unit"],
        "properties": {
            "fuel_type": {
                "type": "string",
                "enum": ["natural_gas", "coal", "diesel"],
                "description": "Type of fuel"
            },
            "amount": {
                "type": "number",
                "minimum": 0,
                "description": "Fuel consumption amount"
            },
            "unit": {
                "type": "string",
                "description": "Unit of measurement"
            }
        }
    }
}


def get_tool_schemas() -> List[Dict]:
    """Get all tool schemas for LLM."""
    return [CALCULATE_EMISSIONS_TOOL_SCHEMA]


def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Execute tool by name."""
    if tool_name == "calculate_emissions":
        return calculate_emissions_tool(**tool_input)
    else:
        raise ValueError(f"Unknown tool: {tool_name}")
```

#### RAG Integration

```python
from greenlang.rag import RAGRetriever

class FuelAgentAI(BaseAgent):
    """Fuel Agent with RAG integration."""

    def __init__(self, config: FuelAgentConfig):
        super().__init__(config)

        # RAG collections from AgentSpec
        self.rag_retriever = RAGRetriever(
            collections=["ghg_protocol_corp", "ipcc_ar6"]
        )

    def _get_context(self, query: str) -> str:
        """Retrieve relevant context from RAG."""
        results = self.rag_retriever.retrieve(
            query=query,
            max_results=5
        )
        return "\n\n".join([r.content for r in results])

    def process(self, input_data: FuelInput) -> FuelOutput:
        """Process with RAG context."""
        # Retrieve context
        context = self._get_context(
            f"Calculate {input_data.fuel_type} emissions for {input_data.amount} {input_data.unit}"
        )

        # Add context to system prompt
        system_prompt = get_system_prompt({"rag_context": context})

        # ... rest of processing
```

---

## 5. Factors Section Mapping

### 5.1 Emission Factors Reference

```yaml
compute:
  factors:
    - ref: "ef://ipcc/natural-gas-combustion"
      gwp_set: "AR6GWP100"
    - ref: "ef://epa/fossil-fuels"
      gwp_set: "AR6GWP100"
```

### 5.2 Generated Code

```python
from greenlang.emission_factors import EmissionFactorRegistry

class FuelAgentAI(BaseAgent):
    """Fuel Agent with emission factor integration."""

    # Emission factor references from AgentSpec
    EMISSION_FACTORS = [
        {"ref": "ef://ipcc/natural-gas-combustion", "gwp_set": "AR6GWP100"},
        {"ref": "ef://epa/fossil-fuels", "gwp_set": "AR6GWP100"}
    ]

    def __init__(self, config: FuelAgentConfig):
        super().__init__(config)
        self.ef_registry = EmissionFactorRegistry()
        self._load_emission_factors()

    def _load_emission_factors(self) -> None:
        """Load emission factors from registry."""
        for ef_spec in self.EMISSION_FACTORS:
            ef = self.ef_registry.get(
                ref=ef_spec["ref"],
                gwp_set=ef_spec["gwp_set"]
            )
            logger.info(f"Loaded emission factor: {ef.ref} (CID: {ef.cid})")

    def _get_emission_factor(self, fuel_type: str, country: str = "US") -> EmissionFactor:
        """Get emission factor for fuel type and country."""
        # Search loaded factors
        for ef_spec in self.EMISSION_FACTORS:
            ef = self.ef_registry.get(
                ref=ef_spec["ref"],
                gwp_set=ef_spec["gwp_set"],
                filters={"fuel_type": fuel_type, "country": country}
            )
            if ef:
                return ef

        raise ValueError(f"No emission factor found for {fuel_type} in {country}")
```

---

## 6. Realtime Section Mapping

### 6.1 Realtime Configuration

```yaml
realtime:
  default_mode: "replay"
  snapshot_path: "./snapshots/"
  connectors: []
```

### 6.2 Generated Code

```python
from greenlang.realtime import RealtimeMode, SnapshotManager
from pathlib import Path

class RealtimeConfig(BaseModel):
    """Realtime configuration from AgentSpec."""

    default_mode: Literal["live", "replay"] = Field(
        default="replay",
        description="Default realtime mode"
    )

    snapshot_path: Path = Field(
        default=Path("./snapshots/"),
        description="Path to snapshot storage"
    )


class FuelAgentAI(BaseAgent):
    """Fuel Agent with realtime capabilities."""

    def __init__(self, config: FuelAgentConfig):
        super().__init__(config)
        self.realtime_config = RealtimeConfig()
        self.snapshot_manager = SnapshotManager(
            snapshot_path=self.realtime_config.snapshot_path
        )

    def process(
        self,
        input_data: FuelInput,
        mode: Optional[RealtimeMode] = None
    ) -> FuelOutput:
        """Process with realtime mode support."""
        mode = mode or self.realtime_config.default_mode

        if mode == "replay":
            # Use snapshot for deterministic replay
            snapshot = self.snapshot_manager.load_snapshot(
                agent_id=self.AGENT_ID,
                input_hash=self._hash_input(input_data)
            )
            if snapshot:
                logger.info("Using snapshot for replay mode")
                return FuelOutput(**snapshot.output)

        # Live mode - process normally
        result = self._process_core_logic(input_data)

        # Save snapshot for future replay
        if mode == "live":
            self.snapshot_manager.save_snapshot(
                agent_id=self.AGENT_ID,
                input_data=input_data,
                output_data=result
            )

        return result
```

---

## 7. Provenance Section Mapping

### 7.1 Provenance Configuration

```yaml
provenance:
  pin_ef: true
  gwp_set: "AR6GWP100"
  record: ["seed", "temperature", "tool_calls", "ef_cids", "calculation_steps"]
```

### 7.2 Generated Code

```python
from greenlang.provenance import ProvenanceTracker, ProvenanceRecord
import hashlib

class ProvenanceConfig(BaseModel):
    """Provenance configuration from AgentSpec."""

    pin_ef: bool = Field(
        default=True,
        description="Pin emission factors with content IDs"
    )

    gwp_set: str = Field(
        default="AR6GWP100",
        description="GWP set for emissions"
    )

    record_fields: List[str] = Field(
        default=["seed", "temperature", "tool_calls", "ef_cids", "calculation_steps"],
        description="Fields to record in provenance"
    )


class FuelAgentAI(BaseAgent):
    """Fuel Agent with provenance tracking."""

    def __init__(self, config: FuelAgentConfig):
        super().__init__(config)
        self.provenance_config = ProvenanceConfig()
        self.provenance_tracker = ProvenanceTracker()

    def process(self, input_data: FuelInput) -> FuelOutput:
        """Process with complete provenance tracking."""
        # Initialize provenance record
        provenance = ProvenanceRecord(
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            gwp_set=self.provenance_config.gwp_set
        )

        # Track seed (if deterministic)
        if "seed" in self.provenance_config.record_fields:
            provenance.seed = self.config.seed

        # Track temperature
        if "temperature" in self.provenance_config.record_fields:
            provenance.temperature = self.llm_config.temperature

        # Process with tool call tracking
        tool_calls = []
        result = self._process_with_tracking(input_data, tool_calls)

        # Track tool calls
        if "tool_calls" in self.provenance_config.record_fields:
            provenance.tool_calls = tool_calls

        # Track emission factor CIDs
        if "ef_cids" in self.provenance_config.record_fields:
            provenance.ef_cids = self._extract_ef_cids(tool_calls)

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            input_data, result, provenance
        )

        # Save provenance record
        self.provenance_tracker.save(provenance)

        return FuelOutput(
            **result,
            provenance_hash=provenance_hash
        )

    def _calculate_provenance_hash(
        self,
        input_data: FuelInput,
        output_data: Dict,
        provenance: ProvenanceRecord
    ) -> str:
        """Calculate SHA-256 hash for complete audit trail."""
        provenance_str = (
            f"{input_data.json()}"
            f"{json.dumps(output_data, sort_keys=True)}"
            f"{provenance.json()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _extract_ef_cids(self, tool_calls: List[Dict]) -> List[str]:
        """Extract emission factor CIDs from tool calls."""
        ef_cids = []
        for call in tool_calls:
            if "citations" in call.get("result", {}):
                for citation in call["result"]["citations"]:
                    if "ef_cid" in citation:
                        ef_cids.append(citation["ef_cid"])
        return ef_cids
```

---

## 8. Tools Section Detailed Mapping

### 8.1 Tool Definition

```yaml
ai:
  tools:
    - name: "calculate_emissions"
      description: "Calculate fuel combustion emissions"
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
      impl: "python://greenlang.calculators.fuel:calculate_fuel_emissions"
      safety: "deterministic"
```

### 8.2 Generated Tool Wrapper

```python
from typing import Dict, Any, TypedDict
from pydantic import BaseModel, Field, validator
from greenlang.calculators.fuel import calculate_fuel_emissions


class CalculateEmissionsInput(BaseModel):
    """Input schema for calculate_emissions tool."""

    fuel_type: str = Field(
        ...,
        description="Type of fuel"
    )

    amount: float = Field(
        ...,
        description="Fuel consumption amount"
    )

    unit: str = Field(
        ...,
        description="Unit of measurement"
    )


class CalculateEmissionsOutput(BaseModel):
    """Output schema for calculate_emissions tool."""

    emissions_kg_co2e: float = Field(
        ...,
        description="CO2 equivalent emissions in kg"
    )


def calculate_emissions_tool(
    fuel_type: str,
    amount: float,
    unit: str
) -> CalculateEmissionsOutput:
    """
    Calculate fuel combustion emissions.

    SAFETY: deterministic (zero-hallucination calculator)

    Implementation: greenlang.calculators.fuel:calculate_fuel_emissions

    Args:
        fuel_type: Type of fuel
        amount: Fuel consumption amount
        unit: Unit of measurement

    Returns:
        Emissions calculation result

    Raises:
        ValueError: If input validation fails
        CalculationError: If calculation fails
    """
    # Validate input
    input_data = CalculateEmissionsInput(
        fuel_type=fuel_type,
        amount=amount,
        unit=unit
    )

    # Call GreenLang calculator (imported from impl path)
    result = calculate_fuel_emissions(
        fuel_type=input_data.fuel_type,
        amount=input_data.amount,
        unit=input_data.unit
    )

    # Validate output
    output = CalculateEmissionsOutput(
        emissions_kg_co2e=result.emissions_kg_co2e
    )

    return output


# LLM tool schema
CALCULATE_EMISSIONS_TOOL_SCHEMA = {
    "name": "calculate_emissions",
    "description": "Calculate fuel combustion emissions",
    "input_schema": {
        "type": "object",
        "required": ["fuel_type", "amount", "unit"],
        "properties": {
            "fuel_type": {"type": "string"},
            "amount": {"type": "number"},
            "unit": {"type": "string"}
        }
    }
}
```

---

## 9. Graph Configuration Mapping

### 9.1 Generated LangGraph Configuration

```yaml
# File: graph/agent_graph.yaml
# Auto-generated from AgentSpec v2: emissions/fuel_agent_v1

name: fuel_agent_graph
version: 1.0.0

nodes:
  - id: validate_input
    type: validator
    implementation: fuel_agent_ai.validators.validate_input
    config:
      schema: FuelInput
      strict: true

  - id: retrieve_context
    type: retriever
    implementation: fuel_agent_ai.rag.retrieve_context
    config:
      collections: ["ghg_protocol_corp", "ipcc_ar6"]
      max_results: 5

  - id: llm_orchestrator
    type: llm
    implementation: fuel_agent_ai.llm.call_llm
    config:
      model: claude-3-opus-20240229
      temperature: 0.0
      max_tokens: 15000
      tools:
        - calculate_emissions

  - id: validate_output
    type: validator
    implementation: fuel_agent_ai.validators.validate_output
    config:
      schema: FuelOutput
      strict: true

  - id: track_provenance
    type: tracker
    implementation: fuel_agent_ai.provenance.track
    config:
      fields: ["seed", "temperature", "tool_calls", "ef_cids"]

edges:
  - from: __start__
    to: validate_input

  - from: validate_input
    to: retrieve_context

  - from: retrieve_context
    to: llm_orchestrator

  - from: llm_orchestrator
    to: validate_output

  - from: validate_output
    to: track_provenance

  - from: track_provenance
    to: __end__

error_handling:
  - node: validate_input
    on_error: return_error
  - node: llm_orchestrator
    on_error: retry_3_times
  - node: validate_output
    on_error: return_error
```

---

## 10. Test Generation Mapping

### 10.1 Generated Test Suite

```python
# File: tests/test_fuel_agent.py
# Auto-generated from AgentSpec v2

import pytest
from pathlib import Path
from fuel_agent_ai import FuelAgentAI, FuelInput, FuelOutput, FuelAgentConfig


class TestFuelAgent:
    """Test suite for Fuel Agent."""

    @pytest.fixture
    def agent(self):
        """Create agent instance for testing."""
        config = FuelAgentConfig(deterministic=True, seed=42)
        return FuelAgentAI(config)

    @pytest.fixture
    def valid_input(self):
        """Valid input fixture."""
        return FuelInput(
            fuel_type="natural_gas",
            amount=1000.0,
            country="US",
            renewable_percentage=0.0
        )

    def test_valid_input_processes_successfully(self, agent, valid_input):
        """Test that valid input processes successfully."""
        result = agent.process(valid_input)

        assert isinstance(result, FuelOutput)
        assert result.co2e_emissions_kg > 0
        assert result.provenance_hash is not None
        assert result.processing_time_ms > 0

    def test_missing_required_field_raises_error(self, agent):
        """Test that missing required field raises validation error."""
        with pytest.raises(ValueError, match="fuel_type"):
            # Missing fuel_type (required field)
            FuelInput(amount=1000.0)

    def test_negative_amount_raises_error(self, agent):
        """Test that negative amount raises validation error."""
        with pytest.raises(ValueError, match="must be non-negative"):
            FuelInput(
                fuel_type="natural_gas",
                amount=-100.0  # Invalid: ge constraint violated
            )

    def test_invalid_percentage_raises_error(self, agent):
        """Test that invalid percentage raises validation error."""
        with pytest.raises(ValueError, match="between 0 and 100"):
            FuelInput(
                fuel_type="natural_gas",
                amount=1000.0,
                renewable_percentage=150.0  # Invalid: le constraint violated
            )

    def test_deterministic_execution(self, agent, valid_input):
        """Test that execution is deterministic."""
        result1 = agent.process(valid_input)
        result2 = agent.process(valid_input)

        assert result1.co2e_emissions_kg == result2.co2e_emissions_kg
        assert result1.provenance_hash == result2.provenance_hash

    def test_provenance_tracking(self, agent, valid_input):
        """Test that provenance is tracked correctly."""
        result = agent.process(valid_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex digest

    def test_output_schema_compliance(self, agent, valid_input):
        """Test that output complies with schema."""
        result = agent.process(valid_input)

        # All required outputs present
        assert hasattr(result, 'co2e_emissions_kg')
        assert hasattr(result, 'emission_factor')
        assert hasattr(result, 'citations')

        # Correct types
        assert isinstance(result.co2e_emissions_kg, float)
        assert isinstance(result.emission_factor, float)
        assert isinstance(result.citations, list)
```

---

## 11. Documentation Generation Mapping

### 11.1 Generated README.md

```markdown
# Fuel Emissions Agent

Calculate fuel combustion emissions

**Agent ID**: emissions/fuel_agent_v1
**Version**: 1.0.0
**License**: MIT
**Owners**: greenlang-team

## Overview

{spec.summary}

This agent was auto-generated from AgentSpec v2 on {timestamp}.

## Features

- **Deterministic**: Guaranteed reproducible results (temperature=0.0, seed=42)
- **Zero Hallucination**: Uses GreenLang calculators for all numeric calculations
- **Complete Provenance**: Full audit trail with SHA-256 hashing
- **Type Safe**: Pydantic models with comprehensive validation
- **Production Ready**: 85%+ test coverage, documented, monitored

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from fuel_agent_ai import FuelAgentAI, FuelInput, FuelAgentConfig

# Create agent
config = FuelAgentConfig(deterministic=True)
agent = FuelAgentAI(config)

# Process input
input_data = FuelInput(
    fuel_type="natural_gas",
    amount=1000.0,
    country="US"
)

result = agent.process(input_data)
print(f"Emissions: {result.co2e_emissions_kg} kg CO2e")
```

## Input Schema

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| fuel_type | string | Yes | Type of fuel | enum: [natural_gas, coal, diesel] |
| amount | float | Yes | Fuel consumption amount | >= 0.0 |
| country | string | No | Country code | default: "US" |
| renewable_percentage | float | No | Renewable energy percentage | 0.0 <= x <= 100.0 |

## Output Schema

| Field | Type | Description |
|-------|------|-------------|
| co2e_emissions_kg | float | Total CO2 equivalent emissions |
| emission_factor | float | Emission factor used |
| citations | array | List of emission factor citations |
| provenance_hash | string | SHA-256 hash for audit trail |

## Tools

### calculate_emissions

Calculate fuel combustion emissions using deterministic calculator.

**Safety**: deterministic (zero-hallucination)
**Implementation**: greenlang.calculators.fuel:calculate_fuel_emissions

## Testing

```bash
pytest tests/
```

## Deployment

See [deployment/](deployment/) for Kubernetes manifests and Docker configuration.

## Documentation

- [Architecture](ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Tool Specifications](docs/TOOLS.md)

## License

MIT

## Support

- Email: greenlang-team@example.com
- Issues: [GitHub Issues](https://github.com/greenlang/agents/issues)
```

---

## 12. Summary: Complete Mapping Table

| AgentSpec Section | Generated Artifact | File Location |
|-------------------|-------------------|---------------|
| `metadata` | Module docstring, class attributes | `agent.py` |
| `compute.entrypoint` | Entrypoint function | `agent.py` |
| `compute.inputs` | Pydantic Input model | `agent.py` |
| `compute.outputs` | Pydantic Output model | `agent.py` |
| `compute.factors` | Emission factor loading | `agent.py` |
| `ai.system_prompt` | Prompt template | `prompts.py` |
| `ai.budget` | LLM config model | `agent.py` |
| `ai.tools` | Tool wrappers + schemas | `tools.py` |
| `ai.rag_collections` | RAG retriever config | `agent.py` |
| `realtime` | Realtime config + snapshot manager | `agent.py` |
| `provenance` | Provenance tracker integration | `agent.py` |
| Complete spec | LangGraph configuration | `graph/agent_graph.yaml` |
| Complete spec | Test suite | `tests/test_agent.py` |
| Complete spec | Documentation | `README.md`, `docs/` |
| Complete spec | Deployment configs | `deployment/` |
| Complete spec | Pack manifest (copy) | `pack.yaml` |

---

**Document Status**: Design Complete
**Next Step**: Implement template system with these mappings
