# Migration Guide: Standardizing to AgentSpecV2Base + Category Mixins

## Overview

This guide provides step-by-step instructions for migrating existing agents to use the standardized `AgentSpecV2Base` inheritance pattern with category mixins.

**Target Architecture:**
```python
AgentSpecV2Base[InT, OutT] + CategoryMixin → Standardized Agent
```

**Why Migrate?**
- **Unified lifecycle**: Standard initialize → validate → execute → finalize flow
- **Type safety**: Generic typing with Pydantic validation
- **Category clarity**: Explicit agent category (DETERMINISTIC, REASONING, INSIGHT)
- **Audit compliance**: Built-in provenance tracking and audit trails
- **Better testing**: Consistent hooks and lifecycle management

---

## Category Mixin Decision Tree

Before migrating, determine which category mixin your agent needs:

### DeterministicMixin
**Use when:** Agent performs pure calculations with ZERO AI/LLM usage
- ✅ Emission calculations (CBAM, CSRD, GHG Protocol)
- ✅ Compliance validation
- ✅ Factor lookups
- ✅ Regulatory reporting
- ❌ NO LLM in calculation path
- ❌ NO creative reasoning

**Example agents:** FuelAgent, CarbonAgent, EmissionsCalculatorAgent

### ReasoningMixin
**Use when:** Agent uses AI for reasoning and recommendations
- ✅ Technology recommendations
- ✅ Strategic planning
- ✅ Optimization analysis
- ✅ What-if scenarios
- ✅ RAG + ChatSession + tools
- ✅ Temperature ≥ 0.5
- ❌ NON-CRITICAL PATH (not for regulatory calculations)

**Example agents:** DecarbonizationRoadmapAgent, RecommendationAgent

### InsightMixin
**Use when:** Agent combines deterministic calculations + AI insights
- ✅ Anomaly detection (ML) + explanation (AI)
- ✅ Forecast calculation (stats) + interpretation (AI)
- ✅ Benchmark calculation (math) + insights (AI)
- ✅ Deterministic numbers + AI narratives
- ✅ Temperature ≤ 0.7 for consistency

**Example agents:** AnomalyInvestigationAgent, ForecastExplanationAgent

---

## Migration Steps

### Step 1: Identify Current Pattern

Check your agent's current base class:

```python
# Pattern 1: Generic Agent[InT, OutT]
class FuelAgent(Agent[FuelInput, FuelOutput]):
    ...

# Pattern 2: BaseAgent
class CarbonAgent(BaseAgent):
    ...

# Pattern 3: Category-specific base (old pattern)
class AnomalyAgent(InsightAgent):
    ...

# Pattern 4: No standard base
class CustomAgent:
    ...
```

### Step 2: Choose Category Mixin

Based on the decision tree above, select the appropriate mixin:
- **DeterministicMixin**: Zero-hallucination calculations
- **ReasoningMixin**: AI-powered reasoning
- **InsightMixin**: Hybrid calculation + AI

### Step 3: Define Input/Output Models (Pydantic)

Ensure your agent has Pydantic models for inputs and outputs:

```python
from pydantic import BaseModel, Field

class FuelInput(BaseModel):
    """Input model for FuelAgent."""
    fuel_type: str = Field(..., description="Type of fuel (natural_gas, diesel, etc.)")
    amount: float = Field(..., ge=0, description="Fuel consumption amount")
    unit: str = Field(..., description="Unit of measurement (kWh, gallons, etc.)")
    country: str = Field(default="US", description="Country code for emission factors")
    year: int = Field(default=2025, ge=2000, le=2050, description="Year for emission factors")

class FuelOutput(BaseModel):
    """Output model for FuelAgent."""
    co2e_emissions_kg: float = Field(..., description="CO2e emissions in kilograms")
    fuel_type: str = Field(..., description="Fuel type processed")
    emission_factor: float = Field(..., description="Emission factor used")
    emission_factor_unit: str = Field(..., description="Emission factor unit")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
```

### Step 4: Migrate Agent Class

#### BEFORE (Old Pattern):
```python
from greenlang.types import Agent, AgentResult

class FuelAgent(Agent[FuelInput, FuelOutput]):
    """Agent for calculating emissions based on fuel consumption."""

    agent_id: str = "fuel"
    name: str = "Fuel Emissions Calculator"
    version: str = "0.0.1"

    def __init__(self) -> None:
        self.emission_factors = EmissionFactors()
        # ... setup code ...

    def validate(self, payload: FuelInput) -> bool:
        # Validation logic
        return True

    def run(self, payload: FuelInput) -> AgentResult[FuelOutput]:
        # Calculation logic
        ...
```

#### AFTER (New Pattern):
```python
from greenlang.agents.agentspec_v2_base import AgentSpecV2Base, AgentExecutionContext
from greenlang.agents.mixins import DeterministicMixin

class FuelAgent(AgentSpecV2Base[FuelInput, FuelOutput], DeterministicMixin):
    """
    Agent for calculating emissions based on fuel consumption.

    Category: DETERMINISTIC
    - Zero-hallucination guarantee
    - Full audit trail
    - No LLM usage in calculation path
    - Regulatory compliance ready
    """

    def __init__(self, **kwargs):
        """Initialize FuelAgent."""
        super().__init__(
            agent_id="fuel_agent_v2",
            enable_metrics=True,
            enable_citations=False,  # Not needed for deterministic calculations
            enable_validation=True,
            **kwargs
        )
        # Custom initialization
        self.emission_factors = EmissionFactors()

    def initialize_impl(self) -> None:
        """Custom initialization logic."""
        # Load resources, setup databases, etc.
        self.logger.info("FuelAgent initialized")

    def validate_input_impl(self, input_data: FuelInput, context: AgentExecutionContext) -> FuelInput:
        """Custom input validation logic."""
        # Additional validation beyond Pydantic schema
        if input_data.amount < 0:
            raise ValueError("Fuel amount cannot be negative")

        return input_data

    def execute_impl(self, validated_input: FuelInput, context: AgentExecutionContext) -> FuelOutput:
        """
        Core execution logic - ZERO HALLUCINATION.

        This is the main calculation method. It MUST be deterministic.
        NO LLM calls allowed here.
        """
        # 1. Get emission factor (deterministic database lookup)
        emission_factor = self.emission_factors.get_factor(
            fuel_type=validated_input.fuel_type,
            unit=validated_input.unit,
            region=validated_input.country
        )

        # 2. Calculate emissions (pure arithmetic)
        co2e_emissions_kg = validated_input.amount * emission_factor

        # 3. Capture audit trail (required for DeterministicMixin)
        calculation_trace = [
            f"amount = {validated_input.amount} {validated_input.unit}",
            f"emission_factor = {emission_factor} kgCO2e/{validated_input.unit}",
            f"emissions = {validated_input.amount} * {emission_factor} = {co2e_emissions_kg} kg CO2e"
        ]

        self.capture_audit_entry(
            operation="fuel_emissions_calculation",
            inputs=validated_input.dict(),
            outputs={"co2e_emissions_kg": co2e_emissions_kg},
            calculation_trace=calculation_trace,
            metadata={
                "country": validated_input.country,
                "year": validated_input.year
            }
        )

        # 4. Calculate provenance hash
        provenance_hash = self.calculate_provenance_hash(
            inputs=validated_input.dict(),
            outputs={"co2e_emissions_kg": co2e_emissions_kg}
        )

        # 5. Return validated output
        return FuelOutput(
            co2e_emissions_kg=co2e_emissions_kg,
            fuel_type=validated_input.fuel_type,
            emission_factor=emission_factor,
            emission_factor_unit=f"kgCO2e/{validated_input.unit}",
            provenance_hash=provenance_hash
        )

    def validate_output_impl(self, output: FuelOutput, context: AgentExecutionContext) -> FuelOutput:
        """Custom output validation logic."""
        # Validate output makes sense
        if output.co2e_emissions_kg < 0:
            raise ValueError("Emissions cannot be negative")

        return output

    def finalize_impl(self, result: AgentResult[FuelOutput], context: AgentExecutionContext) -> AgentResult[FuelOutput]:
        """Custom finalization logic."""
        # Add audit trail to result metadata
        result.metadata["audit_trail_count"] = len(self.get_audit_trail())

        return result
```

### Step 5: Update Tests

#### BEFORE:
```python
def test_fuel_agent():
    agent = FuelAgent()

    payload = {
        "fuel_type": "natural_gas",
        "amount": 100,
        "unit": "therms"
    }

    result = agent.run(payload)

    assert result["success"] is True
    assert "co2e_emissions_kg" in result["data"]
```

#### AFTER:
```python
def test_fuel_agent():
    agent = FuelAgent()

    # Use Pydantic model
    input_data = FuelInput(
        fuel_type="natural_gas",
        amount=100,
        unit="therms"
    )

    # Call run() method
    result = agent.run(input_data)

    # Validate result
    assert result.success is True
    assert "co2e_emissions_kg" in result.data
    assert "provenance_hash" in result.data

    # Validate audit trail (for DeterministicMixin)
    audit_trail = agent.get_audit_trail()
    assert len(audit_trail) > 0
    assert audit_trail[0]["operation"] == "fuel_emissions_calculation"
```

---

## Complete Migration Examples

### Example 1: DeterministicMixin (FuelAgent)

```python
from greenlang.agents.agentspec_v2_base import AgentSpecV2Base, AgentExecutionContext
from greenlang.agents.mixins import DeterministicMixin
from pydantic import BaseModel, Field

class FuelInput(BaseModel):
    fuel_type: str
    amount: float = Field(ge=0)
    unit: str

class FuelOutput(BaseModel):
    co2e_emissions_kg: float
    provenance_hash: str

class FuelAgent(AgentSpecV2Base[FuelInput, FuelOutput], DeterministicMixin):
    """Deterministic fuel emissions calculator."""

    def execute_impl(self, validated_input: FuelInput, context: AgentExecutionContext) -> FuelOutput:
        # Pure calculation - NO LLM
        emission_factor = self.get_emission_factor(validated_input.fuel_type, validated_input.unit)
        emissions = validated_input.amount * emission_factor

        # Audit trail
        self.capture_audit_entry(
            operation="emissions_calc",
            inputs=validated_input.dict(),
            outputs={"emissions": emissions},
            calculation_trace=[f"{validated_input.amount} * {emission_factor} = {emissions}"]
        )

        # Provenance hash
        provenance_hash = self.calculate_provenance_hash(
            inputs=validated_input.dict(),
            outputs={"emissions": emissions}
        )

        return FuelOutput(co2e_emissions_kg=emissions, provenance_hash=provenance_hash)
```

### Example 2: ReasoningMixin (DecarbonizationAgent)

```python
from greenlang.agents.agentspec_v2_base import AgentSpecV2Base, AgentExecutionContext
from greenlang.agents.mixins import ReasoningMixin
from pydantic import BaseModel

class PlanInput(BaseModel):
    industry: str
    current_emissions: float
    target_year: int

class PlanOutput(BaseModel):
    recommendations: list
    implementation_plan: str

class DecarbonizationAgent(AgentSpecV2Base[PlanInput, PlanOutput], ReasoningMixin):
    """AI-powered decarbonization planner."""

    async def execute_impl(self, validated_input: PlanInput, context: AgentExecutionContext) -> PlanOutput:
        # 1. RAG retrieval
        knowledge = await self.rag_retrieve(
            query=f"Decarbonization for {validated_input.industry}",
            collections=["case_studies", "best_practices"],
            top_k=5
        )

        # 2. LLM reasoning (temperature=0.7 for creativity)
        response = await self._chat_session.chat(
            messages=[{
                "role": "user",
                "content": f"Create decarbonization plan for {validated_input.industry}. Context: {self.format_rag_results(knowledge)}"
            }],
            temperature=0.7
        )

        return PlanOutput.parse_obj(response)
```

### Example 3: InsightMixin (AnomalyAgent)

```python
from greenlang.agents.agentspec_v2_base import AgentSpecV2Base, AgentExecutionContext
from greenlang.agents.mixins import InsightMixin
from pydantic import BaseModel

class AnomalyInput(BaseModel):
    data: list
    metric: str

class AnomalyOutput(BaseModel):
    anomalies: list
    explanation: str

class AnomalyAgent(AgentSpecV2Base[AnomalyInput, AnomalyOutput], InsightMixin):
    """Hybrid anomaly detector + explainer."""

    def execute_impl(self, validated_input: AnomalyInput, context: AgentExecutionContext) -> AnomalyOutput:
        # 1. DETERMINISTIC: Detect anomalies (no LLM)
        anomalies = self.isolation_forest.detect(validated_input.data)

        # Audit trail for calculation
        self.capture_calculation_audit(
            operation="anomaly_detection",
            inputs=validated_input.dict(),
            outputs={"anomalies": anomalies},
            calculation_trace=["IsolationForest.fit_predict()"]
        )

        # 2. AI-POWERED: Explain anomalies (with LLM)
        explanation = await self._generate_explanation(anomalies, validated_input)

        return AnomalyOutput(anomalies=anomalies, explanation=explanation)

    async def _generate_explanation(self, anomalies, input_data):
        # RAG for context
        context = await self.rag_retrieve(
            query=f"Anomalies in {input_data.metric}",
            collections=["historical_data"],
            top_k=3
        )

        # LLM for explanation (temperature=0.6 for consistency)
        response = await self._chat_session.chat(
            messages=[{
                "role": "user",
                "content": f"Explain these anomalies: {anomalies}. Context: {context}"
            }],
            temperature=0.6
        )

        return response.text
```

---

## Migration Checklist

Use this checklist when migrating an agent:

- [ ] **Choose category mixin** (DeterministicMixin, ReasoningMixin, or InsightMixin)
- [ ] **Create Pydantic models** for Input and Output
- [ ] **Update class inheritance** to `AgentSpecV2Base[InT, OutT] + CategoryMixin`
- [ ] **Implement `execute_impl()`** method (required)
- [ ] **Add audit trail** (for DeterministicMixin and InsightMixin)
- [ ] **Calculate provenance hash** (for DeterministicMixin)
- [ ] **Update tests** to use Pydantic models and new API
- [ ] **Validate determinism** (for DeterministicMixin - no LLM in calculation)
- [ ] **Set up RAG/ChatSession** (for ReasoningMixin and InsightMixin)
- [ ] **Document category** in agent docstring
- [ ] **Export audit trail** capability (for compliance agents)

---

## Common Migration Pitfalls

### Pitfall 1: Using LLM in DeterministicMixin
❌ **WRONG:**
```python
class EmissionsAgent(AgentSpecV2Base[Input, Output], DeterministicMixin):
    def execute_impl(self, validated_input, context):
        # ❌ NO LLM ALLOWED IN DETERMINISTIC AGENTS!
        emissions = await llm.calculate_emissions(validated_input)
        return Output(emissions=emissions)
```

✅ **CORRECT:**
```python
class EmissionsAgent(AgentSpecV2Base[Input, Output], DeterministicMixin):
    def execute_impl(self, validated_input, context):
        # ✅ Pure calculation only
        emission_factor = self.db.lookup_factor(validated_input.fuel_type)
        emissions = validated_input.amount * emission_factor
        return Output(emissions=emissions)
```

### Pitfall 2: Forgetting Audit Trail
❌ **WRONG:**
```python
class FuelAgent(AgentSpecV2Base[Input, Output], DeterministicMixin):
    def execute_impl(self, validated_input, context):
        emissions = validated_input.amount * emission_factor
        # ❌ Missing audit trail!
        return Output(emissions=emissions)
```

✅ **CORRECT:**
```python
class FuelAgent(AgentSpecV2Base[Input, Output], DeterministicMixin):
    def execute_impl(self, validated_input, context):
        emissions = validated_input.amount * emission_factor

        # ✅ Capture audit trail
        self.capture_audit_entry(
            operation="emissions_calc",
            inputs=validated_input.dict(),
            outputs={"emissions": emissions},
            calculation_trace=[f"{validated_input.amount} * {emission_factor}"]
        )

        return Output(emissions=emissions)
```

### Pitfall 3: Multiple Category Mixins
❌ **WRONG:**
```python
# ❌ Cannot inherit from multiple category mixins!
class MyAgent(AgentSpecV2Base[Input, Output], DeterministicMixin, ReasoningMixin):
    ...
```

✅ **CORRECT:**
```python
# ✅ Choose ONE category mixin
# If you need both calculation + reasoning, use InsightMixin
class MyAgent(AgentSpecV2Base[Input, Output], InsightMixin):
    def execute_impl(self, validated_input, context):
        # Deterministic calculation
        result = self.calculate(validated_input)

        # AI explanation
        explanation = await self.explain(result, validated_input)

        return Output(result=result, explanation=explanation)
```

---

## Validation Script

After migration, run the validation script to ensure compliance:

```bash
python scripts/validate_agent_inheritance.py
```

This will check:
- ✅ All agents inherit from `AgentSpecV2Base`
- ✅ Each agent has exactly one category mixin
- ✅ DeterministicMixin agents have no LLM calls in `execute_impl()`
- ✅ Audit trail methods are implemented correctly
- ✅ Pydantic models are properly defined

---

## Support

If you encounter issues during migration:
1. Check this guide for examples
2. Review existing migrated agents (FuelAgent, CarbonAgent)
3. Run validation script for specific errors
4. Contact GreenLang framework team for assistance

---

## Version History

- **v1.0.0** (2025-12-01): Initial migration guide
- Category mixins (DeterministicMixin, ReasoningMixin, InsightMixin)
- AgentSpecV2Base standardization
- Complete migration examples and checklists
