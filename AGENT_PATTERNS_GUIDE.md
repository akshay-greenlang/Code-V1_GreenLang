# Agent Patterns Guide - Intelligence Paradox Architecture

**Version:** 1.0
**Date:** 2025-11-06
**Purpose:** Comprehensive patterns for three agent categories

---

## Table of Contents

1. [Pattern Overview](#pattern-overview)
2. [Pattern 1: Deterministic Agent (CRITICAL PATH)](#pattern-1-deterministic-agent)
3. [Pattern 2: Reasoning Agent (RECOMMENDATION PATH)](#pattern-2-reasoning-agent)
4. [Pattern 3: Insight Agent (INSIGHT PATH)](#pattern-3-insight-agent)
5. [Pattern Selection Decision Tree](#pattern-selection-decision-tree)
6. [Migration Guide](#migration-guide)
7. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)

---

## Pattern Overview

### Three Agent Architectures

```
┌────────────────────────────────────────────────────────────┐
│  PATTERN 1: DETERMINISTIC AGENT (CRITICAL PATH)            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  INPUT → CALCULATION → AUDIT TRAIL → OUTPUT          │  │
│  └──────────────────────────────────────────────────────┘  │
│  Zero AI | 100% Reproducible | Regulatory Compliance      │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  PATTERN 2: REASONING AGENT (RECOMMENDATION PATH)          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  INPUT → RAG → LLM REASONING → TOOL ORCHESTRATION    │  │
│  │         ↓                      ↓                      │  │
│  │    KNOWLEDGE              DETERMINISTIC TOOLS         │  │
│  └──────────────────────────────────────────────────────┘  │
│  Full AI | Creative | Non-Critical                         │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  PATTERN 3: INSIGHT AGENT (HYBRID)                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  INPUT → DETERMINISTIC CALCULATION → NUMBERS         │  │
│  │            ↓                                          │  │
│  │         NUMBERS + RAG → LLM EXPLANATION → INSIGHTS   │  │
│  └──────────────────────────────────────────────────────┘  │
│  Numbers Deterministic | Insights AI-Generated             │
└────────────────────────────────────────────────────────────┘
```

---

## Pattern 1: Deterministic Agent (CRITICAL PATH)

### When to Use

✅ **YES** - Use for:
- Regulatory calculations (CBAM, CSRD, GHG Protocol)
- Compliance validation
- Emission factor lookups
- Audit-critical calculations
- Financial reporting metrics

❌ **NO** - Don't use for:
- Recommendations
- Insights
- Narrative generation
- Strategic planning

### Architecture

```python
from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from typing import Dict, Any, List


class EmissionsCalculator(DeterministicAgent):
    """
    Example CRITICAL PATH agent for emissions calculations.

    Follows Zero Hallucination Guarantee pattern.
    """

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="emissions_calculator",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="GHG Protocol Scope 1 emissions calculator"
    )

    def __init__(self, factor_database):
        super().__init__(enable_audit_trail=True)
        self.factor_db = factor_database

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate emissions using GHG Protocol methodology.

        Deterministic calculation with full audit trail.
        """
        # Step 1: Extract inputs
        consumption_kwh = inputs["consumption_kwh"]
        fuel_type = inputs["fuel_type"]
        region = inputs["region"]

        # Step 2: Lookup emission factor (deterministic)
        emission_factor = self.factor_db.get_factor(
            fuel_type=fuel_type,
            region=region,
            year=inputs.get("year", 2024)
        )

        # Step 3: Calculate emissions (deterministic formula)
        emissions_tco2e = consumption_kwh * emission_factor

        # Step 4: Build calculation trace
        calculation_trace = [
            f"Input: {consumption_kwh} kWh of {fuel_type} in {region}",
            f"Lookup: Emission factor = {emission_factor} kg CO2e/kWh",
            f"Formula: Emissions = Consumption × Emission Factor",
            f"Calculation: {emissions_tco2e} = {consumption_kwh} × {emission_factor}",
            f"Output: {emissions_tco2e} tCO2e"
        ]

        # Step 5: Prepare outputs
        outputs = {
            "emissions_tco2e": emissions_tco2e,
            "emission_factor": emission_factor,
            "methodology": "GHG Protocol Scope 1",
            "factor_source": self.factor_db.get_source(fuel_type, region),
            "calculation_date": datetime.utcnow().isoformat() + "Z"
        }

        # Step 6: Capture audit trail
        self._capture_audit_entry(
            operation="calculate_emissions",
            inputs=inputs,
            outputs=outputs,
            calculation_trace=calculation_trace,
            metadata={
                "ghg_protocol_version": "2023",
                "factor_vintage": "2024"
            }
        )

        return outputs
```

### Key Principles

1. **Zero AI**: No LLM calls, no ChatSession, no RAG
2. **Deterministic**: Same inputs always produce same outputs
3. **Traceable**: Full calculation trace for auditors
4. **Fast**: No network calls, pure computation
5. **Reproducible**: Hash inputs and outputs for verification

### Testing Pattern

```python
import pytest


def test_emissions_calculator_deterministic():
    """Verify deterministic behavior."""
    calculator = EmissionsCalculator(factor_db)

    inputs = {
        "consumption_kwh": 10000,
        "fuel_type": "natural_gas",
        "region": "US"
    }

    # Run twice
    result1 = calculator.execute(inputs)
    result2 = calculator.execute(inputs)

    # Must be identical
    assert result1 == result2
    assert result1["emissions_tco2e"] == result2["emissions_tco2e"]


def test_emissions_calculator_audit_trail():
    """Verify audit trail capture."""
    calculator = EmissionsCalculator(factor_db)

    inputs = {"consumption_kwh": 5000, "fuel_type": "coal", "region": "EU"}
    calculator.execute(inputs)

    # Check audit trail
    audit = calculator.get_audit_trail()
    assert len(audit) == 1
    assert audit[0].operation == "calculate_emissions"
    assert audit[0].input_hash is not None
    assert len(audit[0].calculation_trace) > 0


def test_emissions_calculator_reproducibility():
    """Verify reproducibility across runs."""
    calculator = EmissionsCalculator(factor_db)

    inputs = {"consumption_kwh": 7500, "fuel_type": "diesel", "region": "UK"}
    expected = {"emissions_tco2e": 2.1825}  # Pre-computed expected value

    is_reproducible, error = calculator.verify_reproducibility(inputs, expected)
    assert is_reproducible, error
```

---

## Pattern 2: Reasoning Agent (RECOMMENDATION PATH)

### When to Use

✅ **YES** - Use for:
- Technology recommendations
- Strategic planning
- Optimization analysis
- What-if scenarios
- Decision support

❌ **NO** - Don't use for:
- Regulatory calculations
- Compliance validation
- Financial reporting
- Audit-critical data

### Architecture

```python
from greenlang.agents.base_agents import ReasoningAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from typing import Dict, Any, List


class DecarbonizationPlanner(ReasoningAgent):
    """
    Example RECOMMENDATION PATH agent using full AI reasoning.

    Uses RAG + ChatSession + Multi-Tool Orchestration.
    """

    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name="decarbonization_planner",
        category=AgentCategory.RECOMMENDATION,
        uses_chat_session=True,
        uses_rag=True,
        uses_tools=True,
        critical_for_compliance=False,
        transformation_priority="LOW (Already AI)",
        description="AI-powered decarbonization roadmap generation"
    )

    async def reason(
        self,
        context: Dict[str, Any],
        session,  # ChatSession instance
        rag_engine,  # RAGEngine instance
        tools: List[Any] = None
    ) -> Dict[str, Any]:
        """
        Generate AI-powered decarbonization roadmap.

        Uses RAG for knowledge, ChatSession for reasoning, tools for validation.
        """
        # Step 1: RAG retrieval for contextual knowledge
        rag_result = await self._rag_retrieve(
            query=f"""
            Decarbonization strategies for {context['industry']} facility
            with {context['annual_consumption_kwh']} kWh/year consumption
            in {context['region']}
            """,
            rag_engine=rag_engine,
            collections=[
                "case_studies",
                "technology_database",
                "best_practices",
                "regulatory_incentives"
            ],
            top_k=8
        )

        formatted_knowledge = self._format_rag_results(rag_result)

        # Step 2: Initial AI reasoning
        initial_response = await session.chat(
            messages=[
                {
                    "role": "system",
                    "content": """You are a decarbonization expert. Analyze the facility
                    and use available tools to develop a comprehensive roadmap. Prioritize
                    cost-effective solutions with proven ROI."""
                },
                {
                    "role": "user",
                    "content": f"""
                    Develop a decarbonization roadmap for this facility:

                    **Facility Profile:**
                    - Industry: {context['industry']}
                    - Annual consumption: {context['annual_consumption_kwh']} kWh
                    - Current emissions: {context['emissions_tco2e']} tCO2e/year
                    - Current heating: {context['heating_system']}
                    - Budget: ${context['budget']}
                    - Space available: {context['space_sqm']} m²
                    - Location: {context['region']}
                    - Target reduction: {context['target_reduction_pct']}%

                    **Relevant Knowledge:**
                    {formatted_knowledge}

                    Use the available tools to:
                    1. Check technology compatibility with this facility
                    2. Calculate financial metrics (ROI, payback period)
                    3. Verify spatial constraints
                    4. Assess grid integration requirements
                    5. Evaluate regulatory compliance
                    6. Model emission reduction scenarios

                    Develop a multi-year roadmap with specific technologies, costs, timelines,
                    and expected reductions.
                    """
                }
            ],
            tools=tools or self._get_default_tools(),
            temperature=0.7,  # Allow creative reasoning
            tool_choice="auto"
        )

        # Step 3: Multi-step tool orchestration
        conversation_history = [initial_response]
        tool_execution_trace = []

        while initial_response.tool_calls:
            tool_results = []

            # Execute all tool calls
            for tool_call in initial_response.tool_calls:
                result = await self._execute_tool(
                    tool_call=tool_call,
                    tool_registry=self._build_tool_registry()
                )

                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "content": json.dumps(result)
                })

                tool_execution_trace.append({
                    "tool": tool_call["name"],
                    "arguments": json.loads(tool_call["arguments"]),
                    "result": result,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })

            # Continue conversation with tool results
            conversation_history.extend(tool_results)

            next_response = await session.chat(
                messages=conversation_history,
                tools=tools or self._get_default_tools(),
                temperature=0.7
            )

            conversation_history.append(next_response)
            initial_response = next_response

        # Step 4: Parse and structure final roadmap
        roadmap = self._parse_roadmap(initial_response.text)

        return {
            "roadmap": roadmap,
            "reasoning": initial_response.text,
            "tool_execution_trace": tool_execution_trace,
            "rag_context": {
                "chunks_retrieved": len(rag_result.chunks),
                "collections_searched": rag_result.collections,
                "relevance_scores": rag_result.relevance_scores
            },
            "confidence": self._extract_confidence(initial_response.text),
            "metadata": {
                "model": initial_response.provider_info["model"],
                "tokens_used": initial_response.usage["total_tokens"],
                "cost_usd": initial_response.usage["total_cost"],
                "tools_called": len(tool_execution_trace),
                "temperature": 0.7
            }
        }

    def _get_default_tools(self) -> List[Any]:
        """Get default tool definitions."""
        from greenlang.intelligence.schemas.tools import ToolDef

        return [
            ToolDef(
                name="technology_compatibility_check",
                description="Check if a technology is compatible with facility constraints",
                parameters={
                    "type": "object",
                    "properties": {
                        "technology": {"type": "string"},
                        "facility_type": {"type": "string"},
                        "consumption_kwh": {"type": "number"}
                    },
                    "required": ["technology", "facility_type"]
                }
            ),
            ToolDef(
                name="financial_analysis",
                description="Calculate ROI, payback period, NPV for technology investment",
                parameters={
                    "type": "object",
                    "properties": {
                        "technology": {"type": "string"},
                        "capex": {"type": "number"},
                        "annual_savings": {"type": "number"},
                        "lifetime_years": {"type": "number"}
                    },
                    "required": ["technology", "capex", "annual_savings"]
                }
            ),
            ToolDef(
                name="emission_reduction_model",
                description="Model emission reduction for a technology",
                parameters={
                    "type": "object",
                    "properties": {
                        "technology": {"type": "string"},
                        "baseline_emissions_tco2e": {"type": "number"}
                    },
                    "required": ["technology", "baseline_emissions_tco2e"]
                }
            )
        ]

    def _build_tool_registry(self) -> Dict[str, Any]:
        """Build registry of tool implementations."""
        return {
            "technology_compatibility_check": self._check_technology_compatibility,
            "financial_analysis": self._analyze_financial,
            "emission_reduction_model": self._model_emission_reduction
        }

    async def _check_technology_compatibility(
        self,
        technology: str,
        facility_type: str,
        consumption_kwh: float = None
    ) -> Dict[str, Any]:
        """Check technology compatibility (deterministic tool)."""
        # Implementation here
        return {"compatible": True, "confidence": 0.9, "notes": "..."}

    async def _analyze_financial(
        self,
        technology: str,
        capex: float,
        annual_savings: float,
        lifetime_years: int = 20
    ) -> Dict[str, Any]:
        """Calculate financial metrics (deterministic tool)."""
        # Implementation here
        payback_years = capex / annual_savings
        return {
            "payback_years": payback_years,
            "roi_pct": (annual_savings * lifetime_years - capex) / capex * 100,
            "npv_usd": "..."  # NPV calculation
        }

    async def _model_emission_reduction(
        self,
        technology: str,
        baseline_emissions_tco2e: float
    ) -> Dict[str, Any]:
        """Model emission reduction (deterministic tool)."""
        # Implementation here
        return {
            "reduction_pct": 65.0,
            "new_emissions_tco2e": baseline_emissions_tco2e * 0.35
        }

    def _parse_roadmap(self, llm_text: str) -> Dict[str, Any]:
        """Parse structured roadmap from LLM text."""
        # Implementation: Extract phases, technologies, costs, timelines
        return {"phases": [...], "total_cost": ..., "total_reduction": ...}

    def _extract_confidence(self, llm_text: str) -> float:
        """Extract confidence score from LLM text."""
        # Implementation: Parse confidence indicators
        return 0.85
```

### Key Principles

1. **RAG First**: Always retrieve knowledge before reasoning
2. **Multi-Tool**: Let LLM orchestrate multiple tool calls
3. **Temperature ≥ 0.5**: Allow creative problem-solving
4. **Structured Output**: Parse LLM text into structured data
5. **Audit LLM Usage**: Track tokens, cost, model used

### Testing Pattern

```python
import pytest
from unittest.mock import Mock, AsyncMock


@pytest.mark.asyncio
async def test_decarbonization_planner_uses_rag():
    """Verify RAG is used for knowledge retrieval."""
    planner = DecarbonizationPlanner()
    mock_session = Mock()
    mock_rag = AsyncMock()

    mock_rag.query.return_value = Mock(chunks=[...])
    mock_session.chat = AsyncMock(return_value=Mock(
        text="Roadmap...",
        tool_calls=None,
        usage={"total_tokens": 1000}
    ))

    context = {
        "industry": "manufacturing",
        "annual_consumption_kwh": 50000,
        "budget": 100000
    }

    result = await planner.reason(context, mock_session, mock_rag)

    # Verify RAG was called
    assert mock_rag.query.called
    assert "case_studies" in mock_rag.query.call_args[1]["collections"]


@pytest.mark.asyncio
async def test_decarbonization_planner_tool_orchestration():
    """Verify tool calls are orchestrated."""
    planner = DecarbonizationPlanner()
    mock_session = Mock()
    mock_rag = AsyncMock()

    # Mock tool calls
    mock_session.chat = AsyncMock(side_effect=[
        Mock(
            text="Checking compatibility...",
            tool_calls=[{
                "id": "call_1",
                "name": "technology_compatibility_check",
                "arguments": json.dumps({"technology": "heat_pump"})
            }]
        ),
        Mock(text="Final roadmap...", tool_calls=None)
    ])

    context = {...}
    result = await planner.reason(context, mock_session, mock_rag)

    # Verify tools were called
    assert len(result["tool_execution_trace"]) > 0
    assert result["tool_execution_trace"][0]["tool"] == "technology_compatibility_check"
```

---

## Pattern 3: Insight Agent (HYBRID)

### When to Use

✅ **YES** - Use for:
- Anomaly investigation
- Forecast explanation
- Benchmark insights
- Trend analysis
- Pattern recognition

❌ **NO** - Don't use for:
- Pure recommendations (use Pattern 2)
- Pure calculations (use Pattern 1)

### Architecture

```python
from greenlang.agents.base_agents import InsightAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from typing import Dict, Any


class AnomalyInvestigator(InsightAgent):
    """
    Example INSIGHT PATH agent with hybrid architecture.

    Numbers are deterministic, insights are AI-generated.
    """

    category = AgentCategory.INSIGHT
    metadata = AgentMetadata(
        name="anomaly_investigator",
        category=AgentCategory.INSIGHT,
        uses_chat_session=True,
        uses_rag=True,
        critical_for_compliance=False,
        transformation_priority="HIGH",
        description="Anomaly detection (deterministic) + root cause investigation (AI)"
    )

    def __init__(self, anomaly_detector):
        super().__init__(enable_audit_trail=True)
        self.detector = anomaly_detector  # sklearn IsolationForest

    def calculate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministic anomaly detection using Isolation Forest.

        No AI involved - pure statistical method.
        """
        # Step 1: Extract data
        emissions_data = inputs["emissions_timeseries"]
        metadata = inputs.get("metadata", {})

        # Step 2: Detect anomalies (deterministic ML model)
        anomaly_scores = self.detector.decision_function(emissions_data)
        anomalies = self.detector.predict(emissions_data)

        # Identify anomaly points
        anomaly_indices = np.where(anomalies == -1)[0]
        anomaly_values = emissions_data[anomaly_indices]

        # Calculate statistics
        baseline_mean = np.mean(emissions_data)
        baseline_std = np.std(emissions_data)

        deviations = []
        for idx, value in zip(anomaly_indices, anomaly_values):
            deviation_pct = ((value - baseline_mean) / baseline_mean) * 100
            deviations.append({
                "index": int(idx),
                "timestamp": metadata.get("timestamps", [])[idx] if metadata.get("timestamps") else None,
                "value": float(value),
                "baseline_mean": float(baseline_mean),
                "deviation_pct": float(deviation_pct),
                "anomaly_score": float(anomaly_scores[idx])
            })

        # Build calculation trace
        calculation_trace = [
            f"Input: {len(emissions_data)} data points",
            f"Method: Isolation Forest anomaly detection",
            f"Baseline Mean: {baseline_mean:.2f}",
            f"Baseline Std Dev: {baseline_std:.2f}",
            f"Anomalies Detected: {len(anomaly_indices)}",
            f"Anomaly Indices: {anomaly_indices.tolist()}"
        ]

        outputs = {
            "anomalies_detected": len(anomaly_indices),
            "anomaly_details": deviations,
            "baseline_statistics": {
                "mean": float(baseline_mean),
                "std_dev": float(baseline_std),
                "min": float(np.min(emissions_data)),
                "max": float(np.max(emissions_data))
            },
            "methodology": "Isolation Forest",
            "detection_date": datetime.utcnow().isoformat() + "Z"
        }

        # Capture audit trail for deterministic calculation
        self._capture_calculation_audit(
            operation="detect_anomalies",
            inputs=inputs,
            outputs=outputs,
            calculation_trace=calculation_trace
        )

        return outputs

    async def explain(
        self,
        calculation_result: Dict[str, Any],
        context: Dict[str, Any],
        session,  # ChatSession instance
        rag_engine,  # RAGEngine instance
        temperature: float = 0.6
    ) -> str:
        """
        Generate AI-powered root cause hypothesis.

        Uses RAG for historical context, ChatSession for narrative.
        """
        # Step 1: Extract anomaly details
        anomalies = calculation_result["anomaly_details"]
        baseline = calculation_result["baseline_statistics"]

        # Step 2: RAG retrieval for historical context
        rag_result = await rag_engine.query(
            query=f"""
            Historical anomalies in {context.get('metric', 'emissions')}
            for site {context.get('site_id', 'unknown')}
            """,
            collections=[
                "historical_data",
                "maintenance_logs",
                "operational_events"
            ],
            top_k=5
        )

        formatted_context = self._format_rag_results(rag_result)

        # Step 3: AI investigation
        response = await session.chat(
            messages=[
                {
                    "role": "system",
                    "content": """You are a facility operations expert. Analyze anomalies
                    and provide root cause hypotheses based on historical patterns and
                    maintenance records."""
                },
                {
                    "role": "user",
                    "content": f"""
                    Investigate these anomalies:

                    **Detection Summary:**
                    - Anomalies Detected: {calculation_result['anomalies_detected']}
                    - Baseline Mean: {baseline['mean']:.2f}
                    - Baseline Std Dev: {baseline['std_dev']:.2f}

                    **Anomaly Details:**
                    {json.dumps(anomalies, indent=2)}

                    **Site Context:**
                    - Site ID: {context.get('site_id')}
                    - Metric: {context.get('metric')}
                    - Period: {context.get('period')}

                    **Historical Context:**
                    {formatted_context}

                    Provide:
                    1. Root cause hypothesis for each anomaly
                    2. Pattern analysis (one-time spike vs sustained change)
                    3. Likelihood assessment (high/medium/low confidence)
                    4. Recommended investigation steps
                    5. Potential corrective actions

                    Be specific and reference historical patterns where relevant.
                    """
                }
            ],
            temperature=temperature,  # Allow some reasoning, but stay consistent
            tool_choice="none"  # No tools needed for narrative generation
        )

        return response.text

    def _format_rag_results(self, rag_result) -> str:
        """Format RAG results for LLM context."""
        if not rag_result or not rag_result.chunks:
            return "No historical context available."

        formatted = []
        for i, chunk in enumerate(rag_result.chunks, 1):
            formatted.append(f"{i}. {chunk.text}")

        return "\n\n".join(formatted)
```

### Key Principles

1. **Separation of Concerns**: Numbers from `calculate()`, insights from `explain()`
2. **Deterministic First**: Always run calculations first
3. **AI for Why, Not What**: LLM explains WHY anomaly occurred, not WHAT the value is
4. **Temperature ≤ 0.7**: More consistent than recommendation agents
5. **Optional RAG**: Use for historical context, not always required

### Testing Pattern

```python
import pytest


def test_anomaly_investigator_calculate_deterministic():
    """Verify anomaly detection is deterministic."""
    investigator = AnomalyInvestigator(anomaly_detector)

    inputs = {
        "emissions_timeseries": np.array([100, 105, 98, 102, 200, 99, 103])
    }

    # Run twice
    result1 = investigator.calculate(inputs)
    result2 = investigator.calculate(inputs)

    # Anomaly detection must be identical
    assert result1["anomalies_detected"] == result2["anomalies_detected"]
    assert result1["anomaly_details"] == result2["anomaly_details"]


@pytest.mark.asyncio
async def test_anomaly_investigator_explain_uses_rag():
    """Verify AI explanation uses RAG for context."""
    investigator = AnomalyInvestigator(anomaly_detector)
    mock_session = Mock()
    mock_rag = AsyncMock()

    mock_rag.query.return_value = Mock(chunks=[...])
    mock_session.chat = AsyncMock(return_value=Mock(
        text="Root cause hypothesis..."
    ))

    calculation_result = {
        "anomalies_detected": 1,
        "anomaly_details": [...],
        "baseline_statistics": {"mean": 100}
    }

    explanation = await investigator.explain(
        calculation_result, {}, mock_session, mock_rag
    )

    # Verify RAG was used
    assert mock_rag.query.called
    assert "historical_data" in mock_rag.query.call_args[1]["collections"]

    # Verify explanation was generated
    assert len(explanation) > 0


@pytest.mark.asyncio
async def test_anomaly_investigator_hybrid_workflow():
    """Test full hybrid workflow: calculate + explain."""
    investigator = AnomalyInvestigator(anomaly_detector)
    mock_session = Mock()
    mock_rag = AsyncMock()

    # Step 1: Deterministic calculation
    inputs = {"emissions_timeseries": np.array([...])}
    calculation = investigator.calculate(inputs)

    assert calculation["anomalies_detected"] > 0

    # Step 2: AI explanation
    mock_rag.query.return_value = Mock(chunks=[])
    mock_session.chat = AsyncMock(return_value=Mock(text="Explanation..."))

    explanation = await investigator.explain(calculation, {}, mock_session, mock_rag)

    assert explanation is not None
```

---

## Pattern Selection Decision Tree

```
START: "I need to create an agent"
    │
    ├─→ Is this for regulatory/compliance calculations?
    │   ├─→ YES → Use PATTERN 1: Deterministic Agent
    │   │         - Zero AI
    │   │         - Full audit trail
    │   │         - 100% reproducible
    │   │
    │   └─→ NO (continue)
    │
    ├─→ Does this involve AI-driven recommendations or strategic planning?
    │   ├─→ YES → Use PATTERN 2: Reasoning Agent
    │   │         - Full AI with RAG
    │   │         - Multi-tool orchestration
    │   │         - Creative problem-solving
    │   │
    │   └─→ NO (continue)
    │
    ├─→ Does this analyze data and provide insights?
    │   ├─→ YES → Use PATTERN 3: Insight Agent
    │   │         - Deterministic calculations
    │   │         - AI-generated explanations
    │   │         - Hybrid architecture
    │   │
    │   └─→ NO → This may be UTILITY code (base classes, testing)
```

### Quick Reference Table

| Characteristic | Pattern 1 (Deterministic) | Pattern 2 (Reasoning) | Pattern 3 (Insight) |
|----------------|---------------------------|----------------------|---------------------|
| **Uses ChatSession** | ❌ NO | ✅ YES | ✅ YES |
| **Uses RAG** | ❌ NO | ✅ YES | ✅ Optional |
| **Uses Tools** | ❌ NO | ✅ YES | ❌ Usually NO |
| **Temperature** | N/A | 0.5-0.8 | 0.5-0.7 |
| **Deterministic** | ✅ 100% | ❌ NO | ✅ Calculations only |
| **Audit Trail** | ✅ Required | ❌ NO | ✅ For calculations |
| **Compliance-Critical** | ✅ YES | ❌ NO | ❌ NO |
| **Example Use Cases** | Emissions calculations | Technology recommendations | Anomaly investigation |

---

## Migration Guide

### Migrating from Old Pattern to New Patterns

#### Scenario 1: Remove AI from Critical Path Agent

**BEFORE (WRONG):**
```python
# carbon_agent_ai.py - Uses ChatSession for calculations (BAD!)
emissions = calculate_emissions(data)  # Deterministic
summary = await session.chat(f"Summarize: {emissions}", temperature=0.0)
```

**AFTER (CORRECT):**
```python
# carbon_agent.py - Pure deterministic (GOOD!)
class CarbonCalculator(DeterministicAgent):
    def execute(self, inputs):
        emissions = self._calculate_emissions(inputs)
        return {"emissions_tco2e": emissions, "trace": [...]}
```

**Migration Steps:**
1. Remove ChatSession entirely
2. Inherit from `DeterministicAgent`
3. Add audit trail
4. Remove "_ai" suffix from filename
5. Update all callers to use new version

---

#### Scenario 2: Add AI to Recommendation Agent

**BEFORE (WRONG):**
```python
# recommendation_agent.py - Static database lookup
recommendation = db.query(building_type=input.type).first()
return recommendation
```

**AFTER (CORRECT):**
```python
# recommendation_agent_ai.py - Full AI reasoning
class RecommendationAgent(ReasoningAgent):
    async def reason(self, context, session, rag_engine, tools):
        # 1. RAG retrieval
        knowledge = await self._rag_retrieve(...)

        # 2. AI reasoning with tools
        response = await session.chat(
            messages=[...],
            tools=[tech_db, financial_analysis, spatial_check],
            temperature=0.7
        )

        return parse_recommendation(response)
```

**Migration Steps:**
1. Create new file with "_ai" suffix
2. Inherit from `ReasoningAgent`
3. Add RAG retrieval
4. Define tools
5. Implement multi-tool orchestration
6. Test with real scenarios

---

#### Scenario 3: Split Calculation and Insight

**BEFORE (MIXED):**
```python
# benchmark_agent.py - Mixed concerns
peer_avg = db.query(industry).mean()
ratio = emissions / peer_avg
insight = f"You are {'above' if ratio > 1 else 'below'} average"  # Static text
```

**AFTER (SPLIT):**
```python
# benchmark_agent.py - Split into hybrid
class BenchmarkAgent(InsightAgent):
    def calculate(self, inputs):
        # Deterministic comparison
        peer_avg = self.db.query(inputs["industry"]).mean()
        ratio = inputs["emissions"] / peer_avg
        return {"ratio": ratio, "peer_avg": peer_avg}

    async def explain(self, calculation_result, context, session, rag_engine):
        # AI-generated insight
        best_practices = await rag_engine.query("best in class for ...")

        response = await session.chat(
            messages=[{
                "role": "user",
                "content": f"""
                Performance ratio: {calculation_result['ratio']}
                Best practices: {best_practices}

                Provide competitive analysis and improvement roadmap.
                """
            }],
            temperature=0.6
        )

        return response.text
```

**Migration Steps:**
1. Split into `calculate()` and `explain()`
2. Inherit from `InsightAgent`
3. Add RAG to `explain()`
4. Keep calculations deterministic
5. Use AI only for narratives

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: AI in Critical Path ❌

```python
# WRONG - LLM for regulatory calculation
emissions = await session.chat(
    f"Calculate emissions for {consumption} kWh of {fuel}",
    temperature=0.0
)
```

**Why Wrong:**
- LLMs can hallucinate numbers
- Not reproducible
- Can't be audited
- Fails regulatory compliance

**Fix:** Use deterministic calculation, remove LLM entirely.

---

### Anti-Pattern 2: Static Lookups for Recommendations ❌

```python
# WRONG - Static database for recommendations
recommendation = db.query(building_type="office").first()
```

**Why Wrong:**
- No contextual reasoning
- Ignores facility specifics
- Can't adapt to constraints
- Misses better solutions

**Fix:** Use AI reasoning with RAG and tools.

---

### Anti-Pattern 3: Recalculating in Explain ❌

```python
# WRONG - Recalculating in explain method
async def explain(self, calculation_result, context, session, rag_engine):
    # DON'T DO THIS - recalculating!
    emissions = self._calculate_emissions(context)  # Already done!

    response = await session.chat(f"Explain {emissions}")
```

**Why Wrong:**
- Wastes computation
- Can produce different numbers
- Breaks separation of concerns

**Fix:** Use `calculation_result` parameter, don't recalculate.

---

### Anti-Pattern 4: No Tool Orchestration ❌

```python
# WRONG - Single LLM call, no tools
response = await session.chat(
    "Recommend technology with costs",
    temperature=0.7
)
# LLM will hallucinate costs!
```

**Why Wrong:**
- LLM makes up numbers
- No validation of feasibility
- No deterministic checks

**Fix:** Use tools for calculations, LLM only for reasoning.

---

### Anti-Pattern 5: High Temperature for Critical Insights ❌

```python
# WRONG - Temperature too high for consistent insights
async def explain(...):
    response = await session.chat(..., temperature=0.9)  # TOO HIGH
```

**Why Wrong:**
- Insights vary wildly between runs
- Users get inconsistent advice
- Hard to reproduce results

**Fix:** Use temperature ≤ 0.7 for insight agents.

---

## Summary

### Three Patterns, One Goal

**Goal:** Leverage AI intelligence while maintaining regulatory compliance.

**How:**
1. **Pattern 1**: Pure deterministic for compliance-critical calculations
2. **Pattern 2**: Full AI reasoning for strategic recommendations
3. **Pattern 3**: Hybrid (deterministic + AI) for analysis and insights

**Result:** Best of both worlds - AI intelligence + regulatory compliance.

---

**Version History:**
- 1.0 (2025-11-06): Initial patterns guide
- Author: GreenLang Architecture Team
- Next Review: After Phase 2 transformations complete
