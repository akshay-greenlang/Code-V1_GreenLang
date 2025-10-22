# GreenLang Agent Usage Guide
## Calculator vs Assistant Agents - Clear Distinction

**Document Version:** 1.0.0
**Date:** October 25, 2025
**Author:** Head of AI & Climate Intelligence
**Status:** AUTHORITATIVE GUIDE - Required Reading for All Users

---

## EXECUTIVE SUMMARY

GreenLang employs a **DUAL-TIER ARCHITECTURE** with two types of agents:

1. **Calculator Agents** - Fast, deterministic, API-first computation engines
2. **Assistant Agents** - Conversational, AI-powered, explanation-rich interfaces

**Both are necessary. Both serve different purposes. Neither is redundant.**

---

## THE CONFUSION PROBLEM

### What Users See (Confusing)

```
Available Agents:
- FuelAgent
- FuelAgentAI
- CarbonAgent
- CarbonAgentAI
- GridFactorAgent
- GridFactorAgentAI
- RecommendationAgent
- RecommendationAgentAI
- ReportAgent
- ReportAgentAI
```

**User Question:** "Which one do I use? They seem to do the same thing!"

**Answer:** They work TOGETHER, not separately. Let me explain...

---

## ARCHITECTURE: PARENT-CHILD RELATIONSHIP

### The Truth: It's a Layered Design

```
┌─────────────────────────────────────────────────────┐
│  USER REQUEST                                        │
│  "What are my Q3 fuel emissions from natural gas?"  │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  FuelAgentAI (ASSISTANT - AI Orchestrator)          │
│  - Understands natural language                     │
│  - Uses ChatSession (Claude/GPT-4)                  │
│  - Generates explanations                           │
│  - Provides recommendations                         │
│  - Cost: $0.001-0.01 per query                      │
│  - Speed: 2-5 seconds                               │
└───────────────────┬─────────────────────────────────┘
                    │
                    │ CALLS AS A TOOL
                    ▼
┌─────────────────────────────────────────────────────┐
│  FuelAgent (CALCULATOR - Computation Engine)        │
│  - Pure deterministic math                          │
│  - emissions = amount × emission_factor             │
│  - Database lookups only                            │
│  - Zero hallucination guarantee                     │
│  - Cost: FREE (no LLM)                              │
│  - Speed: <10ms                                     │
└─────────────────────────────────────────────────────┘
```

**Key Insight:** FuelAgentAI is a WRAPPER around FuelAgent, not a duplicate.

---

## WHEN TO USE CALCULATOR AGENTS

### Use Calculator Agents For:

✅ **API Integrations**
```python
# Direct API call for systems integration
from greenlang.agents import FuelAgent

agent = FuelAgent()
result = agent.run({
    "fuel_type": "natural_gas",
    "amount": 1000,
    "unit": "therms"
})

# Result: {"emissions_kg_co2e": 5306.0, "calculation_time_ms": 3}
# Fast, exact, no LLM cost
```

✅ **Batch Processing**
```python
# Process 10,000 records in seconds
results = [agent.run(record) for record in shipment_data]
# Total time: ~30 seconds (3ms × 10,000)
# Total cost: $0 (no LLM)
```

✅ **Enterprise ERP Integration**
```python
# SAP/Oracle integration - needs speed and reliability
sap_connector.register_calculator(FuelAgent())
# Processes 100,000 transactions/day
# Zero LLM costs = $0/month
```

✅ **Microservices Architecture**
```python
# Calculator as a microservice
@app.post("/api/v1/calculate/fuel")
def calculate_fuel(request: FuelInput):
    return FuelAgent().run(request.dict())
# <10ms response time, no LLM dependency
```

### Calculator Agent Characteristics

| Feature | Calculator Agent |
|---------|------------------|
| **Input** | Structured JSON/dict |
| **Output** | Exact numbers only |
| **Speed** | <10ms per calculation |
| **Cost** | $0 (no LLM) |
| **Use Case** | API, batch, integration |
| **User Type** | Developers, systems |
| **Hallucination Risk** | ZERO (pure math) |
| **Explanation** | None (numbers only) |

---

## WHEN TO USE ASSISTANT AGENTS

### Use Assistant Agents For:

✅ **Natural Language Queries**
```python
# Business user asking a question
from greenlang.agents import FuelAgentAI

agent = FuelAgentAI()
result = agent.run({
    "query": "What are my Q3 emissions from natural gas? "
             "How do they compare to last year?"
})

# Result includes:
# - Exact emissions calculation (via FuelAgent)
# - Year-over-year comparison
# - Natural language explanation
# - Recommendations for reduction
# - Cost: ~$0.003, Time: ~3 seconds
```

✅ **Interactive Dashboards**
```python
# Sustainability dashboard with chat interface
user_question = "Show me fuel emissions by facility"

response = FuelAgentAI().run({"query": user_question})
# Returns: data + insights + visualizations + explanations
```

✅ **Executive Reporting**
```python
# Generate narrative for board presentation
report = ReportAgentAI().run({
    "query": "Create an executive summary of our Q3 carbon footprint "
             "with year-over-year trends and key recommendations"
})

# Result: Multi-page report with:
# - Executive summary (AI-generated narrative)
# - Exact calculations (from Calculator agents)
# - Visualizations
# - Recommendations prioritized by ROI
```

✅ **Conversational AI / Chatbots**
```python
# ChatGPT-style interface for sustainability team
chatbot.register_agent(FuelAgentAI())

# User: "What's our biggest source of emissions?"
# Assistant: [Analyzes data using Calculator + AI reasoning]
# Returns comprehensive answer with context
```

### Assistant Agent Characteristics

| Feature | Assistant Agent |
|---------|-----------------|
| **Input** | Natural language questions |
| **Output** | Numbers + explanations + recommendations |
| **Speed** | 2-5 seconds (LLM processing) |
| **Cost** | $0.001-0.01 per query |
| **Use Case** | Dashboards, chat, reports |
| **User Type** | Business users, executives |
| **Hallucination Risk** | ZERO for numbers (uses Calculator tools) |
| **Explanation** | Rich AI-generated insights |

---

## DECISION TREE: WHICH AGENT TO USE?

```
START: I need to calculate emissions
    │
    ▼
┌─────────────────────────────────────┐
│ Do you have structured input data? │
│ (JSON, CSV, database records)      │
└─────────────┬───────────────────────┘
              │
        ┌─────┴─────┐
        │           │
       YES         NO
        │           │
        ▼           ▼
┌───────────┐  ┌──────────────┐
│ Is speed  │  │ Do you need  │
│ critical? │  │ explanations?│
│ (<100ms)  │  │ or insights? │
└─────┬─────┘  └──────┬───────┘
      │                │
  ┌───┴───┐        ┌───┴───┐
 YES     NO       YES     NO
  │       │        │       │
  ▼       ▼        ▼       ▼
┌──────────────┐  ┌──────────────┐
│ CALCULATOR   │  │ ASSISTANT    │
│ AGENT        │  │ AGENT        │
│              │  │              │
│ FuelAgent    │  │ FuelAgentAI  │
│ CarbonAgent  │  │ CarbonAgentAI│
│ etc.         │  │ etc.         │
└──────────────┘  └──────────────┘
  Fast, Free       Smart, Insightful
```

### Quick Decision Guide

| Your Need | Use This |
|-----------|----------|
| "I have 1000 shipments to process" | **Calculator** (FuelAgent) |
| "What's causing our high emissions?" | **Assistant** (FuelAgentAI) |
| "API endpoint for ERP integration" | **Calculator** (CarbonAgent) |
| "Generate board report with insights" | **Assistant** (ReportAgentAI) |
| "Real-time dashboard calculations" | **Calculator** (GridFactorAgent) |
| "Answer user questions in chatbot" | **Assistant** (GridFactorAgentAI) |
| "Batch overnight processing" | **Calculator** (all) |
| "Interactive analysis with users" | **Assistant** (all) |

---

## COST COMPARISON

### Example: Processing 10,000 Fuel Records

| Approach | Time | Cost | Best For |
|----------|------|------|----------|
| **Calculator Only** | 30 seconds | $0 | Batch processing, APIs |
| **Assistant Only** | 8 hours | $30-100 | NOT RECOMMENDED for batch |
| **Hybrid** | 35 seconds | $0.10 | Best: Calculator for data, Assistant for summary |

### Recommended Hybrid Pattern

```python
# BEST PRACTICE: Use Calculator for bulk, Assistant for summary

# Step 1: Process all records with Calculator (fast, free)
calculator = FuelAgent()
results = [calculator.run(record) for record in records]
# Time: 30 seconds, Cost: $0

# Step 2: Summarize with Assistant (smart, insightful)
assistant = FuelAgentAI()
summary = assistant.run({
    "query": f"Analyze these {len(results)} calculations and "
             f"provide key insights and recommendations",
    "data": results
})
# Time: 5 seconds, Cost: $0.01

# TOTAL: 35 seconds, $0.01 (vs $100 using Assistant for all)
```

---

## REAL-WORLD EXAMPLES

### Example 1: Manufacturing Company Dashboard

**Requirement:** Real-time emissions dashboard for 50 facilities

**Solution:**
```python
# Backend API: Calculator Agents
@app.get("/api/emissions/{facility_id}")
def get_emissions(facility_id: str):
    # Fast calculation: <10ms response
    return FuelAgent().run(get_facility_data(facility_id))

# Frontend Chat: Assistant Agent
@app.post("/api/chat")
def chat(question: str):
    # Natural language interface: ~3s response
    return FuelAgentAI().run({"query": question})
```

**Result:**
- Dashboard loads instantly (<100ms per facility)
- Users can ask questions ("Why is Facility A higher?")
- Total cost: ~$5/month (chat queries only)

---

### Example 2: EU CBAM Compliance

**Requirement:** Process 100,000 shipments for CBAM reporting

**Solution:**
```python
# Use GL-CBAM-APP (all Calculator agents)
from cbam.agents import EmissionsCalculatorAgent

calculator = EmissionsCalculatorAgent()
results = calculator.process_batch(shipments)
# Time: <5 minutes for 100,000 shipments
# Cost: $0 (no LLM)

# Generate narrative report with Assistant
from greenlang.agents import ReportAgentAI

report = ReportAgentAI().run({
    "query": "Create EU CBAM compliance report with "
             "executive summary and risk analysis",
    "data": results
})
# Time: ~10 seconds, Cost: $0.05
```

**Result:**
- Fast processing (Calculator)
- Insightful reporting (Assistant)
- Total cost: $0.05 (vs $500 if using Assistant for all calculations)

---

### Example 3: Sustainability Chatbot

**Requirement:** ChatGPT-style interface for sustainability team

**Solution:**
```python
# User asks: "What are our top 3 emission sources?"

# Backend: Assistant orchestrates, Calculators compute
assistant = CarbonAgentAI()
response = assistant.run({
    "query": "Analyze all emission sources and identify top 3 contributors"
})

# How it works:
# 1. Assistant breaks down question
# 2. Calls FuelAgent (Calculator) for fuel emissions
# 3. Calls GridFactorAgent (Calculator) for electricity
# 4. Calls CarbonAgent (Calculator) for aggregation
# 5. Assistant ranks and explains results

# User gets: Numbers (exact) + Explanation (AI-generated)
```

---

## MIGRATION PATH: CURRENT STATE → FUTURE STATE

### Current State (Confusing)

```
Two separate agents with unclear relationship:
- FuelAgent (what is this?)
- FuelAgentAI (what is this?)
```

### Future State (Clear) - Coming in v2.0

```python
# UNIFIED AGENT - Auto-detects input type

from greenlang.agents import FuelAgent

agent = FuelAgent()

# Structured input → Fast Calculator path
result = agent.run({
    "fuel_type": "natural_gas",
    "amount": 1000,
    "unit": "therms"
})
# Time: <10ms, Cost: $0

# Natural language → Smart Assistant path
result = agent.run({
    "query": "What are my emissions from 1000 therms of natural gas?"
})
# Time: ~3s, Cost: $0.003

# SAME AGENT, AUTOMATIC PATH SELECTION
```

### Migration Timeline

**Q1 2026 (v1.5):**
- ✅ Add clear documentation (this guide)
- ✅ Update all examples
- ✅ Add deprecation warnings

**Q2 2026 (v2.0):**
- ✅ Implement unified interface
- ✅ Maintain backward compatibility
- ✅ Old names still work (with warnings)

**Q4 2026 (v3.0):**
- ✅ Remove old naming
- ✅ Single agent per function
- ✅ Clean, simple API

---

## FREQUENTLY ASKED QUESTIONS

### Q: Why have two agents if AI can do everything?

**A:** Because AI is expensive and slow for bulk calculations.

- Calculator: $0, <10ms per calculation
- Assistant: $0.01, ~3s per calculation

For 100,000 calculations:
- Calculator: $0, 16 minutes
- Assistant: $1,000, 83 hours

### Q: Does the Assistant Agent hallucinate numbers?

**A:** NO. The Assistant uses Calculator tools for ALL numbers.

```python
# What happens inside FuelAgentAI:
def run(self, query):
    # AI understands question
    # AI calls FuelAgent.run() as a tool
    # AI explains the exact result from FuelAgent
    # AI NEVER generates numbers itself
```

### Q: Can I use Calculator Agents directly in production?

**A:** YES! They're production-grade:
- Zero-hallucination guarantee
- <10ms response time
- Fully deterministic
- Complete audit trail

### Q: Will Calculator agents be deprecated?

**A:** NO. They'll be integrated into unified agents but the fast path remains.

### Q: Which agents are Calculator vs Assistant?

**Current GreenLang Platform:**

| Calculator Agent | Assistant Agent |
|------------------|-----------------|
| FuelAgent | FuelAgentAI |
| CarbonAgent | CarbonAgentAI |
| GridFactorAgent | GridFactorAgentAI |
| RecommendationAgent | RecommendationAgentAI |
| ReportAgent | ReportAgentAI |
| BoilerAgent | (no AI version yet) |
| IntensityAgent | (no AI version yet) |
| BenchmarkAgent | (no AI version yet) |

**GL-CSRD-APP (all Calculator except MaterialityAgent):**
- IntakeAgent, CalculatorAgent, AggregatorAgent, AuditAgent, ReportingAgent (Calculator)
- MaterialityAgent (AI Assistant - requires human review)

**GL-CBAM-APP (all Calculator):**
- ShipmentIntakeAgent, EmissionsCalculatorAgent, ReportingPackagerAgent

---

## BEST PRACTICES

### ✅ DO THIS

```python
# USE CALCULATOR for batch processing
results = [FuelAgent().run(r) for r in records]

# USE ASSISTANT for user questions
answer = FuelAgentAI().run({"query": user_question})

# USE HYBRID for optimal performance
data = FuelAgent().batch_process(records)  # Fast
summary = FuelAgentAI().summarize(data)    # Smart
```

### ❌ DON'T DO THIS

```python
# DON'T use Assistant for batch (slow & expensive)
results = [FuelAgentAI().run(r) for r in records]  # ❌ $1000+, 83 hours

# DON'T use Calculator for natural language (won't work)
result = FuelAgent().run("What are my emissions?")  # ❌ Error

# DON'T duplicate work
calc_result = FuelAgent().run(data)
ai_result = FuelAgentAI().run(data)  # ❌ FuelAgentAI already calls FuelAgent!
```

---

## SUMMARY: THE GOLDEN RULES

1. **Calculator Agents** = Fast, free, exact numbers for APIs/batch
2. **Assistant Agents** = Smart, insightful, conversational for humans
3. **Assistant calls Calculator** = No duplication, it's a wrapper
4. **Use Calculator when you can** = Save money and time
5. **Use Assistant when you need AI** = Get insights and explanations
6. **Hybrid approach is best** = Calculator for data, Assistant for summary

---

## SUPPORT & FEEDBACK

**Questions?** Contact: ai-support@greenlang.io

**Found this confusing?** We want feedback: feedback@greenlang.io

**Want training?** Request a workshop: training@greenlang.io

---

## INDUSTRIAL AI AGENTS: ORCHESTRATOR + TOOLS PATTERN

### The New Pattern (Agent #1, #2, #12)

The latest GreenLang agents use a **TOOL-FIRST ARCHITECTURE**:

```
┌────────────────────────────────────────────────────────┐
│  USER INPUT                                             │
│  {facility_data, fuel_consumption, electricity, ...}   │
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│  IndustrialProcessHeatAgent_AI                         │
│  AI ORCHESTRATOR                                        │
│  - Uses ChatSession (Claude/GPT-4)                     │
│  - temperature=0.0, seed=42 (deterministic)            │
│  - Budget controlled ($0.40 default)                   │
│  - Has 7 deterministic TOOLS                           │
│  - Speed: 10-30 seconds                                │
│  - Cost: $0.10-0.40 per analysis                       │
└───────────────────┬────────────────────────────────────┘
                    │
                    │ CALLS TOOLS (deterministic)
                    ▼
┌────────────────────────────────────────────────────────┐
│  DETERMINISTIC TOOLS (7 tools)                         │
│  1. calculate_process_heat_demand()                    │
│  2. calculate_temperature_requirements()               │
│  3. calculate_energy_intensity()                       │
│  4. estimate_solar_thermal_fraction()                  │
│  5. calculate_backup_fuel_requirements()               │
│  6. estimate_emissions_baseline()                      │
│  7. calculate_decarbonization_potential()              │
│                                                         │
│  All tools: Pure math, zero hallucination, <10ms each │
└────────────────────────────────────────────────────────┘
```

**Key Differences from Calculator/Assistant Pattern:**

| Feature | Old Pattern (FuelAgent) | New Pattern (Agent #1/2/12) |
|---------|-------------------------|----------------------------|
| **Structure** | Two separate agents | Single AI agent + tools |
| **Calculator** | Standalone agent | Built-in tools |
| **AI Usage** | Optional (AssistantAI) | Mandatory (orchestrator) |
| **Determinism** | Calculator: 100% | Tools: 100%, AI: temp=0.0 |
| **Cost** | Calculator: $0, AI: variable | $0.10-0.40 per analysis |
| **Speed** | Calculator: <10ms | 10-30 seconds (full analysis) |
| **Use Case** | Simple calculations | Complex multi-step analysis |

---

## AGENT #1: IndustrialProcessHeatAgent_AI

**Purpose:** Analyze industrial process heat requirements and decarbonization potential

**Input:** Facility data with fuel consumption, processes, temperatures
**Output:** Comprehensive analysis with heat demand, solar thermal sizing, emissions baseline, ROI

### Example Usage

```python
from greenlang.agents import IndustrialProcessHeatAgent_AI

# Initialize with budget
agent = IndustrialProcessHeatAgent_AI(budget_usd=0.40)

# Input data
facility_data = {
    "facility_id": "PLANT-001",
    "facility_name": "Food Processing Plant",
    "industry_type": "Food & Beverage",
    "processes": [
        {
            "process_name": "Pasteurization",
            "temperature_required_c": 72,
            "annual_hours": 6000,
            "thermal_load_mmbtu_hr": 5.0
        }
    ],
    "fuel_consumption": {
        "natural_gas": 50000  # MMBtu/year
    },
    "latitude": 35.0,
    "grid_region": "CAISO"
}

# Run analysis
result = agent.run(facility_data)

# Result includes:
# - Total heat demand (MMBtu/year)
# - Process temperature distribution
# - Solar thermal potential (% of load)
# - Backup fuel requirements
# - Emissions baseline (kg CO2e/year)
# - Decarbonization roadmap
# - Financial analysis (NPV, IRR, payback)
```

**When to Use:**
- Analyzing industrial heating needs
- Sizing solar thermal systems
- Calculating process-specific emissions
- Evaluating decarbonization options for manufacturing

**Cost:** ~$0.20-0.40 per facility analysis
**Speed:** 15-30 seconds

---

## AGENT #2: BoilerReplacementAgent_AI

**Purpose:** Analyze boiler replacement options with solar thermal, heat pumps, and hybrid systems

**Input:** Current boiler data, facility requirements, capital budget
**Output:** Technology comparison, hybrid system design, financial analysis, implementation plan

### Example Usage

```python
from greenlang.agents import BoilerReplacementAgent_AI

agent = BoilerReplacementAgent_AI(budget_usd=0.50)

boiler_data = {
    "facility_id": "PLANT-002",
    "current_boiler": {
        "fuel_type": "natural_gas",
        "capacity_mmbtu_hr": 10.0,
        "efficiency_percent": 80,
        "age_years": 25
    },
    "annual_fuel_consumption_mmbtu": 50000,
    "peak_demand_mmbtu_hr": 8.0,
    "required_temperature_f": 180,
    "facility_sqft": 50000,
    "latitude": 40.0,
    "capital_budget_usd": 500000
}

result = agent.run(boiler_data)

# Result includes:
# - Current boiler efficiency analysis
# - Solar thermal sizing and performance
# - Heat pump COP and capacity
# - Hybrid system configuration
# - Technology comparison matrix
# - Financial analysis (NPV, IRR, payback with IRA 30% ITC)
# - Implementation roadmap
```

**When to Use:**
- Replacing aging boilers
- Evaluating heat pump + solar hybrid systems
- Calculating IRA 2022 tax incentives (30% ITC)
- Designing decarbonized heating systems

**Cost:** ~$0.30-0.50 per analysis
**Speed:** 20-40 seconds

---

## AGENT #12: DecarbonizationRoadmapAgent_AI

**Purpose:** Generate comprehensive decarbonization roadmaps for industrial facilities

**Input:** Complete facility profile with fuel, electricity, capital budget, targets
**Output:** GHG inventory, technology assessment, 3-phase implementation plan, compliance analysis

### Example Usage

```python
from greenlang.agents import DecarbonizationRoadmapAgent_AI

agent = DecarbonizationRoadmapAgent_AI(budget_usd=2.0)

facility_profile = {
    "facility_id": "DEMO-001",
    "facility_name": "Sample Food Processing Plant",
    "industry_type": "Food & Beverage",
    "latitude": 35.0,
    "fuel_consumption": {
        "natural_gas": 50000,  # MMBtu/year
        "fuel_oil": 5000
    },
    "electricity_consumption_kwh": 15000000,
    "grid_region": "CAISO",
    "capital_budget_usd": 10000000,
    "target_year": 2030,
    "target_reduction_percent": 50,
    "risk_tolerance": "moderate",
    "facility_sqft": 100000
}

result = agent.run(facility_profile)

# Result includes:
# - GHG inventory (Scope 1, 2, 3)
# - Baseline emissions (kg CO2e/year)
# - Technology assessment (solar, heat pumps, WHR, efficiency)
# - 3-phase implementation plan
# - Financial analysis (NPV, IRR, LCOA, federal incentives)
# - Risk assessment
# - Compliance analysis (CBAM, CSRD, SEC Climate Rule)
# - Recommended pathway with prioritized actions
```

**When to Use:**
- Creating facility-wide decarbonization strategies
- SEC Climate Rule disclosures
- EU CBAM compliance planning
- CSRD reporting
- Capital planning for sustainability investments

**Cost:** ~$1.00-2.00 per comprehensive roadmap
**Speed:** 30-60 seconds

**CLI Usage:**
```bash
# Generate roadmap from JSON input
gl decarbonization --input facility.json --output roadmap.json

# Run demo with sample data
gl decarbonization demo

# Generate example template
gl decarbonization example
```

---

## INTEGRATION PATTERNS: COMBINING AGENTS

### Pattern 1: Sequential Analysis (Agent #1 → Agent #2)

**Use Case:** Deep dive on specific process after facility-wide analysis

```python
# Step 1: Analyze all processes with Agent #1
agent1 = IndustrialProcessHeatAgent_AI()
heat_analysis = agent1.run(facility_data)

# Step 2: Focus on boiler replacement with Agent #2
# Extract boiler-specific data from Agent #1 results
boiler_data = {
    "facility_id": facility_data["facility_id"],
    "current_boiler": facility_data["current_boiler"],
    "annual_fuel_consumption_mmbtu": heat_analysis["data"]["total_heat_demand_mmbtu_year"],
    "peak_demand_mmbtu_hr": heat_analysis["data"]["peak_demand_mmbtu_hr"],
    # ... other fields from Agent #1 output
}

agent2 = BoilerReplacementAgent_AI()
boiler_plan = agent2.run(boiler_data)

# Combined output: Process heat analysis + detailed boiler replacement plan
```

**Total Cost:** ~$0.60-0.90 per facility
**Total Time:** 40-70 seconds

---

### Pattern 2: Parallel Analysis (Agent #1 || Agent #2)

**Use Case:** Analyze multiple facilities simultaneously

```python
import asyncio

async def analyze_facility(facility_data):
    """Analyze single facility with both agents in parallel"""

    agent1 = IndustrialProcessHeatAgent_AI()
    agent2 = BoilerReplacementAgent_AI()

    # Run both analyses in parallel
    results = await asyncio.gather(
        agent1._run_async(facility_data),
        agent2._run_async(prepare_boiler_data(facility_data))
    )

    return {
        "heat_analysis": results[0],
        "boiler_analysis": results[1]
    }

# Analyze 10 facilities in parallel
facilities = [...]  # List of 10 facilities
results = await asyncio.gather(*[analyze_facility(f) for f in facilities])

# Total time: ~40 seconds (parallelized, not 400 seconds sequential)
# Total cost: ~$9 (10 facilities × $0.90)
```

---

### Pattern 3: Orchestration with Agent #12

**Use Case:** Comprehensive roadmap that coordinates multiple specialist agents

```python
# Agent #12 orchestrates Agent #1, Agent #2, and other agents internally

agent12 = DecarbonizationRoadmapAgent_AI(budget_usd=2.0)

# Agent #12's internal workflow (automatic):
# 1. Analyzes facility profile
# 2. Calls Agent #1 (process heat analysis)
# 3. Calls Agent #2 (boiler replacement options)
# 4. Calls other agents (electricity, efficiency, etc.)
# 5. Synthesizes all results into comprehensive roadmap

roadmap = agent12.run(facility_profile)

# Single call → comprehensive analysis
# Agent #12 handles all coordination internally
```

**When to Use Agent #12 vs Individual Agents:**

| Scenario | Use This |
|----------|----------|
| Need full decarbonization roadmap | **Agent #12** (comprehensive) |
| Only need process heat analysis | **Agent #1** (focused, cheaper) |
| Only evaluating boiler options | **Agent #2** (focused, faster) |
| Need heat + boiler + electricity + everything | **Agent #12** (orchestrates all) |

---

### Pattern 4: Batch + Summary (Hybrid)

**Use Case:** Process multiple facilities, then generate executive summary

```python
# Step 1: Analyze all facilities (parallel)
facilities = load_facility_data(50)  # 50 facilities

results = await asyncio.gather(*[
    DecarbonizationRoadmapAgent_AI().run(f)
    for f in facilities
])
# Time: ~60 seconds (parallel), Cost: ~$100 (50 × $2)

# Step 2: Generate portfolio summary with ReportAgentAI
from greenlang.agents import ReportAgentAI

summary = ReportAgentAI().run({
    "query": "Create executive summary of decarbonization roadmaps "
             "for 50 facilities with portfolio-level insights, "
             "total CAPEX, total emissions reduction, and prioritization",
    "data": results
})
# Time: ~10 seconds, Cost: ~$0.20

# Total: 70 seconds, $100.20 for 50 facilities + executive summary
```

---

## DECISION TREE: WHICH INDUSTRIAL AGENT?

```
START: I need industrial decarbonization analysis
    │
    ▼
┌─────────────────────────────────────────┐
│ What's your scope?                      │
└─────────────┬───────────────────────────┘
              │
        ┌─────┴─────────────┐
        │                   │
   Specific Focus      Comprehensive
        │                   │
        ▼                   ▼
┌────────────────┐   ┌──────────────────┐
│ What aspect?   │   │ Use AGENT #12    │
└────┬───────────┘   │ Decarbonization  │
     │               │ RoadmapAgent     │
     │               └──────────────────┘
     │               Full facility analysis
┌────┴────┐
│         │
Process   Boiler
Heat      Replacement
│         │
▼         ▼
┌────────────┐  ┌────────────┐
│ AGENT #1   │  │ AGENT #2   │
│ Process    │  │ Boiler     │
│ Heat       │  │ Replacement│
└────────────┘  └────────────┘
  Focused         Focused
  Cheaper         Faster
  $0.20-0.40      $0.30-0.50
```

### Quick Selection Guide

| Your Question | Use This Agent |
|---------------|----------------|
| "What's my total heat demand across all processes?" | **Agent #1** (IndustrialProcessHeatAgent_AI) |
| "Should I replace my boiler with heat pump + solar?" | **Agent #2** (BoilerReplacementAgent_AI) |
| "Create full decarbonization plan for my facility" | **Agent #12** (DecarbonizationRoadmapAgent_AI) |
| "How much solar thermal can I install?" | **Agent #1** (includes solar thermal sizing) |
| "What's my payback with IRA 30% ITC?" | **Agent #2** (includes IRA incentives) |
| "I need SEC Climate Rule disclosure data" | **Agent #12** (includes compliance analysis) |
| "Compare multiple boiler technologies" | **Agent #2** (technology comparison matrix) |
| "Analyze 10 processes for decarbonization" | **Agent #1** (multi-process analysis) |
| "3-phase implementation plan needed" | **Agent #12** (includes phased roadmap) |

---

## COST OPTIMIZATION STRATEGIES

### Strategy 1: Use Focused Agents When Possible

```python
# ❌ EXPENSIVE: Use Agent #12 for simple boiler question
agent12 = DecarbonizationRoadmapAgent_AI(budget_usd=2.0)
result = agent12.run(data)  # $2.00, 60 seconds

# ✅ CHEAPER: Use Agent #2 for boiler-specific question
agent2 = BoilerReplacementAgent_AI(budget_usd=0.50)
result = agent2.run(boiler_data)  # $0.50, 30 seconds

# SAVINGS: $1.50 per query, 50% faster
```

### Strategy 2: Batch Processing

```python
# ❌ EXPENSIVE: Sequential analysis
for facility in facilities:
    result = agent.run(facility)  # 50 facilities × 60s = 50 minutes

# ✅ CHEAPER: Parallel analysis
results = await asyncio.gather(*[agent.run(f) for f in facilities])
# 50 facilities in ~60 seconds (same time as 1)
```

### Strategy 3: Budget Control

```python
# Set strict budget limits
agent = IndustrialProcessHeatAgent_AI(budget_usd=0.20)  # Lower budget

try:
    result = agent.run(data)
except BudgetExceededError:
    # Fall back to simpler analysis
    result = simple_calculator.run(data)
```

---

## TESTING AND VALIDATION

### Unit Tests (Deterministic Tools)

```python
# All tools are deterministic → 100% reproducible tests

def test_solar_thermal_fraction():
    """Test solar thermal fraction calculation (Agent #1 tool)"""

    agent = IndustrialProcessHeatAgent_AI()

    # Same input → Same output (every time)
    result1 = agent._estimate_solar_thermal_fraction_impl(
        latitude=35.0,
        annual_heat_demand_mmbtu=50000,
        avg_process_temp_c=80
    )

    result2 = agent._estimate_solar_thermal_fraction_impl(
        latitude=35.0,
        annual_heat_demand_mmbtu=50000,
        avg_process_temp_c=80
    )

    assert result1 == result2  # ✅ Deterministic
    assert result1["solar_fraction_percent"] == 45.2  # ✅ Exact value
```

### Integration Tests (Agent #1 + Agent #2)

```python
def test_agent1_agent2_integration():
    """Test data flow from Agent #1 to Agent #2"""

    # Step 1: Agent #1 analysis
    agent1 = IndustrialProcessHeatAgent_AI()
    heat_result = agent1.run(facility_data)

    assert heat_result["success"]
    assert "total_heat_demand_mmbtu_year" in heat_result["data"]

    # Step 2: Extract data for Agent #2
    boiler_input = {
        "facility_id": facility_data["facility_id"],
        "annual_fuel_consumption_mmbtu": heat_result["data"]["total_heat_demand_mmbtu_year"],
        # ... other fields
    }

    # Step 3: Agent #2 analysis
    agent2 = BoilerReplacementAgent_AI()
    boiler_result = agent2.run(boiler_input)

    assert boiler_result["success"]
    assert boiler_result["data"]["federal_itc_percent"] == 30  # IRA 2022

    # Step 4: Validate consistency
    # Agent #2's fuel consumption should match Agent #1's heat demand
    assert abs(
        boiler_result["data"]["annual_fuel_consumption_mmbtu"] -
        heat_result["data"]["total_heat_demand_mmbtu_year"]
    ) < 0.01  # Within rounding error
```

### End-to-End Tests (Agent #12 Orchestration)

```python
def test_agent12_orchestration():
    """Test Agent #12 orchestrating Agent #1 and Agent #2"""

    agent12 = DecarbonizationRoadmapAgent_AI(budget_usd=2.0)

    result = agent12.run(facility_profile)

    # Verify comprehensive output
    assert result["success"]
    assert "baseline_emissions_kg_co2e" in result["data"]
    assert "total_reduction_potential_kg_co2e" in result["data"]
    assert "recommended_pathway" in result["data"]
    assert "npv_usd" in result["data"]
    assert "irr_percent" in result["data"]

    # Verify Agent #12 called sub-agents (check metadata)
    assert result["metadata"]["tools_called"] >= 5  # Multiple tools used
    assert result["metadata"]["deterministic"] == True
```

---

## PRODUCTION DEPLOYMENT CHECKLIST

### Pre-Deployment

- [ ] All agents tested with ≥80% code coverage
- [ ] Integration tests passing
- [ ] Budget limits configured
- [ ] Error handling validated
- [ ] Input validation working
- [ ] Output schema validated

### Agent #1 Deployment

- [ ] IndustrialProcessHeatAgent_AI tested with real facility data
- [ ] 7 tools validated (calculate_process_heat_demand, etc.)
- [ ] Coverage: 85.97% ✅ (exceeds 85% target)
- [ ] API endpoint configured: `/api/v1/analyze/process-heat`
- [ ] Budget: $0.40 default (configurable)

### Agent #2 Deployment

- [ ] BoilerReplacementAgent_AI tested with real boiler data
- [ ] 8 tools validated (calculate_boiler_efficiency, etc.)
- [ ] Coverage: 83.05% ✅ (meets 80% minimum)
- [ ] IRA 2022 30% ITC validated
- [ ] API endpoint configured: `/api/v1/analyze/boiler-replacement`
- [ ] Budget: $0.50 default (configurable)

### Agent #12 Deployment

- [ ] DecarbonizationRoadmapAgent_AI tested end-to-end
- [ ] CLI command working: `gl decarbonization`
- [ ] Sub-agent orchestration validated
- [ ] Coverage: 100% ✅
- [ ] API endpoint configured: `/api/v1/roadmap/decarbonization`
- [ ] Budget: $2.00 default (configurable)

### Monitoring

- [ ] Token usage tracking enabled
- [ ] Cost monitoring dashboards configured
- [ ] Error rate alerts set up
- [ ] Performance metrics tracked (response time, success rate)
- [ ] Budget alerts configured (warn at 80%, block at 100%)

---

## MIGRATION GUIDE: v1.0 → v2.0

### Current State (v1.0)

Three agent types:
1. Calculator Agents (FuelAgent, CarbonAgent, etc.) - Standalone
2. Assistant Agents (FuelAgentAI, CarbonAgentAI, etc.) - Wrappers
3. Industrial AI Agents (Agent #1, #2, #12) - Tool-first architecture

### Future State (v2.0) - Q2 2026

Unified interface for all agents:

```python
# FUTURE: All agents auto-detect input type

from greenlang.agents import FuelAgent, IndustrialProcessHeatAgent

# Simple calculation → Fast path (no LLM)
fuel_result = FuelAgent.calculate({
    "fuel_type": "natural_gas",
    "amount": 1000
})  # <10ms, $0

# Natural language → AI path
fuel_result = FuelAgent.analyze({
    "query": "What are my emissions from 1000 therms?"
})  # ~3s, $0.003

# Complex analysis → Always AI (like Agent #1)
heat_result = IndustrialProcessHeatAgent.analyze(facility_data)
# ~20s, $0.40
```

### Backward Compatibility

```python
# OLD CODE (v1.0) - Still works in v2.0
from greenlang.agents import FuelAgent, FuelAgentAI

calc = FuelAgent()  # Deprecated warning
ai = FuelAgentAI()  # Deprecated warning

# NEW CODE (v2.0) - Recommended
from greenlang.agents import FuelAgent

agent = FuelAgent()  # Unified interface
result = agent.calculate(...)  # Fast path
result = agent.analyze(...)    # AI path
```

---

## APPENDIX: COMPLETE AGENT INVENTORY

### Industrial AI Agents (Tool-First Architecture)

| Agent ID | Agent Name | Status | Tools | Coverage | Cost |
|----------|-----------|--------|-------|----------|------|
| #1 | IndustrialProcessHeatAgent_AI | 100% ✅ | 7 | 85.97% | $0.20-0.40 |
| #2 | BoilerReplacementAgent_AI | 97% ✅ | 8 | 83.05% | $0.30-0.50 |
| #12 | DecarbonizationRoadmapAgent_AI | 100% ✅ | Multiple | 100% | $1.00-2.00 |

### Classic AI-Enhanced Agents (Wrapper Pattern)

| Calculator Agent | Assistant Agent | Use Case |
|------------------|-----------------|----------|
| FuelAgent | FuelAgentAI | Fuel emissions calculation |
| CarbonAgent | CarbonAgentAI | Carbon footprint aggregation |
| GridFactorAgent | GridFactorAgentAI | Electricity grid emissions |
| RecommendationAgent | RecommendationAgentAI | Reduction recommendations |
| ReportAgent | ReportAgentAI | Reporting and narratives |

### Specialized Calculators (No AI Version)

| Agent | Domain | Use Case |
|-------|--------|----------|
| BoilerAgent | HVAC | Boiler efficiency calculation |
| IntensityAgent | Benchmarking | Emissions intensity metrics |
| BenchmarkAgent | Benchmarking | Industry benchmarking |

### CSRD Platform Agents (GL-CSRD-APP)

| Agent | Type | Use Case |
|-------|------|----------|
| IntakeAgent | Calculator | Data ingestion |
| CalculatorAgent | Calculator | CSRD metrics calculation |
| AggregatorAgent | Calculator | Data aggregation |
| AuditAgent | Calculator | Audit trail generation |
| ReportingAgent | Calculator | CSRD report generation |
| MaterialityAgent | AI Assistant | Double materiality assessment |

### CBAM Platform Agents (GL-CBAM-APP)

| Agent | Type | Use Case |
|-------|------|----------|
| ShipmentIntakeAgent | Calculator | Import data ingestion |
| EmissionsCalculatorAgent | Calculator | Embedded emissions calculation |
| ReportingPackagerAgent | Calculator | CBAM XML report generation |

---

**Document Status:** AUTHORITATIVE GUIDE v1.1.0 (Updated with Industrial AI Agents)
**Last Updated:** December 2025
**Next Review:** Q1 2026 (before v2.0 unified interface release)

---

**END OF GUIDE**
