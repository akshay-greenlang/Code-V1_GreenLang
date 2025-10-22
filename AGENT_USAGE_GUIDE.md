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

**Document Status:** AUTHORITATIVE GUIDE v1.0.0
**Last Updated:** October 25, 2025
**Next Review:** Q1 2026 (before v2.0 unified interface release)

---

**END OF GUIDE**
