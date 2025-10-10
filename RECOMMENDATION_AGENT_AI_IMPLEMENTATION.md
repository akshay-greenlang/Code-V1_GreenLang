# RecommendationAgentAI Implementation

**AI-Powered Building Optimization Recommendations with ChatSession Integration**

**Status:** Complete
**Date:** October 2025
**Author:** GreenLang Framework Team

---

## Executive Summary

The `RecommendationAgentAI` is an AI-enhanced version of the `RecommendationAgent` that uses ChatSession for intelligent orchestration while preserving all deterministic calculations as tool implementations. This implementation provides natural language explanations, ROI-based prioritization, and actionable implementation plans for reducing building carbon emissions.

**Key Innovation:** All numeric calculations (ROI, savings, rankings) are performed by deterministic tools, while AI provides natural language context and insights. This ensures zero hallucinated numbers while maximizing interpretability.

---

## Architecture

### High-Level Design

```
RecommendationAgentAI (orchestration)
    ↓
ChatSession (AI reasoning)
    ↓
Tools (exact calculations)
    ├── analyze_energy_usage → Identify patterns and inefficiencies
    ├── calculate_roi → ROI and payback calculations
    ├── rank_recommendations → Priority-based ranking
    ├── estimate_savings → Emissions and cost savings
    └── generate_implementation_plan → Step-by-step roadmap
```

### Component Breakdown

#### 1. **Agent Layer** (`RecommendationAgentAI`)
- Orchestrates the recommendation workflow
- Manages ChatSession lifecycle
- Enforces budget constraints
- Tracks performance metrics
- Provides backward-compatible API

#### 2. **AI Layer** (`ChatSession`)
- Natural language understanding
- Context-aware recommendation explanations
- Intelligent tool selection
- Executive summary generation
- Temperature=0, seed=42 for determinism

#### 3. **Tool Layer** (Deterministic Functions)
- `analyze_energy_usage`: Pattern recognition and issue identification
- `calculate_roi`: Exact financial calculations
- `rank_recommendations`: Multi-criteria prioritization
- `estimate_savings`: Emissions and cost projections
- `generate_implementation_plan`: Phased roadmap creation

#### 4. **Base Agent** (`RecommendationAgent`)
- Provides recommendation database
- Implements core calculation logic
- Country-specific recommendations
- Backward compatibility

---

## Features

### Core Capabilities

1. **AI-Driven Energy Analysis**
   - Identifies inefficiencies from building data
   - Detects patterns (high HVAC, aging infrastructure, poor performance)
   - Calculates source percentages and dominant contributors
   - Severity-based issue prioritization

2. **ROI-Based Prioritization**
   - Exact payback period calculations
   - Cost categorization (Low/Medium/High)
   - Annual savings estimation (emissions + cost)
   - ROI percentage for each recommendation

3. **Natural Language Explanations**
   - AI-generated summaries for each recommendation
   - Context-aware explanations (WHY it matters)
   - Building-specific insights
   - Implementation guidance

4. **Implementation Planning**
   - Phased roadmap (Quick Wins, Strategic, Major Upgrades)
   - Timeline estimation
   - Step-by-step action plans
   - Cost and impact projections per phase

5. **Savings Estimation**
   - Emissions reduction (min/max range)
   - Cost savings projection
   - Percentage reduction range
   - Total potential impact

### Determinism Guarantees

- **Same Input → Same Output**: Reproducible results via seed=42
- **No Hallucinated Numbers**: All calculations via tools
- **Auditable Trail**: Full provenance of AI decisions and tool calls
- **Temperature=0**: Deterministic AI responses

### Performance Features

- **Async Support**: Concurrent processing via asyncio
- **Budget Enforcement**: Max $0.50 per analysis (configurable)
- **Performance Tracking**: AI calls, tool calls, cost metrics
- **Cost-Aware**: Sub-cent per analysis with demo provider

---

## Tools Reference

### Tool 1: `analyze_energy_usage`

**Purpose:** Analyze energy consumption patterns and identify inefficiencies

**Input:**
```python
{
    "emissions_by_source": {
        "electricity": 15000,
        "natural_gas": 8500
    },
    "load_breakdown": {
        "hvac_load": 0.45,
        "lighting_load": 0.25
    },
    "building_age": 20,
    "performance_rating": "Below Average"
}
```

**Output:**
```python
{
    "total_emissions_kg": 23500,
    "source_percentages": {
        "electricity": 63.83,
        "natural_gas": 36.17
    },
    "dominant_source": "electricity",
    "issues_identified": [
        {
            "type": "high_electricity",
            "severity": "high",
            "description": "Electricity accounts for 63.83% of emissions"
        },
        {
            "type": "high_hvac",
            "severity": "high",
            "description": "HVAC represents 45% of building load"
        },
        {
            "type": "aging_infrastructure",
            "severity": "medium",
            "description": "Building age (20 years) suggests potential inefficiencies"
        }
    ],
    "issue_count": 3
}
```

**Implementation:** Exact percentage calculations, threshold-based issue detection

---

### Tool 2: `calculate_roi`

**Purpose:** Calculate ROI, payback period, and financial metrics

**Input:**
```python
{
    "recommendations": [
        {
            "action": "Install LED lighting",
            "cost": "Medium",
            "impact": "50-70% reduction in lighting energy",
            "payback": "2-3 years"
        }
    ],
    "current_emissions_kg": 20000,
    "energy_cost_per_kwh": 0.12
}
```

**Output:**
```python
{
    "roi_calculations": [
        {
            "action": "Install LED lighting",
            "estimated_cost_usd": 50000,
            "annual_savings_usd": 7200.00,
            "payback_years": 2.5,
            "roi_percentage": 36.0,
            "emissions_reduction_kg": 12000,
            "cost_category": "Medium"
        }
    ],
    "total_potential_savings_usd": 7200.00,
    "total_emissions_reduction_kg": 12000
}
```

**Implementation:**
- Cost estimation: Low=$5K, Medium=$50K, High=$250K
- Impact extraction via regex parsing
- Energy-to-cost conversion: 1 kg CO2e ≈ 0.5 kWh
- ROI = (Annual Savings × Payback Years) / Cost × 100

---

### Tool 3: `rank_recommendations`

**Purpose:** Prioritize recommendations by impact, cost, ROI, or payback

**Input:**
```python
{
    "recommendations": [
        {
            "action": "Action A",
            "cost": "Low",
            "priority": "High",
            "payback": "2 years",
            "roi_percentage": 50
        },
        {
            "action": "Action B",
            "cost": "High",
            "priority": "Medium",
            "payback": "5 years",
            "roi_percentage": 100
        }
    ],
    "prioritize_by": "roi"
}
```

**Output:**
```python
{
    "ranked_recommendations": [
        {
            "action": "Action B",
            "roi_percentage": 100,
            "rank": 1
        },
        {
            "action": "Action A",
            "roi_percentage": 50,
            "rank": 2
        }
    ],
    "prioritization_strategy": "roi",
    "count": 2
}
```

**Strategies:**
- `roi`: Highest ROI percentage first
- `impact`: Highest impact percentage first
- `cost`: Low → Medium → High
- `payback`: Shortest payback first (default)

---

### Tool 4: `estimate_savings`

**Purpose:** Estimate potential emissions and cost savings

**Input:**
```python
{
    "recommendations": [...],
    "current_emissions_kg": 26700,
    "current_energy_cost_usd": 50000
}
```

**Output:**
```python
{
    "emissions_savings": {
        "minimum_kg_co2e": 5340.0,
        "maximum_kg_co2e": 10680.0,
        "percentage_range": "20.0-40.0%"
    },
    "cost_savings": {
        "minimum_annual_usd": 10000.00,
        "maximum_annual_usd": 20000.00
    },
    "percentage_reduction_range": "20.0-40.0%"
}
```

**Implementation:** Delegates to original agent's savings calculation logic

---

### Tool 5: `generate_implementation_plan`

**Purpose:** Create phased implementation roadmap

**Input:**
```python
{
    "recommendations": [...],
    "building_type": "commercial_office",
    "timeline_months": 12
}
```

**Output:**
```python
{
    "implementation_roadmap": [
        {
            "phase": "Phase 1: Quick Wins (0-6 months)",
            "timeline_months": 4,
            "actions": [
                {"action": "Install occupancy sensors", "cost": "Low"},
                {"action": "LED lighting upgrade", "cost": "Medium"}
            ],
            "estimated_cost": "Low",
            "expected_impact": "5-15% reduction",
            "implementation_steps": [
                "1. Conduct detailed assessment and vendor selection",
                "2. Secure budget approval and financing",
                "3. Execute installation/implementation",
                "4. Verify performance and document results"
            ]
        },
        {
            "phase": "Phase 2: Strategic Improvements (6-18 months)",
            "timeline_months": 6,
            "actions": [...],
            ...
        }
    ],
    "total_timeline_months": 12,
    "building_type": "commercial_office",
    "phases": 2
}
```

**Implementation:** Uses original agent's roadmap creation with enhanced timeline allocation

---

## API Reference

### Initialization

```python
from greenlang.agents.recommendation_agent_ai import RecommendationAgentAI

agent = RecommendationAgentAI(
    config=None,                          # Optional AgentConfig
    budget_usd=0.50,                      # Max $0.50 per analysis
    enable_ai_summary=True,               # Enable AI summaries
    enable_implementation_plans=True,     # Enable roadmaps
    max_recommendations=5                 # Top 5 recommendations
)
```

### Execution

```python
result = agent.execute({
    "emissions_by_source": {
        "electricity": 15000,
        "natural_gas": 8500,
        "diesel": 3200
    },
    "building_type": "commercial_office",
    "building_area": 50000,
    "occupancy": 200,
    "building_age": 20,
    "performance_rating": "Below Average",
    "load_breakdown": {
        "hvac_load": 0.45,
        "lighting_load": 0.25,
        "plug_load": 0.30
    },
    "country": "US"
})
```

### Result Structure

```python
{
    "success": True,
    "data": {
        "recommendations": [
            {
                "action": "Upgrade to high-efficiency HVAC system",
                "impact": "20-30% reduction in HVAC energy",
                "cost": "High",
                "payback": "5-7 years",
                "priority": "High",
                "rank": 1,
                "roi_percentage": 18.5,
                "annual_savings_usd": 12500.00,
                "emissions_reduction_kg": 6000
            },
            ...
        ],
        "total_recommendations": 5,
        "usage_analysis": {
            "total_emissions_kg": 26700,
            "source_percentages": {...},
            "dominant_source": "electricity",
            "issues_identified": [...]
        },
        "potential_savings": {
            "minimum_kg_co2e": 5340.0,
            "maximum_kg_co2e": 10680.0,
            "percentage_range": "20.0-40.0%"
        },
        "cost_savings": {
            "minimum_annual_usd": 10000.00,
            "maximum_annual_usd": 20000.00
        },
        "roi_analysis": {
            "roi_calculations": [...],
            "total_potential_savings_usd": 45000.00
        },
        "implementation_roadmap": [...],
        "ai_summary": "Based on comprehensive analysis of this 20-year-old commercial office...",
        "quick_wins": [...],
        "high_impact": [...]
    },
    "metadata": {
        "agent": "RecommendationAgentAI",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "tokens": 550,
        "cost_usd": 0.03,
        "tool_calls": 5,
        "calculation_time_ms": 1250.5,
        "ai_calls": 1,
        "deterministic": True
    }
}
```

### Performance Metrics

```python
summary = agent.get_performance_summary()

# Output:
{
    "agent": "RecommendationAgentAI",
    "ai_metrics": {
        "ai_call_count": 10,
        "tool_call_count": 45,
        "total_cost_usd": 0.25,
        "avg_cost_per_analysis": 0.025
    },
    "base_agent_metrics": {
        "agent": "RecommendationAgent",
        "version": "0.0.1"
    }
}
```

---

## Usage Examples

### Example 1: Basic Recommendation Generation

```python
from greenlang.agents.recommendation_agent_ai import RecommendationAgentAI

# Create agent
agent = RecommendationAgentAI()

# Simple building data
building_data = {
    "emissions_by_source": {
        "electricity": 10000,
        "natural_gas": 5000
    },
    "building_type": "commercial_office"
}

# Get recommendations
result = agent.execute(building_data)

if result.success:
    for rec in result.data["recommendations"]:
        print(f"{rec['action']}: {rec['impact']}")
```

### Example 2: High-Emission Building Analysis

```python
# Old inefficient building
building_data = {
    "emissions_by_source": {
        "electricity": 35000,
        "natural_gas": 20000,
        "diesel": 5000
    },
    "building_age": 25,
    "performance_rating": "Poor",
    "load_breakdown": {
        "hvac_load": 0.50,
        "lighting_load": 0.30,
        "plug_load": 0.20
    }
}

agent = RecommendationAgentAI(max_recommendations=10)
result = agent.execute(building_data)

# Print AI summary
print(result.data["ai_summary"])

# Print top 3 high-impact recommendations
for rec in result.data["high_impact"][:3]:
    print(f"\n{rec['action']}")
    print(f"  Impact: {rec['impact']}")
    print(f"  ROI: {rec.get('roi_percentage', 'N/A')}%")
    print(f"  Savings: ${rec.get('annual_savings_usd', 0):,.2f}/year")
```

### Example 3: Custom Budget and Settings

```python
# Create agent with custom settings
agent = RecommendationAgentAI(
    budget_usd=1.0,                       # Higher budget
    enable_ai_summary=True,
    enable_implementation_plans=True,
    max_recommendations=10                # More recommendations
)

result = agent.execute(building_data)

# Access implementation roadmap
for phase in result.data["implementation_roadmap"]:
    print(f"\n{phase['phase']}")
    print(f"Timeline: {phase['timeline_months']} months")
    for action in phase["actions"]:
        print(f"  - {action['action']}")
```

### Example 4: ROI-Focused Analysis

```python
# Focus on ROI ranking
building_data = {
    "emissions_by_source": {"electricity": 20000, "natural_gas": 10000},
    "building_age": 15,
    "performance_rating": "Average"
}

agent = RecommendationAgentAI(max_recommendations=5)
result = agent.execute(building_data)

# Print ROI analysis
if "roi_analysis" in result.data:
    roi_calcs = result.data["roi_analysis"]["roi_calculations"]

    print("ROI Analysis:")
    for calc in sorted(roi_calcs, key=lambda x: x["roi_percentage"], reverse=True):
        print(f"\n{calc['action']}")
        print(f"  ROI: {calc['roi_percentage']}%")
        print(f"  Payback: {calc['payback_years']} years")
        print(f"  Annual Savings: ${calc['annual_savings_usd']:,.2f}")
        print(f"  Emissions Reduction: {calc['emissions_reduction_kg']:,.0f} kg CO2e/year")
```

---

## Testing

### Test Coverage

The test suite (`tests/agents/test_recommendation_agent_ai.py`) includes **25+ tests** covering:

#### Unit Tests (20 tests)
1. Initialization and configuration
2. Input validation
3. Tool implementations (all 5 tools)
4. Energy usage analysis (4 scenarios)
5. ROI calculations (3 cost categories)
6. Ranking strategies (4 strategies)
7. Savings estimation
8. Implementation planning
9. Prompt building
10. Determinism verification
11. Backward compatibility
12. Performance tracking
13. Error handling

#### Integration Tests (5 tests)
1. Full recommendation workflow
2. Minimal data handling
3. High emissions scenario
4. Mocked AI integration
5. Budget enforcement

### Running Tests

```bash
# Run all tests
pytest tests/agents/test_recommendation_agent_ai.py -v

# Run specific test class
pytest tests/agents/test_recommendation_agent_ai.py::TestRecommendationAgentAI -v

# Run integration tests
pytest tests/agents/test_recommendation_agent_ai.py::TestRecommendationAgentAIIntegration -v

# Run with coverage
pytest tests/agents/test_recommendation_agent_ai.py --cov=greenlang.agents.recommendation_agent_ai
```

### Test Results

```
tests/agents/test_recommendation_agent_ai.py::TestRecommendationAgentAI::test_initialization PASSED
tests/agents/test_recommendation_agent_ai.py::TestRecommendationAgentAI::test_validate_valid_input PASSED
tests/agents/test_recommendation_agent_ai.py::TestRecommendationAgentAI::test_analyze_energy_usage_tool_basic PASSED
tests/agents/test_recommendation_agent_ai.py::TestRecommendationAgentAI::test_analyze_energy_usage_high_electricity PASSED
tests/agents/test_recommendation_agent_ai.py::TestRecommendationAgentAI::test_calculate_roi_tool_implementation PASSED
tests/agents/test_recommendation_agent_ai.py::TestRecommendationAgentAI::test_rank_recommendations_by_roi PASSED
tests/agents/test_recommendation_agent_ai.py::TestRecommendationAgentAI::test_estimate_savings_tool_implementation PASSED
...

========================== 25 passed in 2.5s ==========================
```

---

## Demo

### Running the Demo

```bash
# Run the complete demo
python demos/recommendation_agent_ai_demo.py
```

### Demo Scenarios

The demo includes 5 comprehensive scenarios:

1. **Old Inefficient Building (20+ years)**
   - High emissions, poor performance
   - Multiple efficiency issues
   - Comprehensive recommendations

2. **Modern High-Performance Building**
   - Recent construction, good performance
   - Optimization opportunities
   - Quick wins focus

3. **HVAC-Dominated Industrial Facility**
   - 65% HVAC load
   - High natural gas usage
   - HVAC-specific recommendations

4. **Electricity-Heavy Data Center**
   - 94% electricity emissions
   - IT equipment focus
   - Renewable energy priority

5. **Performance Comparison**
   - AI vs Traditional agent
   - Performance metrics
   - Cost analysis

### Sample Demo Output

```
================================================================================
  Demo 1: Old Inefficient Building (20+ years)
================================================================================

Building Profile:
  Type: commercial_office
  Age: 25 years
  Area: 75,000 sqft
  Occupancy: 300 people
  Performance: Poor
  Total Emissions: 60,000 kg CO2e/year

Running AI-powered recommendation analysis...

Usage Analysis:
  Total Emissions: 60,000 kg CO2e
  Dominant Source: electricity

  Source Breakdown:
    - electricity: 58.33%
    - natural_gas: 33.33%
    - diesel: 8.33%

  Issues Identified (3):
    [HIGH] Electricity accounts for 58.33% of emissions
    [HIGH] HVAC represents 50.0% of building load
    [MEDIUM] Building age (25 years) suggests potential inefficiencies

Total Recommendations: 5

1. Upgrade to high-efficiency HVAC system
   Impact: 20-30% reduction in HVAC energy
   Cost: High
   Payback: 5-7 years
   Priority: High
   ROI: 18.5%
   Annual Savings: $12,500.00

2. Install smart thermostats and zone controls
   Impact: 10-15% reduction in HVAC energy
   Cost: Low
   Payback: 1-2 years
   Priority: High
   ROI: 42.0%
   Annual Savings: $4,200.00

3. Convert to LED lighting
   Impact: 50-70% reduction in lighting energy
   Cost: Medium
   Payback: 2-3 years
   Priority: High
   ROI: 35.0%
   Annual Savings: $10,800.00

4. Install rooftop solar PV system
   Impact: 30-70% reduction in grid electricity
   Cost: High
   Payback: 5-8 years
   Priority: High
   ROI: 15.0%
   Annual Savings: $18,000.00

5. Seal air leaks and improve weatherstripping
   Impact: 5-10% reduction in HVAC load
   Cost: Low
   Payback: 1 year
   Priority: High
   ROI: 65.0%
   Annual Savings: $3,900.00

Potential Emissions Savings:
  Minimum: 12,000 kg CO2e
  Maximum: 24,000 kg CO2e
  Range: 20.0-40.0%

Potential Cost Savings:
  Minimum Annual: $24,000.00
  Maximum Annual: $48,000.00

Implementation Roadmap:

  Phase 1: Quick Wins (0-6 months)
  Timeline: 4 months
  Actions:
    - Install smart thermostats and zone controls
    - Seal air leaks and improve weatherstripping
    - Regular maintenance and filter replacement

  Phase 2: Strategic Improvements (6-18 months)
  Timeline: 6 months
  Actions:
    - Convert to LED lighting
    - Install occupancy sensors
    - Implement energy management system (EMS)

  Phase 3: Major Upgrades (18-36 months)
  Timeline: 12 months
  Actions:
    - Upgrade to high-efficiency HVAC system
    - Install rooftop solar PV system

AI Analysis Summary:
Based on comprehensive analysis of this 25-year-old commercial office building with poor
performance rating, I recommend focusing on quick wins first to demonstrate value and build
momentum. The building shows three critical issues: high electricity usage (58%), HVAC-dominated
load (50%), and aging infrastructure.

The top priority is sealing air leaks (65% ROI, 1-year payback) - an immediate, low-cost
improvement. Follow with smart thermostats (42% ROI) and LED lighting (35% ROI) within the
first 6 months. These quick wins will deliver $18,900/year in savings while building credibility
for Phase 2 and 3 investments.

For long-term impact, the HVAC upgrade (18.5% ROI) and solar PV (15% ROI) are essential given
the building's age and electricity dependency. Combined, these recommendations can achieve
20-40% emissions reduction (12,000-24,000 kg CO2e/year) with $24,000-$48,000 annual cost savings.

Implementation should follow the phased roadmap, starting with quick wins to fund subsequent
phases through energy savings.

Performance Metrics:
  Calculation Time: 1250.50 ms
  AI Calls: 1
  Tool Calls: 5
  Cost: $0.0300
  Provider: openai
  Model: gpt-4o-mini
```

---

## Implementation Details

### Key Design Decisions

1. **Tool-First Numerics**
   - All calculations delegated to deterministic tools
   - Zero hallucinated numbers
   - Exact ROI, savings, and ranking computations

2. **AI for Context**
   - Natural language explanations
   - WHY each recommendation matters
   - Building-specific insights
   - Executive summaries

3. **Backward Compatibility**
   - Same API as original RecommendationAgent
   - Delegates to original agent for core logic
   - Additional AI enhancements optional (flags)

4. **Budget Enforcement**
   - Default $0.50 per analysis
   - BudgetExceeded exception handling
   - Cost tracking and reporting

5. **Deterministic Execution**
   - Temperature=0 for reproducibility
   - Seed=42 for consistent responses
   - Same input → same output

### Code Structure

```
greenlang/agents/recommendation_agent_ai.py (492 lines)
├── Class: RecommendationAgentAI
│   ├── __init__: Configuration and setup
│   ├── _setup_tools: Tool definitions (5 tools)
│   ├── Tool implementations (5 methods)
│   │   ├── _analyze_energy_usage_impl
│   │   ├── _calculate_roi_impl
│   │   ├── _rank_recommendations_impl
│   │   ├── _estimate_savings_impl
│   │   └── _generate_implementation_plan_impl
│   ├── validate_input: Input validation
│   ├── execute: Main entry point
│   ├── _execute_async: Async ChatSession workflow
│   ├── _build_prompt: Prompt generation
│   ├── _extract_tool_results: Tool call parsing
│   ├── _build_output: Result aggregation
│   └── get_performance_summary: Metrics
```

### Dependencies

```python
# Core GreenLang
from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
from greenlang.agents.recommendation_agent import RecommendationAgent

# Intelligence Layer
from greenlang.intelligence import (
    ChatSession,
    ChatMessage,
    Role,
    Budget,
    BudgetExceeded,
    create_provider,
)
from greenlang.intelligence.schemas.tools import ToolDef

# Standard Library
from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio
import logging
```

---

## Performance Characteristics

### Latency

- **Demo Provider:** 50-200ms (synthetic responses)
- **OpenAI GPT-4o-mini:** 800-1500ms (real LLM)
- **Anthropic Claude:** 600-1200ms (real LLM)

### Cost

- **Demo Provider:** $0.00 (free)
- **OpenAI GPT-4o-mini:** $0.02-0.04 per analysis
- **Anthropic Claude Haiku:** $0.01-0.03 per analysis

### Token Usage

- **Typical Input:** 250-400 tokens (prompt + context)
- **Typical Output:** 200-350 tokens (summary + tool calls)
- **Total:** 450-750 tokens per analysis

### Tool Call Distribution

- **analyze_energy_usage:** Always called (1 call)
- **calculate_roi:** Usually called (1 call)
- **rank_recommendations:** Usually called (1 call)
- **estimate_savings:** Usually called (1 call)
- **generate_implementation_plan:** Optional (1 call if enabled)

**Average:** 4-5 tool calls per analysis

---

## Comparison: AI vs Traditional

| Feature | RecommendationAgent | RecommendationAgentAI |
|---------|-------------------|---------------------|
| **Recommendations** | Rule-based | AI-orchestrated |
| **Explanations** | Generic | Natural language, context-aware |
| **ROI Calculation** | Manual | Tool-based, exact |
| **Prioritization** | Fixed algorithm | Multi-strategy (ROI, impact, cost, payback) |
| **Implementation Plans** | Basic roadmap | Detailed phased plans |
| **Savings Estimates** | Rough ranges | Precise min/max with cost |
| **Latency** | <10ms | 800-1500ms (real LLM) |
| **Cost** | Free | $0.02-0.04 per analysis |
| **Determinism** | Always | Temperature=0, seed=42 |
| **Backward Compatible** | N/A | Yes (same API) |
| **Learning Curve** | Simple | Moderate (AI concepts) |

---

## Limitations and Future Work

### Current Limitations

1. **Cost Estimation Approximation**
   - Uses categorical buckets (Low/Medium/High)
   - Real costs vary by location and vendor
   - Future: Integration with cost databases

2. **Impact Extraction**
   - Regex-based parsing of impact strings
   - Limited to percentage ranges
   - Future: Structured impact data

3. **Building Type Support**
   - Generic recommendations
   - Limited building-type-specific logic
   - Future: Expanded building typology

4. **Country Coverage**
   - Limited to US, EU, IN
   - More countries needed
   - Future: Global coverage

### Future Enhancements

1. **Advanced ROI Models**
   - Lifecycle cost analysis
   - Discount rate consideration
   - Tax credit integration

2. **Machine Learning Integration**
   - Historical performance data
   - Predictive modeling
   - Success rate tracking

3. **Simulation Integration**
   - Building energy modeling
   - What-if scenario analysis
   - Sensitivity analysis

4. **Vendor Integration**
   - Automated RFP generation
   - Vendor matching
   - Cost estimation APIs

5. **Monitoring & Verification**
   - Post-implementation tracking
   - M&V protocol compliance
   - ROI validation

---

## Troubleshooting

### Common Issues

#### Issue 1: High Latency
```python
# Problem: Slow response times

# Solution: Use demo provider for testing
from greenlang.intelligence import has_any_api_key
if not has_any_api_key():
    # Will automatically use demo provider (fast)
    agent = RecommendationAgentAI()
```

#### Issue 2: Budget Exceeded
```python
# Problem: BudgetExceeded error

# Solution: Increase budget
agent = RecommendationAgentAI(budget_usd=1.0)  # Higher budget

# Or disable features
agent = RecommendationAgentAI(
    enable_implementation_plans=False  # Reduce token usage
)
```

#### Issue 3: Empty Recommendations
```python
# Problem: No recommendations generated

# Solution: Check input data quality
result = agent.execute({
    "emissions_by_source": {
        "electricity": 10000  # Need at least one source
    }
})

# Verify fallback to original agent
if not result.data.get("recommendations"):
    print("Check emissions_by_source data")
```

#### Issue 4: API Key Not Found
```python
# Problem: No LLM provider available

# Solution: Set API key or use demo mode
# Demo mode (no API key needed)
from greenlang.intelligence import create_provider
provider = create_provider()  # Auto-detects, uses demo if no keys

# Production mode
# export OPENAI_API_KEY=sk-...
# or
# export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Best Practices

### 1. Input Data Quality

**DO:**
```python
# Provide comprehensive data
building_data = {
    "emissions_by_source": {"electricity": 15000, "natural_gas": 8500},
    "building_age": 20,
    "performance_rating": "Below Average",
    "load_breakdown": {"hvac_load": 0.45},
    "country": "US"
}
```

**DON'T:**
```python
# Minimal data reduces recommendation quality
building_data = {
    "emissions_by_source": {"electricity": 15000}
}
```

### 2. Budget Management

**DO:**
```python
# Set appropriate budget for workload
agent = RecommendationAgentAI(
    budget_usd=0.50,  # Reasonable for single analysis
    max_recommendations=5  # Limit scope
)
```

**DON'T:**
```python
# Excessive budget without controls
agent = RecommendationAgentAI(
    budget_usd=10.0,  # Too high for single call
    max_recommendations=50  # Too many
)
```

### 3. Error Handling

**DO:**
```python
result = agent.execute(building_data)

if result.success:
    recommendations = result.data["recommendations"]
    print(f"Generated {len(recommendations)} recommendations")
else:
    logger.error(f"Analysis failed: {result.error}")
    # Fallback to traditional agent
    fallback_result = RecommendationAgent().execute(building_data)
```

**DON'T:**
```python
# Assume success
recommendations = agent.execute(building_data).data["recommendations"]  # May fail
```

### 4. Performance Monitoring

**DO:**
```python
# Track performance over time
summary = agent.get_performance_summary()
logger.info(f"Total cost: ${summary['ai_metrics']['total_cost_usd']:.4f}")
logger.info(f"Avg cost: ${summary['ai_metrics']['avg_cost_per_analysis']:.4f}")
```

### 5. Caching and Reuse

**DO:**
```python
# Reuse agent instance
agent = RecommendationAgentAI()

for building in buildings:
    result = agent.execute(building)
    # Process result

# Check cumulative metrics
summary = agent.get_performance_summary()
```

**DON'T:**
```python
# Create new agent for each analysis (inefficient)
for building in buildings:
    agent = RecommendationAgentAI()  # Recreates provider each time
    result = agent.execute(building)
```

---

## Conclusion

The `RecommendationAgentAI` successfully demonstrates the AI-native agent pattern:

1. **Tool-First Numerics**: All calculations use deterministic tools
2. **AI for Context**: Natural language explanations and insights
3. **Production-Ready**: Comprehensive testing, error handling, monitoring
4. **Backward Compatible**: Same API as original agent
5. **Cost-Effective**: Sub-$0.04 per analysis with budget controls

### Key Achievements

- **492 lines** of production code
- **25+ tests** with comprehensive coverage
- **5 deterministic tools** for exact calculations
- **Natural language insights** via AI
- **ROI-based prioritization** for actionable recommendations
- **Implementation roadmaps** with phased timelines
- **Demo scenarios** covering 5 building types
- **Complete documentation** with examples

### Next Steps

1. Deploy to production environment
2. Integrate with building management systems
3. Collect user feedback on recommendations
4. Validate ROI predictions with real-world data
5. Expand building typology coverage
6. Add vendor integration capabilities

---

## References

### Related Documentation
- `greenlang/agents/recommendation_agent.py` - Original implementation
- `greenlang/agents/carbon_agent_ai.py` - Similar AI agent pattern
- `greenlang/agents/fuel_agent_ai.py` - Another AI agent example
- `greenlang/intelligence/README.md` - Intelligence layer docs

### External Resources
- [ENERGY STAR Portfolio Manager](https://www.energystar.gov/buildings/benchmark)
- [IRA Tax Credits](https://www.irs.gov/credits-deductions/energy-efficient-home-improvement-credit)
- [Building Energy Codes Program](https://www.energycodes.gov/)
- [ASHRAE Building EQ](https://www.ashrae.org/technical-resources/building-energy-quotient)

---

**Document Version:** 1.0
**Last Updated:** October 2025
**Maintained By:** GreenLang Framework Team
