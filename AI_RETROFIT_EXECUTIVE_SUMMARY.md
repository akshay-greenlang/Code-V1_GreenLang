# AI Agent Retrofit - Executive Summary

**Document:** 4-Week Sprint Plan for AI Integration
**Date:** October 10, 2025
**Team:** 2 AI/ML Engineers
**Timeline:** 4 weeks (Oct 10 - Nov 7, 2025)

---

## Overview

This plan transforms 5 deterministic GreenLang agents into AI-augmented agents using ChatSession infrastructure, enabling natural language understanding while preserving calculation accuracy.

## Key Innovation

**Hybrid Architecture:** LLMs orchestrate workflow and generate explanations; deterministic tools perform all calculations.

```
User Query (Natural Language)
    ↓
ChatSession + LLM Provider
    ↓
Tool Calls (JSON)
    ↓
Deterministic Calculators (existing agents)
    ↓
Structured Results (Quantity objects)
    ↓
LLM Explanation (with provenance)
    ↓
Final Response
```

---

## Agents to Retrofit

| Agent | Complexity | Tools Needed | Target Cost | Target Latency |
|-------|-----------|--------------|-------------|----------------|
| FuelAgent | Simple | 2 | $0.03 | 1.5s |
| CarbonAgent | Simple | 2 | $0.02 | 1.0s |
| GridFactorAgent | Medium | 2 | $0.02 | 0.8s |
| RecommendationAgent | Complex | 3+ | $0.15 | 8.0s |
| ReportAgent | Medium | 2 | $0.08 | 3.0s |

**Full Pipeline:** $0.30 total, 15s latency

---

## Week-by-Week Breakdown

### Week 1: Foundation (FuelAgent + CarbonAgent)
**Goal:** Establish retrofit pattern

**Deliverables:**
- FuelAgentAI with 2 tools (lookup emission factor, calculate emissions)
- CarbonAgentAI with 2 tools (aggregate emissions, calculate intensity)
- System prompts with "No Naked Numbers" enforcement
- 10+ unit tests
- Integration test (fuel → carbon pipeline)
- Pattern documentation

**Success Criteria:**
- All calculations match deterministic agents (±0.01%)
- No naked numbers in responses
- Cost < $0.05 per agent
- Pattern repeatable for remaining agents

---

### Week 2: Real-Time Data (GridFactorAgent)
**Goal:** Integrate caching and data freshness

**Deliverables:**
- GridFactorAgentAI with 2 tools (lookup intensity, compare regions)
- TTL-based caching infrastructure (24hr for static, 5min for live)
- Multi-region comparison capability
- Integration with FuelAgent (grid → fuel emissions)

**Success Criteria:**
- Cache hit rate > 90%
- Live mode blocked in Replay (security)
- Data quality indicators in responses
- Cost < $0.03 (due to caching)

---

### Week 3: Complex Reasoning (RecommendationAgent)
**Goal:** Multi-step reasoning chains

**Deliverables:**
- RecommendationAgentAI with 3+ tools (analyze breakdown, get strategies, calculate potential)
- Chain-of-thought prompting
- Quantified reduction strategies (kgCO2e savings)
- Implementation roadmap generation (Phase 1/2/3)

**Success Criteria:**
- 5-10 tool calls per query (multi-step chain)
- Recommendations are actionable and quantified
- No hallucinated numbers (100% tool-sourced)
- Cost < $0.20 per query

---

### Week 4: Production Readiness (ReportAgent + Integration)
**Goal:** Complete pipeline and production deployment

**Deliverables:**
- ReportAgentAI for narrative report generation
- End-to-end integration tests (all 5 agents)
- Performance benchmarks (cost, latency, accuracy)
- Production deployment guide
- Demo script for stakeholders

**Success Criteria:**
- Complete pipeline < $0.50 total cost
- Complete pipeline < 30s latency
- 50+ tests passing (unit + integration)
- Professional documentation
- Demo runs successfully

---

## Technical Architecture

### Tool Runtime Pattern

Every agent follows this pattern:

```python
from greenlang.intelligence.runtime.session import ChatSession
from greenlang.intelligence.runtime.tools import ToolRegistry

class FuelAgentAI(Agent):
    def __init__(self, provider: LLMProvider):
        self.session = ChatSession(provider)
        self.registry = ToolRegistry()

        # Register domain tools
        self.registry.register(LOOKUP_EMISSION_FACTOR_TOOL)
        self.registry.register(CALCULATE_EMISSIONS_TOOL)

    async def run(self, payload: dict) -> AgentResult:
        # LLM calls tools, tools return Quantity objects
        response = await self.session.chat(
            messages=[system_prompt, user_message],
            tools=self.registry.get_tool_defs(),
            budget=Budget(max_usd=0.05)
        )
        return parse_response(response)
```

### No Naked Numbers Enforcement

All numeric values must be in Quantity format:

```json
{
  "emissions": {
    "value": 1021.0,
    "unit": "kgCO2e"
  }
}
```

LLM references values via macros: `{{claim:0}}`

ToolRuntime validates: No raw numbers allowed.

---

## Cost Breakdown

### Per-Agent Costs
- **Simple agents** (Fuel, Carbon, Grid): $0.02-0.03 per query
- **Complex agents** (Recommendation): $0.15 per query
- **Report generation**: $0.08 per query

### Monthly Budget Projection
```
Queries/month: 10,000
Average cost/query: $0.05
Monthly cost: $500
+ 20% buffer: $600/month
```

**ROI:**
- Current: Human analyst at $80/hr, 30 min/report = $40/report
- AI: $0.30/report
- **Savings: 99.25% per report**

---

## Risk Mitigation

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LLM hallucination | Medium | High | No naked numbers + ToolRuntime enforcement |
| Cost overrun | Low | Medium | Budget caps + cheap models + caching |
| Latency issues | Low | Medium | Async execution + caching + fast models |
| API rate limits | Low | Low | Exponential backoff + multi-provider |

### Schedule Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Week 1 pattern not established | Low | High | Pair programming + FakeProvider for iteration |
| Week 3 complexity underestimated | Medium | Medium | Start simple, iterate to complex + buffer time |
| Week 4 integration issues | Low | Medium | Integration tests throughout + mock providers |

---

## Success Metrics

### Technical Metrics
- [ ] All 5 agents AI-augmented
- [ ] 50+ tests passing
- [ ] Zero naked numbers violations
- [ ] Pipeline cost < $0.50
- [ ] Pipeline latency < 30s
- [ ] Cache hit rate > 90%

### Business Metrics
- [ ] Natural language queries working
- [ ] Explanations clear and actionable
- [ ] Recommendations quantified
- [ ] Reports professional-quality
- [ ] 100% backward compatible

### Quality Metrics
- [ ] Calculation accuracy: ±0.01% vs deterministic
- [ ] Test coverage: >80%
- [ ] Documentation: Complete
- [ ] Demo: Successful

---

## Team Allocation

### Engineer 1 (Backend/Tools)
- **Week 1:** Tool wrappers for Fuel + Carbon
- **Week 2:** Grid tools + caching infrastructure
- **Week 3:** Recommendation tool chains
- **Week 4:** Integration testing + optimization

### Engineer 2 (LLM/Prompts)
- **Week 1:** System prompts + AI agent wrappers
- **Week 2:** Grid prompts + response parsing
- **Week 3:** Chain-of-thought prompting
- **Week 4:** Report agent + demo

### Collaboration
- **Daily standups** (15 min)
- **Weekly demos** to stakeholders
- **Pair programming** on complex components
- **Code reviews** before merge

---

## Deliverables Checklist

### Code
- [ ] `greenlang/agents/fuel_agent_ai.py`
- [ ] `greenlang/agents/carbon_agent_ai.py`
- [ ] `greenlang/agents/grid_factor_agent_ai.py`
- [ ] `greenlang/agents/recommendation_agent_ai.py`
- [ ] `greenlang/agents/report_agent_ai.py`
- [ ] Tool files for each agent (5 files)

### Tests
- [ ] `tests/agents/test_fuel_agent_ai.py` (5+ tests)
- [ ] `tests/agents/test_carbon_agent_ai.py` (5+ tests)
- [ ] `tests/agents/test_grid_factor_agent_ai.py` (5+ tests)
- [ ] `tests/agents/test_recommendation_agent_ai.py` (8+ tests)
- [ ] `tests/agents/test_report_agent_ai.py` (5+ tests)
- [ ] `tests/integration/test_complete_ai_pipeline.py` (3+ tests)

### Documentation
- [ ] `docs/agents/ai_retrofit_guide.md`
- [ ] `docs/AI_AGENTS_PRODUCTION_GUIDE.md`
- [ ] `examples/ai_agents_demo.py`
- [ ] `README_AI_AGENTS.md`

### Deployment
- [ ] Environment variable setup guide
- [ ] Budget configuration
- [ ] Monitoring/telemetry setup
- [ ] Rollback procedure

---

## Example: Complete Pipeline Flow

### Input (Natural Language)
```
"Generate a carbon report for my 50,000 sqft office building in California.
We use 10,000 kWh electricity and 200 therms natural gas per month."
```

### Pipeline Execution

**Step 1: FuelAgentAI** (2 calls)
- Calculate electricity emissions: 3,850 kgCO2e
- Calculate gas emissions: 1,060 kgCO2e

**Step 2: CarbonAgentAI** (1 call)
- Aggregate total: 4,910 kgCO2e/month
- Calculate intensity: 0.098 kgCO2e/sqft

**Step 3: GridFactorAgentAI** (1 call)
- Look up CA grid intensity: 0.237 kgCO2e/kWh
- Context: 54% renewable, low carbon grid

**Step 4: RecommendationAgentAI** (1 call)
- Analyze breakdown: Electricity 78%, Gas 22%
- Top recommendation: Solar PV (2,700 kgCO2e reduction)
- Second: Heat pump (530 kgCO2e reduction)
- Roadmap: Phase 1/2/3 with costs and timelines

**Step 5: ReportAgentAI** (1 call)
- Generate professional markdown report
- Include all sections: Summary, Analysis, Recommendations
- All numbers traced to tool calls (no hallucination)

### Output
Professional 2-page carbon report with:
- Executive summary
- Emissions breakdown
- Grid context
- Top 3 quantified recommendations
- Implementation roadmap
- Total potential reduction: 3,230 kgCO2e (66%)

**Total Cost:** $0.30
**Total Time:** 15 seconds

---

## Next Steps

### Immediate (Week 0)
1. Review plan with team
2. Set up development environment
3. Configure API keys (OpenAI/Anthropic)
4. Create Git branch: `feature/ai-agent-retrofit`

### Week 1 Kickoff
1. Pair programming session: FuelAgent tool design
2. Implement first system prompt
3. Write first integration test
4. Daily standups at 9am

### Ongoing
- Track metrics daily (cost, latency, accuracy)
- Demo progress weekly to stakeholders
- Adjust plan based on learnings
- Document patterns and best practices

---

## References

- **Main Plan:** `AI_AGENT_RETROFIT_4WEEK_PLAN.md` (35 pages, detailed implementation)
- **ChatSession:** `greenlang/intelligence/runtime/session.py`
- **Tool Runtime:** `greenlang/intelligence/runtime/tools.py`
- **Example:** `examples/intelligence/complete_demo.py`

---

## Questions?

Contact: AI Integration Lead
Date: October 10, 2025

**Ready to transform GreenLang agents with AI! Let's build.**
