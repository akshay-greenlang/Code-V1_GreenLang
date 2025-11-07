# GreenLang Agent Categorization Audit

**Version:** 1.0
**Date:** 2025-11-06
**Total Agents Audited:** 49
**Purpose:** Categorize all agents for Intelligence Paradox fix (Phase 2)

---

## Executive Summary

| Category | Count | Percentage | Action Required |
|----------|-------|------------|-----------------|
| **CRITICAL PATH** | 23 | 47% | Keep deterministic, remove ChatSession if present |
| **RECOMMENDATION PATH** | 10 | 20% | Full AI transformation (most already done) |
| **INSIGHT PATH** | 13 | 27% | New AI analysis/investigation capabilities |
| **UTILITY** | 3 | 6% | Framework code, no changes needed |

**Key Findings:**
- ✅ 47% of agents are correctly in critical regulatory path (deterministic)
- ✅ 22% already use ChatSession (good AI transformation progress)
- ⚠️ 0% currently use RAG (huge opportunity for knowledge enhancement)
- 🎯 4 HIGH priority transformation candidates identified

---

## CATEGORY 1: CRITICAL PATH (23 agents - 47%)

**Definition:** Regulatory/compliance calculations that MUST remain deterministic for audit compliance.

**Action:** Keep 100% deterministic, remove ChatSession if present, enhance audit trails.

| # | Agent Name | Location | Critical Function | Status |
|---|------------|----------|-------------------|--------|
| 1 | emissions_calculator_agent_refactored | GL-CBAM-APP/CBAM-Refactored | EU CBAM emissions | ✅ Optimal |
| 2 | intake_agent_refactored | GL-CBAM-APP/CBAM-Refactored | CBAM data validation | ✅ Optimal |
| 3 | reporting_packager_agent_refactored | GL-CBAM-APP/CBAM-Refactored | CBAM report generation | ✅ Optimal |
| 4 | calculator_agent | GL-CSRD-APP/CSRD-Reporting-Platform | ESRS metrics (Zero Hallucination Guarantee) | ✅ Optimal |
| 5 | audit_agent | GL-CSRD-APP/CSRD-Reporting-Platform | ESRS compliance (215+ rules) | ✅ Optimal |
| 6 | intake_agent | GL-CSRD-APP/CSRD-Reporting-Platform | CSRD data ingestion | ✅ Optimal |
| 7 | calculator/agent | GL-VCCI-Carbon-APP/VCCI-Scope3-Platform | Scope 3 emissions (ISO 14083) | ✅ Optimal |
| 8 | intake/agent | GL-VCCI-Carbon-APP/VCCI-Scope3-Platform | Value chain data ingestion | ✅ Optimal |
| 9 | boiler_agent | greenlang/agents | Boiler emissions | ✅ Optimal |
| 10 | grid_factor_agent | greenlang/agents | Grid emission factors | ✅ Optimal |
| 11 | fuel_agent | greenlang/agents | Scope 1/2 fuel emissions | ✅ Optimal |
| 12 | validator_agent | greenlang/agents | Input validation | ✅ Optimal |
| 13 | site_input_agent | greenlang/agents | Site data collection | ✅ Optimal |
| 14 | energy_balance_agent | greenlang/agents | Solar energy balance | ✅ Optimal |
| 15 | field_layout_agent | greenlang/agents | Solar field sizing | ✅ Optimal |
| 16 | load_profile_agent | greenlang/agents | Thermal load profiles | ✅ Optimal |
| 17 | solar_resource_agent | greenlang/agents | TMY solar data | ✅ Optimal |
| 18 | emissions_calculator | packs/cement-lca | Cement LCA emissions | ✅ Optimal |
| 19 | boiler_analyzer | packs/boiler-solar | Boiler efficiency | ✅ Optimal |
| 20 | solar_estimator | packs/boiler-solar | Solar potential | ✅ Optimal |
| 21 | energy_calculator | packs/hvac-measures | HVAC energy | ✅ Optimal |
| 22 | material_analyzer | packs/cement-lca | Material composition | ✅ Optimal |
| 23 | impact_assessor | packs/cement-lca | Environmental impact | ✅ Optimal |
| 24 | intensity_agent (calculations) | greenlang/agents | Carbon intensity calcs | ✅ Optimal |

**Why Critical Path:**
- EU CBAM compliance (regulatory fines)
- EU CSRD compliance (mandatory disclosure)
- GHG Protocol Scope 3 (ISO 14083 conformance)
- Energy audits (building code compliance)
- LCA standards (ISO 14040/14044)

**Audit Trail Requirements:**
- All calculations must be reproducible
- All input/output pairs logged
- All emission factors traceable to sources
- All formulas documented with references
- Zero hallucination guarantee maintained

---

## CATEGORY 2: RECOMMENDATION PATH (10 agents - 20%)

**Definition:** Non-critical decision support that benefits from AI reasoning.

**Action:** Full AI transformation using RAG + ChatSession + multi-tool orchestration.

| # | Agent Name | Location | Function | Priority | Status |
|---|------------|----------|----------|----------|--------|
| 1 | recommendation_agent | greenlang/agents | Static recommendations | **HIGH** | 🔄 Needs AI |
| 2 | reporting_agent | GL-CSRD-APP | CSRD report narratives | **HIGH** | 🔄 Needs AI |
| 3 | recommendation_agent_ai | greenlang/agents | AI recommendations | LOW | ✅ Already AI |
| 4 | report_agent_ai | greenlang/agents | AI report generation | LOW | ✅ Already AI |
| 5 | carbon_agent_ai | greenlang/agents | Carbon insights | LOW | ✅ Already AI |
| 6 | decarbonization_roadmap_agent_ai | greenlang/agents | Decarbonization planning | LOW | ✅ Already AI |
| 7 | industrial_process_heat_agent_ai | greenlang/agents | Process heat analysis | LOW | ✅ Already AI |
| 8 | boiler_replacement_agent_ai | greenlang/agents | Boiler selection | LOW | ✅ Already AI |
| 9 | waste_heat_recovery_agent_ai | greenlang/agents | WHR opportunities | LOW | ✅ Already AI |
| 10 | industrial_heat_pump_agent_ai | greenlang/agents | Heat pump analysis | LOW | ✅ Already AI |

**Additional AI Agents (11-20):**
- thermal_storage_agent_ai
- cogeneration_chp_agent_ai

**Transformation Pattern:**
```python
# BEFORE (recommendation_agent.py):
recommendation = static_lookup(building_type, emissions)

# AFTER (AI transformation):
rag_result = await rag_engine.query("best practices for {building_type}")
recommendation = await session.chat(
    messages=[{"role": "user", "content": f"Recommend solutions for {context}"}],
    tools=[technology_db, financial_analysis, spatial_constraints],
    temperature=0.7
)
```

**RAG Opportunities:**
- Technology database (heat pumps, CHP, solar thermal specs)
- Case studies (successful implementations)
- Best practices (industry benchmarks)
- Regulatory incentives (regional programs)

---

## CATEGORY 3: INSIGHT PATH (13 agents - 27%)

**Definition:** Analysis, investigation, and narrative generation from deterministic data.

**Action:** Add AI reasoning layer while keeping calculations deterministic.

| # | Agent Name | Location | Function | Priority | Transformation |
|---|------------|----------|----------|----------|----------------|
| 1 | benchmark_agent | greenlang/agents | Static benchmarks | **HIGH** | Add AI insights |
| 2 | hotspot/agent | GL-VCCI-Carbon-APP | Emissions hotspot analysis | **HIGH** | Add AI investigation |
| 3 | engagement/agent | GL-VCCI-Carbon-APP | Supplier engagement | MEDIUM | Personalize campaigns |
| 4 | anomaly_agent_iforest | greenlang/agents | Anomaly detection | MEDIUM | Add AI explanations |
| 5 | forecast_agent_sarima | greenlang/agents | Time series forecasting | MEDIUM | Add AI narratives |
| 6 | intensity_agent (insights) | greenlang/agents | Intensity trends | MEDIUM | Add AI analysis |
| 7 | building_profile_agent | greenlang/agents | Building categorization | MEDIUM | Add AI recommendations |
| 8 | aggregator_agent | GL-CSRD-APP | CSRD data aggregation | MEDIUM | Add AI insights |
| 9 | materiality_agent | GL-CSRD-APP | Double materiality | MEDIUM | Add AI reasoning |
| 10 | reporting_agent | GL-CSRD-APP | CSRD report assembly | MEDIUM | Add AI narratives |

**Hybrid Pattern (Deterministic + AI):**
```python
# Step 1: Deterministic calculation
hotspots = pareto_analysis(emissions_data)  # 80/20 rule
segments = segment_by_category(emissions_data)

# Step 2: AI investigation
investigation = await session.chat(
    messages=[{
        "role": "user",
        "content": f"""
        Analyze these emissions hotspots:
        {format_hotspots(hotspots)}

        Context from knowledge base:
        {rag_result}

        Provide:
        1. Root cause hypotheses
        2. Reduction opportunities
        3. Priority actions
        """
    }],
    tools=[historical_trends, industry_benchmarks, best_practices],
    temperature=0.7
)
```

**New Insight Agents to Create:**
1. **AnomalyInvestigationAgent** - Root cause analysis for detected anomalies
2. **ForecastExplanationAgent** - Narrative generation for SARIMA forecasts
3. **BenchmarkInsightAgent** - Competitive analysis and peer comparisons
4. **TrendAnalysisAgent** - Pattern recognition in historical data
5. **ScenarioModelingAgent** - What-if analysis for planning

---

## CATEGORY 4: UTILITY (3 agents - 6%)

**Definition:** Framework code and testing infrastructure.

**Action:** No changes needed.

| # | Agent Name | Location | Purpose |
|---|------------|----------|---------|
| 1 | demo_agent | greenlang/agents | Demo/example |
| 2 | mock | greenlang/agents (2x) | Testing |
| 3 | base, calculator, data_processor, reporter | greenlang/agents | Framework |

---

## Detailed Transformation Plans

### HIGH PRIORITY: recommendation_agent.py

**Current State:**
```python
# Static database lookup
recommendations = db.query(
    building_type=input.building_type,
    emissions_level=input.emissions
)
return recommendations[0]  # First match
```

**Target State:**
```python
async def recommend(self, context: Dict) -> Recommendation:
    # 1. RAG retrieval for context
    case_studies = await rag_engine.query(
        query=f"Decarbonization for {context['building_type']} in {context['region']}",
        collections=["case_studies", "technology_database"],
        top_k=5
    )

    # 2. Multi-tool reasoning
    response = await session.chat(
        messages=[{
            "role": "user",
            "content": f"""
            Recommend decarbonization strategies for:
            - Building: {context['building_type']}
            - Emissions: {context['emissions_tco2e']} tCO2e/year
            - Budget: ${context['budget']}

            Relevant case studies:
            {format_rag_results(case_studies)}

            Use tools to verify feasibility and calculate ROI.
            """
        }],
        tools=[
            technology_compatibility_tool,
            financial_analysis_tool,
            spatial_constraints_tool,
            grid_integration_tool
        ],
        temperature=0.7
    )

    # 3. Parse structured recommendation
    return parse_recommendation(response)
```

**Estimated Effort:** 8-12 hours
**Impact:** HIGH - This agent is used extensively

---

### HIGH PRIORITY: benchmark_agent.py

**Current State:**
```python
# Static peer group comparison
peer_avg = db.query(industry=input.industry).mean()
ratio = input.emissions / peer_avg
return {"status": "above average" if ratio > 1 else "below average"}
```

**Target State:**
```python
async def benchmark(self, context: Dict) -> BenchmarkInsight:
    # 1. Deterministic comparison (keep this)
    peer_stats = self.db.query(
        industry=context['industry'],
        size_category=context['size_category']
    )

    # 2. RAG for best practices
    best_practices = await rag_engine.query(
        query=f"Best in class for {context['industry']}",
        collections=["industry_benchmarks", "best_practices"],
        top_k=3
    )

    # 3. AI-generated insights
    insight = await session.chat(
        messages=[{
            "role": "user",
            "content": f"""
            Provide competitive analysis:

            Company: {context['emissions_tco2e']} tCO2e/year
            Peer Average: {peer_stats['mean']} tCO2e/year
            Peer 25th percentile: {peer_stats['p25']}
            Peer 75th percentile: {peer_stats['p75']}

            Best practices:
            {format_rag_results(best_practices)}

            Provide:
            1. Performance assessment
            2. Gap analysis
            3. Improvement roadmap
            """
        }],
        temperature=0.7
    )

    return BenchmarkInsight(
        metrics=peer_stats,
        narrative=insight.text,
        recommendations=parse_recommendations(insight)
    )
```

**Estimated Effort:** 6-8 hours
**Impact:** MEDIUM-HIGH

---

### HIGH PRIORITY: hotspot/agent.py

**Current State:**
```python
# Pareto analysis (80/20 rule)
hotspots = emissions_data.sort_values(descending=True).head(10)
segments = emissions_data.groupby('category').sum()
return {"hotspots": hotspots, "segments": segments}
```

**Target State:**
```python
async def analyze_hotspots(self, emissions: DataFrame) -> HotspotAnalysis:
    # 1. Deterministic analysis (keep this)
    hotspots = self._pareto_analysis(emissions)  # 80/20 rule
    segments = self._segment_by_category(emissions)
    trends = self._detect_trends(emissions, historical_data)

    # 2. RAG for similar cases
    similar_cases = await rag_engine.query(
        query=f"Emissions hotspots in {context['industry']}",
        collections=["case_studies", "reduction_strategies"],
        top_k=5
    )

    # 3. AI investigation
    investigation = await session.chat(
        messages=[{
            "role": "user",
            "content": f"""
            Investigate these emissions hotspots:

            Top 10 Sources (80% of emissions):
            {format_hotspots(hotspots)}

            Segment Breakdown:
            {format_segments(segments)}

            Trends:
            {format_trends(trends)}

            Similar Cases:
            {format_rag_results(similar_cases)}

            Provide:
            1. Root cause analysis for top hotspots
            2. Quick win opportunities (low-hanging fruit)
            3. Long-term reduction strategies
            4. Priority action plan
            """
        }],
        tools=[
            historical_comparison_tool,
            technology_match_tool,
            cost_benefit_tool
        ],
        temperature=0.7
    )

    return HotspotAnalysis(
        hotspots=hotspots,
        segments=segments,
        investigation=investigation.text,
        action_plan=parse_actions(investigation)
    )
```

**Estimated Effort:** 10-12 hours
**Impact:** HIGH - Value chain visibility

---

## RAG Knowledge Base Requirements

### Collections Needed:

1. **technology_database** (Phase 1 - Already exists ✅)
   - Heat pumps (COP curves, sizing, applications)
   - Solar thermal (irradiance requirements, ROI)
   - CHP/Cogeneration (fuel flexibility, economics)
   - Already created: 3 documents

2. **case_studies** (Phase 1 - Already exists ✅)
   - Industrial implementations
   - ROI data, payback periods
   - Lessons learned
   - Already created: 1 document with 3 cases

3. **ghg_protocol** (Phase 1 - Already exists ✅)
   - Scope 1/2/3 definitions
   - Emission factors
   - Calculation methodologies
   - Already created: 3 documents

4. **industry_benchmarks** (NEW - Phase 2)
   - Energy use intensity by sector
   - Emissions intensity by industry
   - Best-in-class performance
   - Peer group statistics

5. **best_practices** (NEW - Phase 2)
   - Technology selection criteria
   - Implementation guides
   - Common pitfalls
   - Success factors

6. **regulatory_incentives** (NEW - Phase 2)
   - Tax credits (ITC, PTC)
   - Grants and subsidies
   - Regional programs
   - Compliance deadlines

7. **reduction_strategies** (NEW - Phase 2)
   - Quick wins by industry
   - Long-term roadmaps
   - Prioritization frameworks
   - Cost curves

---

## Tool Library Requirements

### Existing Tools (Deterministic)
1. ✅ Emission factor lookups
2. ✅ Grid factor lookups
3. ✅ Calculation tools
4. ✅ Validation tools

### NEW Tools Needed (Phase 2)

**Technology Tools:**
1. `technology_compatibility_check` - Verify tech fits facility constraints
2. `technology_database_search` - Find matching technologies
3. `technology_sizing_calculator` - Size equipment for load

**Financial Tools:**
4. `financial_analysis` - Calculate ROI, NPV, payback
5. `cost_benefit_analysis` - Compare multiple options
6. `incentive_calculator` - Calculate available incentives

**Feasibility Tools:**
7. `spatial_constraints_check` - Verify space requirements
8. `grid_integration_assessment` - Check grid capacity
9. `regulatory_compliance_check` - Verify code compliance

**Analysis Tools:**
10. `historical_comparison` - Compare to historical data
11. `peer_benchmarking` - Compare to peer group
12. `trend_analysis` - Identify patterns
13. `scenario_modeling` - Model what-if scenarios

---

## Implementation Roadmap

### Phase 2.1: Agent Standards (Week 1)
- ✅ Create AgentCategory enum
- ✅ Define interfaces for each category
- ✅ Create base classes (DeterministicAgent, ReasoningAgent, InsightAgent)
- ✅ Document patterns

### Phase 2.2: High-Priority Transformations (Week 2-3)
- 🔄 Transform recommendation_agent.py → recommendation_agent_ai.py
- 🔄 Transform benchmark_agent.py → benchmark_agent_ai.py
- 🔄 Enhance hotspot/agent.py with AI investigation
- 🔄 Transform reporting_agent.py (CSRD) → reporting_agent_ai.py

### Phase 2.3: Tool Library Expansion (Week 3-4)
- 🔄 Create 10+ new tools for reasoning agents
- 🔄 Implement tool registry system
- 🔄 Add tool security and validation

### Phase 2.4: Knowledge Base Expansion (Week 4-5)
- 🔄 Create industry_benchmarks collection
- 🔄 Create best_practices collection
- 🔄 Create regulatory_incentives collection
- 🔄 Create reduction_strategies collection

### Phase 2.5: Medium-Priority Transformations (Week 5-7)
- 🔄 Create AnomalyInvestigationAgent
- 🔄 Create ForecastExplanationAgent
- 🔄 Create BenchmarkInsightAgent
- 🔄 Enhance engagement/agent with AI personalization

---

## Success Metrics

### Quantitative
- [ ] 4 HIGH priority agents transformed
- [ ] 7 MEDIUM priority agents enhanced
- [ ] 10+ new tools created
- [ ] 4+ new RAG collections added
- [ ] 100% critical path agents remain deterministic

### Qualitative
- [ ] Recommendation acceptance rate >70%
- [ ] User satisfaction score >8/10
- [ ] RAG retrieval relevance (NDCG) >0.7
- [ ] Tool orchestration success rate >90%

---

## Risk Mitigation

### Risk: Accidentally transform critical path agent
**Mitigation:** Double-check categorization before transformation. Run full test suite.

### Risk: AI recommendations conflict with regulations
**Mitigation:** All regulatory checks remain deterministic. AI provides guidance only.

### Risk: Performance degradation from LLM calls
**Mitigation:** Cache RAG results. Use async/parallel tool calls. Monitor latency.

### Risk: Cost explosion from excessive LLM usage
**Mitigation:** Enforce budgets with ChatSession. Use batch processing where possible.

---

## Approval Required

### Stakeholder Sign-Off

**Technical Lead:** _________________ Date: _______

**Compliance Officer:** _________________ Date: _______

**Product Owner:** _________________ Date: _______

**Confirmation:**
- [ ] I confirm that CRITICAL PATH agents will remain 100% deterministic
- [ ] I confirm that AI transformation is limited to RECOMMENDATION and INSIGHT paths
- [ ] I confirm that all regulatory calculations maintain audit trails
- [ ] I confirm that this categorization aligns with compliance requirements

---

**Version History:**
- 1.0 (2025-11-06): Initial audit and categorization
- Author: GreenLang Architecture Team
- Next Review: Upon completion of Phase 2.1
