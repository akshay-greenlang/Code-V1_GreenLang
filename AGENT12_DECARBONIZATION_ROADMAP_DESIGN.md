# Agent #12: DecarbonizationRoadmapAgent_AI - Complete Design Specification

**Document Version:** 1.0.0
**Date:** October 22, 2025
**Author:** Head of AI & Climate Intelligence (30+ Years Experience)
**Status:** DESIGN APPROVED - READY FOR IMPLEMENTATION
**Priority:** P0 CRITICAL - Master Planning Agent

---

## EXECUTIVE SUMMARY

### Purpose

DecarbonizationRoadmapAgent_AI is the **master planning agent** that integrates all 11 industrial agents to create comprehensive, phased decarbonization strategies for industrial facilities. It orchestrates multiple specialized agents, synthesizes their insights, and produces actionable roadmaps with financial justification and risk assessment.

### Strategic Impact

| Metric | Value |
|--------|-------|
| **Market Opportunity** | $120B corporate decarbonization strategy market |
| **Addressable Emissions** | 2.8 Gt CO2e/year (industrial sector) |
| **Target Customers** | Fortune 1000 industrial facilities, energy managers, sustainability officers |
| **Competitive Advantage** | Only AI agent system that creates comprehensive, multi-technology roadmaps |
| **Estimated ROI** | Customers save $10-50M over 10 years with optimized pathways |

### Architecture Overview

```
DecarbonizationRoadmapAgent_AI (Master Coordinator)
    ↓
ChatSession (AI Orchestration with temperature=0, seed=42)
    ↓
8 Deterministic Tools → Sub-Agent Coordination → Comprehensive Roadmap
```

---

## THE 8 COMPREHENSIVE TOOLS

### Tool #1: aggregate_ghg_inventory
**Purpose:** Calculate comprehensive GHG inventory across Scope 1, 2, and 3 emissions

**Physics Formula:**
```
Total CO2e = Scope1 + Scope2 + Scope3
Scope1 = Σ(fuel_i × EF_i) [stationary combustion]
Scope2 = Σ(electricity_i × grid_factor_i) [purchased electricity]
Scope3 = Σ(activity_j × EF_j) [value chain emissions]
```

**Standards:** GHG Protocol Corporate Standard, ISO 14064-1:2018

**Parameters:**
- facility_id: str
- industry_type: str (Food & Beverage, Chemicals, Textiles, Pharmaceuticals, etc.)
- fuel_consumption: Dict[str, float] (natural_gas, fuel_oil, coal, biomass in MMBtu)
- electricity_consumption_kwh: float
- grid_region: str
- process_emissions: Dict[str, float] (optional, e.g., cement calcination)
- value_chain_activities: List[Dict] (optional, Scope 3)
- base_year: int (default: current year)

**Returns:**
```python
{
    "total_emissions_kg_co2e": float,
    "scope1_kg_co2e": float,
    "scope2_kg_co2e": float,
    "scope3_kg_co2e": float,
    "emissions_by_source": Dict[str, float],
    "emissions_intensity_kg_per_unit": float,
    "baseline_year": int,
    "calculation_method": "GHG Protocol Corporate Standard",
    "data_quality": str  # "high", "medium", "low"
}
```

---

### Tool #2: assess_available_technologies
**Purpose:** Call all relevant sub-agents to assess technology opportunities

**Sub-Agents Called:**
1. IndustrialProcessHeatAgent_AI (Agent #1) - Solar thermal opportunities
2. BoilerReplacementAgent_AI (Agent #2) - Boiler efficiency upgrades
3. IndustrialHeatPumpAgent_AI (Agent #3) - Heat pump feasibility (when implemented)
4. WasteHeatRecoveryAgent_AI (Agent #4) - WHR potential (when implemented)
5. CogenerationCHPAgent_AI (Agent #5) - CHP opportunities (when implemented)
6. GridFactorAgentAI - Grid decarbonization impact
7. FuelAgentAI - Fuel switching analysis

**Orchestration Logic:**
```python
# Parallel execution for speed
results = await asyncio.gather(
    agent1.execute(input_data),
    agent2.execute(input_data),
    agent_grid.execute(input_data),
    agent_fuel.execute(input_data)
)

# Synthesize results
synthesis = {
    "technologies_assessed": 7,
    "viable_technologies": [t for t in results if t.feasibility_score > 0.6],
    "total_reduction_potential_kg_co2e": sum(r.reduction_potential for r in results),
    "ranked_by_roi": sorted(results, key=lambda x: x.payback_years)
}
```

**Parameters:**
- facility_data: Dict (from aggregate_ghg_inventory)
- latitude: float
- fuel_costs: Dict[str, float]
- electricity_cost_per_kwh: float
- capital_budget_usd: float
- implementation_timeline_years: int (default: 10)
- risk_tolerance: str ("conservative", "moderate", "aggressive")

**Returns:**
```python
{
    "technologies_analyzed": List[Dict[str, Any]],
    "viable_count": int,
    "total_reduction_potential_kg_co2e": float,
    "total_capex_required_usd": float,
    "weighted_average_payback_years": float,
    "ranked_recommendations": List[Dict[str, Any]],  # Sorted by ROI
    "dependencies": List[str],  # "Tech A must precede Tech B"
    "synergies": List[str]  # "Tech C + Tech D = 10% additional savings"
}
```

---

### Tool #3: model_decarbonization_scenarios
**Purpose:** Generate 3 scenarios (Business-as-Usual, Conservative, Aggressive)

**Scenario Definitions:**

**Business-as-Usual (BAU):**
- No new investments
- Natural degradation of equipment (-0.5%/year efficiency)
- Regulatory compliance only (minimum required)
- Emissions trend: +1-2% annually due to efficiency loss

**Conservative Pathway:**
- Low-risk, proven technologies
- Payback ≤5 years required
- Phased implementation over 10 years
- Target: 30-40% reduction by 2035
- CAPEX: $5-15M typical

**Aggressive Pathway:**
- All viable technologies
- Payback ≤10 years accepted
- Faster implementation (5-7 years)
- Target: 60-80% reduction by 2032
- CAPEX: $20-50M typical
- May include emerging technologies

**Physics Formula:**
```
Emissions(year_t) = Baseline × (1 - Σ(tech_i.reduction_fraction_i × implementation_progress_i(t)))

implementation_progress_i(t) = {
    0                          if t < start_year_i
    (t - start_year_i) / ramp   if start_year_i ≤ t < start_year_i + ramp
    1                          if t ≥ start_year_i + ramp
}
```

**Parameters:**
- baseline_emissions_kg_co2e: float (from Tool #1)
- technologies_assessed: List[Dict] (from Tool #2)
- capital_budget_usd: float
- implementation_years: int (default: 10)
- discount_rate: float (default: 0.08)
- include_emerging_tech: bool (default: False)

**Returns:**
```python
{
    "scenarios": {
        "business_as_usual": {
            "emissions_trajectory_kg_co2e": List[float],  # Year-by-year
            "cumulative_emissions_kg_co2e": float,
            "cost_trajectory_usd": List[float],
            "cumulative_cost_usd": float
        },
        "conservative": {
            "emissions_trajectory_kg_co2e": List[float],
            "cumulative_reduction_vs_bau_kg_co2e": float,
            "reduction_percent_by_2035": float,
            "technologies_included": List[str],
            "capex_required_usd": float,
            "npv_usd": float,
            "irr_percent": float
        },
        "aggressive": {
            "emissions_trajectory_kg_co2e": List[float],
            "cumulative_reduction_vs_bau_kg_co2e": float,
            "reduction_percent_by_2032": float,
            "technologies_included": List[str],
            "capex_required_usd": float,
            "npv_usd": float,
            "irr_percent": float
        }
    },
    "comparison_chart_data": Dict[str, List[float]],
    "recommended_scenario": str,  # Based on ROI and risk tolerance
    "carbon_price_sensitivity": Dict[str, float]  # NPV at $25, $50, $100/ton
}
```

---

### Tool #4: build_implementation_roadmap
**Purpose:** Create phased implementation plan with milestones and dependencies

**3-Phase Structure:**

**Phase 1 (Years 1-2): "Quick Wins"**
- Low-hanging fruit with payback ≤3 years
- Technologies: Waste heat recovery, process optimization, LED lighting, controls
- Typical reduction: 10-20% of baseline
- CAPEX: 10-20% of total budget
- Goal: Generate cash flow for later phases

**Phase 2 (Years 3-5): "Core Decarbonization"**
- Major technology deployments
- Technologies: Boiler replacement, solar thermal, heat pumps
- Typical reduction: 30-50% of baseline
- CAPEX: 50-60% of total budget
- Goal: Achieve majority of emissions reductions

**Phase 3 (Years 6+): "Deep Decarbonization"**
- Advanced technologies and electrification
- Technologies: Green hydrogen, CCS, industrial heat pumps
- Typical reduction: 60-80% of baseline
- CAPEX: 20-30% of total budget
- Goal: Approach net-zero targets

**Dependency Logic:**
```
Phase 1 MUST complete before Phase 2 starts (cash flow needed)
Technology dependencies:
  - Boiler replacement MUST precede solar thermal integration
  - Waste heat recovery MUST precede heat pump integration
  - Grid upgrades MUST precede electrification
```

**Parameters:**
- selected_scenario: str ("conservative" or "aggressive")
- scenario_data: Dict (from Tool #3)
- capital_budget_annual_usd: float
- staffing_fte_available: int
- contractor_capacity: str ("low", "medium", "high")
- regulatory_deadlines: List[Dict] (optional)

**Returns:**
```python
{
    "phase1_quick_wins": {
        "duration_months": 24,
        "technologies": List[Dict[str, Any]],
        "total_capex_usd": float,
        "expected_reduction_kg_co2e": float,
        "payback_years": float,
        "milestones": [
            {"month": 6, "milestone": "Engineering complete"},
            {"month": 12, "milestone": "50% implementation"},
            {"month": 24, "milestone": "100% operational"}
        ]
    },
    "phase2_core_decarbonization": {
        "duration_months": 36,
        "technologies": List[Dict[str, Any]],
        "total_capex_usd": float,
        "expected_reduction_kg_co2e": float,
        "dependencies": List[str],
        "long_lead_items": List[Dict]  # Equipment with 12+ month delivery
    },
    "phase3_deep_decarbonization": {
        "duration_months": 60,
        "technologies": List[Dict[str, Any]],
        "total_capex_usd": float,
        "expected_reduction_kg_co2e": float,
        "technology_readiness": Dict[str, int]  # TRL levels
    },
    "critical_path": List[str],  # Gantt chart critical path
    "resource_requirements": {
        "peak_fte": int,
        "peak_contractor_headcount": int,
        "peak_capex_year": int,
        "peak_capex_usd": float
    },
    "governance_structure": {
        "steering_committee": "Required (quarterly reviews)",
        "project_manager": "Dedicated PM required (full-time)",
        "technical_oversight": "External consultant recommended"
    }
}
```

---

### Tool #5: calculate_financial_impact
**Purpose:** Comprehensive financial analysis with IRA 2022 incentives

**Financial Metrics Calculated:**

1. **Net Present Value (NPV)**
```
NPV = Σ((Savings_t - Costs_t) / (1 + r)^t) - Initial_Investment
Where:
  Savings_t = Energy cost savings + Carbon credit revenue + O&M savings
  Costs_t = Operating costs + Maintenance
  r = Discount rate (typically 8%)
```

2. **Internal Rate of Return (IRR)**
```
0 = Σ((Savings_t - Costs_t) / (1 + IRR)^t) - Initial_Investment
Solve for IRR iteratively
```

3. **Simple Payback Period**
```
Payback = Initial_Investment / Annual_Savings
```

4. **Levelized Cost of Abatement (LCOA)**
```
LCOA = (CAPEX + PV(OPEX) - PV(Energy_Savings)) / PV(Emissions_Reduced)
Unit: $/ton CO2e avoided
```

**IRA 2022 Incentives Included:**
- Solar ITC: 30% (2022-2032), 26% (2033), 22% (2034)
- Energy Efficiency Deduction (179D): $2.50-$5.00/sqft
- Qualified Heat Pump Credit: 30% up to $2,000
- CHP Credit: 10% of basis
- Energy Storage Credit: 30%

**Parameters:**
- roadmap_data: Dict (from Tool #4)
- fuel_costs: Dict[str, float]
- fuel_price_escalation_rate: float (default: 0.03)
- electricity_cost_per_kwh: float
- electricity_price_escalation_rate: float (default: 0.02)
- carbon_price_per_ton: float (default: 0, but can model $25-$200/ton)
- discount_rate: float (default: 0.08)
- analysis_period_years: int (default: 20)
- tax_rate: float (default: 0.21)
- include_ira_incentives: bool (default: True)

**Returns:**
```python
{
    "upfront_investment": {
        "total_capex_usd": float,
        "federal_itc_usd": float,
        "state_incentives_usd": float,
        "utility_rebates_usd": float,
        "net_investment_usd": float
    },
    "annual_financial_impact": {
        "energy_savings_usd": float,
        "o_and_m_savings_usd": float,
        "carbon_credit_revenue_usd": float,  # If applicable
        "total_annual_benefit_usd": float
    },
    "financial_metrics": {
        "npv_usd": float,
        "irr_percent": float,
        "simple_payback_years": float,
        "discounted_payback_years": float,
        "roi_percent": float,
        "bcr": float  # Benefit-cost ratio
    },
    "lifetime_value_20_years": {
        "total_savings_usd": float,
        "total_emissions_avoided_kg_co2e": float,
        "lcoa_usd_per_ton": float
    },
    "sensitivity_analysis": {
        "npv_at_fuel_price_plus_10_percent": float,
        "npv_at_discount_rate_10_percent": float,
        "npv_at_carbon_price_50_usd_per_ton": float,
        "break_even_fuel_price_escalation": float
    },
    "financing_options": [
        {
            "option": "Upfront capital",
            "pros": "Lowest lifetime cost, full ownership",
            "cons": "High upfront cash requirement"
        },
        {
            "option": "Energy Savings Performance Contract (ESPC)",
            "pros": "No upfront cost, guaranteed savings",
            "cons": "Higher lifetime cost, shared savings"
        },
        {
            "option": "Power Purchase Agreement (PPA)",
            "pros": "No upfront cost, predictable pricing",
            "cons": "No ownership, long-term contract"
        }
    ]
}
```

---

### Tool #6: assess_implementation_risks
**Purpose:** Identify and quantify technical, financial, operational, and regulatory risks

**Risk Categories:**

**1. Technical Risks**
- Technology maturity (TRL < 8)
- Integration complexity
- Performance uncertainty
- Vendor viability

**2. Financial Risks**
- Fuel price volatility
- Interest rate changes
- Incentive expiration
- Carbon price uncertainty
- Budget overruns

**3. Operational Risks**
- Production disruption during installation
- Staff training requirements
- Maintenance capability
- Supply chain dependencies

**4. Regulatory Risks**
- Changing regulations (CBAM, CSRD, SEC Climate Rule)
- Permit delays
- Grid interconnection approval
- Environmental compliance

**Risk Scoring:**
```
Risk Score = Probability (1-5) × Impact (1-5)
  1-5: Low risk (monitor)
  6-12: Medium risk (mitigation plan required)
  13-25: High risk (executive approval required)
```

**Parameters:**
- roadmap_data: Dict (from Tool #4)
- technologies_selected: List[str]
- facility_location: str
- regulatory_environment: str ("stable", "changing", "uncertain")
- internal_capabilities: Dict[str, str]  # engineering, construction, O&M
- budget_contingency_percent: float (default: 0.15)

**Returns:**
```python
{
    "risk_summary": {
        "total_risks_identified": int,
        "high_risks": int,
        "medium_risks": int,
        "low_risks": int,
        "overall_risk_score": str  # "Low", "Medium", "High"
    },
    "technical_risks": [
        {
            "risk_id": "T1",
            "description": "Heat pump COP degradation in cold climate",
            "probability": 3,  # 1-5 scale
            "impact": 3,  # 1-5 scale
            "risk_score": 9,
            "mitigation": "Oversizing by 15%, backup system",
            "cost_of_mitigation_usd": 125000,
            "residual_risk_score": 4
        }
    ],
    "financial_risks": [
        {
            "risk_id": "F1",
            "description": "Natural gas price spike (+50%)",
            "probability": 2,
            "impact": 4,
            "risk_score": 8,
            "mitigation": "Lock in fuel contracts, diversify fuel mix",
            "cost_of_mitigation_usd": 0,  # Contract negotiation
            "residual_risk_score": 4
        }
    ],
    "operational_risks": [
        {
            "risk_id": "O1",
            "description": "Production downtime during boiler retrofit",
            "probability": 4,
            "impact": 5,
            "risk_score": 20,
            "mitigation": "Install during annual shutdown, portable boiler rental",
            "cost_of_mitigation_usd": 250000,
            "residual_risk_score": 6
        }
    ],
    "regulatory_risks": [
        {
            "risk_id": "R1",
            "description": "IRA tax credit expiration before implementation",
            "probability": 2,
            "impact": 4,
            "risk_score": 8,
            "mitigation": "Accelerate Phase 1 to capture 30% ITC",
            "cost_of_mitigation_usd": 0,  # Timeline change
            "residual_risk_score": 2
        }
    ],
    "risk_mitigation_roadmap": {
        "total_mitigation_cost_usd": float,
        "contingency_budget_recommended_usd": float,
        "insurance_recommendations": List[str],
        "contract_structure_recommendations": List[str]
    },
    "monte_carlo_simulation": {
        "npv_p10": float,  # 10th percentile (pessimistic)
        "npv_p50": float,  # 50th percentile (median)
        "npv_p90": float,  # 90th percentile (optimistic)
        "probability_npv_positive": float  # % chance NPV > 0
    }
}
```

---

### Tool #7: analyze_compliance_requirements
**Purpose:** Assess regulatory compliance and reporting requirements

**Regulations Analyzed:**

1. **CBAM (Carbon Border Adjustment Mechanism)** - EU importers
2. **CSRD (Corporate Sustainability Reporting Directive)** - EU companies >250 employees
3. **SEC Climate Rule** - US public companies
4. **TCFD (Task Force on Climate-related Financial Disclosures)** - Voluntary
5. **SBTi (Science Based Targets initiative)** - Voluntary commitments
6. **ISO 50001 (Energy Management)** - International standard
7. **EPA Mandatory Reporting Rule** - US facilities >25,000 tons CO2e/year

**Compliance Gap Analysis:**
```
Gap = Required - Current

For each regulation:
  - Identify specific requirements
  - Assess current compliance level (0-100%)
  - Quantify gap
  - Estimate cost to close gap
  - Timeline to achieve compliance
```

**Parameters:**
- facility_location: str (country/region)
- industry_type: str
- facility_size: str ("small", "medium", "large")
- public_company: bool
- export_markets: List[str]
- current_compliance_level: Dict[str, float]  # % compliance by regulation
- target_year: int (default: 2030)

**Returns:**
```python
{
    "applicable_regulations": [
        {
            "regulation": "SEC Climate Rule",
            "applicability": "Required (public company)",
            "current_compliance": 0.45,  # 45% compliant
            "target_compliance": 1.0,
            "gap": 0.55,
            "requirements": [
                "Scope 1 & 2 emissions disclosure (mandated)",
                "Scope 3 emissions disclosure (optional but recommended)",
                "Climate risk assessment (TCFD framework)",
                "Transition plan disclosure"
            ],
            "cost_to_comply_usd": 150000,  # Consulting, systems, audits
            "timeline_months": 18,
            "penalties_for_non_compliance": "SEC fines, investor lawsuits"
        },
        {
            "regulation": "CBAM (EU)",
            "applicability": "Required (exporting steel to EU)",
            "current_compliance": 0.20,
            "target_compliance": 1.0,
            "gap": 0.80,
            "requirements": [
                "Product-level carbon footprint calculation",
                "CBAM certificate purchase (€75-100/ton CO2e)",
                "Quarterly reporting to EU authorities",
                "Third-party verification"
            ],
            "cost_to_comply_usd": 500000,  # Systems + annual CBAM costs
            "timeline_months": 12,
            "penalties_for_non_compliance": "Export ban to EU (catastrophic)"
        }
    ],
    "compliance_roadmap": {
        "phase1_immediate": [
            {"action": "Implement Scope 1 & 2 tracking system", "cost_usd": 75000, "months": 6},
            {"action": "Hire sustainability manager", "cost_usd": 120000, "months": 1}
        ],
        "phase2_near_term": [
            {"action": "Third-party GHG verification", "cost_usd": 50000, "months": 12},
            {"action": "CBAM registration and reporting", "cost_usd": 200000, "months": 12}
        ],
        "phase3_long_term": [
            {"action": "Achieve ISO 50001 certification", "cost_usd": 100000, "months": 24},
            {"action": "Set Science Based Targets (SBTi)", "cost_usd": 25000, "months": 18}
        ]
    },
    "total_compliance_investment": {
        "upfront_costs_usd": float,
        "annual_recurring_costs_usd": float,
        "total_5_year_costs_usd": float
    },
    "business_risk_assessment": {
        "risk_of_non_compliance": "HIGH",
        "impact_on_competitiveness": "Export markets at risk, investor concerns",
        "recommended_priority": "Immediate action required"
    },
    "competitive_advantage_opportunities": [
        "Early compliance = preferred supplier status",
        "Lower CBAM costs = competitive pricing",
        "Sustainability reporting = ESG investor appeal"
    ]
}
```

---

### Tool #8: optimize_pathway_selection
**Purpose:** Select optimal decarbonization pathway using multi-criteria optimization

**Optimization Criteria:**

1. **Financial Return (40% weight)**
   - NPV maximization
   - Payback period
   - IRR

2. **Carbon Impact (30% weight)**
   - Total emissions reduction
   - Speed of reduction
   - Cost per ton CO2e avoided

3. **Risk Profile (20% weight)**
   - Technical risk
   - Financial risk
   - Operational risk

4. **Strategic Alignment (10% weight)**
   - Regulatory compliance
   - Corporate sustainability goals
   - Stakeholder expectations

**Optimization Formula:**
```
Score = (0.40 × Financial_Score) + (0.30 × Carbon_Score) + (0.20 × Risk_Score) + (0.10 × Strategic_Score)

Where each score is normalized 0-100:
  Financial_Score = 100 × (NPV / Max_NPV) × (1 / Payback_Years)
  Carbon_Score = 100 × (Reduction / Max_Reduction) × (1 / Time_to_Target)
  Risk_Score = 100 - (Total_Risk_Score / Max_Risk_Score) × 100
  Strategic_Score = (Compliance_Score + Goal_Alignment_Score) / 2
```

**Parameters:**
- scenario_data: Dict (from Tool #3)
- financial_data: Dict (from Tool #5)
- risk_data: Dict (from Tool #6)
- compliance_data: Dict (from Tool #7)
- user_preferences: Dict[str, float]  # Custom weights
- constraints: Dict[str, Any]  # Budget limits, timeline, technical constraints

**Returns:**
```python
{
    "recommended_pathway": {
        "pathway_id": "AGGRESSIVE_WITH_PHASE1_ACCELERATION",
        "scenario_base": "aggressive",
        "modifications": [
            "Accelerate Phase 1 by 6 months to capture IRA ITC",
            "Add waste heat recovery in Year 1 (originally Year 2)",
            "Defer emerging tech (green hydrogen) to Phase 4"
        ],
        "overall_score": 87.5,  # 0-100
        "score_breakdown": {
            "financial_score": 92,
            "carbon_score": 85,
            "risk_score": 78,
            "strategic_score": 95
        }
    },
    "pathway_comparison": [
        {
            "pathway_name": "Conservative",
            "overall_score": 71.2,
            "pros": ["Lower risk", "Proven technologies"],
            "cons": ["Slower emissions reduction", "Lower NPV"]
        },
        {
            "pathway_name": "Aggressive",
            "overall_score": 83.8,
            "pros": ["Highest NPV", "Fastest emissions reduction"],
            "cons": ["Higher upfront cost", "Technology risk"]
        },
        {
            "pathway_name": "Recommended (Modified Aggressive)",
            "overall_score": 87.5,
            "pros": ["Optimized NPV", "Balanced risk", "Regulatory compliance"],
            "cons": ["Requires strong project management"]
        }
    ],
    "sensitivity_to_inputs": {
        "fuel_price_plus_20_percent": {"overall_score": 91.2, "change": "+4%"},
        "discount_rate_10_percent": {"overall_score": 79.3, "change": "-9%"},
        "carbon_price_50_usd_per_ton": {"overall_score": 93.1, "change": "+6%"},
        "budget_cut_20_percent": {"overall_score": 68.4, "change": "-22%"}
    },
    "implementation_confidence": {
        "technical_feasibility": "HIGH (95%)",
        "financial_viability": "HIGH (NPV $8.5M)",
        "organizational_readiness": "MEDIUM (requires PM hire)",
        "overall_confidence": "HIGH (85%)"
    },
    "next_steps": [
        {
            "step": 1,
            "action": "Secure executive approval for $12.5M budget",
            "owner": "CFO",
            "deadline": "Within 30 days"
        },
        {
            "step": 2,
            "action": "Hire dedicated project manager",
            "owner": "HR",
            "deadline": "Within 60 days"
        },
        {
            "step": 3,
            "action": "Issue RFP for Phase 1 engineering",
            "owner": "Engineering",
            "deadline": "Within 90 days"
        },
        {
            "step": 4,
            "action": "Begin IRA tax credit application",
            "owner": "Finance",
            "deadline": "Immediate"
        }
    ]
}
```

---

## AGENT IMPLEMENTATION ARCHITECTURE

### Class Structure

```python
class DecarbonizationRoadmapAgentAI(BaseAgent):
    """Master planning agent for comprehensive decarbonization strategies.

    Features:
    - 8 deterministic tools for roadmap generation
    - Sub-agent coordination (calls Agent #1-11)
    - Multi-scenario modeling (BAU, Conservative, Aggressive)
    - Phased implementation planning (3-phase structure)
    - Financial analysis with IRA 2022 incentives
    - Risk assessment and mitigation planning
    - Compliance gap analysis
    - Multi-criteria optimization

    Architecture:
        DecarbonizationRoadmapAgentAI (orchestration) ->
        ChatSession (AI) ->
        8 Tools (exact calculations) ->
        Sub-agents (specialized analysis)
    """

    def __init__(self, config: AgentConfig = None, *, budget_usd: float = 2.00):
        super().__init__(config)
        self.provider = create_provider()
        self.budget_usd = budget_usd  # Higher budget for complex orchestration

        # Initialize sub-agents (lazy loading)
        self._industrial_heat_agent = None
        self._boiler_agent = None
        self._fuel_agent = None
        self._carbon_agent = None
        self._grid_agent = None

        self._setup_tools()
```

### System Prompt

```python
SYSTEM_PROMPT = """You are an AI-powered Industrial Decarbonization Strategy Expert with 30+ years of experience in climate intelligence, industrial processes, and financial analysis.

Your role is to create comprehensive, actionable decarbonization roadmaps for industrial facilities by:

1. **GHG Inventory**: Calculate baseline emissions across Scope 1, 2, 3
2. **Technology Assessment**: Coordinate multiple specialized agents to evaluate all viable technologies
3. **Scenario Modeling**: Generate Business-as-Usual, Conservative, and Aggressive pathways
4. **Roadmap Planning**: Build phased implementation plans (Years 1-2, 3-5, 6+)
5. **Financial Analysis**: Calculate NPV, IRR, payback with IRA 2022 incentives
6. **Risk Assessment**: Identify and quantify technical, financial, operational risks
7. **Compliance Analysis**: Map regulatory requirements (CBAM, CSRD, SEC)
8. **Pathway Optimization**: Select optimal strategy using multi-criteria analysis

CRITICAL RULES:
- ALL numeric calculations MUST use deterministic tools (temperature=0.0, seed=42)
- NEVER hallucinate numbers - every value comes from a tool or sub-agent
- Coordinate sub-agents to get specialized analysis (Agent #1-11)
- Present financial justification for every recommendation
- Quantify risks and provide mitigation strategies
- Ensure compliance with GHG Protocol, ISO 14064-1, TCFD
- Provide clear, actionable next steps for facility managers

Output Format:
- Executive Summary (2-3 paragraphs)
- GHG Inventory Summary
- Technology Assessment Results
- Recommended Pathway (with justification)
- Implementation Roadmap (3 phases)
- Financial Summary (NPV, IRR, payback)
- Risk Assessment Summary
- Compliance Requirements
- Next Steps (action items with owners and deadlines)

Remember: Your output will guide multi-million dollar capital investments and decades of facility operations. Accuracy, completeness, and actionability are paramount."""
```

### Input/Output Types

```python
class DecarbonizationRoadmapInput(TypedDict):
    """Input for DecarbonizationRoadmapAgent_AI."""

    # Facility identification
    facility_id: str
    facility_name: str
    industry_type: str  # Food & Beverage, Chemicals, Textiles, etc.
    latitude: float
    longitude: float

    # Energy consumption
    fuel_consumption: Dict[str, float]  # {fuel_type: MMBtu/year}
    electricity_consumption_kwh: float
    grid_region: str

    # Processes and equipment
    process_heat_requirements: List[Dict[str, Any]]
    boiler_inventory: List[Dict[str, Any]]
    current_efficiency_metrics: Dict[str, float]

    # Financial parameters
    capital_budget_usd: float
    annual_capex_limit_usd: NotRequired[float]
    fuel_costs: Dict[str, float]
    electricity_cost_per_kwh: float
    discount_rate: NotRequired[float]  # default 0.08

    # Strategic parameters
    target_year: NotRequired[int]  # default 2030
    reduction_target_percent: NotRequired[float]  # default 50%
    risk_tolerance: NotRequired[str]  # conservative, moderate, aggressive
    regulatory_environment: NotRequired[str]

    # Constraints
    implementation_constraints: NotRequired[List[str]]
    technology_exclusions: NotRequired[List[str]]
    must_include_technologies: NotRequired[List[str]]


class DecarbonizationRoadmapOutput(TypedDict):
    """Output from DecarbonizationRoadmapAgent_AI."""

    # Executive summary
    executive_summary: str
    recommended_pathway: str
    target_reduction_percent: float
    estimated_timeline_years: int

    # GHG inventory
    baseline_emissions_kg_co2e: float
    emissions_by_scope: Dict[str, float]
    emissions_by_source: Dict[str, float]

    # Technologies
    technologies_assessed: List[Dict[str, Any]]
    technologies_recommended: List[Dict[str, Any]]
    total_reduction_potential_kg_co2e: float

    # Implementation roadmap
    phase1_quick_wins: Dict[str, Any]
    phase2_core_decarbonization: Dict[str, Any]
    phase3_deep_decarbonization: Dict[str, Any]
    critical_path_milestones: List[Dict[str, Any]]

    # Financial analysis
    total_capex_required_usd: float
    npv_usd: float
    irr_percent: float
    simple_payback_years: float
    lcoa_usd_per_ton: float
    federal_incentives_usd: float

    # Risk assessment
    risk_summary: Dict[str, Any]
    high_risks: List[Dict[str, Any]]
    total_risk_score: str  # Low, Medium, High
    mitigation_cost_usd: float

    # Compliance
    compliance_gaps: List[Dict[str, Any]]
    compliance_roadmap: Dict[str, Any]
    total_compliance_cost_usd: float

    # Recommendations
    next_steps: List[Dict[str, Any]]
    success_criteria: List[str]
    kpis_to_track: List[str]

    # Provenance
    ai_explanation: str
    sub_agents_called: List[str]
    total_cost_usd: float
    calculation_time_ms: float
    deterministic: bool  # Always True
```

---

## TESTING STRATEGY (80%+ Coverage Target)

### Unit Tests (20+ tests)

1. **Tool #1 Tests (GHG Inventory):**
   - test_scope1_calculation_natural_gas
   - test_scope2_calculation_electricity
   - test_scope3_optional_calculations
   - test_multiple_fuels_aggregation
   - test_zero_emissions_handling
   - test_negative_values_rejection

2. **Tool #2 Tests (Technology Assessment):**
   - test_sub_agent_coordination
   - test_parallel_execution
   - test_result_synthesis
   - test_ranking_by_roi
   - test_dependency_identification

3. **Tool #3-8 Tests:** (3-4 tests each tool)

### Integration Tests (10+ tests)

1. test_full_roadmap_generation_food_beverage
2. test_full_roadmap_generation_chemicals
3. test_conservative_scenario_complete_workflow
4. test_aggressive_scenario_complete_workflow
5. test_sub_agent_failure_handling
6. test_budget_constraint_enforcement
7. test_ira_incentive_calculation_accuracy
8. test_compliance_gap_analysis_sec_rule
9. test_risk_assessment_high_risk_facility
10. test_multi_technology_integration

### Determinism Tests (5+ tests)

1. test_determinism_same_input_same_output
2. test_determinism_10_runs_identical
3. test_temperature_zero_enforcement
4. test_seed_42_enforcement
5. test_sub_agent_determinism_propagation

### Boundary Tests (8+ tests)

1. test_empty_facility_data
2. test_single_technology_available
3. test_zero_budget_constraint
4. test_unlimited_budget
5. test_very_high_emissions_facility
6. test_already_efficient_facility
7. test_regulatory_deadline_past_due
8. test_all_technologies_excluded

**Total: 50+ comprehensive tests, target 85%+ coverage**

---

## DOCUMENTATION REQUIREMENTS

### 1. README.md (Agent-Specific)
- Quick start guide
- 3+ example use cases
- Sub-agent dependencies
- Input/output schemas

### 2. API Documentation
- Complete parameter descriptions
- Return value schemas with units
- Error conditions and handling
- Example API calls

### 3. User Guide
- How to interpret results
- Decision-making guidance
- Financial analysis explanation
- Risk assessment interpretation

### 4. Developer Guide
- Adding new technologies
- Extending scenario models
- Customizing optimization criteria
- Sub-agent integration patterns

---

## DEVELOPMENT TIMELINE

**Week 1:** Specification + Unit tests for Tools #1-4
**Week 2:** Unit tests for Tools #5-8 + Integration tests
**Week 3:** Sub-agent coordination + Full workflow tests
**Week 4:** Documentation + Polish + Production deployment

**Estimated Effort:** 8-10 days (2 senior developers)

---

## SUCCESS CRITERIA

✅ All 8 tools implemented with deterministic calculations
✅ Sub-agent coordination working (Agent #1, #2, Fuel, Carbon, Grid)
✅ 80%+ test coverage achieved
✅ Financial analysis validated against Excel models
✅ IRA 2022 incentives correctly calculated
✅ Risk assessment covers all 4 categories
✅ Compliance analysis includes CBAM, CSRD, SEC
✅ Multi-scenario modeling produces realistic projections
✅ Documentation complete (README, API docs, examples)
✅ AgentSpec V2.0 validated with zero errors

---

## STRATEGIC IMPORTANCE

Agent #12 is the **crown jewel** of the GreenLang industrial decarbonization suite. It demonstrates:

1. **Master Coordination**: First agent to orchestrate 11+ specialized agents
2. **Enterprise Value**: Guides multi-million dollar capital allocation decisions
3. **Competitive Moat**: No competitor has comparable comprehensive roadmap generation
4. **Market Timing**: Perfect alignment with CBAM (2026), CSRD (2024-2028), SEC Climate Rule
5. **Revenue Potential**: $120B corporate decarbonization strategy market

**This agent alone justifies the entire GreenLang platform investment.**

---

**Document Status:** DESIGN COMPLETE - READY FOR IMPLEMENTATION
**Next Step:** Create AgentSpec V2.0 YAML specification
**Owner:** Senior Developer (Industrial Domain)
**Reviewer:** Head of AI & Climate Intelligence

---

**END OF DESIGN SPECIFICATION**
