# PRD-PACK-026: SME Net Zero Pack

**Pack ID:** PACK-026-sme-net-zero
**Category:** Net Zero Packs
**Tier:** Standalone
**Version:** 1.0.0
**Status:** Draft
**Author:** GreenLang Product Team
**Date:** 2026-03-18
**Prerequisite:** None (standalone; enhanced with PACK-021/022/023 if present)

---

## 1. Executive Summary

### 1.1 Problem Statement

Small and Medium Enterprises (SMEs) -- defined as businesses with fewer than 250 employees -- represent 99% of all businesses globally, employ approximately 70% of the workforce, and collectively account for over 50% of global greenhouse gas emissions. Despite this enormous aggregate impact, SMEs are systematically underserved by existing climate tools, frameworks, and advisory services, which are overwhelmingly designed for large enterprises with dedicated sustainability teams, substantial budgets, and deep technical expertise. SMEs face a distinct set of barriers that prevent meaningful climate action:

1. **Resource constraints**: The typical SME has zero dedicated sustainability staff, an annual sustainability budget of less than $10,000, and no internal expertise in GHG accounting, science-based targets, or decarbonization planning. Existing net-zero tools (including PACK-021/022) assume organizational capacity that SMEs simply do not have: full-time sustainability managers, data management systems, and consultant relationships. An SME owner-operator cannot spend 400+ hours building a GHG baseline -- they need it done in under 2 hours.

2. **Data availability gap**: SMEs typically lack metered energy data, detailed procurement records, fleet management systems, and supply chain visibility. They have utility bills (often quarterly, not monthly), bank statements, and tax returns -- not activity data feeds. Any credible SME climate solution must work with spend-based emission factors, industry averages, and minimal primary data, progressively upgrading data quality as the organization matures.

3. **Complexity barrier**: The GHG Protocol Corporate Standard, SBTi Net-Zero Standard, and ISO 14064-1 are designed for organizations with the capacity to implement complex methodologies across multiple scopes and categories. An SME bakery or plumbing company cannot navigate 15 Scope 3 categories, choose between ACA and SDA pathways, or calculate a Marginal Abatement Cost Curve. SMEs need a simplified, guided experience that abstracts complexity while maintaining methodological credibility.

4. **Cost-benefit focus**: Unlike large enterprises driven by regulatory compliance (CSRD, SEC Climate Rule) and investor pressure (CDP, TCFD), SMEs are motivated by cost savings, customer requirements (supply chain mandates), and competitive differentiation. Every climate action must demonstrate a clear return on investment (ROI) with payback periods under 3 years. SMEs will not invest in actions that do not pay for themselves.

5. **Funding access**: SMEs are eligible for thousands of grants, subsidies, tax incentives, and green loan programs globally (EU SME Initiative, US DOE Small Business grants, UK Energy Efficiency grants, local government programs), but discovering and applying for these programs is itself a resource-intensive task. Most SMEs are unaware of available funding, miss application deadlines, or lack the capacity to prepare competitive applications.

6. **Certification confusion**: SMEs face a bewildering landscape of climate certifications and commitments: SME Climate Hub (UN-backed, free), B Corp (certification fee, comprehensive assessment), ISO 14001 (formal EMS, audit-based), Carbon Trust Standard (UK-focused), Climate Active (Australia), and numerous national schemes. Each has different requirements, costs, and market recognition. SMEs need guidance on which certifications are most valuable for their specific context.

7. **Scope 3 dominance without data**: For service-sector SMEs (the majority), Scope 3 emissions (particularly Categories 1, 6, and 7 -- purchased goods/services, business travel, and employee commuting) typically represent 80%+ of total emissions. Yet these are precisely the categories where SMEs have the least data and the least control. Practical Scope 3 management for SMEs requires spend-based estimation, simplified supplier engagement, and focused action on the 2-3 categories that matter most.

8. **Isolation and lack of peer support**: Large enterprises have industry associations, sustainability consortia, and peer networks for benchmarking and knowledge sharing. SMEs often operate in isolation, with no visibility into what similar businesses are achieving. Peer benchmarking -- comparing emissions intensity, reduction progress, and action adoption rates against similar-sized businesses in the same sector -- is a powerful motivator but unavailable in existing tools.

### 1.2 Solution Overview

PACK-026 is the **SME Net Zero Pack** -- the sixth pack in the "Net Zero Packs" category. It is purpose-built for Small and Medium Enterprises with fewer than 250 employees, providing a simplified, cost-effective, and practical net-zero journey that respects the reality of SME constraints: limited time, limited budget, limited data, and limited expertise. The pack wraps and orchestrates existing GreenLang platform components into an SME-optimized experience with 8 new engines, 6 workflows, 8 templates, 10 integrations, and 6 presets.

Unlike PACK-021 (Starter, designed for general-purpose net-zero strategy) or PACK-022 (Acceleration, designed for large corporates), PACK-026 is fundamentally different in philosophy:

- **Time-optimized**: Full baseline in under 2 hours of data collection, not 8+ hours
- **Spend-based first**: Uses financial data (invoices, bank statements, tax returns) as primary input, not activity data
- **Quick wins focus**: Prioritizes actions with <3-year payback and immediate cost savings
- **Grant-aware**: Automatically matches SMEs to available funding programs
- **Certification-guided**: Clear pathways to SME Climate Hub, B Corp, ISO 14001
- **Peer-benchmarked**: Compare against similar businesses by sector, size, and region
- **Mobile-friendly**: Dashboards and reports designed for phone/tablet access
- **DIY-first**: Minimal consultant dependency; the pack is the consultant

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Spreadsheet / Generic Tool | PACK-026 SME Net Zero Pack |
|-----------|--------------------------------------|----------------------------|
| Time to baseline | 40-100+ hours (if attempted at all) | <2 hours data collection, <30 min calculation |
| Data requirement | Activity data, meter readings, fleet logs | Spend data, utility bills, headcount -- minimal burden |
| Cost to implement | $20,000-$100,000+ (consultant-driven) | <$5,000 for micro, <$20,000 for medium (DIY) |
| Actions identified | Generic action lists from consultants | 500+ SME-specific actions ranked by ROI, payback, ease |
| Funding discovery | Manual research across hundreds of programs | Automated grant matching (10,000+ grants globally) |
| Payback visibility | Qualitative cost descriptions | Quantified NPV, IRR, payback for every action |
| Certification path | Unclear requirements, multiple consultants | Guided readiness assessment with gap closure plans |
| Peer comparison | No visibility into peer performance | Benchmarking by NACE code, employee count, region |
| Progress tracking | Annual spreadsheet updates | Simple annual tracker with KPI dashboard |
| Scope 3 approach | Complex 15-category analysis | Focused on Cat 1, 6, 7 (typically 80%+ of SME Scope 3) |
| Audit readiness | No documentation trail | SHA-256 provenance, full calculation lineage |

### 1.4 SME Market Overview

| Statistic | Value | Source |
|-----------|-------|--------|
| SMEs as % of global businesses | 99% | OECD (2023) |
| SMEs as % of global employment | 70% | World Bank (2023) |
| SMEs as % of global GHG emissions | 50-70% (estimated) | CDP/UNFCCC (2023) |
| SMEs with net-zero targets | <5% | SME Climate Hub (2024) |
| SMEs with GHG baselines | <10% | CDP Supply Chain (2024) |
| Average SME sustainability budget | <$10,000/year | OECD/EU SME surveys (2023) |
| SMEs citing cost as top barrier | 68% | EU SME Green Transition survey (2023) |
| SMEs citing expertise as top barrier | 54% | EU SME Green Transition survey (2023) |
| SMEs citing data as top barrier | 47% | EU SME Green Transition survey (2023) |

### 1.5 Target Users

**Primary:**
- SME owner-operators and managing directors making climate commitments
- Office managers and operations staff collecting sustainability data
- SME finance managers evaluating cost-benefit of climate actions
- SMEs required by supply chain partners to report emissions (Scope 3 mandates)

**Secondary:**
- SME advisors and accountants providing sustainability guidance
- Industry associations supporting member SMEs with climate programs
- Local chambers of commerce running SME sustainability programs
- Government agencies administering SME green transition grants
- Supply chain sustainability teams onboarding SME suppliers

### 1.6 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to complete GHG baseline | <2 hours data collection | Time from first login to validated baseline |
| Implementation cost (micro, 1-9 employees) | <$5,000 total first year | All-in cost including pack license, data collection time, initial actions |
| Implementation cost (medium, 50-249 employees) | <$20,000 total first year | All-in cost including pack license, data collection, implementation |
| Average payback period of recommended actions | <3 years | Weighted average payback across top 10 recommended actions |
| Grant application success rate | >50% of submitted applications approved | Applications approved / applications submitted |
| SME Climate Hub certification rate | >70% achieve within 12 months | Certified / committed within 12-month window |
| Peer benchmarking satisfaction | >80% find benchmarks useful | User survey response |
| Mobile dashboard usage | >40% of sessions on mobile | Mobile session / total session ratio |
| Year-over-year emission reduction (participants) | >5% average annual reduction | Aggregate participant emission change |
| Customer NPS | >60 | Net Promoter Score survey |

---

## 2. Regulatory & Framework Basis

### 2.1 Primary Frameworks

| Framework | Reference | Pack Relevance |
|-----------|-----------|----------------|
| GHG Protocol Corporate Standard | WRI/WBCSD (2004, 2015 update) | Core GHG inventory methodology, simplified for SME application |
| GHG Protocol Scope 3 Standard | WRI/WBCSD (2011) | Scope 3 methodology; SME focus on spend-based factors for Cat 1, 6, 7 |
| SBTi SME Pathway | SBTi (launched 2023) | Simplified near-term target: 50% absolute reduction by 2030 from a recent baseline, covering Scope 1+2 (mandatory) and Scope 3 (encouraged) |
| SME Climate Hub | UN Race to Zero (2020) | UN-backed SME net-zero commitment: 1-2-3 pledge (halve by 2030, net zero by 2050, disclose annually) |
| IPCC AR6 | IPCC Sixth Assessment Report (2021) | GWP-100 values for all greenhouse gases |
| Paris Agreement | UNFCCC (2015) | 1.5C temperature alignment target |

### 2.2 SME-Specific Standards & Certifications

| Standard / Certification | Reference | Pack Relevance |
|--------------------------|-----------|----------------|
| SBTi SME Target Setting | SBTi SME Route (2023) | Streamlined target validation: no SDA required, simplified Scope 3, immediate validation |
| SME Climate Hub Commitment | SME Climate Hub / UN Race to Zero | Free commitment platform; 1-2-3 pledge; connected to Race to Zero campaign |
| B Corp Certification | B Lab (ongoing) | Comprehensive ESG assessment including climate; B Corp Climate Collective for acceleration |
| ISO 14001:2015 | ISO | Environmental Management System; adaptable to SME scale with simplified documentation |
| Carbon Trust Standard | Carbon Trust (UK) | UK-focused carbon measurement and reduction certification |
| Climate Active | Australian Government | Australian carbon neutral certification with SME pathway |
| CDP Supply Chain | CDP (2024) | SME supplier disclosure via simplified questionnaire (SME module) |
| SBTN (Science Based Targets for Nature) | SBTN (2023) | Nature-related targets; emerging SME guidance |

### 2.3 Supporting Standards

| Standard / Framework | Reference | Pack Relevance |
|---------------------|-----------|----------------|
| GHG Protocol Scope 3 Evaluator Tool | WRI (2013) | Simplified Scope 3 estimation tool methodology (used as reference for spend-based approach) |
| DEFRA Emission Factors | UK Government (annual) | Widely-used emission factors including spend-based factors by SIC code |
| EPA Emission Factors | US EPA (annual) | US emission factors including EEIO (environmentally-extended input-output) models |
| ecoinvent | ecoinvent v3.x | LCA database for product-level emission factors |
| NACE Rev. 2 | Eurostat | Industry classification for peer benchmarking and sector-specific emission factors |
| EU SME Green Transition | EU Commission (2023) | Policy framework for SME sustainability transition |
| OECD SME and Entrepreneurship Outlook | OECD (2023) | SME policy landscape and green transition barriers |

---

## 3. Technical Architecture

### 3.1 Components Overview

| Component Type | Count | Description |
|----------------|-------|-------------|
| Engines | 8 | SME-optimized deterministic calculation engines |
| Workflows | 6 | Simplified multi-phase orchestration workflows |
| Templates | 8 | SME-friendly report and dashboard templates |
| Integrations | 10 | Agent, app, grant, certification, and accounting bridges |
| Presets | 6 | Size-tier and sector-specific configurations |
| Config | 1 | Runtime configuration (Pydantic v2) |
| Demo | 1 | Demo configuration |

### 3.2 Engines

| # | Engine | Purpose |
|---|--------|---------|
| 1 | `sme_baseline_engine.py` | Simplified GHG baseline using spend-based emission factors, industry averages, and minimal data. Accepts utility bills (electricity, gas), fuel receipts, headcount, floor area, and annual revenue/spend as primary inputs. Calculates Scope 1 (natural gas, vehicle fuel), Scope 2 (electricity), and simplified Scope 3 (Cat 1 via spend, Cat 6 via travel spend, Cat 7 via commute estimation) using DEFRA/EPA EEIO factors. Produces a 1-2 page visual baseline with confidence intervals reflecting data quality. |
| 2 | `simplified_target_engine.py` | SME-appropriate target setting aligned with SBTi SME Pathway. Sets near-term targets (50% absolute reduction by 2030 from baseline), validates against SBTi SME Route criteria (no SDA required, simplified Scope 3 treatment), and generates annual milestone pathway. Supports both SBTi-aligned targets (for formal validation) and SME Climate Hub targets (for campaign commitment). Adjusts ambition based on sector and current intensity. |
| 3 | `quick_wins_engine.py` | Identifies high-impact, low-cost actions from a curated database of 500+ SME-specific quick wins. Actions include LED lighting retrofit, HVAC optimization, smart thermostat installation, renewable energy tariff switching, remote/hybrid work policies, fleet electrification (e-van/e-car), waste reduction, water efficiency, and procurement optimization. Each action tagged with cost range, annual savings, payback period, CO2e reduction, difficulty level (DIY/contractor/specialist), and applicability by sector and size tier. Filters and ranks actions by ROI and ease of implementation for the specific SME profile. |
| 4 | `action_prioritization_engine.py` | Ranks all identified actions (from quick_wins_engine and broader DECARB-X library) by composite score combining ROI (30% weight), payback period (25%), ease of implementation (20%), CO2e abatement potential (15%), and co-benefits (10%). Produces a prioritized action list with recommended implementation sequence, resource requirements, and cumulative impact projection. Applies SME-specific constraints: maximum upfront cost thresholds, staff availability, and landlord/tenant considerations (for leased premises). |
| 5 | `sme_progress_tracker.py` | Simple annual progress tracking designed for SMEs. Collects year-over-year data (same minimal inputs as baseline), calculates emission changes, tracks progress against targets, and produces a KPI dashboard with 5-8 key metrics: total emissions (tCO2e), change vs. baseline (%), change vs. prior year (%), emissions intensity (tCO2e/employee and tCO2e/revenue), energy consumption (kWh), renewable energy share (%), and actions completed. Triggers alerts when off-track and recommends corrective actions. |
| 6 | `cost_benefit_engine.py` | Calculates financial metrics for each decarbonization action: Net Present Value (NPV) over 5/10/15-year horizons, Internal Rate of Return (IRR), simple payback period (months), discounted payback period, annual operating cost savings, upfront capital cost, ongoing maintenance cost, and total cost of ownership (TCO). Incorporates available grants/subsidies to calculate net-of-funding costs. Uses SME-appropriate discount rates (8-12% reflecting higher cost of capital). Produces cost-benefit summary tables and charts suitable for board/owner decision-making. |
| 7 | `grant_finder_engine.py` | Matches SME profile (country, region, sector, size, emissions profile, planned actions) against a database of 10,000+ grants, subsidies, tax incentives, and green loan programs globally. Covers EU SME Initiative (Horizon Europe, LIFE, Cohesion Funds), US DOE Small Business grants (SBIR/STTR for clean energy), UK schemes (Energy Efficiency grants, Boiler Upgrade Scheme, Enhanced Capital Allowances), country-specific programs (Germany KfW, France ADEME, Japan METI), and local/regional programs. Provides match score (0-100), eligibility assessment, application deadline, estimated award amount, and pre-filled application template guidance. |
| 8 | `certification_readiness_engine.py` | Assesses readiness for SME-relevant climate certifications and commitments. Covers SME Climate Hub (5-criterion assessment: pledge, measure, reduce, report, offset), B Corp Climate (climate-specific B Impact Assessment questions), ISO 14001 (EMS documentation requirements for SMEs), and Carbon Trust Standard (measurement, management, reduction evidence). Produces per-certification readiness score (0-100), gap list with remediation actions, estimated time-to-certification, and estimated cost. Recommends optimal certification sequence based on SME profile, market context, and supply chain requirements. |

### 3.3 Engine Specifications

#### 3.3.1 Engine 1: SME Baseline Engine

**Purpose:** Calculate a credible GHG baseline from minimal SME data inputs.

**Data Input Tiers (progressive data quality):**

| Tier | Data Available | Approach | Accuracy |
|------|---------------|----------|----------|
| Bronze | Revenue, headcount, sector (NACE code) | Full industry average estimation | +/- 40-60% |
| Silver | Utility bills (electricity, gas), basic fuel data, travel spend | Hybrid: activity data (energy) + spend-based (procurement, travel) | +/- 20-35% |
| Gold | Detailed energy data, fleet records, procurement data, travel logs | Activity-based for Scope 1+2, spend-based for Scope 3 | +/- 10-20% |

**Scope 1 Calculation (simplified):**
- Natural gas: annual kWh from utility bills x emission factor (DEFRA/EPA)
- Vehicle fuel: annual litres/gallons from fuel receipts x emission factor
- Refrigerants: simplified estimation based on equipment type and age (if applicable)
- If no data: industry average tCO2e/employee for the NACE code

**Scope 2 Calculation:**
- Electricity: annual kWh from utility bills x grid emission factor (location-based)
- If renewable tariff: market-based factor applied
- If no data: industry average kWh/m2 x floor area x grid factor

**Scope 3 Calculation (simplified, focused on material categories):**

| Category | SME Relevance | Estimation Method |
|----------|--------------|-------------------|
| Cat 1: Purchased goods/services | High (typically 40-60% of Scope 3) | Annual procurement spend x EEIO emission factors by spend category |
| Cat 2: Capital goods | Low-Medium | Annual CapEx spend x EEIO factors (if significant) |
| Cat 3: Fuel & energy | Auto-calculated | Derived from Scope 1+2 data (WTT and T&D factors) |
| Cat 5: Waste | Low | Estimated from headcount x waste-per-employee x waste-type factors |
| Cat 6: Business travel | Medium (service SMEs) | Travel spend x spend-based factors (by mode: air, rail, car) |
| Cat 7: Employee commuting | Medium | Headcount x average commute distance x mode split x emission factors |
| Cat 8: Upstream leased assets | If applicable | Floor area x energy intensity (if leased premises not in Scope 1+2) |
| Cat 9-15 | Low for most SMEs | Excluded with documented justification per GHG Protocol |

**Key Models:**
- `SMEBaselineInput` - Company profile (NACE code, employees, revenue, region), utility bills, fuel data, spend categories, floor area, commute data
- `SMEBaselineResult` - Total CO2e by scope, category breakdown, data quality tier, confidence interval, industry comparison, visual dashboard data
- `DataQualityTier` - Bronze/Silver/Gold classification with per-source quality flags
- `IndustryComparison` - Comparison to NACE sector average tCO2e/employee and tCO2e/revenue

#### 3.3.2 Engine 2: Simplified Target Engine

**Purpose:** Set SME-appropriate targets aligned with SBTi SME Pathway and SME Climate Hub.

**SBTi SME Pathway Features:**
- Immediate validation (no queuing; SBTi SME targets are auto-validated)
- Near-term target: halve Scope 1+2 emissions by 2030 from a recent baseline (no older than 2 years)
- Scope 3: measure and reduce (no formal Scope 3 target required for SMEs)
- No SDA pathway required (ACA-equivalent 50% by 2030 is the default)
- Annual linear reduction rate: approximately 7-8% per year (to reach 50% by 2030 from a 2023/2024 baseline)

**Target Types Supported:**

| Target Type | Scope | Methodology | Ambition |
|-------------|-------|-------------|----------|
| SBTi SME Near-Term | Scope 1+2 | Absolute contraction, 50% by 2030 | 1.5C-aligned |
| SME Climate Hub 1-2-3 | All scopes | Halve by 2030, net zero by 2050 | 1.5C-aligned |
| Custom Reduction | Configurable | Absolute or intensity, custom % | User-defined (30-50% by 2030 range) |
| Sector Benchmark | Scope 1+2 | Match sector median/leader intensity | Peer-aligned |

**Key Models:**
- `SMETargetInput` - Baseline result, sector, ambition preference, certification pathway
- `SMETargetResult` - Target definition, annual milestones, SBTi SME compliance check, estimated effort level
- `AnnualMilestone` - Year, target emissions, required annual reduction, cumulative reduction %
- `TargetFeasibility` - Assessment of target achievability given identified quick wins and actions

#### 3.3.3 Engine 3: Quick Wins Engine

**Purpose:** Identify and prioritize high-impact, low-cost decarbonization actions for SMEs.

**Quick Wins Database (500+ actions):**

| Category | Example Actions | Typical Payback | Difficulty |
|----------|----------------|-----------------|-----------|
| Lighting | LED retrofit (whole building) | 6-18 months | DIY/Contractor |
| Lighting | Motion sensors and daylight dimming | 12-24 months | Contractor |
| HVAC | Smart thermostat installation | 3-12 months | DIY |
| HVAC | Boiler upgrade (gas to heat pump) | 3-7 years | Specialist |
| HVAC | Draught-proofing and insulation | 12-36 months | Contractor |
| Energy | Switch to renewable electricity tariff | Immediate (0 months) | DIY |
| Energy | Solar PV installation (rooftop) | 5-10 years | Specialist |
| Energy | Voltage optimization | 18-36 months | Specialist |
| Transport | Fleet electrification (EV/e-van) | 3-5 years | DIY/Specialist |
| Transport | Route optimization software | 6-12 months | DIY |
| Transport | Cycle-to-work scheme | Immediate | DIY |
| Transport | Remote/hybrid work policy | Immediate | DIY |
| IT | Server virtualization / cloud migration | 12-24 months | Specialist |
| IT | Equipment power management | 1-6 months | DIY |
| IT | Green web hosting | Immediate | DIY |
| Waste | Waste segregation and recycling program | 3-12 months | DIY |
| Waste | Food waste reduction (hospitality/retail) | 1-6 months | DIY |
| Waste | Packaging optimization | 6-18 months | DIY/Contractor |
| Water | Low-flow fixtures | 6-18 months | Contractor |
| Water | Rainwater harvesting | 3-7 years | Specialist |
| Procurement | Sustainable procurement policy | Immediate | DIY |
| Procurement | Local sourcing preference | Immediate | DIY |
| Procurement | Supplier sustainability questionnaires | 3-12 months | DIY |
| Office | Paperless operations | 1-6 months | DIY |
| Office | Green cleaning products | Immediate | DIY |
| Manufacturing | Compressed air leak detection | 1-6 months | Contractor |
| Manufacturing | Motor efficiency upgrade (IE4/IE5) | 2-4 years | Specialist |
| Manufacturing | Process heat recovery | 2-5 years | Specialist |
| Retail | Refrigeration door retrofitting | 12-36 months | Specialist |
| Retail | POS system energy optimization | 6-18 months | DIY |

**Action Attributes:**
- `action_id`: Unique identifier
- `category`: Energy, Transport, IT, Waste, Water, Procurement, Office, Manufacturing, Retail
- `name`: Action name
- `description`: Plain-language description
- `cost_range_min/max`: Upfront cost range (currency)
- `annual_savings_min/max`: Annual cost savings range
- `payback_months`: Expected payback period
- `co2e_reduction_pct`: Estimated % emission reduction for applicable scope
- `co2e_reduction_tonnes`: Estimated absolute reduction (tCO2e/year)
- `difficulty`: DIY / Contractor / Specialist
- `applicable_sectors`: List of NACE codes
- `applicable_sizes`: micro / small / medium
- `landlord_consent_needed`: Boolean (for leased premises)
- `grant_eligible`: Boolean + grant program references
- `co_benefits`: List (cost saving, employee wellbeing, customer appeal, compliance)

**Key Models:**
- `QuickWinsInput` - SME profile, baseline result, sector, size, premises type (owned/leased), budget constraints
- `QuickWinsResult` - Ranked action list, total potential reduction (tCO2e), total potential savings, implementation roadmap
- `QuickWinAction` - Single action with all attributes, ROI score, priority rank

#### 3.3.4 Engine 4: Action Prioritization Engine

**Purpose:** Rank and sequence all identified actions by composite score.

**Composite Scoring Formula:**
```
Score = (ROI_norm * 0.30) + (Payback_norm * 0.25) + (Ease_norm * 0.20)
      + (CO2e_norm * 0.15) + (CoBenefit_norm * 0.10)
```

Where:
- `ROI_norm`: Normalized ROI (0-1), higher = better return
- `Payback_norm`: Normalized payback (0-1), shorter = higher score; 0-6 months = 1.0, 6-12 months = 0.8, 12-24 months = 0.6, 24-36 months = 0.4, 36-60 months = 0.2, >60 months = 0.1
- `Ease_norm`: Difficulty-adjusted ease (DIY = 1.0, Contractor = 0.6, Specialist = 0.3); reduced by 0.2 if landlord consent needed
- `CO2e_norm`: Normalized abatement potential (0-1), higher reduction = higher score
- `CoBenefit_norm`: Number of co-benefits / max co-benefits, normalized (0-1)

**SME-Specific Constraints:**
- Maximum upfront cost filter (set by SME budget)
- Staff availability filter (DIY actions only if no maintenance staff)
- Premises type filter (exclude landlord-consent actions for tenants unless pre-approved)
- Sector relevance filter (exclude manufacturing actions for service SMEs)
- Regulatory filter (exclude actions requiring permits the SME cannot obtain)

**Output:** Top 5-10 prioritized actions with implementation sequence, resource plan, and cumulative emission reduction trajectory.

**Key Models:**
- `PrioritizationInput` - All identified actions, SME constraints (budget, staff, premises), weighting preferences
- `PrioritizationResult` - Ranked action list with composite scores, implementation sequence, cumulative impact chart data
- `ConstraintFilter` - Applied constraints with filtered/included action counts

#### 3.3.5 Engine 5: SME Progress Tracker

**Purpose:** Simple annual tracking with SME-appropriate KPIs.

**KPI Dashboard (8 core metrics):**

| # | KPI | Unit | Frequency | Data Source |
|---|-----|------|-----------|-------------|
| 1 | Total emissions | tCO2e | Annual | Baseline/annual recalculation |
| 2 | Change vs. baseline | % | Annual | Derived |
| 3 | Change vs. prior year | % | Annual | Derived |
| 4 | Emissions per employee | tCO2e/FTE | Annual | Emissions / headcount |
| 5 | Emissions per revenue | tCO2e/$M revenue | Annual | Emissions / revenue |
| 6 | Energy consumption | kWh | Annual | Utility bills |
| 7 | Renewable energy share | % | Annual | Tariff type / on-site generation |
| 8 | Actions completed | count (of planned) | Annual | User-reported action status |

**Progress Status:**
- **On Track (GREEN):** Cumulative reductions at or ahead of target pathway
- **Close (AMBER):** Within 5 percentage points of target pathway
- **Behind (RED):** More than 5 percentage points behind target pathway
- **No Data (GREY):** Annual data not yet submitted

**Corrective Action Triggers:**
- Off-track status triggers recommendation of additional quick wins
- Energy consumption increase triggers energy audit recommendation
- Low action completion triggers barrier assessment
- Intensity increase (per-employee or per-revenue) despite absolute decrease triggers growth-adjusted analysis

**Key Models:**
- `ProgressInput` - Current year data (same minimal inputs as baseline), prior year data, target pathway
- `ProgressResult` - KPI values, status (RAG), trajectory chart data, corrective recommendations
- `KPIDataPoint` - Single KPI with current value, prior year value, baseline value, target value, status

#### 3.3.6 Engine 6: Cost-Benefit Engine

**Purpose:** Calculate financial metrics for each action to support SME investment decisions.

**Financial Metrics Calculated:**

| Metric | Description | Presentation |
|--------|-------------|-------------|
| Simple Payback | Upfront cost / annual savings | Months |
| Discounted Payback | Time for cumulative discounted savings to exceed cost | Months |
| NPV (5-year) | Net present value over 5-year horizon | Currency |
| NPV (10-year) | Net present value over 10-year horizon | Currency |
| IRR | Internal rate of return | % |
| Annual Savings | Net annual operating cost reduction | Currency/year |
| Total Cost of Ownership | Upfront + ongoing costs over equipment life | Currency |
| Net-of-Grant Cost | Upfront cost minus applicable grant/subsidy | Currency |
| Carbon Price Impact | Value of avoided emissions at shadow carbon price | Currency/year |

**SME-Specific Financial Parameters:**
- Discount rate: 8-12% (higher than corporate, reflecting SME cost of capital)
- Equipment life: per-action database (LED: 10yr, heat pump: 15yr, solar PV: 25yr)
- Energy price escalation: 3-5% per year (adjustable by country)
- Carbon price escalation: aligned with country carbon pricing trajectory (or shadow price)
- Maintenance cost assumptions: per-action database
- Tax treatment: per-country capital allowances and deductions

**Key Models:**
- `CostBenefitInput` - Action details, SME financial profile (discount rate, tax rate), energy prices, grant availability
- `CostBenefitResult` - All financial metrics per action, summary table, decision chart (NPV vs. payback bubble chart)
- `FinancialMetrics` - Single action financial analysis with all metrics
- `GrantImpact` - Before/after grant financial comparison

#### 3.3.7 Engine 7: Grant Finder Engine

**Purpose:** Match SMEs to available grants, subsidies, tax incentives, and green loan programs.

**Grant Database Coverage (10,000+ programs):**

| Region | Key Programs | Coverage |
|--------|-------------|----------|
| EU | Horizon Europe SME Instrument, LIFE Programme, Cohesion Fund Green, EIC Accelerator, EU ETS Innovation Fund (small-scale), national programs (per 27 member states) | 3,000+ programs |
| UK | Energy Efficiency Grant (SMEs), Boiler Upgrade Scheme, Enhanced Capital Allowances, Innovate UK Smart Grants, Local Authority grants, Workplace Charging Scheme | 500+ programs |
| US | DOE SBIR/STTR (clean energy), IRA tax credits (30C, 45L, 179D), SBA Green Loans, state-level PACE financing, utility rebate programs, USDA REAP | 2,500+ programs |
| Germany | KfW Energy-Efficient Construction/Renovation, BAFA Energy Audit subsidy, Federal/State environment programs | 400+ programs |
| France | ADEME (Tremplin, Diag Eco-Flux, PRO-SMEn), Certificats d'Economie d'Energie, regional programs | 300+ programs |
| Japan | METI energy efficiency subsidies, local government programs, J-Credit scheme | 200+ programs |
| Australia | Clean Energy Finance Corporation, ARENA, state-level energy saver programs, Climate Active SME | 200+ programs |
| Canada | NRCan programs, CIB Green Infrastructure, provincial programs | 200+ programs |
| Other | Country-specific programs across 50+ additional countries | 2,700+ programs |

**Grant Matching Algorithm:**
1. Filter by country and region
2. Filter by sector (NACE/SIC code)
3. Filter by company size (employees, revenue thresholds)
4. Filter by eligible actions/technologies
5. Filter by application status (open/closed/upcoming)
6. Score match quality (0-100) based on eligibility fit
7. Sort by match score, then by deadline proximity
8. Flag high-probability matches (>80 score) for immediate action

**Key Models:**
- `GrantFinderInput` - SME profile (country, region, sector, size, planned actions), search preferences
- `GrantFinderResult` - Matched grants sorted by relevance, total potential funding, application timeline
- `GrantMatch` - Single grant with program name, funder, amount range, match score, eligibility status, deadline, application guidance
- `FundingCalendar` - Timeline of upcoming deadlines and application windows

#### 3.3.8 Engine 8: Certification Readiness Engine

**Purpose:** Assess readiness for SME-relevant climate certifications.

**Certifications Assessed:**

| # | Certification | Criteria Count | Typical Cost (SME) | Time to Achieve |
|---|--------------|---------------|--------------------|-----------------|
| 1 | SME Climate Hub | 5 | Free | 1-3 months |
| 2 | B Corp (Climate) | 12 (climate subset) | $1,000-$5,000/yr | 6-18 months |
| 3 | ISO 14001 | 20 (SME-relevant) | $5,000-$15,000 | 6-12 months |
| 4 | Carbon Trust Standard | 8 | $3,000-$8,000 | 3-6 months |
| 5 | Climate Active (AU) | 10 | $2,000-$10,000 | 3-9 months |
| 6 | Carbon Neutral (PAS 2060/ISO 14068-1) | 15 | $5,000-$20,000 | 6-12 months |

**SME Climate Hub Assessment (5 criteria):**

| Criterion | Requirement | Evidence |
|-----------|------------|---------|
| Pledge | Commit to halve emissions by 2030 and reach net zero by 2050 | Public pledge statement |
| Measure | Measure Scope 1+2 emissions (Scope 3 encouraged) | GHG baseline report |
| Reduce | Take immediate action to reduce emissions | Action plan + evidence of implementation |
| Report | Report progress annually | Annual progress report |
| Offset | Offset residual emissions (optional, but encouraged) | Credit retirement records (if applicable) |

**B Corp Climate Collective Assessment:**
- Climate-specific questions from B Impact Assessment (BIA)
- Environmental management practices scoring
- GHG measurement and reduction evidence
- Climate governance and targets
- Supply chain environmental engagement

**ISO 14001 Readiness (SME-simplified):**
- Environmental policy document
- Significant aspects and impacts register
- Legal and other requirements register
- Objectives and targets with programs
- Operational controls for significant aspects
- Emergency preparedness
- Monitoring and measurement procedures
- Internal audit program
- Management review records

**Key Models:**
- `CertificationInput` - SME profile, current sustainability practices, existing documentation, target certifications
- `CertificationResult` - Per-certification readiness score, gap list, remediation plan, estimated timeline, estimated cost
- `CertificationAssessment` - Single certification with per-criterion pass/fail/partial and evidence gaps
- `CertificationRoadmap` - Recommended certification sequence with dependencies and timeline

### 3.4 Workflows

| # | Workflow | Phases | Purpose |
|---|----------|--------|---------|
| 1 | `sme_onboarding_workflow.py` | 4: CompanyProfile -> DataCollection -> BaselineCalc -> TargetSetting | Guided onboarding from company profile creation through baseline calculation to target commitment. Phase 1 collects NACE code, headcount, revenue, region, premises type. Phase 2 guides utility bill upload and spend categorization. Phase 3 runs sme_baseline_engine. Phase 4 runs simplified_target_engine with certification pathway selection. |
| 2 | `quick_assessment_workflow.py` | 3: RapidBaseline -> QuickWinsIdentification -> ActionPlan | Rapid assessment workflow for SMEs wanting quick results. Phase 1 runs Bronze-tier baseline (revenue + headcount + sector only). Phase 2 identifies top 10 quick wins via quick_wins_engine. Phase 3 produces a simplified action plan with cost-benefit summary. Designed to complete in <1 hour. |
| 3 | `action_planning_workflow.py` | 5: Prioritization -> CostBenefit -> GrantSearch -> Timeline -> Approval | Detailed action planning workflow. Phase 1 runs action_prioritization_engine. Phase 2 runs cost_benefit_engine for top actions. Phase 3 runs grant_finder_engine to identify funding. Phase 4 generates implementation timeline. Phase 5 produces approval-ready business case document for owner/board. |
| 4 | `implementation_support_workflow.py` | 4: VendorSelection -> Monitoring -> Verification -> Adjustment | Implementation support workflow. Phase 1 provides vendor/contractor selection guidance for specialist actions. Phase 2 sets up monitoring checkpoints. Phase 3 verifies implemented actions are delivering expected savings. Phase 4 adjusts plan if actions underperform or new opportunities arise. |
| 5 | `progress_review_workflow.py` | 3: AnnualDataCollection -> Tracking -> Reporting | Annual progress review workflow. Phase 1 guides annual data collection (same minimal inputs as baseline). Phase 2 runs sme_progress_tracker with year-over-year comparison. Phase 3 generates annual progress report and certification-specific disclosures (SME Climate Hub, B Corp, CDP). |
| 6 | `certification_pathway_workflow.py` | 5: ReadinessAssessment -> GapClosure -> Documentation -> Submission -> Verification | Certification achievement workflow. Phase 1 runs certification_readiness_engine. Phase 2 guides gap closure with specific actions. Phase 3 prepares certification documentation. Phase 4 supports submission to certification body. Phase 5 manages verification/audit process (if applicable). |

### 3.5 Templates

| # | Template | Formats | Purpose |
|---|----------|---------|---------|
| 1 | `sme_baseline_report.py` | MD, HTML, JSON | 1-2 page visual baseline report with pie charts (scope split), bar charts (category breakdown), industry comparison, and data quality indicators. Designed for non-technical audiences. Plain-language explanations of what Scope 1/2/3 mean for the specific business. |
| 2 | `simplified_action_plan.py` | MD, HTML, JSON | Practical action plan focused on top 5-10 prioritized actions. Each action has a 1-paragraph description, cost/savings summary, payback period, implementation steps, and responsible person/role. Includes total projected emission reduction and cost savings. |
| 3 | `quick_wins_roadmap.py` | MD, HTML, JSON | Visual 6-24 month timeline showing quick wins organized by quarter. Gantt-style chart with actions color-coded by category (energy, transport, waste, etc.). Each action shows cost, savings, and CO2e impact. Designed for wall display or team briefing. |
| 4 | `cost_benefit_analysis.py` | MD, HTML, JSON | Financial analysis report with NPV, IRR, and payback tables for each action. Includes bubble chart (NPV vs. payback with bubble size = CO2e reduction), cumulative savings curve, and grant-adjusted investment summary. Designed for owner/finance decision-making. |
| 5 | `grant_application_support.py` | MD, HTML, JSON | Grant application support package with matched grant summaries, eligibility confirmation, pre-populated application content (company description, project description, environmental impact), and submission checklist. Templates for the 5 most commonly-used grant programs per country. |
| 6 | `progress_dashboard.py` | MD, HTML, JSON | Annual KPI tracking dashboard with sparkline charts for each metric, year-over-year comparison, target pathway visualization, and RAG status indicators. Mobile-friendly responsive layout. Includes simple explanations of what each metric means and what the SME should do if off-track. |
| 7 | `certification_submission.py` | MD, HTML, JSON | Certification-ready submission documents for SME Climate Hub (1-2-3 commitment letter and evidence), B Corp BIA climate section (pre-populated responses), and ISO 14001 documentation package (environmental policy, aspects register, objectives). Format adapts to selected certification. |
| 8 | `stakeholder_communication.py` | MD, HTML, JSON | Communication templates for three audiences: (a) Board/owner briefing: 1-page executive summary with financial case, (b) Employee engagement: team announcement with individual actions, (c) Customer messaging: sustainability commitment statement for website/marketing. Plain language, jargon-free. |

### 3.6 Integrations

| # | Integration | Purpose |
|---|-------------|---------|
| 1 | `pack_orchestrator.py` | 6-phase DAG pipeline with retry, provenance, conditional grant/certification phases. SME-optimized: shorter pipeline than enterprise packs, fewer dependencies, faster execution. |
| 2 | `mrv_bridge.py` | Routes to relevant MRV agents for emission calculations. SME-optimized: uses only MRV-001 (Stationary Combustion), MRV-003 (Mobile Combustion), MRV-009 (Scope 2 Location-Based), MRV-010 (Scope 2 Market-Based), MRV-014 (Cat 1), MRV-019 (Cat 6), MRV-020 (Cat 7) by default. Additional agents activated only if SME profile indicates relevance. |
| 3 | `data_bridge.py` | Bridges to DATA agents for data intake. SME-optimized: primarily uses DATA-002 (Excel/CSV Normalizer for utility bills), DATA-009 (Spend Data Categorizer for procurement), and DATA-010 (Data Quality Profiler). |
| 4 | `grant_database_bridge.py` | Bridges to external grant databases: EU funding portal API, Grants.gov API (US), UK Government grants API, country-specific grant aggregators. Maintains local cache of 10,000+ programs with weekly refresh. Provides structured search, eligibility checking, and deadline monitoring. |
| 5 | `sme_climate_hub_bridge.py` | Bridges to UN SME Climate Hub platform for commitment registration, progress reporting, and community features. Supports pledge submission, annual disclosure, and peer network connection. Maps PACK-026 outputs to SME Climate Hub reporting format. |
| 6 | `accounting_software_bridge.py` | Bridges to common SME accounting platforms for automated spend data extraction: QuickBooks (API), Xero (API), Sage (API), FreshBooks (API), and CSV export format for other platforms. Extracts spend-by-category data for spend-based Scope 3 calculation. Maps accounting categories to EEIO emission factor categories. |
| 7 | `certification_body_bridge.py` | Bridges to certification body APIs and portals: B Lab B Impact Assessment API, ISO certification body lookup, Carbon Trust Standard portal, Climate Active registry. Supports application submission, status tracking, and certification evidence upload. |
| 8 | `peer_network_bridge.py` | Bridges to anonymized peer benchmarking data sources: industry associations, local chambers of commerce, government SME surveys, and GreenLang platform aggregate data. Provides sector/size/region benchmarks while maintaining strict anonymization (k-anonymity >= 10 for any comparison group). |
| 9 | `supplier_engagement_bridge.py` | Simplified Scope 3 supplier data collection for SMEs. Provides lightweight supplier questionnaire (10 questions, 15-minute completion), supplier emission estimation from public data (where available), and aggregated supplier engagement scoring. Designed for SMEs with 10-100 key suppliers. |
| 10 | `setup_wizard.py` | 4-step guided configuration wizard: (1) Company profile (name, sector, size, region), (2) Data availability assessment (what bills/records available), (3) Ambition level selection (quick assessment vs. full baseline), (4) Certification interest (SME Climate Hub, B Corp, ISO 14001). Produces initial configuration for all engines. |

### 3.7 Presets

| # | Preset | Target | Key Characteristics |
|---|--------|--------|---------------------|
| 1 | `micro_business.yaml` | 1-9 employees | Minimal data inputs (revenue + headcount + sector only for Bronze tier). Industry average emission factors. Top 5 quick wins only. Simplified dashboard (5 KPIs). Basic action plan. SME Climate Hub certification pathway. Budget ceiling: $5,000. No Scope 3 deep dive. |
| 2 | `small_business.yaml` | 10-49 employees | Moderate data collection (utility bills + spend categories). Mix of activity data (energy) and spend-based (procurement). Top 10 quick wins. Full dashboard (8 KPIs). Standard action plan with cost-benefit. SME Climate Hub + B Corp pathway. Budget ceiling: $15,000. Simplified Scope 3 (Cat 1, 6, 7). |
| 3 | `medium_business.yaml` | 50-249 employees | More detailed data collection (utility bills + fleet data + procurement + travel logs). Mostly activity-based Scope 1+2, spend-based Scope 3. Full quick wins database. Full dashboard + benchmarking. Detailed action plan with grant matching. ISO 14001 + SBTi SME pathway. Budget ceiling: $50,000. Comprehensive Scope 3 (all material categories). |
| 4 | `service_sme.yaml` | Office-based services (any size) | Scope 2 dominant (electricity), Scope 3 dominant (Cat 1 procurement, Cat 6 travel, Cat 7 commuting). Quick wins focus: renewable tariff, LED, remote work, travel policy. Low Scope 1 (no fleet, no process). Office-specific actions only. |
| 5 | `manufacturing_sme.yaml` | Manufacturing/production (any size) | Scope 1 significant (natural gas, process heat). Scope 2 significant (electricity for machinery). Manufacturing-specific actions: compressed air, motor efficiency, heat recovery, process optimization. Fleet considerations if distribution. |
| 6 | `retail_sme.yaml` | Retail/hospitality (any size) | Scope 2 dominant (lighting, refrigeration, HVAC for customer space). Packaging and waste focus. Refrigeration-specific actions. Customer-facing sustainability messaging templates. Scope 3 Cat 1 (purchased goods for resale) dominant. |

---

## 4. Agent Dependencies

### 4.1 MRV Agents (subset of 30)

SME-relevant MRV agents via `mrv_bridge.py`:
- **Scope 1 (2 primary):** MRV-001 (Stationary Combustion -- gas heating), MRV-003 (Mobile Combustion -- fleet vehicles)
- **Scope 1 (2 optional):** MRV-002 (Refrigerants -- for retail/food service), MRV-007 (Waste Treatment -- for manufacturing)
- **Scope 2 (2):** MRV-009 (Location-Based), MRV-010 (Market-Based)
- **Scope 3 (3 primary):** MRV-014 (Cat 1 -- Purchased Goods), MRV-019 (Cat 6 -- Business Travel), MRV-020 (Cat 7 -- Employee Commuting)
- **Scope 3 (2 optional):** MRV-018 (Cat 5 -- Waste Generated), MRV-016 (Cat 3 -- Fuel & Energy Activities, auto-calculated)
- **Cross-cutting (2):** MRV-029 (Scope 3 Category Mapper), MRV-030 (Audit Trail)

### 4.2 Decarbonization Agents (subset of 21)

SME-relevant DECARB-X agents via `decarb_bridge.py` (limited integration):
- DECARB-X-001: Abatement Options Library (filtered to SME-appropriate actions)
- DECARB-X-005: Investment Prioritization (with SME financial parameters)
- DECARB-X-010: Renewable Energy Planner (simplified for SME scale)
- DECARB-X-013: Energy Efficiency Identifier (core SME use case)
- DECARB-X-018: Progress Monitoring Agent (simplified annual tracking)
- DECARB-X-020: Cost-Benefit Analyzer (SME discount rates and horizons)

### 4.3 Application Dependencies

- GL-GHG-APP: GHG inventory management (simplified SME mode)
- GL-SBTi-APP: SBTi SME Pathway target validation

### 4.4 Data Agents (subset of 20)

SME-relevant DATA agents via `data_bridge.py`:
- DATA-002: Excel/CSV Normalizer (utility bill parsing)
- DATA-009: Spend Data Categorizer (procurement spend classification)
- DATA-010: Data Quality Profiler (data completeness assessment)
- DATA-008: Supplier Questionnaire Processor (simplified supplier engagement)

### 4.5 Foundation Agents (10)

All 10 AGENT-FOUND agents for orchestration, schema, units, audit, etc.

### 4.6 Optional Pack Dependencies

- PACK-021 Net Zero Starter Pack: Full baseline upgrade path for growing SMEs via integration if present
- PACK-022 Net Zero Acceleration Pack: Advanced capabilities for medium SMEs approaching 250-employee threshold
- PACK-023 SBTi Alignment Pack: Full SBTi lifecycle if SME graduates to formal SBTi corporate track
- PACK-024 Carbon Neutral Pack: Carbon neutrality claim support if SME pursues PAS 2060/ISO 14068-1

---

## 5. SME-Specific Considerations

### 5.1 Data Collection Burden Minimization

PACK-026 is designed around the principle that **less data collected well is better than more data collected poorly**. The pack never asks for data that the SME does not already have. The minimum viable baseline requires only:

| Data Item | Source | Collection Time |
|-----------|--------|----------------|
| Company name, sector (NACE), employee count | Known by owner | 2 minutes |
| Annual revenue | Tax return / accounts | 2 minutes |
| Annual electricity bill (kWh + cost) | Utility bill | 5 minutes |
| Annual gas bill (kWh + cost) | Utility bill | 5 minutes |
| Annual fuel spend (vehicles) | Fuel receipts / bank statement | 10 minutes |
| Annual procurement spend (top 5 categories) | Bank statement / accounts | 20 minutes |
| Business travel spend (air, rail, car) | Bank statement / expense reports | 15 minutes |
| Employee count and average commute estimate | HR records / estimate | 10 minutes |
| Floor area (m2) | Lease agreement / estimate | 5 minutes |
| Premises type (owned/leased) | Known | 1 minute |

**Total: approximately 75 minutes** for a Silver-tier baseline.

### 5.2 Landlord/Tenant Considerations

Many SMEs lease their premises, which limits their ability to implement building-level actions (insulation, solar PV, boiler replacement). The pack handles this by:
- Flagging actions that require landlord consent
- Providing landlord engagement templates (letters requesting permission for energy improvements)
- Identifying tenant-controlled actions (lighting, equipment, behavior) vs. landlord-controlled actions (building fabric, HVAC systems, roof access)
- Supporting green lease clause recommendations for lease renewals

### 5.3 Multi-Site SMEs

Some SMEs operate across multiple locations (e.g., retail chains with 5-20 stores, service companies with regional offices). The pack supports:
- Per-site data collection with rollup to organization level
- Site comparison dashboards (identify best/worst performing sites)
- Site-specific action plans (different sites may need different interventions)
- Consolidated reporting for certification purposes

### 5.4 Seasonal Business Adjustments

Many SMEs have seasonal revenue and emission patterns (hospitality, agriculture, retail). The pack:
- Accepts annual data (not monthly) to reduce collection burden
- Normalizes emissions by revenue/output for fair year-over-year comparison
- Adjusts baselines for known seasonal patterns
- Flags unusual year-over-year changes for investigation (e.g., COVID impact years)

### 5.5 Growth-Adjusted Tracking

SMEs frequently grow (or contract) significantly year-over-year, making absolute emission tracking misleading. The pack:
- Tracks both absolute (tCO2e) and intensity (tCO2e/FTE, tCO2e/$M revenue) metrics
- Highlights when absolute increases are driven by growth (intensity stable or declining)
- Adjusts target pathways for organic growth using the SBTi base year recalculation approach (simplified)
- Communicates progress in growth-adjusted terms to stakeholders

### 5.6 Supply Chain Mandate Support

Increasingly, SMEs are required by large corporate customers to report emissions as part of Scope 3 supply chain programs (e.g., CDP Supply Chain, Walmart Project Gigaton, Apple supplier program). The pack:
- Generates supplier-ready emission reports in standard formats
- Maps outputs to CDP Supply Chain SME questionnaire
- Provides "carbon credentials" documentation for customer RFPs
- Tracks multiple customer reporting requirements simultaneously

---

## 6. Performance Targets

| Metric | Target |
|--------|--------|
| Bronze baseline (revenue + headcount only) | <2 minutes |
| Silver baseline (utility bills + spend data) | <5 minutes |
| Gold baseline (detailed activity data) | <15 minutes |
| Quick wins identification (500+ actions filtered) | <2 minutes |
| Action prioritization (top 10 ranking) | <1 minute |
| Cost-benefit analysis (10 actions) | <3 minutes |
| Grant matching (10,000+ programs searched) | <5 minutes |
| Certification readiness (6 certifications) | <5 minutes |
| Full SME onboarding workflow (end-to-end) | <30 minutes (excluding data collection) |
| Quick assessment workflow (Bronze tier) | <10 minutes |
| Annual progress review | <15 minutes (excluding data collection) |
| Memory ceiling | 1024 MB |
| Cache hit target | 80% |
| Max sites per SME | 50 |
| Max employees modeled | 250 |
| Max suppliers (simplified engagement) | 500 |
| Mobile dashboard load time | <3 seconds |

---

## 7. Security Requirements

- JWT RS256 authentication
- RBAC with 5 roles: `sme_owner` (full access), `sme_manager` (read/write data, run engines), `sme_viewer` (read-only dashboards), `advisor` (read access + report generation for external accountants/advisors), `admin`
- AES-256-GCM encryption at rest for all emission and financial data
- TLS 1.3 for data in transit
- SHA-256 provenance hashing on all calculation outputs
- Full audit trail per SEC-005
- Accounting software API credentials encrypted via Vault (SEC-006)
- GDPR compliance for employee commuting data (anonymization of individual commute patterns)
- Grant application data access controls (sensitive financial data separation)

---

## 8. Database Migrations

Inherits platform migrations V001-V128. Pack-specific migrations:

| Migration | Table | Purpose |
|-----------|-------|---------|
| V083-PACK026-001 | `sme_profiles` | SME company profiles with NACE code, size tier, region, premises type |
| V083-PACK026-002 | `sme_baselines` | GHG baseline records with data quality tier, confidence intervals, scope breakdown |
| V083-PACK026-003 | `sme_targets` | Target records (SBTi SME, SME Climate Hub, custom) with annual milestones |
| V083-PACK026-004 | `sme_actions` | Identified and planned actions with prioritization scores, implementation status |
| V083-PACK026-005 | `sme_cost_benefit` | Cost-benefit analysis records with NPV, IRR, payback per action |
| V083-PACK026-006 | `sme_progress` | Annual progress tracking records with KPI values and RAG status |
| V083-PACK026-007 | `sme_grants` | Grant matches with eligibility status, application status, award outcome |
| V083-PACK026-008 | `sme_certifications` | Certification readiness assessments with per-criterion scores and gap lists |
| V083-PACK026-009 | `sme_quick_wins` | Quick wins database (500+ actions) with cost, savings, and applicability metadata |
| V083-PACK026-010 | `sme_peer_benchmarks` | Anonymized peer benchmark data by NACE code, size tier, and region |

---

## 9. File Structure

```
packs/net-zero/PACK-026-sme-net-zero/
  __init__.py
  pack.yaml
  config/
    __init__.py
    pack_config.py
    demo/
      __init__.py
      demo_config.yaml
    presets/
      __init__.py
      micro_business.yaml
      small_business.yaml
      medium_business.yaml
      service_sme.yaml
      manufacturing_sme.yaml
      retail_sme.yaml
  engines/
    __init__.py
    sme_baseline_engine.py
    simplified_target_engine.py
    quick_wins_engine.py
    action_prioritization_engine.py
    sme_progress_tracker.py
    cost_benefit_engine.py
    grant_finder_engine.py
    certification_readiness_engine.py
  workflows/
    __init__.py
    sme_onboarding_workflow.py
    quick_assessment_workflow.py
    action_planning_workflow.py
    implementation_support_workflow.py
    progress_review_workflow.py
    certification_pathway_workflow.py
  templates/
    __init__.py
    sme_baseline_report.py
    simplified_action_plan.py
    quick_wins_roadmap.py
    cost_benefit_analysis.py
    grant_application_support.py
    progress_dashboard.py
    certification_submission.py
    stakeholder_communication.py
  integrations/
    __init__.py
    pack_orchestrator.py
    mrv_bridge.py
    data_bridge.py
    grant_database_bridge.py
    sme_climate_hub_bridge.py
    accounting_software_bridge.py
    certification_body_bridge.py
    peer_network_bridge.py
    supplier_engagement_bridge.py
    setup_wizard.py
  data/
    __init__.py
    quick_wins_database.json
    eeio_emission_factors.json
    industry_benchmarks.json
    grant_programs.json
  tests/
    __init__.py
    conftest.py
    test_manifest.py
    test_config.py
    test_baseline_engine.py
    test_target_engine.py
    test_quick_wins_engine.py
    test_prioritization_engine.py
    test_progress_tracker.py
    test_cost_benefit_engine.py
    test_grant_finder_engine.py
    test_certification_engine.py
    test_workflows.py
    test_templates.py
    test_integrations.py
    test_presets.py
    test_quick_wins_database.py
    test_grant_database.py
    test_e2e.py
    test_orchestrator.py
```

---

## 10. Testing Requirements

| Test Type | Coverage Target | Scope |
|-----------|-----------------|-------|
| Unit Tests | >90% line coverage | All 8 engines, all config models, all presets, quick wins database |
| Workflow Tests | >85% | All 6 workflows with synthetic SME data (micro, small, medium) |
| Template Tests | 100% | All 8 templates in 3 formats (MD, HTML, JSON) |
| Integration Tests | >80% | All 10 integrations with mock agents, mock grant APIs, mock accounting APIs |
| E2E Tests | Core happy path | Full pipeline from onboarding to progress review for each size tier |
| Baseline Tests | 100% | All 3 data quality tiers (Bronze, Silver, Gold) with edge cases |
| Quick Wins Tests | 100% | Database completeness, filtering by sector/size/premises, ranking accuracy |
| Cost-Benefit Tests | >90% | NPV, IRR, payback calculations with known-value verification |
| Grant Finder Tests | >85% | Matching algorithm with synthetic grant databases across 5 countries |
| Certification Tests | 100% | All 6 certifications with pass/fail/partial scenarios |
| Preset Tests | 100% | All 6 presets with representative SME profiles |
| Mobile Responsiveness Tests | >80% | Dashboard templates render correctly at mobile breakpoints |
| Manifest Tests | 100% | pack.yaml validation, component counts, version |

**Test Count Target:** 700+ tests (50-70 per engine, 30-40 integration, 20-30 E2E, 50+ quick wins database, 30+ grant database)

---

## 11. Upgrade Path

PACK-026 is designed as the entry point for SMEs, with clear upgrade paths as organizations grow:

| Trigger | Current Pack | Upgrade To | What Changes |
|---------|-------------|-----------|-------------|
| SME exceeds 250 employees | PACK-026 | PACK-021 | Full GHG Protocol methodology, all 15 Scope 3 categories, MACC curves |
| SME wants formal SBTi corporate target | PACK-026 | PACK-023 | Full 42-criterion SBTi validation, SDA/FLAG pathways |
| SME wants multi-scenario planning | PACK-026 | PACK-022 | Monte Carlo scenarios, supplier engagement at scale, temperature scoring |
| SME wants carbon neutral claim | PACK-026 | PACK-024 | ISO 14068-1 compliance, credit portfolio management, verification |
| SME joins Race to Zero | PACK-026 | PACK-025 | Campaign lifecycle, Starting Line criteria, HLEG credibility |
| SME graduates to medium enterprise | PACK-026 (medium preset) | PACK-021 + PACK-023 | Full enterprise capabilities |

**Data Continuity:** All PACK-026 baseline data, targets, action history, and progress records are preserved and automatically migrated when upgrading to any other Net Zero pack. The baseline is recalculated with enhanced methodology (activity-based replacing spend-based where data quality improves).

---

## 12. Release Plan

| Phase | Deliverable | Timeline |
|-------|-------------|----------|
| Phase 1 | PRD Approval | 2026-03-18 |
| Phase 2 | Engine implementation (8 engines) | 2026-03-19 |
| Phase 3 | Workflow implementation (6 workflows) | 2026-03-19 |
| Phase 4 | Template implementation (8 templates) | 2026-03-19 |
| Phase 5 | Integration implementation (10 integrations) | 2026-03-19 |
| Phase 6 | Quick wins + grant database population | 2026-03-20 |
| Phase 7 | Test suite (700+ tests) | 2026-03-20 |
| Phase 8 | Documentation & Release | 2026-03-20 |

---

## 13. Appendix: SME Climate Hub 1-2-3 Commitment

### The 1-2-3 Pledge

The SME Climate Hub (smeclimatehub.org) is a UN-backed initiative within the Race to Zero campaign, designed specifically for SMEs. The commitment is simple:

1. **Halve** greenhouse gas emissions before 2030
2. **Achieve net zero** emissions before 2050
3. **Disclose** progress on a yearly basis

### Eligibility

- Any business with fewer than 500 employees (definition varies by country; PACK-026 focuses on <250)
- No minimum size -- sole traders and micro-businesses welcome
- No cost to join (free platform)
- No external verification required (self-declaration with reporting)

### Benefits of SME Climate Hub Membership

- Recognition as a Race to Zero participant
- Access to free tools and resources
- Peer network and community
- Customer/supply chain credibility
- Listed on public commitment platform
- Access to SME-specific climate action guides

### PACK-026 Alignment

PACK-026 produces all outputs required for SME Climate Hub participation:
- Pledge statement (via `stakeholder_communication.py` template)
- GHG measurement (via `sme_baseline_engine.py`)
- Reduction actions (via `quick_wins_engine.py` + `action_prioritization_engine.py`)
- Annual disclosure (via `sme_progress_tracker.py` + `progress_dashboard.py`)

---

## 14. Appendix: SBTi SME Pathway

### SBTi SME Target Setting Route (launched 2023)

The SBTi SME Route provides a streamlined pathway for small and medium-sized enterprises to set science-based targets:

| Feature | SBTi Corporate Route | SBTi SME Route |
|---------|---------------------|----------------|
| Eligibility | Any company | <500 employees (varies) |
| Target requirement | Near-term + Long-term | Near-term only |
| Scope 1+2 target | 4.2%/yr or SDA | Halve by 2030 |
| Scope 3 target | 67%+ coverage | Measure and reduce (no formal target required) |
| Pathway | ACA, SDA, or FLAG | Simplified absolute contraction |
| Validation | Queued review (weeks-months) | Immediate auto-validation |
| Cost | Varies by revenue | Free |
| Review cycle | 5-year revalidation | Annual self-assessment |

### PACK-026 SBTi SME Support

- Automatic eligibility assessment (employee count, revenue check)
- Near-term target calculation: 50% absolute reduction in Scope 1+2 from baseline by 2030
- Base year validation: must be within last 2 years
- Scope 3 guidance: measure Cat 1, 6, 7 (minimum); set informal reduction goals
- Auto-generated SBTi SME commitment letter
- Annual progress tracking aligned with SBTi reporting requirements

---

## 15. Appendix: Quick Wins Database Schema

### Database Structure

```json
{
  "action_id": "QW-001",
  "category": "Lighting",
  "name": "LED Lighting Retrofit",
  "description": "Replace all fluorescent and halogen lighting with LED equivalents...",
  "cost_range": {"min": 500, "max": 5000, "currency": "USD"},
  "annual_savings": {"min": 200, "max": 2000, "currency": "USD"},
  "payback_months": {"min": 6, "max": 36},
  "co2e_reduction": {"pct": 0.02, "tonnes_min": 0.5, "tonnes_max": 5.0},
  "difficulty": "contractor",
  "applicable_sectors": ["*"],
  "applicable_sizes": ["micro", "small", "medium"],
  "landlord_consent": false,
  "grant_eligible": true,
  "grant_programs": ["UK-EEG", "US-UTILITY-REBATE"],
  "co_benefits": ["cost_saving", "improved_lighting_quality", "reduced_maintenance"],
  "implementation_steps": [
    "Audit current lighting (count fixtures, note types)",
    "Get 2-3 quotes from LED suppliers/installers",
    "Check for utility rebates and local grants",
    "Schedule installation (typically 1-2 days for small office)",
    "Dispose of old fixtures responsibly (WEEE regulations)"
  ],
  "data_sources": ["Carbon Trust", "Energy Saving Trust", "US DOE"],
  "last_updated": "2026-03-01"
}
```

### Database Statistics

| Category | Action Count | Avg Payback (months) | Avg CO2e Reduction |
|----------|-------------|---------------------|-------------------|
| Lighting | 35 | 18 | 2-5% of Scope 2 |
| HVAC | 45 | 24 | 5-15% of Scope 1+2 |
| Energy | 60 | 36 | 10-30% of Scope 2 |
| Transport | 55 | 30 | 5-20% of Scope 1 |
| IT | 40 | 12 | 1-5% of Scope 2 |
| Waste | 30 | 6 | 1-3% of Scope 3 |
| Water | 25 | 24 | <1% (co-benefit focus) |
| Procurement | 50 | 0 (policy) | 5-15% of Scope 3 |
| Office | 35 | 3 | 1-3% of Scope 2 |
| Manufacturing | 65 | 30 | 10-30% of Scope 1+2 |
| Retail | 40 | 24 | 5-15% of Scope 2 |
| Hospitality | 20 | 18 | 5-10% of Scope 1+2 |
| **Total** | **500** | **20 (avg)** | **Varies** |

---

## 16. Appendix: Peer Benchmarking Methodology

### Benchmark Data Sources

- GreenLang platform aggregate data (anonymized, k-anonymity >= 10)
- Published industry emission intensity data (DEFRA sector factors, EPA benchmarks)
- SME Climate Hub reported data (aggregated, anonymized)
- Government SME energy surveys (per country)
- Industry association sustainability reports

### Benchmark Dimensions

| Dimension | Metric | Comparison Group |
|-----------|--------|-----------------|
| Emission intensity (per employee) | tCO2e/FTE | Same NACE code + same size tier |
| Emission intensity (per revenue) | tCO2e/$M revenue | Same NACE code + same size tier |
| Energy intensity | kWh/m2 or kWh/FTE | Same NACE code + same region |
| Renewable energy share | % of electricity from renewables | Same country + same size tier |
| Action adoption rate | % of applicable quick wins implemented | Same NACE code + same size tier |
| Reduction trajectory | Annual % change in emissions | Same NACE code |

### Benchmark Presentation

- Percentile ranking (e.g., "Your emission intensity places you in the 65th percentile for your sector")
- Quartile classification: Leader (top 25%), Above Average (25-50%), Below Average (50-75%), Laggard (bottom 25%)
- Gap-to-leader: specific tCO2e/FTE or kWh/m2 gap to reach sector leader performance
- Anonymized peer examples: "A similar-sized [sector] business achieved X% reduction by implementing [action]"

### Privacy and Anonymization

- All benchmarks computed on aggregated data (minimum 10 entities per comparison group)
- No individual entity data is ever exposed
- Comparison groups broadened (e.g., from 4-digit NACE to 2-digit) if group size < 10
- Regional benchmarks use country-level if sub-national groups too small
- GDPR-compliant data processing for all European SME data

---

## 17. Future Roadmap

- **PACK-026 v1.1: Enhanced Grant Intelligence** -- AI-powered grant matching with success probability scoring, application drafting assistance, and deadline management with automated reminders
- **PACK-026 v1.2: Supplier Cascade** -- Extended supplier engagement module enabling SMEs to cascade net-zero requirements to their own suppliers (Scope 3 Cat 1 reduction at source)
- **PACK-026 v1.3: Sector Deep Dives** -- Additional sector-specific presets: construction_sme, agriculture_sme, transport_sme, healthcare_sme, food_service_sme
- **PACK-027: SME Net Zero Network Pack** -- Multi-SME orchestration for industry associations, chambers of commerce, and supply chain programs managing hundreds of SMEs collectively
- **PACK-028: SME Climate Finance Pack** -- Green loan readiness assessment, ESG-linked lending preparation, climate risk assessment for SME lending, and investor-ready sustainability reports

---

*Document Version: 1.0.0 | Last Updated: 2026-03-18 | Status: Draft*
