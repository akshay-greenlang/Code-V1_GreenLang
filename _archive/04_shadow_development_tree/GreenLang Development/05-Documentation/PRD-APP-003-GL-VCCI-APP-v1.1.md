# PRD: APP-003 -- GL-VCCI-APP v1.1 Enhancement

## Document Info
| Field | Value |
|-------|-------|
| PRD ID | PRD-APP-003 |
| Application | GL-VCCI-APP v1.1 |
| Priority | P0 (Critical) |
| Version | 1.1.0 |
| Status | Approved |
| Author | GL-ProductManager + GL-RegulatoryIntelligence |
| Date | 2026-02-26 |
| Base | GL-VCCI-APP v1.0 (applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/) |

---

## 1. Overview

### 1.1 Purpose
Enhance GL-VCCI-APP from v1.0 (89/100 production-ready, 170K+ lines) to v1.1 by:
1. **Monte Carlo UI** -- Frontend visualization for uncertainty analysis (distributions, sensitivity tornado, confidence interval bands, scenario comparison)
2. **CDP Integration Enhancement** -- Full CDP Climate Change questionnaire editor, section completion tracking, auto-population improvement (90% -> 95%+), multi-year comparison
3. **Advanced Sensitivity Analysis** -- Sobol indices, Morris screening, tornado diagram data generation
4. **Compliance Dashboard** -- Multi-standard compliance scorecard (GHG Protocol, ESRS E1, CDP, IFRS S2, ISO 14083)
5. **Backend API Expansion** -- New endpoints for uncertainty details, sensitivity analysis, CDP questionnaire management

### 1.2 Current State (v1.0 Gap Analysis)

| Component | v1.0 State | v1.1 Target |
|-----------|-----------|-------------|
| Monte Carlo engine | Backend complete (622 lines, 10K iterations) | No change (backend stays) |
| Monte Carlo UI | Not implemented (no frontend components) | Full visualization suite (6 components) |
| Uncertainty display | Not shown on dashboard | CI bands on all charts, distribution charts |
| Sensitivity analysis | Basic Pearson correlation only | Sobol indices, Morris screening, tornado charts |
| CDP generator | Basic auto-population (57 lines, 90%) | Enhanced 95%+ auto-population, all sections |
| CDP UI | Report generation dialog only | Full questionnaire editor, section tracker |
| CDP validation | Not implemented | Section-by-section validation with gap analysis |
| Compliance dashboard | Not implemented | Multi-standard scorecard with coverage metrics |
| Settings persistence | Frontend placeholder only | Full backend persistence |
| Scenario comparison | Backend scenario modeling exists | UI with uncertainty bands and probability |

### 1.3 Technical Context
- **Frontend Stack**: React 18 + TypeScript + Material-UI 5 + Redux Toolkit + Recharts + Vite
- **Backend Stack**: FastAPI + Python 3.9+ + NumPy/SciPy + Pydantic
- **Existing Monte Carlo**: MonteCarloSimulator (622 lines), UncertaintyQuantifier (667 lines), UncertaintyEngine (145 lines)
- **Existing CDP**: CDPGenerator (57 lines), ReportingAgent with CDP route
- **Frontend Pages**: Dashboard, DataUpload, Reports, SupplierManagement, Settings (5 pages)
- **API Client**: Axios-based with JWT auth, error handling (235 lines)

---

## 2. Enhancement Areas

### 2.1 Monte Carlo UI Components (Frontend)

#### 2.1.1 UncertaintyDistribution Component
- Histogram of Monte Carlo simulation results (10,000 samples)
- Violin plot option for compact distribution view
- Overlaid kernel density estimation (KDE) curve
- Marked percentile lines (p5, p25, p50, p75, p95)
- Mean vs. median indicator
- Configurable bin count and color scheme
- Responsive resize with Recharts
- Export distribution data as CSV

#### 2.1.2 SensitivityTornado Component
- Tornado diagram showing parameter sensitivity ranking
- Horizontal bars: positive (right) and negative (left) impact
- Color-coded by parameter category (activity data, emission factor, etc.)
- Ranked by absolute contribution to variance
- Show Sobol total-order indices
- Interactive hover with parameter details
- Configurable top-N parameters display

#### 2.1.3 ConfidenceIntervalChart Component
- Enhanced emissions trend chart with CI bands
- Shaded regions: 90% CI (light), 95% CI (medium), 99% CI (dark)
- Toggle between confidence levels
- Show upper/lower bounds as dashed lines
- Combined with existing MonthlyTrendChart enhancement
- Support for category-level and aggregate views

#### 2.1.4 MonteCarloResultsPanel Component
- Summary statistics card: mean, median, std_dev, CV, skewness, kurtosis
- Distribution type indicator (normal, lognormal)
- Iteration count and computation time
- Data quality tier indicator
- Convergence assessment (is 10K enough?)
- Tabbed view: Summary | Distribution | Sensitivity | Raw Data
- Export full results (PDF, CSV, JSON)

#### 2.1.5 ScenarioComparisonChart Component
- Compare 2-5 scenarios side-by-side with uncertainty bands
- Box-and-whisker plots per scenario
- Probability of meeting reduction targets
- Scenario parameter overlay (what's different)
- Risk assessment color coding
- Show statistical significance of differences

#### 2.1.6 UncertaintyMetricsCard Component
- Enhanced StatCard for uncertainty-aware dashboard
- Show value with ± range (e.g., "1,250 ± 180 tCO2e")
- Mini sparkline with CI band
- Data quality indicator (Tier 1/2/3 badge)
- Coefficient of variation warning (>50% = high uncertainty)

### 2.2 CDP Integration Enhancement

#### 2.2.1 CDPQuestionnaireEditor Component (Frontend)
- Full CDP Climate Change questionnaire structure
- Section navigation: C0 (Intro), C1 (Governance), C2 (Risks), C3 (Strategy), C4 (Targets), C5 (Emissions Methodology), C6 (Emissions Data), C7 (Emissions Breakdown), C8 (Energy), C9 (Additional), C10 (Verification), C11 (Carbon Pricing), C12 (Engagement)
- Per-question form fields with validation
- Auto-populated fields highlighted in green
- Manual-entry fields highlighted in yellow
- Required vs. optional field indicators
- In-line help text from CDP guidance
- Save draft and resume later
- Version tracking per reporting year

#### 2.2.2 CDPProgressTracker Component (Frontend)
- Section-by-section completion percentage
- Overall completion score (0-100%)
- Auto-populated vs. manual-entry breakdown
- Data gap identification with recommended actions
- Deadline countdown (CDP submission window)
- Visual timeline with milestones
- Exportable progress report

#### 2.2.3 CDPDataMapping Component (Frontend)
- Visual mapping: emissions data → CDP questions
- Shows which data populates which CDP fields
- Data source tracking (ERP, manual, calculated)
- Confidence level per mapping
- Unmapped questions highlighted
- Drag-and-drop data assignment (advanced)

#### 2.2.4 Enhanced CDP Backend
- Full CDP questionnaire schema (all 13 sections, 200+ questions)
- Question-level validation rules
- Auto-population engine enhancement (90% -> 95%+):
  - C1 (Governance): Board oversight, management responsibility
  - C2 (Risks): Physical + transition risk assessment
  - C3 (Strategy): Scenario analysis results
  - C4 (Targets): SBTi-aligned targets
  - C5 (Emissions Methodology): GHG Protocol alignment
  - C6 (Emissions): Scope 1/2/3 with categories
  - C7 (Breakdown): By country, business division, activity
  - C8 (Energy): Total, renewable %, by source
  - C11 (Carbon Pricing): Internal carbon price, ETS exposure
- Multi-year comparison (YoY changes)
- Data gap analysis with severity scoring
- CDP scoring prediction (A/A-, B/B-, C/C-, D/D-)

#### 2.2.5 ComplianceScorecard Component (Frontend)
- Multi-standard compliance overview:
  - GHG Protocol Scope 3 Standard (2011)
  - ESRS E1 (EU CSRD)
  - CDP Climate Change (2025)
  - IFRS S2 Climate Disclosures
  - ISO 14083 Transport
- Coverage percentage per standard
- Requirement checklist with status
- Evidence trail linkage
- Action items for gaps
- Exportable compliance report

### 2.3 Advanced Sensitivity Analysis (Backend)

#### 2.3.1 Sobol Sensitivity Analysis
- First-order Sobol indices (Si) for each parameter
- Total-order Sobol indices (ST) including interactions
- Saltelli sampling scheme for efficiency
- 2^N + 2 × N × 2^N quasi-random samples
- Results formatted for tornado diagram
- Parameter interaction detection

#### 2.3.2 Morris Screening
- One-at-a-time (OAT) elementary effects
- Mean effect (μ*) for parameter importance
- Standard deviation (σ) for interaction/non-linearity
- Configurable trajectory count (r=10 default)
- Results for Morris scatter plot (μ* vs σ)
- Cost-effective screening for large parameter spaces

#### 2.3.3 Tornado Diagram Data Generator
- Generate ranked parameter sensitivity data
- Positive and negative impact quantification
- Baseline (p50) and extremes (p5, p95) per parameter
- Support for one-way and two-way sensitivity
- Export format compatible with frontend tornado component

### 2.4 Backend API Expansion

#### 2.4.1 Uncertainty API Endpoints
- POST /api/v1/uncertainty/analyze -- Run full uncertainty analysis
- GET /api/v1/uncertainty/{calculation_id} -- Get uncertainty results
- GET /api/v1/uncertainty/{calculation_id}/distribution -- Distribution data for charting
- POST /api/v1/uncertainty/sensitivity -- Run sensitivity analysis (Sobol/Morris)
- GET /api/v1/uncertainty/convergence/{calculation_id} -- Convergence assessment
- POST /api/v1/uncertainty/compare-scenarios -- Compare scenarios with uncertainty

#### 2.4.2 CDP API Endpoints
- GET /api/v1/cdp/questionnaire/{year} -- Full questionnaire structure
- PUT /api/v1/cdp/questionnaire/{year}/section/{section} -- Update section
- POST /api/v1/cdp/questionnaire/{year}/auto-populate -- Run auto-population
- GET /api/v1/cdp/questionnaire/{year}/progress -- Completion tracking
- POST /api/v1/cdp/questionnaire/{year}/validate -- Full validation
- GET /api/v1/cdp/questionnaire/{year}/gaps -- Data gap analysis
- GET /api/v1/cdp/questionnaire/{year}/score-prediction -- Score prediction
- GET /api/v1/cdp/questionnaire/compare/{year1}/{year2} -- Year comparison
- POST /api/v1/cdp/questionnaire/{year}/export -- Export (Excel, PDF, JSON)
- GET /api/v1/cdp/sections -- CDP section metadata

#### 2.4.3 Compliance API Endpoints
- GET /api/v1/compliance/scorecard -- Multi-standard scorecard
- GET /api/v1/compliance/standard/{standard} -- Standard-specific coverage
- GET /api/v1/compliance/gaps -- Cross-standard gap analysis
- GET /api/v1/compliance/evidence/{requirement_id} -- Evidence trail

#### 2.4.4 Settings Persistence API
- GET /api/v1/settings -- Get user settings
- PUT /api/v1/settings -- Update user settings
- GET /api/v1/settings/defaults -- Get default settings

---

## 3. File Structure (New Files)

```
applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/
    frontend/src/
        components/
            uncertainty/
                UncertaintyDistribution.tsx       (~300 lines)
                SensitivityTornado.tsx             (~250 lines)
                ConfidenceIntervalChart.tsx        (~200 lines)
                MonteCarloResultsPanel.tsx         (~350 lines)
                ScenarioComparisonChart.tsx        (~280 lines)
                UncertaintyMetricsCard.tsx         (~120 lines)
                index.ts                           (~30 lines)
            cdp/
                CDPQuestionnaireEditor.tsx         (~500 lines)
                CDPProgressTracker.tsx             (~250 lines)
                CDPDataMapping.tsx                 (~300 lines)
                ComplianceScorecard.tsx            (~350 lines)
                index.ts                           (~20 lines)
        pages/
            UncertaintyAnalysis.tsx               (~250 lines)
            CDPManagement.tsx                      (~300 lines)
            ComplianceDashboard.tsx                (~250 lines)
        store/
            uncertaintySlice.ts                    (~150 lines)
            cdpSlice.ts                            (~200 lines)
            complianceSlice.ts                     (~120 lines)
            settingsSlice.ts                       (~100 lines)
    services/
        methodologies/
            sensitivity_analysis.py               (~600 lines)
        agents/
            reporting/
                standards/
                    cdp_enhanced.py                (~800 lines)
                    cdp_questionnaire_schema.py    (~600 lines)
                    compliance_scorecard.py        (~500 lines)
        api/
            uncertainty_routes.py                  (~500 lines)
            cdp_routes.py                          (~600 lines)
            compliance_routes.py                   (~400 lines)
            settings_routes.py                     (~200 lines)
```

### 3.1 Modified Files
```
    frontend/src/
        App.tsx                                    (add routes for new pages)
        services/api.ts                            (add uncertainty/CDP/compliance endpoints)
        pages/Dashboard.tsx                        (integrate uncertainty metrics)
        pages/Reports.tsx                          (add CDP questionnaire link)
        components/EmissionsChart.tsx               (add CI band support)
        store/dashboardSlice.ts                     (add uncertainty fields)
    services/
        methodologies/uncertainty.py               (enhance SensitivityAnalyzer)
        agents/reporting/standards/cdp.py           (enhance CDP auto-population)
        agents/reporting/agent.py                   (add CDP questionnaire methods)
```

---

## 4. Development Tasks (Parallel Build Plan)

### Task Group A: Monte Carlo UI Components (Agent 1 - Frontend)
- A1: Build uncertainty/ component directory (6 components + index)
- A2: Build UncertaintyAnalysis.tsx page
- A3: Build uncertaintySlice.ts Redux state
- A4: Update Dashboard.tsx with uncertainty metrics
- A5: Update EmissionsChart.tsx with CI bands
- A6: Update api.ts with uncertainty endpoints

### Task Group B: CDP UI Components (Agent 2 - Frontend)
- B1: Build cdp/ component directory (4 components + index)
- B2: Build CDPManagement.tsx page
- B3: Build cdpSlice.ts Redux state
- B4: Build ComplianceDashboard.tsx page
- B5: Build complianceSlice.ts and settingsSlice.ts
- B6: Update App.tsx with new routes
- B7: Update Reports.tsx with CDP questionnaire link

### Task Group C: Backend - Sensitivity Analysis + Uncertainty API (Agent 3)
- C1: Build sensitivity_analysis.py (Sobol indices, Morris screening, tornado data)
- C2: Build uncertainty_routes.py (6 endpoints)
- C3: Enhance uncertainty.py SensitivityAnalyzer (replace stub)

### Task Group D: Backend - CDP Enhancement + Compliance (Agent 4)
- D1: Build cdp_enhanced.py (95%+ auto-population, all 13 sections)
- D2: Build cdp_questionnaire_schema.py (200+ questions, validation rules)
- D3: Build compliance_scorecard.py (5 standards)
- D4: Build cdp_routes.py (10 endpoints)
- D5: Build compliance_routes.py (4 endpoints)
- D6: Build settings_routes.py (3 endpoints)
- D7: Enhance existing cdp.py auto-population

---

## 5. Acceptance Criteria

1. Monte Carlo distribution visualization (histogram, violin, KDE overlay)
2. Sensitivity tornado diagram with Sobol indices
3. Confidence interval bands on emissions trend charts
4. Monte Carlo results panel with summary statistics
5. Scenario comparison with uncertainty bands
6. CDP questionnaire editor with all 13 sections
7. CDP auto-population rate improved to 95%+
8. CDP section completion tracking with progress bar
9. CDP data gap analysis with severity scoring
10. CDP score prediction (A through D-)
11. Multi-standard compliance scorecard
12. Sobol and Morris sensitivity analysis methods
13. New API endpoints for uncertainty, CDP, compliance
14. Settings persistence (backend)
15. All new code with comprehensive test coverage
