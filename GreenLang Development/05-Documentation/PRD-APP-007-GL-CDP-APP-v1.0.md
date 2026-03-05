# PRD: APP-007 -- GL-CDP-APP v1.0

## Document Info
| Field | Value |
|-------|-------|
| PRD ID | PRD-APP-007 |
| Application | GL-CDP-APP v1.0 (Beta) |
| Priority | P1 (High) |
| Version | 1.0.0 |
| Status | Approved |
| Author | GL-ProductManager + GL-RegulatoryIntelligence |
| Date | 2026-03-02 |
| Standard | CDP Climate Change Questionnaire (2025/2026 Integrated Format) aligned with IFRS S2, ESRS E1, TCFD, GRI |
| Base | 30 MRV Agents (750K+ lines), GL-GHG-APP v1.0, GL-ISO14064-APP v1.0, Scope 3 Category Mapper, Audit Trail & Lineage |
| Ralphy Task ID | APP-007 |

---

## 1. Overview

### 1.1 Purpose
Build GL-CDP-APP v1.0 as a comprehensive CDP Climate Change disclosure management platform enabling organizations to prepare, simulate scoring, optimize, and submit CDP questionnaire responses. This application orchestrates all 30 existing MRV agents for auto-population of emissions data, provides a CDP scoring simulator, gap analysis for score improvement, sector benchmarking, supply chain module, and 1.5C transition plan builder.

**Key differentiator**: While GL-GHG-APP handles GHG Protocol compliance and GL-ISO14064-APP handles ISO 14064-1 compliance, GL-CDP-APP focuses on CDP's investor-focused disclosure framework with its unique scoring methodology (D- through A), question-level response management, and competitive benchmarking -- plus auto-population from existing MRV data.

Core capabilities:
1. **Questionnaire Engine** -- Full CDP Climate Change questionnaire (13 modules, 200+ questions)
2. **Response Manager** -- Draft, review, approve workflow with rich-text editor and evidence attachment
3. **Data Auto-Population** -- Pull Scope 1/2/3 data from 30 MRV agents via GL-GHG-APP connectors
4. **Scoring Simulator** -- Predict CDP score (D- to A) with 17 scoring categories and weightings
5. **Gap Analysis** -- Identify response gaps and provide actionable improvement recommendations
6. **Benchmarking Engine** -- Compare against sector peers, regional peers, and CDP global averages
7. **Supply Chain Module** -- CDP Supply Chain questionnaire for supplier engagement
8. **Transition Plan Builder** -- 1.5C-aligned transition plan (required for A-level scoring)
9. **Verification Tracker** -- Track third-party verification status (required for A-level)
10. **Historical Tracker** -- Year-over-year score progression and trend analysis
11. **Report Generator** -- Submission-ready export (PDF, Excel, XML for CDP Online Response System)
12. **Dashboard & Analytics** -- Executive dashboard with score simulation, timeline, readiness metrics

### 1.2 CDP Climate Change Questionnaire Module Structure (2025/2026)

| Module | Name | Key Areas | Questions |
|--------|------|-----------|-----------|
| **M0** | Introduction | Organization profile, reporting boundary, base year | ~15 |
| **M1** | Governance | Board oversight, management responsibility, incentives | ~20 |
| **M2** | Policies & Commitments | Climate policies, commitments, deforestation-free | ~15 |
| **M3** | Risks & Opportunities | Climate risk assessment, physical/transition risks | ~25 |
| **M4** | Strategy | Business strategy alignment, scenario analysis | ~20 |
| **M5** | Transition Plans | 1.5C pathway, decarbonization roadmap, milestones | ~20 |
| **M6** | Implementation | Emissions reduction initiatives, investments, R&D | ~20 |
| **M7** | Environmental Performance -- Climate Change | Scope 1/2/3 emissions, methodology, verification | ~35 |
| **M8** | Environmental Performance -- Forests | Commodity-driven deforestation (if applicable) | ~15 |
| **M9** | Environmental Performance -- Water Security | Water dependencies (if applicable) | ~15 |
| **M10** | Supply Chain | Supplier engagement, Scope 3 collaboration | ~15 |
| **M11** | Additional Metrics | Sector-specific metrics, energy mix | ~10 |
| **M12** | Financial Services | Portfolio emissions, financed emissions (if FS) | ~20 |
| **M13** | Sign Off | Authorization, verification statement | ~5 |

### 1.3 CDP Scoring Methodology

#### Scoring Levels
| Level | Band | Score Range | Requirements |
|-------|------|-------------|-------------|
| **A** | Leadership | 80-100% | Full disclosure + best practice + 1.5C transition plan + verified emissions |
| **A-** | Leadership | 70-79% | Full disclosure + strong management + most best practices |
| **B** | Management | 60-69% | Evidence of environmental management actions |
| **B-** | Management | 50-59% | Some management actions documented |
| **C** | Awareness | 40-49% | Awareness of environmental issues and impacts |
| **C-** | Awareness | 30-39% | Basic awareness demonstrated |
| **D** | Disclosure | 20-29% | Basic disclosure provided |
| **D-** | Disclosure | 0-19% | Minimal/incomplete disclosure |

#### 17 Scoring Categories (Climate Change)
| # | Category | Weight (Mgmt) | Weight (Lead) |
|---|----------|---------------|---------------|
| 1 | Governance | 7% | 7% |
| 2 | Risk management processes | 6% | 5% |
| 3 | Risk disclosure | 5% | 4% |
| 4 | Opportunity disclosure | 5% | 4% |
| 5 | Business strategy | 6% | 5% |
| 6 | Scenario analysis | 5% | 5% |
| 7 | Targets | 8% | 8% |
| 8 | Emissions reduction initiatives | 7% | 7% |
| 9 | Scope 1 & 2 emissions (incl. verification) | 10% | 10% |
| 10 | Scope 3 emissions (incl. verification) | 8% | 8% |
| 11 | Energy | 6% | 6% |
| 12 | Carbon pricing | 4% | 4% |
| 13 | Value chain engagement | 6% | 6% |
| 14 | Public policy engagement | 3% | 3% |
| 15 | Transition plan | 6% | 8% |
| 16 | Portfolio climate performance (FS only) | 5% | 7% |
| 17 | Financial impact assessment | 3% | 3% |

#### A-Level Requirements (Critical for scoring engine)
1. Publicly available 1.5C-aligned transition plan
2. Complete emissions inventory with no material exclusions
3. Third-party verification of 100% Scope 1 and Scope 2 emissions
4. Third-party verification of >= 70% of at least one Scope 3 category
5. SBTi-validated or 1.5C-aligned target (>= 4.2% annual absolute reduction)

### 1.4 CDP-to-MRV Agent Mapping

| CDP Module / Category | Data Required | MRV Agents |
|----------------------|---------------|------------|
| **M7: Scope 1 Emissions** | Stationary, mobile, process, fugitive, refrigerants | MRV-001, 002, 003, 004, 005 |
| **M7: Scope 1 Land Use** | LULUCF, agricultural, waste | MRV-006, 007, 008 |
| **M7: Scope 2 Emissions** | Location-based, market-based, dual reporting | MRV-009, 010, 011, 012, 013 |
| **M7: Scope 3 Cat 1-5** | Upstream categories | MRV-014, 015, 016, 017, 018 |
| **M7: Scope 3 Cat 6-8** | Travel, commuting, leased assets | MRV-019, 020, 021 |
| **M7: Scope 3 Cat 9-15** | Downstream categories | MRV-022 through 028 |
| **M7: Cross-cutting** | Category mapping, audit trail | MRV-029 (Mapper), MRV-030 (Audit) |
| **M6: Reduction initiatives** | Management plan data | GL-ISO14064-APP management plan |
| **M5: Transition plan** | Targets, pathways, milestones | GL-GHG-APP + new transition engine |
| **M11: Energy** | Electricity, heat, steam, cooling | MRV-009, 010, 011, 012 |

### 1.5 Technical Context
- **Backend**: FastAPI + Python 3.11+ + Pydantic v2
- **Frontend**: React 18 + TypeScript + Material-UI 5 + Redux Toolkit + Recharts + Vite
- **MRV Agents**: 30 production-ready agents at `greenlang/`
- **Existing Apps**: GL-GHG-APP v1.0, GL-ISO14064-APP v1.0 for data sourcing
- **Database**: PostgreSQL + TimescaleDB (V001-V084 deployed)
- **Frameworks Aligned**: IFRS S2, ESRS E1, TCFD, GRI, SBTi

---

## 2. Application Components

### 2.1 Backend: Questionnaire Engine
- Full CDP Climate Change questionnaire structure (13 modules)
- Question registry with question_id, module, sub-section, question_text, guidance_text
- Question types: text, numeric, percentage, table, multi-select, single-select, yes/no
- Conditional logic: skip patterns, dependency chains, sector-specific routing
- Question versioning: 2024 vs 2025 vs 2026 questionnaire versions
- Module completion tracking with progress percentage
- Sector-specific question routing (e.g., Financial Services -> Module 12)
- Question weighting for scoring categories
- Guidance text and example responses per question

### 2.2 Backend: Response Manager
- Response lifecycle: draft -> in_review -> approved -> submitted
- Rich-text response editor with markdown support
- Evidence attachment: documents, data tables, links, screenshots
- Multi-user collaboration: assign questions to team members
- Review workflow: data owner -> sustainability manager -> C-suite sign-off
- Version control: track all response edits with timestamps
- Comment/annotation system for reviewers
- Response templates: reusable answers for recurring questions
- Bulk import: import previous year responses
- Auto-save with conflict resolution

### 2.3 Backend: Data Auto-Population Engine
- Connect to 30 MRV agents for Scope 1/2/3 emissions data
- Auto-populate M7 (Climate Change) emissions tables
- Auto-populate M11 (Energy) from Scope 2 agents
- Map MRV output formats to CDP table formats
- Unit conversion and rounding per CDP requirements
- Data freshness validation (alert if data older than reporting period)
- Manual override with justification tracking
- Confidence indicator per auto-populated field
- Reconciliation between auto-populated and manually entered data

### 2.4 Backend: Scoring Simulator
- CDP scoring algorithm implementation (17 categories)
- Question-level scoring: disclosure (D), awareness (C), management (B), leadership (A)
- Category weighting application (management and leadership weights)
- Overall score calculation with band determination
- What-if analysis: simulate score changes from improving specific responses
- Score breakdown by category with individual grades
- Score trajectory: predict score based on current completion
- A-level eligibility checker (5 mandatory requirements)
- Score comparison: current draft vs. previous submission
- Confidence interval on predicted score

### 2.5 Backend: Gap Analysis Engine
- Compare current responses against scoring criteria for each question
- Identify missing responses (disclosure-level gaps)
- Identify weak responses (awareness-level gaps)
- Identify management action gaps (management-level)
- Identify best-practice gaps (leadership-level)
- Priority ranking: highest-impact gaps to improve score
- Actionable recommendations per gap with example responses
- Effort estimation: low/medium/high effort to close each gap
- Score uplift prediction: expected score improvement per gap closed
- Gap tracking over time: progress dashboard

### 2.6 Backend: Benchmarking Engine
- Sector-level benchmarking (GICS sector classification)
- Regional benchmarking (geography-based peer groups)
- Score distribution: histogram of scores in peer group
- Category-by-category comparison against sector average
- Top quartile identification: what leaders are doing differently
- Historical trend: sector average score progression
- A-list rate: percentage of sector achieving A/A-
- Custom peer group: user-defined comparator set
- Anonymous benchmarking data (no company-specific data exposed)

### 2.7 Backend: Supply Chain Module
- CDP Supply Chain questionnaire structure
- Supplier invitation and tracking
- Supplier response status dashboard
- Aggregated supplier emissions data
- Supplier engagement scoring
- Cascade request management
- Supplier data quality assessment
- Supply chain emissions hotspot identification
- Supplier improvement tracking over time

### 2.8 Backend: Transition Plan Builder
- 1.5C pathway definition (required for A-level)
- Decarbonization milestone setting (short/medium/long-term)
- Technology roadmap: identified decarbonization levers
- Investment plan: CapEx/OpEx for decarbonization
- Revenue alignment: % revenue from low-carbon products
- Scope 1/2/3 reduction pathway modeling
- SBTi alignment checker
- Progress tracking against transition milestones
- Board oversight documentation
- Public disclosure readiness assessment

### 2.9 Backend: Verification Tracker
- Third-party verification status tracking per scope
- Verification coverage percentage (Scope 1, 2, 3)
- Verifier details: organization, standard, assurance level
- Verification statement document management
- Limited vs. reasonable assurance tracking
- Verification schedule management
- A-level verification requirements checker:
  - 100% Scope 1+2 verified
  - >= 70% of at least one Scope 3 category verified

### 2.10 Backend: Historical Tracker
- Year-over-year score comparison (5+ year history)
- Score progression chart with band transitions
- Category-level trend analysis
- Response reuse: carry forward previous year answers
- Change log: what changed between submissions
- Improvement rate calculation
- Year-over-year emissions data comparison
- Submission timeline tracking

### 2.11 Backend: Report Generator
- CDP Online Response System (ORS) compatible XML export
- PDF report: formatted questionnaire with all responses
- Excel export: tabular response data
- Executive summary: key metrics and scores
- Board report: governance-focused summary
- Verification package: evidence bundle for verifiers
- Submission checklist: completeness validation
- Multi-language support (English primary)

### 2.12 Frontend: Dashboard
- CDP score gauge (D- to A) with band indicator
- Score simulator quick view (predicted score)
- Module completion progress bars (13 modules)
- Gap count by severity (critical/high/medium/low)
- Timeline: submission deadline countdown
- Readiness score: % questions answered, % reviewed, % approved
- Year-over-year score trend chart
- Category radar chart (17 categories)
- A-level eligibility status (5 requirements checklist)
- Recent activity feed

### 2.13 Frontend: Questionnaire Wizard
- Module-by-module navigation (M0-M13)
- Question list with status indicators (not started/draft/reviewed/approved)
- Rich-text response editor with formatting toolbar
- Evidence attachment panel (drag-and-drop)
- Auto-populated data display with source attribution
- Guidance panel: CDP guidance text + example responses
- Score impact indicator per question
- Assignment panel: assign to team member
- Comment thread per question
- Previous year response reference panel

### 2.14 Frontend: Scoring Simulator Page
- Overall score gauge with predicted band
- 17-category breakdown bar chart
- What-if scenario builder (toggle improvements)
- Score delta visualization (current vs. improved)
- A-level requirements checklist with status
- Category drill-down with question-level scores
- Score confidence meter
- Export score report

### 2.15 Frontend: Gap Analysis Page
- Gap list with severity color coding
- Filter by module, category, severity, effort
- Gap detail view: current response vs. ideal response
- Improvement recommendations with action items
- Score uplift calculator
- Priority matrix (impact vs. effort)
- Gap resolution tracking
- Export gap report

### 2.16 Frontend: Benchmarking Page
- Sector selection and peer group configuration
- Score distribution histogram
- Category comparison spider chart
- Peer ranking table
- A-list rate visualization
- Historical trend comparison
- Best practice highlights
- Export benchmark report

### 2.17 Frontend: Supply Chain Page
- Supplier list with response status
- Engagement dashboard (invited/responded/scored)
- Aggregated supplier emissions chart
- Hotspot identification map
- Supplier improvement tracker
- Cascade request manager

### 2.18 Frontend: Transition Plan Page
- Pathway visualization (current to 2050)
- Milestone timeline with progress indicators
- Technology lever breakdown
- Investment plan summary
- SBTi alignment status
- Revenue alignment tracker
- Board oversight documentation

### 2.19 Frontend: Reports Page
- Report builder with template selection
- Preview panel with formatting
- Export options (PDF, Excel, XML/ORS)
- Submission checklist with validation
- Report history and version management
- Verification package builder

### 2.20 Frontend: Settings Page
- Organization profile configuration
- Reporting year and boundary settings
- Team member management and permissions
- Sector classification (GICS)
- Notification preferences
- API integrations (MRV agent connections)
- Previous year data import

---

## 3. File Structure

```
applications/GL-CDP-APP/CDP-Disclosure-Platform/
    config/
        cdp_config.yaml                              (~350 lines)
    services/
        __init__.py                                  (~100 lines)
        config.py                                    (~500 lines)
        models.py                                    (~1,400 lines)
        questionnaire_engine.py                      (~900 lines)
        response_manager.py                          (~800 lines)
        scoring_simulator.py                         (~1,000 lines)
        data_connector.py                            (~700 lines)
        gap_analysis_engine.py                       (~800 lines)
        benchmarking_engine.py                       (~600 lines)
        supply_chain_module.py                       (~700 lines)
        transition_plan_engine.py                    (~700 lines)
        verification_tracker.py                      (~500 lines)
        historical_tracker.py                        (~500 lines)
        report_generator.py                          (~800 lines)
        setup.py                                     (~500 lines)
        api/
            __init__.py                              (~50 lines)
            questionnaire_routes.py                  (~600 lines)
            response_routes.py                       (~600 lines)
            scoring_routes.py                        (~500 lines)
            gap_analysis_routes.py                   (~450 lines)
            benchmarking_routes.py                   (~400 lines)
            supply_chain_routes.py                   (~500 lines)
            transition_plan_routes.py                (~450 lines)
            reporting_routes.py                      (~500 lines)
            dashboard_routes.py                      (~400 lines)
            settings_routes.py                       (~300 lines)
    frontend/
        package.json, tsconfig.json, vite.config.ts, index.html
        src/
            main.tsx, App.tsx
            types/index.ts                           (~600 lines)
            services/api.ts                          (~700 lines)
            store/ (index.ts, hooks.ts, 12 slices)
            components/
                layout/ (Sidebar, Header, Layout)
                common/ (ScoreGauge, ProgressBar, StatusChip, DataTable, LoadingSpinner)
                dashboard/ (ScoreCard, ModuleProgress, GapSummary, TimelineCountdown, CategoryRadar, ReadinessScore, TrendChart, AListChecklist)
                questionnaire/ (ModuleNav, QuestionCard, ResponseEditor, EvidencePanel, GuidancePanel, AutoPopulatedData, AssignmentPanel)
                scoring/ (ScoreGaugeDetail, CategoryBreakdown, WhatIfBuilder, ScoreDelta, ARequirementsCheck)
                gaps/ (GapList, GapDetail, PriorityMatrix, UpliftCalculator, RecommendationCard)
                benchmarking/ (SectorSelector, ScoreHistogram, SpiderChart, PeerTable, AListRate)
                supply_chain/ (SupplierList, EngagementDashboard, HotspotMap, SupplierTracker)
                transition/ (PathwayChart, MilestoneTimeline, TechLeverBreakdown, InvestmentPlan, SBTiStatus)
                reports/ (ReportBuilder, PreviewPanel, ExportDialog, SubmissionChecklist, VerificationPackage)
                verification/ (VerificationStatus, CoverageTracker, VerifierDetails, AssuranceLevel)
                historical/ (YearComparison, TrendChart, ChangeLog, ScoreProgression)
            pages/
                Dashboard.tsx
                QuestionnaireWizard.tsx
                ModuleDetail.tsx
                ScoringSimulator.tsx
                GapAnalysis.tsx
                Benchmarking.tsx
                SupplyChain.tsx
                TransitionPlan.tsx
                Verification.tsx
                Reports.tsx
                Historical.tsx
                Settings.tsx
            utils/ (formatters.ts, validators.ts, scoringHelpers.ts)
    tests/
        __init__.py
        test_models.py
        test_questionnaire_engine.py
        test_response_manager.py
        test_scoring_simulator.py
        test_data_connector.py
        test_gap_analysis_engine.py
        test_benchmarking_engine.py
        test_supply_chain_module.py
        test_transition_plan_engine.py
        test_verification_tracker.py
        test_historical_tracker.py
        test_report_generator.py
        test_api_routes.py
```

---

## 4. Database Migration: V085

```sql
-- V085__cdp_app_service.sql
-- Tables: ~20 tables + 3 hypertables + 2 continuous aggregates

-- Core tables:
-- gl_cdp_organizations              (CDP org profile, sector, region)
-- gl_cdp_questionnaires             (questionnaire instances per year)
-- gl_cdp_modules                    (13 modules per questionnaire)
-- gl_cdp_questions                  (200+ questions with metadata)
-- gl_cdp_responses                  (response per question per org)
-- gl_cdp_response_versions          (version history of responses)
-- gl_cdp_evidence_attachments       (documents/data linked to responses)
-- gl_cdp_review_workflows           (draft -> review -> approve)
-- gl_cdp_scoring_results            (overall + category scores)
-- gl_cdp_category_scores            (17 category-level scores)
-- gl_cdp_gap_analyses               (gap identification results)
-- gl_cdp_gap_items                  (individual gap items)
-- gl_cdp_benchmarks                 (sector/regional benchmarks)
-- gl_cdp_peer_comparisons           (peer group comparisons)
-- gl_cdp_supply_chain_requests      (supplier engagement requests)
-- gl_cdp_supplier_responses         (supplier questionnaire responses)
-- gl_cdp_transition_plans           (1.5C transition plans)
-- gl_cdp_transition_milestones      (decarbonization milestones)
-- gl_cdp_verification_records       (third-party verification)
-- gl_cdp_submissions                (final submission records)

-- Hypertables: gl_cdp_responses, gl_cdp_scoring_results, gl_cdp_gap_analyses
-- Continuous aggregates: monthly_response_progress, quarterly_score_trends
```

---

## 5. Development Tasks (Parallel Build Plan)

### Task Group A: Backend Core Engines (Agent 1 - gl-backend-developer)
- A1: config.py, models.py (enums, domain models, 60+ Pydantic schemas)
- A2: questionnaire_engine.py (13 modules, 200+ questions, conditional logic)
- A3: response_manager.py (lifecycle, versioning, review workflow)
- A4: scoring_simulator.py (17 categories, 4 scoring levels, weightings)
- A5: data_connector.py (30 MRV agent integration, auto-population)
- A6: gap_analysis_engine.py (gap identification, recommendations, priority)
- A7: benchmarking_engine.py (sector, regional, custom peer groups)
- A8: supply_chain_module.py (supplier engagement, aggregation)
- A9: transition_plan_engine.py (1.5C pathway, milestones, SBTi)
- A10: verification_tracker.py (coverage, assurance level, A-level check)
- A11: historical_tracker.py (YoY comparison, trend analysis)
- A12: report_generator.py (PDF, Excel, XML/ORS export)
- A13: setup.py + __init__.py + cdp_config.yaml

### Task Group B: Backend API Layer (Agent 2 - gl-api-developer)
- B1: api/__init__.py, questionnaire_routes.py (10+ endpoints)
- B2: response_routes.py (CRUD, workflow, bulk operations)
- B3: scoring_routes.py, gap_analysis_routes.py
- B4: benchmarking_routes.py, supply_chain_routes.py
- B5: transition_plan_routes.py, reporting_routes.py
- B6: dashboard_routes.py, settings_routes.py

### Task Group C: Frontend Core + Layout + Store (Agent 3 - gl-frontend-developer)
- C1: Config files, entry points, types, API client
- C2: Redux store (12 slices)
- C3: Layout + common components (ScoreGauge, ProgressBar, etc.)
- C4: App.tsx with routing

### Task Group D: Frontend Pages + Domain Components (Agent 4 - gl-frontend-developer)
- D1: Dashboard components (ScoreCard, ModuleProgress, CategoryRadar, etc.)
- D2: Questionnaire components (ModuleNav, QuestionCard, ResponseEditor, etc.)
- D3: Scoring + Gap components (WhatIfBuilder, GapList, PriorityMatrix, etc.)
- D4: Benchmarking + Supply Chain components
- D5: Transition + Verification + Historical + Reports components
- D6: All 12 pages

### Task Group E: Tests + DB Migration (Agent 5 - gl-test-engineer)
- E1: 13 backend test files (400+ tests)
- E2: V085 database migration

---

## 6. Acceptance Criteria

1. Full CDP Climate Change questionnaire with 13 modules and 200+ questions
2. Response manager with draft/review/approve workflow and versioning
3. Data auto-population from 30 MRV agents (Scope 1/2/3)
4. CDP scoring simulator with 17 categories and 4 scoring levels (D- to A)
5. What-if analysis for score improvement scenarios
6. Gap analysis with actionable recommendations and effort estimation
7. Sector benchmarking against peer groups
8. CDP Supply Chain module for supplier engagement
9. 1.5C transition plan builder with SBTi alignment checking
10. Verification tracking with A-level requirements validation
11. Historical year-over-year comparison (5+ year support)
12. Report export: PDF, Excel, XML (CDP ORS compatible)
13. Full React dashboard with 12 pages
14. 60+ REST API endpoints
15. 400+ backend tests
16. V085 database migration with TimescaleDB hypertables

---

## 7. Framework Alignment

| Framework | Alignment |
|-----------|-----------|
| **IFRS S2** | Climate-related disclosures alignment (governance, strategy, risk, metrics) |
| **ESRS E1** | Climate change mitigation/adaptation (CSRD) |
| **TCFD** | 4-pillar alignment (governance, strategy, risk management, metrics & targets) |
| **GRI 305** | GHG emissions disclosure |
| **SBTi** | Science-based target validation and 1.5C pathway |
| **GHG Protocol** | Scope 1/2/3 methodology (via MRV agents) |
| **ISO 14064-1** | Organization-level GHG quantification (via GL-ISO14064-APP) |
