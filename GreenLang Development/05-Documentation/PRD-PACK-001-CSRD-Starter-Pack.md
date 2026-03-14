# PRD-PACK-001: CSRD Starter Pack

## Document Control

| Field | Value |
|-------|-------|
| PRD ID | PRD-PACK-001 |
| Title | CSRD Starter Pack |
| Category | Solution Packs > EU Compliance |
| Version | 1.0.0 |
| Status | APPROVED |
| Author | GreenLang Product Team |
| Created | 2026-03-14 |
| Priority | P0 - Critical Path |
| Target Users | EU companies subject to CSRD (50,000+ globally) |

---

## 1. Executive Summary

The CSRD Starter Pack is GreenLang's first Solution Pack - a curated, deployable bundle that combines the GL-CSRD-APP (6-agent pipeline), GreenLang MRV calculation engines (30 agents), data intake connectors (20 agents), and foundation infrastructure (10 agents) into a single, ready-to-deploy product for EU Corporate Sustainability Reporting Directive compliance.

### 1.1 Why a Pack?

Individual agents and apps are building blocks. Customers don't buy building blocks - they buy solutions. The CSRD Starter Pack is the first "solution" that packages:
- **Pre-configured workflows** for common CSRD reporting scenarios
- **Sector-specific presets** for different industries
- **Report templates** matching auditor expectations
- **Quick-start onboarding** to reduce time-to-first-report
- **Integration wiring** that connects 66+ agents into a cohesive pipeline

### 1.2 What's New vs. What's Reused

| Component | Status | Source |
|-----------|--------|--------|
| GL-CSRD-APP (6 agents) | REUSE | `applications/GL-CSRD-APP/` |
| MRV Scope 1 Agents (8) | REUSE | `greenlang/scope1_agents/` |
| MRV Scope 2 Agents (5) | REUSE | `greenlang/scope2_agents/` |
| MRV Scope 3 Agents (17) | REUSE | `greenlang/scope3_agents/` |
| Data Intake Agents (9) | REUSE | `greenlang/data/` |
| Data Quality Agents (11) | REUSE | `greenlang/quality_agents/` |
| Foundation Agents (10) | REUSE | `greenlang/foundation_agents/` |
| Auth/RBAC/Security | REUSE | `greenlang/auth/`, `greenlang/access_guard/` |
| Pack Manifest System | **NEW** | `packs/eu-compliance/PACK-001-csrd-starter/` |
| Workflow Templates (5) | **NEW** | `packs/.../workflows/` |
| Configuration Presets (4) | **NEW** | `packs/.../config/` |
| Report Templates (6) | **NEW** | `packs/.../templates/` |
| Integration Layer | **NEW** | `packs/.../integrations/` |
| Pack-Level E2E Tests | **NEW** | `packs/.../tests/` |
| Quick-Start Wizard | **NEW** | `packs/.../integrations/setup_wizard.py` |
| Demo Mode & Sample Data | **NEW** | `packs/.../config/demo/` |

---

## 2. Problem Statement

### 2.1 Customer Pain Points

1. **Complexity**: CSRD requires reporting across 12 ESRS standards, 1,082 data points, Scope 1/2/3 GHG emissions, double materiality assessment, and XBRL-tagged digital reports. Companies don't know where to start.

2. **Integration Gap**: Even with all agents built, connecting them into a working end-to-end solution requires deep platform knowledge.

3. **Time Pressure**: Large EU companies (>500 employees) must report starting FY2024. Mid-size companies start FY2025. SMEs start FY2026. The clock is ticking.

4. **Audit Readiness**: External auditors require specific documentation formats, calculation provenance, and compliance evidence packages.

### 2.2 Solution

A "batteries-included" pack that:
- Deploys in under 1 hour with guided setup
- Produces a compliant CSRD report within 30 minutes of data input
- Generates auditor-ready packages automatically
- Supports 12 ESRS standards (1,082 data points)
- Provides sector-specific configurations out of the box

---

## 3. Pack Architecture

### 3.1 Component Registry

```
PACK-001-csrd-starter/
├── pack.yaml                           # Pack manifest & component registry
├── README.md                           # Pack documentation & quick-start
│
├── workflows/                          # Pre-built orchestration workflows
│   ├── __init__.py
│   ├── annual_reporting.py             # Full annual CSRD reporting cycle
│   ├── quarterly_update.py             # Quarterly data refresh & tracking
│   ├── materiality_assessment.py       # Standalone double materiality
│   ├── data_onboarding.py             # First-time data import & validation
│   └── audit_preparation.py           # Pre-audit compliance check & package
│
├── config/                             # Configuration presets
│   ├── __init__.py
│   ├── pack_config.py                  # Pack configuration manager
│   ├── presets/
│   │   ├── large_enterprise.yaml       # >10,000 employees, listed
│   │   ├── mid_market.yaml             # 1,000-10,000 employees
│   │   ├── sme.yaml                    # 250-1,000 employees (simplified)
│   │   └── first_time_reporter.yaml    # No prior ESG reporting history
│   ├── sectors/
│   │   ├── manufacturing.yaml          # Heavy industry, process emissions
│   │   ├── financial_services.yaml     # Financed emissions, PCAF
│   │   ├── technology.yaml             # Data centers, Scope 2 heavy
│   │   ├── retail.yaml                 # Supply chain, Scope 3 heavy
│   │   └── energy.yaml                 # Fossil fuels, transition plans
│   └── demo/
│       ├── demo_config.yaml            # Demo mode configuration
│       ├── demo_company_profile.json   # Sample company data
│       └── demo_esg_data.csv           # Sample ESG dataset (500 records)
│
├── templates/                          # Report & output templates
│   ├── __init__.py
│   ├── executive_summary.py            # Board-level CSRD summary
│   ├── esrs_disclosure.py              # Full ESRS disclosure narrative
│   ├── materiality_matrix.py           # Visual materiality assessment
│   ├── ghg_emissions_report.py         # Scope 1/2/3 breakdown report
│   ├── auditor_package.py              # External auditor evidence package
│   └── compliance_dashboard.py         # Real-time compliance status
│
├── integrations/                       # Agent wiring & integration layer
│   ├── __init__.py
│   ├── pack_orchestrator.py            # Master orchestrator connecting all agents
│   ├── mrv_bridge.py                   # Bridge: MRV engines ↔ CSRD Calculator
│   ├── data_pipeline_bridge.py         # Bridge: Data agents ↔ CSRD Intake
│   ├── setup_wizard.py                 # Interactive guided setup
│   └── health_check.py                # Pack health verification
│
└── tests/                              # Pack-level tests
    ├── __init__.py
    ├── conftest.py                     # Shared test fixtures
    ├── test_pack_manifest.py           # Validate pack.yaml integrity
    ├── test_workflows.py               # Workflow E2E tests
    ├── test_config_presets.py          # Configuration preset validation
    ├── test_templates.py               # Template rendering tests
    ├── test_integrations.py            # Integration bridge tests
    ├── test_demo_mode.py              # Demo mode E2E test
    └── test_e2e_annual_report.py      # Full annual reporting E2E
```

### 3.2 Agent Dependencies

The pack orchestrates these existing agents:

**Tier 1 - Data Intake (from `greenlang/data/`)**
- AGENT-DATA-001: PDF & Invoice Extractor
- AGENT-DATA-002: Excel/CSV Normalizer
- AGENT-DATA-003: ERP/Finance Connector
- AGENT-DATA-008: Supplier Questionnaire Processor

**Tier 2 - Data Quality (from `greenlang/quality_agents/`)**
- AGENT-DATA-010: Data Quality Profiler
- AGENT-DATA-011: Duplicate Detection
- AGENT-DATA-012: Missing Value Imputer
- AGENT-DATA-013: Outlier Detection
- AGENT-DATA-019: Validation Rule Engine

**Tier 3 - MRV Calculation (from `greenlang/scope*_agents/`)**
- AGENT-MRV-001 through 008: Scope 1 (8 engines)
- AGENT-MRV-009 through 013: Scope 2 (5 engines)
- AGENT-MRV-014 through 030: Scope 3 (17 engines)

**Tier 4 - CSRD Pipeline (from `applications/GL-CSRD-APP/`)**
- IntakeAgent: ESG data validation against ESRS catalog
- MaterialityAgent: AI-powered double materiality
- CalculatorAgent: 524 ESRS formulas (zero hallucination)
- AggregatorAgent: Cross-framework mapping (TCFD/GRI/SASB)
- ReportingAgent: XBRL tagging, iXBRL, ESEF packaging
- AuditAgent: 235 compliance rules

**Tier 5 - Foundation (from `greenlang/foundation_agents/`)**
- Orchestrator (DAG), Schema Compiler, Citations, Assumptions, Audit Trail

---

## 4. Detailed Component Specifications

### 4.1 Pack Manifest (`pack.yaml`)

The pack manifest is the single source of truth for what's included in the pack, its version, dependencies, and deployment requirements.

```yaml
name: csrd-starter-pack
version: 1.0.0
category: eu-compliance
display_name: "CSRD Starter Pack"
description: "Complete CSRD compliance solution"
components:
  apps: [GL-CSRD-APP]
  agents:
    data: [001, 002, 003, 008]
    quality: [010, 011, 012, 013, 019]
    mrv_scope1: [001-008]
    mrv_scope2: [009-013]
    mrv_scope3: [014-030]
    foundation: [001-010]
  workflows: [annual, quarterly, materiality, onboarding, audit]
  presets: [large, mid, sme, first_time]
  templates: [exec_summary, esrs, materiality, ghg, auditor, dashboard]
requirements:
  python: ">=3.11"
  postgresql: ">=14"
  redis: ">=7"
  memory: "16GB minimum"
  storage: "50GB minimum"
```

### 4.2 Workflow Specifications

#### 4.2.1 Annual Reporting Workflow (`annual_reporting.py`)

The primary workflow that orchestrates a full CSRD reporting cycle:

```
Phase 1: Data Collection (Weeks 1-2)
  ├── Activate data connectors (ERP, Excel, questionnaire)
  ├── Run data quality checks (profiling, dedup, validation)
  ├── Flag missing/incomplete data points
  └── Generate data readiness report

Phase 2: Materiality Assessment (Week 3)
  ├── Run double materiality assessment (AI-powered)
  ├── Generate materiality matrix
  ├── Queue for human review
  └── Apply materiality results to scope determination

Phase 3: Emissions Calculation (Week 4)
  ├── Execute Scope 1 calculations (8 MRV agents)
  ├── Execute Scope 2 calculations (5 MRV agents)
  ├── Execute Scope 3 calculations (17 MRV agents)
  ├── Run dual reporting reconciliation
  └── Generate calculation audit trail

Phase 4: Report Generation (Week 5)
  ├── Aggregate across frameworks (TCFD, GRI, SASB → ESRS)
  ├── Generate XBRL-tagged iXBRL report
  ├── Generate PDF management report
  ├── Create ESEF submission package
  └── Draft narrative sections (AI + human review)

Phase 5: Compliance & Audit (Week 6)
  ├── Execute 235 ESRS compliance rules
  ├── Cross-reference validation
  ├── Calculation re-verification
  ├── Generate external auditor package
  └── Produce compliance certification
```

#### 4.2.2 Quarterly Update Workflow (`quarterly_update.py`)

Lighter-weight quarterly data refresh:
- Re-run data intake with new quarter data
- Recalculate emissions for updated period
- Update trend analysis and benchmarks
- Generate quarterly progress report
- Flag deviations from annual targets

#### 4.2.3 Data Onboarding Workflow (`data_onboarding.py`)

First-time customer data import:
- Guided data source configuration
- Sample data validation against ESRS data point catalog
- Auto-detection of data format and field mapping
- Gap analysis (which ESRS data points are missing)
- Data quality baseline assessment
- Recommended actions for data completeness

#### 4.2.4 Materiality Assessment Workflow (`materiality_assessment.py`)

Standalone materiality workflow:
- Company context collection
- Stakeholder identification and analysis
- Impact materiality scoring (severity x scope x irremediability)
- Financial materiality scoring (magnitude x likelihood)
- Double materiality matrix generation
- Human review queue with approval workflow
- Material topic documentation for auditors

#### 4.2.5 Audit Preparation Workflow (`audit_preparation.py`)

Pre-audit compliance verification:
- Full compliance rule execution (235 rules)
- Calculation verification and re-computation
- Data lineage documentation
- Evidence package assembly
- Gap identification and remediation suggestions
- Auditor-ready documentation generation

### 4.3 Configuration Presets

#### 4.3.1 Large Enterprise Preset
- Full 12 ESRS standards, all 1,082 data points
- All Scope 3 categories (15) enabled
- XBRL tagging with full taxonomy
- Multi-language reporting (EN, DE, FR, ES)
- Multi-subsidiary consolidation enabled
- External auditor package generation

#### 4.3.2 Mid-Market Preset
- 12 ESRS standards, material topics focus
- Scope 3: Top 5 categories by materiality
- XBRL tagging (essential fields)
- Single-language reporting
- Simplified consolidation

#### 4.3.3 SME Preset
- ESRS simplified standards (LSME/VSME)
- Scope 1 + 2 mandatory, Scope 3 optional
- Basic XBRL tagging
- Guided data collection forms
- Simplified materiality assessment

#### 4.3.4 First-Time Reporter Preset
- Step-by-step guided mode
- Extended data onboarding workflow
- Pre-populated templates with examples
- Increased AI assistance for narrative sections
- Tutorial mode for materiality assessment

### 4.4 Report Templates

#### 4.4.1 Executive Summary Template
- Board-level 2-page CSRD summary
- Key metrics dashboard (Scope 1/2/3, materiality, compliance)
- Year-over-year trends
- Regulatory deadline tracking
- Risk heatmap

#### 4.4.2 ESRS Disclosure Template
- Full narrative disclosure per ESRS standard
- Auto-populated metrics with provenance
- Cross-reference to GRI/TCFD/SASB equivalents
- AI-drafted narrative sections (marked for review)

#### 4.4.3 GHG Emissions Report Template
- Scope 1/2/3 breakdown with waterfall chart
- Emission intensity metrics
- Year-over-year comparison
- Science-based targets progress (if applicable)
- Methodology documentation

#### 4.4.4 Materiality Matrix Template
- Interactive 2D materiality matrix
- Impact vs. financial materiality scatter plot
- Material topic prioritization
- Stakeholder engagement summary
- Methodology documentation

#### 4.4.5 Auditor Package Template
- Complete calculation audit trail
- Data lineage documentation
- Source data references
- Compliance checklist (235 rules)
- Calculation verification evidence
- Methodology notes

#### 4.4.6 Compliance Dashboard Template
- Real-time compliance status by ESRS standard
- Data completeness heatmap
- Outstanding actions and deadlines
- Historical compliance trends
- Alert summary

### 4.5 Integration Layer

#### 4.5.1 Pack Orchestrator (`pack_orchestrator.py`)

The master orchestrator that coordinates all agents within the pack:

```python
class CSRDPackOrchestrator:
    """
    Master orchestrator for CSRD Starter Pack.

    Connects 66+ agents into a cohesive pipeline:
    - Data intake agents → CSRD IntakeAgent
    - MRV calculation engines → CSRD CalculatorAgent
    - Quality agents → Data validation pipeline
    - Foundation agents → Orchestration & audit trail
    """

    def __init__(self, config: PackConfig):
        self.config = config
        self.workflow_engine = WorkflowEngine()
        self.mrv_bridge = MRVBridge()
        self.data_bridge = DataPipelineBridge()

    async def run_workflow(self, workflow_name: str, params: dict) -> WorkflowResult:
        """Execute a named workflow with given parameters."""

    async def get_status(self) -> PackStatus:
        """Get current pack health and status."""
```

#### 4.5.2 MRV Bridge (`mrv_bridge.py`)

Bridges the 30 MRV agents into the CSRD Calculator:

```python
class MRVBridge:
    """
    Connects GreenLang MRV agents to CSRD CalculatorAgent.

    Maps ESRS E1 climate metrics to appropriate MRV calculation engines.
    Ensures zero-hallucination guarantee is maintained across the bridge.
    """

    def route_calculation(self, metric_code: str, data: dict) -> CalculationResult:
        """Route ESRS metric calculation to appropriate MRV agent."""
        # E1-1 (Scope 1) → stationary_combustion + mobile + process + fugitive
        # E1-2 (Scope 2) → location_based + market_based
        # E1-3 (Scope 3) → 15 category agents
```

#### 4.5.3 Data Pipeline Bridge (`data_pipeline_bridge.py`)

Connects data intake agents to the CSRD pipeline:

```python
class DataPipelineBridge:
    """
    Connects GreenLang Data agents to CSRD IntakeAgent.

    Routes incoming data through appropriate connectors:
    - PDF/Invoice → PDF Extractor → CSRD Intake
    - Excel/CSV → Excel Normalizer → CSRD Intake
    - ERP → ERP Connector → CSRD Intake
    - Questionnaire → Questionnaire Processor → CSRD Intake

    Applies quality pipeline before CSRD intake.
    """
```

#### 4.5.4 Setup Wizard (`setup_wizard.py`)

Interactive guided setup for new pack deployments:

```python
class CSRDSetupWizard:
    """
    Guided setup for CSRD Starter Pack.

    Steps:
    1. Company profile collection
    2. Reporting scope selection (ESRS standards)
    3. Data source configuration
    4. Sector preset selection
    5. Integration configuration
    6. Health check & validation
    7. Demo run with sample data
    """
```

#### 4.5.5 Health Check (`health_check.py`)

Pack health verification:

```python
class PackHealthCheck:
    """
    Verifies all pack components are operational.

    Checks:
    - All referenced agents are available
    - Database connectivity
    - Configuration validity
    - Data file integrity
    - API endpoint availability
    - Auth/RBAC configuration
    """
```

---

## 5. Technical Requirements

### 5.1 Performance Targets

| Metric | Target |
|--------|--------|
| Full annual report generation | < 30 minutes (10,000 data points) |
| Data onboarding (first time) | < 1 hour guided setup |
| Quarterly update | < 15 minutes |
| Materiality assessment | < 10 minutes + human review |
| Audit preparation | < 20 minutes |
| Pack health check | < 30 seconds |

### 5.2 Quality Targets

| Metric | Target |
|--------|--------|
| Calculation accuracy | 100% (zero hallucination) |
| ESRS compliance rules | 235 automated checks |
| ESRS data point coverage | 1,082 (96% automation) |
| XBRL taxonomy compliance | 100% |
| Test coverage | 85%+ for new pack code |

### 5.3 Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | >= 3.11 | Runtime |
| PostgreSQL | >= 14 | Data storage |
| Redis | >= 7 | Caching |
| GL-CSRD-APP | 1.1.0 | Core CSRD pipeline |
| greenlang core | >= 0.3.0 | Agent framework |
| pydantic | >= 2.5.0 | Data validation |
| pyyaml | >= 6.0 | Configuration |
| jinja2 | >= 3.1.0 | Report templates |

---

## 6. Testing Strategy

### 6.1 Test Categories

| Category | Count | Purpose |
|----------|-------|---------|
| Pack manifest validation | 10 | Verify pack.yaml integrity |
| Configuration preset tests | 20 | Validate all presets |
| Workflow E2E tests | 25 | End-to-end workflow testing |
| Template rendering tests | 18 | Report template output verification |
| Integration bridge tests | 20 | MRV ↔ CSRD, Data ↔ CSRD bridges |
| Demo mode E2E | 5 | Full demo mode verification |
| Health check tests | 10 | Pack health verification |
| **Total** | **108** | |

### 6.2 Test Data

- Demo company profile (mid-market manufacturing company)
- 500-record ESG dataset covering all 12 ESRS standards
- Pre-computed expected results for calculation verification
- Sample materiality assessment results

---

## 7. Delivery Milestones

| Phase | Components | Files | Tests |
|-------|-----------|-------|-------|
| Phase 1 | Pack manifest, config system, presets | 12 | 30 |
| Phase 2 | Workflows (5 workflows) | 6 | 25 |
| Phase 3 | Report templates (6 templates) | 7 | 18 |
| Phase 4 | Integration layer (orchestrator, bridges, wizard) | 6 | 20 |
| Phase 5 | Tests & demo mode | 10 | 15 |
| **Total** | | **41** | **108** |

---

## 8. Success Criteria

1. Pack deploys successfully with `greenlang pack install`
2. Demo mode produces a valid CSRD report from sample data
3. All 5 workflows execute successfully end-to-end
4. All 4 configuration presets pass validation
5. All 6 report templates render correctly
6. MRV bridge correctly routes to all 30 calculation engines
7. Data pipeline bridge correctly routes to all 4 intake connectors
8. Health check passes all component verifications
9. 108+ tests pass with 85%+ coverage
10. Full annual report completes in < 30 minutes
