# PRD-PACK-005: CBAM Complete Pack

**Document ID**: PRD-PACK-005
**Version**: 1.1
**Status**: APPROVED & DELIVERED
**Author**: GreenLang Product Team
**Created**: 2026-03-14
**Last Updated**: 2026-03-14
**Approval**: APPROVED

---

## 1. Executive Summary

PACK-005 CBAM Complete Pack is a comprehensive EU Carbon Border Adjustment Mechanism compliance solution that extends PACK-004 CBAM Readiness Pack with full definitive-period capabilities. While PACK-004 provides foundational CBAM compliance (calculate, report, plan), PACK-005 delivers operational CBAM compliance at enterprise scale (trade, submit, optimize, audit).

Key additions over PACK-004:
- **Certificate Trading**: Full buy/hold/surrender/re-sell/cancel lifecycle with portfolio optimization
- **Precursor Chain Resolution**: Multi-tier recursive upstream emissions for complex goods (Article 35)
- **Multi-Entity Consolidation**: Corporate group management with multi-EORI, subsidiary rollup
- **CBAM Registry Integration**: Direct API connectivity for certificate and declaration management
- **Advanced Analytics**: Sourcing optimization, scenario analysis, Monte Carlo cost modeling
- **Customs Automation**: TARIC integration, customs declaration parsing, anti-circumvention detection
- **Cross-Regulation Mapping**: CBAM ↔ CSRD ↔ EU ETS ↔ CDP ↔ SBTi ↔ EU Taxonomy alignment
- **Audit Management**: Evidence chain-of-custody, data rooms, NCA examination readiness

**Regulation**: EU CBAM (EU) 2023/956 + Implementing Regulation (EU) 2023/1773 + December 2025 Implementing/Delegated Acts
**Extends**: PACK-004 CBAM Readiness Pack (62 files, ~39.5K lines, 268 tests)
**Target**: Enterprise importers, customs brokers, multi-entity corporate groups

---

## 2. Background & Motivation

### 2.1 Regulatory Context

The EU CBAM entered its transitional period on October 1, 2023. The definitive period begins January 1, 2026, with the first CBAM certificates required to be purchased and surrendered. Key definitive-period obligations:

1. **Authorized CBAM Declarant** status required (Articles 5-6)
2. **Certificate purchasing** from National Competent Authorities (Article 20-22)
3. **Annual CBAM Declaration** by September 30 (updated from May 31 per Omnibus simplification)
4. **Certificate surrendering** matching embedded emissions of imports (Article 22)
5. **Verification** of actual emissions by accredited verifiers (Article 18)
6. **Penalties** for non-compliance (Articles 26-27): EUR per tCO2e of non-surrendered certificates
7. **Anti-circumvention** measures including downstream product expansion (Dec 2025 proposal)

### 2.2 PACK-004 vs PACK-005 Positioning

| Dimension | PACK-004 Readiness | PACK-005 Complete |
|-----------|-------------------|-------------------|
| Period Focus | Transitional + early definitive | Full definitive period operations |
| Entity Scope | Single importer | Multi-entity corporate groups |
| Submission | Manual XML upload | API-automated Registry submission |
| Certificates | Obligation calculation | Full trading lifecycle |
| Precursors | Single-level flat | Multi-tier recursive chains |
| Analytics | Basic cost projection | Portfolio optimization, Monte Carlo |
| Customs | CN code lookup | TARIC integration, SAD parsing |
| Regulations | CBAM standalone | Cross-regulation alignment |
| Audit | Verification engagement | Enterprise audit management |

### 2.3 Target Users

1. **Large importers** (>1000 tCO2e/year embedded emissions)
2. **Multi-national corporate groups** with imports across multiple EU member states
3. **Customs brokers** managing CBAM for multiple clients (delegated compliance)
4. **Compliance officers** needing cross-regulation visibility
5. **Finance teams** managing CBAM certificate budgets
6. **Procurement teams** optimizing sourcing to minimize CBAM exposure

---

## 3. Goals & Objectives

### 3.1 Primary Goals

1. **Full Definitive Period Compliance**: Cover all Article 20-27 certificate obligations
2. **Enterprise Scale**: Support multi-entity corporate groups with 10+ subsidiaries
3. **Automated Registry Integration**: Zero-manual-touch certificate and declaration management
4. **Cross-Regulation Intelligence**: Single source of truth for CBAM data across 6 regulations
5. **Financial Optimization**: Minimize CBAM certificate costs through portfolio and sourcing optimization

### 3.2 Success Metrics

- Certificate obligation accuracy: 99.9% vs manual calculation
- Automated Registry submission: <30 seconds per declaration
- Cross-regulation data reuse: >80% of CBAM data shared with CSRD/CDP/SBTi
- Total cost of compliance reduction: 40-60% vs manual processes
- Audit preparation time: <2 hours for full evidence assembly

---

## 4. Technical Architecture

### 4.1 Pack Structure

```
PACK-005-cbam-complete/
├── pack.yaml                          # Pack manifest (extends PACK-004)
├── README.md                          # Documentation
├── config/
│   ├── __init__.py
│   ├── pack_config.py                 # CBAMCompleteConfig (extends CBAMPackConfig)
│   ├── presets/
│   │   ├── enterprise_importer.yaml   # Large multi-entity group
│   │   ├── customs_broker.yaml        # Broker managing multiple clients
│   │   ├── steel_group.yaml           # Vertically integrated steel group
│   │   └── multi_commodity_group.yaml # Diverse commodity importer
│   ├── sectors/
│   │   ├── automotive_oem.yaml        # Automotive (steel + aluminium heavy)
│   │   ├── construction.yaml          # Construction (cement + steel)
│   │   └── chemical_manufacturing.yaml # Chemicals (fertilizers + hydrogen)
│   └── demo/
│       ├── demo_config.yaml
│       ├── demo_group_structure.json   # 3-entity group hierarchy
│       └── demo_import_portfolio.csv   # 500-row multi-entity imports
├── engines/
│   ├── __init__.py
│   ├── certificate_trading_engine.py
│   ├── precursor_chain_engine.py
│   ├── multi_entity_engine.py
│   ├── registry_api_engine.py
│   ├── advanced_analytics_engine.py
│   ├── customs_automation_engine.py
│   ├── cross_regulation_engine.py
│   └── audit_management_engine.py
├── workflows/
│   ├── __init__.py
│   ├── certificate_trading.py
│   ├── multi_entity_consolidation.py
│   ├── registry_submission.py
│   ├── cross_regulation_sync.py
│   ├── customs_integration.py
│   └── audit_preparation.py
├── templates/
│   ├── __init__.py
│   ├── certificate_portfolio_report.py
│   ├── group_consolidation_report.py
│   ├── sourcing_scenario_analysis.py
│   ├── cross_regulation_mapping_report.py
│   ├── customs_integration_report.py
│   └── audit_readiness_scorecard.py
├── integrations/
│   ├── __init__.py
│   ├── pack_orchestrator.py
│   ├── registry_client.py
│   ├── taric_client.py
│   ├── ets_registry_bridge.py
│   ├── cross_pack_bridge.py
│   ├── setup_wizard.py
│   └── health_check.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_manifest.py
    ├── test_config.py
    ├── test_certificate_trading.py
    ├── test_precursor_chain.py
    ├── test_multi_entity.py
    ├── test_registry_api.py
    ├── test_analytics.py
    ├── test_customs.py
    ├── test_cross_regulation.py
    ├── test_audit_management.py
    ├── test_workflows.py
    ├── test_templates.py
    ├── test_integrations.py
    ├── test_demo.py
    └── test_e2e.py
```

### 4.2 Dependency Graph

```
PACK-005 CBAM Complete
├── extends: PACK-004 CBAM Readiness (7 engines, 7 workflows, 8 templates)
├── bridges: GL-CBAM-APP v1.1 (3 agents, SDK, backend API)
├── bridges: cbam-pack-mvp (CLI pipeline, factor library)
├── bridges: greenlang/agents/policy/cbam_compliance_agent.py
├── bridges: greenlang/agents/policy/regulatory_mapping_agent.py
├── bridges: greenlang/agents/mrv/industrial/ (steel, cement, aluminium, chemicals)
├── bridges: greenlang/agents/calculation/industry/ (production-route calculators)
├── bridges: greenlang/data/cbam_benchmarks.py
├── optional: PACK-001/002/003 CSRD packs (cross-regulation mapping)
└── optional: GL-Taxonomy-APP, GL-CDP-APP, GL-SBTi-APP (cross-regulation)
```

---

## 5. Engine Specifications

### 5.1 Certificate Trading Engine (`certificate_trading_engine.py`)

**Purpose**: Full CBAM certificate lifecycle management per Articles 20-25.

**Capabilities**:
- **Portfolio Management**: Track purchased, held, surrendered, re-sold, cancelled, expired certificates
- **Purchase Orders**: Create purchase orders with quantity, timing, price limits; auto-execute at market price
- **Price Integration**: Weekly EU ETS auction clearing price per Article 22(1); forward curve modeling
- **Repurchase Facility**: Sell-back up to 1/3 of certificates within 12 months (Article 23)
- **Expiry Management**: Auto-flag certificates expiring within 90/60/30 days (2-year validity)
- **Inventory Valuation**: FIFO, weighted average cost, mark-to-market methods
- **Quarterly Holding Check**: Verify 50% target holding per Omnibus simplified rule
- **Budget Integration**: Certificate cost allocation to cost centers, business units, products
- **Intra-Group Transfer**: Certificate transfer between entities in same corporate group
- **Surrender Optimization**: Minimize cost by surrendering lowest-cost certificates first (FIFO/LIFO configurable)

**Data Model**:
```python
class CertificatePortfolio(BaseModel):
    entity_id: str
    certificates: List[Certificate]
    total_purchased: int
    total_surrendered: int
    total_resold: int
    total_cancelled: int
    current_balance: int
    weighted_avg_cost: Decimal
    mark_to_market_value: Decimal

class Certificate(BaseModel):
    certificate_id: str
    purchase_date: date
    purchase_price: Decimal  # EUR per tCO2e
    quantity: int  # number of certificates (1 cert = 1 tCO2e)
    status: CertificateStatus  # PURCHASED, HELD, SURRENDERED, RESOLD, CANCELLED, EXPIRED
    expiry_date: date
    surrender_date: Optional[date]
    linked_declaration_id: Optional[str]

class PurchaseOrder(BaseModel):
    order_id: str
    entity_id: str
    quantity: int
    price_type: PriceType  # MARKET, LIMIT, TRAILING_STOP
    limit_price: Optional[Decimal]
    execution_window: Tuple[date, date]
    status: OrderStatus
```

### 5.2 Precursor Chain Engine (`precursor_chain_engine.py`)

**Purpose**: Multi-tier recursive precursor emission resolution per Article 35 and Annex III.

**Capabilities**:
- **Recursive Resolution**: Resolve precursor chains up to 5 tiers deep (e.g., bauxite → alumina → primary aluminium → rolled sheet → fabricated part)
- **Allocation Methods**: Mass-based, economic, energy-based allocation for multi-product installations
- **Composition Tracking**: Track % composition of input materials (e.g., 70% primary + 30% secondary aluminium)
- **Default Fallback Waterfall**: Actual verified → Supplier default → Country default → EU Commission default → Maximum observed penalty default
- **Installation Attribution**: Link each precursor tier to specific supplier installations
- **Gap Analysis**: Identify missing precursor data and estimate impact of gaps
- **Production Route Mapping**: BF-BOF vs EAF (steel), dry vs wet (cement), Bayer+Hall-Héroult (aluminium)
- **Scrap Classification**: Pre-consumer vs post-consumer scrap per Dec 2025 anti-circumvention rules
- **Mass Balance Validation**: Verify input/output mass balance at each production tier

**Precursor Chains by Goods Category**:
- **Iron/Steel**: Iron ore → pig iron → crude steel → hot-rolled → cold-rolled → coated/fabricated
- **Aluminium**: Bauxite → alumina → primary aluminium → alloy → rolled/extruded → fabricated
- **Cement**: Limestone → clinker → cement (CEM I-V with varying clinker ratios)
- **Fertilizers**: Natural gas → ammonia → urea/AN/CAN; nitric acid → nitrates
- **Hydrogen**: Natural gas → grey H2; NG+CCS → blue H2; electricity → green H2

### 5.3 Multi-Entity Consolidation Engine (`multi_entity_engine.py`)

**Purpose**: Corporate group CBAM management with multi-EORI support.

**Capabilities**:
- **Entity Hierarchy**: Parent → subsidiary → customs declarant tree with unlimited depth
- **Multi-EORI Management**: Track multiple EORI numbers across EU member states
- **Consolidated Obligation**: Aggregate certificate obligations across all group entities
- **Entity-Level Declarations**: Generate per-entity annual declarations with group context
- **De Minimis Aggregation**: Cross-entity de minimis threshold assessment (group-level 50-tonne check)
- **Cost Allocation**: Allocate CBAM certificate costs to entities using configurable methods (volume, revenue, profit center)
- **Delegated Compliance**: Customs broker mode — manage CBAM for external client entities
- **Transfer Pricing**: Internal CBAM cost transfer between group entities
- **Member State Coordination**: Track which NCA handles which entity's obligations
- **Financial Guarantee Pooling**: Group-level financial guarantee management

**Data Model**:
```python
class EntityGroup(BaseModel):
    group_id: str
    group_name: str
    parent_entity: Entity
    subsidiaries: List[Entity]
    cost_allocation_method: CostAllocationMethod
    financial_guarantee: FinancialGuarantee
    consolidation_currency: str  # EUR

class Entity(BaseModel):
    entity_id: str
    legal_name: str
    eori_number: str
    member_state: str  # ISO 3166-1 alpha-2
    nca_authority: str
    declarant_status: DeclarantStatus
    role: EntityRole  # PARENT, SUBSIDIARY, CUSTOMS_REPRESENTATIVE
    imports: List[ImportRecord]
    certificate_account: CertificatePortfolio
```

### 5.4 Registry API Engine (`registry_api_engine.py`)

**Purpose**: Direct integration with EU CBAM Registry (transitional + definitive portals).

**Capabilities**:
- **Transitional Registry**: Submit/amend quarterly reports via XML upload API
- **Definitive Registry**: Certificate purchase, surrender, re-sell via Registry API
- **Authorisation Management Module (AMM)**: Declarant application status tracking
- **Third-Country Installation Registration**: Track operator registrations in CBAM Registry
- **Authentication**: eIDAS certificate-based authentication, OAuth 2.0 fallback
- **Submission Lifecycle**: Draft → Validated → Submitted → Accepted/Rejected → Amended
- **Status Polling**: Automated status checks with configurable intervals
- **Error Handling**: Structured error parsing, retry with exponential backoff
- **Sandbox Support**: EU CBAM Registry test/sandbox environment for development
- **Response Parsing**: Parse Registry confirmations, receipts, error details
- **Audit Logging**: Log all Registry interactions for compliance trail

**API Endpoints Modeled**:
```
POST   /api/v1/declarations            # Submit annual declaration
PUT    /api/v1/declarations/{id}        # Amend declaration
GET    /api/v1/declarations/{id}/status # Check submission status
POST   /api/v1/certificates/purchase    # Purchase certificates
POST   /api/v1/certificates/surrender   # Surrender certificates
POST   /api/v1/certificates/resell      # Re-sell certificates
GET    /api/v1/certificates/balance     # Check certificate balance
GET    /api/v1/price/current            # Current certificate price
POST   /api/v1/installations/register   # Register third-country installation
GET    /api/v1/declarant/status         # Declarant authorization status
```

### 5.5 Advanced Analytics Engine (`advanced_analytics_engine.py`)

**Purpose**: Strategic intelligence for CBAM cost optimization and procurement decision support.

**Capabilities**:
- **Sourcing Optimization**: LP/MILP solver recommending supplier/country mix to minimize CBAM cost subject to volume, quality, and lead time constraints
- **Scenario Analysis**: What-if modeling for supplier switching, volume changes, ETS price movements
- **Monte Carlo Simulation**: Certificate cost distributions with 10,000+ iterations
- **Carbon Price Modeling**: ETS forward curve integration (ICE EUA futures), mean reversion, GBM models
- **Free Allocation Impact**: Year-by-year cost escalation as free allocation declines 2026-2034
- **Decarbonization ROI**: Model cost savings if supplier X reduces emissions by Y%
- **Peer Benchmarking**: Compare CBAM exposure vs industry averages (emission intensity, cost per unit)
- **Procurement TCO**: Total cost of ownership including CBAM certificates for sourcing decisions
- **Budget Forecasting**: Multi-year CBAM budget projections with confidence intervals
- **Sensitivity Analysis**: Identify which variables (price, volume, EF, country) drive most cost variation

**Output Models**:
```python
class SourcingScenario(BaseModel):
    scenario_name: str
    supplier_mix: Dict[str, float]  # supplier_id → % allocation
    total_embedded_emissions: Decimal  # tCO2e
    total_certificate_cost: Decimal  # EUR
    cost_per_unit: Decimal
    savings_vs_current: Decimal
    risk_score: float  # 0-100

class MonteCarloResult(BaseModel):
    iterations: int
    mean_cost: Decimal
    median_cost: Decimal
    p5_cost: Decimal   # 5th percentile
    p95_cost: Decimal  # 95th percentile
    std_dev: Decimal
    distribution: List[Decimal]
```

### 5.6 Customs Automation Engine (`customs_automation_engine.py`)

**Purpose**: Customs system integration and anti-circumvention detection.

**Capabilities**:
- **TARIC Integration**: Real-time CN code validation against EU TARIC database
- **CN Code Versioning**: Annual CN nomenclature updates with backward compatibility
- **Customs Declaration Parsing**: Extract CBAM-relevant data from SAD (Single Administrative Document)
- **AEO Status Checking**: Verify Authorized Economic Operator status
- **Import Procedure Tracking**: Inward processing, customs warehousing, free zone applicability
- **Multi-Port Coordination**: Track imports across multiple EU entry points
- **Anti-Circumvention Detection**: Flag suspicious patterns:
  - Sudden country-of-origin changes
  - CN code reclassification attempts
  - Unusual scrap ratios
  - Supply chain restructuring indicators
  - Minor processing avoidance patterns
- **Downstream Product Monitoring**: Track 180+ downstream products proposed for 2028 expansion
- **EORI Validation**: Real-time EORI number validation via EU VIES/EOS
- **Duty/CBAM Combined Calculation**: Total import cost including customs duty + CBAM certificates

### 5.7 Cross-Regulation Engine (`cross_regulation_engine.py`)

**Purpose**: Map CBAM data to 6 related regulatory frameworks for data reuse optimization.

**Capabilities**:
- **CBAM → CSRD**: Map embedded emissions to ESRS E1 Climate Change disclosures; Scope 3 Category 1 (purchased goods) using CBAM supplier-specific emission factors
- **CBAM → EU ETS**: Cross-reference free allocation benchmarks; verify phase-out consistency; benchmark comparison (CBAM default vs ETS benchmark)
- **CBAM → CDP**: Populate CDP Climate Change Questionnaire sections on carbon pricing, Scope 3 upstream, supplier engagement
- **CBAM → SBTi**: Feed CBAM emission intensities into SBTi Scope 3 Category 1 target tracking; decarbonization pathway alignment
- **CBAM → EU Taxonomy**: Map CBAM-covered activities to Taxonomy climate mitigation criteria; substantial contribution thresholds for CBAM sectors
- **CBAM → EUDR**: Trace fertilizer supply chains for deforestation-free commodity verification
- **Data Reuse Optimization**: Identify which CBAM data points serve multiple regulations; single-entry, multi-regulation output
- **Third-Country Carbon Pricing**: Comprehensive equivalence database for 50+ countries' carbon pricing schemes (ETS, carbon tax, CORSIA)
- **Regulatory Change Monitor**: Track proposed amendments across all 6 regulations

**Mapping Matrix**:
```
CBAM Data Point          → CSRD  → CDP  → SBTi → Taxonomy → ETS  → EUDR
─────────────────────────────────────────────────────────────────────────
Embedded emissions/t     → E1-4  → C6.1 → S3.1 → DNSH     → Bench → n/a
Supplier EF (tCO2/t)    → E1-6  → C7.3 → S3.1 → TSC      → n/a   → n/a
Country of origin        → E1-9  → C7.6 → n/a  → n/a      → n/a   → DDS
Certificate cost (EUR)   → E1-3  → C11  → n/a  → CapEx    → n/a   → n/a
Free alloc phase-out     → E1-3  → C11  → n/a  → n/a      → FAL   → n/a
Supplier engagement      → S2-4  → C12  → S3T  → n/a      → n/a   → n/a
Verification status      → E1-9  → C10  → n/a  → Art.8    → MRV   → n/a
```

### 5.8 Audit Management Engine (`audit_management_engine.py`)

**Purpose**: Enterprise-grade audit trail and NCA examination readiness.

**Capabilities**:
- **Multi-Year Audit Repository**: Searchable archive of all CBAM declarations, certificates, verifications across reporting periods
- **Evidence Chain-of-Custody**: Track who accessed, modified, exported evidence; immutable audit log
- **Data Room Management**: Secure controlled-access data rooms for verifiers and NCA inspectors
- **Remediation Tracking**: Corrective Action Plans (CAPs) with deadlines, assignees, status, escalation
- **NCA Examination Support**: Pre-packaged evidence bundles matching NCA request templates
- **Continuous Monitoring**: Real-time anomaly detection on emission patterns (sudden changes, outliers)
- **Penalty Risk Assessment**: Calculate potential penalty exposure based on compliance gaps
- **Audit Committee Reporting**: Board-level summary of CBAM audit status, findings, remediation
- **Verifier Accreditation Validation**: Check verifier credentials against EU published accreditation lists
- **Regulatory Correspondence Log**: Track all NCA/competent authority communications with response deadlines

---

## 6. Workflow Specifications

### 6.1 Certificate Trading Workflow (`certificate_trading.py`)

**Phases** (6-phase weekly cycle):
1. **Price Monitor**: Fetch current EU ETS clearing price; compare to budget thresholds
2. **Obligation Forecast**: Calculate upcoming certificate needs based on import pipeline
3. **Purchase Decision**: Apply buying strategy (budget-paced, price-triggered, bulk quarterly)
4. **Order Execution**: Submit purchase orders to Registry API
5. **Portfolio Rebalance**: Check expiry, surplus, deficit; trigger re-sell or additional purchases
6. **Reporting**: Update certificate dashboard, notify finance team

### 6.2 Multi-Entity Consolidation Workflow (`multi_entity_consolidation.py`)

**Phases** (5-phase quarterly):
1. **Data Collection**: Gather import data from all group entities across member states
2. **Entity-Level Calculation**: Run PACK-004 calculation engine per entity
3. **Group Aggregation**: Consolidate obligations, apply group-level de minimis, cross-entity netting
4. **Cost Allocation**: Distribute CBAM costs to entities per configured method
5. **Reporting**: Generate group consolidation report + per-entity declarations

### 6.3 Registry Submission Workflow (`registry_submission.py`)

**Phases** (4-phase per submission):
1. **Pre-Validation**: Validate declaration/report against definitive-period XML schema
2. **Submit**: POST to CBAM Registry API with eIDAS authentication
3. **Monitor**: Poll status until Accepted/Rejected; handle validation errors
4. **Confirm**: Log acceptance receipt, trigger next workflow steps

### 6.4 Cross-Regulation Sync Workflow (`cross_regulation_sync.py`)

**Phases** (4-phase triggered by data changes):
1. **Change Detection**: Identify new/modified CBAM data since last sync
2. **Mapping**: Apply regulation-specific mapping rules (CSRD/CDP/SBTi/Taxonomy/ETS)
3. **Output Generation**: Generate regulation-specific data exports
4. **Notification**: Alert compliance teams of updated cross-regulation data

### 6.5 Customs Integration Workflow (`customs_integration.py`)

**Phases** (3-phase per import declaration):
1. **Intake**: Parse customs declaration (SAD/CDS) to extract CN codes, quantities, origins
2. **Enrichment**: Validate CN codes via TARIC, check CBAM applicability, flag anti-circumvention risks
3. **CBAM Linkage**: Create CBAM import records, calculate preliminary embedded emissions

### 6.6 Audit Preparation Workflow (`audit_preparation.py`)

**Phases** (5-phase annual):
1. **Completeness Scan**: Check all required evidence exists for reporting period
2. **Gap Analysis**: Identify missing verifications, incomplete supplier data, unresolved findings
3. **Evidence Assembly**: Package evidence into structured data room
4. **Quality Review**: Cross-check consistency of declarations vs certificates vs customs records
5. **Readiness Score**: Generate audit readiness scorecard with actionable gaps

---

## 7. Template Specifications

### 7.1 Certificate Portfolio Report (`certificate_portfolio_report.py`)

- Holdings summary: purchased, held, surrendered, re-sold, cancelled, expired
- Cost analysis: weighted average cost, mark-to-market, unrealized P&L
- Expiry timeline: certificates expiring in 30/60/90/180 days
- Budget vs actual: certificate spend against budget
- Quarterly holding compliance: 50% threshold check

### 7.2 Group Consolidation Report (`group_consolidation_report.py`)

- Entity hierarchy visualization
- Per-entity import volumes and embedded emissions
- Consolidated group obligation
- Cost allocation breakdown by entity
- De minimis status per entity and at group level

### 7.3 Sourcing Scenario Analysis (`sourcing_scenario_analysis.py`)

- Current sourcing profile (suppliers, countries, emission intensities)
- Alternative scenarios with cost/emission impact
- Monte Carlo confidence intervals
- Recommended actions with expected savings
- Sensitivity analysis charts (tornado diagram)

### 7.4 Cross-Regulation Mapping Report (`cross_regulation_mapping_report.py`)

- Data flow diagram: CBAM → CSRD/CDP/SBTi/Taxonomy/ETS
- Data reuse statistics: how many data points serve multiple regulations
- Consistency checks: flag conflicting values across regulations
- Gap analysis: CBAM data not yet mapped to target regulations

### 7.5 Customs Integration Report (`customs_integration_report.py`)

- SAD-to-CBAM reconciliation by customs office
- CN code validation results (TARIC cross-reference)
- Anti-circumvention risk flags
- Import procedure summary (standard, inward processing, warehousing)
- Duty + CBAM combined cost per shipment

### 7.6 Audit Readiness Scorecard (`audit_readiness_scorecard.py`)

- Overall readiness score (0-100)
- Evidence completeness by category
- Unresolved findings and CAP status
- Verifier engagement status
- NCA correspondence log summary
- Risk areas requiring attention before examination

---

## 8. Integration Specifications

### 8.1 Pack Orchestrator (`pack_orchestrator.py`)

Extends PACK-004 orchestrator with 10-phase CBAM Complete execution pipeline:
1. Health Check (PACK-004 base + PACK-005 components)
2. Configuration Loading (merge PACK-004 + PACK-005 configs)
3. Import Data Intake (customs integration workflow)
4. Emission Calculations (PACK-004 engine + precursor chain resolution)
5. Certificate Obligation (PACK-004 engine + multi-entity consolidation)
6. Certificate Trading (PACK-005 trading workflow)
7. Registry Submission (PACK-005 submission workflow)
8. Cross-Regulation Sync (PACK-005 mapping workflow)
9. Audit Trail Update (PACK-005 audit management)
10. Reporting (all templates, dashboards, exports)

### 8.2 Registry Client (`registry_client.py`)

HTTP client for EU CBAM Registry APIs:
- REST client with aiohttp for async requests
- eIDAS certificate-based mutual TLS authentication
- OAuth 2.0 token management with refresh
- Request/response logging for audit trail
- Retry with exponential backoff (3 attempts, 1s/2s/4s)
- Sandbox/production environment switching
- Response schema validation
- Rate limiting compliance

### 8.3 TARIC Client (`taric_client.py`)

EU TARIC database integration:
- REST API client for TARIC consultation
- CN code hierarchy navigation (2/4/6/8 digit)
- Tariff measure lookup (duty rates, quotas)
- CBAM applicability determination per CN code
- Annual nomenclature update tracking
- Offline fallback with local CN code cache
- Downstream product monitoring (2028 expansion list)

### 8.4 ETS Registry Bridge (`ets_registry_bridge.py`)

EU ETS Union Registry (EUTL) data integration:
- Free allocation data by installation
- Benchmark values per product
- Compliance status of covered installations
- Cross-reference CBAM imports with ETS installations
- Phase-out consistency verification

### 8.5 Cross-Pack Bridge (`cross_pack_bridge.py`)

Bridge to other GreenLang Solution Packs (graceful degradation if not installed):
- PACK-001/002/003 CSRD: Push CBAM data to ESRS E1 disclosures
- GL-CDP-APP: Push CBAM data to CDP Climate sections
- GL-SBTi-APP: Push CBAM emission intensities to Scope 3 tracking
- GL-Taxonomy-APP: Push CBAM sector data to Taxonomy alignment
- GL-EUDR-APP: Link fertilizer CBAM with deforestation-free supply chains

### 8.6 Setup Wizard (`setup_wizard.py`)

10-step interactive setup for CBAM Complete:
1. Import PACK-004 configuration (or start fresh)
2. Configure entity group hierarchy
3. Set up CBAM Registry API credentials (sandbox first)
4. Configure TARIC API access
5. Set certificate trading strategy and budget
6. Map cross-regulation connections
7. Configure customs system integration
8. Set audit management preferences
9. Run demo with sample data (500-row portfolio)
10. Health check and readiness verification

### 8.7 Health Check (`health_check.py`)

18-category health verification:
1. PACK-004 Base (all 12 PACK-004 categories)
2. Certificate Trading Engine
3. Precursor Chain Engine
4. Multi-Entity Engine
5. Registry API Connectivity
6. TARIC API Connectivity
7. ETS Data Feed
8. Cross-Pack Bridges
9. Analytics Engine
10. Customs Automation
11. Cross-Regulation Mappings
12. Audit Management
13. Configuration Completeness
14. Demo Data Availability
15. Template Registry
16. Workflow Engine
17. Entity Group Integrity
18. Certificate Portfolio Consistency

---

## 9. Configuration Model

### 9.1 CBAMCompleteConfig

```python
class CBAMCompleteConfig(CBAMPackConfig):  # Extends PACK-004
    """PACK-005 CBAM Complete Pack configuration."""

    # Certificate Trading
    trading: CertificateTradingConfig

    # Multi-Entity
    entity_group: EntityGroupConfig

    # Registry API
    registry: RegistryAPIConfig

    # Advanced Analytics
    analytics: AdvancedAnalyticsConfig

    # Customs Automation
    customs: CustomsAutomationConfig

    # Cross-Regulation
    cross_regulation: CrossRegulationConfig

    # Audit Management
    audit: AuditManagementConfig

    # Precursor Chains
    precursor: PrecursorChainConfig
```

### 9.2 Sub-Configurations

- **CertificateTradingConfig**: buying_strategy, price_alerts, repurchase_threshold, valuation_method, budget_limit, auto_purchase_enabled
- **EntityGroupConfig**: entities, hierarchy, cost_allocation_method, consolidation_currency, delegated_compliance_mode
- **RegistryAPIConfig**: base_url, auth_type, certificate_path, sandbox_mode, polling_interval_seconds, max_retries
- **AdvancedAnalyticsConfig**: monte_carlo_iterations, optimization_solver, benchmark_source, forecast_horizon_years
- **CustomsAutomationConfig**: taric_api_url, sad_format, aeo_check_enabled, anti_circumvention_rules, downstream_monitoring
- **CrossRegulationConfig**: csrd_enabled, cdp_enabled, sbti_enabled, taxonomy_enabled, ets_enabled, eudr_enabled, sync_frequency
- **AuditManagementConfig**: retention_years, data_room_enabled, auto_remediation_alerts, nca_response_deadline_days
- **PrecursorChainConfig**: max_chain_depth, allocation_method, default_fallback_waterfall, scrap_classification_enabled

---

## 10. Presets

### 10.1 Enterprise Importer (`enterprise_importer.yaml`)
Multi-entity group with 5+ subsidiaries, all 6 goods categories, full certificate trading, Registry API enabled, cross-regulation sync to CSRD/CDP/SBTi.

### 10.2 Customs Broker (`customs_broker.yaml`)
Delegated compliance mode, multi-client entity management, customs declaration parsing, TARIC integration, white-label reporting per client.

### 10.3 Steel Group (`steel_group.yaml`)
Vertically integrated steel importer, deep precursor chains (ore → pig iron → crude steel → rolled → coated), BF-BOF and EAF production routes, scrap classification.

### 10.4 Multi-Commodity Group (`multi_commodity_group.yaml`)
Diverse commodity importer (steel + aluminium + fertilizers), cross-commodity precursor tracking, portfolio optimization across goods categories.

### 10.5 Automotive OEM (`automotive_oem.yaml`)
Steel and aluminium heavy, downstream product monitoring (2028 expansion), multi-tier supplier engagement, decarbonization ROI modeling.

### 10.6 Construction (`construction.yaml`)
Cement and steel focus, clinker ratio optimization, project-level CBAM cost allocation, regional sourcing analysis.

### 10.7 Chemical Manufacturing (`chemical_manufacturing.yaml`)
Fertilizers and hydrogen, ammonia production chains, N2O monitoring, green hydrogen pathway tracking, EUDR cross-regulation for agricultural inputs.

---

## 11. Test Plan

### 11.1 Test Categories

| Category | Tests | Focus |
|----------|-------|-------|
| Manifest | 15 | pack.yaml validation, PACK-004 extension, component listing |
| Config | 50 | CBAMCompleteConfig, all sub-configs, presets, validation |
| Certificate Trading | 30 | Portfolio, orders, repurchase, expiry, valuation, surrender |
| Precursor Chain | 25 | Recursive resolution, allocation, composition, gap analysis |
| Multi-Entity | 25 | Hierarchy, consolidation, cost allocation, de minimis |
| Registry API | 20 | Submission lifecycle, authentication, polling, error handling |
| Analytics | 25 | Monte Carlo, optimization, scenarios, benchmarking |
| Customs | 20 | TARIC, SAD parsing, anti-circumvention, CN versioning |
| Cross-Regulation | 20 | All 6 regulation mappings, data reuse, consistency |
| Audit Management | 20 | Repository, chain-of-custody, data room, penalties |
| Workflows | 30 | All 6 workflows end-to-end |
| Templates | 25 | All 6 templates render correctly in md/html/json |
| Integrations | 25 | All 7 integrations bridge correctly |
| Demo | 10 | Setup wizard, demo data, demo execution |
| E2E | 15 | Full pipeline from customs intake to cross-regulation output |
| **Total** | **355** | |

### 11.2 Test Patterns

- All tests use `sys.path.insert(0, os.path.dirname(__file__))` for conftest imports
- Pydantic BaseModel for all test fixtures
- SHA-256 provenance hashing on all outputs
- Deterministic calculations (Decimal arithmetic, no floats for money)
- No external API dependencies (all mocked/stubbed)
- conftest.py provides shared fixtures: sample entity groups, certificate portfolios, customs declarations

---

## 12. Assets Leveraged

### 12.1 PACK-004 CBAM Readiness (Direct Extension)
- 7 engines (calculation, certificate, quarterly, supplier, deminimis, verification, policy)
- 7 workflows
- 8 templates
- 6 integrations
- All 268 tests

### 12.2 GL-CBAM-APP v1.1 (Bridged)
- ShipmentIntakeAgent v2 (customs data intake)
- EmissionsCalculatorAgent v2 (production-route calculations)
- ReportingPackagerAgent v2 (XML generation)
- CBAM SDK (client library)
- 340+ tests

### 12.3 Industrial MRV Agents (Bridged)
- Steel MRV (BF-BOF, EAF, DRI pathways)
- Cement MRV (clinker ratios, kiln fuels)
- Aluminium MRV (smelting, PFC, Söderberg/prebake)
- Chemicals MRV (ammonia, urea, nitric acid, N2O)

### 12.4 Policy Agents (Bridged)
- CBAM Compliance Agent (695 lines, classification, sector mapping)
- Regulatory Mapping Agent (cross-regulation applicability)
- Compliance Gap Analyzer (readiness assessment)

### 12.5 Data Assets (Referenced)
- cbam_benchmarks.py (12 product benchmarks, 259 lines)
- cbam_defaults_2024.json (country-specific emission factors)
- cn_codes.json (CN code database)
- emission_factors.py (14 factor variants)

---

## 13. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| EU CBAM Registry API not yet stable | Cannot test live submission | Sandbox environment + comprehensive mocks |
| TARIC API access restrictions | Cannot validate CN codes in real-time | Local CN code cache with periodic sync |
| December 2025 regulation changes still being finalized | Feature scope may shift | Modular engine design; configuration-driven rule updates |
| Multi-entity complexity high | Testing combinatorial explosion | Preset-based testing; focus on 3-5 common entity structures |
| Cross-regulation packs may not be installed | Bridge failures | Graceful degradation with informative messages |
| ETS price feed availability | Mock data may diverge from reality | Configurable price source (mock, API, manual) |

---

## 14. Build Results

**Status**: COMPLETE - All components built and tested
**Date**: 2026-03-14
**Total Files**: 100
**Total Lines**: ~44,770

### 14.1 File Inventory

| Layer | Files | Lines | Key Components |
|-------|-------|-------|----------------|
| Root | 2 | ~2,000 | pack.yaml, README.md |
| Config | 14 | ~6,500 | pack_config.py (2,500+), 4 presets, 3 sectors, 3 demo files |
| Engines | 9 | ~9,775 | 8 engines + __init__.py |
| Workflows | 7 | ~8,883 | 6 workflows + __init__.py |
| Templates | 7 | ~6,962 | 6 templates + __init__.py |
| Integrations | 8 | ~8,000 | 7 integrations + __init__.py |
| Tests | 17 | ~5,500 | 15 test files + conftest.py + __init__.py |

### 14.2 Test Results

```
367 passed in 1.56s
```

| Test File | Tests | Status |
|-----------|-------|--------|
| test_manifest.py | 15 | PASS |
| test_config.py | 50 | PASS |
| test_certificate_trading.py | 30 | PASS |
| test_precursor_chain.py | 25 | PASS |
| test_multi_entity.py | 25 | PASS |
| test_registry_api.py | 20 | PASS |
| test_analytics.py | 25 | PASS |
| test_customs.py | 20 | PASS |
| test_cross_regulation.py | 20 | PASS |
| test_audit_management.py | 20 | PASS |
| test_workflows.py | 42 | PASS |
| test_templates.py | 25 | PASS |
| test_integrations.py | 25 | PASS |
| test_demo.py | 10 | PASS |
| test_e2e.py | 15 | PASS |
| **Total** | **367** | **ALL PASS** |

---

## Appendix A: CBAM Regulatory Timeline

| Date | Event |
|------|-------|
| 2023-10-01 | Transitional period begins |
| 2025-01-01 | First transitional reports due |
| 2025-12-31 | Transitional period ends |
| 2026-01-01 | Definitive period begins; authorized declarants required |
| 2026-01-01 | First certificate purchases (Q1 2026 imports) |
| 2026-09-30 | First annual CBAM declaration due (updated from May 31) |
| 2026-12-31 | 97.5% free allocation |
| 2027-12-31 | 95.0% free allocation |
| 2028-01-01 | Downstream product expansion (proposed) |
| 2028-12-31 | 90.0% free allocation |
| 2029-12-31 | 77.5% free allocation |
| 2030-12-31 | 51.5% free allocation |
| 2031-12-31 | 39.0% free allocation |
| 2032-12-31 | 26.5% free allocation |
| 2033-12-31 | 14.0% free allocation |
| 2034-12-31 | 0% free allocation (CBAM fully operational) |

## Appendix B: CN Code Coverage

PACK-005 targets full Annex I coverage:
- **Cement**: 2523 10 00, 2523 21 00, 2523 29 00, 2523 90 00 (4 codes)
- **Electricity**: 2716 00 00 (1 code)
- **Fertilizers**: 2808 00 00, 2814 10 00, 2814 20 00, 3102 10 xx, 3102 30 xx, 3102 40 xx, 3102 50 xx, 3102 60 xx, 3102 80 00, 3102 90 xx, 3105 10 xx, 3105 20 xx, 3105 30 xx, 3105 40 xx, 3105 51 00, 3105 59 00, 3105 90 xx (25+ codes)
- **Iron & Steel**: 7201-7229, 7301-7326 (100+ codes)
- **Aluminium**: 7601-7616 (30+ codes)
- **Hydrogen**: 2804 10 00 (1 code)
- **Total**: 160+ CN codes (expanded from 107 in PACK-004)
