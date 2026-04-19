# PACK-026 SME Net Zero Pack -- Changelog

All notable changes to PACK-026 SME Net Zero Pack will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.0.0] - 2026-03-18

### Summary

Initial release of the SME Net Zero Pack -- purpose-built for Small and Medium Enterprises with fewer than 250 employees. Provides a complete net-zero journey from 15-minute express onboarding through baseline calculation, target setting, quick wins identification, grant matching, certification readiness, and annual progress tracking.

**Key Numbers:**
- 67 files, ~43,500 lines of code
- 738 tests, 100% pass rate
- 91.2% code coverage
- 8 engines, 6 workflows, 8 templates, 13 integrations, 6 presets

### Added

#### Engines (8)

- **SME Baseline Engine** (`sme_baseline_engine.py`)
  - Three-tier data approach: Bronze (industry averages), Silver (utility bills), Gold (detailed activity data)
  - Bronze baseline from 5 data points in under 2 seconds
  - DEFRA 2024 and IEA 2024 emission factors (deterministic, no LLM)
  - Scope 1 (gas, fuel, refrigerants), Scope 2 (electricity), Scope 3 (Cat 1, 5, 6, 7)
  - Industry peer comparison by NACE sector code
  - Data quality scoring with confidence intervals
  - SHA-256 provenance hashing on all calculation outputs

- **Simplified Target Engine** (`simplified_target_engine.py`)
  - SBTi SME Pathway aligned targets (50% reduction by 2030)
  - SME Climate Hub 1-2-3 pledge targets (halve by 2030, net zero by 2050)
  - Custom reduction targets (30-50% by 2030, configurable)
  - Annual linear reduction milestones with pathway points
  - Automatic feasibility assessment based on identified quick wins
  - SBTi SME Route auto-validation compliance check

- **Quick Wins Engine** (`quick_wins_engine.py`)
  - Database of 500+ SME-specific decarbonization actions
  - Categories: lighting, HVAC, renewable energy, transport, waste, IT, procurement, behaviour
  - Each action tagged with cost range, annual savings, payback period, tCO2e reduction, difficulty
  - Sector and size-tier filtering (micro/small/medium)
  - Landlord/tenant premise type filtering
  - Budget-constrained action selection
  - ROI-ranked output with implementation sequence

- **Action Prioritization Engine** (`action_prioritization_engine.py`)
  - Composite scoring: ROI (30%), payback (25%), ease (20%), CO2e (15%), co-benefits (10%)
  - SME-specific constraints: budget limits, staff availability, premises type
  - Implementation sequencing with dependency management
  - Cumulative impact projection (tCO2e and cost savings over time)

- **SME Progress Tracker** (`sme_progress_tracker.py`)
  - 8 core KPIs: total emissions, change vs baseline, change vs prior year, per-employee intensity, per-revenue intensity, energy consumption, renewable share, actions completed
  - RAG status (Green/Amber/Red) with 5 percentage-point thresholds
  - Corrective action triggers for off-track metrics
  - Year-over-year trend analysis
  - Growth-adjusted intensity comparison

- **Cost-Benefit Engine** (`cost_benefit_engine.py`)
  - NPV calculation over 5/10/15-year horizons
  - IRR calculation for each action
  - Simple and discounted payback periods
  - Net-of-grant cost calculation (factors in available subsidies)
  - SME-appropriate discount rates (8-12%)
  - Total Cost of Ownership (TCO) analysis
  - Carbon price impact at shadow carbon prices

- **Grant Finder Engine** (`grant_finder_engine.py`)
  - Database of 50+ grant, subsidy, tax incentive, and loan programs
  - Coverage: UK (IETF, BUS, ECA, Salix), EU (Cohesion, LIFE, Horizon), US (SBIR, REAP, IRA)
  - Profile-based matching: country, region, sector, size, planned actions
  - Match score (0-100) with eligibility assessment
  - Application deadline tracking and reminder triggers
  - Pre-filled application template guidance
  - Monthly database sync from government sources

- **Certification Readiness Engine** (`certification_readiness_engine.py`)
  - SME Climate Hub: 5-criterion assessment (pledge, measure, reduce, report, offset)
  - B Corp Climate: climate-specific B Impact Assessment scoring
  - ISO 14001: EMS documentation requirements assessment
  - Carbon Trust Standard: measurement, management, reduction evidence check
  - Per-certification readiness score (0-100) with gap list
  - Estimated time-to-certification and cost
  - Recommended certification sequence based on profile

#### Workflows (6)

- **Express Onboarding Workflow** (`express_onboarding_workflow.py`)
  - 4-phase workflow completing in 15-20 minutes
  - Phase 1: Organization Profile (5 min -- collect company info)
  - Phase 2: Quick Baseline (5 min -- Bronze tier from spend + headcount)
  - Phase 3: Auto-Target (instant -- 1.5C-aligned SBTi SME pathway)
  - Phase 4: Quick Wins (5 min -- top 5 actions ranked by ROI)
  - Full provenance hashing and audit trail
  - Mobile-friendly phase summaries

- **Standard Setup Workflow** (`standard_setup_workflow.py`)
  - 6-phase workflow for detailed Silver/Gold baseline (1-2 hours)
  - Includes accounting software connection and auto-classification
  - Detailed Scope 3 estimation with spend category mapping
  - Enhanced peer benchmarking with percentile ranking

- **Quick Wins Implementation Workflow** (`quick_wins_workflow.py`)
  - 3-phase action planning and tracking workflow (30 min)
  - Phase 1: Action selection (from prioritized list)
  - Phase 2: Implementation planning (timelines, responsibilities)
  - Phase 3: Progress tracking (status updates, savings validation)

- **Grant Application Workflow** (`grant_application_workflow.py`)
  - 4-phase grant discovery and application workflow (45 min)
  - Phase 1: Profile matching (auto-match to eligible programs)
  - Phase 2: Eligibility verification (detailed criteria check)
  - Phase 3: Application preparation (pre-filled templates)
  - Phase 4: Submission tracking (deadline reminders, status updates)

- **Quarterly Progress Review Workflow** (`quarterly_review_workflow.py`)
  - 4-phase review workflow (15-30 min)
  - Phase 1: Data collection (same minimal inputs as baseline)
  - Phase 2: Progress calculation (vs baseline and vs target)
  - Phase 3: Corrective actions (if off-track)
  - Phase 4: Dashboard update (KPIs, charts, status)

- **Certification Pathway Workflow** (`certification_pathway_workflow.py`)
  - 5-phase certification readiness workflow (1 hour)
  - Phase 1: Certification selection (based on profile and market)
  - Phase 2: Gap assessment (per-certification readiness check)
  - Phase 3: Gap closure plan (actions to achieve readiness)
  - Phase 4: Evidence compilation (data collection for application)
  - Phase 5: Application submission guidance

#### Templates (8)

- **SME Baseline Report** (`sme_baseline_report.py`)
  - 1-2 page visual emissions dashboard
  - 4 output formats: Markdown, HTML, JSON, Excel
  - Sections: executive summary, scope breakdown, visual bars, peer comparison, data quality badge, top 3 sources, next steps
  - Mobile-responsive HTML with CSS grid layout
  - Green colour scheme with accessibility compliance
  - SHA-256 provenance hashing

- **Quick Wins Action Plan** -- 2-3 page prioritized action list with ROI tables
- **Progress Dashboard** -- Single-page HTML dashboard optimized for mobile
- **Grant Application Brief** -- 3-5 page grant-ready summary with financials
- **Certification Readiness Report** -- 2-3 page readiness assessment with gaps
- **Cost-Benefit Summary** -- 2-3 page financial analysis with NPV/IRR tables
- **Annual Report (Simple)** -- 4-6 page stakeholder-friendly annual summary
- **Peer Benchmark Report** -- 1-2 page sector comparison with percentile ranking

#### Integrations (13)

- **Pack Orchestrator** -- 6-phase master pipeline for all SME workflows
- **GHG App Bridge** -- Connects to GL-GHG-APP for Scope 1/2/3 calculation
- **SBTi App Bridge** -- Connects to GL-SBTi-APP for target validation
- **MRV Bridge (Simplified)** -- Routes simplified data to relevant MRV agents
- **Data Bridge** -- Connects to AGENT-DATA agents for intake and quality
- **Foundation Bridge** -- Platform infrastructure (auth, schema, audit, registry)
- **Health Check** -- System verification and connectivity testing (8 categories)
- **Setup Wizard** -- Guided configuration with preset selection
- **Xero Integration** -- OAuth2 connection to Xero accounting software
- **QuickBooks Integration** -- OAuth2 connection to QuickBooks Online
- **Sage Integration** -- API key connection to Sage accounting software
- **Grant Database Sync** -- Scheduled sync of 50+ grant programs (UK/EU/US)
- **Certification Registry** -- Links to SME Climate Hub, B Corp, ISO registries

#### Presets (6)

- **Micro Office** -- 1-9 employees, office-based, minimal inputs
- **Small Retail** -- 10-49 employees, retail/hospitality, waste + refrigeration focus
- **Small Services** -- 10-49 employees, professional services, travel-heavy
- **Medium Manufacturing** -- 50-249 employees, light manufacturing, process energy
- **Medium Technology** -- 50-249 employees, tech sector, cloud + commuting focus
- **General SME** -- Any sector, 1-249 employees, default balanced configuration

#### Database

- 8 database migrations (V129-PACK026-001 through V129-PACK026-008)
- 16 tables with Row-Level Security (RLS) enabled
- 3 materialized views for dashboard, action plan, and benchmark data
- 36 indexes optimized for common SME query patterns

#### Security

- JWT RS256 authentication on all endpoints
- 5-role RBAC model: sme_owner, sme_manager, sme_viewer, advisor, admin
- AES-256-GCM encryption at rest for all financial and emissions data
- TLS 1.3 encryption in transit
- OAuth2 for all accounting software integrations (no password storage)
- Full audit logging (27 event types)
- PII redaction in logs
- GDPR-compliant data subject request support

### Known Limitations

| Limitation | Impact | Workaround |
|-----------|--------|------------|
| No process emissions | Bronze/Silver underestimate for heavy manufacturing | Use PACK-021/022 for process-intensive sectors |
| Spend-based Scope 3 only | Lower accuracy for Scope 3 vs activity-based | Upgrade to Gold tier or PACK-021 for supplier-specific data |
| Single entity per free tier | Multi-site businesses need premium | Multi-Entity add-on ($100/entity/year) |
| No ETS/carbon pricing | Does not factor in emissions trading costs | Use PACK-022 for carbon pricing analysis |
| Grant database lag | New programs may take 1-2 months to appear | Manual search at gov.uk/grants, ec.europa.eu |
| 11 sectors only | Niche sectors default to "other" | Custom sector mapping available in config |
| No real-time accounting sync in free tier | Monthly sync only | Premium add-on for real-time sync |
| UK/EU/US grants only | Other regions not yet covered | Submit feature request for additional regions |

### Dependencies

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Python | >= 3.11 | Runtime |
| Pydantic | >= 2.0 | Data validation and models |
| httpx | >= 0.24 | Async HTTP for accounting APIs |
| psycopg | >= 3.1 | PostgreSQL connection |
| psycopg_pool | >= 3.1 | Connection pooling |
| PyYAML | >= 6.0 | Configuration files |
| openpyxl | >= 3.1 | Excel report generation |
| jinja2 | >= 3.1 | HTML template rendering |

---

## [Unreleased] -- Planned for v1.1.0

### Planned Features

#### Additional Quick Wins
- Heat pump sizing calculator with grant integration
- Solar PV payback calculator with Feed-in Tariff modelling
- EV fleet transition planner with TCO comparison
- Water efficiency actions (20+ new actions)
- Circular economy actions (15+ new actions)
- Remote work carbon impact calculator

#### Additional Grant Programs
- Canada (Clean Growth Hub, SIF)
- Australia (Climate Active, ERF)
- Germany (KfW Energy Efficiency, BAFA)
- France (ADEME, MaPrimeRenov)
- Japan (METI Green Innovation)
- India (BEE, SIDBI Green Finance)
- 200+ additional regional programs

#### Mobile App
- Native iOS and Android companion app
- Push notification for grant deadlines
- Offline baseline data entry
- Photo capture for utility bills (OCR)
- Quick progress update in under 2 minutes

#### Enhanced Reporting
- Interactive HTML5 charts (Chart.js integration)
- Board-ready presentation export (PPTX)
- Automated email report distribution
- Comparison report across multiple entities
- Customer-facing sustainability badge (embeddable)

#### Scope 3 Improvements
- Supplier engagement questionnaire (simplified)
- Top-10 supplier emissions estimation
- Category 2 (Capital Goods) spend-based support
- Category 3 (Fuel & Energy) auto-calculation
- Category 8 (Upstream Leased Assets) estimation

#### Integration Enhancements
- FreeAgent accounting integration
- Wave accounting integration
- Xero payroll integration (for commuting estimates)
- Smart meter API integration (UK SMETS2)
- Utility bill OCR (auto-extract from uploaded images)

### Planned Improvements

- Performance optimization: Bronze baseline target < 100 ms
- Enhanced peer benchmarking with regional granularity (county/state level)
- Multi-currency baseline comparison (normalize to tCO2e/USD equivalent)
- Action implementation status tracking with photo evidence
- Carbon literacy training module for SME staff

---

## Breaking Changes Policy

### v1.0.0

No breaking changes -- this is the initial release.

### Future Versions

PACK-026 follows semantic versioning:
- **Patch versions** (1.0.x): Bug fixes, emission factor updates, grant database updates. No breaking changes.
- **Minor versions** (1.x.0): New features, new engines, new integrations. Backward-compatible. Database migrations auto-applied.
- **Major versions** (x.0.0): Potential breaking changes to API or data models. Migration guide provided. Minimum 6-month deprecation notice.

---

## Migration Guide

### From Manual Carbon Accounting (Spreadsheets)

If you have been tracking emissions manually in spreadsheets, follow these steps to migrate to PACK-026:

1. **Export your existing data** as CSV (columns: Year, Scope 1, Scope 2, Scope 3, Total, Employees, Revenue)

2. **Run express onboarding** with your latest year's energy spend data

3. **Compare results** -- your spreadsheet total should be within the accuracy band for your data tier:
   - Bronze: +/- 40% is normal (you may have more accurate data)
   - Silver: +/- 15% -- if your spreadsheet used actual bills, expect close match
   - Gold: +/- 5% -- should match closely

4. **Import historical data** (optional):
   ```python
   from packs.net_zero.PACK_026_sme_net_zero.engines.sme_progress_tracker import SMEProgressTracker

   tracker = SMEProgressTracker()
   await tracker.import_historical_baseline(
       entity_id="your-entity-id",
       historical_data=[
           {"year": 2022, "total_tco2e": 95.4, "employees": 22},
           {"year": 2023, "total_tco2e": 88.2, "employees": 24},
           {"year": 2024, "total_tco2e": 82.4, "employees": 25},
       ],
   )
   ```

5. **Set your base year** to the earliest year with reliable data

6. **Continue tracking** with PACK-026's quarterly review workflow

### From Another Carbon Accounting Tool

If you are migrating from another tool (e.g., Normative, Watershed, Plan A):

1. **Export your GHG inventory** as CSV or JSON from the existing tool
2. **Map the scopes** to PACK-026 format (Scope 1/2/3 breakdown)
3. **Import via API** or manually enter into the standard setup workflow
4. **Verify baseline** -- compare PACK-026 result against your previous tool's output
5. **Set baseline and targets** -- PACK-026 will generate new SBTi-aligned targets

---

## Upgrade Path

### To PACK-021 Net Zero Starter Pack (Enterprise)

When your business grows beyond 250 employees or needs enterprise-grade features:

| Feature | PACK-026 (SME) | PACK-021 (Enterprise) |
|---------|:-------------:|:--------------------:|
| Employees | 1-249 | Unlimited |
| Scope 3 categories | Cat 1, 5, 6, 7 | All 15 categories |
| Scope 3 methods | Spend-based | Supplier-specific, hybrid, activity-based |
| Pathways | ACA (SBTi SME) | ACA, SDA, FLAG |
| MACC analysis | Simple ROI ranking | Full NPV/IRR optimization |
| Offset portfolio | Guidance only | Full portfolio management (VCMI, Oxford) |
| Reporting | 8 SME templates | CDP, TCFD, SBTi, ESRS E1 |
| Facilities | Single entity | Multi-facility, multi-region |
| Data sources | Utility bills, accounting | ERP, fleet systems, BMS, IoT |

**Migration path:**
```bash
greenlang pack upgrade --from PACK-026 --to PACK-021 --entity YOUR_ENTITY_ID
```

All PACK-026 data (baselines, targets, progress) is automatically migrated and preserved as the baseline for PACK-021.

### To PACK-022 Net Zero Acceleration Pack

For advanced decarbonization planning, scenario modelling, and supplier engagement:

| Feature | PACK-026 (SME) | PACK-022 (Acceleration) |
|---------|:-------------:|:----------------------:|
| Action library | 500+ SME actions | 2,000+ enterprise actions |
| Scenario modelling | No | Multi-scenario comparison |
| Supplier engagement | No | Top-100 supplier program |
| Carbon pricing | No | Internal carbon price + ETS |
| Board reporting | Simple dashboard | Executive briefing package |

### To PACK-023 SBTi Alignment Pack

For formal SBTi target validation and progress reporting:

| Feature | PACK-026 (SME) | PACK-023 (SBTi) |
|---------|:-------------:|:---------------:|
| SBTi pathway | SME Route only | ACA, SDA, FLAG |
| Scope 3 coverage | Encouraged, not required | 67% minimum threshold |
| Target validation | Auto-validated (SME) | Formal validation process |
| Progress reporting | Annual tracker | SBTi annual progress report |

---

*Changelog maintained by GreenLang Platform Team*
*Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)*
*Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)*
