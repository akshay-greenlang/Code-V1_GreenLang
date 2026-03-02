# PRD: APP-002 -- GL-CBAM-APP v1.1 Enhancement

## Document Info
| Field | Value |
|-------|-------|
| PRD ID | PRD-APP-002 |
| Application | GL-CBAM-APP v1.1 |
| Priority | P0 (Critical) |
| Version | 1.1.0 |
| Status | Approved |
| Author | GL-ProductManager + GL-RegulatoryIntelligence |
| Date | 2026-02-26 |
| Base | GL-CBAM-APP v1.0 (applications/GL-CBAM-APP/CBAM-Importer-Copilot/) |

---

## 1. Overview

### 1.1 Purpose
Enhance GL-CBAM-APP from v1.0 (production-certified 3-agent pipeline) to v1.1 by:
1. **Supplier Portal** -- Self-service portal for third-country installation operators to submit emissions data
2. **Quarterly Automation** -- End-to-end quarterly CBAM report generation with deadline tracking
3. **Definitive Period Support** -- CBAM certificate calculation, EU ETS pricing, free allocation adjustments
4. **De Minimis Exemption Engine** -- 50-tonne annual threshold checking (2026 Omnibus simplification)
5. **Verification Workflow** -- Accredited verifier management, site visit scheduling, materiality checks
6. **Carbon Price Deduction Tracker** -- Track carbon prices paid in origin countries for certificate deductions

### 1.2 Current State (v1.0 Gap Analysis)

| Component | v1.0 State | v1.1 Target |
|-----------|-----------|-------------|
| Supplier data intake | Manual YAML upload | Self-service supplier portal with API |
| Supplier management | Static supplier profiles | Dynamic registry with verification status |
| Quarterly reporting | Manual pipeline trigger | Automated quarterly generation + deadline tracking |
| Report amendments | Not supported | 2-month correction window with version control |
| CBAM certificates | Not implemented | Certificate calculation + EU ETS price feed |
| Free allocation | Not implemented | Proportional deduction engine |
| Carbon price deductions | Not implemented | Third-country carbon price tracking |
| De minimis check | Not implemented | 50-tonne annual threshold per CN code group |
| Verification workflow | Not implemented | Verifier management, site visits, materiality |
| Default value markups | Not implemented | +10%/+20%/+30% progressive markup engine |
| CN codes | 30 codes | 50+ codes with sector expansion |
| Emission factors | 14 product variants | 30+ product variants with regional factors |

### 1.3 Regulatory Context
- **CBAM Regulation (EU) 2023/956** -- Carbon Border Adjustment Mechanism
- **Implementing Regulation (EU) 2023/1773** -- Transitional period reporting
- **Omnibus Simplification (Oct 2025)** -- De minimis 50t, certificate delays, deadline extensions
- **Transitional Period**: Oct 2023 - Dec 2025 (quarterly reporting, no certificates)
- **Definitive Period**: Jan 2026+ (annual reporting, certificates required)
- **Certificate Sales Start**: Feb 1, 2027
- **First Certificate Surrender**: Sep 30, 2027 (for 2026 emissions)

---

## 2. Enhancement Areas

### 2.1 Supplier Portal Module

#### 2.1.1 Supplier Registration & Management
- Supplier self-registration with EORI number validation
- Installation profile management (name, address, country, processes)
- Multi-installation support per supplier
- Supplier verification status tracking (unverified, pending, verified, expired)
- Supplier-importer linkage management
- Communication log (requests, responses, data submissions)

#### 2.1.2 Emissions Data Submission
- Web-form and API-based emissions data upload
- Support for EU calculation method format
- Installation-level emissions data (direct + indirect)
- Product-specific embedded emissions (per CN code)
- Precursor emissions tracking (recursive upstream)
- Supporting documentation upload (verification reports, process descriptions)
- Data versioning with amendment history

#### 2.1.3 Supplier Dashboard
- Emissions data submission status
- Verification timeline and requirements
- Data quality scoring (completeness, consistency)
- Historical submission archive
- Upcoming deadline notifications

#### 2.1.4 Importer-Supplier Data Exchange
- Supplier search in third-country installation registry
- EORI-based data access authorization
- Emissions data retrieval API for importers
- Automated data refresh notifications
- Bulk supplier data import/export (CSV, JSON, XML)

### 2.2 Quarterly Automation Engine

#### 2.2.1 Quarterly Report Lifecycle
- Automated quarterly period detection (Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec)
- Report generation trigger: T+15 days after quarter end
- Deadline management: T+30 days submission deadline
- Status workflow: draft -> review -> submitted -> accepted/rejected
- Amendment window: T+60 days for corrections

#### 2.2.2 Automated Report Assembly
- Pull all shipments for quarter from intake pipeline
- Aggregate emissions by:
  - CN code (8-digit level)
  - Country of origin
  - Installation
  - Product group (cement, steel, aluminium, fertilizers, electricity, hydrogen)
- Apply calculation method hierarchy (supplier actual -> regional factor -> default value)
- Generate EU CBAM Registry format XML output
- Create human-readable summary report (PDF + Markdown)

#### 2.2.3 Deadline Tracking & Notifications
- Quarterly deadline calendar with configurable alerts
- Alert levels: 30-day, 14-day, 7-day, 3-day, 1-day, overdue
- Email/webhook notification integration
- Amendment deadline tracking (2-month correction window)
- Multi-user notification routing (compliance officer, finance, management)

#### 2.2.4 Report Amendment Management
- Version-controlled report amendments
- Diff visualization (what changed between versions)
- Amendment reason tracking (regulatory requirement)
- Audit trail for all amendments
- Resubmission workflow with approval

### 2.3 CBAM Certificate Engine

#### 2.3.1 Certificate Requirement Calculator
- Calculate annual certificate obligation per CN code
- Formula: `certificates_required = SUM(quantity_mt × embedded_emissions_tCO2e_per_mt)`
- Apply free allocation adjustment factor
- Apply carbon price deductions
- Net certificate requirement calculation
- Quarterly holding requirement (50% of annual estimate per 2026 Omnibus)

#### 2.3.2 EU ETS Price Integration
- Weekly EU ETS auction clearing price feed
- Quarterly volume-weighted average calculation (for 2026)
- Historical price database
- Certificate cost estimation: `cost = certificates × weekly_ets_price`
- Price trend analytics and forecasting
- Budget impact projections

#### 2.3.3 Free Allocation Adjustment
- Track EU free allocation factors by product benchmark
- Calculate proportional deduction for importers
- Annual free allocation factor updates
- Phase-out schedule tracking (free allocation declining to zero by 2034)
- Benchmark values per product category (cement clinker, hot metal, aluminium)

#### 2.3.4 Carbon Price Deduction Tracker
- Track carbon prices paid in third countries
- Verified evidence management (payment receipts, tax certificates)
- Currency conversion to EUR (ECB exchange rates)
- Deduction calculation per installation/product
- Eligible carbon pricing scheme registry (ETS, carbon tax, etc.)
- Commission-published default carbon prices (2027+)

### 2.4 De Minimis Exemption Engine

#### 2.4.1 Threshold Monitoring
- Track cumulative annual imports by CN code sector group
- 50-tonne threshold per year for cement, steel, aluminium, fertilizers
- Electricity and hydrogen excluded from threshold
- Real-time threshold approach alerts (80%, 90%, 95%, 100%)
- Historical import volume tracking
- Forecast threshold breach date based on import velocity

#### 2.4.2 Exemption Status Management
- Automatic exemption determination per year
- Exemption certificate generation
- Status transitions: exempt -> approaching_threshold -> subject_to_cbam
- Mid-year exemption loss handling (retroactive reporting required)
- SME identification and simplified compliance path

### 2.5 Verification Workflow Manager

#### 2.5.1 Verifier Registry
- Accredited verifier database
- NAB (National Accreditation Body) credentials tracking
- Verifier expertise by sector (cement, steel, aluminium, etc.)
- Accreditation expiry monitoring
- Conflict-of-interest checking

#### 2.5.2 Verification Scheduling
- Annual physical site visit planning
- Biennial visit schedule (2027+)
- Remote audit scheduling (between on-site visits)
- Verifier assignment and notification
- Visit outcome tracking (pass/fail/conditional)

#### 2.5.3 Materiality Assessment
- 5% materiality threshold per CN code
- Automatic materiality calculation (direct + indirect emissions)
- Threshold breach flagging
- Recommended verification scope based on materiality
- Historical materiality trend analysis

### 2.6 Data & Reference Expansion

#### 2.6.1 CN Code Expansion
- Expand from 30 to 50+ CN codes
- Add downstream product codes
- Add precursor material codes
- Include sector annexes per CBAM Regulation Annex I
- CN code versioning (annual EU updates)

#### 2.6.2 Emission Factor Expansion
- Expand from 14 to 30+ product variants
- Add regional emission factors (by country/region)
- Default value database with markup schedule
- JRC (Joint Research Centre) published values
- Historical emission factor versioning

#### 2.6.3 Country & Carbon Pricing Database
- 50+ country profiles with carbon pricing status
- Carbon pricing scheme details (ETS, carbon tax, hybrid)
- EU ETS equivalent pricing
- Grid emission factors by country
- Country-specific calculation adjustments

---

## 3. File Structure (New Files)

```
applications/GL-CBAM-APP/CBAM-Importer-Copilot/
    supplier_portal/
        __init__.py                              (~150 lines)
        models.py                                (~1,200 lines)
        supplier_registry.py                     (~1,500 lines)
        emissions_submission.py                  (~1,200 lines)
        supplier_dashboard.py                    (~800 lines)
        data_exchange.py                         (~1,000 lines)
        api/
            __init__.py                          (~10 lines)
            supplier_routes.py                   (~1,500 lines)
    quarterly_engine/
        __init__.py                              (~150 lines)
        models.py                                (~800 lines)
        quarterly_scheduler.py                   (~1,000 lines)
        report_assembler.py                      (~1,500 lines)
        amendment_manager.py                     (~800 lines)
        deadline_tracker.py                      (~600 lines)
        notification_service.py                  (~500 lines)
        api/
            __init__.py                          (~10 lines)
            quarterly_routes.py                  (~1,200 lines)
    certificate_engine/
        __init__.py                              (~150 lines)
        models.py                                (~800 lines)
        certificate_calculator.py                (~1,200 lines)
        ets_price_service.py                     (~800 lines)
        free_allocation.py                       (~700 lines)
        carbon_price_deduction.py                (~900 lines)
        api/
            __init__.py                          (~10 lines)
            certificate_routes.py                (~1,000 lines)
    deminimis_engine/
        __init__.py                              (~100 lines)
        threshold_monitor.py                     (~800 lines)
        exemption_manager.py                     (~600 lines)
    verification_workflow/
        __init__.py                              (~100 lines)
        verifier_registry.py                     (~800 lines)
        verification_scheduler.py                (~700 lines)
        materiality_assessor.py                  (~600 lines)
    data/
        cn_codes_v2.json                         (~1,500 lines, expanded)
        emission_factors_v2.py                   (~1,200 lines, expanded)
        country_carbon_pricing.json              (~800 lines)
        default_value_markups.json               (~200 lines)
        free_allocation_benchmarks.json          (~300 lines)
```

### 3.1 Modified Files
```
    agents/
        shipment_intake_agent.py                 (integrate supplier portal data)
        emissions_calculator_agent.py            (add default markup, regional factors)
        reporting_packager_agent.py              (quarterly format, XML output)
    config/
        cbam_config.yaml                         (v1.1 sections for new modules)
    rules/
        cbam_rules.yaml                          (expanded rules for certificates, deminimis)
```

---

## 4. Development Tasks (Parallel Build Plan)

### Task Group A: Supplier Portal (Agent 1)
- A1: Build supplier_portal/models.py (all data models)
- A2: Build supplier_registry.py (registration, EORI, verification status)
- A3: Build emissions_submission.py (data upload, versioning)
- A4: Build supplier_dashboard.py (status, quality scoring)
- A5: Build data_exchange.py (importer-supplier API)
- A6: Build supplier_routes.py (REST API endpoints)

### Task Group B: Quarterly Automation (Agent 2)
- B1: Build quarterly_engine/models.py
- B2: Build quarterly_scheduler.py (period detection, lifecycle)
- B3: Build report_assembler.py (aggregation, XML output)
- B4: Build amendment_manager.py (versioning, diffs)
- B5: Build deadline_tracker.py + notification_service.py
- B6: Build quarterly_routes.py (REST API)

### Task Group C: Certificate Engine (Agent 3)
- C1: Build certificate_engine/models.py
- C2: Build certificate_calculator.py (obligation, net calculation)
- C3: Build ets_price_service.py (weekly/quarterly prices)
- C4: Build free_allocation.py (benchmark deductions)
- C5: Build carbon_price_deduction.py (third-country tracking)
- C6: Build certificate_routes.py (REST API)

### Task Group D: De Minimis + Verification + Data (Agent 4)
- D1: Build threshold_monitor.py + exemption_manager.py
- D2: Build verifier_registry.py + verification_scheduler.py + materiality_assessor.py
- D3: Expand cn_codes_v2.json (30 -> 50+ codes)
- D4: Expand emission_factors_v2.py (14 -> 30+ variants)
- D5: Build country_carbon_pricing.json, default_value_markups.json, free_allocation_benchmarks.json
- D6: Update cbam_config.yaml, cbam_rules.yaml, and integrate with existing agents

---

## 5. Acceptance Criteria

1. Supplier self-registration with EORI validation and multi-installation support
2. Emissions data submission with versioning and verification status tracking
3. Automated quarterly report generation with Q1-Q4 lifecycle management
4. Report amendment support with 2-month correction window and audit trail
5. CBAM certificate requirement calculation with free allocation adjustment
6. EU ETS price integration (weekly/quarterly) for cost estimation
7. Carbon price deduction tracking with evidence management
8. De minimis 50-tonne threshold monitoring with real-time alerts
9. Verification workflow with verifier registry and materiality assessment
10. Expanded CN codes (50+) and emission factors (30+ variants)
11. Country carbon pricing database (50+ countries)
12. All new code with comprehensive test coverage
