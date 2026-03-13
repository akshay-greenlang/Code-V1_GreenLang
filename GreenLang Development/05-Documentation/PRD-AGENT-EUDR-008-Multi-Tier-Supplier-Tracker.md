# PRD: AGENT-EUDR-008 -- Multi-Tier Supplier Tracker

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-008 |
| **Agent ID** | GL-EUDR-MST-008 |
| **Component** | Multi-Tier Supplier Tracker Agent |
| **Category** | EUDR Regulatory Agent -- Supply Chain Traceability |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Approved |
| **Approved Date** | 2026-03-08 |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-08 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR) |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

EUDR Article 4 and Article 10 require operators and traders to exercise due diligence on the **entire** supply chain -- not just direct (Tier 1) suppliers. Operators must demonstrate that every product placed on the EU market is traceable to its origin production plots through every intermediary (Article 9). In practice, real-world commodity supply chains have 5-15+ tiers of intermediaries between the farm gate and the EU border:

- **Tier 1**: Direct supplier (exporter/trader)
- **Tier 2**: Processor/refiner/aggregator
- **Tier 3**: Regional collector/cooperative
- **Tier 4**: Local aggregation point
- **Tier N**: Smallholder farmer/production plot

The challenges operators face:

- **Limited visibility beyond Tier 1**: 85%+ of EU importers have no systematic visibility into sub-tier suppliers. They know their direct supplier but cannot identify the processor, cooperative, or farmer.
- **Supplier relationship opacity**: Supply chains change dynamically -- suppliers switch sub-suppliers, new intermediaries appear, seasonal variations shift sourcing patterns. Without continuous tracking, mapped relationships go stale.
- **Missing supplier metadata**: Sub-tier suppliers often lack formal registration, legal entity identifiers, GPS coordinates, or certification status. Data quality degrades with each tier deeper.
- **No tier depth scoring**: Operators cannot quantify how deep their supply chain visibility extends or where the visibility drops off.
- **Supplier onboarding friction**: Discovering and onboarding sub-tier suppliers requires manual outreach, questionnaires, and verification -- a process that takes weeks per supplier.
- **No supplier risk inheritance**: Risk at a Tier 4 cooperative (e.g., proximity to deforested area) is not automatically propagated to Tier 1 and the EU importer.
- **Regulatory escalation**: Under EUDR Article 14, competent authorities can demand full supply chain documentation for any product within 5 years. Operators without multi-tier records face penalties up to 4% of annual EU turnover.
- **Dynamic compliance status**: A supplier that was compliant last quarter may become non-compliant due to new deforestation in their sourcing area, changed certifications, or expired DDS.

### 1.2 Solution Overview

Agent-EUDR-008: Multi-Tier Supplier Tracker is a specialized agent that discovers, maps, monitors, and maintains the complete multi-tier supplier hierarchy for EUDR-regulated commodity supply chains. It tracks every supplier from Tier 1 down to the origin farm/plot, maintains supplier profiles with compliance metadata, monitors tier depth and visibility gaps, propagates risk signals through the supplier hierarchy, and ensures continuous compliance status monitoring.

Core capabilities:

1. **Supplier hierarchy discovery** -- Automatically discover sub-tier suppliers from ERP data, supplier declarations, questionnaires, shipping documents, and certification databases. Build recursive supplier trees from Tier 1 to Tier N.
2. **Supplier profile management** -- Maintain comprehensive supplier profiles including legal entity, location (GPS/country/region), certifications, commodity types, capacity, compliance status, and risk rating.
3. **Tier depth tracking** -- Quantify supply chain visibility depth per product/commodity, identify where visibility drops off, and score tier coverage completeness.
4. **Relationship lifecycle management** -- Track supplier relationships through time: onboarding, active, suspended, terminated. Maintain relationship history for audit.
5. **Supplier risk propagation** -- Propagate risk scores from deep-tier suppliers upstream to Tier 1 and the operator. Risk categories: deforestation proximity, country risk, certification gaps, compliance history, data quality.
6. **Compliance status monitoring** -- Continuously monitor supplier compliance status: DDS validity, certification expiry, geolocation coverage, deforestation-free verification. Alert on status changes.
7. **Gap analysis and remediation** -- Identify suppliers with missing data (no GPS, no certification, no legal entity), quantify coverage gaps, and generate remediation action plans.
8. **Audit trail and reporting** -- Immutable audit trail of all supplier data changes, relationship changes, and compliance status transitions. Generate EUDR Article 14 audit-ready reports.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Tier depth discovery | Average >= 4 tiers mapped per supply chain | Mean tier depth across all active chains |
| Supplier profile completeness | >= 90% of fields populated for Tier 1-3 | Profile completeness score per tier |
| Sub-tier discovery rate | >= 80% of known sub-tier suppliers captured | Cross-validation against manual audits |
| Risk propagation accuracy | 100% deterministic, reproducible | Bit-perfect reproducibility tests |
| Compliance monitoring latency | < 24 hours from status change to alert | Time-to-alert measurement |
| Gap identification precision | >= 95% of gaps correctly identified | Precision/recall against audit results |
| Supplier onboarding time | < 48 hours for Tier 1-2; < 7 days for Tier 3+ | Mean time from discovery to profile creation |
| Audit report generation | < 30 seconds for full supplier chain report | p99 latency benchmark |
| EUDR Article 14 readiness | 100% of supplier data audit-retrievable within 5 years | Retention and retrieval validation |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM**: 400,000+ EUDR-affected operators needing multi-tier supplier visibility, representing a supplier management market of 2-4 billion EUR.
- **SAM**: 100,000+ EU importers requiring systematic sub-tier supplier tracking, estimated at 400-800M EUR.
- **SOM**: Target 500+ enterprise customers in Year 1, representing 20-35M EUR in supplier tracking module ARR.

### 2.2 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Tier 1 supplier databases (SAP Ariba, Coupa) | Enterprise integration | No sub-tier visibility; no EUDR-specific compliance | Multi-tier recursive discovery; EUDR-native |
| Supply chain mapping tools (Sourcemap, Altana) | Visual mapping; AI discovery | Generic (not EUDR-specific); expensive; limited risk propagation | EUDR Article 9/10 compliant; deterministic risk scoring |
| Certification platforms (RSPO, FSC, UTZ) | Commodity expertise; trusted | Single-commodity; no cross-commodity; Tier 1-2 only | All 7 commodities; unlimited tier depth |
| Manual questionnaires + spreadsheets | Low cost; flexible | Error-prone; slow; no automation; no real-time monitoring | Automated discovery; real-time compliance monitoring |

---

## 3. Goals and Non-Goals

### 3.1 Goals

1. Discover and map supplier hierarchies from Tier 1 to Tier N for all 7 EUDR commodities
2. Maintain comprehensive supplier profiles with compliance metadata
3. Track tier depth and visibility coverage with quantified scores
4. Propagate risk from deep-tier suppliers to Tier 1 and operators
5. Monitor supplier compliance status continuously with alerting
6. Identify and quantify data gaps with remediation guidance
7. Generate EUDR Article 14 audit-ready reports
8. Process 100,000+ supplier records in batch mode

### 3.2 Non-Goals

1. Supply chain graph visualization (AGENT-EUDR-001 handles graph topology)
2. GPS coordinate validation (AGENT-EUDR-007)
3. Satellite monitoring of supplier locations (AGENT-EUDR-003)
4. Forest cover analysis at supplier plots (AGENT-EUDR-004)
5. Supplier payment or procurement management
6. Contract negotiation or pricing

---

## 4. User Personas

### 4.1 Supply Chain Manager -- Carlos (Primary)
- **Role**: EUDR supply chain lead at a major cocoa importer
- **Goal**: Map and monitor all suppliers from direct trader down to smallholder cooperatives
- **Pain point**: Only knows Tier 1 supplier; cannot identify the 500+ cooperatives and 50,000+ farmers behind each shipment
- **Key features**: Tier discovery, supplier hierarchy visualization, gap analysis

### 4.2 Compliance Officer -- Maria (Primary)
- **Role**: EUDR compliance lead at a multinational food company
- **Goal**: Ensure all suppliers meet EUDR requirements before DDS submission
- **Pain point**: Cannot verify sub-tier supplier compliance; no visibility into Tier 3+ certification status
- **Key features**: Compliance monitoring, risk propagation, audit reports

### 4.3 Procurement Analyst -- Aiko (Secondary)
- **Role**: Strategic sourcing analyst at a rubber importer
- **Goal**: Understand supplier concentration risk and diversification needs
- **Pain point**: No data on sub-tier supplier dependencies; cannot quantify single-source risk
- **Key features**: Supplier profiles, tier depth analysis, concentration metrics

### 4.4 Auditor -- Klaus (Secondary)
- **Role**: Third-party EUDR auditor
- **Goal**: Verify that operators have complete multi-tier supplier documentation
- **Pain point**: No standardized supplier hierarchy format; manual document review
- **Key features**: Audit trail, supplier chain reports, compliance certificates

---

## 5. Regulatory Requirements

### 5.1 EUDR Article 4 -- Due Diligence Obligation

| Requirement | EUDR Reference | Implementation |
|-------------|---------------|----------------|
| Due diligence on full supply chain | Article 4(1) | Multi-tier supplier discovery and tracking |
| Identify all operators and traders | Article 4(2) | Supplier profile management |
| Maintain due diligence system | Article 4(5) | Continuous compliance monitoring |

### 5.2 EUDR Article 9 -- Traceability Information

| Requirement | EUDR Reference | Implementation |
|-------------|---------------|----------------|
| Name and address of suppliers | Article 9(1)(a) | Supplier profile with legal entity |
| Name and address of buyers | Article 9(1)(b) | Buyer/importer profile linkage |
| Country of production | Article 9(1)(c) | Supplier location tracking |
| Geolocation of production plots | Article 9(1)(d) | GPS linkage per supplier |

### 5.3 EUDR Article 10 -- Trader Obligations

| Requirement | EUDR Reference | Implementation |
|-------------|---------------|----------------|
| Traders must identify operators | Article 10(1) | Upstream supplier discovery |
| Retain DDS references | Article 10(2) | DDS reference tracking per supplier |
| Simplified due diligence (SMEs) | Article 10(3) | Tier-appropriate compliance checking |

### 5.4 EUDR Article 14 -- Competent Authority Requests

| Requirement | Implementation |
|-------------|----------------|
| 5-year retention of supplier data | Immutable audit trail with 5-year retention |
| Full supply chain documentation on request | Audit report generation within 30 seconds |
| Proof of due diligence completeness | Tier depth and coverage scoring |

---

## 6. Features and Requirements

### 6.1 Feature 1: Supplier Hierarchy Discovery (P0)

**Requirements**:
- F1.1: Discover Tier 2+ suppliers from Tier 1 supplier declarations
- F1.2: Extract supplier relationships from shipping documents (BL, packing lists)
- F1.3: Parse supplier questionnaire responses to build hierarchy links
- F1.4: Discover suppliers from certification database cross-references (RSPO, FSC, UTZ)
- F1.5: Auto-detect tier depth from commodity flow analysis
- F1.6: Handle many-to-many supplier relationships (one supplier feeds multiple buyers)
- F1.7: Recursive discovery with configurable max depth (default 15 tiers)
- F1.8: Confidence scoring for discovered relationships (verified, declared, inferred, suspected)
- F1.9: Batch discovery from bulk supplier data uploads
- F1.10: Deduplication of suppliers across multiple discovery sources

### 6.2 Feature 2: Supplier Profile Management (P0)

**Requirements**:
- F2.1: Comprehensive supplier profile: legal entity name, registration ID, tax ID, DUNS
- F2.2: Location data: country (ISO 3166), admin region, GPS coordinates, address
- F2.3: Commodity types handled (7 EUDR commodities + derived products)
- F2.4: Certifications: type (FSC, RSPO, UTZ, Rainforest Alliance), certificate ID, validity dates
- F2.5: Capacity: annual volume, processing capacity, number of upstream suppliers
- F2.6: Risk profile: country risk, deforestation proximity risk, compliance history
- F2.7: Contact information: primary contact, compliance contact
- F2.8: DDS references: linked DDS IDs from EU Information System
- F2.9: Profile completeness score (0-100) with missing field identification
- F2.10: Profile versioning with full change history

### 6.3 Feature 3: Tier Depth Tracking (P0)

**Requirements**:
- F3.1: Calculate tier depth for each supplier in the hierarchy
- F3.2: Visibility score: percentage of known suppliers at each tier level
- F3.3: Coverage score: percentage of commodity volume with full origin traceability
- F3.4: Tier gap detection: identify where visibility drops off
- F3.5: Commodity-specific tier depth (different commodities have different chain lengths)
- F3.6: Time-series tier depth tracking (visibility improving or degrading over time)
- F3.7: Benchmark tier depth against industry averages
- F3.8: Tier depth alerts: notify when visibility drops below configured threshold

### 6.4 Feature 4: Relationship Lifecycle Management (P0)

**Requirements**:
- F4.1: Relationship states: prospective, onboarding, active, suspended, terminated
- F4.2: Relationship attributes: start date, commodity, volume, frequency, exclusivity
- F4.3: Relationship change tracking with reason codes
- F4.4: Seasonal relationship patterns (e.g., harvest season only)
- F4.5: Relationship strength scoring based on transaction frequency, volume, duration
- F4.6: Relationship conflict detection (circular dependencies, inconsistent declarations)
- F4.7: Relationship timeline visualization data
- F4.8: Bulk relationship import from ERP/procurement systems

### 6.5 Feature 5: Supplier Risk Propagation (P0)

**Requirements**:
- F5.1: Risk categories: deforestation_proximity, country_risk, certification_gap, compliance_history, data_quality, concentration_risk
- F5.2: Risk score per supplier: 0-100 composite with category breakdown
- F5.3: Upstream risk propagation: deep-tier risk flows to Tier 1 and operator
- F5.4: Propagation methods: max (worst-case), weighted average, volume-weighted
- F5.5: Risk inheritance rules configurable per commodity and tier
- F5.6: Risk change alerts: notify when supplier risk crosses threshold
- F5.7: Risk trend analysis: improving, stable, or degrading over time
- F5.8: Concentration risk: single-source dependency, geographic concentration, certification concentration

### 6.6 Feature 6: Compliance Status Monitoring (P0)

**Requirements**:
- F6.1: Compliance dimensions: DDS validity, certification status, geolocation coverage, deforestation-free status
- F6.2: Composite compliance score per supplier (0-100)
- F6.3: Compliance status: compliant, conditionally_compliant, non_compliant, unverified, expired
- F6.4: DDS validity tracking with expiry alerts (30/14/7 day warnings)
- F6.5: Certification expiry monitoring with auto-renewal tracking
- F6.6: Geolocation coverage: percentage of supplier volume with GPS-verified origin
- F6.7: Deforestation-free verification status linkage (AGENT-EUDR-003/004/005)
- F6.8: Compliance history timeline with trend analysis
- F6.9: Non-compliance escalation workflow with remediation deadlines
- F6.10: Supplier compliance dashboard data generation

### 6.7 Feature 7: Gap Analysis and Remediation (P0)

**Requirements**:
- F7.1: Data gap detection: missing GPS, missing certification, missing legal entity, missing DDS
- F7.2: Coverage gap: tiers without any known suppliers
- F7.3: Verification gap: suppliers with outdated or unverified data
- F7.4: Gap severity classification: critical (blocks DDS), major (regulatory risk), minor (data quality)
- F7.5: Remediation action plan generation with prioritized steps
- F7.6: Remediation progress tracking with completion percentage
- F7.7: Auto-generated supplier questionnaires for gap filling
- F7.8: Gap trend analysis over time (improving or worsening)

### 6.8 Feature 8: Audit Trail and Reporting (P0)

**Requirements**:
- F8.1: Immutable audit log of all supplier data changes
- F8.2: Relationship change audit trail with actor and timestamp
- F8.3: Compliance status change history with evidence
- F8.4: EUDR Article 14 audit report generation (full supplier chain)
- F8.5: Supplier chain report: JSON, PDF, CSV, EUDR XML formats
- F8.6: Tier depth summary report with coverage metrics
- F8.7: Risk propagation report showing inheritance paths
- F8.8: DDS submission readiness report per supplier chain

---

## 7. Technical Requirements

### 7.1 Architecture

```
greenlang/agents/eudr/multi_tier_supplier/
    __init__.py                          # Package exports (80+ symbols)
    config.py                            # MultiTierSupplierConfig singleton
    models.py                            # Pydantic v2 models, enums
    provenance.py                        # SHA-256 chain hashing
    metrics.py                           # Prometheus metrics (gl_eudr_mst_ prefix)
    supplier_discovery_engine.py         # Engine 1: Tier discovery
    supplier_profile_manager.py          # Engine 2: Profile CRUD
    tier_depth_tracker.py                # Engine 3: Depth scoring
    relationship_manager.py              # Engine 4: Lifecycle management
    risk_propagation_engine.py           # Engine 5: Risk flow
    compliance_monitor.py                # Engine 6: Status monitoring
    gap_analyzer.py                      # Engine 7: Gap detection
    audit_reporter.py                    # Engine 8: Reports
    setup.py                             # MultiTierSupplierService facade
    reference_data/
        __init__.py
        country_risk_scores.py           # Country-level deforestation risk
        certification_standards.py       # Certification types and rules
        commodity_supply_chains.py       # Typical chain structure per commodity
    api/
        __init__.py
        router.py
        schemas.py
        dependencies.py
        discovery_routes.py
        profile_routes.py
        tier_routes.py
        compliance_routes.py
        report_routes.py
        batch_routes.py
```

### 7.2 Database Schema (V096)

| Table | Type | Description |
|-------|------|-------------|
| `gl_eudr_mst_suppliers` | regular | Supplier profile records |
| `gl_eudr_mst_relationships` | hypertable (monthly) | Supplier-to-supplier relationships |
| `gl_eudr_mst_tier_scores` | hypertable (monthly) | Tier depth and visibility scores |
| `gl_eudr_mst_risk_scores` | hypertable (monthly) | Supplier risk assessment records |
| `gl_eudr_mst_compliance_status` | hypertable (monthly) | Compliance monitoring records |
| `gl_eudr_mst_certifications` | regular | Supplier certification records |
| `gl_eudr_mst_gaps` | regular | Data gap analysis results |
| `gl_eudr_mst_remediation_plans` | regular | Remediation action plans |
| `gl_eudr_mst_batch_jobs` | regular | Batch processing jobs |
| `gl_eudr_mst_audit_log` | regular | Immutable audit trail |

### 7.3 Prometheus Metrics (18 metrics, `gl_eudr_mst_` prefix)

| Metric | Type | Description |
|--------|------|-------------|
| `gl_eudr_mst_suppliers_discovered_total` | Counter | Total suppliers discovered |
| `gl_eudr_mst_suppliers_onboarded_total` | Counter | Total suppliers onboarded |
| `gl_eudr_mst_relationships_created_total` | Counter | Total relationships created |
| `gl_eudr_mst_tier_depth_assessments_total` | Counter | Total tier depth assessments |
| `gl_eudr_mst_risk_assessments_total` | Counter | Total risk assessments |
| `gl_eudr_mst_risk_alerts_total` | Counter | Risk threshold alerts |
| `gl_eudr_mst_compliance_checks_total` | Counter | Total compliance checks |
| `gl_eudr_mst_compliance_alerts_total` | Counter | Compliance status change alerts |
| `gl_eudr_mst_gaps_detected_total` | Counter | Total gaps detected |
| `gl_eudr_mst_gaps_remediated_total` | Counter | Total gaps remediated |
| `gl_eudr_mst_reports_generated_total` | Counter | Total reports generated |
| `gl_eudr_mst_batch_jobs_total` | Counter | Total batch jobs |
| `gl_eudr_mst_discovery_duration_seconds` | Histogram | Discovery operation latency |
| `gl_eudr_mst_risk_propagation_duration_seconds` | Histogram | Risk propagation latency |
| `gl_eudr_mst_compliance_check_duration_seconds` | Histogram | Compliance check latency |
| `gl_eudr_mst_active_suppliers` | Gauge | Currently active suppliers |
| `gl_eudr_mst_avg_tier_depth` | Gauge | Average tier depth |
| `gl_eudr_mst_api_errors_total` | Counter | Total API errors |

### 7.4 API Endpoints (~32 endpoints)

| Group | Method | Path | Description |
|-------|--------|------|-------------|
| Discovery | POST | `/api/v1/eudr-mst/discover` | Discover sub-tier suppliers |
| | POST | `/api/v1/eudr-mst/discover/batch` | Batch discovery |
| | POST | `/api/v1/eudr-mst/discover/from-declaration` | Discovery from supplier declaration |
| | POST | `/api/v1/eudr-mst/discover/from-questionnaire` | Discovery from questionnaire |
| Profiles | POST | `/api/v1/eudr-mst/suppliers` | Create supplier profile |
| | GET | `/api/v1/eudr-mst/suppliers/{supplier_id}` | Get supplier profile |
| | PUT | `/api/v1/eudr-mst/suppliers/{supplier_id}` | Update supplier profile |
| | DELETE | `/api/v1/eudr-mst/suppliers/{supplier_id}` | Deactivate supplier |
| | POST | `/api/v1/eudr-mst/suppliers/search` | Search suppliers |
| | POST | `/api/v1/eudr-mst/suppliers/batch` | Batch create/update |
| Tiers | GET | `/api/v1/eudr-mst/tiers/{supplier_id}` | Get tier depth for supplier |
| | POST | `/api/v1/eudr-mst/tiers/assess` | Assess tier depth for chain |
| | GET | `/api/v1/eudr-mst/tiers/visibility` | Get visibility scores |
| | GET | `/api/v1/eudr-mst/tiers/gaps` | Get tier coverage gaps |
| Relationships | POST | `/api/v1/eudr-mst/relationships` | Create relationship |
| | PUT | `/api/v1/eudr-mst/relationships/{rel_id}` | Update relationship |
| | GET | `/api/v1/eudr-mst/relationships/{supplier_id}` | Get supplier relationships |
| | POST | `/api/v1/eudr-mst/relationships/history` | Get relationship history |
| Risk | POST | `/api/v1/eudr-mst/risk/assess` | Assess supplier risk |
| | POST | `/api/v1/eudr-mst/risk/propagate` | Propagate risk through chain |
| | GET | `/api/v1/eudr-mst/risk/{supplier_id}` | Get supplier risk profile |
| | POST | `/api/v1/eudr-mst/risk/batch` | Batch risk assessment |
| Compliance | POST | `/api/v1/eudr-mst/compliance/check` | Check supplier compliance |
| | GET | `/api/v1/eudr-mst/compliance/{supplier_id}` | Get compliance status |
| | POST | `/api/v1/eudr-mst/compliance/batch` | Batch compliance check |
| | GET | `/api/v1/eudr-mst/compliance/alerts` | Get compliance alerts |
| Reports | POST | `/api/v1/eudr-mst/reports/audit` | Generate audit report |
| | POST | `/api/v1/eudr-mst/reports/tier-summary` | Tier depth summary |
| | POST | `/api/v1/eudr-mst/reports/gaps` | Gap analysis report |
| | GET | `/api/v1/eudr-mst/reports/{report_id}` | Get report |
| | GET | `/api/v1/eudr-mst/reports/{report_id}/download` | Download report |
| Batch | POST | `/api/v1/eudr-mst/batch` | Submit batch job |
| | DELETE | `/api/v1/eudr-mst/batch/{batch_id}` | Cancel batch job |
| Health | GET | `/api/v1/eudr-mst/health` | Health check |

---

## 8. Test Strategy

### 8.1 Unit Tests (500+)
- Supplier discovery from 5+ data source types
- Profile CRUD with validation and deduplication
- Tier depth calculation for linear, branching, and diamond-shaped chains
- Risk propagation with max, weighted average, and volume-weighted methods
- Compliance status transitions and alert generation
- Gap detection across all data categories
- Report generation in all formats
- Relationship lifecycle state transitions
- Certification expiry calculations
- Country risk score lookups

### 8.2 Performance Tests
- Batch discovery of 100,000 supplier records
- Risk propagation through 10,000-node supply chains
- Compliance check throughput measurement

---

## Appendices

### A. Typical Supply Chain Depths by Commodity

| Commodity | Typical Tiers | Chain Structure |
|-----------|--------------|-----------------|
| Cocoa | 6-8 | Farmer -> Cooperative -> Aggregator -> Processor -> Trader -> Refiner -> Importer |
| Coffee | 5-7 | Farmer -> Cooperative -> Mill -> Exporter -> Trader -> Roaster -> Importer |
| Palm oil | 5-7 | Smallholder -> Mill -> Refinery -> Trader -> Processor -> Importer |
| Soya | 4-6 | Farm -> Silo -> Crusher -> Trader -> Processor -> Importer |
| Rubber | 5-7 | Smallholder -> Dealer -> Factory -> Trader -> Processor -> Importer |
| Cattle | 3-5 | Ranch -> Feedlot -> Slaughterhouse -> Trader -> Importer |
| Wood/Timber | 4-6 | Forest -> Sawmill -> Processor -> Trader -> Importer |

### B. Risk Category Weights

| Risk Category | Default Weight | Description |
|---------------|---------------|-------------|
| deforestation_proximity | 0.30 | Distance to recent deforestation events |
| country_risk | 0.20 | Country-level deforestation/governance risk |
| certification_gap | 0.15 | Missing or expired certifications |
| compliance_history | 0.15 | Historical compliance violations |
| data_quality | 0.10 | Profile completeness and data freshness |
| concentration_risk | 0.10 | Single-source or geographic concentration |

### C. Compliance Status Definitions

| Status | Definition | DDS Impact |
|--------|-----------|------------|
| COMPLIANT | All checks pass; valid DDS, certifications, GPS | Can include in DDS |
| CONDITIONALLY_COMPLIANT | Minor gaps; remediation in progress | Can include with disclosure |
| NON_COMPLIANT | Critical gaps; failed checks | Cannot include in DDS |
| UNVERIFIED | Not yet assessed; insufficient data | Cannot include in DDS |
| EXPIRED | Previously compliant; certifications/DDS expired | Cannot include until renewed |

### D. Supplier Profile Completeness Scoring

| Field Category | Weight | Fields |
|---------------|--------|--------|
| Legal identity | 25% | Legal name, registration ID, country |
| Location | 20% | GPS coordinates, address, admin region |
| Commodity | 15% | Commodity types, volumes, capacity |
| Certification | 15% | Certification type, ID, validity |
| Compliance | 15% | DDS reference, deforestation status |
| Contact | 10% | Primary contact, compliance contact |
