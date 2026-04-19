# PRD: AGENT-EUDR-011 -- Mass Balance Calculator Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-011 |
| **Agent ID** | GL-EUDR-MBC-011 |
| **Component** | Mass Balance Calculator Agent |
| **Category** | EUDR Regulatory Agent -- Mass Balance Accounting & Credit Reconciliation |
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

EUDR permits operators to use the **Mass Balance** (MB) chain of custody model, which allows physical mixing of EUDR-compliant and non-compliant material while maintaining administrative accounting that ensures the total volume of compliant output does not exceed the volume of compliant input over a defined credit period. This model is widely used in palm oil (RSPO), cocoa (UTZ/Rainforest Alliance), and wood (FSC) supply chains because it is commercially practical -- it does not require physical segregation at every stage.

However, mass balance accounting is notoriously error-prone and subject to abuse:

- **Credit period mismanagement**: An operator claims 500 tonnes of compliant palm oil output in January based on 300 tonnes of compliant input received in December, violating the 3-month RSPO credit period because the input credits expired before sufficient volume was accumulated.
- **Conversion factor manipulation**: A cocoa processor applies a 0.90 yield ratio (beans to liquor) when the industry standard is 0.70, inflating the compliant output volume by 28% and creating phantom compliance credits.
- **Cross-facility credit transfers**: Compliant input credits from Facility A are used to offset non-compliant output at Facility B, despite the MB model requiring per-facility accounting. Without centralized ledger management, this fraud goes undetected.
- **Overdraft tolerance abuse**: A facility consistently draws more compliant output than it has compliant input, relying on "reconciliation at period end" to paper over systematic overdrafts that violate the intent of mass balance accounting.
- **Multi-commodity blending errors**: A facility processes both cocoa and soya, but the MB ledger fails to separate commodity-specific balances, allowing cocoa input credits to be used against soya output claims.
- **Carry-forward expiry failures**: Credits from previous periods are carried forward indefinitely, when standard-specific rules (RSPO: 3 months, FSC: 12 months) require expiry of unused credits.
- **Loss and waste under-reporting**: Processing losses of 15-25% are not properly deducted from the ledger, creating an artificial surplus of compliant volume that doesn't physically exist.
- **Period-end reconciliation gaming**: Operators time their compliant input purchases just before period-end reconciliation to create a snapshot of compliance, then immediately redirect the material to non-compliant channels.

Without a rigorous, deterministic mass balance calculation engine, operators using the MB model cannot prove to competent authorities that their administrative tracking accurately reflects the physical flow of compliant material, resulting in non-compliant DDS submissions and penalties of up to 4% of annual EU turnover.

### 1.2 Solution Overview

Agent-EUDR-011: Mass Balance Calculator Agent provides a production-grade, zero-hallucination mass balance accounting engine for EUDR compliance. It maintains per-facility, per-commodity input/output ledgers with configurable credit periods, enforces conversion factor validation against peer-reviewed reference data, detects overdraft conditions in real-time, manages carry-forward with standard-specific expiry rules, and generates auditor-ready reconciliation reports.

Core capabilities:

1. **Ledger management** -- Maintains double-entry input/output ledgers per facility per commodity per credit period, with running balance calculations, transaction history, and immutable audit trail.
2. **Credit period engine** -- Manages credit period lifecycle (open, active, reconciling, closed) with configurable durations (3-month RSPO, 12-month FSC, custom EUDR), automatic period rollover, and carry-forward with expiry.
3. **Conversion factor validator** -- Validates conversion factors (yield ratios) against peer-reviewed commodity-specific reference data, flagging deviations beyond configurable tolerance thresholds.
4. **Overdraft detection** -- Real-time monitoring of ledger balance with configurable overdraft thresholds, automatic alerting when compliant output approaches or exceeds compliant input, and overdraft severity classification.
5. **Loss and waste tracker** -- Tracks processing losses, waste, and by-products as mandatory deductions from the mass balance, enforcing commodity-specific maximum loss tolerances.
6. **Carry-forward manager** -- Manages credit carry-forward between periods with standard-specific expiry rules, partial carry-forward, and expiry notification.
7. **Reconciliation engine** -- Period-end reconciliation with variance analysis, trend detection, anomaly flagging, and competent authority report generation.
8. **Multi-facility consolidation** -- Consolidates mass balance data across multiple facilities for enterprise-level reporting while enforcing per-facility accounting boundaries.

### 1.3 Dependencies

| Dependency | Component | Integration |
|------------|-----------|-------------|
| AGENT-EUDR-001 | Supply Chain Mapping Master | Facility nodes, supply chain graph |
| AGENT-EUDR-008 | Multi-Tier Supplier Tracker | Supplier profiles, facility capabilities |
| AGENT-EUDR-009 | Chain of Custody Agent | CoC model assignments, batch data, custody events, existing mass balance ledger |
| AGENT-EUDR-010 | Segregation Verifier | Segregation status for MB/SG model transitions |
| AGENT-DATA-005 | EUDR Traceability Connector | Raw mass balance data intake |

---

## 2. Regulatory Context

### 2.1 EUDR Articles Addressed

| Article | Requirement | Agent Feature |
|---------|-------------|---------------|
| Art. 4 | Due diligence obligation | Mass balance proves compliant volume accounting |
| Art. 9(1)(f) | Quantity/weight of product | Ledger tracks exact quantities per batch |
| Art. 10(2)(f) | Adequate compliance information | Reconciliation reports provide audit evidence |
| Art. 12 | Simplified due diligence for low-risk | Credit period flexibility per risk level |
| Art. 14 | 5-year record retention | Immutable ledger with full transaction history |
| Art. 16 | Risk mitigation measures | Overdraft detection as risk mitigation |
| Art. 31 | Review and reporting | Period-end reconciliation and analytics |

### 2.2 Mass Balance Standards

| Standard | Credit Period | Carry-Forward | Conversion Factors | Overdraft Rule |
|----------|--------------|---------------|--------------------|--------------------|
| RSPO SCC 2020 | 3 months | Allowed, expires at period end | MPOB reference yields | Zero overdraft |
| FSC-STD-40-004 v3 | 12 months | Allowed, no expiry within period | FSC conversion tables | Zero overdraft |
| ISCC 202 v4 | 12 months | Allowed, expires at period end | ISCC default values | 5% tolerance |
| UTZ/RA CoC | 12 months | Allowed, limited carry-forward | Industry averages | Zero overdraft |
| EUDR (proposed) | Configurable | Standard-specific | Peer-reviewed reference | Zero overdraft (default) |
| Fairtrade | 12 months | Limited carry-forward | Fairtrade tables | Zero overdraft |

---

## 3. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Ledger accuracy | 100% bit-perfect reproducibility | Same inputs produce identical balances |
| Overdraft detection rate | 100% of overdrafts detected within 1 transaction | Real-time monitoring |
| Conversion factor validation | 100% of factors checked against reference | Factor coverage score |
| Reconciliation accuracy | <= 0.01% variance from expected | Period-end variance |
| Credit expiry enforcement | 100% of expired credits detected and voided | Expiry coverage |
| Processing throughput | >= 2,000 ledger transactions/second | Batch processing benchmark |
| Reconciliation time | < 1 second per facility per period | P95 latency |
| Test coverage | >= 500 unit tests | Pytest count |

---

## 4. Scope

### 4.1 In Scope
- All 7 EUDR commodities + derived products under Mass Balance CoC model
- Double-entry input/output ledger per facility per commodity
- Credit period management (3-month, 12-month, custom)
- Conversion factor validation against reference data
- Real-time overdraft detection and alerting
- Processing loss and waste tracking
- Credit carry-forward with standard-specific expiry
- Period-end reconciliation with variance analysis
- Multi-facility consolidation reporting
- Anomaly and trend detection in mass balance patterns
- 5-year immutable transaction history per Article 14

### 4.2 Out of Scope
- Physical segregation verification (handled by EUDR-010)
- CoC model assignment and enforcement (handled by EUDR-009)
- Batch lifecycle management (handled by EUDR-009)
- Financial cost tracking
- IoT-based real-time weight monitoring

---

## 5. Zero-Hallucination Principles

1. All balance calculations use deterministic arithmetic (addition, subtraction, multiplication) -- no LLM inference.
2. Conversion factors come from peer-reviewed reference data with documented sources (MPOB, ICCO, ICO, NOPA, IRSG, ITTO, FAO).
3. Overdraft detection is pure arithmetic comparison (output > input = overdraft) -- no probabilistic assessment.
4. Credit expiry is deterministic datetime comparison -- no approximation.
5. Reconciliation variance is exact arithmetic (expected - actual = variance) -- no statistical estimation.
6. SHA-256 provenance hashing ensures tamper detection on all ledger entries and results.

---

## 6. Feature Requirements

### 6.1 Feature 1: Ledger Management (P0)

**Requirements**:
- F1.1: Create and manage double-entry ledgers per facility per commodity per credit period
- F1.2: Ledger entry types: input (compliant material received), output (compliant material dispatched), adjustment (correction), loss (processing loss), waste (waste/by-product), carry_forward_in (from previous period), carry_forward_out (to next period), expiry (expired credits voided)
- F1.3: Each entry records: entry_id, ledger_id, entry_type, batch_id, quantity_kg, compliance_status, source/destination, conversion_factor_applied, timestamp, operator_id
- F1.4: Running balance calculation: sum(inputs + carry_forward_in) - sum(outputs + losses + waste + carry_forward_out + expiry) = current_balance
- F1.5: Balance must be >= 0 at all times (or within configurable overdraft tolerance)
- F1.6: Ledger immutability: entries cannot be deleted or modified; corrections via adjustment entries only
- F1.7: Transaction ordering: entries processed in strict chronological order
- F1.8: Ledger search: by facility, commodity, period, date range, batch_id, entry_type
- F1.9: Ledger summary: total_inputs, total_outputs, total_losses, total_waste, current_balance, utilization_rate
- F1.10: Bulk entry import from EDI/CSV/XML sources

### 6.2 Feature 2: Credit Period Engine (P0)

**Requirements**:
- F2.1: Credit period lifecycle: pending -> active -> reconciling -> closed
- F2.2: Configurable period durations: 3 months (RSPO), 12 months (FSC/ISCC/UTZ), custom (30-365 days)
- F2.3: Automatic period creation on first entry for facility+commodity combination
- F2.4: Automatic period rollover when current period expires
- F2.5: Period-end lock: no new entries after period enters reconciling state
- F2.6: Grace period: configurable window (default 5 business days) after period end for late entries
- F2.7: Period metadata: period_id, facility_id, commodity, start_date, end_date, status, standard, carry_forward_balance
- F2.8: Historical period browsing: view any past period's ledger and reconciliation
- F2.9: Period overlap prevention: no two active periods for same facility+commodity
- F2.10: Period extension: authorized extension of period end date with audit trail

### 6.3 Feature 3: Conversion Factor Validator (P0)

**Requirements**:
- F3.1: Validate conversion factors against commodity-specific reference data (MPOB, ICCO, ICO, NOPA, IRSG, ITTO, FAO)
- F3.2: 30+ commodity conversion pairs with reference yield ratios and acceptable ranges
- F3.3: Tolerance bands: warn (>5% deviation), reject (>15% deviation) from reference, configurable per commodity
- F3.4: Conversion factor application: input_quantity * conversion_factor = expected_output_quantity
- F3.5: Multi-step conversion chains: raw -> intermediate -> final with cumulative factor validation
- F3.6: Seasonal adjustment factors for agricultural commodities (cocoa, coffee, palm oil)
- F3.7: Process-specific factors: different yields for different processing methods (wet vs dry processing)
- F3.8: Conversion factor history: track all factors used per facility with timestamp
- F3.9: Custom factor approval workflow: facility-specific factors require approval with justification
- F3.10: Factor deviation reporting: trend analysis of actual vs reference factors per facility

### 6.4 Feature 4: Overdraft Detection (P0)

**Requirements**:
- F4.1: Real-time balance check on every output entry: reject if balance would go negative (zero-overdraft mode)
- F4.2: Configurable overdraft tolerance: percentage-based (e.g., 5% ISCC) or absolute quantity
- F4.3: Overdraft severity classification: warning (approaching limit), violation (exceeded limit), critical (>2x limit)
- F4.4: Overdraft alert generation with affected batch_ids, quantities, and recommended actions
- F4.5: Overdraft history tracking per facility per commodity
- F4.6: Overdraft resolution: require matching input entry within configurable timeframe (default 48 hours)
- F4.7: Automatic output rejection when in critical overdraft state
- F4.8: Overdraft trend analysis: detect facilities with recurring overdraft patterns
- F4.9: Pre-output balance forecast: simulate output impact before recording
- F4.10: Overdraft exemption management: authorized temporary overdraft with approval and expiry

### 6.5 Feature 5: Loss and Waste Tracker (P0)

**Requirements**:
- F5.1: Mandatory loss recording for every processing transformation
- F5.2: Commodity-specific maximum loss tolerances (e.g., cocoa beans->nibs: 10-15% loss, palm FFB->CPO: 77-80% loss)
- F5.3: Loss types: processing_loss, transport_loss, storage_loss, quality_rejection, spillage, contamination_loss
- F5.4: Waste types: by_product (valuable), waste_material (non-valuable), hazardous_waste
- F5.5: By-product credit: valuable by-products credited back to balance at by-product conversion rate
- F5.6: Loss validation: flag losses outside expected range (too low = under-reporting, too high = potential fraud)
- F5.7: Cumulative loss tracking: total loss percentage across all processing steps per batch
- F5.8: Loss trend analysis per facility per commodity per process type
- F5.9: Waste documentation linkage: link waste entries to waste certificates/disposal records
- F5.10: Loss allocation: when batch is split, losses allocated proportionally to sub-batches

### 6.6 Feature 6: Carry-Forward Manager (P0)

**Requirements**:
- F6.1: Automatic carry-forward of positive balance at period end to next period
- F6.2: Standard-specific expiry rules: RSPO (expires at end of receiving period), FSC (no expiry within period), ISCC (expires at period end)
- F6.3: Partial carry-forward: option to carry forward only a portion of balance
- F6.4: Carry-forward cap: configurable maximum carry-forward as percentage of period inputs (default 100%)
- F6.5: Carry-forward entry creation: automatic carry_forward_out in closing period, carry_forward_in in opening period
- F6.6: Expiry notification: alert when carry-forward credits are approaching expiry date
- F6.7: Expired credit voiding: automatic entry of expiry entries to void expired carry-forward credits
- F6.8: Carry-forward audit trail: full history of carry-forward amounts between periods
- F6.9: Negative balance at period end: flag as critical non-compliance, prevent period closure
- F6.10: Carry-forward reporting: summary of credits carried, credits expired, credits utilized per period

### 6.7 Feature 7: Reconciliation Engine (P0)

**Requirements**:
- F7.1: Period-end reconciliation: compare expected balance (inputs - outputs - losses) with recorded balance
- F7.2: Variance calculation: absolute and percentage variance between expected and recorded
- F7.3: Variance classification: acceptable (<= 1%), warning (1-3%), violation (>3%)
- F7.4: Anomaly detection: flag unusual patterns (sudden spikes, consistent small overdrafts, timing anomalies)
- F7.5: Trend analysis: balance trends over multiple periods per facility
- F7.6: Cross-facility comparison: benchmark facility mass balance performance against peers
- F7.7: Reconciliation sign-off: authorized user approval of reconciliation results
- F7.8: Reconciliation report generation: detailed breakdown of all entries, balance movements, variances
- F7.9: Regulatory compliance check: verify reconciliation meets RSPO/FSC/ISCC/EUDR requirements
- F7.10: Automatic re-reconciliation: trigger re-reconciliation when late entries are received during grace period

### 6.8 Feature 8: Multi-Facility Consolidation (P0)

**Requirements**:
- F8.1: Enterprise-level dashboard: aggregate mass balance across all facilities
- F8.2: Facility-level isolation: strict per-facility accounting with no cross-facility credit transfer
- F8.3: Cross-facility transfer tracking: record material transfers between facilities with proper ledger entries at both ends
- F8.4: Consolidation reporting: enterprise-wide compliance summary, facility comparison, commodity breakdown
- F8.5: Facility grouping: organize facilities by region, country, commodity, or custom hierarchy
- F8.6: Consolidated reconciliation: enterprise-wide reconciliation with drill-down to facility level
- F8.7: Report formats: JSON, PDF, CSV, EUDR XML
- F8.8: Regulatory evidence package: compiled documentation for competent authority inspections

---

## 7. Technical Requirements

### 7.1 Architecture

```
greenlang/agents/eudr/mass_balance_calculator/
    __init__.py                          # Package exports (80+ symbols)
    config.py                            # MassBalanceCalculatorConfig singleton
    models.py                            # Pydantic v2 models, enums
    provenance.py                        # SHA-256 chain hashing
    metrics.py                           # Prometheus metrics (gl_eudr_mbc_ prefix)
    ledger_manager.py                    # Engine 1: Double-entry ledger management
    credit_period_engine.py              # Engine 2: Period lifecycle management
    conversion_factor_validator.py       # Engine 3: Factor validation
    overdraft_detector.py                # Engine 4: Real-time overdraft detection
    loss_waste_tracker.py                # Engine 5: Loss and waste tracking
    carry_forward_manager.py             # Engine 6: Credit carry-forward with expiry
    reconciliation_engine.py             # Engine 7: Period-end reconciliation
    consolidation_reporter.py            # Engine 8: Multi-facility consolidation
    setup.py                             # MassBalanceCalculatorService facade
    reference_data/
        __init__.py
        conversion_factors.py            # Commodity conversion/yield ratios (30+)
        loss_tolerances.py               # Commodity-specific loss limits
        credit_period_rules.py           # Standard-specific period/carry-forward rules
    api/
        __init__.py
        router.py
        schemas.py
        dependencies.py
        ledger_routes.py                 # Ledger CRUD and transaction routes
        period_routes.py                 # Credit period management routes
        factor_routes.py                 # Conversion factor validation routes
        overdraft_routes.py              # Overdraft detection and alert routes
        loss_routes.py                   # Loss and waste tracking routes
        reconciliation_routes.py         # Reconciliation and reporting routes
        consolidation_routes.py          # Multi-facility consolidation routes
```

### 7.2 Database Schema (V099)

| Table | Type | Description |
|-------|------|-------------|
| `gl_eudr_mbc_ledgers` | regular | Ledger master records (facility+commodity+period) |
| `gl_eudr_mbc_ledger_entries` | hypertable (monthly) | Double-entry transaction records |
| `gl_eudr_mbc_credit_periods` | regular | Credit period lifecycle records |
| `gl_eudr_mbc_conversion_factors` | regular | Applied conversion factors with validation status |
| `gl_eudr_mbc_overdraft_events` | hypertable (monthly) | Overdraft detection and alert records |
| `gl_eudr_mbc_loss_records` | hypertable (monthly) | Processing loss and waste tracking |
| `gl_eudr_mbc_carry_forwards` | regular | Credit carry-forward between periods |
| `gl_eudr_mbc_reconciliations` | regular | Period-end reconciliation results |
| `gl_eudr_mbc_facility_groups` | regular | Facility grouping for consolidation |
| `gl_eudr_mbc_consolidation_reports` | regular | Multi-facility consolidated reports |
| `gl_eudr_mbc_batch_jobs` | regular | Batch processing jobs |
| `gl_eudr_mbc_audit_log` | regular | Immutable audit trail |

### 7.3 Prometheus Metrics (18 metrics, `gl_eudr_mbc_` prefix)

| Metric | Type | Description |
|--------|------|-------------|
| `gl_eudr_mbc_ledger_entries_total` | Counter | Total ledger entries recorded |
| `gl_eudr_mbc_input_entries_total` | Counter | Total compliant input entries |
| `gl_eudr_mbc_output_entries_total` | Counter | Total compliant output entries |
| `gl_eudr_mbc_overdrafts_detected_total` | Counter | Overdraft events detected |
| `gl_eudr_mbc_overdrafts_critical_total` | Counter | Critical overdraft events |
| `gl_eudr_mbc_conversion_validations_total` | Counter | Conversion factor validations |
| `gl_eudr_mbc_conversion_rejections_total` | Counter | Conversion factor rejections |
| `gl_eudr_mbc_losses_recorded_total` | Counter | Loss/waste entries recorded |
| `gl_eudr_mbc_credits_expired_total` | Counter | Carry-forward credits expired |
| `gl_eudr_mbc_reconciliations_total` | Counter | Period reconciliations completed |
| `gl_eudr_mbc_reports_generated_total` | Counter | Reports generated |
| `gl_eudr_mbc_batch_jobs_total` | Counter | Batch processing jobs |
| `gl_eudr_mbc_api_errors_total` | Counter | API errors |
| `gl_eudr_mbc_entry_recording_duration_seconds` | Histogram | Ledger entry recording latency |
| `gl_eudr_mbc_reconciliation_duration_seconds` | Histogram | Reconciliation processing latency |
| `gl_eudr_mbc_overdraft_check_duration_seconds` | Histogram | Overdraft check latency |
| `gl_eudr_mbc_active_ledgers` | Gauge | Currently active ledgers |
| `gl_eudr_mbc_total_balance_kg` | Gauge | Total compliant balance across all ledgers |

### 7.4 API Endpoints (~37 endpoints)

| Group | Method | Path | Description |
|-------|--------|------|-------------|
| Ledger | POST | `/api/v1/eudr-mbc/ledgers` | Create ledger |
| | GET | `/api/v1/eudr-mbc/ledgers/{ledger_id}` | Get ledger details |
| | POST | `/api/v1/eudr-mbc/ledgers/entries` | Record ledger entry |
| | POST | `/api/v1/eudr-mbc/ledgers/entries/bulk` | Bulk entry import |
| | GET | `/api/v1/eudr-mbc/ledgers/{ledger_id}/balance` | Get current balance |
| | GET | `/api/v1/eudr-mbc/ledgers/{ledger_id}/history` | Get entry history |
| | POST | `/api/v1/eudr-mbc/ledgers/search` | Search ledgers |
| Period | POST | `/api/v1/eudr-mbc/periods` | Create credit period |
| | GET | `/api/v1/eudr-mbc/periods/{period_id}` | Get period details |
| | PUT | `/api/v1/eudr-mbc/periods/{period_id}` | Update period (extend) |
| | POST | `/api/v1/eudr-mbc/periods/rollover` | Trigger period rollover |
| | GET | `/api/v1/eudr-mbc/periods/active/{facility_id}` | Get active periods |
| Factor | POST | `/api/v1/eudr-mbc/factors/validate` | Validate conversion factor |
| | GET | `/api/v1/eudr-mbc/factors/reference/{commodity}` | Get reference factors |
| | POST | `/api/v1/eudr-mbc/factors/custom` | Register custom factor |
| | GET | `/api/v1/eudr-mbc/factors/history/{facility_id}` | Get factor usage history |
| Overdraft | POST | `/api/v1/eudr-mbc/overdraft/check` | Check overdraft status |
| | GET | `/api/v1/eudr-mbc/overdraft/alerts/{facility_id}` | Get active alerts |
| | POST | `/api/v1/eudr-mbc/overdraft/forecast` | Forecast output impact |
| | POST | `/api/v1/eudr-mbc/overdraft/exemption` | Request overdraft exemption |
| | GET | `/api/v1/eudr-mbc/overdraft/history/{facility_id}` | Get overdraft history |
| Loss | POST | `/api/v1/eudr-mbc/losses` | Record processing loss |
| | GET | `/api/v1/eudr-mbc/losses/{facility_id}` | Get loss records |
| | POST | `/api/v1/eudr-mbc/losses/validate` | Validate loss against tolerance |
| | GET | `/api/v1/eudr-mbc/losses/trends/{facility_id}` | Get loss trends |
| Reconciliation | POST | `/api/v1/eudr-mbc/reconciliation` | Run period reconciliation |
| | GET | `/api/v1/eudr-mbc/reconciliation/{reconciliation_id}` | Get reconciliation result |
| | POST | `/api/v1/eudr-mbc/reconciliation/sign-off` | Sign off reconciliation |
| | GET | `/api/v1/eudr-mbc/reconciliation/history/{facility_id}` | Get reconciliation history |
| Consolidation | POST | `/api/v1/eudr-mbc/consolidation/report` | Generate consolidation report |
| | POST | `/api/v1/eudr-mbc/consolidation/groups` | Create facility group |
| | GET | `/api/v1/eudr-mbc/consolidation/dashboard` | Get enterprise dashboard |
| | GET | `/api/v1/eudr-mbc/consolidation/report/{report_id}` | Get report |
| | GET | `/api/v1/eudr-mbc/consolidation/report/{report_id}/download` | Download report |
| Batch | POST | `/api/v1/eudr-mbc/batch` | Submit batch job |
| | DELETE | `/api/v1/eudr-mbc/batch/{job_id}` | Cancel batch job |
| Health | GET | `/api/v1/eudr-mbc/health` | Health check |

---

## 8. Test Strategy

### 8.1 Unit Tests (500+)
- Ledger creation, entry recording, balance calculation for all entry types
- Credit period lifecycle (pending -> active -> reconciling -> closed)
- Conversion factor validation against reference data with tolerance bands
- Overdraft detection for zero-tolerance and percentage-tolerance modes
- Loss and waste recording with tolerance validation
- Carry-forward with standard-specific expiry rules (RSPO 3mo, FSC 12mo, ISCC 12mo)
- Period-end reconciliation with variance analysis
- Multi-facility consolidation with per-facility isolation
- Anomaly detection in mass balance patterns
- Edge cases: zero balance, maximum carry-forward, concurrent entries

### 8.2 Performance Tests
- Bulk entry import of 100,000 ledger transactions
- Reconciliation across 1,000 facilities
- Consolidation reporting for 500-facility enterprise

---

## Appendices

### A. Conversion Factors by Commodity (Reference Data)

| Commodity | Process | Input | Output | Yield Ratio | Acceptable Range | Source |
|-----------|---------|-------|--------|-------------|------------------|--------|
| Cocoa | Beans -> Nibs | Beans | Nibs | 0.87 | 0.82-0.92 | ICCO |
| Cocoa | Nibs -> Liquor | Nibs | Liquor | 0.80 | 0.75-0.85 | ICCO |
| Cocoa | Liquor -> Butter | Liquor | Butter | 0.45 | 0.40-0.50 | ICCO |
| Cocoa | Liquor -> Powder | Liquor | Powder | 0.55 | 0.50-0.60 | ICCO |
| Palm oil | FFB -> CPO | FFB | CPO | 0.215 | 0.18-0.25 | MPOB |
| Palm oil | FFB -> PKO | FFB | PKO | 0.035 | 0.025-0.045 | MPOB |
| Palm oil | CPO -> RBD PO | CPO | RBD PO | 0.92 | 0.88-0.96 | PORAM |
| Coffee | Cherry -> Green | Cherry | Green | 0.185 | 0.15-0.22 | ICO |
| Coffee | Green -> Roasted | Green | Roasted | 0.825 | 0.78-0.87 | SCA |
| Soya | Beans -> Oil | Beans | Oil | 0.19 | 0.16-0.22 | NOPA |
| Soya | Beans -> Meal | Beans | Meal | 0.795 | 0.75-0.84 | NOPA |
| Rubber | Latex -> Sheet RSS | Latex | RSS | 0.325 | 0.28-0.37 | IRSG |
| Rubber | Latex -> Block | Latex | Block | 0.30 | 0.25-0.35 | IRSG |
| Wood | Log -> Sawn Timber | Log | Sawn | 0.50 | 0.40-0.60 | ITTO |
| Wood | Log -> Veneer | Log | Veneer | 0.55 | 0.45-0.65 | ITTO |
| Cattle | Live -> Carcass | Live | Carcass | 0.55 | 0.48-0.62 | FAO |
| Cattle | Live -> Hide | Live | Hide | 0.07 | 0.05-0.09 | ICHSLTA |

### B. Loss Tolerances by Commodity

| Commodity | Process | Expected Loss % | Max Tolerance % | Source |
|-----------|---------|-----------------|-----------------|--------|
| Cocoa | Beans -> Nibs (shelling) | 13% | 20% | ICCO |
| Cocoa | Nibs -> Liquor (grinding) | 2% | 5% | ICCO |
| Palm oil | FFB -> CPO (milling) | 78.5% | 82% | MPOB |
| Coffee | Cherry -> Green (processing) | 81.5% | 85% | ICO |
| Soya | Beans -> Oil (extraction) | 1% | 3% | NOPA |
| Rubber | Latex -> Sheet (drying) | 67.5% | 72% | IRSG |
| Wood | Log -> Sawn (sawing) | 50% | 60% | ITTO |
| Cattle | Live -> Carcass (slaughter) | 45% | 52% | FAO |

### C. Credit Period Rules by Standard

| Standard | Duration | Carry-Forward | Expiry Rule | Grace Period | Overdraft |
|----------|----------|---------------|-------------|-------------|-----------|
| RSPO SCC | 3 months | Allowed | Expires end of receiving period | 5 business days | Zero |
| FSC-STD-40-004 | 12 months | Allowed | No expiry within active period | 10 business days | Zero |
| ISCC 202 | 12 months | Allowed | Expires end of receiving period | 5 business days | 5% tolerance |
| UTZ/RA CoC | 12 months | Limited (50%) | Expires end of receiving period | 5 business days | Zero |
| Fairtrade | 12 months | Limited (25%) | Expires end of receiving period | 10 business days | Zero |
| EUDR default | 12 months | Allowed | Configurable | 5 business days | Zero |

### D. Overdraft Severity Classification

| Level | Threshold | Action | Resolution Time |
|-------|-----------|--------|-----------------|
| Warning | Balance < 10% of period inputs | Alert generated | Informational |
| Violation | Balance < 0 (or < -tolerance) | Output flagged | 48 hours |
| Critical | Balance < -2x tolerance | Output rejected | Immediate |
