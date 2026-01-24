# GreenLang MVP 2026 - Decision Log

This document records the key decisions made during PRD development based on stakeholder interviews.

---

## Decision Summary Table

| # | Area | Question | Decision | Rationale |
|---|------|----------|----------|-----------|
| 1 | Target User | Primary customer segment | Small/Mid EU Importers (10-100 lines/quarter) | Largest underserved segment, validation-sensitive |
| 2 | Business Model | Pricing approach | Open-source + paid support | Builds trust, enables enterprise upsell |
| 3 | Regulatory | CBAM scope | Transitional period only (2024-2025) | Simpler requirements, faster to market |
| 4 | Interface | Primary interaction method | CLI-first (`gl run cbam`) | Matches developer-first positioning |
| 5 | Input | PDF extraction support | Excluded from MVP | Reduces complexity, avoids nondeterminism |
| 6 | Data | Supplier data handling | Defaults-first, optional override | EU Commission defaults as baseline |
| 7 | Scope | Product categories | Steel & Iron + Aluminum only | ~70% of CBAM imports by value |
| 8 | Output | XML schema version | Latest EU Transitional Registry schema | Official compliance requirement |
| 9 | Audit | Audit bundle detail level | Full lineage + evidence | Critical for compliance defensibility |
| 10 | Errors | Validation error handling | Fail fast with actionable errors | Clear, fixable messages for users |
| 11 | Architecture | Agent chain size | Minimal (7 agents) | Manageable complexity |
| 12 | Deployment | Packaging approach | CLI + Docker Compose | No K8s required for pilots |
| 13 | Determinism | Reproducibility guarantee | Version-pinned determinism | Same inputs + versions = identical outputs |
| 14 | Testing | Coverage target | Golden datasets + unit tests (80%) | Balance speed with quality |
| 15 | Security | Security posture | Local-first, no network | Data never leaves user's machine |
| 16 | Output | Human-readable summary | Excel summary required | Review before registry submission |
| 17 | Edge Case | Missing emission factors | Fail with clear error | No silent fallbacks |
| 18 | Updates | Regulatory update handling | Versioned packs | Users pin to specific version |
| 19 | Language | Supported languages | English only | Sufficient for MVP |
| 20 | Risk | Top concern | Regulatory accuracy | Calculations must match EU methodology |
| 21 | Validation | Regulatory accuracy method | Test against official examples + expert review | Authoritative verification |
| 22 | Input | Ledger template design | Minimal mandatory + optional fields | ~10 required columns |
| 23 | Onboarding | User learning approach | Quick start + example data | README with demo_shipments.csv |
| 24 | Metrics | Success KPIs | XSD validation rate + user completion | Focus on core outcomes |
| 25 | Features | Gap report inclusion | Yes, include in MVP | Identifies data quality improvements |
| 26 | CLI | Command structure | `gl run cbam --config ...` | Follows GreenLang patterns |
| 27 | Versioning | Reproducibility tracking | Semantic versioning + hashes | Pack, factor, config, input hashes |
| 28 | Emissions | Indirect emissions handling | Include using default electricity factors | EU-published country factors |
| 29 | Aggregation | Multi-installation handling | Aggregate by product/origin | Weight-average if different installations |
| 30 | Amendments | Correction support | No, new run for corrections | Each run is standalone |
| 31 | Docs | Documentation scope | README + Quick Start + Examples | Focused, practical docs |

---

## Detailed Decision Records

### D1: Target Customer Selection

**Date:** January 2026
**Decision:** Small/Mid EU Importers (10-100 import lines/quarter)

**Context:**
- Options considered: SMB importers, large enterprises, customs consultants, freight forwarders
- SMB importers are the largest underserved segment
- Limited compliance staff means high pain from manual processes
- Lower technical barriers than large enterprises

**Consequences:**
- MVP features optimized for single-user CLI workflow
- Template design prioritizes simplicity over completeness
- Batch processing deferred to post-MVP

---

### D2: Business Model

**Date:** January 2026
**Decision:** Open-source + paid support

**Context:**
- Options: Open-source + support, freemium features, SaaS subscription, per-report pricing
- Open-source builds trust in compliance tool
- Enterprise upsell path for premium support and SLAs
- Avoids SaaS complexity (hosting, multi-tenancy)

**Consequences:**
- All core functionality free and open
- Revenue from enterprise support contracts
- Must demonstrate value before commercial adoption

---

### D3: Regulatory Scope

**Date:** January 2026
**Decision:** CBAM Transitional period only (2024-2025)

**Context:**
- Transitional period has simpler requirements (reporting only)
- Definitive phase (2026+) requires certificate purchase tracking
- Building for transitional period is faster to market

**Consequences:**
- Certificate tracking deferred to v2.0
- Schema versioning must support future expansion
- Users on MVP will need upgrade path for 2026

---

### D5: PDF Extraction Exclusion

**Date:** January 2026
**Decision:** PDF extraction excluded from MVP

**Context:**
- PDF/OCR introduces nondeterminism and low-confidence extractions
- Structured inputs (CSV/XLSX) provide reliable, auditable data
- Reducing scope accelerates delivery

**Consequences:**
- Users must prepare structured data manually
- PDFs can be attached as evidence but not parsed
- May reconsider for v1.1 based on user feedback

---

### D20: Regulatory Accuracy as Top Concern

**Date:** January 2026
**Decision:** Regulatory accuracy is the primary MVP concern

**Context:**
- Compliance tools must produce correct calculations
- Incorrect reports could lead to penalties
- User trust depends on accuracy

**Consequences:**
- Expert review required before release
- Golden datasets must include verified expected outputs
- Methodology documentation with regulatory citations
- Factor library cross-referenced against official sources

---

## Change History

| Date | Change | Rationale |
|------|--------|-----------|
| January 2026 | Initial decisions captured | PRD development interviews |

---

## Pending Decisions

| Topic | Options | Target Date | Owner |
|-------|---------|-------------|-------|
| Pilot customer selection | TBD | Before M6 | Product |
| Factor library hosting | Bundled vs. external | M3 | Engineering |
| Telemetry opt-in design | TBD | M6 | Product |
