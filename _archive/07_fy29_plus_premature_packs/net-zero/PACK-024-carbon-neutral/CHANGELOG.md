# PACK-024 Carbon Neutral Pack - Changelog

All notable changes to this pack are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-03-18

### Summary

Initial production release of the PACK-024 Carbon Neutral Pack. Provides a
complete, standalone carbon neutrality management solution covering the full
lifecycle from GHG footprint quantification through carbon credit procurement,
registry retirement, neutralization balance, claims substantiation, third-party
verification, and annual cycle management.

### Added

#### Engines (10)

- **Footprint Quantification Engine** -- ISO 14064-1:2018 aligned GHG footprint
  quantification with Scope 1/2/3 breakdown, multi-facility aggregation, three
  consolidation approaches (equity share, operational control, financial control),
  data quality scoring with 5-tier DQIS rubric, and uncertainty assessment per
  IPCC 2006 Guidelines.

- **Carbon Management Plan Engine** -- Reduction-first strategy generation per
  ISO 14068-1 and PAS 2060 with MACC (Marginal Abatement Cost Curve) analysis,
  multi-year reduction trajectory with annual milestones, action prioritization
  by cost-effectiveness, and residual emissions projection.

- **Credit Quality Engine** -- 12-dimension ICVCM Core Carbon Principles scoring
  (additionality 0.15, permanence 0.12, quantification 0.10, third-party
  validation 0.08, unique claim 0.08, co-benefits 0.07, safeguards 0.07,
  net-zero contribution 0.08, environmental integrity 0.07, social integrity
  0.06, governance 0.06, transparency 0.06). Quality rating from A+ to F.
  Supports 14 credit project types.

- **Portfolio Optimization Engine** -- Markowitz-inspired credit portfolio
  optimization across avoidance/removal and nature-based/technology-based axes.
  Configurable constraints for minimum removal percentage, maximum nature-based
  percentage, vintage age limits, and geographic diversification. Oxford
  Principles alignment with increasing removal share over time.

- **Registry Retirement Engine** -- Retirement tracking across 6 registries:
  Verra VCS, Gold Standard, ACR, CAR, Puro.earth, and Isometric. Serial number
  verification, retirement certificate generation, double-counting prevention,
  and Article 6 corresponding adjustment support.

- **Neutralization Balance Engine** -- Balance calculation per ISO 14068-1:2023
  Clause 9 and PAS 2060:2014. Six balance methods: corporate_total,
  per_unit_produced, event_total, project_total, entity_level, and
  service_revenue. Surplus/deficit position, buffer adequacy, confidence
  interval, and qualifying explanatory statement generation.

- **Claims Substantiation Engine** -- Carbon neutrality claims validation per
  ISO 14068-1, PAS 2060, and VCMI Claims Code of Practice. VCMI precondition
  checks (science-aligned targets, Scope 1+2 reductions, public disclosure).
  Platinum/Gold/Silver tier assessment. Qualifying statement generation with
  required caveats and limitations.

- **Verification Package Engine** -- ISAE 3410 aligned evidence package assembly
  with SHA-256 content hashing for integrity verification. Evidence index
  compilation, methodology documentation, gap analysis with remediation
  guidance. Supports limited assurance and reasonable assurance engagement types.

- **Annual Cycle Engine** -- Multi-year carbon neutrality cycle management with
  configurable milestone frequency (monthly, quarterly, biannual). Annual
  renewal process management, base year recalculation trigger detection,
  year-over-year comparison, and forward projection generation.

- **Permanence Risk Engine** -- Reversal risk assessment across nature-based
  (25-100 year permanence), technology-based (1000+ year permanence), and
  hybrid credit categories. Buffer pool contribution calculation (10-20%),
  reversal event monitoring, climate hazard integration for forward-looking
  risk projections.

#### Workflows (8)

- **Full Annual Cycle Workflow** -- 10-phase end-to-end orchestration covering
  the complete carbon neutrality lifecycle. Annual schedule, 120-minute target.

- **Footprint Assessment Workflow** -- 4-phase GHG quantification: data
  collection, calculation, quality assessment, and reporting.

- **Carbon Management Plan Workflow** -- 5-phase planning: baseline review,
  action identification, MACC analysis, trajectory generation, and plan review.

- **Credit Procurement Workflow** -- 4-phase procurement: requirements
  assessment, market survey, quality screening, and procurement execution.

- **Retirement Workflow** -- 3-phase registry retirement: credit selection,
  retirement execution, and documentation generation.

- **Neutralization Workflow** -- 5-phase balance calculation: aggregation,
  balance calculation, surplus/deficit assessment, gap remediation, and
  statement generation.

- **Claims Validation Workflow** -- 4-phase claims process: precondition
  verification, tier assessment, statement generation, and disclosure
  preparation.

- **Verification Workflow** -- 4-phase ISAE 3410 package: evidence compilation,
  package assembly, gap check, and final delivery.

#### Templates (10)

- **Footprint Report** -- Scope 1/2/3 breakdown with data quality scores,
  emission factor sources, uncertainty ranges, and year-over-year comparison.

- **Carbon Management Plan Report** -- MACC curve visualization, reduction
  trajectory with milestones, action prioritization, and investment timeline.

- **Credit Portfolio Report** -- Per-credit ICVCM quality assessment, benchmark
  comparison, SDG contribution analysis, and improvement recommendations.

- **Registry Retirement Report** -- Retirement certificates with serial numbers,
  registry references, beneficiary designation, and QR code links.

- **Neutralization Statement Report** -- Balance statement per ISO 14068-1 with
  qualifying explanatory statement per PAS 2060 Section 9.

- **Claims Substantiation Report** -- Public claims disclosure with VCMI tier,
  verification status, carbon management plan summary, and credit composition.

- **Verification Package Report** -- ISAE 3410 evidence index with SHA-256
  hashes, methodology summary, and assurance opinion template.

- **Annual Progress Report** -- Multi-year footprint trend, reduction progress,
  credit portfolio evolution, balance history, and forward projections.

- **Permanence Risk Report** -- Credit-level and portfolio-level risk scores,
  reversal monitoring status, buffer pool adequacy, and climate hazard exposure.

- **Public Disclosure Report** -- Public-facing carbon neutrality statement
  with qualifying explanations and links to supporting evidence.

#### Integrations (12)

- **Pack Orchestrator** -- 10-phase DAG pipeline with dependency resolution,
  exponential backoff retry, and SHA-256 provenance tracking.

- **MRV Bridge** -- Routes emission data to all 30 AGENT-MRV agents for
  complete Scope 1/2/3 quantification.

- **GHG App Bridge** -- Bidirectional connection to GL-GHG-APP v1.0 for
  inventory, base year, scope aggregation, and data quality.

- **DECARB Bridge** -- Routes to DECARB agents for reduction planning,
  technology assessment, MACC generation, and progress monitoring.

- **Data Bridge** -- Connects to all 20 AGENT-DATA agents for data intake
  (PDF/Excel/ERP/API) and quality management.

- **Registry Bridge** -- API integration with Verra, Gold Standard, ACR, CAR,
  Puro.earth, and Isometric for retirement and validation.

- **Credit Marketplace Bridge** -- Price discovery, credit availability,
  procurement automation, and ICVCM CCP quality screening.

- **Verification Body Bridge** -- ISAE 3410 engagement management, evidence
  package delivery, and verification opinion tracking.

- **PACK-021 Bridge** -- Optional bridge to Net Zero Starter Pack for baseline,
  gap analysis, and roadmap capabilities.

- **PACK-023 Bridge** -- Optional bridge to SBTi Alignment Pack for target
  validation, pathway analysis, and temperature scoring.

- **Health Check** -- 20-category system verification covering engines,
  registries, databases, agents, and reference data.

- **Setup Wizard** -- 6-step guided configuration for organization profile,
  boundaries, credit preferences, portfolio strategy, claims, and validation.

#### Presets (8)

- `corporate_neutrality` -- Full organizational Scope 1+2+3, ISO 14068-1 +
  PAS 2060, minimum quality score 70, reasonable assurance.

- `sme_neutrality` -- Simplified SME Scope 1+2 only, PAS 2060, minimum quality
  score 55, limited assurance, budget-conscious portfolio.

- `event_neutrality` -- Event-specific emission sources, ISO 20121 alignment,
  event-total balance method, up to 10,000 attendees.

- `product_neutrality` -- ISO 14067 LCA methodology, cradle-to-gate/grave/gate
  boundaries, per-unit-produced balance method.

- `building_neutrality` -- CRREM pathway alignment, weather-normalized energy
  intensity, multi-building portfolio up to 200 buildings.

- `service_neutrality` -- Office and cloud operations, FTE and revenue
  normalization, cloud provider emission data integration.

- `project_neutrality` -- ISO 14064-2 and PAS 2080, lifecycle modules A1-D,
  embodied carbon quantification with EPD data.

- `portfolio_neutrality` -- Multi-entity consolidation up to 500 entities,
  equity share consolidation, PCAF methodology for financed emissions.

#### Configuration

- Pydantic v2 runtime configuration with 15+ sub-config models.
- Configuration hierarchy: pack.yaml -> preset -> environment -> runtime.
- Environment variable overrides with `CARBON_NEUTRAL_*` prefix.
- SHA-256 config hashing for reproducibility.

#### Database Migrations

- 10 new migrations (V084-PACK024-001 through V084-PACK024-010) covering
  footprint data, management plans, credit quality scores, portfolio
  allocations, registry retirement records, neutralization balance,
  claims records, verification packages, annual cycle milestones, and
  permanence risk scores.

#### Testing

- 693 tests across 15 test modules with 100% pass rate.
- Engine unit tests (10 modules), workflow tests, template tests, integration
  tests, config tests, and preset validation tests.
- 91.9% code coverage across all modules.

#### Security

- JWT (RS256) authentication with 6 role-based access levels.
- AES-256-GCM encryption at rest, TLS 1.3 in transit.
- Audit logging for all engine operations.
- PII detection and redaction.

### Known Limitations

- Credit marketplace bridge requires external marketplace API credentials for
  live price discovery; operates in simulation mode without credentials.
- Registry API bridge requires registry-specific API keys for real-time
  retirement verification; supports offline verification via manual serial
  number entry.
- Satellite connector (AGENT-DATA-007) for nature-based credit permanence
  monitoring requires separate satellite imagery subscription.
- Maximum 500 entities for portfolio_neutrality preset; contact support for
  larger portfolios.
- VCMI Claims Code assessment is based on the V1.0 (June 2023) edition;
  future versions will require configuration updates.
- Product neutrality preset requires ecoinvent 3.10 LCA database license for
  full product carbon footprint calculation.

### Dependencies

- Python >= 3.11
- PostgreSQL >= 16 with pgvector and TimescaleDB
- Redis >= 7
- GreenLang Platform >= 2.0.0
- 30 AGENT-MRV agents (v1.0.0)
- 20 AGENT-DATA agents (v1.0.0)
- 10 AGENT-FOUND agents (v1.0.0)

---

*Maintained by GreenLang Platform Team*
