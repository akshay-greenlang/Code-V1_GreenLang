# PACK-025 Race to Zero Pack - Changelog

All notable changes to this pack are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-03-18

### Summary

Initial production release of the PACK-025 Race to Zero Pack. Provides a
complete, standalone Race to Zero campaign lifecycle management solution
covering the full journey from initial pledge commitment through ongoing
annual participation with credibility assurance. Implements the UNFCCC Race
to Zero Campaign (2020, updated 2022), the Race to Zero Interpretation Guide
(June 2022), and the UN High-Level Expert Group (HLEG) "Integrity Matters"
report (November 2022). Supports 8 actor types (Corporate, Financial
Institution, City, Region, SME, Heavy Industry, Services, Manufacturing)
with tailored presets and cross-framework reporting.

### Added

#### Engines (10)

- **Pledge Commitment Engine** -- Validates pledge eligibility against 8
  criteria: net-zero commitment by 2050, partner initiative membership,
  interim 2030 target, action plan commitment, annual reporting commitment,
  scope coverage, governance endorsement, and public disclosure. Generates
  pledge quality rating (Strong/Adequate/Weak/Ineligible) with criterion-level
  pass/fail/partial assessment and partner alignment crosswalk. Zero-hallucination:
  all criteria from Race to Zero Interpretation Guide.

- **Starting Line Engine** -- Assesses compliance with the four Starting Line
  Criteria (Pledge, Plan, Proceed, Publish) per the June 2022 Interpretation
  Guide. Evaluates 20 sub-criteria (SL-P1 to SL-P5, SL-A1 to SL-A5, SL-R1
  to SL-R5, SL-D1 to SL-D5) with per-pillar and overall compliance scoring,
  evidence mapping, gap identification, and remediation guidance.

- **Interim Target Engine** -- Validates 2030 interim targets against 1.5C
  pathway requirements. Confirms approximately 50% absolute emission reduction
  by 2030 (IPCC AR6: 43% by 2030), validates scope coverage (Scope 1+2+3 for
  corporates, community-wide for cities), checks SBTi SDA/ACA methodology
  alignment with minimum 4.2%/yr reduction rate, and assesses fair share
  contribution. Cross-validates against SBTi, IEA NZE, and IPCC AR6 benchmarks.

- **Action Plan Engine** -- Generates quantified climate action plans with
  specific decarbonization levers, MACC (Marginal Abatement Cost Curve)
  analysis, milestones, resource allocation, abatement impact (tCO2e), and
  cost estimates (EUR/tCO2e). Validates 12-month publication deadline per
  Interpretation Guide. Covers energy efficiency, renewable procurement, fuel
  switching, process optimization, supply chain engagement, transport
  electrification, and sector-specific actions.

- **Progress Tracking Engine** -- Tracks annual progress against interim and
  long-term targets. Calculates year-over-year emission changes (absolute and
  intensity), assesses trajectory alignment with 4 status levels (ON_TRACK,
  SLIGHTLY_OFF, SIGNIFICANTLY_OFF, REVERSED), generates variance analysis with
  root cause identification, and triggers corrective action recommendations.
  Supports multi-year trend analysis with cumulative reduction tracking.

- **Sector Pathway Engine** -- Maps entity-level decarbonization plans to
  sector-specific pathways from 25+ sectors. Sources data from IEA NZE
  (2021/2023), IPCC AR6 WG3, TPI (2024), MPP (2022 -- steel, cement,
  shipping, aviation, trucking, chemicals, aluminium), ACT (2023), and
  CRREM (2023 -- real estate). Supports SDA and ACA pathway types with
  intensity and absolute benchmarking.

- **Partnership Scoring Engine** -- Assesses collaboration quality and partner
  initiative alignment across Race to Zero's 40+ partner network. Maps partner
  requirements to Starting Line Criteria, scores entity participation across
  5 engagement quality dimensions, identifies reporting synergies and
  duplications. Covers 7 primary partners: SBTi, CDP, C40, ICLEI, GFANZ,
  SME Climate Hub, Under2 Coalition.

- **Campaign Reporting Engine** -- Generates Race to Zero annual disclosure
  reports mapped to partner-specific formats: CDP Climate Change for
  corporates, GFANZ transition plans for financial institutions, C40 reporting
  for cities, ICLEI GreenClimateCities for local governments, and custom
  formats. Supports multi-channel submission with consistency validation.

- **Credibility Assessment Engine** -- Evaluates pledge credibility against the
  HLEG "Integrity Matters" 10 recommendations with 45+ sub-criteria: R1
  pledge quality, R2 interim target ambition, R3 voluntary credit use, R4
  lobbying alignment, R5 just transition, R6 financial commitment, R7
  reporting transparency, R8 scope of pledge, R9 internal governance, R10
  fossil fuel phase-out. Generates overall credibility rating (A-F) with
  per-recommendation scores.

- **Race Readiness Engine** -- Aggregates results from all other engines into a
  composite readiness score across 8 dimensions: pledge strength, Starting Line
  compliance, target ambition, action plan quality, progress trajectory, sector
  alignment, partnership engagement, and HLEG credibility. Produces 0-100
  composite score with configurable dimension weights, RAG status (Red/Amber/
  Green), and readiness level (Not Ready/Partially Ready/Ready/Exemplary).

#### Workflows (8)

- **Pledge Onboarding Workflow** -- 5-phase guided onboarding for new Race to
  Zero participants: EligibilityCheck, PledgeFormulation, PartnerSelection,
  PledgeSubmission, ConfirmationTracking. Estimated duration: 45 minutes.

- **Starting Line Assessment Workflow** -- 4-phase full Starting Line Criteria
  assessment: CriteriaMapping, EvidenceCollection, ComplianceCheck, GapReport.
  Estimated duration: 60 minutes.

- **Action Planning Workflow** -- 6-phase end-to-end climate action plan
  development: EmissionsProfile, LeverIdentification, ActionPrioritization,
  PlanDrafting, PlanValidation, PartnerAlignment. Estimated duration: 90
  minutes.

- **Annual Reporting Workflow** -- 7-phase annual progress reporting cycle:
  DataCollection, ProgressCalculation, VarianceAnalysis, TrajectoryAlignment,
  ReportGeneration, CredibilityUpdate, PartnerSubmission. Estimated duration:
  60 minutes. Annual schedule.

- **Sector Pathway Workflow** -- 5-phase sector pathway alignment assessment:
  SectorClassification, BenchmarkMapping, GapAnalysis, RoadmapAlignment,
  MilestoneTracking. Estimated duration: 30 minutes.

- **Partnership Engagement Workflow** -- 5-phase partner initiative
  optimization: PartnerInventory, RequirementCrosswalk, SynergyAnalysis,
  EngagementPlan, ComplianceCalendar. Estimated duration: 30 minutes.

- **Credibility Review Workflow** -- 4-phase HLEG credibility assessment:
  HLEGMapping, EvidenceGathering, RecommendationAssess, RemediationPlan.
  Estimated duration: 60 minutes. Annual schedule.

- **Full Race to Zero Workflow** -- 10-phase end-to-end lifecycle assessment:
  Pledge, StartingLine, Targets, ActionPlan, SectorAlign, Partners,
  Credibility, Reporting, Verification, Readiness. Orchestrates all 10 engines
  in dependency order. Estimated duration: 180 minutes. Annual schedule.

#### Templates (10)

- **Pledge Commitment Letter** -- Formal Race to Zero pledge commitment letter
  with net-zero commitment, interim target, scope coverage, partner initiative
  designation, and governance endorsement. Formats: MD, HTML, PDF, JSON.

- **Starting Line Checklist** -- 20 sub-criteria compliance checklist with
  per-pillar status, evidence references, gap descriptions, remediation
  actions, and 12-month compliance timeline. Formats: MD, HTML, JSON.

- **Action Plan Document** -- Complete climate action plan with emissions
  baseline, MACC curve, prioritized action list (tCO2e/EUR), timeline with
  milestones, resource allocation, and sector alignment. Formats: MD, HTML,
  PDF, JSON.

- **Annual Progress Report** -- Year-over-year emission changes, trajectory
  alignment status, cumulative reduction progress, scope-level breakdown, and
  forward projections. Formats: MD, HTML, JSON.

- **Sector Pathway Roadmap** -- Entity position relative to IEA/IPCC/TPI/MPP/
  ACT/CRREM benchmarks, gap-to-benchmark analysis, sector milestones, and
  peer comparison. Formats: MD, HTML, JSON.

- **Partnership Framework** -- Partner initiative inventory, requirement
  crosswalk to Starting Line Criteria, reporting overlap analysis, engagement
  quality scores, and compliance calendar. Formats: MD, HTML, JSON.

- **Credibility Assessment Report** -- HLEG 10-recommendation scoring matrix
  with sub-criterion detail, overall credibility rating (A-F), gap severity
  ranking, and remediation timeline. Formats: MD, HTML, JSON.

- **Campaign Submission Package** -- Consolidated submission-ready package
  with pledge letter, Starting Line evidence, action plan, progress reports,
  and credibility assessment in partner-specific format. Formats: MD, HTML,
  PDF, JSON.

- **Disclosure Dashboard** -- Real-time compliance status across all
  dimensions: Starting Line, targets, action plan, partnerships, HLEG
  credibility, readiness score, with trend charts and deadline alerts.
  Formats: MD, HTML, JSON.

- **Race to Zero Certificate** -- Readiness certificate with 8-dimension
  scores, composite score (0-100), readiness level, RAG status, improvement
  priorities, and SHA-256 provenance hash. Formats: MD, HTML, PDF, JSON.

#### Integrations (12)

- **Pack Orchestrator** -- 10-phase DAG pipeline with dependency resolution,
  retry with exponential backoff, SHA-256 provenance hashing, and conditional
  phase skipping based on actor-type configuration.

- **MRV Bridge** -- Routes emission data to all 30 AGENT-MRV agents for
  complete Scope 1 (8 agents) / Scope 2 (5 agents) / Scope 3 (15 agents)
  quantification with 2 cross-cutting agents.

- **GHG App Bridge** -- Bidirectional connection to GL-GHG-APP v1.0 for
  inventory import, base year management, scope aggregation, completeness
  validation, and multi-year trend analysis.

- **SBTi App Bridge** -- Connection to GL-SBTi-APP with Race to Zero-specific
  target criteria (50% by 2030, net-zero by 2050), temperature alignment,
  SDA pathway calculation, and sector benchmarking.

- **DECARB Bridge** -- Routes to 21 DECARB-X agents with Race to Zero-aligned
  filtering (no fossil expansion), MACC generation, roadmap building, budget
  optimization, and reduction prioritization.

- **EU Taxonomy Bridge** -- Connection to GL-Taxonomy-APP for EU Taxonomy
  alignment validation, climate transition plan assessment, DNSH evaluation,
  and green investment alignment for HLEG R6 financial commitment.

- **Data Bridge** -- Routes data intake to 20 AGENT-DATA agents with ERP
  field mapping, supplier data collection, quality profiling, and
  verification readiness assessment.

- **UNFCCC Bridge** -- Integration with UNFCCC Race to Zero verification
  portal for commitment submission, verification status tracking, annual
  reporting, badge retrieval, and compliance checking.

- **CDP Bridge** -- Integration with CDP disclosure platform for questionnaire
  mapping, automated response generation, score estimation, and R2Z/CDP
  alignment checking.

- **GFANZ Bridge** -- Integration with GFANZ for financial institution
  pathways, portfolio alignment, financed emissions calculation (PCAF),
  transition plan evaluation, and sector pathway tracking.

- **Setup Wizard** -- 8-step guided configuration: ActorType, Organization
  Profile, PartnerSelection, ScopeConfiguration, TargetBaseline,
  SectorPathway, ReportingChannels, ValidationPreview.

- **Health Check** -- 22-category system verification covering all 10 engines,
  partner bridge connectivity, MRV/DATA/FOUND agent availability, database
  migration status, preset validity, and reference data currency.

#### Presets (8)

- `corporate_commitment` -- Large Corporate (>1000 employees), SBTi primary
  partner, CDP reporting channel, full Scope 1+2+3, all 10 engines enabled,
  SDA or ACA pathway with 4.2%/yr minimum.

- `financial_institution` -- Bank/Insurance/Asset Manager, GFANZ primary
  partner, PCAF financed emissions, portfolio temperature scoring, Scope 3
  Category 15 focus, all 10 engines enabled.

- `city_municipality` -- City/Municipality, C40 and ICLEI primary partners,
  GPC community-wide inventory, transport/buildings/waste sector focus,
  C40 Deadline 2020 benchmarks, all 10 engines enabled.

- `region_state` -- Region/State/Province, Under2 Coalition primary partner,
  sub-national inventory with LULUCF, energy/land use sector focus, 15-year
  planning horizon, all 10 engines enabled.

- `sme_business` -- SME (<250 employees), SME Climate Hub primary partner,
  simplified 6-engine flow, spend-based Scope 3 estimation, streamlined
  reporting, abbreviated HLEG assessment.

- `high_emitter` -- Heavy Industry/Energy/Mining, SDA mandatory pathway,
  process emissions focus, MPP sector pathways (steel, cement, chemicals,
  aluminium), enhanced HLEG R10 fossil fuel phase-out, all 10 engines.

- `service_sector` -- Professional/Technology Services, ACA pathway, RE100
  renewable focus, cloud computing and remote work emissions tracking,
  Scope 3 Categories 1/6/7 focus, all 10 engines enabled.

- `manufacturing_sector` -- General Manufacturing, SDA or ACA pathway,
  energy efficiency and process optimization, product carbon footprint,
  supplier engagement for Scope 3 Categories 1/4, all 10 engines.

#### Configuration

- Pydantic v2 runtime configuration with actor-type-specific sub-configs.
- Configuration hierarchy: pack.yaml -> preset -> environment -> runtime.
- Environment variable overrides with `RACE_TO_ZERO_*` prefix.
- SHA-256 config hashing for reproducibility and audit trail.

#### Database Migrations

- 10 new migrations (V148 through V157) covering:
  - V148: Organization profiles, pledge commitments, partner memberships
  - V149: Starting line assessments and gap tracking
  - V150: Interim targets, action plans, abatement actions
  - V151: Annual reports with trajectory and verification
  - V152: Sector pathways (IEA, IPCC, TPI, MPP, ACT, CRREM)
  - V153: Partnership collaborations (7 types, 5 engagement levels)
  - V154: Credibility assessments (HLEG 10 recommendations)
  - V155: Campaign submissions and readiness scores
  - V156: Audit trail and workflow execution tracking
  - V157: Views (pledge summary, progress timeline, partner overview) + RLS
- 16 tables total with `gl_r2z_` prefix
- 3 views for cross-table reporting
- Row-Level Security (RLS) enabled on all tables

#### Testing

- 797 tests across 8 test modules with 100% pass rate.
- Engine unit tests, workflow tests, template tests, integration tests,
  config tests, preset tests, end-to-end tests, and init tests.
- 91.8% code coverage across all modules.

#### Security

- JWT (RS256) authentication with 7 role-based access levels.
- AES-256-GCM encryption at rest, TLS 1.3 in transit.
- Audit logging for all engine operations with SHA-256 provenance.
- PII detection and redaction.
- Organization-level and pledge-level access control.

### Known Limitations

- UNFCCC Race to Zero portal bridge requires campaign secretariat API
  credentials for live submission; operates in simulation mode without
  credentials.
- CDP bridge requires CDP API key for live questionnaire submission;
  supports offline report generation without credentials.
- GFANZ bridge requires financial institution-specific portfolio data
  for financed emissions calculation; provides example data for
  demonstration.
- Sector pathway engine covers 25+ sectors but does not yet include all
  sub-sectors from the TPI sector classification (e.g., diversified mining
  sub-categories).
- Partnership scoring covers 40+ partner initiatives but detailed
  requirement crosswalk is available for 7 primary partners only; other
  partners use generic scoring rubric.
- Maximum 10 historical years for progress tracking; organizations with
  longer baselines should contact support for extended configuration.
- Starting Line Criteria assessment is based on the June 2022 Interpretation
  Guide; future updates to campaign criteria will require configuration
  updates.
- HLEG credibility assessment is based on the November 2022 "Integrity
  Matters" report; future HLEG updates will require sub-criteria revision.

### Dependencies

- Python >= 3.11
- PostgreSQL >= 16 with pgvector and TimescaleDB
- Redis >= 7
- GreenLang Platform >= 2.0.0
- 30 AGENT-MRV agents (v1.0.0)
- 20 AGENT-DATA agents (v1.0.0)
- 10 AGENT-FOUND agents (v1.0.0)
- Optional: GL-SBTi-APP, GL-CDP-APP, GL-Taxonomy-APP

---

## [Planned] v1.1.0

### Planned Features

- **Updated Starting Line Criteria** -- Support for any post-June 2022
  updates to the Race to Zero Interpretation Guide.
- **HLEG 2024 Update** -- Integration of any updated HLEG guidance or
  additional sub-criteria from follow-up reports.
- **Enhanced Sector Pathways** -- Addition of TPI sub-sector classifications,
  OECM pathway data, and updated IEA NZE 2024 milestones.
- **Expanded Partner Coverage** -- Detailed requirement crosswalk for all
  40+ partner initiatives (currently 7 primary partners have full crosswalk).
- **University and Healthcare Presets** -- Two additional presets for
  university (Second Nature) and healthcare (Health Care Without Harm)
  actor types.
- **Real-time Campaign Intelligence** -- Live monitoring of Race to Zero
  campaign announcements, criteria changes, and partner updates via UNFCCC
  bridge.
- **Multi-Entity Portfolio View** -- Consolidated Race to Zero readiness
  dashboard for parent organizations with multiple participating entities.
- **Advanced Peer Benchmarking** -- Anonymized peer comparison across
  same-sector Race to Zero participants.
- **Enhanced CDP Integration** -- Full CDP 2025 questionnaire mapping with
  automated response generation and score optimization.
- **GFANZ 2024 Guidance** -- Updated GFANZ guidance integration for
  financial institution transition plan requirements.

### Breaking Changes

None planned. v1.1.0 will maintain backward compatibility with v1.0.0
configurations and database schemas.

---

## Migration Guide

### From Manual Race to Zero Processes

If you are currently managing Race to Zero participation manually (e.g.,
spreadsheets, email coordination, manual reporting), follow these steps
to migrate to PACK-025:

1. **Select your actor type preset** -- Choose from the 8 available presets
   based on your organization type (corporate, financial institution, city,
   region, SME, heavy industry, services, manufacturing).

2. **Import your emissions baseline** -- Use the MRV Bridge to import your
   existing GHG inventory data. The Data Bridge supports PDF invoices, Excel
   spreadsheets, and ERP connections.

3. **Map your existing pledge** -- Use the Pledge Commitment Engine to
   validate your existing Race to Zero pledge against the 8 eligibility
   criteria. The engine will identify any gaps.

4. **Run Starting Line assessment** -- Use the Starting Line Assessment
   Workflow to evaluate your current compliance with the 20 sub-criteria.
   This will produce a gap report with remediation actions.

5. **Import your action plan** -- Use the Action Plan Engine to validate
   your existing climate action plan or generate a new one with MACC
   analysis.

6. **Configure reporting channels** -- Set up partner-specific reporting
   outputs (CDP, GFANZ, C40, ICLEI) through the Campaign Reporting Engine.

7. **Run credibility assessment** -- Use the Credibility Review Workflow
   to evaluate your pledge against HLEG "Integrity Matters" recommendations.

8. **Generate readiness score** -- Run the Race Readiness Engine to get
   your composite readiness score and improvement priorities.

---

*Maintained by GreenLang Platform Team*
