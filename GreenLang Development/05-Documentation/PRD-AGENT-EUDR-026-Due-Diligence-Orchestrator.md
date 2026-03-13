# PRD: AGENT-EUDR-026 -- Due Diligence Orchestrator Agent

## Document Info

| Field | Value |
|-------|-------|
| **PRD ID** | PRD-AGENT-EUDR-026 |
| **Agent ID** | GL-EUDR-DDO-026 |
| **Component** | Due Diligence Orchestrator Agent |
| **Category** | EUDR Regulatory Agent -- Due Diligence Workflow Orchestration |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Approved Date** | Pending |
| **Author** | GL-ProductManager |
| **Date** | 2026-03-11 |
| **Regulation** | Regulation (EU) 2023/1115 -- EU Deforestation Regulation (EUDR), Articles 4, 8, 9, 10, 11, 12, 13; ISO 19011:2018 Auditing Management Systems |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |

---

## 1. Executive Summary

### 1.1 Problem Statement

The EU Deforestation Regulation (Regulation (EU) 2023/1115) Article 8 mandates that operators and traders establish and implement a due diligence system consisting of three mandatory, sequential phases: (1) information gathering (Article 9), (2) risk assessment (Article 10), and (3) risk mitigation (Article 11). Article 4 requires the submission of a Due Diligence Statement (DDS) to the EU Information System before placing products on the EU market. Articles 12 and 13 define the content and procedural requirements for these statements. The enforcement date for large operators was December 30, 2025, and SME enforcement follows on June 30, 2026.

The GreenLang platform has built a comprehensive suite of 25 specialized EUDR agents spanning the full due diligence lifecycle: Supply Chain Traceability agents (EUDR-001 through EUDR-015) handle information gathering -- supply chain mapping, geolocation verification, satellite monitoring, forest cover analysis, land use change detection, plot boundary management, GPS coordinate validation, multi-tier supplier tracking, chain of custody, segregation verification, mass balance calculation, document authentication, blockchain integration, QR code generation, and mobile data collection. Risk Assessment agents (EUDR-016 through EUDR-025) handle multi-dimensional risk scoring -- country risk evaluation, supplier risk scoring, commodity risk analysis, corruption index monitoring, deforestation alert systems, indigenous rights checking, protected area validation, legal compliance verification, third-party audit management, and risk mitigation advisory.

However, these 25 agents currently operate as independent, loosely coupled services. There is no central orchestration engine that coordinates their execution in the correct sequence, manages dependencies between them, enforces quality gates between phases, handles partial failures with intelligent retry, or compiles their outputs into a unified, audit-ready due diligence package. Today, operators face the following critical gaps:

- **No end-to-end workflow orchestration**: Each of the 25 agents must be invoked manually or through ad-hoc scripting. There is no DAG-based workflow engine that understands the dependency topology between agents, resolves execution order, and drives the complete due diligence process from raw data intake to DDS-ready output. Compliance officers must manually coordinate the sequence: run supply chain mapping first, then geolocation verification, then satellite monitoring in parallel with forest cover analysis, then risk assessment only after all information gathering is complete, then mitigation only after risk assessment is complete. This manual coordination is error-prone, slow, and does not scale.

- **No workflow state management**: When a due diligence workflow is interrupted -- by agent failure, data unavailability, external dependency timeout, or user-initiated pause -- there is no mechanism to checkpoint the workflow state, persist intermediate results, and resume from the last successful step. Operators must restart the entire process from scratch, wasting hours of computation and duplicating API calls to external services (satellite data providers, certification databases, government registries).

- **No quality gates between phases**: EUDR Article 8 requires that each phase of due diligence be adequate before proceeding to the next. Information gathering must be sufficiently complete before risk assessment begins; risk assessment must identify all material risks before mitigation is designed. There is no automated validation engine that enforces completeness, accuracy, and consistency thresholds between phases, preventing premature progression and ensuring regulatory defensibility.

- **No intelligent parallelization**: Many of the 25 agents can execute concurrently. For example, geolocation verification (EUDR-002), satellite monitoring (EUDR-003), forest cover analysis (EUDR-004), and land use change detection (EUDR-005) can all run in parallel once supply chain mapping (EUDR-001) provides plot data. Similarly, all 10 risk assessment agents (EUDR-016 through 025) can run concurrently once information gathering is complete. Without intelligent parallelization, workflows execute sequentially, taking 5-10x longer than necessary.

- **No error recovery and retry**: Individual agent failures are inevitable in production -- satellite data providers experience outages, external APIs rate-limit requests, database connections timeout under load, and upstream data quality issues cause validation failures. There is no circuit breaker, exponential backoff, or fallback strategy to handle these failures gracefully. A single agent failure currently halts the entire due diligence process with no recovery path.

- **No unified due diligence package**: The 25 agents produce outputs in their own formats -- supply chain graphs, geolocation verification reports, satellite analysis results, risk scores, mitigation plans, audit findings. There is no aggregation engine that compiles these disparate outputs into a single, coherent, audit-ready due diligence evidence package that satisfies Articles 12 and 13 requirements for DDS content and that competent authorities can inspect under Articles 14-16.

- **No real-time progress tracking**: Compliance officers initiating a due diligence workflow have no visibility into which agents have completed, which are currently executing, which are blocked on dependencies, and what the estimated time to completion is. Without progress tracking, they cannot plan their work, escalate delays, or report status to management.

- **No simplified due diligence support**: Article 13 provides for simplified due diligence when commodities originate from countries classified as low-risk under Article 29. Simplified workflows require fewer agents and reduced evidence thresholds. There is no workflow variant engine that adjusts the agent topology, quality gate thresholds, and evidence requirements based on the applicable due diligence level.

Without solving these problems, operators cannot efficiently execute the complete EUDR due diligence obligation across their commodity portfolios. Manual orchestration of 25 agents is operationally unsustainable for large operators processing hundreds of products across seven commodity categories. The risk of incomplete due diligence, missed quality gates, and inadequate documentation exposes operators to penalties of up to 4% of annual EU turnover, goods confiscation, market exclusion, and public naming under Articles 23-25.

### 1.2 Solution Overview

Agent-EUDR-026: Due Diligence Orchestrator is the central workflow orchestration engine for end-to-end EUDR due diligence. It coordinates the execution of all 25 upstream agents (Supply Chain Traceability EUDR-001 through EUDR-015, Risk Assessment EUDR-016 through EUDR-025) in a DAG-based topology, manages workflow state with checkpointing and resume, enforces quality gates between the three mandatory due diligence phases, optimizes execution through intelligent parallelization, handles errors with exponential backoff and circuit breakers, and produces audit-ready due diligence packages for DDS submission. It is the 26th agent in the EUDR agent family and establishes the Due Diligence Workflow Orchestration sub-category.

Core capabilities:

1. **Workflow Definition Engine** -- DAG-based orchestration with configurable 25-agent topology, dependency resolution using topological sorting, support for standard and simplified due diligence workflow variants, commodity-specific workflow templates for all 7 EUDR commodities, and runtime workflow modification for dynamic agent addition or removal.

2. **Information Gathering Coordinator** -- Orchestrates EUDR-001 through EUDR-015 in the correct dependency order for comprehensive supply chain data collection. Manages parallel execution of independent agents, coordinates data handoff between dependent agents, and validates information gathering completeness against Article 9 requirements before allowing phase transition to risk assessment.

3. **Risk Assessment Coordinator** -- Orchestrates EUDR-016 through EUDR-025 for multi-dimensional risk scoring. Launches all 10 risk assessment agents in parallel once information gathering passes quality gate. Aggregates risk scores into a unified risk profile. Validates risk assessment completeness against Article 10 requirements before allowing phase transition to risk mitigation.

4. **Risk Mitigation Coordinator** -- Integrates EUDR-025 Risk Mitigation Advisor outputs. Validates that mitigation measures are adequate and proportionate per Article 11. Verifies that residual risk has been reduced to negligible levels. Coordinates the mitigation evidence compilation for the due diligence package.

5. **Quality Gate Engine** -- Validates completeness, accuracy, and consistency across all three due diligence phases. Enforces configurable thresholds: information gathering completeness (default >= 90%), risk assessment coverage (default >= 95%), mitigation adequacy (default residual risk <= 15). Provides detailed gap reports when quality gates fail with actionable remediation guidance.

6. **Workflow State Manager** -- Persistent checkpointing after every agent completion. Workflow resume from any checkpoint without re-executing completed steps. Rollback to previous checkpoint on critical failure. Complete audit trail of every state transition with timestamps, actor, and provenance hashes. Support for workflow pause, cancel, and clone operations.

7. **Parallel Execution Engine** -- Analyzes the DAG topology to identify independent agent groups that can execute concurrently. Manages configurable concurrency limits (default 10 agents per workflow, max 25). Implements work-stealing for load balancing across agent pools. Provides execution timeline visualization showing critical path and parallelization efficiency.

8. **Error Recovery & Retry Manager** -- Exponential backoff with configurable base delay (1s), max delay (300s), and max retries (5). Circuit breaker pattern with configurable failure threshold (5 failures), reset timeout (60s), and half-open probe. Fallback strategies: cached results, degraded mode (skip non-critical agents), manual override. Dead letter queue for permanently failed agent invocations.

9. **Due Diligence Package Generator** -- Compiles all 25 agent outputs into a single, structured, audit-ready evidence bundle. Generates DDS-compatible JSON for EU Information System submission. Produces human-readable PDF report with executive summary, detailed findings, evidence annexes, and provenance chain. Includes SHA-256 integrity hashes on every evidence artifact. Supports multi-language output (EN, FR, DE, ES, PT).

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| End-to-end workflow completion time (standard) | < 5 minutes for 1,000-shipment portfolio | Time from workflow start to package generation |
| End-to-end workflow completion time (simplified) | < 2 minutes for low-risk commodity | Time from workflow start to package generation |
| Concurrent workflow capacity | 1,000+ simultaneous workflows | Load test with sustained concurrent workflows |
| Workflow resume success rate | >= 99% successful resume from checkpoint | Resume attempts vs. successful completions |
| Quality gate accuracy | 100% of passed workflows meet regulatory requirements | Audit sample validation |
| Parallelization efficiency | >= 70% CPU utilization during workflow execution | Execution timeline analysis |
| Error recovery success rate | >= 95% of transient failures recovered automatically | Retry success vs. total retry attempts |
| Due diligence package completeness | 100% of required DDS fields populated | Schema validation against EU Information System spec |
| Agent coordination overhead | < 5% of total workflow time spent on orchestration | Orchestration time vs. agent execution time |
| Audit trail completeness | 100% of state transitions recorded with provenance | Audit trail gap analysis |
| All 7 commodities supported | 7/7 commodity-specific workflow templates | Commodity coverage matrix |
| Determinism | 100% reproducible workflow execution (same inputs produce same outputs) | Bit-perfect reproducibility tests |
| EUDR regulatory coverage | Full coverage of Articles 4, 8, 9, 10, 11, 12, 13 | Regulatory compliance matrix |

---

## 2. Market and Competitive Analysis

### 2.1 Market Opportunity

- **TAM (Total Addressable Market)**: 400,000+ operators and traders affected by EUDR across the EU, with an estimated compliance workflow automation market of 4-7 billion EUR as end-to-end due diligence systems become mandatory operational infrastructure.
- **SAM (Serviceable Addressable Market)**: 100,000+ EU importers of the 7 regulated commodities requiring automated due diligence workflow orchestration, estimated at 1-2B EUR.
- **SOM (Serviceable Obtainable Market)**: Target 500+ enterprise customers in Year 1 using GreenLang's full due diligence orchestration, representing 60-100M EUR in orchestration module ARR.

### 2.2 Target Customers

**Primary:**
- Large EU importers (> 10,000 shipments/year) managing EUDR due diligence across multiple commodity categories simultaneously
- Multinational food and beverage companies with cocoa, coffee, palm oil, and soya supply chains requiring full Article 8 due diligence
- Timber and paper industry operators with complex multi-tier supply chains and high documentation requirements
- Automotive and tire manufacturers (rubber) with global supply chain orchestration needs
- Meat and leather importers (cattle) with multi-country sourcing requiring coordinated due diligence

**Secondary:**
- Compliance consulting firms delivering EUDR due diligence-as-a-service to multiple operator clients
- Certification bodies (FSC, RSPO, PEFC, Rainforest Alliance) requiring evidence of systematic due diligence processes
- Customs brokers and freight forwarders coordinating EUDR documentation for importer clients
- SME importers (1,000-10,000 shipments/year) -- enforcement from June 30, 2026
- Financial institutions requiring portfolio-level EUDR due diligence evidence for ESG reporting

### 2.3 Competitive Landscape

| Competitor Category | Strengths | Weaknesses | GreenLang Advantage |
|---------------------|-----------|------------|---------------------|
| Manual / Spreadsheet orchestration | No technology cost; familiar | Cannot manage 25 agents; no parallelization; no checkpointing; 10x slower; error-prone | DAG-based automation; 5x faster; zero manual coordination |
| Generic workflow tools (Apache Airflow, Prefect, Temporal) | Mature DAG engines; community support | Not EUDR-specific; no quality gates for Article 8/9/10/11; no DDS integration; require custom development | Purpose-built for EUDR due diligence; regulatory quality gates; native DDS generation |
| Generic GRC platforms (SAP GRC, MetricStream, OneTrust) | Enterprise risk management; audit workflows | No EUDR agent ecosystem; no satellite/GIS integration; no commodity-specific workflows | 25-agent native integration; commodity-specific templates; full geospatial capability |
| Niche EUDR platforms (Preferred by Nature, Ecosphere+) | Commodity domain expertise | Limited to single commodity; no orchestration engine; manual risk-to-mitigation handoff | All 7 commodities; automated end-to-end orchestration; intelligent parallelization |
| In-house custom orchestration | Tailored to org; full control | 12-18 month build; no regulatory updates; no quality gate framework; maintenance burden | Ready now; continuous regulatory updates; production-grade resilience |

### 2.4 Differentiation Strategy

1. **25-agent native orchestration** -- The only platform that orchestrates all 25 purpose-built EUDR agents in a single DAG-based workflow, eliminating manual coordination entirely.
2. **Regulatory quality gates** -- Article-specific quality gates (Art. 9 for information gathering, Art. 10 for risk assessment, Art. 11 for mitigation) enforced automatically between phases, providing regulatory defensibility.
3. **Commodity-specific workflow templates** -- Pre-configured DAG topologies for all 7 EUDR commodities, accounting for commodity-specific supply chain archetypes and risk profiles.
4. **Checkpoint and resume** -- Persistent workflow state with sub-second resume from any checkpoint, eliminating wasted computation on failure and enabling multi-day workflows for complex supply chains.
5. **Intelligent parallelization** -- DAG-aware concurrent execution achieving 3-5x speedup over sequential orchestration, with configurable concurrency limits for resource management.
6. **Audit-ready package generation** -- Single-click compilation of all 25 agent outputs into DDS-compatible evidence bundles with SHA-256 provenance chain and multi-language support.

---

## 3. Regulatory and Legal Requirements

### 3.1 EUDR Articles Addressed

| Article | Requirement | Agent Implementation |
|---------|-------------|---------------------|
| **Art. 4(1)** | Operators shall not place or make available relevant products on the Union market unless they are deforestation-free, produced in accordance with relevant legislation, and covered by a due diligence statement | Orchestrator ensures complete due diligence workflow execution before DDS generation; quality gates verify all three conditions |
| **Art. 4(2)** | Operators shall exercise due diligence with regard to all relevant products | Workflow templates cover all 7 commodities and derived products; commodity classifier routes to correct workflow |
| **Art. 8(1)** | Operators shall establish and implement a due diligence system comprising information gathering, risk assessment, and risk mitigation | Three-phase workflow architecture maps directly to Art. 8 due diligence system: Phase 1 (EUDR-001-015), Phase 2 (EUDR-016-025), Phase 3 (EUDR-025 mitigation outputs) |
| **Art. 8(2)** | Operators shall make available to competent authorities, upon request, evidence of due diligence system | Due Diligence Package Generator produces inspection-ready evidence bundles; audit trail provides complete workflow provenance |
| **Art. 8(3)** | Due diligence system shall be reviewed at least once a year and updated when relevant | Workflow versioning and annual review automation; template updates on regulatory changes |
| **Art. 9(1)(a-d)** | Information to be collected: description, quantity, geolocation of plots, country, date/period of production, supplier information | Information Gathering Coordinator validates all Art. 9 data fields via quality gate before phase transition |
| **Art. 9(2)** | Operators shall verify and validate the information collected | Quality Gate Engine validates information completeness, consistency, and accuracy using cross-agent validation |
| **Art. 10(1)** | Operators shall assess and identify risk that relevant commodities are non-compliant | Risk Assessment Coordinator orchestrates all 10 risk agents (EUDR-016-025) and validates assessment completeness |
| **Art. 10(2)(a-f)** | Risk factors: supply chain complexity, circumvention risk, country risk, production risk, supplier concerns, complementary information | Each risk factor mapped to specific risk assessment agents; quality gate verifies all factors assessed |
| **Art. 11(1)** | Where risk assessment identifies non-negligible risk, adopt adequate and proportionate risk mitigation measures | Risk Mitigation Coordinator validates mitigation adequacy; quality gate verifies residual risk reduced to negligible |
| **Art. 11(2)(a-c)** | Mitigation measures: additional info from suppliers, independent audits, other measures | Mitigation phase orchestrates EUDR-025 outputs covering all three measure categories |
| **Art. 12(1)** | Prior to placing on market, operator shall submit a DDS to the Information System | Due Diligence Package Generator produces DDS-compatible output for EU Information System submission |
| **Art. 12(2)(a-j)** | DDS content requirements: operator info, products, quantities, countries, geolocation, risk assessment results, mitigation measures | Package Generator maps all 25 agent outputs to Art. 12(2) DDS content fields; schema validation against EU spec |
| **Art. 13** | Simplified due diligence for low-risk countries | Workflow Definition Engine supports simplified variant with reduced agent topology and relaxed quality gate thresholds |
| **Art. 31(1)** | Record keeping for 5 years | Workflow State Manager retains all checkpoints, audit trails, and evidence for minimum 5 years with immutable storage |

### 3.2 ISO 19011:2018 Auditing Management Systems Alignment

The orchestrator's audit trail and quality gate framework aligns with ISO 19011:2018 principles for auditing management systems, ensuring that the due diligence workflow itself is auditable.

| ISO 19011 Clause | Requirement | Agent Implementation |
|-------------------|-------------|---------------------|
| **4 Principles of auditing** | Integrity, fair presentation, due professional care, confidentiality, independence, evidence-based approach | Quality gates enforce evidence-based progression; audit trail provides fair, complete record; RBAC ensures data confidentiality |
| **5.4.2 Audit plan** | Establish a plan that includes objectives, scope, criteria, methods, schedule | Workflow Definition Engine defines the due diligence plan with configurable scope, quality criteria, and schedule |
| **6.3.2 Preparing audit activities** | Prepare document review, sampling plan, audit procedures | Information Gathering Coordinator prepares evidence collection plan; quality gate criteria define acceptance thresholds |
| **6.4.2 Generating audit findings** | Evaluate audit evidence against criteria to generate findings | Quality Gate Engine evaluates agent outputs against regulatory criteria to generate pass/fail findings with evidence |
| **6.4.4 Preparing audit conclusions** | Review findings and determine audit conclusions | Due Diligence Package Generator synthesizes all findings into conclusions with confidence levels and recommendations |
| **6.5 Preparing and distributing audit report** | Prepare a complete, accurate, concise, clear report | Package Generator produces structured reports in multiple formats (PDF, JSON, HTML) with executive summary and detailed evidence |
| **6.6 Completing the audit** | Retain and protect audit records | Workflow State Manager ensures immutable retention of all audit records for 5 years per EUDR Article 31 |

### 3.3 Additional Regulatory Frameworks

| Framework | Relevance | Agent Support |
|-----------|-----------|---------------|
| EU Corporate Sustainability Due Diligence Directive (CSDDD) | Supply chain due diligence orchestration patterns reusable for CSDDD | Modular workflow templates adaptable to CSDDD due diligence requirements |
| ISO 14001:2015 Environmental Management | Systematic environmental management processes | Workflow quality gates align with Plan-Do-Check-Act cycle |
| ISO 31000:2018 Risk Management | Structured risk management process | Three-phase workflow maps to ISO 31000 risk identification, analysis, treatment |
| FSC Chain of Custody (FSC-STD-40-004) | Forestry chain of custody due diligence | Wood commodity workflow template includes FSC-specific evidence requirements |
| RSPO Supply Chain Certification Standard | Palm oil due diligence requirements | Palm oil workflow template includes RSPO-specific verification steps |
| UN Guiding Principles on Business and Human Rights | Human rights due diligence process | Indigenous rights (EUDR-021) and legal compliance (EUDR-023) integrated into standard workflow |

### 3.4 Key Regulatory Dates

| Date | Milestone | Agent Impact |
|------|-----------|-------------|
| December 31, 2020 | EUDR deforestation cutoff date | Baseline date for all deforestation-free verification in information gathering phase |
| June 29, 2023 | Regulation entered into force | Legal basis for all due diligence workflow requirements |
| December 30, 2025 | Enforcement for large operators (ACTIVE) | Full due diligence orchestration must be operational for large operators |
| June 30, 2026 | Enforcement for SMEs | Simplified workflow variant must be available; scale for SME onboarding wave |
| Ongoing (quarterly) | Country benchmarking updates by EC | Workflow templates auto-updated; quality gate thresholds adjusted per new risk classifications |
| Ongoing (annually) | Article 8(3) due diligence system review | Annual workflow review automation; template versioning and effectiveness assessment |

---

## 4. User Stories and Personas

### 4.1 Persona 1: Compliance Officer -- Maria (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Head of Regulatory Compliance at a large EU chocolate manufacturer |
| **Company** | 5,000 employees, sourcing cocoa from 12 countries via 300+ suppliers |
| **EUDR Pressure** | Must execute complete due diligence for 50+ product lines across cocoa, soya, and palm oil commodities |
| **Pain Points** | Manually coordinating 25 agents is taking 2-3 days per product line; no visibility into workflow progress; agent failures require restarting from scratch; cannot demonstrate systematic due diligence process to auditors |
| **Goals** | One-click due diligence execution; real-time progress tracking; automatic error recovery; audit-ready evidence packages for DDS submission |
| **Technical Skill** | Moderate -- comfortable with web applications and dashboards but not a developer |

### 4.2 Persona 2: Supply Chain Analyst -- Lukas (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Senior Supply Chain Analyst at an EU timber importer |
| **Company** | 800 employees, importing tropical and temperate wood from 20+ countries |
| **EUDR Pressure** | Must manage due diligence workflows for 200+ wood product variants with complex multi-tier supply chains |
| **Pain Points** | Cannot parallelize agent execution manually; quality inconsistencies between due diligence runs; no checkpoint capability means 4-hour workflows lost on failure; simplified due diligence for low-risk Nordic sourcing still runs full workflow |
| **Goals** | Automated parallel execution; consistent quality gates; checkpoint and resume for long workflows; simplified workflow for low-risk origins |
| **Technical Skill** | High -- comfortable with data tools, APIs, and workflow configuration |

### 4.3 Persona 3: Compliance Operations Manager -- Stefan (Primary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Compliance Operations Manager at a large EU palm oil refinery |
| **Company** | 3,000 employees, processing palm oil from 200+ plantations across Indonesia and Malaysia |
| **EUDR Pressure** | Must orchestrate due diligence for 1,000+ shipments per quarter with tight submission deadlines |
| **Pain Points** | No batch workflow capability; cannot run due diligence across entire shipment portfolio; no ETA estimates for workflow completion; resource contention when multiple workflows run simultaneously |
| **Goals** | Batch workflow execution for entire quarterly portfolio; accurate ETA estimates; resource management for concurrent workflows; dashboard showing fleet-level progress |
| **Technical Skill** | Moderate-high -- uses operational dashboards and monitors SLAs |

### 4.4 Persona 4: External Auditor -- Dr. Hofmann (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | Lead Auditor at an EU-accredited EUDR verification body |
| **Company** | Third-party audit firm |
| **EUDR Pressure** | Must verify that operators have executed systematic, complete due diligence per Article 8 |
| **Pain Points** | Operators provide fragmented evidence from individual agents; no unified audit trail showing the complete due diligence process; cannot verify that quality gates were properly enforced; no way to confirm workflow was not tampered with |
| **Goals** | Access complete workflow execution audit trail; verify quality gate enforcement; validate evidence package integrity via provenance hashes; confirm all 25 agents executed for each product |
| **Technical Skill** | Moderate -- comfortable with audit software and structured documentation |

### 4.5 Persona 5: IT Operations Engineer -- Katja (Secondary)

| Attribute | Detail |
|-----------|--------|
| **Role** | DevOps Engineer responsible for GreenLang platform operations |
| **Company** | GreenLang infrastructure team |
| **EUDR Pressure** | Must ensure the orchestrator handles 1,000+ concurrent workflows with high availability |
| **Pain Points** | No visibility into workflow execution metrics; cannot tune concurrency limits; error recovery behavior is unpredictable; no circuit breaker to prevent cascade failures when upstream agents are down |
| **Goals** | Prometheus metrics for workflow execution; configurable concurrency and circuit breaker parameters; Grafana dashboards for operational monitoring; alerting on workflow SLA breaches |
| **Technical Skill** | High -- manages Kubernetes, monitoring, and infrastructure |

### 4.6 User Stories

**US-001: One-Click Due Diligence Execution**
```
As a compliance officer,
I want to initiate a complete EUDR due diligence workflow with a single action,
So that all 25 agents execute in the correct order without manual coordination.
```

**US-002: Real-Time Progress Tracking**
```
As a compliance officer,
I want to see real-time progress of my due diligence workflow with agent-level status and ETA,
So that I can plan my work and report status to management.
```

**US-003: Workflow Resume After Failure**
```
As a supply chain analyst,
I want to resume a failed workflow from the last successful checkpoint,
So that I do not lose hours of completed work when a single agent fails.
```

**US-004: Parallel Agent Execution**
```
As a supply chain analyst,
I want independent agents to execute in parallel automatically,
So that my due diligence workflow completes in minutes instead of hours.
```

**US-005: Quality Gate Enforcement**
```
As a compliance officer,
I want the system to validate completeness before transitioning between due diligence phases,
So that my risk assessment is not based on incomplete information and my DDS is defensible.
```

**US-006: Simplified Due Diligence for Low-Risk Origins**
```
As a compliance officer,
I want a simplified workflow variant for commodities from low-risk countries,
So that I spend appropriate effort proportional to the risk level per Article 13.
```

**US-007: Batch Workflow Execution**
```
As a compliance operations manager,
I want to launch due diligence workflows for my entire quarterly shipment portfolio at once,
So that I can meet quarterly DDS submission deadlines without manual bottlenecks.
```

**US-008: Error Recovery Without Manual Intervention**
```
As a compliance operations manager,
I want transient failures (timeouts, rate limits, temporary outages) to be retried automatically,
So that my workflows complete without requiring my intervention for routine infrastructure issues.
```

**US-009: Audit-Ready Evidence Package**
```
As an external auditor,
I want to receive a single, structured evidence package with provenance hashes,
So that I can efficiently verify the completeness and integrity of the operator's due diligence.
```

**US-010: Commodity-Specific Workflow Templates**
```
As a compliance officer,
I want pre-configured workflow templates for each of the 7 EUDR commodities,
So that the workflow includes the correct agents and quality criteria for my specific commodity.
```

**US-011: Concurrent Workflow Management**
```
As an IT operations engineer,
I want configurable concurrency limits and circuit breakers for the orchestrator,
So that the platform remains stable under high workflow load without cascade failures.
```

**US-012: Workflow Audit Trail for Regulatory Inspection**
```
As a compliance officer,
I want an immutable audit trail of every workflow execution with timestamps and provenance,
So that I can demonstrate systematic due diligence to competent authorities under Article 8(2).
```

---

## 5. Feature Requirements

### 5.1 Must-Have Features (P0 -- Launch Blockers)

All 9 features below are P0 launch blockers. The agent cannot ship without all 9 features operational. Features 1-4 form the core orchestration engine; Features 5-6 form the quality and state management layer; Features 7-8 form the execution optimization and resilience layer; Feature 9 is the output delivery layer.

**P0 Features 1-4: Core Orchestration Engine**

---

#### Feature 1: Workflow Definition Engine

**User Story:**
```
As a compliance officer,
I want to define and configure due diligence workflows as directed acyclic graphs (DAGs),
So that the system understands agent dependencies and executes them in the correct order.
```

**Acceptance Criteria:**
- [ ] Defines workflows as DAGs with 25 agent nodes and configurable dependency edges
- [ ] Resolves execution order using topological sorting (Kahn's algorithm)
- [ ] Detects circular dependencies and rejects invalid workflow definitions
- [ ] Provides pre-built workflow templates for all 7 EUDR commodities (cattle, cocoa, coffee, palm oil, rubber, soya, wood)
- [ ] Supports standard due diligence workflow (full 25-agent topology)
- [ ] Supports simplified due diligence workflow (reduced topology per Article 13)
- [ ] Supports runtime workflow modification: add/remove agents, modify dependencies
- [ ] Validates workflow definitions against a schema before execution
- [ ] Supports workflow versioning with immutable version history
- [ ] Supports workflow cloning for template customization

**DAG Topology -- Standard Due Diligence Workflow:**

```
Phase 1: Information Gathering (Article 9)
==========================================

Layer 0 (Entry):
  EUDR-001 Supply Chain Mapping Master
    |
    +---> Layer 1 (Parallel - Geospatial Verification):
    |       EUDR-002 Geolocation Verification
    |       EUDR-006 Plot Boundary Manager
    |       EUDR-007 GPS Coordinate Validator
    |       EUDR-008 Multi-Tier Supplier Tracker
    |
    +---> Layer 2 (Parallel - Satellite & Land, depends on Layer 1):
    |       EUDR-003 Satellite Monitoring
    |       EUDR-004 Forest Cover Analysis
    |       EUDR-005 Land Use Change Detector
    |
    +---> Layer 3 (Parallel - Chain of Custody, depends on Layer 1):
    |       EUDR-009 Chain of Custody
    |       EUDR-010 Segregation Verifier
    |       EUDR-011 Mass Balance Calculator
    |
    +---> Layer 4 (Parallel - Evidence & Traceability, depends on Layers 2-3):
    |       EUDR-012 Document Authentication
    |       EUDR-013 Blockchain Integration
    |       EUDR-014 QR Code Generator
    |       EUDR-015 Mobile Data Collector
    |
    +---> QUALITY GATE 1: Information Gathering Completeness
            (>= 90% data completeness, all Art. 9 fields validated)

Phase 2: Risk Assessment (Article 10)
======================================

Layer 5 (Parallel - All Risk Agents, depends on Quality Gate 1):
  EUDR-016 Country Risk Evaluator
  EUDR-017 Supplier Risk Scorer
  EUDR-018 Commodity Risk Analyzer
  EUDR-019 Corruption Index Monitor
  EUDR-020 Deforestation Alert System
  EUDR-021 Indigenous Rights Checker
  EUDR-022 Protected Area Validator
  EUDR-023 Legal Compliance Verifier
  EUDR-024 Third-Party Audit Manager
  EUDR-025 Risk Mitigation Advisor (assessment inputs)
    |
    +---> QUALITY GATE 2: Risk Assessment Completeness
            (>= 95% risk dimension coverage, all Art. 10 factors assessed)

Phase 3: Risk Mitigation (Article 11)
======================================

Layer 6 (Depends on Quality Gate 2):
  EUDR-025 Risk Mitigation Advisor (mitigation outputs)
    |
    +---> QUALITY GATE 3: Mitigation Adequacy
            (residual risk <= 15, mitigation proportionality verified)

Phase 4: Package Generation
============================

Layer 7 (Depends on Quality Gate 3):
  Due Diligence Package Generator
    |
    +---> OUTPUT: Audit-Ready Due Diligence Package + DDS-Compatible JSON
```

**Simplified Due Diligence Workflow (Article 13):**

```
Layer 0: EUDR-001 Supply Chain Mapping Master
Layer 1: EUDR-002 Geolocation Verification + EUDR-007 GPS Validator
Layer 2: EUDR-003 Satellite Monitoring (reduced scope)
QUALITY GATE 1 (relaxed: >= 80% completeness)
Layer 3: EUDR-016 Country Risk + EUDR-018 Commodity Risk + EUDR-023 Legal Compliance
QUALITY GATE 2 (relaxed: >= 85% coverage)
Layer 4: Package Generator (simplified format)
OUTPUT: Simplified DDS Package
```

**Non-Functional Requirements:**
- Performance: Workflow definition parsing and validation < 100ms
- Scalability: Support up to 50 agent nodes per workflow (future-proofing beyond 25)
- Determinism: Same workflow definition always produces same execution plan

**Dependencies:**
- NetworkX for DAG operations and topological sorting
- Workflow definition schema (JSON/YAML)

**Estimated Effort:** 3 weeks (1 senior backend engineer)

---

#### Feature 2: Information Gathering Coordinator

**User Story:**
```
As a compliance officer,
I want the system to orchestrate all 15 supply chain traceability agents in the correct order,
So that comprehensive information is gathered per EUDR Article 9 before risk assessment begins.
```

**Acceptance Criteria:**
- [ ] Invokes EUDR-001 Supply Chain Mapping Master as the entry point for all workflows
- [ ] Launches EUDR-002, EUDR-006, EUDR-007, EUDR-008 in parallel after EUDR-001 completes
- [ ] Launches EUDR-003, EUDR-004, EUDR-005 after Layer 1 geospatial agents complete (depends on plot data)
- [ ] Launches EUDR-009, EUDR-010, EUDR-011 after Layer 1 completes (depends on supplier/custody data)
- [ ] Launches EUDR-012, EUDR-013, EUDR-014, EUDR-015 after Layers 2-3 complete
- [ ] Manages data handoff between dependent agents (e.g., EUDR-001 plot data feeds EUDR-002, EUDR-003)
- [ ] Tracks individual agent status: PENDING, RUNNING, COMPLETED, FAILED, SKIPPED
- [ ] Calculates information gathering completeness score (0-100) in real time
- [ ] Validates all Article 9 required fields before triggering Quality Gate 1
- [ ] Supports partial execution: skip non-applicable agents for specific commodities

**Article 9 Field Validation Matrix:**

| Art. 9 Field | Source Agent(s) | Validation Rule |
|-------------|-----------------|-----------------|
| Product description | EUDR-001 | Non-empty, matches CN/HS code |
| Quantity | EUDR-001 | Positive numeric, valid unit |
| Country of production | EUDR-001, EUDR-016 | Valid ISO 3166-1 alpha-2 |
| Geolocation of plots | EUDR-002, EUDR-006, EUDR-007 | Valid WGS84 coordinates; polygon for > 4 ha |
| Date/period of production | EUDR-001, EUDR-009 | Valid date, not future, after EUDR cutoff context |
| Supplier information | EUDR-001, EUDR-008, EUDR-009 | Complete chain of custody from producer to importer |
| Deforestation-free evidence | EUDR-003, EUDR-004, EUDR-005 | Satellite verification against Dec 31, 2020 cutoff |
| Legal compliance evidence | EUDR-012, EUDR-023 | Authenticated documents, valid permits/licenses |

**Non-Functional Requirements:**
- Performance: Phase 1 completion < 3 minutes for 1,000-shipment portfolio with full parallelization
- Reliability: Agent invocations are idempotent; safe to retry without side effects
- Observability: Real-time progress events emitted for each agent state transition

**Dependencies:**
- EUDR-001 through EUDR-015 (all built, production-ready)
- Workflow State Manager (Feature 6) for checkpointing
- Parallel Execution Engine (Feature 7) for concurrent agent invocation

**Estimated Effort:** 3 weeks (1 senior backend engineer, 1 integration engineer)

---

#### Feature 3: Risk Assessment Coordinator

**User Story:**
```
As a compliance officer,
I want all 10 risk assessment agents to execute in parallel once information gathering is complete,
So that a comprehensive, multi-dimensional risk profile is generated per EUDR Article 10.
```

**Acceptance Criteria:**
- [ ] Blocks until Quality Gate 1 (information gathering completeness) passes
- [ ] Launches all 10 risk assessment agents (EUDR-016 through EUDR-025) in parallel
- [ ] Passes information gathering outputs as input context to each risk agent
- [ ] Aggregates individual risk scores into a unified risk profile
- [ ] Calculates composite risk score using deterministic weighted formula
- [ ] Maps each risk agent output to specific Article 10(2) risk factors
- [ ] Validates risk assessment completeness: all 10 risk dimensions scored
- [ ] Generates risk assessment summary with highest-risk findings highlighted
- [ ] Triggers Quality Gate 2 upon all risk agents completing
- [ ] Supports degraded mode: proceed with available risk scores if non-critical agents fail (with warning)

**Article 10(2) Risk Factor to Agent Mapping:**

| Art. 10(2) Factor | Risk Agent(s) | Risk Dimension |
|-------------------|---------------|----------------|
| (a) Complexity of supply chain | EUDR-001, EUDR-008 | Supply chain depth, node count, opaque segments |
| (b) Risk of circumvention or mixing | EUDR-010, EUDR-011, EUDR-018 | Segregation integrity, mass balance accuracy, commodity mixing risk |
| (c) Risk of non-compliance of country | EUDR-016, EUDR-019 | Country risk classification, corruption perception index |
| (d) Risk linked to country of production | EUDR-016, EUDR-020, EUDR-021, EUDR-022 | Deforestation rate, indigenous rights, protected area proximity |
| (e) Concerns about the supplier | EUDR-017, EUDR-024 | Supplier compliance history, audit findings, certification status |
| (f) Substantiated concerns / complementary info | EUDR-020, EUDR-021, EUDR-023, EUDR-025 | Deforestation alerts, rights violations, legal gaps, mitigation readiness |

**Composite Risk Score Formula:**
```
Composite_Risk = (
    W_country    * EUDR_016_score +
    W_supplier   * EUDR_017_score +
    W_commodity  * EUDR_018_score +
    W_corruption * EUDR_019_score +
    W_deforest   * EUDR_020_score +
    W_indigenous * EUDR_021_score +
    W_protected  * EUDR_022_score +
    W_legal      * EUDR_023_score +
    W_audit      * EUDR_024_score +
    W_mitigation * EUDR_025_score
)

Where default weights (configurable per operator):
  W_country = 0.15, W_supplier = 0.12, W_commodity = 0.10,
  W_corruption = 0.08, W_deforest = 0.15, W_indigenous = 0.10,
  W_protected = 0.10, W_legal = 0.10, W_audit = 0.05,
  W_mitigation = 0.05
  Sum = 1.00
```

**Non-Functional Requirements:**
- Performance: Phase 2 completion < 90 seconds with all 10 agents running in parallel
- Determinism: Composite risk score is bit-perfect reproducible
- Auditability: Every risk score contribution tracked with agent ID, version, timestamp, and provenance hash

**Dependencies:**
- EUDR-016 through EUDR-025 (all built, production-ready)
- Quality Gate Engine (Feature 5) for phase transition validation
- Parallel Execution Engine (Feature 7) for concurrent agent invocation

**Estimated Effort:** 3 weeks (1 senior backend engineer)

---

#### Feature 4: Risk Mitigation Coordinator

**User Story:**
```
As a compliance officer,
I want the system to coordinate risk mitigation activities and verify adequacy per Article 11,
So that identified risks are reduced to negligible levels before DDS submission.
```

**Acceptance Criteria:**
- [ ] Blocks until Quality Gate 2 (risk assessment completeness) passes
- [ ] Determines whether risk mitigation is required based on composite risk threshold (non-negligible risk)
- [ ] Invokes EUDR-025 Risk Mitigation Advisor with full risk assessment context
- [ ] Validates that mitigation strategies address all identified non-negligible risks
- [ ] Verifies mitigation adequacy: residual risk reduced to configurable threshold (default <= 15 on 0-100 scale)
- [ ] Validates mitigation proportionality: effort proportionate to risk level per Article 11(1)
- [ ] Collects mitigation evidence: remediation plans, capacity building enrollments, audit schedules, documentation
- [ ] Supports bypass for low-risk workflows where risk assessment shows negligible risk (skip to Package Generation)
- [ ] Triggers Quality Gate 3 upon mitigation completion and adequacy verification
- [ ] Generates mitigation summary for inclusion in due diligence package

**Risk-to-Mitigation Decision Logic:**
```
IF composite_risk_score <= NEGLIGIBLE_THRESHOLD (default: 20):
    SKIP mitigation phase -> proceed to Package Generation
    Document: "Risk assessment identified negligible risk; no mitigation required"

ELSE IF composite_risk_score <= STANDARD_THRESHOLD (default: 50):
    EXECUTE standard mitigation via EUDR-025
    REQUIRE: residual_risk <= 15 after mitigation

ELSE (composite_risk_score > 50):
    EXECUTE enhanced mitigation via EUDR-025
    REQUIRE: residual_risk <= 15 after mitigation
    REQUIRE: enhanced evidence documentation (independent audits, supplier site visits)
```

**Non-Functional Requirements:**
- Performance: Mitigation coordination < 60 seconds (excluding actual mitigation implementation time)
- Auditability: Complete record of mitigation decisions, bypass justifications, and adequacy determinations
- Determinism: Mitigation requirement determination is reproducible

**Dependencies:**
- EUDR-025 Risk Mitigation Advisor (built, production-ready)
- Quality Gate Engine (Feature 5) for phase transition validation
- Risk Assessment Coordinator (Feature 3) for risk inputs

**Estimated Effort:** 2 weeks (1 senior backend engineer)

---

**P0 Features 5-6: Quality and State Management Layer**

---

#### Feature 5: Quality Gate Engine

**User Story:**
```
As a compliance officer,
I want the system to automatically validate completeness and quality between due diligence phases,
So that my DDS is built on adequate information, thorough risk assessment, and sufficient mitigation.
```

**Acceptance Criteria:**
- [ ] Enforces Quality Gate 1 (Information Gathering): validates Art. 9 field completeness (>= 90%), cross-agent data consistency, geolocation coverage, custody chain integrity
- [ ] Enforces Quality Gate 2 (Risk Assessment): validates all 10 risk dimensions scored (>= 95% coverage), composite risk score calculated, all Art. 10(2) factors assessed
- [ ] Enforces Quality Gate 3 (Mitigation Adequacy): validates residual risk <= threshold, mitigation evidence documented, proportionality verified
- [ ] Provides configurable thresholds per quality gate (adjustable per operator, commodity, risk level)
- [ ] Generates detailed quality gate report: pass/fail with individual check results, scores, and evidence
- [ ] Identifies specific gaps causing quality gate failure with remediation guidance
- [ ] Supports quality gate override with mandatory justification (for exceptional cases, logged in audit trail)
- [ ] Supports quality gate relaxation for simplified due diligence (Article 13)
- [ ] Emits events on quality gate evaluation: PASSED, FAILED, OVERRIDDEN
- [ ] Maintains quality gate evaluation history for audit trail

**Quality Gate Specifications:**

| Gate | Phase Transition | Default Threshold | Simplified Threshold | Checks |
|------|-----------------|-------------------|---------------------|--------|
| QG-1 | Info Gathering -> Risk Assessment | >= 90% completeness | >= 80% completeness | Art. 9 fields, geolocation, custody chain, satellite verification |
| QG-2 | Risk Assessment -> Mitigation | >= 95% risk coverage | >= 85% risk coverage | All 10 risk dimensions, Art. 10(2) factors, composite risk score |
| QG-3 | Mitigation -> Package Generation | Residual risk <= 15 | Residual risk <= 25 | Mitigation evidence, adequacy, proportionality |

**Quality Gate 1 Detailed Checks:**

| Check | Weight | Pass Criteria | Remediation on Failure |
|-------|--------|---------------|----------------------|
| Product description completeness | 10% | All products have CN/HS code and description | Complete product classification via EUDR-001 |
| Quantity data completeness | 10% | All shipments have quantity and unit | Update shipment records |
| Country of production identified | 10% | All products linked to production country | Run supplier discovery (EUDR-008) |
| Plot geolocation coverage | 20% | >= threshold% of plots have valid GPS coordinates | Trigger geolocation collection (EUDR-002, EUDR-007) |
| Polygon coverage (> 4 ha plots) | 10% | All plots > 4 ha have polygon boundaries | Trigger boundary mapping (EUDR-006) |
| Custody chain integrity | 15% | Unbroken chain from producer to importer | Run chain of custody verification (EUDR-009) |
| Satellite verification | 15% | Deforestation-free status verified for all plots | Run satellite analysis (EUDR-003, EUDR-004, EUDR-005) |
| Document authentication | 10% | Key documents authenticated | Run authentication (EUDR-012) |

**Non-Functional Requirements:**
- Performance: Quality gate evaluation < 5 seconds for 10,000-shipment workflow
- Determinism: Same inputs always produce same gate result
- Auditability: Every gate evaluation stored with full evidence and provenance hash

**Dependencies:**
- All 25 upstream agents (outputs feed quality gate checks)
- Workflow State Manager (Feature 6) for persisting gate results

**Estimated Effort:** 3 weeks (1 senior backend engineer)

---

#### Feature 6: Workflow State Manager

**User Story:**
```
As a supply chain analyst,
I want workflow state to be persistently checkpointed after every agent completion,
So that I can resume long-running workflows from the last checkpoint without losing work.
```

**Acceptance Criteria:**
- [ ] Checkpoints workflow state after every agent completion (agent outputs, status, timestamps)
- [ ] Checkpoints after every quality gate evaluation (gate result, scores, evidence)
- [ ] Supports workflow resume from any checkpoint without re-executing completed agents
- [ ] Supports workflow rollback to a previous checkpoint on critical failure
- [ ] Supports workflow pause: save state and stop execution, resume later
- [ ] Supports workflow cancel: terminate execution, mark workflow as cancelled, retain state for audit
- [ ] Supports workflow clone: create new workflow from existing workflow state (useful for re-runs)
- [ ] Maintains immutable audit trail: every state transition logged with timestamp, actor, and reason
- [ ] Generates provenance hash (SHA-256) for each checkpoint covering all accumulated state
- [ ] Retains workflow state for minimum 5 years per EUDR Article 31
- [ ] Supports concurrent access: multiple users can view workflow state while execution continues
- [ ] Provides real-time progress tracking: current phase, active agents, completed agents, ETA estimate

**Workflow State Machine:**

```
                    +----------+
                    |  CREATED |
                    +----+-----+
                         |
                    start_workflow()
                         |
                    +----v-----+
           +------->  RUNNING  <--------+
           |        +----+-----+        |
           |             |              |
      resume()    +------+------+    retry()
           |      |      |      |      |
           |  complete  fail   pause   |
           |      |      |      |      |
           |  +---v--+ +-v----+ +v---+ |
           |  |PASSED| |FAILED| |PAUSED|
           |  +---+--+ +-+----+ +--+-+ |
           |      |       |         |   |
           |   (next    (retry     resume()
           |    agent)   or           |
           |      |    escalate)      |
           |      v       |           |
           |   RUNNING    +------->---+
           |      |
           |   (all agents done)
           |      |
           |  +---v-----------+
           |  | QUALITY_GATE  |
           |  +---+-------+---+
           |      |       |
           |    pass    fail
           |      |       |
           |  +---v--+ +--v----+
           |  |PASSED| |GATE_  |
           |  +---+--+ |FAILED |
           |      |     +---+---+
           |   (next        |
           |    phase)  (remediate & retry)
           |      |         |
           +------+---------+

Terminal States:
  COMPLETED  -- All phases passed, package generated
  CANCELLED  -- User cancelled workflow
  TERMINATED -- Unrecoverable failure after max retries
```

**Checkpoint Data Structure:**
```python
class WorkflowCheckpoint(BaseModel):
    checkpoint_id: str           # UUID
    workflow_id: str             # Parent workflow UUID
    sequence_number: int         # Monotonic sequence within workflow
    phase: DueDiligencePhase     # INFORMATION_GATHERING / RISK_ASSESSMENT / MITIGATION / PACKAGE_GENERATION
    agent_id: Optional[str]      # Agent that just completed (if agent checkpoint)
    gate_id: Optional[str]       # Quality gate evaluated (if gate checkpoint)
    agent_status: Dict[str, AgentExecutionStatus]  # Status of all agents
    agent_outputs: Dict[str, Any]  # Accumulated outputs from completed agents
    quality_gate_results: Dict[str, QualityGateResult]  # Gate evaluations
    cumulative_provenance_hash: str  # SHA-256 covering all state up to this point
    created_at: datetime
    created_by: str              # User or system that triggered checkpoint
```

**Non-Functional Requirements:**
- Performance: Checkpoint write < 500ms (async, non-blocking to workflow execution)
- Performance: Workflow resume from checkpoint < 2 seconds
- Durability: Checkpoints persisted to PostgreSQL with WAL for crash safety
- Retention: 5 years per EUDR Article 31
- Concurrency: Support 1,000+ concurrent workflow state reads

**Dependencies:**
- PostgreSQL + TimescaleDB for persistent state storage
- Redis for real-time progress tracking and pub/sub
- S3 for large agent output storage (referenced by checkpoint)

**Estimated Effort:** 3 weeks (1 senior backend engineer)

---

**P0 Features 7-8: Execution Optimization and Resilience Layer**

---

#### Feature 7: Parallel Execution Engine

**User Story:**
```
As a supply chain analyst,
I want independent agents to execute concurrently to minimize workflow completion time,
So that due diligence for a 1,000-shipment portfolio completes in minutes, not hours.
```

**Acceptance Criteria:**
- [ ] Analyzes DAG topology to identify independent agent groups per execution layer
- [ ] Executes agents within the same layer concurrently using async task pool
- [ ] Supports configurable maximum concurrency per workflow (default: 10 agents, max: 25)
- [ ] Supports configurable global concurrency limit across all workflows (default: 100 agents)
- [ ] Implements work-stealing: idle workers pick up tasks from busy workers' queues
- [ ] Calculates critical path through DAG for ETA estimation
- [ ] Provides execution timeline data for visualization (Gantt chart of agent execution)
- [ ] Handles agent completion events and triggers dependent agents immediately
- [ ] Supports priority-based scheduling: critical-path agents get priority over non-critical
- [ ] Reports parallelization efficiency: actual speedup vs. theoretical maximum

**Parallelization Analysis for Standard Workflow:**

| Layer | Agents | Max Parallel | Sequential Time | Parallel Time | Speedup |
|-------|--------|-------------|-----------------|---------------|---------|
| 0 | EUDR-001 | 1 | 30s | 30s | 1.0x |
| 1 | EUDR-002, 006, 007, 008 | 4 | 120s | 30s | 4.0x |
| 2 | EUDR-003, 004, 005 | 3 | 90s | 30s | 3.0x |
| 3 | EUDR-009, 010, 011 | 3 | 90s | 30s | 3.0x |
| 4 | EUDR-012, 013, 014, 015 | 4 | 120s | 30s | 4.0x |
| QG-1 | Quality Gate | 1 | 5s | 5s | 1.0x |
| 5 | EUDR-016-025 | 10 | 300s | 30s | 10.0x |
| QG-2 | Quality Gate | 1 | 5s | 5s | 1.0x |
| 6 | EUDR-025 (mitigation) | 1 | 30s | 30s | 1.0x |
| QG-3 | Quality Gate | 1 | 5s | 5s | 1.0x |
| 7 | Package Generator | 1 | 15s | 15s | 1.0x |
| **Total** | **25 agents + 3 gates** | -- | **810s (13.5 min)** | **240s (4 min)** | **3.4x** |

**Non-Functional Requirements:**
- Performance: Achieve >= 70% of theoretical maximum parallelization speedup
- Scalability: Handle 1,000+ concurrent workflows without degradation
- Fairness: No single workflow monopolizes resources at the expense of others
- Observability: Per-workflow and per-agent execution metrics in Prometheus

**Dependencies:**
- Python asyncio for concurrent task execution
- Redis Streams for inter-workflow coordination and work distribution
- Workflow Definition Engine (Feature 1) for DAG topology

**Estimated Effort:** 3 weeks (1 senior backend engineer)

---

#### Feature 8: Error Recovery & Retry Manager

**User Story:**
```
As a compliance operations manager,
I want transient failures to be automatically retried with exponential backoff,
So that my workflows recover from temporary infrastructure issues without manual intervention.
```

**Acceptance Criteria:**
- [ ] Implements exponential backoff with jitter: delay = min(base * 2^attempt + random_jitter, max_delay)
- [ ] Configurable retry parameters: base_delay (default 1s), max_delay (default 300s), max_retries (default 5)
- [ ] Implements circuit breaker pattern per agent type with configurable thresholds
- [ ] Circuit breaker states: CLOSED (normal), OPEN (failing, reject calls), HALF_OPEN (test single call)
- [ ] Configurable circuit breaker: failure_threshold (default 5), reset_timeout (default 60s), success_threshold (default 2)
- [ ] Classifies errors: transient (retry), permanent (fail immediately), degraded (use fallback)
- [ ] Fallback strategies: cached_result (use last known good output), degraded_mode (skip non-critical agent), manual_override (pause for human decision)
- [ ] Dead letter queue for permanently failed agent invocations with full context for investigation
- [ ] Supports per-agent retry configuration overrides (e.g., satellite agents may need longer timeouts)
- [ ] Emits retry/circuit-breaker events for monitoring and alerting

**Error Classification Matrix:**

| Error Type | Classification | Action | Example |
|-----------|---------------|--------|---------|
| HTTP 429 (Rate Limited) | Transient | Retry with backoff | External API rate limit |
| HTTP 503 (Service Unavailable) | Transient | Retry with backoff | Agent temporarily down |
| Connection Timeout | Transient | Retry with backoff | Network instability |
| HTTP 400 (Bad Request) | Permanent | Fail immediately | Invalid input data |
| HTTP 401/403 (Auth Error) | Permanent | Fail immediately | Credential issue |
| Data Validation Error | Permanent | Fail with details | Upstream data quality issue |
| Partial Result Available | Degraded | Use fallback + warning | Agent completed partially |
| External Service Outage | Transient -> Circuit Break | Backoff then circuit break | Satellite provider outage |

**Circuit Breaker State Machine:**
```
CLOSED --[failure_count >= threshold]--> OPEN
OPEN --[reset_timeout elapsed]--> HALF_OPEN
HALF_OPEN --[probe succeeds >= success_threshold]--> CLOSED
HALF_OPEN --[probe fails]--> OPEN
```

**Non-Functional Requirements:**
- Recovery rate: >= 95% of transient failures recovered within max_retries
- Circuit breaker: Prevent cascade failures within 5 seconds of detection
- Dead letter: All permanently failed invocations captured with full context
- Observability: Per-agent retry counts, circuit breaker state changes in Prometheus

**Dependencies:**
- Workflow State Manager (Feature 6) for checkpointing before retry
- Redis for circuit breaker state sharing across workflow instances

**Estimated Effort:** 2 weeks (1 senior backend engineer)

---

**P0 Feature 9: Output Delivery Layer**

---

#### Feature 9: Due Diligence Package Generator

**User Story:**
```
As a compliance officer,
I want a single, structured evidence package compiling all 25 agent outputs,
So that I can submit a complete DDS and provide audit-ready documentation to competent authorities.
```

**Acceptance Criteria:**
- [ ] Compiles all 25 agent outputs into a structured evidence bundle
- [ ] Generates DDS-compatible JSON matching EU Information System submission schema
- [ ] Generates human-readable PDF report with: executive summary, information gathering findings, risk assessment results, mitigation measures, evidence annexes, workflow provenance chain
- [ ] Includes SHA-256 integrity hash on every evidence artifact and on the complete package
- [ ] Maps agent outputs to EUDR Article 12(2) DDS content requirements
- [ ] Validates package against EU Information System DDS schema before finalization
- [ ] Supports multi-language output: EN, FR, DE, ES, PT
- [ ] Supports batch package generation for multiple products/shipments in a single workflow
- [ ] Includes workflow execution metadata: duration, agents executed, quality gate results, retry events
- [ ] Provides package download in multiple formats: JSON, PDF, HTML, ZIP (complete bundle)
- [ ] Supports package versioning and amendment for previously submitted DDS updates

**DDS Content Mapping (Article 12(2)):**

| Art. 12(2) | DDS Field | Source Agent(s) | Package Section |
|-----------|-----------|-----------------|-----------------|
| (a) | Operator name and contact | System config | Cover page |
| (b) | Product description, trade name, CN code | EUDR-001 | Section 1: Product Identification |
| (c) | Quantity | EUDR-001, EUDR-011 | Section 1: Product Identification |
| (d) | Country of production | EUDR-001, EUDR-016 | Section 2: Origin |
| (e) | Geolocation of plots | EUDR-002, EUDR-006, EUDR-007 | Section 3: Geolocation Evidence |
| (f) | Date/period of production | EUDR-001, EUDR-009 | Section 2: Origin |
| (g) | Deforestation-free verification | EUDR-003, EUDR-004, EUDR-005, EUDR-020 | Section 4: Deforestation Verification |
| (h) | Legal compliance evidence | EUDR-012, EUDR-023 | Section 5: Legal Compliance |
| (i) | Risk assessment results | EUDR-016-025 | Section 6: Risk Assessment |
| (j) | Risk mitigation measures | EUDR-025 | Section 7: Risk Mitigation |
| -- | Supply chain traceability | EUDR-001, EUDR-008-011, EUDR-013 | Section 8: Supply Chain Evidence |
| -- | Workflow provenance | DDO-026 | Section 9: Audit Trail |

**Non-Functional Requirements:**
- Performance: Package generation < 30 seconds for standard workflow with 1,000 shipments
- Completeness: 100% of Art. 12(2) fields populated (or explicitly flagged as N/A with justification)
- Integrity: Package-level SHA-256 hash verifiable by auditors
- Size: PDF report < 50 MB; ZIP bundle < 200 MB

**Dependencies:**
- ReportLab + WeasyPrint for PDF generation
- All 25 upstream agents for output compilation
- EU Information System DDS schema specification
- S3 for package storage

**Estimated Effort:** 3 weeks (1 senior backend engineer, 1 frontend engineer for PDF template)

---

### 5.2 Could-Have Features (P2 -- Nice to Have)

#### Feature 10: Workflow Analytics Dashboard
- Historical workflow execution statistics (completion rates, average duration, failure hotspots)
- Agent-level performance benchmarking (slowest agents, most error-prone agents)
- Trend analysis: workflow efficiency improvements over time
- Predictive ETA based on historical execution patterns

#### Feature 11: Workflow Template Marketplace
- Community-contributed workflow templates for industry-specific due diligence patterns
- Template rating and review system
- Template versioning with regulatory update notifications
- Export/import of custom workflow definitions

#### Feature 12: Multi-Entity Workflow Coordination
- Coordinate due diligence across multiple legal entities in a corporate group
- Shared evidence reuse: one entity's information gathering results reused by another
- Consolidated group-level due diligence reporting
- Inter-entity quality gate dependencies

---

### 5.3 Won't-Have Features (P3 -- Out of Scope for v1.0)

- AI-powered workflow optimization (automatic DAG restructuring based on execution history -- defer to Phase 2)
- Real-time streaming workflow execution via WebSocket (polling-based progress for v1.0)
- Mobile native workflow management application (web responsive only)
- Integration with non-EUDR due diligence frameworks (CSDDD, CSRD -- defer to Phase 3)
- Blockchain-based workflow immutability (SHA-256 provenance sufficient for v1.0)
- Natural language workflow definition (DAG-based configuration only)

---

## 6. Technical Architecture Overview

### 6.1 Architecture Diagram

```
+------------------------------------------------------------------+
|                     GL-EUDR-APP v1.0                             |
|                   Frontend (React/TS)                            |
|  [Workflow Dashboard] [Progress Tracker] [Package Viewer]        |
+-------------------------------+----------------------------------+
                                |
+-------------------------------v----------------------------------+
|                     Unified API Layer (FastAPI)                   |
+-------------------------------+----------------------------------+
                                |
+-------------------------------v----------------------------------+
|              AGENT-EUDR-026: Due Diligence Orchestrator          |
|                                                                  |
|  +-----------------+  +------------------+  +------------------+ |
|  | Workflow        |  | Quality Gate     |  | Workflow State   | |
|  | Definition      |  | Engine           |  | Manager          | |
|  | Engine (DAG)    |  | (QG-1, QG-2,    |  | (Checkpoint,     | |
|  | (Feature 1)     |  |  QG-3)           |  |  Resume, Audit)  | |
|  +-----------------+  | (Feature 5)      |  | (Feature 6)      | |
|                       +------------------+  +------------------+ |
|  +-----------------+  +------------------+  +------------------+ |
|  | Parallel        |  | Error Recovery   |  | DD Package       | |
|  | Execution       |  | & Retry          |  | Generator        | |
|  | Engine          |  | Manager          |  | (DDS + PDF)      | |
|  | (Feature 7)     |  | (Feature 8)      |  | (Feature 9)      | |
|  +-----------------+  +------------------+  +------------------+ |
|                                                                  |
|  +-----------------+  +------------------+  +------------------+ |
|  | Info Gathering  |  | Risk Assessment  |  | Risk Mitigation  | |
|  | Coordinator     |  | Coordinator      |  | Coordinator      | |
|  | (Feature 2)     |  | (Feature 3)      |  | (Feature 4)      | |
|  +-----------------+  +------------------+  +------------------+ |
+------+---------------------------+---------------------------+---+
       |                           |                           |
+------v-----------+  +------------v-----------+  +------------v---+
| Phase 1: Supply  |  | Phase 2: Risk          |  | Phase 3: Risk  |
| Chain Traceability|  | Assessment             |  | Mitigation     |
| (15 Agents)      |  | (10 Agents)            |  | (1 Agent)      |
|                  |  |                        |  |                |
| EUDR-001 SCM     |  | EUDR-016 Country Risk  |  | EUDR-025 Risk  |
| EUDR-002 GeoVer  |  | EUDR-017 Supplier Risk |  | Mitigation     |
| EUDR-003 Sat.Mon |  | EUDR-018 Commodity     |  | Advisor        |
| EUDR-004 Forest  |  | EUDR-019 Corruption    |  |                |
| EUDR-005 LandUse |  | EUDR-020 Deforestation |  +----------------+
| EUDR-006 PlotBnd |  | EUDR-021 Indigenous    |
| EUDR-007 GPSVal  |  | EUDR-022 Protected     |
| EUDR-008 MultiTr |  | EUDR-023 Legal         |
| EUDR-009 CoC     |  | EUDR-024 Audit         |
| EUDR-010 Segreg  |  | EUDR-025 Risk Assess   |
| EUDR-011 MassBal |  +------------------------+
| EUDR-012 DocAuth |
| EUDR-013 BlockCh |
| EUDR-014 QRCode  |
| EUDR-015 Mobile  |
+------------------+
       |
+------v--------------------------+
| Data & Infrastructure Layer     |
| PostgreSQL + TimescaleDB        |
| Redis (Cache + Streams + PubSub)|
| S3 Object Storage               |
+----------------------------------+
```

### 6.2 Module Structure

```
greenlang/agents/eudr/due_diligence_orchestrator/
    __init__.py                              # Public API exports
    config.py                                # DueDiligenceOrchestratorConfig with GL_EUDR_DDO_ env prefix
    models.py                                # Pydantic v2 models: workflows, checkpoints, gates, packages
    workflow_definition_engine.py             # WorkflowDefinitionEngine: DAG creation, validation, templates
    information_gathering_coordinator.py      # InformationGatheringCoordinator: Phase 1 orchestration
    risk_assessment_coordinator.py           # RiskAssessmentCoordinator: Phase 2 orchestration
    risk_mitigation_coordinator.py           # RiskMitigationCoordinator: Phase 3 orchestration
    quality_gate_engine.py                   # QualityGateEngine: phase transition validation
    workflow_state_manager.py                # WorkflowStateManager: checkpoint, resume, rollback, audit
    parallel_execution_engine.py             # ParallelExecutionEngine: concurrent agent execution
    error_recovery_manager.py                # ErrorRecoveryManager: retry, circuit breaker, fallback
    due_diligence_package_generator.py       # DueDiligencePackageGenerator: DDS JSON + PDF report
    agent_client.py                          # AgentClient: unified interface for invoking all 25 agents
    provenance.py                            # ProvenanceTracker: SHA-256 hash chains for audit trail
    metrics.py                               # 20 Prometheus self-monitoring metrics
    setup.py                                 # DueDiligenceOrchestratorService facade
    workflow_templates/
        __init__.py
        standard_workflow.py                 # Full 25-agent standard due diligence DAG
        simplified_workflow.py               # Reduced Article 13 simplified due diligence DAG
        cattle_workflow.py                   # Cattle-specific workflow template
        cocoa_workflow.py                    # Cocoa-specific workflow template
        coffee_workflow.py                   # Coffee-specific workflow template
        palm_oil_workflow.py                 # Palm oil-specific workflow template
        rubber_workflow.py                   # Rubber-specific workflow template
        soya_workflow.py                     # Soya-specific workflow template
        wood_workflow.py                     # Wood-specific workflow template
    api/
        __init__.py
        router.py                            # FastAPI router (30+ endpoints)
        workflow_routes.py                   # Workflow CRUD and execution endpoints
        phase_routes.py                      # Phase coordination endpoints
        quality_gate_routes.py               # Quality gate management endpoints
        state_routes.py                      # Workflow state and checkpoint endpoints
        execution_routes.py                  # Parallel execution monitoring endpoints
        package_routes.py                    # Package generation and download endpoints
        monitoring_routes.py                 # Health, metrics, and operational endpoints
```

### 6.3 Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| Language | Python 3.11+ | GreenLang platform standard |
| Web Framework | FastAPI | Async, OpenAPI docs, Pydantic v2 native |
| DAG Engine | NetworkX | Topological sorting, dependency resolution, critical path analysis |
| Async Runtime | Python asyncio + anyio | Concurrent agent execution with structured concurrency |
| Task Queue | Redis Streams | Distributed task dispatch, work-stealing, event streaming |
| Database | PostgreSQL + TimescaleDB | Persistent workflow state + time-series checkpoint hypertables |
| Cache | Redis | Agent output caching, circuit breaker state, real-time progress |
| Object Storage | S3 | Large agent outputs, evidence packages, PDF reports |
| Serialization | Pydantic v2 | Type-safe, validated, JSON-compatible workflow models |
| PDF Generation | ReportLab + WeasyPrint | Audit-ready PDF due diligence reports |
| Authentication | JWT (RS256) via SEC-001 | Standard GreenLang auth |
| Authorization | RBAC via SEC-002 | Role-based workflow access control |
| Monitoring | Prometheus + Grafana | 20 metrics + dedicated orchestration dashboard |
| Tracing | OpenTelemetry | End-to-end distributed tracing across all 25 agent calls |
| CI/CD | GitHub Actions | Standard GreenLang pipeline |
| Deployment | Kubernetes (EKS) | Standard GreenLang deployment with HPA for auto-scaling |

---

## 7. Data Model and Schemas

### 7.1 Core Domain Models

```python
# Due Diligence Phase Enumeration
class DueDiligencePhase(str, Enum):
    INFORMATION_GATHERING = "information_gathering"  # Article 9
    RISK_ASSESSMENT = "risk_assessment"              # Article 10
    RISK_MITIGATION = "risk_mitigation"              # Article 11
    PACKAGE_GENERATION = "package_generation"         # Article 12

# Workflow Type
class WorkflowType(str, Enum):
    STANDARD = "standard"          # Full 25-agent due diligence
    SIMPLIFIED = "simplified"      # Article 13 reduced workflow
    CUSTOM = "custom"              # Operator-customized workflow

# Agent Execution Status
class AgentExecutionStatus(str, Enum):
    PENDING = "pending"            # Not yet started
    QUEUED = "queued"              # In execution queue
    RUNNING = "running"            # Currently executing
    COMPLETED = "completed"        # Finished successfully
    FAILED = "failed"              # Failed (may retry)
    RETRYING = "retrying"          # Retrying after failure
    SKIPPED = "skipped"            # Skipped (not applicable or degraded mode)
    CIRCUIT_BROKEN = "circuit_broken"  # Circuit breaker open

# Workflow Status
class WorkflowStatus(str, Enum):
    CREATED = "created"            # Workflow defined but not started
    RUNNING = "running"            # Actively executing agents
    PAUSED = "paused"              # Execution paused by user
    QUALITY_GATE = "quality_gate"  # Evaluating quality gate
    GATE_FAILED = "gate_failed"    # Quality gate failed, awaiting remediation
    COMPLETED = "completed"        # All phases complete, package generated
    CANCELLED = "cancelled"        # Cancelled by user
    TERMINATED = "terminated"      # Unrecoverable failure

# Quality Gate Result
class QualityGateResult(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    OVERRIDDEN = "overridden"      # Failed but manually overridden with justification

# Circuit Breaker State
class CircuitBreakerState(str, Enum):
    CLOSED = "closed"              # Normal operation
    OPEN = "open"                  # Rejecting calls
    HALF_OPEN = "half_open"        # Testing recovery
```

### 7.2 Workflow Definition Model

```python
class WorkflowDefinition(BaseModel):
    workflow_def_id: str                     # UUID
    name: str                                # Human-readable workflow name
    description: str                         # Workflow purpose
    workflow_type: WorkflowType              # Standard / Simplified / Custom
    commodity: Optional[EUDRCommodity]       # Target commodity (None for multi-commodity)
    version: int                             # Definition version number
    agent_nodes: List[AgentNode]             # Agent definitions in the workflow
    dependency_edges: List[DependencyEdge]   # Dependencies between agents
    quality_gates: List[QualityGateDefinition]  # Gate definitions
    config: WorkflowConfig                   # Execution configuration
    created_at: datetime
    updated_at: datetime

class AgentNode(BaseModel):
    agent_id: str                            # e.g., "EUDR-001"
    agent_name: str                          # e.g., "Supply Chain Mapping Master"
    phase: DueDiligencePhase                 # Which DD phase this agent belongs to
    layer: int                               # Execution layer (0-7)
    is_critical: bool                        # If True, failure blocks workflow
    is_required: bool                        # If False, can be skipped in simplified mode
    timeout_seconds: int                     # Maximum execution time
    retry_config: Optional[RetryConfig]      # Agent-specific retry overrides
    input_mapping: Dict[str, str]            # Maps upstream outputs to this agent's inputs

class DependencyEdge(BaseModel):
    source_agent_id: str                     # Upstream agent
    target_agent_id: str                     # Downstream agent
    dependency_type: str                     # "data" (output required) or "completion" (just needs to finish)

class QualityGateDefinition(BaseModel):
    gate_id: str                             # e.g., "QG-1"
    gate_name: str                           # e.g., "Information Gathering Completeness"
    phase_before: DueDiligencePhase          # Phase being validated
    phase_after: DueDiligencePhase           # Phase to enter if gate passes
    checks: List[QualityCheck]               # Individual checks
    threshold: float                         # Overall pass threshold (0-100)
    allow_override: bool                     # Can be manually overridden
    relaxed_threshold: Optional[float]       # Threshold for simplified DD

class WorkflowConfig(BaseModel):
    max_concurrency: int = 10                # Max concurrent agents per workflow
    max_retries: int = 5                     # Default max retries per agent
    base_retry_delay_seconds: float = 1.0    # Exponential backoff base
    max_retry_delay_seconds: float = 300.0   # Maximum retry delay
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_reset_timeout_seconds: int = 60
    checkpoint_enabled: bool = True          # Enable state checkpointing
    package_languages: List[str] = ["en"]    # Output languages
```

### 7.3 Workflow Execution Model

```python
class WorkflowExecution(BaseModel):
    workflow_id: str                          # UUID - unique execution instance
    workflow_def_id: str                      # Reference to workflow definition
    operator_id: str                          # Operator who initiated
    commodity: EUDRCommodity                  # Target commodity
    product_ids: List[str]                    # Products being evaluated
    status: WorkflowStatus                   # Current workflow status
    current_phase: DueDiligencePhase         # Current execution phase
    agent_statuses: Dict[str, AgentExecution]  # Per-agent execution state
    quality_gate_results: Dict[str, GateEvaluation]  # Gate evaluation results
    progress: WorkflowProgress               # Real-time progress data
    error_summary: Optional[ErrorSummary]    # Summary of errors encountered
    package_id: Optional[str]                # Generated package ID (when complete)
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    estimated_completion: Optional[datetime]  # ETA based on progress
    created_at: datetime
    updated_at: datetime

class AgentExecution(BaseModel):
    agent_id: str
    status: AgentExecutionStatus
    attempt_number: int                      # Current attempt (1 = first try)
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]
    output_ref: Optional[str]                # S3 reference for output data
    output_summary: Optional[Dict[str, Any]] # Key output metrics
    error: Optional[AgentError]
    retry_history: List[RetryAttempt]

class GateEvaluation(BaseModel):
    gate_id: str
    result: QualityGateResult
    overall_score: float                     # 0-100
    check_results: List[CheckResult]         # Individual check results
    threshold_used: float                    # Threshold that was applied
    override_justification: Optional[str]    # If overridden
    evaluated_at: datetime
    evaluated_by: str

class WorkflowProgress(BaseModel):
    total_agents: int
    completed_agents: int
    running_agents: int
    failed_agents: int
    skipped_agents: int
    pending_agents: int
    completion_percentage: float             # 0-100
    current_phase: DueDiligencePhase
    phase_progress: Dict[str, float]         # Per-phase completion %
    estimated_remaining_seconds: Optional[int]
    critical_path_agents: List[str]          # Agents on critical path
```

### 7.4 Due Diligence Package Model

```python
class DueDiligencePackage(BaseModel):
    package_id: str                          # UUID
    workflow_id: str                         # Source workflow
    operator_id: str                         # Operator
    commodity: EUDRCommodity
    product_ids: List[str]
    workflow_type: WorkflowType              # Standard / Simplified
    dds_json: Dict[str, Any]                 # DDS-compatible JSON for EU submission
    sections: List[PackageSection]           # Report sections with evidence
    quality_gate_summary: Dict[str, GateEvaluation]
    risk_profile: RiskProfile                # Aggregated risk scores
    mitigation_summary: Optional[MitigationSummary]
    provenance_chain: List[ProvenanceEntry]  # SHA-256 hash chain
    package_hash: str                        # SHA-256 of complete package
    languages: List[str]                     # Generated languages
    created_at: datetime
    version: int

class PackageSection(BaseModel):
    section_number: int
    title: str                               # e.g., "Product Identification"
    dds_article_ref: str                     # e.g., "Art. 12(2)(b)"
    source_agents: List[str]                 # Agents that contributed
    content: Dict[str, Any]                  # Section content
    evidence_refs: List[str]                 # S3 references to evidence files
    evidence_hash: str                       # SHA-256 of all evidence in section
```

### 7.5 Database Schema (New Migration: V109)

```sql
CREATE SCHEMA IF NOT EXISTS eudr_due_diligence_orchestrator;

-- Workflow definitions (templates and custom)
CREATE TABLE eudr_due_diligence_orchestrator.workflow_definitions (
    workflow_def_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) NOT NULL,
    description TEXT,
    workflow_type VARCHAR(20) NOT NULL DEFAULT 'standard',
    commodity VARCHAR(50),
    version INTEGER NOT NULL DEFAULT 1,
    agent_nodes JSONB NOT NULL DEFAULT '[]',
    dependency_edges JSONB NOT NULL DEFAULT '[]',
    quality_gates JSONB NOT NULL DEFAULT '[]',
    config JSONB NOT NULL DEFAULT '{}',
    is_system_template BOOLEAN DEFAULT FALSE,
    created_by VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Workflow executions (one per due diligence run)
CREATE TABLE eudr_due_diligence_orchestrator.workflow_executions (
    workflow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_def_id UUID NOT NULL REFERENCES eudr_due_diligence_orchestrator.workflow_definitions(workflow_def_id),
    operator_id UUID NOT NULL,
    commodity VARCHAR(50) NOT NULL,
    product_ids JSONB NOT NULL DEFAULT '[]',
    status VARCHAR(30) NOT NULL DEFAULT 'created',
    current_phase VARCHAR(30) NOT NULL DEFAULT 'information_gathering',
    agent_statuses JSONB NOT NULL DEFAULT '{}',
    progress JSONB NOT NULL DEFAULT '{}',
    error_summary JSONB,
    package_id UUID,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    estimated_completion TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_operator FOREIGN KEY (operator_id) REFERENCES auth.users(id)
);

-- Workflow checkpoints (hypertable for time-series)
CREATE TABLE eudr_due_diligence_orchestrator.workflow_checkpoints (
    checkpoint_id UUID DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL,
    sequence_number INTEGER NOT NULL,
    phase VARCHAR(30) NOT NULL,
    agent_id VARCHAR(20),
    gate_id VARCHAR(10),
    checkpoint_type VARCHAR(20) NOT NULL,  -- 'agent_complete', 'gate_evaluated', 'phase_transition'
    agent_statuses JSONB NOT NULL DEFAULT '{}',
    agent_outputs_ref VARCHAR(500),         -- S3 reference for large outputs
    quality_gate_results JSONB DEFAULT '{}',
    cumulative_provenance_hash VARCHAR(64) NOT NULL,
    created_by VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_due_diligence_orchestrator.workflow_checkpoints', 'created_at');

-- Quality gate evaluations (hypertable)
CREATE TABLE eudr_due_diligence_orchestrator.quality_gate_evaluations (
    evaluation_id UUID DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL,
    gate_id VARCHAR(10) NOT NULL,
    result VARCHAR(20) NOT NULL,
    overall_score NUMERIC(5,2) NOT NULL,
    check_results JSONB NOT NULL DEFAULT '[]',
    threshold_used NUMERIC(5,2) NOT NULL,
    override_justification TEXT,
    evaluated_by VARCHAR(100),
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_due_diligence_orchestrator.quality_gate_evaluations', 'evaluated_at');

-- Agent execution log (hypertable)
CREATE TABLE eudr_due_diligence_orchestrator.agent_execution_log (
    log_id UUID DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL,
    agent_id VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    attempt_number INTEGER NOT NULL DEFAULT 1,
    duration_seconds NUMERIC(10,3),
    output_ref VARCHAR(500),
    output_summary JSONB,
    error_type VARCHAR(50),
    error_message TEXT,
    error_classification VARCHAR(20),  -- 'transient', 'permanent', 'degraded'
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_due_diligence_orchestrator.agent_execution_log', 'completed_at');

-- Circuit breaker state
CREATE TABLE eudr_due_diligence_orchestrator.circuit_breaker_state (
    agent_id VARCHAR(20) PRIMARY KEY,
    state VARCHAR(15) NOT NULL DEFAULT 'closed',
    failure_count INTEGER NOT NULL DEFAULT 0,
    last_failure_at TIMESTAMPTZ,
    last_success_at TIMESTAMPTZ,
    opened_at TIMESTAMPTZ,
    half_open_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Due diligence packages
CREATE TABLE eudr_due_diligence_orchestrator.due_diligence_packages (
    package_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES eudr_due_diligence_orchestrator.workflow_executions(workflow_id),
    operator_id UUID NOT NULL,
    commodity VARCHAR(50) NOT NULL,
    product_ids JSONB NOT NULL DEFAULT '[]',
    workflow_type VARCHAR(20) NOT NULL,
    dds_json JSONB NOT NULL,
    sections JSONB NOT NULL DEFAULT '[]',
    quality_gate_summary JSONB NOT NULL DEFAULT '{}',
    risk_profile JSONB NOT NULL DEFAULT '{}',
    mitigation_summary JSONB,
    provenance_chain JSONB NOT NULL DEFAULT '[]',
    package_hash VARCHAR(64) NOT NULL,
    languages JSONB NOT NULL DEFAULT '["en"]',
    pdf_ref VARCHAR(500),              -- S3 reference for PDF report
    zip_ref VARCHAR(500),              -- S3 reference for ZIP bundle
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Workflow audit trail (hypertable)
CREATE TABLE eudr_due_diligence_orchestrator.workflow_audit_trail (
    audit_id UUID DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL,   -- 'workflow_started', 'agent_completed', 'gate_evaluated', etc.
    event_data JSONB NOT NULL DEFAULT '{}',
    actor VARCHAR(100) NOT NULL,       -- User or system
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('eudr_due_diligence_orchestrator.workflow_audit_trail', 'created_at');

-- Dead letter queue for permanently failed invocations
CREATE TABLE eudr_due_diligence_orchestrator.dead_letter_queue (
    dlq_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL,
    agent_id VARCHAR(20) NOT NULL,
    attempt_number INTEGER NOT NULL,
    error_type VARCHAR(50) NOT NULL,
    error_message TEXT NOT NULL,
    input_data JSONB,
    retry_history JSONB NOT NULL DEFAULT '[]',
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(100),
    resolution_notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_wf_exec_operator ON eudr_due_diligence_orchestrator.workflow_executions(operator_id);
CREATE INDEX idx_wf_exec_status ON eudr_due_diligence_orchestrator.workflow_executions(status);
CREATE INDEX idx_wf_exec_commodity ON eudr_due_diligence_orchestrator.workflow_executions(commodity);
CREATE INDEX idx_checkpoints_workflow ON eudr_due_diligence_orchestrator.workflow_checkpoints(workflow_id);
CREATE INDEX idx_gate_eval_workflow ON eudr_due_diligence_orchestrator.quality_gate_evaluations(workflow_id);
CREATE INDEX idx_agent_log_workflow ON eudr_due_diligence_orchestrator.agent_execution_log(workflow_id);
CREATE INDEX idx_agent_log_agent ON eudr_due_diligence_orchestrator.agent_execution_log(agent_id);
CREATE INDEX idx_audit_workflow ON eudr_due_diligence_orchestrator.workflow_audit_trail(workflow_id);
CREATE INDEX idx_audit_event ON eudr_due_diligence_orchestrator.workflow_audit_trail(event_type);
CREATE INDEX idx_packages_operator ON eudr_due_diligence_orchestrator.due_diligence_packages(operator_id);
CREATE INDEX idx_packages_workflow ON eudr_due_diligence_orchestrator.due_diligence_packages(workflow_id);
CREATE INDEX idx_dlq_workflow ON eudr_due_diligence_orchestrator.dead_letter_queue(workflow_id);
CREATE INDEX idx_dlq_unresolved ON eudr_due_diligence_orchestrator.dead_letter_queue(resolved) WHERE resolved = FALSE;
```

---

## 8. API and Integration Points

### 8.1 REST API Endpoints (30+)

| Method | Path | Description |
|--------|------|-------------|
| **Workflow Management** | | |
| POST | `/v1/workflows` | Create and start a new due diligence workflow |
| GET | `/v1/workflows` | List workflows (with filters: status, commodity, operator, date range) |
| GET | `/v1/workflows/{workflow_id}` | Get workflow details with real-time progress |
| POST | `/v1/workflows/{workflow_id}/pause` | Pause a running workflow |
| POST | `/v1/workflows/{workflow_id}/resume` | Resume a paused or failed workflow from checkpoint |
| POST | `/v1/workflows/{workflow_id}/cancel` | Cancel a workflow |
| POST | `/v1/workflows/{workflow_id}/clone` | Clone a workflow for re-execution |
| DELETE | `/v1/workflows/{workflow_id}` | Archive a workflow (soft delete) |
| **Workflow Templates** | | |
| GET | `/v1/templates` | List available workflow templates |
| GET | `/v1/templates/{template_id}` | Get template definition with DAG |
| POST | `/v1/templates` | Create custom workflow template |
| PUT | `/v1/templates/{template_id}` | Update custom template |
| **Phase Coordination** | | |
| GET | `/v1/workflows/{workflow_id}/phases` | Get phase-level progress summary |
| GET | `/v1/workflows/{workflow_id}/phases/{phase}/agents` | Get agent-level status for a phase |
| POST | `/v1/workflows/{workflow_id}/phases/{phase}/retry` | Retry all failed agents in a phase |
| **Quality Gates** | | |
| GET | `/v1/workflows/{workflow_id}/gates` | Get all quality gate evaluations |
| GET | `/v1/workflows/{workflow_id}/gates/{gate_id}` | Get detailed gate evaluation with check results |
| POST | `/v1/workflows/{workflow_id}/gates/{gate_id}/override` | Override a failed quality gate (with justification) |
| POST | `/v1/workflows/{workflow_id}/gates/{gate_id}/re-evaluate` | Re-evaluate a gate after remediation |
| **Workflow State** | | |
| GET | `/v1/workflows/{workflow_id}/checkpoints` | List workflow checkpoints |
| GET | `/v1/workflows/{workflow_id}/checkpoints/{checkpoint_id}` | Get checkpoint details |
| POST | `/v1/workflows/{workflow_id}/rollback/{checkpoint_id}` | Rollback to specific checkpoint |
| GET | `/v1/workflows/{workflow_id}/audit-trail` | Get complete workflow audit trail |
| **Execution Monitoring** | | |
| GET | `/v1/workflows/{workflow_id}/execution/timeline` | Get execution timeline (Gantt data) |
| GET | `/v1/workflows/{workflow_id}/execution/critical-path` | Get critical path analysis |
| GET | `/v1/workflows/{workflow_id}/agents/{agent_id}/retries` | Get retry history for an agent |
| **Package Generation** | | |
| POST | `/v1/workflows/{workflow_id}/package/generate` | Generate due diligence package |
| GET | `/v1/workflows/{workflow_id}/package` | Get package metadata |
| GET | `/v1/workflows/{workflow_id}/package/download/{format}` | Download package (json, pdf, html, zip) |
| POST | `/v1/workflows/{workflow_id}/package/validate` | Validate package against DDS schema |
| **Batch Operations** | | |
| POST | `/v1/workflows/batch` | Create batch of workflows for multiple products |
| GET | `/v1/workflows/batch/{batch_id}` | Get batch execution summary |
| **Circuit Breakers** | | |
| GET | `/v1/circuit-breakers` | Get circuit breaker states for all agents |
| POST | `/v1/circuit-breakers/{agent_id}/reset` | Manually reset a circuit breaker |
| **Dead Letter Queue** | | |
| GET | `/v1/dead-letter-queue` | List dead letter entries |
| POST | `/v1/dead-letter-queue/{dlq_id}/resolve` | Resolve a dead letter entry |
| **Health** | | |
| GET | `/health` | Service health check |

### 8.2 Upstream Agent Integration (25 Agents)

#### Phase 1: Supply Chain Traceability (EUDR-001 through EUDR-015)

| Agent | Integration Method | Input From | Output To | Invocation Pattern |
|-------|-------------------|------------|-----------|-------------------|
| EUDR-001 Supply Chain Mapping | REST API | Operator product/shipment data | Plot data, supplier graph -> EUDR-002,006,007,008 | Synchronous, Layer 0 |
| EUDR-002 Geolocation Verification | REST API | Plot coordinates from EUDR-001 | Verified coordinates -> EUDR-003,004,005 | Async parallel, Layer 1 |
| EUDR-003 Satellite Monitoring | REST API | Verified plot locations from EUDR-002 | Satellite imagery, change detection -> QG-1 | Async parallel, Layer 2 |
| EUDR-004 Forest Cover Analysis | REST API | Plot boundaries from EUDR-006 | Forest cover metrics -> QG-1 | Async parallel, Layer 2 |
| EUDR-005 Land Use Change Detector | REST API | Plot boundaries from EUDR-006 | Land use change events -> QG-1 | Async parallel, Layer 2 |
| EUDR-006 Plot Boundary Manager | REST API | Plot data from EUDR-001 | Polygon boundaries -> EUDR-003,004,005 | Async parallel, Layer 1 |
| EUDR-007 GPS Coordinate Validator | REST API | GPS coordinates from EUDR-001 | Validated coordinates -> QG-1 | Async parallel, Layer 1 |
| EUDR-008 Multi-Tier Supplier Tracker | REST API | Supplier graph from EUDR-001 | Sub-tier supplier data -> EUDR-009 | Async parallel, Layer 1 |
| EUDR-009 Chain of Custody | REST API | Supplier data from EUDR-008 | Custody chain records -> EUDR-010,011 | Async parallel, Layer 3 |
| EUDR-010 Segregation Verifier | REST API | Custody chain from EUDR-009 | Segregation integrity -> QG-1 | Async parallel, Layer 3 |
| EUDR-011 Mass Balance Calculator | REST API | Custody chain from EUDR-009 | Mass balance verification -> QG-1 | Async parallel, Layer 3 |
| EUDR-012 Document Authentication | REST API | Documents from Layers 2-3 | Authenticated documents -> QG-1 | Async parallel, Layer 4 |
| EUDR-013 Blockchain Integration | REST API | Traceability data from Layers 2-3 | Immutable records -> QG-1 | Async parallel, Layer 4 |
| EUDR-014 QR Code Generator | REST API | Product/batch data from Layers 2-3 | QR codes for products -> QG-1 | Async parallel, Layer 4 |
| EUDR-015 Mobile Data Collector | REST API | Field data requirements from Layers 2-3 | Mobile-collected evidence -> QG-1 | Async parallel, Layer 4 |

#### Phase 2: Risk Assessment (EUDR-016 through EUDR-025)

| Agent | Integration Method | Input From | Output To | Invocation Pattern |
|-------|-------------------|------------|-----------|-------------------|
| EUDR-016 Country Risk Evaluator | REST API | Countries from Phase 1 | Country risk scores -> Composite risk | Async parallel, Layer 5 |
| EUDR-017 Supplier Risk Scorer | REST API | Suppliers from Phase 1 | Supplier risk scores -> Composite risk | Async parallel, Layer 5 |
| EUDR-018 Commodity Risk Analyzer | REST API | Commodities from Phase 1 | Commodity risk profiles -> Composite risk | Async parallel, Layer 5 |
| EUDR-019 Corruption Index Monitor | REST API | Countries from Phase 1 | Corruption scores -> Composite risk | Async parallel, Layer 5 |
| EUDR-020 Deforestation Alert System | REST API | Plot locations from Phase 1 | Deforestation alerts -> Composite risk | Async parallel, Layer 5 |
| EUDR-021 Indigenous Rights Checker | REST API | Plot locations from Phase 1 | Rights overlap results -> Composite risk | Async parallel, Layer 5 |
| EUDR-022 Protected Area Validator | REST API | Plot locations from Phase 1 | Protected area overlaps -> Composite risk | Async parallel, Layer 5 |
| EUDR-023 Legal Compliance Verifier | REST API | Supplier/country data from Phase 1 | Legal compliance status -> Composite risk | Async parallel, Layer 5 |
| EUDR-024 Third-Party Audit Manager | REST API | Supplier data from Phase 1 | Audit findings -> Composite risk | Async parallel, Layer 5 |
| EUDR-025 Risk Mitigation Advisor | REST API | All risk scores from Layer 5 | Risk assessment summary -> QG-2 | Async parallel, Layer 5 |

#### Phase 3: Risk Mitigation (EUDR-025)

| Agent | Integration Method | Input From | Output To | Invocation Pattern |
|-------|-------------------|------------|-----------|-------------------|
| EUDR-025 Risk Mitigation Advisor (mitigation mode) | REST API | Composite risk profile from Phase 2 | Mitigation strategies, evidence -> QG-3 | Synchronous, Layer 6 |

### 8.3 Downstream Consumers

| Consumer | Integration | Data Flow |
|----------|-------------|-----------|
| GL-EUDR-APP v1.0 | REST API | Workflow progress, package download -> frontend display |
| Future EUDR-027 DDS Generator | REST API | Due diligence package -> DDS submission to EU Information System |
| External Auditors | Read-only API + package export | Audit trail, evidence package -> third-party verification |
| Prometheus / Grafana | Metrics endpoint | Workflow metrics -> operational dashboards |
| OpenTelemetry / Tempo | Trace propagation | Distributed traces -> trace visualization |

---

## 9. Security and Compliance

### 9.1 RBAC Permissions (SEC-002 Integration)

| Permission | Description | Roles |
|------------|-------------|-------|
| `eudr-ddo:workflows:read` | View workflow status and progress | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ddo:workflows:create` | Create and start new workflows | Analyst, Compliance Officer, Admin |
| `eudr-ddo:workflows:manage` | Pause, resume, cancel, clone workflows | Compliance Officer, Admin |
| `eudr-ddo:workflows:delete` | Archive workflows | Admin |
| `eudr-ddo:templates:read` | View workflow templates | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ddo:templates:manage` | Create and modify custom templates | Compliance Officer, Admin |
| `eudr-ddo:gates:read` | View quality gate evaluations | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ddo:gates:override` | Override failed quality gates | Compliance Officer, Admin |
| `eudr-ddo:checkpoints:read` | View checkpoint data | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ddo:checkpoints:rollback` | Rollback to checkpoint | Compliance Officer, Admin |
| `eudr-ddo:audit-trail:read` | View workflow audit trail | Auditor (read-only), Compliance Officer, Admin |
| `eudr-ddo:packages:read` | View due diligence packages | Viewer, Analyst, Compliance Officer, Admin |
| `eudr-ddo:packages:generate` | Generate due diligence packages | Compliance Officer, Admin |
| `eudr-ddo:packages:download` | Download packages (PDF, ZIP, JSON) | Compliance Officer, Admin |
| `eudr-ddo:batch:manage` | Create and manage batch workflows | Compliance Officer, Admin |
| `eudr-ddo:circuit-breakers:read` | View circuit breaker states | Analyst, Compliance Officer, Admin, Ops Engineer |
| `eudr-ddo:circuit-breakers:manage` | Reset circuit breakers | Admin, Ops Engineer |
| `eudr-ddo:dlq:read` | View dead letter queue | Analyst, Compliance Officer, Admin, Ops Engineer |
| `eudr-ddo:dlq:manage` | Resolve dead letter entries | Admin, Ops Engineer |

### 9.2 Data Security

| Requirement | Implementation |
|-------------|---------------|
| Encryption at rest | AES-256-GCM via SEC-003 for all workflow state and packages |
| Encryption in transit | TLS 1.3 via SEC-004 for all API communication and inter-agent calls |
| Data integrity | SHA-256 provenance hashes on every checkpoint and package artifact |
| Audit trail immutability | TimescaleDB hypertable with append-only pattern; no UPDATE/DELETE on audit tables |
| Secrets management | Vault integration via SEC-006 for agent credentials and API keys |
| PII protection | PII detection and redaction via SEC-011 in package outputs |
| Access logging | All API access logged via SEC-005 Centralized Audit Logging |
| Record retention | 5 years minimum per EUDR Article 31; configurable retention policy |

### 9.3 Multi-Tenant Isolation

| Isolation Layer | Implementation |
|-----------------|---------------|
| Workflow isolation | Each workflow scoped to operator_id; cross-operator access blocked by RBAC |
| Agent output isolation | Agent outputs stored in operator-scoped S3 prefixes |
| Circuit breaker isolation | Per-operator circuit breaker state (one operator's failures do not affect another) |
| Resource isolation | Kubernetes resource quotas per operator tier |

---

## 10. Performance and Scalability

### 10.1 Performance Targets

| Metric | Target | Test Condition |
|--------|--------|----------------|
| Standard workflow end-to-end | < 5 minutes | 1,000-shipment portfolio, all 25 agents |
| Simplified workflow end-to-end | < 2 minutes | Low-risk commodity, reduced agent set |
| Workflow creation and start | < 500ms | Single workflow, includes DAG validation |
| Checkpoint write latency | < 500ms | Async write to PostgreSQL |
| Checkpoint resume latency | < 2 seconds | Resume from any checkpoint |
| Quality gate evaluation | < 5 seconds | 10,000-shipment workflow |
| Package generation (JSON) | < 10 seconds | Standard workflow, 1,000 shipments |
| Package generation (PDF) | < 30 seconds | Standard workflow with full evidence |
| Batch workflow creation | < 5 seconds | 100 workflows in a batch |
| API response time (read) | < 200ms p95 | Workflow status, progress queries |
| API response time (write) | < 500ms p95 | Workflow creation, gate overrides |

### 10.2 Scalability Targets

| Metric | Target | Scaling Strategy |
|--------|--------|-----------------|
| Concurrent workflows | 1,000+ | Horizontal scaling via K8s HPA; Redis Streams for distributed coordination |
| Concurrent agent executions | 5,000+ | Async execution pool; distributed across multiple orchestrator pods |
| Workflow state storage | 10M+ checkpoints | TimescaleDB hypertable with compression; S3 for large outputs |
| Package storage | 100,000+ packages | S3 with lifecycle policies; hot/warm/cold tiering |
| Database connections | 100+ per pod | Connection pooling via psycopg_pool; read replicas for queries |

### 10.3 Auto-Scaling Configuration

```yaml
# Kubernetes HPA for orchestrator pods
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: eudr-ddo-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: eudr-due-diligence-orchestrator
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: gl_eudr_ddo_active_workflows
      target:
        type: AverageValue
        averageValue: 50    # Scale at 50 active workflows per pod
```

---

## 11. Testing and Quality Assurance

### 11.1 Test Categories

| Category | Test Count (Target) | Description |
|----------|--------------------|----|
| Workflow Definition Tests | 100+ | DAG creation, validation, topological sort, cycle detection, template loading |
| Information Gathering Coordinator Tests | 80+ | Phase 1 orchestration, dependency resolution, data handoff, completeness scoring |
| Risk Assessment Coordinator Tests | 80+ | Phase 2 orchestration, parallel risk agent execution, composite scoring |
| Risk Mitigation Coordinator Tests | 60+ | Phase 3 orchestration, mitigation adequacy, bypass logic |
| Quality Gate Tests | 100+ | All 3 gates, all checks, threshold enforcement, override handling, relaxed mode |
| Workflow State Manager Tests | 120+ | Checkpoint, resume, rollback, pause, cancel, clone, audit trail, provenance |
| Parallel Execution Engine Tests | 80+ | Concurrency, work-stealing, critical path, ETA, resource limits |
| Error Recovery Tests | 100+ | Exponential backoff, circuit breaker state machine, fallback strategies, DLQ |
| Package Generator Tests | 80+ | DDS JSON, PDF generation, multi-language, schema validation, integrity hash |
| API Tests | 100+ | All 30+ endpoints, auth, pagination, error handling, batch operations |
| Golden Workflow Tests | 49+ | 7 commodities x 7 scenarios (complete, partial, high-risk, simplified, etc.) |
| Integration Tests | 40+ | End-to-end with mock agents, cross-phase data flow, quality gate transitions |
| Performance Tests | 30+ | 1K/5K/10K shipment workflows, concurrent workflow load, checkpoint throughput |
| Chaos Tests | 20+ | Agent failure injection, network partition, database unavailability, timeout |
| **Total** | **1,040+** | |

### 11.2 Golden Workflow Test Scenarios

Each of the 7 commodities will have dedicated golden test workflows:

| # | Scenario | Expected Outcome |
|---|----------|-----------------|
| 1 | Complete standard workflow | All 25 agents complete; all 3 quality gates pass; package generated |
| 2 | Simplified workflow (low-risk country) | Reduced agent set executes; relaxed gates pass; simplified package |
| 3 | Quality Gate 1 failure (missing geolocation) | Phase 1 gate fails; gap report identifies missing plots; remediation guidance |
| 4 | Quality Gate 2 failure (incomplete risk coverage) | Phase 2 gate fails; missing risk dimensions identified |
| 5 | High-risk workflow requiring enhanced mitigation | Full mitigation phase executes; enhanced evidence required |
| 6 | Agent failure with successful retry | Transient failure retried; workflow completes normally |
| 7 | Agent failure with circuit breaker activation | Circuit breaker opens; degraded mode with fallback |

Total: 7 commodities x 7 scenarios = 49 golden workflow tests

### 11.3 Chaos Testing

| Test | Description | Expected Behavior |
|------|-------------|-------------------|
| Agent timeout during Phase 1 | Simulate EUDR-003 timeout | Retry with backoff; resume from checkpoint if max retries exceeded |
| Database unavailability | Kill PostgreSQL connection | Checkpoint queue buffers; resume when connection restored |
| Redis outage | Simulate Redis failure | Fall back to database-only state management; reduced performance |
| Concurrent workflow overload | Launch 2,000 simultaneous workflows | HPA scales pods; backpressure prevents resource exhaustion |
| Mid-workflow pod restart | Kill orchestrator pod during execution | New pod resumes workflow from last checkpoint |

---

## 12. Documentation Requirements

### 12.1 Developer Documentation

| Document | Description | Format |
|----------|-------------|--------|
| API Reference | OpenAPI 3.1 specification for all 30+ endpoints | Swagger UI + YAML |
| Architecture Guide | System architecture, DAG engine, state machine, integration patterns | Markdown |
| Agent Integration Guide | How to integrate a new agent into the orchestrator DAG | Markdown + examples |
| Workflow Template Guide | How to create custom workflow definitions | Markdown + YAML examples |
| Error Handling Guide | Error classification, retry behavior, circuit breaker configuration | Markdown |

### 12.2 Operator Documentation

| Document | Description | Format |
|----------|-------------|--------|
| User Guide | End-to-end workflow execution, progress monitoring, package download | PDF + web |
| Quality Gate Guide | Understanding quality gates, interpreting results, remediation steps | PDF + web |
| Workflow Template Catalog | Descriptions of all 7 commodity templates and simplified variants | PDF + web |
| Troubleshooting Guide | Common errors, resolution steps, support escalation | PDF + web |
| DDS Submission Guide | How to submit generated DDS packages to EU Information System | PDF + web |

### 12.3 Compliance Documentation

| Document | Description | Format |
|----------|-------------|--------|
| EUDR Article Mapping | Detailed mapping of every workflow step to EUDR article requirements | PDF |
| Audit Trail Specification | Format, content, and integrity verification of workflow audit trails | PDF |
| Evidence Package Specification | Structure, content, and hash verification of due diligence packages | PDF |
| Quality Gate Criteria | Detailed quality gate check definitions and threshold justifications | PDF |

---

## 13. Implementation Roadmap

### Phase 1: Core Orchestration Engine (Weeks 1-6)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 1-2 | Workflow Definition Engine (Feature 1): DAG creation, topological sort, validation, 7 commodity templates | Senior Backend Engineer |
| 2-3 | Workflow State Manager (Feature 6): checkpoint, resume, rollback, audit trail, provenance hashing | Senior Backend Engineer |
| 3-4 | Parallel Execution Engine (Feature 7): async agent pool, concurrency limits, critical path analysis | Senior Backend Engineer |
| 4-5 | Error Recovery & Retry Manager (Feature 8): exponential backoff, circuit breaker, fallback, DLQ | Senior Backend Engineer |
| 5-6 | Agent Client (agent_client.py): unified interface for invoking all 25 upstream agents via REST API | Integration Engineer |

**Milestone: Core engine operational with DAG execution, checkpointing, parallelization, and error recovery (Week 6)**

### Phase 2: Phase Coordinators and Quality Gates (Weeks 7-10)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 7-8 | Information Gathering Coordinator (Feature 2): Phase 1 orchestration, Art. 9 validation, data handoff | Senior Backend Engineer |
| 8-9 | Risk Assessment Coordinator (Feature 3): Phase 2 orchestration, composite risk scoring, Art. 10 mapping | Senior Backend Engineer |
| 9 | Risk Mitigation Coordinator (Feature 4): Phase 3 orchestration, mitigation adequacy, bypass logic | Senior Backend Engineer |
| 9-10 | Quality Gate Engine (Feature 5): 3 gates, configurable thresholds, override, relaxed mode for simplified DD | Senior Backend Engineer |

**Milestone: Full three-phase orchestration with quality gate enforcement (Week 10)**

### Phase 3: Package Generation, API, and Integration (Weeks 11-14)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 11-12 | Due Diligence Package Generator (Feature 9): DDS JSON, PDF report, multi-language, SHA-256 provenance | Senior Backend Engineer + Frontend Engineer |
| 12-13 | REST API layer: 30+ endpoints, authentication, pagination, batch operations | Senior Backend Engineer |
| 13-14 | Integration testing with all 25 upstream agents (mock and live); RBAC permission registration | Integration Engineer |

**Milestone: All 9 P0 features implemented with full API (Week 14)**

### Phase 4: Testing and Launch (Weeks 15-18)

| Week | Deliverable | Owner |
|------|-------------|-------|
| 15-16 | Complete test suite: 1,040+ tests, golden workflow tests for all 7 commodities, chaos tests | Test Engineer |
| 16-17 | Performance testing (1K concurrent workflows), security audit, load testing | DevOps + Security |
| 17 | Database migration V109 finalized and tested | DevOps |
| 17-18 | Beta customer onboarding (5 customers), Grafana dashboard deployment | Product + Engineering |
| 18 | Launch readiness review and go-live | All |

**Milestone: Production launch with all 9 P0 features verified (Week 18)**

### Phase 5: Enhancements (Weeks 19-26)

- Workflow Analytics Dashboard (Feature 10)
- Workflow Template Marketplace (Feature 11)
- Multi-Entity Workflow Coordination (Feature 12)
- WebSocket-based real-time progress streaming
- AI-powered workflow optimization

---

## 14. User Experience

### 14.1 User Flows

#### Flow 1: Standard Due Diligence Execution (Compliance Officer)

```
1. Maria logs in to GL-EUDR-APP
2. Navigates to "Due Diligence" module
3. Clicks "New Due Diligence Workflow"
4. Selects commodity: "Cocoa"
5. Selects products/shipments from portfolio (50 products)
6. System auto-selects "Standard" workflow template for cocoa
7. Maria reviews DAG topology and quality gate thresholds
8. Clicks "Start Workflow"
9. Dashboard shows real-time progress:
   - Phase 1: Information Gathering [=====>     ] 60%
   - Active: EUDR-003 Satellite, EUDR-004 Forest Cover
   - Completed: EUDR-001, EUDR-002, EUDR-006, EUDR-007, EUDR-008
   - ETA: 2 minutes 30 seconds
10. Quality Gate 1 passes (score: 94%)
11. Phase 2: Risk Assessment starts (10 agents in parallel)
12. Quality Gate 2 passes (score: 97%)
13. Phase 3: Risk Mitigation (3 high-risk suppliers identified)
14. Quality Gate 3 passes (residual risk: 12)
15. Package generated: Maria downloads PDF + JSON
16. Maria submits DDS JSON to EU Information System
```

#### Flow 2: Workflow Resume After Failure (Supply Chain Analyst)

```
1. Lukas has a running workflow for 200 wood products
2. EUDR-003 Satellite Monitoring fails (satellite provider timeout)
3. System auto-retries with exponential backoff (1s, 2s, 4s, 8s, 16s)
4. After 5 retries, agent marked as FAILED
5. Workflow checkpointed at last successful state (10 of 15 Phase 1 agents complete)
6. Lukas receives notification: "Workflow paused - EUDR-003 failed after 5 retries"
7. 2 hours later, satellite provider is back online
8. Lukas clicks "Resume Workflow"
9. System loads checkpoint, skips 10 completed agents
10. Resumes from EUDR-003 (and any other incomplete agents)
11. EUDR-003 succeeds; workflow continues to completion
12. Total wall time: 2 hours 4 minutes; actual compute time: 4 minutes
```

#### Flow 3: Batch Quarterly Execution (Operations Manager)

```
1. Stefan uploads quarterly shipment portfolio (1,000 shipments across 3 commodities)
2. System creates batch: 3 workflow groups (cocoa: 400, palm oil: 350, soya: 250)
3. System launches workflows respecting global concurrency limit (100 agents max)
4. Stefan monitors batch dashboard:
   - Cocoa: [========> ] 80% (320/400 shipments processed)
   - Palm Oil: [=====>    ] 55% (193/350 shipments processed)
   - Soya: [==>       ] 30% (75/250 shipments processed)
   - Total ETA: 18 minutes
5. 12 workflows fail quality gate 1 (missing geolocation data)
6. Stefan reviews failures, sends supplier data requests
7. After data received, Stefan clicks "Re-evaluate Gate" for affected workflows
8. All workflows complete; Stefan downloads batch summary report
```

### 14.2 Key Screen Descriptions

**Workflow Dashboard:**
- Summary cards: Active Workflows, Completed Today, Failed/Paused, Average Duration
- Workflow list with status badges, progress bars, commodity tags, and ETA
- Quick actions: Start New, Resume, Cancel, View Package
- Filter/sort by commodity, status, date, operator

**Workflow Execution View:**
- DAG visualization showing agent nodes colored by status (green=complete, blue=running, grey=pending, red=failed)
- Right panel: selected agent details (status, duration, output summary, retry history)
- Bottom timeline: Gantt chart of agent execution with critical path highlighted
- Quality gate indicators between phases (checkmark=passed, X=failed, clock=pending)
- Real-time ETA counter and progress percentage

**Quality Gate Detail View:**
- Overall score with pass/fail indicator and threshold display
- Individual check results with scores and evidence
- Gap identification with remediation guidance
- Override button (for authorized users) with justification text field
- History of previous evaluations

**Package Viewer:**
- Table of contents matching DDS Article 12(2) sections
- Expandable sections with source agent attribution
- Evidence file listing with SHA-256 hash display
- Download buttons: JSON, PDF, HTML, ZIP
- Language selector for multi-language output

---

## 15. Success Criteria

### 15.1 Launch Criteria (Go/No-Go)

- [ ] All 9 P0 features (Features 1-9) implemented and tested
  - [ ] Feature 1: Workflow Definition Engine -- DAG creation, 7 commodity templates, simplified variant
  - [ ] Feature 2: Information Gathering Coordinator -- Phase 1 orchestration, 15-agent coordination
  - [ ] Feature 3: Risk Assessment Coordinator -- Phase 2 orchestration, 10-agent parallel execution
  - [ ] Feature 4: Risk Mitigation Coordinator -- Phase 3 orchestration, adequacy verification
  - [ ] Feature 5: Quality Gate Engine -- 3 gates, configurable thresholds, override capability
  - [ ] Feature 6: Workflow State Manager -- checkpoint, resume, rollback, 5-year audit trail
  - [ ] Feature 7: Parallel Execution Engine -- concurrent execution, critical path, ETA
  - [ ] Feature 8: Error Recovery & Retry Manager -- backoff, circuit breaker, fallback, DLQ
  - [ ] Feature 9: Due Diligence Package Generator -- DDS JSON, PDF, SHA-256, multi-language
- [ ] >= 85% test coverage achieved
- [ ] 1,040+ tests passing (unit, integration, golden, chaos, performance)
- [ ] Security audit passed (JWT + RBAC integrated, encryption verified)
- [ ] Performance targets met (< 5 min standard workflow, 1,000 concurrent workflows)
- [ ] All 7 commodity workflow templates tested with golden fixtures
- [ ] Quality gates validated against manually audited due diligence packages
- [ ] Database migration V109 tested and validated
- [ ] Integration with all 25 upstream EUDR agents verified
- [ ] API documentation complete (OpenAPI 3.1 spec)
- [ ] Grafana dashboard for workflow monitoring deployed
- [ ] 5 beta customers successfully executing due diligence workflows
- [ ] No critical or high-severity bugs in backlog

### 15.2 Post-Launch Metrics (30/60/90 Days)

**30 Days:**
- 100+ due diligence workflows completed by customers
- Average workflow completion time < 5 minutes
- >= 95% workflow completion rate (no unrecoverable failures)
- < 5 support tickets per customer
- Quality gate false-positive rate < 5%

**60 Days:**
- 500+ due diligence workflows completed
- 50+ batch workflow executions
- Average error recovery success rate >= 95%
- Zero DDS packages rejected by EU Information System
- NPS > 40 from compliance officer persona

**90 Days:**
- 2,000+ due diligence workflows completed
- 1,000+ due diligence packages generated
- Average workflow completion time < 4 minutes (optimization improvements)
- 99.9% orchestrator uptime
- NPS > 50

---

## 16. Dependencies

### 16.1 Internal Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EUDR-001 Supply Chain Mapping Master | BUILT (100%) | Low | Stable, production-ready |
| EUDR-002 Geolocation Verification | BUILT (100%) | Low | Stable, production-ready |
| EUDR-003 Satellite Monitoring | BUILT (100%) | Low | Stable, production-ready |
| EUDR-004 Forest Cover Analysis | BUILT (100%) | Low | Stable, production-ready |
| EUDR-005 Land Use Change Detector | BUILT (100%) | Low | Stable, production-ready |
| EUDR-006 Plot Boundary Manager | BUILT (100%) | Low | Stable, production-ready |
| EUDR-007 GPS Coordinate Validator | BUILT (100%) | Low | Stable, production-ready |
| EUDR-008 Multi-Tier Supplier Tracker | BUILT (100%) | Low | Stable, production-ready |
| EUDR-009 Chain of Custody | BUILT (100%) | Low | Stable, production-ready |
| EUDR-010 Segregation Verifier | BUILT (100%) | Low | Stable, production-ready |
| EUDR-011 Mass Balance Calculator | BUILT (100%) | Low | Stable, production-ready |
| EUDR-012 Document Authentication | BUILT (100%) | Low | Stable, production-ready |
| EUDR-013 Blockchain Integration | BUILT (100%) | Low | Stable, production-ready |
| EUDR-014 QR Code Generator | BUILT (100%) | Low | Stable, production-ready |
| EUDR-015 Mobile Data Collector | BUILT (100%) | Low | Stable, production-ready |
| EUDR-016 Country Risk Evaluator | BUILT (100%) | Low | Stable, production-ready |
| EUDR-017 Supplier Risk Scorer | BUILT (100%) | Low | Stable, production-ready |
| EUDR-018 Commodity Risk Analyzer | BUILT (100%) | Low | Stable, production-ready |
| EUDR-019 Corruption Index Monitor | BUILT (100%) | Low | Stable, production-ready |
| EUDR-020 Deforestation Alert System | BUILT (100%) | Low | Stable, production-ready |
| EUDR-021 Indigenous Rights Checker | BUILT (100%) | Low | Stable, production-ready |
| EUDR-022 Protected Area Validator | BUILT (100%) | Low | Stable, production-ready |
| EUDR-023 Legal Compliance Verifier | BUILT (100%) | Low | Stable, production-ready |
| EUDR-024 Third-Party Audit Manager | BUILT (100%) | Low | Stable, production-ready |
| EUDR-025 Risk Mitigation Advisor | BUILT (100%) | Low | Stable, production-ready |
| SEC-001 JWT Authentication | BUILT (100%) | Low | Standard auth integration |
| SEC-002 RBAC Authorization | BUILT (100%) | Low | Standard RBAC integration |
| SEC-003 Encryption at Rest | BUILT (100%) | Low | AES-256-GCM |
| SEC-005 Centralized Audit Logging | BUILT (100%) | Low | Standard logging |
| PostgreSQL + TimescaleDB | Production Ready | Low | Standard infrastructure |
| Redis | Production Ready | Low | Standard cache/streams |
| S3 | Production Ready | Low | Standard object storage |

### 16.2 External Dependencies

| Dependency | Status | Risk | Mitigation |
|------------|--------|------|------------|
| EU Information System DDS schema | Published (v1.x) | Medium | Adapter pattern for schema version changes; schema cached locally |
| EC country benchmarking list | Published; updated periodically | Medium | Database-driven; hot-reloadable |
| Satellite data providers (Sentinel-2, Landsat) | Available | Low | Multi-provider fallback in EUDR-003/020 |
| EUDR implementing regulations (evolving) | Evolving | Medium | Configuration-driven workflow rules; modular quality gate criteria |

---

## 17. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | EU Information System DDS schema changes | Medium | High | Adapter pattern isolates schema mapping; Package Generator can be updated independently of orchestrator |
| R2 | Upstream agent API contract changes | Medium | High | Agent Client abstraction layer; version negotiation; backward-compatible contracts enforced by CI |
| R3 | Workflow complexity exceeds performance targets at scale | Low | High | Lazy loading of agent outputs; S3 offload for large payloads; continuous performance benchmarking |
| R4 | Quality gate thresholds too strict/lenient for real-world data | Medium | Medium | Configurable per-operator thresholds; override capability; threshold tuning based on beta feedback |
| R5 | Circuit breaker cascade: one agent's outage blocks many workflows | Medium | High | Per-agent, per-operator circuit breaker isolation; degraded mode for non-critical agents |
| R6 | Checkpoint data volume grows unbounded | Low | Medium | TimescaleDB compression; S3 tiering for old checkpoints; configurable retention with 5-year minimum |
| R7 | Concurrent workflow overload exhausts database connections | Medium | High | Connection pooling; read replicas; Redis-based rate limiting; HPA auto-scaling |
| R8 | Simplified due diligence misapplied to standard-risk commodities | Low | High | Workflow type validation against country risk classification; audit trail flags workflow type selection |
| R9 | Package generation fails for very large portfolios (10K+ shipments) | Low | Medium | Streaming package generation; chunked PDF rendering; async generation with notification |
| R10 | Regulatory changes to Article 8 due diligence structure | Low | High | Three-phase architecture is modular; phases can be extended/modified via workflow templates |

---

## 18. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **EUDR** | EU Deforestation Regulation (Regulation (EU) 2023/1115) |
| **DDS** | Due Diligence Statement -- formal declaration required by EUDR Article 4 |
| **DAG** | Directed Acyclic Graph -- workflow topology where agents are nodes and dependencies are directed edges |
| **Quality Gate** | Automated validation checkpoint between due diligence phases that enforces completeness and accuracy thresholds |
| **Checkpoint** | Persistent snapshot of workflow state taken after each agent completion, enabling resume on failure |
| **Circuit Breaker** | Resilience pattern that prevents repeated calls to a failing agent, allowing it time to recover |
| **Exponential Backoff** | Retry strategy where delay between retries increases exponentially (1s, 2s, 4s, 8s, ...) |
| **Dead Letter Queue** | Queue for permanently failed agent invocations that require manual investigation |
| **Topological Sort** | Algorithm that orders DAG nodes such that every node comes after its dependencies |
| **Critical Path** | The longest sequential chain of dependent agents in the DAG, determining minimum workflow duration |
| **Work-Stealing** | Load balancing strategy where idle execution threads take tasks from busy threads' queues |
| **Provenance Hash** | SHA-256 hash chain that provides tamper-evident integrity verification for audit trails |
| **Competent Authority** | EU Member State authority responsible for EUDR enforcement (Articles 14-16) |
| **CN Code** | Combined Nomenclature -- EU product classification code |
| **HS Code** | Harmonized System -- international product classification code |

### Appendix B: EUDR Article 8 Due Diligence System Structure

Per Article 8(1), the due diligence system must include:

| Due Diligence Phase | EUDR Article | Orchestrator Phase | Quality Gate |
|--------------------|--------------|--------------------|-------------|
| Information gathering | Article 9 | Phase 1 (Layers 0-4) | QG-1: Information Gathering Completeness |
| Risk assessment | Article 10 | Phase 2 (Layer 5) | QG-2: Risk Assessment Completeness |
| Risk mitigation | Article 11 | Phase 3 (Layer 6) | QG-3: Mitigation Adequacy |
| DDS submission | Article 12 | Phase 4 (Layer 7) | DDS schema validation |

### Appendix C: Agent Execution Layer Summary

| Layer | Phase | Agents | Max Concurrency | Typical Duration | Dependencies |
|-------|-------|--------|-----------------|------------------|-------------|
| 0 | Info Gathering | EUDR-001 | 1 | 30s | None (entry point) |
| 1 | Info Gathering | EUDR-002, 006, 007, 008 | 4 | 30s | Layer 0 |
| 2 | Info Gathering | EUDR-003, 004, 005 | 3 | 30s | Layer 1 |
| 3 | Info Gathering | EUDR-009, 010, 011 | 3 | 30s | Layer 1 |
| 4 | Info Gathering | EUDR-012, 013, 014, 015 | 4 | 30s | Layers 2-3 |
| QG-1 | Gate | Quality Gate 1 | 1 | 5s | Layer 4 |
| 5 | Risk Assessment | EUDR-016-025 | 10 | 30s | QG-1 |
| QG-2 | Gate | Quality Gate 2 | 1 | 5s | Layer 5 |
| 6 | Risk Mitigation | EUDR-025 (mitigation) | 1 | 30s | QG-2 |
| QG-3 | Gate | Quality Gate 3 | 1 | 5s | Layer 6 |
| 7 | Package Gen | Package Generator | 1 | 15s | QG-3 |

### Appendix D: Prometheus Self-Monitoring Metrics (20)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_eudr_ddo_workflows_created_total` | Counter | Workflows created by type and commodity |
| 2 | `gl_eudr_ddo_workflows_completed_total` | Counter | Workflows completed successfully |
| 3 | `gl_eudr_ddo_workflows_failed_total` | Counter | Workflows terminated due to failure |
| 4 | `gl_eudr_ddo_workflows_cancelled_total` | Counter | Workflows cancelled by user |
| 5 | `gl_eudr_ddo_agent_invocations_total` | Counter | Agent invocations by agent ID and status |
| 6 | `gl_eudr_ddo_agent_retries_total` | Counter | Agent retry attempts by agent ID |
| 7 | `gl_eudr_ddo_quality_gates_evaluated_total` | Counter | Quality gate evaluations by gate ID and result |
| 8 | `gl_eudr_ddo_quality_gates_overridden_total` | Counter | Quality gate overrides |
| 9 | `gl_eudr_ddo_checkpoints_created_total` | Counter | Checkpoints created |
| 10 | `gl_eudr_ddo_workflow_resumes_total` | Counter | Workflow resumes from checkpoint |
| 11 | `gl_eudr_ddo_packages_generated_total` | Counter | Due diligence packages generated by type |
| 12 | `gl_eudr_ddo_circuit_breaker_trips_total` | Counter | Circuit breaker state transitions |
| 13 | `gl_eudr_ddo_dlq_entries_total` | Counter | Dead letter queue entries |
| 14 | `gl_eudr_ddo_workflow_duration_seconds` | Histogram | End-to-end workflow duration by type |
| 15 | `gl_eudr_ddo_agent_duration_seconds` | Histogram | Individual agent execution duration by agent ID |
| 16 | `gl_eudr_ddo_gate_evaluation_duration_seconds` | Histogram | Quality gate evaluation duration |
| 17 | `gl_eudr_ddo_package_generation_duration_seconds` | Histogram | Package generation duration |
| 18 | `gl_eudr_ddo_active_workflows` | Gauge | Currently active (running) workflows |
| 19 | `gl_eudr_ddo_active_agents` | Gauge | Currently executing agent invocations |
| 20 | `gl_eudr_ddo_avg_parallelization_efficiency` | Gauge | Average parallelization efficiency across active workflows |

### Appendix E: Commodity-Specific Workflow Variations

| Commodity | Key Agents Emphasized | Special Considerations |
|-----------|----------------------|----------------------|
| **Cattle** | EUDR-001, EUDR-002, EUDR-007, EUDR-021 | Animal movement tracking; pasture rotation GPS trails; indigenous territory overlap in Amazon/Cerrado |
| **Cocoa** | EUDR-001, EUDR-008, EUDR-011, EUDR-021 | Thousands of smallholders per cooperative; mass balance critical; West Africa indigenous rights |
| **Coffee** | EUDR-001, EUDR-002, EUDR-008, EUDR-023 | Altitude/origin segregation; wet/dry mill processing chain; legal compliance (land titles) |
| **Palm Oil** | EUDR-001, EUDR-011, EUDR-020, EUDR-022 | RSPO mass balance; high deforestation risk; protected area proximity; peatland concerns |
| **Rubber** | EUDR-001, EUDR-008, EUDR-010, EUDR-021 | Latex aggregation destroys traceability; smallholder dominance; indigenous territory overlap in SE Asia |
| **Soya** | EUDR-001, EUDR-005, EUDR-016, EUDR-020 | Large volumes, co-mingling at silos; land use change (Cerrado); country risk (Brazil, Argentina) |
| **Wood** | EUDR-001, EUDR-009, EUDR-012, EUDR-023 | Multi-step processing (forest->sawmill->veneer->furniture); species mixing; FSC chain of custody; legality |

### Appendix F: References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 (EU Deforestation Regulation)
2. EU Deforestation Regulation Guidance Document (European Commission)
3. EUDR Technical Specifications for the Information System
4. ISO 19011:2018 -- Guidelines for auditing management systems
5. ISO 31000:2018 -- Risk management -- Guidelines
6. ISO 14001:2015 -- Environmental management systems -- Requirements with guidance for use
7. FSC Chain of Custody Standard (FSC-STD-40-004)
8. RSPO Supply Chain Certification Standard
9. Kahn's Algorithm for Topological Sorting (Kahn, 1962)
10. Circuit Breaker Pattern (Nygard, M. "Release It!", 2007)
11. Exponential Backoff and Jitter (AWS Architecture Blog)
12. Directed Acyclic Graphs for Workflow Orchestration (Apache Airflow documentation)

---

**Approval Signatures:**

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Product Manager | GL-ProductManager | 2026-03-11 | APPROVED |
| Engineering Lead | ___________________ | __________ | __________ |
| EUDR Regulatory Advisor | ___________________ | __________ | __________ |
| CEO | ___________________ | __________ | __________ |

---

**Document History:**

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0-draft | 2026-03-11 | GL-ProductManager | Initial draft created: 9 P0 features, 25-agent DAG orchestration, 3 quality gates (Art. 8/9/10/11), 7 commodity workflow templates, checkpoint/resume, circuit breaker, DDS package generation, V109 migration schema, 1,040+ test targets, 20 Prometheus metrics, 30+ API endpoints |
