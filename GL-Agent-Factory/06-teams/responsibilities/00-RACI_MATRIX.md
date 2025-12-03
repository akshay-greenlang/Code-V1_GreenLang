# Agent Factory: RACI Matrix

**Version:** 1.0
**Date:** 2025-12-03
**Program:** Agent Factory

---

## RACI Legend

- **R (Responsible):** Does the work to complete the task
- **A (Accountable):** Ultimately answerable for completion and has decision authority (only one "A" per deliverable)
- **C (Consulted):** Provides input and expertise (two-way communication)
- **I (Informed):** Kept up-to-date on progress (one-way communication)

---

## Team Abbreviations

| Abbreviation | Team |
|--------------|------|
| **ML** | ML Platform Team |
| **AI** | AI/Agent Team |
| **CS** | Climate Science & Policy Team |
| **PL** | Platform/Development Team |
| **DE** | Data Engineering Team |
| **DO** | DevOps/SRE/Security Team |
| **PM** | Product Manager |
| **EL** | Engineering Lead |

---

## Phase 1: Foundation (Weeks 1-16)

### 1.1 Model Infrastructure

| Deliverable | ML | AI | CS | PL | DE | DO | PM | EL |
|-------------|----|----|----|----|----|----|----|----|
| **Model Registry** | R,A | C | I | C | I | I | I | C |
| **Model Serving API** | R,A | C | I | C | I | C | I | C |
| **Model Versioning** | R,A | I | I | C | I | I | I | C |
| **Evaluation Harness** | R,A | C | C | I | I | I | I | C |
| **Golden Test Suite (100 tests)** | R | C | R,A | I | I | I | C | C |
| **Determinism Validator** | R,A | C | C | I | I | I | I | C |
| **Model Telemetry** | R,A | I | I | C | I | R | I | C |
| **Grafana Dashboard (Model Health)** | R | I | I | I | I | R,A | I | C |

**Accountable:** ML Platform Team (Model infrastructure)

---

### 1.2 Agent Factory

| Deliverable | ML | AI | CS | PL | DE | DO | PM | EL |
|-------------|----|----|----|----|----|----|----|----|
| **AgentSpec Schema v1.0** | C | R,A | C | C | C | I | C | C |
| **AgentSpec Validator** | C | R,A | C | C | C | I | C | C |
| **Code Generator (MVP)** | C | R,A | I | C | I | I | C | C |
| **Test Generator** | C | R,A | C | I | I | I | C | C |
| **Documentation Generator** | C | R,A | I | C | I | I | C | C |
| **Quality Validator** | C | R,A | C | I | I | R | C | C |
| **Agent Package Builder** | C | R,A | I | C | I | R | C | C |

**Accountable:** AI/Agent Team (Agent Factory)

---

### 1.3 Agent SDK

| Deliverable | ML | AI | CS | PL | DE | DO | PM | EL |
|-------------|----|----|----|----|----|----|----|----|
| **SDK Core Library** | I | R,A | I | R | C | I | C | C |
| **Authentication Module** | I | C | I | R,A | I | C | C | C |
| **Logging Framework** | I | C | I | R,A | I | C | C | C |
| **Error Handling** | I | C | I | R,A | I | C | C | C |
| **Data Validation Framework** | I | R | C | R,A | R | I | C | C |
| **Provenance Tracking** | I | R | C | R,A | R | I | C | C |

**Accountable:** Platform Team (SDK core), AI/Agent Team (Agent runtime)

---

### 1.4 Validation Hooks

| Deliverable | ML | AI | CS | PL | DE | DO | PM | EL |
|-------------|----|----|----|----|----|----|----|----|
| **CBAM Validation Hooks** | I | C | R,A | I | C | I | C | C |
| **Emission Factor Database** | C | I | R,A | I | C | I | C | C |
| **CN Code Validator** | I | C | R,A | I | C | I | C | C |
| **Validation SDK** | I | C | R,A | C | I | I | C | C |
| **Certification Framework** | I | C | R,A | I | I | I | C | C |
| **CBAM Certification Criteria** | I | C | R,A | I | I | I | C | C |

**Accountable:** Climate Science Team (Domain validation)

---

### 1.5 Data Contracts

| Deliverable | ML | AI | CS | PL | DE | DO | PM | EL |
|-------------|----|----|----|----|----|----|----|----|
| **CBAM Data Contracts** | I | C | C | C | R,A | I | C | C |
| **Contract Validator** | I | C | C | C | R,A | I | C | C |
| **Ingestion Pipelines** | I | C | I | C | R,A | C | C | C |
| **Transformation Pipelines** | I | C | C | C | R,A | C | C | C |
| **Data Quality Framework** | I | C | C | C | R,A | I | C | C |

**Accountable:** Data Engineering Team (Data contracts & pipelines)

---

### 1.6 Agent Registry

| Deliverable | ML | AI | CS | PL | DE | DO | PM | EL |
|-------------|----|----|----|----|----|----|----|----|
| **Registry Database (PostgreSQL)** | I | C | I | R,A | C | R | C | C |
| **Registry API** | I | C | I | R,A | I | C | C | C |
| **Agent Storage (S3)** | I | C | I | R,A | I | R | C | C |
| **Version Management** | I | C | I | R,A | I | C | C | C |
| **CLI Tools** | I | C | I | R,A | I | C | C | C |
| **API Gateway** | I | C | I | R,A | I | R | C | C |

**Accountable:** Platform Team (Registry & CLI)

---

### 1.7 Infrastructure & DevOps

| Deliverable | ML | AI | CS | PL | DE | DO | PM | EL |
|-------------|----|----|----|----|----|----|----|----|
| **Kubernetes Cluster** | I | I | I | C | C | R,A | I | C |
| **CI/CD Pipelines** | C | C | C | C | C | R,A | I | C |
| **Terraform Infrastructure** | C | C | I | C | C | R,A | I | C |
| **Helm Charts** | C | C | I | C | C | R,A | I | C |
| **Prometheus + Grafana** | C | C | I | C | C | R,A | I | C |
| **ELK Stack (Logging)** | C | C | I | C | C | R,A | I | C |
| **Security Scanning** | C | C | C | C | C | R,A | I | C |
| **RBAC Configuration** | C | C | I | C | C | R,A | I | C |

**Accountable:** DevOps/SRE Team (Infrastructure & operations)

---

## Phase 2: Production Scale (Weeks 17-28)

### 2.1 Advanced Agent Generation

| Deliverable | ML | AI | CS | PL | DE | DO | PM | EL |
|-------------|----|----|----|----|----|----|----|----|
| **Multi-Language Support** | C | R,A | C | I | I | I | C | C |
| **Advanced Templates** | C | R,A | C | I | I | I | C | C |
| **LLM Fine-Tuning** | R,A | C | I | I | C | I | C | C |
| **Model A/B Testing** | R,A | C | I | I | I | I | C | C |
| **Performance Optimization** | R | R,A | I | C | I | C | C | C |

**Accountable:** AI/Agent Team (Generation), ML Platform Team (Models)

---

### 2.2 Multi-Regulation Validation

| Deliverable | ML | AI | CS | PL | DE | DO | PM | EL |
|-------------|----|----|----|----|----|----|----|----|
| **EUDR Validation Hooks** | I | C | R,A | I | C | I | C | C |
| **CSRD Validation Hooks** | I | C | R,A | I | C | I | C | C |
| **Golden Test Suite (2,000 tests)** | R | C | R,A | I | I | I | C | C |
| **Automated Test Generation** | C | C | R,A | I | I | I | C | C |

**Accountable:** Climate Science Team (Validation)

---

### 2.3 ERP Integrations

| Deliverable | ML | AI | CS | PL | DE | DO | PM | EL |
|-------------|----|----|----|----|----|----|----|----|
| **SAP Connector** | I | C | I | C | R,A | C | C | C |
| **Oracle Connector** | I | C | I | C | R,A | C | C | C |
| **Workday Connector** | I | C | I | C | R,A | C | C | C |
| **Real-Time Streaming (Kafka)** | I | C | I | C | R,A | R | C | C |

**Accountable:** Data Engineering Team (ERP integrations)

---

### 2.4 Registry Enhancements

| Deliverable | ML | AI | CS | PL | DE | DO | PM | EL |
|-------------|----|----|----|----|----|----|----|----|
| **Registry UI (Web Dashboard)** | I | C | I | R,A | I | C | C | C |
| **Full-Text Search** | I | C | I | R,A | I | C | C | C |
| **Analytics Dashboard** | I | C | I | R,A | C | C | C | C |
| **API Versioning (v2)** | C | C | I | R,A | I | C | C | C |

**Accountable:** Platform Team (Registry)

---

### 2.5 GitOps & Multi-Region

| Deliverable | ML | AI | CS | PL | DE | DO | PM | EL |
|-------------|----|----|----|----|----|----|----|----|
| **ArgoCD Deployment** | C | C | I | C | C | R,A | I | C |
| **Multi-Region Setup (3 regions)** | C | C | I | C | C | R,A | I | C |
| **Regional Failover** | C | C | I | C | C | R,A | I | C |
| **Disaster Recovery** | C | C | I | C | C | R,A | I | C |

**Accountable:** DevOps/SRE Team (Infrastructure)

---

## Phase 3: Enterprise Ready (Weeks 29-40)

### 3.1 Enterprise Features

| Deliverable | ML | AI | CS | PL | DE | DO | PM | EL |
|-------------|----|----|----|----|----|----|----|----|
| **Multi-Tenancy** | C | C | I | R,A | C | R | C | C |
| **RBAC for Agent Generation** | C | C | I | R,A | I | R | C | C |
| **Audit Logging** | C | C | C | R,A | C | R | C | C |
| **Developer Portal** | I | C | I | R,A | I | C | C | C |

**Accountable:** Platform Team (Multi-tenancy), DevOps Team (RBAC)

---

### 3.2 Compliance & Security

| Deliverable | ML | AI | CS | PL | DE | DO | PM | EL |
|-------------|----|----|----|----|----|----|----|----|
| **SOC 2 Type II Certification** | C | C | C | C | C | R,A | C | C |
| **ISO 27001 Compliance** | C | C | C | C | C | R,A | C | C |
| **Penetration Testing** | C | C | C | C | C | R,A | C | C |
| **Third-Party Auditor Interface** | I | C | R,A | I | I | C | C | C |

**Accountable:** DevOps/SRE Team (Compliance), Climate Science Team (Auditor interface)

---

### 3.3 Advanced Deployment

| Deliverable | ML | AI | CS | PL | DE | DO | PM | EL |
|-------------|----|----|----|----|----|----|----|----|
| **Canary Deployments** | C | C | I | C | I | R,A | I | C |
| **Progressive Delivery (Flagger)** | C | C | I | C | I | R,A | I | C |
| **Chaos Engineering** | C | C | I | C | I | R,A | I | C |
| **Cost Optimization** | C | C | I | C | C | R,A | C | C |

**Accountable:** DevOps/SRE Team (Deployment & optimization)

---

### 3.4 Scale & Optimization

| Deliverable | ML | AI | CS | PL | DE | DO | PM | EL |
|-------------|----|----|----|----|----|----|----|----|
| **Batch Agent Generation (100+ agents)** | C | R,A | C | C | I | C | C | C |
| **Automated Re-Certification** | C | C | R,A | I | I | I | C | C |
| **Data Warehouse (Snowflake/BigQuery)** | I | I | I | C | R,A | C | C | C |
| **Lineage Visualization** | I | I | C | C | R,A | I | C | C |

**Accountable:** AI/Agent Team (Batch generation), Climate Science Team (Re-certification), Data Engineering Team (Data warehouse)

---

## Cross-Cutting Responsibilities

### Program Management

| Activity | ML | AI | CS | PL | DE | DO | PM | EL |
|----------|----|----|----|----|----|----|----|----|
| **Product Roadmap** | C | C | C | C | C | C | R,A | I |
| **Sprint Planning** | C | C | C | C | C | C | R,A | C |
| **Stakeholder Communication** | I | I | I | I | I | I | R,A | C |
| **Risk Management** | C | C | C | C | C | C | R,A | R,A |

**Accountable:** Product Manager (Roadmap), Engineering Lead (Risk)

---

### Architecture & Design

| Activity | ML | AI | CS | PL | DE | DO | PM | EL |
|----------|----|----|----|----|----|----|----|----|
| **System Architecture** | C | C | C | C | C | C | I | R,A |
| **API Contracts** | R | R | R | R | R | C | I | A |
| **Data Schemas** | C | C | C | C | R,A | I | I | C |
| **Security Architecture** | C | C | C | C | C | R,A | I | C |

**Accountable:** Engineering Lead (System architecture), Data Engineering Team (Data schemas), DevOps Team (Security architecture)

---

### Quality & Testing

| Activity | ML | AI | CS | PL | DE | DO | PM | EL |
|----------|----|----|----|----|----|----|----|----|
| **Test Strategy** | C | C | C | C | C | C | C | R,A |
| **Golden Tests** | R | C | R | I | I | I | I | C |
| **Integration Tests** | R | R | R | R | R | R | I | C |
| **Performance Tests** | R | R | I | C | C | R | I | C |
| **Security Tests** | C | C | C | C | C | R,A | I | C |

**Accountable:** Engineering Lead (Test strategy), ML Platform + Climate Science (Golden tests), DevOps Team (Security tests)

---

### Documentation

| Activity | ML | AI | CS | PL | DE | DO | PM | EL |
|----------|----|----|----|----|----|----|----|----|
| **API Documentation** | R | R | R | R | R | R | I | C |
| **User Guides** | C | R,A | C | C | I | I | C | I |
| **Architecture Docs (ADRs)** | C | C | C | C | C | C | I | R,A |
| **Runbooks (Operations)** | C | C | C | C | C | R,A | I | C |

**Accountable:** AI/Agent Team (User guides), Engineering Lead (ADRs), DevOps Team (Runbooks)

---

### Incident Management

| Activity | ML | AI | CS | PL | DE | DO | PM | EL |
|----------|----|----|----|----|----|----|----|----|
| **On-Call Rotation** | R | R | R | R | R | R | I | I |
| **Incident Response** | R | R | R | R | R | R,A | I | C |
| **Postmortems** | C | C | C | C | C | R,A | I | C |
| **Action Item Tracking** | R | R | R | R | R | R | C | A |

**Accountable:** DevOps Team (Incident response), Engineering Lead (Action items)

---

## Decision Rights

### Strategic Decisions

| Decision Type | Recommender | Approver | Informed |
|--------------|-------------|----------|----------|
| **Product Roadmap** | PM + EL | PM | All Teams |
| **Architecture Changes** | EL + Tech Leads | EL | All Teams |
| **Budget Allocation** | PM + EL | PM | All Teams |
| **Hiring Priorities** | Team Leads | EL + PM | All Teams |

---

### Tactical Decisions

| Decision Type | Recommender | Approver | Informed |
|--------------|-------------|----------|----------|
| **Sprint Scope** | Team Leads | Team Leads | PM |
| **Technology Stack** | Team Leads | EL | PM, Other Teams |
| **API Contracts** | Team Leads | EL | All Teams |
| **Release Timing** | DO Team | PM + EL | All Teams |

---

### Operational Decisions

| Decision Type | Recommender | Approver | Informed |
|--------------|-------------|----------|----------|
| **Deployment Go/No-Go** | DO Team | DO Team | PM, EL |
| **Incident Escalation** | On-Call Engineer | DO Lead | PM, EL |
| **Emergency Rollback** | DO Team | DO Team | PM, EL |
| **Resource Scaling** | DO Team | DO Team | PM, EL |

---

## Escalation Matrix

### Level 1: Team-Level (Response: <1 hour)
- **Contact:** Team Lead
- **Scope:** Team-specific issues, blockers
- **Examples:** Code review delays, CI failures, test failures

### Level 2: Cross-Team (Response: <4 hours)
- **Contact:** Tech Lead Council (all tech leads)
- **Scope:** Cross-team dependencies, integration issues
- **Examples:** API contract conflicts, data schema mismatches

### Level 3: Architecture (Response: <1 day)
- **Contact:** Engineering Lead
- **Scope:** Architectural changes, major technical decisions
- **Examples:** Technology stack changes, system design changes

### Level 4: Executive (Response: <1 day)
- **Contact:** Product Manager + Engineering Lead
- **Scope:** Scope changes, budget overruns, timeline risks
- **Examples:** Major scope additions, resource constraints

---

## Change Management

### Major Changes (Require RACI Review)
- New teams added to program
- Major scope additions/deletions
- Organizational restructuring
- Budget reallocation

### Process for RACI Updates
1. Propose change in Tech Lead Council meeting
2. Review with all affected teams (5-day review period)
3. Engineering Lead approves
4. Product Manager updates RACI matrix
5. Communicate changes to all teams

---

## Appendix: RACI Summary by Team

### ML Platform Team
- **Accountable for:** Model infrastructure, evaluation harness, model telemetry
- **Responsible for:** Model registry, serving API, golden tests (technical), fine-tuning
- **Consulted on:** Agent generation, validation hooks, SDK, infrastructure

### AI/Agent Team
- **Accountable for:** Agent Factory, AgentSpec, agent generation
- **Responsible for:** Code generator, SDK (agent runtime), templates
- **Consulted on:** Model infrastructure, data contracts, validation

### Climate Science Team
- **Accountable for:** Validation hooks, certification, golden tests (domain)
- **Responsible for:** Emission factors, regulatory compliance, auditor interface
- **Consulted on:** Agent generation, data quality, testing

### Platform Team
- **Accountable for:** Agent registry, SDK core, CLI tools, API gateway
- **Responsible for:** Authentication, logging, registry database/API
- **Consulted on:** Agent generation, data pipelines, infrastructure

### Data Engineering Team
- **Accountable for:** Data contracts, pipelines, data quality
- **Responsible for:** Ingestion, transformation, ERP connectors, data warehouse
- **Consulted on:** Agent SDK, validation, storage

### DevOps/SRE Team
- **Accountable for:** Infrastructure, CI/CD, security, observability
- **Responsible for:** Kubernetes, deployments, monitoring, compliance
- **Consulted on:** All technical deliverables (infrastructure needs)

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-03 | GL Product Manager | Initial RACI matrix |

---

**Approvals:**

- Engineering Lead: ___________________
- Product Manager: ___________________
- All Tech Leads: ___________________
