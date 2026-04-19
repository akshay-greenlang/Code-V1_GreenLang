# GreenLang Climate OS - Year 1 (2026) Detailed TODO

**Period:** January 2026 - December 2026
**Goal:** Foundation & Regulatory Sprint
**Target:** 500 Agents, 75 Apps, 300 Packs, $40M ARR

---

## Q1 2026: Core Platform & EUDR

### Week 1-2: Foundation Setup

#### Infrastructure
- [ ] Deploy Kubernetes cluster (production)
- [ ] Configure PostgreSQL primary/replica
- [ ] Set up TimescaleDB for time-series
- [ ] Deploy Redis cluster
- [ ] Configure S3 buckets (artifacts, backups)
- [ ] Set up CI/CD pipeline base

#### Agents
- [ ] GL-FOUND-X-001 GreenLang Orchestrator (Start)
- [ ] GL-FOUND-X-002 Schema Compiler (Start)

### Week 3-4: Core Agents

#### Infrastructure
- [ ] Deploy API Gateway
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure alerting
- [ ] Implement feature flags

#### Agents
- [ ] GL-FOUND-X-001 GreenLang Orchestrator (Complete)
- [ ] GL-FOUND-X-002 Schema Compiler (Complete)
- [ ] GL-FOUND-X-003 Unit Normalizer (Start)
- [ ] GL-FOUND-X-004 Assumptions Registry (Start)
- [ ] GL-FOUND-X-007 Agent Registry (Start)

### Week 5-6: Foundation Complete

#### Infrastructure
- [ ] Agent Factory v1.0 deployment
- [ ] Security: JWT auth implementation
- [ ] Security: RBAC layer

#### Agents
- [ ] GL-FOUND-X-003 Unit Normalizer (Complete)
- [ ] GL-FOUND-X-004 Assumptions Registry (Complete)
- [ ] GL-FOUND-X-005 Citations Agent (Start)
- [ ] GL-FOUND-X-006 Policy Guard (Start)
- [ ] GL-FOUND-X-007 Agent Registry (Complete)
- [ ] GL-FOUND-X-009 QA Test Harness (Start)
- [ ] GL-FOUND-X-010 Observability Agent (Start)

### Week 7-8: Data Layer Start

#### Agents (Foundation Complete)
- [ ] GL-FOUND-X-005 Citations Agent (Complete)
- [ ] GL-FOUND-X-006 Policy Guard (Complete)
- [ ] GL-FOUND-X-008 Reproducibility Agent (Start)
- [ ] GL-FOUND-X-009 QA Test Harness (Complete)
- [ ] GL-FOUND-X-010 Observability Agent (Complete)

#### Agents (Data Layer)
- [ ] GL-DATA-X-001 PDF Extractor (Start)
- [ ] GL-DATA-X-002 Excel Normalizer (Start)
- [ ] GL-DATA-X-003 ERP Connector (Start)
- [ ] GL-DATA-X-013 Data Quality Profiler (Start)
- [ ] GL-DATA-REG-004 EUDR Traceability (Start)
- [ ] GL-DATA-GEO-001 GIS Connector (Start)

### Week 9-10: MRV Core

#### Agents (Foundation Final)
- [ ] GL-FOUND-X-008 Reproducibility Agent (Complete)
- [ ] ALL 10 Foundation Agents Complete ✓

#### Agents (Data)
- [ ] GL-DATA-X-001 PDF Extractor (Complete)
- [ ] GL-DATA-X-002 Excel Normalizer (Complete)
- [ ] GL-DATA-X-003 ERP Connector (Complete)
- [ ] GL-DATA-X-013 Data Quality Profiler (Complete)
- [ ] GL-DATA-REG-004 EUDR Traceability (Complete)
- [ ] GL-DATA-GEO-001 GIS Connector (Complete)
- [ ] +8 more data agents

#### Agents (MRV Start)
- [ ] GL-MRV-X-001 Stationary Combustion
- [ ] GL-MRV-X-002 Refrigerants & F-Gas
- [ ] GL-MRV-X-003 Mobile Combustion
- [ ] GL-MRV-X-020 Scope 2 Location-Based
- [ ] GL-MRV-X-021 Scope 2 Market-Based

### Week 11-12: EUDR Launch

#### Agents (MRV Complete)
- [ ] Complete all 8 Scope 1 agents
- [ ] Complete all 5 Scope 2 agents
- [ ] Start 15 Scope 3 category agents

#### Agents (EUDR)
- [ ] Complete 40 EUDR-specific agents

#### Applications
- [ ] GL-EUDR-APP v1.0 Launch
- [ ] GL-CSRD-APP v1.1 Release
- [ ] GL-CBAM-APP v1.1 Release
- [ ] GL-VCCI-APP v1.1 Release

#### Packs
- [ ] 50 solution packs published

---

## Q2 2026: Industrial & SB 253

### Week 13-16: Planning Layer

#### Agents
- [ ] GL-PLAN-X-001 through GL-PLAN-X-015 (Abatement)
- [ ] GL-PLAN-X-020 through GL-PLAN-X-030 (Roadmap)
- [ ] GL-PLAN-IND-001 through GL-PLAN-IND-005 (Industrial)

#### Applications
- [ ] GL-SB253-APP Beta
- [ ] GL-FoodBev-APP Beta

### Week 17-20: Finance Layer

#### Agents
- [ ] GL-FIN-X-001 through GL-FIN-X-012 (Project Finance)
- [ ] GL-FIN-X-060 through GL-FIN-X-067 (Enterprise Finance)
- [ ] GL-SB253-001 through GL-SB253-025 (California)

#### Applications
- [ ] GL-SB253-APP v1.0 Launch
- [ ] GL-FoodBev-APP v1.0 Launch
- [ ] GL-Energy-APP v1.0 Launch

### Week 21-24: Sector Connectors

#### Agents
- [ ] All industrial connectors (GL-DATA-IND-*)
- [ ] All building connectors (GL-DATA-BLD-*)
- [ ] All transport connectors (GL-DATA-TRN-*)
- [ ] All utility connectors (GL-DATA-UTL-*)

#### Applications
- [ ] GL-Boiler-APP v1.0 Launch
- [ ] GL-Enterprise-APP v1.0 Launch
- [ ] +8 Beta applications

#### Packs
- [ ] 50 more packs (100 total)

---

## Q3 2026: Buildings & Transport

### Week 25-30: Building Optimization

#### Agents (50 building agents)
- [ ] GL-PLAN-BLD-001 through GL-PLAN-BLD-005
- [ ] GL-OPS-X-001 through GL-OPS-X-015
- [ ] GL-RISK-X-001 through GL-RISK-X-010
- [ ] Additional data/reporting agents

#### Applications
- [ ] GL-BuildingBPS-APP v1.0 Launch
- [ ] GL-HVAC-APP v1.0 Launch
- [ ] GL-NYC-LL97-APP v1.0 Launch
- [ ] GL-BERDO-APP v1.0 Launch

### Week 31-36: Transport & Fleet

#### Agents (40 transport agents)
- [ ] GL-PLAN-TRN-001 through GL-PLAN-TRN-005
- [ ] GL-OPS-TRN-001 through GL-OPS-TRN-010
- [ ] GL-FIN-TRN-001 through GL-FIN-TRN-010

#### Applications
- [ ] GL-EVFleet-APP v1.0 Launch
- [ ] GL-ChargingInfra-APP v1.0 Launch
- [ ] GL-FleetTCO-APP v1.0 Launch

#### Packs
- [ ] 100 more packs (200 total)

---

## Q4 2026: Supply Chain & Scope 3

### Week 37-42: Supply Chain Agents

#### Agents (60 supply chain agents)
- [ ] Enhanced Scope 3 agents
- [ ] Supplier engagement agents
- [ ] Multi-tier mapping agents

#### Applications
- [ ] GL-Scope3-Enhanced v2.0 Launch
- [ ] GL-SupplierPortal-APP v1.0 Launch
- [ ] GL-Hotspot-APP v1.0 Launch

### Week 43-48: Risk, Policy, DevTools

#### Agents
- [ ] GL-RISK-X-020 through GL-RISK-X-054 (40 agents)
- [ ] GL-POL-X-001 through GL-POL-X-034 (30 agents)
- [ ] GL-DEV-X-001 through GL-DEV-X-017 (17 agents)

#### Applications
- [ ] GL-ClimateRisk-APP Beta
- [ ] All remaining apps to 75 total

#### Packs
- [ ] 100 more packs (300 total)

#### Infrastructure
- [ ] SOC 2 Type II certified
- [ ] ISO 27001 started
- [ ] GreenLang Hub marketplace beta

---

## Year 1 Summary Checkpoints

### Q1 Exit Criteria (March 31, 2026)
- [ ] 100 agents in production
- [ ] 10 applications live
- [ ] 50 solution packs
- [ ] 30 customers
- [ ] $5M ARR
- [ ] Agent Factory v1.0 deployed

### Q2 Exit Criteria (June 30, 2026)
- [ ] 200 agents in production
- [ ] 25 applications live
- [ ] 100 solution packs
- [ ] 75 customers
- [ ] $12M ARR
- [ ] Multi-region deployed (US+EU)

### Q3 Exit Criteria (September 30, 2026)
- [ ] 350 agents in production
- [ ] 50 applications live
- [ ] 200 solution packs
- [ ] 150 customers
- [ ] $25M ARR
- [ ] SOC 2 Type II certified

### Q4 Exit Criteria (December 31, 2026)
- [ ] 500 agents in production
- [ ] 75 applications live
- [ ] 300 solution packs
- [ ] 250 customers
- [ ] $40M ARR
- [ ] GreenLang Hub beta live

---

## Team Scaling Year 1

| Role | Q1 | Q2 | Q3 | Q4 |
|------|-----|-----|-----|-----|
| Engineering | 25 | 40 | 55 | 70 |
| Product | 5 | 7 | 9 | 12 |
| Sales | 3 | 5 | 7 | 10 |
| Customer Success | 2 | 4 | 6 | 8 |
| **Total** | **35** | **56** | **77** | **100** |

---

*End of Year 1 (2026) TODO*
