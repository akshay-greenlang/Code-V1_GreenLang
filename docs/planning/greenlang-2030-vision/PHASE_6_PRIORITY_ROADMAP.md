# Phase 6: Priority Implementation Roadmap

**Document:** Go-to-Market Readiness Implementation Plan
**Version:** 1.0
**Date:** December 5, 2025
**Product Manager:** GL-ProductManager

---

## Executive Summary

This document provides a prioritized implementation roadmap for completing Phase 6 of the Process Heat Agent commercialization. Based on the gap analysis, the true completion is 40% (not 0%), with 14 tasks remaining.

### Current Status

| Category | Complete | Partial | Not Started | Total |
|----------|----------|---------|-------------|-------|
| 6.1 Product Specifications | 4 | 2 | 4 | 10 |
| 6.2 Pricing & Business Model | 3 | 3 | 4 | 10 |
| 6.3 Sales Enablement | 4 | 2 | 4 | 10 |
| **TOTAL** | **11** | **7** | **12** | **30** |

### Revenue Target Alignment

| Year | Target ARR | Required Sales | Readiness Impact |
|------|------------|----------------|------------------|
| Y1 2026 | $45M | ~200 customers | Critical - need sales enablement |
| Y2 2027 | $165M | ~600 customers | Demo environments essential |
| Y3 2028 | $405M | ~1,200 customers | Partner enablement required |

---

## Priority 1: Launch Blockers (Weeks 1-4)

These tasks must be completed before commercial launch.

### TASK-215: Module Specifications (8 Modules)

**Owner:** Product Manager
**Effort:** Large (2 weeks)
**Dependencies:** None

**Deliverables:**
1. `GL-003_STEAM_ANALYTICS_SPEC.md`
2. `GL-007_FURNACE_PERFORMANCE_SPEC.md`
3. `GL-011_FUEL_OPTIMIZATION_SPEC.md`
4. `GL-013_PREDICTIVE_MAINTENANCE_SPEC.md`
5. `GL-014_HEAT_EXCHANGER_SPEC.md`
6. `GL-015_INSULATION_ANALYSIS_SPEC.md`
7. `GL-019_LOAD_SCHEDULING_SPEC.md`
8. `GL-020_ECONOMIZER_SPEC.md`

**Template:** Use `MODULE_SPECIFICATION_TEMPLATE.md`

**Week 1:**
- Complete GL-003, GL-007, GL-011, GL-013 (ThermalCommand modules)

**Week 2:**
- Complete GL-014, GL-015 (WasteHeatRecovery modules)
- Complete GL-019, GL-020 (remaining modules)

---

### TASK-228: Usage Metering Infrastructure

**Owner:** Engineering (Backend)
**Effort:** Large (3 weeks)
**Dependencies:** Infrastructure team

**Scope:**
```yaml
usage_metering:
  dimensions:
    - assets_monitored
    - data_points_ingested
    - api_calls
    - reports_generated
    - ml_inferences
    - storage_gb

  implementation:
    collector: OpenTelemetry
    aggregation: TimescaleDB
    billing_api: REST endpoints
    real_time: Kafka streams

  deliverables:
    - Usage collection service
    - Aggregation pipeline
    - Usage API endpoints
    - Usage dashboard
    - Billing export
```

**Week 1:** Design and architecture review
**Week 2:** Core implementation
**Week 3:** Testing and integration

---

### TASK-229: Billing Integration

**Owner:** Engineering (Backend) + Finance
**Effort:** Large (2 weeks)
**Dependencies:** TASK-228 (Usage Metering)

**Scope:**
```yaml
billing_integration:
  platform: Stripe Billing (recommended)

  features:
    - subscription_management
    - usage_based_billing
    - invoice_generation
    - payment_processing
    - revenue_recognition
    - multi_currency

  integrations:
    - salesforce_opportunities
    - netsuite_erp
    - usage_metering_api

  deliverables:
    - Stripe account configuration
    - Product/price catalog
    - Subscription workflows
    - Invoice templates
    - Payment webhooks
    - Admin portal
```

**Week 1:** Platform setup and configuration
**Week 2:** Integration testing and documentation

---

### TASK-234: Demo Environment

**Owner:** Solution Architecture + DevOps
**Effort:** Large (3 weeks)
**Dependencies:** Data Engineering for synthetic data

**Deliverables:**
1. Demo platform infrastructure (Kubernetes)
2. Sandbox provisioning service
3. 5 industry data scenarios
4. Real-time simulation engine
5. Demo portal (self-service)
6. Guided tour framework

**Reference:** See `DEMO_ENVIRONMENT_ARCHITECTURE.md`

**Week 1:** Infrastructure and base platform
**Week 2:** Data scenarios and simulation
**Week 3:** Portal and tours

---

### TASK-230: Contract Templates

**Owner:** Legal + Sales Operations
**Effort:** Medium (1 week)
**Dependencies:** TASK-220 (Licensing Framework)

**Deliverables:**
1. Master Services Agreement (MSA)
2. SaaS Subscription Agreement
3. Enterprise License Agreement (ELA)
4. Data Processing Agreement (DPA)
5. Service Level Agreement (SLA)
6. Statement of Work (SOW) template
7. Order Form template

**Week 1:**
- Days 1-2: Draft templates
- Days 3-4: Legal review
- Day 5: Finalization

---

### TASK-220: Licensing Framework

**Owner:** Product + Legal
**Effort:** Medium (1 week)
**Dependencies:** TASK-228 for usage definitions

**Deliverables:**

```markdown
LICENSING_FRAMEWORK.md:

1. License Types
   - SaaS Subscription (primary)
   - On-Premise Perpetual (enterprise)
   - Term License (annual)
   - Usage-Based (API)

2. Entitlements by Edition
   - Standard: Core features, limited assets
   - Professional: Full features, moderate scale
   - Enterprise: Unlimited, custom SLA

3. Module Licensing
   - Add-on model
   - Bundle discounts
   - Upgrade paths

4. Usage Limits
   - Asset limits
   - User limits
   - API rate limits
   - Storage limits

5. Special Programs
   - Academic: 50% discount
   - Government: Custom terms
   - Startups: Reduced pricing
   - Partners: Margin structures
```

---

## Priority 2: Sales Readiness (Weeks 3-6)

These tasks enable effective sales execution.

### TASK-216: Consolidated Feature Matrix

**Owner:** Product Manager
**Effort:** Small (3 days)
**Dependencies:** TASK-215 (Module Specs)

**Deliverable:** `FEATURE_MATRIX.md`

```markdown
| Feature | TC Std | TC Pro | TC Ent | BO Std | BO Pro | BO Ent | WHR Std | WHR Pro | WHR Ent | EG Std | EG Pro | EG Ent |
|---------|--------|--------|--------|--------|--------|--------|---------|---------|---------|--------|--------|--------|
| Real-time monitoring | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| AI analytics | - | Yes | Yes | - | Yes | Yes | - | Yes | Yes | - | Yes | Yes |
| Predictive alerts | - | Yes | Yes | - | Yes | Yes | - | Yes | Yes | - | Yes | Yes |
| ...
```

---

### TASK-217: Comparison Charts

**Owner:** Product Marketing
**Effort:** Small (3 days)
**Dependencies:** TASK-216 (Feature Matrix)

**Deliverables:**
1. Edition comparison chart (visual)
2. Product comparison chart (vs. competitors)
3. TCO comparison chart (vs. manual/alternatives)
4. Module value chart (ROI by module)

---

### TASK-226: ROI Calculators (3 products)

**Owner:** Product Manager + Technical Writer
**Effort:** Medium (1 week)
**Dependencies:** None

**Deliverables:**
1. `BOILEROPTIMIZER_ROI_CALCULATOR.md` - Expand existing
2. `WASTEHEATRECOVERY_ROI_CALCULATOR.md` - Expand existing
3. `EMISSIONSGUARDIAN_ROI_CALCULATOR.md` - Expand existing

**Template:** Use ThermalCommand ROI Calculator as reference (507 lines)

---

### TASK-227: TCO Calculators

**Owner:** Product Manager
**Effort:** Medium (1 week)
**Dependencies:** TASK-226

**Deliverables:**
1. Standalone TCO Calculator methodology
2. TCO vs. Status Quo comparison
3. TCO vs. Competitors comparison
4. Interactive TCO tool specifications

---

### TASK-232: Solution Briefs (5 industries)

**Owner:** Product Marketing
**Effort:** Medium (1 week)
**Dependencies:** None

**Deliverables:**
1. `SOLUTION_BRIEF_OIL_GAS.md`
2. `SOLUTION_BRIEF_CHEMICALS.md`
3. `SOLUTION_BRIEF_STEEL.md`
4. `SOLUTION_BRIEF_FOOD_BEVERAGE.md`
5. `SOLUTION_BRIEF_CEMENT.md`

**Structure:**
- Industry challenges (1 page)
- GreenLang solution (1 page)
- Customer results (1 page)
- Getting started (1 page)

---

### TASK-233: Competitive Battlecards (validate/complete)

**Owner:** Product Marketing + Sales
**Effort:** Medium (1 week)
**Dependencies:** Competitor research

**Deliverables:**
1. Validate ThermalCommand battlecard
2. Complete BoilerOptimizer battlecard (vs. Honeywell, Emerson)
3. Complete WasteHeatRecovery battlecard (vs. Schneider, ABB)
4. Complete EmissionsGuardian battlecard (vs. Sphera, Enablon, Persefoni)

---

### TASK-239: Sales Playbooks

**Owner:** Sales Operations + Product
**Effort:** Medium (1 week)
**Dependencies:** TASK-232, TASK-233

**Deliverables:**
1. `THERMALCOMMAND_SALES_PLAYBOOK.md`
2. `BOILEROPTIMIZER_SALES_PLAYBOOK.md`
3. `WASTEHEATRECOVERY_SALES_PLAYBOOK.md`
4. `EMISSIONSGUARDIAN_SALES_PLAYBOOK.md`

---

## Priority 3: Scale Enablement (Weeks 5-8)

These tasks enable growth and partnerships.

### TASK-219: Packaging Guidelines

**Owner:** Product Manager
**Effort:** Small (3 days)
**Dependencies:** TASK-215

**Deliverable:** `PACKAGING_GUIDELINES.md`

---

### TASK-223: Performance-Based Pricing (all products)

**Owner:** Product + Finance
**Effort:** Medium (1 week)
**Dependencies:** None

**Deliverable:** Extend gain-share model to all 4 products

---

### TASK-224: Cross-Product Bundle Pricing

**Owner:** Product + Finance
**Effort:** Small (3 days)
**Dependencies:** TASK-219

**Deliverables:**
1. ThermalCommand + EmissionsGuardian bundle
2. BoilerOptimizer + WasteHeatRecovery bundle
3. Full platform bundle (all 4 products)

---

### TASK-235: EmissionsGuardian Sales Collateral

**Owner:** Product Marketing
**Effort:** Small (2 days)
**Dependencies:** None

**Deliverables:**
1. `EMISSIONSGUARDIAN_DEMO_SCRIPT.md`
2. `EMISSIONSGUARDIAN_CASE_STUDY_TEMPLATE.md`

---

### TASK-236: Proposal Templates

**Owner:** Sales Operations
**Effort:** Medium (1 week)
**Dependencies:** TASK-230

**Deliverables:**
1. Executive proposal template
2. Technical proposal template
3. Commercial proposal template
4. Implementation plan template

---

### TASK-237: Interactive Pricing Calculator

**Owner:** Engineering (Frontend)
**Effort:** Medium (2 weeks)
**Dependencies:** Pricing data

**Scope:**
- Web-based configurator
- Real-time price calculation
- PDF export
- CRM integration

---

### TASK-238: Objection Handling Guides

**Owner:** Sales Operations
**Effort:** Small (3 days)
**Dependencies:** TASK-233

**Deliverable:** `OBJECTION_HANDLING_GUIDE.md`

---

### TASK-240: Partner Enablement Kit

**Owner:** Partner Management
**Effort:** Large (2 weeks)
**Dependencies:** All above

**Deliverables:**
1. Partner program overview
2. Partner onboarding guide
3. Partner technical training
4. Partner sales training
5. Deal registration process
6. Partner pricing/margins
7. Co-branded templates
8. Partner success metrics

---

## Implementation Timeline

```
PHASE 6 IMPLEMENTATION GANTT:

WEEK 1   WEEK 2   WEEK 3   WEEK 4   WEEK 5   WEEK 6   WEEK 7   WEEK 8
|--------|--------|--------|--------|--------|--------|--------|--------|

PRIORITY 1 (Launch Blockers):
TASK-215 Module Specs      [========]
TASK-228 Usage Metering    [================]
TASK-229 Billing           .........[=======]
TASK-234 Demo Environment  [========================]
TASK-230 Contract Templates.........[====]
TASK-220 Licensing         .........[====]

PRIORITY 2 (Sales Readiness):
TASK-216 Feature Matrix    .........[==]
TASK-217 Comparison Charts ...........[==]
TASK-226 ROI Calculators   .........[======]
TASK-227 TCO Calculators   ...............[======]
TASK-232 Solution Briefs   .........[======]
TASK-233 Battlecards       ...............[======]
TASK-239 Sales Playbooks   ..................[======]

PRIORITY 3 (Scale Enablement):
TASK-219 Packaging         ...............[==]
TASK-223 Perf-Based Pricing................[====]
TASK-224 Bundle Pricing    ..................[==]
TASK-235 EG Collateral     ...............[==]
TASK-236 Proposal Templates................[====]
TASK-237 Pricing Calculator................[========]
TASK-238 Objection Handling................[==]
TASK-240 Partner Kit       ........................[========]

MILESTONES:
[M1] Week 2: Module specs complete
[M2] Week 4: Core infrastructure ready (metering, billing, demo)
[M3] Week 6: Sales team enabled
[M4] Week 8: Partner program ready
```

---

## Resource Requirements

### Team Allocation

| Role | Phase 6 Allocation | Primary Tasks |
|------|-------------------|---------------|
| Product Manager | 100% | Specs, pricing, roadmap |
| Technical Writer | 50% | ROI/TCO calculators, docs |
| Product Marketing | 75% | Briefs, battlecards, charts |
| Sales Operations | 50% | Playbooks, templates |
| Solution Architect | 50% | Demo environment |
| Backend Engineering | 100% | Metering, billing |
| Frontend Engineering | 25% | Pricing calculator |
| DevOps | 50% | Demo infrastructure |
| Legal | 25% | Contracts, licensing |
| Partner Management | 25% | Partner enablement |

### External Dependencies

| Dependency | Owner | Risk | Mitigation |
|------------|-------|------|------------|
| Stripe setup | Finance | Low | Start immediately |
| Legal review | Legal | Medium | Time-box to 1 week |
| Demo data | Data Engineering | Medium | Parallel with infrastructure |
| Sales feedback | Sales | Low | Involve in development |

---

## Success Criteria

### Phase 6 Completion Checklist

**6.1 Product Specifications:**
- [ ] All 4 core product specs updated with 95+ improvements
- [ ] All 8 module specifications complete
- [ ] Feature matrix consolidated
- [ ] Comparison charts created
- [ ] Edition tiers standardized
- [ ] Packaging guidelines documented
- [ ] Licensing framework approved

**6.2 Pricing & Business Model:**
- [ ] All pricing tiers finalized
- [ ] Module pricing reconciled with Master Plan
- [ ] Performance-based pricing extended
- [ ] Bundle discounts defined
- [ ] Implementation services priced
- [ ] ROI calculators complete (all 4 products)
- [ ] TCO calculators complete
- [ ] Usage metering operational
- [ ] Billing integration live
- [ ] Contract templates approved

**6.3 Sales Enablement:**
- [ ] All datasheets current
- [ ] Solution briefs for 5 industries
- [ ] Battlecards for all 4 products
- [ ] Demo environment operational
- [ ] Case study templates complete
- [ ] Proposal templates ready
- [ ] Pricing calculator functional
- [ ] Objection handling guide complete
- [ ] Sales playbooks for all 4 products
- [ ] Partner enablement kit ready

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Demo environment delays | Medium | High | Start Week 1, parallel track |
| Billing complexity | Medium | High | Use proven platform (Stripe) |
| Legal review delays | Medium | Medium | Engage Legal Week 1 |
| Sales team availability | Low | Medium | Schedule reviews in advance |
| Resource conflicts | Medium | Medium | Dedicated Phase 6 team |

---

## Appendix: Task Assignments

| Task ID | Task Name | Owner | Start | End | Status |
|---------|-----------|-------|-------|-----|--------|
| TASK-215 | Module Specifications | PM | W1D1 | W2D5 | Not Started |
| TASK-216 | Feature Matrix | PM | W3D1 | W3D3 | Not Started |
| TASK-217 | Comparison Charts | PMM | W3D3 | W3D5 | Not Started |
| TASK-218 | Edition Tiers | PM | - | - | Complete |
| TASK-219 | Packaging Guidelines | PM | W5D1 | W5D3 | Not Started |
| TASK-220 | Licensing Framework | PM/Legal | W3D1 | W3D5 | Not Started |
| TASK-221 | SaaS Pricing | PM | - | - | Complete |
| TASK-222 | Module Pricing | PM | - | - | Complete (needs reconcile) |
| TASK-223 | Performance Pricing | PM/Fin | W5D1 | W5D5 | Partial |
| TASK-224 | Bundle Discounts | PM/Fin | W6D1 | W6D3 | Partial |
| TASK-225 | Services Pricing | PM | - | - | Complete |
| TASK-226 | ROI Calculators | PM/TW | W3D1 | W4D5 | Partial (1/4) |
| TASK-227 | TCO Calculators | PM | W5D1 | W5D5 | Not Started |
| TASK-228 | Usage Metering | Eng | W1D1 | W3D5 | Not Started |
| TASK-229 | Billing Integration | Eng | W3D1 | W4D5 | Not Started |
| TASK-230 | Contract Templates | Legal | W3D1 | W3D5 | Not Started |
| TASK-231 | Product Datasheets | PMM | - | - | Complete |
| TASK-232 | Solution Briefs | PMM | W3D1 | W4D5 | Not Started |
| TASK-233 | Battlecards | PMM | W5D1 | W5D5 | Partial (needs validation) |
| TASK-234 | Demo Environment | SA/DevOps | W1D1 | W4D5 | Not Started |
| TASK-235 | Case Study Templates | PMM | W5D1 | W5D3 | Partial (3/4) |
| TASK-236 | Proposal Templates | SalesOps | W5D1 | W5D5 | Not Started |
| TASK-237 | Pricing Calculator | FE Eng | W5D1 | W6D5 | Not Started |
| TASK-238 | Objection Handling | SalesOps | W6D1 | W6D3 | Not Started |
| TASK-239 | Sales Playbooks | SalesOps | W6D1 | W7D5 | Not Started |
| TASK-240 | Partner Enablement | Partner | W7D1 | W8D5 | Not Started |

**Legend:**
- PM = Product Manager
- PMM = Product Marketing
- TW = Technical Writer
- Eng = Engineering
- SA = Solution Architect
- FE = Frontend
- Fin = Finance

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-05 | GL-ProductManager | Initial roadmap |

---

**END OF DOCUMENT**
