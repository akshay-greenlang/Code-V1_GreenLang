# GreenLang Agent Factory: Phased Implementation Roadmap

**Version:** 1.0
**Date:** 2025-12-03
**Product Manager:** GL-ProductManager
**Status:** Active

---

## Executive Summary

The GreenLang Agent Factory is a multi-phase initiative to transform how climate compliance agents are built, validated, and deployed. This roadmap outlines the journey from alignment (Phase 0) through self-service ecosystem enablement (Phase 4).

**Program Vision:** Move from "write agents by hand" to "generate from spec" - enabling GreenLang to scale from 15 agents to 100+ certified agents with 20x faster development velocity.

**Total Timeline:** 40-44 weeks (excluding Phase 4)
**Investment:** 23-29 FTEs across 6 teams

---

## Phase Overview

| Phase | Name | Duration | Dates | Core Goal |
|-------|------|----------|-------|-----------|
| **Phase 0** | Alignment & Scoping | 2 weeks | Dec 3-17, 2025 | Lock scope, roles, first wedge use cases |
| **Phase 1** | Agent SDK v1 | 10 weeks | Dec 18, 2025 - Feb 28, 2026 | Unify how agents are represented and executed |
| **Phase 2** | Factory Core | 12 weeks | Mar 1 - May 23, 2026 | Move from "write by hand" to "generate from spec" |
| **Phase 3** | Registry & Runtime | 12 weeks | May 24 - Aug 15, 2026 | Treat agents as first-class, governed assets |
| **Phase 4** | Agent Studio (Optional) | 16 weeks | Aug 16 - Dec 5, 2026 | Self-service and ecosystem enablement |

---

## Timeline Visualization

```
2025                           2026
Dec        Jan        Feb        Mar        Apr        May        Jun        Jul        Aug        Sep        Oct        Nov        Dec
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
[P0]       [===== Phase 1: Agent SDK v1 =====]
                                 [======== Phase 2: Factory Core ========]
                                                                         [======== Phase 3: Registry & Runtime ========]
                                                                                                                        [===== Phase 4: Agent Studio (Optional) =====]
           M1        M2         M3         M4         M5         M6         M7         M8         M9         M10        M11        M12
```

**Milestones:**
- **M1 (Week 2):** Phase 0 Complete - Scope locked, teams aligned
- **M2 (Week 6):** AgentSpec v1 finalized, base classes complete
- **M3 (Week 12):** Agent SDK v1 released, first agent migrated
- **M4 (Week 16):** Generator MVP functional, generates first agent
- **M5 (Week 20):** Evaluation framework operational, certification running
- **M6 (Week 24):** Factory Core complete, 10 agents generated
- **M7 (Week 28):** Agent Registry launched, first agents registered
- **M8 (Week 32):** Runtime governance operational
- **M9 (Week 36):** Registry & Runtime complete, 50 agents deployed
- **M10 (Week 40):** Agent Studio MVP (if Phase 4 approved)
- **M11 (Week 44):** Partner enablement live
- **M12 (Week 48):** Full ecosystem operational, 100+ agents

---

## Phase Details Summary

### Phase 0: Alignment & Scoping (Now - Week 2)

**Goal:** Lock scope, assign team ownership, define first wedge use cases.

**Key Deliverables:**
- Approved AgentSpec v1 requirements
- Team RACI matrix finalized
- First 2-3 flagship use cases defined
- Success metrics baseline established
- Risk register initialized

**Exit Criteria:**
- All team leads have reviewed and approved scope
- Budget and headcount confirmed
- First sprint planned

---

### Phase 1: Agent SDK v1 (Weeks 3-12)

**Goal:** Unify how agents are represented, validated, and executed.

**Key Deliverables:**
- AgentSpec v1 schema (YAML/JSON)
- BaseAgent class and runtime
- Validation engine with Pydantic models
- Agent graph DSL (LangGraph compatible)
- First 3 existing agents migrated

**Team Ownership:**
| Component | Owner | Support |
|-----------|-------|---------|
| AgentSpec schema | AI/Agent Team | Climate Science |
| BaseAgent class | AI/Agent Team | ML Platform |
| Validation engine | ML Platform Team | AI/Agent |
| Agent graph DSL | AI/Agent Team | Platform |
| Migration | AI/Agent Team | All teams |

**Exit Criteria to Phase 2:**
- AgentSpec v1 schema finalized and documented
- BaseAgent class passing all tests (85%+ coverage)
- 3+ existing agents successfully migrated
- Validation engine catching 95%+ of schema errors
- Performance baseline: agent load time <100ms

---

### Phase 2: Factory Core (Weeks 13-24)

**Goal:** Move from "write agents by hand" to "generate from spec."

**Key Deliverables:**
- AgentSpec v2 (with generation metadata)
- Agent Generator (spec-to-agent code)
- Evaluation framework with certification
- Golden test suite (100+ tests)
- 10 agents generated from spec

**Team Ownership:**
| Component | Owner | Support |
|-----------|-------|---------|
| AgentSpec v2 | AI/Agent Team | All |
| Agent Generator | AI/Agent Team | ML Platform |
| Evaluation framework | ML Platform Team | Climate Science |
| Golden tests | Climate Science Team | AI/Agent |
| Certification | Climate Science Team | DevOps |

**Exit Criteria to Phase 3:**
- Generator produces valid, executable agents
- Certification pass rate >90% on first attempt
- 10+ agents generated and certified
- Generation time <2 hours per agent
- Documentation complete for generator usage

---

### Phase 3: Registry & Runtime Governance (Weeks 25-36)

**Goal:** Treat agents as first-class, governed assets with lifecycle management.

**Key Deliverables:**
- Agent Registry with versioning
- Lifecycle management (CRUD + deployment)
- Runtime governance engine
- Observability and monitoring
- 50 agents deployed via registry

**Team Ownership:**
| Component | Owner | Support |
|-----------|-------|---------|
| Agent Registry API | Platform Team | DevOps |
| Lifecycle management | Platform Team | AI/Agent |
| Governance engine | DevOps/Security Team | Climate Science |
| Observability | DevOps/SRE Team | ML Platform |
| Deployment | DevOps Team | Platform |

**Exit Criteria to Phase 4:**
- Registry serving 50+ agents
- All agents discoverable via registry API
- Governance policies enforced (version control, approval workflow)
- 99.9% uptime for registry service
- Full audit trail for all agent changes

---

### Phase 4: Agent Studio (Weeks 37-52) - OPTIONAL

**Goal:** Enable self-service agent creation and ecosystem participation.

**Key Deliverables:**
- Web-based Agent Studio UI
- Visual agent builder
- Partner onboarding portal
- Marketplace foundation
- Third-party agent submission workflow

**Team Ownership:**
| Component | Owner | Support |
|-----------|-------|---------|
| Agent Studio UI | Platform Team | AI/Agent |
| Visual builder | AI/Agent Team | Platform |
| Partner portal | Platform Team | DevOps |
| Marketplace | Platform Team | All |

**Strategic Note:** Phase 4 is optional but strategic for ecosystem growth. Decision to proceed based on Phase 3 success and market demand.

---

## Dependencies Between Phases

### Sequential Dependencies (Must Complete Before Next Phase)

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4
   │            │            │            │            │
   │            │            │            │            └─► Agent Studio UI
   │            │            │            │                 └─► Requires: Registry, Governance
   │            │            │            │
   │            │            │            └─► Registry & Runtime
   │            │            │                 └─► Requires: Evaluation, Generator
   │            │            │
   │            │            └─► Factory Core (Generator, Evaluation)
   │            │                 └─► Requires: SDK, AgentSpec v1
   │            │
   │            └─► Agent SDK v1
   │                 └─► Requires: Scope locked, Use cases defined
   │
   └─► Alignment & Scoping
        └─► Requires: Team commitment, Budget approval
```

### Parallel Workstreams Within Phases

**Phase 1 Parallel Tracks:**
- Track A: AgentSpec v1 schema design
- Track B: BaseAgent class implementation
- Track C: Validation engine development

**Phase 2 Parallel Tracks:**
- Track A: AgentSpec v2 finalization
- Track B: Generator development
- Track C: Evaluation framework
- Track D: Golden test creation

**Phase 3 Parallel Tracks:**
- Track A: Registry development
- Track B: Governance engine
- Track C: Observability integration

---

## Success Criteria by Phase

### Phase 0 Success Criteria

| Criteria | Target | Measurement |
|----------|--------|-------------|
| Scope document signed off | 100% | All team leads approve |
| Use cases defined | 2-3 | Documented in 01-INITIAL_USE_CASES.md |
| Team RACI complete | 100% | Matrix published |
| Baseline metrics captured | 100% | Current state documented |

### Phase 1 Success Criteria

| Criteria | Target | Measurement |
|----------|--------|-------------|
| AgentSpec v1 coverage | 100% | All required fields defined |
| BaseAgent test coverage | 85%+ | pytest --cov report |
| Agents migrated | 3+ | Successfully running on new SDK |
| Validation error detection | 95%+ | Schema errors caught |
| Agent load time | <100ms | Performance benchmark |

### Phase 2 Success Criteria

| Criteria | Target | Measurement |
|----------|--------|-------------|
| Generator success rate | 90%+ | % of specs that generate valid code |
| Certification pass rate | 90%+ | First-attempt pass rate |
| Agents generated | 10+ | From AgentSpec v2 |
| Generation time | <2 hours | Per agent, end-to-end |
| Golden test coverage | 100+ tests | Climate Science approved |

### Phase 3 Success Criteria

| Criteria | Target | Measurement |
|----------|--------|-------------|
| Registry uptime | 99.9% | Monitoring dashboard |
| Agents registered | 50+ | Registry database |
| Governance compliance | 100% | All agents pass policy checks |
| Deployment automation | 100% | No manual deployments |
| Audit completeness | 100% | Full trail for all changes |

### Phase 4 Success Criteria (if approved)

| Criteria | Target | Measurement |
|----------|--------|-------------|
| Studio users | 100+ | Active monthly users |
| Agents created via Studio | 20+ | Self-service created |
| Partner submissions | 5+ | Third-party agents |
| Time to first agent | <1 hour | New user onboarding |

---

## Investment by Phase

### Headcount Allocation

| Phase | Duration | Total FTE-Weeks | Primary Teams |
|-------|----------|-----------------|---------------|
| Phase 0 | 2 weeks | 50 | All teams (partial) |
| Phase 1 | 10 weeks | 250 | AI/Agent (5), ML Platform (4), Platform (3) |
| Phase 2 | 12 weeks | 300 | AI/Agent (5), ML Platform (4), Climate Science (3), DevOps (3) |
| Phase 3 | 12 weeks | 320 | Platform (5), DevOps (5), AI/Agent (3), Data (3) |
| Phase 4 | 16 weeks | 400 | Platform (5), AI/Agent (4), DevOps (3), Data (2) |

### Budget Estimates (USD)

| Phase | Engineering | Infrastructure | Total |
|-------|-------------|----------------|-------|
| Phase 0 | $75K | $0 | $75K |
| Phase 1 | $625K | $50K | $675K |
| Phase 2 | $750K | $100K | $850K |
| Phase 3 | $800K | $150K | $950K |
| Phase 4 | $1,000K | $200K | $1,200K |
| **Total** | **$3,250K** | **$500K** | **$3,750K** |

*Note: Phase 4 budget contingent on approval*

---

## Risk Summary

### Phase-Specific Risks

| Phase | Key Risk | Mitigation |
|-------|----------|------------|
| Phase 0 | Scope creep before lock | Strict 2-week timebox; PM owns scope |
| Phase 1 | SDK breaks existing agents | Migration testing; phased rollout |
| Phase 2 | Generator quality issues | Extensive golden testing; human review |
| Phase 3 | Registry performance | Load testing; horizontal scaling |
| Phase 4 | Low adoption | User research; iterate on UX |

### Program-Level Risks

| Risk | Likelihood | Impact | Owner | Mitigation |
|------|------------|--------|-------|------------|
| Regulatory changes | Medium | High | Climate Science | Modular architecture; monitoring |
| Team attrition | Medium | High | All | Documentation; knowledge sharing |
| Integration complexity | High | Medium | Engineering Lead | Clear contracts; weekly syncs |
| Model hallucination | Low | Critical | ML Platform | Zero-hallucination design |

---

## Governance & Reviews

### Phase Gate Reviews

| Gate | Date | Reviewers | Decision |
|------|------|-----------|----------|
| Phase 0 Exit | Week 2 | All leads + PM | Go/No-Go to Phase 1 |
| Phase 1 Exit | Week 12 | All leads + PM | Go/No-Go to Phase 2 |
| Phase 2 Exit | Week 24 | All leads + PM + Exec | Go/No-Go to Phase 3 |
| Phase 3 Exit | Week 36 | All leads + PM + Exec | Go/No-Go to Phase 4 |
| Phase 4 Approval | Week 34 | Exec + Board | Approve/Defer Phase 4 |

### Monthly Executive Updates

- **Audience:** CEO, VP Engineering, VP Product
- **Cadence:** First Monday of each month
- **Content:** Phase progress, metrics, risks, budget

---

## Document References

- `phase-0/00-PHASE_0_ALIGNMENT.md` - Phase 0 detailed plan
- `phase-0/01-INITIAL_USE_CASES.md` - First flagship use cases
- `phase-1/00-PHASE_1_AGENT_SDK.md` - Phase 1 detailed plan
- `phase-2/00-PHASE_2_FACTORY_CORE.md` - Phase 2 detailed plan
- `phase-3/00-PHASE_3_REGISTRY_RUNTIME.md` - Phase 3 detailed plan
- `phase-4/00-PHASE_4_AGENT_STUDIO.md` - Phase 4 detailed plan
- `00-DEPENDENCIES_CRITICAL_PATH.md` - Critical path analysis

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-03 | GL-ProductManager | Initial phased roadmap |

---

**Approvals:**

- Product Manager: ___________________
- Engineering Lead: ___________________
- VP Engineering: ___________________
- CEO: ___________________
