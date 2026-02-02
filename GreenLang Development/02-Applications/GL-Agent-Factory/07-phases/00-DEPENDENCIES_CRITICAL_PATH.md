# GreenLang Agent Factory: Dependencies and Critical Path Analysis

**Version:** 1.0
**Date:** 2025-12-03
**Product Manager:** GL-ProductManager
**Status:** Active

---

## Executive Summary

This document analyzes the dependencies between phases, components, and teams in the Agent Factory program. It identifies the critical path - the sequence of tasks that determines the minimum project duration - and provides risk mitigation strategies for potential bottlenecks.

**Key Finding:** The critical path runs through AgentSpec v1 -> BaseAgent -> Generator -> Registry, with Climate Science certification as a parallel critical dependency.

---

## Phase Dependencies Overview

### Phase Dependency Chain

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                         Phase Dependencies                                     │
│                                                                               │
│   Phase 0         Phase 1           Phase 2           Phase 3        Phase 4  │
│   Alignment       Agent SDK         Factory Core      Registry       Studio   │
│                                                                      (Opt)    │
│   ┌───────┐       ┌───────┐         ┌───────┐        ┌───────┐     ┌───────┐ │
│   │ Scope │       │AgentSpec│       │AgentSpec│       │Registry│    │ Studio│ │
│   │ Lock  │──────►│  v1    │──────► │  v2    │──────► │  API   │───►│  UI   │ │
│   └───────┘       └───────┘         └───────┘        └───────┘     └───────┘ │
│       │               │                 │                │             │      │
│       │               ▼                 ▼                ▼             ▼      │
│       │           ┌───────┐         ┌───────┐        ┌───────┐     ┌───────┐ │
│       │           │BaseAgent│        │Generator│       │Lifecycle│   │Builder│ │
│       └──────────►│ Class  │──────► │        │──────► │Mgmt    │───►│       │ │
│                   └───────┘         └───────┘        └───────┘     └───────┘ │
│                       │                 │                │             │      │
│                       ▼                 ▼                ▼             ▼      │
│                   ┌───────┐         ┌───────┐        ┌───────┐     ┌───────┐ │
│                   │Validatn│         │Evaluatn│        │Governce│    │Partner│ │
│                   │Engine  │──────► │Framewrk│──────► │Engine  │───►│Portal │ │
│                   └───────┘         └───────┘        └───────┘     └───────┘ │
│                                         │                │                    │
│                                         ▼                ▼                    │
│                                     ┌───────┐        ┌───────┐               │
│                                     │Certifi-│        │Observ- │               │
│                                     │cation  │──────► │ability │               │
│                                     └───────┘        └───────┘               │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Sequential vs. Parallel Dependencies

| Dependency Type | Examples |
|-----------------|----------|
| **Sequential (Blocking)** | AgentSpec v1 must complete before BaseAgent |
| **Parallel (Independent)** | Validation Engine can be built alongside BaseAgent |
| **Soft Dependencies** | Documentation can start before code complete |

---

## Critical Path Analysis

### Critical Path Definition

The critical path is the longest sequence of dependent tasks that determines the minimum project duration. Any delay in critical path tasks delays the entire project.

### Critical Path Identification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CRITICAL PATH                                      │
│                                                                             │
│  Week 0        Week 2      Week 6       Week 12      Week 20      Week 36   │
│    │             │           │            │            │            │        │
│    ▼             ▼           ▼            ▼            ▼            ▼        │
│  ┌─────┐     ┌───────┐   ┌───────┐    ┌───────┐   ┌───────┐   ┌───────┐   │
│  │Scope│────►│AgentSpec──►│BaseAgent──►│Generator──►│10 Agents──►│50 Agents │
│  │Lock │     │  v1   │   │ Class │    │       │   │Generated│  │Registered│ │
│  └─────┘     └───────┘   └───────┘    └───────┘   └───────┘   └───────┘   │
│    2 wks       4 wks      6 wks        8 wks       8 wks        16 wks     │
│                                                                             │
│  Total Critical Path Duration: 44 weeks                                     │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│                    PARALLEL CRITICAL PATH (Certification)                    │
│                                                                             │
│            Week 8      Week 16       Week 20      Week 28      Week 36     │
│              │            │            │            │            │          │
│              ▼            ▼            ▼            ▼            ▼          │
│           ┌───────┐   ┌───────┐    ┌───────┐   ┌───────┐   ┌───────┐      │
│           │Domain │──►│Golden │───►│Evaluate│──►│Certify│──►│Certify│      │
│           │Rules  │   │ Tests │    │10 Agents│  │10     │   │50     │      │
│           └───────┘   └───────┘    └───────┘   └───────┘   └───────┘      │
│              8 wks      8 wks        4 wks       8 wks       8 wks         │
│                                                                             │
│  Parallel Path Duration: 36 weeks (starts Week 8)                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Critical Path Tasks

| Task | Duration | Dependencies | Float | Critical? |
|------|----------|--------------|-------|-----------|
| Scope Lock | 2 weeks | None | 0 | YES |
| AgentSpec v1 | 4 weeks | Scope Lock | 0 | YES |
| BaseAgent Class | 6 weeks | AgentSpec v1 | 0 | YES |
| Validation Engine | 6 weeks | AgentSpec v1 | 2 weeks | No |
| Agent Generator | 8 weeks | BaseAgent Class | 0 | YES |
| Evaluation Framework | 6 weeks | Validation Engine | 2 weeks | No |
| Generate 10 Agents | 8 weeks | Generator | 0 | YES |
| Certify 10 Agents | 8 weeks | Evaluation + Generate | 0 | YES |
| Registry API | 6 weeks | Generator | 4 weeks | No |
| Deploy 50 Agents | 16 weeks | Registry + Certify 10 | 0 | YES |

**Total Critical Path:** 44 weeks
**Float on Non-Critical:** 2-4 weeks

---

## Component Dependencies

### Phase 1: Agent SDK Dependencies

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Phase 1 Dependencies                             │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     AgentSpec v1                              │  │
│  │                         │                                     │  │
│  │          ┌──────────────┼──────────────┐                      │  │
│  │          │              │              │                      │  │
│  │          ▼              ▼              ▼                      │  │
│  │    ┌──────────┐  ┌──────────┐  ┌──────────┐                  │  │
│  │    │ BaseAgent│  │Validation│  │Agent Graph│                  │  │
│  │    │   Class  │  │  Engine  │  │   DSL    │                  │  │
│  │    └────┬─────┘  └────┬─────┘  └────┬─────┘                  │  │
│  │         │             │             │                         │  │
│  │         └─────────────┴─────────────┘                         │  │
│  │                       │                                       │  │
│  │                       ▼                                       │  │
│  │              ┌──────────────┐                                 │  │
│  │              │Agent Migration│                                 │  │
│  │              │  (3 agents)  │                                 │  │
│  │              └──────────────┘                                 │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Dependency Matrix:                                                 │
│  ┌────────────────┬─────────┬─────────┬─────────┬──────────────┐   │
│  │ Component      │AgentSpec│BaseAgent│Valid Eng│Agent Graph   │   │
│  ├────────────────┼─────────┼─────────┼─────────┼──────────────┤   │
│  │ AgentSpec v1   │    -    │         │         │              │   │
│  │ BaseAgent Class│   X     │    -    │         │              │   │
│  │ Validation Eng │   X     │         │    -    │              │   │
│  │ Agent Graph DSL│   X     │         │         │      -       │   │
│  │ Migration      │   X     │    X    │    X    │      X       │   │
│  └────────────────┴─────────┴─────────┴─────────┴──────────────┘   │
│                                                                     │
│  X = Depends on                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 2: Factory Core Dependencies

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Phase 2 Dependencies                             │
│                                                                     │
│  External (Phase 1):                                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ AgentSpec v1 + BaseAgent + Validation Engine                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     AgentSpec v2                              │  │
│  │                         │                                     │  │
│  │          ┌──────────────┴──────────────┐                      │  │
│  │          │                             │                      │  │
│  │          ▼                             ▼                      │  │
│  │    ┌──────────┐                 ┌──────────┐                  │  │
│  │    │ Agent    │                 │ Evaluation│                  │  │
│  │    │Generator │                 │ Framework │                  │  │
│  │    └────┬─────┘                 └────┬─────┘                  │  │
│  │         │                            │                        │  │
│  │         │                            │                        │  │
│  │         └────────────┬───────────────┘                        │  │
│  │                      │                                        │  │
│  │                      ▼                                        │  │
│  │              ┌──────────────┐                                 │  │
│  │              │   Golden     │                                 │  │
│  │              │  Test Suite  │                                 │  │
│  │              └──────┬───────┘                                 │  │
│  │                     │                                         │  │
│  │                     ▼                                         │  │
│  │              ┌──────────────┐                                 │  │
│  │              │ Certification│                                 │  │
│  │              │   Pipeline   │                                 │  │
│  │              └──────────────┘                                 │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Parallel Streams:                                                  │
│  Stream A: AgentSpec v2 -> Generator -> Generate Agents             │
│  Stream B: Evaluation Framework -> Golden Tests -> Certification    │
│  Sync Point: Agent certification requires both streams              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 3: Registry Dependencies

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Phase 3 Dependencies                             │
│                                                                     │
│  External (Phase 2):                                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Generator + Evaluation + Certification (10 certified agents)  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│          ┌───────────────────┴───────────────────┐                  │
│          │                                       │                  │
│          ▼                                       ▼                  │
│    ┌──────────┐                          ┌──────────┐               │
│    │ Registry │                          │Governance│               │
│    │   API    │                          │  Engine  │               │
│    └────┬─────┘                          └────┬─────┘               │
│         │                                     │                     │
│         │       ┌─────────────────────────────┘                     │
│         │       │                                                   │
│         ▼       ▼                                                   │
│    ┌──────────────┐                                                 │
│    │  Lifecycle   │                                                 │
│    │  Management  │                                                 │
│    └──────┬───────┘                                                 │
│           │                                                         │
│           │                                                         │
│           ▼                                                         │
│    ┌──────────────┐                                                 │
│    │ Observability│                                                 │
│    │Infrastructure│                                                 │
│    └──────┬───────┘                                                 │
│           │                                                         │
│           ▼                                                         │
│    ┌──────────────┐                                                 │
│    │Deploy 50 Agts│                                                 │
│    └──────────────┘                                                 │
│                                                                     │
│  Parallel Streams:                                                  │
│  Stream A: Registry API -> Lifecycle -> Deploy                      │
│  Stream B: Governance Engine -> Lifecycle                           │
│  Stream C: Observability (can start with Registry)                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Team Dependencies

### Inter-Team Dependency Matrix

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       Inter-Team Dependencies                                │
│                                                                              │
│              ┌─────────┬─────────┬─────────┬─────────┬─────────┬──────────┐ │
│              │AI/Agent │ML Platfm│Climate  │Platform │Data Eng │DevOps    │ │
│  ────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼──────────┤ │
│  AI/Agent    │    -    │ Receive │ Receive │ Provide │         │          │ │
│              │         │ models  │ rules   │ SDK     │         │          │ │
│  ────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼──────────┤ │
│  ML Platform │ Provide │    -    │ Receive │         │         │ Receive  │ │
│              │ models  │         │ metrics │         │         │ infra    │ │
│  ────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼──────────┤ │
│  Climate Sci │ Provide │ Provide │    -    │         │         │          │ │
│              │ rules   │ metrics │         │         │         │          │ │
│  ────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼──────────┤ │
│  Platform    │ Receive │         │         │    -    │ Receive │ Receive  │ │
│              │ SDK     │         │         │         │ data    │ deploy   │ │
│  ────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼──────────┤ │
│  Data Eng    │         │         │         │ Provide │    -    │ Receive  │ │
│              │         │         │         │ data    │         │ pipelines│ │
│  ────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼──────────┤ │
│  DevOps      │         │ Provide │         │ Provide │ Provide │    -     │ │
│              │         │ infra   │         │ deploy  │pipelines│          │ │
│  ────────────┴─────────┴─────────┴─────────┴─────────┴─────────┴──────────┘ │
│                                                                              │
│  Key Dependency Flows:                                                       │
│  1. AI/Agent ←→ Climate Science: Domain rules and validation                │
│  2. AI/Agent ←→ ML Platform: Model invocation interface                     │
│  3. Platform ←→ DevOps: Deployment and infrastructure                       │
│  4. Data Eng → All: Data pipelines and quality                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Critical Team Handoffs

| Handoff | From | To | Phase | Critical? | Risk |
|---------|------|----|----- |-----------|------|
| Model Interface | ML Platform | AI/Agent | 1 | YES | Interface changes break agents |
| Domain Rules | Climate Science | ML Platform | 1 | YES | Rules not ready delays validation |
| AgentSpec v1 | AI/Agent | All | 1 | YES | Schema changes ripple everywhere |
| Evaluation API | ML Platform | AI/Agent | 2 | YES | Integration complexity |
| Golden Tests | Climate Science | ML Platform | 2 | YES | Test creation bottleneck |
| Registry API | Platform | AI/Agent | 3 | NO | Can use mock initially |
| Governance API | DevOps | Platform | 3 | NO | Can add later |

---

## Blocking Dependencies

### Absolute Blockers (Cannot Proceed Without)

| Blocker | Blocks | Phase | Mitigation |
|---------|--------|-------|------------|
| AgentSpec v1 not finalized | All Phase 1 development | 1 | Timebox design to 2 weeks |
| BaseAgent not working | Generator development | 2 | Parallel spike on generator |
| No domain rules | Validation Engine | 1 | Start with CBAM only |
| Generator produces invalid code | Agent generation | 2 | Manual code review fallback |
| No golden tests | Certification | 2 | Prioritize test creation |
| Registry API not ready | Agent deployment | 3 | File-based fallback |

### Soft Blockers (Can Workaround)

| Blocker | Impact | Workaround |
|---------|--------|------------|
| Elasticsearch not ready | No search | PostgreSQL full-text search |
| Grafana not configured | No dashboards | Prometheus raw metrics |
| LangGraph integration issues | Graph DSL broken | Custom graph engine |
| Partner portal delayed | No external partners | Internal-only launch |

---

## Risk Mitigation Strategies

### Strategy 1: Parallel Spikes

Run parallel technical spikes to de-risk critical path items:

```
Week 1-2: Parallel Spikes
├── Spike A: AgentSpec v1 schema validation
├── Spike B: LangGraph integration feasibility
├── Spike C: Generator template approach
└── Spike D: Registry performance with 100 agents

Result: Identify risks early, adjust timeline if needed
```

### Strategy 2: Interface Contracts First

Define interfaces before implementation:

```
Week 1-4: Interface Definition
├── AgentSpec v1 schema (locked by Week 4)
├── BaseAgent interface (locked by Week 4)
├── Validation API (locked by Week 4)
├── Generator API (locked by Week 8)
└── Registry API (locked by Week 16)

Benefit: Teams can develop in parallel against stable contracts
```

### Strategy 3: Feature Flags and Progressive Rollout

Enable incomplete features behind flags:

```python
class FeatureFlags:
    ENABLE_NEW_VALIDATION_ENGINE = False
    ENABLE_GRAPH_DSL = False
    ENABLE_REGISTRY_SEARCH = False
    ENABLE_GOVERNANCE_POLICIES = False

# Usage
if FeatureFlags.ENABLE_NEW_VALIDATION_ENGINE:
    result = new_validation_engine.validate(agent)
else:
    result = legacy_validation.validate(agent)
```

### Strategy 4: Buffer Time on Critical Path

Add buffer at phase gates:

| Phase Gate | Planned | Buffer | Total |
|------------|---------|--------|-------|
| Phase 0 Exit | Week 2 | 0 | Week 2 |
| Phase 1 Exit | Week 12 | +2 | Week 14 |
| Phase 2 Exit | Week 24 | +2 | Week 26 |
| Phase 3 Exit | Week 36 | +2 | Week 38 |

### Strategy 5: Parallel Certification Track

Run certification in parallel with development:

```
Development Track:     ──────────────────────────────►
                      Generator      |     Generate Agents
                                     |
Certification Track:  ────────────────────────────────►
                      Golden Tests   |     Certify as generated
                                     |
                                     ▼
                              Continuous certification
                              (no end-of-phase bottleneck)
```

---

## Gantt Chart Summary

```
2025                                  2026
Dec   Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct   Nov   Dec
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
[P0]
     [========= Phase 1: Agent SDK =========]
          ├─ AgentSpec v1
          ├─ BaseAgent Class
          ├─ Validation Engine (parallel)
          └─ Migration
                              [========= Phase 2: Factory Core =========]
                                   ├─ AgentSpec v2
                                   ├─ Generator ►────────────────────────────┐
                                   ├─ Evaluation (parallel)                  │
                                   └─ Golden Tests (parallel)                │
                                                                             │
                                        [========= Phase 3: Registry ========▼====]
                                             ├─ Registry API                 │
                                             ├─ Governance (parallel)        │
                                             ├─ Observability (parallel)     │
                                             └─ Deploy 50 Agents ◄───────────┘

                                                          [=== Phase 4 (Optional) ===]
                                                               ├─ Studio UI
                                                               ├─ Builder
                                                               └─ Partner Portal

Critical Path: ════════════════════════════════════════════════════════════════
               Scope → AgentSpec v1 → BaseAgent → Generator → Gen 10 → Deploy 50
```

---

## Dependency Risk Matrix

### High-Risk Dependencies

| Dependency | Risk Level | Impact | Owner | Mitigation Status |
|------------|------------|--------|-------|-------------------|
| AgentSpec v1 finalization | HIGH | Blocks all P1 | AI/Agent | Timebox + reviews |
| Climate Science rules | HIGH | Blocks validation | Climate | Early start |
| Generator quality | HIGH | Blocks certification | AI/Agent | Template approach |
| Golden test creation | HIGH | Blocks certification | Climate | Parallel track |
| LangGraph stability | MEDIUM | May need fallback | AI/Agent | Abstraction layer |

### Dependency Tracking

**Weekly Dependency Review:**
- Review all blocking dependencies
- Update status (Green/Yellow/Red)
- Escalate Red items to Engineering Lead

**Dependency Status Dashboard:**
```
Dependency Health (Week N):
├── AgentSpec v1:        [GREEN]  On track
├── BaseAgent Class:     [GREEN]  On track
├── Domain Rules:        [YELLOW] Need 1 more week
├── Generator Template:  [GREEN]  On track
├── Golden Tests:        [YELLOW] 60% complete
└── Registry API:        [GREEN]  Design approved
```

---

## Contingency Plans

### Contingency 1: AgentSpec v1 Delayed

**Trigger:** AgentSpec v1 not locked by Week 4
**Impact:** 2-4 week delay to Phase 1
**Action:**
1. Extend Phase 0 by 1 week
2. Lock minimal spec (identity + inputs/outputs only)
3. Defer advanced fields to v1.1
4. Communicate delay to all teams

### Contingency 2: Generator Quality Issues

**Trigger:** Generator success rate <70% by Week 20
**Impact:** Cannot certify 10 agents
**Action:**
1. Add human review step for all generated code
2. Focus on 5 high-value agents (not 10)
3. Extend Phase 2 by 2 weeks
4. Add 2 engineers to generator team

### Contingency 3: Certification Bottleneck

**Trigger:** <3 agents certified by Week 22
**Impact:** Cannot exit Phase 2
**Action:**
1. Train 2 additional reviewers
2. Implement parallel review (2 reviewers per agent)
3. Reduce golden tests to 50 (from 100)
4. Accept higher risk for Phase 2 exit

### Contingency 4: Registry Performance

**Trigger:** Registry latency >500ms at 50 agents
**Impact:** Poor user experience
**Action:**
1. Implement aggressive caching
2. Deploy read replicas
3. Reduce search scope
4. Add CDN for artifacts

---

## Appendices

### Appendix A: Detailed Task Dependencies (Phase 1)

| Task ID | Task Name | Duration | Predecessors | Team |
|---------|-----------|----------|--------------|------|
| P1.1 | AgentSpec v1 design | 2 weeks | P0.1 | AI/Agent |
| P1.2 | AgentSpec v1 implementation | 2 weeks | P1.1 | AI/Agent |
| P1.3 | BaseAgent design | 1 week | P1.1 | AI/Agent |
| P1.4 | BaseAgent implementation | 3 weeks | P1.3 | AI/Agent |
| P1.5 | BaseAgent testing | 2 weeks | P1.4 | AI/Agent |
| P1.6 | Validation Engine design | 1 week | P1.1 | ML Platform |
| P1.7 | Schema Validator | 2 weeks | P1.6 | ML Platform |
| P1.8 | Domain Validator | 3 weeks | P1.7 | ML Platform, Climate |
| P1.9 | Agent Graph DSL | 3 weeks | P1.2 | AI/Agent |
| P1.10 | Migration Agent 1 | 2 weeks | P1.5, P1.8 | AI/Agent |
| P1.11 | Migration Agent 2 | 1 week | P1.10 | AI/Agent |
| P1.12 | Migration Agent 3 | 1 week | P1.11 | AI/Agent |

### Appendix B: Critical Path Calculations

**Forward Pass (Earliest Start/Finish):**
```
P0.1: ES=0, EF=2
P1.1: ES=2, EF=4
P1.2: ES=4, EF=6
P1.3: ES=4, EF=5
P1.4: ES=5, EF=8
P1.5: ES=8, EF=10
...
```

**Backward Pass (Latest Start/Finish):**
```
P3.END: LS=36, LF=36
P3.5: LS=32, LF=36
...
P1.5: LS=8, LF=10 (Float=0, Critical)
P1.6: LS=4, LF=5 (Float=2, Not Critical)
```

### Appendix C: Dependency Communication Protocol

**When a dependency is blocked:**
1. Team lead notifies Engineering Lead within 4 hours
2. Engineering Lead convenes dependency sync within 24 hours
3. Mitigation plan documented and communicated
4. Daily updates until resolved

**When a dependency is at risk:**
1. Team lead raises in daily standup
2. Escalate to weekly integration sync if not resolved
3. Document in risk register

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-03 | GL-ProductManager | Initial critical path analysis |

---

**Approvals:**

- Product Manager: ___________________
- Engineering Lead: ___________________
- All Tech Leads: ___________________
