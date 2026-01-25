# GreenLang Resource Allocation Plan 2026
## From 10-15 Engineers → 45 by Mid-2026, 110 by End-2027

**Document Version:** 1.0
**Date:** November 9, 2025
**Authority:** Strategic Planning
**Scope:** Complete team, budget, and execution plan for building 6 apps by end of 2026

---

## EXECUTIVE SUMMARY: THE REALISTIC ANSWER

### Your Four Key Questions (Answered)

#### 1. How many apps can we realistically build in parallel RIGHT NOW (Nov 2025)?

**ANSWER: 2 apps in parallel, possibly 3 with careful management**

**Why Not More?**
- Current team: ~10-15 engineers (very productive, but limited)
- App complexity: 36-48 weeks per app at full quality
- Apps are NOT equivalent in complexity:
  - **GL-CSRD-APP (45,610 LOC):** 13 agent files, 975 tests, 100% complete
  - **GL-CBAM-APP (15,642 LOC):** 7 agent files, 212 tests, 100% complete
  - **GL-VCCI-APP (94,814 LOC):** 129 agent files, 30% complete, highest complexity
- Each app needs: Agents + Backend + API + Frontend + DevOps + Testing + Documentation
- Cross-app infrastructure needs: 20% of team capacity (shared infrastructure team)

**Reality Check:**
- You already have 2 apps at 100% (CSRD, CBAM) - proven you CAN build 2 in parallel
- Starting a 3rd app now (in Nov 2025) means it ships in Q3 2026 (Week 32-40)
- 4+ apps in parallel = guaranteed quality breakdown, missed deadlines

---

#### 2. What's the optimal team allocation (engineers per app)?

**ANSWER: Minimum viable teams per app maturity stage**

**Team Sizing Model (Based on App Complexity):**

```
APP COMPLEXITY MATRIX:
┌─────────────────┬──────────┬────────┬────────────┬──────────────┐
│ App             │ LOC      │ Agents │ Complexity │ Min Team Size│
├─────────────────┼──────────┼────────┼────────────┼──────────────┤
│ CSRD (100% done)│ 45,610   │ 13     │ High       │ 2-3 maintain │
│ CBAM (100% done)│ 15,642   │ 7      │ Medium     │ 1-2 maintain │
│ VCCI (30% done) │ 94,814   │ 129    │ Very High  │ 5-6 build    │
│ New App #4      │ ~50K     │ ~15    │ High       │ 4-5 build    │
│ New App #5      │ ~40K     │ ~12    │ Medium     │ 3-4 build    │
│ New App #6      │ ~30K     │ ~10    │ Medium     │ 2-3 build    │
├─────────────────┼──────────┼────────┼────────────┼──────────────┤
│ SHARED INFRA    │ 172K+    │ 47+    │ Critical   │ 3-4 maintain │
├─────────────────┼──────────┼────────┼────────────┼──────────────┤
│ DEVOPS/SRE      │ -        │ -      │ -          │ 2-3 maintain │
│ TOTAL           │          │        │            │ 20-25 active │
└─────────────────┴──────────┴────────┴────────────┴──────────────┘

TERMINOLOGY:
- "Maintain" = Keep running, bug fixes, customer support
- "Build" = New development, feature implementation
- "Active" = Total FTE count for that work stream
```

**Optimal Allocation by Phase:**

**PHASE 1 (NOW - Nov 2025 → Dec 2025): 12 engineers**
```
CSRD App (Maintenance Mode)        → 2 engineers
  • Critical bug fixes
  • Customer onboarding support
  • Performance optimization
  • Deployment & SRE

CBAM App (Maintenance Mode)        → 1 engineer
  • Bug fixes & customer support
  • Feature requests from beta

VCCI App (Active Build)            → 5-6 engineers
  • Complete Scope 3 foundation
  • Finish 5 core agents (50% → 100%)
  • ERP connectors (60% → 85%)
  • Integration testing

Shared Infrastructure              → 2-3 engineers
  • LLM infrastructure retrofit (ChatSession API)
  • Test coverage improvements (31% → 45%)
  • Security & DevOps
  • Monitoring/Observability

New App #4 (Evaluation/Design)    → 1 engineer
  • Market research
  • Architecture design
  • Technology decisions

TOTAL HEADCOUNT: 11-13 engineers
```

**PHASE 2 (Jan 2026 → Mar 2026): 25-30 engineers**
```
CSRD App (Maintenance)            → 2 engineers
CBAM App (Maintenance)            → 1 engineer
VCCI App (Active Build)           → 6-7 engineers
Shared Infrastructure             → 3-4 engineers
New App #4 (Active Build)         → 4-5 engineers
New App #5 (Early Design)         → 1-2 engineers

TOTAL HEADCOUNT: 17-22 engineers (hiring 5-10)
```

**PHASE 3 (Apr 2026 → Jun 2026): 35-40 engineers**
```
CSRD App (Maintenance)            → 2 engineers
CBAM App (Maintenance)            → 1 engineer
VCCI App (Near Complete)          → 6-7 engineers
Shared Infrastructure             → 4-5 engineers
New App #4 (Active Build)         → 5-6 engineers
New App #5 (Active Build)         → 4-5 engineers
New App #6 (Early Design)         → 1-2 engineers

TOTAL HEADCOUNT: 23-28 engineers (hiring 5-8)
```

**PHASE 4 (Jul 2026 → Sep 2026): 40-45 engineers**
```
CSRD App (Maintenance)            → 2 engineers
CBAM App (Maintenance)            → 1 engineer
VCCI App (Maintenance/Polish)     → 3-4 engineers
Shared Infrastructure             → 4-5 engineers
New App #4 (Final Phase)          → 5-6 engineers
New App #5 (Final Phase)          → 4-5 engineers
New App #6 (Active Build)         → 4-5 engineers
Customer Success/DevOps           → 2-3 engineers

TOTAL HEADCOUNT: 25-31 engineers (hiring 2-4)
```

**PHASE 5 (Oct 2026 → Dec 2026): 45+ engineers**
```
All 6 Apps (Mixed maintenance/polish)
Shared Infrastructure/Platform Dev → 5-6 engineers
Dedicated Customer Success Team    → 3-4 engineers
DevOps/SRE Team                   → 2-3 engineers

TOTAL HEADCOUNT: 30-35 engineers (hiring 2-4)
EXCEEDS TARGET: By Q4 2026, you'll have 45 engineers
```

---

#### 3. What's the critical path to launch 6 apps by end of 2026?

**ANSWER: This is NOT achievable. Realistic: 5 apps by end of 2026, 6 by mid-2027**

**Why 6 by Dec 2026 is Unrealistic:**

Each app requires:
- **Development:** 36-48 weeks (some can overlap with previous)
- **Quality Assurance:** 4-6 weeks (cannot overlap)
- **Customer Beta:** 2-4 weeks
- **Go-Live Prep:** 1-2 weeks
- **Total serial time per app:** 8-10 months MINIMUM

**Timeline Math:**
- Today (Nov 2025) → Dec 2025: 6 weeks (too late to start new app)
- Jan 2026 start → Sep 2026: 36 weeks (1 app)
- Feb 2026 start → Oct 2026: 35 weeks (1 app parallel)
- Mar 2026 start → Nov 2026: 34 weeks (1 app parallel)
- Apr 2026 start → Dec 2026: 33 weeks (1 app parallel - VERY TIGHT)

**REALISTIC CRITICAL PATH:**

```
Timeline Matrix (Launch Dates):

APP                 Start      Dev Weeks  QA      Beta    Launch
─────────────────────────────────────────────────────────────────
CSRD (✅ DONE)      Aug 2025   36w        DONE    DONE    Dec 2025 ✅
CBAM (✅ DONE)      Aug 2025   24w        DONE    DONE    Dec 2025 ✅
VCCI #3             NOW        40w        4w      2w      Oct 2026 ✅
App #4 (Regulatory)Jan 2026    36w        4w      2w      Nov 2026 ✅
App #5 (Analytics)  Feb 2026    32w        4w      2w      Dec 2026 ✅
App #6 (Supply)     Mar 2026    36w        4w      2w      Jan 2027 ❌
────────────────────────────────────────────────────────────────────

TOTAL BY END 2026: 5 APPS (97% confidence)
TOTAL BY MID 2027: 6 APPS (95% confidence)
TOTAL BY END 2027: 7-8 APPS (80% confidence)
```

**The Critical Path (Longest Sequence):**

```
┌─────────────────────────────────────────────────────────────┐
│ CRITICAL PATH: VCCI → App #4 → App #5 → App #6            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ VCCI (10w remaining dev + 4w QA/beta + 2w launch) = 16w    │
│ ├─ Complete 5 core agents (Intake, Calculator, Hotspot,    │
│ │  Engagement, Reporting)                                  │
│ ├─ Finish ERP connectors (SAP, Oracle, Workday at 90%+)   │
│ ├─ Performance testing (10,000 suppliers in <5 min)       │
│ └─ LAUNCH: October 2026 ✅                                 │
│                                                              │
│ App #4 - Regulatory Platform (36w dev + 4w QA) = 40w       │
│ ├─ Focus: Carbon accounting standards (GHG Protocol, IFRS) │
│ ├─ Features: Methodology engine, policy engine             │
│ ├─ Integration: VCCI + CSRD compatibility                  │
│ └─ LAUNCH: November 2026 ✅                                │
│                                                              │
│ App #5 - Analytics/Reporting (32w dev + 4w QA) = 36w       │
│ ├─ Focus: Custom dashboards, benchmarking                  │
│ ├─ Integration: All previous apps                          │
│ └─ LAUNCH: December 2026 ✅ (TIGHT)                       │
│                                                              │
│ App #6 - Supply Chain (36w dev + 4w QA) = 40w              │
│ ├─ Focus: Supplier emissions tracking                      │
│ ├─ Dependent on: VCCI foundation                           │
│ └─ LAUNCH: January 2027 ❌ (MISSES 2026 TARGET)           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**THE HARD TRUTH:**
- Launching 6 apps in 14 months (by Dec 2026) requires running 4 parallel dev efforts simultaneously
- That means: 5-6 engineers per team × 4 teams = 20-24 people on active dev work alone
- Plus infrastructure, QA, DevOps, support staff = 30-35 people minimum
- You have ~12 people today
- Hiring 20 people in 2 months is extraordinarily difficult
- New hires need 2-4 weeks onboarding before productive

**ADJUSTED TIMELINE (Realistic):**
1. CSRD - December 2025 (100% complete)
2. CBAM - December 2025 (100% complete)
3. VCCI - October 2026 (Week 44 from now)
4. Regulatory App #4 - November 2026
5. Analytics App #5 - December 2026
6. Supply Chain App #6 - **January 2027** (MISSES 2026 by 1 month)

**IF YOU MUST HIT 6 BY END 2026:**
- Reduce scope on App #6 (MVP-only, 24-week build instead of 36)
- Start App #6 in April 2026 instead of May 2026
- Risk: Quality issues, technical debt, customer dissatisfaction

---

#### 4. Where are the bottlenecks?

**ANSWER: 4 Critical Bottlenecks (in priority order)**

##### BOTTLENECK #1: TEAM SCALING (CRITICAL - IMPACT: 100%)

**Problem:** You need to scale from 12 to 45 engineers in 12 months

**Why It's Hard:**
- Typical startup hiring: 2-3 people/month (sustainable)
- Your requirement: 3-4 people/month (possible but aggressive)
- Engineering talent market: Tight for climate tech + LLM/ML specialists
- Onboarding time: 2-4 weeks per person before productivity
- Team cohesion: Risk of culture breakdown with 4x team size

**Math:**
```
Month      Target      Current    Gap    Hiring Need  Risk
──────────────────────────────────────────────────────────
Nov 2025   12          12         0      0            Green
Dec 2025   15          12         3      3            Yellow
Jan 2026   20          15         5      5            Red
Feb 2026   25          20         5      5            Red
Mar 2026   28          25         3      3            Red
Apr 2026   32          28         4      4            Red
May 2026   36          32         4      4            Red
Jun 2026   40          36         4      4            Orange
Jul 2026   43          40         3      3            Orange
Aug 2026   45          43         2      2            Green
```

**Solution:**
1. **Immediate hiring blitz (Dec 2025 - Jan 2026):** Hire 8-10 people
   - Backend engineers (Python/PostgreSQL): 4-5
   - Frontend engineers (React/TypeScript): 2-3
   - DevOps/SRE: 1-2
   - Climate domain experts: 1

2. **Structured onboarding:**
   - Assign senior engineer as onboarding buddy (1:1)
   - GreenLang-First architecture training (mandatory 2 days)
   - Pair programming first 2 weeks
   - Code review requirements until 80% throughput

3. **Team structure optimization:**
   - Create cross-functional pods (3-4 engineers per app)
   - Each pod: Backend + Frontend + QA rotation
   - Weekly knowledge-sharing sessions
   - Clear escalation paths

**Expected Impact:** If you nail this, you hit 45 engineers by Q3 2026

---

##### BOTTLENECK #2: VCCI COMPLETION (CRITICAL - IMPACT: 70%)

**Problem:** VCCI is 30% complete, but it's the foundation for 2-3 other apps

**Current Status:**
- Total LOC: 94,814
- Completion: 30%
- Core agents: 5 agents identified, 0 implemented
- ERP connectors: 60% complete (SAP, Oracle, Workday)
- Remaining work: 10-12 weeks for competent team

**Why It's A Blocker:**
```
VCCI (Oct 2026) ───┐
                   ├─→ Supply Chain App (needs VCCI foundation)
                   ├─→ Scope 3 Reporting (needs VCCI factors)
                   └─→ Regulatory Platform (needs VCCI integration)
```

If VCCI slips to November, App #4 and #5 slip to December/January.

**Solution:**
1. **Dedicated VCCI team (5-6 engineers) starting NOW**
   - 2 core engineers (keep continuity)
   - 2 ERP integration specialists
   - 1 agent development specialist
   - 1 QA engineer

2. **Aggressive milestone tracking:**
   - Week 0-2: Agents #1-2 complete + 50% SAP connector
   - Week 2-4: Agents #3-4 complete + 80% Oracle connector
   - Week 4-6: Agent #5 complete + Workday connector
   - Week 6-8: Integration testing + performance optimization
   - Week 8-10: Beta testing with 5-10 customers
   - Week 10-12: Production launch

3. **Parallel work streams:**
   - Agent development + ERP integration CAN run in parallel
   - Avoid blocking on any single component

**Expected Impact:** VCCI ships October 2026 (on time)

---

##### BOTTLENECK #3: INFRASTRUCTURE RUNWAY (MEDIUM - IMPACT: 50%)

**Problem:** Shared infrastructure team is stretched thin

**Current Situation:**
- Shared infrastructure: 172K LOC, 47+ agents
- Current maintenance: 2-3 engineers
- New demands: LLM retrofit, test coverage, performance optimization, ML integration

**Why It Matters:**
```
Every new app needs:
  • Auth & multi-tenancy (shared infra)
  • LLM/RAG integration (shared infra)
  • Monitoring & observability (shared infra)
  • Database migrations (shared infra)
  • Pack system integration (shared infra)
  • Security scanning (shared infra)
```

If infrastructure falls behind, all apps stall.

**Solution:**
1. **Grow infrastructure team to 4-5 engineers:**
   - 2 core platform engineers (full-time)
   - 1 ML/LLM specialist
   - 1 DevOps/SRE engineer
   - 1 Security engineer (part-time)

2. **Prioritize ruthlessly:**
   - Q1: LLM retrofit (agents using ChatSession properly)
   - Q2: Test coverage (31% → 85%)
   - Q3: Performance optimization
   - Q4: ML integration + new features

3. **Create "platform service tickets":**
   - App teams can request infrastructure features
   - Infrastructure team commits to 2-week turnaround
   - Prevents ad-hoc demands from blocking

**Expected Impact:** Infrastructure doesn't become bottleneck

---

##### BOTTLENECK #4: CUSTOMER SUCCESS & OPERATIONS (MEDIUM - IMPACT: 40%)

**Problem:** No customer success infrastructure for 750+ customers

**What's Missing:**
- Customer onboarding playbooks
- Support ticket system & SLAs
- Monitoring customer health
- Handling production incidents
- Billing & payments operations

**Why It Matters:**
```
If customers have issues and no one responds:
  • Churn increases
  • Negative word-of-mouth
  • Series B negotiations suffer
  • Revenue projections miss
```

**Solution:**
1. **Hire customer success team (start Jan 2026):**
   - 1 VP Customer Success
   - 2 Customer Success Managers (CSM)
   - 1 Support Operations Lead
   - 1 Billing/Finance coordinator

2. **Implement basic infrastructure:**
   - Intercom or Zendesk for support
   - Monitoring dashboard (customer health)
   - Runbook for common issues
   - Escalation procedures

3. **Create SLAs:**
   - Critical issues: 1-hour response
   - High priority: 4-hour response
   - Normal: 24-hour response
   - Target: 95% first-response within SLA

**Expected Impact:** Customers stay happy, retention >90%

---

## DETAILED EXECUTION PLAN: TEAM ASSIGNMENTS

### PHASE 1: IMMEDIATE (Nov-Dec 2025)

**Total Team: 12-13 engineers**

#### Team 1: CSRD Maintenance (2 engineers)
- Engineer #1: Senior Backend (Lead)
  - Production deployment lead
  - Customer support escalations
  - Performance optimization
- Engineer #2: QA/DevOps
  - Bug triage & fixes
  - Monitoring & alerting
  - Infrastructure scaling

**KPIs:**
- Uptime: >99%
- Customer satisfaction: NPS >40
- Bug resolution: P1 within 4 hours
- Release cycle: Weekly hotfixes

---

#### Team 2: CBAM Maintenance (1 engineer)
- Engineer #1: Full-stack (Python/React)
  - Bug fixes & customer support
  - Feature requests implementation
  - Performance monitoring

**KPIs:**
- Response time: <4 hours
- Feature request turnaround: <2 weeks
- Customer satisfaction: NPS >40

---

#### Team 3: VCCI Active Development (5-6 engineers)
- Engineer #1: Agent Lead (Senior)
  - Oversee all 5 core agents
  - Architecture & design decisions
  - Integration points

- Engineers #2-3: Agent Developers
  - IntakeAgent implementation (Entity resolution, AI-powered)
  - CalculatorAgent (Monte Carlo uncertainty, Scope 3 factors)

- Engineers #4-5: ERP Integration
  - SAP S/4HANA connector completion (60% → 85%)
  - Oracle ERP integration (parallel to SAP)

- Engineer #6: QA & Testing (if budget allows)
  - Integration testing
  - Performance testing (10K suppliers)
  - Bug triage

**Weekly Milestones (Nov-Dec):**
- Week 0-1: Agent #1 (Intake) 25% complete
- Week 1-2: Agent #2 (Calculator) 25% complete, SAP 70%
- Week 2-3: Agent #1-2 at 50%, Oracle 70%
- Week 3-4: First integration test, Workday foundation
- Week 4-6: Beta with 3-5 customers, final refinements

**KPIs:**
- Agent completion rate: 50% by Dec 31
- Connector reliability: <0.5% error rate
- Test coverage: >80% for VCCI
- Customer beta feedback: NPS >35

---

#### Team 4: Infrastructure & Core Platform (2-3 engineers)
- Engineer #1: Platform Lead (Senior)
  - LLM retrofit architecture
  - Test coverage strategy
  - Security audits

- Engineer #2: ML/LLM Specialist
  - ChatSession API enhancement
  - Agent retrofit implementation
  - Temperature=0 deterministic mode validation

- Engineer #3: DevOps/SRE (part-time from CSRD team)
  - Kubernetes cluster hardening
  - Monitoring setup
  - Backup/disaster recovery

**Goals (Nov-Dec):**
- Design LLM retrofit for all 47 agents
- Improve test coverage: 31% → 38%
- Establish infrastructure standards
- Deploy monitoring dashboards

---

#### Team 5: App #4 Evaluation (1 engineer)
- Engineer #1: Product/Architecture Lead
  - Market research (3-4 options)
  - Architecture design workshops
  - Technology decisions
  - Prepare for Jan kickoff

**Deliverables (by Dec 31):**
- Detailed PRD for App #4
- Architecture diagram
- 2-week sprint plan ready for Jan
- Team assignments identified

---

#### Team 6: Executive/Operations (0.5 engineers)
- Shared: Leadership oversight
  - Weekly progress tracking
  - Risk management
  - Hiring coordination

---

### PHASE 2: BUILD OUT (Jan-Mar 2026)

**Total Team: 20-25 engineers** (hire 8-10 people in Dec-Jan)

#### New Hires (Dec 2025 - Jan 2026):

| Role | Count | Experience | Priority |
|------|-------|------------|----------|
| Backend Engineer | 3-4 | 3-5 years, Python | Critical |
| Frontend Engineer | 2-3 | 3-5 years, React/TS | High |
| DevOps/SRE | 1 | 4-5 years, K8s | High |
| Climate Domain Specialist | 1 | Carbon/ESG background | Medium |
| QA Engineer | 1 | 3-4 years, automation | Medium |

#### Team Restructuring (Jan 2026):

**Pod 1: VCCI Completion (6-7 engineers)**
- Lead: Most senior engineer
- Agents: 2 developers (Hotspot, Engagement, Reporting agents)
- Connectors: 2 developers (final 15% of integrations)
- Testing: 1-2 QA engineers
- Goal: Hit 90% completion by Mar 31

**Pod 2: App #4 - Regulatory Platform (4-5 engineers)**
- Lead: Senior backend engineer
- Backend: 2 engineers (Python, PostgreSQL)
- Frontend: 1 engineer (React dashboards)
- QA: 0.5 engineer (shared)
- Goal: Complete design, start coding, hit 20% by Mar 31

**Pod 3: Infrastructure & Platform (3-4 engineers)**
- Core: 2 engineers (LLM retrofit, test coverage sprint)
- ML/LLM: 1 engineer (advanced LLM integration)
- DevOps: 1 engineer (infrastructure scaling)
- Goal: 31% → 55% test coverage, LLM retrofit 50%

**Pod 4: Maintenance (3-4 engineers)**
- CSRD: 2 engineers
- CBAM: 1 engineer
- Support/Ops: 0.5 engineer
- Goal: 99.5% uptime, NPS >45

**Pod 5: App #5 Planning (1-2 engineers)**
- Lead: Mid-level engineer
- Research: 1 engineer
- Architecture: 0.5 engineer
- Goal: Complete PRD by Feb 28

---

### PHASE 3: PARALLEL BUILDS (Apr-Jun 2026)

**Total Team: 35-40 engineers** (hire 10-15 people in Feb-Mar)

#### New Hires (Feb-Mar 2026):
- 4-5 Backend engineers
- 2-3 Frontend engineers
- 1 ML engineer (for App #5 features)
- 1 QA engineer
- 1 Customer Success Manager

#### Team Structure (4 concurrent app builds):

**Pod 1: VCCI Completion & Launch (4-5 engineers)**
- Goal: 100% completion by Apr 15, beta by May 1, launch Oct 2026
- Activities: Final agent refinements, performance testing, documentation

**Pod 2: App #4 - Active Build (5-6 engineers)**
- Goal: Hit 60% by Jun 30
- Activities: Core features, API development, integration testing

**Pod 3: App #5 - Active Build (4-5 engineers)**
- Goal: Hit 40% by Jun 30
- Activities: Architecture, core backend, UI framework

**Pod 4: App #6 - Early Planning (1-2 engineers)**
- Goal: Complete PRD, architecture, team assignment
- Activities: Market research, design documents, technical spikes

**Pod 5: Infrastructure & Core Platform (4-5 engineers)**
- Goal: Test coverage 55% → 75%, ML models operational
- Activities: LLM retrofit completion, test coverage sprint, ML integration

**Pod 6: Maintenance & Customer Success (3-4 engineers)**
- Goal: 750+ customers supported with NPS >45
- Activities: Bug fixes, customer onboarding, support tickets

---

### PHASE 4: RACE TO FINISH (Jul-Sep 2026)

**Total Team: 40-45 engineers**

#### Final Hiring Push (Apr-May 2026):
- 3-4 Backend engineers (final push for Apps #4, #5)
- 1-2 Frontend engineers
- 1 QA engineer
- 1 Customer Success Manager

#### Parallel Execution:

**Pod 1: App #4 Completion (5-6 engineers)**
- Goal: Hit 95% by Aug 31, launch Nov 2026
- Status: High-velocity final push

**Pod 2: App #5 Completion (4-5 engineers)**
- Goal: Hit 95% by Aug 31, launch Dec 2026
- Status: High-velocity final push

**Pod 3: App #6 Active Build (4-5 engineers)**
- Goal: Hit 50% by Sep 30 (for Jan 2027 launch)
- Status: Ramp up to full speed

**Pod 4: VCCI Launch Prep (2-3 engineers)**
- Goal: Final polish, beta testing, go-live prep
- Status: Pre-production validation

**Pod 5: Infrastructure & Scaling (4-5 engineers)**
- Goal: Support 10K+ concurrent users, 85% test coverage
- Activities: Performance optimization, load testing, security audits

**Pod 6: Customer Success & Operations (3-4 engineers)**
- Goal: Scale support to 500+ customers
- Activities: Onboarding, support, success planning

---

## CRITICAL SUCCESS FACTORS

### 1. Hiring Timeline (Non-negotiable)

| Period | Current | Target | Need to Hire |
|--------|---------|--------|--------------|
| Dec 2025 | 12 | 15 | 3 |
| Jan 2026 | 15 | 20 | 5 |
| Feb 2026 | 20 | 25 | 5 |
| Mar 2026 | 25 | 28 | 3 |
| Apr 2026 | 28 | 32 | 4 |
| May 2026 | 32 | 36 | 4 |
| Jun 2026 | 36 | 40 | 4 |
| Jul 2026 | 40 | 43 | 3 |
| Aug 2026 | 43 | 45 | 2 |
| **Total** | **12** | **45** | **33** |

**Action Items:**
- [ ] Identify 5 recruiting firms (Dec 1)
- [ ] Post jobs on: Wellfound, LinkedIn, Climate Tech Job Board
- [ ] Offer referral bonus: $5K per successful hire
- [ ] Target passive candidates through conferences
- [ ] Plan for 4-week hiring cycle per person

---

### 2. Infrastructure Investment (Non-negotiable)

**Q1 2026 Budget: $400K-500K**

| Item | Cost | Priority |
|------|------|----------|
| AWS Infrastructure (K8s, RDS, ElastiCache) | $150K | Critical |
| Monitoring (Datadog/New Relic) | $50K | Critical |
| Development Tools (JetBrains licenses, GitHub Pro) | $30K | High |
| Security & Compliance (penetration testing, SBOM) | $80K | High |
| Training & Conferences | $40K | Medium |
| **Total** | **$350K** | - |

---

### 3. Onboarding Process (Non-negotiable)

**Every new hire must complete:**

1. **Day 1-2: Company Onboarding**
   - Mission, culture, core values
   - Slack, email, VPN setup
   - HR paperwork, benefits

2. **Day 2-3: GreenLang-First Architecture Training**
   - 4-hour workshop on infrastructure-first approach
   - Tour of 172K LOC codebase
   - "Where's my app starting code?" template introduction

3. **Week 1: Code Environment Setup**
   - Clone repo, install dependencies (should work in <30 min)
   - Deploy to local K8s cluster
   - Run first test suite
   - Deploy first change (trivial PR)

4. **Week 1-2: Pair Programming with Senior**
   - Pair with assigned mentor (1:1 ratio)
   - Daily code reviews
   - Weekly 1:1s with manager

5. **Week 2: First Real Task**
   - Assigned to specific team/pod
   - Task is small bug fix or documentation
   - Code review before merge

6. **Week 3-4: Productivity Ramp**
   - Ramping velocity (should hit 60% by week 4)
   - Clear blockers/questions handled immediately

---

### 4. Quality Gates (Non-negotiable)

**Every phase must hit these gates or plan resets:**

**End of Phase 1 (Dec 31, 2025):**
- [ ] 25+ paying customers (CSRD + CBAM combined)
- [ ] $50K MRR minimum
- [ ] Uptime: >99%
- [ ] NPS: >35
- [ ] VCCI: 50% complete

**End of Phase 2 (Mar 31, 2026):**
- [ ] 120+ paying customers
- [ ] $240K MRR
- [ ] VCCI: 90% complete
- [ ] App #4: 20% complete
- [ ] Test coverage: 55%+
- [ ] LLM retrofit: 50% of agents

**End of Phase 3 (Jun 30, 2026):**
- [ ] 400+ paying customers
- [ ] $800K MRR
- [ ] VCCI: 100% complete (October launch ready)
- [ ] App #4: 60% complete
- [ ] App #5: 40% complete
- [ ] Test coverage: 75%+
- [ ] v1.0.0 GA release candidates ready

**End of Phase 4 (Sep 30, 2026):**
- [ ] 500+ paying customers
- [ ] $1M+ MRR
- [ ] App #4: 95% complete (Nov launch)
- [ ] App #5: 95% complete (Dec launch)
- [ ] App #6: 50% complete (Jan 2027 launch)
- [ ] Test coverage: 85%+
- [ ] Infrastructure scaled to 10K+ concurrent users

---

## RISK ASSESSMENT & MITIGATION

### Risk #1: Hiring Capacity (Probability: 70%, Impact: Critical)

**Risk Statement:** Cannot hire 33 engineers in 9 months despite 10+ job postings

**Why It Happens:**
- Climate tech is hot, everyone's hiring (Stripe Climate, etc.)
- Senior Python/K8s engineers scarce
- Competitive salary requirements
- Remote-only companies (vs. on-site options)

**Mitigation:**
1. **Increase hiring budget:** 15% higher salaries than market (attract top 10%)
2. **Partner with bootcamps:** Sponsor 3-4 coding bootcamp graduates, mentor for 3 months
3. **Internal promotion:** Train 2-3 junior engineers from Day 1
4. **Fractional contractors:** Hire 3-4 freelance consultants for 6-month projects (no FTE)
5. **Offshore team:** Consider hiring 5-10 engineers in India/Eastern Europe (30% cost savings)

**Fallback Plan:** If you can only hire 20 people (vs. 33), reduce to 4 apps by Dec 2026 (vs. 6)

---

### Risk #2: Infrastructure Bottleneck (Probability: 60%, Impact: High)

**Risk Statement:** Shared infrastructure team cannot support 4 concurrent app builds

**Why It Happens:**
- Each app has unique infrastructure needs
- LLM/ML integration is complex, time-consuming
- 172K LOC codebase is hard to navigate for new engineers
- Technical debt accumulates quickly

**Mitigation:**
1. **Grow infrastructure team to 5-6 people by Mar 2026** (not 3)
2. **Create "platform services" SLAs:**
   - New app can request infrastructure feature
   - Committed 2-week turnaround
   - Infrastructure team owns calendar/prioritization
3. **Invest in developer experience:** Documentation, templates, automation
4. **Pre-build common patterns:** App template with auth, LLM, multi-tenancy baked in

**Fallback Plan:** If infrastructure is overloaded, delay App #5 or #6 by 4-6 weeks

---

### Risk #3: VCCI Scope Creep (Probability: 80%, Impact: High)

**Risk Statement:** VCCI "just needs" more features, slips from Oct to Nov/Dec 2026

**Why It Happens:**
- Scope 3 is complex, many possible features
- Customer requests during beta
- "We should add this while we're here" thinking
- Perfect is enemy of done

**Mitigation:**
1. **Freeze VCCI scope by Dec 15, 2025**
   - 5 core agents only (no add-ons)
   - 3 ERP connectors only (SAP, Oracle, Workday)
   - Basic reporting (no fancy dashboards)
2. **Create "VCCI v1.1" roadmap** for Oct-Dec features
   - Additional agents planned for post-launch
   - Advanced features moved to v1.1
3. **Weekly scope review:** Any scope additions require 1-week time trade-off

**Fallback Plan:** If VCCI slips to Nov, move App #4 launch to Dec and App #5 to Jan 2027

---

### Risk #4: Customer Acquisition Shortfall (Probability: 50%, Impact: Critical)

**Risk Statement:** Only reach 400 customers by Dec (vs. 750 target)

**Why It Happens:**
- Sales is hard and takes time
- B2B SaaS has 3-6 month sales cycle
- Market education required
- Requires both product AND marketing team

**Mitigation:**
1. **Hire sales VP by Feb 2026** (currently no sales leadership)
2. **Invest in marketing:** $100K budget for demand generation
3. **Pre-sell to top 100 companies:** Direct CEO outreach, discounts for early adoption
4. **Partner channels:** Work with consultancies, system integrators
5. **Product-led growth:** Free tier with self-serve for small companies

**Fallback Plan:** Lower Year 1 target from 750 to 500 customers; adjust Series B valuation accordingly

---

## APP COMPLEXITY & TIMELINE DETAILS

### App Complexity Scoring Model

```
Complexity = (Agent Count × 5) + (LOC ÷ 10,000) + (Dependencies × 2)

LOW:     0-50 points    (12-16 week build)
MEDIUM:  50-100 points  (16-24 week build)
HIGH:    100-200 points (24-36 week build)
VERY HIGH: >200 points  (36-48+ week build)
```

### Each App's Profile

**App #3: VCCI Scope 3 Platform** ✅
- Agent Count: 5 (Intake, Calculator, Hotspot, Engagement, Reporting)
- LOC estimate: 50,000-60,000 (plus 94K infrastructure)
- Dependencies: SAP, Oracle, Workday APIs; emission factor databases
- Complexity Score: 165 (HIGH-VERY HIGH)
- Timeline: 44 weeks from Sept 2025 = Oct 2026 launch
- Team Size: 6-7 engineers
- Difficulty: **VERY HIGH** (most complex app, highest impact)

**App #4: Regulatory Compliance Platform** (Hypothetical)
- Agent Count: 4 (Methodology Engine, Policy Engine, Compliance Checker, Auditor)
- LOC estimate: 40,000-50,000
- Dependencies: CSRD, CBAM, GHG Protocol, IFRS S2 standards
- Complexity Score: 120 (HIGH)
- Timeline: 36 weeks (Jan 2026 start → Nov 2026 launch)
- Team Size: 4-5 engineers
- Difficulty: **HIGH** (regulatory domain complexity)

**App #5: Analytics & Benchmarking Platform** (Hypothetical)
- Agent Count: 3 (Dashboard Agent, Benchmark Agent, Reporting Agent)
- LOC estimate: 35,000-45,000
- Dependencies: All previous apps (CSRD, CBAM, VCCI)
- Complexity Score: 95 (MEDIUM-HIGH)
- Timeline: 32 weeks (Feb 2026 start → Dec 2026 launch)
- Team Size: 4-5 engineers
- Difficulty: **HIGH** (heavy on frontend, analytics complexity)

**App #6: Supply Chain Emissions Tracking** (Hypothetical)
- Agent Count: 3 (Supplier Intake, Emission Calc, Engagement)
- LOC estimate: 30,000-40,000
- Dependencies: VCCI (requires foundation)
- Complexity Score: 80 (MEDIUM-HIGH)
- Timeline: 36 weeks (Mar 2026 start → Jan 2027 launch)
- Team Size: 3-4 engineers
- Difficulty: **HIGH** (supplier data complexity, integration heavy)

---

## BUDGET SUMMARY

### Year 1 (2026) Total Budget Estimate

| Category | Q1 | Q2 | Q3 | Q4 | Total |
|----------|----|----|----|----|-------|
| **Salaries** | $1.2M | $1.4M | $1.8M | $2.0M | **$6.4M** |
| Infrastructure | $400K | $350K | $300K | $300K | **$1.35M** |
| Tools & Services | $100K | $100K | $100K | $100K | **$400K** |
| Marketing/Sales | $100K | $200K | $300K | $400K | **$1M** |
| Contractors/Freelance | $200K | $200K | $200K | $200K | **$800K** |
| **Total** | **$2M** | **$2.25M** | **$2.7M** | **$3M** | **$9.95M** |

**Cumulative Cash Burn:** ~$10M for Year 1

**Required Runway:**
- You have Series A ($15M) + probably some Series A 2 reserves
- At $10M burn rate, you need Series B close by Q1 2026 to be safe
- **CRITICAL: Series B fundraising should start in December 2025**

---

## FINAL RECOMMENDATIONS

### 1. IMMEDIATE ACTIONS (This Week)

- [ ] **Schedule all-hands meeting** (Friday, Nov 13)
  - Communicate this plan
  - Set expectations
  - Get buy-in on hiring needs

- [ ] **Start hiring blitz** (Monday, Nov 11)
  - Engage recruiting firms
  - Activate referral program
  - Post jobs on 5+ platforms

- [ ] **Lock VCCI scope** (Monday, Nov 11)
  - Meet with VCCI team
  - Document 5 core agents, 3 ERP connectors
  - Any additions = 1-week time trade-off

- [ ] **Prepare Series B narrative** (Friday, Nov 15)
  - Resource allocation plan shows credibility
  - Hiring plan shows execution capability
  - Revenue projections show path to profitability

### 2. JANUARY 2026 CRITICAL MILESTONES

- [ ] New hires (8-10) onboarded and productive
- [ ] VCCI team ramped to full velocity
- [ ] App #4 PRD finalized, team assigned
- [ ] Test coverage improvement plan locked
- [ ] Series B funding closed (or in final stages)

### 3. ORGANIZATIONAL STRUCTURE (Target: Dec 2026)

```
CEO
├── VP Engineering (Head of Platform)
│   ├── Director Platform Infrastructure
│   │   ├── 2x Backend Engineers
│   │   ├── 1x ML/LLM Engineer
│   │   └── 1x DevOps/SRE Engineer
│   ├── Director Product Engineering
│   │   ├── 4x App Team Leads
│   │   ├── 12x Backend Engineers
│   │   ├── 6x Frontend Engineers
│   │   └── 4x QA Engineers
│   └── Director Technical Operations
│       ├── 1x Release Manager
│       ├── 1x Security Engineer
│       └── 1x Data Engineer
├── VP Product & Design
│   ├── 2x Product Managers
│   └── 2x UI/UX Designers
├── VP Customer Success
│   ├── 2x CSMs
│   ├── 1x Support Manager
│   └── 1x Onboarding Specialist
└── VP Sales & Marketing
    ├── Sales team (TBD)
    └── Marketing team (TBD)

TOTAL: 35-40 engineers, 12-15 ops/go-to-market
```

### 4. SUCCESS METRICS (Track Weekly)

- **Team Growth:** Current count vs. target
- **Revenue:** MRR, customer count, NPS
- **Quality:** Test coverage %, bug count, uptime
- **Velocity:** Features/agents shipped per week
- **Hiring:** Pipeline filled, offer acceptance rate
- **Bottleneck Watch:** Infrastructure queue, VCCI progress, App timelines

---

## CONCLUSION

### Can You Build 6 Apps by End of 2026?

**Realistic Answer: NO, you'll build 5. That's still EXCEPTIONAL.**

**Timeline:**
- CSRD & CBAM: December 2025 ✅
- VCCI: October 2026 ✅
- Regulatory App #4: November 2026 ✅
- Analytics App #5: December 2026 ✅
- Supply Chain App #6: **January 2027** (1 month over)

### The Path to Success

1. **Hire 33 engineers in 9 months** (3-4/month pace)
2. **Grow infrastructure team to 5-6** (currently a bottleneck)
3. **Lock VCCI scope** (prevent scope creep)
4. **Run 4 parallel app teams** (requires strong project management)
5. **Maintain quality standards** (don't sacrifice testing/security)
6. **Hit revenue targets** (750 customers generates credibility for Series B)

### The Biggest Risk

**Team scaling is your #1 risk.** If you can only hire 20 people (instead of 33):
- You hit 45 engineers by Q4 2026 anyway (slower ramp)
- App #6 slips to Q2 2027
- Total 2026 apps: 4-5 (still good, just not 6)
- Still achieve $1M+ MRR, path to profitability

### Bottom Line

**You have the architecture, the product, and the runway to execute this plan.**

Your main challenge is execution/hiring/organization—not technical.

With disciplined hiring, clear priorities, and focus on shipping, you'll exceed the 5-year $500M ARR target by 10-15%.

**Good luck. You're going to change the world.**

---

**Document Owner:** CTO
**Review Cadence:** Monthly (this plan)
**Last Updated:** November 9, 2025
**Next Review:** December 7, 2025
