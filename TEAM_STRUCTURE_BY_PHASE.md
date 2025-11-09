# GreenLang Team Structure & Organization
## Evolution from Current State → December 2026

**Document Version:** 1.0
**Date:** November 9, 2025
**Scope:** Detailed org chart, role descriptions, and hiring timeline

---

## CURRENT STATE (November 2025)

**Total Headcount: 10-15 engineers** (estimate based on 172K LOC at 8-10 LOC/engineer-day productivity)

### Current Organization (Estimated)

```
CTO / Founder
├── Team Lead - CSRD-APP
│   ├── 2-3 backend engineers
│   ├── 1-2 frontend engineers
│   └── 1 QA engineer
├── Team Lead - CBAM-APP
│   ├── 1-2 backend engineers
│   └── 1 frontend engineer
├── Team Lead - VCCI-APP
│   ├── 2-3 backend engineers
│   ├── 1 ERP specialist
│   └── 1 agent developer
└── Infrastructure/DevOps
    └── 1-2 engineers
```

**Roles (Estimated):**
- 5-6 backend/core engineers
- 3-4 frontend engineers
- 1-2 DevOps/infrastructure engineers
- 1 QA engineer
- Other: (CEO, CFO, probably 1-2 non-technical founders)

---

## PHASE 1: IMMEDIATE (Nov 15 - Dec 31, 2025)

**Target Headcount: 12-15 engineers** (No new hires, just reorganize)

### Organizational Chart (Dec 31, 2025)

```
CEO
│
├── CTO / VP Engineering
│   ├── Eng Lead - CSRD-APP (Maintenance)
│   │   ├── Senior Backend Engineer #1
│   │   └── QA/DevOps Engineer #1
│   │
│   ├── Eng Lead - CBAM-APP (Maintenance)
│   │   └── Full-Stack Engineer #1
│   │
│   ├── Eng Lead - VCCI-APP (Active Development)
│   │   ├── Senior Agent Architect (Lead)
│   │   ├── Agent Developer #1
│   │   ├── Agent Developer #2
│   │   ├── ERP Integration Specialist #1
│   │   └── QA Engineer #1
│   │
│   └── Eng Lead - Platform/Infrastructure (Shared)
│       ├── Platform Architect (Senior)
│       └── ML/LLM Specialist
│
├── CFO
│
├── HR Lead
│
└── Early-stage founder role (if still active)
```

**Key Changes from Current:**
- CSRD team: Move to maintenance mode (reduce from 3-4 to 2)
- CBAM team: Move to maintenance mode (reduce from 2-3 to 1)
- VCCI team: Consolidate and focus (dedicate 5-6 engineers)
- Infrastructure: Formalize structure, add LLM specialist

---

## PHASE 2: BUILD OUT (Jan 1 - Mar 31, 2026)

**Target Headcount: 20-25 engineers** (Hire 8-10 people)

### Hiring Plan (Dec 2025 - Jan 2026)

**Priority Order:**

1. **Senior Backend Engineer** (Week 1 of Jan)
   - 5+ years Python, PostgreSQL
   - Kubernetes preferred
   - Can lead App #4 development
   - Salary: $220-270K

2. **Backend Engineer #2** (Week 2 of Jan)
   - 3-5 years, full-stack optional
   - Joins App #4 team or VCCI team
   - Salary: $150-170K

3. **Frontend Engineer** (Week 2 of Jan)
   - 3-5 years React/TypeScript
   - Joins App #4 or existing team
   - Salary: $150-180K

4. **ML/LLM Specialist** (Week 3 of Jan) [CRITICAL]
   - AI/ML background, Python expert
   - Works on ChatSession API retrofit
   - Salary: $200-240K

5. **Backend Engineer #3** (Week 1 of Feb)
   - Joins VCCI team
   - Salary: $150-170K

6. **Frontend Engineer #2** (Week 2 of Feb)
   - Joins App #4 team
   - Salary: $150-180K

7. **QA Engineer** (Week 3 of Feb)
   - Automation focused
   - Test coverage improvement
   - Salary: $120-150K

8. **DevOps/SRE Engineer** (Week 1 of Mar) [CRITICAL]
   - Infrastructure scaling
   - Kubernetes, monitoring
   - Salary: $180-220K

9. **Backend Engineer #4** (Week 2 of Mar)
   - Junior/mid-level
   - Joins any team needing capacity
   - Salary: $130-160K

10. **Climate Domain Specialist** (Week 3 of Mar)
    - Carbon accounting, emissions background
    - Works with product/agent teams
    - Salary: $140-180K

### Organizational Chart (Mar 31, 2026)

```
CEO
│
├── VP Engineering (CTO or new hire)
│   ├── Director Platform Infrastructure
│   │   ├── Platform Architect (Senior)
│   │   ├── ML/LLM Engineer
│   │   ├── DevOps/SRE Engineer
│   │   └── Security Engineer (part-time)
│   │
│   ├── Director Product Engineering
│   │   ├── Eng Lead - CSRD-APP (Maintenance)
│   │   │   ├── Senior Backend Engineer
│   │   │   └── QA/DevOps Engineer
│   │   │
│   │   ├── Eng Lead - CBAM-APP (Maintenance)
│   │   │   └── Full-Stack Engineer
│   │   │
│   │   ├── Eng Lead - VCCI-APP (Active Development)
│   │   │   ├── Senior Agent Architect
│   │   │   ├── Agent Developer #1
│   │   │   ├── Agent Developer #2
│   │   │   ├── ERP Integration Specialist #1
│   │   │   ├── ERP Integration Specialist #2
│   │   │   └── QA Engineer
│   │   │
│   │   └── Eng Lead - App #4 (Early Development)
│   │       ├── Senior Backend Engineer (new)
│   │       ├── Backend Engineer
│   │       ├── Frontend Engineer
│   │       └── QA Engineer (shared)
│   │
│   └── Tech Lead (1 engineer per major component)
│
├── VP Product & Design (new hire by Mar)
│   ├── Product Manager - Growth
│   └── UI/UX Designer
│
├── CFO
│   └── Finance Operations
│
├── HR Lead
│   └── Recruiting Coordinator (part-time)
│
└── Early-stage operations
```

**Total Distribution (Mar 31, 2026):**
- CSRD (Maintenance): 2
- CBAM (Maintenance): 1
- VCCI (Active Build): 6-7
- App #4 (Early Build): 3-4
- Infrastructure/Platform: 3-4
- Non-engineering leadership: 4-5
- **Total: 20-25 engineers**

---

## PHASE 3: PARALLEL BUILDS (Apr 1 - Jun 30, 2026)

**Target Headcount: 35-40 engineers** (Hire 10-15 people)

### Hiring Plan (Apr-May 2026)

**New Hires:**
1. Backend Engineer #5 (early Apr)
2. Backend Engineer #6 (early Apr)
3. Frontend Engineer #3 (mid Apr)
4. ML Engineer (mid Apr) [For App #5 analytics]
5. QA Engineer #2 (late Apr)
6. Backend Engineer #7 (early May)
7. Backend Engineer #8 (mid May)
8. Frontend Engineer #4 (mid May)
9. Customer Success Manager #1 (late May)
10. Billing/Operations Coordinator (late May)
11-15. Other roles TBD based on progress

### Organizational Chart (Jun 30, 2026)

```
CEO
│
├── VP Engineering
│   ├── Director Platform Infrastructure (4-5 engineers)
│   │   ├── Platform Architect
│   │   ├── ML/LLM Engineer
│   │   ├── DevOps/SRE Engineer
│   │   └── Security Engineer
│   │
│   ├── Director Product Engineering (18-20 engineers)
│   │   ├── CSRD-APP Team (2)
│   │   ├── CBAM-APP Team (1)
│   │   ├── VCCI-APP Team (6-7)
│   │   │   └── Agent Lead, Agent Devs, ERP Specs, QA
│   │   ├── App #4 Team (5-6)
│   │   │   └── Tech Lead, Backend (2), Frontend, QA
│   │   └── App #5 Team (4-5)
│   │       └── Tech Lead, Backend (2), Frontend, QA
│   │
│   └── Tech Leads (1 per major system)
│       ├── VCCI Agent Lead
│       ├── App #4 Lead
│       ├── App #5 Lead
│       └── Infrastructure Lead
│
├── VP Product & Design
│   ├── Product Manager - Platform
│   ├── Product Manager - Apps
│   └── UI/UX Designer (1-2)
│
├── VP Customer Success
│   ├── Customer Success Manager #1
│   ├── Support Operations Lead
│   └── Billing/Onboarding Specialist
│
├── VP Marketing & Sales (hiring by Jun)
│   ├── Head of Marketing
│   └── Marketing Associate
│
├── CFO
│
├── HR Lead
│   └── Recruiting Coordinator
│
└── Other Operations
```

**Total Distribution (Jun 30, 2026):**
- Engineering: 28-30
- Product & Design: 2-3
- Customer Success: 2
- Operations: 2-3
- Leadership: 4-5
- **Total: 35-40 engineers + operations**

---

## PHASE 4: RACE TO FINISH (Jul 1 - Sep 30, 2026)

**Target Headcount: 40-45 engineers**

### Key Changes:
- App #4 enters final phase (80% → 95%)
- App #5 enters final phase (50% → 95%)
- App #6 begins active development
- Sales & Marketing team grows to 3-5 people
- Customer Success grows to 3-4 people

### Organizational Chart (Sep 30, 2026)

```
CEO
│
├── CTO / VP Engineering
│   ├── Director Platform Infrastructure (5 engineers)
│   ├── Director Product Engineering (20-22 engineers)
│   │   ├── CSRD-APP (2) - Maintenance
│   │   ├── CBAM-APP (1) - Maintenance
│   │   ├── VCCI-APP (3-4) - Pre-launch polish
│   │   ├── App #4 (5-6) - Final push
│   │   ├── App #5 (5-6) - Final push
│   │   └── App #6 (3-4) - Active development
│   ├── Engineering Manager (1-2)
│   └── Tech Ops Lead
│
├── VP Product
│   ├── Product Manager x2
│   ├── Product Operations Lead
│   └── Designer x2
│
├── VP Customer Success
│   ├── Customer Success Manager x2
│   ├── Support Lead
│   └── Billing & Operations Coordinator
│
├── VP Sales & Marketing (new by Jul)
│   ├── Sales Manager
│   ├── Sales Development Rep (SDR)
│   ├── Head of Marketing
│   ├── Demand Generation Specialist
│   └── Content/Communications Lead
│
├── CFO & Finance (expanded)
│
├── HR & People Ops (expanded)
│
└── Office Management / Admin
```

**Total Distribution (Sep 30, 2026):**
- Engineering: 27-32
- Product & Design: 4-5
- Customer Success: 3-4
- Sales & Marketing: 4-5
- Operations & Finance: 3-4
- **Total: 40-45 engineers + 15-20 non-engineering**

---

## FINAL STATE (Dec 31, 2026)

**Target Headcount: 45 engineers + 15-20 operations/sales/marketing**

### Organizational Chart (Dec 31, 2026)

```
CEO
│
├── CTO / VP Engineering (40-50 direct/indirect)
│   ├── Director Platform Infrastructure (5-6)
│   │   └── Platform teams, DevOps, Security, ML/LLM
│   ├── Director Product Engineering (25-30)
│   │   └── 6 app teams (CSRD, CBAM, VCCI, #4, #5, #6)
│   ├── Engineering Manager (2-3)
│   └── Tech Ops & Quality Lead
│
├── VP Product & Design (4-5)
│   ├── Sr Product Manager - Platform
│   ├── Product Manager - Growth/Analytics
│   └── Lead Designer + UI/UX team
│
├── VP Customer Success (4-5)
│   ├── Manager, Customer Success
│   ├── Customer Success Manager x2
│   ├── Customer Support Lead
│   └── Success Operations Specialist
│
├── VP Sales & Revenue (5-8)
│   ├── Sales Manager
│   ├── Account Executive x2-3
│   ├── Sales Development Rep
│   └── Revenue Operations
│
├── Head of Marketing (2-3)
│   ├── Demand Generation Lead
│   ├── Content/Communications Specialist
│   └── Marketing Operations
│
├── CFO (2-3)
│   ├── Finance Manager
│   └── Accountant
│
├── VP People & Culture (2-3)
│   ├── Recruiting Lead
│   ├── HR Business Partner
│   └── People Operations
│
└── Office Manager / Admin (1)
```

**Total by Dec 31, 2026: 45 engineers + 25-35 non-engineering = 70-80 total headcount**

---

## ROLE DESCRIPTIONS & QUALIFICATIONS

### Engineering Roles

#### Senior Backend Engineer (Python/PostgreSQL)
**Level:** Staff/Lead (5-10 years experience)
**Salary Range:** $220-270K
**Key Responsibilities:**
- Lead backend architecture decisions
- Mentor 2-3 junior engineers
- Own critical infrastructure components
- Performance optimization
- Security review & compliance

**Qualifications:**
- 5+ years professional Python development
- PostgreSQL expertise (query optimization, schema design)
- Kubernetes or similar orchestration platform
- Distributed systems understanding
- Mentorship experience

**Key Hires for This Role:**
- Infrastructure Lead (Jan 2026) - From VCCI team
- App #4 Tech Lead (Feb 2026) - External hire
- App #5 Tech Lead (Apr 2026) - External hire

---

#### Backend Engineer (Python/FastAPI)
**Level:** Mid-level (3-5 years experience)
**Salary Range:** $150-180K
**Key Responsibilities:**
- Backend API development
- Database schema design
- Integration with ERP systems
- Agent implementation & debugging

**Qualifications:**
- 3-5 years Python development
- FastAPI or similar framework
- PostgreSQL experience
- REST API design
- Basic DevOps understanding (Docker, deployment)

**Hiring Schedule:**
- Week 1 Jan: 1 hire
- Week 3 Jan: 1 hire
- Week 2 Feb: 1 hire
- Week 1 Mar: 1 hire
- Total Year 1: 6-8 backend engineers

---

#### Frontend Engineer (React/TypeScript)
**Level:** Mid-level (3-5 years experience)
**Salary Range:** $150-180K
**Key Responsibilities:**
- React component development
- Dashboard creation
- Data visualization
- User experience optimization

**Qualifications:**
- 3-5 years React development
- TypeScript proficiency
- CSS/styling (Tailwind preferred)
- API integration
- Accessibility standards (WCAG)

**Hiring Schedule:**
- Week 2 Jan: 1 hire
- Week 2 Feb: 1 hire
- Week 1 Apr: 1 hire
- Week 3 May: 1 hire
- Total Year 1: 4 frontend engineers

---

#### ML/LLM Specialist
**Level:** Senior (4-6 years AI/ML)
**Salary Range:** $200-240K
**Key Responsibilities:**
- LLM integration (ChatSession API)
- RAG system enhancement
- Agent prompt engineering
- Model evaluation & optimization
- Deterministic calculation validation

**Qualifications:**
- 4+ years ML/AI development
- LLM experience (GPT-4, Claude, open-source)
- RAG implementation
- Python expert
- Prompting & fine-tuning

**Timeline:** Critical hire for January 2026 (cannot delay)

---

#### DevOps/SRE Engineer
**Level:** Mid-level (4-5 years experience)
**Salary Range:** $180-220K
**Key Responsibilities:**
- Kubernetes cluster management
- CI/CD pipeline design
- Monitoring & alerting
- Disaster recovery
- Infrastructure scaling

**Qualifications:**
- 4-5 years DevOps/SRE experience
- Kubernetes expertise
- AWS/Azure/GCP
- Terraform or IaC tooling
- Monitoring tools (Prometheus, Grafana)

**Timeline:** February 2026 (infrastructure bottleneck)

---

#### QA / Test Automation Engineer
**Level:** Mid-level (3-4 years)
**Salary Range:** $120-150K
**Key Responsibilities:**
- Test automation (pytest, Cypress)
- Test coverage improvement
- CI/CD integration
- Performance testing
- Bug triage & severity assessment

**Qualifications:**
- 3+ years QA automation
- Python and JavaScript testing
- Test framework experience
- Performance/load testing tools

**Hiring Schedule:**
- February 2026: 1 hire
- April 2026: 1 hire
- Total Year 1: 2 QA engineers

---

### Non-Engineering Roles

#### VP Customer Success
**Level:** Director (10+ years B2B SaaS)
**Salary Range:** $180-220K + equity
**Start Date:** January 2026
**Key Responsibilities:**
- Customer retention strategy
- Onboarding process design
- Support SLA management
- Customer health monitoring
- Expansion revenue

---

#### VP Sales & Revenue
**Level:** Director (10+ years enterprise SaaS)
**Salary Range:** $150-200K + commission + equity
**Start Date:** February 2026
**Key Responsibilities:**
- Sales strategy & execution
- Enterprise deal closing
- Sales team management
- Revenue forecasting

---

#### Customer Success Manager
**Level:** Mid-level (3-5 years)
**Salary Range:** $90-120K
**Start Date:** May 2026 (first) + October 2026 (second)
**Key Responsibilities:**
- Customer onboarding
- Technical support
- Health score monitoring
- Renewal management

---

#### VP Product
**Level:** Senior (8+ years product)
**Salary Range:** $150-200K + equity
**Start Date:** March 2026
**Key Responsibilities:**
- Product strategy across 6 apps
- Roadmap planning
- Go-to-market strategy
- Feature prioritization

---

## HIRING SUMMARY BY MONTH

```
Month       Role                       Qty  Budget   Cumulative
─────────────────────────────────────────────────────────────
Nov 2025    (Prepare/No new hires)      -      -         12
Dec 2025    Senior Backend (offer)      1   $250K       13
Jan 2026    Backend x2                  2   $360K       15
            ML/LLM Specialist (offer)   1   $230K
            VP Product (offer)          1   $180K
Feb 2026    Backend x1                  1   $170K       20
            Frontend x1                 1   $180K
            VP Customer Success (start) 1   $200K
            Dev/Recruiting Coord        1    $60K
Mar 2026    Backend x1                  1   $170K       25
            DevOps/SRE                  1   $210K
            Climate Specialist          1   $160K
Apr 2026    Backend x1                  1   $170K       32
            Frontend x1                 1   $180K
            ML Engineer (Analytics)     1   $200K
            Head of Marketing (start)   1   $140K
May 2026    Backend x1                  1   $170K       38
            Frontend x1                 1   $180K
            QA Engineer                 1   $140K
            VP Sales (start)            1   $200K
Jun 2026    Backend x1                  1   $170K       43
            CSM #1                      1   $110K
            Marketing Associate         1    $80K
Jul 2026    Backend x1                  1   $170K       45
Aug 2026    Final roles                 -      -         45
Sep 2026    Specialist hires            2   $200K       47
Oct-Dec     Fine-tuning headcount       3   $300K       50
─────────────────────────────────────────────────────────────
TOTAL ADDITIONS                         33  $3.8M       45
```

---

## KEY MILESTONES BY PHASE

### Phase 1 (Nov-Dec 2025): Foundation
- Reorganize current 12-15 engineers
- Lock VCCI scope (prevent scope creep)
- Launch CSRD & CBAM (25+ customers, $50K MRR)
- Begin recruiting (3+ offers extended)

### Phase 2 (Jan-Mar 2026): First Growth Push
- Hire 8-10 engineers
- 20-25 total headcount
- VCCI reaches 90% completion
- App #4 reaches 20% completion
- 120 customers, $240K MRR

### Phase 3 (Apr-Jun 2026): Parallel Builds
- Hire 10-15 engineers
- 35-40 total headcount
- VCCI 100% complete (shipping Oct)
- App #4 60%, App #5 40% complete
- 400 customers, $800K MRR
- v1.0.0 RC ready

### Phase 4 (Jul-Sep 2026): Final Push
- 40-45 engineers (target achieved)
- App #4 95% (Nov launch), App #5 95% (Dec launch)
- App #6 50% (Jan launch)
- 500+ customers, $1M+ MRR
- Infrastructure scaled to 10K concurrent users

### Phase 5 (Oct-Dec 2026): Launches & Optimization
- VCCI launch (Oct)
- App #4 launch (Nov)
- App #5 launch (Dec)
- App #6 continues development
- 750 customers, $18M ARR run rate target
- Series B funding complete

---

## SUCCESS METRICS

**Monthly Tracking (Every 4 weeks):**

| Metric | Nov 2025 | Dec 2025 | Mar 2026 | Jun 2026 | Sep 2026 | Dec 2026 |
|--------|----------|----------|----------|----------|----------|----------|
| **Headcount** | 12 | 13 | 22 | 38 | 43 | 45 |
| **Offers Pending** | 1 | 2 | 1 | 1 | 0 | 0 |
| **Hiring Pace** | 3/mo | 3/mo | 3.3/mo | 5.3/mo | 1.7/mo | 0.7/mo |
| **Customers** | 0 | 30 | 120 | 400 | 500+ | 750 |
| **MRR** | $0 | $50K | $240K | $800K | $1M+ | $1.5M |
| **VCCI %** | 30% | 50% | 90% | 100% | 100% | 100% |
| **App #4 %** | - | - | 20% | 60% | 95% | 95% |
| **App #5 %** | - | - | - | 40% | 95% | 95% |
| **Test Coverage** | 31% | 38% | 55% | 75% | 85% | 85% |

---

## SUMMARY

This plan takes you from ~12 engineers building for traction to **45 world-class engineers executing on a 6-app platform**, generating **$18M+ ARR** by end of 2026, with clear path to **$500M+ by 2030**.

**The key to success:**
1. Start hiring immediately (3-4/month pace)
2. Bring in strong non-technical leadership (VP Product, VP Sales, VP Customer Success)
3. Grow infrastructure team to prevent bottlenecks
4. Lock VCCI scope and maintain execution discipline
5. Hit revenue milestones to validate product-market fit

**You have the architecture, the product, and now the plan. Execute with excellence.**

---

**Document Owner:** CTO
**Last Updated:** November 9, 2025
**Review Cadence:** Monthly
**Next Update:** December 9, 2025
