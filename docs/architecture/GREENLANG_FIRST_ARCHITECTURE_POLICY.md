# GreenLang-First Architecture Policy
## Mandatory Infrastructure Reuse Framework

**Version:** 1.0.0
**Status:** MANDATORY - Effective Immediately
**Scope:** All GreenLang applications, agents, and services
**Authority:** CTO-Approved Strategic Decision
**Date:** 2025-11-09

---

## EXECUTIVE MANDATE

**CRITICAL PIVOT:** All GreenLang development MUST prioritize existing GreenLang infrastructure before considering custom implementations. This is a life-changing decision for the organization's efficiency, maintainability, and time-to-market.

**Violation Consequence:** Code reviews WILL be rejected. CI/CD pipelines WILL fail. Pull requests WILL be blocked.

---

## THE GREENLANG-FIRST PRINCIPLE

```
┌──────────────────────────────────────────────────────────┐
│  DECISION TREE: Can I Use Existing GreenLang Infra?     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. CHECK: Does greenlang.* provide this? ────┐         │
│                                                │         │
│     YES ──> USE IT (MANDATORY)                 │         │
│                                                │         │
│     NO ───> 2. CHECK: Can it be extended? ────┤         │
│                                                │         │
│              YES ──> Extend & contribute back  │         │
│                                                │         │
│              NO ───> 3. Justify in ADR ────────┤         │
│                                                │         │
│                       Approved ──> Build custom│         │
│                       Rejected ──> Use infra   │         │
│                                                │         │
└──────────────────────────────────────────────────────────┘
```

---

## MANDATORY INFRASTRUCTURE COMPONENTS

### 1. LLM INFRASTRUCTURE (99% Coverage Required)

**MUST USE:**
```python
# ✅ CORRECT - Uses GreenLang infrastructure
from greenlang.intelligence.runtime.session import ChatSession
from greenlang.intelligence.rag import RAGEngine
from greenlang.intelligence.semantic_cache import SemanticCache

# ❌ FORBIDDEN - Custom LLM wrappers
class CustomLLMClient:  # WILL BE REJECTED
    def __init__(self, api_key):
        self.openai_client = OpenAI(api_key)  # NO!
```

**Enforcement:**
- Pre-commit hook WILL reject `import openai` or `import anthropic` without greenlang wrapper
- Linter WILL flag custom LLM client classes
- CI/CD WILL fail if `greenlang.intelligence` is not imported in files with LLM logic

**Required Components:**
- `greenlang.intelligence.ChatSession` - All LLM interactions
- `greenlang.intelligence.rag.RAGEngine` - All RAG implementations
- `greenlang.intelligence.semantic_cache` - All LLM caching (30%+ cost savings)
- `greenlang.intelligence.streaming` - All streaming responses
- `greenlang.intelligence.security.PromptGuard` - All prompt security

**Exemptions:** NONE. Zero tolerance.

---

### 2. AGENT FRAMEWORK (95% Coverage Required)

**MUST USE:**
```python
# ✅ CORRECT - Inherits from GreenLang base
from greenlang.sdk.base import Agent, Pipeline
from greenlang.factory import AgentFactory

class MyAgent(Agent):  # REQUIRED
    def execute(self, input_data):
        return self.process(input_data)

# ❌ FORBIDDEN - Custom agent classes from scratch
class MyCustomAgent:  # WILL BE REJECTED
    def __init__(self):
        self.custom_logic = True  # NO!
```

**Enforcement:**
- All agent classes MUST inherit from `greenlang.sdk.base.Agent`
- Pipeline orchestration MUST use `greenlang.sdk.base.Pipeline`
- AgentFactory SHOULD be used for new agent generation
- Custom batch processing loops are FORBIDDEN (use Agent.batch_process)

**Required Components:**
- `greenlang.sdk.base.Agent` - Base class for all agents
- `greenlang.sdk.base.Pipeline` - Pipeline orchestration
- `greenlang.factory.AgentFactory` - Agent code generation
- `greenlang.runtime.executor.Executor` - Agent execution
- `greenlang.agents.categories.AgentMetadata` - Agent categorization

**Exemptions:** Only if base Agent class doesn't support the use case (requires ADR)

---

### 3. DATA STORAGE & CACHING (90% Coverage Required)

**MUST USE:**
```python
# ✅ CORRECT - Uses GreenLang data layer
from greenlang.db import get_engine, get_session
from greenlang.cache import CacheManager

cache = CacheManager()  # REQUIRED
result = cache.get_or_compute(key, expensive_function)

# ❌ FORBIDDEN - Direct database/cache usage
import redis
r = redis.Redis()  # WILL BE REJECTED
```

**Enforcement:**
- Direct `import redis`, `import pymongo` WILL be flagged
- Direct SQLAlchemy engine creation WILL be flagged (use get_engine)
- Files without CacheManager for expensive operations WILL be flagged

**Required Components:**
- `greenlang.cache.CacheManager` - L1/L2/L3 caching
- `greenlang.cache.emission_factor_cache` - Domain-specific caching
- `greenlang.db.get_engine()` - Database connections
- `greenlang.db.get_session()` - Session management
- `greenlang.db.connection.DatabaseConnectionPool` - Connection pooling

**Exemptions:** Performance-critical paths (requires benchmarks in ADR)

---

### 4. API FRAMEWORK (85% Coverage Required)

**MUST USE:**
```python
# ✅ CORRECT - Uses GreenLang GraphQL API
from greenlang.api.graphql import create_graphql_app

app = create_graphql_app()  # REQUIRED

# ❌ FORBIDDEN - Custom FastAPI apps
from fastapi import FastAPI
app = FastAPI()  # WILL BE REJECTED (unless justified)
```

**Enforcement:**
- New APIs MUST use `greenlang.api.graphql`
- REST APIs MUST justify why GraphQL is insufficient (ADR required)
- Custom FastAPI apps require CTO approval

**Required Components:**
- `greenlang.api.graphql.create_graphql_app()` - GraphQL server
- `greenlang.api.websocket` - Real-time streaming
- `greenlang.middleware.rate_limiter` - Rate limiting
- `greenlang.middleware.error_handler` - Error handling

**Exemptions:** Legacy integrations, third-party requirements (requires ADR)

---

### 5. AUTHENTICATION & AUTHORIZATION (100% Coverage Required)

**MUST USE:**
```python
# ✅ CORRECT - Uses GreenLang auth
from greenlang.auth import AuthManager, AuthToken
from greenlang.auth.rbac import check_permission

auth = AuthManager()  # REQUIRED
token = auth.create_token(user_id)

# ❌ FORBIDDEN - Custom JWT, custom RBAC
from jose import jwt
token = jwt.encode(payload, secret)  # WILL BE REJECTED
```

**Enforcement:**
- ZERO tolerance for custom auth implementations
- All auth code MUST use `greenlang.auth`
- Pre-commit hooks WILL block `import jose`, `import jwt`, `import passlib` without greenlang wrapper

**Required Components:**
- `greenlang.auth.AuthManager` - All authentication
- `greenlang.auth.rbac` - All authorization
- `greenlang.auth.tenant` - Multi-tenancy
- `greenlang.auth.audit` - Audit trails
- `greenlang.db.models_auth` - Auth database models

**Exemptions:** NONE. Security is non-negotiable.

---

### 6. VALIDATION & SECURITY (100% Coverage Required)

**MUST USE:**
```python
# ✅ CORRECT - Uses GreenLang validation
from greenlang.validation import ValidationFramework
from greenlang.security.validators import validate_safe_path

validator = ValidationFramework()  # REQUIRED
validator.validate_schema(data, schema)

# ❌ FORBIDDEN - Custom validation
import jsonschema
jsonschema.validate(data, schema)  # WILL BE REJECTED
```

**Enforcement:**
- All validation MUST use `greenlang.validation.ValidationFramework`
- All path validation MUST use `greenlang.security.validators`
- Direct `import jsonschema` WILL be flagged

**Required Components:**
- `greenlang.validation.framework.ValidationFramework` - All validation
- `greenlang.validation.schema.SchemaValidator` - JSON Schema validation
- `greenlang.security.validators` - XSS, SQLi, command injection protection
- `greenlang.security.paths.validate_safe_path()` - Path traversal protection

**Exemptions:** NONE. Security is non-negotiable.

---

### 7. MONITORING & TELEMETRY (90% Coverage Required)

**MUST USE:**
```python
# ✅ CORRECT - Uses GreenLang telemetry
from greenlang.telemetry import MetricsCollector, StructuredLogger
from greenlang.monitoring.health import create_health_app

logger = StructuredLogger(__name__)  # REQUIRED
metrics = MetricsCollector()

# ❌ FORBIDDEN - Custom metrics
from prometheus_client import Counter
counter = Counter('my_metric', 'desc')  # WILL BE FLAGGED
```

**Enforcement:**
- All metrics MUST use `greenlang.telemetry.MetricsCollector`
- All logging MUST use `greenlang.telemetry.StructuredLogger`
- Direct prometheus_client usage requires justification

**Required Components:**
- `greenlang.telemetry.metrics.MetricsCollector` - All metrics
- `greenlang.telemetry.logging.StructuredLogger` - All logging
- `greenlang.telemetry.tracing.TracingManager` - Distributed tracing
- `greenlang.monitoring.health.create_health_app()` - Health checks

**Exemptions:** Custom domain-specific metrics (requires ADR)

---

### 8. CONFIGURATION MANAGEMENT (95% Coverage Required)

**MUST USE:**
```python
# ✅ CORRECT - Uses GreenLang config
from greenlang.config import ConfigManager, ServiceContainer
from greenlang.config.schemas import GreenLangConfig

config = ConfigManager()  # REQUIRED
service_container = ServiceContainer()

# ❌ FORBIDDEN - Custom config loading
import yaml
with open('config.yaml') as f:
    config = yaml.load(f)  # WILL BE FLAGGED
```

**Enforcement:**
- All config MUST use `greenlang.config.ConfigManager`
- Dependency injection MUST use `greenlang.config.ServiceContainer`
- Direct YAML/JSON loading for config is FORBIDDEN

**Required Components:**
- `greenlang.config.manager.ConfigManager` - All configuration
- `greenlang.config.container.ServiceContainer` - Dependency injection
- `greenlang.config.schemas.GreenLangConfig` - Type-safe config
- `greenlang.config.providers.create_provider_from_config()` - Provider factory

**Exemptions:** Data file loading (non-config) is allowed

---

## ENFORCEMENT MECHANISMS

### Level 1: Pre-Commit Hooks (LOCAL)

**File:** `.greenlang/hooks/pre-commit`

Checks before commit:
- ✅ All LLM code uses `greenlang.intelligence`
- ✅ All agents inherit from `greenlang.sdk.base.Agent`
- ✅ All auth uses `greenlang.auth`
- ✅ All validation uses `greenlang.validation`
- ✅ No direct imports of: `openai`, `anthropic`, `redis`, `jose`, `jwt`

**Action on Failure:** BLOCK commit with error message and documentation link

### Level 2: CI/CD Pipeline (REMOTE)

**File:** `.github/workflows/greenlang-first-enforcement.yml`

Checks on PR:
- ✅ Architecture Decision Record (ADR) exists for all custom implementations
- ✅ Infrastructure usage metrics meet thresholds (95%+ for new code)
- ✅ Static analysis passes (no forbidden imports)
- ✅ Code coverage includes infrastructure integration tests

**Action on Failure:** FAIL pipeline, BLOCK merge

### Level 3: OPA Policy Engine (RUNTIME)

**File:** `.greenlang/policies/infrastructure-first.rego`

Runtime policies:
- ✅ All API calls route through greenlang middleware
- ✅ All LLM calls route through ChatSession
- ✅ All cache hits/misses tracked
- ✅ All auth tokens validated by greenlang.auth

**Action on Violation:** LOG alert, DENY operation

### Level 4: Code Review Checklist (HUMAN)

**File:** `.github/PULL_REQUEST_TEMPLATE.md`

Mandatory checklist:
- [ ] I have checked if existing GreenLang infrastructure can be used
- [ ] If custom code is required, I have created an ADR documenting why
- [ ] I have added infrastructure usage metrics to this PR
- [ ] I have updated documentation with infrastructure usage examples
- [ ] All new agents inherit from `greenlang.sdk.base.Agent`
- [ ] All new LLM calls use `greenlang.intelligence.ChatSession`

**Action on Incomplete:** REJECT PR

---

## ARCHITECTURE DECISION RECORDS (ADR)

When custom implementation is required, create an ADR:

**Template:** `.greenlang/adr/TEMPLATE.md`

```markdown
# ADR-XXX: [Title]

## Status
Proposed | Accepted | Rejected | Superseded

## Context
Why is custom implementation being considered?

## Decision
What custom code will be built?

## GreenLang Infrastructure Evaluation
What infrastructure was evaluated and why was it insufficient?

| Component | Evaluated? | Reason for Rejection |
|-----------|------------|----------------------|
| greenlang.intelligence.ChatSession | Yes | [Specific reason] |
| greenlang.cache.CacheManager | Yes | [Specific reason] |

## Consequences
- Maintenance burden: [Estimate LOC, person-hours]
- Performance impact: [Benchmarks]
- Alternative timeline: Using infra would take [X days], custom takes [Y days]

## Approval
- [ ] Tech Lead
- [ ] CTO (required for core infrastructure bypass)
```

**Storage:** `C:\Users\aksha\Code-V1_GreenLang\.greenlang\adr\`

**Process:**
1. Developer creates ADR
2. Tech lead reviews
3. CTO approves if core infrastructure bypass
4. ADR linked in code comments

---

## MIGRATION TIMELINE

### Phase 1: Immediate (Week 1) - ENFORCEMENT
- ✅ Create pre-commit hooks
- ✅ Create CI/CD enforcement
- ✅ Create OPA policies
- ✅ Update PR template
- ✅ Train all developers

### Phase 2: Quick Wins (Month 1) - HIGH-VALUE REFACTORS
- ✅ GL-CSRD-APP: Replace custom LLMClient with ChatSession
- ✅ GL-CSRD-APP: Replace custom RAG with RAGEngine
- ✅ All apps: Add CacheManager for expensive operations
- ✅ All apps: Replace custom validation with ValidationFramework

### Phase 3: Structural (Quarter 1) - AGENT REFACTORS
- ✅ GL-CBAM-APP: Adopt CBAM-Refactored (56.9% LOC reduction)
- ✅ GL-CSRD-APP: Refactor all 6 agents to inherit from Agent base
- ✅ GL-VCCI-APP: Enhance to use GraphQL API
- ✅ All apps: Migrate to greenlang.sdk.base.Pipeline

### Phase 4: Strategic (Quarter 2) - ECOSYSTEM
- ✅ Extract FactorBroker to greenlang.services.factor_broker
- ✅ Extract EntityMDM to greenlang.services.entity_mdm
- ✅ Create agent library (IntakeAgent, CalculatorAgent, ReportingAgent)
- ✅ Deprecate all custom infrastructure

---

## METRICS & ACCOUNTABILITY

### Infrastructure Usage Metrics (IUM)

**Formula:**
```
IUM = (Lines using greenlang.*) / (Total lines) × 100%
```

**Thresholds:**
- New code: ≥95% (MANDATORY)
- Existing apps (post-refactor): ≥70% (TARGET)
- Legacy code: ≥50% (MINIMUM)

**Tracking:**
- Dashboard: `greenlang.monitoring.dashboards.infrastructure_usage`
- Reports: Weekly to CTO
- Alerts: <95% on new PRs

### Team Accountability

**Leaderboard (Monthly):**
1. Highest IUM improvement
2. Most infrastructure contributions
3. Best ADR documentation

**Consequences:**
- Top 3 teams: Recognition, bonuses
- Bottom 3 teams: Mandatory training, code review backlog

---

## EXCEPTIONS & WAIVERS

**Permanent Exceptions:**
1. **Zero Hallucination Requirements:** If calculations MUST be deterministic and LLM-free (e.g., CBAM), document in ADR
2. **Regulatory Compliance:** If regulator mandates specific implementation (e.g., XBRL), document in ADR
3. **Performance Critical:** If benchmarks show >50% performance degradation, optimize infrastructure first, then consider waiver

**Temporary Waivers (6-month max):**
1. **Infrastructure Not Ready:** If feature doesn't exist yet, create GitHub issue, get CTO approval, sunset custom code when ready
2. **Third-Party Integration:** If vendor API requires specific implementation, document in ADR

**NO Waivers For:**
- Authentication/authorization (security)
- Validation (security)
- LLM security (PromptGuard)

---

## DEVELOPER WORKFLOW

### Before Writing Code:

```bash
# 1. Check infrastructure
greenlang search-infra "what I want to build"

# 2. Review documentation
greenlang docs intelligence.ChatSession

# 3. If unsure, ask
greenlang ask "Can I use existing infra for [feature]?"

# 4. If custom needed, create ADR
greenlang adr create "Custom implementation for [reason]"
```

### During Development:

```python
# ALWAYS import from greenlang first
from greenlang.intelligence import ChatSession  # ✅
from greenlang.sdk.base import Agent  # ✅
from greenlang.cache import CacheManager  # ✅

# NOT
from openai import OpenAI  # ❌
class MyCustomAgent:  # ❌
import redis  # ❌
```

### Before Committing:

```bash
# Pre-commit hook will run automatically
# If it fails, fix the issues or create an ADR

git add .
git commit -m "feat: Add new agent"
# Hook runs -> Checks infrastructure usage -> PASS/FAIL
```

### During Code Review:

- Reviewer checks: "Could this use existing infrastructure?"
- Reviewer verifies: ADR exists if custom code present
- Reviewer confirms: IUM ≥95% for new code

---

## SUCCESS CRITERIA

**3 Months:**
- ✅ All new code: 95%+ IUM
- ✅ GL-CBAM-APP: Refactored to 45% custom (from 98%)
- ✅ GL-CSRD-APP: Refactored to 50% custom (from 99%)
- ✅ GL-VCCI-APP: Enhanced to 55% custom (from 75%)
- ✅ Zero security incidents from custom auth/validation

**6 Months:**
- ✅ Codebase reduction: -40,000+ LOC
- ✅ LLM cost reduction: -30% (semantic caching)
- ✅ Developer velocity: +50% (less custom code to maintain)
- ✅ Infrastructure contributions: 100+ PRs to greenlang.*

**12 Months:**
- ✅ All apps: 60%+ IUM
- ✅ Shared services: FactorBroker, EntityMDM, PCFExchange
- ✅ Agent library: 15+ reusable agents
- ✅ Zero custom LLM wrappers, zero custom auth, zero custom validation

---

## COMMUNICATION PLAN

**Week 1: Announcement**
- All-hands meeting: CTO presents vision
- Training sessions: Infrastructure deep-dives
- Documentation: Update all READMEs

**Ongoing:**
- Weekly infrastructure office hours
- Monthly infrastructure showcase
- Quarterly infrastructure roadmap updates

**Feedback:**
- Slack channel: #greenlang-first
- GitHub discussions: Infrastructure requests
- Monthly surveys: Developer satisfaction

---

## APPENDIX: INFRASTRUCTURE CATALOG

**Complete catalog:** See `GREENLANG_INFRASTRUCTURE_CATALOG.md`

**Quick reference:**
- LLM: `greenlang.intelligence.*`
- Agents: `greenlang.sdk.base.*`, `greenlang.factory.*`
- Data: `greenlang.db.*`, `greenlang.cache.*`
- API: `greenlang.api.*`
- Auth: `greenlang.auth.*`
- Security: `greenlang.security.*`
- Validation: `greenlang.validation.*`
- Monitoring: `greenlang.telemetry.*`, `greenlang.monitoring.*`
- Config: `greenlang.config.*`

---

## VERSION HISTORY

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-09 | CTO | Initial policy creation - MANDATORY enforcement |

---

**END OF POLICY**

**Questions? Contact:**
- Slack: #greenlang-first
- Email: architecture@greenlang.io
- Office Hours: Every Friday 2-4 PM

**This policy is MANDATORY and EFFECTIVE IMMEDIATELY.**
