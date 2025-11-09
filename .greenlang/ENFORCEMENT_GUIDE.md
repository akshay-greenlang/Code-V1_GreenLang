# GreenLang Infrastructure-First Enforcement Guide

**Version:** 1.0
**Last Updated:** 2024-11-09
**Status:** Active

## Table of Contents

1. [Overview](#overview)
2. [Philosophy](#philosophy)
3. [Enforcement Mechanisms](#enforcement-mechanisms)
4. [How to Comply](#how-to-comply)
5. [Bypass Process (ADR)](#bypass-process-adr)
6. [Common Violations & Fixes](#common-violations--fixes)
7. [Troubleshooting](#troubleshooting)
8. [FAQs](#faqs)

---

## Overview

The **Infrastructure-First** principle ensures all GreenLang applications use shared, tested, and optimized infrastructure rather than custom implementations. This guide explains how enforcement works and how to comply.

### Key Benefits

- **Consistency:** All apps use the same patterns
- **Quality:** Infrastructure is tested and optimized
- **Security:** Centralized security controls
- **Maintainability:** Updates propagate automatically
- **Performance:** Shared optimizations (caching, retries, etc.)

### Enforcement Layers

1. **Pre-commit hooks** - Local validation before commit
2. **Static analysis** - AST-based code scanning
3. **GitHub Actions** - PR validation and reporting
4. **OPA policies** - Runtime enforcement
5. **Infrastructure Usage Metrics (IUM)** - Compliance scoring

---

## Philosophy

### The Golden Rule

> **Use GreenLang infrastructure first. Only implement custom solutions when infrastructure cannot support your use case, and document why via ADR.**

### What This Means

- Use `greenlang.intelligence.ChatSession` instead of OpenAI/Anthropic clients
- Use `greenlang.auth.AuthManager` instead of custom JWT/password handling
- Use `greenlang.cache.CacheManager` instead of direct Redis
- Use `greenlang.sdk.base.Agent` for all agent classes
- Use `greenlang.db` connectors instead of direct pymongo/SQLAlchemy

### When Custom Code Is OK

- **After documenting via ADR** - Explain why infrastructure can't work
- **With team approval** - Get 2+ reviewers to approve
- **With monitoring** - Custom code requires extra logging/metrics
- **With plan to migrate** - Custom code should be temporary

---

## Enforcement Mechanisms

### 1. Pre-Commit Hook

**Location:** `.git/hooks/pre-commit`
**Runs:** Before every commit (local)

**Checks:**
- ❌ Forbidden imports (openai, anthropic, redis, pymongo, jose, jwt, passlib)
- ✅ Required imports when LLM code detected (greenlang.intelligence)
- ✅ Required imports when auth code detected (greenlang.auth)
- ✅ All Agent classes inherit from `greenlang.sdk.base.Agent`
- ✅ ADR exists if custom implementation detected

**Installation:**
```bash
cp .greenlang/hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

**Output Example:**
```
================================================================================
GreenLang Infrastructure-First Pre-Commit Check
================================================================================

Checking 3 Python file(s)...

================================================================================
INFRASTRUCTURE-FIRST VIOLATIONS DETECTED
================================================================================

✗ agents/custom_agent.py:15
  Agent class 'CustomAgent' does not inherit from greenlang.sdk.base.Agent
  → Change to: class CustomAgent(Agent) and import: from greenlang.sdk.base import Agent

✗ services/llm_service.py:8
  Forbidden import: 'openai'
  → Use greenlang.intelligence.ChatSession instead

================================================================================
COMMIT BLOCKED - Fix violations above
================================================================================
```

### 2. Static Analysis Linter

**Location:** `.greenlang/linters/infrastructure_first.py`
**Runs:** On-demand or in CI

**Usage:**
```bash
# Lint entire codebase
python .greenlang/linters/infrastructure_first.py

# Lint specific directory
python .greenlang/linters/infrastructure_first.py --path apps/my-app

# JSON output
python .greenlang/linters/infrastructure_first.py --output json > violations.json
```

**Violation Codes:**
- `FORBIDDEN_IMPORT` - Direct import of forbidden module
- `CUSTOM_AGENT` - Agent class not inheriting from greenlang.sdk.base.Agent
- `CUSTOM_LLM` - Custom LLM client detected
- `CUSTOM_AUTH` - Custom auth implementation detected
- `DIRECT_DB` - Direct database access detected

### 3. GitHub Actions Workflow

**Location:** `.github/workflows/greenlang-first-enforcement.yml`
**Runs:** On every PR and push to main

**Steps:**
1. Run static analysis (forbidden imports)
2. Calculate Infrastructure Usage Metrics (IUM)
3. Check for ADRs if custom code detected
4. Run OPA policy validation
5. Generate comprehensive report
6. Comment on PR with results
7. Fail PR if violations found and no ADR

**Success Criteria:**
- No static analysis violations OR
- IUM >= 95% OR
- ADR exists and approved

### 4. OPA Policy

**Location:** `.greenlang/policies/infrastructure-first.rego`
**Runs:** At runtime (if OPA integrated)

**Enforces:**
- All API calls must have greenlang auth token
- All LLM calls must route through ChatSession
- All cache operations must use CacheManager
- All agent executions must use proper Agent base class

**Test Policy:**
```bash
opa test .greenlang/policies/infrastructure-first.rego
```

### 5. Infrastructure Usage Metrics (IUM)

**Location:** `.greenlang/scripts/calculate_ium.py`
**Runs:** On-demand or in CI

**Calculates:**
- Import compliance (greenlang vs external)
- Agent inheritance compliance
- LLM call compliance
- Auth operation compliance
- Cache operation compliance
- Database operation compliance

**Overall IUM Score:** Weighted average of all categories

**Usage:**
```bash
# Calculate overall IUM
python .greenlang/scripts/calculate_ium.py

# Calculate for specific app
python .greenlang/scripts/calculate_ium.py --app GL-CBAM-APP

# JSON output
python .greenlang/scripts/calculate_ium.py --output json --output-file ium.json

# Markdown report
python .greenlang/scripts/calculate_ium.py --output markdown --markdown-file ium.md
```

---

## How to Comply

### Step 1: Know the Infrastructure

**Intelligence (LLM):**
```python
# ❌ WRONG
from openai import OpenAI
client = OpenAI(api_key="...")
response = client.chat.completions.create(...)

# ✅ RIGHT
from greenlang.intelligence import ChatSession

session = ChatSession()
response = session.chat("Your prompt here")
```

**Auth:**
```python
# ❌ WRONG
from jose import jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"])
hashed = pwd_context.hash(password)

# ✅ RIGHT
from greenlang.auth import AuthManager

auth = AuthManager()
hashed = auth.hash_password(password)
token = auth.create_token(user_id)
```

**Caching:**
```python
# ❌ WRONG
import redis
r = redis.Redis(host='localhost')
r.set('key', 'value')

# ✅ RIGHT
from greenlang.cache import CacheManager

cache = CacheManager()
cache.set('key', 'value')
```

**Agents:**
```python
# ❌ WRONG
class MyAgent:
    def process(self, data):
        pass

# ✅ RIGHT
from greenlang.sdk.base import Agent

class MyAgent(Agent):
    def validate(self, input_data):
        return True

    def process(self, input_data):
        # Your logic here
        return result
```

**Database:**
```python
# ❌ WRONG
from pymongo import MongoClient
client = MongoClient('mongodb://localhost')

# ✅ RIGHT
from greenlang.db import get_connector

db = get_connector('mongodb')
```

### Step 2: Check Before Commit

```bash
# Run linter
python .greenlang/linters/infrastructure_first.py --path .

# Calculate IUM
python .greenlang/scripts/calculate_ium.py

# Commit (pre-commit hook runs automatically)
git add .
git commit -m "feat: Add new feature"
```

### Step 3: Fix Violations

Use the suggestions in violation messages:

```
✗ Line 25:4 [FORBIDDEN_IMPORT] Direct import of 'openai' is forbidden
  → Use greenlang.intelligence instead. Use ChatSession for LLM calls
```

### Step 4: Submit PR

The PR checklist will remind you:

- [x] I checked if GreenLang infrastructure can be used
- [x] No forbidden imports
- [x] All agents inherit from greenlang.sdk.base.Agent
- [x] All LLM calls use greenlang.intelligence.ChatSession
- [x] All auth uses greenlang.auth

---

## Bypass Process (ADR)

### When You Need Custom Code

Sometimes GreenLang infrastructure cannot support your use case. In these cases:

1. **Document why** via Architecture Decision Record (ADR)
2. **Get approval** from 2+ team members
3. **Reference ADR** in your PR
4. **Plan migration** if custom code is temporary

### Creating an ADR

**Location:** `.greenlang/adrs/YYYYMMDD-short-title.md`

**Template:**
```markdown
# ADR-001: Custom LLM Provider for Specialized Model

Date: 2024-11-09
Status: Proposed

## Context

We need to use a specialized climate modeling LLM that is not available
through standard OpenAI/Anthropic APIs supported by greenlang.intelligence.

## Decision

Implement custom client for ClimateGPT API while maintaining greenlang
patterns for monitoring, budgets, and error handling.

## Consequences

### Positive
- Access to specialized climate model
- Better accuracy for domain-specific tasks

### Negative
- Custom code to maintain
- Won't benefit from greenlang.intelligence updates automatically
- Need to implement own retry/fallback logic

## Alternatives Considered

### 1. Use greenlang.intelligence with fine-tuning
**Rejected:** Fine-tuning Claude/GPT-4 would be too expensive

### 2. Contribute ClimateGPT provider to greenlang
**Future:** We plan to contribute this back to core infrastructure

## Implementation Plan

1. Create custom client in `services/climate_llm.py`
2. Implement greenlang-compatible interfaces
3. Add comprehensive logging and metrics
4. Document in team wiki
5. Create issue to migrate to greenlang.intelligence when supported

## Review

Approved by:
- @alice (Engineering Lead)
- @bob (Security)

## Migration Plan

When ClimateGPT is added to greenlang.intelligence (estimated Q2 2025),
migrate to standard ChatSession and deprecate custom client.
```

### ADR Approval Process

1. **Create ADR** in `.greenlang/adrs/`
2. **Submit PR** with ADR
3. **Get reviews** from 2+ team members
4. **Update status** to "Accepted" after approval
5. **Reference ADR** in code comments:

```python
# ADR-001: Custom LLM client for ClimateGPT
# See: .greenlang/adrs/20241109-climate-gpt-client.md
from services.climate_llm import ClimateGPTClient
```

### ADR Override in OPA

If you have an approved ADR, you can provide override metadata:

```python
{
    "type": "llm_call",
    "metadata": {
        "caller": "custom.climate_llm.Client"
    },
    "override": {
        "adr_exists": True,
        "adr_approved": True,
        "adr_id": "ADR-001",
        "reason": "Specialized climate model not in greenlang"
    },
    "user": "engineer@company.com"
}
```

---

## Common Violations & Fixes

### Violation 1: Forbidden Import (openai)

**Error:**
```
✗ services/llm.py:3
  Forbidden import: 'openai'
  → Use greenlang.intelligence.ChatSession instead
```

**Fix:**
```python
# Before
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

# After
from greenlang.intelligence import ChatSession

session = ChatSession(model="gpt-4")
response = session.chat(prompt)
```

### Violation 2: Custom Agent Class

**Error:**
```
✗ agents/custom.py:10
  Agent class 'CustomAgent' does not inherit from greenlang.sdk.base.Agent
  → Add 'from greenlang.sdk.base import Agent' and inherit from Agent
```

**Fix:**
```python
# Before
class CustomAgent:
    def __init__(self):
        self.name = "custom"

    def run(self, data):
        return self.process(data)

# After
from greenlang.sdk.base import Agent

class CustomAgent(Agent):
    def validate(self, input_data):
        # Add validation logic
        return True

    def process(self, input_data):
        # Your processing logic
        return result
```

### Violation 3: Direct Redis Usage

**Error:**
```
✗ services/cache.py:5
  Forbidden import: 'redis'
  → Use greenlang.cache.CacheManager instead
```

**Fix:**
```python
# Before
import redis

r = redis.Redis(host='localhost', port=6379, db=0)
r.setex('key', 3600, 'value')
value = r.get('key')

# After
from greenlang.cache import CacheManager

cache = CacheManager()
cache.set('key', 'value', ttl=3600)
value = cache.get('key')
```

### Violation 4: Custom JWT Handling

**Error:**
```
✗ auth/jwt_service.py:2
  Forbidden import: 'from jose import jwt'
  → Use greenlang.auth.AuthManager instead
```

**Fix:**
```python
# Before
from jose import jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=30)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# After
from greenlang.auth import AuthManager

auth = AuthManager()

def create_access_token(user_id: str):
    return auth.create_token(user_id, expires_in=1800)
```

### Violation 5: LLM Code Without greenlang.intelligence

**Error:**
```
✗ services/generator.py:1
  LLM-related code detected but greenlang.intelligence not imported
  → Add: from greenlang.intelligence import ChatSession
```

**Fix:**
```python
# Before
def generate_report(data):
    # Uses LLM somehow but doesn't import greenlang.intelligence
    prompt = f"Generate report for {data}"
    # ... custom LLM logic

# After
from greenlang.intelligence import ChatSession

def generate_report(data):
    session = ChatSession()
    prompt = f"Generate report for {data}"
    return session.chat(prompt)
```

---

## Troubleshooting

### Pre-commit Hook Not Running

**Problem:** Committing without seeing enforcement checks

**Solutions:**
1. Verify hook is installed:
   ```bash
   ls -la .git/hooks/pre-commit
   ```

2. Reinstall if missing:
   ```bash
   cp .greenlang/hooks/pre-commit .git/hooks/pre-commit
   chmod +x .git/hooks/pre-commit
   ```

3. Test manually:
   ```bash
   .git/hooks/pre-commit
   ```

### False Positives

**Problem:** Linter flags code that is actually compliant

**Solutions:**
1. Check if you're using the right imports:
   ```python
   # This might be flagged as non-compliant
   from greenlang.intelligence.providers.openai import OpenAIProvider

   # Use this instead
   from greenlang.intelligence import ChatSession
   ```

2. If genuinely a false positive, create an ADR to document exception

### IUM Score Too Low

**Problem:** IUM score is below 95%

**Solutions:**
1. Run detailed report:
   ```bash
   python .greenlang/scripts/calculate_ium.py --output markdown
   ```

2. Identify low-scoring categories
3. Prioritize fixing:
   - Agent inheritance (weight: 3)
   - LLM calls (weight: 3)
   - Imports (weight: 2)
   - Auth (weight: 2)
   - Cache (weight: 1)
   - Database (weight: 1)

4. Create ADR for legitimate custom implementations

### GitHub Actions Failing

**Problem:** PR blocked by enforcement workflow

**Solutions:**
1. Check workflow output for specific violations
2. Run linter locally:
   ```bash
   python .greenlang/linters/infrastructure_first.py
   ```
3. Fix violations or create ADR
4. Re-push to trigger workflow again

### OPA Policy Rejecting Valid Code

**Problem:** Runtime policy blocking legitimate operations

**Solutions:**
1. Verify you're using greenlang infrastructure correctly
2. Check metadata being passed to OPA
3. If you have approved ADR, include override metadata:
   ```python
   {
       "override": {
           "adr_exists": True,
           "adr_approved": True,
           "adr_id": "ADR-XXX"
       }
   }
   ```

---

## FAQs

### Q: Why is this enforced?

**A:** To ensure consistency, quality, security, and maintainability across all GreenLang applications. Shared infrastructure is tested, optimized, and maintained by the core team.

### Q: What if greenlang doesn't support my use case?

**A:** Create an ADR documenting why you need custom code. Get approval from 2+ team members. Reference the ADR in your PR.

### Q: Can I temporarily bypass enforcement during development?

**A:** No. Enforcement runs on every commit and PR. If you need custom code, create an ADR first. This ensures technical debt is documented.

### Q: How do I contribute new features to greenlang infrastructure?

**A:**
1. Create a feature request issue
2. Discuss with core team
3. Submit PR to `core/greenlang/`
4. Once merged, update your app code to use it

### Q: What's the difference between WARNING and ERROR violations?

**A:**
- **ERROR:** Blocks commit/PR, must fix or create ADR
- **WARNING:** Informational, should fix but won't block

### Q: How is IUM calculated?

**A:** Weighted average of:
- Imports (weight 2)
- Agent inheritance (weight 3)
- LLM calls (weight 3)
- Auth operations (weight 2)
- Cache operations (weight 1)
- Database operations (weight 1)

### Q: Do I need 100% IUM?

**A:** No. 95%+ is the target. Some custom code is expected for specialized use cases (documented via ADR).

### Q: Can I disable enforcement for specific files?

**A:** No. Enforcement is codebase-wide. If a file needs custom code, document via ADR.

### Q: What if I disagree with the infrastructure-first approach?

**A:** Discuss with the architecture team. The approach is based on proven practices for large-scale systems, but feedback is welcome.

### Q: How long does an ADR approval take?

**A:** Usually 1-2 business days. Tag relevant reviewers in your PR.

---

## Support

**Issues:** Create issue in GitHub with `enforcement` label
**Questions:** Ask in #greenlang-infrastructure Slack channel
**Documentation:** [GreenLang Infrastructure Docs](../docs/INFRASTRUCTURE.md)

---

**Last Updated:** 2024-11-09
**Version:** 1.0
**Maintained by:** GreenLang Infrastructure Team
