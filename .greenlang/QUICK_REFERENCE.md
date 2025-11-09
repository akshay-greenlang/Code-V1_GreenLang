# GreenLang Infrastructure-First Quick Reference

**TL;DR:** Use GreenLang infrastructure first. Custom code requires ADR.

---

## The Golden Rule

> **Use `greenlang.*` imports. Only write custom code when infrastructure cannot support your use case, and document why via ADR.**

---

## Quick Checks

### Before Committing

```bash
# Check for violations
python .greenlang/linters/infrastructure_first.py --path .

# Check your score
python .greenlang/scripts/calculate_ium.py

# Commit (pre-commit hook runs automatically)
git commit -m "Your message"
```

### Before Creating PR

```bash
# Ensure IUM >= 95%
python .greenlang/scripts/calculate_ium.py

# If <95%, either:
# 1. Fix violations, OR
# 2. Create ADR for custom code
```

---

## Forbidden → Allowed Mapping

| ❌ DON'T USE | ✅ USE INSTEAD |
|-------------|---------------|
| `import openai` | `from greenlang.intelligence import ChatSession` |
| `import anthropic` | `from greenlang.intelligence import ChatSession` |
| `import redis` | `from greenlang.cache import CacheManager` |
| `import pymongo` | `from greenlang.db import get_connector` |
| `from jose import jwt` | `from greenlang.auth import AuthManager` |
| `import passlib` | `from greenlang.auth import AuthManager` |
| `class MyAgent:` | `from greenlang.sdk.base import Agent` |

---

## Common Patterns

### LLM Calls

```python
# ❌ WRONG
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(...)

# ✅ RIGHT
from greenlang.intelligence import ChatSession
session = ChatSession()
response = session.chat("Your prompt")
```

### Agents

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
        # Your logic
        return result
```

### Auth

```python
# ❌ WRONG
from jose import jwt
token = jwt.encode({"sub": user_id}, SECRET)

# ✅ RIGHT
from greenlang.auth import AuthManager
auth = AuthManager()
token = auth.create_token(user_id)
```

### Caching

```python
# ❌ WRONG
import redis
r = redis.Redis()
r.set('key', 'value')

# ✅ RIGHT
from greenlang.cache import CacheManager
cache = CacheManager()
cache.set('key', 'value')
```

---

## Creating an ADR

When you MUST use custom code:

```bash
# 1. Copy template
cp .greenlang/adrs/TEMPLATE.md .greenlang/adrs/20241109-my-case.md

# 2. Fill in ALL sections
vim .greenlang/adrs/20241109-my-case.md

# 3. Get 2+ approvals

# 4. Update status to "Accepted"

# 5. Reference in code
# ADR-001: Custom implementation for X
# See: .greenlang/adrs/20241109-my-case.md

# 6. Reference in PR description
```

---

## Fixing Violations

### Step 1: See the violation

```bash
python .greenlang/linters/infrastructure_first.py
```

Output:
```
✗ my_file.py:15
  Forbidden import: 'openai'
  → Use greenlang.intelligence.ChatSession instead
```

### Step 2: Fix it

```python
# Replace this
import openai

# With this
from greenlang.intelligence import ChatSession
```

### Step 3: Verify

```bash
python .greenlang/linters/infrastructure_first.py
# Should show: ✓ No violations found
```

---

## IUM Score Interpretation

- **95-100%** ✅ Excellent, fully compliant
- **85-94%** ⚠️ Good, some custom code
- **75-84%** ⚠️ Needs improvement
- **<75%** ❌ Review custom code

```bash
# See your score
python .greenlang/scripts/calculate_ium.py

# See detailed breakdown
python .greenlang/scripts/calculate_ium.py --output markdown
```

---

## Installation (One-Time)

```bash
# Quick install
bash .greenlang/scripts/install_enforcement.sh

# Verify
.git/hooks/pre-commit
opa test .greenlang/policies/infrastructure-first.rego
```

---

## Common Commands

```bash
# Lint entire codebase
python .greenlang/linters/infrastructure_first.py

# Lint specific app
python .greenlang/linters/infrastructure_first.py --path apps/GL-CBAM-APP

# Calculate IUM
python .greenlang/scripts/calculate_ium.py

# Calculate IUM for specific app
python .greenlang/scripts/calculate_ium.py --app GL-CBAM-APP

# JSON output (for CI)
python .greenlang/linters/infrastructure_first.py --output json

# Test OPA policy
opa test .greenlang/policies/infrastructure-first.rego
```

---

## PR Checklist

Before submitting PR, ensure:

- [ ] Ran linter (0 violations)
- [ ] IUM >= 95% OR ADR created
- [ ] All agents inherit from Agent
- [ ] All LLM calls use ChatSession
- [ ] All auth uses greenlang.auth
- [ ] Pre-commit hook passed

---

## Getting Help

- **Documentation:** `.greenlang/ENFORCEMENT_GUIDE.md`
- **Examples:** `.greenlang/adrs/EXAMPLE-*.md`
- **Issues:** GitHub with `enforcement` label
- **Questions:** #greenlang-infrastructure Slack

---

## Bypass Process

**Only if absolutely necessary:**

1. Create ADR documenting why
2. Get 2+ team approvals
3. Reference ADR in code and PR
4. Plan migration back to infrastructure

**Do NOT:**
- Skip enforcement without ADR
- Commit with violations
- Ignore pre-commit hook failures

---

## Key Files

- **Enforcement Guide:** `.greenlang/ENFORCEMENT_GUIDE.md`
- **Pre-commit Hook:** `.git/hooks/pre-commit`
- **Linter:** `.greenlang/linters/infrastructure_first.py`
- **IUM Calculator:** `.greenlang/scripts/calculate_ium.py`
- **ADR Template:** `.greenlang/adrs/TEMPLATE.md`
- **OPA Policy:** `.greenlang/policies/infrastructure-first.rego`

---

## Quick Win Examples

### Win 1: Replace OpenAI

```diff
- from openai import OpenAI
- client = OpenAI()
- response = client.chat.completions.create(model="gpt-4", messages=[...])

+ from greenlang.intelligence import ChatSession
+ session = ChatSession(model="gpt-4")
+ response = session.chat("Your prompt")
```

**Impact:** +15% IUM score

### Win 2: Fix Agent Inheritance

```diff
- class MyAgent:
-     def process(self, data):
-         return data

+ from greenlang.sdk.base import Agent
+ class MyAgent(Agent):
+     def validate(self, input_data):
+         return True
+     def process(self, input_data):
+         return input_data
```

**Impact:** +20% IUM score

### Win 3: Use CacheManager

```diff
- import redis
- r = redis.Redis()
- r.setex('key', 3600, 'value')

+ from greenlang.cache import CacheManager
+ cache = CacheManager()
+ cache.set('key', 'value', ttl=3600)
```

**Impact:** +10% IUM score

---

**Last Updated:** 2024-11-09
**Version:** 1.0
**Print This:** Keep at your desk!
