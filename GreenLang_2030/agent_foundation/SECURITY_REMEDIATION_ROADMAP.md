# GreenLang Agent Foundation - Security Remediation Roadmap

**Project:** Security Vulnerability Remediation
**Status:** CRITICAL - PRODUCTION BLOCKED
**Start Date:** 2025-01-15
**Target Completion:** 2025-01-30 (15 days)
**Team Size:** 3-4 developers
**Total Effort:** 80 hours (2 weeks)

---

## Executive Summary

This roadmap provides a prioritized, actionable plan to remediate all security vulnerabilities identified in the comprehensive security scan. The plan is structured in 4 phases with clear ownership, timelines, and acceptance criteria.

**Critical Path:** CRITICAL issues (Phase 1) MUST be completed before any production deployment.

---

## Phase 1: CRITICAL BLOCKERS (Days 1-2) - PRODUCTION BLOCKING

**Deadline:** 2025-01-17 (48 hours)
**Team:** 2 senior developers + 1 QA engineer
**Status:** BLOCKING PRODUCTION DEPLOYMENT

### Objective
Eliminate all 6 critical security vulnerabilities that allow remote code execution or authentication bypass.

### Issues to Fix

#### CRITICAL-001: eval() in reasoning.py
**File:** `capabilities/reasoning.py:1596`
**Owner:** Dev 1
**Effort:** 15 minutes
**Priority:** P0

**Current Code:**
```python
source_dict = eval(source)  # In production, use safe evaluation
```

**Fixed Code:**
```python
import ast
try:
    source_dict = ast.literal_eval(source)
except (ValueError, SyntaxError) as e:
    logger.error(f"Invalid source format: {e}")
    return None
```

**Testing:**
```python
# Test safe evaluation
assert _extract_solution("{'solution': 42}") == 42

# Test attack blocked
result = _extract_solution("__import__('os').system('whoami')")
assert result is None  # Attack blocked
```

**Acceptance Criteria:**
- [ ] eval() replaced with ast.literal_eval()
- [ ] Error handling added
- [ ] Unit tests pass
- [ ] Attack test confirms exploit blocked

---

#### CRITICAL-002: eval() in routing.py
**File:** `orchestration/routing.py:94`
**Owner:** Dev 1
**Effort:** 30 minutes
**Priority:** P0

**Current Code:**
```python
return eval(self.condition, {"__builtins__": {}}, context)
```

**Fixed Code:**
```python
from simpleeval import simple_eval, NameNotDefined, InvalidExpression

try:
    return simple_eval(self.condition, names=context)
except (SyntaxError, NameNotDefined, InvalidExpression) as e:
    logger.error(f"Invalid routing condition: {e}")
    return False
```

**Testing:**
```python
# Test safe expressions
rule = RouteRule(name="test", condition="priority > 5")
assert rule.evaluate(Message(priority=10)) == True

# Test attack blocked
rule = RouteRule(name="attack", condition="__import__('os').system('whoami')")
assert rule.evaluate(Message()) == False  # Attack blocked
```

**Acceptance Criteria:**
- [ ] eval() replaced with simpleeval
- [ ] All routing tests pass
- [ ] Performance benchmark shows <5ms overhead
- [ ] Attack test confirms exploit blocked

---

#### CRITICAL-003: eval() in pipeline.py
**File:** `orchestration/pipeline.py:604`
**Owner:** Dev 1
**Effort:** 30 minutes
**Priority:** P0

**Same fix as CRITICAL-002**

**Acceptance Criteria:**
- [ ] eval() replaced with simpleeval
- [ ] All pipeline tests pass
- [ ] Attack test confirms exploit blocked

---

#### CRITICAL-004: pickle.loads() in task_executor.py
**File:** `capabilities/task_executor.py:816`
**Owner:** Dev 2
**Effort:** 1 hour
**Priority:** P0

**Current Code:**
```python
return pickle.loads(data)
```

**Fixed Code (Option 1 - JSON):**
```python
import json

try:
    text_data = data.decode('utf-8')
    return json.loads(text_data)
except (UnicodeDecodeError, json.JSONDecodeError) as e:
    logger.error(f"Failed to load checkpoint: {e}")
    return None
```

**Fixed Code (Option 2 - Signed Pickle - if complex objects needed):**
```python
import hmac
import hashlib
import pickle
import os

SECRET_KEY = os.getenv("CHECKPOINT_SIGNING_KEY")
if not SECRET_KEY:
    raise ValueError("CHECKPOINT_SIGNING_KEY environment variable required")

# Extract signature and payload
if len(data) < 32:
    raise ValueError("Checkpoint data too short")

signature = data[:32]
payload = data[32:]

# Verify HMAC signature
expected_sig = hmac.new(
    SECRET_KEY.encode(),
    payload,
    hashlib.sha256
).digest()

if not hmac.compare_digest(signature, expected_sig):
    raise ValueError("Checkpoint signature invalid - possible tampering detected")

# Only deserialize if signature valid
return pickle.loads(payload)
```

**Save Code Update:**
```python
# When saving checkpoint
import hmac
import hashlib
import pickle

payload = pickle.dumps(checkpoint_data)
signature = hmac.new(
    SECRET_KEY.encode(),
    payload,
    hashlib.sha256
).digest()

data = signature + payload
await f.write(data)
```

**Testing:**
```python
# Test legitimate checkpoint
checkpoint = {"task_id": "123", "progress": 0.5}
saved = save_checkpoint(checkpoint)
loaded = load_checkpoint(saved)
assert loaded == checkpoint

# Test tampered checkpoint blocked
tampered = saved[:32] + b"malicious" + saved[41:]
try:
    load_checkpoint(tampered)
    assert False, "Should have raised exception"
except ValueError as e:
    assert "signature invalid" in str(e)

# Test RCE exploit blocked
import pickle, os
class Exploit:
    def __reduce__(self):
        return (os.system, ('echo PWNED',))

malicious = pickle.dumps(Exploit())
# This should fail signature verification
try:
    load_checkpoint(malicious)
    assert False, "Should have raised exception"
except ValueError:
    pass  # GOOD
```

**Acceptance Criteria:**
- [ ] pickle.loads() secured with HMAC signature
- [ ] Environment variable CHECKPOINT_SIGNING_KEY configured
- [ ] All checkpoint tests pass
- [ ] RCE exploit test confirms attack blocked
- [ ] Documentation updated

---

#### CRITICAL-005: pickle.load() in meta_cognition.py
**File:** `capabilities/meta_cognition.py:1190`
**Owner:** Dev 2
**Effort:** 1 hour
**Priority:** P0

**Same fix as CRITICAL-004**

**Acceptance Criteria:**
- [ ] pickle.load() secured with HMAC signature
- [ ] All meta-cognition tests pass
- [ ] RCE exploit test confirms attack blocked

---

#### CRITICAL-006: JWT Signature Verification Disabled
**File:** `auth/oauth.py:495`
**Owner:** Dev 1
**Effort:** 30 minutes
**Priority:** P0

**Current Code:**
```python
def decode_token_without_validation(self, token: str) -> Dict[str, Any]:
    """
    Warning: DO NOT use for authentication - validation is disabled
    """
    payload = jwt.decode(
        token,
        options={"verify_signature": False, "verify_exp": False}
    )
    return payload
```

**Fixed Code:**
```python
import os

def decode_token_without_validation(self, token: str) -> Dict[str, Any]:
    """
    INTERNAL DEBUG ONLY - NEVER EXPOSE TO PRODUCTION CODE

    This method disables JWT signature verification and is ONLY for:
    - Debugging token contents in development
    - Logging token claims (non-security)
    - Development tooling

    Raises:
        NotImplementedError: In production environments
        SecurityWarning: In non-production with audit logging

    NEVER use this method for authentication/authorization.
    ALWAYS use validate_token() for security decisions.
    """
    # Hard block in production
    environment = os.getenv("ENVIRONMENT", "production")
    if environment == "production":
        raise NotImplementedError(
            "decode_token_without_validation() is DISABLED in production. "
            "This method bypasses signature verification and MUST NOT be used "
            "for authentication. Use validate_token() instead."
        )

    # Log usage in non-production for security audit
    logger.warning(
        "JWT signature validation BYPASSED (DEBUG ONLY)",
        extra={
            "method": "decode_token_without_validation",
            "environment": environment,
            "stack_trace": "".join(traceback.format_stack())
        }
    )

    try:
        payload = jwt.decode(
            token,
            options={"verify_signature": False, "verify_exp": False}
        )
        return payload
    except Exception as e:
        logger.error(f"Failed to decode token: {e}")
        raise
```

**Code Scanning Rule:**
```python
# Add to factory/validation.py:
SECURITY_PATTERNS = [
    (r'decode_token_without_validation',
     "NEVER use for authentication - signature verification disabled",
     50),
]
```

**Testing:**
```python
# Test production blocking
os.environ["ENVIRONMENT"] = "production"
try:
    oauth.decode_token_without_validation(token)
    assert False, "Should have raised NotImplementedError"
except NotImplementedError as e:
    assert "DISABLED in production" in str(e)

# Test development warning
os.environ["ENVIRONMENT"] = "development"
with captured_logs() as logs:
    oauth.decode_token_without_validation(token)
    assert "BYPASSED" in logs
    assert "DEBUG ONLY" in logs
```

**Acceptance Criteria:**
- [ ] Production blocker added
- [ ] Development logging added
- [ ] Code scanning rule added
- [ ] Unit tests pass
- [ ] Production test confirms method blocked
- [ ] Documentation updated

---

### Phase 1 Summary

**Total Effort:** 3.5 hours development + 4 hours testing = 7.5 hours
**Timeline:** 2 days (includes code review, testing, deployment)
**Deliverables:**
- All 6 critical vulnerabilities fixed
- Unit tests for all fixes
- Attack tests confirming exploits blocked
- Code review approval from security lead
- Documentation updated

**Phase 1 Exit Criteria:**
- [ ] All CRITICAL issues resolved
- [ ] All unit tests passing
- [ ] All attack tests confirm exploits blocked
- [ ] Code review approved by 2 senior developers
- [ ] Security lead sign-off obtained
- [ ] Changes deployed to staging environment
- [ ] Staging security scan shows 0 critical issues

**PRODUCTION DEPLOYMENT GATE:** Phase 1 MUST be 100% complete before production deployment.

---

## Phase 2: HIGH PRIORITY (Days 3-7) - SECURITY HARDENING

**Deadline:** 2025-01-22 (5 days after Phase 1)
**Team:** 3 developers + 1 QA engineer
**Status:** SECURITY HARDENING

### Objective
Eliminate high-severity vulnerabilities and replace weak cryptography.

### Issues to Fix

#### HIGH-001: Replace MD5 with BLAKE2/SHA-256 (20+ instances)
**Owner:** Dev 3
**Effort:** 2 hours
**Priority:** P1

**Files to Update:**
- `agent_intelligence.py:597`
- `agents/worker_agent.py:642`
- `agents/reporter_agent.py:345, 588`
- `agents/compliance_agent.py:400`
- `agents/integrator_agent.py:473`
- `orchestration/routing.py:393, 413`
- `rag/retrieval_strategies.py:50, 524`
- `rag/document_processor.py:89, 90`
- `rag/rag_system.py:57, 336, 347, 581`
- `rag/knowledge_graph.py:42, 70`

**Find/Replace Script:**
```python
# find_replace_md5.py
import os
import re

def replace_md5_in_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern 1: hashlib.md5(...).hexdigest()
    original = content
    content = re.sub(
        r'hashlib\.md5\((.*?)\)\.hexdigest\(\)',
        r'hashlib.blake2b(\1, digest_size=16).hexdigest()',
        content
    )

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Updated: {filepath}")

# Run on all Python files
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py'):
            replace_md5_in_file(os.path.join(root, file))
```

**Testing:**
```python
# Verify hash outputs are different but consistent
import hashlib

data = b"test data"

# Old MD5
md5_hash = hashlib.md5(data).hexdigest()

# New BLAKE2
blake2_hash = hashlib.blake2b(data, digest_size=16).hexdigest()

# Verify different algorithms
assert md5_hash != blake2_hash

# Verify consistency
assert hashlib.blake2b(data, digest_size=16).hexdigest() == blake2_hash

# Verify length (both 32 hex chars = 16 bytes)
assert len(md5_hash) == 32
assert len(blake2_hash) == 32
```

**Data Migration:**
```python
# If hash values stored in database, need migration
# migration_md5_to_blake2.py

async def migrate_hashes():
    """Regenerate all hash values with new algorithm"""
    # RAG documents
    documents = await db.fetch_all("SELECT id, content FROM rag_documents")
    for doc in documents:
        old_hash = hashlib.md5(doc['content'].encode()).hexdigest()
        new_hash = hashlib.blake2b(doc['content'].encode(), digest_size=16).hexdigest()
        await db.execute(
            "UPDATE rag_documents SET doc_id = $1 WHERE id = $2",
            new_hash, doc['id']
        )

    # Cache keys - can just invalidate
    await redis.flushdb()  # Clear all cache (will regenerate)
```

**Acceptance Criteria:**
- [ ] All 20+ MD5 instances replaced
- [ ] Unit tests updated and passing
- [ ] Performance benchmark shows no regression
- [ ] Data migration script tested
- [ ] Documentation updated

---

#### HIGH-002: Add Import Guard to security/examples.py
**Owner:** Dev 3
**Effort:** 15 minutes
**Priority:** P1

**Fixed Code:**
```python
"""
Security Examples - INTENTIONALLY VULNERABLE CODE FOR TRAINING

WARNING: This file contains DELIBERATELY INSECURE code patterns to demonstrate
         security vulnerabilities. DO NOT use any code from this file in production.

All insecure functions are prefixed with 'example_*_insecure' for clarity.
All secure functions are prefixed with 'example_*_secure' for comparison.
"""

import sys
import os

# IMPORT GUARD: Prevent accidental imports
if not sys.argv[0].endswith('examples.py') and not os.getenv('ALLOW_SECURITY_EXAMPLES'):
    raise ImportError(
        "security/examples.py contains INTENTIONALLY VULNERABLE code and should "
        "NEVER be imported. This file is for security training and documentation only.\n\n"
        "If you need to run examples for training, execute the file directly:\n"
        "  python security/examples.py\n\n"
        "To bypass this check in tests only:\n"
        "  ALLOW_SECURITY_EXAMPLES=1 pytest tests/test_security_examples.py"
    )

# Rest of file...
```

**Testing:**
```python
# Test import blocked
try:
    import security.examples
    assert False, "Import should be blocked"
except ImportError as e:
    assert "INTENTIONALLY VULNERABLE" in str(e)

# Test direct execution allowed
subprocess.run(["python", "security/examples.py"], check=True)

# Test test bypass
os.environ["ALLOW_SECURITY_EXAMPLES"] = "1"
import security.examples  # Should work now
```

**Acceptance Criteria:**
- [ ] Import guard added
- [ ] Direct execution still works
- [ ] Import from other modules blocked
- [ ] Test bypass mechanism works
- [ ] Documentation updated

---

#### HIGH-003-006: Subprocess and SQL Review
**Owner:** Dev 2
**Effort:** 2 hours
**Priority:** P1

**Subprocess Review Checklist:**
```python
# Review all subprocess calls for:
# 1. shell=False (REQUIRED)
# 2. Command as list, not string
# 3. Input validation
# 4. Timeout specified
# 5. Error handling

# GOOD example:
result = subprocess.run(
    ["kubectl", "get", "pods", validated_namespace],  # List, not string
    shell=False,  # CRITICAL
    capture_output=True,
    text=True,
    timeout=30,  # Prevent hanging
    check=True
)

# BAD example:
subprocess.run(f"kubectl get pods {namespace}", shell=True)  # DANGEROUS
```

**SQL Review Checklist:**
```python
# Review all SQL queries for:
# 1. Parameterized queries ($1, $2) - NOT string interpolation
# 2. Whitelist validation for identifiers (table/column names)
# 3. Type validation for values
# 4. Length limits enforced

# GOOD example:
query = "SELECT * FROM users WHERE tenant_id = $1 AND status = $2"
params = [validated_tenant_id, validated_status]
result = await db.fetch_all(query, *params)

# BAD example:
query = f"SELECT * FROM {table} WHERE tenant_id = '{tenant_id}'"  # DANGEROUS
```

**Acceptance Criteria:**
- [ ] All subprocess calls reviewed and documented
- [ ] All SQL queries reviewed and documented
- [ ] Any unsafe patterns fixed
- [ ] Review findings documented
- [ ] Code comments added

---

#### HIGH-007: Dependency Vulnerability Scanning
**Owner:** Dev 1
**Effort:** 4 hours
**Priority:** P1

**Setup Automated Scanning:**
```bash
# Install scanners
pip install safety pip-audit

# Run scans
safety check --json --output safety_report.json
pip-audit --format json --output pip_audit_report.json

# Review findings
python scripts/review_vulnerabilities.py
```

**Review Script:**
```python
# scripts/review_vulnerabilities.py
import json

# Load scan results
with open('safety_report.json') as f:
    safety_data = json.load(f)

with open('pip_audit_report.json') as f:
    audit_data = json.load(f)

# Categorize by severity
critical = []
high = []
medium = []

for vuln in safety_data.get('vulnerabilities', []):
    severity = vuln.get('severity', 'unknown').upper()
    if severity == 'CRITICAL':
        critical.append(vuln)
    elif severity == 'HIGH':
        high.append(vuln)
    else:
        medium.append(vuln)

# Generate report
print(f"CRITICAL: {len(critical)}")
print(f"HIGH: {len(high)}")
print(f"MEDIUM: {len(medium)}")

# Output upgrade commands
for vuln in critical + high:
    pkg = vuln['package']
    fixed_version = vuln.get('fixed_in', ['latest'])[0]
    print(f"pip install {pkg}>={fixed_version}")
```

**Update Dependencies:**
```bash
# Update vulnerable packages
pip install cryptography>=42.0.5
pip install aiohttp>=3.9.3
pip install jinja2>=3.1.3

# Re-run scans to verify
safety check
pip-audit

# Update requirements.txt with new versions
pip freeze > requirements.txt
```

**Acceptance Criteria:**
- [ ] safety and pip-audit installed
- [ ] Initial vulnerability scan complete
- [ ] All CRITICAL CVEs patched
- [ ] All HIGH CVEs patched
- [ ] requirements.txt updated
- [ ] Re-scan confirms 0 critical/high vulnerabilities
- [ ] All tests pass with updated dependencies

---

#### HIGH-008: Update Test Credentials to Use Environment Variables
**Owner:** Dev 3
**Effort:** 30 minutes
**Priority:** P2

**Files to Update:**
- `tests/integration/test_llm_providers.py`
- `tests/integration/test_llm_failover.py`
- `llm_capable_agent.py`

**Fix Pattern:**
```diff
- api_key="invalid-key-12345",
+ api_key=os.getenv("TEST_API_KEY", "invalid-key-for-testing"),
```

**Acceptance Criteria:**
- [ ] All hardcoded test keys replaced
- [ ] Environment variable fallbacks added
- [ ] Tests still pass
- [ ] Documentation updated

---

### Phase 2 Summary

**Total Effort:** 8.5 hours development + 8 hours testing = 16.5 hours
**Timeline:** 5 days (includes thorough testing and code review)
**Deliverables:**
- All high-priority vulnerabilities fixed
- MD5 replaced globally
- Dependency vulnerabilities patched
- Code review and security testing complete

**Phase 2 Exit Criteria:**
- [ ] All HIGH issues resolved
- [ ] Dependency scan shows 0 critical/high CVEs
- [ ] All unit tests passing
- [ ] Performance benchmarks within acceptable range
- [ ] Code review approved
- [ ] Security scan shows significant improvement

---

## Phase 3: MEDIUM PRIORITY (Days 8-12) - BEST PRACTICES

**Deadline:** 2025-01-27 (5 days after Phase 2)
**Team:** 2 developers + 1 QA engineer
**Status:** SECURITY BEST PRACTICES

### Objectives
- Implement security monitoring
- Add rate limiting
- Enhance input validation
- Improve documentation

### Tasks

#### MEDIUM-001: Security Event Logging
**Owner:** Dev 2
**Effort:** 3 hours
**Priority:** P2

**Events to Log:**
- Failed authentication attempts
- Authorization failures
- Suspicious input patterns
- Rate limit violations
- Security exceptions
- Admin actions

**Implementation:**
```python
# observability/security_logging.py
import structlog
from datetime import datetime
from typing import Optional

logger = structlog.get_logger()

class SecurityEventLogger:
    """Centralized security event logging"""

    @staticmethod
    def log_auth_failure(
        username: str,
        reason: str,
        ip_address: str,
        user_agent: Optional[str] = None
    ):
        """Log failed authentication attempt"""
        logger.warning(
            "authentication_failed",
            event_type="security.auth.failed",
            username=username,
            reason=reason,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow().isoformat(),
            severity="warning"
        )

    @staticmethod
    def log_authz_failure(
        user_id: str,
        resource: str,
        action: str,
        tenant_id: str
    ):
        """Log authorization failure"""
        logger.warning(
            "authorization_failed",
            event_type="security.authz.failed",
            user_id=user_id,
            resource=resource,
            action=action,
            tenant_id=tenant_id,
            timestamp=datetime.utcnow().isoformat(),
            severity="warning"
        )

    # Additional methods for other security events...
```

**Acceptance Criteria:**
- [ ] Security event logger implemented
- [ ] All authentication/authorization failures logged
- [ ] SIEM integration ready (structured JSON logs)
- [ ] Tests verify logging
- [ ] Documentation updated

---

#### MEDIUM-002: Rate Limiting
**Owner:** Dev 3
**Effort:** 4 hours
**Priority:** P2

**Implementation:**
```python
# api/rate_limiting.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import FastAPI, Request

# Create limiter
limiter = Limiter(key_func=get_remote_address)

# Add to FastAPI app
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Apply to endpoints
@app.post("/api/v1/agents/execute")
@limiter.limit("100/minute")  # 100 requests per minute per IP
async def execute_agent(request: Request):
    ...

# Per-tenant rate limiting
@limiter.limit("1000/hour", key_func=lambda: get_tenant_id())
async def tenant_limited_endpoint():
    ...
```

**Rate Limits:**
- Authentication: 5/minute per IP
- API endpoints: 100/minute per IP
- Agent execution: 1000/hour per tenant
- Admin operations: 10/minute per user

**Acceptance Criteria:**
- [ ] Rate limiting implemented
- [ ] Per-IP and per-tenant limits
- [ ] Rate limit headers returned (X-RateLimit-*)
- [ ] Tests verify limits enforced
- [ ] Documentation updated

---

#### MEDIUM-003: Input Validation Enhancement
**Owner:** Dev 1
**Effort:** 4 hours
**Priority:** P2

**Validation Framework:**
```python
# validation/validators.py
from pydantic import BaseModel, validator, Field
from typing import Optional
import re

class TenantIdValidator(BaseModel):
    """Validate tenant ID format"""
    tenant_id: str = Field(..., min_length=3, max_length=255)

    @validator('tenant_id')
    def validate_tenant_id(cls, v):
        # Alphanumeric, hyphens, underscores only
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Tenant ID must be alphanumeric')
        return v

class EmailValidator(BaseModel):
    """Validate email format"""
    email: str

    @validator('email')
    def validate_email(cls, v):
        from email_validator import validate_email, EmailNotValidError
        try:
            validate_email(v)
            return v
        except EmailNotValidError as e:
            raise ValueError(str(e))

# Apply to all API endpoints
@app.post("/api/v1/tenant/create")
async def create_tenant(data: TenantIdValidator):
    ...
```

**Acceptance Criteria:**
- [ ] Validation framework implemented
- [ ] All user inputs validated
- [ ] Pydantic models for all API requests
- [ ] Tests verify validation
- [ ] Documentation updated

---

#### MEDIUM-004: CORS Configuration Review
**Owner:** Dev 2
**Effort:** 1 hour
**Priority:** P2

**Current Config:**
```python
# api/main.py
origins = [
    "https://*.greenlang.io",
    "http://localhost:3000",  # Development
    "http://localhost:8000",  # Development
]
```

**Enhanced Config:**
```python
import os

# Environment-specific CORS
environment = os.getenv("ENVIRONMENT", "production")

if environment == "production":
    origins = [
        "https://*.greenlang.io",
        "https://app.greenlang.io",
        "https://api.greenlang.io"
    ]
elif environment == "staging":
    origins = [
        "https://*.greenlang-staging.io",
        "http://localhost:3000"
    ]
else:  # development
    origins = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600
)
```

**Acceptance Criteria:**
- [ ] Environment-specific CORS
- [ ] Production config restrictive
- [ ] Development config flexible
- [ ] Tests verify CORS headers
- [ ] Documentation updated

---

#### MEDIUM-005: Security Documentation
**Owner:** All Developers
**Effort:** 2 hours
**Priority:** P2

**Documentation to Create:**
1. Security Architecture Document
2. Incident Response Plan
3. Security Best Practices Guide
4. Developer Security Training

**Acceptance Criteria:**
- [ ] All documents created
- [ ] Architecture diagrams included
- [ ] Runbooks for common incidents
- [ ] Training materials available
- [ ] Team training scheduled

---

### Phase 3 Summary

**Total Effort:** 14 hours development + 6 hours testing = 20 hours
**Timeline:** 5 days
**Deliverables:**
- Security monitoring implemented
- Rate limiting active
- Input validation comprehensive
- Documentation complete

**Phase 3 Exit Criteria:**
- [ ] All MEDIUM issues resolved
- [ ] Security monitoring operational
- [ ] Rate limiting tested and verified
- [ ] Documentation reviewed and approved
- [ ] Team trained on security practices

---

## Phase 4: ONGOING SECURITY (Continuous)

**Timeline:** Ongoing (post-deployment)
**Team:** All developers + Security Lead

### Continuous Security Activities

#### Weekly Tasks
- [ ] Run automated dependency scans (safety, pip-audit)
- [ ] Review security event logs
- [ ] Monitor failed authentication attempts
- [ ] Check rate limiting metrics

#### Monthly Tasks
- [ ] Manual security code review
- [ ] Dependency updates (patch versions)
- [ ] Security metrics review
- [ ] Incident response drill

#### Quarterly Tasks
- [ ] Penetration testing (external)
- [ ] Security architecture review
- [ ] Dependency major version updates
- [ ] Compliance audit preparation

#### Annual Tasks
- [ ] Full security audit (external)
- [ ] SOC2/ISO 27001 audit
- [ ] Security training refresh
- [ ] Threat model update

---

## Automation Setup

### CI/CD Security Pipeline

**GitHub Actions Workflow:**
```yaml
# .github/workflows/security.yml
name: Security Scan

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install bandit safety pip-audit semgrep

      - name: Run Bandit
        run: bandit -r . -f json -o bandit_report.json

      - name: Run Safety
        run: safety check --json --output safety_report.json

      - name: Run pip-audit
        run: pip-audit --format json --output pip_audit_report.json

      - name: Run Semgrep
        run: semgrep --config=auto --json --output semgrep_report.json .

      - name: Upload reports
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: '*_report.json'

      - name: Check for critical issues
        run: |
          python scripts/check_security_blockers.py
          # Fail build if critical issues found
```

**Pre-commit Hooks:**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-ll', '-i']  # Low severity, ignore

  - repo: https://github.com/returntocorp/semgrep
    rev: v1.52.0
    hooks:
      - id: semgrep
        args: ['--config=auto']
```

---

## Success Metrics

### Security Metrics

**Code Security:**
- Critical vulnerabilities: 0 (target: 0)
- High vulnerabilities: 0 (target: 0)
- Medium vulnerabilities: <5 (target: <3)
- Low vulnerabilities: <10 (target: <5)

**Dependency Security:**
- Critical CVEs: 0 (target: 0)
- High CVEs: 0 (target: 0)
- Medium CVEs: <3 (target: <2)
- Outdated packages: <10% (target: <5%)

**Authentication Security:**
- Failed auth rate: <1% (target: <0.5%)
- JWT validation errors: 0 (target: 0)
- Authentication bypasses: 0 (target: 0)

**Runtime Security:**
- Security exceptions: <10/day (target: <5/day)
- Rate limit violations: <100/day (target: <50/day)
- Suspicious activities: 0 (target: 0)

---

## Risk Management

### Risks and Mitigation

**Risk 1: Regression During Fixes**
- Probability: Medium
- Impact: High
- Mitigation: Comprehensive testing, code review, staging deployment

**Risk 2: Performance Degradation**
- Probability: Low
- Impact: Medium
- Mitigation: Performance benchmarks, load testing

**Risk 3: Incomplete Dependency Updates**
- Probability: Medium
- Impact: Medium
- Mitigation: Automated scanning, manual review

**Risk 4: Timeline Delay**
- Probability: Low
- Impact: High
- Mitigation: Buffer time, prioritization, parallel work

---

## Communication Plan

### Stakeholder Updates

**Daily Standups (Phase 1-2):**
- Progress on critical/high issues
- Blockers and dependencies
- Test results

**Weekly Status Reports:**
- Completed tasks
- Upcoming milestones
- Risk updates
- Metrics dashboard

**Phase Completion Reviews:**
- Demo of fixes
- Security scan results
- Compliance status
- Go/no-go decision for next phase

---

## Appendix: Quick Reference

### Critical Commands

**Run Security Scans:**
```bash
bandit -r . -ll
safety check
pip-audit
semgrep --config=auto .
```

**Run Tests:**
```bash
pytest tests/ -v
pytest tests/security/ -v --log-level=DEBUG
```

**Deploy to Staging:**
```bash
git checkout staging
git merge main
./scripts/deploy_staging.sh
```

**Emergency Rollback:**
```bash
./scripts/rollback.sh
```

---

**END OF REMEDIATION ROADMAP**
