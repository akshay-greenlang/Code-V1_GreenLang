# GreenLang Security Quick Reference Guide

**For:** All GreenLang Developers
**Last Updated:** 2025-11-09
**Version:** 1.0

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Total Vulnerabilities Found | 58 |
| Critical Issues | 8 |
| Security Score | 70/100 |
| Production Ready | NO |
| Est. Fix Time | 2-4 weeks |

---

## DO NOT Deploy Checklist

Before deploying to production, ensure:

- ⬜ 0 Critical vulnerabilities
- ⬜ SQL injection fixed (`execute_raw` deprecated)
- ⬜ Budget bypass patched
- ⬜ CSV/XBRL injection vulnerabilities fixed
- ⬜ API keys in secret management (not .env files)
- ⬜ Redis TLS enabled
- ⬜ Session management in Redis (not memory)
- ⬜ All security scans passing in CI

---

## Critical Vulnerabilities - MUST FIX NOW

### 1. SQL Injection (greenlang.db)
**File:** `greenlang/db/connection.py:504`

**DON'T:**
```python
query = f"SELECT * FROM users WHERE username = '{user_input}'"
await pool.execute_raw(query)  # VULNERABLE!
```

**DO:**
```python
from sqlalchemy import select
query = select(User).where(User.username == user_input)
result = await session.execute(query)
```

---

### 2. Budget Bypass (greenlang.intelligence)
**File:** `greenlang/intelligence/runtime/budget.py`

**DON'T:**
```python
budget.add(add_usd=-0.50, add_tokens=1000)  # BYPASS!
```

**DO:**
```python
def add(self, add_usd: float, add_tokens: int):
    if add_usd < 0 or add_tokens < 0:
        raise ValueError("Costs cannot be negative")
    # ... rest of implementation
```

---

### 3. CSV Injection (GL-CBAM-APP)
**File:** CSV upload handlers

**DON'T:**
```python
df = pd.read_csv(file)  # DANGEROUS!
df.to_excel("report.xlsx")
```

**DO:**
```python
def sanitize_cell(value: str) -> str:
    if value and value[0] in ['=', '+', '-', '@']:
        return "'" + value  # Neutralize formula
    return value

df = pd.read_csv(file)
df = df.applymap(sanitize_cell)
```

---

### 4. API Keys in Code
**Files:** All provider implementations

**DON'T:**
```python
api_key = os.getenv("OPENAI_API_KEY")  # INSECURE!
```

**DO:**
```python
from greenlang.secrets import SecretManager
secret_mgr = SecretManager()
api_key = secret_mgr.get_secret("OPENAI_API_KEY", encrypted=True)
```

---

## Security Checklist for PRs

Before submitting a PR, verify:

### Code Security
- ⬜ No hardcoded secrets (run `python security/scripts/scan_secrets.py`)
- ⬜ All user input validated
- ⬜ SQL queries use parameterization
- ⬜ File uploads sanitized
- ⬜ Output encoded (HTML/XML/JSON)
- ⬜ Error messages don't leak sensitive data

### Dependencies
- ⬜ No new vulnerable dependencies (run `python security/scripts/scan_dependencies.py`)
- ⬜ License compliance checked
- ⬜ Dependencies pinned with exact versions

### Authentication & Authorization
- ⬜ Endpoints have auth decorators
- ⬜ RBAC permissions checked
- ⬜ Audit logging enabled
- ⬜ No privilege escalation possible

### Data Security
- ⬜ Sensitive data encrypted
- ⬜ PII marked and protected
- ⬜ Data retention policies followed
- ⬜ No data leakage in logs

---

## Common Vulnerabilities & Fixes

### XSS (Cross-Site Scripting)

**Vulnerable:**
```python
# Jinja2 template
<div>{{ user_input }}</div>
```

**Fixed:**
```python
# Jinja2 with auto-escaping
<div>{{ user_input | e }}</div>

# Or in Python:
import html
safe_output = html.escape(user_input, quote=True)
```

---

### Path Traversal

**Vulnerable:**
```python
file_path = f"/uploads/{filename}"  # DANGEROUS!
with open(file_path) as f:
    return f.read()
```

**Fixed:**
```python
from pathlib import Path

upload_dir = Path("/uploads").resolve()
file_path = (upload_dir / filename).resolve()

if not file_path.is_relative_to(upload_dir):
    raise ValueError("Invalid file path")

with open(file_path) as f:
    return f.read()
```

---

### Command Injection

**Vulnerable:**
```python
subprocess.run(f"convert {filename} output.pdf", shell=True)  # DANGEROUS!
```

**Fixed:**
```python
subprocess.run(
    ["convert", filename, "output.pdf"],
    shell=False,  # Never use shell=True with user input!
    check=True
)
```

---

### Insecure Deserialization

**Vulnerable:**
```python
import pickle
data = pickle.loads(user_input)  # DANGEROUS!
```

**Fixed:**
```python
import json
data = json.loads(user_input)  # Use JSON, not pickle

# Or with validation:
from pydantic import BaseModel
class DataModel(BaseModel):
    field1: str
    field2: int

data = DataModel.parse_raw(user_input)
```

---

## Security Tools

### Run Before Committing

```bash
# 1. Secret scan
python security/scripts/scan_secrets.py

# 2. Dependency check
python security/scripts/scan_dependencies.py

# 3. Format code
black greenlang/
isort greenlang/

# 4. Lint
flake8 greenlang/
mypy greenlang/

# 5. Run tests
pytest tests/
```

---

### Install Pre-Commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Or manually:
python security/scripts/scan_secrets.py --install-hook
```

---

## Quick Fixes

### Fix: Weak Password Hashing

**Before:**
```python
password_hash = hashlib.sha256(password.encode()).hexdigest()  # WEAK!
```

**After:**
```python
import bcrypt
password_hash = bcrypt.hashpw(
    password.encode(),
    bcrypt.gensalt(rounds=14)  # Strong work factor
).decode()
```

---

### Fix: Insecure Session Management

**Before:**
```python
# In-memory sessions (lost on restart)
sessions = {}
sessions[token] = user_data
```

**After:**
```python
# Redis-backed sessions
await redis.setex(
    f"session:{token}",
    3600,  # 1 hour TTL
    json.dumps(user_data)
)
```

---

### Fix: Missing Input Validation

**Before:**
```python
@app.post("/api/users")
async def create_user(username: str, email: str):
    # No validation!
    user = User(username=username, email=email)
    await db.save(user)
```

**After:**
```python
from pydantic import BaseModel, EmailStr, Field

class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, regex=r'^[a-zA-Z0-9_]+$')
    email: EmailStr

@app.post("/api/users")
async def create_user(request: CreateUserRequest):
    user = User(username=request.username, email=request.email)
    await db.save(user)
```

---

## Environment Variables

### DON'T Store in .env (Committed to Git)

```bash
# .env (NEVER commit this!)
OPENAI_API_KEY=sk-abc123...
DATABASE_URL=postgresql://user:password@localhost/db
```

### DO Use Secret Management

```bash
# .env.example (Safe to commit)
OPENAI_API_KEY=<set-in-secret-manager>
DATABASE_URL=<set-in-secret-manager>

# Fetch from Vault/AWS Secrets Manager
export OPENAI_API_KEY=$(vault kv get -field=key secret/openai)
```

---

## OWASP Top 10 Quick Reference

| # | Vulnerability | GreenLang Status | Priority |
|---|---------------|------------------|----------|
| A01 | Broken Access Control | ⚠️ Needs Work | HIGH |
| A02 | Cryptographic Failures | ⚠️ Needs Work | HIGH |
| A03 | Injection | ❌ SQL Injection | CRITICAL |
| A04 | Insecure Design | ✅ Good | - |
| A05 | Security Misconfiguration | ⚠️ Redis TLS | HIGH |
| A06 | Vulnerable Components | ⚠️ Outdated Deps | MEDIUM |
| A07 | Authentication Failures | ⚠️ Session Mgmt | HIGH |
| A08 | Software/Data Integrity | ⚠️ No Signing | HIGH |
| A09 | Logging Failures | ⚠️ Not Immutable | MEDIUM |
| A10 | SSRF | ✅ Protected | - |

---

## Incident Response

### If You Find a Security Issue:

1. **DO NOT** commit the finding to git
2. **DO NOT** discuss in public channels
3. **DO** email security@greenlang.io immediately
4. **DO** include:
   - Description of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if known)

---

### If Secrets Are Committed:

```bash
# 1. Rotate the secret IMMEDIATELY
# 2. Remove from git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/secret/file" \
  --prune-empty --tag-name-filter cat -- --all

# 3. Force push (coordinate with team first!)
git push origin --force --all

# 4. Report incident
```

---

## Security Resources

### Documentation
- [Infrastructure Security Audit](./audits/INFRASTRUCTURE_SECURITY_AUDIT.md)
- [Application Security Audit](./audits/APPLICATION_SECURITY_AUDIT.md)
- [OWASP Top 10](https://owasp.org/Top10/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

### Tools
- Dependency Scanner: `python security/scripts/scan_dependencies.py`
- Secret Scanner: `python security/scripts/scan_secrets.py`
- SAST: `bandit -r greenlang/`
- GitHub Actions: `.github/workflows/security-scan.yml`

### Training
- OWASP Top 10: https://owasp.org/www-project-top-ten/
- Secure Coding: https://www.securecoding.cert.org/
- Python Security: https://python.readthedocs.io/en/latest/library/security_warnings.html

---

## Emergency Contacts

| Role | Contact |
|------|---------|
| Security Team | security@greenlang.io |
| On-Call Engineer | +1-XXX-XXX-XXXX |
| CISO | ciso@greenlang.io |
| Incident Response | incident@greenlang.io |

---

## Compliance Quick Check

### GDPR
- ⬜ PII encrypted at rest
- ⬜ Right to erasure implemented
- ⬜ Data retention policies active
- ⬜ Consent management configured

### SOC 2
- ⬜ Audit logs immutable
- ⬜ Sessions persistent (Redis)
- ⬜ No secrets in logs
- ⬜ Change management process

### Industry-Specific
- ⬜ CBAM: Provenance signed
- ⬜ CSRD: ESRS data signed
- ⬜ VCCI: Supplier data encrypted

---

## Remember

**Security is everyone's responsibility!**

When in doubt:
1. Ask the security team
2. Fail closed (deny by default)
3. Validate all inputs
4. Encrypt sensitive data
5. Log security events

---

**Questions?** Slack: #security | Email: security@greenlang.io

---

Last Updated: 2025-11-09
Version: 1.0
