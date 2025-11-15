# Dependency Security Guide - GreenLang Agent Foundation

**Version:** 1.0.0
**Last Updated:** 2025-01-15
**Owner:** DevOps & Security Team

---

## Table of Contents

1. [Overview](#overview)
2. [Dependency Management Strategy](#dependency-management-strategy)
3. [Security Update Process](#security-update-process)
4. [CVE Response Procedures](#cve-response-procedures)
5. [Automated Security Scanning](#automated-security-scanning)
6. [Manual Audit Procedures](#manual-audit-procedures)
7. [Rollback Procedures](#rollback-procedures)
8. [License Compliance](#license-compliance)
9. [SBOM Generation](#sbom-generation)
10. [Incident Response](#incident-response)

---

## Overview

GreenLang uses a **security-first dependency management strategy** with exact version pinning to prevent supply chain attacks and unintended version drift.

### Key Principles

1. **Exact Version Pinning** - All dependencies use `==` instead of `>=` or `~=`
2. **Automated Security Scanning** - Daily vulnerability checks via GitHub Actions
3. **Rapid CVE Response** - Critical/High CVEs patched within 24 hours
4. **License Compliance** - No GPL/LGPL dependencies in production
5. **SBOM Tracking** - Maintain Software Bill of Materials for audits

---

## Dependency Management Strategy

### Version Pinning Policy

**Production (`requirements.txt`)**
```txt
# ✅ CORRECT - Exact pinning
cryptography==42.0.5
requests==2.31.0
fastapi==0.109.2

# ❌ INCORRECT - Range-based versioning
cryptography>=42.0.0
requests~=2.31
fastapi>=0.109
```

**Why Exact Pinning?**

| Risk | Range Versioning | Exact Pinning |
|------|------------------|---------------|
| Supply Chain Attack | ⚠️ HIGH | ✅ LOW |
| Breaking Changes | ⚠️ HIGH | ✅ LOW |
| Reproducible Builds | ❌ NO | ✅ YES |
| CI/CD Consistency | ⚠️ VARIABLE | ✅ GUARANTEED |

### Dependency Update Cadence

| Update Type | Frequency | Owner |
|-------------|-----------|-------|
| Security Patches (Critical/High) | Immediate | Security Team |
| Security Patches (Medium) | Weekly | DevOps Team |
| Security Patches (Low) | Monthly | DevOps Team |
| Feature Updates | Quarterly | Engineering Team |
| Major Version Upgrades | Annually | Architecture Team |

---

## Security Update Process

### 1. Automated Dependabot Updates

**Configuration:** `.github/dependabot.yml`

```yaml
schedule:
  interval: "weekly"
  day: "monday"
  time: "09:00"
allow:
  - dependency-type: "direct"
    update-type: "security"
  - dependency-type: "indirect"
    update-type: "security"
```

**Process:**
1. Dependabot creates PR with security patch
2. CI/CD runs automated tests
3. Security team reviews CVE details
4. Auto-merge if tests pass (optional)
5. Deploy to staging → production

### 2. Manual Security Updates

**When to use:**
- Dependabot doesn't support the package
- Pre-release security fixes needed
- Emergency CVE response

**Steps:**
```bash
# 1. Check current version
pip show cryptography

# 2. Check for vulnerabilities
pip-audit --package cryptography

# 3. Update to patched version
# Edit requirements.txt
cryptography==42.0.5

# 4. Test locally
pip install -r requirements.txt
pytest

# 5. Commit and deploy
git add requirements.txt
git commit -m "security: Update cryptography to 42.0.5 (CVE-2024-0727)"
git push
```

### 3. Transitive Dependency Updates

**Problem:** Vulnerability in sub-dependency (e.g., `urllib3` via `requests`)

**Solution:**
```bash
# 1. Identify parent package
pip show urllib3 | grep Required-by
# Required-by: requests

# 2. Update parent to version that uses patched sub-dependency
# Check parent package's dependency constraints
pip show requests | grep Requires
# Requires: urllib3>=2.0.7

# 3. Update parent package
# Edit requirements.txt
requests==2.31.0  # This enforces urllib3>=2.0.7

# 4. Verify sub-dependency updated
pip install -r requirements.txt
pip show urllib3 | grep Version
# Version: 2.0.7 ✅
```

---

## CVE Response Procedures

### Severity-Based Response SLA

| Severity | Response Time | Patch Time | Approval Required |
|----------|---------------|------------|-------------------|
| **CRITICAL** (CVSS 9.0-10.0) | 1 hour | 4 hours | Security Lead |
| **HIGH** (CVSS 7.0-8.9) | 4 hours | 24 hours | DevOps Lead |
| **MEDIUM** (CVSS 4.0-6.9) | 24 hours | 1 week | Team Lead |
| **LOW** (CVSS 0.1-3.9) | 1 week | 30 days | Engineering Team |

### CVE Response Workflow

```
┌─────────────────────┐
│ CVE Alert Received  │
│ (Dependabot/Email)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Assess Severity     │
│ - CVSS Score        │
│ - Exploitability    │
│ - Impact            │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Check Patch         │
│ - Patched version?  │
│ - Breaking changes? │
│ - Compatibility?    │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌────────┐   ┌────────────┐
│ PATCH  │   │ WORKAROUND │
│ EXISTS │   │ or DEFER   │
└───┬────┘   └─────┬──────┘
    │              │
    ▼              ▼
┌────────────┐  ┌──────────────┐
│ Update Dep │  │ Implement    │
│ Run Tests  │  │ Mitigation   │
│ Deploy     │  │ (WAF, etc.)  │
└────────────┘  └──────────────┘
```

### Example: CVE-2024-0727 Response

**CVE:** CVE-2024-0727
**Package:** cryptography <42.0.5
**Severity:** CRITICAL (CVSS 9.1)
**Description:** OpenSSL DoS via malicious PKCS#12 files

**Response:**

1. **Alert Received:** 2025-01-10 09:00 UTC
2. **Severity Assessment:** CRITICAL - Production impact HIGH
3. **Patch Available:** cryptography==42.0.5
4. **Testing:** Regression tests PASSED
5. **Deployment:**
   - Staging: 2025-01-10 11:00 UTC ✅
   - Production: 2025-01-10 13:00 UTC ✅
6. **Total Response Time:** 4 hours (within SLA) ✅

**Documentation:**
- Updated SECURITY_AUDIT.md
- Created incident report
- Notified stakeholders

---

## Automated Security Scanning

### Daily GitHub Actions Workflow

**File:** `.github/workflows/security-audit.yml`

**Scans Performed:**
1. **Safety Check** - PyPA advisory database
2. **pip-audit** - Official PyPI vulnerability scanner (OSV)
3. **Bandit** - Python code security linter
4. **Semgrep** - Static analysis security patterns
5. **License Check** - GPL/LGPL detection

**Schedule:**
```yaml
on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight UTC
  pull_request:
    paths:
      - '**/requirements*.txt'
  push:
    branches: [main, master]
```

**Notifications:**
- GitHub Actions summary
- Slack #security-alerts (CRITICAL/HIGH)
- Email to security team

---

## Manual Audit Procedures

### Monthly Dependency Audit

**Checklist:**

1. **Run Security Scanners**
   ```bash
   # Safety check
   safety check --json > audit/safety-$(date +%Y%m%d).json

   # pip-audit
   pip-audit --format json > audit/pip-audit-$(date +%Y%m%d).json

   # Bandit code scan
   bandit -r . -f json > audit/bandit-$(date +%Y%m%d).json
   ```

2. **Review Vulnerability Reports**
   - Identify new CVEs
   - Assess severity and impact
   - Check for available patches

3. **Check for Abandoned Packages**
   ```bash
   # Check last release date
   pip show <package> | grep Version
   # Research on PyPI for maintenance status
   ```

4. **License Compliance Audit**
   ```bash
   pip-licenses --format=csv --output-file=licenses-audit.csv
   # Review for GPL/LGPL/AGPL
   ```

5. **Update SECURITY_AUDIT.md**
   - Document findings
   - Track remediation status
   - Update next audit date

### Quarterly Comprehensive Review

**Additional Steps:**

1. **Dependency Tree Analysis**
   ```bash
   pipdeptree --json > dependency-tree.json
   # Identify unnecessary transitive dependencies
   ```

2. **Package Size Audit**
   ```bash
   # Check installed package sizes
   pip list --format=json | jq -r '.[] | "\(.name): \(.version)"' | while read line; do
     size=$(pip show ${line%:*} | grep Location | awk '{print $2}' | xargs du -sh)
     echo "$line - $size"
   done
   ```

3. **Alternative Package Evaluation**
   - Research newer/better alternatives
   - Evaluate performance improvements
   - Assess security track record

---

## Rollback Procedures

### When to Rollback

- New dependency version breaks production
- Unexpected side effects discovered
- Performance degradation
- Compatibility issues

### Rollback Steps

1. **Identify Previous Working Version**
   ```bash
   git log --oneline requirements.txt
   # Find commit before problematic update
   ```

2. **Revert to Previous Version**
   ```bash
   # Option 1: Revert specific file
   git checkout <commit-hash> requirements.txt

   # Option 2: Revert entire commit
   git revert <commit-hash>
   ```

3. **Test Rollback**
   ```bash
   pip install -r requirements.txt
   pytest
   ```

4. **Deploy Rollback**
   ```bash
   git commit -m "revert: Rollback cryptography to 42.0.4 due to production issues"
   git push
   ```

5. **Document Incident**
   - Update SECURITY_AUDIT.md
   - Create incident report
   - Notify stakeholders

### Emergency Rollback

**For production outages:**

```bash
# 1. Immediate container rollback
kubectl rollout undo deployment/greenlang-app -n production

# 2. Verify previous version running
kubectl get pods -n production

# 3. Document and investigate
# Create incident ticket
# Schedule post-mortem
```

---

## License Compliance

### Approved Licenses (Production)

| License | Risk Level | Usage |
|---------|------------|-------|
| MIT | ✅ LOW | Unrestricted |
| Apache 2.0 | ✅ LOW | Unrestricted |
| BSD 3-Clause | ✅ LOW | Unrestricted |
| BSD 2-Clause | ✅ LOW | Unrestricted |
| PSF (Python) | ✅ LOW | Unrestricted |

### Prohibited Licenses (Production)

| License | Risk Level | Reason |
|---------|------------|--------|
| GPL v2/v3 | 🚫 CRITICAL | Copyleft - Requires source disclosure |
| LGPL v2/v3 | ⚠️ HIGH | Weak copyleft - Linking restrictions |
| AGPL v3 | 🚫 CRITICAL | Network copyleft - SaaS restrictions |
| Commons Clause | ⚠️ HIGH | Commercial use restrictions |

### License Audit Process

```bash
# 1. Generate license report
pip-licenses --format=json --output-file=licenses.json

# 2. Check for prohibited licenses
pip-licenses --fail-on 'GNU General Public License'

# 3. Review dual-licensed packages
pip-licenses --format=markdown > LICENSES.md
```

---

## SBOM Generation

### CycloneDX SBOM

**Software Bill of Materials** for compliance and security tracking.

```bash
# Install SBOM generator
pip install cyclonedx-bom

# Generate SBOM
cyclonedx-py \
  --format json \
  --output sbom.json \
  --requirements requirements.txt

# Validate SBOM
cyclonedx-cli validate --input-file sbom.json
```

### SBOM Use Cases

1. **Regulatory Compliance** - SOC 2, ISO 27001, NIST
2. **Vulnerability Tracking** - Map CVEs to components
3. **License Audits** - Track all licenses in one document
4. **Supply Chain Security** - Verify integrity of dependencies

### SBOM Format

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.5",
  "version": 1,
  "components": [
    {
      "type": "library",
      "name": "cryptography",
      "version": "42.0.5",
      "purl": "pkg:pypi/cryptography@42.0.5",
      "licenses": [{"license": {"id": "Apache-2.0"}}],
      "hashes": [{"alg": "SHA-256", "content": "..."}]
    }
  ]
}
```

---

## Incident Response

### Security Incident Types

1. **Supply Chain Compromise** - Malicious package or dependency
2. **Zero-Day Vulnerability** - Unpatched CVE in production
3. **License Violation** - GPL package discovered in production
4. **Abandoned Package** - Unmaintained dependency with vulnerabilities

### Incident Response Plan

```
┌─────────────────┐
│ DETECTION       │
│ - Scan alerts   │
│ - CVE announce  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CONTAINMENT     │
│ - Assess impact │
│ - Block traffic │
│ - Isolate       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ERADICATION     │
│ - Remove threat │
│ - Patch systems │
│ - Update deps   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ RECOVERY        │
│ - Restore       │
│ - Verify        │
│ - Monitor       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ POST-MORTEM     │
│ - Root cause    │
│ - Lessons       │
│ - Improvements  │
└─────────────────┘
```

### Contact Information

**Security Team:**
- Email: security@greenlang.io
- Slack: #security-incidents
- On-call: +1-XXX-XXX-XXXX

**Escalation Path:**
1. Security Engineer (1 hour response)
2. Security Lead (4 hour response)
3. CISO (immediate escalation for CRITICAL)

---

## Tools Reference

### Security Scanning Tools

| Tool | Purpose | Command |
|------|---------|---------|
| Safety | PyPA vulnerability DB | `safety check` |
| pip-audit | Official PyPI scanner | `pip-audit` |
| Bandit | Python code security | `bandit -r .` |
| Semgrep | Static analysis | `semgrep --config=auto` |
| pip-licenses | License checker | `pip-licenses` |
| CycloneDX | SBOM generator | `cyclonedx-py` |

### Installation

```bash
pip install safety pip-audit bandit semgrep pip-licenses cyclonedx-bom
```

---

## Appendix: CVE Database Sources

### Primary Sources

1. **NVD** - National Vulnerability Database (NIST)
   - https://nvd.nist.gov/
   - CVSS scoring authority

2. **OSV** - Open Source Vulnerabilities
   - https://osv.dev/
   - Used by pip-audit

3. **PyPA Advisory Database**
   - https://github.com/pypa/advisory-database
   - Used by Safety

4. **GitHub Security Advisories**
   - https://github.com/advisories
   - Integrated with Dependabot

### Monitoring Services

1. **Snyk** - https://snyk.io/
2. **WhiteSource** - https://www.whitesourcesoftware.com/
3. **Socket.dev** - https://socket.dev/
4. **Tidelift** - https://tidelift.com/

---

**Document Version:** 1.0.0
**Last Review:** 2025-01-15
**Next Review:** 2025-02-15
**Owner:** DevOps & Security Team
