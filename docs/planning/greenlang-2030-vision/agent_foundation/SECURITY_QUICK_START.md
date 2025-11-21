# Security Quick Start Guide

**For:** Developers, DevOps Engineers, Security Team
**Updated:** 2025-01-15
**Version:** 1.0.0

---

## 5-Minute Security Setup

### 1. Install Pinned Dependencies

```bash
# Navigate to agent_foundation directory
cd GreenLang_2030/agent_foundation

# Install production dependencies (exact versions)
pip install -r requirements.txt

# Install development dependencies (includes security tools)
pip install -r requirements-dev.txt

# Verify no vulnerabilities
safety check
pip-audit
```

### 2. Enable Pre-commit Hooks

```bash
# Install pre-commit framework
pip install pre-commit

# Install hooks in your repository
pre-commit install

# Run on all files (first-time setup)
pre-commit run --all-files
```

### 3. Configure GitHub Actions (if repo owner)

```bash
# Files already created:
# - .github/workflows/security-audit.yml (daily scans)
# - .github/dependabot.yml (weekly updates)

# Enable in GitHub Settings:
# Settings > Security > Dependabot alerts: ON
# Settings > Security > Dependabot security updates: ON
# Settings > Code security > Dependency graph: ON
```

---

## Daily Security Workflow

### Before Coding

```bash
# 1. Pull latest dependencies
git pull origin main

# 2. Check for security updates
safety check
pip-audit

# 3. Update if needed
pip install -r requirements.txt --upgrade
```

### While Coding

Pre-commit hooks automatically run on `git commit`:
- Bandit (security scanner)
- Safety (vulnerability check)
- Black (code formatter)
- MyPy (type checker)
- Secret detection

### Before Committing

```bash
# 1. Run security scans manually
safety check
pip-audit
bandit -r . -f json

# 2. Commit (hooks run automatically)
git commit -m "feat: Add new feature"

# 3. If hooks fail, fix and retry
git commit -m "security: Fix vulnerability"
```

---

## Responding to CVEs

### Critical/High Severity (CVSS 7.0+)

**Response Time: 4 hours**

```bash
# 1. Check vulnerability details
safety check --json
pip-audit --format json

# 2. Update affected package
# Edit requirements.txt
cryptography==42.0.5  # Update to patched version

# 3. Test locally
pip install -r requirements.txt
pytest

# 4. Commit and deploy
git commit -m "security: Update cryptography to 42.0.5 (CVE-2024-0727)"
git push

# 5. Monitor deployment
kubectl rollout status deployment/greenlang-app -n production
```

### Medium/Low Severity

**Response Time: 1 week**

- Wait for Dependabot PR
- Review automated tests
- Approve and merge

---

## Emergency Rollback

### If new dependency breaks production:

```bash
# 1. Immediate rollback (Kubernetes)
kubectl rollout undo deployment/greenlang-app -n production

# 2. Revert dependency change
git log --oneline requirements.txt
git revert <commit-hash>

# 3. Deploy rollback
git push

# 4. Document incident
# Create ticket in issue tracker
# Update SECURITY_AUDIT.md
```

---

## Common Security Commands

### Dependency Scanning

```bash
# Check for vulnerabilities
safety check

# Detailed vulnerability report
pip-audit --format json > audit-report.json

# Code security scan
bandit -r . -f json -o bandit-report.json

# License compliance
pip-licenses --format=csv
```

### Package Updates

```bash
# Show outdated packages
pip list --outdated

# Update specific package
pip install package-name==new-version

# Freeze all dependencies
pip freeze > requirements-frozen.txt

# Generate SBOM
cyclonedx-py --format json --output sbom.json
```

### Hash Verification (Maximum Security)

```bash
# Generate hashes
pip-compile --generate-hashes requirements.txt

# Install with hash verification
pip install --require-hashes -r requirements.txt
```

---

## Security Alerts

### Where to find alerts:

1. **GitHub Dependabot**
   - Repository > Security > Dependabot alerts
   - Automated PRs for security updates

2. **GitHub Actions**
   - Repository > Actions > Security Audit
   - Daily scan results and reports

3. **Email Notifications**
   - Configure in GitHub Settings > Notifications
   - Email on critical/high vulnerabilities

4. **Slack (if configured)**
   - #security-alerts channel
   - Real-time notifications for critical issues

---

## Documentation Reference

| Document | Purpose |
|----------|---------|
| `SECURITY_AUDIT.md` | Comprehensive CVE audit report |
| `DEPENDENCY_SECURITY_GUIDE.md` | Detailed security procedures |
| `SECURITY_HARDENING_SUMMARY.md` | Executive summary |
| `SECURITY_QUICK_START.md` | This file - quick reference |

---

## Security Team Contacts

**Email:** security@greenlang.io
**Slack:** #security-team
**On-call:** See PagerDuty rotation

**Response SLA:**
- Critical: 1 hour response, 4 hour patch
- High: 4 hour response, 24 hour patch
- Medium: 24 hour response, 1 week patch
- Low: 1 week response, 30 day patch

---

## Troubleshooting

### Pre-commit hooks failing?

```bash
# Update hooks
pre-commit autoupdate

# Run specific hook
pre-commit run bandit --all-files

# Skip hooks temporarily (use sparingly!)
git commit --no-verify
```

### Dependency conflicts?

```bash
# Show dependency tree
pipdeptree

# Check specific package dependencies
pip show package-name

# Create clean virtual environment
python -m venv venv-clean
source venv-clean/bin/activate  # Linux/Mac
# or
venv-clean\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Security scan false positives?

```bash
# Safety: Ignore specific vulnerability
safety check --ignore 12345

# Bandit: Skip specific test
bandit -r . --skip B101

# Add to pre-commit config if persistent
```

---

## Best Practices

1. **Always use exact version pinning** (`==` not `>=`)
2. **Run security scans before committing**
3. **Update dependencies weekly** (via Dependabot)
4. **Test updates in staging first**
5. **Document all security changes**
6. **Never skip pre-commit hooks** (without good reason)
7. **Review Dependabot PRs promptly**
8. **Monitor security alerts daily**

---

## Resources

### Security Tools

- **Safety:** https://pyup.io/safety/
- **pip-audit:** https://github.com/pypa/pip-audit
- **Bandit:** https://bandit.readthedocs.io/
- **Semgrep:** https://semgrep.dev/
- **Dependabot:** https://github.com/dependabot

### Vulnerability Databases

- **NVD:** https://nvd.nist.gov/
- **OSV:** https://osv.dev/
- **PyPA Advisory:** https://github.com/pypa/advisory-database
- **GitHub Advisories:** https://github.com/advisories

### Training

- **OWASP Top 10:** https://owasp.org/Top10/
- **Python Security:** https://cheatsheetseries.owasp.org/cheatsheets/Python_Security_Cheat_Sheet.html
- **Supply Chain Security:** https://slsa.dev/

---

**Remember:** Security is everyone's responsibility!

When in doubt, ask the security team. Better safe than sorry.

---

**Document Version:** 1.0.0
**Last Updated:** 2025-01-15
**Maintained by:** DevOps & Security Team
