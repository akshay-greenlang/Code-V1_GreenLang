# Dependency Security Policy

## Overview
This policy defines how we handle security vulnerabilities in dependencies.

## Severity Thresholds

### Merge Blocking (Automatic)
| Severity | Known Fix Available | Action | Timeline |
|----------|-------------------|---------|----------|
| CRITICAL | Yes | **BLOCK MERGE** | Fix immediately |
| CRITICAL | No | **BLOCK MERGE** | Mitigate within 24h |
| HIGH | Yes | **BLOCK MERGE** | Fix within 24h |
| HIGH | No | Warn + Track | Mitigate within 72h |
| MEDIUM | Yes | Warn | Fix within 1 week |
| MEDIUM | No | Track | Monitor for fix |
| LOW | Any | Track | Fix in next release |

### Exceptions Process
To merge with known vulnerabilities, you must:
1. Add label: `security-exception`
2. Document justification in PR description
3. Get approval from Security team member
4. Add suppression with expiration date

## Vulnerability Response

### CRITICAL Severity (CVSS 9.0-10.0)
**Examples**: Remote code execution, authentication bypass, data exposure

**Response**:
```bash
# 1. Immediate notification
@security-team @on-call

# 2. Check if exploited
grep -r "vulnerable_function" logs/
aws cloudtrail lookup-events --lookup-attributes AttributeKey=EventName,AttributeValue=SuspiciousAPICall

# 3. Patch or mitigate
pip install package_name --upgrade
# OR if no patch:
pip uninstall vulnerable_package

# 4. Deploy hotfix
git checkout -b hotfix/CVE-2025-xxxxx
# make changes
git push
gh pr create --title "CRITICAL: Fix CVE-2025-xxxxx" --label "security-critical"
```

### HIGH Severity (CVSS 7.0-8.9)
**Examples**: Privilege escalation, significant data exposure

**Response**:
```bash
# 1. Create tracking issue
gh issue create --title "HIGH: Fix CVE-2025-xxxxx in package_name" \
  --label "security-high" \
  --assignee "@security-team"

# 2. Update dependency
pip install package_name --upgrade

# 3. Test thoroughly
pytest tests/security/
pytest tests/integration/

# 4. Normal PR process with priority review
```

### MEDIUM Severity (CVSS 4.0-6.9)
**Examples**: Cross-site scripting, information disclosure

**Response**:
- Add to security backlog
- Fix in next sprint
- Bundle with other updates

### LOW Severity (CVSS 0.1-3.9)
**Examples**: Minor information leaks, requires unusual configuration

**Response**:
- Track in quarterly review
- Fix when touching related code
- Update in major releases

## Dependency Management Best Practices

### 1. Version Pinning Strategy
```txt
# requirements.txt - PRODUCTION (pin exact versions)
django==4.2.5
requests==2.31.0
numpy==1.24.3

# requirements-dev.txt - DEVELOPMENT (allow ranges)
pytest>=7.0.0,<8.0.0
black>=22.0.0
mypy>=1.0.0
```

### 2. Update Schedule
- **Daily**: Security patches (automated via Dependabot)
- **Weekly**: Patch versions (x.x.PATCH)
- **Monthly**: Minor versions (x.MINOR.x)
- **Quarterly**: Major versions (MAJOR.x.x)

### 3. Pre-update Checklist
```bash
# Before updating dependencies
□ Run current tests: pytest
□ Check breaking changes: pip show package_name
□ Review changelog: https://github.com/org/package/releases
□ Test in isolated env: python -m venv test-env
□ Check license changes: pip-licenses
□ Scan for new vulns: pip-audit
```

## Supply Chain Security

### Approved Package Registries
✅ **Allowed**:
- PyPI (https://pypi.org)
- GitHub Packages (npm.pkg.github.com)
- Docker Hub (Official images only)

❌ **Blocked**:
- Private/unknown registries
- Direct git URLs (without hash)
- Local file:// references

### Package Verification
```bash
# Verify package signatures (when available)
pip install --require-hashes -r requirements.txt

# Generate hash file
pip-compile --generate-hashes requirements.in

# Verify package integrity
pip hash dist/package.whl
```

### New Dependency Approval

Before adding a new dependency, verify:

```bash
# 1. Check package health
pip install pipgrip
pipgrip package_name --tree  # Check sub-dependencies

# 2. Security scan
pip-audit package_name

# 3. License compatibility
pip-licenses --from=mixed | grep package_name

# 4. Maintenance status
curl -s https://pypi.org/pypi/package_name/json | jq '.info.home_page'
# Check: Last release date, GitHub stars, open issues

# 5. Size and complexity
pip install package_name --target /tmp/test
du -sh /tmp/test  # Should be reasonable size
```

### Dependency Review Checklist
- [ ] **Popularity**: >1000 weekly downloads
- [ ] **Maintenance**: Updated within last 6 months
- [ ] **Security**: No known vulnerabilities
- [ ] **License**: Compatible (MIT, Apache, BSD)
- [ ] **Dependencies**: <10 transitive dependencies
- [ ] **Size**: <10MB installed
- [ ] **Purpose**: Can't be easily implemented in-house

## Automated Enforcement

### GitHub Branch Protection Rules
```yaml
# Required status checks:
- pip-audit / audit
- trivy-scan / trivy
- secret-scan / trufflehog-pr-diff

# Dismiss stale reviews when new commits are pushed
dismiss_stale_reviews: true

# Require review from CODEOWNERS
require_code_owner_reviews: true
```

### Dependabot Configuration
```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "daily"
      time: "06:00"
    open-pull-requests-limit: 10
    labels:
      - "dependencies"
      - "security"
    reviewers:
      - "security-team"
    groups:
      development-dependencies:
        patterns:
          - "pytest*"
          - "black"
          - "mypy"
```

## Vulnerability Database References

### Primary Sources
1. **GitHub Advisory Database**: https://github.com/advisories
2. **PyPA Advisory DB**: https://github.com/pypa/advisory-database
3. **CVE Database**: https://cve.mitre.org
4. **NVD**: https://nvd.nist.gov

### Monitoring Services
- **Snyk**: https://snyk.io/vuln
- **OWASP Dependency Check**: https://jeremylong.github.io/DependencyCheck/
- **Safety**: https://pyup.io/safety/

## Metrics and Reporting

### Monthly Security Metrics
```python
# Track these KPIs
metrics = {
    "critical_vulns": 0,  # Target: 0
    "high_vulns": 2,      # Target: <5
    "medium_vulns": 8,    # Target: <20
    "mean_time_to_patch_days": 3.5,  # Target: <7
    "dependencies_total": 45,
    "dependencies_outdated": 5,  # Target: <10%
    "last_audit_date": "2025-01-15"
}
```

### Compliance Report Template
```markdown
## Dependency Security Report - [Month Year]

### Summary
- Total Dependencies: X
- Vulnerable Dependencies: Y
- Patches Applied: Z
- MTTR: N days

### Critical Issues
[List any CRITICAL vulnerabilities]

### Trends
[Graph of vulnerability count over time]

### Next Steps
[Planned updates and improvements]
```

## Exception Records

Document all security exceptions here:

| Date | Package | Version | CVE | Severity | Justification | Expiry | Approved By |
|------|---------|---------|-----|----------|---------------|--------|-------------|
| Example: 2025-01-15 | pillow | 9.0.0 | CVE-2025-001 | HIGH | No exploit available, isolated usage | 2025-02-15 | @security-lead |

## Contact

- **Security Team**: security@greenlang.com
- **Dependency Updates**: Create issue with label `dependencies`
- **Emergency**: Page on-call via PagerDuty

## Appendix: Quick Commands

```bash
# Full security check
make security-check

# Update all dependencies safely
pip list --outdated
pip-review --auto --preview

# Generate SBOM
pip install pip-audit
pip-audit --format cyclonedx --output sbom.json

# Check licenses
pip-licenses --format=markdown --output-file=licenses.md

# Find unused dependencies
pip install pip-autoremove
pip-autoremove --list
```