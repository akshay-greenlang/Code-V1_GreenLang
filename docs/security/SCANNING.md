# Security Scanning Guide

## Overview
This guide covers how to run security scans locally and interpret results.

## Secret Scanning with TruffleHog

### Installation
```bash
# Install via pip
pip install trufflehog3

# Or via pipx (recommended)
pipx install trufflehog3

# Or via Docker
docker pull trufflesecurity/trufflehog:latest
```

### Local Scanning

#### Quick Scan (Current Directory)
```bash
# Basic scan with JSON output
trufflehog3 . -o scan-results.json --format json

# Scan with specific rules
trufflehog3 . --rules rules.json

# Scan specific file types only
trufflehog3 . --include "*.py,*.js,*.env"

# Exclude paths
trufflehog3 . --exclude ".git,node_modules,venv"
```

#### Git History Scan
```bash
# Full history scan
git log --all --format=%H | while read commit; do
  echo "Scanning commit: $commit"
  git checkout $commit 2>/dev/null
  trufflehog3 . -o "scan-$commit.json" --format json
done
git checkout main  # Return to main branch

# Scan since specific commit
FIRST_COMMIT=$(git rev-list --max-parents=0 HEAD)
trufflehog3 . --since-commit "$FIRST_COMMIT"
```

#### Pre-commit Hook
```bash
# Add to .git/hooks/pre-commit
#!/bin/bash
echo "Running secret scan..."
trufflehog3 --staged --format json -o /tmp/pre-commit-scan.json

if [ -s /tmp/pre-commit-scan.json ]; then
  echo "❌ Potential secrets detected!"
  cat /tmp/pre-commit-scan.json
  exit 1
fi
echo "✅ No secrets detected"
```

### Interpreting Results

#### Understanding Finding Severity
```json
{
  "path": "config/database.yml",
  "line": 5,
  "type": "Password",
  "content": "password: mysecretpass123",
  "reason": "High entropy string",
  "confidence": "HIGH"
}
```

**Confidence Levels:**
- **HIGH**: Almost certainly a real secret (90%+ confidence)
- **MEDIUM**: Likely a secret, needs review (60-90% confidence)
- **LOW**: Possible false positive (30-60% confidence)

#### Common False Positives
1. **Example/test keys**: Often contain "example", "test", "demo"
2. **Lorem ipsum text**: High entropy but meaningless
3. **Base64 encoded non-secrets**: Images, fonts, etc.
4. **UUIDs/GUIDs**: Random but not secret

### Creating Custom Rules

Create `trufflehog-rules.json`:
```json
{
  "rules": [
    {
      "name": "GreenLang API Key",
      "pattern": "GL_[A-Z0-9]{32}",
      "confidence": "HIGH",
      "severity": "CRITICAL"
    },
    {
      "name": "Internal Token",
      "pattern": "token['\"]?[:\\s]+['\"]?[a-zA-Z0-9]{40}",
      "confidence": "MEDIUM",
      "severity": "HIGH"
    }
  ]
}
```

## Dependency Scanning

### Using pip-audit

#### Installation
```bash
pip install pip-audit
```

#### Basic Usage
```bash
# Scan current environment
pip-audit

# Scan requirements file
pip-audit -r requirements.txt

# Output as JSON
pip-audit --format json -o audit-results.json

# Strict mode (fail on any vulnerability)
pip-audit --strict

# Include description of vulnerabilities
pip-audit --desc
```

#### Understanding Results
```
Found 2 known vulnerabilities in 1 package

Name       Version  ID             Fix Versions
---------- -------- -------------- ------------
requests   2.25.0   PYSEC-2021-1   2.25.1

Description:
The Requests library through 2.25.0 has an issue with
improper handling of Transfer-Encoding headers...
```

**Severity Interpretation:**
- **CRITICAL**: Exploit available, immediate action required
- **HIGH**: Serious vulnerability, patch ASAP
- **MEDIUM**: Should be fixed in next release
- **LOW**: Consider fixing if convenient

### Docker Image Scanning with Trivy

#### Installation
```bash
# Install via script
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Or via Docker
docker pull aquasecurity/trivy:latest
```

#### Scanning Docker Images
```bash
# Build and scan local image
docker build -t myapp:latest .
trivy image myapp:latest

# Scan with specific severity
trivy image --severity HIGH,CRITICAL myapp:latest

# Output as JSON
trivy image -f json -o trivy-results.json myapp:latest

# Scan and fail on vulnerabilities
trivy image --exit-code 1 --severity HIGH,CRITICAL myapp:latest
```

#### Scanning Other Targets
```bash
# Scan filesystem
trivy fs .

# Scan git repository
trivy repo https://github.com/user/repo

# Scan Kubernetes cluster
trivy k8s --report summary cluster
```

## CI/CD Integration

### GitHub Actions Status Checks
The following checks run automatically:
1. **secret-scan**: On every PR and weekly full scan
2. **pip-audit**: On every PR and nightly
3. **trivy-scan**: On every PR for Docker images

### Manual Workflow Triggers
```bash
# Trigger secret scan manually
gh workflow run secret-scan

# Trigger dependency audit
gh workflow run pip-audit

# Trigger container scan
gh workflow run trivy-scan
```

## Handling Findings

### Priority Matrix
| Severity | In Production | In Development | In Test |
|----------|--------------|----------------|---------|
| CRITICAL | Immediate | Within 24h | Within 3 days |
| HIGH | Within 24h | Within 3 days | Within 1 week |
| MEDIUM | Within 1 week | Within 2 weeks | Next release |
| LOW | Next release | Track only | Optional |

### Suppression (When Necessary)

#### TruffleHog Suppressions
Create `.trufflehogignore`:
```
# Ignore test fixtures
tests/fixtures/fake_keys.txt

# Ignore specific finding by hash
finding:abc123def456

# Ignore by pattern
*.test
```

#### pip-audit Suppressions
```bash
# Ignore specific vulnerability
pip-audit --ignore-vuln PYSEC-2021-1

# Or in pyproject.toml
[tool.pip-audit]
ignore-vulns = ["PYSEC-2021-1"]
```

#### Trivy Suppressions
Create `.trivyignore`:
```
# Accept risk until 2025-03-01
CVE-2021-12345 exp:2025-03-01

# Accept specific CVE with justification
CVE-2021-67890 # False positive, not applicable to our usage
```

## Metrics and Reporting

### Dashboard Queries
```sql
-- Weekly security metrics
SELECT
  week,
  COUNT(CASE WHEN severity = 'CRITICAL' THEN 1 END) as critical,
  COUNT(CASE WHEN severity = 'HIGH' THEN 1 END) as high,
  COUNT(CASE WHEN severity = 'MEDIUM' THEN 1 END) as medium,
  AVG(time_to_fix_hours) as mttr
FROM security_findings
GROUP BY week
ORDER BY week DESC;
```

### KPIs to Track
1. **Mean Time to Remediation (MTTR)**: Target < 24h for CRITICAL
2. **Open vulnerability count**: Trend should be downward
3. **False positive rate**: Should decrease over time
4. **Scan coverage**: 100% of repos and images

## Troubleshooting

### Common Issues

#### TruffleHog consuming too much memory
```bash
# Limit memory usage
ulimit -v 2097152  # 2GB limit
trufflehog3 .

# Or scan in chunks
find . -type f -name "*.py" | xargs -n 10 trufflehog3
```

#### pip-audit fails with dependency conflicts
```bash
# Use isolated environment
python -m venv scan-env
source scan-env/bin/activate
pip install pip-audit
pip-audit -r requirements.txt
```

#### Trivy rate limiting
```bash
# Use local DB
trivy image --download-db-only
trivy image --skip-update myapp:latest

# Or use cache
export TRIVY_CACHE_DIR=/tmp/trivy-cache
trivy image myapp:latest
```

## References
- [TruffleHog Documentation](https://github.com/trufflesecurity/trufflehog)
- [pip-audit Documentation](https://github.com/pypa/pip-audit)
- [Trivy Documentation](https://aquasecurity.github.io/trivy/)
- [OWASP Dependency Check](https://owasp.org/www-project-dependency-check/)
- [GitHub Advanced Security](https://docs.github.com/en/code-security)