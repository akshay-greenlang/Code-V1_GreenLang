# PRD-SEC-007: Security Scanning Pipeline

**Status:** APPROVED
**Version:** 1.0
**Created:** 2026-02-06
**Priority:** P0 - CRITICAL
**Depends On:** INFRA-007 (CI/CD), SEC-001-006 (Security Foundation)

---

## 1. Overview

### 1.1 Purpose
Formalize, complete, and enhance the security scanning pipeline for GreenLang Climate OS. This PRD documents the existing scanning infrastructure and defines enhancements needed for a production-grade, compliance-ready security program covering SAST, DAST, SCA, container security, supply chain integrity, and vulnerability management.

### 1.2 Current State
Significant security scanning infrastructure exists:
- **GitHub Actions**: `unified-security-scan.yml` (Trivy, Snyk, Bandit, Gitleaks, License, SBOM)
- **IaC Scanning**: `infra-validate.yml` (TFSec, Checkov, Kubeconform)
- **Pre-commit Hooks**: 9 security hook groups in `.pre-commit-config.yaml`
- **Config Files**: `.gitleaks.toml`, `.bandit` configurations
- **K8s Security**: Trivy Operator, Falco rules, network policies
- **Scripts**: `scan_dependencies.py`, `scan_secrets.py`

### 1.3 Scope
- **In Scope:**
  - Unified security scanning orchestration module
  - DAST integration (OWASP ZAP)
  - Advanced SAST (Semgrep, CodeQL)
  - Container image signing (Cosign/Sigstore)
  - SBOM signing and attestation
  - Vulnerability deduplication and management
  - Security findings API and dashboard
  - PII/sensitive data scanning
  - Compliance mapping (SOC 2, ISO 27001, GDPR)
  - Automated remediation workflows
- **Out of Scope:**
  - Penetration testing automation
  - Bug bounty platform integration
  - Third-party security vendor integrations

### 1.4 Success Criteria
- All code changes scanned before merge (SAST, SCA, secrets)
- Container images signed and verified before deployment
- SBOM generated and signed for every release
- Vulnerabilities deduplicated and prioritized by risk
- Security findings queryable via API
- Compliance reports generated automatically
- <5 minute scan time for PR checks
- Zero false positive rate >5%

---

## 2. Technical Requirements

### TR-001: Security Scanning Orchestrator
**Priority:** P0
**Description:** Create a unified Python module to orchestrate all security scanning operations.

**Requirements:**
1. `greenlang/infrastructure/security_scanning/` module:
   - `ScannerConfig` dataclass with tool-specific settings
   - `ScanOrchestrator` class coordinating all scanners
   - `ScanResult` dataclass with findings, severity, tool source
   - `ScanReport` aggregating results across tools
2. Scanner integrations:
   - SAST: Bandit, Semgrep, CodeQL
   - SCA: Trivy, Snyk, pip-audit, Safety
   - Secrets: Gitleaks, TruffleHog
   - Container: Trivy, Grype
   - IaC: TFSec, Checkov
   - License: pip-licenses
3. Deduplication:
   - CVE-based deduplication across scanners
   - Fingerprint-based finding correlation
   - Severity normalization (CVSS 3.1)
4. Output formats:
   - SARIF for GitHub Security tab
   - JSON for API consumption
   - HTML for human review
   - CycloneDX for SBOM

**Acceptance Criteria:**
- [ ] All scanners invokable via unified interface
- [ ] Findings deduplicated by CVE/fingerprint
- [ ] Results available in SARIF, JSON, HTML formats
- [ ] <2 minute orchestration overhead

### TR-002: DAST Integration (OWASP ZAP)
**Priority:** P0
**Description:** Add Dynamic Application Security Testing to scan running applications.

**Requirements:**
1. OWASP ZAP integration:
   - Baseline scan for CI/CD (fast, <5 min)
   - Full scan for nightly (comprehensive)
   - API scan for OpenAPI/GraphQL endpoints
   - Authenticated scan support
2. GitHub Actions workflow:
   - `dast-scan.yml` workflow
   - Triggers on deployment to staging
   - Posts findings to PR as comments
   - Fails on HIGH/CRITICAL findings
3. Configuration:
   - ZAP automation framework YAML
   - Custom scan policies for GreenLang APIs
   - Authentication scripts for JWT
4. K8s integration:
   - OWASP ZAP deployment for continuous scanning
   - ServiceMonitor for Prometheus metrics

**Acceptance Criteria:**
- [ ] Baseline scan <5 minutes
- [ ] Full scan coverage >80% endpoints
- [ ] Findings reported in SARIF format
- [ ] No HIGH/CRITICAL findings in production APIs

### TR-003: Advanced SAST (Semgrep + CodeQL)
**Priority:** P0
**Description:** Enhance SAST with multi-language, semantic analysis.

**Requirements:**
1. Semgrep integration:
   - Custom rules for GreenLang patterns
   - Python, JavaScript, Terraform, Dockerfile coverage
   - CI/CD integration via GitHub Action
   - SARIF output for GitHub Security tab
2. CodeQL integration:
   - CodeQL workflow for Python
   - Custom queries for GreenLang
   - Weekly deep analysis schedule
   - Security advisories integration
3. Rule management:
   - Custom rule repository
   - Rule versioning and deployment
   - False positive tracking
   - Rule effectiveness metrics

**Acceptance Criteria:**
- [ ] Semgrep scanning in <2 minutes
- [ ] CodeQL weekly deep scan
- [ ] Custom rules for top 10 GreenLang patterns
- [ ] False positive rate <5%

### TR-004: Container Image Signing (Cosign/Sigstore)
**Priority:** P0
**Description:** Sign and verify container images for supply chain integrity.

**Requirements:**
1. Cosign integration:
   - Keyless signing via Sigstore/Fulcio
   - Image signing in CI/CD pipeline
   - Signature verification before deployment
   - Public transparency log (Rekor)
2. Kubernetes admission:
   - Kyverno/OPA policy for signature verification
   - Reject unsigned images in production
   - Allow unsigned in dev with warning
3. SLSA provenance:
   - Build provenance attestation (SLSA Level 2)
   - Provenance verification
   - Supply chain security metadata

**Acceptance Criteria:**
- [ ] All production images signed
- [ ] Signatures verifiable via cosign verify
- [ ] Unsigned images rejected in prod
- [ ] SLSA Level 2 provenance attached

### TR-005: SBOM Signing and Attestation
**Priority:** P1
**Description:** Sign SBOMs and attach attestations for compliance.

**Requirements:**
1. SBOM signing:
   - Sign CycloneDX SBOM with Cosign
   - Attach SBOM as image attestation
   - Store in OCI registry
2. SBOM validation:
   - Verify SBOM signature before deployment
   - Check for known vulnerabilities in SBOM components
   - License compliance validation
3. SBOM API:
   - Query SBOM by image digest
   - Search components across all SBOMs
   - Vulnerability correlation

**Acceptance Criteria:**
- [ ] All SBOMs signed and attested
- [ ] SBOM queryable via API
- [ ] Vulnerability correlation automated

### TR-006: Vulnerability Management Service
**Priority:** P1
**Description:** Centralized service for vulnerability tracking and prioritization.

**Requirements:**
1. `greenlang/infrastructure/security_scanning/vulnerability_service.py`:
   - Vulnerability ingestion from all scanners
   - CVE deduplication and correlation
   - Risk scoring (CVSS + exploitability + context)
   - SLA tracking (time to remediation)
2. Database schema:
   - `security.vulnerabilities` table
   - `security.findings` table
   - `security.remediation_sla` table
   - `security.exceptions` table (risk acceptance)
3. API endpoints:
   - `GET /api/v1/security/vulnerabilities` - List with filtering
   - `GET /api/v1/security/vulnerabilities/{id}` - Details
   - `POST /api/v1/security/vulnerabilities/{id}/accept` - Risk acceptance
   - `POST /api/v1/security/vulnerabilities/{id}/remediate` - Mark fixed
   - `GET /api/v1/security/dashboard` - Summary statistics
4. Prioritization:
   - EPSS (Exploit Prediction Scoring System)
   - CISA KEV (Known Exploited Vulnerabilities)
   - Asset criticality weighting

**Acceptance Criteria:**
- [ ] All vulnerabilities centralized
- [ ] Risk scoring includes exploitability
- [ ] SLA tracking with alerts
- [ ] API for querying and management

### TR-007: Security Findings Dashboard
**Priority:** P1
**Description:** Grafana dashboard for security posture visibility.

**Requirements:**
1. Dashboard panels (20+):
   - Vulnerability count by severity (gauge)
   - Vulnerability trend over time (graph)
   - Open vs closed findings (pie)
   - Mean time to remediation (stat)
   - Scanner coverage (table)
   - Top 10 vulnerable dependencies (table)
   - Container image scan results (table)
   - Secret detection events (graph)
   - Compliance score by framework (gauge)
   - SLA breach rate (stat)
   - DAST findings by category (bar)
   - Code quality trends (graph)
2. Prometheus metrics:
   - `gl_security_vulnerabilities_total` (severity, status)
   - `gl_security_scan_duration_seconds` (scanner)
   - `gl_security_findings_total` (scanner, severity)
   - `gl_security_remediation_sla_seconds` (severity)
   - `gl_security_images_signed_total`
   - `gl_security_sbom_components_total`

**Acceptance Criteria:**
- [ ] Dashboard with 20+ panels
- [ ] Real-time vulnerability counts
- [ ] Trend analysis over 90 days
- [ ] Drill-down to individual findings

### TR-008: PII/Sensitive Data Scanning
**Priority:** P1
**Description:** Detect PII and sensitive data in code and data stores.

**Requirements:**
1. PII scanner module:
   - Pattern-based detection (SSN, credit card, email, phone)
   - ML-based entity recognition (Microsoft Presidio)
   - Context-aware false positive reduction
   - Data classification (PII, PHI, PCI)
2. Scan targets:
   - Source code (pre-commit, CI/CD)
   - Log files (Loki integration)
   - Database samples (opt-in)
   - S3 objects (sampling)
3. Alerts:
   - Real-time alerts on PII detection
   - Integration with audit logging
   - Remediation guidance

**Acceptance Criteria:**
- [ ] 95%+ detection rate for known PII patterns
- [ ] <10% false positive rate
- [ ] Integrated with CI/CD and pre-commit
- [ ] Alerts routed to data stewards

### TR-009: Compliance Automation
**Priority:** P2
**Description:** Automated compliance checks and reporting.

**Requirements:**
1. Compliance frameworks:
   - SOC 2 Type II mapping
   - ISO 27001 control mapping
   - GDPR technical controls
   - PCI DSS (if applicable)
2. Automated checks:
   - Security scanning coverage
   - Encryption at rest verification
   - Access control audit
   - Logging completeness
3. Report generation:
   - Evidence collection automation
   - Control effectiveness scoring
   - Gap analysis reports
   - Auditor-ready exports (PDF, Excel)

**Acceptance Criteria:**
- [ ] 80%+ controls automated
- [ ] Reports generated on-demand
- [ ] Evidence collected automatically
- [ ] Auditor-ready format

### TR-010: Automated Remediation
**Priority:** P2
**Description:** Automated fix suggestions and PR creation for vulnerabilities.

**Requirements:**
1. Auto-fix capabilities:
   - Dependency version bumps
   - Secret rotation triggers
   - Configuration hardening
   - License violation resolution
2. PR automation:
   - Create fix PRs for known vulnerabilities
   - Include remediation guidance
   - Link to security advisories
   - Auto-merge for low-risk fixes
3. Notification:
   - Slack/Teams integration
   - Email alerts for critical findings
   - PagerDuty for production issues

**Acceptance Criteria:**
- [ ] Auto-fix PRs for dependency updates
- [ ] Remediation guidance included
- [ ] Notifications to relevant teams

---

## 3. Architecture

### 3.1 Security Scanning Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Developer Workflow                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │   Pre-commit Hooks    │
    │  (Bandit, Gitleaks,   │
    │   TruffleHog, Semgrep)│
    └───────────┬───────────┘
                │
                ▼
    ┌───────────────────────┐
    │    Pull Request       │
    └───────────┬───────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CI/CD Security Scanning                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    SAST     │  │    SCA      │  │   Secrets   │  │  Container  │        │
│  │ Bandit      │  │ Trivy       │  │ Gitleaks    │  │ Trivy       │        │
│  │ Semgrep     │  │ Snyk        │  │ TruffleHog  │  │ Grype       │        │
│  │ CodeQL      │  │ pip-audit   │  │ detect-     │  │ Cosign      │        │
│  └─────────────┘  └─────────────┘  │ secrets     │  │ verify      │        │
│                                    └─────────────┘  └─────────────┘        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    IaC      │  │   License   │  │    SBOM     │  │    PII      │        │
│  │ TFSec       │  │ pip-licenses│  │ CycloneDX   │  │ Presidio    │        │
│  │ Checkov     │  │ License     │  │ Cosign sign │  │ Pattern     │        │
│  │ Kubeconform │  │ Compliance  │  │             │  │ Detection   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │                    Scan Orchestrator                                   │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
    │  │ Deduplication│ │ Correlation │  │ Prioritization│ │ SARIF Gen  │  │
    │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │
    └───────────────────────────────────────────────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │                    Vulnerability Management                            │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │  PostgreSQL: security.vulnerabilities, security.findings        │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
    │  │ Risk Scoring│  │ SLA Tracking│  │ Exceptions  │  │ Remediation │  │
    │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │
    └───────────────────────────────────────────────────────────────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
┌──────────────┐  ┌──────────────┐
│  Grafana     │  │  API         │
│  Dashboard   │  │  Endpoints   │
└──────────────┘  └──────────────┘
```

### 3.2 Staging DAST Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Staging Deployment                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │                         OWASP ZAP                                      │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │
    │  │ Baseline    │  │ API Scan    │  │ Full Scan   │                    │
    │  │ (5 min)     │  │ (OpenAPI)   │  │ (Nightly)   │                    │
    │  └─────────────┘  └─────────────┘  └─────────────┘                    │
    └───────────────────────────────────────────────────────────────────────┘
                │
                ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │                    Findings Processing                                 │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │  SARIF → GitHub Security → Vulnerability Management             │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    └───────────────────────────────────────────────────────────────────────┘
```

---

## 4. Implementation Phases

### Phase 1: Scanning Orchestrator (P0)
- Create `greenlang/infrastructure/security_scanning/` module
- Implement `ScanOrchestrator` with all scanner integrations
- Add deduplication and correlation logic
- SARIF, JSON, HTML output generation

### Phase 2: DAST Integration (P0)
- OWASP ZAP baseline scan workflow
- API scan with OpenAPI schema
- GitHub Actions integration
- K8s deployment for continuous scanning

### Phase 3: Advanced SAST (P0)
- Semgrep CI/CD integration
- Custom Semgrep rules for GreenLang
- CodeQL workflow setup
- Custom CodeQL queries

### Phase 4: Supply Chain Security (P0)
- Cosign image signing in CI/CD
- Kyverno admission policy
- SBOM signing and attestation
- SLSA provenance generation

### Phase 5: Vulnerability Management (P1)
- Database schema for vulnerabilities
- Vulnerability service implementation
- API endpoints for management
- Risk scoring with EPSS/CISA KEV

### Phase 6: Dashboard & Monitoring (P1)
- Prometheus metrics for security
- Grafana dashboard (20+ panels)
- Alert rules for security events
- Integration with existing monitoring

### Phase 7: PII Scanning (P1)
- PII scanner module implementation
- Pattern and ML-based detection
- Integration with CI/CD and Loki
- Alert routing to data stewards

### Phase 8: Compliance & Remediation (P2)
- Compliance framework mapping
- Automated evidence collection
- Auto-fix PR generation
- Notification integrations

### Phase 9: Testing (P2)
- Unit tests for all components
- Integration tests with real scanners
- Load tests for orchestrator
- Compliance verification tests

---

## 5. Security Considerations

### 5.1 Scanner Security
- Scanners run in isolated containers
- No network access except to targets
- Results encrypted at rest
- Findings access controlled by RBAC

### 5.2 Supply Chain
- All scanner images signed and verified
- Pinned versions for reproducibility
- No arbitrary code execution
- Audit trail for all scans

### 5.3 Sensitive Data
- Secrets never logged in findings
- PII redacted in reports
- Findings stored with encryption
- Access logged to audit service

---

## 6. Compliance Mapping

| Requirement | SOC 2 | ISO 27001 | GDPR | PCI DSS |
|-------------|-------|-----------|------|---------|
| Vulnerability scanning | CC7.1 | A.12.6.1 | Art. 32 | 11.2 |
| Code review | CC7.2 | A.14.2.1 | Art. 32 | 6.3.2 |
| Penetration testing | CC7.1 | A.18.2.3 | Art. 32 | 11.3 |
| Secure coding | CC7.1 | A.14.2.5 | Art. 25 | 6.5 |
| Patch management | CC7.1 | A.12.6.1 | Art. 32 | 6.2 |

---

## 7. Deliverables Summary

| Component | Files | Priority |
|-----------|-------|----------|
| Scanning Orchestrator | 8 | P0 |
| DAST Integration | 6 | P0 |
| Advanced SAST | 4 | P0 |
| Supply Chain Security | 6 | P0 |
| Vulnerability Management | 6 | P1 |
| Dashboard & Monitoring | 3 | P1 |
| PII Scanning | 4 | P1 |
| Compliance & Remediation | 4 | P2 |
| Testing | 12 | P2 |
| **TOTAL** | **~53** | - |

---

## 8. Appendix

### A. Scanner Comparison

| Scanner | Type | Speed | Coverage | False Positives |
|---------|------|-------|----------|-----------------|
| Bandit | SAST | Fast | Python | Low |
| Semgrep | SAST | Fast | Multi-lang | Very Low |
| CodeQL | SAST | Slow | Deep | Low |
| Trivy | SCA/Container | Fast | Broad | Low |
| Snyk | SCA | Medium | Broad | Low |
| Gitleaks | Secrets | Fast | Comprehensive | Medium |
| OWASP ZAP | DAST | Medium | APIs/Web | Medium |

### B. Severity Mapping

| Scanner Severity | Normalized | SLA |
|-----------------|------------|-----|
| CRITICAL | P0 | 24 hours |
| HIGH | P1 | 7 days |
| MEDIUM | P2 | 30 days |
| LOW | P3 | 90 days |
| INFO | P4 | Best effort |

### C. SARIF Output Example

```json
{
  "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [
    {
      "tool": {
        "driver": {
          "name": "GreenLang Security Scanner",
          "version": "1.0.0"
        }
      },
      "results": [
        {
          "ruleId": "CVE-2024-1234",
          "level": "error",
          "message": {
            "text": "Vulnerable dependency: requests<2.32.0"
          },
          "locations": [
            {
              "physicalLocation": {
                "artifactLocation": {
                  "uri": "requirements.txt",
                  "uriBaseId": "%SRCROOT%"
                },
                "region": {
                  "startLine": 15
                }
              }
            }
          ]
        }
      ]
    }
  ]
}
```
