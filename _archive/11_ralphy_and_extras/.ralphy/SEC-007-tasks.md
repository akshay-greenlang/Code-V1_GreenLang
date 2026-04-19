# SEC-007: Security Scanning Pipeline - Development Tasks

**Status:** COMPLETE
**Created:** 2026-02-06
**Completed:** 2026-02-06
**Priority:** P0 - CRITICAL
**Depends On:** INFRA-007, SEC-001-006
**Existing Code:** unified-security-scan.yml, infra-validate.yml, pre-commit hooks, trivy/falco configs
**Result:** 55+ new files, ~22,000 lines, 250+ tests

---

## Phase 1: Scanning Orchestrator (P0) - COMPLETE

### 1.1 Package Init
- [x] Create `greenlang/infrastructure/security_scanning/__init__.py`:
  - Public API exports: ScanOrchestrator, ScanResult, ScanReport, ScannerConfig
  - Version constant
  - Scanner availability flags

### 1.2 Scanner Configuration
- [x] Create `greenlang/infrastructure/security_scanning/config.py`:
  - `ScannerConfig` dataclass with tool-specific settings
  - `ScanOrchestratorConfig` with orchestration settings
  - Environment variable mapping
  - Severity thresholds, timeout settings

### 1.3 Scan Result Models
- [x] Create `greenlang/infrastructure/security_scanning/models.py`:
  - `ScanResult` dataclass (finding_id, severity, tool, cve, description, location)
  - `ScanFinding` with detailed vulnerability info
  - `ScanReport` aggregating results across tools
  - `VulnerabilityInfo` with CVSS, EPSS, references
  - `RemediationInfo` with fix suggestions

### 1.4 Scanner Base Class
- [x] Create `greenlang/infrastructure/security_scanning/scanners/base.py`:
  - `BaseScanner` abstract class
  - `scan()`, `parse_results()`, `to_sarif()` abstract methods
  - Common utilities for subprocess execution
  - Timeout handling, error recovery

### 1.5 SAST Scanners
- [x] Create `greenlang/infrastructure/security_scanning/scanners/sast.py`:
  - `BanditScanner` class wrapping Bandit CLI
  - `SemgrepScanner` class wrapping Semgrep CLI
  - `CodeQLScanner` class for CodeQL analysis
  - SARIF output parsing for each

### 1.6 SCA Scanners
- [x] Create `greenlang/infrastructure/security_scanning/scanners/sca.py`:
  - `TrivyScanner` class for dependency scanning
  - `SnykScanner` class with Snyk CLI
  - `PipAuditScanner` class for pip-audit
  - `SafetyScanner` class for Safety
  - CVE deduplication across scanners

### 1.7 Secret Scanners
- [x] Create `greenlang/infrastructure/security_scanning/scanners/secrets.py`:
  - `GitleaksScanner` class
  - `TrufflehogScanner` class
  - `DetectSecretsScanner` class
  - Entropy analysis integration

### 1.8 Container Scanners
- [x] Create `greenlang/infrastructure/security_scanning/scanners/container.py`:
  - `TrivyContainerScanner` for image scanning
  - `GrypeScanner` for Anchore Grype
  - `CosignVerifier` for signature verification
  - Layer-by-layer vulnerability tracking

### 1.9 IaC Scanners
- [x] Create `greenlang/infrastructure/security_scanning/scanners/iac.py`:
  - `TfsecScanner` for Terraform
  - `CheckovScanner` for multi-IaC
  - `KubeconformScanner` for K8s manifests
  - Policy violation categorization

### 1.10 Deduplication Engine
- [x] Create `greenlang/infrastructure/security_scanning/deduplication.py`:
  - `DeduplicationEngine` class
  - CVE-based correlation
  - Fingerprint-based finding matching
  - Cross-scanner result merging
  - Severity normalization to CVSS 3.1

### 1.11 SARIF Generator
- [x] Create `greenlang/infrastructure/security_scanning/sarif_generator.py`:
  - `SARIFGenerator` class
  - Unified SARIF 2.1.0 output
  - GitHub Security tab integration
  - Custom rules metadata

### 1.12 Scan Orchestrator
- [x] Create `greenlang/infrastructure/security_scanning/orchestrator.py`:
  - `ScanOrchestrator` class coordinating all scanners
  - Parallel scan execution
  - Result aggregation and deduplication
  - HTML, JSON, SARIF report generation
  - Global instance management

---

## Phase 2: DAST Integration (P0) - COMPLETE

### 2.1 OWASP ZAP Scanner
- [x] Create `greenlang/infrastructure/security_scanning/scanners/dast.py`:
  - `ZAPScanner` class
  - Baseline scan mode (fast)
  - Full scan mode (comprehensive)
  - API scan mode (OpenAPI/GraphQL)
  - Authenticated scan support

### 2.2 ZAP Automation Framework
- [x] Create `deployment/security/dast/zap-automation.yaml`:
  - ZAP Automation Framework configuration
  - Baseline scan profile
  - Full scan profile
  - API scan profile
  - Custom scan policies

### 2.3 ZAP GitHub Action
- [x] Create `.github/workflows/dast-scan.yml`:
  - Trigger on staging deployment
  - Baseline scan for PRs
  - Full scan nightly
  - SARIF output to GitHub Security
  - Fail on HIGH/CRITICAL

### 2.4 ZAP K8s Deployment
- [x] Create `deployment/kubernetes/security-scanning/zap-deployment.yaml`:
  - OWASP ZAP deployment
  - ServiceMonitor for Prometheus
  - NetworkPolicy for scan targets
  - ConfigMap for scan profiles

### 2.5 ZAP Authentication Scripts
- [x] Create `deployment/security/dast/auth-scripts/`:
  - `jwt_auth.py` - JWT authentication hook for ZAP

### 2.6 ZAP Custom Policies
- [x] Create `deployment/security/dast/policies/`:
  - `greenlang-api-policy.xml` - API-specific rules

---

## Phase 3: Advanced SAST (P0) - COMPLETE

### 3.1 Semgrep Integration
- [x] Update `.github/workflows/unified-security-scan.yml`:
  - Add Semgrep job
  - SARIF output to GitHub Security
  - Custom rules integration

### 3.2 Custom Semgrep Rules
- [x] Create `deployment/security/semgrep-rules/`:
  - `greenlang-security.yaml` - GreenLang-specific rules (40+ rules)
  - `python-security.yaml` - Enhanced Python rules (35+ rules)
  - `terraform-security.yaml` - IaC security rules (30+ rules)
  - `dockerfile-security.yaml` - Container best practices (25+ rules)

### 3.3 CodeQL Workflow
- [x] Create `.github/workflows/codeql-analysis.yml`:
  - Python CodeQL analysis
  - Weekly deep scan schedule (Sunday 2 AM UTC)
  - Custom queries integration
  - Security advisory correlation

### 3.4 Custom CodeQL Queries
- [x] Create `deployment/security/codeql-queries/`:
  - `greenlang-queries.qls` - Query suite definition
  - `injection-vulnerabilities.ql` - SQL, command, LDAP injection detection
  - `auth-bypass.ql` - Authentication bypass, hardcoded credentials
  - `data-exposure.ql` - PII exposure, stack traces, sensitive data

---

## Phase 4: Supply Chain Security (P0) - COMPLETE

### 4.1 Cosign Image Signing
- [x] Create `.github/workflows/docker-build.yml`:
  - Add Cosign signing step after docker push
  - Keyless signing via Sigstore/Fulcio
  - Push signature to registry
  - Log to Rekor transparency log
  - SBOM generation with Syft
  - SLSA provenance generation

### 4.2 Cosign Verification Action
- [x] Create `.github/actions/cosign-verify/action.yml`:
  - Composite action for signature verification
  - Input: image reference
  - Verify signature, SBOM, and provenance
  - Configurable failure modes
  - Supports cache results

### 4.3 Kyverno Admission Policy
- [x] Create `deployment/kubernetes/security-scanning/kyverno-policies/`:
  - `require-image-signature.yaml` - ClusterPolicy to reject unsigned images
  - `verify-sbom-attestation.yaml` - Require SBOM attestation
  - `verify-provenance.yaml` - SLSA provenance check
  - `kustomization.yaml` - Include all policies

### 4.4 SBOM Signing
- [x] Create `greenlang/infrastructure/security_scanning/sbom_signing.py`:
  - `SBOMSigner` class
  - Sign CycloneDX SBOM with Cosign
  - Attach as OCI attestation
  - Store in registry
  - Verification support

### 4.5 SLSA Provenance
- [x] Update `.github/workflows/release-orchestration.yml`:
  - Add SLSA provenance generation
  - Use Cosign for attestation
  - Attach provenance attestation
  - SLSA Level 2 compliance

### 4.6 Supply Chain Verification
- [x] Create `greenlang/infrastructure/security_scanning/supply_chain.py`:
  - `SupplyChainVerifier` class
  - `verify_image_signature(image_ref) -> bool`
  - `verify_sbom_attestation(image_ref) -> SBOMInfo`
  - `verify_provenance(image_ref) -> ProvenanceInfo`
  - Integration with Cosign CLI
  - Configurable trust policies

---

## Phase 5: Vulnerability Management (P1) - COMPLETE

### 5.1 Database Schema
- [x] Create `deployment/database/migrations/sql/V015__security_scanning.sql`:
  - `security.vulnerabilities` table (id, cve, severity, status, discovered_at)
  - `security.findings` table (id, vulnerability_id, tool, location, raw_data)
  - `security.remediation_sla` table (severity, max_days)
  - `security.exceptions` table (vulnerability_id, reason, expires_at, approved_by)
  - `security.scan_runs` table (id, scanner, started_at, completed_at, findings_count)
  - `security.pii_findings` table for PII-specific tracking
  - Indexes and RLS policies

### 5.2 Vulnerability Service
- [x] Create `greenlang/infrastructure/security_scanning/vulnerability_service.py`:
  - `VulnerabilityService` class
  - Ingest findings from all scanners
  - CVE correlation and deduplication
  - Risk scoring (CVSS + EPSS + CISA KEV)
  - SLA tracking and alerting

### 5.3 Risk Scoring Engine
- [x] Create `greenlang/infrastructure/security_scanning/risk_scoring.py`:
  - `RiskScorer` class
  - CVSS 3.1 base score
  - EPSS exploitability adjustment
  - CISA KEV known exploited check
  - Asset criticality weighting
  - Context-aware prioritization

### 5.4 Vulnerability API Routes
- [x] Create `greenlang/infrastructure/security_scanning/api/vulnerabilities_routes.py`:
  - `GET /api/v1/security/vulnerabilities` - List with filters
  - `GET /api/v1/security/vulnerabilities/{id}` - Details
  - `POST /api/v1/security/vulnerabilities/{id}/accept` - Risk acceptance
  - `POST /api/v1/security/vulnerabilities/{id}/remediate` - Mark fixed
  - `GET /api/v1/security/vulnerabilities/stats` - Statistics

### 5.5 Scan API Routes
- [x] Create `greenlang/infrastructure/security_scanning/api/scans_routes.py`:
  - `POST /api/v1/security/scans` - Trigger scan
  - `GET /api/v1/security/scans` - List scan runs
  - `GET /api/v1/security/scans/{id}` - Scan details
  - `GET /api/v1/security/scans/{id}/findings` - Scan findings

### 5.6 Dashboard API
- [x] Create `greenlang/infrastructure/security_scanning/api/dashboard_routes.py`:
  - `GET /api/v1/security/dashboard` - Summary statistics
  - `GET /api/v1/security/dashboard/trends` - Trend data
  - `GET /api/v1/security/dashboard/coverage` - Scanner coverage
  - `GET /api/v1/security/dashboard/sla` - SLA compliance

---

## Phase 6: Dashboard & Monitoring (P1) - COMPLETE

### 6.1 Security Metrics
- [x] Create `greenlang/infrastructure/security_scanning/metrics.py`:
  - `gl_security_vulnerabilities_total` Counter (severity, status)
  - `gl_security_scan_duration_seconds` Histogram (scanner)
  - `gl_security_findings_total` Counter (scanner, severity)
  - `gl_security_remediation_days` Histogram (severity)
  - `gl_security_images_signed_total` Counter
  - `gl_security_sbom_components_total` Gauge
  - `gl_security_sla_breach_total` Counter (severity)
  - `gl_security_pii_findings_total` Counter
  - Lazy initialization pattern

### 6.2 Grafana Dashboard
- [x] Create `deployment/monitoring/dashboards/security-scanning.json`:
  - 28 panels across 8 rows
  - Vulnerability Overview, Trends, Remediation metrics
  - Scanner coverage, Top Issues, Container Security
  - PII & Secrets detection, Compliance scores
  - Template variables: $datasource, $severity, $scanner

### 6.3 Prometheus Alerts
- [x] Create `deployment/monitoring/alerts/security-scanning-alerts.yaml`:
  - 17 alert rules across 7 groups
  - CriticalVulnerabilityDetected
  - HighVulnerabilitySLABreach
  - ScannerFailed
  - UnsignedImageDeployed
  - SecretDetected
  - DASTHighFinding
  - PIIDetected, PHIDetected
  - KEVDetected (CISA known exploited)
  - ComplianceScoreDrop

---

## Phase 7: PII Scanning (P1) - COMPLETE

### 7.1 PII Scanner Module
- [x] Create `greenlang/infrastructure/security_scanning/pii_scanner.py`:
  - `PIIScanner` class
  - Pattern-based detection (SSN, credit card with Luhn, email, phone)
  - 15+ regex patterns with context validation
  - False positive reduction logic
  - Data classification (PII, PHI, PCI)

### 7.2 ML-based PII Detection
- [x] Create `greenlang/infrastructure/security_scanning/pii_ml.py`:
  - `PresidioPIIScanner` class (Microsoft Presidio integration)
  - `HybridPIIScanner` combining regex + ML
  - Named entity recognition
  - Custom entity types for GreenLang
  - Confidence scoring

### 7.3 PII Scan Integration
- [x] Update scanning orchestrator:
  - Add PII scanner to pipeline
  - `scan_pii()` method for dedicated PII scans
  - `scan_with_pii()` for unified scanning
  - PII-to-ScanFinding conversion

### 7.4 PII Alert Routing
- [x] Create `greenlang/infrastructure/security_scanning/pii_alerts.py`:
  - `PIIAlertRouter` class
  - Route alerts to data stewards by classification
  - PII -> Legal, PHI -> Compliance, PCI -> Security
  - Integration with audit logging
  - Remediation guidance generation

---

## Phase 8: Compliance & Remediation (P2) - COMPLETE

### 8.1 Compliance Framework Mapping
- [x] Create `greenlang/infrastructure/security_scanning/compliance/`:
  - `__init__.py` - Exports
  - `base.py` - ComplianceFramework abstract class
  - `soc2_mapping.py` - SOC 2 Type II controls (CC6.1, CC7.1, CC7.2)
  - `iso27001_mapping.py` - ISO 27001 controls (A.12.6.1, A.14.2.1)
  - `gdpr_mapping.py` - GDPR technical controls (Art. 25, 32)

### 8.2 Automated Evidence Collection
- [x] Create `greenlang/infrastructure/security_scanning/compliance/evidence_collector.py`:
  - `EvidenceCollector` class
  - collect_scan_evidence() - Aggregate scan results
  - collect_config_evidence() - Configuration snapshots
  - collect_access_evidence() - RBAC audit
  - collect_encryption_evidence() - Encryption status
  - generate_evidence_package(framework, period) -> EvidencePackage

### 8.3 Auto-Fix PR Generator
- [x] Create `greenlang/infrastructure/security_scanning/remediation/auto_fix.py`:
  - `AutoFixGenerator` class
  - generate_dependency_fix(vuln) -> FixPR
  - generate_secret_rotation(finding) -> RotationTask
  - generate_config_fix(finding) -> ConfigPatch
  - create_github_pr(fix) -> PR URL

### 8.4 Notification Integration
- [x] Create `greenlang/infrastructure/security_scanning/notifications.py`:
  - `NotificationService` class
  - send_slack(channel, message, severity)
  - send_email(to, subject, body)
  - send_pagerduty(severity, title, details)
  - send_teams(webhook, message)

---

## Phase 9: Integration (P1) - COMPLETE

### 9.1 Auth Setup Integration
- [x] Modify `greenlang/infrastructure/auth_service/auth_setup.py`:
  - Import security_scanning routers (with SECURITY_SCANNING_AVAILABLE flag)
  - Include security_router at /api/v1/security prefix
  - Wire VulnerabilityService dependency

### 9.2 Route Protector Update
- [x] Update `greenlang/infrastructure/auth_service/route_protector.py`:
  - Add security permission mappings to PERMISSION_MAP:
    - GET:/api/v1/security/vulnerabilities -> security:read
    - POST:/api/v1/security/vulnerabilities/{id}/accept -> security:admin
    - POST:/api/v1/security/scans -> security:scan
    - GET:/api/v1/security/dashboard -> security:read

### 9.3 API Init
- [x] Create `greenlang/infrastructure/security_scanning/api/__init__.py`:
  - Import vulnerabilities_routes, scans_routes, dashboard_routes
  - Create combined security_router
  - Export security_router

---

## Phase 10: Testing (P2) - COMPLETE

### 10.1 Unit Tests
- [x] Create `tests/unit/security_scanning/__init__.py`
- [x] Create `tests/unit/security_scanning/conftest.py` - Shared fixtures
- [x] Create `tests/unit/security_scanning/test_orchestrator.py` - 30+ tests
- [x] Create `tests/unit/security_scanning/test_scanners.py` - 40+ tests
- [x] Create `tests/unit/security_scanning/test_deduplication.py` - 20+ tests
- [x] Create `tests/unit/security_scanning/test_vulnerability_service.py` - 25+ tests
- [x] Create `tests/unit/security_scanning/test_risk_scoring.py` - 20+ tests
- [x] Create `tests/unit/security_scanning/test_pii_scanner.py` - 25+ tests
- [x] Create `tests/unit/security_scanning/test_routes.py` - 30+ tests

### 10.2 Integration Tests
- [x] Create `tests/integration/security_scanning/__init__.py`
- [x] Create `tests/integration/security_scanning/conftest.py`
- [x] Create `tests/integration/security_scanning/test_scanner_integration.py` - 20+ tests
- [x] Create `tests/integration/security_scanning/test_dast_integration.py` - 15+ tests
- [x] Create `tests/integration/security_scanning/test_supply_chain.py` - 15+ tests

### 10.3 Load Tests
- [x] Create `tests/load/security_scanning/__init__.py`
- [x] Create `tests/load/security_scanning/conftest.py`
- [x] Create `tests/load/security_scanning/test_scan_throughput.py` - 10+ tests

---

## Summary

| Phase | Tasks | Priority | Status |
|-------|-------|----------|--------|
| Phase 1: Scanning Orchestrator | 12/12 | P0 | COMPLETE |
| Phase 2: DAST Integration | 6/6 | P0 | COMPLETE |
| Phase 3: Advanced SAST | 4/4 | P0 | COMPLETE |
| Phase 4: Supply Chain Security | 6/6 | P0 | COMPLETE |
| Phase 5: Vulnerability Management | 6/6 | P1 | COMPLETE |
| Phase 6: Dashboard & Monitoring | 3/3 | P1 | COMPLETE |
| Phase 7: PII Scanning | 4/4 | P1 | COMPLETE |
| Phase 8: Compliance & Remediation | 4/4 | P2 | COMPLETE |
| Phase 9: Integration | 3/3 | P1 | COMPLETE |
| Phase 10: Testing | 14/14 | P2 | COMPLETE |
| **TOTAL** | **62/62** | - | **COMPLETE** |

---

## Files Created

### Python Modules (30+ files)
- `greenlang/infrastructure/security_scanning/__init__.py`
- `greenlang/infrastructure/security_scanning/config.py`
- `greenlang/infrastructure/security_scanning/models.py`
- `greenlang/infrastructure/security_scanning/orchestrator.py`
- `greenlang/infrastructure/security_scanning/deduplication.py`
- `greenlang/infrastructure/security_scanning/sarif_generator.py`
- `greenlang/infrastructure/security_scanning/vulnerability_service.py`
- `greenlang/infrastructure/security_scanning/risk_scoring.py`
- `greenlang/infrastructure/security_scanning/metrics.py`
- `greenlang/infrastructure/security_scanning/pii_scanner.py`
- `greenlang/infrastructure/security_scanning/pii_ml.py`
- `greenlang/infrastructure/security_scanning/pii_alerts.py`
- `greenlang/infrastructure/security_scanning/sbom_signing.py`
- `greenlang/infrastructure/security_scanning/supply_chain.py`
- `greenlang/infrastructure/security_scanning/notifications.py`
- `greenlang/infrastructure/security_scanning/scanners/base.py`
- `greenlang/infrastructure/security_scanning/scanners/sast.py`
- `greenlang/infrastructure/security_scanning/scanners/sca.py`
- `greenlang/infrastructure/security_scanning/scanners/secrets.py`
- `greenlang/infrastructure/security_scanning/scanners/container.py`
- `greenlang/infrastructure/security_scanning/scanners/iac.py`
- `greenlang/infrastructure/security_scanning/scanners/dast.py`
- `greenlang/infrastructure/security_scanning/api/__init__.py`
- `greenlang/infrastructure/security_scanning/api/vulnerabilities_routes.py`
- `greenlang/infrastructure/security_scanning/api/scans_routes.py`
- `greenlang/infrastructure/security_scanning/api/dashboard_routes.py`
- `greenlang/infrastructure/security_scanning/compliance/__init__.py`
- `greenlang/infrastructure/security_scanning/compliance/base.py`
- `greenlang/infrastructure/security_scanning/compliance/soc2_mapping.py`
- `greenlang/infrastructure/security_scanning/compliance/iso27001_mapping.py`
- `greenlang/infrastructure/security_scanning/compliance/gdpr_mapping.py`
- `greenlang/infrastructure/security_scanning/compliance/evidence_collector.py`
- `greenlang/infrastructure/security_scanning/remediation/auto_fix.py`

### GitHub Workflows & Actions (4 files)
- `.github/workflows/dast-scan.yml`
- `.github/workflows/codeql-analysis.yml`
- `.github/workflows/docker-build.yml`
- `.github/actions/cosign-verify/action.yml`

### Semgrep Rules (4 files)
- `deployment/security/semgrep-rules/greenlang-security.yaml`
- `deployment/security/semgrep-rules/python-security.yaml`
- `deployment/security/semgrep-rules/terraform-security.yaml`
- `deployment/security/semgrep-rules/dockerfile-security.yaml`

### CodeQL Queries (4 files)
- `deployment/security/codeql-queries/greenlang-queries.qls`
- `deployment/security/codeql-queries/injection-vulnerabilities.ql`
- `deployment/security/codeql-queries/auth-bypass.ql`
- `deployment/security/codeql-queries/data-exposure.ql`

### Kubernetes Manifests (5 files)
- `deployment/kubernetes/security-scanning/zap-deployment.yaml`
- `deployment/kubernetes/security-scanning/kyverno-policies/require-image-signature.yaml`
- `deployment/kubernetes/security-scanning/kyverno-policies/verify-sbom-attestation.yaml`
- `deployment/kubernetes/security-scanning/kyverno-policies/verify-provenance.yaml`
- `deployment/kubernetes/security-scanning/kyverno-policies/kustomization.yaml`

### DAST Configuration (3 files)
- `deployment/security/dast/zap-automation.yaml`
- `deployment/security/dast/auth-scripts/jwt_auth.py`
- `deployment/security/dast/policies/greenlang-api-policy.xml`

### Database Migration (1 file)
- `deployment/database/migrations/sql/V015__security_scanning.sql`

### Monitoring (2 files)
- `deployment/monitoring/dashboards/security-scanning.json`
- `deployment/monitoring/alerts/security-scanning-alerts.yaml`

### Tests (17 files)
- `tests/unit/security_scanning/__init__.py`
- `tests/unit/security_scanning/conftest.py`
- `tests/unit/security_scanning/test_orchestrator.py`
- `tests/unit/security_scanning/test_scanners.py`
- `tests/unit/security_scanning/test_deduplication.py`
- `tests/unit/security_scanning/test_vulnerability_service.py`
- `tests/unit/security_scanning/test_risk_scoring.py`
- `tests/unit/security_scanning/test_pii_scanner.py`
- `tests/unit/security_scanning/test_routes.py`
- `tests/integration/security_scanning/__init__.py`
- `tests/integration/security_scanning/conftest.py`
- `tests/integration/security_scanning/test_scanner_integration.py`
- `tests/integration/security_scanning/test_dast_integration.py`
- `tests/integration/security_scanning/test_supply_chain.py`
- `tests/load/security_scanning/__init__.py`
- `tests/load/security_scanning/conftest.py`
- `tests/load/security_scanning/test_scan_throughput.py`

### Modified Files (3 files)
- `greenlang/infrastructure/auth_service/auth_setup.py`
- `greenlang/infrastructure/auth_service/route_protector.py`
- `.github/workflows/unified-security-scan.yml`
