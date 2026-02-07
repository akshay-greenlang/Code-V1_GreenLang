# PRD-SEC-009: SOC 2 Type II Audit Preparation

**Version:** 1.0
**Status:** APPROVED
**Author:** Security & Compliance Team
**Created:** 2026-02-06
**Last Updated:** 2026-02-06

---

## 1. Executive Summary

### 1.1 Purpose

This PRD defines the operational preparation system for SOC 2 Type II audit execution. While SEC-001 through SEC-008 built the technical controls and policy framework achieving 100% criteria coverage, SEC-009 focuses on the **audit execution infrastructure** - the tools, processes, and operational procedures required to successfully complete a SOC 2 Type II audit.

### 1.2 Scope

- Audit preparation toolkit and automation
- Auditor portal with secure access management
- Evidence packaging and delivery system
- Pre-audit self-assessment and readiness scoring
- Internal control testing program
- Audit project management and timeline tracking
- Management attestation workflow
- Findings management and remediation tracking
- Continuous compliance monitoring dashboard

### 1.3 Success Criteria

| Metric | Target |
|--------|--------|
| Self-assessment coverage | 100% of 48 criteria |
| Evidence collection automation | >80% automated |
| Auditor request SLA compliance | 95% within SLA |
| Pre-audit readiness score | >95% |
| Control testing completion | 100% before audit |
| Management attestation | 100% signed |

---

## 2. Background

### 2.1 Current State

GreenLang has achieved 100% SOC 2 criteria mapping through:
- SEC-001: JWT Authentication (CC6 controls)
- SEC-002: RBAC Authorization (CC6.1 controls)
- SEC-003: Encryption at Rest (C1 controls)
- SEC-004: TLS 1.3 (CC6.7 controls)
- SEC-005: Audit Logging (CC7, CC8 controls)
- SEC-006: Secrets Management (CC6.1 controls)
- SEC-007: Security Scanning (CC7.1 controls)
- SEC-008: Security Policies (CC1-CC5 controls)

**What's Missing:**
1. No operational toolkit to execute the audit
2. No auditor portal for secure evidence sharing
3. No pre-audit self-assessment system
4. No internal control testing automation
5. No audit project management system
6. No management attestation workflow
7. No findings remediation tracker
8. No continuous compliance dashboard

### 2.2 SOC 2 Type II Requirements

| Component | Requirement |
|-----------|-------------|
| Audit Period | 12 months minimum |
| Trust Services | Security (required), Availability, Confidentiality, Processing Integrity, Privacy (optional) |
| Evidence Types | Policies, procedures, population samples, configuration evidence, logs |
| Testing | Design effectiveness + Operating effectiveness |
| Output | Type II Report with auditor opinion |

---

## 3. Technical Architecture

### 3.1 System Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SOC 2 Audit Preparation Platform                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │  Self-Assessment │  │  Evidence Mgmt  │  │  Auditor Portal │          │
│  │     Engine       │  │     System      │  │    (Read-Only)  │          │
│  │                  │  │                 │  │                 │          │
│  │ - 48 Criteria    │  │ - Auto-collect  │  │ - SSO/MFA       │          │
│  │ - Scoring        │  │ - Package       │  │ - File sharing  │          │
│  │ - Gap detection  │  │ - Versioning    │  │ - Request mgmt  │          │
│  └────────┬─────────┘  └────────┬────────┘  └────────┬────────┘          │
│           │                     │                     │                   │
│           └─────────────────────┼─────────────────────┘                   │
│                                 │                                         │
│  ┌─────────────────┐  ┌────────┴────────┐  ┌─────────────────┐          │
│  │ Control Testing │  │   PostgreSQL    │  │  Findings Mgmt  │          │
│  │    Framework    │  │   + Redis       │  │     System      │          │
│  │                 │  │                 │  │                 │          │
│  │ - Test cases    │  │ - Audit data    │  │ - Track issues  │          │
│  │ - Automation    │  │ - Evidence      │  │ - Remediation   │          │
│  │ - Reporting     │  │ - Results       │  │ - Closure       │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
│                                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │   Attestation   │  │ Audit Project   │  │   Compliance    │          │
│  │    Workflow     │  │   Management    │  │   Dashboard     │          │
│  │                 │  │                 │  │                 │          │
│  │ - Signatures    │  │ - Timeline      │  │ - Real-time     │          │
│  │ - Approvals     │  │ - Tasks         │  │ - Trends        │          │
│  │ - Versioning    │  │ - Status        │  │ - Alerts        │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Module Structure

```
greenlang/infrastructure/soc2_preparation/
├── __init__.py
├── config.py                    # Configuration management
├── models.py                    # Data models
│
├── self_assessment/             # Self-Assessment Engine
│   ├── __init__.py
│   ├── criteria.py              # 48 TSC criteria definitions
│   ├── assessor.py              # Assessment execution
│   ├── scorer.py                # Scoring algorithm
│   └── gap_analyzer.py          # Gap detection
│
├── evidence/                    # Evidence Management
│   ├── __init__.py
│   ├── collector.py             # Auto-collection from systems
│   ├── packager.py              # Evidence packaging
│   ├── versioner.py             # Version control
│   └── validator.py             # Evidence validation
│
├── control_testing/             # Internal Control Testing
│   ├── __init__.py
│   ├── test_framework.py        # Test execution framework
│   ├── test_cases.py            # Pre-built test cases
│   ├── automation.py            # Automated testing
│   └── reporter.py              # Test reporting
│
├── auditor_portal/              # Auditor Access Management
│   ├── __init__.py
│   ├── access_manager.py        # Access provisioning
│   ├── request_handler.py       # Evidence request handling
│   └── activity_logger.py       # Portal activity logging
│
├── findings/                    # Findings Management
│   ├── __init__.py
│   ├── tracker.py               # Finding tracker
│   ├── remediation.py           # Remediation workflow
│   └── closure.py               # Finding closure
│
├── attestation/                 # Management Attestation
│   ├── __init__.py
│   ├── workflow.py              # Attestation workflow
│   ├── templates.py             # Document templates
│   └── signer.py                # Digital signature handling
│
├── project/                     # Audit Project Management
│   ├── __init__.py
│   ├── timeline.py              # Audit timeline
│   ├── tasks.py                 # Task management
│   └── status.py                # Status tracking
│
├── dashboard/                   # Compliance Dashboard
│   ├── __init__.py
│   ├── metrics.py               # Compliance metrics
│   ├── trends.py                # Trend analysis
│   └── alerts.py                # Compliance alerts
│
├── api/                         # REST API
│   ├── __init__.py
│   ├── assessment_routes.py     # Self-assessment endpoints
│   ├── evidence_routes.py       # Evidence endpoints
│   ├── testing_routes.py        # Control testing endpoints
│   ├── portal_routes.py         # Auditor portal endpoints
│   ├── findings_routes.py       # Findings endpoints
│   ├── attestation_routes.py    # Attestation endpoints
│   ├── project_routes.py        # Project management endpoints
│   └── dashboard_routes.py      # Dashboard endpoints
│
└── metrics.py                   # Prometheus metrics
```

---

## 4. Detailed Specifications

### 4.1 Self-Assessment Engine

#### 4.1.1 Trust Services Criteria

```python
# All 48 SOC 2 criteria organized by category
TSC_CRITERIA = {
    "CC1": {  # Control Environment
        "CC1.1": "COSO Principle 1 - Demonstrates commitment to integrity and ethical values",
        "CC1.2": "COSO Principle 2 - Board exercises oversight responsibility",
        "CC1.3": "COSO Principle 3 - Management establishes structures and reporting lines",
        "CC1.4": "COSO Principle 4 - Commitment to competence",
        "CC1.5": "COSO Principle 5 - Enforces accountability",
    },
    "CC2": {  # Communication and Information
        "CC2.1": "COSO Principle 13 - Uses relevant information",
        "CC2.2": "COSO Principle 14 - Communicates internally",
        "CC2.3": "COSO Principle 15 - Communicates externally",
    },
    "CC3": {  # Risk Assessment
        "CC3.1": "COSO Principle 6 - Specifies suitable objectives",
        "CC3.2": "COSO Principle 7 - Identifies and analyzes risk",
        "CC3.3": "COSO Principle 8 - Assesses fraud risk",
        "CC3.4": "COSO Principle 9 - Identifies and analyzes significant change",
    },
    "CC4": {  # Monitoring Activities
        "CC4.1": "COSO Principle 16 - Selects and develops ongoing evaluations",
        "CC4.2": "COSO Principle 17 - Evaluates and communicates deficiencies",
    },
    "CC5": {  # Control Activities
        "CC5.1": "COSO Principle 10 - Selects and develops control activities",
        "CC5.2": "COSO Principle 11 - Selects and develops technology controls",
        "CC5.3": "COSO Principle 12 - Deploys through policies and procedures",
    },
    "CC6": {  # Logical and Physical Access Controls
        "CC6.1": "Implements logical access security",
        "CC6.2": "Prior to issuing system credentials and access",
        "CC6.3": "Removes access to protected information assets",
        "CC6.4": "Restricts physical access",
        "CC6.5": "Disposes of protected assets and media",
        "CC6.6": "Implements controls to prevent unauthorized software",
        "CC6.7": "Restricts transmission and data movement",
        "CC6.8": "Implements controls to protect against malware",
    },
    "CC7": {  # System Operations
        "CC7.1": "Uses detection and monitoring procedures",
        "CC7.2": "Monitors system components",
        "CC7.3": "Evaluates security events",
        "CC7.4": "Responds to security incidents",
        "CC7.5": "Identifies and recovers from incidents",
    },
    "CC8": {  # Change Management
        "CC8.1": "Manages changes throughout lifecycle",
    },
    "CC9": {  # Risk Mitigation
        "CC9.1": "Identifies and assesses vendor risk",
        "CC9.2": "Assesses and manages risks related to subservice organizations",
    },
    "A1": {  # Availability
        "A1.1": "Maintains current processing capacity",
        "A1.2": "Implements environmental protections",
        "A1.3": "Supports complete and timely recovery",
    },
    "C1": {  # Confidentiality
        "C1.1": "Identifies and maintains confidential information",
        "C1.2": "Disposes of confidential information",
    },
    "PI1": {  # Processing Integrity
        "PI1.1": "Obtains or generates accurate data",
        "PI1.2": "Implements policies for complete and timely processing",
    },
    "P1": {"P1.0": "Privacy notice"},
    "P2": {"P2.0": "Choice and consent"},
    "P3": {"P3.0": "Collection"},
    "P4": {"P4.0": "Use, retention, and disposal"},
    "P5": {"P5.0": "Access"},
    "P6": {"P6.0": "Disclosure and notification"},
    "P7": {"P7.0": "Quality"},
    "P8": {"P8.0": "Monitoring and enforcement"},
}
```

#### 4.1.2 Assessment Scoring

```python
class AssessmentScore:
    """Scoring model for each criterion"""

    # Score levels
    NOT_IMPLEMENTED = 0    # Control does not exist
    PARTIALLY_IMPLEMENTED = 1  # Control exists but gaps
    IMPLEMENTED = 2        # Control implemented but not tested
    TESTED = 3             # Control tested but not documented
    FULLY_COMPLIANT = 4    # Control documented, implemented, tested

    # Evidence requirements
    EVIDENCE_TYPES = [
        "policy",           # Written policy
        "procedure",        # Documented procedure
        "screenshot",       # Configuration screenshot
        "export",           # System export/report
        "log",              # Audit log sample
        "attestation",      # Management attestation
        "test_result",      # Control test result
    ]
```

#### 4.1.3 Gap Analysis

```python
class GapAnalyzer:
    """Identify and prioritize compliance gaps"""

    def analyze_gaps(self, assessment: Assessment) -> List[Gap]:
        """
        Returns prioritized list of gaps with:
        - Criterion ID
        - Current score vs required score
        - Missing evidence
        - Remediation recommendation
        - Estimated effort (hours)
        - Risk level if not remediated
        """
```

### 4.2 Evidence Management System

#### 4.2.1 Evidence Sources

| Source | Evidence Type | Collection Method | Frequency |
|--------|---------------|-------------------|-----------|
| AWS CloudTrail | Access logs | API | Continuous |
| GitHub | Code changes, PRs | API | Continuous |
| PostgreSQL | Audit tables | Direct query | On-demand |
| Loki | Application logs | LogQL | On-demand |
| Okta/Auth Service | Authentication logs | API | Continuous |
| JIRA/Linear | Change tickets | API | On-demand |
| GRC Platform | Policy acknowledgments | API | On-demand |
| Trivy/Snyk | Vulnerability scans | File export | Weekly |
| Terraform | Infrastructure config | State file | On-change |
| Kubernetes | Cluster config | kubectl export | On-demand |

#### 4.2.2 Evidence Packaging

```python
class EvidencePackage:
    """SOC 2 evidence package structure"""

    def create_package(self, criteria: List[str], period: DateRange) -> Package:
        """
        Creates audit-ready evidence package:

        evidence_package/
        ├── manifest.json           # Package metadata
        ├── CC1/                    # Control Environment
        │   ├── CC1.1/
        │   │   ├── evidence.json   # Evidence metadata
        │   │   ├── policies/
        │   │   ├── screenshots/
        │   │   └── attestations/
        │   └── ...
        ├── CC6/                    # Access Controls
        │   ├── CC6.1/
        │   │   ├── access_logs/
        │   │   ├── user_lists/
        │   │   └── config_exports/
        │   └── ...
        ├── populations/            # Sample populations
        │   ├── access_requests.csv
        │   ├── change_tickets.csv
        │   └── incidents.csv
        └── hashes.sha256          # Integrity verification
        """
```

#### 4.2.3 Population Sampling

```python
class PopulationSampler:
    """Generate statistically valid samples for auditor testing"""

    SAMPLE_SIZES = {
        # Based on AICPA guidance
        "weekly_control": {25: 1, 52: 2, 100: 15, 250: 25},
        "monthly_control": {12: 2, 24: 3},
        "quarterly_control": {4: 2},
        "annual_control": {1: 1},
        "on_demand": {  # Based on population size
            25: 1, 50: 3, 100: 8, 250: 15, 500: 25, 1000: 45
        }
    }

    def generate_sample(
        self,
        population: List[Any],
        control_frequency: str
    ) -> List[Any]:
        """Return statistically valid random sample"""
```

### 4.3 Control Testing Framework

#### 4.3.1 Test Case Structure

```python
class ControlTestCase:
    """Individual control test case"""

    criterion_id: str           # e.g., "CC6.1"
    test_id: str                # e.g., "CC6.1-T01"
    test_type: TestType         # DESIGN or OPERATING
    description: str
    procedure: str              # Step-by-step test procedure
    expected_result: str
    evidence_required: List[str]
    automation_available: bool

class TestType(Enum):
    DESIGN = "design"           # Does control exist as designed?
    OPERATING = "operating"     # Did control operate effectively?
```

#### 4.3.2 Automated Test Examples

```python
# Example automated tests for CC6 (Access Controls)

class CC6Tests:
    """Automated control tests for CC6 criteria"""

    async def test_cc6_1_mfa_enforcement(self) -> TestResult:
        """CC6.1-T01: Verify MFA is enforced for all users"""
        users = await self.auth_service.list_users()
        mfa_enabled = [u for u in users if u.mfa_enabled]

        return TestResult(
            passed=len(mfa_enabled) == len(users),
            evidence=f"{len(mfa_enabled)}/{len(users)} users have MFA",
            sample=users[:10]  # Sample for auditor
        )

    async def test_cc6_2_access_provisioning(self) -> TestResult:
        """CC6.1-T02: Verify access requests are approved before provisioning"""
        requests = await self.get_access_requests(last_90_days)
        approved_before_provisioned = [
            r for r in requests
            if r.approved_at < r.provisioned_at
        ]

        return TestResult(
            passed=len(approved_before_provisioned) == len(requests),
            evidence=f"{len(approved_before_provisioned)}/{len(requests)} properly sequenced",
            exceptions=[r for r in requests if r not in approved_before_provisioned]
        )

    async def test_cc6_3_termination_access_removal(self) -> TestResult:
        """CC6.3-T01: Verify terminated users have access removed within 24 hours"""
        terminations = await self.hr_service.get_terminations(last_90_days)
        timely_removals = [
            t for t in terminations
            if (t.access_removed_at - t.termination_date).hours <= 24
        ]

        return TestResult(
            passed=len(timely_removals) == len(terminations),
            evidence=f"{len(timely_removals)}/{len(terminations)} within SLA",
            exceptions=[t for t in terminations if t not in timely_removals]
        )
```

#### 4.3.3 Test Schedule

| Control Type | Test Frequency | Automation | Sample Size |
|--------------|----------------|------------|-------------|
| Preventive | Quarterly | Full | Per control |
| Detective | Monthly | Full | 25/month |
| Corrective | On occurrence | Partial | All |
| Manual | Quarterly | N/A | 15-25 |

### 4.4 Auditor Portal

#### 4.4.1 Portal Features

```python
class AuditorPortal:
    """Secure auditor access to evidence and requests"""

    features = {
        "authentication": {
            "sso": True,              # SAML/OIDC integration
            "mfa": "required",        # Hardware key preferred
            "session_timeout": 30,    # Minutes
        },
        "authorization": {
            "role": "auditor",        # Read-only access
            "scope": ["evidence", "reports", "requests"],
            "audit_logging": True,
        },
        "file_sharing": {
            "upload": False,          # Auditor cannot upload
            "download": True,         # Can download evidence
            "max_download_size": "500MB",
        },
        "request_management": {
            "create_request": True,   # Can submit evidence requests
            "track_request": True,    # Can view request status
            "escalate": True,         # Can flag overdue requests
        }
    }
```

#### 4.4.2 Evidence Request Workflow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Auditor   │     │  Compliance │     │   Control   │     │   Auditor   │
│   Portal    │     │    Team     │     │   Owner     │     │   Portal    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │                   │
       │ Submit Request    │                   │                   │
       │──────────────────>│                   │                   │
       │                   │                   │                   │
       │                   │ Triage & Assign   │                   │
       │                   │──────────────────>│                   │
       │                   │                   │                   │
       │                   │                   │ Gather Evidence   │
       │                   │                   │───────────────────│
       │                   │                   │                   │
       │                   │<──────────────────│                   │
       │                   │  Upload Evidence  │                   │
       │                   │                   │                   │
       │                   │ Review & Approve  │                   │
       │                   │───────────────────│                   │
       │                   │                   │                   │
       │<───────────────────────────────────────────────────────────│
       │           Evidence Available                               │
       │                   │                   │                   │
```

#### 4.4.3 Request SLAs

| Priority | Response Time | Resolution Time | Escalation |
|----------|---------------|-----------------|------------|
| Critical | 4 hours | 1 business day | CISO |
| High | 1 business day | 2 business days | Compliance Manager |
| Normal | 2 business days | 5 business days | Team Lead |
| Low | 3 business days | 10 business days | - |

### 4.5 Findings Management

#### 4.5.1 Finding Classification

```python
class FindingClassification(Enum):
    """SOC 2 finding severity levels"""

    EXCEPTION = "exception"           # One-time deviation, control otherwise effective
    CONTROL_DEFICIENCY = "deficiency" # Control design gap
    SIGNIFICANT_DEFICIENCY = "significant"  # Multiple deficiencies or pervasive issue
    MATERIAL_WEAKNESS = "material"    # Control failure affecting report opinion
```

#### 4.5.2 Remediation Workflow

```python
class RemediationWorkflow:
    """Finding remediation process"""

    states = [
        "identified",       # Finding identified by auditor
        "acknowledged",     # Management acknowledges finding
        "planned",          # Remediation plan created
        "in_progress",      # Remediation underway
        "implemented",      # Fix implemented
        "tested",           # Fix tested
        "closed",           # Auditor confirms closure
    ]

    sla_by_severity = {
        "material": 30,     # Days to remediation plan
        "significant": 60,
        "deficiency": 90,
        "exception": 120,
    }
```

### 4.6 Management Attestation

#### 4.6.1 Attestation Documents

| Document | Signers | Frequency | Purpose |
|----------|---------|-----------|---------|
| SOC 2 Readiness Attestation | CEO, CISO | Pre-audit | Confirm readiness |
| Management Assertion Letter | CEO, CFO | With report | Affirm control operation |
| Control Owner Attestations | Control owners | Quarterly | Confirm control operation |
| Subservice Organization List | CISO | Annual | Identify carve-out/inclusive |
| Complementary User Entity Controls | CISO | Annual | Customer responsibilities |

#### 4.6.2 Digital Signature Integration

```python
class AttestationSigner:
    """Digital signature for attestation documents"""

    async def sign_document(
        self,
        document: Document,
        signers: List[User],
        signature_method: str = "docusign"  # or "adobe_sign"
    ) -> SignedDocument:
        """
        Returns document with:
        - Digital signatures from all signers
        - Timestamp
        - Certificate chain
        - Audit trail
        """
```

### 4.7 Compliance Dashboard

#### 4.7.1 Dashboard Panels

| Panel | Metrics | Refresh |
|-------|---------|---------|
| Readiness Score | Overall % by category | Real-time |
| Evidence Status | Collected vs required | Hourly |
| Control Tests | Passed/failed/pending | On test |
| Open Findings | By severity, age | Real-time |
| Request Status | Open/closed, SLA % | Real-time |
| Attestation Status | Signed/pending | On change |
| Audit Timeline | Milestones, % complete | Daily |
| Risk Heat Map | By control category | Weekly |

#### 4.7.2 Grafana Dashboard

```json
{
  "title": "SOC 2 Type II Compliance Dashboard",
  "panels": [
    {
      "title": "Overall Readiness Score",
      "type": "gauge",
      "targets": [{"expr": "soc2_readiness_score_percent"}]
    },
    {
      "title": "Evidence Collection Progress",
      "type": "stat",
      "targets": [{"expr": "soc2_evidence_collected / soc2_evidence_required * 100"}]
    },
    {
      "title": "Control Test Results",
      "type": "piechart",
      "targets": [{"expr": "soc2_control_tests_total by (result)"}]
    },
    {
      "title": "Open Findings by Severity",
      "type": "bargauge",
      "targets": [{"expr": "soc2_findings_open by (severity)"}]
    },
    {
      "title": "Auditor Request SLA Compliance",
      "type": "stat",
      "targets": [{"expr": "soc2_requests_within_sla / soc2_requests_total * 100"}]
    }
  ]
}
```

---

## 5. API Specifications

### 5.1 Self-Assessment Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/soc2/assessment` | Get current assessment |
| POST | `/api/v1/soc2/assessment/run` | Execute new assessment |
| GET | `/api/v1/soc2/assessment/score` | Get readiness score |
| GET | `/api/v1/soc2/assessment/gaps` | Get identified gaps |
| PUT | `/api/v1/soc2/assessment/criteria/{id}` | Update criterion status |

### 5.2 Evidence Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/soc2/evidence` | List all evidence |
| GET | `/api/v1/soc2/evidence/{criterion}` | Get evidence for criterion |
| POST | `/api/v1/soc2/evidence/collect` | Trigger evidence collection |
| POST | `/api/v1/soc2/evidence/package` | Create evidence package |
| GET | `/api/v1/soc2/evidence/package/{id}` | Download package |

### 5.3 Control Testing Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/soc2/tests` | List all test cases |
| POST | `/api/v1/soc2/tests/run` | Execute test suite |
| GET | `/api/v1/soc2/tests/{id}/result` | Get test result |
| GET | `/api/v1/soc2/tests/report` | Get test report |

### 5.4 Auditor Portal Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/soc2/portal/evidence` | List available evidence (auditor) |
| POST | `/api/v1/soc2/portal/requests` | Submit evidence request |
| GET | `/api/v1/soc2/portal/requests/{id}` | Get request status |
| GET | `/api/v1/soc2/portal/download/{id}` | Download evidence file |

### 5.5 Findings Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/soc2/findings` | List all findings |
| POST | `/api/v1/soc2/findings` | Create finding |
| PUT | `/api/v1/soc2/findings/{id}` | Update finding |
| POST | `/api/v1/soc2/findings/{id}/remediation` | Add remediation plan |
| PUT | `/api/v1/soc2/findings/{id}/close` | Close finding |

### 5.6 Attestation Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/soc2/attestations` | List attestations |
| POST | `/api/v1/soc2/attestations` | Create attestation |
| POST | `/api/v1/soc2/attestations/{id}/sign` | Request signatures |
| GET | `/api/v1/soc2/attestations/{id}/status` | Get signature status |

### 5.7 Dashboard Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/soc2/dashboard/summary` | Get dashboard summary |
| GET | `/api/v1/soc2/dashboard/timeline` | Get audit timeline |
| GET | `/api/v1/soc2/dashboard/metrics` | Get all metrics |

---

## 6. Database Schema

### 6.1 Migration: V016__soc2_preparation.sql

```sql
-- SOC 2 Audit Preparation Schema
CREATE SCHEMA IF NOT EXISTS soc2;

-- Self-Assessment Results
CREATE TABLE soc2.assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    assessed_by UUID REFERENCES security.users(id),
    overall_score DECIMAL(5,2),
    status VARCHAR(20) NOT NULL DEFAULT 'in_progress',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE soc2.assessment_criteria (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID REFERENCES soc2.assessments(id) ON DELETE CASCADE,
    criterion_id VARCHAR(10) NOT NULL,  -- e.g., "CC6.1"
    score INTEGER NOT NULL CHECK (score BETWEEN 0 AND 4),
    notes TEXT,
    evidence_ids UUID[],
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(assessment_id, criterion_id)
);

-- Evidence Management
CREATE TABLE soc2.evidence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    criterion_id VARCHAR(10) NOT NULL,
    evidence_type VARCHAR(50) NOT NULL,
    file_path VARCHAR(500),
    s3_key VARCHAR(500),
    file_hash VARCHAR(64),  -- SHA-256
    collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    collected_by UUID REFERENCES security.users(id),
    valid_from TIMESTAMPTZ,
    valid_to TIMESTAMPTZ,
    metadata JSONB
);

CREATE INDEX idx_evidence_criterion ON soc2.evidence(criterion_id);
CREATE INDEX idx_evidence_collected ON soc2.evidence(collected_at);

-- Evidence Packages
CREATE TABLE soc2.evidence_packages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    package_name VARCHAR(200) NOT NULL,
    criteria VARCHAR(10)[],
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID REFERENCES security.users(id),
    s3_key VARCHAR(500),
    package_hash VARCHAR(64),
    status VARCHAR(20) NOT NULL DEFAULT 'creating'
);

-- Control Tests
CREATE TABLE soc2.control_tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_id VARCHAR(20) NOT NULL,  -- e.g., "CC6.1-T01"
    criterion_id VARCHAR(10) NOT NULL,
    test_type VARCHAR(20) NOT NULL,  -- design/operating
    description TEXT,
    procedure TEXT,
    expected_result TEXT,
    is_automated BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE soc2.control_test_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_id UUID REFERENCES soc2.control_tests(id),
    executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    executed_by UUID REFERENCES security.users(id),
    passed BOOLEAN NOT NULL,
    actual_result TEXT,
    evidence_ids UUID[],
    exceptions JSONB,
    notes TEXT
);

CREATE INDEX idx_test_results_test ON soc2.control_test_results(test_id);
CREATE INDEX idx_test_results_executed ON soc2.control_test_results(executed_at);

-- Auditor Requests
CREATE TABLE soc2.auditor_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_number VARCHAR(20) NOT NULL UNIQUE,
    auditor_id UUID NOT NULL,
    auditor_firm VARCHAR(100),
    criterion_id VARCHAR(10),
    request_type VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    priority VARCHAR(20) NOT NULL DEFAULT 'normal',
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    due_at TIMESTAMPTZ,
    assigned_to UUID REFERENCES security.users(id),
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT
);

CREATE INDEX idx_requests_status ON soc2.auditor_requests(status);
CREATE INDEX idx_requests_due ON soc2.auditor_requests(due_at) WHERE status = 'open';

-- Findings
CREATE TABLE soc2.findings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    finding_number VARCHAR(20) NOT NULL UNIQUE,
    criterion_id VARCHAR(10) NOT NULL,
    classification VARCHAR(30) NOT NULL,  -- exception/deficiency/significant/material
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    identified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    identified_by VARCHAR(100),  -- Auditor name
    status VARCHAR(20) NOT NULL DEFAULT 'identified',
    owner_id UUID REFERENCES security.users(id),
    target_remediation_date DATE,
    actual_remediation_date DATE,
    management_response TEXT,
    root_cause TEXT,
    remediation_plan TEXT,
    verification_notes TEXT,
    closed_at TIMESTAMPTZ
);

CREATE INDEX idx_findings_status ON soc2.findings(status);
CREATE INDEX idx_findings_criterion ON soc2.findings(criterion_id);

-- Attestations
CREATE TABLE soc2.attestations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    attestation_type VARCHAR(50) NOT NULL,
    document_name VARCHAR(200) NOT NULL,
    version VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID REFERENCES security.users(id),
    s3_key VARCHAR(500),
    status VARCHAR(20) NOT NULL DEFAULT 'draft',
    effective_date DATE
);

CREATE TABLE soc2.attestation_signatures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    attestation_id UUID REFERENCES soc2.attestations(id) ON DELETE CASCADE,
    signer_id UUID REFERENCES security.users(id),
    signer_role VARCHAR(100) NOT NULL,
    signed_at TIMESTAMPTZ,
    signature_method VARCHAR(50),
    certificate_hash VARCHAR(64),
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
);

-- Audit Project
CREATE TABLE soc2.audit_projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_name VARCHAR(200) NOT NULL,
    audit_firm VARCHAR(100),
    audit_period_start DATE NOT NULL,
    audit_period_end DATE NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'planning',
    lead_auditor VARCHAR(100),
    internal_liaison_id UUID REFERENCES security.users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    kickoff_date DATE,
    fieldwork_start DATE,
    fieldwork_end DATE,
    report_date DATE
);

CREATE TABLE soc2.audit_milestones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES soc2.audit_projects(id) ON DELETE CASCADE,
    milestone_name VARCHAR(100) NOT NULL,
    description TEXT,
    planned_date DATE,
    actual_date DATE,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    notes TEXT
);

-- Auditor Access Log (audit trail for portal)
CREATE TABLE soc2.auditor_access_log (
    id BIGSERIAL PRIMARY KEY,
    auditor_id UUID NOT NULL,
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    ip_address INET,
    user_agent TEXT,
    accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Convert to hypertable for time-series queries
SELECT create_hypertable('soc2.auditor_access_log', 'accessed_at');

-- Permissions
INSERT INTO security.permissions (name, resource, action, description) VALUES
    ('soc2:assessment:read', 'soc2_assessment', 'read', 'View SOC 2 assessments'),
    ('soc2:assessment:write', 'soc2_assessment', 'write', 'Run SOC 2 assessments'),
    ('soc2:evidence:read', 'soc2_evidence', 'read', 'View SOC 2 evidence'),
    ('soc2:evidence:write', 'soc2_evidence', 'write', 'Collect SOC 2 evidence'),
    ('soc2:evidence:package', 'soc2_evidence', 'package', 'Create evidence packages'),
    ('soc2:tests:read', 'soc2_tests', 'read', 'View control tests'),
    ('soc2:tests:execute', 'soc2_tests', 'execute', 'Execute control tests'),
    ('soc2:portal:access', 'soc2_portal', 'access', 'Access auditor portal'),
    ('soc2:findings:read', 'soc2_findings', 'read', 'View findings'),
    ('soc2:findings:manage', 'soc2_findings', 'manage', 'Manage findings'),
    ('soc2:attestations:read', 'soc2_attestations', 'read', 'View attestations'),
    ('soc2:attestations:sign', 'soc2_attestations', 'sign', 'Sign attestations'),
    ('soc2:project:read', 'soc2_project', 'read', 'View audit project'),
    ('soc2:project:manage', 'soc2_project', 'manage', 'Manage audit project'),
    ('soc2:dashboard:view', 'soc2_dashboard', 'view', 'View compliance dashboard');

-- Role mappings
INSERT INTO security.role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM security.roles r, security.permissions p
WHERE r.name = 'compliance_officer'
  AND p.name LIKE 'soc2:%';

INSERT INTO security.role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM security.roles r, security.permissions p
WHERE r.name = 'auditor'
  AND p.name IN ('soc2:portal:access', 'soc2:evidence:read', 'soc2:findings:read');
```

---

## 7. Monitoring & Alerting

### 7.1 Prometheus Metrics

```python
# SOC 2 Preparation Metrics
SOC2_METRICS = {
    "gl_soc2_readiness_score": Gauge(
        "gl_soc2_readiness_score",
        "Overall SOC 2 readiness score (0-100)",
        ["category"]
    ),
    "gl_soc2_evidence_collected": Counter(
        "gl_soc2_evidence_collected_total",
        "Total evidence items collected",
        ["criterion", "type"]
    ),
    "gl_soc2_tests_executed": Counter(
        "gl_soc2_tests_executed_total",
        "Control tests executed",
        ["criterion", "result"]
    ),
    "gl_soc2_findings_open": Gauge(
        "gl_soc2_findings_open",
        "Open findings by severity",
        ["severity"]
    ),
    "gl_soc2_requests_pending": Gauge(
        "gl_soc2_requests_pending",
        "Pending auditor requests",
        ["priority"]
    ),
    "gl_soc2_request_resolution_seconds": Histogram(
        "gl_soc2_request_resolution_seconds",
        "Time to resolve auditor requests",
        ["priority"]
    ),
    "gl_soc2_attestations_pending": Gauge(
        "gl_soc2_attestations_pending",
        "Attestations awaiting signature"
    ),
}
```

### 7.2 Alert Rules

```yaml
groups:
  - name: soc2_preparation_alerts
    rules:
      - alert: SOC2ReadinessScoreLow
        expr: gl_soc2_readiness_score < 90
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "SOC 2 readiness score below 90%"

      - alert: SOC2AuditorRequestOverdue
        expr: gl_soc2_requests_pending{priority="critical"} > 0
        for: 4h
        labels:
          severity: critical
        annotations:
          summary: "Critical auditor request overdue"

      - alert: SOC2FindingUnaddressed
        expr: gl_soc2_findings_open{severity="material"} > 0
        for: 24h
        labels:
          severity: critical
        annotations:
          summary: "Material finding unaddressed for 24+ hours"

      - alert: SOC2AttestationPendingSignature
        expr: gl_soc2_attestations_pending > 0
        for: 72h
        labels:
          severity: warning
        annotations:
          summary: "Attestation awaiting signature for 72+ hours"

      - alert: SOC2ControlTestFailed
        expr: increase(gl_soc2_tests_executed_total{result="failed"}[1h]) > 0
        labels:
          severity: warning
        annotations:
          summary: "Control test failed"
```

---

## 8. Implementation Phases

### Phase 1: Core Infrastructure (P0)
- [ ] Database migration V016
- [ ] Configuration management
- [ ] Base models and types
- [ ] Package initialization

### Phase 2: Self-Assessment Engine (P0)
- [ ] 48 TSC criteria definitions
- [ ] Assessment executor
- [ ] Scoring algorithm
- [ ] Gap analyzer

### Phase 3: Evidence Management (P0)
- [ ] Evidence collector (multi-source)
- [ ] Evidence packager
- [ ] Version control
- [ ] Validation and integrity

### Phase 4: Control Testing (P1)
- [ ] Test framework
- [ ] Automated test cases (20+)
- [ ] Test execution engine
- [ ] Test reporting

### Phase 5: Auditor Portal (P1)
- [ ] Access management
- [ ] Request handling
- [ ] Activity logging
- [ ] Secure file sharing

### Phase 6: Findings Management (P1)
- [ ] Finding tracker
- [ ] Remediation workflow
- [ ] Status tracking
- [ ] Closure verification

### Phase 7: Attestation Workflow (P2)
- [ ] Document templates
- [ ] Signature workflow
- [ ] Version management
- [ ] Digital signing integration

### Phase 8: Project Management (P2)
- [ ] Audit timeline
- [ ] Milestone tracking
- [ ] Task management
- [ ] Status reporting

### Phase 9: Dashboard & Monitoring (P2)
- [ ] Grafana dashboard
- [ ] Prometheus metrics
- [ ] Alert rules
- [ ] Real-time updates

### Phase 10: API & Integration (P2)
- [ ] REST API endpoints
- [ ] Route protection
- [ ] Auth integration
- [ ] Testing suite

---

## 9. Testing Requirements

### 9.1 Test Coverage

| Component | Unit Tests | Integration Tests | Target Coverage |
|-----------|------------|-------------------|-----------------|
| Self-Assessment | 40+ | 10+ | 85% |
| Evidence Mgmt | 35+ | 15+ | 85% |
| Control Testing | 30+ | 20+ | 85% |
| Auditor Portal | 25+ | 10+ | 80% |
| Findings | 20+ | 10+ | 85% |
| Attestations | 15+ | 5+ | 80% |
| Dashboard | 15+ | 5+ | 80% |
| **Total** | **180+** | **75+** | **85%** |

---

## 10. Dependencies

### 10.1 Internal Dependencies

| Dependency | Purpose | Required |
|------------|---------|----------|
| SEC-001 | Authentication for portal | Yes |
| SEC-002 | RBAC for permissions | Yes |
| SEC-005 | Audit logging | Yes |
| SEC-008 | Policy framework | Yes |
| INFRA-002 | PostgreSQL storage | Yes |
| INFRA-003 | Redis caching | Yes |
| INFRA-004 | S3 evidence storage | Yes |
| INFRA-009 | Loki log queries | Yes |

### 10.2 External Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| boto3 | 1.34+ | AWS S3, CloudTrail |
| httpx | 0.27+ | API integrations |
| cryptography | 42+ | Hash verification |
| python-docx | 1.1+ | Document generation |
| PyPDF2 | 3.0+ | PDF generation |

---

## 11. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Self-assessment automation | >90% | Auto-scored criteria |
| Evidence collection automation | >80% | Auto-collected items |
| Control test automation | >70% | Automated tests |
| Auditor request SLA | >95% | Within SLA |
| Finding remediation rate | 100% | Before report |
| Attestation completion | 100% | All signed |
| Dashboard availability | 99.9% | Uptime |

---

## 12. Appendix

### 12.1 SOC 2 Type II Timeline Template

```
Month -2: Auditor Selection
  - RFP to 3-5 firms
  - Evaluate proposals
  - Contract negotiation
  - Engagement letter

Month -1: Pre-Audit Preparation
  - Internal readiness assessment
  - Evidence organization
  - Control testing completion
  - Management attestations

Month 0: Audit Kickoff
  - Opening meeting
  - Scope confirmation
  - Request list walkthrough
  - Timeline agreement

Months 1-11: Audit Period
  - Interim testing (Month 3, 6, 9)
  - Evidence request handling
  - Finding remediation
  - Status updates

Month 12: Audit Completion
  - Final testing
  - Management letter draft
  - Response preparation
  - Exit meeting

Month 13: Report Issuance
  - Final report review
  - Report publication
  - Customer distribution
  - Year 2 planning
```

### 12.2 Evidence Request Template

```markdown
# Evidence Request

**Request ID:** ER-2026-001
**Auditor:** [Firm Name]
**Date:** 2026-02-06
**Priority:** High

## Request Details

**Criterion:** CC6.1 - Logical Access Security
**Evidence Type:** Population and Sample

## Description

Please provide:
1. Complete list of system users as of [date]
2. Access provisioning requests for audit period
3. Sample of 25 access approvals

## Due Date

2026-02-10 (4 business days)

## Delivery Instructions

Upload to auditor portal: https://audit.greenlang.io
```

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-06 | Security Team | Initial version |
