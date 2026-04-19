# PRD-SEC-010: Security Operations Automation Platform

**Version:** 1.0
**Status:** APPROVED
**Author:** Security & Compliance Team
**Created:** 2026-02-06
**Last Updated:** 2026-02-06

---

## 1. Executive Summary

### 1.1 Purpose

SEC-010 completes the GreenLang Security & Compliance track by implementing operational security automation. While SEC-001 through SEC-009 established technical security controls (authentication, authorization, encryption, logging, scanning, policies, SOC 2), SEC-010 adds the operational layer:
- **Incident Response Automation** - Detect, escalate, and remediate security incidents
- **Threat Modeling** - Systematic security design review
- **DDoS/WAF Protection** - Layer 7 defense and attack mitigation
- **Vulnerability Disclosure** - Responsible disclosure and bug bounty management
- **Multi-Compliance Automation** - ISO 27001, GDPR, PCI-DSS continuous compliance
- **Security Training** - Awareness platform with phishing simulation

### 1.2 Business Value

| Benefit | Impact |
|---------|--------|
| MTTD reduction | <5 min incident detection (from ~30 min) |
| MTTR reduction | <15 min automated response |
| Compliance readiness | ISO 27001 + GDPR + PCI-DSS automation |
| Enterprise sales | Required for regulated customers |
| Trust building | Public VDP shows security maturity |
| Risk reduction | Proactive threat identification |

### 1.3 Scope

| Component | Priority | Files | Lines (Est.) |
|-----------|----------|-------|--------------|
| Incident Response Automation | P0 | 12 | 4,500 |
| Threat Modeling System | P0 | 10 | 3,500 |
| DDoS/WAF Management | P1 | 12 | 4,000 |
| Vulnerability Disclosure Program | P1 | 8 | 3,000 |
| Multi-Compliance Automation | P1 | 14 | 5,000 |
| Security Training Platform | P2 | 10 | 3,500 |
| **Total** | - | **~66** | **~23,500** |

---

## 2. Component 1: Incident Response Automation

### 2.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Incident Response Automation                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │  Alert Sources  │  │ Incident Engine │  │   Notification  │          │
│  │                 │  │                 │  │     System      │          │
│  │ - Prometheus    │  │ - Detector      │  │                 │          │
│  │ - Loki          │──▶│ - Correlator    │──▶│ - PagerDuty    │          │
│  │ - GuardDuty     │  │ - Classifier    │  │ - Slack         │          │
│  │ - CloudTrail    │  │ - Escalator     │  │ - Email/SMS     │          │
│  └─────────────────┘  └────────┬────────┘  └─────────────────┘          │
│                                │                                         │
│                       ┌────────▼────────┐  ┌─────────────────┐          │
│                       │   Playbook      │  │    Tracking     │          │
│                       │   Executor      │  │     System      │          │
│                       │                 │  │                 │          │
│                       │ - Auto-remediate│──▶│ - Jira/Linear  │          │
│                       │ - Containment   │  │ - Timeline      │          │
│                       │ - Recovery      │  │ - Metrics       │          │
│                       └─────────────────┘  └─────────────────┘          │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Module Structure

```
greenlang/infrastructure/incident_response/
├── __init__.py
├── config.py
├── models.py
├── detector.py              # Alert aggregation and correlation
├── correlator.py            # Multi-source event correlation
├── classifier.py            # Incident severity classification
├── escalator.py             # Escalation workflow engine
├── notifier.py              # PagerDuty, Slack, Email integration
├── playbook_executor.py     # Automated remediation playbooks
├── tracker.py               # Incident lifecycle tracking
├── metrics.py               # Prometheus metrics
└── api/
    ├── __init__.py
    └── incident_routes.py   # REST API endpoints
```

### 2.3 Key Classes

```python
class IncidentDetector:
    """Aggregates alerts from multiple sources"""

    async def poll_prometheus(self) -> List[Alert]
    async def poll_loki(self) -> List[Alert]
    async def poll_guardduty(self) -> List[Alert]
    async def poll_cloudtrail_anomalies(self) -> List[Alert]
    async def detect_incidents(self) -> List[Incident]

class IncidentCorrelator:
    """Correlates related alerts into incidents"""

    async def correlate(self, alerts: List[Alert]) -> List[Incident]
    def calculate_similarity(self, a1: Alert, a2: Alert) -> float
    def merge_incidents(self, incidents: List[Incident]) -> Incident

class IncidentClassifier:
    """Classifies incident severity and type"""

    SEVERITY_LEVELS = {
        "P0": {"response_time": 15, "escalation": "immediate"},  # Critical
        "P1": {"response_time": 60, "escalation": "1h"},         # High
        "P2": {"response_time": 240, "escalation": "4h"},        # Medium
        "P3": {"response_time": 1440, "escalation": "24h"},      # Low
    }

    def classify(self, incident: Incident) -> Severity
    def calculate_business_impact(self, incident: Incident) -> Impact

class PlaybookExecutor:
    """Executes automated remediation playbooks"""

    PLAYBOOKS = {
        "credential_compromise": CredentialCompromisePlaybook,
        "ddos_attack": DDoSMitigationPlaybook,
        "data_breach": DataBreachPlaybook,
        "malware_detected": MalwareContainmentPlaybook,
        "unauthorized_access": AccessRevocationPlaybook,
        # ... 20+ playbooks
    }

    async def execute(self, incident: Incident, playbook_id: str) -> ExecutionResult
    async def rollback(self, execution_id: str) -> None
```

### 2.4 Alert Sources Integration

| Source | Method | Frequency | Alert Types |
|--------|--------|-----------|-------------|
| Prometheus | HTTP Pull | 30s | Performance, availability |
| Loki | LogQL Push | Real-time | Error patterns, security events |
| AWS GuardDuty | EventBridge | Real-time | Threats, compromised instances |
| CloudTrail | S3 + Lambda | Near real-time | API anomalies, unauthorized access |
| Security Scanning | Webhook | On completion | New vulnerabilities |

### 2.5 Playbook Examples

```python
class CredentialCompromisePlaybook(BasePlaybook):
    """Automated response to credential compromise"""

    steps = [
        "identify_affected_accounts",
        "revoke_all_sessions",
        "rotate_credentials",
        "enable_mfa_enforcement",
        "notify_affected_users",
        "create_incident_ticket",
        "collect_forensic_evidence",
        "generate_post_mortem",
    ]

    async def execute(self, incident: Incident) -> PlaybookResult:
        # Automated credential rotation and session revocation
        pass

class DDoSMitigationPlaybook(BasePlaybook):
    """Automated DDoS attack response"""

    steps = [
        "identify_attack_vector",
        "enable_shield_advanced",
        "update_waf_rules",
        "scale_infrastructure",
        "enable_geo_blocking",
        "notify_noc",
        "monitor_mitigation",
    ]
```

---

## 3. Component 2: Threat Modeling System

### 3.1 Module Structure

```
greenlang/infrastructure/threat_modeling/
├── __init__.py
├── config.py
├── models.py
├── stride_engine.py         # STRIDE threat identification
├── attack_surface.py        # Attack surface mapping
├── dfd_validator.py         # Data flow diagram validation
├── risk_scorer.py           # Risk scoring engine
├── control_mapper.py        # Map threats to controls
├── metrics.py
└── api/
    ├── __init__.py
    └── threat_routes.py
```

### 3.2 STRIDE Engine

```python
class STRIDEEngine:
    """STRIDE threat identification engine"""

    THREAT_CATEGORIES = {
        "S": "Spoofing",           # Identity attacks
        "T": "Tampering",          # Data integrity
        "R": "Repudiation",        # Non-repudiation
        "I": "Information Disclosure",  # Confidentiality
        "D": "Denial of Service",  # Availability
        "E": "Elevation of Privilege",  # Authorization
    }

    def analyze_component(self, component: Component) -> List[Threat]
    def analyze_data_flow(self, flow: DataFlow) -> List[Threat]
    def generate_threat_model(self, system: System) -> ThreatModel

class RiskScorer:
    """Risk scoring based on CVSS + business impact"""

    def calculate_risk_score(
        self,
        threat: Threat,
        likelihood: float,
        impact: float,
        business_context: dict
    ) -> RiskScore:
        # Composite score = (CVSS * likelihood * impact * business_weight)
        pass

    def prioritize_threats(self, threats: List[Threat]) -> List[RankedThreat]
```

### 3.3 Threat Model Schema

```python
class ThreatModel(BaseModel):
    id: UUID
    service_name: str
    version: str
    created_at: datetime
    created_by: UUID

    # Components
    components: List[Component]
    data_flows: List[DataFlow]
    trust_boundaries: List[TrustBoundary]

    # Analysis
    threats: List[Threat]
    mitigations: List[Mitigation]
    residual_risks: List[Risk]

    # Scoring
    overall_risk_score: float
    category_scores: Dict[str, float]  # S, T, R, I, D, E

    # Audit
    review_status: str
    approved_by: Optional[UUID]
    approved_at: Optional[datetime]
```

---

## 4. Component 3: DDoS Protection & WAF Management

### 4.1 Module Structure

```
greenlang/infrastructure/waf_management/
├── __init__.py
├── config.py
├── models.py
├── rule_builder.py          # WAF rule construction
├── rule_tester.py           # Rule effectiveness testing
├── anomaly_detector.py      # Attack pattern detection
├── shield_manager.py        # AWS Shield Advanced
├── metrics.py
└── api/
    ├── __init__.py
    └── waf_routes.py

deployment/terraform/modules/shield-waf/
├── main.tf                  # Shield Advanced + WAF v2
├── variables.tf
├── outputs.tf
└── waf-rules.tf            # Managed rule groups
```

### 4.2 WAF Rule Management

```python
class WAFRuleBuilder:
    """Build and manage WAF rules"""

    RULE_TYPES = {
        "rate_limit": RateLimitRule,
        "geo_block": GeoBlockRule,
        "ip_reputation": IPReputationRule,
        "sql_injection": SQLInjectionRule,
        "xss": XSSRule,
        "custom_regex": CustomRegexRule,
    }

    def create_rule(self, rule_type: str, config: dict) -> WAFRule
    def deploy_rule(self, rule: WAFRule, web_acl_id: str) -> None
    def test_rule(self, rule: WAFRule, test_requests: List[Request]) -> TestResult

class AnomalyDetector:
    """Detect DDoS and attack patterns"""

    def analyze_traffic(self, metrics: TrafficMetrics) -> List[Anomaly]
    def detect_volumetric_attack(self, rps: int, baseline: int) -> bool
    def detect_slowloris(self, connection_metrics: dict) -> bool
    def detect_application_layer_attack(self, patterns: List[Pattern]) -> bool
    async def auto_mitigate(self, attack: Attack) -> MitigationResult
```

### 4.3 Terraform Module

```hcl
# deployment/terraform/modules/shield-waf/main.tf

resource "aws_shield_protection" "main" {
  name         = "greenlang-shield"
  resource_arn = var.alb_arn
}

resource "aws_wafv2_web_acl" "main" {
  name  = "greenlang-waf"
  scope = "REGIONAL"

  default_action {
    allow {}
  }

  # AWS Managed Rules
  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 1

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesCommonRuleSet"
      }
    }
  }

  # Rate limiting
  rule {
    name     = "RateLimitRule"
    priority = 2

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }
  }
}
```

---

## 5. Component 4: Vulnerability Disclosure Program

### 5.1 Module Structure

```
greenlang/infrastructure/vulnerability_disclosure/
├── __init__.py
├── config.py
├── models.py
├── submission_handler.py    # Vulnerability submission processing
├── triage_workflow.py       # Triage and severity assessment
├── disclosure_tracker.py    # Disclosure timeline management
├── researcher_manager.py    # Researcher profiles and reputation
├── bounty_processor.py      # Bug bounty payment integration
├── metrics.py
└── api/
    ├── __init__.py
    └── vdp_routes.py

docs/security/
├── security.txt             # RFC 9116 security.txt
└── SECURITY.md              # Responsible disclosure policy
```

### 5.2 VDP Workflow

```python
class VulnerabilitySubmissionHandler:
    """Process incoming vulnerability reports"""

    async def submit(self, report: VulnerabilityReport) -> Submission:
        # 1. Validate submission
        # 2. Auto-acknowledge (within 24h requirement)
        # 3. Assign to triage queue
        # 4. Notify security team
        pass

class TriageWorkflow:
    """Triage and assess vulnerabilities"""

    STATES = [
        "submitted",
        "acknowledged",
        "triaging",
        "confirmed",
        "remediation",
        "fixed",
        "disclosed",
        "closed",
    ]

    async def triage(self, submission: Submission) -> TriageResult:
        # CVSS scoring, duplicate detection, severity assignment
        pass

class DisclosureTracker:
    """Manage disclosure timelines"""

    DISCLOSURE_POLICIES = {
        "critical": 7,    # Days to fix before disclosure
        "high": 30,
        "medium": 60,
        "low": 90,
    }

    def calculate_disclosure_date(self, severity: str, reported_at: datetime) -> datetime
    def check_disclosure_readiness(self, vulnerability_id: UUID) -> bool
```

### 5.3 Security.txt

```
# docs/security/security.txt (RFC 9116)

Contact: mailto:security@greenlang.io
Contact: https://greenlang.io/.well-known/security.txt
Encryption: https://greenlang.io/.well-known/pgp-key.txt
Acknowledgments: https://greenlang.io/security/hall-of-fame
Policy: https://greenlang.io/security/disclosure-policy
Hiring: https://greenlang.io/careers/security
Preferred-Languages: en
Canonical: https://greenlang.io/.well-known/security.txt
Expires: 2027-02-06T00:00:00.000Z
```

---

## 6. Component 5: Multi-Compliance Automation

### 6.1 Module Structure

```
greenlang/infrastructure/compliance_automation/
├── __init__.py
├── config.py
├── models.py
├── base_framework.py        # Base compliance framework
├── iso27001/                # ISO 27001:2022
│   ├── __init__.py
│   ├── mapper.py            # Control mapping
│   ├── evidence.py          # Evidence collection
│   └── reporter.py          # Compliance reporting
├── gdpr/                    # GDPR automation
│   ├── __init__.py
│   ├── dsar_processor.py    # Data Subject Access Requests
│   ├── data_discovery.py    # PII discovery
│   ├── retention_enforcer.py# Retention policy enforcement
│   └── consent_manager.py   # Consent tracking
├── pci_dss/                 # PCI-DSS
│   ├── __init__.py
│   ├── card_data_mapper.py  # Cardholder data flow
│   └── encryption_checker.py# Encryption validation
├── ccpa/                    # CCPA/LGPD
│   ├── __init__.py
│   └── consumer_rights.py   # Consumer rights processing
├── metrics.py
└── api/
    ├── __init__.py
    └── compliance_routes.py
```

### 6.2 GDPR DSAR Processor

```python
class DSARProcessor:
    """Process Data Subject Access Requests (GDPR Art. 15-22)"""

    REQUEST_TYPES = {
        "access": "Article 15 - Right of Access",
        "rectification": "Article 16 - Right to Rectification",
        "erasure": "Article 17 - Right to Erasure",
        "portability": "Article 20 - Right to Data Portability",
        "objection": "Article 21 - Right to Object",
        "restriction": "Article 18 - Right to Restriction",
    }

    SLA_DAYS = 30  # GDPR requirement

    async def submit_request(self, request: DSARRequest) -> DSARTicket:
        """Submit and validate DSAR"""
        # Identity verification
        # Request validation
        # Ticket creation
        # 30-day SLA timer start

    async def discover_data(self, user_id: UUID) -> List[DataRecord]:
        """Discover all data for a user across systems"""
        # Query all databases
        # Check S3, logs, backups
        # Compile data inventory

    async def execute_erasure(self, user_id: UUID) -> ErasureResult:
        """Execute right to erasure (right to be forgotten)"""
        # Delete from primary databases
        # Delete from backups (scheduled)
        # Delete from logs (anonymize)
        # Generate deletion certificate

    async def export_data(self, user_id: UUID, format: str = "json") -> bytes:
        """Export data in portable format (Art. 20)"""
        # Machine-readable export
        # Include all personal data
        # Generate checksum
```

### 6.3 ISO 27001 Continuous Compliance

```python
class ISO27001Mapper:
    """Map technical controls to ISO 27001:2022 Annex A"""

    CONTROL_MAPPING = {
        # A.5 Organizational controls
        "A.5.1": {"control": "Policies", "evidence": ["POL-001"]},
        "A.5.7": {"control": "Threat Intelligence", "evidence": ["threat_model"]},

        # A.8 Technological controls
        "A.8.1": {"control": "User endpoint devices", "evidence": ["mdm_report"]},
        "A.8.5": {"control": "Secure authentication", "evidence": ["auth_config"]},
        "A.8.24": {"control": "Cryptography", "evidence": ["encryption_config"]},
        # ... 93 controls total
    }

    async def assess_compliance(self) -> ComplianceReport:
        """Assess current compliance status"""
        results = {}
        for control_id, mapping in self.CONTROL_MAPPING.items():
            evidence = await self.collect_evidence(mapping["evidence"])
            results[control_id] = self.evaluate_control(control_id, evidence)
        return ComplianceReport(results=results)
```

---

## 7. Component 6: Security Training Platform

### 7.1 Module Structure

```
greenlang/infrastructure/security_training/
├── __init__.py
├── config.py
├── models.py
├── content_library.py       # Training content management
├── curriculum_mapper.py     # Role-based training paths
├── assessment_engine.py     # Quizzes and assessments
├── phishing_simulator.py    # Phishing campaign automation
├── completion_tracker.py    # Training completion tracking
├── security_scorer.py       # Employee security score
├── metrics.py
└── api/
    ├── __init__.py
    └── training_routes.py
```

### 7.2 Training System

```python
class CurriculumMapper:
    """Map training to roles"""

    ROLE_CURRICULA = {
        "developer": [
            "secure_coding_fundamentals",
            "owasp_top_10",
            "secure_code_review",
            "dependency_security",
        ],
        "devops": [
            "infrastructure_security",
            "secrets_management",
            "container_security",
            "incident_response",
        ],
        "all_employees": [
            "security_awareness",
            "phishing_recognition",
            "password_hygiene",
            "data_classification",
        ],
    }

    def get_curriculum(self, user: User) -> List[Course]
    def get_required_training(self, user: User) -> List[Course]

class PhishingSimulator:
    """Automated phishing simulation campaigns"""

    TEMPLATE_TYPES = [
        "credential_harvest",
        "malicious_attachment",
        "fake_invoice",
        "urgent_action",
        "ceo_fraud",
    ]

    async def create_campaign(self, campaign: PhishingCampaign) -> Campaign
    async def send_phishing_emails(self, campaign_id: UUID) -> None
    async def track_interactions(self, campaign_id: UUID) -> CampaignMetrics
    async def trigger_training(self, user_id: UUID) -> None  # Auto-train clickers
```

---

## 8. Database Schema

### 8.1 Migration: V017__security_operations.sql

```sql
-- Security Operations Automation Schema
CREATE SCHEMA IF NOT EXISTS security_ops;

-- Incidents
CREATE TABLE security_ops.incidents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_number VARCHAR(20) NOT NULL UNIQUE,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    severity VARCHAR(10) NOT NULL,  -- P0, P1, P2, P3
    status VARCHAR(20) NOT NULL DEFAULT 'detected',
    incident_type VARCHAR(50) NOT NULL,
    source VARCHAR(50) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ,
    assignee_id UUID REFERENCES security.users(id),
    playbook_id VARCHAR(50),
    playbook_execution_id UUID,
    metadata JSONB
);

SELECT create_hypertable('security_ops.incidents', 'detected_at');

-- Alerts
CREATE TABLE security_ops.alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id UUID REFERENCES security_ops.incidents(id),
    alert_source VARCHAR(50) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(10) NOT NULL,
    message TEXT NOT NULL,
    raw_data JSONB,
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('security_ops.alerts', 'received_at');

-- Playbook Executions
CREATE TABLE security_ops.playbook_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id UUID REFERENCES security_ops.incidents(id),
    playbook_id VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'running',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    steps_completed INTEGER DEFAULT 0,
    steps_total INTEGER NOT NULL,
    execution_log JSONB
);

-- Threat Models
CREATE TABLE security_ops.threat_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    service_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID REFERENCES security.users(id),
    status VARCHAR(20) NOT NULL DEFAULT 'draft',
    overall_risk_score DECIMAL(5,2),
    components JSONB,
    data_flows JSONB,
    threats JSONB,
    mitigations JSONB,
    approved_by UUID REFERENCES security.users(id),
    approved_at TIMESTAMPTZ
);

-- WAF Rules
CREATE TABLE security_ops.waf_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_name VARCHAR(100) NOT NULL,
    rule_type VARCHAR(50) NOT NULL,
    priority INTEGER NOT NULL,
    action VARCHAR(20) NOT NULL,
    condition JSONB NOT NULL,
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID REFERENCES security.users(id),
    deployed_at TIMESTAMPTZ,
    metrics JSONB
);

-- Vulnerability Disclosures
CREATE TABLE security_ops.vulnerability_disclosures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id VARCHAR(20) NOT NULL UNIQUE,
    title VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'submitted',
    reporter_email VARCHAR(255) NOT NULL,
    reporter_name VARCHAR(100),
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    triaged_at TIMESTAMPTZ,
    fixed_at TIMESTAMPTZ,
    disclosed_at TIMESTAMPTZ,
    disclosure_deadline TIMESTAMPTZ,
    cvss_score DECIMAL(3,1),
    bounty_amount DECIMAL(10,2),
    bounty_paid_at TIMESTAMPTZ
);

-- GDPR DSARs
CREATE TABLE security_ops.dsar_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_number VARCHAR(20) NOT NULL UNIQUE,
    request_type VARCHAR(30) NOT NULL,
    subject_email VARCHAR(255) NOT NULL,
    subject_name VARCHAR(100),
    status VARCHAR(20) NOT NULL DEFAULT 'submitted',
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    due_date TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    data_discovered JSONB,
    actions_taken JSONB
);

-- Security Training
CREATE TABLE security_ops.training_completions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES security.users(id),
    course_id VARCHAR(50) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    score INTEGER,
    passed BOOLEAN,
    certificate_id VARCHAR(50)
);

-- Phishing Campaigns
CREATE TABLE security_ops.phishing_campaigns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_name VARCHAR(100) NOT NULL,
    template_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'draft',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sent_at TIMESTAMPTZ,
    emails_sent INTEGER DEFAULT 0,
    clicks INTEGER DEFAULT 0,
    reports INTEGER DEFAULT 0
);

-- Permissions
INSERT INTO security.permissions (name, resource, action, description) VALUES
    ('secops:incidents:read', 'incidents', 'read', 'View security incidents'),
    ('secops:incidents:manage', 'incidents', 'manage', 'Manage security incidents'),
    ('secops:playbooks:execute', 'playbooks', 'execute', 'Execute remediation playbooks'),
    ('secops:threats:read', 'threats', 'read', 'View threat models'),
    ('secops:threats:write', 'threats', 'write', 'Create/edit threat models'),
    ('secops:waf:read', 'waf', 'read', 'View WAF rules'),
    ('secops:waf:manage', 'waf', 'manage', 'Manage WAF rules'),
    ('secops:vdp:read', 'vdp', 'read', 'View vulnerability disclosures'),
    ('secops:vdp:manage', 'vdp', 'manage', 'Manage vulnerability disclosures'),
    ('secops:compliance:read', 'compliance', 'read', 'View compliance status'),
    ('secops:compliance:manage', 'compliance', 'manage', 'Manage compliance automation'),
    ('secops:dsar:read', 'dsar', 'read', 'View DSAR requests'),
    ('secops:dsar:process', 'dsar', 'process', 'Process DSAR requests'),
    ('secops:training:read', 'training', 'read', 'View training status'),
    ('secops:training:manage', 'training', 'manage', 'Manage security training'),
    ('secops:phishing:manage', 'phishing', 'manage', 'Manage phishing campaigns');
```

---

## 9. API Specifications

### 9.1 Incident Response Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/secops/incidents` | List incidents |
| GET | `/api/v1/secops/incidents/{id}` | Get incident details |
| POST | `/api/v1/secops/incidents/{id}/acknowledge` | Acknowledge incident |
| POST | `/api/v1/secops/incidents/{id}/execute-playbook` | Execute playbook |
| PUT | `/api/v1/secops/incidents/{id}/resolve` | Resolve incident |
| GET | `/api/v1/secops/incidents/{id}/timeline` | Get incident timeline |

### 9.2 Threat Modeling Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/secops/threats` | List threat models |
| POST | `/api/v1/secops/threats` | Create threat model |
| GET | `/api/v1/secops/threats/{id}` | Get threat model |
| POST | `/api/v1/secops/threats/{id}/analyze` | Run STRIDE analysis |
| PUT | `/api/v1/secops/threats/{id}/approve` | Approve threat model |

### 9.3 WAF Management Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/secops/waf/rules` | List WAF rules |
| POST | `/api/v1/secops/waf/rules` | Create rule |
| POST | `/api/v1/secops/waf/rules/{id}/test` | Test rule |
| POST | `/api/v1/secops/waf/rules/{id}/deploy` | Deploy rule |
| GET | `/api/v1/secops/waf/metrics` | Get WAF metrics |

### 9.4 VDP Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/secops/vdp/submit` | Submit vulnerability (public) |
| GET | `/api/v1/secops/vdp/submissions` | List submissions |
| PUT | `/api/v1/secops/vdp/submissions/{id}/triage` | Triage submission |
| PUT | `/api/v1/secops/vdp/submissions/{id}/close` | Close submission |

### 9.5 Compliance Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/secops/compliance/status` | Get compliance status |
| GET | `/api/v1/secops/compliance/iso27001` | ISO 27001 status |
| GET | `/api/v1/secops/compliance/gdpr` | GDPR status |
| POST | `/api/v1/secops/dsar` | Submit DSAR |
| GET | `/api/v1/secops/dsar/{id}` | Get DSAR status |
| POST | `/api/v1/secops/dsar/{id}/execute` | Execute DSAR |

### 9.6 Training Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/secops/training/courses` | List courses |
| GET | `/api/v1/secops/training/my-progress` | User progress |
| POST | `/api/v1/secops/training/complete` | Mark complete |
| POST | `/api/v1/secops/phishing/campaigns` | Create campaign |
| GET | `/api/v1/secops/phishing/campaigns/{id}/metrics` | Campaign metrics |

---

## 10. Monitoring & Alerting

### 10.1 Prometheus Metrics

```python
SECOPS_METRICS = {
    # Incident Response
    "gl_secops_incidents_total": Counter("Incidents by severity"),
    "gl_secops_incident_mttd_seconds": Histogram("Mean time to detect"),
    "gl_secops_incident_mttr_seconds": Histogram("Mean time to respond"),
    "gl_secops_playbook_executions_total": Counter("Playbook executions"),

    # Threat Modeling
    "gl_secops_threat_models_total": Gauge("Total threat models"),
    "gl_secops_threats_by_category": Gauge("Threats by STRIDE category"),

    # WAF
    "gl_secops_waf_requests_total": Counter("WAF requests"),
    "gl_secops_waf_blocked_total": Counter("WAF blocked requests"),
    "gl_secops_ddos_attacks_total": Counter("DDoS attacks detected"),

    # VDP
    "gl_secops_vdp_submissions_total": Counter("VDP submissions"),
    "gl_secops_vdp_fix_time_days": Histogram("Time to fix vulnerabilities"),

    # Compliance
    "gl_secops_compliance_score": Gauge("Compliance score by framework"),
    "gl_secops_dsar_pending": Gauge("Pending DSARs"),
    "gl_secops_dsar_sla_compliance": Gauge("DSAR SLA compliance %"),

    # Training
    "gl_secops_training_completion_rate": Gauge("Training completion %"),
    "gl_secops_phishing_click_rate": Gauge("Phishing click rate %"),
}
```

### 10.2 Alert Rules

```yaml
groups:
  - name: security_operations_alerts
    rules:
      - alert: P0IncidentDetected
        expr: increase(gl_secops_incidents_total{severity="P0"}[5m]) > 0
        labels:
          severity: critical
        annotations:
          summary: "P0 Security Incident Detected"

      - alert: HighMTTR
        expr: gl_secops_incident_mttr_seconds > 900  # 15 min
        labels:
          severity: warning
        annotations:
          summary: "Incident response time exceeding SLA"

      - alert: DDoSAttackDetected
        expr: increase(gl_secops_ddos_attacks_total[1m]) > 0
        labels:
          severity: critical
        annotations:
          summary: "DDoS attack detected"

      - alert: DSARSLABreach
        expr: gl_secops_dsar_sla_compliance < 0.95
        labels:
          severity: warning
        annotations:
          summary: "DSAR SLA compliance below 95%"
```

---

## 11. Implementation Phases

### Phase 1 (P0) - 2 weeks
- [ ] Incident Response Automation core
- [ ] Threat Modeling System

### Phase 2 (P1) - 3 weeks
- [ ] DDoS/WAF Management
- [ ] Vulnerability Disclosure Program
- [ ] Multi-Compliance Automation (ISO 27001, GDPR)

### Phase 3 (P2) - 2 weeks
- [ ] Security Training Platform
- [ ] Phishing Simulation
- [ ] Integration testing

---

## 12. Success Metrics

| Metric | Target |
|--------|--------|
| MTTD (Mean Time to Detect) | <5 minutes |
| MTTR (Mean Time to Respond) | <15 minutes |
| Playbook automation rate | >70% of incidents |
| Threat model coverage | 100% of services |
| WAF false positive rate | <2% |
| DDoS mitigation success | >99.9% |
| VDP response time | <24 hours |
| DSAR SLA compliance | 100% (30 days) |
| Training completion | 100% annually |
| Phishing click rate | <5% |

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-06 | Security Team | Initial version |
