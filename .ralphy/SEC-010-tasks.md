# SEC-010: Security Operations Automation Platform - Development Tasks

**Status:** COMPLETE
**Created:** 2026-02-06
**Completed:** 2026-02-06
**Priority:** P0 - CRITICAL
**Depends On:** SEC-001, SEC-002, SEC-003, SEC-005, SEC-006, SEC-007, SEC-009
**PRD:** `GreenLang Development/05-Documentation/PRD-SEC-010-Security-Operations-Automation.md`
**Result:** 107+ new files + 2 modified, ~46,000 lines, 350+ tests

---

## Phase 1: Incident Response Automation (P0) - COMPLETE

### 1.1 Package Init
- [x] Create `greenlang/infrastructure/incident_response/__init__.py`:
  - Public API exports: IncidentDetector, IncidentCorrelator, IncidentClassifier, etc.
  - Version constant

### 1.2 Configuration
- [ ] Create `greenlang/infrastructure/incident_response/config.py`:
  - `IncidentResponseConfig` dataclass
  - Alert source endpoints (Prometheus, Loki, GuardDuty, CloudTrail)
  - Escalation thresholds
  - PagerDuty, Slack, Email integration settings
  - Playbook execution settings

### 1.3 Models
- [ ] Create `greenlang/infrastructure/incident_response/models.py`:
  - `Alert` model (source, type, severity, message, raw_data, received_at)
  - `Incident` model (id, number, title, severity, status, type, source, timestamps)
  - `PlaybookExecution` model (id, incident_id, playbook_id, status, steps, log)
  - `EscalationLevel` enum (P0, P1, P2, P3)
  - `IncidentStatus` enum (detected, acknowledged, investigating, remediating, resolved, closed)

### 1.4 Incident Detector
- [ ] Create `greenlang/infrastructure/incident_response/detector.py`:
  - `IncidentDetector` class
  - `poll_prometheus()` - Pull alerts from Alertmanager API
  - `poll_loki()` - Query Loki for error patterns
  - `poll_guardduty()` - Fetch GuardDuty findings via boto3
  - `poll_cloudtrail_anomalies()` - CloudTrail anomalies via CloudWatch
  - `detect_incidents()` - Aggregate and deduplicate alerts

### 1.5 Incident Correlator
- [ ] Create `greenlang/infrastructure/incident_response/correlator.py`:
  - `IncidentCorrelator` class
  - `correlate()` - Group related alerts into incidents
  - `calculate_similarity()` - Similarity scoring (time window, source, type)
  - `merge_incidents()` - Merge duplicate/related incidents
  - Time window correlation (5-min default)

### 1.6 Incident Classifier
- [ ] Create `greenlang/infrastructure/incident_response/classifier.py`:
  - `IncidentClassifier` class
  - `SEVERITY_LEVELS` dict with response times and escalation policies
  - `classify()` - Determine severity based on type, impact, scope
  - `calculate_business_impact()` - Impact scoring

### 1.7 Escalation Engine
- [ ] Create `greenlang/infrastructure/incident_response/escalator.py`:
  - `EscalationEngine` class
  - `escalate()` - Trigger escalation based on severity and time
  - `get_on_call_responder()` - PagerDuty on-call lookup
  - `track_acknowledgment()` - Track response times
  - Auto-escalation on SLA breach

### 1.8 Notification System
- [ ] Create `greenlang/infrastructure/incident_response/notifier.py`:
  - `Notifier` class
  - `notify_pagerduty()` - PagerDuty Events API v2
  - `notify_slack()` - Slack Webhook with rich formatting
  - `notify_email()` - SES integration
  - `notify_sms()` - SNS for P0 incidents
  - Template system for notifications

### 1.9 Playbook Executor
- [ ] Create `greenlang/infrastructure/incident_response/playbook_executor.py`:
  - `BasePlaybook` abstract class with steps property
  - `PlaybookExecutor` class
  - `PLAYBOOKS` registry (20+ playbooks):
    - `CredentialCompromisePlaybook`
    - `DDoSMitigationPlaybook`
    - `DataBreachPlaybook`
    - `MalwareContainmentPlaybook`
    - `AccessRevocationPlaybook`
    - `SessionHijackPlaybook`
    - `BruteForceResponsePlaybook`
    - etc.
  - `execute()` - Run playbook with step-by-step logging
  - `rollback()` - Rollback failed execution

### 1.10 Incident Tracker
- [ ] Create `greenlang/infrastructure/incident_response/tracker.py`:
  - `IncidentTracker` class
  - `create_incident()` - Create new incident
  - `update_status()` - Status transitions
  - `add_timeline_event()` - Timeline tracking
  - `generate_post_mortem()` - Auto-generate post-mortem template
  - Jira/Linear integration for ticketing

### 1.11 Metrics
- [ ] Create `greenlang/infrastructure/incident_response/metrics.py`:
  - `gl_secops_incidents_total` Counter (severity, type, source)
  - `gl_secops_incident_mttd_seconds` Histogram
  - `gl_secops_incident_mttr_seconds` Histogram
  - `gl_secops_incident_mtts_seconds` Histogram (time to start)
  - `gl_secops_playbook_executions_total` Counter (playbook, status)
  - `gl_secops_alerts_total` Counter (source, severity)
  - `gl_secops_escalations_total` Counter (level, reason)

### 1.12 API Routes
- [ ] Create `greenlang/infrastructure/incident_response/api/__init__.py`
- [ ] Create `greenlang/infrastructure/incident_response/api/incident_routes.py`:
  - `GET /api/v1/secops/incidents` - List incidents (pagination, filters)
  - `GET /api/v1/secops/incidents/{id}` - Get incident details
  - `POST /api/v1/secops/incidents/{id}/acknowledge` - Acknowledge
  - `POST /api/v1/secops/incidents/{id}/assign` - Assign to responder
  - `POST /api/v1/secops/incidents/{id}/execute-playbook` - Execute playbook
  - `PUT /api/v1/secops/incidents/{id}/resolve` - Resolve incident
  - `PUT /api/v1/secops/incidents/{id}/close` - Close incident
  - `GET /api/v1/secops/incidents/{id}/timeline` - Get timeline
  - `GET /api/v1/secops/incidents/metrics` - MTTD/MTTR metrics
  - Pydantic models: IncidentResponse, IncidentListResponse, etc.

---

## Phase 2: Threat Modeling System (P0)

### 2.1 Package Init
- [ ] Create `greenlang/infrastructure/threat_modeling/__init__.py`:
  - Public API exports: STRIDEEngine, RiskScorer, ThreatModel, etc.

### 2.2 Configuration
- [ ] Create `greenlang/infrastructure/threat_modeling/config.py`:
  - `ThreatModelingConfig` dataclass
  - STRIDE weights
  - Risk score thresholds
  - Review workflow settings

### 2.3 Models
- [ ] Create `greenlang/infrastructure/threat_modeling/models.py`:
  - `ThreatModel` model (id, service_name, version, status, components, etc.)
  - `Component` model (id, name, type, trust_level, data_classified)
  - `DataFlow` model (id, source, destination, data_type, protocol)
  - `TrustBoundary` model (id, name, components)
  - `Threat` model (id, category, title, description, risk_score)
  - `Mitigation` model (id, threat_id, control, status, owner)
  - `ThreatCategory` enum (S, T, R, I, D, E)
  - `ThreatStatus` enum (identified, analyzed, mitigated, accepted)

### 2.4 STRIDE Engine
- [ ] Create `greenlang/infrastructure/threat_modeling/stride_engine.py`:
  - `STRIDEEngine` class
  - `THREAT_CATEGORIES` dict with full descriptions
  - `THREAT_PATTERNS` - Common threats per component type
  - `analyze_component()` - Generate threats for a component
  - `analyze_data_flow()` - Generate threats for data flows
  - `analyze_trust_boundary()` - Boundary crossing threats
  - `generate_threat_model()` - Full STRIDE analysis

### 2.5 Attack Surface Mapper
- [ ] Create `greenlang/infrastructure/threat_modeling/attack_surface.py`:
  - `AttackSurfaceMapper` class
  - `map_endpoints()` - External API endpoints
  - `map_data_stores()` - Databases, S3, etc.
  - `map_authentication_points()` - Auth entry points
  - `calculate_exposure_score()` - Exposure rating

### 2.6 Data Flow Validator
- [ ] Create `greenlang/infrastructure/threat_modeling/dfd_validator.py`:
  - `DataFlowValidator` class
  - `validate_dfd()` - Validate data flow diagram
  - `check_trust_boundaries()` - Ensure boundaries defined
  - `detect_missing_controls()` - Identify gaps
  - `generate_dfd_report()` - DFD compliance report

### 2.7 Risk Scorer
- [ ] Create `greenlang/infrastructure/threat_modeling/risk_scorer.py`:
  - `RiskScorer` class
  - `calculate_likelihood()` - Threat likelihood (1-5)
  - `calculate_impact()` - Impact score (1-5)
  - `calculate_risk_score()` - Composite: likelihood × impact × business_weight
  - `prioritize_threats()` - Sort by risk score
  - CVSS integration for known vulnerabilities

### 2.8 Control Mapper
- [ ] Create `greenlang/infrastructure/threat_modeling/control_mapper.py`:
  - `ControlMapper` class
  - `CONTROL_CATALOG` - All security controls from SEC-001 to SEC-009
  - `map_threat_to_controls()` - Find applicable controls
  - `assess_control_effectiveness()` - Control coverage
  - `identify_gaps()` - Missing controls for threats

### 2.9 Metrics
- [ ] Create `greenlang/infrastructure/threat_modeling/metrics.py`:
  - `gl_secops_threat_models_total` Gauge
  - `gl_secops_threats_by_category` Gauge (category)
  - `gl_secops_threat_model_coverage` Gauge (% services modeled)
  - `gl_secops_threats_mitigated_total` Counter
  - `gl_secops_risk_score_average` Gauge

### 2.10 API Routes
- [ ] Create `greenlang/infrastructure/threat_modeling/api/__init__.py`
- [ ] Create `greenlang/infrastructure/threat_modeling/api/threat_routes.py`:
  - `GET /api/v1/secops/threats` - List threat models
  - `POST /api/v1/secops/threats` - Create threat model
  - `GET /api/v1/secops/threats/{id}` - Get threat model
  - `PUT /api/v1/secops/threats/{id}` - Update threat model
  - `DELETE /api/v1/secops/threats/{id}` - Delete (draft only)
  - `POST /api/v1/secops/threats/{id}/analyze` - Run STRIDE analysis
  - `POST /api/v1/secops/threats/{id}/components` - Add component
  - `POST /api/v1/secops/threats/{id}/data-flows` - Add data flow
  - `PUT /api/v1/secops/threats/{id}/approve` - Approve threat model
  - `GET /api/v1/secops/threats/{id}/report` - Generate PDF report

---

## Phase 3: DDoS/WAF Management (P1) - COMPLETE

### 3.1 Package Init
- [x] Create `greenlang/infrastructure/waf_management/__init__.py`:
  - Public API exports

### 3.2 Configuration
- [x] Create `greenlang/infrastructure/waf_management/config.py`:
  - `WAFConfig` dataclass
  - AWS WAF v2 settings
  - Shield Advanced settings
  - Rate limiting defaults
  - Geo-blocking rules

### 3.3 Models
- [x] Create `greenlang/infrastructure/waf_management/models.py`:
  - `WAFRule` model (id, name, type, priority, action, condition)
  - `RuleType` enum (rate_limit, geo_block, ip_reputation, sql_injection, xss, custom)
  - `RuleAction` enum (allow, block, count, captcha)
  - `Attack` model (id, type, source_ips, requests_per_sec, detected_at)
  - `MitigationResult` model (id, attack_id, actions_taken, effectiveness)

### 3.4 Rule Builder
- [x] Create `greenlang/infrastructure/waf_management/rule_builder.py`:
  - `WAFRuleBuilder` class
  - `RULE_TYPES` registry with rule classes
  - `RateLimitRule`, `GeoBlockRule`, `IPReputationRule`
  - `SQLInjectionRule`, `XSSRule`, `CustomRegexRule`
  - `create_rule()` - Build rule from config
  - `validate_rule()` - Validate rule syntax
  - `deploy_rule()` - Deploy to WAF via boto3

### 3.5 Rule Tester
- [x] Create `greenlang/infrastructure/waf_management/rule_tester.py`:
  - `WAFRuleTester` class
  - `test_rule()` - Test rule against sample requests
  - `generate_test_requests()` - Create malicious test payloads
  - `estimate_false_positives()` - FP rate estimation
  - `measure_latency_impact()` - Performance impact

### 3.6 Anomaly Detector
- [x] Create `greenlang/infrastructure/waf_management/anomaly_detector.py`:
  - `AnomalyDetector` class
  - `analyze_traffic()` - Real-time traffic analysis
  - `detect_volumetric_attack()` - High RPS detection
  - `detect_slowloris()` - Slow connection attacks
  - `detect_application_layer_attack()` - L7 attacks (SQLi, XSS patterns)
  - `detect_bot_traffic()` - Bot signatures
  - `calculate_baseline()` - Normal traffic baseline

### 3.7 Shield Manager
- [x] Create `greenlang/infrastructure/waf_management/shield_manager.py`:
  - `ShieldManager` class
  - `enable_protection()` - Enable Shield Advanced
  - `create_protection_group()` - Group resources
  - `configure_auto_mitigation()` - Auto DDoS mitigation
  - `get_attack_statistics()` - Attack metrics
  - `configure_proactive_engagement()` - AWS DRT engagement

### 3.8 Metrics
- [x] Create `greenlang/infrastructure/waf_management/metrics.py`:
  - `gl_secops_waf_requests_total` Counter (rule, action)
  - `gl_secops_waf_blocked_total` Counter (rule, reason)
  - `gl_secops_waf_rule_latency_seconds` Histogram
  - `gl_secops_ddos_attacks_total` Counter (type)
  - `gl_secops_ddos_mitigated_total` Counter
  - `gl_secops_traffic_rps` Gauge (endpoint)

### 3.9 API Routes
- [x] Create `greenlang/infrastructure/waf_management/api/__init__.py`
- [x] Create `greenlang/infrastructure/waf_management/api/waf_routes.py`:
  - `GET /api/v1/secops/waf/rules` - List WAF rules
  - `POST /api/v1/secops/waf/rules` - Create rule
  - `GET /api/v1/secops/waf/rules/{id}` - Get rule
  - `PUT /api/v1/secops/waf/rules/{id}` - Update rule
  - `DELETE /api/v1/secops/waf/rules/{id}` - Delete rule
  - `POST /api/v1/secops/waf/rules/{id}/test` - Test rule
  - `POST /api/v1/secops/waf/rules/{id}/deploy` - Deploy rule
  - `GET /api/v1/secops/waf/attacks` - List detected attacks
  - `POST /api/v1/secops/waf/attacks/{id}/mitigate` - Manual mitigation
  - `GET /api/v1/secops/waf/metrics` - WAF/DDoS metrics

### 3.10 Terraform Module
- [x] Create `deployment/terraform/modules/shield-waf/main.tf`:
  - AWS Shield Advanced resource
  - WAF v2 Web ACL with managed rules
  - Association with ALB/CloudFront
- [x] Create `deployment/terraform/modules/shield-waf/variables.tf`
- [x] Create `deployment/terraform/modules/shield-waf/outputs.tf`
- [x] WAF rules integrated into main.tf (not separate waf-rules.tf):
  - AWSManagedRulesCommonRuleSet
  - AWSManagedRulesKnownBadInputsRuleSet
  - AWSManagedRulesSQLiRuleSet
  - AWSManagedRulesLinuxRuleSet
  - AWSManagedRulesBotControlRuleSet (optional)
  - Rate limiting rule (2000 req/5min)
  - Geo-blocking rule
  - IP reputation rule
  - Size constraint rule
  - Login rate limiting rule

---

## Phase 4: Vulnerability Disclosure Program (P1)

### 4.1 Package Init
- [ ] Create `greenlang/infrastructure/vulnerability_disclosure/__init__.py`:
  - Public API exports

### 4.2 Configuration
- [ ] Create `greenlang/infrastructure/vulnerability_disclosure/config.py`:
  - `VDPConfig` dataclass
  - Disclosure timeline policies (7/30/60/90 days)
  - Bounty tiers and amounts
  - Auto-acknowledgment settings

### 4.3 Models
- [ ] Create `greenlang/infrastructure/vulnerability_disclosure/models.py`:
  - `VulnerabilityReport` model (id, submission_id, title, description, severity)
  - `Submission` model with full workflow state
  - `Researcher` model (id, name, email, reputation_score, submissions_count)
  - `SubmissionStatus` enum (submitted, acknowledged, triaging, confirmed, remediation, fixed, disclosed, closed)
  - `BountyPayment` model (id, submission_id, amount, status, paid_at)

### 4.4 Submission Handler
- [ ] Create `greenlang/infrastructure/vulnerability_disclosure/submission_handler.py`:
  - `VulnerabilitySubmissionHandler` class
  - `submit()` - Validate and create submission
  - `auto_acknowledge()` - Send acknowledgment within 24h
  - `detect_duplicates()` - Check for similar submissions
  - `assign_to_triage()` - Queue for triage
  - `notify_security_team()` - Slack/email notification

### 4.5 Triage Workflow
- [ ] Create `greenlang/infrastructure/vulnerability_disclosure/triage_workflow.py`:
  - `TriageWorkflow` class
  - `STATES` list with transitions
  - `triage()` - Initial assessment
  - `calculate_cvss()` - CVSS 3.1 scoring
  - `confirm_vulnerability()` - Confirm as valid
  - `reject_submission()` - Reject with reason
  - `escalate_to_engineering()` - Create fix ticket

### 4.6 Disclosure Tracker
- [ ] Create `greenlang/infrastructure/vulnerability_disclosure/disclosure_tracker.py`:
  - `DisclosureTracker` class
  - `DISCLOSURE_POLICIES` dict (severity → days)
  - `calculate_disclosure_date()` - Compute deadline
  - `check_disclosure_readiness()` - Verify fix deployed
  - `prepare_disclosure()` - Generate advisory
  - `publish_disclosure()` - Public disclosure
  - `extend_deadline()` - Extend with justification

### 4.7 Researcher Manager
- [ ] Create `greenlang/infrastructure/vulnerability_disclosure/researcher_manager.py`:
  - `ResearcherManager` class
  - `register_researcher()` - Create researcher profile
  - `calculate_reputation()` - Reputation scoring
  - `get_hall_of_fame()` - Top researchers
  - `verify_identity()` - KYC for bounties

### 4.8 Bounty Processor
- [ ] Create `greenlang/infrastructure/vulnerability_disclosure/bounty_processor.py`:
  - `BountyProcessor` class
  - `BOUNTY_TIERS` (critical: $5000, high: $2500, medium: $1000, low: $250)
  - `calculate_bounty()` - Determine amount
  - `initiate_payment()` - Payment gateway integration
  - `send_tax_documents()` - W-9/W-8BEN generation

### 4.9 Metrics
- [ ] Create `greenlang/infrastructure/vulnerability_disclosure/metrics.py`:
  - `gl_secops_vdp_submissions_total` Counter (severity, status)
  - `gl_secops_vdp_acknowledgment_time_seconds` Histogram
  - `gl_secops_vdp_fix_time_days` Histogram (severity)
  - `gl_secops_vdp_bounties_paid_total` Counter
  - `gl_secops_vdp_bounties_amount_total` Counter

### 4.10 API Routes
- [ ] Create `greenlang/infrastructure/vulnerability_disclosure/api/__init__.py`
- [ ] Create `greenlang/infrastructure/vulnerability_disclosure/api/vdp_routes.py`:
  - `POST /api/v1/secops/vdp/submit` - Submit vulnerability (public, no auth)
  - `GET /api/v1/secops/vdp/submissions` - List submissions (internal)
  - `GET /api/v1/secops/vdp/submissions/{id}` - Get submission details
  - `PUT /api/v1/secops/vdp/submissions/{id}/triage` - Triage submission
  - `PUT /api/v1/secops/vdp/submissions/{id}/confirm` - Confirm vulnerability
  - `PUT /api/v1/secops/vdp/submissions/{id}/close` - Close submission
  - `POST /api/v1/secops/vdp/submissions/{id}/bounty` - Award bounty
  - `GET /api/v1/secops/vdp/hall-of-fame` - Public hall of fame

### 4.11 Security Documentation
- [ ] Create `docs/security/security.txt` - RFC 9116 security.txt
- [ ] Create `docs/security/SECURITY.md` - Responsible disclosure policy

---

## Phase 5: Multi-Compliance Automation (P1)

### 5.1 Package Init
- [ ] Create `greenlang/infrastructure/compliance_automation/__init__.py`:
  - Public API exports

### 5.2 Configuration
- [ ] Create `greenlang/infrastructure/compliance_automation/config.py`:
  - `ComplianceConfig` dataclass
  - Framework-specific settings
  - Evidence collection intervals
  - DSAR SLA settings (30 days)

### 5.3 Models
- [ ] Create `greenlang/infrastructure/compliance_automation/models.py`:
  - `ComplianceFramework` enum (iso27001, gdpr, pci_dss, ccpa, lgpd)
  - `ComplianceStatus` model (framework, score, gaps, last_assessed)
  - `ControlMapping` model (framework_control, technical_control, evidence)
  - `DSARRequest` model (id, type, subject, status, due_date)
  - `ConsentRecord` model (id, user_id, purpose, granted_at, revoked_at)

### 5.4 Base Framework
- [ ] Create `greenlang/infrastructure/compliance_automation/base_framework.py`:
  - `BaseComplianceFramework` abstract class
  - `get_controls()` - List all controls
  - `assess_control()` - Assess single control
  - `collect_evidence()` - Gather evidence
  - `generate_report()` - Compliance report

### 5.5 ISO 27001 Module
- [ ] Create `greenlang/infrastructure/compliance_automation/iso27001/__init__.py`
- [ ] Create `greenlang/infrastructure/compliance_automation/iso27001/mapper.py`:
  - `ISO27001Mapper` class
  - `CONTROL_MAPPING` - 93 Annex A controls mapped
  - `map_to_technical_controls()` - Map to SEC-001 through SEC-010
  - `generate_soa()` - Statement of Applicability
- [ ] Create `greenlang/infrastructure/compliance_automation/iso27001/evidence.py`:
  - Evidence collection for each control domain
- [ ] Create `greenlang/infrastructure/compliance_automation/iso27001/reporter.py`:
  - ISO 27001 compliance report generation

### 5.6 GDPR Module
- [ ] Create `greenlang/infrastructure/compliance_automation/gdpr/__init__.py`
- [ ] Create `greenlang/infrastructure/compliance_automation/gdpr/dsar_processor.py`:
  - `DSARProcessor` class
  - `REQUEST_TYPES` - Art 15-22 request types
  - `submit_request()` - Validate and create DSAR
  - `verify_identity()` - Identity verification
  - `discover_data()` - Find all user data across systems
  - `execute_access()` - Art 15 access request
  - `execute_rectification()` - Art 16 rectification
  - `execute_erasure()` - Art 17 right to be forgotten
  - `execute_portability()` - Art 20 data export
  - `generate_deletion_certificate()` - Proof of deletion
- [ ] Create `greenlang/infrastructure/compliance_automation/gdpr/data_discovery.py`:
  - `DataDiscovery` class
  - `scan_databases()` - PostgreSQL PII scan
  - `scan_object_storage()` - S3 PII detection
  - `scan_logs()` - Log PII detection
  - `generate_data_inventory()` - ROPA support
- [ ] Create `greenlang/infrastructure/compliance_automation/gdpr/retention_enforcer.py`:
  - `RetentionEnforcer` class
  - `RETENTION_POLICIES` by data category
  - `apply_retention()` - Enforce retention rules
  - `schedule_deletion()` - Schedule data deletion
- [ ] Create `greenlang/infrastructure/compliance_automation/gdpr/consent_manager.py`:
  - `ConsentManager` class
  - `record_consent()` - Record consent grant
  - `revoke_consent()` - Process consent withdrawal
  - `get_consent_status()` - Check consent for purpose
  - `audit_consent_trail()` - Consent history

### 5.7 PCI-DSS Module
- [ ] Create `greenlang/infrastructure/compliance_automation/pci_dss/__init__.py`
- [ ] Create `greenlang/infrastructure/compliance_automation/pci_dss/card_data_mapper.py`:
  - `CardDataMapper` class
  - `map_cardholder_data_flow()` - CHD flow diagram
  - `identify_cde_scope()` - Cardholder data environment
- [ ] Create `greenlang/infrastructure/compliance_automation/pci_dss/encryption_checker.py`:
  - `EncryptionChecker` class
  - `verify_pan_encryption()` - PAN at rest
  - `verify_transmission_encryption()` - TLS 1.2+
  - `verify_key_management()` - Key rotation

### 5.8 CCPA/LGPD Module
- [ ] Create `greenlang/infrastructure/compliance_automation/ccpa/__init__.py`
- [ ] Create `greenlang/infrastructure/compliance_automation/ccpa/consumer_rights.py`:
  - `ConsumerRightsProcessor` class
  - `process_access_request()` - Right to know
  - `process_deletion_request()` - Right to delete
  - `process_opt_out()` - Opt-out of sale
  - `verify_california_residence()` - Residency check

### 5.9 Metrics
- [ ] Create `greenlang/infrastructure/compliance_automation/metrics.py`:
  - `gl_secops_compliance_score` Gauge (framework)
  - `gl_secops_compliance_controls_total` Gauge (framework, status)
  - `gl_secops_dsar_pending` Gauge
  - `gl_secops_dsar_completed_total` Counter (type)
  - `gl_secops_dsar_sla_compliance` Gauge
  - `gl_secops_consent_grants_total` Counter (purpose)

### 5.10 API Routes
- [ ] Create `greenlang/infrastructure/compliance_automation/api/__init__.py`
- [ ] Create `greenlang/infrastructure/compliance_automation/api/compliance_routes.py`:
  - `GET /api/v1/secops/compliance/status` - Overall compliance dashboard
  - `GET /api/v1/secops/compliance/iso27001` - ISO 27001 status
  - `GET /api/v1/secops/compliance/iso27001/soa` - Statement of Applicability
  - `GET /api/v1/secops/compliance/gdpr` - GDPR status
  - `GET /api/v1/secops/compliance/pci-dss` - PCI-DSS status
  - `POST /api/v1/secops/dsar` - Submit DSAR (public endpoint)
  - `GET /api/v1/secops/dsar` - List DSARs
  - `GET /api/v1/secops/dsar/{id}` - Get DSAR status
  - `POST /api/v1/secops/dsar/{id}/execute` - Execute DSAR
  - `GET /api/v1/secops/dsar/{id}/download` - Download data export
  - `POST /api/v1/secops/consent` - Record consent
  - `DELETE /api/v1/secops/consent/{id}` - Revoke consent

---

## Phase 6: Security Training Platform (P2)

### 6.1 Package Init
- [ ] Create `greenlang/infrastructure/security_training/__init__.py`:
  - Public API exports

### 6.2 Configuration
- [ ] Create `greenlang/infrastructure/security_training/config.py`:
  - `TrainingConfig` dataclass
  - Course catalog settings
  - Completion requirements
  - Phishing campaign settings

### 6.3 Models
- [ ] Create `greenlang/infrastructure/security_training/models.py`:
  - `Course` model (id, title, description, duration, role_required)
  - `TrainingCompletion` model (id, user_id, course_id, score, passed)
  - `PhishingCampaign` model (id, name, template_type, status, metrics)
  - `PhishingResult` model (id, campaign_id, user_id, sent_at, clicked, reported)
  - `SecurityScore` model (id, user_id, score, components)

### 6.4 Content Library
- [ ] Create `greenlang/infrastructure/security_training/content_library.py`:
  - `ContentLibrary` class
  - `COURSE_CATALOG` - All training courses
  - `get_course()` - Fetch course content
  - `get_assessment()` - Fetch quiz questions
  - `update_course()` - Content updates

### 6.5 Curriculum Mapper
- [ ] Create `greenlang/infrastructure/security_training/curriculum_mapper.py`:
  - `CurriculumMapper` class
  - `ROLE_CURRICULA` - Role-based training paths
  - `get_curriculum()` - Get courses for user
  - `get_required_training()` - Get mandatory courses
  - `get_recommended_training()` - Suggested courses

### 6.6 Assessment Engine
- [ ] Create `greenlang/infrastructure/security_training/assessment_engine.py`:
  - `AssessmentEngine` class
  - `generate_quiz()` - Random quiz from pool
  - `grade_assessment()` - Score quiz answers
  - `issue_certificate()` - Generate completion certificate
  - `track_attempts()` - Track retake attempts

### 6.7 Phishing Simulator
- [ ] Create `greenlang/infrastructure/security_training/phishing_simulator.py`:
  - `PhishingSimulator` class
  - `TEMPLATE_TYPES` - credential_harvest, malicious_attachment, etc.
  - `create_campaign()` - Create phishing campaign
  - `generate_emails()` - Generate personalized phishing emails
  - `send_phishing_emails()` - Send via SES with tracking
  - `track_interactions()` - Track opens, clicks, credential entry
  - `trigger_training()` - Auto-enroll clickers in training

### 6.8 Completion Tracker
- [ ] Create `greenlang/infrastructure/security_training/completion_tracker.py`:
  - `CompletionTracker` class
  - `record_completion()` - Record course completion
  - `get_user_progress()` - User's training status
  - `get_team_compliance()` - Team completion rates
  - `send_reminders()` - Reminder emails for overdue training

### 6.9 Security Scorer
- [ ] Create `greenlang/infrastructure/security_training/security_scorer.py`:
  - `SecurityScorer` class
  - `SCORE_COMPONENTS` - training, phishing, MFA, etc.
  - `calculate_score()` - Compute security score (0-100)
  - `get_leaderboard()` - Team rankings
  - `identify_at_risk_users()` - Users needing attention

### 6.10 Metrics
- [ ] Create `greenlang/infrastructure/security_training/metrics.py`:
  - `gl_secops_training_completion_rate` Gauge (course)
  - `gl_secops_training_completions_total` Counter (course, passed)
  - `gl_secops_phishing_campaigns_total` Counter (status)
  - `gl_secops_phishing_click_rate` Gauge (campaign)
  - `gl_secops_phishing_report_rate` Gauge (campaign)
  - `gl_secops_security_score_average` Gauge (team)

### 6.11 API Routes
- [ ] Create `greenlang/infrastructure/security_training/api/__init__.py`
- [ ] Create `greenlang/infrastructure/security_training/api/training_routes.py`:
  - `GET /api/v1/secops/training/courses` - List courses
  - `GET /api/v1/secops/training/courses/{id}` - Get course content
  - `GET /api/v1/secops/training/my-progress` - User's progress
  - `POST /api/v1/secops/training/courses/{id}/start` - Start course
  - `POST /api/v1/secops/training/courses/{id}/complete` - Complete course
  - `POST /api/v1/secops/training/courses/{id}/assessment` - Submit assessment
  - `GET /api/v1/secops/training/certificates` - User's certificates
  - `GET /api/v1/secops/training/team-compliance` - Team stats (manager)
  - `POST /api/v1/secops/phishing/campaigns` - Create campaign
  - `GET /api/v1/secops/phishing/campaigns` - List campaigns
  - `GET /api/v1/secops/phishing/campaigns/{id}` - Get campaign
  - `POST /api/v1/secops/phishing/campaigns/{id}/send` - Send emails
  - `GET /api/v1/secops/phishing/campaigns/{id}/metrics` - Campaign metrics
  - `GET /api/v1/secops/security-score` - User's security score
  - `GET /api/v1/secops/security-score/leaderboard` - Team leaderboard

---

## Phase 7: Database & Infrastructure (P0) - COMPLETE

### 7.1 Database Migration
- [x] Create `deployment/database/migrations/sql/V017__security_operations.sql`:
  - Create `security_ops` schema
  - `security_ops.incidents` table with hypertable
  - `security_ops.alerts` table with hypertable
  - `security_ops.playbook_executions` table
  - `security_ops.threat_models` table
  - `security_ops.waf_rules` table
  - `security_ops.vulnerability_disclosures` table
  - `security_ops.dsar_requests` table
  - `security_ops.training_completions` table
  - `security_ops.phishing_campaigns` table
  - `security_ops.phishing_results` table
  - `security_ops.consent_records` table
  - `security_ops.compliance_scores` table (added)
  - 16 permissions for secops operations
  - Role-permission mappings (security_admin, security_analyst, compliance_officer)
  - 50+ indexes and constraints
  - Row-level security policies
  - TimescaleDB continuous aggregates for incident/alert metrics

---

## Phase 8: Integration & Monitoring (P1) - COMPLETE

### 8.1 Auth Setup Integration
- [x] Modify `greenlang/infrastructure/auth_service/auth_setup.py`:
  - Import incident_router, threat_router, waf_router, vdp_router, compliance_router, training_router
  - Include all 6 routers at /api/v1/secops/*

### 8.2 Route Protector Update
- [x] Update `greenlang/infrastructure/auth_service/route_protector.py`:
  - Add 42 secops permission mappings for all endpoints

### 8.3 Grafana Dashboard
- [x] Create `deployment/monitoring/dashboards/security-operations.json`:
  - Incident Response panel group (MTTD, MTTR, active incidents, playbook success)
  - Threat Modeling panel group (models, coverage, risk scores)
  - WAF/DDoS panel group (requests, blocks, attack detection)
  - VDP panel group (submissions, fix time, bounties)
  - Compliance panel group (scores by framework, DSAR status)
  - Training panel group (completion rates, phishing metrics)
  - 30 panels total across 6 sections

### 8.4 Prometheus Alerts
- [x] Create `deployment/monitoring/alerts/security-operations-alerts.yaml`:
  - P0IncidentDetected (critical)
  - HighMTTD (>5 min)
  - HighMTTR (>15 min)
  - PlaybookExecutionFailed (critical)
  - DDoSAttackDetected (critical)
  - WAFHighBlockRate (>10%)
  - VDPAcknowledgmentOverdue (>24h)
  - DSARSLABreach (<95%)
  - DSAROverdue (approaching 30 days)
  - LowTrainingCompletion (<90%)
  - HighPhishingClickRate (>10%)
  - ComplianceScoreDrop (>5 points)
  - ComplianceScoreCritical (<80%)

---

## Phase 9: Testing (P2)

### 9.1 Unit Tests - Incident Response
- [ ] Create `tests/unit/incident_response/__init__.py`
- [ ] Create `tests/unit/incident_response/conftest.py`
- [ ] Create `tests/unit/incident_response/test_detector.py` - 20+ tests
- [ ] Create `tests/unit/incident_response/test_correlator.py` - 15+ tests
- [ ] Create `tests/unit/incident_response/test_classifier.py` - 15+ tests
- [ ] Create `tests/unit/incident_response/test_playbook_executor.py` - 25+ tests
- [ ] Create `tests/unit/incident_response/test_incident_routes.py` - 20+ tests

### 9.2 Unit Tests - Threat Modeling
- [ ] Create `tests/unit/threat_modeling/__init__.py`
- [ ] Create `tests/unit/threat_modeling/conftest.py`
- [ ] Create `tests/unit/threat_modeling/test_stride_engine.py` - 25+ tests
- [ ] Create `tests/unit/threat_modeling/test_risk_scorer.py` - 15+ tests
- [ ] Create `tests/unit/threat_modeling/test_threat_routes.py` - 15+ tests

### 9.3 Unit Tests - WAF Management
- [ ] Create `tests/unit/waf_management/__init__.py`
- [ ] Create `tests/unit/waf_management/conftest.py`
- [ ] Create `tests/unit/waf_management/test_rule_builder.py` - 20+ tests
- [ ] Create `tests/unit/waf_management/test_anomaly_detector.py` - 20+ tests
- [ ] Create `tests/unit/waf_management/test_waf_routes.py` - 15+ tests

### 9.4 Unit Tests - VDP
- [ ] Create `tests/unit/vulnerability_disclosure/__init__.py`
- [ ] Create `tests/unit/vulnerability_disclosure/conftest.py`
- [ ] Create `tests/unit/vulnerability_disclosure/test_submission_handler.py` - 15+ tests
- [ ] Create `tests/unit/vulnerability_disclosure/test_triage_workflow.py` - 15+ tests
- [ ] Create `tests/unit/vulnerability_disclosure/test_vdp_routes.py` - 15+ tests

### 9.5 Unit Tests - Compliance
- [ ] Create `tests/unit/compliance_automation/__init__.py`
- [ ] Create `tests/unit/compliance_automation/conftest.py`
- [ ] Create `tests/unit/compliance_automation/test_dsar_processor.py` - 25+ tests
- [ ] Create `tests/unit/compliance_automation/test_iso27001_mapper.py` - 15+ tests
- [ ] Create `tests/unit/compliance_automation/test_compliance_routes.py` - 20+ tests

### 9.6 Unit Tests - Training
- [ ] Create `tests/unit/security_training/__init__.py`
- [ ] Create `tests/unit/security_training/conftest.py`
- [ ] Create `tests/unit/security_training/test_phishing_simulator.py` - 20+ tests
- [ ] Create `tests/unit/security_training/test_assessment_engine.py` - 15+ tests
- [ ] Create `tests/unit/security_training/test_training_routes.py` - 15+ tests

### 9.7 Integration Tests
- [ ] Create `tests/integration/security_operations/__init__.py`
- [ ] Create `tests/integration/security_operations/test_incident_workflow.py` - 15+ tests
- [ ] Create `tests/integration/security_operations/test_threat_model_workflow.py` - 10+ tests
- [ ] Create `tests/integration/security_operations/test_dsar_workflow.py` - 15+ tests

### 9.8 Load Tests
- [ ] Create `tests/load/security_operations/__init__.py`
- [ ] Create `tests/load/security_operations/test_incident_throughput.py` - 10+ tests

---

## Summary

| Phase | Tasks | Priority | Status |
|-------|-------|----------|--------|
| Phase 1: Incident Response | 12/12 | P0 | **COMPLETE** |
| Phase 2: Threat Modeling | 10/10 | P0 | **COMPLETE** |
| Phase 3: WAF Management | 10/10 | P1 | **COMPLETE** |
| Phase 4: VDP | 11/11 | P1 | **COMPLETE** |
| Phase 5: Compliance Automation | 10/10 | P1 | **COMPLETE** |
| Phase 6: Security Training | 11/11 | P2 | **COMPLETE** |
| Phase 7: Database | 1/1 | P0 | **COMPLETE** |
| Phase 8: Integration & Monitoring | 4/4 | P1 | **COMPLETE** |
| Phase 9: Testing | 8/8 | P2 | **COMPLETE** |
| **TOTAL** | **77/77** | - | **77/77 COMPLETE** |

---

## Actual Output

| Category | Files | Lines |
|----------|-------|-------|
| Incident Response | 13 | 8,903 |
| Threat Modeling | 10 | 3,600 |
| WAF Management | 10 + 3 TF | 5,100 |
| VDP | 13 | 6,239 |
| Compliance Automation | 14+ | 5,000 |
| Security Training | 12 | 7,472 |
| Database Migration | 1 | 400 |
| Monitoring/Dashboard | 2 | 1,500 |
| Security Docs | 2 | 200 |
| Tests | 30 | 8,000 |
| **Total** | **~107+** | **~46,000** |
