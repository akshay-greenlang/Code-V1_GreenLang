-- =============================================================================
-- V017: Security Operations Automation Platform
-- =============================================================================
-- Description: Creates security_ops schema for Security Operations Automation
--              including incident response, threat modeling, WAF management,
--              vulnerability disclosure, compliance automation, and security
--              training platform tables with TimescaleDB hypertables.
-- Author: GreenLang Security Team
-- PRD: SEC-010 Security Operations Automation
-- Requires: TimescaleDB (V002), uuid-ossp (V001), security schema (V009)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Schema Setup
-- -----------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS security_ops;

SET search_path TO security_ops, security, public;

-- -----------------------------------------------------------------------------
-- 1. Incidents Table (TimescaleDB Hypertable)
-- -----------------------------------------------------------------------------
-- Security incident tracking with full lifecycle management.

CREATE TABLE security_ops.incidents (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    incident_number VARCHAR(20) NOT NULL UNIQUE,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    severity VARCHAR(10) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'detected',
    incident_type VARCHAR(50) NOT NULL,
    source VARCHAR(50) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ,
    assignee_id UUID,
    playbook_id VARCHAR(50),
    playbook_execution_id UUID,
    metadata JSONB DEFAULT '{}'::jsonb,
    tenant_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Primary key for hypertable (must include time column)
    PRIMARY KEY (detected_at, id),

    -- Constraints
    CONSTRAINT chk_incident_severity CHECK (
        severity IN ('P0', 'P1', 'P2', 'P3')
    ),
    CONSTRAINT chk_incident_status CHECK (
        status IN (
            'detected', 'acknowledged', 'investigating', 'remediating',
            'resolved', 'closed', 'escalated'
        )
    ),
    CONSTRAINT chk_incident_type CHECK (
        incident_type IN (
            'credential_compromise', 'data_breach', 'ddos_attack', 'malware',
            'unauthorized_access', 'phishing', 'insider_threat', 'ransomware',
            'api_abuse', 'configuration_drift', 'vulnerability_exploitation',
            'brute_force', 'session_hijack', 'privilege_escalation', 'other'
        )
    ),
    CONSTRAINT chk_incident_source CHECK (
        source IN (
            'prometheus', 'loki', 'guardduty', 'cloudtrail', 'waf',
            'security_scanner', 'user_report', 'siem', 'ids', 'manual'
        )
    )
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable(
    'security_ops.incidents',
    'detected_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Set retention policy (7 years for compliance)
SELECT add_retention_policy(
    'security_ops.incidents',
    INTERVAL '2555 days',
    if_not_exists => TRUE
);

COMMENT ON TABLE security_ops.incidents IS
    'Security incident tracking with TimescaleDB hypertable. Tracks full '
    'incident lifecycle from detection through resolution and closure.';

-- -----------------------------------------------------------------------------
-- 2. Alerts Table (TimescaleDB Hypertable)
-- -----------------------------------------------------------------------------
-- Raw security alerts before correlation into incidents.

CREATE TABLE security_ops.alerts (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    incident_id UUID,
    alert_source VARCHAR(50) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(10) NOT NULL,
    message TEXT NOT NULL,
    raw_data JSONB,
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID,

    -- Primary key for hypertable
    PRIMARY KEY (received_at, id),

    -- Constraints
    CONSTRAINT chk_alert_severity CHECK (
        severity IN ('critical', 'high', 'medium', 'low', 'info')
    ),
    CONSTRAINT chk_alert_source CHECK (
        alert_source IN (
            'prometheus', 'loki', 'guardduty', 'cloudtrail', 'waf',
            'security_scanner', 'trivy', 'falco', 'crowdstrike', 'other'
        )
    )
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable(
    'security_ops.alerts',
    'received_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Set retention policy (90 days for raw alerts)
SELECT add_retention_policy(
    'security_ops.alerts',
    INTERVAL '90 days',
    if_not_exists => TRUE
);

COMMENT ON TABLE security_ops.alerts IS
    'Raw security alerts from various sources. Correlated into incidents '
    'by the IncidentCorrelator. Short retention for operational data.';

-- -----------------------------------------------------------------------------
-- 3. Playbook Executions Table
-- -----------------------------------------------------------------------------
-- Tracks automated remediation playbook executions.

CREATE TABLE security_ops.playbook_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id UUID,
    playbook_id VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'running',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    steps_completed INTEGER DEFAULT 0,
    steps_total INTEGER NOT NULL,
    execution_log JSONB DEFAULT '[]'::jsonb,
    rollback_available BOOLEAN DEFAULT FALSE,
    rolled_back_at TIMESTAMPTZ,
    error_message TEXT,
    tenant_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_playbook_status CHECK (
        status IN ('pending', 'running', 'completed', 'failed', 'rolled_back', 'cancelled')
    ),
    CONSTRAINT chk_steps CHECK (
        steps_completed <= steps_total AND steps_total > 0
    )
);

COMMENT ON TABLE security_ops.playbook_executions IS
    'Automated remediation playbook execution tracking. Logs each step '
    'and supports rollback for failed executions.';

-- -----------------------------------------------------------------------------
-- 4. Threat Models Table
-- -----------------------------------------------------------------------------
-- STRIDE-based threat models for services and systems.

CREATE TABLE security_ops.threat_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    service_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID,
    status VARCHAR(20) NOT NULL DEFAULT 'draft',
    overall_risk_score DECIMAL(5,2),
    components JSONB DEFAULT '[]'::jsonb,
    data_flows JSONB DEFAULT '[]'::jsonb,
    trust_boundaries JSONB DEFAULT '[]'::jsonb,
    threats JSONB DEFAULT '[]'::jsonb,
    mitigations JSONB DEFAULT '[]'::jsonb,
    category_scores JSONB DEFAULT '{}'::jsonb,
    approved_by UUID,
    approved_at TIMESTAMPTZ,
    review_notes TEXT,
    tenant_id UUID,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_threat_model_status CHECK (
        status IN ('draft', 'in_review', 'approved', 'archived', 'superseded')
    ),
    CONSTRAINT chk_risk_score CHECK (
        overall_risk_score IS NULL OR (overall_risk_score >= 0 AND overall_risk_score <= 10)
    ),

    -- Unique constraint on service + version
    CONSTRAINT uq_threat_model_version UNIQUE (service_name, version)
);

COMMENT ON TABLE security_ops.threat_models IS
    'STRIDE threat models for services. Stores components, data flows, '
    'trust boundaries, identified threats, and mitigations with risk scores.';

-- -----------------------------------------------------------------------------
-- 5. WAF Rules Table
-- -----------------------------------------------------------------------------
-- Web Application Firewall rule management.

CREATE TABLE security_ops.waf_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_name VARCHAR(100) NOT NULL,
    rule_type VARCHAR(50) NOT NULL,
    priority INTEGER NOT NULL,
    action VARCHAR(20) NOT NULL,
    condition JSONB NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID,
    deployed_at TIMESTAMPTZ,
    metrics JSONB DEFAULT '{}'::jsonb,
    description TEXT,
    web_acl_id VARCHAR(100),
    tenant_id UUID,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_waf_rule_type CHECK (
        rule_type IN (
            'rate_limit', 'geo_block', 'ip_reputation', 'sql_injection',
            'xss', 'custom_regex', 'size_constraint', 'managed', 'bot_control'
        )
    ),
    CONSTRAINT chk_waf_action CHECK (
        action IN ('allow', 'block', 'count', 'captcha', 'challenge')
    ),
    CONSTRAINT chk_waf_priority CHECK (
        priority >= 0 AND priority <= 999
    )
);

COMMENT ON TABLE security_ops.waf_rules IS
    'WAF rule definitions for AWS WAF v2. Tracks rule configuration, '
    'deployment status, and effectiveness metrics.';

-- -----------------------------------------------------------------------------
-- 6. Vulnerability Disclosures Table
-- -----------------------------------------------------------------------------
-- Vulnerability Disclosure Program (VDP) submissions.

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
    cve_id VARCHAR(20),
    bounty_amount DECIMAL(10,2),
    bounty_paid_at TIMESTAMPTZ,
    affected_component VARCHAR(100),
    reproduction_steps TEXT,
    proof_of_concept TEXT,
    fix_details TEXT,
    is_duplicate BOOLEAN DEFAULT FALSE,
    duplicate_of UUID,
    tenant_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_vdp_severity CHECK (
        severity IN ('critical', 'high', 'medium', 'low', 'informational')
    ),
    CONSTRAINT chk_vdp_status CHECK (
        status IN (
            'submitted', 'acknowledged', 'triaging', 'confirmed', 'duplicate',
            'invalid', 'remediation', 'fixed', 'disclosed', 'closed', 'wont_fix'
        )
    ),
    CONSTRAINT chk_cvss_score CHECK (
        cvss_score IS NULL OR (cvss_score >= 0 AND cvss_score <= 10)
    ),
    CONSTRAINT chk_bounty_amount CHECK (
        bounty_amount IS NULL OR bounty_amount >= 0
    )
);

COMMENT ON TABLE security_ops.vulnerability_disclosures IS
    'Vulnerability Disclosure Program submissions. Tracks full lifecycle '
    'from submission through fix and disclosure with bounty management.';

-- -----------------------------------------------------------------------------
-- 7. DSAR Requests Table
-- -----------------------------------------------------------------------------
-- GDPR Data Subject Access Requests.

CREATE TABLE security_ops.dsar_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_number VARCHAR(20) NOT NULL UNIQUE,
    request_type VARCHAR(30) NOT NULL,
    subject_email VARCHAR(255) NOT NULL,
    subject_name VARCHAR(100),
    status VARCHAR(20) NOT NULL DEFAULT 'submitted',
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    due_date TIMESTAMPTZ NOT NULL,
    verified_at TIMESTAMPTZ,
    processing_started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    data_discovered JSONB DEFAULT '[]'::jsonb,
    actions_taken JSONB DEFAULT '[]'::jsonb,
    data_export_path VARCHAR(1024),
    deletion_certificate_id VARCHAR(50),
    extended BOOLEAN DEFAULT FALSE,
    extension_reason TEXT,
    assigned_to UUID,
    notes TEXT,
    tenant_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_dsar_type CHECK (
        request_type IN (
            'access', 'rectification', 'erasure', 'portability',
            'objection', 'restriction', 'withdraw_consent'
        )
    ),
    CONSTRAINT chk_dsar_status CHECK (
        status IN (
            'submitted', 'identity_verification', 'verified', 'processing',
            'awaiting_data', 'review', 'completed', 'rejected', 'withdrawn'
        )
    )
);

COMMENT ON TABLE security_ops.dsar_requests IS
    'GDPR Data Subject Access Requests (Articles 15-22). Tracks 30-day SLA, '
    'data discovery, and execution of access, erasure, and portability requests.';

-- -----------------------------------------------------------------------------
-- 8. Consent Records Table
-- -----------------------------------------------------------------------------
-- GDPR consent tracking.

CREATE TABLE security_ops.consent_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    purpose VARCHAR(100) NOT NULL,
    granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at TIMESTAMPTZ,
    source VARCHAR(50),
    consent_version VARCHAR(20),
    ip_address VARCHAR(45),
    user_agent VARCHAR(512),
    metadata JSONB DEFAULT '{}'::jsonb,
    tenant_id UUID,

    -- Unique constraint on user + purpose (latest consent wins)
    CONSTRAINT uq_consent_user_purpose UNIQUE (user_id, purpose),

    -- Constraints
    CONSTRAINT chk_consent_source CHECK (
        source IS NULL OR source IN (
            'web_form', 'api', 'import', 'email', 'mobile_app', 'cookie_banner'
        )
    )
);

COMMENT ON TABLE security_ops.consent_records IS
    'GDPR consent tracking. Records consent grants and withdrawals '
    'for specific purposes with full audit trail.';

-- -----------------------------------------------------------------------------
-- 9. Training Completions Table
-- -----------------------------------------------------------------------------
-- Security awareness training completions.

CREATE TABLE security_ops.training_completions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    course_id VARCHAR(50) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    score INTEGER,
    passed BOOLEAN,
    certificate_id VARCHAR(50),
    attempts INTEGER DEFAULT 1,
    time_spent_minutes INTEGER,
    tenant_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Unique constraint on user + course + start time
    CONSTRAINT uq_training_attempt UNIQUE (user_id, course_id, started_at),

    -- Constraints
    CONSTRAINT chk_training_score CHECK (
        score IS NULL OR (score >= 0 AND score <= 100)
    ),
    CONSTRAINT chk_training_attempts CHECK (
        attempts >= 1
    )
);

COMMENT ON TABLE security_ops.training_completions IS
    'Security awareness training completion tracking. Records course '
    'completions, scores, and certificates for compliance reporting.';

-- -----------------------------------------------------------------------------
-- 10. Phishing Campaigns Table
-- -----------------------------------------------------------------------------
-- Phishing simulation campaigns.

CREATE TABLE security_ops.phishing_campaigns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_name VARCHAR(100) NOT NULL,
    template_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'draft',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID,
    sent_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    target_count INTEGER DEFAULT 0,
    emails_sent INTEGER DEFAULT 0,
    emails_opened INTEGER DEFAULT 0,
    clicks INTEGER DEFAULT 0,
    credentials_submitted INTEGER DEFAULT 0,
    reports INTEGER DEFAULT 0,
    training_triggered INTEGER DEFAULT 0,
    template_subject VARCHAR(200),
    template_content TEXT,
    landing_page_url VARCHAR(500),
    tenant_id UUID,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_phishing_template CHECK (
        template_type IN (
            'credential_harvest', 'malicious_attachment', 'fake_invoice',
            'urgent_action', 'ceo_fraud', 'password_reset', 'shipping_notice',
            'tax_document', 'it_support', 'custom'
        )
    ),
    CONSTRAINT chk_phishing_status CHECK (
        status IN ('draft', 'scheduled', 'sending', 'active', 'completed', 'cancelled')
    )
);

COMMENT ON TABLE security_ops.phishing_campaigns IS
    'Phishing simulation campaign management. Tracks campaign configuration, '
    'send status, and aggregate results for security awareness metrics.';

-- -----------------------------------------------------------------------------
-- 11. Phishing Results Table
-- -----------------------------------------------------------------------------
-- Individual phishing simulation results.

CREATE TABLE security_ops.phishing_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID NOT NULL REFERENCES security_ops.phishing_campaigns(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    email_address VARCHAR(255) NOT NULL,
    sent_at TIMESTAMPTZ NOT NULL,
    opened_at TIMESTAMPTZ,
    clicked_at TIMESTAMPTZ,
    credentials_entered BOOLEAN DEFAULT FALSE,
    credentials_entered_at TIMESTAMPTZ,
    reported_at TIMESTAMPTZ,
    training_completed BOOLEAN DEFAULT FALSE,
    training_completed_at TIMESTAMPTZ,
    ip_address VARCHAR(45),
    user_agent VARCHAR(512),
    tenant_id UUID,

    -- Unique constraint on campaign + user
    CONSTRAINT uq_phishing_user UNIQUE (campaign_id, user_id)
);

COMMENT ON TABLE security_ops.phishing_results IS
    'Individual phishing simulation results per user. Tracks email delivery, '
    'opens, clicks, credential submission, and reporting for awareness scoring.';

-- -----------------------------------------------------------------------------
-- 12. Compliance Scores Table
-- -----------------------------------------------------------------------------
-- Multi-framework compliance score tracking.

CREATE TABLE security_ops.compliance_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    framework VARCHAR(30) NOT NULL,
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    overall_score DECIMAL(5,2) NOT NULL,
    controls_total INTEGER NOT NULL,
    controls_compliant INTEGER NOT NULL,
    controls_partial INTEGER DEFAULT 0,
    controls_non_compliant INTEGER DEFAULT 0,
    controls_not_applicable INTEGER DEFAULT 0,
    gaps JSONB DEFAULT '[]'::jsonb,
    evidence_collected INTEGER DEFAULT 0,
    assessed_by UUID,
    tenant_id UUID,

    -- Constraints
    CONSTRAINT chk_compliance_framework CHECK (
        framework IN ('iso27001', 'gdpr', 'pci_dss', 'soc2', 'hipaa', 'ccpa', 'nist')
    ),
    CONSTRAINT chk_compliance_score CHECK (
        overall_score >= 0 AND overall_score <= 100
    ),
    CONSTRAINT chk_compliance_controls CHECK (
        controls_compliant + controls_partial + controls_non_compliant + controls_not_applicable <= controls_total
    )
);

COMMENT ON TABLE security_ops.compliance_scores IS
    'Multi-framework compliance score snapshots. Tracks assessment results '
    'for ISO 27001, GDPR, PCI-DSS, SOC 2, and other frameworks.';

-- -----------------------------------------------------------------------------
-- 13. Security Operations Permissions
-- -----------------------------------------------------------------------------
-- Insert 16 permissions for secops operations.

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
    ('secops:phishing:manage', 'phishing', 'manage', 'Manage phishing campaigns')
ON CONFLICT (name) DO NOTHING;

-- -----------------------------------------------------------------------------
-- 14. Role-Permission Mappings
-- -----------------------------------------------------------------------------
-- Grant all secops permissions to security_admin role.

INSERT INTO security.role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM security.roles r, security.permissions p
WHERE r.name = 'security_admin'
  AND p.name LIKE 'secops:%'
ON CONFLICT DO NOTHING;

-- Grant read permissions to security_analyst role
INSERT INTO security.role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM security.roles r, security.permissions p
WHERE r.name = 'security_analyst'
  AND p.name IN (
      'secops:incidents:read',
      'secops:threats:read',
      'secops:waf:read',
      'secops:vdp:read',
      'secops:compliance:read',
      'secops:dsar:read',
      'secops:training:read'
  )
ON CONFLICT DO NOTHING;

-- Grant compliance permissions to compliance_officer role
INSERT INTO security.role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM security.roles r, security.permissions p
WHERE r.name = 'compliance_officer'
  AND p.name IN (
      'secops:compliance:read',
      'secops:compliance:manage',
      'secops:dsar:read',
      'secops:dsar:process',
      'secops:training:read',
      'secops:training:manage'
  )
ON CONFLICT DO NOTHING;

-- -----------------------------------------------------------------------------
-- 15. Indexes - Incidents
-- -----------------------------------------------------------------------------

CREATE INDEX idx_incidents_severity ON security_ops.incidents (severity, status);
CREATE INDEX idx_incidents_status ON security_ops.incidents (status, detected_at DESC);
CREATE INDEX idx_incidents_type ON security_ops.incidents (incident_type, detected_at DESC);
CREATE INDEX idx_incidents_source ON security_ops.incidents (source, detected_at DESC);
CREATE INDEX idx_incidents_assignee ON security_ops.incidents (assignee_id, status)
    WHERE assignee_id IS NOT NULL;
CREATE INDEX idx_incidents_playbook ON security_ops.incidents (playbook_id)
    WHERE playbook_id IS NOT NULL;
CREATE INDEX idx_incidents_tenant ON security_ops.incidents (tenant_id, detected_at DESC)
    WHERE tenant_id IS NOT NULL;
CREATE INDEX idx_incidents_number ON security_ops.incidents (incident_number);

-- -----------------------------------------------------------------------------
-- 16. Indexes - Alerts
-- -----------------------------------------------------------------------------

CREATE INDEX idx_alerts_incident ON security_ops.alerts (incident_id, received_at DESC)
    WHERE incident_id IS NOT NULL;
CREATE INDEX idx_alerts_source ON security_ops.alerts (alert_source, received_at DESC);
CREATE INDEX idx_alerts_severity ON security_ops.alerts (severity, received_at DESC);
CREATE INDEX idx_alerts_type ON security_ops.alerts (alert_type, received_at DESC);
CREATE INDEX idx_alerts_tenant ON security_ops.alerts (tenant_id, received_at DESC)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 17. Indexes - Playbook Executions
-- -----------------------------------------------------------------------------

CREATE INDEX idx_playbook_exec_incident ON security_ops.playbook_executions (incident_id);
CREATE INDEX idx_playbook_exec_status ON security_ops.playbook_executions (status, started_at DESC);
CREATE INDEX idx_playbook_exec_playbook ON security_ops.playbook_executions (playbook_id, status);
CREATE INDEX idx_playbook_exec_tenant ON security_ops.playbook_executions (tenant_id)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 18. Indexes - Threat Models
-- -----------------------------------------------------------------------------

CREATE INDEX idx_threat_models_service ON security_ops.threat_models (service_name, version);
CREATE INDEX idx_threat_models_status ON security_ops.threat_models (status, created_at DESC);
CREATE INDEX idx_threat_models_risk ON security_ops.threat_models (overall_risk_score DESC)
    WHERE status = 'approved';
CREATE INDEX idx_threat_models_tenant ON security_ops.threat_models (tenant_id)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 19. Indexes - WAF Rules
-- -----------------------------------------------------------------------------

CREATE INDEX idx_waf_rules_type ON security_ops.waf_rules (rule_type, enabled);
CREATE INDEX idx_waf_rules_priority ON security_ops.waf_rules (priority ASC)
    WHERE enabled = TRUE;
CREATE INDEX idx_waf_rules_deployed ON security_ops.waf_rules (deployed_at DESC)
    WHERE deployed_at IS NOT NULL;
CREATE INDEX idx_waf_rules_tenant ON security_ops.waf_rules (tenant_id)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 20. Indexes - Vulnerability Disclosures
-- -----------------------------------------------------------------------------

CREATE INDEX idx_vdp_status ON security_ops.vulnerability_disclosures (status, submitted_at DESC);
CREATE INDEX idx_vdp_severity ON security_ops.vulnerability_disclosures (severity, status);
CREATE INDEX idx_vdp_deadline ON security_ops.vulnerability_disclosures (disclosure_deadline)
    WHERE status NOT IN ('disclosed', 'closed', 'invalid', 'duplicate');
CREATE INDEX idx_vdp_reporter ON security_ops.vulnerability_disclosures (reporter_email);
CREATE INDEX idx_vdp_tenant ON security_ops.vulnerability_disclosures (tenant_id)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 21. Indexes - DSAR Requests
-- -----------------------------------------------------------------------------

CREATE INDEX idx_dsar_status ON security_ops.dsar_requests (status, submitted_at DESC);
CREATE INDEX idx_dsar_due_date ON security_ops.dsar_requests (due_date)
    WHERE status NOT IN ('completed', 'rejected', 'withdrawn');
CREATE INDEX idx_dsar_type ON security_ops.dsar_requests (request_type, status);
CREATE INDEX idx_dsar_subject ON security_ops.dsar_requests (subject_email);
CREATE INDEX idx_dsar_assigned ON security_ops.dsar_requests (assigned_to, status)
    WHERE assigned_to IS NOT NULL;
CREATE INDEX idx_dsar_tenant ON security_ops.dsar_requests (tenant_id)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 22. Indexes - Consent Records
-- -----------------------------------------------------------------------------

CREATE INDEX idx_consent_user ON security_ops.consent_records (user_id, purpose);
CREATE INDEX idx_consent_purpose ON security_ops.consent_records (purpose, granted_at DESC);
CREATE INDEX idx_consent_revoked ON security_ops.consent_records (revoked_at)
    WHERE revoked_at IS NOT NULL;
CREATE INDEX idx_consent_tenant ON security_ops.consent_records (tenant_id)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 23. Indexes - Training Completions
-- -----------------------------------------------------------------------------

CREATE INDEX idx_training_user ON security_ops.training_completions (user_id, course_id);
CREATE INDEX idx_training_course ON security_ops.training_completions (course_id, passed);
CREATE INDEX idx_training_completed ON security_ops.training_completions (completed_at DESC)
    WHERE completed_at IS NOT NULL;
CREATE INDEX idx_training_tenant ON security_ops.training_completions (tenant_id)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 24. Indexes - Phishing Campaigns
-- -----------------------------------------------------------------------------

CREATE INDEX idx_phishing_camp_status ON security_ops.phishing_campaigns (status, created_at DESC);
CREATE INDEX idx_phishing_camp_type ON security_ops.phishing_campaigns (template_type);
CREATE INDEX idx_phishing_camp_tenant ON security_ops.phishing_campaigns (tenant_id)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 25. Indexes - Phishing Results
-- -----------------------------------------------------------------------------

CREATE INDEX idx_phishing_res_campaign ON security_ops.phishing_results (campaign_id);
CREATE INDEX idx_phishing_res_user ON security_ops.phishing_results (user_id);
CREATE INDEX idx_phishing_res_clicked ON security_ops.phishing_results (clicked_at)
    WHERE clicked_at IS NOT NULL;
CREATE INDEX idx_phishing_res_reported ON security_ops.phishing_results (reported_at)
    WHERE reported_at IS NOT NULL;
CREATE INDEX idx_phishing_res_tenant ON security_ops.phishing_results (tenant_id)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 26. Indexes - Compliance Scores
-- -----------------------------------------------------------------------------

CREATE INDEX idx_compliance_framework ON security_ops.compliance_scores (framework, assessed_at DESC);
CREATE INDEX idx_compliance_score ON security_ops.compliance_scores (overall_score DESC);
CREATE INDEX idx_compliance_tenant ON security_ops.compliance_scores (tenant_id)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 27. Row-Level Security
-- -----------------------------------------------------------------------------

ALTER TABLE security_ops.incidents ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_ops.alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_ops.playbook_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_ops.threat_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_ops.waf_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_ops.vulnerability_disclosures ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_ops.dsar_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_ops.consent_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_ops.training_completions ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_ops.phishing_campaigns ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_ops.phishing_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_ops.compliance_scores ENABLE ROW LEVEL SECURITY;

-- Generic tenant isolation policy for all tables
CREATE POLICY tenant_isolation ON security_ops.incidents
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    );

CREATE POLICY tenant_isolation ON security_ops.alerts
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    );

CREATE POLICY tenant_isolation ON security_ops.playbook_executions
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    );

CREATE POLICY tenant_isolation ON security_ops.threat_models
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    );

CREATE POLICY tenant_isolation ON security_ops.waf_rules
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    );

CREATE POLICY tenant_isolation ON security_ops.vulnerability_disclosures
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    );

CREATE POLICY tenant_isolation ON security_ops.dsar_requests
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin', 'compliance_officer')
    );

CREATE POLICY tenant_isolation ON security_ops.consent_records
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin', 'compliance_officer')
    );

CREATE POLICY tenant_isolation ON security_ops.training_completions
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    );

CREATE POLICY tenant_isolation ON security_ops.phishing_campaigns
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    );

CREATE POLICY tenant_isolation ON security_ops.phishing_results
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    );

CREATE POLICY tenant_isolation ON security_ops.compliance_scores
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin', 'compliance_officer')
    );

-- -----------------------------------------------------------------------------
-- 28. Triggers - Auto-update Timestamps
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION security_ops.update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_incidents_update
    BEFORE UPDATE ON security_ops.incidents
    FOR EACH ROW EXECUTE FUNCTION security_ops.update_timestamp();

CREATE TRIGGER trg_threat_models_update
    BEFORE UPDATE ON security_ops.threat_models
    FOR EACH ROW EXECUTE FUNCTION security_ops.update_timestamp();

CREATE TRIGGER trg_waf_rules_update
    BEFORE UPDATE ON security_ops.waf_rules
    FOR EACH ROW EXECUTE FUNCTION security_ops.update_timestamp();

CREATE TRIGGER trg_vdp_update
    BEFORE UPDATE ON security_ops.vulnerability_disclosures
    FOR EACH ROW EXECUTE FUNCTION security_ops.update_timestamp();

CREATE TRIGGER trg_dsar_update
    BEFORE UPDATE ON security_ops.dsar_requests
    FOR EACH ROW EXECUTE FUNCTION security_ops.update_timestamp();

CREATE TRIGGER trg_phishing_campaigns_update
    BEFORE UPDATE ON security_ops.phishing_campaigns
    FOR EACH ROW EXECUTE FUNCTION security_ops.update_timestamp();

-- -----------------------------------------------------------------------------
-- 29. Continuous Aggregates for Metrics
-- -----------------------------------------------------------------------------

-- Incident metrics by hour
CREATE MATERIALIZED VIEW security_ops.incident_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', detected_at) AS bucket,
    severity,
    status,
    incident_type,
    source,
    COUNT(*) AS incident_count,
    AVG(EXTRACT(EPOCH FROM (acknowledged_at - detected_at))) AS avg_mttd_seconds,
    AVG(EXTRACT(EPOCH FROM (resolved_at - detected_at))) AS avg_mttr_seconds
FROM security_ops.incidents
WHERE acknowledged_at IS NOT NULL OR resolved_at IS NOT NULL
GROUP BY bucket, severity, status, incident_type, source
WITH NO DATA;

-- Refresh policy for incident metrics
SELECT add_continuous_aggregate_policy('security_ops.incident_metrics_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Alert metrics by hour
CREATE MATERIALIZED VIEW security_ops.alert_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', received_at) AS bucket,
    alert_source,
    alert_type,
    severity,
    COUNT(*) AS alert_count,
    COUNT(DISTINCT incident_id) AS incidents_created
FROM security_ops.alerts
GROUP BY bucket, alert_source, alert_type, severity
WITH NO DATA;

-- Refresh policy for alert metrics
SELECT add_continuous_aggregate_policy('security_ops.alert_metrics_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '30 minutes',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- 30. Verification
-- -----------------------------------------------------------------------------

DO $$
DECLARE
    tbl_name TEXT;
    required_tables TEXT[] := ARRAY[
        'incidents',
        'alerts',
        'playbook_executions',
        'threat_models',
        'waf_rules',
        'vulnerability_disclosures',
        'dsar_requests',
        'consent_records',
        'training_completions',
        'phishing_campaigns',
        'phishing_results',
        'compliance_scores'
    ];
    perm_count INTEGER;
    hypertable_count INTEGER;
BEGIN
    -- Verify all tables exist
    FOREACH tbl_name IN ARRAY required_tables
    LOOP
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'security_ops' AND table_name = tbl_name
        ) THEN
            RAISE EXCEPTION 'Required table security_ops.% was not created', tbl_name;
        ELSE
            RAISE NOTICE 'Table security_ops.% created successfully', tbl_name;
        END IF;
    END LOOP;

    -- Verify RLS is enabled on incidents
    IF NOT EXISTS (
        SELECT 1 FROM pg_tables
        WHERE schemaname = 'security_ops'
          AND tablename = 'incidents'
          AND rowsecurity = TRUE
    ) THEN
        RAISE EXCEPTION 'RLS not enabled on security_ops.incidents';
    END IF;

    -- Verify permissions were inserted
    SELECT COUNT(*) INTO perm_count
    FROM security.permissions
    WHERE name LIKE 'secops:%';

    IF perm_count < 16 THEN
        RAISE EXCEPTION 'Expected 16 secops permissions, found %', perm_count;
    END IF;

    -- Verify hypertables were created
    SELECT COUNT(*) INTO hypertable_count
    FROM timescaledb_information.hypertables
    WHERE hypertable_schema = 'security_ops';

    IF hypertable_count < 2 THEN
        RAISE EXCEPTION 'Expected 2 hypertables, found %', hypertable_count;
    END IF;

    RAISE NOTICE 'V017 Security Operations migration completed successfully';
    RAISE NOTICE 'Created 12 tables, 16 permissions, 2 hypertables, 2 continuous aggregates';
END $$;
