-- =============================================================================
-- V015: Security Scanning Infrastructure
-- =============================================================================
-- Description: Creates security scanning tables for vulnerability management,
--              findings tracking, remediation SLA, risk exceptions, and scan runs
--              for the GreenLang Climate OS security scanning pipeline.
-- Author: GreenLang Infrastructure Team
-- PRD: SEC-007 Security Scanning Pipeline
-- Requires: TimescaleDB (V002), uuid-ossp (V001)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Schema Setup
-- -----------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS security;

SET search_path TO security, public;

-- -----------------------------------------------------------------------------
-- 1. Vulnerabilities Table
-- -----------------------------------------------------------------------------
-- Central registry of discovered vulnerabilities, deduplicated by CVE.
-- Tracks lifecycle from discovery through remediation.

CREATE TABLE security.vulnerabilities (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- CVE Identification
    cve VARCHAR(50),
    cwe VARCHAR(50),
    ghsa VARCHAR(50),

    -- Vulnerability Details
    title VARCHAR(512) NOT NULL,
    description TEXT,
    severity VARCHAR(20) NOT NULL,

    -- Scoring
    cvss_score DECIMAL(3,1),
    cvss_vector VARCHAR(255),
    epss_score DECIMAL(5,4),
    epss_percentile DECIMAL(5,4),

    -- Risk Assessment
    risk_score DECIMAL(4,2) NOT NULL DEFAULT 0,
    is_exploited BOOLEAN NOT NULL DEFAULT FALSE,
    is_kev BOOLEAN NOT NULL DEFAULT FALSE,

    -- Lifecycle
    status VARCHAR(30) NOT NULL DEFAULT 'open',
    discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    sla_due_at TIMESTAMPTZ,

    -- Affected Component
    package_name VARCHAR(255),
    package_version VARCHAR(100),
    fixed_version VARCHAR(100),
    component_type VARCHAR(50),

    -- Asset Context
    asset_type VARCHAR(50),
    asset_criticality VARCHAR(20) DEFAULT 'medium',

    -- Tenant Isolation
    tenant_id UUID,

    -- References
    references JSONB NOT NULL DEFAULT '[]'::jsonb,
    remediation_guidance TEXT,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_vuln_severity CHECK (
        severity IN ('critical', 'high', 'medium', 'low', 'info')
    ),

    CONSTRAINT chk_vuln_status CHECK (
        status IN (
            'open',
            'in_progress',
            'resolved',
            'accepted',
            'false_positive',
            'wont_fix'
        )
    ),

    CONSTRAINT chk_vuln_cvss CHECK (
        cvss_score IS NULL OR (cvss_score >= 0 AND cvss_score <= 10)
    ),

    CONSTRAINT chk_vuln_epss CHECK (
        epss_score IS NULL OR (epss_score >= 0 AND epss_score <= 1)
    ),

    CONSTRAINT chk_vuln_risk CHECK (
        risk_score >= 0 AND risk_score <= 10
    ),

    CONSTRAINT chk_vuln_asset_criticality CHECK (
        asset_criticality IN ('critical', 'high', 'medium', 'low')
    )
);

-- -----------------------------------------------------------------------------
-- 2. Findings Table
-- -----------------------------------------------------------------------------
-- Individual scanner findings linked to vulnerabilities.
-- Multiple findings can map to a single vulnerability (deduplication).

CREATE TABLE security.findings (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Link to Vulnerability
    vulnerability_id UUID
        REFERENCES security.vulnerabilities(id) ON DELETE SET NULL,

    -- Scanner Source
    scanner VARCHAR(50) NOT NULL,
    scan_run_id UUID,

    -- Finding Details
    finding_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    confidence VARCHAR(20) NOT NULL DEFAULT 'high',

    -- Location
    file_path VARCHAR(1024),
    line_start INTEGER,
    line_end INTEGER,
    code_snippet TEXT,

    -- Identification
    fingerprint VARCHAR(128),
    rule_id VARCHAR(255),
    cve VARCHAR(50),
    cwe VARCHAR(50),

    -- Context
    message TEXT NOT NULL,

    -- Raw Data
    raw_data JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Tenant Isolation
    tenant_id UUID,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_finding_severity CHECK (
        severity IN ('critical', 'high', 'medium', 'low', 'info')
    ),

    CONSTRAINT chk_finding_confidence CHECK (
        confidence IN ('high', 'medium', 'low')
    ),

    CONSTRAINT chk_finding_type CHECK (
        finding_type IN (
            'sast',
            'sca',
            'dast',
            'secret',
            'container',
            'iac',
            'pii',
            'license',
            'sbom'
        )
    ),

    CONSTRAINT chk_finding_scanner CHECK (
        scanner IN (
            'bandit',
            'semgrep',
            'codeql',
            'trivy',
            'snyk',
            'pip_audit',
            'safety',
            'gitleaks',
            'trufflehog',
            'detect_secrets',
            'grype',
            'cosign',
            'tfsec',
            'checkov',
            'kubeconform',
            'owasp_zap',
            'presidio',
            'pii_scanner',
            'pip_licenses',
            'cyclonedx',
            'syft'
        )
    )
);

-- -----------------------------------------------------------------------------
-- 3. Remediation SLA Table
-- -----------------------------------------------------------------------------
-- Defines remediation SLA targets by severity.
-- Used to calculate sla_due_at on vulnerabilities.

CREATE TABLE security.remediation_sla (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Severity Level
    severity VARCHAR(20) NOT NULL UNIQUE,

    -- SLA Configuration
    max_days INTEGER NOT NULL,
    warning_days INTEGER NOT NULL,

    -- Notification Settings
    notify_on_breach BOOLEAN NOT NULL DEFAULT TRUE,
    notify_on_warning BOOLEAN NOT NULL DEFAULT TRUE,
    escalation_contacts TEXT[],

    -- Metadata
    description TEXT,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_sla_severity CHECK (
        severity IN ('critical', 'high', 'medium', 'low', 'info')
    ),

    CONSTRAINT chk_sla_days CHECK (
        max_days > 0 AND warning_days > 0 AND warning_days <= max_days
    )
);

-- Insert default SLA values per PRD
INSERT INTO security.remediation_sla (severity, max_days, warning_days, description)
VALUES
    ('critical', 1, 1, 'Critical vulnerabilities must be remediated within 24 hours'),
    ('high', 7, 5, 'High severity vulnerabilities: 7-day SLA'),
    ('medium', 30, 21, 'Medium severity vulnerabilities: 30-day SLA'),
    ('low', 90, 60, 'Low severity vulnerabilities: 90-day SLA'),
    ('info', 365, 270, 'Informational findings: best effort within 1 year')
ON CONFLICT (severity) DO NOTHING;

-- -----------------------------------------------------------------------------
-- 4. Exceptions Table (Risk Acceptance)
-- -----------------------------------------------------------------------------
-- Records risk acceptance decisions for vulnerabilities that won't be fixed.

CREATE TABLE security.exceptions (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Link to Vulnerability
    vulnerability_id UUID NOT NULL
        REFERENCES security.vulnerabilities(id) ON DELETE CASCADE,

    -- Exception Details
    exception_type VARCHAR(30) NOT NULL DEFAULT 'risk_acceptance',
    reason TEXT NOT NULL,
    business_justification TEXT,

    -- Compensating Controls
    compensating_controls TEXT,

    -- Validity
    expires_at TIMESTAMPTZ,
    is_permanent BOOLEAN NOT NULL DEFAULT FALSE,

    -- Approval Chain
    approved_by VARCHAR(255) NOT NULL,
    approved_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    approver_role VARCHAR(100),

    -- Review Tracking
    last_reviewed_at TIMESTAMPTZ,
    next_review_at TIMESTAMPTZ,
    review_notes TEXT,

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    revoked_at TIMESTAMPTZ,
    revoked_by VARCHAR(255),
    revoke_reason TEXT,

    -- Tenant Isolation
    tenant_id UUID,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_exception_type CHECK (
        exception_type IN (
            'risk_acceptance',
            'false_positive',
            'wont_fix',
            'deferred',
            'compensating_control'
        )
    ),

    CONSTRAINT chk_exception_status CHECK (
        status IN ('active', 'expired', 'revoked', 'pending_review')
    ),

    CONSTRAINT chk_exception_expiry CHECK (
        is_permanent = TRUE OR expires_at IS NOT NULL
    )
);

-- -----------------------------------------------------------------------------
-- 5. Scan Runs Table
-- -----------------------------------------------------------------------------
-- Tracks individual scan executions with timing and findings count.

CREATE TABLE security.scan_runs (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Scanner Identification
    scanner VARCHAR(50) NOT NULL,
    scanner_version VARCHAR(50),

    -- Target Information
    target_type VARCHAR(50) NOT NULL,
    target_path VARCHAR(1024),
    target_ref VARCHAR(255),
    commit_sha VARCHAR(64),
    branch VARCHAR(255),

    -- Execution Context
    trigger_type VARCHAR(30) NOT NULL DEFAULT 'manual',
    triggered_by VARCHAR(255),
    job_id VARCHAR(255),
    workflow_run_id VARCHAR(255),

    -- Timing
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    duration_seconds DECIMAL(10,2),

    -- Results Summary
    status VARCHAR(20) NOT NULL DEFAULT 'running',
    exit_code INTEGER,

    -- Finding Counts
    findings_total INTEGER NOT NULL DEFAULT 0,
    findings_critical INTEGER NOT NULL DEFAULT 0,
    findings_high INTEGER NOT NULL DEFAULT 0,
    findings_medium INTEGER NOT NULL DEFAULT 0,
    findings_low INTEGER NOT NULL DEFAULT 0,
    findings_info INTEGER NOT NULL DEFAULT 0,

    -- Deduplication Stats
    findings_new INTEGER NOT NULL DEFAULT 0,
    findings_existing INTEGER NOT NULL DEFAULT 0,
    findings_resolved INTEGER NOT NULL DEFAULT 0,

    -- Output
    report_url VARCHAR(1024),
    sarif_url VARCHAR(1024),
    log_url VARCHAR(1024),

    -- Error Handling
    error_message TEXT,
    error_details JSONB,

    -- Configuration
    scan_config JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Tenant Isolation
    tenant_id UUID,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_scan_status CHECK (
        status IN (
            'pending',
            'running',
            'completed',
            'failed',
            'cancelled',
            'timeout'
        )
    ),

    CONSTRAINT chk_scan_target_type CHECK (
        target_type IN (
            'repository',
            'branch',
            'commit',
            'pull_request',
            'image',
            'manifest',
            'url',
            'api',
            'infrastructure'
        )
    ),

    CONSTRAINT chk_scan_trigger CHECK (
        trigger_type IN (
            'manual',
            'push',
            'pull_request',
            'schedule',
            'api',
            'deployment'
        )
    )
);

-- -----------------------------------------------------------------------------
-- 6. PII Findings Table
-- -----------------------------------------------------------------------------
-- Specialized table for PII/sensitive data findings.

CREATE TABLE security.pii_findings (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Link to General Finding
    finding_id UUID
        REFERENCES security.findings(id) ON DELETE SET NULL,

    -- PII Classification
    data_classification VARCHAR(20) NOT NULL,
    pii_type VARCHAR(50) NOT NULL,

    -- Detection Details
    pattern_name VARCHAR(100),
    confidence_score DECIMAL(3,2) NOT NULL,
    detection_method VARCHAR(30) NOT NULL,

    -- Location
    file_path VARCHAR(1024),
    line_number INTEGER,
    column_start INTEGER,
    column_end INTEGER,

    -- Context (redacted)
    context_before TEXT,
    context_after TEXT,
    matched_text_hash VARCHAR(64),

    -- Risk Assessment
    exposure_risk VARCHAR(20) NOT NULL DEFAULT 'medium',
    data_subject_count INTEGER,

    -- Remediation
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    remediated_at TIMESTAMPTZ,
    remediated_by VARCHAR(255),

    -- Routing
    routed_to VARCHAR(255),
    routed_at TIMESTAMPTZ,

    -- Tenant Isolation
    tenant_id UUID,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_pii_classification CHECK (
        data_classification IN ('pii', 'phi', 'pci', 'secret', 'internal')
    ),

    CONSTRAINT chk_pii_type CHECK (
        pii_type IN (
            'ssn',
            'credit_card',
            'email',
            'phone',
            'address',
            'name',
            'dob',
            'passport',
            'driver_license',
            'ip_address',
            'medical_record',
            'financial_account',
            'api_key',
            'password',
            'token',
            'tenant_id',
            'emission_data',
            'other'
        )
    ),

    CONSTRAINT chk_pii_method CHECK (
        detection_method IN ('regex', 'ml', 'presidio', 'hybrid')
    ),

    CONSTRAINT chk_pii_confidence CHECK (
        confidence_score >= 0 AND confidence_score <= 1
    ),

    CONSTRAINT chk_pii_exposure CHECK (
        exposure_risk IN ('critical', 'high', 'medium', 'low')
    ),

    CONSTRAINT chk_pii_status CHECK (
        status IN ('open', 'remediated', 'accepted', 'false_positive')
    )
);

-- -----------------------------------------------------------------------------
-- 7. Indexes - Vulnerabilities
-- -----------------------------------------------------------------------------

-- CVE lookup (most common query)
CREATE INDEX idx_vuln_cve ON security.vulnerabilities (cve)
    WHERE cve IS NOT NULL;

-- Severity filtering
CREATE INDEX idx_vuln_severity ON security.vulnerabilities (severity, status);

-- Status filtering (for dashboards)
CREATE INDEX idx_vuln_status ON security.vulnerabilities (status, discovered_at DESC);

-- Open vulnerabilities (partial index for hot path)
CREATE INDEX idx_vuln_open ON security.vulnerabilities (severity, sla_due_at)
    WHERE status = 'open';

-- Tenant isolation
CREATE INDEX idx_vuln_tenant ON security.vulnerabilities (tenant_id, status)
    WHERE tenant_id IS NOT NULL;

-- Package lookup
CREATE INDEX idx_vuln_package ON security.vulnerabilities (package_name, package_version)
    WHERE package_name IS NOT NULL;

-- EPSS score ranking (for prioritization)
CREATE INDEX idx_vuln_epss ON security.vulnerabilities (epss_score DESC NULLS LAST)
    WHERE status = 'open';

-- KEV vulnerabilities (high priority)
CREATE INDEX idx_vuln_kev ON security.vulnerabilities (discovered_at DESC)
    WHERE is_kev = TRUE AND status = 'open';

-- SLA breach detection
CREATE INDEX idx_vuln_sla_due ON security.vulnerabilities (sla_due_at)
    WHERE status = 'open' AND sla_due_at IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 8. Indexes - Findings
-- -----------------------------------------------------------------------------

-- Scanner lookup
CREATE INDEX idx_finding_scanner ON security.findings (scanner, created_at DESC);

-- Vulnerability link
CREATE INDEX idx_finding_vuln ON security.findings (vulnerability_id)
    WHERE vulnerability_id IS NOT NULL;

-- Scan run link
CREATE INDEX idx_finding_scan_run ON security.findings (scan_run_id)
    WHERE scan_run_id IS NOT NULL;

-- Fingerprint deduplication
CREATE UNIQUE INDEX idx_finding_fingerprint ON security.findings (fingerprint)
    WHERE fingerprint IS NOT NULL;

-- Severity filtering
CREATE INDEX idx_finding_severity ON security.findings (severity, finding_type);

-- CVE lookup
CREATE INDEX idx_finding_cve ON security.findings (cve)
    WHERE cve IS NOT NULL;

-- File path lookup
CREATE INDEX idx_finding_file ON security.findings (file_path)
    WHERE file_path IS NOT NULL;

-- Tenant isolation
CREATE INDEX idx_finding_tenant ON security.findings (tenant_id, scanner)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 9. Indexes - Exceptions
-- -----------------------------------------------------------------------------

-- Vulnerability lookup
CREATE INDEX idx_exception_vuln ON security.exceptions (vulnerability_id, status);

-- Active exceptions
CREATE INDEX idx_exception_active ON security.exceptions (expires_at)
    WHERE status = 'active' AND is_permanent = FALSE;

-- Approver audit
CREATE INDEX idx_exception_approver ON security.exceptions (approved_by, approved_at DESC);

-- Review scheduling
CREATE INDEX idx_exception_review ON security.exceptions (next_review_at)
    WHERE status = 'active' AND next_review_at IS NOT NULL;

-- Tenant isolation
CREATE INDEX idx_exception_tenant ON security.exceptions (tenant_id, status)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 10. Indexes - Scan Runs
-- -----------------------------------------------------------------------------

-- Scanner lookup
CREATE INDEX idx_scan_scanner ON security.scan_runs (scanner, started_at DESC);

-- Status filtering
CREATE INDEX idx_scan_status ON security.scan_runs (status, scanner);

-- Running scans
CREATE INDEX idx_scan_running ON security.scan_runs (started_at DESC)
    WHERE status = 'running';

-- Completed scans by time
CREATE INDEX idx_scan_completed ON security.scan_runs (completed_at DESC)
    WHERE status = 'completed';

-- Target lookup
CREATE INDEX idx_scan_target ON security.scan_runs (target_type, target_path);

-- Commit lookup
CREATE INDEX idx_scan_commit ON security.scan_runs (commit_sha)
    WHERE commit_sha IS NOT NULL;

-- Tenant isolation
CREATE INDEX idx_scan_tenant ON security.scan_runs (tenant_id, scanner)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 11. Indexes - PII Findings
-- -----------------------------------------------------------------------------

-- Classification lookup
CREATE INDEX idx_pii_classification ON security.pii_findings (data_classification, pii_type);

-- Open findings
CREATE INDEX idx_pii_open ON security.pii_findings (data_classification, created_at DESC)
    WHERE status = 'open';

-- File lookup
CREATE INDEX idx_pii_file ON security.pii_findings (file_path)
    WHERE file_path IS NOT NULL;

-- Routing
CREATE INDEX idx_pii_routed ON security.pii_findings (routed_to, routed_at)
    WHERE routed_to IS NOT NULL;

-- Tenant isolation
CREATE INDEX idx_pii_tenant ON security.pii_findings (tenant_id, data_classification)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 12. Row-Level Security
-- -----------------------------------------------------------------------------

-- Enable RLS on all tables
ALTER TABLE security.vulnerabilities ENABLE ROW LEVEL SECURITY;
ALTER TABLE security.findings ENABLE ROW LEVEL SECURITY;
ALTER TABLE security.exceptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE security.scan_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE security.pii_findings ENABLE ROW LEVEL SECURITY;

-- Vulnerabilities: tenant isolation
CREATE POLICY vuln_tenant_isolation ON security.vulnerabilities
    FOR ALL
    USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    )
    WITH CHECK (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    );

-- Findings: tenant isolation
CREATE POLICY finding_tenant_isolation ON security.findings
    FOR ALL
    USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    )
    WITH CHECK (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    );

-- Exceptions: tenant isolation + security admin required for write
CREATE POLICY exception_read_isolation ON security.exceptions
    FOR SELECT
    USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    );

CREATE POLICY exception_write_isolation ON security.exceptions
    FOR ALL
    USING (
        NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    )
    WITH CHECK (
        NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    );

-- Scan runs: tenant isolation
CREATE POLICY scan_tenant_isolation ON security.scan_runs
    FOR ALL
    USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    )
    WITH CHECK (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    );

-- PII findings: restricted to security team and data stewards
CREATE POLICY pii_read_isolation ON security.pii_findings
    FOR SELECT
    USING (
        NULLIF(current_setting('app.user_role', true), '') IN (
            'admin', 'security_admin', 'data_steward', 'compliance_officer'
        )
        OR (
            tenant_id IS NOT NULL
            AND tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
            AND NULLIF(current_setting('app.user_role', true), '') = 'tenant_admin'
        )
    );

CREATE POLICY pii_write_isolation ON security.pii_findings
    FOR ALL
    USING (
        NULLIF(current_setting('app.user_role', true), '') IN (
            'admin', 'security_admin', 'data_steward'
        )
    )
    WITH CHECK (
        NULLIF(current_setting('app.user_role', true), '') IN (
            'admin', 'security_admin', 'data_steward'
        )
    );

-- -----------------------------------------------------------------------------
-- 13. Triggers - Auto-update timestamps
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION security.update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_vulnerabilities_update
    BEFORE UPDATE ON security.vulnerabilities
    FOR EACH ROW EXECUTE FUNCTION security.update_timestamp();

CREATE TRIGGER trg_findings_update
    BEFORE UPDATE ON security.findings
    FOR EACH ROW EXECUTE FUNCTION security.update_timestamp();

CREATE TRIGGER trg_exceptions_update
    BEFORE UPDATE ON security.exceptions
    FOR EACH ROW EXECUTE FUNCTION security.update_timestamp();

CREATE TRIGGER trg_scan_runs_update
    BEFORE UPDATE ON security.scan_runs
    FOR EACH ROW EXECUTE FUNCTION security.update_timestamp();

CREATE TRIGGER trg_pii_findings_update
    BEFORE UPDATE ON security.pii_findings
    FOR EACH ROW EXECUTE FUNCTION security.update_timestamp();

-- -----------------------------------------------------------------------------
-- 14. Trigger - Auto-calculate SLA due date
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION security.calculate_sla_due()
RETURNS TRIGGER AS $$
DECLARE
    sla_days INTEGER;
BEGIN
    -- Only calculate on insert or when severity changes
    IF TG_OP = 'INSERT' OR OLD.severity != NEW.severity THEN
        SELECT max_days INTO sla_days
        FROM security.remediation_sla
        WHERE severity = NEW.severity;

        IF sla_days IS NOT NULL THEN
            NEW.sla_due_at = NEW.discovered_at + (sla_days || ' days')::INTERVAL;
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_vulnerability_sla
    BEFORE INSERT OR UPDATE ON security.vulnerabilities
    FOR EACH ROW EXECUTE FUNCTION security.calculate_sla_due();

-- -----------------------------------------------------------------------------
-- 15. Trigger - Auto-expire exceptions
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION security.check_exception_expiry()
RETURNS TRIGGER AS $$
BEGIN
    -- Mark expired exceptions
    IF NEW.status = 'active'
       AND NOT NEW.is_permanent
       AND NEW.expires_at IS NOT NULL
       AND NEW.expires_at < NOW() THEN
        NEW.status = 'expired';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_exception_expiry
    BEFORE UPDATE ON security.exceptions
    FOR EACH ROW EXECUTE FUNCTION security.check_exception_expiry();

-- -----------------------------------------------------------------------------
-- 16. Table Comments
-- -----------------------------------------------------------------------------

COMMENT ON TABLE security.vulnerabilities IS
    'Central registry of discovered vulnerabilities. Each unique CVE/finding '
    'is tracked from discovery through remediation with full lifecycle '
    'management and risk scoring (CVSS + EPSS + KEV + asset criticality).';

COMMENT ON TABLE security.findings IS
    'Individual scanner findings from SAST, SCA, DAST, secrets, container, '
    'and IaC scanners. Multiple findings can map to a single vulnerability. '
    'Includes location, fingerprint for deduplication, and raw scanner data.';

COMMENT ON TABLE security.remediation_sla IS
    'SLA configuration by severity level. Default SLAs: Critical=24h, '
    'High=7d, Medium=30d, Low=90d. Used to auto-calculate sla_due_at on '
    'vulnerabilities.';

COMMENT ON TABLE security.exceptions IS
    'Risk acceptance and exception records for vulnerabilities that are '
    'accepted, deferred, or have compensating controls. Requires security '
    'admin approval and periodic review.';

COMMENT ON TABLE security.scan_runs IS
    'Audit log of all security scan executions. Tracks scanner, target, '
    'timing, findings counts by severity, and output locations. Used for '
    'scanner health monitoring and trend analysis.';

COMMENT ON TABLE security.pii_findings IS
    'Specialized table for PII/sensitive data detection findings. Includes '
    'data classification (PII/PHI/PCI), detection method (regex/ML/Presidio), '
    'confidence score, and routing for data steward alerts.';

-- -----------------------------------------------------------------------------
-- 17. Verification
-- -----------------------------------------------------------------------------

DO $$
DECLARE
    tbl_name TEXT;
    required_tables TEXT[] := ARRAY[
        'vulnerabilities',
        'findings',
        'remediation_sla',
        'exceptions',
        'scan_runs',
        'pii_findings'
    ];
BEGIN
    FOREACH tbl_name IN ARRAY required_tables
    LOOP
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'security' AND table_name = tbl_name
        ) THEN
            RAISE EXCEPTION 'Required table security.% was not created', tbl_name;
        ELSE
            RAISE NOTICE 'Table security.% created successfully', tbl_name;
        END IF;
    END LOOP;

    -- Verify RLS is enabled
    IF NOT EXISTS (
        SELECT 1 FROM pg_tables
        WHERE schemaname = 'security'
          AND tablename = 'vulnerabilities'
          AND rowsecurity = TRUE
    ) THEN
        RAISE EXCEPTION 'RLS not enabled on security.vulnerabilities';
    END IF;

    -- Verify default SLAs were inserted
    IF NOT EXISTS (
        SELECT 1 FROM security.remediation_sla WHERE severity = 'critical'
    ) THEN
        RAISE EXCEPTION 'Default SLA values not inserted';
    END IF;

    RAISE NOTICE 'V015 Security Scanning migration completed successfully';
END $$;
