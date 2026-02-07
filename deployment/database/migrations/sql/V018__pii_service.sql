-- =============================================================================
-- V018: PII Service Database Schema
-- =============================================================================
-- Description: Creates pii_service schema for PII Detection/Redaction Service
--              including token vault, allowlist, quarantine, remediation items,
--              remediation log, audit log, deletion certificates, and
--              enforcement policies with TimescaleDB hypertables.
-- Author: GreenLang Security Team
-- PRD: SEC-011 PII Detection/Redaction Enhancements
-- Requires: TimescaleDB (V002), uuid-ossp (V001), security schema (V009)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Schema Setup
-- -----------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS pii_service;

SET search_path TO pii_service, security, public;

-- -----------------------------------------------------------------------------
-- 1. Token Vault Table
-- -----------------------------------------------------------------------------
-- Stores encrypted, reversible tokens for PII values using AES-256-GCM.
-- Tokens are tenant-isolated and have configurable expiration.

CREATE TABLE pii_service.token_vault (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    token_id VARCHAR(64) NOT NULL UNIQUE,
    pii_type VARCHAR(30) NOT NULL,
    original_hash VARCHAR(64) NOT NULL,
    encrypted_value BYTEA NOT NULL,
    tenant_id VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Constraints
    CONSTRAINT chk_token_pii_type CHECK (
        pii_type IN (
            'person_name', 'email', 'phone', 'ssn', 'national_id', 'passport',
            'drivers_license', 'credit_card', 'bank_account', 'iban',
            'medical_record', 'health_insurance_id', 'address', 'ip_address',
            'gps_coordinates', 'username', 'password', 'api_key',
            'organization_name', 'date_of_birth', 'custom'
        )
    ),
    CONSTRAINT chk_token_expires CHECK (expires_at > created_at)
);

CREATE INDEX idx_token_vault_tenant ON pii_service.token_vault(tenant_id);
CREATE INDEX idx_token_vault_expires ON pii_service.token_vault(expires_at);
CREATE INDEX idx_token_vault_pii_type ON pii_service.token_vault(pii_type);
CREATE INDEX idx_token_vault_created ON pii_service.token_vault(created_at DESC);
CREATE INDEX idx_token_vault_hash ON pii_service.token_vault(original_hash);

COMMENT ON TABLE pii_service.token_vault IS
    'Encrypted PII token vault using AES-256-GCM. Tokens are tenant-isolated '
    'and expire after configurable TTL. Supports secure detokenization with '
    'audit trail.';

-- -----------------------------------------------------------------------------
-- 2. Allowlist Table
-- -----------------------------------------------------------------------------
-- Patterns to exclude from PII detection (false positive filtering).
-- Supports regex, exact match, prefix, suffix, and contains patterns.

CREATE TABLE pii_service.allowlist (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pii_type VARCHAR(30) NOT NULL,
    pattern TEXT NOT NULL,
    pattern_type VARCHAR(20) NOT NULL,
    reason TEXT,
    tenant_id VARCHAR(50),  -- NULL = global allowlist
    created_by UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    enabled BOOLEAN DEFAULT true,

    -- Constraints
    CONSTRAINT chk_allowlist_pii_type CHECK (
        pii_type IN (
            'person_name', 'email', 'phone', 'ssn', 'national_id', 'passport',
            'drivers_license', 'credit_card', 'bank_account', 'iban',
            'medical_record', 'health_insurance_id', 'address', 'ip_address',
            'gps_coordinates', 'username', 'password', 'api_key',
            'organization_name', 'date_of_birth', 'custom'
        )
    ),
    CONSTRAINT chk_allowlist_pattern_type CHECK (
        pattern_type IN ('regex', 'exact', 'prefix', 'suffix', 'contains')
    )
);

CREATE INDEX idx_allowlist_pii_type ON pii_service.allowlist(pii_type);
CREATE INDEX idx_allowlist_tenant ON pii_service.allowlist(tenant_id);
CREATE INDEX idx_allowlist_enabled ON pii_service.allowlist(enabled) WHERE enabled = true;
CREATE INDEX idx_allowlist_expires ON pii_service.allowlist(expires_at)
    WHERE expires_at IS NOT NULL;

COMMENT ON TABLE pii_service.allowlist IS
    'PII detection allowlist for filtering false positives. Supports per-tenant '
    'and global patterns with multiple matching strategies.';

-- -----------------------------------------------------------------------------
-- 3. Quarantine Table
-- -----------------------------------------------------------------------------
-- Content quarantined for manual review when PII is detected.

CREATE TABLE pii_service.quarantine (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_hash VARCHAR(64) NOT NULL,
    pii_type VARCHAR(30) NOT NULL,
    detection_confidence DECIMAL(3,2) NOT NULL,
    source_type VARCHAR(30) NOT NULL,
    source_location TEXT,
    tenant_id VARCHAR(50) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    reviewed_by UUID,
    reviewed_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Constraints
    CONSTRAINT chk_quarantine_pii_type CHECK (
        pii_type IN (
            'person_name', 'email', 'phone', 'ssn', 'national_id', 'passport',
            'drivers_license', 'credit_card', 'bank_account', 'iban',
            'medical_record', 'health_insurance_id', 'address', 'ip_address',
            'gps_coordinates', 'username', 'password', 'api_key',
            'organization_name', 'date_of_birth', 'custom'
        )
    ),
    CONSTRAINT chk_quarantine_source_type CHECK (
        source_type IN (
            'api_request', 'api_response', 'storage', 'logging',
            'streaming', 'kafka_stream', 'kinesis_stream', 'batch_processing'
        )
    ),
    CONSTRAINT chk_quarantine_status CHECK (
        status IN ('pending', 'released', 'deleted')
    ),
    CONSTRAINT chk_quarantine_confidence CHECK (
        detection_confidence >= 0 AND detection_confidence <= 1
    )
);

CREATE INDEX idx_quarantine_status ON pii_service.quarantine(status);
CREATE INDEX idx_quarantine_tenant ON pii_service.quarantine(tenant_id);
CREATE INDEX idx_quarantine_expires ON pii_service.quarantine(expires_at);
CREATE INDEX idx_quarantine_pii_type ON pii_service.quarantine(pii_type);
CREATE INDEX idx_quarantine_detected ON pii_service.quarantine(detected_at DESC);

COMMENT ON TABLE pii_service.quarantine IS
    'Quarantine for PII-containing content pending manual review. Items auto-expire '
    'after configurable TTL and can be released or deleted after review.';

-- -----------------------------------------------------------------------------
-- 4. Remediation Items Table
-- -----------------------------------------------------------------------------
-- PII items scheduled for remediation (deletion, anonymization, etc.).

CREATE TABLE pii_service.remediation_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pii_type VARCHAR(30) NOT NULL,
    source_type VARCHAR(30) NOT NULL,
    source_location TEXT NOT NULL,
    record_identifier TEXT NOT NULL,
    tenant_id VARCHAR(50) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    scheduled_for TIMESTAMPTZ NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    approved_by UUID,
    approved_at TIMESTAMPTZ,
    executed_at TIMESTAMPTZ,
    deletion_certificate_id UUID,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Constraints
    CONSTRAINT chk_remediation_pii_type CHECK (
        pii_type IN (
            'person_name', 'email', 'phone', 'ssn', 'national_id', 'passport',
            'drivers_license', 'credit_card', 'bank_account', 'iban',
            'medical_record', 'health_insurance_id', 'address', 'ip_address',
            'gps_coordinates', 'username', 'password', 'api_key',
            'organization_name', 'date_of_birth', 'custom'
        )
    ),
    CONSTRAINT chk_remediation_source_type CHECK (
        source_type IN ('postgresql', 's3', 'redis', 'elasticsearch', 'kafka', 'file')
    ),
    CONSTRAINT chk_remediation_status CHECK (
        status IN ('pending', 'approved', 'executing', 'completed', 'failed', 'cancelled')
    )
);

CREATE INDEX idx_remediation_status ON pii_service.remediation_items(status);
CREATE INDEX idx_remediation_scheduled ON pii_service.remediation_items(scheduled_for);
CREATE INDEX idx_remediation_tenant ON pii_service.remediation_items(tenant_id);
CREATE INDEX idx_remediation_pii_type ON pii_service.remediation_items(pii_type);

COMMENT ON TABLE pii_service.remediation_items IS
    'PII remediation queue for scheduled deletion/anonymization. Tracks approval '
    'workflow and execution status with deletion certificate linkage.';

-- -----------------------------------------------------------------------------
-- 5. Remediation Log Table (TimescaleDB Hypertable)
-- -----------------------------------------------------------------------------
-- Historical log of all remediation actions for audit trail.

CREATE TABLE pii_service.remediation_log (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    pii_type VARCHAR(30) NOT NULL,
    action VARCHAR(20) NOT NULL,
    source_type VARCHAR(30) NOT NULL,
    source_location TEXT,
    tenant_id VARCHAR(50) NOT NULL,
    executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    executed_by VARCHAR(50),  -- 'system' or user_id
    deletion_certificate_id UUID,
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Primary key for hypertable (must include time column)
    PRIMARY KEY (executed_at, id),

    -- Constraints
    CONSTRAINT chk_remediation_log_action CHECK (
        action IN ('delete', 'anonymize', 'archive', 'notify_only')
    )
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable(
    'pii_service.remediation_log',
    'executed_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Set retention policy (7 years for compliance)
SELECT add_retention_policy(
    'pii_service.remediation_log',
    INTERVAL '2555 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_remediation_log_tenant ON pii_service.remediation_log(tenant_id, executed_at DESC);
CREATE INDEX idx_remediation_log_pii_type ON pii_service.remediation_log(pii_type, executed_at DESC);
CREATE INDEX idx_remediation_log_action ON pii_service.remediation_log(action, executed_at DESC);

COMMENT ON TABLE pii_service.remediation_log IS
    'TimescaleDB hypertable for PII remediation audit trail. 7-year retention '
    'for regulatory compliance (GDPR, CCPA).';

-- -----------------------------------------------------------------------------
-- 6. Deletion Certificates Table
-- -----------------------------------------------------------------------------
-- Cryptographic proof of PII deletion for compliance audits.

CREATE TABLE pii_service.deletion_certificates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    remediation_item_id UUID REFERENCES pii_service.remediation_items(id),
    pii_type VARCHAR(30) NOT NULL,
    source_type VARCHAR(30) NOT NULL,
    source_location TEXT,
    deleted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    verification_hash VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(50) NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Constraints
    CONSTRAINT chk_cert_pii_type CHECK (
        pii_type IN (
            'person_name', 'email', 'phone', 'ssn', 'national_id', 'passport',
            'drivers_license', 'credit_card', 'bank_account', 'iban',
            'medical_record', 'health_insurance_id', 'address', 'ip_address',
            'gps_coordinates', 'username', 'password', 'api_key',
            'organization_name', 'date_of_birth', 'custom'
        )
    )
);

CREATE INDEX idx_deletion_cert_tenant ON pii_service.deletion_certificates(tenant_id);
CREATE INDEX idx_deletion_cert_deleted ON pii_service.deletion_certificates(deleted_at DESC);
CREATE INDEX idx_deletion_cert_remediation ON pii_service.deletion_certificates(remediation_item_id);

COMMENT ON TABLE pii_service.deletion_certificates IS
    'GDPR-compliant deletion certificates with cryptographic verification hash. '
    'Provides audit evidence for data subject erasure requests.';

-- -----------------------------------------------------------------------------
-- 7. Audit Log Table (TimescaleDB Hypertable)
-- -----------------------------------------------------------------------------
-- Comprehensive audit trail for all PII service operations.

CREATE TABLE pii_service.audit_log (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    action VARCHAR(30) NOT NULL,
    pii_type VARCHAR(30),
    tenant_id VARCHAR(50) NOT NULL,
    user_id UUID,
    content_hash VARCHAR(64),
    details JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Primary key for hypertable
    PRIMARY KEY (created_at, id),

    -- Constraints
    CONSTRAINT chk_audit_action CHECK (
        action IN (
            'detect', 'redact', 'tokenize', 'detokenize', 'detokenize_denied',
            'quarantine_add', 'quarantine_release', 'quarantine_delete',
            'allowlist_add', 'allowlist_remove', 'policy_update',
            'remediation_schedule', 'remediation_execute', 'remediation_fail',
            'enforcement_block', 'enforcement_redact', 'enforcement_allow'
        )
    )
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable(
    'pii_service.audit_log',
    'created_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Set retention policy (365 days for operational audit)
SELECT add_retention_policy(
    'pii_service.audit_log',
    INTERVAL '365 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_audit_log_tenant ON pii_service.audit_log(tenant_id, created_at DESC);
CREATE INDEX idx_audit_log_action ON pii_service.audit_log(action, created_at DESC);
CREATE INDEX idx_audit_log_user ON pii_service.audit_log(user_id, created_at DESC)
    WHERE user_id IS NOT NULL;
CREATE INDEX idx_audit_log_pii_type ON pii_service.audit_log(pii_type, created_at DESC)
    WHERE pii_type IS NOT NULL;

COMMENT ON TABLE pii_service.audit_log IS
    'TimescaleDB hypertable for PII service audit trail. Tracks all operations '
    'for security monitoring and compliance reporting.';

-- -----------------------------------------------------------------------------
-- 8. Enforcement Policies Table
-- -----------------------------------------------------------------------------
-- Per-tenant enforcement policy overrides.

CREATE TABLE pii_service.enforcement_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pii_type VARCHAR(30) NOT NULL,
    tenant_id VARCHAR(50) NOT NULL,
    action VARCHAR(20) NOT NULL,
    min_confidence DECIMAL(3,2) NOT NULL DEFAULT 0.8,
    contexts TEXT[] NOT NULL DEFAULT '{*}',
    notify BOOLEAN DEFAULT true,
    quarantine_ttl_hours INTEGER DEFAULT 72,
    custom_placeholder TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Unique constraint on pii_type + tenant
    CONSTRAINT uq_enforcement_policy UNIQUE (pii_type, tenant_id),

    -- Constraints
    CONSTRAINT chk_enforcement_pii_type CHECK (
        pii_type IN (
            'person_name', 'email', 'phone', 'ssn', 'national_id', 'passport',
            'drivers_license', 'credit_card', 'bank_account', 'iban',
            'medical_record', 'health_insurance_id', 'address', 'ip_address',
            'gps_coordinates', 'username', 'password', 'api_key',
            'organization_name', 'date_of_birth', 'custom'
        )
    ),
    CONSTRAINT chk_enforcement_action CHECK (
        action IN ('allow', 'redact', 'block', 'quarantine', 'transform')
    ),
    CONSTRAINT chk_enforcement_confidence CHECK (
        min_confidence >= 0 AND min_confidence <= 1
    ),
    CONSTRAINT chk_enforcement_ttl CHECK (
        quarantine_ttl_hours >= 1 AND quarantine_ttl_hours <= 8760
    )
);

CREATE INDEX idx_enforcement_tenant ON pii_service.enforcement_policies(tenant_id);
CREATE INDEX idx_enforcement_pii_type ON pii_service.enforcement_policies(pii_type);

COMMENT ON TABLE pii_service.enforcement_policies IS
    'Per-tenant enforcement policy overrides. Allows tenants to customize '
    'PII handling behavior beyond system defaults.';

-- -----------------------------------------------------------------------------
-- 9. PII Service Permissions
-- -----------------------------------------------------------------------------
-- Insert 11 permissions for PII service operations.

INSERT INTO security.permissions (name, resource, action, description) VALUES
    ('pii:detect', 'pii', 'detect', 'Detect PII in content'),
    ('pii:redact', 'pii', 'redact', 'Redact PII from content'),
    ('pii:tokenize', 'pii', 'tokenize', 'Create PII tokens'),
    ('pii:detokenize', 'pii', 'detokenize', 'Retrieve original from tokens'),
    ('pii:policies:read', 'pii_policies', 'read', 'View PII enforcement policies'),
    ('pii:policies:write', 'pii_policies', 'write', 'Modify PII enforcement policies'),
    ('pii:allowlist:read', 'pii_allowlist', 'read', 'View PII allowlist entries'),
    ('pii:allowlist:write', 'pii_allowlist', 'write', 'Modify PII allowlist entries'),
    ('pii:quarantine:read', 'pii_quarantine', 'read', 'View quarantined items'),
    ('pii:quarantine:manage', 'pii_quarantine', 'manage', 'Release or delete quarantined items'),
    ('pii:audit:read', 'pii_audit', 'read', 'View PII audit logs and metrics')
ON CONFLICT (name) DO NOTHING;

-- -----------------------------------------------------------------------------
-- 10. Role-Permission Mappings
-- -----------------------------------------------------------------------------
-- Grant all PII permissions to security_admin role.

INSERT INTO security.role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM security.roles r, security.permissions p
WHERE r.name = 'security_admin'
  AND p.name LIKE 'pii:%'
ON CONFLICT DO NOTHING;

-- Grant read and operational permissions to compliance_officer role
INSERT INTO security.role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM security.roles r, security.permissions p
WHERE r.name = 'compliance_officer'
  AND p.name IN (
      'pii:detect',
      'pii:redact',
      'pii:policies:read',
      'pii:allowlist:read',
      'pii:quarantine:read',
      'pii:audit:read'
  )
ON CONFLICT DO NOTHING;

-- Grant basic PII detection to data_steward role (if exists)
INSERT INTO security.role_permissions (role_id, permission_id)
SELECT r.id, p.id
FROM security.roles r, security.permissions p
WHERE r.name = 'data_steward'
  AND p.name IN (
      'pii:detect',
      'pii:redact',
      'pii:quarantine:read',
      'pii:quarantine:manage'
  )
ON CONFLICT DO NOTHING;

-- -----------------------------------------------------------------------------
-- 11. Row-Level Security
-- -----------------------------------------------------------------------------

ALTER TABLE pii_service.token_vault ENABLE ROW LEVEL SECURITY;
ALTER TABLE pii_service.allowlist ENABLE ROW LEVEL SECURITY;
ALTER TABLE pii_service.quarantine ENABLE ROW LEVEL SECURITY;
ALTER TABLE pii_service.remediation_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE pii_service.remediation_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE pii_service.deletion_certificates ENABLE ROW LEVEL SECURITY;
ALTER TABLE pii_service.audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE pii_service.enforcement_policies ENABLE ROW LEVEL SECURITY;

-- Token vault: strict tenant isolation
CREATE POLICY tenant_isolation ON pii_service.token_vault
    FOR ALL USING (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    );

-- Allowlist: allow global entries (tenant_id IS NULL) plus tenant-specific
CREATE POLICY tenant_isolation ON pii_service.allowlist
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    );

-- Quarantine: tenant isolation
CREATE POLICY tenant_isolation ON pii_service.quarantine
    FOR ALL USING (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin', 'compliance_officer')
    );

-- Remediation items: tenant isolation
CREATE POLICY tenant_isolation ON pii_service.remediation_items
    FOR ALL USING (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin', 'compliance_officer')
    );

-- Remediation log: tenant isolation
CREATE POLICY tenant_isolation ON pii_service.remediation_log
    FOR ALL USING (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin', 'compliance_officer')
    );

-- Deletion certificates: tenant isolation
CREATE POLICY tenant_isolation ON pii_service.deletion_certificates
    FOR ALL USING (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin', 'compliance_officer')
    );

-- Audit log: tenant isolation
CREATE POLICY tenant_isolation ON pii_service.audit_log
    FOR ALL USING (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    );

-- Enforcement policies: tenant isolation
CREATE POLICY tenant_isolation ON pii_service.enforcement_policies
    FOR ALL USING (
        tenant_id = NULLIF(current_setting('app.tenant_id', true), '')
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'security_admin')
    );

-- -----------------------------------------------------------------------------
-- 12. Triggers - Auto-update Timestamps
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION pii_service.update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_enforcement_policies_update
    BEFORE UPDATE ON pii_service.enforcement_policies
    FOR EACH ROW EXECUTE FUNCTION pii_service.update_timestamp();

-- -----------------------------------------------------------------------------
-- 13. Trigger - Token Access Count Update
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION pii_service.update_token_access()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE pii_service.token_vault
    SET access_count = access_count + 1,
        last_accessed_at = NOW()
    WHERE id = NEW.id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- -----------------------------------------------------------------------------
-- 14. Continuous Aggregates for Metrics
-- -----------------------------------------------------------------------------

-- Detection metrics by hour
CREATE MATERIALIZED VIEW pii_service.detection_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at) AS bucket,
    action,
    pii_type,
    tenant_id,
    COUNT(*) AS operation_count
FROM pii_service.audit_log
WHERE action IN ('detect', 'redact', 'enforcement_block', 'enforcement_redact', 'enforcement_allow')
GROUP BY bucket, action, pii_type, tenant_id
WITH NO DATA;

-- Refresh policy for detection metrics
SELECT add_continuous_aggregate_policy('pii_service.detection_metrics_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Tokenization metrics by hour
CREATE MATERIALIZED VIEW pii_service.tokenization_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at) AS bucket,
    action,
    pii_type,
    tenant_id,
    COUNT(*) AS operation_count
FROM pii_service.audit_log
WHERE action IN ('tokenize', 'detokenize', 'detokenize_denied')
GROUP BY bucket, action, pii_type, tenant_id
WITH NO DATA;

-- Refresh policy for tokenization metrics
SELECT add_continuous_aggregate_policy('pii_service.tokenization_metrics_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- 15. Helper Functions
-- -----------------------------------------------------------------------------

-- Function to clean expired tokens
CREATE OR REPLACE FUNCTION pii_service.cleanup_expired_tokens()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM pii_service.token_vault
    WHERE expires_at < NOW();

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION pii_service.cleanup_expired_tokens IS
    'Remove expired tokens from the vault. Should be called periodically via cron.';

-- Function to clean expired quarantine items
CREATE OR REPLACE FUNCTION pii_service.cleanup_expired_quarantine()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Mark expired items as deleted
    UPDATE pii_service.quarantine
    SET status = 'deleted'
    WHERE expires_at < NOW() AND status = 'pending';

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION pii_service.cleanup_expired_quarantine IS
    'Mark expired quarantine items as deleted. Should be called periodically via cron.';

-- -----------------------------------------------------------------------------
-- 16. Verification
-- -----------------------------------------------------------------------------

DO $$
DECLARE
    tbl_name TEXT;
    required_tables TEXT[] := ARRAY[
        'token_vault',
        'allowlist',
        'quarantine',
        'remediation_items',
        'remediation_log',
        'deletion_certificates',
        'audit_log',
        'enforcement_policies'
    ];
    perm_count INTEGER;
    hypertable_count INTEGER;
BEGIN
    -- Verify all tables exist
    FOREACH tbl_name IN ARRAY required_tables
    LOOP
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'pii_service' AND table_name = tbl_name
        ) THEN
            RAISE EXCEPTION 'Required table pii_service.% was not created', tbl_name;
        ELSE
            RAISE NOTICE 'Table pii_service.% created successfully', tbl_name;
        END IF;
    END LOOP;

    -- Verify RLS is enabled on token_vault
    IF NOT EXISTS (
        SELECT 1 FROM pg_tables
        WHERE schemaname = 'pii_service'
          AND tablename = 'token_vault'
          AND rowsecurity = TRUE
    ) THEN
        RAISE EXCEPTION 'RLS not enabled on pii_service.token_vault';
    END IF;

    -- Verify permissions were inserted
    SELECT COUNT(*) INTO perm_count
    FROM security.permissions
    WHERE name LIKE 'pii:%';

    IF perm_count < 11 THEN
        RAISE EXCEPTION 'Expected 11 pii permissions, found %', perm_count;
    END IF;

    -- Verify hypertables were created
    SELECT COUNT(*) INTO hypertable_count
    FROM timescaledb_information.hypertables
    WHERE hypertable_schema = 'pii_service';

    IF hypertable_count < 2 THEN
        RAISE EXCEPTION 'Expected 2 hypertables, found %', hypertable_count;
    END IF;

    RAISE NOTICE 'V018 PII Service migration completed successfully';
    RAISE NOTICE 'Created 8 tables, 11 permissions, 2 hypertables, 2 continuous aggregates';
END $$;
