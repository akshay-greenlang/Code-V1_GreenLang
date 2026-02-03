-- =============================================================================
-- GreenLang Climate OS - Audit Hypertables
-- =============================================================================
-- File: 05_audit_hypertables.sql
-- Description: TimescaleDB hypertables for audit logs and API request tracking.
--              These tables support compliance requirements with long retention.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Audit Log Hypertable
-- -----------------------------------------------------------------------------
-- Immutable audit trail for all significant system actions.
-- Required for regulatory compliance (SEC, SOX, GHG Protocol verification).
CREATE TABLE IF NOT EXISTS audit.audit_log (
    -- Time of the action
    time TIMESTAMPTZ NOT NULL,

    -- User who performed the action (NULL for system actions)
    user_id UUID,

    -- Organization context
    org_id UUID,

    -- Session identifier for correlation
    session_id UUID,

    -- Action classification
    action VARCHAR(100) NOT NULL,
    action_category VARCHAR(50) NOT NULL,

    -- Resource affected
    resource_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255),
    resource_name VARCHAR(255),

    -- Change details
    -- For updates: {"before": {...}, "after": {...}}
    -- For creates: {"created": {...}}
    -- For deletes: {"deleted": {...}}
    changes JSONB,

    -- Request context
    ip_address INET,
    user_agent TEXT,
    request_id UUID,

    -- Geographic information (from IP)
    geo_country CHAR(2),
    geo_region VARCHAR(100),
    geo_city VARCHAR(100),

    -- API key used (if applicable)
    api_key_id UUID,

    -- Result of the action
    success BOOLEAN NOT NULL DEFAULT true,
    error_message TEXT,
    error_code VARCHAR(50),

    -- Additional context
    metadata JSONB DEFAULT '{}',

    -- Compliance flags
    -- Indicates if this action is relevant for specific compliance frameworks
    compliance_relevant BOOLEAN DEFAULT false,
    compliance_frameworks VARCHAR(50)[] DEFAULT '{}',

    -- Record creation (immutable)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create hypertable with 1 day chunks
-- Daily chunks are appropriate for audit logs (medium volume, long retention)
SELECT create_hypertable(
    'audit.audit_log',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Primary access pattern: user's actions
CREATE INDEX IF NOT EXISTS idx_audit_log_user
    ON audit.audit_log(user_id, time DESC)
    WHERE user_id IS NOT NULL;

-- Organization audit queries
CREATE INDEX IF NOT EXISTS idx_audit_log_org
    ON audit.audit_log(org_id, time DESC)
    WHERE org_id IS NOT NULL;

-- Resource-specific audit trail
CREATE INDEX IF NOT EXISTS idx_audit_log_resource
    ON audit.audit_log(resource_type, resource_id, time DESC);

-- Action-based queries
CREATE INDEX IF NOT EXISTS idx_audit_log_action
    ON audit.audit_log(action, time DESC);

-- Failed actions (security monitoring)
CREATE INDEX IF NOT EXISTS idx_audit_log_failures
    ON audit.audit_log(success, time DESC)
    WHERE success = false;

-- IP address queries (security investigation)
CREATE INDEX IF NOT EXISTS idx_audit_log_ip
    ON audit.audit_log(ip_address, time DESC)
    WHERE ip_address IS NOT NULL;

-- Session correlation
CREATE INDEX IF NOT EXISTS idx_audit_log_session
    ON audit.audit_log(session_id, time DESC)
    WHERE session_id IS NOT NULL;

-- Compliance-relevant actions
CREATE INDEX IF NOT EXISTS idx_audit_log_compliance
    ON audit.audit_log(time DESC)
    WHERE compliance_relevant = true;

-- GIN index for searching changes JSONB
CREATE INDEX IF NOT EXISTS idx_audit_log_changes
    ON audit.audit_log USING GIN(changes jsonb_path_ops)
    WHERE changes IS NOT NULL;

-- GIN index for compliance frameworks
CREATE INDEX IF NOT EXISTS idx_audit_log_frameworks
    ON audit.audit_log USING GIN(compliance_frameworks)
    WHERE compliance_relevant = true;

COMMENT ON TABLE audit.audit_log IS 'Immutable audit log with 1-day chunks, 10-year retention for compliance';
COMMENT ON COLUMN audit.audit_log.changes IS 'Before/after state for changes, or created/deleted content';
COMMENT ON COLUMN audit.audit_log.compliance_frameworks IS 'Compliance frameworks this action is relevant for (GHG_PROTOCOL, SOX, SEC, etc.)';

-- -----------------------------------------------------------------------------
-- API Requests Hypertable
-- -----------------------------------------------------------------------------
-- Detailed log of all API requests for monitoring, debugging, and rate limiting.
CREATE TABLE IF NOT EXISTS audit.api_requests (
    -- Time request was received
    time TIMESTAMPTZ NOT NULL,

    -- API key used
    key_id UUID,

    -- User context (resolved from API key or session)
    user_id UUID,
    org_id UUID,

    -- Request details
    request_id UUID NOT NULL,
    method VARCHAR(10) NOT NULL,
    path VARCHAR(500) NOT NULL,
    query_params JSONB,

    -- Response details
    status INTEGER NOT NULL,

    -- Performance metrics
    duration_ms INTEGER NOT NULL,

    -- Size metrics
    request_size INTEGER,
    response_size INTEGER,

    -- Request headers (selected, not all for privacy)
    content_type VARCHAR(100),
    accept_type VARCHAR(100),
    user_agent TEXT,

    -- Client information
    ip_address INET,

    -- Rate limiting info
    rate_limit_remaining INTEGER,
    rate_limit_reset TIMESTAMPTZ,

    -- Error information (if applicable)
    error_code VARCHAR(50),
    error_message TEXT,

    -- Tracing
    trace_id VARCHAR(64),
    span_id VARCHAR(32),

    -- Record creation
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create hypertable with 1 hour chunks
-- Small chunks for high-volume API requests (frequent writes, short retention)
SELECT create_hypertable(
    'audit.api_requests',
    'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- API key usage analysis
CREATE INDEX IF NOT EXISTS idx_api_requests_key
    ON audit.api_requests(key_id, time DESC)
    WHERE key_id IS NOT NULL;

-- User request history
CREATE INDEX IF NOT EXISTS idx_api_requests_user
    ON audit.api_requests(user_id, time DESC)
    WHERE user_id IS NOT NULL;

-- Organization request history
CREATE INDEX IF NOT EXISTS idx_api_requests_org
    ON audit.api_requests(org_id, time DESC)
    WHERE org_id IS NOT NULL;

-- Path analysis (which endpoints are used most)
CREATE INDEX IF NOT EXISTS idx_api_requests_path
    ON audit.api_requests(path, time DESC);

-- Error analysis
CREATE INDEX IF NOT EXISTS idx_api_requests_errors
    ON audit.api_requests(status, time DESC)
    WHERE status >= 400;

-- Performance analysis (slow requests)
CREATE INDEX IF NOT EXISTS idx_api_requests_slow
    ON audit.api_requests(duration_ms DESC, time DESC)
    WHERE duration_ms > 1000;

-- IP-based analysis
CREATE INDEX IF NOT EXISTS idx_api_requests_ip
    ON audit.api_requests(ip_address, time DESC);

-- Tracing correlation
CREATE INDEX IF NOT EXISTS idx_api_requests_trace
    ON audit.api_requests(trace_id, time DESC)
    WHERE trace_id IS NOT NULL;

-- Request ID lookup
CREATE INDEX IF NOT EXISTS idx_api_requests_request_id
    ON audit.api_requests(request_id);

COMMENT ON TABLE audit.api_requests IS 'API request log with 1-hour chunks, 90-day retention';

-- -----------------------------------------------------------------------------
-- Security Events Table
-- -----------------------------------------------------------------------------
-- Security-specific events (login attempts, permission changes, etc.)
CREATE TABLE IF NOT EXISTS audit.security_events (
    -- Time of the event
    time TIMESTAMPTZ NOT NULL,

    -- Event classification
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL DEFAULT 'info',

    -- User context
    user_id UUID,
    org_id UUID,

    -- Event details
    description TEXT NOT NULL,

    -- Related entities
    target_user_id UUID,
    target_resource_type VARCHAR(100),
    target_resource_id VARCHAR(255),

    -- Client information
    ip_address INET,
    user_agent TEXT,

    -- Geographic information
    geo_country CHAR(2),
    geo_region VARCHAR(100),

    -- Risk assessment
    risk_score INTEGER CHECK (risk_score BETWEEN 0 AND 100),
    risk_factors JSONB DEFAULT '{}',

    -- Response actions taken
    actions_taken VARCHAR(100)[] DEFAULT '{}',

    -- Investigation status
    investigation_status VARCHAR(50) DEFAULT 'new',
    investigated_by UUID,
    investigation_notes TEXT,

    -- Additional context
    metadata JSONB DEFAULT '{}',

    -- Record creation
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create hypertable with 1 day chunks
SELECT create_hypertable(
    'audit.security_events',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_security_events_type
    ON audit.security_events(event_type, time DESC);

CREATE INDEX IF NOT EXISTS idx_security_events_severity
    ON audit.security_events(severity, time DESC);

CREATE INDEX IF NOT EXISTS idx_security_events_user
    ON audit.security_events(user_id, time DESC)
    WHERE user_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_security_events_org
    ON audit.security_events(org_id, time DESC)
    WHERE org_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_security_events_ip
    ON audit.security_events(ip_address, time DESC)
    WHERE ip_address IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_security_events_risk
    ON audit.security_events(risk_score DESC, time DESC)
    WHERE risk_score >= 70;

CREATE INDEX IF NOT EXISTS idx_security_events_investigation
    ON audit.security_events(investigation_status, time DESC)
    WHERE investigation_status != 'closed';

COMMENT ON TABLE audit.security_events IS 'Security events with risk scoring and investigation tracking';

-- -----------------------------------------------------------------------------
-- Helper function to log audit event
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION audit.log_action(
    p_user_id UUID,
    p_org_id UUID,
    p_action VARCHAR,
    p_action_category VARCHAR,
    p_resource_type VARCHAR,
    p_resource_id VARCHAR,
    p_resource_name VARCHAR DEFAULT NULL,
    p_changes JSONB DEFAULT NULL,
    p_ip_address INET DEFAULT NULL,
    p_user_agent TEXT DEFAULT NULL,
    p_request_id UUID DEFAULT NULL,
    p_success BOOLEAN DEFAULT true,
    p_error_message TEXT DEFAULT NULL,
    p_compliance_relevant BOOLEAN DEFAULT false,
    p_compliance_frameworks VARCHAR[] DEFAULT '{}'
)
RETURNS UUID AS $$
DECLARE
    v_audit_id UUID;
BEGIN
    INSERT INTO audit.audit_log (
        time, user_id, org_id, action, action_category,
        resource_type, resource_id, resource_name, changes,
        ip_address, user_agent, request_id,
        success, error_message,
        compliance_relevant, compliance_frameworks
    ) VALUES (
        NOW(), p_user_id, p_org_id, p_action, p_action_category,
        p_resource_type, p_resource_id, p_resource_name, p_changes,
        p_ip_address, p_user_agent, p_request_id,
        p_success, p_error_message,
        p_compliance_relevant, p_compliance_frameworks
    )
    RETURNING ctid INTO v_audit_id;

    RETURN v_audit_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION audit.log_action IS 'Helper function to create audit log entries';

-- -----------------------------------------------------------------------------
-- Helper function to log security event
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION audit.log_security_event(
    p_event_type VARCHAR,
    p_severity VARCHAR,
    p_description TEXT,
    p_user_id UUID DEFAULT NULL,
    p_org_id UUID DEFAULT NULL,
    p_ip_address INET DEFAULT NULL,
    p_risk_score INTEGER DEFAULT NULL,
    p_risk_factors JSONB DEFAULT '{}',
    p_metadata JSONB DEFAULT '{}'
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO audit.security_events (
        time, event_type, severity, description,
        user_id, org_id, ip_address,
        risk_score, risk_factors, metadata
    ) VALUES (
        NOW(), p_event_type, p_severity, p_description,
        p_user_id, p_org_id, p_ip_address,
        p_risk_score, p_risk_factors, p_metadata
    );
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION audit.log_security_event IS 'Helper function to create security event entries';
