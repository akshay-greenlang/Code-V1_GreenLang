-- =============================================================================
-- V005: Audit Tables
-- =============================================================================
-- Description: Creates comprehensive audit infrastructure including
--              audit_log, user_activity, and api_requests hypertables.
-- Author: GreenLang Data Integration Team
-- Requires: TimescaleDB (V002)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Audit Event Types
-- -----------------------------------------------------------------------------

CREATE TYPE audit.event_category AS ENUM (
    'authentication',
    'authorization',
    'data_access',
    'data_modification',
    'configuration',
    'integration',
    'reporting',
    'admin_action',
    'security',
    'system'
);

CREATE TYPE audit.severity_level AS ENUM (
    'debug',
    'info',
    'notice',
    'warning',
    'error',
    'critical',
    'alert',
    'emergency'
);

CREATE TYPE audit.api_method AS ENUM (
    'GET',
    'POST',
    'PUT',
    'PATCH',
    'DELETE',
    'HEAD',
    'OPTIONS'
);

-- -----------------------------------------------------------------------------
-- Enhanced Audit Log Table (if not already hypertable from V001)
-- -----------------------------------------------------------------------------

-- Drop the simple audit_log if exists and recreate with more fields
DROP TABLE IF EXISTS audit.audit_log CASCADE;

CREATE TABLE audit.audit_log (
    -- Time dimension
    performed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Event Identification
    id UUID NOT NULL DEFAULT uuid_generate_v4(),
    event_id VARCHAR(100),  -- External correlation ID
    trace_id VARCHAR(100),  -- Distributed tracing ID
    span_id VARCHAR(100),

    -- Classification
    category event_category NOT NULL DEFAULT 'data_modification',
    severity severity_level NOT NULL DEFAULT 'info',
    event_type VARCHAR(100) NOT NULL,

    -- What Changed
    schema_name VARCHAR(100),
    table_name VARCHAR(100),
    record_id UUID,
    operation VARCHAR(20) NOT NULL,  -- INSERT, UPDATE, DELETE, SELECT, EXECUTE

    -- Change Details
    old_data JSONB,
    new_data JSONB,
    changed_fields TEXT[],
    change_summary TEXT,

    -- Who
    organization_id UUID,
    user_id UUID,
    user_email VARCHAR(255),
    user_role VARCHAR(50),
    service_account VARCHAR(100),
    impersonated_by UUID,

    -- From Where
    ip_address INET,
    user_agent TEXT,
    request_id UUID,
    session_id UUID,

    -- Context
    resource_type VARCHAR(100),
    resource_path TEXT,
    action VARCHAR(100),
    outcome VARCHAR(20) DEFAULT 'success',  -- success, failure, error
    error_message TEXT,
    error_code VARCHAR(50),

    -- Additional Data
    metadata JSONB DEFAULT '{}'::jsonb,
    tags TEXT[],

    -- Compliance
    data_classification VARCHAR(50),  -- public, internal, confidential, restricted
    retention_days INTEGER,
    gdpr_relevant BOOLEAN DEFAULT FALSE,

    -- Primary key for hypertable
    PRIMARY KEY (performed_at, id)
);

-- Convert to hypertable
SELECT create_hypertable(
    'audit.audit_log',
    'performed_at',
    chunk_time_interval => INTERVAL '${chunk_interval}',
    if_not_exists => TRUE
);

-- Enable compression
ALTER TABLE audit.audit_log SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'organization_id, category',
    timescaledb.compress_orderby = 'performed_at DESC, id'
);

SELECT add_compression_policy(
    'audit.audit_log',
    INTERVAL '${compression_after_days} days',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'audit.audit_log',
    INTERVAL '${retention_days} days',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- User Activity Hypertable
-- -----------------------------------------------------------------------------

CREATE TABLE audit.user_activity (
    -- Time dimension
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Identification
    id UUID NOT NULL DEFAULT uuid_generate_v4(),
    session_id UUID,

    -- User Information
    organization_id UUID,
    user_id UUID NOT NULL,
    user_email VARCHAR(255),

    -- Activity Details
    activity_type VARCHAR(100) NOT NULL,
    activity_category VARCHAR(50),
    description TEXT,

    -- Page/Feature Tracking
    page_path TEXT,
    page_title VARCHAR(255),
    feature_name VARCHAR(100),
    component_name VARCHAR(100),

    -- Interaction Details
    action VARCHAR(100),  -- click, view, submit, download, etc.
    target_type VARCHAR(100),
    target_id VARCHAR(255),
    target_name VARCHAR(255),

    -- Navigation
    referrer TEXT,
    entry_point TEXT,

    -- Duration and Performance
    duration_ms INTEGER,
    time_on_page_ms INTEGER,

    -- Device and Browser
    device_type VARCHAR(50),  -- desktop, mobile, tablet
    browser VARCHAR(100),
    browser_version VARCHAR(50),
    os VARCHAR(100),
    os_version VARCHAR(50),
    screen_resolution VARCHAR(20),

    -- Location (from IP)
    ip_address INET,
    country_code CHAR(2),
    region VARCHAR(100),
    city VARCHAR(100),

    -- Additional Context
    utm_source VARCHAR(255),
    utm_medium VARCHAR(255),
    utm_campaign VARCHAR(255),
    metadata JSONB DEFAULT '{}'::jsonb,

    PRIMARY KEY (timestamp, id)
);

SELECT create_hypertable(
    'audit.user_activity',
    'timestamp',
    'organization_id',
    4,
    chunk_time_interval => INTERVAL '${chunk_interval}',
    if_not_exists => TRUE
);

ALTER TABLE audit.user_activity SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'organization_id, user_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy(
    'audit.user_activity',
    INTERVAL '${compression_after_days} days',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'audit.user_activity',
    INTERVAL '${retention_days} days',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- API Requests Hypertable
-- -----------------------------------------------------------------------------

CREATE TABLE audit.api_requests (
    -- Time dimension
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Identification
    id UUID NOT NULL DEFAULT uuid_generate_v4(),
    request_id UUID,
    trace_id VARCHAR(100),

    -- Request Details
    method audit.api_method NOT NULL,
    path TEXT NOT NULL,
    query_string TEXT,
    host VARCHAR(255),
    protocol VARCHAR(10),

    -- Authentication
    organization_id UUID,
    user_id UUID,
    api_key_id UUID,
    auth_method VARCHAR(50),  -- jwt, api_key, oauth2, none

    -- Request Body (sanitized)
    request_content_type VARCHAR(100),
    request_size_bytes INTEGER,
    request_body_sample TEXT,  -- First N bytes, sanitized

    -- Response
    status_code INTEGER NOT NULL,
    response_content_type VARCHAR(100),
    response_size_bytes INTEGER,
    response_time_ms INTEGER,

    -- Error Information
    error_type VARCHAR(100),
    error_message TEXT,
    error_stack TEXT,

    -- Rate Limiting
    rate_limit_remaining INTEGER,
    rate_limit_reset_at TIMESTAMPTZ,
    rate_limited BOOLEAN DEFAULT FALSE,

    -- Client Information
    ip_address INET,
    user_agent TEXT,
    client_id VARCHAR(255),
    client_version VARCHAR(50),
    sdk_name VARCHAR(100),
    sdk_version VARCHAR(50),

    -- Geolocation
    country_code CHAR(2),
    region VARCHAR(100),

    -- Performance Metrics
    db_query_count INTEGER,
    db_query_time_ms INTEGER,
    cache_hits INTEGER,
    cache_misses INTEGER,

    -- Tags and Metadata
    endpoint_name VARCHAR(100),
    endpoint_version VARCHAR(20),
    tags TEXT[],
    metadata JSONB DEFAULT '{}'::jsonb,

    PRIMARY KEY (timestamp, id)
);

SELECT create_hypertable(
    'audit.api_requests',
    'timestamp',
    'organization_id',
    4,
    chunk_time_interval => INTERVAL '${chunk_interval}',
    if_not_exists => TRUE
);

ALTER TABLE audit.api_requests SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'organization_id, path',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy(
    'audit.api_requests',
    INTERVAL '${compression_after_days} days',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'audit.api_requests',
    INTERVAL '${retention_days} days',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- Indexes for Audit Log
-- -----------------------------------------------------------------------------

CREATE INDEX idx_audit_log_org_time ON audit.audit_log (organization_id, performed_at DESC);
CREATE INDEX idx_audit_log_user ON audit.audit_log (user_id, performed_at DESC);
CREATE INDEX idx_audit_log_table ON audit.audit_log (schema_name, table_name, performed_at DESC);
CREATE INDEX idx_audit_log_record ON audit.audit_log (record_id, performed_at DESC);
CREATE INDEX idx_audit_log_category ON audit.audit_log (category, performed_at DESC);
CREATE INDEX idx_audit_log_severity ON audit.audit_log (severity, performed_at DESC)
    WHERE severity IN ('warning', 'error', 'critical', 'alert', 'emergency');
CREATE INDEX idx_audit_log_event_type ON audit.audit_log (event_type, performed_at DESC);
CREATE INDEX idx_audit_log_trace ON audit.audit_log (trace_id) WHERE trace_id IS NOT NULL;
CREATE INDEX idx_audit_log_request ON audit.audit_log (request_id) WHERE request_id IS NOT NULL;
CREATE INDEX idx_audit_log_gdpr ON audit.audit_log (performed_at DESC) WHERE gdpr_relevant = TRUE;
CREATE INDEX idx_audit_log_tags ON audit.audit_log USING gin(tags);
CREATE INDEX idx_audit_log_metadata ON audit.audit_log USING gin(metadata jsonb_path_ops);

-- -----------------------------------------------------------------------------
-- Indexes for User Activity
-- -----------------------------------------------------------------------------

CREATE INDEX idx_user_activity_org ON audit.user_activity (organization_id, timestamp DESC);
CREATE INDEX idx_user_activity_user ON audit.user_activity (user_id, timestamp DESC);
CREATE INDEX idx_user_activity_session ON audit.user_activity (session_id, timestamp DESC);
CREATE INDEX idx_user_activity_type ON audit.user_activity (activity_type, timestamp DESC);
CREATE INDEX idx_user_activity_page ON audit.user_activity (page_path, timestamp DESC);
CREATE INDEX idx_user_activity_feature ON audit.user_activity (feature_name, timestamp DESC)
    WHERE feature_name IS NOT NULL;

-- -----------------------------------------------------------------------------
-- Indexes for API Requests
-- -----------------------------------------------------------------------------

CREATE INDEX idx_api_requests_org ON audit.api_requests (organization_id, timestamp DESC);
CREATE INDEX idx_api_requests_user ON audit.api_requests (user_id, timestamp DESC);
CREATE INDEX idx_api_requests_path ON audit.api_requests (path, timestamp DESC);
CREATE INDEX idx_api_requests_status ON audit.api_requests (status_code, timestamp DESC);
CREATE INDEX idx_api_requests_errors ON audit.api_requests (timestamp DESC)
    WHERE status_code >= 400;
CREATE INDEX idx_api_requests_slow ON audit.api_requests (response_time_ms DESC, timestamp DESC)
    WHERE response_time_ms > 1000;
CREATE INDEX idx_api_requests_rate_limited ON audit.api_requests (timestamp DESC)
    WHERE rate_limited = TRUE;
CREATE INDEX idx_api_requests_trace ON audit.api_requests (trace_id) WHERE trace_id IS NOT NULL;
CREATE INDEX idx_api_requests_endpoint ON audit.api_requests (endpoint_name, timestamp DESC)
    WHERE endpoint_name IS NOT NULL;

-- -----------------------------------------------------------------------------
-- Continuous Aggregates for Audit Data
-- -----------------------------------------------------------------------------

-- Hourly audit event summary
CREATE MATERIALIZED VIEW IF NOT EXISTS audit.hourly_audit_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', performed_at) AS bucket,
    organization_id,
    category,
    COUNT(*) AS event_count,
    COUNT(DISTINCT user_id) AS unique_users,
    COUNT(*) FILTER (WHERE severity IN ('error', 'critical', 'alert', 'emergency')) AS error_count,
    COUNT(*) FILTER (WHERE outcome = 'failure') AS failure_count
FROM audit.audit_log
GROUP BY time_bucket('1 hour', performed_at), organization_id, category
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'audit.hourly_audit_summary',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '30 minutes',
    if_not_exists => TRUE
);

-- Daily user activity summary
CREATE MATERIALIZED VIEW IF NOT EXISTS audit.daily_user_activity_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS bucket,
    organization_id,
    user_id,
    COUNT(*) AS activity_count,
    COUNT(DISTINCT session_id) AS session_count,
    COUNT(DISTINCT page_path) AS pages_visited,
    COUNT(DISTINCT feature_name) AS features_used,
    SUM(duration_ms) AS total_duration_ms,
    MAX(timestamp) AS last_activity
FROM audit.user_activity
GROUP BY time_bucket('1 day', timestamp), organization_id, user_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'audit.daily_user_activity_summary',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Hourly API metrics
CREATE MATERIALIZED VIEW IF NOT EXISTS audit.hourly_api_metrics
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS bucket,
    organization_id,
    path,
    method,
    COUNT(*) AS request_count,
    COUNT(*) FILTER (WHERE status_code >= 200 AND status_code < 300) AS success_count,
    COUNT(*) FILTER (WHERE status_code >= 400 AND status_code < 500) AS client_error_count,
    COUNT(*) FILTER (WHERE status_code >= 500) AS server_error_count,
    AVG(response_time_ms)::INTEGER AS avg_response_time_ms,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY response_time_ms)::INTEGER AS p50_response_time_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms)::INTEGER AS p95_response_time_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY response_time_ms)::INTEGER AS p99_response_time_ms,
    MAX(response_time_ms) AS max_response_time_ms,
    SUM(request_size_bytes) AS total_request_bytes,
    SUM(response_size_bytes) AS total_response_bytes,
    COUNT(*) FILTER (WHERE rate_limited) AS rate_limited_count
FROM audit.api_requests
GROUP BY time_bucket('1 hour', timestamp), organization_id, path, method
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'audit.hourly_api_metrics',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '30 minutes',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- Audit Helper Functions
-- -----------------------------------------------------------------------------

-- Function to log an audit event
CREATE OR REPLACE FUNCTION audit.log_event(
    p_category event_category,
    p_event_type VARCHAR(100),
    p_resource_type VARCHAR(100) DEFAULT NULL,
    p_resource_id UUID DEFAULT NULL,
    p_action VARCHAR(100) DEFAULT NULL,
    p_outcome VARCHAR(20) DEFAULT 'success',
    p_old_data JSONB DEFAULT NULL,
    p_new_data JSONB DEFAULT NULL,
    p_metadata JSONB DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_event_id UUID;
    v_changed_fields TEXT[];
BEGIN
    -- Calculate changed fields if both old and new data provided
    IF p_old_data IS NOT NULL AND p_new_data IS NOT NULL THEN
        SELECT ARRAY(
            SELECT key
            FROM jsonb_each(p_old_data)
            WHERE p_old_data->key IS DISTINCT FROM p_new_data->key
        ) INTO v_changed_fields;
    END IF;

    INSERT INTO audit.audit_log (
        category,
        event_type,
        resource_type,
        record_id,
        action,
        outcome,
        old_data,
        new_data,
        changed_fields,
        organization_id,
        user_id,
        user_email,
        ip_address,
        request_id,
        session_id,
        metadata
    ) VALUES (
        p_category,
        p_event_type,
        p_resource_type,
        p_resource_id,
        p_action,
        p_outcome,
        p_old_data,
        p_new_data,
        v_changed_fields,
        NULLIF(current_setting('app.organization_id', true), '')::UUID,
        NULLIF(current_setting('app.current_user_id', true), '')::UUID,
        NULLIF(current_setting('app.current_user_email', true), ''),
        NULLIF(current_setting('app.client_ip', true), '')::INET,
        NULLIF(current_setting('app.request_id', true), '')::UUID,
        NULLIF(current_setting('app.session_id', true), '')::UUID,
        COALESCE(p_metadata, '{}'::jsonb)
    )
    RETURNING id INTO v_event_id;

    RETURN v_event_id;
END;
$$ LANGUAGE plpgsql;

-- Function to get audit trail for a record
CREATE OR REPLACE FUNCTION audit.get_record_history(
    p_table_name VARCHAR(100),
    p_record_id UUID,
    p_limit INTEGER DEFAULT 100
) RETURNS TABLE (
    performed_at TIMESTAMPTZ,
    operation VARCHAR(20),
    user_email VARCHAR(255),
    changed_fields TEXT[],
    old_data JSONB,
    new_data JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        al.performed_at,
        al.operation,
        al.user_email,
        al.changed_fields,
        al.old_data,
        al.new_data
    FROM audit.audit_log al
    WHERE al.table_name = p_table_name
      AND al.record_id = p_record_id
    ORDER BY al.performed_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to get user activity summary
CREATE OR REPLACE FUNCTION audit.get_user_activity_summary(
    p_user_id UUID,
    p_start_date TIMESTAMPTZ,
    p_end_date TIMESTAMPTZ
) RETURNS TABLE (
    total_activities BIGINT,
    unique_sessions BIGINT,
    pages_visited BIGINT,
    features_used BIGINT,
    total_time_minutes NUMERIC,
    most_used_feature VARCHAR(100),
    last_activity TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*) AS total_activities,
        COUNT(DISTINCT session_id) AS unique_sessions,
        COUNT(DISTINCT page_path) AS pages_visited,
        COUNT(DISTINCT feature_name) AS features_used,
        ROUND(SUM(duration_ms)::NUMERIC / 60000, 2) AS total_time_minutes,
        MODE() WITHIN GROUP (ORDER BY feature_name) AS most_used_feature,
        MAX(timestamp) AS last_activity
    FROM audit.user_activity
    WHERE user_id = p_user_id
      AND timestamp >= p_start_date
      AND timestamp < p_end_date;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to get API health metrics
CREATE OR REPLACE FUNCTION audit.get_api_health(
    p_organization_id UUID DEFAULT NULL,
    p_hours INTEGER DEFAULT 24
) RETURNS TABLE (
    endpoint VARCHAR(100),
    total_requests BIGINT,
    success_rate NUMERIC,
    avg_response_time_ms INTEGER,
    p95_response_time_ms INTEGER,
    error_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ar.endpoint_name AS endpoint,
        COUNT(*) AS total_requests,
        ROUND(
            COUNT(*) FILTER (WHERE status_code < 400)::NUMERIC /
            NULLIF(COUNT(*), 0) * 100, 2
        ) AS success_rate,
        AVG(ar.response_time_ms)::INTEGER AS avg_response_time_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ar.response_time_ms)::INTEGER AS p95_response_time_ms,
        COUNT(*) FILTER (WHERE ar.status_code >= 400) AS error_count
    FROM audit.api_requests ar
    WHERE ar.timestamp >= NOW() - (p_hours || ' hours')::INTERVAL
      AND (p_organization_id IS NULL OR ar.organization_id = p_organization_id)
    GROUP BY ar.endpoint_name
    ORDER BY error_count DESC, total_requests DESC;
END;
$$ LANGUAGE plpgsql STABLE;

-- -----------------------------------------------------------------------------
-- Security and Compliance Views
-- -----------------------------------------------------------------------------

-- View for security-relevant events
CREATE OR REPLACE VIEW audit.security_events AS
SELECT
    al.*
FROM audit.audit_log al
WHERE al.category IN ('authentication', 'authorization', 'security')
   OR al.severity IN ('warning', 'error', 'critical', 'alert', 'emergency')
   OR al.event_type LIKE '%failed%'
   OR al.outcome = 'failure';

-- View for GDPR-relevant data access
CREATE OR REPLACE VIEW audit.gdpr_data_access AS
SELECT
    al.performed_at,
    al.user_id,
    al.user_email,
    al.organization_id,
    al.table_name,
    al.record_id,
    al.operation,
    al.ip_address,
    al.request_id
FROM audit.audit_log al
WHERE al.gdpr_relevant = TRUE
   OR al.table_name IN ('public.users', 'public.organizations')
ORDER BY al.performed_at DESC;

-- View for failed authentication attempts
CREATE OR REPLACE VIEW audit.failed_auth_attempts AS
SELECT
    al.performed_at,
    al.ip_address,
    al.user_email,
    al.error_message,
    al.metadata
FROM audit.audit_log al
WHERE al.category = 'authentication'
  AND al.outcome = 'failure'
ORDER BY al.performed_at DESC;

-- -----------------------------------------------------------------------------
-- Row Level Security
-- -----------------------------------------------------------------------------

ALTER TABLE audit.audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit.user_activity ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit.api_requests ENABLE ROW LEVEL SECURITY;

-- Audit log: Users can see their org's audit logs (auditors only)
CREATE POLICY audit_log_select_policy ON audit.audit_log
    FOR SELECT
    USING (
        organization_id IN (
            SELECT om.organization_id
            FROM public.organization_members om
            WHERE om.user_id = NULLIF(current_setting('app.current_user_id', true), '')::UUID
              AND om.role IN ('owner', 'admin', 'auditor')
        )
        OR organization_id IS NULL  -- System events
    );

-- User activity: Users can see their org's activity
CREATE POLICY user_activity_select_policy ON audit.user_activity
    FOR SELECT
    USING (
        organization_id IN (
            SELECT organization_id
            FROM public.organization_members
            WHERE user_id = NULLIF(current_setting('app.current_user_id', true), '')::UUID
        )
    );

-- API requests: Admins can see their org's API requests
CREATE POLICY api_requests_select_policy ON audit.api_requests
    FOR SELECT
    USING (
        organization_id IN (
            SELECT om.organization_id
            FROM public.organization_members om
            WHERE om.user_id = NULLIF(current_setting('app.current_user_id', true), '')::UUID
              AND om.role IN ('owner', 'admin')
        )
    );

-- -----------------------------------------------------------------------------
-- Documentation
-- -----------------------------------------------------------------------------

COMMENT ON TABLE audit.audit_log IS 'Comprehensive audit trail for all data modifications and security events';
COMMENT ON TABLE audit.user_activity IS 'User interaction tracking for analytics and UX improvement';
COMMENT ON TABLE audit.api_requests IS 'API request logging for performance monitoring and debugging';

COMMENT ON MATERIALIZED VIEW audit.hourly_audit_summary IS 'Continuous aggregate: Hourly audit event counts by category';
COMMENT ON MATERIALIZED VIEW audit.daily_user_activity_summary IS 'Continuous aggregate: Daily user activity metrics';
COMMENT ON MATERIALIZED VIEW audit.hourly_api_metrics IS 'Continuous aggregate: Hourly API performance metrics';

COMMENT ON FUNCTION audit.log_event IS 'Helper function to create audit log entries with context';
COMMENT ON FUNCTION audit.get_record_history IS 'Get complete audit trail for a specific record';
COMMENT ON FUNCTION audit.get_user_activity_summary IS 'Get user activity summary for a date range';
COMMENT ON FUNCTION audit.get_api_health IS 'Get API health metrics for monitoring';

COMMENT ON VIEW audit.security_events IS 'Filtered view of security-relevant audit events';
COMMENT ON VIEW audit.gdpr_data_access IS 'Filtered view of GDPR-relevant data access events';
COMMENT ON VIEW audit.failed_auth_attempts IS 'Filtered view of failed authentication attempts';
