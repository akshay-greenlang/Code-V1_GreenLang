-- ============================================================
-- V009: Auth Service Schema (SEC-001)
-- ============================================================
-- PRD: SEC-001 - Core JWT Authentication Service
-- Creates 4 tables in the `security` schema with Row-Level Security,
-- indexes, cleanup functions, and retention automation for the
-- JWT authentication service.
--
-- Tables:
--   security.token_blacklist    - JTI revocation records (L2 durable layer)
--   security.refresh_tokens     - Opaque refresh tokens with family tracking
--   security.password_history   - Password change audit trail
--   security.login_attempts     - Login attempt tracking for lockout/analytics
-- ============================================================

-- Ensure security schema exists
CREATE SCHEMA IF NOT EXISTS security;

-- ============================================================
-- 1. Token Blacklist
-- ============================================================
-- L2 durable layer for revoked JTIs.  Redis is L1 (hot path).
-- Entries are pruned after original_expiry passes since the token
-- can no longer pass signature validation anyway.
-- ============================================================

CREATE TABLE IF NOT EXISTS security.token_blacklist (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    jti             VARCHAR(128) NOT NULL UNIQUE,
    user_id         VARCHAR(128) NOT NULL,
    tenant_id       VARCHAR(128) NOT NULL,
    token_type      VARCHAR(16)  NOT NULL DEFAULT 'access'
                    CHECK (token_type IN ('access', 'refresh')),
    reason          VARCHAR(64)  NOT NULL DEFAULT 'logout'
                    CHECK (reason IN (
                        'logout', 'password_change', 'admin_revoke',
                        'account_compromise', 'token_rotation',
                        'family_revoke', 'bulk_revoke',
                        'revoke_endpoint', 'session_timeout'
                    )),
    revoked_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    original_expiry TIMESTAMPTZ,
    metadata        JSONB        NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_tb_jti        ON security.token_blacklist (jti);
CREATE INDEX idx_tb_user       ON security.token_blacklist (user_id);
CREATE INDEX idx_tb_tenant     ON security.token_blacklist (tenant_id);
CREATE INDEX idx_tb_expiry     ON security.token_blacklist (original_expiry)
    WHERE original_expiry IS NOT NULL;
CREATE INDEX idx_tb_revoked_at ON security.token_blacklist (revoked_at);

-- Row-Level Security
ALTER TABLE security.token_blacklist ENABLE ROW LEVEL SECURITY;

CREATE POLICY token_blacklist_tenant_isolation ON security.token_blacklist
    USING (tenant_id = current_setting('app.current_tenant', TRUE));

CREATE POLICY token_blacklist_service_access ON security.token_blacklist
    FOR ALL
    TO greenlang_service
    USING (TRUE);

-- ============================================================
-- 2. Refresh Tokens
-- ============================================================
-- Opaque refresh tokens stored as SHA-256 hashes.  Supports
-- family-based rotation tracking and reuse detection.
-- ============================================================

CREATE TABLE IF NOT EXISTS security.refresh_tokens (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    token_hash         VARCHAR(128)  NOT NULL UNIQUE,
    user_id            VARCHAR(128)  NOT NULL,
    tenant_id          VARCHAR(128)  NOT NULL,
    family_id          VARCHAR(128)  NOT NULL,
    status             VARCHAR(16)   NOT NULL DEFAULT 'active'
                       CHECK (status IN ('active', 'rotated', 'revoked')),
    device_fingerprint VARCHAR(256),
    ip_address         VARCHAR(45),
    user_agent         VARCHAR(512),
    created_at         TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    expires_at         TIMESTAMPTZ   NOT NULL,
    rotated_at         TIMESTAMPTZ,
    revoked_at         TIMESTAMPTZ,
    revoke_reason      VARCHAR(64)
);

CREATE INDEX idx_rt_hash       ON security.refresh_tokens (token_hash);
CREATE INDEX idx_rt_user       ON security.refresh_tokens (user_id);
CREATE INDEX idx_rt_tenant     ON security.refresh_tokens (tenant_id);
CREATE INDEX idx_rt_family     ON security.refresh_tokens (family_id);
CREATE INDEX idx_rt_status     ON security.refresh_tokens (status)
    WHERE status = 'active';
CREATE INDEX idx_rt_expires    ON security.refresh_tokens (expires_at);

-- Row-Level Security
ALTER TABLE security.refresh_tokens ENABLE ROW LEVEL SECURITY;

CREATE POLICY refresh_tokens_tenant_isolation ON security.refresh_tokens
    USING (tenant_id = current_setting('app.current_tenant', TRUE));

CREATE POLICY refresh_tokens_service_access ON security.refresh_tokens
    FOR ALL
    TO greenlang_service
    USING (TRUE);

-- ============================================================
-- 3. Password History
-- ============================================================
-- Tracks previous password hashes to enforce password-reuse
-- policies.  Stores only hashes, never plaintext.
-- ============================================================

CREATE TABLE IF NOT EXISTS security.password_history (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       VARCHAR(128) NOT NULL,
    tenant_id     VARCHAR(128) NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    changed_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    changed_by    VARCHAR(128),
    ip_address    VARCHAR(45),
    metadata      JSONB        NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_ph_user      ON security.password_history (user_id);
CREATE INDEX idx_ph_tenant    ON security.password_history (tenant_id);
CREATE INDEX idx_ph_changed   ON security.password_history (changed_at);

-- Row-Level Security
ALTER TABLE security.password_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY password_history_tenant_isolation ON security.password_history
    USING (tenant_id = current_setting('app.current_tenant', TRUE));

CREATE POLICY password_history_service_access ON security.password_history
    FOR ALL
    TO greenlang_service
    USING (TRUE);

-- ============================================================
-- 4. Login Attempts
-- ============================================================
-- Records every authentication attempt for security analytics,
-- lockout enforcement, and compliance reporting.
-- ============================================================

CREATE TABLE IF NOT EXISTS security.login_attempts (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username     VARCHAR(256) NOT NULL,
    tenant_id    VARCHAR(128),
    success      BOOLEAN      NOT NULL,
    ip_address   VARCHAR(45),
    user_agent   VARCHAR(512),
    failure_reason VARCHAR(128),
    mfa_used     BOOLEAN      NOT NULL DEFAULT FALSE,
    attempted_at TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    metadata     JSONB        NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_la_username   ON security.login_attempts (username);
CREATE INDEX idx_la_tenant     ON security.login_attempts (tenant_id);
CREATE INDEX idx_la_success    ON security.login_attempts (success)
    WHERE success = FALSE;
CREATE INDEX idx_la_ip         ON security.login_attempts (ip_address);
CREATE INDEX idx_la_attempted  ON security.login_attempts (attempted_at);

-- Convert to TimescaleDB hypertable for time-series analytics
-- (only if TimescaleDB extension is available)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable(
            'security.login_attempts',
            'attempted_at',
            if_not_exists => TRUE,
            migrate_data  => TRUE
        );
    END IF;
END $$;

-- Row-Level Security
ALTER TABLE security.login_attempts ENABLE ROW LEVEL SECURITY;

CREATE POLICY login_attempts_tenant_isolation ON security.login_attempts
    USING (tenant_id = current_setting('app.current_tenant', TRUE));

CREATE POLICY login_attempts_service_access ON security.login_attempts
    FOR ALL
    TO greenlang_service
    USING (TRUE);

-- ============================================================
-- Cleanup Functions
-- ============================================================

-- Cleanup expired blacklist entries
CREATE OR REPLACE FUNCTION security.cleanup_expired_blacklist()
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM security.token_blacklist
    WHERE original_expiry IS NOT NULL
      AND original_expiry < NOW();

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    RAISE NOTICE 'Cleaned up % expired token_blacklist entries', deleted_count;
    RETURN deleted_count;
END;
$$;

-- Cleanup expired and old rotated refresh tokens
CREATE OR REPLACE FUNCTION security.cleanup_expired_refresh_tokens(
    retention_days INTEGER DEFAULT 14
)
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    deleted_count INTEGER;
    cutoff TIMESTAMPTZ;
BEGIN
    cutoff := NOW() - (retention_days || ' days')::INTERVAL;

    DELETE FROM security.refresh_tokens
    WHERE (expires_at < NOW() AND status IN ('revoked', 'rotated'))
       OR (revoked_at IS NOT NULL AND revoked_at < cutoff);

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    RAISE NOTICE 'Cleaned up % expired/old refresh_tokens entries', deleted_count;
    RETURN deleted_count;
END;
$$;

-- Cleanup old login attempts (retention: 90 days default)
CREATE OR REPLACE FUNCTION security.cleanup_old_login_attempts(
    retention_days INTEGER DEFAULT 90
)
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    deleted_count INTEGER;
    cutoff TIMESTAMPTZ;
BEGIN
    cutoff := NOW() - (retention_days || ' days')::INTERVAL;

    DELETE FROM security.login_attempts
    WHERE attempted_at < cutoff;

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    RAISE NOTICE 'Cleaned up % old login_attempts entries', deleted_count;
    RETURN deleted_count;
END;
$$;

-- Cleanup old password history (keep last 12 per user)
CREATE OR REPLACE FUNCTION security.cleanup_old_password_history(
    keep_count INTEGER DEFAULT 12
)
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    WITH ranked AS (
        SELECT id,
               ROW_NUMBER() OVER (
                   PARTITION BY user_id
                   ORDER BY changed_at DESC
               ) AS rn
        FROM security.password_history
    )
    DELETE FROM security.password_history
    WHERE id IN (
        SELECT id FROM ranked WHERE rn > keep_count
    );

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    RAISE NOTICE 'Cleaned up % old password_history entries', deleted_count;
    RETURN deleted_count;
END;
$$;

-- ============================================================
-- Master cleanup function (called by K8s CronJob)
-- ============================================================

CREATE OR REPLACE FUNCTION security.run_auth_cleanup()
RETURNS TABLE (
    component TEXT,
    deleted_count INTEGER
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    component := 'token_blacklist';
    deleted_count := security.cleanup_expired_blacklist();
    RETURN NEXT;

    component := 'refresh_tokens';
    deleted_count := security.cleanup_expired_refresh_tokens(14);
    RETURN NEXT;

    component := 'login_attempts';
    deleted_count := security.cleanup_old_login_attempts(90);
    RETURN NEXT;

    component := 'password_history';
    deleted_count := security.cleanup_old_password_history(12);
    RETURN NEXT;
END;
$$;

-- ============================================================
-- Continuous Aggregate: login attempts per hour (if TimescaleDB)
-- ============================================================

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        EXECUTE '
            CREATE MATERIALIZED VIEW IF NOT EXISTS security.login_attempts_hourly
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket(''1 hour'', attempted_at) AS bucket,
                tenant_id,
                username,
                COUNT(*) FILTER (WHERE success = TRUE)  AS success_count,
                COUNT(*) FILTER (WHERE success = FALSE) AS failure_count,
                COUNT(DISTINCT ip_address)               AS unique_ips
            FROM security.login_attempts
            GROUP BY bucket, tenant_id, username
            WITH NO DATA
        ';

        -- Refresh policy: refresh every 15 minutes, lag 1 hour
        PERFORM add_continuous_aggregate_policy(
            'security.login_attempts_hourly',
            start_offset    => INTERVAL '3 hours',
            end_offset      => INTERVAL '1 hour',
            schedule_interval => INTERVAL '15 minutes',
            if_not_exists   => TRUE
        );
    END IF;
END $$;

-- ============================================================
-- Grants
-- ============================================================

DO $$
BEGIN
    -- Service role (used by the application)
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_service') THEN
        GRANT USAGE ON SCHEMA security TO greenlang_service;
        GRANT SELECT, INSERT, UPDATE, DELETE
            ON ALL TABLES IN SCHEMA security TO greenlang_service;
        GRANT EXECUTE
            ON ALL FUNCTIONS IN SCHEMA security TO greenlang_service;
    END IF;

    -- Read-only role (for reporting/analytics)
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_readonly') THEN
        GRANT USAGE ON SCHEMA security TO greenlang_readonly;
        GRANT SELECT
            ON ALL TABLES IN SCHEMA security TO greenlang_readonly;
    END IF;
END $$;

-- ============================================================
-- Comments
-- ============================================================

COMMENT ON TABLE security.token_blacklist IS
    'L2 durable layer for revoked JWT JTIs. L1 is Redis. SEC-001.';

COMMENT ON TABLE security.refresh_tokens IS
    'Opaque refresh tokens stored as SHA-256 hashes with family-based rotation tracking. SEC-001.';

COMMENT ON TABLE security.password_history IS
    'Password change audit trail for reuse-prevention policies. SEC-001.';

COMMENT ON TABLE security.login_attempts IS
    'Authentication attempt log for lockout enforcement and security analytics. SEC-001.';

COMMENT ON FUNCTION security.run_auth_cleanup() IS
    'Master cleanup function for all auth tables. Called by K8s CronJob daily. SEC-001.';
