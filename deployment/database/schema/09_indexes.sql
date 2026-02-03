-- =============================================================================
-- GreenLang Climate OS - Optimized Indexes
-- =============================================================================
-- File: 09_indexes.sql
-- Description: Additional optimized indexes for common query patterns,
--              including partial indexes, GiST/GIN indexes for spatial/text.
-- =============================================================================

-- =============================================================================
-- ORGANIZATION INDEXES
-- =============================================================================

-- Full-text search on organization name
CREATE INDEX IF NOT EXISTS idx_organizations_name_trgm
    ON public.organizations USING GIN(name gin_trgm_ops);

-- JSONB settings search
CREATE INDEX IF NOT EXISTS idx_organizations_settings_gin
    ON public.organizations USING GIN(settings jsonb_path_ops);

-- Active organizations with recent activity (partial index)
CREATE INDEX IF NOT EXISTS idx_organizations_active_recent
    ON public.organizations(updated_at DESC)
    WHERE is_active = true AND deleted_at IS NULL;

-- =============================================================================
-- USER INDEXES
-- =============================================================================

-- Full-text search on user names
CREATE INDEX IF NOT EXISTS idx_users_name_trgm
    ON public.users USING GIN(
        (COALESCE(first_name, '') || ' ' || COALESCE(last_name, '')) gin_trgm_ops
    );

-- Email domain analysis (for enterprise features)
CREATE INDEX IF NOT EXISTS idx_users_email_domain
    ON public.users(split_part(email, '@', 2));

-- Active users by role (for permission checks)
CREATE INDEX IF NOT EXISTS idx_users_org_role_active
    ON public.users(org_id, role)
    WHERE is_active = true AND deleted_at IS NULL;

-- Users with pending MFA setup
CREATE INDEX IF NOT EXISTS idx_users_mfa_pending
    ON public.users(org_id, created_at)
    WHERE mfa_enabled = false AND is_active = true;

-- Recently active users (for session management)
CREATE INDEX IF NOT EXISTS idx_users_last_login
    ON public.users(last_login_at DESC NULLS LAST)
    WHERE is_active = true;

-- =============================================================================
-- PROJECT INDEXES
-- =============================================================================

-- Full-text search on project name and description
CREATE INDEX IF NOT EXISTS idx_projects_name_trgm
    ON public.projects USING GIN(name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_projects_description_trgm
    ON public.projects USING GIN(description gin_trgm_ops)
    WHERE description IS NOT NULL;

-- Active projects by type for dashboard
CREATE INDEX IF NOT EXISTS idx_projects_org_type_active
    ON public.projects(org_id, project_type)
    WHERE status = 'active' AND deleted_at IS NULL;

-- Projects by boundary type (for reporting)
CREATE INDEX IF NOT EXISTS idx_projects_boundary
    ON public.projects(boundary_type, org_id)
    WHERE status = 'active';

-- Team member queries (for access control)
CREATE INDEX IF NOT EXISTS idx_projects_team_gin
    ON public.projects USING GIN(team_members jsonb_path_ops);

-- =============================================================================
-- EMISSION SOURCE INDEXES
-- =============================================================================

-- Scope and category filtering
CREATE INDEX IF NOT EXISTS idx_emission_sources_scope_category
    ON metrics.emission_sources(scope, category, org_id)
    WHERE is_active = true;

-- Scope 3 category analysis
CREATE INDEX IF NOT EXISTS idx_emission_sources_scope3
    ON metrics.emission_sources(scope3_category, org_id)
    WHERE scope = 3 AND is_active = true;

-- Location-based queries (JSONB)
CREATE INDEX IF NOT EXISTS idx_emission_sources_location
    ON metrics.emission_sources USING GIN(location jsonb_path_ops)
    WHERE location IS NOT NULL AND location != '{}';

-- =============================================================================
-- EMISSION MEASUREMENTS INDEXES (Hypertable)
-- =============================================================================
-- Note: Basic time-based indexes are created with the hypertable.
-- These are additional indexes for specific query patterns.

-- Verification status filtering
CREATE INDEX IF NOT EXISTS idx_emission_measurements_verification
    ON metrics.emission_measurements(verification_status, time DESC)
    WHERE verification_status != 'verified';

-- Import batch tracking
CREATE INDEX IF NOT EXISTS idx_emission_measurements_batch
    ON metrics.emission_measurements(import_batch_id, time DESC)
    WHERE import_batch_id IS NOT NULL;

-- Low quality data identification
CREATE INDEX IF NOT EXISTS idx_emission_measurements_low_quality
    ON metrics.emission_measurements(data_quality_score, time DESC)
    WHERE data_quality_score IS NOT NULL AND data_quality_score < 50;

-- Emission factor reference
CREATE INDEX IF NOT EXISTS idx_emission_measurements_factor
    ON metrics.emission_measurements(emission_factor_id, time DESC)
    WHERE emission_factor_id IS NOT NULL;

-- =============================================================================
-- EMISSION FACTORS INDEXES
-- =============================================================================

-- Category + region + validity (most common lookup pattern)
CREATE INDEX IF NOT EXISTS idx_emission_factors_lookup
    ON metrics.emission_factors(category, region, valid_from DESC)
    WHERE is_active = true;

-- Source-based filtering
CREATE INDEX IF NOT EXISTS idx_emission_factors_source
    ON metrics.emission_factors(source, category)
    WHERE is_active = true;

-- Country-specific factors
CREATE INDEX IF NOT EXISTS idx_emission_factors_country
    ON metrics.emission_factors(country_code, category)
    WHERE country_code IS NOT NULL AND is_active = true;

-- Expiring factors (for maintenance)
CREATE INDEX IF NOT EXISTS idx_emission_factors_expiring
    ON metrics.emission_factors(valid_to)
    WHERE valid_to IS NOT NULL AND valid_to > NOW();

-- =============================================================================
-- DEVICE INDEXES
-- =============================================================================

-- Device type analysis
CREATE INDEX IF NOT EXISTS idx_devices_type_status
    ON metrics.devices(device_type, status, org_id);

-- Geographic queries (for map views)
CREATE INDEX IF NOT EXISTS idx_devices_geo
    ON metrics.devices(latitude, longitude)
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL;

-- Offline devices (for alerting)
CREATE INDEX IF NOT EXISTS idx_devices_offline
    ON metrics.devices(last_seen_at, org_id)
    WHERE status = 'active' AND is_online = false;

-- Calibration due (for maintenance scheduling)
CREATE INDEX IF NOT EXISTS idx_devices_calibration_due
    ON metrics.devices(next_calibration_date, org_id)
    WHERE next_calibration_date IS NOT NULL AND status = 'active';

-- Connection protocol analysis
CREATE INDEX IF NOT EXISTS idx_devices_protocol
    ON metrics.devices(communication_protocol, org_id)
    WHERE status = 'active';

-- =============================================================================
-- SENSOR READINGS INDEXES (Hypertable)
-- =============================================================================
-- Note: Additional indexes beyond the basic time-based ones.

-- Raw vs processed readings
CREATE INDEX IF NOT EXISTS idx_sensor_readings_raw
    ON metrics.sensor_readings(device_id, time DESC)
    WHERE is_raw = true;

-- Readings needing reprocessing
CREATE INDEX IF NOT EXISTS idx_sensor_readings_reprocess
    ON metrics.sensor_readings(device_id, time DESC)
    WHERE quality = 'bad' OR quality = 'uncertain';

-- =============================================================================
-- AUDIT LOG INDEXES (Hypertable)
-- =============================================================================

-- Action category analysis
CREATE INDEX IF NOT EXISTS idx_audit_log_category
    ON audit.audit_log(action_category, time DESC);

-- Specific action tracking
CREATE INDEX IF NOT EXISTS idx_audit_log_action_resource
    ON audit.audit_log(action, resource_type, time DESC);

-- Changes JSONB search (for "what changed" queries)
CREATE INDEX IF NOT EXISTS idx_audit_log_changes_path
    ON audit.audit_log USING GIN(changes jsonb_path_ops)
    WHERE changes IS NOT NULL;

-- Geographic analysis
CREATE INDEX IF NOT EXISTS idx_audit_log_geo
    ON audit.audit_log(geo_country, time DESC)
    WHERE geo_country IS NOT NULL;

-- Error tracking
CREATE INDEX IF NOT EXISTS idx_audit_log_errors
    ON audit.audit_log(error_code, time DESC)
    WHERE success = false AND error_code IS NOT NULL;

-- =============================================================================
-- API REQUESTS INDEXES (Hypertable)
-- =============================================================================

-- Method + path analysis (API usage patterns)
CREATE INDEX IF NOT EXISTS idx_api_requests_method_path
    ON audit.api_requests(method, path, time DESC);

-- High latency requests
CREATE INDEX IF NOT EXISTS idx_api_requests_latency
    ON audit.api_requests(duration_ms DESC, time DESC)
    WHERE duration_ms > 500;

-- Large responses (for optimization)
CREATE INDEX IF NOT EXISTS idx_api_requests_large_response
    ON audit.api_requests(response_size DESC, time DESC)
    WHERE response_size > 100000;

-- Status code distribution
CREATE INDEX IF NOT EXISTS idx_api_requests_status_time
    ON audit.api_requests(status, time DESC);

-- =============================================================================
-- API KEYS INDEXES
-- =============================================================================

-- Scope-based filtering
CREATE INDEX IF NOT EXISTS idx_api_keys_scope_active
    ON public.api_keys USING GIN(scopes)
    WHERE is_active = true AND revoked_at IS NULL;

-- Usage tracking (most used keys)
CREATE INDEX IF NOT EXISTS idx_api_keys_usage
    ON public.api_keys(total_requests DESC)
    WHERE is_active = true;

-- Expiration monitoring
CREATE INDEX IF NOT EXISTS idx_api_keys_expiring_soon
    ON public.api_keys(expires_at)
    WHERE expires_at IS NOT NULL
      AND expires_at > NOW()
      AND expires_at < NOW() + INTERVAL '30 days'
      AND is_active = true;

-- =============================================================================
-- SECURITY EVENTS INDEXES (Hypertable)
-- =============================================================================

-- High-risk events
CREATE INDEX IF NOT EXISTS idx_security_events_high_risk
    ON audit.security_events(risk_score DESC, time DESC)
    WHERE risk_score >= 80;

-- Open investigations
CREATE INDEX IF NOT EXISTS idx_security_events_investigating
    ON audit.security_events(investigation_status, time DESC)
    WHERE investigation_status IN ('new', 'investigating');

-- Event type patterns
CREATE INDEX IF NOT EXISTS idx_security_events_type_severity
    ON audit.security_events(event_type, severity, time DESC);

-- =============================================================================
-- COMPOSITE INDEXES FOR COMMON DASHBOARD QUERIES
-- =============================================================================

-- Organization dashboard: recent emissions by scope
CREATE INDEX IF NOT EXISTS idx_dashboard_org_emissions
    ON metrics.emission_measurements(org_id, scope, time DESC)
    INCLUDE (emission_value);

-- Project dashboard: emissions by source
CREATE INDEX IF NOT EXISTS idx_dashboard_project_emissions
    ON metrics.emission_measurements(project_id, source_id, time DESC)
    INCLUDE (emission_value, data_quality_score);

-- Device dashboard: recent readings with quality
CREATE INDEX IF NOT EXISTS idx_dashboard_device_readings
    ON metrics.sensor_readings(device_id, metric_type, time DESC)
    INCLUDE (value, quality);

-- =============================================================================
-- INDEX MAINTENANCE FUNCTIONS
-- =============================================================================

-- Function to identify unused indexes
CREATE OR REPLACE FUNCTION metrics.get_unused_indexes()
RETURNS TABLE (
    schema_name TEXT,
    table_name TEXT,
    index_name TEXT,
    index_size TEXT,
    index_scans BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        schemaname::TEXT,
        relname::TEXT,
        indexrelname::TEXT,
        pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
        idx_scan AS index_scans
    FROM pg_stat_user_indexes
    WHERE idx_scan = 0
      AND schemaname IN ('public', 'metrics', 'audit')
    ORDER BY pg_relation_size(indexrelid) DESC;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION metrics.get_unused_indexes
    IS 'Identify indexes that have never been used (candidates for removal)';

-- Function to get index statistics
CREATE OR REPLACE FUNCTION metrics.get_index_stats()
RETURNS TABLE (
    schema_name TEXT,
    table_name TEXT,
    index_name TEXT,
    index_size TEXT,
    index_scans BIGINT,
    rows_read BIGINT,
    rows_fetched BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        schemaname::TEXT,
        relname::TEXT,
        indexrelname::TEXT,
        pg_size_pretty(pg_relation_size(indexrelid)),
        idx_scan,
        idx_tup_read,
        idx_tup_fetch
    FROM pg_stat_user_indexes
    WHERE schemaname IN ('public', 'metrics', 'audit')
    ORDER BY idx_scan DESC
    LIMIT 50;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION metrics.get_index_stats
    IS 'Get usage statistics for all indexes';

-- =============================================================================
-- SUMMARY
-- =============================================================================
DO $$
BEGIN
    RAISE NOTICE '=============================================================';
    RAISE NOTICE 'GreenLang Index Summary';
    RAISE NOTICE '=============================================================';
    RAISE NOTICE '';
    RAISE NOTICE 'Index Types Created:';
    RAISE NOTICE '  - B-tree: Standard lookups and range queries';
    RAISE NOTICE '  - GIN: JSONB, array, and trigram searches';
    RAISE NOTICE '  - Partial: Filter inactive/deleted records';
    RAISE NOTICE '  - Covering (INCLUDE): Dashboard query optimization';
    RAISE NOTICE '';
    RAISE NOTICE 'Use metrics.get_unused_indexes() to find unused indexes';
    RAISE NOTICE 'Use metrics.get_index_stats() to monitor index usage';
    RAISE NOTICE '=============================================================';
END $$;
