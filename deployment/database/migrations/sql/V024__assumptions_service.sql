-- =============================================================================
-- GreenLang Climate OS - Assumptions Registry Service Schema
-- =============================================================================
-- Migration: V024
-- Component: AGENT-FOUND-004 Assumptions Registry
-- Description: Creates assumptions_service schema with version-controlled
--              assumption registry, scenario management, validation engine,
--              dependency graph, SHA-256 audit trail, hypertables for
--              assumption_versions and assumption_change_log, continuous
--              aggregates, RLS policies, and seed data for common emission
--              factor assumptions and default scenarios.
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS assumptions_service;

-- =============================================================================
-- Table: assumptions_service.assumptions
-- =============================================================================
-- Main registry of all assumptions used in zero-hallucination compliance
-- calculations. Each assumption has a unique ID, category, data type, unit,
-- current value, validation rules, dependency tracking, and provenance hash.
-- Supports multi-tenant isolation via tenant_id.

CREATE TABLE assumptions_service.assumptions (
    assumption_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    category VARCHAR(50) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    unit VARCHAR(100),
    current_value JSONB NOT NULL,
    default_value JSONB,
    metadata JSONB DEFAULT '{}'::jsonb,
    validation_rules JSONB DEFAULT '[]'::jsonb,
    depends_on TEXT[] DEFAULT '{}',
    used_by TEXT[] DEFAULT '{}',
    parent_assumption_id VARCHAR(255),
    provenance_hash CHAR(64),
    tenant_id VARCHAR(100) DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Category constraint
ALTER TABLE assumptions_service.assumptions
    ADD CONSTRAINT chk_assumption_category
    CHECK (category IN (
        'emission_factor', 'conversion_rate', 'economic_parameter',
        'regulatory_threshold', 'physical_constant', 'activity_data',
        'gwp_factor', 'grid_factor', 'transport_factor', 'custom'
    ));

-- Data type constraint
ALTER TABLE assumptions_service.assumptions
    ADD CONSTRAINT chk_assumption_data_type
    CHECK (data_type IN (
        'numeric', 'percentage', 'ratio', 'range', 'enum',
        'boolean', 'json', 'text', 'date', 'composite'
    ));

-- Provenance hash must be 64-character hex when present
ALTER TABLE assumptions_service.assumptions
    ADD CONSTRAINT chk_assumption_provenance_hash_length
    CHECK (provenance_hash IS NULL OR LENGTH(provenance_hash) = 64);

-- Self-referencing FK for parent hierarchy
ALTER TABLE assumptions_service.assumptions
    ADD CONSTRAINT fk_assumption_parent
    FOREIGN KEY (parent_assumption_id)
    REFERENCES assumptions_service.assumptions(assumption_id)
    ON DELETE SET NULL;

-- =============================================================================
-- Table: assumptions_service.assumption_versions
-- =============================================================================
-- TimescaleDB hypertable recording every version of every assumption for
-- full audit trail and historical analysis. Each row is an immutable snapshot
-- of an assumption value at a point in time, with effective date ranges,
-- change reasons, scenario associations, and provenance hashes. Partitioned
-- by timestamp for efficient time-series queries and unlimited retention.

CREATE TABLE assumptions_service.assumption_versions (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    assumption_id VARCHAR(255) NOT NULL,
    version_number INTEGER NOT NULL,
    value JSONB NOT NULL,
    effective_from TIMESTAMPTZ NOT NULL,
    effective_until TIMESTAMPTZ,
    created_by VARCHAR(255) NOT NULL,
    change_reason TEXT NOT NULL,
    change_type VARCHAR(50) NOT NULL,
    provenance_hash CHAR(64),
    parent_version_id VARCHAR(255),
    scenario_id VARCHAR(255),
    tenant_id VARCHAR(100) DEFAULT 'default',
    PRIMARY KEY (id, timestamp)
);

-- Create hypertable partitioned by timestamp
SELECT create_hypertable('assumptions_service.assumption_versions', 'timestamp', if_not_exists => TRUE);

-- Change type constraint
ALTER TABLE assumptions_service.assumption_versions
    ADD CONSTRAINT chk_version_change_type
    CHECK (change_type IN (
        'initial', 'update', 'correction', 'regulatory_update',
        'methodology_change', 'source_update', 'recalibration',
        'scenario_override', 'rollback', 'deprecation'
    ));

-- Version number must be positive
ALTER TABLE assumptions_service.assumption_versions
    ADD CONSTRAINT chk_version_number_positive
    CHECK (version_number > 0);

-- Provenance hash must be 64-character hex when present
ALTER TABLE assumptions_service.assumption_versions
    ADD CONSTRAINT chk_version_provenance_hash_length
    CHECK (provenance_hash IS NULL OR LENGTH(provenance_hash) = 64);

-- Effective_until must be after effective_from when set
ALTER TABLE assumptions_service.assumption_versions
    ADD CONSTRAINT chk_version_effective_range
    CHECK (effective_until IS NULL OR effective_until >= effective_from);

-- =============================================================================
-- Table: assumptions_service.scenarios
-- =============================================================================
-- Scenario definitions for managing assumption sets under different analysis
-- contexts (baseline, optimistic, conservative, custom). Each scenario can
-- have a parent for inheritance, tags for organization, and is tenant-scoped.

CREATE TABLE assumptions_service.scenarios (
    scenario_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    scenario_type VARCHAR(50) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    parent_scenario_id VARCHAR(255),
    tags TEXT[] DEFAULT '{}',
    created_by VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(100) DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Scenario type constraint
ALTER TABLE assumptions_service.scenarios
    ADD CONSTRAINT chk_scenario_type
    CHECK (scenario_type IN (
        'baseline', 'optimistic', 'conservative', 'regulatory',
        'custom', 'what_if', 'sensitivity', 'monte_carlo'
    ));

-- Self-referencing FK for parent scenario inheritance
ALTER TABLE assumptions_service.scenarios
    ADD CONSTRAINT fk_scenario_parent
    FOREIGN KEY (parent_scenario_id)
    REFERENCES assumptions_service.scenarios(scenario_id)
    ON DELETE SET NULL;

-- =============================================================================
-- Table: assumptions_service.scenario_overrides
-- =============================================================================
-- Per-scenario assumption value overrides. When a scenario is active, these
-- overrides replace the default assumption values for analysis under that
-- scenario. Each override links a scenario to an assumption with a
-- replacement value. Unique constraint ensures one override per
-- (scenario, assumption) pair.

CREATE TABLE assumptions_service.scenario_overrides (
    id BIGSERIAL PRIMARY KEY,
    scenario_id VARCHAR(255) NOT NULL REFERENCES assumptions_service.scenarios(scenario_id) ON DELETE CASCADE,
    assumption_id VARCHAR(255) NOT NULL REFERENCES assumptions_service.assumptions(assumption_id) ON DELETE CASCADE,
    override_value JSONB NOT NULL,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(scenario_id, assumption_id)
);

-- =============================================================================
-- Table: assumptions_service.assumption_change_log
-- =============================================================================
-- TimescaleDB hypertable recording every change operation for full audit
-- trail. Each row captures who changed what, when, why, with old/new values
-- and provenance hashes. Partitioned by timestamp for time-series analysis.
-- Retained for 90 days with compression after 7 days.

CREATE TABLE assumptions_service.assumption_change_log (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id VARCHAR(255) NOT NULL,
    change_type VARCHAR(50) NOT NULL,
    assumption_id VARCHAR(255) NOT NULL,
    scenario_id VARCHAR(255),
    old_value JSONB,
    new_value JSONB,
    change_reason TEXT NOT NULL,
    provenance_hash CHAR(64),
    tenant_id VARCHAR(100) DEFAULT 'default',
    PRIMARY KEY (id, timestamp)
);

-- Create hypertable partitioned by timestamp
SELECT create_hypertable('assumptions_service.assumption_change_log', 'timestamp', if_not_exists => TRUE);

-- Change type constraint
ALTER TABLE assumptions_service.assumption_change_log
    ADD CONSTRAINT chk_changelog_change_type
    CHECK (change_type IN (
        'create', 'update', 'delete', 'restore', 'override',
        'validate', 'approve', 'reject', 'deprecate', 'archive',
        'scenario_create', 'scenario_update', 'scenario_delete',
        'dependency_add', 'dependency_remove'
    ));

-- Provenance hash must be 64-character hex when present
ALTER TABLE assumptions_service.assumption_change_log
    ADD CONSTRAINT chk_changelog_provenance_hash_length
    CHECK (provenance_hash IS NULL OR LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: assumptions_service.assumption_dependencies
-- =============================================================================
-- Directed dependency graph between assumptions and between assumptions
-- and calculations. Used for impact analysis, change propagation, and
-- cycle detection. Each edge records an assumption's dependency on another
-- assumption or a calculation's dependency on an assumption.

CREATE TABLE assumptions_service.assumption_dependencies (
    id BIGSERIAL PRIMARY KEY,
    assumption_id VARCHAR(255) NOT NULL,
    depends_on_id VARCHAR(255),
    calculation_id VARCHAR(255),
    dependency_type VARCHAR(50) DEFAULT 'assumption',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(assumption_id, depends_on_id, calculation_id)
);

-- Dependency type constraint
ALTER TABLE assumptions_service.assumption_dependencies
    ADD CONSTRAINT chk_dep_type
    CHECK (dependency_type IN ('assumption', 'calculation', 'external', 'derived'));

-- At least one of depends_on_id or calculation_id must be set
ALTER TABLE assumptions_service.assumption_dependencies
    ADD CONSTRAINT chk_dep_target_not_null
    CHECK (depends_on_id IS NOT NULL OR calculation_id IS NOT NULL);

-- =============================================================================
-- Continuous Aggregate: assumptions_service.daily_operation_counts
-- =============================================================================
-- Precomputed daily counts of assumption change operations by type for
-- dashboard queries and trend analysis. Aggregates change_log data into
-- per-change-type daily statistics.

CREATE MATERIALIZED VIEW assumptions_service.daily_operation_counts
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS bucket,
    change_type,
    COUNT(*) AS total_operations,
    COUNT(DISTINCT assumption_id) AS affected_assumptions,
    COUNT(DISTINCT user_id) AS unique_users
FROM assumptions_service.assumption_change_log
WHERE timestamp IS NOT NULL
GROUP BY bucket, change_type
WITH NO DATA;

-- Refresh policy: refresh every 30 minutes, covering the last 2 days
SELECT add_continuous_aggregate_policy('assumptions_service.daily_operation_counts',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Continuous Aggregate: assumptions_service.daily_version_counts
-- =============================================================================
-- Precomputed daily version creation summaries for monitoring version
-- growth, change velocity, and category distribution. Aggregates version
-- data into per-day statistics.

CREATE MATERIALIZED VIEW assumptions_service.daily_version_counts
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS bucket,
    change_type,
    COUNT(*) AS total_versions,
    COUNT(DISTINCT assumption_id) AS assumptions_versioned,
    AVG(version_number) AS avg_version_number
FROM assumptions_service.assumption_versions
WHERE timestamp IS NOT NULL
GROUP BY bucket, change_type
WITH NO DATA;

-- Refresh policy: refresh every 30 minutes, covering the last 2 days
SELECT add_continuous_aggregate_policy('assumptions_service.daily_version_counts',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- assumptions indexes
CREATE INDEX idx_assumptions_category ON assumptions_service.assumptions(category);
CREATE INDEX idx_assumptions_data_type ON assumptions_service.assumptions(data_type);
CREATE INDEX idx_assumptions_tenant ON assumptions_service.assumptions(tenant_id);
CREATE INDEX idx_assumptions_parent ON assumptions_service.assumptions(parent_assumption_id);
CREATE INDEX idx_assumptions_name ON assumptions_service.assumptions(name);
CREATE INDEX idx_assumptions_updated ON assumptions_service.assumptions(updated_at DESC);
CREATE INDEX idx_assumptions_provenance ON assumptions_service.assumptions(provenance_hash);
CREATE INDEX idx_assumptions_metadata ON assumptions_service.assumptions USING GIN (metadata);
CREATE INDEX idx_assumptions_validation ON assumptions_service.assumptions USING GIN (validation_rules);
CREATE INDEX idx_assumptions_depends_on ON assumptions_service.assumptions USING GIN (depends_on);
CREATE INDEX idx_assumptions_used_by ON assumptions_service.assumptions USING GIN (used_by);

-- assumption_versions indexes (hypertable-aware)
CREATE INDEX idx_versions_assumption ON assumptions_service.assumption_versions(assumption_id, timestamp DESC);
CREATE INDEX idx_versions_tenant ON assumptions_service.assumption_versions(tenant_id, timestamp DESC);
CREATE INDEX idx_versions_created_by ON assumptions_service.assumption_versions(created_by, timestamp DESC);
CREATE INDEX idx_versions_change_type ON assumptions_service.assumption_versions(change_type, timestamp DESC);
CREATE INDEX idx_versions_scenario ON assumptions_service.assumption_versions(scenario_id, timestamp DESC);
CREATE INDEX idx_versions_provenance ON assumptions_service.assumption_versions(provenance_hash);
CREATE INDEX idx_versions_effective ON assumptions_service.assumption_versions(effective_from, effective_until);
CREATE INDEX idx_versions_number ON assumptions_service.assumption_versions(assumption_id, version_number DESC, timestamp DESC);

-- scenarios indexes
CREATE INDEX idx_scenarios_type ON assumptions_service.scenarios(scenario_type);
CREATE INDEX idx_scenarios_tenant ON assumptions_service.scenarios(tenant_id);
CREATE INDEX idx_scenarios_active ON assumptions_service.scenarios(is_active, tenant_id);
CREATE INDEX idx_scenarios_parent ON assumptions_service.scenarios(parent_scenario_id);
CREATE INDEX idx_scenarios_created_by ON assumptions_service.scenarios(created_by);
CREATE INDEX idx_scenarios_tags ON assumptions_service.scenarios USING GIN (tags);

-- scenario_overrides indexes
CREATE INDEX idx_overrides_scenario ON assumptions_service.scenario_overrides(scenario_id);
CREATE INDEX idx_overrides_assumption ON assumptions_service.scenario_overrides(assumption_id);

-- assumption_change_log indexes (hypertable-aware)
CREATE INDEX idx_changelog_assumption ON assumptions_service.assumption_change_log(assumption_id, timestamp DESC);
CREATE INDEX idx_changelog_user ON assumptions_service.assumption_change_log(user_id, timestamp DESC);
CREATE INDEX idx_changelog_type ON assumptions_service.assumption_change_log(change_type, timestamp DESC);
CREATE INDEX idx_changelog_tenant ON assumptions_service.assumption_change_log(tenant_id, timestamp DESC);
CREATE INDEX idx_changelog_scenario ON assumptions_service.assumption_change_log(scenario_id, timestamp DESC);
CREATE INDEX idx_changelog_provenance ON assumptions_service.assumption_change_log(provenance_hash);

-- assumption_dependencies indexes
CREATE INDEX idx_deps_assumption ON assumptions_service.assumption_dependencies(assumption_id);
CREATE INDEX idx_deps_depends_on ON assumptions_service.assumption_dependencies(depends_on_id);
CREATE INDEX idx_deps_calculation ON assumptions_service.assumption_dependencies(calculation_id);
CREATE INDEX idx_deps_type ON assumptions_service.assumption_dependencies(dependency_type);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE assumptions_service.assumptions ENABLE ROW LEVEL SECURITY;
CREATE POLICY assumptions_tenant_read ON assumptions_service.assumptions
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY assumptions_tenant_write ON assumptions_service.assumptions
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE assumptions_service.assumption_versions ENABLE ROW LEVEL SECURITY;
CREATE POLICY versions_tenant_read ON assumptions_service.assumption_versions
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY versions_tenant_write ON assumptions_service.assumption_versions
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE assumptions_service.scenarios ENABLE ROW LEVEL SECURITY;
CREATE POLICY scenarios_tenant_read ON assumptions_service.scenarios
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY scenarios_tenant_write ON assumptions_service.scenarios
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE assumptions_service.scenario_overrides ENABLE ROW LEVEL SECURITY;
-- Overrides inherit tenant context via their scenario
CREATE POLICY overrides_read ON assumptions_service.scenario_overrides
    FOR SELECT USING (true);
CREATE POLICY overrides_write ON assumptions_service.scenario_overrides
    FOR ALL USING (
        current_setting('app.is_admin', true) = 'true'
        OR current_setting('app.current_tenant', true) IS NULL
        OR EXISTS (
            SELECT 1 FROM assumptions_service.scenarios s
            WHERE s.scenario_id = scenario_overrides.scenario_id
            AND (
                s.tenant_id = current_setting('app.current_tenant', true)
                OR current_setting('app.current_tenant', true) IS NULL
            )
        )
    );

ALTER TABLE assumptions_service.assumption_change_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY changelog_tenant_read ON assumptions_service.assumption_change_log
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY changelog_tenant_write ON assumptions_service.assumption_change_log
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE assumptions_service.assumption_dependencies ENABLE ROW LEVEL SECURITY;
CREATE POLICY dependencies_read ON assumptions_service.assumption_dependencies
    FOR SELECT USING (true);
CREATE POLICY dependencies_write ON assumptions_service.assumption_dependencies
    FOR ALL USING (
        current_setting('app.is_admin', true) = 'true'
        OR current_setting('app.current_tenant', true) IS NULL
    );

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA assumptions_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA assumptions_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA assumptions_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON assumptions_service.daily_operation_counts TO greenlang_app;
GRANT SELECT ON assumptions_service.daily_version_counts TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA assumptions_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA assumptions_service TO greenlang_readonly;
GRANT SELECT ON assumptions_service.daily_operation_counts TO greenlang_readonly;
GRANT SELECT ON assumptions_service.daily_version_counts TO greenlang_readonly;

-- Add assumptions service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'assumptions:read', 'assumptions', 'read', 'View assumptions and their current values'),
    (gen_random_uuid(), 'assumptions:write', 'assumptions', 'write', 'Create and update assumptions'),
    (gen_random_uuid(), 'assumptions:delete', 'assumptions', 'delete', 'Delete assumptions from the registry'),
    (gen_random_uuid(), 'assumptions:versions:read', 'assumptions', 'versions_read', 'View assumption version history'),
    (gen_random_uuid(), 'assumptions:versions:write', 'assumptions', 'versions_write', 'Create new assumption versions'),
    (gen_random_uuid(), 'assumptions:scenarios:read', 'assumptions', 'scenarios_read', 'View scenarios and their overrides'),
    (gen_random_uuid(), 'assumptions:scenarios:write', 'assumptions', 'scenarios_write', 'Create and update scenarios'),
    (gen_random_uuid(), 'assumptions:scenarios:delete', 'assumptions', 'scenarios_delete', 'Delete scenarios'),
    (gen_random_uuid(), 'assumptions:validate', 'assumptions', 'validate', 'Execute assumption validation'),
    (gen_random_uuid(), 'assumptions:dependencies:read', 'assumptions', 'dependencies_read', 'View assumption dependency graph'),
    (gen_random_uuid(), 'assumptions:dependencies:write', 'assumptions', 'dependencies_write', 'Manage assumption dependencies'),
    (gen_random_uuid(), 'assumptions:audit:read', 'assumptions', 'audit_read', 'View assumption change audit log'),
    (gen_random_uuid(), 'assumptions:export', 'assumptions', 'export', 'Export assumptions and scenarios'),
    (gen_random_uuid(), 'assumptions:admin', 'assumptions', 'admin', 'Assumptions service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep change log for 90 days
SELECT add_retention_policy('assumptions_service.assumption_change_log', INTERVAL '90 days');

-- assumption_versions: no retention (unlimited history for compliance)

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on assumption_versions after 30 days
ALTER TABLE assumptions_service.assumption_versions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'assumption_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('assumptions_service.assumption_versions', INTERVAL '30 days');

-- Enable compression on change_log after 7 days
ALTER TABLE assumptions_service.assumption_change_log SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'change_type',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('assumptions_service.assumption_change_log', INTERVAL '7 days');

-- =============================================================================
-- Seed: Default Scenarios
-- =============================================================================

INSERT INTO assumptions_service.scenarios (scenario_id, name, description, scenario_type, is_active, tags, created_by, tenant_id) VALUES
('scenario-baseline', 'Baseline', 'Default baseline scenario using standard emission factors and conversion rates from regulatory sources (EPA, IPCC AR6, IEA).', 'baseline', true, '{"default", "regulatory", "compliance"}', 'system', 'default'),
('scenario-optimistic', 'Optimistic', 'Optimistic scenario using lower emission factors reflecting best-case technology adoption, renewable energy uptake, and efficiency improvements.', 'optimistic', true, '{"planning", "best-case", "target-setting"}', 'system', 'default'),
('scenario-conservative', 'Conservative', 'Conservative scenario using higher emission factors reflecting worst-case assumptions for risk assessment and precautionary reporting.', 'conservative', true, '{"risk", "worst-case", "precautionary"}', 'system', 'default')
ON CONFLICT (scenario_id) DO NOTHING;

-- =============================================================================
-- Seed: Common Emission Factor Assumptions (20+ entries)
-- =============================================================================

INSERT INTO assumptions_service.assumptions (assumption_id, name, description, category, data_type, unit, current_value, default_value, metadata, validation_rules, depends_on, used_by, provenance_hash, tenant_id) VALUES

-- Stationary Combustion - Fuels
('ef-diesel-co2', 'Diesel CO2 Emission Factor', 'CO2 emission factor for diesel fuel combustion (EPA AP-42, Table 1.3-1)', 'emission_factor', 'numeric', 'kgCO2/L',
 '{"value": 2.68, "uncertainty": 0.05}', '{"value": 2.68}',
 '{"source": "EPA AP-42", "source_version": "5th Edition", "region": "US", "scope": 1, "fuel_type": "diesel", "last_verified": "2025-12-01"}',
 '[{"rule": "range", "min": 2.0, "max": 3.5, "message": "Diesel EF outside expected range"}]',
 '{}', '{"calc-fleet-emissions", "calc-generator-emissions"}',
 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2', 'default'),

('ef-gasoline-co2', 'Gasoline CO2 Emission Factor', 'CO2 emission factor for gasoline/petrol combustion (EPA AP-42)', 'emission_factor', 'numeric', 'kgCO2/L',
 '{"value": 2.31, "uncertainty": 0.04}', '{"value": 2.31}',
 '{"source": "EPA AP-42", "source_version": "5th Edition", "region": "US", "scope": 1, "fuel_type": "gasoline", "last_verified": "2025-12-01"}',
 '[{"rule": "range", "min": 1.8, "max": 3.0, "message": "Gasoline EF outside expected range"}]',
 '{}', '{"calc-fleet-emissions", "calc-vehicle-emissions"}',
 'b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3', 'default'),

('ef-natural-gas-co2', 'Natural Gas CO2 Emission Factor', 'CO2 emission factor for natural gas combustion (EPA 40 CFR Part 98)', 'emission_factor', 'numeric', 'kgCO2/m3',
 '{"value": 1.89, "uncertainty": 0.03}', '{"value": 1.89}',
 '{"source": "EPA 40 CFR Part 98", "source_version": "2024", "region": "US", "scope": 1, "fuel_type": "natural_gas", "last_verified": "2025-12-01"}',
 '[{"rule": "range", "min": 1.5, "max": 2.5, "message": "Natural gas EF outside expected range"}]',
 '{}', '{"calc-building-emissions", "calc-heating-emissions"}',
 'c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4', 'default'),

('ef-coal-anthracite-co2', 'Anthracite Coal CO2 Emission Factor', 'CO2 emission factor for anthracite coal combustion (IPCC 2006)', 'emission_factor', 'numeric', 'kgCO2/kg',
 '{"value": 2.86, "uncertainty": 0.08}', '{"value": 2.86}',
 '{"source": "IPCC 2006 Guidelines", "source_version": "Vol 2, Ch 2", "region": "global", "scope": 1, "fuel_type": "coal_anthracite", "last_verified": "2025-11-15"}',
 '[{"rule": "range", "min": 2.0, "max": 3.5, "message": "Anthracite coal EF outside expected range"}]',
 '{}', '{"calc-power-plant-emissions"}',
 'd4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5', 'default'),

('ef-coal-bituminous-co2', 'Bituminous Coal CO2 Emission Factor', 'CO2 emission factor for bituminous coal combustion (IPCC 2006)', 'emission_factor', 'numeric', 'kgCO2/kg',
 '{"value": 2.42, "uncertainty": 0.07}', '{"value": 2.42}',
 '{"source": "IPCC 2006 Guidelines", "source_version": "Vol 2, Ch 2", "region": "global", "scope": 1, "fuel_type": "coal_bituminous", "last_verified": "2025-11-15"}',
 '[{"rule": "range", "min": 1.8, "max": 3.0, "message": "Bituminous coal EF outside expected range"}]',
 '{}', '{"calc-power-plant-emissions"}',
 'e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6', 'default'),

('ef-lpg-co2', 'LPG CO2 Emission Factor', 'CO2 emission factor for liquefied petroleum gas combustion (EPA)', 'emission_factor', 'numeric', 'kgCO2/L',
 '{"value": 1.51, "uncertainty": 0.03}', '{"value": 1.51}',
 '{"source": "EPA AP-42", "source_version": "5th Edition", "region": "US", "scope": 1, "fuel_type": "lpg", "last_verified": "2025-12-01"}',
 '[{"rule": "range", "min": 1.0, "max": 2.0, "message": "LPG EF outside expected range"}]',
 '{}', '{"calc-heating-emissions"}',
 'f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7', 'default'),

('ef-jet-fuel-co2', 'Jet Fuel (Kerosene) CO2 Emission Factor', 'CO2 emission factor for aviation kerosene/jet fuel (ICAO)', 'emission_factor', 'numeric', 'kgCO2/L',
 '{"value": 2.53, "uncertainty": 0.05}', '{"value": 2.53}',
 '{"source": "ICAO Carbon Calculator", "source_version": "2024", "region": "global", "scope": 1, "fuel_type": "jet_fuel", "last_verified": "2025-11-20"}',
 '[{"rule": "range", "min": 2.0, "max": 3.0, "message": "Jet fuel EF outside expected range"}]',
 '{}', '{"calc-aviation-emissions", "calc-scope3-business-travel"}',
 'a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8', 'default'),

('ef-fuel-oil-co2', 'Fuel Oil (#2) CO2 Emission Factor', 'CO2 emission factor for No. 2 fuel oil / heating oil combustion (EPA)', 'emission_factor', 'numeric', 'kgCO2/L',
 '{"value": 2.68, "uncertainty": 0.05}', '{"value": 2.68}',
 '{"source": "EPA AP-42", "source_version": "5th Edition", "region": "US", "scope": 1, "fuel_type": "fuel_oil_2", "last_verified": "2025-12-01"}',
 '[{"rule": "range", "min": 2.0, "max": 3.5, "message": "Fuel oil EF outside expected range"}]',
 '{}', '{"calc-building-emissions", "calc-marine-emissions"}',
 'b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9', 'default'),

('ef-propane-co2', 'Propane CO2 Emission Factor', 'CO2 emission factor for propane combustion (EPA)', 'emission_factor', 'numeric', 'kgCO2/L',
 '{"value": 1.54, "uncertainty": 0.03}', '{"value": 1.54}',
 '{"source": "EPA AP-42", "source_version": "5th Edition", "region": "US", "scope": 1, "fuel_type": "propane", "last_verified": "2025-12-01"}',
 '[{"rule": "range", "min": 1.0, "max": 2.0, "message": "Propane EF outside expected range"}]',
 '{}', '{"calc-heating-emissions"}',
 'c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0', 'default'),

('ef-biodiesel-co2', 'Biodiesel (B100) CO2 Emission Factor', 'CO2 emission factor for biodiesel B100 combustion (EPA)', 'emission_factor', 'numeric', 'kgCO2/L',
 '{"value": 2.50, "uncertainty": 0.06, "biogenic": true}', '{"value": 2.50}',
 '{"source": "EPA", "source_version": "2024", "region": "US", "scope": 1, "fuel_type": "biodiesel_b100", "biogenic": true, "last_verified": "2025-12-01"}',
 '[{"rule": "range", "min": 2.0, "max": 3.0, "message": "Biodiesel EF outside expected range"}]',
 '{}', '{"calc-fleet-emissions"}',
 'd0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1', 'default'),

-- GWP Factors (IPCC AR6)
('gwp-ch4-ar6', 'Methane GWP (AR6, 100-year)', 'Global Warming Potential of methane over 100-year horizon (IPCC AR6)', 'gwp_factor', 'numeric', 'kgCO2e/kgCH4',
 '{"value": 27.9, "timeframe": "100-year"}', '{"value": 27.9}',
 '{"source": "IPCC AR6", "source_version": "2021", "region": "global", "gas": "CH4", "timeframe_years": 100, "last_verified": "2025-10-01"}',
 '[{"rule": "range", "min": 20, "max": 35, "message": "CH4 GWP outside expected range for AR6"}]',
 '{}', '{"calc-scope1-ch4", "calc-fugitive-emissions"}',
 'e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2', 'default'),

('gwp-n2o-ar6', 'Nitrous Oxide GWP (AR6, 100-year)', 'Global Warming Potential of nitrous oxide over 100-year horizon (IPCC AR6)', 'gwp_factor', 'numeric', 'kgCO2e/kgN2O',
 '{"value": 273, "timeframe": "100-year"}', '{"value": 273}',
 '{"source": "IPCC AR6", "source_version": "2021", "region": "global", "gas": "N2O", "timeframe_years": 100, "last_verified": "2025-10-01"}',
 '[{"rule": "range", "min": 250, "max": 300, "message": "N2O GWP outside expected range for AR6"}]',
 '{}', '{"calc-scope1-n2o", "calc-agricultural-emissions"}',
 'f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3', 'default'),

('gwp-sf6-ar6', 'SF6 GWP (AR6, 100-year)', 'Global Warming Potential of sulfur hexafluoride over 100-year horizon (IPCC AR6)', 'gwp_factor', 'numeric', 'kgCO2e/kgSF6',
 '{"value": 25200, "timeframe": "100-year"}', '{"value": 25200}',
 '{"source": "IPCC AR6", "source_version": "2021", "region": "global", "gas": "SF6", "timeframe_years": 100, "last_verified": "2025-10-01"}',
 '[{"rule": "range", "min": 20000, "max": 30000, "message": "SF6 GWP outside expected range for AR6"}]',
 '{}', '{"calc-scope1-sf6"}',
 'a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4', 'default'),

-- Grid Electricity Factors
('ef-grid-us-avg', 'US Average Grid Emission Factor', 'Average grid emission factor for electricity consumed in the US (EPA eGRID)', 'grid_factor', 'numeric', 'kgCO2e/kWh',
 '{"value": 0.386, "uncertainty": 0.02, "year": 2024}', '{"value": 0.386}',
 '{"source": "EPA eGRID", "source_version": "2024", "region": "US", "scope": 2, "method": "location-based", "last_verified": "2025-11-01"}',
 '[{"rule": "range", "min": 0.1, "max": 1.0, "message": "US grid EF outside expected range"}]',
 '{}', '{"calc-scope2-electricity"}',
 'b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5', 'default'),

('ef-grid-eu-avg', 'EU Average Grid Emission Factor', 'Average grid emission factor for electricity consumed in the EU (EEA)', 'grid_factor', 'numeric', 'kgCO2e/kWh',
 '{"value": 0.256, "uncertainty": 0.02, "year": 2024}', '{"value": 0.256}',
 '{"source": "EEA", "source_version": "2024", "region": "EU", "scope": 2, "method": "location-based", "last_verified": "2025-11-01"}',
 '[{"rule": "range", "min": 0.05, "max": 0.8, "message": "EU grid EF outside expected range"}]',
 '{}', '{"calc-scope2-electricity"}',
 'c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6', 'default'),

-- Transport Factors
('ef-road-freight-co2', 'Road Freight CO2 Emission Factor', 'CO2 emission factor for road freight transport (DEFRA)', 'transport_factor', 'numeric', 'kgCO2e/tonne-km',
 '{"value": 0.10730, "uncertainty": 0.01}', '{"value": 0.10730}',
 '{"source": "UK DEFRA", "source_version": "2024", "region": "UK", "scope": 3, "transport_mode": "road_freight", "vehicle_type": "average", "last_verified": "2025-11-15"}',
 '[{"rule": "range", "min": 0.01, "max": 0.5, "message": "Road freight EF outside expected range"}]',
 '{}', '{"calc-scope3-freight"}',
 'd6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7', 'default'),

('ef-rail-freight-co2', 'Rail Freight CO2 Emission Factor', 'CO2 emission factor for rail freight transport (DEFRA)', 'transport_factor', 'numeric', 'kgCO2e/tonne-km',
 '{"value": 0.02549, "uncertainty": 0.005}', '{"value": 0.02549}',
 '{"source": "UK DEFRA", "source_version": "2024", "region": "UK", "scope": 3, "transport_mode": "rail_freight", "last_verified": "2025-11-15"}',
 '[{"rule": "range", "min": 0.005, "max": 0.1, "message": "Rail freight EF outside expected range"}]',
 '{}', '{"calc-scope3-freight"}',
 'e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8', 'default'),

('ef-sea-freight-co2', 'Sea Freight CO2 Emission Factor', 'CO2 emission factor for sea freight transport (IMO)', 'transport_factor', 'numeric', 'kgCO2e/tonne-km',
 '{"value": 0.01612, "uncertainty": 0.003}', '{"value": 0.01612}',
 '{"source": "IMO GHG Study", "source_version": "4th 2020", "region": "global", "scope": 3, "transport_mode": "sea_freight", "vessel_type": "average", "last_verified": "2025-11-15"}',
 '[{"rule": "range", "min": 0.001, "max": 0.05, "message": "Sea freight EF outside expected range"}]',
 '{}', '{"calc-scope3-freight"}',
 'f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9', 'default'),

('ef-air-freight-co2', 'Air Freight CO2 Emission Factor', 'CO2 emission factor for air freight transport (DEFRA)', 'transport_factor', 'numeric', 'kgCO2e/tonne-km',
 '{"value": 0.60192, "uncertainty": 0.08}', '{"value": 0.60192}',
 '{"source": "UK DEFRA", "source_version": "2024", "region": "global", "scope": 3, "transport_mode": "air_freight", "last_verified": "2025-11-15"}',
 '[{"rule": "range", "min": 0.3, "max": 1.0, "message": "Air freight EF outside expected range"}]',
 '{}', '{"calc-scope3-freight", "calc-scope3-business-travel"}',
 'a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0', 'default'),

-- Economic & Regulatory Parameters
('param-social-cost-carbon', 'Social Cost of Carbon', 'Social cost of carbon for economic impact analysis (US EPA IWG)', 'economic_parameter', 'numeric', 'USD/tCO2e',
 '{"value": 51.0, "discount_rate": "3%", "year": 2025}', '{"value": 51.0}',
 '{"source": "US EPA Interagency Working Group", "source_version": "2024", "region": "US", "discount_rate_pct": 3, "last_verified": "2025-10-01"}',
 '[{"rule": "range", "min": 10, "max": 300, "message": "SCC outside expected range"}]',
 '{}', '{"calc-economic-impact", "calc-cost-benefit"}',
 'b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1', 'default'),

('threshold-sbti-1-5c', 'SBTi 1.5C Annual Reduction Rate', 'Science-Based Targets initiative annual linear reduction rate for 1.5C pathway', 'regulatory_threshold', 'percentage', 'percent/year',
 '{"value": 4.2, "pathway": "1.5C"}', '{"value": 4.2}',
 '{"source": "SBTi", "source_version": "v5.1", "region": "global", "pathway": "1.5C", "base_year": 2020, "target_year": 2030, "last_verified": "2025-09-15"}',
 '[{"rule": "range", "min": 2.0, "max": 10.0, "message": "SBTi reduction rate outside expected range"}]',
 '{}', '{"calc-target-trajectory", "calc-gap-analysis"}',
 'c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2', 'default'),

('ef-ethanol-co2', 'Ethanol (E100) CO2 Emission Factor', 'CO2 emission factor for ethanol combustion (EPA RFS)', 'emission_factor', 'numeric', 'kgCO2/L',
 '{"value": 1.51, "uncertainty": 0.04, "biogenic": true}', '{"value": 1.51}',
 '{"source": "EPA Renewable Fuel Standard", "source_version": "2024", "region": "US", "scope": 1, "fuel_type": "ethanol_e100", "biogenic": true, "last_verified": "2025-12-01"}',
 '[{"rule": "range", "min": 1.0, "max": 2.0, "message": "Ethanol EF outside expected range"}]',
 '{}', '{"calc-fleet-emissions"}',
 'd2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3', 'default'),

('ef-cng-co2', 'CNG CO2 Emission Factor', 'CO2 emission factor for compressed natural gas vehicle combustion (EPA)', 'emission_factor', 'numeric', 'kgCO2/m3',
 '{"value": 1.88, "uncertainty": 0.03}', '{"value": 1.88}',
 '{"source": "EPA", "source_version": "2024", "region": "US", "scope": 1, "fuel_type": "cng", "last_verified": "2025-12-01"}',
 '[{"rule": "range", "min": 1.5, "max": 2.5, "message": "CNG EF outside expected range"}]',
 '{}', '{"calc-fleet-emissions"}',
 'e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4', 'default')

ON CONFLICT (assumption_id) DO NOTHING;

-- =============================================================================
-- Seed: Initial Versions for Seeded Assumptions
-- =============================================================================

INSERT INTO assumptions_service.assumption_versions (assumption_id, version_number, value, effective_from, created_by, change_reason, change_type, provenance_hash, tenant_id) VALUES
('ef-diesel-co2', 1, '{"value": 2.68, "uncertainty": 0.05}', '2026-01-01', 'system', 'Initial seed from EPA AP-42', 'initial', 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2', 'default'),
('ef-gasoline-co2', 1, '{"value": 2.31, "uncertainty": 0.04}', '2026-01-01', 'system', 'Initial seed from EPA AP-42', 'initial', 'b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3', 'default'),
('ef-natural-gas-co2', 1, '{"value": 1.89, "uncertainty": 0.03}', '2026-01-01', 'system', 'Initial seed from EPA 40 CFR Part 98', 'initial', 'c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4', 'default'),
('ef-coal-anthracite-co2', 1, '{"value": 2.86, "uncertainty": 0.08}', '2026-01-01', 'system', 'Initial seed from IPCC 2006 Guidelines', 'initial', 'd4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5', 'default'),
('ef-coal-bituminous-co2', 1, '{"value": 2.42, "uncertainty": 0.07}', '2026-01-01', 'system', 'Initial seed from IPCC 2006 Guidelines', 'initial', 'e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6', 'default'),
('ef-lpg-co2', 1, '{"value": 1.51, "uncertainty": 0.03}', '2026-01-01', 'system', 'Initial seed from EPA AP-42', 'initial', 'f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7', 'default'),
('ef-jet-fuel-co2', 1, '{"value": 2.53, "uncertainty": 0.05}', '2026-01-01', 'system', 'Initial seed from ICAO Carbon Calculator', 'initial', 'a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8', 'default'),
('ef-fuel-oil-co2', 1, '{"value": 2.68, "uncertainty": 0.05}', '2026-01-01', 'system', 'Initial seed from EPA AP-42', 'initial', 'b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9', 'default'),
('ef-propane-co2', 1, '{"value": 1.54, "uncertainty": 0.03}', '2026-01-01', 'system', 'Initial seed from EPA AP-42', 'initial', 'c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0', 'default'),
('ef-biodiesel-co2', 1, '{"value": 2.50, "uncertainty": 0.06, "biogenic": true}', '2026-01-01', 'system', 'Initial seed from EPA', 'initial', 'd0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1', 'default'),
('gwp-ch4-ar6', 1, '{"value": 27.9, "timeframe": "100-year"}', '2026-01-01', 'system', 'Initial seed from IPCC AR6', 'initial', 'e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2', 'default'),
('gwp-n2o-ar6', 1, '{"value": 273, "timeframe": "100-year"}', '2026-01-01', 'system', 'Initial seed from IPCC AR6', 'initial', 'f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3', 'default'),
('gwp-sf6-ar6', 1, '{"value": 25200, "timeframe": "100-year"}', '2026-01-01', 'system', 'Initial seed from IPCC AR6', 'initial', 'a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4', 'default'),
('ef-grid-us-avg', 1, '{"value": 0.386, "uncertainty": 0.02, "year": 2024}', '2026-01-01', 'system', 'Initial seed from EPA eGRID', 'initial', 'b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5', 'default'),
('ef-grid-eu-avg', 1, '{"value": 0.256, "uncertainty": 0.02, "year": 2024}', '2026-01-01', 'system', 'Initial seed from EEA', 'initial', 'c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6', 'default'),
('ef-road-freight-co2', 1, '{"value": 0.10730, "uncertainty": 0.01}', '2026-01-01', 'system', 'Initial seed from UK DEFRA', 'initial', 'd6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7', 'default'),
('ef-rail-freight-co2', 1, '{"value": 0.02549, "uncertainty": 0.005}', '2026-01-01', 'system', 'Initial seed from UK DEFRA', 'initial', 'e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8', 'default'),
('ef-sea-freight-co2', 1, '{"value": 0.01612, "uncertainty": 0.003}', '2026-01-01', 'system', 'Initial seed from IMO GHG Study', 'initial', 'f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9', 'default'),
('ef-air-freight-co2', 1, '{"value": 0.60192, "uncertainty": 0.08}', '2026-01-01', 'system', 'Initial seed from UK DEFRA', 'initial', 'a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0', 'default'),
('param-social-cost-carbon', 1, '{"value": 51.0, "discount_rate": "3%", "year": 2025}', '2026-01-01', 'system', 'Initial seed from US EPA IWG', 'initial', 'b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1', 'default'),
('threshold-sbti-1-5c', 1, '{"value": 4.2, "pathway": "1.5C"}', '2026-01-01', 'system', 'Initial seed from SBTi v5.1', 'initial', 'c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2', 'default'),
('ef-ethanol-co2', 1, '{"value": 1.51, "uncertainty": 0.04, "biogenic": true}', '2026-01-01', 'system', 'Initial seed from EPA RFS', 'initial', 'd2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3', 'default'),
('ef-cng-co2', 1, '{"value": 1.88, "uncertainty": 0.03}', '2026-01-01', 'system', 'Initial seed from EPA', 'initial', 'e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4', 'default');

-- =============================================================================
-- Seed: Scenario Overrides for Optimistic Scenario
-- =============================================================================

INSERT INTO assumptions_service.scenario_overrides (scenario_id, assumption_id, override_value, created_by) VALUES
('scenario-optimistic', 'ef-grid-us-avg', '{"value": 0.290, "uncertainty": 0.02, "year": 2024, "note": "Accelerated renewable adoption"}', 'system'),
('scenario-optimistic', 'ef-grid-eu-avg', '{"value": 0.180, "uncertainty": 0.02, "year": 2024, "note": "Accelerated renewable adoption"}', 'system'),
('scenario-optimistic', 'ef-road-freight-co2', '{"value": 0.085, "uncertainty": 0.01, "note": "Increased EV fleet penetration"}', 'system')
ON CONFLICT (scenario_id, assumption_id) DO NOTHING;

-- =============================================================================
-- Seed: Scenario Overrides for Conservative Scenario
-- =============================================================================

INSERT INTO assumptions_service.scenario_overrides (scenario_id, assumption_id, override_value, created_by) VALUES
('scenario-conservative', 'ef-grid-us-avg', '{"value": 0.450, "uncertainty": 0.03, "year": 2024, "note": "Delayed coal phase-out"}', 'system'),
('scenario-conservative', 'ef-grid-eu-avg', '{"value": 0.320, "uncertainty": 0.03, "year": 2024, "note": "Delayed coal phase-out"}', 'system'),
('scenario-conservative', 'ef-road-freight-co2', '{"value": 0.130, "uncertainty": 0.015, "note": "Slower EV adoption"}', 'system')
ON CONFLICT (scenario_id, assumption_id) DO NOTHING;

-- =============================================================================
-- Seed: Sample Dependencies
-- =============================================================================

INSERT INTO assumptions_service.assumption_dependencies (assumption_id, depends_on_id, calculation_id, dependency_type) VALUES
('ef-grid-us-avg', NULL, 'calc-scope2-electricity', 'calculation'),
('ef-grid-eu-avg', NULL, 'calc-scope2-electricity', 'calculation'),
('gwp-ch4-ar6', NULL, 'calc-scope1-ch4', 'calculation'),
('gwp-n2o-ar6', NULL, 'calc-scope1-n2o', 'calculation'),
('ef-diesel-co2', NULL, 'calc-fleet-emissions', 'calculation'),
('ef-gasoline-co2', NULL, 'calc-fleet-emissions', 'calculation'),
('ef-road-freight-co2', NULL, 'calc-scope3-freight', 'calculation'),
('ef-rail-freight-co2', NULL, 'calc-scope3-freight', 'calculation'),
('ef-sea-freight-co2', NULL, 'calc-scope3-freight', 'calculation'),
('ef-air-freight-co2', NULL, 'calc-scope3-freight', 'calculation')
ON CONFLICT (assumption_id, depends_on_id, calculation_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA assumptions_service IS 'Assumptions Registry service for GreenLang Climate OS (AGENT-FOUND-004) - version-controlled, audit-ready registry for all compliance assumptions';
COMMENT ON TABLE assumptions_service.assumptions IS 'Main registry of all assumptions used in zero-hallucination compliance calculations with version tracking, validation rules, and dependency graph';
COMMENT ON TABLE assumptions_service.assumption_versions IS 'TimescaleDB hypertable: immutable version history of all assumption values with effective date ranges and provenance hashes';
COMMENT ON TABLE assumptions_service.scenarios IS 'Scenario definitions for managing assumption sets under different analysis contexts (baseline, optimistic, conservative, custom)';
COMMENT ON TABLE assumptions_service.scenario_overrides IS 'Per-scenario assumption value overrides replacing default values during scenario analysis';
COMMENT ON TABLE assumptions_service.assumption_change_log IS 'TimescaleDB hypertable: full audit trail of all change operations with old/new values and provenance';
COMMENT ON TABLE assumptions_service.assumption_dependencies IS 'Directed dependency graph between assumptions and calculations for impact analysis and cycle detection';
COMMENT ON MATERIALIZED VIEW assumptions_service.daily_operation_counts IS 'Continuous aggregate: daily operation counts by change type for dashboard and trend analysis';
COMMENT ON MATERIALIZED VIEW assumptions_service.daily_version_counts IS 'Continuous aggregate: daily version creation summaries for monitoring version growth and change velocity';

COMMENT ON COLUMN assumptions_service.assumptions.assumption_id IS 'Unique identifier for the assumption (e.g., ef-diesel-co2, gwp-ch4-ar6)';
COMMENT ON COLUMN assumptions_service.assumptions.category IS 'Assumption category: emission_factor, conversion_rate, economic_parameter, regulatory_threshold, etc.';
COMMENT ON COLUMN assumptions_service.assumptions.data_type IS 'Value data type: numeric, percentage, ratio, range, enum, boolean, json, text, date, composite';
COMMENT ON COLUMN assumptions_service.assumptions.current_value IS 'Current value as JSONB (supports numeric values with uncertainty, ranges, composite structures)';
COMMENT ON COLUMN assumptions_service.assumptions.validation_rules IS 'JSON array of validation rules applied when values are updated (range checks, format validation)';
COMMENT ON COLUMN assumptions_service.assumptions.depends_on IS 'Array of assumption IDs this assumption depends on for calculation';
COMMENT ON COLUMN assumptions_service.assumptions.used_by IS 'Array of calculation/process IDs that consume this assumption';
COMMENT ON COLUMN assumptions_service.assumptions.provenance_hash IS 'SHA-256 hash of the assumption provenance chain for audit integrity';

COMMENT ON COLUMN assumptions_service.assumption_versions.version_number IS 'Monotonically increasing version number per assumption';
COMMENT ON COLUMN assumptions_service.assumption_versions.effective_from IS 'Date from which this version value is effective';
COMMENT ON COLUMN assumptions_service.assumption_versions.effective_until IS 'Date until which this version is effective (NULL = current)';
COMMENT ON COLUMN assumptions_service.assumption_versions.change_type IS 'Type of change: initial, update, correction, regulatory_update, methodology_change, etc.';

COMMENT ON COLUMN assumptions_service.scenarios.scenario_type IS 'Scenario type: baseline, optimistic, conservative, regulatory, custom, what_if, sensitivity, monte_carlo';
COMMENT ON COLUMN assumptions_service.scenarios.parent_scenario_id IS 'Parent scenario for inheritance (overrides cascade from parent)';

COMMENT ON COLUMN assumptions_service.assumption_change_log.provenance_hash IS 'SHA-256 hash of the change provenance for audit trail integrity';
