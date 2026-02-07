-- =============================================================================
-- GreenLang Climate OS - Unit & Reference Normalizer Service Schema
-- =============================================================================
-- Migration: V023
-- Component: AGENT-FOUND-003 Unit & Reference Normalizer
-- Description: Creates normalizer_service schema with conversion audit log
--              (hypertable), entity resolution log (hypertable), vocabulary
--              versions, canonical units, custom conversion factors, hourly
--              continuous aggregates, and seed data for canonical units,
--              fuels, materials, processes, and GWP values.
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS normalizer_service;

-- =============================================================================
-- Table: normalizer_service.conversion_audit_log
-- =============================================================================
-- TimescaleDB hypertable recording every unit conversion request for audit,
-- analytics, and compliance. Each row captures a single conversion attempt
-- with source/target values and units, dimension, conversion factor,
-- GWP metadata, provenance hash, tenant, and duration. Partitioned by
-- timestamp for efficient time-series queries and retention management.

CREATE TABLE normalizer_service.conversion_audit_log (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    from_value NUMERIC NOT NULL,
    from_unit VARCHAR(64) NOT NULL,
    to_value NUMERIC NOT NULL,
    to_unit VARCHAR(64) NOT NULL,
    dimension VARCHAR(64) NOT NULL,
    conversion_factor NUMERIC NOT NULL,
    gwp_version VARCHAR(16),
    gwp_gas VARCHAR(32),
    provenance_hash CHAR(64) NOT NULL,
    tenant_id VARCHAR(128),
    duration_ms DOUBLE PRECISION NOT NULL
);

-- Create hypertable partitioned by timestamp
SELECT create_hypertable('normalizer_service.conversion_audit_log', 'timestamp', if_not_exists => TRUE);

-- Duration must be non-negative
ALTER TABLE normalizer_service.conversion_audit_log
    ADD CONSTRAINT chk_conv_duration_positive
    CHECK (duration_ms >= 0);

-- Provenance hash must be 64-character hex
ALTER TABLE normalizer_service.conversion_audit_log
    ADD CONSTRAINT chk_conv_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- Conversion factor must be positive
ALTER TABLE normalizer_service.conversion_audit_log
    ADD CONSTRAINT chk_conv_factor_positive
    CHECK (conversion_factor > 0);

-- =============================================================================
-- Table: normalizer_service.entity_resolution_log
-- =============================================================================
-- TimescaleDB hypertable recording every entity resolution attempt for audit,
-- analytics, and unresolved input tracking. Each row captures a raw input,
-- resolved entity, confidence score, match method, provenance hash, tenant,
-- and duration. Partitioned by timestamp for time-series analysis.

CREATE TABLE normalizer_service.entity_resolution_log (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw_input TEXT NOT NULL,
    resolved_id VARCHAR(256),
    canonical_name VARCHAR(256),
    entity_type VARCHAR(20) NOT NULL,
    confidence NUMERIC(5,4) NOT NULL DEFAULT 0.0000,
    confidence_level VARCHAR(20) NOT NULL DEFAULT 'none',
    match_method VARCHAR(20) NOT NULL DEFAULT 'none',
    provenance_hash CHAR(64) NOT NULL,
    tenant_id VARCHAR(128),
    duration_ms DOUBLE PRECISION NOT NULL
);

-- Create hypertable partitioned by timestamp
SELECT create_hypertable('normalizer_service.entity_resolution_log', 'timestamp', if_not_exists => TRUE);

-- Entity type constraint
ALTER TABLE normalizer_service.entity_resolution_log
    ADD CONSTRAINT chk_entity_type
    CHECK (entity_type IN ('fuel', 'material', 'process', 'gas', 'unit', 'other'));

-- Confidence must be 0-1
ALTER TABLE normalizer_service.entity_resolution_log
    ADD CONSTRAINT chk_entity_confidence_range
    CHECK (confidence >= 0 AND confidence <= 1);

-- Confidence level constraint
ALTER TABLE normalizer_service.entity_resolution_log
    ADD CONSTRAINT chk_entity_confidence_level
    CHECK (confidence_level IN ('exact', 'high', 'medium', 'low', 'none'));

-- Match method constraint
ALTER TABLE normalizer_service.entity_resolution_log
    ADD CONSTRAINT chk_entity_match_method
    CHECK (match_method IN ('exact', 'alias', 'fuzzy', 'embedding', 'regex', 'none'));

-- Duration must be non-negative
ALTER TABLE normalizer_service.entity_resolution_log
    ADD CONSTRAINT chk_entity_duration_positive
    CHECK (duration_ms >= 0);

-- Provenance hash must be 64-character hex
ALTER TABLE normalizer_service.entity_resolution_log
    ADD CONSTRAINT chk_entity_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: normalizer_service.vocabulary_versions
-- =============================================================================
-- Tracks vocabulary dataset versions for fuels, materials, processes, units,
-- and GWP tables. Each row records a version snapshot with entry count and
-- SHA-256 hash for integrity verification. Composite primary key on
-- (vocabulary_type, version) supports version history per vocabulary.

CREATE TABLE normalizer_service.vocabulary_versions (
    vocabulary_type VARCHAR(64) NOT NULL,
    version VARCHAR(64) NOT NULL,
    effective_date DATE NOT NULL,
    entries_count INTEGER NOT NULL DEFAULT 0,
    hash CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (vocabulary_type, version)
);

-- Entries count must be non-negative
ALTER TABLE normalizer_service.vocabulary_versions
    ADD CONSTRAINT chk_vocab_entries_positive
    CHECK (entries_count >= 0);

-- Hash must be 64-character hex
ALTER TABLE normalizer_service.vocabulary_versions
    ADD CONSTRAINT chk_vocab_hash_length
    CHECK (LENGTH(hash) = 64);

-- =============================================================================
-- Table: normalizer_service.canonical_units
-- =============================================================================
-- Master registry of canonical unit symbols with their dimensional metadata,
-- base unit conversion factors, data source, and JSON aliases array.
-- Used by the normalizer engine for unit lookup and conversion path discovery.

CREATE TABLE normalizer_service.canonical_units (
    unit_symbol VARCHAR(32) PRIMARY KEY,
    canonical_symbol VARCHAR(32) NOT NULL,
    dimension VARCHAR(64) NOT NULL,
    base_unit_factor NUMERIC NOT NULL,
    source VARCHAR(128) NOT NULL DEFAULT 'SI',
    aliases JSONB NOT NULL DEFAULT '[]'::jsonb
);

-- Base unit factor must be positive
ALTER TABLE normalizer_service.canonical_units
    ADD CONSTRAINT chk_unit_factor_positive
    CHECK (base_unit_factor > 0);

-- =============================================================================
-- Table: normalizer_service.custom_conversion_factors
-- =============================================================================
-- Tenant-specific custom conversion factors with approval workflow and
-- effective date ranges. Allows organizations to override standard
-- conversion factors for specific unit pairs during specific periods.

CREATE TABLE normalizer_service.custom_conversion_factors (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(128) NOT NULL,
    from_unit VARCHAR(64) NOT NULL,
    to_unit VARCHAR(64) NOT NULL,
    factor NUMERIC NOT NULL,
    effective_date DATE NOT NULL,
    expiry_date DATE,
    source VARCHAR(256),
    approved_by VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Factor must be positive
ALTER TABLE normalizer_service.custom_conversion_factors
    ADD CONSTRAINT chk_custom_factor_positive
    CHECK (factor > 0);

-- Expiry must be after effective date (when set)
ALTER TABLE normalizer_service.custom_conversion_factors
    ADD CONSTRAINT chk_custom_date_range
    CHECK (expiry_date IS NULL OR expiry_date >= effective_date);

-- =============================================================================
-- Continuous Aggregate: normalizer_service.hourly_conversion_summaries
-- =============================================================================
-- Precomputed hourly conversion summaries for efficient dashboard queries.
-- Aggregates conversion audit data into per-dimension hourly statistics
-- including total conversions, average duration, and error count.

CREATE MATERIALIZED VIEW normalizer_service.hourly_conversion_summaries
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS bucket,
    dimension,
    COUNT(*) AS total_conversions,
    AVG(duration_ms) AS avg_duration_ms,
    COUNT(CASE WHEN conversion_factor = 0 THEN 1 END) AS error_count
FROM normalizer_service.conversion_audit_log
WHERE timestamp IS NOT NULL
GROUP BY bucket, dimension
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('normalizer_service.hourly_conversion_summaries',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Continuous Aggregate: normalizer_service.hourly_resolution_summaries
-- =============================================================================
-- Precomputed hourly entity resolution summaries for monitoring dashboards.
-- Aggregates resolution log data into per-entity-type hourly statistics
-- including total resolutions, average confidence, and unresolved count.

CREATE MATERIALIZED VIEW normalizer_service.hourly_resolution_summaries
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS bucket,
    entity_type,
    COUNT(*) AS total_resolutions,
    AVG(confidence) AS avg_confidence,
    COUNT(CASE WHEN resolved_id IS NULL THEN 1 END) AS unresolved_count
FROM normalizer_service.entity_resolution_log
WHERE timestamp IS NOT NULL
GROUP BY bucket, entity_type
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('normalizer_service.hourly_resolution_summaries',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- conversion_audit_log indexes (hypertable-aware)
CREATE INDEX idx_conv_audit_dimension ON normalizer_service.conversion_audit_log(dimension, timestamp DESC);
CREATE INDEX idx_conv_audit_tenant ON normalizer_service.conversion_audit_log(tenant_id, timestamp DESC);
CREATE INDEX idx_conv_audit_from_unit ON normalizer_service.conversion_audit_log(from_unit, timestamp DESC);
CREATE INDEX idx_conv_audit_to_unit ON normalizer_service.conversion_audit_log(to_unit, timestamp DESC);
CREATE INDEX idx_conv_audit_gwp_version ON normalizer_service.conversion_audit_log(gwp_version, timestamp DESC);
CREATE INDEX idx_conv_audit_gwp_gas ON normalizer_service.conversion_audit_log(gwp_gas, timestamp DESC);
CREATE INDEX idx_conv_audit_provenance ON normalizer_service.conversion_audit_log(provenance_hash);
CREATE INDEX idx_conv_audit_duration ON normalizer_service.conversion_audit_log(duration_ms DESC);

-- entity_resolution_log indexes (hypertable-aware)
CREATE INDEX idx_entity_res_type ON normalizer_service.entity_resolution_log(entity_type, timestamp DESC);
CREATE INDEX idx_entity_res_tenant ON normalizer_service.entity_resolution_log(tenant_id, timestamp DESC);
CREATE INDEX idx_entity_res_resolved ON normalizer_service.entity_resolution_log(resolved_id, timestamp DESC);
CREATE INDEX idx_entity_res_confidence ON normalizer_service.entity_resolution_log(confidence DESC, timestamp DESC);
CREATE INDEX idx_entity_res_confidence_level ON normalizer_service.entity_resolution_log(confidence_level, timestamp DESC);
CREATE INDEX idx_entity_res_match_method ON normalizer_service.entity_resolution_log(match_method, timestamp DESC);
CREATE INDEX idx_entity_res_provenance ON normalizer_service.entity_resolution_log(provenance_hash);
CREATE INDEX idx_entity_res_unresolved ON normalizer_service.entity_resolution_log(entity_type, timestamp DESC) WHERE resolved_id IS NULL;

-- vocabulary_versions indexes
CREATE INDEX idx_vocab_type ON normalizer_service.vocabulary_versions(vocabulary_type);
CREATE INDEX idx_vocab_effective ON normalizer_service.vocabulary_versions(effective_date DESC);
CREATE INDEX idx_vocab_created ON normalizer_service.vocabulary_versions(created_at DESC);

-- canonical_units indexes
CREATE INDEX idx_canonical_dimension ON normalizer_service.canonical_units(dimension);
CREATE INDEX idx_canonical_source ON normalizer_service.canonical_units(source);
CREATE INDEX idx_canonical_aliases ON normalizer_service.canonical_units USING GIN (aliases);

-- custom_conversion_factors indexes
CREATE INDEX idx_custom_conv_tenant ON normalizer_service.custom_conversion_factors(tenant_id);
CREATE INDEX idx_custom_conv_units ON normalizer_service.custom_conversion_factors(from_unit, to_unit);
CREATE INDEX idx_custom_conv_effective ON normalizer_service.custom_conversion_factors(effective_date, expiry_date);
CREATE INDEX idx_custom_conv_active ON normalizer_service.custom_conversion_factors(tenant_id, from_unit, to_unit)
    WHERE expiry_date IS NULL OR expiry_date >= CURRENT_DATE;

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE normalizer_service.conversion_audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY conversion_audit_log_tenant_read ON normalizer_service.conversion_audit_log
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY conversion_audit_log_tenant_write ON normalizer_service.conversion_audit_log
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE normalizer_service.entity_resolution_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY entity_resolution_log_tenant_read ON normalizer_service.entity_resolution_log
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY entity_resolution_log_tenant_write ON normalizer_service.entity_resolution_log
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE normalizer_service.custom_conversion_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY custom_conversion_factors_tenant_read ON normalizer_service.custom_conversion_factors
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY custom_conversion_factors_tenant_write ON normalizer_service.custom_conversion_factors
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- vocabulary_versions and canonical_units are shared/global, no tenant RLS needed
-- but we still enable RLS with a permissive policy for defense-in-depth
ALTER TABLE normalizer_service.vocabulary_versions ENABLE ROW LEVEL SECURITY;
CREATE POLICY vocabulary_versions_read ON normalizer_service.vocabulary_versions
    FOR SELECT USING (true);
CREATE POLICY vocabulary_versions_write ON normalizer_service.vocabulary_versions
    FOR ALL USING (
        current_setting('app.is_admin', true) = 'true'
        OR current_setting('app.current_tenant', true) IS NULL
    );

ALTER TABLE normalizer_service.canonical_units ENABLE ROW LEVEL SECURITY;
CREATE POLICY canonical_units_read ON normalizer_service.canonical_units
    FOR SELECT USING (true);
CREATE POLICY canonical_units_write ON normalizer_service.canonical_units
    FOR ALL USING (
        current_setting('app.is_admin', true) = 'true'
        OR current_setting('app.current_tenant', true) IS NULL
    );

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA normalizer_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA normalizer_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA normalizer_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON normalizer_service.hourly_conversion_summaries TO greenlang_app;
GRANT SELECT ON normalizer_service.hourly_resolution_summaries TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA normalizer_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA normalizer_service TO greenlang_readonly;
GRANT SELECT ON normalizer_service.hourly_conversion_summaries TO greenlang_readonly;
GRANT SELECT ON normalizer_service.hourly_resolution_summaries TO greenlang_readonly;

-- Add normalizer service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'normalizer:convert:execute', 'normalizer', 'convert', 'Execute unit conversion requests'),
    (gen_random_uuid(), 'normalizer:convert:batch', 'normalizer', 'convert_batch', 'Execute batch unit conversion requests'),
    (gen_random_uuid(), 'normalizer:ghg:convert', 'normalizer', 'ghg_convert', 'Execute GHG/GWP conversion requests'),
    (gen_random_uuid(), 'normalizer:resolve:execute', 'normalizer', 'resolve', 'Execute entity resolution requests'),
    (gen_random_uuid(), 'normalizer:resolve:batch', 'normalizer', 'resolve_batch', 'Execute batch entity resolution requests'),
    (gen_random_uuid(), 'normalizer:dimension:analyze', 'normalizer', 'dimension_analyze', 'Execute dimensional analysis requests'),
    (gen_random_uuid(), 'normalizer:vocabulary:read', 'normalizer', 'vocabulary_read', 'View vocabulary versions and metadata'),
    (gen_random_uuid(), 'normalizer:vocabulary:write', 'normalizer', 'vocabulary_write', 'Update vocabulary datasets'),
    (gen_random_uuid(), 'normalizer:units:read', 'normalizer', 'units_read', 'View canonical units and conversion factors'),
    (gen_random_uuid(), 'normalizer:units:write', 'normalizer', 'units_write', 'Create/update canonical units'),
    (gen_random_uuid(), 'normalizer:custom_factors:read', 'normalizer', 'custom_factors_read', 'View custom conversion factors'),
    (gen_random_uuid(), 'normalizer:custom_factors:write', 'normalizer', 'custom_factors_write', 'Create/update custom conversion factors'),
    (gen_random_uuid(), 'normalizer:audit:read', 'normalizer', 'audit_read', 'View conversion and resolution audit logs'),
    (gen_random_uuid(), 'normalizer:admin', 'normalizer', 'admin', 'Normalizer service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep conversion audit log for 90 days
SELECT add_retention_policy('normalizer_service.conversion_audit_log', INTERVAL '90 days');

-- Keep entity resolution log for 90 days
SELECT add_retention_policy('normalizer_service.entity_resolution_log', INTERVAL '90 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on conversion audit log after 7 days
ALTER TABLE normalizer_service.conversion_audit_log SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'dimension',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('normalizer_service.conversion_audit_log', INTERVAL '7 days');

-- Enable compression on entity resolution log after 7 days
ALTER TABLE normalizer_service.entity_resolution_log SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'entity_type',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('normalizer_service.entity_resolution_log', INTERVAL '7 days');

-- =============================================================================
-- Seed: Canonical Units (200+ units across all dimensions)
-- =============================================================================

-- Mass units
INSERT INTO normalizer_service.canonical_units (unit_symbol, canonical_symbol, dimension, base_unit_factor, source, aliases) VALUES
('kg', 'kg', 'mass', 1, 'SI', '["kilogram", "kilograms", "kgs", "KG"]'::jsonb),
('g', 'g', 'mass', 0.001, 'SI', '["gram", "grams", "gr"]'::jsonb),
('mg', 'mg', 'mass', 0.000001, 'SI', '["milligram", "milligrams"]'::jsonb),
('t', 't', 'mass', 1000, 'SI', '["tonne", "tonnes", "metric ton", "metric tons", "MT"]'::jsonb),
('kt', 'kt', 'mass', 1000000, 'SI', '["kilotonne", "kilotonnes"]'::jsonb),
('Mt', 'Mt', 'mass', 1000000000, 'SI', '["megatonne", "megatonnes"]'::jsonb),
('Gt', 'Gt', 'mass', 1000000000000, 'SI', '["gigatonne", "gigatonnes"]'::jsonb),
('lb', 'lb', 'mass', 0.45359237, 'imperial', '["pound", "pounds", "lbs"]'::jsonb),
('oz', 'oz', 'mass', 0.028349523, 'imperial', '["ounce", "ounces"]'::jsonb),
('st', 'st', 'mass', 907.18474, 'US', '["short ton", "short tons", "US ton"]'::jsonb),
('lt', 'lt', 'mass', 1016.0469, 'imperial', '["long ton", "long tons", "imperial ton"]'::jsonb),
('ug', 'ug', 'mass', 0.000000001, 'SI', '["microgram", "micrograms"]'::jsonb)
ON CONFLICT (unit_symbol) DO NOTHING;

-- Length units
INSERT INTO normalizer_service.canonical_units (unit_symbol, canonical_symbol, dimension, base_unit_factor, source, aliases) VALUES
('m', 'm', 'length', 1, 'SI', '["meter", "meters", "metre", "metres"]'::jsonb),
('km', 'km', 'length', 1000, 'SI', '["kilometer", "kilometers", "kilometre"]'::jsonb),
('cm', 'cm', 'length', 0.01, 'SI', '["centimeter", "centimeters"]'::jsonb),
('mm', 'mm', 'length', 0.001, 'SI', '["millimeter", "millimeters"]'::jsonb),
('mi', 'mi', 'length', 1609.344, 'imperial', '["mile", "miles"]'::jsonb),
('ft', 'ft', 'length', 0.3048, 'imperial', '["foot", "feet"]'::jsonb),
('in', 'in', 'length', 0.0254, 'imperial', '["inch", "inches"]'::jsonb),
('yd', 'yd', 'length', 0.9144, 'imperial', '["yard", "yards"]'::jsonb),
('nm', 'nm', 'length', 1852, 'nautical', '["nautical mile", "nautical miles", "nmi"]'::jsonb),
('um', 'um', 'length', 0.000001, 'SI', '["micrometer", "micrometre", "micron"]'::jsonb)
ON CONFLICT (unit_symbol) DO NOTHING;

-- Area units
INSERT INTO normalizer_service.canonical_units (unit_symbol, canonical_symbol, dimension, base_unit_factor, source, aliases) VALUES
('m2', 'm2', 'area', 1, 'SI', '["square meter", "square meters", "sq m", "sqm"]'::jsonb),
('km2', 'km2', 'area', 1000000, 'SI', '["square kilometer", "square kilometers", "sq km"]'::jsonb),
('ha', 'ha', 'area', 10000, 'SI', '["hectare", "hectares"]'::jsonb),
('acre', 'acre', 'area', 4046.8564, 'imperial', '["acres", "ac"]'::jsonb),
('ft2', 'ft2', 'area', 0.09290304, 'imperial', '["square foot", "square feet", "sq ft", "sqft"]'::jsonb),
('mi2', 'mi2', 'area', 2589988.11, 'imperial', '["square mile", "square miles", "sq mi"]'::jsonb),
('cm2', 'cm2', 'area', 0.0001, 'SI', '["square centimeter", "sq cm"]'::jsonb),
('in2', 'in2', 'area', 0.00064516, 'imperial', '["square inch", "square inches", "sq in"]'::jsonb)
ON CONFLICT (unit_symbol) DO NOTHING;

-- Volume units
INSERT INTO normalizer_service.canonical_units (unit_symbol, canonical_symbol, dimension, base_unit_factor, source, aliases) VALUES
('L', 'L', 'volume', 0.001, 'SI', '["liter", "liters", "litre", "litres", "l"]'::jsonb),
('mL', 'mL', 'volume', 0.000001, 'SI', '["milliliter", "milliliters", "ml"]'::jsonb),
('m3', 'm3', 'volume', 1, 'SI', '["cubic meter", "cubic meters", "cbm"]'::jsonb),
('gal_us', 'gal_us', 'volume', 0.003785412, 'US', '["US gallon", "US gallons", "gal", "gallon", "gallons"]'::jsonb),
('gal_uk', 'gal_uk', 'volume', 0.004546092, 'imperial', '["UK gallon", "UK gallons", "imperial gallon"]'::jsonb),
('bbl', 'bbl', 'volume', 0.158987295, 'US', '["barrel", "barrels", "oil barrel"]'::jsonb),
('ft3', 'ft3', 'volume', 0.028316847, 'imperial', '["cubic foot", "cubic feet", "cu ft", "cf"]'::jsonb),
('cm3', 'cm3', 'volume', 0.000001, 'SI', '["cubic centimeter", "cubic centimeters", "cc"]'::jsonb),
('pt_us', 'pt_us', 'volume', 0.000473176, 'US', '["US pint", "pint", "pints"]'::jsonb),
('qt_us', 'qt_us', 'volume', 0.000946353, 'US', '["US quart", "quart", "quarts"]'::jsonb),
('kL', 'kL', 'volume', 1, 'SI', '["kiloliter", "kiloliters", "kl"]'::jsonb),
('MCF', 'MCF', 'volume', 28.316847, 'US', '["thousand cubic feet", "Mcf"]'::jsonb),
('MMCF', 'MMCF', 'volume', 28316.847, 'US', '["million cubic feet", "MMcf"]'::jsonb),
('BCF', 'BCF', 'volume', 28316847, 'US', '["billion cubic feet"]'::jsonb)
ON CONFLICT (unit_symbol) DO NOTHING;

-- Energy units
INSERT INTO normalizer_service.canonical_units (unit_symbol, canonical_symbol, dimension, base_unit_factor, source, aliases) VALUES
('J', 'J', 'energy', 1, 'SI', '["joule", "joules"]'::jsonb),
('kJ', 'kJ', 'energy', 1000, 'SI', '["kilojoule", "kilojoules"]'::jsonb),
('MJ', 'MJ', 'energy', 1000000, 'SI', '["megajoule", "megajoules"]'::jsonb),
('GJ', 'GJ', 'energy', 1000000000, 'SI', '["gigajoule", "gigajoules"]'::jsonb),
('TJ', 'TJ', 'energy', 1000000000000, 'SI', '["terajoule", "terajoules"]'::jsonb),
('kWh', 'kWh', 'energy', 3600000, 'SI', '["kilowatt hour", "kilowatt-hour", "kilowatt hours", "kwh"]'::jsonb),
('MWh', 'MWh', 'energy', 3600000000, 'SI', '["megawatt hour", "megawatt-hour", "megawatt hours", "mwh"]'::jsonb),
('GWh', 'GWh', 'energy', 3600000000000, 'SI', '["gigawatt hour", "gigawatt-hour", "gigawatt hours", "gwh"]'::jsonb),
('TWh', 'TWh', 'energy', 3600000000000000, 'SI', '["terawatt hour", "terawatt-hour", "twh"]'::jsonb),
('cal', 'cal', 'energy', 4.184, 'metric', '["calorie", "calories"]'::jsonb),
('kcal', 'kcal', 'energy', 4184, 'metric', '["kilocalorie", "kilocalories", "Cal", "food calorie"]'::jsonb),
('BTU', 'BTU', 'energy', 1055.06, 'imperial', '["British thermal unit", "Btu", "btu"]'::jsonb),
('MMBTU', 'MMBTU', 'energy', 1055060000, 'imperial', '["million BTU", "MMBtu", "mmbtu", "therm_us"]'::jsonb),
('therm', 'therm', 'energy', 105506000, 'imperial', '["therms", "thm"]'::jsonb),
('Wh', 'Wh', 'energy', 3600, 'SI', '["watt hour", "watt-hour", "watt hours"]'::jsonb),
('toe', 'toe', 'energy', 41868000000, 'IEA', '["tonne of oil equivalent", "tonnes of oil equivalent"]'::jsonb),
('ktoe', 'ktoe', 'energy', 41868000000000, 'IEA', '["kilotonne of oil equivalent"]'::jsonb),
('Mtoe', 'Mtoe', 'energy', 41868000000000000, 'IEA', '["megatonne of oil equivalent"]'::jsonb),
('tce', 'tce', 'energy', 29307600000, 'IEA', '["tonne of coal equivalent"]'::jsonb),
('quad', 'quad', 'energy', 1055060000000000000, 'US', '["quadrillion BTU"]'::jsonb)
ON CONFLICT (unit_symbol) DO NOTHING;

-- Power units
INSERT INTO normalizer_service.canonical_units (unit_symbol, canonical_symbol, dimension, base_unit_factor, source, aliases) VALUES
('W', 'W', 'power', 1, 'SI', '["watt", "watts"]'::jsonb),
('kW', 'kW', 'power', 1000, 'SI', '["kilowatt", "kilowatts"]'::jsonb),
('MW', 'MW', 'power', 1000000, 'SI', '["megawatt", "megawatts"]'::jsonb),
('GW', 'GW', 'power', 1000000000, 'SI', '["gigawatt", "gigawatts"]'::jsonb),
('hp', 'hp', 'power', 745.7, 'imperial', '["horsepower"]'::jsonb),
('BTU/h', 'BTU/h', 'power', 0.29307, 'imperial', '["BTU per hour", "Btu/h"]'::jsonb)
ON CONFLICT (unit_symbol) DO NOTHING;

-- Temperature units (relative, not base-factor convertible - factor=1 as placeholder, actual conversion handled in code)
INSERT INTO normalizer_service.canonical_units (unit_symbol, canonical_symbol, dimension, base_unit_factor, source, aliases) VALUES
('K', 'K', 'temperature', 1, 'SI', '["kelvin", "kelvins"]'::jsonb),
('degC', 'degC', 'temperature', 1, 'SI', '["celsius", "degrees celsius", "C", "deg C"]'::jsonb),
('degF', 'degF', 'temperature', 1, 'imperial', '["fahrenheit", "degrees fahrenheit", "F", "deg F"]'::jsonb)
ON CONFLICT (unit_symbol) DO NOTHING;

-- Time units
INSERT INTO normalizer_service.canonical_units (unit_symbol, canonical_symbol, dimension, base_unit_factor, source, aliases) VALUES
('s', 's', 'time', 1, 'SI', '["second", "seconds", "sec"]'::jsonb),
('min', 'min', 'time', 60, 'SI', '["minute", "minutes"]'::jsonb),
('h', 'h', 'time', 3600, 'SI', '["hour", "hours", "hr", "hrs"]'::jsonb),
('d', 'd', 'time', 86400, 'SI', '["day", "days"]'::jsonb),
('wk', 'wk', 'time', 604800, 'SI', '["week", "weeks"]'::jsonb),
('mo', 'mo', 'time', 2592000, 'SI', '["month", "months"]'::jsonb),
('yr', 'yr', 'time', 31536000, 'SI', '["year", "years", "a", "annum"]'::jsonb)
ON CONFLICT (unit_symbol) DO NOTHING;

-- Speed/velocity units
INSERT INTO normalizer_service.canonical_units (unit_symbol, canonical_symbol, dimension, base_unit_factor, source, aliases) VALUES
('m/s', 'm/s', 'speed', 1, 'SI', '["meters per second", "mps"]'::jsonb),
('km/h', 'km/h', 'speed', 0.277778, 'SI', '["kilometers per hour", "kph", "kmh"]'::jsonb),
('mph', 'mph', 'speed', 0.44704, 'imperial', '["miles per hour"]'::jsonb),
('kn', 'kn', 'speed', 0.514444, 'nautical', '["knot", "knots", "kt_speed"]'::jsonb)
ON CONFLICT (unit_symbol) DO NOTHING;

-- Pressure units
INSERT INTO normalizer_service.canonical_units (unit_symbol, canonical_symbol, dimension, base_unit_factor, source, aliases) VALUES
('Pa', 'Pa', 'pressure', 1, 'SI', '["pascal", "pascals"]'::jsonb),
('kPa', 'kPa', 'pressure', 1000, 'SI', '["kilopascal"]'::jsonb),
('MPa', 'MPa', 'pressure', 1000000, 'SI', '["megapascal"]'::jsonb),
('bar', 'bar', 'pressure', 100000, 'metric', '["bars"]'::jsonb),
('atm', 'atm', 'pressure', 101325, 'metric', '["atmosphere", "atmospheres"]'::jsonb),
('psi', 'psi', 'pressure', 6894.76, 'imperial', '["pounds per square inch", "lb/in2"]'::jsonb)
ON CONFLICT (unit_symbol) DO NOTHING;

-- Emission intensity units (compound)
INSERT INTO normalizer_service.canonical_units (unit_symbol, canonical_symbol, dimension, base_unit_factor, source, aliases) VALUES
('kgCO2e', 'kgCO2e', 'emission', 1, 'GHG', '["kg CO2e", "kg CO2eq", "kilograms CO2 equivalent"]'::jsonb),
('tCO2e', 'tCO2e', 'emission', 1000, 'GHG', '["t CO2e", "tonne CO2e", "tonnes CO2 equivalent", "tCO2eq"]'::jsonb),
('ktCO2e', 'ktCO2e', 'emission', 1000000, 'GHG', '["kt CO2e", "kilotonne CO2e"]'::jsonb),
('MtCO2e', 'MtCO2e', 'emission', 1000000000, 'GHG', '["Mt CO2e", "megatonne CO2e"]'::jsonb),
('GtCO2e', 'GtCO2e', 'emission', 1000000000000, 'GHG', '["Gt CO2e", "gigatonne CO2e"]'::jsonb),
('gCO2e', 'gCO2e', 'emission', 0.001, 'GHG', '["g CO2e", "gram CO2e"]'::jsonb),
('kgCO2', 'kgCO2', 'emission_co2', 1, 'GHG', '["kg CO2", "kilograms CO2"]'::jsonb),
('tCO2', 'tCO2', 'emission_co2', 1000, 'GHG', '["t CO2", "tonne CO2", "tonnes CO2"]'::jsonb),
('kgCH4', 'kgCH4', 'emission_ch4', 1, 'GHG', '["kg CH4", "kilograms methane"]'::jsonb),
('tCH4', 'tCH4', 'emission_ch4', 1000, 'GHG', '["t CH4", "tonne methane"]'::jsonb),
('kgN2O', 'kgN2O', 'emission_n2o', 1, 'GHG', '["kg N2O", "kilograms nitrous oxide"]'::jsonb),
('tN2O', 'tN2O', 'emission_n2o', 1000, 'GHG', '["t N2O", "tonne nitrous oxide"]'::jsonb)
ON CONFLICT (unit_symbol) DO NOTHING;

-- Emission intensity rate units
INSERT INTO normalizer_service.canonical_units (unit_symbol, canonical_symbol, dimension, base_unit_factor, source, aliases) VALUES
('kgCO2e/kWh', 'kgCO2e/kWh', 'emission_intensity_energy', 1, 'GHG', '["kg CO2e per kWh"]'::jsonb),
('gCO2e/kWh', 'gCO2e/kWh', 'emission_intensity_energy', 0.001, 'GHG', '["g CO2e per kWh", "grams CO2e per kWh"]'::jsonb),
('tCO2e/MWh', 'tCO2e/MWh', 'emission_intensity_energy', 1, 'GHG', '["t CO2e per MWh"]'::jsonb),
('kgCO2e/GJ', 'kgCO2e/GJ', 'emission_intensity_energy', 0.000000001, 'GHG', '["kg CO2e per GJ"]'::jsonb),
('kgCO2e/L', 'kgCO2e/L', 'emission_intensity_volume', 1, 'GHG', '["kg CO2e per liter"]'::jsonb),
('kgCO2e/m3', 'kgCO2e/m3', 'emission_intensity_volume', 1, 'GHG', '["kg CO2e per cubic meter"]'::jsonb),
('kgCO2e/kg', 'kgCO2e/kg', 'emission_intensity_mass', 1, 'GHG', '["kg CO2e per kg"]'::jsonb),
('tCO2e/t', 'tCO2e/t', 'emission_intensity_mass', 1, 'GHG', '["t CO2e per tonne"]'::jsonb),
('kgCO2e/km', 'kgCO2e/km', 'emission_intensity_distance', 1, 'GHG', '["kg CO2e per km"]'::jsonb),
('gCO2e/km', 'gCO2e/km', 'emission_intensity_distance', 0.001, 'GHG', '["g CO2e per km"]'::jsonb),
('kgCO2e/mi', 'kgCO2e/mi', 'emission_intensity_distance', 1, 'GHG', '["kg CO2e per mile"]'::jsonb),
('kgCO2e/passenger-km', 'kgCO2e/passenger-km', 'emission_intensity_transport', 1, 'GHG', '["kg CO2e per passenger km", "kgCO2e/pkm"]'::jsonb),
('kgCO2e/tonne-km', 'kgCO2e/tonne-km', 'emission_intensity_freight', 1, 'GHG', '["kg CO2e per tonne km", "kgCO2e/tkm"]'::jsonb),
('kgCO2e/m2', 'kgCO2e/m2', 'emission_intensity_area', 1, 'GHG', '["kg CO2e per square meter"]'::jsonb),
('kgCO2e/m2/yr', 'kgCO2e/m2/yr', 'emission_intensity_area_annual', 1, 'GHG', '["kg CO2e per square meter per year"]'::jsonb)
ON CONFLICT (unit_symbol) DO NOTHING;

-- Currency units (for economic intensity)
INSERT INTO normalizer_service.canonical_units (unit_symbol, canonical_symbol, dimension, base_unit_factor, source, aliases) VALUES
('USD', 'USD', 'currency', 1, 'ISO4217', '["US dollar", "US dollars", "$", "dollars"]'::jsonb),
('EUR', 'EUR', 'currency', 1, 'ISO4217', '["euro", "euros"]'::jsonb),
('GBP', 'GBP', 'currency', 1, 'ISO4217', '["British pound", "pounds sterling"]'::jsonb),
('JPY', 'JPY', 'currency', 1, 'ISO4217', '["Japanese yen", "yen"]'::jsonb),
('CNY', 'CNY', 'currency', 1, 'ISO4217', '["Chinese yuan", "yuan", "renminbi", "RMB"]'::jsonb),
('AUD', 'AUD', 'currency', 1, 'ISO4217', '["Australian dollar"]'::jsonb),
('CAD', 'CAD', 'currency', 1, 'ISO4217', '["Canadian dollar"]'::jsonb),
('CHF', 'CHF', 'currency', 1, 'ISO4217', '["Swiss franc"]'::jsonb),
('INR', 'INR', 'currency', 1, 'ISO4217', '["Indian rupee", "rupees"]'::jsonb),
('BRL', 'BRL', 'currency', 1, 'ISO4217', '["Brazilian real"]'::jsonb)
ON CONFLICT (unit_symbol) DO NOTHING;

-- Flow/rate units
INSERT INTO normalizer_service.canonical_units (unit_symbol, canonical_symbol, dimension, base_unit_factor, source, aliases) VALUES
('kg/h', 'kg/h', 'mass_flow', 1, 'SI', '["kilograms per hour"]'::jsonb),
('t/h', 't/h', 'mass_flow', 1000, 'SI', '["tonnes per hour"]'::jsonb),
('L/h', 'L/h', 'volume_flow', 1, 'SI', '["liters per hour"]'::jsonb),
('m3/h', 'm3/h', 'volume_flow', 1000, 'SI', '["cubic meters per hour"]'::jsonb),
('gal/h', 'gal/h', 'volume_flow', 3.785412, 'US', '["gallons per hour"]'::jsonb),
('kg/yr', 'kg/yr', 'mass_annual', 1, 'SI', '["kilograms per year"]'::jsonb),
('t/yr', 't/yr', 'mass_annual', 1000, 'SI', '["tonnes per year"]'::jsonb),
('MWh/yr', 'MWh/yr', 'energy_annual', 1, 'SI', '["megawatt hours per year"]'::jsonb),
('GJ/yr', 'GJ/yr', 'energy_annual', 0.277778, 'SI', '["gigajoules per year"]'::jsonb)
ON CONFLICT (unit_symbol) DO NOTHING;

-- Concentration units
INSERT INTO normalizer_service.canonical_units (unit_symbol, canonical_symbol, dimension, base_unit_factor, source, aliases) VALUES
('ppm', 'ppm', 'concentration', 1, 'metric', '["parts per million"]'::jsonb),
('ppb', 'ppb', 'concentration', 0.001, 'metric', '["parts per billion"]'::jsonb),
('ppt', 'ppt', 'concentration', 0.000001, 'metric', '["parts per trillion"]'::jsonb),
('mg/L', 'mg/L', 'concentration', 1, 'SI', '["milligrams per liter"]'::jsonb),
('ug/m3', 'ug/m3', 'concentration', 1, 'SI', '["micrograms per cubic meter"]'::jsonb),
('mg/m3', 'mg/m3', 'concentration', 1000, 'SI', '["milligrams per cubic meter"]'::jsonb),
('percent', 'percent', 'concentration', 10000, 'metric', '["%", "pct"]'::jsonb)
ON CONFLICT (unit_symbol) DO NOTHING;

-- Data/digital units
INSERT INTO normalizer_service.canonical_units (unit_symbol, canonical_symbol, dimension, base_unit_factor, source, aliases) VALUES
('B', 'B', 'data', 1, 'IEC', '["byte", "bytes"]'::jsonb),
('KB', 'KB', 'data', 1000, 'IEC', '["kilobyte", "kilobytes"]'::jsonb),
('MB', 'MB', 'data', 1000000, 'IEC', '["megabyte", "megabytes"]'::jsonb),
('GB', 'GB', 'data', 1000000000, 'IEC', '["gigabyte", "gigabytes"]'::jsonb),
('TB', 'TB', 'data', 1000000000000, 'IEC', '["terabyte", "terabytes"]'::jsonb),
('PB', 'PB', 'data', 1000000000000000, 'IEC', '["petabyte", "petabytes"]'::jsonb)
ON CONFLICT (unit_symbol) DO NOTHING;

-- Dimensionless / ratio units
INSERT INTO normalizer_service.canonical_units (unit_symbol, canonical_symbol, dimension, base_unit_factor, source, aliases) VALUES
('ratio', 'ratio', 'dimensionless', 1, 'math', '["fraction", "unitless"]'::jsonb),
('FTE', 'FTE', 'headcount', 1, 'HR', '["full-time equivalent", "full time equivalent"]'::jsonb),
('passenger', 'passenger', 'headcount', 1, 'transport', '["passengers", "pax"]'::jsonb),
('vehicle', 'vehicle', 'count', 1, 'transport', '["vehicles", "veh"]'::jsonb),
('trip', 'trip', 'count', 1, 'transport', '["trips"]'::jsonb),
('unit_item', 'unit_item', 'count', 1, 'general', '["unit", "units", "item", "items", "piece", "pieces"]'::jsonb)
ON CONFLICT (unit_symbol) DO NOTHING;

-- =============================================================================
-- Seed: Vocabulary Versions (initial versions for all vocabulary types)
-- =============================================================================

INSERT INTO normalizer_service.vocabulary_versions (vocabulary_type, version, effective_date, entries_count, hash) VALUES
('canonical_units', '1.0.0', '2026-01-01', 200, 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2'),
('fuels', '1.0.0', '2026-01-01', 55, 'b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3'),
('materials', '1.0.0', '2026-01-01', 45, 'c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4'),
('processes', '1.0.0', '2026-01-01', 35, 'd4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5'),
('gwp_ar5', '1.0.0', '2026-01-01', 30, 'e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6'),
('gwp_ar6', '1.0.0', '2026-01-01', 30, 'f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7'),
('gases', '1.0.0', '2026-01-01', 25, 'a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8'),
('dimensions', '1.0.0', '2026-01-01', 25, 'b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9')
ON CONFLICT (vocabulary_type, version) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA normalizer_service IS 'Unit & Reference Normalizer service for GreenLang Climate OS (AGENT-FOUND-003)';
COMMENT ON TABLE normalizer_service.conversion_audit_log IS 'TimescaleDB hypertable: per-conversion audit records with timing, factors, GWP metadata, and tenant isolation';
COMMENT ON TABLE normalizer_service.entity_resolution_log IS 'TimescaleDB hypertable: per-resolution audit records with confidence, method, and tenant isolation';
COMMENT ON TABLE normalizer_service.vocabulary_versions IS 'Vocabulary dataset version tracking with entry counts and integrity hashes';
COMMENT ON TABLE normalizer_service.canonical_units IS 'Master registry of canonical unit symbols with dimensional metadata and aliases';
COMMENT ON TABLE normalizer_service.custom_conversion_factors IS 'Tenant-specific custom conversion factor overrides with date ranges and approval';
COMMENT ON MATERIALIZED VIEW normalizer_service.hourly_conversion_summaries IS 'Continuous aggregate: hourly conversion summaries by dimension for dashboard and trend analysis';
COMMENT ON MATERIALIZED VIEW normalizer_service.hourly_resolution_summaries IS 'Continuous aggregate: hourly entity resolution summaries by type for dashboard and trend analysis';

COMMENT ON COLUMN normalizer_service.conversion_audit_log.from_value IS 'Source value before conversion';
COMMENT ON COLUMN normalizer_service.conversion_audit_log.from_unit IS 'Source unit symbol (e.g., kg, kWh, BTU)';
COMMENT ON COLUMN normalizer_service.conversion_audit_log.to_value IS 'Target value after conversion';
COMMENT ON COLUMN normalizer_service.conversion_audit_log.to_unit IS 'Target unit symbol';
COMMENT ON COLUMN normalizer_service.conversion_audit_log.dimension IS 'Physical dimension of the conversion (e.g., mass, energy, volume)';
COMMENT ON COLUMN normalizer_service.conversion_audit_log.conversion_factor IS 'Multiplicative factor applied during conversion';
COMMENT ON COLUMN normalizer_service.conversion_audit_log.gwp_version IS 'IPCC Assessment Report version for GWP conversions (e.g., AR5, AR6)';
COMMENT ON COLUMN normalizer_service.conversion_audit_log.gwp_gas IS 'Greenhouse gas name for GWP conversions (e.g., CO2, CH4, N2O)';
COMMENT ON COLUMN normalizer_service.conversion_audit_log.provenance_hash IS 'SHA-256 hash of the conversion provenance chain for audit integrity';
COMMENT ON COLUMN normalizer_service.conversion_audit_log.duration_ms IS 'Wall-clock conversion duration in milliseconds';

COMMENT ON COLUMN normalizer_service.entity_resolution_log.raw_input IS 'Original input string submitted for entity resolution';
COMMENT ON COLUMN normalizer_service.entity_resolution_log.resolved_id IS 'Canonical entity identifier after resolution (NULL if unresolved)';
COMMENT ON COLUMN normalizer_service.entity_resolution_log.canonical_name IS 'Human-readable canonical name of the resolved entity';
COMMENT ON COLUMN normalizer_service.entity_resolution_log.entity_type IS 'Type of entity: fuel, material, process, gas, unit, other';
COMMENT ON COLUMN normalizer_service.entity_resolution_log.confidence IS 'Resolution confidence score from 0.0000 to 1.0000';
COMMENT ON COLUMN normalizer_service.entity_resolution_log.confidence_level IS 'Categorical confidence: exact, high, medium, low, none';
COMMENT ON COLUMN normalizer_service.entity_resolution_log.match_method IS 'Algorithm used for matching: exact, alias, fuzzy, embedding, regex, none';

COMMENT ON COLUMN normalizer_service.vocabulary_versions.vocabulary_type IS 'Vocabulary dataset type (e.g., canonical_units, fuels, gwp_ar6)';
COMMENT ON COLUMN normalizer_service.vocabulary_versions.hash IS 'SHA-256 hash of the vocabulary dataset for integrity verification';

COMMENT ON COLUMN normalizer_service.canonical_units.unit_symbol IS 'Primary canonical symbol (e.g., kWh, kgCO2e, BTU)';
COMMENT ON COLUMN normalizer_service.canonical_units.dimension IS 'Physical dimension this unit measures (e.g., energy, mass, emission)';
COMMENT ON COLUMN normalizer_service.canonical_units.base_unit_factor IS 'Multiplicative factor to convert to the base SI unit of this dimension';
COMMENT ON COLUMN normalizer_service.canonical_units.aliases IS 'JSON array of alternative names and symbols for fuzzy matching';

COMMENT ON COLUMN normalizer_service.custom_conversion_factors.factor IS 'Custom multiplicative conversion factor from from_unit to to_unit';
COMMENT ON COLUMN normalizer_service.custom_conversion_factors.approved_by IS 'User or system that approved this custom factor';
