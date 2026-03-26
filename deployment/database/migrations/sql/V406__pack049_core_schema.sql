-- =============================================================================
-- PACK-049 GHG Multi-Site Management Pack
-- Migration: V406 - Core Schema
-- =============================================================================
-- Pack:         PACK-049 (GHG Multi-Site Management Pack)
-- Migration:    001 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates the ghg_multisite schema and core configuration/reporting period
-- tables. All tables use UUID primary keys, NUMERIC(20,6) for emissions,
-- NUMERIC(10,4) for percentages, JSONB for flexible data, and full
-- tenant isolation via RLS.
--
-- Tables (2):
--   1. ghg_multisite.gl_ms_configurations
--   2. ghg_multisite.gl_ms_reporting_periods
--
-- Also includes: schema, RLS policies, indexes, comments.
-- Next: V407__pack049_site_registry.sql
-- =============================================================================

-- Create schema
CREATE SCHEMA IF NOT EXISTS ghg_multisite;
SET search_path TO ghg_multisite, public;

-- =============================================================================
-- Table 1: ghg_multisite.gl_ms_configurations
-- =============================================================================
-- Master configuration for a multi-site management instance. Each tenant
-- can have multiple configurations (e.g., one per business unit or
-- reporting entity) but only one active at a time.

CREATE TABLE ghg_multisite.gl_ms_configurations (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_name                 VARCHAR(255)    NOT NULL,
    preset_name                 VARCHAR(100),
    consolidation_approach      VARCHAR(30)     NOT NULL DEFAULT 'OPERATIONAL_CONTROL',
    collection_period           VARCHAR(20)     NOT NULL DEFAULT 'MONTHLY',
    materiality_threshold       NUMERIC(10,4)   NOT NULL DEFAULT 5.0000,
    de_minimis_threshold        NUMERIC(10,4)   NOT NULL DEFAULT 1.0000,
    completeness_target         NUMERIC(10,4)   NOT NULL DEFAULT 95.0000,
    config_data                 JSONB           DEFAULT '{}',
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p049_cfg_approach CHECK (
        consolidation_approach IN (
            'OPERATIONAL_CONTROL', 'FINANCIAL_CONTROL', 'EQUITY_SHARE'
        )
    ),
    CONSTRAINT chk_p049_cfg_period CHECK (
        collection_period IN ('MONTHLY', 'QUARTERLY', 'ANNUAL')
    ),
    CONSTRAINT chk_p049_cfg_materiality CHECK (
        materiality_threshold >= 0 AND materiality_threshold <= 100
    ),
    CONSTRAINT chk_p049_cfg_deminimis CHECK (
        de_minimis_threshold >= 0 AND de_minimis_threshold <= 100
    ),
    CONSTRAINT chk_p049_cfg_completeness CHECK (
        completeness_target >= 0 AND completeness_target <= 100
    ),
    CONSTRAINT uq_p049_cfg_tenant_name UNIQUE (tenant_id, config_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_cfg_tenant          ON ghg_multisite.gl_ms_configurations(tenant_id);
CREATE INDEX idx_p049_cfg_active          ON ghg_multisite.gl_ms_configurations(tenant_id, is_active)
    WHERE is_active = true;
CREATE INDEX idx_p049_cfg_approach        ON ghg_multisite.gl_ms_configurations(consolidation_approach);
CREATE INDEX idx_p049_cfg_preset          ON ghg_multisite.gl_ms_configurations(preset_name)
    WHERE preset_name IS NOT NULL;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_configurations ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_cfg_tenant_isolation ON ghg_multisite.gl_ms_configurations
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 2: ghg_multisite.gl_ms_reporting_periods
-- =============================================================================
-- Reporting periods linked to a configuration. Each period defines a date
-- range, submission deadline, and lifecycle status.

CREATE TABLE ghg_multisite.gl_ms_reporting_periods (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_configurations(id) ON DELETE CASCADE,
    period_name                 VARCHAR(100)    NOT NULL,
    period_start                DATE            NOT NULL,
    period_end                  DATE            NOT NULL,
    deadline                    DATE,
    status                      VARCHAR(20)     NOT NULL DEFAULT 'OPEN',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_rp_dates CHECK (period_end > period_start),
    CONSTRAINT chk_p049_rp_status CHECK (
        status IN ('OPEN', 'COLLECTION', 'REVIEW', 'CLOSED', 'ARCHIVED')
    ),
    CONSTRAINT uq_p049_rp_cfg_name UNIQUE (config_id, period_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_rp_tenant           ON ghg_multisite.gl_ms_reporting_periods(tenant_id);
CREATE INDEX idx_p049_rp_config           ON ghg_multisite.gl_ms_reporting_periods(config_id);
CREATE INDEX idx_p049_rp_status           ON ghg_multisite.gl_ms_reporting_periods(status);
CREATE INDEX idx_p049_rp_dates            ON ghg_multisite.gl_ms_reporting_periods(period_start, period_end);
CREATE INDEX idx_p049_rp_deadline         ON ghg_multisite.gl_ms_reporting_periods(deadline)
    WHERE deadline IS NOT NULL;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_reporting_periods ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_rp_tenant_isolation ON ghg_multisite.gl_ms_reporting_periods
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA ghg_multisite IS
    'PACK-049: GHG Multi-Site Management schema for portfolio-wide emission consolidation.';
COMMENT ON TABLE ghg_multisite.gl_ms_configurations IS
    'PACK-049: Master configuration with consolidation approach, collection period, and thresholds.';
COMMENT ON TABLE ghg_multisite.gl_ms_reporting_periods IS
    'PACK-049: Reporting periods with date range, deadline, and lifecycle status.';
