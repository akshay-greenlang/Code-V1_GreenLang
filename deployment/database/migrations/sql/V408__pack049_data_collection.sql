-- =============================================================================
-- PACK-049 GHG Multi-Site Management Pack
-- Migration: V408 - Data Collection
-- =============================================================================
-- Pack:         PACK-049 (GHG Multi-Site Management Pack)
-- Migration:    003 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates data collection tables for site-level emission data submission.
-- Collection rounds define the reporting window; site submissions track
-- per-site response lifecycle; submission data holds the actual emission
-- records; validation results capture rule-level pass/fail outcomes.
--
-- Tables (4):
--   1. ghg_multisite.gl_ms_collection_rounds
--   2. ghg_multisite.gl_ms_site_submissions
--   3. ghg_multisite.gl_ms_submission_data
--   4. ghg_multisite.gl_ms_validation_results
--
-- Also includes: indexes, RLS, comments.
-- Previous: V407__pack049_site_registry.sql
-- =============================================================================

SET search_path TO ghg_multisite, public;

-- =============================================================================
-- Table 1: ghg_multisite.gl_ms_collection_rounds
-- =============================================================================
-- A collection round represents a single data-gathering cycle for a reporting
-- period. Multiple rounds may exist (initial, correction, resubmission).

CREATE TABLE ghg_multisite.gl_ms_collection_rounds (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_configurations(id) ON DELETE CASCADE,
    period_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_reporting_periods(id) ON DELETE CASCADE,
    round_number                INTEGER         NOT NULL DEFAULT 1,
    round_type                  VARCHAR(30)     NOT NULL DEFAULT 'INITIAL',
    status                      VARCHAR(20)     NOT NULL DEFAULT 'OPEN',
    opens_at                    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    closes_at                   TIMESTAMPTZ,
    reminder_sent_at            TIMESTAMPTZ,
    sites_expected              INTEGER         NOT NULL DEFAULT 0,
    sites_submitted             INTEGER         NOT NULL DEFAULT 0,
    sites_approved              INTEGER         NOT NULL DEFAULT 0,
    completeness_pct            NUMERIC(10,4)   NOT NULL DEFAULT 0.0000,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p049_cr_type CHECK (
        round_type IN ('INITIAL', 'CORRECTION', 'RESUBMISSION', 'AUDIT')
    ),
    CONSTRAINT chk_p049_cr_status CHECK (
        status IN ('DRAFT', 'OPEN', 'COLLECTION', 'REVIEW', 'CLOSED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p049_cr_round_num CHECK (round_number >= 1 AND round_number <= 99),
    CONSTRAINT chk_p049_cr_completeness CHECK (
        completeness_pct >= 0 AND completeness_pct <= 100
    ),
    CONSTRAINT chk_p049_cr_sites_expected CHECK (sites_expected >= 0),
    CONSTRAINT chk_p049_cr_sites_submitted CHECK (sites_submitted >= 0),
    CONSTRAINT chk_p049_cr_sites_approved CHECK (sites_approved >= 0),
    CONSTRAINT uq_p049_cr_period_round UNIQUE (period_id, round_number)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_cr_tenant          ON ghg_multisite.gl_ms_collection_rounds(tenant_id);
CREATE INDEX idx_p049_cr_config          ON ghg_multisite.gl_ms_collection_rounds(config_id);
CREATE INDEX idx_p049_cr_period          ON ghg_multisite.gl_ms_collection_rounds(period_id);
CREATE INDEX idx_p049_cr_status          ON ghg_multisite.gl_ms_collection_rounds(status);
CREATE INDEX idx_p049_cr_type            ON ghg_multisite.gl_ms_collection_rounds(round_type);
CREATE INDEX idx_p049_cr_open            ON ghg_multisite.gl_ms_collection_rounds(status)
    WHERE status IN ('OPEN', 'COLLECTION');
CREATE INDEX idx_p049_cr_closes          ON ghg_multisite.gl_ms_collection_rounds(closes_at)
    WHERE closes_at IS NOT NULL;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_collection_rounds ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_cr_tenant_isolation ON ghg_multisite.gl_ms_collection_rounds
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 2: ghg_multisite.gl_ms_site_submissions
-- =============================================================================
-- Tracks the submission lifecycle for each site within a collection round.
-- Status flows: NOT_STARTED -> DRAFT -> SUBMITTED -> UNDER_REVIEW ->
-- APPROVED / REJECTED / NEEDS_REVISION.

CREATE TABLE ghg_multisite.gl_ms_site_submissions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    round_id                    UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_collection_rounds(id) ON DELETE CASCADE,
    site_id                     UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    status                      VARCHAR(20)     NOT NULL DEFAULT 'NOT_STARTED',
    submitted_at                TIMESTAMPTZ,
    submitted_by                UUID,
    reviewed_at                 TIMESTAMPTZ,
    reviewed_by                 UUID,
    reviewer_comments           TEXT,
    revision_number             INTEGER         NOT NULL DEFAULT 0,
    data_quality_score          NUMERIC(10,4),
    record_count                INTEGER         NOT NULL DEFAULT 0,
    scope1_tco2e                NUMERIC(20,6)   NOT NULL DEFAULT 0,
    scope2_location_tco2e       NUMERIC(20,6)   NOT NULL DEFAULT 0,
    scope2_market_tco2e         NUMERIC(20,6)   NOT NULL DEFAULT 0,
    scope3_tco2e                NUMERIC(20,6)   NOT NULL DEFAULT 0,
    total_tco2e                 NUMERIC(20,6)   NOT NULL DEFAULT 0,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_ss_status CHECK (
        status IN (
            'NOT_STARTED', 'DRAFT', 'SUBMITTED', 'UNDER_REVIEW',
            'APPROVED', 'REJECTED', 'NEEDS_REVISION'
        )
    ),
    CONSTRAINT chk_p049_ss_revision CHECK (revision_number >= 0),
    CONSTRAINT chk_p049_ss_dq CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p049_ss_records CHECK (record_count >= 0),
    CONSTRAINT chk_p049_ss_scope1 CHECK (scope1_tco2e >= 0),
    CONSTRAINT chk_p049_ss_scope2l CHECK (scope2_location_tco2e >= 0),
    CONSTRAINT chk_p049_ss_scope2m CHECK (scope2_market_tco2e >= 0),
    CONSTRAINT chk_p049_ss_scope3 CHECK (scope3_tco2e >= 0),
    CONSTRAINT chk_p049_ss_total CHECK (total_tco2e >= 0),
    CONSTRAINT uq_p049_ss_round_site UNIQUE (round_id, site_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_ss_tenant          ON ghg_multisite.gl_ms_site_submissions(tenant_id);
CREATE INDEX idx_p049_ss_round           ON ghg_multisite.gl_ms_site_submissions(round_id);
CREATE INDEX idx_p049_ss_site            ON ghg_multisite.gl_ms_site_submissions(site_id);
CREATE INDEX idx_p049_ss_status          ON ghg_multisite.gl_ms_site_submissions(status);
CREATE INDEX idx_p049_ss_pending         ON ghg_multisite.gl_ms_site_submissions(status)
    WHERE status IN ('NOT_STARTED', 'DRAFT', 'NEEDS_REVISION');
CREATE INDEX idx_p049_ss_review          ON ghg_multisite.gl_ms_site_submissions(status)
    WHERE status IN ('SUBMITTED', 'UNDER_REVIEW');
CREATE INDEX idx_p049_ss_approved        ON ghg_multisite.gl_ms_site_submissions(round_id, status)
    WHERE status = 'APPROVED';
CREATE INDEX idx_p049_ss_quality         ON ghg_multisite.gl_ms_site_submissions(data_quality_score)
    WHERE data_quality_score IS NOT NULL;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_site_submissions ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_ss_tenant_isolation ON ghg_multisite.gl_ms_site_submissions
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 3: ghg_multisite.gl_ms_submission_data
-- =============================================================================
-- Individual emission data records within a site submission. Each row
-- represents one activity data point (e.g., electricity consumption for
-- January) with its emission factor, calculated emissions, and data source.

CREATE TABLE ghg_multisite.gl_ms_submission_data (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    submission_id               UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_site_submissions(id) ON DELETE CASCADE,
    site_id                     UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    scope                       VARCHAR(10)     NOT NULL,
    category                    VARCHAR(100)    NOT NULL,
    subcategory                 VARCHAR(100),
    source_description          VARCHAR(500),
    activity_value              NUMERIC(20,6)   NOT NULL,
    activity_unit               VARCHAR(50)     NOT NULL,
    emission_factor_id          UUID,
    emission_factor_value       NUMERIC(20,10),
    emission_factor_unit        VARCHAR(100),
    emission_factor_source      VARCHAR(255),
    co2_tco2e                   NUMERIC(20,6)   NOT NULL DEFAULT 0,
    ch4_tco2e                   NUMERIC(20,6)   NOT NULL DEFAULT 0,
    n2o_tco2e                   NUMERIC(20,6)   NOT NULL DEFAULT 0,
    other_ghg_tco2e             NUMERIC(20,6)   NOT NULL DEFAULT 0,
    total_tco2e                 NUMERIC(20,6)   NOT NULL DEFAULT 0,
    data_quality_tier           INTEGER,
    data_source                 VARCHAR(50)     NOT NULL DEFAULT 'MANUAL',
    is_estimated                BOOLEAN         NOT NULL DEFAULT false,
    estimation_method           VARCHAR(100),
    reporting_month             DATE,
    evidence_ref                VARCHAR(255),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_sd_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3')
    ),
    CONSTRAINT chk_p049_sd_source CHECK (
        data_source IN (
            'MANUAL', 'ERP', 'METER', 'INVOICE', 'ESTIMATE',
            'API', 'IOT', 'SUPPLIER', 'CALCULATED'
        )
    ),
    CONSTRAINT chk_p049_sd_tier CHECK (
        data_quality_tier IS NULL OR (data_quality_tier >= 1 AND data_quality_tier <= 5)
    ),
    CONSTRAINT chk_p049_sd_activity CHECK (activity_value >= 0),
    CONSTRAINT chk_p049_sd_co2 CHECK (co2_tco2e >= 0),
    CONSTRAINT chk_p049_sd_ch4 CHECK (ch4_tco2e >= 0),
    CONSTRAINT chk_p049_sd_n2o CHECK (n2o_tco2e >= 0),
    CONSTRAINT chk_p049_sd_other CHECK (other_ghg_tco2e >= 0),
    CONSTRAINT chk_p049_sd_total CHECK (total_tco2e >= 0)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_sd_tenant          ON ghg_multisite.gl_ms_submission_data(tenant_id);
CREATE INDEX idx_p049_sd_submission      ON ghg_multisite.gl_ms_submission_data(submission_id);
CREATE INDEX idx_p049_sd_site            ON ghg_multisite.gl_ms_submission_data(site_id);
CREATE INDEX idx_p049_sd_scope           ON ghg_multisite.gl_ms_submission_data(scope);
CREATE INDEX idx_p049_sd_category        ON ghg_multisite.gl_ms_submission_data(category);
CREATE INDEX idx_p049_sd_source          ON ghg_multisite.gl_ms_submission_data(data_source);
CREATE INDEX idx_p049_sd_month           ON ghg_multisite.gl_ms_submission_data(reporting_month)
    WHERE reporting_month IS NOT NULL;
CREATE INDEX idx_p049_sd_estimated       ON ghg_multisite.gl_ms_submission_data(is_estimated)
    WHERE is_estimated = true;
CREATE INDEX idx_p049_sd_tier            ON ghg_multisite.gl_ms_submission_data(data_quality_tier)
    WHERE data_quality_tier IS NOT NULL;
CREATE INDEX idx_p049_sd_ef              ON ghg_multisite.gl_ms_submission_data(emission_factor_id)
    WHERE emission_factor_id IS NOT NULL;
CREATE INDEX idx_p049_sd_meta            ON ghg_multisite.gl_ms_submission_data USING gin(metadata);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_submission_data ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_sd_tenant_isolation ON ghg_multisite.gl_ms_submission_data
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 4: ghg_multisite.gl_ms_validation_results
-- =============================================================================
-- Validation results produced by the data quality engine during submission
-- review. Each row is one rule evaluation (pass/fail/warning) against a
-- specific submission or individual data record.

CREATE TABLE ghg_multisite.gl_ms_validation_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    submission_id               UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_site_submissions(id) ON DELETE CASCADE,
    data_record_id              UUID            REFERENCES ghg_multisite.gl_ms_submission_data(id) ON DELETE SET NULL,
    rule_code                   VARCHAR(100)    NOT NULL,
    rule_name                   VARCHAR(255)    NOT NULL,
    rule_category               VARCHAR(50)     NOT NULL DEFAULT 'COMPLETENESS',
    severity                    VARCHAR(20)     NOT NULL DEFAULT 'WARNING',
    result                      VARCHAR(10)     NOT NULL DEFAULT 'PASS',
    message                     TEXT,
    expected_value              VARCHAR(255),
    actual_value                VARCHAR(255),
    remediation_hint            TEXT,
    is_blocking                 BOOLEAN         NOT NULL DEFAULT false,
    validated_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_vr_category CHECK (
        rule_category IN (
            'COMPLETENESS', 'ACCURACY', 'CONSISTENCY', 'TIMELINESS',
            'PLAUSIBILITY', 'CROSS_CHECK', 'BOUNDARY', 'FACTOR',
            'THRESHOLD', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p049_vr_severity CHECK (
        severity IN ('CRITICAL', 'ERROR', 'WARNING', 'INFO')
    ),
    CONSTRAINT chk_p049_vr_result CHECK (
        result IN ('PASS', 'FAIL', 'WARNING', 'SKIP', 'ERROR')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_vr_tenant          ON ghg_multisite.gl_ms_validation_results(tenant_id);
CREATE INDEX idx_p049_vr_submission      ON ghg_multisite.gl_ms_validation_results(submission_id);
CREATE INDEX idx_p049_vr_record          ON ghg_multisite.gl_ms_validation_results(data_record_id)
    WHERE data_record_id IS NOT NULL;
CREATE INDEX idx_p049_vr_rule            ON ghg_multisite.gl_ms_validation_results(rule_code);
CREATE INDEX idx_p049_vr_category        ON ghg_multisite.gl_ms_validation_results(rule_category);
CREATE INDEX idx_p049_vr_severity        ON ghg_multisite.gl_ms_validation_results(severity);
CREATE INDEX idx_p049_vr_result          ON ghg_multisite.gl_ms_validation_results(result);
CREATE INDEX idx_p049_vr_failures        ON ghg_multisite.gl_ms_validation_results(submission_id, result)
    WHERE result IN ('FAIL', 'ERROR');
CREATE INDEX idx_p049_vr_blocking        ON ghg_multisite.gl_ms_validation_results(submission_id)
    WHERE is_blocking = true AND result = 'FAIL';

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_validation_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_vr_tenant_isolation ON ghg_multisite.gl_ms_validation_results
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_multisite.gl_ms_collection_rounds IS
    'PACK-049: Collection rounds with lifecycle status, completeness tracking, and site counts.';
COMMENT ON TABLE ghg_multisite.gl_ms_site_submissions IS
    'PACK-049: Per-site submission with review lifecycle (7 statuses), quality score, and scope totals.';
COMMENT ON TABLE ghg_multisite.gl_ms_submission_data IS
    'PACK-049: Individual emission records with activity data, emission factors, GHG breakdown, and quality tier.';
COMMENT ON TABLE ghg_multisite.gl_ms_validation_results IS
    'PACK-049: Rule-level validation results (10 categories, 4 severities) with remediation hints.';
