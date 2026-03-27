-- =============================================================================
-- PACK-050 GHG Consolidation Pack
-- Migration: V420 - Data Collection
-- =============================================================================
-- Pack:         PACK-050 (GHG Consolidation Pack)
-- Migration:    005 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates data collection tables for requesting and receiving entity-level
-- GHG submissions. Data requests are sent to each entity within the
-- consolidation boundary, and entity submissions contain scope-level
-- emission data with methodology, data quality, and provenance tracking.
--
-- Tables (2):
--   1. ghg_consolidation.gl_cons_data_requests
--   2. ghg_consolidation.gl_cons_entity_submissions
--
-- Also includes: indexes, RLS, constraints, comments.
-- Previous: V419__pack050_boundary.sql
-- Next:     V421__pack050_consolidation.sql
-- =============================================================================

SET search_path TO ghg_consolidation, public;

-- =============================================================================
-- Table 1: ghg_consolidation.gl_cons_data_requests
-- =============================================================================
-- Represents a request for GHG data from a specific entity for a given
-- reporting period. Tracks the request lifecycle from creation through
-- assignment, submission, validation, and approval. Supports template
-- versioning for consistent data collection formats.

CREATE TABLE ghg_consolidation.gl_cons_data_requests (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    boundary_id                 UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_boundaries(id) ON DELETE CASCADE,
    entity_id                   UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE CASCADE,
    reporting_period_start      DATE            NOT NULL,
    reporting_period_end        DATE            NOT NULL,
    template_version            VARCHAR(20)     NOT NULL DEFAULT '1.0',
    template_config             JSONB           DEFAULT '{}',
    assigned_to                 UUID,
    assigned_to_name            VARCHAR(255),
    assigned_to_email           VARCHAR(255),
    due_date                    DATE            NOT NULL,
    reminder_sent_at            TIMESTAMPTZ,
    reminder_count              INTEGER         NOT NULL DEFAULT 0,
    status                      VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    submitted_at                TIMESTAMPTZ,
    submitted_by                UUID,
    validated_at                TIMESTAMPTZ,
    validated_by                UUID,
    validation_notes            TEXT,
    approved_at                 TIMESTAMPTZ,
    approved_by                 UUID,
    rejection_reason            TEXT,
    scopes_requested            VARCHAR(30)[]   DEFAULT ARRAY['SCOPE_1', 'SCOPE_2', 'SCOPE_3']::VARCHAR(30)[],
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p050_dr_dates CHECK (
        reporting_period_end > reporting_period_start
    ),
    CONSTRAINT chk_p050_dr_due_date CHECK (
        due_date >= reporting_period_start
    ),
    CONSTRAINT chk_p050_dr_status CHECK (
        status IN (
            'PENDING', 'ASSIGNED', 'IN_PROGRESS', 'SUBMITTED',
            'VALIDATING', 'VALIDATED', 'REJECTED', 'APPROVED',
            'OVERDUE', 'CANCELLED'
        )
    ),
    CONSTRAINT chk_p050_dr_reminder_count CHECK (reminder_count >= 0),
    CONSTRAINT chk_p050_dr_rejection CHECK (
        status != 'REJECTED' OR rejection_reason IS NOT NULL
    ),
    CONSTRAINT uq_p050_dr_boundary_entity UNIQUE (boundary_id, entity_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_dr_tenant          ON ghg_consolidation.gl_cons_data_requests(tenant_id);
CREATE INDEX idx_p050_dr_boundary        ON ghg_consolidation.gl_cons_data_requests(boundary_id);
CREATE INDEX idx_p050_dr_entity          ON ghg_consolidation.gl_cons_data_requests(entity_id);
CREATE INDEX idx_p050_dr_status          ON ghg_consolidation.gl_cons_data_requests(status);
CREATE INDEX idx_p050_dr_assigned        ON ghg_consolidation.gl_cons_data_requests(assigned_to)
    WHERE assigned_to IS NOT NULL;
CREATE INDEX idx_p050_dr_due_date        ON ghg_consolidation.gl_cons_data_requests(due_date);
CREATE INDEX idx_p050_dr_overdue         ON ghg_consolidation.gl_cons_data_requests(due_date, status)
    WHERE status NOT IN ('APPROVED', 'CANCELLED');
CREATE INDEX idx_p050_dr_period          ON ghg_consolidation.gl_cons_data_requests(reporting_period_start, reporting_period_end);
CREATE INDEX idx_p050_dr_pending         ON ghg_consolidation.gl_cons_data_requests(tenant_id, status)
    WHERE status IN ('PENDING', 'ASSIGNED', 'IN_PROGRESS');
CREATE INDEX idx_p050_dr_submitted       ON ghg_consolidation.gl_cons_data_requests(tenant_id, status)
    WHERE status = 'SUBMITTED';

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_data_requests ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_dr_tenant_isolation ON ghg_consolidation.gl_cons_data_requests
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 2: ghg_consolidation.gl_cons_entity_submissions
-- =============================================================================
-- Entity-level emission submissions linked to a data request. Each row
-- represents one scope/category combination for an entity. Captures
-- the emission value, activity data, emission factor, methodology,
-- and data quality tier. Supports multiple submissions per request
-- (e.g., one per scope or per emission category).

CREATE TABLE ghg_consolidation.gl_cons_entity_submissions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    request_id                  UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_data_requests(id) ON DELETE CASCADE,
    entity_id                   UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE CASCADE,
    scope                       VARCHAR(20)     NOT NULL,
    category                    VARCHAR(50),
    category_description        VARCHAR(255),
    emissions_tco2e             NUMERIC(20,6)   NOT NULL,
    co2_tco2e                   NUMERIC(20,6),
    ch4_tco2e                   NUMERIC(20,6),
    n2o_tco2e                   NUMERIC(20,6),
    hfc_tco2e                   NUMERIC(20,6),
    pfc_tco2e                   NUMERIC(20,6),
    sf6_tco2e                   NUMERIC(20,6),
    nf3_tco2e                   NUMERIC(20,6),
    activity_data               NUMERIC(20,6),
    activity_unit               VARCHAR(50),
    emission_factor             NUMERIC(20,10),
    emission_factor_unit        VARCHAR(100),
    emission_factor_source      VARCHAR(500),
    emission_factor_year        INTEGER,
    methodology                 VARCHAR(100)    NOT NULL DEFAULT 'GHG_PROTOCOL',
    methodology_detail          TEXT,
    calculation_approach        VARCHAR(50),
    data_quality_tier           INTEGER         NOT NULL DEFAULT 1,
    data_quality_score          NUMERIC(10,4),
    data_quality_notes          TEXT,
    is_estimated                BOOLEAN         NOT NULL DEFAULT false,
    estimation_method           VARCHAR(100),
    estimation_coverage_pct     NUMERIC(10,4),
    uncertainty_pct             NUMERIC(10,4),
    biogenic_emissions_tco2e    NUMERIC(20,6),
    evidence_refs               TEXT[],
    submitted_by                UUID,
    submitted_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    validated_by                UUID,
    validated_at                TIMESTAMPTZ,
    validation_status           VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    validation_notes            TEXT,
    provenance_hash             VARCHAR(64),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p050_es_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2_LOCATION', 'SCOPE_2_MARKET', 'SCOPE_3')
    ),
    CONSTRAINT chk_p050_es_emissions CHECK (emissions_tco2e >= 0),
    CONSTRAINT chk_p050_es_co2 CHECK (co2_tco2e IS NULL OR co2_tco2e >= 0),
    CONSTRAINT chk_p050_es_ch4 CHECK (ch4_tco2e IS NULL OR ch4_tco2e >= 0),
    CONSTRAINT chk_p050_es_n2o CHECK (n2o_tco2e IS NULL OR n2o_tco2e >= 0),
    CONSTRAINT chk_p050_es_hfc CHECK (hfc_tco2e IS NULL OR hfc_tco2e >= 0),
    CONSTRAINT chk_p050_es_pfc CHECK (pfc_tco2e IS NULL OR pfc_tco2e >= 0),
    CONSTRAINT chk_p050_es_sf6 CHECK (sf6_tco2e IS NULL OR sf6_tco2e >= 0),
    CONSTRAINT chk_p050_es_nf3 CHECK (nf3_tco2e IS NULL OR nf3_tco2e >= 0),
    CONSTRAINT chk_p050_es_dq_tier CHECK (
        data_quality_tier >= 1 AND data_quality_tier <= 5
    ),
    CONSTRAINT chk_p050_es_dq_score CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p050_es_uncertainty CHECK (
        uncertainty_pct IS NULL OR (uncertainty_pct >= 0 AND uncertainty_pct <= 100)
    ),
    CONSTRAINT chk_p050_es_est_coverage CHECK (
        estimation_coverage_pct IS NULL OR (estimation_coverage_pct >= 0 AND estimation_coverage_pct <= 100)
    ),
    CONSTRAINT chk_p050_es_biogenic CHECK (
        biogenic_emissions_tco2e IS NULL OR biogenic_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p050_es_validation CHECK (
        validation_status IN ('PENDING', 'VALID', 'INVALID', 'WARNING', 'OVERRIDE')
    ),
    CONSTRAINT chk_p050_es_methodology CHECK (
        methodology IN (
            'GHG_PROTOCOL', 'ISO_14064', 'EPA', 'DEFRA', 'IPCC',
            'SUPPLIER_SPECIFIC', 'SPEND_BASED', 'AVERAGE_DATA',
            'HYBRID', 'DIRECT_MEASUREMENT', 'OTHER'
        )
    ),
    CONSTRAINT chk_p050_es_ef_year CHECK (
        emission_factor_year IS NULL OR (emission_factor_year >= 1990 AND emission_factor_year <= 2100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_es_tenant          ON ghg_consolidation.gl_cons_entity_submissions(tenant_id);
CREATE INDEX idx_p050_es_request         ON ghg_consolidation.gl_cons_entity_submissions(request_id);
CREATE INDEX idx_p050_es_entity          ON ghg_consolidation.gl_cons_entity_submissions(entity_id);
CREATE INDEX idx_p050_es_scope           ON ghg_consolidation.gl_cons_entity_submissions(scope);
CREATE INDEX idx_p050_es_category        ON ghg_consolidation.gl_cons_entity_submissions(category)
    WHERE category IS NOT NULL;
CREATE INDEX idx_p050_es_validation      ON ghg_consolidation.gl_cons_entity_submissions(validation_status);
CREATE INDEX idx_p050_es_dq_tier         ON ghg_consolidation.gl_cons_entity_submissions(data_quality_tier);
CREATE INDEX idx_p050_es_estimated       ON ghg_consolidation.gl_cons_entity_submissions(request_id)
    WHERE is_estimated = true;
CREATE INDEX idx_p050_es_entity_scope    ON ghg_consolidation.gl_cons_entity_submissions(entity_id, scope);
CREATE INDEX idx_p050_es_request_scope   ON ghg_consolidation.gl_cons_entity_submissions(request_id, scope);
CREATE INDEX idx_p050_es_submitted       ON ghg_consolidation.gl_cons_entity_submissions(submitted_at);
CREATE INDEX idx_p050_es_methodology     ON ghg_consolidation.gl_cons_entity_submissions(methodology);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_entity_submissions ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_es_tenant_isolation ON ghg_consolidation.gl_cons_entity_submissions
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_consolidation.gl_cons_data_requests IS
    'PACK-050: Data requests to entities with assignment, due date, and 10-status lifecycle.';
COMMENT ON TABLE ghg_consolidation.gl_cons_entity_submissions IS
    'PACK-050: Entity emission submissions by scope/category with gas breakdown, methodology, and data quality.';
