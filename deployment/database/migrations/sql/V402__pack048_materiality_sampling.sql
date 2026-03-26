-- =============================================================================
-- PACK-048 GHG Assurance Prep Pack
-- Migration: V402 - Materiality & Sampling
-- =============================================================================
-- Pack:         PACK-048 (GHG Assurance Prep Pack)
-- Migration:    007 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for materiality assessment and statistical sampling plans.
-- Materiality assessments determine thresholds for overall, performance,
-- and clearly-trivial materiality with scope-level breakdowns and
-- qualitative factors. Sampling plans implement MUS, random, systematic,
-- stratified, and judgmental methods with confidence levels and tolerable
-- misstatement. Sample selections track strata with items tested and
-- projected misstatement.
--
-- Tables (3):
--   1. ghg_assurance.gl_ap_materiality_assessments
--   2. ghg_assurance.gl_ap_sampling_plans
--   3. ghg_assurance.gl_ap_sample_selections
--
-- Also includes: indexes, RLS, comments.
-- Previous: V401__pack048_verifier.sql
-- =============================================================================

SET search_path TO ghg_assurance, public;

-- =============================================================================
-- Table 1: ghg_assurance.gl_ap_materiality_assessments
-- =============================================================================
-- Materiality threshold determination for the assurance engagement.
-- Calculates overall materiality (typically 5% of total emissions),
-- performance materiality (typically 50-75% of overall), and clearly
-- trivial threshold (typically 5% of overall). Includes scope-level
-- breakdowns and qualitative factors that may adjust thresholds.

CREATE TABLE ghg_assurance.gl_ap_materiality_assessments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_configurations(id) ON DELETE CASCADE,
    total_emissions             NUMERIC(20,6)   NOT NULL,
    overall_materiality         NUMERIC(20,6)   NOT NULL,
    materiality_pct             NUMERIC(5,2)    NOT NULL,
    performance_materiality     NUMERIC(20,6)   NOT NULL,
    performance_pct             NUMERIC(5,2)    NOT NULL,
    clearly_trivial             NUMERIC(20,6)   NOT NULL,
    trivial_pct                 NUMERIC(5,2)    NOT NULL,
    scope_materiality           JSONB           NOT NULL DEFAULT '{}',
    qualitative_factors         JSONB           DEFAULT '[]',
    methodology_narrative       TEXT,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    assessed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p048_mat_total CHECK (
        total_emissions >= 0
    ),
    CONSTRAINT chk_p048_mat_overall CHECK (
        overall_materiality >= 0
    ),
    CONSTRAINT chk_p048_mat_pct CHECK (
        materiality_pct >= 0 AND materiality_pct <= 100
    ),
    CONSTRAINT chk_p048_mat_perf CHECK (
        performance_materiality >= 0
    ),
    CONSTRAINT chk_p048_mat_perf_pct CHECK (
        performance_pct >= 0 AND performance_pct <= 100
    ),
    CONSTRAINT chk_p048_mat_trivial CHECK (
        clearly_trivial >= 0
    ),
    CONSTRAINT chk_p048_mat_trivial_pct CHECK (
        trivial_pct >= 0 AND trivial_pct <= 100
    ),
    CONSTRAINT chk_p048_mat_hierarchy CHECK (
        clearly_trivial <= performance_materiality
        AND performance_materiality <= overall_materiality
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_mat_tenant           ON ghg_assurance.gl_ap_materiality_assessments(tenant_id);
CREATE INDEX idx_p048_mat_config           ON ghg_assurance.gl_ap_materiality_assessments(config_id);
CREATE INDEX idx_p048_mat_total            ON ghg_assurance.gl_ap_materiality_assessments(total_emissions);
CREATE INDEX idx_p048_mat_overall          ON ghg_assurance.gl_ap_materiality_assessments(overall_materiality);
CREATE INDEX idx_p048_mat_assessed         ON ghg_assurance.gl_ap_materiality_assessments(assessed_at DESC);
CREATE INDEX idx_p048_mat_created          ON ghg_assurance.gl_ap_materiality_assessments(created_at DESC);
CREATE INDEX idx_p048_mat_provenance       ON ghg_assurance.gl_ap_materiality_assessments(provenance_hash);
CREATE INDEX idx_p048_mat_scope            ON ghg_assurance.gl_ap_materiality_assessments USING GIN(scope_materiality);
CREATE INDEX idx_p048_mat_qualitative      ON ghg_assurance.gl_ap_materiality_assessments USING GIN(qualitative_factors);
CREATE INDEX idx_p048_mat_metadata         ON ghg_assurance.gl_ap_materiality_assessments USING GIN(metadata);

-- Composite: tenant + config for scoped listing
CREATE INDEX idx_p048_mat_tenant_config    ON ghg_assurance.gl_ap_materiality_assessments(tenant_id, config_id);

-- Composite: config + assessed_at for time series
CREATE INDEX idx_p048_mat_config_date      ON ghg_assurance.gl_ap_materiality_assessments(config_id, assessed_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_mat_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_materiality_assessments
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_assurance.gl_ap_sampling_plans
-- =============================================================================
-- Statistical sampling plans for assurance testing. Defines population
-- parameters, confidence level, tolerable misstatement, sampling method,
-- and calculated sample sizes for high-value items, key items, and
-- remaining population. Plans progress from DRAFT to COMPLETED.

CREATE TABLE ghg_assurance.gl_ap_sampling_plans (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_configurations(id) ON DELETE CASCADE,
    materiality_id              UUID            REFERENCES ghg_assurance.gl_ap_materiality_assessments(id) ON DELETE SET NULL,
    population_count            INTEGER         NOT NULL,
    population_value            NUMERIC(20,6)   NOT NULL,
    confidence_level            NUMERIC(5,2)    NOT NULL DEFAULT 95.00,
    tolerable_misstatement      NUMERIC(20,6)   NOT NULL,
    sampling_method             VARCHAR(20)     NOT NULL,
    total_sample_size           INTEGER         NOT NULL,
    high_value_count            INTEGER         NOT NULL DEFAULT 0,
    key_item_count              INTEGER         NOT NULL DEFAULT 0,
    remaining_sample            INTEGER         NOT NULL DEFAULT 0,
    plan_status                 VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p048_sp_pop_count CHECK (
        population_count >= 0
    ),
    CONSTRAINT chk_p048_sp_pop_value CHECK (
        population_value >= 0
    ),
    CONSTRAINT chk_p048_sp_confidence CHECK (
        confidence_level >= 50 AND confidence_level <= 99.99
    ),
    CONSTRAINT chk_p048_sp_tolerable CHECK (
        tolerable_misstatement >= 0
    ),
    CONSTRAINT chk_p048_sp_method CHECK (
        sampling_method IN (
            'MUS', 'RANDOM', 'SYSTEMATIC',
            'STRATIFIED', 'JUDGMENTAL'
        )
    ),
    CONSTRAINT chk_p048_sp_total_sample CHECK (
        total_sample_size >= 0
    ),
    CONSTRAINT chk_p048_sp_high_value CHECK (
        high_value_count >= 0
    ),
    CONSTRAINT chk_p048_sp_key_item CHECK (
        key_item_count >= 0
    ),
    CONSTRAINT chk_p048_sp_remaining CHECK (
        remaining_sample >= 0
    ),
    CONSTRAINT chk_p048_sp_status CHECK (
        plan_status IN ('DRAFT', 'APPROVED', 'IN_PROGRESS', 'COMPLETED')
    ),
    CONSTRAINT chk_p048_sp_sample_sum CHECK (
        total_sample_size >= high_value_count + key_item_count
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_sp_tenant            ON ghg_assurance.gl_ap_sampling_plans(tenant_id);
CREATE INDEX idx_p048_sp_config            ON ghg_assurance.gl_ap_sampling_plans(config_id);
CREATE INDEX idx_p048_sp_materiality       ON ghg_assurance.gl_ap_sampling_plans(materiality_id);
CREATE INDEX idx_p048_sp_method            ON ghg_assurance.gl_ap_sampling_plans(sampling_method);
CREATE INDEX idx_p048_sp_status            ON ghg_assurance.gl_ap_sampling_plans(plan_status);
CREATE INDEX idx_p048_sp_confidence        ON ghg_assurance.gl_ap_sampling_plans(confidence_level);
CREATE INDEX idx_p048_sp_created           ON ghg_assurance.gl_ap_sampling_plans(created_at DESC);
CREATE INDEX idx_p048_sp_provenance        ON ghg_assurance.gl_ap_sampling_plans(provenance_hash);
CREATE INDEX idx_p048_sp_metadata          ON ghg_assurance.gl_ap_sampling_plans USING GIN(metadata);

-- Composite: config + status for plan tracking
CREATE INDEX idx_p048_sp_config_status     ON ghg_assurance.gl_ap_sampling_plans(config_id, plan_status);

-- Composite: tenant + config for scoped listing
CREATE INDEX idx_p048_sp_tenant_config     ON ghg_assurance.gl_ap_sampling_plans(tenant_id, config_id);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_sp_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_sampling_plans
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_assurance.gl_ap_sample_selections
-- =============================================================================
-- Individual strata within a sampling plan. Each stratum defines a named
-- population segment with count, value, sample size, and selection method.
-- After testing, tracks items tested, errors found, and projected
-- misstatement for the stratum.

CREATE TABLE ghg_assurance.gl_ap_sample_selections (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    plan_id                     UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_sampling_plans(id) ON DELETE CASCADE,
    stratum_name                VARCHAR(100)    NOT NULL,
    stratum_count               INTEGER         NOT NULL,
    stratum_value               NUMERIC(20,6)   NOT NULL,
    sample_size                 INTEGER         NOT NULL,
    selection_method            VARCHAR(50),
    items_selected              JSONB           DEFAULT '[]',
    items_tested                INTEGER         NOT NULL DEFAULT 0,
    errors_found                INTEGER         NOT NULL DEFAULT 0,
    projected_misstatement      NUMERIC(20,6),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    tested_at                   TIMESTAMPTZ,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p048_ss_count CHECK (
        stratum_count >= 0
    ),
    CONSTRAINT chk_p048_ss_value CHECK (
        stratum_value >= 0
    ),
    CONSTRAINT chk_p048_ss_sample CHECK (
        sample_size >= 0
    ),
    CONSTRAINT chk_p048_ss_tested CHECK (
        items_tested >= 0 AND items_tested <= sample_size
    ),
    CONSTRAINT chk_p048_ss_errors CHECK (
        errors_found >= 0 AND errors_found <= items_tested
    ),
    CONSTRAINT chk_p048_ss_misstatement CHECK (
        projected_misstatement IS NULL OR projected_misstatement >= 0
    ),
    CONSTRAINT uq_p048_ss_plan_stratum UNIQUE (plan_id, stratum_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_ss_tenant            ON ghg_assurance.gl_ap_sample_selections(tenant_id);
CREATE INDEX idx_p048_ss_plan              ON ghg_assurance.gl_ap_sample_selections(plan_id);
CREATE INDEX idx_p048_ss_stratum           ON ghg_assurance.gl_ap_sample_selections(stratum_name);
CREATE INDEX idx_p048_ss_tested_at         ON ghg_assurance.gl_ap_sample_selections(tested_at DESC);
CREATE INDEX idx_p048_ss_created           ON ghg_assurance.gl_ap_sample_selections(created_at DESC);
CREATE INDEX idx_p048_ss_items             ON ghg_assurance.gl_ap_sample_selections USING GIN(items_selected);
CREATE INDEX idx_p048_ss_metadata          ON ghg_assurance.gl_ap_sample_selections USING GIN(metadata);

-- Composite: plan + stratum for ordered retrieval
CREATE INDEX idx_p048_ss_plan_stratum      ON ghg_assurance.gl_ap_sample_selections(plan_id, stratum_name);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_ss_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_sample_selections
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_assurance.gl_ap_materiality_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_assurance.gl_ap_sampling_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_assurance.gl_ap_sample_selections ENABLE ROW LEVEL SECURITY;

CREATE POLICY p048_mat_tenant_isolation
    ON ghg_assurance.gl_ap_materiality_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_mat_service_bypass
    ON ghg_assurance.gl_ap_materiality_assessments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p048_sp_tenant_isolation
    ON ghg_assurance.gl_ap_sampling_plans
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_sp_service_bypass
    ON ghg_assurance.gl_ap_sampling_plans
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p048_ss_tenant_isolation
    ON ghg_assurance.gl_ap_sample_selections
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_ss_service_bypass
    ON ghg_assurance.gl_ap_sample_selections
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_materiality_assessments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_sampling_plans TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_sample_selections TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_assurance.gl_ap_materiality_assessments IS
    'Materiality threshold determination with overall, performance, and clearly-trivial levels plus scope breakdowns.';
COMMENT ON TABLE ghg_assurance.gl_ap_sampling_plans IS
    'Statistical sampling plans with population parameters, confidence levels, and method-specific sample sizing.';
COMMENT ON TABLE ghg_assurance.gl_ap_sample_selections IS
    'Individual strata within sampling plans tracking selections, testing progress, and projected misstatement.';

COMMENT ON COLUMN ghg_assurance.gl_ap_materiality_assessments.overall_materiality IS 'Overall materiality threshold in tCO2e. Typically 5% of total emissions for GHG assurance.';
COMMENT ON COLUMN ghg_assurance.gl_ap_materiality_assessments.performance_materiality IS 'Performance materiality (50-75% of overall). Used for sampling and evaluating results.';
COMMENT ON COLUMN ghg_assurance.gl_ap_materiality_assessments.clearly_trivial IS 'Clearly-trivial threshold (typically 5% of overall materiality). Errors below this are not aggregated.';
COMMENT ON COLUMN ghg_assurance.gl_ap_materiality_assessments.scope_materiality IS 'JSON: scope-level materiality, e.g. {"SCOPE_1":{"materiality":500,"pct":5.0},"SCOPE_2":{"materiality":300,"pct":5.0}}.';
COMMENT ON COLUMN ghg_assurance.gl_ap_materiality_assessments.qualitative_factors IS 'JSON array of qualitative adjustments, e.g. [{"factor":"first_year_reporting","adjustment_pct":-10}].';
COMMENT ON COLUMN ghg_assurance.gl_ap_sampling_plans.sampling_method IS 'MUS (monetary unit), RANDOM, SYSTEMATIC (every nth), STRATIFIED (by strata), JUDGMENTAL (professional judgement).';
COMMENT ON COLUMN ghg_assurance.gl_ap_sampling_plans.high_value_count IS 'Items individually significant (> performance materiality). All tested.';
COMMENT ON COLUMN ghg_assurance.gl_ap_sampling_plans.key_item_count IS 'Items selected based on specific risk characteristics.';
COMMENT ON COLUMN ghg_assurance.gl_ap_sampling_plans.remaining_sample IS 'Statistically sampled items from remaining population after high-value and key item removal.';
COMMENT ON COLUMN ghg_assurance.gl_ap_sample_selections.projected_misstatement IS 'Projected misstatement for the stratum based on errors found, extrapolated to full stratum population.';
