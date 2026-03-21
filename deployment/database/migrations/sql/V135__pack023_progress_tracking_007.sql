-- =============================================================================
-- V135: PACK-023-sbti-alignment-007: Progress Tracking and Recalculation Records
-- =============================================================================
-- Pack:         PACK-023 (SBTi Alignment Pack)
-- Date:         March 2026
--
-- Pack-level tables for annual SBTi progress tracking and base year recalculation
-- management. Covers on-track/off-track assessment, variance analysis, corrective
-- action triggers, trajectory projection, and structural recalculations for M&A,
-- divestitures, methodology changes with 5% significance threshold.
--
-- EXTENDS:
--   V087: GL-SBTi-APP v1.0 (progress baseline)
--   V129: PACK-023 Target Definitions
--   V081: Audit Trail & Lineage Service
--
-- These tables sit in the pack023_sbti_alignment schema and provide
-- the progress tracking and recalculation assessment layer for the pack.
-- =============================================================================
-- Tables (4):
--   1. pack023_progress_tracking_records     - Annual progress assessment (HT)
--   2. pack023_progress_variance_analysis    - Detailed variance breakdown
--   3. pack023_recalculation_events          - Base year recalculation records
--   4. pack023_recalculation_adjustments     - Per-scope recalculation details
--
-- Hypertables (1):
--   pack023_progress_tracking_records on tracking_date (chunk: 3 months)
--
-- Also includes: 40+ indexes, update triggers, security grants, and comments.
-- Previous: V134__pack023_temperature_rating_006.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack023_sbti_alignment.pack023_progress_tracking_records
-- =============================================================================
-- Annual progress assessment records showing emissions vs. target pathway,
-- on-track/off-track status, and variance from reduction requirement.

CREATE TABLE pack023_sbti_alignment.pack023_progress_tracking_records (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    target_definition_id    UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_sbti_target_definitions(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    tracking_date           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    reporting_year          INTEGER         NOT NULL,
    target_year             INTEGER,
    years_from_baseline     INTEGER,
    baseline_emissions      DECIMAL(18,6),
    target_emissions        DECIMAL(18,6),
    actual_emissions        DECIMAL(18,6),
    emissions_variance      DECIMAL(18,6),
    variance_percentage     DECIMAL(8,4),
    required_annual_reduction DECIMAL(18,6),
    actual_annual_reduction DECIMAL(18,6),
    reduction_gap           DECIMAL(18,6),
    required_reduction_rate DECIMAL(6,4),
    actual_reduction_rate   DECIMAL(6,4),
    rate_variance           DECIMAL(6,4),
    status_rag              VARCHAR(30),
    on_track_status         VARCHAR(30),
    trajectory_2030         DECIMAL(18,6),
    trajectory_2050         DECIMAL(18,6),
    trajectory_alignment    VARCHAR(50),
    budget_remaining        DECIMAL(18,6),
    budget_remaining_pct    DECIMAL(6,2),
    corrective_action_triggered BOOLEAN     DEFAULT FALSE,
    corrective_action_type  VARCHAR(200),
    corrective_action_plan  TEXT,
    data_quality_assessment VARCHAR(30),
    confidence_level        VARCHAR(30),
    underlying_drivers      TEXT,
    external_factors        TEXT[],
    methodology_changes     TEXT[],
    growth_adjusted         BOOLEAN         DEFAULT FALSE,
    growth_impact           DECIMAL(8,4),
    structural_changes      TEXT[],
    structural_impact       DECIMAL(18,6),
    efficiency_improvement  DECIMAL(8,4),
    renewable_energy_contribution DECIMAL(8,4),
    scope3_contribution     DECIMAL(8,4),
    assessed_by             VARCHAR(255),
    approval_status         VARCHAR(30),
    approved_by             VARCHAR(255),
    approved_date           DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_prog_rag CHECK (
        status_rag IN ('GREEN', 'YELLOW', 'RED')
    ),
    CONSTRAINT chk_pk_prog_status CHECK (
        on_track_status IN ('ON_TRACK', 'OFF_TRACK', 'CRITICAL', 'INSUFFICIENT_DATA')
    ),
    CONSTRAINT chk_pk_prog_variance CHECK (
        variance_percentage IS NULL OR (variance_percentage >= -100 AND variance_percentage <= 100)
    )
);

-- Hypertable
SELECT create_hypertable(
    'pack023_sbti_alignment.pack023_progress_tracking_records',
    'tracking_date',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '3 months'
);

-- Indexes
CREATE INDEX idx_pk_prog_target_id ON pack023_sbti_alignment.pack023_progress_tracking_records(target_definition_id);
CREATE INDEX idx_pk_prog_tenant ON pack023_sbti_alignment.pack023_progress_tracking_records(tenant_id);
CREATE INDEX idx_pk_prog_org ON pack023_sbti_alignment.pack023_progress_tracking_records(org_id);
CREATE INDEX idx_pk_prog_date ON pack023_sbti_alignment.pack023_progress_tracking_records(tracking_date DESC);
CREATE INDEX idx_pk_prog_year ON pack023_sbti_alignment.pack023_progress_tracking_records(reporting_year);
CREATE INDEX idx_pk_prog_status ON pack023_sbti_alignment.pack023_progress_tracking_records(on_track_status);
CREATE INDEX idx_pk_prog_rag ON pack023_sbti_alignment.pack023_progress_tracking_records(status_rag);
CREATE INDEX idx_pk_prog_corrective ON pack023_sbti_alignment.pack023_progress_tracking_records(corrective_action_triggered);
CREATE INDEX idx_pk_prog_org_year ON pack023_sbti_alignment.pack023_progress_tracking_records(org_id, reporting_year DESC);
CREATE INDEX idx_pk_prog_external_factors ON pack023_sbti_alignment.pack023_progress_tracking_records USING GIN(external_factors);
CREATE INDEX idx_pk_prog_metadata ON pack023_sbti_alignment.pack023_progress_tracking_records USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_pk_prog_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_progress_tracking_records
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 2: pack023_sbti_alignment.pack023_progress_variance_analysis
-- =============================================================================
-- Detailed variance analysis breakdown by scope showing contributions to
-- overall variance from Scope 1, 2, and 3 with driver analysis.

CREATE TABLE pack023_sbti_alignment.pack023_progress_variance_analysis (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    progress_record_id      UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_progress_tracking_records(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    reporting_year          INTEGER,
    scope                   VARCHAR(30)     NOT NULL,
    baseline_emissions      DECIMAL(18,6),
    target_emissions        DECIMAL(18,6),
    actual_emissions        DECIMAL(18,6),
    variance_mt             DECIMAL(18,6),
    variance_pct            DECIMAL(8,4),
    primary_variance_driver VARCHAR(200),
    activity_variance       DECIMAL(18,6),
    activity_variance_pct   DECIMAL(6,2),
    efficiency_variance     DECIMAL(18,6),
    efficiency_variance_pct DECIMAL(6,2),
    factor_variance         DECIMAL(18,6),
    factor_variance_pct     DECIMAL(6,2),
    methodology_variance    DECIMAL(18,6),
    methodology_variance_pct DECIMAL(6,2),
    structural_variance     DECIMAL(18,6),
    structural_variance_pct DECIMAL(6,2),
    scope1_fuel_mix_change  DECIMAL(6,2),
    scope1_output_change    DECIMAL(6,2),
    scope2_grid_factor_change DECIMAL(6,2),
    scope2_renewable_increase DECIMAL(6,2),
    scope3_category_changes TEXT[],
    scope3_supplier_engagement DECIMAL(6,2),
    key_drivers             TEXT[],
    supporting_evidence     TEXT[],
    data_quality_tier       VARCHAR(30),
    verification_status     VARCHAR(30),
    notes                   TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_var_scope CHECK (
        scope IN ('SCOPE1', 'SCOPE2', 'SCOPE3')
    ),
    CONSTRAINT chk_pk_var_breakdown CHECK (
        ABS(activity_variance_pct) + ABS(efficiency_variance_pct) +
        ABS(factor_variance_pct) + ABS(methodology_variance_pct) > 0
    )
);

-- Indexes
CREATE INDEX idx_pk_var_prog_id ON pack023_sbti_alignment.pack023_progress_variance_analysis(progress_record_id);
CREATE INDEX idx_pk_var_tenant ON pack023_sbti_alignment.pack023_progress_variance_analysis(tenant_id);
CREATE INDEX idx_pk_var_org ON pack023_sbti_alignment.pack023_progress_variance_analysis(org_id);
CREATE INDEX idx_pk_var_year ON pack023_sbti_alignment.pack023_progress_variance_analysis(reporting_year);
CREATE INDEX idx_pk_var_scope ON pack023_sbti_alignment.pack023_progress_variance_analysis(scope);
CREATE INDEX idx_pk_var_driver ON pack023_sbti_alignment.pack023_progress_variance_analysis(primary_variance_driver);
CREATE INDEX idx_pk_var_created_at ON pack023_sbti_alignment.pack023_progress_variance_analysis(created_at DESC);
CREATE INDEX idx_pk_var_drivers ON pack023_sbti_alignment.pack023_progress_variance_analysis USING GIN(key_drivers);
CREATE INDEX idx_pk_var_categories ON pack023_sbti_alignment.pack023_progress_variance_analysis USING GIN(scope3_category_changes);

-- Updated_at trigger
CREATE TRIGGER trg_pk_var_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_progress_variance_analysis
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 3: pack023_sbti_alignment.pack023_recalculation_events
-- =============================================================================
-- Base year recalculation event records tracking M&A, divestitures, methodology
-- changes with pre/post emissions and significance assessment (5% threshold).

CREATE TABLE pack023_sbti_alignment.pack023_recalculation_events (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    target_definition_id    UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_sbti_target_definitions(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    recalc_event_date       DATE            NOT NULL,
    event_type              VARCHAR(100)    NOT NULL,
    event_description       TEXT,
    effective_date          DATE,
    base_year_before        INTEGER,
    base_year_after         INTEGER,
    business_context        TEXT,
    is_retrospective        BOOLEAN         DEFAULT FALSE,
    retrospective_years     INTEGER,
    triggering_threshold    DECIMAL(6,2)    DEFAULT 5.0,
    emissions_impact_pct    DECIMAL(8,4),
    emissions_impact_mt     DECIMAL(18,6),
    exceeds_significance_threshold BOOLEAN  DEFAULT FALSE,
    requires_target_revision BOOLEAN        DEFAULT FALSE,
    revised_target_emissions DECIMAL(18,6),
    revised_reduction_pct   DECIMAL(8,4),
    equity_share_adjustment DECIMAL(6,2),
    scope1_adjustment       DECIMAL(18,6),
    scope2_adjustment       DECIMAL(18,6),
    scope3_adjustment       DECIMAL(18,6),
    methodology_changes     TEXT[],
    new_emission_factors    VARCHAR(200)[],
    data_sources_updated    VARCHAR(200)[],
    assurance_obtained      BOOLEAN         DEFAULT FALSE,
    assurance_level         VARCHAR(30),
    assurance_provider      VARCHAR(255),
    documented_by           VARCHAR(255),
    approval_status         VARCHAR(30),
    approved_by             VARCHAR(255),
    approved_date           DATE,
    sbti_notification_date  DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_recalc_type CHECK (
        event_type IN ('ACQUISITION', 'DIVESTITURE', 'MERGER', 'METHODOLOGY_CHANGE',
                      'STRUCTURAL_CHANGE', 'ORGANIC_GROWTH', 'BASELINE_REVISION')
    ),
    CONSTRAINT chk_pk_recalc_impact CHECK (
        emissions_impact_pct IS NULL OR (emissions_impact_pct >= -100 AND emissions_impact_pct <= 100)
    )
);

-- Indexes
CREATE INDEX idx_pk_recalc_target_id ON pack023_sbti_alignment.pack023_recalculation_events(target_definition_id);
CREATE INDEX idx_pk_recalc_tenant ON pack023_sbti_alignment.pack023_recalculation_events(tenant_id);
CREATE INDEX idx_pk_recalc_org ON pack023_sbti_alignment.pack023_recalculation_events(org_id);
CREATE INDEX idx_pk_recalc_date ON pack023_sbti_alignment.pack023_recalculation_events(recalc_event_date DESC);
CREATE INDEX idx_pk_recalc_type ON pack023_sbti_alignment.pack023_recalculation_events(event_type);
CREATE INDEX idx_pk_recalc_significant ON pack023_sbti_alignment.pack023_recalculation_events(exceeds_significance_threshold);
CREATE INDEX idx_pk_recalc_approval ON pack023_sbti_alignment.pack023_recalculation_events(approval_status);
CREATE INDEX idx_pk_recalc_base_year ON pack023_sbti_alignment.pack023_recalculation_events(base_year_before, base_year_after);
CREATE INDEX idx_pk_recalc_methods ON pack023_sbti_alignment.pack023_recalculation_events USING GIN(methodology_changes);
CREATE INDEX idx_pk_recalc_factors ON pack023_sbti_alignment.pack023_recalculation_events USING GIN(new_emission_factors);

-- Updated_at trigger
CREATE TRIGGER trg_pk_recalc_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_recalculation_events
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 4: pack023_sbti_alignment.pack023_recalculation_adjustments
-- =============================================================================
-- Per-scope recalculation adjustment details showing baseline revision by scope
-- with detailed calculation rationale and supporting evidence.

CREATE TABLE pack023_sbti_alignment.pack023_recalculation_adjustments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    recalculation_event_id  UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_recalculation_events(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    adjustment_scope        VARCHAR(30)     NOT NULL,
    base_year               INTEGER,
    reporting_year_affected INTEGER,
    baseline_emissions_before DECIMAL(18,6),
    baseline_emissions_after DECIMAL(18,6),
    adjustment_amount       DECIMAL(18,6),
    adjustment_percentage   DECIMAL(8,4),
    adjustment_rationale    TEXT,
    calculation_method      VARCHAR(200),
    data_sources            TEXT[],
    entity_portion          DECIMAL(6,2),
    equity_share_portion    DECIMAL(6,2),
    supporting_documentation TEXT[],
    verification_status     VARCHAR(30),
    verified_by             VARCHAR(255),
    verified_date           DATE,
    notes                   TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_adj_scope CHECK (
        adjustment_scope IN ('SCOPE1', 'SCOPE2', 'SCOPE3_TOTAL', 'SCOPE3_CATEGORY')
    ),
    CONSTRAINT chk_pk_adj_portion CHECK (
        (entity_portion IS NULL OR (entity_portion >= 0 AND entity_portion <= 100)) AND
        (equity_share_portion IS NULL OR (equity_share_portion >= 0 AND equity_share_portion <= 100))
    )
);

-- Indexes
CREATE INDEX idx_pk_adj_recalc_id ON pack023_sbti_alignment.pack023_recalculation_adjustments(recalculation_event_id);
CREATE INDEX idx_pk_adj_tenant ON pack023_sbti_alignment.pack023_recalculation_adjustments(tenant_id);
CREATE INDEX idx_pk_adj_org ON pack023_sbti_alignment.pack023_recalculation_adjustments(org_id);
CREATE INDEX idx_pk_adj_scope ON pack023_sbti_alignment.pack023_recalculation_adjustments(adjustment_scope);
CREATE INDEX idx_pk_adj_base_year ON pack023_sbti_alignment.pack023_recalculation_adjustments(base_year);
CREATE INDEX idx_pk_adj_verification ON pack023_sbti_alignment.pack023_recalculation_adjustments(verification_status);
CREATE INDEX idx_pk_adj_created_at ON pack023_sbti_alignment.pack023_recalculation_adjustments(created_at DESC);
CREATE INDEX idx_pk_adj_sources ON pack023_sbti_alignment.pack023_recalculation_adjustments USING GIN(data_sources);
CREATE INDEX idx_pk_adj_docs ON pack023_sbti_alignment.pack023_recalculation_adjustments USING GIN(supporting_documentation);

-- Updated_at trigger
CREATE TRIGGER trg_pk_adj_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_recalculation_adjustments
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Permissions & Grants
-- =============================================================================

GRANT USAGE ON SCHEMA pack023_sbti_alignment TO public;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA pack023_sbti_alignment TO public;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA pack023_sbti_alignment TO public;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE pack023_sbti_alignment.pack023_progress_tracking_records IS
'Annual progress assessment records showing emissions vs. pathway targets with on-track/off-track status and variance analysis.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_progress_variance_analysis IS
'Detailed variance breakdown by scope showing contributions to variance from activity, efficiency, factors, and methodology changes.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_recalculation_events IS
'Base year recalculation event records for M&A, divestitures, methodology changes with impact assessment and significance threshold checking.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_recalculation_adjustments IS
'Per-scope baseline adjustment details for recalculation events with calculation rationale and verification status.';
