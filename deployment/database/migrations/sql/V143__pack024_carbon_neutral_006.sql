-- =============================================================================
-- V143: PACK-024-carbon-neutral-006: Neutralization Balance Reconciliation
-- =============================================================================
-- Pack:         PACK-024 (Carbon Neutral Pack)
-- Date:         March 2026
--
-- Pack-level tables for neutralization balance tracking and reconciliation
-- with emissions footprint, carbon credit holdings, and retirement records
-- to verify carbon neutral status achievement and maintenance.
--
-- EXTENDS:
--   V142: Registry Retirements
--
-- These tables provide the reconciliation and balance verification framework
-- for carbon neutral claim substantiation and ongoing maintenance.
-- =============================================================================
-- Tables (4):
--   1. pack024_carbon_neutral.pack024_neutralization_balance         - Balance tracking
--   2. pack024_carbon_neutral.pack024_balance_reconciliation         - Reconciliation records
--   3. pack024_carbon_neutral.pack024_net_zero_achievement          - Achievement tracking
--   4. pack024_carbon_neutral.pack024_balance_trend_analysis        - Historical trends
--
-- Also includes: 50+ indexes, update triggers, security grants, and comments.
-- Previous: V142__pack024_carbon_neutral_005.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack024_carbon_neutral.pack024_neutralization_balance
-- =============================================================================
-- Current neutralization balance with emissions, credits, and net position.

CREATE TABLE pack024_carbon_neutral.pack024_neutralization_balance (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    reporting_year          INTEGER         NOT NULL,
    reporting_period_start  DATE            NOT NULL,
    reporting_period_end    DATE            NOT NULL,
    balance_calculation_date DATE            NOT NULL,
    scope                   VARCHAR(50),
    total_emissions_tco2e   DECIMAL(18,4)   NOT NULL,
    emissions_breakdown     JSONB           DEFAULT '{}',
    fugitive_emissions      DECIMAL(18,4),
    residual_emissions      DECIMAL(18,4),
    unavoidable_emissions   DECIMAL(18,4),
    hard_to_abate_emissions DECIMAL(18,4),
    total_credits_retired   DECIMAL(18,2)   NOT NULL,
    retired_credits_breakdown JSONB         DEFAULT '{}',
    credits_pending_retirement DECIMAL(18,2),
    credits_awaiting_verification DECIMAL(18,2),
    credits_surplus         DECIMAL(18,2),
    credits_deficit         DECIMAL(18,2),
    net_balance_position    VARCHAR(30),
    carbon_neutral_achieved BOOLEAN         DEFAULT FALSE,
    carbon_negative_achieved BOOLEAN        DEFAULT FALSE,
    balance_status          VARCHAR(30),
    balance_variance        DECIMAL(6,2),
    variance_reason         TEXT,
    variance_explanation    TEXT,
    confidence_in_balance   DECIMAL(5,2),
    data_completeness_pct   DECIMAL(6,2),
    missing_data_issues     TEXT[],
    calculation_methodology VARCHAR(255),
    calculation_date        DATE,
    calculated_by           VARCHAR(255),
    verified_by             VARCHAR(255),
    verification_date       DATE,
    verification_status     VARCHAR(30),
    audit_status            VARCHAR(30),
    audit_findings          TEXT[],
    restatement_required    BOOLEAN         DEFAULT FALSE,
    restatement_reason      TEXT,
    prior_year_comparison   JSONB           DEFAULT '{}',
    trend_analysis_result   VARCHAR(255),
    sustainability_of_balance TEXT,
    materiality_assessment  DECIMAL(5,2),
    materiality_threshold   DECIMAL(6,2),
    approved_for_disclosure BOOLEAN         DEFAULT FALSE,
    approval_date           DATE,
    approved_by             VARCHAR(255),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_bal_year CHECK (
        reporting_year >= 2000 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_pack024_bal_emissions_non_neg CHECK (
        total_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_pack024_bal_credits_non_neg CHECK (
        total_credits_retired >= 0
    ),
    CONSTRAINT chk_pack024_bal_confidence CHECK (
        confidence_in_balance >= 0 AND confidence_in_balance <= 100
    ),
    CONSTRAINT chk_pack024_bal_completeness CHECK (
        data_completeness_pct >= 0 AND data_completeness_pct <= 100
    )
);

-- Indexes
CREATE INDEX idx_pack024_bal_org ON pack024_carbon_neutral.pack024_neutralization_balance(org_id);
CREATE INDEX idx_pack024_bal_tenant ON pack024_carbon_neutral.pack024_neutralization_balance(tenant_id);
CREATE INDEX idx_pack024_bal_year ON pack024_carbon_neutral.pack024_neutralization_balance(reporting_year DESC);
CREATE INDEX idx_pack024_bal_calculation_date ON pack024_carbon_neutral.pack024_neutralization_balance(balance_calculation_date DESC);
CREATE INDEX idx_pack024_bal_scope ON pack024_carbon_neutral.pack024_neutralization_balance(scope);
CREATE INDEX idx_pack024_bal_net_position ON pack024_carbon_neutral.pack024_neutralization_balance(net_balance_position);
CREATE INDEX idx_pack024_bal_neutral_achieved ON pack024_carbon_neutral.pack024_neutralization_balance(carbon_neutral_achieved);
CREATE INDEX idx_pack024_bal_status ON pack024_carbon_neutral.pack024_neutralization_balance(balance_status);
CREATE INDEX idx_pack024_bal_verified ON pack024_carbon_neutral.pack024_neutralization_balance(verification_status);
CREATE INDEX idx_pack024_bal_audit ON pack024_carbon_neutral.pack024_neutralization_balance(audit_status);
CREATE INDEX idx_pack024_bal_approved ON pack024_carbon_neutral.pack024_neutralization_balance(approved_for_disclosure);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_bal_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_neutralization_balance
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 2: pack024_carbon_neutral.pack024_balance_reconciliation
-- =============================================================================
-- Reconciliation records between emissions and credit records.

CREATE TABLE pack024_carbon_neutral.pack024_balance_reconciliation (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    balance_id              UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_neutralization_balance(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    reconciliation_date     DATE            NOT NULL,
    reconciliation_type     VARCHAR(50)     NOT NULL,
    reconciliation_period   VARCHAR(100),
    emissions_source        VARCHAR(255),
    emission_record_id      UUID,
    emission_quantity       DECIMAL(18,4),
    emission_unit           VARCHAR(30),
    credit_source           VARCHAR(255),
    credit_record_id        UUID,
    credit_quantity         DECIMAL(18,2),
    credit_unit             VARCHAR(30),
    coverage_ratio          DECIMAL(6,4),
    coverage_percentage     DECIMAL(6,2),
    match_status            VARCHAR(30),
    match_quality_score     DECIMAL(5,2),
    discrepancy_detected    BOOLEAN         DEFAULT FALSE,
    discrepancy_amount      DECIMAL(18,4),
    discrepancy_percentage  DECIMAL(6,2),
    discrepancy_reason      TEXT,
    discrepancy_justification TEXT,
    reconciliation_action   VARCHAR(255),
    adjustment_made         BOOLEAN         DEFAULT FALSE,
    adjustment_type         VARCHAR(50),
    adjustment_amount       DECIMAL(18,4),
    adjustment_approved_by  VARCHAR(255),
    adjustment_approval_date DATE,
    source_of_adjustment    VARCHAR(100),
    variance_explanation    TEXT,
    materiality_assessment  DECIMAL(5,2),
    is_material             BOOLEAN         DEFAULT FALSE,
    impact_on_balance       TEXT,
    remediation_status      VARCHAR(30),
    remediation_action      TEXT,
    remediation_deadline    DATE,
    remediation_completion_date DATE,
    documentation_attached  BOOLEAN         DEFAULT TRUE,
    documentation_links     TEXT[],
    reconciliation_status   VARCHAR(30)     DEFAULT 'completed',
    approval_status         VARCHAR(30),
    approved_by             VARCHAR(255),
    approval_date           DATE,
    reconciled_by           VARCHAR(255),
    reviewed_by             VARCHAR(255),
    review_date             DATE,
    review_comments         TEXT,
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_recon_type CHECK (
        reconciliation_type IN ('QUARTERLY', 'ANNUAL', 'TRIGGERED', 'AUDIT_REQUIRED', 'SPECIAL_REQUEST')
    ),
    CONSTRAINT chk_pack024_recon_status CHECK (
        reconciliation_status IN ('IN_PROGRESS', 'COMPLETED', 'FAILED', 'REQUIRES_CORRECTION')
    ),
    CONSTRAINT chk_pack024_recon_coverage_valid CHECK (
        coverage_percentage IS NULL OR (coverage_percentage >= 0 AND coverage_percentage <= 100)
    )
);

-- Indexes
CREATE INDEX idx_pack024_recon_balance_id ON pack024_carbon_neutral.pack024_balance_reconciliation(balance_id);
CREATE INDEX idx_pack024_recon_org ON pack024_carbon_neutral.pack024_balance_reconciliation(org_id);
CREATE INDEX idx_pack024_recon_tenant ON pack024_carbon_neutral.pack024_balance_reconciliation(tenant_id);
CREATE INDEX idx_pack024_recon_date ON pack024_carbon_neutral.pack024_balance_reconciliation(reconciliation_date DESC);
CREATE INDEX idx_pack024_recon_type ON pack024_carbon_neutral.pack024_balance_reconciliation(reconciliation_type);
CREATE INDEX idx_pack024_recon_match ON pack024_carbon_neutral.pack024_balance_reconciliation(match_status);
CREATE INDEX idx_pack024_recon_discrepancy ON pack024_carbon_neutral.pack024_balance_reconciliation(discrepancy_detected);
CREATE INDEX idx_pack024_recon_status ON pack024_carbon_neutral.pack024_balance_reconciliation(reconciliation_status);
CREATE INDEX idx_pack024_recon_material ON pack024_carbon_neutral.pack024_balance_reconciliation(is_material);
CREATE INDEX idx_pack024_recon_remediation ON pack024_carbon_neutral.pack024_balance_reconciliation(remediation_status);
CREATE INDEX idx_pack024_recon_approval ON pack024_carbon_neutral.pack024_balance_reconciliation(approval_status);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_recon_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_balance_reconciliation
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 3: pack024_carbon_neutral.pack024_net_zero_achievement
-- =============================================================================
-- Carbon neutral/net zero achievement tracking and status monitoring.

CREATE TABLE pack024_carbon_neutral.pack024_net_zero_achievement (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    achievement_date        DATE            NOT NULL,
    assessment_year         INTEGER         NOT NULL,
    net_zero_claim_type     VARCHAR(50)     NOT NULL,
    target_net_zero_year    INTEGER,
    baseline_emissions      DECIMAL(18,4)   NOT NULL,
    baseline_year           INTEGER,
    target_emissions        DECIMAL(18,4),
    actual_emissions        DECIMAL(18,4),
    reduction_from_baseline DECIMAL(6,2),
    actual_removals         DECIMAL(18,4),
    removals_percentage     DECIMAL(6,2),
    remaining_emissions     DECIMAL(18,4),
    offset_credits_retired  DECIMAL(18,2),
    carbon_neutral_achieved BOOLEAN         DEFAULT FALSE,
    scopes_included         TEXT[],
    scope1_neutral          BOOLEAN         DEFAULT FALSE,
    scope2_neutral          BOOLEAN         DEFAULT FALSE,
    scope3_neutral          BOOLEAN         DEFAULT FALSE,
    carbon_negative_claimed BOOLEAN         DEFAULT FALSE,
    carbon_negative_amount  DECIMAL(18,4),
    achievement_methodology VARCHAR(255),
    methodology_description TEXT,
    residual_emissions_strategy TEXT,
    removal_strategy        TEXT,
    removal_methods         TEXT[],
    high_integrity_removals BOOLEAN         DEFAULT FALSE,
    removal_verification    BOOLEAN         DEFAULT FALSE,
    removal_verifier        VARCHAR(255),
    verification_date       DATE,
    additionality_verified  BOOLEAN         DEFAULT TRUE,
    permanence_guaranteed   INTEGER,
    permanence_assurance    TEXT,
    leakage_addressed       BOOLEAN         DEFAULT TRUE,
    leakage_mitigation      TEXT,
    double_counting_prevented BOOLEAN       DEFAULT TRUE,
    anti_double_counting_evidence TEXT,
    public_commitment_made  BOOLEAN         DEFAULT FALSE,
    commitment_details      TEXT,
    commitment_date         DATE,
    governance_oversight    BOOLEAN         DEFAULT TRUE,
    governance_structure    VARCHAR(255),
    board_responsibility    BOOLEAN         DEFAULT FALSE,
    executive_accountability BOOLEAN        DEFAULT FALSE,
    stakeholder_communication BOOLEAN       DEFAULT TRUE,
    disclosure_planned      BOOLEAN         DEFAULT FALSE,
    disclosure_date         DATE,
    disclosure_scope        VARCHAR(100),
    third_party_verification BOOLEAN        DEFAULT FALSE,
    verifier_name           VARCHAR(255),
    verification_completed  BOOLEAN         DEFAULT FALSE,
    verification_opinion    VARCHAR(50),
    certification_obtained  BOOLEAN         DEFAULT FALSE,
    certification_standard  VARCHAR(100),
    certification_body      VARCHAR(255),
    certification_date      DATE,
    near_term_targets_aligned BOOLEAN       DEFAULT FALSE,
    sbti_alignment          BOOLEAN         DEFAULT FALSE,
    paris_agreement_aligned BOOLEAN         DEFAULT FALSE,
    achievement_status      VARCHAR(30)     DEFAULT 'assessed',
    sustainability          VARCHAR(30),
    maintenance_plan        TEXT,
    monitoring_frequency    VARCHAR(50),
    review_schedule         VARCHAR(50),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_nz_type CHECK (
        net_zero_claim_type IN ('CARBON_NEUTRAL', 'NET_ZERO', 'CLIMATE_POSITIVE', 'CARBON_NEGATIVE')
    ),
    CONSTRAINT chk_pack024_nz_year CHECK (
        assessment_year >= 2000 AND assessment_year <= 2100
    ),
    CONSTRAINT chk_pack024_nz_emissions_non_neg CHECK (
        actual_emissions >= 0
    )
);

-- Indexes
CREATE INDEX idx_pack024_nz_org ON pack024_carbon_neutral.pack024_net_zero_achievement(org_id);
CREATE INDEX idx_pack024_nz_tenant ON pack024_carbon_neutral.pack024_net_zero_achievement(tenant_id);
CREATE INDEX idx_pack024_nz_date ON pack024_carbon_neutral.pack024_net_zero_achievement(achievement_date DESC);
CREATE INDEX idx_pack024_nz_year ON pack024_carbon_neutral.pack024_net_zero_achievement(assessment_year DESC);
CREATE INDEX idx_pack024_nz_claim_type ON pack024_carbon_neutral.pack024_net_zero_achievement(net_zero_claim_type);
CREATE INDEX idx_pack024_nz_target_year ON pack024_carbon_neutral.pack024_net_zero_achievement(target_net_zero_year);
CREATE INDEX idx_pack024_nz_achieved ON pack024_carbon_neutral.pack024_net_zero_achievement(carbon_neutral_achieved);
CREATE INDEX idx_pack024_nz_negative ON pack024_carbon_neutral.pack024_net_zero_achievement(carbon_negative_claimed);
CREATE INDEX idx_pack024_nz_verified ON pack024_carbon_neutral.pack024_net_zero_achievement(third_party_verification);
CREATE INDEX idx_pack024_nz_certified ON pack024_carbon_neutral.pack024_net_zero_achievement(certification_obtained);
CREATE INDEX idx_pack024_nz_sbti ON pack024_carbon_neutral.pack024_net_zero_achievement(sbti_alignment);
CREATE INDEX idx_pack024_nz_status ON pack024_carbon_neutral.pack024_net_zero_achievement(achievement_status);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_nz_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_net_zero_achievement
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 4: pack024_carbon_neutral.pack024_balance_trend_analysis
-- =============================================================================
-- Historical trend analysis for balance and progress tracking.

CREATE TABLE pack024_carbon_neutral.pack024_balance_trend_analysis (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    analysis_date           DATE            NOT NULL,
    analysis_period         VARCHAR(30)     NOT NULL,
    period_start_date       DATE,
    period_end_date         DATE,
    baseline_year           INTEGER,
    reporting_years_covered INTEGER[],
    baseline_emissions      DECIMAL(18,4),
    current_emissions       DECIMAL(18,4),
    emissions_reduction_pct DECIMAL(6,2),
    emissions_trend         VARCHAR(30),
    trend_slope             DECIMAL(6,4),
    reduction_trajectory    VARCHAR(255),
    trajectory_alignment    VARCHAR(30),
    on_track_for_target     BOOLEAN         DEFAULT FALSE,
    target_year             INTEGER,
    years_to_target         INTEGER,
    annual_reduction_rate   DECIMAL(6,4),
    required_annual_rate    DECIMAL(6,4),
    pace_assessment         VARCHAR(30),
    velocity_trend          VARCHAR(30),
    carbon_neutrality_progression JSONB     DEFAULT '{}',
    neutrality_trend_direction VARCHAR(30),
    credit_procurement_trend DECIMAL(6,2),
    credit_cost_trend       DECIMAL(6,2),
    portfolio_diversification_trend DECIMAL(6,2),
    portfolio_concentration_trend DECIMAL(6,2),
    geographic_diversification_trend DECIMAL(6,2),
    standard_concentration_trend DECIMAL(6,2),
    net_position_trend      VARCHAR(30),
    surplus_deficit_trend   DECIMAL(18,4),
    progress_visualization  JSONB           DEFAULT '{}',
    milestone_achievement   JSONB           DEFAULT '{}',
    completion_percentage   DECIMAL(6,2),
    forecast_analysis       JSONB           DEFAULT '{}',
    forecast_net_zero_year  INTEGER,
    forecast_confidence     DECIMAL(5,2),
    risk_assessment_trend   VARCHAR(30),
    risk_mitigation_effectiveness DECIMAL(5,2),
    scenario_analysis_results JSONB         DEFAULT '{}',
    best_case_year          INTEGER,
    worst_case_year         INTEGER,
    base_case_year          INTEGER,
    analysis_confidence     DECIMAL(5,2),
    data_quality_assessment VARCHAR(30),
    missing_data_impact     TEXT,
    recommendations         TEXT[],
    action_items            TEXT[],
    analyst_name            VARCHAR(255),
    analyst_comments        TEXT,
    approved_by             VARCHAR(255),
    approval_date           DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_trend_period CHECK (
        analysis_period IN ('MONTHLY', 'QUARTERLY', 'ANNUAL', 'CUSTOM')
    ),
    CONSTRAINT chk_pack024_trend_on_track CHECK (
        pace_assessment IN ('AHEAD', 'ON_TRACK', 'AT_RISK', 'BEHIND', 'UNKNOWN')
    )
);

-- Indexes
CREATE INDEX idx_pack024_trend_org ON pack024_carbon_neutral.pack024_balance_trend_analysis(org_id);
CREATE INDEX idx_pack024_trend_tenant ON pack024_carbon_neutral.pack024_balance_trend_analysis(tenant_id);
CREATE INDEX idx_pack024_trend_date ON pack024_carbon_neutral.pack024_balance_trend_analysis(analysis_date DESC);
CREATE INDEX idx_pack024_trend_period ON pack024_carbon_neutral.pack024_balance_trend_analysis(analysis_period);
CREATE INDEX idx_pack024_trend_baseline_year ON pack024_carbon_neutral.pack024_balance_trend_analysis(baseline_year);
CREATE INDEX idx_pack024_trend_target_year ON pack024_carbon_neutral.pack024_balance_trend_analysis(target_year);
CREATE INDEX idx_pack024_trend_on_track ON pack024_carbon_neutral.pack024_balance_trend_analysis(on_track_for_target);
CREATE INDEX idx_pack024_trend_trajectory ON pack024_carbon_neutral.pack024_balance_trend_analysis(trajectory_alignment);
CREATE INDEX idx_pack024_trend_pace ON pack024_carbon_neutral.pack024_balance_trend_analysis(pace_assessment);
CREATE INDEX idx_pack024_trend_velocity ON pack024_carbon_neutral.pack024_balance_trend_analysis(velocity_trend);
CREATE INDEX idx_pack024_trend_completion ON pack024_carbon_neutral.pack024_balance_trend_analysis(completion_percentage DESC);
CREATE INDEX idx_pack024_trend_confidence ON pack024_carbon_neutral.pack024_balance_trend_analysis(analysis_confidence DESC);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_trend_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_balance_trend_analysis
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Permissions & Grants
-- =============================================================================

GRANT USAGE ON SCHEMA pack024_carbon_neutral TO public;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA pack024_carbon_neutral TO public;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA pack024_carbon_neutral TO public;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE pack024_carbon_neutral.pack024_neutralization_balance IS
'Current neutralization balance with total emissions, credits retired, and net position verification for carbon neutral status.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_balance_reconciliation IS
'Reconciliation records between emissions and credit records with discrepancy detection and remediation tracking.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_net_zero_achievement IS
'Carbon neutral/net zero achievement tracking and status monitoring with verification and certification documentation.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_balance_trend_analysis IS
'Historical trend analysis for balance and progress tracking with forecasting and on-track assessment relative to targets.';
