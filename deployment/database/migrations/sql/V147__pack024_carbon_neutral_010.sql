-- =============================================================================
-- V147: PACK-024-carbon-neutral-010: Permanence Assessments
-- =============================================================================
-- Pack:         PACK-024 (Carbon Neutral Pack)
-- Date:         March 2026
--
-- Pack-level tables for carbon offset permanence assessment and risk management
-- including durability verification, reversal risk tracking, and mitigation
-- measures to ensure long-term environmental integrity of carbon neutral claims.
--
-- EXTENDS:
--   V146: Annual Cycles
--
-- These tables support the environmental integrity verification requirements
-- for carbon offset permanence and ensure claim credibility over time.
-- =============================================================================
-- Tables (4):
--   1. pack024_carbon_neutral.pack024_permanence_assessments       - Permanence risk analysis
--   2. pack024_carbon_neutral.pack024_permanence_risk_factors      - Individual risk factors
--   3. pack024_carbon_neutral.pack024_permanence_monitoring        - Ongoing monitoring
--   4. pack024_carbon_neutral.pack024_permanence_insurance         - Insurance/guarantees
--
-- Also includes: 50+ indexes, update triggers, security grants, and comments.
-- Previous: V146__pack024_carbon_neutral_009.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack024_carbon_neutral.pack024_permanence_assessments
-- =============================================================================
-- Comprehensive permanence risk assessments for offset portfolio.

CREATE TABLE pack024_carbon_neutral.pack024_permanence_assessments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    assessment_date         DATE            NOT NULL,
    assessment_year         INTEGER         NOT NULL,
    assessment_type         VARCHAR(50)     NOT NULL,
    portfolio_assessment    BOOLEAN         DEFAULT FALSE,
    project_specific        BOOLEAN         DEFAULT FALSE,
    project_id              VARCHAR(100),
    credit_inventory_id     UUID,
    assessment_period_years INTEGER,
    assessment_scope        VARCHAR(255),
    overall_permanence_risk VARCHAR(30)     NOT NULL,
    risk_score              DECIMAL(5,2)    NOT NULL,
    risk_rating             VARCHAR(30),
    confidence_level        DECIMAL(5,2),
    assessment_methodology  VARCHAR(255),
    methodology_standard    VARCHAR(100),
    baseline_conditions     TEXT,
    future_scenarios_analyzed BOOLEAN       DEFAULT FALSE,
    scenario_timeframe_years INTEGER,
    best_case_scenario      TEXT,
    worst_case_scenario     TEXT,
    most_likely_scenario    TEXT,
    permanence_guarantee_years INTEGER,
    guarantee_mechanism     VARCHAR(100),
    guarantee_provider      VARCHAR(255),
    guarantee_terms         TEXT,
    buffer_pool_participation BOOLEAN       DEFAULT FALSE,
    buffer_pool_percentage  DECIMAL(6,2),
    buffer_pool_justification TEXT,
    insurance_applied       BOOLEAN         DEFAULT FALSE,
    insurance_provider      VARCHAR(255),
    insurance_coverage_pct  DECIMAL(6,2),
    insurance_coverage_years INTEGER,
    material_risk_identified BOOLEAN        DEFAULT FALSE,
    material_risk_description TEXT,
    material_risk_mitigation TEXT,
    risk_threshold_exceeded BOOLEAN         DEFAULT FALSE,
    threshold_exceeded_action TEXT,
    monitoring_plan_in_place BOOLEAN        DEFAULT FALSE,
    monitoring_frequency    VARCHAR(50),
    monitoring_parameters   TEXT[],
    long_term_viability     VARCHAR(30),
    operational_continuity  VARCHAR(30),
    technical_permanence    VARCHAR(30),
    environmental_permanence VARCHAR(30),
    financial_permanence    VARCHAR(30),
    institutional_permanence VARCHAR(30),
    overall_assessment_conclusion TEXT,
    assessment_limitations  TEXT[],
    assumptions_documented  BOOLEAN         DEFAULT TRUE,
    assumption_details      JSONB           DEFAULT '{}',
    expert_judgment_applied BOOLEAN         DEFAULT FALSE,
    expert_names            TEXT[],
    expert_conclusion       TEXT,
    assessed_by             VARCHAR(255),
    assessment_organization VARCHAR(255),
    reviewer_name           VARCHAR(255),
    review_date             DATE,
    review_approval         BOOLEAN         DEFAULT FALSE,
    approval_comments       TEXT,
    next_assessment_date    DATE,
    assessment_frequency    VARCHAR(50)     DEFAULT 'annual',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_perm_risk CHECK (
        overall_permanence_risk IN ('VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH')
    ),
    CONSTRAINT chk_pack024_perm_score CHECK (
        risk_score >= 0 AND risk_score <= 100
    ),
    CONSTRAINT chk_pack024_perm_type CHECK (
        assessment_type IN ('INITIAL_ASSESSMENT', 'PERIODIC_ASSESSMENT', 'PRE_RETIREMENT',
                           'INCIDENT_INVESTIGATION', 'PORTFOLIO_REVIEW')
    ),
    CONSTRAINT chk_pack024_perm_confidence CHECK (
        confidence_level >= 0 AND confidence_level <= 100
    )
);

-- Indexes
CREATE INDEX idx_pack024_perm_org ON pack024_carbon_neutral.pack024_permanence_assessments(org_id);
CREATE INDEX idx_pack024_perm_tenant ON pack024_carbon_neutral.pack024_permanence_assessments(tenant_id);
CREATE INDEX idx_pack024_perm_date ON pack024_carbon_neutral.pack024_permanence_assessments(assessment_date DESC);
CREATE INDEX idx_pack024_perm_year ON pack024_carbon_neutral.pack024_permanence_assessments(assessment_year DESC);
CREATE INDEX idx_pack024_perm_type ON pack024_carbon_neutral.pack024_permanence_assessments(assessment_type);
CREATE INDEX idx_pack024_perm_project_id ON pack024_carbon_neutral.pack024_permanence_assessments(project_id);
CREATE INDEX idx_pack024_perm_risk ON pack024_carbon_neutral.pack024_permanence_assessments(overall_permanence_risk);
CREATE INDEX idx_pack024_perm_score ON pack024_carbon_neutral.pack024_permanence_assessments(risk_score);
CREATE INDEX idx_pack024_perm_rating ON pack024_carbon_neutral.pack024_permanence_assessments(risk_rating);
CREATE INDEX idx_pack024_perm_material_risk ON pack024_carbon_neutral.pack024_permanence_assessments(material_risk_identified);
CREATE INDEX idx_pack024_perm_insurance ON pack024_carbon_neutral.pack024_permanence_assessments(insurance_applied);
CREATE INDEX idx_pack024_perm_approval ON pack024_carbon_neutral.pack024_permanence_assessments(review_approval);
CREATE INDEX idx_pack024_perm_next_assessment ON pack024_carbon_neutral.pack024_permanence_assessments(next_assessment_date);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_perm_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_permanence_assessments
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 2: pack024_carbon_neutral.pack024_permanence_risk_factors
-- =============================================================================
-- Individual risk factors contributing to permanence assessment.

CREATE TABLE pack024_carbon_neutral.pack024_permanence_risk_factors (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    permanence_assessment_id UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_permanence_assessments(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    risk_factor_category    VARCHAR(100)    NOT NULL,
    risk_factor_name        VARCHAR(255)    NOT NULL,
    risk_factor_description TEXT,
    baseline_risk_level     VARCHAR(30),
    baseline_risk_score     DECIMAL(5,2),
    current_risk_level      VARCHAR(30),
    current_risk_score      DECIMAL(5,2),
    risk_trend              VARCHAR(30),
    risk_probability        DECIMAL(6,2),
    impact_if_occurs        VARCHAR(255),
    impact_on_carbon_reductions DECIMAL(6,2),
    impact_units            VARCHAR(50),
    affected_timeframe      VARCHAR(100),
    timeframe_years         INTEGER,
    risk_materiality        VARCHAR(30),
    is_material             BOOLEAN         DEFAULT FALSE,
    mitigation_available    BOOLEAN         DEFAULT FALSE,
    mitigation_strategy     TEXT,
    mitigation_effectiveness DECIMAL(5,2),
    residual_risk_score     DECIMAL(5,2),
    mitigation_cost_usd     DECIMAL(18,2),
    mitigation_timeline     VARCHAR(100),
    responsible_party       VARCHAR(255),
    monitoring_required     BOOLEAN         DEFAULT FALSE,
    monitoring_frequency    VARCHAR(50),
    monitoring_method       VARCHAR(255),
    monitoring_metric       VARCHAR(100),
    acceptable_threshold    VARCHAR(100),
    escalation_trigger      VARCHAR(100),
    escalation_procedure    TEXT,
    regulatory_relevance    VARCHAR(30),
    regulatory_requirement  VARCHAR(255),
    compliance_status       VARCHAR(30),
    control_in_place        BOOLEAN         DEFAULT FALSE,
    control_effectiveness   DECIMAL(5,2),
    evidence_of_control     TEXT[],
    historical_incidents    BOOLEAN         DEFAULT FALSE,
    incident_count          INTEGER         DEFAULT 0,
    incident_details        TEXT[],
    industry_prevalence     VARCHAR(30),
    comparable_projects     TEXT[],
    comparable_analysis     TEXT,
    risk_owner              VARCHAR(255),
    risk_acceptance_status  VARCHAR(30),
    accepted_by             VARCHAR(255),
    acceptance_date         DATE,
    acceptance_rationale    TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_factor_category CHECK (
        risk_factor_category IN ('NATURAL_HAZARDS', 'HUMAN_ACTIVITY', 'CLIMATE_CHANGE', 'FINANCIAL',
                                'INSTITUTIONAL', 'TECHNICAL', 'MARKET', 'REGULATORY', 'REPUTATIONAL')
    ),
    CONSTRAINT chk_pack024_factor_level CHECK (
        baseline_risk_level IN ('VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH')
    ),
    CONSTRAINT chk_pack024_factor_probability CHECK (
        risk_probability >= 0 AND risk_probability <= 100
    )
);

-- Indexes
CREATE INDEX idx_pack024_factor_assessment_id ON pack024_carbon_neutral.pack024_permanence_risk_factors(permanence_assessment_id);
CREATE INDEX idx_pack024_factor_org ON pack024_carbon_neutral.pack024_permanence_risk_factors(org_id);
CREATE INDEX idx_pack024_factor_tenant ON pack024_carbon_neutral.pack024_permanence_risk_factors(tenant_id);
CREATE INDEX idx_pack024_factor_category ON pack024_carbon_neutral.pack024_permanence_risk_factors(risk_factor_category);
CREATE INDEX idx_pack024_factor_name ON pack024_carbon_neutral.pack024_permanence_risk_factors(risk_factor_name);
CREATE INDEX idx_pack024_factor_current_level ON pack024_carbon_neutral.pack024_permanence_risk_factors(current_risk_level);
CREATE INDEX idx_pack024_factor_score ON pack024_carbon_neutral.pack024_permanence_risk_factors(current_risk_score DESC);
CREATE INDEX idx_pack024_factor_material ON pack024_carbon_neutral.pack024_permanence_risk_factors(is_material);
CREATE INDEX idx_pack024_factor_mitigation_available ON pack024_carbon_neutral.pack024_permanence_risk_factors(mitigation_available);
CREATE INDEX idx_pack024_factor_residual_score ON pack024_carbon_neutral.pack024_permanence_risk_factors(residual_risk_score);
CREATE INDEX idx_pack024_factor_acceptance ON pack024_carbon_neutral.pack024_permanence_risk_factors(risk_acceptance_status);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_factor_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_permanence_risk_factors
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 3: pack024_carbon_neutral.pack024_permanence_monitoring
-- =============================================================================
-- Ongoing permanence monitoring activities and results.

CREATE TABLE pack024_carbon_neutral.pack024_permanence_monitoring (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    monitoring_date         DATE            NOT NULL,
    monitoring_year         INTEGER         NOT NULL,
    project_id              VARCHAR(100),
    credit_inventory_id     UUID,
    monitoring_activity     VARCHAR(100)    NOT NULL,
    monitoring_frequency    VARCHAR(50),
    monitoring_scope        VARCHAR(255),
    monitoring_period_start DATE,
    monitoring_period_end   DATE,
    monitoring_method       VARCHAR(255)    NOT NULL,
    monitoring_tools_used   VARCHAR(255)[],
    data_collected          VARCHAR(255)[],
    data_quality            VARCHAR(30),
    data_completeness_pct   DECIMAL(6,2),
    monitoring_parameter_1  VARCHAR(100),
    parameter_1_value       VARCHAR(255),
    parameter_1_unit        VARCHAR(50),
    parameter_1_threshold   VARCHAR(100),
    parameter_1_status      VARCHAR(30),
    monitoring_parameter_2  VARCHAR(100),
    parameter_2_value       VARCHAR(255),
    parameter_2_unit        VARCHAR(50),
    parameter_2_threshold   VARCHAR(100),
    parameter_2_status      VARCHAR(30),
    monitoring_parameter_3  VARCHAR(100),
    parameter_3_value       VARCHAR(255),
    parameter_3_unit        VARCHAR(50),
    parameter_3_threshold   VARCHAR(100),
    parameter_3_status      VARCHAR(30),
    anomalies_detected      BOOLEAN         DEFAULT FALSE,
    anomaly_description     TEXT,
    anomaly_severity        VARCHAR(30),
    anomaly_investigation   TEXT,
    remedial_action_required BOOLEAN        DEFAULT FALSE,
    remedial_action         TEXT,
    remedial_action_deadline DATE,
    remedial_action_status  VARCHAR(30),
    reversal_risk_detected  BOOLEAN         DEFAULT FALSE,
    reversal_risk_type      VARCHAR(100),
    reversal_risk_magnitude DECIMAL(18,4),
    reversal_mitigation     TEXT,
    reversals_prevented     DECIMAL(18,4),
    monitoring_cost_usd     DECIMAL(18,2),
    monitoring_status       VARCHAR(30)     DEFAULT 'completed',
    report_prepared         BOOLEAN         DEFAULT FALSE,
    report_date             DATE,
    report_reviewer         VARCHAR(255),
    review_date             DATE,
    review_approval         BOOLEAN         DEFAULT FALSE,
    review_comments         TEXT,
    third_party_verified    BOOLEAN         DEFAULT FALSE,
    verifier_name           VARCHAR(255),
    verification_date       DATE,
    verification_opinion    VARCHAR(50),
    monitoring_performed_by VARCHAR(255),
    monitoring_organization VARCHAR(255),
    findings_summary        TEXT,
    concerns_raised         TEXT[],
    corrective_actions_recommended TEXT[],
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_monitor_activity CHECK (
        monitoring_activity IN ('SITE_INSPECTION', 'SATELLITE_MONITORING', 'COMMUNITY_ENGAGEMENT',
                               'DOCUMENT_REVIEW', 'INTERVIEW_STAKEHOLDERS', 'FIRE_MONITORING',
                               'PEST_MONITORING', 'CLIMATE_ASSESSMENT', 'HEALTH_CHECK')
    ),
    CONSTRAINT chk_pack024_monitor_status CHECK (
        monitoring_status IN ('PLANNED', 'IN_PROGRESS', 'COMPLETED', 'DEFERRED', 'CANCELLED')
    ),
    CONSTRAINT chk_pack024_monitor_data_quality CHECK (
        data_completeness_pct >= 0 AND data_completeness_pct <= 100
    )
);

-- Indexes
CREATE INDEX idx_pack024_monitor_org ON pack024_carbon_neutral.pack024_permanence_monitoring(org_id);
CREATE INDEX idx_pack024_monitor_tenant ON pack024_carbon_neutral.pack024_permanence_monitoring(tenant_id);
CREATE INDEX idx_pack024_monitor_date ON pack024_carbon_neutral.pack024_permanence_monitoring(monitoring_date DESC);
CREATE INDEX idx_pack024_monitor_year ON pack024_carbon_neutral.pack024_permanence_monitoring(monitoring_year DESC);
CREATE INDEX idx_pack024_monitor_project_id ON pack024_carbon_neutral.pack024_permanence_monitoring(project_id);
CREATE INDEX idx_pack024_monitor_activity ON pack024_carbon_neutral.pack024_permanence_monitoring(monitoring_activity);
CREATE INDEX idx_pack024_monitor_status ON pack024_carbon_neutral.pack024_permanence_monitoring(monitoring_status);
CREATE INDEX idx_pack024_monitor_anomalies ON pack024_carbon_neutral.pack024_permanence_monitoring(anomalies_detected);
CREATE INDEX idx_pack024_monitor_reversal_risk ON pack024_carbon_neutral.pack024_permanence_monitoring(reversal_risk_detected);
CREATE INDEX idx_pack024_monitor_verified ON pack024_carbon_neutral.pack024_permanence_monitoring(third_party_verified);
CREATE INDEX idx_pack024_monitor_approval ON pack024_carbon_neutral.pack024_permanence_monitoring(review_approval);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_monitor_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_permanence_monitoring
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 4: pack024_carbon_neutral.pack024_permanence_insurance
-- =============================================================================
-- Permanence insurance and guarantee coverage for carbon offset integrity.

CREATE TABLE pack024_carbon_neutral.pack024_permanence_insurance (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    policy_date             DATE            NOT NULL,
    policy_type             VARCHAR(50)     NOT NULL,
    project_id              VARCHAR(100),
    credit_inventory_id     UUID,
    insurance_provider      VARCHAR(255)    NOT NULL,
    provider_type           VARCHAR(100),
    policy_number           VARCHAR(100),
    policy_version          VARCHAR(20),
    policy_start_date       DATE            NOT NULL,
    policy_end_date         DATE,
    coverage_period_years   INTEGER,
    renewal_date            DATE,
    renewal_status          VARCHAR(30),
    insurance_amount_usd    DECIMAL(18,2)   NOT NULL,
    coverage_units          DECIMAL(18,2),
    coverage_unit_type      VARCHAR(30),
    annual_premium_usd      DECIMAL(18,2),
    total_premium_paid_usd  DECIMAL(18,2),
    coverage_type           VARCHAR(100),
    covered_risks           TEXT[],
    exclusions              TEXT[],
    policy_terms            TEXT,
    claim_procedure         TEXT,
    claim_deadline          VARCHAR(100),
    deductible_usd          DECIMAL(18,2),
    deductible_percentage   DECIMAL(6,2),
    coverage_limits         JSONB           DEFAULT '{}',
    buffer_pool_equivalent  DECIMAL(18,2),
    buffer_pool_percentage  DECIMAL(6,2),
    guarantee_mechanism     VARCHAR(100),
    guarantee_type          VARCHAR(50),
    guarantor_name          VARCHAR(255),
    guarantee_amount_usd    DECIMAL(18,2),
    guarantee_terms         TEXT,
    guarantee_period_years  INTEGER,
    claim_filed             BOOLEAN         DEFAULT FALSE,
    claim_date              DATE,
    claim_amount_usd        DECIMAL(18,2),
    claim_reason            TEXT,
    claim_status            VARCHAR(30),
    claim_approved          BOOLEAN         DEFAULT FALSE,
    claim_approved_date     DATE,
    claim_paid_amount_usd   DECIMAL(18,2),
    claim_payment_date      DATE,
    insurance_status        VARCHAR(30)     DEFAULT 'active',
    policy_compliance       BOOLEAN         DEFAULT TRUE,
    compliance_status_date  DATE,
    insurance_rating        VARCHAR(30),
    provider_financial_strength VARCHAR(30),
    regulatory_approval     BOOLEAN         DEFAULT FALSE,
    regulatory_body         VARCHAR(100),
    certification_standard  VARCHAR(100),
    certificate_number      VARCHAR(100),
    certificate_expiry_date DATE,
    policy_review_date      DATE,
    reviewed_by             VARCHAR(255),
    review_approval         BOOLEAN         DEFAULT FALSE,
    policy_notes            TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_insure_type CHECK (
        policy_type IN ('PERMANENCE_INSURANCE', 'REVERSAL_GUARANTEE', 'BUFFER_POOL', 'ESCROW_ACCOUNT',
                       'PERFORMANCE_BOND', 'LETTER_OF_CREDIT', 'INSURANCE_TRUST', 'OTHER')
    ),
    CONSTRAINT chk_pack024_insure_status CHECK (
        insurance_status IN ('ACTIVE', 'INACTIVE', 'EXPIRED', 'CANCELLED', 'LAPSED')
    ),
    CONSTRAINT chk_pack024_insure_claim_status CHECK (
        claim_status IN ('NOT_FILED', 'FILED', 'UNDER_REVIEW', 'APPROVED', 'DENIED', 'CLOSED')
    )
);

-- Indexes
CREATE INDEX idx_pack024_insure_org ON pack024_carbon_neutral.pack024_permanence_insurance(org_id);
CREATE INDEX idx_pack024_insure_tenant ON pack024_carbon_neutral.pack024_permanence_insurance(tenant_id);
CREATE INDEX idx_pack024_insure_date ON pack024_carbon_neutral.pack024_permanence_insurance(policy_date DESC);
CREATE INDEX idx_pack024_insure_type ON pack024_carbon_neutral.pack024_permanence_insurance(policy_type);
CREATE INDEX idx_pack024_insure_provider ON pack024_carbon_neutral.pack024_permanence_insurance(insurance_provider);
CREATE INDEX idx_pack024_insure_project_id ON pack024_carbon_neutral.pack024_permanence_insurance(project_id);
CREATE INDEX idx_pack024_insure_status ON pack024_carbon_neutral.pack024_permanence_insurance(insurance_status);
CREATE INDEX idx_pack024_insure_claim_filed ON pack024_carbon_neutral.pack024_permanence_insurance(claim_filed);
CREATE INDEX idx_pack024_insure_claim_status ON pack024_carbon_neutral.pack024_permanence_insurance(claim_status);
CREATE INDEX idx_pack024_insure_approved ON pack024_carbon_neutral.pack024_permanence_insurance(policy_compliance);
CREATE INDEX idx_pack024_insure_expiry ON pack024_carbon_neutral.pack024_permanence_insurance(policy_end_date);
CREATE INDEX idx_pack024_insure_renewal_date ON pack024_carbon_neutral.pack024_permanence_insurance(renewal_date);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_insure_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_permanence_insurance
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

COMMENT ON TABLE pack024_carbon_neutral.pack024_permanence_assessments IS
'Comprehensive permanence risk assessments for offset portfolio with scenario analysis and mitigation planning.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_permanence_risk_factors IS
'Individual risk factors contributing to permanence assessment with mitigation strategies and monitoring plans.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_permanence_monitoring IS
'Ongoing permanence monitoring activities and results with anomaly detection and reversal risk tracking.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_permanence_insurance IS
'Permanence insurance and guarantee coverage for carbon offset integrity with policy management and claim tracking.';
