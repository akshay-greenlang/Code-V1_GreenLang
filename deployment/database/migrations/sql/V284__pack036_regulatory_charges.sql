-- =============================================================================
-- V284: PACK-036 Utility Analysis Pack - Regulatory Charges & Optimizations
-- =============================================================================
-- Pack:         PACK-036 (Utility Analysis Pack)
-- Migration:    009 of 010
-- Date:         March 2026
--
-- Tables for tracking non-commodity regulatory charges (network charges,
-- levies, taxes, surcharges), optimization analyses to reduce these
-- charges, and exemption/reduction eligibility assessments.
--
-- Tables (3):
--   1. pack036_utility_analysis.gl_regulatory_charges
--   2. pack036_utility_analysis.gl_charge_optimizations
--   3. pack036_utility_analysis.gl_exemption_assessments
--
-- Previous: V283__pack036_benchmarks.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack036_utility_analysis.gl_regulatory_charges
-- =============================================================================
-- Non-commodity charges on utility bills including network charges,
-- renewable levies, capacity charges, climate levies, and other
-- regulated components. Tracks methodology, jurisdiction, and
-- optimization potential.

CREATE TABLE pack036_utility_analysis.gl_regulatory_charges (
    charge_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    charge_type             VARCHAR(50)     NOT NULL,
    name                    VARCHAR(255)    NOT NULL,
    description             TEXT,
    methodology             VARCHAR(100),
    rate                    NUMERIC(14,8),
    unit                    VARCHAR(30),
    volume_basis            NUMERIC(16,4),
    annual_amount_eur       NUMERIC(14,2)   NOT NULL,
    share_of_bill_pct       NUMERIC(6,2),
    jurisdiction            VARCHAR(100)    NOT NULL,
    regulatory_body         VARCHAR(255),
    legislation_reference   VARCHAR(255),
    optimizable             BOOLEAN         NOT NULL DEFAULT false,
    optimization_potential_eur NUMERIC(14,2),
    commodity               VARCHAR(30)     NOT NULL DEFAULT 'ELECTRICITY',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_rgc_charge_type CHECK (
        charge_type IN (
            'NETWORK_USE', 'TRANSMISSION', 'DISTRIBUTION', 'SYSTEM_OPERATOR',
            'CAPACITY', 'RENEWABLE_LEVY', 'CHP_LEVY', 'CLIMATE_LEVY',
            'ENERGY_TAX', 'VAT', 'CONCESSION_FEE', 'METER_CHARGE',
            'REACTIVE_POWER', 'INTERRUPTIBILITY', 'BALANCING',
            'STRANDED_COST', 'UNIVERSAL_SERVICE', 'OTHER'
        )
    ),
    CONSTRAINT chk_p036_rgc_period CHECK (
        period_end >= period_start
    ),
    CONSTRAINT chk_p036_rgc_amount CHECK (
        annual_amount_eur >= 0
    ),
    CONSTRAINT chk_p036_rgc_share CHECK (
        share_of_bill_pct IS NULL OR (share_of_bill_pct >= 0 AND share_of_bill_pct <= 100)
    ),
    CONSTRAINT chk_p036_rgc_commodity CHECK (
        commodity IN (
            'ELECTRICITY', 'NATURAL_GAS', 'WATER', 'ALL'
        )
    ),
    CONSTRAINT chk_p036_rgc_opt_potential CHECK (
        optimization_potential_eur IS NULL OR optimization_potential_eur >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_rgc_tenant        ON pack036_utility_analysis.gl_regulatory_charges(tenant_id);
CREATE INDEX idx_p036_rgc_facility      ON pack036_utility_analysis.gl_regulatory_charges(facility_id);
CREATE INDEX idx_p036_rgc_period        ON pack036_utility_analysis.gl_regulatory_charges(period_start DESC);
CREATE INDEX idx_p036_rgc_charge_type   ON pack036_utility_analysis.gl_regulatory_charges(charge_type);
CREATE INDEX idx_p036_rgc_jurisdiction  ON pack036_utility_analysis.gl_regulatory_charges(jurisdiction);
CREATE INDEX idx_p036_rgc_commodity     ON pack036_utility_analysis.gl_regulatory_charges(commodity);
CREATE INDEX idx_p036_rgc_optimizable   ON pack036_utility_analysis.gl_regulatory_charges(optimizable);
CREATE INDEX idx_p036_rgc_amount        ON pack036_utility_analysis.gl_regulatory_charges(annual_amount_eur DESC);
CREATE INDEX idx_p036_rgc_created       ON pack036_utility_analysis.gl_regulatory_charges(created_at DESC);
CREATE INDEX idx_p036_rgc_metadata      ON pack036_utility_analysis.gl_regulatory_charges USING GIN(metadata);

-- Composite: facility + period for time-series charge analysis
CREATE INDEX idx_p036_rgc_fac_period    ON pack036_utility_analysis.gl_regulatory_charges(facility_id, period_start DESC);

-- Partial: optimizable charges for cost reduction dashboard
CREATE INDEX idx_p036_rgc_opt_charges   ON pack036_utility_analysis.gl_regulatory_charges(facility_id, charge_type, optimization_potential_eur DESC)
    WHERE optimizable = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_rgc_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_regulatory_charges
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack036_utility_analysis.gl_charge_optimizations
-- =============================================================================
-- Facility-level non-commodity cost optimization analysis results.
-- Identifies total non-commodity costs, achievable reductions, and
-- specific action plans with implementation steps.

CREATE TABLE pack036_utility_analysis.gl_charge_optimizations (
    optimization_id         UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    analysis_date           DATE            NOT NULL DEFAULT CURRENT_DATE,
    analysis_period_start   DATE,
    analysis_period_end     DATE,
    commodity               VARCHAR(30)     NOT NULL DEFAULT 'ELECTRICITY',
    current_non_commodity_eur NUMERIC(14,2) NOT NULL,
    non_commodity_share_pct NUMERIC(6,2),
    optimized_eur           NUMERIC(14,2)   NOT NULL,
    total_savings_eur       NUMERIC(14,2)   NOT NULL DEFAULT 0,
    savings_pct             NUMERIC(8,4),
    actions                 JSONB           NOT NULL DEFAULT '[]',
    implementation_timeline VARCHAR(30),
    risk_level              VARCHAR(20)     DEFAULT 'LOW',
    status                  VARCHAR(30)     NOT NULL DEFAULT 'COMPLETED',
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_co_commodity CHECK (
        commodity IN (
            'ELECTRICITY', 'NATURAL_GAS', 'WATER', 'ALL'
        )
    ),
    CONSTRAINT chk_p036_co_current CHECK (
        current_non_commodity_eur >= 0
    ),
    CONSTRAINT chk_p036_co_optimized CHECK (
        optimized_eur >= 0
    ),
    CONSTRAINT chk_p036_co_savings CHECK (
        total_savings_eur >= 0
    ),
    CONSTRAINT chk_p036_co_share CHECK (
        non_commodity_share_pct IS NULL OR (non_commodity_share_pct >= 0 AND non_commodity_share_pct <= 100)
    ),
    CONSTRAINT chk_p036_co_timeline CHECK (
        implementation_timeline IS NULL OR implementation_timeline IN (
            'IMMEDIATE', '1_MONTH', '3_MONTHS', '6_MONTHS', '12_MONTHS', 'LONG_TERM'
        )
    ),
    CONSTRAINT chk_p036_co_risk CHECK (
        risk_level IN ('LOW', 'MEDIUM', 'HIGH')
    ),
    CONSTRAINT chk_p036_co_status CHECK (
        status IN ('IN_PROGRESS', 'COMPLETED', 'APPROVED', 'IMPLEMENTED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p036_co_period CHECK (
        analysis_period_start IS NULL OR analysis_period_end IS NULL
        OR analysis_period_end >= analysis_period_start
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_co_tenant         ON pack036_utility_analysis.gl_charge_optimizations(tenant_id);
CREATE INDEX idx_p036_co_facility       ON pack036_utility_analysis.gl_charge_optimizations(facility_id);
CREATE INDEX idx_p036_co_date           ON pack036_utility_analysis.gl_charge_optimizations(analysis_date DESC);
CREATE INDEX idx_p036_co_commodity      ON pack036_utility_analysis.gl_charge_optimizations(commodity);
CREATE INDEX idx_p036_co_savings        ON pack036_utility_analysis.gl_charge_optimizations(total_savings_eur DESC);
CREATE INDEX idx_p036_co_status         ON pack036_utility_analysis.gl_charge_optimizations(status);
CREATE INDEX idx_p036_co_created        ON pack036_utility_analysis.gl_charge_optimizations(created_at DESC);
CREATE INDEX idx_p036_co_actions        ON pack036_utility_analysis.gl_charge_optimizations USING GIN(actions);

-- Composite: facility + date for historical analysis lookup
CREATE INDEX idx_p036_co_fac_date       ON pack036_utility_analysis.gl_charge_optimizations(facility_id, analysis_date DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_co_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_charge_optimizations
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack036_utility_analysis.gl_exemption_assessments
-- =============================================================================
-- Assessments of eligibility for regulatory charge exemptions or
-- reductions (e.g., EEG surcharge reduction in Germany, CCL reduction
-- in UK, renewable energy levy exemptions). Tracks eligibility criteria,
-- current charges, potential savings, and application deadlines.

CREATE TABLE pack036_utility_analysis.gl_exemption_assessments (
    exemption_id            UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    exemption_type          VARCHAR(50)     NOT NULL,
    exemption_name          VARCHAR(255)    NOT NULL,
    jurisdiction            VARCHAR(100)    NOT NULL,
    regulatory_basis        VARCHAR(255),
    eligible                BOOLEAN         NOT NULL DEFAULT false,
    eligibility_criteria    JSONB           DEFAULT '{}',
    eligibility_notes       TEXT,
    current_charge_eur      NUMERIC(14,2)   NOT NULL DEFAULT 0,
    reduced_charge_eur      NUMERIC(14,2),
    savings_eur             NUMERIC(14,2)   NOT NULL DEFAULT 0,
    application_deadline    DATE,
    application_status      VARCHAR(30)     DEFAULT 'NOT_STARTED',
    assessment_date         DATE            NOT NULL DEFAULT CURRENT_DATE,
    valid_from              DATE,
    valid_to                DATE,
    required_documents      TEXT[],
    assessor                TEXT,
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_ea_exemption_type CHECK (
        exemption_type IN (
            'RENEWABLE_LEVY', 'ENERGY_TAX', 'NETWORK_CHARGE',
            'CLIMATE_LEVY', 'CHP_LEVY', 'CONCESSION_FEE',
            'CAPACITY_CHARGE', 'INTERRUPTIBILITY', 'SELF_GENERATION',
            'COGENERATION', 'ENERGY_INTENSIVE', 'TRADE_EXPOSED',
            'OTHER'
        )
    ),
    CONSTRAINT chk_p036_ea_current CHECK (
        current_charge_eur >= 0
    ),
    CONSTRAINT chk_p036_ea_reduced CHECK (
        reduced_charge_eur IS NULL OR reduced_charge_eur >= 0
    ),
    CONSTRAINT chk_p036_ea_savings CHECK (
        savings_eur >= 0
    ),
    CONSTRAINT chk_p036_ea_app_status CHECK (
        application_status IN (
            'NOT_STARTED', 'PREPARING', 'SUBMITTED', 'UNDER_REVIEW',
            'APPROVED', 'REJECTED', 'EXPIRED', 'RENEWED'
        )
    ),
    CONSTRAINT chk_p036_ea_validity CHECK (
        valid_from IS NULL OR valid_to IS NULL OR valid_to >= valid_from
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_ea_tenant         ON pack036_utility_analysis.gl_exemption_assessments(tenant_id);
CREATE INDEX idx_p036_ea_facility       ON pack036_utility_analysis.gl_exemption_assessments(facility_id);
CREATE INDEX idx_p036_ea_type           ON pack036_utility_analysis.gl_exemption_assessments(exemption_type);
CREATE INDEX idx_p036_ea_jurisdiction   ON pack036_utility_analysis.gl_exemption_assessments(jurisdiction);
CREATE INDEX idx_p036_ea_eligible       ON pack036_utility_analysis.gl_exemption_assessments(eligible);
CREATE INDEX idx_p036_ea_savings        ON pack036_utility_analysis.gl_exemption_assessments(savings_eur DESC);
CREATE INDEX idx_p036_ea_deadline       ON pack036_utility_analysis.gl_exemption_assessments(application_deadline);
CREATE INDEX idx_p036_ea_app_status     ON pack036_utility_analysis.gl_exemption_assessments(application_status);
CREATE INDEX idx_p036_ea_created        ON pack036_utility_analysis.gl_exemption_assessments(created_at DESC);

-- Partial: eligible exemptions with upcoming deadlines for alerts
CREATE INDEX idx_p036_ea_upcoming       ON pack036_utility_analysis.gl_exemption_assessments(application_deadline)
    WHERE eligible = true AND application_status IN ('NOT_STARTED', 'PREPARING');

-- Composite: facility + exemption type for assessment lookup
CREATE INDEX idx_p036_ea_fac_type       ON pack036_utility_analysis.gl_exemption_assessments(facility_id, exemption_type);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_ea_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_exemption_assessments
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack036_utility_analysis.gl_regulatory_charges ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_charge_optimizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_exemption_assessments ENABLE ROW LEVEL SECURITY;

CREATE POLICY p036_rgc_tenant_isolation
    ON pack036_utility_analysis.gl_regulatory_charges
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_rgc_service_bypass
    ON pack036_utility_analysis.gl_regulatory_charges
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_co_tenant_isolation
    ON pack036_utility_analysis.gl_charge_optimizations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_co_service_bypass
    ON pack036_utility_analysis.gl_charge_optimizations
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_ea_tenant_isolation
    ON pack036_utility_analysis.gl_exemption_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_ea_service_bypass
    ON pack036_utility_analysis.gl_exemption_assessments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_regulatory_charges TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_charge_optimizations TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_exemption_assessments TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack036_utility_analysis.gl_regulatory_charges IS
    'Non-commodity regulatory charges on utility bills including network charges, levies, taxes, and surcharges with optimization potential.';

COMMENT ON TABLE pack036_utility_analysis.gl_charge_optimizations IS
    'Facility-level non-commodity cost optimization analysis with current vs optimized costs and action plans.';

COMMENT ON TABLE pack036_utility_analysis.gl_exemption_assessments IS
    'Assessments of eligibility for regulatory charge exemptions or reductions with savings and application tracking.';

COMMENT ON COLUMN pack036_utility_analysis.gl_regulatory_charges.charge_id IS
    'Unique identifier for the regulatory charge record.';
COMMENT ON COLUMN pack036_utility_analysis.gl_regulatory_charges.charge_type IS
    'Charge type: NETWORK_USE, TRANSMISSION, DISTRIBUTION, RENEWABLE_LEVY, CLIMATE_LEVY, ENERGY_TAX, etc.';
COMMENT ON COLUMN pack036_utility_analysis.gl_regulatory_charges.methodology IS
    'Methodology used to calculate the charge (e.g., volumetric, capacity-based, peak-based).';
COMMENT ON COLUMN pack036_utility_analysis.gl_regulatory_charges.share_of_bill_pct IS
    'Percentage share of total utility bill attributable to this charge.';
COMMENT ON COLUMN pack036_utility_analysis.gl_regulatory_charges.optimizable IS
    'Whether this charge can be optimized through operational or contractual changes.';
COMMENT ON COLUMN pack036_utility_analysis.gl_regulatory_charges.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack036_utility_analysis.gl_charge_optimizations.optimization_id IS
    'Unique identifier for the optimization analysis.';
COMMENT ON COLUMN pack036_utility_analysis.gl_charge_optimizations.current_non_commodity_eur IS
    'Current total non-commodity charges in EUR.';
COMMENT ON COLUMN pack036_utility_analysis.gl_charge_optimizations.actions IS
    'JSON array of optimization actions with type, description, savings, and implementation steps.';
COMMENT ON COLUMN pack036_utility_analysis.gl_charge_optimizations.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack036_utility_analysis.gl_exemption_assessments.exemption_id IS
    'Unique identifier for the exemption assessment.';
COMMENT ON COLUMN pack036_utility_analysis.gl_exemption_assessments.exemption_type IS
    'Exemption type: RENEWABLE_LEVY, ENERGY_TAX, NETWORK_CHARGE, CLIMATE_LEVY, ENERGY_INTENSIVE, etc.';
COMMENT ON COLUMN pack036_utility_analysis.gl_exemption_assessments.eligible IS
    'Whether the facility is eligible for this exemption based on assessment criteria.';
COMMENT ON COLUMN pack036_utility_analysis.gl_exemption_assessments.current_charge_eur IS
    'Current annual charge amount before exemption in EUR.';
COMMENT ON COLUMN pack036_utility_analysis.gl_exemption_assessments.savings_eur IS
    'Estimated annual savings if exemption is applied.';
COMMENT ON COLUMN pack036_utility_analysis.gl_exemption_assessments.application_deadline IS
    'Deadline for submitting the exemption application.';
COMMENT ON COLUMN pack036_utility_analysis.gl_exemption_assessments.application_status IS
    'Application status: NOT_STARTED, PREPARING, SUBMITTED, UNDER_REVIEW, APPROVED, REJECTED, EXPIRED, RENEWED.';
COMMENT ON COLUMN pack036_utility_analysis.gl_exemption_assessments.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
