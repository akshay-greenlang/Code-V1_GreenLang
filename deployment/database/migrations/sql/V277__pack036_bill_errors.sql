-- =============================================================================
-- V277: PACK-036 Utility Analysis Pack - Bill Errors & Audit Results
-- =============================================================================
-- Pack:         PACK-036 (Utility Analysis Pack)
-- Migration:    002 of 010
-- Date:         March 2026
--
-- Tables for bill validation errors detected during automated auditing
-- and facility-level audit result summaries. Supports error classification,
-- financial impact estimation, auto-correction tracking, and resolution
-- workflow.
--
-- Tables (2):
--   1. pack036_utility_analysis.gl_bill_errors
--   2. pack036_utility_analysis.gl_bill_audit_results
--
-- Previous: V276__pack036_utility_bills.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack036_utility_analysis.gl_bill_errors
-- =============================================================================
-- Individual errors detected on utility bills during automated validation.
-- Each error captures the expected vs. actual value, financial impact,
-- and whether auto-correction is possible.

CREATE TABLE pack036_utility_analysis.gl_bill_errors (
    error_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    bill_id                 UUID            NOT NULL REFERENCES pack036_utility_analysis.gl_utility_bills(bill_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    error_type              VARCHAR(50)     NOT NULL,
    severity                VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    field_name              VARCHAR(100),
    expected_value          TEXT,
    actual_value            TEXT,
    financial_impact_eur    NUMERIC(14,2)   DEFAULT 0,
    description             TEXT            NOT NULL,
    auto_correctable        BOOLEAN         NOT NULL DEFAULT false,
    resolved                BOOLEAN         NOT NULL DEFAULT false,
    resolved_at             TIMESTAMPTZ,
    resolved_by             UUID,
    resolution_notes        TEXT,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_be_error_type CHECK (
        error_type IN (
            'DUPLICATE_BILL', 'ESTIMATED_READ', 'USAGE_SPIKE', 'USAGE_DROP',
            'RATE_MISMATCH', 'TAX_ERROR', 'DEMAND_SPIKE', 'METER_ROLLOVER',
            'MISSING_DATA', 'DATE_OVERLAP', 'CALCULATION_ERROR',
            'LATE_CHARGE', 'WRONG_RATE', 'POWER_FACTOR_PENALTY',
            'CONTRACT_VIOLATION', 'UNIT_MISMATCH', 'OTHER'
        )
    ),
    CONSTRAINT chk_p036_be_severity CHECK (
        severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p036_be_resolved_at CHECK (
        (resolved = false AND resolved_at IS NULL) OR
        (resolved = true AND resolved_at IS NOT NULL)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_be_bill           ON pack036_utility_analysis.gl_bill_errors(bill_id);
CREATE INDEX idx_p036_be_tenant         ON pack036_utility_analysis.gl_bill_errors(tenant_id);
CREATE INDEX idx_p036_be_error_type     ON pack036_utility_analysis.gl_bill_errors(error_type);
CREATE INDEX idx_p036_be_severity       ON pack036_utility_analysis.gl_bill_errors(severity);
CREATE INDEX idx_p036_be_resolved       ON pack036_utility_analysis.gl_bill_errors(resolved);
CREATE INDEX idx_p036_be_created        ON pack036_utility_analysis.gl_bill_errors(created_at DESC);
CREATE INDEX idx_p036_be_financial      ON pack036_utility_analysis.gl_bill_errors(financial_impact_eur DESC);

-- Composite: unresolved errors by severity for operational dashboard
CREATE INDEX idx_p036_be_unresolved_sev ON pack036_utility_analysis.gl_bill_errors(severity, error_type)
    WHERE resolved = false;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_be_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_bill_errors
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack036_utility_analysis.gl_bill_audit_results
-- =============================================================================
-- Facility-level audit result summaries aggregating errors found,
-- financial impact, and overall audit pass/fail status.

CREATE TABLE pack036_utility_analysis.gl_bill_audit_results (
    audit_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    audit_date              DATE            NOT NULL DEFAULT CURRENT_DATE,
    audit_type              VARCHAR(30)     NOT NULL DEFAULT 'AUTOMATED',
    period_start            DATE,
    period_end              DATE,
    bills_audited           INTEGER         NOT NULL DEFAULT 0,
    errors_found            INTEGER         NOT NULL DEFAULT 0,
    errors_critical         INTEGER         DEFAULT 0,
    errors_high             INTEGER         DEFAULT 0,
    errors_medium           INTEGER         DEFAULT 0,
    errors_low              INTEGER         DEFAULT 0,
    auto_corrected          INTEGER         DEFAULT 0,
    total_financial_impact_eur NUMERIC(14,2) DEFAULT 0,
    recovered_amount_eur    NUMERIC(14,2)   DEFAULT 0,
    status                  VARCHAR(30)     NOT NULL DEFAULT 'COMPLETED',
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_bar_audit_type CHECK (
        audit_type IN ('AUTOMATED', 'MANUAL', 'PERIODIC', 'AD_HOC')
    ),
    CONSTRAINT chk_p036_bar_status CHECK (
        status IN ('IN_PROGRESS', 'COMPLETED', 'APPROVED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p036_bar_bills CHECK (
        bills_audited >= 0
    ),
    CONSTRAINT chk_p036_bar_errors CHECK (
        errors_found >= 0
    ),
    CONSTRAINT chk_p036_bar_period CHECK (
        period_start IS NULL OR period_end IS NULL OR period_end >= period_start
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_bar_tenant        ON pack036_utility_analysis.gl_bill_audit_results(tenant_id);
CREATE INDEX idx_p036_bar_facility      ON pack036_utility_analysis.gl_bill_audit_results(facility_id);
CREATE INDEX idx_p036_bar_audit_date    ON pack036_utility_analysis.gl_bill_audit_results(audit_date DESC);
CREATE INDEX idx_p036_bar_status        ON pack036_utility_analysis.gl_bill_audit_results(status);
CREATE INDEX idx_p036_bar_created       ON pack036_utility_analysis.gl_bill_audit_results(created_at DESC);

-- Composite: facility + date for time-series audit history
CREATE INDEX idx_p036_bar_fac_date      ON pack036_utility_analysis.gl_bill_audit_results(facility_id, audit_date DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_bar_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_bill_audit_results
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack036_utility_analysis.gl_bill_errors ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_bill_audit_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY p036_be_tenant_isolation
    ON pack036_utility_analysis.gl_bill_errors
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p036_be_service_bypass
    ON pack036_utility_analysis.gl_bill_errors
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_bar_tenant_isolation
    ON pack036_utility_analysis.gl_bill_audit_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p036_bar_service_bypass
    ON pack036_utility_analysis.gl_bill_audit_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_bill_errors TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_bill_audit_results TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack036_utility_analysis.gl_bill_errors IS
    'Individual errors detected on utility bills during automated validation with expected vs actual values and financial impact.';

COMMENT ON TABLE pack036_utility_analysis.gl_bill_audit_results IS
    'Facility-level audit result summaries aggregating errors found, financial impact, and recovery amounts.';

COMMENT ON COLUMN pack036_utility_analysis.gl_bill_errors.error_id IS
    'Unique identifier for the bill error.';
COMMENT ON COLUMN pack036_utility_analysis.gl_bill_errors.bill_id IS
    'Reference to the utility bill containing this error.';
COMMENT ON COLUMN pack036_utility_analysis.gl_bill_errors.tenant_id IS
    'Multi-tenant isolation key.';
COMMENT ON COLUMN pack036_utility_analysis.gl_bill_errors.error_type IS
    'Error classification: DUPLICATE_BILL, ESTIMATED_READ, USAGE_SPIKE, RATE_MISMATCH, etc.';
COMMENT ON COLUMN pack036_utility_analysis.gl_bill_errors.severity IS
    'Error severity: LOW, MEDIUM, HIGH, CRITICAL.';
COMMENT ON COLUMN pack036_utility_analysis.gl_bill_errors.field_name IS
    'Name of the bill field where the error was detected.';
COMMENT ON COLUMN pack036_utility_analysis.gl_bill_errors.expected_value IS
    'Expected value based on validation rules or historical patterns.';
COMMENT ON COLUMN pack036_utility_analysis.gl_bill_errors.actual_value IS
    'Actual value found on the bill.';
COMMENT ON COLUMN pack036_utility_analysis.gl_bill_errors.financial_impact_eur IS
    'Estimated financial impact of this error in EUR.';
COMMENT ON COLUMN pack036_utility_analysis.gl_bill_errors.auto_correctable IS
    'Whether this error can be automatically corrected without human intervention.';
COMMENT ON COLUMN pack036_utility_analysis.gl_bill_errors.resolved IS
    'Whether this error has been resolved (corrected or accepted).';
COMMENT ON COLUMN pack036_utility_analysis.gl_bill_errors.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack036_utility_analysis.gl_bill_audit_results.audit_id IS
    'Unique identifier for the audit result.';
COMMENT ON COLUMN pack036_utility_analysis.gl_bill_audit_results.facility_id IS
    'Reference to the facility audited.';
COMMENT ON COLUMN pack036_utility_analysis.gl_bill_audit_results.bills_audited IS
    'Total number of bills included in this audit run.';
COMMENT ON COLUMN pack036_utility_analysis.gl_bill_audit_results.total_financial_impact_eur IS
    'Total financial impact of all errors found in EUR.';
COMMENT ON COLUMN pack036_utility_analysis.gl_bill_audit_results.recovered_amount_eur IS
    'Amount successfully recovered through dispute resolution.';
COMMENT ON COLUMN pack036_utility_analysis.gl_bill_audit_results.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
