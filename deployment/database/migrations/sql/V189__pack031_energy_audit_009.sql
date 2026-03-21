-- =============================================================================
-- V189: PACK-031 Industrial Energy Audit - Benchmarks & Compliance
-- =============================================================================
-- Pack:         PACK-031 (Industrial Energy Audit Pack)
-- Migration:    009 of 010
-- Date:         March 2026
--
-- Energy benchmarking, BAT/AEL (Best Available Technique / Associated
-- Emission Levels) comparisons from BREF documents, EU Energy Efficiency
-- Directive (EED) compliance tracking, and ISO 50001 certification records.
--
-- Tables (4):
--   1. pack031_energy_audit.energy_benchmarks
--   2. pack031_energy_audit.bat_ael_comparisons
--   3. pack031_energy_audit.eed_compliance
--   4. pack031_energy_audit.iso_50001_records
--
-- Previous: V188__pack031_energy_audit_008.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack031_energy_audit.energy_benchmarks
-- =============================================================================
-- Energy benchmarking records comparing facility SEC (Specific Energy
-- Consumption) against sector averages and best practice values.

CREATE TABLE pack031_energy_audit.energy_benchmarks (
    benchmark_id            UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    period                  DATE            NOT NULL,
    product_type            VARCHAR(100),
    production_volume       NUMERIC(14,4),
    production_unit         VARCHAR(50),
    sec_value               NUMERIC(14,6)   NOT NULL,
    sec_unit                VARCHAR(50)     NOT NULL,
    sector_average          NUMERIC(14,6),
    sector_median           NUMERIC(14,6),
    best_practice           NUMERIC(14,6),
    top_quartile            NUMERIC(14,6),
    percentile_rank         INTEGER,
    energy_rating           VARCHAR(10),
    gap_to_best_pct         NUMERIC(8,4),
    gap_to_average_pct      NUMERIC(8,4),
    benchmark_source        VARCHAR(255),
    benchmark_year          INTEGER,
    data_quality            VARCHAR(20),
    normalized              BOOLEAN         DEFAULT FALSE,
    normalization_factors   JSONB           DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_bench_sec CHECK (sec_value >= 0),
    CONSTRAINT chk_p031_bench_avg CHECK (
        sector_average IS NULL OR sector_average >= 0
    ),
    CONSTRAINT chk_p031_bench_best CHECK (
        best_practice IS NULL OR best_practice >= 0
    ),
    CONSTRAINT chk_p031_bench_percentile CHECK (
        percentile_rank IS NULL OR (percentile_rank >= 0 AND percentile_rank <= 100)
    ),
    CONSTRAINT chk_p031_bench_rating CHECK (
        energy_rating IS NULL OR energy_rating IN ('A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G')
    ),
    CONSTRAINT chk_p031_bench_quality CHECK (
        data_quality IS NULL OR data_quality IN ('high', 'medium', 'low', 'estimated')
    )
);

-- Indexes
CREATE INDEX idx_p031_bench_facility   ON pack031_energy_audit.energy_benchmarks(facility_id);
CREATE INDEX idx_p031_bench_tenant     ON pack031_energy_audit.energy_benchmarks(tenant_id);
CREATE INDEX idx_p031_bench_period     ON pack031_energy_audit.energy_benchmarks(period);
CREATE INDEX idx_p031_bench_sec        ON pack031_energy_audit.energy_benchmarks(sec_value);
CREATE INDEX idx_p031_bench_rating     ON pack031_energy_audit.energy_benchmarks(energy_rating);
CREATE INDEX idx_p031_bench_percentile ON pack031_energy_audit.energy_benchmarks(percentile_rank);
CREATE INDEX idx_p031_bench_gap        ON pack031_energy_audit.energy_benchmarks(gap_to_best_pct);

-- =============================================================================
-- Table 2: pack031_energy_audit.bat_ael_comparisons
-- =============================================================================
-- BAT-AEL (Best Available Technique - Associated Emission/Energy Levels)
-- comparisons from EU BREF (BAT Reference) documents for IED compliance.

CREATE TABLE pack031_energy_audit.bat_ael_comparisons (
    comparison_id           UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    bref_document           VARCHAR(255)    NOT NULL,
    bref_version            VARCHAR(50),
    bref_publication_date   DATE,
    process                 VARCHAR(255)    NOT NULL,
    parameter               VARCHAR(100),
    bat_ael_min             NUMERIC(14,6),
    bat_ael_max             NUMERIC(14,6),
    bat_ael_unit            VARCHAR(50),
    facility_value          NUMERIC(14,6)   NOT NULL,
    facility_unit           VARCHAR(50),
    compliance_status       VARCHAR(30)     NOT NULL DEFAULT 'unknown',
    gap_pct                 NUMERIC(8,4),
    gap_to_upper_pct        NUMERIC(8,4),
    derogation_applied      BOOLEAN         DEFAULT FALSE,
    derogation_reason       TEXT,
    improvement_actions     TEXT,
    assessment_date         DATE,
    next_review_date        DATE,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_bat_ael_range CHECK (
        bat_ael_min IS NULL OR bat_ael_max IS NULL OR bat_ael_min <= bat_ael_max
    ),
    CONSTRAINT chk_p031_bat_compliance CHECK (
        compliance_status IN ('compliant', 'non_compliant', 'marginal', 'derogation', 'unknown')
    )
);

-- Indexes
CREATE INDEX idx_p031_bat_facility     ON pack031_energy_audit.bat_ael_comparisons(facility_id);
CREATE INDEX idx_p031_bat_tenant       ON pack031_energy_audit.bat_ael_comparisons(tenant_id);
CREATE INDEX idx_p031_bat_bref         ON pack031_energy_audit.bat_ael_comparisons(bref_document);
CREATE INDEX idx_p031_bat_process      ON pack031_energy_audit.bat_ael_comparisons(process);
CREATE INDEX idx_p031_bat_compliance   ON pack031_energy_audit.bat_ael_comparisons(compliance_status);
CREATE INDEX idx_p031_bat_gap          ON pack031_energy_audit.bat_ael_comparisons(gap_pct);

-- =============================================================================
-- Table 3: pack031_energy_audit.eed_compliance
-- =============================================================================
-- EU Energy Efficiency Directive (EED) Article 8 compliance tracking
-- for mandatory energy audit obligations and exemptions.

CREATE TABLE pack031_energy_audit.eed_compliance (
    compliance_id           UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    obligation_applies      BOOLEAN         NOT NULL DEFAULT TRUE,
    obligation_basis        VARCHAR(100),
    enterprise_type         VARCHAR(30),
    last_audit_date         DATE,
    last_audit_id           UUID            REFERENCES pack031_energy_audit.energy_audits(audit_id) ON DELETE SET NULL,
    next_audit_due          DATE,
    audit_frequency_years   INTEGER         DEFAULT 4,
    iso50001_exempt         BOOLEAN         DEFAULT FALSE,
    emas_exempt             BOOLEAN         DEFAULT FALSE,
    en16247_compliant       BOOLEAN         DEFAULT FALSE,
    compliance_status       VARCHAR(30)     NOT NULL DEFAULT 'unknown',
    national_authority      TEXT,
    national_legislation    VARCHAR(255),
    registration_number     VARCHAR(100),
    penalties_risk          BOOLEAN         DEFAULT FALSE,
    days_until_due          INTEGER,
    last_notification_date  DATE,
    responsible_person      VARCHAR(255),
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_eed_enterprise CHECK (
        enterprise_type IS NULL OR enterprise_type IN ('LARGE', 'SME', 'MICRO', 'PUBLIC_BODY')
    ),
    CONSTRAINT chk_p031_eed_compliance CHECK (
        compliance_status IN ('compliant', 'non_compliant', 'exempt', 'overdue', 'pending', 'unknown')
    ),
    CONSTRAINT chk_p031_eed_frequency CHECK (
        audit_frequency_years IS NULL OR (audit_frequency_years >= 1 AND audit_frequency_years <= 10)
    )
);

-- Indexes
CREATE INDEX idx_p031_eed_facility     ON pack031_energy_audit.eed_compliance(facility_id);
CREATE INDEX idx_p031_eed_tenant       ON pack031_energy_audit.eed_compliance(tenant_id);
CREATE INDEX idx_p031_eed_obligation   ON pack031_energy_audit.eed_compliance(obligation_applies);
CREATE INDEX idx_p031_eed_compliance   ON pack031_energy_audit.eed_compliance(compliance_status);
CREATE INDEX idx_p031_eed_next_due     ON pack031_energy_audit.eed_compliance(next_audit_due);
CREATE INDEX idx_p031_eed_iso_exempt   ON pack031_energy_audit.eed_compliance(iso50001_exempt);
CREATE INDEX idx_p031_eed_last_audit   ON pack031_energy_audit.eed_compliance(last_audit_date);

-- Trigger
CREATE TRIGGER trg_p031_eed_updated
    BEFORE UPDATE ON pack031_energy_audit.eed_compliance
    FOR EACH ROW EXECUTE FUNCTION pack031_energy_audit.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack031_energy_audit.iso_50001_records
-- =============================================================================
-- ISO 50001 Energy Management System certification and internal
-- audit records with EnMS maturity level tracking.

CREATE TABLE pack031_energy_audit.iso_50001_records (
    record_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    certification_status    VARCHAR(30)     NOT NULL DEFAULT 'not_certified',
    certification_body      VARCHAR(255),
    certificate_number      VARCHAR(100),
    certification_date      DATE,
    expiry_date             DATE,
    last_surveillance_audit DATE,
    next_surveillance_due   DATE,
    last_recertification    DATE,
    last_internal_audit     DATE,
    internal_audit_findings INTEGER         DEFAULT 0,
    last_management_review  DATE,
    enms_maturity_level     VARCHAR(30),
    scope_description       TEXT,
    enpi_count              INTEGER,
    continual_improvement_pct NUMERIC(5,2),
    energy_policy_reviewed  BOOLEAN         DEFAULT FALSE,
    legal_register_current  BOOLEAN         DEFAULT FALSE,
    objectives_set          BOOLEAN         DEFAULT FALSE,
    responsible_person      VARCHAR(255),
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_iso_cert_status CHECK (
        certification_status IN ('not_certified', 'in_progress', 'certified',
                                  'suspended', 'withdrawn', 'expired')
    ),
    CONSTRAINT chk_p031_iso_maturity CHECK (
        enms_maturity_level IS NULL OR enms_maturity_level IN (
            'INITIAL', 'DEVELOPING', 'DEFINED', 'MANAGED', 'OPTIMIZING'
        )
    ),
    CONSTRAINT chk_p031_iso_findings CHECK (
        internal_audit_findings IS NULL OR internal_audit_findings >= 0
    ),
    CONSTRAINT chk_p031_iso_improvement CHECK (
        continual_improvement_pct IS NULL OR (continual_improvement_pct >= -100 AND continual_improvement_pct <= 100)
    )
);

-- Indexes
CREATE INDEX idx_p031_iso_facility     ON pack031_energy_audit.iso_50001_records(facility_id);
CREATE INDEX idx_p031_iso_tenant       ON pack031_energy_audit.iso_50001_records(tenant_id);
CREATE INDEX idx_p031_iso_cert_status  ON pack031_energy_audit.iso_50001_records(certification_status);
CREATE INDEX idx_p031_iso_expiry       ON pack031_energy_audit.iso_50001_records(expiry_date);
CREATE INDEX idx_p031_iso_maturity     ON pack031_energy_audit.iso_50001_records(enms_maturity_level);
CREATE INDEX idx_p031_iso_next_surv    ON pack031_energy_audit.iso_50001_records(next_surveillance_due);

-- Trigger
CREATE TRIGGER trg_p031_iso_updated
    BEFORE UPDATE ON pack031_energy_audit.iso_50001_records
    FOR EACH ROW EXECUTE FUNCTION pack031_energy_audit.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack031_energy_audit.energy_benchmarks ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.bat_ael_comparisons ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.eed_compliance ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.iso_50001_records ENABLE ROW LEVEL SECURITY;

CREATE POLICY p031_bench_tenant_isolation ON pack031_energy_audit.energy_benchmarks
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_bench_service_bypass ON pack031_energy_audit.energy_benchmarks
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_bat_tenant_isolation ON pack031_energy_audit.bat_ael_comparisons
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_bat_service_bypass ON pack031_energy_audit.bat_ael_comparisons
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_eed_tenant_isolation ON pack031_energy_audit.eed_compliance
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_eed_service_bypass ON pack031_energy_audit.eed_compliance
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_iso_tenant_isolation ON pack031_energy_audit.iso_50001_records
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_iso_service_bypass ON pack031_energy_audit.iso_50001_records
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.energy_benchmarks TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.bat_ael_comparisons TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.eed_compliance TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.iso_50001_records TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack031_energy_audit.energy_benchmarks IS
    'Energy benchmarking records comparing facility SEC against sector averages and best practice values.';
COMMENT ON TABLE pack031_energy_audit.bat_ael_comparisons IS
    'BAT-AEL comparisons from EU BREF documents for Industrial Emissions Directive compliance.';
COMMENT ON TABLE pack031_energy_audit.eed_compliance IS
    'EU Energy Efficiency Directive Article 8 compliance tracking for mandatory audit obligations.';
COMMENT ON TABLE pack031_energy_audit.iso_50001_records IS
    'ISO 50001 EnMS certification, surveillance audits, internal audits, and maturity tracking.';

COMMENT ON COLUMN pack031_energy_audit.energy_benchmarks.sec_value IS
    'Specific Energy Consumption - primary benchmarking metric (e.g., kWh/tonne, kWh/m2).';
COMMENT ON COLUMN pack031_energy_audit.energy_benchmarks.energy_rating IS
    'Energy rating label from A+ (best) to G (worst).';
COMMENT ON COLUMN pack031_energy_audit.bat_ael_comparisons.bref_document IS
    'EU BREF document reference (e.g., FDM BREF, LCP BREF, ENE BREF).';
COMMENT ON COLUMN pack031_energy_audit.eed_compliance.audit_frequency_years IS
    'Required audit frequency in years (typically 4 years under EED Article 8).';
COMMENT ON COLUMN pack031_energy_audit.iso_50001_records.enms_maturity_level IS
    'Energy Management System maturity: INITIAL, DEVELOPING, DEFINED, MANAGED, OPTIMIZING.';
