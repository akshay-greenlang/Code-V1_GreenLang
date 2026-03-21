-- =============================================================================
-- V272: PACK-035 Energy Benchmark Pack - Performance Rating Tables
-- =============================================================================
-- Pack:         PACK-035 (Energy Benchmark Pack)
-- Migration:    007 of 010
-- Date:         March 2026
--
-- Performance rating systems: generic rating records, EU Energy Performance
-- Certificates (EPC), and CRREM (Carbon Risk Real Estate Monitor) stranding
-- assessments. Covers multiple rating frameworks including ENERGY STAR,
-- NABERS, EPC (A-G), DEC, BREEAM In-Use, and GRESB.
--
-- Tables (3):
--   1. pack035_energy_benchmark.performance_ratings
--   2. pack035_energy_benchmark.epc_certificates
--   3. pack035_energy_benchmark.crrem_assessments
--
-- Previous: V271__pack035_energy_benchmark_006.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack035_energy_benchmark.performance_ratings
-- =============================================================================
-- Generic performance rating records supporting multiple rating systems.
-- Stores both the label (e.g., "B") and numeric score where applicable.

CREATE TABLE pack035_energy_benchmark.performance_ratings (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack035_energy_benchmark.facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    rating_system           VARCHAR(50)     NOT NULL,
    rating_date             DATE            NOT NULL,
    rating_value            VARCHAR(20)     NOT NULL,
    numeric_score           DECIMAL(10, 4),
    primary_energy_kwh_m2   DECIMAL(10, 4),
    co2_kg_m2               DECIMAL(10, 4),
    valid_from              DATE,
    valid_until             DATE,
    assessor                VARCHAR(255),
    assessor_accreditation  VARCHAR(100),
    methodology_version     VARCHAR(50),
    certificate_url         TEXT,
    is_current              BOOLEAN         DEFAULT true,
    provenance_hash         VARCHAR(64)     NOT NULL,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_pr_system CHECK (
        rating_system IN (
            'ENERGY_STAR', 'EPC', 'DEC', 'NABERS', 'BREEAM_IN_USE',
            'LEED_EBOM', 'GRESB', 'DGNB', 'HQE', 'VERDE',
            'GREEN_STAR', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p035_pr_pe CHECK (
        primary_energy_kwh_m2 IS NULL OR primary_energy_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p035_pr_co2 CHECK (
        co2_kg_m2 IS NULL OR co2_kg_m2 >= 0
    ),
    CONSTRAINT chk_p035_pr_validity CHECK (
        valid_from IS NULL OR valid_until IS NULL OR valid_from <= valid_until
    )
);

-- Indexes
CREATE INDEX idx_p035_pr_facility        ON pack035_energy_benchmark.performance_ratings(facility_id);
CREATE INDEX idx_p035_pr_tenant          ON pack035_energy_benchmark.performance_ratings(tenant_id);
CREATE INDEX idx_p035_pr_system          ON pack035_energy_benchmark.performance_ratings(rating_system);
CREATE INDEX idx_p035_pr_date            ON pack035_energy_benchmark.performance_ratings(rating_date DESC);
CREATE INDEX idx_p035_pr_value           ON pack035_energy_benchmark.performance_ratings(rating_value);
CREATE INDEX idx_p035_pr_current         ON pack035_energy_benchmark.performance_ratings(is_current);
CREATE INDEX idx_p035_pr_valid_until     ON pack035_energy_benchmark.performance_ratings(valid_until);
CREATE INDEX idx_p035_pr_fac_system      ON pack035_energy_benchmark.performance_ratings(facility_id, rating_system);

-- Trigger
CREATE TRIGGER trg_p035_pr_updated
    BEFORE UPDATE ON pack035_energy_benchmark.performance_ratings
    FOR EACH ROW EXECUTE FUNCTION pack035_energy_benchmark.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack035_energy_benchmark.epc_certificates
-- =============================================================================
-- EU Energy Performance Certificate (EPC) records per EPBD (Energy
-- Performance of Buildings Directive). Tracks certificate number, rating
-- (A-G), primary energy, CO2, assessed floor area, and registry details.

CREATE TABLE pack035_energy_benchmark.epc_certificates (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack035_energy_benchmark.facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    certificate_number      VARCHAR(100)    NOT NULL,
    rating                  CHAR(1)         NOT NULL,
    rating_numeric          INTEGER,
    primary_energy_kwh_m2   DECIMAL(10, 4)  NOT NULL,
    co2_emissions_kg_m2     DECIMAL(10, 4),
    floor_area_m2           DECIMAL(12, 2),
    issue_date              DATE            NOT NULL,
    expiry_date             DATE,
    assessor_name           VARCHAR(255),
    assessor_number         VARCHAR(50),
    issuing_body            VARCHAR(255),
    registry_url            TEXT,
    country_code            CHAR(2),
    -- EPC recommendation tracking
    recommendations_count   INTEGER         DEFAULT 0,
    estimated_savings_eur   DECIMAL(14, 4),
    potential_rating        CHAR(1),
    potential_energy_kwh_m2 DECIMAL(10, 4),
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_epc_rating CHECK (
        rating IN ('A', 'B', 'C', 'D', 'E', 'F', 'G')
    ),
    CONSTRAINT chk_p035_epc_pe CHECK (
        primary_energy_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p035_epc_co2 CHECK (
        co2_emissions_kg_m2 IS NULL OR co2_emissions_kg_m2 >= 0
    ),
    CONSTRAINT chk_p035_epc_area CHECK (
        floor_area_m2 IS NULL OR floor_area_m2 > 0
    ),
    CONSTRAINT chk_p035_epc_dates CHECK (
        expiry_date IS NULL OR expiry_date > issue_date
    ),
    CONSTRAINT chk_p035_epc_potential CHECK (
        potential_rating IS NULL OR potential_rating IN ('A', 'B', 'C', 'D', 'E', 'F', 'G')
    )
);

-- Indexes
CREATE INDEX idx_p035_epc_facility       ON pack035_energy_benchmark.epc_certificates(facility_id);
CREATE INDEX idx_p035_epc_tenant         ON pack035_energy_benchmark.epc_certificates(tenant_id);
CREATE INDEX idx_p035_epc_cert           ON pack035_energy_benchmark.epc_certificates(certificate_number);
CREATE INDEX idx_p035_epc_rating         ON pack035_energy_benchmark.epc_certificates(rating);
CREATE INDEX idx_p035_epc_issue          ON pack035_energy_benchmark.epc_certificates(issue_date DESC);
CREATE INDEX idx_p035_epc_expiry         ON pack035_energy_benchmark.epc_certificates(expiry_date);
CREATE INDEX idx_p035_epc_country        ON pack035_energy_benchmark.epc_certificates(country_code);

-- =============================================================================
-- Table 3: pack035_energy_benchmark.crrem_assessments
-- =============================================================================
-- CRREM (Carbon Risk Real Estate Monitor) stranding risk assessments.
-- Evaluates a facility against CRREM decarbonisation pathways to
-- determine the year when the building becomes stranded (exceeds
-- the carbon pathway target).

CREATE TABLE pack035_energy_benchmark.crrem_assessments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack035_energy_benchmark.facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    assessment_date         DATE            NOT NULL,
    crrem_version           VARCHAR(20)     DEFAULT '2.0',
    pathway_scenario        VARCHAR(50)     NOT NULL,
    building_type           VARCHAR(50)     NOT NULL,
    country_code            CHAR(2)         NOT NULL,
    -- Current performance
    current_carbon_intensity DECIMAL(10, 4) NOT NULL,
    current_energy_intensity DECIMAL(10, 4),
    intensity_unit          VARCHAR(30)     DEFAULT 'kgCO2/m2/yr',
    -- Pathway targets
    pathway_target_current  DECIMAL(10, 4),
    pathway_target_2030     DECIMAL(10, 4),
    pathway_target_2040     DECIMAL(10, 4),
    pathway_target_2050     DECIMAL(10, 4),
    -- Stranding analysis
    stranding_year          INTEGER,
    stranding_risk          VARCHAR(20)     NOT NULL,
    years_until_stranding   INTEGER,
    excess_emissions_kg_m2  DECIMAL(10, 4),
    -- Required actions
    annual_reduction_needed_pct DECIMAL(8, 4),
    capex_estimate_eur      DECIMAL(16, 4),
    target_energy_intensity DECIMAL(10, 4),
    -- Metadata
    assessor                VARCHAR(255),
    provenance_hash         VARCHAR(64)     NOT NULL,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_crrem_scenario CHECK (
        pathway_scenario IN ('1.5C', '2.0C', 'BELOW_2C', 'NDC', 'CUSTOM')
    ),
    CONSTRAINT chk_p035_crrem_risk CHECK (
        stranding_risk IN ('HIGH', 'MEDIUM', 'LOW', 'NONE', 'ALREADY_STRANDED')
    ),
    CONSTRAINT chk_p035_crrem_ci CHECK (
        current_carbon_intensity >= 0
    ),
    CONSTRAINT chk_p035_crrem_ei CHECK (
        current_energy_intensity IS NULL OR current_energy_intensity >= 0
    ),
    CONSTRAINT chk_p035_crrem_stranding_yr CHECK (
        stranding_year IS NULL OR (stranding_year >= 2020 AND stranding_year <= 2100)
    )
);

-- Indexes
CREATE INDEX idx_p035_crrem_facility     ON pack035_energy_benchmark.crrem_assessments(facility_id);
CREATE INDEX idx_p035_crrem_tenant       ON pack035_energy_benchmark.crrem_assessments(tenant_id);
CREATE INDEX idx_p035_crrem_date         ON pack035_energy_benchmark.crrem_assessments(assessment_date DESC);
CREATE INDEX idx_p035_crrem_scenario     ON pack035_energy_benchmark.crrem_assessments(pathway_scenario);
CREATE INDEX idx_p035_crrem_risk         ON pack035_energy_benchmark.crrem_assessments(stranding_risk);
CREATE INDEX idx_p035_crrem_strand_yr    ON pack035_energy_benchmark.crrem_assessments(stranding_year);
CREATE INDEX idx_p035_crrem_country      ON pack035_energy_benchmark.crrem_assessments(country_code);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack035_energy_benchmark.performance_ratings ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack035_energy_benchmark.epc_certificates ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack035_energy_benchmark.crrem_assessments ENABLE ROW LEVEL SECURITY;

CREATE POLICY p035_pr_tenant_isolation ON pack035_energy_benchmark.performance_ratings
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_pr_service_bypass ON pack035_energy_benchmark.performance_ratings
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p035_epc_tenant_isolation ON pack035_energy_benchmark.epc_certificates
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_epc_service_bypass ON pack035_energy_benchmark.epc_certificates
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p035_crrem_tenant_isolation ON pack035_energy_benchmark.crrem_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_crrem_service_bypass ON pack035_energy_benchmark.crrem_assessments
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.performance_ratings TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.epc_certificates TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.crrem_assessments TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack035_energy_benchmark.performance_ratings IS
    'Generic performance ratings supporting ENERGY STAR, EPC, DEC, NABERS, BREEAM In-Use, GRESB, and custom systems.';
COMMENT ON TABLE pack035_energy_benchmark.epc_certificates IS
    'EU Energy Performance Certificates per EPBD with rating (A-G), primary energy, CO2, and improvement recommendations.';
COMMENT ON TABLE pack035_energy_benchmark.crrem_assessments IS
    'CRREM stranding risk assessments evaluating facilities against 1.5C/2C decarbonisation pathways.';

COMMENT ON COLUMN pack035_energy_benchmark.crrem_assessments.stranding_year IS
    'Year when the building is projected to become stranded (exceed carbon pathway). NULL if not stranded before 2050.';
COMMENT ON COLUMN pack035_energy_benchmark.crrem_assessments.stranding_risk IS
    'Stranding risk level: ALREADY_STRANDED (exceeds now), HIGH (<5yr), MEDIUM (5-15yr), LOW (15-25yr), NONE (>25yr or never).';
COMMENT ON COLUMN pack035_energy_benchmark.epc_certificates.potential_rating IS
    'Potential EPC rating achievable through recommended improvement measures.';
