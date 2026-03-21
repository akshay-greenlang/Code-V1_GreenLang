-- =============================================================================
-- V153: PACK-025 Race to Zero - Sector Pathways
-- =============================================================================
-- Pack:         PACK-025 (Race to Zero Pack)
-- Migration:    006 of 010
-- Date:         March 2026
--
-- Sector-specific decarbonization pathways with IEA NZE benchmark data
-- pre-populated for 25 sectors. Organization-level sector alignment
-- assessment with gap analysis and recommended actions.
--
-- Tables (2):
--   1. pack025_race_to_zero.sector_pathways       (25 sectors pre-populated)
--   2. pack025_race_to_zero.sector_alignment
--
-- Previous: V152__pack025_race_to_zero_005.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack025_race_to_zero.sector_pathways
-- =============================================================================
-- Reference table of sector-specific decarbonization pathways with benchmark
-- emissions intensity data, technology readiness levels, and policy requirements.

CREATE TABLE pack025_race_to_zero.sector_pathways (
    pathway_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sector_nace             VARCHAR(10)     NOT NULL,
    sector_name             VARCHAR(255)    NOT NULL,
    sub_sector              VARCHAR(255),
    year                    INTEGER         NOT NULL,
    benchmark_emissions_intensity DECIMAL(18,6) NOT NULL,
    intensity_unit          VARCHAR(100)    NOT NULL,
    pathway_source          VARCHAR(50)     NOT NULL DEFAULT 'IEA_NZE',
    pathway_version         VARCHAR(30)     DEFAULT '2023',
    technology_trl          INTEGER,
    key_technologies        TEXT[],
    policy_requirements     TEXT,
    investment_needs_usd_bn DECIMAL(12,2),
    employment_impact       VARCHAR(50),
    regional_variations     JSONB           DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p025_spw_year CHECK (
        year >= 2020 AND year <= 2100
    ),
    CONSTRAINT chk_p025_spw_source CHECK (
        pathway_source IN ('IEA_NZE', 'IPCC_AR6', 'TPI', 'MPP', 'ACT', 'CRREM', 'SBTi_SDA', 'CUSTOM')
    ),
    CONSTRAINT chk_p025_spw_trl CHECK (
        technology_trl IS NULL OR (technology_trl >= 1 AND technology_trl <= 9)
    ),
    CONSTRAINT chk_p025_spw_intensity_non_neg CHECK (
        benchmark_emissions_intensity >= 0
    ),
    CONSTRAINT uq_p025_spw_sector_year_source UNIQUE (sector_nace, year, pathway_source)
);

-- ---------------------------------------------------------------------------
-- Indexes for sector_pathways
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p025_spw_sector      ON pack025_race_to_zero.sector_pathways(sector_nace);
CREATE INDEX idx_p025_spw_name        ON pack025_race_to_zero.sector_pathways(sector_name);
CREATE INDEX idx_p025_spw_year        ON pack025_race_to_zero.sector_pathways(year);
CREATE INDEX idx_p025_spw_source      ON pack025_race_to_zero.sector_pathways(pathway_source);
CREATE INDEX idx_p025_spw_sector_year ON pack025_race_to_zero.sector_pathways(sector_nace, year);
CREATE INDEX idx_p025_spw_trl         ON pack025_race_to_zero.sector_pathways(technology_trl);
CREATE INDEX idx_p025_spw_created     ON pack025_race_to_zero.sector_pathways(created_at DESC);
CREATE INDEX idx_p025_spw_regional    ON pack025_race_to_zero.sector_pathways USING GIN(regional_variations);
CREATE INDEX idx_p025_spw_metadata    ON pack025_race_to_zero.sector_pathways USING GIN(metadata);

-- =============================================================================
-- Pre-populate 25 sectors with IEA NZE 2030 benchmark data
-- =============================================================================
-- Benchmark emissions intensity values based on IEA Net Zero by 2050 scenario.
-- Values represent 2030 intermediate benchmarks.

INSERT INTO pack025_race_to_zero.sector_pathways (sector_nace, sector_name, year, benchmark_emissions_intensity, intensity_unit, pathway_source, technology_trl, policy_requirements) VALUES
-- Energy & Power
('D35.11', 'Electricity Generation',        2030, 0.138,  'tCO2/MWh',       'IEA_NZE', 9, 'Phase out unabated coal; 60% renewables share'),
('D35.21', 'Gas Distribution',              2030, 0.042,  'tCO2/GJ',        'IEA_NZE', 8, 'Methane leak reduction; hydrogen blending targets'),
('B06',    'Oil & Gas Extraction',           2030, 0.015,  'tCO2/boe',       'IEA_NZE', 7, 'Zero routine flaring; methane intensity < 0.5%'),
-- Heavy Industry
('C24.10', 'Iron & Steel',                   2030, 1.400,  'tCO2/t steel',   'IEA_NZE', 6, 'Green hydrogen DRI; scrap-based EAF expansion'),
('C23.51', 'Cement',                         2030, 0.520,  'tCO2/t cement',  'IEA_NZE', 5, 'Clinker substitution; CCS on kiln emissions'),
('C20.11', 'Basic Chemicals',                2030, 0.800,  'tCO2/t HVC',     'IEA_NZE', 6, 'Electrification of steam crackers; bio-feedstocks'),
('C24.42', 'Aluminium',                      2030, 6.500,  'tCO2/t Al',      'IEA_NZE', 7, 'Inert anodes; renewable power for smelting'),
('C17.11', 'Pulp & Paper',                   2030, 0.300,  'tCO2/t product', 'IEA_NZE', 8, 'Biomass boilers; energy efficiency improvements'),
-- Transport
('H49.10', 'Rail Transport',                 2030, 0.020,  'tCO2/tkm',       'IEA_NZE', 9, 'Full electrification of mainlines'),
('H50.10', 'Maritime Shipping',              2030, 0.008,  'tCO2/tkm',       'IEA_NZE', 5, 'Ammonia/methanol fuels; wind-assisted propulsion'),
('H51.10', 'Aviation',                       2030, 0.800,  'tCO2/RPK',       'IEA_NZE', 4, '10% SAF mandate; operational efficiency'),
('H49.31', 'Road Freight',                   2030, 0.062,  'tCO2/tkm',       'IEA_NZE', 7, 'Battery-electric and hydrogen trucks; modal shift'),
('C29.10', 'Automotive Manufacturing',       2030, 0.400,  'tCO2/vehicle',   'IEA_NZE', 8, '60% EV share of new sales; supply chain decarb'),
-- Buildings & Real Estate
('L68.20', 'Real Estate - Commercial',       2030, 0.040,  'tCO2/m2',        'IEA_NZE', 8, 'NZEB standards; heat pump deployment'),
('L68.10', 'Real Estate - Residential',      2030, 0.025,  'tCO2/m2',        'IEA_NZE', 8, 'Deep renovation; all-electric new builds'),
-- Agriculture & Food
('A01',    'Agriculture - Crops',            2030, 1.200,  'tCO2e/ha',       'IEA_NZE', 6, 'Precision agriculture; reduced fertilizer use'),
('A01.4',  'Agriculture - Livestock',        2030, 12.000, 'tCO2e/t protein','IEA_NZE', 4, 'Feed additives; manure management; herd optimization'),
('C10',    'Food Processing',                2030, 0.350,  'tCO2/t product', 'IEA_NZE', 7, 'Electrified process heat; cold chain efficiency'),
-- Finance
('K64',    'Financial Services - Banking',   2030, 0.070,  'tCO2e/M$ AUM',   'IEA_NZE', 8, 'Portfolio alignment; financed emissions disclosure'),
('K65',    'Insurance',                      2030, 0.050,  'tCO2e/M$ AUM',   'IEA_NZE', 7, 'Underwriting policy; climate risk integration'),
-- ICT & Services
('J62',    'IT & Software',                  2030, 0.030,  'tCO2/M$ revenue','IEA_NZE', 9, '100% renewable data centers; PUE < 1.2'),
('J61',    'Telecommunications',             2030, 0.015,  'tCO2/subscriber','IEA_NZE', 9, 'Network energy efficiency; renewable procurement'),
-- Mining & Resources
('B07',    'Mining',                          2030, 0.800,  'tCO2/kt ore',    'IEA_NZE', 6, 'Electric haul trucks; renewable power on site'),
-- Textiles & Fashion
('C13',    'Textiles & Apparel',             2030, 15.000, 'tCO2/t fiber',   'IEA_NZE', 6, 'Recycled fibers; energy-efficient dyeing'),
-- Healthcare
('Q86',    'Healthcare Services',            2030, 0.050,  'tCO2/M$ revenue','IEA_NZE', 7, 'Anesthetic gas management; building decarbonization');

-- =============================================================================
-- Table 2: pack025_race_to_zero.sector_alignment
-- =============================================================================
-- Organization-level sector alignment assessment comparing entity intensity
-- against benchmark pathways with gap analysis and recommended actions.

CREATE TABLE pack025_race_to_zero.sector_alignment (
    alignment_id            UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES pack025_race_to_zero.organization_profiles(org_id) ON DELETE CASCADE,
    pledge_id               UUID            REFERENCES pack025_race_to_zero.pledges(pledge_id) ON DELETE SET NULL,
    tenant_id               UUID            NOT NULL,
    sector                  VARCHAR(255)    NOT NULL,
    sector_nace             VARCHAR(10),
    assessment_date         DATE            NOT NULL,
    baseline_intensity      DECIMAL(18,6)   NOT NULL,
    target_intensity        DECIMAL(18,6)   NOT NULL,
    benchmark_intensity     DECIMAL(18,6)   NOT NULL,
    intensity_unit          VARCHAR(100)    NOT NULL,
    gap_pct                 DECIMAL(8,3),
    alignment_status        VARCHAR(30)     DEFAULT 'PENDING',
    actions_json            JSONB           DEFAULT '[]',
    technology_assessment   JSONB           DEFAULT '{}',
    investment_required_usd DECIMAL(18,2),
    timeline_years          INTEGER,
    pathway_source          VARCHAR(50)     DEFAULT 'IEA_NZE',
    recommendations         TEXT[],
    warnings                TEXT[]          DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p025_sa_alignment CHECK (
        alignment_status IN ('ALIGNED', 'PARTIALLY_ALIGNED', 'MISALIGNED', 'AHEAD', 'PENDING')
    ),
    CONSTRAINT chk_p025_sa_source CHECK (
        pathway_source IN ('IEA_NZE', 'IPCC_AR6', 'TPI', 'MPP', 'ACT', 'CRREM', 'SBTi_SDA', 'CUSTOM')
    ),
    CONSTRAINT chk_p025_sa_intensity_non_neg CHECK (
        baseline_intensity >= 0 AND target_intensity >= 0 AND benchmark_intensity >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for sector_alignment
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p025_sa_org             ON pack025_race_to_zero.sector_alignment(org_id);
CREATE INDEX idx_p025_sa_pledge          ON pack025_race_to_zero.sector_alignment(pledge_id);
CREATE INDEX idx_p025_sa_tenant          ON pack025_race_to_zero.sector_alignment(tenant_id);
CREATE INDEX idx_p025_sa_sector          ON pack025_race_to_zero.sector_alignment(sector);
CREATE INDEX idx_p025_sa_nace            ON pack025_race_to_zero.sector_alignment(sector_nace);
CREATE INDEX idx_p025_sa_date            ON pack025_race_to_zero.sector_alignment(assessment_date);
CREATE INDEX idx_p025_sa_alignment       ON pack025_race_to_zero.sector_alignment(alignment_status);
CREATE INDEX idx_p025_sa_source          ON pack025_race_to_zero.sector_alignment(pathway_source);
CREATE INDEX idx_p025_sa_gap             ON pack025_race_to_zero.sector_alignment(gap_pct);
CREATE INDEX idx_p025_sa_created         ON pack025_race_to_zero.sector_alignment(created_at DESC);
CREATE INDEX idx_p025_sa_actions         ON pack025_race_to_zero.sector_alignment USING GIN(actions_json);
CREATE INDEX idx_p025_sa_tech            ON pack025_race_to_zero.sector_alignment USING GIN(technology_assessment);
CREATE INDEX idx_p025_sa_metadata        ON pack025_race_to_zero.sector_alignment USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p025_sector_pathways_updated
    BEFORE UPDATE ON pack025_race_to_zero.sector_pathways
    FOR EACH ROW EXECUTE FUNCTION pack025_race_to_zero.fn_set_updated_at();

CREATE TRIGGER trg_p025_sector_alignment_updated
    BEFORE UPDATE ON pack025_race_to_zero.sector_alignment
    FOR EACH ROW EXECUTE FUNCTION pack025_race_to_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
-- sector_pathways is a reference table, no tenant isolation needed but enable RLS
ALTER TABLE pack025_race_to_zero.sector_pathways ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack025_race_to_zero.sector_alignment ENABLE ROW LEVEL SECURITY;

CREATE POLICY p025_spw_read_all
    ON pack025_race_to_zero.sector_pathways
    FOR SELECT
    USING (TRUE);

CREATE POLICY p025_spw_service_write
    ON pack025_race_to_zero.sector_pathways
    FOR ALL
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p025_sa_tenant_isolation
    ON pack025_race_to_zero.sector_alignment
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p025_sa_service_bypass
    ON pack025_race_to_zero.sector_alignment
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT ON pack025_race_to_zero.sector_pathways TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack025_race_to_zero.sector_alignment TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack025_race_to_zero.sector_pathways IS
    'Reference table of sector-specific decarbonization pathways with IEA NZE benchmark data for 25 sectors.';
COMMENT ON TABLE pack025_race_to_zero.sector_alignment IS
    'Organization-level sector alignment assessment comparing entity intensity against benchmark pathways.';

COMMENT ON COLUMN pack025_race_to_zero.sector_pathways.pathway_id IS 'Unique pathway benchmark identifier.';
COMMENT ON COLUMN pack025_race_to_zero.sector_pathways.sector_nace IS 'NACE Rev.2 sector classification code.';
COMMENT ON COLUMN pack025_race_to_zero.sector_pathways.benchmark_emissions_intensity IS 'Benchmark emissions intensity for the target year.';
COMMENT ON COLUMN pack025_race_to_zero.sector_pathways.technology_trl IS 'Technology Readiness Level (1-9) for key decarbonization tech.';
COMMENT ON COLUMN pack025_race_to_zero.sector_pathways.policy_requirements IS 'Key policy requirements for sector pathway achievement.';
COMMENT ON COLUMN pack025_race_to_zero.sector_alignment.alignment_id IS 'Unique sector alignment assessment identifier.';
COMMENT ON COLUMN pack025_race_to_zero.sector_alignment.gap_pct IS 'Percentage gap between entity intensity and benchmark.';
COMMENT ON COLUMN pack025_race_to_zero.sector_alignment.actions_json IS 'JSONB array of recommended actions to close the alignment gap.';
