-- =============================================================================
-- V190: PACK-028 Sector Pathway Pack - IEA NZE Milestones
-- =============================================================================
-- Pack:         PACK-028 (Sector Pathway Pack)
-- Migration:    010 of 015
-- Date:         March 2026
--
-- IEA Net Zero by 2050 technology milestones (400+) with sector mapping,
-- regional variants, progress tracking, and company milestone alignment
-- assessment across all IEA NZE 2050 chapters.
--
-- Tables (1):
--   1. pack028_sector_pathway.gl_iea_technology_milestones
--
-- Previous: V189__PACK028_sbti_sda_data.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack028_sector_pathway.gl_iea_technology_milestones
-- =============================================================================

CREATE TABLE pack028_sector_pathway.gl_iea_technology_milestones (
    milestone_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    company_id                  UUID,
    -- IEA reference
    iea_milestone_code          VARCHAR(40)     NOT NULL,
    iea_report                  VARCHAR(100)    NOT NULL DEFAULT 'IEA NZE 2050',
    iea_report_year             INTEGER         NOT NULL DEFAULT 2023,
    iea_chapter                 VARCHAR(100)    NOT NULL,
    iea_chapter_number          INTEGER,
    -- Sector mapping
    sector                      VARCHAR(80)     NOT NULL,
    sector_code                 VARCHAR(20)     NOT NULL,
    sub_sector                  VARCHAR(80),
    -- Milestone definition
    milestone_name              VARCHAR(500)    NOT NULL,
    milestone_description       TEXT,
    milestone_category          VARCHAR(50)     NOT NULL,
    milestone_type              VARCHAR(30)     NOT NULL DEFAULT 'TECHNOLOGY',
    -- Target
    target_year                 INTEGER         NOT NULL,
    target_value                DECIMAL(18,4),
    target_unit                 VARCHAR(80),
    target_description          VARCHAR(500),
    -- Baseline
    baseline_year               INTEGER         DEFAULT 2020,
    baseline_value              DECIMAL(18,4),
    baseline_unit               VARCHAR(80),
    -- Interim milestones
    interim_2025_value          DECIMAL(18,4),
    interim_2030_value          DECIMAL(18,4),
    interim_2035_value          DECIMAL(18,4),
    interim_2040_value          DECIMAL(18,4),
    interim_2045_value          DECIMAL(18,4),
    annual_trajectory           JSONB           DEFAULT '{}',
    -- Regional variants
    region                      VARCHAR(30)     NOT NULL DEFAULT 'GLOBAL',
    oecd_value                  DECIMAL(18,4),
    non_oecd_value              DECIMAL(18,4),
    regional_variants           JSONB           DEFAULT '{}',
    -- Company progress tracking
    company_current_value       DECIMAL(18,4),
    company_current_year        INTEGER,
    company_progress_pct        DECIMAL(6,2),
    company_on_track            BOOLEAN,
    company_gap_to_milestone    DECIMAL(18,4),
    company_gap_pct             DECIMAL(8,4),
    company_required_rate       DECIMAL(8,4),
    -- Progress status
    global_progress_status      VARCHAR(20)     DEFAULT 'NOT_ASSESSED',
    global_current_value        DECIMAL(18,4),
    global_progress_pct         DECIMAL(6,2),
    -- Technology details
    technology_name             VARCHAR(255),
    technology_trl              INTEGER,
    technology_commercial_year  INTEGER,
    technology_cost_current     DECIMAL(14,2),
    technology_cost_target      DECIMAL(14,2),
    technology_cost_unit        VARCHAR(50),
    cost_decline_pct            DECIMAL(6,2),
    learning_rate_pct           DECIMAL(6,2),
    -- Dependencies
    prerequisite_milestones     TEXT[]          DEFAULT '{}',
    enabling_policies           TEXT[]          DEFAULT '{}',
    infrastructure_requirements TEXT[]          DEFAULT '{}',
    -- Investment requirements
    global_investment_required  DECIMAL(18,2),
    investment_currency         VARCHAR(3)      DEFAULT 'USD',
    investment_timeframe        VARCHAR(20),
    -- Emissions impact
    emissions_reduction_mtco2   DECIMAL(14,2),
    emissions_reduction_pct     DECIMAL(6,2),
    -- Confidence
    achievement_probability     DECIMAL(5,2),
    confidence_level            VARCHAR(20)     DEFAULT 'MEDIUM',
    -- Classification
    priority_level              VARCHAR(20)     DEFAULT 'HIGH',
    is_critical_milestone       BOOLEAN         DEFAULT FALSE,
    is_no_regret_action         BOOLEAN         DEFAULT FALSE,
    -- Metadata
    is_reference_data           BOOLEAN         DEFAULT FALSE,
    is_active                   BOOLEAN         DEFAULT TRUE,
    data_source                 VARCHAR(200),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p028_iem_milestone_category CHECK (
        milestone_category IN (
            'RENEWABLE_CAPACITY', 'FOSSIL_PHASE_OUT', 'ELECTRIFICATION',
            'HYDROGEN_PRODUCTION', 'CCS_DEPLOYMENT', 'ENERGY_EFFICIENCY',
            'EV_ADOPTION', 'BATTERY_STORAGE', 'NUCLEAR_CAPACITY',
            'GRID_INFRASTRUCTURE', 'SUSTAINABLE_FUELS', 'BUILDING_RETROFIT',
            'INDUSTRIAL_PROCESS', 'METHANE_REDUCTION', 'CARBON_REMOVAL',
            'POLICY_REGULATION', 'INVESTMENT_FINANCE', 'INNOVATION_RD',
            'BEHAVIORAL_DEMAND', 'INFRASTRUCTURE', 'OTHER'
        )
    ),
    CONSTRAINT chk_p028_iem_milestone_type CHECK (
        milestone_type IN ('TECHNOLOGY', 'POLICY', 'INVESTMENT', 'BEHAVIORAL',
                           'INFRASTRUCTURE', 'MARKET', 'REGULATORY')
    ),
    CONSTRAINT chk_p028_iem_region CHECK (
        region IN ('GLOBAL', 'OECD', 'NON_OECD', 'EU', 'NORTH_AMERICA', 'ASIA_PACIFIC',
                   'LATIN_AMERICA', 'AFRICA', 'MIDDLE_EAST', 'CHINA', 'INDIA', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_iem_progress_status CHECK (
        global_progress_status IN ('NOT_ASSESSED', 'ON_TRACK', 'BEHIND',
                                   'WELL_BEHIND', 'ACHIEVED', 'NOT_STARTED')
    ),
    CONSTRAINT chk_p028_iem_target_year CHECK (
        target_year >= 2020 AND target_year <= 2100
    ),
    CONSTRAINT chk_p028_iem_trl CHECK (
        technology_trl IS NULL OR (technology_trl >= 1 AND technology_trl <= 9)
    ),
    CONSTRAINT chk_p028_iem_confidence CHECK (
        confidence_level IN ('VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW', 'VERY_LOW')
    ),
    CONSTRAINT chk_p028_iem_priority CHECK (
        priority_level IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
    ),
    CONSTRAINT chk_p028_iem_progress_pct CHECK (
        company_progress_pct IS NULL OR (company_progress_pct >= 0 AND company_progress_pct <= 200)
    ),
    CONSTRAINT chk_p028_iem_global_progress CHECK (
        global_progress_pct IS NULL OR (global_progress_pct >= 0 AND global_progress_pct <= 200)
    ),
    CONSTRAINT chk_p028_iem_probability CHECK (
        achievement_probability IS NULL OR (achievement_probability >= 0 AND achievement_probability <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p028_iem_tenant            ON pack028_sector_pathway.gl_iea_technology_milestones(tenant_id);
CREATE INDEX idx_p028_iem_company           ON pack028_sector_pathway.gl_iea_technology_milestones(company_id);
CREATE INDEX idx_p028_iem_milestone_code    ON pack028_sector_pathway.gl_iea_technology_milestones(iea_milestone_code);
CREATE INDEX idx_p028_iem_chapter           ON pack028_sector_pathway.gl_iea_technology_milestones(iea_chapter);
CREATE INDEX idx_p028_iem_chapter_num       ON pack028_sector_pathway.gl_iea_technology_milestones(iea_chapter_number);
CREATE INDEX idx_p028_iem_sector            ON pack028_sector_pathway.gl_iea_technology_milestones(sector_code);
CREATE INDEX idx_p028_iem_category          ON pack028_sector_pathway.gl_iea_technology_milestones(milestone_category);
CREATE INDEX idx_p028_iem_type              ON pack028_sector_pathway.gl_iea_technology_milestones(milestone_type);
CREATE INDEX idx_p028_iem_target_year       ON pack028_sector_pathway.gl_iea_technology_milestones(target_year);
CREATE INDEX idx_p028_iem_region            ON pack028_sector_pathway.gl_iea_technology_milestones(region);
CREATE INDEX idx_p028_iem_progress          ON pack028_sector_pathway.gl_iea_technology_milestones(global_progress_status);
CREATE INDEX idx_p028_iem_on_track          ON pack028_sector_pathway.gl_iea_technology_milestones(company_on_track) WHERE company_on_track IS NOT NULL;
CREATE INDEX idx_p028_iem_critical          ON pack028_sector_pathway.gl_iea_technology_milestones(is_critical_milestone) WHERE is_critical_milestone = TRUE;
CREATE INDEX idx_p028_iem_no_regret         ON pack028_sector_pathway.gl_iea_technology_milestones(is_no_regret_action) WHERE is_no_regret_action = TRUE;
CREATE INDEX idx_p028_iem_priority          ON pack028_sector_pathway.gl_iea_technology_milestones(priority_level);
CREATE INDEX idx_p028_iem_reference         ON pack028_sector_pathway.gl_iea_technology_milestones(is_reference_data) WHERE is_reference_data = TRUE;
CREATE INDEX idx_p028_iem_active            ON pack028_sector_pathway.gl_iea_technology_milestones(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p028_iem_sector_year       ON pack028_sector_pathway.gl_iea_technology_milestones(sector_code, target_year);
CREATE INDEX idx_p028_iem_sector_category   ON pack028_sector_pathway.gl_iea_technology_milestones(sector_code, milestone_category);
CREATE INDEX idx_p028_iem_company_sector    ON pack028_sector_pathway.gl_iea_technology_milestones(company_id, sector_code);
CREATE INDEX idx_p028_iem_behind            ON pack028_sector_pathway.gl_iea_technology_milestones(global_progress_status) WHERE global_progress_status IN ('BEHIND', 'WELL_BEHIND');
CREATE INDEX idx_p028_iem_created           ON pack028_sector_pathway.gl_iea_technology_milestones(created_at DESC);
CREATE INDEX idx_p028_iem_trajectory        ON pack028_sector_pathway.gl_iea_technology_milestones USING GIN(annual_trajectory);
CREATE INDEX idx_p028_iem_regional_var      ON pack028_sector_pathway.gl_iea_technology_milestones USING GIN(regional_variants);
CREATE INDEX idx_p028_iem_prereqs           ON pack028_sector_pathway.gl_iea_technology_milestones USING GIN(prerequisite_milestones);
CREATE INDEX idx_p028_iem_metadata          ON pack028_sector_pathway.gl_iea_technology_milestones USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p028_iea_milestones_updated
    BEFORE UPDATE ON pack028_sector_pathway.gl_iea_technology_milestones
    FOR EACH ROW EXECUTE FUNCTION pack028_sector_pathway.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack028_sector_pathway.gl_iea_technology_milestones ENABLE ROW LEVEL SECURITY;

CREATE POLICY p028_iem_tenant_isolation
    ON pack028_sector_pathway.gl_iea_technology_milestones
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p028_iem_service_bypass
    ON pack028_sector_pathway.gl_iea_technology_milestones
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack028_sector_pathway.gl_iea_technology_milestones TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack028_sector_pathway.gl_iea_technology_milestones IS
    'IEA Net Zero by 2050 technology milestones (400+) with sector mapping, regional variants, progress tracking, and company milestone alignment assessment.';

COMMENT ON COLUMN pack028_sector_pathway.gl_iea_technology_milestones.milestone_id IS 'Unique milestone record identifier.';
COMMENT ON COLUMN pack028_sector_pathway.gl_iea_technology_milestones.iea_milestone_code IS 'IEA milestone reference code (e.g., NZE-PWR-001, NZE-STL-015).';
COMMENT ON COLUMN pack028_sector_pathway.gl_iea_technology_milestones.milestone_category IS 'Milestone category: RENEWABLE_CAPACITY, FOSSIL_PHASE_OUT, HYDROGEN_PRODUCTION, CCS_DEPLOYMENT, etc.';
COMMENT ON COLUMN pack028_sector_pathway.gl_iea_technology_milestones.company_on_track IS 'Whether the company is on track to meet this milestone at current trajectory.';
COMMENT ON COLUMN pack028_sector_pathway.gl_iea_technology_milestones.is_critical_milestone IS 'IEA-flagged critical milestone that must be achieved for NZE pathway.';
COMMENT ON COLUMN pack028_sector_pathway.gl_iea_technology_milestones.is_no_regret_action IS 'IEA-flagged no-regret action beneficial regardless of scenario.';
COMMENT ON COLUMN pack028_sector_pathway.gl_iea_technology_milestones.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
