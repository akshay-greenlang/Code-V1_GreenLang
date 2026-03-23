-- =============================================================================
-- V294: PACK-037 Demand Response Pack - Carbon Impact Assessment
-- =============================================================================
-- Pack:         PACK-037 (Demand Response Pack)
-- Migration:    009 of 010
-- Date:         March 2026
--
-- Carbon impact assessment tables linking demand response to emissions
-- reduction. Tracks marginal emission factors by grid region, per-event
-- carbon impact calculations, annual carbon summaries, carbon reports
-- for ESG/sustainability teams, and SBTi contributions from DR.
--
-- Tables (5):
--   1. pack037_demand_response.dr_marginal_emission_factors
--   2. pack037_demand_response.dr_event_carbon_impacts
--   3. pack037_demand_response.dr_annual_carbon_summaries
--   4. pack037_demand_response.dr_carbon_reports
--   5. pack037_demand_response.dr_sbti_contributions
--
-- Previous: V293__pack037_demand_response_008.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack037_demand_response.dr_marginal_emission_factors
-- =============================================================================
-- Marginal emission factors by ISO/RTO region, hour, and season.
-- DR curtailment avoids marginal generation, so marginal (not average)
-- factors are used for accurate carbon impact calculation.

CREATE TABLE pack037_demand_response.dr_marginal_emission_factors (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    iso_rto_region          VARCHAR(30)     NOT NULL,
    node_zone               VARCHAR(100),
    factor_date             DATE            NOT NULL,
    hour_of_day             INTEGER         NOT NULL,
    season                  VARCHAR(20)     NOT NULL,
    marginal_co2_kg_per_mwh NUMERIC(10,4)   NOT NULL,
    marginal_ch4_kg_per_mwh NUMERIC(10,6),
    marginal_n2o_kg_per_mwh NUMERIC(10,6),
    marginal_co2e_kg_per_mwh NUMERIC(10,4)  NOT NULL,
    marginal_fuel_type      VARCHAR(50),
    average_co2_kg_per_mwh  NUMERIC(10,4),
    residual_co2_kg_per_mwh NUMERIC(10,4),
    data_source             VARCHAR(100)    NOT NULL,
    data_vintage_year       INTEGER         NOT NULL,
    methodology             VARCHAR(50)     NOT NULL,
    confidence_level        VARCHAR(20)     DEFAULT 'HIGH',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_mef_region CHECK (
        iso_rto_region IN (
            'PJM', 'ERCOT', 'CAISO', 'ISO_NE', 'NYISO', 'MISO', 'SPP',
            'UK_NGESO', 'DE_TENNET', 'DE_AMPRION', 'DE_50HZ', 'DE_TRANSNET',
            'FR_RTE', 'NL_TENNET', 'ES_REE', 'IT_TERNA', 'AU_AEMO',
            'JP_TEPCO', 'JP_KEPCO', 'OTHER'
        )
    ),
    CONSTRAINT chk_p037_mef_hour CHECK (
        hour_of_day >= 0 AND hour_of_day <= 23
    ),
    CONSTRAINT chk_p037_mef_season CHECK (
        season IN ('SUMMER', 'WINTER', 'SHOULDER', 'ALL_YEAR')
    ),
    CONSTRAINT chk_p037_mef_co2 CHECK (
        marginal_co2_kg_per_mwh >= 0
    ),
    CONSTRAINT chk_p037_mef_co2e CHECK (
        marginal_co2e_kg_per_mwh >= 0
    ),
    CONSTRAINT chk_p037_mef_fuel CHECK (
        marginal_fuel_type IS NULL OR marginal_fuel_type IN (
            'NATURAL_GAS_CCGT', 'NATURAL_GAS_CT', 'COAL', 'OIL',
            'NUCLEAR', 'HYDRO', 'WIND', 'SOLAR', 'BIOMASS',
            'MIXED', 'UNKNOWN'
        )
    ),
    CONSTRAINT chk_p037_mef_methodology CHECK (
        methodology IN (
            'WATTTIME_MOER', 'EPA_EGRID', 'EIA_930', 'UNFCCC_CDM',
            'IEA_NATIONAL', 'ELECTRICITY_MAP', 'ENTSOE', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p037_mef_confidence CHECK (
        confidence_level IS NULL OR confidence_level IN ('HIGH', 'MEDIUM', 'LOW')
    ),
    CONSTRAINT chk_p037_mef_vintage CHECK (
        data_vintage_year >= 2015 AND data_vintage_year <= 2100
    ),
    CONSTRAINT uq_p037_mef_region_date_hour UNIQUE (iso_rto_region, node_zone, factor_date, hour_of_day)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_mef_region         ON pack037_demand_response.dr_marginal_emission_factors(iso_rto_region);
CREATE INDEX idx_p037_mef_zone           ON pack037_demand_response.dr_marginal_emission_factors(node_zone);
CREATE INDEX idx_p037_mef_date           ON pack037_demand_response.dr_marginal_emission_factors(factor_date DESC);
CREATE INDEX idx_p037_mef_hour           ON pack037_demand_response.dr_marginal_emission_factors(hour_of_day);
CREATE INDEX idx_p037_mef_season         ON pack037_demand_response.dr_marginal_emission_factors(season);
CREATE INDEX idx_p037_mef_source         ON pack037_demand_response.dr_marginal_emission_factors(data_source);
CREATE INDEX idx_p037_mef_methodology    ON pack037_demand_response.dr_marginal_emission_factors(methodology);
CREATE INDEX idx_p037_mef_created        ON pack037_demand_response.dr_marginal_emission_factors(created_at DESC);

-- Composite: region + date + hour for event-time lookups
CREATE INDEX idx_p037_mef_region_date_hr ON pack037_demand_response.dr_marginal_emission_factors(iso_rto_region, factor_date, hour_of_day);

-- =============================================================================
-- Table 2: pack037_demand_response.dr_event_carbon_impacts
-- =============================================================================
-- Per-event carbon impact calculation showing the emissions avoided
-- by curtailing load during each DR event. Uses marginal emission
-- factors for the specific region, date, and hour of curtailment.

CREATE TABLE pack037_demand_response.dr_event_carbon_impacts (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id                UUID            NOT NULL REFERENCES pack037_demand_response.dr_events(id) ON DELETE CASCADE,
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id),
    tenant_id               UUID            NOT NULL,
    curtailment_mwh         NUMERIC(14,6)   NOT NULL,
    event_hours             NUMERIC(8,2)    NOT NULL,
    avg_marginal_co2e_kg_mwh NUMERIC(10,4)  NOT NULL,
    avoided_co2_kg          NUMERIC(14,4)   NOT NULL,
    avoided_co2_tonnes      NUMERIC(14,6)   NOT NULL,
    avoided_ch4_kg          NUMERIC(10,6),
    avoided_n2o_kg          NUMERIC(10,6),
    avoided_co2e_kg         NUMERIC(14,4)   NOT NULL,
    avoided_co2e_tonnes     NUMERIC(14,6)   NOT NULL,
    rebound_co2e_kg         NUMERIC(14,4)   DEFAULT 0,
    net_avoided_co2e_kg     NUMERIC(14,4)   NOT NULL,
    net_avoided_co2e_tonnes NUMERIC(14,6)   NOT NULL,
    marginal_fuel_displaced VARCHAR(50),
    emission_factor_source  VARCHAR(100),
    calculation_methodology VARCHAR(50)     NOT NULL,
    data_quality_score      NUMERIC(5,2),
    hourly_breakdown        JSONB           DEFAULT '[]',
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_eci_mwh CHECK (
        curtailment_mwh >= 0
    ),
    CONSTRAINT chk_p037_eci_hours CHECK (
        event_hours > 0
    ),
    CONSTRAINT chk_p037_eci_co2 CHECK (
        avoided_co2_kg >= 0
    ),
    CONSTRAINT chk_p037_eci_co2e CHECK (
        avoided_co2e_kg >= 0
    ),
    CONSTRAINT chk_p037_eci_net CHECK (
        net_avoided_co2e_kg >= 0
    ),
    CONSTRAINT chk_p037_eci_rebound CHECK (
        rebound_co2e_kg >= 0
    ),
    CONSTRAINT chk_p037_eci_methodology CHECK (
        calculation_methodology IN (
            'MARGINAL_HOURLY', 'MARGINAL_AVERAGE', 'AVERAGE_GRID',
            'RESIDUAL_MIX', 'LOCATION_BASED', 'MARKET_BASED'
        )
    ),
    CONSTRAINT chk_p037_eci_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT uq_p037_eci_event UNIQUE (event_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_eci_event          ON pack037_demand_response.dr_event_carbon_impacts(event_id);
CREATE INDEX idx_p037_eci_facility       ON pack037_demand_response.dr_event_carbon_impacts(facility_profile_id);
CREATE INDEX idx_p037_eci_tenant         ON pack037_demand_response.dr_event_carbon_impacts(tenant_id);
CREATE INDEX idx_p037_eci_co2e           ON pack037_demand_response.dr_event_carbon_impacts(net_avoided_co2e_tonnes DESC);
CREATE INDEX idx_p037_eci_methodology    ON pack037_demand_response.dr_event_carbon_impacts(calculation_methodology);
CREATE INDEX idx_p037_eci_created        ON pack037_demand_response.dr_event_carbon_impacts(created_at DESC);

-- =============================================================================
-- Table 3: pack037_demand_response.dr_annual_carbon_summaries
-- =============================================================================
-- Annual aggregated carbon impact summaries per facility, suitable for
-- sustainability reporting, GHG inventories, and ESG disclosures.

CREATE TABLE pack037_demand_response.dr_annual_carbon_summaries (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id),
    tenant_id               UUID            NOT NULL,
    reporting_year          INTEGER         NOT NULL,
    total_events            INTEGER         NOT NULL DEFAULT 0,
    total_curtailment_mwh   NUMERIC(14,4)   NOT NULL DEFAULT 0,
    total_avoided_co2e_tonnes NUMERIC(14,6) NOT NULL DEFAULT 0,
    summer_avoided_co2e_tonnes NUMERIC(14,6) DEFAULT 0,
    winter_avoided_co2e_tonnes NUMERIC(14,6) DEFAULT 0,
    total_rebound_co2e_tonnes NUMERIC(14,6) DEFAULT 0,
    net_avoided_co2e_tonnes NUMERIC(14,6)   NOT NULL DEFAULT 0,
    avg_marginal_ef_kg_mwh  NUMERIC(10,4),
    weighted_avg_ef_kg_mwh  NUMERIC(10,4),
    scope2_equivalent_pct   NUMERIC(6,2),
    yoy_change_pct          NUMERIC(8,4),
    carbon_intensity_kg_per_kw NUMERIC(10,4),
    equivalent_trees_planted INTEGER,
    equivalent_cars_removed  INTEGER,
    equivalent_homes_powered INTEGER,
    methodology_used        VARCHAR(50),
    verification_status     VARCHAR(20)     DEFAULT 'UNVERIFIED',
    verified_by             VARCHAR(255),
    verified_at             TIMESTAMPTZ,
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_acs_year CHECK (
        reporting_year >= 2020 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p037_acs_events CHECK (
        total_events >= 0
    ),
    CONSTRAINT chk_p037_acs_mwh CHECK (
        total_curtailment_mwh >= 0
    ),
    CONSTRAINT chk_p037_acs_co2e CHECK (
        total_avoided_co2e_tonnes >= 0
    ),
    CONSTRAINT chk_p037_acs_net CHECK (
        net_avoided_co2e_tonnes >= 0
    ),
    CONSTRAINT chk_p037_acs_scope2 CHECK (
        scope2_equivalent_pct IS NULL OR (scope2_equivalent_pct >= 0 AND scope2_equivalent_pct <= 100)
    ),
    CONSTRAINT chk_p037_acs_verification CHECK (
        verification_status IS NULL OR verification_status IN (
            'UNVERIFIED', 'SELF_VERIFIED', 'THIRD_PARTY_VERIFIED', 'AUDITED'
        )
    ),
    CONSTRAINT uq_p037_acs_facility_year UNIQUE (facility_profile_id, reporting_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_acs_facility       ON pack037_demand_response.dr_annual_carbon_summaries(facility_profile_id);
CREATE INDEX idx_p037_acs_tenant         ON pack037_demand_response.dr_annual_carbon_summaries(tenant_id);
CREATE INDEX idx_p037_acs_year           ON pack037_demand_response.dr_annual_carbon_summaries(reporting_year DESC);
CREATE INDEX idx_p037_acs_co2e           ON pack037_demand_response.dr_annual_carbon_summaries(net_avoided_co2e_tonnes DESC);
CREATE INDEX idx_p037_acs_verification   ON pack037_demand_response.dr_annual_carbon_summaries(verification_status);
CREATE INDEX idx_p037_acs_created        ON pack037_demand_response.dr_annual_carbon_summaries(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_acs_updated
    BEFORE UPDATE ON pack037_demand_response.dr_annual_carbon_summaries
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack037_demand_response.dr_carbon_reports
-- =============================================================================
-- Generated carbon impact reports for ESG, sustainability, GHG inventory,
-- and regulatory disclosures. Supports multiple reporting frameworks
-- (GHG Protocol, CDP, TCFD, CSRD, SBTi).

CREATE TABLE pack037_demand_response.dr_carbon_reports (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_profile_id     UUID            REFERENCES pack037_demand_response.dr_facility_profiles(id),
    tenant_id               UUID            NOT NULL,
    report_name             VARCHAR(255)    NOT NULL,
    report_type             VARCHAR(50)     NOT NULL,
    reporting_framework     VARCHAR(50)     NOT NULL,
    reporting_year          INTEGER         NOT NULL,
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    scope                   VARCHAR(30)     NOT NULL DEFAULT 'SCOPE_2',
    total_avoided_co2e_tonnes NUMERIC(14,6) NOT NULL,
    net_avoided_co2e_tonnes NUMERIC(14,6)   NOT NULL,
    total_curtailment_mwh   NUMERIC(14,4),
    report_narrative        TEXT,
    report_status           VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    generated_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    approved_by             VARCHAR(255),
    approved_at             TIMESTAMPTZ,
    report_file_path        TEXT,
    report_format           VARCHAR(20)     DEFAULT 'PDF',
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_crpt_type CHECK (
        report_type IN (
            'GHG_INVENTORY_SUPPLEMENT', 'ESG_DISCLOSURE', 'SUSTAINABILITY_REPORT',
            'CDP_RESPONSE', 'TCFD_DISCLOSURE', 'CSRD_SUPPLEMENT',
            'REGULATORY_FILING', 'INTERNAL_DASHBOARD', 'BOARD_REPORT'
        )
    ),
    CONSTRAINT chk_p037_crpt_framework CHECK (
        reporting_framework IN (
            'GHG_PROTOCOL', 'CDP', 'TCFD', 'CSRD', 'SASB', 'GRI',
            'SBTi', 'ISO_14064', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p037_crpt_year CHECK (
        reporting_year >= 2020 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p037_crpt_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'ALL_SCOPES')
    ),
    CONSTRAINT chk_p037_crpt_dates CHECK (
        period_end >= period_start
    ),
    CONSTRAINT chk_p037_crpt_status CHECK (
        report_status IN (
            'DRAFT', 'REVIEW', 'APPROVED', 'PUBLISHED', 'ARCHIVED'
        )
    ),
    CONSTRAINT chk_p037_crpt_format CHECK (
        report_format IS NULL OR report_format IN ('PDF', 'EXCEL', 'CSV', 'XML', 'JSON', 'HTML')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_crpt_facility      ON pack037_demand_response.dr_carbon_reports(facility_profile_id);
CREATE INDEX idx_p037_crpt_tenant        ON pack037_demand_response.dr_carbon_reports(tenant_id);
CREATE INDEX idx_p037_crpt_type          ON pack037_demand_response.dr_carbon_reports(report_type);
CREATE INDEX idx_p037_crpt_framework     ON pack037_demand_response.dr_carbon_reports(reporting_framework);
CREATE INDEX idx_p037_crpt_year          ON pack037_demand_response.dr_carbon_reports(reporting_year DESC);
CREATE INDEX idx_p037_crpt_status        ON pack037_demand_response.dr_carbon_reports(report_status);
CREATE INDEX idx_p037_crpt_scope         ON pack037_demand_response.dr_carbon_reports(scope);
CREATE INDEX idx_p037_crpt_created       ON pack037_demand_response.dr_carbon_reports(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_crpt_updated
    BEFORE UPDATE ON pack037_demand_response.dr_carbon_reports
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack037_demand_response.dr_sbti_contributions
-- =============================================================================
-- Tracks how DR participation contributes to Science Based Targets
-- initiative (SBTi) emission reduction targets. Links DR-avoided
-- emissions to the overall decarbonization pathway.

CREATE TABLE pack037_demand_response.dr_sbti_contributions (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_profile_id     UUID            REFERENCES pack037_demand_response.dr_facility_profiles(id),
    tenant_id               UUID            NOT NULL,
    target_year             INTEGER         NOT NULL,
    sbti_target_type        VARCHAR(30)     NOT NULL,
    base_year               INTEGER         NOT NULL,
    base_year_emissions_tco2e NUMERIC(14,4) NOT NULL,
    target_emissions_tco2e  NUMERIC(14,4)   NOT NULL,
    current_emissions_tco2e NUMERIC(14,4),
    dr_avoided_tco2e        NUMERIC(14,6)   NOT NULL,
    dr_contribution_pct     NUMERIC(6,2)    NOT NULL,
    cumulative_dr_tco2e     NUMERIC(14,4),
    remaining_gap_tco2e     NUMERIC(14,4),
    on_track                BOOLEAN,
    trajectory_year         INTEGER         NOT NULL,
    linear_target_tco2e     NUMERIC(14,4),
    pathway_alignment       VARCHAR(20),
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_sbti_target_type CHECK (
        sbti_target_type IN (
            'NEAR_TERM_1_5C', 'NEAR_TERM_WB2C', 'LONG_TERM_NET_ZERO',
            'SECTOR_SPECIFIC', 'ABSOLUTE_CONTRACTION', 'INTENSITY_TARGET'
        )
    ),
    CONSTRAINT chk_p037_sbti_target_year CHECK (
        target_year >= 2025 AND target_year <= 2100
    ),
    CONSTRAINT chk_p037_sbti_base_year CHECK (
        base_year >= 2015 AND base_year <= 2030
    ),
    CONSTRAINT chk_p037_sbti_base_emissions CHECK (
        base_year_emissions_tco2e > 0
    ),
    CONSTRAINT chk_p037_sbti_target_emissions CHECK (
        target_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p037_sbti_dr_avoided CHECK (
        dr_avoided_tco2e >= 0
    ),
    CONSTRAINT chk_p037_sbti_contribution CHECK (
        dr_contribution_pct >= 0 AND dr_contribution_pct <= 100
    ),
    CONSTRAINT chk_p037_sbti_trajectory CHECK (
        trajectory_year >= base_year AND trajectory_year <= target_year
    ),
    CONSTRAINT chk_p037_sbti_alignment CHECK (
        pathway_alignment IS NULL OR pathway_alignment IN (
            '1_5C_ALIGNED', 'WB2C_ALIGNED', 'NOT_ALIGNED', 'AHEAD_OF_TARGET', 'BEHIND_TARGET'
        )
    ),
    CONSTRAINT uq_p037_sbti_fac_target_year UNIQUE (facility_profile_id, sbti_target_type, trajectory_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_sbti_facility      ON pack037_demand_response.dr_sbti_contributions(facility_profile_id);
CREATE INDEX idx_p037_sbti_tenant        ON pack037_demand_response.dr_sbti_contributions(tenant_id);
CREATE INDEX idx_p037_sbti_target_year   ON pack037_demand_response.dr_sbti_contributions(target_year);
CREATE INDEX idx_p037_sbti_type          ON pack037_demand_response.dr_sbti_contributions(sbti_target_type);
CREATE INDEX idx_p037_sbti_traj_year     ON pack037_demand_response.dr_sbti_contributions(trajectory_year DESC);
CREATE INDEX idx_p037_sbti_on_track      ON pack037_demand_response.dr_sbti_contributions(on_track);
CREATE INDEX idx_p037_sbti_alignment     ON pack037_demand_response.dr_sbti_contributions(pathway_alignment);
CREATE INDEX idx_p037_sbti_created       ON pack037_demand_response.dr_sbti_contributions(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_sbti_updated
    BEFORE UPDATE ON pack037_demand_response.dr_sbti_contributions
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack037_demand_response.dr_marginal_emission_factors ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_event_carbon_impacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_annual_carbon_summaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_carbon_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_sbti_contributions ENABLE ROW LEVEL SECURITY;

-- Marginal emission factors are shared reference data
CREATE POLICY p037_mef_read_all ON pack037_demand_response.dr_marginal_emission_factors
    FOR SELECT USING (TRUE);
CREATE POLICY p037_mef_service_bypass ON pack037_demand_response.dr_marginal_emission_factors
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_eci_tenant_isolation ON pack037_demand_response.dr_event_carbon_impacts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_eci_service_bypass ON pack037_demand_response.dr_event_carbon_impacts
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_acs_tenant_isolation ON pack037_demand_response.dr_annual_carbon_summaries
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_acs_service_bypass ON pack037_demand_response.dr_annual_carbon_summaries
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_crpt_tenant_isolation ON pack037_demand_response.dr_carbon_reports
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_crpt_service_bypass ON pack037_demand_response.dr_carbon_reports
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_sbti_tenant_isolation ON pack037_demand_response.dr_sbti_contributions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_sbti_service_bypass ON pack037_demand_response.dr_sbti_contributions
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_marginal_emission_factors TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_event_carbon_impacts TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_annual_carbon_summaries TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_carbon_reports TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_sbti_contributions TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack037_demand_response.dr_marginal_emission_factors IS
    'Marginal emission factors by ISO/RTO region, hour, and season for accurate carbon impact calculation of DR curtailment.';
COMMENT ON TABLE pack037_demand_response.dr_event_carbon_impacts IS
    'Per-event carbon impact calculation showing emissions avoided by curtailing load using marginal emission factors.';
COMMENT ON TABLE pack037_demand_response.dr_annual_carbon_summaries IS
    'Annual aggregated carbon impact summaries per facility for sustainability reporting and GHG inventories.';
COMMENT ON TABLE pack037_demand_response.dr_carbon_reports IS
    'Carbon impact reports for ESG, sustainability, and regulatory disclosures across GHG Protocol, CDP, TCFD, CSRD, SBTi frameworks.';
COMMENT ON TABLE pack037_demand_response.dr_sbti_contributions IS
    'Science Based Targets initiative contribution tracking linking DR-avoided emissions to decarbonization pathways.';

COMMENT ON COLUMN pack037_demand_response.dr_marginal_emission_factors.marginal_co2e_kg_per_mwh IS 'Marginal CO2-equivalent emission factor in kg/MWh including CH4 and N2O GWP-weighted.';
COMMENT ON COLUMN pack037_demand_response.dr_marginal_emission_factors.marginal_fuel_type IS 'Fuel type of the marginal generator displaced by DR curtailment.';
COMMENT ON COLUMN pack037_demand_response.dr_marginal_emission_factors.methodology IS 'Data methodology: WATTTIME_MOER, EPA_EGRID, EIA_930, UNFCCC_CDM, IEA_NATIONAL, ELECTRICITY_MAP, ENTSOE.';

COMMENT ON COLUMN pack037_demand_response.dr_event_carbon_impacts.rebound_co2e_kg IS 'CO2e emissions from post-event rebound energy consumption.';
COMMENT ON COLUMN pack037_demand_response.dr_event_carbon_impacts.net_avoided_co2e_tonnes IS 'Net avoided CO2e after subtracting rebound emissions, in metric tonnes.';
COMMENT ON COLUMN pack037_demand_response.dr_event_carbon_impacts.hourly_breakdown IS 'JSONB array with per-hour curtailment MWh, marginal EF, and avoided CO2e.';
COMMENT ON COLUMN pack037_demand_response.dr_event_carbon_impacts.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack037_demand_response.dr_annual_carbon_summaries.scope2_equivalent_pct IS 'DR-avoided emissions as percentage of total Scope 2 emissions for context.';
COMMENT ON COLUMN pack037_demand_response.dr_annual_carbon_summaries.equivalent_trees_planted IS 'Equivalent number of trees planted (EPA equivalency factor: 0.06 tCO2e/tree/year).';
COMMENT ON COLUMN pack037_demand_response.dr_annual_carbon_summaries.equivalent_cars_removed IS 'Equivalent number of cars removed from roads (EPA: 4.6 tCO2e/car/year).';

COMMENT ON COLUMN pack037_demand_response.dr_sbti_contributions.dr_contribution_pct IS 'Percentage of total required emission reduction achieved through DR participation.';
COMMENT ON COLUMN pack037_demand_response.dr_sbti_contributions.pathway_alignment IS 'SBTi pathway alignment: 1_5C_ALIGNED, WB2C_ALIGNED, NOT_ALIGNED, AHEAD_OF_TARGET, BEHIND_TARGET.';
