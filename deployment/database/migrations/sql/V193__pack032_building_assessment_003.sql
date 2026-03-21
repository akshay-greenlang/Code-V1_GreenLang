-- =============================================================================
-- V193: PACK-032 Building Energy Assessment - HVAC Systems
-- =============================================================================
-- Pack:         PACK-032 (Building Energy Assessment Pack)
-- Migration:    003 of 010
-- Date:         March 2026
--
-- Creates HVAC system tables for heating, cooling, ventilation, and
-- refrigerant tracking in building energy assessments.
--
-- Tables (4):
--   1. pack032_building_assessment.heating_systems
--   2. pack032_building_assessment.cooling_systems
--   3. pack032_building_assessment.ventilation_systems
--   4. pack032_building_assessment.refrigerant_records
--
-- Previous: V192__pack032_building_assessment_002.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack032_building_assessment.heating_systems
-- =============================================================================
-- Heating plant and distribution details including fuel type, efficiency,
-- controls strategy, and condensing capability.

CREATE TABLE pack032_building_assessment.heating_systems (
    system_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    system_type             VARCHAR(100)    NOT NULL,
    fuel_type               VARCHAR(100)    NOT NULL,
    rated_output_kw         NUMERIC(12,2),
    efficiency_pct          NUMERIC(6,2),
    age_years               INTEGER,
    condensing              BOOLEAN         DEFAULT FALSE,
    controls                VARCHAR(255),
    distribution_type       VARCHAR(100),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_hs_rated_output CHECK (
        rated_output_kw IS NULL OR rated_output_kw > 0
    ),
    CONSTRAINT chk_p032_hs_efficiency CHECK (
        efficiency_pct IS NULL OR (efficiency_pct > 0 AND efficiency_pct <= 500)
    ),
    CONSTRAINT chk_p032_hs_age CHECK (
        age_years IS NULL OR age_years >= 0
    ),
    CONSTRAINT chk_p032_hs_system_type CHECK (
        system_type IN ('BOILER', 'HEAT_PUMP_ASHP', 'HEAT_PUMP_GSHP', 'HEAT_PUMP_WSHP',
                         'CHP', 'DISTRICT_HEATING', 'ELECTRIC_RESISTANCE', 'BIOMASS_BOILER',
                         'INFRARED_HEATER', 'WARM_AIR', 'SOLAR_THERMAL', 'OTHER')
    ),
    CONSTRAINT chk_p032_hs_fuel_type CHECK (
        fuel_type IN ('NATURAL_GAS', 'LPG', 'OIL', 'ELECTRICITY', 'BIOMASS_PELLET',
                       'BIOMASS_CHIP', 'BIOMASS_LOG', 'COAL', 'DISTRICT_HEAT',
                       'HYDROGEN', 'BIOGAS', 'OTHER')
    ),
    CONSTRAINT chk_p032_hs_distribution CHECK (
        distribution_type IS NULL OR distribution_type IN ('RADIATORS', 'UNDERFLOOR',
                                                             'FAN_COIL', 'WARM_AIR',
                                                             'RADIANT_PANEL', 'CONVECTORS', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_hs_building    ON pack032_building_assessment.heating_systems(building_id);
CREATE INDEX idx_p032_hs_tenant      ON pack032_building_assessment.heating_systems(tenant_id);
CREATE INDEX idx_p032_hs_system_type ON pack032_building_assessment.heating_systems(system_type);
CREATE INDEX idx_p032_hs_fuel_type   ON pack032_building_assessment.heating_systems(fuel_type);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_hs_updated
    BEFORE UPDATE ON pack032_building_assessment.heating_systems
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack032_building_assessment.cooling_systems
-- =============================================================================
-- Cooling plant with SEER/EER ratings, refrigerant type, and charge details.

CREATE TABLE pack032_building_assessment.cooling_systems (
    system_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    system_type             VARCHAR(100)    NOT NULL,
    rated_capacity_kw       NUMERIC(12,2),
    seer                    NUMERIC(8,2),
    eer                     NUMERIC(8,2),
    refrigerant_type        VARCHAR(50),
    refrigerant_charge_kg   NUMERIC(10,3),
    age_years               INTEGER,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_cs_capacity CHECK (
        rated_capacity_kw IS NULL OR rated_capacity_kw > 0
    ),
    CONSTRAINT chk_p032_cs_seer CHECK (
        seer IS NULL OR seer > 0
    ),
    CONSTRAINT chk_p032_cs_eer CHECK (
        eer IS NULL OR eer > 0
    ),
    CONSTRAINT chk_p032_cs_charge CHECK (
        refrigerant_charge_kg IS NULL OR refrigerant_charge_kg >= 0
    ),
    CONSTRAINT chk_p032_cs_age CHECK (
        age_years IS NULL OR age_years >= 0
    ),
    CONSTRAINT chk_p032_cs_system_type CHECK (
        system_type IN ('SPLIT_SYSTEM', 'VRF', 'CHILLER_AIR_COOLED', 'CHILLER_WATER_COOLED',
                         'DX_PACKAGED', 'DISTRICT_COOLING', 'ABSORPTION_CHILLER',
                         'EVAPORATIVE_COOLING', 'FREE_COOLING', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_cs_building    ON pack032_building_assessment.cooling_systems(building_id);
CREATE INDEX idx_p032_cs_tenant      ON pack032_building_assessment.cooling_systems(tenant_id);
CREATE INDEX idx_p032_cs_system_type ON pack032_building_assessment.cooling_systems(system_type);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_cs_updated
    BEFORE UPDATE ON pack032_building_assessment.cooling_systems
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack032_building_assessment.ventilation_systems
-- =============================================================================
-- Ventilation system details with specific fan power (SFP) and heat recovery.

CREATE TABLE pack032_building_assessment.ventilation_systems (
    system_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    system_type             VARCHAR(100)    NOT NULL,
    supply_rate_ls          NUMERIC(12,2),
    extract_rate_ls         NUMERIC(12,2),
    sfp_w_per_ls            NUMERIC(8,4),
    heat_recovery_type      VARCHAR(100),
    heat_recovery_effectiveness_pct NUMERIC(6,2),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_vs_supply CHECK (
        supply_rate_ls IS NULL OR supply_rate_ls >= 0
    ),
    CONSTRAINT chk_p032_vs_extract CHECK (
        extract_rate_ls IS NULL OR extract_rate_ls >= 0
    ),
    CONSTRAINT chk_p032_vs_sfp CHECK (
        sfp_w_per_ls IS NULL OR sfp_w_per_ls >= 0
    ),
    CONSTRAINT chk_p032_vs_hr_eff CHECK (
        heat_recovery_effectiveness_pct IS NULL OR
        (heat_recovery_effectiveness_pct >= 0 AND heat_recovery_effectiveness_pct <= 100)
    ),
    CONSTRAINT chk_p032_vs_system_type CHECK (
        system_type IN ('NATURAL', 'MECHANICAL_EXTRACT', 'MECHANICAL_SUPPLY_EXTRACT',
                         'MVHR', 'AHU', 'MIXED_MODE', 'DEMAND_CONTROLLED', 'OTHER')
    ),
    CONSTRAINT chk_p032_vs_hr_type CHECK (
        heat_recovery_type IS NULL OR heat_recovery_type IN ('PLATE', 'ROTARY_WHEEL',
                                                               'RUN_AROUND_COIL', 'HEAT_PIPE',
                                                               'NONE', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_vs_building    ON pack032_building_assessment.ventilation_systems(building_id);
CREATE INDEX idx_p032_vs_tenant      ON pack032_building_assessment.ventilation_systems(tenant_id);
CREATE INDEX idx_p032_vs_system_type ON pack032_building_assessment.ventilation_systems(system_type);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_vs_updated
    BEFORE UPDATE ON pack032_building_assessment.ventilation_systems
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack032_building_assessment.refrigerant_records
-- =============================================================================
-- Refrigerant tracking records for F-gas regulation compliance with GWP values.

CREATE TABLE pack032_building_assessment.refrigerant_records (
    record_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    system_id               UUID            NOT NULL REFERENCES pack032_building_assessment.cooling_systems(system_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    refrigerant_type        VARCHAR(50)     NOT NULL,
    charge_kg               NUMERIC(10,3)   NOT NULL,
    gwp                     INTEGER         NOT NULL,
    annual_leak_rate_pct    NUMERIC(6,2),
    last_service_date       DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_rr_charge CHECK (
        charge_kg >= 0
    ),
    CONSTRAINT chk_p032_rr_gwp CHECK (
        gwp >= 0
    ),
    CONSTRAINT chk_p032_rr_leak_rate CHECK (
        annual_leak_rate_pct IS NULL OR (annual_leak_rate_pct >= 0 AND annual_leak_rate_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_rr_system      ON pack032_building_assessment.refrigerant_records(system_id);
CREATE INDEX idx_p032_rr_tenant      ON pack032_building_assessment.refrigerant_records(tenant_id);
CREATE INDEX idx_p032_rr_type        ON pack032_building_assessment.refrigerant_records(refrigerant_type);
CREATE INDEX idx_p032_rr_service     ON pack032_building_assessment.refrigerant_records(last_service_date DESC);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack032_building_assessment.heating_systems ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.cooling_systems ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.ventilation_systems ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.refrigerant_records ENABLE ROW LEVEL SECURITY;

CREATE POLICY p032_hs_tenant_isolation
    ON pack032_building_assessment.heating_systems
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_hs_service_bypass
    ON pack032_building_assessment.heating_systems
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_cs_tenant_isolation
    ON pack032_building_assessment.cooling_systems
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_cs_service_bypass
    ON pack032_building_assessment.cooling_systems
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_vs_tenant_isolation
    ON pack032_building_assessment.ventilation_systems
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_vs_service_bypass
    ON pack032_building_assessment.ventilation_systems
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_rr_tenant_isolation
    ON pack032_building_assessment.refrigerant_records
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_rr_service_bypass
    ON pack032_building_assessment.refrigerant_records
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.heating_systems TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.cooling_systems TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.ventilation_systems TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.refrigerant_records TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack032_building_assessment.heating_systems IS
    'Heating plant and distribution systems with fuel type, efficiency, condensing capability, and controls strategy.';

COMMENT ON TABLE pack032_building_assessment.cooling_systems IS
    'Cooling plant systems with SEER/EER ratings, refrigerant type, and charge details for F-gas compliance.';

COMMENT ON TABLE pack032_building_assessment.ventilation_systems IS
    'Ventilation systems with specific fan power (SFP), heat recovery type, and effectiveness for energy performance.';

COMMENT ON TABLE pack032_building_assessment.refrigerant_records IS
    'Refrigerant tracking records for F-gas regulation compliance with GWP values and leak rate monitoring.';

COMMENT ON COLUMN pack032_building_assessment.heating_systems.efficiency_pct IS
    'Seasonal efficiency percentage (allows >100% for heat pumps expressed as COP x 100).';
COMMENT ON COLUMN pack032_building_assessment.cooling_systems.seer IS
    'Seasonal Energy Efficiency Ratio.';
COMMENT ON COLUMN pack032_building_assessment.cooling_systems.eer IS
    'Energy Efficiency Ratio at full load.';
COMMENT ON COLUMN pack032_building_assessment.ventilation_systems.sfp_w_per_ls IS
    'Specific Fan Power in W/(l/s).';
COMMENT ON COLUMN pack032_building_assessment.refrigerant_records.gwp IS
    'Global Warming Potential (100-year, AR5 or AR6).';
