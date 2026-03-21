-- =============================================================================
-- V257: PACK-034 ISO 50001 Energy Management System - SEU Tables
-- =============================================================================
-- Pack:         PACK-034 (ISO 50001 Energy Management System Pack)
-- Migration:    002 of 010
-- Date:         March 2026
--
-- Creates Significant Energy Use (SEU) tables for identifying, categorizing,
-- and tracking major energy consumers. Tracks equipment details and energy
-- drivers that influence consumption patterns.
--
-- Tables (3):
--   1. pack034_iso50001.significant_energy_uses
--   2. pack034_iso50001.seu_equipment
--   3. pack034_iso50001.energy_drivers
--
-- Previous: V256__pack034_iso50001_001.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack034_iso50001.significant_energy_uses
-- =============================================================================
-- Significant Energy Uses as defined by ISO 50001. Identifies energy uses
-- that account for substantial consumption and/or offer significant
-- improvement potential.

CREATE TABLE pack034_iso50001.significant_energy_uses (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    enms_id                     UUID            NOT NULL REFERENCES pack034_iso50001.energy_management_systems(id) ON DELETE CASCADE,
    seu_name                    VARCHAR(500)    NOT NULL,
    seu_category                VARCHAR(30)     NOT NULL,
    annual_consumption_kwh      DECIMAL(18,4),
    percentage_of_total         DECIMAL(8,4),
    is_significant              BOOLEAN         NOT NULL DEFAULT FALSE,
    determination_method        TEXT,
    energy_driver               VARCHAR(255),
    operating_hours_annual      INTEGER,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'identified',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_seu_category CHECK (
        seu_category IN ('hvac', 'lighting', 'compressed_air', 'motors', 'process_heat',
                         'refrigeration', 'steam', 'pumps', 'other')
    ),
    CONSTRAINT chk_p034_seu_consumption CHECK (
        annual_consumption_kwh IS NULL OR annual_consumption_kwh >= 0
    ),
    CONSTRAINT chk_p034_seu_pct CHECK (
        percentage_of_total IS NULL OR (percentage_of_total >= 0 AND percentage_of_total <= 100)
    ),
    CONSTRAINT chk_p034_seu_hours CHECK (
        operating_hours_annual IS NULL OR (operating_hours_annual >= 0 AND operating_hours_annual <= 8784)
    ),
    CONSTRAINT chk_p034_seu_status CHECK (
        status IN ('identified', 'under_review', 'confirmed', 'optimized', 'decommissioned')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_seu_enms           ON pack034_iso50001.significant_energy_uses(enms_id);
CREATE INDEX idx_p034_seu_category       ON pack034_iso50001.significant_energy_uses(seu_category);
CREATE INDEX idx_p034_seu_significant    ON pack034_iso50001.significant_energy_uses(is_significant);
CREATE INDEX idx_p034_seu_consumption    ON pack034_iso50001.significant_energy_uses(annual_consumption_kwh DESC);
CREATE INDEX idx_p034_seu_pct            ON pack034_iso50001.significant_energy_uses(percentage_of_total DESC);
CREATE INDEX idx_p034_seu_status         ON pack034_iso50001.significant_energy_uses(status);
CREATE INDEX idx_p034_seu_created        ON pack034_iso50001.significant_energy_uses(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_seu_updated
    BEFORE UPDATE ON pack034_iso50001.significant_energy_uses
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack034_iso50001.seu_equipment
-- =============================================================================
-- Individual equipment items linked to a Significant Energy Use, with rated
-- power, load factor, efficiency, and replacement planning data.

CREATE TABLE pack034_iso50001.seu_equipment (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    seu_id                      UUID            NOT NULL REFERENCES pack034_iso50001.significant_energy_uses(id) ON DELETE CASCADE,
    equipment_name              VARCHAR(500)    NOT NULL,
    equipment_type              VARCHAR(255),
    rated_power_kw              DECIMAL(12,4),
    load_factor                 DECIMAL(6,4),
    efficiency                  DECIMAL(6,4),
    age_years                   INTEGER,
    replacement_due             BOOLEAN         DEFAULT FALSE,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_equip_power CHECK (
        rated_power_kw IS NULL OR rated_power_kw >= 0
    ),
    CONSTRAINT chk_p034_equip_load CHECK (
        load_factor IS NULL OR (load_factor >= 0 AND load_factor <= 1.5)
    ),
    CONSTRAINT chk_p034_equip_eff CHECK (
        efficiency IS NULL OR (efficiency >= 0 AND efficiency <= 1)
    ),
    CONSTRAINT chk_p034_equip_age CHECK (
        age_years IS NULL OR age_years >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_equip_seu          ON pack034_iso50001.seu_equipment(seu_id);
CREATE INDEX idx_p034_equip_type         ON pack034_iso50001.seu_equipment(equipment_type);
CREATE INDEX idx_p034_equip_power        ON pack034_iso50001.seu_equipment(rated_power_kw DESC);
CREATE INDEX idx_p034_equip_eff          ON pack034_iso50001.seu_equipment(efficiency);
CREATE INDEX idx_p034_equip_replace      ON pack034_iso50001.seu_equipment(replacement_due);
CREATE INDEX idx_p034_equip_created      ON pack034_iso50001.seu_equipment(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_equip_updated
    BEFORE UPDATE ON pack034_iso50001.seu_equipment
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack034_iso50001.energy_drivers
-- =============================================================================
-- Variables that significantly affect energy consumption for a given SEU.
-- Includes production volume, weather, occupancy, and other relevant
-- variables used in baseline regression models.

CREATE TABLE pack034_iso50001.energy_drivers (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    seu_id                      UUID            NOT NULL REFERENCES pack034_iso50001.significant_energy_uses(id) ON DELETE CASCADE,
    driver_name                 VARCHAR(255)    NOT NULL,
    driver_type                 VARCHAR(30)     NOT NULL,
    correlation_coefficient     DECIMAL(8,6),
    unit_of_measure             VARCHAR(50),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_driver_type CHECK (
        driver_type IN ('production_volume', 'weather', 'occupancy', 'operating_hours', 'other')
    ),
    CONSTRAINT chk_p034_driver_corr CHECK (
        correlation_coefficient IS NULL OR (correlation_coefficient >= -1 AND correlation_coefficient <= 1)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_driver_seu         ON pack034_iso50001.energy_drivers(seu_id);
CREATE INDEX idx_p034_driver_type        ON pack034_iso50001.energy_drivers(driver_type);
CREATE INDEX idx_p034_driver_corr        ON pack034_iso50001.energy_drivers(correlation_coefficient DESC);
CREATE INDEX idx_p034_driver_created     ON pack034_iso50001.energy_drivers(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_driver_updated
    BEFORE UPDATE ON pack034_iso50001.energy_drivers
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack034_iso50001.significant_energy_uses ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.seu_equipment ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.energy_drivers ENABLE ROW LEVEL SECURITY;

CREATE POLICY p034_seu_tenant_isolation
    ON pack034_iso50001.significant_energy_uses
    USING (enms_id IN (
        SELECT id FROM pack034_iso50001.energy_management_systems
        WHERE organization_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p034_seu_service_bypass
    ON pack034_iso50001.significant_energy_uses
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_equip_tenant_isolation
    ON pack034_iso50001.seu_equipment
    USING (seu_id IN (
        SELECT id FROM pack034_iso50001.significant_energy_uses
        WHERE enms_id IN (
            SELECT id FROM pack034_iso50001.energy_management_systems
            WHERE organization_id = current_setting('app.current_tenant')::UUID
        )
    ));
CREATE POLICY p034_equip_service_bypass
    ON pack034_iso50001.seu_equipment
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_driver_tenant_isolation
    ON pack034_iso50001.energy_drivers
    USING (seu_id IN (
        SELECT id FROM pack034_iso50001.significant_energy_uses
        WHERE enms_id IN (
            SELECT id FROM pack034_iso50001.energy_management_systems
            WHERE organization_id = current_setting('app.current_tenant')::UUID
        )
    ));
CREATE POLICY p034_driver_service_bypass
    ON pack034_iso50001.energy_drivers
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.significant_energy_uses TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.seu_equipment TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.energy_drivers TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack034_iso50001.significant_energy_uses IS
    'Significant Energy Uses (SEUs) as defined by ISO 50001 - energy uses accounting for substantial consumption or offering significant improvement potential.';

COMMENT ON TABLE pack034_iso50001.seu_equipment IS
    'Individual equipment items linked to SEUs with rated power, load factor, efficiency, and replacement planning data.';

COMMENT ON TABLE pack034_iso50001.energy_drivers IS
    'Variables that significantly affect energy consumption for a given SEU, used in baseline regression models.';

COMMENT ON COLUMN pack034_iso50001.significant_energy_uses.seu_category IS
    'Category of energy use: hvac, lighting, compressed_air, motors, process_heat, refrigeration, steam, pumps, or other.';
COMMENT ON COLUMN pack034_iso50001.significant_energy_uses.is_significant IS
    'Whether this energy use has been determined to be significant per ISO 50001 criteria.';
COMMENT ON COLUMN pack034_iso50001.significant_energy_uses.determination_method IS
    'Method used to determine significance (e.g., Pareto analysis, engineering judgment, regression).';
COMMENT ON COLUMN pack034_iso50001.significant_energy_uses.energy_driver IS
    'Primary variable driving energy consumption for this SEU.';
COMMENT ON COLUMN pack034_iso50001.seu_equipment.load_factor IS
    'Average operating load as fraction of rated capacity (0.0 to 1.5 for overloaded equipment).';
COMMENT ON COLUMN pack034_iso50001.seu_equipment.efficiency IS
    'Operating efficiency as a fraction (0.0 to 1.0).';
COMMENT ON COLUMN pack034_iso50001.seu_equipment.replacement_due IS
    'Whether this equipment is due for replacement based on age, efficiency, or condition.';
COMMENT ON COLUMN pack034_iso50001.energy_drivers.driver_type IS
    'Type of energy driver: production_volume, weather, occupancy, operating_hours, or other.';
COMMENT ON COLUMN pack034_iso50001.energy_drivers.correlation_coefficient IS
    'Pearson correlation coefficient between the driver and energy consumption (-1 to +1).';
