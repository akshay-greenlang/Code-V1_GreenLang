-- =============================================================================
-- V192: PACK-032 Building Energy Assessment - Building Envelope Elements
-- =============================================================================
-- Pack:         PACK-032 (Building Energy Assessment Pack)
-- Migration:    002 of 010
-- Date:         March 2026
--
-- Creates building envelope element tables for thermal performance analysis
-- including walls, roofs, floors, windows, thermal bridges, and airtightness.
--
-- Tables (6):
--   1. pack032_building_assessment.envelope_walls
--   2. pack032_building_assessment.envelope_roofs
--   3. pack032_building_assessment.envelope_floors
--   4. pack032_building_assessment.envelope_windows
--   5. pack032_building_assessment.thermal_bridges
--   6. pack032_building_assessment.airtightness_tests
--
-- Previous: V191__pack032_building_assessment_001.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack032_building_assessment.envelope_walls
-- =============================================================================
-- Wall elements with U-value, construction details, insulation, and orientation.

CREATE TABLE pack032_building_assessment.envelope_walls (
    wall_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    wall_type               VARCHAR(100)    NOT NULL,
    construction            VARCHAR(255),
    u_value                 NUMERIC(8,4),
    area_m2                 NUMERIC(12,2),
    orientation             VARCHAR(20),
    insulation_type         VARCHAR(100),
    insulation_thickness_mm NUMERIC(8,2),
    age_band                VARCHAR(50),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_ew_u_value CHECK (
        u_value IS NULL OR u_value >= 0
    ),
    CONSTRAINT chk_p032_ew_area CHECK (
        area_m2 IS NULL OR area_m2 > 0
    ),
    CONSTRAINT chk_p032_ew_insulation_thick CHECK (
        insulation_thickness_mm IS NULL OR insulation_thickness_mm >= 0
    ),
    CONSTRAINT chk_p032_ew_wall_type CHECK (
        wall_type IN ('SOLID_BRICK', 'CAVITY', 'TIMBER_FRAME', 'STEEL_FRAME',
                       'CONCRETE', 'CURTAIN_WALL', 'CLADDING', 'STONE', 'OTHER')
    ),
    CONSTRAINT chk_p032_ew_orientation CHECK (
        orientation IS NULL OR orientation IN ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_ew_building    ON pack032_building_assessment.envelope_walls(building_id);
CREATE INDEX idx_p032_ew_tenant      ON pack032_building_assessment.envelope_walls(tenant_id);
CREATE INDEX idx_p032_ew_wall_type   ON pack032_building_assessment.envelope_walls(wall_type);
CREATE INDEX idx_p032_ew_orientation ON pack032_building_assessment.envelope_walls(orientation);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_ew_updated
    BEFORE UPDATE ON pack032_building_assessment.envelope_walls
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack032_building_assessment.envelope_roofs
-- =============================================================================
-- Roof elements with U-value, insulation type, and thermal characteristics.

CREATE TABLE pack032_building_assessment.envelope_roofs (
    roof_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    roof_type               VARCHAR(100)    NOT NULL,
    u_value                 NUMERIC(8,4),
    area_m2                 NUMERIC(12,2),
    insulation_type         VARCHAR(100),
    insulation_thickness_mm NUMERIC(8,2),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_er_u_value CHECK (
        u_value IS NULL OR u_value >= 0
    ),
    CONSTRAINT chk_p032_er_area CHECK (
        area_m2 IS NULL OR area_m2 > 0
    ),
    CONSTRAINT chk_p032_er_insulation_thick CHECK (
        insulation_thickness_mm IS NULL OR insulation_thickness_mm >= 0
    ),
    CONSTRAINT chk_p032_er_roof_type CHECK (
        roof_type IN ('FLAT', 'PITCHED', 'MANSARD', 'GREEN_ROOF', 'METAL_DECK',
                       'CONCRETE_SLAB', 'THATCHED', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_er_building    ON pack032_building_assessment.envelope_roofs(building_id);
CREATE INDEX idx_p032_er_tenant      ON pack032_building_assessment.envelope_roofs(tenant_id);
CREATE INDEX idx_p032_er_roof_type   ON pack032_building_assessment.envelope_roofs(roof_type);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_er_updated
    BEFORE UPDATE ON pack032_building_assessment.envelope_roofs
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack032_building_assessment.envelope_floors
-- =============================================================================
-- Floor elements with U-value, ground contact, and perimeter details.

CREATE TABLE pack032_building_assessment.envelope_floors (
    floor_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    floor_type              VARCHAR(100)    NOT NULL,
    u_value                 NUMERIC(8,4),
    area_m2                 NUMERIC(12,2),
    perimeter_m             NUMERIC(10,2),
    ground_contact          BOOLEAN         DEFAULT TRUE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_ef_u_value CHECK (
        u_value IS NULL OR u_value >= 0
    ),
    CONSTRAINT chk_p032_ef_area CHECK (
        area_m2 IS NULL OR area_m2 > 0
    ),
    CONSTRAINT chk_p032_ef_perimeter CHECK (
        perimeter_m IS NULL OR perimeter_m > 0
    ),
    CONSTRAINT chk_p032_ef_floor_type CHECK (
        floor_type IN ('SOLID_GROUND', 'SUSPENDED_TIMBER', 'SUSPENDED_CONCRETE',
                        'BEAM_BLOCK', 'OVER_PARKING', 'OVER_PASSAGEWAY', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_ef_building    ON pack032_building_assessment.envelope_floors(building_id);
CREATE INDEX idx_p032_ef_tenant      ON pack032_building_assessment.envelope_floors(tenant_id);
CREATE INDEX idx_p032_ef_floor_type  ON pack032_building_assessment.envelope_floors(floor_type);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_ef_updated
    BEFORE UPDATE ON pack032_building_assessment.envelope_floors
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack032_building_assessment.envelope_windows
-- =============================================================================
-- Window and glazing elements with U-value, g-value, orientation, and shading.

CREATE TABLE pack032_building_assessment.envelope_windows (
    window_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    window_type             VARCHAR(100)    NOT NULL,
    glazing_type            VARCHAR(100),
    frame_material          VARCHAR(100),
    u_value                 NUMERIC(8,4),
    g_value                 NUMERIC(6,4),
    area_m2                 NUMERIC(12,2),
    orientation             VARCHAR(20),
    shading_type            VARCHAR(100),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_ewn_u_value CHECK (
        u_value IS NULL OR u_value >= 0
    ),
    CONSTRAINT chk_p032_ewn_g_value CHECK (
        g_value IS NULL OR (g_value >= 0 AND g_value <= 1)
    ),
    CONSTRAINT chk_p032_ewn_area CHECK (
        area_m2 IS NULL OR area_m2 > 0
    ),
    CONSTRAINT chk_p032_ewn_orientation CHECK (
        orientation IS NULL OR orientation IN ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')
    ),
    CONSTRAINT chk_p032_ewn_window_type CHECK (
        window_type IN ('SINGLE_GLAZED', 'DOUBLE_GLAZED', 'TRIPLE_GLAZED', 'SECONDARY_GLAZED',
                         'ROOFLIGHT', 'CURTAIN_WALL_GLAZED', 'DOOR_GLAZED', 'OTHER')
    ),
    CONSTRAINT chk_p032_ewn_frame CHECK (
        frame_material IS NULL OR frame_material IN ('UPVC', 'TIMBER', 'ALUMINIUM',
                                                       'STEEL', 'COMPOSITE', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_ewn_building    ON pack032_building_assessment.envelope_windows(building_id);
CREATE INDEX idx_p032_ewn_tenant      ON pack032_building_assessment.envelope_windows(tenant_id);
CREATE INDEX idx_p032_ewn_window_type ON pack032_building_assessment.envelope_windows(window_type);
CREATE INDEX idx_p032_ewn_orientation ON pack032_building_assessment.envelope_windows(orientation);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_ewn_updated
    BEFORE UPDATE ON pack032_building_assessment.envelope_windows
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack032_building_assessment.thermal_bridges
-- =============================================================================
-- Thermal bridge junctions with psi-values and lengths for heat loss calculation.

CREATE TABLE pack032_building_assessment.thermal_bridges (
    bridge_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    junction_type           VARCHAR(100)    NOT NULL,
    psi_value               NUMERIC(8,4),
    length_m                NUMERIC(10,2),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_tb_psi CHECK (
        psi_value IS NULL OR psi_value >= 0
    ),
    CONSTRAINT chk_p032_tb_length CHECK (
        length_m IS NULL OR length_m > 0
    ),
    CONSTRAINT chk_p032_tb_junction_type CHECK (
        junction_type IN ('WALL_ROOF', 'WALL_FLOOR', 'WALL_WALL_CORNER', 'WINDOW_WALL',
                           'DOOR_WALL', 'LINTEL', 'SILL', 'JAMB', 'BALCONY',
                           'PARAPET', 'EAVES', 'PARTY_WALL', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_tb_building      ON pack032_building_assessment.thermal_bridges(building_id);
CREATE INDEX idx_p032_tb_tenant        ON pack032_building_assessment.thermal_bridges(tenant_id);
CREATE INDEX idx_p032_tb_junction_type ON pack032_building_assessment.thermal_bridges(junction_type);

-- =============================================================================
-- Table 6: pack032_building_assessment.airtightness_tests
-- =============================================================================
-- Blower door / airtightness test results for building infiltration assessment.

CREATE TABLE pack032_building_assessment.airtightness_tests (
    test_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    test_date               DATE            NOT NULL,
    n50_value               NUMERIC(8,4),
    q50_value               NUMERIC(8,4),
    test_standard           VARCHAR(100),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_at_n50 CHECK (
        n50_value IS NULL OR n50_value >= 0
    ),
    CONSTRAINT chk_p032_at_q50 CHECK (
        q50_value IS NULL OR q50_value >= 0
    ),
    CONSTRAINT chk_p032_at_standard CHECK (
        test_standard IS NULL OR test_standard IN ('EN_13829', 'ISO_9972', 'ATTMA_TSL1',
                                                     'ATTMA_TSL2', 'ASTM_E779', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_at_building ON pack032_building_assessment.airtightness_tests(building_id);
CREATE INDEX idx_p032_at_tenant   ON pack032_building_assessment.airtightness_tests(tenant_id);
CREATE INDEX idx_p032_at_date     ON pack032_building_assessment.airtightness_tests(test_date DESC);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack032_building_assessment.envelope_walls ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.envelope_roofs ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.envelope_floors ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.envelope_windows ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.thermal_bridges ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.airtightness_tests ENABLE ROW LEVEL SECURITY;

CREATE POLICY p032_ew_tenant_isolation
    ON pack032_building_assessment.envelope_walls
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_ew_service_bypass
    ON pack032_building_assessment.envelope_walls
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_er_tenant_isolation
    ON pack032_building_assessment.envelope_roofs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_er_service_bypass
    ON pack032_building_assessment.envelope_roofs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_ef_tenant_isolation
    ON pack032_building_assessment.envelope_floors
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_ef_service_bypass
    ON pack032_building_assessment.envelope_floors
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_ewn_tenant_isolation
    ON pack032_building_assessment.envelope_windows
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_ewn_service_bypass
    ON pack032_building_assessment.envelope_windows
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_tb_tenant_isolation
    ON pack032_building_assessment.thermal_bridges
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_tb_service_bypass
    ON pack032_building_assessment.thermal_bridges
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_at_tenant_isolation
    ON pack032_building_assessment.airtightness_tests
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_at_service_bypass
    ON pack032_building_assessment.airtightness_tests
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.envelope_walls TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.envelope_roofs TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.envelope_floors TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.envelope_windows TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.thermal_bridges TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.airtightness_tests TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack032_building_assessment.envelope_walls IS
    'Wall elements with U-value, construction details, insulation type/thickness, and orientation for heat loss calculation.';

COMMENT ON TABLE pack032_building_assessment.envelope_roofs IS
    'Roof elements with U-value, insulation type/thickness for thermal performance analysis.';

COMMENT ON TABLE pack032_building_assessment.envelope_floors IS
    'Floor elements with U-value, perimeter, and ground contact details for heat loss assessment.';

COMMENT ON TABLE pack032_building_assessment.envelope_windows IS
    'Window and glazing elements with U-value, solar heat gain coefficient (g-value), orientation, and shading for energy modelling.';

COMMENT ON TABLE pack032_building_assessment.thermal_bridges IS
    'Thermal bridge junctions with linear transmittance (psi-value) and length for supplementary heat loss calculation.';

COMMENT ON TABLE pack032_building_assessment.airtightness_tests IS
    'Blower door / airtightness test results (n50, q50) for building infiltration assessment per EN 13829 / ISO 9972.';

COMMENT ON COLUMN pack032_building_assessment.envelope_walls.u_value IS
    'Thermal transmittance in W/(m2.K).';
COMMENT ON COLUMN pack032_building_assessment.envelope_windows.g_value IS
    'Solar heat gain coefficient (total solar energy transmittance, 0-1).';
COMMENT ON COLUMN pack032_building_assessment.thermal_bridges.psi_value IS
    'Linear thermal transmittance in W/(m.K).';
COMMENT ON COLUMN pack032_building_assessment.airtightness_tests.n50_value IS
    'Air changes per hour at 50 Pa pressure difference (ACH).';
COMMENT ON COLUMN pack032_building_assessment.airtightness_tests.q50_value IS
    'Air permeability at 50 Pa in m3/(h.m2).';
