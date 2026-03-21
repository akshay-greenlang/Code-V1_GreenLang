-- =============================================================================
-- V248: PACK-033 Quick Wins Identifier - Energy Savings Estimation
-- =============================================================================
-- Pack:         PACK-033 (Quick Wins Identifier Pack)
-- Migration:    003 of 010
-- Date:         March 2026
--
-- Creates energy savings estimation tables including three-point estimates
-- (low/expected/high), climate adjustments, and interactive effects between
-- multiple actions implemented together.
--
-- Tables (2):
--   1. pack033_quick_wins.savings_estimates
--   2. pack033_quick_wins.interactive_effects
--
-- Previous: V247__pack033_quick_wins_002.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack033_quick_wins.savings_estimates
-- =============================================================================
-- Three-point energy savings estimates with climate zone adjustments,
-- operating hours, load factors, and rebound effect corrections.

CREATE TABLE pack033_quick_wins.savings_estimates (
    estimate_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id                 UUID            NOT NULL REFERENCES pack033_quick_wins.quick_wins_scans(scan_id) ON DELETE CASCADE,
    action_id               UUID,
    energy_type             VARCHAR(50)     NOT NULL,
    baseline_consumption    NUMERIC(16,2)   NOT NULL,
    estimated_savings_low   NUMERIC(14,2)   NOT NULL,
    estimated_savings_expected NUMERIC(14,2) NOT NULL,
    estimated_savings_high  NUMERIC(14,2)   NOT NULL,
    savings_unit            VARCHAR(20)     NOT NULL DEFAULT 'kWh',
    operating_hours         INTEGER,
    load_factor             NUMERIC(6,4),
    climate_zone            VARCHAR(10),
    hdd_adjustment          NUMERIC(8,4),
    cdd_adjustment          NUMERIC(8,4),
    rebound_factor          NUMERIC(6,4)    DEFAULT 1.0,
    confidence_pct          NUMERIC(5,2),
    methodology             VARCHAR(100),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p033_se_energy_type CHECK (
        energy_type IN ('ELECTRICITY', 'NATURAL_GAS', 'DISTRICT_HEATING', 'DISTRICT_COOLING',
                         'FUEL_OIL', 'LPG', 'BIOMASS', 'STEAM', 'CHILLED_WATER', 'OTHER')
    ),
    CONSTRAINT chk_p033_se_baseline CHECK (
        baseline_consumption > 0
    ),
    CONSTRAINT chk_p033_se_savings_low CHECK (
        estimated_savings_low >= 0
    ),
    CONSTRAINT chk_p033_se_savings_expected CHECK (
        estimated_savings_expected >= 0
    ),
    CONSTRAINT chk_p033_se_savings_high CHECK (
        estimated_savings_high >= 0
    ),
    CONSTRAINT chk_p033_se_savings_order CHECK (
        estimated_savings_low <= estimated_savings_expected
        AND estimated_savings_expected <= estimated_savings_high
    ),
    CONSTRAINT chk_p033_se_savings_unit CHECK (
        savings_unit IN ('kWh', 'MWh', 'GJ', 'therms', 'MMBtu', 'kWh_th')
    ),
    CONSTRAINT chk_p033_se_operating_hours CHECK (
        operating_hours IS NULL OR (operating_hours >= 0 AND operating_hours <= 8784)
    ),
    CONSTRAINT chk_p033_se_load_factor CHECK (
        load_factor IS NULL OR (load_factor > 0 AND load_factor <= 1)
    ),
    CONSTRAINT chk_p033_se_rebound CHECK (
        rebound_factor IS NULL OR (rebound_factor >= 0 AND rebound_factor <= 1)
    ),
    CONSTRAINT chk_p033_se_confidence CHECK (
        confidence_pct IS NULL OR (confidence_pct >= 0 AND confidence_pct <= 100)
    ),
    CONSTRAINT chk_p033_se_methodology CHECK (
        methodology IS NULL OR methodology IN (
            'ENGINEERING_ESTIMATE', 'BILLING_ANALYSIS', 'SIMULATION',
            'MEASUREMENT_VERIFICATION', 'DEEMED_SAVINGS', 'HYBRID'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_se_scan          ON pack033_quick_wins.savings_estimates(scan_id);
CREATE INDEX idx_p033_se_action        ON pack033_quick_wins.savings_estimates(action_id);
CREATE INDEX idx_p033_se_energy_type   ON pack033_quick_wins.savings_estimates(energy_type);
CREATE INDEX idx_p033_se_climate_zone  ON pack033_quick_wins.savings_estimates(climate_zone);
CREATE INDEX idx_p033_se_methodology   ON pack033_quick_wins.savings_estimates(methodology);
CREATE INDEX idx_p033_se_created       ON pack033_quick_wins.savings_estimates(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p033_se_updated
    BEFORE UPDATE ON pack033_quick_wins.savings_estimates
    FOR EACH ROW EXECUTE FUNCTION pack033_quick_wins.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack033_quick_wins.interactive_effects
-- =============================================================================
-- Interaction effects between co-implemented actions. Captures synergistic
-- or antagonistic savings adjustments when multiple measures are bundled.

CREATE TABLE pack033_quick_wins.interactive_effects (
    effect_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id                 UUID            NOT NULL REFERENCES pack033_quick_wins.quick_wins_scans(scan_id) ON DELETE CASCADE,
    action_a_id             UUID            NOT NULL,
    action_b_id             UUID            NOT NULL,
    interaction_type        VARCHAR(30)     NOT NULL,
    adjustment_factor       NUMERIC(6,4)    NOT NULL,
    combined_savings        NUMERIC(14,2),
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p033_ie_interaction_type CHECK (
        interaction_type IN ('SYNERGISTIC', 'ANTAGONISTIC', 'INDEPENDENT', 'SEQUENTIAL')
    ),
    CONSTRAINT chk_p033_ie_adjustment CHECK (
        adjustment_factor > 0 AND adjustment_factor <= 2.0
    ),
    CONSTRAINT chk_p033_ie_combined CHECK (
        combined_savings IS NULL OR combined_savings >= 0
    ),
    CONSTRAINT chk_p033_ie_different_actions CHECK (
        action_a_id != action_b_id
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_ie_scan          ON pack033_quick_wins.interactive_effects(scan_id);
CREATE INDEX idx_p033_ie_action_a      ON pack033_quick_wins.interactive_effects(action_a_id);
CREATE INDEX idx_p033_ie_action_b      ON pack033_quick_wins.interactive_effects(action_b_id);
CREATE INDEX idx_p033_ie_type          ON pack033_quick_wins.interactive_effects(interaction_type);
CREATE INDEX idx_p033_ie_pair          ON pack033_quick_wins.interactive_effects(action_a_id, action_b_id);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p033_ie_updated
    BEFORE UPDATE ON pack033_quick_wins.interactive_effects
    FOR EACH ROW EXECUTE FUNCTION pack033_quick_wins.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack033_quick_wins.savings_estimates ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack033_quick_wins.interactive_effects ENABLE ROW LEVEL SECURITY;

CREATE POLICY p033_se_tenant_isolation
    ON pack033_quick_wins.savings_estimates
    USING (scan_id IN (
        SELECT scan_id FROM pack033_quick_wins.quick_wins_scans
        WHERE tenant_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p033_se_service_bypass
    ON pack033_quick_wins.savings_estimates
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p033_ie_tenant_isolation
    ON pack033_quick_wins.interactive_effects
    USING (scan_id IN (
        SELECT scan_id FROM pack033_quick_wins.quick_wins_scans
        WHERE tenant_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p033_ie_service_bypass
    ON pack033_quick_wins.interactive_effects
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.savings_estimates TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.interactive_effects TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack033_quick_wins.savings_estimates IS
    'Three-point energy savings estimates with climate zone adjustments, operating hours, load factors, and rebound effect corrections.';

COMMENT ON TABLE pack033_quick_wins.interactive_effects IS
    'Interaction effects between co-implemented actions capturing synergistic or antagonistic savings adjustments.';

COMMENT ON COLUMN pack033_quick_wins.savings_estimates.rebound_factor IS
    'Rebound effect factor (0-1) reducing estimated savings; 1.0 = no rebound, 0.8 = 20% rebound.';
COMMENT ON COLUMN pack033_quick_wins.savings_estimates.hdd_adjustment IS
    'Heating Degree Day adjustment factor for climate normalization.';
COMMENT ON COLUMN pack033_quick_wins.savings_estimates.cdd_adjustment IS
    'Cooling Degree Day adjustment factor for climate normalization.';
COMMENT ON COLUMN pack033_quick_wins.savings_estimates.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack033_quick_wins.interactive_effects.adjustment_factor IS
    'Multiplier applied to combined savings (>1.0 synergistic, <1.0 antagonistic).';
COMMENT ON COLUMN pack033_quick_wins.interactive_effects.interaction_type IS
    'Type of interaction: SYNERGISTIC (>additive), ANTAGONISTIC (<additive), INDEPENDENT, SEQUENTIAL (order matters).';
