-- =============================================================================
-- V367: PACK-045 Base Year Management Pack - Base Year Inventory
-- =============================================================================
-- Pack:         PACK-045 (Base Year Management Pack)
-- Migration:    002 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates the base year inventory table that stores the detailed emission line
-- items composing the base year total. Each row represents a single emission
-- source within the base year, including activity data, emission factor,
-- calculated tCO2e, gas type, GWP version, methodology tier, and data quality
-- score. This table provides the granular data needed for recalculation when
-- structural or methodological changes occur.
--
-- Tables (1):
--   1. ghg_base_year.gl_by_inventories
--
-- Also includes: indexes, RLS, comments.
-- Previous: V366__pack045_core_schema.sql
-- =============================================================================

SET search_path TO ghg_base_year, public;

-- =============================================================================
-- Table 1: ghg_base_year.gl_by_inventories
-- =============================================================================
-- Granular inventory line items for a base year. Each row is one emission
-- source (e.g., natural gas consumption at Facility A, grid electricity at
-- Facility B). Stores the activity data quantity, emission factor used,
-- resulting tCO2e, gas breakdown, GWP version, methodology tier, and a
-- data quality score per GHG Protocol data quality indicators.

CREATE TABLE ghg_base_year.gl_by_inventories (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    base_year_id                UUID            NOT NULL REFERENCES ghg_base_year.gl_by_base_years(id) ON DELETE CASCADE,
    scope                       VARCHAR(10)     NOT NULL,
    category                    VARCHAR(60)     NOT NULL,
    sub_category                VARCHAR(60),
    source_id                   UUID,
    source_description          VARCHAR(500),
    facility_id                 UUID,
    facility_name               VARCHAR(500),
    activity_data               NUMERIC(18,6)   NOT NULL,
    activity_data_unit          VARCHAR(50)     NOT NULL,
    emission_factor             NUMERIC(18,10)  NOT NULL,
    emission_factor_unit        VARCHAR(100)    NOT NULL,
    emission_factor_source      VARCHAR(100),
    emission_factor_year        INTEGER,
    tco2e                       NUMERIC(14,3)   NOT NULL,
    co2_tonnes                  NUMERIC(14,3),
    ch4_tonnes                  NUMERIC(14,6),
    n2o_tonnes                  NUMERIC(14,6),
    hfc_tonnes                  NUMERIC(14,6),
    pfc_tonnes                  NUMERIC(14,6),
    sf6_tonnes                  NUMERIC(14,6),
    nf3_tonnes                  NUMERIC(14,6),
    gas_type                    VARCHAR(30)     DEFAULT 'CO2',
    gwp_version                 VARCHAR(10)     DEFAULT 'AR5',
    gwp_value                   NUMERIC(10,2),
    methodology_tier            VARCHAR(20)     DEFAULT 'TIER_1',
    data_quality_score          NUMERIC(5,2),
    uncertainty_pct             NUMERIC(6,2),
    is_estimated                BOOLEAN         NOT NULL DEFAULT false,
    estimation_method           VARCHAR(60),
    is_biogenic                 BOOLEAN         NOT NULL DEFAULT false,
    country                     VARCHAR(3),
    region                      VARCHAR(100),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p045_inv_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2_LOCATION', 'SCOPE_2_MARKET', 'SCOPE_3')
    ),
    CONSTRAINT chk_p045_inv_category CHECK (
        category IN (
            'STATIONARY_COMBUSTION', 'MOBILE_COMBUSTION', 'PROCESS_EMISSIONS',
            'FUGITIVE_EMISSIONS', 'REFRIGERANT_LEAKAGE', 'LAND_USE_CHANGE',
            'WASTE_TREATMENT', 'AGRICULTURAL_EMISSIONS',
            'PURCHASED_ELECTRICITY', 'PURCHASED_STEAM',
            'PURCHASED_COOLING', 'PURCHASED_HEATING',
            'SCOPE3_CAT1', 'SCOPE3_CAT2', 'SCOPE3_CAT3', 'SCOPE3_CAT4',
            'SCOPE3_CAT5', 'SCOPE3_CAT6', 'SCOPE3_CAT7', 'SCOPE3_CAT8',
            'SCOPE3_CAT9', 'SCOPE3_CAT10', 'SCOPE3_CAT11', 'SCOPE3_CAT12',
            'SCOPE3_CAT13', 'SCOPE3_CAT14', 'SCOPE3_CAT15',
            'OTHER'
        )
    ),
    CONSTRAINT chk_p045_inv_activity CHECK (
        activity_data >= 0
    ),
    CONSTRAINT chk_p045_inv_ef CHECK (
        emission_factor >= 0
    ),
    CONSTRAINT chk_p045_inv_tco2e CHECK (
        tco2e >= 0
    ),
    CONSTRAINT chk_p045_inv_gas_type CHECK (
        gas_type IN ('CO2', 'CH4', 'N2O', 'HFC', 'PFC', 'SF6', 'NF3', 'MIXED')
    ),
    CONSTRAINT chk_p045_inv_gwp CHECK (
        gwp_version IN ('SAR', 'TAR', 'AR4', 'AR5', 'AR6')
    ),
    CONSTRAINT chk_p045_inv_tier CHECK (
        methodology_tier IS NULL OR methodology_tier IN (
            'TIER_1', 'TIER_2', 'TIER_3', 'DIRECT_MEASUREMENT', 'HYBRID'
        )
    ),
    CONSTRAINT chk_p045_inv_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p045_inv_uncertainty CHECK (
        uncertainty_pct IS NULL OR (uncertainty_pct >= 0 AND uncertainty_pct <= 200)
    ),
    CONSTRAINT chk_p045_inv_ef_year CHECK (
        emission_factor_year IS NULL OR (emission_factor_year >= 1990 AND emission_factor_year <= 2100)
    ),
    CONSTRAINT chk_p045_inv_country CHECK (
        country IS NULL OR LENGTH(country) BETWEEN 2 AND 3
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p045_inv_tenant         ON ghg_base_year.gl_by_inventories(tenant_id);
CREATE INDEX idx_p045_inv_base_year      ON ghg_base_year.gl_by_inventories(base_year_id);
CREATE INDEX idx_p045_inv_scope          ON ghg_base_year.gl_by_inventories(scope);
CREATE INDEX idx_p045_inv_category       ON ghg_base_year.gl_by_inventories(category);
CREATE INDEX idx_p045_inv_source         ON ghg_base_year.gl_by_inventories(source_id);
CREATE INDEX idx_p045_inv_facility       ON ghg_base_year.gl_by_inventories(facility_id);
CREATE INDEX idx_p045_inv_gas_type       ON ghg_base_year.gl_by_inventories(gas_type);
CREATE INDEX idx_p045_inv_tier           ON ghg_base_year.gl_by_inventories(methodology_tier);
CREATE INDEX idx_p045_inv_quality        ON ghg_base_year.gl_by_inventories(data_quality_score);
CREATE INDEX idx_p045_inv_estimated      ON ghg_base_year.gl_by_inventories(is_estimated) WHERE is_estimated = true;
CREATE INDEX idx_p045_inv_biogenic       ON ghg_base_year.gl_by_inventories(is_biogenic) WHERE is_biogenic = true;
CREATE INDEX idx_p045_inv_country        ON ghg_base_year.gl_by_inventories(country);
CREATE INDEX idx_p045_inv_created        ON ghg_base_year.gl_by_inventories(created_at DESC);
CREATE INDEX idx_p045_inv_metadata       ON ghg_base_year.gl_by_inventories USING GIN(metadata);
CREATE INDEX idx_p045_inv_provenance     ON ghg_base_year.gl_by_inventories(provenance_hash);

-- Composite: base_year + scope + category for aggregation queries
CREATE INDEX idx_p045_inv_by_scope_cat   ON ghg_base_year.gl_by_inventories(base_year_id, scope, category);

-- Composite: base_year + facility for facility-level rollup
CREATE INDEX idx_p045_inv_by_facility    ON ghg_base_year.gl_by_inventories(base_year_id, facility_id);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p045_inv_updated
    BEFORE UPDATE ON ghg_base_year.gl_by_inventories
    FOR EACH ROW EXECUTE FUNCTION ghg_base_year.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_base_year.gl_by_inventories ENABLE ROW LEVEL SECURITY;

CREATE POLICY p045_inv_tenant_isolation
    ON ghg_base_year.gl_by_inventories
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p045_inv_service_bypass
    ON ghg_base_year.gl_by_inventories
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_base_year.gl_by_inventories TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_base_year.gl_by_inventories IS
    'Granular base year inventory line items with activity data, emission factors, gas breakdown, methodology tier, and data quality scores per GHG Protocol.';

COMMENT ON COLUMN ghg_base_year.gl_by_inventories.activity_data IS 'Quantity of activity (e.g., litres of fuel, kWh of electricity, tonnes of material).';
COMMENT ON COLUMN ghg_base_year.gl_by_inventories.emission_factor IS 'Emission factor applied (e.g., kgCO2e per kWh, tCO2e per litre).';
COMMENT ON COLUMN ghg_base_year.gl_by_inventories.tco2e IS 'Calculated total CO2 equivalent emissions for this line item.';
COMMENT ON COLUMN ghg_base_year.gl_by_inventories.data_quality_score IS 'Data quality indicator (0-100) per GHG Protocol guidance Chapter 7.';
COMMENT ON COLUMN ghg_base_year.gl_by_inventories.methodology_tier IS 'GHG Protocol methodology tier: TIER_1 (default factors), TIER_2 (country/supplier), TIER_3 (facility measured).';
COMMENT ON COLUMN ghg_base_year.gl_by_inventories.provenance_hash IS 'SHA-256 hash linking to source data for audit trail.';
