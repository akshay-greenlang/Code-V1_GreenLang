-- =============================================================================
-- V327: PACK-041 Scope 1-2 Complete Pack - Emission Factor Registry
-- =============================================================================
-- Pack:         PACK-041 (Scope 1-2 Complete Pack)
-- Migration:    002 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates emission factor management tables including a multi-source factor
-- registry, factor override mechanism with approval workflow, and GWP values
-- for all IPCC Assessment Report generations. The emission factor registry
-- supports tiered factors (Tier 1 default, Tier 2 country-specific, Tier 3
-- supplier-specific) with provenance tracking for full auditability.
--
-- Tables (3):
--   1. ghg_scope12.emission_factor_registry
--   2. ghg_scope12.factor_overrides
--   3. ghg_scope12.gwp_values
--
-- Also includes: indexes, RLS, comments, seed data for GWP values.
-- Previous: V326__pack041_core_schema.sql
-- =============================================================================

SET search_path TO ghg_scope12, public;

-- =============================================================================
-- Table 1: ghg_scope12.emission_factor_registry
-- =============================================================================
-- Centralized registry of emission factors from authoritative sources (EPA,
-- DEFRA, IPCC, IEA, national inventories). Each factor is versioned by year,
-- geography, and tier. Factors include individual GHG components (CO2, CH4,
-- N2O) and the pre-calculated CO2e using specified GWP values. Provenance
-- hash ensures factor integrity for audit and reproducibility.

CREATE TABLE ghg_scope12.emission_factor_registry (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    factor_type                 VARCHAR(50)     NOT NULL,
    fuel_type                   VARCHAR(100)    NOT NULL,
    fuel_sub_type               VARCHAR(100),
    geography                   VARCHAR(100)    NOT NULL DEFAULT 'GLOBAL',
    year                        INTEGER         NOT NULL,
    tier                        VARCHAR(20)     NOT NULL DEFAULT 'TIER_1',
    source                      VARCHAR(100)    NOT NULL,
    source_document             VARCHAR(500),
    source_table_ref            VARCHAR(100),
    co2_factor                  DECIMAL(12,6)   NOT NULL DEFAULT 0,
    ch4_factor                  DECIMAL(12,6)   NOT NULL DEFAULT 0,
    n2o_factor                  DECIMAL(12,6)   NOT NULL DEFAULT 0,
    co2e_factor                 DECIMAL(12,6)   NOT NULL DEFAULT 0,
    unit                        VARCHAR(50)     NOT NULL,
    unit_numerator              VARCHAR(30)     NOT NULL DEFAULT 'kg_CO2e',
    unit_denominator            VARCHAR(50)     NOT NULL,
    gwp_source                  VARCHAR(20)     NOT NULL DEFAULT 'AR5',
    oxidation_factor            DECIMAL(6,4)    DEFAULT 1.0000,
    heating_value               DECIMAL(12,4),
    heating_value_unit          VARCHAR(30),
    heating_value_basis         VARCHAR(10)     DEFAULT 'NCV',
    density                     DECIMAL(10,6),
    density_unit                VARCHAR(30),
    uncertainty_pct             DECIMAL(8,4),
    data_quality_score          NUMERIC(5,2),
    is_default                  BOOLEAN         NOT NULL DEFAULT false,
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    valid_from                  DATE            NOT NULL,
    valid_to                    DATE,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p041_ef_factor_type CHECK (
        factor_type IN (
            'STATIONARY_COMBUSTION', 'MOBILE_COMBUSTION', 'FUGITIVE',
            'PROCESS', 'REFRIGERANT', 'GRID_ELECTRICITY', 'STEAM',
            'COOLING', 'HEATING', 'COMBINED_HEAT_POWER', 'RESIDUAL_MIX',
            'SUPPLIER_SPECIFIC', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p041_ef_tier CHECK (
        tier IN ('TIER_1', 'TIER_2', 'TIER_3', 'DIRECT_MEASUREMENT', 'SUPPLIER_SPECIFIC')
    ),
    CONSTRAINT chk_p041_ef_source CHECK (
        source IN (
            'EPA', 'DEFRA', 'IPCC', 'IEA', 'EEA', 'AIB',
            'EGRID', 'NATIONAL_INVENTORY', 'SUPPLIER',
            'VERIFIED_CUSTOM', 'GHG_PROTOCOL', 'ECOINVENT',
            'ADEME', 'OTHER'
        )
    ),
    CONSTRAINT chk_p041_ef_gwp_source CHECK (
        gwp_source IN ('AR4', 'AR5', 'AR6', 'SAR', 'TAR')
    ),
    CONSTRAINT chk_p041_ef_year CHECK (
        year >= 1990 AND year <= 2100
    ),
    CONSTRAINT chk_p041_ef_co2_factor CHECK (
        co2_factor >= 0
    ),
    CONSTRAINT chk_p041_ef_ch4_factor CHECK (
        ch4_factor >= 0
    ),
    CONSTRAINT chk_p041_ef_n2o_factor CHECK (
        n2o_factor >= 0
    ),
    CONSTRAINT chk_p041_ef_co2e_factor CHECK (
        co2e_factor >= 0
    ),
    CONSTRAINT chk_p041_ef_oxidation CHECK (
        oxidation_factor IS NULL OR (oxidation_factor >= 0 AND oxidation_factor <= 1)
    ),
    CONSTRAINT chk_p041_ef_heating_basis CHECK (
        heating_value_basis IS NULL OR heating_value_basis IN ('NCV', 'GCV', 'HHV', 'LHV')
    ),
    CONSTRAINT chk_p041_ef_uncertainty CHECK (
        uncertainty_pct IS NULL OR (uncertainty_pct >= 0 AND uncertainty_pct <= 200)
    ),
    CONSTRAINT chk_p041_ef_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p041_ef_valid_dates CHECK (
        valid_to IS NULL OR valid_from <= valid_to
    ),
    CONSTRAINT uq_p041_ef_composite UNIQUE (fuel_type, geography, year, source, tier, tenant_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_ef_tenant             ON ghg_scope12.emission_factor_registry(tenant_id);
CREATE INDEX idx_p041_ef_factor_type        ON ghg_scope12.emission_factor_registry(factor_type);
CREATE INDEX idx_p041_ef_fuel_type          ON ghg_scope12.emission_factor_registry(fuel_type);
CREATE INDEX idx_p041_ef_geography          ON ghg_scope12.emission_factor_registry(geography);
CREATE INDEX idx_p041_ef_year               ON ghg_scope12.emission_factor_registry(year);
CREATE INDEX idx_p041_ef_tier               ON ghg_scope12.emission_factor_registry(tier);
CREATE INDEX idx_p041_ef_source             ON ghg_scope12.emission_factor_registry(source);
CREATE INDEX idx_p041_ef_gwp_source         ON ghg_scope12.emission_factor_registry(gwp_source);
CREATE INDEX idx_p041_ef_default            ON ghg_scope12.emission_factor_registry(is_default) WHERE is_default = true;
CREATE INDEX idx_p041_ef_active             ON ghg_scope12.emission_factor_registry(is_active) WHERE is_active = true;
CREATE INDEX idx_p041_ef_valid_from         ON ghg_scope12.emission_factor_registry(valid_from);
CREATE INDEX idx_p041_ef_created            ON ghg_scope12.emission_factor_registry(created_at DESC);
CREATE INDEX idx_p041_ef_metadata           ON ghg_scope12.emission_factor_registry USING GIN(metadata);

-- Composite: common lookup pattern (fuel + geography + year + active)
CREATE INDEX idx_p041_ef_lookup             ON ghg_scope12.emission_factor_registry(fuel_type, geography, year)
    WHERE is_active = true;

-- Composite: factor type + tier for methodology selection
CREATE INDEX idx_p041_ef_type_tier          ON ghg_scope12.emission_factor_registry(factor_type, tier, year DESC)
    WHERE is_active = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_ef_updated
    BEFORE UPDATE ON ghg_scope12.emission_factor_registry
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_scope12.factor_overrides
-- =============================================================================
-- Allows organizations to override default emission factors with custom or
-- supplier-specific values. Each override requires justification and approval
-- for audit compliance. Overrides are tracked with provenance and effective
-- dates to support reproducibility of historical calculations.

CREATE TABLE ghg_scope12.factor_overrides (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    original_factor_id          UUID            NOT NULL REFERENCES ghg_scope12.emission_factor_registry(id) ON DELETE CASCADE,
    organization_id             UUID            REFERENCES ghg_scope12.organizations(id) ON DELETE SET NULL,
    facility_id                 UUID            REFERENCES ghg_scope12.facilities(id) ON DELETE SET NULL,
    override_co2_factor         DECIMAL(12,6),
    override_ch4_factor         DECIMAL(12,6),
    override_n2o_factor         DECIMAL(12,6),
    override_co2e_factor        DECIMAL(12,6),
    override_reason             VARCHAR(50)     NOT NULL,
    justification               TEXT            NOT NULL,
    supporting_document         VARCHAR(500),
    status                      VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    effective_date              DATE            NOT NULL,
    expiry_date                 DATE,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p041_fo_reason CHECK (
        override_reason IN (
            'SUPPLIER_SPECIFIC', 'SITE_MEASUREMENT', 'VERIFIED_CUSTOM',
            'REGULATORY_REQUIREMENT', 'METHODOLOGY_UPDATE', 'CORRECTION',
            'HIGHER_TIER', 'OTHER'
        )
    ),
    CONSTRAINT chk_p041_fo_status CHECK (
        status IN ('PENDING', 'APPROVED', 'REJECTED', 'EXPIRED', 'SUPERSEDED')
    ),
    CONSTRAINT chk_p041_fo_co2 CHECK (
        override_co2_factor IS NULL OR override_co2_factor >= 0
    ),
    CONSTRAINT chk_p041_fo_ch4 CHECK (
        override_ch4_factor IS NULL OR override_ch4_factor >= 0
    ),
    CONSTRAINT chk_p041_fo_n2o CHECK (
        override_n2o_factor IS NULL OR override_n2o_factor >= 0
    ),
    CONSTRAINT chk_p041_fo_co2e CHECK (
        override_co2e_factor IS NULL OR override_co2e_factor >= 0
    ),
    CONSTRAINT chk_p041_fo_dates CHECK (
        expiry_date IS NULL OR effective_date <= expiry_date
    ),
    CONSTRAINT chk_p041_fo_has_override CHECK (
        override_co2_factor IS NOT NULL OR override_ch4_factor IS NOT NULL OR
        override_n2o_factor IS NOT NULL OR override_co2e_factor IS NOT NULL
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_fo_tenant             ON ghg_scope12.factor_overrides(tenant_id);
CREATE INDEX idx_p041_fo_original           ON ghg_scope12.factor_overrides(original_factor_id);
CREATE INDEX idx_p041_fo_org               ON ghg_scope12.factor_overrides(organization_id);
CREATE INDEX idx_p041_fo_facility           ON ghg_scope12.factor_overrides(facility_id);
CREATE INDEX idx_p041_fo_status             ON ghg_scope12.factor_overrides(status);
CREATE INDEX idx_p041_fo_reason             ON ghg_scope12.factor_overrides(override_reason);
CREATE INDEX idx_p041_fo_effective          ON ghg_scope12.factor_overrides(effective_date);
CREATE INDEX idx_p041_fo_created            ON ghg_scope12.factor_overrides(created_at DESC);

-- Composite: active overrides for a given original factor
CREATE INDEX idx_p041_fo_active_overrides   ON ghg_scope12.factor_overrides(original_factor_id, effective_date)
    WHERE status = 'APPROVED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_fo_updated
    BEFORE UPDATE ON ghg_scope12.factor_overrides
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_scope12.gwp_values
-- =============================================================================
-- Global Warming Potential (GWP) values for greenhouse gases across IPCC
-- Assessment Report generations. Used to convert individual gas masses to
-- CO2-equivalent. Contains AR4, AR5, and AR6 values for all seven Kyoto
-- Protocol gases plus key HFCs/PFCs.

CREATE TABLE ghg_scope12.gwp_values (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    gas_name                    VARCHAR(100)    NOT NULL,
    gas_formula                 VARCHAR(30)     NOT NULL,
    cas_number                  VARCHAR(20),
    gas_category                VARCHAR(30)     NOT NULL DEFAULT 'KYOTO',
    ar4_value                   DECIMAL(10,2),
    ar5_value                   DECIMAL(10,2),
    ar6_value                   DECIMAL(10,2),
    sar_value                   DECIMAL(10,2),
    tar_value                   DECIMAL(10,2),
    atmospheric_lifetime_years  DECIMAL(10,2),
    notes                       TEXT,
    source_reference            VARCHAR(500),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p041_gwp_category CHECK (
        gas_category IN ('KYOTO', 'HFC', 'PFC', 'OTHER_FLUORINATED', 'OZONE_DEPLETING', 'OTHER')
    ),
    CONSTRAINT chk_p041_gwp_ar4 CHECK (
        ar4_value IS NULL OR ar4_value >= 0
    ),
    CONSTRAINT chk_p041_gwp_ar5 CHECK (
        ar5_value IS NULL OR ar5_value >= 0
    ),
    CONSTRAINT chk_p041_gwp_ar6 CHECK (
        ar6_value IS NULL OR ar6_value >= 0
    ),
    CONSTRAINT uq_p041_gwp_formula UNIQUE (gas_formula)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_gwp_name              ON ghg_scope12.gwp_values(gas_name);
CREATE INDEX idx_p041_gwp_formula           ON ghg_scope12.gwp_values(gas_formula);
CREATE INDEX idx_p041_gwp_category          ON ghg_scope12.gwp_values(gas_category);
CREATE INDEX idx_p041_gwp_cas               ON ghg_scope12.gwp_values(cas_number);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_gwp_updated
    BEFORE UPDATE ON ghg_scope12.gwp_values
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_scope12.emission_factor_registry ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.factor_overrides ENABLE ROW LEVEL SECURITY;
-- gwp_values is reference data shared across tenants; no RLS needed

CREATE POLICY p041_ef_tenant_isolation
    ON ghg_scope12.emission_factor_registry
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_ef_service_bypass
    ON ghg_scope12.emission_factor_registry
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_fo_tenant_isolation
    ON ghg_scope12.factor_overrides
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_fo_service_bypass
    ON ghg_scope12.factor_overrides
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.emission_factor_registry TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.factor_overrides TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.gwp_values TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Seed Data: GWP Values (100-year time horizon)
-- ---------------------------------------------------------------------------
-- Sources: IPCC AR4 (2007), AR5 (2014), AR6 (2021)

INSERT INTO ghg_scope12.gwp_values (gas_name, gas_formula, cas_number, gas_category, ar4_value, ar5_value, ar6_value, sar_value, tar_value, atmospheric_lifetime_years, source_reference) VALUES
    ('Carbon dioxide',         'CO2',    '124-38-9',   'KYOTO',  1,      1,      1,      1,      1,      NULL,   'IPCC AR6 WG1 Table 7.SM.7'),
    ('Methane',                'CH4',    '74-82-8',    'KYOTO',  25,     28,     27.9,   21,     23,     11.8,   'IPCC AR6 WG1 Table 7.SM.7'),
    ('Nitrous oxide',          'N2O',    '10024-97-2', 'KYOTO',  298,    265,    273,    310,    296,    109,    'IPCC AR6 WG1 Table 7.SM.7'),
    ('Sulfur hexafluoride',    'SF6',    '2551-62-4',  'KYOTO',  22800,  23500,  25200,  23900,  22200,  3200,   'IPCC AR6 WG1 Table 7.SM.7'),
    ('Nitrogen trifluoride',   'NF3',    '7783-54-2',  'KYOTO',  17200,  16100,  17400,  NULL,   NULL,   500,    'IPCC AR6 WG1 Table 7.SM.7'),
    ('HFC-134a',               'CH2FCF3','811-97-2',   'HFC',    1430,   1300,   1530,   1300,   1300,   14,     'IPCC AR6 WG1 Table 7.SM.7'),
    ('HFC-32',                 'CH2F2',  '75-10-5',    'HFC',    675,    677,    771,    650,    550,    5.4,    'IPCC AR6 WG1 Table 7.SM.7'),
    ('HFC-125',                'CHF2CF3','354-33-6',   'HFC',    3500,   3170,   3740,   2800,   3400,   30,     'IPCC AR6 WG1 Table 7.SM.7'),
    ('HFC-143a',               'CH3CF3', '420-46-2',   'HFC',    4470,   4800,   5810,   3800,   4300,   51,     'IPCC AR6 WG1 Table 7.SM.7'),
    ('HFC-152a',               'CH3CHF2','75-37-6',    'HFC',    124,    138,    164,    140,    120,    1.6,    'IPCC AR6 WG1 Table 7.SM.7'),
    ('HFC-227ea',              'CF3CHFCF3','431-89-0', 'HFC',    3220,   3350,   3600,   2900,   3500,   36,     'IPCC AR6 WG1 Table 7.SM.7'),
    ('HFC-236fa',              'CF3CH2CF3','690-39-1',  'HFC',    9810,   8060,   8690,   6300,   9400,   213,    'IPCC AR6 WG1 Table 7.SM.7'),
    ('HFC-245fa',              'CHF2CH2CF3','460-73-1', 'HFC',    1030,   858,    962,    NULL,   NULL,   7.9,    'IPCC AR6 WG1 Table 7.SM.7'),
    ('HFC-365mfc',             'CH3CF2CH2CF3','406-58-6','HFC',   794,    804,    914,    NULL,   NULL,   8.7,    'IPCC AR6 WG1 Table 7.SM.7'),
    ('HFC-23',                 'CHF3',   '75-46-7',    'HFC',    14800,  12400,  14600,  11700,  12000,  228,    'IPCC AR6 WG1 Table 7.SM.7'),
    ('HFC-43-10mee',           'CF3CHFCHFCF2CF3','138495-42-8','HFC', 1640, 1650, 1600, NULL, NULL, 17, 'IPCC AR6 WG1 Table 7.SM.7'),
    ('PFC-14 (CF4)',           'CF4',    '75-73-0',    'PFC',    7390,   6630,   7380,   6500,   5700,   50000,  'IPCC AR6 WG1 Table 7.SM.7'),
    ('PFC-116 (C2F6)',         'C2F6',   '76-16-4',    'PFC',    12200,  11100,  12400,  9200,   11900,  10000,  'IPCC AR6 WG1 Table 7.SM.7'),
    ('PFC-218 (C3F8)',         'C3F8',   '76-19-7',    'PFC',    8830,   8900,   9290,   7000,   8600,   2600,   'IPCC AR6 WG1 Table 7.SM.7'),
    ('PFC-318 (c-C4F8)',       'c-C4F8', '115-25-3',   'PFC',    10300,  9540,   10200,  8700,   10000,  3200,   'IPCC AR6 WG1 Table 7.SM.7');

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_scope12.emission_factor_registry IS
    'Centralized emission factor registry supporting multi-source, multi-tier factors with CO2/CH4/N2O components, GWP references, and provenance tracking.';
COMMENT ON TABLE ghg_scope12.factor_overrides IS
    'Organization/facility-level emission factor overrides with approval workflow, justification, and effective date tracking.';
COMMENT ON TABLE ghg_scope12.gwp_values IS
    'IPCC Global Warming Potential values across assessment report generations (SAR, TAR, AR4, AR5, AR6) for all Kyoto Protocol gases and key HFCs/PFCs.';

COMMENT ON COLUMN ghg_scope12.emission_factor_registry.id IS 'Unique identifier for the emission factor.';
COMMENT ON COLUMN ghg_scope12.emission_factor_registry.tenant_id IS 'Multi-tenant isolation key.';
COMMENT ON COLUMN ghg_scope12.emission_factor_registry.factor_type IS 'Emission source category: STATIONARY_COMBUSTION, GRID_ELECTRICITY, REFRIGERANT, etc.';
COMMENT ON COLUMN ghg_scope12.emission_factor_registry.fuel_type IS 'Fuel or energy carrier name (e.g., NATURAL_GAS, DIESEL, GRID_AVERAGE).';
COMMENT ON COLUMN ghg_scope12.emission_factor_registry.tier IS 'Methodology tier: TIER_1 (default), TIER_2 (country-specific), TIER_3 (facility-specific).';
COMMENT ON COLUMN ghg_scope12.emission_factor_registry.co2_factor IS 'CO2 emission factor component in unit_numerator per unit_denominator.';
COMMENT ON COLUMN ghg_scope12.emission_factor_registry.co2e_factor IS 'Total CO2-equivalent factor including all GHG components using specified GWP source.';
COMMENT ON COLUMN ghg_scope12.emission_factor_registry.gwp_source IS 'IPCC Assessment Report used for GWP conversion: AR4, AR5, AR6, SAR, TAR.';
COMMENT ON COLUMN ghg_scope12.emission_factor_registry.oxidation_factor IS 'Fraction of carbon oxidized during combustion (0-1, default 1.0).';
COMMENT ON COLUMN ghg_scope12.emission_factor_registry.heating_value_basis IS 'Heating value basis: NCV (net calorific value) or GCV (gross calorific value).';
COMMENT ON COLUMN ghg_scope12.emission_factor_registry.provenance_hash IS 'SHA-256 hash of source data for factor integrity verification.';

COMMENT ON COLUMN ghg_scope12.factor_overrides.override_reason IS 'Reason for override: SUPPLIER_SPECIFIC, SITE_MEASUREMENT, VERIFIED_CUSTOM, etc.';
COMMENT ON COLUMN ghg_scope12.factor_overrides.status IS 'Approval status: PENDING, APPROVED, REJECTED, EXPIRED, SUPERSEDED.';

COMMENT ON COLUMN ghg_scope12.gwp_values.gas_formula IS 'Chemical formula of the greenhouse gas (unique identifier).';
COMMENT ON COLUMN ghg_scope12.gwp_values.ar4_value IS 'GWP from IPCC Fourth Assessment Report (2007), 100-year time horizon.';
COMMENT ON COLUMN ghg_scope12.gwp_values.ar5_value IS 'GWP from IPCC Fifth Assessment Report (2014), 100-year time horizon.';
COMMENT ON COLUMN ghg_scope12.gwp_values.ar6_value IS 'GWP from IPCC Sixth Assessment Report (2021), 100-year time horizon.';
COMMENT ON COLUMN ghg_scope12.gwp_values.atmospheric_lifetime_years IS 'Atmospheric lifetime in years. NULL for CO2 (complex decay).';
