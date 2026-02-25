-- ==========================================================================
-- V061: Scope 2 Market-Based Emissions Service Schema
-- AGENT-MRV-010 (GL-MRV-SCOPE2-010)
--
-- Tables: 14 (s2m_ prefix)
-- Hypertables: 3
-- Continuous Aggregates: 2
-- Indexes: 40+
--
-- GHG Protocol Scope 2 Guidance (2015) -- Market-based method
-- Supports contractual instruments (RECs, GOs, I-RECs, PPAs),
-- supplier-specific factors, residual mix factors, energy source factors,
-- dual reporting with location-based, and GHG Protocol Scope 2 Quality
-- Criteria assessment.
--
-- Author: GreenLang Platform Team
-- Date: February 2026
-- Previous: V060__scope2_location_based_service.sql
-- ==========================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS scope2_market_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ==========================================================================
-- Function: Auto-update updated_at timestamp
-- ==========================================================================

CREATE OR REPLACE FUNCTION scope2_market_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- ==========================================================================
-- Table 1: s2m_facilities -- Facility registration with grid region
-- ==========================================================================
CREATE TABLE IF NOT EXISTS scope2_market_service.s2m_facilities (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID            NOT NULL,
    name            VARCHAR(255)    NOT NULL,
    facility_type   VARCHAR(50)     NOT NULL DEFAULT 'office',
    country_code    VARCHAR(3)      NOT NULL,
    grid_region     VARCHAR(100),
    latitude        DECIMAL(10, 7),
    longitude       DECIMAL(10, 7),
    is_active       BOOLEAN         NOT NULL DEFAULT TRUE,
    metadata        JSONB           DEFAULT '{}',
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE scope2_market_service.s2m_facilities IS 'Scope 2 market-based facility registration with grid region assignment';
COMMENT ON COLUMN scope2_market_service.s2m_facilities.facility_type IS 'Type: office, warehouse, manufacturing, retail, data_center, hospital, school, other';

CREATE TRIGGER trg_s2m_facilities_updated_at
    BEFORE UPDATE ON scope2_market_service.s2m_facilities
    FOR EACH ROW EXECUTE FUNCTION scope2_market_service.set_updated_at();

-- ==========================================================================
-- Table 2: s2m_instruments -- Contractual instruments (RECs, GOs, PPAs)
-- ==========================================================================
CREATE TABLE IF NOT EXISTS scope2_market_service.s2m_instruments (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    instrument_type     VARCHAR(50)     NOT NULL,
    energy_source       VARCHAR(50),
    quantity_mwh        DECIMAL(18, 6),
    emission_factor_kgco2e DECIMAL(18, 8),
    vintage_year        INTEGER,
    tracking_system     VARCHAR(100),
    certificate_id      VARCHAR(200)    UNIQUE,
    status              VARCHAR(30)     NOT NULL DEFAULT 'active',
    region              VARCHAR(100),
    supplier_id         VARCHAR(100),
    verified            BOOLEAN         NOT NULL DEFAULT TRUE,
    delivery_start      DATE,
    delivery_end        DATE,
    retired_at          TIMESTAMPTZ,
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_s2m_instruments_type CHECK (instrument_type IN (
        'rec', 'go', 'i-rec', 'tigr', 'ppa', 'vppa', 'green_tariff',
        'direct_contract', 'unbundled_rec', 'bundled_rec', 'other'
    )),
    CONSTRAINT chk_s2m_instruments_status CHECK (status IN (
        'active', 'allocated', 'retired', 'expired', 'cancelled'
    ))
);

COMMENT ON TABLE scope2_market_service.s2m_instruments IS 'Contractual instruments: RECs, Guarantees of Origin, I-RECs, PPAs, green tariffs';
COMMENT ON COLUMN scope2_market_service.s2m_instruments.instrument_type IS 'Type: rec, go, i-rec, tigr, ppa, vppa, green_tariff, direct_contract, unbundled_rec, bundled_rec, other';
COMMENT ON COLUMN scope2_market_service.s2m_instruments.tracking_system IS 'Registry: M-RETS, NAR, NEPOOL-GIS, WREGIS, ERCOT, PJM-GATS, AIB, I-REC Standard';
COMMENT ON COLUMN scope2_market_service.s2m_instruments.status IS 'Lifecycle: active, allocated, retired, expired, cancelled';

CREATE TRIGGER trg_s2m_instruments_updated_at
    BEFORE UPDATE ON scope2_market_service.s2m_instruments
    FOR EACH ROW EXECUTE FUNCTION scope2_market_service.set_updated_at();

-- ==========================================================================
-- Table 3: s2m_instrument_allocations -- Allocation of instruments to calcs
-- ==========================================================================
CREATE TABLE IF NOT EXISTS scope2_market_service.s2m_instrument_allocations (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    instrument_id       UUID            NOT NULL REFERENCES scope2_market_service.s2m_instruments(id),
    purchase_id         UUID,
    calculation_id      UUID,
    allocated_mwh       DECIMAL(18, 6)  NOT NULL,
    priority_level      INTEGER         NOT NULL DEFAULT 1,
    allocation_method   VARCHAR(50)     DEFAULT 'fifo',
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_s2m_alloc_method CHECK (allocation_method IN (
        'fifo', 'lifo', 'pro_rata', 'priority', 'manual'
    ))
);

COMMENT ON TABLE scope2_market_service.s2m_instrument_allocations IS 'Allocation of contractual instruments to energy purchases and calculations';
COMMENT ON COLUMN scope2_market_service.s2m_instrument_allocations.allocation_method IS 'Method: fifo, lifo, pro_rata, priority, manual';

-- ==========================================================================
-- Table 4: s2m_residual_mix_factors -- Grid residual mix emission factors
-- ==========================================================================
CREATE TABLE IF NOT EXISTS scope2_market_service.s2m_residual_mix_factors (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    region              VARCHAR(100)    NOT NULL,
    factor_kgco2e_kwh   DECIMAL(18, 8)  NOT NULL,
    source              VARCHAR(100),
    year                INTEGER,
    country_code        VARCHAR(2),
    is_custom           BOOLEAN         NOT NULL DEFAULT FALSE,
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE(region, year, source)
);

COMMENT ON TABLE scope2_market_service.s2m_residual_mix_factors IS 'Grid residual mix emission factors after contractual instruments are removed from the grid average';
COMMENT ON COLUMN scope2_market_service.s2m_residual_mix_factors.factor_kgco2e_kwh IS 'Residual mix factor in kg CO2e per kWh';

CREATE TRIGGER trg_s2m_residual_mix_updated_at
    BEFORE UPDATE ON scope2_market_service.s2m_residual_mix_factors
    FOR EACH ROW EXECUTE FUNCTION scope2_market_service.set_updated_at();

-- ==========================================================================
-- Table 5: s2m_supplier_factors -- Supplier-specific emission factors
-- ==========================================================================
CREATE TABLE IF NOT EXISTS scope2_market_service.s2m_supplier_factors (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    supplier_id         VARCHAR(100)    NOT NULL,
    supplier_name       VARCHAR(255),
    country_code        VARCHAR(2),
    ef_kgco2e_kwh       DECIMAL(18, 8),
    fuel_mix            JSONB           DEFAULT '{}',
    year                INTEGER,
    verified            BOOLEAN         NOT NULL DEFAULT FALSE,
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, supplier_id, year)
);

COMMENT ON TABLE scope2_market_service.s2m_supplier_factors IS 'Supplier-specific emission factors with fuel mix disclosure';
COMMENT ON COLUMN scope2_market_service.s2m_supplier_factors.fuel_mix IS 'JSON fuel mix breakdown e.g. {"coal": 0.30, "gas": 0.25, "wind": 0.20, "solar": 0.15, "nuclear": 0.10}';

CREATE TRIGGER trg_s2m_supplier_factors_updated_at
    BEFORE UPDATE ON scope2_market_service.s2m_supplier_factors
    FOR EACH ROW EXECUTE FUNCTION scope2_market_service.set_updated_at();

-- ==========================================================================
-- Table 6: s2m_energy_source_factors -- Energy source emission factors
-- ==========================================================================
CREATE TABLE IF NOT EXISTS scope2_market_service.s2m_energy_source_factors (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    source              VARCHAR(50)     NOT NULL UNIQUE,
    ef_kgco2e_kwh       DECIMAL(18, 8),
    is_renewable        BOOLEAN         NOT NULL DEFAULT FALSE,
    is_biogenic         BOOLEAN         NOT NULL DEFAULT FALSE,
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE scope2_market_service.s2m_energy_source_factors IS 'Energy source emission factors (solar, wind, coal, gas, nuclear, etc.)';
COMMENT ON COLUMN scope2_market_service.s2m_energy_source_factors.is_renewable IS 'True if energy source qualifies as renewable under GHG Protocol';

-- ==========================================================================
-- Table 7: s2m_energy_purchases -- Energy purchase records
-- ==========================================================================
CREATE TABLE IF NOT EXISTS scope2_market_service.s2m_energy_purchases (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    facility_id         UUID            REFERENCES scope2_market_service.s2m_facilities(id),
    quantity            DECIMAL(18, 6)  NOT NULL,
    unit                VARCHAR(20)     NOT NULL DEFAULT 'mwh',
    region              VARCHAR(100),
    supplier_id         VARCHAR(100),
    period_start        TIMESTAMPTZ,
    period_end          TIMESTAMPTZ,
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_s2m_purchase_unit CHECK (unit IN ('kwh', 'mwh', 'gwh', 'gj', 'mmbtu', 'therms'))
);

COMMENT ON TABLE scope2_market_service.s2m_energy_purchases IS 'Energy purchase records linked to facilities and suppliers';

-- ==========================================================================
-- Table 8: s2m_calculations -- Market-based calculation results
-- ==========================================================================
CREATE TABLE IF NOT EXISTS scope2_market_service.s2m_calculations (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    facility_id                 UUID,
    total_mwh                   DECIMAL(18, 6),
    covered_mwh                 DECIMAL(18, 6),
    uncovered_mwh               DECIMAL(18, 6),
    coverage_pct                DECIMAL(8, 4),
    covered_emissions_tco2e     DECIMAL(18, 6),
    uncovered_emissions_tco2e   DECIMAL(18, 6),
    total_emissions_tco2e       DECIMAL(18, 6)  NOT NULL,
    gwp_source                  VARCHAR(20)     DEFAULT 'AR5',
    calculation_method          VARCHAR(50)     DEFAULT 'market_based',
    provenance_hash             VARCHAR(64)     NOT NULL,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_s2m_calc_gwp CHECK (gwp_source IN ('AR4', 'AR5', 'AR6', 'AR6_20YR'))
);

COMMENT ON TABLE scope2_market_service.s2m_calculations IS 'Scope 2 market-based calculation results with covered/uncovered split and provenance';
COMMENT ON COLUMN scope2_market_service.s2m_calculations.coverage_pct IS 'Percentage of energy covered by contractual instruments (0-100)';
COMMENT ON COLUMN scope2_market_service.s2m_calculations.provenance_hash IS 'SHA-256 hash of complete calculation provenance chain';

-- ==========================================================================
-- Table 9: s2m_calculation_details -- Per-instrument/per-gas breakdown
-- ==========================================================================
CREATE TABLE IF NOT EXISTS scope2_market_service.s2m_calculation_details (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id      UUID            NOT NULL REFERENCES scope2_market_service.s2m_calculations(id),
    instrument_id       UUID,
    instrument_type     VARCHAR(50),
    energy_source       VARCHAR(50),
    mwh_allocated       DECIMAL(18, 6),
    ef_kgco2e           DECIMAL(18, 8),
    emissions_kg        DECIMAL(18, 6),
    co2e_kg             DECIMAL(18, 6),
    gas                 VARCHAR(10),
    is_covered          BOOLEAN         NOT NULL DEFAULT TRUE,
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE scope2_market_service.s2m_calculation_details IS 'Per-instrument and per-gas breakdown of market-based calculations';
COMMENT ON COLUMN scope2_market_service.s2m_calculation_details.is_covered IS 'True if covered by contractual instrument, false if residual mix';

-- ==========================================================================
-- Table 10: s2m_dual_reporting -- Location-based vs market-based comparison
-- ==========================================================================
CREATE TABLE IF NOT EXISTS scope2_market_service.s2m_dual_reporting (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID,
    location_based_tco2e    DECIMAL(18, 6),
    market_based_tco2e      DECIMAL(18, 6),
    difference_tco2e        DECIMAL(18, 6),
    difference_pct          DECIMAL(8, 4),
    location_calc_id        UUID,
    market_calc_id          UUID,
    reporting_period        VARCHAR(20),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE scope2_market_service.s2m_dual_reporting IS 'GHG Protocol dual reporting: location-based vs market-based comparison';
COMMENT ON COLUMN scope2_market_service.s2m_dual_reporting.difference_pct IS 'Percentage difference = (location - market) / location * 100';

-- ==========================================================================
-- Table 11: s2m_compliance_records -- Multi-framework compliance checks
-- ==========================================================================
CREATE TABLE IF NOT EXISTS scope2_market_service.s2m_compliance_records (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    calculation_id      UUID            REFERENCES scope2_market_service.s2m_calculations(id),
    framework           VARCHAR(50)     NOT NULL,
    status              VARCHAR(20)     NOT NULL,
    findings            JSONB           DEFAULT '[]',
    score               DECIMAL(5, 2),
    provenance_hash     VARCHAR(64),
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_s2m_compl_framework CHECK (framework IN (
        'ghg_protocol_scope2', 'ghg_protocol_quality_criteria', 'iso_14064',
        'csrd_esrs', 're100', 'cdp', 'tcfd', 'sec_climate'
    )),
    CONSTRAINT chk_s2m_compl_status CHECK (status IN (
        'compliant', 'non_compliant', 'partial', 'not_assessed'
    ))
);

COMMENT ON TABLE scope2_market_service.s2m_compliance_records IS 'Multi-framework regulatory compliance check results for market-based calculations';
COMMENT ON COLUMN scope2_market_service.s2m_compliance_records.framework IS 'Framework: ghg_protocol_scope2, ghg_protocol_quality_criteria, iso_14064, csrd_esrs, re100, cdp, tcfd, sec_climate';

-- ==========================================================================
-- Table 12: s2m_certificate_retirements -- Instrument retirement records
-- ==========================================================================
CREATE TABLE IF NOT EXISTS scope2_market_service.s2m_certificate_retirements (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    instrument_id       UUID            NOT NULL REFERENCES scope2_market_service.s2m_instruments(id),
    retirement_date     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    calculation_id      UUID,
    retired_by          VARCHAR(100),
    notes               TEXT,
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE scope2_market_service.s2m_certificate_retirements IS 'Certificate/instrument retirement records for double-counting prevention';

-- ==========================================================================
-- Table 13: s2m_quality_assessments -- GHG Protocol Scope 2 Quality Criteria
-- ==========================================================================
CREATE TABLE IF NOT EXISTS scope2_market_service.s2m_quality_assessments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    instrument_id           UUID            REFERENCES scope2_market_service.s2m_instruments(id),
    unique_claim            BOOLEAN         NOT NULL DEFAULT FALSE,
    associated_delivery     BOOLEAN         NOT NULL DEFAULT FALSE,
    temporal_match          BOOLEAN         NOT NULL DEFAULT FALSE,
    geographic_match        BOOLEAN         NOT NULL DEFAULT FALSE,
    no_double_count         BOOLEAN         NOT NULL DEFAULT FALSE,
    recognized_registry     BOOLEAN         NOT NULL DEFAULT FALSE,
    represents_generation   BOOLEAN         NOT NULL DEFAULT FALSE,
    overall_score           DECIMAL(5, 4),
    passes_threshold        BOOLEAN         NOT NULL DEFAULT FALSE,
    assessed_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE scope2_market_service.s2m_quality_assessments IS 'GHG Protocol Scope 2 Quality Criteria assessment for contractual instruments';
COMMENT ON COLUMN scope2_market_service.s2m_quality_assessments.unique_claim IS 'Criterion 1: Only one party claims the attributes';
COMMENT ON COLUMN scope2_market_service.s2m_quality_assessments.associated_delivery IS 'Criterion 2: Associated with energy delivery to the reporter';
COMMENT ON COLUMN scope2_market_service.s2m_quality_assessments.temporal_match IS 'Criterion 3: Generated within the same reporting period';
COMMENT ON COLUMN scope2_market_service.s2m_quality_assessments.geographic_match IS 'Criterion 4: From the same market as consumption';
COMMENT ON COLUMN scope2_market_service.s2m_quality_assessments.no_double_count IS 'Criterion 5: Not double-counted across reporters';
COMMENT ON COLUMN scope2_market_service.s2m_quality_assessments.recognized_registry IS 'Criterion 6: Tracked in recognized tracking system';
COMMENT ON COLUMN scope2_market_service.s2m_quality_assessments.represents_generation IS 'Criterion 7: Represents actual electricity generation';

-- ==========================================================================
-- Table 14: s2m_audit_entries -- Provenance/audit trail
-- ==========================================================================
CREATE TABLE IF NOT EXISTS scope2_market_service.s2m_audit_entries (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    entity_type         VARCHAR(50)     NOT NULL,
    entity_id           UUID            NOT NULL,
    action              VARCHAR(50)     NOT NULL,
    previous_hash       VARCHAR(64),
    current_hash        VARCHAR(64)     NOT NULL,
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_s2m_audit_action CHECK (action IN (
        'CREATE', 'UPDATE', 'DELETE', 'CALCULATE', 'ALLOCATE',
        'RETIRE', 'VALIDATE', 'ASSESS', 'IMPORT', 'EXPORT',
        'DUAL_REPORT', 'COMPLIANCE_CHECK'
    ))
);

COMMENT ON TABLE scope2_market_service.s2m_audit_entries IS 'SHA-256 provenance chain audit trail for market-based calculation reproducibility';

-- ==========================================================================
-- Hypertable 1: s2m_calculation_events -- Calculation time-series
-- ==========================================================================
CREATE TABLE IF NOT EXISTS scope2_market_service.s2m_calculation_events (
    event_id            UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    calculation_id      UUID            NOT NULL,
    instrument_type     VARCHAR(50),
    emissions_tco2e     DECIMAL(18, 6)  NOT NULL,
    coverage_pct        DECIMAL(8, 4),
    event_time          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'scope2_market_service.s2m_calculation_events',
    'event_time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

COMMENT ON TABLE scope2_market_service.s2m_calculation_events IS 'Time-series of Scope 2 market-based calculation events';

-- ==========================================================================
-- Hypertable 2: s2m_instrument_events -- Instrument lifecycle time-series
-- ==========================================================================
CREATE TABLE IF NOT EXISTS scope2_market_service.s2m_instrument_events (
    event_id            UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    instrument_id       UUID            NOT NULL,
    instrument_type     VARCHAR(50)     NOT NULL,
    event_type          VARCHAR(50)     NOT NULL,
    quantity_mwh        DECIMAL(18, 6),
    event_time          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'scope2_market_service.s2m_instrument_events',
    'event_time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

COMMENT ON TABLE scope2_market_service.s2m_instrument_events IS 'Time-series of instrument lifecycle events (created, allocated, retired)';

-- ==========================================================================
-- Hypertable 3: s2m_compliance_events -- Compliance time-series
-- ==========================================================================
CREATE TABLE IF NOT EXISTS scope2_market_service.s2m_compliance_events (
    event_id            UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    calculation_id      UUID            NOT NULL,
    framework           VARCHAR(50)     NOT NULL,
    status              VARCHAR(20)     NOT NULL,
    score               DECIMAL(5, 2),
    event_time          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'scope2_market_service.s2m_compliance_events',
    'event_time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

COMMENT ON TABLE scope2_market_service.s2m_compliance_events IS 'Time-series of compliance check events';

-- ==========================================================================
-- Continuous Aggregate 1: Hourly calculation statistics
-- ==========================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS scope2_market_service.s2m_hourly_calculation_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', event_time) AS bucket,
    tenant_id,
    instrument_type,
    COUNT(*) AS calculation_count,
    SUM(emissions_tco2e) AS total_emissions_tco2e,
    AVG(emissions_tco2e) AS avg_emissions_tco2e,
    AVG(coverage_pct) AS avg_coverage_pct,
    MIN(emissions_tco2e) AS min_emissions_tco2e,
    MAX(emissions_tco2e) AS max_emissions_tco2e
FROM scope2_market_service.s2m_calculation_events
GROUP BY bucket, tenant_id, instrument_type
WITH NO DATA;

-- ==========================================================================
-- Continuous Aggregate 2: Daily emission totals
-- ==========================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS scope2_market_service.s2m_daily_emission_totals
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', event_time) AS bucket,
    tenant_id,
    instrument_type,
    COUNT(*) AS calculation_count,
    SUM(emissions_tco2e) AS total_emissions_tco2e,
    AVG(coverage_pct) AS avg_coverage_pct,
    MIN(emissions_tco2e) AS min_emissions_tco2e,
    MAX(emissions_tco2e) AS max_emissions_tco2e
FROM scope2_market_service.s2m_calculation_events
GROUP BY bucket, tenant_id, instrument_type
WITH NO DATA;

-- ==========================================================================
-- Indexes: Core tables (40+)
-- ==========================================================================

-- s2m_facilities (6 indexes)
CREATE INDEX IF NOT EXISTS idx_s2m_facilities_tenant ON scope2_market_service.s2m_facilities(tenant_id);
CREATE INDEX IF NOT EXISTS idx_s2m_facilities_country ON scope2_market_service.s2m_facilities(country_code);
CREATE INDEX IF NOT EXISTS idx_s2m_facilities_region ON scope2_market_service.s2m_facilities(grid_region);
CREATE INDEX IF NOT EXISTS idx_s2m_facilities_type ON scope2_market_service.s2m_facilities(facility_type);
CREATE INDEX IF NOT EXISTS idx_s2m_facilities_active ON scope2_market_service.s2m_facilities(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_s2m_facilities_meta ON scope2_market_service.s2m_facilities USING GIN(metadata);

-- s2m_instruments (10 indexes)
CREATE INDEX IF NOT EXISTS idx_s2m_instruments_tenant ON scope2_market_service.s2m_instruments(tenant_id);
CREATE INDEX IF NOT EXISTS idx_s2m_instruments_type ON scope2_market_service.s2m_instruments(instrument_type);
CREATE INDEX IF NOT EXISTS idx_s2m_instruments_status ON scope2_market_service.s2m_instruments(status);
CREATE INDEX IF NOT EXISTS idx_s2m_instruments_source ON scope2_market_service.s2m_instruments(energy_source);
CREATE INDEX IF NOT EXISTS idx_s2m_instruments_region ON scope2_market_service.s2m_instruments(region);
CREATE INDEX IF NOT EXISTS idx_s2m_instruments_supplier ON scope2_market_service.s2m_instruments(supplier_id);
CREATE INDEX IF NOT EXISTS idx_s2m_instruments_vintage ON scope2_market_service.s2m_instruments(vintage_year);
CREATE INDEX IF NOT EXISTS idx_s2m_instruments_verified ON scope2_market_service.s2m_instruments(verified) WHERE verified = TRUE;
CREATE INDEX IF NOT EXISTS idx_s2m_instruments_active ON scope2_market_service.s2m_instruments(status) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_s2m_instruments_meta ON scope2_market_service.s2m_instruments USING GIN(metadata);

-- s2m_instrument_allocations (4 indexes)
CREATE INDEX IF NOT EXISTS idx_s2m_alloc_tenant ON scope2_market_service.s2m_instrument_allocations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_s2m_alloc_instrument ON scope2_market_service.s2m_instrument_allocations(instrument_id);
CREATE INDEX IF NOT EXISTS idx_s2m_alloc_calculation ON scope2_market_service.s2m_instrument_allocations(calculation_id);
CREATE INDEX IF NOT EXISTS idx_s2m_alloc_purchase ON scope2_market_service.s2m_instrument_allocations(purchase_id);

-- s2m_residual_mix_factors (3 indexes)
CREATE INDEX IF NOT EXISTS idx_s2m_rmf_region ON scope2_market_service.s2m_residual_mix_factors(region);
CREATE INDEX IF NOT EXISTS idx_s2m_rmf_country ON scope2_market_service.s2m_residual_mix_factors(country_code);
CREATE INDEX IF NOT EXISTS idx_s2m_rmf_year ON scope2_market_service.s2m_residual_mix_factors(year);

-- s2m_supplier_factors (4 indexes)
CREATE INDEX IF NOT EXISTS idx_s2m_sf_tenant ON scope2_market_service.s2m_supplier_factors(tenant_id);
CREATE INDEX IF NOT EXISTS idx_s2m_sf_supplier ON scope2_market_service.s2m_supplier_factors(supplier_id);
CREATE INDEX IF NOT EXISTS idx_s2m_sf_country ON scope2_market_service.s2m_supplier_factors(country_code);
CREATE INDEX IF NOT EXISTS idx_s2m_sf_year ON scope2_market_service.s2m_supplier_factors(year);

-- s2m_energy_purchases (5 indexes)
CREATE INDEX IF NOT EXISTS idx_s2m_purchases_tenant ON scope2_market_service.s2m_energy_purchases(tenant_id);
CREATE INDEX IF NOT EXISTS idx_s2m_purchases_facility ON scope2_market_service.s2m_energy_purchases(facility_id);
CREATE INDEX IF NOT EXISTS idx_s2m_purchases_supplier ON scope2_market_service.s2m_energy_purchases(supplier_id);
CREATE INDEX IF NOT EXISTS idx_s2m_purchases_region ON scope2_market_service.s2m_energy_purchases(region);
CREATE INDEX IF NOT EXISTS idx_s2m_purchases_period ON scope2_market_service.s2m_energy_purchases(period_start, period_end);

-- s2m_calculations (5 indexes)
CREATE INDEX IF NOT EXISTS idx_s2m_calc_tenant ON scope2_market_service.s2m_calculations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_s2m_calc_facility ON scope2_market_service.s2m_calculations(facility_id);
CREATE INDEX IF NOT EXISTS idx_s2m_calc_method ON scope2_market_service.s2m_calculations(calculation_method);
CREATE INDEX IF NOT EXISTS idx_s2m_calc_provenance ON scope2_market_service.s2m_calculations(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_s2m_calc_meta ON scope2_market_service.s2m_calculations USING GIN(metadata);

-- s2m_calculation_details (4 indexes)
CREATE INDEX IF NOT EXISTS idx_s2m_details_calc ON scope2_market_service.s2m_calculation_details(calculation_id);
CREATE INDEX IF NOT EXISTS idx_s2m_details_instrument ON scope2_market_service.s2m_calculation_details(instrument_id);
CREATE INDEX IF NOT EXISTS idx_s2m_details_type ON scope2_market_service.s2m_calculation_details(instrument_type);
CREATE INDEX IF NOT EXISTS idx_s2m_details_covered ON scope2_market_service.s2m_calculation_details(is_covered);

-- s2m_dual_reporting (3 indexes)
CREATE INDEX IF NOT EXISTS idx_s2m_dual_tenant ON scope2_market_service.s2m_dual_reporting(tenant_id);
CREATE INDEX IF NOT EXISTS idx_s2m_dual_facility ON scope2_market_service.s2m_dual_reporting(facility_id);
CREATE INDEX IF NOT EXISTS idx_s2m_dual_period ON scope2_market_service.s2m_dual_reporting(reporting_period);

-- s2m_compliance_records (4 indexes)
CREATE INDEX IF NOT EXISTS idx_s2m_compliance_tenant ON scope2_market_service.s2m_compliance_records(tenant_id);
CREATE INDEX IF NOT EXISTS idx_s2m_compliance_calc ON scope2_market_service.s2m_compliance_records(calculation_id);
CREATE INDEX IF NOT EXISTS idx_s2m_compliance_framework ON scope2_market_service.s2m_compliance_records(framework);
CREATE INDEX IF NOT EXISTS idx_s2m_compliance_status ON scope2_market_service.s2m_compliance_records(status);

-- s2m_certificate_retirements (3 indexes)
CREATE INDEX IF NOT EXISTS idx_s2m_retire_tenant ON scope2_market_service.s2m_certificate_retirements(tenant_id);
CREATE INDEX IF NOT EXISTS idx_s2m_retire_instrument ON scope2_market_service.s2m_certificate_retirements(instrument_id);
CREATE INDEX IF NOT EXISTS idx_s2m_retire_date ON scope2_market_service.s2m_certificate_retirements(retirement_date);

-- s2m_quality_assessments (3 indexes)
CREATE INDEX IF NOT EXISTS idx_s2m_qa_tenant ON scope2_market_service.s2m_quality_assessments(tenant_id);
CREATE INDEX IF NOT EXISTS idx_s2m_qa_instrument ON scope2_market_service.s2m_quality_assessments(instrument_id);
CREATE INDEX IF NOT EXISTS idx_s2m_qa_passes ON scope2_market_service.s2m_quality_assessments(passes_threshold) WHERE passes_threshold = TRUE;

-- s2m_audit_entries (4 indexes)
CREATE INDEX IF NOT EXISTS idx_s2m_audit_tenant ON scope2_market_service.s2m_audit_entries(tenant_id);
CREATE INDEX IF NOT EXISTS idx_s2m_audit_entity ON scope2_market_service.s2m_audit_entries(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_s2m_audit_action ON scope2_market_service.s2m_audit_entries(action);
CREATE INDEX IF NOT EXISTS idx_s2m_audit_date ON scope2_market_service.s2m_audit_entries(created_at);

-- ==========================================================================
-- Hypertable indexes
-- ==========================================================================
CREATE INDEX IF NOT EXISTS idx_s2m_calc_events_tenant ON scope2_market_service.s2m_calculation_events(tenant_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_s2m_calc_events_type ON scope2_market_service.s2m_calculation_events(instrument_type, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_s2m_calc_events_calc ON scope2_market_service.s2m_calculation_events(calculation_id, event_time DESC);

CREATE INDEX IF NOT EXISTS idx_s2m_instr_events_tenant ON scope2_market_service.s2m_instrument_events(tenant_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_s2m_instr_events_type ON scope2_market_service.s2m_instrument_events(instrument_type, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_s2m_instr_events_instr ON scope2_market_service.s2m_instrument_events(instrument_id, event_time DESC);

CREATE INDEX IF NOT EXISTS idx_s2m_comp_events_tenant ON scope2_market_service.s2m_compliance_events(tenant_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_s2m_comp_events_framework ON scope2_market_service.s2m_compliance_events(framework, event_time DESC);

-- ==========================================================================
-- Continuous Aggregate Refresh Policies
-- ==========================================================================
SELECT add_continuous_aggregate_policy('scope2_market_service.s2m_hourly_calculation_stats',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

SELECT add_continuous_aggregate_policy('scope2_market_service.s2m_daily_emission_totals',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- ==========================================================================
-- Retention Policies (5 years)
-- ==========================================================================
SELECT add_retention_policy('scope2_market_service.s2m_calculation_events', INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('scope2_market_service.s2m_instrument_events', INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('scope2_market_service.s2m_compliance_events', INTERVAL '5 years', if_not_exists => TRUE);

-- ==========================================================================
-- Compression Policies (30 days)
-- ==========================================================================
ALTER TABLE scope2_market_service.s2m_calculation_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'instrument_type',
         timescaledb.compress_orderby   = 'event_time DESC');
SELECT add_compression_policy('scope2_market_service.s2m_calculation_events', INTERVAL '30 days');

ALTER TABLE scope2_market_service.s2m_instrument_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'instrument_type',
         timescaledb.compress_orderby   = 'event_time DESC');
SELECT add_compression_policy('scope2_market_service.s2m_instrument_events', INTERVAL '30 days');

ALTER TABLE scope2_market_service.s2m_compliance_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'framework',
         timescaledb.compress_orderby   = 'event_time DESC');
SELECT add_compression_policy('scope2_market_service.s2m_compliance_events', INTERVAL '30 days');

-- ==========================================================================
-- Row-Level Security
-- ==========================================================================

-- s2m_facilities: tenant-isolated
ALTER TABLE scope2_market_service.s2m_facilities ENABLE ROW LEVEL SECURITY;
CREATE POLICY s2m_fac_read  ON scope2_market_service.s2m_facilities FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY s2m_fac_write ON scope2_market_service.s2m_facilities FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- s2m_instruments: tenant-isolated
ALTER TABLE scope2_market_service.s2m_instruments ENABLE ROW LEVEL SECURITY;
CREATE POLICY s2m_inst_read  ON scope2_market_service.s2m_instruments FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY s2m_inst_write ON scope2_market_service.s2m_instruments FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- s2m_instrument_allocations: tenant-isolated
ALTER TABLE scope2_market_service.s2m_instrument_allocations ENABLE ROW LEVEL SECURITY;
CREATE POLICY s2m_alloc_read  ON scope2_market_service.s2m_instrument_allocations FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY s2m_alloc_write ON scope2_market_service.s2m_instrument_allocations FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- s2m_residual_mix_factors: shared reference data (open read, admin write)
ALTER TABLE scope2_market_service.s2m_residual_mix_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY s2m_rmf_read  ON scope2_market_service.s2m_residual_mix_factors FOR SELECT USING (TRUE);
CREATE POLICY s2m_rmf_write ON scope2_market_service.s2m_residual_mix_factors FOR ALL USING (
    current_setting('app.is_admin', true) = 'true'
);

-- s2m_supplier_factors: tenant-isolated
ALTER TABLE scope2_market_service.s2m_supplier_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY s2m_sf_read  ON scope2_market_service.s2m_supplier_factors FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY s2m_sf_write ON scope2_market_service.s2m_supplier_factors FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- s2m_energy_source_factors: shared reference data (open read, admin write)
ALTER TABLE scope2_market_service.s2m_energy_source_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY s2m_esf_read  ON scope2_market_service.s2m_energy_source_factors FOR SELECT USING (TRUE);
CREATE POLICY s2m_esf_write ON scope2_market_service.s2m_energy_source_factors FOR ALL USING (
    current_setting('app.is_admin', true) = 'true'
);

-- s2m_energy_purchases: tenant-isolated
ALTER TABLE scope2_market_service.s2m_energy_purchases ENABLE ROW LEVEL SECURITY;
CREATE POLICY s2m_ep_read  ON scope2_market_service.s2m_energy_purchases FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY s2m_ep_write ON scope2_market_service.s2m_energy_purchases FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- s2m_calculations: tenant-isolated
ALTER TABLE scope2_market_service.s2m_calculations ENABLE ROW LEVEL SECURITY;
CREATE POLICY s2m_calc_read  ON scope2_market_service.s2m_calculations FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY s2m_calc_write ON scope2_market_service.s2m_calculations FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- s2m_calculation_details: open read/write (linked via FK to tenant-isolated s2m_calculations)
ALTER TABLE scope2_market_service.s2m_calculation_details ENABLE ROW LEVEL SECURITY;
CREATE POLICY s2m_cd_read  ON scope2_market_service.s2m_calculation_details FOR SELECT USING (TRUE);
CREATE POLICY s2m_cd_write ON scope2_market_service.s2m_calculation_details FOR ALL   USING (TRUE);

-- s2m_dual_reporting: tenant-isolated
ALTER TABLE scope2_market_service.s2m_dual_reporting ENABLE ROW LEVEL SECURITY;
CREATE POLICY s2m_dual_read  ON scope2_market_service.s2m_dual_reporting FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY s2m_dual_write ON scope2_market_service.s2m_dual_reporting FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- s2m_compliance_records: tenant-isolated
ALTER TABLE scope2_market_service.s2m_compliance_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY s2m_cr_read  ON scope2_market_service.s2m_compliance_records FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY s2m_cr_write ON scope2_market_service.s2m_compliance_records FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- s2m_certificate_retirements: tenant-isolated
ALTER TABLE scope2_market_service.s2m_certificate_retirements ENABLE ROW LEVEL SECURITY;
CREATE POLICY s2m_ret_read  ON scope2_market_service.s2m_certificate_retirements FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY s2m_ret_write ON scope2_market_service.s2m_certificate_retirements FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- s2m_quality_assessments: tenant-isolated
ALTER TABLE scope2_market_service.s2m_quality_assessments ENABLE ROW LEVEL SECURITY;
CREATE POLICY s2m_qa_read  ON scope2_market_service.s2m_quality_assessments FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY s2m_qa_write ON scope2_market_service.s2m_quality_assessments FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- s2m_audit_entries: tenant-isolated
ALTER TABLE scope2_market_service.s2m_audit_entries ENABLE ROW LEVEL SECURITY;
CREATE POLICY s2m_ae_read  ON scope2_market_service.s2m_audit_entries FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY s2m_ae_write ON scope2_market_service.s2m_audit_entries FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- s2m_calculation_events: open read/write (time-series telemetry)
ALTER TABLE scope2_market_service.s2m_calculation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY s2m_cae_read  ON scope2_market_service.s2m_calculation_events FOR SELECT USING (TRUE);
CREATE POLICY s2m_cae_write ON scope2_market_service.s2m_calculation_events FOR ALL   USING (TRUE);

-- s2m_instrument_events: open read/write (time-series telemetry)
ALTER TABLE scope2_market_service.s2m_instrument_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY s2m_ine_read  ON scope2_market_service.s2m_instrument_events FOR SELECT USING (TRUE);
CREATE POLICY s2m_ine_write ON scope2_market_service.s2m_instrument_events FOR ALL   USING (TRUE);

-- s2m_compliance_events: open read/write (time-series telemetry)
ALTER TABLE scope2_market_service.s2m_compliance_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY s2m_coe_read  ON scope2_market_service.s2m_compliance_events FOR SELECT USING (TRUE);
CREATE POLICY s2m_coe_write ON scope2_market_service.s2m_compliance_events FOR ALL   USING (TRUE);

-- ==========================================================================
-- Permissions
-- ==========================================================================

GRANT USAGE ON SCHEMA scope2_market_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA scope2_market_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA scope2_market_service TO greenlang_app;
GRANT SELECT ON scope2_market_service.s2m_hourly_calculation_stats TO greenlang_app;
GRANT SELECT ON scope2_market_service.s2m_daily_emission_totals TO greenlang_app;

GRANT USAGE ON SCHEMA scope2_market_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA scope2_market_service TO greenlang_readonly;
GRANT SELECT ON scope2_market_service.s2m_hourly_calculation_stats TO greenlang_readonly;
GRANT SELECT ON scope2_market_service.s2m_daily_emission_totals TO greenlang_readonly;

GRANT ALL ON SCHEMA scope2_market_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA scope2_market_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA scope2_market_service TO greenlang_admin;

INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'scope2-market:read',                  'scope2-market', 'read',                  'View all Scope 2 market-based emissions data including instruments, calculations, dual reporting, and compliance records'),
    (gen_random_uuid(), 'scope2-market:write',                 'scope2-market', 'write',                 'Create, update, and manage Scope 2 market-based emissions data'),
    (gen_random_uuid(), 'scope2-market:execute',               'scope2-market', 'execute',               'Execute market-based emission calculations with full audit trail and provenance'),
    (gen_random_uuid(), 'scope2-market:instruments:read',      'scope2-market', 'instruments_read',      'View contractual instruments (RECs, GOs, I-RECs, PPAs) including certificate IDs, tracking systems, vintages, and verification status'),
    (gen_random_uuid(), 'scope2-market:instruments:write',     'scope2-market', 'instruments_write',     'Create, update, retire, and manage contractual instrument records'),
    (gen_random_uuid(), 'scope2-market:allocations:read',      'scope2-market', 'allocations_read',      'View instrument allocation records with priority levels and allocation methods'),
    (gen_random_uuid(), 'scope2-market:allocations:write',     'scope2-market', 'allocations_write',     'Create and manage instrument allocation to energy purchases'),
    (gen_random_uuid(), 'scope2-market:factors:read',          'scope2-market', 'factors_read',          'View residual mix factors, supplier-specific factors, and energy source factors'),
    (gen_random_uuid(), 'scope2-market:factors:write',         'scope2-market', 'factors_write',         'Create and manage emission factor entries for market-based calculations'),
    (gen_random_uuid(), 'scope2-market:calculations:read',     'scope2-market', 'calculations_read',     'View market-based calculation results with covered/uncovered breakdown, per-instrument details, and provenance hashes'),
    (gen_random_uuid(), 'scope2-market:calculations:write',    'scope2-market', 'calculations_write',    'Create and manage market-based emission calculation records'),
    (gen_random_uuid(), 'scope2-market:dual-reporting:read',   'scope2-market', 'dual_reporting_read',   'View dual reporting records comparing location-based and market-based emissions'),
    (gen_random_uuid(), 'scope2-market:dual-reporting:write',  'scope2-market', 'dual_reporting_write',  'Create and manage dual reporting comparison records'),
    (gen_random_uuid(), 'scope2-market:quality:read',          'scope2-market', 'quality_read',          'View GHG Protocol Scope 2 Quality Criteria assessments for instruments'),
    (gen_random_uuid(), 'scope2-market:quality:write',         'scope2-market', 'quality_write',         'Execute quality criteria assessments against GHG Protocol 7-criteria framework'),
    (gen_random_uuid(), 'scope2-market:compliance:read',       'scope2-market', 'compliance_read',       'View regulatory compliance records for GHG Protocol, ISO 14064, CSRD, RE100, CDP, TCFD, and SEC Climate'),
    (gen_random_uuid(), 'scope2-market:compliance:execute',    'scope2-market', 'compliance_execute',    'Execute regulatory compliance checks against multiple frameworks'),
    (gen_random_uuid(), 'scope2-market:admin',                 'scope2-market', 'admin',                 'Full administrative access to Scope 2 market-based emissions service including configuration, diagnostics, and bulk operations')
ON CONFLICT DO NOTHING;

-- ==========================================================================
-- Seed Data: Residual Mix Factors (60+ regions)
-- ==========================================================================

-- US eGRID subregion residual mix factors (2022 data, kgCO2e/kWh)
INSERT INTO scope2_market_service.s2m_residual_mix_factors (region, factor_kgco2e_kwh, source, year, country_code) VALUES
    ('US-AKGD',  0.51500000, 'eGRID_residual', 2022, 'US'),
    ('US-AKMS',  0.28700000, 'eGRID_residual', 2022, 'US'),
    ('US-AZNM',  0.43200000, 'eGRID_residual', 2022, 'US'),
    ('US-CAMX',  0.24100000, 'eGRID_residual', 2022, 'US'),
    ('US-ERCT',  0.39600000, 'eGRID_residual', 2022, 'US'),
    ('US-FRCC',  0.40800000, 'eGRID_residual', 2022, 'US'),
    ('US-HIMS',  0.67200000, 'eGRID_residual', 2022, 'US'),
    ('US-HIOA',  0.70300000, 'eGRID_residual', 2022, 'US'),
    ('US-MROE',  0.62800000, 'eGRID_residual', 2022, 'US'),
    ('US-MROW',  0.52100000, 'eGRID_residual', 2022, 'US'),
    ('US-NEWE',  0.23600000, 'eGRID_residual', 2022, 'US'),
    ('US-NWPP',  0.30200000, 'eGRID_residual', 2022, 'US'),
    ('US-NYCW',  0.28400000, 'eGRID_residual', 2022, 'US'),
    ('US-NYLI',  0.53700000, 'eGRID_residual', 2022, 'US'),
    ('US-NYUP',  0.15800000, 'eGRID_residual', 2022, 'US'),
    ('US-PRMS',  0.80200000, 'eGRID_residual', 2022, 'US'),
    ('US-RFCE',  0.34600000, 'eGRID_residual', 2022, 'US'),
    ('US-RFCM',  0.52300000, 'eGRID_residual', 2022, 'US'),
    ('US-RFCW',  0.55100000, 'eGRID_residual', 2022, 'US'),
    ('US-RMPA',  0.59400000, 'eGRID_residual', 2022, 'US'),
    ('US-SPNO',  0.63500000, 'eGRID_residual', 2022, 'US'),
    ('US-SPSO',  0.47900000, 'eGRID_residual', 2022, 'US'),
    ('US-SRMV',  0.38300000, 'eGRID_residual', 2022, 'US'),
    ('US-SRMW',  0.71700000, 'eGRID_residual', 2022, 'US'),
    ('US-SRSO',  0.42800000, 'eGRID_residual', 2022, 'US'),
    ('US-SRTV',  0.38900000, 'eGRID_residual', 2022, 'US'),
    ('US-SRVC',  0.33800000, 'eGRID_residual', 2022, 'US')
ON CONFLICT (region, year, source) DO NOTHING;

-- EU country residual mix factors (AIB 2022, kgCO2e/kWh)
INSERT INTO scope2_market_service.s2m_residual_mix_factors (region, factor_kgco2e_kwh, source, year, country_code) VALUES
    ('EU-AT',  0.26500000, 'AIB_residual', 2022, 'AT'),
    ('EU-BE',  0.21600000, 'AIB_residual', 2022, 'BE'),
    ('EU-BG',  0.67400000, 'AIB_residual', 2022, 'BG'),
    ('EU-HR',  0.31500000, 'AIB_residual', 2022, 'HR'),
    ('EU-CY',  0.70200000, 'AIB_residual', 2022, 'CY'),
    ('EU-CZ',  0.63100000, 'AIB_residual', 2022, 'CZ'),
    ('EU-DK',  0.34200000, 'AIB_residual', 2022, 'DK'),
    ('EU-EE',  0.85300000, 'AIB_residual', 2022, 'EE'),
    ('EU-FI',  0.16400000, 'AIB_residual', 2022, 'FI'),
    ('EU-FR',  0.07500000, 'AIB_residual', 2022, 'FR'),
    ('EU-DE',  0.59100000, 'AIB_residual', 2022, 'DE'),
    ('EU-GR',  0.56800000, 'AIB_residual', 2022, 'GR'),
    ('EU-HU',  0.37200000, 'AIB_residual', 2022, 'HU'),
    ('EU-IE',  0.43200000, 'AIB_residual', 2022, 'IE'),
    ('EU-IT',  0.43300000, 'AIB_residual', 2022, 'IT'),
    ('EU-LV',  0.20100000, 'AIB_residual', 2022, 'LV'),
    ('EU-LT',  0.23800000, 'AIB_residual', 2022, 'LT'),
    ('EU-LU',  0.37600000, 'AIB_residual', 2022, 'LU'),
    ('EU-MT',  0.62500000, 'AIB_residual', 2022, 'MT'),
    ('EU-NL',  0.49300000, 'AIB_residual', 2022, 'NL'),
    ('EU-PL',  0.82100000, 'AIB_residual', 2022, 'PL'),
    ('EU-PT',  0.30400000, 'AIB_residual', 2022, 'PT'),
    ('EU-RO',  0.43600000, 'AIB_residual', 2022, 'RO'),
    ('EU-SK',  0.26200000, 'AIB_residual', 2022, 'SK'),
    ('EU-SI',  0.38100000, 'AIB_residual', 2022, 'SI'),
    ('EU-ES',  0.25100000, 'AIB_residual', 2022, 'ES'),
    ('EU-SE',  0.04500000, 'AIB_residual', 2022, 'SE')
ON CONFLICT (region, year, source) DO NOTHING;

-- APAC residual mix factors (IEA/national sources 2022, kgCO2e/kWh)
INSERT INTO scope2_market_service.s2m_residual_mix_factors (region, factor_kgco2e_kwh, source, year, country_code) VALUES
    ('APAC-CN',  0.58100000, 'IEA_residual', 2022, 'CN'),
    ('APAC-IN',  0.72300000, 'IEA_residual', 2022, 'IN'),
    ('APAC-JP',  0.47100000, 'IEA_residual', 2022, 'JP'),
    ('APAC-KR',  0.45900000, 'IEA_residual', 2022, 'KR'),
    ('APAC-AU',  0.68200000, 'IEA_residual', 2022, 'AU'),
    ('APAC-SG',  0.40800000, 'IEA_residual', 2022, 'SG'),
    ('APAC-TH',  0.49500000, 'IEA_residual', 2022, 'TH'),
    ('APAC-MY',  0.58600000, 'IEA_residual', 2022, 'MY'),
    ('APAC-ID',  0.71600000, 'IEA_residual', 2022, 'ID'),
    ('APAC-PH',  0.62300000, 'IEA_residual', 2022, 'PH'),
    ('APAC-VN',  0.53700000, 'IEA_residual', 2022, 'VN'),
    ('APAC-NZ',  0.11500000, 'IEA_residual', 2022, 'NZ')
ON CONFLICT (region, year, source) DO NOTHING;

-- Americas residual mix factors (kgCO2e/kWh)
INSERT INTO scope2_market_service.s2m_residual_mix_factors (region, factor_kgco2e_kwh, source, year, country_code) VALUES
    ('AMER-CA',  0.12600000, 'IEA_residual', 2022, 'CA'),
    ('AMER-MX',  0.42300000, 'IEA_residual', 2022, 'MX'),
    ('AMER-BR',  0.07400000, 'IEA_residual', 2022, 'BR'),
    ('AMER-CL',  0.34100000, 'IEA_residual', 2022, 'CL'),
    ('AMER-CO',  0.17200000, 'IEA_residual', 2022, 'CO'),
    ('AMER-AR',  0.32800000, 'IEA_residual', 2022, 'AR')
ON CONFLICT (region, year, source) DO NOTHING;

-- Global/other residual mix factors (kgCO2e/kWh)
INSERT INTO scope2_market_service.s2m_residual_mix_factors (region, factor_kgco2e_kwh, source, year, country_code) VALUES
    ('GLOBAL',       0.49400000, 'IEA_global', 2022, NULL),
    ('OTHER-GB',     0.23100000, 'DEFRA_residual', 2022, 'GB'),
    ('OTHER-ZA',     0.92800000, 'IEA_residual', 2022, 'ZA'),
    ('OTHER-AE',     0.43600000, 'IEA_residual', 2022, 'AE'),
    ('OTHER-SA',     0.63400000, 'IEA_residual', 2022, 'SA'),
    ('OTHER-IL',     0.52700000, 'IEA_residual', 2022, 'IL'),
    ('OTHER-NO',     0.01600000, 'NVE_residual', 2022, 'NO'),
    ('OTHER-IS',     0.00000000, 'IEA_residual', 2022, 'IS')
ON CONFLICT (region, year, source) DO NOTHING;

-- ==========================================================================
-- Seed Data: Energy Source Factors (11 sources)
-- ==========================================================================
INSERT INTO scope2_market_service.s2m_energy_source_factors (source, ef_kgco2e_kwh, is_renewable, is_biogenic) VALUES
    ('solar',           0.00000000, TRUE,  FALSE),
    ('wind',            0.00000000, TRUE,  FALSE),
    ('hydro',           0.00000000, TRUE,  FALSE),
    ('geothermal',      0.00000000, TRUE,  FALSE),
    ('biomass',         0.00000000, TRUE,  TRUE),
    ('nuclear',         0.00000000, FALSE, FALSE),
    ('natural_gas',     0.41000000, FALSE, FALSE),
    ('coal',            0.91000000, FALSE, FALSE),
    ('oil',             0.65000000, FALSE, FALSE),
    ('waste',           0.33000000, FALSE, FALSE),
    ('grid_average',    0.49400000, FALSE, FALSE)
ON CONFLICT (source) DO NOTHING;

-- ==========================================================================
-- Seed Data: Instrument Type Reference (10 types)
-- ==========================================================================
-- This reference data is stored as comments and can be used by the
-- application layer for UI display and validation.
--
-- Instrument Types:
--   rec              - Renewable Energy Certificate (North America)
--   go               - Guarantee of Origin (EU/EEA)
--   i-rec            - International REC (global markets)
--   tigr             - Tradable Instrument for Global Renewables
--   ppa              - Power Purchase Agreement (physical)
--   vppa             - Virtual Power Purchase Agreement (financial)
--   green_tariff     - Utility green tariff program
--   direct_contract  - Direct contract with generator
--   unbundled_rec    - Unbundled renewable energy certificate
--   bundled_rec      - Bundled renewable energy certificate
--   other            - Other contractual instrument

-- ==========================================================================
-- Seed Data: Register Agent in Agent Registry
-- ==========================================================================

INSERT INTO agent_registry_service.agents (
    agent_id, name, description, layer, execution_mode,
    idempotency_support, deterministic, max_concurrent_runs,
    glip_version, supports_checkpointing, author,
    documentation_url, enabled, tenant_id
) VALUES (
    'GL-MRV-SCOPE2-010',
    'Scope 2 Market-Based Emissions Agent',
    'Market-based Scope 2 emission calculator for GreenLang Climate OS. Manages contractual instruments (RECs, Guarantees of Origin, I-RECs, TIGRs, PPAs, VPPAs, green tariffs, direct contracts, bundled/unbundled certificates) with certificate tracking, vintage years, tracking system registry (M-RETS, NAR, NEPOOL-GIS, WREGIS, ERCOT, PJM-GATS, AIB, I-REC Standard), instrument lifecycle management (active/allocated/retired/expired/cancelled), and allocation to energy purchases using FIFO/LIFO/pro-rata/priority/manual methods. Maintains residual mix emission factor database for 60+ regions (US eGRID 27 subregions, EU 27 member states via AIB, APAC 12 countries, Americas 6 countries, global fallback) with kgCO2e/kWh factors and source attribution. Stores supplier-specific emission factors with fuel mix disclosure and verification status. Tracks energy source factors for 11 generation types (solar, wind, hydro, geothermal, biomass, nuclear, natural gas, coal, oil, waste, grid average) with renewable and biogenic classification. Executes deterministic market-based calculations with covered/uncovered energy split, coverage percentage, per-instrument emission allocation, residual mix application for uncovered portion, multi-gas breakdown (CO2/CH4/N2O), and GWP weighting (AR4/AR5/AR6/AR6_20YR). Produces GHG Protocol dual reporting comparing location-based and market-based results with difference calculation. Implements GHG Protocol Scope 2 Quality Criteria 7-point assessment (unique claim, associated delivery, temporal match, geographic match, no double counting, recognized registry, represents generation) with overall scoring and threshold evaluation. Checks regulatory compliance against 8 frameworks (GHG Protocol Scope 2, Quality Criteria, ISO 14064, CSRD ESRS, RE100, CDP, TCFD, SEC Climate). Manages certificate retirement records for double-counting prevention. Generates entity-level audit trail entries with action tracking (CREATE/UPDATE/DELETE/CALCULATE/ALLOCATE/RETIRE/VALIDATE/ASSESS/IMPORT/EXPORT/DUAL_REPORT/COMPLIANCE_CHECK), prev_hash/current_hash chaining for tamper-evident provenance, and actor attribution. SHA-256 provenance chains for zero-hallucination audit trail.',
    2, 'async', true, true, 10, '1.0.0', true,
    'GreenLang MRV Team',
    'https://docs.greenlang.ai/agents/scope2-market-based',
    true, 'default'
) ON CONFLICT (agent_id) DO NOTHING;

INSERT INTO agent_registry_service.agent_versions (
    agent_id, version, resource_profile, container_spec,
    tags, sectors, provenance_hash
) VALUES (
    'GL-MRV-SCOPE2-010', '1.0.0',
    '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "512Mi", "memory_limit": "2Gi", "gpu": false}'::jsonb,
    '{"image": "greenlang/scope2-market-based-service", "tag": "1.0.0", "port": 8000}'::jsonb,
    '{"scope-2", "market-based", "rec", "go", "i-rec", "ppa", "dual-reporting", "quality-criteria", "ghg-protocol", "residual-mix", "mrv"}',
    '{"energy", "utilities", "corporate", "manufacturing", "technology", "financial-services", "cross-sector"}',
    'b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3'
) ON CONFLICT (agent_id, version) DO NOTHING;

INSERT INTO agent_registry_service.agent_capabilities (
    agent_id, version, name, category,
    description, input_types, output_types, parameters
) VALUES
(
    'GL-MRV-SCOPE2-010', '1.0.0',
    'instrument_management',
    'configuration',
    'Register and manage contractual instruments (RECs, GOs, I-RECs, TIGRs, PPAs, VPPAs, green tariffs, direct contracts) with certificate tracking, vintage years, tracking system registry, verification, and lifecycle management.',
    '{"instrument_type", "energy_source", "quantity_mwh", "emission_factor", "vintage_year", "tracking_system", "certificate_id"}',
    '{"instrument_id", "status", "registration_result"}',
    '{"instrument_types": ["rec", "go", "i-rec", "tigr", "ppa", "vppa", "green_tariff", "direct_contract", "unbundled_rec", "bundled_rec", "other"], "tracking_systems": ["M-RETS", "NAR", "NEPOOL-GIS", "WREGIS", "ERCOT", "PJM-GATS", "AIB", "I-REC Standard"]}'::jsonb
),
(
    'GL-MRV-SCOPE2-010', '1.0.0',
    'market_based_calculation',
    'calculation',
    'Execute market-based emission calculations with covered/uncovered split, instrument allocation, residual mix for uncovered portion, per-instrument breakdown, and SHA-256 provenance hashing.',
    '{"facility_id", "total_mwh", "instruments", "residual_mix_region", "gwp_source"}',
    '{"calculation_id", "covered_mwh", "uncovered_mwh", "coverage_pct", "total_emissions_tco2e", "provenance_hash"}',
    '{"gwp_sources": ["AR4", "AR5", "AR6", "AR6_20YR"], "calculation_methods": ["market_based", "supplier_specific", "residual_mix", "hybrid"]}'::jsonb
),
(
    'GL-MRV-SCOPE2-010', '1.0.0',
    'dual_reporting',
    'reporting',
    'Generate GHG Protocol dual reporting comparing location-based and market-based emission totals with difference calculation and percentage variance.',
    '{"facility_id", "location_calc_id", "market_calc_id", "reporting_period"}',
    '{"dual_report_id", "location_tco2e", "market_tco2e", "difference_tco2e", "difference_pct"}',
    '{"reporting_periods": ["Q1", "Q2", "Q3", "Q4", "annual", "custom"]}'::jsonb
),
(
    'GL-MRV-SCOPE2-010', '1.0.0',
    'quality_criteria_assessment',
    'validation',
    'Assess contractual instruments against GHG Protocol Scope 2 Quality Criteria (7 criteria: unique claim, associated delivery, temporal match, geographic match, no double counting, recognized registry, represents generation).',
    '{"instrument_id", "consumption_facility", "consumption_period", "consumption_region"}',
    '{"assessment_id", "overall_score", "passes_threshold", "criteria_results"}',
    '{"criteria_count": 7, "threshold": 0.7143}'::jsonb
),
(
    'GL-MRV-SCOPE2-010', '1.0.0',
    'compliance_checking',
    'compliance',
    'Check market-based calculations against regulatory frameworks (GHG Protocol Scope 2, Quality Criteria, ISO 14064, CSRD ESRS, RE100, CDP, TCFD, SEC Climate).',
    '{"calculation_id", "frameworks"}',
    '{"compliance_records", "overall_status", "score"}',
    '{"frameworks": ["ghg_protocol_scope2", "ghg_protocol_quality_criteria", "iso_14064", "csrd_esrs", "re100", "cdp", "tcfd", "sec_climate"]}'::jsonb
)
ON CONFLICT DO NOTHING;

-- ==========================================================================
-- Migration complete: V061 Scope 2 Market-Based Emissions Service
-- ==========================================================================
