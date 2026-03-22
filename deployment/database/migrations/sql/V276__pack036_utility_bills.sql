-- =============================================================================
-- V276: PACK-036 Utility Analysis Pack - Core Schema & Utility Bills
-- =============================================================================
-- Pack:         PACK-036 (Utility Analysis Pack)
-- Migration:    001 of 010
-- Date:         March 2026
--
-- Creates the pack036_utility_analysis schema and foundational tables for
-- utility bill management, line-item tracking, and meter readings. Provides
-- the data foundation for all downstream rate analysis, demand profiling,
-- cost allocation, and regulatory charge optimization.
--
-- Tables (3):
--   1. pack036_utility_analysis.gl_utility_bills
--   2. pack036_utility_analysis.gl_bill_line_items
--   3. pack036_utility_analysis.gl_meter_readings
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V275__pack035_energy_benchmark_010.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS pack036_utility_analysis;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION pack036_utility_analysis.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: pack036_utility_analysis.gl_utility_bills
-- =============================================================================
-- Utility bills ingested from multiple formats (PDF, EDI, CSV, API).
-- Each bill captures consumption, demand, costs, and rate schedule for
-- a single billing period and commodity type at a service address.

CREATE TABLE pack036_utility_analysis.gl_utility_bills (
    bill_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    account_number          VARCHAR(100)    NOT NULL,
    meter_number            VARCHAR(100),
    commodity_type          VARCHAR(30)     NOT NULL,
    service_address         TEXT,
    billing_period_start    DATE            NOT NULL,
    billing_period_end      DATE            NOT NULL,
    days_in_period          INTEGER,
    total_consumption       NUMERIC(16,4),
    consumption_unit        VARCHAR(20)     NOT NULL DEFAULT 'kWh',
    peak_demand_kw          NUMERIC(12,4),
    power_factor            NUMERIC(5,4),
    rate_schedule           VARCHAR(100),
    total_amount            NUMERIC(14,2)   NOT NULL,
    taxes                   NUMERIC(14,2)   DEFAULT 0,
    bill_status             VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    bill_format             VARCHAR(30)     NOT NULL DEFAULT 'MANUAL',
    parsed_at               TIMESTAMPTZ,
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              UUID,
    -- Constraints
    CONSTRAINT chk_p036_ub_commodity CHECK (
        commodity_type IN (
            'ELECTRICITY', 'NATURAL_GAS', 'WATER', 'STEAM',
            'CHILLED_WATER', 'FUEL_OIL', 'LPG', 'DISTRICT_HEATING',
            'DISTRICT_COOLING', 'SEWER', 'OTHER'
        )
    ),
    CONSTRAINT chk_p036_ub_status CHECK (
        bill_status IN (
            'PENDING', 'VALIDATED', 'ESTIMATED', 'DISPUTED',
            'CORRECTED', 'APPROVED', 'ARCHIVED', 'ERROR'
        )
    ),
    CONSTRAINT chk_p036_ub_format CHECK (
        bill_format IN (
            'MANUAL', 'PDF', 'EDI', 'CSV', 'API', 'XML', 'SCRAPE'
        )
    ),
    CONSTRAINT chk_p036_ub_period CHECK (
        billing_period_end >= billing_period_start
    ),
    CONSTRAINT chk_p036_ub_days CHECK (
        days_in_period IS NULL OR (days_in_period >= 1 AND days_in_period <= 366)
    ),
    CONSTRAINT chk_p036_ub_consumption CHECK (
        total_consumption IS NULL OR total_consumption >= 0
    ),
    CONSTRAINT chk_p036_ub_demand CHECK (
        peak_demand_kw IS NULL OR peak_demand_kw >= 0
    ),
    CONSTRAINT chk_p036_ub_pf CHECK (
        power_factor IS NULL OR (power_factor >= 0 AND power_factor <= 1)
    ),
    CONSTRAINT chk_p036_ub_amount CHECK (
        total_amount >= 0
    ),
    CONSTRAINT chk_p036_ub_taxes CHECK (
        taxes IS NULL OR taxes >= 0
    ),
    CONSTRAINT chk_p036_ub_unit CHECK (
        consumption_unit IN (
            'kWh', 'MWh', 'therms', 'CCF', 'MCF', 'GJ', 'MJ',
            'kL', 'gallons', 'litres', 'tonnes', 'kg', 'lbs', 'BBL'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_ub_tenant         ON pack036_utility_analysis.gl_utility_bills(tenant_id);
CREATE INDEX idx_p036_ub_facility       ON pack036_utility_analysis.gl_utility_bills(facility_id);
CREATE INDEX idx_p036_ub_account        ON pack036_utility_analysis.gl_utility_bills(account_number);
CREATE INDEX idx_p036_ub_meter          ON pack036_utility_analysis.gl_utility_bills(meter_number);
CREATE INDEX idx_p036_ub_commodity      ON pack036_utility_analysis.gl_utility_bills(commodity_type);
CREATE INDEX idx_p036_ub_period_start   ON pack036_utility_analysis.gl_utility_bills(billing_period_start DESC);
CREATE INDEX idx_p036_ub_period_end     ON pack036_utility_analysis.gl_utility_bills(billing_period_end DESC);
CREATE INDEX idx_p036_ub_status         ON pack036_utility_analysis.gl_utility_bills(bill_status);
CREATE INDEX idx_p036_ub_created        ON pack036_utility_analysis.gl_utility_bills(created_at DESC);
CREATE INDEX idx_p036_ub_metadata       ON pack036_utility_analysis.gl_utility_bills USING GIN(metadata);

-- Composite: facility + commodity + period for common query pattern
CREATE INDEX idx_p036_ub_fac_comm_period ON pack036_utility_analysis.gl_utility_bills(facility_id, commodity_type, billing_period_start DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_ub_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_utility_bills
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack036_utility_analysis.gl_bill_line_items
-- =============================================================================
-- Individual charge line items from a utility bill. Enables granular cost
-- breakdown by charge category (energy, demand, fixed, taxes, riders, etc.).

CREATE TABLE pack036_utility_analysis.gl_bill_line_items (
    line_item_id            UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    bill_id                 UUID            NOT NULL REFERENCES pack036_utility_analysis.gl_utility_bills(bill_id) ON DELETE CASCADE,
    charge_category         VARCHAR(50)     NOT NULL,
    description             TEXT,
    quantity                NUMERIC(14,4),
    unit_rate               NUMERIC(14,6),
    amount                  NUMERIC(14,2)   NOT NULL,
    tax_rate                NUMERIC(6,4),
    tax_amount              NUMERIC(14,2)   DEFAULT 0,
    total                   NUMERIC(14,2)   NOT NULL,
    sort_order              INTEGER         DEFAULT 0,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_bli_category CHECK (
        charge_category IN (
            'ENERGY', 'DEMAND', 'FIXED', 'TRANSMISSION', 'DISTRIBUTION',
            'GENERATION', 'CAPACITY', 'REACTIVE_POWER', 'FUEL_ADJUSTMENT',
            'RENEWABLE_SURCHARGE', 'TAX', 'FRANCHISE_FEE', 'REGULATORY',
            'RIDER', 'CREDIT', 'LATE_FEE', 'DEPOSIT', 'OTHER'
        )
    ),
    CONSTRAINT chk_p036_bli_sort CHECK (
        sort_order >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_bli_bill          ON pack036_utility_analysis.gl_bill_line_items(bill_id);
CREATE INDEX idx_p036_bli_category     ON pack036_utility_analysis.gl_bill_line_items(charge_category);
CREATE INDEX idx_p036_bli_created      ON pack036_utility_analysis.gl_bill_line_items(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_bli_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_bill_line_items
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack036_utility_analysis.gl_meter_readings
-- =============================================================================
-- Meter reads associated with utility bills. Supports actual, estimated,
-- and customer reads with multiplier and previous-read delta calculation.

CREATE TABLE pack036_utility_analysis.gl_meter_readings (
    reading_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    bill_id                 UUID            NOT NULL REFERENCES pack036_utility_analysis.gl_utility_bills(bill_id) ON DELETE CASCADE,
    meter_id                VARCHAR(100)    NOT NULL,
    read_date               DATE            NOT NULL,
    read_value              NUMERIC(16,4)   NOT NULL,
    read_type               VARCHAR(20)     NOT NULL DEFAULT 'ACTUAL',
    units                   VARCHAR(20)     NOT NULL DEFAULT 'kWh',
    multiplier              NUMERIC(10,4)   DEFAULT 1.0,
    previous_read           NUMERIC(16,4),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_mr_read_type CHECK (
        read_type IN ('ACTUAL', 'ESTIMATED', 'CUSTOMER', 'DEMAND', 'NET', 'CHECK')
    ),
    CONSTRAINT chk_p036_mr_multiplier CHECK (
        multiplier > 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_mr_bill           ON pack036_utility_analysis.gl_meter_readings(bill_id);
CREATE INDEX idx_p036_mr_meter          ON pack036_utility_analysis.gl_meter_readings(meter_id);
CREATE INDEX idx_p036_mr_read_date      ON pack036_utility_analysis.gl_meter_readings(read_date DESC);
CREATE INDEX idx_p036_mr_type           ON pack036_utility_analysis.gl_meter_readings(read_type);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack036_utility_analysis.gl_utility_bills ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_bill_line_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_meter_readings ENABLE ROW LEVEL SECURITY;

CREATE POLICY p036_ub_tenant_isolation
    ON pack036_utility_analysis.gl_utility_bills
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p036_ub_service_bypass
    ON pack036_utility_analysis.gl_utility_bills
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_bli_tenant_isolation
    ON pack036_utility_analysis.gl_bill_line_items
    USING (bill_id IN (
        SELECT bill_id FROM pack036_utility_analysis.gl_utility_bills
        WHERE tenant_id = current_setting('app.current_tenant')::UUID
    ));

CREATE POLICY p036_bli_service_bypass
    ON pack036_utility_analysis.gl_bill_line_items
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_mr_tenant_isolation
    ON pack036_utility_analysis.gl_meter_readings
    USING (bill_id IN (
        SELECT bill_id FROM pack036_utility_analysis.gl_utility_bills
        WHERE tenant_id = current_setting('app.current_tenant')::UUID
    ));

CREATE POLICY p036_mr_service_bypass
    ON pack036_utility_analysis.gl_meter_readings
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA pack036_utility_analysis TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_utility_bills TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_bill_line_items TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_meter_readings TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack036_utility_analysis.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA pack036_utility_analysis IS
    'PACK-036 Utility Analysis Pack - utility bill management, rate analysis, demand profiling, cost allocation, procurement, benchmarking, and regulatory charge optimization.';

COMMENT ON TABLE pack036_utility_analysis.gl_utility_bills IS
    'Utility bills ingested from multiple formats with consumption, demand, costs, and rate schedule per billing period.';

COMMENT ON TABLE pack036_utility_analysis.gl_bill_line_items IS
    'Individual charge line items from a utility bill for granular cost breakdown by charge category.';

COMMENT ON TABLE pack036_utility_analysis.gl_meter_readings IS
    'Meter reads associated with utility bills supporting actual, estimated, and customer reads.';

COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.bill_id IS
    'Unique identifier for the utility bill.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.tenant_id IS
    'Multi-tenant isolation key.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.facility_id IS
    'Reference to the facility this bill belongs to.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.account_number IS
    'Utility account number as shown on the bill.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.meter_number IS
    'Physical meter number for this billing point.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.commodity_type IS
    'Commodity type: ELECTRICITY, NATURAL_GAS, WATER, STEAM, etc.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.service_address IS
    'Service address as printed on the utility bill.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.billing_period_start IS
    'First day of the billing period.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.billing_period_end IS
    'Last day of the billing period.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.days_in_period IS
    'Number of days in the billing period (for daily normalization).';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.total_consumption IS
    'Total metered consumption for the billing period.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.consumption_unit IS
    'Unit of measure for total_consumption (kWh, therms, CCF, etc.).';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.peak_demand_kw IS
    'Peak demand in kW recorded during the billing period.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.power_factor IS
    'Average power factor for the billing period (0 to 1).';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.rate_schedule IS
    'Utility tariff / rate schedule code applied to this bill.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.total_amount IS
    'Total bill amount in local currency before taxes.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.taxes IS
    'Total taxes and surcharges on the bill.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.bill_status IS
    'Bill processing status: PENDING, VALIDATED, ESTIMATED, DISPUTED, etc.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.bill_format IS
    'Source format of the bill: MANUAL, PDF, EDI, CSV, API, XML, SCRAPE.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.parsed_at IS
    'Timestamp when the bill was parsed/extracted from its source format.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_bills.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack036_utility_analysis.gl_bill_line_items.charge_category IS
    'Charge category: ENERGY, DEMAND, FIXED, TRANSMISSION, DISTRIBUTION, TAX, etc.';
COMMENT ON COLUMN pack036_utility_analysis.gl_bill_line_items.sort_order IS
    'Display order of line items on the bill.';
COMMENT ON COLUMN pack036_utility_analysis.gl_meter_readings.read_type IS
    'Type of meter read: ACTUAL, ESTIMATED, CUSTOMER, DEMAND, NET, CHECK.';
COMMENT ON COLUMN pack036_utility_analysis.gl_meter_readings.multiplier IS
    'Meter multiplier (CT ratio) to convert dial reading to actual consumption.';
COMMENT ON COLUMN pack036_utility_analysis.gl_meter_readings.previous_read IS
    'Previous meter reading value for delta calculation.';
