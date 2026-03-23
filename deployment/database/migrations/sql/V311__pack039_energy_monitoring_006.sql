-- =============================================================================
-- V311: PACK-039 Energy Monitoring Pack - Cost Allocation
-- =============================================================================
-- Pack:         PACK-039 (Energy Monitoring Pack)
-- Migration:    006 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates cost allocation tables for distributing energy costs across
-- tenants, departments, cost centers, and equipment. Includes tenant
-- account management, configurable allocation rules, calculated cost
-- allocations, billing record generation, and reconciliation logs for
-- ensuring cost balance integrity.
--
-- Tables (5):
--   1. pack039_energy_monitoring.em_tenant_accounts
--   2. pack039_energy_monitoring.em_allocation_rules
--   3. pack039_energy_monitoring.em_cost_allocations
--   4. pack039_energy_monitoring.em_billing_records
--   5. pack039_energy_monitoring.em_reconciliation_logs
--
-- Previous: V310__pack039_energy_monitoring_005.sql
-- =============================================================================

SET search_path TO pack039_energy_monitoring, public;

-- =============================================================================
-- Table 1: pack039_energy_monitoring.em_tenant_accounts
-- =============================================================================
-- Manages tenant/department/cost center accounts for energy cost allocation.
-- Each account represents an entity that consumes energy and receives cost
-- allocations. Accounts can be hierarchical (company -> division -> department)
-- and linked to specific meters for direct metering or shared meters for
-- proportional allocation.

CREATE TABLE pack039_energy_monitoring.em_tenant_accounts (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    account_name            VARCHAR(255)    NOT NULL,
    account_code            VARCHAR(50)     NOT NULL,
    account_type            VARCHAR(30)     NOT NULL DEFAULT 'DEPARTMENT',
    parent_account_id       UUID            REFERENCES pack039_energy_monitoring.em_tenant_accounts(id),
    hierarchy_level         INTEGER         NOT NULL DEFAULT 1,
    cost_center_code        VARCHAR(50),
    gl_account_number       VARCHAR(50),
    erp_entity_id           VARCHAR(100),
    contact_name            VARCHAR(255),
    contact_email           VARCHAR(255),
    contact_phone           VARCHAR(50),
    billing_address         TEXT,
    meter_ids               UUID[]          DEFAULT '{}',
    floor_area_m2           NUMERIC(12,2),
    floor_area_pct          NUMERIC(7,4),
    headcount               INTEGER,
    headcount_pct           NUMERIC(7,4),
    operating_hours_weekly  NUMERIC(6,2),
    default_allocation_pct  NUMERIC(7,4),
    allocation_method       VARCHAR(30)     NOT NULL DEFAULT 'METERED',
    billing_frequency       VARCHAR(20)     NOT NULL DEFAULT 'MONTHLY',
    billing_currency        VARCHAR(3)      NOT NULL DEFAULT 'USD',
    billing_start_date      DATE,
    contract_end_date       DATE,
    security_deposit        NUMERIC(12,2),
    credit_limit            NUMERIC(12,2),
    current_balance         NUMERIC(12,2)   NOT NULL DEFAULT 0,
    ytd_energy_kwh          NUMERIC(15,3)   NOT NULL DEFAULT 0,
    ytd_cost                NUMERIC(12,2)   NOT NULL DEFAULT 0,
    account_status          VARCHAR(20)     NOT NULL DEFAULT 'ACTIVE',
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_ta_type CHECK (
        account_type IN (
            'TENANT', 'DEPARTMENT', 'COST_CENTER', 'BUILDING',
            'FLOOR', 'ZONE', 'EQUIPMENT_GROUP', 'PROCESS',
            'COMMON_AREA', 'EXTERNAL', 'OTHER'
        )
    ),
    CONSTRAINT chk_p039_ta_level CHECK (
        hierarchy_level >= 1 AND hierarchy_level <= 10
    ),
    CONSTRAINT chk_p039_ta_alloc_method CHECK (
        allocation_method IN (
            'METERED', 'FLOOR_AREA', 'HEADCOUNT', 'FIXED_PCT',
            'OPERATING_HOURS', 'CONNECTED_LOAD', 'HYBRID',
            'EQUAL_SPLIT', 'FORMULA', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_ta_billing_freq CHECK (
        billing_frequency IN ('WEEKLY', 'BIWEEKLY', 'MONTHLY', 'QUARTERLY', 'ANNUAL')
    ),
    CONSTRAINT chk_p039_ta_status CHECK (
        account_status IN (
            'ACTIVE', 'INACTIVE', 'SUSPENDED', 'CLOSED', 'PENDING'
        )
    ),
    CONSTRAINT chk_p039_ta_floor_area CHECK (
        floor_area_m2 IS NULL OR floor_area_m2 > 0
    ),
    CONSTRAINT chk_p039_ta_floor_pct CHECK (
        floor_area_pct IS NULL OR (floor_area_pct >= 0 AND floor_area_pct <= 100)
    ),
    CONSTRAINT chk_p039_ta_headcount CHECK (
        headcount IS NULL OR headcount >= 0
    ),
    CONSTRAINT chk_p039_ta_headcount_pct CHECK (
        headcount_pct IS NULL OR (headcount_pct >= 0 AND headcount_pct <= 100)
    ),
    CONSTRAINT chk_p039_ta_alloc_pct CHECK (
        default_allocation_pct IS NULL OR (default_allocation_pct >= 0 AND default_allocation_pct <= 100)
    ),
    CONSTRAINT chk_p039_ta_no_self_parent CHECK (
        parent_account_id IS NULL OR parent_account_id != id
    ),
    CONSTRAINT uq_p039_ta_tenant_code UNIQUE (tenant_id, account_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_ta_tenant          ON pack039_energy_monitoring.em_tenant_accounts(tenant_id);
CREATE INDEX idx_p039_ta_facility        ON pack039_energy_monitoring.em_tenant_accounts(facility_id);
CREATE INDEX idx_p039_ta_code            ON pack039_energy_monitoring.em_tenant_accounts(account_code);
CREATE INDEX idx_p039_ta_type            ON pack039_energy_monitoring.em_tenant_accounts(account_type);
CREATE INDEX idx_p039_ta_parent          ON pack039_energy_monitoring.em_tenant_accounts(parent_account_id);
CREATE INDEX idx_p039_ta_level           ON pack039_energy_monitoring.em_tenant_accounts(hierarchy_level);
CREATE INDEX idx_p039_ta_cost_center     ON pack039_energy_monitoring.em_tenant_accounts(cost_center_code);
CREATE INDEX idx_p039_ta_gl_account      ON pack039_energy_monitoring.em_tenant_accounts(gl_account_number);
CREATE INDEX idx_p039_ta_status          ON pack039_energy_monitoring.em_tenant_accounts(account_status);
CREATE INDEX idx_p039_ta_alloc_method    ON pack039_energy_monitoring.em_tenant_accounts(allocation_method);
CREATE INDEX idx_p039_ta_created         ON pack039_energy_monitoring.em_tenant_accounts(created_at DESC);
CREATE INDEX idx_p039_ta_meters          ON pack039_energy_monitoring.em_tenant_accounts USING GIN(meter_ids);
CREATE INDEX idx_p039_ta_metadata        ON pack039_energy_monitoring.em_tenant_accounts USING GIN(metadata);

-- Composite: active accounts by facility
CREATE INDEX idx_p039_ta_fac_active      ON pack039_energy_monitoring.em_tenant_accounts(facility_id, account_type)
    WHERE account_status = 'ACTIVE';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_ta_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_tenant_accounts
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack039_energy_monitoring.em_allocation_rules
-- =============================================================================
-- Configurable rules for distributing energy costs from source meters
-- to tenant accounts. Rules define the allocation methodology (metered,
-- proportional, formula-based), applicable periods, rate structures,
-- and priority ordering. Multiple rules can apply to a single account
-- for blended allocation methods.

CREATE TABLE pack039_energy_monitoring.em_allocation_rules (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    rule_name               VARCHAR(255)    NOT NULL,
    rule_code               VARCHAR(50)     NOT NULL,
    account_id              UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_tenant_accounts(id) ON DELETE CASCADE,
    source_meter_id         UUID            REFERENCES pack039_energy_monitoring.em_meters(id),
    allocation_method       VARCHAR(30)     NOT NULL DEFAULT 'METERED',
    allocation_pct          NUMERIC(7,4),
    allocation_formula      TEXT,
    rate_type               VARCHAR(30)     NOT NULL DEFAULT 'PASS_THROUGH',
    rate_value              NUMERIC(12,6),
    rate_unit               VARCHAR(30),
    rate_schedule            JSONB           DEFAULT '{}',
    markup_pct              NUMERIC(7,4)    DEFAULT 0,
    markup_fixed            NUMERIC(12,2)   DEFAULT 0,
    common_area_share_pct   NUMERIC(7,4),
    demand_allocation_method VARCHAR(30),
    demand_allocation_pct   NUMERIC(7,4),
    reactive_power_charge   BOOLEAN         NOT NULL DEFAULT false,
    power_factor_penalty_threshold NUMERIC(5,4),
    minimum_charge          NUMERIC(12,2),
    maximum_charge          NUMERIC(12,2),
    effective_from          DATE            NOT NULL DEFAULT CURRENT_DATE,
    effective_to            DATE,
    priority                INTEGER         NOT NULL DEFAULT 100,
    applies_to_energy_types VARCHAR(50)[]   DEFAULT '{ELECTRICITY}',
    applies_to_tariff_periods VARCHAR(30)[] DEFAULT '{}',
    proration_method        VARCHAR(20)     DEFAULT 'DAILY',
    is_enabled              BOOLEAN         NOT NULL DEFAULT true,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_alr_method CHECK (
        allocation_method IN (
            'METERED', 'FLOOR_AREA', 'HEADCOUNT', 'FIXED_PCT',
            'OPERATING_HOURS', 'CONNECTED_LOAD', 'HYBRID',
            'EQUAL_SPLIT', 'FORMULA', 'RESIDUAL', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_alr_rate_type CHECK (
        rate_type IN (
            'PASS_THROUGH', 'FIXED_RATE', 'TIERED', 'TOU',
            'BLENDED', 'MARKET_RATE', 'CONTRACT', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_alr_demand_method CHECK (
        demand_allocation_method IS NULL OR demand_allocation_method IN (
            'COINCIDENT_PEAK', 'NON_COINCIDENT_PEAK', 'AVERAGE_DEMAND',
            'FIXED_PCT', 'METERED', 'EQUAL_SPLIT'
        )
    ),
    CONSTRAINT chk_p039_alr_proration CHECK (
        proration_method IS NULL OR proration_method IN (
            'DAILY', 'CALENDAR', 'BUSINESS_DAY', 'NONE'
        )
    ),
    CONSTRAINT chk_p039_alr_alloc_pct CHECK (
        allocation_pct IS NULL OR (allocation_pct >= 0 AND allocation_pct <= 100)
    ),
    CONSTRAINT chk_p039_alr_demand_pct CHECK (
        demand_allocation_pct IS NULL OR (demand_allocation_pct >= 0 AND demand_allocation_pct <= 100)
    ),
    CONSTRAINT chk_p039_alr_common CHECK (
        common_area_share_pct IS NULL OR (common_area_share_pct >= 0 AND common_area_share_pct <= 100)
    ),
    CONSTRAINT chk_p039_alr_markup_pct CHECK (
        markup_pct IS NULL OR markup_pct >= -100
    ),
    CONSTRAINT chk_p039_alr_pf_threshold CHECK (
        power_factor_penalty_threshold IS NULL OR
        (power_factor_penalty_threshold >= 0 AND power_factor_penalty_threshold <= 1)
    ),
    CONSTRAINT chk_p039_alr_min_max CHECK (
        minimum_charge IS NULL OR maximum_charge IS NULL OR minimum_charge <= maximum_charge
    ),
    CONSTRAINT chk_p039_alr_dates CHECK (
        effective_to IS NULL OR effective_from <= effective_to
    ),
    CONSTRAINT chk_p039_alr_priority CHECK (
        priority >= 1 AND priority <= 9999
    ),
    CONSTRAINT uq_p039_alr_tenant_code UNIQUE (tenant_id, rule_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_alr_tenant         ON pack039_energy_monitoring.em_allocation_rules(tenant_id);
CREATE INDEX idx_p039_alr_facility       ON pack039_energy_monitoring.em_allocation_rules(facility_id);
CREATE INDEX idx_p039_alr_code           ON pack039_energy_monitoring.em_allocation_rules(rule_code);
CREATE INDEX idx_p039_alr_account        ON pack039_energy_monitoring.em_allocation_rules(account_id);
CREATE INDEX idx_p039_alr_source_meter   ON pack039_energy_monitoring.em_allocation_rules(source_meter_id);
CREATE INDEX idx_p039_alr_method         ON pack039_energy_monitoring.em_allocation_rules(allocation_method);
CREATE INDEX idx_p039_alr_rate_type      ON pack039_energy_monitoring.em_allocation_rules(rate_type);
CREATE INDEX idx_p039_alr_enabled        ON pack039_energy_monitoring.em_allocation_rules(is_enabled) WHERE is_enabled = true;
CREATE INDEX idx_p039_alr_effective      ON pack039_energy_monitoring.em_allocation_rules(effective_from, effective_to);
CREATE INDEX idx_p039_alr_priority       ON pack039_energy_monitoring.em_allocation_rules(priority);
CREATE INDEX idx_p039_alr_created        ON pack039_energy_monitoring.em_allocation_rules(created_at DESC);
CREATE INDEX idx_p039_alr_energy_types   ON pack039_energy_monitoring.em_allocation_rules USING GIN(applies_to_energy_types);
CREATE INDEX idx_p039_alr_rate_sched     ON pack039_energy_monitoring.em_allocation_rules USING GIN(rate_schedule);

-- Composite: active rules by account for billing
CREATE INDEX idx_p039_alr_acct_active    ON pack039_energy_monitoring.em_allocation_rules(account_id, priority)
    WHERE is_enabled = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_alr_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_allocation_rules
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack039_energy_monitoring.em_cost_allocations
-- =============================================================================
-- Calculated cost allocation results for each account and billing period.
-- Each row represents the energy quantity and cost allocated to an account
-- for a specific period, broken down by energy type, tariff period, and
-- cost component (energy charge, demand charge, fixed charge, etc.).

CREATE TABLE pack039_energy_monitoring.em_cost_allocations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id              UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_tenant_accounts(id) ON DELETE CASCADE,
    allocation_rule_id      UUID            REFERENCES pack039_energy_monitoring.em_allocation_rules(id),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    billing_period_start    DATE            NOT NULL,
    billing_period_end      DATE            NOT NULL,
    energy_type             VARCHAR(50)     NOT NULL DEFAULT 'ELECTRICITY',
    energy_consumed_kwh     NUMERIC(15,3)   NOT NULL DEFAULT 0,
    energy_unit             VARCHAR(20)     NOT NULL DEFAULT 'kWh',
    peak_demand_kw          NUMERIC(12,3),
    average_demand_kw       NUMERIC(12,3),
    power_factor_avg        NUMERIC(5,4),
    energy_charge           NUMERIC(12,2)   NOT NULL DEFAULT 0,
    demand_charge           NUMERIC(12,2)   DEFAULT 0,
    fixed_charge            NUMERIC(12,2)   DEFAULT 0,
    reactive_power_charge   NUMERIC(12,2)   DEFAULT 0,
    power_factor_penalty    NUMERIC(12,2)   DEFAULT 0,
    common_area_charge      NUMERIC(12,2)   DEFAULT 0,
    markup_amount           NUMERIC(12,2)   DEFAULT 0,
    tax_amount              NUMERIC(12,2)   DEFAULT 0,
    credit_amount           NUMERIC(12,2)   DEFAULT 0,
    adjustment_amount       NUMERIC(12,2)   DEFAULT 0,
    total_charge            NUMERIC(12,2)   NOT NULL DEFAULT 0,
    currency                VARCHAR(3)      NOT NULL DEFAULT 'USD',
    cost_per_kwh            NUMERIC(10,6),
    allocation_pct_applied  NUMERIC(7,4),
    on_peak_kwh             NUMERIC(15,3),
    off_peak_kwh            NUMERIC(15,3),
    mid_peak_kwh            NUMERIC(15,3),
    tariff_breakdown        JSONB           DEFAULT '{}',
    calculation_method      VARCHAR(30),
    data_quality_pct        NUMERIC(5,2),
    estimated_pct           NUMERIC(5,2)    DEFAULT 0,
    allocation_status       VARCHAR(20)     NOT NULL DEFAULT 'CALCULATED',
    is_approved             BOOLEAN         NOT NULL DEFAULT false,
    approved_by             UUID,
    approved_at             TIMESTAMPTZ,
    invoice_id              UUID,
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_ca_energy_type CHECK (
        energy_type IN (
            'ELECTRICITY', 'NATURAL_GAS', 'STEAM', 'CHILLED_WATER',
            'HOT_WATER', 'COMPRESSED_AIR', 'TOTAL', 'OTHER'
        )
    ),
    CONSTRAINT chk_p039_ca_status CHECK (
        allocation_status IN (
            'CALCULATED', 'REVIEWED', 'APPROVED', 'INVOICED',
            'DISPUTED', 'ADJUSTED', 'CANCELLED'
        )
    ),
    CONSTRAINT chk_p039_ca_energy CHECK (
        energy_consumed_kwh >= 0
    ),
    CONSTRAINT chk_p039_ca_pf CHECK (
        power_factor_avg IS NULL OR (power_factor_avg >= 0 AND power_factor_avg <= 1)
    ),
    CONSTRAINT chk_p039_ca_alloc_pct CHECK (
        allocation_pct_applied IS NULL OR (allocation_pct_applied >= 0 AND allocation_pct_applied <= 100)
    ),
    CONSTRAINT chk_p039_ca_quality CHECK (
        data_quality_pct IS NULL OR (data_quality_pct >= 0 AND data_quality_pct <= 100)
    ),
    CONSTRAINT chk_p039_ca_estimated CHECK (
        estimated_pct IS NULL OR (estimated_pct >= 0 AND estimated_pct <= 100)
    ),
    CONSTRAINT chk_p039_ca_dates CHECK (
        billing_period_start <= billing_period_end
    ),
    CONSTRAINT uq_p039_ca_account_period_energy UNIQUE (account_id, billing_period_start, energy_type)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_ca_account         ON pack039_energy_monitoring.em_cost_allocations(account_id);
CREATE INDEX idx_p039_ca_rule            ON pack039_energy_monitoring.em_cost_allocations(allocation_rule_id);
CREATE INDEX idx_p039_ca_tenant          ON pack039_energy_monitoring.em_cost_allocations(tenant_id);
CREATE INDEX idx_p039_ca_facility        ON pack039_energy_monitoring.em_cost_allocations(facility_id);
CREATE INDEX idx_p039_ca_period_start    ON pack039_energy_monitoring.em_cost_allocations(billing_period_start DESC);
CREATE INDEX idx_p039_ca_energy_type     ON pack039_energy_monitoring.em_cost_allocations(energy_type);
CREATE INDEX idx_p039_ca_total           ON pack039_energy_monitoring.em_cost_allocations(total_charge DESC);
CREATE INDEX idx_p039_ca_status          ON pack039_energy_monitoring.em_cost_allocations(allocation_status);
CREATE INDEX idx_p039_ca_approved        ON pack039_energy_monitoring.em_cost_allocations(is_approved) WHERE is_approved = false;
CREATE INDEX idx_p039_ca_invoice         ON pack039_energy_monitoring.em_cost_allocations(invoice_id);
CREATE INDEX idx_p039_ca_created         ON pack039_energy_monitoring.em_cost_allocations(created_at DESC);
CREATE INDEX idx_p039_ca_tariff          ON pack039_energy_monitoring.em_cost_allocations USING GIN(tariff_breakdown);

-- Composite: account + monthly for billing history
CREATE INDEX idx_p039_ca_acct_monthly    ON pack039_energy_monitoring.em_cost_allocations(account_id, billing_period_start DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_ca_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_cost_allocations
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack039_energy_monitoring.em_billing_records
-- =============================================================================
-- Generated billing records (invoices) for tenant accounts. Each billing
-- record aggregates cost allocations for a billing period and generates
-- a payable or internal charge-back document. Tracks payment status,
-- dispute resolution, and credit/debit adjustments.

CREATE TABLE pack039_energy_monitoring.em_billing_records (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id              UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_tenant_accounts(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    invoice_number          VARCHAR(50)     NOT NULL,
    billing_period_start    DATE            NOT NULL,
    billing_period_end      DATE            NOT NULL,
    issue_date              DATE            NOT NULL DEFAULT CURRENT_DATE,
    due_date                DATE            NOT NULL,
    energy_total_kwh        NUMERIC(15,3)   NOT NULL DEFAULT 0,
    energy_charge_total     NUMERIC(12,2)   NOT NULL DEFAULT 0,
    demand_charge_total     NUMERIC(12,2)   DEFAULT 0,
    fixed_charges_total     NUMERIC(12,2)   DEFAULT 0,
    taxes_total             NUMERIC(12,2)   DEFAULT 0,
    credits_total           NUMERIC(12,2)   DEFAULT 0,
    adjustments_total       NUMERIC(12,2)   DEFAULT 0,
    late_fee                NUMERIC(12,2)   DEFAULT 0,
    subtotal                NUMERIC(12,2)   NOT NULL DEFAULT 0,
    grand_total             NUMERIC(12,2)   NOT NULL DEFAULT 0,
    currency                VARCHAR(3)      NOT NULL DEFAULT 'USD',
    previous_balance        NUMERIC(12,2)   DEFAULT 0,
    payments_received       NUMERIC(12,2)   DEFAULT 0,
    balance_forward         NUMERIC(12,2)   DEFAULT 0,
    amount_due              NUMERIC(12,2)   NOT NULL DEFAULT 0,
    billing_type            VARCHAR(20)     NOT NULL DEFAULT 'INTERNAL',
    payment_method          VARCHAR(30),
    payment_status          VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    payment_date            DATE,
    payment_reference       VARCHAR(100),
    dispute_status          VARCHAR(20),
    dispute_reason          TEXT,
    dispute_resolved_at     TIMESTAMPTZ,
    dispute_resolution      TEXT,
    erp_sync_status         VARCHAR(20)     DEFAULT 'PENDING',
    erp_sync_reference      VARCHAR(100),
    erp_synced_at           TIMESTAMPTZ,
    document_url            TEXT,
    line_items              JSONB           DEFAULT '[]',
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_br_billing_type CHECK (
        billing_type IN (
            'INTERNAL', 'EXTERNAL', 'CHARGEBACK', 'CREDIT_NOTE', 'ADJUSTMENT'
        )
    ),
    CONSTRAINT chk_p039_br_payment_method CHECK (
        payment_method IS NULL OR payment_method IN (
            'BANK_TRANSFER', 'CREDIT_CARD', 'DIRECT_DEBIT',
            'CHECK', 'INTERNAL_TRANSFER', 'ERP_JOURNAL', 'OTHER'
        )
    ),
    CONSTRAINT chk_p039_br_payment_status CHECK (
        payment_status IN (
            'PENDING', 'SENT', 'PARTIAL', 'PAID', 'OVERDUE',
            'CANCELLED', 'REFUNDED', 'WRITE_OFF'
        )
    ),
    CONSTRAINT chk_p039_br_dispute CHECK (
        dispute_status IS NULL OR dispute_status IN (
            'OPEN', 'UNDER_REVIEW', 'RESOLVED', 'ESCALATED', 'CLOSED'
        )
    ),
    CONSTRAINT chk_p039_br_erp_sync CHECK (
        erp_sync_status IS NULL OR erp_sync_status IN (
            'PENDING', 'IN_PROGRESS', 'SYNCED', 'FAILED', 'NOT_APPLICABLE'
        )
    ),
    CONSTRAINT chk_p039_br_dates CHECK (
        billing_period_start <= billing_period_end AND
        issue_date <= due_date
    ),
    CONSTRAINT uq_p039_br_invoice UNIQUE (tenant_id, invoice_number)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_br_account         ON pack039_energy_monitoring.em_billing_records(account_id);
CREATE INDEX idx_p039_br_tenant          ON pack039_energy_monitoring.em_billing_records(tenant_id);
CREATE INDEX idx_p039_br_facility        ON pack039_energy_monitoring.em_billing_records(facility_id);
CREATE INDEX idx_p039_br_invoice_num     ON pack039_energy_monitoring.em_billing_records(invoice_number);
CREATE INDEX idx_p039_br_period_start    ON pack039_energy_monitoring.em_billing_records(billing_period_start DESC);
CREATE INDEX idx_p039_br_issue_date      ON pack039_energy_monitoring.em_billing_records(issue_date DESC);
CREATE INDEX idx_p039_br_due_date        ON pack039_energy_monitoring.em_billing_records(due_date);
CREATE INDEX idx_p039_br_payment_status  ON pack039_energy_monitoring.em_billing_records(payment_status);
CREATE INDEX idx_p039_br_billing_type    ON pack039_energy_monitoring.em_billing_records(billing_type);
CREATE INDEX idx_p039_br_dispute         ON pack039_energy_monitoring.em_billing_records(dispute_status) WHERE dispute_status IS NOT NULL;
CREATE INDEX idx_p039_br_erp_sync        ON pack039_energy_monitoring.em_billing_records(erp_sync_status);
CREATE INDEX idx_p039_br_grand_total     ON pack039_energy_monitoring.em_billing_records(grand_total DESC);
CREATE INDEX idx_p039_br_created         ON pack039_energy_monitoring.em_billing_records(created_at DESC);
CREATE INDEX idx_p039_br_line_items      ON pack039_energy_monitoring.em_billing_records USING GIN(line_items);

-- Composite: overdue invoices for collection
CREATE INDEX idx_p039_br_overdue         ON pack039_energy_monitoring.em_billing_records(due_date, amount_due DESC)
    WHERE payment_status IN ('PENDING', 'SENT', 'OVERDUE');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_br_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_billing_records
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack039_energy_monitoring.em_reconciliation_logs
-- =============================================================================
-- Reconciliation records ensuring cost allocation integrity. Each
-- reconciliation run compares total utility costs with the sum of all
-- allocations to identify imbalances, unmetered areas, and allocation
-- gaps. Tracks discrepancies and their resolution for audit compliance.

CREATE TABLE pack039_energy_monitoring.em_reconciliation_logs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    reconciliation_period_start DATE        NOT NULL,
    reconciliation_period_end   DATE        NOT NULL,
    reconciliation_type     VARCHAR(30)     NOT NULL DEFAULT 'ENERGY',
    energy_type             VARCHAR(50)     NOT NULL DEFAULT 'ELECTRICITY',
    source_total_kwh        NUMERIC(15,3)   NOT NULL,
    allocated_total_kwh     NUMERIC(15,3)   NOT NULL,
    difference_kwh          NUMERIC(15,3)   NOT NULL,
    difference_pct          NUMERIC(10,4)   NOT NULL,
    source_total_cost       NUMERIC(12,2),
    allocated_total_cost    NUMERIC(12,2),
    cost_difference         NUMERIC(12,2),
    cost_difference_pct     NUMERIC(10,4),
    unmetered_energy_kwh    NUMERIC(15,3)   DEFAULT 0,
    distribution_loss_kwh   NUMERIC(15,3)   DEFAULT 0,
    common_area_kwh         NUMERIC(15,3)   DEFAULT 0,
    accounts_counted        INTEGER         NOT NULL DEFAULT 0,
    meters_counted          INTEGER         NOT NULL DEFAULT 0,
    tolerance_pct           NUMERIC(5,2)    NOT NULL DEFAULT 5.0,
    is_balanced             BOOLEAN         NOT NULL,
    reconciliation_status   VARCHAR(20)     NOT NULL DEFAULT 'COMPLETED',
    discrepancy_details     JSONB           DEFAULT '[]',
    resolution_notes        TEXT,
    resolved_by             UUID,
    resolved_at             TIMESTAMPTZ,
    adjustment_billing_id   UUID,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_rl_type CHECK (
        reconciliation_type IN (
            'ENERGY', 'COST', 'DEMAND', 'FULL', 'SPOT_CHECK'
        )
    ),
    CONSTRAINT chk_p039_rl_status CHECK (
        reconciliation_status IN (
            'COMPLETED', 'BALANCED', 'IMBALANCED', 'UNDER_REVIEW',
            'ADJUSTED', 'ACCEPTED', 'ESCALATED'
        )
    ),
    CONSTRAINT chk_p039_rl_tolerance CHECK (
        tolerance_pct >= 0 AND tolerance_pct <= 50
    ),
    CONSTRAINT chk_p039_rl_accounts CHECK (
        accounts_counted >= 0
    ),
    CONSTRAINT chk_p039_rl_meters CHECK (
        meters_counted >= 0
    ),
    CONSTRAINT chk_p039_rl_dates CHECK (
        reconciliation_period_start <= reconciliation_period_end
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_rl_tenant          ON pack039_energy_monitoring.em_reconciliation_logs(tenant_id);
CREATE INDEX idx_p039_rl_facility        ON pack039_energy_monitoring.em_reconciliation_logs(facility_id);
CREATE INDEX idx_p039_rl_period_start    ON pack039_energy_monitoring.em_reconciliation_logs(reconciliation_period_start DESC);
CREATE INDEX idx_p039_rl_type            ON pack039_energy_monitoring.em_reconciliation_logs(reconciliation_type);
CREATE INDEX idx_p039_rl_energy_type     ON pack039_energy_monitoring.em_reconciliation_logs(energy_type);
CREATE INDEX idx_p039_rl_balanced        ON pack039_energy_monitoring.em_reconciliation_logs(is_balanced) WHERE is_balanced = false;
CREATE INDEX idx_p039_rl_status          ON pack039_energy_monitoring.em_reconciliation_logs(reconciliation_status);
CREATE INDEX idx_p039_rl_diff_pct        ON pack039_energy_monitoring.em_reconciliation_logs(difference_pct DESC);
CREATE INDEX idx_p039_rl_created         ON pack039_energy_monitoring.em_reconciliation_logs(created_at DESC);
CREATE INDEX idx_p039_rl_discrepancies   ON pack039_energy_monitoring.em_reconciliation_logs USING GIN(discrepancy_details);

-- Composite: imbalanced reconciliations for review queue
CREATE INDEX idx_p039_rl_imbalanced      ON pack039_energy_monitoring.em_reconciliation_logs(facility_id, reconciliation_period_start DESC)
    WHERE is_balanced = false AND reconciliation_status IN ('COMPLETED', 'IMBALANCED');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_rl_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_reconciliation_logs
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack039_energy_monitoring.em_tenant_accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_allocation_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_cost_allocations ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_billing_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_reconciliation_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY p039_ta_tenant_isolation
    ON pack039_energy_monitoring.em_tenant_accounts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_ta_service_bypass
    ON pack039_energy_monitoring.em_tenant_accounts
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_alr_tenant_isolation
    ON pack039_energy_monitoring.em_allocation_rules
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_alr_service_bypass
    ON pack039_energy_monitoring.em_allocation_rules
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_ca_tenant_isolation
    ON pack039_energy_monitoring.em_cost_allocations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_ca_service_bypass
    ON pack039_energy_monitoring.em_cost_allocations
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_br_tenant_isolation
    ON pack039_energy_monitoring.em_billing_records
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_br_service_bypass
    ON pack039_energy_monitoring.em_billing_records
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_rl_tenant_isolation
    ON pack039_energy_monitoring.em_reconciliation_logs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_rl_service_bypass
    ON pack039_energy_monitoring.em_reconciliation_logs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_tenant_accounts TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_allocation_rules TO PUBLIC;
GRANT SELECT, INSERT, UPDATE ON pack039_energy_monitoring.em_cost_allocations TO PUBLIC;
GRANT SELECT, INSERT, UPDATE ON pack039_energy_monitoring.em_billing_records TO PUBLIC;
GRANT SELECT, INSERT, UPDATE ON pack039_energy_monitoring.em_reconciliation_logs TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack039_energy_monitoring.em_tenant_accounts IS
    'Tenant/department/cost center accounts for energy cost allocation with hierarchy, metering links, and billing configuration.';
COMMENT ON TABLE pack039_energy_monitoring.em_allocation_rules IS
    'Configurable rules for distributing energy costs from source meters to accounts using various allocation methodologies.';
COMMENT ON TABLE pack039_energy_monitoring.em_cost_allocations IS
    'Calculated cost allocation results per account, period, and energy type with charge component breakdown.';
COMMENT ON TABLE pack039_energy_monitoring.em_billing_records IS
    'Generated billing records (invoices) with payment tracking, dispute resolution, and ERP synchronization.';
COMMENT ON TABLE pack039_energy_monitoring.em_reconciliation_logs IS
    'Reconciliation records ensuring utility costs match sum of allocations with discrepancy tracking and resolution.';

COMMENT ON COLUMN pack039_energy_monitoring.em_tenant_accounts.allocation_method IS 'Default cost allocation method: METERED, FLOOR_AREA, HEADCOUNT, FIXED_PCT, FORMULA, etc.';
COMMENT ON COLUMN pack039_energy_monitoring.em_tenant_accounts.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack039_energy_monitoring.em_allocation_rules.rate_type IS 'Rate structure: PASS_THROUGH (utility rate), FIXED_RATE, TIERED, TOU, BLENDED, CONTRACT.';
COMMENT ON COLUMN pack039_energy_monitoring.em_allocation_rules.common_area_share_pct IS 'Share of common area energy costs allocated to this account.';
COMMENT ON COLUMN pack039_energy_monitoring.em_allocation_rules.demand_allocation_method IS 'Method for allocating demand charges: COINCIDENT_PEAK, NON_COINCIDENT_PEAK, etc.';

COMMENT ON COLUMN pack039_energy_monitoring.em_cost_allocations.cost_per_kwh IS 'Blended cost per kWh including all charges (energy, demand, fixed, taxes, markup).';
COMMENT ON COLUMN pack039_energy_monitoring.em_cost_allocations.estimated_pct IS 'Percentage of the allocation based on estimated (not measured) data.';

COMMENT ON COLUMN pack039_energy_monitoring.em_billing_records.erp_sync_status IS 'ERP journal entry synchronization status: PENDING, SYNCED, FAILED.';
COMMENT ON COLUMN pack039_energy_monitoring.em_billing_records.line_items IS 'JSON array of billing line items: [{description, quantity, unit, rate, amount}].';

COMMENT ON COLUMN pack039_energy_monitoring.em_reconciliation_logs.is_balanced IS 'Whether the difference is within the configured tolerance percentage.';
COMMENT ON COLUMN pack039_energy_monitoring.em_reconciliation_logs.unmetered_energy_kwh IS 'Energy not captured by any sub-meter (e.g., common areas, losses).';
