-- =============================================================================
-- GreenLang Climate OS - ERP/Finance Connector Service Schema
-- =============================================================================
-- Migration: V033
-- Component: AGENT-DATA-003 ERP/Finance Connector
-- Description: Creates erp_connector_service schema with erp_connections,
--              vendor_mappings, material_mappings, spend_records (hypertable),
--              purchase_orders, purchase_order_lines, inventory_snapshots,
--              sync_jobs (hypertable), emission_calculations,
--              erp_audit_log (hypertable), continuous aggregates for hourly
--              spend stats and hourly audit stats, 50+ indexes (including
--              GIN indexes on JSONB), RLS policies per tenant, 14 security
--              permissions, retention policies (30-day sync_jobs, 365-day
--              audit_log), compression, and seed data registering the
--              ERP/Finance Connector Agent (GL-DATA-X-004) with capabilities
--              in the agent registry.
-- Previous: V032 / V031__pdf_extractor_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS erp_connector_service;

-- =============================================================================
-- Table: erp_connector_service.erp_connections
-- =============================================================================
-- Connection registry. Each connection record captures ERP system type, host,
-- port, client credentials, company code, connection status, last sync time,
-- sync count, error count, tenant scope, provenance hash for integrity
-- verification, and timestamps. Tenant-scoped.

CREATE TABLE erp_connector_service.erp_connections (
    connection_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    erp_system VARCHAR(50) NOT NULL,
    host VARCHAR(500) NOT NULL,
    port INTEGER NOT NULL DEFAULT 443,
    client_id VARCHAR(255) NOT NULL,
    username VARCHAR(255) NOT NULL,
    company_code VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    last_sync TIMESTAMPTZ DEFAULT NULL,
    sync_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255) DEFAULT 'system'
);

-- ERP system constraint
ALTER TABLE erp_connector_service.erp_connections
    ADD CONSTRAINT chk_erp_system_type
    CHECK (erp_system IN (
        'sap_s4hana', 'sap_ecc', 'oracle_erp_cloud', 'oracle_ebs',
        'microsoft_dynamics_365', 'microsoft_dynamics_gp', 'netsuite',
        'workday', 'sage_intacct', 'sage_x3', 'infor_m3',
        'epicor', 'ifs', 'quickbooks', 'xero', 'custom'
    ));

-- Status constraint
ALTER TABLE erp_connector_service.erp_connections
    ADD CONSTRAINT chk_connection_status
    CHECK (status IN (
        'active', 'inactive', 'error', 'syncing', 'suspended',
        'pending', 'disconnected'
    ));

-- Host must not be empty
ALTER TABLE erp_connector_service.erp_connections
    ADD CONSTRAINT chk_connection_host_not_empty
    CHECK (LENGTH(TRIM(host)) > 0);

-- Client ID must not be empty
ALTER TABLE erp_connector_service.erp_connections
    ADD CONSTRAINT chk_connection_client_id_not_empty
    CHECK (LENGTH(TRIM(client_id)) > 0);

-- Username must not be empty
ALTER TABLE erp_connector_service.erp_connections
    ADD CONSTRAINT chk_connection_username_not_empty
    CHECK (LENGTH(TRIM(username)) > 0);

-- Company code must not be empty
ALTER TABLE erp_connector_service.erp_connections
    ADD CONSTRAINT chk_connection_company_code_not_empty
    CHECK (LENGTH(TRIM(company_code)) > 0);

-- Port must be valid
ALTER TABLE erp_connector_service.erp_connections
    ADD CONSTRAINT chk_connection_port_range
    CHECK (port > 0 AND port <= 65535);

-- Sync count must be non-negative
ALTER TABLE erp_connector_service.erp_connections
    ADD CONSTRAINT chk_connection_sync_count_non_negative
    CHECK (sync_count >= 0);

-- Error count must be non-negative
ALTER TABLE erp_connector_service.erp_connections
    ADD CONSTRAINT chk_connection_error_count_non_negative
    CHECK (error_count >= 0);

-- Provenance hash must be 64-character hex (SHA-256)
ALTER TABLE erp_connector_service.erp_connections
    ADD CONSTRAINT chk_connection_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: erp_connector_service.vendor_mappings
-- =============================================================================
-- Vendor classification and emission factor mapping. Each vendor record maps
-- an ERP vendor ID to GreenLang spend categories and Scope 3 emission factors.
-- Emission factors are expressed in kgCO2e per USD for spend-based calculations.
-- Tenant-scoped.

CREATE TABLE erp_connector_service.vendor_mappings (
    mapping_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_id VARCHAR(100) NOT NULL,
    vendor_name VARCHAR(500) NOT NULL,
    primary_category VARCHAR(100) NOT NULL,
    secondary_category VARCHAR(100) DEFAULT NULL,
    spend_category VARCHAR(100) NOT NULL,
    emission_factor_kgco2e_per_dollar DOUBLE PRECISION NOT NULL DEFAULT 0,
    emission_factor_source VARCHAR(200) NOT NULL DEFAULT 'EEIO',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    metadata JSONB DEFAULT '{}'::jsonb,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Vendor name must not be empty
ALTER TABLE erp_connector_service.vendor_mappings
    ADD CONSTRAINT chk_vendor_name_not_empty
    CHECK (LENGTH(TRIM(vendor_name)) > 0);

-- Vendor ID must not be empty
ALTER TABLE erp_connector_service.vendor_mappings
    ADD CONSTRAINT chk_vendor_id_not_empty
    CHECK (LENGTH(TRIM(vendor_id)) > 0);

-- Primary category must not be empty
ALTER TABLE erp_connector_service.vendor_mappings
    ADD CONSTRAINT chk_vendor_primary_category_not_empty
    CHECK (LENGTH(TRIM(primary_category)) > 0);

-- Spend category must not be empty
ALTER TABLE erp_connector_service.vendor_mappings
    ADD CONSTRAINT chk_vendor_spend_category_not_empty
    CHECK (LENGTH(TRIM(spend_category)) > 0);

-- Emission factor must be non-negative
ALTER TABLE erp_connector_service.vendor_mappings
    ADD CONSTRAINT chk_vendor_emission_factor_non_negative
    CHECK (emission_factor_kgco2e_per_dollar >= 0);

-- Confidence must be between 0 and 1
ALTER TABLE erp_connector_service.vendor_mappings
    ADD CONSTRAINT chk_vendor_confidence_range
    CHECK (confidence >= 0 AND confidence <= 1);

-- Provenance hash must be 64-character hex
ALTER TABLE erp_connector_service.vendor_mappings
    ADD CONSTRAINT chk_vendor_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- Unique vendor per tenant
CREATE UNIQUE INDEX uq_vendor_mapping_tenant
    ON erp_connector_service.vendor_mappings (vendor_id, tenant_id);

-- =============================================================================
-- Table: erp_connector_service.material_mappings
-- =============================================================================
-- Material classification and emission factor mapping. Each material record
-- maps an ERP material ID to GreenLang material groups, spend categories,
-- and activity-based emission factors. Emission factors are expressed in
-- kgCO2e per unit for activity-based calculations. Tenant-scoped.

CREATE TABLE erp_connector_service.material_mappings (
    mapping_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    material_id VARCHAR(100) NOT NULL,
    material_name VARCHAR(500) NOT NULL,
    material_group VARCHAR(100) NOT NULL,
    category VARCHAR(100) NOT NULL,
    spend_category VARCHAR(100) NOT NULL,
    unit VARCHAR(30) NOT NULL DEFAULT 'kg',
    emission_factor_kgco2e_per_unit DOUBLE PRECISION NOT NULL DEFAULT 0,
    emission_factor_source VARCHAR(200) NOT NULL DEFAULT 'EEIO',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    metadata JSONB DEFAULT '{}'::jsonb,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Material name must not be empty
ALTER TABLE erp_connector_service.material_mappings
    ADD CONSTRAINT chk_material_name_not_empty
    CHECK (LENGTH(TRIM(material_name)) > 0);

-- Material ID must not be empty
ALTER TABLE erp_connector_service.material_mappings
    ADD CONSTRAINT chk_material_id_not_empty
    CHECK (LENGTH(TRIM(material_id)) > 0);

-- Material group must not be empty
ALTER TABLE erp_connector_service.material_mappings
    ADD CONSTRAINT chk_material_group_not_empty
    CHECK (LENGTH(TRIM(material_group)) > 0);

-- Category must not be empty
ALTER TABLE erp_connector_service.material_mappings
    ADD CONSTRAINT chk_material_category_not_empty
    CHECK (LENGTH(TRIM(category)) > 0);

-- Emission factor must be non-negative
ALTER TABLE erp_connector_service.material_mappings
    ADD CONSTRAINT chk_material_emission_factor_non_negative
    CHECK (emission_factor_kgco2e_per_unit >= 0);

-- Confidence must be between 0 and 1
ALTER TABLE erp_connector_service.material_mappings
    ADD CONSTRAINT chk_material_confidence_range
    CHECK (confidence >= 0 AND confidence <= 1);

-- Provenance hash must be 64-character hex
ALTER TABLE erp_connector_service.material_mappings
    ADD CONSTRAINT chk_material_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- Unique material per tenant
CREATE UNIQUE INDEX uq_material_mapping_tenant
    ON erp_connector_service.material_mappings (material_id, tenant_id);

-- =============================================================================
-- Table: erp_connector_service.spend_records (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording spend/procurement transaction records
-- extracted from the ERP system. Each spend record captures transaction
-- details including type, date, vendor, amount, currency, USD conversion,
-- description, material group, cost center, GL account, Scope 3 category,
-- spend category, estimated emissions, connection reference, and provenance
-- hash. Partitioned by transaction_date for time-series queries. Retained
-- for 730 days (2 years) with compression after 30 days.

CREATE TABLE erp_connector_service.spend_records (
    record_id UUID NOT NULL DEFAULT gen_random_uuid(),
    transaction_type VARCHAR(30) NOT NULL,
    transaction_date TIMESTAMPTZ NOT NULL,
    vendor_id VARCHAR(100) DEFAULT NULL,
    vendor_name VARCHAR(500) DEFAULT NULL,
    amount DOUBLE PRECISION NOT NULL DEFAULT 0,
    currency VARCHAR(10) NOT NULL DEFAULT 'USD',
    amount_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    description TEXT DEFAULT '',
    material_group VARCHAR(100) DEFAULT NULL,
    cost_center VARCHAR(100) DEFAULT NULL,
    gl_account VARCHAR(50) DEFAULT NULL,
    scope3_category VARCHAR(50) DEFAULT NULL,
    spend_category VARCHAR(100) DEFAULT NULL,
    estimated_emissions_kgco2e DOUBLE PRECISION DEFAULT NULL,
    connection_id UUID NOT NULL,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (record_id, transaction_date)
);

-- Create hypertable partitioned by transaction_date
SELECT create_hypertable('erp_connector_service.spend_records', 'transaction_date', if_not_exists => TRUE);

-- Transaction type constraint
ALTER TABLE erp_connector_service.spend_records
    ADD CONSTRAINT chk_spend_transaction_type
    CHECK (transaction_type IN (
        'invoice', 'credit_note', 'debit_note', 'payment',
        'purchase_order', 'goods_receipt', 'service_entry',
        'expense_report', 'journal_entry', 'accrual', 'other'
    ));

-- Currency constraint (ISO 4217 common currencies)
ALTER TABLE erp_connector_service.spend_records
    ADD CONSTRAINT chk_spend_currency
    CHECK (LENGTH(TRIM(currency)) >= 3 AND LENGTH(TRIM(currency)) <= 10);

-- Scope 3 category constraint
ALTER TABLE erp_connector_service.spend_records
    ADD CONSTRAINT chk_spend_scope3_category
    CHECK (scope3_category IS NULL OR scope3_category IN (
        'cat1_purchased_goods', 'cat2_capital_goods', 'cat3_fuel_energy',
        'cat4_upstream_transport', 'cat5_waste', 'cat6_business_travel',
        'cat7_employee_commuting', 'cat8_upstream_leased',
        'cat9_downstream_transport', 'cat10_processing',
        'cat11_use_of_products', 'cat12_eol_treatment',
        'cat13_downstream_leased', 'cat14_franchises',
        'cat15_investments', 'unclassified'
    ));

-- Amount USD must be non-negative
ALTER TABLE erp_connector_service.spend_records
    ADD CONSTRAINT chk_spend_amount_usd_non_negative
    CHECK (amount_usd >= 0);

-- Provenance hash must be 64-character hex
ALTER TABLE erp_connector_service.spend_records
    ADD CONSTRAINT chk_spend_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: erp_connector_service.purchase_orders
-- =============================================================================
-- Purchase order header records extracted from the ERP system. Each PO
-- captures the PO number, vendor, dates, total amount, currency, status,
-- company code, Scope 3 category, spend category, connection reference,
-- and provenance hash. Tenant-scoped.

CREATE TABLE erp_connector_service.purchase_orders (
    po_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    po_number VARCHAR(100) NOT NULL,
    vendor_id VARCHAR(100) NOT NULL,
    vendor_name VARCHAR(500) NOT NULL,
    order_date TIMESTAMPTZ NOT NULL,
    delivery_date TIMESTAMPTZ DEFAULT NULL,
    total_amount DOUBLE PRECISION NOT NULL DEFAULT 0,
    currency VARCHAR(10) NOT NULL DEFAULT 'USD',
    amount_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    status VARCHAR(30) NOT NULL DEFAULT 'open',
    company_code VARCHAR(50) NOT NULL,
    scope3_category VARCHAR(50) DEFAULT NULL,
    spend_category VARCHAR(100) DEFAULT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    connection_id UUID NOT NULL REFERENCES erp_connector_service.erp_connections(connection_id) ON DELETE CASCADE,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- PO status constraint
ALTER TABLE erp_connector_service.purchase_orders
    ADD CONSTRAINT chk_po_status
    CHECK (status IN (
        'open', 'approved', 'partially_received', 'fully_received',
        'closed', 'cancelled', 'draft', 'pending_approval'
    ));

-- PO number must not be empty
ALTER TABLE erp_connector_service.purchase_orders
    ADD CONSTRAINT chk_po_number_not_empty
    CHECK (LENGTH(TRIM(po_number)) > 0);

-- Vendor name must not be empty
ALTER TABLE erp_connector_service.purchase_orders
    ADD CONSTRAINT chk_po_vendor_name_not_empty
    CHECK (LENGTH(TRIM(vendor_name)) > 0);

-- Total amount must be non-negative
ALTER TABLE erp_connector_service.purchase_orders
    ADD CONSTRAINT chk_po_total_amount_non_negative
    CHECK (total_amount >= 0);

-- Amount USD must be non-negative
ALTER TABLE erp_connector_service.purchase_orders
    ADD CONSTRAINT chk_po_amount_usd_non_negative
    CHECK (amount_usd >= 0);

-- Provenance hash must be 64-character hex
ALTER TABLE erp_connector_service.purchase_orders
    ADD CONSTRAINT chk_po_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- Scope 3 category constraint
ALTER TABLE erp_connector_service.purchase_orders
    ADD CONSTRAINT chk_po_scope3_category
    CHECK (scope3_category IS NULL OR scope3_category IN (
        'cat1_purchased_goods', 'cat2_capital_goods', 'cat3_fuel_energy',
        'cat4_upstream_transport', 'cat5_waste', 'cat6_business_travel',
        'cat7_employee_commuting', 'cat8_upstream_leased',
        'cat9_downstream_transport', 'cat10_processing',
        'cat11_use_of_products', 'cat12_eol_treatment',
        'cat13_downstream_leased', 'cat14_franchises',
        'cat15_investments', 'unclassified'
    ));

-- Unique PO number per connection and tenant
CREATE UNIQUE INDEX uq_po_number_connection_tenant
    ON erp_connector_service.purchase_orders (po_number, connection_id, tenant_id);

-- =============================================================================
-- Table: erp_connector_service.purchase_order_lines
-- =============================================================================
-- Purchase order line item records. Each line captures the material,
-- description, quantity, unit, unit price, total price, currency, delivery
-- date, cost center, and GL account. Linked to the parent PO.

CREATE TABLE erp_connector_service.purchase_order_lines (
    line_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    po_id UUID NOT NULL REFERENCES erp_connector_service.purchase_orders(po_id) ON DELETE CASCADE,
    line_number INTEGER NOT NULL,
    material_id VARCHAR(100) DEFAULT NULL,
    description TEXT NOT NULL DEFAULT '',
    quantity DOUBLE PRECISION NOT NULL DEFAULT 0,
    unit VARCHAR(30) NOT NULL DEFAULT 'EA',
    unit_price DOUBLE PRECISION NOT NULL DEFAULT 0,
    total_price DOUBLE PRECISION NOT NULL DEFAULT 0,
    currency VARCHAR(10) NOT NULL DEFAULT 'USD',
    delivery_date TIMESTAMPTZ DEFAULT NULL,
    cost_center VARCHAR(100) DEFAULT NULL,
    gl_account VARCHAR(50) DEFAULT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Line number must be positive
ALTER TABLE erp_connector_service.purchase_order_lines
    ADD CONSTRAINT chk_po_line_number_positive
    CHECK (line_number > 0);

-- Quantity must be non-negative
ALTER TABLE erp_connector_service.purchase_order_lines
    ADD CONSTRAINT chk_po_line_quantity_non_negative
    CHECK (quantity >= 0);

-- Unit price must be non-negative
ALTER TABLE erp_connector_service.purchase_order_lines
    ADD CONSTRAINT chk_po_line_unit_price_non_negative
    CHECK (unit_price >= 0);

-- Total price must be non-negative
ALTER TABLE erp_connector_service.purchase_order_lines
    ADD CONSTRAINT chk_po_line_total_price_non_negative
    CHECK (total_price >= 0);

-- Unique line number per PO
ALTER TABLE erp_connector_service.purchase_order_lines
    ADD CONSTRAINT uq_po_line_number
    UNIQUE (po_id, line_number);

-- =============================================================================
-- Table: erp_connector_service.inventory_snapshots
-- =============================================================================
-- Inventory position snapshots from the ERP system. Each snapshot captures
-- the material, quantity on hand, unit cost, total value, warehouse location,
-- last receipt date, connection reference, and snapshot date. Tenant-scoped.

CREATE TABLE erp_connector_service.inventory_snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    material_id VARCHAR(100) NOT NULL,
    material_name VARCHAR(500) NOT NULL,
    material_group VARCHAR(100) DEFAULT NULL,
    quantity_on_hand DOUBLE PRECISION NOT NULL DEFAULT 0,
    unit VARCHAR(30) NOT NULL DEFAULT 'EA',
    unit_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
    total_value DOUBLE PRECISION NOT NULL DEFAULT 0,
    warehouse_id VARCHAR(100) DEFAULT NULL,
    last_receipt_date TIMESTAMPTZ DEFAULT NULL,
    connection_id UUID NOT NULL REFERENCES erp_connector_service.erp_connections(connection_id) ON DELETE CASCADE,
    snapshot_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Material name must not be empty
ALTER TABLE erp_connector_service.inventory_snapshots
    ADD CONSTRAINT chk_inv_material_name_not_empty
    CHECK (LENGTH(TRIM(material_name)) > 0);

-- Material ID must not be empty
ALTER TABLE erp_connector_service.inventory_snapshots
    ADD CONSTRAINT chk_inv_material_id_not_empty
    CHECK (LENGTH(TRIM(material_id)) > 0);

-- Quantity on hand must be non-negative
ALTER TABLE erp_connector_service.inventory_snapshots
    ADD CONSTRAINT chk_inv_quantity_non_negative
    CHECK (quantity_on_hand >= 0);

-- Unit cost must be non-negative
ALTER TABLE erp_connector_service.inventory_snapshots
    ADD CONSTRAINT chk_inv_unit_cost_non_negative
    CHECK (unit_cost >= 0);

-- Total value must be non-negative
ALTER TABLE erp_connector_service.inventory_snapshots
    ADD CONSTRAINT chk_inv_total_value_non_negative
    CHECK (total_value >= 0);

-- Provenance hash must be 64-character hex
ALTER TABLE erp_connector_service.inventory_snapshots
    ADD CONSTRAINT chk_inv_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: erp_connector_service.sync_jobs (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording ERP synchronization job executions.
-- Each sync job captures the connection, sync mode (full/incremental/delta),
-- query type (spend/po/inventory/vendor/material), status, records synced,
-- records skipped, errors (JSONB), start/completion times, duration, and
-- provenance hash. Partitioned by started_at for time-series queries.
-- Retained for 30 days with compression after 3 days.

CREATE TABLE erp_connector_service.sync_jobs (
    job_id UUID NOT NULL DEFAULT gen_random_uuid(),
    connection_id UUID NOT NULL,
    sync_mode VARCHAR(20) NOT NULL DEFAULT 'incremental',
    query_type VARCHAR(30) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    records_synced INTEGER NOT NULL DEFAULT 0,
    records_skipped INTEGER NOT NULL DEFAULT 0,
    errors JSONB DEFAULT '[]'::jsonb,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ DEFAULT NULL,
    duration_seconds DOUBLE PRECISION NOT NULL DEFAULT 0,
    provenance_hash VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_by VARCHAR(255) DEFAULT 'system',
    PRIMARY KEY (job_id, started_at)
);

-- Create hypertable partitioned by started_at
SELECT create_hypertable('erp_connector_service.sync_jobs', 'started_at', if_not_exists => TRUE);

-- Sync mode constraint
ALTER TABLE erp_connector_service.sync_jobs
    ADD CONSTRAINT chk_sync_mode
    CHECK (sync_mode IN ('full', 'incremental', 'delta', 'manual', 'scheduled'));

-- Query type constraint
ALTER TABLE erp_connector_service.sync_jobs
    ADD CONSTRAINT chk_sync_query_type
    CHECK (query_type IN (
        'spend', 'purchase_orders', 'purchase_order_lines',
        'inventory', 'vendors', 'materials', 'gl_accounts',
        'cost_centers', 'company_codes', 'all'
    ));

-- Status constraint
ALTER TABLE erp_connector_service.sync_jobs
    ADD CONSTRAINT chk_sync_status
    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'timeout'));

-- Records synced must be non-negative
ALTER TABLE erp_connector_service.sync_jobs
    ADD CONSTRAINT chk_sync_records_synced_non_negative
    CHECK (records_synced >= 0);

-- Records skipped must be non-negative
ALTER TABLE erp_connector_service.sync_jobs
    ADD CONSTRAINT chk_sync_records_skipped_non_negative
    CHECK (records_skipped >= 0);

-- Duration must be non-negative
ALTER TABLE erp_connector_service.sync_jobs
    ADD CONSTRAINT chk_sync_duration_non_negative
    CHECK (duration_seconds >= 0);

-- Provenance hash must be 64-character hex
ALTER TABLE erp_connector_service.sync_jobs
    ADD CONSTRAINT chk_sync_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: erp_connector_service.emission_calculations
-- =============================================================================
-- Emission calculation results derived from spend records using vendor or
-- material emission factors. Each calculation links a spend record to an
-- estimated emission value with methodology (spend-based, activity-based,
-- hybrid) and emission factor source. Tenant-scoped.

CREATE TABLE erp_connector_service.emission_calculations (
    calc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    spend_record_id UUID NOT NULL,
    vendor_id VARCHAR(100) DEFAULT NULL,
    amount_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    emission_factor DOUBLE PRECISION NOT NULL DEFAULT 0,
    methodology VARCHAR(30) NOT NULL DEFAULT 'spend_based',
    estimated_kgco2e DOUBLE PRECISION NOT NULL DEFAULT 0,
    emission_factor_source VARCHAR(200) NOT NULL DEFAULT 'EEIO',
    scope3_category VARCHAR(50) DEFAULT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    metadata JSONB DEFAULT '{}'::jsonb,
    connection_id UUID NOT NULL REFERENCES erp_connector_service.erp_connections(connection_id) ON DELETE CASCADE,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Methodology constraint
ALTER TABLE erp_connector_service.emission_calculations
    ADD CONSTRAINT chk_calc_methodology
    CHECK (methodology IN (
        'spend_based', 'activity_based', 'hybrid',
        'supplier_specific', 'average_data', 'manual'
    ));

-- Amount USD must be non-negative
ALTER TABLE erp_connector_service.emission_calculations
    ADD CONSTRAINT chk_calc_amount_usd_non_negative
    CHECK (amount_usd >= 0);

-- Emission factor must be non-negative
ALTER TABLE erp_connector_service.emission_calculations
    ADD CONSTRAINT chk_calc_emission_factor_non_negative
    CHECK (emission_factor >= 0);

-- Estimated kgCO2e must be non-negative
ALTER TABLE erp_connector_service.emission_calculations
    ADD CONSTRAINT chk_calc_estimated_kgco2e_non_negative
    CHECK (estimated_kgco2e >= 0);

-- Confidence must be between 0 and 1
ALTER TABLE erp_connector_service.emission_calculations
    ADD CONSTRAINT chk_calc_confidence_range
    CHECK (confidence >= 0 AND confidence <= 1);

-- Provenance hash must be 64-character hex
ALTER TABLE erp_connector_service.emission_calculations
    ADD CONSTRAINT chk_calc_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- Scope 3 category constraint
ALTER TABLE erp_connector_service.emission_calculations
    ADD CONSTRAINT chk_calc_scope3_category
    CHECK (scope3_category IS NULL OR scope3_category IN (
        'cat1_purchased_goods', 'cat2_capital_goods', 'cat3_fuel_energy',
        'cat4_upstream_transport', 'cat5_waste', 'cat6_business_travel',
        'cat7_employee_commuting', 'cat8_upstream_leased',
        'cat9_downstream_transport', 'cat10_processing',
        'cat11_use_of_products', 'cat12_eol_treatment',
        'cat13_downstream_leased', 'cat14_franchises',
        'cat15_investments', 'unclassified'
    ));

-- =============================================================================
-- Table: erp_connector_service.erp_audit_log (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording comprehensive audit events for all
-- ERP connector operations. Each event captures the operation, entity type,
-- entity ID, details (JSONB), user, tenant, provenance hash, and timestamp.
-- Partitioned by created_at for time-series queries. Retained for 365 days
-- with compression after 30 days.

CREATE TABLE erp_connector_service.erp_audit_log (
    log_id UUID NOT NULL DEFAULT gen_random_uuid(),
    operation VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    details JSONB DEFAULT '{}'::jsonb,
    user_id VARCHAR(100) DEFAULT 'system',
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (log_id, created_at)
);

-- Create hypertable partitioned by created_at
SELECT create_hypertable('erp_connector_service.erp_audit_log', 'created_at', if_not_exists => TRUE);

-- Entity type constraint
ALTER TABLE erp_connector_service.erp_audit_log
    ADD CONSTRAINT chk_erp_audit_entity_type
    CHECK (entity_type IN (
        'erp_connection', 'vendor_mapping', 'material_mapping',
        'spend_record', 'purchase_order', 'purchase_order_line',
        'inventory_snapshot', 'sync_job', 'emission_calculation',
        'system'
    ));

-- Operation constraint
ALTER TABLE erp_connector_service.erp_audit_log
    ADD CONSTRAINT chk_erp_audit_operation
    CHECK (operation IN (
        'create', 'update', 'delete', 'sync_start', 'sync_complete',
        'sync_fail', 'connect', 'disconnect', 'map_vendor', 'map_material',
        'calculate_emissions', 'currency_convert', 'classify_spend',
        'activate', 'deactivate', 'system', 'admin'
    ));

-- Provenance hash must be 64-character hex
ALTER TABLE erp_connector_service.erp_audit_log
    ADD CONSTRAINT chk_erp_audit_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Continuous Aggregate: erp_connector_service.hourly_spend_stats
-- =============================================================================
-- Precomputed hourly spend statistics by Scope 3 category, spend category,
-- and currency for dashboard queries, trend analysis, and SLI tracking.

CREATE MATERIALIZED VIEW erp_connector_service.hourly_spend_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', transaction_date) AS bucket,
    scope3_category,
    spend_category,
    currency,
    tenant_id,
    COUNT(*) AS record_count,
    SUM(amount_usd) AS total_amount_usd,
    AVG(amount_usd) AS avg_amount_usd,
    MAX(amount_usd) AS max_amount_usd,
    SUM(COALESCE(estimated_emissions_kgco2e, 0)) AS total_emissions_kgco2e,
    AVG(COALESCE(estimated_emissions_kgco2e, 0)) AS avg_emissions_kgco2e,
    COUNT(DISTINCT vendor_id) AS unique_vendors
FROM erp_connector_service.spend_records
WHERE transaction_date IS NOT NULL
GROUP BY bucket, scope3_category, spend_category, currency, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('erp_connector_service.hourly_spend_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Continuous Aggregate: erp_connector_service.hourly_audit_stats
-- =============================================================================
-- Precomputed hourly counts of audit events by entity type and operation
-- for compliance reporting, dashboard queries, and long-term trend analysis.

CREATE MATERIALIZED VIEW erp_connector_service.hourly_audit_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at) AS bucket,
    entity_type,
    operation,
    tenant_id,
    COUNT(*) AS event_count,
    COUNT(DISTINCT entity_id) AS unique_entities,
    COUNT(DISTINCT user_id) AS unique_users
FROM erp_connector_service.erp_audit_log
WHERE created_at IS NOT NULL
GROUP BY bucket, entity_type, operation, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 30 minutes, covering the last 2 days
SELECT add_continuous_aggregate_policy('erp_connector_service.hourly_audit_stats',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- erp_connections indexes
CREATE INDEX idx_ec_erp_system ON erp_connector_service.erp_connections(erp_system);
CREATE INDEX idx_ec_status ON erp_connector_service.erp_connections(status);
CREATE INDEX idx_ec_company_code ON erp_connector_service.erp_connections(company_code);
CREATE INDEX idx_ec_last_sync ON erp_connector_service.erp_connections(last_sync DESC);
CREATE INDEX idx_ec_created_at ON erp_connector_service.erp_connections(created_at DESC);
CREATE INDEX idx_ec_updated_at ON erp_connector_service.erp_connections(updated_at DESC);
CREATE INDEX idx_ec_provenance_hash ON erp_connector_service.erp_connections(provenance_hash);
CREATE INDEX idx_ec_tenant ON erp_connector_service.erp_connections(tenant_id);
CREATE INDEX idx_ec_tenant_system ON erp_connector_service.erp_connections(tenant_id, erp_system);
CREATE INDEX idx_ec_tenant_status ON erp_connector_service.erp_connections(tenant_id, status);
CREATE INDEX idx_ec_metadata ON erp_connector_service.erp_connections USING GIN (metadata);

-- vendor_mappings indexes
CREATE INDEX idx_vm_vendor_id ON erp_connector_service.vendor_mappings(vendor_id);
CREATE INDEX idx_vm_vendor_name ON erp_connector_service.vendor_mappings(vendor_name);
CREATE INDEX idx_vm_primary_category ON erp_connector_service.vendor_mappings(primary_category);
CREATE INDEX idx_vm_spend_category ON erp_connector_service.vendor_mappings(spend_category);
CREATE INDEX idx_vm_emission_factor ON erp_connector_service.vendor_mappings(emission_factor_kgco2e_per_dollar);
CREATE INDEX idx_vm_provenance_hash ON erp_connector_service.vendor_mappings(provenance_hash);
CREATE INDEX idx_vm_created_at ON erp_connector_service.vendor_mappings(created_at DESC);
CREATE INDEX idx_vm_tenant ON erp_connector_service.vendor_mappings(tenant_id);
CREATE INDEX idx_vm_tenant_category ON erp_connector_service.vendor_mappings(tenant_id, primary_category);
CREATE INDEX idx_vm_metadata ON erp_connector_service.vendor_mappings USING GIN (metadata);

-- material_mappings indexes
CREATE INDEX idx_mm_material_id ON erp_connector_service.material_mappings(material_id);
CREATE INDEX idx_mm_material_name ON erp_connector_service.material_mappings(material_name);
CREATE INDEX idx_mm_material_group ON erp_connector_service.material_mappings(material_group);
CREATE INDEX idx_mm_category ON erp_connector_service.material_mappings(category);
CREATE INDEX idx_mm_spend_category ON erp_connector_service.material_mappings(spend_category);
CREATE INDEX idx_mm_emission_factor ON erp_connector_service.material_mappings(emission_factor_kgco2e_per_unit);
CREATE INDEX idx_mm_provenance_hash ON erp_connector_service.material_mappings(provenance_hash);
CREATE INDEX idx_mm_created_at ON erp_connector_service.material_mappings(created_at DESC);
CREATE INDEX idx_mm_tenant ON erp_connector_service.material_mappings(tenant_id);
CREATE INDEX idx_mm_tenant_group ON erp_connector_service.material_mappings(tenant_id, material_group);
CREATE INDEX idx_mm_metadata ON erp_connector_service.material_mappings USING GIN (metadata);

-- spend_records indexes (hypertable-aware)
CREATE INDEX idx_sr_vendor ON erp_connector_service.spend_records(vendor_id, transaction_date DESC);
CREATE INDEX idx_sr_vendor_name ON erp_connector_service.spend_records(vendor_name, transaction_date DESC);
CREATE INDEX idx_sr_type ON erp_connector_service.spend_records(transaction_type, transaction_date DESC);
CREATE INDEX idx_sr_scope3 ON erp_connector_service.spend_records(scope3_category, transaction_date DESC);
CREATE INDEX idx_sr_spend_category ON erp_connector_service.spend_records(spend_category, transaction_date DESC);
CREATE INDEX idx_sr_currency ON erp_connector_service.spend_records(currency, transaction_date DESC);
CREATE INDEX idx_sr_material_group ON erp_connector_service.spend_records(material_group, transaction_date DESC);
CREATE INDEX idx_sr_cost_center ON erp_connector_service.spend_records(cost_center, transaction_date DESC);
CREATE INDEX idx_sr_gl_account ON erp_connector_service.spend_records(gl_account, transaction_date DESC);
CREATE INDEX idx_sr_connection ON erp_connector_service.spend_records(connection_id, transaction_date DESC);
CREATE INDEX idx_sr_provenance_hash ON erp_connector_service.spend_records(provenance_hash);
CREATE INDEX idx_sr_tenant ON erp_connector_service.spend_records(tenant_id, transaction_date DESC);
CREATE INDEX idx_sr_tenant_scope3 ON erp_connector_service.spend_records(tenant_id, scope3_category, transaction_date DESC);
CREATE INDEX idx_sr_tenant_vendor ON erp_connector_service.spend_records(tenant_id, vendor_id, transaction_date DESC);

-- purchase_orders indexes
CREATE INDEX idx_po_po_number ON erp_connector_service.purchase_orders(po_number);
CREATE INDEX idx_po_vendor_id ON erp_connector_service.purchase_orders(vendor_id);
CREATE INDEX idx_po_vendor_name ON erp_connector_service.purchase_orders(vendor_name);
CREATE INDEX idx_po_order_date ON erp_connector_service.purchase_orders(order_date DESC);
CREATE INDEX idx_po_delivery_date ON erp_connector_service.purchase_orders(delivery_date DESC);
CREATE INDEX idx_po_status ON erp_connector_service.purchase_orders(status);
CREATE INDEX idx_po_scope3 ON erp_connector_service.purchase_orders(scope3_category);
CREATE INDEX idx_po_connection ON erp_connector_service.purchase_orders(connection_id);
CREATE INDEX idx_po_provenance_hash ON erp_connector_service.purchase_orders(provenance_hash);
CREATE INDEX idx_po_created_at ON erp_connector_service.purchase_orders(created_at DESC);
CREATE INDEX idx_po_tenant ON erp_connector_service.purchase_orders(tenant_id);
CREATE INDEX idx_po_tenant_status ON erp_connector_service.purchase_orders(tenant_id, status);
CREATE INDEX idx_po_tenant_vendor ON erp_connector_service.purchase_orders(tenant_id, vendor_id);
CREATE INDEX idx_po_metadata ON erp_connector_service.purchase_orders USING GIN (metadata);

-- purchase_order_lines indexes
CREATE INDEX idx_pol_po ON erp_connector_service.purchase_order_lines(po_id);
CREATE INDEX idx_pol_material ON erp_connector_service.purchase_order_lines(material_id);
CREATE INDEX idx_pol_line_number ON erp_connector_service.purchase_order_lines(line_number);
CREATE INDEX idx_pol_cost_center ON erp_connector_service.purchase_order_lines(cost_center);
CREATE INDEX idx_pol_gl_account ON erp_connector_service.purchase_order_lines(gl_account);
CREATE INDEX idx_pol_created_at ON erp_connector_service.purchase_order_lines(created_at DESC);
CREATE INDEX idx_pol_metadata ON erp_connector_service.purchase_order_lines USING GIN (metadata);

-- inventory_snapshots indexes
CREATE INDEX idx_is_material ON erp_connector_service.inventory_snapshots(material_id);
CREATE INDEX idx_is_material_name ON erp_connector_service.inventory_snapshots(material_name);
CREATE INDEX idx_is_material_group ON erp_connector_service.inventory_snapshots(material_group);
CREATE INDEX idx_is_warehouse ON erp_connector_service.inventory_snapshots(warehouse_id);
CREATE INDEX idx_is_connection ON erp_connector_service.inventory_snapshots(connection_id);
CREATE INDEX idx_is_snapshot_date ON erp_connector_service.inventory_snapshots(snapshot_date DESC);
CREATE INDEX idx_is_provenance_hash ON erp_connector_service.inventory_snapshots(provenance_hash);
CREATE INDEX idx_is_created_at ON erp_connector_service.inventory_snapshots(created_at DESC);
CREATE INDEX idx_is_tenant ON erp_connector_service.inventory_snapshots(tenant_id);
CREATE INDEX idx_is_tenant_material ON erp_connector_service.inventory_snapshots(tenant_id, material_id);
CREATE INDEX idx_is_metadata ON erp_connector_service.inventory_snapshots USING GIN (metadata);

-- sync_jobs indexes (hypertable-aware)
CREATE INDEX idx_sj_connection ON erp_connector_service.sync_jobs(connection_id, started_at DESC);
CREATE INDEX idx_sj_sync_mode ON erp_connector_service.sync_jobs(sync_mode, started_at DESC);
CREATE INDEX idx_sj_query_type ON erp_connector_service.sync_jobs(query_type, started_at DESC);
CREATE INDEX idx_sj_status ON erp_connector_service.sync_jobs(status, started_at DESC);
CREATE INDEX idx_sj_tenant ON erp_connector_service.sync_jobs(tenant_id, started_at DESC);
CREATE INDEX idx_sj_tenant_status ON erp_connector_service.sync_jobs(tenant_id, status, started_at DESC);
CREATE INDEX idx_sj_tenant_connection ON erp_connector_service.sync_jobs(tenant_id, connection_id, started_at DESC);
CREATE INDEX idx_sj_errors ON erp_connector_service.sync_jobs USING GIN (errors);

-- emission_calculations indexes
CREATE INDEX idx_ecalc_spend_record ON erp_connector_service.emission_calculations(spend_record_id);
CREATE INDEX idx_ecalc_vendor ON erp_connector_service.emission_calculations(vendor_id);
CREATE INDEX idx_ecalc_methodology ON erp_connector_service.emission_calculations(methodology);
CREATE INDEX idx_ecalc_scope3 ON erp_connector_service.emission_calculations(scope3_category);
CREATE INDEX idx_ecalc_emission_source ON erp_connector_service.emission_calculations(emission_factor_source);
CREATE INDEX idx_ecalc_connection ON erp_connector_service.emission_calculations(connection_id);
CREATE INDEX idx_ecalc_provenance_hash ON erp_connector_service.emission_calculations(provenance_hash);
CREATE INDEX idx_ecalc_created_at ON erp_connector_service.emission_calculations(created_at DESC);
CREATE INDEX idx_ecalc_tenant ON erp_connector_service.emission_calculations(tenant_id);
CREATE INDEX idx_ecalc_tenant_vendor ON erp_connector_service.emission_calculations(tenant_id, vendor_id);
CREATE INDEX idx_ecalc_tenant_scope3 ON erp_connector_service.emission_calculations(tenant_id, scope3_category);
CREATE INDEX idx_ecalc_metadata ON erp_connector_service.emission_calculations USING GIN (metadata);

-- erp_audit_log indexes (hypertable-aware)
CREATE INDEX idx_eal_entity_type ON erp_connector_service.erp_audit_log(entity_type, created_at DESC);
CREATE INDEX idx_eal_entity_id ON erp_connector_service.erp_audit_log(entity_id, created_at DESC);
CREATE INDEX idx_eal_operation ON erp_connector_service.erp_audit_log(operation, created_at DESC);
CREATE INDEX idx_eal_user ON erp_connector_service.erp_audit_log(user_id, created_at DESC);
CREATE INDEX idx_eal_tenant ON erp_connector_service.erp_audit_log(tenant_id, created_at DESC);
CREATE INDEX idx_eal_provenance_hash ON erp_connector_service.erp_audit_log(provenance_hash);
CREATE INDEX idx_eal_details ON erp_connector_service.erp_audit_log USING GIN (details);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE erp_connector_service.erp_connections ENABLE ROW LEVEL SECURITY;
CREATE POLICY ec_tenant_read ON erp_connector_service.erp_connections
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ec_tenant_write ON erp_connector_service.erp_connections
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE erp_connector_service.vendor_mappings ENABLE ROW LEVEL SECURITY;
CREATE POLICY vm_tenant_read ON erp_connector_service.vendor_mappings
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY vm_tenant_write ON erp_connector_service.vendor_mappings
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE erp_connector_service.material_mappings ENABLE ROW LEVEL SECURITY;
CREATE POLICY mm_tenant_read ON erp_connector_service.material_mappings
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY mm_tenant_write ON erp_connector_service.material_mappings
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE erp_connector_service.spend_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY sr_tenant_read ON erp_connector_service.spend_records
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY sr_tenant_write ON erp_connector_service.spend_records
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE erp_connector_service.purchase_orders ENABLE ROW LEVEL SECURITY;
CREATE POLICY po_tenant_read ON erp_connector_service.purchase_orders
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY po_tenant_write ON erp_connector_service.purchase_orders
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE erp_connector_service.purchase_order_lines ENABLE ROW LEVEL SECURITY;
CREATE POLICY pol_tenant_read ON erp_connector_service.purchase_order_lines
    FOR SELECT USING (TRUE);
CREATE POLICY pol_tenant_write ON erp_connector_service.purchase_order_lines
    FOR ALL USING (TRUE);

ALTER TABLE erp_connector_service.inventory_snapshots ENABLE ROW LEVEL SECURITY;
CREATE POLICY is_tenant_read ON erp_connector_service.inventory_snapshots
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY is_tenant_write ON erp_connector_service.inventory_snapshots
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE erp_connector_service.sync_jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY sj_tenant_read ON erp_connector_service.sync_jobs
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY sj_tenant_write ON erp_connector_service.sync_jobs
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE erp_connector_service.emission_calculations ENABLE ROW LEVEL SECURITY;
CREATE POLICY ecalc_tenant_read ON erp_connector_service.emission_calculations
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ecalc_tenant_write ON erp_connector_service.emission_calculations
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE erp_connector_service.erp_audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY eal_tenant_read ON erp_connector_service.erp_audit_log
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY eal_tenant_write ON erp_connector_service.erp_audit_log
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA erp_connector_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA erp_connector_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA erp_connector_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON erp_connector_service.hourly_spend_stats TO greenlang_app;
GRANT SELECT ON erp_connector_service.hourly_audit_stats TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA erp_connector_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA erp_connector_service TO greenlang_readonly;
GRANT SELECT ON erp_connector_service.hourly_spend_stats TO greenlang_readonly;
GRANT SELECT ON erp_connector_service.hourly_audit_stats TO greenlang_readonly;

-- Add ERP connector service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'erp_connector:connections:read', 'erp_connector', 'connections_read', 'View ERP connections and sync status'),
    (gen_random_uuid(), 'erp_connector:connections:write', 'erp_connector', 'connections_write', 'Create and manage ERP connections'),
    (gen_random_uuid(), 'erp_connector:vendors:read', 'erp_connector', 'vendors_read', 'View vendor mappings and emission factors'),
    (gen_random_uuid(), 'erp_connector:vendors:write', 'erp_connector', 'vendors_write', 'Create and manage vendor mappings'),
    (gen_random_uuid(), 'erp_connector:materials:read', 'erp_connector', 'materials_read', 'View material mappings and emission factors'),
    (gen_random_uuid(), 'erp_connector:materials:write', 'erp_connector', 'materials_write', 'Create and manage material mappings'),
    (gen_random_uuid(), 'erp_connector:spend:read', 'erp_connector', 'spend_read', 'View spend records and procurement transactions'),
    (gen_random_uuid(), 'erp_connector:spend:write', 'erp_connector', 'spend_write', 'Create and manage spend records'),
    (gen_random_uuid(), 'erp_connector:orders:read', 'erp_connector', 'orders_read', 'View purchase orders and line items'),
    (gen_random_uuid(), 'erp_connector:orders:write', 'erp_connector', 'orders_write', 'Create and manage purchase orders'),
    (gen_random_uuid(), 'erp_connector:sync:read', 'erp_connector', 'sync_read', 'View sync job status and history'),
    (gen_random_uuid(), 'erp_connector:sync:write', 'erp_connector', 'sync_write', 'Create and manage sync jobs'),
    (gen_random_uuid(), 'erp_connector:audit:read', 'erp_connector', 'audit_read', 'View ERP connector audit event log'),
    (gen_random_uuid(), 'erp_connector:admin', 'erp_connector', 'admin', 'ERP connector service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep spend records for 730 days (2 years)
SELECT add_retention_policy('erp_connector_service.spend_records', INTERVAL '730 days');

-- Keep sync jobs for 30 days
SELECT add_retention_policy('erp_connector_service.sync_jobs', INTERVAL '30 days');

-- Keep audit events for 365 days
SELECT add_retention_policy('erp_connector_service.erp_audit_log', INTERVAL '365 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on spend_records after 30 days
ALTER TABLE erp_connector_service.spend_records SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'transaction_date DESC'
);

SELECT add_compression_policy('erp_connector_service.spend_records', INTERVAL '30 days');

-- Enable compression on sync_jobs after 3 days
ALTER TABLE erp_connector_service.sync_jobs SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'started_at DESC'
);

SELECT add_compression_policy('erp_connector_service.sync_jobs', INTERVAL '3 days');

-- Enable compression on erp_audit_log after 30 days
ALTER TABLE erp_connector_service.erp_audit_log SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_compression_policy('erp_connector_service.erp_audit_log', INTERVAL '30 days');

-- =============================================================================
-- Seed: Register the ERP/Finance Connector Agent (GL-DATA-X-004) in Agent Registry
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES
('GL-DATA-X-004', 'ERP/Finance Connector',
 'Connects to enterprise ERP and finance systems (SAP S/4HANA, SAP ECC, Oracle ERP Cloud, Oracle EBS, Microsoft Dynamics 365, NetSuite, Workday, Sage Intacct, Xero, QuickBooks) to extract spend data, purchase orders, vendor master data, material master data, and inventory positions. Maps vendors and materials to GHG Protocol Scope 3 categories with EEIO emission factors. Provides spend-based and activity-based emission calculations with SHA-256 provenance hash chains for every synced record.',
 2, 'async', true, true, 5, '1.0.0', true,
 'GreenLang Data Team', 'https://docs.greenlang.ai/agents/erp-connector', true, 'default')
ON CONFLICT (agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Agent Version for ERP/Finance Connector
-- =============================================================================

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES
('GL-DATA-X-004', '1.0.0',
 '{"cpu_request": "250m", "cpu_limit": "1000m", "memory_request": "512Mi", "memory_limit": "1Gi", "gpu": false}'::jsonb,
 '{"image": "greenlang/erp-connector-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"data", "erp", "finance", "spend", "procurement", "scope3", "emissions"}',
 '{"cross-sector"}',
 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2')
ON CONFLICT (agent_id, version) DO NOTHING;

-- =============================================================================
-- Seed: Agent Capabilities for ERP/Finance Connector
-- =============================================================================

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES

('GL-DATA-X-004', '1.0.0', 'erp_connection_management', 'connectivity',
 'Create, update, test, and manage connections to enterprise ERP systems with health monitoring and automatic reconnection',
 '{"erp_system", "host", "port", "credentials"}', '{"connection_id", "status", "last_sync"}',
 '{"supported_systems": ["sap_s4hana", "sap_ecc", "oracle_erp_cloud", "oracle_ebs", "microsoft_dynamics_365", "netsuite", "workday", "sage_intacct", "xero", "quickbooks"], "health_check_interval": 60}'::jsonb),

('GL-DATA-X-004', '1.0.0', 'spend_data_sync', 'ingestion',
 'Synchronize spend and procurement transaction records from ERP systems with full/incremental/delta sync modes and currency conversion to USD',
 '{"connection_id", "sync_mode", "date_range"}', '{"records_synced", "records_skipped", "total_amount_usd"}',
 '{"sync_modes": ["full", "incremental", "delta"], "batch_size": 1000, "currency_conversion": true}'::jsonb),

('GL-DATA-X-004', '1.0.0', 'purchase_order_sync', 'ingestion',
 'Synchronize purchase order headers and line items from ERP systems with status tracking and Scope 3 category classification',
 '{"connection_id", "sync_mode"}', '{"orders_synced", "lines_synced"}',
 '{"include_line_items": true, "status_filter": ["open", "approved", "partially_received"]}'::jsonb),

('GL-DATA-X-004', '1.0.0', 'vendor_mapping', 'classification',
 'Map ERP vendor master data to GreenLang spend categories and assign EEIO emission factors (kgCO2e/USD) for spend-based Scope 3 calculations',
 '{"vendor_id", "vendor_name"}', '{"mapping_id", "spend_category", "emission_factor"}',
 '{"emission_factor_databases": ["EEIO", "EXIOBASE", "GaBi", "ecoinvent"], "auto_classify": true}'::jsonb),

('GL-DATA-X-004', '1.0.0', 'material_mapping', 'classification',
 'Map ERP material master data to GreenLang material groups and assign activity-based emission factors (kgCO2e/unit) for procurement emissions',
 '{"material_id", "material_name"}', '{"mapping_id", "material_group", "emission_factor"}',
 '{"emission_factor_databases": ["EEIO", "EXIOBASE", "GaBi", "ecoinvent"], "unit_normalization": true}'::jsonb),

('GL-DATA-X-004', '1.0.0', 'scope3_emission_calculation', 'computation',
 'Calculate Scope 3 emissions from spend records using vendor or material emission factors with spend-based, activity-based, or hybrid methodology',
 '{"spend_record_ids"}', '{"calculations", "total_kgco2e", "by_category"}',
 '{"methodologies": ["spend_based", "activity_based", "hybrid", "supplier_specific"], "ghg_protocol_compliant": true}'::jsonb),

('GL-DATA-X-004', '1.0.0', 'inventory_sync', 'ingestion',
 'Synchronize inventory position snapshots from ERP systems with material quantities, costs, and warehouse locations',
 '{"connection_id", "snapshot_date"}', '{"materials_synced", "total_value"}',
 '{"include_warehouse": true, "snapshot_frequency": "daily"}'::jsonb),

('GL-DATA-X-004', '1.0.0', 'currency_conversion', 'computation',
 'Convert spend amounts from source currencies to USD using configurable exchange rate providers for consistent emission factor application',
 '{"amount", "source_currency", "target_currency", "date"}', '{"converted_amount", "exchange_rate", "provider"}',
 '{"providers": ["ecb", "openexchangerates", "fixer", "manual"], "base_currency": "USD"}'::jsonb)

ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: Agent Dependencies for ERP/Finance Connector
-- =============================================================================

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES

-- ERP Connector depends on Schema Compiler for input/output validation
('GL-DATA-X-004', 'GL-FOUND-X-002', '>=1.0.0', false,
 'Spend records, purchase orders, and emission calculations are validated against JSON Schema definitions'),

-- ERP Connector depends on Unit Normalizer for unit conversion
('GL-DATA-X-004', 'GL-FOUND-X-003', '>=1.0.0', false,
 'Material quantities and emission factors are normalized to standard units for consistent calculations'),

-- ERP Connector depends on Registry for agent discovery
('GL-DATA-X-004', 'GL-FOUND-X-007', '>=1.0.0', false,
 'Agent version and capability lookup for sync pipeline orchestration'),

-- ERP Connector optionally uses Citations for provenance tracking
('GL-DATA-X-004', 'GL-FOUND-X-005', '>=1.0.0', true,
 'Spend data and emission calculation provenance chains are registered with the citation service for audit trail'),

-- ERP Connector optionally uses Reproducibility for determinism
('GL-DATA-X-004', 'GL-FOUND-X-008', '>=1.0.0', true,
 'Emission calculations are verified for reproducibility across re-sync runs')

ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Service Catalog Entry for ERP/Finance Connector
-- =============================================================================

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES
('GL-DATA-X-004', 'ERP/Finance Connector',
 'Multi-system ERP connector for spend data, purchase orders, vendor/material master data, and inventory. Maps procurement to GHG Protocol Scope 3 categories with EEIO emission factors. Supports SAP, Oracle, Dynamics 365, NetSuite, Workday, and more.',
 'data', 'active', 'default')
ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA erp_connector_service IS 'ERP/Finance Connector for GreenLang Climate OS (AGENT-DATA-003) - multi-system ERP integration, spend data sync, purchase order sync, vendor/material mapping, Scope 3 emission calculation, and provenance tracking';
COMMENT ON TABLE erp_connector_service.erp_connections IS 'ERP connection registry with system type, host, credentials, company code, sync status, and provenance hash';
COMMENT ON TABLE erp_connector_service.vendor_mappings IS 'Vendor classification mapping with spend categories and EEIO emission factors (kgCO2e/USD) for spend-based Scope 3 calculations';
COMMENT ON TABLE erp_connector_service.material_mappings IS 'Material classification mapping with material groups and activity-based emission factors (kgCO2e/unit) for procurement emissions';
COMMENT ON TABLE erp_connector_service.spend_records IS 'TimescaleDB hypertable: spend/procurement transaction records with vendor, amount, currency, Scope 3 category, and estimated emissions';
COMMENT ON TABLE erp_connector_service.purchase_orders IS 'Purchase order headers with vendor, dates, amounts, status, and Scope 3 category classification';
COMMENT ON TABLE erp_connector_service.purchase_order_lines IS 'Purchase order line items with material, quantity, unit price, cost center, and GL account';
COMMENT ON TABLE erp_connector_service.inventory_snapshots IS 'Inventory position snapshots with material quantities, costs, warehouse locations, and snapshot dates';
COMMENT ON TABLE erp_connector_service.sync_jobs IS 'TimescaleDB hypertable: ERP sync job execution records with mode, query type, status, records synced, and duration';
COMMENT ON TABLE erp_connector_service.emission_calculations IS 'Emission calculation results with spend-based/activity-based methodology, emission factors, and Scope 3 category';
COMMENT ON TABLE erp_connector_service.erp_audit_log IS 'TimescaleDB hypertable: comprehensive audit events for all ERP connector operations with provenance hash';
COMMENT ON MATERIALIZED VIEW erp_connector_service.hourly_spend_stats IS 'Continuous aggregate: hourly spend statistics by Scope 3 category, spend category, and currency for dashboard queries and SLI tracking';
COMMENT ON MATERIALIZED VIEW erp_connector_service.hourly_audit_stats IS 'Continuous aggregate: hourly audit event counts by entity type and operation for compliance reporting';

COMMENT ON COLUMN erp_connector_service.erp_connections.erp_system IS 'ERP system type: sap_s4hana, sap_ecc, oracle_erp_cloud, oracle_ebs, microsoft_dynamics_365, netsuite, workday, sage_intacct, xero, quickbooks, custom';
COMMENT ON COLUMN erp_connector_service.erp_connections.status IS 'Connection status: active, inactive, error, syncing, suspended, pending, disconnected';
COMMENT ON COLUMN erp_connector_service.vendor_mappings.emission_factor_kgco2e_per_dollar IS 'EEIO emission factor in kgCO2e per USD spent for spend-based Scope 3 calculations';
COMMENT ON COLUMN erp_connector_service.material_mappings.emission_factor_kgco2e_per_unit IS 'Activity-based emission factor in kgCO2e per material unit for procurement emission calculations';
COMMENT ON COLUMN erp_connector_service.spend_records.scope3_category IS 'GHG Protocol Scope 3 category: cat1 through cat15 or unclassified';
COMMENT ON COLUMN erp_connector_service.spend_records.estimated_emissions_kgco2e IS 'Estimated Scope 3 emissions in kgCO2e calculated from amount_usd * vendor emission factor';
COMMENT ON COLUMN erp_connector_service.sync_jobs.sync_mode IS 'Sync mode: full, incremental, delta, manual, scheduled';
COMMENT ON COLUMN erp_connector_service.sync_jobs.query_type IS 'Query type: spend, purchase_orders, purchase_order_lines, inventory, vendors, materials, gl_accounts, cost_centers, company_codes, all';
COMMENT ON COLUMN erp_connector_service.emission_calculations.methodology IS 'Emission calculation methodology: spend_based, activity_based, hybrid, supplier_specific, average_data, manual';
COMMENT ON COLUMN erp_connector_service.erp_audit_log.operation IS 'Audit operation: create, update, delete, sync_start, sync_complete, sync_fail, connect, disconnect, map_vendor, map_material, calculate_emissions, currency_convert, classify_spend';
