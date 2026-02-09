-- =============================================================================
-- V039: Spend Categorizer Service Schema
-- =============================================================================
-- Component: AGENT-DATA-009 (Spend Data Categorizer)
-- Agent ID:  GL-DATA-SUP-002
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Spend Data Categorizer Agent (GL-DATA-SUP-002) with capabilities
-- for spend record ingestion and normalization, multi-taxonomy classification
-- (UNSPSC, NAICS, eCl@ss, ISIC, SIC, CPV, HS/CN), GHG Protocol Scope 3
-- category assignment (categories 1-15), emission factor database management
-- (EPA EEIO, EXIOBASE, DEFRA, Ecoinvent), per-record emission calculations,
-- custom categorization rule engines, analytics snapshot aggregation,
-- and batch ingestion tracking with provenance chains.
-- =============================================================================
-- Tables (10):
--   1. spend_records              - Ingested and normalized spend records
--   2. taxonomy_mappings          - Taxonomy classification results per record
--   3. scope3_classifications     - Scope 3 category assignments per record
--   4. emission_factors           - Emission factor database (EPA EEIO, EXIOBASE, DEFRA, Ecoinvent, custom)
--   5. emission_calculations      - Per-record emission calculation results
--   6. category_rules             - Custom categorization rules with pattern matching
--   7. analytics_snapshots        - Periodic analytics aggregation snapshots
--   8. ingestion_batches          - Batch ingestion tracking with status lifecycle
--   9. categorization_events      - Categorization event time-series (hypertable)
--  10. emission_calculations_ts   - Emission calculation time-series (hypertable)
--
-- Additional Hypertable:
--  11. analytics_events           - Analytics event time-series (hypertable)
--
-- Continuous Aggregates (2):
--   1. hourly_categorization_stats - Hourly categorization event statistics
--   2. hourly_emission_stats       - Hourly emission calculation statistics
--
-- Also includes: 65+ indexes (B-tree, GIN), 20 RLS policies per tenant,
-- retention policies (90 days on hypertables), compression policies (7 days),
-- updated_at trigger, security permissions for gl_spend_categorizer_role,
-- seed emission factors (EPA EEIO, EXIOBASE, DEFRA), and agent registry
-- seed data registering GL-DATA-SUP-002.
-- Previous: V038__supplier_questionnaire_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS spend_categorizer_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================
-- Reusable trigger function for tables with updated_at columns.

CREATE OR REPLACE FUNCTION spend_categorizer_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: spend_categorizer_service.spend_records
-- =============================================================================
-- Ingested and normalized spend records. Each record captures vendor
-- details, transaction amount with multi-currency normalization to USD,
-- GL account and cost center classification, material grouping,
-- processing status lifecycle, and provenance hash. Tenant-scoped.

CREATE TABLE spend_categorizer_service.spend_records (
    record_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID NOT NULL,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    source VARCHAR(50) NOT NULL,
    vendor_id VARCHAR(200),
    vendor_name VARCHAR(500),
    normalized_vendor_name VARCHAR(500),
    description TEXT,
    amount DECIMAL(18,4) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    amount_usd DECIMAL(18,4),
    transaction_date DATE,
    cost_center VARCHAR(100),
    gl_account VARCHAR(100),
    department VARCHAR(200),
    material_group VARCHAR(200),
    status VARCHAR(30) NOT NULL DEFAULT 'raw',
    metadata JSONB DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Source constraint
ALTER TABLE spend_categorizer_service.spend_records
    ADD CONSTRAINT chk_sr_source
    CHECK (source IN ('erp_extract', 'csv_file', 'excel_file', 'api_feed', 'manual_entry'));

-- Status constraint
ALTER TABLE spend_categorizer_service.spend_records
    ADD CONSTRAINT chk_sr_status
    CHECK (status IN (
        'raw', 'normalized', 'classified', 'calculated', 'validated',
        'error', 'duplicate', 'excluded', 'archived'
    ));

-- Currency must be 3-character ISO code
ALTER TABLE spend_categorizer_service.spend_records
    ADD CONSTRAINT chk_sr_currency_length
    CHECK (LENGTH(TRIM(currency)) = 3);

-- Amount must not be zero (allow negative for credits/refunds)
ALTER TABLE spend_categorizer_service.spend_records
    ADD CONSTRAINT chk_sr_amount_not_zero
    CHECK (amount <> 0);

-- Provenance hash must not be empty
ALTER TABLE spend_categorizer_service.spend_records
    ADD CONSTRAINT chk_sr_provenance_hash_not_empty
    CHECK (LENGTH(TRIM(provenance_hash)) > 0);

-- Batch ID must not be null (enforced by NOT NULL, but semantic constraint)
ALTER TABLE spend_categorizer_service.spend_records
    ADD CONSTRAINT chk_sr_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- updated_at trigger
CREATE TRIGGER trg_sr_updated_at
    BEFORE UPDATE ON spend_categorizer_service.spend_records
    FOR EACH ROW
    EXECUTE FUNCTION spend_categorizer_service.set_updated_at();

-- =============================================================================
-- Table 2: spend_categorizer_service.taxonomy_mappings
-- =============================================================================
-- Taxonomy classification results linking spend records to industry
-- classification codes. Each mapping captures the taxonomy system
-- (UNSPSC, NAICS, eCl@ss, ISIC, SIC, CPV, HS/CN), hierarchical code
-- with level and parent, classification confidence, and method used.
-- Linked to spend_records via record_id.

CREATE TABLE spend_categorizer_service.taxonomy_mappings (
    mapping_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    record_id UUID NOT NULL,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    taxonomy_system VARCHAR(20) NOT NULL,
    taxonomy_code VARCHAR(20) NOT NULL,
    code_description TEXT,
    level INTEGER,
    parent_code VARCHAR(20),
    confidence DECIMAL(5,4),
    classification_method VARCHAR(50),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Foreign key to spend_records
ALTER TABLE spend_categorizer_service.taxonomy_mappings
    ADD CONSTRAINT fk_tm_record_id
    FOREIGN KEY (record_id) REFERENCES spend_categorizer_service.spend_records(record_id)
    ON DELETE CASCADE;

-- Taxonomy system constraint
ALTER TABLE spend_categorizer_service.taxonomy_mappings
    ADD CONSTRAINT chk_tm_taxonomy_system
    CHECK (taxonomy_system IN ('unspsc', 'naics', 'eclass', 'isic', 'sic', 'cpv', 'hs_cn'));

-- Taxonomy code must not be empty
ALTER TABLE spend_categorizer_service.taxonomy_mappings
    ADD CONSTRAINT chk_tm_taxonomy_code_not_empty
    CHECK (LENGTH(TRIM(taxonomy_code)) > 0);

-- Confidence must be between 0 and 1 if specified
ALTER TABLE spend_categorizer_service.taxonomy_mappings
    ADD CONSTRAINT chk_tm_confidence_range
    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1));

-- Level must be positive if specified
ALTER TABLE spend_categorizer_service.taxonomy_mappings
    ADD CONSTRAINT chk_tm_level_positive
    CHECK (level IS NULL OR level > 0);

-- Provenance hash must not be empty
ALTER TABLE spend_categorizer_service.taxonomy_mappings
    ADD CONSTRAINT chk_tm_provenance_hash_not_empty
    CHECK (LENGTH(TRIM(provenance_hash)) > 0);

-- Tenant ID must not be empty
ALTER TABLE spend_categorizer_service.taxonomy_mappings
    ADD CONSTRAINT chk_tm_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- updated_at trigger
CREATE TRIGGER trg_tm_updated_at
    BEFORE UPDATE ON spend_categorizer_service.taxonomy_mappings
    FOR EACH ROW
    EXECUTE FUNCTION spend_categorizer_service.set_updated_at();

-- =============================================================================
-- Table 3: spend_categorizer_service.scope3_classifications
-- =============================================================================
-- Scope 3 category assignments for spend records following the GHG
-- Protocol Corporate Value Chain (Scope 3) Standard. Each classification
-- captures the Scope 3 category (1-15), confidence, mapping rule used,
-- capital expenditure flag, and split percentage for multi-category
-- allocation. Linked to spend_records via record_id.

CREATE TABLE spend_categorizer_service.scope3_classifications (
    classification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    record_id UUID NOT NULL,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    scope3_category VARCHAR(50) NOT NULL,
    category_number INTEGER,
    confidence DECIMAL(5,4),
    mapping_rule TEXT,
    is_capex BOOLEAN DEFAULT FALSE,
    split_pct DECIMAL(5,4) DEFAULT 1.0,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Foreign key to spend_records
ALTER TABLE spend_categorizer_service.scope3_classifications
    ADD CONSTRAINT fk_s3c_record_id
    FOREIGN KEY (record_id) REFERENCES spend_categorizer_service.spend_records(record_id)
    ON DELETE CASCADE;

-- Scope 3 category constraint
ALTER TABLE spend_categorizer_service.scope3_classifications
    ADD CONSTRAINT chk_s3c_scope3_category
    CHECK (scope3_category IN (
        'cat1_purchased_goods', 'cat2_capital_goods', 'cat3_fuel_energy',
        'cat4_upstream_transport', 'cat5_waste', 'cat6_business_travel',
        'cat7_employee_commuting', 'cat8_upstream_leased', 'cat9_downstream_transport',
        'cat10_processing', 'cat11_use_of_sold', 'cat12_end_of_life',
        'cat13_downstream_leased', 'cat14_franchises', 'cat15_investments',
        'unclassified'
    ));

-- Category number must be between 1 and 15
ALTER TABLE spend_categorizer_service.scope3_classifications
    ADD CONSTRAINT chk_s3c_category_number_range
    CHECK (category_number IS NULL OR (category_number >= 1 AND category_number <= 15));

-- Confidence must be between 0 and 1 if specified
ALTER TABLE spend_categorizer_service.scope3_classifications
    ADD CONSTRAINT chk_s3c_confidence_range
    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1));

-- Split percentage must be between 0 and 1
ALTER TABLE spend_categorizer_service.scope3_classifications
    ADD CONSTRAINT chk_s3c_split_pct_range
    CHECK (split_pct >= 0 AND split_pct <= 1);

-- Provenance hash must not be empty
ALTER TABLE spend_categorizer_service.scope3_classifications
    ADD CONSTRAINT chk_s3c_provenance_hash_not_empty
    CHECK (LENGTH(TRIM(provenance_hash)) > 0);

-- Tenant ID must not be empty
ALTER TABLE spend_categorizer_service.scope3_classifications
    ADD CONSTRAINT chk_s3c_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 4: spend_categorizer_service.emission_factors
-- =============================================================================
-- Emission factor database storing factors from multiple authoritative
-- sources (EPA EEIO, EXIOBASE, DEFRA, Ecoinvent, custom). Each factor
-- captures the taxonomy mapping (NAICS, UNSPSC), geographic region,
-- factor value with units, reference year, version, and descriptive
-- metadata. Used for deterministic spend-based emission calculations.

CREATE TABLE spend_categorizer_service.emission_factors (
    factor_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source VARCHAR(50) NOT NULL,
    taxonomy_code VARCHAR(20),
    taxonomy_system VARCHAR(20),
    naics_code VARCHAR(10),
    unspsc_code VARCHAR(20),
    region VARCHAR(10) DEFAULT 'global',
    factor_value DECIMAL(18,8) NOT NULL,
    factor_unit VARCHAR(30) NOT NULL,
    year INTEGER NOT NULL,
    version VARCHAR(20),
    description TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Source constraint
ALTER TABLE spend_categorizer_service.emission_factors
    ADD CONSTRAINT chk_ef_source
    CHECK (source IN ('epa_eeio', 'exiobase', 'defra', 'ecoinvent', 'custom'));

-- Factor value must be positive
ALTER TABLE spend_categorizer_service.emission_factors
    ADD CONSTRAINT chk_ef_factor_value_positive
    CHECK (factor_value > 0);

-- Factor unit must not be empty
ALTER TABLE spend_categorizer_service.emission_factors
    ADD CONSTRAINT chk_ef_factor_unit_not_empty
    CHECK (LENGTH(TRIM(factor_unit)) > 0);

-- Year must be reasonable (1990-2100)
ALTER TABLE spend_categorizer_service.emission_factors
    ADD CONSTRAINT chk_ef_year_range
    CHECK (year >= 1990 AND year <= 2100);

-- Region must not be empty
ALTER TABLE spend_categorizer_service.emission_factors
    ADD CONSTRAINT chk_ef_region_not_empty
    CHECK (LENGTH(TRIM(region)) > 0);

-- updated_at trigger
CREATE TRIGGER trg_ef_updated_at
    BEFORE UPDATE ON spend_categorizer_service.emission_factors
    FOR EACH ROW
    EXECUTE FUNCTION spend_categorizer_service.set_updated_at();

-- =============================================================================
-- Table 5: spend_categorizer_service.emission_calculations
-- =============================================================================
-- Per-record emission calculation results linking spend records to
-- emission factors. Each calculation captures the spend amount in USD,
-- calculated emissions in both kgCO2e and tCO2e, calculation method,
-- data quality score, and provenance hash. Linked to spend_records
-- via record_id and emission_factors via factor_id.

CREATE TABLE spend_categorizer_service.emission_calculations (
    calculation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    record_id UUID NOT NULL,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    factor_id UUID NOT NULL,
    spend_usd DECIMAL(18,4),
    emissions_kgco2e DECIMAL(18,6),
    emissions_tco2e DECIMAL(18,9),
    calculation_method VARCHAR(50),
    data_quality_score DECIMAL(5,4),
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Foreign key to spend_records
ALTER TABLE spend_categorizer_service.emission_calculations
    ADD CONSTRAINT fk_ec_record_id
    FOREIGN KEY (record_id) REFERENCES spend_categorizer_service.spend_records(record_id)
    ON DELETE CASCADE;

-- Foreign key to emission_factors
ALTER TABLE spend_categorizer_service.emission_calculations
    ADD CONSTRAINT fk_ec_factor_id
    FOREIGN KEY (factor_id) REFERENCES spend_categorizer_service.emission_factors(factor_id)
    ON DELETE RESTRICT;

-- Emissions must be non-negative if specified
ALTER TABLE spend_categorizer_service.emission_calculations
    ADD CONSTRAINT chk_ec_emissions_kgco2e_non_negative
    CHECK (emissions_kgco2e IS NULL OR emissions_kgco2e >= 0);

-- Emissions tCO2e must be non-negative if specified
ALTER TABLE spend_categorizer_service.emission_calculations
    ADD CONSTRAINT chk_ec_emissions_tco2e_non_negative
    CHECK (emissions_tco2e IS NULL OR emissions_tco2e >= 0);

-- Data quality score must be between 0 and 1 if specified
ALTER TABLE spend_categorizer_service.emission_calculations
    ADD CONSTRAINT chk_ec_data_quality_score_range
    CHECK (data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 1));

-- Provenance hash must not be empty
ALTER TABLE spend_categorizer_service.emission_calculations
    ADD CONSTRAINT chk_ec_provenance_hash_not_empty
    CHECK (LENGTH(TRIM(provenance_hash)) > 0);

-- Tenant ID must not be empty
ALTER TABLE spend_categorizer_service.emission_calculations
    ADD CONSTRAINT chk_ec_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 6: spend_categorizer_service.category_rules
-- =============================================================================
-- Custom categorization rules enabling tenant-specific pattern matching
-- for automated spend classification. Each rule captures the match type
-- (exact, contains, regex, fuzzy, starts_with, ends_with), match field,
-- pattern, target taxonomy and Scope 3 category, priority ordering,
-- activation state, and match count tracking. Tenant-scoped.

CREATE TABLE spend_categorizer_service.category_rules (
    rule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    name VARCHAR(200) NOT NULL,
    description TEXT,
    match_type VARCHAR(20) NOT NULL,
    match_field VARCHAR(50) NOT NULL,
    pattern TEXT NOT NULL,
    target_taxonomy VARCHAR(20),
    target_code VARCHAR(20),
    target_scope3_category VARCHAR(50),
    priority INTEGER DEFAULT 100,
    is_active BOOLEAN DEFAULT TRUE,
    match_count INTEGER DEFAULT 0,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Name must not be empty
ALTER TABLE spend_categorizer_service.category_rules
    ADD CONSTRAINT chk_cr_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- Match type constraint
ALTER TABLE spend_categorizer_service.category_rules
    ADD CONSTRAINT chk_cr_match_type
    CHECK (match_type IN ('exact', 'contains', 'regex', 'fuzzy', 'starts_with', 'ends_with'));

-- Match field constraint
ALTER TABLE spend_categorizer_service.category_rules
    ADD CONSTRAINT chk_cr_match_field
    CHECK (match_field IN ('description', 'vendor_name', 'gl_account', 'cost_center', 'material_group'));

-- Pattern must not be empty
ALTER TABLE spend_categorizer_service.category_rules
    ADD CONSTRAINT chk_cr_pattern_not_empty
    CHECK (LENGTH(TRIM(pattern)) > 0);

-- Priority must be non-negative
ALTER TABLE spend_categorizer_service.category_rules
    ADD CONSTRAINT chk_cr_priority_non_negative
    CHECK (priority >= 0);

-- Match count must be non-negative
ALTER TABLE spend_categorizer_service.category_rules
    ADD CONSTRAINT chk_cr_match_count_non_negative
    CHECK (match_count >= 0);

-- Target taxonomy constraint if specified
ALTER TABLE spend_categorizer_service.category_rules
    ADD CONSTRAINT chk_cr_target_taxonomy
    CHECK (target_taxonomy IS NULL OR target_taxonomy IN ('unspsc', 'naics', 'eclass', 'isic', 'sic', 'cpv', 'hs_cn'));

-- Target scope3 category constraint if specified
ALTER TABLE spend_categorizer_service.category_rules
    ADD CONSTRAINT chk_cr_target_scope3_category
    CHECK (target_scope3_category IS NULL OR target_scope3_category IN (
        'cat1_purchased_goods', 'cat2_capital_goods', 'cat3_fuel_energy',
        'cat4_upstream_transport', 'cat5_waste', 'cat6_business_travel',
        'cat7_employee_commuting', 'cat8_upstream_leased', 'cat9_downstream_transport',
        'cat10_processing', 'cat11_use_of_sold', 'cat12_end_of_life',
        'cat13_downstream_leased', 'cat14_franchises', 'cat15_investments',
        'unclassified'
    ));

-- Provenance hash must not be empty
ALTER TABLE spend_categorizer_service.category_rules
    ADD CONSTRAINT chk_cr_provenance_hash_not_empty
    CHECK (LENGTH(TRIM(provenance_hash)) > 0);

-- Tenant ID must not be empty
ALTER TABLE spend_categorizer_service.category_rules
    ADD CONSTRAINT chk_cr_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- updated_at trigger
CREATE TRIGGER trg_cr_updated_at
    BEFORE UPDATE ON spend_categorizer_service.category_rules
    FOR EACH ROW
    EXECUTE FUNCTION spend_categorizer_service.set_updated_at();

-- =============================================================================
-- Table 7: spend_categorizer_service.analytics_snapshots
-- =============================================================================
-- Periodic analytics aggregation snapshots capturing spend and emission
-- summaries for dashboard display and trend analysis. Each snapshot
-- captures the reporting period, total spend in USD, total emissions
-- in tCO2e, record count, and JSONB breakdowns by Scope 3 category,
-- taxonomy category, vendor, and hotspot identification. Tenant-scoped.

CREATE TABLE spend_categorizer_service.analytics_snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    total_spend_usd DECIMAL(18,4),
    total_emissions_tco2e DECIMAL(18,9),
    records_count INTEGER,
    scope3_breakdown JSONB DEFAULT '{}'::jsonb,
    category_breakdown JSONB DEFAULT '{}'::jsonb,
    vendor_breakdown JSONB DEFAULT '{}'::jsonb,
    hotspots JSONB DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Period end must be after period start
ALTER TABLE spend_categorizer_service.analytics_snapshots
    ADD CONSTRAINT chk_as_period_valid
    CHECK (period_end >= period_start);

-- Total spend must be non-negative if specified
ALTER TABLE spend_categorizer_service.analytics_snapshots
    ADD CONSTRAINT chk_as_total_spend_non_negative
    CHECK (total_spend_usd IS NULL OR total_spend_usd >= 0);

-- Total emissions must be non-negative if specified
ALTER TABLE spend_categorizer_service.analytics_snapshots
    ADD CONSTRAINT chk_as_total_emissions_non_negative
    CHECK (total_emissions_tco2e IS NULL OR total_emissions_tco2e >= 0);

-- Records count must be non-negative if specified
ALTER TABLE spend_categorizer_service.analytics_snapshots
    ADD CONSTRAINT chk_as_records_count_non_negative
    CHECK (records_count IS NULL OR records_count >= 0);

-- Provenance hash must not be empty
ALTER TABLE spend_categorizer_service.analytics_snapshots
    ADD CONSTRAINT chk_as_provenance_hash_not_empty
    CHECK (LENGTH(TRIM(provenance_hash)) > 0);

-- Tenant ID must not be empty
ALTER TABLE spend_categorizer_service.analytics_snapshots
    ADD CONSTRAINT chk_as_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 8: spend_categorizer_service.ingestion_batches
-- =============================================================================
-- Batch ingestion tracking records capturing the lifecycle of spend
-- data import operations. Each batch captures the source, file name,
-- record counts (total, processed, error), processing status lifecycle,
-- error details, and provenance hash. Tenant-scoped.

CREATE TABLE spend_categorizer_service.ingestion_batches (
    batch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    source VARCHAR(50) NOT NULL,
    file_name VARCHAR(500),
    total_records INTEGER,
    processed_records INTEGER,
    error_records INTEGER,
    status VARCHAR(30) NOT NULL DEFAULT 'pending',
    error_details JSONB DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Source constraint
ALTER TABLE spend_categorizer_service.ingestion_batches
    ADD CONSTRAINT chk_ib_source
    CHECK (source IN ('erp_extract', 'csv_file', 'excel_file', 'api_feed', 'manual_entry'));

-- Status constraint
ALTER TABLE spend_categorizer_service.ingestion_batches
    ADD CONSTRAINT chk_ib_status
    CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled', 'partial'));

-- Total records must be non-negative if specified
ALTER TABLE spend_categorizer_service.ingestion_batches
    ADD CONSTRAINT chk_ib_total_records_non_negative
    CHECK (total_records IS NULL OR total_records >= 0);

-- Processed records must be non-negative if specified
ALTER TABLE spend_categorizer_service.ingestion_batches
    ADD CONSTRAINT chk_ib_processed_records_non_negative
    CHECK (processed_records IS NULL OR processed_records >= 0);

-- Error records must be non-negative if specified
ALTER TABLE spend_categorizer_service.ingestion_batches
    ADD CONSTRAINT chk_ib_error_records_non_negative
    CHECK (error_records IS NULL OR error_records >= 0);

-- Provenance hash must not be empty
ALTER TABLE spend_categorizer_service.ingestion_batches
    ADD CONSTRAINT chk_ib_provenance_hash_not_empty
    CHECK (LENGTH(TRIM(provenance_hash)) > 0);

-- Tenant ID must not be empty
ALTER TABLE spend_categorizer_service.ingestion_batches
    ADD CONSTRAINT chk_ib_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 9: spend_categorizer_service.categorization_events (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording categorization lifecycle events as a
-- time-series. Each event captures the record reference, event type
-- (ingested, classified, mapped, calculated, rule_matched), taxonomy
-- system and code applied, Scope 3 category, confidence, processing
-- duration, and metadata. Partitioned by event_time for time-series
-- queries. Retained for 90 days with compression after 7 days.

CREATE TABLE spend_categorizer_service.categorization_events (
    event_id UUID DEFAULT gen_random_uuid(),
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    record_id UUID,
    event_type VARCHAR(50) NOT NULL,
    taxonomy_system VARCHAR(20),
    taxonomy_code VARCHAR(20),
    scope3_category VARCHAR(50),
    confidence DECIMAL(5,4),
    processing_ms DECIMAL(10,2),
    metadata JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (event_id, event_time)
);

-- Create hypertable partitioned by event_time with 7-day chunks
SELECT create_hypertable('spend_categorizer_service.categorization_events', 'event_time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Event type constraint
ALTER TABLE spend_categorizer_service.categorization_events
    ADD CONSTRAINT chk_ce_event_type
    CHECK (event_type IN (
        'ingested', 'classified', 'mapped', 'calculated', 'rule_matched',
        'normalized', 'validated', 'error', 'reclassified', 'split'
    ));

-- Taxonomy system constraint if specified
ALTER TABLE spend_categorizer_service.categorization_events
    ADD CONSTRAINT chk_ce_taxonomy_system
    CHECK (taxonomy_system IS NULL OR taxonomy_system IN ('unspsc', 'naics', 'eclass', 'isic', 'sic', 'cpv', 'hs_cn'));

-- Confidence must be between 0 and 1 if specified
ALTER TABLE spend_categorizer_service.categorization_events
    ADD CONSTRAINT chk_ce_confidence_range
    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1));

-- Processing time must be non-negative if specified
ALTER TABLE spend_categorizer_service.categorization_events
    ADD CONSTRAINT chk_ce_processing_ms_non_negative
    CHECK (processing_ms IS NULL OR processing_ms >= 0);

-- Tenant ID must not be empty
ALTER TABLE spend_categorizer_service.categorization_events
    ADD CONSTRAINT chk_ce_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 10: spend_categorizer_service.emission_calculations_ts (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording emission calculation events as a
-- time-series. Each event captures the record reference, factor source,
-- spend amount, calculated emissions, factor value, and processing
-- duration. Partitioned by event_time for time-series queries.
-- Retained for 90 days with compression after 7 days.

CREATE TABLE spend_categorizer_service.emission_calculations_ts (
    event_id UUID DEFAULT gen_random_uuid(),
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    record_id UUID,
    factor_source VARCHAR(50),
    spend_usd DECIMAL(18,4),
    emissions_kgco2e DECIMAL(18,6),
    factor_value DECIMAL(18,8),
    processing_ms DECIMAL(10,2),
    PRIMARY KEY (event_id, event_time)
);

-- Create hypertable partitioned by event_time with 7-day chunks
SELECT create_hypertable('spend_categorizer_service.emission_calculations_ts', 'event_time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Factor source constraint if specified
ALTER TABLE spend_categorizer_service.emission_calculations_ts
    ADD CONSTRAINT chk_ects_factor_source
    CHECK (factor_source IS NULL OR factor_source IN ('epa_eeio', 'exiobase', 'defra', 'ecoinvent', 'custom'));

-- Emissions must be non-negative if specified
ALTER TABLE spend_categorizer_service.emission_calculations_ts
    ADD CONSTRAINT chk_ects_emissions_non_negative
    CHECK (emissions_kgco2e IS NULL OR emissions_kgco2e >= 0);

-- Factor value must be positive if specified
ALTER TABLE spend_categorizer_service.emission_calculations_ts
    ADD CONSTRAINT chk_ects_factor_value_positive
    CHECK (factor_value IS NULL OR factor_value > 0);

-- Processing time must be non-negative if specified
ALTER TABLE spend_categorizer_service.emission_calculations_ts
    ADD CONSTRAINT chk_ects_processing_ms_non_negative
    CHECK (processing_ms IS NULL OR processing_ms >= 0);

-- Tenant ID must not be empty
ALTER TABLE spend_categorizer_service.emission_calculations_ts
    ADD CONSTRAINT chk_ects_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 11: spend_categorizer_service.analytics_events (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording analytics events as a time-series.
-- Each event captures the event type (aggregation, hotspot, trend,
-- report), Scope 3 category, spend and emission values, and metadata.
-- Partitioned by event_time for time-series queries. Retained for
-- 90 days with compression after 7 days.

CREATE TABLE spend_categorizer_service.analytics_events (
    event_id UUID DEFAULT gen_random_uuid(),
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    event_type VARCHAR(50) NOT NULL,
    scope3_category VARCHAR(50),
    spend_usd DECIMAL(18,4),
    emissions_kgco2e DECIMAL(18,6),
    metadata JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (event_id, event_time)
);

-- Create hypertable partitioned by event_time with 7-day chunks
SELECT create_hypertable('spend_categorizer_service.analytics_events', 'event_time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Event type constraint
ALTER TABLE spend_categorizer_service.analytics_events
    ADD CONSTRAINT chk_ae_event_type
    CHECK (event_type IN ('aggregation', 'hotspot', 'trend', 'report', 'benchmark', 'threshold_breach'));

-- Scope 3 category constraint if specified
ALTER TABLE spend_categorizer_service.analytics_events
    ADD CONSTRAINT chk_ae_scope3_category
    CHECK (scope3_category IS NULL OR scope3_category IN (
        'cat1_purchased_goods', 'cat2_capital_goods', 'cat3_fuel_energy',
        'cat4_upstream_transport', 'cat5_waste', 'cat6_business_travel',
        'cat7_employee_commuting', 'cat8_upstream_leased', 'cat9_downstream_transport',
        'cat10_processing', 'cat11_use_of_sold', 'cat12_end_of_life',
        'cat13_downstream_leased', 'cat14_franchises', 'cat15_investments',
        'unclassified'
    ));

-- Tenant ID must not be empty
ALTER TABLE spend_categorizer_service.analytics_events
    ADD CONSTRAINT chk_ae_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Continuous Aggregate: spend_categorizer_service.hourly_categorization_stats
-- =============================================================================
-- Precomputed hourly categorization event statistics by event type and
-- taxonomy system for dashboard queries, classification monitoring,
-- and confidence trend analysis.

CREATE MATERIALIZED VIEW spend_categorizer_service.hourly_categorization_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', event_time) AS bucket,
    event_type,
    COUNT(*) AS total_events,
    COUNT(DISTINCT record_id) AS unique_records,
    COUNT(*) FILTER (WHERE taxonomy_system = 'unspsc') AS unspsc_count,
    COUNT(*) FILTER (WHERE taxonomy_system = 'naics') AS naics_count,
    COUNT(*) FILTER (WHERE taxonomy_system = 'eclass') AS eclass_count,
    COUNT(*) FILTER (WHERE taxonomy_system = 'isic') AS isic_count,
    COUNT(*) FILTER (WHERE taxonomy_system = 'sic') AS sic_count,
    COUNT(*) FILTER (WHERE taxonomy_system = 'cpv') AS cpv_count,
    COUNT(*) FILTER (WHERE taxonomy_system = 'hs_cn') AS hs_cn_count,
    AVG(confidence) AS avg_confidence,
    AVG(processing_ms) AS avg_processing_ms
FROM spend_categorizer_service.categorization_events
WHERE event_time IS NOT NULL
GROUP BY bucket, event_type
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('spend_categorizer_service.hourly_categorization_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Continuous Aggregate: spend_categorizer_service.hourly_emission_stats
-- =============================================================================
-- Precomputed hourly emission calculation statistics for dashboard
-- queries, emission trend monitoring, and factor utilization analysis.

CREATE MATERIALIZED VIEW spend_categorizer_service.hourly_emission_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', event_time) AS bucket,
    factor_source,
    SUM(spend_usd) AS total_spend_usd,
    SUM(emissions_kgco2e) AS total_emissions_kgco2e,
    COUNT(*) AS total_calculations,
    COUNT(DISTINCT record_id) AS unique_records,
    AVG(factor_value) AS avg_factor_value,
    AVG(processing_ms) AS avg_processing_ms
FROM spend_categorizer_service.emission_calculations_ts
WHERE event_time IS NOT NULL
GROUP BY bucket, factor_source
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('spend_categorizer_service.hourly_emission_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- spend_records indexes
CREATE INDEX idx_sr_record_id ON spend_categorizer_service.spend_records(record_id);
CREATE INDEX idx_sr_batch_id ON spend_categorizer_service.spend_records(batch_id);
CREATE INDEX idx_sr_tenant_id ON spend_categorizer_service.spend_records(tenant_id);
CREATE INDEX idx_sr_source ON spend_categorizer_service.spend_records(source);
CREATE INDEX idx_sr_vendor_id ON spend_categorizer_service.spend_records(vendor_id);
CREATE INDEX idx_sr_vendor_name ON spend_categorizer_service.spend_records(vendor_name);
CREATE INDEX idx_sr_normalized_vendor ON spend_categorizer_service.spend_records(normalized_vendor_name);
CREATE INDEX idx_sr_currency ON spend_categorizer_service.spend_records(currency);
CREATE INDEX idx_sr_transaction_date ON spend_categorizer_service.spend_records(transaction_date);
CREATE INDEX idx_sr_cost_center ON spend_categorizer_service.spend_records(cost_center);
CREATE INDEX idx_sr_gl_account ON spend_categorizer_service.spend_records(gl_account);
CREATE INDEX idx_sr_department ON spend_categorizer_service.spend_records(department);
CREATE INDEX idx_sr_material_group ON spend_categorizer_service.spend_records(material_group);
CREATE INDEX idx_sr_status ON spend_categorizer_service.spend_records(status);
CREATE INDEX idx_sr_created_at ON spend_categorizer_service.spend_records(created_at DESC);
CREATE INDEX idx_sr_updated_at ON spend_categorizer_service.spend_records(updated_at DESC);
CREATE INDEX idx_sr_provenance ON spend_categorizer_service.spend_records(provenance_hash);
CREATE INDEX idx_sr_tenant_status ON spend_categorizer_service.spend_records(tenant_id, status);
CREATE INDEX idx_sr_tenant_source ON spend_categorizer_service.spend_records(tenant_id, source);
CREATE INDEX idx_sr_tenant_vendor ON spend_categorizer_service.spend_records(tenant_id, vendor_id);
CREATE INDEX idx_sr_tenant_date ON spend_categorizer_service.spend_records(tenant_id, transaction_date);
CREATE INDEX idx_sr_tenant_batch ON spend_categorizer_service.spend_records(tenant_id, batch_id);
CREATE INDEX idx_sr_metadata ON spend_categorizer_service.spend_records USING GIN (metadata);

-- taxonomy_mappings indexes
CREATE INDEX idx_tm_mapping_id ON spend_categorizer_service.taxonomy_mappings(mapping_id);
CREATE INDEX idx_tm_record_id ON spend_categorizer_service.taxonomy_mappings(record_id);
CREATE INDEX idx_tm_tenant_id ON spend_categorizer_service.taxonomy_mappings(tenant_id);
CREATE INDEX idx_tm_taxonomy_system ON spend_categorizer_service.taxonomy_mappings(taxonomy_system);
CREATE INDEX idx_tm_taxonomy_code ON spend_categorizer_service.taxonomy_mappings(taxonomy_code);
CREATE INDEX idx_tm_confidence ON spend_categorizer_service.taxonomy_mappings(confidence DESC);
CREATE INDEX idx_tm_classification_method ON spend_categorizer_service.taxonomy_mappings(classification_method);
CREATE INDEX idx_tm_provenance ON spend_categorizer_service.taxonomy_mappings(provenance_hash);
CREATE INDEX idx_tm_created_at ON spend_categorizer_service.taxonomy_mappings(created_at DESC);
CREATE INDEX idx_tm_updated_at ON spend_categorizer_service.taxonomy_mappings(updated_at DESC);
CREATE INDEX idx_tm_record_taxonomy ON spend_categorizer_service.taxonomy_mappings(record_id, taxonomy_system);
CREATE INDEX idx_tm_tenant_taxonomy ON spend_categorizer_service.taxonomy_mappings(tenant_id, taxonomy_system);
CREATE INDEX idx_tm_tenant_code ON spend_categorizer_service.taxonomy_mappings(tenant_id, taxonomy_code);
CREATE INDEX idx_tm_system_code ON spend_categorizer_service.taxonomy_mappings(taxonomy_system, taxonomy_code);
CREATE INDEX idx_tm_parent_code ON spend_categorizer_service.taxonomy_mappings(parent_code);

-- scope3_classifications indexes
CREATE INDEX idx_s3c_classification_id ON spend_categorizer_service.scope3_classifications(classification_id);
CREATE INDEX idx_s3c_record_id ON spend_categorizer_service.scope3_classifications(record_id);
CREATE INDEX idx_s3c_tenant_id ON spend_categorizer_service.scope3_classifications(tenant_id);
CREATE INDEX idx_s3c_scope3_category ON spend_categorizer_service.scope3_classifications(scope3_category);
CREATE INDEX idx_s3c_category_number ON spend_categorizer_service.scope3_classifications(category_number);
CREATE INDEX idx_s3c_confidence ON spend_categorizer_service.scope3_classifications(confidence DESC);
CREATE INDEX idx_s3c_is_capex ON spend_categorizer_service.scope3_classifications(is_capex);
CREATE INDEX idx_s3c_provenance ON spend_categorizer_service.scope3_classifications(provenance_hash);
CREATE INDEX idx_s3c_created_at ON spend_categorizer_service.scope3_classifications(created_at DESC);
CREATE INDEX idx_s3c_tenant_category ON spend_categorizer_service.scope3_classifications(tenant_id, scope3_category);
CREATE INDEX idx_s3c_record_category ON spend_categorizer_service.scope3_classifications(record_id, scope3_category);
CREATE INDEX idx_s3c_tenant_capex ON spend_categorizer_service.scope3_classifications(tenant_id, is_capex);

-- emission_factors indexes
CREATE INDEX idx_ef_factor_id ON spend_categorizer_service.emission_factors(factor_id);
CREATE INDEX idx_ef_source ON spend_categorizer_service.emission_factors(source);
CREATE INDEX idx_ef_taxonomy_code ON spend_categorizer_service.emission_factors(taxonomy_code);
CREATE INDEX idx_ef_taxonomy_system ON spend_categorizer_service.emission_factors(taxonomy_system);
CREATE INDEX idx_ef_naics_code ON spend_categorizer_service.emission_factors(naics_code);
CREATE INDEX idx_ef_unspsc_code ON spend_categorizer_service.emission_factors(unspsc_code);
CREATE INDEX idx_ef_region ON spend_categorizer_service.emission_factors(region);
CREATE INDEX idx_ef_year ON spend_categorizer_service.emission_factors(year);
CREATE INDEX idx_ef_version ON spend_categorizer_service.emission_factors(version);
CREATE INDEX idx_ef_created_at ON spend_categorizer_service.emission_factors(created_at DESC);
CREATE INDEX idx_ef_updated_at ON spend_categorizer_service.emission_factors(updated_at DESC);
CREATE INDEX idx_ef_source_naics ON spend_categorizer_service.emission_factors(source, naics_code);
CREATE INDEX idx_ef_source_unspsc ON spend_categorizer_service.emission_factors(source, unspsc_code);
CREATE INDEX idx_ef_source_region ON spend_categorizer_service.emission_factors(source, region);
CREATE INDEX idx_ef_source_year ON spend_categorizer_service.emission_factors(source, year);
CREATE INDEX idx_ef_naics_region ON spend_categorizer_service.emission_factors(naics_code, region);
CREATE INDEX idx_ef_metadata ON spend_categorizer_service.emission_factors USING GIN (metadata);

-- emission_calculations indexes
CREATE INDEX idx_ec_calculation_id ON spend_categorizer_service.emission_calculations(calculation_id);
CREATE INDEX idx_ec_record_id ON spend_categorizer_service.emission_calculations(record_id);
CREATE INDEX idx_ec_tenant_id ON spend_categorizer_service.emission_calculations(tenant_id);
CREATE INDEX idx_ec_factor_id ON spend_categorizer_service.emission_calculations(factor_id);
CREATE INDEX idx_ec_calculation_method ON spend_categorizer_service.emission_calculations(calculation_method);
CREATE INDEX idx_ec_data_quality ON spend_categorizer_service.emission_calculations(data_quality_score DESC);
CREATE INDEX idx_ec_provenance ON spend_categorizer_service.emission_calculations(provenance_hash);
CREATE INDEX idx_ec_created_at ON spend_categorizer_service.emission_calculations(created_at DESC);
CREATE INDEX idx_ec_tenant_record ON spend_categorizer_service.emission_calculations(tenant_id, record_id);
CREATE INDEX idx_ec_tenant_factor ON spend_categorizer_service.emission_calculations(tenant_id, factor_id);
CREATE INDEX idx_ec_record_factor ON spend_categorizer_service.emission_calculations(record_id, factor_id);

-- category_rules indexes
CREATE INDEX idx_cr_rule_id ON spend_categorizer_service.category_rules(rule_id);
CREATE INDEX idx_cr_tenant_id ON spend_categorizer_service.category_rules(tenant_id);
CREATE INDEX idx_cr_name ON spend_categorizer_service.category_rules(name);
CREATE INDEX idx_cr_match_type ON spend_categorizer_service.category_rules(match_type);
CREATE INDEX idx_cr_match_field ON spend_categorizer_service.category_rules(match_field);
CREATE INDEX idx_cr_target_taxonomy ON spend_categorizer_service.category_rules(target_taxonomy);
CREATE INDEX idx_cr_target_code ON spend_categorizer_service.category_rules(target_code);
CREATE INDEX idx_cr_target_scope3 ON spend_categorizer_service.category_rules(target_scope3_category);
CREATE INDEX idx_cr_priority ON spend_categorizer_service.category_rules(priority);
CREATE INDEX idx_cr_is_active ON spend_categorizer_service.category_rules(is_active);
CREATE INDEX idx_cr_provenance ON spend_categorizer_service.category_rules(provenance_hash);
CREATE INDEX idx_cr_created_at ON spend_categorizer_service.category_rules(created_at DESC);
CREATE INDEX idx_cr_updated_at ON spend_categorizer_service.category_rules(updated_at DESC);
CREATE INDEX idx_cr_tenant_active ON spend_categorizer_service.category_rules(tenant_id, is_active);
CREATE INDEX idx_cr_tenant_priority ON spend_categorizer_service.category_rules(tenant_id, priority);
CREATE INDEX idx_cr_active_priority ON spend_categorizer_service.category_rules(is_active, priority);

-- analytics_snapshots indexes
CREATE INDEX idx_as_snapshot_id ON spend_categorizer_service.analytics_snapshots(snapshot_id);
CREATE INDEX idx_as_tenant_id ON spend_categorizer_service.analytics_snapshots(tenant_id);
CREATE INDEX idx_as_period_start ON spend_categorizer_service.analytics_snapshots(period_start);
CREATE INDEX idx_as_period_end ON spend_categorizer_service.analytics_snapshots(period_end);
CREATE INDEX idx_as_provenance ON spend_categorizer_service.analytics_snapshots(provenance_hash);
CREATE INDEX idx_as_created_at ON spend_categorizer_service.analytics_snapshots(created_at DESC);
CREATE INDEX idx_as_tenant_period ON spend_categorizer_service.analytics_snapshots(tenant_id, period_start, period_end);
CREATE INDEX idx_as_scope3_breakdown ON spend_categorizer_service.analytics_snapshots USING GIN (scope3_breakdown);
CREATE INDEX idx_as_category_breakdown ON spend_categorizer_service.analytics_snapshots USING GIN (category_breakdown);
CREATE INDEX idx_as_vendor_breakdown ON spend_categorizer_service.analytics_snapshots USING GIN (vendor_breakdown);
CREATE INDEX idx_as_hotspots ON spend_categorizer_service.analytics_snapshots USING GIN (hotspots);

-- ingestion_batches indexes
CREATE INDEX idx_ib_batch_id ON spend_categorizer_service.ingestion_batches(batch_id);
CREATE INDEX idx_ib_tenant_id ON spend_categorizer_service.ingestion_batches(tenant_id);
CREATE INDEX idx_ib_source ON spend_categorizer_service.ingestion_batches(source);
CREATE INDEX idx_ib_status ON spend_categorizer_service.ingestion_batches(status);
CREATE INDEX idx_ib_provenance ON spend_categorizer_service.ingestion_batches(provenance_hash);
CREATE INDEX idx_ib_created_at ON spend_categorizer_service.ingestion_batches(created_at DESC);
CREATE INDEX idx_ib_completed_at ON spend_categorizer_service.ingestion_batches(completed_at DESC);
CREATE INDEX idx_ib_tenant_status ON spend_categorizer_service.ingestion_batches(tenant_id, status);
CREATE INDEX idx_ib_tenant_source ON spend_categorizer_service.ingestion_batches(tenant_id, source);
CREATE INDEX idx_ib_error_details ON spend_categorizer_service.ingestion_batches USING GIN (error_details);

-- categorization_events indexes (hypertable-aware)
CREATE INDEX idx_ce_record_id ON spend_categorizer_service.categorization_events(record_id, event_time DESC);
CREATE INDEX idx_ce_tenant_id ON spend_categorizer_service.categorization_events(tenant_id, event_time DESC);
CREATE INDEX idx_ce_event_type ON spend_categorizer_service.categorization_events(event_type, event_time DESC);
CREATE INDEX idx_ce_taxonomy_system ON spend_categorizer_service.categorization_events(taxonomy_system, event_time DESC);
CREATE INDEX idx_ce_scope3_category ON spend_categorizer_service.categorization_events(scope3_category, event_time DESC);
CREATE INDEX idx_ce_metadata ON spend_categorizer_service.categorization_events USING GIN (metadata);

-- emission_calculations_ts indexes (hypertable-aware)
CREATE INDEX idx_ects_record_id ON spend_categorizer_service.emission_calculations_ts(record_id, event_time DESC);
CREATE INDEX idx_ects_tenant_id ON spend_categorizer_service.emission_calculations_ts(tenant_id, event_time DESC);
CREATE INDEX idx_ects_factor_source ON spend_categorizer_service.emission_calculations_ts(factor_source, event_time DESC);

-- analytics_events indexes (hypertable-aware)
CREATE INDEX idx_ae_tenant_id ON spend_categorizer_service.analytics_events(tenant_id, event_time DESC);
CREATE INDEX idx_ae_event_type ON spend_categorizer_service.analytics_events(event_type, event_time DESC);
CREATE INDEX idx_ae_scope3_category ON spend_categorizer_service.analytics_events(scope3_category, event_time DESC);
CREATE INDEX idx_ae_metadata ON spend_categorizer_service.analytics_events USING GIN (metadata);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

-- spend_records: tenant-scoped
ALTER TABLE spend_categorizer_service.spend_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY sr_tenant_read ON spend_categorizer_service.spend_records
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY sr_tenant_write ON spend_categorizer_service.spend_records
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- taxonomy_mappings: tenant-scoped
ALTER TABLE spend_categorizer_service.taxonomy_mappings ENABLE ROW LEVEL SECURITY;
CREATE POLICY tm_tenant_read ON spend_categorizer_service.taxonomy_mappings
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY tm_tenant_write ON spend_categorizer_service.taxonomy_mappings
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- scope3_classifications: tenant-scoped
ALTER TABLE spend_categorizer_service.scope3_classifications ENABLE ROW LEVEL SECURITY;
CREATE POLICY s3c_tenant_read ON spend_categorizer_service.scope3_classifications
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY s3c_tenant_write ON spend_categorizer_service.scope3_classifications
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- emission_factors: global (no tenant_id column, shared reference data)
ALTER TABLE spend_categorizer_service.emission_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY ef_tenant_read ON spend_categorizer_service.emission_factors
    FOR SELECT USING (TRUE);
CREATE POLICY ef_tenant_write ON spend_categorizer_service.emission_factors
    FOR ALL USING (TRUE);

-- emission_calculations: tenant-scoped
ALTER TABLE spend_categorizer_service.emission_calculations ENABLE ROW LEVEL SECURITY;
CREATE POLICY ec_tenant_read ON spend_categorizer_service.emission_calculations
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ec_tenant_write ON spend_categorizer_service.emission_calculations
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- category_rules: tenant-scoped
ALTER TABLE spend_categorizer_service.category_rules ENABLE ROW LEVEL SECURITY;
CREATE POLICY cr_tenant_read ON spend_categorizer_service.category_rules
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY cr_tenant_write ON spend_categorizer_service.category_rules
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- analytics_snapshots: tenant-scoped
ALTER TABLE spend_categorizer_service.analytics_snapshots ENABLE ROW LEVEL SECURITY;
CREATE POLICY as_tenant_read ON spend_categorizer_service.analytics_snapshots
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY as_tenant_write ON spend_categorizer_service.analytics_snapshots
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- ingestion_batches: tenant-scoped
ALTER TABLE spend_categorizer_service.ingestion_batches ENABLE ROW LEVEL SECURITY;
CREATE POLICY ib_tenant_read ON spend_categorizer_service.ingestion_batches
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ib_tenant_write ON spend_categorizer_service.ingestion_batches
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- categorization_events: open (hypertable, no tenant column in PK)
ALTER TABLE spend_categorizer_service.categorization_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY ce_tenant_read ON spend_categorizer_service.categorization_events
    FOR SELECT USING (TRUE);
CREATE POLICY ce_tenant_write ON spend_categorizer_service.categorization_events
    FOR ALL USING (TRUE);

-- emission_calculations_ts: open (hypertable)
ALTER TABLE spend_categorizer_service.emission_calculations_ts ENABLE ROW LEVEL SECURITY;
CREATE POLICY ects_tenant_read ON spend_categorizer_service.emission_calculations_ts
    FOR SELECT USING (TRUE);
CREATE POLICY ects_tenant_write ON spend_categorizer_service.emission_calculations_ts
    FOR ALL USING (TRUE);

-- analytics_events: open (hypertable)
ALTER TABLE spend_categorizer_service.analytics_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY ae_tenant_read ON spend_categorizer_service.analytics_events
    FOR SELECT USING (TRUE);
CREATE POLICY ae_tenant_write ON spend_categorizer_service.analytics_events
    FOR ALL USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA spend_categorizer_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA spend_categorizer_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA spend_categorizer_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON spend_categorizer_service.hourly_categorization_stats TO greenlang_app;
GRANT SELECT ON spend_categorizer_service.hourly_emission_stats TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA spend_categorizer_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA spend_categorizer_service TO greenlang_readonly;
GRANT SELECT ON spend_categorizer_service.hourly_categorization_stats TO greenlang_readonly;
GRANT SELECT ON spend_categorizer_service.hourly_emission_stats TO greenlang_readonly;

-- Add spend categorizer service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'spend_categorizer:records:read', 'spend_categorizer', 'records_read', 'View spend records and normalized transaction data'),
    (gen_random_uuid(), 'spend_categorizer:records:write', 'spend_categorizer', 'records_write', 'Create and manage spend records and batch ingestion'),
    (gen_random_uuid(), 'spend_categorizer:taxonomy:read', 'spend_categorizer', 'taxonomy_read', 'View taxonomy classification mappings and confidence scores'),
    (gen_random_uuid(), 'spend_categorizer:taxonomy:write', 'spend_categorizer', 'taxonomy_write', 'Create and manage taxonomy classification mappings'),
    (gen_random_uuid(), 'spend_categorizer:scope3:read', 'spend_categorizer', 'scope3_read', 'View Scope 3 category classifications and split allocations'),
    (gen_random_uuid(), 'spend_categorizer:scope3:write', 'spend_categorizer', 'scope3_write', 'Create and manage Scope 3 category classifications'),
    (gen_random_uuid(), 'spend_categorizer:emissions:read', 'spend_categorizer', 'emissions_read', 'View emission factors and calculation results'),
    (gen_random_uuid(), 'spend_categorizer:emissions:write', 'spend_categorizer', 'emissions_write', 'Create and manage emission calculations and factors'),
    (gen_random_uuid(), 'spend_categorizer:rules:read', 'spend_categorizer', 'rules_read', 'View custom categorization rules and match statistics'),
    (gen_random_uuid(), 'spend_categorizer:rules:write', 'spend_categorizer', 'rules_write', 'Create and manage custom categorization rules'),
    (gen_random_uuid(), 'spend_categorizer:analytics:read', 'spend_categorizer', 'analytics_read', 'View analytics snapshots, hotspots, and trend reports'),
    (gen_random_uuid(), 'spend_categorizer:analytics:write', 'spend_categorizer', 'analytics_write', 'Generate analytics snapshots and aggregation reports'),
    (gen_random_uuid(), 'spend_categorizer:batches:read', 'spend_categorizer', 'batches_read', 'View ingestion batch status and processing metrics'),
    (gen_random_uuid(), 'spend_categorizer:admin', 'spend_categorizer', 'admin', 'Spend categorizer service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep categorization event records for 90 days
SELECT add_retention_policy('spend_categorizer_service.categorization_events', INTERVAL '90 days');

-- Keep emission calculation time-series records for 90 days
SELECT add_retention_policy('spend_categorizer_service.emission_calculations_ts', INTERVAL '90 days');

-- Keep analytics event records for 90 days
SELECT add_retention_policy('spend_categorizer_service.analytics_events', INTERVAL '90 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on categorization_events after 7 days
ALTER TABLE spend_categorizer_service.categorization_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'event_time DESC'
);

SELECT add_compression_policy('spend_categorizer_service.categorization_events', INTERVAL '7 days');

-- Enable compression on emission_calculations_ts after 7 days
ALTER TABLE spend_categorizer_service.emission_calculations_ts SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'event_time DESC'
);

SELECT add_compression_policy('spend_categorizer_service.emission_calculations_ts', INTERVAL '7 days');

-- Enable compression on analytics_events after 7 days
ALTER TABLE spend_categorizer_service.analytics_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'event_time DESC'
);

SELECT add_compression_policy('spend_categorizer_service.analytics_events', INTERVAL '7 days');

-- =============================================================================
-- Seed: EPA EEIO Emission Factors (20 representative factors)
-- =============================================================================
-- US EPA Environmentally-Extended Input-Output (EEIO) model factors.
-- Source: US EPA Supply Chain GHG Emission Factors v1.2 (2024)
-- Units: kgCO2e per USD spent

INSERT INTO spend_categorizer_service.emission_factors (source, naics_code, taxonomy_system, taxonomy_code, region, factor_value, factor_unit, year, version, description, metadata) VALUES
-- Manufacturing sectors
('epa_eeio', '311', 'naics', '311', 'US', 0.74200000, 'kgCO2e/USD', 2024, '1.2', 'Food manufacturing', '{"sector": "food", "detail_level": "3-digit"}'::jsonb),
('epa_eeio', '324', 'naics', '324', 'US', 1.58300000, 'kgCO2e/USD', 2024, '1.2', 'Petroleum and coal products manufacturing', '{"sector": "energy", "detail_level": "3-digit"}'::jsonb),
('epa_eeio', '325', 'naics', '325', 'US', 0.62100000, 'kgCO2e/USD', 2024, '1.2', 'Chemical manufacturing', '{"sector": "chemicals", "detail_level": "3-digit"}'::jsonb),
('epa_eeio', '326', 'naics', '326', 'US', 0.54800000, 'kgCO2e/USD', 2024, '1.2', 'Plastics and rubber products manufacturing', '{"sector": "plastics", "detail_level": "3-digit"}'::jsonb),
('epa_eeio', '327', 'naics', '327', 'US', 0.89200000, 'kgCO2e/USD', 2024, '1.2', 'Nonmetallic mineral product manufacturing', '{"sector": "minerals", "detail_level": "3-digit"}'::jsonb),
('epa_eeio', '331', 'naics', '331', 'US', 1.12400000, 'kgCO2e/USD', 2024, '1.2', 'Primary metal manufacturing', '{"sector": "metals", "detail_level": "3-digit"}'::jsonb),
('epa_eeio', '332', 'naics', '332', 'US', 0.43600000, 'kgCO2e/USD', 2024, '1.2', 'Fabricated metal product manufacturing', '{"sector": "metals", "detail_level": "3-digit"}'::jsonb),
('epa_eeio', '333', 'naics', '333', 'US', 0.35200000, 'kgCO2e/USD', 2024, '1.2', 'Machinery manufacturing', '{"sector": "machinery", "detail_level": "3-digit"}'::jsonb),
('epa_eeio', '334', 'naics', '334', 'US', 0.18900000, 'kgCO2e/USD', 2024, '1.2', 'Computer and electronic product manufacturing', '{"sector": "electronics", "detail_level": "3-digit"}'::jsonb),
('epa_eeio', '335', 'naics', '335', 'US', 0.31500000, 'kgCO2e/USD', 2024, '1.2', 'Electrical equipment and appliance manufacturing', '{"sector": "electrical", "detail_level": "3-digit"}'::jsonb),
-- Transportation and services
('epa_eeio', '481', 'naics', '481', 'US', 1.02300000, 'kgCO2e/USD', 2024, '1.2', 'Air transportation', '{"sector": "transport", "detail_level": "3-digit"}'::jsonb),
('epa_eeio', '484', 'naics', '484', 'US', 0.81700000, 'kgCO2e/USD', 2024, '1.2', 'Truck transportation', '{"sector": "transport", "detail_level": "3-digit"}'::jsonb),
('epa_eeio', '221', 'naics', '221', 'US', 1.89500000, 'kgCO2e/USD', 2024, '1.2', 'Utilities (electric power generation)', '{"sector": "utilities", "detail_level": "3-digit"}'::jsonb),
('epa_eeio', '236', 'naics', '236', 'US', 0.37800000, 'kgCO2e/USD', 2024, '1.2', 'Construction of buildings', '{"sector": "construction", "detail_level": "3-digit"}'::jsonb),
('epa_eeio', '541', 'naics', '541', 'US', 0.12400000, 'kgCO2e/USD', 2024, '1.2', 'Professional, scientific, and technical services', '{"sector": "services", "detail_level": "3-digit"}'::jsonb),
('epa_eeio', '561', 'naics', '561', 'US', 0.15200000, 'kgCO2e/USD', 2024, '1.2', 'Administrative and support services', '{"sector": "services", "detail_level": "3-digit"}'::jsonb),
('epa_eeio', '511', 'naics', '511', 'US', 0.09800000, 'kgCO2e/USD', 2024, '1.2', 'Publishing industries (except Internet)', '{"sector": "information", "detail_level": "3-digit"}'::jsonb),
('epa_eeio', '517', 'naics', '517', 'US', 0.11300000, 'kgCO2e/USD', 2024, '1.2', 'Telecommunications', '{"sector": "information", "detail_level": "3-digit"}'::jsonb),
('epa_eeio', '721', 'naics', '721', 'US', 0.28600000, 'kgCO2e/USD', 2024, '1.2', 'Accommodation', '{"sector": "hospitality", "detail_level": "3-digit"}'::jsonb),
('epa_eeio', '722', 'naics', '722', 'US', 0.39400000, 'kgCO2e/USD', 2024, '1.2', 'Food services and drinking places', '{"sector": "hospitality", "detail_level": "3-digit"}'::jsonb)
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: EXIOBASE Emission Factors (10 representative factors)
-- =============================================================================
-- EXIOBASE 3 multi-regional environmentally extended supply and use /
-- input-output database. European and global average factors.
-- Units: kgCO2e per EUR spent (converted to USD equivalent)

INSERT INTO spend_categorizer_service.emission_factors (source, taxonomy_system, taxonomy_code, region, factor_value, factor_unit, year, version, description, metadata) VALUES
('exiobase', 'isic', 'A01', 'EU', 0.95600000, 'kgCO2e/EUR', 2023, '3.8.2', 'Crop and animal production, hunting and related service activities', '{"database": "exiobase3", "resolution": "product"}'::jsonb),
('exiobase', 'isic', 'B05-09', 'EU', 1.23400000, 'kgCO2e/EUR', 2023, '3.8.2', 'Mining and quarrying', '{"database": "exiobase3", "resolution": "product"}'::jsonb),
('exiobase', 'isic', 'C10-12', 'EU', 0.68900000, 'kgCO2e/EUR', 2023, '3.8.2', 'Manufacture of food products, beverages and tobacco', '{"database": "exiobase3", "resolution": "product"}'::jsonb),
('exiobase', 'isic', 'C19', 'EU', 2.14700000, 'kgCO2e/EUR', 2023, '3.8.2', 'Manufacture of coke and refined petroleum products', '{"database": "exiobase3", "resolution": "product"}'::jsonb),
('exiobase', 'isic', 'C20', 'EU', 0.78300000, 'kgCO2e/EUR', 2023, '3.8.2', 'Manufacture of chemicals and chemical products', '{"database": "exiobase3", "resolution": "product"}'::jsonb),
('exiobase', 'isic', 'C24', 'EU', 1.34200000, 'kgCO2e/EUR', 2023, '3.8.2', 'Manufacture of basic metals', '{"database": "exiobase3", "resolution": "product"}'::jsonb),
('exiobase', 'isic', 'D35', 'EU', 2.08900000, 'kgCO2e/EUR', 2023, '3.8.2', 'Electricity, gas, steam and air conditioning supply', '{"database": "exiobase3", "resolution": "product"}'::jsonb),
('exiobase', 'isic', 'F41-43', 'EU', 0.41200000, 'kgCO2e/EUR', 2023, '3.8.2', 'Construction', '{"database": "exiobase3", "resolution": "product"}'::jsonb),
('exiobase', 'isic', 'H49', 'EU', 0.92100000, 'kgCO2e/EUR', 2023, '3.8.2', 'Land transport and transport via pipelines', '{"database": "exiobase3", "resolution": "product"}'::jsonb),
('exiobase', 'isic', 'H51', 'EU', 1.18600000, 'kgCO2e/EUR', 2023, '3.8.2', 'Air transport', '{"database": "exiobase3", "resolution": "product"}'::jsonb)
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: DEFRA Emission Factors (10 representative factors)
-- =============================================================================
-- UK Department for Environment, Food and Rural Affairs (DEFRA)
-- Greenhouse gas reporting conversion factors 2024.
-- Units: kgCO2e per GBP spent

INSERT INTO spend_categorizer_service.emission_factors (source, taxonomy_system, taxonomy_code, region, factor_value, factor_unit, year, version, description, metadata) VALUES
('defra', 'sic', '01', 'UK', 0.87400000, 'kgCO2e/GBP', 2024, '2024.1', 'Crop and animal production', '{"database": "defra_conversion_factors", "scope": "scope3_upstream"}'::jsonb),
('defra', 'sic', '10-12', 'UK', 0.63200000, 'kgCO2e/GBP', 2024, '2024.1', 'Manufacture of food products, beverages and tobacco', '{"database": "defra_conversion_factors", "scope": "scope3_upstream"}'::jsonb),
('defra', 'sic', '19', 'UK', 1.76500000, 'kgCO2e/GBP', 2024, '2024.1', 'Manufacture of coke and refined petroleum products', '{"database": "defra_conversion_factors", "scope": "scope3_upstream"}'::jsonb),
('defra', 'sic', '20', 'UK', 0.71800000, 'kgCO2e/GBP', 2024, '2024.1', 'Manufacture of chemicals and chemical products', '{"database": "defra_conversion_factors", "scope": "scope3_upstream"}'::jsonb),
('defra', 'sic', '24', 'UK', 1.21300000, 'kgCO2e/GBP', 2024, '2024.1', 'Manufacture of basic metals', '{"database": "defra_conversion_factors", "scope": "scope3_upstream"}'::jsonb),
('defra', 'sic', '35', 'UK', 1.94200000, 'kgCO2e/GBP', 2024, '2024.1', 'Electricity, gas, steam and air conditioning supply', '{"database": "defra_conversion_factors", "scope": "scope3_upstream"}'::jsonb),
('defra', 'sic', '41-43', 'UK', 0.38900000, 'kgCO2e/GBP', 2024, '2024.1', 'Construction', '{"database": "defra_conversion_factors", "scope": "scope3_upstream"}'::jsonb),
('defra', 'sic', '49', 'UK', 0.85600000, 'kgCO2e/GBP', 2024, '2024.1', 'Land transport and transport via pipelines', '{"database": "defra_conversion_factors", "scope": "scope3_upstream"}'::jsonb),
('defra', 'sic', '51', 'UK', 1.14200000, 'kgCO2e/GBP', 2024, '2024.1', 'Air transport', '{"database": "defra_conversion_factors", "scope": "scope3_upstream"}'::jsonb),
('defra', 'sic', '55', 'UK', 0.31200000, 'kgCO2e/GBP', 2024, '2024.1', 'Accommodation', '{"database": "defra_conversion_factors", "scope": "scope3_upstream"}'::jsonb)
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: Register the Spend Data Categorizer Agent (GL-DATA-SUP-002)
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES
('GL-DATA-SUP-002', 'Spend Data Categorizer',
 'Categorizes and calculates emissions from organizational spend data for GreenLang Climate OS. Ingests spend records from multiple sources (ERP, CSV, Excel, API), normalizes vendor names and currency to USD, classifies spend against multiple taxonomy systems (UNSPSC, NAICS, eCl@ss, ISIC, SIC, CPV, HS/CN), assigns GHG Protocol Scope 3 categories (1-15) with confidence scoring and split allocation, calculates per-record emissions using authoritative emission factor databases (EPA EEIO, EXIOBASE, DEFRA, Ecoinvent), supports custom categorization rules with pattern matching, and produces analytics snapshots with hotspot identification.',
 2, 'async', true, true, 5, '1.0.0', true,
 'GreenLang Data Team', 'https://docs.greenlang.ai/agents/spend-categorizer', true, 'default')
ON CONFLICT (agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Agent Version for Spend Data Categorizer
-- =============================================================================

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES
('GL-DATA-SUP-002', '1.0.0',
 '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "1Gi", "memory_limit": "4Gi", "gpu": false}'::jsonb,
 '{"image": "greenlang/spend-categorizer-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"spend", "categorization", "taxonomy", "scope3", "emissions", "eeio", "naics", "unspsc", "ghg"}',
 '{"cross-sector", "manufacturing", "retail", "energy", "finance", "healthcare"}',
 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2')
ON CONFLICT (agent_id, version) DO NOTHING;

-- =============================================================================
-- Seed: Agent Capabilities for Spend Data Categorizer
-- =============================================================================

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES

('GL-DATA-SUP-002', '1.0.0', 'spend_ingestion', 'ingestion',
 'Ingest and normalize spend records from multiple sources (ERP extracts, CSV files, Excel files, API feeds, manual entry) with vendor name normalization, multi-currency conversion to USD, and batch tracking',
 '{"records", "source", "batch_config"}', '{"batch_id", "total_records", "processed_records", "error_records"}',
 '{"sources": ["erp_extract", "csv_file", "excel_file", "api_feed", "manual_entry"], "max_batch_size": 50000, "supported_currencies": 170, "auto_normalize_vendors": true}'::jsonb),

('GL-DATA-SUP-002', '1.0.0', 'taxonomy_classification', 'classification',
 'Classify spend records against multiple industry taxonomy systems (UNSPSC, NAICS, eCl@ss, ISIC, SIC, CPV, HS/CN) with hierarchical code assignment, confidence scoring, and classification method tracking',
 '{"record_id", "taxonomy_systems"}', '{"mappings", "confidence", "classification_method"}',
 '{"taxonomy_systems": ["unspsc", "naics", "eclass", "isic", "sic", "cpv", "hs_cn"], "min_confidence_threshold": 0.5, "hierarchical_levels": true, "batch_classification": true}'::jsonb),

('GL-DATA-SUP-002', '1.0.0', 'scope3_assignment', 'classification',
 'Assign GHG Protocol Scope 3 categories (1-15) to spend records based on taxonomy codes, vendor data, and custom rules with confidence scoring, CAPEX flagging, and multi-category split allocation',
 '{"record_id", "mapping_rules"}', '{"scope3_category", "category_number", "confidence", "split_pct"}',
 '{"categories": 15, "split_allocation": true, "capex_detection": true, "rule_based_override": true, "min_confidence": 0.4}'::jsonb),

('GL-DATA-SUP-002', '1.0.0', 'emission_calculation', 'computation',
 'Calculate per-record emissions using authoritative emission factor databases (EPA EEIO, EXIOBASE, DEFRA, Ecoinvent, custom) with deterministic spend-based methodology, data quality scoring, and provenance tracking',
 '{"record_id", "factor_source"}', '{"emissions_kgco2e", "emissions_tco2e", "data_quality_score", "provenance_hash"}',
 '{"factor_sources": ["epa_eeio", "exiobase", "defra", "ecoinvent", "custom"], "calculation_method": "spend_based", "zero_hallucination": true, "deterministic": true}'::jsonb),

('GL-DATA-SUP-002', '1.0.0', 'custom_rules_engine', 'classification',
 'Apply tenant-specific custom categorization rules using pattern matching (exact, contains, regex, fuzzy, starts_with, ends_with) against spend fields (description, vendor_name, gl_account, cost_center, material_group) with priority ordering',
 '{"record_id", "rules"}', '{"matched_rules", "taxonomy_code", "scope3_category"}',
 '{"match_types": ["exact", "contains", "regex", "fuzzy", "starts_with", "ends_with"], "match_fields": ["description", "vendor_name", "gl_account", "cost_center", "material_group"], "priority_ordering": true, "match_count_tracking": true}'::jsonb),

('GL-DATA-SUP-002', '1.0.0', 'analytics_aggregation', 'reporting',
 'Generate analytics snapshots with total spend and emission summaries, Scope 3 category breakdowns, vendor analysis, hotspot identification, and trend reports for dashboard consumption',
 '{"tenant_id", "period_start", "period_end"}', '{"snapshot_id", "total_spend_usd", "total_emissions_tco2e", "hotspots"}',
 '{"breakdowns": ["scope3", "category", "vendor", "hotspot"], "export_formats": ["json", "csv", "xlsx"], "trend_analysis": true, "benchmark_comparison": true}'::jsonb),

('GL-DATA-SUP-002', '1.0.0', 'batch_processing', 'orchestration',
 'Process large spend datasets in configurable batches with parallel execution, error tracking, partial completion support, and lifecycle management from pending through processing to completed or failed',
 '{"batch_id", "batch_config"}', '{"status", "processed_records", "error_records", "processing_time_ms"}',
 '{"max_batch_size": 50000, "parallel_workers": 4, "retry_on_error": true, "partial_completion": true, "status_lifecycle": ["pending", "processing", "completed", "failed", "cancelled", "partial"]}'::jsonb)

ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: Agent Dependencies for Spend Data Categorizer
-- =============================================================================

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES

-- Spend Categorizer depends on Schema Compiler for input/output validation
('GL-DATA-SUP-002', 'GL-FOUND-X-002', '>=1.0.0', false,
 'Spend records, taxonomy mappings, and emission calculations are validated against JSON Schema definitions'),

-- Spend Categorizer depends on Registry for agent discovery
('GL-DATA-SUP-002', 'GL-FOUND-X-007', '>=1.0.0', false,
 'Agent version and capability lookup for spend categorization pipeline orchestration'),

-- Spend Categorizer depends on Access Guard for policy enforcement
('GL-DATA-SUP-002', 'GL-FOUND-X-006', '>=1.0.0', false,
 'Data classification and access control enforcement for spend records and emission data'),

-- Spend Categorizer depends on Observability Agent for metrics
('GL-DATA-SUP-002', 'GL-FOUND-X-010', '>=1.0.0', false,
 'Categorization metrics, emission calculation statistics, and batch processing telemetry are reported to observability'),

-- Spend Categorizer depends on Unit Normalizer for currency conversion
('GL-DATA-SUP-002', 'GL-FOUND-X-003', '>=1.0.0', false,
 'Multi-currency spend amounts are normalized to USD using authoritative exchange rates'),

-- Spend Categorizer optionally uses Citations for provenance tracking
('GL-DATA-SUP-002', 'GL-FOUND-X-005', '>=1.0.0', true,
 'Emission factor provenance and calculation audit trails are registered with the citation service'),

-- Spend Categorizer optionally uses Reproducibility for determinism
('GL-DATA-SUP-002', 'GL-FOUND-X-008', '>=1.0.0', true,
 'Emission calculations are verified for reproducibility across re-execution with identical inputs'),

-- Spend Categorizer optionally integrates with ERP Connector
('GL-DATA-SUP-002', 'GL-DATA-X-003', '>=1.0.0', true,
 'ERP spend extracts are ingested through the ERP/Finance Connector agent for SAP, Oracle, NetSuite etc.'),

-- Spend Categorizer optionally integrates with Excel Normalizer
('GL-DATA-SUP-002', 'GL-DATA-X-002', '>=1.0.0', true,
 'Excel and CSV spend files are pre-processed through the Excel/CSV Normalizer agent'),

-- Spend Categorizer optionally integrates with Supplier Questionnaire
('GL-DATA-SUP-002', 'GL-DATA-SUP-001', '>=1.0.0', true,
 'Supplier-specific emission factors from questionnaire responses can override default EEIO factors')

ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Service Catalog Entry for Spend Data Categorizer
-- =============================================================================

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES
('GL-DATA-SUP-002', 'Spend Data Categorizer',
 'End-to-end spend data categorization and emission calculation engine. Ingests spend records from ERP, CSV, Excel, and API sources with vendor normalization and multi-currency conversion. Classifies against 7 taxonomy systems (UNSPSC, NAICS, eCl@ss, ISIC, SIC, CPV, HS/CN). Assigns GHG Protocol Scope 3 categories 1-15 with split allocation. Calculates emissions using EPA EEIO, EXIOBASE, DEFRA, and Ecoinvent factors. Custom rule engine for tenant-specific categorization. Analytics dashboards with hotspot identification. SHA-256 provenance chains for zero-hallucination audit trail.',
 'data', 'active', 'default')
ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA spend_categorizer_service IS 'Spend Data Categorizer for GreenLang Climate OS (AGENT-DATA-009) - spend ingestion, multi-taxonomy classification, Scope 3 assignment, emission factor management, per-record emission calculation, custom rules engine, and analytics aggregation with provenance chains';
COMMENT ON TABLE spend_categorizer_service.spend_records IS 'Ingested and normalized spend records with vendor details, multi-currency amount (normalized to USD), GL account, cost center, department, material group, processing status lifecycle, and SHA-256 provenance hash';
COMMENT ON TABLE spend_categorizer_service.taxonomy_mappings IS 'Taxonomy classification results linking spend records to industry codes (UNSPSC/NAICS/eCl@ss/ISIC/SIC/CPV/HS-CN) with hierarchical level, parent code, confidence scoring, and classification method';
COMMENT ON TABLE spend_categorizer_service.scope3_classifications IS 'GHG Protocol Scope 3 category assignments (categories 1-15) for spend records with confidence scoring, CAPEX flagging, split percentage for multi-category allocation, and provenance hash';
COMMENT ON TABLE spend_categorizer_service.emission_factors IS 'Emission factor database from authoritative sources (EPA EEIO, EXIOBASE, DEFRA, Ecoinvent, custom) with taxonomy mapping, geographic region, factor value and units, reference year, and version tracking';
COMMENT ON TABLE spend_categorizer_service.emission_calculations IS 'Per-record emission calculation results linking spend records to emission factors with spend in USD, emissions in kgCO2e and tCO2e, calculation method, data quality score, and provenance hash';
COMMENT ON TABLE spend_categorizer_service.category_rules IS 'Custom tenant-specific categorization rules with pattern matching (exact/contains/regex/fuzzy/starts_with/ends_with) against spend fields, target taxonomy and Scope 3 category, priority ordering, and match count tracking';
COMMENT ON TABLE spend_categorizer_service.analytics_snapshots IS 'Periodic analytics aggregation snapshots with total spend and emissions, record counts, JSONB breakdowns by Scope 3 category, taxonomy category, vendor, and hotspot identification';
COMMENT ON TABLE spend_categorizer_service.ingestion_batches IS 'Batch ingestion tracking with source, file name, record counts (total/processed/error), processing status lifecycle, error details, and provenance hash';
COMMENT ON TABLE spend_categorizer_service.categorization_events IS 'TimescaleDB hypertable: categorization lifecycle event time-series with event type (ingested/classified/mapped/calculated/rule_matched), taxonomy system and code, Scope 3 category, confidence, and processing duration';
COMMENT ON TABLE spend_categorizer_service.emission_calculations_ts IS 'TimescaleDB hypertable: emission calculation event time-series with factor source, spend amount, emissions, factor value, and processing duration';
COMMENT ON TABLE spend_categorizer_service.analytics_events IS 'TimescaleDB hypertable: analytics event time-series with event type (aggregation/hotspot/trend/report), Scope 3 category, spend, emissions, and metadata';
COMMENT ON MATERIALIZED VIEW spend_categorizer_service.hourly_categorization_stats IS 'Continuous aggregate: hourly categorization event statistics by event type with total events, unique records, per-taxonomy-system counts, average confidence, and average processing time';
COMMENT ON MATERIALIZED VIEW spend_categorizer_service.hourly_emission_stats IS 'Continuous aggregate: hourly emission calculation statistics by factor source with total spend, total emissions, calculation count, unique records, average factor value, and average processing time';

COMMENT ON COLUMN spend_categorizer_service.spend_records.source IS 'Data source: erp_extract, csv_file, excel_file, api_feed, manual_entry';
COMMENT ON COLUMN spend_categorizer_service.spend_records.status IS 'Processing status: raw, normalized, classified, calculated, validated, error, duplicate, excluded, archived';
COMMENT ON COLUMN spend_categorizer_service.spend_records.amount_usd IS 'Spend amount normalized to USD using exchange rate at transaction date';
COMMENT ON COLUMN spend_categorizer_service.spend_records.normalized_vendor_name IS 'Vendor name after normalization (trimming, case folding, entity resolution)';
COMMENT ON COLUMN spend_categorizer_service.spend_records.provenance_hash IS 'SHA-256 provenance hash of record content for integrity verification and audit trail';
COMMENT ON COLUMN spend_categorizer_service.taxonomy_mappings.taxonomy_system IS 'Taxonomy system: unspsc, naics, eclass, isic, sic, cpv, hs_cn';
COMMENT ON COLUMN spend_categorizer_service.taxonomy_mappings.confidence IS 'Classification confidence score (0-1) indicating reliability of taxonomy assignment';
COMMENT ON COLUMN spend_categorizer_service.taxonomy_mappings.classification_method IS 'Method used for classification: rule_based, ml_classifier, llm_assisted, manual, inherited';
COMMENT ON COLUMN spend_categorizer_service.scope3_classifications.scope3_category IS 'GHG Protocol Scope 3 category: cat1_purchased_goods through cat15_investments, or unclassified';
COMMENT ON COLUMN spend_categorizer_service.scope3_classifications.split_pct IS 'Split allocation percentage (0-1) for records spanning multiple Scope 3 categories';
COMMENT ON COLUMN spend_categorizer_service.scope3_classifications.is_capex IS 'Flag indicating capital expenditure (routes to Category 2: Capital Goods)';
COMMENT ON COLUMN spend_categorizer_service.emission_factors.source IS 'Emission factor source: epa_eeio, exiobase, defra, ecoinvent, custom';
COMMENT ON COLUMN spend_categorizer_service.emission_factors.factor_value IS 'Emission factor value in the specified factor_unit (e.g., kgCO2e per monetary unit spent)';
COMMENT ON COLUMN spend_categorizer_service.emission_factors.factor_unit IS 'Unit of the emission factor (e.g., kgCO2e/USD, kgCO2e/EUR, kgCO2e/GBP)';
COMMENT ON COLUMN spend_categorizer_service.emission_calculations.emissions_kgco2e IS 'Calculated emissions in kilograms of CO2 equivalent (spend_usd * factor_value)';
COMMENT ON COLUMN spend_categorizer_service.emission_calculations.emissions_tco2e IS 'Calculated emissions in metric tonnes of CO2 equivalent (emissions_kgco2e / 1000)';
COMMENT ON COLUMN spend_categorizer_service.emission_calculations.data_quality_score IS 'Data quality score (0-1) based on factor specificity, data recency, and input completeness';
COMMENT ON COLUMN spend_categorizer_service.category_rules.match_type IS 'Pattern match type: exact, contains, regex, fuzzy, starts_with, ends_with';
COMMENT ON COLUMN spend_categorizer_service.category_rules.match_field IS 'Spend record field to match against: description, vendor_name, gl_account, cost_center, material_group';
COMMENT ON COLUMN spend_categorizer_service.category_rules.match_count IS 'Running count of records matched by this rule (updated on each successful match)';
COMMENT ON COLUMN spend_categorizer_service.analytics_snapshots.hotspots IS 'JSONB identifying top emission hotspots by vendor, category, and Scope 3 category for reduction targeting';
COMMENT ON COLUMN spend_categorizer_service.ingestion_batches.status IS 'Batch processing status: pending, processing, completed, failed, cancelled, partial';
COMMENT ON COLUMN spend_categorizer_service.categorization_events.event_type IS 'Categorization event type: ingested, classified, mapped, calculated, rule_matched, normalized, validated, error, reclassified, split';
COMMENT ON COLUMN spend_categorizer_service.analytics_events.event_type IS 'Analytics event type: aggregation, hotspot, trend, report, benchmark, threshold_breach';
