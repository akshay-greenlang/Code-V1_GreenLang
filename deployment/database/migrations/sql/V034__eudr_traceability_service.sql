-- =============================================================================
-- V034: EUDR Traceability Service Schema
-- =============================================================================
-- Component: AGENT-DATA-005 (EUDR Traceability Connector Agent)
-- Agent ID:  GL-DATA-EUDR-001
-- Regulation: EU Deforestation Regulation (EU 2023/1115)
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- EUDR Traceability Connector Agent (GL-DATA-EUDR-001) with capabilities
-- for production plot registry, chain of custody tracking,
-- due diligence statement management, risk assessment,
-- commodity classification, compliance verification,
-- and EU Information System integration.
-- =============================================================================
-- Tables (10):
--   1. production_plots           - Geolocation-tracked production plots
--   2. plot_monitoring            - Satellite/field monitoring (hypertable)
--   3. custody_transfers          - Chain of custody transfer records
--   4. custody_batches            - Batch tracking for mass balance
--   5. due_diligence_statements   - DDS records per EU 2023/1115
--   6. dds_submissions            - EU IS submission log (hypertable)
--   7. risk_assessments           - Country/commodity/supplier risk scores
--   8. commodity_classifications  - CN/HS code commodity mapping
--   9. supplier_declarations      - Supplier deforestation-free declarations
--  10. compliance_checks          - Article-level compliance checks (hypertable)
--
-- Continuous Aggregates (2):
--   1. eudr_compliance_checks_hourly  - Hourly compliance check aggregates
--   2. eudr_dds_submissions_hourly    - Hourly DDS submission aggregates
--
-- Also includes: 50+ indexes (B-tree, GIN), RLS policies per tenant,
-- retention policies, compression policies, updated_at trigger,
-- security permissions, and seed data registering GL-DATA-EUDR-001
-- in the agent registry.
-- Previous: V033__erp_connector_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS eudr_traceability_service;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================
-- Reusable trigger function for tables with updated_at columns.

CREATE OR REPLACE FUNCTION eudr_traceability_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: eudr_traceability_service.production_plots
-- =============================================================================
-- Production plot registry. Each plot record captures the geolocation of a
-- production area for EUDR-regulated commodities. Includes GPS coordinates,
-- GeoJSON polygon boundaries, producer information, harvest details,
-- deforestation-free status, legal compliance declaration, and risk level.
-- Core reference table for traceability chain. Tenant-scoped.

CREATE TABLE eudr_traceability_service.production_plots (
    plot_id VARCHAR(50) PRIMARY KEY,
    commodity VARCHAR(20) NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    polygon_geojson JSONB,
    plot_area_hectares NUMERIC(12,4),
    country_code VARCHAR(3) NOT NULL,
    region VARCHAR(100),
    producer_id VARCHAR(100),
    producer_name VARCHAR(255),
    production_date DATE,
    harvest_date DATE,
    quantity_kg NUMERIC(15,4),
    unit VARCHAR(20) DEFAULT 'kg',
    certification VARCHAR(100),
    land_use_type VARCHAR(30) DEFAULT 'other',
    deforestation_free BOOLEAN DEFAULT FALSE,
    deforestation_cutoff_date DATE DEFAULT '2020-12-31',
    legal_compliance BOOLEAN DEFAULT FALSE,
    supporting_documents JSONB DEFAULT '[]'::jsonb,
    risk_level VARCHAR(20) DEFAULT 'unknown',
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Commodity constraint per EU 2023/1115 Article 2
ALTER TABLE eudr_traceability_service.production_plots
    ADD CONSTRAINT chk_plot_commodity
    CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood',
        'derived_cattle', 'derived_cocoa', 'derived_coffee', 'derived_oil_palm',
        'derived_rubber', 'derived_soya', 'derived_wood'
    ));

-- Latitude must be between -90 and 90
ALTER TABLE eudr_traceability_service.production_plots
    ADD CONSTRAINT chk_plot_latitude_range
    CHECK (latitude >= -90 AND latitude <= 90);

-- Longitude must be between -180 and 180
ALTER TABLE eudr_traceability_service.production_plots
    ADD CONSTRAINT chk_plot_longitude_range
    CHECK (longitude >= -180 AND longitude <= 180);

-- Plot area must be positive if specified
ALTER TABLE eudr_traceability_service.production_plots
    ADD CONSTRAINT chk_plot_area_positive
    CHECK (plot_area_hectares IS NULL OR plot_area_hectares > 0);

-- Country code must be 2 or 3 characters (ISO 3166)
ALTER TABLE eudr_traceability_service.production_plots
    ADD CONSTRAINT chk_plot_country_code_length
    CHECK (LENGTH(TRIM(country_code)) >= 2 AND LENGTH(TRIM(country_code)) <= 3);

-- Quantity must be non-negative if specified
ALTER TABLE eudr_traceability_service.production_plots
    ADD CONSTRAINT chk_plot_quantity_non_negative
    CHECK (quantity_kg IS NULL OR quantity_kg >= 0);

-- Risk level constraint
ALTER TABLE eudr_traceability_service.production_plots
    ADD CONSTRAINT chk_plot_risk_level
    CHECK (risk_level IN ('low', 'standard', 'high', 'unknown'));

-- Land use type constraint
ALTER TABLE eudr_traceability_service.production_plots
    ADD CONSTRAINT chk_plot_land_use_type
    CHECK (land_use_type IN (
        'cropland', 'pasture', 'plantation', 'agroforestry',
        'forest', 'degraded_forest', 'other'
    ));

-- Plot ID must not be empty
ALTER TABLE eudr_traceability_service.production_plots
    ADD CONSTRAINT chk_plot_id_not_empty
    CHECK (LENGTH(TRIM(plot_id)) > 0);

-- Updated_at trigger for production_plots
CREATE TRIGGER trg_production_plots_updated_at
    BEFORE UPDATE ON eudr_traceability_service.production_plots
    FOR EACH ROW
    EXECUTE FUNCTION eudr_traceability_service.set_updated_at();

-- =============================================================================
-- Table 2: eudr_traceability_service.plot_monitoring (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording monitoring events for production plots.
-- Each monitoring event captures satellite imagery analysis, field visit
-- results, or certification checks. Records forest cover change detection,
-- deforestation alerts, and assessor notes. Partitioned by monitored_at
-- for time-series queries. Retained for 730 days with compression after
-- 30 days. Tenant-scoped.

CREATE TABLE eudr_traceability_service.plot_monitoring (
    monitoring_id VARCHAR(50) NOT NULL,
    plot_id VARCHAR(50) NOT NULL,
    monitoring_type VARCHAR(50) NOT NULL,
    monitored_at TIMESTAMPTZ NOT NULL,
    deforestation_detected BOOLEAN DEFAULT FALSE,
    forest_cover_change_pct NUMERIC(8,4),
    satellite_source VARCHAR(100),
    image_reference VARCHAR(255),
    assessment_notes TEXT,
    assessor VARCHAR(100),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (monitoring_id, monitored_at)
);

-- Create hypertable partitioned by monitored_at
SELECT create_hypertable('eudr_traceability_service.plot_monitoring', 'monitored_at', if_not_exists => TRUE);

-- Monitoring type constraint
ALTER TABLE eudr_traceability_service.plot_monitoring
    ADD CONSTRAINT chk_monitoring_type
    CHECK (monitoring_type IN (
        'satellite', 'field_visit', 'remote_assessment', 'certification_check'
    ));

-- Forest cover change percentage must be between -100 and 100
ALTER TABLE eudr_traceability_service.plot_monitoring
    ADD CONSTRAINT chk_monitoring_forest_cover_range
    CHECK (forest_cover_change_pct IS NULL OR (forest_cover_change_pct >= -100 AND forest_cover_change_pct <= 100));

-- Monitoring ID must not be empty
ALTER TABLE eudr_traceability_service.plot_monitoring
    ADD CONSTRAINT chk_monitoring_id_not_empty
    CHECK (LENGTH(TRIM(monitoring_id)) > 0);

-- Plot ID must not be empty
ALTER TABLE eudr_traceability_service.plot_monitoring
    ADD CONSTRAINT chk_monitoring_plot_id_not_empty
    CHECK (LENGTH(TRIM(plot_id)) > 0);

-- =============================================================================
-- Table 3: eudr_traceability_service.custody_transfers
-- =============================================================================
-- Chain of custody transfer records tracking commodity movements between
-- operators. Each transfer captures source and target operators, commodity,
-- quantity, batch number, origin plot references, custody model (segregation,
-- mass balance, identity preserved), transport details, customs declarations,
-- CN/HS codes, and verification status. Core traceability table linking
-- production plots to end operators. Tenant-scoped.

CREATE TABLE eudr_traceability_service.custody_transfers (
    transfer_id VARCHAR(50) PRIMARY KEY,
    transaction_id VARCHAR(50) NOT NULL,
    source_operator_id VARCHAR(100) NOT NULL,
    source_operator_name VARCHAR(255) NOT NULL,
    target_operator_id VARCHAR(100) NOT NULL,
    target_operator_name VARCHAR(255) NOT NULL,
    commodity VARCHAR(20) NOT NULL,
    product_description TEXT,
    quantity NUMERIC(15,4) NOT NULL,
    unit VARCHAR(20) DEFAULT 'kg',
    batch_number VARCHAR(100),
    origin_plot_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    custody_model VARCHAR(30) DEFAULT 'mass_balance',
    transaction_date TIMESTAMPTZ NOT NULL,
    transport_mode VARCHAR(50),
    transport_documents JSONB DEFAULT '[]'::jsonb,
    customs_declaration VARCHAR(100),
    cn_code VARCHAR(20),
    hs_code VARCHAR(20),
    verification_status VARCHAR(30) DEFAULT 'pending_verification',
    verified_by VARCHAR(100),
    verified_at TIMESTAMPTZ,
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Commodity constraint
ALTER TABLE eudr_traceability_service.custody_transfers
    ADD CONSTRAINT chk_transfer_commodity
    CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood',
        'derived_cattle', 'derived_cocoa', 'derived_coffee', 'derived_oil_palm',
        'derived_rubber', 'derived_soya', 'derived_wood'
    ));

-- Custody model constraint
ALTER TABLE eudr_traceability_service.custody_transfers
    ADD CONSTRAINT chk_transfer_custody_model
    CHECK (custody_model IN (
        'identity_preserved', 'segregation', 'mass_balance',
        'controlled_blending', 'book_and_claim'
    ));

-- Verification status constraint
ALTER TABLE eudr_traceability_service.custody_transfers
    ADD CONSTRAINT chk_transfer_verification_status
    CHECK (verification_status IN (
        'pending_verification', 'verified', 'rejected',
        'under_review', 'expired', 'revoked'
    ));

-- Quantity must be positive
ALTER TABLE eudr_traceability_service.custody_transfers
    ADD CONSTRAINT chk_transfer_quantity_positive
    CHECK (quantity > 0);

-- Transfer ID must not be empty
ALTER TABLE eudr_traceability_service.custody_transfers
    ADD CONSTRAINT chk_transfer_id_not_empty
    CHECK (LENGTH(TRIM(transfer_id)) > 0);

-- Transaction ID must not be empty
ALTER TABLE eudr_traceability_service.custody_transfers
    ADD CONSTRAINT chk_transfer_transaction_id_not_empty
    CHECK (LENGTH(TRIM(transaction_id)) > 0);

-- Source operator ID must not be empty
ALTER TABLE eudr_traceability_service.custody_transfers
    ADD CONSTRAINT chk_transfer_source_operator_not_empty
    CHECK (LENGTH(TRIM(source_operator_id)) > 0);

-- Source operator name must not be empty
ALTER TABLE eudr_traceability_service.custody_transfers
    ADD CONSTRAINT chk_transfer_source_name_not_empty
    CHECK (LENGTH(TRIM(source_operator_name)) > 0);

-- Target operator ID must not be empty
ALTER TABLE eudr_traceability_service.custody_transfers
    ADD CONSTRAINT chk_transfer_target_operator_not_empty
    CHECK (LENGTH(TRIM(target_operator_id)) > 0);

-- Target operator name must not be empty
ALTER TABLE eudr_traceability_service.custody_transfers
    ADD CONSTRAINT chk_transfer_target_name_not_empty
    CHECK (LENGTH(TRIM(target_operator_name)) > 0);

-- Transport mode constraint
ALTER TABLE eudr_traceability_service.custody_transfers
    ADD CONSTRAINT chk_transfer_transport_mode
    CHECK (transport_mode IS NULL OR transport_mode IN (
        'sea', 'air', 'road', 'rail', 'pipeline', 'multimodal', 'other'
    ));

-- =============================================================================
-- Table 4: eudr_traceability_service.custody_batches
-- =============================================================================
-- Batch tracking records for mass balance and segregation custody models.
-- Each batch captures parent batch lineage, commodity, quantity, origin
-- plot references, and batch status. Used to maintain traceability through
-- processing, blending, and transformation steps. Tenant-scoped.

CREATE TABLE eudr_traceability_service.custody_batches (
    batch_id VARCHAR(50) PRIMARY KEY,
    parent_batch_ids JSONB DEFAULT '[]'::jsonb,
    commodity VARCHAR(20) NOT NULL,
    product_description TEXT,
    quantity NUMERIC(15,4) NOT NULL,
    unit VARCHAR(20) DEFAULT 'kg',
    origin_plot_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    custody_model VARCHAR(30) DEFAULT 'mass_balance',
    batch_status VARCHAR(20) DEFAULT 'active',
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Commodity constraint
ALTER TABLE eudr_traceability_service.custody_batches
    ADD CONSTRAINT chk_batch_commodity
    CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood',
        'derived_cattle', 'derived_cocoa', 'derived_coffee', 'derived_oil_palm',
        'derived_rubber', 'derived_soya', 'derived_wood'
    ));

-- Custody model constraint
ALTER TABLE eudr_traceability_service.custody_batches
    ADD CONSTRAINT chk_batch_custody_model
    CHECK (custody_model IN (
        'identity_preserved', 'segregation', 'mass_balance',
        'controlled_blending', 'book_and_claim'
    ));

-- Batch status constraint
ALTER TABLE eudr_traceability_service.custody_batches
    ADD CONSTRAINT chk_batch_status
    CHECK (batch_status IN (
        'active', 'consumed', 'expired', 'split', 'merged', 'cancelled'
    ));

-- Quantity must be positive
ALTER TABLE eudr_traceability_service.custody_batches
    ADD CONSTRAINT chk_batch_quantity_positive
    CHECK (quantity > 0);

-- Batch ID must not be empty
ALTER TABLE eudr_traceability_service.custody_batches
    ADD CONSTRAINT chk_batch_id_not_empty
    CHECK (LENGTH(TRIM(batch_id)) > 0);

-- =============================================================================
-- Table 5: eudr_traceability_service.due_diligence_statements
-- =============================================================================
-- Due Diligence Statement (DDS) records per EU 2023/1115 Articles 4 and 33.
-- Each DDS captures operator information, commodity details, CN codes,
-- origin plot references, risk assessment linkage, deforestation-free and
-- legal compliance declarations, risk mitigation measures, EU reference
-- number, digital signature, submission dates, and validity period.
-- Central compliance record for EUDR obligations. Tenant-scoped.

CREATE TABLE eudr_traceability_service.due_diligence_statements (
    dds_id VARCHAR(50) PRIMARY KEY,
    operator_id VARCHAR(100) NOT NULL,
    operator_name VARCHAR(255) NOT NULL,
    operator_country VARCHAR(3) NOT NULL,
    operator_eori VARCHAR(50),
    dds_type VARCHAR(30) DEFAULT 'import_placement',
    commodity VARCHAR(20) NOT NULL,
    product_description TEXT,
    cn_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
    quantity NUMERIC(15,4) NOT NULL,
    unit VARCHAR(20) DEFAULT 'kg',
    origin_countries JSONB DEFAULT '[]'::jsonb,
    origin_plot_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    custody_transfer_ids JSONB DEFAULT '[]'::jsonb,
    risk_assessment_id VARCHAR(50),
    risk_level VARCHAR(20) DEFAULT 'standard',
    deforestation_free_declaration BOOLEAN DEFAULT FALSE,
    legal_compliance_declaration BOOLEAN DEFAULT FALSE,
    risk_mitigation_measures JSONB DEFAULT '[]'::jsonb,
    supporting_evidence JSONB DEFAULT '[]'::jsonb,
    status VARCHAR(20) DEFAULT 'draft',
    eu_reference_number VARCHAR(100),
    digital_signature TEXT,
    submission_date TIMESTAMPTZ,
    validity_start DATE,
    validity_end DATE,
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Commodity constraint
ALTER TABLE eudr_traceability_service.due_diligence_statements
    ADD CONSTRAINT chk_dds_commodity
    CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood',
        'derived_cattle', 'derived_cocoa', 'derived_coffee', 'derived_oil_palm',
        'derived_rubber', 'derived_soya', 'derived_wood'
    ));

-- DDS type constraint per EU 2023/1115 Articles 4-5
ALTER TABLE eudr_traceability_service.due_diligence_statements
    ADD CONSTRAINT chk_dds_type
    CHECK (dds_type IN (
        'import_placement', 'export', 'making_available',
        'trader_simplified', 'sme_simplified'
    ));

-- Risk level constraint per Article 29 benchmarking
ALTER TABLE eudr_traceability_service.due_diligence_statements
    ADD CONSTRAINT chk_dds_risk_level
    CHECK (risk_level IN ('low', 'standard', 'high'));

-- Status constraint
ALTER TABLE eudr_traceability_service.due_diligence_statements
    ADD CONSTRAINT chk_dds_status
    CHECK (status IN (
        'draft', 'pending_review', 'submitted', 'accepted',
        'rejected', 'withdrawn', 'expired', 'superseded'
    ));

-- Quantity must be positive
ALTER TABLE eudr_traceability_service.due_diligence_statements
    ADD CONSTRAINT chk_dds_quantity_positive
    CHECK (quantity > 0);

-- DDS ID must not be empty
ALTER TABLE eudr_traceability_service.due_diligence_statements
    ADD CONSTRAINT chk_dds_id_not_empty
    CHECK (LENGTH(TRIM(dds_id)) > 0);

-- Operator ID must not be empty
ALTER TABLE eudr_traceability_service.due_diligence_statements
    ADD CONSTRAINT chk_dds_operator_id_not_empty
    CHECK (LENGTH(TRIM(operator_id)) > 0);

-- Operator name must not be empty
ALTER TABLE eudr_traceability_service.due_diligence_statements
    ADD CONSTRAINT chk_dds_operator_name_not_empty
    CHECK (LENGTH(TRIM(operator_name)) > 0);

-- Operator country code must be 2 or 3 characters (ISO 3166)
ALTER TABLE eudr_traceability_service.due_diligence_statements
    ADD CONSTRAINT chk_dds_operator_country_length
    CHECK (LENGTH(TRIM(operator_country)) >= 2 AND LENGTH(TRIM(operator_country)) <= 3);

-- Validity end must be after validity start if both specified
ALTER TABLE eudr_traceability_service.due_diligence_statements
    ADD CONSTRAINT chk_dds_validity_dates
    CHECK (validity_end IS NULL OR validity_start IS NULL OR validity_end >= validity_start);

-- Updated_at trigger for due_diligence_statements
CREATE TRIGGER trg_due_diligence_statements_updated_at
    BEFORE UPDATE ON eudr_traceability_service.due_diligence_statements
    FOR EACH ROW
    EXECUTE FUNCTION eudr_traceability_service.set_updated_at();

-- =============================================================================
-- Table 6: eudr_traceability_service.dds_submissions (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording DDS submission attempts to the EU
-- Information System. Each submission event captures the DDS reference,
-- submission status, EU reference number, request/response payloads,
-- retry count, and error messages. Partitioned by submitted_at for
-- time-series queries. Retained for 730 days with compression after
-- 30 days. Tenant-scoped.

CREATE TABLE eudr_traceability_service.dds_submissions (
    submission_id VARCHAR(50) NOT NULL,
    dds_id VARCHAR(50) NOT NULL,
    submission_status VARCHAR(30) DEFAULT 'pending',
    eu_reference VARCHAR(100),
    submitted_at TIMESTAMPTZ NOT NULL,
    response_at TIMESTAMPTZ,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    request_payload JSONB,
    response_payload JSONB,
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (submission_id, submitted_at)
);

-- Create hypertable partitioned by submitted_at
SELECT create_hypertable('eudr_traceability_service.dds_submissions', 'submitted_at', if_not_exists => TRUE);

-- Submission status constraint
ALTER TABLE eudr_traceability_service.dds_submissions
    ADD CONSTRAINT chk_submission_status
    CHECK (submission_status IN (
        'pending', 'submitted', 'accepted', 'rejected',
        'error', 'timeout', 'retrying', 'cancelled'
    ));

-- Retry count must be non-negative
ALTER TABLE eudr_traceability_service.dds_submissions
    ADD CONSTRAINT chk_submission_retry_count_non_negative
    CHECK (retry_count >= 0);

-- Submission ID must not be empty
ALTER TABLE eudr_traceability_service.dds_submissions
    ADD CONSTRAINT chk_submission_id_not_empty
    CHECK (LENGTH(TRIM(submission_id)) > 0);

-- DDS ID must not be empty
ALTER TABLE eudr_traceability_service.dds_submissions
    ADD CONSTRAINT chk_submission_dds_id_not_empty
    CHECK (LENGTH(TRIM(dds_id)) > 0);

-- =============================================================================
-- Table 7: eudr_traceability_service.risk_assessments
-- =============================================================================
-- Risk assessment records per EU 2023/1115 Article 10. Each assessment
-- captures country risk (Article 29 benchmarking), commodity risk, supplier
-- risk, traceability risk, and an overall composite score. Risk factors and
-- mitigation measures are stored as JSONB arrays. Supports target types
-- including plot, operator, commodity, and country. Tenant-scoped.

CREATE TABLE eudr_traceability_service.risk_assessments (
    assessment_id VARCHAR(50) PRIMARY KEY,
    target_type VARCHAR(20) NOT NULL,
    target_id VARCHAR(100) NOT NULL,
    country_risk_score NUMERIC(6,2) DEFAULT 0,
    commodity_risk_score NUMERIC(6,2) DEFAULT 0,
    supplier_risk_score NUMERIC(6,2) DEFAULT 0,
    traceability_risk_score NUMERIC(6,2) DEFAULT 0,
    overall_risk_score NUMERIC(6,2) DEFAULT 0,
    risk_level VARCHAR(20) DEFAULT 'standard',
    risk_factors JSONB DEFAULT '[]'::jsonb,
    mitigation_measures JSONB DEFAULT '[]'::jsonb,
    methodology VARCHAR(100) DEFAULT 'EUDR_Standard_Risk_Assessment_v1',
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Target type constraint
ALTER TABLE eudr_traceability_service.risk_assessments
    ADD CONSTRAINT chk_risk_target_type
    CHECK (target_type IN (
        'plot', 'operator', 'commodity', 'country', 'supplier', 'batch'
    ));

-- Risk level constraint
ALTER TABLE eudr_traceability_service.risk_assessments
    ADD CONSTRAINT chk_risk_level
    CHECK (risk_level IN ('low', 'standard', 'high'));

-- Risk scores must be between 0 and 100
ALTER TABLE eudr_traceability_service.risk_assessments
    ADD CONSTRAINT chk_risk_country_score_range
    CHECK (country_risk_score >= 0 AND country_risk_score <= 100);

ALTER TABLE eudr_traceability_service.risk_assessments
    ADD CONSTRAINT chk_risk_commodity_score_range
    CHECK (commodity_risk_score >= 0 AND commodity_risk_score <= 100);

ALTER TABLE eudr_traceability_service.risk_assessments
    ADD CONSTRAINT chk_risk_supplier_score_range
    CHECK (supplier_risk_score >= 0 AND supplier_risk_score <= 100);

ALTER TABLE eudr_traceability_service.risk_assessments
    ADD CONSTRAINT chk_risk_traceability_score_range
    CHECK (traceability_risk_score >= 0 AND traceability_risk_score <= 100);

ALTER TABLE eudr_traceability_service.risk_assessments
    ADD CONSTRAINT chk_risk_overall_score_range
    CHECK (overall_risk_score >= 0 AND overall_risk_score <= 100);

-- Assessment ID must not be empty
ALTER TABLE eudr_traceability_service.risk_assessments
    ADD CONSTRAINT chk_risk_assessment_id_not_empty
    CHECK (LENGTH(TRIM(assessment_id)) > 0);

-- Target ID must not be empty
ALTER TABLE eudr_traceability_service.risk_assessments
    ADD CONSTRAINT chk_risk_target_id_not_empty
    CHECK (LENGTH(TRIM(target_id)) > 0);

-- =============================================================================
-- Table 8: eudr_traceability_service.commodity_classifications
-- =============================================================================
-- Commodity classification records mapping products to EUDR-regulated
-- commodities, CN codes (Combined Nomenclature), and HS codes
-- (Harmonized System). Supports derived product identification and
-- product composition tracking for mixed/processed goods. Tenant-scoped.

CREATE TABLE eudr_traceability_service.commodity_classifications (
    classification_id VARCHAR(50) PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    commodity VARCHAR(20) NOT NULL,
    cn_code VARCHAR(20),
    hs_code VARCHAR(20),
    is_derived_product BOOLEAN DEFAULT FALSE,
    primary_commodity VARCHAR(20),
    product_composition JSONB,
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    classified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Commodity constraint
ALTER TABLE eudr_traceability_service.commodity_classifications
    ADD CONSTRAINT chk_classification_commodity
    CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood',
        'derived_cattle', 'derived_cocoa', 'derived_coffee', 'derived_oil_palm',
        'derived_rubber', 'derived_soya', 'derived_wood'
    ));

-- Primary commodity constraint (same values, nullable)
ALTER TABLE eudr_traceability_service.commodity_classifications
    ADD CONSTRAINT chk_classification_primary_commodity
    CHECK (primary_commodity IS NULL OR primary_commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    ));

-- Classification ID must not be empty
ALTER TABLE eudr_traceability_service.commodity_classifications
    ADD CONSTRAINT chk_classification_id_not_empty
    CHECK (LENGTH(TRIM(classification_id)) > 0);

-- Product name must not be empty
ALTER TABLE eudr_traceability_service.commodity_classifications
    ADD CONSTRAINT chk_classification_product_name_not_empty
    CHECK (LENGTH(TRIM(product_name)) > 0);

-- =============================================================================
-- Table 9: eudr_traceability_service.supplier_declarations
-- =============================================================================
-- Supplier declaration records confirming deforestation-free production,
-- legal compliance, and traceability for EUDR-regulated commodities.
-- Each declaration captures supplier details, covered commodities,
-- confirmation flags, validity period, documentation provided, and
-- signatory information. Used as supporting evidence for DDS. Tenant-scoped.

CREATE TABLE eudr_traceability_service.supplier_declarations (
    declaration_id VARCHAR(50) PRIMARY KEY,
    supplier_id VARCHAR(100) NOT NULL,
    supplier_name VARCHAR(255) NOT NULL,
    supplier_country VARCHAR(3) NOT NULL,
    declaration_date DATE NOT NULL,
    commodities_covered JSONB NOT NULL DEFAULT '[]'::jsonb,
    confirms_deforestation_free BOOLEAN DEFAULT FALSE,
    confirms_legal_production BOOLEAN DEFAULT FALSE,
    confirms_traceability BOOLEAN DEFAULT FALSE,
    valid_from DATE NOT NULL,
    valid_until DATE,
    documentation_provided JSONB DEFAULT '[]'::jsonb,
    signatory_name VARCHAR(255),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Declaration ID must not be empty
ALTER TABLE eudr_traceability_service.supplier_declarations
    ADD CONSTRAINT chk_declaration_id_not_empty
    CHECK (LENGTH(TRIM(declaration_id)) > 0);

-- Supplier ID must not be empty
ALTER TABLE eudr_traceability_service.supplier_declarations
    ADD CONSTRAINT chk_declaration_supplier_id_not_empty
    CHECK (LENGTH(TRIM(supplier_id)) > 0);

-- Supplier name must not be empty
ALTER TABLE eudr_traceability_service.supplier_declarations
    ADD CONSTRAINT chk_declaration_supplier_name_not_empty
    CHECK (LENGTH(TRIM(supplier_name)) > 0);

-- Supplier country code must be 2 or 3 characters (ISO 3166)
ALTER TABLE eudr_traceability_service.supplier_declarations
    ADD CONSTRAINT chk_declaration_supplier_country_length
    CHECK (LENGTH(TRIM(supplier_country)) >= 2 AND LENGTH(TRIM(supplier_country)) <= 3);

-- Valid_until must be after valid_from if specified
ALTER TABLE eudr_traceability_service.supplier_declarations
    ADD CONSTRAINT chk_declaration_validity_dates
    CHECK (valid_until IS NULL OR valid_until >= valid_from);

-- =============================================================================
-- Table 10: eudr_traceability_service.compliance_checks (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording article-level compliance verification
-- checks against EU 2023/1115 requirements. Each check record captures the
-- target entity, article reference, requirement description, compliance
-- status, details, and remediation guidance. Partitioned by checked_at
-- for time-series queries. Retained for 730 days with compression after
-- 30 days. Tenant-scoped.

CREATE TABLE eudr_traceability_service.compliance_checks (
    check_id VARCHAR(50) NOT NULL,
    target_type VARCHAR(20) NOT NULL,
    target_id VARCHAR(100) NOT NULL,
    article_checked VARCHAR(20) NOT NULL,
    requirement TEXT NOT NULL,
    is_compliant BOOLEAN DEFAULT FALSE,
    details TEXT,
    remediation TEXT,
    checked_at TIMESTAMPTZ NOT NULL,
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (check_id, checked_at)
);

-- Create hypertable partitioned by checked_at
SELECT create_hypertable('eudr_traceability_service.compliance_checks', 'checked_at', if_not_exists => TRUE);

-- Target type constraint
ALTER TABLE eudr_traceability_service.compliance_checks
    ADD CONSTRAINT chk_compliance_target_type
    CHECK (target_type IN (
        'plot', 'transfer', 'batch', 'dds', 'operator',
        'supplier', 'commodity', 'declaration'
    ));

-- Article checked must reference EU 2023/1115 articles
ALTER TABLE eudr_traceability_service.compliance_checks
    ADD CONSTRAINT chk_compliance_article_not_empty
    CHECK (LENGTH(TRIM(article_checked)) > 0);

-- Requirement must not be empty
ALTER TABLE eudr_traceability_service.compliance_checks
    ADD CONSTRAINT chk_compliance_requirement_not_empty
    CHECK (LENGTH(TRIM(requirement)) > 0);

-- Check ID must not be empty
ALTER TABLE eudr_traceability_service.compliance_checks
    ADD CONSTRAINT chk_compliance_check_id_not_empty
    CHECK (LENGTH(TRIM(check_id)) > 0);

-- Target ID must not be empty
ALTER TABLE eudr_traceability_service.compliance_checks
    ADD CONSTRAINT chk_compliance_target_id_not_empty
    CHECK (LENGTH(TRIM(target_id)) > 0);

-- =============================================================================
-- Continuous Aggregate: eudr_traceability_service.eudr_compliance_checks_hourly
-- =============================================================================
-- Precomputed hourly compliance check statistics by target type, article,
-- and compliance status for dashboard queries, trend analysis, and SLI tracking.

CREATE MATERIALIZED VIEW eudr_traceability_service.eudr_compliance_checks_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', checked_at) AS bucket,
    target_type,
    article_checked,
    tenant_id,
    COUNT(*) AS total_checks,
    COUNT(*) FILTER (WHERE is_compliant = TRUE) AS compliant_count,
    COUNT(*) FILTER (WHERE is_compliant = FALSE) AS non_compliant_count,
    COUNT(DISTINCT target_id) AS unique_targets
FROM eudr_traceability_service.compliance_checks
WHERE checked_at IS NOT NULL
GROUP BY bucket, target_type, article_checked, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('eudr_traceability_service.eudr_compliance_checks_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Continuous Aggregate: eudr_traceability_service.eudr_dds_submissions_hourly
-- =============================================================================
-- Precomputed hourly DDS submission statistics by submission status for
-- monitoring EU IS integration health, retry rates, and submission throughput.

CREATE MATERIALIZED VIEW eudr_traceability_service.eudr_dds_submissions_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', submitted_at) AS bucket,
    submission_status,
    tenant_id,
    COUNT(*) AS total_submissions,
    COUNT(*) FILTER (WHERE submission_status = 'accepted') AS accepted_count,
    COUNT(*) FILTER (WHERE submission_status = 'rejected') AS rejected_count,
    COUNT(*) FILTER (WHERE submission_status = 'error') AS error_count,
    SUM(retry_count) AS total_retries,
    AVG(retry_count) AS avg_retries,
    COUNT(DISTINCT dds_id) AS unique_dds
FROM eudr_traceability_service.dds_submissions
WHERE submitted_at IS NOT NULL
GROUP BY bucket, submission_status, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('eudr_traceability_service.eudr_dds_submissions_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- production_plots indexes
CREATE INDEX idx_pp_commodity ON eudr_traceability_service.production_plots(commodity);
CREATE INDEX idx_pp_country_code ON eudr_traceability_service.production_plots(country_code);
CREATE INDEX idx_pp_risk_level ON eudr_traceability_service.production_plots(risk_level);
CREATE INDEX idx_pp_tenant ON eudr_traceability_service.production_plots(tenant_id);
CREATE INDEX idx_pp_lat_lon ON eudr_traceability_service.production_plots(latitude, longitude);
CREATE INDEX idx_pp_deforestation_free ON eudr_traceability_service.production_plots(deforestation_free);
CREATE INDEX idx_pp_created_at ON eudr_traceability_service.production_plots(created_at DESC);
CREATE INDEX idx_pp_updated_at ON eudr_traceability_service.production_plots(updated_at DESC);
CREATE INDEX idx_pp_producer_id ON eudr_traceability_service.production_plots(producer_id);
CREATE INDEX idx_pp_production_date ON eudr_traceability_service.production_plots(production_date DESC);
CREATE INDEX idx_pp_harvest_date ON eudr_traceability_service.production_plots(harvest_date DESC);
CREATE INDEX idx_pp_legal_compliance ON eudr_traceability_service.production_plots(legal_compliance);
CREATE INDEX idx_pp_land_use_type ON eudr_traceability_service.production_plots(land_use_type);
CREATE INDEX idx_pp_tenant_commodity ON eudr_traceability_service.production_plots(tenant_id, commodity);
CREATE INDEX idx_pp_tenant_country ON eudr_traceability_service.production_plots(tenant_id, country_code);
CREATE INDEX idx_pp_tenant_risk ON eudr_traceability_service.production_plots(tenant_id, risk_level);
CREATE INDEX idx_pp_polygon_geojson ON eudr_traceability_service.production_plots USING GIN (polygon_geojson);
CREATE INDEX idx_pp_supporting_docs ON eudr_traceability_service.production_plots USING GIN (supporting_documents);
CREATE INDEX idx_pp_metadata ON eudr_traceability_service.production_plots USING GIN (metadata);

-- plot_monitoring indexes (hypertable-aware)
CREATE INDEX idx_pm_plot ON eudr_traceability_service.plot_monitoring(plot_id, monitored_at DESC);
CREATE INDEX idx_pm_monitoring_type ON eudr_traceability_service.plot_monitoring(monitoring_type, monitored_at DESC);
CREATE INDEX idx_pm_deforestation ON eudr_traceability_service.plot_monitoring(deforestation_detected, monitored_at DESC);
CREATE INDEX idx_pm_satellite_source ON eudr_traceability_service.plot_monitoring(satellite_source, monitored_at DESC);
CREATE INDEX idx_pm_assessor ON eudr_traceability_service.plot_monitoring(assessor, monitored_at DESC);
CREATE INDEX idx_pm_tenant ON eudr_traceability_service.plot_monitoring(tenant_id, monitored_at DESC);
CREATE INDEX idx_pm_tenant_plot ON eudr_traceability_service.plot_monitoring(tenant_id, plot_id, monitored_at DESC);
CREATE INDEX idx_pm_metadata ON eudr_traceability_service.plot_monitoring USING GIN (metadata);

-- custody_transfers indexes
CREATE INDEX idx_ct_commodity ON eudr_traceability_service.custody_transfers(commodity);
CREATE INDEX idx_ct_source_operator ON eudr_traceability_service.custody_transfers(source_operator_id);
CREATE INDEX idx_ct_target_operator ON eudr_traceability_service.custody_transfers(target_operator_id);
CREATE INDEX idx_ct_batch_number ON eudr_traceability_service.custody_transfers(batch_number);
CREATE INDEX idx_ct_custody_model ON eudr_traceability_service.custody_transfers(custody_model);
CREATE INDEX idx_ct_transaction_date ON eudr_traceability_service.custody_transfers(transaction_date DESC);
CREATE INDEX idx_ct_transaction_id ON eudr_traceability_service.custody_transfers(transaction_id);
CREATE INDEX idx_ct_verification_status ON eudr_traceability_service.custody_transfers(verification_status);
CREATE INDEX idx_ct_cn_code ON eudr_traceability_service.custody_transfers(cn_code);
CREATE INDEX idx_ct_hs_code ON eudr_traceability_service.custody_transfers(hs_code);
CREATE INDEX idx_ct_transport_mode ON eudr_traceability_service.custody_transfers(transport_mode);
CREATE INDEX idx_ct_created_at ON eudr_traceability_service.custody_transfers(created_at DESC);
CREATE INDEX idx_ct_tenant ON eudr_traceability_service.custody_transfers(tenant_id);
CREATE INDEX idx_ct_tenant_commodity ON eudr_traceability_service.custody_transfers(tenant_id, commodity);
CREATE INDEX idx_ct_tenant_source ON eudr_traceability_service.custody_transfers(tenant_id, source_operator_id);
CREATE INDEX idx_ct_tenant_target ON eudr_traceability_service.custody_transfers(tenant_id, target_operator_id);
CREATE INDEX idx_ct_origin_plot_ids ON eudr_traceability_service.custody_transfers USING GIN (origin_plot_ids);
CREATE INDEX idx_ct_transport_documents ON eudr_traceability_service.custody_transfers USING GIN (transport_documents);
CREATE INDEX idx_ct_metadata ON eudr_traceability_service.custody_transfers USING GIN (metadata);

-- custody_batches indexes
CREATE INDEX idx_cb_commodity ON eudr_traceability_service.custody_batches(commodity);
CREATE INDEX idx_cb_custody_model ON eudr_traceability_service.custody_batches(custody_model);
CREATE INDEX idx_cb_batch_status ON eudr_traceability_service.custody_batches(batch_status);
CREATE INDEX idx_cb_created_at ON eudr_traceability_service.custody_batches(created_at DESC);
CREATE INDEX idx_cb_tenant ON eudr_traceability_service.custody_batches(tenant_id);
CREATE INDEX idx_cb_tenant_commodity ON eudr_traceability_service.custody_batches(tenant_id, commodity);
CREATE INDEX idx_cb_tenant_status ON eudr_traceability_service.custody_batches(tenant_id, batch_status);
CREATE INDEX idx_cb_parent_batch_ids ON eudr_traceability_service.custody_batches USING GIN (parent_batch_ids);
CREATE INDEX idx_cb_origin_plot_ids ON eudr_traceability_service.custody_batches USING GIN (origin_plot_ids);
CREATE INDEX idx_cb_metadata ON eudr_traceability_service.custody_batches USING GIN (metadata);

-- due_diligence_statements indexes
CREATE INDEX idx_dds_operator_id ON eudr_traceability_service.due_diligence_statements(operator_id);
CREATE INDEX idx_dds_commodity ON eudr_traceability_service.due_diligence_statements(commodity);
CREATE INDEX idx_dds_status ON eudr_traceability_service.due_diligence_statements(status);
CREATE INDEX idx_dds_eu_reference ON eudr_traceability_service.due_diligence_statements(eu_reference_number);
CREATE INDEX idx_dds_submission_date ON eudr_traceability_service.due_diligence_statements(submission_date DESC);
CREATE INDEX idx_dds_dds_type ON eudr_traceability_service.due_diligence_statements(dds_type);
CREATE INDEX idx_dds_risk_level ON eudr_traceability_service.due_diligence_statements(risk_level);
CREATE INDEX idx_dds_risk_assessment ON eudr_traceability_service.due_diligence_statements(risk_assessment_id);
CREATE INDEX idx_dds_operator_country ON eudr_traceability_service.due_diligence_statements(operator_country);
CREATE INDEX idx_dds_validity_start ON eudr_traceability_service.due_diligence_statements(validity_start);
CREATE INDEX idx_dds_validity_end ON eudr_traceability_service.due_diligence_statements(validity_end);
CREATE INDEX idx_dds_created_at ON eudr_traceability_service.due_diligence_statements(created_at DESC);
CREATE INDEX idx_dds_updated_at ON eudr_traceability_service.due_diligence_statements(updated_at DESC);
CREATE INDEX idx_dds_tenant ON eudr_traceability_service.due_diligence_statements(tenant_id);
CREATE INDEX idx_dds_tenant_commodity ON eudr_traceability_service.due_diligence_statements(tenant_id, commodity);
CREATE INDEX idx_dds_tenant_status ON eudr_traceability_service.due_diligence_statements(tenant_id, status);
CREATE INDEX idx_dds_tenant_operator ON eudr_traceability_service.due_diligence_statements(tenant_id, operator_id);
CREATE INDEX idx_dds_cn_codes ON eudr_traceability_service.due_diligence_statements USING GIN (cn_codes);
CREATE INDEX idx_dds_origin_countries ON eudr_traceability_service.due_diligence_statements USING GIN (origin_countries);
CREATE INDEX idx_dds_origin_plot_ids ON eudr_traceability_service.due_diligence_statements USING GIN (origin_plot_ids);
CREATE INDEX idx_dds_custody_transfer_ids ON eudr_traceability_service.due_diligence_statements USING GIN (custody_transfer_ids);
CREATE INDEX idx_dds_risk_mitigation ON eudr_traceability_service.due_diligence_statements USING GIN (risk_mitigation_measures);
CREATE INDEX idx_dds_supporting_evidence ON eudr_traceability_service.due_diligence_statements USING GIN (supporting_evidence);
CREATE INDEX idx_dds_metadata ON eudr_traceability_service.due_diligence_statements USING GIN (metadata);

-- dds_submissions indexes (hypertable-aware)
CREATE INDEX idx_dsub_dds ON eudr_traceability_service.dds_submissions(dds_id, submitted_at DESC);
CREATE INDEX idx_dsub_status ON eudr_traceability_service.dds_submissions(submission_status, submitted_at DESC);
CREATE INDEX idx_dsub_eu_reference ON eudr_traceability_service.dds_submissions(eu_reference, submitted_at DESC);
CREATE INDEX idx_dsub_tenant ON eudr_traceability_service.dds_submissions(tenant_id, submitted_at DESC);
CREATE INDEX idx_dsub_tenant_dds ON eudr_traceability_service.dds_submissions(tenant_id, dds_id, submitted_at DESC);
CREATE INDEX idx_dsub_tenant_status ON eudr_traceability_service.dds_submissions(tenant_id, submission_status, submitted_at DESC);
CREATE INDEX idx_dsub_request_payload ON eudr_traceability_service.dds_submissions USING GIN (request_payload);
CREATE INDEX idx_dsub_response_payload ON eudr_traceability_service.dds_submissions USING GIN (response_payload);
CREATE INDEX idx_dsub_metadata ON eudr_traceability_service.dds_submissions USING GIN (metadata);

-- risk_assessments indexes
CREATE INDEX idx_ra_target_type_id ON eudr_traceability_service.risk_assessments(target_type, target_id);
CREATE INDEX idx_ra_risk_level ON eudr_traceability_service.risk_assessments(risk_level);
CREATE INDEX idx_ra_assessed_at ON eudr_traceability_service.risk_assessments(assessed_at DESC);
CREATE INDEX idx_ra_overall_score ON eudr_traceability_service.risk_assessments(overall_risk_score DESC);
CREATE INDEX idx_ra_country_score ON eudr_traceability_service.risk_assessments(country_risk_score DESC);
CREATE INDEX idx_ra_methodology ON eudr_traceability_service.risk_assessments(methodology);
CREATE INDEX idx_ra_tenant ON eudr_traceability_service.risk_assessments(tenant_id);
CREATE INDEX idx_ra_tenant_type ON eudr_traceability_service.risk_assessments(tenant_id, target_type);
CREATE INDEX idx_ra_tenant_risk ON eudr_traceability_service.risk_assessments(tenant_id, risk_level);
CREATE INDEX idx_ra_risk_factors ON eudr_traceability_service.risk_assessments USING GIN (risk_factors);
CREATE INDEX idx_ra_mitigation ON eudr_traceability_service.risk_assessments USING GIN (mitigation_measures);
CREATE INDEX idx_ra_metadata ON eudr_traceability_service.risk_assessments USING GIN (metadata);

-- commodity_classifications indexes
CREATE INDEX idx_cc_commodity ON eudr_traceability_service.commodity_classifications(commodity);
CREATE INDEX idx_cc_cn_code ON eudr_traceability_service.commodity_classifications(cn_code);
CREATE INDEX idx_cc_hs_code ON eudr_traceability_service.commodity_classifications(hs_code);
CREATE INDEX idx_cc_is_derived ON eudr_traceability_service.commodity_classifications(is_derived_product);
CREATE INDEX idx_cc_primary_commodity ON eudr_traceability_service.commodity_classifications(primary_commodity);
CREATE INDEX idx_cc_product_name ON eudr_traceability_service.commodity_classifications(product_name);
CREATE INDEX idx_cc_classified_at ON eudr_traceability_service.commodity_classifications(classified_at DESC);
CREATE INDEX idx_cc_tenant ON eudr_traceability_service.commodity_classifications(tenant_id);
CREATE INDEX idx_cc_tenant_commodity ON eudr_traceability_service.commodity_classifications(tenant_id, commodity);
CREATE INDEX idx_cc_product_composition ON eudr_traceability_service.commodity_classifications USING GIN (product_composition);
CREATE INDEX idx_cc_metadata ON eudr_traceability_service.commodity_classifications USING GIN (metadata);

-- supplier_declarations indexes
CREATE INDEX idx_sd_supplier_id ON eudr_traceability_service.supplier_declarations(supplier_id);
CREATE INDEX idx_sd_supplier_country ON eudr_traceability_service.supplier_declarations(supplier_country);
CREATE INDEX idx_sd_declaration_date ON eudr_traceability_service.supplier_declarations(declaration_date DESC);
CREATE INDEX idx_sd_valid_from ON eudr_traceability_service.supplier_declarations(valid_from);
CREATE INDEX idx_sd_valid_until ON eudr_traceability_service.supplier_declarations(valid_until);
CREATE INDEX idx_sd_created_at ON eudr_traceability_service.supplier_declarations(created_at DESC);
CREATE INDEX idx_sd_tenant ON eudr_traceability_service.supplier_declarations(tenant_id);
CREATE INDEX idx_sd_tenant_supplier ON eudr_traceability_service.supplier_declarations(tenant_id, supplier_id);
CREATE INDEX idx_sd_tenant_country ON eudr_traceability_service.supplier_declarations(tenant_id, supplier_country);
CREATE INDEX idx_sd_commodities ON eudr_traceability_service.supplier_declarations USING GIN (commodities_covered);
CREATE INDEX idx_sd_documentation ON eudr_traceability_service.supplier_declarations USING GIN (documentation_provided);
CREATE INDEX idx_sd_metadata ON eudr_traceability_service.supplier_declarations USING GIN (metadata);

-- compliance_checks indexes (hypertable-aware)
CREATE INDEX idx_chk_target_type ON eudr_traceability_service.compliance_checks(target_type, checked_at DESC);
CREATE INDEX idx_chk_target_id ON eudr_traceability_service.compliance_checks(target_id, checked_at DESC);
CREATE INDEX idx_chk_article ON eudr_traceability_service.compliance_checks(article_checked, checked_at DESC);
CREATE INDEX idx_chk_compliant ON eudr_traceability_service.compliance_checks(is_compliant, checked_at DESC);
CREATE INDEX idx_chk_tenant ON eudr_traceability_service.compliance_checks(tenant_id, checked_at DESC);
CREATE INDEX idx_chk_tenant_type ON eudr_traceability_service.compliance_checks(tenant_id, target_type, checked_at DESC);
CREATE INDEX idx_chk_tenant_article ON eudr_traceability_service.compliance_checks(tenant_id, article_checked, checked_at DESC);
CREATE INDEX idx_chk_metadata ON eudr_traceability_service.compliance_checks USING GIN (metadata);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE eudr_traceability_service.production_plots ENABLE ROW LEVEL SECURITY;
CREATE POLICY pp_tenant_read ON eudr_traceability_service.production_plots
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY pp_tenant_write ON eudr_traceability_service.production_plots
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE eudr_traceability_service.plot_monitoring ENABLE ROW LEVEL SECURITY;
CREATE POLICY pm_tenant_read ON eudr_traceability_service.plot_monitoring
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY pm_tenant_write ON eudr_traceability_service.plot_monitoring
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE eudr_traceability_service.custody_transfers ENABLE ROW LEVEL SECURITY;
CREATE POLICY ct_tenant_read ON eudr_traceability_service.custody_transfers
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ct_tenant_write ON eudr_traceability_service.custody_transfers
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE eudr_traceability_service.custody_batches ENABLE ROW LEVEL SECURITY;
CREATE POLICY cb_tenant_read ON eudr_traceability_service.custody_batches
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY cb_tenant_write ON eudr_traceability_service.custody_batches
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE eudr_traceability_service.due_diligence_statements ENABLE ROW LEVEL SECURITY;
CREATE POLICY dds_tenant_read ON eudr_traceability_service.due_diligence_statements
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY dds_tenant_write ON eudr_traceability_service.due_diligence_statements
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE eudr_traceability_service.dds_submissions ENABLE ROW LEVEL SECURITY;
CREATE POLICY dsub_tenant_read ON eudr_traceability_service.dds_submissions
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY dsub_tenant_write ON eudr_traceability_service.dds_submissions
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE eudr_traceability_service.risk_assessments ENABLE ROW LEVEL SECURITY;
CREATE POLICY ra_tenant_read ON eudr_traceability_service.risk_assessments
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ra_tenant_write ON eudr_traceability_service.risk_assessments
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE eudr_traceability_service.commodity_classifications ENABLE ROW LEVEL SECURITY;
CREATE POLICY cc_tenant_read ON eudr_traceability_service.commodity_classifications
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY cc_tenant_write ON eudr_traceability_service.commodity_classifications
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE eudr_traceability_service.supplier_declarations ENABLE ROW LEVEL SECURITY;
CREATE POLICY sd_tenant_read ON eudr_traceability_service.supplier_declarations
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY sd_tenant_write ON eudr_traceability_service.supplier_declarations
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE eudr_traceability_service.compliance_checks ENABLE ROW LEVEL SECURITY;
CREATE POLICY chk_tenant_read ON eudr_traceability_service.compliance_checks
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY chk_tenant_write ON eudr_traceability_service.compliance_checks
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA eudr_traceability_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA eudr_traceability_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA eudr_traceability_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON eudr_traceability_service.eudr_compliance_checks_hourly TO greenlang_app;
GRANT SELECT ON eudr_traceability_service.eudr_dds_submissions_hourly TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA eudr_traceability_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA eudr_traceability_service TO greenlang_readonly;
GRANT SELECT ON eudr_traceability_service.eudr_compliance_checks_hourly TO greenlang_readonly;
GRANT SELECT ON eudr_traceability_service.eudr_dds_submissions_hourly TO greenlang_readonly;

-- Add EUDR traceability service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'eudr_traceability:plots:read', 'eudr_traceability', 'plots_read', 'View production plots and geolocation data'),
    (gen_random_uuid(), 'eudr_traceability:plots:write', 'eudr_traceability', 'plots_write', 'Create and manage production plots'),
    (gen_random_uuid(), 'eudr_traceability:monitoring:read', 'eudr_traceability', 'monitoring_read', 'View plot monitoring events and satellite data'),
    (gen_random_uuid(), 'eudr_traceability:monitoring:write', 'eudr_traceability', 'monitoring_write', 'Create and manage monitoring events'),
    (gen_random_uuid(), 'eudr_traceability:custody:read', 'eudr_traceability', 'custody_read', 'View chain of custody transfers and batches'),
    (gen_random_uuid(), 'eudr_traceability:custody:write', 'eudr_traceability', 'custody_write', 'Create and manage custody transfers and batches'),
    (gen_random_uuid(), 'eudr_traceability:dds:read', 'eudr_traceability', 'dds_read', 'View due diligence statements and submissions'),
    (gen_random_uuid(), 'eudr_traceability:dds:write', 'eudr_traceability', 'dds_write', 'Create and manage due diligence statements'),
    (gen_random_uuid(), 'eudr_traceability:risk:read', 'eudr_traceability', 'risk_read', 'View risk assessments and scores'),
    (gen_random_uuid(), 'eudr_traceability:risk:write', 'eudr_traceability', 'risk_write', 'Create and manage risk assessments'),
    (gen_random_uuid(), 'eudr_traceability:compliance:read', 'eudr_traceability', 'compliance_read', 'View compliance checks and results'),
    (gen_random_uuid(), 'eudr_traceability:compliance:write', 'eudr_traceability', 'compliance_write', 'Create and manage compliance checks'),
    (gen_random_uuid(), 'eudr_traceability:declarations:read', 'eudr_traceability', 'declarations_read', 'View supplier declarations'),
    (gen_random_uuid(), 'eudr_traceability:declarations:write', 'eudr_traceability', 'declarations_write', 'Create and manage supplier declarations'),
    (gen_random_uuid(), 'eudr_traceability:classifications:read', 'eudr_traceability', 'classifications_read', 'View commodity classifications and CN/HS codes'),
    (gen_random_uuid(), 'eudr_traceability:classifications:write', 'eudr_traceability', 'classifications_write', 'Create and manage commodity classifications'),
    (gen_random_uuid(), 'eudr_traceability:admin', 'eudr_traceability', 'admin', 'EUDR traceability service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep plot monitoring records for 730 days (2 years)
SELECT add_retention_policy('eudr_traceability_service.plot_monitoring', INTERVAL '730 days');

-- Keep DDS submission records for 730 days (2 years)
SELECT add_retention_policy('eudr_traceability_service.dds_submissions', INTERVAL '730 days');

-- Keep compliance check records for 730 days (2 years)
SELECT add_retention_policy('eudr_traceability_service.compliance_checks', INTERVAL '730 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on plot_monitoring after 30 days
ALTER TABLE eudr_traceability_service.plot_monitoring SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'monitored_at DESC'
);

SELECT add_compression_policy('eudr_traceability_service.plot_monitoring', INTERVAL '30 days');

-- Enable compression on dds_submissions after 30 days
ALTER TABLE eudr_traceability_service.dds_submissions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'submitted_at DESC'
);

SELECT add_compression_policy('eudr_traceability_service.dds_submissions', INTERVAL '30 days');

-- Enable compression on compliance_checks after 30 days
ALTER TABLE eudr_traceability_service.compliance_checks SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'checked_at DESC'
);

SELECT add_compression_policy('eudr_traceability_service.compliance_checks', INTERVAL '30 days');

-- =============================================================================
-- Seed: Register the EUDR Traceability Connector Agent (GL-DATA-EUDR-001)
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES
('GL-DATA-EUDR-001', 'EUDR Traceability Connector',
 'Manages end-to-end traceability for EU Deforestation Regulation (EU 2023/1115) compliance. Tracks production plots with GPS/polygon geolocation, monitors deforestation-free status via satellite imagery and field visits, manages chain of custody transfers between operators with mass balance and segregation models, generates and submits Due Diligence Statements (DDS) to the EU Information System, performs multi-factor risk assessments (country/commodity/supplier/traceability), classifies commodities by CN/HS codes, verifies article-level compliance, and maintains supplier declaration records.',
 2, 'async', true, true, 5, '1.0.0', true,
 'GreenLang Data Team', 'https://docs.greenlang.ai/agents/eudr-traceability', true, 'default')
ON CONFLICT (agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Agent Version for EUDR Traceability Connector
-- =============================================================================

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES
('GL-DATA-EUDR-001', '1.0.0',
 '{"cpu_request": "250m", "cpu_limit": "1000m", "memory_request": "512Mi", "memory_limit": "1Gi", "gpu": false}'::jsonb,
 '{"image": "greenlang/eudr-traceability-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"data", "eudr", "deforestation", "traceability", "due-diligence", "compliance", "supply-chain"}',
 '{"agriculture", "forestry", "cross-sector"}',
 'b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3')
ON CONFLICT (agent_id, version) DO NOTHING;

-- =============================================================================
-- Seed: Agent Capabilities for EUDR Traceability Connector
-- =============================================================================

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES

('GL-DATA-EUDR-001', '1.0.0', 'production_plot_registry', 'data_management',
 'Register and manage production plots with GPS coordinates, polygon boundaries, producer information, harvest details, deforestation-free status, and EU 2023/1115 Article 9 geolocation requirements',
 '{"latitude", "longitude", "commodity", "country_code", "producer_id"}', '{"plot_id", "deforestation_free", "risk_level"}',
 '{"supported_commodities": ["cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"], "geojson_support": true, "deforestation_cutoff": "2020-12-31"}'::jsonb),

('GL-DATA-EUDR-001', '1.0.0', 'plot_monitoring', 'monitoring',
 'Monitor production plots using satellite imagery (Sentinel-2, Landsat, PRODES), field visits, and remote assessments for deforestation detection and forest cover change analysis',
 '{"plot_id", "monitoring_type", "satellite_source"}', '{"monitoring_id", "deforestation_detected", "forest_cover_change_pct"}',
 '{"satellite_sources": ["sentinel_2", "landsat_8", "prodes", "global_forest_watch", "planet"], "detection_threshold_pct": 0.5}'::jsonb),

('GL-DATA-EUDR-001', '1.0.0', 'chain_of_custody_tracking', 'traceability',
 'Track commodity movements between operators with full chain of custody documentation, supporting identity preserved, segregation, mass balance, and controlled blending models per EUDR Article 3',
 '{"source_operator_id", "target_operator_id", "commodity", "quantity"}', '{"transfer_id", "batch_number", "verification_status"}',
 '{"custody_models": ["identity_preserved", "segregation", "mass_balance", "controlled_blending"], "transport_modes": ["sea", "air", "road", "rail", "pipeline", "multimodal"]}'::jsonb),

('GL-DATA-EUDR-001', '1.0.0', 'due_diligence_statement_management', 'compliance',
 'Generate, validate, and manage Due Diligence Statements (DDS) per EU 2023/1115 Articles 4 and 33 with operator information, commodity details, geolocation data, risk assessments, and compliance declarations',
 '{"operator_id", "commodity", "origin_plot_ids", "risk_assessment_id"}', '{"dds_id", "status", "eu_reference_number"}',
 '{"dds_types": ["import_placement", "export", "making_available", "trader_simplified", "sme_simplified"], "auto_risk_assessment": true}'::jsonb),

('GL-DATA-EUDR-001', '1.0.0', 'eu_information_system_submission', 'integration',
 'Submit Due Diligence Statements to the EU Information System (EU IS) with retry logic, status tracking, and response handling per EUDR Article 33',
 '{"dds_id"}', '{"submission_id", "submission_status", "eu_reference"}',
 '{"max_retries": 3, "retry_backoff_seconds": [30, 120, 600], "timeout_seconds": 300}'::jsonb),

('GL-DATA-EUDR-001', '1.0.0', 'risk_assessment', 'analysis',
 'Perform multi-factor risk assessments combining country risk (Article 29 benchmarking), commodity risk, supplier risk, and traceability completeness scoring with configurable methodology',
 '{"target_type", "target_id"}', '{"assessment_id", "overall_risk_score", "risk_level"}',
 '{"risk_factors": ["country_benchmark", "commodity_prevalence", "supplier_history", "traceability_completeness", "certification_status"], "scoring_method": "weighted_average"}'::jsonb),

('GL-DATA-EUDR-001', '1.0.0', 'commodity_classification', 'classification',
 'Classify products into EUDR-regulated commodity categories with CN code (Combined Nomenclature) and HS code (Harmonized System) mapping for derived and composite products',
 '{"product_name", "product_description"}', '{"classification_id", "commodity", "cn_code", "hs_code", "is_derived_product"}',
 '{"auto_classify": true, "cn_code_database": "EU_CN_2024", "hs_code_database": "WCO_HS_2022"}'::jsonb),

('GL-DATA-EUDR-001', '1.0.0', 'compliance_verification', 'verification',
 'Verify compliance against specific EU 2023/1115 articles and requirements with detailed findings, remediation guidance, and compliance scoring',
 '{"target_type", "target_id", "articles_to_check"}', '{"check_results", "overall_compliant", "non_compliant_articles"}',
 '{"supported_articles": ["Art.3", "Art.4", "Art.5", "Art.9", "Art.10", "Art.11", "Art.12", "Art.29", "Art.33"], "include_remediation": true}'::jsonb)

ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: Agent Dependencies for EUDR Traceability Connector
-- =============================================================================

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES

-- EUDR Connector depends on Schema Compiler for input/output validation
('GL-DATA-EUDR-001', 'GL-FOUND-X-002', '>=1.0.0', false,
 'Production plots, custody transfers, DDS, and compliance checks are validated against JSON Schema definitions'),

-- EUDR Connector depends on Unit Normalizer for quantity conversion
('GL-DATA-EUDR-001', 'GL-FOUND-X-003', '>=1.0.0', false,
 'Commodity quantities are normalized to standard units (kg) for consistent traceability across custody chain'),

-- EUDR Connector depends on Registry for agent discovery
('GL-DATA-EUDR-001', 'GL-FOUND-X-007', '>=1.0.0', false,
 'Agent version and capability lookup for EUDR pipeline orchestration'),

-- EUDR Connector depends on Access Guard for policy enforcement
('GL-DATA-EUDR-001', 'GL-FOUND-X-006', '>=1.0.0', false,
 'Data classification and access control enforcement for sensitive geolocation and supplier data'),

-- EUDR Connector optionally uses Citations for provenance tracking
('GL-DATA-EUDR-001', 'GL-FOUND-X-005', '>=1.0.0', true,
 'Production plot data, custody transfers, and DDS provenance chains are registered with the citation service for audit trail'),

-- EUDR Connector optionally uses Reproducibility for determinism
('GL-DATA-EUDR-001', 'GL-FOUND-X-008', '>=1.0.0', true,
 'Risk assessments and compliance checks are verified for reproducibility across re-evaluation runs')

ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Service Catalog Entry for EUDR Traceability Connector
-- =============================================================================

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES
('GL-DATA-EUDR-001', 'EUDR Traceability Connector',
 'End-to-end EU Deforestation Regulation (EU 2023/1115) compliance. Tracks production plots with GPS geolocation, monitors deforestation via satellite, manages chain of custody (mass balance, segregation, identity preserved), generates Due Diligence Statements, submits to EU Information System, performs multi-factor risk assessments, classifies commodities by CN/HS codes, and verifies article-level compliance.',
 'data', 'active', 'default')
ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA eudr_traceability_service IS 'EUDR Traceability Connector for GreenLang Climate OS (AGENT-DATA-005) - EU Deforestation Regulation (EU 2023/1115) compliance with production plot registry, chain of custody tracking, due diligence statements, risk assessments, commodity classification, and compliance verification';
COMMENT ON TABLE eudr_traceability_service.production_plots IS 'Production plot registry with GPS coordinates, polygon boundaries, producer details, commodity type, harvest data, deforestation-free status, and risk level per EU 2023/1115 Article 9';
COMMENT ON TABLE eudr_traceability_service.plot_monitoring IS 'TimescaleDB hypertable: satellite and field monitoring events for production plots with deforestation detection and forest cover change analysis';
COMMENT ON TABLE eudr_traceability_service.custody_transfers IS 'Chain of custody transfer records tracking commodity movements between operators with batch numbers, origin plot references, CN/HS codes, and verification status';
COMMENT ON TABLE eudr_traceability_service.custody_batches IS 'Batch tracking records for mass balance and segregation custody models with parent batch lineage and origin plot references';
COMMENT ON TABLE eudr_traceability_service.due_diligence_statements IS 'Due Diligence Statement (DDS) records per EU 2023/1115 Articles 4 and 33 with operator info, commodity details, risk assessments, and compliance declarations';
COMMENT ON TABLE eudr_traceability_service.dds_submissions IS 'TimescaleDB hypertable: DDS submission attempts to the EU Information System with status tracking, retry counts, and request/response payloads';
COMMENT ON TABLE eudr_traceability_service.risk_assessments IS 'Multi-factor risk assessment records per EU 2023/1115 Article 10 with country, commodity, supplier, and traceability risk scores';
COMMENT ON TABLE eudr_traceability_service.commodity_classifications IS 'Commodity classification records mapping products to EUDR-regulated commodities with CN codes and HS codes for derived/composite products';
COMMENT ON TABLE eudr_traceability_service.supplier_declarations IS 'Supplier deforestation-free, legal production, and traceability declarations with validity periods and supporting documentation';
COMMENT ON TABLE eudr_traceability_service.compliance_checks IS 'TimescaleDB hypertable: article-level compliance verification checks against EU 2023/1115 requirements with remediation guidance';
COMMENT ON MATERIALIZED VIEW eudr_traceability_service.eudr_compliance_checks_hourly IS 'Continuous aggregate: hourly compliance check statistics by target type, article, and compliance status for dashboard queries and SLI tracking';
COMMENT ON MATERIALIZED VIEW eudr_traceability_service.eudr_dds_submissions_hourly IS 'Continuous aggregate: hourly DDS submission statistics by submission status for EU IS integration monitoring';

COMMENT ON COLUMN eudr_traceability_service.production_plots.commodity IS 'EUDR-regulated commodity: cattle, cocoa, coffee, oil_palm, rubber, soya, wood, or derived variants per EU 2023/1115 Article 2';
COMMENT ON COLUMN eudr_traceability_service.production_plots.latitude IS 'GPS latitude of production plot center point (-90 to 90 degrees)';
COMMENT ON COLUMN eudr_traceability_service.production_plots.longitude IS 'GPS longitude of production plot center point (-180 to 180 degrees)';
COMMENT ON COLUMN eudr_traceability_service.production_plots.polygon_geojson IS 'GeoJSON polygon boundary of the production plot for plots larger than 4 hectares per EUDR Article 9';
COMMENT ON COLUMN eudr_traceability_service.production_plots.deforestation_free IS 'Declaration that the plot has not been subject to deforestation after the cutoff date (31 Dec 2020)';
COMMENT ON COLUMN eudr_traceability_service.production_plots.deforestation_cutoff_date IS 'EUDR deforestation cutoff date per Article 2(13): default 31 December 2020';
COMMENT ON COLUMN eudr_traceability_service.production_plots.risk_level IS 'Risk classification: low, standard, high per Article 29 country benchmarking';
COMMENT ON COLUMN eudr_traceability_service.custody_transfers.custody_model IS 'Chain of custody model: identity_preserved, segregation, mass_balance, controlled_blending, book_and_claim';
COMMENT ON COLUMN eudr_traceability_service.custody_transfers.cn_code IS 'EU Combined Nomenclature code for customs classification';
COMMENT ON COLUMN eudr_traceability_service.custody_transfers.hs_code IS 'Harmonized System code for international trade classification';
COMMENT ON COLUMN eudr_traceability_service.custody_transfers.verification_status IS 'Verification status: pending_verification, verified, rejected, under_review, expired, revoked';
COMMENT ON COLUMN eudr_traceability_service.due_diligence_statements.dds_type IS 'DDS type per EU 2023/1115: import_placement, export, making_available, trader_simplified, sme_simplified';
COMMENT ON COLUMN eudr_traceability_service.due_diligence_statements.eu_reference_number IS 'Reference number assigned by the EU Information System upon successful DDS submission';
COMMENT ON COLUMN eudr_traceability_service.due_diligence_statements.risk_level IS 'Risk level per Article 29 benchmarking: low, standard, high';
COMMENT ON COLUMN eudr_traceability_service.risk_assessments.target_type IS 'Assessment target: plot, operator, commodity, country, supplier, batch';
COMMENT ON COLUMN eudr_traceability_service.risk_assessments.overall_risk_score IS 'Composite risk score (0-100) combining country, commodity, supplier, and traceability risk factors';
COMMENT ON COLUMN eudr_traceability_service.risk_assessments.methodology IS 'Risk assessment methodology identifier (default: EUDR_Standard_Risk_Assessment_v1)';
COMMENT ON COLUMN eudr_traceability_service.compliance_checks.article_checked IS 'EU 2023/1115 article reference being verified (e.g., Art.3, Art.4, Art.9, Art.10)';
COMMENT ON COLUMN eudr_traceability_service.compliance_checks.is_compliant IS 'Whether the target entity meets the requirements of the checked article';
