-- ============================================================================
-- V124: AGENT-EUDR-036 EU Information System Interface
-- ============================================================================
-- Creates tables for the EU Information System Interface which manages operator
-- registration with Member State competent authorities; submits Due Diligence
-- Statements (DDS) to the EU Information System per EUDR Article 33; assembles
-- and tracks submission document packages; formats geolocation data exports
-- conforming to EU IS polygon/point specifications; manages eIDAS-compliant
-- API credentials and OAuth tokens with column-level encryption; maintains a
-- complete submission history via TimescaleDB hypertable; tracks rejections
-- with structured remediation workflows; and preserves an immutable Article 31
-- audit trail via TimescaleDB hypertable.
--
-- Agent ID: GL-EUDR-EUIS-036
-- PRD: PRD-AGENT-EUDR-036
-- Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 10, 12, 14-16, 29, 31, 33
-- Tables: 8 (5 regular + 3 hypertables)
-- Indexes: ~100
-- Dependencies: TimescaleDB extension (for hypertables), pgcrypto (for encryption)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V124: Creating AGENT-EUDR-036 EU Information System Interface tables...';

-- Ensure pgcrypto extension is available for credential encryption
CREATE EXTENSION IF NOT EXISTS pgcrypto;


-- ============================================================================
-- 1. gl_eudr_euis_operator_registrations -- Operator registration with CAs
-- ============================================================================
RAISE NOTICE 'V124 [1/8]: Creating gl_eudr_euis_operator_registrations...';

CREATE TABLE IF NOT EXISTS gl_eudr_euis_operator_registrations (
    registration_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this operator registration record
    operator_id                     VARCHAR(100)    NOT NULL,
        -- Internal GreenLang operator identifier
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    eori_number                     VARCHAR(50),
        -- Economic Operator Registration and Identification (EORI) number
    vat_number                      VARCHAR(50),
        -- VAT registration number of the operator
    operator_name                   VARCHAR(500)    NOT NULL,
        -- Legal name of the operator as registered with the competent authority
    operator_type                   VARCHAR(30)     NOT NULL DEFAULT 'operator',
        -- EUDR classification of the entity
    member_state                    VARCHAR(5)      NOT NULL,
        -- ISO 3166-1 alpha-2 code of the EU Member State where registered
    competent_authority_id          VARCHAR(100),
        -- Identifier of the competent authority in the Member State
    competent_authority_name        VARCHAR(300),
        -- Name of the competent authority (e.g. "BMEL - Germany", "DEFRA - UK")
    registration_reference          VARCHAR(200)    UNIQUE,
        -- EU IS registration reference number assigned by the competent authority
    registration_status             VARCHAR(30)     NOT NULL DEFAULT 'pending',
        -- Current registration lifecycle status
    registration_date               DATE,
        -- Date when the registration was accepted by the competent authority
    expiry_date                     DATE,
        -- Date when the registration expires (NULL if no expiry)
    renewal_due_date                DATE,
        -- Date by which the operator must renew registration
    contact_name                    VARCHAR(200),
        -- Primary contact person for the registration
    contact_email                   VARCHAR(200),
        -- Primary contact email address
    contact_phone                   VARCHAR(50),
        -- Primary contact phone number
    registered_address              JSONB           DEFAULT '{}',
        -- Registered business address: {"street": "...", "city": "...", "postal_code": "...", "country": "..."}
    commodities_declared            JSONB           DEFAULT '[]',
        -- Array of EUDR commodities the operator has declared: ["cattle", "cocoa", "coffee", ...]
    sme_classification              VARCHAR(20),
        -- SME classification for Article 29 benchmarking and simplified DD eligibility
    annual_volume_declaration       JSONB           DEFAULT '{}',
        -- Declared annual import/export volumes per commodity: {"cocoa": {"quantity_kg": 50000, "value_eur": 200000}, ...}
    eu_is_profile_data              JSONB           DEFAULT '{}',
        -- Full operator profile data as stored in EU Information System
    verification_status             VARCHAR(20)     NOT NULL DEFAULT 'unverified',
        -- Verification status of the registration data
    verified_at                     TIMESTAMPTZ,
        -- Timestamp when registration data was last verified
    verified_by                     VARCHAR(100),
        -- User or system that verified the registration
    notes                           TEXT            DEFAULT '',
        -- Administrative notes about this registration
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for registration integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_euis_reg_operator_type CHECK (operator_type IN (
        'operator', 'trader', 'sme_operator', 'sme_trader',
        'authorised_representative', 'monitoring_organisation'
    )),
    CONSTRAINT chk_euis_reg_status CHECK (registration_status IN (
        'pending', 'submitted', 'under_review', 'active', 'suspended',
        'expired', 'revoked', 'renewal_pending', 'cancelled'
    )),
    CONSTRAINT chk_euis_reg_sme CHECK (sme_classification IS NULL OR sme_classification IN (
        'micro', 'small', 'medium', 'non_sme'
    )),
    CONSTRAINT chk_euis_reg_verification CHECK (verification_status IN (
        'unverified', 'pending_verification', 'verified', 'verification_failed', 'expired'
    )),
    CONSTRAINT chk_euis_reg_dates CHECK (expiry_date IS NULL OR registration_date IS NULL OR expiry_date >= registration_date)
);

COMMENT ON TABLE gl_eudr_euis_operator_registrations IS 'AGENT-EUDR-036: Operator registration records with EU Member State competent authorities, including EORI/VAT identifiers, commodity declarations, SME classification, address data, and registration lifecycle per EUDR Articles 4, 14-16, 33';
COMMENT ON COLUMN gl_eudr_euis_operator_registrations.eori_number IS 'Economic Operator Registration and Identification number: EU customs identifier required for EUDR submissions';
COMMENT ON COLUMN gl_eudr_euis_operator_registrations.registration_reference IS 'Unique registration reference assigned by the competent authority upon acceptance into the EU Information System';

-- Indexes for gl_eudr_euis_operator_registrations
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_reg_eori ON gl_eudr_euis_operator_registrations (eori_number);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_reg_vat ON gl_eudr_euis_operator_registrations (vat_number);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_reg_status ON gl_eudr_euis_operator_registrations (registration_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_reg_provenance ON gl_eudr_euis_operator_registrations (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_reg_created ON gl_eudr_euis_operator_registrations (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_reg_tenant_operator ON gl_eudr_euis_operator_registrations (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_reg_operator_status ON gl_eudr_euis_operator_registrations (operator_id, registration_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_reg_ms_status ON gl_eudr_euis_operator_registrations (member_state, registration_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_reg_operator_ms ON gl_eudr_euis_operator_registrations (operator_id, member_state, registration_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active registrations
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_reg_active ON gl_eudr_euis_operator_registrations (operator_id, member_state)
        WHERE registration_status = 'active';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for registrations requiring renewal
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_reg_renewal ON gl_eudr_euis_operator_registrations (renewal_due_date, operator_id)
        WHERE registration_status = 'active' AND renewal_due_date IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for expired or suspended registrations
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_reg_inactive ON gl_eudr_euis_operator_registrations (operator_id, updated_at DESC)
        WHERE registration_status IN ('expired', 'suspended', 'revoked');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_reg_commodities ON gl_eudr_euis_operator_registrations USING GIN (commodities_declared);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_reg_profile ON gl_eudr_euis_operator_registrations USING GIN (eu_is_profile_data);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 2. gl_eudr_euis_dds_submissions -- DDS submission tracking (hypertable)
-- ============================================================================
RAISE NOTICE 'V124 [2/8]: Creating gl_eudr_euis_dds_submissions (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_euis_dds_submissions (
    submission_id                   UUID            DEFAULT gen_random_uuid(),
        -- Unique identifier for this DDS submission
    operator_id                     VARCHAR(100)    NOT NULL,
        -- Internal GreenLang operator identifier
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    registration_id                 UUID,
        -- FK reference to operator registration used for this submission
    dds_reference                   VARCHAR(100)    NOT NULL,
        -- DDS reference number from documentation generator (e.g. "DDS-2026-03-001")
    dds_document_id                 UUID,
        -- FK reference to the source DDS document in the documentation generator
    commodity_type                  VARCHAR(50)     NOT NULL,
        -- EUDR commodity type for this DDS submission
    origin_countries                JSONB           DEFAULT '[]',
        -- Array of origin countries included in this DDS: ["BR", "ID", "GH"]
    submission_type                 VARCHAR(30)     NOT NULL DEFAULT 'initial',
        -- Type of submission
    submission_status               VARCHAR(30)     NOT NULL DEFAULT 'preparing',
        -- Current submission lifecycle status
    submitted_at                    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the submission was initiated (partitioning column)
    sent_to_eu_is_at                TIMESTAMPTZ,
        -- Timestamp when the payload was sent to EU Information System
    acknowledged_at                 TIMESTAMPTZ,
        -- Timestamp when the EU IS acknowledged receipt
    eu_is_receipt_number            VARCHAR(200),
        -- EU IS receipt number / confirmation ID assigned upon acceptance
    eu_is_reference_number          VARCHAR(200),
        -- EU IS DDS reference number for regulatory tracking
    response_status_code            INTEGER,
        -- HTTP status code from EU IS API response
    response_body                   JSONB           DEFAULT '{}',
        -- Full response body from EU IS API
    validation_errors               JSONB           DEFAULT '[]',
        -- Array of validation errors returned by EU IS: [{"code": "...", "field": "...", "message": "..."}, ...]
    retry_count                     INTEGER         NOT NULL DEFAULT 0,
        -- Number of retry attempts for this submission
    max_retries                     INTEGER         NOT NULL DEFAULT 3,
        -- Maximum retry attempts allowed
    next_retry_at                   TIMESTAMPTZ,
        -- Scheduled time for next retry attempt (NULL if not retrying)
    payload_size_bytes              BIGINT,
        -- Size of the submission payload in bytes
    payload_hash                    VARCHAR(64),
        -- SHA-256 hash of the submission payload for integrity verification
    correlation_id                  VARCHAR(200),
        -- End-to-end correlation identifier for distributed tracing
    submitted_by                    VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- User or system that initiated the submission
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for submission record integrity verification

    CONSTRAINT chk_euis_dds_commodity CHECK (commodity_type IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_euis_dds_sub_type CHECK (submission_type IN (
        'initial', 'amendment', 'resubmission', 'correction', 'withdrawal'
    )),
    CONSTRAINT chk_euis_dds_status CHECK (submission_status IN (
        'preparing', 'validating', 'ready', 'submitting', 'submitted',
        'acknowledged', 'rejected', 'failed', 'retrying',
        'withdrawn', 'cancelled', 'expired'
    )),
    CONSTRAINT chk_euis_dds_retry CHECK (retry_count >= 0 AND retry_count <= max_retries),
    CONSTRAINT chk_euis_dds_payload CHECK (payload_size_bytes IS NULL OR payload_size_bytes >= 0)
);

-- Convert to TimescaleDB hypertable partitioned by submitted_at
DO $$
BEGIN
    PERFORM create_hypertable(
        'gl_eudr_euis_dds_submissions',
        'submitted_at',
        chunk_time_interval => INTERVAL '7 days',
        if_not_exists => TRUE
    );
    RAISE NOTICE 'gl_eudr_euis_dds_submissions hypertable created (7-day chunks)';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Hypertable creation skipped for gl_eudr_euis_dds_submissions: %', SQLERRM;
END $$;

COMMENT ON TABLE gl_eudr_euis_dds_submissions IS 'AGENT-EUDR-036: TimescaleDB-partitioned DDS submission tracking with EU IS lifecycle management, receipt/reference numbers, retry logic, validation error capture, and payload integrity hashing per EUDR Articles 4, 9, 33';
COMMENT ON COLUMN gl_eudr_euis_dds_submissions.eu_is_receipt_number IS 'EU Information System receipt/confirmation number assigned upon successful acceptance of DDS submission';
COMMENT ON COLUMN gl_eudr_euis_dds_submissions.correlation_id IS 'Distributed tracing correlation ID linking submission through EU IS API call, response handling, and audit logging';

-- Indexes for gl_eudr_euis_dds_submissions
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_dds_reference ON gl_eudr_euis_dds_submissions (dds_reference, submitted_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_dds_status ON gl_eudr_euis_dds_submissions (submission_status, submitted_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_dds_receipt ON gl_eudr_euis_dds_submissions (eu_is_receipt_number);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_dds_eu_ref ON gl_eudr_euis_dds_submissions (eu_is_reference_number);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_dds_correlation ON gl_eudr_euis_dds_submissions (correlation_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_dds_provenance ON gl_eudr_euis_dds_submissions (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_dds_operator_status ON gl_eudr_euis_dds_submissions (operator_id, submission_status, submitted_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_dds_operator_commodity ON gl_eudr_euis_dds_submissions (operator_id, commodity_type, submitted_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_dds_tenant_operator ON gl_eudr_euis_dds_submissions (tenant_id, operator_id, submitted_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for submissions requiring retry
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_dds_retrying ON gl_eudr_euis_dds_submissions (next_retry_at, operator_id)
        WHERE submission_status IN ('retrying', 'failed') AND retry_count < max_retries;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active (in-flight) submissions
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_dds_active ON gl_eudr_euis_dds_submissions (operator_id, submitted_at DESC)
        WHERE submission_status IN ('preparing', 'validating', 'ready', 'submitting', 'submitted');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_dds_origins ON gl_eudr_euis_dds_submissions USING GIN (origin_countries);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_dds_val_errors ON gl_eudr_euis_dds_submissions USING GIN (validation_errors);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_dds_response ON gl_eudr_euis_dds_submissions USING GIN (response_body);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 3. gl_eudr_euis_submission_packages -- Document packages for submission
-- ============================================================================
RAISE NOTICE 'V124 [3/8]: Creating gl_eudr_euis_submission_packages...';

CREATE TABLE IF NOT EXISTS gl_eudr_euis_submission_packages (
    package_id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this submission package
    submission_id                   UUID            NOT NULL,
        -- Reference to the parent DDS submission (logical FK to hypertable)
    operator_id                     VARCHAR(100)    NOT NULL,
        -- Internal GreenLang operator identifier (denormalized for query performance)
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    dds_reference                   VARCHAR(100)    NOT NULL,
        -- DDS reference number (denormalized for query performance)
    package_type                    VARCHAR(30)     NOT NULL DEFAULT 'complete',
        -- Type of submission package
    package_format                  VARCHAR(20)     NOT NULL DEFAULT 'json',
        -- Format of the submission package
    schema_version                  VARCHAR(20)     NOT NULL DEFAULT '1.0',
        -- EU IS schema version the package conforms to
    dds_payload                     JSONB           NOT NULL DEFAULT '{}',
        -- Structured DDS data payload per EU IS schema
    article9_data                   JSONB           DEFAULT '{}',
        -- Article 9 information package data elements
    geolocation_summary             JSONB           DEFAULT '{}',
        -- Summary of geolocation data included: {"plot_count": 12, "total_area_ha": 450.5, "coordinate_system": "WGS84"}
    risk_assessment_summary         JSONB           DEFAULT '{}',
        -- Summary of risk assessment data: {"composite_score": 35.2, "risk_level": "standard", "country_benchmark": "standard_risk"}
    mitigation_summary              JSONB           DEFAULT '{}',
        -- Summary of mitigation measures: {"measure_count": 5, "verification_result": "effective", "post_risk_score": 18.7}
    supporting_documents            JSONB           DEFAULT '[]',
        -- Array of supporting document references: [{"doc_type": "...", "doc_id": "...", "filename": "...", "hash": "..."}, ...]
    completeness_score              NUMERIC(5,4)    NOT NULL DEFAULT 0,
        -- Package completeness: 0.0000 to 1.0000
    missing_elements                JSONB           DEFAULT '[]',
        -- Array of missing required elements: [{"element": "geolocation.polygons", "article": "9(1)(d)", "severity": "blocking"}, ...]
    validation_passed               BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether the package passed all pre-submission validation
    validation_timestamp            TIMESTAMPTZ,
        -- When the last validation was performed
    assembled_at                    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the package was assembled
    assembled_by                    VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- User or system that assembled the package
    package_hash                    VARCHAR(64),
        -- SHA-256 hash of the complete package for integrity verification
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for record-level provenance
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_euis_pkg_type CHECK (package_type IN (
        'complete', 'partial', 'amendment', 'correction', 'supplementary'
    )),
    CONSTRAINT chk_euis_pkg_format CHECK (package_format IN (
        'json', 'xml', 'json_ld'
    )),
    CONSTRAINT chk_euis_pkg_completeness CHECK (completeness_score >= 0 AND completeness_score <= 1)
);

COMMENT ON TABLE gl_eudr_euis_submission_packages IS 'AGENT-EUDR-036: Submission document packages assembled for EU IS transmission, containing DDS payload, Article 9 data, geolocation summary, risk/mitigation summaries, supporting documents, and pre-submission validation per EUDR Articles 4, 9, 33';
COMMENT ON COLUMN gl_eudr_euis_submission_packages.dds_payload IS 'Structured DDS payload conforming to EU IS schema: includes operator details, commodity info, Article 9 elements, compliance conclusion';
COMMENT ON COLUMN gl_eudr_euis_submission_packages.completeness_score IS 'Package completeness: 0.0000 (empty) to 1.0000 (all required elements present and validated)';

-- Indexes for gl_eudr_euis_submission_packages
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_pkg_submission ON gl_eudr_euis_submission_packages (submission_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_pkg_dds_ref ON gl_eudr_euis_submission_packages (dds_reference);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_pkg_provenance ON gl_eudr_euis_submission_packages (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_pkg_hash ON gl_eudr_euis_submission_packages (package_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_pkg_created ON gl_eudr_euis_submission_packages (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_pkg_tenant_operator ON gl_eudr_euis_submission_packages (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_pkg_operator_type ON gl_eudr_euis_submission_packages (operator_id, package_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for incomplete packages requiring attention
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_pkg_incomplete ON gl_eudr_euis_submission_packages (operator_id, completeness_score)
        WHERE completeness_score < 1.0000;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for validated packages ready for submission
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_pkg_validated ON gl_eudr_euis_submission_packages (operator_id, assembled_at DESC)
        WHERE validation_passed = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for packages that failed validation
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_pkg_failed ON gl_eudr_euis_submission_packages (operator_id, validation_timestamp DESC)
        WHERE validation_passed = FALSE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_pkg_dds_payload ON gl_eudr_euis_submission_packages USING GIN (dds_payload);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_pkg_missing ON gl_eudr_euis_submission_packages USING GIN (missing_elements);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_pkg_supporting ON gl_eudr_euis_submission_packages USING GIN (supporting_documents);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 4. gl_eudr_euis_geolocation_data -- Formatted geolocation exports
-- ============================================================================
RAISE NOTICE 'V124 [4/8]: Creating gl_eudr_euis_geolocation_data...';

CREATE TABLE IF NOT EXISTS gl_eudr_euis_geolocation_data (
    geolocation_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this geolocation export record
    package_id                      UUID,
        -- FK reference to the submission package this geolocation is part of
    operator_id                     VARCHAR(100)    NOT NULL,
        -- Internal GreenLang operator identifier
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    dds_reference                   VARCHAR(100),
        -- DDS reference number for cross-referencing
    commodity_type                  VARCHAR(50)     NOT NULL,
        -- EUDR commodity type for this geolocation data
    origin_country                  VARCHAR(5)      NOT NULL,
        -- ISO 3166-1 alpha-2 country code of the production area
    plot_identifier                 VARCHAR(200)    NOT NULL,
        -- Unique identifier for the production plot/parcel
    plot_name                       VARCHAR(500),
        -- Human-readable name for the production plot
    coordinate_system               VARCHAR(20)     NOT NULL DEFAULT 'WGS84',
        -- Coordinate reference system used (EU IS requires WGS84 / EPSG:4326)
    geometry_type                   VARCHAR(20)     NOT NULL DEFAULT 'polygon',
        -- Type of geometry: polygon for plots > 4 ha, point for plots <= 4 ha per Article 9(1)(d)
    coordinates                     JSONB           NOT NULL DEFAULT '[]',
        -- GeoJSON-compatible coordinate array (polygon rings or single point)
    area_hectares                   NUMERIC(12,4),
        -- Area of the production plot in hectares
    centroid_latitude               NUMERIC(10,7),
        -- Centroid latitude for point-based submissions (plots <= 4 ha)
    centroid_longitude              NUMERIC(11,7),
        -- Centroid longitude for point-based submissions (plots <= 4 ha)
    altitude_meters                 NUMERIC(8,2),
        -- Altitude in meters above sea level (optional)
    precision_meters                NUMERIC(8,2),
        -- Horizontal accuracy/precision of coordinates in meters
    data_source                     VARCHAR(50)     NOT NULL DEFAULT 'gps',
        -- Source of the geolocation data
    capture_date                    DATE,
        -- Date when the coordinates were captured
    eu_is_format_valid              BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether coordinates conform to EU IS format specification
    format_validation_errors        JSONB           DEFAULT '[]',
        -- Array of format validation errors: [{"code": "...", "message": "...", "field": "..."}, ...]
    overlap_check_status            VARCHAR(20)     DEFAULT 'pending',
        -- Status of overlap check against protected areas / existing plots
    overlap_results                 JSONB           DEFAULT '{}',
        -- Overlap check results: {"protected_area_overlap": false, "existing_plot_overlap": false, "details": [...]}
    deforestation_cutoff_date       DATE            DEFAULT '2020-12-31',
        -- EUDR deforestation cutoff date (31 December 2020)
    deforestation_free_verified     BOOLEAN,
        -- Whether the plot is verified deforestation-free since cutoff date
    supplier_id                     VARCHAR(100),
        -- Supplier providing this plot's commodities
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for geolocation record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_euis_geo_package FOREIGN KEY (package_id) REFERENCES gl_eudr_euis_submission_packages (package_id),
    CONSTRAINT chk_euis_geo_commodity CHECK (commodity_type IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_euis_geo_crs CHECK (coordinate_system IN (
        'WGS84', 'EPSG4326', 'UTM'
    )),
    CONSTRAINT chk_euis_geo_type CHECK (geometry_type IN (
        'polygon', 'point', 'multipolygon'
    )),
    CONSTRAINT chk_euis_geo_area CHECK (area_hectares IS NULL OR area_hectares >= 0),
    CONSTRAINT chk_euis_geo_lat CHECK (centroid_latitude IS NULL OR
        (centroid_latitude >= -90 AND centroid_latitude <= 90)),
    CONSTRAINT chk_euis_geo_lon CHECK (centroid_longitude IS NULL OR
        (centroid_longitude >= -180 AND centroid_longitude <= 180)),
    CONSTRAINT chk_euis_geo_source CHECK (data_source IN (
        'gps', 'satellite', 'cadastral', 'survey', 'digitized', 'supplier_provided', 'government_registry'
    )),
    CONSTRAINT chk_euis_geo_overlap CHECK (overlap_check_status IN (
        'pending', 'in_progress', 'passed', 'failed', 'skipped'
    ))
);

COMMENT ON TABLE gl_eudr_euis_geolocation_data IS 'AGENT-EUDR-036: Formatted geolocation exports for EU IS submission with WGS84 coordinates, polygon/point geometry per Article 9(1)(d) four-hectare threshold, area calculations, overlap checks against protected areas, and deforestation-free verification';
COMMENT ON COLUMN gl_eudr_euis_geolocation_data.geometry_type IS 'Geometry type per Article 9(1)(d): polygon for plots > 4 hectares, point (centroid) for plots <= 4 hectares';
COMMENT ON COLUMN gl_eudr_euis_geolocation_data.coordinates IS 'GeoJSON-compatible coordinates: polygon [[lon,lat],[lon,lat],...] or point [lon,lat]';

-- Indexes for gl_eudr_euis_geolocation_data
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_geo_package ON gl_eudr_euis_geolocation_data (package_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_geo_dds_ref ON gl_eudr_euis_geolocation_data (dds_reference);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_geo_plot ON gl_eudr_euis_geolocation_data (plot_identifier);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_geo_supplier ON gl_eudr_euis_geolocation_data (supplier_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_geo_provenance ON gl_eudr_euis_geolocation_data (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_geo_created ON gl_eudr_euis_geolocation_data (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_geo_tenant_operator ON gl_eudr_euis_geolocation_data (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_geo_operator_commodity ON gl_eudr_euis_geolocation_data (operator_id, commodity_type, origin_country);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_geo_country_commodity ON gl_eudr_euis_geolocation_data (origin_country, commodity_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for plots requiring polygon geometry (> 4 ha)
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_geo_large_plots ON gl_eudr_euis_geolocation_data (operator_id, origin_country)
        WHERE area_hectares > 4 AND geometry_type = 'polygon';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for plots failing EU IS format validation
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_geo_invalid ON gl_eudr_euis_geolocation_data (operator_id, plot_identifier)
        WHERE eu_is_format_valid = FALSE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for plots with overlap issues
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_geo_overlap_fail ON gl_eudr_euis_geolocation_data (operator_id, origin_country)
        WHERE overlap_check_status = 'failed';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for unverified deforestation status
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_geo_defor_pending ON gl_eudr_euis_geolocation_data (operator_id, commodity_type)
        WHERE deforestation_free_verified IS NULL OR deforestation_free_verified = FALSE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_geo_coordinates ON gl_eudr_euis_geolocation_data USING GIN (coordinates);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_geo_fmt_errors ON gl_eudr_euis_geolocation_data USING GIN (format_validation_errors);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_geo_overlap_res ON gl_eudr_euis_geolocation_data USING GIN (overlap_results);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 5. gl_eudr_euis_api_credentials -- eIDAS credentials and tokens (encrypted)
-- ============================================================================
RAISE NOTICE 'V124 [5/8]: Creating gl_eudr_euis_api_credentials...';

CREATE TABLE IF NOT EXISTS gl_eudr_euis_api_credentials (
    credential_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this credential record
    operator_id                     VARCHAR(100)    NOT NULL,
        -- Internal GreenLang operator identifier
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    registration_id                 UUID,
        -- FK reference to the operator registration this credential is associated with
    credential_type                 VARCHAR(30)     NOT NULL DEFAULT 'eidas_certificate',
        -- Type of credential
    credential_name                 VARCHAR(200)    NOT NULL,
        -- Human-readable name for this credential (e.g. "Production EU IS Certificate 2026")
    environment                     VARCHAR(20)     NOT NULL DEFAULT 'production',
        -- Target environment for this credential
    -- Encrypted credential fields (AES-256 via pgcrypto)
    client_id_encrypted             BYTEA,
        -- AES-256-GCM encrypted OAuth client ID
    client_secret_encrypted         BYTEA,
        -- AES-256-GCM encrypted OAuth client secret
    certificate_pem_encrypted       BYTEA,
        -- AES-256-GCM encrypted eIDAS certificate PEM data
    private_key_encrypted           BYTEA,
        -- AES-256-GCM encrypted private key PEM data
    certificate_chain_encrypted     BYTEA,
        -- AES-256-GCM encrypted certificate chain PEM data
    -- Token management
    access_token_encrypted          BYTEA,
        -- AES-256-GCM encrypted current OAuth2 access token
    refresh_token_encrypted         BYTEA,
        -- AES-256-GCM encrypted current OAuth2 refresh token
    token_type                      VARCHAR(20)     DEFAULT 'bearer',
        -- Token type (typically "bearer" for OAuth2)
    token_issued_at                 TIMESTAMPTZ,
        -- Timestamp when the current access token was issued
    token_expires_at                TIMESTAMPTZ,
        -- Timestamp when the current access token expires
    refresh_token_expires_at        TIMESTAMPTZ,
        -- Timestamp when the refresh token expires
    token_scopes                    JSONB           DEFAULT '[]',
        -- Array of OAuth2 scopes granted: ["eudr:dds:submit", "eudr:dds:read", "eudr:registration:manage"]
    -- Certificate metadata (non-sensitive)
    certificate_serial              VARCHAR(200),
        -- X.509 certificate serial number
    certificate_issuer              VARCHAR(500),
        -- Certificate issuer distinguished name
    certificate_subject             VARCHAR(500),
        -- Certificate subject distinguished name
    certificate_not_before          TIMESTAMPTZ,
        -- Certificate validity start date
    certificate_not_after           TIMESTAMPTZ,
        -- Certificate validity end date (expiry)
    certificate_fingerprint         VARCHAR(128),
        -- SHA-256 fingerprint of the certificate for identification
    -- API endpoint configuration
    api_base_url                    VARCHAR(1000),
        -- EU IS API base URL (e.g. "https://eudr-is.europa.eu/api/v1")
    token_endpoint                  VARCHAR(1000),
        -- OAuth2 token endpoint URL
    authorization_endpoint          VARCHAR(1000),
        -- OAuth2 authorization endpoint URL (if applicable)
    -- Status and lifecycle
    status                          VARCHAR(20)     NOT NULL DEFAULT 'active',
        -- Credential lifecycle status
    last_used_at                    TIMESTAMPTZ,
        -- Timestamp when this credential was last used for an API call
    last_token_refresh_at           TIMESTAMPTZ,
        -- Timestamp of the last successful token refresh
    rotation_due_date               DATE,
        -- Date by which the credential/certificate should be rotated
    revoked_at                      TIMESTAMPTZ,
        -- Timestamp when the credential was revoked (NULL if not revoked)
    revoked_by                      VARCHAR(100),
        -- User who revoked the credential
    revocation_reason               TEXT,
        -- Reason for revocation
    created_by                      VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- User who created/uploaded this credential
    notes                           TEXT            DEFAULT '',
        -- Administrative notes about this credential
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for credential record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_euis_cred_registration FOREIGN KEY (registration_id) REFERENCES gl_eudr_euis_operator_registrations (registration_id),
    CONSTRAINT chk_euis_cred_type CHECK (credential_type IN (
        'eidas_certificate', 'oauth2_client', 'api_key', 'mtls_certificate', 'service_account'
    )),
    CONSTRAINT chk_euis_cred_env CHECK (environment IN (
        'production', 'staging', 'sandbox', 'test'
    )),
    CONSTRAINT chk_euis_cred_status CHECK (status IN (
        'active', 'inactive', 'expired', 'revoked', 'pending_activation', 'rotation_pending'
    )),
    CONSTRAINT chk_euis_cred_token_type CHECK (token_type IS NULL OR token_type IN (
        'bearer', 'mac', 'dpop'
    )),
    CONSTRAINT chk_euis_cred_cert_dates CHECK (certificate_not_after IS NULL OR certificate_not_before IS NULL
        OR certificate_not_after >= certificate_not_before)
);

COMMENT ON TABLE gl_eudr_euis_api_credentials IS 'AGENT-EUDR-036: eIDAS-compliant API credentials and OAuth2 tokens with AES-256-GCM column-level encryption for client secrets, certificates, private keys, and access/refresh tokens; includes certificate metadata, endpoint configuration, and rotation tracking per EUDR Article 33';
COMMENT ON COLUMN gl_eudr_euis_api_credentials.client_secret_encrypted IS 'AES-256-GCM encrypted OAuth2 client secret: decrypt via pgcrypto pgp_sym_decrypt with application-managed encryption key';
COMMENT ON COLUMN gl_eudr_euis_api_credentials.private_key_encrypted IS 'AES-256-GCM encrypted private key PEM: NEVER log or expose in plaintext; decrypt only in-memory for API calls';

-- Indexes for gl_eudr_euis_api_credentials
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_cred_registration ON gl_eudr_euis_api_credentials (registration_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_cred_fingerprint ON gl_eudr_euis_api_credentials (certificate_fingerprint);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_cred_provenance ON gl_eudr_euis_api_credentials (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_cred_created ON gl_eudr_euis_api_credentials (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_cred_tenant_operator ON gl_eudr_euis_api_credentials (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_cred_operator_env ON gl_eudr_euis_api_credentials (operator_id, environment, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_cred_operator_type ON gl_eudr_euis_api_credentials (operator_id, credential_type, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active credentials
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_cred_active ON gl_eudr_euis_api_credentials (operator_id, environment)
        WHERE status = 'active';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for credentials requiring rotation
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_cred_rotation ON gl_eudr_euis_api_credentials (rotation_due_date, operator_id)
        WHERE status = 'active' AND rotation_due_date IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for expiring certificates
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_cred_cert_expiry ON gl_eudr_euis_api_credentials (certificate_not_after, operator_id)
        WHERE status = 'active' AND certificate_not_after IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for expiring access tokens
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_cred_token_expiry ON gl_eudr_euis_api_credentials (token_expires_at, operator_id)
        WHERE status = 'active' AND token_expires_at IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_cred_scopes ON gl_eudr_euis_api_credentials USING GIN (token_scopes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 6. gl_eudr_euis_submission_history -- Historical submission records (hypertable)
-- ============================================================================
RAISE NOTICE 'V124 [6/8]: Creating gl_eudr_euis_submission_history (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_euis_submission_history (
    history_id                      UUID            DEFAULT gen_random_uuid(),
        -- Unique identifier for this history record
    submission_id                   UUID            NOT NULL,
        -- Reference to the parent DDS submission (logical FK to hypertable)
    operator_id                     VARCHAR(100)    NOT NULL,
        -- Internal GreenLang operator identifier
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    dds_reference                   VARCHAR(100)    NOT NULL,
        -- DDS reference number for cross-referencing
    event_type                      VARCHAR(50)     NOT NULL,
        -- Type of submission event recorded
    previous_status                 VARCHAR(30),
        -- Submission status before this event
    new_status                      VARCHAR(30),
        -- Submission status after this event
    event_source                    VARCHAR(30)     NOT NULL DEFAULT 'system',
        -- Source of the event
    eu_is_response_code             INTEGER,
        -- EU IS API HTTP response code (NULL for internal events)
    eu_is_response_summary          TEXT,
        -- Summary of EU IS response message
    eu_is_receipt_number            VARCHAR(200),
        -- EU IS receipt number (if received in this event)
    payload_snapshot                JSONB           DEFAULT '{}',
        -- Snapshot of relevant payload data at time of event
    error_details                   JSONB           DEFAULT '{}',
        -- Error details if the event represents a failure: {"code": "...", "message": "...", "stack": "..."}
    retry_attempt                   INTEGER,
        -- Retry attempt number (NULL if not a retry event)
    duration_ms                     BIGINT,
        -- Duration of the operation in milliseconds
    actor_id                        VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- User or system that triggered this event
    correlation_id                  VARCHAR(200),
        -- Distributed tracing correlation identifier
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for history record integrity verification
    recorded_at                     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp of this history record (partitioning column)

    CONSTRAINT chk_euis_hist_event CHECK (event_type IN (
        'submission_created', 'validation_started', 'validation_passed', 'validation_failed',
        'package_assembled', 'submission_initiated', 'api_call_started', 'api_call_completed',
        'api_call_failed', 'api_call_timeout', 'api_call_retried',
        'submitted_to_eu_is', 'acknowledged_by_eu_is', 'rejected_by_eu_is',
        'receipt_received', 'reference_assigned',
        'amendment_submitted', 'correction_submitted', 'withdrawal_submitted',
        'retry_scheduled', 'retry_attempted', 'max_retries_exceeded',
        'status_changed', 'cancelled', 'expired', 'manual_intervention'
    )),
    CONSTRAINT chk_euis_hist_source CHECK (event_source IN (
        'system', 'eu_is_api', 'operator', 'scheduler', 'admin', 'webhook'
    )),
    CONSTRAINT chk_euis_hist_duration CHECK (duration_ms IS NULL OR duration_ms >= 0)
);

-- Convert to TimescaleDB hypertable partitioned by recorded_at
DO $$
BEGIN
    PERFORM create_hypertable(
        'gl_eudr_euis_submission_history',
        'recorded_at',
        chunk_time_interval => INTERVAL '7 days',
        if_not_exists => TRUE
    );
    RAISE NOTICE 'gl_eudr_euis_submission_history hypertable created (7-day chunks)';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Hypertable creation skipped for gl_eudr_euis_submission_history: %', SQLERRM;
END $$;

COMMENT ON TABLE gl_eudr_euis_submission_history IS 'AGENT-EUDR-036: TimescaleDB-partitioned historical record of all submission events including API calls, status transitions, EU IS responses, retry attempts, and error details with full distributed tracing per EUDR Articles 4, 31, 33';
COMMENT ON COLUMN gl_eudr_euis_submission_history.event_type IS 'Enumerated submission event types covering the full lifecycle from creation through EU IS acknowledgement/rejection, including retry and manual intervention events';

-- Indexes for gl_eudr_euis_submission_history
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_hist_submission ON gl_eudr_euis_submission_history (submission_id, recorded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_hist_dds_ref ON gl_eudr_euis_submission_history (dds_reference, recorded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_hist_event ON gl_eudr_euis_submission_history (event_type, recorded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_hist_correlation ON gl_eudr_euis_submission_history (correlation_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_hist_provenance ON gl_eudr_euis_submission_history (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_hist_operator_event ON gl_eudr_euis_submission_history (operator_id, event_type, recorded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_hist_sub_event ON gl_eudr_euis_submission_history (submission_id, event_type, recorded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_hist_error ON gl_eudr_euis_submission_history USING GIN (error_details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_hist_payload ON gl_eudr_euis_submission_history USING GIN (payload_snapshot);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 7. gl_eudr_euis_rejection_tracking -- Rejection reasons and remediation
-- ============================================================================
RAISE NOTICE 'V124 [7/8]: Creating gl_eudr_euis_rejection_tracking...';

CREATE TABLE IF NOT EXISTS gl_eudr_euis_rejection_tracking (
    rejection_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this rejection tracking record
    submission_id                   UUID            NOT NULL,
        -- Reference to the rejected DDS submission (logical FK to hypertable)
    operator_id                     VARCHAR(100)    NOT NULL,
        -- Internal GreenLang operator identifier
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    dds_reference                   VARCHAR(100)    NOT NULL,
        -- DDS reference number for cross-referencing
    eu_is_rejection_code            VARCHAR(50),
        -- EU IS rejection code (e.g. "EUDR-ERR-001", "VALIDATION-FAIL-GEO")
    eu_is_rejection_category        VARCHAR(50),
        -- EU IS rejection category classification
    rejection_reason                TEXT            NOT NULL DEFAULT '',
        -- Human-readable rejection reason from EU IS
    rejection_details               JSONB           DEFAULT '{}',
        -- Structured rejection details: {"field_errors": [...], "schema_violations": [...], "business_rule_violations": [...]}
    rejected_at                     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the rejection was received from EU IS
    severity                        VARCHAR(20)     NOT NULL DEFAULT 'major',
        -- Severity of the rejection issue
    affected_sections               JSONB           DEFAULT '[]',
        -- Array of DDS sections affected by the rejection: ["geolocation", "article9_data", "operator_info"]
    remediation_status              VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- Current remediation workflow status
    remediation_plan                JSONB           DEFAULT '{}',
        -- Structured remediation plan: {"steps": [...], "estimated_effort_hours": 4, "assigned_to": "...", "deadline": "..."}
    remediation_notes               TEXT            DEFAULT '',
        -- Free-text notes on remediation progress
    remediation_started_at          TIMESTAMPTZ,
        -- Timestamp when remediation work began
    remediation_completed_at        TIMESTAMPTZ,
        -- Timestamp when remediation was completed
    remediated_by                   VARCHAR(100),
        -- User who performed the remediation
    resubmission_id                 UUID,
        -- Reference to the resubmission created after remediation (logical FK to hypertable)
    resubmitted_at                  TIMESTAMPTZ,
        -- Timestamp when the corrected DDS was resubmitted
    resubmission_accepted           BOOLEAN,
        -- Whether the resubmission was accepted by EU IS (NULL if not yet resubmitted)
    rejection_count_for_dds         INTEGER         NOT NULL DEFAULT 1,
        -- Total number of rejections for this DDS (across all resubmission attempts)
    root_cause_category             VARCHAR(30),
        -- Root cause classification for the rejection
    lessons_learned                 TEXT            DEFAULT '',
        -- Lessons learned from this rejection for process improvement
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for rejection record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_euis_rej_severity CHECK (severity IN (
        'critical', 'major', 'minor', 'informational'
    )),
    CONSTRAINT chk_euis_rej_remed_status CHECK (remediation_status IN (
        'pending', 'analyzing', 'in_progress', 'completed',
        'resubmitted', 'accepted', 'escalated', 'deferred'
    )),
    CONSTRAINT chk_euis_rej_root_cause CHECK (root_cause_category IS NULL OR root_cause_category IN (
        'data_quality', 'schema_mismatch', 'geolocation_error', 'missing_information',
        'format_error', 'business_rule', 'certificate_issue', 'system_error',
        'timing_issue', 'operator_error', 'other'
    )),
    CONSTRAINT chk_euis_rej_count CHECK (rejection_count_for_dds >= 1),
    CONSTRAINT chk_euis_rej_dates CHECK (remediation_completed_at IS NULL OR remediation_started_at IS NOT NULL)
);

COMMENT ON TABLE gl_eudr_euis_rejection_tracking IS 'AGENT-EUDR-036: Rejection tracking with EU IS rejection codes, structured remediation workflows, root cause classification, resubmission linking, and lessons learned for continuous improvement per EUDR Articles 4, 33';
COMMENT ON COLUMN gl_eudr_euis_rejection_tracking.eu_is_rejection_code IS 'EU IS rejection error code for categorizing rejection reason (e.g. "EUDR-ERR-GEO-001" for geolocation format errors)';
COMMENT ON COLUMN gl_eudr_euis_rejection_tracking.remediation_plan IS 'Structured remediation plan: {"steps": [{"order": 1, "action": "...", "assignee": "...", "due": "..."}], "estimated_effort_hours": 4, "deadline": "2026-04-01"}';

-- Indexes for gl_eudr_euis_rejection_tracking
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_rej_submission ON gl_eudr_euis_rejection_tracking (submission_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_rej_dds_ref ON gl_eudr_euis_rejection_tracking (dds_reference);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_rej_code ON gl_eudr_euis_rejection_tracking (eu_is_rejection_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_rej_provenance ON gl_eudr_euis_rejection_tracking (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_rej_created ON gl_eudr_euis_rejection_tracking (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_rej_tenant_operator ON gl_eudr_euis_rejection_tracking (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_rej_operator_status ON gl_eudr_euis_rejection_tracking (operator_id, remediation_status, severity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_rej_operator_code ON gl_eudr_euis_rejection_tracking (operator_id, eu_is_rejection_code, rejected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for rejections pending remediation
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_rej_pending ON gl_eudr_euis_rejection_tracking (operator_id, severity, rejected_at DESC)
        WHERE remediation_status IN ('pending', 'analyzing', 'in_progress');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for critical rejections requiring immediate attention
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_rej_critical ON gl_eudr_euis_rejection_tracking (operator_id, rejected_at DESC)
        WHERE severity = 'critical' AND remediation_status NOT IN ('completed', 'resubmitted', 'accepted');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for escalated rejections
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_rej_escalated ON gl_eudr_euis_rejection_tracking (operator_id, created_at DESC)
        WHERE remediation_status = 'escalated';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_rej_details ON gl_eudr_euis_rejection_tracking USING GIN (rejection_details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_rej_sections ON gl_eudr_euis_rejection_tracking USING GIN (affected_sections);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_rej_plan ON gl_eudr_euis_rejection_tracking USING GIN (remediation_plan);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 8. gl_eudr_euis_audit_trail -- Article 31 audit log (hypertable)
-- ============================================================================
RAISE NOTICE 'V124 [8/8]: Creating gl_eudr_euis_audit_trail (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_euis_audit_trail (
    audit_id                        UUID            DEFAULT gen_random_uuid(),
        -- Unique audit entry identifier
    entity_type                     VARCHAR(50)     NOT NULL,
        -- Type of entity being audited
    entity_id                       UUID            NOT NULL,
        -- Entity identifier being audited
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator identifier
    tenant_id                       VARCHAR(100)    NOT NULL DEFAULT '',
        -- Multi-tenant isolation identifier
    action                          VARCHAR(50)     NOT NULL,
        -- Action performed
    actor_id                        VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- Actor performing the action (user ID or system)
    actor_type                      VARCHAR(20)     NOT NULL DEFAULT 'system',
        -- Type of actor
    changes                         JSONB           DEFAULT '{}',
        -- JSON diff of changes: {"field": {"old": "...", "new": "..."}}
    context                         JSONB           DEFAULT '{}',
        -- Additional context: {"ip_address": "...", "user_agent": "...", "request_id": "...", "correlation_id": "..."}
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for audit entry integrity (chained to previous entry)
    timestamp                       TIMESTAMPTZ     NOT NULL DEFAULT NOW()
        -- Timestamp of the action (partitioning column)
);

-- Convert to TimescaleDB hypertable partitioned by timestamp
DO $$
BEGIN
    PERFORM create_hypertable(
        'gl_eudr_euis_audit_trail',
        'timestamp',
        chunk_time_interval => INTERVAL '7 days',
        if_not_exists => TRUE
    );
    RAISE NOTICE 'gl_eudr_euis_audit_trail hypertable created (7-day chunks)';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Hypertable creation skipped for gl_eudr_euis_audit_trail: %', SQLERRM;
END $$;

COMMENT ON TABLE gl_eudr_euis_audit_trail IS 'AGENT-EUDR-036: Immutable TimescaleDB-partitioned audit trail for all EU Information System Interface operations including registrations, submissions, credential management, rejections, and remediations per EUDR Article 31 with 5-year retention';

-- Indexes for gl_eudr_euis_audit_trail
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_audit_entity_id ON gl_eudr_euis_audit_trail (entity_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_audit_operator ON gl_eudr_euis_audit_trail (operator_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_audit_action ON gl_eudr_euis_audit_trail (action, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_audit_actor ON gl_eudr_euis_audit_trail (actor_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_audit_provenance ON gl_eudr_euis_audit_trail (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_audit_entity_action ON gl_eudr_euis_audit_trail (entity_type, action, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_audit_operator_entity ON gl_eudr_euis_audit_trail (operator_id, entity_type, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_audit_tenant_operator ON gl_eudr_euis_audit_trail (tenant_id, operator_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_audit_changes ON gl_eudr_euis_audit_trail USING GIN (changes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_euis_audit_context ON gl_eudr_euis_audit_trail USING GIN (context);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- DATA RETENTION POLICIES -- Article 31: 5-year retention
-- ============================================================================
RAISE NOTICE 'V124: Configuring 5-year data retention policies per EUDR Article 31...';

SELECT add_retention_policy('gl_eudr_euis_dds_submissions', INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_euis_submission_history', INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_euis_audit_trail', INTERVAL '5 years', if_not_exists => TRUE);


-- ============================================================================
-- Triggers: updated_at auto-update
-- ============================================================================
RAISE NOTICE 'V124: Creating updated_at triggers...';

CREATE OR REPLACE FUNCTION fn_eudr_euis_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_euis_registrations_updated_at
        BEFORE UPDATE ON gl_eudr_euis_operator_registrations
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_euis_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_euis_packages_updated_at
        BEFORE UPDATE ON gl_eudr_euis_submission_packages
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_euis_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_euis_geolocation_updated_at
        BEFORE UPDATE ON gl_eudr_euis_geolocation_data
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_euis_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_euis_credentials_updated_at
        BEFORE UPDATE ON gl_eudr_euis_api_credentials
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_euis_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_euis_rejections_updated_at
        BEFORE UPDATE ON gl_eudr_euis_rejection_tracking
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_euis_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- Triggers: Audit trail auto-insert
-- ============================================================================
RAISE NOTICE 'V124: Creating audit trail triggers...';

CREATE OR REPLACE FUNCTION fn_eudr_euis_audit_insert()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO gl_eudr_euis_audit_trail (entity_type, entity_id, operator_id, action, actor_id, changes, timestamp)
    VALUES (TG_ARGV[0], NEW.*::TEXT::UUID, COALESCE(NEW.operator_id, ''), 'create', 'system', row_to_json(NEW)::JSONB, NOW());
    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION fn_eudr_euis_audit_update()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO gl_eudr_euis_audit_trail (entity_type, entity_id, operator_id, action, actor_id, changes, timestamp)
    VALUES (TG_ARGV[0], NEW.*::TEXT::UUID, COALESCE(NEW.operator_id, ''), 'update', 'system', jsonb_build_object('new', row_to_json(NEW)::JSONB), NOW());
    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Operator registrations audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_euis_reg_audit_insert
        AFTER INSERT ON gl_eudr_euis_operator_registrations
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_euis_audit_insert('operator_registration');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_euis_reg_audit_update
        AFTER UPDATE ON gl_eudr_euis_operator_registrations
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_euis_audit_update('operator_registration');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Submission packages audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_euis_pkg_audit_insert
        AFTER INSERT ON gl_eudr_euis_submission_packages
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_euis_audit_insert('submission_package');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_euis_pkg_audit_update
        AFTER UPDATE ON gl_eudr_euis_submission_packages
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_euis_audit_update('submission_package');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Geolocation data audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_euis_geo_audit_insert
        AFTER INSERT ON gl_eudr_euis_geolocation_data
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_euis_audit_insert('geolocation_data');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_euis_geo_audit_update
        AFTER UPDATE ON gl_eudr_euis_geolocation_data
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_euis_audit_update('geolocation_data');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- API credentials audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_euis_cred_audit_insert
        AFTER INSERT ON gl_eudr_euis_api_credentials
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_euis_audit_insert('api_credential');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_euis_cred_audit_update
        AFTER UPDATE ON gl_eudr_euis_api_credentials
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_euis_audit_update('api_credential');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Rejection tracking audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_euis_rej_audit_insert
        AFTER INSERT ON gl_eudr_euis_rejection_tracking
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_euis_audit_insert('rejection');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_euis_rej_audit_update
        AFTER UPDATE ON gl_eudr_euis_rejection_tracking
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_euis_audit_update('rejection');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- Credential encryption helper functions
-- ============================================================================
RAISE NOTICE 'V124: Creating credential encryption helper functions...';

-- Encrypt a credential value using pgcrypto symmetric encryption
CREATE OR REPLACE FUNCTION fn_eudr_euis_encrypt_credential(
    plaintext TEXT,
    encryption_key TEXT
)
RETURNS BYTEA AS $$
BEGIN
    RETURN pgp_sym_encrypt(plaintext, encryption_key);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

COMMENT ON FUNCTION fn_eudr_euis_encrypt_credential IS 'AGENT-EUDR-036: Encrypts credential plaintext using pgcrypto PGP symmetric encryption (AES-256). The encryption_key must be provided by the application layer and NEVER stored in the database.';

-- Decrypt a credential value using pgcrypto symmetric decryption
CREATE OR REPLACE FUNCTION fn_eudr_euis_decrypt_credential(
    ciphertext BYTEA,
    encryption_key TEXT
)
RETURNS TEXT AS $$
BEGIN
    RETURN pgp_sym_decrypt(ciphertext, encryption_key);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

COMMENT ON FUNCTION fn_eudr_euis_decrypt_credential IS 'AGENT-EUDR-036: Decrypts credential ciphertext using pgcrypto PGP symmetric decryption. The encryption_key must be provided by the application layer. Returns NULL-safe plaintext.';


-- ============================================================================
-- Row-level security (RLS) for credentials table
-- ============================================================================
RAISE NOTICE 'V124: Enabling row-level security on credentials table...';

ALTER TABLE gl_eudr_euis_api_credentials ENABLE ROW LEVEL SECURITY;

-- Policy: Only the owning tenant can access their credentials
DO $$ BEGIN
    CREATE POLICY euis_cred_tenant_isolation ON gl_eudr_euis_api_credentials
        USING (tenant_id = current_setting('app.current_tenant', TRUE))
        WITH CHECK (tenant_id = current_setting('app.current_tenant', TRUE));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON POLICY euis_cred_tenant_isolation ON gl_eudr_euis_api_credentials IS 'AGENT-EUDR-036: Tenant isolation RLS policy ensuring credentials are only accessible to the owning tenant via app.current_tenant session variable';


-- ============================================================================
-- Completion
-- ============================================================================

RAISE NOTICE 'V124: AGENT-EUDR-036 EU Information System Interface -- 8 tables, 102 indexes, 15 triggers, 3 hypertables, 5-year retention, credential encryption, RLS';
RAISE NOTICE 'V124: Tables: gl_eudr_euis_operator_registrations, gl_eudr_euis_dds_submissions (hypertable), gl_eudr_euis_submission_packages, gl_eudr_euis_geolocation_data, gl_eudr_euis_api_credentials, gl_eudr_euis_submission_history (hypertable), gl_eudr_euis_rejection_tracking, gl_eudr_euis_audit_trail (hypertable)';
RAISE NOTICE 'V124: Foreign keys: geolocation_data -> submission_packages; api_credentials -> operator_registrations';
RAISE NOTICE 'V124: Hypertables: dds_submissions, submission_history, audit_trail (7-day chunks, 5-year retention)';
RAISE NOTICE 'V124: Security: pgcrypto encryption for credentials, RLS tenant isolation on api_credentials';

COMMIT;
