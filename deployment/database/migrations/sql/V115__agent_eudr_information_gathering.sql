-- ============================================================================
-- V115: AGENT-EUDR-027 Information Gathering Agent
-- ============================================================================
-- Creates tables for the Information Gathering Agent which orchestrates
-- collection, verification, normalization, and packaging of all information
-- required for EUDR due diligence statements per Article 9.  Covers external
-- database queries (EU TRACES, FLEGT, CITES, EITI), certification verification
-- (FSC, PEFC, RSPO, Rainforest Alliance, UTZ, Bonsucro, RTRS, ISCC, ISO 14001),
-- supplier profile aggregation, public data harvesting, data normalization,
-- completeness validation, information package assembly, and Article 31
-- audit trail.
--
-- Tables: 9 (all regular, no hypertables)
-- Indexes: ~95
--
-- Dependencies: None (standard PostgreSQL)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V115: Creating AGENT-EUDR-027 Information Gathering Agent tables...';


-- ============================================================================
-- 1. eudr_iga_gathering_operations — Top-level gathering operations
-- ============================================================================
RAISE NOTICE 'V115 [1/9]: Creating eudr_iga_gathering_operations...';

CREATE TABLE IF NOT EXISTS eudr_iga_gathering_operations (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    operation_id                    VARCHAR(100)    UNIQUE NOT NULL,
        -- Unique identifier for this gathering operation (e.g. "iga-op-2026-03-001")
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator initiating the information gathering
    commodity                       VARCHAR(50)     NOT NULL,
        -- EUDR commodity (cattle, cocoa, coffee, oil_palm, rubber, soya, wood)
    workflow_id                     VARCHAR(100),
        -- Reference to the due diligence workflow orchestrating this operation
    status                          VARCHAR(50)     NOT NULL DEFAULT 'initiated',
        -- Operation lifecycle status
    sources_queried                 JSONB           DEFAULT '[]',
        -- Array of external source identifiers that were queried
    sources_completed               JSONB           DEFAULT '[]',
        -- Array of source identifiers that returned results successfully
    sources_failed                  JSONB           DEFAULT '[]',
        -- Array of source identifiers that failed or timed out
    completeness_score              DECIMAL(10,4)   DEFAULT 0,
        -- Overall completeness score (0.0000 to 1.0000) of gathered information
    completeness_classification     VARCHAR(50),
        -- Classification: 'comprehensive', 'adequate', 'partial', 'insufficient'
    total_records_collected          INTEGER         DEFAULT 0,
        -- Total number of records collected across all sources
    total_suppliers_resolved         INTEGER         DEFAULT 0,
        -- Total number of supplier profiles successfully resolved
    total_certificates_verified      INTEGER         DEFAULT 0,
        -- Total number of certificates verified against certification bodies
    package_id                      VARCHAR(100),
        -- Reference to the assembled information package
    started_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the gathering operation began
    completed_at                    TIMESTAMPTZ,
        -- Timestamp when the gathering operation completed (NULL if in-progress)
    duration_ms                     INTEGER,
        -- Total duration in milliseconds
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for operation integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_iga_op_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_iga_op_status CHECK (status IN (
        'initiated', 'querying', 'verifying', 'aggregating',
        'normalizing', 'validating', 'packaging', 'completed', 'failed'
    )),
    CONSTRAINT chk_iga_op_classification CHECK (completeness_classification IS NULL OR completeness_classification IN (
        'comprehensive', 'adequate', 'partial', 'insufficient'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_op_operation ON eudr_iga_gathering_operations (operation_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_op_operator ON eudr_iga_gathering_operations (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_op_commodity ON eudr_iga_gathering_operations (commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_op_workflow ON eudr_iga_gathering_operations (workflow_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_op_status ON eudr_iga_gathering_operations (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_op_completeness ON eudr_iga_gathering_operations (completeness_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_op_classification ON eudr_iga_gathering_operations (completeness_classification);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_op_package ON eudr_iga_gathering_operations (package_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_op_started ON eudr_iga_gathering_operations (started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_op_completed ON eudr_iga_gathering_operations (completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_op_provenance ON eudr_iga_gathering_operations (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_op_created ON eudr_iga_gathering_operations (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_op_updated ON eudr_iga_gathering_operations (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_op_commodity_status ON eudr_iga_gathering_operations (commodity, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_op_operator_commodity ON eudr_iga_gathering_operations (operator_id, commodity, started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active (non-terminal) operations
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_op_active ON eudr_iga_gathering_operations (started_at DESC)
        WHERE status NOT IN ('completed', 'failed');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_op_sources_queried ON eudr_iga_gathering_operations USING GIN (sources_queried);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_op_sources_completed ON eudr_iga_gathering_operations USING GIN (sources_completed);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_op_sources_failed ON eudr_iga_gathering_operations USING GIN (sources_failed);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_iga_gathering_operations IS 'AGENT-EUDR-027: Top-level information gathering operations tracking source queries, completeness, and packaging for EUDR Article 9 due diligence';
COMMENT ON COLUMN eudr_iga_gathering_operations.completeness_score IS 'Weighted completeness: 0.0000 (no data) to 1.0000 (all Article 9 elements present)';
COMMENT ON COLUMN eudr_iga_gathering_operations.completeness_classification IS 'comprehensive (>=0.90), adequate (>=0.70), partial (>=0.40), insufficient (<0.40)';


-- ============================================================================
-- 2. eudr_iga_query_results — External database query results
-- ============================================================================
RAISE NOTICE 'V115 [2/9]: Creating eudr_iga_query_results...';

CREATE TABLE IF NOT EXISTS eudr_iga_query_results (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id                        VARCHAR(100)    UNIQUE NOT NULL,
        -- Unique identifier for this query (e.g. "iga-q-traces-2026-03-001")
    operation_id                    VARCHAR(100)    NOT NULL,
        -- Reference to the parent gathering operation
    source                          VARCHAR(50)     NOT NULL,
        -- External data source queried
    query_parameters                JSONB           DEFAULT '{}',
        -- Parameters sent to the external source
    status                          VARCHAR(50)     NOT NULL DEFAULT 'success',
        -- Query result status
    records                         JSONB           DEFAULT '[]',
        -- Array of records returned by the source
    record_count                    INTEGER         DEFAULT 0,
        -- Number of records returned
    query_timestamp                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the query was executed
    response_time_ms                INTEGER         DEFAULT 0,
        -- Response time from the external source in milliseconds
    cached                          BOOLEAN         DEFAULT FALSE,
        -- Whether the result was served from cache
    cache_age_seconds               INTEGER,
        -- Age of cached result in seconds (NULL if not cached)
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for query result integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_iga_qr_operation FOREIGN KEY (operation_id)
        REFERENCES eudr_iga_gathering_operations (operation_id),
    CONSTRAINT chk_iga_qr_source CHECK (source IN (
        'eu_traces', 'flegt', 'cites', 'eiti', 'fao',
        'global_forest_watch', 'world_bank', 'transparency_international',
        'eu_timber_regulation', 'national_registry', 'custom'
    )),
    CONSTRAINT chk_iga_qr_status CHECK (status IN (
        'success', 'partial', 'empty', 'error', 'timeout', 'rate_limited'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_qr_query ON eudr_iga_query_results (query_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_qr_operation ON eudr_iga_query_results (operation_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_qr_source ON eudr_iga_query_results (source);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_qr_status ON eudr_iga_query_results (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_qr_timestamp ON eudr_iga_query_results (query_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_qr_response_time ON eudr_iga_query_results (response_time_ms DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_qr_cached ON eudr_iga_query_results (cached);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_qr_provenance ON eudr_iga_query_results (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_qr_created ON eudr_iga_query_results (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_qr_op_source ON eudr_iga_query_results (operation_id, source);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_qr_source_status ON eudr_iga_query_results (source, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for failed queries requiring attention
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_qr_failures ON eudr_iga_query_results (created_at DESC, source)
        WHERE status IN ('error', 'timeout', 'rate_limited');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_qr_params ON eudr_iga_query_results USING GIN (query_parameters);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_qr_records ON eudr_iga_query_results USING GIN (records);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_iga_query_results IS 'AGENT-EUDR-027: External database query results from EU TRACES, FLEGT, CITES, EITI, and other authoritative sources';
COMMENT ON COLUMN eudr_iga_query_results.source IS 'External data source: eu_traces, flegt, cites, eiti, fao, global_forest_watch, world_bank, transparency_international, eu_timber_regulation, national_registry, custom';
COMMENT ON COLUMN eudr_iga_query_results.cached IS 'TRUE if result was served from internal cache rather than live query';


-- ============================================================================
-- 3. eudr_iga_certificate_verifications — Certificate verification results
-- ============================================================================
RAISE NOTICE 'V115 [3/9]: Creating eudr_iga_certificate_verifications...';

CREATE TABLE IF NOT EXISTS eudr_iga_certificate_verifications (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    certificate_id                  VARCHAR(200)    NOT NULL,
        -- Certificate number or identifier from the certification body
    certification_body              VARCHAR(50)     NOT NULL,
        -- Certification body that issued the certificate
    operation_id                    VARCHAR(100),
        -- Reference to the parent gathering operation (nullable for standalone checks)
    holder_name                     VARCHAR(500),
        -- Name of the certificate holder (company or individual)
    verification_status             VARCHAR(50)     DEFAULT 'not_found',
        -- Verification result status
    valid_from                      TIMESTAMPTZ,
        -- Certificate validity start date
    valid_until                     TIMESTAMPTZ,
        -- Certificate expiration date
    scope                           JSONB           DEFAULT '[]',
        -- Array of scope items covered by the certificate
    commodity_scope                 JSONB           DEFAULT '[]',
        -- Array of EUDR commodities covered by the certificate
    chain_of_custody_model          VARCHAR(100),
        -- Chain of custody model (e.g. "FSC Mix", "PEFC Controlled Sources", "RSPO Mass Balance")
    days_until_expiry               INTEGER,
        -- Calculated days until certificate expiry (negative = expired)
    last_verified                   TIMESTAMPTZ     DEFAULT NOW(),
        -- Timestamp of the most recent verification check
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for verification integrity
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_iga_cv_operation FOREIGN KEY (operation_id)
        REFERENCES eudr_iga_gathering_operations (operation_id),
    CONSTRAINT chk_iga_cv_body CHECK (certification_body IN (
        'fsc', 'pefc', 'rspo', 'rainforest_alliance', 'utz',
        'bonsucro', 'rtrs', 'iscc', 'iso_14001', 'eu_organic', 'custom'
    )),
    CONSTRAINT chk_iga_cv_status CHECK (verification_status IN (
        'valid', 'expired', 'suspended', 'revoked', 'not_found',
        'pending_verification', 'verification_error'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cv_certificate ON eudr_iga_certificate_verifications (certificate_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cv_body ON eudr_iga_certificate_verifications (certification_body);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cv_operation ON eudr_iga_certificate_verifications (operation_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cv_holder ON eudr_iga_certificate_verifications (holder_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cv_status ON eudr_iga_certificate_verifications (verification_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cv_valid_from ON eudr_iga_certificate_verifications (valid_from);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cv_valid_until ON eudr_iga_certificate_verifications (valid_until);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cv_coc_model ON eudr_iga_certificate_verifications (chain_of_custody_model);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cv_expiry ON eudr_iga_certificate_verifications (days_until_expiry);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cv_last_verified ON eudr_iga_certificate_verifications (last_verified DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cv_provenance ON eudr_iga_certificate_verifications (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cv_created ON eudr_iga_certificate_verifications (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cv_body_status ON eudr_iga_certificate_verifications (certification_body, verification_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cv_cert_body ON eudr_iga_certificate_verifications (certificate_id, certification_body);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for expiring certificates (within 90 days)
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cv_expiring ON eudr_iga_certificate_verifications (days_until_expiry, valid_until)
        WHERE verification_status = 'valid' AND days_until_expiry IS NOT NULL AND days_until_expiry <= 90;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for invalid certificates requiring attention
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cv_invalid ON eudr_iga_certificate_verifications (created_at DESC, certification_body)
        WHERE verification_status IN ('expired', 'suspended', 'revoked');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cv_scope ON eudr_iga_certificate_verifications USING GIN (scope);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cv_commodity_scope ON eudr_iga_certificate_verifications USING GIN (commodity_scope);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_iga_certificate_verifications IS 'AGENT-EUDR-027: Certificate verification results from FSC, PEFC, RSPO, Rainforest Alliance, UTZ, Bonsucro, RTRS, ISCC, ISO 14001';
COMMENT ON COLUMN eudr_iga_certificate_verifications.days_until_expiry IS 'Calculated days until expiry: positive = days remaining, zero = expires today, negative = expired N days ago';
COMMENT ON COLUMN eudr_iga_certificate_verifications.chain_of_custody_model IS 'Chain of custody model such as FSC Mix, PEFC Controlled Sources, RSPO Mass Balance, RSPO Segregation';


-- ============================================================================
-- 4. eudr_iga_supplier_profiles — Aggregated supplier profiles
-- ============================================================================
RAISE NOTICE 'V115 [4/9]: Creating eudr_iga_supplier_profiles...';

CREATE TABLE IF NOT EXISTS eudr_iga_supplier_profiles (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id                     VARCHAR(100)    UNIQUE NOT NULL,
        -- Unique supplier identifier
    name                            VARCHAR(500)    NOT NULL,
        -- Primary supplier name
    alternative_names               JSONB           DEFAULT '[]',
        -- Array of alternative/trade names for entity resolution
    postal_address                  TEXT,
        -- Full postal address of the supplier
    country_code                    VARCHAR(5),
        -- ISO 3166-1 alpha-2 or alpha-3 country code
    email                           VARCHAR(320),
        -- Primary contact email address
    registration_number             VARCHAR(200),
        -- Business registration or tax identification number
    commodities                     JSONB           DEFAULT '[]',
        -- Array of EUDR commodities supplied
    certifications                  JSONB           DEFAULT '[]',
        -- Array of active certifications with verification status
    plot_ids                        JSONB           DEFAULT '[]',
        -- Array of associated production plot identifiers
    tier_depth                      INTEGER         DEFAULT 0,
        -- Supply chain tier depth (0 = direct supplier, 1 = tier-1, etc.)
    data_sources                    JSONB           DEFAULT '[]',
        -- Array of data source identifiers used to build this profile
    completeness_score              DECIMAL(10,4)   DEFAULT 0,
        -- Profile completeness (0.0000 to 1.0000) based on required fields
    confidence_score                DECIMAL(10,4)   DEFAULT 0,
        -- Data confidence score (0.0000 to 1.0000) based on source reliability
    discrepancies                   JSONB           DEFAULT '[]',
        -- Array of detected data discrepancies across sources
    last_updated                    TIMESTAMPTZ     DEFAULT NOW(),
        -- Timestamp of most recent profile update
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for profile integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_sp_supplier ON eudr_iga_supplier_profiles (supplier_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_sp_name ON eudr_iga_supplier_profiles (name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_sp_country ON eudr_iga_supplier_profiles (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_sp_registration ON eudr_iga_supplier_profiles (registration_number);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_sp_tier ON eudr_iga_supplier_profiles (tier_depth);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_sp_completeness ON eudr_iga_supplier_profiles (completeness_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_sp_confidence ON eudr_iga_supplier_profiles (confidence_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_sp_last_updated ON eudr_iga_supplier_profiles (last_updated DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_sp_provenance ON eudr_iga_supplier_profiles (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_sp_created ON eudr_iga_supplier_profiles (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_sp_updated ON eudr_iga_supplier_profiles (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_sp_country_tier ON eudr_iga_supplier_profiles (country_code, tier_depth);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for incomplete profiles needing enrichment
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_sp_incomplete ON eudr_iga_supplier_profiles (completeness_score, last_updated DESC)
        WHERE completeness_score < 0.7000;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_sp_alt_names ON eudr_iga_supplier_profiles USING GIN (alternative_names);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_sp_commodities ON eudr_iga_supplier_profiles USING GIN (commodities);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_sp_certifications ON eudr_iga_supplier_profiles USING GIN (certifications);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_sp_plot_ids ON eudr_iga_supplier_profiles USING GIN (plot_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_sp_data_sources ON eudr_iga_supplier_profiles USING GIN (data_sources);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_sp_discrepancies ON eudr_iga_supplier_profiles USING GIN (discrepancies);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_iga_supplier_profiles IS 'AGENT-EUDR-027: Aggregated supplier profiles compiled from multiple sources with completeness scoring, confidence assessment, and discrepancy tracking';
COMMENT ON COLUMN eudr_iga_supplier_profiles.completeness_score IS 'Profile completeness: 0.0000 (empty) to 1.0000 (all required EUDR Article 9 fields populated)';
COMMENT ON COLUMN eudr_iga_supplier_profiles.confidence_score IS 'Data confidence: 0.0000 (unverified) to 1.0000 (cross-referenced across 3+ authoritative sources)';


-- ============================================================================
-- 5. eudr_iga_completeness_reports — Completeness validation reports
-- ============================================================================
RAISE NOTICE 'V115 [5/9]: Creating eudr_iga_completeness_reports...';

CREATE TABLE IF NOT EXISTS eudr_iga_completeness_reports (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    operation_id                    VARCHAR(100)    NOT NULL,
        -- Reference to the parent gathering operation
    commodity                       VARCHAR(50)     NOT NULL,
        -- EUDR commodity being validated for completeness
    elements                        JSONB           NOT NULL,
        -- Detailed element-by-element completeness assessment per Article 9
    completeness_score              DECIMAL(10,4)   DEFAULT 0,
        -- Overall completeness score (0.0000 to 1.0000)
    completeness_classification     VARCHAR(50)     DEFAULT 'insufficient',
        -- Classification: 'comprehensive', 'adequate', 'partial', 'insufficient'
    gap_report                      JSONB           DEFAULT '{}',
        -- Detailed gap analysis identifying missing Article 9 elements
    is_simplified_dd                BOOLEAN         DEFAULT FALSE,
        -- TRUE if simplified due diligence applies per EUDR Article 13
    validated_at                    TIMESTAMPTZ     DEFAULT NOW(),
        -- Timestamp of the completeness validation
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for validation integrity
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_iga_cr_operation FOREIGN KEY (operation_id)
        REFERENCES eudr_iga_gathering_operations (operation_id),
    CONSTRAINT chk_iga_cr_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_iga_cr_classification CHECK (completeness_classification IN (
        'comprehensive', 'adequate', 'partial', 'insufficient'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cr_operation ON eudr_iga_completeness_reports (operation_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cr_commodity ON eudr_iga_completeness_reports (commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cr_score ON eudr_iga_completeness_reports (completeness_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cr_classification ON eudr_iga_completeness_reports (completeness_classification);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cr_simplified ON eudr_iga_completeness_reports (is_simplified_dd);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cr_validated ON eudr_iga_completeness_reports (validated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cr_provenance ON eudr_iga_completeness_reports (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cr_created ON eudr_iga_completeness_reports (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cr_commodity_class ON eudr_iga_completeness_reports (commodity, completeness_classification);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for insufficient completeness needing remediation
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cr_gaps ON eudr_iga_completeness_reports (created_at DESC, commodity)
        WHERE completeness_classification IN ('insufficient', 'partial');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cr_elements ON eudr_iga_completeness_reports USING GIN (elements);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_cr_gap_report ON eudr_iga_completeness_reports USING GIN (gap_report);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_iga_completeness_reports IS 'AGENT-EUDR-027: Completeness validation reports checking all EUDR Article 9 information elements are present with gap analysis';
COMMENT ON COLUMN eudr_iga_completeness_reports.elements IS 'Element-by-element JSON: {"operator_info": {"present": true, "score": 1.0}, "product_description": {...}, ...}';
COMMENT ON COLUMN eudr_iga_completeness_reports.is_simplified_dd IS 'TRUE if country is benchmarked low-risk per EUDR Article 29 and simplified due diligence per Article 13 applies';


-- ============================================================================
-- 6. eudr_iga_information_packages — Assembled information packages
-- ============================================================================
RAISE NOTICE 'V115 [6/9]: Creating eudr_iga_information_packages...';

CREATE TABLE IF NOT EXISTS eudr_iga_information_packages (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    package_id                      VARCHAR(100)    UNIQUE NOT NULL,
        -- Unique package identifier (e.g. "iga-pkg-2026-03-001-v1")
    operation_id                    VARCHAR(100)    NOT NULL,
        -- Reference to the parent gathering operation
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator who owns this package
    commodity                       VARCHAR(50)     NOT NULL,
        -- EUDR commodity for which the package was assembled
    version                         INTEGER         DEFAULT 1,
        -- Package version (incremented on repackaging)
    article_9_elements              JSONB           DEFAULT '{}',
        -- Structured Article 9 information elements with field-level data
    completeness_score              DECIMAL(10,4)   DEFAULT 0,
        -- Package completeness score (0.0000 to 1.0000)
    completeness_classification     VARCHAR(50),
        -- Classification: 'comprehensive', 'adequate', 'partial', 'insufficient'
    supplier_profiles               JSONB           DEFAULT '[]',
        -- Array of enriched supplier profile summaries included in the package
    external_data                   JSONB           DEFAULT '{}',
        -- Aggregated external data from EU TRACES, FLEGT, CITES, EITI queries
    certification_results           JSONB           DEFAULT '[]',
        -- Array of certification verification result summaries
    public_data                     JSONB           DEFAULT '{}',
        -- Public data harvest summaries (FAO, GFW, WB, TI)
    normalization_log               JSONB           DEFAULT '[]',
        -- Array of normalization actions applied to package data
    gap_report                      JSONB           DEFAULT '{}',
        -- Gap analysis identifying any missing Article 9 elements
    evidence_artifacts              JSONB           DEFAULT '[]',
        -- Array of evidence artifact references (document hashes, URLs)
    provenance_chain                JSONB           DEFAULT '[]',
        -- Complete provenance chain from source to package
    package_hash                    VARCHAR(64),
        -- SHA-256 hash of the complete package contents
    assembled_at                    TIMESTAMPTZ     DEFAULT NOW(),
        -- Timestamp when the package was assembled
    valid_until                     TIMESTAMPTZ,
        -- Package validity expiration (based on certificate/data freshness)
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_iga_ip_operation FOREIGN KEY (operation_id)
        REFERENCES eudr_iga_gathering_operations (operation_id),
    CONSTRAINT chk_iga_ip_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_iga_ip_classification CHECK (completeness_classification IS NULL OR completeness_classification IN (
        'comprehensive', 'adequate', 'partial', 'insufficient'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_package ON eudr_iga_information_packages (package_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_operation ON eudr_iga_information_packages (operation_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_operator ON eudr_iga_information_packages (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_commodity ON eudr_iga_information_packages (commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_version ON eudr_iga_information_packages (version);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_completeness ON eudr_iga_information_packages (completeness_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_classification ON eudr_iga_information_packages (completeness_classification);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_hash ON eudr_iga_information_packages (package_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_assembled ON eudr_iga_information_packages (assembled_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_valid_until ON eudr_iga_information_packages (valid_until);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_created ON eudr_iga_information_packages (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_updated ON eudr_iga_information_packages (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_operator_commodity ON eudr_iga_information_packages (operator_id, commodity, assembled_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for packages expiring soon (within 30 days)
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_expiring ON eudr_iga_information_packages (valid_until, operator_id)
        WHERE valid_until IS NOT NULL AND valid_until <= (NOW() + INTERVAL '30 days');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_article9 ON eudr_iga_information_packages USING GIN (article_9_elements);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_suppliers ON eudr_iga_information_packages USING GIN (supplier_profiles);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_external ON eudr_iga_information_packages USING GIN (external_data);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_certs ON eudr_iga_information_packages USING GIN (certification_results);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_provenance ON eudr_iga_information_packages USING GIN (provenance_chain);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_ip_evidence ON eudr_iga_information_packages USING GIN (evidence_artifacts);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_iga_information_packages IS 'AGENT-EUDR-027: Assembled information packages containing all Article 9 data elements for EUDR due diligence statements';
COMMENT ON COLUMN eudr_iga_information_packages.article_9_elements IS 'Structured Article 9 fields: operator_info, product_description, country_of_production, geolocation, quantity, supplier_info, compliance_verification';
COMMENT ON COLUMN eudr_iga_information_packages.package_hash IS 'SHA-256 hash of the complete package contents for tamper detection and provenance verification';


-- ============================================================================
-- 7. eudr_iga_normalization_records — Data normalization audit trail
-- ============================================================================
RAISE NOTICE 'V115 [7/9]: Creating eudr_iga_normalization_records...';

CREATE TABLE IF NOT EXISTS eudr_iga_normalization_records (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    operation_id                    VARCHAR(100)    NOT NULL,
        -- Reference to the parent gathering operation
    field_name                      VARCHAR(200)    NOT NULL,
        -- Name of the field that was normalized (e.g. "country_code", "unit_of_measure")
    source_value                    TEXT            NOT NULL,
        -- Original value before normalization
    normalized_value                TEXT            NOT NULL,
        -- Value after normalization
    normalization_type              VARCHAR(50)     NOT NULL,
        -- Type of normalization applied
    confidence                      DECIMAL(10,4)   DEFAULT 1.0,
        -- Confidence in the normalization (0.0000 to 1.0000)
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_iga_nr_operation FOREIGN KEY (operation_id)
        REFERENCES eudr_iga_gathering_operations (operation_id),
    CONSTRAINT chk_iga_nr_type CHECK (normalization_type IN (
        'country_code', 'unit_conversion', 'date_format', 'name_standardization',
        'address_normalization', 'commodity_mapping', 'currency_conversion',
        'encoding_fix', 'whitespace_trim', 'case_normalization', 'custom'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_nr_operation ON eudr_iga_normalization_records (operation_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_nr_field ON eudr_iga_normalization_records (field_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_nr_type ON eudr_iga_normalization_records (normalization_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_nr_confidence ON eudr_iga_normalization_records (confidence);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_nr_created ON eudr_iga_normalization_records (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_nr_op_type ON eudr_iga_normalization_records (operation_id, normalization_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_iga_normalization_records IS 'AGENT-EUDR-027: Data normalization audit trail recording all transformations applied during information gathering';
COMMENT ON COLUMN eudr_iga_normalization_records.normalization_type IS 'Normalization category: country_code, unit_conversion, date_format, name_standardization, address_normalization, commodity_mapping, etc.';


-- ============================================================================
-- 8. eudr_iga_harvest_results — Public data harvest results
-- ============================================================================
RAISE NOTICE 'V115 [8/9]: Creating eudr_iga_harvest_results...';

CREATE TABLE IF NOT EXISTS eudr_iga_harvest_results (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    source                          VARCHAR(50)     NOT NULL,
        -- Public data source harvested
    data_type                       VARCHAR(100)    NOT NULL,
        -- Type of data harvested (e.g. "deforestation_rates", "governance_index")
    country_code                    VARCHAR(5),
        -- ISO country code for geographically scoped data
    commodity                       VARCHAR(50),
        -- EUDR commodity for commodity-scoped data
    records_harvested               INTEGER         DEFAULT 0,
        -- Number of records harvested in this batch
    data_timestamp                  TIMESTAMPTZ,
        -- Timestamp of the source data (data vintage)
    is_incremental                  BOOLEAN         DEFAULT FALSE,
        -- TRUE if this is an incremental update (not full refresh)
    freshness_status                VARCHAR(50)     DEFAULT 'fresh',
        -- Data freshness classification
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for harvest integrity verification
    harvested_at                    TIMESTAMPTZ     DEFAULT NOW(),
        -- Timestamp when the harvest was performed
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_iga_hr_source CHECK (source IN (
        'fao', 'global_forest_watch', 'world_bank', 'transparency_international',
        'unep', 'iucn', 'eu_commission', 'national_statistics', 'custom'
    )),
    CONSTRAINT chk_iga_hr_freshness CHECK (freshness_status IN (
        'fresh', 'recent', 'aging', 'stale', 'expired'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_hr_source ON eudr_iga_harvest_results (source);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_hr_data_type ON eudr_iga_harvest_results (data_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_hr_country ON eudr_iga_harvest_results (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_hr_commodity ON eudr_iga_harvest_results (commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_hr_data_timestamp ON eudr_iga_harvest_results (data_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_hr_freshness ON eudr_iga_harvest_results (freshness_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_hr_provenance ON eudr_iga_harvest_results (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_hr_harvested ON eudr_iga_harvest_results (harvested_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_hr_created ON eudr_iga_harvest_results (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_hr_source_country ON eudr_iga_harvest_results (source, country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_hr_source_commodity ON eudr_iga_harvest_results (source, commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for stale or expired data needing refresh
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_hr_stale ON eudr_iga_harvest_results (harvested_at DESC, source)
        WHERE freshness_status IN ('stale', 'expired');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_iga_harvest_results IS 'AGENT-EUDR-027: Public data harvest results from FAO, Global Forest Watch, World Bank, Transparency International, and other public sources';
COMMENT ON COLUMN eudr_iga_harvest_results.freshness_status IS 'Data freshness: fresh (<24h), recent (<7d), aging (<30d), stale (<90d), expired (>90d)';
COMMENT ON COLUMN eudr_iga_harvest_results.is_incremental IS 'TRUE = incremental delta update; FALSE = full data refresh';


-- ============================================================================
-- 9. eudr_iga_audit_log — Audit log for Article 31 compliance
-- ============================================================================
RAISE NOTICE 'V115 [9/9]: Creating eudr_iga_audit_log...';

CREATE TABLE IF NOT EXISTS eudr_iga_audit_log (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    entry_id                        VARCHAR(100)    UNIQUE NOT NULL,
        -- Unique audit entry identifier (e.g. "iga-audit-2026-03-001")
    operation                       VARCHAR(100)    NOT NULL,
        -- Operation that was performed
    entity_type                     VARCHAR(100)    NOT NULL,
        -- Type of entity affected by the operation
    entity_id                       VARCHAR(200)    NOT NULL,
        -- Identifier of the entity affected
    actor                           VARCHAR(100)    DEFAULT 'gl-eudr-iga-027',
        -- Actor who performed the operation (system agent or user)
    details                         JSONB           DEFAULT '{}',
        -- Detailed audit context (changed fields, old/new values, reason)
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for audit integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_iga_al_operation CHECK (operation IN (
        'gathering_started', 'gathering_completed', 'gathering_failed',
        'source_queried', 'source_query_failed',
        'certificate_verified', 'certificate_verification_failed',
        'supplier_profile_created', 'supplier_profile_updated', 'supplier_profile_merged',
        'data_normalized', 'normalization_warning',
        'completeness_validated', 'completeness_gap_detected',
        'package_assembled', 'package_versioned', 'package_expired',
        'public_data_harvested', 'public_data_stale',
        'discrepancy_detected', 'discrepancy_resolved',
        'config_updated', 'manual_override'
    )),
    CONSTRAINT chk_iga_al_entity_type CHECK (entity_type IN (
        'gathering_operation', 'query_result', 'certificate_verification',
        'supplier_profile', 'completeness_report', 'information_package',
        'normalization_record', 'harvest_result', 'configuration'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_al_entry ON eudr_iga_audit_log (entry_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_al_operation ON eudr_iga_audit_log (operation);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_al_entity_type ON eudr_iga_audit_log (entity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_al_entity_id ON eudr_iga_audit_log (entity_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_al_actor ON eudr_iga_audit_log (actor);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_al_provenance ON eudr_iga_audit_log (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_al_created ON eudr_iga_audit_log (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_al_entity_op ON eudr_iga_audit_log (entity_type, operation, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_al_entity_id_time ON eudr_iga_audit_log (entity_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_iga_al_details ON eudr_iga_audit_log USING GIN (details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_iga_audit_log IS 'AGENT-EUDR-027: Article 31 compliant audit trail for all information gathering operations, queries, verifications, and package assembly';
COMMENT ON COLUMN eudr_iga_audit_log.actor IS 'Default actor is gl-eudr-iga-027 (system agent); overridden for manual user actions';
COMMENT ON COLUMN eudr_iga_audit_log.provenance_hash IS 'SHA-256 hash chained to previous entry for tamper-evident audit trail';


-- ============================================================================
-- FINALIZE
-- ============================================================================

RAISE NOTICE 'V115: AGENT-EUDR-027 Information Gathering Agent tables created successfully!';
RAISE NOTICE 'V115: Created 9 tables, ~95 indexes (B-tree, GIN, partial)';
RAISE NOTICE 'V115: Foreign keys: query_results, certificate_verifications, completeness_reports, information_packages, normalization_records -> gathering_operations';

COMMIT;
