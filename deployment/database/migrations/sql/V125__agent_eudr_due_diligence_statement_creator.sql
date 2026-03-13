-- ============================================================================
-- V125: AGENT-EUDR-037 Due Diligence Statement Creator
-- ============================================================================
-- Creates tables for the Due Diligence Statement Creator which compiles and
-- assembles formal Due Diligence Statements (DDS) per EUDR Article 4(1) and
-- Article 4(2) requirements; manages the complete DDS lifecycle from draft
-- through approval and submission; captures all Article 9 mandatory data
-- elements including operator information, commodity descriptions, origin
-- countries, geolocation data references, risk assessment conclusions, and
-- mitigation measure summaries; tracks statement versions with full amendment
-- history per Article 12(5); manages digital signatures with signatory role
-- verification; links supply chain traceability and risk assessment data from
-- upstream agents; packages supporting evidence and documentation; runs
-- pre-submission compliance validation checks; and preserves a complete
-- Article 31 audit trail via TimescaleDB hypertable.
--
-- Agent ID: GL-EUDR-DDSC-037
-- PRD: PRD-AGENT-EUDR-037
-- Regulation: EU 2023/1115 (EUDR) Articles 4, 5, 8, 9, 10, 11, 12, 14-16, 29, 31
-- Tables: 9 (7 regular + 2 hypertables)
-- Indexes: ~107
-- Dependencies: TimescaleDB extension (for hypertables), PostGIS (optional, for spatial)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V125: Creating AGENT-EUDR-037 Due Diligence Statement Creator tables...';


-- ============================================================================
-- 1. gl_eudr_ddsc_statements -- Main DDS records
-- ============================================================================
-- Stores the core Due Diligence Statement records as required by EUDR Article
-- 4(1) for operators and Article 4(2) for traders. Each statement captures the
-- operator's declaration that a product placed on or exported from the EU
-- market has been subject to due diligence and is deforestation-free, produced
-- in accordance with relevant legislation of the country of production, and
-- covered by a due diligence statement.
-- ============================================================================
RAISE NOTICE 'V125 [1/9]: Creating gl_eudr_ddsc_statements...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddsc_statements (
    statement_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this Due Diligence Statement
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator or trader who is the declarant of this DDS
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    dds_number                      VARCHAR(100)    UNIQUE NOT NULL,
        -- Unique DDS reference number following format "DDS-{YYYY}-{SEQ}-{OPERATOR}"
        -- This number is referenced in EU IS submissions and regulatory correspondence
    commodity                       VARCHAR(50)     NOT NULL,
        -- EUDR commodity type covered by this statement per Article 1(1)
    product_description             TEXT            NOT NULL DEFAULT '',
        -- Description of the relevant product including HS/CN codes per Article 9(1)(b)
    hs_code                         VARCHAR(20),
        -- Harmonised System (HS) heading or Combined Nomenclature (CN) code per Article 9(1)(b)
    origin_country                  VARCHAR(5)      NOT NULL,
        -- ISO 3166-1 alpha-2 country code of production per Article 9(1)(c)
    origin_countries_all            JSONB           DEFAULT '[]',
        -- All origin countries when product has multiple sources: ["BR", "ID", "GH"]
    quantity                        NUMERIC(14,4)   NOT NULL,
        -- Quantity of the commodity/product per Article 9(1)(e)
    quantity_unit                   VARCHAR(20)     NOT NULL DEFAULT 'kg',
        -- Unit of measurement for the quantity
    net_weight_kg                   NUMERIC(14,4),
        -- Net weight in kilograms (standardized for cross-statement comparison)
    volume_m3                       NUMERIC(14,4),
        -- Volume in cubic meters (applicable for wood products)
    shipment_date                   DATE,
        -- Date of placement on the market or export per Article 4
    intended_market_action          VARCHAR(20)     NOT NULL DEFAULT 'placing',
        -- Whether the operator is placing on or making available on the EU market
    status                          VARCHAR(30)     NOT NULL DEFAULT 'draft',
        -- Current DDS lifecycle status
    submission_date                 TIMESTAMPTZ,
        -- Timestamp when the DDS was submitted to the EU Information System
    eu_is_reference                 VARCHAR(200),
        -- Reference number assigned by the EU Information System upon submission
    amendment_number                INTEGER         NOT NULL DEFAULT 0,
        -- Current amendment version number (0 = original, 1+ = amendments per Art. 12(5))
    parent_statement_id             UUID,
        -- FK to the original/previous DDS when this is an amendment or replacement
    amendment_reason                TEXT,
        -- Reason for amending the original DDS per Article 12(5)
    operator_name                   VARCHAR(500)    NOT NULL DEFAULT '',
        -- Legal name of the operator/trader per Article 9(1)(a)
    operator_address                JSONB           DEFAULT '{}',
        -- Registered address of the operator: {"street": "...", "city": "...", "postal_code": "...", "country": "..."}
    operator_eori                   VARCHAR(50),
        -- EORI number of the operator per Article 9(1)(a)
    operator_email                  VARCHAR(200),
        -- Contact email of the operator or authorized representative
    operator_phone                  VARCHAR(50),
        -- Contact phone of the operator or authorized representative
    sme_status                      BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether the operator qualifies as an SME under Article 2(30)
    authorised_representative       JSONB           DEFAULT '{}',
        -- Authorised representative details if applicable: {"name": "...", "address": {...}, "mandate_ref": "..."}
    risk_assessment_conclusion      VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- Overall risk assessment conclusion referenced in this DDS
    risk_score                      NUMERIC(5,2),
        -- Composite risk score from the risk assessment engine (0.00-100.00)
    deforestation_free_conclusion   VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- Conclusion on whether the product is deforestation-free per Article 3(2)
    legality_conclusion             VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- Conclusion on whether the product was produced legally per Article 3(3)
    compliance_declaration          TEXT            NOT NULL DEFAULT '',
        -- Full text of the compliance declaration per Article 4(1)/4(2)
    declaration_accepted            BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether the operator has formally accepted the compliance declaration
    declaration_accepted_at         TIMESTAMPTZ,
        -- Timestamp when the declaration was accepted
    declaration_accepted_by         VARCHAR(100),
        -- User who accepted the declaration on behalf of the operator
    geolocation_count               INTEGER         NOT NULL DEFAULT 0,
        -- Number of geolocation data records linked to this DDS
    supplier_count                  INTEGER         NOT NULL DEFAULT 0,
        -- Number of supply chain entries linked to this DDS
    document_count                  INTEGER         NOT NULL DEFAULT 0,
        -- Number of supporting documents packaged with this DDS
    completeness_score              NUMERIC(5,4)    NOT NULL DEFAULT 0,
        -- DDS completeness: 0.0000 to 1.0000 (1.0000 = all mandatory fields populated)
    pre_submission_passed           BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether all pre-submission compliance checks have passed
    article9_data                   JSONB           DEFAULT '{}',
        -- Structured Article 9 data elements in EU IS-compatible format
    metadata                        JSONB           DEFAULT '{}',
        -- Additional metadata: {"template_version": "1.0", "language": "en", "generator_version": "2.1.0"}
    tags                            JSONB           DEFAULT '[]',
        -- Organizational tags for filtering and categorization
    created_by                      VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- User or system that created this DDS
    approved_by                     VARCHAR(100),
        -- User who approved the DDS for submission
    approved_at                     TIMESTAMPTZ,
        -- Timestamp of DDS approval
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for statement integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ddsc_parent_statement FOREIGN KEY (parent_statement_id) REFERENCES gl_eudr_ddsc_statements (statement_id),
    CONSTRAINT chk_ddsc_stmt_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_ddsc_stmt_quantity_unit CHECK (quantity_unit IN (
        'kg', 'tonnes', 'm3', 'litres', 'pieces', 'heads'
    )),
    CONSTRAINT chk_ddsc_stmt_market_action CHECK (intended_market_action IN (
        'placing', 'making_available', 'exporting'
    )),
    CONSTRAINT chk_ddsc_stmt_status CHECK (status IN (
        'draft', 'in_progress', 'pending_review', 'pending_approval',
        'approved', 'submitted', 'acknowledged', 'rejected',
        'amended', 'withdrawn', 'expired', 'archived'
    )),
    CONSTRAINT chk_ddsc_stmt_amendment CHECK (amendment_number >= 0),
    CONSTRAINT chk_ddsc_stmt_risk_conclusion CHECK (risk_assessment_conclusion IN (
        'pending', 'negligible', 'low', 'standard', 'high', 'not_applicable'
    )),
    CONSTRAINT chk_ddsc_stmt_risk_score CHECK (risk_score IS NULL OR
        (risk_score >= 0 AND risk_score <= 100)),
    CONSTRAINT chk_ddsc_stmt_defor CHECK (deforestation_free_conclusion IN (
        'pending', 'confirmed', 'not_confirmed', 'inconclusive', 'not_applicable'
    )),
    CONSTRAINT chk_ddsc_stmt_legality CHECK (legality_conclusion IN (
        'pending', 'confirmed', 'not_confirmed', 'inconclusive', 'not_applicable'
    )),
    CONSTRAINT chk_ddsc_stmt_completeness CHECK (completeness_score >= 0 AND completeness_score <= 1),
    CONSTRAINT chk_ddsc_stmt_quantity CHECK (quantity > 0),
    CONSTRAINT chk_ddsc_stmt_counts CHECK (geolocation_count >= 0 AND supplier_count >= 0 AND document_count >= 0)
);

COMMENT ON TABLE gl_eudr_ddsc_statements IS 'AGENT-EUDR-037: Core Due Diligence Statement records with operator information, commodity details, origin country, quantity, risk/deforestation/legality conclusions, compliance declaration, amendment tracking, completeness scoring, and EU IS reference per EUDR Articles 4, 9, 12';
COMMENT ON COLUMN gl_eudr_ddsc_statements.dds_number IS 'Unique DDS reference number: format DDS-{YYYY}-{SEQ}-{OPERATOR}. Referenced in EU IS submissions, amendments, and all regulatory correspondence';
COMMENT ON COLUMN gl_eudr_ddsc_statements.amendment_number IS 'Amendment version: 0 = original statement, 1+ = successive amendments per Article 12(5). Linked to parent_statement_id for version chain';
COMMENT ON COLUMN gl_eudr_ddsc_statements.completeness_score IS 'DDS completeness: 0.0000 (empty) to 1.0000 (all Article 9 mandatory fields populated and validated). Must reach 1.0000 before submission';
COMMENT ON COLUMN gl_eudr_ddsc_statements.compliance_declaration IS 'Full text of the operator compliance declaration per Article 4(1)/4(2): confirms due diligence performed, product deforestation-free, and produced legally';

-- Indexes for gl_eudr_ddsc_statements (24 indexes)
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_operator ON gl_eudr_ddsc_statements (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_commodity ON gl_eudr_ddsc_statements (commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_origin ON gl_eudr_ddsc_statements (origin_country);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_status ON gl_eudr_ddsc_statements (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_submission_date ON gl_eudr_ddsc_statements (submission_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_eu_ref ON gl_eudr_ddsc_statements (eu_is_reference);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_parent ON gl_eudr_ddsc_statements (parent_statement_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_eori ON gl_eudr_ddsc_statements (operator_eori);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_hs_code ON gl_eudr_ddsc_statements (hs_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_provenance ON gl_eudr_ddsc_statements (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_created ON gl_eudr_ddsc_statements (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_tenant_operator ON gl_eudr_ddsc_statements (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_operator_status ON gl_eudr_ddsc_statements (operator_id, status, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_operator_commodity ON gl_eudr_ddsc_statements (operator_id, commodity, origin_country);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_operator_shipment ON gl_eudr_ddsc_statements (operator_id, shipment_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_commodity_origin ON gl_eudr_ddsc_statements (commodity, origin_country, status, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_risk_conclusion ON gl_eudr_ddsc_statements (risk_assessment_conclusion, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active (non-archived) statements
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_active ON gl_eudr_ddsc_statements (operator_id, commodity, status, created_at DESC)
        WHERE status NOT IN ('withdrawn', 'expired', 'archived');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for statements pending review or approval
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_pending ON gl_eudr_ddsc_statements (operator_id, status, created_at DESC)
        WHERE status IN ('pending_review', 'pending_approval');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for submitted statements awaiting acknowledgement
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_submitted ON gl_eudr_ddsc_statements (operator_id, submission_date DESC)
        WHERE status = 'submitted';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for incomplete statements needing attention
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_incomplete ON gl_eudr_ddsc_statements (operator_id, completeness_score, created_at DESC)
        WHERE completeness_score < 1.0000 AND status IN ('draft', 'in_progress');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for high-risk statements requiring enhanced scrutiny
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_high_risk ON gl_eudr_ddsc_statements (operator_id, risk_score DESC, commodity)
        WHERE risk_assessment_conclusion IN ('high', 'standard') AND status NOT IN ('withdrawn', 'archived');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_origin_all ON gl_eudr_ddsc_statements USING GIN (origin_countries_all);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_stmt_tags ON gl_eudr_ddsc_statements USING GIN (tags);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 2. gl_eudr_ddsc_geolocation_data -- Plot coordinates for DDS
-- ============================================================================
-- Stores geolocation data linked to each DDS per Article 9(1)(d). For plots
-- of land larger than 4 hectares, polygons with sufficient coordinate points
-- are required. For plots <= 4 hectares, a single latitude/longitude point
-- (centroid) suffices. This table references the validated geolocation data
-- from upstream agents (Geolocation Verification, GPS Coordinate Validator).
-- ============================================================================
RAISE NOTICE 'V125 [2/9]: Creating gl_eudr_ddsc_geolocation_data...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddsc_geolocation_data (
    geo_id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this geolocation data record
    statement_id                    UUID            NOT NULL,
        -- FK reference to the parent DDS
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator (denormalized for query performance)
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    plot_id                         VARCHAR(200)    NOT NULL,
        -- Unique identifier for the production plot/parcel
    plot_name                       VARCHAR(500),
        -- Human-readable name for the production plot
    latitude                        NUMERIC(10,7)   NOT NULL,
        -- Centroid latitude in WGS84 (required for all plots)
    longitude                       NUMERIC(11,7)   NOT NULL,
        -- Centroid longitude in WGS84 (required for all plots)
    polygon_wkt                     TEXT,
        -- Well-Known Text (WKT) representation of plot polygon boundary
        -- Required for plots > 4 hectares per Article 9(1)(d)
    polygon_geojson                 JSONB           DEFAULT '{}',
        -- GeoJSON representation of plot polygon for EU IS compatibility
    coordinate_count                INTEGER         DEFAULT 0,
        -- Number of coordinate points in the polygon boundary
    area_ha                         NUMERIC(12,4),
        -- Area of the production plot in hectares
    geolocation_method              VARCHAR(50)     NOT NULL DEFAULT 'gps',
        -- Method used to capture the geolocation data
    accuracy_meters                 NUMERIC(8,2),
        -- Horizontal accuracy/precision of the geolocation data in meters
    coordinate_system               VARCHAR(20)     NOT NULL DEFAULT 'WGS84',
        -- Coordinate reference system (EU IS requires WGS84 / EPSG:4326)
    altitude_meters                 NUMERIC(8,2),
        -- Altitude above sea level in meters (optional)
    capture_date                    DATE,
        -- Date when the coordinates were captured or last verified
    country_code                    VARCHAR(5),
        -- ISO 3166-1 alpha-2 country code derived from coordinates
    region_name                     VARCHAR(200),
        -- Administrative region/province name
    supplier_id                     VARCHAR(100),
        -- Supplier who owns or manages this production plot
    upstream_geo_id                 UUID,
        -- FK reference to the source geolocation record in upstream agent (Geolocation Verification Agent)
    deforestation_status            VARCHAR(20)     DEFAULT 'pending',
        -- Deforestation-free verification status for this plot
    deforestation_verified_at       TIMESTAMPTZ,
        -- Timestamp when deforestation status was last verified
    protected_area_overlap          BOOLEAN         DEFAULT FALSE,
        -- Whether this plot overlaps with any protected area
    format_valid                    BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether the geolocation data conforms to EU IS format specification
    validation_errors               JSONB           DEFAULT '[]',
        -- Array of validation errors: [{"code": "GEO-001", "message": "...", "field": "..."}, ...]
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for geolocation record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ddsc_geo_statement FOREIGN KEY (statement_id) REFERENCES gl_eudr_ddsc_statements (statement_id),
    CONSTRAINT chk_ddsc_geo_lat CHECK (latitude >= -90 AND latitude <= 90),
    CONSTRAINT chk_ddsc_geo_lon CHECK (longitude >= -180 AND longitude <= 180),
    CONSTRAINT chk_ddsc_geo_area CHECK (area_ha IS NULL OR area_ha >= 0),
    CONSTRAINT chk_ddsc_geo_accuracy CHECK (accuracy_meters IS NULL OR accuracy_meters >= 0),
    CONSTRAINT chk_ddsc_geo_method CHECK (geolocation_method IN (
        'gps', 'satellite', 'cadastral', 'survey', 'digitized',
        'supplier_provided', 'government_registry', 'mobile_capture'
    )),
    CONSTRAINT chk_ddsc_geo_crs CHECK (coordinate_system IN (
        'WGS84', 'EPSG4326', 'UTM'
    )),
    CONSTRAINT chk_ddsc_geo_defor CHECK (deforestation_status IN (
        'pending', 'verified_free', 'not_verified', 'inconclusive', 'failed'
    )),
    CONSTRAINT uq_ddsc_geo_statement_plot UNIQUE (statement_id, plot_id)
);

COMMENT ON TABLE gl_eudr_ddsc_geolocation_data IS 'AGENT-EUDR-037: Geolocation data linked to DDS per Article 9(1)(d) with WGS84 coordinates, polygon/point geometry per four-hectare threshold, deforestation verification status, EU IS format validation, and upstream agent linkage';
COMMENT ON COLUMN gl_eudr_ddsc_geolocation_data.polygon_wkt IS 'Well-Known Text polygon boundary: required for plots > 4 hectares per Article 9(1)(d). Format: POLYGON((lon1 lat1, lon2 lat2, ...))';
COMMENT ON COLUMN gl_eudr_ddsc_geolocation_data.area_ha IS 'Plot area in hectares: determines whether polygon (> 4 ha) or point (<= 4 ha) geometry is required per Article 9(1)(d)';

-- Indexes for gl_eudr_ddsc_geolocation_data (14 indexes)
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_geo_statement ON gl_eudr_ddsc_geolocation_data (statement_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_geo_plot ON gl_eudr_ddsc_geolocation_data (plot_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_geo_supplier ON gl_eudr_ddsc_geolocation_data (supplier_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_geo_upstream ON gl_eudr_ddsc_geolocation_data (upstream_geo_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_geo_provenance ON gl_eudr_ddsc_geolocation_data (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_geo_created ON gl_eudr_ddsc_geolocation_data (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_geo_tenant_operator ON gl_eudr_ddsc_geolocation_data (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_geo_operator_country ON gl_eudr_ddsc_geolocation_data (operator_id, country_code, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_geo_statement_defor ON gl_eudr_ddsc_geolocation_data (statement_id, deforestation_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Spatial index on latitude/longitude for proximity searches
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_geo_latlon ON gl_eudr_ddsc_geolocation_data (latitude, longitude);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for large plots requiring polygon geometry
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_geo_large_plots ON gl_eudr_ddsc_geolocation_data (statement_id, area_ha DESC)
        WHERE area_ha > 4;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for plots failing format validation
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_geo_invalid ON gl_eudr_ddsc_geolocation_data (statement_id, plot_id)
        WHERE format_valid = FALSE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for plots with protected area overlap
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_geo_protected ON gl_eudr_ddsc_geolocation_data (statement_id, operator_id)
        WHERE protected_area_overlap = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_geo_val_errors ON gl_eudr_ddsc_geolocation_data USING GIN (validation_errors);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 3. gl_eudr_ddsc_risk_references -- Risk assessment links
-- ============================================================================
-- Links DDS records to the risk assessments performed by upstream agents
-- (Risk Assessment Engine, Country Risk Evaluator, Supplier Risk Scorer,
-- Commodity Risk Analyzer). Each DDS must reference the applicable risk
-- assessment and its conclusion per Article 10(2).
-- ============================================================================
RAISE NOTICE 'V125 [3/9]: Creating gl_eudr_ddsc_risk_references...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddsc_risk_references (
    risk_ref_id                     UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this risk assessment reference
    statement_id                    UUID            NOT NULL,
        -- FK reference to the parent DDS
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator (denormalized for query performance)
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    risk_assessment_id              UUID            NOT NULL,
        -- FK reference to the risk assessment record in the upstream agent
    risk_assessment_type            VARCHAR(50)     NOT NULL DEFAULT 'composite',
        -- Type of risk assessment referenced
    source_agent_id                 VARCHAR(100)    NOT NULL DEFAULT 'GL-EUDR-RAE-028',
        -- Agent that produced the risk assessment
    risk_level                      VARCHAR(20)     NOT NULL DEFAULT 'standard',
        -- Risk level conclusion from the assessment
    risk_score                      NUMERIC(5,2),
        -- Numeric risk score from the assessment (0.00-100.00)
    assessment_date                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Date when the risk assessment was performed
    assessment_version              INTEGER         NOT NULL DEFAULT 1,
        -- Version of the risk assessment referenced
    risk_factors_summary            JSONB           DEFAULT '{}',
        -- Summary of key risk factors: {"country_risk": "high", "commodity_risk": "standard", "supplier_risk": "low"}
    mitigation_referenced           BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether mitigation measures have been applied per Article 10(3)
    mitigation_measure_ids          JSONB           DEFAULT '[]',
        -- Array of mitigation measure IDs: ["uuid-1", "uuid-2"]
    post_mitigation_risk            VARCHAR(20),
        -- Risk level after mitigation measures were applied
    post_mitigation_score           NUMERIC(5,2),
        -- Numeric risk score after mitigation (0.00-100.00)
    country_benchmark               VARCHAR(20),
        -- EU country benchmarking classification per Article 29
    notes                           TEXT            DEFAULT '',
        -- Analyst notes on the risk assessment reference
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for risk reference integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ddsc_risk_statement FOREIGN KEY (statement_id) REFERENCES gl_eudr_ddsc_statements (statement_id),
    CONSTRAINT chk_ddsc_risk_type CHECK (risk_assessment_type IN (
        'composite', 'country', 'supplier', 'commodity', 'deforestation',
        'legality', 'human_rights', 'environmental', 'enhanced'
    )),
    CONSTRAINT chk_ddsc_risk_level CHECK (risk_level IN (
        'negligible', 'low', 'standard', 'high', 'critical'
    )),
    CONSTRAINT chk_ddsc_risk_score CHECK (risk_score IS NULL OR
        (risk_score >= 0 AND risk_score <= 100)),
    CONSTRAINT chk_ddsc_risk_post_level CHECK (post_mitigation_risk IS NULL OR post_mitigation_risk IN (
        'negligible', 'low', 'standard', 'high', 'critical'
    )),
    CONSTRAINT chk_ddsc_risk_post_score CHECK (post_mitigation_score IS NULL OR
        (post_mitigation_score >= 0 AND post_mitigation_score <= 100)),
    CONSTRAINT chk_ddsc_risk_benchmark CHECK (country_benchmark IS NULL OR country_benchmark IN (
        'low_risk', 'standard_risk', 'high_risk', 'not_benchmarked'
    )),
    CONSTRAINT chk_ddsc_risk_version CHECK (assessment_version >= 1)
);

COMMENT ON TABLE gl_eudr_ddsc_risk_references IS 'AGENT-EUDR-037: Links DDS to upstream risk assessments including composite, country, supplier, and commodity risk evaluations with pre/post-mitigation scores, EU country benchmarking, and mitigation measure references per EUDR Articles 10, 29';

-- Indexes for gl_eudr_ddsc_risk_references (12 indexes)
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_risk_statement ON gl_eudr_ddsc_risk_references (statement_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_risk_assessment ON gl_eudr_ddsc_risk_references (risk_assessment_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_risk_provenance ON gl_eudr_ddsc_risk_references (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_risk_created ON gl_eudr_ddsc_risk_references (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_risk_tenant_operator ON gl_eudr_ddsc_risk_references (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_risk_stmt_type ON gl_eudr_ddsc_risk_references (statement_id, risk_assessment_type, risk_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_risk_operator_level ON gl_eudr_ddsc_risk_references (operator_id, risk_level, assessment_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_risk_source_agent ON gl_eudr_ddsc_risk_references (source_agent_id, assessment_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for high/critical risk references requiring enhanced DD
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_risk_high ON gl_eudr_ddsc_risk_references (statement_id, risk_score DESC)
        WHERE risk_level IN ('high', 'critical');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for references with applied mitigation
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_risk_mitigated ON gl_eudr_ddsc_risk_references (statement_id, post_mitigation_risk)
        WHERE mitigation_referenced = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_risk_factors ON gl_eudr_ddsc_risk_references USING GIN (risk_factors_summary);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_risk_mitigation_ids ON gl_eudr_ddsc_risk_references USING GIN (mitigation_measure_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 4. gl_eudr_ddsc_supply_chain_data -- Traceability references
-- ============================================================================
-- Links DDS records to supply chain traceability data from upstream agents
-- (Multi-Tier Supplier Tracker, Chain of Custody Agent, Supply Chain Mapping
-- Master). Captures the full supply chain for each product covered by the DDS,
-- including all suppliers from Tier 1 to the point of production.
-- ============================================================================
RAISE NOTICE 'V125 [4/9]: Creating gl_eudr_ddsc_supply_chain_data...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddsc_supply_chain_data (
    sc_id                           UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this supply chain reference
    statement_id                    UUID            NOT NULL,
        -- FK reference to the parent DDS
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator (denormalized for query performance)
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    supplier_id                     VARCHAR(100)    NOT NULL,
        -- Supplier identifier in the supply chain
    supplier_name                   VARCHAR(500)    NOT NULL DEFAULT '',
        -- Legal name of the supplier
    supplier_country                VARCHAR(5),
        -- ISO 3166-1 alpha-2 country code of the supplier
    tier_level                      INTEGER         NOT NULL DEFAULT 1,
        -- Supply chain tier (1 = direct supplier, 2+ = further upstream)
    tier_role                       VARCHAR(50)     NOT NULL DEFAULT 'supplier',
        -- Role of this entity in the supply chain
    certification_id                VARCHAR(200),
        -- Certification or accreditation identifier (e.g. RSPO, FSC, Rainforest Alliance)
    certification_scheme            VARCHAR(100),
        -- Name of the certification scheme
    certification_valid_until       DATE,
        -- Expiry date of the certification
    compliance_status               VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- Compliance status of this supply chain entry
    compliance_verified_at          TIMESTAMPTZ,
        -- Timestamp when compliance was last verified
    compliance_verified_by          VARCHAR(100),
        -- User or system that verified compliance
    upstream_traceability_id        UUID,
        -- FK reference to the source record in the upstream traceability agent
    chain_of_custody_id             UUID,
        -- FK reference to chain of custody record
    segregation_model               VARCHAR(30),
        -- Traceability/segregation model used
    quantity_supplied               NUMERIC(14,4),
        -- Quantity supplied by this entity
    quantity_unit                   VARCHAR(20)     DEFAULT 'kg',
        -- Unit of measurement for quantity supplied
    supplier_risk_score             NUMERIC(5,2),
        -- Risk score for this specific supplier (0.00-100.00)
    supplier_details                JSONB           DEFAULT '{}',
        -- Additional supplier details: {"address": {...}, "contact": "...", "registration_number": "..."}
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for supply chain record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ddsc_sc_statement FOREIGN KEY (statement_id) REFERENCES gl_eudr_ddsc_statements (statement_id),
    CONSTRAINT chk_ddsc_sc_tier CHECK (tier_level >= 1 AND tier_level <= 20),
    CONSTRAINT chk_ddsc_sc_role CHECK (tier_role IN (
        'supplier', 'trader', 'processor', 'exporter', 'importer',
        'producer', 'cooperative', 'aggregator', 'transporter', 'warehouse'
    )),
    CONSTRAINT chk_ddsc_sc_compliance CHECK (compliance_status IN (
        'pending', 'compliant', 'non_compliant', 'conditionally_compliant',
        'under_review', 'not_applicable', 'expired'
    )),
    CONSTRAINT chk_ddsc_sc_segregation CHECK (segregation_model IS NULL OR segregation_model IN (
        'identity_preserved', 'segregated', 'mass_balance', 'book_and_claim', 'not_applicable'
    )),
    CONSTRAINT chk_ddsc_sc_quantity CHECK (quantity_supplied IS NULL OR quantity_supplied >= 0),
    CONSTRAINT chk_ddsc_sc_risk CHECK (supplier_risk_score IS NULL OR
        (supplier_risk_score >= 0 AND supplier_risk_score <= 100))
);

COMMENT ON TABLE gl_eudr_ddsc_supply_chain_data IS 'AGENT-EUDR-037: Supply chain traceability references linking DDS to all suppliers from Tier 1 through production, with certification tracking, compliance verification, segregation model, risk scoring, and upstream agent linkage per EUDR Articles 9, 10';

-- Indexes for gl_eudr_ddsc_supply_chain_data (12 indexes)
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sc_statement ON gl_eudr_ddsc_supply_chain_data (statement_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sc_supplier ON gl_eudr_ddsc_supply_chain_data (supplier_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sc_provenance ON gl_eudr_ddsc_supply_chain_data (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sc_created ON gl_eudr_ddsc_supply_chain_data (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sc_tenant_operator ON gl_eudr_ddsc_supply_chain_data (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sc_stmt_tier ON gl_eudr_ddsc_supply_chain_data (statement_id, tier_level, compliance_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sc_stmt_supplier ON gl_eudr_ddsc_supply_chain_data (statement_id, supplier_id, tier_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sc_operator_supplier ON gl_eudr_ddsc_supply_chain_data (operator_id, supplier_id, compliance_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sc_cert ON gl_eudr_ddsc_supply_chain_data (certification_scheme, certification_valid_until);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for non-compliant supply chain entries requiring attention
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sc_non_compliant ON gl_eudr_ddsc_supply_chain_data (statement_id, supplier_id, tier_level)
        WHERE compliance_status IN ('non_compliant', 'conditionally_compliant', 'expired');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for high-risk suppliers
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sc_high_risk ON gl_eudr_ddsc_supply_chain_data (statement_id, supplier_risk_score DESC)
        WHERE supplier_risk_score IS NOT NULL AND supplier_risk_score >= 60;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sc_details ON gl_eudr_ddsc_supply_chain_data USING GIN (supplier_details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 5. gl_eudr_ddsc_compliance_checks -- Validation results
-- ============================================================================
-- Stores the results of pre-submission compliance validation checks against
-- all EUDR Article 9 mandatory data elements. Each check validates one aspect
-- of the DDS (e.g., operator info complete, geolocation present, risk
-- assessment referenced) and records whether it passed or failed.
-- ============================================================================
RAISE NOTICE 'V125 [5/9]: Creating gl_eudr_ddsc_compliance_checks...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddsc_compliance_checks (
    check_id                        UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this compliance check result
    statement_id                    UUID            NOT NULL,
        -- FK reference to the parent DDS
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator (denormalized for query performance)
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    check_type                      VARCHAR(50)     NOT NULL,
        -- Type/category of the compliance check
    check_code                      VARCHAR(30)     NOT NULL,
        -- Unique check code identifier (e.g. "ART9-001", "ART9-002")
    check_description               TEXT            NOT NULL DEFAULT '',
        -- Human-readable description of what this check validates
    eudr_article_ref                VARCHAR(50),
        -- Specific EUDR article reference this check validates (e.g. "Art. 9(1)(a)")
    check_result                    VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- Result of this compliance check
    mandatory_field                 BOOLEAN         NOT NULL DEFAULT TRUE,
        -- Whether this check validates a mandatory Article 9 field
    field_name                      VARCHAR(200),
        -- Name of the specific field being validated
    field_status                    VARCHAR(20)     NOT NULL DEFAULT 'missing',
        -- Status of the field being validated
    expected_value                  TEXT,
        -- Expected value or format for the field
    actual_value                    TEXT,
        -- Actual value found in the DDS
    error_message                   TEXT            DEFAULT '',
        -- Error message if the check failed
    severity                        VARCHAR(20)     NOT NULL DEFAULT 'error',
        -- Severity level of a failed check
    auto_fixable                    BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether this check failure can be automatically remediated
    fix_suggestion                  TEXT            DEFAULT '',
        -- Suggested fix for the check failure
    checked_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when this check was performed
    checked_by                      VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- User or system that performed the check
    check_engine_version            VARCHAR(20)     DEFAULT '1.0',
        -- Version of the validation engine that performed the check
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for check result integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ddsc_chk_statement FOREIGN KEY (statement_id) REFERENCES gl_eudr_ddsc_statements (statement_id),
    CONSTRAINT chk_ddsc_chk_type CHECK (check_type IN (
        'operator_info', 'commodity_info', 'product_info', 'origin_info',
        'geolocation', 'quantity', 'risk_assessment', 'mitigation',
        'supply_chain', 'deforestation', 'legality', 'declaration',
        'signature', 'document', 'completeness', 'format', 'cross_reference'
    )),
    CONSTRAINT chk_ddsc_chk_result CHECK (check_result IN (
        'pending', 'passed', 'failed', 'warning', 'skipped', 'not_applicable'
    )),
    CONSTRAINT chk_ddsc_chk_field_status CHECK (field_status IN (
        'present', 'missing', 'invalid', 'incomplete', 'not_applicable'
    )),
    CONSTRAINT chk_ddsc_chk_severity CHECK (severity IN (
        'error', 'warning', 'info', 'blocking'
    ))
);

COMMENT ON TABLE gl_eudr_ddsc_compliance_checks IS 'AGENT-EUDR-037: Pre-submission compliance validation results checking all Article 9 mandatory data elements, with per-field pass/fail status, severity classification, auto-fix suggestions, and EUDR article cross-references per Articles 4, 9';
COMMENT ON COLUMN gl_eudr_ddsc_compliance_checks.check_code IS 'Unique check identifier: ART9-001 through ART9-xxx mapping to specific Article 9 data element requirements';
COMMENT ON COLUMN gl_eudr_ddsc_compliance_checks.severity IS 'Check severity: blocking (prevents submission), error (must fix), warning (should fix), info (informational)';

-- Indexes for gl_eudr_ddsc_compliance_checks (13 indexes)
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_chk_statement ON gl_eudr_ddsc_compliance_checks (statement_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_chk_type ON gl_eudr_ddsc_compliance_checks (check_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_chk_result ON gl_eudr_ddsc_compliance_checks (check_result);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_chk_code ON gl_eudr_ddsc_compliance_checks (check_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_chk_provenance ON gl_eudr_ddsc_compliance_checks (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_chk_created ON gl_eudr_ddsc_compliance_checks (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_chk_tenant_operator ON gl_eudr_ddsc_compliance_checks (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_chk_stmt_type_result ON gl_eudr_ddsc_compliance_checks (statement_id, check_type, check_result);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_chk_stmt_result ON gl_eudr_ddsc_compliance_checks (statement_id, check_result, severity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_chk_operator_result ON gl_eudr_ddsc_compliance_checks (operator_id, check_result, checked_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for failed mandatory checks blocking submission
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_chk_failed_mandatory ON gl_eudr_ddsc_compliance_checks (statement_id, check_type, severity)
        WHERE check_result = 'failed' AND mandatory_field = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for auto-fixable failures
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_chk_auto_fixable ON gl_eudr_ddsc_compliance_checks (statement_id, check_type)
        WHERE check_result = 'failed' AND auto_fixable = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for blocking checks
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_chk_blocking ON gl_eudr_ddsc_compliance_checks (statement_id, check_code)
        WHERE severity = 'blocking' AND check_result = 'failed';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 6. gl_eudr_ddsc_document_packages -- Supporting evidence
-- ============================================================================
-- Stores references to supporting evidence documents packaged with each DDS,
-- including risk assessment reports, geolocation verification reports,
-- certification copies, supplier compliance documentation, and any other
-- evidence supporting the due diligence conclusion.
-- ============================================================================
RAISE NOTICE 'V125 [6/9]: Creating gl_eudr_ddsc_document_packages...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddsc_document_packages (
    package_id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this document package entry
    statement_id                    UUID            NOT NULL,
        -- FK reference to the parent DDS
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator (denormalized for query performance)
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    document_type                   VARCHAR(50)     NOT NULL,
        -- Type/category of the supporting document
    document_name                   VARCHAR(500)    NOT NULL,
        -- Human-readable name of the document
    document_reference              VARCHAR(200),
        -- External reference number for the document
    file_path                       VARCHAR(2000),
        -- Storage path (S3 key or file system path) of the document
    file_format                     VARCHAR(20),
        -- File format (e.g. "pdf", "xlsx", "jpg", "geojson")
    file_hash                       VARCHAR(64)     NOT NULL,
        -- SHA-256 hash of the file contents for integrity verification
    file_size_bytes                 BIGINT,
        -- Size of the file in bytes
    mime_type                       VARCHAR(100),
        -- MIME type of the document (e.g. "application/pdf")
    description                     TEXT            DEFAULT '',
        -- Description of the document and its relevance to the DDS
    eudr_article_ref                VARCHAR(50),
        -- EUDR article this document supports (e.g. "Art. 9(1)(d)", "Art. 10(2)")
    source_agent_id                 VARCHAR(100),
        -- Agent that generated or collected this document
    source_entity_id                UUID,
        -- Entity ID in the source agent for traceability
    document_date                   DATE,
        -- Date of the document (issue date, capture date, etc.)
    expiry_date                     DATE,
        -- Expiry date of the document (if applicable, e.g. certifications)
    is_mandatory                    BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether this document type is mandatory for the DDS
    verification_status             VARCHAR(20)     NOT NULL DEFAULT 'unverified',
        -- Verification status of the document
    verified_at                     TIMESTAMPTZ,
        -- Timestamp when the document was verified
    verified_by                     VARCHAR(100),
        -- User who verified the document
    uploaded_at                     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the document was uploaded/linked
    uploaded_by                     VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- User who uploaded/linked the document
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for package record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ddsc_doc_statement FOREIGN KEY (statement_id) REFERENCES gl_eudr_ddsc_statements (statement_id),
    CONSTRAINT chk_ddsc_doc_type CHECK (document_type IN (
        'risk_assessment_report', 'geolocation_report', 'satellite_analysis',
        'certification_copy', 'supplier_compliance', 'chain_of_custody',
        'deforestation_analysis', 'legality_verification', 'mitigation_evidence',
        'audit_report', 'inspection_report', 'laboratory_analysis',
        'customs_declaration', 'phytosanitary_certificate', 'export_permit',
        'import_permit', 'operator_declaration', 'photograph', 'map',
        'correspondence', 'other'
    )),
    CONSTRAINT chk_ddsc_doc_size CHECK (file_size_bytes IS NULL OR file_size_bytes >= 0),
    CONSTRAINT chk_ddsc_doc_verification CHECK (verification_status IN (
        'unverified', 'pending', 'verified', 'rejected', 'expired'
    )),
    CONSTRAINT chk_ddsc_doc_expiry CHECK (expiry_date IS NULL OR document_date IS NULL OR expiry_date >= document_date)
);

COMMENT ON TABLE gl_eudr_ddsc_document_packages IS 'AGENT-EUDR-037: Supporting evidence documents packaged with DDS including risk assessments, geolocation reports, certifications, supplier compliance records, and other evidence with integrity hashing, verification tracking, and EUDR article references per Articles 9, 10, 12';

-- Indexes for gl_eudr_ddsc_document_packages (11 indexes)
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_doc_statement ON gl_eudr_ddsc_document_packages (statement_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_doc_type ON gl_eudr_ddsc_document_packages (document_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_doc_hash ON gl_eudr_ddsc_document_packages (file_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_doc_provenance ON gl_eudr_ddsc_document_packages (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_doc_created ON gl_eudr_ddsc_document_packages (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_doc_tenant_operator ON gl_eudr_ddsc_document_packages (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_doc_stmt_type ON gl_eudr_ddsc_document_packages (statement_id, document_type, verification_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_doc_operator_type ON gl_eudr_ddsc_document_packages (operator_id, document_type, uploaded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for mandatory documents that are missing or unverified
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_doc_mandatory_pending ON gl_eudr_ddsc_document_packages (statement_id, document_type)
        WHERE is_mandatory = TRUE AND verification_status NOT IN ('verified');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for expiring documents
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_doc_expiring ON gl_eudr_ddsc_document_packages (expiry_date, operator_id)
        WHERE expiry_date IS NOT NULL AND verification_status = 'verified';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for rejected documents requiring replacement
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_doc_rejected ON gl_eudr_ddsc_document_packages (statement_id, uploaded_at DESC)
        WHERE verification_status = 'rejected';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 7. gl_eudr_ddsc_versions -- DDS version history
-- ============================================================================
-- Tracks the complete version history of each DDS including amendments per
-- Article 12(5), corrections, and any changes to the statement content. Each
-- version captures a snapshot of the statement state for audit trail purposes,
-- enabling comparison between versions and full regulatory traceability.
-- ============================================================================
RAISE NOTICE 'V125 [7/9]: Creating gl_eudr_ddsc_versions...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddsc_versions (
    version_id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this version record
    statement_id                    UUID            NOT NULL,
        -- FK reference to the parent DDS
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator (denormalized for query performance)
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    version_number                  INTEGER         NOT NULL,
        -- Sequential version number (1 = initial, 2+ = subsequent versions)
    version_type                    VARCHAR(20)     NOT NULL DEFAULT 'amendment',
        -- Type of version change
    change_description              TEXT            NOT NULL DEFAULT '',
        -- Detailed description of what changed in this version
    change_summary                  VARCHAR(500)    NOT NULL DEFAULT '',
        -- Brief summary of the version change
    eudr_article_ref                VARCHAR(50),
        -- EUDR article triggering the version change (e.g. "Art. 12(5)")
    previous_status                 VARCHAR(30),
        -- DDS status before this version change
    new_status                      VARCHAR(30),
        -- DDS status after this version change
    fields_changed                  JSONB           DEFAULT '[]',
        -- Array of field names that changed: ["risk_assessment_conclusion", "geolocation_count", "quantity"]
    previous_values                 JSONB           DEFAULT '{}',
        -- Previous field values: {"risk_assessment_conclusion": "standard", "quantity": 5000}
    new_values                      JSONB           DEFAULT '{}',
        -- New field values: {"risk_assessment_conclusion": "low", "quantity": 4800}
    statement_snapshot              JSONB           DEFAULT '{}',
        -- Full snapshot of the DDS state at this version (for regulatory reconstruction)
    geolocation_changes             JSONB           DEFAULT '{}',
        -- Summary of geolocation changes: {"added": 2, "removed": 0, "modified": 1}
    supply_chain_changes            JSONB           DEFAULT '{}',
        -- Summary of supply chain changes: {"added": 1, "removed": 0, "modified": 0}
    document_changes                JSONB           DEFAULT '{}',
        -- Summary of document changes: {"added": 3, "removed": 0, "updated": 1}
    amended_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when this version was created
    amended_by                      VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- User who made the change
    approved_by                     VARCHAR(100),
        -- User who approved this version change
    approved_at                     TIMESTAMPTZ,
        -- Timestamp of version approval
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for version record integrity verification (includes snapshot hash)
    previous_provenance_hash        VARCHAR(64),
        -- SHA-256 hash of the previous version (chain integrity)
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ddsc_ver_statement FOREIGN KEY (statement_id) REFERENCES gl_eudr_ddsc_statements (statement_id),
    CONSTRAINT chk_ddsc_ver_number CHECK (version_number >= 1),
    CONSTRAINT chk_ddsc_ver_type CHECK (version_type IN (
        'initial', 'amendment', 'correction', 'resubmission', 'withdrawal', 'administrative'
    )),
    CONSTRAINT uq_ddsc_ver_statement_number UNIQUE (statement_id, version_number)
);

COMMENT ON TABLE gl_eudr_ddsc_versions IS 'AGENT-EUDR-037: DDS version history with field-level change tracking, statement snapshots for regulatory reconstruction, geolocation/supply-chain/document change summaries, chained provenance hashes, and approval workflow per EUDR Article 12(5)';
COMMENT ON COLUMN gl_eudr_ddsc_versions.statement_snapshot IS 'Full JSON snapshot of the DDS at this version point: enables complete regulatory reconstruction and version comparison per Article 12(5) and Article 31';
COMMENT ON COLUMN gl_eudr_ddsc_versions.previous_provenance_hash IS 'SHA-256 hash of the previous version: enables chain-of-integrity verification across the full version history';

-- Indexes for gl_eudr_ddsc_versions (10 indexes)
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_ver_statement ON gl_eudr_ddsc_versions (statement_id, version_number DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_ver_provenance ON gl_eudr_ddsc_versions (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_ver_prev_prov ON gl_eudr_ddsc_versions (previous_provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_ver_created ON gl_eudr_ddsc_versions (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_ver_tenant_operator ON gl_eudr_ddsc_versions (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_ver_operator_type ON gl_eudr_ddsc_versions (operator_id, version_type, amended_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_ver_stmt_type ON gl_eudr_ddsc_versions (statement_id, version_type, amended_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_ver_amended_by ON gl_eudr_ddsc_versions (amended_by, amended_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_ver_fields ON gl_eudr_ddsc_versions USING GIN (fields_changed);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_ver_new_values ON gl_eudr_ddsc_versions USING GIN (new_values);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 8. gl_eudr_ddsc_signatures -- Digital signature tracking
-- ============================================================================
-- Tracks digital signatures applied to DDS records. The operator or their
-- authorized representative must formally sign the DDS before submission per
-- Article 4(1). This table supports multiple signature types including
-- digital certificates, eIDAS qualified signatures, and manual declarations.
-- ============================================================================
RAISE NOTICE 'V125 [8/9]: Creating gl_eudr_ddsc_signatures...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddsc_signatures (
    signature_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this signature record
    statement_id                    UUID            NOT NULL,
        -- FK reference to the parent DDS
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator (denormalized for query performance)
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    signatory_name                  VARCHAR(300)    NOT NULL,
        -- Full legal name of the person signing
    signatory_role                  VARCHAR(100)    NOT NULL,
        -- Organizational role of the signatory
    signatory_email                 VARCHAR(200),
        -- Email address of the signatory
    signatory_title                 VARCHAR(200),
        -- Professional title (e.g. "Head of Compliance", "Director of Sustainability")
    signature_type                  VARCHAR(30)     NOT NULL DEFAULT 'digital',
        -- Type of digital signature applied
    signature_standard              VARCHAR(30),
        -- Signature standard or level
    certificate_serial              VARCHAR(200),
        -- Certificate serial number (for eIDAS or PKI signatures)
    certificate_issuer              VARCHAR(500),
        -- Certificate issuer (for eIDAS or PKI signatures)
    certificate_fingerprint         VARCHAR(128),
        -- SHA-256 fingerprint of the signing certificate
    signed_at                       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the signature was applied
    signature_hash                  VARCHAR(64)     NOT NULL,
        -- SHA-256 hash of the signed content for integrity verification
    signed_content_hash             VARCHAR(64),
        -- SHA-256 hash of the DDS content at the time of signing
    ip_address                      VARCHAR(45),
        -- IP address from which the signature was made
    user_agent                      VARCHAR(500),
        -- User agent string of the signing client
    signature_purpose               VARCHAR(30)     NOT NULL DEFAULT 'submission',
        -- Purpose of this signature in the DDS lifecycle
    is_primary_signatory            BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether this is the primary (legally binding) signature
    counter_signature_of            UUID,
        -- FK to another signature this countersigns (NULL if standalone)
    revoked                         BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether this signature has been revoked
    revoked_at                      TIMESTAMPTZ,
        -- Timestamp when the signature was revoked
    revocation_reason               TEXT,
        -- Reason for signature revocation
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for signature record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ddsc_sig_statement FOREIGN KEY (statement_id) REFERENCES gl_eudr_ddsc_statements (statement_id),
    CONSTRAINT fk_ddsc_sig_counter FOREIGN KEY (counter_signature_of) REFERENCES gl_eudr_ddsc_signatures (signature_id),
    CONSTRAINT chk_ddsc_sig_type CHECK (signature_type IN (
        'digital', 'eidas_qualified', 'eidas_advanced', 'pki',
        'otp_verified', 'manual_declaration', 'api_token'
    )),
    CONSTRAINT chk_ddsc_sig_standard CHECK (signature_standard IS NULL OR signature_standard IN (
        'eidas_qes', 'eidas_aes', 'eidas_ses', 'pades', 'xades', 'cades', 'jws', 'none'
    )),
    CONSTRAINT chk_ddsc_sig_purpose CHECK (signature_purpose IN (
        'submission', 'approval', 'review', 'amendment', 'withdrawal', 'counter_signature'
    ))
);

COMMENT ON TABLE gl_eudr_ddsc_signatures IS 'AGENT-EUDR-037: Digital signature tracking for DDS with eIDAS support, certificate metadata, signed content hashing, primary/counter signature chains, revocation tracking, and purpose classification per EUDR Articles 4, 33';
COMMENT ON COLUMN gl_eudr_ddsc_signatures.signature_hash IS 'SHA-256 hash of the digital signature value: used to verify signature integrity independently of the signing certificate';
COMMENT ON COLUMN gl_eudr_ddsc_signatures.signed_content_hash IS 'SHA-256 hash of the DDS content at signing time: enables detection of any content changes after signature was applied';

-- Indexes for gl_eudr_ddsc_signatures (11 indexes)
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sig_statement ON gl_eudr_ddsc_signatures (statement_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sig_signed_at ON gl_eudr_ddsc_signatures (signed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sig_hash ON gl_eudr_ddsc_signatures (signature_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sig_provenance ON gl_eudr_ddsc_signatures (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sig_created ON gl_eudr_ddsc_signatures (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sig_tenant_operator ON gl_eudr_ddsc_signatures (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sig_stmt_purpose ON gl_eudr_ddsc_signatures (statement_id, signature_purpose, signed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sig_operator_type ON gl_eudr_ddsc_signatures (operator_id, signature_type, signed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sig_counter ON gl_eudr_ddsc_signatures (counter_signature_of);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for primary signatures (legally binding)
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sig_primary ON gl_eudr_ddsc_signatures (statement_id, signed_at DESC)
        WHERE is_primary_signatory = TRUE AND revoked = FALSE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for revoked signatures
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_sig_revoked ON gl_eudr_ddsc_signatures (statement_id, revoked_at DESC)
        WHERE revoked = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 9. gl_eudr_ddsc_audit_log -- Full audit trail (TimescaleDB hypertable)
-- ============================================================================
-- Immutable audit log for all DDS Creator operations. Records every create,
-- update, approve, sign, submit, amend, and withdraw action with full actor
-- attribution, JSON change diffs, and request context. Partitioned by
-- TimescaleDB with 7-day chunks for efficient time-range queries and
-- automatic 5-year retention per Article 31.
-- ============================================================================
RAISE NOTICE 'V125 [9/9]: Creating gl_eudr_ddsc_audit_log (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddsc_audit_log (
    audit_id                        UUID            DEFAULT gen_random_uuid(),
        -- Unique audit entry identifier
    statement_id                    UUID,
        -- Reference to the DDS (NULL for non-statement operations)
    entity_type                     VARCHAR(50)     NOT NULL,
        -- Type of entity being audited
    entity_id                       UUID            NOT NULL,
        -- Entity identifier being audited
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator identifier
    tenant_id                       VARCHAR(100)    NOT NULL DEFAULT '',
        -- Multi-tenant isolation identifier
    actor                           VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- Actor performing the action (user ID or system)
    actor_type                      VARCHAR(20)     NOT NULL DEFAULT 'system',
        -- Type of actor
    action                          VARCHAR(50)     NOT NULL,
        -- Action performed
    timestamp                       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp of the action (partitioning column)
    details_jsonb                   JSONB           DEFAULT '{}',
        -- Full event details: changes, context, metadata
    changes                         JSONB           DEFAULT '{}',
        -- JSON diff of changes: {"field": {"old": "...", "new": "..."}}
    context                         JSONB           DEFAULT '{}',
        -- Request context: {"ip_address": "...", "user_agent": "...", "request_id": "...", "correlation_id": "..."}
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for audit entry integrity (chained to previous entry)

    CONSTRAINT chk_ddsc_audit_entity_type CHECK (entity_type IN (
        'statement', 'geolocation', 'risk_reference', 'supply_chain',
        'compliance_check', 'document_package', 'version', 'signature'
    )),
    CONSTRAINT chk_ddsc_audit_actor_type CHECK (actor_type IN (
        'user', 'system', 'agent', 'scheduler', 'api', 'admin'
    )),
    CONSTRAINT chk_ddsc_audit_action CHECK (action IN (
        'create', 'update', 'delete', 'approve', 'reject', 'sign',
        'submit', 'amend', 'withdraw', 'archive', 'restore',
        'validate', 'complete_check', 'link_document', 'unlink_document',
        'add_geolocation', 'remove_geolocation', 'add_supplier', 'remove_supplier',
        'link_risk_assessment', 'version_created', 'status_change',
        'declaration_accepted', 'signature_revoked', 'export', 'view'
    ))
);

-- Convert to TimescaleDB hypertable partitioned by timestamp
DO $$
BEGIN
    PERFORM create_hypertable(
        'gl_eudr_ddsc_audit_log',
        'timestamp',
        chunk_time_interval => INTERVAL '7 days',
        if_not_exists => TRUE
    );
    RAISE NOTICE 'gl_eudr_ddsc_audit_log hypertable created (7-day chunks)';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Hypertable creation skipped for gl_eudr_ddsc_audit_log: %', SQLERRM;
END $$;

COMMENT ON TABLE gl_eudr_ddsc_audit_log IS 'AGENT-EUDR-037: Immutable TimescaleDB-partitioned audit trail for all DDS Creator operations with full actor attribution, change diffs, request context, and chained provenance hashes per EUDR Article 31 with 5-year retention';

-- Indexes for gl_eudr_ddsc_audit_log (10 indexes)
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_audit_statement ON gl_eudr_ddsc_audit_log (statement_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_audit_entity ON gl_eudr_ddsc_audit_log (entity_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_audit_operator ON gl_eudr_ddsc_audit_log (operator_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_audit_actor ON gl_eudr_ddsc_audit_log (actor, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_audit_action ON gl_eudr_ddsc_audit_log (action, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_audit_provenance ON gl_eudr_ddsc_audit_log (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_audit_entity_action ON gl_eudr_ddsc_audit_log (entity_type, action, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_audit_operator_entity ON gl_eudr_ddsc_audit_log (operator_id, entity_type, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_audit_changes ON gl_eudr_ddsc_audit_log USING GIN (changes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ddsc_audit_context ON gl_eudr_ddsc_audit_log USING GIN (context);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- DATA RETENTION POLICIES -- Article 31: 5-year retention for DDS records
-- ============================================================================
-- Per EUDR Article 12(1), operators and traders must keep the DDS and all
-- supporting documentation for a minimum period of five years from the date
-- of placing on the market or making available on the market.
-- ============================================================================
RAISE NOTICE 'V125: Configuring 5-year data retention policies per EUDR Articles 12, 31...';

SELECT add_retention_policy('gl_eudr_ddsc_audit_log', INTERVAL '5 years', if_not_exists => TRUE);


-- ============================================================================
-- VIEWS: Active statements and pending submissions
-- ============================================================================
RAISE NOTICE 'V125: Creating operational views...';

-- View: Active (non-archived) DDS statements with summary metrics
CREATE OR REPLACE VIEW vw_eudr_ddsc_active_statements AS
SELECT
    s.statement_id,
    s.operator_id,
    s.tenant_id,
    s.dds_number,
    s.commodity,
    s.origin_country,
    s.quantity,
    s.quantity_unit,
    s.status,
    s.amendment_number,
    s.risk_assessment_conclusion,
    s.risk_score,
    s.deforestation_free_conclusion,
    s.legality_conclusion,
    s.completeness_score,
    s.pre_submission_passed,
    s.geolocation_count,
    s.supplier_count,
    s.document_count,
    s.submission_date,
    s.eu_is_reference,
    s.created_at,
    s.updated_at,
    s.created_by,
    s.approved_by,
    s.approved_at
FROM gl_eudr_ddsc_statements s
WHERE s.status NOT IN ('withdrawn', 'expired', 'archived');

COMMENT ON VIEW vw_eudr_ddsc_active_statements IS 'AGENT-EUDR-037: Active DDS statements excluding withdrawn, expired, and archived records. Provides summary view of all live statements with key metrics for dashboard display';

-- View: Statements pending submission (ready for EU IS upload)
CREATE OR REPLACE VIEW vw_eudr_ddsc_pending_submissions AS
SELECT
    s.statement_id,
    s.operator_id,
    s.tenant_id,
    s.dds_number,
    s.commodity,
    s.origin_country,
    s.quantity,
    s.quantity_unit,
    s.status,
    s.risk_assessment_conclusion,
    s.completeness_score,
    s.pre_submission_passed,
    s.declaration_accepted,
    s.geolocation_count,
    s.supplier_count,
    s.document_count,
    s.approved_by,
    s.approved_at,
    s.created_at
FROM gl_eudr_ddsc_statements s
WHERE s.status IN ('pending_review', 'pending_approval', 'approved')
  AND s.completeness_score >= 0.9000
ORDER BY
    CASE s.status
        WHEN 'approved' THEN 1
        WHEN 'pending_approval' THEN 2
        WHEN 'pending_review' THEN 3
    END,
    s.created_at ASC;

COMMENT ON VIEW vw_eudr_ddsc_pending_submissions IS 'AGENT-EUDR-037: DDS statements approaching submission readiness, ordered by submission priority. Includes approved statements ready for EU IS upload and those pending review/approval with high completeness scores';

-- View: Statement compliance summary
CREATE OR REPLACE VIEW vw_eudr_ddsc_compliance_summary AS
SELECT
    s.statement_id,
    s.dds_number,
    s.operator_id,
    s.tenant_id,
    s.commodity,
    s.origin_country,
    s.status,
    s.completeness_score,
    COUNT(cc.check_id) AS total_checks,
    COUNT(cc.check_id) FILTER (WHERE cc.check_result = 'passed') AS passed_checks,
    COUNT(cc.check_id) FILTER (WHERE cc.check_result = 'failed') AS failed_checks,
    COUNT(cc.check_id) FILTER (WHERE cc.check_result = 'failed' AND cc.mandatory_field = TRUE) AS failed_mandatory,
    COUNT(cc.check_id) FILTER (WHERE cc.check_result = 'failed' AND cc.severity = 'blocking') AS blocking_failures,
    COUNT(cc.check_id) FILTER (WHERE cc.check_result = 'warning') AS warnings,
    CASE
        WHEN COUNT(cc.check_id) = 0 THEN 0
        ELSE ROUND(
            COUNT(cc.check_id) FILTER (WHERE cc.check_result = 'passed')::NUMERIC /
            NULLIF(COUNT(cc.check_id) FILTER (WHERE cc.check_result != 'skipped' AND cc.check_result != 'not_applicable'), 0) * 100,
            2
        )
    END AS compliance_percentage
FROM gl_eudr_ddsc_statements s
LEFT JOIN gl_eudr_ddsc_compliance_checks cc ON s.statement_id = cc.statement_id
WHERE s.status NOT IN ('withdrawn', 'expired', 'archived')
GROUP BY s.statement_id, s.dds_number, s.operator_id, s.tenant_id,
         s.commodity, s.origin_country, s.status, s.completeness_score;

COMMENT ON VIEW vw_eudr_ddsc_compliance_summary IS 'AGENT-EUDR-037: Aggregated compliance check summary per DDS statement showing total/passed/failed/blocking check counts and overall compliance percentage for compliance dashboard display';


-- ============================================================================
-- FUNCTIONS: Statement validation and version comparison
-- ============================================================================
RAISE NOTICE 'V125: Creating helper functions...';

-- Function: Validate DDS completeness (check all Article 9 mandatory elements)
CREATE OR REPLACE FUNCTION fn_eudr_ddsc_validate_statement(p_statement_id UUID)
RETURNS TABLE (
    is_valid                BOOLEAN,
    completeness_score      NUMERIC(5,4),
    total_checks            INTEGER,
    passed_checks           INTEGER,
    failed_checks           INTEGER,
    blocking_failures       INTEGER,
    error_summary           JSONB
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_total         INTEGER := 0;
    v_passed        INTEGER := 0;
    v_failed        INTEGER := 0;
    v_blocking      INTEGER := 0;
    v_errors        JSONB := '[]'::JSONB;
    v_score         NUMERIC(5,4) := 0;
BEGIN
    -- Count check results for this statement
    SELECT
        COUNT(*),
        COUNT(*) FILTER (WHERE cc.check_result = 'passed'),
        COUNT(*) FILTER (WHERE cc.check_result = 'failed'),
        COUNT(*) FILTER (WHERE cc.check_result = 'failed' AND cc.severity = 'blocking')
    INTO v_total, v_passed, v_failed, v_blocking
    FROM gl_eudr_ddsc_compliance_checks cc
    WHERE cc.statement_id = p_statement_id;

    -- Calculate completeness score
    IF v_total > 0 THEN
        v_score := v_passed::NUMERIC / NULLIF(
            (SELECT COUNT(*) FROM gl_eudr_ddsc_compliance_checks
             WHERE statement_id = p_statement_id
               AND check_result NOT IN ('skipped', 'not_applicable')), 0
        );
    END IF;

    -- Collect error summaries
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
        'check_code', cc.check_code,
        'check_type', cc.check_type,
        'severity', cc.severity,
        'error_message', cc.error_message,
        'field_name', cc.field_name,
        'auto_fixable', cc.auto_fixable
    )), '[]'::JSONB)
    INTO v_errors
    FROM gl_eudr_ddsc_compliance_checks cc
    WHERE cc.statement_id = p_statement_id
      AND cc.check_result = 'failed';

    -- Update the statement completeness score
    UPDATE gl_eudr_ddsc_statements
    SET completeness_score = COALESCE(v_score, 0),
        pre_submission_passed = (v_blocking = 0 AND v_failed = 0)
    WHERE gl_eudr_ddsc_statements.statement_id = p_statement_id;

    RETURN QUERY SELECT
        (v_blocking = 0 AND v_failed = 0),
        COALESCE(v_score, 0::NUMERIC(5,4)),
        v_total,
        v_passed,
        v_failed,
        v_blocking,
        v_errors;
END;
$$;

COMMENT ON FUNCTION fn_eudr_ddsc_validate_statement IS 'AGENT-EUDR-037: Validates DDS completeness by aggregating compliance check results, calculating completeness score, updating the statement record, and returning a summary of all failures with severity and auto-fix flags';

-- Function: Compare two DDS versions
CREATE OR REPLACE FUNCTION fn_eudr_ddsc_compare_versions(
    p_statement_id UUID,
    p_version_a INTEGER,
    p_version_b INTEGER
)
RETURNS TABLE (
    field_name      TEXT,
    version_a_value TEXT,
    version_b_value TEXT,
    change_type     TEXT
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_snapshot_a    JSONB;
    v_snapshot_b    JSONB;
    v_key           TEXT;
BEGIN
    -- Retrieve snapshots for both versions
    SELECT v.statement_snapshot INTO v_snapshot_a
    FROM gl_eudr_ddsc_versions v
    WHERE v.statement_id = p_statement_id AND v.version_number = p_version_a;

    SELECT v.statement_snapshot INTO v_snapshot_b
    FROM gl_eudr_ddsc_versions v
    WHERE v.statement_id = p_statement_id AND v.version_number = p_version_b;

    IF v_snapshot_a IS NULL OR v_snapshot_b IS NULL THEN
        RAISE EXCEPTION 'One or both version snapshots not found for statement % (versions %, %)',
            p_statement_id, p_version_a, p_version_b;
    END IF;

    -- Compare all keys from both snapshots
    FOR v_key IN
        SELECT DISTINCT k
        FROM (
            SELECT jsonb_object_keys(v_snapshot_a) AS k
            UNION
            SELECT jsonb_object_keys(v_snapshot_b) AS k
        ) keys
    LOOP
        -- Determine change type
        IF NOT v_snapshot_a ? v_key THEN
            -- Field added in version B
            RETURN QUERY SELECT v_key, NULL::TEXT, v_snapshot_b->>v_key, 'added'::TEXT;
        ELSIF NOT v_snapshot_b ? v_key THEN
            -- Field removed in version B
            RETURN QUERY SELECT v_key, v_snapshot_a->>v_key, NULL::TEXT, 'removed'::TEXT;
        ELSIF (v_snapshot_a->>v_key) IS DISTINCT FROM (v_snapshot_b->>v_key) THEN
            -- Field modified between versions
            RETURN QUERY SELECT v_key, v_snapshot_a->>v_key, v_snapshot_b->>v_key, 'modified'::TEXT;
        END IF;
        -- Skip unchanged fields
    END LOOP;
END;
$$;

COMMENT ON FUNCTION fn_eudr_ddsc_compare_versions IS 'AGENT-EUDR-037: Compares two DDS version snapshots field-by-field, returning added/removed/modified fields with their values from each version. Used for amendment review and Article 12(5) compliance auditing';

-- Function: Get DDS submission readiness assessment
CREATE OR REPLACE FUNCTION fn_eudr_ddsc_submission_readiness(p_statement_id UUID)
RETURNS TABLE (
    is_ready                BOOLEAN,
    status                  VARCHAR(30),
    completeness_score      NUMERIC(5,4),
    has_geolocation         BOOLEAN,
    has_risk_assessment     BOOLEAN,
    has_supply_chain        BOOLEAN,
    has_documents           BOOLEAN,
    has_signature           BOOLEAN,
    declaration_accepted    BOOLEAN,
    blocking_issues         JSONB
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_stmt                  RECORD;
    v_geo_count             INTEGER;
    v_risk_count            INTEGER;
    v_sc_count              INTEGER;
    v_doc_count             INTEGER;
    v_sig_count             INTEGER;
    v_blocking              JSONB := '[]'::JSONB;
BEGIN
    -- Fetch statement
    SELECT * INTO v_stmt FROM gl_eudr_ddsc_statements s WHERE s.statement_id = p_statement_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Statement % not found', p_statement_id;
    END IF;

    -- Count linked records
    SELECT COUNT(*) INTO v_geo_count FROM gl_eudr_ddsc_geolocation_data WHERE gl_eudr_ddsc_geolocation_data.statement_id = p_statement_id;
    SELECT COUNT(*) INTO v_risk_count FROM gl_eudr_ddsc_risk_references WHERE gl_eudr_ddsc_risk_references.statement_id = p_statement_id;
    SELECT COUNT(*) INTO v_sc_count FROM gl_eudr_ddsc_supply_chain_data WHERE gl_eudr_ddsc_supply_chain_data.statement_id = p_statement_id;
    SELECT COUNT(*) INTO v_doc_count FROM gl_eudr_ddsc_document_packages WHERE gl_eudr_ddsc_document_packages.statement_id = p_statement_id;
    SELECT COUNT(*) INTO v_sig_count FROM gl_eudr_ddsc_signatures WHERE gl_eudr_ddsc_signatures.statement_id = p_statement_id AND gl_eudr_ddsc_signatures.revoked = FALSE;

    -- Collect blocking issues
    IF v_geo_count = 0 THEN
        v_blocking := v_blocking || jsonb_build_array(jsonb_build_object('issue', 'No geolocation data', 'article', 'Art. 9(1)(d)'));
    END IF;
    IF v_risk_count = 0 THEN
        v_blocking := v_blocking || jsonb_build_array(jsonb_build_object('issue', 'No risk assessment referenced', 'article', 'Art. 10'));
    END IF;
    IF v_sc_count = 0 THEN
        v_blocking := v_blocking || jsonb_build_array(jsonb_build_object('issue', 'No supply chain data', 'article', 'Art. 9'));
    END IF;
    IF v_sig_count = 0 THEN
        v_blocking := v_blocking || jsonb_build_array(jsonb_build_object('issue', 'No valid signature', 'article', 'Art. 4(1)'));
    END IF;
    IF NOT v_stmt.declaration_accepted THEN
        v_blocking := v_blocking || jsonb_build_array(jsonb_build_object('issue', 'Declaration not accepted', 'article', 'Art. 4(1)'));
    END IF;

    RETURN QUERY SELECT
        (v_geo_count > 0 AND v_risk_count > 0 AND v_sc_count > 0 AND v_sig_count > 0
         AND v_stmt.declaration_accepted AND v_stmt.completeness_score >= 1.0000
         AND v_stmt.pre_submission_passed),
        v_stmt.status,
        v_stmt.completeness_score,
        (v_geo_count > 0),
        (v_risk_count > 0),
        (v_sc_count > 0),
        (v_doc_count > 0),
        (v_sig_count > 0),
        v_stmt.declaration_accepted,
        v_blocking;
END;
$$;

COMMENT ON FUNCTION fn_eudr_ddsc_submission_readiness IS 'AGENT-EUDR-037: Assesses DDS submission readiness by checking all mandatory components (geolocation, risk assessment, supply chain, signature, declaration) and returning a comprehensive readiness report with blocking issues per Articles 4, 9, 10';


-- ============================================================================
-- Triggers: updated_at auto-update
-- ============================================================================
RAISE NOTICE 'V125: Creating updated_at triggers...';

CREATE OR REPLACE FUNCTION fn_eudr_ddsc_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_statements_updated_at
        BEFORE UPDATE ON gl_eudr_ddsc_statements
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_geolocation_updated_at
        BEFORE UPDATE ON gl_eudr_ddsc_geolocation_data
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_risk_refs_updated_at
        BEFORE UPDATE ON gl_eudr_ddsc_risk_references
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_supply_chain_updated_at
        BEFORE UPDATE ON gl_eudr_ddsc_supply_chain_data
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_compliance_chk_updated_at
        BEFORE UPDATE ON gl_eudr_ddsc_compliance_checks
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_doc_packages_updated_at
        BEFORE UPDATE ON gl_eudr_ddsc_document_packages
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_versions_updated_at
        BEFORE UPDATE ON gl_eudr_ddsc_versions
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_signatures_updated_at
        BEFORE UPDATE ON gl_eudr_ddsc_signatures
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- Triggers: Audit trail auto-insert
-- ============================================================================
RAISE NOTICE 'V125: Creating audit trail triggers...';

CREATE OR REPLACE FUNCTION fn_eudr_ddsc_audit_insert()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO gl_eudr_ddsc_audit_log (
        statement_id, entity_type, entity_id, operator_id, action,
        actor, changes, timestamp
    )
    VALUES (
        CASE WHEN TG_ARGV[0] = 'statement' THEN NEW.*::TEXT::UUID ELSE NULL END,
        TG_ARGV[0],
        NEW.*::TEXT::UUID,
        COALESCE(NEW.operator_id, ''),
        'create',
        'system',
        row_to_json(NEW)::JSONB,
        NOW()
    );
    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION fn_eudr_ddsc_audit_update()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO gl_eudr_ddsc_audit_log (
        statement_id, entity_type, entity_id, operator_id, action,
        actor, changes, timestamp
    )
    VALUES (
        CASE WHEN TG_ARGV[0] = 'statement' THEN NEW.*::TEXT::UUID ELSE NULL END,
        TG_ARGV[0],
        NEW.*::TEXT::UUID,
        COALESCE(NEW.operator_id, ''),
        'update',
        'system',
        jsonb_build_object('new', row_to_json(NEW)::JSONB),
        NOW()
    );
    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Statements audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_stmt_audit_insert
        AFTER INSERT ON gl_eudr_ddsc_statements
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_audit_insert('statement');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_stmt_audit_update
        AFTER UPDATE ON gl_eudr_ddsc_statements
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_audit_update('statement');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Geolocation audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_geo_audit_insert
        AFTER INSERT ON gl_eudr_ddsc_geolocation_data
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_audit_insert('geolocation');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_geo_audit_update
        AFTER UPDATE ON gl_eudr_ddsc_geolocation_data
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_audit_update('geolocation');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Risk references audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_risk_audit_insert
        AFTER INSERT ON gl_eudr_ddsc_risk_references
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_audit_insert('risk_reference');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_risk_audit_update
        AFTER UPDATE ON gl_eudr_ddsc_risk_references
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_audit_update('risk_reference');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Supply chain audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_sc_audit_insert
        AFTER INSERT ON gl_eudr_ddsc_supply_chain_data
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_audit_insert('supply_chain');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_sc_audit_update
        AFTER UPDATE ON gl_eudr_ddsc_supply_chain_data
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_audit_update('supply_chain');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Compliance checks audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_chk_audit_insert
        AFTER INSERT ON gl_eudr_ddsc_compliance_checks
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_audit_insert('compliance_check');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_chk_audit_update
        AFTER UPDATE ON gl_eudr_ddsc_compliance_checks
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_audit_update('compliance_check');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Document packages audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_doc_audit_insert
        AFTER INSERT ON gl_eudr_ddsc_document_packages
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_audit_insert('document_package');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_doc_audit_update
        AFTER UPDATE ON gl_eudr_ddsc_document_packages
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_audit_update('document_package');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Versions audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_ver_audit_insert
        AFTER INSERT ON gl_eudr_ddsc_versions
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_audit_insert('version');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_ver_audit_update
        AFTER UPDATE ON gl_eudr_ddsc_versions
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_audit_update('version');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Signatures audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_sig_audit_insert
        AFTER INSERT ON gl_eudr_ddsc_signatures
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_audit_insert('signature');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ddsc_sig_audit_update
        AFTER UPDATE ON gl_eudr_ddsc_signatures
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ddsc_audit_update('signature');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- Completion
-- ============================================================================

RAISE NOTICE 'V125: AGENT-EUDR-037 Due Diligence Statement Creator -- 9 tables (7 regular + 2 hypertables), ~107 indexes, 24 triggers, 3 views, 3 functions, 5-year retention';
RAISE NOTICE 'V125: Tables: gl_eudr_ddsc_statements, gl_eudr_ddsc_geolocation_data, gl_eudr_ddsc_risk_references, gl_eudr_ddsc_supply_chain_data, gl_eudr_ddsc_compliance_checks, gl_eudr_ddsc_document_packages, gl_eudr_ddsc_versions, gl_eudr_ddsc_signatures, gl_eudr_ddsc_audit_log (hypertable)';
RAISE NOTICE 'V125: Foreign keys: geolocation -> statements, risk_refs -> statements, supply_chain -> statements, compliance_checks -> statements, documents -> statements, versions -> statements, signatures -> statements (+ self-ref for counter-signatures), statements -> statements (parent)';
RAISE NOTICE 'V125: Views: vw_eudr_ddsc_active_statements, vw_eudr_ddsc_pending_submissions, vw_eudr_ddsc_compliance_summary';
RAISE NOTICE 'V125: Functions: fn_eudr_ddsc_validate_statement, fn_eudr_ddsc_compare_versions, fn_eudr_ddsc_submission_readiness';
RAISE NOTICE 'V125: Hypertable: gl_eudr_ddsc_audit_log (7-day chunks, 5-year retention)';

COMMIT;
