-- ============================================================================
-- V118: AGENT-EUDR-030 Documentation Generator
-- ============================================================================
-- Creates tables for the Documentation Generator which produces Due Diligence
-- Statement (DDS) documents per EUDR Article 4, assembles Article 9 information
-- packages with all required data elements, generates risk assessment and
-- mitigation documentation, builds complete compliance packages for authority
-- submission, tracks document versions with content hashing, records EU
-- Information System submission lifecycle, validates documents against EUDR
-- schema requirements, and preserves a complete Article 31 audit trail via
-- TimescaleDB hypertable.
--
-- Agent ID: GL-EUDR-DGN-030
-- PRD: PRD-AGENT-EUDR-030
-- Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 10, 11, 12, 14-16, 29, 31
-- Tables: 9 (8 regular + 1 hypertable)
-- Indexes: ~120
--
-- Dependencies: TimescaleDB extension (for eudr_dgn_audit_log hypertable)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V118: Creating AGENT-EUDR-030 Documentation Generator tables...';


-- ============================================================================
-- 1. eudr_dgn_dds_documents -- Due Diligence Statement records
-- ============================================================================
RAISE NOTICE 'V118 [1/9]: Creating eudr_dgn_dds_documents...';

CREATE TABLE IF NOT EXISTS eudr_dgn_dds_documents (
    dds_id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Internal primary key for DDS record
    reference_number                VARCHAR(100)    UNIQUE NOT NULL,
        -- Unique DDS reference number (e.g. "DDS-2026-03-001")
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator filing the Due Diligence Statement
    commodity                       VARCHAR(50)     NOT NULL,
        -- EUDR commodity covered by this DDS (cattle, cocoa, coffee, oil_palm, rubber, soya, wood)
    products                        JSONB,
        -- Array of product descriptions placed on or exported from the EU market
    article9_package_id             UUID,
        -- Reference to the assembled Article 9 information package
    risk_assessment_doc_id          UUID,
        -- Reference to the generated risk assessment documentation
    mitigation_doc_id               UUID,
        -- Reference to the generated mitigation documentation
    status                          VARCHAR(30)     DEFAULT 'draft',
        -- DDS lifecycle status
    compliance_conclusion           VARCHAR(50),
        -- Overall compliance conclusion reached by the operator
    dds_content                     JSONB,
        -- Full structured DDS content matching EU Information System schema
    schema_version                  VARCHAR(20),
        -- DDS schema version used for generation (e.g. "1.0", "2.0")
    generated_at                    TIMESTAMPTZ     DEFAULT NOW(),
        -- Timestamp when the DDS was generated
    submitted_at                    TIMESTAMPTZ,
        -- Timestamp when the DDS was submitted to EU Information System
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for DDS integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_dgn_dds_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_dgn_dds_status CHECK (status IN (
        'draft', 'in_review', 'approved', 'submitted', 'acknowledged',
        'rejected', 'amended', 'superseded', 'archived'
    )),
    CONSTRAINT chk_dgn_dds_conclusion CHECK (compliance_conclusion IS NULL OR compliance_conclusion IN (
        'compliant', 'compliant_with_conditions', 'non_compliant',
        'negligible_risk', 'standard_risk', 'pending_assessment'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_reference ON eudr_dgn_dds_documents (reference_number);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_operator ON eudr_dgn_dds_documents (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_commodity ON eudr_dgn_dds_documents (commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_article9 ON eudr_dgn_dds_documents (article9_package_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_risk_doc ON eudr_dgn_dds_documents (risk_assessment_doc_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_mitigation_doc ON eudr_dgn_dds_documents (mitigation_doc_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_status ON eudr_dgn_dds_documents (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_conclusion ON eudr_dgn_dds_documents (compliance_conclusion);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_schema_version ON eudr_dgn_dds_documents (schema_version);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_generated ON eudr_dgn_dds_documents (generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_submitted ON eudr_dgn_dds_documents (submitted_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_provenance ON eudr_dgn_dds_documents (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_created ON eudr_dgn_dds_documents (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_updated ON eudr_dgn_dds_documents (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_operator_commodity ON eudr_dgn_dds_documents (operator_id, commodity, generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_commodity_status ON eudr_dgn_dds_documents (commodity, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_operator_status ON eudr_dgn_dds_documents (operator_id, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_status_conclusion ON eudr_dgn_dds_documents (status, compliance_conclusion);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active (non-terminal) DDS documents
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_active ON eudr_dgn_dds_documents (generated_at DESC, operator_id)
        WHERE status NOT IN ('archived', 'superseded');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for draft DDS requiring completion
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_drafts ON eudr_dgn_dds_documents (created_at DESC, operator_id)
        WHERE status = 'draft';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for submitted DDS awaiting response
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_submitted_pending ON eudr_dgn_dds_documents (submitted_at DESC, operator_id)
        WHERE status = 'submitted';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for rejected DDS requiring amendment
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_rejected ON eudr_dgn_dds_documents (updated_at DESC, operator_id)
        WHERE status = 'rejected';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_products ON eudr_dgn_dds_documents USING GIN (products);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dds_content ON eudr_dgn_dds_documents USING GIN (dds_content);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_dgn_dds_documents IS 'AGENT-EUDR-030: Due Diligence Statement (DDS) records per EUDR Article 4, linking Article 9 information packages, risk assessments, and mitigation documentation with lifecycle tracking for EU Information System submission';
COMMENT ON COLUMN eudr_dgn_dds_documents.reference_number IS 'Unique DDS reference number for EU IS submission tracking (e.g. "DDS-2026-03-001")';
COMMENT ON COLUMN eudr_dgn_dds_documents.compliance_conclusion IS 'Operator compliance conclusion: compliant, compliant_with_conditions, non_compliant, negligible_risk, standard_risk, pending_assessment';


-- ============================================================================
-- 2. eudr_dgn_article9_packages -- Article 9 information packages
-- ============================================================================
RAISE NOTICE 'V118 [2/9]: Creating eudr_dgn_article9_packages...';

CREATE TABLE IF NOT EXISTS eudr_dgn_article9_packages (
    package_id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Internal primary key for Article 9 package
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator assembling the information package
    commodity                       VARCHAR(50)     NOT NULL,
        -- EUDR commodity for this package
    elements                        JSONB           NOT NULL DEFAULT '{}',
        -- Object of Article 9 required elements (description, HS codes, quantities, etc.)
    completeness_score              NUMERIC(5,4),
        -- Completeness score: 0.0000 (empty) to 1.0000 (all elements present)
    missing_elements                JSONB           DEFAULT '[]',
        -- Array of Article 9 elements that are still missing
    geolocations                    JSONB           DEFAULT '[]',
        -- Array of geolocation data per Article 9(1)(d)
    suppliers                       JSONB           DEFAULT '[]',
        -- Array of supplier information per Article 9(1)(e)
    products                        JSONB           DEFAULT '[]',
        -- Array of product identifiers and descriptions
    production_date_start           TIMESTAMPTZ,
        -- Start of production date range per Article 9(1)(f)
    production_date_end             TIMESTAMPTZ,
        -- End of production date range per Article 9(1)(f)
    certifications                  JSONB           DEFAULT '[]',
        -- Array of relevant certifications and standards
    assembled_at                    TIMESTAMPTZ     DEFAULT NOW(),
        -- Timestamp when the package was assembled
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for package integrity verification

    CONSTRAINT chk_dgn_a9_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_a9_operator ON eudr_dgn_article9_packages (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_a9_commodity ON eudr_dgn_article9_packages (commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_a9_completeness ON eudr_dgn_article9_packages (completeness_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_a9_prod_start ON eudr_dgn_article9_packages (production_date_start);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_a9_prod_end ON eudr_dgn_article9_packages (production_date_end);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_a9_assembled ON eudr_dgn_article9_packages (assembled_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_a9_provenance ON eudr_dgn_article9_packages (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_a9_operator_commodity ON eudr_dgn_article9_packages (operator_id, commodity, assembled_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_a9_commodity_completeness ON eudr_dgn_article9_packages (commodity, completeness_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for incomplete packages requiring attention
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_a9_incomplete ON eudr_dgn_article9_packages (completeness_score, operator_id)
        WHERE completeness_score < 1.0000;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_a9_elements ON eudr_dgn_article9_packages USING GIN (elements);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_a9_missing ON eudr_dgn_article9_packages USING GIN (missing_elements);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_a9_geolocations ON eudr_dgn_article9_packages USING GIN (geolocations);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_a9_suppliers ON eudr_dgn_article9_packages USING GIN (suppliers);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_a9_products ON eudr_dgn_article9_packages USING GIN (products);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_a9_certifications ON eudr_dgn_article9_packages USING GIN (certifications);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_dgn_article9_packages IS 'AGENT-EUDR-030: Article 9 information packages assembling all required data elements (geolocation, supplier, product, production dates, certifications) with completeness tracking per EUDR Article 9';
COMMENT ON COLUMN eudr_dgn_article9_packages.completeness_score IS 'Package completeness: 0.0000 (empty) to 1.0000 (all Article 9 elements present and validated)';
COMMENT ON COLUMN eudr_dgn_article9_packages.elements IS 'Object of Article 9 elements: {"description": "...", "hs_codes": [...], "quantity": {...}, "country_of_production": "...", ...}';


-- ============================================================================
-- 3. eudr_dgn_risk_assessment_docs -- Risk assessment documentation
-- ============================================================================
RAISE NOTICE 'V118 [3/9]: Creating eudr_dgn_risk_assessment_docs...';

CREATE TABLE IF NOT EXISTS eudr_dgn_risk_assessment_docs (
    doc_id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Internal primary key for risk assessment document
    assessment_id                   VARCHAR(100)    NOT NULL,
        -- Reference to the upstream risk assessment that produced this documentation
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator for this risk assessment documentation
    commodity                       VARCHAR(50)     NOT NULL,
        -- EUDR commodity assessed
    composite_score                 NUMERIC(5,2),
        -- Overall composite risk score from the assessment (0.00 to 100.00)
    risk_level                      VARCHAR(20),
        -- Classified risk level from the composite score
    criterion_evaluations           JSONB           DEFAULT '[]',
        -- Array of individual risk criterion evaluations with scores and evidence
    country_benchmark               VARCHAR(20),
        -- Country benchmarking status per EUDR Article 29
    simplified_dd_eligible          BOOLEAN         DEFAULT FALSE,
        -- Whether the operator qualifies for simplified due diligence per Article 13
    risk_dimensions                 JSONB           DEFAULT '{}',
        -- Object of per-dimension risk scores (country, supplier, commodity, etc.)
    generated_at                    TIMESTAMPTZ     DEFAULT NOW(),
        -- Timestamp when the risk assessment document was generated
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for document integrity verification

    CONSTRAINT chk_dgn_rad_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_dgn_rad_risk_level CHECK (risk_level IS NULL OR risk_level IN (
        'negligible', 'low', 'standard', 'elevated', 'high', 'critical'
    )),
    CONSTRAINT chk_dgn_rad_benchmark CHECK (country_benchmark IS NULL OR country_benchmark IN (
        'low_risk', 'standard_risk', 'high_risk', 'not_benchmarked'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_rad_assessment ON eudr_dgn_risk_assessment_docs (assessment_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_rad_operator ON eudr_dgn_risk_assessment_docs (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_rad_commodity ON eudr_dgn_risk_assessment_docs (commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_rad_score ON eudr_dgn_risk_assessment_docs (composite_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_rad_risk_level ON eudr_dgn_risk_assessment_docs (risk_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_rad_benchmark ON eudr_dgn_risk_assessment_docs (country_benchmark);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_rad_simplified ON eudr_dgn_risk_assessment_docs (simplified_dd_eligible);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_rad_generated ON eudr_dgn_risk_assessment_docs (generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_rad_provenance ON eudr_dgn_risk_assessment_docs (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_rad_operator_commodity ON eudr_dgn_risk_assessment_docs (operator_id, commodity, generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_rad_commodity_risk ON eudr_dgn_risk_assessment_docs (commodity, risk_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_rad_risk_score ON eudr_dgn_risk_assessment_docs (risk_level, composite_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for high/critical risk assessments requiring mitigation
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_rad_high_risk ON eudr_dgn_risk_assessment_docs (composite_score DESC, operator_id)
        WHERE risk_level IN ('high', 'critical');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for simplified DD eligible assessments
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_rad_simplified_eligible ON eudr_dgn_risk_assessment_docs (generated_at DESC, operator_id)
        WHERE simplified_dd_eligible = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_rad_criteria ON eudr_dgn_risk_assessment_docs USING GIN (criterion_evaluations);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_rad_dimensions ON eudr_dgn_risk_assessment_docs USING GIN (risk_dimensions);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_dgn_risk_assessment_docs IS 'AGENT-EUDR-030: Risk assessment documentation capturing composite scores, criterion evaluations, country benchmarking, and dimension breakdowns for EUDR Articles 10, 29 compliance';
COMMENT ON COLUMN eudr_dgn_risk_assessment_docs.composite_score IS 'Overall composite risk score: 0.00 (negligible) to 100.00 (critical), sourced from upstream risk assessment agents';
COMMENT ON COLUMN eudr_dgn_risk_assessment_docs.country_benchmark IS 'Article 29 country benchmarking: low_risk (simplified DD), standard_risk (full DD), high_risk (enhanced scrutiny), not_benchmarked';


-- ============================================================================
-- 4. eudr_dgn_mitigation_docs -- Mitigation documentation
-- ============================================================================
RAISE NOTICE 'V118 [4/9]: Creating eudr_dgn_mitigation_docs...';

CREATE TABLE IF NOT EXISTS eudr_dgn_mitigation_docs (
    doc_id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Internal primary key for mitigation document
    strategy_id                     VARCHAR(100)    NOT NULL,
        -- Reference to the upstream mitigation strategy this documentation covers
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator for this mitigation documentation
    commodity                       VARCHAR(50)     NOT NULL,
        -- EUDR commodity mitigated
    pre_mitigation_score            NUMERIC(5,2),
        -- Composite risk score before mitigation (0.00 to 100.00)
    post_mitigation_score           NUMERIC(5,2),
        -- Composite risk score after mitigation (0.00 to 100.00)
    measures_summary                JSONB           DEFAULT '[]',
        -- Array of measure summaries with Article 11 categories and outcomes
    verification_result             VARCHAR(30),
        -- Overall verification outcome from mitigation measures
    generated_at                    TIMESTAMPTZ     DEFAULT NOW(),
        -- Timestamp when the mitigation document was generated
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for document integrity verification

    CONSTRAINT chk_dgn_md_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_dgn_md_verification CHECK (verification_result IS NULL OR verification_result IN (
        'effective', 'partially_effective', 'ineffective',
        'inconclusive', 'pending_verification', 'not_required'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_md_strategy ON eudr_dgn_mitigation_docs (strategy_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_md_operator ON eudr_dgn_mitigation_docs (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_md_commodity ON eudr_dgn_mitigation_docs (commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_md_pre_score ON eudr_dgn_mitigation_docs (pre_mitigation_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_md_post_score ON eudr_dgn_mitigation_docs (post_mitigation_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_md_verification ON eudr_dgn_mitigation_docs (verification_result);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_md_generated ON eudr_dgn_mitigation_docs (generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_md_provenance ON eudr_dgn_mitigation_docs (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_md_operator_commodity ON eudr_dgn_mitigation_docs (operator_id, commodity, generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_md_commodity_verification ON eudr_dgn_mitigation_docs (commodity, verification_result);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for ineffective mitigations requiring further action
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_md_ineffective ON eudr_dgn_mitigation_docs (generated_at DESC, operator_id)
        WHERE verification_result IN ('ineffective', 'inconclusive');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_md_measures ON eudr_dgn_mitigation_docs USING GIN (measures_summary);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_dgn_mitigation_docs IS 'AGENT-EUDR-030: Mitigation documentation capturing pre/post risk scores, Article 11 measure summaries, and verification outcomes for DDS inclusion per EUDR Articles 10-11';
COMMENT ON COLUMN eudr_dgn_mitigation_docs.measures_summary IS 'Array of measure summaries: [{"measure_id": "...", "title": "...", "article11_category": "...", "status": "...", "risk_reduction": 0.15}, ...]';
COMMENT ON COLUMN eudr_dgn_mitigation_docs.verification_result IS 'Verification outcome: effective, partially_effective, ineffective, inconclusive, pending_verification, not_required';


-- ============================================================================
-- 5. eudr_dgn_compliance_packages -- Complete compliance audit packages
-- ============================================================================
RAISE NOTICE 'V118 [5/9]: Creating eudr_dgn_compliance_packages...';

CREATE TABLE IF NOT EXISTS eudr_dgn_compliance_packages (
    package_id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Internal primary key for compliance package
    dds_id                          UUID            NOT NULL,
        -- Reference to the parent DDS document
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator for this compliance package
    commodity                       VARCHAR(50)     NOT NULL,
        -- EUDR commodity covered by this package
    sections                        JSONB           NOT NULL DEFAULT '{}',
        -- Object of compliance package sections (executive_summary, article9_data, risk_assessment, mitigation, supporting_evidence, etc.)
    table_of_contents               JSONB           DEFAULT '[]',
        -- Array of table-of-contents entries for navigating the package
    cross_references                JSONB           DEFAULT '{}',
        -- Object mapping cross-references between sections and source documents
    generated_at                    TIMESTAMPTZ     DEFAULT NOW(),
        -- Timestamp when the compliance package was built
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for package integrity verification

    CONSTRAINT fk_dgn_cp_dds FOREIGN KEY (dds_id)
        REFERENCES eudr_dgn_dds_documents (dds_id),
    CONSTRAINT chk_dgn_cp_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_cp_dds ON eudr_dgn_compliance_packages (dds_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_cp_operator ON eudr_dgn_compliance_packages (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_cp_commodity ON eudr_dgn_compliance_packages (commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_cp_generated ON eudr_dgn_compliance_packages (generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_cp_provenance ON eudr_dgn_compliance_packages (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_cp_operator_commodity ON eudr_dgn_compliance_packages (operator_id, commodity, generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_cp_commodity_time ON eudr_dgn_compliance_packages (commodity, generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_cp_sections ON eudr_dgn_compliance_packages USING GIN (sections);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_cp_toc ON eudr_dgn_compliance_packages USING GIN (table_of_contents);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_cp_xrefs ON eudr_dgn_compliance_packages USING GIN (cross_references);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_dgn_compliance_packages IS 'AGENT-EUDR-030: Complete compliance audit packages assembling DDS, Article 9 data, risk assessment, mitigation documentation, and supporting evidence with table-of-contents and cross-references for authority review';
COMMENT ON COLUMN eudr_dgn_compliance_packages.sections IS 'Package sections: {"executive_summary": {...}, "article9_data": {...}, "risk_assessment": {...}, "mitigation": {...}, "supporting_evidence": {...}, "appendices": {...}}';
COMMENT ON COLUMN eudr_dgn_compliance_packages.cross_references IS 'Cross-reference mapping: {"risk_assessment.criterion_1": "article9.geolocation", "mitigation.measure_1": "evidence.doc_abc", ...}';


-- ============================================================================
-- 6. eudr_dgn_document_versions -- Document version tracking
-- ============================================================================
RAISE NOTICE 'V118 [6/9]: Creating eudr_dgn_document_versions...';

CREATE TABLE IF NOT EXISTS eudr_dgn_document_versions (
    version_id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Internal primary key for version record
    document_id                     VARCHAR(100)    NOT NULL,
        -- Identifier of the versioned document (DDS reference, package ID, etc.)
    document_type                   VARCHAR(50)     NOT NULL,
        -- Type of document being versioned
    version_number                  INTEGER         NOT NULL DEFAULT 1,
        -- Sequential version number (1, 2, 3, ...)
    status                          VARCHAR(30)     DEFAULT 'draft',
        -- Version lifecycle status
    content_hash                    VARCHAR(64)     NOT NULL,
        -- SHA-256 hash of the document content for change detection
    created_by                      VARCHAR(100),
        -- User or system actor that created this version
    amendment_reason                TEXT,
        -- Reason for creating this version (NULL for initial version)
    superseded_by                   UUID,
        -- Reference to the version that supersedes this one (NULL if current)
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_dgn_dv_type CHECK (document_type IN (
        'dds_document', 'article9_package', 'risk_assessment_doc',
        'mitigation_doc', 'compliance_package', 'submission_record'
    )),
    CONSTRAINT chk_dgn_dv_status CHECK (status IN (
        'draft', 'in_review', 'approved', 'published',
        'superseded', 'archived', 'withdrawn'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dv_document ON eudr_dgn_document_versions (document_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dv_type ON eudr_dgn_document_versions (document_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dv_version ON eudr_dgn_document_versions (document_id, version_number DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dv_status ON eudr_dgn_document_versions (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dv_content_hash ON eudr_dgn_document_versions (content_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dv_created_by ON eudr_dgn_document_versions (created_by);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dv_superseded ON eudr_dgn_document_versions (superseded_by);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dv_created ON eudr_dgn_document_versions (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dv_doc_type_status ON eudr_dgn_document_versions (document_id, document_type, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dv_type_status ON eudr_dgn_document_versions (document_type, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dv_type_created ON eudr_dgn_document_versions (document_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for current (non-superseded) versions
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dv_current ON eudr_dgn_document_versions (document_id, version_number DESC)
        WHERE superseded_by IS NULL AND status NOT IN ('superseded', 'archived', 'withdrawn');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for draft versions requiring review
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_dv_drafts ON eudr_dgn_document_versions (created_at DESC, document_type)
        WHERE status = 'draft';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Unique constraint: one active version per document+version number
DO $$ BEGIN
    CREATE UNIQUE INDEX idx_eudr_dgn_dv_unique_version ON eudr_dgn_document_versions (document_id, version_number);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_dgn_document_versions IS 'AGENT-EUDR-030: Document version tracking with content hashing for change detection, amendment history, and supersession chains per Article 31 audit requirements';
COMMENT ON COLUMN eudr_dgn_document_versions.content_hash IS 'SHA-256 hash of document content: used for change detection and integrity verification between versions';
COMMENT ON COLUMN eudr_dgn_document_versions.version_number IS 'Sequential version number within a document: initial version is 1, each amendment increments by 1';


-- ============================================================================
-- 7. eudr_dgn_submission_records -- EU Information System submission tracking
-- ============================================================================
RAISE NOTICE 'V118 [7/9]: Creating eudr_dgn_submission_records...';

CREATE TABLE IF NOT EXISTS eudr_dgn_submission_records (
    submission_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Internal primary key for submission record
    dds_id                          UUID            NOT NULL,
        -- Reference to the DDS document being submitted
    status                          VARCHAR(30)     DEFAULT 'pending',
        -- Submission lifecycle status
    submitted_at                    TIMESTAMPTZ,
        -- Timestamp when the submission was sent to EU IS
    acknowledged_at                 TIMESTAMPTZ,
        -- Timestamp when EU IS acknowledged receipt
    rejected_at                     TIMESTAMPTZ,
        -- Timestamp when EU IS rejected the submission
    rejection_reason                TEXT,
        -- Reason provided by EU IS for rejection
    receipt_number                  VARCHAR(100),
        -- EU IS receipt number assigned upon acknowledgement
    resubmission_count              INTEGER         DEFAULT 0,
        -- Number of times this DDS has been resubmitted after rejection
    response_data                   JSONB           DEFAULT '{}',
        -- Full response data from EU Information System
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_dgn_sr_dds FOREIGN KEY (dds_id)
        REFERENCES eudr_dgn_dds_documents (dds_id),
    CONSTRAINT chk_dgn_sr_status CHECK (status IN (
        'pending', 'submitting', 'submitted', 'acknowledged',
        'rejected', 'resubmitting', 'failed', 'cancelled'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_sr_dds ON eudr_dgn_submission_records (dds_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_sr_status ON eudr_dgn_submission_records (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_sr_submitted ON eudr_dgn_submission_records (submitted_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_sr_acknowledged ON eudr_dgn_submission_records (acknowledged_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_sr_rejected ON eudr_dgn_submission_records (rejected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_sr_receipt ON eudr_dgn_submission_records (receipt_number);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_sr_resubmission ON eudr_dgn_submission_records (resubmission_count DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_sr_created ON eudr_dgn_submission_records (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_sr_updated ON eudr_dgn_submission_records (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_sr_dds_status ON eudr_dgn_submission_records (dds_id, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_sr_status_time ON eudr_dgn_submission_records (status, submitted_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for pending submissions awaiting processing
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_sr_pending ON eudr_dgn_submission_records (created_at DESC, dds_id)
        WHERE status = 'pending';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for rejected submissions requiring resubmission
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_sr_rejected_active ON eudr_dgn_submission_records (rejected_at DESC, dds_id)
        WHERE status = 'rejected';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for submissions in transit (submitted but not yet acknowledged)
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_sr_in_transit ON eudr_dgn_submission_records (submitted_at DESC, dds_id)
        WHERE status = 'submitted';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_sr_response ON eudr_dgn_submission_records USING GIN (response_data);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_dgn_submission_records IS 'AGENT-EUDR-030: EU Information System submission lifecycle tracking with receipt numbers, rejection handling, resubmission counts, and full response data per EUDR Article 4 submission requirements';
COMMENT ON COLUMN eudr_dgn_submission_records.receipt_number IS 'EU IS receipt number: assigned upon successful acknowledgement, used for reference in compliance correspondence';
COMMENT ON COLUMN eudr_dgn_submission_records.resubmission_count IS 'Number of resubmissions: 0 for first submission, increments on each rejection+resubmission cycle';


-- ============================================================================
-- 8. eudr_dgn_validation_results -- Document validation records
-- ============================================================================
RAISE NOTICE 'V118 [8/9]: Creating eudr_dgn_validation_results...';

CREATE TABLE IF NOT EXISTS eudr_dgn_validation_results (
    validation_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Internal primary key for validation result
    document_id                     VARCHAR(100)    NOT NULL,
        -- Identifier of the document that was validated
    document_type                   VARCHAR(50)     NOT NULL,
        -- Type of document validated
    is_valid                        BOOLEAN         NOT NULL,
        -- Whether the document passed all validation rules
    errors                          JSONB           DEFAULT '[]',
        -- Array of validation errors (blocking issues)
    warnings                        JSONB           DEFAULT '[]',
        -- Array of validation warnings (non-blocking issues)
    info_items                      JSONB           DEFAULT '[]',
        -- Array of informational items (suggestions, best practices)
    validated_at                    TIMESTAMPTZ     DEFAULT NOW(),
        -- Timestamp when validation was performed

    CONSTRAINT chk_dgn_vr_type CHECK (document_type IN (
        'dds_document', 'article9_package', 'risk_assessment_doc',
        'mitigation_doc', 'compliance_package', 'submission_payload'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_vr_document ON eudr_dgn_validation_results (document_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_vr_type ON eudr_dgn_validation_results (document_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_vr_valid ON eudr_dgn_validation_results (is_valid);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_vr_validated ON eudr_dgn_validation_results (validated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_vr_doc_type ON eudr_dgn_validation_results (document_id, document_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_vr_type_valid ON eudr_dgn_validation_results (document_type, is_valid);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_vr_doc_time ON eudr_dgn_validation_results (document_id, validated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for failed validations requiring correction
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_vr_failed ON eudr_dgn_validation_results (validated_at DESC, document_type)
        WHERE is_valid = FALSE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_vr_errors ON eudr_dgn_validation_results USING GIN (errors);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_vr_warnings ON eudr_dgn_validation_results USING GIN (warnings);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_vr_info ON eudr_dgn_validation_results USING GIN (info_items);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_dgn_validation_results IS 'AGENT-EUDR-030: Document validation records capturing errors, warnings, and informational items against EUDR schema requirements for pre-submission quality assurance';
COMMENT ON COLUMN eudr_dgn_validation_results.is_valid IS 'Validation result: TRUE if zero errors (warnings allowed), FALSE if any blocking errors found';
COMMENT ON COLUMN eudr_dgn_validation_results.errors IS 'Array of blocking errors: [{"code": "ERR-001", "field": "geolocation", "message": "Missing required polygon coordinates", "article": "9(1)(d)"}, ...]';


-- ============================================================================
-- 9. eudr_dgn_audit_log -- Immutable audit trail (TimescaleDB hypertable)
-- ============================================================================
RAISE NOTICE 'V118 [9/9]: Creating eudr_dgn_audit_log (hypertable)...';

CREATE TABLE IF NOT EXISTS eudr_dgn_audit_log (
    log_id                          UUID            DEFAULT gen_random_uuid(),
        -- Unique audit log entry identifier
    entity_type                     VARCHAR(50)     NOT NULL,
        -- Type of entity affected by the operation
    entity_id                       VARCHAR(100)    NOT NULL,
        -- Identifier of the entity affected
    action                          VARCHAR(50)     NOT NULL,
        -- Action that was performed on the entity
    actor                           VARCHAR(100)    NOT NULL DEFAULT 'gl-eudr-dgn-030',
        -- Actor who performed the action (system agent or user)
    old_state                       JSONB,
        -- Previous state of the entity (NULL for creation actions)
    new_state                       JSONB,
        -- New state of the entity after the action
    metadata                        JSONB           DEFAULT '{}',
        -- Additional context (IP address, session ID, correlation ID, etc.)
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash chained to previous entry for tamper-evident audit trail
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_dgn_al_entity_type CHECK (entity_type IN (
        'dds_document', 'article9_package', 'risk_assessment_doc',
        'mitigation_doc', 'compliance_package', 'document_version',
        'submission_record', 'validation_result', 'configuration'
    )),
    CONSTRAINT chk_dgn_al_action CHECK (action IN (
        'dds_created', 'dds_updated', 'dds_approved', 'dds_submitted',
        'dds_acknowledged', 'dds_rejected', 'dds_amended', 'dds_superseded', 'dds_archived',
        'article9_assembled', 'article9_updated', 'article9_completed',
        'risk_doc_generated', 'risk_doc_updated',
        'mitigation_doc_generated', 'mitigation_doc_updated',
        'package_built', 'package_updated', 'package_finalized',
        'version_created', 'version_approved', 'version_published',
        'version_superseded', 'version_withdrawn',
        'submission_initiated', 'submission_sent', 'submission_acknowledged',
        'submission_rejected', 'submission_resubmitted', 'submission_failed', 'submission_cancelled',
        'validation_performed', 'validation_passed', 'validation_failed',
        'config_updated', 'manual_action'
    ))
);

-- Convert to TimescaleDB hypertable partitioned on created_at
SELECT create_hypertable('eudr_dgn_audit_log', 'created_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_al_entity_type ON eudr_dgn_audit_log (entity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_al_entity_id ON eudr_dgn_audit_log (entity_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_al_action ON eudr_dgn_audit_log (action);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_al_actor ON eudr_dgn_audit_log (actor);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_al_provenance ON eudr_dgn_audit_log (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_al_created ON eudr_dgn_audit_log (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_al_entity_action ON eudr_dgn_audit_log (entity_type, action, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_al_entity_id_time ON eudr_dgn_audit_log (entity_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_al_actor_time ON eudr_dgn_audit_log (actor, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_al_old_state ON eudr_dgn_audit_log USING GIN (old_state);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_al_new_state ON eudr_dgn_audit_log USING GIN (new_state);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_dgn_al_metadata ON eudr_dgn_audit_log USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- DATA RETENTION POLICY -- Article 31: 5-year retention for EUDR audit data
-- ============================================================================
RAISE NOTICE 'V118: Configuring 5-year data retention policy per EUDR Article 31...';

-- Audit log retention: 5 years (60 months) per Article 31 requirement
SELECT add_retention_policy('eudr_dgn_audit_log',
    INTERVAL '5 years',
    if_not_exists => TRUE
);


-- ============================================================================
-- TABLE COMMENTS
-- ============================================================================

COMMENT ON TABLE eudr_dgn_audit_log IS 'AGENT-EUDR-030: Article 31 compliant immutable audit trail (TimescaleDB hypertable, 1-month chunks) for all documentation generation, versioning, validation, and submission operations with 5-year retention';
COMMENT ON COLUMN eudr_dgn_audit_log.actor IS 'Default actor is gl-eudr-dgn-030 (system agent); overridden for manual user actions such as DDS approvals and document reviews';
COMMENT ON COLUMN eudr_dgn_audit_log.provenance_hash IS 'SHA-256 hash chained to previous entry for tamper-evident audit trail per EUDR Article 31';


-- ============================================================================
-- FINALIZE
-- ============================================================================

RAISE NOTICE 'V118: AGENT-EUDR-030 Documentation Generator tables created successfully!';
RAISE NOTICE 'V118: Created 9 tables (8 regular + 1 hypertable), ~120 indexes (B-tree, GIN, partial, unique)';
RAISE NOTICE 'V118: Foreign keys: compliance_packages -> dds_documents; submission_records -> dds_documents';
RAISE NOTICE 'V118: Hypertable: eudr_dgn_audit_log (1-month chunks on created_at, 5-year retention)';
RAISE NOTICE 'V118: Retention policy: 5 years per EUDR Article 31';

COMMIT;
