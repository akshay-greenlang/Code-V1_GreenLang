-- ============================================================================
-- V116: AGENT-EUDR-028 Risk Assessment Engine
-- ============================================================================
-- Creates tables for the Risk Assessment Engine which computes composite risk
-- scores from multi-dimensional factor inputs, evaluates Article 10(2) criteria,
-- applies Article 29 country benchmarking, classifies risk levels (negligible,
-- low, standard, elevated, high, critical), supports manual risk overrides with
-- audit justification, tracks risk trends over time via TimescaleDB hypertable,
-- generates risk assessment reports with DDS readiness flags, and maintains a
-- complete Article 31 audit trail.
--
-- Tables: 9 (8 regular + 1 hypertable)
-- Indexes: ~110
--
-- Dependencies: TimescaleDB extension (for eudr_rae_risk_trends hypertable)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V116: Creating AGENT-EUDR-028 Risk Assessment Engine tables...';


-- ============================================================================
-- 1. eudr_rae_risk_assessments -- Top-level risk assessment operations
-- ============================================================================
RAISE NOTICE 'V116 [1/9]: Creating eudr_rae_risk_assessments...';

CREATE TABLE IF NOT EXISTS eudr_rae_risk_assessments (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    operation_id                    VARCHAR(100)    UNIQUE NOT NULL,
        -- Unique identifier for this risk assessment operation (e.g. "rae-op-2026-03-001")
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator initiating the risk assessment
    commodity                       VARCHAR(50)     NOT NULL,
        -- EUDR commodity being assessed (cattle, cocoa, coffee, oil_palm, rubber, soya, wood)
    workflow_id                     VARCHAR(100),
        -- Reference to the due diligence workflow orchestrating this assessment
    status                          VARCHAR(50)     NOT NULL DEFAULT 'initiated',
        -- Assessment lifecycle status
    composite_score                 DECIMAL(10,4)   DEFAULT 0,
        -- Final weighted composite risk score (0.0000 to 1.0000)
    risk_level                      VARCHAR(20)     DEFAULT 'standard',
        -- Risk level classification derived from composite score
    country_codes                   JSONB           DEFAULT '[]',
        -- Array of ISO 3166 country codes involved in the assessment
    supplier_ids                    JSONB           DEFAULT '[]',
        -- Array of supplier identifiers included in the assessment
    factor_inputs                   JSONB           DEFAULT '[]',
        -- Array of risk factor inputs from upstream agents (EUDR-016 to EUDR-025)
    article10_result                JSONB,
        -- Structured Article 10(2) evaluation result summary
    simplified_dd_eligible          BOOLEAN         DEFAULT FALSE,
        -- TRUE if assessment qualifies for simplified due diligence per Article 13
    report_id                       VARCHAR(100),
        -- Reference to the generated risk assessment report
    started_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the risk assessment began
    completed_at                    TIMESTAMPTZ,
        -- Timestamp when the risk assessment completed (NULL if in-progress)
    duration_ms                     INTEGER,
        -- Total assessment duration in milliseconds
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for assessment integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_rae_ra_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_rae_ra_status CHECK (status IN (
        'initiated', 'collecting_factors', 'aggregating', 'evaluating_article10',
        'benchmarking', 'classifying', 'generating_report', 'completed', 'failed'
    )),
    CONSTRAINT chk_rae_ra_risk_level CHECK (risk_level IN (
        'negligible', 'low', 'standard', 'elevated', 'high', 'critical'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_operation ON eudr_rae_risk_assessments (operation_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_operator ON eudr_rae_risk_assessments (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_commodity ON eudr_rae_risk_assessments (commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_workflow ON eudr_rae_risk_assessments (workflow_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_status ON eudr_rae_risk_assessments (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_composite ON eudr_rae_risk_assessments (composite_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_risk_level ON eudr_rae_risk_assessments (risk_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_simplified ON eudr_rae_risk_assessments (simplified_dd_eligible);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_report ON eudr_rae_risk_assessments (report_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_started ON eudr_rae_risk_assessments (started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_completed ON eudr_rae_risk_assessments (completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_provenance ON eudr_rae_risk_assessments (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_created ON eudr_rae_risk_assessments (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_updated ON eudr_rae_risk_assessments (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_commodity_status ON eudr_rae_risk_assessments (commodity, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_operator_commodity ON eudr_rae_risk_assessments (operator_id, commodity, started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_risk_commodity ON eudr_rae_risk_assessments (risk_level, commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active (non-terminal) assessments
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_active ON eudr_rae_risk_assessments (started_at DESC)
        WHERE status NOT IN ('completed', 'failed');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for high/critical risk assessments requiring attention
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_high_risk ON eudr_rae_risk_assessments (composite_score DESC, operator_id)
        WHERE risk_level IN ('high', 'critical');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_country_codes ON eudr_rae_risk_assessments USING GIN (country_codes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_supplier_ids ON eudr_rae_risk_assessments USING GIN (supplier_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_factor_inputs ON eudr_rae_risk_assessments USING GIN (factor_inputs);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ra_article10 ON eudr_rae_risk_assessments USING GIN (article10_result);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_rae_risk_assessments IS 'AGENT-EUDR-028: Top-level risk assessment operations computing composite risk scores from multi-dimensional factor inputs for EUDR Article 10 due diligence';
COMMENT ON COLUMN eudr_rae_risk_assessments.composite_score IS 'Weighted composite risk score: 0.0000 (negligible risk) to 1.0000 (critical risk)';
COMMENT ON COLUMN eudr_rae_risk_assessments.risk_level IS 'Risk classification: negligible (<0.10), low (0.10-0.25), standard (0.25-0.50), elevated (0.50-0.70), high (0.70-0.85), critical (>=0.85)';


-- ============================================================================
-- 2. eudr_rae_composite_scores -- Computed composite risk scores
-- ============================================================================
RAISE NOTICE 'V116 [2/9]: Creating eudr_rae_composite_scores...';

CREATE TABLE IF NOT EXISTS eudr_rae_composite_scores (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id                   VARCHAR(100)    NOT NULL,
        -- Reference to the parent risk assessment operation
    overall_score                   DECIMAL(10,4)   NOT NULL,
        -- Final computed composite score (0.0000 to 1.0000)
    risk_level                      VARCHAR(20)     NOT NULL,
        -- Risk level classification derived from overall_score
    dimension_scores                JSONB           NOT NULL DEFAULT '[]',
        -- Array of individual dimension score objects with weights and sources
    total_weight                    DECIMAL(10,4),
        -- Sum of all dimension weights (should equal 1.0000 for valid scoring)
    effective_confidence            DECIMAL(10,4),
        -- Weighted average confidence across all dimensions (0.0000 to 1.0000)
    country_benchmark_applied       BOOLEAN         DEFAULT FALSE,
        -- TRUE if Article 29 country benchmarking multiplier was applied
    benchmark_multiplier            DECIMAL(10,4)   DEFAULT 1.0,
        -- Country benchmark multiplier (< 1.0 for low-risk, > 1.0 for high-risk)
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for composite score integrity verification
    calculated_at                   TIMESTAMPTZ     DEFAULT NOW(),
        -- Timestamp when the composite score was calculated

    CONSTRAINT fk_rae_cs_assessment FOREIGN KEY (assessment_id)
        REFERENCES eudr_rae_risk_assessments (operation_id),
    CONSTRAINT chk_rae_cs_risk_level CHECK (risk_level IN (
        'negligible', 'low', 'standard', 'elevated', 'high', 'critical'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cs_assessment ON eudr_rae_composite_scores (assessment_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cs_overall ON eudr_rae_composite_scores (overall_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cs_risk_level ON eudr_rae_composite_scores (risk_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cs_confidence ON eudr_rae_composite_scores (effective_confidence DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cs_benchmark ON eudr_rae_composite_scores (country_benchmark_applied);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cs_multiplier ON eudr_rae_composite_scores (benchmark_multiplier);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cs_provenance ON eudr_rae_composite_scores (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cs_calculated ON eudr_rae_composite_scores (calculated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cs_assessment_calc ON eudr_rae_composite_scores (assessment_id, calculated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for high/critical scores requiring review
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cs_high_risk ON eudr_rae_composite_scores (overall_score DESC, assessment_id)
        WHERE risk_level IN ('high', 'critical');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cs_dimensions ON eudr_rae_composite_scores USING GIN (dimension_scores);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_rae_composite_scores IS 'AGENT-EUDR-028: Computed composite risk scores with dimension breakdown, confidence weighting, and Article 29 country benchmark adjustments';
COMMENT ON COLUMN eudr_rae_composite_scores.overall_score IS 'Final composite: 0.0000 (negligible) to 1.0000 (critical), computed as weighted sum of dimension scores with benchmark multiplier';
COMMENT ON COLUMN eudr_rae_composite_scores.benchmark_multiplier IS 'Article 29 multiplier: <1.0 for EC low-risk benchmarked countries, 1.0 for standard, >1.0 for high-risk';


-- ============================================================================
-- 3. eudr_rae_dimension_scores -- Individual dimension score breakdown
-- ============================================================================
RAISE NOTICE 'V116 [3/9]: Creating eudr_rae_dimension_scores...';

CREATE TABLE IF NOT EXISTS eudr_rae_dimension_scores (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    composite_score_id              UUID            NOT NULL,
        -- Reference to the parent composite score record
    dimension                       VARCHAR(50)     NOT NULL,
        -- Risk dimension being scored
    raw_score                       DECIMAL(10,4)   NOT NULL,
        -- Raw unweighted score for this dimension (0.0000 to 1.0000)
    weighted_score                  DECIMAL(10,4)   NOT NULL,
        -- Weighted score (raw_score * weight) contributing to composite
    weight                          DECIMAL(10,4)   NOT NULL,
        -- Weight assigned to this dimension (all weights should sum to 1.0000)
    confidence                      DECIMAL(10,4)   NOT NULL,
        -- Data confidence for this dimension (0.0000 to 1.0000)
    source_agent                    VARCHAR(50),
        -- Upstream agent that provided the factor data (e.g. "eudr-016", "eudr-017")
    explanation                     TEXT,
        -- Human-readable explanation of the score derivation
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_rae_ds_composite FOREIGN KEY (composite_score_id)
        REFERENCES eudr_rae_composite_scores (id),
    CONSTRAINT chk_rae_ds_dimension CHECK (dimension IN (
        'country_risk', 'supplier_risk', 'commodity_risk', 'corruption_risk',
        'deforestation_risk', 'indigenous_rights', 'protected_areas',
        'legal_compliance', 'traceability', 'certification'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ds_composite ON eudr_rae_dimension_scores (composite_score_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ds_dimension ON eudr_rae_dimension_scores (dimension);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ds_raw_score ON eudr_rae_dimension_scores (raw_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ds_weighted ON eudr_rae_dimension_scores (weighted_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ds_confidence ON eudr_rae_dimension_scores (confidence);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ds_source ON eudr_rae_dimension_scores (source_agent);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ds_created ON eudr_rae_dimension_scores (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ds_composite_dim ON eudr_rae_dimension_scores (composite_score_id, dimension);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ds_dim_source ON eudr_rae_dimension_scores (dimension, source_agent);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for high-risk dimensions (raw score >= 0.70) needing attention
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ds_high_dim ON eudr_rae_dimension_scores (raw_score DESC, dimension)
        WHERE raw_score >= 0.7000;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_rae_dimension_scores IS 'AGENT-EUDR-028: Individual risk dimension scores with weights, confidence levels, and source agent attribution for full composite score decomposition';
COMMENT ON COLUMN eudr_rae_dimension_scores.dimension IS 'Risk dimension: country_risk, supplier_risk, commodity_risk, corruption_risk, deforestation_risk, indigenous_rights, protected_areas, legal_compliance, traceability, certification';
COMMENT ON COLUMN eudr_rae_dimension_scores.source_agent IS 'Upstream EUDR agent ID that provided factor data: eudr-016 (Country), eudr-017 (Supplier), eudr-018 (Commodity), eudr-019 (Corruption), eudr-020 (Deforestation), eudr-021 (Indigenous), eudr-022 (Protected), eudr-023 (Legal)';


-- ============================================================================
-- 4. eudr_rae_article10_evaluations -- Article 10(2) criteria evaluations
-- ============================================================================
RAISE NOTICE 'V116 [4/9]: Creating eudr_rae_article10_evaluations...';

CREATE TABLE IF NOT EXISTS eudr_rae_article10_evaluations (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id                   VARCHAR(100)    NOT NULL,
        -- Reference to the parent risk assessment operation
    criterion                       VARCHAR(100)    NOT NULL,
        -- Article 10(2) criterion being evaluated
    result                          VARCHAR(20)     NOT NULL,
        -- Evaluation result for this criterion
    score                           DECIMAL(10,4)   DEFAULT 0,
        -- Numeric score for this criterion evaluation (0.0000 to 1.0000)
    evidence_summary                TEXT,
        -- Summary of evidence supporting the evaluation result
    data_sources                    JSONB           DEFAULT '[]',
        -- Array of data source identifiers that informed this evaluation
    evaluated_at                    TIMESTAMPTZ     DEFAULT NOW(),
        -- Timestamp when this criterion was evaluated

    CONSTRAINT fk_rae_a10_assessment FOREIGN KEY (assessment_id)
        REFERENCES eudr_rae_risk_assessments (operation_id),
    CONSTRAINT chk_rae_a10_result CHECK (result IN (
        'pass', 'concern', 'fail', 'not_evaluated'
    )),
    CONSTRAINT chk_rae_a10_criterion CHECK (criterion IN (
        'deforestation_free', 'legal_compliance', 'country_of_production',
        'geolocation_verified', 'quantity_verified', 'supplier_identified',
        'product_description', 'operator_information', 'risk_complexity',
        'country_benchmarking', 'third_party_certification', 'supply_chain_complexity',
        'conflict_affected_area', 'prevalence_of_deforestation', 'indigenous_rights_respect'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_a10_assessment ON eudr_rae_article10_evaluations (assessment_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_a10_criterion ON eudr_rae_article10_evaluations (criterion);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_a10_result ON eudr_rae_article10_evaluations (result);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_a10_score ON eudr_rae_article10_evaluations (score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_a10_evaluated ON eudr_rae_article10_evaluations (evaluated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_a10_assess_crit ON eudr_rae_article10_evaluations (assessment_id, criterion);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_a10_crit_result ON eudr_rae_article10_evaluations (criterion, result);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for failed or concern criteria requiring remediation
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_a10_failures ON eudr_rae_article10_evaluations (evaluated_at DESC, criterion)
        WHERE result IN ('fail', 'concern');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_a10_sources ON eudr_rae_article10_evaluations USING GIN (data_sources);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_rae_article10_evaluations IS 'AGENT-EUDR-028: Article 10(2) criteria evaluations assessing deforestation-free status, legal compliance, geolocation, and other EUDR due diligence requirements';
COMMENT ON COLUMN eudr_rae_article10_evaluations.criterion IS 'Article 10(2) criteria: deforestation_free, legal_compliance, country_of_production, geolocation_verified, quantity_verified, supplier_identified, etc.';
COMMENT ON COLUMN eudr_rae_article10_evaluations.result IS 'Evaluation outcome: pass (criterion met), concern (partial compliance), fail (criterion not met), not_evaluated (insufficient data)';


-- ============================================================================
-- 5. eudr_rae_country_benchmarks -- Article 29 country benchmarking data
-- ============================================================================
RAISE NOTICE 'V116 [5/9]: Creating eudr_rae_country_benchmarks...';

CREATE TABLE IF NOT EXISTS eudr_rae_country_benchmarks (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code                    VARCHAR(5)      NOT NULL,
        -- ISO 3166-1 alpha-2 or alpha-3 country code
    benchmark_level                 VARCHAR(20)     NOT NULL,
        -- Article 29 benchmark classification assigned by the European Commission
    effective_date                  DATE            NOT NULL,
        -- Date from which this benchmark classification is effective
    source                          VARCHAR(200),
        -- Source document or EU Official Journal reference for the benchmark
    governance_score                DECIMAL(10,4),
        -- Governance quality score (0.0000 to 1.0000) from World Bank WGI or similar
    deforestation_rate              DECIMAL(10,6),
        -- Annual deforestation rate (percentage) for the country
    confidence                      DECIMAL(10,4),
        -- Confidence in the benchmark data (0.0000 to 1.0000)
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_rae_cb_level CHECK (benchmark_level IN (
        'low', 'standard', 'high'
    )),
    CONSTRAINT uq_rae_cb_country_date UNIQUE (country_code, effective_date)
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cb_country ON eudr_rae_country_benchmarks (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cb_level ON eudr_rae_country_benchmarks (benchmark_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cb_effective ON eudr_rae_country_benchmarks (effective_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cb_governance ON eudr_rae_country_benchmarks (governance_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cb_deforestation ON eudr_rae_country_benchmarks (deforestation_rate DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cb_confidence ON eudr_rae_country_benchmarks (confidence DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cb_created ON eudr_rae_country_benchmarks (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cb_updated ON eudr_rae_country_benchmarks (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cb_country_level ON eudr_rae_country_benchmarks (country_code, benchmark_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cb_level_effective ON eudr_rae_country_benchmarks (benchmark_level, effective_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for high-risk benchmarked countries
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_cb_high_risk ON eudr_rae_country_benchmarks (country_code, effective_date DESC)
        WHERE benchmark_level = 'high';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_rae_country_benchmarks IS 'AGENT-EUDR-028: Article 29 country benchmarking data classifying countries as low/standard/high risk for EUDR compliance with governance and deforestation metrics';
COMMENT ON COLUMN eudr_rae_country_benchmarks.benchmark_level IS 'EC benchmark: low (simplified DD per Article 13), standard (full DD), high (enhanced DD per Article 10)';
COMMENT ON COLUMN eudr_rae_country_benchmarks.deforestation_rate IS 'Annual deforestation rate as a percentage (e.g. 0.001234 = 0.1234% per year)';


-- ============================================================================
-- 6. eudr_rae_risk_overrides -- Manual risk overrides
-- ============================================================================
RAISE NOTICE 'V116 [6/9]: Creating eudr_rae_risk_overrides...';

CREATE TABLE IF NOT EXISTS eudr_rae_risk_overrides (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    override_id                     VARCHAR(100)    UNIQUE NOT NULL,
        -- Unique identifier for this override (e.g. "rae-ovr-2026-03-001")
    assessment_id                   VARCHAR(100)    NOT NULL,
        -- Reference to the risk assessment being overridden
    original_score                  DECIMAL(10,4)   NOT NULL,
        -- Original composite score before override
    overridden_score                DECIMAL(10,4)   NOT NULL,
        -- New composite score after override
    original_level                  VARCHAR(20)     NOT NULL,
        -- Original risk level before override
    overridden_level                VARCHAR(20)     NOT NULL,
        -- New risk level after override
    reason                          VARCHAR(50)     NOT NULL,
        -- Reason category for the override
    justification                   TEXT            NOT NULL,
        -- Detailed justification narrative for the override (required for audit)
    overridden_by                   VARCHAR(100)    NOT NULL,
        -- User or role that initiated the override
    approved_by                     VARCHAR(100),
        -- User or role that approved the override (NULL if pending approval)
    valid_until                     TIMESTAMPTZ,
        -- Override expiration (NULL for indefinite; time-bound overrides auto-revert)
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_rae_ro_assessment FOREIGN KEY (assessment_id)
        REFERENCES eudr_rae_risk_assessments (operation_id),
    CONSTRAINT chk_rae_ro_original_level CHECK (original_level IN (
        'negligible', 'low', 'standard', 'elevated', 'high', 'critical'
    )),
    CONSTRAINT chk_rae_ro_overridden_level CHECK (overridden_level IN (
        'negligible', 'low', 'standard', 'elevated', 'high', 'critical'
    )),
    CONSTRAINT chk_rae_ro_reason CHECK (reason IN (
        'additional_evidence', 'expert_judgment', 'regulatory_guidance',
        'certification_update', 'supplier_remediation', 'data_correction',
        'temporary_exemption', 'escalation', 'de_escalation', 'other'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ro_override ON eudr_rae_risk_overrides (override_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ro_assessment ON eudr_rae_risk_overrides (assessment_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ro_original_score ON eudr_rae_risk_overrides (original_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ro_overridden_score ON eudr_rae_risk_overrides (overridden_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ro_original_level ON eudr_rae_risk_overrides (original_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ro_overridden_level ON eudr_rae_risk_overrides (overridden_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ro_reason ON eudr_rae_risk_overrides (reason);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ro_overridden_by ON eudr_rae_risk_overrides (overridden_by);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ro_approved_by ON eudr_rae_risk_overrides (approved_by);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ro_valid_until ON eudr_rae_risk_overrides (valid_until);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ro_created ON eudr_rae_risk_overrides (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ro_assess_reason ON eudr_rae_risk_overrides (assessment_id, reason);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for pending overrides awaiting approval
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ro_pending ON eudr_rae_risk_overrides (created_at DESC, assessment_id)
        WHERE approved_by IS NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for time-bound overrides approaching expiration
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_ro_expiring ON eudr_rae_risk_overrides (valid_until, assessment_id)
        WHERE valid_until IS NOT NULL AND valid_until <= (NOW() + INTERVAL '30 days');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_rae_risk_overrides IS 'AGENT-EUDR-028: Manual risk score overrides with audit-grade justification, dual-approval workflow, and optional time-bound validity for EUDR Article 10 compliance';
COMMENT ON COLUMN eudr_rae_risk_overrides.reason IS 'Override reason: additional_evidence, expert_judgment, regulatory_guidance, certification_update, supplier_remediation, data_correction, temporary_exemption, escalation, de_escalation, other';
COMMENT ON COLUMN eudr_rae_risk_overrides.valid_until IS 'Override expiration: NULL for indefinite override, timestamp for time-bound override that auto-reverts upon expiration';


-- ============================================================================
-- 7. eudr_rae_risk_trends -- Historical risk trend data (TimescaleDB hypertable)
-- ============================================================================
RAISE NOTICE 'V116 [7/9]: Creating eudr_rae_risk_trends (hypertable)...';

CREATE TABLE IF NOT EXISTS eudr_rae_risk_trends (
    id                              UUID            DEFAULT gen_random_uuid(),
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator for trend tracking
    commodity                       VARCHAR(50)     NOT NULL,
        -- EUDR commodity for trend tracking
    composite_score                 DECIMAL(10,4)   NOT NULL,
        -- Composite risk score at this point in time
    risk_level                      VARCHAR(20)     NOT NULL,
        -- Risk level classification at this point in time
    key_changes                     JSONB           DEFAULT '{}',
        -- JSON object describing key changes from previous assessment
    assessed_at                     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp of the assessment (hypertable partition key)

    CONSTRAINT chk_rae_rt_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_rae_rt_risk_level CHECK (risk_level IN (
        'negligible', 'low', 'standard', 'elevated', 'high', 'critical'
    ))
);

-- Convert to TimescaleDB hypertable partitioned on assessed_at
SELECT create_hypertable('eudr_rae_risk_trends', 'assessed_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rt_operator ON eudr_rae_risk_trends (operator_id, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rt_commodity ON eudr_rae_risk_trends (commodity, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rt_composite ON eudr_rae_risk_trends (composite_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rt_risk_level ON eudr_rae_risk_trends (risk_level, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rt_assessed ON eudr_rae_risk_trends (assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rt_op_commodity ON eudr_rae_risk_trends (operator_id, commodity, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rt_key_changes ON eudr_rae_risk_trends USING GIN (key_changes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_rae_risk_trends IS 'AGENT-EUDR-028: TimescaleDB hypertable for historical risk trend tracking with 30-day chunk intervals, enabling time-series analysis of operator risk evolution';
COMMENT ON COLUMN eudr_rae_risk_trends.key_changes IS 'Key changes from previous assessment: {"dimensions_changed": [...], "score_delta": 0.05, "level_changed": false}';
COMMENT ON COLUMN eudr_rae_risk_trends.assessed_at IS 'Hypertable partition key: assessment timestamp for efficient time-range queries';


-- ============================================================================
-- 8. eudr_rae_risk_reports -- Generated risk assessment reports
-- ============================================================================
RAISE NOTICE 'V116 [8/9]: Creating eudr_rae_risk_reports...';

CREATE TABLE IF NOT EXISTS eudr_rae_risk_reports (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id                       VARCHAR(100)    UNIQUE NOT NULL,
        -- Unique report identifier (e.g. "rae-rpt-2026-03-001")
    assessment_id                   VARCHAR(100)    NOT NULL,
        -- Reference to the parent risk assessment operation
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator for whom the report was generated
    commodity                       VARCHAR(50)     NOT NULL,
        -- EUDR commodity covered by the report
    composite_score                 DECIMAL(10,4),
        -- Composite risk score included in the report
    risk_level                      VARCHAR(20),
        -- Risk level classification included in the report
    dimension_breakdown             JSONB           DEFAULT '{}',
        -- Detailed per-dimension score breakdown for report rendering
    article10_summary               JSONB           DEFAULT '{}',
        -- Summary of Article 10(2) criteria evaluation results
    country_benchmarks              JSONB           DEFAULT '[]',
        -- Array of country benchmark data included in the report
    simplified_dd_eligible          BOOLEAN         DEFAULT FALSE,
        -- TRUE if simplified due diligence is eligible per assessment
    trend_summary                   JSONB,
        -- Historical trend summary (score evolution, direction, velocity)
    overrides                       JSONB           DEFAULT '[]',
        -- Array of risk overrides applied to this assessment
    recommendations                 JSONB           DEFAULT '[]',
        -- Array of risk mitigation recommendations
    dds_ready                       BOOLEAN         DEFAULT FALSE,
        -- TRUE if report data is sufficient for due diligence statement submission
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for report integrity verification
    generated_at                    TIMESTAMPTZ     DEFAULT NOW(),
        -- Timestamp when the report was generated

    CONSTRAINT fk_rae_rr_assessment FOREIGN KEY (assessment_id)
        REFERENCES eudr_rae_risk_assessments (operation_id),
    CONSTRAINT chk_rae_rr_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_rae_rr_risk_level CHECK (risk_level IS NULL OR risk_level IN (
        'negligible', 'low', 'standard', 'elevated', 'high', 'critical'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rr_report ON eudr_rae_risk_reports (report_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rr_assessment ON eudr_rae_risk_reports (assessment_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rr_operator ON eudr_rae_risk_reports (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rr_commodity ON eudr_rae_risk_reports (commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rr_composite ON eudr_rae_risk_reports (composite_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rr_risk_level ON eudr_rae_risk_reports (risk_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rr_simplified ON eudr_rae_risk_reports (simplified_dd_eligible);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rr_dds_ready ON eudr_rae_risk_reports (dds_ready);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rr_provenance ON eudr_rae_risk_reports (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rr_generated ON eudr_rae_risk_reports (generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rr_operator_commodity ON eudr_rae_risk_reports (operator_id, commodity, generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for DDS-ready reports
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rr_dds ON eudr_rae_risk_reports (generated_at DESC, operator_id)
        WHERE dds_ready = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for high/critical risk reports
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rr_high_risk ON eudr_rae_risk_reports (composite_score DESC, operator_id)
        WHERE risk_level IN ('high', 'critical');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rr_dimensions ON eudr_rae_risk_reports USING GIN (dimension_breakdown);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rr_article10 ON eudr_rae_risk_reports USING GIN (article10_summary);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rr_benchmarks ON eudr_rae_risk_reports USING GIN (country_benchmarks);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rr_overrides ON eudr_rae_risk_reports USING GIN (overrides);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_rr_recommendations ON eudr_rae_risk_reports USING GIN (recommendations);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_rae_risk_reports IS 'AGENT-EUDR-028: Generated risk assessment reports with composite scores, dimension breakdowns, Article 10(2) summaries, country benchmarks, trends, overrides, and DDS readiness flags';
COMMENT ON COLUMN eudr_rae_risk_reports.dds_ready IS 'TRUE if report data meets minimum Article 10 requirements for due diligence statement submission to EU Information System';
COMMENT ON COLUMN eudr_rae_risk_reports.trend_summary IS 'Historical trend: {"direction": "improving|stable|deteriorating", "velocity": 0.02, "period_days": 90, "previous_scores": [...]}';


-- ============================================================================
-- 9. eudr_rae_audit_log -- Audit trail for all risk assessment operations
-- ============================================================================
RAISE NOTICE 'V116 [9/9]: Creating eudr_rae_audit_log...';

CREATE TABLE IF NOT EXISTS eudr_rae_audit_log (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    entry_id                        VARCHAR(100)    UNIQUE NOT NULL,
        -- Unique audit entry identifier (e.g. "rae-audit-2026-03-001")
    operation                       VARCHAR(100)    NOT NULL,
        -- Operation that was performed
    entity_type                     VARCHAR(100)    NOT NULL,
        -- Type of entity affected by the operation
    entity_id                       VARCHAR(200)    NOT NULL,
        -- Identifier of the entity affected
    actor                           VARCHAR(100)    DEFAULT 'gl-eudr-rae-028',
        -- Actor who performed the operation (system agent or user)
    details                         JSONB           DEFAULT '{}',
        -- Detailed audit context (changed fields, old/new values, reason)
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for audit integrity verification (chained to previous entry)
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_rae_al_operation CHECK (operation IN (
        'assessment_started', 'assessment_completed', 'assessment_failed',
        'factors_collected', 'factors_aggregated',
        'composite_score_calculated', 'dimension_score_computed',
        'article10_evaluated', 'article10_criterion_passed', 'article10_criterion_failed',
        'benchmark_applied', 'benchmark_updated',
        'risk_classified', 'risk_escalated', 'risk_de_escalated',
        'override_requested', 'override_approved', 'override_rejected', 'override_expired',
        'report_generated', 'report_updated',
        'trend_recorded', 'trend_anomaly_detected',
        'simplified_dd_determined',
        'config_updated', 'manual_action'
    )),
    CONSTRAINT chk_rae_al_entity_type CHECK (entity_type IN (
        'risk_assessment', 'composite_score', 'dimension_score',
        'article10_evaluation', 'country_benchmark', 'risk_override',
        'risk_trend', 'risk_report', 'configuration'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_al_entry ON eudr_rae_audit_log (entry_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_al_operation ON eudr_rae_audit_log (operation);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_al_entity_type ON eudr_rae_audit_log (entity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_al_entity_id ON eudr_rae_audit_log (entity_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_al_actor ON eudr_rae_audit_log (actor);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_al_provenance ON eudr_rae_audit_log (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_al_created ON eudr_rae_audit_log (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_al_entity_op ON eudr_rae_audit_log (entity_type, operation, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_al_entity_id_time ON eudr_rae_audit_log (entity_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_rae_al_details ON eudr_rae_audit_log USING GIN (details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_rae_audit_log IS 'AGENT-EUDR-028: Article 31 compliant audit trail for all risk assessment operations, score computations, Article 10 evaluations, benchmarking, overrides, and report generation';
COMMENT ON COLUMN eudr_rae_audit_log.actor IS 'Default actor is gl-eudr-rae-028 (system agent); overridden for manual user actions such as risk overrides';
COMMENT ON COLUMN eudr_rae_audit_log.provenance_hash IS 'SHA-256 hash chained to previous entry for tamper-evident audit trail per Article 31';


-- ============================================================================
-- FINALIZE
-- ============================================================================

RAISE NOTICE 'V116: AGENT-EUDR-028 Risk Assessment Engine tables created successfully!';
RAISE NOTICE 'V116: Created 9 tables (8 regular + 1 hypertable), ~110 indexes (B-tree, GIN, partial)';
RAISE NOTICE 'V116: Foreign keys: composite_scores, article10_evaluations, risk_overrides, risk_reports -> risk_assessments; dimension_scores -> composite_scores';
RAISE NOTICE 'V116: Hypertable: eudr_rae_risk_trends (30-day chunks on assessed_at)';

COMMIT;
