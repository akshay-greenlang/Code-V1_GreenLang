-- ============================================================================
-- V117: AGENT-EUDR-029 Mitigation Measure Designer
-- ============================================================================
-- Creates tables for the Mitigation Measure Designer which designs risk
-- mitigation strategies linked to risk assessments, manages individual
-- mitigation measures with full lifecycle tracking, provides a curated
-- template library of Article 11 measures, collects evidence/document
-- attachments, estimates effectiveness with conservative/moderate/optimistic
-- projections, records post-mitigation verification results, maintains
-- workflow state machine records, tracks implementation milestones, and
-- preserves a complete Article 31 audit trail via TimescaleDB hypertable.
--
-- Agent ID: GL-EUDR-MMD-029
-- PRD: PRD-AGENT-EUDR-029
-- Regulation: EU 2023/1115 (EUDR) Articles 10, 11, 12, 13, 14-16, 29, 31
-- Tables: 9 (8 regular + 1 hypertable)
-- Indexes: ~115
--
-- Dependencies: TimescaleDB extension (for eudr_mmd_audit_log hypertable)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V117: Creating AGENT-EUDR-029 Mitigation Measure Designer tables...';


-- ============================================================================
-- 1. eudr_mmd_mitigation_strategies -- Strategy records linked to risk assessments
-- ============================================================================
RAISE NOTICE 'V117 [1/9]: Creating eudr_mmd_mitigation_strategies...';

CREATE TABLE IF NOT EXISTS eudr_mmd_mitigation_strategies (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id                     VARCHAR(100)    UNIQUE NOT NULL,
        -- Unique identifier for this mitigation strategy (e.g. "mmd-str-2026-03-001")
    workflow_id                     VARCHAR(100),
        -- Reference to the due diligence workflow orchestrating this strategy
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator initiating the mitigation strategy design
    commodity                       VARCHAR(50)     NOT NULL,
        -- EUDR commodity being mitigated (cattle, cocoa, coffee, oil_palm, rubber, soya, wood)
    risk_assessment_id              VARCHAR(100)    NOT NULL,
        -- Reference to the upstream risk assessment that triggered this strategy
    pre_mitigation_score            DECIMAL(10,4)   NOT NULL,
        -- Composite risk score before mitigation (0.0000 to 1.0000)
    target_score                    DECIMAL(10,4)   NOT NULL DEFAULT 0.3000,
        -- Target risk score after mitigation completion (default: 0.30 = low risk)
    post_mitigation_score           DECIMAL(10,4),
        -- Actual composite risk score after mitigation (NULL if not yet verified)
    risk_level                      VARCHAR(20)     NOT NULL,
        -- Pre-mitigation risk level classification
    risk_dimensions                 JSONB           DEFAULT '{}',
        -- Object of risk dimension scores requiring mitigation
    status                          VARCHAR(30)     NOT NULL DEFAULT 'designed',
        -- Strategy lifecycle status
    designed_by                     VARCHAR(100),
        -- User or system actor that designed the strategy
    designed_at                     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the strategy was designed
    approved_by                     VARCHAR(100),
        -- User who approved the strategy for implementation
    approved_at                     TIMESTAMPTZ,
        -- Timestamp when the strategy was approved
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for strategy integrity verification

    CONSTRAINT chk_mmd_ms_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_mmd_ms_status CHECK (status IN (
        'designed', 'pending_approval', 'approved', 'in_progress',
        'partially_complete', 'completed', 'verified', 'failed', 'cancelled'
    )),
    CONSTRAINT chk_mmd_ms_risk_level CHECK (risk_level IN (
        'negligible', 'low', 'standard', 'elevated', 'high', 'critical'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_strategy ON eudr_mmd_mitigation_strategies (strategy_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_workflow ON eudr_mmd_mitigation_strategies (workflow_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_operator ON eudr_mmd_mitigation_strategies (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_commodity ON eudr_mmd_mitigation_strategies (commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_risk_assessment ON eudr_mmd_mitigation_strategies (risk_assessment_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_pre_score ON eudr_mmd_mitigation_strategies (pre_mitigation_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_target_score ON eudr_mmd_mitigation_strategies (target_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_post_score ON eudr_mmd_mitigation_strategies (post_mitigation_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_risk_level ON eudr_mmd_mitigation_strategies (risk_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_status ON eudr_mmd_mitigation_strategies (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_designed_by ON eudr_mmd_mitigation_strategies (designed_by);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_designed_at ON eudr_mmd_mitigation_strategies (designed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_approved_by ON eudr_mmd_mitigation_strategies (approved_by);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_updated ON eudr_mmd_mitigation_strategies (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_provenance ON eudr_mmd_mitigation_strategies (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_operator_commodity ON eudr_mmd_mitigation_strategies (operator_id, commodity, designed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_commodity_status ON eudr_mmd_mitigation_strategies (commodity, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_risk_commodity ON eudr_mmd_mitigation_strategies (risk_level, commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_status_risk ON eudr_mmd_mitigation_strategies (status, risk_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active (non-terminal) strategies
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_active ON eudr_mmd_mitigation_strategies (designed_at DESC)
        WHERE status NOT IN ('completed', 'verified', 'failed', 'cancelled');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for high/critical risk strategies requiring priority attention
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_high_risk ON eudr_mmd_mitigation_strategies (pre_mitigation_score DESC, operator_id)
        WHERE risk_level IN ('high', 'critical');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for strategies pending approval
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_pending ON eudr_mmd_mitigation_strategies (designed_at DESC, operator_id)
        WHERE status = 'pending_approval';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ms_dimensions ON eudr_mmd_mitigation_strategies USING GIN (risk_dimensions);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_mmd_mitigation_strategies IS 'AGENT-EUDR-029: Mitigation strategy records linking risk assessments to designed mitigation plans with pre/post scores and lifecycle tracking for EUDR Articles 10-11 due diligence';
COMMENT ON COLUMN eudr_mmd_mitigation_strategies.pre_mitigation_score IS 'Composite risk score before mitigation: 0.0000 (negligible) to 1.0000 (critical), sourced from EUDR-028 Risk Assessment Engine';
COMMENT ON COLUMN eudr_mmd_mitigation_strategies.target_score IS 'Target risk score after all measures are implemented: default 0.3000 (low risk threshold per Article 11)';


-- ============================================================================
-- 2. eudr_mmd_mitigation_measures -- Individual measures with status tracking
-- ============================================================================
RAISE NOTICE 'V117 [2/9]: Creating eudr_mmd_mitigation_measures...';

CREATE TABLE IF NOT EXISTS eudr_mmd_mitigation_measures (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    measure_id                      VARCHAR(100)    UNIQUE NOT NULL,
        -- Unique identifier for this measure (e.g. "mmd-msr-2026-03-001")
    strategy_id                     VARCHAR(100)    NOT NULL,
        -- Reference to the parent mitigation strategy
    template_id                     VARCHAR(50),
        -- Reference to the measure template used (NULL if custom measure)
    title                           VARCHAR(500)    NOT NULL,
        -- Human-readable measure title
    description                     TEXT,
        -- Detailed description of the mitigation measure
    article11_category              VARCHAR(30)     NOT NULL,
        -- EUDR Article 11 risk mitigation category
    target_dimension                VARCHAR(50)     NOT NULL,
        -- Risk dimension this measure targets for reduction
    status                          VARCHAR(30)     NOT NULL DEFAULT 'proposed',
        -- Measure lifecycle status
    priority                        VARCHAR(20)     NOT NULL DEFAULT 'medium',
        -- Implementation priority
    assigned_to                     VARCHAR(100),
        -- User or team assigned to implement this measure
    deadline                        TIMESTAMPTZ,
        -- Target completion deadline
    started_at                      TIMESTAMPTZ,
        -- Timestamp when implementation began
    completed_at                    TIMESTAMPTZ,
        -- Timestamp when measure was completed
    cancelled_at                    TIMESTAMPTZ,
        -- Timestamp when measure was cancelled (NULL if active)
    cancel_reason                   TEXT,
        -- Justification for cancellation (required when cancelled)
    expected_risk_reduction         DECIMAL(10,4),
        -- Expected risk score reduction (0.0000 to 1.0000)
    actual_risk_reduction           DECIMAL(10,4),
        -- Actual verified risk score reduction (NULL until verified)
    evidence_count                  INTEGER         DEFAULT 0,
        -- Count of evidence documents attached to this measure
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_mmd_mm_strategy FOREIGN KEY (strategy_id)
        REFERENCES eudr_mmd_mitigation_strategies (strategy_id),
    CONSTRAINT chk_mmd_mm_category CHECK (article11_category IN (
        'additional_information', 'independent_survey', 'third_party_verification',
        'supplier_engagement', 'supply_chain_restructuring', 'certification',
        'monitoring_system', 'field_inspection', 'document_review', 'other'
    )),
    CONSTRAINT chk_mmd_mm_dimension CHECK (target_dimension IN (
        'country_risk', 'supplier_risk', 'commodity_risk', 'corruption_risk',
        'deforestation_risk', 'indigenous_rights', 'protected_areas',
        'legal_compliance', 'traceability', 'certification'
    )),
    CONSTRAINT chk_mmd_mm_status CHECK (status IN (
        'proposed', 'approved', 'in_progress', 'completed', 'verified',
        'overdue', 'cancelled', 'failed'
    )),
    CONSTRAINT chk_mmd_mm_priority CHECK (priority IN (
        'critical', 'high', 'medium', 'low'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_measure ON eudr_mmd_mitigation_measures (measure_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_strategy ON eudr_mmd_mitigation_measures (strategy_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_template ON eudr_mmd_mitigation_measures (template_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_category ON eudr_mmd_mitigation_measures (article11_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_dimension ON eudr_mmd_mitigation_measures (target_dimension);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_status ON eudr_mmd_mitigation_measures (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_priority ON eudr_mmd_mitigation_measures (priority);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_assigned ON eudr_mmd_mitigation_measures (assigned_to);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_deadline ON eudr_mmd_mitigation_measures (deadline);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_started ON eudr_mmd_mitigation_measures (started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_completed ON eudr_mmd_mitigation_measures (completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_expected_reduction ON eudr_mmd_mitigation_measures (expected_risk_reduction DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_actual_reduction ON eudr_mmd_mitigation_measures (actual_risk_reduction DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_created ON eudr_mmd_mitigation_measures (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_updated ON eudr_mmd_mitigation_measures (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_strategy_status ON eudr_mmd_mitigation_measures (strategy_id, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_category_status ON eudr_mmd_mitigation_measures (article11_category, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_dimension_status ON eudr_mmd_mitigation_measures (target_dimension, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_priority_status ON eudr_mmd_mitigation_measures (priority, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active (non-terminal) measures
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_active ON eudr_mmd_mitigation_measures (deadline, priority)
        WHERE status NOT IN ('completed', 'verified', 'cancelled', 'failed');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for overdue measures requiring escalation
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_overdue ON eudr_mmd_mitigation_measures (deadline, strategy_id)
        WHERE status IN ('proposed', 'approved', 'in_progress', 'overdue')
        AND deadline IS NOT NULL AND deadline < NOW();
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for critical priority measures
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mm_critical ON eudr_mmd_mitigation_measures (created_at DESC, strategy_id)
        WHERE priority = 'critical' AND status NOT IN ('completed', 'verified', 'cancelled', 'failed');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_mmd_mitigation_measures IS 'AGENT-EUDR-029: Individual mitigation measures with Article 11 categorization, risk dimension targeting, lifecycle tracking, and expected/actual risk reduction metrics';
COMMENT ON COLUMN eudr_mmd_mitigation_measures.article11_category IS 'Article 11 category: additional_information, independent_survey, third_party_verification, supplier_engagement, supply_chain_restructuring, certification, monitoring_system, field_inspection, document_review, other';
COMMENT ON COLUMN eudr_mmd_mitigation_measures.target_dimension IS 'Risk dimension targeted: country_risk, supplier_risk, commodity_risk, corruption_risk, deforestation_risk, indigenous_rights, protected_areas, legal_compliance, traceability, certification';


-- ============================================================================
-- 3. eudr_mmd_measure_templates -- Curated template library
-- ============================================================================
RAISE NOTICE 'V117 [3/9]: Creating eudr_mmd_measure_templates...';

CREATE TABLE IF NOT EXISTS eudr_mmd_measure_templates (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    template_id                     VARCHAR(50)     UNIQUE NOT NULL,
        -- Unique template identifier (e.g. "tmpl-cert-fsc-001")
    title                           VARCHAR(500)    NOT NULL,
        -- Human-readable template title
    description                     TEXT            NOT NULL,
        -- Detailed template description with implementation guidance
    article11_category              VARCHAR(30)     NOT NULL,
        -- EUDR Article 11 risk mitigation category
    applicable_dimensions           JSONB           NOT NULL DEFAULT '[]',
        -- Array of risk dimensions this template can address
    applicable_commodities          JSONB           NOT NULL DEFAULT '[]',
        -- Array of EUDR commodities this template applies to (empty = all)
    base_effectiveness              DECIMAL(10,4)   NOT NULL,
        -- Base effectiveness score (0.0000 to 1.0000) under ideal conditions
    typical_timeline_days           INTEGER         NOT NULL DEFAULT 30,
        -- Typical implementation timeline in calendar days
    evidence_requirements           JSONB           NOT NULL DEFAULT '[]',
        -- Array of required evidence types for measure verification
    regulatory_reference            VARCHAR(200),
        -- EUDR article or regulatory reference for this template
    version                         VARCHAR(20)     NOT NULL DEFAULT '1.0',
        -- Template version for change tracking
    is_active                       BOOLEAN         NOT NULL DEFAULT TRUE,
        -- Whether template is currently available for use
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mt_template ON eudr_mmd_measure_templates (template_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mt_category ON eudr_mmd_measure_templates (article11_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mt_effectiveness ON eudr_mmd_measure_templates (base_effectiveness DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mt_timeline ON eudr_mmd_measure_templates (typical_timeline_days);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mt_version ON eudr_mmd_measure_templates (version);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mt_active ON eudr_mmd_measure_templates (is_active);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mt_created ON eudr_mmd_measure_templates (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mt_updated ON eudr_mmd_measure_templates (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mt_category_active ON eudr_mmd_measure_templates (article11_category, is_active);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mt_category_eff ON eudr_mmd_measure_templates (article11_category, base_effectiveness DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active templates only (primary lookup pattern)
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mt_active_templates ON eudr_mmd_measure_templates (article11_category, base_effectiveness DESC)
        WHERE is_active = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mt_dimensions ON eudr_mmd_measure_templates USING GIN (applicable_dimensions);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mt_commodities ON eudr_mmd_measure_templates USING GIN (applicable_commodities);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_mt_evidence ON eudr_mmd_measure_templates USING GIN (evidence_requirements);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_mmd_measure_templates IS 'AGENT-EUDR-029: Curated library of Article 11 mitigation measure templates with base effectiveness ratings, evidence requirements, and commodity/dimension applicability';
COMMENT ON COLUMN eudr_mmd_measure_templates.base_effectiveness IS 'Base effectiveness: 0.0000 (no effect) to 1.0000 (full risk elimination) under ideal implementation conditions';
COMMENT ON COLUMN eudr_mmd_measure_templates.evidence_requirements IS 'Array of evidence types: ["certification_document", "audit_report", "satellite_imagery", "field_inspection_report", "supplier_declaration", ...]';


-- ============================================================================
-- 4. eudr_mmd_measure_evidence -- Evidence/document attachments
-- ============================================================================
RAISE NOTICE 'V117 [4/9]: Creating eudr_mmd_measure_evidence...';

CREATE TABLE IF NOT EXISTS eudr_mmd_measure_evidence (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    evidence_id                     VARCHAR(100)    UNIQUE NOT NULL,
        -- Unique evidence identifier (e.g. "mmd-evd-2026-03-001")
    measure_id                      VARCHAR(100)    NOT NULL,
        -- Reference to the mitigation measure this evidence supports
    evidence_type                   VARCHAR(50)     NOT NULL,
        -- Type of evidence document
    title                           VARCHAR(500)    NOT NULL,
        -- Human-readable evidence title
    file_reference                  VARCHAR(1000),
        -- S3/object storage reference for the evidence file
    file_size_bytes                 BIGINT,
        -- File size in bytes
    mime_type                       VARCHAR(100),
        -- MIME type of the evidence file (e.g. "application/pdf")
    checksum                        VARCHAR(64),
        -- SHA-256 checksum of the file content for integrity verification
    uploaded_by                     VARCHAR(100)    NOT NULL,
        -- User who uploaded the evidence
    uploaded_at                     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when evidence was uploaded
    verified                        BOOLEAN         DEFAULT FALSE,
        -- Whether evidence has been verified by a reviewer
    verified_by                     VARCHAR(100),
        -- User who verified the evidence
    verified_at                     TIMESTAMPTZ,
        -- Timestamp when evidence was verified
    rejection_reason                TEXT,
        -- Reason for evidence rejection (NULL if not rejected)

    CONSTRAINT fk_mmd_me_measure FOREIGN KEY (measure_id)
        REFERENCES eudr_mmd_mitigation_measures (measure_id),
    CONSTRAINT chk_mmd_me_type CHECK (evidence_type IN (
        'certification_document', 'audit_report', 'satellite_imagery',
        'field_inspection_report', 'supplier_declaration', 'geolocation_data',
        'legal_document', 'trade_record', 'training_certificate',
        'photo_evidence', 'monitoring_report', 'third_party_assessment', 'other'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_me_evidence ON eudr_mmd_measure_evidence (evidence_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_me_measure ON eudr_mmd_measure_evidence (measure_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_me_type ON eudr_mmd_measure_evidence (evidence_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_me_uploaded_by ON eudr_mmd_measure_evidence (uploaded_by);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_me_uploaded_at ON eudr_mmd_measure_evidence (uploaded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_me_verified ON eudr_mmd_measure_evidence (verified);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_me_verified_by ON eudr_mmd_measure_evidence (verified_by);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_me_verified_at ON eudr_mmd_measure_evidence (verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_me_checksum ON eudr_mmd_measure_evidence (checksum);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_me_measure_type ON eudr_mmd_measure_evidence (measure_id, evidence_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_me_type_verified ON eudr_mmd_measure_evidence (evidence_type, verified);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for unverified evidence requiring review
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_me_unverified ON eudr_mmd_measure_evidence (uploaded_at DESC, measure_id)
        WHERE verified = FALSE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_mmd_measure_evidence IS 'AGENT-EUDR-029: Evidence and document attachments for mitigation measures with file references, integrity checksums, and verification status per Article 31 audit requirements';
COMMENT ON COLUMN eudr_mmd_measure_evidence.evidence_type IS 'Evidence type: certification_document, audit_report, satellite_imagery, field_inspection_report, supplier_declaration, geolocation_data, legal_document, trade_record, training_certificate, photo_evidence, monitoring_report, third_party_assessment, other';
COMMENT ON COLUMN eudr_mmd_measure_evidence.checksum IS 'SHA-256 hash of file content for tamper detection and integrity verification';


-- ============================================================================
-- 5. eudr_mmd_effectiveness_estimates -- Predicted risk reductions
-- ============================================================================
RAISE NOTICE 'V117 [5/9]: Creating eudr_mmd_effectiveness_estimates...';

CREATE TABLE IF NOT EXISTS eudr_mmd_effectiveness_estimates (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    estimate_id                     VARCHAR(100)    UNIQUE NOT NULL,
        -- Unique estimate identifier (e.g. "mmd-eff-2026-03-001")
    measure_id                      VARCHAR(100)    NOT NULL,
        -- Reference to the mitigation measure being estimated
    strategy_id                     VARCHAR(100)    NOT NULL,
        -- Reference to the parent mitigation strategy
    conservative_estimate           DECIMAL(10,4)   NOT NULL,
        -- Conservative (pessimistic) risk reduction estimate (0.0000 to 1.0000)
    moderate_estimate               DECIMAL(10,4)   NOT NULL,
        -- Moderate (expected) risk reduction estimate (0.0000 to 1.0000)
    optimistic_estimate             DECIMAL(10,4)   NOT NULL,
        -- Optimistic (best-case) risk reduction estimate (0.0000 to 1.0000)
    applicability_factor            DECIMAL(10,4)   NOT NULL DEFAULT 1.0000,
        -- Factor adjusting for measure applicability to this specific context (0.0 to 1.0)
    quality_factor                  DECIMAL(10,4)   NOT NULL DEFAULT 1.0000,
        -- Factor adjusting for expected implementation quality (0.0 to 1.0)
    confidence                      DECIMAL(10,4)   NOT NULL DEFAULT 0.7000,
        -- Confidence level of the effectiveness estimate (0.0000 to 1.0000)
    methodology                     VARCHAR(50),
        -- Estimation methodology used (e.g. "template_based", "historical_analogy", "expert_judgment")
    calculated_at                   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the estimate was calculated
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for estimate integrity verification

    CONSTRAINT fk_mmd_ee_measure FOREIGN KEY (measure_id)
        REFERENCES eudr_mmd_mitigation_measures (measure_id),
    CONSTRAINT fk_mmd_ee_strategy FOREIGN KEY (strategy_id)
        REFERENCES eudr_mmd_mitigation_strategies (strategy_id),
    CONSTRAINT chk_mmd_ee_methodology CHECK (methodology IS NULL OR methodology IN (
        'template_based', 'historical_analogy', 'expert_judgment',
        'statistical_model', 'composite', 'manual_override'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ee_estimate ON eudr_mmd_effectiveness_estimates (estimate_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ee_measure ON eudr_mmd_effectiveness_estimates (measure_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ee_strategy ON eudr_mmd_effectiveness_estimates (strategy_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ee_conservative ON eudr_mmd_effectiveness_estimates (conservative_estimate DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ee_moderate ON eudr_mmd_effectiveness_estimates (moderate_estimate DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ee_optimistic ON eudr_mmd_effectiveness_estimates (optimistic_estimate DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ee_applicability ON eudr_mmd_effectiveness_estimates (applicability_factor);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ee_quality ON eudr_mmd_effectiveness_estimates (quality_factor);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ee_confidence ON eudr_mmd_effectiveness_estimates (confidence DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ee_methodology ON eudr_mmd_effectiveness_estimates (methodology);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ee_calculated ON eudr_mmd_effectiveness_estimates (calculated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ee_provenance ON eudr_mmd_effectiveness_estimates (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ee_strategy_calc ON eudr_mmd_effectiveness_estimates (strategy_id, calculated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ee_measure_calc ON eudr_mmd_effectiveness_estimates (measure_id, calculated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for high-confidence estimates
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ee_high_conf ON eudr_mmd_effectiveness_estimates (moderate_estimate DESC, strategy_id)
        WHERE confidence >= 0.8000;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_mmd_effectiveness_estimates IS 'AGENT-EUDR-029: Predicted risk reduction estimates with conservative/moderate/optimistic projections, applicability and quality adjustment factors, and confidence levels';
COMMENT ON COLUMN eudr_mmd_effectiveness_estimates.conservative_estimate IS 'Pessimistic risk reduction: lowest expected impact under unfavorable conditions (0.0000 to 1.0000)';
COMMENT ON COLUMN eudr_mmd_effectiveness_estimates.applicability_factor IS 'Context adjustment: 1.0 for perfectly applicable template, lower for partial applicability (e.g. 0.7 for different commodity)';


-- ============================================================================
-- 6. eudr_mmd_verification_results -- Post-mitigation verification results
-- ============================================================================
RAISE NOTICE 'V117 [6/9]: Creating eudr_mmd_verification_results...';

CREATE TABLE IF NOT EXISTS eudr_mmd_verification_results (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    verification_id                 VARCHAR(100)    UNIQUE NOT NULL,
        -- Unique verification identifier (e.g. "mmd-vrf-2026-03-001")
    strategy_id                     VARCHAR(100)    NOT NULL,
        -- Reference to the mitigation strategy being verified
    pre_mitigation_score            DECIMAL(10,4)   NOT NULL,
        -- Composite risk score before mitigation was applied
    post_mitigation_score           DECIMAL(10,4)   NOT NULL,
        -- Composite risk score after mitigation was applied
    risk_reduction                  DECIMAL(10,4)   NOT NULL,
        -- Absolute risk score reduction (pre - post)
    reduction_percentage            DECIMAL(10,4)   NOT NULL,
        -- Percentage risk reduction ((pre - post) / pre * 100)
    result                          VARCHAR(30)     NOT NULL,
        -- Verification outcome
    target_met                      BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether the strategy target score was met
    additional_measures_needed      BOOLEAN         DEFAULT FALSE,
        -- Whether additional measures are recommended
    dimension_breakdown             JSONB           DEFAULT '{}',
        -- Per-dimension reduction breakdown
    recommendations                 JSONB           DEFAULT '[]',
        -- Array of post-verification recommendations
    verified_by                     VARCHAR(100),
        -- User or system that performed the verification
    verified_at                     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when verification was completed
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for verification integrity

    CONSTRAINT fk_mmd_vr_strategy FOREIGN KEY (strategy_id)
        REFERENCES eudr_mmd_mitigation_strategies (strategy_id),
    CONSTRAINT chk_mmd_vr_result CHECK (result IN (
        'effective', 'partially_effective', 'ineffective',
        'inconclusive', 'pending_review'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_vr_verification ON eudr_mmd_verification_results (verification_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_vr_strategy ON eudr_mmd_verification_results (strategy_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_vr_pre_score ON eudr_mmd_verification_results (pre_mitigation_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_vr_post_score ON eudr_mmd_verification_results (post_mitigation_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_vr_reduction ON eudr_mmd_verification_results (risk_reduction DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_vr_reduction_pct ON eudr_mmd_verification_results (reduction_percentage DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_vr_result ON eudr_mmd_verification_results (result);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_vr_target_met ON eudr_mmd_verification_results (target_met);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_vr_additional ON eudr_mmd_verification_results (additional_measures_needed);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_vr_verified_by ON eudr_mmd_verification_results (verified_by);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_vr_verified_at ON eudr_mmd_verification_results (verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_vr_provenance ON eudr_mmd_verification_results (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_vr_strategy_time ON eudr_mmd_verification_results (strategy_id, verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_vr_result_time ON eudr_mmd_verification_results (result, verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for ineffective verifications requiring follow-up
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_vr_ineffective ON eudr_mmd_verification_results (verified_at DESC, strategy_id)
        WHERE result IN ('ineffective', 'inconclusive');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for verifications needing additional measures
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_vr_needs_more ON eudr_mmd_verification_results (verified_at DESC, strategy_id)
        WHERE additional_measures_needed = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_vr_dimensions ON eudr_mmd_verification_results USING GIN (dimension_breakdown);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_vr_recommendations ON eudr_mmd_verification_results USING GIN (recommendations);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_mmd_verification_results IS 'AGENT-EUDR-029: Post-mitigation verification results recording actual risk reduction, target attainment, and recommendations for Article 11 compliance verification';
COMMENT ON COLUMN eudr_mmd_verification_results.result IS 'Verification outcome: effective (>= target), partially_effective (>50% of target), ineffective (<50% of target), inconclusive (insufficient data), pending_review';
COMMENT ON COLUMN eudr_mmd_verification_results.reduction_percentage IS 'Percentage risk reduction: ((pre_score - post_score) / pre_score) * 100';


-- ============================================================================
-- 7. eudr_mmd_workflow_states -- Workflow state machine records
-- ============================================================================
RAISE NOTICE 'V117 [7/9]: Creating eudr_mmd_workflow_states...';

CREATE TABLE IF NOT EXISTS eudr_mmd_workflow_states (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id                     VARCHAR(100)    UNIQUE NOT NULL,
        -- Unique workflow identifier (e.g. "mmd-wfl-2026-03-001")
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator for this workflow
    commodity                       VARCHAR(50)     NOT NULL,
        -- EUDR commodity for this workflow
    strategy_id                     VARCHAR(100),
        -- Reference to the associated mitigation strategy (NULL until strategy designed)
    status                          VARCHAR(30)     NOT NULL DEFAULT 'initiated',
        -- Workflow lifecycle status
    current_phase                   VARCHAR(50),
        -- Current workflow phase
    risk_trigger_data               JSONB           NOT NULL DEFAULT '{}',
        -- JSON object describing the risk trigger that initiated this workflow
    started_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the workflow was initiated
    closed_at                       TIMESTAMPTZ,
        -- Timestamp when the workflow was closed (NULL if open)
    escalated_at                    TIMESTAMPTZ,
        -- Timestamp when the workflow was escalated (NULL if not escalated)
    escalation_reason               TEXT,
        -- Reason for workflow escalation
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_mmd_ws_strategy FOREIGN KEY (strategy_id)
        REFERENCES eudr_mmd_mitigation_strategies (strategy_id),
    CONSTRAINT chk_mmd_ws_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_mmd_ws_status CHECK (status IN (
        'initiated', 'designing', 'pending_approval', 'implementing',
        'monitoring', 'verifying', 'closed', 'escalated', 'failed'
    )),
    CONSTRAINT chk_mmd_ws_phase CHECK (current_phase IS NULL OR current_phase IN (
        'risk_analysis', 'template_selection', 'measure_design',
        'effectiveness_estimation', 'approval', 'implementation',
        'evidence_collection', 'verification', 'reporting', 'closure'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ws_workflow ON eudr_mmd_workflow_states (workflow_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ws_operator ON eudr_mmd_workflow_states (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ws_commodity ON eudr_mmd_workflow_states (commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ws_strategy ON eudr_mmd_workflow_states (strategy_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ws_status ON eudr_mmd_workflow_states (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ws_phase ON eudr_mmd_workflow_states (current_phase);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ws_started ON eudr_mmd_workflow_states (started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ws_closed ON eudr_mmd_workflow_states (closed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ws_escalated ON eudr_mmd_workflow_states (escalated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ws_updated ON eudr_mmd_workflow_states (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ws_operator_commodity ON eudr_mmd_workflow_states (operator_id, commodity, started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ws_commodity_status ON eudr_mmd_workflow_states (commodity, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ws_status_phase ON eudr_mmd_workflow_states (status, current_phase);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active (non-terminal) workflows
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ws_active ON eudr_mmd_workflow_states (started_at DESC, operator_id)
        WHERE status NOT IN ('closed', 'failed');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for escalated workflows requiring attention
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ws_escalated_active ON eudr_mmd_workflow_states (escalated_at DESC, operator_id)
        WHERE status = 'escalated';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_ws_trigger_data ON eudr_mmd_workflow_states USING GIN (risk_trigger_data);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_mmd_workflow_states IS 'AGENT-EUDR-029: Workflow state machine records tracking mitigation design lifecycle from initiation through design, implementation, verification, and closure';
COMMENT ON COLUMN eudr_mmd_workflow_states.current_phase IS 'Workflow phase: risk_analysis, template_selection, measure_design, effectiveness_estimation, approval, implementation, evidence_collection, verification, reporting, closure';
COMMENT ON COLUMN eudr_mmd_workflow_states.risk_trigger_data IS 'Risk trigger context: {"risk_assessment_id": "...", "composite_score": 0.75, "risk_level": "high", "trigger_dimensions": [...]}';


-- ============================================================================
-- 8. eudr_mmd_implementation_milestones -- Milestone tracking
-- ============================================================================
RAISE NOTICE 'V117 [8/9]: Creating eudr_mmd_implementation_milestones...';

CREATE TABLE IF NOT EXISTS eudr_mmd_implementation_milestones (
    id                              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    milestone_id                    VARCHAR(100)    UNIQUE NOT NULL,
        -- Unique milestone identifier (e.g. "mmd-mls-2026-03-001")
    measure_id                      VARCHAR(100)    NOT NULL,
        -- Reference to the parent mitigation measure
    title                           VARCHAR(500)    NOT NULL,
        -- Human-readable milestone title
    description                     TEXT,
        -- Detailed description of what the milestone entails
    sequence_number                 INTEGER         NOT NULL DEFAULT 1,
        -- Ordering within the measure implementation plan
    due_date                        TIMESTAMPTZ     NOT NULL,
        -- Target completion date for this milestone
    completed_at                    TIMESTAMPTZ,
        -- Actual completion timestamp (NULL if not yet completed)
    completed_by                    VARCHAR(100),
        -- User who marked the milestone as complete
    status                          VARCHAR(30)     NOT NULL DEFAULT 'pending',
        -- Milestone lifecycle status
    deliverable                     TEXT,
        -- Expected deliverable or output for this milestone
    notes                           TEXT,
        -- Implementation notes or observations
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_mmd_im_measure FOREIGN KEY (measure_id)
        REFERENCES eudr_mmd_mitigation_measures (measure_id),
    CONSTRAINT chk_mmd_im_status CHECK (status IN (
        'pending', 'in_progress', 'completed', 'overdue', 'skipped', 'cancelled'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_im_milestone ON eudr_mmd_implementation_milestones (milestone_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_im_measure ON eudr_mmd_implementation_milestones (measure_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_im_sequence ON eudr_mmd_implementation_milestones (measure_id, sequence_number);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_im_due_date ON eudr_mmd_implementation_milestones (due_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_im_completed ON eudr_mmd_implementation_milestones (completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_im_completed_by ON eudr_mmd_implementation_milestones (completed_by);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_im_status ON eudr_mmd_implementation_milestones (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_im_created ON eudr_mmd_implementation_milestones (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_im_updated ON eudr_mmd_implementation_milestones (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_im_measure_status ON eudr_mmd_implementation_milestones (measure_id, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_im_status_due ON eudr_mmd_implementation_milestones (status, due_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for overdue milestones requiring escalation
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_im_overdue ON eudr_mmd_implementation_milestones (due_date, measure_id)
        WHERE status IN ('pending', 'in_progress', 'overdue')
        AND due_date < NOW();
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active milestones
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_im_active ON eudr_mmd_implementation_milestones (due_date, measure_id)
        WHERE status NOT IN ('completed', 'skipped', 'cancelled');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE eudr_mmd_implementation_milestones IS 'AGENT-EUDR-029: Implementation milestones tracking progress of individual mitigation measures with due dates, sequencing, and completion status';
COMMENT ON COLUMN eudr_mmd_implementation_milestones.sequence_number IS 'Execution order within the measure: milestones are implemented sequentially (1, 2, 3, ...)';
COMMENT ON COLUMN eudr_mmd_implementation_milestones.status IS 'Milestone status: pending (not started), in_progress, completed, overdue (past due_date), skipped (intentionally bypassed), cancelled';


-- ============================================================================
-- 9. eudr_mmd_audit_log -- Immutable audit trail (TimescaleDB hypertable)
-- ============================================================================
RAISE NOTICE 'V117 [9/9]: Creating eudr_mmd_audit_log (hypertable)...';

CREATE TABLE IF NOT EXISTS eudr_mmd_audit_log (
    id                              UUID            DEFAULT gen_random_uuid(),
    entry_id                        VARCHAR(100)    NOT NULL,
        -- Unique audit entry identifier (e.g. "mmd-audit-2026-03-001")
    operation                       VARCHAR(100)    NOT NULL,
        -- Operation that was performed
    entity_type                     VARCHAR(100)    NOT NULL,
        -- Type of entity affected by the operation
    entity_id                       VARCHAR(200)    NOT NULL,
        -- Identifier of the entity affected
    actor                           VARCHAR(100)    DEFAULT 'gl-eudr-mmd-029',
        -- Actor who performed the operation (system agent or user)
    old_state                       JSONB,
        -- Previous state of the entity (NULL for creation operations)
    new_state                       JSONB,
        -- New state of the entity after the operation
    metadata                        JSONB           DEFAULT '{}',
        -- Additional context (IP address, session ID, correlation ID, etc.)
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash chained to previous entry for tamper-evident audit trail
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_mmd_al_operation CHECK (operation IN (
        'strategy_designed', 'strategy_approved', 'strategy_started',
        'strategy_completed', 'strategy_verified', 'strategy_failed', 'strategy_cancelled',
        'measure_proposed', 'measure_approved', 'measure_started',
        'measure_completed', 'measure_verified', 'measure_overdue',
        'measure_cancelled', 'measure_failed',
        'template_created', 'template_updated', 'template_deactivated',
        'evidence_uploaded', 'evidence_verified', 'evidence_rejected',
        'effectiveness_estimated', 'effectiveness_updated',
        'verification_completed', 'verification_reviewed',
        'workflow_initiated', 'workflow_phase_changed', 'workflow_escalated',
        'workflow_closed', 'workflow_failed',
        'milestone_created', 'milestone_started', 'milestone_completed',
        'milestone_overdue', 'milestone_skipped', 'milestone_cancelled',
        'config_updated', 'manual_action'
    )),
    CONSTRAINT chk_mmd_al_entity_type CHECK (entity_type IN (
        'mitigation_strategy', 'mitigation_measure', 'measure_template',
        'measure_evidence', 'effectiveness_estimate', 'verification_result',
        'workflow_state', 'implementation_milestone', 'configuration'
    ))
);

-- Convert to TimescaleDB hypertable partitioned on created_at
SELECT create_hypertable('eudr_mmd_audit_log', 'created_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_al_entry ON eudr_mmd_audit_log (entry_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_al_operation ON eudr_mmd_audit_log (operation);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_al_entity_type ON eudr_mmd_audit_log (entity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_al_entity_id ON eudr_mmd_audit_log (entity_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_al_actor ON eudr_mmd_audit_log (actor);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_al_provenance ON eudr_mmd_audit_log (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_al_created ON eudr_mmd_audit_log (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_al_entity_op ON eudr_mmd_audit_log (entity_type, operation, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_al_entity_id_time ON eudr_mmd_audit_log (entity_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_al_actor_time ON eudr_mmd_audit_log (actor, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_al_old_state ON eudr_mmd_audit_log USING GIN (old_state);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_al_new_state ON eudr_mmd_audit_log USING GIN (new_state);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_mmd_al_metadata ON eudr_mmd_audit_log USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- DATA RETENTION POLICY -- Article 31: 5-year retention for EUDR audit data
-- ============================================================================
RAISE NOTICE 'V117: Configuring 5-year data retention policy per EUDR Article 31...';

-- Audit log retention: 5 years (60 months) per Article 31 requirement
SELECT add_retention_policy('eudr_mmd_audit_log',
    INTERVAL '5 years',
    if_not_exists => TRUE
);


-- ============================================================================
-- TABLE COMMENTS
-- ============================================================================

COMMENT ON TABLE eudr_mmd_audit_log IS 'AGENT-EUDR-029: Article 31 compliant immutable audit trail (TimescaleDB hypertable, 1-month chunks) for all mitigation measure design operations, evidence management, verifications, and workflow transitions with 5-year retention';
COMMENT ON COLUMN eudr_mmd_audit_log.actor IS 'Default actor is gl-eudr-mmd-029 (system agent); overridden for manual user actions such as approvals and evidence verification';
COMMENT ON COLUMN eudr_mmd_audit_log.provenance_hash IS 'SHA-256 hash chained to previous entry for tamper-evident audit trail per EUDR Article 31';


-- ============================================================================
-- FINALIZE
-- ============================================================================

RAISE NOTICE 'V117: AGENT-EUDR-029 Mitigation Measure Designer tables created successfully!';
RAISE NOTICE 'V117: Created 9 tables (8 regular + 1 hypertable), ~115 indexes (B-tree, GIN, partial)';
RAISE NOTICE 'V117: Foreign keys: measures -> strategies; evidence -> measures; effectiveness -> measures + strategies; verification -> strategies; workflow -> strategies; milestones -> measures';
RAISE NOTICE 'V117: Hypertable: eudr_mmd_audit_log (1-month chunks on created_at, 5-year retention)';
RAISE NOTICE 'V117: Retention policy: 5 years per EUDR Article 31';

COMMIT;
