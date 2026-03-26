-- =============================================================================
-- V333: PACK-041 Scope 1-2 Complete Pack - Compliance Mapping
-- =============================================================================
-- Pack:         PACK-041 (Scope 1-2 Complete Pack)
-- Migration:    008 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates compliance assessment tables for mapping GHG inventory completeness
-- against multiple reporting frameworks. Supports GHG Protocol Corporate
-- Standard, ISO 14064-1, CSRD/ESRS E1, CDP Climate, TCFD, SEC Climate Rule,
-- and national mandatory reporting schemes. Each assessment scores compliance
-- per framework, identifies gaps with prioritized remediation actions, and
-- provides an overall readiness classification.
--
-- Tables (3):
--   1. ghg_scope12.compliance_assessments
--   2. ghg_scope12.framework_results
--   3. ghg_scope12.compliance_gaps
--
-- Also includes: indexes, RLS, comments.
-- Previous: V332__pack041_trend_analysis.sql
-- =============================================================================

SET search_path TO ghg_scope12, public;

-- =============================================================================
-- Table 1: ghg_scope12.compliance_assessments
-- =============================================================================
-- Top-level compliance assessment for an organization and reporting year.
-- Evaluates the completeness, accuracy, consistency, transparency, and
-- relevance of the GHG inventory against selected regulatory and voluntary
-- frameworks. The overall score is a weighted average across frameworks.

CREATE TABLE ghg_scope12.compliance_assessments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL REFERENCES ghg_scope12.organizations(id) ON DELETE CASCADE,
    reporting_year              INTEGER         NOT NULL,
    assessment_name             VARCHAR(255),
    assessment_type             VARCHAR(30)     NOT NULL DEFAULT 'AUTOMATED',
    -- Scope
    scope1_inventory_id         UUID            REFERENCES ghg_scope12.scope1_inventories(id) ON DELETE SET NULL,
    scope2_inventory_id         UUID            REFERENCES ghg_scope12.scope2_inventories(id) ON DELETE SET NULL,
    base_year_id                UUID            REFERENCES ghg_scope12.base_years(id) ON DELETE SET NULL,
    -- Overall results
    overall_score               DECIMAL(5,2)    NOT NULL DEFAULT 0,
    overall_classification      VARCHAR(30),
    frameworks_assessed         INTEGER         NOT NULL DEFAULT 0,
    frameworks_compliant        INTEGER         NOT NULL DEFAULT 0,
    frameworks_partial          INTEGER         NOT NULL DEFAULT 0,
    frameworks_non_compliant    INTEGER         NOT NULL DEFAULT 0,
    total_requirements          INTEGER         NOT NULL DEFAULT 0,
    total_met                   INTEGER         NOT NULL DEFAULT 0,
    total_gaps                  INTEGER         NOT NULL DEFAULT 0,
    critical_gaps               INTEGER         NOT NULL DEFAULT 0,
    -- GHG Protocol principles assessment
    completeness_score          DECIMAL(5,2),
    accuracy_score              DECIMAL(5,2),
    consistency_score           DECIMAL(5,2),
    transparency_score          DECIMAL(5,2),
    relevance_score             DECIMAL(5,2),
    -- Workflow
    assessed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    assessed_by                 VARCHAR(255),
    reviewed_by                 VARCHAR(255),
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    notes                       TEXT,
    executive_summary           TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p041_ca_year CHECK (
        reporting_year >= 1990 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p041_ca_type CHECK (
        assessment_type IN ('AUTOMATED', 'MANUAL', 'HYBRID', 'THIRD_PARTY')
    ),
    CONSTRAINT chk_p041_ca_overall_score CHECK (
        overall_score >= 0 AND overall_score <= 100
    ),
    CONSTRAINT chk_p041_ca_classification CHECK (
        overall_classification IS NULL OR overall_classification IN (
            'FULLY_COMPLIANT', 'SUBSTANTIALLY_COMPLIANT', 'PARTIALLY_COMPLIANT',
            'NON_COMPLIANT', 'NOT_ASSESSED'
        )
    ),
    CONSTRAINT chk_p041_ca_status CHECK (
        status IN ('DRAFT', 'IN_PROGRESS', 'COMPLETED', 'REVIEWED', 'APPROVED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p041_ca_completeness CHECK (
        completeness_score IS NULL OR (completeness_score >= 0 AND completeness_score <= 100)
    ),
    CONSTRAINT chk_p041_ca_accuracy CHECK (
        accuracy_score IS NULL OR (accuracy_score >= 0 AND accuracy_score <= 100)
    ),
    CONSTRAINT chk_p041_ca_consistency CHECK (
        consistency_score IS NULL OR (consistency_score >= 0 AND consistency_score <= 100)
    ),
    CONSTRAINT chk_p041_ca_transparency CHECK (
        transparency_score IS NULL OR (transparency_score >= 0 AND transparency_score <= 100)
    ),
    CONSTRAINT chk_p041_ca_relevance CHECK (
        relevance_score IS NULL OR (relevance_score >= 0 AND relevance_score <= 100)
    ),
    CONSTRAINT chk_p041_ca_frameworks CHECK (
        frameworks_assessed >= 0
    ),
    CONSTRAINT chk_p041_ca_requirements CHECK (
        total_requirements >= 0 AND total_met >= 0 AND total_gaps >= 0
    ),
    CONSTRAINT chk_p041_ca_met_le_total CHECK (
        total_met <= total_requirements
    ),
    CONSTRAINT uq_p041_ca_org_year_type UNIQUE (organization_id, reporting_year, assessment_type)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_ca_tenant             ON ghg_scope12.compliance_assessments(tenant_id);
CREATE INDEX idx_p041_ca_org               ON ghg_scope12.compliance_assessments(organization_id);
CREATE INDEX idx_p041_ca_year              ON ghg_scope12.compliance_assessments(reporting_year);
CREATE INDEX idx_p041_ca_type              ON ghg_scope12.compliance_assessments(assessment_type);
CREATE INDEX idx_p041_ca_score             ON ghg_scope12.compliance_assessments(overall_score DESC);
CREATE INDEX idx_p041_ca_classification    ON ghg_scope12.compliance_assessments(overall_classification);
CREATE INDEX idx_p041_ca_status            ON ghg_scope12.compliance_assessments(status);
CREATE INDEX idx_p041_ca_critical_gaps     ON ghg_scope12.compliance_assessments(critical_gaps DESC);
CREATE INDEX idx_p041_ca_created           ON ghg_scope12.compliance_assessments(created_at DESC);
CREATE INDEX idx_p041_ca_metadata          ON ghg_scope12.compliance_assessments USING GIN(metadata);

-- Composite: org + year for history
CREATE INDEX idx_p041_ca_org_year          ON ghg_scope12.compliance_assessments(organization_id, reporting_year DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_ca_updated
    BEFORE UPDATE ON ghg_scope12.compliance_assessments
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_scope12.framework_results
-- =============================================================================
-- Per-framework compliance scoring within an assessment. Each framework is
-- evaluated independently against its specific requirements. Scores reflect
-- the percentage of requirements met, and a classification determines the
-- overall status (compliant, partially compliant, non-compliant).

CREATE TABLE ghg_scope12.framework_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    assessment_id               UUID            NOT NULL REFERENCES ghg_scope12.compliance_assessments(id) ON DELETE CASCADE,
    framework                   VARCHAR(50)     NOT NULL,
    framework_version           VARCHAR(30),
    framework_category          VARCHAR(30),
    -- Scoring
    score                       DECIMAL(5,2)    NOT NULL DEFAULT 0,
    classification              VARCHAR(30)     NOT NULL DEFAULT 'NOT_ASSESSED',
    -- Requirements breakdown
    total_requirements          INTEGER         NOT NULL DEFAULT 0,
    met                         INTEGER         NOT NULL DEFAULT 0,
    partially_met               INTEGER         NOT NULL DEFAULT 0,
    not_met                     INTEGER         NOT NULL DEFAULT 0,
    not_applicable              INTEGER         NOT NULL DEFAULT 0,
    -- Details
    mandatory_requirements      INTEGER         DEFAULT 0,
    mandatory_met               INTEGER         DEFAULT 0,
    optional_requirements       INTEGER         DEFAULT 0,
    optional_met                INTEGER         DEFAULT 0,
    -- Framework-specific fields
    disclosure_readiness_pct    DECIMAL(5,2),
    reporting_deadline          DATE,
    regulatory_authority        VARCHAR(255),
    regulatory_reference        VARCHAR(500),
    -- Workflow
    assessed_by                 VARCHAR(255),
    notes                       TEXT,
    recommendations             TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p041_fr_framework CHECK (
        framework IN (
            'GHG_PROTOCOL', 'ISO_14064_1', 'CSRD_ESRS_E1', 'CDP_CLIMATE',
            'TCFD', 'SEC_CLIMATE', 'UK_SECR', 'EU_ETS', 'EPA_GHGRP',
            'NGER_AUSTRALIA', 'JAPAN_GHGRP', 'KOREA_GHGRP', 'CHINA_GHGRP',
            'SBTI', 'ISSB_S2', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p041_fr_category CHECK (
        framework_category IS NULL OR framework_category IN (
            'VOLUNTARY', 'MANDATORY', 'REGULATORY', 'INVESTOR', 'CUSTOMER', 'OTHER'
        )
    ),
    CONSTRAINT chk_p041_fr_score CHECK (
        score >= 0 AND score <= 100
    ),
    CONSTRAINT chk_p041_fr_classification CHECK (
        classification IN (
            'FULLY_COMPLIANT', 'SUBSTANTIALLY_COMPLIANT', 'PARTIALLY_COMPLIANT',
            'NON_COMPLIANT', 'NOT_ASSESSED', 'NOT_APPLICABLE'
        )
    ),
    CONSTRAINT chk_p041_fr_requirements CHECK (
        total_requirements >= 0 AND met >= 0 AND partially_met >= 0 AND
        not_met >= 0 AND not_applicable >= 0
    ),
    CONSTRAINT chk_p041_fr_req_sum CHECK (
        met + partially_met + not_met + not_applicable <= total_requirements
    ),
    CONSTRAINT chk_p041_fr_mandatory CHECK (
        mandatory_requirements IS NULL OR mandatory_requirements >= 0
    ),
    CONSTRAINT chk_p041_fr_disclosure CHECK (
        disclosure_readiness_pct IS NULL OR (disclosure_readiness_pct >= 0 AND disclosure_readiness_pct <= 100)
    ),
    CONSTRAINT uq_p041_fr_assessment_framework UNIQUE (assessment_id, framework)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_fr_tenant             ON ghg_scope12.framework_results(tenant_id);
CREATE INDEX idx_p041_fr_assessment         ON ghg_scope12.framework_results(assessment_id);
CREATE INDEX idx_p041_fr_framework          ON ghg_scope12.framework_results(framework);
CREATE INDEX idx_p041_fr_category           ON ghg_scope12.framework_results(framework_category);
CREATE INDEX idx_p041_fr_score              ON ghg_scope12.framework_results(score DESC);
CREATE INDEX idx_p041_fr_classification     ON ghg_scope12.framework_results(classification);
CREATE INDEX idx_p041_fr_deadline           ON ghg_scope12.framework_results(reporting_deadline);
CREATE INDEX idx_p041_fr_created            ON ghg_scope12.framework_results(created_at DESC);
CREATE INDEX idx_p041_fr_metadata           ON ghg_scope12.framework_results USING GIN(metadata);

-- Composite: assessment + non-compliant for gap identification
CREATE INDEX idx_p041_fr_assess_gaps        ON ghg_scope12.framework_results(assessment_id, score)
    WHERE classification IN ('PARTIALLY_COMPLIANT', 'NON_COMPLIANT');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_fr_updated
    BEFORE UPDATE ON ghg_scope12.framework_results
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_scope12.compliance_gaps
-- =============================================================================
-- Individual compliance gaps (unmet or partially met requirements) within
-- a framework assessment. Each gap identifies the specific requirement,
-- describes the gap, and proposes a prioritized remediation action with
-- effort estimate and responsible party assignment.

CREATE TABLE ghg_scope12.compliance_gaps (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    framework_result_id         UUID            NOT NULL REFERENCES ghg_scope12.framework_results(id) ON DELETE CASCADE,
    -- Requirement
    requirement_id              VARCHAR(50)     NOT NULL,
    requirement_section         VARCHAR(100),
    requirement_description     TEXT            NOT NULL,
    requirement_type            VARCHAR(20)     NOT NULL DEFAULT 'MANDATORY',
    -- Gap details
    status                      VARCHAR(30)     NOT NULL DEFAULT 'NOT_MET',
    gap_description             TEXT            NOT NULL,
    gap_severity                VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    current_state               TEXT,
    evidence_available          TEXT,
    -- Remediation
    remediation_action          TEXT            NOT NULL,
    remediation_category        VARCHAR(30)     NOT NULL DEFAULT 'DATA_COLLECTION',
    estimated_effort_hours      DECIMAL(8,2),
    estimated_cost              DECIMAL(18,2),
    cost_currency               VARCHAR(3)      DEFAULT 'USD',
    implementation_complexity   VARCHAR(20)     DEFAULT 'MEDIUM',
    priority                    INTEGER         NOT NULL DEFAULT 3,
    -- Assignment
    assigned_to                 VARCHAR(255),
    due_date                    DATE,
    -- Tracking
    resolved_at                 TIMESTAMPTZ,
    resolved_by                 VARCHAR(255),
    resolution_notes            TEXT,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p041_cg_req_type CHECK (
        requirement_type IN ('MANDATORY', 'RECOMMENDED', 'OPTIONAL', 'BEST_PRACTICE')
    ),
    CONSTRAINT chk_p041_cg_status CHECK (
        status IN (
            'NOT_MET', 'PARTIALLY_MET', 'IN_PROGRESS', 'REMEDIATED',
            'ACCEPTED_RISK', 'NOT_APPLICABLE', 'WAIVED'
        )
    ),
    CONSTRAINT chk_p041_cg_severity CHECK (
        gap_severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFORMATIONAL')
    ),
    CONSTRAINT chk_p041_cg_remediation_cat CHECK (
        remediation_category IN (
            'DATA_COLLECTION', 'METHODOLOGY_CHANGE', 'PROCESS_IMPROVEMENT',
            'DOCUMENTATION', 'CALCULATION_FIX', 'BOUNDARY_ADJUSTMENT',
            'FACTOR_UPDATE', 'VERIFICATION', 'SYSTEM_IMPLEMENTATION',
            'TRAINING', 'OTHER'
        )
    ),
    CONSTRAINT chk_p041_cg_complexity CHECK (
        implementation_complexity IS NULL OR implementation_complexity IN (
            'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH'
        )
    ),
    CONSTRAINT chk_p041_cg_priority CHECK (
        priority >= 1 AND priority <= 5
    ),
    CONSTRAINT chk_p041_cg_effort CHECK (
        estimated_effort_hours IS NULL OR estimated_effort_hours >= 0
    ),
    CONSTRAINT chk_p041_cg_cost CHECK (
        estimated_cost IS NULL OR estimated_cost >= 0
    ),
    CONSTRAINT uq_p041_cg_framework_req UNIQUE (framework_result_id, requirement_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_cg_tenant             ON ghg_scope12.compliance_gaps(tenant_id);
CREATE INDEX idx_p041_cg_framework_result   ON ghg_scope12.compliance_gaps(framework_result_id);
CREATE INDEX idx_p041_cg_requirement_id     ON ghg_scope12.compliance_gaps(requirement_id);
CREATE INDEX idx_p041_cg_status             ON ghg_scope12.compliance_gaps(status);
CREATE INDEX idx_p041_cg_severity           ON ghg_scope12.compliance_gaps(gap_severity);
CREATE INDEX idx_p041_cg_priority           ON ghg_scope12.compliance_gaps(priority);
CREATE INDEX idx_p041_cg_assigned           ON ghg_scope12.compliance_gaps(assigned_to);
CREATE INDEX idx_p041_cg_due_date           ON ghg_scope12.compliance_gaps(due_date);
CREATE INDEX idx_p041_cg_remediation_cat    ON ghg_scope12.compliance_gaps(remediation_category);
CREATE INDEX idx_p041_cg_created            ON ghg_scope12.compliance_gaps(created_at DESC);
CREATE INDEX idx_p041_cg_metadata           ON ghg_scope12.compliance_gaps USING GIN(metadata);

-- Composite: open gaps by priority
CREATE INDEX idx_p041_cg_open_priority      ON ghg_scope12.compliance_gaps(framework_result_id, priority, gap_severity)
    WHERE status IN ('NOT_MET', 'PARTIALLY_MET', 'IN_PROGRESS');

-- Composite: critical/high severity open gaps
CREATE INDEX idx_p041_cg_critical           ON ghg_scope12.compliance_gaps(framework_result_id, due_date)
    WHERE gap_severity IN ('CRITICAL', 'HIGH') AND status IN ('NOT_MET', 'PARTIALLY_MET', 'IN_PROGRESS');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_cg_updated
    BEFORE UPDATE ON ghg_scope12.compliance_gaps
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_scope12.compliance_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.framework_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.compliance_gaps ENABLE ROW LEVEL SECURITY;

CREATE POLICY p041_ca_tenant_isolation
    ON ghg_scope12.compliance_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_ca_service_bypass
    ON ghg_scope12.compliance_assessments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_fr_tenant_isolation
    ON ghg_scope12.framework_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_fr_service_bypass
    ON ghg_scope12.framework_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_cg_tenant_isolation
    ON ghg_scope12.compliance_gaps
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_cg_service_bypass
    ON ghg_scope12.compliance_gaps
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.compliance_assessments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.framework_results TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.compliance_gaps TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_scope12.compliance_assessments IS
    'Top-level compliance assessment evaluating GHG inventory against multiple regulatory and voluntary frameworks with overall scoring and GHG Protocol principles assessment.';
COMMENT ON TABLE ghg_scope12.framework_results IS
    'Per-framework compliance scoring with requirements breakdown (met, partially met, not met, N/A) and classification.';
COMMENT ON TABLE ghg_scope12.compliance_gaps IS
    'Individual compliance gaps with requirement details, gap description, prioritized remediation actions, and assignment tracking.';

COMMENT ON COLUMN ghg_scope12.compliance_assessments.overall_score IS 'Weighted average compliance score across all assessed frameworks (0-100).';
COMMENT ON COLUMN ghg_scope12.compliance_assessments.overall_classification IS 'Overall compliance classification: FULLY_COMPLIANT (>90), SUBSTANTIALLY (70-90), PARTIALLY (50-70), NON (<50).';
COMMENT ON COLUMN ghg_scope12.compliance_assessments.completeness_score IS 'GHG Protocol principle: Completeness - are all relevant sources included?';
COMMENT ON COLUMN ghg_scope12.compliance_assessments.accuracy_score IS 'GHG Protocol principle: Accuracy - are calculations free from systematic errors?';
COMMENT ON COLUMN ghg_scope12.compliance_assessments.consistency_score IS 'GHG Protocol principle: Consistency - are methodologies applied uniformly over time?';
COMMENT ON COLUMN ghg_scope12.compliance_assessments.transparency_score IS 'GHG Protocol principle: Transparency - are assumptions and methodologies disclosed?';
COMMENT ON COLUMN ghg_scope12.compliance_assessments.relevance_score IS 'GHG Protocol principle: Relevance - does the inventory reflect actual emissions?';

COMMENT ON COLUMN ghg_scope12.framework_results.framework IS 'Reporting framework: GHG_PROTOCOL, ISO_14064_1, CSRD_ESRS_E1, CDP_CLIMATE, TCFD, SEC_CLIMATE, etc.';
COMMENT ON COLUMN ghg_scope12.framework_results.classification IS 'Framework-level compliance: FULLY_COMPLIANT, SUBSTANTIALLY_COMPLIANT, PARTIALLY_COMPLIANT, NON_COMPLIANT.';
COMMENT ON COLUMN ghg_scope12.framework_results.disclosure_readiness_pct IS 'Percentage readiness for disclosure submission to the framework authority.';

COMMENT ON COLUMN ghg_scope12.compliance_gaps.gap_severity IS 'Gap severity: CRITICAL (blocks compliance), HIGH (significant risk), MEDIUM (moderate risk), LOW (minor), INFORMATIONAL.';
COMMENT ON COLUMN ghg_scope12.compliance_gaps.priority IS 'Remediation priority 1 (highest/urgent) to 5 (lowest/deferred).';
COMMENT ON COLUMN ghg_scope12.compliance_gaps.remediation_category IS 'Category of remediation action needed: DATA_COLLECTION, METHODOLOGY_CHANGE, DOCUMENTATION, etc.';
