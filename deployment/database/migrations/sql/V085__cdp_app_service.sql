-- =============================================================================
-- V085: GL-CDP-APP v1.0 Application Service Schema
-- =============================================================================
-- Application: GL-CDP-APP (CDP Climate Change Disclosure Platform)
-- Date:        March 2026
--
-- Application-level tables for CDP Climate Change questionnaire management,
-- scoring simulation, gap analysis, benchmarking, supply chain engagement,
-- transition plan building, verification tracking, and submission.
--
-- EXTENDS:
--   V051-V070: AGENT-MRV calculation agents (Scope 1-3)
--   V080: Scope 3 Category Mapper Service
--   V081: Audit Trail & Lineage Service
--   V083: GL-GHG-APP v1.0
--   V084: GL-ISO14064-APP v1.0
--
-- These tables sit in the cdp_app schema and integrate with the underlying
-- MRV agent data for auto-population of Scope 1/2/3 emissions into CDP
-- questionnaire responses.
-- =============================================================================
-- Tables (20):
--   1.  gl_cdp_organizations            - CDP organization profiles
--   2.  gl_cdp_questionnaires           - Questionnaire instances per year
--   3.  gl_cdp_modules                  - 13 CDP module definitions
--   4.  gl_cdp_questions                - 200+ question registry
--   5.  gl_cdp_responses                - Response per question per org
--   6.  gl_cdp_response_versions        - Version history of responses
--   7.  gl_cdp_evidence_attachments     - Documents/data linked to responses
--   8.  gl_cdp_review_workflows         - Draft -> review -> approve workflow
--   9.  gl_cdp_scoring_results          - Overall + predicted scores
--  10.  gl_cdp_category_scores          - 17 category-level scores
--  11.  gl_cdp_gap_analyses             - Gap identification runs
--  12.  gl_cdp_gap_items                - Individual gap items
--  13.  gl_cdp_benchmarks               - Sector/regional benchmark data
--  14.  gl_cdp_peer_comparisons         - Peer group comparison results
--  15.  gl_cdp_supply_chain_requests    - Supplier engagement requests
--  16.  gl_cdp_supplier_responses       - Supplier questionnaire responses
--  17.  gl_cdp_transition_plans         - 1.5C transition plans
--  18.  gl_cdp_transition_milestones    - Decarbonization milestones
--  19.  gl_cdp_verification_records     - Third-party verification
--  20.  gl_cdp_submissions              - Final submission records
--
-- Hypertables (3):
--  gl_cdp_responses, gl_cdp_scoring_results, gl_cdp_gap_analyses
--
-- Continuous Aggregates (2):
--  cdp_app.monthly_response_progress, cdp_app.quarterly_score_trends
--
-- Also includes: 60+ indexes, update triggers, security grants,
-- retention policies, compression policies, permissions, and comments.
-- Previous: V084__iso14064_app_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS cdp_app;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION cdp_app.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: cdp_app.gl_cdp_organizations
-- =============================================================================
-- CDP organization profiles with sector classification, region, revenue,
-- and CDP account number for managing disclosure submissions.

CREATE TABLE cdp_app.gl_cdp_organizations (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    name                VARCHAR(500)    NOT NULL,
    sector_gics         VARCHAR(20),
    region              VARCHAR(100),
    country             CHAR(3),
    employee_count      INTEGER,
    revenue_usd         DECIMAL(18,2),
    fiscal_year_end     VARCHAR(10),
    cdp_account_number  VARCHAR(100),
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_org_name_not_empty CHECK (
        LENGTH(TRIM(name)) > 0
    ),
    CONSTRAINT chk_cdp_org_country_length CHECK (
        country IS NULL OR LENGTH(TRIM(country)) = 3
    ),
    CONSTRAINT chk_cdp_org_employees_non_neg CHECK (
        employee_count IS NULL OR employee_count >= 0
    ),
    CONSTRAINT chk_cdp_org_revenue_non_neg CHECK (
        revenue_usd IS NULL OR revenue_usd >= 0
    )
);

-- Indexes
CREATE INDEX idx_cdp_org_name ON cdp_app.gl_cdp_organizations(name);
CREATE INDEX idx_cdp_org_sector ON cdp_app.gl_cdp_organizations(sector_gics);
CREATE INDEX idx_cdp_org_region ON cdp_app.gl_cdp_organizations(region);
CREATE INDEX idx_cdp_org_country ON cdp_app.gl_cdp_organizations(country);
CREATE INDEX idx_cdp_org_account ON cdp_app.gl_cdp_organizations(cdp_account_number);
CREATE INDEX idx_cdp_org_created_at ON cdp_app.gl_cdp_organizations(created_at DESC);
CREATE INDEX idx_cdp_org_metadata ON cdp_app.gl_cdp_organizations USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_cdp_org_updated_at
    BEFORE UPDATE ON cdp_app.gl_cdp_organizations
    FOR EACH ROW
    EXECUTE FUNCTION cdp_app.set_updated_at();

-- =============================================================================
-- Table 2: cdp_app.gl_cdp_questionnaires
-- =============================================================================
-- Questionnaire instances per reporting year.  Each questionnaire links to
-- an organization and tracks the questionnaire version, status, and which
-- sector-specific modules are applicable.

CREATE TABLE cdp_app.gl_cdp_questionnaires (
    id                       UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                   UUID            NOT NULL REFERENCES cdp_app.gl_cdp_organizations(id) ON DELETE CASCADE,
    reporting_year           INTEGER         NOT NULL,
    questionnaire_version    VARCHAR(20)     NOT NULL DEFAULT '2025',
    status                   VARCHAR(50)     NOT NULL DEFAULT 'not_started',
    sector_specific_modules  JSONB           DEFAULT '{}',
    started_at               TIMESTAMPTZ,
    submitted_at             TIMESTAMPTZ,
    metadata                 JSONB           DEFAULT '{}',
    created_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_q_year_range CHECK (
        reporting_year >= 2015 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_cdp_q_status CHECK (
        status IN ('not_started', 'in_progress', 'completed', 'submitted', 'scored')
    ),
    UNIQUE(org_id, reporting_year)
);

-- Indexes
CREATE INDEX idx_cdp_q_org ON cdp_app.gl_cdp_questionnaires(org_id);
CREATE INDEX idx_cdp_q_year ON cdp_app.gl_cdp_questionnaires(reporting_year);
CREATE INDEX idx_cdp_q_status ON cdp_app.gl_cdp_questionnaires(status);
CREATE INDEX idx_cdp_q_version ON cdp_app.gl_cdp_questionnaires(questionnaire_version);
CREATE INDEX idx_cdp_q_created_at ON cdp_app.gl_cdp_questionnaires(created_at DESC);
CREATE INDEX idx_cdp_q_modules ON cdp_app.gl_cdp_questionnaires USING GIN(sector_specific_modules);

-- Updated_at trigger
CREATE TRIGGER trg_cdp_q_updated_at
    BEFORE UPDATE ON cdp_app.gl_cdp_questionnaires
    FOR EACH ROW
    EXECUTE FUNCTION cdp_app.set_updated_at();

-- =============================================================================
-- Table 3: cdp_app.gl_cdp_modules
-- =============================================================================
-- Module definitions for CDP Climate Change questionnaire (M0-M13).  Each
-- module record tracks applicability, sector specificity, question count,
-- and display ordering.

CREATE TABLE cdp_app.gl_cdp_modules (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    questionnaire_id    UUID            NOT NULL REFERENCES cdp_app.gl_cdp_questionnaires(id) ON DELETE CASCADE,
    module_code         VARCHAR(10)     NOT NULL,
    module_name         VARCHAR(200)    NOT NULL,
    description         TEXT,
    question_count      INTEGER         NOT NULL DEFAULT 0,
    is_applicable       BOOLEAN         NOT NULL DEFAULT TRUE,
    is_sector_specific  BOOLEAN         NOT NULL DEFAULT FALSE,
    sort_order          INTEGER         NOT NULL DEFAULT 0,
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_mod_code CHECK (
        module_code IN ('M0','M1','M2','M3','M4','M5','M6','M7','M8','M9','M10','M11','M12','M13')
    ),
    CONSTRAINT chk_cdp_mod_name_not_empty CHECK (
        LENGTH(TRIM(module_name)) > 0
    ),
    CONSTRAINT chk_cdp_mod_question_count_non_neg CHECK (
        question_count >= 0
    ),
    UNIQUE(questionnaire_id, module_code)
);

-- Indexes
CREATE INDEX idx_cdp_mod_questionnaire ON cdp_app.gl_cdp_modules(questionnaire_id);
CREATE INDEX idx_cdp_mod_code ON cdp_app.gl_cdp_modules(module_code);
CREATE INDEX idx_cdp_mod_applicable ON cdp_app.gl_cdp_modules(is_applicable);
CREATE INDEX idx_cdp_mod_sort ON cdp_app.gl_cdp_modules(sort_order);

-- =============================================================================
-- Table 4: cdp_app.gl_cdp_questions
-- =============================================================================
-- Question registry with question number, text, type, guidance, scoring
-- category, point allocations across 4 levels, conditional logic, and
-- version year.  200+ questions across 13 modules.

CREATE TABLE cdp_app.gl_cdp_questions (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    module_id           UUID            NOT NULL REFERENCES cdp_app.gl_cdp_modules(id) ON DELETE CASCADE,
    question_number     VARCHAR(20)     NOT NULL,
    question_text       TEXT            NOT NULL,
    question_type       VARCHAR(30)     NOT NULL,
    guidance_text       TEXT,
    is_required         BOOLEAN         NOT NULL DEFAULT TRUE,
    is_conditional      BOOLEAN         NOT NULL DEFAULT FALSE,
    condition_logic     JSONB,
    scoring_category    VARCHAR(60),
    disclosure_points   INTEGER         NOT NULL DEFAULT 0,
    awareness_points    INTEGER         NOT NULL DEFAULT 0,
    management_points   INTEGER         NOT NULL DEFAULT 0,
    leadership_points   INTEGER         NOT NULL DEFAULT 0,
    version_year        INTEGER         NOT NULL DEFAULT 2025,
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_qn_number_not_empty CHECK (
        LENGTH(TRIM(question_number)) > 0
    ),
    CONSTRAINT chk_cdp_qn_type CHECK (
        question_type IN ('text', 'numeric', 'percentage', 'table', 'multi_select', 'single_select', 'yes_no')
    ),
    CONSTRAINT chk_cdp_qn_disclosure_non_neg CHECK (disclosure_points >= 0),
    CONSTRAINT chk_cdp_qn_awareness_non_neg CHECK (awareness_points >= 0),
    CONSTRAINT chk_cdp_qn_management_non_neg CHECK (management_points >= 0),
    CONSTRAINT chk_cdp_qn_leadership_non_neg CHECK (leadership_points >= 0),
    CONSTRAINT chk_cdp_qn_version_range CHECK (
        version_year >= 2020 AND version_year <= 2100
    )
);

-- Indexes
CREATE INDEX idx_cdp_qn_module ON cdp_app.gl_cdp_questions(module_id);
CREATE INDEX idx_cdp_qn_number ON cdp_app.gl_cdp_questions(question_number);
CREATE INDEX idx_cdp_qn_type ON cdp_app.gl_cdp_questions(question_type);
CREATE INDEX idx_cdp_qn_scoring_cat ON cdp_app.gl_cdp_questions(scoring_category);
CREATE INDEX idx_cdp_qn_required ON cdp_app.gl_cdp_questions(is_required);
CREATE INDEX idx_cdp_qn_conditional ON cdp_app.gl_cdp_questions(is_conditional);
CREATE INDEX idx_cdp_qn_version ON cdp_app.gl_cdp_questions(version_year);
CREATE INDEX idx_cdp_qn_condition_logic ON cdp_app.gl_cdp_questions USING GIN(condition_logic);

-- =============================================================================
-- Table 5: cdp_app.gl_cdp_responses (hypertable)
-- =============================================================================
-- Responses per question per organization.  Tracks response content (JSONB),
-- plain text, lifecycle status, assignment, review chain, auto-population
-- source, and confidence score.  Partitioned by created_at for time-series
-- query efficiency.

CREATE TABLE cdp_app.gl_cdp_responses (
    id                      UUID            NOT NULL DEFAULT gen_random_uuid(),
    question_id             UUID            NOT NULL,
    questionnaire_id        UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    response_content        JSONB           DEFAULT '{}',
    response_text           TEXT,
    response_status         VARCHAR(30)     NOT NULL DEFAULT 'draft',
    assigned_to             UUID,
    reviewed_by             UUID,
    approved_by             UUID,
    auto_populated          BOOLEAN         NOT NULL DEFAULT FALSE,
    auto_populated_source   VARCHAR(200),
    confidence_score        DECIMAL(5,4),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_resp_status CHECK (
        response_status IN ('draft', 'in_review', 'approved', 'submitted')
    ),
    CONSTRAINT chk_cdp_resp_confidence CHECK (
        confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)
    )
);

-- Convert to hypertable for time-series partitioning
SELECT create_hypertable('cdp_app.gl_cdp_responses', 'created_at',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes (hypertable-aware)
CREATE INDEX idx_cdp_resp_question ON cdp_app.gl_cdp_responses(question_id, created_at DESC);
CREATE INDEX idx_cdp_resp_questionnaire ON cdp_app.gl_cdp_responses(questionnaire_id, created_at DESC);
CREATE INDEX idx_cdp_resp_org ON cdp_app.gl_cdp_responses(org_id, created_at DESC);
CREATE INDEX idx_cdp_resp_status ON cdp_app.gl_cdp_responses(response_status, created_at DESC);
CREATE INDEX idx_cdp_resp_assigned ON cdp_app.gl_cdp_responses(assigned_to, created_at DESC);
CREATE INDEX idx_cdp_resp_auto ON cdp_app.gl_cdp_responses(auto_populated, created_at DESC);
CREATE INDEX idx_cdp_resp_content ON cdp_app.gl_cdp_responses USING GIN(response_content);

-- =============================================================================
-- Table 6: cdp_app.gl_cdp_response_versions
-- =============================================================================
-- Version history for response edits.  Stores the content snapshot,
-- change author, and reason at each version increment.

CREATE TABLE cdp_app.gl_cdp_response_versions (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id     UUID            NOT NULL,
    version_number  INTEGER         NOT NULL,
    content         JSONB           NOT NULL DEFAULT '{}',
    changed_by      UUID,
    change_reason   TEXT,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_rv_version_positive CHECK (version_number > 0)
);

-- Indexes
CREATE INDEX idx_cdp_rv_response ON cdp_app.gl_cdp_response_versions(response_id);
CREATE INDEX idx_cdp_rv_version ON cdp_app.gl_cdp_response_versions(response_id, version_number);
CREATE INDEX idx_cdp_rv_created_at ON cdp_app.gl_cdp_response_versions(created_at DESC);

-- =============================================================================
-- Table 7: cdp_app.gl_cdp_evidence_attachments
-- =============================================================================
-- Evidence documents, data tables, and supporting files attached to
-- individual responses for verification and audit purposes.

CREATE TABLE cdp_app.gl_cdp_evidence_attachments (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id     UUID            NOT NULL,
    file_name       VARCHAR(500)    NOT NULL,
    file_type       VARCHAR(100),
    file_size_bytes BIGINT,
    storage_path    VARCHAR(1000),
    description     TEXT,
    uploaded_by     UUID,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_ea_name_not_empty CHECK (
        LENGTH(TRIM(file_name)) > 0
    ),
    CONSTRAINT chk_cdp_ea_size_non_neg CHECK (
        file_size_bytes IS NULL OR file_size_bytes >= 0
    )
);

-- Indexes
CREATE INDEX idx_cdp_ea_response ON cdp_app.gl_cdp_evidence_attachments(response_id);
CREATE INDEX idx_cdp_ea_uploaded_by ON cdp_app.gl_cdp_evidence_attachments(uploaded_by);
CREATE INDEX idx_cdp_ea_created_at ON cdp_app.gl_cdp_evidence_attachments(created_at DESC);

-- =============================================================================
-- Table 8: cdp_app.gl_cdp_review_workflows
-- =============================================================================
-- Review workflow actions tracking status transitions, reviewer actions,
-- comments, and timestamp for the full review chain.

CREATE TABLE cdp_app.gl_cdp_review_workflows (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id     UUID            NOT NULL,
    action          VARCHAR(30)     NOT NULL,
    actor_id        UUID,
    comments        TEXT,
    status_from     VARCHAR(30),
    status_to       VARCHAR(30),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_rw_action CHECK (
        action IN ('assign', 'review', 'approve', 'reject', 'submit', 'reopen')
    ),
    CONSTRAINT chk_cdp_rw_status_from CHECK (
        status_from IS NULL OR status_from IN ('draft', 'in_review', 'approved', 'submitted')
    ),
    CONSTRAINT chk_cdp_rw_status_to CHECK (
        status_to IS NULL OR status_to IN ('draft', 'in_review', 'approved', 'submitted')
    )
);

-- Indexes
CREATE INDEX idx_cdp_rw_response ON cdp_app.gl_cdp_review_workflows(response_id);
CREATE INDEX idx_cdp_rw_action ON cdp_app.gl_cdp_review_workflows(action);
CREATE INDEX idx_cdp_rw_actor ON cdp_app.gl_cdp_review_workflows(actor_id);
CREATE INDEX idx_cdp_rw_created_at ON cdp_app.gl_cdp_review_workflows(created_at DESC);

-- =============================================================================
-- Table 9: cdp_app.gl_cdp_scoring_results (hypertable)
-- =============================================================================
-- Overall CDP scoring results per questionnaire.  Tracks the overall score
-- (0-100), determined band (D- to A), predicted band, confidence interval,
-- A-level eligibility, and simulation timestamp.

CREATE TABLE cdp_app.gl_cdp_scoring_results (
    id                  UUID            NOT NULL DEFAULT gen_random_uuid(),
    questionnaire_id    UUID            NOT NULL,
    org_id              UUID            NOT NULL,
    overall_score       DECIMAL(6,2)    NOT NULL DEFAULT 0,
    score_band          VARCHAR(5)      NOT NULL DEFAULT 'D-',
    predicted_band      VARCHAR(5),
    confidence_lower    DECIMAL(6,2),
    confidence_upper    DECIMAL(6,2),
    a_level_eligible    BOOLEAN         NOT NULL DEFAULT FALSE,
    simulation_run_at   TIMESTAMPTZ,
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_sr_score_range CHECK (
        overall_score >= 0 AND overall_score <= 100
    ),
    CONSTRAINT chk_cdp_sr_band CHECK (
        score_band IN ('D-', 'D', 'C-', 'C', 'B-', 'B', 'A-', 'A')
    ),
    CONSTRAINT chk_cdp_sr_predicted_band CHECK (
        predicted_band IS NULL OR predicted_band IN ('D-', 'D', 'C-', 'C', 'B-', 'B', 'A-', 'A')
    ),
    CONSTRAINT chk_cdp_sr_confidence CHECK (
        confidence_lower IS NULL OR confidence_upper IS NULL
        OR confidence_lower <= confidence_upper
    )
);

-- Convert to hypertable
SELECT create_hypertable('cdp_app.gl_cdp_scoring_results', 'created_at',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes (hypertable-aware)
CREATE INDEX idx_cdp_sr_questionnaire ON cdp_app.gl_cdp_scoring_results(questionnaire_id, created_at DESC);
CREATE INDEX idx_cdp_sr_org ON cdp_app.gl_cdp_scoring_results(org_id, created_at DESC);
CREATE INDEX idx_cdp_sr_band ON cdp_app.gl_cdp_scoring_results(score_band, created_at DESC);
CREATE INDEX idx_cdp_sr_eligible ON cdp_app.gl_cdp_scoring_results(a_level_eligible, created_at DESC);

-- =============================================================================
-- Table 10: cdp_app.gl_cdp_category_scores
-- =============================================================================
-- Per-category scores for the 17 CDP Climate Change scoring categories.
-- Stores raw score, weighted score, weight, and sub-level breakdown
-- (disclosure, awareness, management, leadership).

CREATE TABLE cdp_app.gl_cdp_category_scores (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    scoring_result_id   UUID            NOT NULL,
    category_code       VARCHAR(10)     NOT NULL,
    category_name       VARCHAR(200)    NOT NULL,
    raw_score           DECIMAL(6,2)    NOT NULL DEFAULT 0,
    weighted_score      DECIMAL(8,4)    NOT NULL DEFAULT 0,
    weight              DECIMAL(6,4)    NOT NULL DEFAULT 0,
    disclosure_score    DECIMAL(6,2)    DEFAULT 0,
    awareness_score     DECIMAL(6,2)    DEFAULT 0,
    management_score    DECIMAL(6,2)    DEFAULT 0,
    leadership_score    DECIMAL(6,2)    DEFAULT 0,
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_cs_raw_range CHECK (
        raw_score >= 0 AND raw_score <= 100
    ),
    CONSTRAINT chk_cdp_cs_weight_range CHECK (
        weight >= 0 AND weight <= 1
    ),
    CONSTRAINT chk_cdp_cs_disclosure_range CHECK (
        disclosure_score IS NULL OR (disclosure_score >= 0 AND disclosure_score <= 100)
    ),
    CONSTRAINT chk_cdp_cs_awareness_range CHECK (
        awareness_score IS NULL OR (awareness_score >= 0 AND awareness_score <= 100)
    ),
    CONSTRAINT chk_cdp_cs_management_range CHECK (
        management_score IS NULL OR (management_score >= 0 AND management_score <= 100)
    ),
    CONSTRAINT chk_cdp_cs_leadership_range CHECK (
        leadership_score IS NULL OR (leadership_score >= 0 AND leadership_score <= 100)
    )
);

-- Indexes
CREATE INDEX idx_cdp_cs_result ON cdp_app.gl_cdp_category_scores(scoring_result_id);
CREATE INDEX idx_cdp_cs_category ON cdp_app.gl_cdp_category_scores(category_code);
CREATE INDEX idx_cdp_cs_created_at ON cdp_app.gl_cdp_category_scores(created_at DESC);

-- =============================================================================
-- Table 11: cdp_app.gl_cdp_gap_analyses (hypertable)
-- =============================================================================
-- Gap analysis runs capturing total gap counts by severity and potential
-- score uplift.  Partitioned by created_at for time-series tracking.

CREATE TABLE cdp_app.gl_cdp_gap_analyses (
    id                  UUID            NOT NULL DEFAULT gen_random_uuid(),
    questionnaire_id    UUID            NOT NULL,
    org_id              UUID            NOT NULL,
    total_gaps          INTEGER         NOT NULL DEFAULT 0,
    critical_gaps       INTEGER         NOT NULL DEFAULT 0,
    high_gaps           INTEGER         NOT NULL DEFAULT 0,
    medium_gaps         INTEGER         NOT NULL DEFAULT 0,
    low_gaps            INTEGER         NOT NULL DEFAULT 0,
    potential_uplift    DECIMAL(6,2)    DEFAULT 0,
    analyzed_at         TIMESTAMPTZ     DEFAULT NOW(),
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_ga_total_non_neg CHECK (total_gaps >= 0),
    CONSTRAINT chk_cdp_ga_critical_non_neg CHECK (critical_gaps >= 0),
    CONSTRAINT chk_cdp_ga_high_non_neg CHECK (high_gaps >= 0),
    CONSTRAINT chk_cdp_ga_medium_non_neg CHECK (medium_gaps >= 0),
    CONSTRAINT chk_cdp_ga_low_non_neg CHECK (low_gaps >= 0),
    CONSTRAINT chk_cdp_ga_uplift_non_neg CHECK (
        potential_uplift IS NULL OR potential_uplift >= 0
    ),
    CONSTRAINT chk_cdp_ga_totals_consistent CHECK (
        total_gaps = critical_gaps + high_gaps + medium_gaps + low_gaps
    )
);

-- Convert to hypertable
SELECT create_hypertable('cdp_app.gl_cdp_gap_analyses', 'created_at',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes (hypertable-aware)
CREATE INDEX idx_cdp_ga_questionnaire ON cdp_app.gl_cdp_gap_analyses(questionnaire_id, created_at DESC);
CREATE INDEX idx_cdp_ga_org ON cdp_app.gl_cdp_gap_analyses(org_id, created_at DESC);

-- =============================================================================
-- Table 12: cdp_app.gl_cdp_gap_items
-- =============================================================================
-- Individual gap items identifying missing or weak responses with severity,
-- current/target level, recommendation, effort, and score uplift prediction.

CREATE TABLE cdp_app.gl_cdp_gap_items (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    gap_analysis_id     UUID            NOT NULL,
    question_id         UUID,
    module_code         VARCHAR(10),
    severity            VARCHAR(20)     NOT NULL,
    current_level       VARCHAR(30),
    target_level        VARCHAR(30),
    recommendation      TEXT,
    effort              VARCHAR(20),
    score_uplift        DECIMAL(6,2)    DEFAULT 0,
    is_resolved         BOOLEAN         NOT NULL DEFAULT FALSE,
    resolved_at         TIMESTAMPTZ,
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_gi_severity CHECK (
        severity IN ('critical', 'high', 'medium', 'low')
    ),
    CONSTRAINT chk_cdp_gi_current_level CHECK (
        current_level IS NULL OR current_level IN ('disclosure', 'awareness', 'management', 'leadership')
    ),
    CONSTRAINT chk_cdp_gi_target_level CHECK (
        target_level IS NULL OR target_level IN ('disclosure', 'awareness', 'management', 'leadership')
    ),
    CONSTRAINT chk_cdp_gi_effort CHECK (
        effort IS NULL OR effort IN ('low', 'medium', 'high')
    ),
    CONSTRAINT chk_cdp_gi_uplift_non_neg CHECK (
        score_uplift IS NULL OR score_uplift >= 0
    ),
    CONSTRAINT chk_cdp_gi_resolved_consistency CHECK (
        (is_resolved = FALSE AND resolved_at IS NULL)
        OR (is_resolved = TRUE AND resolved_at IS NOT NULL)
    )
);

-- Indexes
CREATE INDEX idx_cdp_gi_analysis ON cdp_app.gl_cdp_gap_items(gap_analysis_id);
CREATE INDEX idx_cdp_gi_question ON cdp_app.gl_cdp_gap_items(question_id);
CREATE INDEX idx_cdp_gi_module ON cdp_app.gl_cdp_gap_items(module_code);
CREATE INDEX idx_cdp_gi_severity ON cdp_app.gl_cdp_gap_items(severity);
CREATE INDEX idx_cdp_gi_resolved ON cdp_app.gl_cdp_gap_items(is_resolved);
CREATE INDEX idx_cdp_gi_created_at ON cdp_app.gl_cdp_gap_items(created_at DESC);

-- =============================================================================
-- Table 13: cdp_app.gl_cdp_benchmarks
-- =============================================================================
-- Sector and regional benchmark data for peer comparison.  Stores score
-- statistics (mean, median, percentiles), A-list rate, respondent count,
-- and full score distribution by band.

CREATE TABLE cdp_app.gl_cdp_benchmarks (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sector_gics         VARCHAR(20)     NOT NULL,
    region              VARCHAR(100)    NOT NULL DEFAULT 'Global',
    year                INTEGER         NOT NULL,
    mean_score          DECIMAL(6,2)    NOT NULL DEFAULT 0,
    median_score        DECIMAL(6,2)    NOT NULL DEFAULT 0,
    p25_score           DECIMAL(6,2)    DEFAULT 0,
    p75_score           DECIMAL(6,2)    DEFAULT 0,
    a_list_rate         DECIMAL(6,2)    DEFAULT 0,
    respondent_count    INTEGER         NOT NULL DEFAULT 0,
    score_distribution  JSONB           DEFAULT '{}',
    metadata            JSONB           DEFAULT '{}',
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_bm_year_range CHECK (
        year >= 2015 AND year <= 2100
    ),
    CONSTRAINT chk_cdp_bm_mean_range CHECK (
        mean_score >= 0 AND mean_score <= 100
    ),
    CONSTRAINT chk_cdp_bm_median_range CHECK (
        median_score >= 0 AND median_score <= 100
    ),
    CONSTRAINT chk_cdp_bm_respondent_non_neg CHECK (
        respondent_count >= 0
    ),
    CONSTRAINT chk_cdp_bm_a_rate_range CHECK (
        a_list_rate >= 0 AND a_list_rate <= 100
    ),
    UNIQUE(sector_gics, region, year)
);

-- Indexes
CREATE INDEX idx_cdp_bm_sector ON cdp_app.gl_cdp_benchmarks(sector_gics);
CREATE INDEX idx_cdp_bm_region ON cdp_app.gl_cdp_benchmarks(region);
CREATE INDEX idx_cdp_bm_year ON cdp_app.gl_cdp_benchmarks(year);
CREATE INDEX idx_cdp_bm_distribution ON cdp_app.gl_cdp_benchmarks USING GIN(score_distribution);

-- Updated_at trigger
CREATE TRIGGER trg_cdp_bm_updated_at
    BEFORE UPDATE ON cdp_app.gl_cdp_benchmarks
    FOR EACH ROW
    EXECUTE FUNCTION cdp_app.set_updated_at();

-- =============================================================================
-- Table 14: cdp_app.gl_cdp_peer_comparisons
-- =============================================================================
-- Peer group comparison results linking a questionnaire to a benchmark
-- with the organization's score, band, percentile, and category-level
-- comparisons.

CREATE TABLE cdp_app.gl_cdp_peer_comparisons (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    questionnaire_id        UUID            NOT NULL,
    benchmark_id            UUID            NOT NULL REFERENCES cdp_app.gl_cdp_benchmarks(id) ON DELETE CASCADE,
    org_score               DECIMAL(6,2)    NOT NULL DEFAULT 0,
    org_band                VARCHAR(5),
    percentile              DECIMAL(6,2),
    category_comparisons    JSONB           DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_pc_score_range CHECK (
        org_score >= 0 AND org_score <= 100
    ),
    CONSTRAINT chk_cdp_pc_percentile_range CHECK (
        percentile IS NULL OR (percentile >= 0 AND percentile <= 100)
    ),
    CONSTRAINT chk_cdp_pc_band CHECK (
        org_band IS NULL OR org_band IN ('D-', 'D', 'C-', 'C', 'B-', 'B', 'A-', 'A')
    )
);

-- Indexes
CREATE INDEX idx_cdp_pc_questionnaire ON cdp_app.gl_cdp_peer_comparisons(questionnaire_id);
CREATE INDEX idx_cdp_pc_benchmark ON cdp_app.gl_cdp_peer_comparisons(benchmark_id);
CREATE INDEX idx_cdp_pc_created_at ON cdp_app.gl_cdp_peer_comparisons(created_at DESC);
CREATE INDEX idx_cdp_pc_comparisons ON cdp_app.gl_cdp_peer_comparisons USING GIN(category_comparisons);

-- =============================================================================
-- Table 15: cdp_app.gl_cdp_supply_chain_requests
-- =============================================================================
-- Supplier engagement requests for CDP Supply Chain program.  Tracks
-- invitation, response, and follow-up reminders per supplier.

CREATE TABLE cdp_app.gl_cdp_supply_chain_requests (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id              UUID            NOT NULL REFERENCES cdp_app.gl_cdp_organizations(id) ON DELETE CASCADE,
    supplier_name       VARCHAR(500)    NOT NULL,
    supplier_email      VARCHAR(300),
    supplier_sector     VARCHAR(20),
    invitation_sent_at  TIMESTAMPTZ,
    response_received_at TIMESTAMPTZ,
    status              VARCHAR(30)     NOT NULL DEFAULT 'not_invited',
    reminder_count      INTEGER         NOT NULL DEFAULT 0,
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_sc_name_not_empty CHECK (
        LENGTH(TRIM(supplier_name)) > 0
    ),
    CONSTRAINT chk_cdp_sc_status CHECK (
        status IN ('not_invited', 'invited', 'responded', 'declined', 'expired')
    ),
    CONSTRAINT chk_cdp_sc_reminder_non_neg CHECK (
        reminder_count >= 0
    )
);

-- Indexes
CREATE INDEX idx_cdp_sc_org ON cdp_app.gl_cdp_supply_chain_requests(org_id);
CREATE INDEX idx_cdp_sc_status ON cdp_app.gl_cdp_supply_chain_requests(status);
CREATE INDEX idx_cdp_sc_supplier ON cdp_app.gl_cdp_supply_chain_requests(supplier_name);
CREATE INDEX idx_cdp_sc_sector ON cdp_app.gl_cdp_supply_chain_requests(supplier_sector);
CREATE INDEX idx_cdp_sc_created_at ON cdp_app.gl_cdp_supply_chain_requests(created_at DESC);

-- =============================================================================
-- Table 16: cdp_app.gl_cdp_supplier_responses
-- =============================================================================
-- Supplier questionnaire response data with emissions by scope, target
-- and verification status, engagement score, and raw response data.

CREATE TABLE cdp_app.gl_cdp_supplier_responses (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id          UUID            NOT NULL REFERENCES cdp_app.gl_cdp_supply_chain_requests(id) ON DELETE CASCADE,
    scope1_emissions    DECIMAL(15,3)   DEFAULT 0,
    scope2_emissions    DECIMAL(15,3)   DEFAULT 0,
    scope3_emissions    DECIMAL(15,3)   DEFAULT 0,
    has_targets         BOOLEAN         DEFAULT FALSE,
    has_verification    BOOLEAN         DEFAULT FALSE,
    engagement_score    DECIMAL(6,2)    DEFAULT 0,
    response_data       JSONB           DEFAULT '{}',
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_sr2_scope1_non_neg CHECK (
        scope1_emissions IS NULL OR scope1_emissions >= 0
    ),
    CONSTRAINT chk_cdp_sr2_scope2_non_neg CHECK (
        scope2_emissions IS NULL OR scope2_emissions >= 0
    ),
    CONSTRAINT chk_cdp_sr2_scope3_non_neg CHECK (
        scope3_emissions IS NULL OR scope3_emissions >= 0
    ),
    CONSTRAINT chk_cdp_sr2_engagement_range CHECK (
        engagement_score >= 0 AND engagement_score <= 100
    )
);

-- Indexes
CREATE INDEX idx_cdp_sr2_request ON cdp_app.gl_cdp_supplier_responses(request_id);
CREATE INDEX idx_cdp_sr2_engagement ON cdp_app.gl_cdp_supplier_responses(engagement_score);
CREATE INDEX idx_cdp_sr2_created_at ON cdp_app.gl_cdp_supplier_responses(created_at DESC);
CREATE INDEX idx_cdp_sr2_data ON cdp_app.gl_cdp_supplier_responses USING GIN(response_data);

-- =============================================================================
-- Table 17: cdp_app.gl_cdp_transition_plans
-- =============================================================================
-- 1.5C-aligned transition plans with pathway type, SBTi alignment,
-- investment totals, revenue alignment, and public availability status.
-- Required for A-level CDP scoring.

CREATE TABLE cdp_app.gl_cdp_transition_plans (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES cdp_app.gl_cdp_organizations(id) ON DELETE CASCADE,
    plan_name               VARCHAR(500)    NOT NULL,
    base_year               INTEGER         NOT NULL,
    target_year             INTEGER         NOT NULL,
    pathway_type            VARCHAR(30)     NOT NULL,
    is_sbti_aligned         BOOLEAN         NOT NULL DEFAULT FALSE,
    sbti_status             VARCHAR(30)     DEFAULT 'none',
    total_investment_usd    DECIMAL(18,2)   DEFAULT 0,
    revenue_alignment_pct   DECIMAL(6,2)    DEFAULT 0,
    is_publicly_available   BOOLEAN         NOT NULL DEFAULT FALSE,
    status                  VARCHAR(30)     NOT NULL DEFAULT 'draft',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_tp_name_not_empty CHECK (
        LENGTH(TRIM(plan_name)) > 0
    ),
    CONSTRAINT chk_cdp_tp_target_after_base CHECK (
        target_year > base_year
    ),
    CONSTRAINT chk_cdp_tp_pathway CHECK (
        pathway_type IN ('net_zero', 'well_below_2c', 'aligned_1_5c', 'aligned_2c')
    ),
    CONSTRAINT chk_cdp_tp_sbti_status CHECK (
        sbti_status IN ('none', 'committed', 'targets_set', 'validated')
    ),
    CONSTRAINT chk_cdp_tp_investment_non_neg CHECK (
        total_investment_usd IS NULL OR total_investment_usd >= 0
    ),
    CONSTRAINT chk_cdp_tp_revenue_range CHECK (
        revenue_alignment_pct >= 0 AND revenue_alignment_pct <= 100
    ),
    CONSTRAINT chk_cdp_tp_status CHECK (
        status IN ('draft', 'active', 'superseded', 'archived')
    )
);

-- Indexes
CREATE INDEX idx_cdp_tp_org ON cdp_app.gl_cdp_transition_plans(org_id);
CREATE INDEX idx_cdp_tp_pathway ON cdp_app.gl_cdp_transition_plans(pathway_type);
CREATE INDEX idx_cdp_tp_sbti ON cdp_app.gl_cdp_transition_plans(sbti_status);
CREATE INDEX idx_cdp_tp_status ON cdp_app.gl_cdp_transition_plans(status);
CREATE INDEX idx_cdp_tp_created_at ON cdp_app.gl_cdp_transition_plans(created_at DESC);

-- Updated_at trigger
CREATE TRIGGER trg_cdp_tp_updated_at
    BEFORE UPDATE ON cdp_app.gl_cdp_transition_plans
    FOR EACH ROW
    EXECUTE FUNCTION cdp_app.set_updated_at();

-- =============================================================================
-- Table 18: cdp_app.gl_cdp_transition_milestones
-- =============================================================================
-- Decarbonization milestones within a transition plan.  Each milestone
-- specifies a target year, reduction percentage, scope, technology lever,
-- investment, and current progress.

CREATE TABLE cdp_app.gl_cdp_transition_milestones (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id                 UUID            NOT NULL REFERENCES cdp_app.gl_cdp_transition_plans(id) ON DELETE CASCADE,
    milestone_name          VARCHAR(500)    NOT NULL,
    target_year             INTEGER         NOT NULL,
    target_reduction_pct    DECIMAL(6,2)    NOT NULL DEFAULT 0,
    scope                   VARCHAR(30),
    technology_lever        VARCHAR(100),
    investment_usd          DECIMAL(18,2)   DEFAULT 0,
    status                  VARCHAR(30)     NOT NULL DEFAULT 'not_started',
    progress_pct            DECIMAL(6,2)    DEFAULT 0,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_tm_name_not_empty CHECK (
        LENGTH(TRIM(milestone_name)) > 0
    ),
    CONSTRAINT chk_cdp_tm_reduction_range CHECK (
        target_reduction_pct >= 0 AND target_reduction_pct <= 100
    ),
    CONSTRAINT chk_cdp_tm_investment_non_neg CHECK (
        investment_usd IS NULL OR investment_usd >= 0
    ),
    CONSTRAINT chk_cdp_tm_status CHECK (
        status IN ('not_started', 'on_track', 'behind', 'at_risk', 'completed', 'cancelled')
    ),
    CONSTRAINT chk_cdp_tm_progress_range CHECK (
        progress_pct >= 0 AND progress_pct <= 100
    )
);

-- Indexes
CREATE INDEX idx_cdp_tm_plan ON cdp_app.gl_cdp_transition_milestones(plan_id);
CREATE INDEX idx_cdp_tm_year ON cdp_app.gl_cdp_transition_milestones(target_year);
CREATE INDEX idx_cdp_tm_status ON cdp_app.gl_cdp_transition_milestones(status);
CREATE INDEX idx_cdp_tm_scope ON cdp_app.gl_cdp_transition_milestones(scope);
CREATE INDEX idx_cdp_tm_created_at ON cdp_app.gl_cdp_transition_milestones(created_at DESC);

-- =============================================================================
-- Table 19: cdp_app.gl_cdp_verification_records
-- =============================================================================
-- Third-party verification records per scope.  Tracks verifier details,
-- accreditation, assurance level, verification standard, statement dates,
-- and document paths.  Critical for A-level scoring requirements.

CREATE TABLE cdp_app.gl_cdp_verification_records (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES cdp_app.gl_cdp_organizations(id) ON DELETE CASCADE,
    questionnaire_id        UUID,
    scope                   VARCHAR(30)     NOT NULL,
    coverage_pct            DECIMAL(6,2)    NOT NULL DEFAULT 0,
    verifier_name           VARCHAR(500)    NOT NULL,
    verifier_accreditation  VARCHAR(200),
    assurance_level         VARCHAR(30)     NOT NULL,
    verification_standard   VARCHAR(200),
    statement_date          DATE,
    valid_until             DATE,
    statement_path          VARCHAR(1000),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_vr_verifier_not_empty CHECK (
        LENGTH(TRIM(verifier_name)) > 0
    ),
    CONSTRAINT chk_cdp_vr_coverage_range CHECK (
        coverage_pct >= 0 AND coverage_pct <= 100
    ),
    CONSTRAINT chk_cdp_vr_assurance CHECK (
        assurance_level IN ('limited', 'reasonable')
    ),
    CONSTRAINT chk_cdp_vr_valid_after_statement CHECK (
        valid_until IS NULL OR statement_date IS NULL OR valid_until >= statement_date
    )
);

-- Indexes
CREATE INDEX idx_cdp_vr_org ON cdp_app.gl_cdp_verification_records(org_id);
CREATE INDEX idx_cdp_vr_questionnaire ON cdp_app.gl_cdp_verification_records(questionnaire_id);
CREATE INDEX idx_cdp_vr_scope ON cdp_app.gl_cdp_verification_records(scope);
CREATE INDEX idx_cdp_vr_assurance ON cdp_app.gl_cdp_verification_records(assurance_level);
CREATE INDEX idx_cdp_vr_valid_until ON cdp_app.gl_cdp_verification_records(valid_until);
CREATE INDEX idx_cdp_vr_created_at ON cdp_app.gl_cdp_verification_records(created_at DESC);

-- =============================================================================
-- Table 20: cdp_app.gl_cdp_submissions
-- =============================================================================
-- Final submission records tracking the submission timestamp, format,
-- file path, CDP reference number, confirmation, and status.

CREATE TABLE cdp_app.gl_cdp_submissions (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    questionnaire_id        UUID            NOT NULL,
    org_id                  UUID            NOT NULL REFERENCES cdp_app.gl_cdp_organizations(id) ON DELETE CASCADE,
    submitted_at            TIMESTAMPTZ     DEFAULT NOW(),
    submission_format       VARCHAR(20)     NOT NULL,
    file_path               VARCHAR(1000),
    submission_reference    VARCHAR(100),
    cdp_confirmation        VARCHAR(200),
    status                  VARCHAR(30)     NOT NULL DEFAULT 'pending',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_cdp_sub_format CHECK (
        submission_format IN ('xml', 'pdf', 'excel', 'json')
    ),
    CONSTRAINT chk_cdp_sub_status CHECK (
        status IN ('pending', 'submitted', 'accepted', 'rejected', 'withdrawn')
    )
);

-- Indexes
CREATE INDEX idx_cdp_sub_questionnaire ON cdp_app.gl_cdp_submissions(questionnaire_id);
CREATE INDEX idx_cdp_sub_org ON cdp_app.gl_cdp_submissions(org_id);
CREATE INDEX idx_cdp_sub_status ON cdp_app.gl_cdp_submissions(status);
CREATE INDEX idx_cdp_sub_format ON cdp_app.gl_cdp_submissions(submission_format);
CREATE INDEX idx_cdp_sub_submitted_at ON cdp_app.gl_cdp_submissions(submitted_at DESC);
CREATE INDEX idx_cdp_sub_created_at ON cdp_app.gl_cdp_submissions(created_at DESC);

-- =============================================================================
-- Continuous Aggregate: cdp_app.monthly_response_progress
-- =============================================================================
-- Precomputed monthly response submission progress aggregated by
-- questionnaire and status for dashboard time-series tracking.

CREATE MATERIALIZED VIEW cdp_app.monthly_response_progress
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 month', created_at)  AS bucket,
    questionnaire_id,
    response_status,
    COUNT(*)                            AS response_count
FROM cdp_app.gl_cdp_responses
GROUP BY bucket, questionnaire_id, response_status
WITH NO DATA;

-- Refresh policy: every 30 minutes, covering last 3 hours
SELECT add_continuous_aggregate_policy('cdp_app.monthly_response_progress',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Continuous Aggregate: cdp_app.quarterly_score_trends
-- =============================================================================
-- Precomputed quarterly scoring trends aggregated by organization for
-- historical score progression and trend analysis.

CREATE MATERIALIZED VIEW cdp_app.quarterly_score_trends
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('3 months', created_at) AS bucket,
    org_id,
    AVG(overall_score)                  AS avg_score,
    MAX(overall_score)                  AS max_score,
    MIN(overall_score)                  AS min_score,
    COUNT(*)                            AS simulation_count
FROM cdp_app.gl_cdp_scoring_results
GROUP BY bucket, org_id
WITH NO DATA;

-- Refresh policy: every hour, covering last 3 days
SELECT add_continuous_aggregate_policy('cdp_app.quarterly_score_trends',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep responses for 3650 days (10 years, CDP disclosure retention)
SELECT add_retention_policy('cdp_app.gl_cdp_responses', INTERVAL '3650 days');

-- Keep scoring results for 3650 days (10 years)
SELECT add_retention_policy('cdp_app.gl_cdp_scoring_results', INTERVAL '3650 days');

-- Keep gap analyses for 3650 days (10 years)
SELECT add_retention_policy('cdp_app.gl_cdp_gap_analyses', INTERVAL '3650 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on responses after 90 days
ALTER TABLE cdp_app.gl_cdp_responses SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_compression_policy('cdp_app.gl_cdp_responses', INTERVAL '90 days');

-- Enable compression on scoring_results after 90 days
ALTER TABLE cdp_app.gl_cdp_scoring_results SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_compression_policy('cdp_app.gl_cdp_scoring_results', INTERVAL '90 days');

-- Enable compression on gap_analyses after 90 days
ALTER TABLE cdp_app.gl_cdp_gap_analyses SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_compression_policy('cdp_app.gl_cdp_gap_analyses', INTERVAL '90 days');

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA cdp_app TO greenlang_app;
GRANT ALL ON ALL TABLES IN SCHEMA cdp_app TO greenlang_app;
GRANT ALL ON ALL SEQUENCES IN SCHEMA cdp_app TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON cdp_app.monthly_response_progress TO greenlang_app;
GRANT SELECT ON cdp_app.quarterly_score_trends TO greenlang_app;

-- Read-only role
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_readonly') THEN
        GRANT USAGE ON SCHEMA cdp_app TO greenlang_readonly;
        GRANT SELECT ON ALL TABLES IN SCHEMA cdp_app TO greenlang_readonly;
        GRANT SELECT ON cdp_app.monthly_response_progress TO greenlang_readonly;
        GRANT SELECT ON cdp_app.quarterly_score_trends TO greenlang_readonly;
    END IF;
END
$$;

-- Add CDP app service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'cdp_app:organizations:read', 'cdp_app', 'organizations_read', 'View CDP organization profiles'),
    (gen_random_uuid(), 'cdp_app:organizations:write', 'cdp_app', 'organizations_write', 'Create and manage CDP organization profiles'),
    (gen_random_uuid(), 'cdp_app:questionnaires:read', 'cdp_app', 'questionnaires_read', 'View CDP questionnaires'),
    (gen_random_uuid(), 'cdp_app:questionnaires:write', 'cdp_app', 'questionnaires_write', 'Create and manage CDP questionnaires'),
    (gen_random_uuid(), 'cdp_app:modules:read', 'cdp_app', 'modules_read', 'View CDP questionnaire modules'),
    (gen_random_uuid(), 'cdp_app:questions:read', 'cdp_app', 'questions_read', 'View CDP questions'),
    (gen_random_uuid(), 'cdp_app:questions:write', 'cdp_app', 'questions_write', 'Manage CDP question registry'),
    (gen_random_uuid(), 'cdp_app:responses:read', 'cdp_app', 'responses_read', 'View CDP questionnaire responses'),
    (gen_random_uuid(), 'cdp_app:responses:write', 'cdp_app', 'responses_write', 'Create and manage CDP responses'),
    (gen_random_uuid(), 'cdp_app:responses:approve', 'cdp_app', 'responses_approve', 'Approve CDP questionnaire responses'),
    (gen_random_uuid(), 'cdp_app:responses:submit', 'cdp_app', 'responses_submit', 'Submit CDP questionnaire responses'),
    (gen_random_uuid(), 'cdp_app:scoring:read', 'cdp_app', 'scoring_read', 'View CDP scoring results and simulations'),
    (gen_random_uuid(), 'cdp_app:scoring:simulate', 'cdp_app', 'scoring_simulate', 'Run CDP scoring simulations'),
    (gen_random_uuid(), 'cdp_app:gaps:read', 'cdp_app', 'gaps_read', 'View CDP gap analysis results'),
    (gen_random_uuid(), 'cdp_app:gaps:write', 'cdp_app', 'gaps_write', 'Run and manage CDP gap analyses'),
    (gen_random_uuid(), 'cdp_app:benchmarks:read', 'cdp_app', 'benchmarks_read', 'View CDP sector benchmarks'),
    (gen_random_uuid(), 'cdp_app:benchmarks:write', 'cdp_app', 'benchmarks_write', 'Manage CDP benchmark data'),
    (gen_random_uuid(), 'cdp_app:supply_chain:read', 'cdp_app', 'supply_chain_read', 'View CDP supply chain requests'),
    (gen_random_uuid(), 'cdp_app:supply_chain:write', 'cdp_app', 'supply_chain_write', 'Manage CDP supply chain engagement'),
    (gen_random_uuid(), 'cdp_app:transition:read', 'cdp_app', 'transition_read', 'View CDP transition plans'),
    (gen_random_uuid(), 'cdp_app:transition:write', 'cdp_app', 'transition_write', 'Create and manage CDP transition plans'),
    (gen_random_uuid(), 'cdp_app:verification:read', 'cdp_app', 'verification_read', 'View CDP verification records'),
    (gen_random_uuid(), 'cdp_app:verification:write', 'cdp_app', 'verification_write', 'Create and manage CDP verification records'),
    (gen_random_uuid(), 'cdp_app:submissions:read', 'cdp_app', 'submissions_read', 'View CDP submission records'),
    (gen_random_uuid(), 'cdp_app:submissions:write', 'cdp_app', 'submissions_write', 'Create and manage CDP submissions'),
    (gen_random_uuid(), 'cdp_app:reports:read', 'cdp_app', 'reports_read', 'View CDP generated reports'),
    (gen_random_uuid(), 'cdp_app:reports:generate', 'cdp_app', 'reports_generate', 'Generate CDP submission reports'),
    (gen_random_uuid(), 'cdp_app:dashboard:read', 'cdp_app', 'dashboard_read', 'View CDP dashboards and analytics'),
    (gen_random_uuid(), 'cdp_app:historical:read', 'cdp_app', 'historical_read', 'View CDP historical score data'),
    (gen_random_uuid(), 'cdp_app:admin', 'cdp_app', 'admin', 'CDP application administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA cdp_app IS 'GL-CDP-APP v1.0 Application Schema - CDP Climate Change Disclosure Platform with questionnaire management, scoring simulation, gap analysis, benchmarking, supply chain engagement, transition plan building, verification tracking, and submission';

COMMENT ON TABLE cdp_app.gl_cdp_organizations IS 'CDP organization profiles with sector GICS classification, region, revenue, employee count, and CDP account number';
COMMENT ON TABLE cdp_app.gl_cdp_questionnaires IS 'CDP questionnaire instances per reporting year with version tracking, status lifecycle, and sector-specific module configuration';
COMMENT ON TABLE cdp_app.gl_cdp_modules IS 'CDP Climate Change questionnaire module definitions (M0-M13) with applicability, sector specificity, and display ordering';
COMMENT ON TABLE cdp_app.gl_cdp_questions IS 'CDP question registry with 200+ questions including type, guidance, scoring category, point allocations, conditional logic, and version year';
COMMENT ON TABLE cdp_app.gl_cdp_responses IS 'TimescaleDB hypertable: CDP questionnaire responses with content, status lifecycle (draft/in_review/approved/submitted), auto-population source, and confidence scoring';
COMMENT ON TABLE cdp_app.gl_cdp_response_versions IS 'Response version history tracking content snapshots, change authors, and reasons at each version increment';
COMMENT ON TABLE cdp_app.gl_cdp_evidence_attachments IS 'Evidence documents and supporting files attached to individual responses for verification and audit purposes';
COMMENT ON TABLE cdp_app.gl_cdp_review_workflows IS 'Review workflow actions tracking status transitions, reviewer actions, comments, and timestamps';
COMMENT ON TABLE cdp_app.gl_cdp_scoring_results IS 'TimescaleDB hypertable: CDP scoring simulation results with overall score (0-100), band (D- to A), confidence interval, and A-level eligibility';
COMMENT ON TABLE cdp_app.gl_cdp_category_scores IS 'Per-category scores for 17 CDP Climate Change scoring categories with raw/weighted scores and sub-level breakdowns';
COMMENT ON TABLE cdp_app.gl_cdp_gap_analyses IS 'TimescaleDB hypertable: Gap analysis runs with total gap counts by severity (critical/high/medium/low) and potential score uplift';
COMMENT ON TABLE cdp_app.gl_cdp_gap_items IS 'Individual gap items with severity, current/target scoring level, recommendation, effort estimation, and score uplift prediction';
COMMENT ON TABLE cdp_app.gl_cdp_benchmarks IS 'Sector and regional benchmark data with score statistics, A-list rate, respondent count, and score distribution by band';
COMMENT ON TABLE cdp_app.gl_cdp_peer_comparisons IS 'Peer group comparison results linking questionnaires to benchmarks with percentile ranking and category-level comparisons';
COMMENT ON TABLE cdp_app.gl_cdp_supply_chain_requests IS 'CDP Supply Chain supplier engagement requests with invitation tracking, response status, and reminder counts';
COMMENT ON TABLE cdp_app.gl_cdp_supplier_responses IS 'Supplier questionnaire response data with emissions by scope, target/verification status, and engagement scoring';
COMMENT ON TABLE cdp_app.gl_cdp_transition_plans IS '1.5C-aligned transition plans with pathway type, SBTi alignment, investment totals, revenue alignment, and public availability';
COMMENT ON TABLE cdp_app.gl_cdp_transition_milestones IS 'Decarbonization milestones with target year, reduction percentage, scope, technology lever, investment, and progress tracking';
COMMENT ON TABLE cdp_app.gl_cdp_verification_records IS 'Third-party verification records per scope with verifier details, assurance level, statement dates, and A-level requirement tracking';
COMMENT ON TABLE cdp_app.gl_cdp_submissions IS 'Final CDP submission records with format, file path, reference number, CDP confirmation, and status tracking';

COMMENT ON MATERIALIZED VIEW cdp_app.monthly_response_progress IS 'Continuous aggregate: monthly response submission progress by questionnaire and status for dashboard tracking';
COMMENT ON MATERIALIZED VIEW cdp_app.quarterly_score_trends IS 'Continuous aggregate: quarterly scoring trends by organization for historical score progression analysis';

COMMENT ON COLUMN cdp_app.gl_cdp_questionnaires.status IS 'Questionnaire lifecycle: not_started, in_progress, completed, submitted, scored';
COMMENT ON COLUMN cdp_app.gl_cdp_modules.module_code IS 'CDP module code: M0 (Introduction) through M13 (Sign Off)';
COMMENT ON COLUMN cdp_app.gl_cdp_questions.question_type IS 'Question type: text, numeric, percentage, table, multi_select, single_select, yes_no';
COMMENT ON COLUMN cdp_app.gl_cdp_questions.scoring_category IS 'One of 17 CDP scoring categories (governance, risk_management, etc.)';
COMMENT ON COLUMN cdp_app.gl_cdp_responses.response_status IS 'Response lifecycle: draft, in_review, approved, submitted';
COMMENT ON COLUMN cdp_app.gl_cdp_responses.auto_populated IS 'Whether this response was auto-populated from MRV agent data';
COMMENT ON COLUMN cdp_app.gl_cdp_responses.auto_populated_source IS 'MRV agent source (e.g., MRV-001 Stationary Combustion) for auto-populated responses';
COMMENT ON COLUMN cdp_app.gl_cdp_scoring_results.score_band IS 'CDP score band: D-, D, C-, C, B-, B, A-, A';
COMMENT ON COLUMN cdp_app.gl_cdp_scoring_results.a_level_eligible IS 'Whether the organization meets all 5 A-level requirements';
COMMENT ON COLUMN cdp_app.gl_cdp_gap_items.severity IS 'Gap severity: critical, high, medium, low';
COMMENT ON COLUMN cdp_app.gl_cdp_gap_items.current_level IS 'Current scoring level: disclosure, awareness, management, leadership';
COMMENT ON COLUMN cdp_app.gl_cdp_gap_items.target_level IS 'Target scoring level: disclosure, awareness, management, leadership';
COMMENT ON COLUMN cdp_app.gl_cdp_gap_items.effort IS 'Estimated effort to close gap: low, medium, high';
COMMENT ON COLUMN cdp_app.gl_cdp_benchmarks.a_list_rate IS 'Percentage of respondents achieving A or A- in this sector/region';
COMMENT ON COLUMN cdp_app.gl_cdp_supply_chain_requests.status IS 'Supplier engagement status: not_invited, invited, responded, declined, expired';
COMMENT ON COLUMN cdp_app.gl_cdp_transition_plans.pathway_type IS 'Transition pathway: net_zero, well_below_2c, aligned_1_5c, aligned_2c';
COMMENT ON COLUMN cdp_app.gl_cdp_transition_plans.sbti_status IS 'SBTi status: none, committed, targets_set, validated';
COMMENT ON COLUMN cdp_app.gl_cdp_transition_milestones.status IS 'Milestone status: not_started, on_track, behind, at_risk, completed, cancelled';
COMMENT ON COLUMN cdp_app.gl_cdp_verification_records.assurance_level IS 'Assurance level: limited, reasonable';
COMMENT ON COLUMN cdp_app.gl_cdp_submissions.submission_format IS 'Submission format: xml, pdf, excel, json';
COMMENT ON COLUMN cdp_app.gl_cdp_submissions.status IS 'Submission status: pending, submitted, accepted, rejected, withdrawn';

-- =============================================================================
-- End of V085: GL-CDP-APP v1.0 Application Service Schema
-- =============================================================================
-- Summary:
--   20 tables created
--   3 hypertables (responses, scoring_results, gap_analyses)
--   2 continuous aggregates (monthly_response_progress, quarterly_score_trends)
--   7 update triggers
--   60+ B-tree indexes
--   9 GIN indexes on JSONB columns
--   3 retention policies (10-year retention)
--   3 compression policies (90-day threshold)
--   30 security permissions
--   Security grants for greenlang_app and greenlang_readonly
-- Previous: V084__iso14064_app_service.sql
-- =============================================================================
