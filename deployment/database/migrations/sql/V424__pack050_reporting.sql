-- =============================================================================
-- PACK-050 GHG Consolidation Pack
-- Migration: V424 - Reporting
-- =============================================================================
-- Pack:         PACK-050 (GHG Consolidation Pack)
-- Migration:    009 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates reporting, signoff, and assurance package tables. Reports are
-- generated from approved consolidation runs and can target multiple
-- frameworks (GHG Protocol, CDP, CSRD, TCFD). Signoffs provide a
-- multi-level approval workflow. Assurance packages bundle all
-- supporting evidence for external verification.
--
-- Tables (3):
--   1. ghg_consolidation.gl_cons_reports
--   2. ghg_consolidation.gl_cons_signoffs
--   3. ghg_consolidation.gl_cons_assurance_packages
--
-- Also includes: indexes, RLS, constraints, comments.
-- Previous: V423__pack050_mna.sql
-- Next:     V425__pack050_views_indexes_seed.sql
-- =============================================================================

SET search_path TO ghg_consolidation, public;

-- =============================================================================
-- Table 1: ghg_consolidation.gl_cons_reports
-- =============================================================================
-- Generated reports from consolidation runs. Each report targets a specific
-- framework and output format. Reports are immutable once finalised;
-- new versions create new rows. Supports GHG Protocol, CDP, CSRD, TCFD,
-- ISO 14064, SECR, and custom frameworks.

CREATE TABLE ghg_consolidation.gl_cons_reports (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    run_id                      UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_consolidation_runs(id) ON DELETE CASCADE,
    report_name                 VARCHAR(255)    NOT NULL,
    report_type                 VARCHAR(30)     NOT NULL,
    framework                   VARCHAR(30)     NOT NULL,
    framework_version           VARCHAR(20),
    format                      VARCHAR(20)     NOT NULL DEFAULT 'PDF',
    version_number              INTEGER         NOT NULL DEFAULT 1,
    status                      VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    reporting_period_start      DATE            NOT NULL,
    reporting_period_end        DATE            NOT NULL,
    scope_coverage              VARCHAR(30)[]   DEFAULT ARRAY['SCOPE_1', 'SCOPE_2', 'SCOPE_3']::VARCHAR(30)[],
    content                     JSONB           DEFAULT '{}',
    summary_data                JSONB           DEFAULT '{}',
    file_reference              VARCHAR(500),
    file_size_bytes             BIGINT,
    file_checksum               VARCHAR(64),
    template_id                 VARCHAR(100),
    include_entity_detail       BOOLEAN         NOT NULL DEFAULT true,
    include_methodology         BOOLEAN         NOT NULL DEFAULT true,
    include_uncertainty         BOOLEAN         NOT NULL DEFAULT false,
    include_trends              BOOLEAN         NOT NULL DEFAULT true,
    generated_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    generated_by                UUID,
    finalised_at                TIMESTAMPTZ,
    finalised_by                UUID,
    provenance_hash             VARCHAR(64),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p050_rpt_type CHECK (
        report_type IN (
            'CORPORATE_INVENTORY', 'SCOPE_DETAIL', 'ENTITY_BREAKDOWN',
            'YEAR_OVER_YEAR', 'BASE_YEAR_COMPARISON', 'INTENSITY_REPORT',
            'ELIMINATION_REPORT', 'DATA_QUALITY_REPORT',
            'ASSURANCE_SUMMARY', 'EXECUTIVE_SUMMARY', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p050_rpt_framework CHECK (
        framework IN (
            'GHG_PROTOCOL', 'CDP', 'CSRD', 'TCFD', 'ISO_14064',
            'SECR', 'NGER', 'EPA', 'SBTi', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p050_rpt_format CHECK (
        format IN ('PDF', 'XLSX', 'CSV', 'JSON', 'XML', 'HTML')
    ),
    CONSTRAINT chk_p050_rpt_status CHECK (
        status IN ('DRAFT', 'REVIEW', 'FINAL', 'PUBLISHED', 'SUPERSEDED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p050_rpt_version CHECK (version_number >= 1),
    CONSTRAINT chk_p050_rpt_dates CHECK (
        reporting_period_end > reporting_period_start
    ),
    CONSTRAINT chk_p050_rpt_file_size CHECK (
        file_size_bytes IS NULL OR file_size_bytes >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_rpt_tenant         ON ghg_consolidation.gl_cons_reports(tenant_id);
CREATE INDEX idx_p050_rpt_run            ON ghg_consolidation.gl_cons_reports(run_id);
CREATE INDEX idx_p050_rpt_type           ON ghg_consolidation.gl_cons_reports(report_type);
CREATE INDEX idx_p050_rpt_framework      ON ghg_consolidation.gl_cons_reports(framework);
CREATE INDEX idx_p050_rpt_status         ON ghg_consolidation.gl_cons_reports(status);
CREATE INDEX idx_p050_rpt_format         ON ghg_consolidation.gl_cons_reports(format);
CREATE INDEX idx_p050_rpt_period         ON ghg_consolidation.gl_cons_reports(reporting_period_start, reporting_period_end);
CREATE INDEX idx_p050_rpt_final          ON ghg_consolidation.gl_cons_reports(run_id, status)
    WHERE status = 'FINAL';
CREATE INDEX idx_p050_rpt_published      ON ghg_consolidation.gl_cons_reports(tenant_id, status)
    WHERE status = 'PUBLISHED';
CREATE INDEX idx_p050_rpt_generated      ON ghg_consolidation.gl_cons_reports(generated_at);
CREATE INDEX idx_p050_rpt_content        ON ghg_consolidation.gl_cons_reports USING gin(content);
CREATE INDEX idx_p050_rpt_summary        ON ghg_consolidation.gl_cons_reports USING gin(summary_data);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_reports ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_rpt_tenant_isolation ON ghg_consolidation.gl_cons_reports
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 2: ghg_consolidation.gl_cons_signoffs
-- =============================================================================
-- Multi-level signoff/approval workflow for consolidation runs. Supports
-- sequential approval levels (preparer, reviewer, approver, executive,
-- board). Each signoff captures the level, actor, timestamp, and
-- optional comments or conditions.

CREATE TABLE ghg_consolidation.gl_cons_signoffs (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    run_id                      UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_consolidation_runs(id) ON DELETE CASCADE,
    report_id                   UUID            REFERENCES ghg_consolidation.gl_cons_reports(id) ON DELETE SET NULL,
    level                       VARCHAR(30)     NOT NULL,
    level_order                 INTEGER         NOT NULL DEFAULT 1,
    status                      VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    signoff_by                  UUID,
    signoff_by_name             VARCHAR(255),
    signoff_by_title            VARCHAR(255),
    signoff_at                  TIMESTAMPTZ,
    due_date                    DATE,
    comments                    TEXT,
    conditions                  TEXT,
    is_conditional              BOOLEAN         NOT NULL DEFAULT false,
    delegation_from             UUID,
    delegation_reason           TEXT,
    reminder_sent_at            TIMESTAMPTZ,
    provenance_hash             VARCHAR(64),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p050_so_level CHECK (
        level IN (
            'PREPARER', 'REVIEWER', 'SENIOR_REVIEWER',
            'APPROVER', 'EXECUTIVE', 'BOARD', 'EXTERNAL'
        )
    ),
    CONSTRAINT chk_p050_so_level_order CHECK (
        level_order >= 1 AND level_order <= 10
    ),
    CONSTRAINT chk_p050_so_status CHECK (
        status IN ('PENDING', 'APPROVED', 'REJECTED', 'CONDITIONAL', 'DELEGATED', 'SKIPPED')
    ),
    CONSTRAINT chk_p050_so_conditional CHECK (
        is_conditional = false OR conditions IS NOT NULL
    ),
    CONSTRAINT chk_p050_so_delegation CHECK (
        delegation_from IS NULL OR delegation_reason IS NOT NULL
    ),
    CONSTRAINT uq_p050_so_run_level UNIQUE (run_id, level, level_order)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_so_tenant          ON ghg_consolidation.gl_cons_signoffs(tenant_id);
CREATE INDEX idx_p050_so_run             ON ghg_consolidation.gl_cons_signoffs(run_id);
CREATE INDEX idx_p050_so_report          ON ghg_consolidation.gl_cons_signoffs(report_id)
    WHERE report_id IS NOT NULL;
CREATE INDEX idx_p050_so_level           ON ghg_consolidation.gl_cons_signoffs(level);
CREATE INDEX idx_p050_so_status          ON ghg_consolidation.gl_cons_signoffs(status);
CREATE INDEX idx_p050_so_signoff_by      ON ghg_consolidation.gl_cons_signoffs(signoff_by)
    WHERE signoff_by IS NOT NULL;
CREATE INDEX idx_p050_so_pending         ON ghg_consolidation.gl_cons_signoffs(run_id, status)
    WHERE status = 'PENDING';
CREATE INDEX idx_p050_so_due_date        ON ghg_consolidation.gl_cons_signoffs(due_date)
    WHERE due_date IS NOT NULL;
CREATE INDEX idx_p050_so_run_order       ON ghg_consolidation.gl_cons_signoffs(run_id, level_order);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_signoffs ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_so_tenant_isolation ON ghg_consolidation.gl_cons_signoffs
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 3: ghg_consolidation.gl_cons_assurance_packages
-- =============================================================================
-- Bundles all supporting evidence and documentation for external assurance
-- (limited or reasonable). Each package references a consolidation run
-- and contains structured data for the assurance provider including
-- methodology documentation, data quality assessments, entity-level
-- evidence, and reconciliation results.

CREATE TABLE ghg_consolidation.gl_cons_assurance_packages (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    run_id                      UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_consolidation_runs(id) ON DELETE CASCADE,
    package_name                VARCHAR(255)    NOT NULL,
    assurance_type              VARCHAR(30)     NOT NULL DEFAULT 'LIMITED',
    assurance_provider          VARCHAR(255),
    assurance_standard          VARCHAR(100),
    engagement_ref              VARCHAR(200),
    package_data                JSONB           NOT NULL DEFAULT '{}',
    methodology_documentation   JSONB           DEFAULT '{}',
    data_quality_summary        JSONB           DEFAULT '{}',
    entity_evidence             JSONB           DEFAULT '{}',
    boundary_documentation      JSONB           DEFAULT '{}',
    elimination_evidence        JSONB           DEFAULT '{}',
    restatement_documentation   JSONB           DEFAULT '{}',
    adjustment_documentation    JSONB           DEFAULT '{}',
    completeness_assessment     JSONB           DEFAULT '{}',
    file_references             TEXT[],
    total_file_count            INTEGER         NOT NULL DEFAULT 0,
    total_file_size_bytes       BIGINT          NOT NULL DEFAULT 0,
    status                      VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    prepared_by                 UUID,
    prepared_at                 TIMESTAMPTZ,
    submitted_to_assurer_at     TIMESTAMPTZ,
    assurance_completed_at      TIMESTAMPTZ,
    assurance_opinion           VARCHAR(30),
    assurance_findings          JSONB           DEFAULT '[]',
    provenance_hash             VARCHAR(64),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p050_ap_type CHECK (
        assurance_type IN ('LIMITED', 'REASONABLE', 'COMBINED', 'INTERNAL', 'NONE')
    ),
    CONSTRAINT chk_p050_ap_status CHECK (
        status IN ('DRAFT', 'COMPILED', 'REVIEW', 'SUBMITTED', 'IN_ASSURANCE', 'COMPLETED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p050_ap_opinion CHECK (
        assurance_opinion IS NULL OR assurance_opinion IN (
            'UNQUALIFIED', 'QUALIFIED', 'ADVERSE', 'DISCLAIMER', 'PENDING'
        )
    ),
    CONSTRAINT chk_p050_ap_file_count CHECK (total_file_count >= 0),
    CONSTRAINT chk_p050_ap_file_size CHECK (total_file_size_bytes >= 0)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_ap_tenant          ON ghg_consolidation.gl_cons_assurance_packages(tenant_id);
CREATE INDEX idx_p050_ap_run             ON ghg_consolidation.gl_cons_assurance_packages(run_id);
CREATE INDEX idx_p050_ap_type            ON ghg_consolidation.gl_cons_assurance_packages(assurance_type);
CREATE INDEX idx_p050_ap_status          ON ghg_consolidation.gl_cons_assurance_packages(status);
CREATE INDEX idx_p050_ap_provider        ON ghg_consolidation.gl_cons_assurance_packages(assurance_provider)
    WHERE assurance_provider IS NOT NULL;
CREATE INDEX idx_p050_ap_opinion         ON ghg_consolidation.gl_cons_assurance_packages(assurance_opinion)
    WHERE assurance_opinion IS NOT NULL;
CREATE INDEX idx_p050_ap_submitted       ON ghg_consolidation.gl_cons_assurance_packages(submitted_to_assurer_at)
    WHERE submitted_to_assurer_at IS NOT NULL;
CREATE INDEX idx_p050_ap_package_data    ON ghg_consolidation.gl_cons_assurance_packages USING gin(package_data);
CREATE INDEX idx_p050_ap_findings        ON ghg_consolidation.gl_cons_assurance_packages USING gin(assurance_findings);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_assurance_packages ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_ap_tenant_isolation ON ghg_consolidation.gl_cons_assurance_packages
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_consolidation.gl_cons_reports IS
    'PACK-050: Generated reports (11 types, 10 frameworks, 6 formats) with versioning and publication status.';
COMMENT ON TABLE ghg_consolidation.gl_cons_signoffs IS
    'PACK-050: Multi-level signoff workflow (7 levels, 6 statuses) with delegation and conditional approval.';
COMMENT ON TABLE ghg_consolidation.gl_cons_assurance_packages IS
    'PACK-050: Assurance evidence packages (5 types, 7 statuses) with structured documentation bundles.';
