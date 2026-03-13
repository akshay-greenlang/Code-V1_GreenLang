-- ============================================================================
-- V122: AGENT-EUDR-034 Annual Review Scheduler Agent
-- ============================================================================
-- Creates tables for the Annual Review Scheduler Agent which orchestrates
-- periodic compliance reviews mandated by EUDR: annual review cycle
-- management with configurable review windows; granular task decomposition
-- for supplier reviews, plot reviews, risk assessment reviews, DDS reviews,
-- documentation reviews, stakeholder consultations, and compliance checks;
-- regulatory and submission deadline tracking with authority integration;
-- review checklist templates and instances with article-level coverage
-- tracking; TimescaleDB-partitioned notification dispatch and acknowledgement;
-- multi-year comparison analytics with change significance classification;
-- unified compliance calendar for cross-entity event coordination;
-- TimescaleDB-partitioned historical review records for trend analysis;
-- and immutable Article 31 audit trail via TimescaleDB hypertable.
--
-- Agent ID: GL-EUDR-ARS-034
-- PRD: PRD-AGENT-EUDR-034
-- Regulation: EU 2023/1115 (EUDR) Articles 4, 8, 9, 10, 11, 12, 14-16, 29, 31
-- Tables: 9 (6 regular + 3 hypertables)
-- Indexes: ~115
--
-- Dependencies: TimescaleDB extension (for hypertables)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V122: Creating AGENT-EUDR-034 Annual Review Scheduler Agent tables...';


-- ============================================================================
-- 1. gl_eudr_ars_review_cycles -- Annual review cycle definitions
-- ============================================================================
RAISE NOTICE 'V122 [1/9]: Creating gl_eudr_ars_review_cycles...';

CREATE TABLE IF NOT EXISTS gl_eudr_ars_review_cycles (
    cycle_id                        UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this annual review cycle
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator whose compliance is being reviewed
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    commodity_type                  VARCHAR(50)     NOT NULL,
        -- EUDR commodity type under review (cattle, cocoa, coffee, oil_palm, rubber, soya, wood)
    origin_country                  VARCHAR(10)     NOT NULL,
        -- ISO 3166-1 alpha-2 country code of commodity origin
    review_year                     INTEGER         NOT NULL,
        -- Calendar year this review cycle covers
    cycle_start_date                DATE            NOT NULL,
        -- Date when the review cycle officially begins
    cycle_end_date                  DATE            NOT NULL,
        -- Date when the review cycle is expected to conclude
    review_window_days              INTEGER         NOT NULL DEFAULT 90,
        -- Number of days allocated for completing the review (default: 90 per EUDR guidance)
    status                          VARCHAR(20)     NOT NULL DEFAULT 'scheduled',
        -- Current lifecycle status of the review cycle
    assigned_to                     VARCHAR(100),
        -- User or team responsible for managing this review cycle
    completion_percentage           NUMERIC(5,2)    NOT NULL DEFAULT 0,
        -- Overall completion percentage across all tasks (0.00-100.00)
    total_tasks                     INTEGER         NOT NULL DEFAULT 0,
        -- Total number of review tasks in this cycle
    completed_tasks                 INTEGER         NOT NULL DEFAULT 0,
        -- Number of tasks completed in this cycle
    overall_compliance_score        NUMERIC(5,2),
        -- Aggregate compliance score at review conclusion (0.00-100.00)
    review_outcome                  VARCHAR(30),
        -- Final outcome of the review cycle
    previous_cycle_id               UUID,
        -- FK reference to the previous year cycle for year-over-year comparison
    escalation_required             BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether this cycle has been flagged for escalation
    review_notes                    TEXT            DEFAULT '',
        -- General notes and observations for this review cycle
    config                          JSONB           DEFAULT '{}',
        -- Cycle-specific configuration: {"auto_assign_tasks": true, "notification_lead_days": 14, "escalation_policy": {...}}
    tags                            JSONB           DEFAULT '[]',
        -- Organizational tags: ["priority", "high-risk-country", "new-supplier"]
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for cycle configuration integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                      VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- User or system actor that created this cycle

    CONSTRAINT chk_ars_cycle_status CHECK (status IN (
        'scheduled', 'in_progress', 'completed', 'overdue', 'cancelled'
    )),
    CONSTRAINT chk_ars_cycle_outcome CHECK (review_outcome IS NULL OR review_outcome IN (
        'compliant', 'partially_compliant', 'non_compliant', 'requires_action', 'deferred'
    )),
    CONSTRAINT chk_ars_cycle_dates CHECK (cycle_end_date >= cycle_start_date),
    CONSTRAINT chk_ars_cycle_window CHECK (review_window_days > 0 AND review_window_days <= 365),
    CONSTRAINT chk_ars_cycle_completion CHECK (completion_percentage >= 0 AND completion_percentage <= 100),
    CONSTRAINT chk_ars_cycle_total_tasks CHECK (total_tasks >= 0),
    CONSTRAINT chk_ars_cycle_completed_tasks CHECK (completed_tasks >= 0),
    CONSTRAINT chk_ars_cycle_task_count CHECK (completed_tasks <= total_tasks),
    CONSTRAINT chk_ars_cycle_compliance CHECK (overall_compliance_score IS NULL OR
        (overall_compliance_score >= 0 AND overall_compliance_score <= 100)),
    CONSTRAINT chk_ars_cycle_year CHECK (review_year >= 2023 AND review_year <= 2100),
    CONSTRAINT uq_ars_cycle_operator_commodity_country_year UNIQUE (operator_id, commodity_type, origin_country, review_year)
);

COMMENT ON TABLE gl_eudr_ars_review_cycles IS 'AGENT-EUDR-034: Annual review cycle definitions for EUDR periodic compliance reviews with configurable review windows, completion tracking, year-over-year linkage, escalation flags, and compliance scoring per Articles 4, 8, 9-12';

-- B-tree indexes for gl_eudr_ars_review_cycles
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cycle_tenant ON gl_eudr_ars_review_cycles (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cycle_commodity ON gl_eudr_ars_review_cycles (commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cycle_country ON gl_eudr_ars_review_cycles (origin_country);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cycle_year ON gl_eudr_ars_review_cycles (review_year DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cycle_status ON gl_eudr_ars_review_cycles (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cycle_assigned ON gl_eudr_ars_review_cycles (assigned_to);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cycle_provenance ON gl_eudr_ars_review_cycles (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cycle_created ON gl_eudr_ars_review_cycles (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cycle_operator_year ON gl_eudr_ars_review_cycles (operator_id, review_year DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cycle_operator_status ON gl_eudr_ars_review_cycles (operator_id, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cycle_tenant_operator ON gl_eudr_ars_review_cycles (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cycle_operator_commodity ON gl_eudr_ars_review_cycles (operator_id, commodity_type, review_year DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cycle_overdue_only ON gl_eudr_ars_review_cycles (operator_id, cycle_end_date)
        WHERE status = 'overdue';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cycle_active_only ON gl_eudr_ars_review_cycles (operator_id, review_year DESC)
        WHERE status IN ('scheduled', 'in_progress');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes on JSONB
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cycle_config ON gl_eudr_ars_review_cycles USING GIN (config);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cycle_tags ON gl_eudr_ars_review_cycles USING GIN (tags);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 2. gl_eudr_ars_review_tasks -- Individual review tasks within a cycle
-- ============================================================================
RAISE NOTICE 'V122 [2/9]: Creating gl_eudr_ars_review_tasks...';

CREATE TABLE IF NOT EXISTS gl_eudr_ars_review_tasks (
    task_id                         UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this review task
    cycle_id                        UUID            NOT NULL,
        -- FK reference to the parent review cycle
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator identifier (denormalized for query performance)
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    task_type                       VARCHAR(30)     NOT NULL,
        -- Classification of the review task
    entity_type                     VARCHAR(50)     NOT NULL,
        -- Type of entity being reviewed (supplier, plot, commodity_flow, certificate, dds_statement)
    entity_id                       VARCHAR(200)    NOT NULL,
        -- Identifier of the specific entity under review
    task_name                       VARCHAR(500)    NOT NULL,
        -- Human-readable name for the review task
    task_description                TEXT            NOT NULL DEFAULT '',
        -- Detailed description of what this task requires
    priority                        VARCHAR(10)     NOT NULL DEFAULT 'medium',
        -- Task priority level for scheduling and assignment
    due_date                        DATE            NOT NULL,
        -- Deadline for task completion
    completed_date                  DATE,
        -- Actual completion date (NULL if not yet completed)
    status                          VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- Current task execution status
    assigned_to                     VARCHAR(100),
        -- User or team responsible for executing this task
    estimated_hours                 NUMERIC(5,1)    DEFAULT 0,
        -- Estimated effort in hours
    actual_hours                    NUMERIC(5,1),
        -- Actual effort spent in hours
    checklist_items                 JSONB           DEFAULT '[]',
        -- Structured checklist: [{"item": "Verify supplier certificate", "required": true, "completed": false, "completed_by": null}, ...]
    total_checklist_items           INTEGER         NOT NULL DEFAULT 0,
        -- Total number of checklist items
    completed_checklist_items       INTEGER         NOT NULL DEFAULT 0,
        -- Number of checklist items completed
    completion_notes                TEXT            DEFAULT '',
        -- Notes recorded upon task completion
    findings                        JSONB           DEFAULT '[]',
        -- Task-level findings: [{"finding": "...", "severity": "high", "action_required": true}, ...]
    dependencies                    JSONB           DEFAULT '[]',
        -- Task dependencies: [{"task_id": "uuid", "dependency_type": "blocks"}, ...]
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for task integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ars_task_cycle FOREIGN KEY (cycle_id) REFERENCES gl_eudr_ars_review_cycles (cycle_id),
    CONSTRAINT chk_ars_task_type CHECK (task_type IN (
        'supplier_review', 'plot_review', 'risk_assessment_review',
        'dds_review', 'documentation_review', 'stakeholder_consultation',
        'compliance_check'
    )),
    CONSTRAINT chk_ars_task_priority CHECK (priority IN (
        'critical', 'high', 'medium', 'low'
    )),
    CONSTRAINT chk_ars_task_status CHECK (status IN (
        'pending', 'in_progress', 'completed', 'skipped', 'blocked', 'overdue'
    )),
    CONSTRAINT chk_ars_task_checklist_total CHECK (total_checklist_items >= 0),
    CONSTRAINT chk_ars_task_checklist_completed CHECK (completed_checklist_items >= 0),
    CONSTRAINT chk_ars_task_checklist_count CHECK (completed_checklist_items <= total_checklist_items),
    CONSTRAINT chk_ars_task_estimated_hours CHECK (estimated_hours IS NULL OR estimated_hours >= 0),
    CONSTRAINT chk_ars_task_actual_hours CHECK (actual_hours IS NULL OR actual_hours >= 0)
);

COMMENT ON TABLE gl_eudr_ars_review_tasks IS 'AGENT-EUDR-034: Granular review tasks within annual cycles covering supplier_review, plot_review, risk_assessment_review, dds_review, documentation_review, stakeholder_consultation, and compliance_check with checklist tracking, effort estimation, findings capture, and dependency management per EUDR Articles 9-12';

-- B-tree indexes for gl_eudr_ars_review_tasks
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_task_cycle ON gl_eudr_ars_review_tasks (cycle_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_task_tenant ON gl_eudr_ars_review_tasks (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_task_type ON gl_eudr_ars_review_tasks (task_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_task_status ON gl_eudr_ars_review_tasks (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_task_assigned ON gl_eudr_ars_review_tasks (assigned_to);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_task_due_date ON gl_eudr_ars_review_tasks (due_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_task_provenance ON gl_eudr_ars_review_tasks (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_task_created ON gl_eudr_ars_review_tasks (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_task_cycle_type ON gl_eudr_ars_review_tasks (cycle_id, task_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_task_cycle_status ON gl_eudr_ars_review_tasks (cycle_id, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_task_operator_status ON gl_eudr_ars_review_tasks (operator_id, status, due_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_task_entity_type_id ON gl_eudr_ars_review_tasks (entity_type, entity_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_task_tenant_operator ON gl_eudr_ars_review_tasks (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_task_overdue_only ON gl_eudr_ars_review_tasks (cycle_id, due_date)
        WHERE status = 'overdue';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_task_pending_only ON gl_eudr_ars_review_tasks (cycle_id, priority, due_date)
        WHERE status IN ('pending', 'in_progress', 'blocked');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes on JSONB
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_task_checklist ON gl_eudr_ars_review_tasks USING GIN (checklist_items);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_task_findings ON gl_eudr_ars_review_tasks USING GIN (findings);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 3. gl_eudr_ars_deadlines -- Regulatory and submission deadlines
-- ============================================================================
RAISE NOTICE 'V122 [3/9]: Creating gl_eudr_ars_deadlines...';

CREATE TABLE IF NOT EXISTS gl_eudr_ars_deadlines (
    deadline_id                     UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this deadline record
    cycle_id                        UUID            NOT NULL,
        -- FK reference to the associated review cycle
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator identifier (denormalized for query performance)
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    deadline_type                   VARCHAR(30)     NOT NULL,
        -- Classification of the regulatory deadline
    deadline_date                   DATE            NOT NULL,
        -- Date by which the obligation must be fulfilled
    submission_date                 DATE,
        -- Actual date the submission was made (NULL if not yet submitted)
    status                          VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- Current deadline fulfillment status
    authority_name                  VARCHAR(200)    NOT NULL DEFAULT '',
        -- Name of the competent authority or regulatory body
    authority_country               VARCHAR(10),
        -- Country of the competent authority (ISO 3166-1 alpha-2)
    submission_reference            VARCHAR(200),
        -- Reference number or identifier assigned by the authority upon submission
    evidence_url                    VARCHAR(2000),
        -- URL or S3 reference to submission evidence documentation
    evidence_hash                   VARCHAR(64),
        -- SHA-256 hash of the submitted evidence document for verification
    reminder_days_before            INTEGER[]       DEFAULT '{30,14,7,3,1}',
        -- Array of days before deadline to send reminders
    last_reminder_sent_at           TIMESTAMPTZ,
        -- Timestamp of the most recent reminder notification
    penalty_risk                    VARCHAR(20)     DEFAULT 'low',
        -- Assessed risk level of penalties for missing this deadline
    notes                           TEXT            DEFAULT '',
        -- Additional notes about this deadline
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for deadline record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ars_deadline_cycle FOREIGN KEY (cycle_id) REFERENCES gl_eudr_ars_review_cycles (cycle_id),
    CONSTRAINT chk_ars_deadline_type CHECK (deadline_type IN (
        'dds_submission', 'ca_registration', 'annual_report',
        'audit_response', 'review_completion'
    )),
    CONSTRAINT chk_ars_deadline_status CHECK (status IN (
        'pending', 'submitted', 'late', 'waived'
    )),
    CONSTRAINT chk_ars_deadline_penalty CHECK (penalty_risk IS NULL OR penalty_risk IN (
        'negligible', 'low', 'moderate', 'high', 'critical'
    ))
);

COMMENT ON TABLE gl_eudr_ars_deadlines IS 'AGENT-EUDR-034: Regulatory and submission deadline tracking for DDS submissions, competent authority registrations, annual reports, audit responses, and review completions with configurable reminder schedules, evidence verification, and penalty risk assessment per EUDR Articles 4, 8, 14-16';

-- B-tree indexes for gl_eudr_ars_deadlines
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_dl_cycle ON gl_eudr_ars_deadlines (cycle_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_dl_tenant ON gl_eudr_ars_deadlines (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_dl_type ON gl_eudr_ars_deadlines (deadline_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_dl_date ON gl_eudr_ars_deadlines (deadline_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_dl_status ON gl_eudr_ars_deadlines (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_dl_provenance ON gl_eudr_ars_deadlines (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_dl_operator_type ON gl_eudr_ars_deadlines (operator_id, deadline_type, deadline_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_dl_operator_status ON gl_eudr_ars_deadlines (operator_id, status, deadline_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_dl_cycle_type ON gl_eudr_ars_deadlines (cycle_id, deadline_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_dl_tenant_operator ON gl_eudr_ars_deadlines (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_dl_pending_only ON gl_eudr_ars_deadlines (operator_id, deadline_date)
        WHERE status = 'pending';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_dl_late_only ON gl_eudr_ars_deadlines (operator_id, deadline_date)
        WHERE status = 'late';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 4. gl_eudr_ars_checklists -- Review checklist templates and instances
-- ============================================================================
RAISE NOTICE 'V122 [4/9]: Creating gl_eudr_ars_checklists...';

CREATE TABLE IF NOT EXISTS gl_eudr_ars_checklists (
    checklist_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this checklist
    cycle_id                        UUID,
        -- FK reference to the review cycle (NULL for templates)
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator identifier
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    commodity_type                  VARCHAR(50)     NOT NULL,
        -- EUDR commodity type this checklist applies to
    checklist_type                  VARCHAR(10)     NOT NULL DEFAULT 'instance',
        -- Whether this is a reusable template or a cycle-specific instance
    checklist_name                  VARCHAR(500)    NOT NULL DEFAULT '',
        -- Human-readable name for the checklist
    checklist_description           TEXT            NOT NULL DEFAULT '',
        -- Description of the checklist scope and purpose
    checklist_items                 JSONB           NOT NULL DEFAULT '[]',
        -- Array of checklist items: [{"item_id": "uuid", "section": "Traceability", "question": "...", "article_ref": "Art. 9(1)(a)", "required": true, "response": null, "evidence_ref": null, "notes": ""}, ...]
    total_items                     INTEGER         NOT NULL DEFAULT 0,
        -- Total number of items in the checklist
    completed_items                 INTEGER         NOT NULL DEFAULT 0,
        -- Number of items that have been completed/answered
    compliant_items                 INTEGER         NOT NULL DEFAULT 0,
        -- Number of items assessed as compliant
    non_compliant_items             INTEGER         NOT NULL DEFAULT 0,
        -- Number of items assessed as non-compliant
    not_applicable_items            INTEGER         NOT NULL DEFAULT 0,
        -- Number of items marked as not applicable
    compliance_score                NUMERIC(5,2)    NOT NULL DEFAULT 0,
        -- Computed compliance score: (compliant_items / applicable_items) * 100
    article_coverage                JSONB           DEFAULT '{}',
        -- EUDR articles assessed: {"art_4": {"covered": true, "items": 5, "compliant": 4}, "art_9": {...}, ...}
    version                         INTEGER         NOT NULL DEFAULT 1,
        -- Checklist version number (incremented on template updates)
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for checklist integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ars_checklist_cycle FOREIGN KEY (cycle_id) REFERENCES gl_eudr_ars_review_cycles (cycle_id),
    CONSTRAINT chk_ars_checklist_type CHECK (checklist_type IN (
        'template', 'instance'
    )),
    CONSTRAINT chk_ars_checklist_total CHECK (total_items >= 0),
    CONSTRAINT chk_ars_checklist_completed CHECK (completed_items >= 0),
    CONSTRAINT chk_ars_checklist_compliant CHECK (compliant_items >= 0),
    CONSTRAINT chk_ars_checklist_non_compliant CHECK (non_compliant_items >= 0),
    CONSTRAINT chk_ars_checklist_na CHECK (not_applicable_items >= 0),
    CONSTRAINT chk_ars_checklist_item_count CHECK (
        compliant_items + non_compliant_items + not_applicable_items <= total_items
    ),
    CONSTRAINT chk_ars_checklist_score CHECK (compliance_score >= 0 AND compliance_score <= 100),
    CONSTRAINT chk_ars_checklist_version CHECK (version >= 1)
);

COMMENT ON TABLE gl_eudr_ars_checklists IS 'AGENT-EUDR-034: Review checklist templates and cycle-specific instances with article-level EUDR coverage tracking, per-item compliance assessment (compliant/non_compliant/not_applicable), versioned template management, and aggregate compliance scoring per Articles 4, 8, 9-12';

-- B-tree indexes for gl_eudr_ars_checklists
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cl_cycle ON gl_eudr_ars_checklists (cycle_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cl_tenant ON gl_eudr_ars_checklists (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cl_commodity ON gl_eudr_ars_checklists (commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cl_type ON gl_eudr_ars_checklists (checklist_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cl_score ON gl_eudr_ars_checklists (compliance_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cl_provenance ON gl_eudr_ars_checklists (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cl_operator_commodity ON gl_eudr_ars_checklists (operator_id, commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cl_operator_type ON gl_eudr_ars_checklists (operator_id, checklist_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cl_cycle_commodity ON gl_eudr_ars_checklists (cycle_id, commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cl_tenant_operator ON gl_eudr_ars_checklists (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cl_templates_only ON gl_eudr_ars_checklists (operator_id, commodity_type, version DESC)
        WHERE checklist_type = 'template';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes on JSONB
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cl_items ON gl_eudr_ars_checklists USING GIN (checklist_items);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cl_article_coverage ON gl_eudr_ars_checklists USING GIN (article_coverage);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 5. gl_eudr_ars_notifications -- Notification tracking (TimescaleDB hypertable)
-- ============================================================================
RAISE NOTICE 'V122 [5/9]: Creating gl_eudr_ars_notifications (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_ars_notifications (
    notification_id                 UUID            DEFAULT gen_random_uuid(),
        -- Unique identifier for this notification
    cycle_id                        UUID,
        -- FK reference to the associated review cycle (nullable for system-wide notifications)
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator targeted by this notification
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    notification_type               VARCHAR(30)     NOT NULL,
        -- Classification of the notification trigger
    recipients                      TEXT[]          NOT NULL DEFAULT '{}',
        -- Array of recipient identifiers (user IDs or email addresses)
    channels                        TEXT[]          NOT NULL DEFAULT '{}',
        -- Array of delivery channels used
    scheduled_at                    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when this notification is scheduled for delivery (partitioning column)
    sent_at                         TIMESTAMPTZ,
        -- Timestamp when the notification was actually sent (NULL if not yet sent)
    acknowledged_at                 TIMESTAMPTZ,
        -- Timestamp when a recipient acknowledged the notification
    acknowledged_by                 VARCHAR(100),
        -- User who acknowledged the notification
    status                          VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- Current notification delivery status
    message_template                VARCHAR(200)    NOT NULL DEFAULT '',
        -- Identifier of the message template used
    message_subject                 VARCHAR(500)    NOT NULL DEFAULT '',
        -- Subject line of the notification message
    message_data                    JSONB           DEFAULT '{}',
        -- Template substitution data: {"cycle_name": "...", "deadline_date": "...", "days_remaining": 14, "task_count": 5}
    delivery_attempts               INTEGER         NOT NULL DEFAULT 0,
        -- Number of delivery attempts made
    last_error                      TEXT,
        -- Error message from the most recent failed delivery attempt
    related_entity_type             VARCHAR(50),
        -- Type of entity this notification relates to (deadline, task, cycle)
    related_entity_id               UUID,
        -- Identifier of the related entity
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for notification integrity verification

    CONSTRAINT chk_ars_notif_type CHECK (notification_type IN (
        'review_starting', 'deadline_approaching', 'task_overdue',
        'review_completed', 'submission_required'
    )),
    CONSTRAINT chk_ars_notif_status CHECK (status IN (
        'pending', 'sent', 'delivered', 'failed', 'cancelled'
    )),
    CONSTRAINT chk_ars_notif_attempts CHECK (delivery_attempts >= 0)
);

-- Convert to hypertable partitioned by scheduled_at
DO $$
BEGIN
    PERFORM create_hypertable(
        'gl_eudr_ars_notifications',
        'scheduled_at',
        chunk_time_interval => INTERVAL '7 days',
        if_not_exists => TRUE
    );
    RAISE NOTICE 'gl_eudr_ars_notifications hypertable created (7-day chunks)';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Hypertable creation skipped for gl_eudr_ars_notifications: %', SQLERRM;
END $$;

COMMENT ON TABLE gl_eudr_ars_notifications IS 'AGENT-EUDR-034: TimescaleDB-partitioned notification tracking for review_starting, deadline_approaching, task_overdue, review_completed, and submission_required events with multi-channel delivery (email, webhook, dashboard), retry tracking, and acknowledgement lifecycle management per EUDR Article 14-16 compliance deadlines';

-- B-tree indexes for gl_eudr_ars_notifications
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_notif_cycle ON gl_eudr_ars_notifications (cycle_id, scheduled_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_notif_operator ON gl_eudr_ars_notifications (operator_id, scheduled_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_notif_tenant ON gl_eudr_ars_notifications (tenant_id, scheduled_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_notif_type ON gl_eudr_ars_notifications (notification_type, scheduled_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_notif_status ON gl_eudr_ars_notifications (status, scheduled_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_notif_provenance ON gl_eudr_ars_notifications (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_notif_operator_type ON gl_eudr_ars_notifications (operator_id, notification_type, scheduled_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_notif_tenant_operator ON gl_eudr_ars_notifications (tenant_id, operator_id, scheduled_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_notif_pending_only ON gl_eudr_ars_notifications (operator_id, scheduled_at)
        WHERE status = 'pending';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes on JSONB
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_notif_message_data ON gl_eudr_ars_notifications USING GIN (message_data);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 6. gl_eudr_ars_year_comparison -- Multi-year comparison results
-- ============================================================================
RAISE NOTICE 'V122 [6/9]: Creating gl_eudr_ars_year_comparison...';

CREATE TABLE IF NOT EXISTS gl_eudr_ars_year_comparison (
    comparison_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this year-over-year comparison
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator whose multi-year data is being compared
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    commodity_type                  VARCHAR(50)     NOT NULL,
        -- EUDR commodity type being compared
    origin_country                  VARCHAR(10),
        -- Country of origin filter (NULL for all countries)
    comparison_years                INTEGER[]       NOT NULL,
        -- Array of years being compared: {2024, 2025, 2026}
    comparison_date                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when this comparison was generated
    base_year                       INTEGER         NOT NULL,
        -- Reference year for delta calculations
    metrics_compared                JSONB           NOT NULL DEFAULT '{}',
        -- Per-year metric snapshots: {"2024": {"supplier_count": 45, "plot_count": 120, "avg_risk_score": 32.5, "alerts": 8, "compliance_score": 88.0}, "2025": {...}}
    changes_detected                JSONB           NOT NULL DEFAULT '{}',
        -- Year-over-year changes: {"supplier_count": {"delta": 5, "pct_change": 11.1}, "avg_risk_score": {"delta": -3.2, "pct_change": -9.8}, ...}
    change_significance             VARCHAR(20)     NOT NULL DEFAULT 'negligible',
        -- Overall significance classification of detected changes
    trend_analysis                  JSONB           DEFAULT '{}',
        -- Trend indicators: {"compliance_trend": "improving", "risk_trend": "stable", "supplier_growth": "increasing"}
    analysis_summary                TEXT            NOT NULL DEFAULT '',
        -- Human-readable narrative summary of the comparison results
    key_findings                    JSONB           DEFAULT '[]',
        -- Top findings: [{"finding": "Supplier count increased 11%", "impact": "positive", "action": "none"}, ...]
    recommendations                 JSONB           DEFAULT '[]',
        -- Recommendations based on comparison: [{"action": "...", "priority": "medium", "rationale": "..."}, ...]
    related_cycle_ids               JSONB           DEFAULT '[]',
        -- Array of review cycle IDs involved in the comparison
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for comparison result integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_ars_yc_significance CHECK (change_significance IN (
        'negligible', 'moderate', 'significant', 'critical'
    )),
    CONSTRAINT chk_ars_yc_base_year CHECK (base_year >= 2023 AND base_year <= 2100)
);

COMMENT ON TABLE gl_eudr_ars_year_comparison IS 'AGENT-EUDR-034: Multi-year comparison analytics with supplier count, plot count, risk score, alert, and compliance score metrics across comparison years, delta calculation against base year, trend analysis, significance classification (negligible/moderate/significant/critical), and actionable recommendations for EUDR continuous improvement per Articles 9-12';

-- B-tree indexes for gl_eudr_ars_year_comparison
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_yc_tenant ON gl_eudr_ars_year_comparison (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_yc_commodity ON gl_eudr_ars_year_comparison (commodity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_yc_significance ON gl_eudr_ars_year_comparison (change_significance);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_yc_date ON gl_eudr_ars_year_comparison (comparison_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_yc_provenance ON gl_eudr_ars_year_comparison (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_yc_operator_commodity ON gl_eudr_ars_year_comparison (operator_id, commodity_type, comparison_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_yc_operator_significance ON gl_eudr_ars_year_comparison (operator_id, change_significance, comparison_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_yc_tenant_operator ON gl_eudr_ars_year_comparison (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes on JSONB
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_yc_metrics ON gl_eudr_ars_year_comparison USING GIN (metrics_compared);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_yc_changes ON gl_eudr_ars_year_comparison USING GIN (changes_detected);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 7. gl_eudr_ars_calendar_events -- Unified compliance calendar
-- ============================================================================
RAISE NOTICE 'V122 [7/9]: Creating gl_eudr_ars_calendar_events...';

CREATE TABLE IF NOT EXISTS gl_eudr_ars_calendar_events (
    event_id                        UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this calendar event
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator this event belongs to
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    event_type                      VARCHAR(30)     NOT NULL,
        -- Classification of the calendar event
    event_date                      DATE            NOT NULL,
        -- Primary date of the event
    event_end_date                  DATE,
        -- End date for multi-day events (NULL for single-day events)
    event_name                      VARCHAR(500)    NOT NULL,
        -- Human-readable name for the calendar event
    event_description               TEXT            NOT NULL DEFAULT '',
        -- Detailed description of the event scope and requirements
    related_cycle_id                UUID,
        -- FK reference to an associated review cycle (nullable)
    related_entity_type             VARCHAR(50),
        -- Type of related entity (deadline, task, review, audit, submission)
    related_entity_id               UUID,
        -- Identifier of the related entity
    participants                    JSONB           DEFAULT '[]',
        -- Array of participants: [{"user_id": "...", "name": "...", "role": "reviewer"}, ...]
    location                        VARCHAR(500),
        -- Physical or virtual location of the event
    is_recurring                    BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether this event recurs on a schedule
    recurrence_rule                 VARCHAR(200),
        -- iCal RRULE string for recurring events (e.g., "FREQ=YEARLY;BYMONTH=3;BYMONTHDAY=15")
    recurrence_end_date             DATE,
        -- Date when recurrence stops
    reminder_minutes                INTEGER[]       DEFAULT '{1440,60}',
        -- Array of minutes before event to send reminders (1440=1day, 60=1hour)
    is_all_day                      BOOLEAN         NOT NULL DEFAULT TRUE,
        -- Whether this is an all-day event
    status                          VARCHAR(20)     NOT NULL DEFAULT 'scheduled',
        -- Current event status
    color_code                      VARCHAR(7),
        -- Hex color code for calendar display (e.g., "#FF5733")
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for calendar event integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ars_event_cycle FOREIGN KEY (related_cycle_id) REFERENCES gl_eudr_ars_review_cycles (cycle_id),
    CONSTRAINT chk_ars_event_type CHECK (event_type IN (
        'review_start', 'review_end', 'deadline', 'task_due',
        'audit_session', 'stakeholder_meeting', 'submission_window',
        'regulatory_date', 'custom'
    )),
    CONSTRAINT chk_ars_event_status CHECK (status IN (
        'scheduled', 'confirmed', 'in_progress', 'completed', 'cancelled', 'postponed'
    )),
    CONSTRAINT chk_ars_event_dates CHECK (event_end_date IS NULL OR event_end_date >= event_date)
);

COMMENT ON TABLE gl_eudr_ars_calendar_events IS 'AGENT-EUDR-034: Unified compliance calendar for review starts/ends, deadlines, task due dates, audit sessions, stakeholder meetings, submission windows, and regulatory dates with iCal recurrence support, participant tracking, and multi-entity linkage for EUDR annual review scheduling coordination';

-- B-tree indexes for gl_eudr_ars_calendar_events
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cal_tenant ON gl_eudr_ars_calendar_events (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cal_type ON gl_eudr_ars_calendar_events (event_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cal_date ON gl_eudr_ars_calendar_events (event_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cal_cycle ON gl_eudr_ars_calendar_events (related_cycle_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cal_status ON gl_eudr_ars_calendar_events (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cal_provenance ON gl_eudr_ars_calendar_events (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cal_operator_type ON gl_eudr_ars_calendar_events (operator_id, event_type, event_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cal_operator_date_range ON gl_eudr_ars_calendar_events (operator_id, event_date, event_end_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cal_tenant_operator ON gl_eudr_ars_calendar_events (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cal_related_entity ON gl_eudr_ars_calendar_events (related_entity_type, related_entity_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cal_upcoming_only ON gl_eudr_ars_calendar_events (operator_id, event_date)
        WHERE status IN ('scheduled', 'confirmed');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes on JSONB
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_cal_participants ON gl_eudr_ars_calendar_events USING GIN (participants);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 8. gl_eudr_ars_review_history -- Historical review records (TimescaleDB hypertable)
-- ============================================================================
RAISE NOTICE 'V122 [8/9]: Creating gl_eudr_ars_review_history (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_ars_review_history (
    history_id                      UUID            DEFAULT gen_random_uuid(),
        -- Unique identifier for this historical review record
    cycle_id                        UUID,
        -- FK reference to the review cycle that generated this record
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator whose review was completed
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    commodity_type                  VARCHAR(50)     NOT NULL,
        -- EUDR commodity type that was reviewed
    origin_country                  VARCHAR(10)     NOT NULL,
        -- Country of origin for the reviewed supply chain
    review_year                     INTEGER         NOT NULL,
        -- Calendar year covered by this review
    review_completed_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the review was finalized (partitioning column)
    review_duration_days            INTEGER,
        -- Number of days the review took from start to completion
    review_outcome                  VARCHAR(30)     NOT NULL DEFAULT 'compliant',
        -- Final outcome classification of the review
    compliance_status               VARCHAR(30)     NOT NULL DEFAULT 'compliant',
        -- Compliance status determination at review conclusion
    overall_compliance_score        NUMERIC(5,2)    NOT NULL DEFAULT 0,
        -- Aggregate compliance score (0.00-100.00)
    dimension_scores                JSONB           DEFAULT '{}',
        -- Per-dimension scores: {"traceability": 92.0, "deforestation_free": 95.0, "legality": 88.5, "due_diligence": 90.0}
    total_tasks_completed           INTEGER         NOT NULL DEFAULT 0,
        -- Total tasks that were completed during this review
    total_findings                  INTEGER         NOT NULL DEFAULT 0,
        -- Total findings identified across all tasks
    findings                        JSONB           DEFAULT '[]',
        -- Consolidated findings: [{"finding_id": "...", "category": "traceability", "severity": "medium", "description": "...", "status": "resolved"}, ...]
    findings_by_severity            JSONB           DEFAULT '{}',
        -- Finding count breakdown: {"critical": 0, "high": 2, "medium": 5, "low": 8}
    recommendations                 JSONB           DEFAULT '[]',
        -- Consolidated recommendations: [{"recommendation": "...", "priority": "high", "responsible": "...", "deadline": "..."}, ...]
    corrective_actions              JSONB           DEFAULT '[]',
        -- Corrective actions required: [{"action": "...", "status": "pending", "due_date": "...", "assigned_to": "..."}, ...]
    next_review_due                 DATE,
        -- Date by which the next annual review should commence
    reviewer_id                     VARCHAR(100),
        -- User who finalized the review
    reviewer_notes                  TEXT            DEFAULT '',
        -- Reviewer sign-off notes
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for historical record integrity verification

    CONSTRAINT chk_ars_hist_outcome CHECK (review_outcome IN (
        'compliant', 'partially_compliant', 'non_compliant', 'requires_action', 'deferred'
    )),
    CONSTRAINT chk_ars_hist_compliance CHECK (compliance_status IN (
        'compliant', 'conditionally_compliant', 'non_compliant', 'under_remediation', 'suspended'
    )),
    CONSTRAINT chk_ars_hist_score CHECK (overall_compliance_score >= 0 AND overall_compliance_score <= 100),
    CONSTRAINT chk_ars_hist_tasks CHECK (total_tasks_completed >= 0),
    CONSTRAINT chk_ars_hist_findings CHECK (total_findings >= 0),
    CONSTRAINT chk_ars_hist_year CHECK (review_year >= 2023 AND review_year <= 2100),
    CONSTRAINT chk_ars_hist_duration CHECK (review_duration_days IS NULL OR review_duration_days >= 0)
);

-- Convert to hypertable partitioned by review_completed_at
DO $$
BEGIN
    PERFORM create_hypertable(
        'gl_eudr_ars_review_history',
        'review_completed_at',
        chunk_time_interval => INTERVAL '30 days',
        if_not_exists => TRUE
    );
    RAISE NOTICE 'gl_eudr_ars_review_history hypertable created (30-day chunks)';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Hypertable creation skipped for gl_eudr_ars_review_history: %', SQLERRM;
END $$;

COMMENT ON TABLE gl_eudr_ars_review_history IS 'AGENT-EUDR-034: TimescaleDB-partitioned historical review records with outcome classification, multi-dimension compliance scoring, consolidated findings by severity, corrective action tracking, and next-review scheduling for EUDR Articles 4, 9-12 continuous compliance trend analysis';

-- B-tree indexes for gl_eudr_ars_review_history
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_hist_cycle ON gl_eudr_ars_review_history (cycle_id, review_completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_hist_operator ON gl_eudr_ars_review_history (operator_id, review_completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_hist_tenant ON gl_eudr_ars_review_history (tenant_id, review_completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_hist_year ON gl_eudr_ars_review_history (review_year DESC, review_completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_hist_outcome ON gl_eudr_ars_review_history (review_outcome, review_completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_hist_compliance ON gl_eudr_ars_review_history (compliance_status, review_completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_hist_provenance ON gl_eudr_ars_review_history (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_hist_operator_commodity ON gl_eudr_ars_review_history (operator_id, commodity_type, review_completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_hist_operator_outcome ON gl_eudr_ars_review_history (operator_id, review_outcome, review_completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_hist_tenant_operator ON gl_eudr_ars_review_history (tenant_id, operator_id, review_completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_hist_non_compliant ON gl_eudr_ars_review_history (operator_id, review_completed_at DESC)
        WHERE compliance_status IN ('non_compliant', 'under_remediation', 'suspended');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes on JSONB
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_hist_dimension_scores ON gl_eudr_ars_review_history USING GIN (dimension_scores);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_hist_findings ON gl_eudr_ars_review_history USING GIN (findings);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_hist_recommendations ON gl_eudr_ars_review_history USING GIN (recommendations);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_hist_corrective ON gl_eudr_ars_review_history USING GIN (corrective_actions);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 9. gl_eudr_ars_audit_trail -- Audit log (TimescaleDB hypertable)
-- ============================================================================
RAISE NOTICE 'V122 [9/9]: Creating gl_eudr_ars_audit_trail (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_ars_audit_trail (
    audit_id                        UUID            DEFAULT gen_random_uuid(),
        -- Unique audit entry identifier
    entity_type                     VARCHAR(50)     NOT NULL,
        -- Type of entity being audited (review_cycle, review_task, deadline, checklist, notification, year_comparison, calendar_event, review_history)
    entity_id                       UUID            NOT NULL,
        -- Entity identifier being audited
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator identifier
    tenant_id                       VARCHAR(100)    NOT NULL DEFAULT '',
        -- Multi-tenant isolation identifier
    action                          VARCHAR(50)     NOT NULL,
        -- Action performed (create, update, start_review, complete_task, submit_deadline, send_notification, etc.)
    actor_id                        VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- Actor performing the action (user ID or system)
    actor_type                      VARCHAR(20)     NOT NULL DEFAULT 'system',
        -- Type of actor (system, user, api, scheduler)
    changes                         JSONB           DEFAULT '{}',
        -- JSON diff of changes: {"field": {"old": "...", "new": "..."}}
    context                         JSONB           DEFAULT '{}',
        -- Additional context: {"ip_address": "...", "user_agent": "...", "request_id": "...", "cycle_year": 2026}
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for audit entry integrity
    timestamp                       TIMESTAMPTZ     NOT NULL DEFAULT NOW()
        -- Timestamp of the action (partitioning column)
);

-- Convert to hypertable
DO $$
BEGIN
    PERFORM create_hypertable(
        'gl_eudr_ars_audit_trail',
        'timestamp',
        chunk_time_interval => INTERVAL '7 days',
        if_not_exists => TRUE
    );
    RAISE NOTICE 'gl_eudr_ars_audit_trail hypertable created (7-day chunks)';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Hypertable creation skipped for gl_eudr_ars_audit_trail: %', SQLERRM;
END $$;

COMMENT ON TABLE gl_eudr_ars_audit_trail IS 'AGENT-EUDR-034: Immutable TimescaleDB-partitioned audit trail for all annual review scheduler operations per EUDR Article 31, capturing review cycle lifecycle events, task completions, deadline submissions, notification dispatches, and configuration changes with full provenance tracking';

-- B-tree indexes for gl_eudr_ars_audit_trail
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_audit_entity_type ON gl_eudr_ars_audit_trail (entity_type, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_audit_entity_id ON gl_eudr_ars_audit_trail (entity_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_audit_operator ON gl_eudr_ars_audit_trail (operator_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_audit_tenant ON gl_eudr_ars_audit_trail (tenant_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_audit_action ON gl_eudr_ars_audit_trail (action, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_audit_actor ON gl_eudr_ars_audit_trail (actor_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_audit_provenance ON gl_eudr_ars_audit_trail (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_audit_entity_action ON gl_eudr_ars_audit_trail (entity_type, action, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_audit_operator_entity ON gl_eudr_ars_audit_trail (operator_id, entity_type, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_ars_audit_changes ON gl_eudr_ars_audit_trail USING GIN (changes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- Triggers: updated_at auto-update
-- ============================================================================
RAISE NOTICE 'V122: Creating updated_at triggers...';

CREATE OR REPLACE FUNCTION fn_eudr_ars_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ars_review_cycles_updated_at
        BEFORE UPDATE ON gl_eudr_ars_review_cycles
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ars_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ars_review_tasks_updated_at
        BEFORE UPDATE ON gl_eudr_ars_review_tasks
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ars_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ars_deadlines_updated_at
        BEFORE UPDATE ON gl_eudr_ars_deadlines
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ars_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ars_checklists_updated_at
        BEFORE UPDATE ON gl_eudr_ars_checklists
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ars_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ars_calendar_events_updated_at
        BEFORE UPDATE ON gl_eudr_ars_calendar_events
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ars_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- Triggers: Audit trail auto-insert
-- ============================================================================
RAISE NOTICE 'V122: Creating audit trail triggers...';

CREATE OR REPLACE FUNCTION fn_eudr_ars_audit_insert()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO gl_eudr_ars_audit_trail (entity_type, entity_id, operator_id, action, actor_id, changes, timestamp)
    VALUES (TG_ARGV[0], NEW.*::TEXT::UUID, COALESCE(NEW.operator_id, ''), 'create', 'system', row_to_json(NEW)::JSONB, NOW());
    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION fn_eudr_ars_audit_update()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO gl_eudr_ars_audit_trail (entity_type, entity_id, operator_id, action, actor_id, changes, timestamp)
    VALUES (TG_ARGV[0], NEW.*::TEXT::UUID, COALESCE(NEW.operator_id, ''), 'update', 'system', jsonb_build_object('new', row_to_json(NEW)::JSONB), NOW());
    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Review cycles audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ars_cycles_audit_insert
        AFTER INSERT ON gl_eudr_ars_review_cycles
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ars_audit_insert('review_cycle');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ars_cycles_audit_update
        AFTER UPDATE ON gl_eudr_ars_review_cycles
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ars_audit_update('review_cycle');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Review tasks audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ars_tasks_audit_insert
        AFTER INSERT ON gl_eudr_ars_review_tasks
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ars_audit_insert('review_task');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ars_tasks_audit_update
        AFTER UPDATE ON gl_eudr_ars_review_tasks
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ars_audit_update('review_task');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Deadlines audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ars_deadlines_audit_insert
        AFTER INSERT ON gl_eudr_ars_deadlines
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ars_audit_insert('deadline');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ars_deadlines_audit_update
        AFTER UPDATE ON gl_eudr_ars_deadlines
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ars_audit_update('deadline');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Checklists audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ars_checklists_audit_insert
        AFTER INSERT ON gl_eudr_ars_checklists
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ars_audit_insert('checklist');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ars_checklists_audit_update
        AFTER UPDATE ON gl_eudr_ars_checklists
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ars_audit_update('checklist');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Year comparison audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ars_year_comparison_audit_insert
        AFTER INSERT ON gl_eudr_ars_year_comparison
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ars_audit_insert('year_comparison');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Calendar events audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ars_calendar_audit_insert
        AFTER INSERT ON gl_eudr_ars_calendar_events
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ars_audit_insert('calendar_event');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ars_calendar_audit_update
        AFTER UPDATE ON gl_eudr_ars_calendar_events
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ars_audit_update('calendar_event');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Review history audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ars_history_audit_insert
        AFTER INSERT ON gl_eudr_ars_review_history
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ars_audit_insert('review_history');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- Completion
-- ============================================================================

RAISE NOTICE 'V122: AGENT-EUDR-034 Annual Review Scheduler Agent tables created successfully';
RAISE NOTICE 'V122: Tables: 9 (review_cycles, review_tasks, deadlines, checklists, notifications, year_comparison, calendar_events, review_history, audit_trail)';
RAISE NOTICE 'V122: Hypertables: gl_eudr_ars_notifications, gl_eudr_ars_review_history, gl_eudr_ars_audit_trail (7/30/7-day chunks)';
RAISE NOTICE 'V122: Indexes: ~115';
RAISE NOTICE 'V122: Triggers: 5 updated_at + 12 audit trail';

COMMIT;
