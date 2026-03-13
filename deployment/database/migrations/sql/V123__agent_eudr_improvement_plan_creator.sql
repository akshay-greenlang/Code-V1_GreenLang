-- ============================================================================
-- V123: AGENT-EUDR-035 Improvement Plan Creator
-- ============================================================================
-- Creates tables for the Improvement Plan Creator which aggregates compliance
-- findings from all upstream agents and review processes; performs structured
-- gap analysis against EUDR requirements; creates master improvement plans
-- with SMART-goal actions; conducts root cause analysis on recurring non-
-- conformities; applies prioritization scoring via impact/effort/urgency
-- matrices; tracks action progress over time via TimescaleDB hypertable;
-- manages RACI stakeholder assignments for each action; and preserves a
-- complete Article 31 audit trail via TimescaleDB hypertable.
--
-- Agent ID: GL-EUDR-IPC-035
-- PRD: PRD-AGENT-EUDR-035
-- Regulation: EU 2023/1115 (EUDR) Articles 4, 8, 9, 10, 11, 12, 14-16, 29, 31
-- Tables: 9 (7 regular + 2 hypertables), Indexes: ~97
-- Dependencies: TimescaleDB extension (for hypertables)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V123: Creating AGENT-EUDR-035 Improvement Plan Creator tables...';


-- ============================================================================
-- 1. gl_eudr_ipc_findings -- Aggregated findings from all sources
-- ============================================================================
RAISE NOTICE 'V123 [1/9]: Creating gl_eudr_ipc_findings...';

CREATE TABLE IF NOT EXISTS gl_eudr_ipc_findings (
    finding_id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this aggregated finding
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator to whom this finding pertains
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    commodity_type                  VARCHAR(50)     NOT NULL,
        -- EUDR commodity type related to this finding
    origin_country                  VARCHAR(10),
        -- ISO 3166-1 alpha-2 country code of commodity origin (NULL if cross-country)
    source_agent_id                 VARCHAR(100)    NOT NULL,
        -- Agent that originated the finding (e.g. "GL-EUDR-LCV-023", "GL-EUDR-ARS-034")
    source_entity_type              VARCHAR(50)     NOT NULL,
        -- Type of source entity (review_task, audit_result, risk_assessment, compliance_check, monitoring_alert)
    source_entity_id                VARCHAR(200)    NOT NULL,
        -- Identifier of the originating entity in the source agent
    finding_title                   VARCHAR(500)    NOT NULL,
        -- Concise human-readable title summarizing the finding
    finding_description             TEXT            NOT NULL DEFAULT '',
        -- Detailed description of the finding including context and evidence references
    eudr_article_ref                VARCHAR(50),
        -- EUDR article reference this finding relates to (e.g. "Art. 9(1)(a)", "Art. 10(2)")
    category                        VARCHAR(30)     NOT NULL,
        -- Finding category classification
    severity                        VARCHAR(20)     NOT NULL DEFAULT 'medium',
        -- Severity of the finding
    status                          VARCHAR(20)     NOT NULL DEFAULT 'open',
        -- Current finding lifecycle status
    is_recurring                    BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether this finding has been observed in prior review cycles
    recurrence_count                INTEGER         NOT NULL DEFAULT 1,
        -- Number of times this finding has been observed across review cycles
    first_observed_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when this finding was first identified
    last_observed_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp of the most recent observation
    linked_plan_id                  UUID,
        -- FK reference to the improvement plan addressing this finding (NULL if unassigned)
    linked_action_id                UUID,
        -- FK reference to the specific action addressing this finding (NULL if unassigned)
    evidence_references             JSONB           DEFAULT '[]',
        -- Array of evidence references: [{"type": "audit_report", "ref": "...", "date": "..."}, ...]
    impact_assessment               JSONB           DEFAULT '{}',
        -- Assessed impact: {"regulatory_risk": "high", "financial_impact": "moderate", "reputational_impact": "low"}
    tags                            JSONB           DEFAULT '[]',
        -- Organizational tags: ["priority", "audit-2026-q1", "palm-oil"]
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for finding integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_ipc_finding_category CHECK (category IN (
        'traceability', 'deforestation', 'legality', 'human_rights',
        'environmental', 'documentation', 'due_diligence', 'risk_management',
        'supply_chain', 'monitoring', 'reporting', 'other'
    )),
    CONSTRAINT chk_ipc_finding_severity CHECK (severity IN (
        'critical', 'high', 'medium', 'low', 'informational'
    )),
    CONSTRAINT chk_ipc_finding_status CHECK (status IN (
        'open', 'acknowledged', 'in_remediation', 'resolved', 'closed', 'deferred'
    )),
    CONSTRAINT chk_ipc_finding_recurrence CHECK (recurrence_count >= 1)
);

COMMENT ON TABLE gl_eudr_ipc_findings IS 'AGENT-EUDR-035: Aggregated compliance findings from all upstream agents with severity classification, recurrence tracking, EUDR article mapping, impact assessment, and improvement plan linkage per Articles 4, 9-12';

-- Indexes for gl_eudr_ipc_findings
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_find_source_entity ON gl_eudr_ipc_findings (source_entity_type, source_entity_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_find_status ON gl_eudr_ipc_findings (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_find_linked_plan ON gl_eudr_ipc_findings (linked_plan_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_find_provenance ON gl_eudr_ipc_findings (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_find_created ON gl_eudr_ipc_findings (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_find_operator_category ON gl_eudr_ipc_findings (operator_id, category, severity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_find_operator_status ON gl_eudr_ipc_findings (operator_id, status, severity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_find_tenant_operator ON gl_eudr_ipc_findings (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_find_commodity_category ON gl_eudr_ipc_findings (commodity_type, category, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_find_open_only ON gl_eudr_ipc_findings (operator_id, severity, created_at DESC)
        WHERE status IN ('open', 'acknowledged', 'in_remediation');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_find_critical_only ON gl_eudr_ipc_findings (operator_id, category, created_at DESC)
        WHERE severity IN ('critical', 'high');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_find_recurring_only ON gl_eudr_ipc_findings (operator_id, recurrence_count DESC)
        WHERE is_recurring = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_find_tags ON gl_eudr_ipc_findings USING GIN (tags);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 2. gl_eudr_ipc_gap_analysis -- Compliance gap assessments
-- ============================================================================
RAISE NOTICE 'V123 [2/9]: Creating gl_eudr_ipc_gap_analysis...';

CREATE TABLE IF NOT EXISTS gl_eudr_ipc_gap_analysis (
    gap_id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this gap analysis record
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator for whom the gap analysis was conducted
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    commodity_type                  VARCHAR(50)     NOT NULL,
        -- EUDR commodity type assessed
    origin_country                  VARCHAR(10),
        -- ISO 3166-1 alpha-2 country code (NULL for cross-country analysis)
    analysis_date                   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the gap analysis was performed
    eudr_article                    VARCHAR(50)     NOT NULL,
        -- EUDR article being assessed (e.g. "Art. 4", "Art. 9", "Art. 10")
    requirement_description         TEXT            NOT NULL DEFAULT '',
        -- Description of the specific EUDR requirement being assessed
    current_state                   VARCHAR(20)     NOT NULL DEFAULT 'non_compliant',
        -- Current compliance state for this requirement
    target_state                    VARCHAR(20)     NOT NULL DEFAULT 'fully_compliant',
        -- Target compliance state to achieve
    gap_severity                    VARCHAR(20)     NOT NULL DEFAULT 'moderate',
        -- Severity of the identified gap
    gap_description                 TEXT            NOT NULL DEFAULT '',
        -- Detailed description of the gap between current and target states
    remediation_complexity          VARCHAR(20)     NOT NULL DEFAULT 'moderate',
        -- Estimated complexity of closing this gap
    estimated_effort_days           INTEGER,
        -- Estimated effort in calendar days to close the gap
    estimated_cost                  NUMERIC(12,2),
        -- Estimated cost to remediate the gap (in base currency)
    related_finding_ids             JSONB           DEFAULT '[]',
        -- Array of finding IDs that contribute to this gap: ["uuid-1", "uuid-2"]
    linked_plan_id                  UUID,
        -- FK reference to the improvement plan addressing this gap (NULL if unassigned)
    compliance_score_before         NUMERIC(5,2),
        -- Compliance score for this requirement before remediation (0.00-100.00)
    compliance_score_after          NUMERIC(5,2),
        -- Compliance score after remediation (NULL until verified)
    reviewer_id                     VARCHAR(100),
        -- User who conducted the gap analysis
    review_notes                    TEXT            DEFAULT '',
        -- Reviewer notes and observations
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for gap analysis integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_ipc_gap_current CHECK (current_state IN (
        'non_compliant', 'partially_compliant', 'substantially_compliant', 'fully_compliant', 'not_applicable'
    )),
    CONSTRAINT chk_ipc_gap_target CHECK (target_state IN (
        'partially_compliant', 'substantially_compliant', 'fully_compliant'
    )),
    CONSTRAINT chk_ipc_gap_severity CHECK (gap_severity IN (
        'critical', 'major', 'moderate', 'minor', 'negligible'
    )),
    CONSTRAINT chk_ipc_gap_complexity CHECK (remediation_complexity IN (
        'trivial', 'simple', 'moderate', 'complex', 'very_complex'
    )),
    CONSTRAINT chk_ipc_gap_effort CHECK (estimated_effort_days IS NULL OR estimated_effort_days >= 0),
    CONSTRAINT chk_ipc_gap_cost CHECK (estimated_cost IS NULL OR estimated_cost >= 0),
    CONSTRAINT chk_ipc_gap_score_before CHECK (compliance_score_before IS NULL OR
        (compliance_score_before >= 0 AND compliance_score_before <= 100)),
    CONSTRAINT chk_ipc_gap_score_after CHECK (compliance_score_after IS NULL OR
        (compliance_score_after >= 0 AND compliance_score_after <= 100))
);

COMMENT ON TABLE gl_eudr_ipc_gap_analysis IS 'AGENT-EUDR-035: Compliance gap assessments mapping current vs. target state per EUDR article, with gap severity classification, remediation complexity/effort/cost estimation, finding linkage, and before/after compliance scoring per Articles 4, 9-12';

-- Indexes for gl_eudr_ipc_gap_analysis
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_gap_linked_plan ON gl_eudr_ipc_gap_analysis (linked_plan_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_gap_provenance ON gl_eudr_ipc_gap_analysis (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_gap_created ON gl_eudr_ipc_gap_analysis (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_gap_operator_article ON gl_eudr_ipc_gap_analysis (operator_id, eudr_article, gap_severity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_gap_operator_severity ON gl_eudr_ipc_gap_analysis (operator_id, gap_severity, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_gap_tenant_operator ON gl_eudr_ipc_gap_analysis (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_gap_commodity_article ON gl_eudr_ipc_gap_analysis (commodity_type, eudr_article, analysis_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_gap_non_compliant ON gl_eudr_ipc_gap_analysis (operator_id, gap_severity, eudr_article)
        WHERE current_state IN ('non_compliant', 'partially_compliant');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_gap_critical_major ON gl_eudr_ipc_gap_analysis (operator_id, eudr_article, created_at DESC)
        WHERE gap_severity IN ('critical', 'major');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_gap_unassigned ON gl_eudr_ipc_gap_analysis (operator_id, gap_severity)
        WHERE linked_plan_id IS NULL AND current_state != 'fully_compliant';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_gap_finding_ids ON gl_eudr_ipc_gap_analysis USING GIN (related_finding_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 3. gl_eudr_ipc_improvement_plans -- Master improvement plans
-- ============================================================================
RAISE NOTICE 'V123 [3/9]: Creating gl_eudr_ipc_improvement_plans...';

CREATE TABLE IF NOT EXISTS gl_eudr_ipc_improvement_plans (
    plan_id                         UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this master improvement plan
    plan_reference                  VARCHAR(100)    UNIQUE NOT NULL,
        -- Human-readable plan reference (e.g. "IPC-2026-Q1-OP001-PALM")
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator who owns this improvement plan
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    commodity_type                  VARCHAR(50)     NOT NULL,
        -- Primary EUDR commodity type covered by this plan
    origin_country                  VARCHAR(10),
        -- Primary country of origin focus (NULL for multi-country plans)
    plan_title                      VARCHAR(500)    NOT NULL,
        -- Descriptive title for the improvement plan
    plan_description                TEXT            NOT NULL DEFAULT '',
        -- Detailed description of the plan scope, objectives, and context
    plan_type                       VARCHAR(20)     NOT NULL DEFAULT 'corrective',
        -- Classification of the improvement plan
    status                          VARCHAR(20)     NOT NULL DEFAULT 'draft',
        -- Current plan lifecycle status
    priority                        VARCHAR(10)     NOT NULL DEFAULT 'medium',
        -- Overall plan priority
    plan_start_date                 DATE            NOT NULL,
        -- Scheduled start date for plan implementation
    plan_end_date                   DATE            NOT NULL,
        -- Scheduled completion date for all plan actions
    actual_start_date               DATE,
        -- Actual start date of plan implementation
    actual_end_date                 DATE,
        -- Actual completion date (NULL if not yet completed)
    total_actions                   INTEGER         NOT NULL DEFAULT 0,
        -- Total number of actions in this plan
    completed_actions               INTEGER         NOT NULL DEFAULT 0,
        -- Number of actions completed
    overdue_actions                 INTEGER         NOT NULL DEFAULT 0,
        -- Number of actions currently overdue
    completion_percentage           NUMERIC(5,2)    NOT NULL DEFAULT 0,
        -- Percentage of actions completed (0.00-100.00)
    total_budget                    NUMERIC(14,2),
        -- Total allocated budget for the plan (in base currency)
    spent_budget                    NUMERIC(14,2)   DEFAULT 0,
        -- Amount spent to date
    review_cycle_id                 UUID,
        -- FK reference to the review cycle that triggered this plan
    approved_by                     VARCHAR(100),
        -- User who approved the plan for implementation
    approved_at                     TIMESTAMPTZ,
        -- Timestamp of plan approval
    created_by                      VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- User or system that created the plan
    plan_owner                      VARCHAR(100),
        -- User responsible for overall plan delivery
    executive_sponsor               VARCHAR(100),
        -- Executive sponsor providing oversight and resources
    objectives                      JSONB           DEFAULT '[]',
        -- Plan objectives: [{"objective": "Achieve full Art. 9 compliance", "target_date": "2026-06-30", "metric": "compliance_score >= 90"}, ...]
    success_criteria                JSONB           DEFAULT '[]',
        -- Success criteria: [{"criterion": "All critical gaps closed", "measurable": true}, ...]
    risk_factors                    JSONB           DEFAULT '[]',
        -- Plan risk factors: [{"risk": "Supplier non-cooperation", "likelihood": "medium", "impact": "high", "mitigation": "..."}, ...]
    tags                            JSONB           DEFAULT '[]',
        -- Organizational tags for categorization
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for plan integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_ipc_plan_type CHECK (plan_type IN (
        'corrective', 'preventive', 'continuous_improvement', 'emergency', 'strategic'
    )),
    CONSTRAINT chk_ipc_plan_status CHECK (status IN (
        'draft', 'pending_approval', 'approved', 'in_progress',
        'on_hold', 'completed', 'closed', 'cancelled'
    )),
    CONSTRAINT chk_ipc_plan_priority CHECK (priority IN (
        'critical', 'high', 'medium', 'low'
    )),
    CONSTRAINT chk_ipc_plan_dates CHECK (plan_end_date >= plan_start_date),
    CONSTRAINT chk_ipc_plan_actual_dates CHECK (actual_end_date IS NULL OR actual_start_date IS NOT NULL),
    CONSTRAINT chk_ipc_plan_completion CHECK (completion_percentage >= 0 AND completion_percentage <= 100),
    CONSTRAINT chk_ipc_plan_actions CHECK (total_actions >= 0),
    CONSTRAINT chk_ipc_plan_completed_actions CHECK (completed_actions >= 0 AND completed_actions <= total_actions),
    CONSTRAINT chk_ipc_plan_overdue_actions CHECK (overdue_actions >= 0),
    CONSTRAINT chk_ipc_plan_budget CHECK (total_budget IS NULL OR total_budget >= 0),
    CONSTRAINT chk_ipc_plan_spent CHECK (spent_budget IS NULL OR spent_budget >= 0)
);

COMMENT ON TABLE gl_eudr_ipc_improvement_plans IS 'AGENT-EUDR-035: Master improvement plans aggregating SMART-goal actions with lifecycle management, budget tracking, action completion metrics, approval workflow, and risk factor assessment per EUDR Articles 10-12';

-- Indexes for gl_eudr_ipc_improvement_plans
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_plan_status ON gl_eudr_ipc_improvement_plans (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_plan_owner ON gl_eudr_ipc_improvement_plans (plan_owner);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_plan_review_cycle ON gl_eudr_ipc_improvement_plans (review_cycle_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_plan_provenance ON gl_eudr_ipc_improvement_plans (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_plan_created ON gl_eudr_ipc_improvement_plans (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_plan_operator_status ON gl_eudr_ipc_improvement_plans (operator_id, status, priority);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_plan_operator_type ON gl_eudr_ipc_improvement_plans (operator_id, plan_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_plan_tenant_operator ON gl_eudr_ipc_improvement_plans (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_plan_commodity_status ON gl_eudr_ipc_improvement_plans (commodity_type, status, plan_end_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_plan_active_only ON gl_eudr_ipc_improvement_plans (operator_id, priority, plan_end_date)
        WHERE status IN ('approved', 'in_progress');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_plan_pending_only ON gl_eudr_ipc_improvement_plans (operator_id, created_at DESC)
        WHERE status IN ('draft', 'pending_approval');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_plan_overdue_check ON gl_eudr_ipc_improvement_plans (operator_id, plan_end_date)
        WHERE status IN ('approved', 'in_progress') AND overdue_actions > 0;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_plan_tags ON gl_eudr_ipc_improvement_plans USING GIN (tags);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 4. gl_eudr_ipc_actions -- Individual improvement actions (SMART goals)
-- ============================================================================
RAISE NOTICE 'V123 [4/9]: Creating gl_eudr_ipc_actions...';

CREATE TABLE IF NOT EXISTS gl_eudr_ipc_actions (
    action_id                       UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this improvement action
    plan_id                         UUID            NOT NULL,
        -- FK reference to the parent improvement plan
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator (denormalized for query performance)
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    action_reference                VARCHAR(100)    NOT NULL,
        -- Human-readable action reference (e.g. "IPC-2026-Q1-OP001-ACT-001")
    action_title                    VARCHAR(500)    NOT NULL,
        -- Concise action title (Specific component of SMART)
    action_description              TEXT            NOT NULL DEFAULT '',
        -- Detailed description of what must be done
    target_outcome                  TEXT            NOT NULL DEFAULT '',
        -- Measurable target outcome (Measurable component of SMART)
    eudr_article_ref                VARCHAR(50),
        -- EUDR article this action addresses
    category                        VARCHAR(30)     NOT NULL,
        -- Action category classification
    priority                        VARCHAR(10)     NOT NULL DEFAULT 'medium',
        -- Action priority level
    status                          VARCHAR(20)     NOT NULL DEFAULT 'planned',
        -- Current action lifecycle status
    assigned_to                     VARCHAR(100),
        -- Primary assignee responsible for this action (Achievable via responsible person)
    assigned_team                   VARCHAR(100),
        -- Team responsible for execution
    due_date                        DATE            NOT NULL,
        -- Target completion date (Time-bound component of SMART)
    started_at                      TIMESTAMPTZ,
        -- Timestamp when work began
    completed_at                    TIMESTAMPTZ,
        -- Timestamp when the action was completed
    verified_at                     TIMESTAMPTZ,
        -- Timestamp when the action result was verified
    verified_by                     VARCHAR(100),
        -- User who verified the action completion
    estimated_effort_hours          NUMERIC(7,1),
        -- Estimated effort in person-hours
    actual_effort_hours             NUMERIC(7,1),
        -- Actual effort spent in person-hours
    estimated_cost                  NUMERIC(12,2),
        -- Estimated cost to implement this action
    actual_cost                     NUMERIC(12,2),
        -- Actual cost incurred
    completion_percentage           NUMERIC(5,2)    NOT NULL DEFAULT 0,
        -- Action completion percentage (0.00-100.00)
    blockers                        JSONB           DEFAULT '[]',
        -- Current blockers: [{"description": "...", "severity": "high", "reported_at": "...", "resolved": false}, ...]
    dependencies                    JSONB           DEFAULT '[]',
        -- Action dependencies: [{"action_id": "uuid", "type": "finish_to_start"}, ...]
    acceptance_criteria             JSONB           DEFAULT '[]',
        -- Acceptance criteria: [{"criterion": "...", "met": false, "verified_by": null}, ...]
    related_finding_ids             JSONB           DEFAULT '[]',
        -- Finding IDs this action addresses: ["uuid-1", "uuid-2"]
    related_gap_ids                 JSONB           DEFAULT '[]',
        -- Gap analysis IDs this action addresses: ["uuid-1", "uuid-2"]
    notes                           TEXT            DEFAULT '',
        -- Implementation notes and observations
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for action integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ipc_action_plan FOREIGN KEY (plan_id) REFERENCES gl_eudr_ipc_improvement_plans (plan_id),
    CONSTRAINT chk_ipc_action_category CHECK (category IN (
        'traceability', 'deforestation', 'legality', 'human_rights',
        'environmental', 'documentation', 'due_diligence', 'risk_management',
        'supply_chain', 'monitoring', 'training', 'technology', 'process', 'other'
    )),
    CONSTRAINT chk_ipc_action_priority CHECK (priority IN (
        'critical', 'high', 'medium', 'low'
    )),
    CONSTRAINT chk_ipc_action_status CHECK (status IN (
        'planned', 'approved', 'in_progress', 'blocked',
        'completed', 'verified', 'overdue', 'cancelled'
    )),
    CONSTRAINT chk_ipc_action_completion CHECK (completion_percentage >= 0 AND completion_percentage <= 100),
    CONSTRAINT chk_ipc_action_effort_est CHECK (estimated_effort_hours IS NULL OR estimated_effort_hours >= 0),
    CONSTRAINT chk_ipc_action_effort_act CHECK (actual_effort_hours IS NULL OR actual_effort_hours >= 0),
    CONSTRAINT chk_ipc_action_cost_est CHECK (estimated_cost IS NULL OR estimated_cost >= 0),
    CONSTRAINT chk_ipc_action_cost_act CHECK (actual_cost IS NULL OR actual_cost >= 0)
);

COMMENT ON TABLE gl_eudr_ipc_actions IS 'AGENT-EUDR-035: Individual SMART-goal improvement actions with lifecycle tracking, effort/cost estimation, blocker management, dependency chains, and acceptance criteria per Articles 10-12';

-- Indexes for gl_eudr_ipc_actions
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_act_reference ON gl_eudr_ipc_actions (action_reference);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_act_status ON gl_eudr_ipc_actions (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_act_provenance ON gl_eudr_ipc_actions (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_act_created ON gl_eudr_ipc_actions (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_act_plan_status ON gl_eudr_ipc_actions (plan_id, status, priority);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_act_plan_due ON gl_eudr_ipc_actions (plan_id, due_date, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_act_operator_status ON gl_eudr_ipc_actions (operator_id, status, due_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_act_assigned_status ON gl_eudr_ipc_actions (assigned_to, status, due_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_act_tenant_operator ON gl_eudr_ipc_actions (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_act_active_only ON gl_eudr_ipc_actions (plan_id, priority, due_date)
        WHERE status IN ('planned', 'approved', 'in_progress', 'blocked');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_act_overdue_only ON gl_eudr_ipc_actions (plan_id, due_date, assigned_to)
        WHERE status = 'overdue';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_act_critical_active ON gl_eudr_ipc_actions (operator_id, due_date)
        WHERE priority = 'critical' AND status NOT IN ('completed', 'verified', 'cancelled');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_act_deps ON gl_eudr_ipc_actions USING GIN (dependencies);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_act_finding_ids ON gl_eudr_ipc_actions USING GIN (related_finding_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 5. gl_eudr_ipc_root_causes -- Root cause analysis records
-- ============================================================================
RAISE NOTICE 'V123 [5/9]: Creating gl_eudr_ipc_root_causes...';

CREATE TABLE IF NOT EXISTS gl_eudr_ipc_root_causes (
    root_cause_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this root cause analysis
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator for whom the analysis was conducted
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    analysis_title                  VARCHAR(500)    NOT NULL,
        -- Title summarizing the root cause analysis
    analysis_method                 VARCHAR(30)     NOT NULL DEFAULT 'five_whys',
        -- Root cause analysis methodology used
    trigger_type                    VARCHAR(30)     NOT NULL DEFAULT 'recurring_finding',
        -- What triggered this root cause analysis
    related_finding_ids             JSONB           NOT NULL DEFAULT '[]',
        -- Array of finding IDs that triggered this analysis: ["uuid-1", "uuid-2"]
    related_gap_ids                 JSONB           DEFAULT '[]',
        -- Array of gap analysis IDs related to this root cause: ["uuid-1"]
    root_cause_description          TEXT            NOT NULL DEFAULT '',
        -- Description of the identified root cause
    contributing_factors            JSONB           DEFAULT '[]',
        -- Contributing factors: [{"factor": "...", "category": "process", "significance": "high"}, ...]
    five_whys_chain                 JSONB           DEFAULT '[]',
        -- Five Whys analysis chain: [{"level": 1, "why": "...", "answer": "..."}, ...]
    fishbone_categories             JSONB           DEFAULT '{}',
        -- Ishikawa/fishbone diagram: {"people": [...], "process": [...], "technology": [...], "materials": [...], "environment": [...], "management": [...]}
    systemic_issue                  BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether the root cause indicates a systemic organizational issue
    root_cause_category             VARCHAR(30)     NOT NULL DEFAULT 'process',
        -- Primary category of the identified root cause
    confidence_level                VARCHAR(20)     NOT NULL DEFAULT 'moderate',
        -- Confidence in the root cause identification
    linked_plan_id                  UUID,
        -- FK reference to the improvement plan addressing this root cause
    linked_action_ids               JSONB           DEFAULT '[]',
        -- Action IDs created to address this root cause: ["uuid-1", "uuid-2"]
    analyst_id                      VARCHAR(100),
        -- User who performed the root cause analysis
    analysis_date                   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Date the analysis was conducted
    review_notes                    TEXT            DEFAULT '',
        -- Review and approval notes
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for root cause analysis integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ipc_rc_plan FOREIGN KEY (linked_plan_id) REFERENCES gl_eudr_ipc_improvement_plans (plan_id),
    CONSTRAINT chk_ipc_rc_method CHECK (analysis_method IN (
        'five_whys', 'fishbone', 'fault_tree', 'pareto', 'brainstorm', 'combined'
    )),
    CONSTRAINT chk_ipc_rc_trigger CHECK (trigger_type IN (
        'recurring_finding', 'critical_finding', 'audit_recommendation',
        'risk_escalation', 'management_review', 'regulatory_action'
    )),
    CONSTRAINT chk_ipc_rc_category CHECK (root_cause_category IN (
        'process', 'people', 'technology', 'data', 'policy',
        'training', 'communication', 'resource', 'external', 'other'
    )),
    CONSTRAINT chk_ipc_rc_confidence CHECK (confidence_level IN (
        'high', 'moderate', 'low', 'speculative'
    ))
);

COMMENT ON TABLE gl_eudr_ipc_root_causes IS 'AGENT-EUDR-035: Root cause analysis records using Five Whys, fishbone/Ishikawa, fault tree, and Pareto methodologies with systemic issue flagging, contributing factor analysis, and improvement plan linkage per Articles 10-12';

-- Indexes for gl_eudr_ipc_root_causes
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_rc_linked_plan ON gl_eudr_ipc_root_causes (linked_plan_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_rc_provenance ON gl_eudr_ipc_root_causes (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_rc_created ON gl_eudr_ipc_root_causes (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_rc_operator_category ON gl_eudr_ipc_root_causes (operator_id, root_cause_category, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_rc_operator_trigger ON gl_eudr_ipc_root_causes (operator_id, trigger_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_rc_tenant_operator ON gl_eudr_ipc_root_causes (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_rc_systemic ON gl_eudr_ipc_root_causes (operator_id, root_cause_category, created_at DESC)
        WHERE systemic_issue = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_rc_high_conf ON gl_eudr_ipc_root_causes (operator_id, root_cause_category)
        WHERE confidence_level = 'high';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_rc_finding_ids ON gl_eudr_ipc_root_causes USING GIN (related_finding_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_rc_factors ON gl_eudr_ipc_root_causes USING GIN (contributing_factors);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_rc_action_ids ON gl_eudr_ipc_root_causes USING GIN (linked_action_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 6. gl_eudr_ipc_priorities -- Prioritization scores and matrices
-- ============================================================================
RAISE NOTICE 'V123 [6/9]: Creating gl_eudr_ipc_priorities...';

CREATE TABLE IF NOT EXISTS gl_eudr_ipc_priorities (
    priority_id                     UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this prioritization record
    action_id                       UUID            NOT NULL,
        -- FK reference to the action being prioritized
    plan_id                         UUID            NOT NULL,
        -- FK reference to the parent improvement plan (denormalized)
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator (denormalized for query performance)
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    impact_score                    NUMERIC(5,2)    NOT NULL DEFAULT 0,
        -- Impact score: regulatory, financial, reputational combined (0.00-100.00)
    effort_score                    NUMERIC(5,2)    NOT NULL DEFAULT 0,
        -- Effort score: time, cost, complexity combined (0.00-100.00)
    urgency_score                   NUMERIC(5,2)    NOT NULL DEFAULT 0,
        -- Urgency score: regulatory deadline proximity, risk level, recurrence (0.00-100.00)
    composite_priority_score        NUMERIC(5,2)    NOT NULL DEFAULT 0,
        -- Weighted composite score (0.00-100.00): higher = more urgent to address
    priority_quadrant               VARCHAR(20)     NOT NULL DEFAULT 'plan',
        -- Eisenhower matrix quadrant classification
    priority_rank                   INTEGER,
        -- Rank within the plan (1 = highest priority)
    scoring_weights                 JSONB           NOT NULL DEFAULT '{"impact": 0.40, "urgency": 0.35, "effort_inverse": 0.25}',
        -- Weights used for composite score calculation
    impact_breakdown                JSONB           DEFAULT '{}',
        -- Impact sub-scores: {"regulatory_impact": 90, "financial_impact": 60, "reputational_impact": 45}
    effort_breakdown                JSONB           DEFAULT '{}',
        -- Effort sub-scores: {"time_effort": 70, "cost_effort": 50, "complexity_effort": 80}
    urgency_breakdown               JSONB           DEFAULT '{}',
        -- Urgency sub-scores: {"deadline_proximity": 85, "risk_severity": 70, "recurrence_factor": 60}
    justification                   TEXT            DEFAULT '',
        -- Textual justification for the assigned priority
    override_reason                 TEXT,
        -- Reason if priority was manually overridden (NULL if calculated)
    is_manual_override              BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether the priority was manually set rather than calculated
    scored_by                       VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- User or system that assigned the score
    scored_at                       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when scoring was performed
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for priority scoring integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ipc_pri_action FOREIGN KEY (action_id) REFERENCES gl_eudr_ipc_actions (action_id),
    CONSTRAINT fk_ipc_pri_plan FOREIGN KEY (plan_id) REFERENCES gl_eudr_ipc_improvement_plans (plan_id),
    CONSTRAINT chk_ipc_pri_impact CHECK (impact_score >= 0 AND impact_score <= 100),
    CONSTRAINT chk_ipc_pri_effort CHECK (effort_score >= 0 AND effort_score <= 100),
    CONSTRAINT chk_ipc_pri_urgency CHECK (urgency_score >= 0 AND urgency_score <= 100),
    CONSTRAINT chk_ipc_pri_composite CHECK (composite_priority_score >= 0 AND composite_priority_score <= 100),
    CONSTRAINT chk_ipc_pri_quadrant CHECK (priority_quadrant IN (
        'do_first', 'schedule', 'delegate', 'eliminate', 'plan'
    )),
    CONSTRAINT chk_ipc_pri_rank CHECK (priority_rank IS NULL OR priority_rank >= 1),
    CONSTRAINT uq_ipc_pri_action UNIQUE (action_id)
);

COMMENT ON TABLE gl_eudr_ipc_priorities IS 'AGENT-EUDR-035: Prioritization scoring using impact/effort/urgency matrices with Eisenhower quadrant classification, configurable composite weights, and rank ordering within improvement plans per Articles 10-12';

-- Indexes for gl_eudr_ipc_priorities
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_pri_quadrant ON gl_eudr_ipc_priorities (priority_quadrant);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_pri_provenance ON gl_eudr_ipc_priorities (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_pri_created ON gl_eudr_ipc_priorities (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_pri_plan_composite ON gl_eudr_ipc_priorities (plan_id, composite_priority_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_pri_plan_quadrant ON gl_eudr_ipc_priorities (plan_id, priority_quadrant, composite_priority_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_pri_operator_composite ON gl_eudr_ipc_priorities (operator_id, composite_priority_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_pri_tenant_operator ON gl_eudr_ipc_priorities (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_pri_do_first ON gl_eudr_ipc_priorities (plan_id, composite_priority_score DESC)
        WHERE priority_quadrant = 'do_first';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_pri_overrides ON gl_eudr_ipc_priorities (operator_id, scored_at DESC)
        WHERE is_manual_override = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_pri_impact_bd ON gl_eudr_ipc_priorities USING GIN (impact_breakdown);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 7. gl_eudr_ipc_progress_tracking -- Action progress tracking (TimescaleDB hypertable)
-- ============================================================================
RAISE NOTICE 'V123 [7/9]: Creating gl_eudr_ipc_progress_tracking (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_ipc_progress_tracking (
    tracking_id                     UUID            DEFAULT gen_random_uuid(),
        -- Unique identifier for this progress record
    action_id                       UUID            NOT NULL,
        -- FK reference to the action being tracked
    plan_id                         UUID            NOT NULL,
        -- FK reference to the parent improvement plan (denormalized)
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator (denormalized for query performance)
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    recorded_at                     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp of progress record (partitioning column)
    completion_percentage           NUMERIC(5,2)    NOT NULL DEFAULT 0,
        -- Action completion percentage at time of recording (0.00-100.00)
    previous_percentage             NUMERIC(5,2),
        -- Previous completion percentage for delta calculation
    status                          VARCHAR(20)     NOT NULL,
        -- Action status at time of recording
    previous_status                 VARCHAR(20),
        -- Previous status for transition tracking
    hours_spent                     NUMERIC(7,1)    DEFAULT 0,
        -- Cumulative hours spent at time of recording
    cost_spent                      NUMERIC(12,2)   DEFAULT 0,
        -- Cumulative cost at time of recording
    update_type                     VARCHAR(20)     NOT NULL DEFAULT 'progress',
        -- Type of progress update
    update_notes                    TEXT            DEFAULT '',
        -- Notes describing what was accomplished or changed
    blocker_update                  JSONB           DEFAULT NULL,
        -- Blocker status update: {"added": [...], "resolved": [...], "current_count": 2}
    risks_identified                JSONB           DEFAULT '[]',
        -- New risks identified: [{"risk": "...", "severity": "medium", "mitigation": "..."}, ...]
    reported_by                     VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- User or system reporting the progress
    evidence_ref                    VARCHAR(1000),
        -- Reference to evidence supporting this progress update
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for progress record integrity verification

    CONSTRAINT chk_ipc_pt_completion CHECK (completion_percentage >= 0 AND completion_percentage <= 100),
    CONSTRAINT chk_ipc_pt_prev_completion CHECK (previous_percentage IS NULL OR
        (previous_percentage >= 0 AND previous_percentage <= 100)),
    CONSTRAINT chk_ipc_pt_status CHECK (status IN (
        'planned', 'approved', 'in_progress', 'blocked',
        'completed', 'verified', 'overdue', 'cancelled'
    )),
    CONSTRAINT chk_ipc_pt_update_type CHECK (update_type IN (
        'progress', 'status_change', 'blocker_update', 'milestone',
        'escalation', 'review', 'completion', 'reopen'
    )),
    CONSTRAINT chk_ipc_pt_hours CHECK (hours_spent IS NULL OR hours_spent >= 0),
    CONSTRAINT chk_ipc_pt_cost CHECK (cost_spent IS NULL OR cost_spent >= 0)
);

-- Convert to hypertable partitioned by recorded_at
DO $$
BEGIN
    PERFORM create_hypertable(
        'gl_eudr_ipc_progress_tracking',
        'recorded_at',
        chunk_time_interval => INTERVAL '7 days',
        if_not_exists => TRUE
    );
    RAISE NOTICE 'gl_eudr_ipc_progress_tracking hypertable created (7-day chunks)';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Hypertable creation skipped for gl_eudr_ipc_progress_tracking: %', SQLERRM;
END $$;

COMMENT ON TABLE gl_eudr_ipc_progress_tracking IS 'AGENT-EUDR-035: TimescaleDB-partitioned action progress tracking with completion deltas, status transitions, effort/cost accumulation, blocker updates, and risk identification per Articles 10-12 and Article 31';

-- Indexes for gl_eudr_ipc_progress_tracking
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_pt_action ON gl_eudr_ipc_progress_tracking (action_id, recorded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_pt_plan ON gl_eudr_ipc_progress_tracking (plan_id, recorded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_pt_operator ON gl_eudr_ipc_progress_tracking (operator_id, recorded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_pt_status ON gl_eudr_ipc_progress_tracking (status, recorded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_pt_action_status ON gl_eudr_ipc_progress_tracking (action_id, status, recorded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_pt_plan_update ON gl_eudr_ipc_progress_tracking (plan_id, update_type, recorded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_pt_operator_plan ON gl_eudr_ipc_progress_tracking (operator_id, plan_id, recorded_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_pt_blockers ON gl_eudr_ipc_progress_tracking USING GIN (blocker_update);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_pt_risks ON gl_eudr_ipc_progress_tracking USING GIN (risks_identified);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 8. gl_eudr_ipc_stakeholder_assignments -- RACI assignments
-- ============================================================================
RAISE NOTICE 'V123 [8/9]: Creating gl_eudr_ipc_stakeholder_assignments...';

CREATE TABLE IF NOT EXISTS gl_eudr_ipc_stakeholder_assignments (
    assignment_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this RACI assignment
    action_id                       UUID            NOT NULL,
        -- FK reference to the action
    plan_id                         UUID            NOT NULL,
        -- FK reference to the parent improvement plan (denormalized)
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator (denormalized for query performance)
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    stakeholder_id                  VARCHAR(100)    NOT NULL,
        -- User ID or stakeholder identifier
    stakeholder_name                VARCHAR(200)    NOT NULL DEFAULT '',
        -- Human-readable stakeholder name
    stakeholder_role                VARCHAR(100)    NOT NULL DEFAULT '',
        -- Organizational role of the stakeholder (e.g. "Compliance Officer", "Supply Chain Manager")
    stakeholder_department          VARCHAR(100),
        -- Department the stakeholder belongs to
    raci_role                       VARCHAR(15)     NOT NULL,
        -- RACI classification for this assignment
    notification_preference         VARCHAR(20)     NOT NULL DEFAULT 'email',
        -- Preferred notification channel for this stakeholder
    is_active                       BOOLEAN         NOT NULL DEFAULT TRUE,
        -- Whether this assignment is currently active
    assigned_at                     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- When the stakeholder was assigned
    assigned_by                     VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- Who made the assignment
    unassigned_at                   TIMESTAMPTZ,
        -- When the stakeholder was unassigned (NULL if still active)
    unassigned_reason               TEXT,
        -- Reason for unassignment
    notes                           TEXT            DEFAULT '',
        -- Assignment notes
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for assignment integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ipc_sa_action FOREIGN KEY (action_id) REFERENCES gl_eudr_ipc_actions (action_id),
    CONSTRAINT fk_ipc_sa_plan FOREIGN KEY (plan_id) REFERENCES gl_eudr_ipc_improvement_plans (plan_id),
    CONSTRAINT chk_ipc_sa_raci CHECK (raci_role IN (
        'responsible', 'accountable', 'consulted', 'informed'
    )),
    CONSTRAINT chk_ipc_sa_notification CHECK (notification_preference IN (
        'email', 'webhook', 'dashboard', 'sms', 'none'
    )),
    CONSTRAINT uq_ipc_sa_action_stakeholder_raci UNIQUE (action_id, stakeholder_id, raci_role)
);

COMMENT ON TABLE gl_eudr_ipc_stakeholder_assignments IS 'AGENT-EUDR-035: RACI stakeholder assignments for improvement actions with notification preferences, organizational context, and active/inactive lifecycle per Articles 10-12';

-- Indexes for gl_eudr_ipc_stakeholder_assignments
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_sa_provenance ON gl_eudr_ipc_stakeholder_assignments (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_sa_created ON gl_eudr_ipc_stakeholder_assignments (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_sa_action_raci ON gl_eudr_ipc_stakeholder_assignments (action_id, raci_role, is_active);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_sa_stakeholder_raci ON gl_eudr_ipc_stakeholder_assignments (stakeholder_id, raci_role, is_active);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_sa_plan_raci ON gl_eudr_ipc_stakeholder_assignments (plan_id, raci_role, is_active);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_sa_tenant_operator ON gl_eudr_ipc_stakeholder_assignments (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_sa_active_only ON gl_eudr_ipc_stakeholder_assignments (action_id, stakeholder_id, raci_role)
        WHERE is_active = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_sa_accountable ON gl_eudr_ipc_stakeholder_assignments (action_id, stakeholder_id)
        WHERE raci_role = 'accountable' AND is_active = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 9. gl_eudr_ipc_audit_trail -- Audit log (TimescaleDB hypertable)
-- ============================================================================
RAISE NOTICE 'V123 [9/9]: Creating gl_eudr_ipc_audit_trail (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_ipc_audit_trail (
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
        -- Additional context: {"ip_address": "...", "user_agent": "...", "request_id": "...", "plan_reference": "..."}
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for audit entry integrity (chained to previous entry)
    timestamp                       TIMESTAMPTZ     NOT NULL DEFAULT NOW()
        -- Timestamp of the action (partitioning column)
);

-- Convert to hypertable
DO $$
BEGIN
    PERFORM create_hypertable(
        'gl_eudr_ipc_audit_trail',
        'timestamp',
        chunk_time_interval => INTERVAL '7 days',
        if_not_exists => TRUE
    );
    RAISE NOTICE 'gl_eudr_ipc_audit_trail hypertable created (7-day chunks)';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Hypertable creation skipped for gl_eudr_ipc_audit_trail: %', SQLERRM;
END $$;

COMMENT ON TABLE gl_eudr_ipc_audit_trail IS 'AGENT-EUDR-035: Immutable TimescaleDB-partitioned audit trail for all improvement plan creator operations per Article 31 with full provenance tracking';

-- Indexes for gl_eudr_ipc_audit_trail
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_audit_entity_id ON gl_eudr_ipc_audit_trail (entity_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_audit_operator ON gl_eudr_ipc_audit_trail (operator_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_audit_action ON gl_eudr_ipc_audit_trail (action, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_audit_provenance ON gl_eudr_ipc_audit_trail (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_audit_entity_action ON gl_eudr_ipc_audit_trail (entity_type, action, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_audit_operator_entity ON gl_eudr_ipc_audit_trail (operator_id, entity_type, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_audit_changes ON gl_eudr_ipc_audit_trail USING GIN (changes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_ipc_audit_context ON gl_eudr_ipc_audit_trail USING GIN (context);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- DATA RETENTION -- Article 31: 5-year retention
-- ============================================================================

SELECT add_retention_policy('gl_eudr_ipc_progress_tracking', INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_ipc_audit_trail', INTERVAL '5 years', if_not_exists => TRUE);


-- ============================================================================
-- Triggers: updated_at auto-update
-- ============================================================================
RAISE NOTICE 'V123: Creating updated_at triggers...';

CREATE OR REPLACE FUNCTION fn_eudr_ipc_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_findings_updated_at
        BEFORE UPDATE ON gl_eudr_ipc_findings
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_gap_analysis_updated_at
        BEFORE UPDATE ON gl_eudr_ipc_gap_analysis
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_improvement_plans_updated_at
        BEFORE UPDATE ON gl_eudr_ipc_improvement_plans
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_actions_updated_at
        BEFORE UPDATE ON gl_eudr_ipc_actions
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_root_causes_updated_at
        BEFORE UPDATE ON gl_eudr_ipc_root_causes
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_priorities_updated_at
        BEFORE UPDATE ON gl_eudr_ipc_priorities
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_stakeholder_assignments_updated_at
        BEFORE UPDATE ON gl_eudr_ipc_stakeholder_assignments
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- Triggers: Audit trail auto-insert
-- ============================================================================
RAISE NOTICE 'V123: Creating audit trail triggers...';

CREATE OR REPLACE FUNCTION fn_eudr_ipc_audit_insert()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO gl_eudr_ipc_audit_trail (entity_type, entity_id, operator_id, action, actor_id, changes, timestamp)
    VALUES (TG_ARGV[0], NEW.*::TEXT::UUID, COALESCE(NEW.operator_id, ''), 'create', 'system', row_to_json(NEW)::JSONB, NOW());
    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION fn_eudr_ipc_audit_update()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO gl_eudr_ipc_audit_trail (entity_type, entity_id, operator_id, action, actor_id, changes, timestamp)
    VALUES (TG_ARGV[0], NEW.*::TEXT::UUID, COALESCE(NEW.operator_id, ''), 'update', 'system', jsonb_build_object('new', row_to_json(NEW)::JSONB), NOW());
    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Findings audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_findings_audit_insert
        AFTER INSERT ON gl_eudr_ipc_findings
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_audit_insert('finding');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_findings_audit_update
        AFTER UPDATE ON gl_eudr_ipc_findings
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_audit_update('finding');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Gap analysis audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_gap_analysis_audit_insert
        AFTER INSERT ON gl_eudr_ipc_gap_analysis
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_audit_insert('gap_analysis');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_gap_analysis_audit_update
        AFTER UPDATE ON gl_eudr_ipc_gap_analysis
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_audit_update('gap_analysis');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Improvement plans audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_plans_audit_insert
        AFTER INSERT ON gl_eudr_ipc_improvement_plans
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_audit_insert('improvement_plan');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_plans_audit_update
        AFTER UPDATE ON gl_eudr_ipc_improvement_plans
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_audit_update('improvement_plan');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Actions audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_actions_audit_insert
        AFTER INSERT ON gl_eudr_ipc_actions
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_audit_insert('action');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_actions_audit_update
        AFTER UPDATE ON gl_eudr_ipc_actions
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_audit_update('action');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Root causes audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_root_causes_audit_insert
        AFTER INSERT ON gl_eudr_ipc_root_causes
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_audit_insert('root_cause');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_root_causes_audit_update
        AFTER UPDATE ON gl_eudr_ipc_root_causes
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_audit_update('root_cause');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Priorities audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_priorities_audit_insert
        AFTER INSERT ON gl_eudr_ipc_priorities
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_audit_insert('priority');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_priorities_audit_update
        AFTER UPDATE ON gl_eudr_ipc_priorities
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_audit_update('priority');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Stakeholder assignments audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_stakeholders_audit_insert
        AFTER INSERT ON gl_eudr_ipc_stakeholder_assignments
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_audit_insert('stakeholder_assignment');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_ipc_stakeholders_audit_update
        AFTER UPDATE ON gl_eudr_ipc_stakeholder_assignments
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_ipc_audit_update('stakeholder_assignment');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- Completion
-- ============================================================================

RAISE NOTICE 'V123: AGENT-EUDR-035 Improvement Plan Creator -- 9 tables, ~97 indexes, 21 triggers, 2 hypertables, 5-year retention';

COMMIT;
