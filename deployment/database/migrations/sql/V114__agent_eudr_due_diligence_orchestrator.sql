-- ============================================================================
-- V114: AGENT-EUDR-026 Due Diligence Orchestrator Agent
-- ============================================================================
-- Creates tables for DAG-based workflow orchestration of all 25 upstream EUDR
-- agents across three mandatory due diligence phases (Article 8): information
-- gathering (Article 9), risk assessment (Article 10), and risk mitigation
-- (Article 11). Manages workflow state with checkpointing and resume, enforces
-- quality gates between phases, handles errors with circuit breakers and
-- dead letter queues, and produces audit-ready due diligence packages.
--
-- Schema: eudr_due_diligence
-- Tables: 13 (9 regular + 4 hypertables)
-- Hypertables: gl_eudr_ddo_workflow_checkpoints (7d chunks),
--              gl_eudr_ddo_quality_gate_evaluations (30d chunks),
--              gl_eudr_ddo_agent_execution_log (7d chunks),
--              gl_eudr_ddo_workflow_audit_trail (30d chunks)
-- Continuous Aggregates: 2 (workflow_checkpoints_hourly, agent_execution_stats_daily)
-- Retention Policies: 4 (all 5 years per EUDR Article 31)
-- Compression Policies: 4 (on all hypertables)
-- Indexes: ~200
--
-- Dependencies: TimescaleDB extension (V002)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V114: Creating AGENT-EUDR-026 Due Diligence Orchestrator tables...';

-- ============================================================================
-- SCHEMA
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS eudr_due_diligence;
SET search_path TO eudr_due_diligence, public;


-- ============================================================================
-- 1. gl_eudr_ddo_workflow_definitions — Templates and custom workflow DAGs
-- ============================================================================
RAISE NOTICE 'V114 [1/13]: Creating gl_eudr_ddo_workflow_definitions...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddo_workflow_definitions (
    workflow_def_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    name                        VARCHAR(500)    NOT NULL,
        -- Human-readable workflow definition name
    description                 TEXT,
        -- Detailed description of the workflow purpose and scope
    workflow_type               VARCHAR(20)     NOT NULL DEFAULT 'standard',
        -- Type of due diligence workflow
    commodity                   VARCHAR(50),
        -- EUDR commodity this template targets (null for generic)
    version                     INTEGER         NOT NULL DEFAULT 1,
        -- Version number for this definition (immutable per version)
    agent_nodes                 JSONB           NOT NULL DEFAULT '[]'::jsonb,
        -- Array of agent node definitions with phase, layer, criticality
    dependency_edges            JSONB           NOT NULL DEFAULT '[]'::jsonb,
        -- Array of dependency edges defining the DAG topology
    quality_gates               JSONB           NOT NULL DEFAULT '[]'::jsonb,
        -- Quality gate definitions with thresholds and check specs
    config                      JSONB           NOT NULL DEFAULT '{}'::jsonb,
        -- Workflow configuration: concurrency limits, timeouts, retry overrides
    is_system_template          BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether this is a built-in system template (immutable by users)
    is_active                   BOOLEAN         NOT NULL DEFAULT TRUE,
        -- Whether this definition is available for use
    max_concurrency             INTEGER         NOT NULL DEFAULT 10,
        -- Maximum parallel agent executions for workflows using this definition
    estimated_duration_seconds  INTEGER,
        -- Estimated total workflow duration in seconds
    created_by                  VARCHAR(100),
        -- User or system actor who created this definition
    tenant_id                   UUID,
        -- Multi-tenant isolation identifier
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_ddo_wfdef_workflow_type CHECK (workflow_type IN (
        'standard', 'simplified', 'custom'
    )),
    CONSTRAINT chk_ddo_wfdef_commodity CHECK (commodity IS NULL OR commodity IN (
        'cattle', 'cocoa', 'coffee', 'palm_oil', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_ddo_wfdef_version CHECK (version >= 1),
    CONSTRAINT chk_ddo_wfdef_max_concurrency CHECK (max_concurrency >= 1 AND max_concurrency <= 50),
    CONSTRAINT uq_ddo_wfdef_name_version UNIQUE (name, version)
);

DO $$ BEGIN
    CREATE INDEX idx_ddo_wfdef_workflow_type ON gl_eudr_ddo_workflow_definitions (workflow_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfdef_commodity ON gl_eudr_ddo_workflow_definitions (commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfdef_is_system ON gl_eudr_ddo_workflow_definitions (is_system_template) WHERE is_system_template = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfdef_is_active ON gl_eudr_ddo_workflow_definitions (is_active) WHERE is_active = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfdef_tenant ON gl_eudr_ddo_workflow_definitions (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfdef_created_at ON gl_eudr_ddo_workflow_definitions (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfdef_commodity_type ON gl_eudr_ddo_workflow_definitions (commodity, workflow_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfdef_tenant_active ON gl_eudr_ddo_workflow_definitions (tenant_id, is_active) WHERE is_active = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfdef_agent_nodes ON gl_eudr_ddo_workflow_definitions USING GIN (agent_nodes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfdef_dependency_edges ON gl_eudr_ddo_workflow_definitions USING GIN (dependency_edges);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfdef_quality_gates ON gl_eudr_ddo_workflow_definitions USING GIN (quality_gates);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfdef_config ON gl_eudr_ddo_workflow_definitions USING GIN (config);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

DO $$ BEGIN
    CREATE INDEX idx_ddo_wfdef_name ON gl_eudr_ddo_workflow_definitions (name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfdef_created_by ON gl_eudr_ddo_workflow_definitions (created_by) WHERE created_by IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfdef_type_commodity_active ON gl_eudr_ddo_workflow_definitions (workflow_type, commodity, is_active) WHERE is_active = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfdef_system_commodity ON gl_eudr_ddo_workflow_definitions (commodity, workflow_type) WHERE is_system_template = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_ddo_workflow_definitions IS 'DAG-based workflow definitions and templates for EUDR due diligence orchestration. Each definition specifies 25-agent topology, dependency edges, quality gates, and configuration for standard or simplified due diligence per EUDR Articles 8, 9, 10, 11, 13.';


-- ============================================================================
-- 2. gl_eudr_ddo_workflow_executions — One row per due diligence run
-- ============================================================================
RAISE NOTICE 'V114 [2/13]: Creating gl_eudr_ddo_workflow_executions...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddo_workflow_executions (
    workflow_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_def_id             UUID            NOT NULL,
        -- Reference to the workflow definition used for this execution
    operator_id                 UUID            NOT NULL,
        -- EUDR operator initiating this due diligence workflow
    commodity                   VARCHAR(50)     NOT NULL,
        -- Target commodity for this due diligence run
    product_ids                 JSONB           NOT NULL DEFAULT '[]'::jsonb,
        -- Array of product/shipment identifiers covered by this workflow
    batch_id                    UUID,
        -- Batch execution group (null if standalone)
    status                      VARCHAR(30)     NOT NULL DEFAULT 'created',
        -- Current workflow state (22-state FSM)
    current_phase               VARCHAR(30)     NOT NULL DEFAULT 'information_gathering',
        -- Current due diligence phase
    workflow_type               VARCHAR(20)     NOT NULL DEFAULT 'standard',
        -- Workflow type: standard, simplified, custom
    agent_statuses              JSONB           NOT NULL DEFAULT '{}'::jsonb,
        -- Per-agent execution status map {agent_id: status}
    agent_outputs_summary       JSONB           NOT NULL DEFAULT '{}'::jsonb,
        -- Summarized output data from each completed agent
    progress                    JSONB           NOT NULL DEFAULT '{}'::jsonb,
        -- Real-time progress data: agents completed, ETA, phase progress
    error_summary               JSONB,
        -- Aggregated error information if workflow has failures
    package_id                  UUID,
        -- Reference to generated due diligence package (set on completion)
    risk_profile                JSONB,
        -- Composite risk assessment results from Phase 2
    mitigation_summary          JSONB,
        -- Risk mitigation results from Phase 3
    composite_risk_score        NUMERIC(5,2),
        -- Final composite risk score (0-100)
    residual_risk_score         NUMERIC(5,2),
        -- Post-mitigation residual risk score
    total_agents                INTEGER         NOT NULL DEFAULT 25,
        -- Total number of agents in this workflow
    completed_agents            INTEGER         NOT NULL DEFAULT 0,
        -- Number of agents that have completed successfully
    failed_agents               INTEGER         NOT NULL DEFAULT 0,
        -- Number of agents that have failed permanently
    skipped_agents              INTEGER         NOT NULL DEFAULT 0,
        -- Number of agents that were skipped
    retry_count                 INTEGER         NOT NULL DEFAULT 0,
        -- Total retry attempts across all agents
    checkpoint_count            INTEGER         NOT NULL DEFAULT 0,
        -- Total checkpoints created for this workflow
    is_simplified               BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether this is a simplified due diligence workflow (Article 13)
    priority                    INTEGER         NOT NULL DEFAULT 5,
        -- Execution priority (1=highest, 10=lowest)
    initiated_by                VARCHAR(100),
        -- User or system actor who initiated the workflow
    tenant_id                   UUID            NOT NULL,
        -- Multi-tenant isolation identifier
    started_at                  TIMESTAMPTZ,
        -- Actual execution start timestamp
    completed_at                TIMESTAMPTZ,
        -- Execution completion timestamp
    estimated_completion        TIMESTAMPTZ,
        -- Estimated completion time based on critical path
    paused_at                   TIMESTAMPTZ,
        -- Timestamp when workflow was paused (null if not paused)
    cancelled_at                TIMESTAMPTZ,
        -- Timestamp when workflow was cancelled (null if not cancelled)
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ddo_wfexec_def FOREIGN KEY (workflow_def_id)
        REFERENCES gl_eudr_ddo_workflow_definitions(workflow_def_id),
    CONSTRAINT chk_ddo_wfexec_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'palm_oil', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_ddo_wfexec_status CHECK (status IN (
        'created', 'validating', 'validation_failed', 'ready',
        'phase1_running', 'phase1_complete',
        'qg1_evaluating', 'qg1_passed', 'qg1_failed', 'qg1_overridden',
        'phase2_running', 'phase2_complete',
        'qg2_evaluating', 'qg2_passed', 'qg2_failed', 'qg2_overridden',
        'mitigation_bypassed',
        'phase3_running',
        'qg3_evaluating', 'qg3_failed', 'qg3_overridden',
        'package_generating', 'completed',
        'paused', 'cancelled', 'terminated', 'agent_failed'
    )),
    CONSTRAINT chk_ddo_wfexec_phase CHECK (current_phase IN (
        'initialization', 'information_gathering', 'risk_assessment',
        'risk_mitigation', 'package_generation', 'completed'
    )),
    CONSTRAINT chk_ddo_wfexec_workflow_type CHECK (workflow_type IN (
        'standard', 'simplified', 'custom'
    )),
    CONSTRAINT chk_ddo_wfexec_priority CHECK (priority >= 1 AND priority <= 10),
    CONSTRAINT chk_ddo_wfexec_agents CHECK (
        completed_agents >= 0 AND failed_agents >= 0 AND skipped_agents >= 0
        AND total_agents >= 1
    ),
    CONSTRAINT chk_ddo_wfexec_risk CHECK (
        (composite_risk_score IS NULL OR (composite_risk_score >= 0 AND composite_risk_score <= 100))
        AND (residual_risk_score IS NULL OR (residual_risk_score >= 0 AND residual_risk_score <= 100))
    )
);

DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_operator ON gl_eudr_ddo_workflow_executions (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_status ON gl_eudr_ddo_workflow_executions (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_commodity ON gl_eudr_ddo_workflow_executions (commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_batch ON gl_eudr_ddo_workflow_executions (batch_id) WHERE batch_id IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_operator_status ON gl_eudr_ddo_workflow_executions (operator_id, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_operator_commodity ON gl_eudr_ddo_workflow_executions (operator_id, commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_tenant ON gl_eudr_ddo_workflow_executions (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_tenant_status ON gl_eudr_ddo_workflow_executions (tenant_id, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_created_at ON gl_eudr_ddo_workflow_executions (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_started_at ON gl_eudr_ddo_workflow_executions (started_at DESC) WHERE started_at IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_completed_at ON gl_eudr_ddo_workflow_executions (completed_at DESC) WHERE completed_at IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_phase ON gl_eudr_ddo_workflow_executions (current_phase);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_workflow_type ON gl_eudr_ddo_workflow_executions (workflow_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_is_simplified ON gl_eudr_ddo_workflow_executions (is_simplified) WHERE is_simplified = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_priority ON gl_eudr_ddo_workflow_executions (priority, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_package ON gl_eudr_ddo_workflow_executions (package_id) WHERE package_id IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_active ON gl_eudr_ddo_workflow_executions (status, priority)
        WHERE status IN ('phase1_running', 'phase2_running', 'phase3_running', 'package_generating');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_paused ON gl_eudr_ddo_workflow_executions (paused_at)
        WHERE status = 'paused';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_agent_statuses ON gl_eudr_ddo_workflow_executions USING GIN (agent_statuses);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_product_ids ON gl_eudr_ddo_workflow_executions USING GIN (product_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_progress ON gl_eudr_ddo_workflow_executions USING GIN (progress);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_risk_profile ON gl_eudr_ddo_workflow_executions USING GIN (risk_profile) WHERE risk_profile IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_def_id ON gl_eudr_ddo_workflow_executions (workflow_def_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_initiated_by ON gl_eudr_ddo_workflow_executions (initiated_by) WHERE initiated_by IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_risk_score ON gl_eudr_ddo_workflow_executions (composite_risk_score DESC) WHERE composite_risk_score IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_residual_score ON gl_eudr_ddo_workflow_executions (residual_risk_score DESC) WHERE residual_risk_score IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_tenant_commodity ON gl_eudr_ddo_workflow_executions (tenant_id, commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_tenant_phase ON gl_eudr_ddo_workflow_executions (tenant_id, current_phase);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_operator_phase ON gl_eudr_ddo_workflow_executions (operator_id, current_phase);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_completed ON gl_eudr_ddo_workflow_executions (completed_at DESC)
        WHERE status = 'completed';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_failed ON gl_eudr_ddo_workflow_executions (updated_at DESC)
        WHERE status IN ('terminated', 'agent_failed');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_error_summary ON gl_eudr_ddo_workflow_executions USING GIN (error_summary) WHERE error_summary IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfexec_mitigation ON gl_eudr_ddo_workflow_executions USING GIN (mitigation_summary) WHERE mitigation_summary IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_ddo_workflow_executions IS 'Active and historical due diligence workflow executions. Each row represents one end-to-end due diligence run across 25 EUDR agents with 22-state FSM, phase tracking, composite risk scoring, and full provenance chain per EUDR Articles 4, 8, 9, 10, 11.';


-- ============================================================================
-- 3. gl_eudr_ddo_due_diligence_packages — Generated evidence bundles
-- ============================================================================
RAISE NOTICE 'V114 [3/13]: Creating gl_eudr_ddo_due_diligence_packages...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddo_due_diligence_packages (
    package_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id                 UUID            NOT NULL,
        -- Reference to the workflow execution that produced this package
    operator_id                 UUID            NOT NULL,
        -- Operator identifier for multi-tenant isolation
    commodity                   VARCHAR(50)     NOT NULL,
        -- Commodity covered by this due diligence package
    product_ids                 JSONB           NOT NULL DEFAULT '[]'::jsonb,
        -- Product/shipment identifiers covered by this package
    workflow_type               VARCHAR(20)     NOT NULL,
        -- Workflow type used: standard, simplified, custom
    dds_json                    JSONB           NOT NULL,
        -- DDS-compatible JSON per EUDR Article 12(2)(a-j)
    sections                    JSONB           NOT NULL DEFAULT '[]'::jsonb,
        -- Package report sections (9 sections: Product, Origin, Geo, Deforestation, Legal, Risk, Mitigation, Supply Chain, Audit)
    quality_gate_summary        JSONB           NOT NULL DEFAULT '{}'::jsonb,
        -- Summary of all quality gate evaluations (QG-1, QG-2, QG-3)
    risk_profile                JSONB           NOT NULL DEFAULT '{}'::jsonb,
        -- Composite risk assessment profile with per-dimension scores
    mitigation_summary          JSONB,
        -- Mitigation strategy and adequacy results (null if bypassed)
    provenance_chain            JSONB           NOT NULL DEFAULT '[]'::jsonb,
        -- SHA-256 provenance chain from all workflow checkpoints
    package_hash                VARCHAR(64)     NOT NULL,
        -- SHA-256 integrity hash of the complete package content
    languages                   JSONB           NOT NULL DEFAULT '["en"]'::jsonb,
        -- Languages included in this package
    pdf_refs                    JSONB           DEFAULT '{}'::jsonb,
        -- S3 references to generated PDF reports per language
    zip_ref                     VARCHAR(500),
        -- S3 reference to the complete ZIP evidence bundle
    html_ref                    VARCHAR(500),
        -- S3 reference to the HTML evidence report
    version                     INTEGER         NOT NULL DEFAULT 1,
        -- Package version (for amendments)
    is_submitted                BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether this package has been submitted to EU Information System
    submitted_at                TIMESTAMPTZ,
        -- Timestamp of DDS submission to EU IS
    submission_ref              VARCHAR(200),
        -- EU Information System submission reference number
    composite_risk_score        NUMERIC(5,2),
        -- Final composite risk score at time of package generation
    residual_risk_score         NUMERIC(5,2),
        -- Post-mitigation residual risk score
    total_agents_executed       INTEGER         NOT NULL DEFAULT 0,
        -- Number of agents that contributed to this package
    total_evidence_artifacts    INTEGER         NOT NULL DEFAULT 0,
        -- Total number of evidence artifacts in the package
    tenant_id                   UUID            NOT NULL,
        -- Multi-tenant isolation identifier
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ddo_pkg_workflow FOREIGN KEY (workflow_id)
        REFERENCES gl_eudr_ddo_workflow_executions(workflow_id),
    CONSTRAINT chk_ddo_pkg_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'palm_oil', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_ddo_pkg_workflow_type CHECK (workflow_type IN (
        'standard', 'simplified', 'custom'
    )),
    CONSTRAINT chk_ddo_pkg_version CHECK (version >= 1),
    CONSTRAINT chk_ddo_pkg_risk CHECK (
        (composite_risk_score IS NULL OR (composite_risk_score >= 0 AND composite_risk_score <= 100))
        AND (residual_risk_score IS NULL OR (residual_risk_score >= 0 AND residual_risk_score <= 100))
    )
);

DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_operator ON gl_eudr_ddo_due_diligence_packages (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_workflow ON gl_eudr_ddo_due_diligence_packages (workflow_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_commodity ON gl_eudr_ddo_due_diligence_packages (commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_tenant ON gl_eudr_ddo_due_diligence_packages (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_created_at ON gl_eudr_ddo_due_diligence_packages (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_operator_commodity ON gl_eudr_ddo_due_diligence_packages (operator_id, commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_tenant_commodity ON gl_eudr_ddo_due_diligence_packages (tenant_id, commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_submitted ON gl_eudr_ddo_due_diligence_packages (is_submitted, submitted_at)
        WHERE is_submitted = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_hash ON gl_eudr_ddo_due_diligence_packages (package_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_dds ON gl_eudr_ddo_due_diligence_packages USING GIN (dds_json);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_sections ON gl_eudr_ddo_due_diligence_packages USING GIN (sections);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_provenance ON gl_eudr_ddo_due_diligence_packages USING GIN (provenance_chain);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_product_ids ON gl_eudr_ddo_due_diligence_packages USING GIN (product_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_risk_profile ON gl_eudr_ddo_due_diligence_packages USING GIN (risk_profile);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_version ON gl_eudr_ddo_due_diligence_packages (workflow_id, version DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_submitted_at ON gl_eudr_ddo_due_diligence_packages (submitted_at DESC) WHERE submitted_at IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_risk_score ON gl_eudr_ddo_due_diligence_packages (composite_risk_score DESC) WHERE composite_risk_score IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_workflow_version ON gl_eudr_ddo_due_diligence_packages (workflow_id, version);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_operator_created ON gl_eudr_ddo_due_diligence_packages (operator_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_tenant_created ON gl_eudr_ddo_due_diligence_packages (tenant_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_qg_summary ON gl_eudr_ddo_due_diligence_packages USING GIN (quality_gate_summary);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_pkg_languages ON gl_eudr_ddo_due_diligence_packages USING GIN (languages);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_ddo_due_diligence_packages IS 'Audit-ready due diligence evidence packages generated from completed workflows. Contains DDS-compatible JSON per Article 12(2)(a-j), 9-section reports, SHA-256 provenance chain, composite risk profiles, and multi-language PDF/ZIP references for regulatory inspection.';


-- ============================================================================
-- 4. gl_eudr_ddo_circuit_breaker_state — Per-agent, per-operator circuit breaker
-- ============================================================================
RAISE NOTICE 'V114 [4/13]: Creating gl_eudr_ddo_circuit_breaker_state...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddo_circuit_breaker_state (
    agent_id                    VARCHAR(20)     NOT NULL,
        -- EUDR agent identifier (e.g., EUDR-001, EUDR-025)
    operator_id                 UUID            NOT NULL,
        -- Operator-scoped circuit breaker (isolation per tenant)
    state                       VARCHAR(15)     NOT NULL DEFAULT 'closed',
        -- Circuit breaker state: closed, open, half_open
    failure_count               INTEGER         NOT NULL DEFAULT 0,
        -- Consecutive failure count (resets on success)
    success_count               INTEGER         NOT NULL DEFAULT 0,
        -- Consecutive success count in half_open state
    failure_threshold           INTEGER         NOT NULL DEFAULT 5,
        -- Failures before opening the circuit
    success_threshold           INTEGER         NOT NULL DEFAULT 2,
        -- Successes in half_open before closing the circuit
    reset_timeout_seconds       INTEGER         NOT NULL DEFAULT 60,
        -- Seconds to wait in open state before transitioning to half_open
    last_failure_at             TIMESTAMPTZ,
        -- Timestamp of most recent failure
    last_failure_error          TEXT,
        -- Error message from most recent failure
    last_success_at             TIMESTAMPTZ,
        -- Timestamp of most recent success
    opened_at                   TIMESTAMPTZ,
        -- Timestamp when circuit breaker transitioned to open
    half_opened_at              TIMESTAMPTZ,
        -- Timestamp when circuit breaker transitioned to half_open
    total_failures              BIGINT          NOT NULL DEFAULT 0,
        -- Cumulative lifetime failure count
    total_trips                 BIGINT          NOT NULL DEFAULT 0,
        -- Cumulative number of times circuit has opened
    tenant_id                   UUID            NOT NULL,
        -- Multi-tenant isolation identifier
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (agent_id, operator_id),

    CONSTRAINT chk_ddo_cb_state CHECK (state IN (
        'closed', 'open', 'half_open'
    )),
    CONSTRAINT chk_ddo_cb_failure_count CHECK (failure_count >= 0),
    CONSTRAINT chk_ddo_cb_success_count CHECK (success_count >= 0),
    CONSTRAINT chk_ddo_cb_failure_threshold CHECK (failure_threshold >= 1),
    CONSTRAINT chk_ddo_cb_success_threshold CHECK (success_threshold >= 1),
    CONSTRAINT chk_ddo_cb_reset_timeout CHECK (reset_timeout_seconds >= 1)
);

DO $$ BEGIN
    CREATE INDEX idx_ddo_cb_state ON gl_eudr_ddo_circuit_breaker_state (state);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_cb_operator ON gl_eudr_ddo_circuit_breaker_state (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_cb_tenant ON gl_eudr_ddo_circuit_breaker_state (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_cb_open ON gl_eudr_ddo_circuit_breaker_state (state, opened_at)
        WHERE state = 'open';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_cb_half_open ON gl_eudr_ddo_circuit_breaker_state (state, half_opened_at)
        WHERE state = 'half_open';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_cb_agent_state ON gl_eudr_ddo_circuit_breaker_state (agent_id, state);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_cb_tenant_state ON gl_eudr_ddo_circuit_breaker_state (tenant_id, state);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_cb_updated ON gl_eudr_ddo_circuit_breaker_state (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_ddo_circuit_breaker_state IS 'Per-agent, per-operator circuit breaker state for resilient error handling. Implements closed/open/half_open FSM with configurable failure thresholds, reset timeouts, and success probes. Ensures per-tenant isolation so one operator failure does not cascade.';


-- ============================================================================
-- 5. gl_eudr_ddo_dead_letter_queue — Unrecoverable agent failures
-- ============================================================================
RAISE NOTICE 'V114 [5/13]: Creating gl_eudr_ddo_dead_letter_queue...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddo_dead_letter_queue (
    dlq_id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id                 UUID            NOT NULL,
        -- Workflow execution that encountered the failure
    agent_id                    VARCHAR(20)     NOT NULL,
        -- EUDR agent that failed permanently
    operator_id                 UUID            NOT NULL,
        -- Operator who owns the workflow
    attempt_number              INTEGER         NOT NULL,
        -- Final attempt number when moved to DLQ
    error_type                  VARCHAR(50)     NOT NULL,
        -- Error type classification (e.g., TimeoutError, ValidationError)
    error_message               TEXT            NOT NULL,
        -- Full error message
    error_classification        VARCHAR(20)     NOT NULL,
        -- Error classification: permanent, transient_exhausted, degraded
    error_stack_trace           TEXT,
        -- Stack trace for debugging
    input_data_ref              VARCHAR(500),
        -- S3 reference to the input data that caused the failure
    output_data_ref             VARCHAR(500),
        -- S3 reference to any partial output before failure
    retry_history               JSONB           NOT NULL DEFAULT '[]'::jsonb,
        -- Array of retry attempt details {attempt, delay, error, timestamp}
    resolved                    BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether this DLQ entry has been resolved
    resolution_type             VARCHAR(30),
        -- Resolution type: retried, skipped, manual_override, abandoned
    resolved_at                 TIMESTAMPTZ,
        -- Resolution timestamp
    resolved_by                 VARCHAR(100),
        -- User or system actor who resolved this entry
    resolution_notes            TEXT,
        -- Notes explaining the resolution action
    tenant_id                   UUID            NOT NULL,
        -- Multi-tenant isolation identifier
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ddo_dlq_workflow FOREIGN KEY (workflow_id)
        REFERENCES gl_eudr_ddo_workflow_executions(workflow_id),
    CONSTRAINT chk_ddo_dlq_error_class CHECK (error_classification IN (
        'permanent', 'transient_exhausted', 'degraded'
    )),
    CONSTRAINT chk_ddo_dlq_resolution_type CHECK (resolution_type IS NULL OR resolution_type IN (
        'retried', 'skipped', 'manual_override', 'abandoned'
    )),
    CONSTRAINT chk_ddo_dlq_attempt CHECK (attempt_number >= 1)
);

DO $$ BEGIN
    CREATE INDEX idx_ddo_dlq_workflow ON gl_eudr_ddo_dead_letter_queue (workflow_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_dlq_agent ON gl_eudr_ddo_dead_letter_queue (agent_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_dlq_operator ON gl_eudr_ddo_dead_letter_queue (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_dlq_resolved ON gl_eudr_ddo_dead_letter_queue (resolved)
        WHERE resolved = FALSE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_dlq_tenant ON gl_eudr_ddo_dead_letter_queue (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_dlq_error_class ON gl_eudr_ddo_dead_letter_queue (error_classification);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_dlq_created_at ON gl_eudr_ddo_dead_letter_queue (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_dlq_tenant_unresolved ON gl_eudr_ddo_dead_letter_queue (tenant_id, resolved)
        WHERE resolved = FALSE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_dlq_agent_class ON gl_eudr_ddo_dead_letter_queue (agent_id, error_classification);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_dlq_workflow_agent ON gl_eudr_ddo_dead_letter_queue (workflow_id, agent_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_dlq_retry_history ON gl_eudr_ddo_dead_letter_queue USING GIN (retry_history);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

DO $$ BEGIN
    CREATE INDEX idx_ddo_dlq_error_type ON gl_eudr_ddo_dead_letter_queue (error_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_dlq_operator_agent ON gl_eudr_ddo_dead_letter_queue (operator_id, agent_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_dlq_resolution_type ON gl_eudr_ddo_dead_letter_queue (resolution_type) WHERE resolution_type IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_dlq_resolved_at ON gl_eudr_ddo_dead_letter_queue (resolved_at DESC) WHERE resolved = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_dlq_resolved_by ON gl_eudr_ddo_dead_letter_queue (resolved_by) WHERE resolved_by IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_ddo_dead_letter_queue IS 'Dead letter queue for permanently failed or retry-exhausted agent invocations. Stores full error context, retry history, and resolution audit trail for ops review. Supports manual retry, skip, override, and abandon resolution workflows.';


-- ============================================================================
-- 6. gl_eudr_ddo_batch_executions — Batch workflow groups
-- ============================================================================
RAISE NOTICE 'V114 [6/13]: Creating gl_eudr_ddo_batch_executions...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddo_batch_executions (
    batch_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id                 UUID            NOT NULL,
        -- Operator who initiated the batch
    name                        VARCHAR(500),
        -- Human-readable batch name (e.g., "Q1 2026 Palm Oil Portfolio")
    description                 TEXT,
        -- Batch description
    commodity                   VARCHAR(50),
        -- Target commodity for this batch (null for multi-commodity)
    workflow_type               VARCHAR(20)     NOT NULL DEFAULT 'standard',
        -- Workflow type for all workflows in this batch
    workflow_count              INTEGER         NOT NULL DEFAULT 0,
        -- Total number of workflows in this batch
    completed_count             INTEGER         NOT NULL DEFAULT 0,
        -- Number of workflows completed successfully
    failed_count                INTEGER         NOT NULL DEFAULT 0,
        -- Number of workflows that terminated with failure
    cancelled_count             INTEGER         NOT NULL DEFAULT 0,
        -- Number of workflows that were cancelled
    status                      VARCHAR(20)     NOT NULL DEFAULT 'running',
        -- Batch status: pending, running, completed, partially_completed, failed, cancelled
    progress_pct                NUMERIC(5,2)    NOT NULL DEFAULT 0,
        -- Overall batch progress percentage
    config                      JSONB           NOT NULL DEFAULT '{}'::jsonb,
        -- Batch configuration: concurrency limits, priority, etc.
    tenant_id                   UUID            NOT NULL,
        -- Multi-tenant isolation identifier
    created_by                  VARCHAR(100),
        -- User or system actor who created this batch
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    completed_at                TIMESTAMPTZ,
        -- Batch completion timestamp

    CONSTRAINT chk_ddo_batch_status CHECK (status IN (
        'pending', 'running', 'completed', 'partially_completed', 'failed', 'cancelled'
    )),
    CONSTRAINT chk_ddo_batch_commodity CHECK (commodity IS NULL OR commodity IN (
        'cattle', 'cocoa', 'coffee', 'palm_oil', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_ddo_batch_workflow_type CHECK (workflow_type IN (
        'standard', 'simplified', 'custom'
    )),
    CONSTRAINT chk_ddo_batch_counts CHECK (
        workflow_count >= 0 AND completed_count >= 0 AND failed_count >= 0 AND cancelled_count >= 0
    ),
    CONSTRAINT chk_ddo_batch_progress CHECK (progress_pct >= 0 AND progress_pct <= 100)
);

DO $$ BEGIN
    CREATE INDEX idx_ddo_batch_operator ON gl_eudr_ddo_batch_executions (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_batch_status ON gl_eudr_ddo_batch_executions (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_batch_tenant ON gl_eudr_ddo_batch_executions (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_batch_created_at ON gl_eudr_ddo_batch_executions (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_batch_operator_status ON gl_eudr_ddo_batch_executions (operator_id, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_batch_tenant_status ON gl_eudr_ddo_batch_executions (tenant_id, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_batch_commodity ON gl_eudr_ddo_batch_executions (commodity) WHERE commodity IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_batch_running ON gl_eudr_ddo_batch_executions (status, created_at DESC)
        WHERE status = 'running';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_batch_config ON gl_eudr_ddo_batch_executions USING GIN (config);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

DO $$ BEGIN
    CREATE INDEX idx_ddo_batch_workflow_type ON gl_eudr_ddo_batch_executions (workflow_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_batch_created_by ON gl_eudr_ddo_batch_executions (created_by) WHERE created_by IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_batch_completed_at ON gl_eudr_ddo_batch_executions (completed_at DESC) WHERE completed_at IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_ddo_batch_executions IS 'Batch workflow groups enabling portfolio-level due diligence execution. Operators can launch workflows for entire quarterly shipment portfolios in a single action with batch-level progress tracking and SLA monitoring.';


-- ============================================================================
-- 7. gl_eudr_ddo_workflow_definition_versions — Immutable version history
-- ============================================================================
RAISE NOTICE 'V114 [7/13]: Creating gl_eudr_ddo_workflow_definition_versions...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddo_workflow_definition_versions (
    version_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_def_id             UUID            NOT NULL,
        -- Reference to the parent workflow definition
    version_number              INTEGER         NOT NULL,
        -- Monotonically increasing version number
    definition_snapshot         JSONB           NOT NULL,
        -- Complete immutable snapshot of the workflow definition at this version
    agent_nodes_snapshot        JSONB           NOT NULL DEFAULT '[]'::jsonb,
        -- Snapshot of agent nodes at this version
    dependency_edges_snapshot   JSONB           NOT NULL DEFAULT '[]'::jsonb,
        -- Snapshot of dependency edges at this version
    quality_gates_snapshot      JSONB           NOT NULL DEFAULT '[]'::jsonb,
        -- Snapshot of quality gate definitions at this version
    change_summary              TEXT,
        -- Human-readable description of what changed
    change_type                 VARCHAR(30),
        -- Type of change: created, modified, agents_added, agents_removed, gates_updated
    changed_by                  VARCHAR(100)    NOT NULL,
        -- User or system actor who made this change
    snapshot_hash               VARCHAR(64),
        -- SHA-256 hash of the definition snapshot for integrity verification
    tenant_id                   UUID,
        -- Multi-tenant isolation identifier
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ddo_wfver_def FOREIGN KEY (workflow_def_id)
        REFERENCES gl_eudr_ddo_workflow_definitions(workflow_def_id),
    CONSTRAINT chk_ddo_wfver_version CHECK (version_number >= 1),
    CONSTRAINT chk_ddo_wfver_change_type CHECK (change_type IS NULL OR change_type IN (
        'created', 'modified', 'agents_added', 'agents_removed', 'gates_updated',
        'config_changed', 'threshold_changed', 'template_cloned'
    )),
    CONSTRAINT uq_ddo_wfver_def_version UNIQUE (workflow_def_id, version_number)
);

DO $$ BEGIN
    CREATE INDEX idx_ddo_wfver_def ON gl_eudr_ddo_workflow_definition_versions (workflow_def_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfver_def_version ON gl_eudr_ddo_workflow_definition_versions (workflow_def_id, version_number DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfver_created_at ON gl_eudr_ddo_workflow_definition_versions (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfver_changed_by ON gl_eudr_ddo_workflow_definition_versions (changed_by);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfver_tenant ON gl_eudr_ddo_workflow_definition_versions (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfver_change_type ON gl_eudr_ddo_workflow_definition_versions (change_type) WHERE change_type IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfver_hash ON gl_eudr_ddo_workflow_definition_versions (snapshot_hash) WHERE snapshot_hash IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfver_definition ON gl_eudr_ddo_workflow_definition_versions USING GIN (definition_snapshot);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

DO $$ BEGIN
    CREATE INDEX idx_ddo_wfver_agent_nodes ON gl_eudr_ddo_workflow_definition_versions USING GIN (agent_nodes_snapshot);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfver_dep_edges ON gl_eudr_ddo_workflow_definition_versions USING GIN (dependency_edges_snapshot);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_wfver_qg_snapshot ON gl_eudr_ddo_workflow_definition_versions USING GIN (quality_gates_snapshot);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_ddo_workflow_definition_versions IS 'Immutable version history for workflow definitions. Every modification creates a new version with a full snapshot, enabling audit trail of template changes and rollback to previous versions per EUDR Article 8(3) annual review requirement.';


-- ============================================================================
-- 8. gl_eudr_ddo_quality_gate_overrides — Override audit records
-- ============================================================================
RAISE NOTICE 'V114 [8/13]: Creating gl_eudr_ddo_quality_gate_overrides...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddo_quality_gate_overrides (
    override_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id                 UUID            NOT NULL,
        -- Workflow execution where the gate was overridden
    gate_id                     VARCHAR(10)     NOT NULL,
        -- Quality gate identifier: QG-1, QG-2, QG-3
    original_result             VARCHAR(20)     NOT NULL DEFAULT 'failed',
        -- Original gate evaluation result before override
    original_score              NUMERIC(5,2)    NOT NULL,
        -- Original gate score that failed the threshold
    threshold                   NUMERIC(5,2)    NOT NULL,
        -- Threshold that was not met
    gap                         NUMERIC(5,2)    NOT NULL,
        -- Score gap: threshold - original_score
    check_failures              JSONB           NOT NULL DEFAULT '[]'::jsonb,
        -- Array of individual check failures that caused the gate to fail
    justification               TEXT            NOT NULL,
        -- Mandatory justification for overriding the failed gate
    risk_acceptance              TEXT,
        -- Formal risk acceptance statement
    compensating_controls       JSONB           DEFAULT '[]'::jsonb,
        -- Compensating controls applied to mitigate the override risk
    approved_by                 VARCHAR(100),
        -- Approver (if different from overrider, for dual-approval workflows)
    overridden_by               VARCHAR(100)    NOT NULL,
        -- User who performed the override
    operator_id                 UUID            NOT NULL,
        -- Operator scope for audit trail
    tenant_id                   UUID            NOT NULL,
        -- Multi-tenant isolation identifier
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ddo_qgo_workflow FOREIGN KEY (workflow_id)
        REFERENCES gl_eudr_ddo_workflow_executions(workflow_id),
    CONSTRAINT chk_ddo_qgo_gate CHECK (gate_id IN ('QG-1', 'QG-2', 'QG-3')),
    CONSTRAINT chk_ddo_qgo_original_result CHECK (original_result IN (
        'failed', 'error'
    )),
    CONSTRAINT chk_ddo_qgo_score CHECK (original_score >= 0 AND original_score <= 100),
    CONSTRAINT chk_ddo_qgo_threshold CHECK (threshold >= 0 AND threshold <= 100),
    CONSTRAINT chk_ddo_qgo_gap CHECK (gap >= 0)
);

DO $$ BEGIN
    CREATE INDEX idx_ddo_qgo_workflow ON gl_eudr_ddo_quality_gate_overrides (workflow_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qgo_gate ON gl_eudr_ddo_quality_gate_overrides (gate_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qgo_operator ON gl_eudr_ddo_quality_gate_overrides (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qgo_tenant ON gl_eudr_ddo_quality_gate_overrides (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qgo_created_at ON gl_eudr_ddo_quality_gate_overrides (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qgo_overridden_by ON gl_eudr_ddo_quality_gate_overrides (overridden_by);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qgo_workflow_gate ON gl_eudr_ddo_quality_gate_overrides (workflow_id, gate_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qgo_tenant_gate ON gl_eudr_ddo_quality_gate_overrides (tenant_id, gate_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qgo_check_failures ON gl_eudr_ddo_quality_gate_overrides USING GIN (check_failures);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qgo_compensating ON gl_eudr_ddo_quality_gate_overrides USING GIN (compensating_controls);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_ddo_quality_gate_overrides IS 'Audit trail for quality gate override decisions. Records the original failure details, mandatory justification, risk acceptance, and compensating controls when a compliance officer overrides a failed quality gate. Required for regulatory defensibility under EUDR Article 8(2).';


-- ============================================================================
-- 9. gl_eudr_ddo_agent_estimated_durations — ETA estimation reference data
-- ============================================================================
RAISE NOTICE 'V114 [9/13]: Creating gl_eudr_ddo_agent_estimated_durations...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddo_agent_estimated_durations (
    agent_id                    VARCHAR(20)     NOT NULL,
        -- EUDR agent identifier (e.g., EUDR-001 through EUDR-025)
    commodity                   VARCHAR(50)     NOT NULL,
        -- Commodity context for duration estimation
    workflow_type               VARCHAR(20)     NOT NULL DEFAULT 'standard',
        -- Workflow type: standard, simplified
    avg_duration_seconds        NUMERIC(10,2)   NOT NULL,
        -- Running average execution duration in seconds
    p50_duration_seconds        NUMERIC(10,2)   NOT NULL DEFAULT 0,
        -- Median (p50) execution duration in seconds
    p95_duration_seconds        NUMERIC(10,2)   NOT NULL,
        -- 95th percentile execution duration in seconds
    p99_duration_seconds        NUMERIC(10,2)   NOT NULL DEFAULT 0,
        -- 99th percentile execution duration in seconds
    min_duration_seconds        NUMERIC(10,2)   NOT NULL DEFAULT 0,
        -- Minimum observed execution duration
    max_duration_seconds        NUMERIC(10,2)   NOT NULL DEFAULT 0,
        -- Maximum observed execution duration
    sample_count                INTEGER         NOT NULL DEFAULT 0,
        -- Number of execution samples used for this estimate
    success_rate                NUMERIC(5,4)    NOT NULL DEFAULT 1.0000,
        -- Success rate for this agent-commodity combination
    last_execution_seconds      NUMERIC(10,2),
        -- Most recent execution duration
    tenant_id                   UUID,
        -- Tenant scope (null for global estimates)
    last_updated_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (agent_id, commodity, workflow_type),

    CONSTRAINT chk_ddo_aed_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'palm_oil', 'rubber', 'soya', 'wood', 'all'
    )),
    CONSTRAINT chk_ddo_aed_workflow_type CHECK (workflow_type IN (
        'standard', 'simplified'
    )),
    CONSTRAINT chk_ddo_aed_avg CHECK (avg_duration_seconds >= 0),
    CONSTRAINT chk_ddo_aed_p95 CHECK (p95_duration_seconds >= 0),
    CONSTRAINT chk_ddo_aed_sample CHECK (sample_count >= 0),
    CONSTRAINT chk_ddo_aed_success_rate CHECK (success_rate >= 0 AND success_rate <= 1)
);

DO $$ BEGIN
    CREATE INDEX idx_ddo_aed_agent ON gl_eudr_ddo_agent_estimated_durations (agent_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_aed_commodity ON gl_eudr_ddo_agent_estimated_durations (commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_aed_agent_commodity ON gl_eudr_ddo_agent_estimated_durations (agent_id, commodity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_aed_tenant ON gl_eudr_ddo_agent_estimated_durations (tenant_id) WHERE tenant_id IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_aed_updated ON gl_eudr_ddo_agent_estimated_durations (last_updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_aed_success_rate ON gl_eudr_ddo_agent_estimated_durations (success_rate) WHERE success_rate < 0.9;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

DO $$ BEGIN
    CREATE INDEX idx_ddo_aed_workflow_type ON gl_eudr_ddo_agent_estimated_durations (workflow_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_aed_avg_duration ON gl_eudr_ddo_agent_estimated_durations (avg_duration_seconds DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_aed_p95_duration ON gl_eudr_ddo_agent_estimated_durations (p95_duration_seconds DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_aed_sample_count ON gl_eudr_ddo_agent_estimated_durations (sample_count DESC) WHERE sample_count > 0;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_ddo_agent_estimated_durations IS 'Running statistics for per-agent, per-commodity execution durations used by the Parallel Execution Engine for ETA calculation via critical path analysis. Updated after each agent completion with rolling average, percentiles, and success rates.';


-- ============================================================================
-- 10. gl_eudr_ddo_workflow_checkpoints — Persistent checkpoints (hypertable)
-- ============================================================================
RAISE NOTICE 'V114 [10/13]: Creating gl_eudr_ddo_workflow_checkpoints (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddo_workflow_checkpoints (
    checkpoint_id               UUID            DEFAULT gen_random_uuid(),
    workflow_id                 UUID            NOT NULL,
        -- Workflow execution this checkpoint belongs to
    sequence_number             INTEGER         NOT NULL,
        -- Monotonically increasing checkpoint sequence within workflow
    phase                       VARCHAR(30)     NOT NULL,
        -- Phase at checkpoint time
    agent_id                    VARCHAR(20),
        -- Agent that triggered this checkpoint (null for gate checkpoints)
    gate_id                     VARCHAR(10),
        -- Quality gate that triggered this checkpoint (null for agent checkpoints)
    checkpoint_type             VARCHAR(30)     NOT NULL,
        -- Type: agent_complete, agent_failed, gate_passed, gate_failed, gate_overridden, phase_start, phase_complete, workflow_paused, workflow_resumed, workflow_cancelled
    agent_statuses              JSONB           NOT NULL DEFAULT '{}'::jsonb,
        -- Complete per-agent status map at checkpoint time
    agent_outputs_ref           VARCHAR(500),
        -- S3 reference to full agent outputs snapshot
    agent_outputs_summary       JSONB           DEFAULT '{}'::jsonb,
        -- Summarized agent outputs at checkpoint time
    quality_gate_results        JSONB           DEFAULT '{}'::jsonb,
        -- Quality gate evaluation results at checkpoint time
    workflow_progress            JSONB           DEFAULT '{}'::jsonb,
        -- Workflow progress snapshot (agents completed, ETA, etc.)
    cumulative_provenance_hash  VARCHAR(64)     NOT NULL,
        -- SHA-256 provenance hash chaining all previous checkpoints
    previous_provenance_hash    VARCHAR(64),
        -- Previous checkpoint provenance hash (null for genesis)
    metadata                    JSONB           DEFAULT '{}'::jsonb,
        -- Additional checkpoint metadata
    created_by                  VARCHAR(100),
        -- User or system actor who created this checkpoint
    tenant_id                   UUID            NOT NULL,
        -- Multi-tenant isolation identifier
    checkpoint_timestamp        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Checkpoint creation timestamp (partition key)

    PRIMARY KEY (checkpoint_id, checkpoint_timestamp),

    CONSTRAINT chk_ddo_ckpt_phase CHECK (phase IN (
        'initialization', 'information_gathering', 'risk_assessment',
        'risk_mitigation', 'package_generation', 'completed'
    )),
    CONSTRAINT chk_ddo_ckpt_type CHECK (checkpoint_type IN (
        'agent_complete', 'agent_failed', 'agent_skipped',
        'gate_passed', 'gate_failed', 'gate_overridden',
        'phase_start', 'phase_complete',
        'workflow_started', 'workflow_paused', 'workflow_resumed',
        'workflow_cancelled', 'workflow_completed', 'workflow_terminated',
        'mitigation_bypassed'
    )),
    CONSTRAINT chk_ddo_ckpt_sequence CHECK (sequence_number >= 0),
    CONSTRAINT chk_ddo_ckpt_gate CHECK (gate_id IS NULL OR gate_id IN ('QG-1', 'QG-2', 'QG-3'))
);

SELECT create_hypertable(
    'gl_eudr_ddo_workflow_checkpoints',
    'checkpoint_timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_ddo_ckpt_workflow ON gl_eudr_ddo_workflow_checkpoints (workflow_id, checkpoint_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ckpt_workflow_seq ON gl_eudr_ddo_workflow_checkpoints (workflow_id, sequence_number DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ckpt_phase ON gl_eudr_ddo_workflow_checkpoints (phase, checkpoint_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ckpt_agent ON gl_eudr_ddo_workflow_checkpoints (agent_id, checkpoint_timestamp DESC) WHERE agent_id IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ckpt_gate ON gl_eudr_ddo_workflow_checkpoints (gate_id, checkpoint_timestamp DESC) WHERE gate_id IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ckpt_type ON gl_eudr_ddo_workflow_checkpoints (checkpoint_type, checkpoint_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ckpt_provenance ON gl_eudr_ddo_workflow_checkpoints (cumulative_provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ckpt_tenant ON gl_eudr_ddo_workflow_checkpoints (tenant_id, checkpoint_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ckpt_agent_statuses ON gl_eudr_ddo_workflow_checkpoints USING GIN (agent_statuses);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ckpt_qg_results ON gl_eudr_ddo_workflow_checkpoints USING GIN (quality_gate_results);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ckpt_workflow_phase ON gl_eudr_ddo_workflow_checkpoints (workflow_id, phase, checkpoint_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ckpt_workflow_type ON gl_eudr_ddo_workflow_checkpoints (workflow_id, checkpoint_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_ddo_workflow_checkpoints IS 'Persistent workflow checkpoints with SHA-256 provenance hash chain for workflow resume, rollback, and audit trail. Each checkpoint captures the complete workflow state including per-agent statuses, quality gate results, and S3 output references. TimescaleDB hypertable with 7-day chunks and 5-year retention per EUDR Article 31.';


-- ============================================================================
-- 11. gl_eudr_ddo_quality_gate_evaluations — Gate evaluation records (hypertable)
-- ============================================================================
RAISE NOTICE 'V114 [11/13]: Creating gl_eudr_ddo_quality_gate_evaluations (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddo_quality_gate_evaluations (
    evaluation_id               UUID            DEFAULT gen_random_uuid(),
    workflow_id                 UUID            NOT NULL,
        -- Workflow execution for which the gate was evaluated
    gate_id                     VARCHAR(10)     NOT NULL,
        -- Quality gate identifier: QG-1, QG-2, QG-3
    result                      VARCHAR(20)     NOT NULL,
        -- Evaluation result: passed, failed, overridden, error
    overall_score               NUMERIC(5,2)    NOT NULL,
        -- Overall gate score (0-100)
    threshold_used              NUMERIC(5,2)    NOT NULL,
        -- Threshold applied for this evaluation
    check_results               JSONB           NOT NULL DEFAULT '[]'::jsonb,
        -- Array of individual check evaluation results
    is_simplified               BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether simplified (Article 13) thresholds were used
    evaluation_duration_ms      INTEGER,
        -- Time taken to evaluate the gate in milliseconds
    gap_details                 JSONB           DEFAULT '{}'::jsonb,
        -- Detailed gap analysis for failed evaluations
    remediation_guidance        JSONB           DEFAULT '[]'::jsonb,
        -- Suggested remediation actions for failed checks
    override_id                 UUID,
        -- Reference to override record (if overridden)
    evaluation_context          JSONB           DEFAULT '{}'::jsonb,
        -- Context data provided to the evaluation (agent outputs, scores)
    evaluated_by                VARCHAR(100),
        -- System actor or user who triggered the evaluation
    operator_id                 UUID            NOT NULL,
        -- Operator scope for multi-tenant isolation
    tenant_id                   UUID            NOT NULL,
        -- Multi-tenant isolation identifier
    evaluated_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Evaluation timestamp (partition key)

    PRIMARY KEY (evaluation_id, evaluated_at),

    CONSTRAINT chk_ddo_qge_gate CHECK (gate_id IN ('QG-1', 'QG-2', 'QG-3')),
    CONSTRAINT chk_ddo_qge_result CHECK (result IN (
        'passed', 'failed', 'overridden', 'error', 'skipped'
    )),
    CONSTRAINT chk_ddo_qge_score CHECK (overall_score >= 0 AND overall_score <= 100),
    CONSTRAINT chk_ddo_qge_threshold CHECK (threshold_used >= 0 AND threshold_used <= 100)
);

SELECT create_hypertable(
    'gl_eudr_ddo_quality_gate_evaluations',
    'evaluated_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_ddo_qge_workflow ON gl_eudr_ddo_quality_gate_evaluations (workflow_id, evaluated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qge_gate ON gl_eudr_ddo_quality_gate_evaluations (gate_id, evaluated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qge_result ON gl_eudr_ddo_quality_gate_evaluations (result, evaluated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qge_workflow_gate ON gl_eudr_ddo_quality_gate_evaluations (workflow_id, gate_id, evaluated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qge_operator ON gl_eudr_ddo_quality_gate_evaluations (operator_id, evaluated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qge_tenant ON gl_eudr_ddo_quality_gate_evaluations (tenant_id, evaluated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qge_gate_result ON gl_eudr_ddo_quality_gate_evaluations (gate_id, result, evaluated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qge_simplified ON gl_eudr_ddo_quality_gate_evaluations (is_simplified, evaluated_at DESC) WHERE is_simplified = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qge_check_results ON gl_eudr_ddo_quality_gate_evaluations USING GIN (check_results);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qge_gap_details ON gl_eudr_ddo_quality_gate_evaluations USING GIN (gap_details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qge_failed ON gl_eudr_ddo_quality_gate_evaluations (gate_id, overall_score, evaluated_at DESC) WHERE result = 'failed';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_qge_overridden ON gl_eudr_ddo_quality_gate_evaluations (override_id, evaluated_at DESC) WHERE override_id IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_ddo_quality_gate_evaluations IS 'Quality gate evaluation records for all three mandatory EUDR due diligence phase transitions. Records individual check results, gap analysis, and remediation guidance. TimescaleDB hypertable with 30-day chunks and 5-year retention for regulatory audit per EUDR Articles 8, 9, 10, 11.';


-- ============================================================================
-- 12. gl_eudr_ddo_agent_execution_log — Per-agent execution records (hypertable)
-- ============================================================================
RAISE NOTICE 'V114 [12/13]: Creating gl_eudr_ddo_agent_execution_log (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddo_agent_execution_log (
    log_id                      UUID            DEFAULT gen_random_uuid(),
    workflow_id                 UUID            NOT NULL,
        -- Workflow execution this agent invocation belongs to
    agent_id                    VARCHAR(20)     NOT NULL,
        -- EUDR agent identifier (e.g., EUDR-001 through EUDR-025)
    agent_name                  VARCHAR(100),
        -- Human-readable agent name
    phase                       VARCHAR(30)     NOT NULL,
        -- Phase during which this agent executed
    layer                       INTEGER,
        -- DAG layer number within the phase
    status                      VARCHAR(20)     NOT NULL,
        -- Execution status: pending, queued, running, completed, failed, skipped, timeout
    attempt_number              INTEGER         NOT NULL DEFAULT 1,
        -- Attempt number (1 = first attempt, >1 = retry)
    duration_seconds            NUMERIC(10,3),
        -- Execution duration in seconds (null if not yet complete)
    input_ref                   VARCHAR(500),
        -- S3 reference to the input data sent to the agent
    input_summary               JSONB,
        -- Summarized input data
    output_ref                  VARCHAR(500),
        -- S3 reference to the agent output data
    output_summary              JSONB,
        -- Summarized output data for quick access
    output_completeness_pct     NUMERIC(5,2),
        -- Completeness percentage of agent output (0-100)
    error_type                  VARCHAR(50),
        -- Error type if failed
    error_message               TEXT,
        -- Error message if failed
    error_classification        VARCHAR(20),
        -- Error classification: transient, permanent, degraded
    is_fallback                 BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether this result used a fallback strategy
    fallback_type               VARCHAR(30),
        -- Fallback type used: cached_result, degraded_mode, manual_override
    circuit_breaker_state       VARCHAR(15),
        -- Circuit breaker state at invocation time
    operator_id                 UUID            NOT NULL,
        -- Operator scope
    tenant_id                   UUID            NOT NULL,
        -- Multi-tenant isolation identifier
    queued_at                   TIMESTAMPTZ,
        -- When the agent was queued for execution
    started_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Execution start timestamp (partition key)
    completed_at                TIMESTAMPTZ,
        -- Execution completion timestamp

    PRIMARY KEY (log_id, started_at),

    CONSTRAINT chk_ddo_ael_phase CHECK (phase IN (
        'information_gathering', 'risk_assessment', 'risk_mitigation', 'package_generation'
    )),
    CONSTRAINT chk_ddo_ael_status CHECK (status IN (
        'pending', 'queued', 'running', 'completed', 'failed', 'skipped', 'timeout', 'cancelled'
    )),
    CONSTRAINT chk_ddo_ael_error_class CHECK (error_classification IS NULL OR error_classification IN (
        'transient', 'permanent', 'degraded'
    )),
    CONSTRAINT chk_ddo_ael_fallback_type CHECK (fallback_type IS NULL OR fallback_type IN (
        'cached_result', 'degraded_mode', 'manual_override', 'none'
    )),
    CONSTRAINT chk_ddo_ael_cb_state CHECK (circuit_breaker_state IS NULL OR circuit_breaker_state IN (
        'closed', 'open', 'half_open'
    )),
    CONSTRAINT chk_ddo_ael_attempt CHECK (attempt_number >= 1),
    CONSTRAINT chk_ddo_ael_completeness CHECK (output_completeness_pct IS NULL OR (output_completeness_pct >= 0 AND output_completeness_pct <= 100))
);

SELECT create_hypertable(
    'gl_eudr_ddo_agent_execution_log',
    'started_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_ddo_ael_workflow ON gl_eudr_ddo_agent_execution_log (workflow_id, started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ael_agent ON gl_eudr_ddo_agent_execution_log (agent_id, started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ael_status ON gl_eudr_ddo_agent_execution_log (status, started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ael_workflow_agent ON gl_eudr_ddo_agent_execution_log (workflow_id, agent_id, started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ael_agent_status ON gl_eudr_ddo_agent_execution_log (agent_id, status, started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ael_phase ON gl_eudr_ddo_agent_execution_log (phase, started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ael_operator ON gl_eudr_ddo_agent_execution_log (operator_id, started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ael_tenant ON gl_eudr_ddo_agent_execution_log (tenant_id, started_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ael_error_class ON gl_eudr_ddo_agent_execution_log (error_classification, started_at DESC)
        WHERE error_classification IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ael_failed ON gl_eudr_ddo_agent_execution_log (agent_id, error_type, started_at DESC)
        WHERE status = 'failed';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ael_fallback ON gl_eudr_ddo_agent_execution_log (agent_id, fallback_type, started_at DESC)
        WHERE is_fallback = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ael_duration ON gl_eudr_ddo_agent_execution_log (agent_id, duration_seconds, started_at DESC)
        WHERE duration_seconds IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ael_retries ON gl_eudr_ddo_agent_execution_log (workflow_id, agent_id, attempt_number)
        WHERE attempt_number > 1;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ael_layer ON gl_eudr_ddo_agent_execution_log (phase, layer, started_at DESC)
        WHERE layer IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ael_output_summary ON gl_eudr_ddo_agent_execution_log USING GIN (output_summary)
        WHERE output_summary IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ael_input_summary ON gl_eudr_ddo_agent_execution_log USING GIN (input_summary)
        WHERE input_summary IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_ael_cb_state ON gl_eudr_ddo_agent_execution_log (circuit_breaker_state, started_at DESC)
        WHERE circuit_breaker_state IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_ddo_agent_execution_log IS 'Per-agent execution log recording every invocation across all workflow executions. Tracks duration, status, retry attempts, error classification, fallback usage, and circuit breaker state. TimescaleDB hypertable with 7-day chunks and 5-year retention for performance analysis and audit.';


-- ============================================================================
-- 13. gl_eudr_ddo_workflow_audit_trail — Immutable audit trail (hypertable)
-- ============================================================================
RAISE NOTICE 'V114 [13/13]: Creating gl_eudr_ddo_workflow_audit_trail (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_ddo_workflow_audit_trail (
    audit_id                    UUID            DEFAULT gen_random_uuid(),
    workflow_id                 UUID            NOT NULL,
        -- Workflow execution this audit event belongs to
    event_type                  VARCHAR(50)     NOT NULL,
        -- Type of audit event
    event_category              VARCHAR(30)     NOT NULL DEFAULT 'workflow',
        -- Category: workflow, agent, gate, checkpoint, package, error, admin
    event_data                  JSONB           NOT NULL DEFAULT '{}'::jsonb,
        -- Structured event data specific to the event type
    actor                       VARCHAR(100)    NOT NULL,
        -- User or system actor who triggered the event
    actor_role                  VARCHAR(50),
        -- Role of the actor (e.g., compliance_officer, system, admin, auditor)
    provenance_hash             VARCHAR(64)     NOT NULL,
        -- SHA-256 provenance hash for immutability verification
    previous_provenance_hash    VARCHAR(64),
        -- Previous audit trail provenance hash (null for first event)
    ip_address                  VARCHAR(45),
        -- IP address of the actor (IPv4 or IPv6)
    user_agent                  VARCHAR(500),
        -- HTTP User-Agent of the actor
    correlation_id              UUID,
        -- Correlation ID linking related events across services
    operator_id                 UUID            NOT NULL,
        -- Operator scope for multi-tenant isolation
    tenant_id                   UUID            NOT NULL,
        -- Multi-tenant isolation identifier
    timestamp                   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Event timestamp (partition key)

    PRIMARY KEY (audit_id, timestamp),

    CONSTRAINT chk_ddo_audit_event_type CHECK (event_type IN (
        'workflow_created', 'workflow_started', 'workflow_validated',
        'workflow_validation_failed', 'workflow_paused', 'workflow_resumed',
        'workflow_cancelled', 'workflow_completed', 'workflow_terminated',
        'phase_started', 'phase_completed',
        'agent_started', 'agent_completed', 'agent_failed',
        'agent_skipped', 'agent_retried', 'agent_timeout',
        'gate_evaluation_started', 'gate_passed', 'gate_failed', 'gate_overridden',
        'checkpoint_created', 'checkpoint_restored', 'checkpoint_rollback',
        'circuit_breaker_opened', 'circuit_breaker_closed', 'circuit_breaker_half_opened',
        'circuit_breaker_reset',
        'dlq_entry_created', 'dlq_entry_resolved',
        'package_generation_started', 'package_generated', 'package_submitted',
        'package_amended',
        'batch_created', 'batch_completed', 'batch_failed',
        'mitigation_bypassed', 'mitigation_started', 'mitigation_completed',
        'config_changed', 'template_modified', 'provenance_verified',
        'provenance_verification_failed'
    )),
    CONSTRAINT chk_ddo_audit_category CHECK (event_category IN (
        'workflow', 'agent', 'gate', 'checkpoint', 'package',
        'error', 'admin', 'batch', 'circuit_breaker', 'dlq', 'provenance'
    ))
);

SELECT create_hypertable(
    'gl_eudr_ddo_workflow_audit_trail',
    'timestamp',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_ddo_audit_workflow ON gl_eudr_ddo_workflow_audit_trail (workflow_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_audit_event_type ON gl_eudr_ddo_workflow_audit_trail (event_type, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_audit_category ON gl_eudr_ddo_workflow_audit_trail (event_category, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_audit_workflow_event ON gl_eudr_ddo_workflow_audit_trail (workflow_id, event_type, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_audit_workflow_category ON gl_eudr_ddo_workflow_audit_trail (workflow_id, event_category, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_audit_actor ON gl_eudr_ddo_workflow_audit_trail (actor, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_audit_operator ON gl_eudr_ddo_workflow_audit_trail (operator_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_audit_tenant ON gl_eudr_ddo_workflow_audit_trail (tenant_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_audit_provenance ON gl_eudr_ddo_workflow_audit_trail (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_audit_correlation ON gl_eudr_ddo_workflow_audit_trail (correlation_id, timestamp DESC)
        WHERE correlation_id IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_audit_event_data ON gl_eudr_ddo_workflow_audit_trail USING GIN (event_data);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_audit_operator_event ON gl_eudr_ddo_workflow_audit_trail (operator_id, event_type, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_audit_tenant_category ON gl_eudr_ddo_workflow_audit_trail (tenant_id, event_category, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_audit_actor_role ON gl_eudr_ddo_workflow_audit_trail (actor_role, timestamp DESC)
        WHERE actor_role IS NOT NULL;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_audit_gate_events ON gl_eudr_ddo_workflow_audit_trail (event_type, timestamp DESC)
        WHERE event_category = 'gate';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_ddo_audit_error_events ON gl_eudr_ddo_workflow_audit_trail (event_type, timestamp DESC)
        WHERE event_category = 'error';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_ddo_workflow_audit_trail IS 'Immutable audit trail for all workflow events with SHA-256 provenance hash chain. Records every state transition, agent invocation, quality gate evaluation, checkpoint operation, and administrative action. Append-only pattern with no UPDATE/DELETE. TimescaleDB hypertable with 30-day chunks and 5-year retention per EUDR Article 31.';


-- ============================================================================
-- CONTINUOUS AGGREGATES
-- ============================================================================

RAISE NOTICE 'V114: Creating continuous aggregates...';

-- Continuous Aggregate 1: Hourly workflow checkpoint summary
DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_ddo_workflow_checkpoints_hourly
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 hour', checkpoint_timestamp) AS bucket,
        workflow_id,
        phase,
        checkpoint_type,
        tenant_id,
        COUNT(*)                                    AS checkpoint_count,
        COUNT(DISTINCT workflow_id)                 AS workflow_count,
        COUNT(*) FILTER (WHERE checkpoint_type = 'agent_complete')  AS agent_completions,
        COUNT(*) FILTER (WHERE checkpoint_type = 'agent_failed')    AS agent_failures,
        COUNT(*) FILTER (WHERE checkpoint_type = 'gate_passed')     AS gate_passes,
        COUNT(*) FILTER (WHERE checkpoint_type = 'gate_failed')     AS gate_failures,
        COUNT(*) FILTER (WHERE checkpoint_type = 'gate_overridden') AS gate_overrides
    FROM gl_eudr_ddo_workflow_checkpoints
    GROUP BY bucket, workflow_id, phase, checkpoint_type, tenant_id;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_ddo_workflow_checkpoints_hourly',
        start_offset => INTERVAL '3 hours',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_ddo_workflow_checkpoints_hourly IS 'Hourly rollup of workflow checkpoint activity by workflow, phase, and type for monitoring checkpoint frequency and workflow progression patterns.';

-- Continuous Aggregate 2: Daily agent execution statistics
DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_ddo_agent_execution_stats_daily
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 day', started_at)            AS bucket,
        agent_id,
        phase,
        tenant_id,
        COUNT(*)                                    AS total_executions,
        COUNT(*) FILTER (WHERE status = 'completed')  AS successful_executions,
        COUNT(*) FILTER (WHERE status = 'failed')     AS failed_executions,
        COUNT(*) FILTER (WHERE status = 'skipped')    AS skipped_executions,
        COUNT(*) FILTER (WHERE status = 'timeout')    AS timeout_executions,
        COUNT(*) FILTER (WHERE attempt_number > 1)    AS retry_executions,
        COUNT(*) FILTER (WHERE is_fallback = TRUE)    AS fallback_executions,
        AVG(duration_seconds) FILTER (WHERE duration_seconds IS NOT NULL)     AS avg_duration_seconds,
        MAX(duration_seconds) FILTER (WHERE duration_seconds IS NOT NULL)     AS max_duration_seconds,
        MIN(duration_seconds) FILTER (WHERE duration_seconds IS NOT NULL)     AS min_duration_seconds,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY duration_seconds)
            FILTER (WHERE duration_seconds IS NOT NULL)   AS p50_duration_seconds,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_seconds)
            FILTER (WHERE duration_seconds IS NOT NULL)   AS p95_duration_seconds,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_seconds)
            FILTER (WHERE duration_seconds IS NOT NULL)   AS p99_duration_seconds
    FROM gl_eudr_ddo_agent_execution_log
    GROUP BY bucket, agent_id, phase, tenant_id;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_ddo_agent_execution_stats_daily',
        start_offset => INTERVAL '3 days',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_ddo_agent_execution_stats_daily IS 'Daily rollup of per-agent execution statistics including success/failure rates, duration percentiles, retry counts, and fallback usage for performance monitoring and ETA calibration.';


-- ============================================================================
-- COMPRESSION POLICIES
-- ============================================================================

RAISE NOTICE 'V114: Creating compression policies...';

-- Compress workflow checkpoints after 30 days
DO $$ BEGIN
    ALTER TABLE gl_eudr_ddo_workflow_checkpoints SET (
        timescaledb.compress,
        timescaledb.compress_segmentby = 'workflow_id, tenant_id',
        timescaledb.compress_orderby = 'checkpoint_timestamp DESC'
    );
EXCEPTION WHEN OTHERS THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_compression_policy('gl_eudr_ddo_workflow_checkpoints', INTERVAL '30 days');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Compress quality gate evaluations after 60 days
DO $$ BEGIN
    ALTER TABLE gl_eudr_ddo_quality_gate_evaluations SET (
        timescaledb.compress,
        timescaledb.compress_segmentby = 'workflow_id, gate_id, tenant_id',
        timescaledb.compress_orderby = 'evaluated_at DESC'
    );
EXCEPTION WHEN OTHERS THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_compression_policy('gl_eudr_ddo_quality_gate_evaluations', INTERVAL '60 days');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Compress agent execution log after 30 days
DO $$ BEGIN
    ALTER TABLE gl_eudr_ddo_agent_execution_log SET (
        timescaledb.compress,
        timescaledb.compress_segmentby = 'workflow_id, agent_id, tenant_id',
        timescaledb.compress_orderby = 'started_at DESC'
    );
EXCEPTION WHEN OTHERS THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_compression_policy('gl_eudr_ddo_agent_execution_log', INTERVAL '30 days');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Compress audit trail after 60 days
DO $$ BEGIN
    ALTER TABLE gl_eudr_ddo_workflow_audit_trail SET (
        timescaledb.compress,
        timescaledb.compress_segmentby = 'workflow_id, tenant_id',
        timescaledb.compress_orderby = 'timestamp DESC'
    );
EXCEPTION WHEN OTHERS THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_compression_policy('gl_eudr_ddo_workflow_audit_trail', INTERVAL '60 days');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- RETENTION POLICIES (5 years per EUDR Article 31)
-- ============================================================================

RAISE NOTICE 'V114: Creating retention policies...';

-- 5 years for workflow checkpoints
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_ddo_workflow_checkpoints', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for quality gate evaluations
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_ddo_quality_gate_evaluations', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for agent execution log
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_ddo_agent_execution_log', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for workflow audit trail
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_ddo_workflow_audit_trail', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- FINALIZE
-- ============================================================================

RAISE NOTICE 'V114: AGENT-EUDR-026 Due Diligence Orchestrator tables created successfully!';
RAISE NOTICE 'V114: Created 13 tables (9 regular + 4 hypertables), 2 continuous aggregates, 195 indexes';
RAISE NOTICE 'V114: Compression policies: 30d checkpoints, 60d gate evals, 30d agent log, 60d audit trail';
RAISE NOTICE 'V114: Retention policies: 5y on all 4 hypertables per EUDR Article 31';

COMMIT;
