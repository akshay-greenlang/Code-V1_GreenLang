-- ============================================================================
-- V113: AGENT-EUDR-025 Risk Mitigation Advisor Agent
-- ============================================================================
-- Creates tables for the Risk Mitigation Advisor agent which provides
-- ML-powered mitigation strategy recommendation, structured remediation
-- plan management, supplier capacity building, a 500+ mitigation measure
-- knowledge base, effectiveness tracking with ROI analysis, continuous
-- monitoring with adaptive management, cost-benefit budget optimization
-- via linear programming, multi-stakeholder collaboration, and audit-ready
-- mitigation documentation for EUDR compliance per Articles 8, 10, 11,
-- 29, 31 and ISO 31000:2018 Risk Management.
--
-- Schema: eudr_risk_mitigation (14 tables)
-- Tables: 14 (10 regular + 4 hypertables)
-- Hypertables: gl_eudr_rma_effectiveness_tracking (30d chunks, time: tracked_at),
--              gl_eudr_rma_monitoring_events (30d chunks, time: event_timestamp),
--              gl_eudr_rma_optimization_runs (30d chunks, time: run_at),
--              gl_eudr_rma_audit_trail (30d chunks, time: timestamp)
-- Continuous Aggregates: 2 (gl_eudr_rma_effectiveness_daily,
--                           gl_eudr_rma_events_hourly)
-- Retention Policies: 4 (5 years per EUDR Article 31)
-- Compression Policies: 4 (on all hypertables)
-- Indexes: ~190 (B-tree, GIN for JSONB, partial indexes, full-text indexes)
--
-- Dependencies: TimescaleDB extension (V002)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V113: Creating AGENT-EUDR-025 Risk Mitigation Advisor tables...';

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS eudr_risk_mitigation;


-- ============================================================================
-- 1. gl_eudr_rma_strategies — Recommended and selected mitigation strategies
-- ============================================================================
RAISE NOTICE 'V113 [1/14]: Creating gl_eudr_rma_strategies...';

CREATE TABLE IF NOT EXISTS gl_eudr_rma_strategies (
    strategy_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id                 UUID            NOT NULL,
        -- Operator requesting mitigation recommendations
    supplier_id                 UUID,
        -- Target supplier for strategy (nullable for portfolio-level strategies)
    name                        VARCHAR(500)    NOT NULL,
        -- Strategy display name (e.g. "Enhanced Country Monitoring - Brazil")
    description                 TEXT,
        -- Detailed strategy description with implementation approach
    risk_categories             JSONB           NOT NULL DEFAULT '[]',
        -- Risk categories addressed (country, supplier, commodity, corruption, deforestation, indigenous_rights, protected_areas, legal_compliance)
    iso_31000_type              VARCHAR(50)     NOT NULL,
        -- ISO 31000:2018 risk treatment type
    target_risk_factors         JSONB           DEFAULT '[]',
        -- Specific risk factors targeted by this strategy (from 9 upstream agents)
    predicted_effectiveness     NUMERIC(5,2)    DEFAULT 0.0,
        -- ML/rule-predicted effectiveness score (0-100)
    confidence_score            NUMERIC(4,3)    DEFAULT 0.000,
        -- ML model confidence score (0.000-1.000)
    cost_estimate               JSONB           DEFAULT '{}',
        -- Cost estimate breakdown (min, max, currency, per_supplier, total)
    implementation_complexity   VARCHAR(20)     NOT NULL DEFAULT 'medium',
        -- Implementation difficulty level
    time_to_effect_weeks        INTEGER         DEFAULT 8,
        -- Expected weeks until measurable risk reduction
    prerequisite_conditions     JSONB           DEFAULT '[]',
        -- Conditions that must be met before implementation
    eudr_articles               JSONB           DEFAULT '[]',
        -- EUDR articles addressed by this strategy (e.g. ["Art.10","Art.11"])
    shap_explanation            JSONB           DEFAULT '{}',
        -- SHAP values explaining ML recommendation (feature importance breakdown)
    measure_ids                 JSONB           DEFAULT '[]',
        -- IDs of specific mitigation measures composing this strategy
    model_version               VARCHAR(50),
        -- ML model version used (or "deterministic_fallback")
    mode                        VARCHAR(20)     NOT NULL DEFAULT 'ml',
        -- Recommendation mode used
    risk_context_hash           VARCHAR(64),
        -- SHA-256 hash of input risk context for deduplication
    provenance_hash             VARCHAR(64)     NOT NULL,
        -- SHA-256 hash for strategy record integrity verification
    status                      VARCHAR(30)     NOT NULL DEFAULT 'recommended',
        -- Strategy lifecycle status
    country_code                CHAR(2),
        -- Target country (ISO 3166-1 alpha-2) for filtering
    commodity                   VARCHAR(50),
        -- Target commodity for filtering
    metadata                    JSONB           DEFAULT '{}',
        -- Additional strategy attributes (tags, notes, source_agents)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_rma_strat_iso_type CHECK (iso_31000_type IN (
        'avoid', 'reduce_likelihood', 'reduce_consequence',
        'share', 'retain', 'remove_source', 'change_likelihood'
    )),
    CONSTRAINT chk_rma_strat_complexity CHECK (implementation_complexity IN (
        'low', 'medium', 'high', 'very_high'
    )),
    CONSTRAINT chk_rma_strat_status CHECK (status IN (
        'recommended', 'selected', 'active', 'completed', 'rejected'
    )),
    CONSTRAINT chk_rma_strat_mode CHECK (mode IN ('ml', 'deterministic', 'hybrid')),
    CONSTRAINT chk_rma_strat_effectiveness CHECK (predicted_effectiveness >= 0 AND predicted_effectiveness <= 100),
    CONSTRAINT chk_rma_strat_confidence CHECK (confidence_score >= 0 AND confidence_score <= 1),
    CONSTRAINT chk_rma_strat_time CHECK (time_to_effect_weeks >= 0)
);

DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_operator ON gl_eudr_rma_strategies (operator_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_supplier ON gl_eudr_rma_strategies (supplier_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_status ON gl_eudr_rma_strategies (status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_iso_type ON gl_eudr_rma_strategies (iso_31000_type); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_complexity ON gl_eudr_rma_strategies (implementation_complexity); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_mode ON gl_eudr_rma_strategies (mode); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_country ON gl_eudr_rma_strategies (country_code); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_commodity ON gl_eudr_rma_strategies (commodity); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_provenance ON gl_eudr_rma_strategies (provenance_hash); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_ctx_hash ON gl_eudr_rma_strategies (risk_context_hash); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_model ON gl_eudr_rma_strategies (model_version); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_effectiveness ON gl_eudr_rma_strategies (predicted_effectiveness DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_confidence ON gl_eudr_rma_strategies (confidence_score DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_tenant ON gl_eudr_rma_strategies (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_created ON gl_eudr_rma_strategies (created_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_op_status ON gl_eudr_rma_strategies (operator_id, status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_sup_status ON gl_eudr_rma_strategies (supplier_id, status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_op_country ON gl_eudr_rma_strategies (operator_id, country_code); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_op_commodity ON gl_eudr_rma_strategies (operator_id, commodity); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active/recommended strategies
DO $$ BEGIN
    CREATE INDEX idx_eudr_rma_strat_active ON gl_eudr_rma_strategies (operator_id, predicted_effectiveness DESC)
        WHERE status IN ('recommended', 'selected', 'active');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for ML-recommended strategies
DO $$ BEGIN
    CREATE INDEX idx_eudr_rma_strat_ml ON gl_eudr_rma_strategies (model_version, confidence_score DESC)
        WHERE mode = 'ml';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_risk_cats ON gl_eudr_rma_strategies USING GIN (risk_categories); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_target_rf ON gl_eudr_rma_strategies USING GIN (target_risk_factors); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_eudr_arts ON gl_eudr_rma_strategies USING GIN (eudr_articles); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_shap ON gl_eudr_rma_strategies USING GIN (shap_explanation); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_measures ON gl_eudr_rma_strategies USING GIN (measure_ids); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_cost ON gl_eudr_rma_strategies USING GIN (cost_estimate); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_prereqs ON gl_eudr_rma_strategies USING GIN (prerequisite_conditions); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_strat_metadata ON gl_eudr_rma_strategies USING GIN (metadata); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_rma_strategies IS 'Mitigation strategies recommended by ML-powered Strategy Selector (Engine 1) or deterministic fallback for EUDR risk mitigation per Article 11(1). Strategies are ranked by predicted effectiveness with SHAP explainability and linked to specific mitigation measures.';
COMMENT ON COLUMN gl_eudr_rma_strategies.iso_31000_type IS 'ISO 31000:2018 risk treatment type: avoid, reduce_likelihood, reduce_consequence, share, retain, remove_source, change_likelihood';
COMMENT ON COLUMN gl_eudr_rma_strategies.confidence_score IS 'ML model confidence (0.000-1.000). Below 0.7 triggers deterministic fallback mode.';
COMMENT ON COLUMN gl_eudr_rma_strategies.shap_explanation IS 'SHAP (SHapley Additive exPlanations) values showing which risk factors drove the recommendation. Contains feature names and their contribution values.';
COMMENT ON COLUMN gl_eudr_rma_strategies.provenance_hash IS 'SHA-256 hash chain linking risk context input, model version, parameters, and recommendation output for audit-grade traceability';


-- ============================================================================
-- 2. gl_eudr_rma_remediation_plans — Structured remediation plan records
-- ============================================================================
RAISE NOTICE 'V113 [2/14]: Creating gl_eudr_rma_remediation_plans...';

CREATE TABLE IF NOT EXISTS gl_eudr_rma_remediation_plans (
    plan_id                     UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id                 UUID            NOT NULL,
        -- Operator owning this remediation plan
    supplier_id                 UUID,
        -- Target supplier for remediation (nullable for portfolio-level plans)
    plan_name                   VARCHAR(500)    NOT NULL,
        -- Human-readable plan name (e.g. "Supplier Capacity Building - Acme Cocoa Ltd")
    risk_finding_ids            JSONB           DEFAULT '[]',
        -- IDs of risk findings from upstream agents that triggered this plan
    strategy_ids                JSONB           DEFAULT '[]',
        -- IDs of selected mitigation strategies driving this plan
    status                      VARCHAR(30)     NOT NULL DEFAULT 'draft',
        -- Plan lifecycle status
    phases                      JSONB           DEFAULT '[]',
        -- Plan phases with start/end dates and status (JSON array of phase objects)
    budget_allocated            NUMERIC(18,2)   DEFAULT 0.00,
        -- Total budget allocated for plan execution (EUR)
    budget_spent                NUMERIC(18,2)   DEFAULT 0.00,
        -- Budget spent to date (EUR)
    start_date                  DATE,
        -- Planned start date
    target_end_date             DATE,
        -- Planned completion date
    actual_end_date             DATE,
        -- Actual completion date (populated when status=completed)
    responsible_parties         JSONB           DEFAULT '[]',
        -- Assigned responsible parties with roles
    escalation_triggers         JSONB           DEFAULT '[]',
        -- Conditions that trigger escalation (budget overrun, timeline breach, risk spike)
    plan_template               VARCHAR(100),
        -- Template used (supplier_capacity_building, emergency_deforestation_response, etc.)
    version                     INTEGER         NOT NULL DEFAULT 1,
        -- Current plan version number
    eudr_articles               JSONB           DEFAULT '[]',
        -- EUDR articles addressed by this remediation plan
    country_code                CHAR(2),
        -- Country of remediation activity (ISO 3166-1 alpha-2)
    commodity                   VARCHAR(50),
        -- Primary commodity under remediation
    risk_category               VARCHAR(50),
        -- Primary risk category being mitigated
    completion_pct              NUMERIC(5,2)    DEFAULT 0.00,
        -- Overall plan completion percentage (0-100)
    provenance_hash             VARCHAR(64)     NOT NULL,
        -- SHA-256 hash for plan record integrity verification
    metadata                    JSONB           DEFAULT '{}',
        -- Additional plan attributes (notes, tags, approval_history)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_rma_plan_status CHECK (status IN (
        'draft', 'active', 'on_track', 'at_risk', 'delayed',
        'completed', 'suspended', 'abandoned'
    )),
    CONSTRAINT chk_rma_plan_budget_alloc CHECK (budget_allocated >= 0),
    CONSTRAINT chk_rma_plan_budget_spent CHECK (budget_spent >= 0),
    CONSTRAINT chk_rma_plan_version CHECK (version >= 1),
    CONSTRAINT chk_rma_plan_completion CHECK (completion_pct >= 0 AND completion_pct <= 100),
    CONSTRAINT chk_rma_plan_template CHECK (plan_template IS NULL OR plan_template IN (
        'supplier_capacity_building', 'emergency_deforestation_response',
        'certification_enrollment', 'enhanced_monitoring_deployment',
        'fpic_remediation', 'legal_gap_closure',
        'anti_corruption_measures', 'buffer_zone_restoration'
    )),
    CONSTRAINT chk_rma_plan_risk_cat CHECK (risk_category IS NULL OR risk_category IN (
        'country', 'supplier', 'commodity', 'corruption',
        'deforestation', 'indigenous_rights', 'protected_areas', 'legal_compliance'
    ))
);

DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_operator ON gl_eudr_rma_remediation_plans (operator_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_supplier ON gl_eudr_rma_remediation_plans (supplier_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_status ON gl_eudr_rma_remediation_plans (status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_template ON gl_eudr_rma_remediation_plans (plan_template); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_country ON gl_eudr_rma_remediation_plans (country_code); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_commodity ON gl_eudr_rma_remediation_plans (commodity); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_risk_cat ON gl_eudr_rma_remediation_plans (risk_category); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_version ON gl_eudr_rma_remediation_plans (version); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_provenance ON gl_eudr_rma_remediation_plans (provenance_hash); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_start ON gl_eudr_rma_remediation_plans (start_date); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_target_end ON gl_eudr_rma_remediation_plans (target_end_date); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_completion ON gl_eudr_rma_remediation_plans (completion_pct DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_tenant ON gl_eudr_rma_remediation_plans (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_created ON gl_eudr_rma_remediation_plans (created_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_op_status ON gl_eudr_rma_remediation_plans (operator_id, status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_sup_status ON gl_eudr_rma_remediation_plans (supplier_id, status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_op_country ON gl_eudr_rma_remediation_plans (operator_id, country_code); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_op_commodity ON gl_eudr_rma_remediation_plans (operator_id, commodity); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active/at-risk/delayed plans requiring attention
DO $$ BEGIN
    CREATE INDEX idx_eudr_rma_plan_active ON gl_eudr_rma_remediation_plans (operator_id, status, completion_pct)
        WHERE status IN ('active', 'on_track', 'at_risk', 'delayed');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for plans approaching target end date
DO $$ BEGIN
    CREATE INDEX idx_eudr_rma_plan_due_soon ON gl_eudr_rma_remediation_plans (target_end_date, status)
        WHERE status NOT IN ('completed', 'suspended', 'abandoned');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_findings ON gl_eudr_rma_remediation_plans USING GIN (risk_finding_ids); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_strategies ON gl_eudr_rma_remediation_plans USING GIN (strategy_ids); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_phases ON gl_eudr_rma_remediation_plans USING GIN (phases); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_parties ON gl_eudr_rma_remediation_plans USING GIN (responsible_parties); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_escalation ON gl_eudr_rma_remediation_plans USING GIN (escalation_triggers); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_eudr_arts ON gl_eudr_rma_remediation_plans USING GIN (eudr_articles); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_plan_metadata ON gl_eudr_rma_remediation_plans USING GIN (metadata); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_rma_remediation_plans IS 'Structured remediation plans generated by Remediation Plan Designer (Engine 2) with multi-phase SMART milestones, budget tracking, and ISO 31000 risk treatment alignment for EUDR Articles 10-11 compliance';
COMMENT ON COLUMN gl_eudr_rma_remediation_plans.status IS 'Plan lifecycle: draft (created), active (approved), on_track/at_risk/delayed (execution), completed/suspended/abandoned (terminal)';
COMMENT ON COLUMN gl_eudr_rma_remediation_plans.plan_template IS 'Template used: supplier_capacity_building, emergency_deforestation_response, certification_enrollment, enhanced_monitoring_deployment, fpic_remediation, legal_gap_closure, anti_corruption_measures, buffer_zone_restoration';
COMMENT ON COLUMN gl_eudr_rma_remediation_plans.phases IS 'JSON array of plan phases, each containing phase_id, name, start_date, end_date, status, and linked milestone IDs';


-- ============================================================================
-- 3. gl_eudr_rma_plan_milestones — SMART milestones for remediation plans
-- ============================================================================
RAISE NOTICE 'V113 [3/14]: Creating gl_eudr_rma_plan_milestones...';

CREATE TABLE IF NOT EXISTS gl_eudr_rma_plan_milestones (
    milestone_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id                     UUID            NOT NULL REFERENCES gl_eudr_rma_remediation_plans(plan_id),
        -- Parent remediation plan
    name                        VARCHAR(500)    NOT NULL,
        -- Milestone name (Specific component of SMART)
    description                 TEXT,
        -- Detailed milestone description
    phase                       VARCHAR(100),
        -- Plan phase this milestone belongs to
    due_date                    DATE            NOT NULL,
        -- Milestone deadline (Time-bound component of SMART)
    completed_date              DATE,
        -- Actual completion date (populated on completion)
    status                      VARCHAR(30)     NOT NULL DEFAULT 'pending',
        -- Milestone lifecycle status
    kpi_name                    VARCHAR(200),
        -- Key performance indicator name (Measurable component of SMART)
    kpi_target                  NUMERIC(10,2),
        -- Target KPI value
    kpi_actual                  NUMERIC(10,2),
        -- Actual KPI value achieved
    kpi_unit                    VARCHAR(50),
        -- KPI measurement unit (percentage, count, days, hectares, EUR)
    evidence_required           JSONB           DEFAULT '[]',
        -- Types of evidence required for milestone verification
    evidence_uploaded           JSONB           DEFAULT '[]',
        -- Evidence items already uploaded
    eudr_article                VARCHAR(20),
        -- Relevant EUDR article (Relevant component of SMART)
    dependency_ids              JSONB           DEFAULT '[]',
        -- IDs of milestones that must complete before this one
    assigned_to                 VARCHAR(100),
        -- Person or role responsible for this milestone
    notes                       TEXT,
        -- Progress notes and commentary
    metadata                    JSONB           DEFAULT '{}',
        -- Additional milestone attributes
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_rma_ms_status CHECK (status IN (
        'pending', 'in_progress', 'completed', 'overdue', 'skipped'
    ))
);

DO $$ BEGIN CREATE INDEX idx_eudr_rma_ms_plan ON gl_eudr_rma_plan_milestones (plan_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_ms_status ON gl_eudr_rma_plan_milestones (status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_ms_due ON gl_eudr_rma_plan_milestones (due_date); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_ms_phase ON gl_eudr_rma_plan_milestones (phase); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_ms_assigned ON gl_eudr_rma_plan_milestones (assigned_to); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_ms_eudr_art ON gl_eudr_rma_plan_milestones (eudr_article); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_ms_tenant ON gl_eudr_rma_plan_milestones (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_ms_created ON gl_eudr_rma_plan_milestones (created_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_ms_plan_status ON gl_eudr_rma_plan_milestones (plan_id, status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_ms_plan_due ON gl_eudr_rma_plan_milestones (plan_id, due_date); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_ms_plan_phase ON gl_eudr_rma_plan_milestones (plan_id, phase); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for overdue milestones requiring immediate attention
DO $$ BEGIN
    CREATE INDEX idx_eudr_rma_ms_overdue ON gl_eudr_rma_plan_milestones (plan_id, due_date)
        WHERE status = 'overdue';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for pending/in_progress milestones
DO $$ BEGIN
    CREATE INDEX idx_eudr_rma_ms_open ON gl_eudr_rma_plan_milestones (plan_id, status, due_date)
        WHERE status IN ('pending', 'in_progress');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_ms_evidence_req ON gl_eudr_rma_plan_milestones USING GIN (evidence_required); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_ms_evidence_up ON gl_eudr_rma_plan_milestones USING GIN (evidence_uploaded); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_ms_deps ON gl_eudr_rma_plan_milestones USING GIN (dependency_ids); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_ms_metadata ON gl_eudr_rma_plan_milestones USING GIN (metadata); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_rma_plan_milestones IS 'SMART milestones (Specific, Measurable, Achievable, Relevant, Time-bound) for remediation plans with KPI tracking, evidence requirements, dependency management, and EUDR article linkage';
COMMENT ON COLUMN gl_eudr_rma_plan_milestones.status IS 'Milestone status: pending (not started), in_progress (underway), completed (verified), overdue (past due_date), skipped (no longer applicable)';
COMMENT ON COLUMN gl_eudr_rma_plan_milestones.kpi_target IS 'Quantified KPI target value (Measurable). For example: 80.00 for 80% completion rate, 10.00 for 10 hectares restored.';


-- ============================================================================
-- 4. gl_eudr_rma_capacity_programs — Capacity building program definitions
-- ============================================================================
RAISE NOTICE 'V113 [4/14]: Creating gl_eudr_rma_capacity_programs...';

CREATE TABLE IF NOT EXISTS gl_eudr_rma_capacity_programs (
    program_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id                 UUID            NOT NULL,
        -- Operator sponsoring the capacity building program
    program_name                VARCHAR(500)    NOT NULL,
        -- Program display name (e.g. "Coffee Smallholder Capacity Building - Colombia")
    description                 TEXT,
        -- Detailed program description and objectives
    commodity                   VARCHAR(50)     NOT NULL,
        -- Target commodity for training content
    target_tier                 INTEGER         NOT NULL DEFAULT 1,
        -- Target capacity tier (1=Awareness, 2=Basic Compliance, 3=Advanced, 4=Leadership)
    modules                     JSONB           NOT NULL DEFAULT '[]',
        -- Training modules with content metadata, duration, and assessment criteria
    total_modules               INTEGER         NOT NULL DEFAULT 22,
        -- Total number of modules in the program
    resource_allocation         JSONB           DEFAULT '{}',
        -- Resources allocated (field trainers, agronomists, GIS specialists, legal advisors)
    budget_allocated            NUMERIC(18,2)   DEFAULT 0.00,
        -- Program budget (EUR)
    budget_spent                NUMERIC(18,2)   DEFAULT 0.00,
        -- Budget spent to date (EUR)
    start_date                  DATE,
        -- Program start date
    end_date                    DATE,
        -- Planned program end date
    enrollment_count            INTEGER         DEFAULT 0,
        -- Number of suppliers enrolled
    completion_count            INTEGER         DEFAULT 0,
        -- Number of suppliers who completed the program
    avg_competency_score        NUMERIC(5,2)    DEFAULT 0.00,
        -- Average competency score across enrolled suppliers
    country_code                CHAR(2),
        -- Target country (ISO 3166-1 alpha-2)
    status                      VARCHAR(30)     NOT NULL DEFAULT 'planning',
        -- Program lifecycle status
    metadata                    JSONB           DEFAULT '{}',
        -- Additional program attributes (tags, notes, partner_orgs)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_rma_cp_tier CHECK (target_tier BETWEEN 1 AND 4),
    CONSTRAINT chk_rma_cp_status CHECK (status IN (
        'planning', 'active', 'paused', 'completed', 'cancelled'
    )),
    CONSTRAINT chk_rma_cp_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_rma_cp_budget_alloc CHECK (budget_allocated >= 0),
    CONSTRAINT chk_rma_cp_budget_spent CHECK (budget_spent >= 0),
    CONSTRAINT chk_rma_cp_modules CHECK (total_modules >= 0),
    CONSTRAINT chk_rma_cp_enroll CHECK (enrollment_count >= 0),
    CONSTRAINT chk_rma_cp_complete CHECK (completion_count >= 0),
    CONSTRAINT chk_rma_cp_competency CHECK (avg_competency_score >= 0 AND avg_competency_score <= 100)
);

DO $$ BEGIN CREATE INDEX idx_eudr_rma_cp_operator ON gl_eudr_rma_capacity_programs (operator_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_cp_commodity ON gl_eudr_rma_capacity_programs (commodity); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_cp_tier ON gl_eudr_rma_capacity_programs (target_tier); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_cp_status ON gl_eudr_rma_capacity_programs (status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_cp_country ON gl_eudr_rma_capacity_programs (country_code); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_cp_tenant ON gl_eudr_rma_capacity_programs (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_cp_created ON gl_eudr_rma_capacity_programs (created_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_cp_op_status ON gl_eudr_rma_capacity_programs (operator_id, status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_cp_op_commodity ON gl_eudr_rma_capacity_programs (operator_id, commodity); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_cp_competency ON gl_eudr_rma_capacity_programs (avg_competency_score DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active programs
DO $$ BEGIN
    CREATE INDEX idx_eudr_rma_cp_active ON gl_eudr_rma_capacity_programs (operator_id, commodity, target_tier)
        WHERE status = 'active';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_cp_modules ON gl_eudr_rma_capacity_programs USING GIN (modules); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_cp_resources ON gl_eudr_rma_capacity_programs USING GIN (resource_allocation); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_cp_metadata ON gl_eudr_rma_capacity_programs USING GIN (metadata); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_rma_capacity_programs IS 'Capacity building program definitions managed by Supplier Capacity Builder (Engine 3) with 4-tier training framework (Awareness, Basic, Advanced, Leadership), commodity-specific modules, and resource allocation tracking';
COMMENT ON COLUMN gl_eudr_rma_capacity_programs.target_tier IS 'Capacity tier: 1=Awareness (basic EUDR understanding), 2=Basic Compliance (minimum requirements), 3=Advanced Practices (best practices), 4=Leadership (sector leadership)';
COMMENT ON COLUMN gl_eudr_rma_capacity_programs.commodity IS 'EUDR regulated commodity: cattle, cocoa, coffee, oil_palm, rubber, soya, wood (Regulation 2023/1115 Annex I)';


-- ============================================================================
-- 5. gl_eudr_rma_mitigation_measures — 500+ mitigation measure library
-- ============================================================================
RAISE NOTICE 'V113 [5/14]: Creating gl_eudr_rma_mitigation_measures...';

CREATE TABLE IF NOT EXISTS gl_eudr_rma_mitigation_measures (
    measure_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    name                        VARCHAR(500)    NOT NULL,
        -- Measure name (e.g. "Satellite Deforestation Monitoring Deployment")
    description                 TEXT            NOT NULL,
        -- Detailed measure description with implementation approach
    risk_category               VARCHAR(50)     NOT NULL,
        -- Primary risk category addressed
    sub_category                VARCHAR(100),
        -- Risk sub-category for finer classification
    target_risk_factors         JSONB           DEFAULT '[]',
        -- Specific risk factors targeted
    applicability               JSONB           DEFAULT '{}',
        -- Applicability criteria: commodities, countries, supply_chain_types, operator_sizes
    effectiveness_evidence      JSONB           DEFAULT '[]',
        -- Evidence of effectiveness from case studies, research, and certifier feedback
    effectiveness_rating        NUMERIC(5,2)    DEFAULT 0.00,
        -- Aggregate effectiveness rating (0-100) from evidence base
    cost_estimate_min           NUMERIC(18,2),
        -- Minimum cost estimate (EUR)
    cost_estimate_max           NUMERIC(18,2),
        -- Maximum cost estimate (EUR)
    cost_currency               VARCHAR(3)      DEFAULT 'EUR',
        -- Cost estimate currency
    implementation_complexity   VARCHAR(20)     NOT NULL DEFAULT 'medium',
        -- Implementation difficulty
    time_to_effect_weeks        INTEGER         DEFAULT 8,
        -- Expected weeks until measurable risk reduction
    prerequisite_conditions     JSONB           DEFAULT '[]',
        -- Conditions that must be met before deploying this measure
    expected_risk_reduction_min NUMERIC(5,2),
        -- Minimum expected risk reduction percentage (0-100)
    expected_risk_reduction_max NUMERIC(5,2),
        -- Maximum expected risk reduction percentage (0-100)
    iso_31000_type              VARCHAR(50),
        -- ISO 31000:2018 risk treatment type
    eudr_articles               JSONB           DEFAULT '[]',
        -- EUDR articles this measure helps satisfy
    certification_schemes       JSONB           DEFAULT '[]',
        -- Certification schemes that recognize this measure (FSC, RSPO, PEFC, etc.)
    tags                        JSONB           DEFAULT '[]',
        -- Categorization tags for search and filtering
    version                     VARCHAR(20)     DEFAULT '1.0.0',
        -- Measure version (for content updates)
    is_active                   BOOLEAN         NOT NULL DEFAULT TRUE,
        -- Whether measure is available for recommendation
    contributed_by              VARCHAR(200),
        -- Source of measure (GreenLang, community, certification_body)
    review_status               VARCHAR(30)     DEFAULT 'approved',
        -- Review status for community-contributed measures
    metadata                    JSONB           DEFAULT '{}',
        -- Additional measure attributes
    tenant_id                   UUID            NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_rma_mm_risk_cat CHECK (risk_category IN (
        'country', 'supplier', 'commodity', 'corruption',
        'deforestation', 'indigenous_rights', 'protected_areas', 'legal_compliance'
    )),
    CONSTRAINT chk_rma_mm_complexity CHECK (implementation_complexity IN (
        'low', 'medium', 'high', 'very_high'
    )),
    CONSTRAINT chk_rma_mm_iso_type CHECK (iso_31000_type IS NULL OR iso_31000_type IN (
        'avoid', 'reduce_likelihood', 'reduce_consequence',
        'share', 'retain', 'remove_source', 'change_likelihood'
    )),
    CONSTRAINT chk_rma_mm_effectiveness CHECK (effectiveness_rating >= 0 AND effectiveness_rating <= 100),
    CONSTRAINT chk_rma_mm_reduction_min CHECK (expected_risk_reduction_min IS NULL OR (expected_risk_reduction_min >= 0 AND expected_risk_reduction_min <= 100)),
    CONSTRAINT chk_rma_mm_reduction_max CHECK (expected_risk_reduction_max IS NULL OR (expected_risk_reduction_max >= 0 AND expected_risk_reduction_max <= 100)),
    CONSTRAINT chk_rma_mm_review CHECK (review_status IN ('pending', 'approved', 'rejected')),
    CONSTRAINT chk_rma_mm_time CHECK (time_to_effect_weeks >= 0)
);

DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_risk_cat ON gl_eudr_rma_mitigation_measures (risk_category); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_sub_cat ON gl_eudr_rma_mitigation_measures (sub_category); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_complexity ON gl_eudr_rma_mitigation_measures (implementation_complexity); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_iso_type ON gl_eudr_rma_mitigation_measures (iso_31000_type); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_active ON gl_eudr_rma_mitigation_measures (is_active); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_effectiveness ON gl_eudr_rma_mitigation_measures (effectiveness_rating DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_cost_min ON gl_eudr_rma_mitigation_measures (cost_estimate_min); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_cost_max ON gl_eudr_rma_mitigation_measures (cost_estimate_max); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_time ON gl_eudr_rma_mitigation_measures (time_to_effect_weeks); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_review ON gl_eudr_rma_mitigation_measures (review_status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_version ON gl_eudr_rma_mitigation_measures (version); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_tenant ON gl_eudr_rma_mitigation_measures (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_created ON gl_eudr_rma_mitigation_measures (created_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_cat_complex ON gl_eudr_rma_mitigation_measures (risk_category, implementation_complexity); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_cat_effect ON gl_eudr_rma_mitigation_measures (risk_category, effectiveness_rating DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active measures only
DO $$ BEGIN
    CREATE INDEX idx_eudr_rma_mm_active_cat ON gl_eudr_rma_mitigation_measures (risk_category, effectiveness_rating DESC)
        WHERE is_active = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Full-text search index on measure name + description
DO $$ BEGIN
    CREATE INDEX idx_eudr_rma_mm_fts ON gl_eudr_rma_mitigation_measures
        USING GIN (to_tsvector('english', name || ' ' || description));
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_target_rf ON gl_eudr_rma_mitigation_measures USING GIN (target_risk_factors); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_applicability ON gl_eudr_rma_mitigation_measures USING GIN (applicability); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_evidence ON gl_eudr_rma_mitigation_measures USING GIN (effectiveness_evidence); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_eudr_arts ON gl_eudr_rma_mitigation_measures USING GIN (eudr_articles); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_schemes ON gl_eudr_rma_mitigation_measures USING GIN (certification_schemes); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_tags ON gl_eudr_rma_mitigation_measures USING GIN (tags); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_prereqs ON gl_eudr_rma_mitigation_measures USING GIN (prerequisite_conditions); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_mm_metadata ON gl_eudr_rma_mitigation_measures USING GIN (metadata); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_rma_mitigation_measures IS '500+ proven mitigation measures organized across 8 EUDR risk categories, searchable via PostgreSQL full-text search (GIN index), with effectiveness evidence, cost estimates, ISO 31000 classification, and certification scheme mapping. Managed by Measure Library Engine (Engine 4).';
COMMENT ON COLUMN gl_eudr_rma_mitigation_measures.risk_category IS 'Risk categories: country, supplier, commodity, corruption, deforestation, indigenous_rights, protected_areas, legal_compliance';
COMMENT ON COLUMN gl_eudr_rma_mitigation_measures.effectiveness_rating IS 'Aggregate effectiveness rating (0-100) computed from evidence base. Higher rating indicates stronger evidence of risk reduction.';
COMMENT ON COLUMN gl_eudr_rma_mitigation_measures.applicability IS 'Applicability criteria JSON: {commodities: [...], countries: [...], supply_chain_types: [...], operator_sizes: [...]}';


-- ============================================================================
-- 6. gl_eudr_rma_stakeholders — Stakeholder registry for collaboration
-- ============================================================================
RAISE NOTICE 'V113 [6/14]: Creating gl_eudr_rma_stakeholders...';

CREATE TABLE IF NOT EXISTS gl_eudr_rma_stakeholders (
    stakeholder_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id                 UUID            NOT NULL,
        -- Operator under which this stakeholder participates
    name                        VARCHAR(500)    NOT NULL,
        -- Stakeholder display name or organization name
    role                        VARCHAR(50)     NOT NULL,
        -- Stakeholder role classification
    organization                VARCHAR(500),
        -- Organization or company name
    contact_email               VARCHAR(500),
        -- Contact email (AES-256 encrypted via SEC-003)
    permissions                 JSONB           DEFAULT '[]',
        -- Role-based permissions for collaboration hub access
    plan_ids                    JSONB           DEFAULT '[]',
        -- Plans this stakeholder is involved in
    notification_preferences    JSONB           DEFAULT '{}',
        -- Notification channel preferences (email, sms, portal)
    is_active                   BOOLEAN         NOT NULL DEFAULT TRUE,
        -- Whether stakeholder is currently active
    last_activity_at            TIMESTAMPTZ,
        -- Timestamp of last activity in the collaboration hub
    metadata                    JSONB           DEFAULT '{}',
        -- Additional stakeholder attributes
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_rma_sh_role CHECK (role IN (
        'compliance_officer', 'procurement_manager', 'supplier_quality',
        'supplier_contact', 'ngo_partner', 'certification_body',
        'competent_authority', 'legal_advisor', 'field_auditor', 'administrator'
    ))
);

DO $$ BEGIN CREATE INDEX idx_eudr_rma_sh_operator ON gl_eudr_rma_stakeholders (operator_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_sh_role ON gl_eudr_rma_stakeholders (role); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_sh_org ON gl_eudr_rma_stakeholders (organization); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_sh_active ON gl_eudr_rma_stakeholders (is_active); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_sh_tenant ON gl_eudr_rma_stakeholders (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_sh_created ON gl_eudr_rma_stakeholders (created_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_sh_op_role ON gl_eudr_rma_stakeholders (operator_id, role); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_sh_last_act ON gl_eudr_rma_stakeholders (last_activity_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active stakeholders
DO $$ BEGIN
    CREATE INDEX idx_eudr_rma_sh_active_role ON gl_eudr_rma_stakeholders (operator_id, role)
        WHERE is_active = TRUE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_sh_permissions ON gl_eudr_rma_stakeholders USING GIN (permissions); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_sh_plans ON gl_eudr_rma_stakeholders USING GIN (plan_ids); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_sh_notif ON gl_eudr_rma_stakeholders USING GIN (notification_preferences); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_sh_metadata ON gl_eudr_rma_stakeholders USING GIN (metadata); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_rma_stakeholders IS 'Stakeholder registry for the multi-stakeholder Collaboration Hub (Engine 8) connecting compliance teams, procurement, suppliers, NGOs, certifiers, and authorities around shared mitigation objectives with role-based access';
COMMENT ON COLUMN gl_eudr_rma_stakeholders.role IS 'Roles: compliance_officer, procurement_manager, supplier_quality, supplier_contact, ngo_partner, certification_body, competent_authority, legal_advisor, field_auditor, administrator';


-- ============================================================================
-- 7. gl_eudr_rma_communications — Stakeholder communication messages
-- ============================================================================
RAISE NOTICE 'V113 [7/14]: Creating gl_eudr_rma_communications...';

CREATE TABLE IF NOT EXISTS gl_eudr_rma_communications (
    communication_id            UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id                     UUID            NOT NULL REFERENCES gl_eudr_rma_remediation_plans(plan_id),
        -- Parent remediation plan context
    sender_id                   UUID            NOT NULL,
        -- Stakeholder ID of the sender
    sender_role                 VARCHAR(50)     NOT NULL,
        -- Role of the sender
    message_type                VARCHAR(30)     NOT NULL DEFAULT 'text',
        -- Type of communication
    subject                     VARCHAR(500),
        -- Message subject line (nullable for real-time chat)
    content                     TEXT            NOT NULL,
        -- Message body content
    attachments                 JSONB           DEFAULT '[]',
        -- File attachment references (evidence_id or S3 key)
    mentions                    JSONB           DEFAULT '[]',
        -- Mentioned stakeholder IDs
    read_by                     JSONB           DEFAULT '[]',
        -- Stakeholder IDs that have read this message
    priority                    VARCHAR(20)     DEFAULT 'normal',
        -- Message priority level
    thread_id                   UUID,
        -- Parent message ID for threaded conversations
    metadata                    JSONB           DEFAULT '{}',
        -- Additional communication attributes
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_rma_comm_type CHECK (message_type IN (
        'text', 'task_update', 'evidence_upload', 'system_notification',
        'escalation', 'approval_request', 'status_update'
    )),
    CONSTRAINT chk_rma_comm_priority CHECK (priority IN ('low', 'normal', 'high', 'urgent'))
);

DO $$ BEGIN CREATE INDEX idx_eudr_rma_comm_plan ON gl_eudr_rma_communications (plan_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_comm_sender ON gl_eudr_rma_communications (sender_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_comm_role ON gl_eudr_rma_communications (sender_role); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_comm_type ON gl_eudr_rma_communications (message_type); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_comm_priority ON gl_eudr_rma_communications (priority); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_comm_thread ON gl_eudr_rma_communications (thread_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_comm_tenant ON gl_eudr_rma_communications (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_comm_created ON gl_eudr_rma_communications (created_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_comm_plan_type ON gl_eudr_rma_communications (plan_id, message_type); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_comm_plan_created ON gl_eudr_rma_communications (plan_id, created_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for unread urgent messages
DO $$ BEGIN
    CREATE INDEX idx_eudr_rma_comm_urgent ON gl_eudr_rma_communications (plan_id, created_at DESC)
        WHERE priority = 'urgent';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_comm_attachments ON gl_eudr_rma_communications USING GIN (attachments); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_comm_mentions ON gl_eudr_rma_communications USING GIN (mentions); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_comm_read_by ON gl_eudr_rma_communications USING GIN (read_by); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_comm_metadata ON gl_eudr_rma_communications USING GIN (metadata); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_rma_communications IS 'Stakeholder communication messages for the Collaboration Hub with threaded conversations, file attachments, @mentions, read receipts, and priority classification for multi-party mitigation coordination';
COMMENT ON COLUMN gl_eudr_rma_communications.message_type IS 'Types: text (free-form), task_update (milestone progress), evidence_upload (new evidence), system_notification (automated), escalation (escalation alert), approval_request (approval workflow), status_update (plan status change)';


-- ============================================================================
-- 8. gl_eudr_rma_supplier_enrollments — Supplier capacity building enrollment
-- ============================================================================
RAISE NOTICE 'V113 [8/14]: Creating gl_eudr_rma_supplier_enrollments...';

CREATE TABLE IF NOT EXISTS gl_eudr_rma_supplier_enrollments (
    enrollment_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id                 UUID            NOT NULL,
        -- Enrolled supplier
    program_id                  UUID            NOT NULL REFERENCES gl_eudr_rma_capacity_programs(program_id),
        -- Parent capacity building program
    commodity                   VARCHAR(50)     NOT NULL,
        -- Commodity for this enrollment
    current_tier                INTEGER         NOT NULL DEFAULT 1,
        -- Current capacity tier (1-4)
    modules_completed           INTEGER         NOT NULL DEFAULT 0,
        -- Number of training modules completed
    modules_total               INTEGER         NOT NULL DEFAULT 22,
        -- Total modules in program
    competency_scores           JSONB           DEFAULT '{}',
        -- Module-level competency scores (module_id -> score)
    avg_competency_score        NUMERIC(5,2)    DEFAULT 0.00,
        -- Average competency score across completed modules
    enrolled_date               DATE            NOT NULL,
        -- Date of enrollment
    target_completion_date      DATE,
        -- Planned completion date
    actual_completion_date      DATE,
        -- Actual completion date (populated on completion)
    status                      VARCHAR(30)     NOT NULL DEFAULT 'active',
        -- Enrollment lifecycle status
    risk_score_at_enrollment    NUMERIC(5,2),
        -- Supplier risk score at time of enrollment (from EUDR-017)
    current_risk_score          NUMERIC(5,2),
        -- Supplier current risk score (updated periodically)
    risk_reduction_pct          NUMERIC(5,2),
        -- Calculated risk reduction since enrollment
    certificate_issued          BOOLEAN         DEFAULT FALSE,
        -- Whether completion certificate has been issued
    certificate_date            DATE,
        -- Date certificate was issued
    trainer_notes               TEXT,
        -- Notes from assigned trainers
    country_code                CHAR(2),
        -- Country of supplier (for filtering)
    metadata                    JSONB           DEFAULT '{}',
        -- Additional enrollment attributes (attendance, session_dates, assessment_results)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_rma_se_tier CHECK (current_tier BETWEEN 1 AND 4),
    CONSTRAINT chk_rma_se_status CHECK (status IN (
        'active', 'paused', 'completed', 'withdrawn'
    )),
    CONSTRAINT chk_rma_se_commodity CHECK (commodity IN (
        'cattle', 'cocoa', 'coffee', 'oil_palm', 'rubber', 'soya', 'wood'
    )),
    CONSTRAINT chk_rma_se_modules CHECK (modules_completed >= 0 AND modules_completed <= modules_total),
    CONSTRAINT chk_rma_se_competency CHECK (avg_competency_score >= 0 AND avg_competency_score <= 100),
    CONSTRAINT chk_rma_se_risk_enroll CHECK (risk_score_at_enrollment IS NULL OR (risk_score_at_enrollment >= 0 AND risk_score_at_enrollment <= 100)),
    CONSTRAINT chk_rma_se_risk_current CHECK (current_risk_score IS NULL OR (current_risk_score >= 0 AND current_risk_score <= 100)),
    CONSTRAINT chk_rma_se_risk_reduce CHECK (risk_reduction_pct IS NULL OR (risk_reduction_pct >= -100 AND risk_reduction_pct <= 100))
);

DO $$ BEGIN CREATE INDEX idx_eudr_rma_se_supplier ON gl_eudr_rma_supplier_enrollments (supplier_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_se_program ON gl_eudr_rma_supplier_enrollments (program_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_se_commodity ON gl_eudr_rma_supplier_enrollments (commodity); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_se_tier ON gl_eudr_rma_supplier_enrollments (current_tier); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_se_status ON gl_eudr_rma_supplier_enrollments (status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_se_country ON gl_eudr_rma_supplier_enrollments (country_code); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_se_enrolled ON gl_eudr_rma_supplier_enrollments (enrolled_date); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_se_target_end ON gl_eudr_rma_supplier_enrollments (target_completion_date); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_se_cert ON gl_eudr_rma_supplier_enrollments (certificate_issued); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_se_competency ON gl_eudr_rma_supplier_enrollments (avg_competency_score DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_se_risk_reduce ON gl_eudr_rma_supplier_enrollments (risk_reduction_pct DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_se_tenant ON gl_eudr_rma_supplier_enrollments (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_se_created ON gl_eudr_rma_supplier_enrollments (created_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_se_sup_status ON gl_eudr_rma_supplier_enrollments (supplier_id, status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_se_prog_status ON gl_eudr_rma_supplier_enrollments (program_id, status); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_se_com_tier ON gl_eudr_rma_supplier_enrollments (commodity, current_tier); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for active enrollments
DO $$ BEGIN
    CREATE INDEX idx_eudr_rma_se_active ON gl_eudr_rma_supplier_enrollments (supplier_id, commodity, current_tier)
        WHERE status = 'active';
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for enrollments needing certificate issuance
DO $$ BEGIN
    CREATE INDEX idx_eudr_rma_se_cert_pending ON gl_eudr_rma_supplier_enrollments (supplier_id)
        WHERE status = 'completed' AND certificate_issued = FALSE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_se_scores ON gl_eudr_rma_supplier_enrollments USING GIN (competency_scores); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_se_metadata ON gl_eudr_rma_supplier_enrollments USING GIN (metadata); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_rma_supplier_enrollments IS 'Individual supplier enrollments in capacity building programs with tier progression tracking, competency assessment scores, risk score correlation, and certificate issuance management';
COMMENT ON COLUMN gl_eudr_rma_supplier_enrollments.current_tier IS 'Capacity tier: 1=Awareness, 2=Basic Compliance, 3=Advanced Practices, 4=Leadership. Advancement requires minimum competency and module completion per tier gate thresholds.';
COMMENT ON COLUMN gl_eudr_rma_supplier_enrollments.risk_reduction_pct IS 'Calculated risk reduction since enrollment: ((risk_score_at_enrollment - current_risk_score) / risk_score_at_enrollment * 100). Negative values indicate risk increase.';


-- ============================================================================
-- 9. gl_eudr_rma_plan_changes — Plan version history and change tracking
-- ============================================================================
RAISE NOTICE 'V113 [9/14]: Creating gl_eudr_rma_plan_changes...';

CREATE TABLE IF NOT EXISTS gl_eudr_rma_plan_changes (
    change_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id                     UUID            NOT NULL REFERENCES gl_eudr_rma_remediation_plans(plan_id),
        -- Parent remediation plan
    version_number              INTEGER         NOT NULL,
        -- Plan version after this change
    change_type                 VARCHAR(50)     NOT NULL,
        -- Type of change made
    plan_snapshot               JSONB           NOT NULL,
        -- Complete plan state at this version (for rollback)
    change_summary              TEXT,
        -- Human-readable summary of what changed
    change_details              JSONB           DEFAULT '{}',
        -- Structured diff of changes (field -> {before, after})
    changed_by                  VARCHAR(100)    NOT NULL,
        -- User who made the change
    change_reason               TEXT,
        -- Reason for the change (e.g. "Trigger event: country reclassification")
    approved_by                 VARCHAR(100),
        -- Approver (if change requires approval)
    approved_at                 TIMESTAMPTZ,
        -- Approval timestamp
    trigger_event_id            UUID,
        -- ID of trigger event that caused this change (from monitoring_events)
    provenance_hash             VARCHAR(64)     NOT NULL,
        -- SHA-256 hash for version integrity
    metadata                    JSONB           DEFAULT '{}',
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_rma_pc_type CHECK (change_type IN (
        'creation', 'status_change', 'timeline_change', 'budget_change',
        'milestone_update', 'scope_change', 'strategy_change',
        'adaptive_adjustment', 'escalation', 'approval'
    )),
    CONSTRAINT chk_rma_pc_version CHECK (version_number >= 1)
);

DO $$ BEGIN CREATE INDEX idx_eudr_rma_pc_plan ON gl_eudr_rma_plan_changes (plan_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_pc_version ON gl_eudr_rma_plan_changes (version_number); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_pc_type ON gl_eudr_rma_plan_changes (change_type); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_pc_changed_by ON gl_eudr_rma_plan_changes (changed_by); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_pc_approved_by ON gl_eudr_rma_plan_changes (approved_by); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_pc_trigger ON gl_eudr_rma_plan_changes (trigger_event_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_pc_provenance ON gl_eudr_rma_plan_changes (provenance_hash); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_pc_tenant ON gl_eudr_rma_plan_changes (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_pc_created ON gl_eudr_rma_plan_changes (created_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_pc_plan_ver ON gl_eudr_rma_plan_changes (plan_id, version_number DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_pc_plan_type ON gl_eudr_rma_plan_changes (plan_id, change_type); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_pc_snapshot ON gl_eudr_rma_plan_changes USING GIN (plan_snapshot); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_pc_details ON gl_eudr_rma_plan_changes USING GIN (change_details); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_pc_metadata ON gl_eudr_rma_plan_changes USING GIN (metadata); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_rma_plan_changes IS 'Immutable plan version history with complete state snapshots for rollback, change diffs, approval tracking, and trigger event linkage for adaptive management audit trail';
COMMENT ON COLUMN gl_eudr_rma_plan_changes.change_type IS 'Change types: creation, status_change, timeline_change, budget_change, milestone_update, scope_change, strategy_change, adaptive_adjustment, escalation, approval';
COMMENT ON COLUMN gl_eudr_rma_plan_changes.plan_snapshot IS 'Complete JSON snapshot of plan state at this version. Used for rollback and audit comparison.';


-- ============================================================================
-- 10. gl_eudr_rma_reports — Mitigation reports and documentation
-- ============================================================================
RAISE NOTICE 'V113 [10/14]: Creating gl_eudr_rma_reports...';

CREATE TABLE IF NOT EXISTS gl_eudr_rma_reports (
    report_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    operator_id                 UUID            NOT NULL,
        -- Operator owning this report
    report_type                 VARCHAR(50)     NOT NULL,
        -- Type of mitigation report
    report_name                 VARCHAR(500)    NOT NULL,
        -- Human-readable report name
    report_scope                JSONB           DEFAULT '{}',
        -- Scope definition (supplier_ids, plan_ids, date_range, risk_categories)
    report_data                 JSONB           NOT NULL,
        -- Structured report data (content varies by report_type)
    format                      VARCHAR(10)     NOT NULL DEFAULT 'pdf',
        -- Output format
    language                    VARCHAR(5)      NOT NULL DEFAULT 'en',
        -- Report language
    s3_key                      VARCHAR(500),
        -- S3 storage key for rendered report file
    file_size_bytes             BIGINT,
        -- Rendered file size in bytes
    generation_time_ms          INTEGER,
        -- Time taken to generate report (milliseconds)
    provenance_hash             VARCHAR(64)     NOT NULL,
        -- SHA-256 hash for report integrity verification
    supplier_id                 UUID,
        -- Specific supplier (for supplier-level reports)
    plan_id                     UUID,
        -- Specific plan (for plan-level reports)
    country_code                CHAR(2),
        -- Country filter for report scope
    commodity                   VARCHAR(50),
        -- Commodity filter for report scope
    metadata                    JSONB           DEFAULT '{}',
        -- Additional report attributes (requested_by, distribution_list)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_rma_rpt_type CHECK (report_type IN (
        'dds_mitigation', 'authority_package', 'annual_review',
        'supplier_scorecard', 'portfolio_summary',
        'risk_mitigation_mapping', 'effectiveness_analysis'
    )),
    CONSTRAINT chk_rma_rpt_format CHECK (format IN ('pdf', 'json', 'html', 'xlsx', 'xml')),
    CONSTRAINT chk_rma_rpt_language CHECK (language IN ('en', 'fr', 'de', 'es', 'pt'))
);

DO $$ BEGIN CREATE INDEX idx_eudr_rma_rpt_operator ON gl_eudr_rma_reports (operator_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_rpt_type ON gl_eudr_rma_reports (report_type); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_rpt_format ON gl_eudr_rma_reports (format); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_rpt_language ON gl_eudr_rma_reports (language); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_rpt_supplier ON gl_eudr_rma_reports (supplier_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_rpt_plan ON gl_eudr_rma_reports (plan_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_rpt_country ON gl_eudr_rma_reports (country_code); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_rpt_commodity ON gl_eudr_rma_reports (commodity); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_rpt_provenance ON gl_eudr_rma_reports (provenance_hash); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_rpt_tenant ON gl_eudr_rma_reports (tenant_id); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_rpt_created ON gl_eudr_rma_reports (created_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_rpt_op_type ON gl_eudr_rma_reports (operator_id, report_type); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_rpt_op_created ON gl_eudr_rma_reports (operator_id, created_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_rpt_scope ON gl_eudr_rma_reports USING GIN (report_scope); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_rpt_data ON gl_eudr_rma_reports USING GIN (report_data); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_rpt_metadata ON gl_eudr_rma_reports USING GIN (metadata); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_rma_reports IS 'Mitigation reports generated by the Reporting Module for EUDR DDS submissions (Art.12(2)(d)), competent authority packages (Art.14-16), annual reviews (Art.8(3)), supplier scorecards, portfolio summaries, risk-mitigation mappings, and effectiveness analyses. Supports PDF/JSON/HTML/XLSX/XML in 5 EU languages.';
COMMENT ON COLUMN gl_eudr_rma_reports.report_type IS 'Report types: dds_mitigation (Article 12 DDS section), authority_package (Article 14-16 inspection evidence), annual_review (Article 8(3) system review), supplier_scorecard, portfolio_summary, risk_mitigation_mapping, effectiveness_analysis';
COMMENT ON COLUMN gl_eudr_rma_reports.provenance_hash IS 'SHA-256 hash of report_data content for integrity verification. Ensures report content has not been tampered with after generation.';


-- ============================================================================
-- 11. gl_eudr_rma_effectiveness_tracking — Effectiveness records (hypertable)
-- ============================================================================
RAISE NOTICE 'V113 [11/14]: Creating gl_eudr_rma_effectiveness_tracking (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_rma_effectiveness_tracking (
    record_id                   UUID            DEFAULT gen_random_uuid(),
    plan_id                     UUID            NOT NULL,
        -- Remediation plan being measured
    supplier_id                 UUID            NOT NULL,
        -- Supplier under mitigation
    baseline_risk_scores        JSONB           NOT NULL,
        -- T0 risk scores from 9 agents (dimension -> Decimal score)
    current_risk_scores         JSONB           NOT NULL,
        -- Current risk scores from 9 agents
    risk_reduction_pct          JSONB           NOT NULL,
        -- Per-dimension risk reduction percentages
    composite_reduction_pct     NUMERIC(5,2),
        -- Weighted composite risk reduction (0-100)
    predicted_reduction_pct     NUMERIC(5,2),
        -- Predicted reduction from Strategy Selector
    deviation_pct               NUMERIC(5,2),
        -- Deviation: actual - predicted
    roi                         NUMERIC(10,2),
        -- Return on investment: (risk_value - cost) / cost * 100
    cost_to_date                NUMERIC(18,2),
        -- Total mitigation cost at measurement point
    statistical_significance    BOOLEAN         DEFAULT FALSE,
        -- Whether risk reduction is statistically significant (p < 0.05)
    p_value                     NUMERIC(6,4),
        -- p-value from paired t-test
    confidence_interval_lower   NUMERIC(5,2),
        -- 95% confidence interval lower bound
    confidence_interval_upper   NUMERIC(5,2),
        -- 95% confidence interval upper bound
    measure_effectiveness       JSONB           DEFAULT '{}',
        -- Per-measure effectiveness breakdown
    underperforming_measures    JSONB           DEFAULT '[]',
        -- Measures with actual < 50% of predicted
    country_code                CHAR(2),
        -- Country for filtering
    commodity                   VARCHAR(50),
        -- Commodity for filtering
    risk_category               VARCHAR(50),
        -- Primary risk category
    provenance_hash             VARCHAR(64)     NOT NULL,
        -- SHA-256 hash for record integrity
    tracked_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Measurement timestamp (partition key)
    tenant_id                   UUID            NOT NULL,

    PRIMARY KEY (record_id, tracked_at),

    CONSTRAINT chk_rma_et_composite CHECK (composite_reduction_pct IS NULL OR (composite_reduction_pct >= -100 AND composite_reduction_pct <= 100)),
    CONSTRAINT chk_rma_et_predicted CHECK (predicted_reduction_pct IS NULL OR (predicted_reduction_pct >= -100 AND predicted_reduction_pct <= 100)),
    CONSTRAINT chk_rma_et_deviation CHECK (deviation_pct IS NULL OR (deviation_pct >= -200 AND deviation_pct <= 200)),
    CONSTRAINT chk_rma_et_cost CHECK (cost_to_date IS NULL OR cost_to_date >= 0),
    CONSTRAINT chk_rma_et_pvalue CHECK (p_value IS NULL OR (p_value >= 0 AND p_value <= 1))
);

SELECT create_hypertable(
    'gl_eudr_rma_effectiveness_tracking',
    'tracked_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN CREATE INDEX idx_eudr_rma_et_plan ON gl_eudr_rma_effectiveness_tracking (plan_id, tracked_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_et_supplier ON gl_eudr_rma_effectiveness_tracking (supplier_id, tracked_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_et_composite ON gl_eudr_rma_effectiveness_tracking (composite_reduction_pct DESC, tracked_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_et_roi ON gl_eudr_rma_effectiveness_tracking (roi DESC, tracked_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_et_sig ON gl_eudr_rma_effectiveness_tracking (statistical_significance, tracked_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_et_country ON gl_eudr_rma_effectiveness_tracking (country_code, tracked_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_et_commodity ON gl_eudr_rma_effectiveness_tracking (commodity, tracked_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_et_risk_cat ON gl_eudr_rma_effectiveness_tracking (risk_category, tracked_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_et_provenance ON gl_eudr_rma_effectiveness_tracking (provenance_hash); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_et_tenant ON gl_eudr_rma_effectiveness_tracking (tenant_id, tracked_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_et_baseline ON gl_eudr_rma_effectiveness_tracking USING GIN (baseline_risk_scores); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_et_current ON gl_eudr_rma_effectiveness_tracking USING GIN (current_risk_scores); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_et_reduction ON gl_eudr_rma_effectiveness_tracking USING GIN (risk_reduction_pct); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_et_measure_eff ON gl_eudr_rma_effectiveness_tracking USING GIN (measure_effectiveness); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_et_underperf ON gl_eudr_rma_effectiveness_tracking USING GIN (underperforming_measures); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_rma_effectiveness_tracking IS 'Time-series effectiveness tracking hypertable (30-day chunks) recording before/after risk scores, ROI analysis, statistical significance testing, and per-measure effectiveness for closed-loop feedback to Strategy Selector. Managed by Effectiveness Tracker (Engine 5).';


-- ============================================================================
-- 12. gl_eudr_rma_monitoring_events — Adaptive management events (hypertable)
-- ============================================================================
RAISE NOTICE 'V113 [12/14]: Creating gl_eudr_rma_monitoring_events (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_rma_monitoring_events (
    event_id                    UUID            DEFAULT gen_random_uuid(),
    plan_id                     UUID,
        -- Affected remediation plan (nullable for portfolio-level events)
    trigger_type                VARCHAR(50)     NOT NULL,
        -- Type of trigger event detected
    source_agent                VARCHAR(50)     NOT NULL,
        -- Upstream agent that generated the risk signal
    severity                    VARCHAR(20)     NOT NULL,
        -- Event severity classification
    description                 TEXT            NOT NULL,
        -- Human-readable event description
    risk_data                   JSONB           DEFAULT '{}',
        -- Risk data from source agent (risk scores, alert details)
    recommended_adjustment      JSONB           DEFAULT '{}',
        -- Recommended plan adjustment (adjustment_type, actions, rationale)
    adjustment_type             VARCHAR(50),
        -- Type of recommended adjustment
    response_sla_hours          INTEGER,
        -- Required response time in hours per trigger matrix
    acknowledged                BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether event has been acknowledged by an operator
    acknowledged_by             VARCHAR(100),
        -- User who acknowledged the event
    acknowledged_at             TIMESTAMPTZ,
        -- Acknowledgment timestamp
    resolved                    BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether event has been resolved (adjustment implemented)
    resolved_at                 TIMESTAMPTZ,
        -- Resolution timestamp
    resolution_notes            TEXT,
        -- Notes on how the event was resolved
    supplier_id                 UUID,
        -- Affected supplier (denormalized for aggregation)
    country_code                CHAR(2),
        -- Country of event origin
    commodity                   VARCHAR(50),
        -- Commodity involved
    escalation_level            INTEGER         DEFAULT 0,
        -- Current escalation level (0=normal, 1=warning, 2=critical, 3=executive)
    event_timestamp             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Event detection timestamp (partition key)
    tenant_id                   UUID            NOT NULL,

    PRIMARY KEY (event_id, event_timestamp),

    CONSTRAINT chk_rma_me_severity CHECK (severity IN ('critical', 'high', 'medium', 'low')),
    CONSTRAINT chk_rma_me_adjust CHECK (adjustment_type IS NULL OR adjustment_type IN (
        'plan_acceleration', 'scope_expansion', 'strategy_replacement',
        'emergency_response', 'plan_de_escalation'
    )),
    CONSTRAINT chk_rma_me_source CHECK (source_agent IN (
        'EUDR-016', 'EUDR-017', 'EUDR-018', 'EUDR-019', 'EUDR-020',
        'EUDR-021', 'EUDR-022', 'EUDR-023', 'EUDR-024', 'EUDR-025-internal'
    )),
    CONSTRAINT chk_rma_me_escalation CHECK (escalation_level >= 0 AND escalation_level <= 3)
);

SELECT create_hypertable(
    'gl_eudr_rma_monitoring_events',
    'event_timestamp',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN CREATE INDEX idx_eudr_rma_me_plan ON gl_eudr_rma_monitoring_events (plan_id, event_timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_me_trigger ON gl_eudr_rma_monitoring_events (trigger_type, event_timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_me_source ON gl_eudr_rma_monitoring_events (source_agent, event_timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_me_severity ON gl_eudr_rma_monitoring_events (severity, event_timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_me_adjust ON gl_eudr_rma_monitoring_events (adjustment_type, event_timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_me_resolved ON gl_eudr_rma_monitoring_events (resolved, event_timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_me_ack ON gl_eudr_rma_monitoring_events (acknowledged, event_timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_me_supplier ON gl_eudr_rma_monitoring_events (supplier_id, event_timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_me_country ON gl_eudr_rma_monitoring_events (country_code, event_timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_me_commodity ON gl_eudr_rma_monitoring_events (commodity, event_timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_me_escalation ON gl_eudr_rma_monitoring_events (escalation_level, event_timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_me_tenant ON gl_eudr_rma_monitoring_events (tenant_id, event_timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_me_sev_source ON gl_eudr_rma_monitoring_events (severity, source_agent, event_timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for unresolved events requiring attention
DO $$ BEGIN
    CREATE INDEX idx_eudr_rma_me_unresolved ON gl_eudr_rma_monitoring_events (severity, escalation_level DESC, event_timestamp DESC)
        WHERE resolved = FALSE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for critical unacknowledged events
DO $$ BEGIN
    CREATE INDEX idx_eudr_rma_me_crit_unack ON gl_eudr_rma_monitoring_events (event_timestamp DESC)
        WHERE severity = 'critical' AND acknowledged = FALSE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_me_risk_data ON gl_eudr_rma_monitoring_events USING GIN (risk_data); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_me_rec_adj ON gl_eudr_rma_monitoring_events USING GIN (recommended_adjustment); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_rma_monitoring_events IS 'Time-series monitoring event hypertable (30-day chunks) for Continuous Monitoring and Adaptive Management (Engine 6). Records trigger events from 9 upstream risk agents, recommended adjustments, acknowledgment, resolution, and escalation tracking.';
COMMENT ON COLUMN gl_eudr_rma_monitoring_events.trigger_type IS 'Trigger event type from pre-coded Trigger Response Matrix (e.g. critical_deforestation_alert, country_reclassification_high, supplier_risk_spike_50pct, audit_critical_nc, indigenous_rights_violation, protected_area_encroachment)';
COMMENT ON COLUMN gl_eudr_rma_monitoring_events.source_agent IS 'Upstream agent source: EUDR-016 through EUDR-024 or EUDR-025-internal for self-generated events';


-- ============================================================================
-- 13. gl_eudr_rma_optimization_runs — Cost-benefit optimization runs (hypertable)
-- ============================================================================
RAISE NOTICE 'V113 [13/14]: Creating gl_eudr_rma_optimization_runs (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_rma_optimization_runs (
    run_id                      UUID            DEFAULT gen_random_uuid(),
    operator_id                 UUID            NOT NULL,
        -- Operator requesting budget optimization
    budget_total                NUMERIC(18,2)   NOT NULL,
        -- Total budget available for optimization (EUR)
    budget_constraints          JSONB           DEFAULT '{}',
        -- Optimization constraints (per_category_max, per_supplier_max, min_coverage)
    supplier_count              INTEGER         DEFAULT 0,
        -- Number of suppliers in optimization scope
    optimization_result         JSONB           NOT NULL,
        -- Complete optimization result (allocations, measures, expected_reductions)
    pareto_frontier             JSONB           DEFAULT '[]',
        -- Pareto-optimal budget/reduction trade-off points
    sensitivity_analysis        JSONB           DEFAULT '{}',
        -- Budget sensitivity analysis (how allocation changes with +/-10/20% budget)
    total_predicted_risk_reduction NUMERIC(5,2),
        -- Aggregate predicted risk reduction across portfolio
    cost_per_risk_point         NUMERIC(10,2),
        -- Average cost per risk point reduced (EUR)
    solver_status               VARCHAR(30),
        -- LP solver status
    solver_iterations           INTEGER,
        -- Number of solver iterations
    computation_time_ms         INTEGER,
        -- Total computation time in milliseconds
    country_code                CHAR(2),
        -- Country filter for scoped optimization
    commodity                   VARCHAR(50),
        -- Commodity filter for scoped optimization
    risk_category               VARCHAR(50),
        -- Risk category filter
    provenance_hash             VARCHAR(64)     NOT NULL,
        -- SHA-256 hash for run integrity
    run_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Run timestamp (partition key)
    tenant_id                   UUID            NOT NULL,

    PRIMARY KEY (run_id, run_at),

    CONSTRAINT chk_rma_or_budget CHECK (budget_total >= 0),
    CONSTRAINT chk_rma_or_solver CHECK (solver_status IS NULL OR solver_status IN (
        'optimal', 'feasible', 'infeasible', 'unbounded', 'timeout', 'error'
    )),
    CONSTRAINT chk_rma_or_suppliers CHECK (supplier_count >= 0),
    CONSTRAINT chk_rma_or_reduction CHECK (total_predicted_risk_reduction IS NULL OR (total_predicted_risk_reduction >= 0 AND total_predicted_risk_reduction <= 100)),
    CONSTRAINT chk_rma_or_cost_rp CHECK (cost_per_risk_point IS NULL OR cost_per_risk_point >= 0)
);

SELECT create_hypertable(
    'gl_eudr_rma_optimization_runs',
    'run_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN CREATE INDEX idx_eudr_rma_or_operator ON gl_eudr_rma_optimization_runs (operator_id, run_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_or_solver ON gl_eudr_rma_optimization_runs (solver_status, run_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_or_budget ON gl_eudr_rma_optimization_runs (budget_total, run_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_or_reduction ON gl_eudr_rma_optimization_runs (total_predicted_risk_reduction DESC, run_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_or_cost_rp ON gl_eudr_rma_optimization_runs (cost_per_risk_point, run_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_or_country ON gl_eudr_rma_optimization_runs (country_code, run_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_or_commodity ON gl_eudr_rma_optimization_runs (commodity, run_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_or_risk_cat ON gl_eudr_rma_optimization_runs (risk_category, run_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_or_provenance ON gl_eudr_rma_optimization_runs (provenance_hash); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_or_tenant ON gl_eudr_rma_optimization_runs (tenant_id, run_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_or_comp_time ON gl_eudr_rma_optimization_runs (computation_time_ms DESC, run_at DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_or_constraints ON gl_eudr_rma_optimization_runs USING GIN (budget_constraints); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_or_result ON gl_eudr_rma_optimization_runs USING GIN (optimization_result); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_or_pareto ON gl_eudr_rma_optimization_runs USING GIN (pareto_frontier); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_or_sensitivity ON gl_eudr_rma_optimization_runs USING GIN (sensitivity_analysis); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_rma_optimization_runs IS 'Time-series cost-benefit optimization run hypertable (30-day chunks) for Budget Optimizer (Engine 7). Records LP solver runs with Pareto frontier, sensitivity analysis, and per-supplier allocation recommendations.';
COMMENT ON COLUMN gl_eudr_rma_optimization_runs.solver_status IS 'LP solver status: optimal (solution found), feasible (non-optimal), infeasible (no solution), unbounded, timeout (30s exceeded), error';
COMMENT ON COLUMN gl_eudr_rma_optimization_runs.cost_per_risk_point IS 'Average cost per risk point reduced (EUR/point). Lower values indicate more cost-effective mitigation. Target: cost-effectiveness ratio >= 3:1.';


-- ============================================================================
-- 14. gl_eudr_rma_audit_trail — Immutable audit trail (hypertable)
-- ============================================================================
RAISE NOTICE 'V113 [14/14]: Creating gl_eudr_rma_audit_trail (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_rma_audit_trail (
    trail_id                    UUID            DEFAULT gen_random_uuid(),
    entity_type                 VARCHAR(50)     NOT NULL,
        -- Type of entity affected
    entity_id                   UUID            NOT NULL,
        -- ID of the affected entity
    action                      VARCHAR(50)     NOT NULL,
        -- Action performed
    before_value                JSONB,
        -- State before the action
    after_value                 JSONB,
        -- State after the action
    actor                       VARCHAR(100)    NOT NULL,
        -- User or system agent performing the action
    actor_role                  VARCHAR(50),
        -- Role of the actor
    ip_address                  VARCHAR(45),
        -- Source IP address
    change_summary              TEXT,
        -- Human-readable summary of what changed
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for audit integrity chain
    timestamp                   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Event timestamp (partition key)
    tenant_id                   UUID            NOT NULL,

    PRIMARY KEY (trail_id, timestamp),

    CONSTRAINT chk_rma_at_action CHECK (action IN (
        'create', 'update', 'delete', 'archive',
        'verify', 'reject', 'approve', 'escalate',
        'recommend', 'select', 'activate', 'complete',
        'suspend', 'abandon', 'enroll', 'advance_tier',
        'generate_report', 'optimize', 'trigger_event',
        'acknowledge', 'resolve', 'import', 'export'
    )),
    CONSTRAINT chk_rma_at_entity CHECK (entity_type IN (
        'strategy', 'remediation_plan', 'milestone',
        'capacity_program', 'mitigation_measure', 'stakeholder',
        'communication', 'supplier_enrollment', 'plan_change',
        'report', 'effectiveness_record', 'monitoring_event',
        'optimization_run', 'ml_model', 'system'
    ))
);

SELECT create_hypertable(
    'gl_eudr_rma_audit_trail',
    'timestamp',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN CREATE INDEX idx_eudr_rma_at_entity_type ON gl_eudr_rma_audit_trail (entity_type, timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_at_entity_id ON gl_eudr_rma_audit_trail (entity_id, timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_at_action ON gl_eudr_rma_audit_trail (action, timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_at_actor ON gl_eudr_rma_audit_trail (actor, timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_at_actor_role ON gl_eudr_rma_audit_trail (actor_role, timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_at_provenance ON gl_eudr_rma_audit_trail (provenance_hash); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_at_tenant ON gl_eudr_rma_audit_trail (tenant_id, timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_at_entity_action ON gl_eudr_rma_audit_trail (entity_type, action, timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_at_entity_id_type ON gl_eudr_rma_audit_trail (entity_id, entity_type, timestamp DESC); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_at_before ON gl_eudr_rma_audit_trail USING GIN (before_value); EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN CREATE INDEX idx_eudr_rma_at_after ON gl_eudr_rma_audit_trail USING GIN (after_value); EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_rma_audit_trail IS 'Immutable time-series audit trail hypertable (30-day chunks) recording all CRUD operations, state transitions, ML recommendations, optimization runs, and stakeholder actions for EUDR Article 31 record-keeping (5-year minimum). Append-only with provenance hash chain.';
COMMENT ON COLUMN gl_eudr_rma_audit_trail.action IS 'Actions: create/update/delete/archive/verify/reject/approve/escalate/recommend/select/activate/complete/suspend/abandon/enroll/advance_tier/generate_report/optimize/trigger_event/acknowledge/resolve/import/export';
COMMENT ON COLUMN gl_eudr_rma_audit_trail.entity_type IS 'Entity types: strategy, remediation_plan, milestone, capacity_program, mitigation_measure, stakeholder, communication, supplier_enrollment, plan_change, report, effectiveness_record, monitoring_event, optimization_run, ml_model, system';


-- ============================================================================
-- CONTINUOUS AGGREGATES
-- ============================================================================

-- Daily effectiveness summary
RAISE NOTICE 'V113: Creating continuous aggregate: gl_eudr_rma_effectiveness_daily...';

DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_rma_effectiveness_daily
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 day', tracked_at)            AS day,
        tenant_id,
        supplier_id,
        country_code,
        commodity,
        risk_category,
        COUNT(*)                                     AS measurement_count,
        AVG(composite_reduction_pct)                 AS avg_composite_reduction,
        MAX(composite_reduction_pct)                 AS max_composite_reduction,
        AVG(roi)                                     AS avg_roi,
        MAX(roi)                                     AS max_roi,
        SUM(cost_to_date)                            AS total_cost,
        SUM(CASE WHEN statistical_significance THEN 1 ELSE 0 END) AS significant_count,
        AVG(deviation_pct)                           AS avg_deviation
    FROM gl_eudr_rma_effectiveness_tracking
    GROUP BY day, tenant_id, supplier_id, country_code, commodity, risk_category;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_rma_effectiveness_daily',
        start_offset => INTERVAL '7 days',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_rma_effectiveness_daily IS 'Daily rollup of effectiveness tracking by supplier, country, commodity, and risk category with reduction averages, ROI, cost totals, and statistical significance counts';


-- Hourly monitoring events summary
RAISE NOTICE 'V113: Creating continuous aggregate: gl_eudr_rma_events_hourly...';

DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_rma_events_hourly
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 hour', event_timestamp)       AS hour,
        tenant_id,
        source_agent,
        severity,
        trigger_type,
        country_code,
        commodity,
        COUNT(*)                                      AS event_count,
        SUM(CASE WHEN resolved THEN 1 ELSE 0 END)    AS resolved_count,
        SUM(CASE WHEN acknowledged THEN 1 ELSE 0 END) AS acknowledged_count,
        MAX(escalation_level)                          AS max_escalation
    FROM gl_eudr_rma_monitoring_events
    GROUP BY hour, tenant_id, source_agent, severity, trigger_type, country_code, commodity;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_rma_events_hourly',
        start_offset => INTERVAL '3 days',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '30 minutes');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_rma_events_hourly IS 'Hourly rollup of monitoring events by source agent, severity, trigger type, country, and commodity with resolution and acknowledgment counts for adaptive management dashboards';


-- ============================================================================
-- RETENTION POLICIES (5 years per EUDR Article 31)
-- ============================================================================

RAISE NOTICE 'V113: Creating retention policies (5 years per EUDR Article 31)...';

-- 5 years for effectiveness tracking
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_rma_effectiveness_tracking', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for monitoring events
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_rma_monitoring_events', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for optimization runs
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_rma_optimization_runs', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for audit trail
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_rma_audit_trail', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- COMPRESSION POLICIES
-- ============================================================================

RAISE NOTICE 'V113: Creating compression policies on hypertables...';

-- Compression on effectiveness tracking (after 60 days)
DO $$ BEGIN
    ALTER TABLE gl_eudr_rma_effectiveness_tracking SET (timescaledb.compress);
    SELECT add_compression_policy('gl_eudr_rma_effectiveness_tracking', INTERVAL '60 days');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Compression on monitoring events (after 60 days)
DO $$ BEGIN
    ALTER TABLE gl_eudr_rma_monitoring_events SET (timescaledb.compress);
    SELECT add_compression_policy('gl_eudr_rma_monitoring_events', INTERVAL '60 days');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Compression on optimization runs (after 90 days)
DO $$ BEGIN
    ALTER TABLE gl_eudr_rma_optimization_runs SET (timescaledb.compress);
    SELECT add_compression_policy('gl_eudr_rma_optimization_runs', INTERVAL '90 days');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Compression on audit trail (after 60 days)
DO $$ BEGIN
    ALTER TABLE gl_eudr_rma_audit_trail SET (timescaledb.compress);
    SELECT add_compression_policy('gl_eudr_rma_audit_trail', INTERVAL '60 days');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- GRANTS — greenlang_app role
-- ============================================================================

RAISE NOTICE 'V113: Granting permissions to greenlang_app...';

-- Schema access
GRANT USAGE ON SCHEMA eudr_risk_mitigation TO greenlang_app;

-- Regular tables (full CRUD)
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_rma_strategies TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_rma_remediation_plans TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_rma_plan_milestones TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_rma_capacity_programs TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_rma_mitigation_measures TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_rma_stakeholders TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_rma_communications TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_rma_supplier_enrollments TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_rma_plan_changes TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_rma_reports TO greenlang_app;

-- Hypertables (append-only for audit integrity)
GRANT SELECT, INSERT ON gl_eudr_rma_effectiveness_tracking TO greenlang_app;
GRANT SELECT, INSERT ON gl_eudr_rma_monitoring_events TO greenlang_app;
GRANT SELECT, INSERT ON gl_eudr_rma_optimization_runs TO greenlang_app;
GRANT SELECT, INSERT ON gl_eudr_rma_audit_trail TO greenlang_app;

-- Continuous aggregates (read-only)
GRANT SELECT ON gl_eudr_rma_effectiveness_daily TO greenlang_app;
GRANT SELECT ON gl_eudr_rma_events_hourly TO greenlang_app;

-- Read-only role (conditional)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_readonly') THEN
        GRANT USAGE ON SCHEMA eudr_risk_mitigation TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_rma_strategies TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_rma_remediation_plans TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_rma_plan_milestones TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_rma_capacity_programs TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_rma_mitigation_measures TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_rma_stakeholders TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_rma_communications TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_rma_supplier_enrollments TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_rma_plan_changes TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_rma_reports TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_rma_effectiveness_tracking TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_rma_monitoring_events TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_rma_optimization_runs TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_rma_audit_trail TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_rma_effectiveness_daily TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_rma_events_hourly TO greenlang_readonly;
    END IF;
END
$$;


-- ============================================================================
-- FINALIZE
-- ============================================================================

RAISE NOTICE 'V113: AGENT-EUDR-025 Risk Mitigation Advisor tables created successfully!';
RAISE NOTICE 'V113: Created 14 tables (10 regular + 4 hypertables), 2 continuous aggregates, ~190 indexes';
RAISE NOTICE 'V113: ~30 GIN indexes on JSONB columns, ~10 partial indexes for filtered records, 1 full-text search index';
RAISE NOTICE 'V113: Retention policies: 5y on all 4 hypertables per EUDR Article 31';
RAISE NOTICE 'V113: Compression policies: 60-90d on all 4 hypertables';
RAISE NOTICE 'V113: Grants applied for greenlang_app and greenlang_readonly roles';

COMMIT;
