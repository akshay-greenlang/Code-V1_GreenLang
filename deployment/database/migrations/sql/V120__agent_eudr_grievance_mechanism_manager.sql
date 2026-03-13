-- ============================================================================
-- V120: AGENT-EUDR-032 Grievance Mechanism Manager
-- ============================================================================
-- Creates tables for the Grievance Mechanism Manager which provides advanced
-- analytics, mediation, remediation tracking, risk scoring, collective
-- grievance handling, regulatory reporting, and audit-ready operations on top
-- of the basic grievance mechanism provided by EUDR-031 (Stakeholder
-- Engagement Tool). This agent reads from gl_eudr_set_grievances but adds
-- its own analytics, mediation, remediation, risk, collective grievance,
-- and regulatory reporting tables.
--
-- Agent ID: GL-EUDR-GMM-032
-- PRD: PRD-AGENT-EUDR-032
-- Regulation: EU 2023/1115 (EUDR) Articles 2, 4, 8, 9, 10, 11, 12, 29, 31;
--             CSDDD Article 8; UNGP Principle 31; ILO Convention 169
-- Tables: 8 regular + 1 hypertable = 9
-- Indexes: ~120
--
-- Dependencies: TimescaleDB extension (for hypertable), V119 (gl_eudr_set_*)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V120: Creating AGENT-EUDR-032 Grievance Mechanism Manager tables...';


-- ============================================================================
-- 1. gl_eudr_gmm_analytics -- Grievance pattern analysis
-- ============================================================================
RAISE NOTICE 'V120 [1/8]: Creating gl_eudr_gmm_analytics...';

CREATE TABLE IF NOT EXISTS gl_eudr_gmm_analytics (
    analytics_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this pattern analysis record
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator whose grievances are being analyzed
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    analysis_period_start           TIMESTAMPTZ     NOT NULL,
        -- Start of the analysis window
    analysis_period_end             TIMESTAMPTZ     NOT NULL,
        -- End of the analysis window
    grievance_ids                   JSONB           DEFAULT '[]',
        -- Array of EUDR-031 grievance UUIDs analyzed: ["uuid-1", "uuid-2"]
    pattern_type                    VARCHAR(30)     NOT NULL,
        -- Detected grievance pattern classification
    pattern_description             TEXT            NOT NULL DEFAULT '',
        -- Human-readable description of the detected pattern
    affected_stakeholder_count      INTEGER         NOT NULL DEFAULT 0,
        -- Number of unique stakeholders affected by this pattern
    root_causes                     JSONB           DEFAULT '[]',
        -- Array of identified root causes: [{"cause": "...", "confidence": 0.85}, ...]
    recommendations                 JSONB           DEFAULT '[]',
        -- Array of recommended actions: [{"action": "...", "priority": "high"}, ...]
    severity_distribution           JSONB           DEFAULT '{}',
        -- Distribution of severities: {"critical": 2, "high": 5, "medium": 10, "low": 3}
    category_distribution           JSONB           DEFAULT '{}',
        -- Distribution of categories: {"environmental": 8, "human_rights": 3, ...}
    trend_direction                 VARCHAR(20)     DEFAULT 'stable',
        -- Trend direction: improving, stable, worsening
    trend_confidence                NUMERIC(5,2)    DEFAULT 0,
        -- Confidence in trend assessment (0-100)
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_gmm_analytics_pattern CHECK (pattern_type IN (
        'recurring', 'clustered', 'systemic', 'isolated', 'escalating'
    )),
    CONSTRAINT chk_gmm_analytics_trend CHECK (trend_direction IN (
        'improving', 'stable', 'worsening'
    )),
    CONSTRAINT chk_gmm_analytics_period CHECK (analysis_period_end >= analysis_period_start),
    CONSTRAINT chk_gmm_analytics_stakeholders CHECK (affected_stakeholder_count >= 0),
    CONSTRAINT chk_gmm_analytics_trend_conf CHECK (trend_confidence >= 0 AND trend_confidence <= 100)
);

-- B-tree indexes for gl_eudr_gmm_analytics
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_ana_operator ON gl_eudr_gmm_analytics (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_ana_tenant ON gl_eudr_gmm_analytics (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_ana_pattern ON gl_eudr_gmm_analytics (pattern_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_ana_period_start ON gl_eudr_gmm_analytics (analysis_period_start);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_ana_period_end ON gl_eudr_gmm_analytics (analysis_period_end);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_ana_trend ON gl_eudr_gmm_analytics (trend_direction);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_ana_created ON gl_eudr_gmm_analytics (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_ana_provenance ON gl_eudr_gmm_analytics (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_ana_operator_pattern ON gl_eudr_gmm_analytics (operator_id, pattern_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_ana_operator_period ON gl_eudr_gmm_analytics (operator_id, analysis_period_start, analysis_period_end);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_ana_tenant_operator ON gl_eudr_gmm_analytics (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_ana_pattern_trend ON gl_eudr_gmm_analytics (pattern_type, trend_direction);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN index on JSONB
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_ana_grievance_ids ON gl_eudr_gmm_analytics USING GIN (grievance_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_ana_root_causes ON gl_eudr_gmm_analytics USING GIN (root_causes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_ana_severity_dist ON gl_eudr_gmm_analytics USING GIN (severity_distribution);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 2. gl_eudr_gmm_root_causes -- Root cause analysis records
-- ============================================================================
RAISE NOTICE 'V120 [2/8]: Creating gl_eudr_gmm_root_causes...';

CREATE TABLE IF NOT EXISTS gl_eudr_gmm_root_causes (
    root_cause_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique identifier for this root cause analysis
    grievance_id                    UUID            NOT NULL,
        -- FK reference to EUDR-031 grievance table (gl_eudr_set_grievances)
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator identifier
    tenant_id                       VARCHAR(100)    NOT NULL DEFAULT '',
        -- Multi-tenant isolation identifier
    analysis_method                 VARCHAR(30)     NOT NULL,
        -- Methodology used for root cause analysis
    primary_cause                   TEXT            NOT NULL DEFAULT '',
        -- Identified primary root cause
    contributing_factors            JSONB           DEFAULT '[]',
        -- Array of contributing factors: [{"factor": "...", "weight": 0.3}, ...]
    analysis_depth                  INTEGER         NOT NULL DEFAULT 1,
        -- Depth of the analysis (1-10, higher = more thorough)
    confidence_score                NUMERIC(5,2)    NOT NULL DEFAULT 0,
        -- Confidence in the analysis result (0-100)
    evidence                        JSONB           DEFAULT '[]',
        -- Supporting evidence: [{"type": "...", "reference": "...", "description": "..."}, ...]
    recommendations                 JSONB           DEFAULT '[]',
        -- Recommended corrective actions: [{"action": "...", "priority": "...", "timeline": "..."}, ...]
    causal_chain                    JSONB           DEFAULT '[]',
        -- Ordered causal chain steps: [{"step": 1, "description": "...", "type": "..."}, ...]
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_gmm_rc_method CHECK (analysis_method IN (
        'five_whys', 'fishbone', 'fault_tree', 'correlation'
    )),
    CONSTRAINT chk_gmm_rc_depth CHECK (analysis_depth >= 1 AND analysis_depth <= 10),
    CONSTRAINT chk_gmm_rc_confidence CHECK (confidence_score >= 0 AND confidence_score <= 100)
);

-- B-tree indexes for gl_eudr_gmm_root_causes
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rc_grievance ON gl_eudr_gmm_root_causes (grievance_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rc_operator ON gl_eudr_gmm_root_causes (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rc_tenant ON gl_eudr_gmm_root_causes (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rc_method ON gl_eudr_gmm_root_causes (analysis_method);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rc_confidence ON gl_eudr_gmm_root_causes (confidence_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rc_depth ON gl_eudr_gmm_root_causes (analysis_depth);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rc_provenance ON gl_eudr_gmm_root_causes (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rc_created ON gl_eudr_gmm_root_causes (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rc_updated ON gl_eudr_gmm_root_causes (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rc_grievance_method ON gl_eudr_gmm_root_causes (grievance_id, analysis_method);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rc_operator_method ON gl_eudr_gmm_root_causes (operator_id, analysis_method);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rc_operator_confidence ON gl_eudr_gmm_root_causes (operator_id, confidence_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rc_factors ON gl_eudr_gmm_root_causes USING GIN (contributing_factors);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rc_evidence ON gl_eudr_gmm_root_causes USING GIN (evidence);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rc_recommendations ON gl_eudr_gmm_root_causes USING GIN (recommendations);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 3. gl_eudr_gmm_mediations -- Multi-party mediation workflows
-- ============================================================================
RAISE NOTICE 'V120 [3/8]: Creating gl_eudr_gmm_mediations...';

CREATE TABLE IF NOT EXISTS gl_eudr_gmm_mediations (
    mediation_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique mediation workflow identifier
    grievance_id                    UUID            NOT NULL,
        -- FK reference to EUDR-031 grievance table
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator identifier
    tenant_id                       VARCHAR(100)    NOT NULL DEFAULT '',
        -- Multi-tenant isolation identifier
    mediation_stage                 VARCHAR(30)     NOT NULL DEFAULT 'initiated',
        -- Current mediation workflow stage
    parties                         JSONB           DEFAULT '[]',
        -- Array of involved parties: [{"role": "complainant", "name": "...", "id": "..."}, ...]
    mediator_id                     VARCHAR(100),
        -- Assigned mediator identifier
    mediator_type                   VARCHAR(30)     NOT NULL DEFAULT 'internal',
        -- Type of mediator assigned
    session_records                 JSONB           DEFAULT '[]',
        -- Array of session records: [{"date": "...", "duration_min": 60, "summary": "...", "attendees": [...]}, ...]
    agreements                      JSONB           DEFAULT '[]',
        -- Array of agreements reached: [{"clause": "...", "agreed_by": [...], "date": "..."}, ...]
    settlement_terms                JSONB           DEFAULT '{}',
        -- Final settlement terms: {"type": "...", "terms": [...], "conditions": [...]}
    settlement_status               VARCHAR(30)     DEFAULT 'pending',
        -- Status of settlement (pending, accepted, rejected, implemented)
    session_count                   INTEGER         DEFAULT 0,
        -- Total number of mediation sessions held
    total_duration_minutes          INTEGER         DEFAULT 0,
        -- Total mediation duration in minutes across all sessions
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    completed_at                    TIMESTAMPTZ,
        -- Timestamp when mediation was completed/closed

    CONSTRAINT chk_gmm_med_stage CHECK (mediation_stage IN (
        'initiated', 'preparation', 'dialogue', 'negotiation',
        'settlement', 'implementation', 'closed'
    )),
    CONSTRAINT chk_gmm_med_mediator_type CHECK (mediator_type IN (
        'internal', 'external', 'community_elder', 'legal'
    )),
    CONSTRAINT chk_gmm_med_settlement_status CHECK (settlement_status IN (
        'pending', 'accepted', 'rejected', 'implemented'
    )),
    CONSTRAINT chk_gmm_med_session_count CHECK (session_count >= 0),
    CONSTRAINT chk_gmm_med_duration CHECK (total_duration_minutes >= 0)
);

-- B-tree indexes for gl_eudr_gmm_mediations
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_med_grievance ON gl_eudr_gmm_mediations (grievance_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_med_operator ON gl_eudr_gmm_mediations (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_med_tenant ON gl_eudr_gmm_mediations (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_med_stage ON gl_eudr_gmm_mediations (mediation_stage);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_med_mediator ON gl_eudr_gmm_mediations (mediator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_med_mediator_type ON gl_eudr_gmm_mediations (mediator_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_med_settlement ON gl_eudr_gmm_mediations (settlement_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_med_provenance ON gl_eudr_gmm_mediations (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_med_created ON gl_eudr_gmm_mediations (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_med_completed ON gl_eudr_gmm_mediations (completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_med_updated ON gl_eudr_gmm_mediations (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_med_operator_stage ON gl_eudr_gmm_mediations (operator_id, mediation_stage);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_med_grievance_stage ON gl_eudr_gmm_mediations (grievance_id, mediation_stage);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_med_operator_settlement ON gl_eudr_gmm_mediations (operator_id, settlement_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_med_tenant_operator ON gl_eudr_gmm_mediations (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_med_parties ON gl_eudr_gmm_mediations USING GIN (parties);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_med_agreements ON gl_eudr_gmm_mediations USING GIN (agreements);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_med_sessions ON gl_eudr_gmm_mediations USING GIN (session_records);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 4. gl_eudr_gmm_remediations -- Remediation effectiveness tracking
-- ============================================================================
RAISE NOTICE 'V120 [4/8]: Creating gl_eudr_gmm_remediations...';

CREATE TABLE IF NOT EXISTS gl_eudr_gmm_remediations (
    remediation_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique remediation tracking identifier
    grievance_id                    UUID            NOT NULL,
        -- FK reference to EUDR-031 grievance table
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator identifier
    tenant_id                       VARCHAR(100)    NOT NULL DEFAULT '',
        -- Multi-tenant isolation identifier
    remediation_type                VARCHAR(30)     NOT NULL,
        -- Type of remediation action taken
    remediation_actions             JSONB           DEFAULT '[]',
        -- Array of actions with deadlines: [{"action": "...", "deadline": "...", "status": "..."}, ...]
    implementation_status           VARCHAR(30)     NOT NULL DEFAULT 'planned',
        -- Current implementation status
    completion_percentage           NUMERIC(5,2)    NOT NULL DEFAULT 0,
        -- Percentage of remediation completed (0-100)
    effectiveness_indicators        JSONB           DEFAULT '{}',
        -- Key performance indicators: {"recurrence_rate": 0.05, "satisfaction": 4.2, ...}
    stakeholder_satisfaction        NUMERIC(3,1)    DEFAULT NULL,
        -- Stakeholder satisfaction rating (1.0-5.0)
    cost_incurred                   NUMERIC(14,2)   DEFAULT 0,
        -- Cost incurred for remediation (in base currency)
    timeline_adherence              NUMERIC(5,2)    DEFAULT 100,
        -- Percentage of milestones met on time (0-100)
    verification_evidence           JSONB           DEFAULT '[]',
        -- Evidence of remediation effectiveness: [{"type": "...", "ref": "...", "date": "..."}, ...]
    lessons_learned                 TEXT            DEFAULT '',
        -- Documented lessons learned from this remediation
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    verified_at                     TIMESTAMPTZ,
        -- Timestamp of remediation effectiveness verification

    CONSTRAINT chk_gmm_rem_type CHECK (remediation_type IN (
        'compensation', 'process_change', 'relationship_repair',
        'policy_reform', 'infrastructure'
    )),
    CONSTRAINT chk_gmm_rem_status CHECK (implementation_status IN (
        'planned', 'in_progress', 'completed', 'verified', 'failed'
    )),
    CONSTRAINT chk_gmm_rem_completion CHECK (completion_percentage >= 0 AND completion_percentage <= 100),
    CONSTRAINT chk_gmm_rem_satisfaction CHECK (stakeholder_satisfaction IS NULL OR
        (stakeholder_satisfaction >= 1.0 AND stakeholder_satisfaction <= 5.0)),
    CONSTRAINT chk_gmm_rem_cost CHECK (cost_incurred >= 0),
    CONSTRAINT chk_gmm_rem_timeline CHECK (timeline_adherence >= 0 AND timeline_adherence <= 100)
);

-- B-tree indexes for gl_eudr_gmm_remediations
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rem_grievance ON gl_eudr_gmm_remediations (grievance_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rem_operator ON gl_eudr_gmm_remediations (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rem_tenant ON gl_eudr_gmm_remediations (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rem_type ON gl_eudr_gmm_remediations (remediation_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rem_status ON gl_eudr_gmm_remediations (implementation_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rem_completion ON gl_eudr_gmm_remediations (completion_percentage);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rem_satisfaction ON gl_eudr_gmm_remediations (stakeholder_satisfaction DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rem_provenance ON gl_eudr_gmm_remediations (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rem_created ON gl_eudr_gmm_remediations (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rem_verified ON gl_eudr_gmm_remediations (verified_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rem_updated ON gl_eudr_gmm_remediations (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rem_operator_type ON gl_eudr_gmm_remediations (operator_id, remediation_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rem_operator_status ON gl_eudr_gmm_remediations (operator_id, implementation_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rem_grievance_type ON gl_eudr_gmm_remediations (grievance_id, remediation_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rem_tenant_operator ON gl_eudr_gmm_remediations (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rem_actions ON gl_eudr_gmm_remediations USING GIN (remediation_actions);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rem_indicators ON gl_eudr_gmm_remediations USING GIN (effectiveness_indicators);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rem_evidence ON gl_eudr_gmm_remediations USING GIN (verification_evidence);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 5. gl_eudr_gmm_risk_scores -- Grievance risk scoring
-- ============================================================================
RAISE NOTICE 'V120 [5/8]: Creating gl_eudr_gmm_risk_scores...';

CREATE TABLE IF NOT EXISTS gl_eudr_gmm_risk_scores (
    risk_score_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique risk score identifier
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator identifier
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    scope                           VARCHAR(30)     NOT NULL,
        -- Scope of the risk score
    scope_identifier                VARCHAR(200)    NOT NULL,
        -- Identifier within scope (e.g., operator_id, supplier_id, commodity name)
    risk_score                      NUMERIC(5,2)    NOT NULL DEFAULT 0,
        -- Computed risk score (0-100)
    risk_level                      VARCHAR(20)     NOT NULL DEFAULT 'low',
        -- Risk level classification
    grievance_frequency             INTEGER         NOT NULL DEFAULT 0,
        -- Number of grievances within scoring window
    average_severity                NUMERIC(5,2)    DEFAULT 0,
        -- Average severity score (0-100) across grievances
    resolution_time_trend           VARCHAR(20)     DEFAULT 'stable',
        -- Trend in resolution times: improving, stable, worsening
    unresolved_count                INTEGER         NOT NULL DEFAULT 0,
        -- Count of unresolved grievances
    escalation_rate                 NUMERIC(5,2)    DEFAULT 0,
        -- Percentage of grievances that were escalated (0-100)
    prediction_confidence           NUMERIC(5,2)    DEFAULT 0,
        -- Confidence in the predictive risk score (0-100)
    score_factors                   JSONB           DEFAULT '{}',
        -- Factor breakdown: {"frequency_weight": 0.3, "severity_weight": 0.25, ...}
    historical_scores               JSONB           DEFAULT '[]',
        -- Previous risk scores: [{"date": "...", "score": 45.0, "level": "moderate"}, ...]
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_gmm_risk_scope CHECK (scope IN (
        'operator', 'supplier', 'commodity', 'region'
    )),
    CONSTRAINT chk_gmm_risk_level CHECK (risk_level IN (
        'negligible', 'low', 'moderate', 'high', 'critical'
    )),
    CONSTRAINT chk_gmm_risk_score CHECK (risk_score >= 0 AND risk_score <= 100),
    CONSTRAINT chk_gmm_risk_frequency CHECK (grievance_frequency >= 0),
    CONSTRAINT chk_gmm_risk_avg_severity CHECK (average_severity >= 0 AND average_severity <= 100),
    CONSTRAINT chk_gmm_risk_trend CHECK (resolution_time_trend IN (
        'improving', 'stable', 'worsening'
    )),
    CONSTRAINT chk_gmm_risk_unresolved CHECK (unresolved_count >= 0),
    CONSTRAINT chk_gmm_risk_escalation CHECK (escalation_rate >= 0 AND escalation_rate <= 100),
    CONSTRAINT chk_gmm_risk_prediction CHECK (prediction_confidence >= 0 AND prediction_confidence <= 100)
);

-- B-tree indexes for gl_eudr_gmm_risk_scores
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_risk_operator ON gl_eudr_gmm_risk_scores (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_risk_tenant ON gl_eudr_gmm_risk_scores (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_risk_scope ON gl_eudr_gmm_risk_scores (scope);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_risk_scope_id ON gl_eudr_gmm_risk_scores (scope_identifier);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_risk_score ON gl_eudr_gmm_risk_scores (risk_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_risk_level ON gl_eudr_gmm_risk_scores (risk_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_risk_frequency ON gl_eudr_gmm_risk_scores (grievance_frequency DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_risk_unresolved ON gl_eudr_gmm_risk_scores (unresolved_count DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_risk_provenance ON gl_eudr_gmm_risk_scores (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_risk_created ON gl_eudr_gmm_risk_scores (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_risk_operator_scope ON gl_eudr_gmm_risk_scores (operator_id, scope);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_risk_operator_level ON gl_eudr_gmm_risk_scores (operator_id, risk_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_risk_scope_level ON gl_eudr_gmm_risk_scores (scope, risk_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_risk_tenant_operator ON gl_eudr_gmm_risk_scores (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_risk_scope_id_created ON gl_eudr_gmm_risk_scores (scope_identifier, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_risk_factors ON gl_eudr_gmm_risk_scores USING GIN (score_factors);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_risk_historical ON gl_eudr_gmm_risk_scores USING GIN (historical_scores);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 6. gl_eudr_gmm_collective_grievances -- Collective/class-action grievances
-- ============================================================================
RAISE NOTICE 'V120 [6/8]: Creating gl_eudr_gmm_collective_grievances...';

CREATE TABLE IF NOT EXISTS gl_eudr_gmm_collective_grievances (
    collective_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique collective grievance identifier
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator identifier
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    title                           VARCHAR(500)    NOT NULL,
        -- Title of the collective grievance
    description                     TEXT            NOT NULL DEFAULT '',
        -- Detailed description of the collective complaint
    grievance_category              VARCHAR(30)     NOT NULL DEFAULT 'process',
        -- Category of the collective grievance
    lead_complainant_id             VARCHAR(100),
        -- Primary/lead complainant identifier
    affected_stakeholder_count      INTEGER         NOT NULL DEFAULT 1,
        -- Number of stakeholders affected
    individual_grievance_ids        JSONB           DEFAULT '[]',
        -- Array of EUDR-031 individual grievance IDs: ["uuid-1", "uuid-2"]
    collective_status               VARCHAR(30)     NOT NULL DEFAULT 'forming',
        -- Current lifecycle status
    spokesperson                    VARCHAR(200),
        -- Designated spokesperson for the group
    representative_body             VARCHAR(200),
        -- Representative organization or body
    collective_demands              JSONB           DEFAULT '[]',
        -- Array of demands: [{"demand": "...", "priority": "high", "negotiable": true}, ...]
    negotiation_status              VARCHAR(30)     DEFAULT 'not_started',
        -- Current negotiation status
    supply_chain_nodes              JSONB           DEFAULT '[]',
        -- Affected supply chain node IDs
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for record integrity verification
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    resolved_at                     TIMESTAMPTZ,
        -- Timestamp when the collective grievance was resolved

    CONSTRAINT chk_gmm_cg_category CHECK (grievance_category IN (
        'environmental', 'human_rights', 'labor', 'land_rights',
        'community_impact', 'process'
    )),
    CONSTRAINT chk_gmm_cg_status CHECK (collective_status IN (
        'forming', 'submitted', 'investigating', 'mediating',
        'resolved', 'closed'
    )),
    CONSTRAINT chk_gmm_cg_negotiation CHECK (negotiation_status IN (
        'not_started', 'in_progress', 'stalled', 'agreement_reached', 'failed'
    )),
    CONSTRAINT chk_gmm_cg_stakeholders CHECK (affected_stakeholder_count >= 1)
);

-- B-tree indexes for gl_eudr_gmm_collective_grievances
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_cg_operator ON gl_eudr_gmm_collective_grievances (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_cg_tenant ON gl_eudr_gmm_collective_grievances (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_cg_category ON gl_eudr_gmm_collective_grievances (grievance_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_cg_status ON gl_eudr_gmm_collective_grievances (collective_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_cg_lead ON gl_eudr_gmm_collective_grievances (lead_complainant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_cg_negotiation ON gl_eudr_gmm_collective_grievances (negotiation_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_cg_stakeholder_count ON gl_eudr_gmm_collective_grievances (affected_stakeholder_count DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_cg_provenance ON gl_eudr_gmm_collective_grievances (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_cg_created ON gl_eudr_gmm_collective_grievances (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_cg_resolved ON gl_eudr_gmm_collective_grievances (resolved_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_cg_updated ON gl_eudr_gmm_collective_grievances (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_cg_operator_status ON gl_eudr_gmm_collective_grievances (operator_id, collective_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_cg_operator_category ON gl_eudr_gmm_collective_grievances (operator_id, grievance_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_cg_tenant_operator ON gl_eudr_gmm_collective_grievances (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_cg_status_negotiation ON gl_eudr_gmm_collective_grievances (collective_status, negotiation_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_cg_individual_ids ON gl_eudr_gmm_collective_grievances USING GIN (individual_grievance_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_cg_demands ON gl_eudr_gmm_collective_grievances USING GIN (collective_demands);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_cg_supply_chain ON gl_eudr_gmm_collective_grievances USING GIN (supply_chain_nodes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 7. gl_eudr_gmm_regulatory_reports -- Generated compliance reports
-- ============================================================================
RAISE NOTICE 'V120 [7/8]: Creating gl_eudr_gmm_regulatory_reports...';

CREATE TABLE IF NOT EXISTS gl_eudr_gmm_regulatory_reports (
    report_id                       UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
        -- Unique report identifier
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator identifier
    tenant_id                       VARCHAR(100)    NOT NULL,
        -- Multi-tenant isolation identifier
    report_type                     VARCHAR(30)     NOT NULL,
        -- Type of regulatory report
    reporting_period_start          TIMESTAMPTZ     NOT NULL,
        -- Start of the reporting period
    reporting_period_end            TIMESTAMPTZ     NOT NULL,
        -- End of the reporting period
    total_grievances                INTEGER         NOT NULL DEFAULT 0,
        -- Total grievances in the reporting period
    resolved_count                  INTEGER         NOT NULL DEFAULT 0,
        -- Number of grievances resolved
    unresolved_count                INTEGER         NOT NULL DEFAULT 0,
        -- Number of grievances still unresolved
    average_resolution_days         NUMERIC(7,2)    DEFAULT 0,
        -- Average resolution time in days
    satisfaction_rating             NUMERIC(3,1)    DEFAULT NULL,
        -- Average complainant satisfaction (1.0-5.0)
    top_categories                  JSONB           DEFAULT '[]',
        -- Top grievance categories: [{"category": "...", "count": 10, "pct": 25.0}, ...]
    top_root_causes                 JSONB           DEFAULT '[]',
        -- Top root causes: [{"cause": "...", "frequency": 8, "pct": 20.0}, ...]
    remediation_effectiveness       NUMERIC(5,2)    DEFAULT 0,
        -- Overall remediation effectiveness score (0-100)
    accessibility_score             NUMERIC(5,2)    DEFAULT 0,
        -- Mechanism accessibility score (0-100)
    report_content                  JSONB           DEFAULT '{}',
        -- Full structured report content with sections
    report_file_reference           VARCHAR(500),
        -- S3/storage reference for the generated report document
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for record integrity verification
    generated_at                    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_gmm_rpt_type CHECK (report_type IN (
        'eudr_article16', 'csddd_article8', 'ungp_effectiveness', 'annual_summary'
    )),
    CONSTRAINT chk_gmm_rpt_period CHECK (reporting_period_end >= reporting_period_start),
    CONSTRAINT chk_gmm_rpt_totals CHECK (total_grievances >= 0),
    CONSTRAINT chk_gmm_rpt_resolved CHECK (resolved_count >= 0),
    CONSTRAINT chk_gmm_rpt_unresolved CHECK (unresolved_count >= 0),
    CONSTRAINT chk_gmm_rpt_avg_resolution CHECK (average_resolution_days >= 0),
    CONSTRAINT chk_gmm_rpt_satisfaction CHECK (satisfaction_rating IS NULL OR
        (satisfaction_rating >= 1.0 AND satisfaction_rating <= 5.0)),
    CONSTRAINT chk_gmm_rpt_effectiveness CHECK (remediation_effectiveness >= 0 AND remediation_effectiveness <= 100),
    CONSTRAINT chk_gmm_rpt_accessibility CHECK (accessibility_score >= 0 AND accessibility_score <= 100)
);

-- B-tree indexes for gl_eudr_gmm_regulatory_reports
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rpt_operator ON gl_eudr_gmm_regulatory_reports (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rpt_tenant ON gl_eudr_gmm_regulatory_reports (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rpt_type ON gl_eudr_gmm_regulatory_reports (report_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rpt_period_start ON gl_eudr_gmm_regulatory_reports (reporting_period_start);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rpt_period_end ON gl_eudr_gmm_regulatory_reports (reporting_period_end);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rpt_generated ON gl_eudr_gmm_regulatory_reports (generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rpt_provenance ON gl_eudr_gmm_regulatory_reports (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rpt_effectiveness ON gl_eudr_gmm_regulatory_reports (remediation_effectiveness DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rpt_operator_type ON gl_eudr_gmm_regulatory_reports (operator_id, report_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rpt_operator_period ON gl_eudr_gmm_regulatory_reports (operator_id, reporting_period_start, reporting_period_end);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rpt_tenant_operator ON gl_eudr_gmm_regulatory_reports (tenant_id, operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rpt_type_generated ON gl_eudr_gmm_regulatory_reports (report_type, generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rpt_categories ON gl_eudr_gmm_regulatory_reports USING GIN (top_categories);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rpt_root_causes ON gl_eudr_gmm_regulatory_reports USING GIN (top_root_causes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_rpt_content ON gl_eudr_gmm_regulatory_reports USING GIN (report_content);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 8. gl_eudr_gmm_audit_trail -- Hypertable for analytics operations
-- ============================================================================
RAISE NOTICE 'V120 [8/8]: Creating gl_eudr_gmm_audit_trail (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_gmm_audit_trail (
    audit_id                        UUID            DEFAULT gen_random_uuid(),
        -- Unique audit entry identifier
    entity_type                     VARCHAR(50)     NOT NULL,
        -- Type of entity being audited (analytics, root_cause, mediation, etc.)
    entity_id                       UUID            NOT NULL,
        -- Entity identifier being audited
    operator_id                     VARCHAR(100)    NOT NULL,
        -- EUDR operator identifier
    action                          VARCHAR(50)     NOT NULL,
        -- Action performed (create, update, advance_stage, close, etc.)
    actor_id                        VARCHAR(100)    NOT NULL DEFAULT 'system',
        -- Actor performing the action (user ID or system)
    changes                         JSONB           DEFAULT '{}',
        -- JSON diff of changes: {"field": {"old": "...", "new": "..."}}
    provenance_hash                 VARCHAR(64),
        -- SHA-256 hash for audit entry integrity
    timestamp                       TIMESTAMPTZ     NOT NULL DEFAULT NOW()
        -- Timestamp of the action (partitioning column)
);

-- Convert to hypertable
DO $$
BEGIN
    PERFORM create_hypertable(
        'gl_eudr_gmm_audit_trail',
        'timestamp',
        chunk_time_interval => INTERVAL '7 days',
        if_not_exists => TRUE
    );
    RAISE NOTICE 'gl_eudr_gmm_audit_trail hypertable created (7-day chunks)';
EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Hypertable creation skipped for gl_eudr_gmm_audit_trail: %', SQLERRM;
END $$;

-- B-tree indexes for gl_eudr_gmm_audit_trail
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_audit_entity_type ON gl_eudr_gmm_audit_trail (entity_type, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_audit_entity_id ON gl_eudr_gmm_audit_trail (entity_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_audit_operator ON gl_eudr_gmm_audit_trail (operator_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_audit_action ON gl_eudr_gmm_audit_trail (action, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_audit_actor ON gl_eudr_gmm_audit_trail (actor_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_audit_provenance ON gl_eudr_gmm_audit_trail (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Composite indexes
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_audit_entity_action ON gl_eudr_gmm_audit_trail (entity_type, action, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_audit_operator_entity ON gl_eudr_gmm_audit_trail (operator_id, entity_type, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- GIN index
DO $$ BEGIN
    CREATE INDEX idx_eudr_gmm_audit_changes ON gl_eudr_gmm_audit_trail USING GIN (changes);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- Triggers: updated_at auto-update
-- ============================================================================
RAISE NOTICE 'V120: Creating updated_at triggers...';

CREATE OR REPLACE FUNCTION fn_eudr_gmm_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_gmm_root_causes_updated_at
        BEFORE UPDATE ON gl_eudr_gmm_root_causes
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_gmm_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_gmm_mediations_updated_at
        BEFORE UPDATE ON gl_eudr_gmm_mediations
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_gmm_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_gmm_remediations_updated_at
        BEFORE UPDATE ON gl_eudr_gmm_remediations
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_gmm_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_gmm_collective_grievances_updated_at
        BEFORE UPDATE ON gl_eudr_gmm_collective_grievances
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_gmm_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- Triggers: Audit trail auto-insert
-- ============================================================================
RAISE NOTICE 'V120: Creating audit trail triggers...';

CREATE OR REPLACE FUNCTION fn_eudr_gmm_audit_insert()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO gl_eudr_gmm_audit_trail (entity_type, entity_id, operator_id, action, actor_id, changes, timestamp)
    VALUES (TG_ARGV[0], NEW.*::TEXT::UUID, COALESCE(NEW.operator_id, ''), 'create', 'system', row_to_json(NEW)::JSONB, NOW());
    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION fn_eudr_gmm_audit_update()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO gl_eudr_gmm_audit_trail (entity_type, entity_id, operator_id, action, actor_id, changes, timestamp)
    VALUES (TG_ARGV[0], NEW.*::TEXT::UUID, COALESCE(NEW.operator_id, ''), 'update', 'system', jsonb_build_object('new', row_to_json(NEW)::JSONB), NOW());
    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Analytics audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_gmm_analytics_audit_insert
        AFTER INSERT ON gl_eudr_gmm_analytics
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_gmm_audit_insert('analytics');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Root causes audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_gmm_root_causes_audit_insert
        AFTER INSERT ON gl_eudr_gmm_root_causes
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_gmm_audit_insert('root_cause');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_gmm_root_causes_audit_update
        AFTER UPDATE ON gl_eudr_gmm_root_causes
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_gmm_audit_update('root_cause');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Mediations audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_gmm_mediations_audit_insert
        AFTER INSERT ON gl_eudr_gmm_mediations
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_gmm_audit_insert('mediation');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_gmm_mediations_audit_update
        AFTER UPDATE ON gl_eudr_gmm_mediations
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_gmm_audit_update('mediation');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Remediations audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_gmm_remediations_audit_insert
        AFTER INSERT ON gl_eudr_gmm_remediations
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_gmm_audit_insert('remediation');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_gmm_remediations_audit_update
        AFTER UPDATE ON gl_eudr_gmm_remediations
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_gmm_audit_update('remediation');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Collective grievances audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_gmm_collective_audit_insert
        AFTER INSERT ON gl_eudr_gmm_collective_grievances
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_gmm_audit_insert('collective_grievance');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TRIGGER trg_eudr_gmm_collective_audit_update
        AFTER UPDATE ON gl_eudr_gmm_collective_grievances
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_gmm_audit_update('collective_grievance');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Regulatory reports audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_gmm_regulatory_reports_audit_insert
        AFTER INSERT ON gl_eudr_gmm_regulatory_reports
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_gmm_audit_insert('regulatory_report');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Risk scores audit triggers
DO $$ BEGIN
    CREATE TRIGGER trg_eudr_gmm_risk_scores_audit_insert
        AFTER INSERT ON gl_eudr_gmm_risk_scores
        FOR EACH ROW EXECUTE FUNCTION fn_eudr_gmm_audit_insert('risk_score');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- Completion
-- ============================================================================

RAISE NOTICE 'V120: AGENT-EUDR-032 Grievance Mechanism Manager tables created successfully';
RAISE NOTICE 'V120: Tables: 8 (analytics, root_causes, mediations, remediations, risk_scores, collective_grievances, regulatory_reports, audit_trail)';
RAISE NOTICE 'V120: Hypertable: gl_eudr_gmm_audit_trail (7-day chunks)';
RAISE NOTICE 'V120: Indexes: ~120';
RAISE NOTICE 'V120: Triggers: 4 updated_at + 11 audit trail';

COMMIT;
