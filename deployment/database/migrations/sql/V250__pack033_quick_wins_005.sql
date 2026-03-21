-- =============================================================================
-- V250: PACK-033 Quick Wins Identifier - Prioritization
-- =============================================================================
-- Pack:         PACK-033 (Quick Wins Identifier Pack)
-- Migration:    005 of 010
-- Date:         March 2026
--
-- Creates prioritization tables for multi-criteria scoring and optimal
-- implementation sequencing of quick-win actions. Supports weighted scoring,
-- Pareto-optimal identification, and dependency-aware sequencing.
--
-- Tables (2):
--   1. pack033_quick_wins.priority_scores
--   2. pack033_quick_wins.implementation_sequences
--
-- Previous: V249__pack033_quick_wins_004.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack033_quick_wins.priority_scores
-- =============================================================================
-- Multi-criteria priority scoring for each action with individual dimension
-- scores, weighted totals, ranking, and Pareto optimality flagging.

CREATE TABLE pack033_quick_wins.priority_scores (
    score_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id                 UUID            NOT NULL REFERENCES pack033_quick_wins.quick_wins_scans(scan_id) ON DELETE CASCADE,
    action_id               UUID            NOT NULL,
    cost_score              NUMERIC(6,2)    NOT NULL DEFAULT 0,
    savings_score           NUMERIC(6,2)    NOT NULL DEFAULT 0,
    risk_score              NUMERIC(6,2)    NOT NULL DEFAULT 0,
    disruption_score        NUMERIC(6,2)    NOT NULL DEFAULT 0,
    complexity_score        NUMERIC(6,2)    NOT NULL DEFAULT 0,
    co_benefits_score       NUMERIC(6,2)    NOT NULL DEFAULT 0,
    weighted_total          NUMERIC(8,2)    NOT NULL DEFAULT 0,
    rank                    INTEGER,
    pareto_optimal          BOOLEAN         DEFAULT FALSE,
    weight_profile          JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p033_ps_cost CHECK (
        cost_score >= 0 AND cost_score <= 100
    ),
    CONSTRAINT chk_p033_ps_savings CHECK (
        savings_score >= 0 AND savings_score <= 100
    ),
    CONSTRAINT chk_p033_ps_risk CHECK (
        risk_score >= 0 AND risk_score <= 100
    ),
    CONSTRAINT chk_p033_ps_disruption CHECK (
        disruption_score >= 0 AND disruption_score <= 100
    ),
    CONSTRAINT chk_p033_ps_complexity CHECK (
        complexity_score >= 0 AND complexity_score <= 100
    ),
    CONSTRAINT chk_p033_ps_co_benefits CHECK (
        co_benefits_score >= 0 AND co_benefits_score <= 100
    ),
    CONSTRAINT chk_p033_ps_weighted CHECK (
        weighted_total >= 0
    ),
    CONSTRAINT chk_p033_ps_rank CHECK (
        rank IS NULL OR rank >= 1
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_ps_scan          ON pack033_quick_wins.priority_scores(scan_id);
CREATE INDEX idx_p033_ps_action        ON pack033_quick_wins.priority_scores(action_id);
CREATE INDEX idx_p033_ps_weighted      ON pack033_quick_wins.priority_scores(weighted_total DESC);
CREATE INDEX idx_p033_ps_rank          ON pack033_quick_wins.priority_scores(rank);
CREATE INDEX idx_p033_ps_pareto        ON pack033_quick_wins.priority_scores(pareto_optimal);
CREATE INDEX idx_p033_ps_created       ON pack033_quick_wins.priority_scores(created_at DESC);
CREATE UNIQUE INDEX idx_p033_ps_scan_action
    ON pack033_quick_wins.priority_scores(scan_id, action_id);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p033_ps_updated
    BEFORE UPDATE ON pack033_quick_wins.priority_scores
    FOR EACH ROW EXECUTE FUNCTION pack033_quick_wins.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack033_quick_wins.implementation_sequences
-- =============================================================================
-- Optimal implementation sequences considering dependencies, resource
-- constraints, and cumulative financial/carbon impacts.

CREATE TABLE pack033_quick_wins.implementation_sequences (
    sequence_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id                 UUID            NOT NULL REFERENCES pack033_quick_wins.quick_wins_scans(scan_id) ON DELETE CASCADE,
    sequence_name           VARCHAR(255)    NOT NULL,
    actions                 JSONB           NOT NULL DEFAULT '[]',
    total_cost              NUMERIC(16,2),
    total_savings           NUMERIC(16,2),
    total_co2e              NUMERIC(14,4),
    dependency_graph        JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p033_is_total_cost CHECK (
        total_cost IS NULL OR total_cost >= 0
    ),
    CONSTRAINT chk_p033_is_total_savings CHECK (
        total_savings IS NULL OR total_savings >= 0
    ),
    CONSTRAINT chk_p033_is_total_co2e CHECK (
        total_co2e IS NULL OR total_co2e >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_is_scan          ON pack033_quick_wins.implementation_sequences(scan_id);
CREATE INDEX idx_p033_is_total_savings ON pack033_quick_wins.implementation_sequences(total_savings DESC);
CREATE INDEX idx_p033_is_total_cost    ON pack033_quick_wins.implementation_sequences(total_cost);
CREATE INDEX idx_p033_is_actions       ON pack033_quick_wins.implementation_sequences USING GIN(actions);
CREATE INDEX idx_p033_is_deps          ON pack033_quick_wins.implementation_sequences USING GIN(dependency_graph);
CREATE INDEX idx_p033_is_created       ON pack033_quick_wins.implementation_sequences(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p033_is_updated
    BEFORE UPDATE ON pack033_quick_wins.implementation_sequences
    FOR EACH ROW EXECUTE FUNCTION pack033_quick_wins.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack033_quick_wins.priority_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack033_quick_wins.implementation_sequences ENABLE ROW LEVEL SECURITY;

CREATE POLICY p033_ps_tenant_isolation
    ON pack033_quick_wins.priority_scores
    USING (scan_id IN (
        SELECT scan_id FROM pack033_quick_wins.quick_wins_scans
        WHERE tenant_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p033_ps_service_bypass
    ON pack033_quick_wins.priority_scores
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p033_is_tenant_isolation
    ON pack033_quick_wins.implementation_sequences
    USING (scan_id IN (
        SELECT scan_id FROM pack033_quick_wins.quick_wins_scans
        WHERE tenant_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p033_is_service_bypass
    ON pack033_quick_wins.implementation_sequences
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.priority_scores TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.implementation_sequences TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack033_quick_wins.priority_scores IS
    'Multi-criteria priority scoring for actions with weighted totals, ranking, and Pareto optimality identification.';

COMMENT ON TABLE pack033_quick_wins.implementation_sequences IS
    'Optimal implementation sequences considering dependencies, resource constraints, and cumulative impacts.';

COMMENT ON COLUMN pack033_quick_wins.priority_scores.cost_score IS
    'Score for implementation cost (0-100, higher = lower cost = better).';
COMMENT ON COLUMN pack033_quick_wins.priority_scores.savings_score IS
    'Score for energy savings potential (0-100, higher = more savings).';
COMMENT ON COLUMN pack033_quick_wins.priority_scores.risk_score IS
    'Score for implementation risk (0-100, higher = lower risk = better).';
COMMENT ON COLUMN pack033_quick_wins.priority_scores.pareto_optimal IS
    'Whether this action is on the Pareto frontier (no other action dominates on all criteria).';
COMMENT ON COLUMN pack033_quick_wins.priority_scores.weight_profile IS
    'JSONB weight profile used for scoring (e.g., {"cost": 0.25, "savings": 0.30, "risk": 0.15, ...}).';
COMMENT ON COLUMN pack033_quick_wins.priority_scores.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack033_quick_wins.implementation_sequences.actions IS
    'Ordered JSONB array of action_ids in the implementation sequence.';
COMMENT ON COLUMN pack033_quick_wins.implementation_sequences.dependency_graph IS
    'JSONB adjacency list of action dependencies (e.g., {"action_a": ["action_b", "action_c"]}).';
