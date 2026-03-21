-- =============================================================================
-- V261: PACK-034 ISO 50001 Energy Management System - Action Plan Tables
-- =============================================================================
-- Pack:         PACK-034 (ISO 50001 Energy Management System Pack)
-- Migration:    006 of 010
-- Date:         March 2026
--
-- Creates energy objectives, targets, action plans, and action items tables
-- per ISO 50001 Clause 6.2. Implements the hierarchical structure:
-- Objective -> Target -> Action Plan -> Action Items.
--
-- Tables (4):
--   1. pack034_iso50001.energy_objectives
--   2. pack034_iso50001.energy_targets
--   3. pack034_iso50001.action_plans
--   4. pack034_iso50001.action_items
--
-- Previous: V260__pack034_iso50001_005.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack034_iso50001.energy_objectives
-- =============================================================================
-- Energy objectives per ISO 50001 Clause 6.2. High-level statements of
-- intended energy performance outcomes consistent with the energy policy.

CREATE TABLE pack034_iso50001.energy_objectives (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    enms_id                     UUID            NOT NULL REFERENCES pack034_iso50001.energy_management_systems(id) ON DELETE CASCADE,
    objective_text              TEXT            NOT NULL,
    objective_type              VARCHAR(30)     NOT NULL,
    target_year                 INTEGER,
    is_measurable               BOOLEAN         NOT NULL DEFAULT TRUE,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'active',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_obj_type CHECK (
        objective_type IN ('reduction', 'efficiency', 'renewable', 'awareness')
    ),
    CONSTRAINT chk_p034_obj_status CHECK (
        status IN ('active', 'completed', 'cancelled')
    ),
    CONSTRAINT chk_p034_obj_year CHECK (
        target_year IS NULL OR (target_year >= 2020 AND target_year <= 2100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_obj_enms           ON pack034_iso50001.energy_objectives(enms_id);
CREATE INDEX idx_p034_obj_type           ON pack034_iso50001.energy_objectives(objective_type);
CREATE INDEX idx_p034_obj_status         ON pack034_iso50001.energy_objectives(status);
CREATE INDEX idx_p034_obj_year           ON pack034_iso50001.energy_objectives(target_year);
CREATE INDEX idx_p034_obj_created        ON pack034_iso50001.energy_objectives(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_obj_updated
    BEFORE UPDATE ON pack034_iso50001.energy_objectives
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack034_iso50001.energy_targets
-- =============================================================================
-- Quantifiable energy targets linked to objectives, with measurable values,
-- baseline references, and achievement tracking.

CREATE TABLE pack034_iso50001.energy_targets (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    objective_id                UUID            NOT NULL REFERENCES pack034_iso50001.energy_objectives(id) ON DELETE CASCADE,
    target_description          TEXT            NOT NULL,
    target_value                DECIMAL(18,6)   NOT NULL,
    target_unit                 VARCHAR(50)     NOT NULL,
    baseline_value              DECIMAL(18,6),
    target_date                 DATE            NOT NULL,
    achievement_pct             DECIMAL(8,4)    DEFAULT 0,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_tgt_achievement CHECK (
        achievement_pct IS NULL OR (achievement_pct >= 0 AND achievement_pct <= 200)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_tgt_objective      ON pack034_iso50001.energy_targets(objective_id);
CREATE INDEX idx_p034_tgt_date           ON pack034_iso50001.energy_targets(target_date);
CREATE INDEX idx_p034_tgt_achievement    ON pack034_iso50001.energy_targets(achievement_pct DESC);
CREATE INDEX idx_p034_tgt_created        ON pack034_iso50001.energy_targets(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_tgt_updated
    BEFORE UPDATE ON pack034_iso50001.energy_targets
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack034_iso50001.action_plans
-- =============================================================================
-- Action plans linked to energy targets, detailing what will be done, by whom,
-- resource requirements, expected savings, timelines, and verification methods.

CREATE TABLE pack034_iso50001.action_plans (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    target_id                   UUID            NOT NULL REFERENCES pack034_iso50001.energy_targets(id) ON DELETE CASCADE,
    plan_name                   VARCHAR(500)    NOT NULL,
    responsible_person          VARCHAR(255)    NOT NULL,
    department                  VARCHAR(255),
    resources_required          TEXT,
    estimated_cost              DECIMAL(14,2),
    estimated_savings_kwh       DECIMAL(18,4),
    estimated_savings_cost      DECIMAL(14,2),
    start_date                  DATE,
    end_date                    DATE,
    verification_method         TEXT,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'planned',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_ap_status CHECK (
        status IN ('planned', 'in_progress', 'completed', 'cancelled')
    ),
    CONSTRAINT chk_p034_ap_dates CHECK (
        start_date IS NULL OR end_date IS NULL OR start_date <= end_date
    ),
    CONSTRAINT chk_p034_ap_cost CHECK (
        estimated_cost IS NULL OR estimated_cost >= 0
    ),
    CONSTRAINT chk_p034_ap_savings_kwh CHECK (
        estimated_savings_kwh IS NULL OR estimated_savings_kwh >= 0
    ),
    CONSTRAINT chk_p034_ap_savings_cost CHECK (
        estimated_savings_cost IS NULL OR estimated_savings_cost >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_ap_target          ON pack034_iso50001.action_plans(target_id);
CREATE INDEX idx_p034_ap_responsible     ON pack034_iso50001.action_plans(responsible_person);
CREATE INDEX idx_p034_ap_department      ON pack034_iso50001.action_plans(department);
CREATE INDEX idx_p034_ap_status          ON pack034_iso50001.action_plans(status);
CREATE INDEX idx_p034_ap_start           ON pack034_iso50001.action_plans(start_date);
CREATE INDEX idx_p034_ap_end             ON pack034_iso50001.action_plans(end_date);
CREATE INDEX idx_p034_ap_savings         ON pack034_iso50001.action_plans(estimated_savings_kwh DESC);
CREATE INDEX idx_p034_ap_created         ON pack034_iso50001.action_plans(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_ap_updated
    BEFORE UPDATE ON pack034_iso50001.action_plans
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack034_iso50001.action_items
-- =============================================================================
-- Granular action items within an action plan, assigned to individuals
-- with due dates and completion tracking.

CREATE TABLE pack034_iso50001.action_items (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id                     UUID            NOT NULL REFERENCES pack034_iso50001.action_plans(id) ON DELETE CASCADE,
    item_description            TEXT            NOT NULL,
    assigned_to                 VARCHAR(255),
    due_date                    DATE,
    completed_date              DATE,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'pending',
    notes                       TEXT,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_ai_status CHECK (
        status IN ('pending', 'in_progress', 'completed', 'cancelled', 'overdue')
    ),
    CONSTRAINT chk_p034_ai_completed CHECK (
        completed_date IS NULL OR due_date IS NULL OR completed_date >= due_date - INTERVAL '365 days'
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_ai_plan            ON pack034_iso50001.action_items(plan_id);
CREATE INDEX idx_p034_ai_assigned        ON pack034_iso50001.action_items(assigned_to);
CREATE INDEX idx_p034_ai_due             ON pack034_iso50001.action_items(due_date);
CREATE INDEX idx_p034_ai_status          ON pack034_iso50001.action_items(status);
CREATE INDEX idx_p034_ai_created         ON pack034_iso50001.action_items(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_ai_updated
    BEFORE UPDATE ON pack034_iso50001.action_items
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack034_iso50001.energy_objectives ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.energy_targets ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.action_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.action_items ENABLE ROW LEVEL SECURITY;

CREATE POLICY p034_obj_tenant_isolation
    ON pack034_iso50001.energy_objectives
    USING (enms_id IN (
        SELECT id FROM pack034_iso50001.energy_management_systems
        WHERE organization_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p034_obj_service_bypass
    ON pack034_iso50001.energy_objectives
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_tgt_tenant_isolation
    ON pack034_iso50001.energy_targets
    USING (objective_id IN (
        SELECT id FROM pack034_iso50001.energy_objectives
        WHERE enms_id IN (
            SELECT id FROM pack034_iso50001.energy_management_systems
            WHERE organization_id = current_setting('app.current_tenant')::UUID
        )
    ));
CREATE POLICY p034_tgt_service_bypass
    ON pack034_iso50001.energy_targets
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_ap_tenant_isolation
    ON pack034_iso50001.action_plans
    USING (target_id IN (
        SELECT id FROM pack034_iso50001.energy_targets
        WHERE objective_id IN (
            SELECT id FROM pack034_iso50001.energy_objectives
            WHERE enms_id IN (
                SELECT id FROM pack034_iso50001.energy_management_systems
                WHERE organization_id = current_setting('app.current_tenant')::UUID
            )
        )
    ));
CREATE POLICY p034_ap_service_bypass
    ON pack034_iso50001.action_plans
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_ai_tenant_isolation
    ON pack034_iso50001.action_items
    USING (plan_id IN (
        SELECT id FROM pack034_iso50001.action_plans
        WHERE target_id IN (
            SELECT id FROM pack034_iso50001.energy_targets
            WHERE objective_id IN (
                SELECT id FROM pack034_iso50001.energy_objectives
                WHERE enms_id IN (
                    SELECT id FROM pack034_iso50001.energy_management_systems
                    WHERE organization_id = current_setting('app.current_tenant')::UUID
                )
            )
        )
    ));
CREATE POLICY p034_ai_service_bypass
    ON pack034_iso50001.action_items
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.energy_objectives TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.energy_targets TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.action_plans TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.action_items TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack034_iso50001.energy_objectives IS
    'Energy objectives per ISO 50001 Clause 6.2 - high-level statements of intended energy performance outcomes.';

COMMENT ON TABLE pack034_iso50001.energy_targets IS
    'Quantifiable energy targets linked to objectives with measurable values and achievement tracking.';

COMMENT ON TABLE pack034_iso50001.action_plans IS
    'Action plans detailing what will be done, by whom, resources, expected savings, and verification methods.';

COMMENT ON TABLE pack034_iso50001.action_items IS
    'Granular action items within action plans with individual assignments, due dates, and completion tracking.';

COMMENT ON COLUMN pack034_iso50001.energy_objectives.objective_type IS
    'Type of objective: reduction (absolute), efficiency (intensity), renewable (energy source), awareness (behavioral).';
COMMENT ON COLUMN pack034_iso50001.energy_objectives.is_measurable IS
    'Whether the objective has quantifiable success criteria as required by ISO 50001.';
COMMENT ON COLUMN pack034_iso50001.energy_targets.achievement_pct IS
    'Percentage of target achieved (0-200%, allowing for over-achievement).';
COMMENT ON COLUMN pack034_iso50001.action_plans.verification_method IS
    'Method to verify that the action plan achieved its intended energy savings.';
COMMENT ON COLUMN pack034_iso50001.action_plans.estimated_savings_kwh IS
    'Estimated annual energy savings in kWh from implementing this action plan.';
COMMENT ON COLUMN pack034_iso50001.action_items.status IS
    'Item status: pending, in_progress, completed, cancelled, or overdue.';
