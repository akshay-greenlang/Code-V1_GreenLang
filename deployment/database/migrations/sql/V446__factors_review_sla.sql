-- V446__factors_review_sla.sql
-- GAP-15: Approval SLA enforcement for the Factors review workflow.
-- Backs greenlang/factors/quality/sla.py (SLAPolicy + SLATimer).
--
-- Default policies (hours per stage):
--   community        initial=72  detailed=168 final=168  deprecation=720
--   pro              initial=48  detailed=120 final=120  deprecation=360
--   enterprise       initial=24  detailed=72  final=72   deprecation=240
--   enterprise_cbam  initial=48  detailed=96  final=48   deprecation=168

BEGIN;

-- -----------------------------------------------------------------------
-- Policies: one row per (stage, tier).
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS factors_sla_policies (
    policy_id               UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    stage                   TEXT          NOT NULL,
    tier                    TEXT          NOT NULL,
    duration_hours          INTEGER       NOT NULL,
    warning_at_pct          REAL          NOT NULL DEFAULT 0.75,
    escalation_level        INTEGER       NOT NULL DEFAULT 1,
    auto_reject_after_hours INTEGER,
    active                  BOOLEAN       NOT NULL DEFAULT TRUE,
    created_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sla_stage CHECK (
        stage IN (
            'initial_review', 'detailed_review',
            'final_approval', 'deprecation_notice'
        )
    ),
    CONSTRAINT chk_sla_duration CHECK (duration_hours > 0),
    CONSTRAINT chk_sla_warning_pct CHECK (
        warning_at_pct >= 0.0 AND warning_at_pct <= 1.0
    ),
    CONSTRAINT chk_sla_escalation CHECK (escalation_level BETWEEN 1 AND 4),
    UNIQUE (stage, tier)
);

COMMENT ON TABLE factors_sla_policies IS
    'Tier + stage SLA durations for the Factors review workflow (GAP-15).';

-- -----------------------------------------------------------------------
-- Timers: one row per factor/stage activation.
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS factors_sla_timers (
    timer_id                UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    factor_id               TEXT          NOT NULL,
    stage                   TEXT          NOT NULL,
    tier                    TEXT,
    started_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    deadline                TIMESTAMPTZ   NOT NULL,
    warning_at              TIMESTAMPTZ   NOT NULL,
    status                  TEXT          NOT NULL,
    escalation_history      JSONB         NOT NULL DEFAULT '[]'::jsonb,
    auto_reject_after_hours INTEGER,
    completed_at            TIMESTAMPTZ,
    CONSTRAINT chk_timer_stage CHECK (
        stage IN (
            'initial_review', 'detailed_review',
            'final_approval', 'deprecation_notice'
        )
    ),
    CONSTRAINT chk_timer_status CHECK (
        status IN ('ACTIVE', 'WARNED', 'ESCALATED', 'EXPIRED', 'COMPLETED')
    )
);

CREATE INDEX IF NOT EXISTS idx_sla_timer_factor
    ON factors_sla_timers (factor_id);
CREATE INDEX IF NOT EXISTS idx_sla_timer_deadline_status
    ON factors_sla_timers (deadline, status);
CREATE INDEX IF NOT EXISTS idx_sla_timer_factor_stage_status
    ON factors_sla_timers (factor_id, stage, status);
CREATE INDEX IF NOT EXISTS idx_sla_timer_active_deadline
    ON factors_sla_timers (deadline)
    WHERE status IN ('ACTIVE', 'WARNED', 'ESCALATED');

COMMENT ON TABLE factors_sla_timers IS
    'Active SLA timers feeding escalation & auto-reject logic (GAP-15).';

-- -----------------------------------------------------------------------
-- Seed default SLA policies.
-- -----------------------------------------------------------------------
INSERT INTO factors_sla_policies (
    stage, tier, duration_hours, warning_at_pct, escalation_level,
    auto_reject_after_hours
) VALUES
    ('initial_review',     'community', 72,  0.75, 1, NULL),
    ('detailed_review',    'community', 168, 0.75, 1, NULL),
    ('final_approval',     'community', 168, 0.75, 2, NULL),
    ('deprecation_notice', 'community', 720, 0.75, 1, NULL),
    ('initial_review',     'pro',       48,  0.75, 1, NULL),
    ('detailed_review',    'pro',       120, 0.75, 2, NULL),
    ('final_approval',     'pro',       120, 0.75, 2, NULL),
    ('deprecation_notice', 'pro',       360, 0.75, 2, NULL),
    ('initial_review',     'enterprise', 24, 0.75, 2, NULL),
    ('detailed_review',    'enterprise', 72, 0.75, 2, NULL),
    ('final_approval',     'enterprise', 72, 0.75, 3, NULL),
    ('deprecation_notice', 'enterprise', 240, 0.75, 2, NULL),
    ('initial_review',     'enterprise_cbam', 48, 0.75, 2, 240),
    ('detailed_review',    'enterprise_cbam', 96, 0.75, 3, 336),
    ('final_approval',     'enterprise_cbam', 48, 0.75, 3, 168),
    ('deprecation_notice', 'enterprise_cbam', 168, 0.75, 3, NULL)
ON CONFLICT (stage, tier) DO NOTHING;

COMMIT;
