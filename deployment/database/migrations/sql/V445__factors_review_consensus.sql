-- V445__factors_review_consensus.sql
-- GAP-14: Multi-reviewer consensus engine for the Factors review workflow.
-- Backs greenlang/factors/quality/consensus.py (ConsensusConfig + ReviewerVote).
--
-- Consensus rules supported:
--   any_of_n    -- one approver suffices (Draft -> Under Review)
--   n_of_m      -- e.g. 2-of-3 methodology leads (Pro tier default)
--   unanimous   -- every named reviewer must approve
--   weighted    -- role-weighted voting (Enterprise default)
--
-- Tier defaults are seeded at the bottom of this file.

BEGIN;

-- -----------------------------------------------------------------------
-- Configuration: one row per (factor_type, tier) decision policy.
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS factors_review_consensus_configs (
    config_id                 UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    factor_type               TEXT          NOT NULL DEFAULT '',
    tier                      TEXT          NOT NULL,
    rule                      TEXT          NOT NULL,
    reviewer_requirements_json JSONB        NOT NULL,
    quorum                    INTEGER       NOT NULL,
    allow_self_approval       BOOLEAN       NOT NULL DEFAULT FALSE,
    dissent_capture_required  BOOLEAN       NOT NULL DEFAULT TRUE,
    sla_hours                 INTEGER,
    active                    BOOLEAN       NOT NULL DEFAULT TRUE,
    created_at                TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_consensus_rule CHECK (
        rule IN ('any_of_n', 'n_of_m', 'unanimous', 'weighted')
    ),
    CONSTRAINT chk_consensus_quorum CHECK (quorum >= 0)
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_consensus_cfg_factor_tier
    ON factors_review_consensus_configs (factor_type, tier)
    WHERE active = TRUE;

COMMENT ON TABLE factors_review_consensus_configs IS
    'Per (factor_type, tier) consensus configuration (GAP-14). '
    'See greenlang.factors.quality.consensus.tier_based_requirements.';

-- -----------------------------------------------------------------------
-- Votes: one row per (factor_id, reviewer_id).  Upsert on re-vote.
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS factors_review_votes (
    vote_id         UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    factor_id       TEXT          NOT NULL,
    reviewer_id     TEXT          NOT NULL,
    reviewer_role   TEXT          NOT NULL,
    decision        TEXT          NOT NULL,
    rationale       TEXT,
    dissent_notes   TEXT,
    weight          INTEGER       NOT NULL DEFAULT 1,
    voted_at        TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_vote_decision CHECK (
        decision IN ('APPROVE', 'REJECT', 'ABSTAIN')
    ),
    CONSTRAINT chk_vote_weight CHECK (weight >= 1)
);

CREATE INDEX IF NOT EXISTS idx_votes_factor
    ON factors_review_votes (factor_id);
CREATE INDEX IF NOT EXISTS idx_votes_reviewer
    ON factors_review_votes (reviewer_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_votes_factor_reviewer
    ON factors_review_votes (factor_id, reviewer_id);
CREATE INDEX IF NOT EXISTS idx_votes_decision_role
    ON factors_review_votes (decision, reviewer_role);

COMMENT ON TABLE factors_review_votes IS
    'Reviewer votes feeding the multi-reviewer consensus engine (GAP-14). '
    'A reviewer may only have one active vote per factor; re-voting upserts.';

-- -----------------------------------------------------------------------
-- Seed default tier configurations.
-- -----------------------------------------------------------------------
INSERT INTO factors_review_consensus_configs (
    factor_type, tier, rule, reviewer_requirements_json,
    quorum, allow_self_approval, dissent_capture_required, sla_hours
) VALUES
    ('', 'community', 'any_of_n',
     '[{"role": "methodology_lead", "min_count": 1, "weight": 1}]'::jsonb,
     1, FALSE, TRUE, 72),
    ('', 'pro', 'n_of_m',
     '[{"role": "methodology_lead", "min_count": 2, "weight": 1}]'::jsonb,
     2, FALSE, TRUE, 48),
    ('', 'enterprise', 'weighted',
     '[{"role": "methodology_lead", "min_count": 1, "weight": 2},
       {"role": "qa_lead", "min_count": 1, "weight": 1},
       {"role": "legal_lead", "min_count": 1, "weight": 1}]'::jsonb,
     3, FALSE, TRUE, 72),
    ('cbam', 'enterprise', 'weighted',
     '[{"role": "methodology_lead", "min_count": 1, "weight": 2},
       {"role": "qa_lead", "min_count": 1, "weight": 1},
       {"role": "legal_lead", "min_count": 1, "weight": 1},
       {"role": "compliance_lead", "min_count": 1, "weight": 2}]'::jsonb,
     4, FALSE, TRUE, 48),
    ('eudr', 'enterprise', 'weighted',
     '[{"role": "methodology_lead", "min_count": 1, "weight": 2},
       {"role": "qa_lead", "min_count": 1, "weight": 1},
       {"role": "legal_lead", "min_count": 1, "weight": 1},
       {"role": "compliance_lead", "min_count": 1, "weight": 2}]'::jsonb,
     4, FALSE, TRUE, 48),
    ('csrd', 'enterprise', 'weighted',
     '[{"role": "methodology_lead", "min_count": 1, "weight": 2},
       {"role": "qa_lead", "min_count": 1, "weight": 1},
       {"role": "legal_lead", "min_count": 1, "weight": 1},
       {"role": "compliance_lead", "min_count": 1, "weight": 2}]'::jsonb,
     4, FALSE, TRUE, 48)
ON CONFLICT DO NOTHING;

COMMIT;
