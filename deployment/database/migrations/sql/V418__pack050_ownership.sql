-- =============================================================================
-- PACK-050 GHG Consolidation Pack
-- Migration: V418 - Ownership
-- =============================================================================
-- Pack:         PACK-050 (GHG Consolidation Pack)
-- Migration:    003 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates ownership tables that track equity stakes, control relationships,
-- and resolved ownership chains between entities. Ownership percentages
-- determine the consolidation fraction under the equity share approach.
-- The equity chain table materialises effective ownership through multi-
-- level indirect holdings. Change history captures all ownership events.
--
-- Tables (3):
--   1. ghg_consolidation.gl_cons_ownership
--   2. ghg_consolidation.gl_cons_equity_chains
--   3. ghg_consolidation.gl_cons_ownership_changes
--
-- Also includes: indexes, RLS, constraints, comments.
-- Previous: V417__pack050_entity_registry.sql
-- Next:     V419__pack050_boundary.sql
-- =============================================================================

SET search_path TO ghg_consolidation, public;

-- =============================================================================
-- Table 1: ghg_consolidation.gl_cons_ownership
-- =============================================================================
-- Records the direct ownership percentage that one entity (owner) holds in
-- another entity (owned). Supports temporal validity with effective_from
-- and effective_to dates. Each ownership record also captures the control
-- type and whether this is a direct or intermediate holding.

CREATE TABLE ghg_consolidation.gl_cons_ownership (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    owner_entity_id             UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE CASCADE,
    owned_entity_id             UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE CASCADE,
    ownership_pct               NUMERIC(10,4)   NOT NULL,
    voting_rights_pct           NUMERIC(10,4),
    effective_from              DATE            NOT NULL,
    effective_to                DATE,
    control_type                ghg_consolidation.control_type NOT NULL DEFAULT 'FINANCIAL',
    ownership_type              ghg_consolidation.ownership_type NOT NULL DEFAULT 'EQUITY',
    is_direct                   BOOLEAN         NOT NULL DEFAULT true,
    is_consolidated             BOOLEAN         NOT NULL DEFAULT true,
    consolidation_pct           NUMERIC(10,4),
    evidence_ref                VARCHAR(500),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p050_own_pct CHECK (
        ownership_pct >= 0 AND ownership_pct <= 100
    ),
    CONSTRAINT chk_p050_own_voting CHECK (
        voting_rights_pct IS NULL OR (voting_rights_pct >= 0 AND voting_rights_pct <= 100)
    ),
    CONSTRAINT chk_p050_own_consol_pct CHECK (
        consolidation_pct IS NULL OR (consolidation_pct >= 0 AND consolidation_pct <= 100)
    ),
    CONSTRAINT chk_p050_own_dates CHECK (
        effective_to IS NULL OR effective_to >= effective_from
    ),
    CONSTRAINT chk_p050_own_no_self CHECK (
        owner_entity_id != owned_entity_id
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_own_tenant         ON ghg_consolidation.gl_cons_ownership(tenant_id);
CREATE INDEX idx_p050_own_owner          ON ghg_consolidation.gl_cons_ownership(owner_entity_id);
CREATE INDEX idx_p050_own_owned          ON ghg_consolidation.gl_cons_ownership(owned_entity_id);
CREATE INDEX idx_p050_own_control        ON ghg_consolidation.gl_cons_ownership(control_type);
CREATE INDEX idx_p050_own_type           ON ghg_consolidation.gl_cons_ownership(ownership_type);
CREATE INDEX idx_p050_own_effective      ON ghg_consolidation.gl_cons_ownership(effective_from, effective_to);
CREATE INDEX idx_p050_own_active         ON ghg_consolidation.gl_cons_ownership(owner_entity_id, owned_entity_id)
    WHERE effective_to IS NULL;
CREATE INDEX idx_p050_own_direct         ON ghg_consolidation.gl_cons_ownership(owner_entity_id, owned_entity_id)
    WHERE is_direct = true;
CREATE INDEX idx_p050_own_consolidated   ON ghg_consolidation.gl_cons_ownership(tenant_id, is_consolidated)
    WHERE is_consolidated = true;
CREATE INDEX idx_p050_own_pair_temporal  ON ghg_consolidation.gl_cons_ownership(
    owner_entity_id, owned_entity_id, effective_from, effective_to
);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_ownership ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_own_tenant_isolation ON ghg_consolidation.gl_cons_ownership
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 2: ghg_consolidation.gl_cons_equity_chains
-- =============================================================================
-- Materialised effective ownership percentages through multi-level indirect
-- holdings. When Entity A owns 60% of Entity B, and Entity B owns 80% of
-- Entity C, the effective ownership of A in C is 48%. This table stores
-- the resolved chain path and effective percentage for each root-to-target
-- pair, recalculated on ownership changes.

CREATE TABLE ghg_consolidation.gl_cons_equity_chains (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    root_entity_id              UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE CASCADE,
    target_entity_id            UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE CASCADE,
    effective_pct               NUMERIC(10,4)   NOT NULL,
    chain_path                  TEXT            NOT NULL,
    chain_depth                 INTEGER         NOT NULL DEFAULT 1,
    intermediate_entities       UUID[]          DEFAULT '{}',
    effective_date              DATE            NOT NULL,
    is_current                  BOOLEAN         NOT NULL DEFAULT true,
    resolved_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    resolution_method           VARCHAR(30)     NOT NULL DEFAULT 'MULTIPLICATIVE',
    provenance_hash             VARCHAR(64),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p050_ec_pct CHECK (
        effective_pct >= 0 AND effective_pct <= 100
    ),
    CONSTRAINT chk_p050_ec_depth CHECK (
        chain_depth >= 1 AND chain_depth <= 50
    ),
    CONSTRAINT chk_p050_ec_no_self CHECK (
        root_entity_id != target_entity_id
    ),
    CONSTRAINT chk_p050_ec_method CHECK (
        resolution_method IN ('MULTIPLICATIVE', 'ADDITIVE', 'HYBRID', 'MANUAL')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_ec_tenant          ON ghg_consolidation.gl_cons_equity_chains(tenant_id);
CREATE INDEX idx_p050_ec_root            ON ghg_consolidation.gl_cons_equity_chains(root_entity_id);
CREATE INDEX idx_p050_ec_target          ON ghg_consolidation.gl_cons_equity_chains(target_entity_id);
CREATE INDEX idx_p050_ec_current         ON ghg_consolidation.gl_cons_equity_chains(root_entity_id, target_entity_id)
    WHERE is_current = true;
CREATE INDEX idx_p050_ec_effective_date  ON ghg_consolidation.gl_cons_equity_chains(effective_date);
CREATE INDEX idx_p050_ec_resolved        ON ghg_consolidation.gl_cons_equity_chains(resolved_at);
CREATE INDEX idx_p050_ec_root_current    ON ghg_consolidation.gl_cons_equity_chains(tenant_id, root_entity_id)
    WHERE is_current = true;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_equity_chains ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_ec_tenant_isolation ON ghg_consolidation.gl_cons_equity_chains
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 3: ghg_consolidation.gl_cons_ownership_changes
-- =============================================================================
-- Tracks all changes to ownership stakes over time. Each record captures
-- the entity affected, the type of change, old and new percentages,
-- effective date, and reason. Used for audit trail and M&A history.

CREATE TABLE ghg_consolidation.gl_cons_ownership_changes (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    ownership_id                UUID            REFERENCES ghg_consolidation.gl_cons_ownership(id) ON DELETE SET NULL,
    entity_id                   UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE CASCADE,
    owner_entity_id             UUID            REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE SET NULL,
    change_type                 VARCHAR(30)     NOT NULL,
    old_pct                     NUMERIC(10,4),
    new_pct                     NUMERIC(10,4),
    old_control_type            VARCHAR(30),
    new_control_type            VARCHAR(30),
    effective_date              DATE            NOT NULL,
    reason                      TEXT            NOT NULL,
    evidence_ref                VARCHAR(500),
    approved_by                 UUID,
    approved_at                 TIMESTAMPTZ,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p050_oc_type CHECK (
        change_type IN (
            'ACQUISITION', 'DISPOSAL', 'DILUTION', 'INCREASE',
            'RESTRUCTURE', 'MERGER', 'DEMERGER', 'IPO',
            'BUYOUT', 'JOINT_VENTURE_ENTRY', 'JOINT_VENTURE_EXIT',
            'CORRECTION', 'OTHER'
        )
    ),
    CONSTRAINT chk_p050_oc_old_pct CHECK (
        old_pct IS NULL OR (old_pct >= 0 AND old_pct <= 100)
    ),
    CONSTRAINT chk_p050_oc_new_pct CHECK (
        new_pct IS NULL OR (new_pct >= 0 AND new_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_oc_tenant          ON ghg_consolidation.gl_cons_ownership_changes(tenant_id);
CREATE INDEX idx_p050_oc_entity          ON ghg_consolidation.gl_cons_ownership_changes(entity_id);
CREATE INDEX idx_p050_oc_owner           ON ghg_consolidation.gl_cons_ownership_changes(owner_entity_id)
    WHERE owner_entity_id IS NOT NULL;
CREATE INDEX idx_p050_oc_ownership       ON ghg_consolidation.gl_cons_ownership_changes(ownership_id)
    WHERE ownership_id IS NOT NULL;
CREATE INDEX idx_p050_oc_type            ON ghg_consolidation.gl_cons_ownership_changes(change_type);
CREATE INDEX idx_p050_oc_effective       ON ghg_consolidation.gl_cons_ownership_changes(effective_date);
CREATE INDEX idx_p050_oc_entity_date     ON ghg_consolidation.gl_cons_ownership_changes(entity_id, effective_date DESC);
CREATE INDEX idx_p050_oc_approved        ON ghg_consolidation.gl_cons_ownership_changes(approved_by)
    WHERE approved_by IS NOT NULL;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_ownership_changes ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_oc_tenant_isolation ON ghg_consolidation.gl_cons_ownership_changes
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_consolidation.gl_cons_ownership IS
    'PACK-050: Direct ownership stakes with temporal validity, control type, and consolidation percentage.';
COMMENT ON TABLE ghg_consolidation.gl_cons_equity_chains IS
    'PACK-050: Resolved effective ownership through multi-level indirect holdings with chain path.';
COMMENT ON TABLE ghg_consolidation.gl_cons_ownership_changes IS
    'PACK-050: Ownership change history (13 change types) for audit trail and M&A tracking.';
