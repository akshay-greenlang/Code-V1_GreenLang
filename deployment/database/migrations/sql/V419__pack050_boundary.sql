-- =============================================================================
-- PACK-050 GHG Consolidation Pack
-- Migration: V419 - Boundary
-- =============================================================================
-- Pack:         PACK-050 (GHG Consolidation Pack)
-- Migration:    004 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates organisational boundary tables per GHG Protocol Chapter 3.
-- A boundary defines which entities are included in a consolidated GHG
-- inventory for a given reporting period, the consolidation approach
-- applied, and the inclusion percentage for each entity. Boundary
-- changes are tracked for audit and base year restatement triggers.
--
-- Tables (3):
--   1. ghg_consolidation.gl_cons_boundaries
--   2. ghg_consolidation.gl_cons_entity_inclusions
--   3. ghg_consolidation.gl_cons_boundary_changes
--
-- Also includes: indexes, RLS, constraints, comments.
-- Previous: V418__pack050_ownership.sql
-- Next:     V420__pack050_data_collection.sql
-- =============================================================================

SET search_path TO ghg_consolidation, public;

-- =============================================================================
-- Table 1: ghg_consolidation.gl_cons_boundaries
-- =============================================================================
-- Defines the organisational boundary for a specific reporting period.
-- Each boundary specifies the consolidation approach and materiality
-- threshold. Once locked, no further changes are permitted without
-- explicit unlock and audit trail.

CREATE TABLE ghg_consolidation.gl_cons_boundaries (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    settings_id                 UUID            REFERENCES ghg_consolidation.gl_cons_settings(id) ON DELETE SET NULL,
    boundary_name               VARCHAR(255)    NOT NULL,
    reporting_period_start      DATE            NOT NULL,
    reporting_period_end        DATE            NOT NULL,
    approach                    ghg_consolidation.consolidation_approach NOT NULL DEFAULT 'OPERATIONAL_CONTROL',
    materiality_threshold_pct   NUMERIC(10,4)   NOT NULL DEFAULT 5.0000,
    de_minimis_threshold_pct    NUMERIC(10,4)   NOT NULL DEFAULT 1.0000,
    status                      VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    entity_count                INTEGER         NOT NULL DEFAULT 0,
    total_inclusion_pct         NUMERIC(10,4),
    boundary_justification      TEXT,
    locked_at                   TIMESTAMPTZ,
    locked_by                   UUID,
    approved_at                 TIMESTAMPTZ,
    approved_by                 UUID,
    provenance_hash             VARCHAR(64),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p050_bnd_dates CHECK (
        reporting_period_end > reporting_period_start
    ),
    CONSTRAINT chk_p050_bnd_materiality CHECK (
        materiality_threshold_pct >= 0 AND materiality_threshold_pct <= 100
    ),
    CONSTRAINT chk_p050_bnd_deminimis CHECK (
        de_minimis_threshold_pct >= 0 AND de_minimis_threshold_pct <= 100
    ),
    CONSTRAINT chk_p050_bnd_status CHECK (
        status IN ('DRAFT', 'REVIEW', 'APPROVED', 'LOCKED', 'SUPERSEDED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p050_bnd_entity_count CHECK (entity_count >= 0),
    CONSTRAINT chk_p050_bnd_total_incl CHECK (
        total_inclusion_pct IS NULL OR (total_inclusion_pct >= 0 AND total_inclusion_pct <= 100)
    ),
    CONSTRAINT chk_p050_bnd_lock_requires_approval CHECK (
        locked_at IS NULL OR approved_at IS NOT NULL
    ),
    CONSTRAINT uq_p050_bnd_tenant_name_period UNIQUE (tenant_id, boundary_name, reporting_period_start)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_bnd_tenant         ON ghg_consolidation.gl_cons_boundaries(tenant_id);
CREATE INDEX idx_p050_bnd_settings       ON ghg_consolidation.gl_cons_boundaries(settings_id)
    WHERE settings_id IS NOT NULL;
CREATE INDEX idx_p050_bnd_approach       ON ghg_consolidation.gl_cons_boundaries(approach);
CREATE INDEX idx_p050_bnd_status         ON ghg_consolidation.gl_cons_boundaries(status);
CREATE INDEX idx_p050_bnd_period         ON ghg_consolidation.gl_cons_boundaries(reporting_period_start, reporting_period_end);
CREATE INDEX idx_p050_bnd_locked         ON ghg_consolidation.gl_cons_boundaries(tenant_id, status)
    WHERE status = 'LOCKED';
CREATE INDEX idx_p050_bnd_approved       ON ghg_consolidation.gl_cons_boundaries(tenant_id, status)
    WHERE status = 'APPROVED';

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_boundaries ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_bnd_tenant_isolation ON ghg_consolidation.gl_cons_boundaries
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 2: ghg_consolidation.gl_cons_entity_inclusions
-- =============================================================================
-- Maps entities into a boundary with their inclusion percentage and control
-- type. Under operational control, included entities are 100%; under equity
-- share, the inclusion percentage matches the effective ownership stake.
-- Each inclusion record provides the reason for inclusion/exclusion.

CREATE TABLE ghg_consolidation.gl_cons_entity_inclusions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    boundary_id                 UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_boundaries(id) ON DELETE CASCADE,
    entity_id                   UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE CASCADE,
    inclusion_status            VARCHAR(20)     NOT NULL DEFAULT 'INCLUDED',
    inclusion_pct               NUMERIC(10,4)   NOT NULL DEFAULT 100.0000,
    control_type                ghg_consolidation.control_type NOT NULL DEFAULT 'OPERATIONAL',
    ownership_pct               NUMERIC(10,4),
    equity_chain_id             UUID            REFERENCES ghg_consolidation.gl_cons_equity_chains(id) ON DELETE SET NULL,
    inclusion_reason            TEXT            NOT NULL DEFAULT 'Within organisational boundary',
    exclusion_justification     TEXT,
    scopes_included             VARCHAR(30)[]   DEFAULT ARRAY['SCOPE_1', 'SCOPE_2', 'SCOPE_3']::VARCHAR(30)[],
    is_material                 BOOLEAN         NOT NULL DEFAULT true,
    materiality_assessment      TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p050_ei_status CHECK (
        inclusion_status IN ('INCLUDED', 'EXCLUDED', 'PARTIAL', 'DE_MINIMIS', 'PENDING_REVIEW')
    ),
    CONSTRAINT chk_p050_ei_pct CHECK (
        inclusion_pct >= 0 AND inclusion_pct <= 100
    ),
    CONSTRAINT chk_p050_ei_ownership CHECK (
        ownership_pct IS NULL OR (ownership_pct >= 0 AND ownership_pct <= 100)
    ),
    CONSTRAINT chk_p050_ei_excluded_justification CHECK (
        inclusion_status != 'EXCLUDED' OR exclusion_justification IS NOT NULL
    ),
    CONSTRAINT uq_p050_ei_boundary_entity UNIQUE (boundary_id, entity_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_ei_tenant          ON ghg_consolidation.gl_cons_entity_inclusions(tenant_id);
CREATE INDEX idx_p050_ei_boundary        ON ghg_consolidation.gl_cons_entity_inclusions(boundary_id);
CREATE INDEX idx_p050_ei_entity          ON ghg_consolidation.gl_cons_entity_inclusions(entity_id);
CREATE INDEX idx_p050_ei_status          ON ghg_consolidation.gl_cons_entity_inclusions(inclusion_status);
CREATE INDEX idx_p050_ei_control         ON ghg_consolidation.gl_cons_entity_inclusions(control_type);
CREATE INDEX idx_p050_ei_included        ON ghg_consolidation.gl_cons_entity_inclusions(boundary_id, inclusion_status)
    WHERE inclusion_status = 'INCLUDED';
CREATE INDEX idx_p050_ei_material        ON ghg_consolidation.gl_cons_entity_inclusions(boundary_id, is_material)
    WHERE is_material = true;
CREATE INDEX idx_p050_ei_equity_chain    ON ghg_consolidation.gl_cons_entity_inclusions(equity_chain_id)
    WHERE equity_chain_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_entity_inclusions ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_ei_tenant_isolation ON ghg_consolidation.gl_cons_entity_inclusions
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 3: ghg_consolidation.gl_cons_boundary_changes
-- =============================================================================
-- Tracks all changes to a boundary definition over time. Each record
-- captures what changed, the old and new values, justification, and
-- whether the change requires a base year restatement.

CREATE TABLE ghg_consolidation.gl_cons_boundary_changes (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    boundary_id                 UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_boundaries(id) ON DELETE CASCADE,
    change_type                 VARCHAR(30)     NOT NULL,
    field_changed               VARCHAR(100),
    old_value                   TEXT,
    new_value                   TEXT,
    entity_id                   UUID            REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE SET NULL,
    justification               TEXT            NOT NULL,
    requires_restatement        BOOLEAN         NOT NULL DEFAULT false,
    restatement_triggered       BOOLEAN         NOT NULL DEFAULT false,
    impact_assessment           TEXT,
    approved_by                 UUID,
    approved_at                 TIMESTAMPTZ,
    metadata                    JSONB           DEFAULT '{}',
    changed_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    changed_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p050_bc_type CHECK (
        change_type IN (
            'ENTITY_ADDED', 'ENTITY_REMOVED', 'ENTITY_PCT_CHANGED',
            'APPROACH_CHANGED', 'THRESHOLD_CHANGED', 'SCOPE_CHANGED',
            'PERIOD_CHANGED', 'STATUS_CHANGED', 'LOCK', 'UNLOCK',
            'CORRECTION', 'OTHER'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_bc_tenant          ON ghg_consolidation.gl_cons_boundary_changes(tenant_id);
CREATE INDEX idx_p050_bc_boundary        ON ghg_consolidation.gl_cons_boundary_changes(boundary_id);
CREATE INDEX idx_p050_bc_entity          ON ghg_consolidation.gl_cons_boundary_changes(entity_id)
    WHERE entity_id IS NOT NULL;
CREATE INDEX idx_p050_bc_type            ON ghg_consolidation.gl_cons_boundary_changes(change_type);
CREATE INDEX idx_p050_bc_changed_at      ON ghg_consolidation.gl_cons_boundary_changes(changed_at);
CREATE INDEX idx_p050_bc_restatement     ON ghg_consolidation.gl_cons_boundary_changes(boundary_id, requires_restatement)
    WHERE requires_restatement = true;
CREATE INDEX idx_p050_bc_boundary_date   ON ghg_consolidation.gl_cons_boundary_changes(boundary_id, changed_at DESC);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_boundary_changes ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_bc_tenant_isolation ON ghg_consolidation.gl_cons_boundary_changes
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_consolidation.gl_cons_boundaries IS
    'PACK-050: Organisational boundaries per GHG Protocol Ch.3 with approach, materiality, and lock lifecycle.';
COMMENT ON TABLE ghg_consolidation.gl_cons_entity_inclusions IS
    'PACK-050: Entity-to-boundary mapping with inclusion percentage, control type, and materiality assessment.';
COMMENT ON TABLE ghg_consolidation.gl_cons_boundary_changes IS
    'PACK-050: Boundary change history (12 change types) with restatement trigger tracking.';
