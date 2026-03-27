-- =============================================================================
-- PACK-050 GHG Consolidation Pack
-- Migration: V417 - Entity Registry
-- =============================================================================
-- Pack:         PACK-050 (GHG Consolidation Pack)
-- Migration:    002 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates the entity registry tables for corporate group structure. Entities
-- represent legal companies, subsidiaries, joint ventures, and other
-- organisational units that participate in GHG consolidation. The hierarchy
-- table provides a closure table for efficient ancestor/descendant queries.
--
-- Tables (2):
--   1. ghg_consolidation.gl_cons_entities
--   2. ghg_consolidation.gl_cons_entity_hierarchy
--
-- Also includes: indexes, RLS, self-referential FK, comments.
-- Previous: V416__pack050_core_schema.sql
-- Next:     V418__pack050_ownership.sql
-- =============================================================================

SET search_path TO ghg_consolidation, public;

-- =============================================================================
-- Table 1: ghg_consolidation.gl_cons_entities
-- =============================================================================
-- Master registry of all entities (legal companies, business units, JVs, etc.)
-- in the corporate group. Each entity has a classification, jurisdiction,
-- lifecycle status, and optional identifiers (LEI, ISIN). Supports
-- self-referential parent hierarchy for direct parent lookups.

CREATE TABLE ghg_consolidation.gl_cons_entities (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    parent_entity_id            UUID,
    entity_code                 VARCHAR(50)     NOT NULL,
    legal_name                  VARCHAR(500)    NOT NULL,
    trading_name                VARCHAR(500),
    entity_type                 ghg_consolidation.entity_type NOT NULL DEFAULT 'SUBSIDIARY',
    lifecycle_status            ghg_consolidation.entity_lifecycle NOT NULL DEFAULT 'ACTIVE',
    jurisdiction                VARCHAR(3)      NOT NULL,
    incorporation_date          DATE,
    registration_number         VARCHAR(100),
    lei                         VARCHAR(20),
    isin                        VARCHAR(12),
    sector_code                 VARCHAR(20),
    industry_classification     VARCHAR(100),
    functional_currency         VARCHAR(3)      NOT NULL DEFAULT 'USD',
    fiscal_year_end_month       INTEGER         NOT NULL DEFAULT 12,
    primary_contact_name        VARCHAR(255),
    primary_contact_email       VARCHAR(255),
    address_line_1              VARCHAR(500),
    address_line_2              VARCHAR(500),
    city                        VARCHAR(100),
    state_province              VARCHAR(100),
    postal_code                 VARCHAR(20),
    country                     VARCHAR(3),
    latitude                    NUMERIC(10,7),
    longitude                   NUMERIC(10,7),
    metadata                    JSONB           DEFAULT '{}',
    is_reporting_entity         BOOLEAN         NOT NULL DEFAULT false,
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Self-referential FK (deferred to allow insertion ordering)
    CONSTRAINT fk_p050_ent_parent FOREIGN KEY (parent_entity_id)
        REFERENCES ghg_consolidation.gl_cons_entities(id)
        ON DELETE SET NULL,
    -- Constraints
    CONSTRAINT chk_p050_ent_lei CHECK (
        lei IS NULL OR LENGTH(lei) = 20
    ),
    CONSTRAINT chk_p050_ent_isin CHECK (
        isin IS NULL OR LENGTH(isin) = 12
    ),
    CONSTRAINT chk_p050_ent_jurisdiction CHECK (
        LENGTH(jurisdiction) BETWEEN 2 AND 3
    ),
    CONSTRAINT chk_p050_ent_fy_month CHECK (
        fiscal_year_end_month >= 1 AND fiscal_year_end_month <= 12
    ),
    CONSTRAINT chk_p050_ent_currency CHECK (
        LENGTH(functional_currency) = 3
    ),
    CONSTRAINT chk_p050_ent_no_self_parent CHECK (
        parent_entity_id IS NULL OR parent_entity_id != id
    ),
    CONSTRAINT uq_p050_ent_tenant_code UNIQUE (tenant_id, entity_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_ent_tenant         ON ghg_consolidation.gl_cons_entities(tenant_id);
CREATE INDEX idx_p050_ent_parent         ON ghg_consolidation.gl_cons_entities(parent_entity_id)
    WHERE parent_entity_id IS NOT NULL;
CREATE INDEX idx_p050_ent_type           ON ghg_consolidation.gl_cons_entities(entity_type);
CREATE INDEX idx_p050_ent_lifecycle      ON ghg_consolidation.gl_cons_entities(lifecycle_status);
CREATE INDEX idx_p050_ent_jurisdiction   ON ghg_consolidation.gl_cons_entities(jurisdiction);
CREATE INDEX idx_p050_ent_sector         ON ghg_consolidation.gl_cons_entities(sector_code)
    WHERE sector_code IS NOT NULL;
CREATE INDEX idx_p050_ent_lei            ON ghg_consolidation.gl_cons_entities(lei)
    WHERE lei IS NOT NULL;
CREATE INDEX idx_p050_ent_reporting      ON ghg_consolidation.gl_cons_entities(tenant_id, is_reporting_entity)
    WHERE is_reporting_entity = true;
CREATE INDEX idx_p050_ent_active         ON ghg_consolidation.gl_cons_entities(tenant_id, is_active)
    WHERE is_active = true;
CREATE INDEX idx_p050_ent_tenant_type    ON ghg_consolidation.gl_cons_entities(tenant_id, entity_type);
CREATE INDEX idx_p050_ent_metadata       ON ghg_consolidation.gl_cons_entities USING gin(metadata);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_entities ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_ent_tenant_isolation ON ghg_consolidation.gl_cons_entities
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 2: ghg_consolidation.gl_cons_entity_hierarchy
-- =============================================================================
-- Closure table for the entity hierarchy. Stores all ancestor-descendant
-- relationships with depth and materialised path for efficient tree queries.
-- Enables O(1) subtree lookups, ancestor queries, and depth-limited traversals
-- without recursive CTEs.

CREATE TABLE ghg_consolidation.gl_cons_entity_hierarchy (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    ancestor_id                 UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE CASCADE,
    descendant_id               UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE CASCADE,
    depth                       INTEGER         NOT NULL DEFAULT 0,
    path                        TEXT            NOT NULL DEFAULT '',
    is_direct                   BOOLEAN         NOT NULL DEFAULT false,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p050_eh_depth CHECK (depth >= 0 AND depth <= 50),
    CONSTRAINT chk_p050_eh_self_depth CHECK (
        (ancestor_id = descendant_id AND depth = 0) OR
        (ancestor_id != descendant_id AND depth > 0)
    ),
    CONSTRAINT uq_p050_eh_ancestor_descendant UNIQUE (tenant_id, ancestor_id, descendant_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_eh_tenant          ON ghg_consolidation.gl_cons_entity_hierarchy(tenant_id);
CREATE INDEX idx_p050_eh_ancestor        ON ghg_consolidation.gl_cons_entity_hierarchy(ancestor_id);
CREATE INDEX idx_p050_eh_descendant      ON ghg_consolidation.gl_cons_entity_hierarchy(descendant_id);
CREATE INDEX idx_p050_eh_depth           ON ghg_consolidation.gl_cons_entity_hierarchy(depth);
CREATE INDEX idx_p050_eh_direct          ON ghg_consolidation.gl_cons_entity_hierarchy(ancestor_id, descendant_id)
    WHERE is_direct = true;
CREATE INDEX idx_p050_eh_subtree         ON ghg_consolidation.gl_cons_entity_hierarchy(ancestor_id, depth);
CREATE INDEX idx_p050_eh_ancestors       ON ghg_consolidation.gl_cons_entity_hierarchy(descendant_id, depth);
CREATE INDEX idx_p050_eh_path            ON ghg_consolidation.gl_cons_entity_hierarchy(path text_pattern_ops);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_entity_hierarchy ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_eh_tenant_isolation ON ghg_consolidation.gl_cons_entity_hierarchy
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_consolidation.gl_cons_entities IS
    'PACK-050: Master entity registry with legal identifiers, jurisdiction, sector, and lifecycle status.';
COMMENT ON TABLE ghg_consolidation.gl_cons_entity_hierarchy IS
    'PACK-050: Closure table for entity hierarchy enabling O(1) subtree and ancestor queries.';
