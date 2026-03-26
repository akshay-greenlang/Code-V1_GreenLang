-- =============================================================================
-- PACK-049 GHG Multi-Site Management Pack
-- Migration: V409 - Boundary Definitions
-- =============================================================================
-- Pack:         PACK-049 (GHG Multi-Site Management Pack)
-- Migration:    004 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates boundary management tables for organisational and operational
-- boundary definitions. Boundaries determine which sites, entities, and
-- emission sources are included in the consolidation scope, following
-- GHG Protocol guidance on operational control, financial control, and
-- equity share approaches.
--
-- Tables (4):
--   1. ghg_multisite.gl_ms_boundary_definitions
--   2. ghg_multisite.gl_ms_entity_ownership
--   3. ghg_multisite.gl_ms_boundary_inclusions
--   4. ghg_multisite.gl_ms_boundary_changes
--
-- Also includes: indexes, RLS, comments.
-- Previous: V408__pack049_data_collection.sql
-- =============================================================================

SET search_path TO ghg_multisite, public;

-- =============================================================================
-- Table 1: ghg_multisite.gl_ms_boundary_definitions
-- =============================================================================
-- Defines the organisational boundary for a configuration. The boundary
-- sets the consolidation approach (operational control, financial control,
-- or equity share) and tracks when it was last reviewed/approved.

CREATE TABLE ghg_multisite.gl_ms_boundary_definitions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_configurations(id) ON DELETE CASCADE,
    period_id                   UUID            REFERENCES ghg_multisite.gl_ms_reporting_periods(id) ON DELETE SET NULL,
    boundary_name               VARCHAR(255)    NOT NULL,
    boundary_type               VARCHAR(30)     NOT NULL DEFAULT 'ORGANISATIONAL',
    consolidation_approach      VARCHAR(30)     NOT NULL DEFAULT 'OPERATIONAL_CONTROL',
    description                 TEXT,
    scope1_included             BOOLEAN         NOT NULL DEFAULT true,
    scope2_included             BOOLEAN         NOT NULL DEFAULT true,
    scope3_included             BOOLEAN         NOT NULL DEFAULT false,
    scope3_categories           JSONB           DEFAULT '[]',
    status                      VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    approved_at                 TIMESTAMPTZ,
    approved_by                 UUID,
    last_reviewed_at            TIMESTAMPTZ,
    review_frequency_months     INTEGER         NOT NULL DEFAULT 12,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p049_bd_type CHECK (
        boundary_type IN ('ORGANISATIONAL', 'OPERATIONAL', 'COMBINED')
    ),
    CONSTRAINT chk_p049_bd_approach CHECK (
        consolidation_approach IN (
            'OPERATIONAL_CONTROL', 'FINANCIAL_CONTROL', 'EQUITY_SHARE'
        )
    ),
    CONSTRAINT chk_p049_bd_status CHECK (
        status IN ('DRAFT', 'UNDER_REVIEW', 'APPROVED', 'SUPERSEDED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p049_bd_review_freq CHECK (
        review_frequency_months >= 1 AND review_frequency_months <= 60
    ),
    CONSTRAINT uq_p049_bd_cfg_name UNIQUE (config_id, boundary_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_bd_tenant          ON ghg_multisite.gl_ms_boundary_definitions(tenant_id);
CREATE INDEX idx_p049_bd_config          ON ghg_multisite.gl_ms_boundary_definitions(config_id);
CREATE INDEX idx_p049_bd_period          ON ghg_multisite.gl_ms_boundary_definitions(period_id)
    WHERE period_id IS NOT NULL;
CREATE INDEX idx_p049_bd_type            ON ghg_multisite.gl_ms_boundary_definitions(boundary_type);
CREATE INDEX idx_p049_bd_approach        ON ghg_multisite.gl_ms_boundary_definitions(consolidation_approach);
CREATE INDEX idx_p049_bd_status          ON ghg_multisite.gl_ms_boundary_definitions(status);
CREATE INDEX idx_p049_bd_approved        ON ghg_multisite.gl_ms_boundary_definitions(config_id, status)
    WHERE status = 'APPROVED';
CREATE INDEX idx_p049_bd_scope3cats      ON ghg_multisite.gl_ms_boundary_definitions USING gin(scope3_categories);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_boundary_definitions ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_bd_tenant_isolation ON ghg_multisite.gl_ms_boundary_definitions
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 2: ghg_multisite.gl_ms_entity_ownership
-- =============================================================================
-- Tracks ownership percentages for legal entities within the boundary.
-- Used by the equity share and financial control approaches to determine
-- the proportion of emissions to consolidate.

CREATE TABLE ghg_multisite.gl_ms_entity_ownership (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    boundary_id                 UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_boundary_definitions(id) ON DELETE CASCADE,
    entity_name                 VARCHAR(255)    NOT NULL,
    entity_type                 VARCHAR(30)     NOT NULL DEFAULT 'SUBSIDIARY',
    legal_entity_id             UUID,
    parent_entity_id            UUID,
    ownership_pct               NUMERIC(10,4)   NOT NULL DEFAULT 100.0000,
    financial_control           BOOLEAN         NOT NULL DEFAULT false,
    operational_control         BOOLEAN         NOT NULL DEFAULT false,
    equity_share_pct            NUMERIC(10,4),
    consolidation_pct           NUMERIC(10,4)   NOT NULL DEFAULT 100.0000,
    country                     VARCHAR(3),
    effective_from              DATE            NOT NULL DEFAULT CURRENT_DATE,
    effective_to                DATE,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_eo_type CHECK (
        entity_type IN (
            'PARENT', 'SUBSIDIARY', 'JOINT_VENTURE', 'ASSOCIATE',
            'FRANCHISE', 'LEASED', 'PARTNERSHIP', 'SPV', 'OTHER'
        )
    ),
    CONSTRAINT chk_p049_eo_ownership CHECK (
        ownership_pct >= 0 AND ownership_pct <= 100
    ),
    CONSTRAINT chk_p049_eo_equity CHECK (
        equity_share_pct IS NULL OR (equity_share_pct >= 0 AND equity_share_pct <= 100)
    ),
    CONSTRAINT chk_p049_eo_consol CHECK (
        consolidation_pct >= 0 AND consolidation_pct <= 100
    ),
    CONSTRAINT chk_p049_eo_dates CHECK (
        effective_to IS NULL OR effective_to > effective_from
    ),
    CONSTRAINT uq_p049_eo_boundary_entity UNIQUE (boundary_id, entity_name, effective_from)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_eo_tenant          ON ghg_multisite.gl_ms_entity_ownership(tenant_id);
CREATE INDEX idx_p049_eo_boundary        ON ghg_multisite.gl_ms_entity_ownership(boundary_id);
CREATE INDEX idx_p049_eo_entity          ON ghg_multisite.gl_ms_entity_ownership(legal_entity_id)
    WHERE legal_entity_id IS NOT NULL;
CREATE INDEX idx_p049_eo_parent          ON ghg_multisite.gl_ms_entity_ownership(parent_entity_id)
    WHERE parent_entity_id IS NOT NULL;
CREATE INDEX idx_p049_eo_type            ON ghg_multisite.gl_ms_entity_ownership(entity_type);
CREATE INDEX idx_p049_eo_country         ON ghg_multisite.gl_ms_entity_ownership(country)
    WHERE country IS NOT NULL;
CREATE INDEX idx_p049_eo_active          ON ghg_multisite.gl_ms_entity_ownership(boundary_id)
    WHERE effective_to IS NULL;
CREATE INDEX idx_p049_eo_dates           ON ghg_multisite.gl_ms_entity_ownership(effective_from, effective_to);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_entity_ownership ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_eo_tenant_isolation ON ghg_multisite.gl_ms_entity_ownership
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 3: ghg_multisite.gl_ms_boundary_inclusions
-- =============================================================================
-- Explicit include/exclude decisions for each site within a boundary.
-- This allows fine-grained control over which sites contribute to the
-- consolidated total, with justification for exclusions.

CREATE TABLE ghg_multisite.gl_ms_boundary_inclusions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    boundary_id                 UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_boundary_definitions(id) ON DELETE CASCADE,
    site_id                     UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    entity_ownership_id         UUID            REFERENCES ghg_multisite.gl_ms_entity_ownership(id) ON DELETE SET NULL,
    inclusion_status            VARCHAR(20)     NOT NULL DEFAULT 'INCLUDED',
    exclusion_reason            VARCHAR(50),
    exclusion_justification     TEXT,
    consolidation_pct           NUMERIC(10,4)   NOT NULL DEFAULT 100.0000,
    is_de_minimis               BOOLEAN         NOT NULL DEFAULT false,
    de_minimis_pct              NUMERIC(10,4),
    effective_from              DATE            NOT NULL DEFAULT CURRENT_DATE,
    effective_to                DATE,
    approved_by                 UUID,
    approved_at                 TIMESTAMPTZ,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_bi_status CHECK (
        inclusion_status IN ('INCLUDED', 'EXCLUDED', 'PARTIAL', 'PENDING_REVIEW')
    ),
    CONSTRAINT chk_p049_bi_reason CHECK (
        exclusion_reason IS NULL OR exclusion_reason IN (
            'DE_MINIMIS', 'NOT_OPERATIONAL', 'SOLD', 'OUTSIDE_BOUNDARY',
            'DATA_UNAVAILABLE', 'REGULATORY_EXEMPT', 'TEMPORARY', 'OTHER'
        )
    ),
    CONSTRAINT chk_p049_bi_consol CHECK (
        consolidation_pct >= 0 AND consolidation_pct <= 100
    ),
    CONSTRAINT chk_p049_bi_deminimis CHECK (
        de_minimis_pct IS NULL OR (de_minimis_pct >= 0 AND de_minimis_pct <= 100)
    ),
    CONSTRAINT chk_p049_bi_dates CHECK (
        effective_to IS NULL OR effective_to > effective_from
    ),
    CONSTRAINT uq_p049_bi_boundary_site UNIQUE (boundary_id, site_id, effective_from)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_bi_tenant          ON ghg_multisite.gl_ms_boundary_inclusions(tenant_id);
CREATE INDEX idx_p049_bi_boundary        ON ghg_multisite.gl_ms_boundary_inclusions(boundary_id);
CREATE INDEX idx_p049_bi_site            ON ghg_multisite.gl_ms_boundary_inclusions(site_id);
CREATE INDEX idx_p049_bi_entity          ON ghg_multisite.gl_ms_boundary_inclusions(entity_ownership_id)
    WHERE entity_ownership_id IS NOT NULL;
CREATE INDEX idx_p049_bi_status          ON ghg_multisite.gl_ms_boundary_inclusions(inclusion_status);
CREATE INDEX idx_p049_bi_included        ON ghg_multisite.gl_ms_boundary_inclusions(boundary_id, inclusion_status)
    WHERE inclusion_status = 'INCLUDED';
CREATE INDEX idx_p049_bi_excluded        ON ghg_multisite.gl_ms_boundary_inclusions(boundary_id, exclusion_reason)
    WHERE inclusion_status = 'EXCLUDED';
CREATE INDEX idx_p049_bi_deminimis       ON ghg_multisite.gl_ms_boundary_inclusions(boundary_id)
    WHERE is_de_minimis = true;
CREATE INDEX idx_p049_bi_active          ON ghg_multisite.gl_ms_boundary_inclusions(boundary_id)
    WHERE effective_to IS NULL;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_boundary_inclusions ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_bi_tenant_isolation ON ghg_multisite.gl_ms_boundary_inclusions
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 4: ghg_multisite.gl_ms_boundary_changes
-- =============================================================================
-- Audit log of all boundary changes (site additions, removals, ownership
-- changes, approach changes). Required by GHG Protocol for base year
-- recalculation trigger assessment.

CREATE TABLE ghg_multisite.gl_ms_boundary_changes (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    boundary_id                 UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_boundary_definitions(id) ON DELETE CASCADE,
    change_type                 VARCHAR(30)     NOT NULL,
    change_description          TEXT            NOT NULL,
    site_id                     UUID            REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE SET NULL,
    entity_ownership_id         UUID            REFERENCES ghg_multisite.gl_ms_entity_ownership(id) ON DELETE SET NULL,
    previous_value              JSONB,
    new_value                   JSONB,
    effective_date              DATE            NOT NULL DEFAULT CURRENT_DATE,
    requires_base_year_recalc   BOOLEAN         NOT NULL DEFAULT false,
    significance_pct            NUMERIC(10,4),
    approved_by                 UUID,
    approved_at                 TIMESTAMPTZ,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p049_bc_type CHECK (
        change_type IN (
            'SITE_ADDED', 'SITE_REMOVED', 'SITE_TRANSFERRED',
            'ENTITY_ADDED', 'ENTITY_REMOVED', 'ENTITY_MERGED',
            'OWNERSHIP_CHANGE', 'APPROACH_CHANGE',
            'SCOPE_CHANGE', 'THRESHOLD_CHANGE',
            'CORRECTION', 'OTHER'
        )
    ),
    CONSTRAINT chk_p049_bc_significance CHECK (
        significance_pct IS NULL OR (significance_pct >= 0 AND significance_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_bc_tenant          ON ghg_multisite.gl_ms_boundary_changes(tenant_id);
CREATE INDEX idx_p049_bc_boundary        ON ghg_multisite.gl_ms_boundary_changes(boundary_id);
CREATE INDEX idx_p049_bc_type            ON ghg_multisite.gl_ms_boundary_changes(change_type);
CREATE INDEX idx_p049_bc_site            ON ghg_multisite.gl_ms_boundary_changes(site_id)
    WHERE site_id IS NOT NULL;
CREATE INDEX idx_p049_bc_entity          ON ghg_multisite.gl_ms_boundary_changes(entity_ownership_id)
    WHERE entity_ownership_id IS NOT NULL;
CREATE INDEX idx_p049_bc_date            ON ghg_multisite.gl_ms_boundary_changes(effective_date);
CREATE INDEX idx_p049_bc_recalc          ON ghg_multisite.gl_ms_boundary_changes(boundary_id)
    WHERE requires_base_year_recalc = true;
CREATE INDEX idx_p049_bc_created         ON ghg_multisite.gl_ms_boundary_changes(created_at);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_boundary_changes ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_bc_tenant_isolation ON ghg_multisite.gl_ms_boundary_changes
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_multisite.gl_ms_boundary_definitions IS
    'PACK-049: Organisational/operational boundary with consolidation approach, scope inclusion, and review cycle.';
COMMENT ON TABLE ghg_multisite.gl_ms_entity_ownership IS
    'PACK-049: Legal entity ownership (9 types) with equity share, control flags, and effective date ranges.';
COMMENT ON TABLE ghg_multisite.gl_ms_boundary_inclusions IS
    'PACK-049: Site-level include/exclude decisions with de minimis handling and consolidation percentages.';
COMMENT ON TABLE ghg_multisite.gl_ms_boundary_changes IS
    'PACK-049: Boundary change audit log (12 types) with base year recalculation trigger assessment.';
