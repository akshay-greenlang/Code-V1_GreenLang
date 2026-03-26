-- =============================================================================
-- V346: PACK-043 Scope 3 Complete Pack - Core Enterprise Schema & Maturity
-- =============================================================================
-- Pack:         PACK-043 (Scope 3 Complete Pack)
-- Migration:    001 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates the ghg_accounting_scope3_complete schema and foundational tables
-- for enterprise-grade Scope 3 inventory management. Establishes multi-entity
-- hierarchy with ownership and control structures, organizational boundary
-- definitions per GHG Protocol, maturity assessments for progressive Scope 3
-- capability building, and per-category maturity tracking with upgrade
-- cost/ROI analysis. Supports complex corporate structures (subsidiaries,
-- joint ventures, associates, franchises, investments) with proper
-- consolidation approaches (operational control, financial control, equity
-- share).
--
-- Tables (4):
--   1. ghg_accounting_scope3_complete.entity_hierarchy
--   2. ghg_accounting_scope3_complete.boundary_definitions
--   3. ghg_accounting_scope3_complete.maturity_assessments
--   4. ghg_accounting_scope3_complete.category_maturity
--
-- Enums (5):
--   1. ghg_accounting_scope3_complete.entity_type
--   2. ghg_accounting_scope3_complete.maturity_level
--   3. ghg_accounting_scope3_complete.assurance_level
--   4. ghg_accounting_scope3_complete.consolidation_approach_type
--   5. ghg_accounting_scope3_complete.scope3_category_type
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V345__pack042_views_indexes_seed.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS ghg_accounting_scope3_complete;

SET search_path TO ghg_accounting_scope3_complete, public;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- ---------------------------------------------------------------------------
-- Enum: entity_type
-- ---------------------------------------------------------------------------
-- Types of legal entities within a corporate hierarchy for GHG Protocol
-- organizational boundary determination.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'entity_type' AND typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'ghg_accounting_scope3_complete')) THEN
        CREATE TYPE ghg_accounting_scope3_complete.entity_type AS ENUM (
            'SUBSIDIARY',   -- Majority-owned entity
            'JV',           -- Joint Venture (shared ownership)
            'ASSOCIATE',    -- Minority stake (typically 20-50%)
            'FRANCHISE',    -- Franchise operation
            'INVESTMENT'    -- Portfolio or financial investment
        );
    END IF;
END;
$$;

-- ---------------------------------------------------------------------------
-- Enum: maturity_level
-- ---------------------------------------------------------------------------
-- Five-level maturity model for Scope 3 capability assessment, from initial
-- spend-based estimates through verified supplier-specific data.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'maturity_level' AND typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'ghg_accounting_scope3_complete')) THEN
        CREATE TYPE ghg_accounting_scope3_complete.maturity_level AS ENUM (
            'LEVEL_1',  -- Screening: spend-based EEIO estimates only
            'LEVEL_2',  -- Foundation: industry-average emission factors
            'LEVEL_3',  -- Intermediate: activity-based with secondary EFs
            'LEVEL_4',  -- Advanced: supplier-specific primary data
            'LEVEL_5'   -- Leading: verified, product-level LCA data
        );
    END IF;
END;
$$;

-- ---------------------------------------------------------------------------
-- Enum: assurance_level
-- ---------------------------------------------------------------------------
-- Levels of third-party assurance per ISAE 3000/3410 standards.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'assurance_level' AND typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'ghg_accounting_scope3_complete')) THEN
        CREATE TYPE ghg_accounting_scope3_complete.assurance_level AS ENUM (
            'LIMITED',      -- Limited assurance engagement
            'REASONABLE'    -- Reasonable assurance engagement
        );
    END IF;
END;
$$;

-- ---------------------------------------------------------------------------
-- Enum: consolidation_approach_type
-- ---------------------------------------------------------------------------
-- GHG Protocol organizational boundary consolidation approaches.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'consolidation_approach_type' AND typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'ghg_accounting_scope3_complete')) THEN
        CREATE TYPE ghg_accounting_scope3_complete.consolidation_approach_type AS ENUM (
            'OPERATIONAL_CONTROL',  -- Entity where org directs operating policies
            'FINANCIAL_CONTROL',    -- Entity where org directs financial policies
            'EQUITY_SHARE'          -- Pro-rata share based on ownership percentage
        );
    END IF;
END;
$$;

-- ---------------------------------------------------------------------------
-- Enum: scope3_category_type
-- ---------------------------------------------------------------------------
-- The 15 categories of Scope 3 emissions per GHG Protocol Corporate Value
-- Chain (Scope 3) Accounting and Reporting Standard.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'scope3_category_type' AND typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'ghg_accounting_scope3_complete')) THEN
        CREATE TYPE ghg_accounting_scope3_complete.scope3_category_type AS ENUM (
            'CAT_1',   -- Purchased Goods and Services
            'CAT_2',   -- Capital Goods
            'CAT_3',   -- Fuel- and Energy-Related Activities
            'CAT_4',   -- Upstream Transportation and Distribution
            'CAT_5',   -- Waste Generated in Operations
            'CAT_6',   -- Business Travel
            'CAT_7',   -- Employee Commuting
            'CAT_8',   -- Upstream Leased Assets
            'CAT_9',   -- Downstream Transportation and Distribution
            'CAT_10',  -- Processing of Sold Products
            'CAT_11',  -- Use of Sold Products
            'CAT_12',  -- End-of-Life Treatment of Sold Products
            'CAT_13',  -- Downstream Leased Assets
            'CAT_14',  -- Franchises
            'CAT_15'   -- Investments
        );
    END IF;
END;
$$;

-- =============================================================================
-- Table 1: ghg_accounting_scope3_complete.entity_hierarchy
-- =============================================================================
-- Represents the corporate structure as a tree of legal entities. Each entity
-- has an optional parent, an entity type (subsidiary, JV, etc.), ownership
-- percentage, and control classification. This hierarchy drives the
-- organizational boundary for GHG reporting -- determining which entities'
-- emissions are included and at what consolidation ratio.

CREATE TABLE ghg_accounting_scope3_complete.entity_hierarchy (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    -- Entity identification
    entity_id                   UUID            NOT NULL,
    parent_id                   UUID,
    entity_type                 ghg_accounting_scope3_complete.entity_type NOT NULL,
    name                        VARCHAR(500)    NOT NULL,
    legal_name                  VARCHAR(500),
    -- Geographic
    country                     VARCHAR(3)      NOT NULL DEFAULT 'US',
    region                      VARCHAR(200),
    -- Ownership and control
    ownership_pct               DECIMAL(5,2)    NOT NULL DEFAULT 100.00,
    control_type                ghg_accounting_scope3_complete.consolidation_approach_type NOT NULL DEFAULT 'OPERATIONAL_CONTROL',
    consolidation_approach      ghg_accounting_scope3_complete.consolidation_approach_type NOT NULL DEFAULT 'OPERATIONAL_CONTROL',
    -- Classification
    sector_naics                VARCHAR(10),
    industry_description        VARCHAR(500),
    -- Status
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    effective_date              DATE            NOT NULL DEFAULT CURRENT_DATE,
    termination_date            DATE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p043_eh_country_len CHECK (
        LENGTH(country) BETWEEN 2 AND 3
    ),
    CONSTRAINT chk_p043_eh_ownership CHECK (
        ownership_pct >= 0 AND ownership_pct <= 100
    ),
    CONSTRAINT chk_p043_eh_dates CHECK (
        termination_date IS NULL OR effective_date <= termination_date
    ),
    CONSTRAINT chk_p043_eh_no_self_parent CHECK (
        parent_id IS NULL OR parent_id != entity_id
    ),
    CONSTRAINT uq_p043_eh_tenant_entity UNIQUE (tenant_id, entity_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_eh_tenant             ON ghg_accounting_scope3_complete.entity_hierarchy(tenant_id);
CREATE INDEX idx_p043_eh_entity             ON ghg_accounting_scope3_complete.entity_hierarchy(entity_id);
CREATE INDEX idx_p043_eh_parent             ON ghg_accounting_scope3_complete.entity_hierarchy(parent_id);
CREATE INDEX idx_p043_eh_type               ON ghg_accounting_scope3_complete.entity_hierarchy(entity_type);
CREATE INDEX idx_p043_eh_country            ON ghg_accounting_scope3_complete.entity_hierarchy(country);
CREATE INDEX idx_p043_eh_ownership          ON ghg_accounting_scope3_complete.entity_hierarchy(ownership_pct);
CREATE INDEX idx_p043_eh_control            ON ghg_accounting_scope3_complete.entity_hierarchy(control_type);
CREATE INDEX idx_p043_eh_consolidation      ON ghg_accounting_scope3_complete.entity_hierarchy(consolidation_approach);
CREATE INDEX idx_p043_eh_active             ON ghg_accounting_scope3_complete.entity_hierarchy(is_active) WHERE is_active = true;
CREATE INDEX idx_p043_eh_naics              ON ghg_accounting_scope3_complete.entity_hierarchy(sector_naics);
CREATE INDEX idx_p043_eh_created            ON ghg_accounting_scope3_complete.entity_hierarchy(created_at DESC);
CREATE INDEX idx_p043_eh_metadata           ON ghg_accounting_scope3_complete.entity_hierarchy USING GIN(metadata);

-- Composite: tenant + parent for tree traversal
CREATE INDEX idx_p043_eh_tenant_parent      ON ghg_accounting_scope3_complete.entity_hierarchy(tenant_id, parent_id)
    WHERE is_active = true;

-- Composite: tenant + type + ownership for boundary analysis
CREATE INDEX idx_p043_eh_tenant_type_own    ON ghg_accounting_scope3_complete.entity_hierarchy(tenant_id, entity_type, ownership_pct DESC)
    WHERE is_active = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_eh_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.entity_hierarchy
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_accounting_scope3_complete.boundary_definitions
-- =============================================================================
-- Defines which entities from the hierarchy are included in a specific
-- Scope 3 inventory and under what consolidation approach. For each entity,
-- records whether it is included or excluded and the rationale. This table
-- implements the GHG Protocol "organizational boundary" requirement and
-- allows different inventories to use different boundary approaches.

CREATE TABLE ghg_accounting_scope3_complete.boundary_definitions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL,
    entity_id                   UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.entity_hierarchy(id) ON DELETE CASCADE,
    -- Inclusion
    included                    BOOLEAN         NOT NULL DEFAULT true,
    approach                    ghg_accounting_scope3_complete.consolidation_approach_type NOT NULL DEFAULT 'OPERATIONAL_CONTROL',
    consolidation_pct           DECIMAL(5,2)    NOT NULL DEFAULT 100.00,
    -- Rationale
    rationale                   TEXT,
    exclusion_justification     TEXT,
    -- Scope 3 relevance
    relevant_categories         ghg_accounting_scope3_complete.scope3_category_type[],
    estimated_scope3_tco2e      DECIMAL(15,3),
    significance_pct            DECIMAL(5,2),
    -- Approval
    approved                    BOOLEAN         NOT NULL DEFAULT false,
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_bd_consolidation_pct CHECK (
        consolidation_pct >= 0 AND consolidation_pct <= 100
    ),
    CONSTRAINT chk_p043_bd_estimated CHECK (
        estimated_scope3_tco2e IS NULL OR estimated_scope3_tco2e >= 0
    ),
    CONSTRAINT chk_p043_bd_significance CHECK (
        significance_pct IS NULL OR (significance_pct >= 0 AND significance_pct <= 100)
    ),
    CONSTRAINT chk_p043_bd_exclusion_reason CHECK (
        included = true OR exclusion_justification IS NOT NULL
    ),
    CONSTRAINT uq_p043_bd_inventory_entity UNIQUE (inventory_id, entity_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_bd_tenant             ON ghg_accounting_scope3_complete.boundary_definitions(tenant_id);
CREATE INDEX idx_p043_bd_inventory          ON ghg_accounting_scope3_complete.boundary_definitions(inventory_id);
CREATE INDEX idx_p043_bd_entity             ON ghg_accounting_scope3_complete.boundary_definitions(entity_id);
CREATE INDEX idx_p043_bd_included           ON ghg_accounting_scope3_complete.boundary_definitions(included) WHERE included = true;
CREATE INDEX idx_p043_bd_approach           ON ghg_accounting_scope3_complete.boundary_definitions(approach);
CREATE INDEX idx_p043_bd_approved           ON ghg_accounting_scope3_complete.boundary_definitions(approved);
CREATE INDEX idx_p043_bd_significance       ON ghg_accounting_scope3_complete.boundary_definitions(significance_pct DESC);
CREATE INDEX idx_p043_bd_created            ON ghg_accounting_scope3_complete.boundary_definitions(created_at DESC);
CREATE INDEX idx_p043_bd_categories         ON ghg_accounting_scope3_complete.boundary_definitions USING GIN(relevant_categories);

-- Composite: inventory + included for boundary scope
CREATE INDEX idx_p043_bd_inv_included       ON ghg_accounting_scope3_complete.boundary_definitions(inventory_id, consolidation_pct DESC)
    WHERE included = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_bd_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.boundary_definitions
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_accounting_scope3_complete.maturity_assessments
-- =============================================================================
-- Organization-wide Scope 3 maturity assessment. Evaluates the overall
-- capability level for Scope 3 measurement, reporting, and reduction across
-- all 15 categories. Tracks budget allocation, timeline for improvement,
-- and overall maturity level on a 5-point scale aligned with the GHG
-- Protocol's progressive methodology upgrade path.

CREATE TABLE ghg_accounting_scope3_complete.maturity_assessments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL,
    -- Assessment
    assessment_date             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    assessment_version          INTEGER         NOT NULL DEFAULT 1,
    assessor                    VARCHAR(255),
    -- Overall maturity
    overall_maturity_level      ghg_accounting_scope3_complete.maturity_level NOT NULL DEFAULT 'LEVEL_1',
    previous_maturity_level     ghg_accounting_scope3_complete.maturity_level,
    target_maturity_level       ghg_accounting_scope3_complete.maturity_level,
    -- Budget and timeline
    budget_usd                  NUMERIC(14,2),
    budget_allocated_usd        NUMERIC(14,2),
    budget_spent_usd            NUMERIC(14,2),
    timeline_months             INTEGER,
    -- Categories summary
    categories_at_level_1       INTEGER         DEFAULT 0,
    categories_at_level_2       INTEGER         DEFAULT 0,
    categories_at_level_3       INTEGER         DEFAULT 0,
    categories_at_level_4       INTEGER         DEFAULT 0,
    categories_at_level_5       INTEGER         DEFAULT 0,
    -- Capability scores
    data_collection_score       DECIMAL(5,2),
    methodology_score           DECIMAL(5,2),
    supplier_engagement_score   DECIMAL(5,2),
    reporting_score             DECIMAL(5,2),
    target_setting_score        DECIMAL(5,2),
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'COMPLETED',
    -- Metadata
    notes                       TEXT,
    recommendations             TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_ma_budget CHECK (
        budget_usd IS NULL OR budget_usd >= 0
    ),
    CONSTRAINT chk_p043_ma_budget_alloc CHECK (
        budget_allocated_usd IS NULL OR budget_allocated_usd >= 0
    ),
    CONSTRAINT chk_p043_ma_budget_spent CHECK (
        budget_spent_usd IS NULL OR budget_spent_usd >= 0
    ),
    CONSTRAINT chk_p043_ma_timeline CHECK (
        timeline_months IS NULL OR (timeline_months >= 0 AND timeline_months <= 120)
    ),
    CONSTRAINT chk_p043_ma_cats_l1 CHECK (
        categories_at_level_1 >= 0 AND categories_at_level_1 <= 15
    ),
    CONSTRAINT chk_p043_ma_cats_l2 CHECK (
        categories_at_level_2 >= 0 AND categories_at_level_2 <= 15
    ),
    CONSTRAINT chk_p043_ma_cats_l3 CHECK (
        categories_at_level_3 >= 0 AND categories_at_level_3 <= 15
    ),
    CONSTRAINT chk_p043_ma_cats_l4 CHECK (
        categories_at_level_4 >= 0 AND categories_at_level_4 <= 15
    ),
    CONSTRAINT chk_p043_ma_cats_l5 CHECK (
        categories_at_level_5 >= 0 AND categories_at_level_5 <= 15
    ),
    CONSTRAINT chk_p043_ma_data_score CHECK (
        data_collection_score IS NULL OR (data_collection_score >= 0 AND data_collection_score <= 100)
    ),
    CONSTRAINT chk_p043_ma_method_score CHECK (
        methodology_score IS NULL OR (methodology_score >= 0 AND methodology_score <= 100)
    ),
    CONSTRAINT chk_p043_ma_supplier_score CHECK (
        supplier_engagement_score IS NULL OR (supplier_engagement_score >= 0 AND supplier_engagement_score <= 100)
    ),
    CONSTRAINT chk_p043_ma_reporting_score CHECK (
        reporting_score IS NULL OR (reporting_score >= 0 AND reporting_score <= 100)
    ),
    CONSTRAINT chk_p043_ma_target_score CHECK (
        target_setting_score IS NULL OR (target_setting_score >= 0 AND target_setting_score <= 100)
    ),
    CONSTRAINT chk_p043_ma_version CHECK (
        assessment_version >= 1
    ),
    CONSTRAINT chk_p043_ma_status CHECK (
        status IN ('DRAFT', 'IN_PROGRESS', 'COMPLETED', 'REVIEWED', 'ARCHIVED')
    ),
    CONSTRAINT uq_p043_ma_inventory_version UNIQUE (inventory_id, assessment_version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_ma_tenant             ON ghg_accounting_scope3_complete.maturity_assessments(tenant_id);
CREATE INDEX idx_p043_ma_inventory          ON ghg_accounting_scope3_complete.maturity_assessments(inventory_id);
CREATE INDEX idx_p043_ma_date               ON ghg_accounting_scope3_complete.maturity_assessments(assessment_date DESC);
CREATE INDEX idx_p043_ma_maturity           ON ghg_accounting_scope3_complete.maturity_assessments(overall_maturity_level);
CREATE INDEX idx_p043_ma_status             ON ghg_accounting_scope3_complete.maturity_assessments(status);
CREATE INDEX idx_p043_ma_created            ON ghg_accounting_scope3_complete.maturity_assessments(created_at DESC);

-- Composite: inventory + latest assessment
CREATE INDEX idx_p043_ma_inv_latest         ON ghg_accounting_scope3_complete.maturity_assessments(inventory_id, assessment_date DESC);

-- Composite: tenant + maturity level for benchmarking
CREATE INDEX idx_p043_ma_tenant_level       ON ghg_accounting_scope3_complete.maturity_assessments(tenant_id, overall_maturity_level);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_ma_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.maturity_assessments
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_accounting_scope3_complete.category_maturity
-- =============================================================================
-- Per-category maturity detail within a maturity assessment. For each of the
-- 15 Scope 3 categories, tracks the current and target methodology tier,
-- data quality rating (DQR), estimated upgrade cost, projected ROI from
-- the upgrade, and a priority rank for investment allocation.

CREATE TABLE ghg_accounting_scope3_complete.category_maturity (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    assessment_id               UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.maturity_assessments(id) ON DELETE CASCADE,
    category                    ghg_accounting_scope3_complete.scope3_category_type NOT NULL,
    -- Current state
    current_tier                ghg_accounting_scope3_complete.maturity_level NOT NULL DEFAULT 'LEVEL_1',
    current_dqr                 DECIMAL(3,1)    NOT NULL DEFAULT 5.0,
    current_tco2e               DECIMAL(15,3),
    -- Target state
    target_tier                 ghg_accounting_scope3_complete.maturity_level NOT NULL DEFAULT 'LEVEL_3',
    target_dqr                  DECIMAL(3,1),
    -- Economics
    upgrade_cost                NUMERIC(14,2),
    upgrade_roi                 DECIMAL(8,2),
    payback_months              INTEGER,
    -- Prioritization
    priority_rank               INTEGER,
    significance_pct            DECIMAL(5,2),
    data_availability           VARCHAR(30)     DEFAULT 'PARTIAL',
    -- Upgrade plan
    upgrade_actions             JSONB           DEFAULT '[]',
    upgrade_timeline_months     INTEGER,
    upgrade_status              VARCHAR(30)     DEFAULT 'PLANNED',
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_cm_dqr CHECK (
        current_dqr >= 1.0 AND current_dqr <= 5.0
    ),
    CONSTRAINT chk_p043_cm_target_dqr CHECK (
        target_dqr IS NULL OR (target_dqr >= 1.0 AND target_dqr <= 5.0)
    ),
    CONSTRAINT chk_p043_cm_tco2e CHECK (
        current_tco2e IS NULL OR current_tco2e >= 0
    ),
    CONSTRAINT chk_p043_cm_cost CHECK (
        upgrade_cost IS NULL OR upgrade_cost >= 0
    ),
    CONSTRAINT chk_p043_cm_priority CHECK (
        priority_rank IS NULL OR (priority_rank >= 1 AND priority_rank <= 15)
    ),
    CONSTRAINT chk_p043_cm_significance CHECK (
        significance_pct IS NULL OR (significance_pct >= 0 AND significance_pct <= 100)
    ),
    CONSTRAINT chk_p043_cm_data_avail CHECK (
        data_availability IS NULL OR data_availability IN (
            'AVAILABLE', 'PARTIAL', 'ESTIMATED', 'NOT_AVAILABLE', 'PLANNED'
        )
    ),
    CONSTRAINT chk_p043_cm_timeline CHECK (
        upgrade_timeline_months IS NULL OR (upgrade_timeline_months >= 0 AND upgrade_timeline_months <= 120)
    ),
    CONSTRAINT chk_p043_cm_upgrade_status CHECK (
        upgrade_status IS NULL OR upgrade_status IN (
            'PLANNED', 'IN_PROGRESS', 'COMPLETED', 'DEFERRED', 'CANCELLED'
        )
    ),
    CONSTRAINT uq_p043_cm_assessment_category UNIQUE (assessment_id, category)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_cm_tenant             ON ghg_accounting_scope3_complete.category_maturity(tenant_id);
CREATE INDEX idx_p043_cm_assessment         ON ghg_accounting_scope3_complete.category_maturity(assessment_id);
CREATE INDEX idx_p043_cm_category           ON ghg_accounting_scope3_complete.category_maturity(category);
CREATE INDEX idx_p043_cm_current_tier       ON ghg_accounting_scope3_complete.category_maturity(current_tier);
CREATE INDEX idx_p043_cm_target_tier        ON ghg_accounting_scope3_complete.category_maturity(target_tier);
CREATE INDEX idx_p043_cm_priority           ON ghg_accounting_scope3_complete.category_maturity(priority_rank);
CREATE INDEX idx_p043_cm_dqr               ON ghg_accounting_scope3_complete.category_maturity(current_dqr);
CREATE INDEX idx_p043_cm_significance       ON ghg_accounting_scope3_complete.category_maturity(significance_pct DESC);
CREATE INDEX idx_p043_cm_cost              ON ghg_accounting_scope3_complete.category_maturity(upgrade_cost);
CREATE INDEX idx_p043_cm_created            ON ghg_accounting_scope3_complete.category_maturity(created_at DESC);
CREATE INDEX idx_p043_cm_actions            ON ghg_accounting_scope3_complete.category_maturity USING GIN(upgrade_actions);

-- Composite: assessment + priority for ranked upgrade path
CREATE INDEX idx_p043_cm_assess_priority    ON ghg_accounting_scope3_complete.category_maturity(assessment_id, priority_rank)
    WHERE upgrade_status IN ('PLANNED', 'IN_PROGRESS');

-- Composite: assessment + cost for budget allocation
CREATE INDEX idx_p043_cm_assess_cost        ON ghg_accounting_scope3_complete.category_maturity(assessment_id, upgrade_cost DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_cm_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.category_maturity
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_accounting_scope3_complete.entity_hierarchy ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.boundary_definitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.maturity_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.category_maturity ENABLE ROW LEVEL SECURITY;

CREATE POLICY p043_eh_tenant_isolation
    ON ghg_accounting_scope3_complete.entity_hierarchy
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_eh_service_bypass
    ON ghg_accounting_scope3_complete.entity_hierarchy
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_bd_tenant_isolation
    ON ghg_accounting_scope3_complete.boundary_definitions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_bd_service_bypass
    ON ghg_accounting_scope3_complete.boundary_definitions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_ma_tenant_isolation
    ON ghg_accounting_scope3_complete.maturity_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_ma_service_bypass
    ON ghg_accounting_scope3_complete.maturity_assessments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_cm_tenant_isolation
    ON ghg_accounting_scope3_complete.category_maturity
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_cm_service_bypass
    ON ghg_accounting_scope3_complete.category_maturity
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA ghg_accounting_scope3_complete TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.entity_hierarchy TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.boundary_definitions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.maturity_assessments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.category_maturity TO PUBLIC;
GRANT EXECUTE ON FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA ghg_accounting_scope3_complete IS
    'PACK-043 Scope 3 Complete Pack - Enterprise-grade GHG Protocol Scope 3 inventory management with multi-entity hierarchy, LCA integration, scenario modelling (MACC), SBTi targets, supplier programmes, climate risk, base year management, sector-specific modules, and assurance/verification.';

COMMENT ON TABLE ghg_accounting_scope3_complete.entity_hierarchy IS
    'Corporate structure tree of legal entities (subsidiaries, JVs, associates, franchises, investments) with ownership percentage and control classification for organizational boundary determination.';
COMMENT ON TABLE ghg_accounting_scope3_complete.boundary_definitions IS
    'Per-inventory organizational boundary defining which entities are included/excluded, the consolidation approach, and consolidation percentage per GHG Protocol.';
COMMENT ON TABLE ghg_accounting_scope3_complete.maturity_assessments IS
    'Organization-wide Scope 3 maturity assessment on a 5-level scale with budget allocation, capability scores, and category-level distribution.';
COMMENT ON TABLE ghg_accounting_scope3_complete.category_maturity IS
    'Per-category maturity detail tracking current/target tier, DQR, upgrade cost/ROI, payback period, and priority rank for investment allocation.';

COMMENT ON COLUMN ghg_accounting_scope3_complete.entity_hierarchy.entity_id IS 'Unique business identifier for the entity (may reference external ERP/legal system).';
COMMENT ON COLUMN ghg_accounting_scope3_complete.entity_hierarchy.ownership_pct IS 'Percentage of ownership held by the reporting organization (0-100).';
COMMENT ON COLUMN ghg_accounting_scope3_complete.entity_hierarchy.control_type IS 'Type of control: OPERATIONAL_CONTROL, FINANCIAL_CONTROL, or EQUITY_SHARE.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.entity_hierarchy.consolidation_approach IS 'Consolidation approach used for this entity in GHG reporting.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.entity_hierarchy.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN ghg_accounting_scope3_complete.boundary_definitions.consolidation_pct IS 'Percentage of entity emissions to consolidate (may differ from ownership_pct for operational control approach).';
COMMENT ON COLUMN ghg_accounting_scope3_complete.boundary_definitions.relevant_categories IS 'Array of Scope 3 categories relevant to this entity within the boundary.';

COMMENT ON COLUMN ghg_accounting_scope3_complete.maturity_assessments.overall_maturity_level IS 'LEVEL_1 (Screening) through LEVEL_5 (Leading) per GreenLang Scope 3 maturity model.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.maturity_assessments.budget_usd IS 'Total budget allocated for Scope 3 capability improvement programme.';

COMMENT ON COLUMN ghg_accounting_scope3_complete.category_maturity.current_dqr IS 'Data Quality Rating (1.0 = best, 5.0 = worst) per GHG Protocol Scope 3 guidance.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.category_maturity.upgrade_roi IS 'Return on investment for upgrading methodology tier -- ratio of accuracy improvement value to cost.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.category_maturity.priority_rank IS 'Priority rank for upgrade investment (1 = highest priority, 15 = lowest).';
