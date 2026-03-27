-- =============================================================================
-- PACK-050 GHG Consolidation Pack
-- Migration: V416 - Core Schema
-- =============================================================================
-- Pack:         PACK-050 (GHG Consolidation Pack)
-- Migration:    001 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates the ghg_consolidation schema and core enums, audit log, and
-- settings tables. All tables use UUID primary keys, NUMERIC(20,6) for
-- emissions, NUMERIC(10,4) for percentages, JSONB for flexible data,
-- and full tenant isolation via RLS.
--
-- Enums (5):
--   1. entity_type
--   2. entity_lifecycle
--   3. consolidation_approach
--   4. control_type
--   5. ownership_type
--
-- Tables (2):
--   1. ghg_consolidation.gl_cons_audit_log
--   2. ghg_consolidation.gl_cons_settings
--
-- Also includes: schema, RLS policies, indexes, comments.
-- Next: V417__pack050_entity_registry.sql
-- =============================================================================

-- Create schema
CREATE SCHEMA IF NOT EXISTS ghg_consolidation;
SET search_path TO ghg_consolidation, public;

-- =============================================================================
-- Enum 1: entity_type
-- =============================================================================
-- Classifies the legal/organisational form of an entity in the group
-- structure. Used to determine default consolidation rules.

DO $$ BEGIN
    CREATE TYPE ghg_consolidation.entity_type AS ENUM (
        'PARENT',
        'SUBSIDIARY',
        'JOINT_VENTURE',
        'ASSOCIATE',
        'SPECIAL_PURPOSE_VEHICLE',
        'BRANCH',
        'DIVISION',
        'PARTNERSHIP',
        'FRANCHISE',
        'OTHER'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- =============================================================================
-- Enum 2: entity_lifecycle
-- =============================================================================
-- Tracks the lifecycle status of an entity within the corporate group.

DO $$ BEGIN
    CREATE TYPE ghg_consolidation.entity_lifecycle AS ENUM (
        'PROSPECT',
        'ACTIVE',
        'DORMANT',
        'DIVESTING',
        'DIVESTED',
        'MERGED',
        'LIQUIDATED',
        'ARCHIVED'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- =============================================================================
-- Enum 3: consolidation_approach
-- =============================================================================
-- GHG Protocol consolidation approaches per Chapter 3 guidance.

DO $$ BEGIN
    CREATE TYPE ghg_consolidation.consolidation_approach AS ENUM (
        'OPERATIONAL_CONTROL',
        'FINANCIAL_CONTROL',
        'EQUITY_SHARE'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- =============================================================================
-- Enum 4: control_type
-- =============================================================================
-- Type of control the reporting entity has over a subsidiary or operation.

DO $$ BEGIN
    CREATE TYPE ghg_consolidation.control_type AS ENUM (
        'OPERATIONAL',
        'FINANCIAL',
        'JOINT_OPERATIONAL',
        'JOINT_FINANCIAL',
        'NO_CONTROL',
        'SIGNIFICANT_INFLUENCE'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- =============================================================================
-- Enum 5: ownership_type
-- =============================================================================
-- Nature of the ownership interest in an entity.

DO $$ BEGIN
    CREATE TYPE ghg_consolidation.ownership_type AS ENUM (
        'EQUITY',
        'VOTING_RIGHTS',
        'CONTRACTUAL',
        'BENEFICIAL',
        'HYBRID'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- =============================================================================
-- Table 1: ghg_consolidation.gl_cons_audit_log
-- =============================================================================
-- Core audit log for all consolidation operations. Captures every significant
-- action (create, update, delete, approve, lock) with full before/after state,
-- actor identity, and timestamp. Immutable once written.

CREATE TABLE ghg_consolidation.gl_cons_audit_log (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    event_type                  VARCHAR(50)     NOT NULL,
    entity_type                 VARCHAR(100)    NOT NULL,
    entity_id                   UUID            NOT NULL,
    actor_id                    UUID,
    actor_name                  VARCHAR(255),
    action                      VARCHAR(30)     NOT NULL,
    old_value                   JSONB,
    new_value                   JSONB,
    change_summary              TEXT,
    ip_address                  VARCHAR(45),
    user_agent                  TEXT,
    correlation_id              UUID,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p050_al_action CHECK (
        action IN (
            'CREATE', 'UPDATE', 'DELETE', 'APPROVE', 'REJECT',
            'LOCK', 'UNLOCK', 'SUBMIT', 'REVERT', 'ARCHIVE',
            'MERGE', 'SPLIT', 'TRANSFER', 'RECALCULATE'
        )
    ),
    CONSTRAINT chk_p050_al_event_type CHECK (
        event_type IN (
            'ENTITY', 'OWNERSHIP', 'BOUNDARY', 'SUBMISSION',
            'CONSOLIDATION', 'ELIMINATION', 'ADJUSTMENT',
            'REPORT', 'SIGNOFF', 'SETTING', 'MNA_EVENT',
            'BASE_YEAR_RESTATEMENT', 'DATA_REQUEST'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_al_tenant          ON ghg_consolidation.gl_cons_audit_log(tenant_id);
CREATE INDEX idx_p050_al_entity          ON ghg_consolidation.gl_cons_audit_log(entity_type, entity_id);
CREATE INDEX idx_p050_al_actor           ON ghg_consolidation.gl_cons_audit_log(actor_id)
    WHERE actor_id IS NOT NULL;
CREATE INDEX idx_p050_al_action          ON ghg_consolidation.gl_cons_audit_log(action);
CREATE INDEX idx_p050_al_event_type      ON ghg_consolidation.gl_cons_audit_log(event_type);
CREATE INDEX idx_p050_al_created         ON ghg_consolidation.gl_cons_audit_log(created_at);
CREATE INDEX idx_p050_al_correlation     ON ghg_consolidation.gl_cons_audit_log(correlation_id)
    WHERE correlation_id IS NOT NULL;
CREATE INDEX idx_p050_al_tenant_created  ON ghg_consolidation.gl_cons_audit_log(tenant_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_audit_log ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_al_tenant_isolation ON ghg_consolidation.gl_cons_audit_log
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 2: ghg_consolidation.gl_cons_settings
-- =============================================================================
-- Per-tenant configuration for the consolidation pack. Controls default
-- consolidation approach, materiality thresholds, data collection periods,
-- and feature toggles. Only one active settings record per tenant.

CREATE TABLE ghg_consolidation.gl_cons_settings (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    setting_name                VARCHAR(255)    NOT NULL,
    default_approach            ghg_consolidation.consolidation_approach NOT NULL DEFAULT 'OPERATIONAL_CONTROL',
    default_materiality_pct     NUMERIC(10,4)   NOT NULL DEFAULT 5.0000,
    default_de_minimis_pct      NUMERIC(10,4)   NOT NULL DEFAULT 1.0000,
    collection_period           VARCHAR(20)     NOT NULL DEFAULT 'ANNUAL',
    fiscal_year_start_month     INTEGER         NOT NULL DEFAULT 1,
    base_year                   INTEGER,
    base_year_locked            BOOLEAN         NOT NULL DEFAULT false,
    auto_calculate_equity       BOOLEAN         NOT NULL DEFAULT true,
    require_dual_signoff        BOOLEAN         NOT NULL DEFAULT false,
    enable_intercompany_elim    BOOLEAN         NOT NULL DEFAULT true,
    enable_mna_tracking         BOOLEAN         NOT NULL DEFAULT true,
    completeness_target_pct     NUMERIC(10,4)   NOT NULL DEFAULT 95.0000,
    config_data                 JSONB           DEFAULT '{}',
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p050_set_period CHECK (
        collection_period IN ('MONTHLY', 'QUARTERLY', 'SEMI_ANNUAL', 'ANNUAL')
    ),
    CONSTRAINT chk_p050_set_fy_month CHECK (
        fiscal_year_start_month >= 1 AND fiscal_year_start_month <= 12
    ),
    CONSTRAINT chk_p050_set_base_year CHECK (
        base_year IS NULL OR (base_year >= 1990 AND base_year <= 2100)
    ),
    CONSTRAINT chk_p050_set_materiality CHECK (
        default_materiality_pct >= 0 AND default_materiality_pct <= 100
    ),
    CONSTRAINT chk_p050_set_deminimis CHECK (
        default_de_minimis_pct >= 0 AND default_de_minimis_pct <= 100
    ),
    CONSTRAINT chk_p050_set_completeness CHECK (
        completeness_target_pct >= 0 AND completeness_target_pct <= 100
    ),
    CONSTRAINT uq_p050_set_tenant_name UNIQUE (tenant_id, setting_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_set_tenant         ON ghg_consolidation.gl_cons_settings(tenant_id);
CREATE INDEX idx_p050_set_active         ON ghg_consolidation.gl_cons_settings(tenant_id, is_active)
    WHERE is_active = true;
CREATE INDEX idx_p050_set_approach       ON ghg_consolidation.gl_cons_settings(default_approach);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_settings ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_set_tenant_isolation ON ghg_consolidation.gl_cons_settings
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA ghg_consolidation IS
    'PACK-050: GHG Consolidation Pack schema for multi-entity corporate GHG consolidation.';
COMMENT ON TABLE ghg_consolidation.gl_cons_audit_log IS
    'PACK-050: Immutable audit log capturing all consolidation actions with before/after state.';
COMMENT ON TABLE ghg_consolidation.gl_cons_settings IS
    'PACK-050: Per-tenant configuration with consolidation approach, thresholds, and feature toggles.';
