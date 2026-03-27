-- =============================================================================
-- PACK-050 GHG Consolidation Pack
-- Migration: V423 - Mergers, Acquisitions & Adjustments
-- =============================================================================
-- Pack:         PACK-050 (GHG Consolidation Pack)
-- Migration:    008 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for mergers and acquisitions (M&A) event tracking, base
-- year restatements triggered by structural changes, and manual/automated
-- adjustments to consolidation runs. Per GHG Protocol Chapter 5, when
-- structural changes occur (acquisitions, divestitures, mergers), the
-- base year inventory must be restated to maintain comparability.
--
-- Tables (3):
--   1. ghg_consolidation.gl_cons_mna_events
--   2. ghg_consolidation.gl_cons_base_year_restatements
--   3. ghg_consolidation.gl_cons_adjustments
--
-- Also includes: indexes, RLS, constraints, comments.
-- Previous: V422__pack050_eliminations.sql
-- Next:     V424__pack050_reporting.sql
-- =============================================================================

SET search_path TO ghg_consolidation, public;

-- =============================================================================
-- Table 1: ghg_consolidation.gl_cons_mna_events
-- =============================================================================
-- Tracks mergers, acquisitions, divestitures, and other structural events
-- that affect the organisational boundary. Each event records the entity
-- involved, event date, boundary impact, pro-rata factor for partial-year
-- accounting, and whether a base year restatement is required.

CREATE TABLE ghg_consolidation.gl_cons_mna_events (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    event_type                  VARCHAR(30)     NOT NULL,
    event_name                  VARCHAR(255)    NOT NULL,
    entity_id                   UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE CASCADE,
    counterparty_entity_id      UUID            REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE SET NULL,
    event_date                  DATE            NOT NULL,
    effective_date              DATE            NOT NULL,
    completion_date             DATE,
    description                 TEXT            NOT NULL,
    boundary_impact             VARCHAR(30)     NOT NULL,
    ownership_before_pct        NUMERIC(10,4),
    ownership_after_pct         NUMERIC(10,4),
    transaction_value           NUMERIC(20,2),
    transaction_currency        VARCHAR(3),
    pro_rata_factor             NUMERIC(10,6),
    pro_rata_start_date         DATE,
    pro_rata_end_date           DATE,
    base_year_restatement_required BOOLEAN      NOT NULL DEFAULT false,
    restatement_triggered       BOOLEAN         NOT NULL DEFAULT false,
    emissions_impact_tco2e      NUMERIC(20,6),
    status                      VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    approved_by                 UUID,
    approved_at                 TIMESTAMPTZ,
    evidence_refs               TEXT[],
    legal_document_ref          VARCHAR(500),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p050_mna_type CHECK (
        event_type IN (
            'ACQUISITION', 'DIVESTITURE', 'MERGER', 'DEMERGER',
            'JOINT_VENTURE_FORMATION', 'JOINT_VENTURE_DISSOLUTION',
            'OUTSOURCING', 'INSOURCING', 'ORGANIC_GROWTH',
            'FACILITY_CLOSURE', 'FACILITY_OPENING',
            'RESTRUCTURE', 'OTHER'
        )
    ),
    CONSTRAINT chk_p050_mna_impact CHECK (
        boundary_impact IN (
            'ENTITY_ADDED', 'ENTITY_REMOVED', 'OWNERSHIP_CHANGED',
            'CONTROL_CHANGED', 'SCOPE_CHANGED', 'NO_CHANGE',
            'PARTIAL_YEAR', 'RESTATEMENT_ONLY'
        )
    ),
    CONSTRAINT chk_p050_mna_status CHECK (
        status IN ('PENDING', 'APPROVED', 'REJECTED', 'APPLIED', 'SUPERSEDED')
    ),
    CONSTRAINT chk_p050_mna_own_before CHECK (
        ownership_before_pct IS NULL OR (ownership_before_pct >= 0 AND ownership_before_pct <= 100)
    ),
    CONSTRAINT chk_p050_mna_own_after CHECK (
        ownership_after_pct IS NULL OR (ownership_after_pct >= 0 AND ownership_after_pct <= 100)
    ),
    CONSTRAINT chk_p050_mna_pro_rata CHECK (
        pro_rata_factor IS NULL OR (pro_rata_factor >= 0 AND pro_rata_factor <= 1)
    ),
    CONSTRAINT chk_p050_mna_pro_rata_dates CHECK (
        (pro_rata_start_date IS NULL AND pro_rata_end_date IS NULL) OR
        (pro_rata_start_date IS NOT NULL AND pro_rata_end_date IS NOT NULL AND pro_rata_end_date >= pro_rata_start_date)
    ),
    CONSTRAINT chk_p050_mna_emissions CHECK (
        emissions_impact_tco2e IS NULL OR emissions_impact_tco2e >= 0
    ),
    CONSTRAINT chk_p050_mna_effective CHECK (
        effective_date >= event_date
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_mna_tenant         ON ghg_consolidation.gl_cons_mna_events(tenant_id);
CREATE INDEX idx_p050_mna_entity         ON ghg_consolidation.gl_cons_mna_events(entity_id);
CREATE INDEX idx_p050_mna_counterparty   ON ghg_consolidation.gl_cons_mna_events(counterparty_entity_id)
    WHERE counterparty_entity_id IS NOT NULL;
CREATE INDEX idx_p050_mna_type           ON ghg_consolidation.gl_cons_mna_events(event_type);
CREATE INDEX idx_p050_mna_impact         ON ghg_consolidation.gl_cons_mna_events(boundary_impact);
CREATE INDEX idx_p050_mna_status         ON ghg_consolidation.gl_cons_mna_events(status);
CREATE INDEX idx_p050_mna_event_date     ON ghg_consolidation.gl_cons_mna_events(event_date);
CREATE INDEX idx_p050_mna_effective_date ON ghg_consolidation.gl_cons_mna_events(effective_date);
CREATE INDEX idx_p050_mna_restatement    ON ghg_consolidation.gl_cons_mna_events(tenant_id, base_year_restatement_required)
    WHERE base_year_restatement_required = true;
CREATE INDEX idx_p050_mna_pending        ON ghg_consolidation.gl_cons_mna_events(tenant_id, status)
    WHERE status = 'PENDING';
CREATE INDEX idx_p050_mna_entity_date    ON ghg_consolidation.gl_cons_mna_events(entity_id, event_date DESC);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_mna_events ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_mna_tenant_isolation ON ghg_consolidation.gl_cons_mna_events
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 2: ghg_consolidation.gl_cons_base_year_restatements
-- =============================================================================
-- Records base year restatements triggered by M&A events or other structural
-- changes per GHG Protocol Chapter 5. Each restatement captures the original
-- and restated base year totals, the adjustment amount, reason, and approval
-- status. Multiple restatements can exist for the same base year.

CREATE TABLE ghg_consolidation.gl_cons_base_year_restatements (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    event_id                    UUID            REFERENCES ghg_consolidation.gl_cons_mna_events(id) ON DELETE SET NULL,
    boundary_id                 UUID            REFERENCES ghg_consolidation.gl_cons_boundaries(id) ON DELETE SET NULL,
    base_year                   INTEGER         NOT NULL,
    restatement_number          INTEGER         NOT NULL DEFAULT 1,
    scope                       VARCHAR(20)     NOT NULL DEFAULT 'TOTAL',
    original_base_year_tco2e    NUMERIC(20,6)   NOT NULL,
    adjustment_amount_tco2e     NUMERIC(20,6)   NOT NULL,
    restated_base_year_tco2e    NUMERIC(20,6)   NOT NULL,
    adjustment_pct              NUMERIC(10,4),
    restatement_reason          TEXT            NOT NULL,
    restatement_methodology     TEXT,
    entity_id                   UUID            REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE SET NULL,
    entity_contribution_tco2e   NUMERIC(20,6),
    significance_threshold_pct  NUMERIC(10,4)   NOT NULL DEFAULT 5.0000,
    is_significant              BOOLEAN         NOT NULL DEFAULT false,
    status                      VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    approved_by                 UUID,
    approved_at                 TIMESTAMPTZ,
    effective_date              DATE            NOT NULL,
    provenance_hash             VARCHAR(64),
    evidence_refs               TEXT[],
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p050_byr_base_year CHECK (
        base_year >= 1990 AND base_year <= 2100
    ),
    CONSTRAINT chk_p050_byr_restatement_num CHECK (
        restatement_number >= 1 AND restatement_number <= 99
    ),
    CONSTRAINT chk_p050_byr_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2_LOCATION', 'SCOPE_2_MARKET', 'SCOPE_3', 'TOTAL')
    ),
    CONSTRAINT chk_p050_byr_original CHECK (original_base_year_tco2e >= 0),
    CONSTRAINT chk_p050_byr_restated CHECK (restated_base_year_tco2e >= 0),
    CONSTRAINT chk_p050_byr_entity_contrib CHECK (
        entity_contribution_tco2e IS NULL OR entity_contribution_tco2e >= 0
    ),
    CONSTRAINT chk_p050_byr_threshold CHECK (
        significance_threshold_pct >= 0 AND significance_threshold_pct <= 100
    ),
    CONSTRAINT chk_p050_byr_status CHECK (
        status IN ('DRAFT', 'REVIEW', 'APPROVED', 'APPLIED', 'SUPERSEDED')
    ),
    CONSTRAINT uq_p050_byr_year_scope_num UNIQUE (tenant_id, base_year, scope, restatement_number)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_byr_tenant         ON ghg_consolidation.gl_cons_base_year_restatements(tenant_id);
CREATE INDEX idx_p050_byr_event          ON ghg_consolidation.gl_cons_base_year_restatements(event_id)
    WHERE event_id IS NOT NULL;
CREATE INDEX idx_p050_byr_boundary       ON ghg_consolidation.gl_cons_base_year_restatements(boundary_id)
    WHERE boundary_id IS NOT NULL;
CREATE INDEX idx_p050_byr_base_year      ON ghg_consolidation.gl_cons_base_year_restatements(base_year);
CREATE INDEX idx_p050_byr_scope          ON ghg_consolidation.gl_cons_base_year_restatements(scope);
CREATE INDEX idx_p050_byr_status         ON ghg_consolidation.gl_cons_base_year_restatements(status);
CREATE INDEX idx_p050_byr_entity         ON ghg_consolidation.gl_cons_base_year_restatements(entity_id)
    WHERE entity_id IS NOT NULL;
CREATE INDEX idx_p050_byr_significant    ON ghg_consolidation.gl_cons_base_year_restatements(tenant_id, is_significant)
    WHERE is_significant = true;
CREATE INDEX idx_p050_byr_effective      ON ghg_consolidation.gl_cons_base_year_restatements(effective_date);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_base_year_restatements ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_byr_tenant_isolation ON ghg_consolidation.gl_cons_base_year_restatements
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 3: ghg_consolidation.gl_cons_adjustments
-- =============================================================================
-- Manual and automated adjustments applied to a consolidation run. Covers
-- corrections, reclassifications, methodology changes, estimation
-- adjustments, and regulatory add-backs. Each adjustment requires
-- justification and optional approval workflow.

CREATE TABLE ghg_consolidation.gl_cons_adjustments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    run_id                      UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_consolidation_runs(id) ON DELETE CASCADE,
    entity_id                   UUID            REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE SET NULL,
    adjustment_type             VARCHAR(30)     NOT NULL,
    scope                       VARCHAR(20)     NOT NULL,
    category                    VARCHAR(50),
    amount_tco2e                NUMERIC(20,6)   NOT NULL,
    is_increase                 BOOLEAN         NOT NULL DEFAULT true,
    reason                      TEXT            NOT NULL,
    justification               TEXT,
    methodology_reference       VARCHAR(500),
    evidence_refs               TEXT[],
    is_recurring                BOOLEAN         NOT NULL DEFAULT false,
    recurrence_period           VARCHAR(20),
    status                      VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    submitted_by                UUID,
    submitted_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    reviewed_by                 UUID,
    reviewed_at                 TIMESTAMPTZ,
    approved_by                 UUID,
    approved_at                 TIMESTAMPTZ,
    rejection_reason            TEXT,
    provenance_hash             VARCHAR(64),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p050_adj_type CHECK (
        adjustment_type IN (
            'CORRECTION', 'RECLASSIFICATION', 'METHODOLOGY_CHANGE',
            'ESTIMATION_ADJUSTMENT', 'REGULATORY_ADDBACK',
            'PRO_RATA_ADJUSTMENT', 'BOUNDARY_CHANGE',
            'DATA_QUALITY_ADJUSTMENT', 'MANUAL_OVERRIDE',
            'ORGANIC_GROWTH', 'OTHER'
        )
    ),
    CONSTRAINT chk_p050_adj_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2_LOCATION', 'SCOPE_2_MARKET', 'SCOPE_3', 'TOTAL')
    ),
    CONSTRAINT chk_p050_adj_amount CHECK (amount_tco2e >= 0),
    CONSTRAINT chk_p050_adj_status CHECK (
        status IN ('PENDING', 'REVIEW', 'APPROVED', 'REJECTED', 'APPLIED', 'REVERSED')
    ),
    CONSTRAINT chk_p050_adj_rejection CHECK (
        status != 'REJECTED' OR rejection_reason IS NOT NULL
    ),
    CONSTRAINT chk_p050_adj_recurrence CHECK (
        recurrence_period IS NULL OR recurrence_period IN ('MONTHLY', 'QUARTERLY', 'ANNUAL')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_adj_tenant         ON ghg_consolidation.gl_cons_adjustments(tenant_id);
CREATE INDEX idx_p050_adj_run            ON ghg_consolidation.gl_cons_adjustments(run_id);
CREATE INDEX idx_p050_adj_entity         ON ghg_consolidation.gl_cons_adjustments(entity_id)
    WHERE entity_id IS NOT NULL;
CREATE INDEX idx_p050_adj_type           ON ghg_consolidation.gl_cons_adjustments(adjustment_type);
CREATE INDEX idx_p050_adj_scope          ON ghg_consolidation.gl_cons_adjustments(scope);
CREATE INDEX idx_p050_adj_status         ON ghg_consolidation.gl_cons_adjustments(status);
CREATE INDEX idx_p050_adj_pending        ON ghg_consolidation.gl_cons_adjustments(run_id, status)
    WHERE status IN ('PENDING', 'REVIEW');
CREATE INDEX idx_p050_adj_approved       ON ghg_consolidation.gl_cons_adjustments(run_id, status)
    WHERE status = 'APPROVED';
CREATE INDEX idx_p050_adj_run_scope      ON ghg_consolidation.gl_cons_adjustments(run_id, scope);
CREATE INDEX idx_p050_adj_recurring      ON ghg_consolidation.gl_cons_adjustments(tenant_id, is_recurring)
    WHERE is_recurring = true;
CREATE INDEX idx_p050_adj_submitted      ON ghg_consolidation.gl_cons_adjustments(submitted_at);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_adjustments ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_adj_tenant_isolation ON ghg_consolidation.gl_cons_adjustments
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_consolidation.gl_cons_mna_events IS
    'PACK-050: M&A events (13 types, 8 impacts) with pro-rata factors and restatement triggers.';
COMMENT ON TABLE ghg_consolidation.gl_cons_base_year_restatements IS
    'PACK-050: Base year restatements per GHG Protocol Ch.5 with significance assessment.';
COMMENT ON TABLE ghg_consolidation.gl_cons_adjustments IS
    'PACK-050: Manual/automated adjustments (11 types, 6 statuses) with approval workflow.';
