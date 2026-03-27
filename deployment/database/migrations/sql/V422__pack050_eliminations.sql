-- =============================================================================
-- PACK-050 GHG Consolidation Pack
-- Migration: V422 - Eliminations
-- =============================================================================
-- Pack:         PACK-050 (GHG Consolidation Pack)
-- Migration:    007 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates intercompany elimination tables to prevent double counting of
-- emissions within the consolidated boundary. When one entity sells
-- energy, goods, or services to another entity in the same group, the
-- associated emissions must be eliminated from the consolidated total.
-- The transfer register records all intercompany flows, the eliminations
-- table records the actual adjustments, and the reconciliation table
-- ensures seller and buyer sides balance.
--
-- Tables (3):
--   1. ghg_consolidation.gl_cons_transfer_register
--   2. ghg_consolidation.gl_cons_eliminations
--   3. ghg_consolidation.gl_cons_elimination_reconciliation
--
-- Also includes: indexes, RLS, constraints, comments.
-- Previous: V421__pack050_consolidation.sql
-- Next:     V423__pack050_mna.sql
-- =============================================================================

SET search_path TO ghg_consolidation, public;

-- =============================================================================
-- Table 1: ghg_consolidation.gl_cons_transfer_register
-- =============================================================================
-- Records all intercompany transfers of energy, goods, waste, or services
-- between entities within the consolidation boundary. Each transfer
-- captures the seller, buyer, transfer type, quantity, emission factor,
-- and resulting emissions. Used as input for elimination calculations.

CREATE TABLE ghg_consolidation.gl_cons_transfer_register (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    seller_entity_id            UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE CASCADE,
    buyer_entity_id             UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE CASCADE,
    boundary_id                 UUID            REFERENCES ghg_consolidation.gl_cons_boundaries(id) ON DELETE SET NULL,
    transfer_type               VARCHAR(30)     NOT NULL,
    transfer_description        TEXT,
    energy_type                 VARCHAR(50),
    quantity                    NUMERIC(20,6)   NOT NULL,
    unit                        VARCHAR(50)     NOT NULL,
    emission_factor             NUMERIC(20,10),
    emission_factor_unit        VARCHAR(100),
    emission_factor_source      VARCHAR(500),
    emissions_tco2e             NUMERIC(20,6)   NOT NULL,
    period_start                DATE            NOT NULL,
    period_end                  DATE            NOT NULL,
    invoice_ref                 VARCHAR(200),
    contract_ref                VARCHAR(200),
    is_verified                 BOOLEAN         NOT NULL DEFAULT false,
    verified_by                 UUID,
    verified_at                 TIMESTAMPTZ,
    seller_scope                VARCHAR(20),
    buyer_scope                 VARCHAR(20),
    evidence_refs               TEXT[],
    provenance_hash             VARCHAR(64),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p050_tr_type CHECK (
        transfer_type IN (
            'ELECTRICITY', 'HEAT', 'STEAM', 'COOLING',
            'RENEWABLE_ENERGY', 'FUEL', 'RAW_MATERIAL',
            'FINISHED_GOODS', 'WASTE', 'TRANSPORT',
            'SHARED_SERVICE', 'WATER', 'OTHER'
        )
    ),
    CONSTRAINT chk_p050_tr_quantity CHECK (quantity > 0),
    CONSTRAINT chk_p050_tr_emissions CHECK (emissions_tco2e >= 0),
    CONSTRAINT chk_p050_tr_dates CHECK (period_end >= period_start),
    CONSTRAINT chk_p050_tr_no_self CHECK (
        seller_entity_id != buyer_entity_id
    ),
    CONSTRAINT chk_p050_tr_seller_scope CHECK (
        seller_scope IS NULL OR seller_scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3')
    ),
    CONSTRAINT chk_p050_tr_buyer_scope CHECK (
        buyer_scope IS NULL OR buyer_scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_tr_tenant          ON ghg_consolidation.gl_cons_transfer_register(tenant_id);
CREATE INDEX idx_p050_tr_seller          ON ghg_consolidation.gl_cons_transfer_register(seller_entity_id);
CREATE INDEX idx_p050_tr_buyer           ON ghg_consolidation.gl_cons_transfer_register(buyer_entity_id);
CREATE INDEX idx_p050_tr_boundary        ON ghg_consolidation.gl_cons_transfer_register(boundary_id)
    WHERE boundary_id IS NOT NULL;
CREATE INDEX idx_p050_tr_type            ON ghg_consolidation.gl_cons_transfer_register(transfer_type);
CREATE INDEX idx_p050_tr_energy_type     ON ghg_consolidation.gl_cons_transfer_register(energy_type)
    WHERE energy_type IS NOT NULL;
CREATE INDEX idx_p050_tr_period          ON ghg_consolidation.gl_cons_transfer_register(period_start, period_end);
CREATE INDEX idx_p050_tr_verified        ON ghg_consolidation.gl_cons_transfer_register(tenant_id, is_verified)
    WHERE is_verified = true;
CREATE INDEX idx_p050_tr_unverified      ON ghg_consolidation.gl_cons_transfer_register(tenant_id, is_verified)
    WHERE is_verified = false;
CREATE INDEX idx_p050_tr_seller_buyer    ON ghg_consolidation.gl_cons_transfer_register(seller_entity_id, buyer_entity_id);
CREATE INDEX idx_p050_tr_invoice         ON ghg_consolidation.gl_cons_transfer_register(invoice_ref)
    WHERE invoice_ref IS NOT NULL;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_transfer_register ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_tr_tenant_isolation ON ghg_consolidation.gl_cons_transfer_register
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 2: ghg_consolidation.gl_cons_eliminations
-- =============================================================================
-- Records the actual elimination entries applied during a consolidation
-- run. Each elimination is linked to a transfer register entry and
-- specifies the scope, amount eliminated, and reason. Eliminations
-- reduce the consolidated total to avoid double counting.

CREATE TABLE ghg_consolidation.gl_cons_eliminations (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    run_id                      UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_consolidation_runs(id) ON DELETE CASCADE,
    transfer_id                 UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_transfer_register(id) ON DELETE CASCADE,
    seller_entity_id            UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE CASCADE,
    buyer_entity_id             UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_entities(id) ON DELETE CASCADE,
    elimination_scope           VARCHAR(20)     NOT NULL,
    elimination_type            VARCHAR(30)     NOT NULL,
    gross_amount_tco2e          NUMERIC(20,6)   NOT NULL,
    elimination_amount_tco2e    NUMERIC(20,6)   NOT NULL,
    net_amount_tco2e            NUMERIC(20,6)   NOT NULL,
    elimination_pct             NUMERIC(10,4)   NOT NULL DEFAULT 100.0000,
    reason                      TEXT            NOT NULL,
    is_automatic                BOOLEAN         NOT NULL DEFAULT false,
    is_partial                  BOOLEAN         NOT NULL DEFAULT false,
    approved_by                 UUID,
    approved_at                 TIMESTAMPTZ,
    provenance_hash             VARCHAR(64),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p050_el_scope CHECK (
        elimination_scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3')
    ),
    CONSTRAINT chk_p050_el_type CHECK (
        elimination_type IN (
            'INTERNAL_ENERGY', 'INTERNAL_TRANSPORT', 'INTERNAL_WASTE',
            'INTERCOMPANY_SALES', 'SHARED_SERVICES', 'INTERNAL_FUEL',
            'RENEWABLE_TRANSFER', 'OTHER'
        )
    ),
    CONSTRAINT chk_p050_el_gross CHECK (gross_amount_tco2e >= 0),
    CONSTRAINT chk_p050_el_amount CHECK (elimination_amount_tco2e >= 0),
    CONSTRAINT chk_p050_el_net CHECK (net_amount_tco2e >= 0),
    CONSTRAINT chk_p050_el_pct CHECK (
        elimination_pct >= 0 AND elimination_pct <= 100
    ),
    CONSTRAINT chk_p050_el_no_self CHECK (
        seller_entity_id != buyer_entity_id
    ),
    CONSTRAINT chk_p050_el_amount_lte_gross CHECK (
        elimination_amount_tco2e <= gross_amount_tco2e
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_el_tenant          ON ghg_consolidation.gl_cons_eliminations(tenant_id);
CREATE INDEX idx_p050_el_run             ON ghg_consolidation.gl_cons_eliminations(run_id);
CREATE INDEX idx_p050_el_transfer        ON ghg_consolidation.gl_cons_eliminations(transfer_id);
CREATE INDEX idx_p050_el_seller          ON ghg_consolidation.gl_cons_eliminations(seller_entity_id);
CREATE INDEX idx_p050_el_buyer           ON ghg_consolidation.gl_cons_eliminations(buyer_entity_id);
CREATE INDEX idx_p050_el_scope           ON ghg_consolidation.gl_cons_eliminations(elimination_scope);
CREATE INDEX idx_p050_el_type            ON ghg_consolidation.gl_cons_eliminations(elimination_type);
CREATE INDEX idx_p050_el_run_scope       ON ghg_consolidation.gl_cons_eliminations(run_id, elimination_scope);
CREATE INDEX idx_p050_el_automatic       ON ghg_consolidation.gl_cons_eliminations(run_id, is_automatic)
    WHERE is_automatic = true;
CREATE INDEX idx_p050_el_partial         ON ghg_consolidation.gl_cons_eliminations(run_id)
    WHERE is_partial = true;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_eliminations ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_el_tenant_isolation ON ghg_consolidation.gl_cons_eliminations
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 3: ghg_consolidation.gl_cons_elimination_reconciliation
-- =============================================================================
-- Reconciles the seller and buyer sides of intercompany eliminations
-- within a consolidation run. Verifies that eliminations are balanced
-- (seller eliminated amount matches buyer eliminated amount) and flags
-- any variances for review.

CREATE TABLE ghg_consolidation.gl_cons_elimination_reconciliation (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    run_id                      UUID            NOT NULL REFERENCES ghg_consolidation.gl_cons_consolidation_runs(id) ON DELETE CASCADE,
    elimination_type            VARCHAR(30)     NOT NULL,
    elimination_scope           VARCHAR(20)     NOT NULL,
    seller_total_tco2e          NUMERIC(20,6)   NOT NULL DEFAULT 0,
    buyer_total_tco2e           NUMERIC(20,6)   NOT NULL DEFAULT 0,
    elimination_total_tco2e     NUMERIC(20,6)   NOT NULL DEFAULT 0,
    variance_tco2e              NUMERIC(20,6)   NOT NULL DEFAULT 0,
    variance_pct                NUMERIC(10,4),
    status                      VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    transfer_count              INTEGER         NOT NULL DEFAULT 0,
    reconciliation_notes        TEXT,
    resolved_by                 UUID,
    resolved_at                 TIMESTAMPTZ,
    provenance_hash             VARCHAR(64),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p050_er_type CHECK (
        elimination_type IN (
            'INTERNAL_ENERGY', 'INTERNAL_TRANSPORT', 'INTERNAL_WASTE',
            'INTERCOMPANY_SALES', 'SHARED_SERVICES', 'INTERNAL_FUEL',
            'RENEWABLE_TRANSFER', 'OTHER', 'ALL'
        )
    ),
    CONSTRAINT chk_p050_er_scope CHECK (
        elimination_scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'ALL')
    ),
    CONSTRAINT chk_p050_er_seller CHECK (seller_total_tco2e >= 0),
    CONSTRAINT chk_p050_er_buyer CHECK (buyer_total_tco2e >= 0),
    CONSTRAINT chk_p050_er_elim CHECK (elimination_total_tco2e >= 0),
    CONSTRAINT chk_p050_er_status CHECK (
        status IN ('PENDING', 'BALANCED', 'VARIANCE', 'RESOLVED', 'WAIVED')
    ),
    CONSTRAINT chk_p050_er_count CHECK (transfer_count >= 0),
    CONSTRAINT chk_p050_er_variance_pct CHECK (
        variance_pct IS NULL OR (variance_pct >= -100 AND variance_pct <= 100)
    ),
    CONSTRAINT uq_p050_er_run_type_scope UNIQUE (run_id, elimination_type, elimination_scope)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p050_er_tenant          ON ghg_consolidation.gl_cons_elimination_reconciliation(tenant_id);
CREATE INDEX idx_p050_er_run             ON ghg_consolidation.gl_cons_elimination_reconciliation(run_id);
CREATE INDEX idx_p050_er_type            ON ghg_consolidation.gl_cons_elimination_reconciliation(elimination_type);
CREATE INDEX idx_p050_er_scope           ON ghg_consolidation.gl_cons_elimination_reconciliation(elimination_scope);
CREATE INDEX idx_p050_er_status          ON ghg_consolidation.gl_cons_elimination_reconciliation(status);
CREATE INDEX idx_p050_er_variance        ON ghg_consolidation.gl_cons_elimination_reconciliation(run_id, status)
    WHERE status = 'VARIANCE';
CREATE INDEX idx_p050_er_run_status      ON ghg_consolidation.gl_cons_elimination_reconciliation(run_id, status);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_consolidation.gl_cons_elimination_reconciliation ENABLE ROW LEVEL SECURITY;

CREATE POLICY p050_er_tenant_isolation ON ghg_consolidation.gl_cons_elimination_reconciliation
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_consolidation.gl_cons_transfer_register IS
    'PACK-050: Intercompany transfer register (13 types) with seller/buyer, quantity, and emissions.';
COMMENT ON TABLE ghg_consolidation.gl_cons_eliminations IS
    'PACK-050: Elimination entries (8 types) applied during consolidation to prevent double counting.';
COMMENT ON TABLE ghg_consolidation.gl_cons_elimination_reconciliation IS
    'PACK-050: Seller/buyer reconciliation per elimination type/scope with variance detection.';
