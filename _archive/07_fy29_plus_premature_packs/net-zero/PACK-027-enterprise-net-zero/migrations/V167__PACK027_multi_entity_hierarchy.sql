-- =============================================================================
-- V167: PACK-027 Enterprise Net Zero - Multi-Entity Hierarchy
-- =============================================================================
-- Pack:         PACK-027 (Enterprise Net Zero Pack)
-- Migration:    002 of 015
-- Date:         March 2026
--
-- Multi-entity corporate hierarchy for GHG Protocol consolidation with
-- parent-child relationships, ownership percentages, control assessments,
-- and intercompany transaction tracking for elimination during consolidation.
--
-- Tables (2):
--   1. pack027_enterprise_net_zero.gl_entity_hierarchy
--   2. pack027_enterprise_net_zero.gl_intercompany_transactions
--
-- Previous: V166__PACK027_enterprise_schema_and_profiles.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack027_enterprise_net_zero.gl_entity_hierarchy
-- =============================================================================
-- Entity tree structure with ownership percentages, control type assessments,
-- consolidation method per entity, and effective dates for tracking
-- acquisitions and divestitures over time.

CREATE TABLE pack027_enterprise_net_zero.gl_entity_hierarchy (
    hierarchy_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    parent_id                   UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    child_id                    UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    -- Relationship
    relationship_type           VARCHAR(30)     NOT NULL,
    ownership_pct               DECIMAL(6,2)    NOT NULL DEFAULT 100.00,
    voting_rights_pct           DECIMAL(6,2),
    control_type                VARCHAR(30)     NOT NULL DEFAULT 'OPERATIONAL',
    consolidation_method        VARCHAR(30)     NOT NULL DEFAULT 'FULL',
    -- Hierarchy depth
    hierarchy_level             INTEGER         NOT NULL DEFAULT 1,
    hierarchy_path              TEXT,
    -- Effective dates
    effective_from              DATE            NOT NULL DEFAULT CURRENT_DATE,
    effective_to                DATE,
    is_active                   BOOLEAN         DEFAULT TRUE,
    -- Acquisition / divestiture
    acquisition_date            DATE,
    divestiture_date            DATE,
    transaction_type            VARCHAR(30),
    pro_rata_months             INTEGER,
    -- Base year impact
    triggers_base_year_recalc   BOOLEAN         DEFAULT FALSE,
    significance_pct            DECIMAL(6,2),
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_eh_relationship_type CHECK (
        relationship_type IN ('SUBSIDIARY', 'JOINT_VENTURE', 'ASSOCIATE', 'SPV',
                              'FRANCHISE', 'JOINT_OPERATION', 'INVESTMENT', 'BRANCH')
    ),
    CONSTRAINT chk_p027_eh_ownership CHECK (
        ownership_pct >= 0 AND ownership_pct <= 100
    ),
    CONSTRAINT chk_p027_eh_voting_rights CHECK (
        voting_rights_pct IS NULL OR (voting_rights_pct >= 0 AND voting_rights_pct <= 100)
    ),
    CONSTRAINT chk_p027_eh_control_type CHECK (
        control_type IN ('FINANCIAL', 'OPERATIONAL', 'JOINT', 'SIGNIFICANT_INFLUENCE', 'NO_CONTROL')
    ),
    CONSTRAINT chk_p027_eh_consolidation CHECK (
        consolidation_method IN ('FULL', 'PROPORTIONAL', 'EQUITY', 'EXCLUDED')
    ),
    CONSTRAINT chk_p027_eh_hierarchy_level CHECK (
        hierarchy_level >= 1 AND hierarchy_level <= 20
    ),
    CONSTRAINT chk_p027_eh_date_order CHECK (
        effective_to IS NULL OR effective_to >= effective_from
    ),
    CONSTRAINT chk_p027_eh_transaction_type CHECK (
        transaction_type IS NULL OR transaction_type IN (
            'ACQUISITION', 'DIVESTITURE', 'MERGER', 'DEMERGER', 'RESTRUCTURE', 'IPO', 'FORMATION'
        )
    ),
    CONSTRAINT chk_p027_eh_pro_rata CHECK (
        pro_rata_months IS NULL OR (pro_rata_months >= 1 AND pro_rata_months <= 12)
    ),
    CONSTRAINT chk_p027_eh_significance CHECK (
        significance_pct IS NULL OR (significance_pct >= 0 AND significance_pct <= 100)
    ),
    CONSTRAINT chk_p027_eh_no_self_reference CHECK (
        parent_id != child_id
    ),
    CONSTRAINT uq_p027_eh_parent_child_active UNIQUE (parent_id, child_id, effective_from)
);

-- =============================================================================
-- Table 2: pack027_enterprise_net_zero.gl_intercompany_transactions
-- =============================================================================
-- Intercompany transaction records for elimination during GHG consolidation.
-- Tracks internal sales, transfers, and services to prevent double-counting.

CREATE TABLE pack027_enterprise_net_zero.gl_intercompany_transactions (
    transaction_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    from_entity                 UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    to_entity                   UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    -- Transaction details
    transaction_date            DATE            NOT NULL,
    fiscal_year                 INTEGER         NOT NULL,
    fiscal_quarter              INTEGER,
    transaction_type            VARCHAR(50)     NOT NULL,
    description                 VARCHAR(500),
    -- Financial
    amount                      DECIMAL(18,2)   NOT NULL,
    currency                    VARCHAR(3)      NOT NULL DEFAULT 'USD',
    amount_usd                  DECIMAL(18,2),
    exchange_rate               DECIMAL(12,6),
    -- Emissions
    emissions_tco2e             DECIMAL(18,4),
    scope_category              VARCHAR(30),
    emission_factor_used        DECIMAL(18,8),
    -- Elimination
    elimination_status          VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    elimination_date            DATE,
    elimination_method          VARCHAR(30),
    eliminated_by               UUID,
    -- Reconciliation
    reconciliation_status       VARCHAR(30)     DEFAULT 'UNMATCHED',
    counterparty_ref            UUID,
    variance_amount             DECIMAL(18,2),
    variance_pct                DECIMAL(6,2),
    -- Metadata
    source_system               VARCHAR(50),
    source_reference            VARCHAR(255),
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_ict_transaction_type CHECK (
        transaction_type IN ('INTERNAL_SALE', 'INTERNAL_SERVICE', 'INTERNAL_TRANSFER',
                              'MANAGEMENT_FEE', 'ROYALTY', 'ENERGY_TRANSFER', 'SHARED_SERVICE',
                              'INTERNAL_TRANSPORT', 'WASTE_TRANSFER', 'OTHER')
    ),
    CONSTRAINT chk_p027_ict_fiscal_year CHECK (
        fiscal_year >= 2015 AND fiscal_year <= 2100
    ),
    CONSTRAINT chk_p027_ict_fiscal_quarter CHECK (
        fiscal_quarter IS NULL OR (fiscal_quarter >= 1 AND fiscal_quarter <= 4)
    ),
    CONSTRAINT chk_p027_ict_currency_len CHECK (
        LENGTH(currency) = 3
    ),
    CONSTRAINT chk_p027_ict_elimination_status CHECK (
        elimination_status IN ('PENDING', 'ELIMINATED', 'PARTIALLY_ELIMINATED', 'NOT_REQUIRED', 'DISPUTED')
    ),
    CONSTRAINT chk_p027_ict_elimination_method CHECK (
        elimination_method IS NULL OR elimination_method IN ('AUTO', 'MANUAL', 'RULE_BASED', 'THRESHOLD')
    ),
    CONSTRAINT chk_p027_ict_recon_status CHECK (
        reconciliation_status IN ('UNMATCHED', 'MATCHED', 'PARTIAL_MATCH', 'DISPUTED', 'EXCLUDED')
    ),
    CONSTRAINT chk_p027_ict_scope_category CHECK (
        scope_category IS NULL OR scope_category IN (
            'SCOPE_1', 'SCOPE_2_LOCATION', 'SCOPE_2_MARKET',
            'SCOPE_3_CAT1', 'SCOPE_3_CAT2', 'SCOPE_3_CAT3', 'SCOPE_3_CAT4', 'SCOPE_3_CAT5',
            'SCOPE_3_CAT6', 'SCOPE_3_CAT7', 'SCOPE_3_CAT8', 'SCOPE_3_CAT9', 'SCOPE_3_CAT10',
            'SCOPE_3_CAT11', 'SCOPE_3_CAT12', 'SCOPE_3_CAT13', 'SCOPE_3_CAT14', 'SCOPE_3_CAT15'
        )
    ),
    CONSTRAINT chk_p027_ict_no_self_transfer CHECK (
        from_entity != to_entity
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for gl_entity_hierarchy
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_eh_tenant             ON pack027_enterprise_net_zero.gl_entity_hierarchy(tenant_id);
CREATE INDEX idx_p027_eh_parent             ON pack027_enterprise_net_zero.gl_entity_hierarchy(parent_id);
CREATE INDEX idx_p027_eh_child              ON pack027_enterprise_net_zero.gl_entity_hierarchy(child_id);
CREATE INDEX idx_p027_eh_parent_child       ON pack027_enterprise_net_zero.gl_entity_hierarchy(parent_id, child_id);
CREATE INDEX idx_p027_eh_relationship       ON pack027_enterprise_net_zero.gl_entity_hierarchy(relationship_type);
CREATE INDEX idx_p027_eh_control            ON pack027_enterprise_net_zero.gl_entity_hierarchy(control_type);
CREATE INDEX idx_p027_eh_consolidation      ON pack027_enterprise_net_zero.gl_entity_hierarchy(consolidation_method);
CREATE INDEX idx_p027_eh_level              ON pack027_enterprise_net_zero.gl_entity_hierarchy(hierarchy_level);
CREATE INDEX idx_p027_eh_active             ON pack027_enterprise_net_zero.gl_entity_hierarchy(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p027_eh_effective          ON pack027_enterprise_net_zero.gl_entity_hierarchy(effective_from, effective_to);
CREATE INDEX idx_p027_eh_acquisition        ON pack027_enterprise_net_zero.gl_entity_hierarchy(acquisition_date);
CREATE INDEX idx_p027_eh_divestiture        ON pack027_enterprise_net_zero.gl_entity_hierarchy(divestiture_date);
CREATE INDEX idx_p027_eh_recalc             ON pack027_enterprise_net_zero.gl_entity_hierarchy(triggers_base_year_recalc) WHERE triggers_base_year_recalc = TRUE;
CREATE INDEX idx_p027_eh_created            ON pack027_enterprise_net_zero.gl_entity_hierarchy(created_at DESC);
CREATE INDEX idx_p027_eh_metadata           ON pack027_enterprise_net_zero.gl_entity_hierarchy USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Indexes for gl_intercompany_transactions
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_ict_tenant            ON pack027_enterprise_net_zero.gl_intercompany_transactions(tenant_id);
CREATE INDEX idx_p027_ict_from              ON pack027_enterprise_net_zero.gl_intercompany_transactions(from_entity);
CREATE INDEX idx_p027_ict_to                ON pack027_enterprise_net_zero.gl_intercompany_transactions(to_entity);
CREATE INDEX idx_p027_ict_from_to           ON pack027_enterprise_net_zero.gl_intercompany_transactions(from_entity, to_entity);
CREATE INDEX idx_p027_ict_date              ON pack027_enterprise_net_zero.gl_intercompany_transactions(transaction_date);
CREATE INDEX idx_p027_ict_fiscal_year       ON pack027_enterprise_net_zero.gl_intercompany_transactions(fiscal_year);
CREATE INDEX idx_p027_ict_type              ON pack027_enterprise_net_zero.gl_intercompany_transactions(transaction_type);
CREATE INDEX idx_p027_ict_elimination       ON pack027_enterprise_net_zero.gl_intercompany_transactions(elimination_status);
CREATE INDEX idx_p027_ict_recon             ON pack027_enterprise_net_zero.gl_intercompany_transactions(reconciliation_status);
CREATE INDEX idx_p027_ict_scope             ON pack027_enterprise_net_zero.gl_intercompany_transactions(scope_category);
CREATE INDEX idx_p027_ict_source            ON pack027_enterprise_net_zero.gl_intercompany_transactions(source_system);
CREATE INDEX idx_p027_ict_counterparty      ON pack027_enterprise_net_zero.gl_intercompany_transactions(counterparty_ref);
CREATE INDEX idx_p027_ict_created           ON pack027_enterprise_net_zero.gl_intercompany_transactions(created_at DESC);
CREATE INDEX idx_p027_ict_metadata          ON pack027_enterprise_net_zero.gl_intercompany_transactions USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p027_entity_hierarchy_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_entity_hierarchy
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

CREATE TRIGGER trg_p027_intercompany_txn_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_intercompany_transactions
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack027_enterprise_net_zero.gl_entity_hierarchy ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack027_enterprise_net_zero.gl_intercompany_transactions ENABLE ROW LEVEL SECURITY;

CREATE POLICY p027_eh_tenant_isolation
    ON pack027_enterprise_net_zero.gl_entity_hierarchy
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p027_eh_service_bypass
    ON pack027_enterprise_net_zero.gl_entity_hierarchy
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p027_ict_tenant_isolation
    ON pack027_enterprise_net_zero.gl_intercompany_transactions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p027_ict_service_bypass
    ON pack027_enterprise_net_zero.gl_intercompany_transactions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_entity_hierarchy TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_intercompany_transactions TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack027_enterprise_net_zero.gl_entity_hierarchy IS
    'Multi-entity corporate hierarchy for GHG Protocol consolidation with ownership percentages, control type, and acquisition/divestiture tracking.';
COMMENT ON TABLE pack027_enterprise_net_zero.gl_intercompany_transactions IS
    'Intercompany transaction records for elimination during GHG consolidation to prevent double-counting across entities.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_entity_hierarchy.hierarchy_id IS 'Unique hierarchy relationship identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_entity_hierarchy.parent_id IS 'Parent entity in the corporate hierarchy.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_entity_hierarchy.child_id IS 'Child entity in the corporate hierarchy.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_entity_hierarchy.relationship_type IS 'Entity relationship type: SUBSIDIARY, JOINT_VENTURE, ASSOCIATE, SPV, FRANCHISE, etc.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_entity_hierarchy.ownership_pct IS 'Ownership percentage held by parent (0-100%).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_entity_hierarchy.control_type IS 'Control assessment: FINANCIAL, OPERATIONAL, JOINT, SIGNIFICANT_INFLUENCE, NO_CONTROL.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_entity_hierarchy.consolidation_method IS 'Consolidation method for this entity: FULL, PROPORTIONAL, EQUITY, EXCLUDED.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_entity_hierarchy.triggers_base_year_recalc IS 'Whether this hierarchy change triggers a GHG Protocol base year recalculation (5% threshold).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_entity_hierarchy.significance_pct IS 'Impact as percentage of group total emissions for 5% significance threshold assessment.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_intercompany_transactions.transaction_id IS 'Unique intercompany transaction identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_intercompany_transactions.from_entity IS 'Selling/providing entity.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_intercompany_transactions.to_entity IS 'Buying/receiving entity.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_intercompany_transactions.elimination_status IS 'Elimination status: PENDING, ELIMINATED, PARTIALLY_ELIMINATED, NOT_REQUIRED, DISPUTED.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_intercompany_transactions.reconciliation_status IS 'Cross-entity reconciliation: UNMATCHED, MATCHED, PARTIAL_MATCH, DISPUTED, EXCLUDED.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_intercompany_transactions.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
