-- =============================================================================
-- V162: PACK-026 SME Net Zero - Accounting Integration
-- =============================================================================
-- Pack:         PACK-026 (SME Net Zero Pack)
-- Migration:    005 of 008
-- Date:         March 2026
--
-- Accounting software connections (Xero, QuickBooks, Sage) with OAuth
-- token management and sync status. Spend category mapping from GL accounts
-- to Scope 3 categories. Individual spend transactions with automated
-- emission estimation.
--
-- Tables (3):
--   1. pack026_sme_net_zero.accounting_connections
--   2. pack026_sme_net_zero.spend_categories
--   3. pack026_sme_net_zero.spend_transactions
--
-- Previous: V161__PACK026_grants_and_certifications.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack026_sme_net_zero.accounting_connections
-- =============================================================================
-- Accounting software OAuth connections for automated spend data import
-- from Xero, QuickBooks, Sage, and other accounting platforms.

CREATE TABLE pack026_sme_net_zero.accounting_connections (
    connection_id           UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sme_id                  UUID            NOT NULL REFERENCES pack026_sme_net_zero.sme_profiles(sme_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    -- Software details
    software                VARCHAR(30)     NOT NULL,
    software_version        VARCHAR(50),
    -- Authentication (encrypted)
    oauth_token_encrypted   TEXT,
    refresh_token_encrypted TEXT,
    token_expiry            TIMESTAMPTZ,
    api_key_encrypted       TEXT,
    tenant_id_external      VARCHAR(255),
    -- Connection
    connection_name         VARCHAR(255),
    organization_name       VARCHAR(255),
    -- Sync status
    last_sync_date          TIMESTAMPTZ,
    last_sync_status        VARCHAR(20)     DEFAULT 'NEVER',
    last_sync_records       INTEGER         DEFAULT 0,
    last_sync_errors        INTEGER         DEFAULT 0,
    last_sync_error_msg     TEXT,
    next_sync_scheduled     TIMESTAMPTZ,
    sync_frequency_hours    INTEGER         DEFAULT 24,
    -- Configuration
    sync_from_date          DATE,
    gl_accounts_mapped      INTEGER         DEFAULT 0,
    auto_categorize         BOOLEAN         DEFAULT TRUE,
    -- Status
    connection_status       VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p026_ac_software CHECK (
        software IN ('XERO', 'QUICKBOOKS', 'SAGE', 'FREEAGENT', 'FRESHBOOKS',
                     'WAVE', 'ZOHO', 'KASHFLOW', 'OTHER')
    ),
    CONSTRAINT chk_p026_ac_sync_status CHECK (
        last_sync_status IN ('NEVER', 'SUCCESS', 'PARTIAL', 'FAILED', 'IN_PROGRESS')
    ),
    CONSTRAINT chk_p026_ac_connection_status CHECK (
        connection_status IN ('PENDING', 'CONNECTED', 'DISCONNECTED', 'ERROR', 'EXPIRED', 'REVOKED')
    ),
    CONSTRAINT chk_p026_ac_sync_frequency CHECK (
        sync_frequency_hours IS NULL OR (sync_frequency_hours >= 1 AND sync_frequency_hours <= 720)
    ),
    CONSTRAINT uq_p026_ac_sme_software UNIQUE (sme_id, software)
);

-- ---------------------------------------------------------------------------
-- Indexes for accounting_connections
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p026_ac_sme              ON pack026_sme_net_zero.accounting_connections(sme_id);
CREATE INDEX idx_p026_ac_tenant           ON pack026_sme_net_zero.accounting_connections(tenant_id);
CREATE INDEX idx_p026_ac_software         ON pack026_sme_net_zero.accounting_connections(software);
CREATE INDEX idx_p026_ac_conn_status      ON pack026_sme_net_zero.accounting_connections(connection_status);
CREATE INDEX idx_p026_ac_sync_status      ON pack026_sme_net_zero.accounting_connections(last_sync_status);
CREATE INDEX idx_p026_ac_last_sync        ON pack026_sme_net_zero.accounting_connections(last_sync_date);
CREATE INDEX idx_p026_ac_next_sync        ON pack026_sme_net_zero.accounting_connections(next_sync_scheduled);
CREATE INDEX idx_p026_ac_created          ON pack026_sme_net_zero.accounting_connections(created_at DESC);
CREATE INDEX idx_p026_ac_metadata         ON pack026_sme_net_zero.accounting_connections USING GIN(metadata);

-- =============================================================================
-- Table 2: pack026_sme_net_zero.spend_categories
-- =============================================================================
-- GL account to Scope 3 category mapping with monthly/annual spend totals
-- and emission factors for spend-based Scope 3 estimation.

CREATE TABLE pack026_sme_net_zero.spend_categories (
    category_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sme_id                  UUID            NOT NULL REFERENCES pack026_sme_net_zero.sme_profiles(sme_id) ON DELETE CASCADE,
    connection_id           UUID            REFERENCES pack026_sme_net_zero.accounting_connections(connection_id) ON DELETE SET NULL,
    tenant_id               UUID            NOT NULL,
    -- GL Account
    gl_account_code         VARCHAR(50)     NOT NULL,
    account_name            VARCHAR(255)    NOT NULL,
    account_type            VARCHAR(50),
    -- Scope 3 mapping
    scope3_category         INTEGER,
    scope3_category_name    VARCHAR(255),
    ghg_scope               VARCHAR(10)     DEFAULT 'SCOPE_3',
    -- Spend
    monthly_spend_eur       DECIMAL(14,2)   DEFAULT 0,
    annual_spend_eur        DECIMAL(16,2)   DEFAULT 0,
    -- Emission factors
    emission_factor_tco2e_per_eur   DECIMAL(18,10),
    emission_factor_source  VARCHAR(100),
    emission_factor_year    INTEGER,
    estimated_annual_tco2e  DECIMAL(14,4),
    -- Quality
    mapping_confidence      VARCHAR(20)     DEFAULT 'AUTO',
    manual_override         BOOLEAN         DEFAULT FALSE,
    -- Status
    active                  BOOLEAN         DEFAULT TRUE,
    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p026_sc_scope3_cat CHECK (
        scope3_category IS NULL OR (scope3_category >= 1 AND scope3_category <= 15)
    ),
    CONSTRAINT chk_p026_sc_ghg_scope CHECK (
        ghg_scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'OUT_OF_SCOPE')
    ),
    CONSTRAINT chk_p026_sc_mapping_confidence CHECK (
        mapping_confidence IN ('AUTO', 'HIGH', 'MEDIUM', 'LOW', 'MANUAL')
    ),
    CONSTRAINT chk_p026_sc_ef_non_neg CHECK (
        emission_factor_tco2e_per_eur IS NULL OR emission_factor_tco2e_per_eur >= 0
    ),
    CONSTRAINT chk_p026_sc_spend_non_neg CHECK (
        monthly_spend_eur >= 0 AND annual_spend_eur >= 0
    ),
    CONSTRAINT uq_p026_sc_sme_gl_code UNIQUE (sme_id, gl_account_code)
);

-- ---------------------------------------------------------------------------
-- Indexes for spend_categories
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p026_sc_sme              ON pack026_sme_net_zero.spend_categories(sme_id);
CREATE INDEX idx_p026_sc_connection       ON pack026_sme_net_zero.spend_categories(connection_id);
CREATE INDEX idx_p026_sc_tenant           ON pack026_sme_net_zero.spend_categories(tenant_id);
CREATE INDEX idx_p026_sc_gl_code          ON pack026_sme_net_zero.spend_categories(gl_account_code);
CREATE INDEX idx_p026_sc_scope3_cat       ON pack026_sme_net_zero.spend_categories(scope3_category);
CREATE INDEX idx_p026_sc_sme_scope3       ON pack026_sme_net_zero.spend_categories(sme_id, scope3_category);
CREATE INDEX idx_p026_sc_active           ON pack026_sme_net_zero.spend_categories(active);
CREATE INDEX idx_p026_sc_confidence       ON pack026_sme_net_zero.spend_categories(mapping_confidence);
CREATE INDEX idx_p026_sc_created          ON pack026_sme_net_zero.spend_categories(created_at DESC);
CREATE INDEX idx_p026_sc_metadata         ON pack026_sme_net_zero.spend_categories USING GIN(metadata);

-- =============================================================================
-- Table 3: pack026_sme_net_zero.spend_transactions
-- =============================================================================
-- Individual spend transactions imported from accounting software with
-- automated Scope 3 category mapping and tCO2e estimation.

CREATE TABLE pack026_sme_net_zero.spend_transactions (
    transaction_id          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    connection_id           UUID            NOT NULL REFERENCES pack026_sme_net_zero.accounting_connections(connection_id) ON DELETE CASCADE,
    sme_id                  UUID            NOT NULL REFERENCES pack026_sme_net_zero.sme_profiles(sme_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    -- Transaction details
    external_id             VARCHAR(255),
    transaction_date        DATE            NOT NULL,
    gl_code                 VARCHAR(50)     NOT NULL,
    account_name            VARCHAR(255),
    description             TEXT,
    supplier_name           VARCHAR(255),
    -- Amounts
    amount_eur              DECIMAL(14,2)   NOT NULL,
    amount_original         DECIMAL(14,2),
    currency_original       VARCHAR(3),
    -- Emission mapping
    scope3_category         INTEGER,
    estimated_tco2e         DECIMAL(14,6),
    emission_factor_used    DECIMAL(18,10),
    emission_factor_source  VARCHAR(100),
    -- Classification
    is_carbon_relevant      BOOLEAN         DEFAULT TRUE,
    classification_method   VARCHAR(20)     DEFAULT 'AUTO',
    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p026_st_scope3_cat CHECK (
        scope3_category IS NULL OR (scope3_category >= 1 AND scope3_category <= 15)
    ),
    CONSTRAINT chk_p026_st_tco2e_non_neg CHECK (
        estimated_tco2e IS NULL OR estimated_tco2e >= 0
    ),
    CONSTRAINT chk_p026_st_ef_non_neg CHECK (
        emission_factor_used IS NULL OR emission_factor_used >= 0
    ),
    CONSTRAINT chk_p026_st_classification CHECK (
        classification_method IN ('AUTO', 'MANUAL', 'ML', 'RULE_BASED')
    ),
    CONSTRAINT chk_p026_st_currency_len CHECK (
        currency_original IS NULL OR LENGTH(currency_original) = 3
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for spend_transactions
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p026_st_sme              ON pack026_sme_net_zero.spend_transactions(sme_id);
CREATE INDEX idx_p026_st_connection       ON pack026_sme_net_zero.spend_transactions(connection_id);
CREATE INDEX idx_p026_st_tenant           ON pack026_sme_net_zero.spend_transactions(tenant_id);
CREATE INDEX idx_p026_st_date             ON pack026_sme_net_zero.spend_transactions(transaction_date);
CREATE INDEX idx_p026_st_sme_date         ON pack026_sme_net_zero.spend_transactions(sme_id, transaction_date);
CREATE INDEX idx_p026_st_gl_code          ON pack026_sme_net_zero.spend_transactions(gl_code);
CREATE INDEX idx_p026_st_scope3           ON pack026_sme_net_zero.spend_transactions(scope3_category);
CREATE INDEX idx_p026_st_external_id      ON pack026_sme_net_zero.spend_transactions(external_id);
CREATE INDEX idx_p026_st_supplier         ON pack026_sme_net_zero.spend_transactions(supplier_name);
CREATE INDEX idx_p026_st_carbon_rel       ON pack026_sme_net_zero.spend_transactions(is_carbon_relevant);
CREATE INDEX idx_p026_st_created          ON pack026_sme_net_zero.spend_transactions(created_at DESC);
CREATE INDEX idx_p026_st_metadata         ON pack026_sme_net_zero.spend_transactions USING GIN(metadata);

-- Composite index for monthly spend aggregation
CREATE INDEX idx_p026_st_sme_date_scope3  ON pack026_sme_net_zero.spend_transactions(sme_id, transaction_date, scope3_category);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p026_accounting_connections_updated
    BEFORE UPDATE ON pack026_sme_net_zero.accounting_connections
    FOR EACH ROW EXECUTE FUNCTION pack026_sme_net_zero.fn_set_updated_at();

CREATE TRIGGER trg_p026_spend_categories_updated
    BEFORE UPDATE ON pack026_sme_net_zero.spend_categories
    FOR EACH ROW EXECUTE FUNCTION pack026_sme_net_zero.fn_set_updated_at();

-- Note: spend_transactions is append-only (imported data), no update trigger needed

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack026_sme_net_zero.accounting_connections ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack026_sme_net_zero.spend_categories ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack026_sme_net_zero.spend_transactions ENABLE ROW LEVEL SECURITY;

CREATE POLICY p026_ac_tenant_isolation
    ON pack026_sme_net_zero.accounting_connections
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p026_ac_service_bypass
    ON pack026_sme_net_zero.accounting_connections
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p026_sc_tenant_isolation
    ON pack026_sme_net_zero.spend_categories
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p026_sc_service_bypass
    ON pack026_sme_net_zero.spend_categories
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p026_st_tenant_isolation
    ON pack026_sme_net_zero.spend_transactions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p026_st_service_bypass
    ON pack026_sme_net_zero.spend_transactions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack026_sme_net_zero.accounting_connections TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack026_sme_net_zero.spend_categories TO PUBLIC;
GRANT SELECT, INSERT ON pack026_sme_net_zero.spend_transactions TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack026_sme_net_zero.accounting_connections IS
    'Accounting software OAuth connections (Xero, QuickBooks, Sage) for automated spend data import and sync management.';
COMMENT ON TABLE pack026_sme_net_zero.spend_categories IS
    'GL account to Scope 3 category mapping with monthly/annual spend totals and emission factors for spend-based estimation.';
COMMENT ON TABLE pack026_sme_net_zero.spend_transactions IS
    'Individual spend transactions imported from accounting software with automated Scope 3 category mapping and tCO2e estimation.';

COMMENT ON COLUMN pack026_sme_net_zero.accounting_connections.connection_id IS 'Unique connection identifier.';
COMMENT ON COLUMN pack026_sme_net_zero.accounting_connections.software IS 'Accounting software: XERO, QUICKBOOKS, SAGE, FREEAGENT, FRESHBOOKS, WAVE, ZOHO, KASHFLOW, OTHER.';
COMMENT ON COLUMN pack026_sme_net_zero.accounting_connections.oauth_token_encrypted IS 'AES-256-GCM encrypted OAuth access token.';
COMMENT ON COLUMN pack026_sme_net_zero.accounting_connections.last_sync_date IS 'Timestamp of last successful or attempted data sync.';
COMMENT ON COLUMN pack026_sme_net_zero.accounting_connections.connection_status IS 'Connection status: PENDING, CONNECTED, DISCONNECTED, ERROR, EXPIRED, REVOKED.';
COMMENT ON COLUMN pack026_sme_net_zero.spend_categories.category_id IS 'Unique spend category identifier.';
COMMENT ON COLUMN pack026_sme_net_zero.spend_categories.gl_account_code IS 'General ledger account code from the accounting software.';
COMMENT ON COLUMN pack026_sme_net_zero.spend_categories.scope3_category IS 'Mapped GHG Protocol Scope 3 category (1-15).';
COMMENT ON COLUMN pack026_sme_net_zero.spend_categories.emission_factor_tco2e_per_eur IS 'Emission factor in tCO2e per EUR of spend.';
COMMENT ON COLUMN pack026_sme_net_zero.spend_transactions.transaction_id IS 'Unique transaction identifier.';
COMMENT ON COLUMN pack026_sme_net_zero.spend_transactions.transaction_date IS 'Date of the financial transaction.';
COMMENT ON COLUMN pack026_sme_net_zero.spend_transactions.estimated_tco2e IS 'Estimated emissions from this transaction based on spend-based emission factors.';
COMMENT ON COLUMN pack026_sme_net_zero.spend_transactions.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
