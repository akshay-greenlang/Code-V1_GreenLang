-- =============================================================================
-- V337: PACK-042 Scope 3 Starter Pack - Spend Classification & EEIO Mapping
-- =============================================================================
-- Pack:         PACK-042 (Scope 3 Starter Pack)
-- Migration:    002 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates spend classification and EEIO (Environmentally Extended Input-Output)
-- sector mapping tables. Stores raw procurement transactions, classification
-- results mapping transactions to Scope 3 categories and EEIO sectors,
-- emission factor intensities from EEIO models, and a NAICS-to-category
-- reference lookup. Supports the Tier 1 spend-based method which is the
-- starting point for most organizations beginning Scope 3 reporting.
--
-- Tables (4):
--   1. ghg_accounting_scope3.spend_transactions
--   2. ghg_accounting_scope3.classification_results
--   3. ghg_accounting_scope3.eeio_sector_factors
--   4. ghg_accounting_scope3.naics_category_mapping
--
-- Seed Data:
--   - 50+ common NAICS-to-Scope 3 category mappings
--   - 30+ EEIO sector emission intensities (Exiobase 3 top sectors)
--
-- Also includes: indexes, RLS, comments.
-- Previous: V336__pack042_core_schema.sql
-- =============================================================================

SET search_path TO ghg_accounting_scope3, public;

-- =============================================================================
-- Table 1: ghg_accounting_scope3.spend_transactions
-- =============================================================================
-- Raw procurement/purchasing transactions extracted from ERP or finance
-- systems. Each transaction represents a line-level purchase with vendor,
-- description, amount, currency, GL account, and optional NAICS code.
-- These transactions feed the spend-based (Tier 1) calculation method.

CREATE TABLE ghg_accounting_scope3.spend_transactions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL REFERENCES ghg_accounting_scope3.scope3_inventories(id) ON DELETE CASCADE,
    -- Vendor information
    vendor_name                 VARCHAR(500)    NOT NULL,
    vendor_id                   VARCHAR(100),
    vendor_country              VARCHAR(3),
    -- Transaction details
    description                 TEXT,
    transaction_date            DATE            NOT NULL,
    amount                      NUMERIC(18,2)   NOT NULL,
    currency                    VARCHAR(3)      NOT NULL DEFAULT 'USD',
    amount_usd                  NUMERIC(18,2),
    exchange_rate               DECIMAL(12,6),
    -- GL/Accounting
    gl_account                  VARCHAR(50),
    gl_account_description      VARCHAR(500),
    cost_center                 VARCHAR(100),
    purchase_order              VARCHAR(100),
    -- Classification hints
    naics_code                  VARCHAR(10),
    naics_description           VARCHAR(500),
    unspsc_code                 VARCHAR(20),
    commodity_code              VARCHAR(50),
    -- Source
    source_system               VARCHAR(100)    DEFAULT 'ERP',
    source_record_id            VARCHAR(200),
    -- Processing
    is_classified               BOOLEAN         NOT NULL DEFAULT false,
    is_excluded                 BOOLEAN         NOT NULL DEFAULT false,
    exclusion_reason            TEXT,
    -- Metadata
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_st_amount CHECK (
        amount >= 0
    ),
    CONSTRAINT chk_p042_st_amount_usd CHECK (
        amount_usd IS NULL OR amount_usd >= 0
    ),
    CONSTRAINT chk_p042_st_exchange_rate CHECK (
        exchange_rate IS NULL OR exchange_rate > 0
    ),
    CONSTRAINT chk_p042_st_currency_len CHECK (
        LENGTH(currency) = 3
    ),
    CONSTRAINT chk_p042_st_vendor_country CHECK (
        vendor_country IS NULL OR LENGTH(vendor_country) BETWEEN 2 AND 3
    ),
    CONSTRAINT chk_p042_st_source CHECK (
        source_system IS NULL OR source_system IN (
            'ERP', 'AP_SYSTEM', 'PROCUREMENT', 'EXPENSE_MGMT',
            'MANUAL_UPLOAD', 'API_IMPORT', 'OTHER'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_st_tenant             ON ghg_accounting_scope3.spend_transactions(tenant_id);
CREATE INDEX idx_p042_st_inventory          ON ghg_accounting_scope3.spend_transactions(inventory_id);
CREATE INDEX idx_p042_st_vendor             ON ghg_accounting_scope3.spend_transactions(vendor_name);
CREATE INDEX idx_p042_st_vendor_id          ON ghg_accounting_scope3.spend_transactions(vendor_id);
CREATE INDEX idx_p042_st_date               ON ghg_accounting_scope3.spend_transactions(transaction_date);
CREATE INDEX idx_p042_st_amount             ON ghg_accounting_scope3.spend_transactions(amount DESC);
CREATE INDEX idx_p042_st_currency           ON ghg_accounting_scope3.spend_transactions(currency);
CREATE INDEX idx_p042_st_gl_account         ON ghg_accounting_scope3.spend_transactions(gl_account);
CREATE INDEX idx_p042_st_naics              ON ghg_accounting_scope3.spend_transactions(naics_code);
CREATE INDEX idx_p042_st_classified         ON ghg_accounting_scope3.spend_transactions(is_classified);
CREATE INDEX idx_p042_st_excluded           ON ghg_accounting_scope3.spend_transactions(is_excluded);
CREATE INDEX idx_p042_st_source_record      ON ghg_accounting_scope3.spend_transactions(source_system, source_record_id);
CREATE INDEX idx_p042_st_created            ON ghg_accounting_scope3.spend_transactions(created_at DESC);
CREATE INDEX idx_p042_st_metadata           ON ghg_accounting_scope3.spend_transactions USING GIN(metadata);

-- Composite: inventory + unclassified for classification queue
CREATE INDEX idx_p042_st_inv_unclassified   ON ghg_accounting_scope3.spend_transactions(inventory_id, amount DESC)
    WHERE is_classified = false AND is_excluded = false;

-- Composite: inventory + vendor for vendor analysis
CREATE INDEX idx_p042_st_inv_vendor         ON ghg_accounting_scope3.spend_transactions(inventory_id, vendor_name);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_st_updated
    BEFORE UPDATE ON ghg_accounting_scope3.spend_transactions
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_accounting_scope3.classification_results
-- =============================================================================
-- Classification output for each spend transaction. Maps the transaction to
-- a Scope 3 category, EEIO sector, and records the classification method
-- and confidence score. Supports both automated (ML/rule-based) and manual
-- classification with override tracking.

CREATE TABLE ghg_accounting_scope3.classification_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    transaction_id              UUID            NOT NULL REFERENCES ghg_accounting_scope3.spend_transactions(id) ON DELETE CASCADE,
    -- Classification output
    scope3_category             ghg_accounting_scope3.scope3_category_type NOT NULL,
    eeio_sector                 VARCHAR(100)    NOT NULL,
    eeio_sector_code            VARCHAR(50),
    -- Confidence
    confidence_score            DECIMAL(3,2)    NOT NULL DEFAULT 0.50,
    classification_method       VARCHAR(30)     NOT NULL DEFAULT 'RULE_BASED',
    -- Override
    is_manual_override          BOOLEAN         NOT NULL DEFAULT false,
    original_category           ghg_accounting_scope3.scope3_category_type,
    original_eeio_sector        VARCHAR(100),
    override_by                 VARCHAR(255),
    override_reason             TEXT,
    -- Calculated emissions
    emission_factor_used        DECIMAL(12,6),
    emission_factor_unit        VARCHAR(50),
    calculated_tco2e            DECIMAL(12,3),
    -- Metadata
    classification_timestamp    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    model_version               VARCHAR(50),
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_cr_confidence CHECK (
        confidence_score >= 0 AND confidence_score <= 1
    ),
    CONSTRAINT chk_p042_cr_method CHECK (
        classification_method IN (
            'RULE_BASED', 'ML_CLASSIFIER', 'NAICS_LOOKUP',
            'GL_MAPPING', 'KEYWORD_MATCH', 'MANUAL', 'HYBRID'
        )
    ),
    CONSTRAINT chk_p042_cr_ef CHECK (
        emission_factor_used IS NULL OR emission_factor_used >= 0
    ),
    CONSTRAINT chk_p042_cr_tco2e CHECK (
        calculated_tco2e IS NULL OR calculated_tco2e >= 0
    ),
    CONSTRAINT uq_p042_cr_transaction UNIQUE (transaction_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_cr_tenant             ON ghg_accounting_scope3.classification_results(tenant_id);
CREATE INDEX idx_p042_cr_transaction        ON ghg_accounting_scope3.classification_results(transaction_id);
CREATE INDEX idx_p042_cr_category           ON ghg_accounting_scope3.classification_results(scope3_category);
CREATE INDEX idx_p042_cr_eeio_sector        ON ghg_accounting_scope3.classification_results(eeio_sector);
CREATE INDEX idx_p042_cr_eeio_code          ON ghg_accounting_scope3.classification_results(eeio_sector_code);
CREATE INDEX idx_p042_cr_confidence         ON ghg_accounting_scope3.classification_results(confidence_score);
CREATE INDEX idx_p042_cr_method             ON ghg_accounting_scope3.classification_results(classification_method);
CREATE INDEX idx_p042_cr_manual             ON ghg_accounting_scope3.classification_results(is_manual_override) WHERE is_manual_override = true;
CREATE INDEX idx_p042_cr_tco2e              ON ghg_accounting_scope3.classification_results(calculated_tco2e DESC);
CREATE INDEX idx_p042_cr_created            ON ghg_accounting_scope3.classification_results(created_at DESC);
CREATE INDEX idx_p042_cr_metadata           ON ghg_accounting_scope3.classification_results USING GIN(metadata);

-- Composite: category + confidence for review queue
CREATE INDEX idx_p042_cr_cat_confidence     ON ghg_accounting_scope3.classification_results(scope3_category, confidence_score)
    WHERE confidence_score < 0.70;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_cr_updated
    BEFORE UPDATE ON ghg_accounting_scope3.classification_results
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_accounting_scope3.eeio_sector_factors
-- =============================================================================
-- EEIO (Environmentally Extended Input-Output) emission factor intensities
-- by economic sector. Sources include Exiobase 3, USEEIO, and CEDA. Each
-- factor represents the GHG intensity per unit of economic output (typically
-- kgCO2e per USD or EUR). Used for Tier 1 spend-based calculations.

CREATE TABLE ghg_accounting_scope3.eeio_sector_factors (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Sector identification
    sector_code                 VARCHAR(50)     NOT NULL,
    sector_name                 VARCHAR(500)    NOT NULL,
    sector_description          TEXT,
    parent_sector_code          VARCHAR(50),
    -- Model source
    model                       VARCHAR(50)     NOT NULL,
    model_version               VARCHAR(20),
    -- Factor
    kgco2e_per_unit             DECIMAL(12,6)   NOT NULL,
    currency                    VARCHAR(3)      NOT NULL DEFAULT 'USD',
    unit_description            VARCHAR(100)    NOT NULL DEFAULT 'kgCO2e per USD spent',
    -- Gas breakdown
    co2_pct                     DECIMAL(5,2),
    ch4_pct                     DECIMAL(5,2),
    n2o_pct                     DECIMAL(5,2),
    other_ghg_pct               DECIMAL(5,2),
    -- Metadata
    year                        INTEGER         NOT NULL,
    geography                   VARCHAR(100)    NOT NULL DEFAULT 'GLOBAL',
    source_reference            VARCHAR(500),
    source_table                VARCHAR(100),
    uncertainty_pct             DECIMAL(8,4),
    data_quality_score          NUMERIC(5,2),
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_ef_kgco2e CHECK (
        kgco2e_per_unit >= 0
    ),
    CONSTRAINT chk_p042_ef_model CHECK (
        model IN (
            'EXIOBASE_3', 'USEEIO', 'CEDA', 'GTAP', 'WIOD',
            'EORA', 'OECD_ICIO', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p042_ef_year CHECK (
        year >= 1990 AND year <= 2100
    ),
    CONSTRAINT chk_p042_ef_currency_len CHECK (
        LENGTH(currency) = 3
    ),
    CONSTRAINT chk_p042_ef_uncertainty CHECK (
        uncertainty_pct IS NULL OR (uncertainty_pct >= 0 AND uncertainty_pct <= 500)
    ),
    CONSTRAINT chk_p042_ef_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p042_ef_gas_pcts CHECK (
        (COALESCE(co2_pct, 0) + COALESCE(ch4_pct, 0) + COALESCE(n2o_pct, 0) + COALESCE(other_ghg_pct, 0)) <= 101
    ),
    CONSTRAINT uq_p042_ef_sector_model_year UNIQUE (sector_code, model, year, geography)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_ef_sector_code        ON ghg_accounting_scope3.eeio_sector_factors(sector_code);
CREATE INDEX idx_p042_ef_sector_name        ON ghg_accounting_scope3.eeio_sector_factors(sector_name);
CREATE INDEX idx_p042_ef_model              ON ghg_accounting_scope3.eeio_sector_factors(model);
CREATE INDEX idx_p042_ef_year               ON ghg_accounting_scope3.eeio_sector_factors(year);
CREATE INDEX idx_p042_ef_geography          ON ghg_accounting_scope3.eeio_sector_factors(geography);
CREATE INDEX idx_p042_ef_active             ON ghg_accounting_scope3.eeio_sector_factors(is_active) WHERE is_active = true;
CREATE INDEX idx_p042_ef_kgco2e             ON ghg_accounting_scope3.eeio_sector_factors(kgco2e_per_unit DESC);
CREATE INDEX idx_p042_ef_created            ON ghg_accounting_scope3.eeio_sector_factors(created_at DESC);

-- Composite: sector + model + active for factor lookup
CREATE INDEX idx_p042_ef_sector_model_active ON ghg_accounting_scope3.eeio_sector_factors(sector_code, model, year DESC)
    WHERE is_active = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_ef_updated
    BEFORE UPDATE ON ghg_accounting_scope3.eeio_sector_factors
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_accounting_scope3.naics_category_mapping
-- =============================================================================
-- Reference lookup mapping NAICS codes to default Scope 3 categories.
-- Provides a fast mapping from vendor NAICS classification to the most
-- likely Scope 3 category. Used as a first-pass classification before
-- more sophisticated ML or rule-based methods are applied.

CREATE TABLE ghg_accounting_scope3.naics_category_mapping (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    naics_code                  VARCHAR(10)     NOT NULL,
    naics_description           VARCHAR(500)    NOT NULL,
    naics_level                 INTEGER         NOT NULL DEFAULT 6,
    -- Mapping
    primary_scope3_category     ghg_accounting_scope3.scope3_category_type NOT NULL,
    secondary_scope3_category   ghg_accounting_scope3.scope3_category_type,
    default_eeio_sector         VARCHAR(100),
    default_eeio_sector_code    VARCHAR(50),
    -- Confidence
    mapping_confidence          DECIMAL(3,2)    NOT NULL DEFAULT 0.80,
    mapping_source              VARCHAR(100)    DEFAULT 'GHG_PROTOCOL_GUIDANCE',
    -- Metadata
    notes                       TEXT,
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_ncm_naics_level CHECK (
        naics_level >= 2 AND naics_level <= 6
    ),
    CONSTRAINT chk_p042_ncm_confidence CHECK (
        mapping_confidence >= 0 AND mapping_confidence <= 1
    ),
    CONSTRAINT uq_p042_ncm_naics UNIQUE (naics_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_ncm_naics             ON ghg_accounting_scope3.naics_category_mapping(naics_code);
CREATE INDEX idx_p042_ncm_category          ON ghg_accounting_scope3.naics_category_mapping(primary_scope3_category);
CREATE INDEX idx_p042_ncm_secondary         ON ghg_accounting_scope3.naics_category_mapping(secondary_scope3_category);
CREATE INDEX idx_p042_ncm_level             ON ghg_accounting_scope3.naics_category_mapping(naics_level);
CREATE INDEX idx_p042_ncm_active            ON ghg_accounting_scope3.naics_category_mapping(is_active) WHERE is_active = true;
CREATE INDEX idx_p042_ncm_eeio              ON ghg_accounting_scope3.naics_category_mapping(default_eeio_sector_code);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_ncm_updated
    BEFORE UPDATE ON ghg_accounting_scope3.naics_category_mapping
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_accounting_scope3.spend_transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.classification_results ENABLE ROW LEVEL SECURITY;
-- eeio_sector_factors and naics_category_mapping are reference data; no RLS needed

CREATE POLICY p042_st_tenant_isolation
    ON ghg_accounting_scope3.spend_transactions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_st_service_bypass
    ON ghg_accounting_scope3.spend_transactions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_cr_tenant_isolation
    ON ghg_accounting_scope3.classification_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_cr_service_bypass
    ON ghg_accounting_scope3.classification_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.spend_transactions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.classification_results TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.eeio_sector_factors TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.naics_category_mapping TO PUBLIC;

-- =============================================================================
-- Seed Data: NAICS to Scope 3 Category Mappings
-- =============================================================================
-- 55 common NAICS codes mapped to their most likely Scope 3 category.
-- Source: GHG Protocol Scope 3 Standard guidance and industry practice.

INSERT INTO ghg_accounting_scope3.naics_category_mapping
    (naics_code, naics_description, naics_level, primary_scope3_category, secondary_scope3_category, default_eeio_sector, mapping_confidence, mapping_source)
VALUES
    -- Manufacturing (Cat 1: Purchased Goods and Services)
    ('311', 'Food Manufacturing', 3, 'CAT_1', NULL, 'Food products', 0.90, 'GHG_PROTOCOL_GUIDANCE'),
    ('312', 'Beverage and Tobacco Manufacturing', 3, 'CAT_1', NULL, 'Beverages', 0.90, 'GHG_PROTOCOL_GUIDANCE'),
    ('313', 'Textile Mills', 3, 'CAT_1', NULL, 'Textiles', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    ('314', 'Textile Product Mills', 3, 'CAT_1', NULL, 'Textile products', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    ('315', 'Apparel Manufacturing', 3, 'CAT_1', NULL, 'Wearing apparel', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    ('321', 'Wood Product Manufacturing', 3, 'CAT_1', NULL, 'Wood products', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    ('322', 'Paper Manufacturing', 3, 'CAT_1', NULL, 'Paper products', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    ('323', 'Printing and Related Support', 3, 'CAT_1', NULL, 'Printing services', 0.80, 'GHG_PROTOCOL_GUIDANCE'),
    ('324', 'Petroleum and Coal Products', 3, 'CAT_1', 'CAT_3', 'Petroleum products', 0.70, 'GHG_PROTOCOL_GUIDANCE'),
    ('325', 'Chemical Manufacturing', 3, 'CAT_1', NULL, 'Chemical products', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    ('326', 'Plastics and Rubber Products', 3, 'CAT_1', NULL, 'Rubber and plastics', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    ('327', 'Nonmetallic Mineral Products', 3, 'CAT_1', NULL, 'Mineral products', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    ('331', 'Primary Metal Manufacturing', 3, 'CAT_1', NULL, 'Basic metals', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    ('332', 'Fabricated Metal Products', 3, 'CAT_1', NULL, 'Metal products', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    ('333', 'Machinery Manufacturing', 3, 'CAT_1', 'CAT_2', 'Machinery', 0.70, 'GHG_PROTOCOL_GUIDANCE'),
    ('334', 'Computer and Electronic Products', 3, 'CAT_1', 'CAT_2', 'Electronics', 0.70, 'GHG_PROTOCOL_GUIDANCE'),
    ('335', 'Electrical Equipment and Appliances', 3, 'CAT_1', 'CAT_2', 'Electrical equipment', 0.70, 'GHG_PROTOCOL_GUIDANCE'),
    ('336', 'Transportation Equipment', 3, 'CAT_2', 'CAT_1', 'Transport equipment', 0.75, 'GHG_PROTOCOL_GUIDANCE'),
    ('337', 'Furniture and Related Products', 3, 'CAT_1', 'CAT_2', 'Furniture', 0.75, 'GHG_PROTOCOL_GUIDANCE'),
    ('339', 'Miscellaneous Manufacturing', 3, 'CAT_1', NULL, 'Other manufacturing', 0.75, 'GHG_PROTOCOL_GUIDANCE'),
    -- Capital Goods (Cat 2)
    ('236', 'Construction of Buildings', 3, 'CAT_2', NULL, 'Construction', 0.90, 'GHG_PROTOCOL_GUIDANCE'),
    ('237', 'Heavy and Civil Engineering', 3, 'CAT_2', NULL, 'Civil engineering', 0.90, 'GHG_PROTOCOL_GUIDANCE'),
    ('238', 'Specialty Trade Contractors', 3, 'CAT_2', 'CAT_1', 'Construction services', 0.80, 'GHG_PROTOCOL_GUIDANCE'),
    -- Fuel and Energy (Cat 3)
    ('211', 'Oil and Gas Extraction', 3, 'CAT_3', NULL, 'Oil and gas', 0.90, 'GHG_PROTOCOL_GUIDANCE'),
    ('213', 'Mining Support Activities', 3, 'CAT_3', 'CAT_1', 'Mining support', 0.75, 'GHG_PROTOCOL_GUIDANCE'),
    ('221', 'Utilities', 3, 'CAT_3', NULL, 'Electricity and gas', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    -- Transportation (Cat 4: Upstream T&D)
    ('481', 'Air Transportation', 3, 'CAT_4', 'CAT_6', 'Air transport', 0.75, 'GHG_PROTOCOL_GUIDANCE'),
    ('482', 'Rail Transportation', 3, 'CAT_4', NULL, 'Rail transport', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    ('483', 'Water Transportation', 3, 'CAT_4', NULL, 'Water transport', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    ('484', 'Truck Transportation', 3, 'CAT_4', NULL, 'Road freight', 0.90, 'GHG_PROTOCOL_GUIDANCE'),
    ('488', 'Transportation Support', 3, 'CAT_4', NULL, 'Transport support', 0.80, 'GHG_PROTOCOL_GUIDANCE'),
    ('491', 'Postal Service', 3, 'CAT_4', NULL, 'Postal services', 0.80, 'GHG_PROTOCOL_GUIDANCE'),
    ('492', 'Couriers and Messengers', 3, 'CAT_4', NULL, 'Courier services', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    ('493', 'Warehousing and Storage', 3, 'CAT_4', NULL, 'Warehousing', 0.80, 'GHG_PROTOCOL_GUIDANCE'),
    -- Waste (Cat 5)
    ('562', 'Waste Management and Remediation', 3, 'CAT_5', NULL, 'Waste services', 0.95, 'GHG_PROTOCOL_GUIDANCE'),
    -- Business Travel (Cat 6)
    ('721', 'Accommodation', 3, 'CAT_6', NULL, 'Accommodation', 0.90, 'GHG_PROTOCOL_GUIDANCE'),
    ('561510', 'Travel Arrangement Services', 6, 'CAT_6', NULL, 'Travel services', 0.95, 'GHG_PROTOCOL_GUIDANCE'),
    ('485', 'Transit and Ground Passenger', 3, 'CAT_6', 'CAT_7', 'Passenger transport', 0.70, 'GHG_PROTOCOL_GUIDANCE'),
    -- Services (typically Cat 1)
    ('511', 'Publishing Industries', 3, 'CAT_1', NULL, 'Publishing', 0.80, 'GHG_PROTOCOL_GUIDANCE'),
    ('512', 'Motion Picture and Sound', 3, 'CAT_1', NULL, 'Media services', 0.80, 'GHG_PROTOCOL_GUIDANCE'),
    ('515', 'Broadcasting', 3, 'CAT_1', NULL, 'Broadcasting', 0.80, 'GHG_PROTOCOL_GUIDANCE'),
    ('517', 'Telecommunications', 3, 'CAT_1', NULL, 'Telecom services', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    ('518', 'Data Processing and Hosting', 3, 'CAT_1', NULL, 'IT services', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    ('541', 'Professional and Technical Services', 3, 'CAT_1', NULL, 'Professional services', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    ('561', 'Administrative Support Services', 3, 'CAT_1', NULL, 'Admin services', 0.80, 'GHG_PROTOCOL_GUIDANCE'),
    -- Real estate / leased assets (Cat 8 or Cat 13)
    ('531', 'Real Estate', 3, 'CAT_8', 'CAT_13', 'Real estate services', 0.75, 'GHG_PROTOCOL_GUIDANCE'),
    -- Financial / Investments (Cat 15)
    ('522', 'Credit Intermediation', 3, 'CAT_15', NULL, 'Financial services', 0.80, 'GHG_PROTOCOL_GUIDANCE'),
    ('523', 'Securities and Investments', 3, 'CAT_15', NULL, 'Investment services', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    ('524', 'Insurance', 3, 'CAT_15', 'CAT_1', 'Insurance services', 0.70, 'GHG_PROTOCOL_GUIDANCE'),
    ('525', 'Funds, Trusts, Financial Vehicles', 3, 'CAT_15', NULL, 'Investment funds', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    -- Agriculture (Cat 1)
    ('111', 'Crop Production', 3, 'CAT_1', NULL, 'Agriculture crops', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    ('112', 'Animal Production and Aquaculture', 3, 'CAT_1', NULL, 'Animal products', 0.85, 'GHG_PROTOCOL_GUIDANCE'),
    -- Mining (Cat 1)
    ('212', 'Mining (except Oil and Gas)', 3, 'CAT_1', NULL, 'Mining products', 0.80, 'GHG_PROTOCOL_GUIDANCE'),
    -- Retail/Wholesale (Cat 9: Downstream T&D for product companies)
    ('423', 'Merchant Wholesalers Durable', 3, 'CAT_1', 'CAT_9', 'Wholesale trade', 0.65, 'GHG_PROTOCOL_GUIDANCE'),
    ('424', 'Merchant Wholesalers Nondurable', 3, 'CAT_1', 'CAT_9', 'Wholesale trade', 0.65, 'GHG_PROTOCOL_GUIDANCE');

-- =============================================================================
-- Seed Data: EEIO Sector Emission Intensities (Exiobase 3 Top Sectors)
-- =============================================================================
-- 35 high-level EEIO sectors with emission intensities from Exiobase 3.
-- Values are in kgCO2e per USD (purchasing power parity adjusted).
-- Source: Exiobase 3 (2024 release), global averages.

INSERT INTO ghg_accounting_scope3.eeio_sector_factors
    (sector_code, sector_name, model, model_version, kgco2e_per_unit, currency, unit_description, year, geography, source_reference, co2_pct, ch4_pct, n2o_pct, other_ghg_pct, provenance_hash)
VALUES
    ('EXIO_01', 'Agriculture crops', 'EXIOBASE_3', '3.8', 0.680, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 45.0, 35.0, 18.0, 2.0, 'seed_exio_01'),
    ('EXIO_02', 'Animal products', 'EXIOBASE_3', '3.8', 1.250, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 30.0, 45.0, 22.0, 3.0, 'seed_exio_02'),
    ('EXIO_03', 'Food products', 'EXIOBASE_3', '3.8', 0.520, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 55.0, 28.0, 15.0, 2.0, 'seed_exio_03'),
    ('EXIO_04', 'Beverages', 'EXIOBASE_3', '3.8', 0.350, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 60.0, 22.0, 16.0, 2.0, 'seed_exio_04'),
    ('EXIO_05', 'Textiles', 'EXIOBASE_3', '3.8', 0.410, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 70.0, 12.0, 8.0, 10.0, 'seed_exio_05'),
    ('EXIO_06', 'Wearing apparel', 'EXIOBASE_3', '3.8', 0.380, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 68.0, 14.0, 10.0, 8.0, 'seed_exio_06'),
    ('EXIO_07', 'Wood products', 'EXIOBASE_3', '3.8', 0.290, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 75.0, 10.0, 8.0, 7.0, 'seed_exio_07'),
    ('EXIO_08', 'Paper products', 'EXIOBASE_3', '3.8', 0.450, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 78.0, 8.0, 6.0, 8.0, 'seed_exio_08'),
    ('EXIO_09', 'Petroleum products', 'EXIOBASE_3', '3.8', 1.850, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 92.0, 5.0, 1.0, 2.0, 'seed_exio_09'),
    ('EXIO_10', 'Chemical products', 'EXIOBASE_3', '3.8', 0.750, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 72.0, 10.0, 5.0, 13.0, 'seed_exio_10'),
    ('EXIO_11', 'Rubber and plastics', 'EXIOBASE_3', '3.8', 0.580, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 80.0, 8.0, 4.0, 8.0, 'seed_exio_11'),
    ('EXIO_12', 'Mineral products', 'EXIOBASE_3', '3.8', 1.120, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 88.0, 5.0, 3.0, 4.0, 'seed_exio_12'),
    ('EXIO_13', 'Basic metals', 'EXIOBASE_3', '3.8', 1.430, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 85.0, 6.0, 3.0, 6.0, 'seed_exio_13'),
    ('EXIO_14', 'Metal products', 'EXIOBASE_3', '3.8', 0.620, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 82.0, 7.0, 4.0, 7.0, 'seed_exio_14'),
    ('EXIO_15', 'Machinery', 'EXIOBASE_3', '3.8', 0.420, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 80.0, 8.0, 4.0, 8.0, 'seed_exio_15'),
    ('EXIO_16', 'Electronics', 'EXIOBASE_3', '3.8', 0.350, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 75.0, 5.0, 3.0, 17.0, 'seed_exio_16'),
    ('EXIO_17', 'Electrical equipment', 'EXIOBASE_3', '3.8', 0.440, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 78.0, 7.0, 4.0, 11.0, 'seed_exio_17'),
    ('EXIO_18', 'Transport equipment', 'EXIOBASE_3', '3.8', 0.480, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 80.0, 7.0, 4.0, 9.0, 'seed_exio_18'),
    ('EXIO_19', 'Construction', 'EXIOBASE_3', '3.8', 0.530, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 82.0, 6.0, 4.0, 8.0, 'seed_exio_19'),
    ('EXIO_20', 'Electricity and gas', 'EXIOBASE_3', '3.8', 2.150, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 90.0, 6.0, 2.0, 2.0, 'seed_exio_20'),
    ('EXIO_21', 'Road freight', 'EXIOBASE_3', '3.8', 0.810, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 92.0, 3.0, 2.0, 3.0, 'seed_exio_21'),
    ('EXIO_22', 'Rail transport', 'EXIOBASE_3', '3.8', 0.350, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 85.0, 4.0, 3.0, 8.0, 'seed_exio_22'),
    ('EXIO_23', 'Water transport', 'EXIOBASE_3', '3.8', 0.720, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 90.0, 4.0, 3.0, 3.0, 'seed_exio_23'),
    ('EXIO_24', 'Air transport', 'EXIOBASE_3', '3.8', 1.350, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 95.0, 2.0, 1.0, 2.0, 'seed_exio_24'),
    ('EXIO_25', 'Warehousing', 'EXIOBASE_3', '3.8', 0.280, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 80.0, 5.0, 3.0, 12.0, 'seed_exio_25'),
    ('EXIO_26', 'Waste services', 'EXIOBASE_3', '3.8', 0.950, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 40.0, 45.0, 10.0, 5.0, 'seed_exio_26'),
    ('EXIO_27', 'Professional services', 'EXIOBASE_3', '3.8', 0.120, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 80.0, 6.0, 4.0, 10.0, 'seed_exio_27'),
    ('EXIO_28', 'IT services', 'EXIOBASE_3', '3.8', 0.150, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 78.0, 5.0, 3.0, 14.0, 'seed_exio_28'),
    ('EXIO_29', 'Financial services', 'EXIOBASE_3', '3.8', 0.080, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 82.0, 5.0, 3.0, 10.0, 'seed_exio_29'),
    ('EXIO_30', 'Real estate services', 'EXIOBASE_3', '3.8', 0.180, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 78.0, 8.0, 5.0, 9.0, 'seed_exio_30'),
    ('EXIO_31', 'Accommodation', 'EXIOBASE_3', '3.8', 0.320, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 75.0, 10.0, 6.0, 9.0, 'seed_exio_31'),
    ('EXIO_32', 'Telecom services', 'EXIOBASE_3', '3.8', 0.140, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 78.0, 5.0, 3.0, 14.0, 'seed_exio_32'),
    ('EXIO_33', 'Printing services', 'EXIOBASE_3', '3.8', 0.310, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 76.0, 8.0, 5.0, 11.0, 'seed_exio_33'),
    ('EXIO_34', 'Furniture', 'EXIOBASE_3', '3.8', 0.340, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 74.0, 9.0, 6.0, 11.0, 'seed_exio_34'),
    ('EXIO_35', 'Other manufacturing', 'EXIOBASE_3', '3.8', 0.400, 'USD', 'kgCO2e per USD spent', 2024, 'GLOBAL', 'Exiobase 3.8 (2024) Satellite Accounts', 76.0, 8.0, 5.0, 11.0, 'seed_exio_35');

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_accounting_scope3.spend_transactions IS
    'Raw procurement transactions from ERP/finance systems for spend-based (Tier 1) Scope 3 calculations with vendor, amount, GL account, and NAICS code.';
COMMENT ON TABLE ghg_accounting_scope3.classification_results IS
    'Classification output mapping spend transactions to Scope 3 categories and EEIO sectors with confidence scores and optional manual overrides.';
COMMENT ON TABLE ghg_accounting_scope3.eeio_sector_factors IS
    'EEIO emission factor intensities (kgCO2e per currency unit) by economic sector from Exiobase, USEEIO, and other IO models.';
COMMENT ON TABLE ghg_accounting_scope3.naics_category_mapping IS
    'Reference lookup mapping NAICS codes to default Scope 3 categories for first-pass transaction classification.';

COMMENT ON COLUMN ghg_accounting_scope3.spend_transactions.amount_usd IS 'Transaction amount converted to USD for standardized EEIO factor application.';
COMMENT ON COLUMN ghg_accounting_scope3.spend_transactions.naics_code IS 'NAICS code for vendor classification, used for initial category mapping.';
COMMENT ON COLUMN ghg_accounting_scope3.spend_transactions.unspsc_code IS 'United Nations Standard Products and Services Code for commodity classification.';

COMMENT ON COLUMN ghg_accounting_scope3.classification_results.confidence_score IS 'Classification confidence (0-1). Scores below 0.70 are flagged for manual review.';
COMMENT ON COLUMN ghg_accounting_scope3.classification_results.classification_method IS 'Method used: RULE_BASED, ML_CLASSIFIER, NAICS_LOOKUP, GL_MAPPING, KEYWORD_MATCH, MANUAL, HYBRID.';

COMMENT ON COLUMN ghg_accounting_scope3.eeio_sector_factors.model IS 'EEIO model source: EXIOBASE_3, USEEIO, CEDA, GTAP, WIOD, EORA, OECD_ICIO.';
COMMENT ON COLUMN ghg_accounting_scope3.eeio_sector_factors.kgco2e_per_unit IS 'GHG intensity in kgCO2e per unit of economic output (currency).';

COMMENT ON COLUMN ghg_accounting_scope3.naics_category_mapping.primary_scope3_category IS 'Most likely Scope 3 category for this NAICS code.';
COMMENT ON COLUMN ghg_accounting_scope3.naics_category_mapping.mapping_confidence IS 'Confidence of the NAICS-to-category mapping (0-1).';
