-- =============================================================================
-- V347: PACK-043 Scope 3 Complete Pack - LCA Integration & Product Carbon Footprints
-- =============================================================================
-- Pack:         PACK-043 (Scope 3 Complete Pack)
-- Migration:    002 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates product lifecycle assessment (LCA) tables for cradle-to-grave
-- carbon footprinting. Supports product master data, bill of materials (BOM)
-- with per-component emission factors and supplier attribution, lifecycle
-- stage results per ISO 14040/14044, and aggregated product carbon footprint
-- (PCF) calculations per ISO 14067 and the PACT Pathfinder framework.
--
-- These tables enable granular product-level calculations for Category 1
-- (Purchased Goods), Category 11 (Use of Sold Products), and Category 12
-- (End-of-Life Treatment) while supporting the PACT data exchange standard.
--
-- Tables (4):
--   1. ghg_accounting_scope3_complete.products
--   2. ghg_accounting_scope3_complete.product_bom
--   3. ghg_accounting_scope3_complete.lifecycle_results
--   4. ghg_accounting_scope3_complete.product_carbon_footprints
--
-- Enums (1):
--   1. ghg_accounting_scope3_complete.lifecycle_stage
--
-- Also includes: indexes, RLS, comments.
-- Previous: V346__pack043_core_enterprise_schema.sql
-- =============================================================================

SET search_path TO ghg_accounting_scope3_complete, public;

-- ---------------------------------------------------------------------------
-- Enum: lifecycle_stage
-- ---------------------------------------------------------------------------
-- Product lifecycle stages per ISO 14040/14044 framework covering the full
-- cradle-to-grave boundary.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'lifecycle_stage' AND typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'ghg_accounting_scope3_complete')) THEN
        CREATE TYPE ghg_accounting_scope3_complete.lifecycle_stage AS ENUM (
            'RAW_MATERIAL_EXTRACTION',  -- A1: Raw material supply
            'MANUFACTURING',            -- A2-A3: Transport + manufacturing
            'DISTRIBUTION',             -- A4: Distribution to customer
            'USE_PHASE',                -- B1-B7: Use, maintenance, repair
            'END_OF_LIFE',              -- C1-C4: Deconstruction, disposal
            'RECYCLING_BENEFIT',        -- D: Benefits beyond system boundary
            'PACKAGING',                -- Packaging production and disposal
            'UPSTREAM_TRANSPORT',       -- Transport of materials to factory
            'DOWNSTREAM_TRANSPORT'      -- Transport from factory to end user
        );
    END IF;
END;
$$;

-- =============================================================================
-- Table 1: ghg_accounting_scope3_complete.products
-- =============================================================================
-- Product master data for LCA-based Scope 3 calculations. Each record
-- represents a product or service with classification, functional unit,
-- annual production volume, and revenue. Products are the anchor for BOM,
-- lifecycle results, and PCF calculations.

CREATE TABLE ghg_accounting_scope3_complete.products (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    -- Identification
    product_id                  VARCHAR(100)    NOT NULL,
    name                        VARCHAR(500)    NOT NULL,
    description                 TEXT,
    -- Classification
    category                    VARCHAR(200),
    sub_category                VARCHAR(200),
    product_group               VARCHAR(200),
    hs_code                     VARCHAR(20),
    cpc_code                    VARCHAR(20),
    -- Functional unit (ISO 14040)
    unit                        VARCHAR(50)     NOT NULL DEFAULT 'unit',
    functional_unit             VARCHAR(200),
    reference_flow              VARCHAR(200),
    -- Volume and revenue
    annual_volume               DECIMAL(15,3),
    volume_unit                 VARCHAR(50),
    revenue                     NUMERIC(18,2),
    revenue_currency            VARCHAR(3)      DEFAULT 'USD',
    revenue_pct_of_total        DECIMAL(5,2),
    -- Physical
    mass_per_unit_kg            DECIMAL(12,6),
    lifetime_years              DECIMAL(6,2),
    energy_consumption_kwh      DECIMAL(12,3),
    -- Supply chain
    primary_supplier_id         UUID,
    manufacturing_country       VARCHAR(3),
    -- Status
    active                      BOOLEAN         NOT NULL DEFAULT true,
    lca_available               BOOLEAN         NOT NULL DEFAULT false,
    pcf_available               BOOLEAN         NOT NULL DEFAULT false,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p043_prod_volume CHECK (
        annual_volume IS NULL OR annual_volume >= 0
    ),
    CONSTRAINT chk_p043_prod_revenue CHECK (
        revenue IS NULL OR revenue >= 0
    ),
    CONSTRAINT chk_p043_prod_rev_pct CHECK (
        revenue_pct_of_total IS NULL OR (revenue_pct_of_total >= 0 AND revenue_pct_of_total <= 100)
    ),
    CONSTRAINT chk_p043_prod_mass CHECK (
        mass_per_unit_kg IS NULL OR mass_per_unit_kg >= 0
    ),
    CONSTRAINT chk_p043_prod_lifetime CHECK (
        lifetime_years IS NULL OR lifetime_years > 0
    ),
    CONSTRAINT chk_p043_prod_energy CHECK (
        energy_consumption_kwh IS NULL OR energy_consumption_kwh >= 0
    ),
    CONSTRAINT uq_p043_prod_tenant_product UNIQUE (tenant_id, product_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_prod_tenant           ON ghg_accounting_scope3_complete.products(tenant_id);
CREATE INDEX idx_p043_prod_product_id       ON ghg_accounting_scope3_complete.products(product_id);
CREATE INDEX idx_p043_prod_name             ON ghg_accounting_scope3_complete.products(name);
CREATE INDEX idx_p043_prod_category         ON ghg_accounting_scope3_complete.products(category);
CREATE INDEX idx_p043_prod_group            ON ghg_accounting_scope3_complete.products(product_group);
CREATE INDEX idx_p043_prod_hs_code          ON ghg_accounting_scope3_complete.products(hs_code);
CREATE INDEX idx_p043_prod_revenue          ON ghg_accounting_scope3_complete.products(revenue DESC);
CREATE INDEX idx_p043_prod_active           ON ghg_accounting_scope3_complete.products(active) WHERE active = true;
CREATE INDEX idx_p043_prod_lca              ON ghg_accounting_scope3_complete.products(lca_available) WHERE lca_available = true;
CREATE INDEX idx_p043_prod_pcf              ON ghg_accounting_scope3_complete.products(pcf_available) WHERE pcf_available = true;
CREATE INDEX idx_p043_prod_country          ON ghg_accounting_scope3_complete.products(manufacturing_country);
CREATE INDEX idx_p043_prod_created          ON ghg_accounting_scope3_complete.products(created_at DESC);
CREATE INDEX idx_p043_prod_metadata         ON ghg_accounting_scope3_complete.products USING GIN(metadata);

-- Composite: tenant + category + revenue
CREATE INDEX idx_p043_prod_tenant_cat       ON ghg_accounting_scope3_complete.products(tenant_id, category, revenue DESC)
    WHERE active = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_prod_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.products
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_accounting_scope3_complete.product_bom
-- =============================================================================
-- Bill of materials per product. Each row is a component or raw material
-- input with mass, supplier attribution, emission factor, and transport.
-- BOM data drives cradle-to-gate LCA and enables Tier 3 (supplier-specific)
-- calculations for Category 1 (Purchased Goods and Services).

CREATE TABLE ghg_accounting_scope3_complete.product_bom (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    product_id                  UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.products(id) ON DELETE CASCADE,
    -- Component
    component_name              VARCHAR(500)    NOT NULL,
    component_code              VARCHAR(100),
    bom_level                   INTEGER         NOT NULL DEFAULT 1,
    parent_component_id         UUID,
    -- Material
    material                    VARCHAR(200)    NOT NULL,
    material_category           VARCHAR(200),
    recycled_content_pct        DECIMAL(5,2)    DEFAULT 0,
    -- Quantity
    mass_kg                     DECIMAL(12,6)   NOT NULL,
    quantity_per_unit            DECIMAL(12,6)   NOT NULL DEFAULT 1,
    wastage_pct                 DECIMAL(5,2)    DEFAULT 0,
    -- Supplier
    supplier_id                 UUID,
    supplier_name               VARCHAR(500),
    origin_country              VARCHAR(3),
    -- Emission factor
    emission_factor             DECIMAL(12,6),
    ef_unit                     VARCHAR(50)     DEFAULT 'kgCO2e/kg',
    ef_source                   VARCHAR(200),
    ef_source_year              INTEGER,
    ef_confidence               DECIMAL(3,2),
    -- Calculated emissions
    component_tco2e             DECIMAL(15,6),
    pct_of_product              DECIMAL(5,2),
    -- Transport to manufacturing
    transport_distance_km       DECIMAL(10,2),
    transport_mode              VARCHAR(50),
    transport_tco2e             DECIMAL(12,6),
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_bom_level CHECK (
        bom_level >= 1 AND bom_level <= 20
    ),
    CONSTRAINT chk_p043_bom_mass CHECK (
        mass_kg >= 0
    ),
    CONSTRAINT chk_p043_bom_qty CHECK (
        quantity_per_unit > 0
    ),
    CONSTRAINT chk_p043_bom_recycled CHECK (
        recycled_content_pct IS NULL OR (recycled_content_pct >= 0 AND recycled_content_pct <= 100)
    ),
    CONSTRAINT chk_p043_bom_wastage CHECK (
        wastage_pct IS NULL OR (wastage_pct >= 0 AND wastage_pct <= 100)
    ),
    CONSTRAINT chk_p043_bom_ef CHECK (
        emission_factor IS NULL OR emission_factor >= 0
    ),
    CONSTRAINT chk_p043_bom_ef_year CHECK (
        ef_source_year IS NULL OR (ef_source_year >= 1990 AND ef_source_year <= 2100)
    ),
    CONSTRAINT chk_p043_bom_confidence CHECK (
        ef_confidence IS NULL OR (ef_confidence >= 0 AND ef_confidence <= 1)
    ),
    CONSTRAINT chk_p043_bom_tco2e CHECK (
        component_tco2e IS NULL OR component_tco2e >= 0
    ),
    CONSTRAINT chk_p043_bom_pct CHECK (
        pct_of_product IS NULL OR (pct_of_product >= 0 AND pct_of_product <= 100)
    ),
    CONSTRAINT chk_p043_bom_transport_dist CHECK (
        transport_distance_km IS NULL OR transport_distance_km >= 0
    ),
    CONSTRAINT chk_p043_bom_transport_tco2e CHECK (
        transport_tco2e IS NULL OR transport_tco2e >= 0
    ),
    CONSTRAINT chk_p043_bom_transport_mode CHECK (
        transport_mode IS NULL OR transport_mode IN (
            'ROAD', 'RAIL', 'SEA', 'AIR', 'INLAND_WATERWAY', 'PIPELINE', 'MULTIMODAL'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_bom_tenant            ON ghg_accounting_scope3_complete.product_bom(tenant_id);
CREATE INDEX idx_p043_bom_product           ON ghg_accounting_scope3_complete.product_bom(product_id);
CREATE INDEX idx_p043_bom_component         ON ghg_accounting_scope3_complete.product_bom(component_name);
CREATE INDEX idx_p043_bom_material          ON ghg_accounting_scope3_complete.product_bom(material);
CREATE INDEX idx_p043_bom_material_cat      ON ghg_accounting_scope3_complete.product_bom(material_category);
CREATE INDEX idx_p043_bom_supplier          ON ghg_accounting_scope3_complete.product_bom(supplier_id);
CREATE INDEX idx_p043_bom_level             ON ghg_accounting_scope3_complete.product_bom(bom_level);
CREATE INDEX idx_p043_bom_parent            ON ghg_accounting_scope3_complete.product_bom(parent_component_id);
CREATE INDEX idx_p043_bom_origin            ON ghg_accounting_scope3_complete.product_bom(origin_country);
CREATE INDEX idx_p043_bom_tco2e             ON ghg_accounting_scope3_complete.product_bom(component_tco2e DESC);
CREATE INDEX idx_p043_bom_created           ON ghg_accounting_scope3_complete.product_bom(created_at DESC);
CREATE INDEX idx_p043_bom_metadata          ON ghg_accounting_scope3_complete.product_bom USING GIN(metadata);

-- Composite: product + level for BOM tree traversal
CREATE INDEX idx_p043_bom_product_level     ON ghg_accounting_scope3_complete.product_bom(product_id, bom_level, mass_kg DESC);

-- Composite: product + emission hotspot identification
CREATE INDEX idx_p043_bom_product_tco2e     ON ghg_accounting_scope3_complete.product_bom(product_id, component_tco2e DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_bom_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.product_bom
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_accounting_scope3_complete.lifecycle_results
-- =============================================================================
-- Per-lifecycle-stage emission results for each product. Each row stores the
-- GHG impact for one stage of a product's life (raw material, manufacturing,
-- use, end-of-life, etc.). These records are aggregated into the product
-- carbon footprint (PCF). Supports ISO 14040/14044 and ISO 14067.

CREATE TABLE ghg_accounting_scope3_complete.lifecycle_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    product_id                  UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.products(id) ON DELETE CASCADE,
    -- Stage
    stage                       ghg_accounting_scope3_complete.lifecycle_stage NOT NULL,
    stage_detail                VARCHAR(200),
    -- Emissions
    tco2e                       DECIMAL(15,6)   NOT NULL DEFAULT 0,
    co2_biogenic                DECIMAL(15,6)   DEFAULT 0,
    pct_of_total                DECIMAL(5,2),
    -- Uncertainty
    uncertainty_lower           DECIMAL(15,6),
    uncertainty_upper           DECIMAL(15,6),
    confidence_level            DECIMAL(3,2),
    -- Methodology
    methodology                 VARCHAR(200)    NOT NULL DEFAULT 'ISO_14040',
    calculation_method          VARCHAR(100),
    data_quality_rating         DECIMAL(3,1),
    primary_data_pct            DECIMAL(5,2),
    -- Emission factor
    ef_source                   VARCHAR(200),
    ef_database                 VARCHAR(100),
    ef_version                  VARCHAR(50),
    -- Allocation
    allocation_method           VARCHAR(50),
    allocation_factor           DECIMAL(6,4),
    -- Provenance
    provenance_hash             VARCHAR(64)     NOT NULL,
    parent_hash                 VARCHAR(64),
    calculation_date            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Metadata
    assumptions                 JSONB           DEFAULT '[]',
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_lr_tco2e CHECK (
        tco2e >= 0
    ),
    CONSTRAINT chk_p043_lr_biogenic CHECK (
        co2_biogenic IS NULL OR co2_biogenic >= 0
    ),
    CONSTRAINT chk_p043_lr_pct CHECK (
        pct_of_total IS NULL OR (pct_of_total >= 0 AND pct_of_total <= 100)
    ),
    CONSTRAINT chk_p043_lr_confidence CHECK (
        confidence_level IS NULL OR (confidence_level >= 0 AND confidence_level <= 1)
    ),
    CONSTRAINT chk_p043_lr_methodology CHECK (
        methodology IN (
            'ISO_14040', 'ISO_14044', 'ISO_14067', 'PAS_2050',
            'PACT_PATHFINDER', 'GHG_PROTOCOL_PRODUCT', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p043_lr_dqr CHECK (
        data_quality_rating IS NULL OR (data_quality_rating >= 1.0 AND data_quality_rating <= 5.0)
    ),
    CONSTRAINT chk_p043_lr_primary CHECK (
        primary_data_pct IS NULL OR (primary_data_pct >= 0 AND primary_data_pct <= 100)
    ),
    CONSTRAINT chk_p043_lr_alloc_method CHECK (
        allocation_method IS NULL OR allocation_method IN (
            'ECONOMIC', 'PHYSICAL', 'MASS', 'ENERGY', 'SYSTEM_EXPANSION', 'CUTOFF'
        )
    ),
    CONSTRAINT chk_p043_lr_alloc_factor CHECK (
        allocation_factor IS NULL OR (allocation_factor >= 0 AND allocation_factor <= 1)
    ),
    CONSTRAINT uq_p043_lr_product_stage UNIQUE (product_id, stage)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_lr_tenant             ON ghg_accounting_scope3_complete.lifecycle_results(tenant_id);
CREATE INDEX idx_p043_lr_product            ON ghg_accounting_scope3_complete.lifecycle_results(product_id);
CREATE INDEX idx_p043_lr_stage              ON ghg_accounting_scope3_complete.lifecycle_results(stage);
CREATE INDEX idx_p043_lr_tco2e              ON ghg_accounting_scope3_complete.lifecycle_results(tco2e DESC);
CREATE INDEX idx_p043_lr_methodology        ON ghg_accounting_scope3_complete.lifecycle_results(methodology);
CREATE INDEX idx_p043_lr_ef_source          ON ghg_accounting_scope3_complete.lifecycle_results(ef_source);
CREATE INDEX idx_p043_lr_provenance         ON ghg_accounting_scope3_complete.lifecycle_results(provenance_hash);
CREATE INDEX idx_p043_lr_parent             ON ghg_accounting_scope3_complete.lifecycle_results(parent_hash);
CREATE INDEX idx_p043_lr_calc_date          ON ghg_accounting_scope3_complete.lifecycle_results(calculation_date DESC);
CREATE INDEX idx_p043_lr_created            ON ghg_accounting_scope3_complete.lifecycle_results(created_at DESC);
CREATE INDEX idx_p043_lr_metadata           ON ghg_accounting_scope3_complete.lifecycle_results USING GIN(metadata);
CREATE INDEX idx_p043_lr_assumptions        ON ghg_accounting_scope3_complete.lifecycle_results USING GIN(assumptions);

-- Composite: product + stage breakdown
CREATE INDEX idx_p043_lr_product_stage      ON ghg_accounting_scope3_complete.lifecycle_results(product_id, stage, tco2e DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_lr_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.lifecycle_results
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_accounting_scope3_complete.product_carbon_footprints
-- =============================================================================
-- Aggregated product carbon footprint (PCF) per ISO 14067 and the PACT
-- Pathfinder framework. Consolidates lifecycle_results into a single
-- cradle-to-grave (or cradle-to-gate) footprint per product with per-unit
-- intensity and phase breakdown.

CREATE TABLE ghg_accounting_scope3_complete.product_carbon_footprints (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    product_id                  UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.products(id) ON DELETE CASCADE,
    -- Totals
    total_tco2e                 DECIMAL(15,6)   NOT NULL,
    per_unit_kgco2e             DECIMAL(12,6)   NOT NULL,
    -- Phase breakdown
    cradle_to_gate              DECIMAL(15,6),
    gate_to_gate                DECIMAL(15,6),
    use_phase                   DECIMAL(15,6),
    end_of_life                 DECIMAL(15,6),
    recycling_benefit           DECIMAL(15,6),
    packaging                   DECIMAL(15,6),
    transport                   DECIMAL(15,6),
    -- Biogenic
    biogenic_tco2e              DECIMAL(15,6)   DEFAULT 0,
    biogenic_removal_tco2e      DECIMAL(15,6)   DEFAULT 0,
    -- Methodology
    methodology                 VARCHAR(200)    NOT NULL DEFAULT 'ISO_14067',
    boundary                    VARCHAR(50)     NOT NULL DEFAULT 'CRADLE_TO_GRAVE',
    gwp_source                  VARCHAR(20)     DEFAULT 'AR5',
    primary_data_pct            DECIMAL(5,2),
    data_quality_rating         DECIMAL(3,1),
    -- Benchmarking
    sector_avg_kgco2e           DECIMAL(12,6),
    benchmark_source            VARCHAR(200),
    pct_vs_sector_avg           DECIMAL(8,2),
    -- PACT Pathfinder compliance
    pact_compliant              BOOLEAN         DEFAULT false,
    pact_version                VARCHAR(20),
    -- Calculation
    calculation_date            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    valid_from                  DATE,
    valid_until                 DATE,
    version                     INTEGER         NOT NULL DEFAULT 1,
    -- Verification
    verified                    BOOLEAN         NOT NULL DEFAULT false,
    verified_by                 VARCHAR(255),
    verified_at                 TIMESTAMPTZ,
    assurance_level             ghg_accounting_scope3_complete.assurance_level,
    -- Provenance
    provenance_hash             VARCHAR(64)     NOT NULL,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_pcf_total CHECK (total_tco2e >= 0),
    CONSTRAINT chk_p043_pcf_per_unit CHECK (per_unit_kgco2e >= 0),
    CONSTRAINT chk_p043_pcf_ctg CHECK (cradle_to_gate IS NULL OR cradle_to_gate >= 0),
    CONSTRAINT chk_p043_pcf_use CHECK (use_phase IS NULL OR use_phase >= 0),
    CONSTRAINT chk_p043_pcf_eol CHECK (end_of_life IS NULL OR end_of_life >= 0),
    CONSTRAINT chk_p043_pcf_methodology CHECK (
        methodology IN ('ISO_14067', 'PAS_2050', 'PACT_PATHFINDER', 'GHG_PROTOCOL_PRODUCT', 'CUSTOM')
    ),
    CONSTRAINT chk_p043_pcf_boundary CHECK (
        boundary IN ('CRADLE_TO_GRAVE', 'CRADLE_TO_GATE', 'GATE_TO_GATE', 'CRADLE_TO_GATE_WITH_OPTIONS')
    ),
    CONSTRAINT chk_p043_pcf_gwp CHECK (
        gwp_source IS NULL OR gwp_source IN ('AR4', 'AR5', 'AR6', 'SAR', 'TAR')
    ),
    CONSTRAINT chk_p043_pcf_primary CHECK (
        primary_data_pct IS NULL OR (primary_data_pct >= 0 AND primary_data_pct <= 100)
    ),
    CONSTRAINT chk_p043_pcf_dqr CHECK (
        data_quality_rating IS NULL OR (data_quality_rating >= 1.0 AND data_quality_rating <= 5.0)
    ),
    CONSTRAINT chk_p043_pcf_version CHECK (version >= 1),
    CONSTRAINT chk_p043_pcf_validity CHECK (
        valid_from IS NULL OR valid_until IS NULL OR valid_from <= valid_until
    ),
    CONSTRAINT uq_p043_pcf_product_version UNIQUE (product_id, version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_pcf_tenant            ON ghg_accounting_scope3_complete.product_carbon_footprints(tenant_id);
CREATE INDEX idx_p043_pcf_product           ON ghg_accounting_scope3_complete.product_carbon_footprints(product_id);
CREATE INDEX idx_p043_pcf_total             ON ghg_accounting_scope3_complete.product_carbon_footprints(total_tco2e DESC);
CREATE INDEX idx_p043_pcf_per_unit          ON ghg_accounting_scope3_complete.product_carbon_footprints(per_unit_kgco2e DESC);
CREATE INDEX idx_p043_pcf_methodology       ON ghg_accounting_scope3_complete.product_carbon_footprints(methodology);
CREATE INDEX idx_p043_pcf_boundary          ON ghg_accounting_scope3_complete.product_carbon_footprints(boundary);
CREATE INDEX idx_p043_pcf_calc_date         ON ghg_accounting_scope3_complete.product_carbon_footprints(calculation_date DESC);
CREATE INDEX idx_p043_pcf_verified          ON ghg_accounting_scope3_complete.product_carbon_footprints(verified) WHERE verified = true;
CREATE INDEX idx_p043_pcf_pact              ON ghg_accounting_scope3_complete.product_carbon_footprints(pact_compliant) WHERE pact_compliant = true;
CREATE INDEX idx_p043_pcf_provenance        ON ghg_accounting_scope3_complete.product_carbon_footprints(provenance_hash);
CREATE INDEX idx_p043_pcf_created           ON ghg_accounting_scope3_complete.product_carbon_footprints(created_at DESC);
CREATE INDEX idx_p043_pcf_metadata          ON ghg_accounting_scope3_complete.product_carbon_footprints USING GIN(metadata);

-- Composite: product + latest version
CREATE INDEX idx_p043_pcf_product_latest    ON ghg_accounting_scope3_complete.product_carbon_footprints(product_id, version DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_pcf_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.product_carbon_footprints
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_accounting_scope3_complete.products ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.product_bom ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.lifecycle_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.product_carbon_footprints ENABLE ROW LEVEL SECURITY;

CREATE POLICY p043_prod_tenant_isolation ON ghg_accounting_scope3_complete.products
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_prod_service_bypass ON ghg_accounting_scope3_complete.products
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_bom_tenant_isolation ON ghg_accounting_scope3_complete.product_bom
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_bom_service_bypass ON ghg_accounting_scope3_complete.product_bom
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_lr_tenant_isolation ON ghg_accounting_scope3_complete.lifecycle_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_lr_service_bypass ON ghg_accounting_scope3_complete.lifecycle_results
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_pcf_tenant_isolation ON ghg_accounting_scope3_complete.product_carbon_footprints
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_pcf_service_bypass ON ghg_accounting_scope3_complete.product_carbon_footprints
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.products TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.product_bom TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.lifecycle_results TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.product_carbon_footprints TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_accounting_scope3_complete.products IS
    'Product master data for LCA-based Scope 3 calculations with classification, functional unit (ISO 14040), volume, and revenue.';
COMMENT ON TABLE ghg_accounting_scope3_complete.product_bom IS
    'Bill of materials per product with component-level mass, supplier, emission factor, and transport for cradle-to-gate LCA.';
COMMENT ON TABLE ghg_accounting_scope3_complete.lifecycle_results IS
    'Per-lifecycle-stage emission results (ISO 14040/14044) with provenance hash chain for audit trail.';
COMMENT ON TABLE ghg_accounting_scope3_complete.product_carbon_footprints IS
    'Aggregated product carbon footprint (PCF) per ISO 14067 / PACT Pathfinder with phase breakdown, benchmarking, and verification.';

COMMENT ON COLUMN ghg_accounting_scope3_complete.products.functional_unit IS 'ISO 14040 functional unit -- the quantified performance of the product system.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.products.hs_code IS 'Harmonized System code for customs/trade classification.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.products.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN ghg_accounting_scope3_complete.product_bom.ef_source IS 'Emission factor database (e.g., ecoinvent 3.9, DEFRA 2025, USEEIO).';
COMMENT ON COLUMN ghg_accounting_scope3_complete.product_bom.ef_confidence IS 'Confidence in emission factor (0.0 to 1.0).';

COMMENT ON COLUMN ghg_accounting_scope3_complete.lifecycle_results.provenance_hash IS 'SHA-256 hash of calculation inputs and outputs for audit trail.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.lifecycle_results.parent_hash IS 'Hash of parent calculation for hash-chain provenance.';

COMMENT ON COLUMN ghg_accounting_scope3_complete.product_carbon_footprints.per_unit_kgco2e IS 'Carbon footprint per functional unit in kgCO2e.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.product_carbon_footprints.pact_compliant IS 'Whether PCF complies with PACT Pathfinder data exchange standard.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.product_carbon_footprints.pct_vs_sector_avg IS 'Percentage difference vs sector average (negative = below average = better).';
