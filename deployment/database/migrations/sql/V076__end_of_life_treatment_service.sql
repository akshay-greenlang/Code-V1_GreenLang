-- =====================================================================================
-- Migration: V076__end_of_life_treatment_service.sql
-- Description: AGENT-MRV-025 End-of-Life Treatment of Sold Products (Scope 3 Category 12)
-- Agent: GL-MRV-S3-012
-- Framework: GHG Protocol Scope 3 Standard, IPCC 2006 Waste (Vol 5), EPA WARM v16,
--            DEFRA 2024, ISO 14064-1, CSRD ESRS E1/E5, CDP, SBTi
-- Created: 2026-02-28
-- =====================================================================================
-- Schema: end_of_life_treatment_service
-- Tables: 21 (10 reference + 8 operational + 3 supporting)
-- Hypertables: 3 (calculations 7-day, compliance_checks 30-day, aggregations 30-day)
-- Continuous Aggregates: 2 (gl_eol_daily_by_treatment, gl_eol_monthly_by_material)
-- RLS Policies: 10 (all tables with tenant_id)
-- Indexes: ~100 (tenant, calc_id, material, treatment, region, hash, GIN JSONB)
-- Check Constraints: ~90 (positive values, fraction 0-1, enum validation)
-- Seed Data: ~120 records (15 materials x 7 treatments + compositions + mixes + params)
-- Agent Registry: 1 INSERT for GL-MRV-S3-012
-- =====================================================================================

-- =====================================================================================
-- SCHEMA CREATION
-- =====================================================================================

CREATE SCHEMA IF NOT EXISTS end_of_life_treatment_service;

COMMENT ON SCHEMA end_of_life_treatment_service IS 'AGENT-MRV-025: End-of-Life Treatment of Sold Products - Scope 3 Category 12 emission calculations (landfill FOD/incineration WtE/recycling cut-off/composting/AD/open burning)';

-- =====================================================================================
-- TABLE 1: gl_eol_material_emission_factors (REFERENCE)
-- Description: Material-level emission factors by treatment method
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_material_emission_factors (
    id SERIAL PRIMARY KEY,
    material_type VARCHAR(100) NOT NULL,
    treatment_method VARCHAR(100) NOT NULL,
    ef_kgco2e_per_kg DECIMAL(12,6) NOT NULL,
    co2_fossil_kgco2_per_kg DECIMAL(12,6) NOT NULL DEFAULT 0,
    co2_biogenic_kgco2_per_kg DECIMAL(12,6) NOT NULL DEFAULT 0,
    ch4_kgch4_per_kg DECIMAL(12,8) NOT NULL DEFAULT 0,
    n2o_kgn2o_per_kg DECIMAL(12,8) NOT NULL DEFAULT 0,
    ef_source VARCHAR(100) NOT NULL,
    region VARCHAR(50) NOT NULL DEFAULT 'GLOBAL',
    year INT NOT NULL,
    gwp_version VARCHAR(20) NOT NULL DEFAULT 'AR5',
    uncertainty_pct DECIMAL(5,2),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eol_ef_year CHECK (year >= 1990 AND year <= 2100),
    CONSTRAINT chk_eol_ef_co2_fossil CHECK (co2_fossil_kgco2_per_kg >= 0),
    CONSTRAINT chk_eol_ef_co2_biogenic CHECK (co2_biogenic_kgco2_per_kg >= 0),
    CONSTRAINT chk_eol_ef_ch4 CHECK (ch4_kgch4_per_kg >= 0),
    CONSTRAINT chk_eol_ef_n2o CHECK (n2o_kgn2o_per_kg >= 0),
    CONSTRAINT chk_eol_ef_uncertainty CHECK (uncertainty_pct IS NULL OR (uncertainty_pct >= 0 AND uncertainty_pct <= 200)),
    CONSTRAINT chk_eol_ef_treatment CHECK (treatment_method IN (
        'LANDFILL', 'INCINERATION', 'INCINERATION_WTE', 'RECYCLING',
        'COMPOSTING', 'ANAEROBIC_DIGESTION', 'OPEN_BURNING', 'WASTEWATER'
    )),
    CONSTRAINT uq_eol_ef_material_treatment_source UNIQUE (material_type, treatment_method, ef_source, region, year)
);

CREATE INDEX idx_eol_ef_material ON end_of_life_treatment_service.gl_eol_material_emission_factors(material_type);
CREATE INDEX idx_eol_ef_treatment ON end_of_life_treatment_service.gl_eol_material_emission_factors(treatment_method);
CREATE INDEX idx_eol_ef_source ON end_of_life_treatment_service.gl_eol_material_emission_factors(ef_source);
CREATE INDEX idx_eol_ef_region ON end_of_life_treatment_service.gl_eol_material_emission_factors(region);
CREATE INDEX idx_eol_ef_year ON end_of_life_treatment_service.gl_eol_material_emission_factors(year);
CREATE INDEX idx_eol_ef_material_treatment ON end_of_life_treatment_service.gl_eol_material_emission_factors(material_type, treatment_method);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_material_emission_factors IS 'Material-level emission factors by treatment method from EPA WARM, DEFRA, IPCC';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_material_emission_factors.ef_kgco2e_per_kg IS 'Total CO2e emission factor (kg CO2e per kg material); can be negative for recycling credits';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_material_emission_factors.co2_fossil_kgco2_per_kg IS 'Fossil CO2 component (kg CO2 per kg material)';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_material_emission_factors.co2_biogenic_kgco2_per_kg IS 'Biogenic CO2 (memo item, kg CO2 per kg material)';

-- =====================================================================================
-- TABLE 2: gl_eol_product_compositions (REFERENCE)
-- Description: Default bill of materials (BOM) for product categories
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_product_compositions (
    id SERIAL PRIMARY KEY,
    product_category VARCHAR(200) NOT NULL,
    material_type VARCHAR(100) NOT NULL,
    fraction DECIMAL(5,4) NOT NULL,
    weight_fraction_kg DECIMAL(12,4),
    description VARCHAR(500),
    source VARCHAR(200),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eol_comp_fraction CHECK (fraction >= 0 AND fraction <= 1),
    CONSTRAINT chk_eol_comp_weight CHECK (weight_fraction_kg IS NULL OR weight_fraction_kg >= 0),
    CONSTRAINT uq_eol_comp_category_material UNIQUE (product_category, material_type)
);

CREATE INDEX idx_eol_comp_category ON end_of_life_treatment_service.gl_eol_product_compositions(product_category);
CREATE INDEX idx_eol_comp_material ON end_of_life_treatment_service.gl_eol_product_compositions(material_type);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_product_compositions IS 'Default bill of materials (BOM) for product categories used in average-data method';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_product_compositions.fraction IS 'Mass fraction of this material in the product (0.0-1.0)';

-- =====================================================================================
-- TABLE 3: gl_eol_regional_treatment_mixes (REFERENCE)
-- Description: Regional waste treatment pathway fractions
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_regional_treatment_mixes (
    id SERIAL PRIMARY KEY,
    region VARCHAR(50) NOT NULL,
    treatment_method VARCHAR(100) NOT NULL,
    fraction DECIMAL(5,4) NOT NULL,
    year INT NOT NULL,
    source VARCHAR(200),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eol_mix_fraction CHECK (fraction >= 0 AND fraction <= 1),
    CONSTRAINT chk_eol_mix_year CHECK (year >= 1990 AND year <= 2100),
    CONSTRAINT chk_eol_mix_treatment CHECK (treatment_method IN (
        'LANDFILL', 'INCINERATION', 'INCINERATION_WTE', 'RECYCLING',
        'COMPOSTING', 'ANAEROBIC_DIGESTION', 'OPEN_BURNING', 'WASTEWATER'
    )),
    CONSTRAINT uq_eol_mix_region_treatment_year UNIQUE (region, treatment_method, year)
);

CREATE INDEX idx_eol_mix_region ON end_of_life_treatment_service.gl_eol_regional_treatment_mixes(region);
CREATE INDEX idx_eol_mix_treatment ON end_of_life_treatment_service.gl_eol_regional_treatment_mixes(treatment_method);
CREATE INDEX idx_eol_mix_year ON end_of_life_treatment_service.gl_eol_regional_treatment_mixes(year);
CREATE INDEX idx_eol_mix_region_year ON end_of_life_treatment_service.gl_eol_regional_treatment_mixes(region, year);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_regional_treatment_mixes IS 'Regional waste treatment pathway fractions (landfill/incineration/recycling/composting) by region and year';

-- =====================================================================================
-- TABLE 4: gl_eol_landfill_parameters (REFERENCE)
-- Description: IPCC First-Order Decay (FOD) parameters by material and climate zone
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_landfill_parameters (
    id SERIAL PRIMARY KEY,
    material_type VARCHAR(100) NOT NULL,
    climate_zone VARCHAR(50) NOT NULL DEFAULT 'TEMPERATE',
    doc DECIMAL(5,4) NOT NULL,
    docf DECIMAL(5,4) NOT NULL,
    mcf DECIMAL(5,4) NOT NULL,
    k_decay DECIMAL(8,6) NOT NULL,
    f_ch4 DECIMAL(5,4) NOT NULL DEFAULT 0.50,
    ox_factor DECIMAL(5,4) NOT NULL DEFAULT 0.10,
    source VARCHAR(200) DEFAULT 'IPCC 2006',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eol_lf_doc CHECK (doc >= 0 AND doc <= 1),
    CONSTRAINT chk_eol_lf_docf CHECK (docf >= 0 AND docf <= 1),
    CONSTRAINT chk_eol_lf_mcf CHECK (mcf >= 0 AND mcf <= 1),
    CONSTRAINT chk_eol_lf_k CHECK (k_decay >= 0),
    CONSTRAINT chk_eol_lf_f_ch4 CHECK (f_ch4 >= 0 AND f_ch4 <= 1),
    CONSTRAINT chk_eol_lf_ox CHECK (ox_factor >= 0 AND ox_factor <= 1),
    CONSTRAINT chk_eol_lf_climate CHECK (climate_zone IN (
        'TROPICAL_WET', 'TROPICAL_DRY', 'TEMPERATE', 'BOREAL'
    )),
    CONSTRAINT uq_eol_lf_material_climate UNIQUE (material_type, climate_zone)
);

CREATE INDEX idx_eol_lf_material ON end_of_life_treatment_service.gl_eol_landfill_parameters(material_type);
CREATE INDEX idx_eol_lf_climate ON end_of_life_treatment_service.gl_eol_landfill_parameters(climate_zone);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_landfill_parameters IS 'IPCC FOD parameters for landfill CH4 generation by material type and climate zone';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_landfill_parameters.doc IS 'Degradable Organic Carbon fraction (0-1)';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_landfill_parameters.docf IS 'Fraction of DOC dissimilated to biogas (0-1)';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_landfill_parameters.mcf IS 'Methane Correction Factor (0-1); 1.0 for managed anaerobic';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_landfill_parameters.k_decay IS 'First-order decay rate constant (1/year)';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_landfill_parameters.f_ch4 IS 'Fraction of CH4 in generated landfill gas (default 0.50)';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_landfill_parameters.ox_factor IS 'Methane oxidation factor in cover soil (default 0.10)';

-- =====================================================================================
-- TABLE 5: gl_eol_incineration_parameters (REFERENCE)
-- Description: Combustion parameters for incineration calculations
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_incineration_parameters (
    id SERIAL PRIMARY KEY,
    material_type VARCHAR(100) NOT NULL,
    dry_matter DECIMAL(5,4) NOT NULL,
    carbon_fraction DECIMAL(5,4) NOT NULL,
    fossil_carbon DECIMAL(5,4) NOT NULL,
    oxidation_factor DECIMAL(5,4) NOT NULL DEFAULT 1.0,
    calorific_value_mj_kg DECIMAL(10,4),
    ch4_ef_kg_per_tonne DECIMAL(10,6) DEFAULT 0,
    n2o_ef_kg_per_tonne DECIMAL(10,6) DEFAULT 0,
    source VARCHAR(200) DEFAULT 'IPCC 2006',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eol_incin_dm CHECK (dry_matter >= 0 AND dry_matter <= 1),
    CONSTRAINT chk_eol_incin_cf CHECK (carbon_fraction >= 0 AND carbon_fraction <= 1),
    CONSTRAINT chk_eol_incin_fc CHECK (fossil_carbon >= 0 AND fossil_carbon <= 1),
    CONSTRAINT chk_eol_incin_ox CHECK (oxidation_factor >= 0 AND oxidation_factor <= 1),
    CONSTRAINT chk_eol_incin_cv CHECK (calorific_value_mj_kg IS NULL OR calorific_value_mj_kg >= 0),
    CONSTRAINT chk_eol_incin_ch4 CHECK (ch4_ef_kg_per_tonne >= 0),
    CONSTRAINT chk_eol_incin_n2o CHECK (n2o_ef_kg_per_tonne >= 0),
    CONSTRAINT uq_eol_incin_material UNIQUE (material_type)
);

CREATE INDEX idx_eol_incin_material ON end_of_life_treatment_service.gl_eol_incineration_parameters(material_type);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_incineration_parameters IS 'Combustion parameters for incineration CO2/CH4/N2O calculations per IPCC Vol 5';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_incineration_parameters.dry_matter IS 'Dry matter content as fraction (0-1)';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_incineration_parameters.carbon_fraction IS 'Carbon content of dry matter (0-1)';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_incineration_parameters.fossil_carbon IS 'Fraction of carbon that is fossil-origin (0-1)';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_incineration_parameters.calorific_value_mj_kg IS 'Net calorific value (MJ/kg) for energy recovery calculations';

-- =====================================================================================
-- TABLE 6: gl_eol_recycling_factors (REFERENCE)
-- Description: Recycling process EFs and avoided virgin material credits
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_recycling_factors (
    id SERIAL PRIMARY KEY,
    material_type VARCHAR(100) NOT NULL,
    transport_ef DECIMAL(12,6) NOT NULL DEFAULT 0,
    mrf_ef DECIMAL(12,6) NOT NULL DEFAULT 0,
    total_processing_ef DECIMAL(12,6) NOT NULL DEFAULT 0,
    avoided_ef DECIMAL(12,6) NOT NULL DEFAULT 0,
    recycling_rate DECIMAL(5,4) NOT NULL DEFAULT 0.85,
    source VARCHAR(200) DEFAULT 'EPA WARM v16',
    region VARCHAR(50) DEFAULT 'GLOBAL',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eol_recycle_transport CHECK (transport_ef >= 0),
    CONSTRAINT chk_eol_recycle_mrf CHECK (mrf_ef >= 0),
    CONSTRAINT chk_eol_recycle_processing CHECK (total_processing_ef >= 0),
    CONSTRAINT chk_eol_recycle_avoided CHECK (avoided_ef >= 0),
    CONSTRAINT chk_eol_recycle_rate CHECK (recycling_rate >= 0 AND recycling_rate <= 1),
    CONSTRAINT uq_eol_recycle_material_region UNIQUE (material_type, region)
);

CREATE INDEX idx_eol_recycle_material ON end_of_life_treatment_service.gl_eol_recycling_factors(material_type);
CREATE INDEX idx_eol_recycle_region ON end_of_life_treatment_service.gl_eol_recycling_factors(region);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_recycling_factors IS 'Recycling process emissions and avoided virgin material credits (cut-off approach)';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_recycling_factors.transport_ef IS 'Transport to MRF (kgCO2e/kg material)';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_recycling_factors.mrf_ef IS 'MRF sorting/processing (kgCO2e/kg material)';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_recycling_factors.avoided_ef IS 'Avoided virgin material emissions (kgCO2e/kg material, positive value)';

-- =====================================================================================
-- TABLE 7: gl_eol_product_weight_defaults (REFERENCE)
-- Description: Default product weights by category
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_product_weight_defaults (
    id SERIAL PRIMARY KEY,
    product_category VARCHAR(200) NOT NULL,
    avg_weight_kg DECIMAL(12,4) NOT NULL,
    min_weight_kg DECIMAL(12,4),
    max_weight_kg DECIMAL(12,4),
    source VARCHAR(200),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eol_weight_avg CHECK (avg_weight_kg > 0),
    CONSTRAINT chk_eol_weight_min CHECK (min_weight_kg IS NULL OR min_weight_kg > 0),
    CONSTRAINT chk_eol_weight_max CHECK (max_weight_kg IS NULL OR max_weight_kg > 0),
    CONSTRAINT chk_eol_weight_range CHECK (
        min_weight_kg IS NULL OR max_weight_kg IS NULL
        OR min_weight_kg <= avg_weight_kg
    ),
    CONSTRAINT uq_eol_weight_category UNIQUE (product_category)
);

CREATE INDEX idx_eol_weight_category ON end_of_life_treatment_service.gl_eol_product_weight_defaults(product_category);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_product_weight_defaults IS 'Default product weights by category for average-data method fallback';

-- =====================================================================================
-- TABLE 8: gl_eol_gas_collection_factors (REFERENCE)
-- Description: Landfill gas collection and flare efficiency by type
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_gas_collection_factors (
    id SERIAL PRIMARY KEY,
    landfill_type VARCHAR(100) NOT NULL,
    collection_efficiency DECIMAL(5,4) NOT NULL,
    flare_efficiency DECIMAL(5,4) NOT NULL DEFAULT 0.99,
    description VARCHAR(500),
    source VARCHAR(200) DEFAULT 'IPCC 2006',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eol_gc_collection CHECK (collection_efficiency >= 0 AND collection_efficiency <= 1),
    CONSTRAINT chk_eol_gc_flare CHECK (flare_efficiency >= 0 AND flare_efficiency <= 1),
    CONSTRAINT chk_eol_gc_landfill_type CHECK (landfill_type IN (
        'MANAGED_ANAEROBIC', 'MANAGED_SEMI', 'UNMANAGED_DEEP', 'UNMANAGED_SHALLOW',
        'SANITARY_WITH_GAS', 'SANITARY_NO_GAS', 'BIOREACTOR'
    )),
    CONSTRAINT uq_eol_gc_type UNIQUE (landfill_type)
);

CREATE INDEX idx_eol_gc_type ON end_of_life_treatment_service.gl_eol_gas_collection_factors(landfill_type);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_gas_collection_factors IS 'Landfill gas collection and flare destruction efficiency by landfill type';

-- =====================================================================================
-- TABLE 9: gl_eol_energy_recovery_factors (REFERENCE)
-- Description: WtE energy recovery and displaced grid emission factors
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_energy_recovery_factors (
    id SERIAL PRIMARY KEY,
    region VARCHAR(50) NOT NULL,
    wte_efficiency DECIMAL(5,4) NOT NULL,
    displaced_grid_ef DECIMAL(12,6) NOT NULL,
    year INT NOT NULL,
    source VARCHAR(200),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eol_er_efficiency CHECK (wte_efficiency >= 0 AND wte_efficiency <= 1),
    CONSTRAINT chk_eol_er_grid_ef CHECK (displaced_grid_ef >= 0),
    CONSTRAINT chk_eol_er_year CHECK (year >= 1990 AND year <= 2100),
    CONSTRAINT uq_eol_er_region_year UNIQUE (region, year)
);

CREATE INDEX idx_eol_er_region ON end_of_life_treatment_service.gl_eol_energy_recovery_factors(region);
CREATE INDEX idx_eol_er_year ON end_of_life_treatment_service.gl_eol_energy_recovery_factors(year);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_energy_recovery_factors IS 'Waste-to-energy conversion efficiency and displaced grid EF by region';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_energy_recovery_factors.wte_efficiency IS 'Net electrical conversion efficiency of WtE facility (0-1)';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_energy_recovery_factors.displaced_grid_ef IS 'Grid emission factor displaced by WtE electricity (kgCO2e/kWh)';

-- =====================================================================================
-- TABLE 10: gl_eol_composting_ad_factors (REFERENCE)
-- Description: Composting and anaerobic digestion emission factors
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_composting_ad_factors (
    id SERIAL PRIMARY KEY,
    treatment_type VARCHAR(100) NOT NULL,
    ch4_ef DECIMAL(12,8) NOT NULL,
    n2o_ef DECIMAL(12,8) NOT NULL,
    biogas_yield DECIMAL(10,4),
    capture_efficiency DECIMAL(5,4),
    source VARCHAR(200) DEFAULT 'IPCC 2006',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eol_cad_ch4 CHECK (ch4_ef >= 0),
    CONSTRAINT chk_eol_cad_n2o CHECK (n2o_ef >= 0),
    CONSTRAINT chk_eol_cad_biogas CHECK (biogas_yield IS NULL OR biogas_yield >= 0),
    CONSTRAINT chk_eol_cad_capture CHECK (capture_efficiency IS NULL OR (capture_efficiency >= 0 AND capture_efficiency <= 1)),
    CONSTRAINT chk_eol_cad_type CHECK (treatment_type IN (
        'COMPOSTING_INDUSTRIAL', 'COMPOSTING_HOME', 'ANAEROBIC_DIGESTION_ENCLOSED',
        'ANAEROBIC_DIGESTION_OPEN', 'ANAEROBIC_DIGESTION_DRY'
    )),
    CONSTRAINT uq_eol_cad_type UNIQUE (treatment_type)
);

CREATE INDEX idx_eol_cad_type ON end_of_life_treatment_service.gl_eol_composting_ad_factors(treatment_type);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_composting_ad_factors IS 'Composting and anaerobic digestion CH4/N2O emission factors and biogas parameters';

-- =====================================================================================
-- TABLE 11: gl_eol_calculations (HYPERTABLE - 7-day chunks)
-- Description: Primary calculation results for end-of-life treatment
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_calculations (
    calc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    org_id VARCHAR(200) NOT NULL,
    year INT NOT NULL,
    method VARCHAR(100) NOT NULL,
    total_mass_tonnes DECIMAL(16,6) NOT NULL DEFAULT 0,
    gross_tco2e DECIMAL(16,8) NOT NULL DEFAULT 0,
    avoided_tco2e DECIMAL(16,8) NOT NULL DEFAULT 0,
    net_tco2e DECIMAL(16,8) NOT NULL DEFAULT 0,
    total_products INT NOT NULL DEFAULT 0,
    dqi_score DECIMAL(3,1),
    gwp_version VARCHAR(20) NOT NULL DEFAULT 'AR5',
    provenance_hash VARCHAR(128),
    status VARCHAR(50) NOT NULL DEFAULT 'COMPLETED',
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eol_calc_year CHECK (year >= 1990 AND year <= 2100),
    CONSTRAINT chk_eol_calc_mass CHECK (total_mass_tonnes >= 0),
    CONSTRAINT chk_eol_calc_gross CHECK (gross_tco2e >= 0),
    CONSTRAINT chk_eol_calc_avoided CHECK (avoided_tco2e >= 0),
    CONSTRAINT chk_eol_calc_products CHECK (total_products >= 0),
    CONSTRAINT chk_eol_calc_dqi CHECK (dqi_score IS NULL OR (dqi_score >= 1 AND dqi_score <= 5)),
    CONSTRAINT chk_eol_calc_method CHECK (method IN (
        'full_pipeline', 'waste_type_specific', 'average_data',
        'producer_specific', 'hybrid', 'landfill', 'incineration',
        'recycling', 'composting', 'anaerobic_digestion'
    )),
    CONSTRAINT chk_eol_calc_status CHECK (status IN ('COMPLETED', 'FAILED', 'PENDING', 'DELETED'))
);

-- Convert to hypertable with 7-day chunks
SELECT create_hypertable(
    'end_of_life_treatment_service.gl_eol_calculations',
    'created_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_eol_calc_tenant ON end_of_life_treatment_service.gl_eol_calculations(tenant_id, created_at DESC);
CREATE INDEX idx_eol_calc_org ON end_of_life_treatment_service.gl_eol_calculations(org_id);
CREATE INDEX idx_eol_calc_year ON end_of_life_treatment_service.gl_eol_calculations(year);
CREATE INDEX idx_eol_calc_method ON end_of_life_treatment_service.gl_eol_calculations(method);
CREATE INDEX idx_eol_calc_status ON end_of_life_treatment_service.gl_eol_calculations(status);
CREATE INDEX idx_eol_calc_hash ON end_of_life_treatment_service.gl_eol_calculations(provenance_hash);
CREATE INDEX idx_eol_calc_deleted ON end_of_life_treatment_service.gl_eol_calculations(is_deleted);
CREATE INDEX idx_eol_calc_tenant_year ON end_of_life_treatment_service.gl_eol_calculations(tenant_id, year);
CREATE INDEX idx_eol_calc_metadata ON end_of_life_treatment_service.gl_eol_calculations USING GIN(metadata);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_calculations IS 'Primary EoL treatment calculation results (HYPERTABLE, 7-day chunks)';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_calculations.gross_tco2e IS 'Gross emissions before avoided credits (tonnes CO2e)';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_calculations.avoided_tco2e IS 'Avoided emissions from recycling/energy recovery (tonnes CO2e)';
COMMENT ON COLUMN end_of_life_treatment_service.gl_eol_calculations.net_tco2e IS 'Net emissions = gross - avoided (tonnes CO2e)';

-- =====================================================================================
-- TABLE 12: gl_eol_calculation_details (OPERATIONAL)
-- Description: Extended input/output JSONB for each calculation
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_calculation_details (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    input_data JSONB NOT NULL DEFAULT '{}',
    output_data JSONB NOT NULL DEFAULT '{}',
    by_treatment JSONB NOT NULL DEFAULT '{}',
    by_material JSONB NOT NULL DEFAULT '{}',
    by_product JSONB NOT NULL DEFAULT '{}',
    method_details JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eol_detail_calc ON end_of_life_treatment_service.gl_eol_calculation_details(calc_id);
CREATE INDEX idx_eol_detail_input ON end_of_life_treatment_service.gl_eol_calculation_details USING GIN(input_data);
CREATE INDEX idx_eol_detail_output ON end_of_life_treatment_service.gl_eol_calculation_details USING GIN(output_data);
CREATE INDEX idx_eol_detail_treatment ON end_of_life_treatment_service.gl_eol_calculation_details USING GIN(by_treatment);
CREATE INDEX idx_eol_detail_material ON end_of_life_treatment_service.gl_eol_calculation_details USING GIN(by_material);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_calculation_details IS 'Extended input/output JSONB for each calculation with per-treatment/material/product breakdowns';

-- =====================================================================================
-- TABLE 13: gl_eol_material_results (OPERATIONAL)
-- Description: Per material x treatment breakdown
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_material_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    material_type VARCHAR(100) NOT NULL,
    treatment_method VARCHAR(100) NOT NULL,
    mass_kg DECIMAL(16,4) NOT NULL DEFAULT 0,
    ef_used DECIMAL(12,6) NOT NULL DEFAULT 0,
    ef_source VARCHAR(100),
    co2_fossil_kg DECIMAL(16,6) NOT NULL DEFAULT 0,
    co2_biogenic_kg DECIMAL(16,6) NOT NULL DEFAULT 0,
    ch4_kg DECIMAL(16,8) NOT NULL DEFAULT 0,
    n2o_kg DECIMAL(16,8) NOT NULL DEFAULT 0,
    co2e_kg DECIMAL(16,6) NOT NULL DEFAULT 0,
    avoided_co2e_kg DECIMAL(16,6) NOT NULL DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eol_mr_mass CHECK (mass_kg >= 0),
    CONSTRAINT chk_eol_mr_co2_fossil CHECK (co2_fossil_kg >= 0),
    CONSTRAINT chk_eol_mr_co2_biogenic CHECK (co2_biogenic_kg >= 0),
    CONSTRAINT chk_eol_mr_ch4 CHECK (ch4_kg >= 0),
    CONSTRAINT chk_eol_mr_n2o CHECK (n2o_kg >= 0),
    CONSTRAINT chk_eol_mr_co2e CHECK (co2e_kg >= 0),
    CONSTRAINT chk_eol_mr_avoided CHECK (avoided_co2e_kg >= 0),
    CONSTRAINT chk_eol_mr_treatment CHECK (treatment_method IN (
        'LANDFILL', 'INCINERATION', 'INCINERATION_WTE', 'RECYCLING',
        'COMPOSTING', 'ANAEROBIC_DIGESTION', 'OPEN_BURNING', 'WASTEWATER'
    ))
);

CREATE INDEX idx_eol_mr_calc ON end_of_life_treatment_service.gl_eol_material_results(calc_id);
CREATE INDEX idx_eol_mr_material ON end_of_life_treatment_service.gl_eol_material_results(material_type);
CREATE INDEX idx_eol_mr_treatment ON end_of_life_treatment_service.gl_eol_material_results(treatment_method);
CREATE INDEX idx_eol_mr_calc_material ON end_of_life_treatment_service.gl_eol_material_results(calc_id, material_type);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_material_results IS 'Per material x treatment breakdown of emissions for each calculation';

-- =====================================================================================
-- TABLE 14: gl_eol_avoided_emissions (OPERATIONAL)
-- Description: Recycling credits and energy recovery credits (separate from gross)
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_avoided_emissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    source_type VARCHAR(100) NOT NULL,
    material_type VARCHAR(100),
    avoided_co2e_kg DECIMAL(16,6) NOT NULL DEFAULT 0,
    description VARCHAR(500),
    methodology VARCHAR(200),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eol_ae_avoided CHECK (avoided_co2e_kg >= 0),
    CONSTRAINT chk_eol_ae_source CHECK (source_type IN (
        'RECYCLING_CREDIT', 'ENERGY_RECOVERY_CREDIT', 'COMPOSTING_CREDIT',
        'AD_BIOGAS_CREDIT', 'MODULE_D_CREDIT'
    ))
);

CREATE INDEX idx_eol_ae_calc ON end_of_life_treatment_service.gl_eol_avoided_emissions(calc_id);
CREATE INDEX idx_eol_ae_source ON end_of_life_treatment_service.gl_eol_avoided_emissions(source_type);
CREATE INDEX idx_eol_ae_material ON end_of_life_treatment_service.gl_eol_avoided_emissions(material_type);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_avoided_emissions IS 'Avoided emissions breakdown: recycling credits, energy recovery credits, Module D';

-- =====================================================================================
-- TABLE 15: gl_eol_compliance_checks (HYPERTABLE - 30-day chunks)
-- Description: Compliance check results per calculation
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_compliance_checks (
    check_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    calc_id UUID NOT NULL,
    framework VARCHAR(100) NOT NULL,
    compliant BOOLEAN NOT NULL,
    score DECIMAL(5,4),
    findings JSONB NOT NULL DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eol_cc_score CHECK (score IS NULL OR (score >= 0 AND score <= 1)),
    CONSTRAINT chk_eol_cc_framework CHECK (framework IN (
        'ghg_protocol', 'iso_14064', 'csrd_esrs', 'cdp', 'sbti', 'sb_253', 'gri'
    ))
);

-- Convert to hypertable with 30-day chunks
SELECT create_hypertable(
    'end_of_life_treatment_service.gl_eol_compliance_checks',
    'checked_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_eol_cc_tenant ON end_of_life_treatment_service.gl_eol_compliance_checks(tenant_id, checked_at DESC);
CREATE INDEX idx_eol_cc_calc ON end_of_life_treatment_service.gl_eol_compliance_checks(calc_id);
CREATE INDEX idx_eol_cc_framework ON end_of_life_treatment_service.gl_eol_compliance_checks(framework);
CREATE INDEX idx_eol_cc_compliant ON end_of_life_treatment_service.gl_eol_compliance_checks(compliant);
CREATE INDEX idx_eol_cc_findings ON end_of_life_treatment_service.gl_eol_compliance_checks USING GIN(findings);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_compliance_checks IS 'Compliance check results (HYPERTABLE, 30-day chunks) for 7 regulatory frameworks';

-- =====================================================================================
-- TABLE 16: gl_eol_aggregations (HYPERTABLE - 30-day chunks)
-- Description: Period aggregations by treatment/material/category
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_aggregations (
    agg_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    org_id VARCHAR(200) NOT NULL,
    period VARCHAR(20) NOT NULL,
    year INT NOT NULL,
    total_mass_tonnes DECIMAL(16,6) NOT NULL DEFAULT 0,
    gross_tco2e DECIMAL(16,8) NOT NULL DEFAULT 0,
    avoided_tco2e DECIMAL(16,8) NOT NULL DEFAULT 0,
    net_tco2e DECIMAL(16,8) NOT NULL DEFAULT 0,
    by_treatment JSONB NOT NULL DEFAULT '{}',
    by_material JSONB NOT NULL DEFAULT '{}',
    by_category JSONB NOT NULL DEFAULT '{}',
    calculation_count INT NOT NULL DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eol_agg_period CHECK (period IN ('DAILY', 'MONTHLY', 'QUARTERLY', 'ANNUAL')),
    CONSTRAINT chk_eol_agg_year CHECK (year >= 1990 AND year <= 2100),
    CONSTRAINT chk_eol_agg_mass CHECK (total_mass_tonnes >= 0),
    CONSTRAINT chk_eol_agg_gross CHECK (gross_tco2e >= 0),
    CONSTRAINT chk_eol_agg_avoided CHECK (avoided_tco2e >= 0),
    CONSTRAINT chk_eol_agg_count CHECK (calculation_count >= 0)
);

-- Convert to hypertable with 30-day chunks
SELECT create_hypertable(
    'end_of_life_treatment_service.gl_eol_aggregations',
    'created_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_eol_agg_tenant ON end_of_life_treatment_service.gl_eol_aggregations(tenant_id, created_at DESC);
CREATE INDEX idx_eol_agg_org ON end_of_life_treatment_service.gl_eol_aggregations(org_id);
CREATE INDEX idx_eol_agg_period ON end_of_life_treatment_service.gl_eol_aggregations(period);
CREATE INDEX idx_eol_agg_year ON end_of_life_treatment_service.gl_eol_aggregations(year);
CREATE INDEX idx_eol_agg_by_treatment ON end_of_life_treatment_service.gl_eol_aggregations USING GIN(by_treatment);
CREATE INDEX idx_eol_agg_by_material ON end_of_life_treatment_service.gl_eol_aggregations USING GIN(by_material);
CREATE INDEX idx_eol_agg_by_category ON end_of_life_treatment_service.gl_eol_aggregations USING GIN(by_category);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_aggregations IS 'Aggregated EoL treatment emissions by period (HYPERTABLE, 30-day chunks)';

-- =====================================================================================
-- TABLE 17: gl_eol_provenance_records (OPERATIONAL)
-- Description: SHA-256 hash chain for provenance tracking
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_provenance_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    stage VARCHAR(50) NOT NULL,
    stage_order INT NOT NULL,
    input_hash VARCHAR(128) NOT NULL,
    output_hash VARCHAR(128) NOT NULL,
    agent_id VARCHAR(100) NOT NULL DEFAULT 'GL-MRV-S3-012',
    agent_version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eol_prov_stage CHECK (stage IN (
        'validate', 'classify', 'normalize', 'resolve_efs', 'calculate',
        'allocate', 'aggregate', 'compliance', 'provenance', 'seal'
    )),
    CONSTRAINT chk_eol_prov_order CHECK (stage_order >= 0 AND stage_order <= 10)
);

CREATE INDEX idx_eol_prov_calc ON end_of_life_treatment_service.gl_eol_provenance_records(calc_id);
CREATE INDEX idx_eol_prov_stage ON end_of_life_treatment_service.gl_eol_provenance_records(stage);
CREATE INDEX idx_eol_prov_hash_in ON end_of_life_treatment_service.gl_eol_provenance_records(input_hash);
CREATE INDEX idx_eol_prov_hash_out ON end_of_life_treatment_service.gl_eol_provenance_records(output_hash);
CREATE INDEX idx_eol_prov_calc_order ON end_of_life_treatment_service.gl_eol_provenance_records(calc_id, stage_order);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_provenance_records IS 'SHA-256 hash chain for provenance tracking across 10 pipeline stages';

-- =====================================================================================
-- TABLE 18: gl_eol_audit_trail (OPERATIONAL)
-- Description: Operation audit log for regulatory compliance
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_audit_trail (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    calc_id UUID,
    operation VARCHAR(100) NOT NULL,
    actor VARCHAR(200),
    details JSONB DEFAULT '{}',
    ip_address INET,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eol_audit_op CHECK (operation IN (
        'CREATE', 'READ', 'UPDATE', 'DELETE', 'CALCULATE', 'BATCH_CALCULATE',
        'COMPLIANCE_CHECK', 'UNCERTAINTY_ANALYSIS', 'PORTFOLIO_ANALYSIS',
        'EXPORT', 'API_CALL'
    ))
);

CREATE INDEX idx_eol_audit_tenant ON end_of_life_treatment_service.gl_eol_audit_trail(tenant_id, created_at DESC);
CREATE INDEX idx_eol_audit_calc ON end_of_life_treatment_service.gl_eol_audit_trail(calc_id);
CREATE INDEX idx_eol_audit_operation ON end_of_life_treatment_service.gl_eol_audit_trail(operation);
CREATE INDEX idx_eol_audit_actor ON end_of_life_treatment_service.gl_eol_audit_trail(actor);
CREATE INDEX idx_eol_audit_details ON end_of_life_treatment_service.gl_eol_audit_trail USING GIN(details);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_audit_trail IS 'Operation audit log for regulatory compliance and data governance';

-- =====================================================================================
-- TABLE 19: gl_eol_batch_jobs (SUPPORTING)
-- Description: Batch processing job tracking
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_batch_jobs (
    batch_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    org_id VARCHAR(200) NOT NULL,
    total_products INT NOT NULL DEFAULT 0,
    successful INT NOT NULL DEFAULT 0,
    failed INT NOT NULL DEFAULT 0,
    gross_tco2e DECIMAL(16,8) NOT NULL DEFAULT 0,
    avoided_tco2e DECIMAL(16,8) NOT NULL DEFAULT 0,
    net_tco2e DECIMAL(16,8) NOT NULL DEFAULT 0,
    status VARCHAR(50) NOT NULL DEFAULT 'PENDING',
    processing_time_ms DECIMAL(12,2),
    error_details JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    CONSTRAINT chk_eol_batch_products CHECK (total_products >= 0),
    CONSTRAINT chk_eol_batch_successful CHECK (successful >= 0),
    CONSTRAINT chk_eol_batch_failed CHECK (failed >= 0),
    CONSTRAINT chk_eol_batch_status CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')),
    CONSTRAINT chk_eol_batch_time CHECK (processing_time_ms IS NULL OR processing_time_ms >= 0)
);

CREATE INDEX idx_eol_batch_tenant ON end_of_life_treatment_service.gl_eol_batch_jobs(tenant_id, created_at DESC);
CREATE INDEX idx_eol_batch_org ON end_of_life_treatment_service.gl_eol_batch_jobs(org_id);
CREATE INDEX idx_eol_batch_status ON end_of_life_treatment_service.gl_eol_batch_jobs(status);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_batch_jobs IS 'Batch processing job tracking with status and error details';

-- =====================================================================================
-- TABLE 20: gl_eol_data_quality_scores (SUPPORTING)
-- Description: Data Quality Indicator (DQI) scoring per calculation
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_data_quality_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    overall_score DECIMAL(3,1) NOT NULL,
    temporal_score DECIMAL(3,1) NOT NULL,
    geographical_score DECIMAL(3,1) NOT NULL,
    technological_score DECIMAL(3,1) NOT NULL,
    completeness_score DECIMAL(3,1) NOT NULL,
    reliability_score DECIMAL(3,1) NOT NULL,
    method_used VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eol_dqi_overall CHECK (overall_score >= 1 AND overall_score <= 5),
    CONSTRAINT chk_eol_dqi_temporal CHECK (temporal_score >= 1 AND temporal_score <= 5),
    CONSTRAINT chk_eol_dqi_geo CHECK (geographical_score >= 1 AND geographical_score <= 5),
    CONSTRAINT chk_eol_dqi_tech CHECK (technological_score >= 1 AND technological_score <= 5),
    CONSTRAINT chk_eol_dqi_complete CHECK (completeness_score >= 1 AND completeness_score <= 5),
    CONSTRAINT chk_eol_dqi_reliable CHECK (reliability_score >= 1 AND reliability_score <= 5)
);

CREATE INDEX idx_eol_dqi_calc ON end_of_life_treatment_service.gl_eol_data_quality_scores(calc_id);
CREATE INDEX idx_eol_dqi_overall ON end_of_life_treatment_service.gl_eol_data_quality_scores(overall_score);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_data_quality_scores IS 'Data Quality Indicator (DQI) 5-dimension scoring (temporal, geo, tech, completeness, reliability)';

-- =====================================================================================
-- TABLE 21: gl_eol_uncertainty_results (SUPPORTING)
-- Description: Uncertainty analysis results
-- =====================================================================================

CREATE TABLE end_of_life_treatment_service.gl_eol_uncertainty_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calc_id UUID NOT NULL,
    method VARCHAR(50) NOT NULL,
    iterations INT,
    confidence_level DECIMAL(4,2) NOT NULL DEFAULT 95.0,
    mean_tco2e DECIMAL(16,8) NOT NULL,
    std_dev_tco2e DECIMAL(16,8) NOT NULL,
    ci_lower_tco2e DECIMAL(16,8) NOT NULL,
    ci_upper_tco2e DECIMAL(16,8) NOT NULL,
    relative_uncertainty_pct DECIMAL(8,2),
    sensitivity_indices JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eol_unc_method CHECK (method IN ('monte_carlo', 'analytical', 'ipcc_tier_2')),
    CONSTRAINT chk_eol_unc_iterations CHECK (iterations IS NULL OR iterations >= 0),
    CONSTRAINT chk_eol_unc_confidence CHECK (confidence_level > 0 AND confidence_level <= 100),
    CONSTRAINT chk_eol_unc_bounds CHECK (ci_lower_tco2e <= mean_tco2e AND mean_tco2e <= ci_upper_tco2e),
    CONSTRAINT chk_eol_unc_rel CHECK (relative_uncertainty_pct IS NULL OR relative_uncertainty_pct >= 0)
);

CREATE INDEX idx_eol_unc_calc ON end_of_life_treatment_service.gl_eol_uncertainty_results(calc_id);
CREATE INDEX idx_eol_unc_method ON end_of_life_treatment_service.gl_eol_uncertainty_results(method);

COMMENT ON TABLE end_of_life_treatment_service.gl_eol_uncertainty_results IS 'Uncertainty analysis results (Monte Carlo, analytical, IPCC Tier 2)';

-- =====================================================================================
-- CONTINUOUS AGGREGATES
-- =====================================================================================

-- Continuous Aggregate 1: Daily by Treatment Method
CREATE MATERIALIZED VIEW end_of_life_treatment_service.gl_eol_daily_by_treatment
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', created_at) AS bucket,
    tenant_id,
    method,
    COUNT(*) AS calculation_count,
    SUM(total_mass_tonnes) AS total_mass_tonnes,
    SUM(gross_tco2e) AS total_gross_tco2e,
    SUM(avoided_tco2e) AS total_avoided_tco2e,
    SUM(net_tco2e) AS total_net_tco2e,
    AVG(net_tco2e) AS avg_net_tco2e,
    MIN(net_tco2e) AS min_net_tco2e,
    MAX(net_tco2e) AS max_net_tco2e
FROM end_of_life_treatment_service.gl_eol_calculations
WHERE is_deleted = FALSE AND status = 'COMPLETED'
GROUP BY bucket, tenant_id, method
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'end_of_life_treatment_service.gl_eol_daily_by_treatment',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW end_of_life_treatment_service.gl_eol_daily_by_treatment IS 'Daily aggregation of EoL emissions by calculation method';

-- Continuous Aggregate 2: Monthly by Material (from calculation details via aggregations)
CREATE MATERIALIZED VIEW end_of_life_treatment_service.gl_eol_monthly_by_material
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('30 days', created_at) AS bucket,
    tenant_id,
    org_id,
    year,
    COUNT(*) AS calculation_count,
    SUM(total_mass_tonnes) AS total_mass_tonnes,
    SUM(gross_tco2e) AS total_gross_tco2e,
    SUM(avoided_tco2e) AS total_avoided_tco2e,
    SUM(net_tco2e) AS total_net_tco2e,
    AVG(dqi_score) AS avg_dqi_score,
    STDDEV(net_tco2e) AS stddev_net_tco2e
FROM end_of_life_treatment_service.gl_eol_calculations
WHERE is_deleted = FALSE AND status = 'COMPLETED'
GROUP BY bucket, tenant_id, org_id, year
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'end_of_life_treatment_service.gl_eol_monthly_by_material',
    start_offset => INTERVAL '90 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '6 hours',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW end_of_life_treatment_service.gl_eol_monthly_by_material IS 'Monthly aggregation of EoL emissions by organization and year';

-- =====================================================================================
-- ROW LEVEL SECURITY (RLS) - 10 policies
-- =====================================================================================

-- Enable RLS on all tables with tenant_id
ALTER TABLE end_of_life_treatment_service.gl_eol_calculations ENABLE ROW LEVEL SECURITY;
ALTER TABLE end_of_life_treatment_service.gl_eol_calculation_details ENABLE ROW LEVEL SECURITY;
ALTER TABLE end_of_life_treatment_service.gl_eol_material_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE end_of_life_treatment_service.gl_eol_avoided_emissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE end_of_life_treatment_service.gl_eol_compliance_checks ENABLE ROW LEVEL SECURITY;
ALTER TABLE end_of_life_treatment_service.gl_eol_aggregations ENABLE ROW LEVEL SECURITY;
ALTER TABLE end_of_life_treatment_service.gl_eol_provenance_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE end_of_life_treatment_service.gl_eol_audit_trail ENABLE ROW LEVEL SECURITY;
ALTER TABLE end_of_life_treatment_service.gl_eol_batch_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE end_of_life_treatment_service.gl_eol_uncertainty_results ENABLE ROW LEVEL SECURITY;

-- RLS Policies using app.current_tenant_id
CREATE POLICY eol_calculations_tenant_isolation
    ON end_of_life_treatment_service.gl_eol_calculations
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY eol_compliance_checks_tenant_isolation
    ON end_of_life_treatment_service.gl_eol_compliance_checks
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY eol_aggregations_tenant_isolation
    ON end_of_life_treatment_service.gl_eol_aggregations
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY eol_audit_trail_tenant_isolation
    ON end_of_life_treatment_service.gl_eol_audit_trail
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

CREATE POLICY eol_batch_jobs_tenant_isolation
    ON end_of_life_treatment_service.gl_eol_batch_jobs
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID);

-- RLS for child tables via calc_id join (using subquery for tenant isolation)
CREATE POLICY eol_calculation_details_tenant_isolation
    ON end_of_life_treatment_service.gl_eol_calculation_details
    USING (calc_id IN (
        SELECT calc_id FROM end_of_life_treatment_service.gl_eol_calculations
        WHERE tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID
    ));

CREATE POLICY eol_material_results_tenant_isolation
    ON end_of_life_treatment_service.gl_eol_material_results
    USING (calc_id IN (
        SELECT calc_id FROM end_of_life_treatment_service.gl_eol_calculations
        WHERE tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID
    ));

CREATE POLICY eol_avoided_emissions_tenant_isolation
    ON end_of_life_treatment_service.gl_eol_avoided_emissions
    USING (calc_id IN (
        SELECT calc_id FROM end_of_life_treatment_service.gl_eol_calculations
        WHERE tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID
    ));

CREATE POLICY eol_provenance_records_tenant_isolation
    ON end_of_life_treatment_service.gl_eol_provenance_records
    USING (calc_id IN (
        SELECT calc_id FROM end_of_life_treatment_service.gl_eol_calculations
        WHERE tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID
    ));

CREATE POLICY eol_uncertainty_results_tenant_isolation
    ON end_of_life_treatment_service.gl_eol_uncertainty_results
    USING (calc_id IN (
        SELECT calc_id FROM end_of_life_treatment_service.gl_eol_calculations
        WHERE tenant_id = current_setting('app.current_tenant_id', TRUE)::UUID
    ));

-- =====================================================================================
-- SEED DATA: MATERIAL EMISSION FACTORS (15 materials x 7 treatments = ~105 rows)
-- Source: EPA WARM v16, DEFRA 2024, IPCC 2006
-- =====================================================================================

INSERT INTO end_of_life_treatment_service.gl_eol_material_emission_factors
(material_type, treatment_method, ef_kgco2e_per_kg, co2_fossil_kgco2_per_kg, co2_biogenic_kgco2_per_kg, ch4_kgch4_per_kg, n2o_kgn2o_per_kg, ef_source, region, year) VALUES
-- HDPE (High-Density Polyethylene)
('HDPE', 'LANDFILL', 0.018, 0.018, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
('HDPE', 'INCINERATION', 2.850, 2.850, 0.000, 0.00002, 0.00001, 'EPA WARM v16', 'GLOBAL', 2024),
('HDPE', 'RECYCLING', 0.042, 0.042, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
-- PET (Polyethylene Terephthalate)
('PET', 'LANDFILL', 0.018, 0.018, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
('PET', 'INCINERATION', 2.290, 2.290, 0.000, 0.00002, 0.00001, 'EPA WARM v16', 'GLOBAL', 2024),
('PET', 'RECYCLING', 0.038, 0.038, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
-- LDPE (Low-Density Polyethylene)
('LDPE', 'LANDFILL', 0.018, 0.018, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
('LDPE', 'INCINERATION', 2.850, 2.850, 0.000, 0.00002, 0.00001, 'EPA WARM v16', 'GLOBAL', 2024),
('LDPE', 'RECYCLING', 0.045, 0.045, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
-- PP (Polypropylene)
('PP', 'LANDFILL', 0.018, 0.018, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
('PP', 'INCINERATION', 2.750, 2.750, 0.000, 0.00002, 0.00001, 'EPA WARM v16', 'GLOBAL', 2024),
('PP', 'RECYCLING', 0.040, 0.040, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
-- STEEL (Carbon Steel)
('STEEL', 'LANDFILL', 0.021, 0.021, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
('STEEL', 'RECYCLING', 0.035, 0.035, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
-- ALUMINUM
('ALUMINUM', 'LANDFILL', 0.021, 0.021, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
('ALUMINUM', 'RECYCLING', 0.065, 0.065, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
-- GLASS
('GLASS', 'LANDFILL', 0.021, 0.021, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
('GLASS', 'RECYCLING', 0.028, 0.028, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
-- CARDBOARD
('CARDBOARD', 'LANDFILL', 0.893, 0.000, 0.893, 0.01200, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
('CARDBOARD', 'INCINERATION', 0.052, 0.005, 0.760, 0.00003, 0.00001, 'EPA WARM v16', 'GLOBAL', 2024),
('CARDBOARD', 'RECYCLING', 0.025, 0.025, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
('CARDBOARD', 'COMPOSTING', 0.147, 0.000, 0.000, 0.00400, 0.00030, 'EPA WARM v16', 'GLOBAL', 2024),
-- PAPER
('PAPER', 'LANDFILL', 0.960, 0.000, 0.960, 0.01300, 0.00000, 'DEFRA 2024', 'GLOBAL', 2024),
('PAPER', 'INCINERATION', 0.052, 0.005, 0.780, 0.00003, 0.00001, 'DEFRA 2024', 'GLOBAL', 2024),
('PAPER', 'RECYCLING', 0.021, 0.021, 0.000, 0.00000, 0.00000, 'DEFRA 2024', 'GLOBAL', 2024),
-- WOOD
('WOOD', 'LANDFILL', 0.679, 0.000, 0.679, 0.00900, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
('WOOD', 'INCINERATION', 0.048, 0.000, 0.720, 0.00003, 0.00001, 'EPA WARM v16', 'GLOBAL', 2024),
('WOOD', 'RECYCLING', 0.015, 0.015, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
('WOOD', 'COMPOSTING', 0.085, 0.000, 0.000, 0.00300, 0.00020, 'EPA WARM v16', 'GLOBAL', 2024),
-- TEXTILES
('TEXTILES', 'LANDFILL', 0.990, 0.000, 0.000, 0.01100, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
('TEXTILES', 'INCINERATION', 1.890, 0.945, 0.945, 0.00003, 0.00001, 'EPA WARM v16', 'GLOBAL', 2024),
('TEXTILES', 'RECYCLING', 0.030, 0.030, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
-- RUBBER
('RUBBER', 'LANDFILL', 0.018, 0.018, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
('RUBBER', 'INCINERATION', 3.200, 3.200, 0.000, 0.00003, 0.00001, 'EPA WARM v16', 'GLOBAL', 2024),
-- ELECTRONICS (mixed PCB, metals, plastics)
('ELECTRONICS', 'LANDFILL', 0.018, 0.018, 0.000, 0.00000, 0.00000, 'DEFRA 2024', 'GLOBAL', 2024),
('ELECTRONICS', 'INCINERATION', 1.200, 0.960, 0.000, 0.00002, 0.00001, 'DEFRA 2024', 'GLOBAL', 2024),
('ELECTRONICS', 'RECYCLING', 0.055, 0.055, 0.000, 0.00000, 0.00000, 'DEFRA 2024', 'GLOBAL', 2024),
-- FOOD_WASTE
('FOOD_WASTE', 'LANDFILL', 0.518, 0.000, 0.000, 0.01500, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
('FOOD_WASTE', 'INCINERATION', 0.039, 0.000, 0.200, 0.00003, 0.00001, 'EPA WARM v16', 'GLOBAL', 2024),
('FOOD_WASTE', 'COMPOSTING', 0.062, 0.000, 0.000, 0.00400, 0.00030, 'EPA WARM v16', 'GLOBAL', 2024),
('FOOD_WASTE', 'ANAEROBIC_DIGESTION', 0.017, 0.000, 0.000, 0.00100, 0.00010, 'EPA WARM v16', 'GLOBAL', 2024),
-- CONCRETE
('CONCRETE', 'LANDFILL', 0.003, 0.003, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024),
('CONCRETE', 'RECYCLING', 0.008, 0.008, 0.000, 0.00000, 0.00000, 'EPA WARM v16', 'GLOBAL', 2024);

-- =====================================================================================
-- SEED DATA: PRODUCT COMPOSITIONS (20 compositions across categories)
-- =====================================================================================

INSERT INTO end_of_life_treatment_service.gl_eol_product_compositions
(product_category, material_type, fraction, weight_fraction_kg, source) VALUES
-- CONSUMER_ELECTRONICS (smartphone ~0.180 kg)
('CONSUMER_ELECTRONICS', 'GLASS', 0.32, 0.058, 'Industry average'),
('CONSUMER_ELECTRONICS', 'ALUMINUM', 0.24, 0.043, 'Industry average'),
('CONSUMER_ELECTRONICS', 'ELECTRONICS', 0.20, 0.036, 'Industry average'),
('CONSUMER_ELECTRONICS', 'STEEL', 0.12, 0.022, 'Industry average'),
('CONSUMER_ELECTRONICS', 'LDPE', 0.12, 0.022, 'Industry average'),
-- PACKAGING_FOOD (typical food packaging ~0.050 kg)
('PACKAGING_FOOD', 'CARDBOARD', 0.40, 0.020, 'Industry average'),
('PACKAGING_FOOD', 'LDPE', 0.25, 0.013, 'Industry average'),
('PACKAGING_FOOD', 'PET', 0.20, 0.010, 'Industry average'),
('PACKAGING_FOOD', 'PAPER', 0.15, 0.008, 'Industry average'),
-- APPLIANCES_LARGE (washing machine ~70 kg)
('APPLIANCES_LARGE', 'STEEL', 0.55, 38.50, 'Industry average'),
('APPLIANCES_LARGE', 'ELECTRONICS', 0.15, 10.50, 'Industry average'),
('APPLIANCES_LARGE', 'PP', 0.15, 10.50, 'Industry average'),
('APPLIANCES_LARGE', 'RUBBER', 0.08, 5.60, 'Industry average'),
('APPLIANCES_LARGE', 'GLASS', 0.07, 4.90, 'Industry average'),
-- TEXTILES_APPAREL (garment ~0.300 kg)
('TEXTILES_APPAREL', 'TEXTILES', 0.90, 0.270, 'Industry average'),
('TEXTILES_APPAREL', 'LDPE', 0.05, 0.015, 'Industry average'),
('TEXTILES_APPAREL', 'CARDBOARD', 0.05, 0.015, 'Industry average'),
-- FURNITURE (office chair ~12 kg)
('FURNITURE', 'STEEL', 0.40, 4.80, 'Industry average'),
('FURNITURE', 'PP', 0.30, 3.60, 'Industry average'),
('FURNITURE', 'TEXTILES', 0.20, 2.40, 'Industry average'),
('FURNITURE', 'WOOD', 0.10, 1.20, 'Industry average');

-- =====================================================================================
-- SEED DATA: REGIONAL TREATMENT MIXES (12 entries across 4 regions)
-- =====================================================================================

INSERT INTO end_of_life_treatment_service.gl_eol_regional_treatment_mixes
(region, treatment_method, fraction, year, source) VALUES
-- US treatment mix (EPA 2022 data)
('US', 'LANDFILL', 0.50, 2024, 'EPA Facts and Figures 2022'),
('US', 'RECYCLING', 0.32, 2024, 'EPA Facts and Figures 2022'),
('US', 'INCINERATION_WTE', 0.12, 2024, 'EPA Facts and Figures 2022'),
('US', 'COMPOSTING', 0.06, 2024, 'EPA Facts and Figures 2022'),
-- EU treatment mix (Eurostat 2022 data)
('EU', 'LANDFILL', 0.23, 2024, 'Eurostat waste statistics 2022'),
('EU', 'RECYCLING', 0.40, 2024, 'Eurostat waste statistics 2022'),
('EU', 'INCINERATION_WTE', 0.27, 2024, 'Eurostat waste statistics 2022'),
('EU', 'COMPOSTING', 0.10, 2024, 'Eurostat waste statistics 2022'),
-- GLOBAL average treatment mix
('GLOBAL', 'LANDFILL', 0.40, 2024, 'World Bank What a Waste 2.0'),
('GLOBAL', 'RECYCLING', 0.18, 2024, 'World Bank What a Waste 2.0'),
('GLOBAL', 'INCINERATION', 0.11, 2024, 'World Bank What a Waste 2.0'),
('GLOBAL', 'OPEN_BURNING', 0.31, 2024, 'World Bank What a Waste 2.0');

-- =====================================================================================
-- SEED DATA: LANDFILL FOD PARAMETERS (15 materials x TEMPERATE)
-- =====================================================================================

INSERT INTO end_of_life_treatment_service.gl_eol_landfill_parameters
(material_type, climate_zone, doc, docf, mcf, k_decay, f_ch4, ox_factor) VALUES
('CARDBOARD', 'TEMPERATE', 0.40, 0.50, 1.0, 0.070, 0.50, 0.10),
('PAPER', 'TEMPERATE', 0.40, 0.50, 1.0, 0.070, 0.50, 0.10),
('WOOD', 'TEMPERATE', 0.43, 0.50, 1.0, 0.035, 0.50, 0.10),
('TEXTILES', 'TEMPERATE', 0.24, 0.50, 1.0, 0.080, 0.50, 0.10),
('FOOD_WASTE', 'TEMPERATE', 0.15, 0.50, 1.0, 0.185, 0.50, 0.10),
('HDPE', 'TEMPERATE', 0.00, 0.00, 1.0, 0.000, 0.50, 0.10),
('PET', 'TEMPERATE', 0.00, 0.00, 1.0, 0.000, 0.50, 0.10),
('LDPE', 'TEMPERATE', 0.00, 0.00, 1.0, 0.000, 0.50, 0.10),
('PP', 'TEMPERATE', 0.00, 0.00, 1.0, 0.000, 0.50, 0.10),
('STEEL', 'TEMPERATE', 0.00, 0.00, 1.0, 0.000, 0.50, 0.10),
('ALUMINUM', 'TEMPERATE', 0.00, 0.00, 1.0, 0.000, 0.50, 0.10),
('GLASS', 'TEMPERATE', 0.00, 0.00, 1.0, 0.000, 0.50, 0.10),
('RUBBER', 'TEMPERATE', 0.39, 0.50, 1.0, 0.040, 0.50, 0.10),
('ELECTRONICS', 'TEMPERATE', 0.00, 0.00, 1.0, 0.000, 0.50, 0.10),
('CONCRETE', 'TEMPERATE', 0.00, 0.00, 1.0, 0.000, 0.50, 0.10);

-- =====================================================================================
-- SEED DATA: INCINERATION PARAMETERS (15 materials)
-- =====================================================================================

INSERT INTO end_of_life_treatment_service.gl_eol_incineration_parameters
(material_type, dry_matter, carbon_fraction, fossil_carbon, oxidation_factor, calorific_value_mj_kg, ch4_ef_kg_per_tonne, n2o_ef_kg_per_tonne) VALUES
('HDPE', 0.99, 0.860, 1.00, 1.0, 43.3, 0.020, 0.010),
('PET', 0.99, 0.625, 1.00, 1.0, 22.0, 0.020, 0.010),
('LDPE', 0.99, 0.860, 1.00, 1.0, 43.3, 0.020, 0.010),
('PP', 0.99, 0.860, 1.00, 1.0, 43.3, 0.020, 0.010),
('CARDBOARD', 0.90, 0.460, 0.01, 1.0, 15.6, 0.030, 0.010),
('PAPER', 0.90, 0.460, 0.01, 1.0, 15.6, 0.030, 0.010),
('WOOD', 0.85, 0.500, 0.00, 1.0, 15.4, 0.030, 0.010),
('TEXTILES', 0.90, 0.500, 0.50, 1.0, 19.0, 0.030, 0.010),
('RUBBER', 0.90, 0.670, 1.00, 1.0, 32.0, 0.030, 0.010),
('FOOD_WASTE', 0.40, 0.150, 0.00, 1.0, 4.0, 0.030, 0.010),
('ELECTRONICS', 0.95, 0.300, 0.80, 1.0, 12.0, 0.020, 0.010),
('STEEL', 1.00, 0.000, 0.00, 1.0, 0.0, 0.000, 0.000),
('ALUMINUM', 1.00, 0.000, 0.00, 1.0, 0.0, 0.000, 0.000),
('GLASS', 1.00, 0.000, 0.00, 1.0, 0.0, 0.000, 0.000),
('CONCRETE', 1.00, 0.000, 0.00, 1.0, 0.0, 0.000, 0.000);

-- =====================================================================================
-- SEED DATA: RECYCLING FACTORS (15 materials)
-- Units: kgCO2e per kg of material
-- =====================================================================================

INSERT INTO end_of_life_treatment_service.gl_eol_recycling_factors
(material_type, transport_ef, mrf_ef, total_processing_ef, avoided_ef, recycling_rate, source, region) VALUES
('HDPE', 0.012, 0.030, 0.042, 1.630, 0.85, 'EPA WARM v16', 'GLOBAL'),
('PET', 0.012, 0.026, 0.038, 1.280, 0.85, 'EPA WARM v16', 'GLOBAL'),
('LDPE', 0.012, 0.033, 0.045, 1.630, 0.80, 'EPA WARM v16', 'GLOBAL'),
('PP', 0.012, 0.028, 0.040, 1.580, 0.85, 'EPA WARM v16', 'GLOBAL'),
('STEEL', 0.015, 0.020, 0.035, 1.580, 0.90, 'EPA WARM v16', 'GLOBAL'),
('ALUMINUM', 0.015, 0.050, 0.065, 9.080, 0.90, 'EPA WARM v16', 'GLOBAL'),
('GLASS', 0.012, 0.016, 0.028, 0.315, 0.85, 'EPA WARM v16', 'GLOBAL'),
('CARDBOARD', 0.010, 0.015, 0.025, 3.640, 0.90, 'EPA WARM v16', 'GLOBAL'),
('PAPER', 0.010, 0.011, 0.021, 3.640, 0.90, 'EPA WARM v16', 'GLOBAL'),
('WOOD', 0.008, 0.007, 0.015, 0.894, 0.75, 'EPA WARM v16', 'GLOBAL'),
('TEXTILES', 0.012, 0.018, 0.030, 4.320, 0.70, 'EPA WARM v16', 'GLOBAL'),
('ELECTRONICS', 0.020, 0.035, 0.055, 0.520, 0.60, 'DEFRA 2024', 'GLOBAL'),
('RUBBER', 0.015, 0.025, 0.040, 1.200, 0.65, 'EPA WARM v16', 'GLOBAL'),
('CONCRETE', 0.005, 0.003, 0.008, 0.082, 0.90, 'EPA WARM v16', 'GLOBAL'),
('FOOD_WASTE', 0.000, 0.000, 0.000, 0.000, 0.00, 'N/A', 'GLOBAL');

-- =====================================================================================
-- SEED DATA: PRODUCT WEIGHT DEFAULTS
-- =====================================================================================

INSERT INTO end_of_life_treatment_service.gl_eol_product_weight_defaults
(product_category, avg_weight_kg, min_weight_kg, max_weight_kg, source) VALUES
('CONSUMER_ELECTRONICS', 0.180, 0.100, 1.500, 'Industry average'),
('PACKAGING_FOOD', 0.050, 0.005, 0.500, 'Industry average'),
('APPLIANCES_LARGE', 70.000, 30.000, 120.000, 'Industry average'),
('APPLIANCES_SMALL', 3.500, 0.500, 15.000, 'Industry average'),
('TEXTILES_APPAREL', 0.300, 0.050, 2.000, 'Industry average'),
('FURNITURE', 12.000, 2.000, 80.000, 'Industry average'),
('AUTOMOTIVE_PARTS', 5.000, 0.100, 50.000, 'Industry average'),
('BUILDING_MATERIALS', 25.000, 1.000, 500.000, 'Industry average'),
('TOYS', 0.500, 0.050, 5.000, 'Industry average'),
('PACKAGING_BEVERAGE', 0.025, 0.010, 0.100, 'Industry average');

-- =====================================================================================
-- SEED DATA: GAS COLLECTION FACTORS
-- =====================================================================================

INSERT INTO end_of_life_treatment_service.gl_eol_gas_collection_factors
(landfill_type, collection_efficiency, flare_efficiency, description) VALUES
('MANAGED_ANAEROBIC', 0.75, 0.99, 'Managed anaerobic landfill with active gas collection'),
('MANAGED_SEMI', 0.50, 0.99, 'Semi-aerobic managed landfill'),
('UNMANAGED_DEEP', 0.00, 0.00, 'Unmanaged deep landfill (>5m depth, no gas collection)'),
('UNMANAGED_SHALLOW', 0.00, 0.00, 'Unmanaged shallow landfill (<5m depth)'),
('SANITARY_WITH_GAS', 0.85, 0.99, 'Modern sanitary landfill with gas-to-energy'),
('SANITARY_NO_GAS', 0.00, 0.00, 'Sanitary landfill without gas collection'),
('BIOREACTOR', 0.90, 0.99, 'Bioreactor landfill with leachate recirculation');

-- =====================================================================================
-- SEED DATA: ENERGY RECOVERY FACTORS
-- =====================================================================================

INSERT INTO end_of_life_treatment_service.gl_eol_energy_recovery_factors
(region, wte_efficiency, displaced_grid_ef, year, source) VALUES
('US', 0.22, 0.420, 2024, 'EPA eGRID 2023'),
('EU', 0.27, 0.295, 2024, 'EEA GHG intensity 2023'),
('UK', 0.25, 0.212, 2024, 'DEFRA 2024'),
('CN', 0.18, 0.581, 2024, 'IEA 2023'),
('JP', 0.20, 0.470, 2024, 'IEA 2023'),
('GLOBAL', 0.22, 0.475, 2024, 'IEA World Average 2023');

-- =====================================================================================
-- SEED DATA: COMPOSTING/AD FACTORS
-- =====================================================================================

INSERT INTO end_of_life_treatment_service.gl_eol_composting_ad_factors
(treatment_type, ch4_ef, n2o_ef, biogas_yield, capture_efficiency) VALUES
('COMPOSTING_INDUSTRIAL', 0.00400, 0.00030, NULL, NULL),
('COMPOSTING_HOME', 0.01000, 0.00060, NULL, NULL),
('ANAEROBIC_DIGESTION_ENCLOSED', 0.00100, 0.00010, 120.0, 0.98),
('ANAEROBIC_DIGESTION_OPEN', 0.00500, 0.00020, 100.0, 0.70),
('ANAEROBIC_DIGESTION_DRY', 0.00200, 0.00015, 80.0, 0.90);

-- =====================================================================================
-- AGENT REGISTRY ENTRY
-- =====================================================================================

INSERT INTO agent_registry.agents (
    agent_code,
    agent_name,
    agent_version,
    agent_type,
    agent_category,
    description,
    status,
    metadata
) VALUES (
    'GL-MRV-S3-012',
    'End-of-Life Treatment of Sold Products Agent',
    '1.0.0',
    'CALCULATION',
    'SCOPE3_EMISSIONS',
    'AGENT-MRV-025: Scope 3 Category 12 - End-of-Life Treatment of Sold Products. Calculates emissions from downstream waste disposal and treatment of products sold by the reporting company. Supports landfill (IPCC FOD), incineration/WtE, recycling (cut-off with avoided credits), composting, anaerobic digestion, and open burning. 4 calculation methods: waste-type-specific (BOM), average-data (composite EFs), producer-specific (EPD/PCF), and hybrid (waterfall). 15 material types, 7 treatment pathways, circularity scoring, portfolio analysis.',
    'ACTIVE',
    jsonb_build_object(
        'scope3_category', 12,
        'category_name', 'End-of-Life Treatment of Sold Products',
        'calculation_methods', jsonb_build_array('waste_type_specific', 'average_data', 'producer_specific', 'hybrid'),
        'material_types', jsonb_build_array(
            'HDPE', 'PET', 'LDPE', 'PP', 'STEEL', 'ALUMINUM', 'GLASS',
            'CARDBOARD', 'PAPER', 'WOOD', 'TEXTILES', 'RUBBER',
            'ELECTRONICS', 'FOOD_WASTE', 'CONCRETE'
        ),
        'treatment_methods', jsonb_build_array(
            'LANDFILL', 'INCINERATION', 'INCINERATION_WTE', 'RECYCLING',
            'COMPOSTING', 'ANAEROBIC_DIGESTION', 'OPEN_BURNING', 'WASTEWATER'
        ),
        'frameworks', jsonb_build_array(
            'GHG Protocol Scope 3', 'ISO 14064-1', 'CSRD ESRS E1/E5',
            'CDP', 'SBTi', 'SB 253', 'GRI 305/306'
        ),
        'material_types_count', 15,
        'treatment_methods_count', 8,
        'supports_fod_model', true,
        'supports_wte_credits', true,
        'supports_recycling_credits', true,
        'supports_circularity_scoring', true,
        'supports_portfolio_analysis', true,
        'supports_epd_module_c_d', true,
        'double_counting_prevention', 'Separate from Cat 5 (own operations waste)',
        'default_ef_source', 'EPA WARM v16',
        'default_gwp', 'AR5',
        'schema', 'end_of_life_treatment_service',
        'table_prefix', 'gl_eol_',
        'hypertables', jsonb_build_array('gl_eol_calculations', 'gl_eol_compliance_checks', 'gl_eol_aggregations'),
        'continuous_aggregates', jsonb_build_array('gl_eol_daily_by_treatment', 'gl_eol_monthly_by_material'),
        'migration_version', 'V076',
        'api_prefix', '/api/v1/end-of-life-treatment',
        'endpoints_count', 22
    )
)
ON CONFLICT (agent_code) DO UPDATE SET
    agent_name = EXCLUDED.agent_name,
    agent_version = EXCLUDED.agent_version,
    description = EXCLUDED.description,
    status = EXCLUDED.status,
    metadata = EXCLUDED.metadata,
    updated_at = NOW();

-- =====================================================================================
-- FINAL COMMENTS
-- =====================================================================================

COMMENT ON SCHEMA end_of_life_treatment_service IS 'Updated: AGENT-MRV-025 complete with 21 tables, 3 hypertables, 2 continuous aggregates, 10 RLS policies, ~120 seed records';

-- =====================================================================================
-- END OF MIGRATION V076
-- =====================================================================================
-- Total Lines: ~1300
-- Total Tables: 21 (10 reference + 8 operational + 3 supporting)
-- Total Hypertables: 3 (calculations 7-day, compliance_checks 30-day, aggregations 30-day)
-- Total Continuous Aggregates: 2 (gl_eol_daily_by_treatment, gl_eol_monthly_by_material)
-- Total RLS Policies: 10
-- Total Indexes: ~100
-- Total Check Constraints: ~90
-- Total Seed Records: ~120
--   Material Emission Factors: 44 (15 materials x avg ~3 treatments)
--   Product Compositions: 21 (5 product categories)
--   Regional Treatment Mixes: 12 (3 regions x 4 pathways)
--   Landfill FOD Parameters: 15
--   Incineration Parameters: 15
--   Recycling Factors: 15
--   Product Weight Defaults: 10
--   Gas Collection Factors: 7
--   Energy Recovery Factors: 6
--   Composting/AD Factors: 5
-- Agent Registry: 1
-- =====================================================================================
