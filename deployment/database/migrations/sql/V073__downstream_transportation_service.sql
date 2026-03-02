-- =====================================================================================
-- Migration: V073__downstream_transportation_service.sql
-- Description: AGENT-MRV-022 Downstream Transportation & Distribution (Scope 3 Category 9)
-- Agent: GL-MRV-S3-009
-- Framework: GHG Protocol Scope 3 Standard, ISO 14083, GLEC Framework v3.0,
--            DEFRA 2024, EPA SmartWay, IMO 2020, ICAO 2024, ICC Incoterms 2020
-- Created: 2026-02-27
-- =====================================================================================
-- Schema: downstream_transportation_service
-- Tables: 21 (16 operational + 5 reference)
-- Hypertables: 3 (calculations, calculation_details, aggregations)
-- Continuous Aggregates: 2 (hourly_emissions, daily_emissions)
-- RLS: Enabled on ALL operational tables with tenant_id isolation
-- Seed Data: 100+ records (transport EFs, cold chain, warehouse EFs, last-mile,
--            EEIO factors, currency rates, CPI deflators, grid EFs,
--            distribution channels, incoterm classification, load factors)
-- =====================================================================================

-- =====================================================================================
-- SCHEMA CREATION
-- =====================================================================================

CREATE SCHEMA IF NOT EXISTS downstream_transportation_service;

COMMENT ON SCHEMA downstream_transportation_service IS 'AGENT-MRV-022: Downstream Transportation & Distribution - Scope 3 Category 9 emission calculations (outbound transport, warehousing, last-mile delivery, cold chain, return logistics)';

-- =====================================================================================
-- TABLE 1: gl_dto_transport_emission_factors (REFERENCE)
-- Description: Emission factors per tonne-km for all transport modes (26 vehicle types)
-- Source: DEFRA 2024, IMO 2020, ICAO 2024, GLEC v3, Industry averages
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_transport_emission_factors (
    vehicle_type VARCHAR(50) PRIMARY KEY,
    mode VARCHAR(30) NOT NULL,
    ef_per_tkm DECIMAL(12,8) NOT NULL,
    wtt_per_tkm DECIMAL(12,8) NOT NULL,
    source VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dto_tef_ef_positive CHECK (ef_per_tkm >= 0),
    CONSTRAINT chk_dto_tef_wtt_positive CHECK (wtt_per_tkm >= 0),
    CONSTRAINT chk_dto_tef_mode CHECK (mode IN (
        'road', 'rail', 'maritime', 'air', 'inland_waterway',
        'courier', 'last_mile', 'pipeline', 'intermodal'
    ))
);

CREATE INDEX idx_dto_tef_mode ON downstream_transportation_service.gl_dto_transport_emission_factors(mode);
CREATE INDEX idx_dto_tef_source ON downstream_transportation_service.gl_dto_transport_emission_factors(source);
CREATE INDEX idx_dto_tef_active ON downstream_transportation_service.gl_dto_transport_emission_factors(is_active);

COMMENT ON TABLE downstream_transportation_service.gl_dto_transport_emission_factors IS 'Transport emission factors per tonne-km for 26 vehicle types across road, rail, maritime, air, inland waterway, courier, and last-mile modes';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_transport_emission_factors.vehicle_type IS 'Vehicle type code (e.g., lgv_petrol, articulated_truck, container_ship_large, freighter_wide)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_transport_emission_factors.ef_per_tkm IS 'Tank-to-wheel emission factor in kgCO2e per tonne-km';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_transport_emission_factors.wtt_per_tkm IS 'Well-to-tank upstream emission factor in kgCO2e per tonne-km';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_transport_emission_factors.source IS 'Source: DEFRA 2024, IMO 2020, ICAO 2024, GLEC v3, Industry avg';

-- =====================================================================================
-- TABLE 2: gl_dto_cold_chain_factors (REFERENCE)
-- Description: Cold chain uplift factors by temperature regime and transport mode
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_cold_chain_factors (
    temperature_regime VARCHAR(30) PRIMARY KEY,
    road_uplift DECIMAL(6,4) NOT NULL DEFAULT 1.0000,
    rail_uplift DECIMAL(6,4) NOT NULL DEFAULT 1.0000,
    maritime_uplift DECIMAL(6,4) NOT NULL DEFAULT 1.0000,
    air_uplift DECIMAL(6,4) NOT NULL DEFAULT 1.0000,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dto_ccf_road_positive CHECK (road_uplift >= 1.0),
    CONSTRAINT chk_dto_ccf_rail_positive CHECK (rail_uplift >= 1.0),
    CONSTRAINT chk_dto_ccf_maritime_positive CHECK (maritime_uplift >= 1.0),
    CONSTRAINT chk_dto_ccf_air_positive CHECK (air_uplift >= 1.0),
    CONSTRAINT chk_dto_ccf_regime CHECK (temperature_regime IN (
        'ambient', 'chilled', 'frozen', 'deep_frozen', 'controlled'
    ))
);

CREATE INDEX idx_dto_ccf_active ON downstream_transportation_service.gl_dto_cold_chain_factors(is_active);

COMMENT ON TABLE downstream_transportation_service.gl_dto_cold_chain_factors IS 'Cold chain uplift factors by temperature regime (ambient, chilled 2-8C, frozen -18 to -25C, deep frozen <-25C, controlled 15-25C)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_cold_chain_factors.temperature_regime IS 'Temperature regime: ambient, chilled, frozen, deep_frozen, controlled';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_cold_chain_factors.road_uplift IS 'Emission multiplier for road transport under this temperature regime (1.0 = no uplift)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_cold_chain_factors.rail_uplift IS 'Emission multiplier for rail transport under this temperature regime';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_cold_chain_factors.maritime_uplift IS 'Emission multiplier for maritime transport (reefer container)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_cold_chain_factors.air_uplift IS 'Emission multiplier for air transport under this temperature regime';

-- =====================================================================================
-- TABLE 3: gl_dto_warehouse_emission_factors (REFERENCE)
-- Description: Warehouse emission factors per square meter per year by type
-- Source: CIBSE TM46, Industry averages
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_warehouse_emission_factors (
    warehouse_type VARCHAR(50) PRIMARY KEY,
    electricity_ef DECIMAL(10,4) NOT NULL,
    gas_ef DECIMAL(10,4) NOT NULL DEFAULT 0,
    total_ef DECIMAL(10,4) NOT NULL,
    source VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dto_wef_elec_positive CHECK (electricity_ef >= 0),
    CONSTRAINT chk_dto_wef_gas_positive CHECK (gas_ef >= 0),
    CONSTRAINT chk_dto_wef_total_positive CHECK (total_ef >= 0),
    CONSTRAINT chk_dto_wef_total_sum CHECK (total_ef >= electricity_ef),
    CONSTRAINT chk_dto_wef_type CHECK (warehouse_type IN (
        'distribution_center', 'cross_dock', 'cold_storage_chilled',
        'cold_storage_frozen', 'retail_store', 'fulfillment_center', 'dark_store'
    ))
);

CREATE INDEX idx_dto_wef_active ON downstream_transportation_service.gl_dto_warehouse_emission_factors(is_active);

COMMENT ON TABLE downstream_transportation_service.gl_dto_warehouse_emission_factors IS 'Warehouse emission factors in kgCO2e per m2 per year by warehouse type (CIBSE TM46 / Industry avg)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_warehouse_emission_factors.warehouse_type IS 'Warehouse type: distribution_center, cross_dock, cold_storage_chilled, cold_storage_frozen, retail_store, fulfillment_center, dark_store';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_warehouse_emission_factors.electricity_ef IS 'Electricity-related emissions in kgCO2e per m2 per year';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_warehouse_emission_factors.gas_ef IS 'Gas/heating-related emissions in kgCO2e per m2 per year';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_warehouse_emission_factors.total_ef IS 'Total emissions in kgCO2e per m2 per year (electricity + gas)';

-- =====================================================================================
-- TABLE 4: gl_dto_last_mile_factors (REFERENCE)
-- Description: Last-mile delivery emission factors per delivery by area type
-- Source: DEFRA 2024, Industry averages
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_last_mile_factors (
    delivery_type VARCHAR(50) PRIMARY KEY,
    urban_ef DECIMAL(10,6) NOT NULL,
    suburban_ef DECIMAL(10,6) NOT NULL,
    rural_ef DECIMAL(10,6),
    source VARCHAR(100) DEFAULT 'DEFRA_2024',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dto_lmf_urban_positive CHECK (urban_ef >= 0),
    CONSTRAINT chk_dto_lmf_suburban_positive CHECK (suburban_ef >= 0),
    CONSTRAINT chk_dto_lmf_rural_positive CHECK (rural_ef IS NULL OR rural_ef >= 0),
    CONSTRAINT chk_dto_lmf_type CHECK (delivery_type IN (
        'parcel_standard', 'parcel_express', 'same_day',
        'click_and_collect', 'locker', 'cargo_bike'
    ))
);

CREATE INDEX idx_dto_lmf_active ON downstream_transportation_service.gl_dto_last_mile_factors(is_active);

COMMENT ON TABLE downstream_transportation_service.gl_dto_last_mile_factors IS 'Last-mile delivery emission factors in kgCO2e per delivery by delivery type and area (urban/suburban/rural)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_last_mile_factors.delivery_type IS 'Last-mile delivery type: parcel_standard, parcel_express, same_day, click_and_collect, locker, cargo_bike';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_last_mile_factors.urban_ef IS 'Emission factor per delivery in urban areas (kgCO2e/delivery)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_last_mile_factors.suburban_ef IS 'Emission factor per delivery in suburban areas (kgCO2e/delivery)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_last_mile_factors.rural_ef IS 'Emission factor per delivery in rural areas (kgCO2e/delivery), NULL if not applicable';

-- =====================================================================================
-- TABLE 5: gl_dto_eeio_factors (REFERENCE)
-- Description: EEIO spend-based emission factors by NAICS code for transport/warehousing
-- Source: EPA USEEIO v2.0
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_eeio_factors (
    naics_code VARCHAR(20) PRIMARY KEY,
    sector VARCHAR(200) NOT NULL,
    ef_per_usd DECIMAL(10,6) NOT NULL,
    base_year INT DEFAULT 2021,
    unit VARCHAR(50) DEFAULT 'kgCO2e/USD',
    source VARCHAR(100) DEFAULT 'EPA_USEEIO_v2',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dto_eeio_ef_positive CHECK (ef_per_usd >= 0),
    CONSTRAINT chk_dto_eeio_year_valid CHECK (base_year >= 1990 AND base_year <= 2100)
);

CREATE INDEX idx_dto_eeio_sector ON downstream_transportation_service.gl_dto_eeio_factors(sector);
CREATE INDEX idx_dto_eeio_source ON downstream_transportation_service.gl_dto_eeio_factors(source);
CREATE INDEX idx_dto_eeio_active ON downstream_transportation_service.gl_dto_eeio_factors(is_active);

COMMENT ON TABLE downstream_transportation_service.gl_dto_eeio_factors IS 'EPA USEEIO v2.0 spend-based emission factors for downstream transport and warehousing services by NAICS code';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_eeio_factors.naics_code IS 'NAICS industry code (e.g., 484110 General Freight Trucking Local)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_eeio_factors.ef_per_usd IS 'Emission factor per USD spent (kgCO2e/USD, base year deflated)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_eeio_factors.base_year IS 'Base year for CPI deflation (default 2021)';

-- =====================================================================================
-- TABLE 6: gl_dto_shipments (OPERATIONAL)
-- Description: Downstream shipment records with route, mode, and logistics metadata
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_shipments (
    shipment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    mode VARCHAR(30) NOT NULL,
    vehicle_type VARCHAR(50),
    origin VARCHAR(200) NOT NULL,
    destination VARCHAR(200) NOT NULL,
    distance_km DECIMAL(15,4),
    weight_tonnes DECIMAL(15,6),
    incoterm VARCHAR(10),
    temperature_regime VARCHAR(30) DEFAULT 'ambient',
    load_factor VARCHAR(20) DEFAULT 'typical',
    return_type VARCHAR(30) DEFAULT 'no_return',
    distribution_channel VARCHAR(50),
    product_id VARCHAR(200),
    customer_id VARCHAR(200),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dto_ship_mode CHECK (mode IN (
        'road', 'rail', 'maritime', 'air', 'inland_waterway',
        'pipeline', 'intermodal', 'courier', 'last_mile'
    )),
    CONSTRAINT chk_dto_ship_distance_positive CHECK (distance_km IS NULL OR distance_km >= 0),
    CONSTRAINT chk_dto_ship_weight_positive CHECK (weight_tonnes IS NULL OR weight_tonnes >= 0),
    CONSTRAINT chk_dto_ship_incoterm CHECK (incoterm IS NULL OR incoterm IN (
        'EXW', 'FCA', 'FAS', 'FOB', 'CFR', 'CIF', 'CIP', 'CPT', 'DAP', 'DPU', 'DDP'
    )),
    CONSTRAINT chk_dto_ship_temp_regime CHECK (temperature_regime IN (
        'ambient', 'chilled', 'frozen', 'deep_frozen', 'controlled'
    )),
    CONSTRAINT chk_dto_ship_load_factor CHECK (load_factor IS NULL OR load_factor IN (
        'empty', 'partial', 'half', 'typical', 'full'
    )),
    CONSTRAINT chk_dto_ship_return_type CHECK (return_type IN (
        'no_return', 'customer_return', 'product_recall', 'reusable_packaging'
    )),
    CONSTRAINT chk_dto_ship_channel CHECK (distribution_channel IS NULL OR distribution_channel IN (
        'direct_to_consumer', 'wholesale', 'retail', 'e_commerce', 'distributor', 'franchise'
    ))
);

CREATE INDEX idx_dto_ship_tenant ON downstream_transportation_service.gl_dto_shipments(tenant_id);
CREATE INDEX idx_dto_ship_mode ON downstream_transportation_service.gl_dto_shipments(mode);
CREATE INDEX idx_dto_ship_vehicle ON downstream_transportation_service.gl_dto_shipments(vehicle_type);
CREATE INDEX idx_dto_ship_incoterm ON downstream_transportation_service.gl_dto_shipments(incoterm);
CREATE INDEX idx_dto_ship_temp ON downstream_transportation_service.gl_dto_shipments(temperature_regime);
CREATE INDEX idx_dto_ship_channel ON downstream_transportation_service.gl_dto_shipments(distribution_channel);
CREATE INDEX idx_dto_ship_product ON downstream_transportation_service.gl_dto_shipments(product_id);
CREATE INDEX idx_dto_ship_customer ON downstream_transportation_service.gl_dto_shipments(customer_id);
CREATE INDEX idx_dto_ship_origin_dest ON downstream_transportation_service.gl_dto_shipments(origin, destination);
CREATE INDEX idx_dto_ship_return ON downstream_transportation_service.gl_dto_shipments(return_type);
CREATE INDEX idx_dto_ship_created ON downstream_transportation_service.gl_dto_shipments(created_at DESC);
CREATE INDEX idx_dto_ship_metadata ON downstream_transportation_service.gl_dto_shipments USING GIN(metadata);

COMMENT ON TABLE downstream_transportation_service.gl_dto_shipments IS 'Downstream shipment records with transport mode, route, Incoterm, temperature regime, and logistics metadata';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_shipments.mode IS 'Transport mode: road, rail, maritime, air, inland_waterway, pipeline, intermodal, courier, last_mile';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_shipments.incoterm IS 'ICC Incoterm 2020 defining Cat 4 vs Cat 9 boundary (EXW, FCA, FAS, FOB, CFR, CIF, CIP, CPT, DAP, DPU, DDP)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_shipments.temperature_regime IS 'Temperature regime: ambient, chilled (2-8C), frozen (-18 to -25C), deep_frozen (<-25C), controlled (15-25C)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_shipments.load_factor IS 'Vehicle load utilization: empty (0%), partial (37%), half (50%), typical (67%), full (92%)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_shipments.return_type IS 'Return logistics type: no_return, customer_return, product_recall, reusable_packaging';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_shipments.distribution_channel IS 'Distribution channel: direct_to_consumer, wholesale, retail, e_commerce, distributor, franchise';

-- =====================================================================================
-- TABLE 7: gl_dto_warehouses (OPERATIONAL)
-- Description: Warehouse and distribution center records for storage emissions
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_warehouses (
    warehouse_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    warehouse_type VARCHAR(50) NOT NULL,
    warehouse_name VARCHAR(300),
    floor_area_m2 DECIMAL(15,4),
    country VARCHAR(10) NOT NULL,
    temperature_regime VARCHAR(30) DEFAULT 'ambient',
    energy_source VARCHAR(30) DEFAULT 'electricity',
    allocation_share DECIMAL(8,6) DEFAULT 1.000000,
    storage_days INT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dto_wh_type CHECK (warehouse_type IN (
        'distribution_center', 'cross_dock', 'cold_storage_chilled',
        'cold_storage_frozen', 'retail_store', 'fulfillment_center', 'dark_store'
    )),
    CONSTRAINT chk_dto_wh_area_positive CHECK (floor_area_m2 IS NULL OR floor_area_m2 > 0),
    CONSTRAINT chk_dto_wh_temp CHECK (temperature_regime IN (
        'ambient', 'chilled', 'frozen', 'deep_frozen', 'controlled'
    )),
    CONSTRAINT chk_dto_wh_energy CHECK (energy_source IN (
        'electricity', 'natural_gas', 'diesel', 'lpg', 'district_heating', 'district_cooling'
    )),
    CONSTRAINT chk_dto_wh_alloc_range CHECK (allocation_share >= 0 AND allocation_share <= 1.0),
    CONSTRAINT chk_dto_wh_storage_days CHECK (storage_days IS NULL OR storage_days >= 0)
);

CREATE INDEX idx_dto_wh_tenant ON downstream_transportation_service.gl_dto_warehouses(tenant_id);
CREATE INDEX idx_dto_wh_type ON downstream_transportation_service.gl_dto_warehouses(warehouse_type);
CREATE INDEX idx_dto_wh_country ON downstream_transportation_service.gl_dto_warehouses(country);
CREATE INDEX idx_dto_wh_temp ON downstream_transportation_service.gl_dto_warehouses(temperature_regime);
CREATE INDEX idx_dto_wh_energy ON downstream_transportation_service.gl_dto_warehouses(energy_source);
CREATE INDEX idx_dto_wh_created ON downstream_transportation_service.gl_dto_warehouses(created_at DESC);

COMMENT ON TABLE downstream_transportation_service.gl_dto_warehouses IS 'Warehouse and distribution center records with floor area, temperature regime, and energy source for storage emission calculations';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_warehouses.warehouse_type IS 'Warehouse type: distribution_center, cross_dock, cold_storage_chilled, cold_storage_frozen, retail_store, fulfillment_center, dark_store';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_warehouses.floor_area_m2 IS 'Total floor area in square meters';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_warehouses.allocation_share IS 'Proportion of warehouse emissions allocated to reporting company products (0.0 to 1.0)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_warehouses.storage_days IS 'Average number of days products are stored at this warehouse';

-- =====================================================================================
-- TABLE 8: gl_dto_currency_rates (REFERENCE)
-- Description: Currency conversion rates to USD for spend-based calculations
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_currency_rates (
    currency VARCHAR(3) PRIMARY KEY,
    rate_to_usd DECIMAL(12,6) NOT NULL,
    year INT DEFAULT 2024,
    source VARCHAR(100) DEFAULT 'ECB_2024',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dto_curr_rate_positive CHECK (rate_to_usd > 0),
    CONSTRAINT chk_dto_curr_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_dto_curr_year ON downstream_transportation_service.gl_dto_currency_rates(year);
CREATE INDEX idx_dto_curr_active ON downstream_transportation_service.gl_dto_currency_rates(is_active);

COMMENT ON TABLE downstream_transportation_service.gl_dto_currency_rates IS 'Currency conversion rates to USD for spend-based emission calculations';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_currency_rates.rate_to_usd IS 'Exchange rate: 1 unit of currency = X USD';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_currency_rates.year IS 'Reference year for the exchange rate';

-- =====================================================================================
-- TABLE 9: gl_dto_cpi_deflators (REFERENCE)
-- Description: CPI deflators for adjusting spend data to EEIO base year
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_cpi_deflators (
    year INT PRIMARY KEY,
    cpi_index DECIMAL(10,4) NOT NULL,
    deflator DECIMAL(10,6) NOT NULL,
    base_year INT DEFAULT 2024,
    source VARCHAR(100) DEFAULT 'BLS_CPI_U',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dto_cpi_index_positive CHECK (cpi_index > 0),
    CONSTRAINT chk_dto_cpi_deflator_positive CHECK (deflator > 0),
    CONSTRAINT chk_dto_cpi_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_dto_cpi_active ON downstream_transportation_service.gl_dto_cpi_deflators(is_active);

COMMENT ON TABLE downstream_transportation_service.gl_dto_cpi_deflators IS 'US CPI deflators for adjusting spend data from reporting year to EEIO base year (2024 = 1.0)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_cpi_deflators.cpi_index IS 'US CPI-U index value for the year';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_cpi_deflators.deflator IS 'Deflator relative to base year (base year = 1.0)';

-- =====================================================================================
-- TABLE 10: gl_dto_grid_emission_factors (REFERENCE)
-- Description: Electricity grid emission factors by country for warehouse calculations
-- Source: EPA eGRID, IEA 2024
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_grid_emission_factors (
    country VARCHAR(10) PRIMARY KEY,
    ef_kwh DECIMAL(12,6) NOT NULL,
    unit VARCHAR(50) DEFAULT 'kgCO2e/kWh',
    source VARCHAR(100) NOT NULL,
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dto_grid_ef_positive CHECK (ef_kwh >= 0),
    CONSTRAINT chk_dto_grid_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_dto_grid_source ON downstream_transportation_service.gl_dto_grid_emission_factors(source);
CREATE INDEX idx_dto_grid_active ON downstream_transportation_service.gl_dto_grid_emission_factors(is_active);

COMMENT ON TABLE downstream_transportation_service.gl_dto_grid_emission_factors IS 'Electricity grid emission factors by country for warehouse and distribution center energy calculations';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_grid_emission_factors.ef_kwh IS 'Grid emission factor in kgCO2e per kWh consumed';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_grid_emission_factors.source IS 'Source: EPA eGRID 2024, IEA 2024, DEFRA 2024';

-- =====================================================================================
-- TABLE 11: gl_dto_distribution_channels (REFERENCE)
-- Description: Default distribution channel parameters for average-data method
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_distribution_channels (
    channel VARCHAR(50) PRIMARY KEY,
    avg_distance_km INT NOT NULL,
    avg_mode VARCHAR(30) NOT NULL,
    avg_legs INT NOT NULL DEFAULT 1,
    storage_days INT NOT NULL DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dto_dc_distance_positive CHECK (avg_distance_km > 0),
    CONSTRAINT chk_dto_dc_mode CHECK (avg_mode IN (
        'road', 'rail', 'maritime', 'air', 'inland_waterway',
        'pipeline', 'intermodal', 'courier', 'last_mile'
    )),
    CONSTRAINT chk_dto_dc_legs_positive CHECK (avg_legs >= 1),
    CONSTRAINT chk_dto_dc_storage_positive CHECK (storage_days >= 0),
    CONSTRAINT chk_dto_dc_channel CHECK (channel IN (
        'direct_to_consumer', 'wholesale', 'retail', 'e_commerce', 'distributor', 'franchise'
    ))
);

CREATE INDEX idx_dto_dc_mode ON downstream_transportation_service.gl_dto_distribution_channels(avg_mode);
CREATE INDEX idx_dto_dc_active ON downstream_transportation_service.gl_dto_distribution_channels(is_active);

COMMENT ON TABLE downstream_transportation_service.gl_dto_distribution_channels IS 'Default distribution channel parameters for average-data emission calculations (distance, mode, legs, storage)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_distribution_channels.channel IS 'Distribution channel: direct_to_consumer, wholesale, retail, e_commerce, distributor, franchise';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_distribution_channels.avg_distance_km IS 'Average total distance in km for this channel';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_distribution_channels.avg_mode IS 'Primary transport mode for this channel';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_distribution_channels.avg_legs IS 'Average number of transport legs in this channel';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_distribution_channels.storage_days IS 'Average days in intermediate storage for this channel';

-- =====================================================================================
-- TABLE 12: gl_dto_incoterm_classification (REFERENCE)
-- Description: Incoterm-based classification of Cat 4 vs Cat 9 scope boundaries
-- Source: ICC Incoterms 2020
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_incoterm_classification (
    incoterm VARCHAR(10) PRIMARY KEY,
    cat4_scope VARCHAR(200) NOT NULL,
    cat9_scope VARCHAR(200) NOT NULL,
    transfer_point VARCHAR(200) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dto_inc_incoterm CHECK (incoterm IN (
        'EXW', 'FCA', 'FAS', 'FOB', 'CFR', 'CIF', 'CIP', 'CPT', 'DAP', 'DPU', 'DDP'
    ))
);

CREATE INDEX idx_dto_inc_active ON downstream_transportation_service.gl_dto_incoterm_classification(is_active);

COMMENT ON TABLE downstream_transportation_service.gl_dto_incoterm_classification IS 'ICC Incoterms 2020 classification defining Cat 4 (seller-paid) vs Cat 9 (buyer-paid) transport boundaries';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_incoterm_classification.incoterm IS 'ICC Incoterm code: EXW, FCA, FAS, FOB, CFR, CIF, CIP, CPT, DAP, DPU, DDP';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_incoterm_classification.cat4_scope IS 'Transport scope assigned to Category 4 (company-paid)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_incoterm_classification.cat9_scope IS 'Transport scope assigned to Category 9 (buyer/3rd-party-paid)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_incoterm_classification.transfer_point IS 'Point at which transport responsibility transfers from seller to buyer';

-- =====================================================================================
-- TABLE 13: gl_dto_load_factors (REFERENCE)
-- Description: Load factor utilization adjustments for emission calculations
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_load_factors (
    load_factor VARCHAR(20) PRIMARY KEY,
    utilization_pct DECIMAL(8,4) NOT NULL,
    adjustment DECIMAL(8,4) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dto_lf_util_range CHECK (utilization_pct >= 0 AND utilization_pct <= 100.0),
    CONSTRAINT chk_dto_lf_adj_positive CHECK (adjustment > 0),
    CONSTRAINT chk_dto_lf_factor CHECK (load_factor IN (
        'empty', 'partial', 'half', 'typical', 'full'
    ))
);

CREATE INDEX idx_dto_lf_active ON downstream_transportation_service.gl_dto_load_factors(is_active);

COMMENT ON TABLE downstream_transportation_service.gl_dto_load_factors IS 'Load factor utilization adjustments for emission calculations (empty vehicle deadheading through full load)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_load_factors.load_factor IS 'Load factor category: empty, partial, half, typical, full';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_load_factors.utilization_pct IS 'Average cargo utilization percentage for this load factor';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_load_factors.adjustment IS 'Emission factor adjustment multiplier (typical = 1.00)';

-- =====================================================================================
-- TABLE 14: gl_dto_calculations (HYPERTABLE)
-- Description: Master calculation records for downstream transportation emissions
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_calculations (
    calculation_id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    reporting_year INT NOT NULL,
    method VARCHAR(50) NOT NULL,
    total_emissions_kg DECIMAL(20,8) NOT NULL,
    transport_emissions_kg DECIMAL(20,8) DEFAULT 0,
    warehouse_emissions_kg DECIMAL(20,8) DEFAULT 0,
    last_mile_emissions_kg DECIMAL(20,8) DEFAULT 0,
    return_emissions_kg DECIMAL(20,8) DEFAULT 0,
    wtt_emissions_kg DECIMAL(20,8) DEFAULT 0,
    shipment_count INT DEFAULT 0,
    dqi_score DECIMAL(5,2),
    provenance_hash VARCHAR(64),
    status VARCHAR(20) DEFAULT 'completed',
    ef_source VARCHAR(100),
    gwp_version VARCHAR(20) DEFAULT 'AR5',
    metadata JSONB DEFAULT '{}',
    is_deleted BOOLEAN DEFAULT FALSE,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (calculation_id, calculated_at),
    CONSTRAINT chk_dto_calc_year_valid CHECK (reporting_year >= 2000 AND reporting_year <= 2100),
    CONSTRAINT chk_dto_calc_method CHECK (method IN (
        'distance_based', 'spend_based', 'average_data', 'supplier_specific'
    )),
    CONSTRAINT chk_dto_calc_total_positive CHECK (total_emissions_kg >= 0),
    CONSTRAINT chk_dto_calc_transport_positive CHECK (transport_emissions_kg IS NULL OR transport_emissions_kg >= 0),
    CONSTRAINT chk_dto_calc_warehouse_positive CHECK (warehouse_emissions_kg IS NULL OR warehouse_emissions_kg >= 0),
    CONSTRAINT chk_dto_calc_lastmile_positive CHECK (last_mile_emissions_kg IS NULL OR last_mile_emissions_kg >= 0),
    CONSTRAINT chk_dto_calc_return_positive CHECK (return_emissions_kg IS NULL OR return_emissions_kg >= 0),
    CONSTRAINT chk_dto_calc_wtt_positive CHECK (wtt_emissions_kg IS NULL OR wtt_emissions_kg >= 0),
    CONSTRAINT chk_dto_calc_shipment_positive CHECK (shipment_count IS NULL OR shipment_count >= 0),
    CONSTRAINT chk_dto_calc_dqi_range CHECK (dqi_score IS NULL OR (dqi_score >= 1.0 AND dqi_score <= 5.0)),
    CONSTRAINT chk_dto_calc_status CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled'))
);

-- Convert to hypertable (7-day chunks)
SELECT create_hypertable('downstream_transportation_service.gl_dto_calculations', 'calculated_at',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX idx_dto_calc_tenant ON downstream_transportation_service.gl_dto_calculations(tenant_id, calculated_at DESC);
CREATE INDEX idx_dto_calc_year ON downstream_transportation_service.gl_dto_calculations(reporting_year);
CREATE INDEX idx_dto_calc_method ON downstream_transportation_service.gl_dto_calculations(method);
CREATE INDEX idx_dto_calc_status ON downstream_transportation_service.gl_dto_calculations(status);
CREATE INDEX idx_dto_calc_hash ON downstream_transportation_service.gl_dto_calculations(provenance_hash);
CREATE INDEX idx_dto_calc_ef_source ON downstream_transportation_service.gl_dto_calculations(ef_source);
CREATE INDEX idx_dto_calc_deleted ON downstream_transportation_service.gl_dto_calculations(is_deleted) WHERE is_deleted = FALSE;
CREATE INDEX idx_dto_calc_metadata ON downstream_transportation_service.gl_dto_calculations USING GIN(metadata);

COMMENT ON TABLE downstream_transportation_service.gl_dto_calculations IS 'Master calculation records for downstream transportation emissions with transport, warehouse, last-mile, and return breakdown (HYPERTABLE, 7-day chunks)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_calculations.method IS 'Calculation method: distance_based, spend_based, average_data, supplier_specific';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_calculations.total_emissions_kg IS 'Total Scope 3 Cat 9 emissions in kgCO2e (transport + warehouse + last_mile + return + WTT)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_calculations.transport_emissions_kg IS 'Transport leg emissions in kgCO2e (TTW only)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_calculations.warehouse_emissions_kg IS 'Warehouse/distribution center emissions in kgCO2e';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_calculations.last_mile_emissions_kg IS 'Last-mile delivery emissions in kgCO2e';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_calculations.return_emissions_kg IS 'Return logistics emissions in kgCO2e';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_calculations.wtt_emissions_kg IS 'Well-to-tank upstream fuel production emissions in kgCO2e';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_calculations.dqi_score IS 'Data Quality Indicator score (1.0=highest to 5.0=lowest)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_calculations.provenance_hash IS 'SHA-256 hash of all calculation inputs for audit trail';

-- =====================================================================================
-- TABLE 15: gl_dto_calculation_details (HYPERTABLE)
-- Description: Per-component emission calculation details with mode breakdown
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_calculation_details (
    detail_id UUID NOT NULL DEFAULT gen_random_uuid(),
    calculation_id UUID NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    component_type VARCHAR(30) NOT NULL,
    mode VARCHAR(30),
    vehicle_type VARCHAR(50),
    distance_km DECIMAL(15,4),
    weight_tonnes DECIMAL(15,6),
    emissions_kg DECIMAL(20,8) NOT NULL,
    wtt_kg DECIMAL(20,8) DEFAULT 0,
    cold_chain_uplift DECIMAL(6,4) DEFAULT 1.0000,
    load_factor_adjustment DECIMAL(8,4) DEFAULT 1.0000,
    ef_used DECIMAL(12,8),
    ef_source VARCHAR(100),
    provenance_hash VARCHAR(64),
    metadata JSONB DEFAULT '{}',
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (detail_id, calculated_at),
    CONSTRAINT chk_dto_det_component CHECK (component_type IN (
        'transport', 'warehouse', 'last_mile', 'return', 'wtt'
    )),
    CONSTRAINT chk_dto_det_mode CHECK (mode IS NULL OR mode IN (
        'road', 'rail', 'maritime', 'air', 'inland_waterway',
        'pipeline', 'intermodal', 'courier', 'last_mile'
    )),
    CONSTRAINT chk_dto_det_emissions_positive CHECK (emissions_kg >= 0),
    CONSTRAINT chk_dto_det_wtt_positive CHECK (wtt_kg IS NULL OR wtt_kg >= 0),
    CONSTRAINT chk_dto_det_distance_positive CHECK (distance_km IS NULL OR distance_km >= 0),
    CONSTRAINT chk_dto_det_weight_positive CHECK (weight_tonnes IS NULL OR weight_tonnes >= 0),
    CONSTRAINT chk_dto_det_cold_chain CHECK (cold_chain_uplift IS NULL OR cold_chain_uplift >= 1.0),
    CONSTRAINT chk_dto_det_load_positive CHECK (load_factor_adjustment IS NULL OR load_factor_adjustment > 0),
    CONSTRAINT chk_dto_det_ef_positive CHECK (ef_used IS NULL OR ef_used >= 0)
);

-- Convert to hypertable (7-day chunks)
SELECT create_hypertable('downstream_transportation_service.gl_dto_calculation_details', 'calculated_at',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX idx_dto_det_calc ON downstream_transportation_service.gl_dto_calculation_details(calculation_id);
CREATE INDEX idx_dto_det_tenant ON downstream_transportation_service.gl_dto_calculation_details(tenant_id, calculated_at DESC);
CREATE INDEX idx_dto_det_component ON downstream_transportation_service.gl_dto_calculation_details(component_type);
CREATE INDEX idx_dto_det_mode ON downstream_transportation_service.gl_dto_calculation_details(mode);
CREATE INDEX idx_dto_det_vehicle ON downstream_transportation_service.gl_dto_calculation_details(vehicle_type);
CREATE INDEX idx_dto_det_ef_source ON downstream_transportation_service.gl_dto_calculation_details(ef_source);
CREATE INDEX idx_dto_det_hash ON downstream_transportation_service.gl_dto_calculation_details(provenance_hash);
CREATE INDEX idx_dto_det_metadata ON downstream_transportation_service.gl_dto_calculation_details USING GIN(metadata);

COMMENT ON TABLE downstream_transportation_service.gl_dto_calculation_details IS 'Per-component emission calculation details with mode, vehicle type, and factor breakdown (HYPERTABLE, 7-day chunks)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_calculation_details.component_type IS 'Emission component: transport, warehouse, last_mile, return, wtt';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_calculation_details.cold_chain_uplift IS 'Cold chain uplift factor applied (1.0 = ambient, 1.35 = frozen)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_calculation_details.load_factor_adjustment IS 'Load factor adjustment multiplier (1.0 = typical 67% utilization)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_calculation_details.ef_used IS 'Emission factor value applied in the calculation';

-- =====================================================================================
-- TABLE 16: gl_dto_compliance_checks (OPERATIONAL)
-- Description: Compliance check results against 7 regulatory frameworks
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_compliance_checks (
    check_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id UUID NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    framework VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    score DECIMAL(5,2),
    total_rules INT DEFAULT 0,
    passed INT DEFAULT 0,
    failed INT DEFAULT 0,
    warnings INT DEFAULT 0,
    findings JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    provenance_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dto_compl_framework CHECK (framework IN (
        'GHG_PROTOCOL', 'ISO_14064', 'ISO_14083', 'CSRD_ESRS', 'CDP', 'SBTI', 'SB_253'
    )),
    CONSTRAINT chk_dto_compl_status CHECK (status IN ('PASS', 'FAIL', 'WARNING', 'NOT_APPLICABLE')),
    CONSTRAINT chk_dto_compl_score_range CHECK (score IS NULL OR (score >= 0 AND score <= 100)),
    CONSTRAINT chk_dto_compl_rules_positive CHECK (total_rules IS NULL OR total_rules >= 0),
    CONSTRAINT chk_dto_compl_passed_positive CHECK (passed IS NULL OR passed >= 0),
    CONSTRAINT chk_dto_compl_failed_positive CHECK (failed IS NULL OR failed >= 0),
    CONSTRAINT chk_dto_compl_warnings_positive CHECK (warnings IS NULL OR warnings >= 0)
);

CREATE INDEX idx_dto_compl_calc ON downstream_transportation_service.gl_dto_compliance_checks(calculation_id);
CREATE INDEX idx_dto_compl_tenant ON downstream_transportation_service.gl_dto_compliance_checks(tenant_id);
CREATE INDEX idx_dto_compl_framework ON downstream_transportation_service.gl_dto_compliance_checks(framework);
CREATE INDEX idx_dto_compl_status ON downstream_transportation_service.gl_dto_compliance_checks(status);
CREATE INDEX idx_dto_compl_findings ON downstream_transportation_service.gl_dto_compliance_checks USING GIN(findings);
CREATE INDEX idx_dto_compl_created ON downstream_transportation_service.gl_dto_compliance_checks(created_at DESC);

COMMENT ON TABLE downstream_transportation_service.gl_dto_compliance_checks IS 'Compliance check results against GHG Protocol, ISO 14064, ISO 14083, CSRD ESRS, CDP, SBTi, SB 253 frameworks';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_compliance_checks.framework IS 'Regulatory framework: GHG_PROTOCOL, ISO_14064, ISO_14083, CSRD_ESRS, CDP, SBTI, SB_253';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_compliance_checks.status IS 'Compliance status: PASS, FAIL, WARNING, NOT_APPLICABLE';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_compliance_checks.findings IS 'JSONB array of compliance findings with severity, rule_id, and detail';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_compliance_checks.recommendations IS 'JSONB array of improvement recommendations';

-- =====================================================================================
-- TABLE 17: gl_dto_aggregations (HYPERTABLE)
-- Description: Period aggregations by mode, channel, and destination
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_aggregations (
    aggregation_id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    by_mode JSONB DEFAULT '{}',
    by_channel JSONB DEFAULT '{}',
    by_destination JSONB DEFAULT '{}',
    by_temperature JSONB DEFAULT '{}',
    by_product JSONB DEFAULT '{}',
    total_emissions_kg DECIMAL(20,8) NOT NULL,
    total_wtt_kg DECIMAL(20,8) DEFAULT 0,
    shipment_count INT DEFAULT 0,
    dqi_avg DECIMAL(5,2),
    metadata JSONB DEFAULT '{}',
    PRIMARY KEY (aggregation_id, period_start),
    CONSTRAINT chk_dto_agg_total_positive CHECK (total_emissions_kg >= 0),
    CONSTRAINT chk_dto_agg_wtt_positive CHECK (total_wtt_kg IS NULL OR total_wtt_kg >= 0),
    CONSTRAINT chk_dto_agg_count_positive CHECK (shipment_count IS NULL OR shipment_count >= 0),
    CONSTRAINT chk_dto_agg_dqi_range CHECK (dqi_avg IS NULL OR (dqi_avg >= 1.0 AND dqi_avg <= 5.0)),
    CONSTRAINT chk_dto_agg_period_valid CHECK (period_end >= period_start)
);

-- Convert to hypertable (30-day chunks)
SELECT create_hypertable('downstream_transportation_service.gl_dto_aggregations', 'period_start',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE INDEX idx_dto_agg_tenant ON downstream_transportation_service.gl_dto_aggregations(tenant_id, period_start DESC);
CREATE INDEX idx_dto_agg_period ON downstream_transportation_service.gl_dto_aggregations(period_start, period_end);
CREATE INDEX idx_dto_agg_by_mode ON downstream_transportation_service.gl_dto_aggregations USING GIN(by_mode);
CREATE INDEX idx_dto_agg_by_channel ON downstream_transportation_service.gl_dto_aggregations USING GIN(by_channel);
CREATE INDEX idx_dto_agg_by_dest ON downstream_transportation_service.gl_dto_aggregations USING GIN(by_destination);
CREATE INDEX idx_dto_agg_by_temp ON downstream_transportation_service.gl_dto_aggregations USING GIN(by_temperature);
CREATE INDEX idx_dto_agg_by_product ON downstream_transportation_service.gl_dto_aggregations USING GIN(by_product);

COMMENT ON TABLE downstream_transportation_service.gl_dto_aggregations IS 'Period aggregations of downstream transport emissions by mode, channel, destination, temperature, and product (HYPERTABLE, 30-day chunks)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_aggregations.by_mode IS 'JSONB breakdown of emissions by transport mode (road, rail, maritime, air, etc.)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_aggregations.by_channel IS 'JSONB breakdown of emissions by distribution channel (direct, wholesale, retail, e-commerce)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_aggregations.by_destination IS 'JSONB breakdown of emissions by destination region/country';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_aggregations.by_temperature IS 'JSONB breakdown of emissions by temperature regime (ambient, chilled, frozen)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_aggregations.by_product IS 'JSONB breakdown of emissions by product/product category';

-- =====================================================================================
-- TABLE 18: gl_dto_product_allocations (OPERATIONAL)
-- Description: Product-level emission allocation records
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_product_allocations (
    allocation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id UUID NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    product_id VARCHAR(200) NOT NULL,
    product_name VARCHAR(500),
    allocation_method VARCHAR(30) NOT NULL,
    allocation_share DECIMAL(10,8) NOT NULL,
    allocated_emissions_kg DECIMAL(20,8) NOT NULL,
    allocated_wtt_kg DECIMAL(20,8) DEFAULT 0,
    mass_kg DECIMAL(15,6),
    volume_m3 DECIMAL(15,6),
    revenue_usd DECIMAL(15,2),
    units_sold INT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dto_alloc_method CHECK (allocation_method IN (
        'mass', 'volume', 'revenue', 'units_sold', 'teu', 'pallet_positions', 'floor_area'
    )),
    CONSTRAINT chk_dto_alloc_share_range CHECK (allocation_share >= 0 AND allocation_share <= 1.0),
    CONSTRAINT chk_dto_alloc_emissions_positive CHECK (allocated_emissions_kg >= 0),
    CONSTRAINT chk_dto_alloc_wtt_positive CHECK (allocated_wtt_kg IS NULL OR allocated_wtt_kg >= 0),
    CONSTRAINT chk_dto_alloc_mass_positive CHECK (mass_kg IS NULL OR mass_kg >= 0),
    CONSTRAINT chk_dto_alloc_volume_positive CHECK (volume_m3 IS NULL OR volume_m3 >= 0),
    CONSTRAINT chk_dto_alloc_revenue_positive CHECK (revenue_usd IS NULL OR revenue_usd >= 0),
    CONSTRAINT chk_dto_alloc_units_positive CHECK (units_sold IS NULL OR units_sold >= 0)
);

CREATE INDEX idx_dto_alloc_calc ON downstream_transportation_service.gl_dto_product_allocations(calculation_id);
CREATE INDEX idx_dto_alloc_tenant ON downstream_transportation_service.gl_dto_product_allocations(tenant_id);
CREATE INDEX idx_dto_alloc_product ON downstream_transportation_service.gl_dto_product_allocations(product_id);
CREATE INDEX idx_dto_alloc_method ON downstream_transportation_service.gl_dto_product_allocations(allocation_method);
CREATE INDEX idx_dto_alloc_created ON downstream_transportation_service.gl_dto_product_allocations(created_at DESC);
CREATE INDEX idx_dto_alloc_metadata ON downstream_transportation_service.gl_dto_product_allocations USING GIN(metadata);

COMMENT ON TABLE downstream_transportation_service.gl_dto_product_allocations IS 'Product-level emission allocation records for product carbon footprints and eco-design compliance';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_product_allocations.allocation_method IS 'Allocation method: mass, volume, revenue, units_sold, teu, pallet_positions, floor_area';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_product_allocations.allocation_share IS 'Proportion of total emissions allocated to this product (0.0 to 1.0)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_product_allocations.allocated_emissions_kg IS 'Emissions allocated to this product in kgCO2e';

-- =====================================================================================
-- TABLE 19: gl_dto_return_logistics (OPERATIONAL)
-- Description: Return logistics emission records for product returns and packaging
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_return_logistics (
    return_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    shipment_id UUID NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    return_type VARCHAR(30) NOT NULL,
    return_multiplier DECIMAL(6,4) NOT NULL,
    outbound_emissions_kg DECIMAL(20,8) NOT NULL,
    return_emissions_kg DECIMAL(20,8) NOT NULL,
    return_distance_km DECIMAL(15,4),
    mode VARCHAR(30),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dto_ret_type CHECK (return_type IN (
        'no_return', 'customer_return', 'product_recall', 'reusable_packaging'
    )),
    CONSTRAINT chk_dto_ret_multiplier_range CHECK (return_multiplier >= 0 AND return_multiplier <= 1.5),
    CONSTRAINT chk_dto_ret_outbound_positive CHECK (outbound_emissions_kg >= 0),
    CONSTRAINT chk_dto_ret_return_positive CHECK (return_emissions_kg >= 0),
    CONSTRAINT chk_dto_ret_distance_positive CHECK (return_distance_km IS NULL OR return_distance_km >= 0),
    CONSTRAINT chk_dto_ret_mode CHECK (mode IS NULL OR mode IN (
        'road', 'rail', 'maritime', 'air', 'inland_waterway',
        'pipeline', 'intermodal', 'courier', 'last_mile'
    ))
);

CREATE INDEX idx_dto_ret_shipment ON downstream_transportation_service.gl_dto_return_logistics(shipment_id);
CREATE INDEX idx_dto_ret_tenant ON downstream_transportation_service.gl_dto_return_logistics(tenant_id);
CREATE INDEX idx_dto_ret_type ON downstream_transportation_service.gl_dto_return_logistics(return_type);
CREATE INDEX idx_dto_ret_mode ON downstream_transportation_service.gl_dto_return_logistics(mode);
CREATE INDEX idx_dto_ret_created ON downstream_transportation_service.gl_dto_return_logistics(created_at DESC);

COMMENT ON TABLE downstream_transportation_service.gl_dto_return_logistics IS 'Return logistics emission records for customer returns, product recalls, and reusable packaging';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_return_logistics.return_type IS 'Return type: no_return, customer_return (85% outbound), product_recall (100% outbound), reusable_packaging (50% outbound)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_return_logistics.return_multiplier IS 'Multiplier applied to outbound emissions (0.0 to 1.5)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_return_logistics.outbound_emissions_kg IS 'Original outbound transport emissions in kgCO2e';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_return_logistics.return_emissions_kg IS 'Calculated return logistics emissions in kgCO2e';

-- =====================================================================================
-- TABLE 20: gl_dto_provenance (OPERATIONAL)
-- Description: Provenance tracking with SHA-256 hash chains for audit trail
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_provenance (
    provenance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id UUID NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    stage VARCHAR(30) NOT NULL,
    stage_index INT NOT NULL DEFAULT 0,
    input_hash VARCHAR(64) NOT NULL,
    output_hash VARCHAR(64) NOT NULL,
    chain_hash VARCHAR(64) NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dto_prov_stage CHECK (stage IN (
        'VALIDATE', 'CLASSIFY', 'NORMALIZE', 'RESOLVE_EFS',
        'CALCULATE', 'ALLOCATE', 'AGGREGATE', 'COMPLIANCE',
        'PROVENANCE', 'SEAL'
    )),
    CONSTRAINT chk_dto_prov_index_positive CHECK (stage_index >= 0)
);

CREATE INDEX idx_dto_prov_calc ON downstream_transportation_service.gl_dto_provenance(calculation_id);
CREATE INDEX idx_dto_prov_tenant ON downstream_transportation_service.gl_dto_provenance(tenant_id);
CREATE INDEX idx_dto_prov_calc_stage ON downstream_transportation_service.gl_dto_provenance(calculation_id, stage_index);
CREATE INDEX idx_dto_prov_stage ON downstream_transportation_service.gl_dto_provenance(stage);
CREATE INDEX idx_dto_prov_input ON downstream_transportation_service.gl_dto_provenance(input_hash);
CREATE INDEX idx_dto_prov_output ON downstream_transportation_service.gl_dto_provenance(output_hash);
CREATE INDEX idx_dto_prov_chain ON downstream_transportation_service.gl_dto_provenance(chain_hash);
CREATE INDEX idx_dto_prov_created ON downstream_transportation_service.gl_dto_provenance(created_at DESC);
CREATE INDEX idx_dto_prov_metadata ON downstream_transportation_service.gl_dto_provenance USING GIN(metadata);

COMMENT ON TABLE downstream_transportation_service.gl_dto_provenance IS 'Provenance tracking for downstream transportation calculations with SHA-256 hash chains across 10 pipeline stages';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_provenance.stage IS 'Processing stage: VALIDATE, CLASSIFY, NORMALIZE, RESOLVE_EFS, CALCULATE, ALLOCATE, AGGREGATE, COMPLIANCE, PROVENANCE, SEAL';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_provenance.stage_index IS 'Sequential order of processing stage within the calculation pipeline (0-9)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_provenance.input_hash IS 'SHA-256 hash of stage input data';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_provenance.output_hash IS 'SHA-256 hash of stage output data';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_provenance.chain_hash IS 'SHA-256 hash chaining input_hash + output_hash + previous chain_hash';

-- =====================================================================================
-- TABLE 21: gl_dto_audit_entries (OPERATIONAL)
-- Description: Audit log entries for all downstream transportation operations
-- =====================================================================================

CREATE TABLE downstream_transportation_service.gl_dto_audit_entries (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    action VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID,
    actor VARCHAR(200),
    details JSONB DEFAULT '{}',
    ip_address VARCHAR(45),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dto_audit_action CHECK (action IN (
        'CREATE', 'UPDATE', 'DELETE', 'CALCULATE', 'RECALCULATE',
        'ALLOCATE', 'AGGREGATE', 'COMPLIANCE_CHECK', 'EXPORT',
        'APPROVE', 'REJECT', 'ARCHIVE'
    )),
    CONSTRAINT chk_dto_audit_entity_type CHECK (entity_type IN (
        'shipment', 'warehouse', 'calculation', 'calculation_detail',
        'compliance_check', 'aggregation', 'product_allocation',
        'return_logistics', 'provenance'
    ))
);

CREATE INDEX idx_dto_audit_tenant ON downstream_transportation_service.gl_dto_audit_entries(tenant_id);
CREATE INDEX idx_dto_audit_action ON downstream_transportation_service.gl_dto_audit_entries(action);
CREATE INDEX idx_dto_audit_entity ON downstream_transportation_service.gl_dto_audit_entries(entity_type, entity_id);
CREATE INDEX idx_dto_audit_actor ON downstream_transportation_service.gl_dto_audit_entries(actor);
CREATE INDEX idx_dto_audit_created ON downstream_transportation_service.gl_dto_audit_entries(created_at DESC);
CREATE INDEX idx_dto_audit_details ON downstream_transportation_service.gl_dto_audit_entries USING GIN(details);

COMMENT ON TABLE downstream_transportation_service.gl_dto_audit_entries IS 'Audit log entries for all downstream transportation operations (create, calculate, allocate, compliance, export)';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_audit_entries.action IS 'Action type: CREATE, UPDATE, DELETE, CALCULATE, RECALCULATE, ALLOCATE, AGGREGATE, COMPLIANCE_CHECK, EXPORT, APPROVE, REJECT, ARCHIVE';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_audit_entries.entity_type IS 'Entity type: shipment, warehouse, calculation, calculation_detail, compliance_check, aggregation, product_allocation, return_logistics, provenance';
COMMENT ON COLUMN downstream_transportation_service.gl_dto_audit_entries.details IS 'JSONB object with action-specific details (before/after values, parameters, etc.)';

-- =====================================================================================
-- CONTINUOUS AGGREGATES
-- =====================================================================================

-- Continuous Aggregate 1: Hourly Emissions
CREATE MATERIALIZED VIEW downstream_transportation_service.gl_dto_hourly_emissions
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', calculated_at) AS bucket,
    tenant_id,
    method,
    COUNT(*) AS calc_count,
    SUM(total_emissions_kg) AS total_emissions_kg,
    SUM(transport_emissions_kg) AS transport_emissions_kg,
    SUM(warehouse_emissions_kg) AS warehouse_emissions_kg,
    SUM(last_mile_emissions_kg) AS last_mile_emissions_kg,
    SUM(return_emissions_kg) AS return_emissions_kg,
    SUM(wtt_emissions_kg) AS wtt_emissions_kg,
    SUM(shipment_count) AS total_shipments,
    AVG(dqi_score) AS avg_dqi_score
FROM downstream_transportation_service.gl_dto_calculations
GROUP BY bucket, tenant_id, method
WITH NO DATA;

-- Refresh policy for hourly emissions (refresh every 1 hour, lag 2 hours)
SELECT add_continuous_aggregate_policy('downstream_transportation_service.gl_dto_hourly_emissions',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '2 hours',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW downstream_transportation_service.gl_dto_hourly_emissions IS 'Hourly aggregation of downstream transportation emissions by method with transport/warehouse/last-mile/return breakdown';

-- Continuous Aggregate 2: Daily Emissions with Mode Breakdown
CREATE MATERIALIZED VIEW downstream_transportation_service.gl_dto_daily_emissions
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', calculated_at) AS bucket,
    tenant_id,
    method,
    COUNT(*) AS calc_count,
    SUM(total_emissions_kg) AS total_emissions_kg,
    SUM(transport_emissions_kg) AS transport_emissions_kg,
    SUM(warehouse_emissions_kg) AS warehouse_emissions_kg,
    SUM(last_mile_emissions_kg) AS last_mile_emissions_kg,
    SUM(return_emissions_kg) AS return_emissions_kg,
    SUM(wtt_emissions_kg) AS wtt_emissions_kg,
    SUM(shipment_count) AS total_shipments,
    AVG(dqi_score) AS avg_dqi_score
FROM downstream_transportation_service.gl_dto_calculations
GROUP BY bucket, tenant_id, method
WITH NO DATA;

-- Refresh policy for daily emissions (refresh every 6 hours, lag 12 hours)
SELECT add_continuous_aggregate_policy('downstream_transportation_service.gl_dto_daily_emissions',
    start_offset => INTERVAL '30 days',
    end_offset => INTERVAL '12 hours',
    schedule_interval => INTERVAL '6 hours',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW downstream_transportation_service.gl_dto_daily_emissions IS 'Daily aggregation of downstream transportation emissions with transport/warehouse/last-mile/return breakdown and DQI scores';

-- =====================================================================================
-- ROW LEVEL SECURITY (RLS) - ALL OPERATIONAL TABLES
-- =====================================================================================

-- Enable RLS on all operational tables with tenant_id
ALTER TABLE downstream_transportation_service.gl_dto_shipments ENABLE ROW LEVEL SECURITY;
ALTER TABLE downstream_transportation_service.gl_dto_warehouses ENABLE ROW LEVEL SECURITY;
ALTER TABLE downstream_transportation_service.gl_dto_calculations ENABLE ROW LEVEL SECURITY;
ALTER TABLE downstream_transportation_service.gl_dto_calculation_details ENABLE ROW LEVEL SECURITY;
ALTER TABLE downstream_transportation_service.gl_dto_compliance_checks ENABLE ROW LEVEL SECURITY;
ALTER TABLE downstream_transportation_service.gl_dto_aggregations ENABLE ROW LEVEL SECURITY;
ALTER TABLE downstream_transportation_service.gl_dto_product_allocations ENABLE ROW LEVEL SECURITY;
ALTER TABLE downstream_transportation_service.gl_dto_return_logistics ENABLE ROW LEVEL SECURITY;
ALTER TABLE downstream_transportation_service.gl_dto_provenance ENABLE ROW LEVEL SECURITY;
ALTER TABLE downstream_transportation_service.gl_dto_audit_entries ENABLE ROW LEVEL SECURITY;

-- RLS Policies: tenant_id isolation on all operational tables
CREATE POLICY dto_shipments_tenant_isolation ON downstream_transportation_service.gl_dto_shipments
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

CREATE POLICY dto_warehouses_tenant_isolation ON downstream_transportation_service.gl_dto_warehouses
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

CREATE POLICY dto_calculations_tenant_isolation ON downstream_transportation_service.gl_dto_calculations
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

CREATE POLICY dto_calculation_details_tenant_isolation ON downstream_transportation_service.gl_dto_calculation_details
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

CREATE POLICY dto_compliance_checks_tenant_isolation ON downstream_transportation_service.gl_dto_compliance_checks
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

CREATE POLICY dto_aggregations_tenant_isolation ON downstream_transportation_service.gl_dto_aggregations
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

CREATE POLICY dto_product_allocations_tenant_isolation ON downstream_transportation_service.gl_dto_product_allocations
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

CREATE POLICY dto_return_logistics_tenant_isolation ON downstream_transportation_service.gl_dto_return_logistics
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

CREATE POLICY dto_provenance_tenant_isolation ON downstream_transportation_service.gl_dto_provenance
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

CREATE POLICY dto_audit_entries_tenant_isolation ON downstream_transportation_service.gl_dto_audit_entries
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- =====================================================================================
-- SEED DATA: TRANSPORT EMISSION FACTORS (26 vehicle types)
-- Source: DEFRA 2024, IMO 2020, ICAO 2024, GLEC v3, Industry averages
-- =====================================================================================

INSERT INTO downstream_transportation_service.gl_dto_transport_emission_factors
(vehicle_type, mode, ef_per_tkm, wtt_per_tkm, source) VALUES
-- Road (9 vehicle types)
('lgv_petrol',            'road',     0.58400000, 0.13900000, 'DEFRA 2024'),
('lgv_diesel',            'road',     0.48000000, 0.11200000, 'DEFRA 2024'),
('lgv_electric',          'road',     0.11800000, 0.02400000, 'DEFRA 2024'),
('rigid_truck_small',     'road',     0.44100000, 0.10300000, 'DEFRA 2024'),
('rigid_truck_medium',    'road',     0.21300000, 0.05000000, 'DEFRA 2024'),
('rigid_truck_large',     'road',     0.15000000, 0.03500000, 'DEFRA 2024'),
('articulated_truck',     'road',     0.10700000, 0.02500000, 'DEFRA 2024'),
('delivery_van',          'road',     0.58000000, 0.13500000, 'DEFRA 2024'),
('cargo_bike',            'road',     0.00000000, 0.00000000, 'Zero emission'),

-- Rail (2 vehicle types)
('freight_train',         'rail',     0.02800000, 0.00600000, 'DEFRA 2024'),
('intermodal_rail',       'rail',     0.02500000, 0.00500000, 'DEFRA 2024'),

-- Maritime (5 vessel types)
('container_ship_small',  'maritime', 0.02200000, 0.00500000, 'IMO 2020'),
('container_ship_medium', 'maritime', 0.01600000, 0.00400000, 'IMO 2020'),
('container_ship_large',  'maritime', 0.00800000, 0.00200000, 'IMO 2020'),
('bulk_carrier',          'maritime', 0.00500000, 0.00100000, 'IMO 2020'),
('ro_ro_ferry',           'maritime', 0.06000000, 0.01400000, 'DEFRA 2024'),

-- Air (3 aircraft types)
('freighter_narrow',      'air',      0.60200000, 0.14300000, 'DEFRA 2024'),
('freighter_wide',        'air',      0.49500000, 0.11800000, 'DEFRA 2024'),
('belly_freight',         'air',      0.44000000, 0.10500000, 'ICAO 2024'),

-- Inland Waterway (1 type)
('barge',                 'inland_waterway', 0.03200000, 0.00700000, 'GLEC v3'),

-- Courier (2 types)
('parcel_standard',       'courier',  0.42000000, 0.09800000, 'DEFRA 2024'),
('parcel_express',        'courier',  0.52000000, 0.12100000, 'DEFRA 2024'),

-- Last Mile (4 types)
('same_day',              'last_mile', 0.68000000, 0.15900000, 'Industry avg'),
('click_and_collect',     'last_mile', 0.05000000, 0.01200000, 'Industry avg'),
('locker',                'last_mile', 0.04000000, 0.00900000, 'Industry avg'),
('cargo_bike_lastmile',   'last_mile', 0.00500000, 0.00100000, 'Industry avg');

-- =====================================================================================
-- SEED DATA: COLD CHAIN UPLIFT FACTORS (5 temperature regimes)
-- =====================================================================================

INSERT INTO downstream_transportation_service.gl_dto_cold_chain_factors
(temperature_regime, road_uplift, rail_uplift, maritime_uplift, air_uplift) VALUES
('ambient',     1.0000, 1.0000, 1.0000, 1.0000),
('chilled',     1.2000, 1.1500, 1.1800, 1.1000),
('frozen',      1.3500, 1.2500, 1.3000, 1.1500),
('deep_frozen', 1.5000, 1.3500, 1.4000, 1.2000),
('controlled',  1.0500, 1.0300, 1.0400, 1.0200);

-- =====================================================================================
-- SEED DATA: WAREHOUSE EMISSION FACTORS (7 warehouse types)
-- Source: CIBSE TM46, Industry averages
-- =====================================================================================

INSERT INTO downstream_transportation_service.gl_dto_warehouse_emission_factors
(warehouse_type, electricity_ef, gas_ef, total_ef, source) VALUES
('distribution_center',  45.0000, 12.0000,  57.0000, 'CIBSE TM46'),
('cross_dock',           30.0000,  8.0000,  38.0000, 'CIBSE TM46'),
('cold_storage_chilled', 120.0000, 5.0000, 125.0000, 'Industry avg'),
('cold_storage_frozen',  180.0000, 3.0000, 183.0000, 'Industry avg'),
('retail_store',          85.0000, 25.0000, 110.0000, 'CIBSE TM46'),
('fulfillment_center',   55.0000, 10.0000,  65.0000, 'Industry avg'),
('dark_store',            95.0000, 15.0000, 110.0000, 'Industry avg');

-- =====================================================================================
-- SEED DATA: LAST MILE EMISSION FACTORS (6 delivery types)
-- Source: DEFRA 2024, Industry averages
-- =====================================================================================

INSERT INTO downstream_transportation_service.gl_dto_last_mile_factors
(delivery_type, urban_ef, suburban_ef, rural_ef, source) VALUES
('parcel_standard',  0.520000, 0.780000, 1.200000, 'DEFRA 2024'),
('parcel_express',   0.680000, 0.950000, 1.500000, 'DEFRA 2024'),
('same_day',         0.850000, 1.200000, 1.800000, 'Industry avg'),
('click_and_collect', 0.050000, 0.050000, 0.050000, 'Store energy'),
('locker',           0.040000, 0.040000, 0.040000, 'Locker energy'),
('cargo_bike',       0.010000, 0.020000, NULL,      'Industry avg');

-- =====================================================================================
-- SEED DATA: EEIO FACTORS (10 NAICS codes for downstream transport/warehousing)
-- Source: EPA USEEIO v2.0
-- =====================================================================================

INSERT INTO downstream_transportation_service.gl_dto_eeio_factors
(naics_code, sector, ef_per_usd, base_year, unit, source) VALUES
('484110', 'General Freight Trucking, Local',           0.470000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2'),
('484121', 'General Freight Trucking, Long-Distance',   0.380000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2'),
('482110', 'Rail Transportation',                       0.280000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2'),
('483111', 'Deep Sea Freight',                          0.210000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2'),
('481112', 'Air Freight',                               1.250000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2'),
('492110', 'Couriers and Express Delivery',             0.520000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2'),
('493110', 'General Warehousing and Storage',           0.340000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2'),
('493120', 'Refrigerated Warehousing and Storage',      0.580000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2'),
('454110', 'Electronic Shopping and Mail-Order',        0.420000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2'),
('493130', 'Farm Product Warehousing',                  0.310000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2');

-- =====================================================================================
-- SEED DATA: CURRENCY CONVERSION RATES (12 currencies)
-- =====================================================================================

INSERT INTO downstream_transportation_service.gl_dto_currency_rates
(currency, rate_to_usd, year, source) VALUES
('USD', 1.000000, 2024, 'ECB_2024'),
('EUR', 1.085000, 2024, 'ECB_2024'),
('GBP', 1.265000, 2024, 'ECB_2024'),
('JPY', 0.006700, 2024, 'ECB_2024'),
('CAD', 0.740000, 2024, 'ECB_2024'),
('AUD', 0.655000, 2024, 'ECB_2024'),
('CHF', 1.130000, 2024, 'ECB_2024'),
('CNY', 0.140000, 2024, 'ECB_2024'),
('INR', 0.012000, 2024, 'ECB_2024'),
('BRL', 0.200000, 2024, 'ECB_2024'),
('KRW', 0.000800, 2024, 'ECB_2024'),
('SEK', 0.096000, 2024, 'ECB_2024');

-- =====================================================================================
-- SEED DATA: CPI DEFLATORS (11 years, base year = 2024)
-- Source: US Bureau of Labor Statistics CPI-U
-- =====================================================================================

INSERT INTO downstream_transportation_service.gl_dto_cpi_deflators
(year, cpi_index, deflator, base_year, source) VALUES
(2015, 237.0000, 0.798300, 2024, 'BLS_CPI_U'),
(2016, 240.0000, 0.808400, 2024, 'BLS_CPI_U'),
(2017, 245.1000, 0.825600, 2024, 'BLS_CPI_U'),
(2018, 251.1000, 0.845800, 2024, 'BLS_CPI_U'),
(2019, 255.7000, 0.861300, 2024, 'BLS_CPI_U'),
(2020, 258.8000, 0.871700, 2024, 'BLS_CPI_U'),
(2021, 270.9000, 0.912500, 2024, 'BLS_CPI_U'),
(2022, 292.7000, 0.985900, 2024, 'BLS_CPI_U'),
(2023, 304.7000, 1.026400, 2024, 'BLS_CPI_U'),
(2024, 296.9000, 1.000000, 2024, 'BLS_CPI_U'),
(2025, 303.5000, 1.022200, 2024, 'BLS_CPI_U');

-- =====================================================================================
-- SEED DATA: GRID EMISSION FACTORS (11 countries/regions)
-- Source: EPA eGRID 2024, IEA 2024, DEFRA 2024
-- =====================================================================================

INSERT INTO downstream_transportation_service.gl_dto_grid_emission_factors
(country, ef_kwh, unit, source, year) VALUES
('US',     0.393700, 'kgCO2e/kWh', 'EPA eGRID 2024', 2024),
('GB',     0.212100, 'kgCO2e/kWh', 'DEFRA 2024',     2024),
('DE',     0.364000, 'kgCO2e/kWh', 'IEA 2024',       2024),
('FR',     0.056900, 'kgCO2e/kWh', 'IEA 2024',       2024),
('JP',     0.457000, 'kgCO2e/kWh', 'IEA 2024',       2024),
('CA',     0.120000, 'kgCO2e/kWh', 'IEA 2024',       2024),
('AU',     0.610000, 'kgCO2e/kWh', 'IEA 2024',       2024),
('IN',     0.708000, 'kgCO2e/kWh', 'IEA 2024',       2024),
('CN',     0.557000, 'kgCO2e/kWh', 'IEA 2024',       2024),
('BR',     0.074000, 'kgCO2e/kWh', 'IEA 2024',       2024),
('GLOBAL', 0.436000, 'kgCO2e/kWh', 'IEA 2024',       2024);

-- =====================================================================================
-- SEED DATA: DISTRIBUTION CHANNEL DEFAULTS (6 channels)
-- =====================================================================================

INSERT INTO downstream_transportation_service.gl_dto_distribution_channels
(channel, avg_distance_km, avg_mode, avg_legs, storage_days) VALUES
('direct_to_consumer', 500,  'road',       2, 0),
('wholesale',          800,  'road',       1, 14),
('retail',             600,  'road',       2, 30),
('e_commerce',         350,  'courier',    3, 7),
('distributor',        1200, 'intermodal', 2, 21),
('franchise',          400,  'road',       1, 7);

-- =====================================================================================
-- SEED DATA: INCOTERM CLASSIFICATION (11 Incoterms)
-- Source: ICC Incoterms 2020
-- =====================================================================================

INSERT INTO downstream_transportation_service.gl_dto_incoterm_classification
(incoterm, cat4_scope, cat9_scope, transfer_point) VALUES
('EXW', 'No',              'All transport',     'Seller''s premises'),
('FCA', 'To carrier',      'After carrier',     'Named place'),
('FAS', 'To port',         'After port',        'Ship''s side'),
('FOB', 'To on board',     'After on board',    'Ship''s rail'),
('CFR', 'Main carriage',   'After discharge',   'Destination port'),
('CIF', 'Main + insurance','After discharge',   'Destination port'),
('CPT', 'To destination',  'After delivery',    'Named place'),
('CIP', 'To dest + ins',   'After delivery',    'Named place'),
('DAP', 'To destination',  'Unloading only',    'Named place'),
('DPU', 'To unloaded',     'None',              'Named place'),
('DDP', 'All transport',   'None',              'Named place');

-- =====================================================================================
-- SEED DATA: LOAD FACTOR ADJUSTMENTS (5 load factors)
-- =====================================================================================

INSERT INTO downstream_transportation_service.gl_dto_load_factors
(load_factor, utilization_pct, adjustment) VALUES
('empty',   0.0000, 0.4000),
('partial', 37.0000, 0.6500),
('half',    50.0000, 0.8000),
('typical', 67.0000, 1.0000),
('full',    92.0000, 1.1500);

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
    'GL-MRV-S3-009',
    'Downstream Transportation & Distribution Agent',
    '1.0.0',
    'CALCULATION',
    'SCOPE3_EMISSIONS',
    'AGENT-MRV-022: Scope 3 Category 9 - Downstream Transportation & Distribution. Calculates emissions from outbound transportation of sold products (9a), distribution center/warehouse operations (9b), retail storage (9c), and last-mile delivery (9d). Supports distance-based (DEFRA/IMO/ICAO/GLEC), spend-based (EPA USEEIO v2), average-data, and supplier-specific methods. Includes 26 transport emission factors, 5 cold chain regimes, 7 warehouse types, 6 last-mile types, 10 EEIO factors, 11 Incoterm classifications, product-level allocation (7 methods), and return logistics (4 types). 7-framework compliance (GHG Protocol, ISO 14064, ISO 14083, CSRD ESRS, CDP, SBTi, SB 253).',
    'ACTIVE',
    jsonb_build_object(
        'scope3_category', 9,
        'category_name', 'Downstream Transportation and Distribution',
        'sub_activities', jsonb_build_array('9a: Outbound transportation', '9b: Distribution/warehousing', '9c: Retail storage', '9d: Last-mile delivery'),
        'calculation_methods', jsonb_build_array('distance_based', 'spend_based', 'average_data', 'supplier_specific'),
        'transport_modes', jsonb_build_array('road', 'rail', 'maritime', 'air', 'inland_waterway', 'pipeline', 'intermodal', 'courier', 'last_mile'),
        'temperature_regimes', jsonb_build_array('ambient', 'chilled', 'frozen', 'deep_frozen', 'controlled'),
        'distribution_channels', jsonb_build_array('direct_to_consumer', 'wholesale', 'retail', 'e_commerce', 'distributor', 'franchise'),
        'allocation_methods', jsonb_build_array('mass', 'volume', 'revenue', 'units_sold', 'teu', 'pallet_positions', 'floor_area'),
        'return_types', jsonb_build_array('no_return', 'customer_return', 'product_recall', 'reusable_packaging'),
        'frameworks', jsonb_build_array('GHG Protocol Scope 3', 'ISO 14064', 'ISO 14083', 'CSRD ESRS E1', 'CDP Climate', 'SBTi', 'SB 253'),
        'transport_ef_count', 26,
        'cold_chain_regime_count', 5,
        'warehouse_type_count', 7,
        'last_mile_type_count', 6,
        'eeio_factor_count', 10,
        'currency_count', 12,
        'cpi_deflator_count', 11,
        'grid_factor_count', 11,
        'distribution_channel_count', 6,
        'incoterm_count', 11,
        'load_factor_count', 5,
        'supports_cold_chain_uplift', true,
        'supports_wtt_emissions', true,
        'supports_load_factor_adjustment', true,
        'supports_return_logistics', true,
        'supports_product_allocation', true,
        'supports_cpi_deflation', true,
        'supports_incoterm_classification', true,
        'default_ef_source', 'DEFRA_2024',
        'default_gwp', 'AR5',
        'schema', 'downstream_transportation_service',
        'table_prefix', 'gl_dto_',
        'hypertables', jsonb_build_array('gl_dto_calculations', 'gl_dto_calculation_details', 'gl_dto_aggregations'),
        'continuous_aggregates', jsonb_build_array('gl_dto_hourly_emissions', 'gl_dto_daily_emissions'),
        'migration_version', 'V073'
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

COMMENT ON SCHEMA downstream_transportation_service IS 'Updated: AGENT-MRV-022 complete with 21 tables, 3 hypertables, 2 continuous aggregates, 10 RLS policies, 100+ seed records';

-- =====================================================================================
-- END OF MIGRATION V073
-- =====================================================================================
-- Total Lines: ~1200
-- Total Tables: 21 (16 operational + 5 reference)
-- Total Hypertables: 3 (calculations, calculation_details, aggregations)
-- Total Continuous Aggregates: 2 (hourly_emissions, daily_emissions)
-- Total RLS Policies: 10 (shipments, warehouses, calculations, calculation_details,
--                         compliance_checks, aggregations, product_allocations,
--                         return_logistics, provenance, audit_entries)
-- Total Seed Records: 107
--   Transport Emission Factors: 26 (road/rail/maritime/air/waterway/courier/last-mile)
--   Cold Chain Uplift Factors: 5 (ambient/chilled/frozen/deep_frozen/controlled)
--   Warehouse Emission Factors: 7 (DC/cross-dock/cold chilled/cold frozen/retail/fulfillment/dark)
--   Last Mile Factors: 6 (standard/express/same-day/click-collect/locker/cargo-bike)
--   EEIO Factors: 10 (NAICS transport/warehousing sectors)
--   Currency Rates: 12 (USD/EUR/GBP/JPY/CAD/AUD/CHF/CNY/INR/BRL/KRW/SEK)
--   CPI Deflators: 11 (2015-2025, base year 2024)
--   Grid Emission Factors: 11 (US/GB/DE/FR/JP/CA/AU/IN/CN/BR/GLOBAL)
--   Distribution Channels: 6 (direct/wholesale/retail/e-commerce/distributor/franchise)
--   Incoterm Classifications: 11 (EXW/FCA/FAS/FOB/CFR/CIF/CIP/CPT/DAP/DPU/DDP)
--   Load Factors: 5 (empty/partial/half/typical/full)
--   Agent Registry: 1
-- =====================================================================================
