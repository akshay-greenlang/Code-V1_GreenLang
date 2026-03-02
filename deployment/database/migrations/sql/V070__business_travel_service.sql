-- =====================================================================================
-- Migration: V070__business_travel_service.sql
-- Description: AGENT-MRV-019 Business Travel (Scope 3 Category 6)
-- Agent: GL-MRV-SCOPE3-006
-- Framework: GHG Protocol Scope 3 Standard, DEFRA 2024, ICAO, EPA EEIO, ISO 14064-1
-- Created: 2026-02-26
-- =====================================================================================
-- Schema: business_travel_service
-- Tables: 16 (7 reference + 9 operational)
-- Hypertables: 3 (calculations, flight_results, aggregations)
-- Continuous Aggregates: 2 (hourly_emissions, daily_emissions)
-- RLS: Enabled on calculations, flight_results, ground_results, hotel_results, spend_results
-- Seed Data: 150+ records (airports, air EFs, ground EFs, hotel EFs, EEIO factors)
-- =====================================================================================

-- =====================================================================================
-- SCHEMA CREATION
-- =====================================================================================

CREATE SCHEMA IF NOT EXISTS business_travel_service;

COMMENT ON SCHEMA business_travel_service IS 'AGENT-MRV-019: Business Travel - Scope 3 Category 6 emission calculations (air/rail/road/hotel/spend-based)';

-- =====================================================================================
-- TABLE 1: gl_bt_trips
-- Description: Trip master records with origin/destination and travel metadata
-- =====================================================================================

CREATE TABLE business_travel_service.gl_bt_trips (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    trip_id VARCHAR(200),
    traveler_id VARCHAR(200),
    mode VARCHAR(50) NOT NULL,
    trip_purpose VARCHAR(50) DEFAULT 'business',
    department VARCHAR(200),
    cost_center VARCHAR(200),
    origin VARCHAR(10),
    destination VARCHAR(10),
    distance_km DECIMAL(20,8),
    travel_date DATE,
    return_date DATE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_bt_mode CHECK (mode IN ('air', 'rail', 'road', 'bus', 'taxi', 'ferry', 'motorcycle', 'hotel')),
    CONSTRAINT chk_bt_distance_positive CHECK (distance_km IS NULL OR distance_km >= 0),
    CONSTRAINT chk_bt_dates_valid CHECK (return_date IS NULL OR return_date >= travel_date)
);

CREATE INDEX idx_bt_trips_tenant ON business_travel_service.gl_bt_trips(tenant_id);
CREATE INDEX idx_bt_trips_trip_id ON business_travel_service.gl_bt_trips(trip_id);
CREATE INDEX idx_bt_trips_traveler ON business_travel_service.gl_bt_trips(traveler_id);
CREATE INDEX idx_bt_trips_mode ON business_travel_service.gl_bt_trips(mode);
CREATE INDEX idx_bt_trips_department ON business_travel_service.gl_bt_trips(department);
CREATE INDEX idx_bt_trips_travel_date ON business_travel_service.gl_bt_trips(travel_date);
CREATE INDEX idx_bt_trips_origin_dest ON business_travel_service.gl_bt_trips(origin, destination);

COMMENT ON TABLE business_travel_service.gl_bt_trips IS 'Trip master records with travel mode, origin/destination, and organizational metadata';
COMMENT ON COLUMN business_travel_service.gl_bt_trips.mode IS 'Travel mode: air, rail, road, bus, taxi, ferry, motorcycle, hotel';
COMMENT ON COLUMN business_travel_service.gl_bt_trips.origin IS 'Origin IATA code or city code';
COMMENT ON COLUMN business_travel_service.gl_bt_trips.destination IS 'Destination IATA code or city code';
COMMENT ON COLUMN business_travel_service.gl_bt_trips.distance_km IS 'Great circle or route distance in km';

-- =====================================================================================
-- TABLE 2: gl_bt_travelers
-- Description: Traveler profile records linked to HR/employee data
-- =====================================================================================

CREATE TABLE business_travel_service.gl_bt_travelers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    employee_id VARCHAR(200),
    name VARCHAR(500),
    department VARCHAR(200),
    cost_center VARCHAR(200),
    default_cabin_class VARCHAR(50) DEFAULT 'economy',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_bt_cabin_class CHECK (default_cabin_class IN ('economy', 'premium_economy', 'business', 'first'))
);

CREATE INDEX idx_bt_travelers_tenant ON business_travel_service.gl_bt_travelers(tenant_id);
CREATE INDEX idx_bt_travelers_employee ON business_travel_service.gl_bt_travelers(employee_id);
CREATE INDEX idx_bt_travelers_department ON business_travel_service.gl_bt_travelers(department);

COMMENT ON TABLE business_travel_service.gl_bt_travelers IS 'Traveler profiles with default preferences and organizational assignments';
COMMENT ON COLUMN business_travel_service.gl_bt_travelers.default_cabin_class IS 'Default cabin class: economy, premium_economy, business, first';

-- =====================================================================================
-- TABLE 3: gl_bt_airports
-- Description: Airport and station database with IATA codes and coordinates
-- =====================================================================================

CREATE TABLE business_travel_service.gl_bt_airports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    iata_code VARCHAR(3) NOT NULL UNIQUE,
    name VARCHAR(500) NOT NULL,
    city VARCHAR(200),
    country_code VARCHAR(2) NOT NULL,
    latitude DECIMAL(10,6) NOT NULL,
    longitude DECIMAL(11,6) NOT NULL,
    timezone VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT chk_bt_latitude_range CHECK (latitude >= -90 AND latitude <= 90),
    CONSTRAINT chk_bt_longitude_range CHECK (longitude >= -180 AND longitude <= 180)
);

CREATE INDEX idx_bt_airports_iata ON business_travel_service.gl_bt_airports(iata_code);
CREATE INDEX idx_bt_airports_country ON business_travel_service.gl_bt_airports(country_code);
CREATE INDEX idx_bt_airports_active ON business_travel_service.gl_bt_airports(is_active);

COMMENT ON TABLE business_travel_service.gl_bt_airports IS 'Airport and station reference database with IATA codes and geographic coordinates';
COMMENT ON COLUMN business_travel_service.gl_bt_airports.iata_code IS 'IATA 3-letter airport code (e.g., JFK, LHR, NRT)';
COMMENT ON COLUMN business_travel_service.gl_bt_airports.latitude IS 'Airport latitude in decimal degrees (-90 to 90)';
COMMENT ON COLUMN business_travel_service.gl_bt_airports.longitude IS 'Airport longitude in decimal degrees (-180 to 180)';

-- =====================================================================================
-- TABLE 4: gl_bt_air_emission_factors
-- Description: Aviation emission factors by distance band and cabin class (DEFRA/ICAO)
-- =====================================================================================

CREATE TABLE business_travel_service.gl_bt_air_emission_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    distance_band VARCHAR(50) NOT NULL,
    cabin_class VARCHAR(50) DEFAULT 'economy',
    ef_without_rf DECIMAL(20,8) NOT NULL,
    ef_with_rf DECIMAL(20,8) NOT NULL,
    wtt_ef DECIMAL(20,8) NOT NULL,
    unit VARCHAR(50) DEFAULT 'kgCO2e/pkm',
    source VARCHAR(100) NOT NULL,
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(distance_band, cabin_class, source),
    CONSTRAINT chk_bt_air_ef_positive CHECK (ef_without_rf >= 0 AND ef_with_rf >= 0 AND wtt_ef >= 0),
    CONSTRAINT chk_bt_air_rf_greater CHECK (ef_with_rf >= ef_without_rf),
    CONSTRAINT chk_bt_air_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_bt_air_ef_distance ON business_travel_service.gl_bt_air_emission_factors(distance_band);
CREATE INDEX idx_bt_air_ef_cabin ON business_travel_service.gl_bt_air_emission_factors(cabin_class);
CREATE INDEX idx_bt_air_ef_source ON business_travel_service.gl_bt_air_emission_factors(source);
CREATE INDEX idx_bt_air_ef_active ON business_travel_service.gl_bt_air_emission_factors(is_active);

COMMENT ON TABLE business_travel_service.gl_bt_air_emission_factors IS 'Aviation emission factors by distance band and cabin class from DEFRA/ICAO';
COMMENT ON COLUMN business_travel_service.gl_bt_air_emission_factors.distance_band IS 'Distance band: short_haul (<3700km), medium_haul (3700-6000km), long_haul (>6000km), domestic';
COMMENT ON COLUMN business_travel_service.gl_bt_air_emission_factors.ef_without_rf IS 'Emission factor excluding radiative forcing (kgCO2e/pkm)';
COMMENT ON COLUMN business_travel_service.gl_bt_air_emission_factors.ef_with_rf IS 'Emission factor including radiative forcing multiplier (kgCO2e/pkm)';
COMMENT ON COLUMN business_travel_service.gl_bt_air_emission_factors.wtt_ef IS 'Well-to-tank emission factor for jet fuel (kgCO2e/pkm)';

-- =====================================================================================
-- TABLE 5: gl_bt_ground_emission_factors
-- Description: Ground transport emission factors for rail, road, bus, taxi, ferry
-- =====================================================================================

CREATE TABLE business_travel_service.gl_bt_ground_emission_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mode VARCHAR(50) NOT NULL,
    vehicle_type VARCHAR(100) NOT NULL,
    ef_per_vkm DECIMAL(20,8),
    ef_per_pkm DECIMAL(20,8),
    wtt_ef DECIMAL(20,8),
    occupancy DECIMAL(10,4),
    unit VARCHAR(50),
    source VARCHAR(100) NOT NULL,
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(mode, vehicle_type, source),
    CONSTRAINT chk_bt_ground_ef_positive CHECK (
        (ef_per_vkm IS NULL OR ef_per_vkm >= 0) AND
        (ef_per_pkm IS NULL OR ef_per_pkm >= 0) AND
        (wtt_ef IS NULL OR wtt_ef >= 0)
    ),
    CONSTRAINT chk_bt_ground_occupancy CHECK (occupancy IS NULL OR occupancy > 0),
    CONSTRAINT chk_bt_ground_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_bt_ground_ef_mode ON business_travel_service.gl_bt_ground_emission_factors(mode);
CREATE INDEX idx_bt_ground_ef_vehicle ON business_travel_service.gl_bt_ground_emission_factors(vehicle_type);
CREATE INDEX idx_bt_ground_ef_source ON business_travel_service.gl_bt_ground_emission_factors(source);
CREATE INDEX idx_bt_ground_ef_active ON business_travel_service.gl_bt_ground_emission_factors(is_active);

COMMENT ON TABLE business_travel_service.gl_bt_ground_emission_factors IS 'Ground transport emission factors for rail, road, bus, taxi, ferry, motorcycle';
COMMENT ON COLUMN business_travel_service.gl_bt_ground_emission_factors.ef_per_vkm IS 'Emission factor per vehicle-km (kgCO2e/vkm)';
COMMENT ON COLUMN business_travel_service.gl_bt_ground_emission_factors.ef_per_pkm IS 'Emission factor per passenger-km (kgCO2e/pkm)';
COMMENT ON COLUMN business_travel_service.gl_bt_ground_emission_factors.wtt_ef IS 'Well-to-tank emission factor (kgCO2e/pkm or kgCO2e/vkm)';
COMMENT ON COLUMN business_travel_service.gl_bt_ground_emission_factors.occupancy IS 'Average vehicle occupancy (passengers per vehicle)';

-- =====================================================================================
-- TABLE 6: gl_bt_hotel_emission_factors
-- Description: Hotel accommodation emission factors by country and class
-- =====================================================================================

CREATE TABLE business_travel_service.gl_bt_hotel_emission_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code VARCHAR(10) NOT NULL,
    hotel_class VARCHAR(50) DEFAULT 'standard',
    ef_per_room_night DECIMAL(20,8) NOT NULL,
    class_multiplier DECIMAL(10,4) DEFAULT 1.0,
    unit VARCHAR(50) DEFAULT 'kgCO2e/room-night',
    source VARCHAR(100) NOT NULL,
    year INT DEFAULT 2024,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(country_code, hotel_class, source),
    CONSTRAINT chk_bt_hotel_ef_positive CHECK (ef_per_room_night >= 0),
    CONSTRAINT chk_bt_hotel_multiplier_positive CHECK (class_multiplier > 0),
    CONSTRAINT chk_bt_hotel_year_valid CHECK (year >= 1990 AND year <= 2100)
);

CREATE INDEX idx_bt_hotel_ef_country ON business_travel_service.gl_bt_hotel_emission_factors(country_code);
CREATE INDEX idx_bt_hotel_ef_class ON business_travel_service.gl_bt_hotel_emission_factors(hotel_class);
CREATE INDEX idx_bt_hotel_ef_source ON business_travel_service.gl_bt_hotel_emission_factors(source);
CREATE INDEX idx_bt_hotel_ef_active ON business_travel_service.gl_bt_hotel_emission_factors(is_active);

COMMENT ON TABLE business_travel_service.gl_bt_hotel_emission_factors IS 'Hotel accommodation emission factors by country and hotel class';
COMMENT ON COLUMN business_travel_service.gl_bt_hotel_emission_factors.ef_per_room_night IS 'Emission factor per room-night (kgCO2e/room-night)';
COMMENT ON COLUMN business_travel_service.gl_bt_hotel_emission_factors.class_multiplier IS 'Multiplier for hotel class (budget=0.75, standard=1.0, luxury=1.5)';

-- =====================================================================================
-- TABLE 7: gl_bt_eeio_factors
-- Description: EEIO spend-based emission factors by NAICS code
-- =====================================================================================

CREATE TABLE business_travel_service.gl_bt_eeio_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    naics_code VARCHAR(20) NOT NULL,
    category_name VARCHAR(200) NOT NULL,
    ef_per_usd DECIMAL(20,8) NOT NULL,
    base_year INT DEFAULT 2021,
    unit VARCHAR(50) DEFAULT 'kgCO2e/USD',
    source VARCHAR(100) DEFAULT 'EPA_USEEIO_v2',
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(naics_code, source),
    CONSTRAINT chk_bt_eeio_ef_positive CHECK (ef_per_usd >= 0),
    CONSTRAINT chk_bt_eeio_year_valid CHECK (base_year >= 1990 AND base_year <= 2100)
);

CREATE INDEX idx_bt_eeio_naics ON business_travel_service.gl_bt_eeio_factors(naics_code);
CREATE INDEX idx_bt_eeio_source ON business_travel_service.gl_bt_eeio_factors(source);
CREATE INDEX idx_bt_eeio_active ON business_travel_service.gl_bt_eeio_factors(is_active);

COMMENT ON TABLE business_travel_service.gl_bt_eeio_factors IS 'EPA USEEIO v2 spend-based emission factors for travel categories by NAICS code';
COMMENT ON COLUMN business_travel_service.gl_bt_eeio_factors.ef_per_usd IS 'Emission factor per USD spent (kgCO2e/USD, base year deflated)';
COMMENT ON COLUMN business_travel_service.gl_bt_eeio_factors.base_year IS 'Base year for CPI deflation (default 2021)';

-- =====================================================================================
-- TABLE 8: gl_bt_calculations (HYPERTABLE)
-- Description: Main calculation results for business travel emissions
-- =====================================================================================

CREATE TABLE business_travel_service.gl_bt_calculations (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id VARCHAR(200) NOT NULL,
    mode VARCHAR(50) NOT NULL,
    method VARCHAR(50) NOT NULL,
    total_co2e_kg DECIMAL(20,8) NOT NULL,
    co2e_without_rf_kg DECIMAL(20,8),
    co2e_with_rf_kg DECIMAL(20,8),
    wtt_co2e_kg DECIMAL(20,8),
    dqi_score DECIMAL(5,2),
    reporting_period VARCHAR(20),
    department VARCHAR(200),
    cost_center VARCHAR(200),
    ef_source VARCHAR(100),
    gwp_version VARCHAR(20) DEFAULT 'AR5',
    metadata JSONB DEFAULT '{}',
    provenance_hash VARCHAR(64),
    is_deleted BOOLEAN DEFAULT FALSE,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, calculated_at),
    CONSTRAINT chk_bt_calc_co2e_positive CHECK (total_co2e_kg >= 0),
    CONSTRAINT chk_bt_calc_method CHECK (method IN ('distance_based', 'fuel_based', 'spend_based', 'supplier_specific', 'hotel_nights')),
    CONSTRAINT chk_bt_calc_dqi_range CHECK (dqi_score IS NULL OR (dqi_score >= 1.0 AND dqi_score <= 5.0))
);

-- Convert to hypertable (7-day chunks)
SELECT create_hypertable('business_travel_service.gl_bt_calculations', 'calculated_at',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX idx_bt_calculations_tenant ON business_travel_service.gl_bt_calculations(tenant_id, calculated_at DESC);
CREATE INDEX idx_bt_calculations_calc_id ON business_travel_service.gl_bt_calculations(calculation_id);
CREATE INDEX idx_bt_calculations_mode ON business_travel_service.gl_bt_calculations(mode);
CREATE INDEX idx_bt_calculations_method ON business_travel_service.gl_bt_calculations(method);
CREATE INDEX idx_bt_calculations_period ON business_travel_service.gl_bt_calculations(reporting_period);
CREATE INDEX idx_bt_calculations_department ON business_travel_service.gl_bt_calculations(department);
CREATE INDEX idx_bt_calculations_hash ON business_travel_service.gl_bt_calculations(provenance_hash);
CREATE INDEX idx_bt_calculations_deleted ON business_travel_service.gl_bt_calculations(is_deleted) WHERE is_deleted = FALSE;

COMMENT ON TABLE business_travel_service.gl_bt_calculations IS 'Main business travel emission calculation results (HYPERTABLE, 7-day chunks)';
COMMENT ON COLUMN business_travel_service.gl_bt_calculations.method IS 'Calculation method: distance_based, fuel_based, spend_based, supplier_specific, hotel_nights';
COMMENT ON COLUMN business_travel_service.gl_bt_calculations.co2e_without_rf_kg IS 'CO2e excluding radiative forcing (aviation only)';
COMMENT ON COLUMN business_travel_service.gl_bt_calculations.co2e_with_rf_kg IS 'CO2e including radiative forcing multiplier (aviation only)';
COMMENT ON COLUMN business_travel_service.gl_bt_calculations.wtt_co2e_kg IS 'Well-to-tank upstream emissions (kgCO2e)';
COMMENT ON COLUMN business_travel_service.gl_bt_calculations.dqi_score IS 'Data Quality Indicator score (1.0=highest to 5.0=lowest)';

-- =====================================================================================
-- TABLE 9: gl_bt_flight_results (HYPERTABLE)
-- Description: Per-flight emission calculation details
-- =====================================================================================

CREATE TABLE business_travel_service.gl_bt_flight_results (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id VARCHAR(200) NOT NULL,
    origin_iata VARCHAR(3) NOT NULL,
    destination_iata VARCHAR(3) NOT NULL,
    distance_km DECIMAL(20,8) NOT NULL,
    distance_band VARCHAR(50) NOT NULL,
    cabin_class VARCHAR(50) NOT NULL,
    passengers INT DEFAULT 1,
    class_multiplier DECIMAL(10,4),
    co2e_without_rf_kg DECIMAL(20,8),
    co2e_with_rf_kg DECIMAL(20,8),
    wtt_co2e_kg DECIMAL(20,8),
    total_co2e_kg DECIMAL(20,8) NOT NULL,
    rf_option VARCHAR(20),
    ef_source VARCHAR(100),
    round_trip BOOLEAN DEFAULT FALSE,
    provenance_hash VARCHAR(64),
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, calculated_at),
    CONSTRAINT chk_bt_flight_distance_positive CHECK (distance_km >= 0),
    CONSTRAINT chk_bt_flight_passengers_positive CHECK (passengers >= 1),
    CONSTRAINT chk_bt_flight_co2e_positive CHECK (total_co2e_kg >= 0)
);

-- Convert to hypertable (7-day chunks)
SELECT create_hypertable('business_travel_service.gl_bt_flight_results', 'calculated_at',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

CREATE INDEX idx_bt_flight_results_tenant ON business_travel_service.gl_bt_flight_results(tenant_id, calculated_at DESC);
CREATE INDEX idx_bt_flight_results_calc_id ON business_travel_service.gl_bt_flight_results(calculation_id);
CREATE INDEX idx_bt_flight_results_route ON business_travel_service.gl_bt_flight_results(origin_iata, destination_iata);
CREATE INDEX idx_bt_flight_results_band ON business_travel_service.gl_bt_flight_results(distance_band);
CREATE INDEX idx_bt_flight_results_cabin ON business_travel_service.gl_bt_flight_results(cabin_class);
CREATE INDEX idx_bt_flight_results_hash ON business_travel_service.gl_bt_flight_results(provenance_hash);

COMMENT ON TABLE business_travel_service.gl_bt_flight_results IS 'Per-flight emission calculation details (HYPERTABLE, 7-day chunks)';
COMMENT ON COLUMN business_travel_service.gl_bt_flight_results.distance_band IS 'Distance band classification: domestic, short_haul, medium_haul, long_haul';
COMMENT ON COLUMN business_travel_service.gl_bt_flight_results.class_multiplier IS 'Cabin class multiplier (economy=1.0, premium_economy=1.6, business=2.9, first=4.0)';
COMMENT ON COLUMN business_travel_service.gl_bt_flight_results.rf_option IS 'Radiative forcing option: with_rf, without_rf';

-- =====================================================================================
-- TABLE 10: gl_bt_ground_results
-- Description: Per-ground-trip emission calculation details
-- =====================================================================================

CREATE TABLE business_travel_service.gl_bt_ground_results (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id VARCHAR(200) NOT NULL,
    mode VARCHAR(50) NOT NULL,
    vehicle_type VARCHAR(100),
    fuel_type VARCHAR(50),
    distance_km DECIMAL(20,8),
    fuel_litres DECIMAL(20,8),
    passengers INT DEFAULT 1,
    co2e_kg DECIMAL(20,8) NOT NULL,
    wtt_co2e_kg DECIMAL(20,8),
    total_co2e_kg DECIMAL(20,8) NOT NULL,
    ef_source VARCHAR(100),
    calculation_method VARCHAR(50),
    provenance_hash VARCHAR(64),
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_bt_ground_distance_positive CHECK (distance_km IS NULL OR distance_km >= 0),
    CONSTRAINT chk_bt_ground_fuel_positive CHECK (fuel_litres IS NULL OR fuel_litres >= 0),
    CONSTRAINT chk_bt_ground_passengers_positive CHECK (passengers >= 1),
    CONSTRAINT chk_bt_ground_co2e_positive CHECK (total_co2e_kg >= 0)
);

CREATE INDEX idx_bt_ground_results_tenant ON business_travel_service.gl_bt_ground_results(tenant_id, calculated_at DESC);
CREATE INDEX idx_bt_ground_results_calc_id ON business_travel_service.gl_bt_ground_results(calculation_id);
CREATE INDEX idx_bt_ground_results_mode ON business_travel_service.gl_bt_ground_results(mode);
CREATE INDEX idx_bt_ground_results_vehicle ON business_travel_service.gl_bt_ground_results(vehicle_type);
CREATE INDEX idx_bt_ground_results_method ON business_travel_service.gl_bt_ground_results(calculation_method);

COMMENT ON TABLE business_travel_service.gl_bt_ground_results IS 'Per-ground-trip emission calculation details for rail, road, bus, taxi, ferry, motorcycle';
COMMENT ON COLUMN business_travel_service.gl_bt_ground_results.calculation_method IS 'Calculation method: distance_based, fuel_based, spend_based';

-- =====================================================================================
-- TABLE 11: gl_bt_hotel_results
-- Description: Per-hotel-stay emission calculation details
-- =====================================================================================

CREATE TABLE business_travel_service.gl_bt_hotel_results (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id VARCHAR(200) NOT NULL,
    country_code VARCHAR(10) NOT NULL,
    hotel_class VARCHAR(50),
    room_nights INT NOT NULL,
    class_multiplier DECIMAL(10,4) DEFAULT 1.0,
    co2e_kg DECIMAL(20,8) NOT NULL,
    total_co2e_kg DECIMAL(20,8) NOT NULL,
    ef_source VARCHAR(100),
    provenance_hash VARCHAR(64),
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_bt_hotel_nights_positive CHECK (room_nights >= 1),
    CONSTRAINT chk_bt_hotel_co2e_positive CHECK (total_co2e_kg >= 0),
    CONSTRAINT chk_bt_hotel_multiplier_positive CHECK (class_multiplier > 0)
);

CREATE INDEX idx_bt_hotel_results_tenant ON business_travel_service.gl_bt_hotel_results(tenant_id, calculated_at DESC);
CREATE INDEX idx_bt_hotel_results_calc_id ON business_travel_service.gl_bt_hotel_results(calculation_id);
CREATE INDEX idx_bt_hotel_results_country ON business_travel_service.gl_bt_hotel_results(country_code);
CREATE INDEX idx_bt_hotel_results_class ON business_travel_service.gl_bt_hotel_results(hotel_class);

COMMENT ON TABLE business_travel_service.gl_bt_hotel_results IS 'Per-hotel-stay emission calculation details by country and class';
COMMENT ON COLUMN business_travel_service.gl_bt_hotel_results.room_nights IS 'Number of room-nights for the stay';
COMMENT ON COLUMN business_travel_service.gl_bt_hotel_results.class_multiplier IS 'Hotel class multiplier (budget=0.75, standard=1.0, luxury=1.5)';

-- =====================================================================================
-- TABLE 12: gl_bt_spend_results
-- Description: Spend-based emission calculation details using EEIO factors
-- =====================================================================================

CREATE TABLE business_travel_service.gl_bt_spend_results (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id VARCHAR(200) NOT NULL,
    naics_code VARCHAR(20) NOT NULL,
    spend_original DECIMAL(20,8) NOT NULL,
    currency VARCHAR(10) NOT NULL,
    spend_usd DECIMAL(20,8) NOT NULL,
    cpi_deflator DECIMAL(10,6),
    eeio_factor DECIMAL(20,8),
    co2e_kg DECIMAL(20,8) NOT NULL,
    ef_source VARCHAR(100),
    provenance_hash VARCHAR(64),
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_bt_spend_positive CHECK (spend_original >= 0 AND spend_usd >= 0),
    CONSTRAINT chk_bt_spend_co2e_positive CHECK (co2e_kg >= 0),
    CONSTRAINT chk_bt_spend_deflator_positive CHECK (cpi_deflator IS NULL OR cpi_deflator > 0)
);

CREATE INDEX idx_bt_spend_results_tenant ON business_travel_service.gl_bt_spend_results(tenant_id, calculated_at DESC);
CREATE INDEX idx_bt_spend_results_calc_id ON business_travel_service.gl_bt_spend_results(calculation_id);
CREATE INDEX idx_bt_spend_results_naics ON business_travel_service.gl_bt_spend_results(naics_code);
CREATE INDEX idx_bt_spend_results_currency ON business_travel_service.gl_bt_spend_results(currency);

COMMENT ON TABLE business_travel_service.gl_bt_spend_results IS 'Spend-based emission calculation details using EPA USEEIO v2 factors';
COMMENT ON COLUMN business_travel_service.gl_bt_spend_results.spend_usd IS 'Spend converted to USD and deflated to EEIO base year';
COMMENT ON COLUMN business_travel_service.gl_bt_spend_results.cpi_deflator IS 'CPI deflator applied to convert current year spend to base year USD';

-- =====================================================================================
-- TABLE 13: gl_bt_compliance_checks
-- Description: Compliance check results against regulatory frameworks
-- =====================================================================================

CREATE TABLE business_travel_service.gl_bt_compliance_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id VARCHAR(200) NOT NULL,
    framework VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    score DECIMAL(5,2),
    findings JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_bt_compliance_status CHECK (status IN ('PASS', 'FAIL', 'WARNING', 'NOT_APPLICABLE')),
    CONSTRAINT chk_bt_compliance_score_range CHECK (score IS NULL OR (score >= 0 AND score <= 100))
);

CREATE INDEX idx_bt_compliance_tenant ON business_travel_service.gl_bt_compliance_checks(tenant_id);
CREATE INDEX idx_bt_compliance_calc_id ON business_travel_service.gl_bt_compliance_checks(calculation_id);
CREATE INDEX idx_bt_compliance_framework ON business_travel_service.gl_bt_compliance_checks(framework);
CREATE INDEX idx_bt_compliance_status ON business_travel_service.gl_bt_compliance_checks(status);
CREATE INDEX idx_bt_compliance_findings ON business_travel_service.gl_bt_compliance_checks USING GIN(findings);

COMMENT ON TABLE business_travel_service.gl_bt_compliance_checks IS 'Compliance check results against GHG Protocol, CSRD, CDP, SBTi, ISO 14064 frameworks';
COMMENT ON COLUMN business_travel_service.gl_bt_compliance_checks.status IS 'Compliance status: PASS, FAIL, WARNING, NOT_APPLICABLE';
COMMENT ON COLUMN business_travel_service.gl_bt_compliance_checks.findings IS 'JSONB array of compliance findings with severity and detail';

-- =====================================================================================
-- TABLE 14: gl_bt_uncertainty_analyses
-- Description: Uncertainty quantification for emission calculations
-- =====================================================================================

CREATE TABLE business_travel_service.gl_bt_uncertainty_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id VARCHAR(200) NOT NULL,
    method VARCHAR(50) NOT NULL,
    iterations INT,
    confidence_level DECIMAL(5,4),
    mean_co2e DECIMAL(20,8),
    std_dev DECIMAL(20,8),
    ci_lower DECIMAL(20,8),
    ci_upper DECIMAL(20,8),
    metadata JSONB DEFAULT '{}',
    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_bt_uncertainty_method CHECK (method IN ('MONTE_CARLO', 'IPCC_TIER2', 'ERROR_PROPAGATION', 'BOOTSTRAP')),
    CONSTRAINT chk_bt_uncertainty_iterations CHECK (iterations IS NULL OR iterations > 0),
    CONSTRAINT chk_bt_uncertainty_confidence CHECK (confidence_level IS NULL OR (confidence_level > 0 AND confidence_level <= 1)),
    CONSTRAINT chk_bt_uncertainty_bounds CHECK (ci_lower IS NULL OR ci_upper IS NULL OR ci_lower <= ci_upper),
    CONSTRAINT chk_bt_uncertainty_std_positive CHECK (std_dev IS NULL OR std_dev >= 0)
);

CREATE INDEX idx_bt_uncertainty_tenant ON business_travel_service.gl_bt_uncertainty_analyses(tenant_id);
CREATE INDEX idx_bt_uncertainty_calc_id ON business_travel_service.gl_bt_uncertainty_analyses(calculation_id);
CREATE INDEX idx_bt_uncertainty_method ON business_travel_service.gl_bt_uncertainty_analyses(method);

COMMENT ON TABLE business_travel_service.gl_bt_uncertainty_analyses IS 'Uncertainty quantification using Monte Carlo, IPCC Tier 2, or error propagation';
COMMENT ON COLUMN business_travel_service.gl_bt_uncertainty_analyses.method IS 'Uncertainty method: MONTE_CARLO, IPCC_TIER2, ERROR_PROPAGATION, BOOTSTRAP';
COMMENT ON COLUMN business_travel_service.gl_bt_uncertainty_analyses.confidence_level IS 'Confidence level as fraction (e.g., 0.95 for 95%)';

-- =====================================================================================
-- TABLE 15: gl_bt_aggregations (HYPERTABLE)
-- Description: Period aggregations by mode, department, cabin class, and country
-- =====================================================================================

CREATE TABLE business_travel_service.gl_bt_aggregations (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    period VARCHAR(20) NOT NULL,
    total_co2e_kg DECIMAL(20,8) NOT NULL,
    by_mode JSONB DEFAULT '{}',
    by_department JSONB DEFAULT '{}',
    by_cabin_class JSONB DEFAULT '{}',
    by_country JSONB DEFAULT '{}',
    trip_count INT DEFAULT 0,
    dqi_avg DECIMAL(5,2),
    metadata JSONB DEFAULT '{}',
    aggregated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, aggregated_at),
    CONSTRAINT chk_bt_agg_co2e_positive CHECK (total_co2e_kg >= 0),
    CONSTRAINT chk_bt_agg_trip_count_positive CHECK (trip_count >= 0),
    CONSTRAINT chk_bt_agg_dqi_range CHECK (dqi_avg IS NULL OR (dqi_avg >= 1.0 AND dqi_avg <= 5.0))
);

-- Convert to hypertable (30-day chunks)
SELECT create_hypertable('business_travel_service.gl_bt_aggregations', 'aggregated_at',
    chunk_time_interval => INTERVAL '30 days', if_not_exists => TRUE);

CREATE INDEX idx_bt_aggregations_tenant ON business_travel_service.gl_bt_aggregations(tenant_id, aggregated_at DESC);
CREATE INDEX idx_bt_aggregations_period ON business_travel_service.gl_bt_aggregations(period);
CREATE INDEX idx_bt_aggregations_by_mode ON business_travel_service.gl_bt_aggregations USING GIN(by_mode);
CREATE INDEX idx_bt_aggregations_by_dept ON business_travel_service.gl_bt_aggregations USING GIN(by_department);
CREATE INDEX idx_bt_aggregations_by_cabin ON business_travel_service.gl_bt_aggregations USING GIN(by_cabin_class);
CREATE INDEX idx_bt_aggregations_by_country ON business_travel_service.gl_bt_aggregations USING GIN(by_country);

COMMENT ON TABLE business_travel_service.gl_bt_aggregations IS 'Period aggregations of business travel emissions (HYPERTABLE, 30-day chunks)';
COMMENT ON COLUMN business_travel_service.gl_bt_aggregations.by_mode IS 'JSONB breakdown of CO2e by travel mode (air, rail, road, hotel)';
COMMENT ON COLUMN business_travel_service.gl_bt_aggregations.by_department IS 'JSONB breakdown of CO2e by department';
COMMENT ON COLUMN business_travel_service.gl_bt_aggregations.by_cabin_class IS 'JSONB breakdown of CO2e by cabin class (economy, business, first)';
COMMENT ON COLUMN business_travel_service.gl_bt_aggregations.by_country IS 'JSONB breakdown of CO2e by destination country';

-- =====================================================================================
-- TABLE 16: gl_bt_provenance
-- Description: Provenance tracking with SHA-256 hash chains
-- =====================================================================================

CREATE TABLE business_travel_service.gl_bt_provenance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    calculation_id VARCHAR(200) NOT NULL,
    stage VARCHAR(50) NOT NULL,
    input_hash VARCHAR(64) NOT NULL,
    output_hash VARCHAR(64) NOT NULL,
    chain_hash VARCHAR(64) NOT NULL,
    stage_index INT NOT NULL,
    metadata JSONB DEFAULT '{}',
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_bt_provenance_stage CHECK (stage IN ('INTAKE', 'DISTANCE_CALC', 'EF_LOOKUP', 'CALCULATION', 'VALIDATION', 'AGGREGATION', 'COMPLIANCE')),
    CONSTRAINT chk_bt_provenance_index_positive CHECK (stage_index >= 0)
);

CREATE INDEX idx_bt_provenance_tenant ON business_travel_service.gl_bt_provenance(tenant_id);
CREATE INDEX idx_bt_provenance_calc_id ON business_travel_service.gl_bt_provenance(calculation_id);
CREATE INDEX idx_bt_provenance_calc_stage ON business_travel_service.gl_bt_provenance(calculation_id, stage_index);
CREATE INDEX idx_bt_provenance_chain ON business_travel_service.gl_bt_provenance(chain_hash);
CREATE INDEX idx_bt_provenance_recorded ON business_travel_service.gl_bt_provenance(recorded_at DESC);

COMMENT ON TABLE business_travel_service.gl_bt_provenance IS 'Provenance tracking for business travel emission calculations with SHA-256 hash chains';
COMMENT ON COLUMN business_travel_service.gl_bt_provenance.stage IS 'Processing stage: INTAKE, DISTANCE_CALC, EF_LOOKUP, CALCULATION, VALIDATION, AGGREGATION, COMPLIANCE';
COMMENT ON COLUMN business_travel_service.gl_bt_provenance.chain_hash IS 'SHA-256 hash chaining input_hash + output_hash + previous chain_hash';

-- =====================================================================================
-- CONTINUOUS AGGREGATES
-- =====================================================================================

-- Continuous Aggregate 1: Hourly Emissions
CREATE MATERIALIZED VIEW business_travel_service.gl_bt_hourly_emissions
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', calculated_at) AS bucket,
    tenant_id,
    mode,
    method,
    COUNT(*) AS calc_count,
    SUM(total_co2e_kg) AS total_co2e_kg,
    AVG(dqi_score) AS avg_dqi_score
FROM business_travel_service.gl_bt_calculations
GROUP BY bucket, tenant_id, mode, method
WITH NO DATA;

-- Refresh policy for hourly emissions (refresh every 1 hour, lag 2 hours)
SELECT add_continuous_aggregate_policy('business_travel_service.gl_bt_hourly_emissions',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '2 hours',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW business_travel_service.gl_bt_hourly_emissions IS 'Hourly aggregation of business travel emissions by mode and method';

-- Continuous Aggregate 2: Daily Emissions
CREATE MATERIALIZED VIEW business_travel_service.gl_bt_daily_emissions
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', calculated_at) AS bucket,
    tenant_id,
    mode,
    method,
    COUNT(*) AS calc_count,
    SUM(total_co2e_kg) AS total_co2e_kg,
    AVG(dqi_score) AS avg_dqi_score
FROM business_travel_service.gl_bt_calculations
GROUP BY bucket, tenant_id, mode, method
WITH NO DATA;

-- Refresh policy for daily emissions (refresh every 6 hours, lag 12 hours)
SELECT add_continuous_aggregate_policy('business_travel_service.gl_bt_daily_emissions',
    start_offset => INTERVAL '30 days',
    end_offset => INTERVAL '12 hours',
    schedule_interval => INTERVAL '6 hours',
    if_not_exists => TRUE
);

COMMENT ON MATERIALIZED VIEW business_travel_service.gl_bt_daily_emissions IS 'Daily aggregation of business travel emissions with mode and method breakdown';

-- =====================================================================================
-- ROW LEVEL SECURITY (RLS)
-- =====================================================================================

-- Enable RLS on operational tables with tenant_id
ALTER TABLE business_travel_service.gl_bt_calculations ENABLE ROW LEVEL SECURITY;
ALTER TABLE business_travel_service.gl_bt_flight_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE business_travel_service.gl_bt_ground_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE business_travel_service.gl_bt_hotel_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE business_travel_service.gl_bt_spend_results ENABLE ROW LEVEL SECURITY;

-- RLS Policy: gl_bt_calculations
CREATE POLICY bt_calculations_tenant_isolation ON business_travel_service.gl_bt_calculations
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- RLS Policy: gl_bt_flight_results
CREATE POLICY bt_flight_results_tenant_isolation ON business_travel_service.gl_bt_flight_results
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- RLS Policy: gl_bt_ground_results
CREATE POLICY bt_ground_results_tenant_isolation ON business_travel_service.gl_bt_ground_results
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- RLS Policy: gl_bt_hotel_results
CREATE POLICY bt_hotel_results_tenant_isolation ON business_travel_service.gl_bt_hotel_results
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- RLS Policy: gl_bt_spend_results
CREATE POLICY bt_spend_results_tenant_isolation ON business_travel_service.gl_bt_spend_results
    USING (tenant_id = current_setting('app.current_tenant_id', TRUE));

-- =====================================================================================
-- SEED DATA: AIRPORTS (50 major airports)
-- =====================================================================================

INSERT INTO business_travel_service.gl_bt_airports
(iata_code, name, city, country_code, latitude, longitude, timezone, is_active) VALUES
-- North America
('JFK', 'John F. Kennedy International Airport', 'New York', 'US', 40.639801, -73.778900, 'America/New_York', TRUE),
('LAX', 'Los Angeles International Airport', 'Los Angeles', 'US', 33.942501, -118.408203, 'America/Los_Angeles', TRUE),
('ORD', 'O''Hare International Airport', 'Chicago', 'US', 41.978600, -87.904800, 'America/Chicago', TRUE),
('ATL', 'Hartsfield-Jackson Atlanta International Airport', 'Atlanta', 'US', 33.636700, -84.428101, 'America/New_York', TRUE),
('DFW', 'Dallas/Fort Worth International Airport', 'Dallas', 'US', 32.896801, -97.038002, 'America/Chicago', TRUE),
('DEN', 'Denver International Airport', 'Denver', 'US', 39.861698, -104.672997, 'America/Denver', TRUE),
('SFO', 'San Francisco International Airport', 'San Francisco', 'US', 37.618999, -122.375000, 'America/Los_Angeles', TRUE),
('SEA', 'Seattle-Tacoma International Airport', 'Seattle', 'US', 47.449001, -122.309998, 'America/Los_Angeles', TRUE),
('MIA', 'Miami International Airport', 'Miami', 'US', 25.795900, -80.287003, 'America/New_York', TRUE),
('BOS', 'Boston Logan International Airport', 'Boston', 'US', 42.364300, -71.005203, 'America/New_York', TRUE),
('IAD', 'Washington Dulles International Airport', 'Washington DC', 'US', 38.944500, -77.455803, 'America/New_York', TRUE),
('EWR', 'Newark Liberty International Airport', 'Newark', 'US', 40.692501, -74.168701, 'America/New_York', TRUE),
('YYZ', 'Toronto Pearson International Airport', 'Toronto', 'CA', 43.677200, -79.630600, 'America/Toronto', TRUE),
('YVR', 'Vancouver International Airport', 'Vancouver', 'CA', 49.193901, -123.184403, 'America/Vancouver', TRUE),

-- Europe
('LHR', 'London Heathrow Airport', 'London', 'GB', 51.470600, -0.461941, 'Europe/London', TRUE),
('LGW', 'London Gatwick Airport', 'London', 'GB', 51.148102, -0.190278, 'Europe/London', TRUE),
('MAN', 'Manchester Airport', 'Manchester', 'GB', 53.353699, -2.274950, 'Europe/London', TRUE),
('EDI', 'Edinburgh Airport', 'Edinburgh', 'GB', 55.950100, -3.372500, 'Europe/London', TRUE),
('CDG', 'Charles de Gaulle Airport', 'Paris', 'FR', 49.012798, 2.550000, 'Europe/Paris', TRUE),
('FRA', 'Frankfurt Airport', 'Frankfurt', 'DE', 50.033333, 8.570556, 'Europe/Berlin', TRUE),
('AMS', 'Amsterdam Airport Schiphol', 'Amsterdam', 'NL', 52.308601, 4.763889, 'Europe/Amsterdam', TRUE),
('MAD', 'Adolfo Suarez Madrid-Barajas Airport', 'Madrid', 'ES', 40.471926, -3.560764, 'Europe/Madrid', TRUE),
('BCN', 'Barcelona-El Prat Airport', 'Barcelona', 'ES', 41.297100, 2.078463, 'Europe/Madrid', TRUE),
('FCO', 'Leonardo da Vinci-Fiumicino Airport', 'Rome', 'IT', 41.800278, 12.238889, 'Europe/Rome', TRUE),
('MXP', 'Milan Malpensa Airport', 'Milan', 'IT', 45.630600, 8.728111, 'Europe/Rome', TRUE),
('MUC', 'Munich Airport', 'Munich', 'DE', 48.353783, 11.786086, 'Europe/Berlin', TRUE),
('ZRH', 'Zurich Airport', 'Zurich', 'CH', 47.464699, 8.549170, 'Europe/Zurich', TRUE),
('CPH', 'Copenhagen Airport', 'Copenhagen', 'DK', 55.617900, 12.655972, 'Europe/Copenhagen', TRUE),
('ARN', 'Stockholm Arlanda Airport', 'Stockholm', 'SE', 59.651901, 17.918600, 'Europe/Stockholm', TRUE),
('OSL', 'Oslo Gardermoen Airport', 'Oslo', 'NO', 60.193901, 11.100399, 'Europe/Oslo', TRUE),
('HEL', 'Helsinki-Vantaa Airport', 'Helsinki', 'FI', 60.317200, 24.963301, 'Europe/Helsinki', TRUE),
('DUB', 'Dublin Airport', 'Dublin', 'IE', 53.421299, -6.270080, 'Europe/Dublin', TRUE),

-- Middle East
('DXB', 'Dubai International Airport', 'Dubai', 'AE', 25.252800, 55.364399, 'Asia/Dubai', TRUE),
('DOH', 'Hamad International Airport', 'Doha', 'QA', 25.273100, 51.608101, 'Asia/Qatar', TRUE),

-- Asia Pacific
('SIN', 'Singapore Changi Airport', 'Singapore', 'SG', 1.350190, 103.994003, 'Asia/Singapore', TRUE),
('HND', 'Tokyo Haneda Airport', 'Tokyo', 'JP', 35.552299, 139.779999, 'Asia/Tokyo', TRUE),
('NRT', 'Narita International Airport', 'Tokyo', 'JP', 35.764702, 140.386002, 'Asia/Tokyo', TRUE),
('PEK', 'Beijing Capital International Airport', 'Beijing', 'CN', 40.080101, 116.584999, 'Asia/Shanghai', TRUE),
('PVG', 'Shanghai Pudong International Airport', 'Shanghai', 'CN', 31.143400, 121.805801, 'Asia/Shanghai', TRUE),
('HKG', 'Hong Kong International Airport', 'Hong Kong', 'HK', 22.308901, 113.914703, 'Asia/Hong_Kong', TRUE),
('ICN', 'Incheon International Airport', 'Seoul', 'KR', 37.469100, 126.450996, 'Asia/Seoul', TRUE),
('DEL', 'Indira Gandhi International Airport', 'New Delhi', 'IN', 28.566500, 77.103104, 'Asia/Kolkata', TRUE),
('BOM', 'Chhatrapati Shivaji International Airport', 'Mumbai', 'IN', 19.088600, 72.867897, 'Asia/Kolkata', TRUE),
('BKK', 'Suvarnabhumi Airport', 'Bangkok', 'TH', 13.681100, 100.747200, 'Asia/Bangkok', TRUE),
('KUL', 'Kuala Lumpur International Airport', 'Kuala Lumpur', 'MY', 2.745580, 101.709999, 'Asia/Kuala_Lumpur', TRUE),

-- Oceania
('SYD', 'Sydney Kingsford Smith Airport', 'Sydney', 'AU', -33.946100, 151.177002, 'Australia/Sydney', TRUE),
('MEL', 'Melbourne Airport', 'Melbourne', 'AU', -37.673302, 144.843002, 'Australia/Melbourne', TRUE),

-- South America
('GRU', 'Sao Paulo-Guarulhos International Airport', 'Sao Paulo', 'BR', -23.435600, -46.473099, 'America/Sao_Paulo', TRUE),

-- Africa
('JNB', 'O.R. Tambo International Airport', 'Johannesburg', 'ZA', -26.139200, 28.246000, 'Africa/Johannesburg', TRUE),
('NBO', 'Jomo Kenyatta International Airport', 'Nairobi', 'KE', -1.319200, 36.927800, 'Africa/Nairobi', TRUE);

-- =====================================================================================
-- SEED DATA: AIR EMISSION FACTORS (DEFRA 2024 - 4 distance bands x economy)
-- =====================================================================================

INSERT INTO business_travel_service.gl_bt_air_emission_factors
(distance_band, cabin_class, ef_without_rf, ef_with_rf, wtt_ef, unit, source, year, is_active) VALUES
-- Domestic flights (<463 km UK / used as short domestic benchmark)
('domestic',        'economy',         0.24587000, 0.27046000, 0.05765000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('domestic',        'premium_economy', 0.39339000, 0.43273000, 0.09224000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('domestic',        'business',        0.71303000, 0.78433000, 0.16719000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('domestic',        'first',           0.98348000, 1.08183000, 0.23060000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),

-- Short-haul flights (<3700 km)
('short_haul',      'economy',         0.15353000, 0.16888000, 0.03600000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('short_haul',      'premium_economy', 0.24565000, 0.27021000, 0.05760000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('short_haul',      'business',        0.44524000, 0.48976000, 0.10440000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('short_haul',      'first',           0.61412000, 0.67553000, 0.14400000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),

-- Medium-haul flights (3700 km - 6000 km)
-- Note: DEFRA groups long haul; these are interpolated for 3700-6000km band
('medium_haul',     'economy',         0.13813000, 0.15194000, 0.03238000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('medium_haul',     'premium_economy', 0.22101000, 0.24311000, 0.05181000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('medium_haul',     'business',        0.40058000, 0.44064000, 0.09390000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('medium_haul',     'first',           0.55254000, 0.60779000, 0.12954000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),

-- Long-haul flights (>6000 km)
('long_haul',       'economy',         0.10231000, 0.11254000, 0.02398000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('long_haul',       'premium_economy', 0.16370000, 0.18007000, 0.03836000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('long_haul',       'business',        0.29670000, 0.32637000, 0.06955000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('long_haul',       'first',           0.40924000, 0.45017000, 0.09594000, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE);

-- =====================================================================================
-- SEED DATA: GROUND EMISSION FACTORS (DEFRA 2024 - rail, road, bus, taxi, ferry, motorcycle)
-- =====================================================================================

INSERT INTO business_travel_service.gl_bt_ground_emission_factors
(mode, vehicle_type, ef_per_vkm, ef_per_pkm, wtt_ef, occupancy, unit, source, year, is_active) VALUES
-- Rail
('rail', 'national_rail',          NULL, 0.03549000, 0.00790000, NULL, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('rail', 'international_rail',     NULL, 0.00446000, 0.00101000, NULL, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('rail', 'light_rail_tram',        NULL, 0.02910000, 0.00670000, NULL, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('rail', 'london_underground',     NULL, 0.02781000, 0.00640000, NULL, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('rail', 'eurostar',               NULL, 0.00446000, 0.00101000, NULL, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('rail', 'high_speed_rail',        NULL, 0.00600000, 0.00140000, NULL, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),

-- Road - Cars
('road', 'average_car',            0.17140000, 0.10963000, 0.02690000, 1.5630, 'kgCO2e/vkm', 'DEFRA_2024', 2024, TRUE),
('road', 'small_car_petrol',       0.14890000, 0.09530000, 0.02330000, 1.5630, 'kgCO2e/vkm', 'DEFRA_2024', 2024, TRUE),
('road', 'medium_car_petrol',      0.18770000, 0.12010000, 0.02940000, 1.5630, 'kgCO2e/vkm', 'DEFRA_2024', 2024, TRUE),
('road', 'large_car_petrol',       0.27870000, 0.17830000, 0.04370000, 1.5630, 'kgCO2e/vkm', 'DEFRA_2024', 2024, TRUE),
('road', 'small_car_diesel',       0.13920000, 0.08910000, 0.02070000, 1.5630, 'kgCO2e/vkm', 'DEFRA_2024', 2024, TRUE),
('road', 'medium_car_diesel',      0.16610000, 0.10630000, 0.02470000, 1.5630, 'kgCO2e/vkm', 'DEFRA_2024', 2024, TRUE),
('road', 'large_car_diesel',       0.20870000, 0.13360000, 0.03100000, 1.5630, 'kgCO2e/vkm', 'DEFRA_2024', 2024, TRUE),
('road', 'hybrid_car',             0.11590000, 0.07420000, 0.01820000, 1.5630, 'kgCO2e/vkm', 'DEFRA_2024', 2024, TRUE),
('road', 'plugin_hybrid_car',      0.06920000, 0.04430000, 0.01440000, 1.5630, 'kgCO2e/vkm', 'DEFRA_2024', 2024, TRUE),
('road', 'battery_ev',             0.04600000, 0.02940000, 0.01330000, 1.5630, 'kgCO2e/vkm', 'DEFRA_2024', 2024, TRUE),

-- Bus and Coach
('bus', 'local_bus',               NULL, 0.10312000, 0.01670000, NULL, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('bus', 'average_local_bus',       NULL, 0.10312000, 0.01670000, NULL, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('bus', 'coach',                   NULL, 0.02732000, 0.00447000, NULL, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),

-- Taxi
('taxi', 'regular_taxi',           NULL, 0.14880000, 0.03600000, NULL, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('taxi', 'black_cab',              NULL, 0.20810000, 0.05040000, NULL, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),

-- Ferry
('ferry', 'foot_passenger',        NULL, 0.01870000, 0.00370000, NULL, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('ferry', 'car_passenger',         NULL, 0.12952000, 0.02560000, NULL, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),
('ferry', 'average_ferry',         NULL, 0.11268000, 0.02230000, NULL, 'kgCO2e/pkm', 'DEFRA_2024', 2024, TRUE),

-- Motorcycle
('motorcycle', 'small_motorcycle',  0.08310000, 0.08310000, 0.01530000, 1.0000, 'kgCO2e/vkm', 'DEFRA_2024', 2024, TRUE),
('motorcycle', 'medium_motorcycle', 0.10100000, 0.10100000, 0.01860000, 1.0000, 'kgCO2e/vkm', 'DEFRA_2024', 2024, TRUE),
('motorcycle', 'large_motorcycle',  0.13240000, 0.13240000, 0.02440000, 1.0000, 'kgCO2e/vkm', 'DEFRA_2024', 2024, TRUE);

-- =====================================================================================
-- SEED DATA: HOTEL EMISSION FACTORS (16 countries + GLOBAL default)
-- =====================================================================================

INSERT INTO business_travel_service.gl_bt_hotel_emission_factors
(country_code, hotel_class, ef_per_room_night, class_multiplier, unit, source, year, is_active) VALUES
-- United States
('US', 'budget',   20.40000000, 0.75, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('US', 'standard', 27.20000000, 1.00, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('US', 'luxury',   40.80000000, 1.50, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),

-- United Kingdom
('GB', 'budget',   12.15000000, 0.75, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('GB', 'standard', 16.20000000, 1.00, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('GB', 'luxury',   24.30000000, 1.50, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),

-- Germany
('DE', 'budget',   13.50000000, 0.75, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('DE', 'standard', 18.00000000, 1.00, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('DE', 'luxury',   27.00000000, 1.50, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),

-- France
('FR', 'budget',    8.25000000, 0.75, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('FR', 'standard', 11.00000000, 1.00, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('FR', 'luxury',   16.50000000, 1.50, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),

-- Japan
('JP', 'budget',   18.00000000, 0.75, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('JP', 'standard', 24.00000000, 1.00, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('JP', 'luxury',   36.00000000, 1.50, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),

-- China
('CN', 'budget',   24.00000000, 0.75, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('CN', 'standard', 32.00000000, 1.00, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('CN', 'luxury',   48.00000000, 1.50, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),

-- India
('IN', 'budget',   15.00000000, 0.75, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('IN', 'standard', 20.00000000, 1.00, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('IN', 'luxury',   30.00000000, 1.50, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),

-- Singapore
('SG', 'budget',   16.50000000, 0.75, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('SG', 'standard', 22.00000000, 1.00, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('SG', 'luxury',   33.00000000, 1.50, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),

-- Australia
('AU', 'budget',   22.50000000, 0.75, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('AU', 'standard', 30.00000000, 1.00, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('AU', 'luxury',   45.00000000, 1.50, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),

-- Brazil
('BR', 'budget',   12.00000000, 0.75, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('BR', 'standard', 16.00000000, 1.00, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('BR', 'luxury',   24.00000000, 1.50, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),

-- Canada
('CA', 'budget',   19.50000000, 0.75, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('CA', 'standard', 26.00000000, 1.00, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('CA', 'luxury',   39.00000000, 1.50, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),

-- United Arab Emirates
('AE', 'budget',   33.75000000, 0.75, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('AE', 'standard', 45.00000000, 1.00, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('AE', 'luxury',   67.50000000, 1.50, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),

-- South Korea
('KR', 'budget',   16.50000000, 0.75, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('KR', 'standard', 22.00000000, 1.00, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('KR', 'luxury',   33.00000000, 1.50, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),

-- South Africa
('ZA', 'budget',   22.50000000, 0.75, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('ZA', 'standard', 30.00000000, 1.00, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('ZA', 'luxury',   45.00000000, 1.50, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),

-- Netherlands
('NL', 'budget',   11.25000000, 0.75, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('NL', 'standard', 15.00000000, 1.00, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('NL', 'luxury',   22.50000000, 1.50, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),

-- Spain
('ES', 'budget',   10.50000000, 0.75, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('ES', 'standard', 14.00000000, 1.00, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('ES', 'luxury',   21.00000000, 1.50, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),

-- GLOBAL default (weighted average)
('GLOBAL', 'budget',   15.00000000, 0.75, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('GLOBAL', 'standard', 20.00000000, 1.00, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE),
('GLOBAL', 'luxury',   30.00000000, 1.50, 'kgCO2e/room-night', 'DEFRA_2024', 2024, TRUE);

-- =====================================================================================
-- SEED DATA: EEIO SPEND-BASED FACTORS (10 NAICS codes for travel categories)
-- =====================================================================================

INSERT INTO business_travel_service.gl_bt_eeio_factors
(naics_code, category_name, ef_per_usd, base_year, unit, source, is_active) VALUES
('481000', 'Air transportation',                             0.47800000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),
('482000', 'Rail transportation',                            0.22300000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),
('485000', 'Transit and ground passenger transportation',    0.35100000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),
('485310', 'Taxi and ridesharing services',                  0.30200000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),
('485510', 'Charter bus industry',                           0.28900000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),
('483000', 'Water transportation',                           0.56100000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),
('721000', 'Accommodation',                                  0.19400000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),
('721110', 'Hotels and motels',                              0.19400000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),
('532100', 'Automotive equipment rental and leasing',        0.25600000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE),
('561500', 'Travel arrangement and reservation services',    0.10200000, 2021, 'kgCO2e/USD', 'EPA_USEEIO_v2', TRUE);

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
    'GL-MRV-SCOPE3-006',
    'Business Travel Agent',
    '1.0.0',
    'CALCULATION',
    'SCOPE3_EMISSIONS',
    'AGENT-MRV-019: Scope 3 Category 6 - Business Travel. Calculates emissions from employee business travel including flights (with/without radiative forcing), rail, road, bus, taxi, ferry, motorcycle, and hotel accommodation. Supports distance-based (DEFRA/ICAO), fuel-based, spend-based (EPA USEEIO v2), and supplier-specific calculation methods. Includes 50 airports, 16 air EFs (4 bands x 4 classes), 27 ground EFs, 51 hotel EFs (17 countries x 3 classes), and 10 EEIO spend factors.',
    'ACTIVE',
    jsonb_build_object(
        'scope3_category', 6,
        'category_name', 'Business Travel',
        'calculation_methods', jsonb_build_array('distance_based', 'fuel_based', 'spend_based', 'supplier_specific', 'hotel_nights'),
        'travel_modes', jsonb_build_array('air', 'rail', 'road', 'bus', 'taxi', 'ferry', 'motorcycle', 'hotel'),
        'air_distance_bands', jsonb_build_array('domestic', 'short_haul', 'medium_haul', 'long_haul'),
        'cabin_classes', jsonb_build_array('economy', 'premium_economy', 'business', 'first'),
        'hotel_classes', jsonb_build_array('budget', 'standard', 'luxury'),
        'frameworks', jsonb_build_array('GHG Protocol Scope 3', 'CSRD ESRS E1', 'CDP Climate', 'SBTi', 'ISO 14064-1', 'DEFRA', 'ICAO'),
        'airports_count', 50,
        'air_ef_count', 16,
        'ground_ef_count', 27,
        'hotel_ef_count', 51,
        'eeio_factor_count', 10,
        'supports_radiative_forcing', true,
        'supports_wtt_emissions', true,
        'supports_cabin_class_multiplier', true,
        'supports_hotel_class_multiplier', true,
        'supports_cpi_deflation', true,
        'supports_great_circle_distance', true,
        'default_ef_source', 'DEFRA_2024',
        'default_gwp', 'AR5',
        'schema', 'business_travel_service',
        'table_prefix', 'gl_bt_',
        'hypertables', jsonb_build_array('gl_bt_calculations', 'gl_bt_flight_results', 'gl_bt_aggregations'),
        'continuous_aggregates', jsonb_build_array('gl_bt_hourly_emissions', 'gl_bt_daily_emissions'),
        'migration_version', 'V070'
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

COMMENT ON SCHEMA business_travel_service IS 'Updated: AGENT-MRV-019 complete with 16 tables, 3 hypertables, 2 continuous aggregates, RLS policies, 150+ seed records';

-- =====================================================================================
-- END OF MIGRATION V070
-- =====================================================================================
-- Total Lines: ~830
-- Total Tables: 16
-- Total Hypertables: 3 (calculations, flight_results, aggregations)
-- Total Continuous Aggregates: 2 (hourly_emissions, daily_emissions)
-- Total RLS Policies: 5 (calculations, flight_results, ground_results, hotel_results, spend_results)
-- Total Seed Records: 154
--   Airports: 50
--   Air Emission Factors: 16 (4 distance bands x 4 cabin classes)
--   Ground Emission Factors: 27 (rail/road/bus/taxi/ferry/motorcycle)
--   Hotel Emission Factors: 51 (17 country/region codes x 3 hotel classes)
--   EEIO Factors: 10 (NAICS codes for travel categories)
--   Agent Registry: 1
-- =====================================================================================
