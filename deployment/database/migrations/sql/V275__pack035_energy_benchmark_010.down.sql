-- =============================================================================
-- V275 DOWN: Drop views, materialized views, audit trail, indexes, seed data
-- =============================================================================

-- Drop view first
DROP VIEW IF EXISTS pack035_energy_benchmark.v_benchmark_dashboard CASCADE;

-- Drop materialized views
DROP MATERIALIZED VIEW IF EXISTS pack035_energy_benchmark.mv_peer_group_stats CASCADE;
DROP MATERIALIZED VIEW IF EXISTS pack035_energy_benchmark.mv_portfolio_summary CASCADE;
DROP MATERIALIZED VIEW IF EXISTS pack035_energy_benchmark.mv_facility_latest_eui CASCADE;

-- Drop additional composite indexes
DROP INDEX IF EXISTS pack035_energy_benchmark.idx_p035_it_fac_active;
DROP INDEX IF EXISTS pack035_energy_benchmark.idx_p035_ga_fac_approved;
DROP INDEX IF EXISTS pack035_energy_benchmark.idx_p035_pa_fac_sev_unack;
DROP INDEX IF EXISTS pack035_energy_benchmark.idx_p035_ecr_fac_period;
DROP INDEX IF EXISTS pack035_energy_benchmark.idx_p035_eui_fac_period;
DROP INDEX IF EXISTS pack035_energy_benchmark.idx_p035_fp_type_country;

-- Drop seed data (weather stations, building type mappings, benchmark sources)
DELETE FROM pack035_energy_benchmark.benchmark_building_types
    WHERE source_id IN (SELECT id FROM pack035_energy_benchmark.benchmark_sources WHERE source_code IN (
        'ENERGY_STAR', 'CIBSE_TM46', 'DIN_V_18599', 'BPIE', 'TABULA', 'NABERS', 'EU_EED', 'ASHRAE_100'
    ));
DELETE FROM pack035_energy_benchmark.benchmark_sources
    WHERE source_code IN ('ENERGY_STAR', 'CIBSE_TM46', 'DIN_V_18599', 'BPIE', 'TABULA', 'NABERS', 'EU_EED', 'ASHRAE_100');
DELETE FROM pack035_energy_benchmark.weather_stations
    WHERE station_id IN ('EGLL', 'LFPG', 'EDDB', 'EHAM', 'LEMD', 'LIRF', 'LOWW', 'EKCH', 'ESSB', 'EFHK', 'LPPT', 'EPWA', 'EIDW', 'LSZH', 'EBBR');

-- Drop RLS policies
DROP POLICY IF EXISTS p035_trail_service_bypass ON pack035_energy_benchmark.pack035_audit_trail;
DROP POLICY IF EXISTS p035_trail_tenant_isolation ON pack035_energy_benchmark.pack035_audit_trail;

-- Disable RLS
ALTER TABLE IF EXISTS pack035_energy_benchmark.pack035_audit_trail DISABLE ROW LEVEL SECURITY;

-- Drop audit trail table
DROP TABLE IF EXISTS pack035_energy_benchmark.pack035_audit_trail CASCADE;
