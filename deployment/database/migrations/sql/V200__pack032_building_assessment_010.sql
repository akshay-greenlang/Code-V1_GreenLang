-- =============================================================================
-- V200: PACK-032 Building Energy Assessment - Audit Trail, Views & Compliance
-- =============================================================================
-- Pack:         PACK-032 (Building Energy Assessment Pack)
-- Migration:    010 of 010
-- Date:         March 2026
--
-- Creates audit trail, compliance records, and analytical views for building
-- energy assessment reporting and regulatory compliance dashboards.
--
-- Tables (2):
--   1. pack032_building_assessment.pack032_audit_trail
--   2. pack032_building_assessment.compliance_records
--
-- Views (4):
--   1. pack032_building_assessment.v_building_performance_summary
--   2. pack032_building_assessment.v_portfolio_benchmarks
--   3. pack032_building_assessment.v_retrofit_portfolio
--   4. pack032_building_assessment.v_compliance_dashboard
--
-- Previous: V199__pack032_building_assessment_009.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack032_building_assessment.pack032_audit_trail
-- =============================================================================
-- Audit trail for all PACK-032 entity changes with old/new values for
-- regulatory compliance and data governance.

CREATE TABLE pack032_building_assessment.pack032_audit_trail (
    entry_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID,
    tenant_id               UUID            NOT NULL,
    action                  VARCHAR(50)     NOT NULL,
    entity_type             VARCHAR(100)    NOT NULL,
    entity_id               UUID            NOT NULL,
    old_values              JSONB,
    new_values              JSONB,
    user_id                 UUID,
    timestamp               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_audit_action CHECK (
        action IN ('CREATE', 'UPDATE', 'DELETE', 'ARCHIVE', 'RESTORE', 'APPROVE', 'REJECT')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_audit_building   ON pack032_building_assessment.pack032_audit_trail(building_id);
CREATE INDEX idx_p032_audit_tenant     ON pack032_building_assessment.pack032_audit_trail(tenant_id);
CREATE INDEX idx_p032_audit_entity     ON pack032_building_assessment.pack032_audit_trail(entity_type, entity_id);
CREATE INDEX idx_p032_audit_user       ON pack032_building_assessment.pack032_audit_trail(user_id);
CREATE INDEX idx_p032_audit_timestamp  ON pack032_building_assessment.pack032_audit_trail(timestamp DESC);
CREATE INDEX idx_p032_audit_action     ON pack032_building_assessment.pack032_audit_trail(action);
CREATE INDEX idx_p032_audit_old_vals   ON pack032_building_assessment.pack032_audit_trail USING GIN(old_values);
CREATE INDEX idx_p032_audit_new_vals   ON pack032_building_assessment.pack032_audit_trail USING GIN(new_values);

-- =============================================================================
-- Table 2: pack032_building_assessment.compliance_records
-- =============================================================================
-- Regulatory compliance tracking for building energy performance requirements
-- (MEES, EPBD, Part L, etc.) with deadlines and penalty information.

CREATE TABLE pack032_building_assessment.compliance_records (
    record_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    regulation              VARCHAR(255)    NOT NULL,
    requirement             TEXT            NOT NULL,
    current_value           VARCHAR(255),
    minimum_value           VARCHAR(255),
    compliant               BOOLEAN         NOT NULL DEFAULT FALSE,
    deadline                DATE,
    penalty_description     TEXT,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_cr_building   ON pack032_building_assessment.compliance_records(building_id);
CREATE INDEX idx_p032_cr_tenant     ON pack032_building_assessment.compliance_records(tenant_id);
CREATE INDEX idx_p032_cr_regulation ON pack032_building_assessment.compliance_records(regulation);
CREATE INDEX idx_p032_cr_compliant  ON pack032_building_assessment.compliance_records(compliant);
CREATE INDEX idx_p032_cr_deadline   ON pack032_building_assessment.compliance_records(deadline);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_cr_updated
    BEFORE UPDATE ON pack032_building_assessment.compliance_records
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack032_building_assessment.pack032_audit_trail ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.compliance_records ENABLE ROW LEVEL SECURITY;

CREATE POLICY p032_audit_tenant_isolation
    ON pack032_building_assessment.pack032_audit_trail
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_audit_service_bypass
    ON pack032_building_assessment.pack032_audit_trail
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_cr_tenant_isolation
    ON pack032_building_assessment.compliance_records
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_cr_service_bypass
    ON pack032_building_assessment.compliance_records
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.pack032_audit_trail TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.compliance_records TO PUBLIC;

-- =============================================================================
-- View 1: v_building_performance_summary
-- =============================================================================
-- Comprehensive building performance summary joining profiles, latest EPC,
-- latest DEC, latest benchmark, and renewable capacity.

CREATE OR REPLACE VIEW pack032_building_assessment.v_building_performance_summary AS
SELECT
    bp.building_id,
    bp.tenant_id,
    bp.building_name,
    bp.building_type,
    bp.country,
    bp.postcode,
    bp.gross_floor_area_m2,
    bp.year_built,
    bp.epc_rating              AS current_epc_rating,
    bp.dec_rating              AS current_dec_rating,
    bp.energy_star_score       AS current_energy_star,
    bp.crrem_aligned,
    -- Latest EPC
    epc.epc_rating             AS latest_epc_rating,
    epc.primary_energy_kwh_m2  AS latest_epc_primary_energy,
    epc.co2_emissions_kg_m2    AS latest_epc_co2,
    epc.assessment_date        AS latest_epc_date,
    epc.expiry_date            AS epc_expiry,
    -- Latest DEC
    dec.operational_rating     AS latest_dec_rating,
    dec.assessment_date        AS latest_dec_date,
    -- Latest Benchmark
    bm.reporting_year          AS latest_benchmark_year,
    bm.eui_kwh_m2             AS latest_eui,
    bm.eui_weather_normalized AS latest_eui_normalized,
    bm.crrem_target_kgco2_m2  AS crrem_target,
    bm.crrem_actual_kgco2_m2  AS crrem_actual,
    bm.stranding_year,
    bm.peer_percentile,
    -- Renewables
    COALESCE(ren.total_capacity_kwp, 0)       AS renewable_capacity_kwp,
    COALESCE(ren.total_generation_kwh, 0)     AS renewable_generation_kwh,
    -- Retrofit
    COALESCE(retro.measure_count, 0)          AS retrofit_measure_count,
    COALESCE(retro.total_savings_kwh, 0)      AS retrofit_total_savings_kwh
FROM pack032_building_assessment.building_profiles bp
LEFT JOIN LATERAL (
    SELECT epc_rating, primary_energy_kwh_m2, co2_emissions_kg_m2, assessment_date, expiry_date
    FROM pack032_building_assessment.epc_certificates e
    WHERE e.building_id = bp.building_id
    ORDER BY e.assessment_date DESC
    LIMIT 1
) epc ON TRUE
LEFT JOIN LATERAL (
    SELECT operational_rating, assessment_date
    FROM pack032_building_assessment.dec_certificates d
    WHERE d.building_id = bp.building_id
    ORDER BY d.assessment_date DESC
    LIMIT 1
) dec ON TRUE
LEFT JOIN LATERAL (
    SELECT reporting_year, eui_kwh_m2, eui_weather_normalized,
           crrem_target_kgco2_m2, crrem_actual_kgco2_m2, stranding_year, peer_percentile
    FROM pack032_building_assessment.building_benchmarks b
    WHERE b.building_id = bp.building_id
    ORDER BY b.reporting_year DESC
    LIMIT 1
) bm ON TRUE
LEFT JOIN LATERAL (
    SELECT SUM(capacity_kwp) AS total_capacity_kwp,
           SUM(annual_generation_kwh) AS total_generation_kwh
    FROM pack032_building_assessment.renewable_systems r
    WHERE r.building_id = bp.building_id
) ren ON TRUE
LEFT JOIN LATERAL (
    SELECT COUNT(*) AS measure_count,
           SUM(energy_savings_kwh) AS total_savings_kwh
    FROM pack032_building_assessment.retrofit_measures rm
    WHERE rm.building_id = bp.building_id
      AND rm.implementation_status != 'rejected'
) retro ON TRUE;

-- =============================================================================
-- View 2: v_portfolio_benchmarks
-- =============================================================================
-- Portfolio-level benchmarking aggregation by building type, country, and year.

CREATE OR REPLACE VIEW pack032_building_assessment.v_portfolio_benchmarks AS
SELECT
    bp.tenant_id,
    bp.building_type,
    bp.country,
    bm.reporting_year,
    COUNT(DISTINCT bp.building_id)           AS building_count,
    SUM(bp.gross_floor_area_m2)              AS total_floor_area_m2,
    AVG(bm.eui_kwh_m2)                      AS avg_eui_kwh_m2,
    AVG(bm.eui_weather_normalized)           AS avg_eui_normalized,
    AVG(bm.energy_star_score)                AS avg_energy_star,
    AVG(bm.crrem_actual_kgco2_m2)           AS avg_carbon_intensity,
    SUM(CASE WHEN bm.crrem_aligned THEN 1 ELSE 0 END) AS crrem_aligned_count,
    AVG(bm.peer_percentile)                  AS avg_peer_percentile,
    MIN(bm.stranding_year)                   AS earliest_stranding_year
FROM pack032_building_assessment.building_profiles bp
INNER JOIN pack032_building_assessment.building_benchmarks bm
    ON bp.building_id = bm.building_id
GROUP BY bp.tenant_id, bp.building_type, bp.country, bm.reporting_year;

-- =============================================================================
-- View 3: v_retrofit_portfolio
-- =============================================================================
-- Portfolio-level retrofit summary with total investment, savings, and payback.

CREATE OR REPLACE VIEW pack032_building_assessment.v_retrofit_portfolio AS
SELECT
    bp.tenant_id,
    bp.building_id,
    bp.building_name,
    bp.building_type,
    bp.epc_rating                             AS current_epc,
    rp.plan_name,
    rp.total_capex_eur,
    rp.total_savings_kwh,
    rp.total_carbon_savings,
    rp.payback_years,
    rp.npv_eur,
    rp.target_epc_rating,
    rp.target_eui,
    COUNT(rm.measure_id)                      AS measure_count,
    SUM(CASE WHEN rm.implementation_status = 'completed' THEN 1 ELSE 0 END) AS completed_measures,
    SUM(CASE WHEN rm.implementation_status = 'in_progress' THEN 1 ELSE 0 END) AS in_progress_measures,
    SUM(CASE WHEN rm.implementation_status = 'proposed' THEN 1 ELSE 0 END) AS proposed_measures
FROM pack032_building_assessment.building_profiles bp
LEFT JOIN pack032_building_assessment.retrofit_plans rp
    ON bp.building_id = rp.building_id
LEFT JOIN pack032_building_assessment.retrofit_measures rm
    ON bp.building_id = rm.building_id
GROUP BY bp.tenant_id, bp.building_id, bp.building_name, bp.building_type,
         bp.epc_rating, rp.plan_name, rp.total_capex_eur, rp.total_savings_kwh,
         rp.total_carbon_savings, rp.payback_years, rp.npv_eur,
         rp.target_epc_rating, rp.target_eui;

-- =============================================================================
-- View 4: v_compliance_dashboard
-- =============================================================================
-- Compliance overview per building with counts of compliant/non-compliant items
-- and upcoming deadlines.

CREATE OR REPLACE VIEW pack032_building_assessment.v_compliance_dashboard AS
SELECT
    bp.tenant_id,
    bp.building_id,
    bp.building_name,
    bp.building_type,
    bp.country,
    bp.epc_rating,
    COUNT(cr.record_id)                                        AS total_requirements,
    SUM(CASE WHEN cr.compliant THEN 1 ELSE 0 END)             AS compliant_count,
    SUM(CASE WHEN NOT cr.compliant THEN 1 ELSE 0 END)         AS non_compliant_count,
    ROUND(
        100.0 * SUM(CASE WHEN cr.compliant THEN 1 ELSE 0 END)
        / NULLIF(COUNT(cr.record_id), 0), 1
    )                                                          AS compliance_pct,
    MIN(CASE WHEN NOT cr.compliant THEN cr.deadline END)       AS nearest_non_compliant_deadline,
    ARRAY_AGG(DISTINCT cr.regulation) FILTER (WHERE NOT cr.compliant) AS non_compliant_regulations
FROM pack032_building_assessment.building_profiles bp
LEFT JOIN pack032_building_assessment.compliance_records cr
    ON bp.building_id = cr.building_id
GROUP BY bp.tenant_id, bp.building_id, bp.building_name, bp.building_type,
         bp.country, bp.epc_rating;

-- ---------------------------------------------------------------------------
-- Grants on views
-- ---------------------------------------------------------------------------
GRANT SELECT ON pack032_building_assessment.v_building_performance_summary TO PUBLIC;
GRANT SELECT ON pack032_building_assessment.v_portfolio_benchmarks TO PUBLIC;
GRANT SELECT ON pack032_building_assessment.v_retrofit_portfolio TO PUBLIC;
GRANT SELECT ON pack032_building_assessment.v_compliance_dashboard TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack032_building_assessment.pack032_audit_trail IS
    'Audit trail for all PACK-032 entity changes with old/new values for regulatory compliance and data governance.';

COMMENT ON TABLE pack032_building_assessment.compliance_records IS
    'Regulatory compliance tracking for building energy performance (MEES, EPBD, Part L, etc.) with deadlines and penalties.';

COMMENT ON VIEW pack032_building_assessment.v_building_performance_summary IS
    'Comprehensive building performance summary joining profiles, latest EPC/DEC, benchmarks, renewables, and retrofit measures.';

COMMENT ON VIEW pack032_building_assessment.v_portfolio_benchmarks IS
    'Portfolio-level benchmarking aggregation by building type, country, and reporting year with CRREM alignment.';

COMMENT ON VIEW pack032_building_assessment.v_retrofit_portfolio IS
    'Portfolio-level retrofit summary with total investment, savings, payback, and measure implementation status.';

COMMENT ON VIEW pack032_building_assessment.v_compliance_dashboard IS
    'Compliance overview per building with compliant/non-compliant counts, compliance percentage, and upcoming deadlines.';

COMMENT ON COLUMN pack032_building_assessment.pack032_audit_trail.entry_id IS
    'Unique identifier for the audit trail entry.';
COMMENT ON COLUMN pack032_building_assessment.pack032_audit_trail.action IS
    'Type of action performed (CREATE, UPDATE, DELETE, ARCHIVE, RESTORE, APPROVE, REJECT).';
COMMENT ON COLUMN pack032_building_assessment.pack032_audit_trail.old_values IS
    'JSON snapshot of entity state before the change.';
COMMENT ON COLUMN pack032_building_assessment.pack032_audit_trail.new_values IS
    'JSON snapshot of entity state after the change.';
COMMENT ON COLUMN pack032_building_assessment.compliance_records.regulation IS
    'Regulatory framework (e.g., MEES, EPBD, Building Regs Part L, SECR).';
COMMENT ON COLUMN pack032_building_assessment.compliance_records.compliant IS
    'Whether the building currently meets this regulatory requirement.';
COMMENT ON COLUMN pack032_building_assessment.compliance_records.deadline IS
    'Date by which compliance must be achieved.';
COMMENT ON COLUMN pack032_building_assessment.compliance_records.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
