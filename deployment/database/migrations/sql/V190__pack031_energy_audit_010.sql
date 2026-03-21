-- =============================================================================
-- V190: PACK-031 Industrial Energy Audit - Process Mapping, Audit Trails, Views
-- =============================================================================
-- Pack:         PACK-031 (Industrial Energy Audit Pack)
-- Migration:    010 of 010
-- Date:         March 2026
--
-- Process energy mapping (Sankey diagram support), energy flow tracking
-- between process nodes, pack-level audit trail, and consolidated views
-- for facility energy summaries, savings portfolios, equipment efficiency
-- gaps, and compliance dashboards.
--
-- Tables (3):
--   1. pack031_energy_audit.process_nodes
--   2. pack031_energy_audit.energy_flows
--   3. pack031_energy_audit.pack031_audit_trail
--
-- Views (4):
--   1. pack031_energy_audit.v_facility_energy_summary
--   2. pack031_energy_audit.v_savings_portfolio
--   3. pack031_energy_audit.v_equipment_efficiency_gaps
--   4. pack031_energy_audit.v_compliance_dashboard
--
-- Previous: V189__pack031_energy_audit_009.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack031_energy_audit.process_nodes
-- =============================================================================
-- Process nodes representing energy-consuming or energy-transforming
-- stages in a production line for Sankey diagram and energy flow analysis.

CREATE TABLE pack031_energy_audit.process_nodes (
    node_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    production_line         VARCHAR(255),
    name                    VARCHAR(255)    NOT NULL,
    process_type            VARCHAR(100),
    input_energy_kwh        NUMERIC(14,4),
    output_energy_kwh       NUMERIC(14,4),
    losses_kwh              NUMERIC(14,4),
    efficiency_pct          NUMERIC(5,2),
    operating_hours         INTEGER,
    carrier_type            VARCHAR(100),
    equipment_id            UUID            REFERENCES pack031_energy_audit.equipment(equipment_id) ON DELETE SET NULL,
    sequence_order          INTEGER,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_node_input CHECK (
        input_energy_kwh IS NULL OR input_energy_kwh >= 0
    ),
    CONSTRAINT chk_p031_node_output CHECK (
        output_energy_kwh IS NULL OR output_energy_kwh >= 0
    ),
    CONSTRAINT chk_p031_node_losses CHECK (
        losses_kwh IS NULL OR losses_kwh >= 0
    ),
    CONSTRAINT chk_p031_node_eff CHECK (
        efficiency_pct IS NULL OR (efficiency_pct >= 0 AND efficiency_pct <= 100)
    ),
    CONSTRAINT chk_p031_node_type CHECK (
        process_type IS NULL OR process_type IN (
            'INPUT', 'CONVERSION', 'DISTRIBUTION', 'END_USE', 'WASTE',
            'STORAGE', 'GENERATION', 'RECOVERY', 'TRANSFORMER', 'OTHER'
        )
    )
);

-- Indexes
CREATE INDEX idx_p031_node_facility    ON pack031_energy_audit.process_nodes(facility_id);
CREATE INDEX idx_p031_node_tenant      ON pack031_energy_audit.process_nodes(tenant_id);
CREATE INDEX idx_p031_node_line        ON pack031_energy_audit.process_nodes(production_line);
CREATE INDEX idx_p031_node_type        ON pack031_energy_audit.process_nodes(process_type);
CREATE INDEX idx_p031_node_equip       ON pack031_energy_audit.process_nodes(equipment_id);

-- =============================================================================
-- Table 2: pack031_energy_audit.energy_flows
-- =============================================================================
-- Directed energy flows between process nodes, supporting Sankey diagram
-- generation with energy type and temperature tracking.

CREATE TABLE pack031_energy_audit.energy_flows (
    flow_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    source_node             UUID            NOT NULL REFERENCES pack031_energy_audit.process_nodes(node_id) ON DELETE CASCADE,
    target_node             UUID            NOT NULL REFERENCES pack031_energy_audit.process_nodes(node_id) ON DELETE CASCADE,
    energy_kwh              NUMERIC(14,4)   NOT NULL,
    energy_type             VARCHAR(50)     NOT NULL,
    temperature_c           NUMERIC(8,2),
    flow_rate_kg_s          NUMERIC(10,4),
    medium                  VARCHAR(50),
    label                   VARCHAR(255),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_flow_energy CHECK (energy_kwh >= 0),
    CONSTRAINT chk_p031_flow_type CHECK (
        energy_type IN ('ELECTRICITY', 'NATURAL_GAS', 'STEAM', 'HOT_WATER',
                        'CHILLED_WATER', 'COMPRESSED_AIR', 'FUEL', 'HEAT',
                        'MECHANICAL', 'RADIATION', 'WASTE_HEAT', 'OTHER')
    ),
    CONSTRAINT chk_p031_flow_no_self_loop CHECK (source_node != target_node)
);

-- Indexes
CREATE INDEX idx_p031_flow_facility    ON pack031_energy_audit.energy_flows(facility_id);
CREATE INDEX idx_p031_flow_tenant      ON pack031_energy_audit.energy_flows(tenant_id);
CREATE INDEX idx_p031_flow_source      ON pack031_energy_audit.energy_flows(source_node);
CREATE INDEX idx_p031_flow_target      ON pack031_energy_audit.energy_flows(target_node);
CREATE INDEX idx_p031_flow_type        ON pack031_energy_audit.energy_flows(energy_type);
CREATE INDEX idx_p031_flow_energy      ON pack031_energy_audit.energy_flows(energy_kwh DESC);

-- =============================================================================
-- Table 3: pack031_energy_audit.pack031_audit_trail
-- =============================================================================
-- Pack-level audit trail logging all significant actions across
-- PACK-031 entities for compliance and provenance tracking.

CREATE TABLE pack031_energy_audit.pack031_audit_trail (
    audit_trail_id          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID,
    tenant_id               UUID,
    action                  VARCHAR(50)     NOT NULL,
    entity_type             VARCHAR(100)    NOT NULL,
    entity_id               UUID,
    actor                   TEXT            NOT NULL,
    actor_role              VARCHAR(50),
    ip_address              VARCHAR(45),
    old_values              JSONB,
    new_values              JSONB,
    details                 JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_trail_action CHECK (
        action IN ('CREATE', 'UPDATE', 'DELETE', 'READ', 'APPROVE', 'REJECT',
                   'SUBMIT', 'EXPORT', 'IMPORT', 'CALCULATE', 'VERIFY',
                   'ARCHIVE', 'RESTORE', 'LOCK', 'UNLOCK')
    )
);

-- Indexes
CREATE INDEX idx_p031_trail_facility   ON pack031_energy_audit.pack031_audit_trail(facility_id);
CREATE INDEX idx_p031_trail_tenant     ON pack031_energy_audit.pack031_audit_trail(tenant_id);
CREATE INDEX idx_p031_trail_action     ON pack031_energy_audit.pack031_audit_trail(action);
CREATE INDEX idx_p031_trail_entity     ON pack031_energy_audit.pack031_audit_trail(entity_type, entity_id);
CREATE INDEX idx_p031_trail_actor      ON pack031_energy_audit.pack031_audit_trail(actor);
CREATE INDEX idx_p031_trail_created    ON pack031_energy_audit.pack031_audit_trail(created_at DESC);
CREATE INDEX idx_p031_trail_details    ON pack031_energy_audit.pack031_audit_trail USING GIN(details);

-- =============================================================================
-- View 1: v_facility_energy_summary
-- =============================================================================
-- Consolidated facility energy overview with total consumption, cost,
-- latest EnPI, and energy rating.

CREATE OR REPLACE VIEW pack031_energy_audit.v_facility_energy_summary AS
SELECT
    f.facility_id,
    f.org_id,
    f.tenant_id,
    f.name AS facility_name,
    f.sector,
    f.sub_sector,
    f.country,
    f.eed_obligation,
    f.iso_50001_certified,
    f.eu_ets_covered,
    a.audit_id AS latest_audit_id,
    a.audit_date AS latest_audit_date,
    a.total_consumption_kwh,
    a.total_cost_eur,
    a.quality_score AS audit_quality_score,
    a.en16247_compliant,
    enpi.enpi_type AS latest_enpi_type,
    enpi.enpi_value AS latest_enpi_value,
    enpi.improvement_pct AS latest_improvement_pct,
    b.energy_rating,
    b.percentile_rank,
    b.gap_to_best_pct
FROM pack031_energy_audit.energy_audit_facilities f
LEFT JOIN LATERAL (
    SELECT audit_id, audit_date, total_consumption_kwh, total_cost_eur,
           quality_score, en16247_compliant
    FROM pack031_energy_audit.energy_audits
    WHERE facility_id = f.facility_id AND status = 'completed'
    ORDER BY audit_date DESC
    LIMIT 1
) a ON TRUE
LEFT JOIN LATERAL (
    SELECT enpi_type, enpi_value, improvement_pct
    FROM pack031_energy_audit.enpi_records
    WHERE facility_id = f.facility_id
    ORDER BY period_end DESC
    LIMIT 1
) enpi ON TRUE
LEFT JOIN LATERAL (
    SELECT energy_rating, percentile_rank, gap_to_best_pct
    FROM pack031_energy_audit.energy_benchmarks
    WHERE facility_id = f.facility_id
    ORDER BY period DESC
    LIMIT 1
) b ON TRUE;

-- =============================================================================
-- View 2: v_savings_portfolio
-- =============================================================================
-- Aggregated savings portfolio per facility showing measure counts,
-- total expected savings, total implementation cost, and average payback.

CREATE OR REPLACE VIEW pack031_energy_audit.v_savings_portfolio AS
SELECT
    f.facility_id,
    f.org_id,
    f.tenant_id,
    f.name AS facility_name,
    COUNT(m.measure_id) AS total_measures,
    COUNT(m.measure_id) FILTER (WHERE m.status = 'implemented') AS implemented_measures,
    COUNT(m.measure_id) FILTER (WHERE m.status IN ('proposed', 'evaluated', 'approved')) AS pending_measures,
    COALESCE(SUM(m.expected_savings_kwh), 0) AS total_expected_savings_kwh,
    COALESCE(SUM(m.expected_savings_kwh) FILTER (WHERE m.status = 'implemented'), 0) AS implemented_savings_kwh,
    COALESCE(SUM(m.implementation_cost_eur), 0) AS total_implementation_cost_eur,
    COALESCE(SUM(m.co2_savings_tonnes), 0) AS total_co2_savings_tonnes,
    ROUND(AVG(fa.simple_payback_years) FILTER (WHERE fa.simple_payback_years IS NOT NULL), 2) AS avg_payback_years,
    ROUND(AVG(fa.irr_pct) FILTER (WHERE fa.irr_pct IS NOT NULL), 2) AS avg_irr_pct,
    COALESCE(SUM(fa.npv_eur), 0) AS total_npv_eur
FROM pack031_energy_audit.energy_audit_facilities f
LEFT JOIN pack031_energy_audit.energy_savings_measures m ON m.facility_id = f.facility_id
LEFT JOIN pack031_energy_audit.financial_analyses fa ON fa.measure_id = m.measure_id
GROUP BY f.facility_id, f.org_id, f.tenant_id, f.name;

-- =============================================================================
-- View 3: v_equipment_efficiency_gaps
-- =============================================================================
-- Equipment efficiency analysis showing current performance vs best
-- practice for motors, pumps, and HVAC systems.

CREATE OR REPLACE VIEW pack031_energy_audit.v_equipment_efficiency_gaps AS
SELECT
    e.equipment_id,
    e.facility_id,
    e.tenant_id,
    e.name AS equipment_name,
    e.type AS equipment_type,
    e.rated_power_kw,
    e.operating_hours,
    e.load_factor_pct,
    e.year_installed,
    e.condition_rating,
    -- Motor-specific
    md.efficiency_class,
    md.has_vsd AS motor_has_vsd,
    md.actual_load_pct AS motor_load_pct,
    md.annual_energy_kwh AS motor_annual_kwh,
    md.replacement_candidate AS motor_replacement_candidate,
    md.savings_with_ie4_kwh,
    -- Pump-specific
    pd.efficiency_pct AS pump_efficiency_pct,
    pd.operating_pct_of_bep AS pump_bep_pct,
    pd.throttle_controlled AS pump_throttle_controlled,
    -- HVAC-specific
    hd.cop AS hvac_cop,
    hd.eer AS hvac_eer,
    hd.refrigerant AS hvac_refrigerant,
    hd.refrigerant_gwp AS hvac_refrigerant_gwp
FROM pack031_energy_audit.equipment e
LEFT JOIN pack031_energy_audit.motor_data md ON md.equipment_id = e.equipment_id
LEFT JOIN pack031_energy_audit.pump_data pd ON pd.equipment_id = e.equipment_id
LEFT JOIN pack031_energy_audit.hvac_data hd ON hd.equipment_id = e.equipment_id
WHERE e.type IN ('MOTOR', 'PUMP', 'FAN', 'HVAC', 'CHILLER', 'COMPRESSOR');

-- =============================================================================
-- View 4: v_compliance_dashboard
-- =============================================================================
-- Compliance overview per facility covering EED status, ISO 50001
-- certification, next audit deadlines, and BAT-AEL compliance gaps.

CREATE OR REPLACE VIEW pack031_energy_audit.v_compliance_dashboard AS
SELECT
    f.facility_id,
    f.org_id,
    f.tenant_id,
    f.name AS facility_name,
    f.sector,
    f.country,
    -- EED compliance
    eed.obligation_applies AS eed_obligation,
    eed.compliance_status AS eed_status,
    eed.last_audit_date AS eed_last_audit,
    eed.next_audit_due AS eed_next_due,
    eed.iso50001_exempt AS eed_iso_exempt,
    eed.days_until_due AS eed_days_until_due,
    -- ISO 50001
    iso.certification_status AS iso_50001_status,
    iso.expiry_date AS iso_50001_expiry,
    iso.enms_maturity_level AS iso_50001_maturity,
    iso.next_surveillance_due AS iso_50001_next_surveillance,
    iso.continual_improvement_pct AS iso_50001_improvement_pct,
    -- BAT-AEL summary
    bat_summary.total_comparisons AS bat_total_comparisons,
    bat_summary.compliant_count AS bat_compliant_count,
    bat_summary.non_compliant_count AS bat_non_compliant_count,
    bat_summary.worst_gap_pct AS bat_worst_gap_pct
FROM pack031_energy_audit.energy_audit_facilities f
LEFT JOIN pack031_energy_audit.eed_compliance eed ON eed.facility_id = f.facility_id
LEFT JOIN pack031_energy_audit.iso_50001_records iso ON iso.facility_id = f.facility_id
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) AS total_comparisons,
        COUNT(*) FILTER (WHERE compliance_status = 'compliant') AS compliant_count,
        COUNT(*) FILTER (WHERE compliance_status = 'non_compliant') AS non_compliant_count,
        MAX(gap_pct) AS worst_gap_pct
    FROM pack031_energy_audit.bat_ael_comparisons
    WHERE facility_id = f.facility_id
) bat_summary ON TRUE;

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack031_energy_audit.process_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.energy_flows ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.pack031_audit_trail ENABLE ROW LEVEL SECURITY;

CREATE POLICY p031_node_tenant_isolation ON pack031_energy_audit.process_nodes
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_node_service_bypass ON pack031_energy_audit.process_nodes
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_flow_tenant_isolation ON pack031_energy_audit.energy_flows
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_flow_service_bypass ON pack031_energy_audit.energy_flows
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_trail_tenant_isolation ON pack031_energy_audit.pack031_audit_trail
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_trail_service_bypass ON pack031_energy_audit.pack031_audit_trail
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.process_nodes TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.energy_flows TO PUBLIC;
GRANT SELECT, INSERT ON pack031_energy_audit.pack031_audit_trail TO PUBLIC;
GRANT SELECT ON pack031_energy_audit.v_facility_energy_summary TO PUBLIC;
GRANT SELECT ON pack031_energy_audit.v_savings_portfolio TO PUBLIC;
GRANT SELECT ON pack031_energy_audit.v_equipment_efficiency_gaps TO PUBLIC;
GRANT SELECT ON pack031_energy_audit.v_compliance_dashboard TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack031_energy_audit.process_nodes IS
    'Process nodes representing energy-consuming/transforming stages for Sankey diagram and energy flow analysis.';
COMMENT ON TABLE pack031_energy_audit.energy_flows IS
    'Directed energy flows between process nodes supporting Sankey diagram generation.';
COMMENT ON TABLE pack031_energy_audit.pack031_audit_trail IS
    'Pack-level audit trail logging all significant actions across PACK-031 entities.';

COMMENT ON VIEW pack031_energy_audit.v_facility_energy_summary IS
    'Consolidated facility energy overview: consumption, cost, EnPI, and energy rating from latest audit.';
COMMENT ON VIEW pack031_energy_audit.v_savings_portfolio IS
    'Aggregated savings portfolio per facility: measure counts, total savings, total cost, average payback.';
COMMENT ON VIEW pack031_energy_audit.v_equipment_efficiency_gaps IS
    'Equipment efficiency analysis showing current performance vs best practice for motors, pumps, HVAC.';
COMMENT ON VIEW pack031_energy_audit.v_compliance_dashboard IS
    'Compliance dashboard: EED status, ISO 50001 certification, next deadlines, BAT-AEL compliance gaps.';
