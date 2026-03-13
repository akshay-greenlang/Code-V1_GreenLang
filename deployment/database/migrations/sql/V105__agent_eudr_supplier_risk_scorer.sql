-- ============================================================================
-- V105: AGENT-EUDR-017 Supplier Risk Scorer Agent
-- ============================================================================
-- Creates tables for supplier risk assessment, due diligence record management,
-- documentation profiling, certification tracking, geographic sourcing analysis,
-- supplier network mapping, monitoring configuration, alert generation, risk
-- reporting, and comprehensive audit trails.
--
-- Tables: 12 (9 regular + 3 hypertables)
-- Hypertables: gl_eudr_srs_supplier_assessments, gl_eudr_srs_alerts,
--              gl_eudr_srs_audit_log
-- Continuous Aggregates: 2 (hourly_assessment_stats + hourly_alert_stats)
-- Retention Policies: 3 (5 years for assessments, 3 years for alerts,
--                        5 years for audit logs)
-- Indexes: ~120
--
-- Dependencies: TimescaleDB extension (V002)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V105: Creating AGENT-EUDR-017 Supplier Risk Scorer tables...';

-- ============================================================================
-- 1. gl_eudr_srs_supplier_assessments — Supplier risk assessments (hypertable, monthly)
-- ============================================================================
RAISE NOTICE 'V105 [1/12]: Creating gl_eudr_srs_supplier_assessments (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_srs_supplier_assessments (
    id                      UUID            DEFAULT gen_random_uuid(),
    supplier_id             UUID            NOT NULL,
    risk_score              NUMERIC(5,2)    NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
    risk_level              VARCHAR(20)     NOT NULL,
        -- 'low', 'standard', 'high', 'critical'
    factor_scores           JSONB,
        -- { "country_risk": 25.5, "certification": 15.2, "documentation": 20.8,
        --   "geographic_concentration": 12.3, "network_complexity": 18.7,
        --   "historical_performance": 7.5 }
    confidence              NUMERIC(5,4)    CHECK (confidence >= 0 AND confidence <= 1),
    trend                   VARCHAR(20),
        -- 'improving', 'stable', 'deteriorating', 'critical'
    data_completeness       NUMERIC(5,4)    CHECK (data_completeness >= 0 AND data_completeness <= 1),
    assessed_by             VARCHAR(255),
    notes                   TEXT,
    metadata                JSONB,
    tenant_id               UUID            NOT NULL,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    assessed_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, assessed_at)
);

SELECT create_hypertable(
    'gl_eudr_srs_supplier_assessments',
    'assessed_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sa_supplier_id ON gl_eudr_srs_supplier_assessments (supplier_id, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sa_risk_score ON gl_eudr_srs_supplier_assessments (risk_score, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sa_risk_level ON gl_eudr_srs_supplier_assessments (risk_level, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sa_confidence ON gl_eudr_srs_supplier_assessments (confidence, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sa_trend ON gl_eudr_srs_supplier_assessments (trend, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sa_completeness ON gl_eudr_srs_supplier_assessments (data_completeness, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sa_assessed_by ON gl_eudr_srs_supplier_assessments (assessed_by, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sa_tenant ON gl_eudr_srs_supplier_assessments (tenant_id, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sa_created ON gl_eudr_srs_supplier_assessments (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sa_updated ON gl_eudr_srs_supplier_assessments (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sa_supplier_level ON gl_eudr_srs_supplier_assessments (supplier_id, risk_level, assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sa_factor_scores ON gl_eudr_srs_supplier_assessments USING GIN (factor_scores);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sa_metadata ON gl_eudr_srs_supplier_assessments USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_srs_supplier_assessments IS 'Supplier risk assessments with multi-factor scoring and confidence metrics';
COMMENT ON COLUMN gl_eudr_srs_supplier_assessments.factor_scores IS 'Breakdown of risk factors contributing to overall score';
COMMENT ON COLUMN gl_eudr_srs_supplier_assessments.confidence IS 'Assessment confidence level (0-1) based on data quality';
COMMENT ON COLUMN gl_eudr_srs_supplier_assessments.data_completeness IS 'Proportion of required data fields available (0-1)';


-- ============================================================================
-- 2. gl_eudr_srs_due_diligence_records — Due diligence activities and results
-- ============================================================================
RAISE NOTICE 'V105 [2/12]: Creating gl_eudr_srs_due_diligence_records...';

CREATE TABLE IF NOT EXISTS gl_eudr_srs_due_diligence_records (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id                 UUID            NOT NULL,
    dd_level                    VARCHAR(20)     NOT NULL,
        -- 'simplified', 'standard', 'enhanced', 'continuous'
    status                      VARCHAR(20)     NOT NULL,
        -- 'planned', 'in_progress', 'completed', 'failed', 'overdue'
    activities                  JSONB,
        -- [{ "activity": "document_review", "status": "completed", "date": "2026-01-15" },
        --  { "activity": "site_visit", "status": "scheduled", "date": "2026-02-20" }]
    non_conformances            JSONB,
        -- [{ "nc_id": "NC-001", "severity": "major", "description": "...",
        --    "finding_date": "2026-01-20", "capa_id": "CAPA-001" }]
    corrective_actions          JSONB,
        -- [{ "capa_id": "CAPA-001", "description": "...", "due_date": "2026-03-15",
        --    "status": "open", "responsible_party": "Supplier X" }]
    cost_eur                    NUMERIC(12,2),
    completed_at                TIMESTAMPTZ,
    next_review_date            DATE,
    assessed_by                 VARCHAR(255),
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dd_supplier_id ON gl_eudr_srs_due_diligence_records (supplier_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dd_level ON gl_eudr_srs_due_diligence_records (dd_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dd_status ON gl_eudr_srs_due_diligence_records (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dd_completed ON gl_eudr_srs_due_diligence_records (completed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dd_next_review ON gl_eudr_srs_due_diligence_records (next_review_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dd_cost ON gl_eudr_srs_due_diligence_records (cost_eur);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dd_tenant ON gl_eudr_srs_due_diligence_records (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dd_created ON gl_eudr_srs_due_diligence_records (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dd_updated ON gl_eudr_srs_due_diligence_records (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dd_supplier_status ON gl_eudr_srs_due_diligence_records (supplier_id, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dd_activities ON gl_eudr_srs_due_diligence_records USING GIN (activities);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dd_non_conformances ON gl_eudr_srs_due_diligence_records USING GIN (non_conformances);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dd_capa ON gl_eudr_srs_due_diligence_records USING GIN (corrective_actions);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dd_metadata ON gl_eudr_srs_due_diligence_records USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_srs_due_diligence_records IS 'Due diligence activities, findings, and corrective actions for suppliers';
COMMENT ON COLUMN gl_eudr_srs_due_diligence_records.dd_level IS 'Level of due diligence based on risk classification';
COMMENT ON COLUMN gl_eudr_srs_due_diligence_records.non_conformances IS 'Identified non-conformances and findings from audits';
COMMENT ON COLUMN gl_eudr_srs_due_diligence_records.corrective_actions IS 'CAPA (Corrective and Preventive Actions) tracking';


-- ============================================================================
-- 3. gl_eudr_srs_documentation_profiles — Supplier documentation quality profiles
-- ============================================================================
RAISE NOTICE 'V105 [3/12]: Creating gl_eudr_srs_documentation_profiles...';

CREATE TABLE IF NOT EXISTS gl_eudr_srs_documentation_profiles (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id                 UUID            NOT NULL,
    documents                   JSONB,
        -- [{ "doc_type": "eudr_ds", "status": "valid", "expiry": "2027-01-15",
        --    "completeness": 0.95, "quality_score": 88.5, "verified": true },
        --  { "doc_type": "certificate", "status": "expired", "expiry": "2025-12-31" }]
    completeness_score          NUMERIC(5,2)    CHECK (completeness_score >= 0 AND completeness_score <= 100),
    quality_score               NUMERIC(5,2)    CHECK (quality_score >= 0 AND quality_score <= 100),
    gaps                        JSONB,
        -- [{ "required_doc": "fsc_certificate", "status": "missing", "priority": "high" },
        --  { "required_doc": "eudr_geo_data", "status": "incomplete", "missing_fields": [...] }]
    last_reviewed_at            TIMESTAMPTZ,
    next_review_date            DATE,
    reviewed_by                 VARCHAR(255),
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dp_supplier_id ON gl_eudr_srs_documentation_profiles (supplier_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dp_completeness ON gl_eudr_srs_documentation_profiles (completeness_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dp_quality ON gl_eudr_srs_documentation_profiles (quality_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dp_last_reviewed ON gl_eudr_srs_documentation_profiles (last_reviewed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dp_next_review ON gl_eudr_srs_documentation_profiles (next_review_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dp_tenant ON gl_eudr_srs_documentation_profiles (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dp_created ON gl_eudr_srs_documentation_profiles (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dp_updated ON gl_eudr_srs_documentation_profiles (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dp_documents ON gl_eudr_srs_documentation_profiles USING GIN (documents);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dp_gaps ON gl_eudr_srs_documentation_profiles USING GIN (gaps);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_dp_metadata ON gl_eudr_srs_documentation_profiles USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_srs_documentation_profiles IS 'Documentation completeness and quality assessment for suppliers';
COMMENT ON COLUMN gl_eudr_srs_documentation_profiles.documents IS 'Inventory of submitted documents with status and quality metrics';
COMMENT ON COLUMN gl_eudr_srs_documentation_profiles.gaps IS 'Identified gaps in required documentation';


-- ============================================================================
-- 4. gl_eudr_srs_certification_records — Certification scheme tracking
-- ============================================================================
RAISE NOTICE 'V105 [4/12]: Creating gl_eudr_srs_certification_records...';

CREATE TABLE IF NOT EXISTS gl_eudr_srs_certification_records (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id                 UUID            NOT NULL,
    scheme                      VARCHAR(30)     NOT NULL,
        -- 'FSC', 'PEFC', 'RSPO', 'Rainforest_Alliance', 'Fairtrade', 'UTZ', 'ISO14001', 'EUTR'
    certificate_number          VARCHAR(100)    NOT NULL,
    status                      VARCHAR(20)     NOT NULL,
        -- 'valid', 'expired', 'suspended', 'withdrawn', 'pending_renewal'
    scope                       JSONB,
        -- { "commodities": ["cocoa", "coffee"], "regions": ["Ghana", "Ivory Coast"],
        --   "operations": ["farming", "processing"], "volume_covered_pct": 85.5 }
    valid_from                  DATE            NOT NULL,
    valid_until                 DATE            NOT NULL,
    chain_of_custody            BOOLEAN         DEFAULT FALSE,
    issuing_body                VARCHAR(255),
    audit_date                  DATE,
    next_surveillance_date      DATE,
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_cr_supplier_id ON gl_eudr_srs_certification_records (supplier_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_cr_scheme ON gl_eudr_srs_certification_records (scheme);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_cr_cert_number ON gl_eudr_srs_certification_records (certificate_number);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_cr_status ON gl_eudr_srs_certification_records (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_cr_valid_from ON gl_eudr_srs_certification_records (valid_from);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_cr_valid_until ON gl_eudr_srs_certification_records (valid_until);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_cr_chain_custody ON gl_eudr_srs_certification_records (chain_of_custody);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_cr_next_surveillance ON gl_eudr_srs_certification_records (next_surveillance_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_cr_tenant ON gl_eudr_srs_certification_records (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_cr_created ON gl_eudr_srs_certification_records (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_cr_updated ON gl_eudr_srs_certification_records (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_cr_supplier_scheme ON gl_eudr_srs_certification_records (supplier_id, scheme);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_cr_scope ON gl_eudr_srs_certification_records USING GIN (scope);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_cr_metadata ON gl_eudr_srs_certification_records USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_srs_certification_records IS 'Tracking of supplier certifications and chain-of-custody status';
COMMENT ON COLUMN gl_eudr_srs_certification_records.chain_of_custody IS 'Indicates if certificate includes chain-of-custody certification';
COMMENT ON COLUMN gl_eudr_srs_certification_records.scope IS 'Certification scope including commodities, regions, and coverage';


-- ============================================================================
-- 5. gl_eudr_srs_geographic_sourcing — Geographic sourcing risk profiles
-- ============================================================================
RAISE NOTICE 'V105 [5/12]: Creating gl_eudr_srs_geographic_sourcing...';

CREATE TABLE IF NOT EXISTS gl_eudr_srs_geographic_sourcing (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id                 UUID            NOT NULL,
    sourcing_locations          JSONB,
        -- [{ "country": "BR", "region": "Para", "latitude": -3.12, "longitude": -52.15,
        --    "commodity": "cattle", "volume_pct": 45.0, "risk_score": 78.5,
        --    "deforestation_alerts": 3, "protected_area_proximity_km": 15.2 }]
    risk_zones                  JSONB,
        -- [{ "zone_id": "RZ-001", "severity": "high", "area_km2": 450.5,
        --    "drivers": ["cattle_ranching"], "mitigation_plan": "..." }]
    concentration_index         NUMERIC(5,4)    CHECK (concentration_index >= 0 AND concentration_index <= 1),
        -- Herfindahl index: 0 (diversified) to 1 (single source)
    country_risk_score          NUMERIC(5,2)    CHECK (country_risk_score >= 0 AND country_risk_score <= 100),
    proximity_alerts            JSONB,
        -- [{ "alert_type": "protected_area", "distance_km": 12.5, "severity": "medium" },
        --  { "alert_type": "indigenous_territory", "distance_km": 8.3, "severity": "high" }]
    last_analyzed_at            TIMESTAMPTZ,
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_gs_supplier_id ON gl_eudr_srs_geographic_sourcing (supplier_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_gs_concentration ON gl_eudr_srs_geographic_sourcing (concentration_index);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_gs_country_risk ON gl_eudr_srs_geographic_sourcing (country_risk_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_gs_last_analyzed ON gl_eudr_srs_geographic_sourcing (last_analyzed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_gs_tenant ON gl_eudr_srs_geographic_sourcing (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_gs_created ON gl_eudr_srs_geographic_sourcing (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_gs_updated ON gl_eudr_srs_geographic_sourcing (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_gs_locations ON gl_eudr_srs_geographic_sourcing USING GIN (sourcing_locations);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_gs_risk_zones ON gl_eudr_srs_geographic_sourcing USING GIN (risk_zones);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_gs_proximity ON gl_eudr_srs_geographic_sourcing USING GIN (proximity_alerts);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_gs_metadata ON gl_eudr_srs_geographic_sourcing USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_srs_geographic_sourcing IS 'Geographic sourcing patterns and associated risk profiles';
COMMENT ON COLUMN gl_eudr_srs_geographic_sourcing.concentration_index IS 'Geographic diversification index (Herfindahl)';
COMMENT ON COLUMN gl_eudr_srs_geographic_sourcing.proximity_alerts IS 'Alerts for proximity to protected areas or indigenous territories';


-- ============================================================================
-- 6. gl_eudr_srs_supplier_networks — Supplier network mapping and analysis
-- ============================================================================
RAISE NOTICE 'V105 [6/12]: Creating gl_eudr_srs_supplier_networks...';

CREATE TABLE IF NOT EXISTS gl_eudr_srs_supplier_networks (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id                 UUID            NOT NULL,
    sub_suppliers               JSONB,
        -- [{ "sub_supplier_id": "uuid-123", "name": "SubSupplier A", "tier": 2,
        --    "risk_score": 65.5, "relationship_type": "direct", "volume_pct": 35.0 }]
    intermediaries              JSONB,
        -- [{ "intermediary_id": "uuid-456", "name": "Trader X", "role": "broker",
        --    "countries": ["BR", "ID"], "risk_flags": ["opacity", "high_volume"] }]
    network_depth               INTEGER         CHECK (network_depth >= 1 AND network_depth <= 10),
        -- Maximum tier depth in supply chain
    risk_propagation_score      NUMERIC(5,2)    CHECK (risk_propagation_score >= 0 AND risk_propagation_score <= 100),
        -- Risk score accounting for upstream propagation
    circular_detected           BOOLEAN         DEFAULT FALSE,
        -- Flag for circular supply chain relationships
    last_mapped_at              TIMESTAMPTZ,
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sn_supplier_id ON gl_eudr_srs_supplier_networks (supplier_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sn_network_depth ON gl_eudr_srs_supplier_networks (network_depth);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sn_risk_propagation ON gl_eudr_srs_supplier_networks (risk_propagation_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sn_circular ON gl_eudr_srs_supplier_networks (circular_detected);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sn_last_mapped ON gl_eudr_srs_supplier_networks (last_mapped_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sn_tenant ON gl_eudr_srs_supplier_networks (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sn_created ON gl_eudr_srs_supplier_networks (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sn_updated ON gl_eudr_srs_supplier_networks (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sn_sub_suppliers ON gl_eudr_srs_supplier_networks USING GIN (sub_suppliers);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sn_intermediaries ON gl_eudr_srs_supplier_networks USING GIN (intermediaries);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sn_metadata ON gl_eudr_srs_supplier_networks USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_srs_supplier_networks IS 'Supplier network topology and risk propagation analysis';
COMMENT ON COLUMN gl_eudr_srs_supplier_networks.network_depth IS 'Maximum tier depth in the supply chain network';
COMMENT ON COLUMN gl_eudr_srs_supplier_networks.risk_propagation_score IS 'Aggregate risk score including upstream suppliers';


-- ============================================================================
-- 7. gl_eudr_srs_monitoring_configs — Supplier monitoring configurations
-- ============================================================================
RAISE NOTICE 'V105 [7/12]: Creating gl_eudr_srs_monitoring_configs...';

CREATE TABLE IF NOT EXISTS gl_eudr_srs_monitoring_configs (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id                 UUID            NOT NULL UNIQUE,
    frequency                   VARCHAR(20)     NOT NULL,
        -- 'daily', 'weekly', 'monthly', 'quarterly', 'ad_hoc'
    alert_thresholds            JSONB,
        -- { "risk_score_increase": 10.0, "certification_expiry_days": 90,
        --   "documentation_gap_critical": true, "country_risk_change": 15.0,
        --   "non_conformance_major": 1 }
    watchlist                   BOOLEAN         DEFAULT FALSE,
        -- High-priority monitoring flag
    last_checked_at             TIMESTAMPTZ,
    next_check_date             TIMESTAMPTZ,
    monitoring_type             VARCHAR(30),
        -- 'automated', 'manual', 'hybrid', 'enhanced'
    enabled                     BOOLEAN         DEFAULT TRUE,
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_mc_supplier_id ON gl_eudr_srs_monitoring_configs (supplier_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_mc_frequency ON gl_eudr_srs_monitoring_configs (frequency);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_mc_watchlist ON gl_eudr_srs_monitoring_configs (watchlist);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_mc_last_checked ON gl_eudr_srs_monitoring_configs (last_checked_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_mc_next_check ON gl_eudr_srs_monitoring_configs (next_check_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_mc_enabled ON gl_eudr_srs_monitoring_configs (enabled);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_mc_tenant ON gl_eudr_srs_monitoring_configs (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_mc_created ON gl_eudr_srs_monitoring_configs (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_mc_updated ON gl_eudr_srs_monitoring_configs (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_mc_thresholds ON gl_eudr_srs_monitoring_configs USING GIN (alert_thresholds);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_mc_metadata ON gl_eudr_srs_monitoring_configs USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_srs_monitoring_configs IS 'Monitoring configuration and alert thresholds for suppliers';
COMMENT ON COLUMN gl_eudr_srs_monitoring_configs.watchlist IS 'High-priority monitoring flag for elevated risk suppliers';
COMMENT ON COLUMN gl_eudr_srs_monitoring_configs.alert_thresholds IS 'Configurable thresholds for automated alert generation';


-- ============================================================================
-- 8. gl_eudr_srs_alerts — Supplier risk alerts (hypertable, monthly)
-- ============================================================================
RAISE NOTICE 'V105 [8/12]: Creating gl_eudr_srs_alerts (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_srs_alerts (
    id                          UUID            DEFAULT gen_random_uuid(),
    supplier_id                 UUID            NOT NULL,
    alert_type                  VARCHAR(30)     NOT NULL,
        -- 'risk_score_spike', 'certification_expiry', 'documentation_gap',
        -- 'country_risk_elevated', 'non_conformance', 'network_risk', 'geographic_alert'
    severity                    VARCHAR(20)     NOT NULL,
        -- 'info', 'low', 'medium', 'high', 'critical'
    message                     TEXT            NOT NULL,
    details                     JSONB,
        -- { "previous_value": 45.5, "current_value": 68.3, "threshold": 55.0,
        --   "recommendation": "Initiate enhanced due diligence", "root_cause": "..." }
    acknowledged                BOOLEAN         DEFAULT FALSE,
    acknowledged_by             VARCHAR(255),
    acknowledged_at             TIMESTAMPTZ,
    resolution_notes            TEXT,
    resolved_at                 TIMESTAMPTZ,
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    triggered_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, triggered_at)
);

SELECT create_hypertable(
    'gl_eudr_srs_alerts',
    'triggered_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_al_supplier_id ON gl_eudr_srs_alerts (supplier_id, triggered_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_al_alert_type ON gl_eudr_srs_alerts (alert_type, triggered_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_al_severity ON gl_eudr_srs_alerts (severity, triggered_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_al_acknowledged ON gl_eudr_srs_alerts (acknowledged, triggered_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_al_acknowledged_by ON gl_eudr_srs_alerts (acknowledged_by, triggered_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_al_resolved ON gl_eudr_srs_alerts (resolved_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_al_tenant ON gl_eudr_srs_alerts (tenant_id, triggered_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_al_created ON gl_eudr_srs_alerts (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_al_updated ON gl_eudr_srs_alerts (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_al_supplier_severity ON gl_eudr_srs_alerts (supplier_id, severity, triggered_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_al_details ON gl_eudr_srs_alerts USING GIN (details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_al_metadata ON gl_eudr_srs_alerts USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_srs_alerts IS 'Real-time alerts for supplier risk changes and threshold breaches';
COMMENT ON COLUMN gl_eudr_srs_alerts.alert_type IS 'Category of alert triggering condition';
COMMENT ON COLUMN gl_eudr_srs_alerts.details IS 'Detailed alert context including values, thresholds, and recommendations';


-- ============================================================================
-- 9. gl_eudr_srs_risk_reports — Generated risk reports
-- ============================================================================
RAISE NOTICE 'V105 [9/12]: Creating gl_eudr_srs_risk_reports...';

CREATE TABLE IF NOT EXISTS gl_eudr_srs_risk_reports (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id                   UUID            NOT NULL UNIQUE,
    report_type                 VARCHAR(30)     NOT NULL,
        -- 'supplier_assessment', 'portfolio_summary', 'risk_heatmap', 'trend_analysis',
        -- 'dd_status', 'certification_expiry', 'executive_summary'
    format                      VARCHAR(10)     NOT NULL,
        -- 'pdf', 'excel', 'json', 'html'
    supplier_ids                JSONB,
        -- ["uuid-1", "uuid-2", ...] or null for portfolio-wide reports
    content_hash                VARCHAR(64),
        -- SHA-256 hash of report content for integrity verification
    file_path                   TEXT,
    file_size_bytes             BIGINT,
    generated_by                VARCHAR(255),
    generated_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    valid_until                 TIMESTAMPTZ,
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_rr_report_id ON gl_eudr_srs_risk_reports (report_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_rr_report_type ON gl_eudr_srs_risk_reports (report_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_rr_format ON gl_eudr_srs_risk_reports (format);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_rr_content_hash ON gl_eudr_srs_risk_reports (content_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_rr_generated_by ON gl_eudr_srs_risk_reports (generated_by);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_rr_generated_at ON gl_eudr_srs_risk_reports (generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_rr_valid_until ON gl_eudr_srs_risk_reports (valid_until);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_rr_tenant ON gl_eudr_srs_risk_reports (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_rr_created ON gl_eudr_srs_risk_reports (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_rr_updated ON gl_eudr_srs_risk_reports (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_rr_supplier_ids ON gl_eudr_srs_risk_reports USING GIN (supplier_ids);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_rr_metadata ON gl_eudr_srs_risk_reports USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_srs_risk_reports IS 'Generated risk reports with integrity verification';
COMMENT ON COLUMN gl_eudr_srs_risk_reports.content_hash IS 'SHA-256 hash for report integrity verification';
COMMENT ON COLUMN gl_eudr_srs_risk_reports.supplier_ids IS 'Array of supplier UUIDs included in report scope';


-- ============================================================================
-- 10. gl_eudr_srs_supplier_profiles — Master supplier profile registry
-- ============================================================================
RAISE NOTICE 'V105 [10/12]: Creating gl_eudr_srs_supplier_profiles...';

CREATE TABLE IF NOT EXISTS gl_eudr_srs_supplier_profiles (
    supplier_id                 UUID            PRIMARY KEY,
    name                        VARCHAR(255)    NOT NULL,
    supplier_type               VARCHAR(30)     NOT NULL,
        -- 'producer', 'processor', 'trader', 'distributor', 'manufacturer'
    country_code                CHAR(3)         NOT NULL,
    region                      VARCHAR(255),
    commodities                 JSONB,
        -- ["cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"]
    registration_date           DATE            NOT NULL,
    active                      BOOLEAN         DEFAULT TRUE,
    contact_info                JSONB,
        -- { "email": "...", "phone": "...", "address": "...", "primary_contact": "..." }
    business_registration       VARCHAR(100),
    tax_id                      VARCHAR(50),
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sp_name ON gl_eudr_srs_supplier_profiles (name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sp_supplier_type ON gl_eudr_srs_supplier_profiles (supplier_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sp_country_code ON gl_eudr_srs_supplier_profiles (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sp_region ON gl_eudr_srs_supplier_profiles (region);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sp_registration ON gl_eudr_srs_supplier_profiles (registration_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sp_active ON gl_eudr_srs_supplier_profiles (active);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sp_tenant ON gl_eudr_srs_supplier_profiles (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sp_created ON gl_eudr_srs_supplier_profiles (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sp_updated ON gl_eudr_srs_supplier_profiles (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sp_commodities ON gl_eudr_srs_supplier_profiles USING GIN (commodities);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sp_contact ON gl_eudr_srs_supplier_profiles USING GIN (contact_info);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_sp_metadata ON gl_eudr_srs_supplier_profiles USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_srs_supplier_profiles IS 'Master registry of supplier profiles and basic information';
COMMENT ON COLUMN gl_eudr_srs_supplier_profiles.commodities IS 'Array of EUDR-relevant commodities supplied';


-- ============================================================================
-- 11. gl_eudr_srs_factor_scores — Detailed risk factor score breakdown
-- ============================================================================
RAISE NOTICE 'V105 [11/12]: Creating gl_eudr_srs_factor_scores...';

CREATE TABLE IF NOT EXISTS gl_eudr_srs_factor_scores (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id               UUID            NOT NULL,
        -- FK to gl_eudr_srs_supplier_assessments(id)
    factor_name                 VARCHAR(50)     NOT NULL,
        -- 'country_risk', 'certification', 'documentation', 'geographic_concentration',
        -- 'network_complexity', 'historical_performance', 'due_diligence_status'
    raw_score                   NUMERIC(10,4),
        -- Raw/unnormalized score from factor calculation
    normalized_score            NUMERIC(5,2)    CHECK (normalized_score >= 0 AND normalized_score <= 100),
        -- Normalized to 0-100 scale
    weight                      NUMERIC(5,2)    CHECK (weight >= 0 AND weight <= 1),
        -- Factor weight in overall assessment (sum = 1.0)
    weighted_score              NUMERIC(5,2)    CHECK (weighted_score >= 0 AND weighted_score <= 100),
        -- normalized_score * weight * 100
    data_sources                JSONB,
        -- ["country_risk_evaluator", "certification_records", "documentation_profile"]
    calculation_method          VARCHAR(50),
    confidence                  NUMERIC(5,4)    CHECK (confidence >= 0 AND confidence <= 1),
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_fs_assessment_id ON gl_eudr_srs_factor_scores (assessment_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_fs_factor_name ON gl_eudr_srs_factor_scores (factor_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_fs_raw_score ON gl_eudr_srs_factor_scores (raw_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_fs_normalized ON gl_eudr_srs_factor_scores (normalized_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_fs_weight ON gl_eudr_srs_factor_scores (weight);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_fs_weighted ON gl_eudr_srs_factor_scores (weighted_score);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_fs_confidence ON gl_eudr_srs_factor_scores (confidence);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_fs_tenant ON gl_eudr_srs_factor_scores (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_fs_created ON gl_eudr_srs_factor_scores (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_fs_updated ON gl_eudr_srs_factor_scores (updated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_fs_assessment_factor ON gl_eudr_srs_factor_scores (assessment_id, factor_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_fs_data_sources ON gl_eudr_srs_factor_scores USING GIN (data_sources);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_fs_metadata ON gl_eudr_srs_factor_scores USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_srs_factor_scores IS 'Detailed breakdown of individual risk factors contributing to overall supplier assessment';
COMMENT ON COLUMN gl_eudr_srs_factor_scores.weighted_score IS 'Factor contribution to overall risk score (normalized_score * weight)';
COMMENT ON COLUMN gl_eudr_srs_factor_scores.data_sources IS 'Source systems/agents providing data for this factor';


-- ============================================================================
-- 12. gl_eudr_srs_audit_log — Comprehensive audit trail (hypertable, monthly)
-- ============================================================================
RAISE NOTICE 'V105 [12/12]: Creating gl_eudr_srs_audit_log (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_srs_audit_log (
    id                          UUID            DEFAULT gen_random_uuid(),
    entity_type                 VARCHAR(30)     NOT NULL,
        -- 'supplier_assessment', 'due_diligence', 'documentation', 'certification',
        -- 'geographic_sourcing', 'network', 'monitoring_config', 'alert', 'report'
    entity_id                   UUID            NOT NULL,
    action                      VARCHAR(30)     NOT NULL,
        -- 'created', 'updated', 'deleted', 'assessed', 'acknowledged', 'resolved',
        -- 'generated', 'archived'
    actor                       VARCHAR(100)    NOT NULL,
        -- User ID or system agent identifier
    details                     JSONB,
        -- { "changed_fields": ["risk_score", "risk_level"],
        --   "old_values": {...}, "new_values": {...}, "reason": "..." }
    ip_address                  INET,
    user_agent                  TEXT,
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for immutability verification
    metadata                    JSONB,
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    logged_at                   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, logged_at)
);

SELECT create_hypertable(
    'gl_eudr_srs_audit_log',
    'logged_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_audit_entity_type ON gl_eudr_srs_audit_log (entity_type, logged_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_audit_entity_id ON gl_eudr_srs_audit_log (entity_id, logged_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_audit_action ON gl_eudr_srs_audit_log (action, logged_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_audit_actor ON gl_eudr_srs_audit_log (actor, logged_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_audit_tenant ON gl_eudr_srs_audit_log (tenant_id, logged_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_audit_created ON gl_eudr_srs_audit_log (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_audit_provenance ON gl_eudr_srs_audit_log (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_audit_entity_action ON gl_eudr_srs_audit_log (entity_type, action, logged_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_audit_details ON gl_eudr_srs_audit_log USING GIN (details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_srs_audit_metadata ON gl_eudr_srs_audit_log USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_srs_audit_log IS 'Comprehensive audit trail for all supplier risk scoring operations';
COMMENT ON COLUMN gl_eudr_srs_audit_log.provenance_hash IS 'SHA-256 hash for immutability verification and audit integrity';


-- ============================================================================
-- CONTINUOUS AGGREGATES
-- ============================================================================

-- Hourly assessment statistics
RAISE NOTICE 'V105: Creating continuous aggregate: hourly_assessment_stats...';

DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_srs_hourly_assessment_stats
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 hour', assessed_at) AS hour,
        tenant_id,
        risk_level,
        COUNT(*) AS assessment_count,
        AVG(risk_score) AS avg_risk_score,
        MIN(risk_score) AS min_risk_score,
        MAX(risk_score) AS max_risk_score,
        AVG(confidence) AS avg_confidence,
        AVG(data_completeness) AS avg_completeness
    FROM gl_eudr_srs_supplier_assessments
    GROUP BY hour, tenant_id, risk_level;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_srs_hourly_assessment_stats',
        start_offset => INTERVAL '7 days',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_srs_hourly_assessment_stats IS 'Hourly rollup of supplier assessment statistics by risk level';


-- Hourly alert statistics
RAISE NOTICE 'V105: Creating continuous aggregate: hourly_alert_stats...';

DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_srs_hourly_alert_stats
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 hour', triggered_at) AS hour,
        tenant_id,
        alert_type,
        severity,
        COUNT(*) AS alert_count,
        SUM(CASE WHEN acknowledged = TRUE THEN 1 ELSE 0 END) AS acknowledged_count,
        SUM(CASE WHEN resolved_at IS NOT NULL THEN 1 ELSE 0 END) AS resolved_count,
        AVG(EXTRACT(EPOCH FROM (COALESCE(acknowledged_at, NOW()) - triggered_at))) AS avg_ack_time_seconds,
        AVG(EXTRACT(EPOCH FROM (COALESCE(resolved_at, NOW()) - triggered_at))) AS avg_resolution_time_seconds
    FROM gl_eudr_srs_alerts
    GROUP BY hour, tenant_id, alert_type, severity;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_srs_hourly_alert_stats',
        start_offset => INTERVAL '7 days',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_srs_hourly_alert_stats IS 'Hourly rollup of alert statistics including acknowledgment and resolution metrics';


-- ============================================================================
-- RETENTION POLICIES
-- ============================================================================

RAISE NOTICE 'V105: Creating retention policies...';

-- 5 years for supplier assessments
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_srs_supplier_assessments', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 3 years for alerts
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_srs_alerts', INTERVAL '3 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for audit logs
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_srs_audit_log', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- FINALIZE
-- ============================================================================

RAISE NOTICE 'V105: AGENT-EUDR-017 Supplier Risk Scorer tables created successfully!';
RAISE NOTICE 'V105: Created 12 tables (3 hypertables), 2 continuous aggregates, ~120 indexes';
RAISE NOTICE 'V105: Retention policies: 5y assessments, 3y alerts, 5y audit logs';

COMMIT;
