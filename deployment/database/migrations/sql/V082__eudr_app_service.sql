-- =============================================================================
-- V082: GL-EUDR-APP v1.0 Application Service Schema
-- =============================================================================
-- Application: GL-EUDR-APP (EU Deforestation Regulation Compliance Platform)
-- Date:        March 2026
--
-- Application-level tables for DDS lifecycle management, document
-- verification, pipeline orchestration, risk tracking, and audit trails.
--
-- EXTENDS:
--   V034__eudr_traceability_service.sql   (production plots, custody, DDS)
--   V037__deforestation_satellite_service.sql (satellite scenes, alerts)
--
-- These tables sit in the eudr_app schema and reference the underlying
-- agent data schemas for satellite, traceability, and risk assessment
-- data. They provide the user-facing application layer including
-- supplier profiles, plot registry, DDS lifecycle, document storage,
-- pipeline orchestration, risk aggregation, alerting, settings, audit
-- trail, and DDS reference number sequencing.
-- =============================================================================
-- Tables (10):
--   1. supplier_profiles          - Supplier application profile
--   2. plot_registry              - Plot application registry
--   3. due_diligence_statements   - DDS lifecycle management
--   4. documents                  - Compliance document storage
--   5. pipeline_runs              - Pipeline orchestration tracking
--   6. risk_assessments           - Risk assessment time-series (hypertable)
--   7. risk_alerts                - Risk alert management (hypertable)
--   8. settings                   - Application settings KV store
--   9. audit_trail                - Audit trail for all entities (hypertable)
--  10. dds_sequences              - DDS reference number sequences
--
-- Continuous Aggregates (2):
--   1. daily_risk_summary         - Daily risk assessment aggregates
--   2. monthly_dds_summary        - Monthly DDS lifecycle aggregates
--
-- Also includes: 20+ indexes (B-tree, GIN), security permissions,
-- and comments.
-- Previous: V081__audit_trail_lineage_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS eudr_app;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION eudr_app.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: eudr_app.supplier_profiles
-- =============================================================================
-- Supplier application profile extending greenlang traceability suppliers.
-- Stores company details, compliance status, risk scoring, and ERP source
-- metadata. Each supplier may deal in multiple EUDR-regulated commodities.
-- Compliance and risk fields are updated by the pipeline orchestrator.

CREATE TABLE eudr_app.supplier_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_supplier_id UUID,
    company_name VARCHAR(500) NOT NULL,
    tax_id VARCHAR(100),
    country_iso3 CHAR(3) NOT NULL,
    country_iso2 CHAR(2),
    address_line1 VARCHAR(500),
    address_line2 VARCHAR(500),
    city VARCHAR(200),
    postal_code VARCHAR(20),
    commodities TEXT[] NOT NULL DEFAULT '{}',
    compliance_status VARCHAR(50) NOT NULL DEFAULT 'pending',
    risk_level VARCHAR(20) NOT NULL DEFAULT 'standard',
    overall_risk_score DECIMAL(5,4) DEFAULT 0,
    erp_source VARCHAR(50),
    erp_external_id VARCHAR(200),
    notes TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(200),
    CONSTRAINT chk_sp_compliance_status CHECK (
        compliance_status IN ('compliant', 'pending', 'non_compliant', 'under_review')
    ),
    CONSTRAINT chk_sp_risk_level CHECK (
        risk_level IN ('low', 'standard', 'high', 'critical')
    ),
    CONSTRAINT chk_sp_risk_score_range CHECK (
        overall_risk_score >= 0 AND overall_risk_score <= 1
    ),
    CONSTRAINT chk_sp_company_name_not_empty CHECK (
        LENGTH(TRIM(company_name)) > 0
    ),
    CONSTRAINT chk_sp_country_iso3_length CHECK (
        LENGTH(TRIM(country_iso3)) = 3
    )
);

-- Indexes for supplier_profiles
CREATE INDEX idx_sp_country ON eudr_app.supplier_profiles(country_iso3);
CREATE INDEX idx_sp_compliance ON eudr_app.supplier_profiles(compliance_status);
CREATE INDEX idx_sp_risk ON eudr_app.supplier_profiles(risk_level);
CREATE INDEX idx_sp_risk_score ON eudr_app.supplier_profiles(overall_risk_score DESC);
CREATE INDEX idx_sp_commodities ON eudr_app.supplier_profiles USING GIN(commodities);
CREATE INDEX idx_sp_erp_source ON eudr_app.supplier_profiles(erp_source);
CREATE INDEX idx_sp_external_id ON eudr_app.supplier_profiles(external_supplier_id);
CREATE INDEX idx_sp_created_at ON eudr_app.supplier_profiles(created_at DESC);
CREATE INDEX idx_sp_metadata ON eudr_app.supplier_profiles USING GIN(metadata);

-- Updated_at trigger for supplier_profiles
CREATE TRIGGER trg_supplier_profiles_updated_at
    BEFORE UPDATE ON eudr_app.supplier_profiles
    FOR EACH ROW
    EXECUTE FUNCTION eudr_app.set_updated_at();

-- =============================================================================
-- Table 2: eudr_app.plot_registry
-- =============================================================================
-- Plot application registry linking production plots to suppliers.
-- Stores GeoJSON polygon coordinates, centroid, area, commodity, satellite
-- assessment status, NDVI baselines, and deforestation detection results.
-- Used by the geo-validation and deforestation risk pipeline stages.

CREATE TABLE eudr_app.plot_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_plot_id UUID,
    supplier_id UUID REFERENCES eudr_app.supplier_profiles(id),
    name VARCHAR(500),
    coordinates JSONB NOT NULL,
    centroid_lat DECIMAL(10,6),
    centroid_lon DECIMAL(10,6),
    area_hectares DECIMAL(15,4),
    commodity VARCHAR(50) NOT NULL,
    country_iso3 CHAR(3) NOT NULL,
    risk_level VARCHAR(20) DEFAULT 'standard',
    satellite_status VARCHAR(50) DEFAULT 'not_assessed',
    last_assessed_at TIMESTAMPTZ,
    ndvi_baseline DECIMAL(5,4),
    ndvi_current DECIMAL(5,4),
    ndvi_change DECIMAL(5,4),
    forest_cover_pct DECIMAL(5,2),
    deforestation_detected BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pr_risk_level CHECK (
        risk_level IN ('low', 'standard', 'high', 'critical')
    ),
    CONSTRAINT chk_pr_satellite_status CHECK (
        satellite_status IN (
            'not_assessed', 'in_progress', 'clear',
            'risk_detected', 'deforestation_confirmed', 'error'
        )
    ),
    CONSTRAINT chk_pr_area_non_negative CHECK (
        area_hectares IS NULL OR area_hectares >= 0
    ),
    CONSTRAINT chk_pr_forest_cover_range CHECK (
        forest_cover_pct IS NULL OR (forest_cover_pct >= 0 AND forest_cover_pct <= 100)
    ),
    CONSTRAINT chk_pr_country_iso3_length CHECK (
        LENGTH(TRIM(country_iso3)) = 3
    )
);

-- Indexes for plot_registry
CREATE INDEX idx_plots_supplier ON eudr_app.plot_registry(supplier_id);
CREATE INDEX idx_plots_commodity ON eudr_app.plot_registry(commodity);
CREATE INDEX idx_plots_risk ON eudr_app.plot_registry(risk_level);
CREATE INDEX idx_plots_country ON eudr_app.plot_registry(country_iso3);
CREATE INDEX idx_plots_satellite ON eudr_app.plot_registry(satellite_status);
CREATE INDEX idx_plots_deforestation ON eudr_app.plot_registry(deforestation_detected);
CREATE INDEX idx_plots_coordinates ON eudr_app.plot_registry USING GIN(coordinates);
CREATE INDEX idx_plots_created_at ON eudr_app.plot_registry(created_at DESC);
CREATE INDEX idx_plots_metadata ON eudr_app.plot_registry USING GIN(metadata);

-- Updated_at trigger for plot_registry
CREATE TRIGGER trg_plot_registry_updated_at
    BEFORE UPDATE ON eudr_app.plot_registry
    FOR EACH ROW
    EXECUTE FUNCTION eudr_app.set_updated_at();

-- =============================================================================
-- Table 3: eudr_app.due_diligence_statements
-- =============================================================================
-- Due Diligence Statement lifecycle management per EU Regulation 2023/1115.
-- Stores all 7 DDS sections (operator, product, country, geolocation,
-- risk assessment, mitigation, conclusion), lifecycle status, validation
-- results, EU submission tracking, and amendment lineage.

CREATE TABLE eudr_app.due_diligence_statements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reference_number VARCHAR(50) UNIQUE NOT NULL,
    supplier_id UUID REFERENCES eudr_app.supplier_profiles(id),
    commodity VARCHAR(50) NOT NULL,
    year INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'draft',
    -- 7 DDS Sections per EU 2023/1115
    operator_info JSONB DEFAULT '{}',
    product_description JSONB DEFAULT '{}',
    country_of_production JSONB DEFAULT '{}',
    geolocation_data JSONB DEFAULT '{}',
    risk_assessment JSONB DEFAULT '{}',
    risk_mitigation JSONB DEFAULT '{}',
    conclusion JSONB DEFAULT '{}',
    -- Linked entities
    plot_ids UUID[] DEFAULT '{}',
    document_ids UUID[] DEFAULT '{}',
    -- Risk and validation
    overall_risk_score DECIMAL(5,4),
    validation_result JSONB,
    -- EU submission
    submission_date TIMESTAMPTZ,
    eu_reference VARCHAR(100),
    eu_response JSONB,
    -- Amendment lineage
    amendment_of UUID REFERENCES eudr_app.due_diligence_statements(id),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(200),
    CONSTRAINT chk_dds_status CHECK (
        status IN ('draft', 'review', 'validated', 'submitted',
                   'accepted', 'rejected', 'amended')
    ),
    CONSTRAINT chk_dds_year_range CHECK (
        year >= 2024 AND year <= 2100
    ),
    CONSTRAINT chk_dds_risk_score_range CHECK (
        overall_risk_score IS NULL OR (overall_risk_score >= 0 AND overall_risk_score <= 1)
    ),
    CONSTRAINT chk_dds_ref_not_empty CHECK (
        LENGTH(TRIM(reference_number)) > 0
    )
);

-- Indexes for due_diligence_statements
CREATE INDEX idx_dds_supplier ON eudr_app.due_diligence_statements(supplier_id);
CREATE INDEX idx_dds_status ON eudr_app.due_diligence_statements(status);
CREATE INDEX idx_dds_commodity ON eudr_app.due_diligence_statements(commodity);
CREATE INDEX idx_dds_year ON eudr_app.due_diligence_statements(year);
CREATE INDEX idx_dds_submission ON eudr_app.due_diligence_statements(submission_date DESC);
CREATE INDEX idx_dds_eu_ref ON eudr_app.due_diligence_statements(eu_reference);
CREATE INDEX idx_dds_amendment ON eudr_app.due_diligence_statements(amendment_of);
CREATE INDEX idx_dds_created_at ON eudr_app.due_diligence_statements(created_at DESC);
CREATE INDEX idx_dds_plot_ids ON eudr_app.due_diligence_statements USING GIN(plot_ids);
CREATE INDEX idx_dds_doc_ids ON eudr_app.due_diligence_statements USING GIN(document_ids);
CREATE INDEX idx_dds_validation ON eudr_app.due_diligence_statements USING GIN(validation_result);
CREATE INDEX idx_dds_operator ON eudr_app.due_diligence_statements USING GIN(operator_info);
CREATE INDEX idx_dds_geolocation ON eudr_app.due_diligence_statements USING GIN(geolocation_data);

-- Updated_at trigger for due_diligence_statements
CREATE TRIGGER trg_dds_updated_at
    BEFORE UPDATE ON eudr_app.due_diligence_statements
    FOR EACH ROW
    EXECUTE FUNCTION eudr_app.set_updated_at();

-- =============================================================================
-- Table 4: eudr_app.documents
-- =============================================================================
-- Compliance document storage for EUDR verification. Supports six document
-- types, OCR text extraction results, verification scoring, and compliance
-- findings. Documents can be linked to suppliers, plots, or DDS records.

CREATE TABLE eudr_app.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(500) NOT NULL,
    doc_type VARCHAR(50) NOT NULL,
    file_path VARCHAR(1000),
    file_size_bytes BIGINT,
    mime_type VARCHAR(100),
    verification_status VARCHAR(50) DEFAULT 'pending',
    verification_score DECIMAL(5,4),
    ocr_text TEXT,
    compliance_findings JSONB DEFAULT '[]',
    linked_supplier_id UUID REFERENCES eudr_app.supplier_profiles(id),
    linked_plot_id UUID REFERENCES eudr_app.plot_registry(id),
    linked_dds_id UUID REFERENCES eudr_app.due_diligence_statements(id),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_doc_type CHECK (
        doc_type IN ('CERTIFICATE', 'PERMIT', 'LAND_TITLE',
                     'INVOICE', 'TRANSPORT', 'OTHER')
    ),
    CONSTRAINT chk_doc_verification CHECK (
        verification_status IN ('pending', 'verified', 'failed', 'expired')
    ),
    CONSTRAINT chk_doc_name_not_empty CHECK (
        LENGTH(TRIM(name)) > 0
    ),
    CONSTRAINT chk_doc_file_size_non_negative CHECK (
        file_size_bytes IS NULL OR file_size_bytes >= 0
    ),
    CONSTRAINT chk_doc_score_range CHECK (
        verification_score IS NULL OR (verification_score >= 0 AND verification_score <= 1)
    )
);

-- Indexes for documents
CREATE INDEX idx_docs_supplier ON eudr_app.documents(linked_supplier_id);
CREATE INDEX idx_docs_plot ON eudr_app.documents(linked_plot_id);
CREATE INDEX idx_docs_dds ON eudr_app.documents(linked_dds_id);
CREATE INDEX idx_docs_type ON eudr_app.documents(doc_type);
CREATE INDEX idx_docs_verification ON eudr_app.documents(verification_status);
CREATE INDEX idx_docs_created_at ON eudr_app.documents(created_at DESC);
CREATE INDEX idx_docs_compliance ON eudr_app.documents USING GIN(compliance_findings);
CREATE INDEX idx_docs_metadata ON eudr_app.documents USING GIN(metadata);

-- Updated_at trigger for documents
CREATE TRIGGER trg_documents_updated_at
    BEFORE UPDATE ON eudr_app.documents
    FOR EACH ROW
    EXECUTE FUNCTION eudr_app.set_updated_at();

-- =============================================================================
-- Table 5: eudr_app.pipeline_runs
-- =============================================================================
-- Pipeline orchestration tracking for the 5-stage EUDR compliance pipeline.
-- Records supplier linkage, commodity, status, current stage, per-stage
-- results (stored as JSONB), plot references, error messages, and timing.

CREATE TABLE eudr_app.pipeline_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id UUID REFERENCES eudr_app.supplier_profiles(id),
    commodity VARCHAR(50),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    current_stage VARCHAR(50),
    stages JSONB DEFAULT '{}',
    plot_ids UUID[] DEFAULT '{}',
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pipeline_status CHECK (
        status IN ('pending', 'running', 'completed', 'failed', 'cancelled')
    ),
    CONSTRAINT chk_pipeline_stage CHECK (
        current_stage IS NULL OR current_stage IN (
            'intake', 'geo_validation', 'deforestation_risk',
            'document_verification', 'dds_reporting'
        )
    ),
    CONSTRAINT chk_pipeline_completed_after_started CHECK (
        completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at
    )
);

-- Indexes for pipeline_runs
CREATE INDEX idx_pipeline_supplier ON eudr_app.pipeline_runs(supplier_id);
CREATE INDEX idx_pipeline_status ON eudr_app.pipeline_runs(status);
CREATE INDEX idx_pipeline_stage ON eudr_app.pipeline_runs(current_stage);
CREATE INDEX idx_pipeline_created_at ON eudr_app.pipeline_runs(created_at DESC);
CREATE INDEX idx_pipeline_started_at ON eudr_app.pipeline_runs(started_at DESC);
CREATE INDEX idx_pipeline_stages ON eudr_app.pipeline_runs USING GIN(stages);
CREATE INDEX idx_pipeline_plot_ids ON eudr_app.pipeline_runs USING GIN(plot_ids);

-- =============================================================================
-- Table 6: eudr_app.risk_assessments (hypertable)
-- =============================================================================
-- Time-series risk assessments combining satellite, country, supplier, and
-- document risk sources. Used for trend tracking and risk monitoring
-- dashboards. Partitioned by assessed_at for efficient time-range queries.

CREATE TABLE eudr_app.risk_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID REFERENCES eudr_app.plot_registry(id),
    supplier_id UUID REFERENCES eudr_app.supplier_profiles(id),
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    satellite_risk DECIMAL(5,4) DEFAULT 0,
    country_risk DECIMAL(5,4) DEFAULT 0,
    supplier_risk DECIMAL(5,4) DEFAULT 0,
    document_risk DECIMAL(5,4) DEFAULT 0,
    overall_risk DECIMAL(5,4) DEFAULT 0,
    risk_level VARCHAR(20),
    factors JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    CONSTRAINT chk_ra_satellite_range CHECK (satellite_risk >= 0 AND satellite_risk <= 1),
    CONSTRAINT chk_ra_country_range CHECK (country_risk >= 0 AND country_risk <= 1),
    CONSTRAINT chk_ra_supplier_range CHECK (supplier_risk >= 0 AND supplier_risk <= 1),
    CONSTRAINT chk_ra_document_range CHECK (document_risk >= 0 AND document_risk <= 1),
    CONSTRAINT chk_ra_overall_range CHECK (overall_risk >= 0 AND overall_risk <= 1),
    CONSTRAINT chk_ra_risk_level CHECK (
        risk_level IS NULL OR risk_level IN ('low', 'standard', 'high', 'critical')
    )
);

-- Make risk_assessments a hypertable for time-series queries
SELECT create_hypertable('eudr_app.risk_assessments', 'assessed_at',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes for risk_assessments (hypertable-aware)
CREATE INDEX idx_ra_plot ON eudr_app.risk_assessments(plot_id, assessed_at DESC);
CREATE INDEX idx_ra_supplier ON eudr_app.risk_assessments(supplier_id, assessed_at DESC);
CREATE INDEX idx_ra_risk_level ON eudr_app.risk_assessments(risk_level, assessed_at DESC);
CREATE INDEX idx_ra_overall ON eudr_app.risk_assessments(overall_risk DESC, assessed_at DESC);
CREATE INDEX idx_ra_factors ON eudr_app.risk_assessments USING GIN(factors);
CREATE INDEX idx_ra_recommendations ON eudr_app.risk_assessments USING GIN(recommendations);

-- =============================================================================
-- Table 7: eudr_app.risk_alerts (hypertable)
-- =============================================================================
-- Risk alerts triggered by deforestation detection, compliance failures,
-- document expirations, or risk score thresholds. Supports acknowledgment
-- workflow and severity-based filtering. Partitioned by created_at.

CREATE TABLE eudr_app.risk_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    plot_id UUID REFERENCES eudr_app.plot_registry(id),
    supplier_id UUID REFERENCES eudr_app.supplier_profiles(id),
    message TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(200),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_alert_severity CHECK (
        severity IN ('low', 'medium', 'high', 'critical')
    ),
    CONSTRAINT chk_alert_message_not_empty CHECK (
        LENGTH(TRIM(message)) > 0
    ),
    CONSTRAINT chk_alert_ack_consistency CHECK (
        (acknowledged = FALSE AND acknowledged_at IS NULL AND acknowledged_by IS NULL)
        OR (acknowledged = TRUE AND acknowledged_at IS NOT NULL)
    )
);

-- Make risk_alerts a hypertable for time-series queries
SELECT create_hypertable('eudr_app.risk_alerts', 'created_at',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes for risk_alerts (hypertable-aware)
CREATE INDEX idx_alerts_severity ON eudr_app.risk_alerts(severity, created_at DESC);
CREATE INDEX idx_alerts_type ON eudr_app.risk_alerts(alert_type, created_at DESC);
CREATE INDEX idx_alerts_plot ON eudr_app.risk_alerts(plot_id, created_at DESC);
CREATE INDEX idx_alerts_supplier ON eudr_app.risk_alerts(supplier_id, created_at DESC);
CREATE INDEX idx_alerts_ack ON eudr_app.risk_alerts(acknowledged, created_at DESC);
CREATE INDEX idx_alerts_details ON eudr_app.risk_alerts USING GIN(details);

-- =============================================================================
-- Table 8: eudr_app.settings
-- =============================================================================
-- Application settings key-value store. Stores configuration such as
-- risk thresholds, notification preferences, EU system endpoints,
-- and default pipeline parameters.

CREATE TABLE eudr_app.settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key VARCHAR(200) UNIQUE NOT NULL,
    value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by VARCHAR(200),
    CONSTRAINT chk_settings_key_not_empty CHECK (
        LENGTH(TRIM(key)) > 0
    )
);

-- Index for settings
CREATE INDEX idx_settings_key ON eudr_app.settings(key);
CREATE INDEX idx_settings_updated ON eudr_app.settings(updated_at DESC);

-- Updated_at trigger for settings
CREATE TRIGGER trg_settings_updated_at
    BEFORE UPDATE ON eudr_app.settings
    FOR EACH ROW
    EXECUTE FUNCTION eudr_app.set_updated_at();

-- =============================================================================
-- Table 9: eudr_app.audit_trail (hypertable)
-- =============================================================================
-- Audit trail for all entity changes in the EUDR application. Records
-- entity type, entity ID, action performed, actor identity, and change
-- details. Partitioned by created_at for efficient time-range queries.
-- Used for regulatory audit compliance (EUDR requires 5-year retention).

CREATE TABLE eudr_app.audit_trail (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID NOT NULL,
    action VARCHAR(50) NOT NULL,
    actor VARCHAR(200),
    details JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_audit_entity_type_not_empty CHECK (
        LENGTH(TRIM(entity_type)) > 0
    ),
    CONSTRAINT chk_audit_action_not_empty CHECK (
        LENGTH(TRIM(action)) > 0
    )
);

-- Make audit_trail a hypertable for time-series queries
SELECT create_hypertable('eudr_app.audit_trail', 'created_at',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes for audit_trail (hypertable-aware)
CREATE INDEX idx_audit_entity ON eudr_app.audit_trail(entity_type, entity_id, created_at DESC);
CREATE INDEX idx_audit_action ON eudr_app.audit_trail(action, created_at DESC);
CREATE INDEX idx_audit_actor ON eudr_app.audit_trail(actor, created_at DESC);
CREATE INDEX idx_audit_entity_type ON eudr_app.audit_trail(entity_type, created_at DESC);
CREATE INDEX idx_audit_details ON eudr_app.audit_trail USING GIN(details);

-- =============================================================================
-- Table 10: eudr_app.dds_sequences
-- =============================================================================
-- DDS reference number sequence tracking. Maintains per-country per-year
-- sequence counters for generating unique reference numbers in the format
-- EUDR-{ISO3}-{YEAR}-{SEQ:06d}. Atomic increment via UPDATE ... RETURNING.

CREATE TABLE eudr_app.dds_sequences (
    country_iso3 CHAR(3) NOT NULL,
    year INTEGER NOT NULL,
    last_sequence INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (country_iso3, year),
    CONSTRAINT chk_seq_year_range CHECK (
        year >= 2024 AND year <= 2100
    ),
    CONSTRAINT chk_seq_non_negative CHECK (
        last_sequence >= 0
    ),
    CONSTRAINT chk_seq_country_length CHECK (
        LENGTH(TRIM(country_iso3)) = 3
    )
);

-- =============================================================================
-- Continuous Aggregate: eudr_app.daily_risk_summary
-- =============================================================================
-- Precomputed daily risk assessment statistics for dashboard queries.
-- Aggregates assessment counts, average/max risk, and critical/high counts.

CREATE MATERIALIZED VIEW eudr_app.daily_risk_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', assessed_at) AS bucket,
    COUNT(*) AS assessment_count,
    AVG(overall_risk) AS avg_risk,
    MAX(overall_risk) AS max_risk,
    MIN(overall_risk) AS min_risk,
    COUNT(*) FILTER (WHERE risk_level = 'critical') AS critical_count,
    COUNT(*) FILTER (WHERE risk_level = 'high') AS high_count,
    COUNT(*) FILTER (WHERE risk_level = 'standard') AS standard_count,
    COUNT(*) FILTER (WHERE risk_level = 'low') AS low_count
FROM eudr_app.risk_assessments
WHERE assessed_at IS NOT NULL
GROUP BY bucket
WITH NO DATA;

-- Refresh policy: refresh every 30 minutes, covering the last 3 hours
SELECT add_continuous_aggregate_policy('eudr_app.daily_risk_summary',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Continuous Aggregate: eudr_app.monthly_dds_summary
-- =============================================================================
-- Precomputed monthly DDS lifecycle statistics for compliance reporting.
-- Aggregates DDS counts by status (draft, submitted, accepted, rejected).

CREATE MATERIALIZED VIEW eudr_app.monthly_dds_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 month', created_at) AS bucket,
    COUNT(*) AS total_dds,
    COUNT(*) FILTER (WHERE status = 'draft') AS draft_count,
    COUNT(*) FILTER (WHERE status = 'review') AS review_count,
    COUNT(*) FILTER (WHERE status = 'validated') AS validated_count,
    COUNT(*) FILTER (WHERE status = 'submitted') AS submitted_count,
    COUNT(*) FILTER (WHERE status = 'accepted') AS accepted_count,
    COUNT(*) FILTER (WHERE status = 'rejected') AS rejected_count,
    COUNT(*) FILTER (WHERE status = 'amended') AS amended_count
FROM eudr_app.due_diligence_statements
WHERE created_at IS NOT NULL
GROUP BY bucket
WITH NO DATA;

-- Refresh policy: refresh every hour, covering the last 3 days
SELECT add_continuous_aggregate_policy('eudr_app.monthly_dds_summary',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep risk assessments for 1825 days (5 years, EUDR requirement)
SELECT add_retention_policy('eudr_app.risk_assessments', INTERVAL '1825 days');

-- Keep risk alerts for 1825 days (5 years)
SELECT add_retention_policy('eudr_app.risk_alerts', INTERVAL '1825 days');

-- Keep audit trail for 1825 days (5 years, EUDR regulatory requirement)
SELECT add_retention_policy('eudr_app.audit_trail', INTERVAL '1825 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on risk_assessments after 90 days
ALTER TABLE eudr_app.risk_assessments SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'assessed_at DESC'
);

SELECT add_compression_policy('eudr_app.risk_assessments', INTERVAL '90 days');

-- Enable compression on risk_alerts after 90 days
ALTER TABLE eudr_app.risk_alerts SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_compression_policy('eudr_app.risk_alerts', INTERVAL '90 days');

-- Enable compression on audit_trail after 90 days
ALTER TABLE eudr_app.audit_trail SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_compression_policy('eudr_app.audit_trail', INTERVAL '90 days');

-- =============================================================================
-- Seed: Default Application Settings
-- =============================================================================

INSERT INTO eudr_app.settings (key, value, description) VALUES
    ('risk.threshold.high', '0.7', 'Score threshold for HIGH risk classification'),
    ('risk.threshold.critical', '0.9', 'Score threshold for CRITICAL risk classification'),
    ('risk.weight.satellite', '0.35', 'Weight of satellite risk in overall score'),
    ('risk.weight.country', '0.25', 'Weight of country risk in overall score'),
    ('risk.weight.supplier', '0.20', 'Weight of supplier risk in overall score'),
    ('risk.weight.document', '0.20', 'Weight of document risk in overall score'),
    ('pipeline.max_concurrent', '10', 'Maximum concurrent pipeline runs'),
    ('pipeline.retry_max', '3', 'Maximum retry attempts per pipeline stage'),
    ('pipeline.timeout_seconds', '300', 'Per-stage timeout in seconds'),
    ('dds.reference_prefix', '"EUDR"', 'Prefix for DDS reference numbers'),
    ('dds.auto_submit', 'false', 'Whether to auto-submit validated DDS'),
    ('eu.system_endpoint', '"https://eudr.ec.europa.eu/api/v1"', 'EU Information System API endpoint'),
    ('satellite.ndvi_threshold', '-0.15', 'NDVI change threshold for deforestation'),
    ('satellite.cache_days', '30', 'Days to cache satellite assessment results'),
    ('document.max_upload_mb', '50', 'Maximum document upload size in megabytes'),
    ('document.retention_days', '3650', 'Document retention period (EUDR requires 5 years)')
ON CONFLICT (key) DO NOTHING;

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA eudr_app TO greenlang_app;
GRANT ALL ON ALL TABLES IN SCHEMA eudr_app TO greenlang_app;
GRANT ALL ON ALL SEQUENCES IN SCHEMA eudr_app TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON eudr_app.daily_risk_summary TO greenlang_app;
GRANT SELECT ON eudr_app.monthly_dds_summary TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA eudr_app TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA eudr_app TO greenlang_readonly;
GRANT SELECT ON eudr_app.daily_risk_summary TO greenlang_readonly;
GRANT SELECT ON eudr_app.monthly_dds_summary TO greenlang_readonly;

-- Add EUDR app service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'eudr_app:suppliers:read', 'eudr_app', 'suppliers_read', 'View EUDR supplier profiles and compliance status'),
    (gen_random_uuid(), 'eudr_app:suppliers:write', 'eudr_app', 'suppliers_write', 'Create and manage EUDR supplier profiles'),
    (gen_random_uuid(), 'eudr_app:plots:read', 'eudr_app', 'plots_read', 'View EUDR plot registry and satellite status'),
    (gen_random_uuid(), 'eudr_app:plots:write', 'eudr_app', 'plots_write', 'Create and manage EUDR plot records'),
    (gen_random_uuid(), 'eudr_app:dds:read', 'eudr_app', 'dds_read', 'View Due Diligence Statements'),
    (gen_random_uuid(), 'eudr_app:dds:write', 'eudr_app', 'dds_write', 'Create, validate, and submit DDS'),
    (gen_random_uuid(), 'eudr_app:documents:read', 'eudr_app', 'documents_read', 'View compliance documents'),
    (gen_random_uuid(), 'eudr_app:documents:write', 'eudr_app', 'documents_write', 'Upload and verify compliance documents'),
    (gen_random_uuid(), 'eudr_app:pipeline:read', 'eudr_app', 'pipeline_read', 'View pipeline run status and history'),
    (gen_random_uuid(), 'eudr_app:pipeline:write', 'eudr_app', 'pipeline_write', 'Start, retry, and cancel pipeline runs'),
    (gen_random_uuid(), 'eudr_app:risk:read', 'eudr_app', 'risk_read', 'View risk assessments, alerts, and trends'),
    (gen_random_uuid(), 'eudr_app:risk:write', 'eudr_app', 'risk_write', 'Create risk assessments and manage alerts'),
    (gen_random_uuid(), 'eudr_app:settings:read', 'eudr_app', 'settings_read', 'View application settings'),
    (gen_random_uuid(), 'eudr_app:settings:write', 'eudr_app', 'settings_write', 'Modify application settings'),
    (gen_random_uuid(), 'eudr_app:audit:read', 'eudr_app', 'audit_read', 'View audit trail records'),
    (gen_random_uuid(), 'eudr_app:admin', 'eudr_app', 'admin', 'EUDR application administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA eudr_app IS 'GL-EUDR-APP v1.0 Application Schema - EU Deforestation Regulation Compliance Platform with supplier profiles, plot registry, DDS lifecycle, document verification, pipeline orchestration, risk aggregation, alerting, settings, and audit trail';

COMMENT ON TABLE eudr_app.supplier_profiles IS 'Supplier application profiles for EUDR compliance with company details, commodities, compliance status, risk scoring, and ERP source tracking';
COMMENT ON TABLE eudr_app.plot_registry IS 'Production plot registry with GeoJSON boundaries, satellite assessment status, NDVI baselines, deforestation detection, and supplier linkage';
COMMENT ON TABLE eudr_app.due_diligence_statements IS 'Due Diligence Statement lifecycle management per EU 2023/1115 with 7 sections, validation, EU submission tracking, and amendment lineage';
COMMENT ON TABLE eudr_app.documents IS 'Compliance document storage with 6 types (CERTIFICATE/PERMIT/LAND_TITLE/INVOICE/TRANSPORT/OTHER), OCR extraction, verification scoring, and entity linking';
COMMENT ON TABLE eudr_app.pipeline_runs IS 'Five-stage EUDR compliance pipeline orchestration (intake/geo_validation/deforestation_risk/document_verification/dds_reporting)';
COMMENT ON TABLE eudr_app.risk_assessments IS 'TimescaleDB hypertable: time-series risk assessments combining satellite, country, supplier, and document risk sources for trend tracking';
COMMENT ON TABLE eudr_app.risk_alerts IS 'TimescaleDB hypertable: risk alerts for deforestation detection, compliance failures, and risk threshold breaches with acknowledgment workflow';
COMMENT ON TABLE eudr_app.settings IS 'Application settings key-value store for risk thresholds, pipeline parameters, EU system endpoints, and notification preferences';
COMMENT ON TABLE eudr_app.audit_trail IS 'TimescaleDB hypertable: regulatory audit trail for all entity changes with 5-year retention per EUDR requirements';
COMMENT ON TABLE eudr_app.dds_sequences IS 'DDS reference number sequence tracking per country per year for format EUDR-{ISO3}-{YEAR}-{SEQ:06d}';

COMMENT ON MATERIALIZED VIEW eudr_app.daily_risk_summary IS 'Continuous aggregate: daily risk assessment statistics with average/max/min risk, critical/high/standard/low counts for dashboard queries';
COMMENT ON MATERIALIZED VIEW eudr_app.monthly_dds_summary IS 'Continuous aggregate: monthly DDS lifecycle statistics with counts by status (draft/review/validated/submitted/accepted/rejected/amended) for compliance reporting';

COMMENT ON COLUMN eudr_app.supplier_profiles.compliance_status IS 'EUDR compliance status: compliant, pending, non_compliant, under_review';
COMMENT ON COLUMN eudr_app.supplier_profiles.risk_level IS 'Risk classification: low, standard, high, critical';
COMMENT ON COLUMN eudr_app.supplier_profiles.overall_risk_score IS 'Composite risk score (0-1) combining satellite, country, supplier, and document risk';
COMMENT ON COLUMN eudr_app.supplier_profiles.commodities IS 'Array of EUDR-regulated commodities: cattle, cocoa, coffee, oil_palm, rubber, soya, wood';
COMMENT ON COLUMN eudr_app.plot_registry.coordinates IS 'GeoJSON Polygon geometry for plot boundary';
COMMENT ON COLUMN eudr_app.plot_registry.ndvi_baseline IS 'NDVI baseline value from EUDR cutoff date (31 Dec 2020)';
COMMENT ON COLUMN eudr_app.plot_registry.ndvi_change IS 'NDVI change from baseline to current (negative indicates vegetation loss)';
COMMENT ON COLUMN eudr_app.plot_registry.satellite_status IS 'Satellite assessment status: not_assessed, in_progress, clear, risk_detected, deforestation_confirmed, error';
COMMENT ON COLUMN eudr_app.due_diligence_statements.reference_number IS 'Unique DDS reference number in format EUDR-{ISO3}-{YEAR}-{SEQ:06d}';
COMMENT ON COLUMN eudr_app.due_diligence_statements.status IS 'DDS lifecycle status: draft, review, validated, submitted, accepted, rejected, amended';
COMMENT ON COLUMN eudr_app.due_diligence_statements.amendment_of IS 'Self-referencing FK: ID of the DDS that this DDS amends';
COMMENT ON COLUMN eudr_app.documents.doc_type IS 'Document type: CERTIFICATE, PERMIT, LAND_TITLE, INVOICE, TRANSPORT, OTHER';
COMMENT ON COLUMN eudr_app.documents.verification_status IS 'Verification status: pending, verified, failed, expired';
COMMENT ON COLUMN eudr_app.pipeline_runs.status IS 'Pipeline status: pending, running, completed, failed, cancelled';
COMMENT ON COLUMN eudr_app.pipeline_runs.current_stage IS 'Current pipeline stage: intake, geo_validation, deforestation_risk, document_verification, dds_reporting';
COMMENT ON COLUMN eudr_app.pipeline_runs.stages IS 'JSONB map of stage names to StageResult (status, timing, output, error)';
COMMENT ON COLUMN eudr_app.risk_assessments.overall_risk IS 'Weighted composite risk score (0-1): 0.35*satellite + 0.25*country + 0.20*supplier + 0.20*document';
COMMENT ON COLUMN eudr_app.risk_alerts.severity IS 'Alert severity: low, medium, high, critical';
COMMENT ON COLUMN eudr_app.audit_trail.entity_type IS 'Type of audited entity: supplier, plot, dds, document, pipeline, risk_assessment, alert, setting';
COMMENT ON COLUMN eudr_app.dds_sequences.last_sequence IS 'Last used sequence number for the given country/year combination';
