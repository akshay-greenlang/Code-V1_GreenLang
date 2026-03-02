-- =====================================================================================
-- Migration: V081__audit_trail_lineage_service.sql
-- Description: AGENT-MRV-030 Audit Trail & Lineage Agent (Cross-Cutting)
-- Agent: GL-MRV-X-042
-- Framework: GHG Protocol, ISO 14064, CSRD ESRS E1, SB 253, CBAM, CDP, TCFD, PCAF, SBTi
-- Created: 2026-03-01
-- =====================================================================================
-- Schema: audit_trail_lineage_service
-- Tables: 12 (4 reference + 6 operational + 2 supporting)
-- Hypertables: 3 (audit_events 7d, evidence_packages 30d, compliance_traces 30d)
-- Continuous Aggregates: 2 (daily_event_stats, monthly_compliance_summary)
-- Indexes: ~80
-- Seed Data: 250+ records
-- =====================================================================================

-- =====================================================================================
-- SCHEMA CREATION
-- =====================================================================================

CREATE SCHEMA IF NOT EXISTS audit_trail_lineage_service;

COMMENT ON SCHEMA audit_trail_lineage_service IS 'AGENT-MRV-030: Audit Trail & Lineage Agent - Cross-cutting agent for immutable audit chains, data lineage DAGs, evidence packaging, and multi-framework compliance tracing';

-- =====================================================================================
-- TABLE 1: gl_atl_event_types (Reference)
-- Description: Audit event type definitions with severity levels and categories
-- =====================================================================================

CREATE TABLE audit_trail_lineage_service.gl_atl_event_types (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) UNIQUE NOT NULL,
    description TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    category VARCHAR(50) NOT NULL,
    requires_lineage BOOLEAN DEFAULT TRUE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_atl_evt_severity CHECK (severity IN ('info', 'low', 'medium', 'high', 'critical')),
    CONSTRAINT chk_atl_evt_category CHECK (category IN (
        'data_intake', 'calculation', 'validation', 'reporting',
        'compliance', 'change_management', 'access', 'system',
        'evidence', 'lineage', 'reconciliation', 'approval'
    ))
);

CREATE INDEX idx_atl_evt_type ON audit_trail_lineage_service.gl_atl_event_types(event_type);
CREATE INDEX idx_atl_evt_severity ON audit_trail_lineage_service.gl_atl_event_types(severity);
CREATE INDEX idx_atl_evt_category ON audit_trail_lineage_service.gl_atl_event_types(category);
CREATE INDEX idx_atl_evt_active ON audit_trail_lineage_service.gl_atl_event_types(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE audit_trail_lineage_service.gl_atl_event_types IS 'Audit event type definitions with severity classification and lineage requirements';
COMMENT ON COLUMN audit_trail_lineage_service.gl_atl_event_types.requires_lineage IS 'Whether this event type must be linked to a lineage node';

-- =====================================================================================
-- TABLE 2: gl_atl_framework_requirements (Reference)
-- Description: Regulatory framework audit/evidence requirements (200+ rows)
-- Sources: GHG Protocol, ISO 14064, CSRD ESRS E1, SB 253, CBAM, CDP, TCFD, PCAF, SBTi
-- =====================================================================================

CREATE TABLE audit_trail_lineage_service.gl_atl_framework_requirements (
    id SERIAL PRIMARY KEY,
    framework VARCHAR(50) NOT NULL,
    framework_version VARCHAR(20) NOT NULL,
    requirement_id VARCHAR(50) NOT NULL,
    requirement_text TEXT NOT NULL,
    section_reference VARCHAR(100),
    data_points_required JSONB DEFAULT '[]',
    evidence_types JSONB DEFAULT '[]',
    assurance_level VARCHAR(20) NOT NULL DEFAULT 'limited',
    is_mandatory BOOLEAN DEFAULT TRUE,
    effective_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(framework, requirement_id),
    CONSTRAINT chk_atl_fw_assurance CHECK (assurance_level IN ('none', 'limited', 'reasonable', 'absolute'))
);

CREATE INDEX idx_atl_fw_framework ON audit_trail_lineage_service.gl_atl_framework_requirements(framework);
CREATE INDEX idx_atl_fw_version ON audit_trail_lineage_service.gl_atl_framework_requirements(framework_version);
CREATE INDEX idx_atl_fw_req_id ON audit_trail_lineage_service.gl_atl_framework_requirements(requirement_id);
CREATE INDEX idx_atl_fw_assurance ON audit_trail_lineage_service.gl_atl_framework_requirements(assurance_level);
CREATE INDEX idx_atl_fw_mandatory ON audit_trail_lineage_service.gl_atl_framework_requirements(is_mandatory) WHERE is_mandatory = TRUE;
CREATE INDEX idx_atl_fw_active ON audit_trail_lineage_service.gl_atl_framework_requirements(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_atl_fw_effective ON audit_trail_lineage_service.gl_atl_framework_requirements(effective_date);

COMMENT ON TABLE audit_trail_lineage_service.gl_atl_framework_requirements IS 'Regulatory framework audit trail and evidence requirements across 9 frameworks (200+ requirements)';
COMMENT ON COLUMN audit_trail_lineage_service.gl_atl_framework_requirements.data_points_required IS 'JSON array of required data point identifiers for this requirement';
COMMENT ON COLUMN audit_trail_lineage_service.gl_atl_framework_requirements.evidence_types IS 'JSON array of acceptable evidence types (document, calculation, measurement, etc.)';

-- =====================================================================================
-- TABLE 3: gl_atl_lineage_node_types (Reference)
-- Description: Lineage DAG node type definitions (10 types)
-- =====================================================================================

CREATE TABLE audit_trail_lineage_service.gl_atl_lineage_node_types (
    id SERIAL PRIMARY KEY,
    node_type VARCHAR(50) UNIQUE NOT NULL,
    description TEXT NOT NULL,
    level_range VARCHAR(20) NOT NULL,
    icon VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT chk_atl_ntype_level CHECK (level_range IN (
        'L0_source', 'L1_intake', 'L2_normalized', 'L3_validated',
        'L4_calculated', 'L5_aggregated', 'L6_reported', 'L7_assured',
        'mixed', 'any'
    ))
);

CREATE INDEX idx_atl_ntype_type ON audit_trail_lineage_service.gl_atl_lineage_node_types(node_type);
CREATE INDEX idx_atl_ntype_level ON audit_trail_lineage_service.gl_atl_lineage_node_types(level_range);
CREATE INDEX idx_atl_ntype_active ON audit_trail_lineage_service.gl_atl_lineage_node_types(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE audit_trail_lineage_service.gl_atl_lineage_node_types IS 'Lineage DAG node type definitions with processing level ranges (L0-L7)';
COMMENT ON COLUMN audit_trail_lineage_service.gl_atl_lineage_node_types.level_range IS 'Data processing level: L0=source through L7=assured';

-- =====================================================================================
-- TABLE 4: gl_atl_change_type_definitions (Reference)
-- Description: Change event type definitions (8 types)
-- =====================================================================================

CREATE TABLE audit_trail_lineage_service.gl_atl_change_type_definitions (
    id SERIAL PRIMARY KEY,
    change_type VARCHAR(50) UNIQUE NOT NULL,
    description TEXT NOT NULL,
    default_severity VARCHAR(20) NOT NULL,
    requires_recalculation BOOLEAN DEFAULT TRUE,
    notification_required BOOLEAN DEFAULT TRUE,
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT chk_atl_chg_severity CHECK (default_severity IN ('info', 'low', 'medium', 'high', 'critical'))
);

CREATE INDEX idx_atl_chg_type ON audit_trail_lineage_service.gl_atl_change_type_definitions(change_type);
CREATE INDEX idx_atl_chg_severity ON audit_trail_lineage_service.gl_atl_change_type_definitions(default_severity);
CREATE INDEX idx_atl_chg_recalc ON audit_trail_lineage_service.gl_atl_change_type_definitions(requires_recalculation) WHERE requires_recalculation = TRUE;
CREATE INDEX idx_atl_chg_active ON audit_trail_lineage_service.gl_atl_change_type_definitions(is_active) WHERE is_active = TRUE;

COMMENT ON TABLE audit_trail_lineage_service.gl_atl_change_type_definitions IS 'Change event type definitions with severity defaults and recalculation requirements';

-- =====================================================================================
-- TABLE 5: gl_atl_audit_events (Operational - Hypertable)
-- Description: Immutable, hash-chained audit event log
-- Chunk interval: 7 days | Compression: 90 days | Retention: 7 years
-- =====================================================================================

CREATE TABLE audit_trail_lineage_service.gl_atl_audit_events (
    id BIGSERIAL,
    event_id UUID NOT NULL DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL,
    agent_id VARCHAR(50) NOT NULL,
    scope VARCHAR(30),
    category SMALLINT,
    organization_id UUID NOT NULL,
    reporting_year SMALLINT NOT NULL,
    calculation_id UUID,
    data_quality_score DECIMAL(5,4),
    payload JSONB NOT NULL DEFAULT '{}',
    prev_event_hash VARCHAR(64) NOT NULL,
    event_hash VARCHAR(64) NOT NULL,
    chain_position BIGINT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at),
    CONSTRAINT chk_atl_ae_year CHECK (reporting_year >= 2000 AND reporting_year <= 2100),
    CONSTRAINT chk_atl_ae_scope CHECK (scope IS NULL OR scope IN ('scope_1', 'scope_2', 'scope_3', 'cross_scope')),
    CONSTRAINT chk_atl_ae_category CHECK (category IS NULL OR (category >= 1 AND category <= 15)),
    CONSTRAINT chk_atl_ae_dq CHECK (data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 1)),
    CONSTRAINT chk_atl_ae_hash_len CHECK (char_length(event_hash) = 64),
    CONSTRAINT chk_atl_ae_prev_hash_len CHECK (char_length(prev_event_hash) = 64),
    CONSTRAINT chk_atl_ae_chain_pos CHECK (chain_position >= 0)
);

SELECT create_hypertable(
    'audit_trail_lineage_service.gl_atl_audit_events',
    'created_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_atl_ae_event_id ON audit_trail_lineage_service.gl_atl_audit_events(event_id, created_at DESC);
CREATE INDEX idx_atl_ae_org_year ON audit_trail_lineage_service.gl_atl_audit_events(organization_id, reporting_year, created_at DESC);
CREATE INDEX idx_atl_ae_agent ON audit_trail_lineage_service.gl_atl_audit_events(agent_id, created_at DESC);
CREATE INDEX idx_atl_ae_event_type ON audit_trail_lineage_service.gl_atl_audit_events(event_type, created_at DESC);
CREATE INDEX idx_atl_ae_calc_id ON audit_trail_lineage_service.gl_atl_audit_events(calculation_id, created_at DESC) WHERE calculation_id IS NOT NULL;
CREATE INDEX idx_atl_ae_chain_pos ON audit_trail_lineage_service.gl_atl_audit_events(chain_position, created_at DESC);
CREATE INDEX idx_atl_ae_event_hash ON audit_trail_lineage_service.gl_atl_audit_events(event_hash, created_at DESC);
CREATE INDEX idx_atl_ae_scope ON audit_trail_lineage_service.gl_atl_audit_events(scope, created_at DESC) WHERE scope IS NOT NULL;
CREATE INDEX idx_atl_ae_category ON audit_trail_lineage_service.gl_atl_audit_events(category, created_at DESC) WHERE category IS NOT NULL;
CREATE INDEX idx_atl_ae_dq ON audit_trail_lineage_service.gl_atl_audit_events(data_quality_score, created_at DESC) WHERE data_quality_score IS NOT NULL;
CREATE INDEX idx_atl_ae_payload ON audit_trail_lineage_service.gl_atl_audit_events USING GIN (payload);
CREATE INDEX idx_atl_ae_metadata ON audit_trail_lineage_service.gl_atl_audit_events USING GIN (metadata);

COMMENT ON TABLE audit_trail_lineage_service.gl_atl_audit_events IS 'Immutable hash-chained audit events (TimescaleDB hypertable, 7-day chunks, 7-year retention)';
COMMENT ON COLUMN audit_trail_lineage_service.gl_atl_audit_events.prev_event_hash IS 'SHA-256 hash of the previous event in the chain (genesis = 64 zeros)';
COMMENT ON COLUMN audit_trail_lineage_service.gl_atl_audit_events.event_hash IS 'SHA-256 hash of (prev_event_hash + event_type + agent_id + payload + created_at)';
COMMENT ON COLUMN audit_trail_lineage_service.gl_atl_audit_events.chain_position IS 'Monotonically increasing position within the organization audit chain';

-- =====================================================================================
-- TABLE 6: gl_atl_lineage_nodes (Operational)
-- Description: Lineage DAG nodes representing data at each processing level
-- =====================================================================================

CREATE TABLE audit_trail_lineage_service.gl_atl_lineage_nodes (
    id BIGSERIAL PRIMARY KEY,
    node_id UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    node_type VARCHAR(50) NOT NULL,
    level VARCHAR(20) NOT NULL,
    agent_id VARCHAR(50),
    qualified_name VARCHAR(500) NOT NULL,
    display_name VARCHAR(200),
    value DECIMAL(20,8),
    unit VARCHAR(50),
    data_quality_score DECIMAL(5,4),
    organization_id UUID NOT NULL,
    reporting_year SMALLINT NOT NULL,
    provenance_hash VARCHAR(64),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_atl_ln_year CHECK (reporting_year >= 2000 AND reporting_year <= 2100),
    CONSTRAINT chk_atl_ln_level CHECK (level IN (
        'L0_source', 'L1_intake', 'L2_normalized', 'L3_validated',
        'L4_calculated', 'L5_aggregated', 'L6_reported', 'L7_assured'
    )),
    CONSTRAINT chk_atl_ln_dq CHECK (data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 1)),
    CONSTRAINT chk_atl_ln_hash_len CHECK (provenance_hash IS NULL OR char_length(provenance_hash) = 64)
);

CREATE INDEX idx_atl_ln_org_year ON audit_trail_lineage_service.gl_atl_lineage_nodes(organization_id, reporting_year);
CREATE INDEX idx_atl_ln_node_type ON audit_trail_lineage_service.gl_atl_lineage_nodes(node_type);
CREATE INDEX idx_atl_ln_level ON audit_trail_lineage_service.gl_atl_lineage_nodes(level);
CREATE INDEX idx_atl_ln_agent ON audit_trail_lineage_service.gl_atl_lineage_nodes(agent_id) WHERE agent_id IS NOT NULL;
CREATE INDEX idx_atl_ln_qname ON audit_trail_lineage_service.gl_atl_lineage_nodes(qualified_name);
CREATE INDEX idx_atl_ln_provenance ON audit_trail_lineage_service.gl_atl_lineage_nodes(provenance_hash) WHERE provenance_hash IS NOT NULL;
CREATE INDEX idx_atl_ln_created ON audit_trail_lineage_service.gl_atl_lineage_nodes(created_at DESC);
CREATE INDEX idx_atl_ln_metadata ON audit_trail_lineage_service.gl_atl_lineage_nodes USING GIN (metadata);

COMMENT ON TABLE audit_trail_lineage_service.gl_atl_lineage_nodes IS 'Lineage DAG nodes at processing levels L0-L7 with provenance tracking';
COMMENT ON COLUMN audit_trail_lineage_service.gl_atl_lineage_nodes.qualified_name IS 'Fully qualified name (e.g., org.scope1.stationary.natural_gas.tier2.2025)';
COMMENT ON COLUMN audit_trail_lineage_service.gl_atl_lineage_nodes.level IS 'Processing level: L0_source through L7_assured';

-- =====================================================================================
-- TABLE 7: gl_atl_lineage_edges (Operational)
-- Description: Lineage DAG edges connecting nodes with transformation metadata
-- =====================================================================================

CREATE TABLE audit_trail_lineage_service.gl_atl_lineage_edges (
    id BIGSERIAL PRIMARY KEY,
    edge_id UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    source_node_id UUID NOT NULL REFERENCES audit_trail_lineage_service.gl_atl_lineage_nodes(node_id),
    target_node_id UUID NOT NULL REFERENCES audit_trail_lineage_service.gl_atl_lineage_nodes(node_id),
    edge_type VARCHAR(50) NOT NULL,
    transformation_description TEXT,
    confidence DECIMAL(5,4) DEFAULT 1.0000,
    organization_id UUID NOT NULL,
    reporting_year SMALLINT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_atl_le_no_self CHECK (source_node_id != target_node_id),
    CONSTRAINT chk_atl_le_year CHECK (reporting_year >= 2000 AND reporting_year <= 2100),
    CONSTRAINT chk_atl_le_confidence CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT chk_atl_le_edge_type CHECK (edge_type IN (
        'derived_from', 'aggregated_from', 'calculated_from', 'normalized_from',
        'validated_from', 'reported_from', 'assured_from', 'split_from',
        'merged_from', 'transformed_from'
    ))
);

CREATE INDEX idx_atl_le_source ON audit_trail_lineage_service.gl_atl_lineage_edges(source_node_id);
CREATE INDEX idx_atl_le_target ON audit_trail_lineage_service.gl_atl_lineage_edges(target_node_id);
CREATE INDEX idx_atl_le_edge_type ON audit_trail_lineage_service.gl_atl_lineage_edges(edge_type);
CREATE INDEX idx_atl_le_org_year ON audit_trail_lineage_service.gl_atl_lineage_edges(organization_id, reporting_year);
CREATE INDEX idx_atl_le_created ON audit_trail_lineage_service.gl_atl_lineage_edges(created_at DESC);
CREATE INDEX idx_atl_le_confidence ON audit_trail_lineage_service.gl_atl_lineage_edges(confidence) WHERE confidence < 1.0;
CREATE INDEX idx_atl_le_metadata ON audit_trail_lineage_service.gl_atl_lineage_edges USING GIN (metadata);

COMMENT ON TABLE audit_trail_lineage_service.gl_atl_lineage_edges IS 'Lineage DAG edges with transformation metadata and confidence scores';
COMMENT ON COLUMN audit_trail_lineage_service.gl_atl_lineage_edges.edge_type IS 'Relationship type: derived_from, aggregated_from, calculated_from, etc.';
COMMENT ON COLUMN audit_trail_lineage_service.gl_atl_lineage_edges.confidence IS 'Confidence score for this transformation (0.0 to 1.0)';

-- =====================================================================================
-- TABLE 8: gl_atl_evidence_packages (Operational - Hypertable)
-- Description: Evidence packages for audit and assurance engagements
-- Chunk interval: 30 days | Compression: 180 days | Retention: 10 years
-- =====================================================================================

CREATE TABLE audit_trail_lineage_service.gl_atl_evidence_packages (
    id BIGSERIAL,
    package_id UUID NOT NULL DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL,
    reporting_year SMALLINT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'draft',
    assurance_level VARCHAR(20),
    frameworks JSONB NOT NULL DEFAULT '[]',
    scope_filter JSONB,
    contents_summary JSONB,
    total_events INT DEFAULT 0,
    total_lineage_nodes INT DEFAULT 0,
    completeness_score DECIMAL(5,4),
    package_hash VARCHAR(64),
    signature TEXT,
    signature_algorithm VARCHAR(30),
    signed_at TIMESTAMPTZ,
    signed_by VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at),
    CONSTRAINT chk_atl_ep_year CHECK (reporting_year >= 2000 AND reporting_year <= 2100),
    CONSTRAINT chk_atl_ep_status CHECK (status IN ('draft', 'assembling', 'review', 'signed', 'submitted', 'rejected')),
    CONSTRAINT chk_atl_ep_assurance CHECK (assurance_level IS NULL OR assurance_level IN ('none', 'limited', 'reasonable', 'absolute')),
    CONSTRAINT chk_atl_ep_completeness CHECK (completeness_score IS NULL OR (completeness_score >= 0 AND completeness_score <= 1)),
    CONSTRAINT chk_atl_ep_hash_len CHECK (package_hash IS NULL OR char_length(package_hash) = 64),
    CONSTRAINT chk_atl_ep_sig_alg CHECK (signature_algorithm IS NULL OR signature_algorithm IN ('RSA-SHA256', 'ECDSA-P256', 'Ed25519'))
);

SELECT create_hypertable(
    'audit_trail_lineage_service.gl_atl_evidence_packages',
    'created_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_atl_ep_package_id ON audit_trail_lineage_service.gl_atl_evidence_packages(package_id, created_at DESC);
CREATE INDEX idx_atl_ep_org_year ON audit_trail_lineage_service.gl_atl_evidence_packages(organization_id, reporting_year, created_at DESC);
CREATE INDEX idx_atl_ep_status ON audit_trail_lineage_service.gl_atl_evidence_packages(status, created_at DESC);
CREATE INDEX idx_atl_ep_frameworks ON audit_trail_lineage_service.gl_atl_evidence_packages USING GIN (frameworks);
CREATE INDEX idx_atl_ep_completeness ON audit_trail_lineage_service.gl_atl_evidence_packages(completeness_score, created_at DESC) WHERE completeness_score IS NOT NULL;
CREATE INDEX idx_atl_ep_signed ON audit_trail_lineage_service.gl_atl_evidence_packages(signed_at, created_at DESC) WHERE signed_at IS NOT NULL;
CREATE INDEX idx_atl_ep_assurance ON audit_trail_lineage_service.gl_atl_evidence_packages(assurance_level, created_at DESC) WHERE assurance_level IS NOT NULL;
CREATE INDEX idx_atl_ep_metadata ON audit_trail_lineage_service.gl_atl_evidence_packages USING GIN (metadata);

COMMENT ON TABLE audit_trail_lineage_service.gl_atl_evidence_packages IS 'Evidence packages for external audit and assurance (TimescaleDB hypertable, 30-day chunks, 10-year retention)';
COMMENT ON COLUMN audit_trail_lineage_service.gl_atl_evidence_packages.package_hash IS 'SHA-256 hash of the entire evidence package contents';
COMMENT ON COLUMN audit_trail_lineage_service.gl_atl_evidence_packages.signature IS 'Cryptographic signature of the package_hash by the signing authority';

-- =====================================================================================
-- TABLE 9: gl_atl_compliance_traces (Operational - Hypertable)
-- Description: Per-requirement compliance trace records
-- Chunk interval: 30 days | Compression: 90 days | Retention: 5 years
-- =====================================================================================

CREATE TABLE audit_trail_lineage_service.gl_atl_compliance_traces (
    id BIGSERIAL,
    trace_id UUID NOT NULL DEFAULT gen_random_uuid(),
    framework VARCHAR(50) NOT NULL,
    organization_id UUID NOT NULL,
    reporting_year SMALLINT NOT NULL,
    requirement_id VARCHAR(50) NOT NULL,
    compliance_status VARCHAR(20) NOT NULL,
    evidence_refs JSONB DEFAULT '[]',
    coverage_pct DECIMAL(5,2),
    gap_description TEXT,
    recommendation TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at),
    CONSTRAINT chk_atl_ct_year CHECK (reporting_year >= 2000 AND reporting_year <= 2100),
    CONSTRAINT chk_atl_ct_status CHECK (compliance_status IN ('compliant', 'partial', 'non_compliant', 'not_applicable', 'pending_review')),
    CONSTRAINT chk_atl_ct_coverage CHECK (coverage_pct IS NULL OR (coverage_pct >= 0 AND coverage_pct <= 100))
);

SELECT create_hypertable(
    'audit_trail_lineage_service.gl_atl_compliance_traces',
    'created_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

CREATE INDEX idx_atl_ct_trace_id ON audit_trail_lineage_service.gl_atl_compliance_traces(trace_id, created_at DESC);
CREATE INDEX idx_atl_ct_fw_org_year ON audit_trail_lineage_service.gl_atl_compliance_traces(framework, organization_id, reporting_year, created_at DESC);
CREATE INDEX idx_atl_ct_req_id ON audit_trail_lineage_service.gl_atl_compliance_traces(requirement_id, created_at DESC);
CREATE INDEX idx_atl_ct_status ON audit_trail_lineage_service.gl_atl_compliance_traces(compliance_status, created_at DESC);
CREATE INDEX idx_atl_ct_coverage ON audit_trail_lineage_service.gl_atl_compliance_traces(coverage_pct, created_at DESC) WHERE coverage_pct IS NOT NULL;
CREATE INDEX idx_atl_ct_evidence ON audit_trail_lineage_service.gl_atl_compliance_traces USING GIN (evidence_refs);
CREATE INDEX idx_atl_ct_metadata ON audit_trail_lineage_service.gl_atl_compliance_traces USING GIN (metadata);

COMMENT ON TABLE audit_trail_lineage_service.gl_atl_compliance_traces IS 'Per-requirement compliance trace records (TimescaleDB hypertable, 30-day chunks, 5-year retention)';
COMMENT ON COLUMN audit_trail_lineage_service.gl_atl_compliance_traces.coverage_pct IS 'Percentage of requirement covered by available evidence (0-100)';

-- =====================================================================================
-- TABLE 10: gl_atl_change_events (Operational)
-- Description: Material change events requiring recalculation assessment
-- =====================================================================================

CREATE TABLE audit_trail_lineage_service.gl_atl_change_events (
    id BIGSERIAL PRIMARY KEY,
    change_id UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    change_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    affected_entity_type VARCHAR(50) NOT NULL,
    affected_entity_id UUID NOT NULL,
    old_value JSONB,
    new_value JSONB,
    trigger VARCHAR(50) NOT NULL,
    materiality_pct DECIMAL(8,4),
    affected_calculations_count INT DEFAULT 0,
    recalculation_required BOOLEAN DEFAULT FALSE,
    recalculation_status VARCHAR(20) DEFAULT 'pending',
    organization_id UUID NOT NULL,
    reporting_year SMALLINT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_atl_ce_year CHECK (reporting_year >= 2000 AND reporting_year <= 2100),
    CONSTRAINT chk_atl_ce_severity CHECK (severity IN ('info', 'low', 'medium', 'high', 'critical')),
    CONSTRAINT chk_atl_ce_trigger CHECK (trigger IN (
        'user_edit', 'data_refresh', 'ef_update', 'methodology_change',
        'scope_change', 'restatement', 'correction', 'system_migration',
        'regulatory_update', 'api_sync'
    )),
    CONSTRAINT chk_atl_ce_recalc_status CHECK (recalculation_status IN (
        'pending', 'queued', 'in_progress', 'completed', 'skipped', 'failed'
    )),
    CONSTRAINT chk_atl_ce_entity_type CHECK (affected_entity_type IN (
        'emission_factor', 'activity_data', 'calculation', 'scope_boundary',
        'methodology', 'allocation_factor', 'organization_structure',
        'reporting_period', 'data_source', 'assumption'
    ))
);

CREATE INDEX idx_atl_ce_change_type ON audit_trail_lineage_service.gl_atl_change_events(change_type);
CREATE INDEX idx_atl_ce_severity ON audit_trail_lineage_service.gl_atl_change_events(severity);
CREATE INDEX idx_atl_ce_entity ON audit_trail_lineage_service.gl_atl_change_events(affected_entity_type, affected_entity_id);
CREATE INDEX idx_atl_ce_org_year ON audit_trail_lineage_service.gl_atl_change_events(organization_id, reporting_year);
CREATE INDEX idx_atl_ce_recalc_status ON audit_trail_lineage_service.gl_atl_change_events(recalculation_status) WHERE recalculation_required = TRUE;
CREATE INDEX idx_atl_ce_trigger ON audit_trail_lineage_service.gl_atl_change_events(trigger);
CREATE INDEX idx_atl_ce_materiality ON audit_trail_lineage_service.gl_atl_change_events(materiality_pct DESC) WHERE materiality_pct IS NOT NULL;
CREATE INDEX idx_atl_ce_created ON audit_trail_lineage_service.gl_atl_change_events(created_at DESC);
CREATE INDEX idx_atl_ce_metadata ON audit_trail_lineage_service.gl_atl_change_events USING GIN (metadata);

COMMENT ON TABLE audit_trail_lineage_service.gl_atl_change_events IS 'Material change events with recalculation impact assessment and cascading dependency tracking';
COMMENT ON COLUMN audit_trail_lineage_service.gl_atl_change_events.materiality_pct IS 'Percentage impact on total emissions (absolute value)';
COMMENT ON COLUMN audit_trail_lineage_service.gl_atl_change_events.recalculation_status IS 'Status of downstream recalculation cascade';

-- =====================================================================================
-- TABLE 11: gl_atl_chain_verifications (Supporting)
-- Description: Hash chain integrity verification records
-- =====================================================================================

CREATE TABLE audit_trail_lineage_service.gl_atl_chain_verifications (
    id BIGSERIAL PRIMARY KEY,
    verification_id UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    organization_id UUID NOT NULL,
    reporting_year SMALLINT NOT NULL,
    chain_status VARCHAR(20) NOT NULL,
    total_events BIGINT NOT NULL,
    verified_events BIGINT NOT NULL,
    break_points JSONB DEFAULT '[]',
    verification_hash VARCHAR(64) NOT NULL,
    verified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    CONSTRAINT chk_atl_cv_year CHECK (reporting_year >= 2000 AND reporting_year <= 2100),
    CONSTRAINT chk_atl_cv_status CHECK (chain_status IN ('intact', 'broken', 'partial', 'unverified')),
    CONSTRAINT chk_atl_cv_events CHECK (verified_events <= total_events),
    CONSTRAINT chk_atl_cv_hash_len CHECK (char_length(verification_hash) = 64)
);

CREATE INDEX idx_atl_cv_org_year ON audit_trail_lineage_service.gl_atl_chain_verifications(organization_id, reporting_year);
CREATE INDEX idx_atl_cv_status ON audit_trail_lineage_service.gl_atl_chain_verifications(chain_status);
CREATE INDEX idx_atl_cv_verified_at ON audit_trail_lineage_service.gl_atl_chain_verifications(verified_at DESC);
CREATE INDEX idx_atl_cv_break ON audit_trail_lineage_service.gl_atl_chain_verifications USING GIN (break_points) WHERE chain_status = 'broken';

COMMENT ON TABLE audit_trail_lineage_service.gl_atl_chain_verifications IS 'Hash chain integrity verification records for audit event chains';
COMMENT ON COLUMN audit_trail_lineage_service.gl_atl_chain_verifications.break_points IS 'JSON array of chain_position values where hash mismatches were detected';

-- =====================================================================================
-- TABLE 12: gl_atl_audit_trail_summaries (Supporting)
-- Description: Pre-computed organization-level audit trail summary statistics
-- =====================================================================================

CREATE TABLE audit_trail_lineage_service.gl_atl_audit_trail_summaries (
    id BIGSERIAL PRIMARY KEY,
    summary_id UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    organization_id UUID NOT NULL,
    reporting_year SMALLINT NOT NULL,
    total_events BIGINT DEFAULT 0,
    events_by_type JSONB DEFAULT '{}',
    events_by_scope JSONB DEFAULT '{}',
    chain_integrity VARCHAR(20) DEFAULT 'unverified',
    lineage_node_count BIGINT DEFAULT 0,
    lineage_edge_count BIGINT DEFAULT 0,
    lineage_max_depth INT DEFAULT 0,
    compliance_coverage JSONB DEFAULT '{}',
    evidence_package_count INT DEFAULT 0,
    change_event_count INT DEFAULT 0,
    last_event_at TIMESTAMPTZ,
    last_verification_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(organization_id, reporting_year),
    CONSTRAINT chk_atl_as_year CHECK (reporting_year >= 2000 AND reporting_year <= 2100),
    CONSTRAINT chk_atl_as_chain CHECK (chain_integrity IN ('intact', 'broken', 'partial', 'unverified')),
    CONSTRAINT chk_atl_as_depth CHECK (lineage_max_depth >= 0 AND lineage_max_depth <= 100)
);

CREATE INDEX idx_atl_as_chain ON audit_trail_lineage_service.gl_atl_audit_trail_summaries(chain_integrity);
CREATE INDEX idx_atl_as_updated ON audit_trail_lineage_service.gl_atl_audit_trail_summaries(updated_at DESC);
CREATE INDEX idx_atl_as_events_type ON audit_trail_lineage_service.gl_atl_audit_trail_summaries USING GIN (events_by_type);
CREATE INDEX idx_atl_as_events_scope ON audit_trail_lineage_service.gl_atl_audit_trail_summaries USING GIN (events_by_scope);
CREATE INDEX idx_atl_as_compliance ON audit_trail_lineage_service.gl_atl_audit_trail_summaries USING GIN (compliance_coverage);

COMMENT ON TABLE audit_trail_lineage_service.gl_atl_audit_trail_summaries IS 'Pre-computed organization-level audit trail summary statistics (one row per org per year)';
COMMENT ON COLUMN audit_trail_lineage_service.gl_atl_audit_trail_summaries.lineage_max_depth IS 'Maximum depth of the lineage DAG (longest path from L0 to L7)';

-- =====================================================================================
-- COMPRESSION POLICIES
-- =====================================================================================

ALTER TABLE audit_trail_lineage_service.gl_atl_audit_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'organization_id',
    timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_compression_policy(
    'audit_trail_lineage_service.gl_atl_audit_events',
    INTERVAL '90 days',
    if_not_exists => TRUE
);

ALTER TABLE audit_trail_lineage_service.gl_atl_evidence_packages SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'organization_id',
    timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_compression_policy(
    'audit_trail_lineage_service.gl_atl_evidence_packages',
    INTERVAL '180 days',
    if_not_exists => TRUE
);

ALTER TABLE audit_trail_lineage_service.gl_atl_compliance_traces SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'organization_id',
    timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_compression_policy(
    'audit_trail_lineage_service.gl_atl_compliance_traces',
    INTERVAL '90 days',
    if_not_exists => TRUE
);

-- =====================================================================================
-- RETENTION POLICIES
-- =====================================================================================

SELECT add_retention_policy(
    'audit_trail_lineage_service.gl_atl_audit_events',
    INTERVAL '7 years',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'audit_trail_lineage_service.gl_atl_evidence_packages',
    INTERVAL '10 years',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'audit_trail_lineage_service.gl_atl_compliance_traces',
    INTERVAL '5 years',
    if_not_exists => TRUE
);

-- =====================================================================================
-- CONTINUOUS AGGREGATES
-- =====================================================================================

-- Daily audit event statistics
CREATE MATERIALIZED VIEW audit_trail_lineage_service.gl_atl_daily_event_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', created_at) AS bucket,
    event_type,
    scope,
    agent_id,
    COUNT(*) AS total_events,
    COUNT(DISTINCT calculation_id) AS distinct_calculations,
    AVG(data_quality_score) AS avg_data_quality,
    MIN(data_quality_score) AS min_data_quality,
    MAX(data_quality_score) AS max_data_quality,
    COUNT(*) FILTER (WHERE data_quality_score >= 0.75) AS high_quality_count,
    COUNT(*) FILTER (WHERE data_quality_score < 0.5) AS low_quality_count,
    MAX(chain_position) AS max_chain_position
FROM audit_trail_lineage_service.gl_atl_audit_events
GROUP BY bucket, event_type, scope, agent_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'audit_trail_lineage_service.gl_atl_daily_event_stats',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Monthly compliance coverage summary
CREATE MATERIALIZED VIEW audit_trail_lineage_service.gl_atl_monthly_compliance_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 month', created_at) AS bucket,
    framework,
    organization_id,
    reporting_year,
    COUNT(*) AS total_traces,
    COUNT(*) FILTER (WHERE compliance_status = 'compliant') AS compliant_count,
    COUNT(*) FILTER (WHERE compliance_status = 'partial') AS partial_count,
    COUNT(*) FILTER (WHERE compliance_status = 'non_compliant') AS non_compliant_count,
    COUNT(*) FILTER (WHERE compliance_status = 'not_applicable') AS not_applicable_count,
    COUNT(*) FILTER (WHERE compliance_status = 'pending_review') AS pending_count,
    AVG(coverage_pct) AS avg_coverage_pct,
    MIN(coverage_pct) AS min_coverage_pct,
    MAX(coverage_pct) AS max_coverage_pct
FROM audit_trail_lineage_service.gl_atl_compliance_traces
GROUP BY bucket, framework, organization_id, reporting_year
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'audit_trail_lineage_service.gl_atl_monthly_compliance_summary',
    start_offset => INTERVAL '3 months',
    end_offset => INTERVAL '1 month',
    schedule_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- =====================================================================================
-- SEED DATA: Event Types (12 types)
-- =====================================================================================

INSERT INTO audit_trail_lineage_service.gl_atl_event_types (event_type, description, severity, category, requires_lineage) VALUES
('data_ingested',        'Raw data ingested from external source (ERP, CSV, API, PDF)',      'info',     'data_intake',        TRUE),
('data_normalized',      'Data normalized to standard units and formats',                     'info',     'data_intake',        TRUE),
('data_validated',       'Data passed validation rules and quality checks',                   'low',      'validation',         TRUE),
('calculation_executed', 'Emission calculation executed by MRV agent',                        'medium',   'calculation',        TRUE),
('aggregation_computed', 'Emissions aggregated across scopes, categories, or periods',        'medium',   'calculation',        TRUE),
('compliance_checked',   'Compliance checked against regulatory framework requirement',       'medium',   'compliance',         FALSE),
('report_generated',     'Regulatory report or disclosure generated',                         'high',     'reporting',          TRUE),
('evidence_packaged',    'Evidence package assembled for assurance engagement',                'high',     'evidence',           FALSE),
('chain_verified',       'Hash chain integrity verified for organization audit trail',        'critical', 'lineage',            FALSE),
('change_detected',      'Material change detected requiring recalculation assessment',       'high',     'change_management',  TRUE),
('approval_granted',     'Data, calculation, or report approved by authorized reviewer',      'medium',   'approval',           FALSE),
('restatement_applied',  'Historical data or calculation restated due to material correction', 'critical', 'change_management', TRUE);

-- =====================================================================================
-- SEED DATA: Lineage Node Types (10 types)
-- =====================================================================================

INSERT INTO audit_trail_lineage_service.gl_atl_lineage_node_types (node_type, description, level_range, icon) VALUES
('raw_source',         'Raw external data source (ERP export, invoice PDF, sensor reading)',            'L0_source',      'database'),
('ingested_record',    'Ingested and parsed data record from intake agent',                             'L1_intake',       'download'),
('normalized_value',   'Unit-normalized and format-standardized value',                                 'L2_normalized',   'transform'),
('validated_datum',    'Value that passed all validation rules and quality checks',                     'L3_validated',    'check'),
('calculated_emission','Emission value calculated by deterministic formula engine',                     'L4_calculated',   'calculator'),
('aggregated_total',   'Aggregated emission total (scope, category, facility, period)',                 'L5_aggregated',   'layers'),
('reported_disclosure','Disclosure data point mapped to regulatory framework requirement',              'L6_reported',     'file-text'),
('assured_value',      'Value verified through limited or reasonable assurance engagement',             'L7_assured',      'shield'),
('emission_factor',    'Emission factor reference value from authoritative source',                     'mixed',           'book'),
('assumption',         'Assumption or proxy value used where primary data unavailable',                 'mixed',           'alert');

-- =====================================================================================
-- SEED DATA: Change Type Definitions (8 types)
-- =====================================================================================

INSERT INTO audit_trail_lineage_service.gl_atl_change_type_definitions (change_type, description, default_severity, requires_recalculation, notification_required) VALUES
('emission_factor_update',     'Emission factor database updated to newer version (e.g., DEFRA 2025 to 2026)',       'high',     TRUE,  TRUE),
('activity_data_correction',   'Activity data corrected due to measurement error or data entry mistake',             'high',     TRUE,  TRUE),
('methodology_change',         'Calculation methodology changed (e.g., spend-based to supplier-specific)',           'critical', TRUE,  TRUE),
('scope_boundary_change',      'Organizational or operational boundary change affecting scope definitions',          'critical', TRUE,  TRUE),
('data_source_change',         'Primary data source replaced or supplemented with new source',                       'medium',   TRUE,  TRUE),
('allocation_factor_update',   'Allocation factor recalculated (e.g., revenue share, floor area, headcount)',        'medium',   TRUE,  TRUE),
('structural_reorganization',  'Organization restructured affecting facility, subsidiary, or JV boundaries',         'critical', TRUE,  TRUE),
('restatement_trigger',        'Material error identified requiring historical period restatement',                   'critical', TRUE,  TRUE);

-- =====================================================================================
-- SEED DATA: Framework Requirements - GHG Protocol (~30 requirements)
-- =====================================================================================

INSERT INTO audit_trail_lineage_service.gl_atl_framework_requirements (framework, framework_version, requirement_id, requirement_text, section_reference, data_points_required, evidence_types, assurance_level, is_mandatory, effective_date) VALUES
('ghg_protocol', '2004-rev', 'GHG-001', 'Document organizational boundaries and consolidation approach (equity share or control)', 'Ch. 3', '["org_boundary","consolidation_approach"]', '["document","calculation"]', 'limited', TRUE, '2004-03-01'),
('ghg_protocol', '2004-rev', 'GHG-002', 'Document operational boundaries separating Scope 1, 2, and 3 emissions', 'Ch. 4', '["scope1_boundary","scope2_boundary","scope3_boundary"]', '["document","calculation"]', 'limited', TRUE, '2004-03-01'),
('ghg_protocol', '2004-rev', 'GHG-003', 'Report Scope 1 emissions separately from Scope 2 emissions', 'Ch. 4', '["scope1_total","scope2_total"]', '["calculation","measurement"]', 'limited', TRUE, '2004-03-01'),
('ghg_protocol', '2004-rev', 'GHG-004', 'Choose and document base year and base year recalculation policy', 'Ch. 5', '["base_year","recalc_policy","recalc_threshold"]', '["document"]', 'limited', TRUE, '2004-03-01'),
('ghg_protocol', '2004-rev', 'GHG-005', 'Track and document emissions over time using consistent methodology', 'Ch. 5', '["annual_emissions_series","methodology_consistency"]', '["calculation","document"]', 'limited', TRUE, '2004-03-01'),
('ghg_protocol', '2004-rev', 'GHG-006', 'Identify and report GHGs: CO2, CH4, N2O, HFCs, PFCs, SF6, NF3', 'Ch. 6', '["ghg_species_reported","gwp_source"]', '["calculation","measurement"]', 'limited', TRUE, '2004-03-01'),
('ghg_protocol', '2004-rev', 'GHG-007', 'Document calculation methodologies for each emission source', 'Ch. 6', '["methodology_by_source","emission_factors_used"]', '["calculation","document"]', 'limited', TRUE, '2004-03-01'),
('ghg_protocol', '2004-rev', 'GHG-008', 'Document emission factors and their sources', 'Ch. 6', '["ef_database","ef_source","ef_year"]', '["document","reference"]', 'limited', TRUE, '2004-03-01'),
('ghg_protocol', '2004-rev', 'GHG-009', 'Document global warming potential (GWP) values used and their source (IPCC AR)', 'Ch. 6', '["gwp_values","gwp_source_ar"]', '["document","reference"]', 'limited', TRUE, '2004-03-01'),
('ghg_protocol', '2004-rev', 'GHG-010', 'Report emissions in metric tonnes of CO2 equivalent (tCO2e)', 'Ch. 6', '["total_tco2e"]', '["calculation"]', 'limited', TRUE, '2004-03-01'),
('ghg_protocol', '2004-rev', 'GHG-011', 'Report Scope 2 emissions using both location-based and market-based methods', 'Scope 2 Guidance', '["scope2_location","scope2_market"]', '["calculation"]', 'limited', TRUE, '2015-01-01'),
('ghg_protocol', '2004-rev', 'GHG-012', 'Document data management plan including roles, data flow, and QA/QC procedures', 'Ch. 7', '["data_mgmt_plan","qaqc_procedures"]', '["document"]', 'limited', TRUE, '2004-03-01'),
('ghg_protocol', '2004-rev', 'GHG-013', 'Perform uncertainty assessment and document data quality indicators', 'Ch. 7', '["uncertainty_assessment","data_quality_scores"]', '["calculation","document"]', 'limited', FALSE, '2004-03-01'),
('ghg_protocol', '2004-rev', 'GHG-014', 'Document any exclusions and justify their insignificance', 'Ch. 8', '["exclusions_list","exclusion_justifications"]', '["document"]', 'limited', TRUE, '2004-03-01'),
('ghg_protocol', '2004-rev', 'GHG-015', 'Obtain third-party verification if claiming external assurance', 'Ch. 10', '["verification_statement","assurance_level"]', '["document","assurance"]', 'reasonable', FALSE, '2004-03-01'),
('ghg_protocol', 'Scope3-v1', 'GHG-S3-001', 'Screen all 15 Scope 3 categories for relevance', 'Ch. 6', '["category_screening_results"]', '["calculation","document"]', 'limited', TRUE, '2011-10-01'),
('ghg_protocol', 'Scope3-v1', 'GHG-S3-002', 'Report Scope 3 emissions for all relevant categories', 'Ch. 6', '["scope3_by_category"]', '["calculation"]', 'limited', TRUE, '2011-10-01'),
('ghg_protocol', 'Scope3-v1', 'GHG-S3-003', 'Document calculation methodology for each Scope 3 category', 'Ch. 7', '["methodology_per_category"]', '["calculation","document"]', 'limited', TRUE, '2011-10-01'),
('ghg_protocol', 'Scope3-v1', 'GHG-S3-004', 'Document data sources and data quality for each Scope 3 category', 'Ch. 7', '["data_sources_per_category","dq_per_category"]', '["document"]', 'limited', TRUE, '2011-10-01'),
('ghg_protocol', 'Scope3-v1', 'GHG-S3-005', 'Avoid double counting between Scope 3 categories', 'Ch. 8', '["double_counting_checks"]', '["calculation","document"]', 'limited', TRUE, '2011-10-01'),
('ghg_protocol', 'Scope3-v1', 'GHG-S3-006', 'Report biogenic CO2 emissions separately from the scopes total', 'Ch. 9', '["biogenic_co2_total"]', '["calculation"]', 'limited', TRUE, '2011-10-01'),
('ghg_protocol', 'Scope3-v1', 'GHG-S3-007', 'Track Scope 3 emissions over time and report trends', 'Ch. 10', '["scope3_time_series"]', '["calculation"]', 'limited', TRUE, '2011-10-01'),
('ghg_protocol', 'Scope3-v1', 'GHG-S3-008', 'Document supplier engagement strategy for improving data quality', 'Ch. 11', '["supplier_engagement_plan"]', '["document"]', 'limited', FALSE, '2011-10-01'),
('ghg_protocol', 'Scope3-v1', 'GHG-S3-009', 'Report percentage of Scope 3 emissions calculated using primary vs secondary data', 'Ch. 7', '["primary_data_pct","secondary_data_pct"]', '["calculation"]', 'limited', TRUE, '2011-10-01'),
('ghg_protocol', 'Scope3-v1', 'GHG-S3-010', 'Disclose any Scope 3 categories excluded with justification', 'Ch. 6', '["excluded_categories","exclusion_reasons"]', '["document"]', 'limited', TRUE, '2011-10-01'),
('ghg_protocol', 'Scope3-v1', 'GHG-S3-011', 'Report upstream and downstream emissions separately', 'Ch. 5', '["upstream_total","downstream_total"]', '["calculation"]', 'limited', TRUE, '2011-10-01'),
('ghg_protocol', 'Scope3-v1', 'GHG-S3-012', 'Document allocation methods for shared transportation and facilities', 'Ch. 8', '["allocation_methods","allocation_basis"]', '["document","calculation"]', 'limited', TRUE, '2011-10-01'),
('ghg_protocol', 'Scope3-v1', 'GHG-S3-013', 'Identify and manage double counting across Scope 1/2/3 boundaries', 'Ch. 8', '["scope_boundary_overlaps"]', '["document"]', 'limited', TRUE, '2011-10-01'),
('ghg_protocol', 'Scope3-v1', 'GHG-S3-014', 'Use Scope 3 Evaluator or equivalent screening tool for materiality assessment', 'Ch. 6', '["screening_tool","materiality_results"]', '["calculation","document"]', 'limited', FALSE, '2011-10-01'),
('ghg_protocol', 'Scope3-v1', 'GHG-S3-015', 'Report Scope 3 data quality improvement roadmap', 'Ch. 11', '["dq_improvement_roadmap"]', '["document"]', 'limited', FALSE, '2011-10-01');

-- =====================================================================================
-- SEED DATA: Framework Requirements - ISO 14064 (~25 requirements)
-- =====================================================================================

INSERT INTO audit_trail_lineage_service.gl_atl_framework_requirements (framework, framework_version, requirement_id, requirement_text, section_reference, data_points_required, evidence_types, assurance_level, is_mandatory, effective_date) VALUES
('iso_14064', '2018-Part1', 'ISO-001', 'Define organizational boundaries using equity share or control approach', '5.1', '["org_boundary","approach"]', '["document"]', 'limited', TRUE, '2018-12-01'),
('iso_14064', '2018-Part1', 'ISO-002', 'Identify and document all direct GHG emission sources (Scope 1)', '5.2.2', '["direct_sources_inventory"]', '["document","measurement"]', 'limited', TRUE, '2018-12-01'),
('iso_14064', '2018-Part1', 'ISO-003', 'Identify and document all energy indirect GHG emissions (Scope 2)', '5.2.3', '["energy_indirect_sources"]', '["document","calculation"]', 'limited', TRUE, '2018-12-01'),
('iso_14064', '2018-Part1', 'ISO-004', 'Identify and document other indirect GHG emissions (Scope 3)', '5.2.4', '["other_indirect_sources"]', '["document","calculation"]', 'limited', TRUE, '2018-12-01'),
('iso_14064', '2018-Part1', 'ISO-005', 'Quantify GHG emissions using documented methodologies', '6.1', '["quantification_methodologies"]', '["calculation","measurement"]', 'limited', TRUE, '2018-12-01'),
('iso_14064', '2018-Part1', 'ISO-006', 'Select and justify emission factors from recognized sources', '6.2', '["ef_sources","ef_justification"]', '["document","reference"]', 'limited', TRUE, '2018-12-01'),
('iso_14064', '2018-Part1', 'ISO-007', 'Report emissions using CO2 equivalent with stated GWP values', '6.3', '["gwp_values","co2e_totals"]', '["calculation"]', 'limited', TRUE, '2018-12-01'),
('iso_14064', '2018-Part1', 'ISO-008', 'Establish a base year and document base year policy', '6.4', '["base_year","base_year_policy"]', '["document"]', 'limited', TRUE, '2018-12-01'),
('iso_14064', '2018-Part1', 'ISO-009', 'Assess and report uncertainty in GHG quantification', '6.5', '["uncertainty_assessment"]', '["calculation","document"]', 'limited', TRUE, '2018-12-01'),
('iso_14064', '2018-Part1', 'ISO-010', 'Develop and maintain a GHG inventory quality management plan', '7.1', '["quality_mgmt_plan"]', '["document"]', 'limited', TRUE, '2018-12-01'),
('iso_14064', '2018-Part1', 'ISO-011', 'Document QA/QC procedures for data collection and calculation', '7.2', '["qaqc_procedures"]', '["document"]', 'limited', TRUE, '2018-12-01'),
('iso_14064', '2018-Part1', 'ISO-012', 'Prepare GHG inventory report with all required elements', '8.1', '["inventory_report"]', '["document"]', 'limited', TRUE, '2018-12-01'),
('iso_14064', '2018-Part1', 'ISO-013', 'Include statement of completeness and materiality threshold', '8.2', '["completeness_statement","materiality_threshold"]', '["document"]', 'limited', TRUE, '2018-12-01'),
('iso_14064', '2018-Part1', 'ISO-014', 'Document any changes in quantification methodology year-over-year', '8.3', '["methodology_changes"]', '["document"]', 'limited', TRUE, '2018-12-01'),
('iso_14064', '2018-Part1', 'ISO-015', 'Maintain records to demonstrate conformity with ISO 14064-1', '9.1', '["conformity_records"]', '["document","assurance"]', 'limited', TRUE, '2018-12-01'),
('iso_14064', '2018-Part3', 'ISO-016', 'Establish verification objectives, scope, and criteria', '6.2', '["verification_scope","verification_criteria"]', '["assurance"]', 'reasonable', FALSE, '2019-04-01'),
('iso_14064', '2018-Part3', 'ISO-017', 'Plan verification activities including sampling strategy', '6.3', '["verification_plan","sampling_strategy"]', '["assurance"]', 'reasonable', FALSE, '2019-04-01'),
('iso_14064', '2018-Part3', 'ISO-018', 'Evaluate GHG information system and data management', '6.4', '["info_system_evaluation"]', '["assurance"]', 'reasonable', FALSE, '2019-04-01'),
('iso_14064', '2018-Part3', 'ISO-019', 'Assess materiality of identified discrepancies', '6.5', '["materiality_assessment"]', '["assurance"]', 'reasonable', FALSE, '2019-04-01'),
('iso_14064', '2018-Part3', 'ISO-020', 'Issue verification opinion (limited or reasonable assurance)', '7.1', '["verification_opinion","assurance_level"]', '["assurance"]', 'reasonable', FALSE, '2019-04-01'),
('iso_14064', '2018-Part3', 'ISO-021', 'Document evidence trail from raw data to reported value', '7.2', '["evidence_trail"]', '["document","assurance"]', 'reasonable', FALSE, '2019-04-01'),
('iso_14064', '2018-Part3', 'ISO-022', 'Report any qualifications, limitations, or caveats in verification statement', '7.3', '["qualifications","limitations"]', '["assurance"]', 'reasonable', FALSE, '2019-04-01'),
('iso_14064', '2018-Part3', 'ISO-023', 'Maintain verification records for at least 5 years', '8.1', '["verification_records_retention"]', '["document"]', 'limited', TRUE, '2019-04-01'),
('iso_14064', '2018-Part3', 'ISO-024', 'Ensure verifier independence and competence requirements met', '5.2', '["verifier_independence","verifier_competence"]', '["assurance"]', 'reasonable', FALSE, '2019-04-01'),
('iso_14064', '2018-Part3', 'ISO-025', 'Conduct data tracing from reported totals to source documents', '6.4.3', '["data_tracing_results"]', '["assurance"]', 'reasonable', FALSE, '2019-04-01'),
('iso_14064', '2018-Part1', 'ISO-026', 'Report emissions from biomass combustion separately', '6.3', '["biomass_emissions"]', '["calculation"]', 'limited', TRUE, '2018-12-01'),
('iso_14064', '2018-Part1', 'ISO-027', 'Document significance thresholds for emission source inclusion', '5.2.1', '["significance_thresholds"]', '["document"]', 'limited', TRUE, '2018-12-01'),
('iso_14064', '2018-Part1', 'ISO-028', 'Report on GHG removals and carbon storage separately from emissions', '6.6', '["ghg_removals","carbon_storage"]', '["calculation","document"]', 'limited', TRUE, '2018-12-01'),
('iso_14064', '2018-Part1', 'ISO-029', 'Document approach to handling data gaps and use of proxy data', '6.2', '["data_gap_approach","proxy_data_used"]', '["document"]', 'limited', TRUE, '2018-12-01'),
('iso_14064', '2018-Part1', 'ISO-030', 'Report on indirect emissions from transportation of products and materials', '5.2.4', '["transport_indirect_emissions"]', '["calculation"]', 'limited', TRUE, '2018-12-01');

-- =====================================================================================
-- SEED DATA: Framework Requirements - CSRD ESRS E1 (~30 requirements)
-- =====================================================================================

INSERT INTO audit_trail_lineage_service.gl_atl_framework_requirements (framework, framework_version, requirement_id, requirement_text, section_reference, data_points_required, evidence_types, assurance_level, is_mandatory, effective_date) VALUES
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-001', 'Disclose transition plan for climate change mitigation aligned with 1.5C', 'E1-1', '["transition_plan","target_year","interim_milestones"]', '["document","calculation"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-002', 'Disclose policies adopted to manage climate change impacts, risks, and opportunities', 'E1-2', '["climate_policies"]', '["document"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-003', 'Disclose actions and resources related to climate change policies and targets', 'E1-3', '["climate_actions","resource_allocation"]', '["document"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-004', 'Disclose GHG emission reduction targets (absolute and/or intensity)', 'E1-4', '["reduction_targets","target_type","target_year","base_year"]', '["calculation","document"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-005', 'Disclose energy consumption and energy mix', 'E1-5', '["energy_consumption_mwh","energy_mix_pct","renewable_share"]', '["calculation","measurement"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-006', 'Disclose gross Scope 1 GHG emissions', 'E1-6(a)', '["scope1_gross_tco2e","scope1_by_country"]', '["calculation","measurement"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-007', 'Disclose gross Scope 2 GHG emissions (location-based)', 'E1-6(b)', '["scope2_location_tco2e"]', '["calculation"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-008', 'Disclose gross Scope 2 GHG emissions (market-based)', 'E1-6(b)', '["scope2_market_tco2e"]', '["calculation"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-009', 'Disclose gross Scope 3 GHG emissions by significant category', 'E1-6(c)', '["scope3_by_category_tco2e","significant_categories"]', '["calculation"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-010', 'Disclose total GHG emissions (Scope 1+2+3)', 'E1-6(d)', '["total_tco2e","scope1_tco2e","scope2_tco2e","scope3_tco2e"]', '["calculation"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-011', 'Disclose GHG intensity per net revenue', 'E1-6(e)', '["ghg_intensity_per_revenue","net_revenue"]', '["calculation"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-012', 'Disclose GHG removals and carbon credits separately from gross emissions', 'E1-7', '["ghg_removals_tco2e","carbon_credits_tco2e"]', '["calculation","document"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-013', 'Disclose internal carbon pricing mechanisms', 'E1-8', '["internal_carbon_price","pricing_mechanism"]', '["document"]', 'limited', FALSE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-014', 'Disclose anticipated financial effects of climate risks and opportunities', 'E1-9', '["financial_effects_climate"]', '["calculation","document"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-015', 'Apply double materiality assessment for climate-related matters', 'ESRS 1', '["double_materiality_assessment"]', '["document"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-016', 'Provide data in machine-readable XBRL format per ESEF requirements', 'ESRS 1 App', '["xbrl_taxonomy","xbrl_filing"]', '["document"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-017', 'Describe GHG accounting policies and methodology applied', 'E1-6 AR', '["ghg_accounting_policies"]', '["document"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-018', 'Disclose data quality limitations and use of estimates', 'E1-6 AR', '["data_quality_assessment","estimate_methodologies"]', '["document"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-019', 'Report Scope 3 using GHG Protocol Scope 3 Standard categories', 'E1-6 AR', '["scope3_ghg_protocol_alignment"]', '["document"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-020', 'Disclose biogenic CO2 emissions and removals from land use', 'E1-6 AR', '["biogenic_co2_land_use"]', '["calculation"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-021', 'Disclose GWP values applied and IPCC Assessment Report version', 'E1-6 AR', '["gwp_values_used","ipcc_ar_version"]', '["document"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-022', 'Provide comparative year-on-year data where available', 'ESRS 1 6.5', '["prior_year_data","yoy_change"]', '["calculation"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-023', 'Disclose percentage of emissions covered by verified data vs estimates', 'E1-6 AR', '["verified_data_pct","estimated_data_pct"]', '["calculation"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-024', 'Subject sustainability report to limited assurance (increasing to reasonable)', 'Directive Art 34', '["assurance_report","assurance_provider"]', '["assurance"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-025', 'Align disclosures with TCFD recommendations', 'ESRS 1 App C', '["tcfd_alignment_mapping"]', '["document"]', 'limited', FALSE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-026', 'Map disclosures to UN SDGs where relevant', 'ESRS 1 App D', '["sdg_mapping"]', '["document"]', 'limited', FALSE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-027', 'Disclose breakdown of Scope 1 emissions by significant country', 'E1-6(a)', '["scope1_by_country"]', '["calculation"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-028', 'Report on EU Taxonomy alignment for climate change mitigation', 'Taxonomy Reg', '["taxonomy_eligible_pct","taxonomy_aligned_pct"]', '["calculation","document"]', 'limited', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-029', 'Maintain audit trail from raw data to disclosed values', 'ESRS 1 QC', '["audit_trail_completeness"]', '["document","assurance"]', 'reasonable', TRUE, '2024-01-01'),
('csrd_esrs_e1', '2023-v1', 'ESRS-E1-030', 'Document board oversight and governance of climate-related matters', 'ESRS 2 GOV', '["governance_climate"]', '["document"]', 'limited', TRUE, '2024-01-01');

-- =====================================================================================
-- SEED DATA: Framework Requirements - SB 253 (~20 requirements)
-- =====================================================================================

INSERT INTO audit_trail_lineage_service.gl_atl_framework_requirements (framework, framework_version, requirement_id, requirement_text, section_reference, data_points_required, evidence_types, assurance_level, is_mandatory, effective_date) VALUES
('sb_253', '2023-CA', 'SB253-001', 'Report Scope 1 emissions annually with third-party assurance', 'Sec 38532(b)', '["scope1_total_tco2e"]', '["calculation","assurance"]', 'limited', TRUE, '2026-01-01'),
('sb_253', '2023-CA', 'SB253-002', 'Report Scope 2 emissions annually with third-party assurance', 'Sec 38532(b)', '["scope2_total_tco2e"]', '["calculation","assurance"]', 'limited', TRUE, '2026-01-01'),
('sb_253', '2023-CA', 'SB253-003', 'Report Scope 3 emissions annually (180-day grace period)', 'Sec 38532(c)', '["scope3_total_tco2e","scope3_by_category"]', '["calculation"]', 'limited', TRUE, '2027-01-01'),
('sb_253', '2023-CA', 'SB253-004', 'Report in conformance with GHG Protocol Corporate Standard', 'Sec 38532(a)', '["ghg_protocol_conformance"]', '["document"]', 'limited', TRUE, '2026-01-01'),
('sb_253', '2023-CA', 'SB253-005', 'Obtain limited assurance for Scope 1 and 2 (increasing to reasonable)', 'Sec 38532(d)', '["assurance_statement","assurance_provider"]', '["assurance"]', 'limited', TRUE, '2026-01-01'),
('sb_253', '2023-CA', 'SB253-006', 'Report on a publicly accessible reporting platform (CARB)', 'Sec 38532(e)', '["carb_submission","public_availability"]', '["document"]', 'limited', TRUE, '2026-01-01'),
('sb_253', '2023-CA', 'SB253-007', 'Applicable to entities with >$1B annual revenue doing business in California', 'Sec 38532(a)', '["revenue_threshold","ca_nexus"]', '["document"]', 'none', TRUE, '2026-01-01'),
('sb_253', '2023-CA', 'SB253-008', 'Use consistent methodology year-over-year', 'Sec 38532(a)', '["methodology_consistency"]', '["document"]', 'limited', TRUE, '2026-01-01'),
('sb_253', '2023-CA', 'SB253-009', 'Report using IPCC AR5 or later GWP values', 'Sec 38532(a)', '["gwp_values","ipcc_ar"]', '["document"]', 'limited', TRUE, '2026-01-01'),
('sb_253', '2023-CA', 'SB253-010', 'Provide data disaggregated by emission source category', 'Sec 38532(b)', '["emissions_by_source"]', '["calculation"]', 'limited', TRUE, '2026-01-01'),
('sb_253', '2023-CA', 'SB253-011', 'Disclose any restatements from prior reporting periods', 'Sec 38532(f)', '["restatement_details"]', '["document"]', 'limited', TRUE, '2026-01-01'),
('sb_253', '2023-CA', 'SB253-012', 'Document estimation methodologies and assumptions for Scope 3', 'Sec 38532(c)', '["scope3_methodology","scope3_assumptions"]', '["document"]', 'limited', TRUE, '2027-01-01'),
('sb_253', '2023-CA', 'SB253-013', 'Report independently verified data from accredited assurance provider', 'Sec 38532(d)', '["verifier_accreditation"]', '["assurance"]', 'limited', TRUE, '2026-01-01'),
('sb_253', '2023-CA', 'SB253-014', 'Include materiality statement and completeness declaration', 'Sec 38532(g)', '["materiality_statement","completeness_declaration"]', '["document"]', 'limited', TRUE, '2026-01-01'),
('sb_253', '2023-CA', 'SB253-015', 'Maintain underlying data and records for minimum 5 years', 'Sec 38532(h)', '["records_retention_years"]', '["document"]', 'none', TRUE, '2026-01-01'),
('sb_253', '2023-CA', 'SB253-016', 'Report emissions in metric tonnes of CO2 equivalent', 'Sec 38532(a)', '["reporting_unit_tco2e"]', '["calculation"]', 'limited', TRUE, '2026-01-01'),
('sb_253', '2023-CA', 'SB253-017', 'Disclose organizational boundary and consolidation approach', 'Sec 38532(a)', '["org_boundary_sb253"]', '["document"]', 'limited', TRUE, '2026-01-01'),
('sb_253', '2023-CA', 'SB253-018', 'Report biogenic emissions separately', 'Sec 38532(a)', '["biogenic_emissions_separate"]', '["calculation"]', 'limited', TRUE, '2026-01-01'),
('sb_253', '2023-CA', 'SB253-019', 'Comply with CARB reporting regulations and deadlines', 'Sec 38532(i)', '["carb_compliance_status"]', '["document"]', 'none', TRUE, '2026-01-01'),
('sb_253', '2023-CA', 'SB253-020', 'Disclose safe harbor for Scope 3 good-faith estimates', 'Sec 38533', '["scope3_safe_harbor_statement"]', '["document"]', 'none', FALSE, '2027-01-01'),
('sb_253', '2023-CA', 'SB253-021', 'Disclose data gaps and estimation techniques used for Scope 3', 'Sec 38532(c)', '["scope3_data_gaps","estimation_techniques"]', '["document"]', 'limited', TRUE, '2027-01-01'),
('sb_253', '2023-CA', 'SB253-022', 'Ensure assurance provider meets CARB qualification requirements', 'Sec 38532(d)', '["assurance_provider_qualifications"]', '["assurance"]', 'limited', TRUE, '2026-01-01'),
('sb_253', '2023-CA', 'SB253-023', 'Report year-over-year change in absolute emissions and intensity', 'Sec 38532(b)', '["yoy_absolute_change","yoy_intensity_change"]', '["calculation"]', 'limited', TRUE, '2027-01-01'),
('sb_253', '2023-CA', 'SB253-024', 'Disclose use of offsets or carbon credits separately from emissions inventory', 'Sec 38532(j)', '["offset_disclosure","credits_retired"]', '["document"]', 'none', FALSE, '2026-01-01'),
('sb_253', '2023-CA', 'SB253-025', 'Provide emissions data at subsidiary or facility level where material', 'Sec 38532(k)', '["facility_level_emissions"]', '["calculation"]', 'limited', FALSE, '2027-01-01');

-- =====================================================================================
-- SEED DATA: Framework Requirements - CBAM (~20 requirements)
-- =====================================================================================

INSERT INTO audit_trail_lineage_service.gl_atl_framework_requirements (framework, framework_version, requirement_id, requirement_text, section_reference, data_points_required, evidence_types, assurance_level, is_mandatory, effective_date) VALUES
('cbam', '2023-EU', 'CBAM-001', 'Report embedded emissions in imported goods for CBAM-covered sectors', 'Art 6', '["embedded_emissions_per_product"]', '["calculation","measurement"]', 'limited', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-002', 'Calculate direct (Scope 1) emissions from production processes', 'Annex III', '["direct_production_emissions"]', '["calculation","measurement"]', 'limited', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-003', 'Calculate indirect (Scope 2) emissions from electricity consumed in production', 'Annex III', '["indirect_electricity_emissions"]', '["calculation"]', 'limited', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-004', 'Report at installation-level granularity for each exporting facility', 'Art 7', '["installation_id","installation_emissions"]', '["measurement","document"]', 'limited', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-005', 'Use actual emissions data or CBAM default values where actual unavailable', 'Annex III', '["data_source_type","default_values_used"]', '["calculation","document"]', 'limited', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-006', 'Cover CBAM goods: cement, iron/steel, aluminium, fertilizers, electricity, hydrogen', 'Annex I', '["cbam_goods_covered","cn_codes"]', '["document"]', 'none', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-007', 'Submit quarterly CBAM reports during transitional period (Oct 2023 - Dec 2025)', 'Art 35', '["quarterly_report"]', '["document"]', 'none', TRUE, '2023-10-01'),
('cbam', '2023-EU', 'CBAM-008', 'Purchase and surrender CBAM certificates corresponding to embedded emissions', 'Art 22', '["certificates_required","certificates_surrendered"]', '["document"]', 'limited', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-009', 'Deduct carbon price paid in country of origin from CBAM certificate obligation', 'Art 9', '["carbon_price_paid_origin","deduction_amount"]', '["document","calculation"]', 'limited', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-010', 'Apply monitoring methodology consistent with EU ETS benchmark approach', 'Annex III', '["monitoring_methodology"]', '["document"]', 'limited', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-011', 'Verify embedded emissions by accredited CBAM verifier', 'Art 8', '["verifier_accreditation","verification_statement"]', '["assurance"]', 'reasonable', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-012', 'Maintain records of all CBAM-relevant data for at least 4 years', 'Art 25', '["records_retention"]', '["document"]', 'none', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-013', 'Register as authorized CBAM declarant in Member State', 'Art 5', '["declarant_registration"]', '["document"]', 'none', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-014', 'Report emission factor source (actual installation data vs country/EU default)', 'Annex III', '["ef_source_per_installation"]', '["document"]', 'limited', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-015', 'Apply product-specific system boundary per CBAM implementing regulation', 'Annex III', '["system_boundary_per_product"]', '["document"]', 'limited', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-016', 'Report precursors emissions for complex goods (e.g., steel in vehicles)', 'Annex III', '["precursor_emissions"]', '["calculation"]', 'limited', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-017', 'Document free allocation equivalent for EU ETS benchmarking deduction', 'Art 31', '["free_allocation_equivalent"]', '["document","calculation"]', 'limited', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-018', 'Ensure data exchange between customs authorities and CBAM registry', 'Art 28', '["customs_data_exchange"]', '["document"]', 'none', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-019', 'Disclose any penalties or non-compliance events', 'Art 26', '["non_compliance_events"]', '["document"]', 'none', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-020', 'Provide product-level CN code-based emission intensity (tCO2e per tonne)', 'Annex III', '["emission_intensity_per_cn"]', '["calculation"]', 'limited', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-021', 'Document monitoring plan for each installation producing CBAM goods', 'Annex IV', '["monitoring_plan_per_installation"]', '["document"]', 'limited', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-022', 'Report on electricity consumed by production process and its carbon intensity', 'Annex III 4.3', '["electricity_consumed_mwh","grid_ef_origin"]', '["calculation","measurement"]', 'limited', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-023', 'Apply mass balance approach for complex goods with multiple production routes', 'Annex III 5', '["mass_balance_results"]', '["calculation"]', 'limited', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-024', 'Report on heat and steam consumed from external sources in production', 'Annex III 4.4', '["heat_consumed_gj","steam_consumed_gj"]', '["calculation","measurement"]', 'limited', TRUE, '2026-01-01'),
('cbam', '2023-EU', 'CBAM-025', 'Maintain chain of custody documentation for goods crossing EU border', 'Art 10', '["chain_of_custody_docs"]', '["document"]', 'none', TRUE, '2026-01-01');

-- =====================================================================================
-- SEED DATA: Framework Requirements - CDP (~25 requirements)
-- =====================================================================================

INSERT INTO audit_trail_lineage_service.gl_atl_framework_requirements (framework, framework_version, requirement_id, requirement_text, section_reference, data_points_required, evidence_types, assurance_level, is_mandatory, effective_date) VALUES
('cdp', '2024', 'CDP-001', 'Report Scope 1 emissions with breakdown by GHG type and source', 'C6.1', '["scope1_by_ghg","scope1_by_source"]', '["calculation"]', 'limited', TRUE, '2024-01-01'),
('cdp', '2024', 'CDP-002', 'Report Scope 2 emissions using both location-based and market-based methods', 'C6.3', '["scope2_location","scope2_market"]', '["calculation"]', 'limited', TRUE, '2024-01-01'),
('cdp', '2024', 'CDP-003', 'Report Scope 3 emissions for all evaluated categories', 'C6.5', '["scope3_by_category"]', '["calculation"]', 'limited', TRUE, '2024-01-01'),
('cdp', '2024', 'CDP-004', 'Describe methodology, assumptions, and allocation for each Scope 3 category', 'C6.5a', '["scope3_methodology_per_cat"]', '["document"]', 'limited', TRUE, '2024-01-01'),
('cdp', '2024', 'CDP-005', 'Report on climate-related governance including board oversight', 'C1.1', '["board_oversight_climate"]', '["document"]', 'none', TRUE, '2024-01-01'),
('cdp', '2024', 'CDP-006', 'Describe climate-related risks and opportunities with financial impact', 'C2.1', '["climate_risks","climate_opportunities"]', '["document"]', 'none', TRUE, '2024-01-01'),
('cdp', '2024', 'CDP-007', 'Report emission reduction targets (absolute and intensity)', 'C4.1', '["reduction_targets"]', '["calculation","document"]', 'limited', TRUE, '2024-01-01'),
('cdp', '2024', 'CDP-008', 'Report progress against emission reduction targets', 'C4.2', '["target_progress_pct"]', '["calculation"]', 'limited', TRUE, '2024-01-01'),
('cdp', '2024', 'CDP-009', 'Describe emissions reduction initiatives and estimated annual reductions', 'C4.3', '["reduction_initiatives","estimated_reductions"]', '["calculation","document"]', 'limited', TRUE, '2024-01-01'),
('cdp', '2024', 'CDP-010', 'Report energy consumption by type (fuel, electricity, heating, cooling, steam)', 'C8.2', '["energy_by_type_mwh"]', '["calculation","measurement"]', 'limited', TRUE, '2024-01-01'),
('cdp', '2024', 'CDP-011', 'Disclose verification/assurance status and attach assurance statement', 'C10.1', '["verification_status","assurance_statement"]', '["assurance"]', 'limited', FALSE, '2024-01-01'),
('cdp', '2024', 'CDP-012', 'Report on use of internal carbon pricing', 'C11.3', '["internal_carbon_price"]', '["document"]', 'none', FALSE, '2024-01-01'),
('cdp', '2024', 'CDP-013', 'Report emissions performance against target using consistent base year', 'C4.2', '["base_year_emissions","current_year_emissions"]', '["calculation"]', 'limited', TRUE, '2024-01-01'),
('cdp', '2024', 'CDP-014', 'Describe strategy for engaging suppliers on climate action', 'C12.1', '["supplier_engagement_strategy"]', '["document"]', 'none', TRUE, '2024-01-01'),
('cdp', '2024', 'CDP-015', 'Report percentage of supplier emissions data that is verified/measured', 'C6.5', '["supplier_data_quality_pct"]', '["calculation"]', 'limited', TRUE, '2024-01-01'),
('cdp', '2024', 'CDP-016', 'Disclose scenario analysis results (2C and below 2C)', 'C3.2', '["scenario_analysis_results"]', '["document","calculation"]', 'none', TRUE, '2024-01-01'),
('cdp', '2024', 'CDP-017', 'Report on water-related risks if applicable (climate nexus)', 'W1.1', '["water_risk_assessment"]', '["document"]', 'none', FALSE, '2024-01-01'),
('cdp', '2024', 'CDP-018', 'Disclose use of renewable energy and low-carbon instruments', 'C8.2e', '["renewable_energy_mwh","rec_purchases"]', '["calculation","document"]', 'limited', TRUE, '2024-01-01'),
('cdp', '2024', 'CDP-019', 'Report on climate-related lobbying and trade association alignment', 'C12.4', '["lobbying_activities","trade_association_alignment"]', '["document"]', 'none', FALSE, '2024-01-01'),
('cdp', '2024', 'CDP-020', 'Disclose biodiversity-related impacts linked to climate change', 'C15.1', '["biodiversity_climate_impacts"]', '["document"]', 'none', FALSE, '2024-01-01'),
('cdp', '2024', 'CDP-021', 'Report on physical climate risks and adaptation measures', 'C2.3', '["physical_risks","adaptation_measures"]', '["document"]', 'none', TRUE, '2024-01-01'),
('cdp', '2024', 'CDP-022', 'Report on transition risks and mitigation strategy', 'C2.4', '["transition_risks","mitigation_strategy"]', '["document"]', 'none', TRUE, '2024-01-01'),
('cdp', '2024', 'CDP-023', 'Disclose carbon credit usage and quality criteria', 'C11.2', '["carbon_credits_retired","credit_quality"]', '["document"]', 'limited', FALSE, '2024-01-01'),
('cdp', '2024', 'CDP-024', 'Align disclosure with ISSB/IFRS S2 where applicable', 'General', '["issb_alignment"]', '["document"]', 'none', FALSE, '2024-01-01'),
('cdp', '2024', 'CDP-025', 'Provide data with sufficient granularity for CDP scoring methodology', 'General', '["scoring_data_completeness"]', '["calculation","document"]', 'none', TRUE, '2024-01-01');

-- =====================================================================================
-- SEED DATA: Framework Requirements - TCFD (~15 requirements)
-- =====================================================================================

INSERT INTO audit_trail_lineage_service.gl_atl_framework_requirements (framework, framework_version, requirement_id, requirement_text, section_reference, data_points_required, evidence_types, assurance_level, is_mandatory, effective_date) VALUES
('tcfd', '2017-v1', 'TCFD-001', 'Disclose board oversight of climate-related risks and opportunities', 'Governance (a)', '["board_oversight_climate"]', '["document"]', 'none', TRUE, '2017-06-01'),
('tcfd', '2017-v1', 'TCFD-002', 'Disclose management role in assessing and managing climate-related risks', 'Governance (b)', '["mgmt_role_climate"]', '["document"]', 'none', TRUE, '2017-06-01'),
('tcfd', '2017-v1', 'TCFD-003', 'Describe climate-related risks and opportunities identified over short/medium/long term', 'Strategy (a)', '["climate_risks_by_horizon"]', '["document"]', 'none', TRUE, '2017-06-01'),
('tcfd', '2017-v1', 'TCFD-004', 'Describe impact of climate-related risks on strategy and financial planning', 'Strategy (b)', '["strategy_impact_climate"]', '["document"]', 'none', TRUE, '2017-06-01'),
('tcfd', '2017-v1', 'TCFD-005', 'Describe resilience of strategy under different climate scenarios (2C or lower)', 'Strategy (c)', '["scenario_analysis","strategy_resilience"]', '["document","calculation"]', 'none', TRUE, '2017-06-01'),
('tcfd', '2017-v1', 'TCFD-006', 'Describe processes for identifying and assessing climate-related risks', 'Risk Mgmt (a)', '["risk_identification_process"]', '["document"]', 'none', TRUE, '2017-06-01'),
('tcfd', '2017-v1', 'TCFD-007', 'Describe processes for managing climate-related risks', 'Risk Mgmt (b)', '["risk_management_process"]', '["document"]', 'none', TRUE, '2017-06-01'),
('tcfd', '2017-v1', 'TCFD-008', 'Describe integration of climate risk management into overall risk management', 'Risk Mgmt (c)', '["risk_integration"]', '["document"]', 'none', TRUE, '2017-06-01'),
('tcfd', '2017-v1', 'TCFD-009', 'Disclose Scope 1, 2 GHG emissions and related risks', 'Metrics (a)', '["scope1_tco2e","scope2_tco2e"]', '["calculation"]', 'limited', TRUE, '2017-06-01'),
('tcfd', '2017-v1', 'TCFD-010', 'Disclose Scope 3 GHG emissions if appropriate', 'Metrics (a)', '["scope3_tco2e"]', '["calculation"]', 'limited', FALSE, '2017-06-01'),
('tcfd', '2017-v1', 'TCFD-011', 'Describe targets used to manage climate-related risks and performance against targets', 'Metrics (b)', '["climate_targets","target_performance"]', '["calculation","document"]', 'limited', TRUE, '2017-06-01'),
('tcfd', '2017-v1', 'TCFD-012', 'Disclose metrics used to assess climate-related risks and opportunities', 'Metrics (c)', '["climate_metrics"]', '["calculation"]', 'none', TRUE, '2017-06-01'),
('tcfd', '2017-v1', 'TCFD-013', 'Use cross-industry metric categories where applicable', 'Metrics', '["cross_industry_metrics"]', '["calculation"]', 'none', TRUE, '2017-06-01'),
('tcfd', '2017-v1', 'TCFD-014', 'Report on transition and physical risk exposure in financial terms', 'Strategy', '["financial_risk_exposure"]', '["calculation","document"]', 'none', TRUE, '2017-06-01'),
('tcfd', '2017-v1', 'TCFD-015', 'Provide disclosures in mainstream financial filings', 'General', '["financial_filing_inclusion"]', '["document"]', 'none', TRUE, '2017-06-01'),
('tcfd', '2017-v1', 'TCFD-016', 'Describe how climate-related risks are factored into remuneration policies', 'Governance', '["remuneration_climate_link"]', '["document"]', 'none', FALSE, '2017-06-01'),
('tcfd', '2017-v1', 'TCFD-017', 'Disclose capital deployment and expenditure related to climate opportunities', 'Strategy', '["climate_capex","climate_opex"]', '["calculation","document"]', 'none', TRUE, '2017-06-01'),
('tcfd', '2017-v1', 'TCFD-018', 'Report on climate-related metrics in line with industry-specific guidance', 'Metrics', '["industry_specific_metrics"]', '["calculation"]', 'none', FALSE, '2017-06-01'),
('tcfd', '2017-v1', 'TCFD-019', 'Describe how climate risk assessment is integrated into enterprise risk management', 'Risk Mgmt', '["erm_integration_climate"]', '["document"]', 'none', TRUE, '2017-06-01'),
('tcfd', '2017-v1', 'TCFD-020', 'Disclose forward-looking metrics and assumptions underpinning scenario analysis', 'Strategy (c)', '["forward_looking_assumptions","scenario_parameters"]', '["calculation","document"]', 'none', TRUE, '2017-06-01');

-- =====================================================================================
-- SEED DATA: Framework Requirements - PCAF (~20 requirements)
-- =====================================================================================

INSERT INTO audit_trail_lineage_service.gl_atl_framework_requirements (framework, framework_version, requirement_id, requirement_text, section_reference, data_points_required, evidence_types, assurance_level, is_mandatory, effective_date) VALUES
('pcaf', '2022-v2', 'PCAF-001', 'Measure and disclose financed emissions for all in-scope asset classes', 'Part A 4.1', '["financed_emissions_by_asset_class"]', '["calculation"]', 'limited', TRUE, '2022-12-01'),
('pcaf', '2022-v2', 'PCAF-002', 'Use attribution factor based on outstanding amount and EVIC or total equity', 'Part A 4.2', '["attribution_factors","outstanding_amounts"]', '["calculation"]', 'limited', TRUE, '2022-12-01'),
('pcaf', '2022-v2', 'PCAF-003', 'Report data quality score (1-5) for each asset class using PCAF criteria', 'Part A 5.1', '["data_quality_scores_by_class"]', '["calculation"]', 'limited', TRUE, '2022-12-01'),
('pcaf', '2022-v2', 'PCAF-004', 'Calculate absolute financed emissions (tCO2e)', 'Part A 4.1', '["absolute_financed_tco2e"]', '["calculation"]', 'limited', TRUE, '2022-12-01'),
('pcaf', '2022-v2', 'PCAF-005', 'Calculate weighted average carbon intensity (WACI)', 'Part A 4.3', '["waci_tco2e_per_m_revenue"]', '["calculation"]', 'limited', TRUE, '2022-12-01'),
('pcaf', '2022-v2', 'PCAF-006', 'Report financed emissions for listed equity and corporate bonds', 'Part B 1', '["listed_equity_emissions","corp_bond_emissions"]', '["calculation"]', 'limited', TRUE, '2022-12-01'),
('pcaf', '2022-v2', 'PCAF-007', 'Report financed emissions for business loans and unlisted equity', 'Part B 2', '["business_loan_emissions","unlisted_equity_emissions"]', '["calculation"]', 'limited', TRUE, '2022-12-01'),
('pcaf', '2022-v2', 'PCAF-008', 'Report financed emissions for project finance', 'Part B 3', '["project_finance_emissions"]', '["calculation"]', 'limited', TRUE, '2022-12-01'),
('pcaf', '2022-v2', 'PCAF-009', 'Report financed emissions for commercial real estate', 'Part B 4', '["cre_emissions","building_eui"]', '["calculation"]', 'limited', TRUE, '2022-12-01'),
('pcaf', '2022-v2', 'PCAF-010', 'Report financed emissions for mortgages', 'Part B 5', '["mortgage_emissions","residential_eui"]', '["calculation"]', 'limited', TRUE, '2022-12-01'),
('pcaf', '2022-v2', 'PCAF-011', 'Report financed emissions for motor vehicle loans', 'Part B 6', '["motor_vehicle_emissions"]', '["calculation"]', 'limited', TRUE, '2022-12-01'),
('pcaf', '2022-v2', 'PCAF-012', 'Report financed emissions for sovereign bonds', 'Part B 7', '["sovereign_bond_emissions","country_attributions"]', '["calculation"]', 'limited', TRUE, '2022-12-01'),
('pcaf', '2022-v2', 'PCAF-013', 'Document data sources and limitations for each asset class', 'Part A 5.2', '["data_sources_per_class","data_limitations"]', '["document"]', 'limited', TRUE, '2022-12-01'),
('pcaf', '2022-v2', 'PCAF-014', 'Report emissions separately for Scope 1, 2, and 3 of investees', 'Part A 4.1', '["investee_scope1","investee_scope2","investee_scope3"]', '["calculation"]', 'limited', TRUE, '2022-12-01'),
('pcaf', '2022-v2', 'PCAF-015', 'Document improvement plan to increase data quality over time', 'Part A 5.3', '["dq_improvement_plan"]', '["document"]', 'none', TRUE, '2022-12-01'),
('pcaf', '2022-v2', 'PCAF-016', 'Disclose portfolio coverage (% of portfolio measured)', 'Part A 6.1', '["portfolio_coverage_pct"]', '["calculation"]', 'limited', TRUE, '2022-12-01'),
('pcaf', '2022-v2', 'PCAF-017', 'Align measurement with GHG Protocol for investee emissions', 'Part A 3', '["ghg_protocol_alignment"]', '["document"]', 'limited', TRUE, '2022-12-01'),
('pcaf', '2022-v2', 'PCAF-018', 'Report asset-class-level aggregation and total portfolio emissions', 'Part A 6.2', '["total_portfolio_emissions","asset_class_breakdown"]', '["calculation"]', 'limited', TRUE, '2022-12-01'),
('pcaf', '2022-v2', 'PCAF-019', 'Track year-on-year changes in financed emissions', 'Part A 6.3', '["yoy_financed_emissions"]', '["calculation"]', 'limited', TRUE, '2022-12-01'),
('pcaf', '2022-v2', 'PCAF-020', 'Support third-party verification of financed emissions calculations', 'Part A 7', '["verification_readiness"]', '["document","assurance"]', 'limited', FALSE, '2022-12-01');

-- =====================================================================================
-- SEED DATA: Framework Requirements - SBTi (~15 requirements)
-- =====================================================================================

INSERT INTO audit_trail_lineage_service.gl_atl_framework_requirements (framework, framework_version, requirement_id, requirement_text, section_reference, data_points_required, evidence_types, assurance_level, is_mandatory, effective_date) VALUES
('sbti', '5.1-2023', 'SBTi-001', 'Set near-term targets covering at least 95% of Scope 1 and 2 emissions', 'C6', '["scope1_coverage_pct","scope2_coverage_pct"]', '["calculation"]', 'limited', TRUE, '2023-07-01'),
('sbti', '5.1-2023', 'SBTi-002', 'Set Scope 3 target if Scope 3 is 40% or more of total Scope 1+2+3', 'C7', '["scope3_share_of_total","scope3_target"]', '["calculation"]', 'limited', TRUE, '2023-07-01'),
('sbti', '5.1-2023', 'SBTi-003', 'Use absolute or intensity pathway aligned with 1.5C for Scope 1 and 2', 'C8', '["pathway_type","temperature_alignment"]', '["calculation","document"]', 'limited', TRUE, '2023-07-01'),
('sbti', '5.1-2023', 'SBTi-004', 'Cover 67% of Scope 3 emissions in near-term Scope 3 targets', 'C7.1', '["scope3_target_coverage_pct"]', '["calculation"]', 'limited', TRUE, '2023-07-01'),
('sbti', '5.1-2023', 'SBTi-005', 'Set target timeframe of 5-10 years from submission date', 'C5', '["target_base_year","target_year"]', '["document"]', 'none', TRUE, '2023-07-01'),
('sbti', '5.1-2023', 'SBTi-006', 'Report progress annually against validated targets', 'C12', '["annual_progress_report","target_tracking"]', '["calculation"]', 'limited', TRUE, '2023-07-01'),
('sbti', '5.1-2023', 'SBTi-007', 'Complete full Scope 3 inventory (all 15 categories screened)', 'C7', '["scope3_full_inventory"]', '["calculation"]', 'limited', TRUE, '2023-07-01'),
('sbti', '5.1-2023', 'SBTi-008', 'Recalculate base year emissions for significant changes (>5% threshold)', 'C11', '["base_year_recalculation","significance_threshold"]', '["calculation","document"]', 'limited', TRUE, '2023-07-01'),
('sbti', '5.1-2023', 'SBTi-009', 'Do not use carbon credits toward target achievement (insetting allowed)', 'C13', '["carbon_credit_exclusion"]', '["document"]', 'none', TRUE, '2023-07-01'),
('sbti', '5.1-2023', 'SBTi-010', 'Use sector-specific pathways where available (e.g., SDA, ACA)', 'C8', '["sector_pathway","decarbonization_approach"]', '["calculation","document"]', 'limited', TRUE, '2023-07-01'),
('sbti', '5.1-2023', 'SBTi-011', 'Document emission reduction levers and implementation strategy', 'C9', '["reduction_levers","implementation_plan"]', '["document"]', 'none', TRUE, '2023-07-01'),
('sbti', '5.1-2023', 'SBTi-012', 'Set long-term net-zero targets (by 2050 at latest for 1.5C)', 'Net-Zero Standard', '["net_zero_target_year","residual_emissions_plan"]', '["document"]', 'none', FALSE, '2023-07-01'),
('sbti', '5.1-2023', 'SBTi-013', 'Report GHG inventory consistent with GHG Protocol standards', 'C4', '["ghg_protocol_conformance"]', '["calculation","document"]', 'limited', TRUE, '2023-07-01'),
('sbti', '5.1-2023', 'SBTi-014', 'Submit target validation data through SBTi online platform', 'C3', '["sbti_submission","validation_status"]', '["document"]', 'none', TRUE, '2023-07-01'),
('sbti', '5.1-2023', 'SBTi-015', 'Disclose annual inventory and target progress via CDP or equivalent', 'C12', '["cdp_disclosure","annual_inventory"]', '["calculation","document"]', 'limited', TRUE, '2023-07-01'),
('sbti', '5.1-2023', 'SBTi-016', 'Include FLAG (Forest, Land, and Agriculture) emissions in targets where applicable', 'FLAG Guidance', '["flag_emissions","flag_target"]', '["calculation","document"]', 'limited', FALSE, '2023-07-01'),
('sbti', '5.1-2023', 'SBTi-017', 'Report on avoided emissions separately from value chain inventory', 'C13', '["avoided_emissions_separate"]', '["calculation","document"]', 'none', TRUE, '2023-07-01'),
('sbti', '5.1-2023', 'SBTi-018', 'Maintain consistency in organizational boundary across reporting periods', 'C11', '["boundary_consistency"]', '["document"]', 'none', TRUE, '2023-07-01'),
('sbti', '5.1-2023', 'SBTi-019', 'Report on Scope 3 supplier engagement percentage where SBTi supplier engagement target set', 'C7.2', '["supplier_sbti_pct"]', '["calculation"]', 'limited', FALSE, '2023-07-01'),
('sbti', '5.1-2023', 'SBTi-020', 'Disclose any target revalidation events and updated base year inventory', 'C11', '["revalidation_events","updated_base_year"]', '["document","calculation"]', 'limited', TRUE, '2023-07-01');

-- =====================================================================================
-- GRANTS
-- =====================================================================================

GRANT USAGE ON SCHEMA audit_trail_lineage_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA audit_trail_lineage_service TO greenlang_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA audit_trail_lineage_service TO greenlang_app;
GRANT SELECT ON audit_trail_lineage_service.gl_atl_daily_event_stats TO greenlang_app;
GRANT SELECT ON audit_trail_lineage_service.gl_atl_monthly_compliance_summary TO greenlang_app;

-- Read-only access for reporting
GRANT USAGE ON SCHEMA audit_trail_lineage_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA audit_trail_lineage_service TO greenlang_readonly;
GRANT SELECT ON audit_trail_lineage_service.gl_atl_daily_event_stats TO greenlang_readonly;
GRANT SELECT ON audit_trail_lineage_service.gl_atl_monthly_compliance_summary TO greenlang_readonly;

-- =====================================================================================
-- MIGRATION METADATA
-- =====================================================================================

COMMENT ON SCHEMA audit_trail_lineage_service IS
'AGENT-MRV-030 (GL-MRV-X-042): Audit Trail & Lineage Agent cross-cutting service.
Tables: 12 (4 reference + 6 operational + 2 supporting)
Hypertables: 3 (audit_events 7d, evidence_packages 30d, compliance_traces 30d)
Continuous Aggregates: 2 (daily_event_stats, monthly_compliance_summary)
Frameworks: GHG Protocol (30), ISO 14064 (30), CSRD ESRS E1 (30), SB 253 (25),
            CBAM (25), CDP (25), TCFD (20), PCAF (20), SBTi (20) = 225 requirements
Seed data: 255 records (12 event types, 10 lineage node types, 8 change type definitions, 225 framework requirements)
Migration: V081 | Created: 2026-03-01';
