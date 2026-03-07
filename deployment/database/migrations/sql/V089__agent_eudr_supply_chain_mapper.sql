-- =============================================================================
-- V089: AGENT-EUDR-001 Supply Chain Mapping Master Schema
-- =============================================================================
-- Agent:       GL-EUDR-SCM-001 (Supply Chain Mapping Master Agent)
-- Date:        March 2026
--
-- Agent-level tables for the EUDR Supply Chain Mapping Master agent.
-- Provides graph-native supply chain modeling with multi-tier recursive
-- mapping, plot-to-product traceability, many-to-many batch topology,
-- risk propagation through supply chain graphs, gap analysis for EUDR
-- compliance, and versioned graph snapshots for audit trail.
--
-- Covers all 7 EUDR-regulated commodities (cattle, cocoa, coffee, palm oil,
-- rubber, soya, wood) and their derived products per Regulation (EU) 2023/1115.
-- Models actors (producers, collectors, processors, traders, importers) as
-- graph nodes, custody transfers as directed edges, with geolocation linkage
-- to production plots (Article 9), batch split/merge tracking, and
-- deterministic risk propagation across the full supply chain DAG.
--
-- EXTENDS:
--   V034: EUDR Traceability Connector Service (custody chains, DDS data)
--   V036: GIS/Mapping Connector Service (geolocation operations)
--   V037: Deforestation Satellite Connector Service (deforestation alerts)
--   V082: GL-EUDR-APP v1.0 (supplier profiles, plot registry, DDS lifecycle)
--
-- These tables sit in the eudr_supply_chain_mapper schema and integrate
-- with the underlying AGENT-DATA and GL-EUDR-APP schemas for enriched
-- supply chain graph construction, risk propagation, and gap analysis.
-- =============================================================================
-- Tables (6):
--   1.  supply_chain_graphs       - Graph metadata (one per operator/commodity)
--   2.  supply_chain_nodes        - Supply chain actors with geolocation
--   3.  supply_chain_edges        - Custody transfer edges between nodes
--   4.  gap_analysis_results      - Gap detection results (hypertable)
--   5.  risk_propagation_log      - Risk propagation audit log (hypertable)
--   6.  graph_snapshots           - Versioned graph snapshots (hypertable)
--
-- Hypertables (3):
--   gap_analysis_results     on detected_at    (chunk: 3 months)
--   risk_propagation_log     on calculated_at  (chunk: 3 months)
--   graph_snapshots          on created_at     (chunk: 3 months)
--
-- Continuous Aggregates (2):
--   eudr_supply_chain_mapper.daily_gap_summary         - Daily gap detection trends
--   eudr_supply_chain_mapper.monthly_risk_propagation   - Monthly risk propagation trends
--
-- Also includes: 60+ indexes (B-tree, GIN), update triggers, security
-- grants, retention policies, compression policies, permissions, and comments.
-- Previous: V088__taxonomy_app_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS eudr_supply_chain_mapper;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION eudr_supply_chain_mapper.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: eudr_supply_chain_mapper.supply_chain_graphs
-- =============================================================================
-- Main graph metadata table.  Each graph represents the full multi-tier
-- supply chain for a single operator and commodity combination.  Tracks
-- graph-level statistics (node/edge counts, max tier depth), compliance
-- readiness and traceability scores, risk summary, and version history
-- for deterministic reproducibility.

CREATE TABLE eudr_supply_chain_mapper.supply_chain_graphs (
    graph_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID,
    operator_id             UUID            NOT NULL,
    commodity               VARCHAR(50)     NOT NULL,
    graph_name              VARCHAR(500),
    description             TEXT,
    total_nodes             INTEGER         NOT NULL DEFAULT 0,
    total_edges             INTEGER         NOT NULL DEFAULT 0,
    max_tier_depth          INTEGER         NOT NULL DEFAULT 0,
    traceability_score      NUMERIC(5,2)    NOT NULL DEFAULT 0.0,
    compliance_readiness    NUMERIC(5,2)    NOT NULL DEFAULT 0.0,
    risk_summary            JSONB           DEFAULT '{}',
    graph_config            JSONB           DEFAULT '{}',
    version                 INTEGER         NOT NULL DEFAULT 1,
    status                  VARCHAR(30)     NOT NULL DEFAULT 'draft',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_scg_commodity CHECK (
        commodity IN (
            'cattle', 'cocoa', 'coffee', 'palm_oil',
            'rubber', 'soya', 'wood',
            'cattle_derived', 'cocoa_derived', 'coffee_derived',
            'palm_oil_derived', 'rubber_derived', 'soya_derived', 'wood_derived'
        )
    ),
    CONSTRAINT chk_scg_graph_name_not_empty CHECK (
        graph_name IS NULL OR LENGTH(TRIM(graph_name)) > 0
    ),
    CONSTRAINT chk_scg_total_nodes_non_neg CHECK (total_nodes >= 0),
    CONSTRAINT chk_scg_total_edges_non_neg CHECK (total_edges >= 0),
    CONSTRAINT chk_scg_max_tier_non_neg CHECK (max_tier_depth >= 0),
    CONSTRAINT chk_scg_traceability_range CHECK (
        traceability_score >= 0.0 AND traceability_score <= 100.0
    ),
    CONSTRAINT chk_scg_compliance_range CHECK (
        compliance_readiness >= 0.0 AND compliance_readiness <= 100.0
    ),
    CONSTRAINT chk_scg_version_positive CHECK (version >= 1),
    CONSTRAINT chk_scg_status CHECK (
        status IN ('draft', 'building', 'active', 'archived', 'error')
    )
);

-- Indexes
CREATE INDEX idx_scg_tenant ON eudr_supply_chain_mapper.supply_chain_graphs(tenant_id);
CREATE INDEX idx_scg_operator ON eudr_supply_chain_mapper.supply_chain_graphs(operator_id);
CREATE INDEX idx_scg_commodity ON eudr_supply_chain_mapper.supply_chain_graphs(commodity);
CREATE INDEX idx_scg_operator_commodity ON eudr_supply_chain_mapper.supply_chain_graphs(operator_id, commodity);
CREATE INDEX idx_scg_status ON eudr_supply_chain_mapper.supply_chain_graphs(status);
CREATE INDEX idx_scg_traceability ON eudr_supply_chain_mapper.supply_chain_graphs(traceability_score);
CREATE INDEX idx_scg_compliance ON eudr_supply_chain_mapper.supply_chain_graphs(compliance_readiness);
CREATE INDEX idx_scg_version ON eudr_supply_chain_mapper.supply_chain_graphs(version);
CREATE INDEX idx_scg_created_at ON eudr_supply_chain_mapper.supply_chain_graphs(created_at DESC);
CREATE INDEX idx_scg_updated_at ON eudr_supply_chain_mapper.supply_chain_graphs(updated_at DESC);
CREATE INDEX idx_scg_risk_summary ON eudr_supply_chain_mapper.supply_chain_graphs USING GIN(risk_summary);
CREATE INDEX idx_scg_graph_config ON eudr_supply_chain_mapper.supply_chain_graphs USING GIN(graph_config);
CREATE INDEX idx_scg_metadata ON eudr_supply_chain_mapper.supply_chain_graphs USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_scg_updated_at
    BEFORE UPDATE ON eudr_supply_chain_mapper.supply_chain_graphs
    FOR EACH ROW
    EXECUTE FUNCTION eudr_supply_chain_mapper.set_updated_at();

-- =============================================================================
-- Table 2: eudr_supply_chain_mapper.supply_chain_nodes
-- =============================================================================
-- Supply chain actors (producers, collectors, processors, traders, importers)
-- modeled as graph nodes.  Each node belongs to a graph, has a tier depth
-- in the supply chain, geolocation coordinates, risk scoring, compliance
-- status, certifications, and links to production plot IDs per Article 9.
-- Supports many-to-many topology where one actor can appear in multiple
-- graphs and one graph contains multiple actors at various tier depths.

CREATE TABLE eudr_supply_chain_mapper.supply_chain_nodes (
    node_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id                UUID            NOT NULL REFERENCES eudr_supply_chain_mapper.supply_chain_graphs(graph_id) ON DELETE CASCADE,
    tenant_id               UUID,
    node_type               VARCHAR(50)     NOT NULL,
    operator_id             VARCHAR(100),
    operator_name           VARCHAR(500)    NOT NULL,
    country_code            CHAR(2)         NOT NULL,
    region                  VARCHAR(200),
    address                 TEXT,
    latitude                DOUBLE PRECISION,
    longitude               DOUBLE PRECISION,
    geolocation_polygon     JSONB,
    commodities             JSONB           DEFAULT '[]',
    tier_depth              INTEGER         NOT NULL DEFAULT 0,
    risk_score              NUMERIC(5,2)    NOT NULL DEFAULT 0.0,
    risk_level              VARCHAR(20)     NOT NULL DEFAULT 'standard',
    compliance_status       VARCHAR(50)     NOT NULL DEFAULT 'pending_verification',
    verification_date       TIMESTAMPTZ,
    certifications          JSONB           DEFAULT '[]',
    plot_ids                JSONB           DEFAULT '[]',
    capacity                JSONB           DEFAULT '{}',
    contact_info            JSONB           DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_scn_node_type CHECK (
        node_type IN (
            'producer', 'collector', 'cooperative', 'aggregator',
            'processor', 'refiner', 'trader', 'exporter',
            'importer', 'distributor', 'retailer', 'warehouse',
            'port', 'customs_broker', 'certification_body'
        )
    ),
    CONSTRAINT chk_scn_operator_name_not_empty CHECK (
        LENGTH(TRIM(operator_name)) > 0
    ),
    CONSTRAINT chk_scn_country_code_length CHECK (
        LENGTH(TRIM(country_code)) = 2
    ),
    CONSTRAINT chk_scn_tier_depth_non_neg CHECK (tier_depth >= 0),
    CONSTRAINT chk_scn_risk_score_range CHECK (
        risk_score >= 0.0 AND risk_score <= 100.0
    ),
    CONSTRAINT chk_scn_risk_level CHECK (
        risk_level IN ('low', 'standard', 'elevated', 'high', 'critical')
    ),
    CONSTRAINT chk_scn_compliance_status CHECK (
        compliance_status IN (
            'pending_verification', 'verified', 'partially_verified',
            'non_compliant', 'suspended', 'expired', 'unknown'
        )
    ),
    CONSTRAINT chk_scn_latitude_range CHECK (
        latitude IS NULL OR (latitude >= -90.0 AND latitude <= 90.0)
    ),
    CONSTRAINT chk_scn_longitude_range CHECK (
        longitude IS NULL OR (longitude >= -180.0 AND longitude <= 180.0)
    )
);

-- Indexes
CREATE INDEX idx_scn_graph_id ON eudr_supply_chain_mapper.supply_chain_nodes(graph_id);
CREATE INDEX idx_scn_tenant ON eudr_supply_chain_mapper.supply_chain_nodes(tenant_id);
CREATE INDEX idx_scn_node_type ON eudr_supply_chain_mapper.supply_chain_nodes(node_type);
CREATE INDEX idx_scn_operator_id ON eudr_supply_chain_mapper.supply_chain_nodes(operator_id);
CREATE INDEX idx_scn_country_code ON eudr_supply_chain_mapper.supply_chain_nodes(country_code);
CREATE INDEX idx_scn_risk_level ON eudr_supply_chain_mapper.supply_chain_nodes(risk_level);
CREATE INDEX idx_scn_risk_score ON eudr_supply_chain_mapper.supply_chain_nodes(risk_score);
CREATE INDEX idx_scn_compliance_status ON eudr_supply_chain_mapper.supply_chain_nodes(compliance_status);
CREATE INDEX idx_scn_tier_depth ON eudr_supply_chain_mapper.supply_chain_nodes(tier_depth);
CREATE INDEX idx_scn_graph_type ON eudr_supply_chain_mapper.supply_chain_nodes(graph_id, node_type);
CREATE INDEX idx_scn_graph_country ON eudr_supply_chain_mapper.supply_chain_nodes(graph_id, country_code);
CREATE INDEX idx_scn_graph_risk ON eudr_supply_chain_mapper.supply_chain_nodes(graph_id, risk_level);
CREATE INDEX idx_scn_graph_tier ON eudr_supply_chain_mapper.supply_chain_nodes(graph_id, tier_depth);
CREATE INDEX idx_scn_latlon ON eudr_supply_chain_mapper.supply_chain_nodes(latitude, longitude) WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
CREATE INDEX idx_scn_created_at ON eudr_supply_chain_mapper.supply_chain_nodes(created_at DESC);
CREATE INDEX idx_scn_commodities ON eudr_supply_chain_mapper.supply_chain_nodes USING GIN(commodities);
CREATE INDEX idx_scn_certifications ON eudr_supply_chain_mapper.supply_chain_nodes USING GIN(certifications);
CREATE INDEX idx_scn_plot_ids ON eudr_supply_chain_mapper.supply_chain_nodes USING GIN(plot_ids);
CREATE INDEX idx_scn_geolocation ON eudr_supply_chain_mapper.supply_chain_nodes USING GIN(geolocation_polygon);
CREATE INDEX idx_scn_capacity ON eudr_supply_chain_mapper.supply_chain_nodes USING GIN(capacity);
CREATE INDEX idx_scn_metadata ON eudr_supply_chain_mapper.supply_chain_nodes USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_scn_updated_at
    BEFORE UPDATE ON eudr_supply_chain_mapper.supply_chain_nodes
    FOR EACH ROW
    EXECUTE FUNCTION eudr_supply_chain_mapper.set_updated_at();

-- =============================================================================
-- Table 3: eudr_supply_chain_mapper.supply_chain_edges
-- =============================================================================
-- Directed edges representing custody transfers between supply chain nodes.
-- Each edge records the commodity flow from source to target node with
-- quantity, unit, batch number, custody model (segregated, mass balance,
-- book and claim), transfer date, CN/HS codes for customs classification,
-- transport mode, and a provenance hash for integrity verification.
-- Supports batch split/merge tracking for many-to-many traceability.

CREATE TABLE eudr_supply_chain_mapper.supply_chain_edges (
    edge_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id                UUID            NOT NULL REFERENCES eudr_supply_chain_mapper.supply_chain_graphs(graph_id) ON DELETE CASCADE,
    source_node_id          UUID            NOT NULL REFERENCES eudr_supply_chain_mapper.supply_chain_nodes(node_id) ON DELETE CASCADE,
    target_node_id          UUID            NOT NULL REFERENCES eudr_supply_chain_mapper.supply_chain_nodes(node_id) ON DELETE CASCADE,
    tenant_id               UUID,
    commodity               VARCHAR(50)     NOT NULL,
    product_description     VARCHAR(1000),
    quantity                NUMERIC(18,4)   NOT NULL,
    unit                    VARCHAR(20)     NOT NULL DEFAULT 'kg',
    batch_number            VARCHAR(100),
    parent_batch_ids        JSONB           DEFAULT '[]',
    child_batch_ids         JSONB           DEFAULT '[]',
    custody_model           VARCHAR(30)     NOT NULL DEFAULT 'segregated',
    transfer_date           TIMESTAMPTZ,
    cn_code                 VARCHAR(20),
    hs_code                 VARCHAR(20),
    transport_mode          VARCHAR(50),
    transport_details       JSONB           DEFAULT '{}',
    documents               JSONB           DEFAULT '[]',
    provenance_hash         VARCHAR(64)     NOT NULL,
    verification_status     VARCHAR(30)     NOT NULL DEFAULT 'unverified',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sce_commodity CHECK (
        commodity IN (
            'cattle', 'cocoa', 'coffee', 'palm_oil',
            'rubber', 'soya', 'wood',
            'cattle_derived', 'cocoa_derived', 'coffee_derived',
            'palm_oil_derived', 'rubber_derived', 'soya_derived', 'wood_derived'
        )
    ),
    CONSTRAINT chk_sce_quantity_positive CHECK (quantity > 0),
    CONSTRAINT chk_sce_unit_not_empty CHECK (
        LENGTH(TRIM(unit)) > 0
    ),
    CONSTRAINT chk_sce_custody_model CHECK (
        custody_model IN ('segregated', 'mass_balance', 'book_and_claim', 'identity_preserved', 'controlled_blending')
    ),
    CONSTRAINT chk_sce_transport_mode CHECK (
        transport_mode IS NULL OR transport_mode IN (
            'road', 'rail', 'sea', 'air', 'inland_waterway',
            'pipeline', 'multimodal', 'unknown'
        )
    ),
    CONSTRAINT chk_sce_provenance_hash_not_empty CHECK (
        LENGTH(TRIM(provenance_hash)) > 0
    ),
    CONSTRAINT chk_sce_verification_status CHECK (
        verification_status IN ('unverified', 'verified', 'failed', 'pending', 'expired')
    ),
    CONSTRAINT chk_sce_no_self_loop CHECK (
        source_node_id != target_node_id
    )
);

-- Indexes
CREATE INDEX idx_sce_graph_id ON eudr_supply_chain_mapper.supply_chain_edges(graph_id);
CREATE INDEX idx_sce_source_node ON eudr_supply_chain_mapper.supply_chain_edges(source_node_id);
CREATE INDEX idx_sce_target_node ON eudr_supply_chain_mapper.supply_chain_edges(target_node_id);
CREATE INDEX idx_sce_tenant ON eudr_supply_chain_mapper.supply_chain_edges(tenant_id);
CREATE INDEX idx_sce_commodity ON eudr_supply_chain_mapper.supply_chain_edges(commodity);
CREATE INDEX idx_sce_batch_number ON eudr_supply_chain_mapper.supply_chain_edges(batch_number);
CREATE INDEX idx_sce_custody_model ON eudr_supply_chain_mapper.supply_chain_edges(custody_model);
CREATE INDEX idx_sce_transfer_date ON eudr_supply_chain_mapper.supply_chain_edges(transfer_date DESC);
CREATE INDEX idx_sce_cn_code ON eudr_supply_chain_mapper.supply_chain_edges(cn_code);
CREATE INDEX idx_sce_hs_code ON eudr_supply_chain_mapper.supply_chain_edges(hs_code);
CREATE INDEX idx_sce_transport_mode ON eudr_supply_chain_mapper.supply_chain_edges(transport_mode);
CREATE INDEX idx_sce_provenance_hash ON eudr_supply_chain_mapper.supply_chain_edges(provenance_hash);
CREATE INDEX idx_sce_verification ON eudr_supply_chain_mapper.supply_chain_edges(verification_status);
CREATE INDEX idx_sce_graph_commodity ON eudr_supply_chain_mapper.supply_chain_edges(graph_id, commodity);
CREATE INDEX idx_sce_graph_source ON eudr_supply_chain_mapper.supply_chain_edges(graph_id, source_node_id);
CREATE INDEX idx_sce_graph_target ON eudr_supply_chain_mapper.supply_chain_edges(graph_id, target_node_id);
CREATE INDEX idx_sce_created_at ON eudr_supply_chain_mapper.supply_chain_edges(created_at DESC);
CREATE INDEX idx_sce_parent_batches ON eudr_supply_chain_mapper.supply_chain_edges USING GIN(parent_batch_ids);
CREATE INDEX idx_sce_child_batches ON eudr_supply_chain_mapper.supply_chain_edges USING GIN(child_batch_ids);
CREATE INDEX idx_sce_transport_details ON eudr_supply_chain_mapper.supply_chain_edges USING GIN(transport_details);
CREATE INDEX idx_sce_documents ON eudr_supply_chain_mapper.supply_chain_edges USING GIN(documents);
CREATE INDEX idx_sce_metadata ON eudr_supply_chain_mapper.supply_chain_edges USING GIN(metadata);

-- =============================================================================
-- Table 4: eudr_supply_chain_mapper.gap_analysis_results (HYPERTABLE)
-- =============================================================================
-- Supply chain gap detection results partitioned by detected_at.  Each record
-- identifies a gap in the supply chain (missing tier, unverified actor,
-- missing geolocation, custody chain break, missing certification, etc.)
-- with severity, affected node/edge, EUDR article reference, remediation
-- guidance, and resolution tracking.

CREATE TABLE eudr_supply_chain_mapper.gap_analysis_results (
    analysis_id             UUID            NOT NULL DEFAULT gen_random_uuid(),
    graph_id                UUID            NOT NULL,
    tenant_id               UUID,
    gap_type                VARCHAR(50)     NOT NULL,
    severity                VARCHAR(20)     NOT NULL,
    affected_node_id        UUID,
    affected_edge_id        UUID,
    description             TEXT            NOT NULL,
    remediation             TEXT,
    eudr_article            VARCHAR(20),
    compliance_impact       VARCHAR(20)     NOT NULL DEFAULT 'blocking',
    confidence              NUMERIC(5,2)    DEFAULT 0.0,
    evidence                JSONB           DEFAULT '{}',
    is_resolved             BOOLEAN         NOT NULL DEFAULT FALSE,
    resolved_at             TIMESTAMPTZ,
    resolved_by             VARCHAR(200),
    resolution_notes        TEXT,
    metadata                JSONB           DEFAULT '{}',
    detected_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_gar_gap_type CHECK (
        gap_type IN (
            'missing_tier', 'unverified_actor', 'missing_geolocation',
            'custody_chain_break', 'missing_certification', 'expired_certification',
            'missing_batch_link', 'incomplete_provenance', 'missing_plot_link',
            'missing_dds_reference', 'quantity_mismatch', 'date_inconsistency',
            'country_risk_unassessed', 'missing_transport_docs', 'orphan_node',
            'disconnected_subgraph', 'duplicate_edge', 'circular_reference'
        )
    ),
    CONSTRAINT chk_gar_severity CHECK (
        severity IN ('critical', 'high', 'medium', 'low', 'info')
    ),
    CONSTRAINT chk_gar_desc_not_empty CHECK (
        LENGTH(TRIM(description)) > 0
    ),
    CONSTRAINT chk_gar_compliance_impact CHECK (
        compliance_impact IN ('blocking', 'degrading', 'informational')
    ),
    CONSTRAINT chk_gar_confidence_range CHECK (
        confidence IS NULL OR (confidence >= 0.0 AND confidence <= 100.0)
    ),
    CONSTRAINT chk_gar_resolved_consistency CHECK (
        (is_resolved = FALSE AND resolved_at IS NULL) OR
        (is_resolved = TRUE AND resolved_at IS NOT NULL)
    )
);

-- Convert to hypertable (3-month chunks)
SELECT create_hypertable('eudr_supply_chain_mapper.gap_analysis_results', 'detected_at',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes (hypertable-aware)
CREATE INDEX idx_gar_graph_id ON eudr_supply_chain_mapper.gap_analysis_results(graph_id, detected_at DESC);
CREATE INDEX idx_gar_tenant ON eudr_supply_chain_mapper.gap_analysis_results(tenant_id, detected_at DESC);
CREATE INDEX idx_gar_gap_type ON eudr_supply_chain_mapper.gap_analysis_results(gap_type, detected_at DESC);
CREATE INDEX idx_gar_severity ON eudr_supply_chain_mapper.gap_analysis_results(severity, detected_at DESC);
CREATE INDEX idx_gar_affected_node ON eudr_supply_chain_mapper.gap_analysis_results(affected_node_id, detected_at DESC);
CREATE INDEX idx_gar_affected_edge ON eudr_supply_chain_mapper.gap_analysis_results(affected_edge_id, detected_at DESC);
CREATE INDEX idx_gar_eudr_article ON eudr_supply_chain_mapper.gap_analysis_results(eudr_article, detected_at DESC);
CREATE INDEX idx_gar_compliance_impact ON eudr_supply_chain_mapper.gap_analysis_results(compliance_impact, detected_at DESC);
CREATE INDEX idx_gar_is_resolved ON eudr_supply_chain_mapper.gap_analysis_results(is_resolved, detected_at DESC);
CREATE INDEX idx_gar_graph_severity ON eudr_supply_chain_mapper.gap_analysis_results(graph_id, severity, detected_at DESC);
CREATE INDEX idx_gar_graph_type ON eudr_supply_chain_mapper.gap_analysis_results(graph_id, gap_type, detected_at DESC);
CREATE INDEX idx_gar_graph_resolved ON eudr_supply_chain_mapper.gap_analysis_results(graph_id, is_resolved, detected_at DESC);
CREATE INDEX idx_gar_evidence ON eudr_supply_chain_mapper.gap_analysis_results USING GIN(evidence);
CREATE INDEX idx_gar_metadata ON eudr_supply_chain_mapper.gap_analysis_results USING GIN(metadata);

-- =============================================================================
-- Table 5: eudr_supply_chain_mapper.risk_propagation_log (HYPERTABLE)
-- =============================================================================
-- Risk propagation audit log partitioned by calculated_at.  Each record
-- captures a risk score change on a supply chain node during a propagation
-- run, recording previous and new risk scores/levels, the propagation source
-- (country risk, deforestation alert, supplier risk, etc.), contributing
-- risk factors, and the propagation algorithm version for reproducibility.

CREATE TABLE eudr_supply_chain_mapper.risk_propagation_log (
    log_id                  UUID            NOT NULL DEFAULT gen_random_uuid(),
    graph_id                UUID            NOT NULL,
    tenant_id               UUID,
    node_id                 UUID            NOT NULL,
    propagation_run_id      UUID            NOT NULL,
    previous_risk_score     NUMERIC(5,2),
    new_risk_score          NUMERIC(5,2)    NOT NULL,
    previous_risk_level     VARCHAR(20),
    new_risk_level          VARCHAR(20)     NOT NULL,
    risk_delta              NUMERIC(5,2)    NOT NULL DEFAULT 0.0,
    propagation_source      VARCHAR(50)     NOT NULL,
    propagation_direction   VARCHAR(20)     NOT NULL DEFAULT 'upstream',
    risk_factors            JSONB           DEFAULT '[]',
    contributing_nodes      JSONB           DEFAULT '[]',
    algorithm_version       VARCHAR(20)     NOT NULL DEFAULT '1.0',
    metadata                JSONB           DEFAULT '{}',
    calculated_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_rpl_new_risk_range CHECK (
        new_risk_score >= 0.0 AND new_risk_score <= 100.0
    ),
    CONSTRAINT chk_rpl_prev_risk_range CHECK (
        previous_risk_score IS NULL OR (previous_risk_score >= 0.0 AND previous_risk_score <= 100.0)
    ),
    CONSTRAINT chk_rpl_new_risk_level CHECK (
        new_risk_level IN ('low', 'standard', 'elevated', 'high', 'critical')
    ),
    CONSTRAINT chk_rpl_prev_risk_level CHECK (
        previous_risk_level IS NULL OR previous_risk_level IN ('low', 'standard', 'elevated', 'high', 'critical')
    ),
    CONSTRAINT chk_rpl_propagation_source CHECK (
        propagation_source IN (
            'country_risk', 'commodity_risk', 'deforestation_alert',
            'supplier_risk', 'certification_expiry', 'compliance_failure',
            'manual_override', 'satellite_detection', 'gap_analysis',
            'batch_contamination', 'regulatory_change', 'initial_assessment'
        )
    ),
    CONSTRAINT chk_rpl_propagation_direction CHECK (
        propagation_direction IN ('upstream', 'downstream', 'bidirectional')
    )
);

-- Convert to hypertable (3-month chunks)
SELECT create_hypertable('eudr_supply_chain_mapper.risk_propagation_log', 'calculated_at',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes (hypertable-aware)
CREATE INDEX idx_rpl_graph_id ON eudr_supply_chain_mapper.risk_propagation_log(graph_id, calculated_at DESC);
CREATE INDEX idx_rpl_tenant ON eudr_supply_chain_mapper.risk_propagation_log(tenant_id, calculated_at DESC);
CREATE INDEX idx_rpl_node_id ON eudr_supply_chain_mapper.risk_propagation_log(node_id, calculated_at DESC);
CREATE INDEX idx_rpl_propagation_run ON eudr_supply_chain_mapper.risk_propagation_log(propagation_run_id, calculated_at DESC);
CREATE INDEX idx_rpl_propagation_source ON eudr_supply_chain_mapper.risk_propagation_log(propagation_source, calculated_at DESC);
CREATE INDEX idx_rpl_new_risk_level ON eudr_supply_chain_mapper.risk_propagation_log(new_risk_level, calculated_at DESC);
CREATE INDEX idx_rpl_direction ON eudr_supply_chain_mapper.risk_propagation_log(propagation_direction, calculated_at DESC);
CREATE INDEX idx_rpl_graph_node ON eudr_supply_chain_mapper.risk_propagation_log(graph_id, node_id, calculated_at DESC);
CREATE INDEX idx_rpl_graph_source ON eudr_supply_chain_mapper.risk_propagation_log(graph_id, propagation_source, calculated_at DESC);
CREATE INDEX idx_rpl_graph_level ON eudr_supply_chain_mapper.risk_propagation_log(graph_id, new_risk_level, calculated_at DESC);
CREATE INDEX idx_rpl_algorithm ON eudr_supply_chain_mapper.risk_propagation_log(algorithm_version, calculated_at DESC);
CREATE INDEX idx_rpl_risk_factors ON eudr_supply_chain_mapper.risk_propagation_log USING GIN(risk_factors);
CREATE INDEX idx_rpl_contributing ON eudr_supply_chain_mapper.risk_propagation_log USING GIN(contributing_nodes);
CREATE INDEX idx_rpl_metadata ON eudr_supply_chain_mapper.risk_propagation_log USING GIN(metadata);

-- =============================================================================
-- Table 6: eudr_supply_chain_mapper.graph_snapshots (HYPERTABLE)
-- =============================================================================
-- Versioned graph snapshots for audit trail, partitioned by created_at.
-- Each snapshot captures the full graph state (nodes, edges, risk scores)
-- at a point in time with a provenance hash for integrity verification.
-- Used for regulatory audits, reproducibility checks, and version
-- comparison (diff between snapshots).

CREATE TABLE eudr_supply_chain_mapper.graph_snapshots (
    snapshot_id             UUID            NOT NULL DEFAULT gen_random_uuid(),
    graph_id                UUID            NOT NULL,
    tenant_id               UUID,
    version                 INTEGER         NOT NULL,
    snapshot_type           VARCHAR(30)     NOT NULL DEFAULT 'full',
    snapshot_data           JSONB           NOT NULL,
    node_count              INTEGER         NOT NULL DEFAULT 0,
    edge_count              INTEGER         NOT NULL DEFAULT 0,
    provenance_hash         VARCHAR(64)     NOT NULL,
    parent_snapshot_id      UUID,
    diff_data               JSONB,
    trigger_reason          VARCHAR(100),
    created_by              VARCHAR(200),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_gs_version_positive CHECK (version >= 1),
    CONSTRAINT chk_gs_snapshot_type CHECK (
        snapshot_type IN ('full', 'incremental', 'risk_update', 'structure_change', 'compliance_check')
    ),
    CONSTRAINT chk_gs_node_count_non_neg CHECK (node_count >= 0),
    CONSTRAINT chk_gs_edge_count_non_neg CHECK (edge_count >= 0),
    CONSTRAINT chk_gs_provenance_hash_not_empty CHECK (
        LENGTH(TRIM(provenance_hash)) > 0
    )
);

-- Convert to hypertable (3-month chunks)
SELECT create_hypertable('eudr_supply_chain_mapper.graph_snapshots', 'created_at',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes (hypertable-aware)
CREATE INDEX idx_gs_graph_id ON eudr_supply_chain_mapper.graph_snapshots(graph_id, created_at DESC);
CREATE INDEX idx_gs_tenant ON eudr_supply_chain_mapper.graph_snapshots(tenant_id, created_at DESC);
CREATE INDEX idx_gs_version ON eudr_supply_chain_mapper.graph_snapshots(graph_id, version, created_at DESC);
CREATE INDEX idx_gs_snapshot_type ON eudr_supply_chain_mapper.graph_snapshots(snapshot_type, created_at DESC);
CREATE INDEX idx_gs_provenance_hash ON eudr_supply_chain_mapper.graph_snapshots(provenance_hash, created_at DESC);
CREATE INDEX idx_gs_parent_snapshot ON eudr_supply_chain_mapper.graph_snapshots(parent_snapshot_id, created_at DESC);
CREATE INDEX idx_gs_created_by ON eudr_supply_chain_mapper.graph_snapshots(created_by, created_at DESC);
CREATE INDEX idx_gs_graph_type ON eudr_supply_chain_mapper.graph_snapshots(graph_id, snapshot_type, created_at DESC);
CREATE INDEX idx_gs_snapshot_data ON eudr_supply_chain_mapper.graph_snapshots USING GIN(snapshot_data);
CREATE INDEX idx_gs_diff_data ON eudr_supply_chain_mapper.graph_snapshots USING GIN(diff_data) WHERE diff_data IS NOT NULL;
CREATE INDEX idx_gs_metadata ON eudr_supply_chain_mapper.graph_snapshots USING GIN(metadata);

-- =============================================================================
-- Continuous Aggregate 1: eudr_supply_chain_mapper.daily_gap_summary
-- =============================================================================
-- Daily aggregation of gap analysis results by graph and severity showing
-- total gap counts, resolved counts, and critical/high gap breakdown.

CREATE MATERIALIZED VIEW eudr_supply_chain_mapper.daily_gap_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', detected_at)                               AS bucket,
    graph_id,
    COUNT(*)                                                        AS total_gaps,
    COUNT(*) FILTER (WHERE severity = 'critical')                   AS critical_count,
    COUNT(*) FILTER (WHERE severity = 'high')                       AS high_count,
    COUNT(*) FILTER (WHERE severity = 'medium')                     AS medium_count,
    COUNT(*) FILTER (WHERE severity = 'low')                        AS low_count,
    COUNT(*) FILTER (WHERE is_resolved = TRUE)                      AS resolved_count,
    COUNT(*) FILTER (WHERE is_resolved = FALSE)                     AS unresolved_count,
    COUNT(*) FILTER (WHERE compliance_impact = 'blocking')          AS blocking_count
FROM eudr_supply_chain_mapper.gap_analysis_results
GROUP BY bucket, graph_id
WITH NO DATA;

-- Refresh policy: every 30 minutes, covering last 6 hours
SELECT add_continuous_aggregate_policy('eudr_supply_chain_mapper.daily_gap_summary',
    start_offset => INTERVAL '6 hours',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Continuous Aggregate 2: eudr_supply_chain_mapper.monthly_risk_propagation
-- =============================================================================
-- Monthly aggregation of risk propagation events by graph and propagation
-- source showing total propagation events, average risk delta, and
-- breakdown by risk level transitions.

CREATE MATERIALIZED VIEW eudr_supply_chain_mapper.monthly_risk_propagation
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 month', calculated_at)                           AS bucket,
    graph_id,
    propagation_source,
    COUNT(*)                                                        AS total_events,
    AVG(new_risk_score)                                             AS avg_new_risk_score,
    AVG(risk_delta)                                                 AS avg_risk_delta,
    MAX(new_risk_score)                                             AS max_risk_score,
    COUNT(*) FILTER (WHERE new_risk_level = 'critical')             AS critical_count,
    COUNT(*) FILTER (WHERE new_risk_level = 'high')                 AS high_count,
    COUNT(*) FILTER (WHERE new_risk_level = 'elevated')             AS elevated_count,
    COUNT(DISTINCT node_id)                                         AS affected_nodes
FROM eudr_supply_chain_mapper.risk_propagation_log
GROUP BY bucket, graph_id, propagation_source
WITH NO DATA;

-- Refresh policy: every 30 minutes, covering last 6 hours
SELECT add_continuous_aggregate_policy('eudr_supply_chain_mapper.monthly_risk_propagation',
    start_offset => INTERVAL '6 hours',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep gap analysis results for 3650 days (10 years, EUDR regulatory retention)
SELECT add_retention_policy('eudr_supply_chain_mapper.gap_analysis_results', INTERVAL '3650 days');

-- Keep risk propagation log for 3650 days (10 years)
SELECT add_retention_policy('eudr_supply_chain_mapper.risk_propagation_log', INTERVAL '3650 days');

-- Keep graph snapshots for 3650 days (10 years)
SELECT add_retention_policy('eudr_supply_chain_mapper.graph_snapshots', INTERVAL '3650 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on gap_analysis_results after 90 days
ALTER TABLE eudr_supply_chain_mapper.gap_analysis_results SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'detected_at DESC'
);

SELECT add_compression_policy('eudr_supply_chain_mapper.gap_analysis_results', INTERVAL '90 days');

-- Enable compression on risk_propagation_log after 90 days
ALTER TABLE eudr_supply_chain_mapper.risk_propagation_log SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'calculated_at DESC'
);

SELECT add_compression_policy('eudr_supply_chain_mapper.risk_propagation_log', INTERVAL '90 days');

-- Enable compression on graph_snapshots after 90 days
ALTER TABLE eudr_supply_chain_mapper.graph_snapshots SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_compression_policy('eudr_supply_chain_mapper.graph_snapshots', INTERVAL '90 days');

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA eudr_supply_chain_mapper TO greenlang_app;
GRANT ALL ON ALL TABLES IN SCHEMA eudr_supply_chain_mapper TO greenlang_app;
GRANT ALL ON ALL SEQUENCES IN SCHEMA eudr_supply_chain_mapper TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON eudr_supply_chain_mapper.daily_gap_summary TO greenlang_app;
GRANT SELECT ON eudr_supply_chain_mapper.monthly_risk_propagation TO greenlang_app;

-- Read-only role
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_readonly') THEN
        GRANT USAGE ON SCHEMA eudr_supply_chain_mapper TO greenlang_readonly;
        GRANT SELECT ON ALL TABLES IN SCHEMA eudr_supply_chain_mapper TO greenlang_readonly;
        GRANT SELECT ON eudr_supply_chain_mapper.daily_gap_summary TO greenlang_readonly;
        GRANT SELECT ON eudr_supply_chain_mapper.monthly_risk_propagation TO greenlang_readonly;
    END IF;
END
$$;

-- Add EUDR Supply Chain Mapper agent permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'eudr_scm:graphs:read', 'eudr_supply_chain_mapper', 'graphs_read', 'View supply chain graphs and metadata'),
    (gen_random_uuid(), 'eudr_scm:graphs:write', 'eudr_supply_chain_mapper', 'graphs_write', 'Create and manage supply chain graphs'),
    (gen_random_uuid(), 'eudr_scm:graphs:delete', 'eudr_supply_chain_mapper', 'graphs_delete', 'Archive and delete supply chain graphs'),
    (gen_random_uuid(), 'eudr_scm:graphs:export', 'eudr_supply_chain_mapper', 'graphs_export', 'Export supply chain graphs (GraphML/JSON-LD/GeoJSON)'),
    (gen_random_uuid(), 'eudr_scm:nodes:read', 'eudr_supply_chain_mapper', 'nodes_read', 'View supply chain nodes and actor details'),
    (gen_random_uuid(), 'eudr_scm:nodes:write', 'eudr_supply_chain_mapper', 'nodes_write', 'Create and manage supply chain nodes'),
    (gen_random_uuid(), 'eudr_scm:nodes:delete', 'eudr_supply_chain_mapper', 'nodes_delete', 'Remove supply chain nodes'),
    (gen_random_uuid(), 'eudr_scm:edges:read', 'eudr_supply_chain_mapper', 'edges_read', 'View custody transfer edges'),
    (gen_random_uuid(), 'eudr_scm:edges:write', 'eudr_supply_chain_mapper', 'edges_write', 'Create and manage custody transfer edges'),
    (gen_random_uuid(), 'eudr_scm:edges:delete', 'eudr_supply_chain_mapper', 'edges_delete', 'Remove custody transfer edges'),
    (gen_random_uuid(), 'eudr_scm:discovery:run', 'eudr_supply_chain_mapper', 'discovery_run', 'Trigger multi-tier supply chain discovery'),
    (gen_random_uuid(), 'eudr_scm:traceability:read', 'eudr_supply_chain_mapper', 'traceability_read', 'View plot-to-product traceability paths'),
    (gen_random_uuid(), 'eudr_scm:traceability:run', 'eudr_supply_chain_mapper', 'traceability_run', 'Execute traceability trace operations'),
    (gen_random_uuid(), 'eudr_scm:risk:read', 'eudr_supply_chain_mapper', 'risk_read', 'View risk propagation results and audit log'),
    (gen_random_uuid(), 'eudr_scm:risk:propagate', 'eudr_supply_chain_mapper', 'risk_propagate', 'Execute risk propagation through supply chain graph'),
    (gen_random_uuid(), 'eudr_scm:gaps:read', 'eudr_supply_chain_mapper', 'gaps_read', 'View gap analysis results'),
    (gen_random_uuid(), 'eudr_scm:gaps:run', 'eudr_supply_chain_mapper', 'gaps_run', 'Execute supply chain gap analysis'),
    (gen_random_uuid(), 'eudr_scm:gaps:resolve', 'eudr_supply_chain_mapper', 'gaps_resolve', 'Mark gaps as resolved'),
    (gen_random_uuid(), 'eudr_scm:snapshots:read', 'eudr_supply_chain_mapper', 'snapshots_read', 'View graph version snapshots'),
    (gen_random_uuid(), 'eudr_scm:snapshots:create', 'eudr_supply_chain_mapper', 'snapshots_create', 'Create graph version snapshots'),
    (gen_random_uuid(), 'eudr_scm:visualization:read', 'eudr_supply_chain_mapper', 'visualization_read', 'View supply chain visualizations and Sankey diagrams'),
    (gen_random_uuid(), 'eudr_scm:dashboard:read', 'eudr_supply_chain_mapper', 'dashboard_read', 'View supply chain mapping dashboards and analytics'),
    (gen_random_uuid(), 'eudr_scm:admin', 'eudr_supply_chain_mapper', 'admin', 'EUDR Supply Chain Mapper administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA eudr_supply_chain_mapper IS 'AGENT-EUDR-001 Supply Chain Mapping Master Schema - Graph-native multi-tier supply chain modeling for EUDR Regulation (EU) 2023/1115 with plot-to-product traceability, many-to-many batch topology, risk propagation, gap analysis, and versioned snapshots covering all 7 regulated commodities (cattle, cocoa, coffee, palm oil, rubber, soya, wood)';

COMMENT ON TABLE eudr_supply_chain_mapper.supply_chain_graphs IS 'Main graph metadata: one graph per operator per commodity with node/edge counts, max tier depth, traceability score, compliance readiness, risk summary, and version history';
COMMENT ON TABLE eudr_supply_chain_mapper.supply_chain_nodes IS 'Supply chain actors (producers, collectors, processors, traders, importers) as graph nodes with geolocation, tier depth, risk scoring, compliance status, certifications, and plot linkage';
COMMENT ON TABLE eudr_supply_chain_mapper.supply_chain_edges IS 'Directed custody transfer edges between nodes with commodity, quantity, batch tracking, custody model (segregated/mass balance/book and claim), CN/HS codes, transport mode, and provenance hash';
COMMENT ON TABLE eudr_supply_chain_mapper.gap_analysis_results IS 'TimescaleDB hypertable: gap detection results with gap type, severity, affected node/edge, EUDR article reference, compliance impact, remediation guidance, and resolution tracking';
COMMENT ON TABLE eudr_supply_chain_mapper.risk_propagation_log IS 'TimescaleDB hypertable: risk propagation audit log with previous/new risk scores, propagation source, direction, risk factors, contributing nodes, and algorithm version';
COMMENT ON TABLE eudr_supply_chain_mapper.graph_snapshots IS 'TimescaleDB hypertable: versioned graph snapshots for audit trail with full/incremental snapshot data, provenance hash, parent snapshot reference, and diff data';

COMMENT ON MATERIALIZED VIEW eudr_supply_chain_mapper.daily_gap_summary IS 'Continuous aggregate: daily gap detection trends by graph showing total/critical/high/resolved/blocking gap counts';
COMMENT ON MATERIALIZED VIEW eudr_supply_chain_mapper.monthly_risk_propagation IS 'Continuous aggregate: monthly risk propagation trends by graph and source showing event counts, avg risk delta, max risk score, and affected node count';

COMMENT ON COLUMN eudr_supply_chain_mapper.supply_chain_graphs.commodity IS 'EUDR-regulated commodity: cattle, cocoa, coffee, palm_oil, rubber, soya, wood (and derived variants)';
COMMENT ON COLUMN eudr_supply_chain_mapper.supply_chain_graphs.traceability_score IS 'Percentage (0-100) indicating how completely the supply chain is traced back to production plots';
COMMENT ON COLUMN eudr_supply_chain_mapper.supply_chain_graphs.compliance_readiness IS 'Percentage (0-100) indicating overall EUDR compliance readiness of the supply chain';
COMMENT ON COLUMN eudr_supply_chain_mapper.supply_chain_graphs.status IS 'Graph status: draft (initial), building (discovery in progress), active (operational), archived, error';
COMMENT ON COLUMN eudr_supply_chain_mapper.supply_chain_nodes.node_type IS 'Actor type: producer, collector, cooperative, aggregator, processor, refiner, trader, exporter, importer, distributor, retailer, warehouse, port, customs_broker, certification_body';
COMMENT ON COLUMN eudr_supply_chain_mapper.supply_chain_nodes.risk_level IS 'Risk level: low, standard, elevated, high, critical';
COMMENT ON COLUMN eudr_supply_chain_mapper.supply_chain_nodes.compliance_status IS 'Compliance status: pending_verification, verified, partially_verified, non_compliant, suspended, expired, unknown';
COMMENT ON COLUMN eudr_supply_chain_mapper.supply_chain_nodes.plot_ids IS 'JSON array of production plot UUIDs linked to this node (Article 9 geolocation requirement)';
COMMENT ON COLUMN eudr_supply_chain_mapper.supply_chain_edges.custody_model IS 'Chain of custody model: segregated (fully separated), mass_balance (proportion tracking), book_and_claim (certificate trading), identity_preserved, controlled_blending';
COMMENT ON COLUMN eudr_supply_chain_mapper.supply_chain_edges.provenance_hash IS 'SHA-256 hash of edge data for integrity verification and tamper detection';
COMMENT ON COLUMN eudr_supply_chain_mapper.supply_chain_edges.verification_status IS 'Edge verification status: unverified, verified, failed, pending, expired';
COMMENT ON COLUMN eudr_supply_chain_mapper.gap_analysis_results.gap_type IS 'Gap type: missing_tier, unverified_actor, missing_geolocation, custody_chain_break, missing_certification, expired_certification, missing_batch_link, incomplete_provenance, missing_plot_link, missing_dds_reference, quantity_mismatch, date_inconsistency, country_risk_unassessed, missing_transport_docs, orphan_node, disconnected_subgraph, duplicate_edge, circular_reference';
COMMENT ON COLUMN eudr_supply_chain_mapper.gap_analysis_results.compliance_impact IS 'Impact on EUDR compliance: blocking (prevents DDS submission), degrading (reduces confidence), informational';
COMMENT ON COLUMN eudr_supply_chain_mapper.gap_analysis_results.eudr_article IS 'Reference to EUDR article (e.g., Art.9, Art.10, Art.4) that the gap affects';
COMMENT ON COLUMN eudr_supply_chain_mapper.risk_propagation_log.propagation_source IS 'Source of risk signal: country_risk, commodity_risk, deforestation_alert, supplier_risk, certification_expiry, compliance_failure, manual_override, satellite_detection, gap_analysis, batch_contamination, regulatory_change, initial_assessment';
COMMENT ON COLUMN eudr_supply_chain_mapper.risk_propagation_log.propagation_direction IS 'Direction of risk propagation: upstream (toward production), downstream (toward import), bidirectional';
COMMENT ON COLUMN eudr_supply_chain_mapper.graph_snapshots.snapshot_type IS 'Snapshot type: full (complete graph), incremental (changes only), risk_update, structure_change, compliance_check';
COMMENT ON COLUMN eudr_supply_chain_mapper.graph_snapshots.provenance_hash IS 'SHA-256 hash of snapshot data for integrity verification';

-- =============================================================================
-- End of V089: AGENT-EUDR-001 Supply Chain Mapping Master Schema
-- =============================================================================
-- Summary:
--   6 tables created (3 regular + 3 hypertables)
--   3 hypertables (gap_analysis_results, risk_propagation_log, graph_snapshots)
--   2 continuous aggregates (daily_gap_summary, monthly_risk_propagation)
--   3 update triggers (graphs, nodes, edges)
--   60+ B-tree indexes (including composite and hypertable-aware)
--   20+ GIN indexes on JSONB columns
--   3 retention policies (10-year regulatory retention)
--   3 compression policies (90-day threshold)
--   23 security permissions
--   Security grants for greenlang_app and greenlang_readonly
-- Previous: V088__taxonomy_app_service.sql
-- =============================================================================
