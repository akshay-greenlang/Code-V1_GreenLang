-- =============================================================================
-- V048: Data Lineage Tracker Service Tables
-- =============================================================================
-- Component: AGENT-DATA-018 (Data Lineage Tracker)
-- Agent ID:  GL-DATA-X-021
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Data Lineage Tracker (GL-DATA-X-021) with capabilities for
-- lineage asset registration (qualified_name/asset_type/owner/tags/
-- classification/status/schema_ref), transformation tracking (12
-- transformation types, agent/pipeline/execution refs, record counts,
-- duration), lineage edge management (dataset-level and column-level
-- edges with source/target field resolution, transformation logic,
-- confidence scoring), graph snapshot capture (node/edge/depth/component
-- counts, coverage scoring, graph hashing), forward/backward impact
-- analysis (blast radius calculation, severity-bucketed affected asset
-- counts), lineage validation (orphan/broken-edge/cycle detection,
-- source coverage, completeness/freshness scoring, issues and
-- recommendations), regulatory compliance reporting (CSRD/ESRS, GHG
-- Protocol, SOC 2, custom, visualization formats including Mermaid/DOT/
-- D3/JSON/text/HTML/PDF), change event tracking (node/edge added/removed,
-- topology changes with severity classification), data quality scoring
-- per asset (source credibility, transformation depth, freshness,
-- documentation, manual intervention counts), and full provenance chain
-- tracking with SHA-256 hashes for zero-hallucination audit trails.
-- =============================================================================
-- Tables (10):
--   1. lineage_assets              - Registered lineage assets (qualified_name/type/owner)
--   2. lineage_transformations     - Transformation operations with record counts
--   3. lineage_edges               - Dataset-level and column-level lineage edges
--   4. lineage_graph_snapshots     - Point-in-time graph topology snapshots
--   5. lineage_impact_analyses     - Forward/backward impact analysis results
--   6. lineage_validations         - Lineage graph validation results
--   7. lineage_reports             - Regulatory compliance and visualization reports
--   8. lineage_change_events       - Lineage graph change event log
--   9. lineage_quality_scores      - Per-asset data quality scores
--  10. lineage_audit_log           - Full audit trail with provenance chains
--
-- Hypertables (3):
--  11. lineage_transformation_events - Transformation event time-series (hypertable)
--  12. lineage_validation_events     - Validation event time-series (hypertable)
--  13. lineage_impact_events         - Impact analysis event time-series (hypertable)
--
-- Continuous Aggregates (2):
--   1. lineage_transformations_hourly_stats - Hourly rollup of lineage_transformation_events
--   2. lineage_validations_hourly_stats     - Hourly rollup of lineage_validation_events
--
-- Also includes: indexes (B-tree, GIN, partial, composite),
-- CHECK constraints, RLS policies per tenant, retention
-- policies (90 days on hypertables), compression policies (7 days),
-- updated_at trigger, security permissions for
-- greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-DATA-X-021.
-- Previous: V047__schema_migration_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS data_lineage_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION data_lineage_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: data_lineage_service.lineage_assets
-- =============================================================================

CREATE TABLE IF NOT EXISTS data_lineage_service.lineage_assets (
    id              UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    qualified_name  VARCHAR(500)  NOT NULL,
    asset_type      VARCHAR(50)   NOT NULL,
    display_name    VARCHAR(255),
    owner           VARCHAR(255)  NOT NULL DEFAULT 'system',
    tags            JSONB         NOT NULL DEFAULT '[]'::jsonb,
    classification  VARCHAR(50)   NOT NULL DEFAULT 'internal',
    status          VARCHAR(50)   NOT NULL DEFAULT 'active',
    schema_ref      VARCHAR(500),
    description     TEXT          NOT NULL DEFAULT '',
    metadata        JSONB         NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

ALTER TABLE data_lineage_service.lineage_assets
    ADD CONSTRAINT uq_la_qualified_name UNIQUE (qualified_name);

ALTER TABLE data_lineage_service.lineage_assets
    ADD CONSTRAINT chk_la_qualified_name_not_empty CHECK (LENGTH(TRIM(qualified_name)) > 0);

ALTER TABLE data_lineage_service.lineage_assets
    ADD CONSTRAINT chk_la_asset_type CHECK (asset_type IN (
        'dataset', 'field', 'agent', 'pipeline', 'report', 'metric', 'external_source'
    ));

ALTER TABLE data_lineage_service.lineage_assets
    ADD CONSTRAINT chk_la_classification CHECK (classification IN (
        'public', 'internal', 'confidential', 'restricted'
    ));

ALTER TABLE data_lineage_service.lineage_assets
    ADD CONSTRAINT chk_la_status CHECK (status IN (
        'active', 'deprecated', 'archived'
    ));

CREATE TRIGGER trg_la_updated_at
    BEFORE UPDATE ON data_lineage_service.lineage_assets
    FOR EACH ROW EXECUTE FUNCTION data_lineage_service.set_updated_at();

-- =============================================================================
-- Table 2: data_lineage_service.lineage_transformations
-- =============================================================================

CREATE TABLE IF NOT EXISTS data_lineage_service.lineage_transformations (
    id                   UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    transformation_type  VARCHAR(50)  NOT NULL,
    agent_id             VARCHAR(255),
    pipeline_id          VARCHAR(255),
    execution_id         VARCHAR(255),
    description          TEXT         NOT NULL DEFAULT '',
    parameters           JSONB        NOT NULL DEFAULT '{}'::jsonb,
    records_in           INTEGER      NOT NULL DEFAULT 0,
    records_out          INTEGER      NOT NULL DEFAULT 0,
    records_filtered     INTEGER      NOT NULL DEFAULT 0,
    records_error        INTEGER      NOT NULL DEFAULT 0,
    duration_ms          INTEGER      NOT NULL DEFAULT 0,
    started_at           TIMESTAMPTZ,
    completed_at         TIMESTAMPTZ,
    metadata             JSONB        NOT NULL DEFAULT '{}'::jsonb
);

ALTER TABLE data_lineage_service.lineage_transformations
    ADD CONSTRAINT chk_lt_transformation_type CHECK (transformation_type IN (
        'filter', 'aggregate', 'join', 'calculate', 'impute', 'deduplicate',
        'enrich', 'merge', 'split', 'validate', 'normalize', 'classify'
    ));

ALTER TABLE data_lineage_service.lineage_transformations
    ADD CONSTRAINT chk_lt_records_in_non_negative CHECK (records_in >= 0);

ALTER TABLE data_lineage_service.lineage_transformations
    ADD CONSTRAINT chk_lt_records_out_non_negative CHECK (records_out >= 0);

ALTER TABLE data_lineage_service.lineage_transformations
    ADD CONSTRAINT chk_lt_records_filtered_non_negative CHECK (records_filtered >= 0);

ALTER TABLE data_lineage_service.lineage_transformations
    ADD CONSTRAINT chk_lt_records_error_non_negative CHECK (records_error >= 0);

ALTER TABLE data_lineage_service.lineage_transformations
    ADD CONSTRAINT chk_lt_duration_ms_non_negative CHECK (duration_ms >= 0);

ALTER TABLE data_lineage_service.lineage_transformations
    ADD CONSTRAINT chk_lt_completed_after_started
        CHECK (completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at);

-- =============================================================================
-- Table 3: data_lineage_service.lineage_edges
-- =============================================================================

CREATE TABLE IF NOT EXISTS data_lineage_service.lineage_edges (
    id                   UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    source_asset_id      UUID         NOT NULL,
    target_asset_id      UUID         NOT NULL,
    transformation_id    UUID,
    edge_type            VARCHAR(50)  NOT NULL DEFAULT 'dataset_level',
    source_field         VARCHAR(255),
    target_field         VARCHAR(255),
    transformation_logic TEXT         NOT NULL DEFAULT '',
    confidence           FLOAT        NOT NULL DEFAULT 1.0,
    created_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

ALTER TABLE data_lineage_service.lineage_edges
    ADD CONSTRAINT fk_le_source_asset_id
        FOREIGN KEY (source_asset_id)
        REFERENCES data_lineage_service.lineage_assets(id)
        ON DELETE CASCADE;

ALTER TABLE data_lineage_service.lineage_edges
    ADD CONSTRAINT fk_le_target_asset_id
        FOREIGN KEY (target_asset_id)
        REFERENCES data_lineage_service.lineage_assets(id)
        ON DELETE CASCADE;

ALTER TABLE data_lineage_service.lineage_edges
    ADD CONSTRAINT fk_le_transformation_id
        FOREIGN KEY (transformation_id)
        REFERENCES data_lineage_service.lineage_transformations(id)
        ON DELETE SET NULL;

ALTER TABLE data_lineage_service.lineage_edges
    ADD CONSTRAINT chk_le_edge_type CHECK (edge_type IN (
        'dataset_level', 'column_level'
    ));

ALTER TABLE data_lineage_service.lineage_edges
    ADD CONSTRAINT chk_le_confidence_range
        CHECK (confidence >= 0.0 AND confidence <= 1.0);

ALTER TABLE data_lineage_service.lineage_edges
    ADD CONSTRAINT chk_le_no_self_loop
        CHECK (source_asset_id != target_asset_id);

-- =============================================================================
-- Table 4: data_lineage_service.lineage_graph_snapshots
-- =============================================================================

CREATE TABLE IF NOT EXISTS data_lineage_service.lineage_graph_snapshots (
    id                    UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_name         VARCHAR(255),
    node_count            INTEGER      NOT NULL DEFAULT 0,
    edge_count            INTEGER      NOT NULL DEFAULT 0,
    max_depth             INTEGER      NOT NULL DEFAULT 0,
    connected_components  INTEGER      NOT NULL DEFAULT 0,
    orphan_count          INTEGER      NOT NULL DEFAULT 0,
    coverage_score        FLOAT        NOT NULL DEFAULT 0.0,
    graph_hash            VARCHAR(128),
    snapshot_data         JSONB        NOT NULL DEFAULT '{}'::jsonb,
    created_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

ALTER TABLE data_lineage_service.lineage_graph_snapshots
    ADD CONSTRAINT chk_lgs_node_count_non_negative CHECK (node_count >= 0);

ALTER TABLE data_lineage_service.lineage_graph_snapshots
    ADD CONSTRAINT chk_lgs_edge_count_non_negative CHECK (edge_count >= 0);

ALTER TABLE data_lineage_service.lineage_graph_snapshots
    ADD CONSTRAINT chk_lgs_max_depth_non_negative CHECK (max_depth >= 0);

ALTER TABLE data_lineage_service.lineage_graph_snapshots
    ADD CONSTRAINT chk_lgs_connected_components_non_negative CHECK (connected_components >= 0);

ALTER TABLE data_lineage_service.lineage_graph_snapshots
    ADD CONSTRAINT chk_lgs_orphan_count_non_negative CHECK (orphan_count >= 0);

ALTER TABLE data_lineage_service.lineage_graph_snapshots
    ADD CONSTRAINT chk_lgs_coverage_score_range
        CHECK (coverage_score >= 0.0 AND coverage_score <= 1.0);

-- =============================================================================
-- Table 5: data_lineage_service.lineage_impact_analyses
-- =============================================================================

CREATE TABLE IF NOT EXISTS data_lineage_service.lineage_impact_analyses (
    id                    UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    root_asset_id         UUID         NOT NULL,
    direction             VARCHAR(20)  NOT NULL,
    depth                 INTEGER      NOT NULL DEFAULT 0,
    affected_assets_count INTEGER      NOT NULL DEFAULT 0,
    critical_count        INTEGER      NOT NULL DEFAULT 0,
    high_count            INTEGER      NOT NULL DEFAULT 0,
    medium_count          INTEGER      NOT NULL DEFAULT 0,
    low_count             INTEGER      NOT NULL DEFAULT 0,
    blast_radius          FLOAT        NOT NULL DEFAULT 0.0,
    analysis_result       JSONB        NOT NULL DEFAULT '{}'::jsonb,
    created_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

ALTER TABLE data_lineage_service.lineage_impact_analyses
    ADD CONSTRAINT fk_lia_root_asset_id
        FOREIGN KEY (root_asset_id)
        REFERENCES data_lineage_service.lineage_assets(id)
        ON DELETE CASCADE;

ALTER TABLE data_lineage_service.lineage_impact_analyses
    ADD CONSTRAINT chk_lia_direction CHECK (direction IN (
        'forward', 'backward'
    ));

ALTER TABLE data_lineage_service.lineage_impact_analyses
    ADD CONSTRAINT chk_lia_depth_non_negative CHECK (depth >= 0);

ALTER TABLE data_lineage_service.lineage_impact_analyses
    ADD CONSTRAINT chk_lia_affected_assets_count_non_negative CHECK (affected_assets_count >= 0);

ALTER TABLE data_lineage_service.lineage_impact_analyses
    ADD CONSTRAINT chk_lia_critical_count_non_negative CHECK (critical_count >= 0);

ALTER TABLE data_lineage_service.lineage_impact_analyses
    ADD CONSTRAINT chk_lia_high_count_non_negative CHECK (high_count >= 0);

ALTER TABLE data_lineage_service.lineage_impact_analyses
    ADD CONSTRAINT chk_lia_medium_count_non_negative CHECK (medium_count >= 0);

ALTER TABLE data_lineage_service.lineage_impact_analyses
    ADD CONSTRAINT chk_lia_low_count_non_negative CHECK (low_count >= 0);

ALTER TABLE data_lineage_service.lineage_impact_analyses
    ADD CONSTRAINT chk_lia_blast_radius_non_negative CHECK (blast_radius >= 0.0);

-- =============================================================================
-- Table 6: data_lineage_service.lineage_validations
-- =============================================================================

CREATE TABLE IF NOT EXISTS data_lineage_service.lineage_validations (
    id                  UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    scope               VARCHAR(255) NOT NULL DEFAULT 'full',
    orphan_nodes        INTEGER      NOT NULL DEFAULT 0,
    broken_edges        INTEGER      NOT NULL DEFAULT 0,
    cycles_detected     INTEGER      NOT NULL DEFAULT 0,
    source_coverage     FLOAT        NOT NULL DEFAULT 0.0,
    completeness_score  FLOAT        NOT NULL DEFAULT 0.0,
    freshness_score     FLOAT        NOT NULL DEFAULT 0.0,
    issues_json         JSONB        NOT NULL DEFAULT '[]'::jsonb,
    recommendations_json JSONB       NOT NULL DEFAULT '[]'::jsonb,
    validated_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

ALTER TABLE data_lineage_service.lineage_validations
    ADD CONSTRAINT chk_lv_orphan_nodes_non_negative CHECK (orphan_nodes >= 0);

ALTER TABLE data_lineage_service.lineage_validations
    ADD CONSTRAINT chk_lv_broken_edges_non_negative CHECK (broken_edges >= 0);

ALTER TABLE data_lineage_service.lineage_validations
    ADD CONSTRAINT chk_lv_cycles_detected_non_negative CHECK (cycles_detected >= 0);

ALTER TABLE data_lineage_service.lineage_validations
    ADD CONSTRAINT chk_lv_source_coverage_range
        CHECK (source_coverage >= 0.0 AND source_coverage <= 1.0);

ALTER TABLE data_lineage_service.lineage_validations
    ADD CONSTRAINT chk_lv_completeness_score_range
        CHECK (completeness_score >= 0.0 AND completeness_score <= 1.0);

ALTER TABLE data_lineage_service.lineage_validations
    ADD CONSTRAINT chk_lv_freshness_score_range
        CHECK (freshness_score >= 0.0 AND freshness_score <= 1.0);

-- =============================================================================
-- Table 7: data_lineage_service.lineage_reports
-- =============================================================================

CREATE TABLE IF NOT EXISTS data_lineage_service.lineage_reports (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    report_type     VARCHAR(50)  NOT NULL,
    format          VARCHAR(20)  NOT NULL,
    scope           VARCHAR(255) NOT NULL DEFAULT 'full',
    parameters      JSONB        NOT NULL DEFAULT '{}'::jsonb,
    content         TEXT         NOT NULL DEFAULT '',
    report_hash     VARCHAR(128),
    generated_by    VARCHAR(255) NOT NULL DEFAULT 'data-lineage-tracker',
    generated_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

ALTER TABLE data_lineage_service.lineage_reports
    ADD CONSTRAINT chk_lr_report_type CHECK (report_type IN (
        'csrd_esrs', 'ghg_protocol', 'soc2', 'custom', 'visualization'
    ));

ALTER TABLE data_lineage_service.lineage_reports
    ADD CONSTRAINT chk_lr_format CHECK (format IN (
        'mermaid', 'dot', 'json', 'd3', 'text', 'html', 'pdf'
    ));

-- =============================================================================
-- Table 8: data_lineage_service.lineage_change_events
-- =============================================================================

CREATE TABLE IF NOT EXISTS data_lineage_service.lineage_change_events (
    id                    UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    previous_snapshot_id  UUID,
    current_snapshot_id   UUID,
    change_type           VARCHAR(50)  NOT NULL,
    entity_id             UUID,
    entity_type           VARCHAR(100),
    details               JSONB        NOT NULL DEFAULT '{}'::jsonb,
    severity              VARCHAR(20)  NOT NULL DEFAULT 'low',
    detected_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

ALTER TABLE data_lineage_service.lineage_change_events
    ADD CONSTRAINT fk_lce_previous_snapshot_id
        FOREIGN KEY (previous_snapshot_id)
        REFERENCES data_lineage_service.lineage_graph_snapshots(id)
        ON DELETE SET NULL;

ALTER TABLE data_lineage_service.lineage_change_events
    ADD CONSTRAINT fk_lce_current_snapshot_id
        FOREIGN KEY (current_snapshot_id)
        REFERENCES data_lineage_service.lineage_graph_snapshots(id)
        ON DELETE SET NULL;

ALTER TABLE data_lineage_service.lineage_change_events
    ADD CONSTRAINT chk_lce_change_type CHECK (change_type IN (
        'node_added', 'node_removed', 'edge_added', 'edge_removed', 'topology_changed'
    ));

ALTER TABLE data_lineage_service.lineage_change_events
    ADD CONSTRAINT chk_lce_severity CHECK (severity IN (
        'low', 'medium', 'high', 'critical'
    ));

-- =============================================================================
-- Table 9: data_lineage_service.lineage_quality_scores
-- =============================================================================

CREATE TABLE IF NOT EXISTS data_lineage_service.lineage_quality_scores (
    id                        UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id                  UUID         NOT NULL,
    source_credibility        FLOAT        NOT NULL DEFAULT 0.0,
    transformation_depth      INTEGER      NOT NULL DEFAULT 0,
    freshness_score           FLOAT        NOT NULL DEFAULT 0.0,
    documentation_score       FLOAT        NOT NULL DEFAULT 0.0,
    manual_intervention_count INTEGER      NOT NULL DEFAULT 0,
    overall_score             FLOAT        NOT NULL DEFAULT 0.0,
    scoring_details           JSONB        NOT NULL DEFAULT '{}'::jsonb,
    scored_at                 TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

ALTER TABLE data_lineage_service.lineage_quality_scores
    ADD CONSTRAINT fk_lqs_asset_id
        FOREIGN KEY (asset_id)
        REFERENCES data_lineage_service.lineage_assets(id)
        ON DELETE CASCADE;

ALTER TABLE data_lineage_service.lineage_quality_scores
    ADD CONSTRAINT chk_lqs_source_credibility_range
        CHECK (source_credibility >= 0.0 AND source_credibility <= 1.0);

ALTER TABLE data_lineage_service.lineage_quality_scores
    ADD CONSTRAINT chk_lqs_transformation_depth_non_negative CHECK (transformation_depth >= 0);

ALTER TABLE data_lineage_service.lineage_quality_scores
    ADD CONSTRAINT chk_lqs_freshness_score_range
        CHECK (freshness_score >= 0.0 AND freshness_score <= 1.0);

ALTER TABLE data_lineage_service.lineage_quality_scores
    ADD CONSTRAINT chk_lqs_documentation_score_range
        CHECK (documentation_score >= 0.0 AND documentation_score <= 1.0);

ALTER TABLE data_lineage_service.lineage_quality_scores
    ADD CONSTRAINT chk_lqs_manual_intervention_count_non_negative CHECK (manual_intervention_count >= 0);

ALTER TABLE data_lineage_service.lineage_quality_scores
    ADD CONSTRAINT chk_lqs_overall_score_range
        CHECK (overall_score >= 0.0 AND overall_score <= 1.0);

-- =============================================================================
-- Table 10: data_lineage_service.lineage_audit_log
-- =============================================================================

CREATE TABLE IF NOT EXISTS data_lineage_service.lineage_audit_log (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    action          VARCHAR(100) NOT NULL,
    entity_type     VARCHAR(100) NOT NULL,
    entity_id       UUID,
    actor           VARCHAR(255) NOT NULL DEFAULT 'system',
    details_json    JSONB        NOT NULL DEFAULT '{}'::jsonb,
    previous_state  JSONB,
    new_state       JSONB,
    provenance_hash VARCHAR(128) NOT NULL DEFAULT '',
    parent_hash     VARCHAR(128) NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

ALTER TABLE data_lineage_service.lineage_audit_log
    ADD CONSTRAINT chk_lal_action_not_empty CHECK (LENGTH(TRIM(action)) > 0);

ALTER TABLE data_lineage_service.lineage_audit_log
    ADD CONSTRAINT chk_lal_entity_type_not_empty CHECK (LENGTH(TRIM(entity_type)) > 0);

ALTER TABLE data_lineage_service.lineage_audit_log
    ADD CONSTRAINT chk_lal_action CHECK (action IN (
        'asset_registered', 'asset_updated', 'asset_deprecated', 'asset_archived',
        'transformation_created', 'transformation_completed', 'transformation_failed',
        'edge_created', 'edge_updated', 'edge_deleted',
        'snapshot_created', 'snapshot_compared',
        'impact_analysis_started', 'impact_analysis_completed',
        'validation_started', 'validation_completed', 'validation_failed',
        'report_generated', 'report_exported',
        'change_detected', 'change_acknowledged', 'change_resolved',
        'quality_scored', 'quality_updated',
        'graph_rebuilt', 'graph_pruned',
        'export_generated', 'import_completed', 'config_changed'
    ));

ALTER TABLE data_lineage_service.lineage_audit_log
    ADD CONSTRAINT chk_lal_entity_type CHECK (entity_type IN (
        'asset', 'transformation', 'edge', 'graph_snapshot',
        'impact_analysis', 'validation', 'report', 'change_event',
        'quality_score', 'config'
    ));

-- =============================================================================
-- Table 11: data_lineage_service.lineage_transformation_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS data_lineage_service.lineage_transformation_events (
    ts                   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    agent_id             VARCHAR(255),
    pipeline_id          VARCHAR(255),
    transformation_type  VARCHAR(50),
    records_in           INTEGER      NOT NULL DEFAULT 0,
    records_out          INTEGER      NOT NULL DEFAULT 0,
    duration_ms          INTEGER      NOT NULL DEFAULT 0
);

SELECT create_hypertable(
    'data_lineage_service.lineage_transformation_events',
    'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE data_lineage_service.lineage_transformation_events
    ADD CONSTRAINT chk_lte_records_in_non_negative CHECK (records_in >= 0);

ALTER TABLE data_lineage_service.lineage_transformation_events
    ADD CONSTRAINT chk_lte_records_out_non_negative CHECK (records_out >= 0);

ALTER TABLE data_lineage_service.lineage_transformation_events
    ADD CONSTRAINT chk_lte_duration_ms_non_negative CHECK (duration_ms >= 0);

-- =============================================================================
-- Table 12: data_lineage_service.lineage_validation_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS data_lineage_service.lineage_validation_events (
    ts                TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    scope             VARCHAR(255),
    orphan_count      INTEGER      NOT NULL DEFAULT 0,
    broken_count      INTEGER      NOT NULL DEFAULT 0,
    coverage_score    FLOAT        NOT NULL DEFAULT 0.0,
    completeness_score FLOAT       NOT NULL DEFAULT 0.0
);

SELECT create_hypertable(
    'data_lineage_service.lineage_validation_events',
    'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE data_lineage_service.lineage_validation_events
    ADD CONSTRAINT chk_lve_orphan_count_non_negative CHECK (orphan_count >= 0);

ALTER TABLE data_lineage_service.lineage_validation_events
    ADD CONSTRAINT chk_lve_broken_count_non_negative CHECK (broken_count >= 0);

ALTER TABLE data_lineage_service.lineage_validation_events
    ADD CONSTRAINT chk_lve_coverage_score_range
        CHECK (coverage_score >= 0.0 AND coverage_score <= 1.0);

ALTER TABLE data_lineage_service.lineage_validation_events
    ADD CONSTRAINT chk_lve_completeness_score_range
        CHECK (completeness_score >= 0.0 AND completeness_score <= 1.0);

-- =============================================================================
-- Table 13: data_lineage_service.lineage_impact_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS data_lineage_service.lineage_impact_events (
    ts              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    root_asset_id   UUID,
    direction       VARCHAR(20),
    affected_count  INTEGER      NOT NULL DEFAULT 0,
    critical_count  INTEGER      NOT NULL DEFAULT 0,
    blast_radius    FLOAT        NOT NULL DEFAULT 0.0
);

SELECT create_hypertable(
    'data_lineage_service.lineage_impact_events',
    'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE data_lineage_service.lineage_impact_events
    ADD CONSTRAINT chk_lie_affected_count_non_negative CHECK (affected_count >= 0);

ALTER TABLE data_lineage_service.lineage_impact_events
    ADD CONSTRAINT chk_lie_critical_count_non_negative CHECK (critical_count >= 0);

ALTER TABLE data_lineage_service.lineage_impact_events
    ADD CONSTRAINT chk_lie_blast_radius_non_negative CHECK (blast_radius >= 0.0);

ALTER TABLE data_lineage_service.lineage_impact_events
    ADD CONSTRAINT chk_lie_direction CHECK (
        direction IS NULL OR direction IN ('forward', 'backward')
    );

-- =============================================================================
-- Continuous Aggregates
-- =============================================================================

-- lineage_transformations_hourly_stats: hourly rollup of lineage_transformation_events
CREATE MATERIALIZED VIEW data_lineage_service.lineage_transformations_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', ts)      AS bucket,
    transformation_type,
    agent_id,
    pipeline_id,
    COUNT(*)                        AS total_events,
    SUM(records_in)                 AS total_records_in,
    SUM(records_out)                AS total_records_out,
    AVG(duration_ms)                AS avg_duration_ms
FROM data_lineage_service.lineage_transformation_events
WHERE ts IS NOT NULL
GROUP BY bucket, transformation_type, agent_id, pipeline_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'data_lineage_service.lineage_transformations_hourly_stats',
    start_offset      => INTERVAL '2 hours',
    end_offset        => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- lineage_validations_hourly_stats: hourly rollup of lineage_validation_events
CREATE MATERIALIZED VIEW data_lineage_service.lineage_validations_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', ts)      AS bucket,
    scope,
    AVG(coverage_score)             AS avg_coverage_score,
    AVG(completeness_score)         AS avg_completeness_score,
    SUM(orphan_count)               AS total_orphan_count,
    SUM(broken_count)               AS total_broken_count
FROM data_lineage_service.lineage_validation_events
WHERE ts IS NOT NULL
GROUP BY bucket, scope
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'data_lineage_service.lineage_validations_hourly_stats',
    start_offset      => INTERVAL '2 hours',
    end_offset        => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- lineage_assets indexes (10)
CREATE INDEX IF NOT EXISTS idx_la_qualified_name       ON data_lineage_service.lineage_assets(qualified_name);
CREATE INDEX IF NOT EXISTS idx_la_asset_type           ON data_lineage_service.lineage_assets(asset_type);
CREATE INDEX IF NOT EXISTS idx_la_owner                ON data_lineage_service.lineage_assets(owner);
CREATE INDEX IF NOT EXISTS idx_la_classification       ON data_lineage_service.lineage_assets(classification);
CREATE INDEX IF NOT EXISTS idx_la_status               ON data_lineage_service.lineage_assets(status);
CREATE INDEX IF NOT EXISTS idx_la_created_at           ON data_lineage_service.lineage_assets(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_la_updated_at           ON data_lineage_service.lineage_assets(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_la_asset_type_status    ON data_lineage_service.lineage_assets(asset_type, status);
CREATE INDEX IF NOT EXISTS idx_la_tags                 ON data_lineage_service.lineage_assets USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_la_metadata             ON data_lineage_service.lineage_assets USING GIN (metadata);

-- lineage_transformations indexes (10)
CREATE INDEX IF NOT EXISTS idx_lt_transformation_type  ON data_lineage_service.lineage_transformations(transformation_type);
CREATE INDEX IF NOT EXISTS idx_lt_agent_id             ON data_lineage_service.lineage_transformations(agent_id);
CREATE INDEX IF NOT EXISTS idx_lt_pipeline_id          ON data_lineage_service.lineage_transformations(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_lt_execution_id         ON data_lineage_service.lineage_transformations(execution_id);
CREATE INDEX IF NOT EXISTS idx_lt_started_at           ON data_lineage_service.lineage_transformations(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_lt_completed_at         ON data_lineage_service.lineage_transformations(completed_at DESC);
CREATE INDEX IF NOT EXISTS idx_lt_duration_ms          ON data_lineage_service.lineage_transformations(duration_ms DESC);
CREATE INDEX IF NOT EXISTS idx_lt_agent_type           ON data_lineage_service.lineage_transformations(agent_id, transformation_type);
CREATE INDEX IF NOT EXISTS idx_lt_pipeline_type        ON data_lineage_service.lineage_transformations(pipeline_id, transformation_type);
CREATE INDEX IF NOT EXISTS idx_lt_parameters           ON data_lineage_service.lineage_transformations USING GIN (parameters);

-- lineage_edges indexes (10)
CREATE INDEX IF NOT EXISTS idx_le_source_asset_id      ON data_lineage_service.lineage_edges(source_asset_id);
CREATE INDEX IF NOT EXISTS idx_le_target_asset_id      ON data_lineage_service.lineage_edges(target_asset_id);
CREATE INDEX IF NOT EXISTS idx_le_transformation_id    ON data_lineage_service.lineage_edges(transformation_id);
CREATE INDEX IF NOT EXISTS idx_le_edge_type            ON data_lineage_service.lineage_edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_le_source_field         ON data_lineage_service.lineage_edges(source_field);
CREATE INDEX IF NOT EXISTS idx_le_target_field         ON data_lineage_service.lineage_edges(target_field);
CREATE INDEX IF NOT EXISTS idx_le_confidence           ON data_lineage_service.lineage_edges(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_le_created_at           ON data_lineage_service.lineage_edges(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_le_source_target        ON data_lineage_service.lineage_edges(source_asset_id, target_asset_id);
CREATE INDEX IF NOT EXISTS idx_le_source_edge_type     ON data_lineage_service.lineage_edges(source_asset_id, edge_type);

-- lineage_graph_snapshots indexes (8)
CREATE INDEX IF NOT EXISTS idx_lgs_snapshot_name       ON data_lineage_service.lineage_graph_snapshots(snapshot_name);
CREATE INDEX IF NOT EXISTS idx_lgs_created_at          ON data_lineage_service.lineage_graph_snapshots(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lgs_graph_hash          ON data_lineage_service.lineage_graph_snapshots(graph_hash);
CREATE INDEX IF NOT EXISTS idx_lgs_node_count          ON data_lineage_service.lineage_graph_snapshots(node_count DESC);
CREATE INDEX IF NOT EXISTS idx_lgs_edge_count          ON data_lineage_service.lineage_graph_snapshots(edge_count DESC);
CREATE INDEX IF NOT EXISTS idx_lgs_coverage_score      ON data_lineage_service.lineage_graph_snapshots(coverage_score DESC);
CREATE INDEX IF NOT EXISTS idx_lgs_orphan_count        ON data_lineage_service.lineage_graph_snapshots(orphan_count DESC);
CREATE INDEX IF NOT EXISTS idx_lgs_snapshot_data       ON data_lineage_service.lineage_graph_snapshots USING GIN (snapshot_data);

-- lineage_impact_analyses indexes (10)
CREATE INDEX IF NOT EXISTS idx_lia_root_asset_id       ON data_lineage_service.lineage_impact_analyses(root_asset_id);
CREATE INDEX IF NOT EXISTS idx_lia_direction           ON data_lineage_service.lineage_impact_analyses(direction);
CREATE INDEX IF NOT EXISTS idx_lia_created_at          ON data_lineage_service.lineage_impact_analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lia_affected_count      ON data_lineage_service.lineage_impact_analyses(affected_assets_count DESC);
CREATE INDEX IF NOT EXISTS idx_lia_blast_radius        ON data_lineage_service.lineage_impact_analyses(blast_radius DESC);
CREATE INDEX IF NOT EXISTS idx_lia_critical_count      ON data_lineage_service.lineage_impact_analyses(critical_count DESC);
CREATE INDEX IF NOT EXISTS idx_lia_root_direction      ON data_lineage_service.lineage_impact_analyses(root_asset_id, direction);
CREATE INDEX IF NOT EXISTS idx_lia_root_created        ON data_lineage_service.lineage_impact_analyses(root_asset_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lia_direction_created   ON data_lineage_service.lineage_impact_analyses(direction, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lia_analysis_result     ON data_lineage_service.lineage_impact_analyses USING GIN (analysis_result);

-- lineage_validations indexes (8)
CREATE INDEX IF NOT EXISTS idx_lv_scope                ON data_lineage_service.lineage_validations(scope);
CREATE INDEX IF NOT EXISTS idx_lv_validated_at         ON data_lineage_service.lineage_validations(validated_at DESC);
CREATE INDEX IF NOT EXISTS idx_lv_completeness_score   ON data_lineage_service.lineage_validations(completeness_score DESC);
CREATE INDEX IF NOT EXISTS idx_lv_freshness_score      ON data_lineage_service.lineage_validations(freshness_score DESC);
CREATE INDEX IF NOT EXISTS idx_lv_source_coverage      ON data_lineage_service.lineage_validations(source_coverage DESC);
CREATE INDEX IF NOT EXISTS idx_lv_orphan_nodes         ON data_lineage_service.lineage_validations(orphan_nodes DESC);
CREATE INDEX IF NOT EXISTS idx_lv_issues               ON data_lineage_service.lineage_validations USING GIN (issues_json);
CREATE INDEX IF NOT EXISTS idx_lv_recommendations      ON data_lineage_service.lineage_validations USING GIN (recommendations_json);

-- lineage_reports indexes (8)
CREATE INDEX IF NOT EXISTS idx_lr_report_type          ON data_lineage_service.lineage_reports(report_type);
CREATE INDEX IF NOT EXISTS idx_lr_format               ON data_lineage_service.lineage_reports(format);
CREATE INDEX IF NOT EXISTS idx_lr_scope                ON data_lineage_service.lineage_reports(scope);
CREATE INDEX IF NOT EXISTS idx_lr_generated_by         ON data_lineage_service.lineage_reports(generated_by);
CREATE INDEX IF NOT EXISTS idx_lr_generated_at         ON data_lineage_service.lineage_reports(generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_lr_report_hash          ON data_lineage_service.lineage_reports(report_hash);
CREATE INDEX IF NOT EXISTS idx_lr_type_format          ON data_lineage_service.lineage_reports(report_type, format);
CREATE INDEX IF NOT EXISTS idx_lr_parameters           ON data_lineage_service.lineage_reports USING GIN (parameters);

-- lineage_change_events indexes (10)
CREATE INDEX IF NOT EXISTS idx_lce_previous_snapshot_id ON data_lineage_service.lineage_change_events(previous_snapshot_id);
CREATE INDEX IF NOT EXISTS idx_lce_current_snapshot_id  ON data_lineage_service.lineage_change_events(current_snapshot_id);
CREATE INDEX IF NOT EXISTS idx_lce_change_type          ON data_lineage_service.lineage_change_events(change_type);
CREATE INDEX IF NOT EXISTS idx_lce_entity_id            ON data_lineage_service.lineage_change_events(entity_id);
CREATE INDEX IF NOT EXISTS idx_lce_entity_type          ON data_lineage_service.lineage_change_events(entity_type);
CREATE INDEX IF NOT EXISTS idx_lce_severity             ON data_lineage_service.lineage_change_events(severity);
CREATE INDEX IF NOT EXISTS idx_lce_detected_at          ON data_lineage_service.lineage_change_events(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_lce_change_severity      ON data_lineage_service.lineage_change_events(change_type, severity);
CREATE INDEX IF NOT EXISTS idx_lce_snapshot_pair        ON data_lineage_service.lineage_change_events(previous_snapshot_id, current_snapshot_id);
CREATE INDEX IF NOT EXISTS idx_lce_details              ON data_lineage_service.lineage_change_events USING GIN (details);

-- lineage_quality_scores indexes (9)
CREATE INDEX IF NOT EXISTS idx_lqs_asset_id             ON data_lineage_service.lineage_quality_scores(asset_id);
CREATE INDEX IF NOT EXISTS idx_lqs_scored_at            ON data_lineage_service.lineage_quality_scores(scored_at DESC);
CREATE INDEX IF NOT EXISTS idx_lqs_overall_score        ON data_lineage_service.lineage_quality_scores(overall_score DESC);
CREATE INDEX IF NOT EXISTS idx_lqs_source_credibility   ON data_lineage_service.lineage_quality_scores(source_credibility DESC);
CREATE INDEX IF NOT EXISTS idx_lqs_freshness_score      ON data_lineage_service.lineage_quality_scores(freshness_score DESC);
CREATE INDEX IF NOT EXISTS idx_lqs_documentation_score  ON data_lineage_service.lineage_quality_scores(documentation_score DESC);
CREATE INDEX IF NOT EXISTS idx_lqs_transformation_depth ON data_lineage_service.lineage_quality_scores(transformation_depth DESC);
CREATE INDEX IF NOT EXISTS idx_lqs_asset_scored         ON data_lineage_service.lineage_quality_scores(asset_id, scored_at DESC);
CREATE INDEX IF NOT EXISTS idx_lqs_scoring_details      ON data_lineage_service.lineage_quality_scores USING GIN (scoring_details);

-- lineage_audit_log indexes (10)
CREATE INDEX IF NOT EXISTS idx_lal_action               ON data_lineage_service.lineage_audit_log(action);
CREATE INDEX IF NOT EXISTS idx_lal_entity_type          ON data_lineage_service.lineage_audit_log(entity_type);
CREATE INDEX IF NOT EXISTS idx_lal_entity_id            ON data_lineage_service.lineage_audit_log(entity_id);
CREATE INDEX IF NOT EXISTS idx_lal_actor                ON data_lineage_service.lineage_audit_log(actor);
CREATE INDEX IF NOT EXISTS idx_lal_created_at           ON data_lineage_service.lineage_audit_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lal_provenance_hash      ON data_lineage_service.lineage_audit_log(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_lal_parent_hash          ON data_lineage_service.lineage_audit_log(parent_hash);
CREATE INDEX IF NOT EXISTS idx_lal_entity_type_id       ON data_lineage_service.lineage_audit_log(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_lal_action_created       ON data_lineage_service.lineage_audit_log(action, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lal_details              ON data_lineage_service.lineage_audit_log USING GIN (details_json);

-- lineage_transformation_events indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_lte_agent_id             ON data_lineage_service.lineage_transformation_events(agent_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_lte_pipeline_id          ON data_lineage_service.lineage_transformation_events(pipeline_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_lte_transformation_type  ON data_lineage_service.lineage_transformation_events(transformation_type, ts DESC);
CREATE INDEX IF NOT EXISTS idx_lte_agent_pipeline       ON data_lineage_service.lineage_transformation_events(agent_id, pipeline_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_lte_records_in           ON data_lineage_service.lineage_transformation_events(records_in DESC, ts DESC);
CREATE INDEX IF NOT EXISTS idx_lte_duration_ms          ON data_lineage_service.lineage_transformation_events(duration_ms DESC, ts DESC);

-- lineage_validation_events indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_lve_scope                ON data_lineage_service.lineage_validation_events(scope, ts DESC);
CREATE INDEX IF NOT EXISTS idx_lve_orphan_count         ON data_lineage_service.lineage_validation_events(orphan_count DESC, ts DESC);
CREATE INDEX IF NOT EXISTS idx_lve_broken_count         ON data_lineage_service.lineage_validation_events(broken_count DESC, ts DESC);
CREATE INDEX IF NOT EXISTS idx_lve_coverage_score       ON data_lineage_service.lineage_validation_events(coverage_score DESC, ts DESC);
CREATE INDEX IF NOT EXISTS idx_lve_completeness_score   ON data_lineage_service.lineage_validation_events(completeness_score DESC, ts DESC);
CREATE INDEX IF NOT EXISTS idx_lve_scope_coverage       ON data_lineage_service.lineage_validation_events(scope, coverage_score DESC, ts DESC);

-- lineage_impact_events indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_lie_root_asset_id        ON data_lineage_service.lineage_impact_events(root_asset_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_lie_direction            ON data_lineage_service.lineage_impact_events(direction, ts DESC);
CREATE INDEX IF NOT EXISTS idx_lie_affected_count       ON data_lineage_service.lineage_impact_events(affected_count DESC, ts DESC);
CREATE INDEX IF NOT EXISTS idx_lie_critical_count       ON data_lineage_service.lineage_impact_events(critical_count DESC, ts DESC);
CREATE INDEX IF NOT EXISTS idx_lie_blast_radius         ON data_lineage_service.lineage_impact_events(blast_radius DESC, ts DESC);
CREATE INDEX IF NOT EXISTS idx_lie_root_direction       ON data_lineage_service.lineage_impact_events(root_asset_id, direction, ts DESC);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

-- lineage_assets: global resource — no tenant_id; admin-gated writes via is_admin
ALTER TABLE data_lineage_service.lineage_assets ENABLE ROW LEVEL SECURITY;
CREATE POLICY la_read  ON data_lineage_service.lineage_assets FOR SELECT USING (TRUE);
CREATE POLICY la_write ON data_lineage_service.lineage_assets FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE data_lineage_service.lineage_transformations ENABLE ROW LEVEL SECURITY;
CREATE POLICY lt_read  ON data_lineage_service.lineage_transformations FOR SELECT USING (TRUE);
CREATE POLICY lt_write ON data_lineage_service.lineage_transformations FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE data_lineage_service.lineage_edges ENABLE ROW LEVEL SECURITY;
CREATE POLICY le_read  ON data_lineage_service.lineage_edges FOR SELECT USING (TRUE);
CREATE POLICY le_write ON data_lineage_service.lineage_edges FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE data_lineage_service.lineage_graph_snapshots ENABLE ROW LEVEL SECURITY;
CREATE POLICY lgs_read  ON data_lineage_service.lineage_graph_snapshots FOR SELECT USING (TRUE);
CREATE POLICY lgs_write ON data_lineage_service.lineage_graph_snapshots FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE data_lineage_service.lineage_impact_analyses ENABLE ROW LEVEL SECURITY;
CREATE POLICY lia_read  ON data_lineage_service.lineage_impact_analyses FOR SELECT USING (TRUE);
CREATE POLICY lia_write ON data_lineage_service.lineage_impact_analyses FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE data_lineage_service.lineage_validations ENABLE ROW LEVEL SECURITY;
CREATE POLICY lv_read  ON data_lineage_service.lineage_validations FOR SELECT USING (TRUE);
CREATE POLICY lv_write ON data_lineage_service.lineage_validations FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE data_lineage_service.lineage_reports ENABLE ROW LEVEL SECURITY;
CREATE POLICY lr_read  ON data_lineage_service.lineage_reports FOR SELECT USING (TRUE);
CREATE POLICY lr_write ON data_lineage_service.lineage_reports FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE data_lineage_service.lineage_change_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY lce_read  ON data_lineage_service.lineage_change_events FOR SELECT USING (TRUE);
CREATE POLICY lce_write ON data_lineage_service.lineage_change_events FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE data_lineage_service.lineage_quality_scores ENABLE ROW LEVEL SECURITY;
CREATE POLICY lqs_read  ON data_lineage_service.lineage_quality_scores FOR SELECT USING (TRUE);
CREATE POLICY lqs_write ON data_lineage_service.lineage_quality_scores FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE data_lineage_service.lineage_audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY lal_read  ON data_lineage_service.lineage_audit_log FOR SELECT USING (TRUE);
CREATE POLICY lal_write ON data_lineage_service.lineage_audit_log FOR ALL   USING (
    current_setting('app.is_admin', true) = 'true'
    OR current_setting('app.current_tenant', true) IS NOT NULL
);

ALTER TABLE data_lineage_service.lineage_transformation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY lte_read  ON data_lineage_service.lineage_transformation_events FOR SELECT USING (TRUE);
CREATE POLICY lte_write ON data_lineage_service.lineage_transformation_events FOR ALL   USING (TRUE);

ALTER TABLE data_lineage_service.lineage_validation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY lve_read  ON data_lineage_service.lineage_validation_events FOR SELECT USING (TRUE);
CREATE POLICY lve_write ON data_lineage_service.lineage_validation_events FOR ALL   USING (TRUE);

ALTER TABLE data_lineage_service.lineage_impact_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY lie_read  ON data_lineage_service.lineage_impact_events FOR SELECT USING (TRUE);
CREATE POLICY lie_write ON data_lineage_service.lineage_impact_events FOR ALL   USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA data_lineage_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA data_lineage_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA data_lineage_service TO greenlang_app;
GRANT SELECT ON data_lineage_service.lineage_transformations_hourly_stats TO greenlang_app;
GRANT SELECT ON data_lineage_service.lineage_validations_hourly_stats TO greenlang_app;

GRANT USAGE ON SCHEMA data_lineage_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA data_lineage_service TO greenlang_readonly;
GRANT SELECT ON data_lineage_service.lineage_transformations_hourly_stats TO greenlang_readonly;
GRANT SELECT ON data_lineage_service.lineage_validations_hourly_stats TO greenlang_readonly;

GRANT ALL ON SCHEMA data_lineage_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA data_lineage_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA data_lineage_service TO greenlang_admin;

INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'data-lineage:assets:read',          'data-lineage', 'assets_read',          'View registered lineage assets and their metadata'),
    (gen_random_uuid(), 'data-lineage:assets:write',         'data-lineage', 'assets_write',         'Register, update, deprecate, and archive lineage assets'),
    (gen_random_uuid(), 'data-lineage:transformations:read', 'data-lineage', 'transformations_read', 'View transformation operations and record counts'),
    (gen_random_uuid(), 'data-lineage:transformations:write','data-lineage', 'transformations_write','Create and track transformation operations'),
    (gen_random_uuid(), 'data-lineage:edges:read',           'data-lineage', 'edges_read',           'View dataset-level and column-level lineage edges'),
    (gen_random_uuid(), 'data-lineage:edges:write',          'data-lineage', 'edges_write',          'Create, update, and delete lineage edges'),
    (gen_random_uuid(), 'data-lineage:snapshots:read',       'data-lineage', 'snapshots_read',       'View graph snapshots and topology metrics'),
    (gen_random_uuid(), 'data-lineage:snapshots:write',      'data-lineage', 'snapshots_write',      'Create graph snapshots and compare topology changes'),
    (gen_random_uuid(), 'data-lineage:impact:read',          'data-lineage', 'impact_read',          'View forward/backward impact analysis results'),
    (gen_random_uuid(), 'data-lineage:impact:write',         'data-lineage', 'impact_write',         'Run impact analyses and calculate blast radius'),
    (gen_random_uuid(), 'data-lineage:validations:read',     'data-lineage', 'validations_read',     'View lineage validation results and quality scores'),
    (gen_random_uuid(), 'data-lineage:validations:write',    'data-lineage', 'validations_write',    'Run lineage validations and detect orphans/cycles'),
    (gen_random_uuid(), 'data-lineage:reports:read',         'data-lineage', 'reports_read',         'View generated lineage compliance and visualization reports'),
    (gen_random_uuid(), 'data-lineage:reports:write',        'data-lineage', 'reports_write',        'Generate CSRD/ESRS, GHG Protocol, SOC 2, and custom lineage reports'),
    (gen_random_uuid(), 'data-lineage:changes:read',         'data-lineage', 'changes_read',         'View lineage graph change events and topology diffs'),
    (gen_random_uuid(), 'data-lineage:changes:write',        'data-lineage', 'changes_write',        'Detect, acknowledge, and resolve lineage change events'),
    (gen_random_uuid(), 'data-lineage:quality:read',         'data-lineage', 'quality_read',         'View per-asset data quality scores and scoring details'),
    (gen_random_uuid(), 'data-lineage:quality:write',        'data-lineage', 'quality_write',        'Calculate and update per-asset data quality scores'),
    (gen_random_uuid(), 'data-lineage:audit:read',           'data-lineage', 'audit_read',           'View lineage audit log entries and provenance chains'),
    (gen_random_uuid(), 'data-lineage:admin',                'data-lineage', 'admin',                'Data lineage tracker service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

SELECT add_retention_policy('data_lineage_service.lineage_transformation_events', INTERVAL '90 days');
SELECT add_retention_policy('data_lineage_service.lineage_validation_events',     INTERVAL '90 days');
SELECT add_retention_policy('data_lineage_service.lineage_impact_events',         INTERVAL '90 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

ALTER TABLE data_lineage_service.lineage_transformation_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'agent_id',
         timescaledb.compress_orderby   = 'ts DESC');
SELECT add_compression_policy('data_lineage_service.lineage_transformation_events', INTERVAL '7 days');

ALTER TABLE data_lineage_service.lineage_validation_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'scope',
         timescaledb.compress_orderby   = 'ts DESC');
SELECT add_compression_policy('data_lineage_service.lineage_validation_events', INTERVAL '7 days');

ALTER TABLE data_lineage_service.lineage_impact_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'root_asset_id',
         timescaledb.compress_orderby   = 'ts DESC');
SELECT add_compression_policy('data_lineage_service.lineage_impact_events', INTERVAL '7 days');

-- =============================================================================
-- Seed: Register the Data Lineage Tracker (GL-DATA-X-021)
-- =============================================================================

INSERT INTO agent_registry_service.agents (
    agent_id, name, description, layer, execution_mode,
    idempotency_support, deterministic, max_concurrent_runs,
    glip_version, supports_checkpointing, author,
    documentation_url, enabled, tenant_id
) VALUES (
    'GL-DATA-X-021',
    'Data Lineage Tracker',
    'Data lineage tracking engine for GreenLang Climate OS. Maintains a registry of lineage assets (dataset/field/agent/pipeline/report/metric/external_source) with qualified names, ownership, classification (public/internal/confidential/restricted), and lifecycle status (active/deprecated/archived). Tracks transformations across 12 types (filter/aggregate/join/calculate/impute/deduplicate/enrich/merge/split/validate/normalize/classify) with agent/pipeline/execution refs, record counts (in/out/filtered/error), and duration tracking. Manages dataset-level and column-level lineage edges with source/target field resolution, transformation logic, and confidence scoring. Captures point-in-time graph snapshots with node/edge/depth/component counts, orphan detection, and coverage scoring with SHA-256 graph hashing. Performs forward/backward impact analysis with blast radius calculation and severity-bucketed affected asset counts (critical/high/medium/low). Validates lineage graph integrity via orphan node detection, broken edge detection, cycle detection, source coverage, and completeness/freshness scoring with structured issues and recommendations. Generates regulatory compliance reports (CSRD/ESRS, GHG Protocol, SOC 2, custom) in multiple visualization formats (Mermaid/DOT/D3/JSON/text/HTML/PDF). Tracks graph change events (node/edge added/removed, topology changes) with severity classification. Scores per-asset data quality (source credibility, transformation depth, freshness, documentation, manual intervention counts). SHA-256 provenance chains for zero-hallucination audit trail.',
    2, 'async', true, true, 5, '1.0.0', true,
    'GreenLang Data Team',
    'https://docs.greenlang.ai/agents/data-lineage-tracker',
    true, 'default'
) ON CONFLICT (agent_id) DO NOTHING;

INSERT INTO agent_registry_service.agent_versions (
    agent_id, version, resource_profile, container_spec,
    tags, sectors, provenance_hash
) VALUES (
    'GL-DATA-X-021', '1.0.0',
    '{"cpu_request": "250m", "cpu_limit": "1000m", "memory_request": "512Mi", "memory_limit": "2Gi", "gpu": false}'::jsonb,
    '{"image": "greenlang/data-lineage-tracker-service", "tag": "1.0.0", "port": 8000}'::jsonb,
    '{"data-lineage", "provenance", "impact-analysis", "graph-tracking", "compliance-reporting", "data-governance"}',
    '{"cross-sector", "manufacturing", "retail", "energy", "finance", "agriculture", "utilities"}',
    'e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6'
) ON CONFLICT (agent_id, version) DO NOTHING;

INSERT INTO agent_registry_service.agent_capabilities (
    agent_id, version, name, category,
    description, input_types, output_types, parameters
) VALUES
(
    'GL-DATA-X-021', '1.0.0',
    'asset_registration',
    'configuration',
    'Register lineage assets with qualified names, asset types, ownership, classification, and schema references.',
    '{"qualified_name", "asset_type", "display_name", "owner", "tags", "classification", "schema_ref"}',
    '{"asset_id", "registration_status", "validation_result"}',
    '{"asset_types": ["dataset", "field", "agent", "pipeline", "report", "metric", "external_source"], "classifications": ["public", "internal", "confidential", "restricted"], "statuses": ["active", "deprecated", "archived"]}'::jsonb
),
(
    'GL-DATA-X-021', '1.0.0',
    'transformation_tracking',
    'processing',
    'Track data transformations across 12 types with agent/pipeline/execution refs, record counts, and duration.',
    '{"transformation_type", "agent_id", "pipeline_id", "execution_id", "parameters"}',
    '{"transformation_id", "records_in", "records_out", "records_filtered", "records_error", "duration_ms"}',
    '{"transformation_types": ["filter", "aggregate", "join", "calculate", "impute", "deduplicate", "enrich", "merge", "split", "validate", "normalize", "classify"]}'::jsonb
),
(
    'GL-DATA-X-021', '1.0.0',
    'edge_management',
    'configuration',
    'Create and manage dataset-level and column-level lineage edges with transformation logic and confidence scoring.',
    '{"source_asset_id", "target_asset_id", "transformation_id", "edge_type", "source_field", "target_field", "transformation_logic"}',
    '{"edge_id", "confidence", "validation_result"}',
    '{"edge_types": ["dataset_level", "column_level"], "confidence_range": [0.0, 1.0]}'::jsonb
),
(
    'GL-DATA-X-021', '1.0.0',
    'graph_snapshot',
    'analysis',
    'Capture point-in-time graph snapshots with topology metrics, orphan detection, and coverage scoring.',
    '{"snapshot_name", "scope"}',
    '{"snapshot_id", "node_count", "edge_count", "max_depth", "connected_components", "orphan_count", "coverage_score", "graph_hash"}',
    '{"captures_topology": true, "sha256_graph_hash": true}'::jsonb
),
(
    'GL-DATA-X-021', '1.0.0',
    'impact_analysis',
    'analysis',
    'Perform forward/backward impact analysis with blast radius calculation and severity-bucketed affected asset counts.',
    '{"root_asset_id", "direction", "max_depth"}',
    '{"analysis_id", "affected_assets_count", "critical_count", "high_count", "medium_count", "low_count", "blast_radius", "analysis_result"}',
    '{"directions": ["forward", "backward"], "severity_buckets": ["critical", "high", "medium", "low"]}'::jsonb
),
(
    'GL-DATA-X-021', '1.0.0',
    'lineage_validation',
    'analysis',
    'Validate lineage graph integrity: orphan nodes, broken edges, cycles, source coverage, completeness, and freshness.',
    '{"scope", "validation_config"}',
    '{"validation_id", "orphan_nodes", "broken_edges", "cycles_detected", "source_coverage", "completeness_score", "freshness_score", "issues", "recommendations"}',
    '{"detects_orphans": true, "detects_broken_edges": true, "detects_cycles": true, "scores_coverage": true}'::jsonb
),
(
    'GL-DATA-X-021', '1.0.0',
    'compliance_reporting',
    'reporting',
    'Generate regulatory compliance reports (CSRD/ESRS, GHG Protocol, SOC 2) in multiple visualization formats.',
    '{"report_type", "format", "scope", "parameters"}',
    '{"report_id", "content", "report_hash", "generated_at"}',
    '{"report_types": ["csrd_esrs", "ghg_protocol", "soc2", "custom", "visualization"], "formats": ["mermaid", "dot", "json", "d3", "text", "html", "pdf"]}'::jsonb
),
(
    'GL-DATA-X-021', '1.0.0',
    'change_detection',
    'monitoring',
    'Detect and track lineage graph changes (node/edge added/removed, topology changes) with severity classification.',
    '{"previous_snapshot_id", "current_snapshot_id"}',
    '{"change_events", "change_summary", "severity_distribution"}',
    '{"change_types": ["node_added", "node_removed", "edge_added", "edge_removed", "topology_changed"], "severities": ["low", "medium", "high", "critical"]}'::jsonb
),
(
    'GL-DATA-X-021', '1.0.0',
    'quality_scoring',
    'analysis',
    'Score per-asset data quality based on source credibility, transformation depth, freshness, documentation, and manual interventions.',
    '{"asset_id", "scoring_config"}',
    '{"score_id", "source_credibility", "transformation_depth", "freshness_score", "documentation_score", "manual_intervention_count", "overall_score", "scoring_details"}',
    '{"score_range": [0.0, 1.0], "factors": ["source_credibility", "transformation_depth", "freshness", "documentation", "manual_intervention"]}'::jsonb
)
ON CONFLICT DO NOTHING;

INSERT INTO agent_registry_service.agent_dependencies (
    agent_id, depends_on_agent_id, version_constraint, optional, reason
) VALUES
    ('GL-DATA-X-021', 'GL-FOUND-X-001', '>=1.0.0', false, 'DAG orchestration for lineage graph traversal and pipeline execution ordering'),
    ('GL-DATA-X-021', 'GL-FOUND-X-007', '>=1.0.0', false, 'Agent version and capability lookup for transformation tracking'),
    ('GL-DATA-X-021', 'GL-FOUND-X-006', '>=1.0.0', false, 'Access control enforcement for asset classification and lineage report access'),
    ('GL-DATA-X-021', 'GL-FOUND-X-010', '>=1.0.0', false, 'Observability metrics for lineage graph health, impact events, and validation scores'),
    ('GL-DATA-X-021', 'GL-FOUND-X-005', '>=1.0.0', true,  'Provenance and audit trail registration with citation service'),
    ('GL-DATA-X-021', 'GL-FOUND-X-008', '>=1.0.0', true,  'Reproducibility verification for deterministic graph snapshot hashing'),
    ('GL-DATA-X-021', 'GL-FOUND-X-009', '>=1.0.0', true,  'QA Test Harness zero-hallucination verification of lineage outputs'),
    ('GL-DATA-X-021', 'GL-DATA-X-013', '>=1.0.0',  true,  'Data quality profiling to feed per-asset quality scoring'),
    ('GL-DATA-X-021', 'GL-DATA-X-020', '>=1.0.0',  true,  'Schema migration agent for schema_ref resolution and drift-aware lineage updates')
ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

INSERT INTO agent_registry_service.service_catalog (
    agent_id, display_name, summary, category, status, tenant_id
) VALUES (
    'GL-DATA-X-021',
    'Data Lineage Tracker',
    'Data lineage tracking engine. Asset registry (7 asset types, 4 classifications, 3 statuses). Transformation tracking (12 types, record counts, duration). Edge management (dataset-level/column-level, confidence scoring). Graph snapshots (topology metrics, orphan detection, coverage scoring, SHA-256 hashing). Impact analysis (forward/backward, blast radius, severity buckets). Lineage validation (orphan/broken-edge/cycle detection, coverage/completeness/freshness scoring). Compliance reporting (CSRD/ESRS, GHG Protocol, SOC 2, 7 output formats). Change detection (5 change types, 4 severity levels). Quality scoring (5 factors, overall score). SHA-256 provenance chains.',
    'data', 'active', 'default'
) ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA data_lineage_service IS
    'Data Lineage Tracker (AGENT-DATA-018) - asset registry, transformation tracking, edge management, graph snapshots, impact analysis, validation, compliance reporting, change detection, quality scoring, provenance chains';

COMMENT ON TABLE data_lineage_service.lineage_assets IS
    'Registered lineage assets: qualified_name (unique), asset_type (7 types), display_name, owner, tags, classification (4 levels), status (3 states), schema_ref, description, metadata';

COMMENT ON TABLE data_lineage_service.lineage_transformations IS
    'Transformation operations: transformation_type (12 types), agent/pipeline/execution refs, description, parameters, records_in/out/filtered/error counts, duration_ms, started/completed timestamps';

COMMENT ON TABLE data_lineage_service.lineage_edges IS
    'Lineage edges: source/target asset refs, transformation ref, edge_type (dataset_level/column_level), source/target field paths, transformation_logic, confidence (0-1)';

COMMENT ON TABLE data_lineage_service.lineage_graph_snapshots IS
    'Point-in-time graph snapshots: snapshot_name, node/edge/depth/component/orphan counts, coverage_score (0-1), SHA-256 graph_hash, snapshot_data JSON';

COMMENT ON TABLE data_lineage_service.lineage_impact_analyses IS
    'Impact analysis results: root_asset ref, direction (forward/backward), depth, affected_assets_count, critical/high/medium/low severity counts, blast_radius, analysis_result JSON';

COMMENT ON TABLE data_lineage_service.lineage_validations IS
    'Lineage validation results: scope, orphan_nodes, broken_edges, cycles_detected, source_coverage (0-1), completeness_score (0-1), freshness_score (0-1), issues/recommendations JSON';

COMMENT ON TABLE data_lineage_service.lineage_reports IS
    'Compliance and visualization reports: report_type (5 types), format (7 formats), scope, parameters JSON, content, SHA-256 report_hash, generated_by';

COMMENT ON TABLE data_lineage_service.lineage_change_events IS
    'Lineage graph change events: previous/current snapshot refs, change_type (5 types), entity_id/type, details JSON, severity (4 levels)';

COMMENT ON TABLE data_lineage_service.lineage_quality_scores IS
    'Per-asset quality scores: asset ref, source_credibility (0-1), transformation_depth, freshness_score (0-1), documentation_score (0-1), manual_intervention_count, overall_score (0-1), scoring_details JSON';

COMMENT ON TABLE data_lineage_service.lineage_audit_log IS
    'Full audit trail: action (28 types), entity_type (10 types), entity_id, actor, details/previous/new state JSON, SHA-256 provenance and parent hashes';

COMMENT ON TABLE data_lineage_service.lineage_transformation_events IS
    'TimescaleDB hypertable: transformation events with agent_id, pipeline_id, transformation_type, records_in/out, duration_ms (7-day chunks, 90-day retention)';

COMMENT ON TABLE data_lineage_service.lineage_validation_events IS
    'TimescaleDB hypertable: validation events with scope, orphan_count, broken_count, coverage_score, completeness_score (7-day chunks, 90-day retention)';

COMMENT ON TABLE data_lineage_service.lineage_impact_events IS
    'TimescaleDB hypertable: impact analysis events with root_asset_id, direction, affected_count, critical_count, blast_radius (7-day chunks, 90-day retention)';

COMMENT ON MATERIALIZED VIEW data_lineage_service.lineage_transformations_hourly_stats IS
    'Continuous aggregate: hourly transformation stats by type/agent/pipeline (total events, total records in/out, avg duration per hour)';

COMMENT ON MATERIALIZED VIEW data_lineage_service.lineage_validations_hourly_stats IS
    'Continuous aggregate: hourly validation stats by scope (avg coverage/completeness scores, total orphan/broken counts per hour)';
