-- =============================================================================
-- V041: Duplicate Detector Service Schema
-- =============================================================================
-- Component: AGENT-DATA-011 (Duplicate Detection Agent)
-- Agent ID:  GL-DATA-X-014
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Duplicate Detection Agent (GL-DATA-X-014) with capabilities
-- for record fingerprinting (SHA-256, SimHash, MinHash), blocking
-- strategies (sorted neighborhood, standard, canopy), pairwise
-- field-level comparison with configurable algorithms, match
-- classification (match/non_match/possible) with confidence scoring,
-- transitive closure clustering with quality/density/diameter metrics,
-- merge decision strategies (keep_first, keep_latest, keep_most_complete,
-- merge_fields, golden_record, custom) with field-level conflict
-- resolution, configurable dedup rule sets with match/possible
-- thresholds, and comprehensive audit logging with provenance chains.
-- =============================================================================
-- Tables (10):
--   1. dedup_rules               - Configurable rule sets
--   2. dedup_jobs                - Job tracking
--   3. dedup_fingerprints        - Record fingerprints
--   4. dedup_blocks              - Blocking results
--   5. dedup_comparisons         - Pairwise comparisons
--   6. dedup_matches             - Classified matches
--   7. dedup_clusters            - Duplicate clusters
--   8. dedup_merge_decisions     - Merge decisions
--   9. dedup_merge_conflicts     - Field conflicts
--  10. dedup_audit_log           - Audit trail
--
-- Hypertables (3):
--  11. dedup_events              - Dedup event time-series (hypertable)
--  12. comparison_events         - Comparison event time-series (hypertable)
--  13. merge_events              - Merge event time-series (hypertable)
--
-- Continuous Aggregates (2):
--   1. dedup_hourly_stats        - Hourly dedup event stats
--   2. comparison_hourly_stats   - Hourly comparison event stats
--
-- Also includes: 150+ indexes (B-tree, GIN, partial, composite),
-- 75+ CHECK constraints, 26 RLS policies per tenant, retention
-- policies (90 days on hypertables), compression policies (7 days),
-- updated_at trigger, security permissions for
-- greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-DATA-X-014.
-- Previous: V040__data_quality_profiler_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS duplicate_detector_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================
-- Reusable trigger function for tables with updated_at columns.

CREATE OR REPLACE FUNCTION duplicate_detector_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: duplicate_detector_service.dedup_rules
-- =============================================================================
-- Configurable dedup rule sets. Each rule set captures field comparison
-- configurations, match and possible thresholds, blocking strategy and
-- fields, merge strategy, activation state, version, and tenant scope.
-- Rules must be created before jobs that reference them.

CREATE TABLE duplicate_detector_service.dedup_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    field_configs JSONB NOT NULL,
    match_threshold NUMERIC(5,4) NOT NULL DEFAULT 0.8500,
    possible_threshold NUMERIC(5,4) NOT NULL DEFAULT 0.6500,
    blocking_strategy VARCHAR(30) NOT NULL DEFAULT 'sorted_neighborhood',
    blocking_fields TEXT[],
    merge_strategy VARCHAR(30) NOT NULL DEFAULT 'keep_most_complete',
    is_active BOOLEAN NOT NULL DEFAULT true,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100) NOT NULL DEFAULT 'system',
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Blocking strategy constraint
ALTER TABLE duplicate_detector_service.dedup_rules
    ADD CONSTRAINT chk_dr_blocking_strategy
    CHECK (blocking_strategy IN ('sorted_neighborhood', 'standard', 'canopy', 'none'));

-- Merge strategy constraint
ALTER TABLE duplicate_detector_service.dedup_rules
    ADD CONSTRAINT chk_dr_merge_strategy
    CHECK (merge_strategy IN (
        'keep_first', 'keep_latest', 'keep_most_complete',
        'merge_fields', 'golden_record', 'custom'
    ));

-- Match threshold must be between 0 and 1
ALTER TABLE duplicate_detector_service.dedup_rules
    ADD CONSTRAINT chk_dr_match_threshold_range
    CHECK (match_threshold >= 0 AND match_threshold <= 1);

-- Possible threshold must be between 0 and 1
ALTER TABLE duplicate_detector_service.dedup_rules
    ADD CONSTRAINT chk_dr_possible_threshold_range
    CHECK (possible_threshold >= 0 AND possible_threshold <= 1);

-- Match threshold must be >= possible threshold
ALTER TABLE duplicate_detector_service.dedup_rules
    ADD CONSTRAINT chk_dr_threshold_ordering
    CHECK (match_threshold >= possible_threshold);

-- Version must be positive
ALTER TABLE duplicate_detector_service.dedup_rules
    ADD CONSTRAINT chk_dr_version_positive
    CHECK (version >= 1);

-- Name must not be empty
ALTER TABLE duplicate_detector_service.dedup_rules
    ADD CONSTRAINT chk_dr_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- Created by must not be empty
ALTER TABLE duplicate_detector_service.dedup_rules
    ADD CONSTRAINT chk_dr_created_by_not_empty
    CHECK (LENGTH(TRIM(created_by)) > 0);

-- Tenant ID must not be empty
ALTER TABLE duplicate_detector_service.dedup_rules
    ADD CONSTRAINT chk_dr_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- Field configs must be a non-empty object
ALTER TABLE duplicate_detector_service.dedup_rules
    ADD CONSTRAINT chk_dr_field_configs_not_empty
    CHECK (field_configs IS NOT NULL AND field_configs::text != '{}' AND field_configs::text != 'null');

-- updated_at trigger
CREATE TRIGGER trg_dr_updated_at
    BEFORE UPDATE ON duplicate_detector_service.dedup_rules
    FOR EACH ROW
    EXECUTE FUNCTION duplicate_detector_service.set_updated_at();

-- =============================================================================
-- Table 2: duplicate_detector_service.dedup_jobs
-- =============================================================================
-- Job tracking for deduplication runs. Each job captures dataset IDs,
-- optional rule reference, processing status and stage, record counts
-- at each pipeline stage (fingerprinted, compared, matched, clustered,
-- merged), duplicate rate, error messages, configuration, provenance
-- hash, and timing information. Tenant-scoped.

CREATE TABLE duplicate_detector_service.dedup_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_ids TEXT[] NOT NULL,
    rule_id UUID REFERENCES duplicate_detector_service.dedup_rules(id),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    stage VARCHAR(20) DEFAULT 'fingerprint',
    total_records INTEGER NOT NULL DEFAULT 0,
    fingerprinted INTEGER NOT NULL DEFAULT 0,
    compared INTEGER NOT NULL DEFAULT 0,
    matched INTEGER NOT NULL DEFAULT 0,
    clustered INTEGER NOT NULL DEFAULT 0,
    merged INTEGER NOT NULL DEFAULT 0,
    duplicate_rate NUMERIC(5,4) DEFAULT 0,
    error_message TEXT,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(64) NOT NULL,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100) NOT NULL DEFAULT 'system',
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Status constraint
ALTER TABLE duplicate_detector_service.dedup_jobs
    ADD CONSTRAINT chk_dj_status
    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'));

-- Stage constraint
ALTER TABLE duplicate_detector_service.dedup_jobs
    ADD CONSTRAINT chk_dj_stage
    CHECK (stage IN ('fingerprint', 'block', 'compare', 'classify', 'cluster', 'merge', 'complete'));

-- Total records must be non-negative
ALTER TABLE duplicate_detector_service.dedup_jobs
    ADD CONSTRAINT chk_dj_total_records_non_negative
    CHECK (total_records >= 0);

-- Fingerprinted must be non-negative
ALTER TABLE duplicate_detector_service.dedup_jobs
    ADD CONSTRAINT chk_dj_fingerprinted_non_negative
    CHECK (fingerprinted >= 0);

-- Compared must be non-negative
ALTER TABLE duplicate_detector_service.dedup_jobs
    ADD CONSTRAINT chk_dj_compared_non_negative
    CHECK (compared >= 0);

-- Matched must be non-negative
ALTER TABLE duplicate_detector_service.dedup_jobs
    ADD CONSTRAINT chk_dj_matched_non_negative
    CHECK (matched >= 0);

-- Clustered must be non-negative
ALTER TABLE duplicate_detector_service.dedup_jobs
    ADD CONSTRAINT chk_dj_clustered_non_negative
    CHECK (clustered >= 0);

-- Merged must be non-negative
ALTER TABLE duplicate_detector_service.dedup_jobs
    ADD CONSTRAINT chk_dj_merged_non_negative
    CHECK (merged >= 0);

-- Duplicate rate must be between 0 and 1
ALTER TABLE duplicate_detector_service.dedup_jobs
    ADD CONSTRAINT chk_dj_duplicate_rate_range
    CHECK (duplicate_rate IS NULL OR (duplicate_rate >= 0 AND duplicate_rate <= 1));

-- Provenance hash must not be empty
ALTER TABLE duplicate_detector_service.dedup_jobs
    ADD CONSTRAINT chk_dj_provenance_hash_not_empty
    CHECK (LENGTH(TRIM(provenance_hash)) > 0);

-- Created by must not be empty
ALTER TABLE duplicate_detector_service.dedup_jobs
    ADD CONSTRAINT chk_dj_created_by_not_empty
    CHECK (LENGTH(TRIM(created_by)) > 0);

-- Tenant ID must not be empty
ALTER TABLE duplicate_detector_service.dedup_jobs
    ADD CONSTRAINT chk_dj_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- Dataset IDs array must not be empty
ALTER TABLE duplicate_detector_service.dedup_jobs
    ADD CONSTRAINT chk_dj_dataset_ids_not_empty
    CHECK (array_length(dataset_ids, 1) > 0);

-- Completed_at must be after started_at if both are set
ALTER TABLE duplicate_detector_service.dedup_jobs
    ADD CONSTRAINT chk_dj_completed_after_started
    CHECK (completed_at IS NULL OR started_at IS NULL OR completed_at >= started_at);

-- updated_at trigger
CREATE TRIGGER trg_dj_updated_at
    BEFORE UPDATE ON duplicate_detector_service.dedup_jobs
    FOR EACH ROW
    EXECUTE FUNCTION duplicate_detector_service.set_updated_at();

-- =============================================================================
-- Table 3: duplicate_detector_service.dedup_fingerprints
-- =============================================================================
-- Record fingerprints generated during the fingerprint stage. Each
-- fingerprint captures the record and dataset identifiers, the set
-- of fields used, the fingerprint hash value, the algorithm used
-- (sha256, simhash, minhash), and normalized field values for
-- downstream comparison. Linked to dedup_jobs. Tenant-scoped.

CREATE TABLE duplicate_detector_service.dedup_fingerprints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL,
    record_id VARCHAR(255) NOT NULL,
    dataset_id VARCHAR(255) NOT NULL,
    field_set TEXT[] NOT NULL,
    fingerprint_hash VARCHAR(128) NOT NULL,
    algorithm VARCHAR(20) NOT NULL,
    normalized_fields JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to dedup_jobs
ALTER TABLE duplicate_detector_service.dedup_fingerprints
    ADD CONSTRAINT fk_df_job_id
    FOREIGN KEY (job_id) REFERENCES duplicate_detector_service.dedup_jobs(id)
    ON DELETE CASCADE;

-- Algorithm constraint
ALTER TABLE duplicate_detector_service.dedup_fingerprints
    ADD CONSTRAINT chk_df_algorithm
    CHECK (algorithm IN ('sha256', 'simhash', 'minhash'));

-- Record ID must not be empty
ALTER TABLE duplicate_detector_service.dedup_fingerprints
    ADD CONSTRAINT chk_df_record_id_not_empty
    CHECK (LENGTH(TRIM(record_id)) > 0);

-- Dataset ID must not be empty
ALTER TABLE duplicate_detector_service.dedup_fingerprints
    ADD CONSTRAINT chk_df_dataset_id_not_empty
    CHECK (LENGTH(TRIM(dataset_id)) > 0);

-- Fingerprint hash must not be empty
ALTER TABLE duplicate_detector_service.dedup_fingerprints
    ADD CONSTRAINT chk_df_fingerprint_hash_not_empty
    CHECK (LENGTH(TRIM(fingerprint_hash)) > 0);

-- Field set must not be empty
ALTER TABLE duplicate_detector_service.dedup_fingerprints
    ADD CONSTRAINT chk_df_field_set_not_empty
    CHECK (array_length(field_set, 1) > 0);

-- Tenant ID must not be empty
ALTER TABLE duplicate_detector_service.dedup_fingerprints
    ADD CONSTRAINT chk_df_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 4: duplicate_detector_service.dedup_blocks
-- =============================================================================
-- Blocking results from the blocking stage. Each block captures
-- the block key, blocking strategy, member record IDs, and record
-- count. Blocking reduces the comparison space by grouping records
-- into blocks that share common attributes. Linked to dedup_jobs.
-- Tenant-scoped.

CREATE TABLE duplicate_detector_service.dedup_blocks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL,
    block_key VARCHAR(255) NOT NULL,
    strategy VARCHAR(30) NOT NULL,
    record_ids TEXT[] NOT NULL,
    record_count INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to dedup_jobs
ALTER TABLE duplicate_detector_service.dedup_blocks
    ADD CONSTRAINT fk_db_job_id
    FOREIGN KEY (job_id) REFERENCES duplicate_detector_service.dedup_jobs(id)
    ON DELETE CASCADE;

-- Strategy constraint
ALTER TABLE duplicate_detector_service.dedup_blocks
    ADD CONSTRAINT chk_db_strategy
    CHECK (strategy IN ('sorted_neighborhood', 'standard', 'canopy', 'none'));

-- Record count must be positive
ALTER TABLE duplicate_detector_service.dedup_blocks
    ADD CONSTRAINT chk_db_record_count_positive
    CHECK (record_count > 0);

-- Block key must not be empty
ALTER TABLE duplicate_detector_service.dedup_blocks
    ADD CONSTRAINT chk_db_block_key_not_empty
    CHECK (LENGTH(TRIM(block_key)) > 0);

-- Record IDs must not be empty
ALTER TABLE duplicate_detector_service.dedup_blocks
    ADD CONSTRAINT chk_db_record_ids_not_empty
    CHECK (array_length(record_ids, 1) > 0);

-- Record count must match record IDs length
ALTER TABLE duplicate_detector_service.dedup_blocks
    ADD CONSTRAINT chk_db_record_count_matches
    CHECK (record_count = array_length(record_ids, 1));

-- Tenant ID must not be empty
ALTER TABLE duplicate_detector_service.dedup_blocks
    ADD CONSTRAINT chk_db_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 5: duplicate_detector_service.dedup_comparisons
-- =============================================================================
-- Pairwise comparison results from the compare stage. Each comparison
-- captures the two record identifiers, per-field similarity scores
-- (JSONB), overall similarity score (0-1), the algorithm used for
-- comparison, and comparison duration. Linked to dedup_jobs.
-- Tenant-scoped.

CREATE TABLE duplicate_detector_service.dedup_comparisons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL,
    record_a_id VARCHAR(255) NOT NULL,
    record_b_id VARCHAR(255) NOT NULL,
    field_scores JSONB NOT NULL,
    overall_score NUMERIC(5,4) NOT NULL,
    algorithm_used VARCHAR(30) NOT NULL,
    comparison_time_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to dedup_jobs
ALTER TABLE duplicate_detector_service.dedup_comparisons
    ADD CONSTRAINT fk_dc_job_id
    FOREIGN KEY (job_id) REFERENCES duplicate_detector_service.dedup_jobs(id)
    ON DELETE CASCADE;

-- Overall score must be between 0 and 1
ALTER TABLE duplicate_detector_service.dedup_comparisons
    ADD CONSTRAINT chk_dc_overall_score_range
    CHECK (overall_score >= 0 AND overall_score <= 1);

-- Record A ID must not be empty
ALTER TABLE duplicate_detector_service.dedup_comparisons
    ADD CONSTRAINT chk_dc_record_a_id_not_empty
    CHECK (LENGTH(TRIM(record_a_id)) > 0);

-- Record B ID must not be empty
ALTER TABLE duplicate_detector_service.dedup_comparisons
    ADD CONSTRAINT chk_dc_record_b_id_not_empty
    CHECK (LENGTH(TRIM(record_b_id)) > 0);

-- Algorithm used must not be empty
ALTER TABLE duplicate_detector_service.dedup_comparisons
    ADD CONSTRAINT chk_dc_algorithm_used_not_empty
    CHECK (LENGTH(TRIM(algorithm_used)) > 0);

-- Comparison time must be non-negative if specified
ALTER TABLE duplicate_detector_service.dedup_comparisons
    ADD CONSTRAINT chk_dc_comparison_time_non_negative
    CHECK (comparison_time_ms IS NULL OR comparison_time_ms >= 0);

-- Records A and B must be different
ALTER TABLE duplicate_detector_service.dedup_comparisons
    ADD CONSTRAINT chk_dc_different_records
    CHECK (record_a_id != record_b_id);

-- Tenant ID must not be empty
ALTER TABLE duplicate_detector_service.dedup_comparisons
    ADD CONSTRAINT chk_dc_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 6: duplicate_detector_service.dedup_matches
-- =============================================================================
-- Classified matches from the classify stage. Each match captures
-- the two record identifiers, classification (match, non_match,
-- possible), confidence score (0-1), per-field similarity scores,
-- overall score, and decision reasoning. Linked to dedup_jobs.
-- Tenant-scoped.

CREATE TABLE duplicate_detector_service.dedup_matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL,
    record_a_id VARCHAR(255) NOT NULL,
    record_b_id VARCHAR(255) NOT NULL,
    classification VARCHAR(20) NOT NULL,
    confidence NUMERIC(5,4) NOT NULL,
    field_scores JSONB NOT NULL,
    overall_score NUMERIC(5,4) NOT NULL,
    decision_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to dedup_jobs
ALTER TABLE duplicate_detector_service.dedup_matches
    ADD CONSTRAINT fk_dm_job_id
    FOREIGN KEY (job_id) REFERENCES duplicate_detector_service.dedup_jobs(id)
    ON DELETE CASCADE;

-- Classification constraint
ALTER TABLE duplicate_detector_service.dedup_matches
    ADD CONSTRAINT chk_dm_classification
    CHECK (classification IN ('match', 'non_match', 'possible'));

-- Confidence must be between 0 and 1
ALTER TABLE duplicate_detector_service.dedup_matches
    ADD CONSTRAINT chk_dm_confidence_range
    CHECK (confidence >= 0 AND confidence <= 1);

-- Overall score must be between 0 and 1
ALTER TABLE duplicate_detector_service.dedup_matches
    ADD CONSTRAINT chk_dm_overall_score_range
    CHECK (overall_score >= 0 AND overall_score <= 1);

-- Record A ID must not be empty
ALTER TABLE duplicate_detector_service.dedup_matches
    ADD CONSTRAINT chk_dm_record_a_id_not_empty
    CHECK (LENGTH(TRIM(record_a_id)) > 0);

-- Record B ID must not be empty
ALTER TABLE duplicate_detector_service.dedup_matches
    ADD CONSTRAINT chk_dm_record_b_id_not_empty
    CHECK (LENGTH(TRIM(record_b_id)) > 0);

-- Records A and B must be different
ALTER TABLE duplicate_detector_service.dedup_matches
    ADD CONSTRAINT chk_dm_different_records
    CHECK (record_a_id != record_b_id);

-- Tenant ID must not be empty
ALTER TABLE duplicate_detector_service.dedup_matches
    ADD CONSTRAINT chk_dm_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 7: duplicate_detector_service.dedup_clusters
-- =============================================================================
-- Duplicate clusters from the cluster stage. Each cluster captures
-- member record IDs, the elected representative record, cluster
-- quality score (0-1), density, diameter, member count, and the
-- clustering algorithm used. Linked to dedup_jobs. Tenant-scoped.

CREATE TABLE duplicate_detector_service.dedup_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL,
    member_record_ids TEXT[] NOT NULL,
    representative_id VARCHAR(255),
    cluster_quality NUMERIC(5,4),
    density NUMERIC(5,4),
    diameter NUMERIC(5,4),
    member_count INTEGER NOT NULL,
    algorithm VARCHAR(30) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to dedup_jobs
ALTER TABLE duplicate_detector_service.dedup_clusters
    ADD CONSTRAINT fk_dcl_job_id
    FOREIGN KEY (job_id) REFERENCES duplicate_detector_service.dedup_jobs(id)
    ON DELETE CASCADE;

-- Cluster quality must be between 0 and 1 if specified
ALTER TABLE duplicate_detector_service.dedup_clusters
    ADD CONSTRAINT chk_dcl_cluster_quality_range
    CHECK (cluster_quality IS NULL OR (cluster_quality >= 0 AND cluster_quality <= 1));

-- Density must be between 0 and 1 if specified
ALTER TABLE duplicate_detector_service.dedup_clusters
    ADD CONSTRAINT chk_dcl_density_range
    CHECK (density IS NULL OR (density >= 0 AND density <= 1));

-- Diameter must be between 0 and 1 if specified
ALTER TABLE duplicate_detector_service.dedup_clusters
    ADD CONSTRAINT chk_dcl_diameter_range
    CHECK (diameter IS NULL OR (diameter >= 0 AND diameter <= 1));

-- Member count must be positive
ALTER TABLE duplicate_detector_service.dedup_clusters
    ADD CONSTRAINT chk_dcl_member_count_positive
    CHECK (member_count > 0);

-- Member record IDs must not be empty
ALTER TABLE duplicate_detector_service.dedup_clusters
    ADD CONSTRAINT chk_dcl_member_record_ids_not_empty
    CHECK (array_length(member_record_ids, 1) > 0);

-- Member count must match member record IDs length
ALTER TABLE duplicate_detector_service.dedup_clusters
    ADD CONSTRAINT chk_dcl_member_count_matches
    CHECK (member_count = array_length(member_record_ids, 1));

-- Algorithm must not be empty
ALTER TABLE duplicate_detector_service.dedup_clusters
    ADD CONSTRAINT chk_dcl_algorithm_not_empty
    CHECK (LENGTH(TRIM(algorithm)) > 0);

-- Tenant ID must not be empty
ALTER TABLE duplicate_detector_service.dedup_clusters
    ADD CONSTRAINT chk_dcl_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 8: duplicate_detector_service.dedup_merge_decisions
-- =============================================================================
-- Merge decisions from the merge stage. Each decision captures the
-- merge strategy, the resulting merged record (JSONB), source records
-- (JSONB), conflict count, provenance hash, and decision timestamp.
-- Linked to dedup_jobs and dedup_clusters. Tenant-scoped.

CREATE TABLE duplicate_detector_service.dedup_merge_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL,
    cluster_id UUID NOT NULL,
    strategy VARCHAR(30) NOT NULL,
    merged_record JSONB NOT NULL,
    source_records JSONB NOT NULL,
    conflict_count INTEGER NOT NULL DEFAULT 0,
    provenance_hash VARCHAR(64) NOT NULL,
    decided_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to dedup_jobs
ALTER TABLE duplicate_detector_service.dedup_merge_decisions
    ADD CONSTRAINT fk_dmd_job_id
    FOREIGN KEY (job_id) REFERENCES duplicate_detector_service.dedup_jobs(id)
    ON DELETE CASCADE;

-- Foreign key to dedup_clusters
ALTER TABLE duplicate_detector_service.dedup_merge_decisions
    ADD CONSTRAINT fk_dmd_cluster_id
    FOREIGN KEY (cluster_id) REFERENCES duplicate_detector_service.dedup_clusters(id)
    ON DELETE CASCADE;

-- Strategy constraint
ALTER TABLE duplicate_detector_service.dedup_merge_decisions
    ADD CONSTRAINT chk_dmd_strategy
    CHECK (strategy IN (
        'keep_first', 'keep_latest', 'keep_most_complete',
        'merge_fields', 'golden_record', 'custom'
    ));

-- Conflict count must be non-negative
ALTER TABLE duplicate_detector_service.dedup_merge_decisions
    ADD CONSTRAINT chk_dmd_conflict_count_non_negative
    CHECK (conflict_count >= 0);

-- Provenance hash must not be empty
ALTER TABLE duplicate_detector_service.dedup_merge_decisions
    ADD CONSTRAINT chk_dmd_provenance_hash_not_empty
    CHECK (LENGTH(TRIM(provenance_hash)) > 0);

-- Tenant ID must not be empty
ALTER TABLE duplicate_detector_service.dedup_merge_decisions
    ADD CONSTRAINT chk_dmd_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 9: duplicate_detector_service.dedup_merge_conflicts
-- =============================================================================
-- Field-level merge conflicts. Each conflict captures the conflicting
-- field name, the array of conflicting values from different source
-- records, the chosen resolved value, the resolution method, and the
-- source record from which the chosen value was taken. Linked to
-- dedup_merge_decisions. Tenant-scoped.

CREATE TABLE duplicate_detector_service.dedup_merge_conflicts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    merge_decision_id UUID NOT NULL,
    field_name VARCHAR(255) NOT NULL,
    conflicting_values JSONB NOT NULL,
    chosen_value TEXT,
    resolution_method VARCHAR(30) NOT NULL,
    source_record_id VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to dedup_merge_decisions
ALTER TABLE duplicate_detector_service.dedup_merge_conflicts
    ADD CONSTRAINT fk_dmc_merge_decision_id
    FOREIGN KEY (merge_decision_id) REFERENCES duplicate_detector_service.dedup_merge_decisions(id)
    ON DELETE CASCADE;

-- Resolution method constraint
ALTER TABLE duplicate_detector_service.dedup_merge_conflicts
    ADD CONSTRAINT chk_dmc_resolution_method
    CHECK (resolution_method IN (
        'most_recent', 'most_frequent', 'longest', 'shortest',
        'first', 'last', 'manual', 'rule_based', 'custom',
        'highest', 'lowest', 'concatenate', 'none'
    ));

-- Field name must not be empty
ALTER TABLE duplicate_detector_service.dedup_merge_conflicts
    ADD CONSTRAINT chk_dmc_field_name_not_empty
    CHECK (LENGTH(TRIM(field_name)) > 0);

-- Tenant ID must not be empty
ALTER TABLE duplicate_detector_service.dedup_merge_conflicts
    ADD CONSTRAINT chk_dmc_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 10: duplicate_detector_service.dedup_audit_log
-- =============================================================================
-- Comprehensive audit trail for all dedup operations. Each entry
-- captures the action performed, entity type and ID, detail payload
-- (JSONB), provenance hash, performer, timestamp, and tenant scope.
-- Linked to dedup_jobs for job-scoped audit queries.

CREATE TABLE duplicate_detector_service.dedup_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID,
    action VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(64),
    performed_by VARCHAR(100) NOT NULL DEFAULT 'system',
    performed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default'
);

-- Foreign key to dedup_jobs (optional, job_id may be null for system-level actions)
ALTER TABLE duplicate_detector_service.dedup_audit_log
    ADD CONSTRAINT fk_dal_job_id
    FOREIGN KEY (job_id) REFERENCES duplicate_detector_service.dedup_jobs(id)
    ON DELETE SET NULL;

-- Action constraint
ALTER TABLE duplicate_detector_service.dedup_audit_log
    ADD CONSTRAINT chk_dal_action
    CHECK (action IN (
        'job_created', 'job_started', 'job_completed', 'job_failed', 'job_cancelled',
        'fingerprint_generated', 'block_created', 'comparison_performed',
        'match_classified', 'cluster_formed', 'merge_decided', 'conflict_resolved',
        'rule_created', 'rule_updated', 'rule_deleted', 'rule_activated', 'rule_deactivated',
        'config_changed', 'threshold_updated', 'manual_review_requested',
        'manual_review_completed', 'export_generated', 'import_completed'
    ));

-- Entity type constraint
ALTER TABLE duplicate_detector_service.dedup_audit_log
    ADD CONSTRAINT chk_dal_entity_type
    CHECK (entity_type IN (
        'job', 'fingerprint', 'block', 'comparison', 'match',
        'cluster', 'merge_decision', 'merge_conflict', 'rule', 'config'
    ));

-- Action must not be empty
ALTER TABLE duplicate_detector_service.dedup_audit_log
    ADD CONSTRAINT chk_dal_action_not_empty
    CHECK (LENGTH(TRIM(action)) > 0);

-- Entity type must not be empty
ALTER TABLE duplicate_detector_service.dedup_audit_log
    ADD CONSTRAINT chk_dal_entity_type_not_empty
    CHECK (LENGTH(TRIM(entity_type)) > 0);

-- Performed by must not be empty
ALTER TABLE duplicate_detector_service.dedup_audit_log
    ADD CONSTRAINT chk_dal_performed_by_not_empty
    CHECK (LENGTH(TRIM(performed_by)) > 0);

-- Tenant ID must not be empty
ALTER TABLE duplicate_detector_service.dedup_audit_log
    ADD CONSTRAINT chk_dal_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 11: duplicate_detector_service.dedup_events (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording deduplication lifecycle events as a
-- time-series. Each event captures the job ID, event type, pipeline
-- stage, record count, duration in milliseconds, details payload,
-- provenance hash, and tenant. Partitioned by event_time for
-- time-series queries. Retained for 90 days with compression
-- after 7 days.

CREATE TABLE duplicate_detector_service.dedup_events (
    event_id UUID DEFAULT gen_random_uuid(),
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    job_id UUID,
    event_type VARCHAR(50) NOT NULL,
    stage VARCHAR(20),
    record_count INTEGER,
    duration_ms INTEGER,
    details JSONB DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    PRIMARY KEY (event_id, event_time)
);

-- Create hypertable partitioned by event_time with 7-day chunks
SELECT create_hypertable('duplicate_detector_service.dedup_events', 'event_time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Event type constraint
ALTER TABLE duplicate_detector_service.dedup_events
    ADD CONSTRAINT chk_de_event_type
    CHECK (event_type IN (
        'job_started', 'job_completed', 'job_failed', 'job_cancelled',
        'fingerprint_started', 'fingerprint_completed', 'fingerprint_failed',
        'blocking_started', 'blocking_completed', 'blocking_failed',
        'comparison_started', 'comparison_completed', 'comparison_failed',
        'classification_started', 'classification_completed', 'classification_failed',
        'clustering_started', 'clustering_completed', 'clustering_failed',
        'merging_started', 'merging_completed', 'merging_failed',
        'stage_transition', 'progress_update', 'threshold_breach'
    ));

-- Stage constraint if specified
ALTER TABLE duplicate_detector_service.dedup_events
    ADD CONSTRAINT chk_de_stage
    CHECK (stage IS NULL OR stage IN (
        'fingerprint', 'block', 'compare', 'classify', 'cluster', 'merge', 'complete'
    ));

-- Record count must be non-negative if specified
ALTER TABLE duplicate_detector_service.dedup_events
    ADD CONSTRAINT chk_de_record_count_non_negative
    CHECK (record_count IS NULL OR record_count >= 0);

-- Duration must be non-negative if specified
ALTER TABLE duplicate_detector_service.dedup_events
    ADD CONSTRAINT chk_de_duration_non_negative
    CHECK (duration_ms IS NULL OR duration_ms >= 0);

-- Tenant ID must not be empty
ALTER TABLE duplicate_detector_service.dedup_events
    ADD CONSTRAINT chk_de_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 12: duplicate_detector_service.comparison_events (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording pairwise comparison events as a
-- time-series. Each event captures the job ID, record pair, algorithm
-- used, similarity score, classification result, and tenant.
-- Partitioned by event_time for time-series queries. Retained for
-- 90 days with compression after 7 days.

CREATE TABLE duplicate_detector_service.comparison_events (
    event_id UUID DEFAULT gen_random_uuid(),
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    job_id UUID,
    record_a_id VARCHAR(255),
    record_b_id VARCHAR(255),
    algorithm VARCHAR(30),
    score NUMERIC(5,4),
    classification VARCHAR(20),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    PRIMARY KEY (event_id, event_time)
);

-- Create hypertable partitioned by event_time with 7-day chunks
SELECT create_hypertable('duplicate_detector_service.comparison_events', 'event_time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Score must be between 0 and 1 if specified
ALTER TABLE duplicate_detector_service.comparison_events
    ADD CONSTRAINT chk_ce_score_range
    CHECK (score IS NULL OR (score >= 0 AND score <= 1));

-- Classification constraint if specified
ALTER TABLE duplicate_detector_service.comparison_events
    ADD CONSTRAINT chk_ce_classification
    CHECK (classification IS NULL OR classification IN ('match', 'non_match', 'possible'));

-- Tenant ID must not be empty
ALTER TABLE duplicate_detector_service.comparison_events
    ADD CONSTRAINT chk_ce_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 13: duplicate_detector_service.merge_events (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording merge decision events as a
-- time-series. Each event captures the job ID, cluster ID, merge
-- strategy, number of records merged, conflict count, provenance
-- hash, and tenant. Partitioned by event_time for time-series
-- queries. Retained for 90 days with compression after 7 days.

CREATE TABLE duplicate_detector_service.merge_events (
    event_id UUID DEFAULT gen_random_uuid(),
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    job_id UUID,
    cluster_id UUID,
    strategy VARCHAR(30),
    records_merged INTEGER,
    conflicts INTEGER,
    provenance_hash VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default',
    PRIMARY KEY (event_id, event_time)
);

-- Create hypertable partitioned by event_time with 7-day chunks
SELECT create_hypertable('duplicate_detector_service.merge_events', 'event_time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Strategy constraint if specified
ALTER TABLE duplicate_detector_service.merge_events
    ADD CONSTRAINT chk_me_strategy
    CHECK (strategy IS NULL OR strategy IN (
        'keep_first', 'keep_latest', 'keep_most_complete',
        'merge_fields', 'golden_record', 'custom'
    ));

-- Records merged must be non-negative if specified
ALTER TABLE duplicate_detector_service.merge_events
    ADD CONSTRAINT chk_me_records_merged_non_negative
    CHECK (records_merged IS NULL OR records_merged >= 0);

-- Conflicts must be non-negative if specified
ALTER TABLE duplicate_detector_service.merge_events
    ADD CONSTRAINT chk_me_conflicts_non_negative
    CHECK (conflicts IS NULL OR conflicts >= 0);

-- Tenant ID must not be empty
ALTER TABLE duplicate_detector_service.merge_events
    ADD CONSTRAINT chk_me_tenant_id_not_empty
    CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Continuous Aggregate: duplicate_detector_service.dedup_hourly_stats
-- =============================================================================
-- Precomputed hourly deduplication statistics by event type for
-- dashboard queries, job monitoring, and throughput analysis.

CREATE MATERIALIZED VIEW duplicate_detector_service.dedup_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', event_time) AS bucket,
    event_type,
    COUNT(*) AS total_events,
    COUNT(DISTINCT job_id) AS unique_jobs,
    AVG(record_count) AS avg_record_count,
    SUM(record_count) AS total_records_processed,
    AVG(duration_ms) AS avg_duration_ms,
    MAX(duration_ms) AS max_duration_ms,
    MIN(duration_ms) AS min_duration_ms,
    COUNT(*) FILTER (WHERE stage = 'fingerprint') AS fingerprint_events,
    COUNT(*) FILTER (WHERE stage = 'block') AS block_events,
    COUNT(*) FILTER (WHERE stage = 'compare') AS compare_events,
    COUNT(*) FILTER (WHERE stage = 'classify') AS classify_events,
    COUNT(*) FILTER (WHERE stage = 'cluster') AS cluster_events,
    COUNT(*) FILTER (WHERE stage = 'merge') AS merge_events
FROM duplicate_detector_service.dedup_events
WHERE event_time IS NOT NULL
GROUP BY bucket, event_type
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('duplicate_detector_service.dedup_hourly_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Continuous Aggregate: duplicate_detector_service.comparison_hourly_stats
-- =============================================================================
-- Precomputed hourly comparison statistics by classification for
-- dashboard queries, match rate monitoring, and algorithm performance.

CREATE MATERIALIZED VIEW duplicate_detector_service.comparison_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', event_time) AS bucket,
    algorithm,
    COUNT(*) AS total_comparisons,
    COUNT(DISTINCT job_id) AS unique_jobs,
    AVG(score) AS avg_score,
    MIN(score) AS min_score,
    MAX(score) AS max_score,
    COUNT(*) FILTER (WHERE classification = 'match') AS match_count,
    COUNT(*) FILTER (WHERE classification = 'non_match') AS non_match_count,
    COUNT(*) FILTER (WHERE classification = 'possible') AS possible_count
FROM duplicate_detector_service.comparison_events
WHERE event_time IS NOT NULL
GROUP BY bucket, algorithm
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('duplicate_detector_service.comparison_hourly_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- dedup_rules indexes (16)
CREATE INDEX idx_dr_name ON duplicate_detector_service.dedup_rules(name);
CREATE INDEX idx_dr_blocking_strategy ON duplicate_detector_service.dedup_rules(blocking_strategy);
CREATE INDEX idx_dr_merge_strategy ON duplicate_detector_service.dedup_rules(merge_strategy);
CREATE INDEX idx_dr_is_active ON duplicate_detector_service.dedup_rules(is_active);
CREATE INDEX idx_dr_version ON duplicate_detector_service.dedup_rules(version);
CREATE INDEX idx_dr_tenant_id ON duplicate_detector_service.dedup_rules(tenant_id);
CREATE INDEX idx_dr_created_by ON duplicate_detector_service.dedup_rules(created_by);
CREATE INDEX idx_dr_created_at ON duplicate_detector_service.dedup_rules(created_at DESC);
CREATE INDEX idx_dr_updated_at ON duplicate_detector_service.dedup_rules(updated_at DESC);
CREATE INDEX idx_dr_match_threshold ON duplicate_detector_service.dedup_rules(match_threshold);
CREATE INDEX idx_dr_possible_threshold ON duplicate_detector_service.dedup_rules(possible_threshold);
CREATE INDEX idx_dr_tenant_active ON duplicate_detector_service.dedup_rules(tenant_id, is_active);
CREATE INDEX idx_dr_tenant_strategy ON duplicate_detector_service.dedup_rules(tenant_id, blocking_strategy);
CREATE INDEX idx_dr_tenant_created ON duplicate_detector_service.dedup_rules(tenant_id, created_at DESC);
CREATE INDEX idx_dr_field_configs ON duplicate_detector_service.dedup_rules USING GIN (field_configs);
CREATE INDEX idx_dr_blocking_fields ON duplicate_detector_service.dedup_rules USING GIN (blocking_fields);

-- dedup_jobs indexes (18)
CREATE INDEX idx_dj_rule_id ON duplicate_detector_service.dedup_jobs(rule_id);
CREATE INDEX idx_dj_status ON duplicate_detector_service.dedup_jobs(status);
CREATE INDEX idx_dj_stage ON duplicate_detector_service.dedup_jobs(stage);
CREATE INDEX idx_dj_tenant_id ON duplicate_detector_service.dedup_jobs(tenant_id);
CREATE INDEX idx_dj_created_by ON duplicate_detector_service.dedup_jobs(created_by);
CREATE INDEX idx_dj_provenance ON duplicate_detector_service.dedup_jobs(provenance_hash);
CREATE INDEX idx_dj_created_at ON duplicate_detector_service.dedup_jobs(created_at DESC);
CREATE INDEX idx_dj_updated_at ON duplicate_detector_service.dedup_jobs(updated_at DESC);
CREATE INDEX idx_dj_started_at ON duplicate_detector_service.dedup_jobs(started_at DESC);
CREATE INDEX idx_dj_completed_at ON duplicate_detector_service.dedup_jobs(completed_at DESC);
CREATE INDEX idx_dj_duplicate_rate ON duplicate_detector_service.dedup_jobs(duplicate_rate DESC);
CREATE INDEX idx_dj_total_records ON duplicate_detector_service.dedup_jobs(total_records DESC);
CREATE INDEX idx_dj_tenant_status ON duplicate_detector_service.dedup_jobs(tenant_id, status);
CREATE INDEX idx_dj_tenant_stage ON duplicate_detector_service.dedup_jobs(tenant_id, stage);
CREATE INDEX idx_dj_tenant_created ON duplicate_detector_service.dedup_jobs(tenant_id, created_at DESC);
CREATE INDEX idx_dj_status_stage ON duplicate_detector_service.dedup_jobs(status, stage);
CREATE INDEX idx_dj_dataset_ids ON duplicate_detector_service.dedup_jobs USING GIN (dataset_ids);
CREATE INDEX idx_dj_config ON duplicate_detector_service.dedup_jobs USING GIN (config);

-- dedup_fingerprints indexes (16)
CREATE INDEX idx_df_job_id ON duplicate_detector_service.dedup_fingerprints(job_id);
CREATE INDEX idx_df_record_id ON duplicate_detector_service.dedup_fingerprints(record_id);
CREATE INDEX idx_df_dataset_id ON duplicate_detector_service.dedup_fingerprints(dataset_id);
CREATE INDEX idx_df_fingerprint_hash ON duplicate_detector_service.dedup_fingerprints(fingerprint_hash);
CREATE INDEX idx_df_algorithm ON duplicate_detector_service.dedup_fingerprints(algorithm);
CREATE INDEX idx_df_tenant_id ON duplicate_detector_service.dedup_fingerprints(tenant_id);
CREATE INDEX idx_df_created_at ON duplicate_detector_service.dedup_fingerprints(created_at DESC);
CREATE INDEX idx_df_job_record ON duplicate_detector_service.dedup_fingerprints(job_id, record_id);
CREATE INDEX idx_df_job_dataset ON duplicate_detector_service.dedup_fingerprints(job_id, dataset_id);
CREATE INDEX idx_df_job_algorithm ON duplicate_detector_service.dedup_fingerprints(job_id, algorithm);
CREATE INDEX idx_df_record_dataset ON duplicate_detector_service.dedup_fingerprints(record_id, dataset_id);
CREATE INDEX idx_df_hash_algorithm ON duplicate_detector_service.dedup_fingerprints(fingerprint_hash, algorithm);
CREATE INDEX idx_df_tenant_job ON duplicate_detector_service.dedup_fingerprints(tenant_id, job_id);
CREATE INDEX idx_df_tenant_record ON duplicate_detector_service.dedup_fingerprints(tenant_id, record_id);
CREATE INDEX idx_df_field_set ON duplicate_detector_service.dedup_fingerprints USING GIN (field_set);
CREATE INDEX idx_df_normalized_fields ON duplicate_detector_service.dedup_fingerprints USING GIN (normalized_fields);

-- dedup_blocks indexes (14)
CREATE INDEX idx_db_job_id ON duplicate_detector_service.dedup_blocks(job_id);
CREATE INDEX idx_db_block_key ON duplicate_detector_service.dedup_blocks(block_key);
CREATE INDEX idx_db_strategy ON duplicate_detector_service.dedup_blocks(strategy);
CREATE INDEX idx_db_record_count ON duplicate_detector_service.dedup_blocks(record_count DESC);
CREATE INDEX idx_db_tenant_id ON duplicate_detector_service.dedup_blocks(tenant_id);
CREATE INDEX idx_db_created_at ON duplicate_detector_service.dedup_blocks(created_at DESC);
CREATE INDEX idx_db_job_block_key ON duplicate_detector_service.dedup_blocks(job_id, block_key);
CREATE INDEX idx_db_job_strategy ON duplicate_detector_service.dedup_blocks(job_id, strategy);
CREATE INDEX idx_db_job_count ON duplicate_detector_service.dedup_blocks(job_id, record_count DESC);
CREATE INDEX idx_db_tenant_job ON duplicate_detector_service.dedup_blocks(tenant_id, job_id);
CREATE INDEX idx_db_tenant_strategy ON duplicate_detector_service.dedup_blocks(tenant_id, strategy);
CREATE INDEX idx_db_strategy_count ON duplicate_detector_service.dedup_blocks(strategy, record_count DESC);
CREATE UNIQUE INDEX idx_db_job_block_unique ON duplicate_detector_service.dedup_blocks(job_id, block_key);
CREATE INDEX idx_db_record_ids ON duplicate_detector_service.dedup_blocks USING GIN (record_ids);

-- dedup_comparisons indexes (16)
CREATE INDEX idx_dc_job_id ON duplicate_detector_service.dedup_comparisons(job_id);
CREATE INDEX idx_dc_record_a_id ON duplicate_detector_service.dedup_comparisons(record_a_id);
CREATE INDEX idx_dc_record_b_id ON duplicate_detector_service.dedup_comparisons(record_b_id);
CREATE INDEX idx_dc_overall_score ON duplicate_detector_service.dedup_comparisons(overall_score DESC);
CREATE INDEX idx_dc_algorithm_used ON duplicate_detector_service.dedup_comparisons(algorithm_used);
CREATE INDEX idx_dc_comparison_time ON duplicate_detector_service.dedup_comparisons(comparison_time_ms);
CREATE INDEX idx_dc_tenant_id ON duplicate_detector_service.dedup_comparisons(tenant_id);
CREATE INDEX idx_dc_created_at ON duplicate_detector_service.dedup_comparisons(created_at DESC);
CREATE INDEX idx_dc_job_score ON duplicate_detector_service.dedup_comparisons(job_id, overall_score DESC);
CREATE INDEX idx_dc_job_algorithm ON duplicate_detector_service.dedup_comparisons(job_id, algorithm_used);
CREATE INDEX idx_dc_job_record_a ON duplicate_detector_service.dedup_comparisons(job_id, record_a_id);
CREATE INDEX idx_dc_job_record_b ON duplicate_detector_service.dedup_comparisons(job_id, record_b_id);
CREATE INDEX idx_dc_record_pair ON duplicate_detector_service.dedup_comparisons(record_a_id, record_b_id);
CREATE INDEX idx_dc_tenant_job ON duplicate_detector_service.dedup_comparisons(tenant_id, job_id);
CREATE INDEX idx_dc_tenant_score ON duplicate_detector_service.dedup_comparisons(tenant_id, overall_score DESC);
CREATE INDEX idx_dc_field_scores ON duplicate_detector_service.dedup_comparisons USING GIN (field_scores);

-- dedup_matches indexes (16)
CREATE INDEX idx_dm_job_id ON duplicate_detector_service.dedup_matches(job_id);
CREATE INDEX idx_dm_record_a_id ON duplicate_detector_service.dedup_matches(record_a_id);
CREATE INDEX idx_dm_record_b_id ON duplicate_detector_service.dedup_matches(record_b_id);
CREATE INDEX idx_dm_classification ON duplicate_detector_service.dedup_matches(classification);
CREATE INDEX idx_dm_confidence ON duplicate_detector_service.dedup_matches(confidence DESC);
CREATE INDEX idx_dm_overall_score ON duplicate_detector_service.dedup_matches(overall_score DESC);
CREATE INDEX idx_dm_tenant_id ON duplicate_detector_service.dedup_matches(tenant_id);
CREATE INDEX idx_dm_created_at ON duplicate_detector_service.dedup_matches(created_at DESC);
CREATE INDEX idx_dm_job_classification ON duplicate_detector_service.dedup_matches(job_id, classification);
CREATE INDEX idx_dm_job_confidence ON duplicate_detector_service.dedup_matches(job_id, confidence DESC);
CREATE INDEX idx_dm_job_score ON duplicate_detector_service.dedup_matches(job_id, overall_score DESC);
CREATE INDEX idx_dm_record_pair ON duplicate_detector_service.dedup_matches(record_a_id, record_b_id);
CREATE INDEX idx_dm_tenant_job ON duplicate_detector_service.dedup_matches(tenant_id, job_id);
CREATE INDEX idx_dm_tenant_classification ON duplicate_detector_service.dedup_matches(tenant_id, classification);
CREATE INDEX idx_dm_tenant_confidence ON duplicate_detector_service.dedup_matches(tenant_id, confidence DESC);
CREATE INDEX idx_dm_field_scores ON duplicate_detector_service.dedup_matches USING GIN (field_scores);

-- dedup_clusters indexes (16)
CREATE INDEX idx_dcl_job_id ON duplicate_detector_service.dedup_clusters(job_id);
CREATE INDEX idx_dcl_representative_id ON duplicate_detector_service.dedup_clusters(representative_id);
CREATE INDEX idx_dcl_cluster_quality ON duplicate_detector_service.dedup_clusters(cluster_quality DESC);
CREATE INDEX idx_dcl_density ON duplicate_detector_service.dedup_clusters(density DESC);
CREATE INDEX idx_dcl_diameter ON duplicate_detector_service.dedup_clusters(diameter DESC);
CREATE INDEX idx_dcl_member_count ON duplicate_detector_service.dedup_clusters(member_count DESC);
CREATE INDEX idx_dcl_algorithm ON duplicate_detector_service.dedup_clusters(algorithm);
CREATE INDEX idx_dcl_tenant_id ON duplicate_detector_service.dedup_clusters(tenant_id);
CREATE INDEX idx_dcl_created_at ON duplicate_detector_service.dedup_clusters(created_at DESC);
CREATE INDEX idx_dcl_job_quality ON duplicate_detector_service.dedup_clusters(job_id, cluster_quality DESC);
CREATE INDEX idx_dcl_job_algorithm ON duplicate_detector_service.dedup_clusters(job_id, algorithm);
CREATE INDEX idx_dcl_job_count ON duplicate_detector_service.dedup_clusters(job_id, member_count DESC);
CREATE INDEX idx_dcl_tenant_job ON duplicate_detector_service.dedup_clusters(tenant_id, job_id);
CREATE INDEX idx_dcl_tenant_algorithm ON duplicate_detector_service.dedup_clusters(tenant_id, algorithm);
CREATE INDEX idx_dcl_tenant_quality ON duplicate_detector_service.dedup_clusters(tenant_id, cluster_quality DESC);
CREATE INDEX idx_dcl_member_record_ids ON duplicate_detector_service.dedup_clusters USING GIN (member_record_ids);

-- dedup_merge_decisions indexes (14)
CREATE INDEX idx_dmd_job_id ON duplicate_detector_service.dedup_merge_decisions(job_id);
CREATE INDEX idx_dmd_cluster_id ON duplicate_detector_service.dedup_merge_decisions(cluster_id);
CREATE INDEX idx_dmd_strategy ON duplicate_detector_service.dedup_merge_decisions(strategy);
CREATE INDEX idx_dmd_conflict_count ON duplicate_detector_service.dedup_merge_decisions(conflict_count DESC);
CREATE INDEX idx_dmd_provenance ON duplicate_detector_service.dedup_merge_decisions(provenance_hash);
CREATE INDEX idx_dmd_decided_at ON duplicate_detector_service.dedup_merge_decisions(decided_at DESC);
CREATE INDEX idx_dmd_tenant_id ON duplicate_detector_service.dedup_merge_decisions(tenant_id);
CREATE INDEX idx_dmd_job_strategy ON duplicate_detector_service.dedup_merge_decisions(job_id, strategy);
CREATE INDEX idx_dmd_job_cluster ON duplicate_detector_service.dedup_merge_decisions(job_id, cluster_id);
CREATE INDEX idx_dmd_job_conflicts ON duplicate_detector_service.dedup_merge_decisions(job_id, conflict_count DESC);
CREATE INDEX idx_dmd_tenant_job ON duplicate_detector_service.dedup_merge_decisions(tenant_id, job_id);
CREATE INDEX idx_dmd_tenant_strategy ON duplicate_detector_service.dedup_merge_decisions(tenant_id, strategy);
CREATE INDEX idx_dmd_merged_record ON duplicate_detector_service.dedup_merge_decisions USING GIN (merged_record);
CREATE INDEX idx_dmd_source_records ON duplicate_detector_service.dedup_merge_decisions USING GIN (source_records);

-- dedup_merge_conflicts indexes (14)
CREATE INDEX idx_dmc_merge_decision_id ON duplicate_detector_service.dedup_merge_conflicts(merge_decision_id);
CREATE INDEX idx_dmc_field_name ON duplicate_detector_service.dedup_merge_conflicts(field_name);
CREATE INDEX idx_dmc_resolution_method ON duplicate_detector_service.dedup_merge_conflicts(resolution_method);
CREATE INDEX idx_dmc_source_record_id ON duplicate_detector_service.dedup_merge_conflicts(source_record_id);
CREATE INDEX idx_dmc_tenant_id ON duplicate_detector_service.dedup_merge_conflicts(tenant_id);
CREATE INDEX idx_dmc_created_at ON duplicate_detector_service.dedup_merge_conflicts(created_at DESC);
CREATE INDEX idx_dmc_decision_field ON duplicate_detector_service.dedup_merge_conflicts(merge_decision_id, field_name);
CREATE INDEX idx_dmc_decision_method ON duplicate_detector_service.dedup_merge_conflicts(merge_decision_id, resolution_method);
CREATE INDEX idx_dmc_field_method ON duplicate_detector_service.dedup_merge_conflicts(field_name, resolution_method);
CREATE INDEX idx_dmc_tenant_decision ON duplicate_detector_service.dedup_merge_conflicts(tenant_id, merge_decision_id);
CREATE INDEX idx_dmc_tenant_field ON duplicate_detector_service.dedup_merge_conflicts(tenant_id, field_name);
CREATE INDEX idx_dmc_tenant_method ON duplicate_detector_service.dedup_merge_conflicts(tenant_id, resolution_method);
CREATE INDEX idx_dmc_tenant_created ON duplicate_detector_service.dedup_merge_conflicts(tenant_id, created_at DESC);
CREATE INDEX idx_dmc_conflicting_values ON duplicate_detector_service.dedup_merge_conflicts USING GIN (conflicting_values);

-- dedup_audit_log indexes (16)
CREATE INDEX idx_dal_job_id ON duplicate_detector_service.dedup_audit_log(job_id);
CREATE INDEX idx_dal_action ON duplicate_detector_service.dedup_audit_log(action);
CREATE INDEX idx_dal_entity_type ON duplicate_detector_service.dedup_audit_log(entity_type);
CREATE INDEX idx_dal_entity_id ON duplicate_detector_service.dedup_audit_log(entity_id);
CREATE INDEX idx_dal_provenance ON duplicate_detector_service.dedup_audit_log(provenance_hash);
CREATE INDEX idx_dal_performed_by ON duplicate_detector_service.dedup_audit_log(performed_by);
CREATE INDEX idx_dal_performed_at ON duplicate_detector_service.dedup_audit_log(performed_at DESC);
CREATE INDEX idx_dal_tenant_id ON duplicate_detector_service.dedup_audit_log(tenant_id);
CREATE INDEX idx_dal_job_action ON duplicate_detector_service.dedup_audit_log(job_id, action);
CREATE INDEX idx_dal_job_entity ON duplicate_detector_service.dedup_audit_log(job_id, entity_type);
CREATE INDEX idx_dal_action_entity ON duplicate_detector_service.dedup_audit_log(action, entity_type);
CREATE INDEX idx_dal_entity_type_id ON duplicate_detector_service.dedup_audit_log(entity_type, entity_id);
CREATE INDEX idx_dal_tenant_job ON duplicate_detector_service.dedup_audit_log(tenant_id, job_id);
CREATE INDEX idx_dal_tenant_action ON duplicate_detector_service.dedup_audit_log(tenant_id, action);
CREATE INDEX idx_dal_tenant_performed ON duplicate_detector_service.dedup_audit_log(tenant_id, performed_at DESC);
CREATE INDEX idx_dal_details ON duplicate_detector_service.dedup_audit_log USING GIN (details);

-- dedup_events indexes (hypertable-aware) (8)
CREATE INDEX idx_de_job_id ON duplicate_detector_service.dedup_events(job_id, event_time DESC);
CREATE INDEX idx_de_event_type ON duplicate_detector_service.dedup_events(event_type, event_time DESC);
CREATE INDEX idx_de_stage ON duplicate_detector_service.dedup_events(stage, event_time DESC);
CREATE INDEX idx_de_tenant_id ON duplicate_detector_service.dedup_events(tenant_id, event_time DESC);
CREATE INDEX idx_de_tenant_job ON duplicate_detector_service.dedup_events(tenant_id, job_id, event_time DESC);
CREATE INDEX idx_de_tenant_type ON duplicate_detector_service.dedup_events(tenant_id, event_type, event_time DESC);
CREATE INDEX idx_de_provenance ON duplicate_detector_service.dedup_events(provenance_hash, event_time DESC);
CREATE INDEX idx_de_details ON duplicate_detector_service.dedup_events USING GIN (details);

-- comparison_events indexes (hypertable-aware) (8)
CREATE INDEX idx_ce_job_id ON duplicate_detector_service.comparison_events(job_id, event_time DESC);
CREATE INDEX idx_ce_record_a ON duplicate_detector_service.comparison_events(record_a_id, event_time DESC);
CREATE INDEX idx_ce_record_b ON duplicate_detector_service.comparison_events(record_b_id, event_time DESC);
CREATE INDEX idx_ce_algorithm ON duplicate_detector_service.comparison_events(algorithm, event_time DESC);
CREATE INDEX idx_ce_classification ON duplicate_detector_service.comparison_events(classification, event_time DESC);
CREATE INDEX idx_ce_tenant_id ON duplicate_detector_service.comparison_events(tenant_id, event_time DESC);
CREATE INDEX idx_ce_tenant_job ON duplicate_detector_service.comparison_events(tenant_id, job_id, event_time DESC);
CREATE INDEX idx_ce_tenant_algorithm ON duplicate_detector_service.comparison_events(tenant_id, algorithm, event_time DESC);

-- merge_events indexes (hypertable-aware) (8)
CREATE INDEX idx_me_job_id ON duplicate_detector_service.merge_events(job_id, event_time DESC);
CREATE INDEX idx_me_cluster_id ON duplicate_detector_service.merge_events(cluster_id, event_time DESC);
CREATE INDEX idx_me_strategy ON duplicate_detector_service.merge_events(strategy, event_time DESC);
CREATE INDEX idx_me_tenant_id ON duplicate_detector_service.merge_events(tenant_id, event_time DESC);
CREATE INDEX idx_me_tenant_job ON duplicate_detector_service.merge_events(tenant_id, job_id, event_time DESC);
CREATE INDEX idx_me_tenant_strategy ON duplicate_detector_service.merge_events(tenant_id, strategy, event_time DESC);
CREATE INDEX idx_me_provenance ON duplicate_detector_service.merge_events(provenance_hash, event_time DESC);
CREATE INDEX idx_me_tenant_cluster ON duplicate_detector_service.merge_events(tenant_id, cluster_id, event_time DESC);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

-- dedup_rules: tenant-scoped
ALTER TABLE duplicate_detector_service.dedup_rules ENABLE ROW LEVEL SECURITY;
CREATE POLICY dr_tenant_read ON duplicate_detector_service.dedup_rules
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY dr_tenant_write ON duplicate_detector_service.dedup_rules
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- dedup_jobs: tenant-scoped
ALTER TABLE duplicate_detector_service.dedup_jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY dj_tenant_read ON duplicate_detector_service.dedup_jobs
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY dj_tenant_write ON duplicate_detector_service.dedup_jobs
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- dedup_fingerprints: tenant-scoped
ALTER TABLE duplicate_detector_service.dedup_fingerprints ENABLE ROW LEVEL SECURITY;
CREATE POLICY df_tenant_read ON duplicate_detector_service.dedup_fingerprints
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY df_tenant_write ON duplicate_detector_service.dedup_fingerprints
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- dedup_blocks: tenant-scoped
ALTER TABLE duplicate_detector_service.dedup_blocks ENABLE ROW LEVEL SECURITY;
CREATE POLICY dbl_tenant_read ON duplicate_detector_service.dedup_blocks
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY dbl_tenant_write ON duplicate_detector_service.dedup_blocks
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- dedup_comparisons: tenant-scoped
ALTER TABLE duplicate_detector_service.dedup_comparisons ENABLE ROW LEVEL SECURITY;
CREATE POLICY dc_tenant_read ON duplicate_detector_service.dedup_comparisons
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY dc_tenant_write ON duplicate_detector_service.dedup_comparisons
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- dedup_matches: tenant-scoped
ALTER TABLE duplicate_detector_service.dedup_matches ENABLE ROW LEVEL SECURITY;
CREATE POLICY dm_tenant_read ON duplicate_detector_service.dedup_matches
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY dm_tenant_write ON duplicate_detector_service.dedup_matches
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- dedup_clusters: tenant-scoped
ALTER TABLE duplicate_detector_service.dedup_clusters ENABLE ROW LEVEL SECURITY;
CREATE POLICY dcl_tenant_read ON duplicate_detector_service.dedup_clusters
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY dcl_tenant_write ON duplicate_detector_service.dedup_clusters
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- dedup_merge_decisions: tenant-scoped
ALTER TABLE duplicate_detector_service.dedup_merge_decisions ENABLE ROW LEVEL SECURITY;
CREATE POLICY dmd_tenant_read ON duplicate_detector_service.dedup_merge_decisions
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY dmd_tenant_write ON duplicate_detector_service.dedup_merge_decisions
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- dedup_merge_conflicts: tenant-scoped
ALTER TABLE duplicate_detector_service.dedup_merge_conflicts ENABLE ROW LEVEL SECURITY;
CREATE POLICY dmc_tenant_read ON duplicate_detector_service.dedup_merge_conflicts
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY dmc_tenant_write ON duplicate_detector_service.dedup_merge_conflicts
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- dedup_audit_log: tenant-scoped
ALTER TABLE duplicate_detector_service.dedup_audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY dal_tenant_read ON duplicate_detector_service.dedup_audit_log
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY dal_tenant_write ON duplicate_detector_service.dedup_audit_log
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- dedup_events: open (hypertable)
ALTER TABLE duplicate_detector_service.dedup_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY de_tenant_read ON duplicate_detector_service.dedup_events
    FOR SELECT USING (TRUE);
CREATE POLICY de_tenant_write ON duplicate_detector_service.dedup_events
    FOR ALL USING (TRUE);

-- comparison_events: open (hypertable)
ALTER TABLE duplicate_detector_service.comparison_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY ce_tenant_read ON duplicate_detector_service.comparison_events
    FOR SELECT USING (TRUE);
CREATE POLICY ce_tenant_write ON duplicate_detector_service.comparison_events
    FOR ALL USING (TRUE);

-- merge_events: open (hypertable)
ALTER TABLE duplicate_detector_service.merge_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY me_tenant_read ON duplicate_detector_service.merge_events
    FOR SELECT USING (TRUE);
CREATE POLICY me_tenant_write ON duplicate_detector_service.merge_events
    FOR ALL USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA duplicate_detector_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA duplicate_detector_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA duplicate_detector_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON duplicate_detector_service.dedup_hourly_stats TO greenlang_app;
GRANT SELECT ON duplicate_detector_service.comparison_hourly_stats TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA duplicate_detector_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA duplicate_detector_service TO greenlang_readonly;
GRANT SELECT ON duplicate_detector_service.dedup_hourly_stats TO greenlang_readonly;
GRANT SELECT ON duplicate_detector_service.comparison_hourly_stats TO greenlang_readonly;

-- Admin role
GRANT ALL ON SCHEMA duplicate_detector_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA duplicate_detector_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA duplicate_detector_service TO greenlang_admin;

-- Add duplicate detector service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'duplicate_detector:jobs:read', 'duplicate_detector', 'jobs_read', 'View deduplication jobs and their progress'),
    (gen_random_uuid(), 'duplicate_detector:jobs:write', 'duplicate_detector', 'jobs_write', 'Create, start, cancel, and manage deduplication jobs'),
    (gen_random_uuid(), 'duplicate_detector:fingerprints:read', 'duplicate_detector', 'fingerprints_read', 'View record fingerprints and normalized fields'),
    (gen_random_uuid(), 'duplicate_detector:fingerprints:write', 'duplicate_detector', 'fingerprints_write', 'Generate and manage record fingerprints'),
    (gen_random_uuid(), 'duplicate_detector:comparisons:read', 'duplicate_detector', 'comparisons_read', 'View pairwise comparison results and field scores'),
    (gen_random_uuid(), 'duplicate_detector:comparisons:write', 'duplicate_detector', 'comparisons_write', 'Create and manage pairwise comparisons'),
    (gen_random_uuid(), 'duplicate_detector:matches:read', 'duplicate_detector', 'matches_read', 'View classified matches and confidence scores'),
    (gen_random_uuid(), 'duplicate_detector:matches:write', 'duplicate_detector', 'matches_write', 'Classify and manage duplicate matches'),
    (gen_random_uuid(), 'duplicate_detector:clusters:read', 'duplicate_detector', 'clusters_read', 'View duplicate clusters and quality metrics'),
    (gen_random_uuid(), 'duplicate_detector:clusters:write', 'duplicate_detector', 'clusters_write', 'Create and manage duplicate clusters'),
    (gen_random_uuid(), 'duplicate_detector:merge:read', 'duplicate_detector', 'merge_read', 'View merge decisions, conflicts, and resolved records'),
    (gen_random_uuid(), 'duplicate_detector:merge:write', 'duplicate_detector', 'merge_write', 'Execute merge decisions and resolve field conflicts'),
    (gen_random_uuid(), 'duplicate_detector:rules:read', 'duplicate_detector', 'rules_read', 'View dedup rule definitions and configurations'),
    (gen_random_uuid(), 'duplicate_detector:rules:write', 'duplicate_detector', 'rules_write', 'Create, update, and manage dedup rule sets'),
    (gen_random_uuid(), 'duplicate_detector:audit:read', 'duplicate_detector', 'audit_read', 'View dedup audit log entries and provenance chains'),
    (gen_random_uuid(), 'duplicate_detector:admin', 'duplicate_detector', 'admin', 'Duplicate detector service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep dedup event records for 90 days
SELECT add_retention_policy('duplicate_detector_service.dedup_events', INTERVAL '90 days');

-- Keep comparison event records for 90 days
SELECT add_retention_policy('duplicate_detector_service.comparison_events', INTERVAL '90 days');

-- Keep merge event records for 90 days
SELECT add_retention_policy('duplicate_detector_service.merge_events', INTERVAL '90 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on dedup_events after 7 days
ALTER TABLE duplicate_detector_service.dedup_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'event_time DESC'
);

SELECT add_compression_policy('duplicate_detector_service.dedup_events', INTERVAL '7 days');

-- Enable compression on comparison_events after 7 days
ALTER TABLE duplicate_detector_service.comparison_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'event_time DESC'
);

SELECT add_compression_policy('duplicate_detector_service.comparison_events', INTERVAL '7 days');

-- Enable compression on merge_events after 7 days
ALTER TABLE duplicate_detector_service.merge_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'event_time DESC'
);

SELECT add_compression_policy('duplicate_detector_service.merge_events', INTERVAL '7 days');

-- =============================================================================
-- Seed: Register the Duplicate Detection Agent (GL-DATA-X-014)
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES
('GL-DATA-X-014', 'Duplicate Detection Agent',
 'Comprehensive duplicate detection and record deduplication engine for GreenLang Climate OS. Generates record fingerprints (SHA-256, SimHash, MinHash) for efficient duplicate candidate identification. Applies blocking strategies (sorted neighborhood, standard, canopy) to reduce comparison space. Performs pairwise field-level comparison with configurable similarity algorithms. Classifies record pairs as match/non_match/possible with confidence scoring and configurable thresholds. Forms duplicate clusters with quality/density/diameter metrics. Executes merge strategies (keep_first, keep_latest, keep_most_complete, merge_fields, golden_record, custom) with field-level conflict resolution. Configurable dedup rule sets with match and possible thresholds. SHA-256 provenance chains for zero-hallucination audit trail.',
 2, 'async', true, true, 5, '1.0.0', true,
 'GreenLang Data Team', 'https://docs.greenlang.ai/agents/duplicate-detector', true, 'default')
ON CONFLICT (agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Agent Version for Duplicate Detection Agent
-- =============================================================================

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES
('GL-DATA-X-014', '1.0.0',
 '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "1Gi", "memory_limit": "4Gi", "gpu": false}'::jsonb,
 '{"image": "greenlang/duplicate-detector-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"deduplication", "fingerprinting", "blocking", "matching", "clustering", "merging", "conflict-resolution", "record-linkage"}',
 '{"cross-sector", "manufacturing", "retail", "energy", "finance", "healthcare", "agriculture"}',
 'c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4')
ON CONFLICT (agent_id, version) DO NOTHING;

-- =============================================================================
-- Seed: Agent Capabilities for Duplicate Detection Agent
-- =============================================================================

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES

('GL-DATA-X-014', '1.0.0', 'record_fingerprinting', 'processing',
 'Generate record fingerprints using SHA-256, SimHash, or MinHash algorithms for efficient duplicate candidate identification. Normalizes field values before hashing to improve fingerprint accuracy. Supports configurable field sets for targeted fingerprinting across multiple datasets',
 '{"records", "field_set", "algorithm", "config"}', '{"fingerprints", "total_fingerprinted", "duplicate_candidates"}',
 '{"algorithms": ["sha256", "simhash", "minhash"], "normalization": ["lowercase", "trim", "strip_punctuation", "phonetic", "metaphone"], "max_field_sets": 20, "batch_size": 10000}'::jsonb),

('GL-DATA-X-014', '1.0.0', 'blocking', 'processing',
 'Apply blocking strategies to reduce the pairwise comparison space from O(n^2) to O(n*b) where b is block size. Supports sorted neighborhood (sliding window), standard blocking (partition by key), and canopy clustering (loose/tight thresholds) strategies with configurable blocking fields',
 '{"fingerprints", "strategy", "blocking_fields", "config"}', '{"blocks", "total_blocks", "avg_block_size", "reduction_ratio"}',
 '{"strategies": ["sorted_neighborhood", "standard", "canopy", "none"], "window_size_default": 5, "canopy_tight_default": 0.8, "canopy_loose_default": 0.4, "max_block_size": 1000}'::jsonb),

('GL-DATA-X-014', '1.0.0', 'pairwise_comparison', 'analysis',
 'Perform pairwise field-level comparison within blocks using configurable similarity algorithms (Levenshtein, Jaro-Winkler, cosine, exact, numeric distance, date proximity, phonetic, token-based). Produces per-field scores and weighted overall similarity score (0-1) for each record pair',
 '{"blocks", "field_configs", "config"}', '{"comparisons", "total_compared", "avg_score", "score_distribution"}',
 '{"algorithms": ["levenshtein", "jaro_winkler", "cosine", "exact", "numeric_distance", "date_proximity", "phonetic", "soundex", "token_sort", "token_set"], "field_weights": true, "missing_value_handling": ["zero", "skip", "average"], "max_comparisons_per_block": 50000}'::jsonb),

('GL-DATA-X-014', '1.0.0', 'match_classification', 'classification',
 'Classify record pairs as match, non_match, or possible based on configurable match and possible thresholds with confidence scoring. Supports Fellegi-Sunter probabilistic matching, rule-based classification, and threshold-based classification. Records decision reasoning for audit trail',
 '{"comparisons", "match_threshold", "possible_threshold", "config"}', '{"matches", "match_count", "non_match_count", "possible_count", "confidence_stats"}',
 '{"methods": ["threshold", "fellegi_sunter", "rule_based"], "default_match_threshold": 0.85, "default_possible_threshold": 0.65, "confidence_bins": [0.0, 0.25, 0.50, 0.75, 0.90, 1.0], "record_decisions": true}'::jsonb),

('GL-DATA-X-014', '1.0.0', 'duplicate_clustering', 'analysis',
 'Form duplicate clusters from classified matches using transitive closure, connected components, or hierarchical clustering. Computes cluster quality (intra-cluster similarity), density (edge ratio), diameter (max pairwise distance), and elects a representative record per cluster',
 '{"matches", "algorithm", "config"}', '{"clusters", "total_clusters", "avg_cluster_size", "quality_stats"}',
 '{"algorithms": ["transitive_closure", "connected_components", "hierarchical", "star"], "representative_selection": ["most_complete", "most_recent", "highest_quality", "centroid"], "min_cluster_size": 2, "max_cluster_size": 100}'::jsonb),

('GL-DATA-X-014', '1.0.0', 'merge_execution', 'processing',
 'Execute merge strategies on duplicate clusters to produce golden records. Supports keep_first, keep_latest, keep_most_complete, merge_fields, golden_record, and custom strategies. Detects and resolves field-level conflicts using configurable resolution methods (most_recent, most_frequent, longest, rule_based). SHA-256 provenance hash on every merge decision',
 '{"clusters", "strategy", "config"}', '{"merge_decisions", "merged_count", "conflict_count", "provenance_hashes"}',
 '{"strategies": ["keep_first", "keep_latest", "keep_most_complete", "merge_fields", "golden_record", "custom"], "resolution_methods": ["most_recent", "most_frequent", "longest", "shortest", "first", "last", "manual", "rule_based", "custom", "highest", "lowest", "concatenate"], "provenance_tracking": true, "rollback_support": true}'::jsonb),

('GL-DATA-X-014', '1.0.0', 'rule_management', 'governance',
 'Create and manage configurable dedup rule sets with field comparison configurations, match and possible thresholds, blocking strategy and field selections, merge strategy preferences, versioning, and activation state. Rules are tenant-scoped and support version history for audit trail',
 '{"rule_definition", "config"}', '{"rule_id", "version", "validation_result"}',
 '{"field_config_options": ["algorithm", "weight", "threshold", "missing_handling", "preprocessing"], "blocking_strategies": ["sorted_neighborhood", "standard", "canopy", "none"], "merge_strategies": ["keep_first", "keep_latest", "keep_most_complete", "merge_fields", "golden_record", "custom"], "version_control": true, "import_export": true}'::jsonb)

ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: Agent Dependencies for Duplicate Detection Agent
-- =============================================================================

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES

-- Duplicate Detector depends on Schema Compiler for input/output validation
('GL-DATA-X-014', 'GL-FOUND-X-002', '>=1.0.0', false,
 'Dedup rules, job configurations, and merge decisions are validated against JSON Schema definitions'),

-- Duplicate Detector depends on Registry for agent discovery
('GL-DATA-X-014', 'GL-FOUND-X-007', '>=1.0.0', false,
 'Agent version and capability lookup for dedup pipeline orchestration'),

-- Duplicate Detector depends on Access Guard for policy enforcement
('GL-DATA-X-014', 'GL-FOUND-X-006', '>=1.0.0', false,
 'Data classification and access control enforcement for dedup jobs and merge decisions'),

-- Duplicate Detector depends on Observability Agent for metrics
('GL-DATA-X-014', 'GL-FOUND-X-010', '>=1.0.0', false,
 'Dedup metrics, comparison throughput, match rates, and merge statistics are reported to observability'),

-- Duplicate Detector optionally uses Citations for provenance tracking
('GL-DATA-X-014', 'GL-FOUND-X-005', '>=1.0.0', true,
 'Merge decision provenance and fingerprint audit trails are registered with the citation service'),

-- Duplicate Detector optionally uses Reproducibility for determinism
('GL-DATA-X-014', 'GL-FOUND-X-008', '>=1.0.0', true,
 'Dedup results are verified for reproducibility across re-execution with identical inputs and rules'),

-- Duplicate Detector optionally uses QA Test Harness
('GL-DATA-X-014', 'GL-FOUND-X-009', '>=1.0.0', true,
 'Dedup results are validated through the QA Test Harness for zero-hallucination verification'),

-- Duplicate Detector optionally integrates with Excel Normalizer
('GL-DATA-X-014', 'GL-DATA-X-002', '>=1.0.0', true,
 'Normalized datasets from Excel/CSV processing are deduplicated before downstream emission calculations'),

-- Duplicate Detector optionally integrates with ERP Connector
('GL-DATA-X-014', 'GL-DATA-X-003', '>=1.0.0', true,
 'ERP-sourced records (suppliers, materials, purchase orders) are deduplicated for data consistency'),

-- Duplicate Detector optionally integrates with Data Quality Profiler
('GL-DATA-X-014', 'GL-DATA-X-013', '>=1.0.0', true,
 'Data quality profiling results inform dedup rule configuration and duplicate rate benchmarking')

ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Service Catalog Entry for Duplicate Detection Agent
-- =============================================================================

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES
('GL-DATA-X-014', 'Duplicate Detection Agent',
 'Comprehensive duplicate detection and record deduplication engine. Generates record fingerprints (SHA-256/SimHash/MinHash) for efficient candidate identification. Applies blocking strategies (sorted neighborhood/standard/canopy) to reduce comparison space. Performs pairwise field-level comparison with 10+ similarity algorithms (Levenshtein, Jaro-Winkler, cosine, phonetic, token-based). Classifies matches with configurable thresholds and confidence scoring. Forms duplicate clusters with quality/density/diameter metrics. Executes merge strategies (keep_first/keep_latest/keep_most_complete/merge_fields/golden_record/custom) with field-level conflict resolution. Configurable dedup rule sets with version history. SHA-256 provenance chains for zero-hallucination audit trail.',
 'data', 'active', 'default')
ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA duplicate_detector_service IS 'Duplicate Detection Agent for GreenLang Climate OS (AGENT-DATA-011) - record fingerprinting, blocking, pairwise comparison, match classification, clustering, merge execution with conflict resolution, configurable rule sets, and comprehensive audit logging with provenance chains';
COMMENT ON TABLE duplicate_detector_service.dedup_rules IS 'Configurable dedup rule sets with field comparison configurations, match/possible thresholds, blocking strategy/fields, merge strategy, versioning, and activation state';
COMMENT ON TABLE duplicate_detector_service.dedup_jobs IS 'Job tracking for dedup runs with dataset IDs, rule reference, status/stage progression, per-stage record counts, duplicate rate, configuration, and SHA-256 provenance hash';
COMMENT ON TABLE duplicate_detector_service.dedup_fingerprints IS 'Record fingerprints with record/dataset identifiers, field set, fingerprint hash (SHA-256/SimHash/MinHash), and normalized field values for downstream comparison';
COMMENT ON TABLE duplicate_detector_service.dedup_blocks IS 'Blocking results with block key, strategy (sorted_neighborhood/standard/canopy/none), member record IDs, and record count for comparison space reduction';
COMMENT ON TABLE duplicate_detector_service.dedup_comparisons IS 'Pairwise comparison results with record pair, per-field similarity scores (JSONB), overall similarity score (0-1), algorithm used, and comparison duration';
COMMENT ON TABLE duplicate_detector_service.dedup_matches IS 'Classified matches with record pair, classification (match/non_match/possible), confidence score, field scores, overall score, and decision reasoning';
COMMENT ON TABLE duplicate_detector_service.dedup_clusters IS 'Duplicate clusters with member record IDs, representative record, cluster quality/density/diameter metrics, member count, and clustering algorithm';
COMMENT ON TABLE duplicate_detector_service.dedup_merge_decisions IS 'Merge decisions with strategy, merged record (JSONB golden record), source records, conflict count, and SHA-256 provenance hash for audit trail';
COMMENT ON TABLE duplicate_detector_service.dedup_merge_conflicts IS 'Field-level merge conflicts with conflicting values, chosen resolved value, resolution method, and source record attribution';
COMMENT ON TABLE duplicate_detector_service.dedup_audit_log IS 'Comprehensive audit trail for all dedup operations with action, entity type/ID, details (JSONB), provenance hash, performer, and timestamp';
COMMENT ON TABLE duplicate_detector_service.dedup_events IS 'TimescaleDB hypertable: dedup lifecycle event time-series with job ID, event type, pipeline stage, record count, duration, and details';
COMMENT ON TABLE duplicate_detector_service.comparison_events IS 'TimescaleDB hypertable: pairwise comparison event time-series with job ID, record pair, algorithm, score, and classification';
COMMENT ON TABLE duplicate_detector_service.merge_events IS 'TimescaleDB hypertable: merge decision event time-series with job ID, cluster ID, strategy, records merged, conflicts, and provenance hash';
COMMENT ON MATERIALIZED VIEW duplicate_detector_service.dedup_hourly_stats IS 'Continuous aggregate: hourly dedup event statistics by event type with total events, unique jobs, avg/sum record counts, avg/max/min duration, and per-stage event counts';
COMMENT ON MATERIALIZED VIEW duplicate_detector_service.comparison_hourly_stats IS 'Continuous aggregate: hourly comparison statistics by algorithm with total comparisons, unique jobs, avg/min/max score, and match/non_match/possible counts';

COMMENT ON COLUMN duplicate_detector_service.dedup_rules.field_configs IS 'JSONB object defining per-field comparison configuration (algorithm, weight, threshold, preprocessing, missing value handling)';
COMMENT ON COLUMN duplicate_detector_service.dedup_rules.match_threshold IS 'Minimum overall similarity score (0-1) to classify a record pair as a definite match';
COMMENT ON COLUMN duplicate_detector_service.dedup_rules.possible_threshold IS 'Minimum overall similarity score (0-1) to classify a record pair as a possible match requiring review';
COMMENT ON COLUMN duplicate_detector_service.dedup_rules.blocking_strategy IS 'Blocking strategy: sorted_neighborhood (sliding window), standard (partition by key), canopy (loose/tight thresholds), none (exhaustive)';
COMMENT ON COLUMN duplicate_detector_service.dedup_rules.blocking_fields IS 'Array of field names used as blocking keys to reduce comparison space';
COMMENT ON COLUMN duplicate_detector_service.dedup_rules.merge_strategy IS 'Merge strategy: keep_first, keep_latest, keep_most_complete, merge_fields, golden_record, custom';
COMMENT ON COLUMN duplicate_detector_service.dedup_jobs.status IS 'Job status: pending, running, completed, failed, cancelled';
COMMENT ON COLUMN duplicate_detector_service.dedup_jobs.stage IS 'Current pipeline stage: fingerprint, block, compare, classify, cluster, merge, complete';
COMMENT ON COLUMN duplicate_detector_service.dedup_jobs.duplicate_rate IS 'Ratio of duplicate records to total records (0-1), computed after clustering';
COMMENT ON COLUMN duplicate_detector_service.dedup_jobs.provenance_hash IS 'SHA-256 provenance hash for integrity verification and audit trail';
COMMENT ON COLUMN duplicate_detector_service.dedup_fingerprints.algorithm IS 'Fingerprint algorithm: sha256 (exact), simhash (near-duplicate), minhash (set similarity)';
COMMENT ON COLUMN duplicate_detector_service.dedup_fingerprints.fingerprint_hash IS 'Computed fingerprint hash value for the record using the specified algorithm';
COMMENT ON COLUMN duplicate_detector_service.dedup_fingerprints.normalized_fields IS 'JSONB object of normalized field values used for fingerprint computation';
COMMENT ON COLUMN duplicate_detector_service.dedup_blocks.strategy IS 'Blocking strategy: sorted_neighborhood, standard, canopy, none';
COMMENT ON COLUMN duplicate_detector_service.dedup_blocks.block_key IS 'Blocking key value that groups records into the same block for comparison';
COMMENT ON COLUMN duplicate_detector_service.dedup_comparisons.overall_score IS 'Weighted average similarity score across all compared fields (0-1)';
COMMENT ON COLUMN duplicate_detector_service.dedup_comparisons.field_scores IS 'JSONB object of per-field similarity scores (0-1) for the record pair';
COMMENT ON COLUMN duplicate_detector_service.dedup_comparisons.algorithm_used IS 'Primary comparison algorithm used for overall score computation';
COMMENT ON COLUMN duplicate_detector_service.dedup_matches.classification IS 'Match classification: match (definite duplicate), non_match (distinct), possible (requires review)';
COMMENT ON COLUMN duplicate_detector_service.dedup_matches.confidence IS 'Classification confidence score (0-1) indicating certainty of the match decision';
COMMENT ON COLUMN duplicate_detector_service.dedup_matches.decision_reason IS 'Human-readable explanation of why the classification was made';
COMMENT ON COLUMN duplicate_detector_service.dedup_clusters.cluster_quality IS 'Intra-cluster similarity score (0-1) measuring average pairwise similarity within the cluster';
COMMENT ON COLUMN duplicate_detector_service.dedup_clusters.density IS 'Edge density ratio (0-1) of actual match edges to possible edges in the cluster';
COMMENT ON COLUMN duplicate_detector_service.dedup_clusters.diameter IS 'Maximum pairwise distance (0-1) between any two members in the cluster';
COMMENT ON COLUMN duplicate_detector_service.dedup_clusters.representative_id IS 'Record ID of the elected cluster representative (most complete, most recent, or highest quality)';
COMMENT ON COLUMN duplicate_detector_service.dedup_merge_decisions.strategy IS 'Merge strategy: keep_first, keep_latest, keep_most_complete, merge_fields, golden_record, custom';
COMMENT ON COLUMN duplicate_detector_service.dedup_merge_decisions.merged_record IS 'JSONB of the resulting golden record after merge execution';
COMMENT ON COLUMN duplicate_detector_service.dedup_merge_decisions.source_records IS 'JSONB array of source records that were merged into the golden record';
COMMENT ON COLUMN duplicate_detector_service.dedup_merge_decisions.provenance_hash IS 'SHA-256 provenance hash of the merge decision for audit trail verification';
COMMENT ON COLUMN duplicate_detector_service.dedup_merge_conflicts.resolution_method IS 'Conflict resolution method: most_recent, most_frequent, longest, shortest, first, last, manual, rule_based, custom, highest, lowest, concatenate, none';
COMMENT ON COLUMN duplicate_detector_service.dedup_merge_conflicts.conflicting_values IS 'JSONB array of conflicting field values from different source records';
COMMENT ON COLUMN duplicate_detector_service.dedup_merge_conflicts.chosen_value IS 'The resolved value chosen for the merged golden record';
COMMENT ON COLUMN duplicate_detector_service.dedup_audit_log.action IS 'Audit action: job_created, job_started, job_completed, job_failed, job_cancelled, fingerprint_generated, block_created, comparison_performed, match_classified, cluster_formed, merge_decided, conflict_resolved, rule_created, rule_updated, rule_deleted, etc.';
COMMENT ON COLUMN duplicate_detector_service.dedup_audit_log.entity_type IS 'Entity type: job, fingerprint, block, comparison, match, cluster, merge_decision, merge_conflict, rule, config';
COMMENT ON COLUMN duplicate_detector_service.dedup_audit_log.provenance_hash IS 'SHA-256 provenance hash linking the audit entry to a specific data state';
COMMENT ON COLUMN duplicate_detector_service.dedup_events.event_type IS 'Dedup event type: job_started/completed/failed/cancelled, fingerprint_started/completed/failed, blocking_started/completed/failed, comparison_started/completed/failed, classification_started/completed/failed, clustering_started/completed/failed, merging_started/completed/failed, stage_transition, progress_update, threshold_breach';
COMMENT ON COLUMN duplicate_detector_service.dedup_events.stage IS 'Pipeline stage: fingerprint, block, compare, classify, cluster, merge, complete';
COMMENT ON COLUMN duplicate_detector_service.comparison_events.classification IS 'Match classification result: match, non_match, possible';
COMMENT ON COLUMN duplicate_detector_service.merge_events.strategy IS 'Merge strategy used: keep_first, keep_latest, keep_most_complete, merge_fields, golden_record, custom';
COMMENT ON COLUMN duplicate_detector_service.merge_events.conflicts IS 'Number of field-level conflicts encountered during merge execution';
