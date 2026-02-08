-- =============================================================================
-- GreenLang Climate OS - Reproducibility Agent Schema
-- =============================================================================
-- Migration: V028
-- Component: AGENT-FOUND-008 Reproducibility Agent
-- Description: Creates reproducibility_service schema with verification_runs,
--              artifact_hashes, environment_fingerprints, seed_configurations,
--              version_manifests, drift_baselines, replay_sessions (hypertable),
--              reproducibility_audit_log (hypertable), continuous aggregates for
--              hourly verification stats and hourly audit stats, 50+ indexes
--              (including GIN indexes on JSONB and arrays), RLS policies per
--              tenant, 14 security permissions, retention policies (30-day
--              replay_sessions, 365-day audit_log), compression, and seed data
--              registering the Reproducibility Agent (GL-FOUND-X-008) with
--              capabilities in the agent registry.
-- Previous: V027__agent_registry_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS reproducibility_service;

-- =============================================================================
-- Table: reproducibility_service.verification_runs
-- =============================================================================
-- Core verification records. Each verification run captures the execution being
-- verified, input/output hashes, environment fingerprint, seed configuration,
-- version manifest, drift baseline, reproducibility result, list of checks
-- performed, non-determinism sources detected, processing time, and a
-- provenance hash for integrity verification. Tenant-scoped.

CREATE TABLE reproducibility_service.verification_runs (
    verification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id VARCHAR(255) NOT NULL,
    status VARCHAR(20) NOT NULL,
    input_hash VARCHAR(64) NOT NULL,
    output_hash VARCHAR(64),
    environment_fingerprint_id UUID,
    seed_config_id UUID,
    version_manifest_id UUID,
    drift_baseline_id UUID,
    is_reproducible BOOLEAN NOT NULL DEFAULT true,
    checks JSONB NOT NULL DEFAULT '[]'::jsonb,
    non_determinism_sources TEXT[] DEFAULT '{}',
    non_determinism_details JSONB DEFAULT '{}'::jsonb,
    absolute_tolerance DOUBLE PRECISION NOT NULL DEFAULT 1e-9,
    relative_tolerance DOUBLE PRECISION NOT NULL DEFAULT 1e-6,
    processing_time_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    provenance_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_by VARCHAR(255) DEFAULT 'system'
);

-- Status constraint
ALTER TABLE reproducibility_service.verification_runs
    ADD CONSTRAINT chk_verification_status
    CHECK (status IN ('pass', 'fail', 'warning', 'skipped'));

-- Input hash must be 64-character hex (SHA-256)
ALTER TABLE reproducibility_service.verification_runs
    ADD CONSTRAINT chk_verification_input_hash_length
    CHECK (LENGTH(input_hash) = 64);

-- Output hash must be 64-character hex when present
ALTER TABLE reproducibility_service.verification_runs
    ADD CONSTRAINT chk_verification_output_hash_length
    CHECK (output_hash IS NULL OR LENGTH(output_hash) = 64);

-- Provenance hash must be 64-character hex
ALTER TABLE reproducibility_service.verification_runs
    ADD CONSTRAINT chk_verification_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- Processing time must be non-negative
ALTER TABLE reproducibility_service.verification_runs
    ADD CONSTRAINT chk_verification_processing_time_positive
    CHECK (processing_time_ms >= 0);

-- Tolerances must be positive
ALTER TABLE reproducibility_service.verification_runs
    ADD CONSTRAINT chk_verification_absolute_tolerance_positive
    CHECK (absolute_tolerance > 0);

ALTER TABLE reproducibility_service.verification_runs
    ADD CONSTRAINT chk_verification_relative_tolerance_positive
    CHECK (relative_tolerance > 0);

-- =============================================================================
-- Table: reproducibility_service.artifact_hashes
-- =============================================================================
-- Computed hash records for artifacts. Each record captures the artifact being
-- hashed, its type, the computed hash, the algorithm used, whether
-- normalization was applied before hashing, and a provenance hash for
-- integrity verification. Used by the Reproducibility Agent to compare
-- input/output artifacts across executions for determinism verification.

CREATE TABLE reproducibility_service.artifact_hashes (
    hash_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_id VARCHAR(255) NOT NULL,
    artifact_type VARCHAR(50) NOT NULL,
    data_hash VARCHAR(64) NOT NULL,
    algorithm VARCHAR(20) NOT NULL DEFAULT 'sha256',
    normalization_applied BOOLEAN NOT NULL DEFAULT true,
    metadata JSONB DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(64) NOT NULL,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_by VARCHAR(255) DEFAULT 'system'
);

-- Artifact type constraint
ALTER TABLE reproducibility_service.artifact_hashes
    ADD CONSTRAINT chk_artifact_type
    CHECK (artifact_type IN (
        'input_data', 'output_data', 'model_weights', 'configuration',
        'emission_factor', 'calculation_result', 'report', 'intermediate',
        'checkpoint', 'schema', 'reference_data', 'seed_state'
    ));

-- Algorithm constraint
ALTER TABLE reproducibility_service.artifact_hashes
    ADD CONSTRAINT chk_artifact_algorithm
    CHECK (algorithm IN ('sha256', 'sha384', 'sha512', 'blake2b', 'blake3'));

-- Data hash must be at least 64 characters (SHA-256 minimum)
ALTER TABLE reproducibility_service.artifact_hashes
    ADD CONSTRAINT chk_artifact_data_hash_length
    CHECK (LENGTH(data_hash) >= 64);

-- Provenance hash must be 64-character hex
ALTER TABLE reproducibility_service.artifact_hashes
    ADD CONSTRAINT chk_artifact_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: reproducibility_service.environment_fingerprints
-- =============================================================================
-- Environment fingerprint records. Each fingerprint captures the full
-- execution environment at the time of a verification run: Python version,
-- platform system/release/machine, hostname, GreenLang version, dependency
-- versions (JSONB), environment variables (JSONB with sensitive values
-- redacted), and a computed hash of the entire fingerprint for comparison.
-- Used to detect environment drift between executions.

CREATE TABLE reproducibility_service.environment_fingerprints (
    fingerprint_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    python_version VARCHAR(20) NOT NULL,
    platform_system VARCHAR(50) NOT NULL,
    platform_release VARCHAR(100),
    platform_machine VARCHAR(50) NOT NULL,
    hostname VARCHAR(255),
    greenlang_version VARCHAR(20) NOT NULL DEFAULT '1.0.0',
    dependency_versions JSONB NOT NULL DEFAULT '{}'::jsonb,
    environment_variables JSONB NOT NULL DEFAULT '{}'::jsonb,
    cpu_count INTEGER,
    total_memory_mb BIGINT,
    gpu_info JSONB DEFAULT '{}'::jsonb,
    environment_hash VARCHAR(64) NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default'
);

-- Platform system constraint
ALTER TABLE reproducibility_service.environment_fingerprints
    ADD CONSTRAINT chk_fingerprint_platform_system
    CHECK (platform_system IN ('Linux', 'Darwin', 'Windows', 'FreeBSD'));

-- Platform machine constraint
ALTER TABLE reproducibility_service.environment_fingerprints
    ADD CONSTRAINT chk_fingerprint_platform_machine
    CHECK (platform_machine IN ('x86_64', 'amd64', 'aarch64', 'arm64', 'i386', 'i686'));

-- Environment hash must be 64-character hex
ALTER TABLE reproducibility_service.environment_fingerprints
    ADD CONSTRAINT chk_fingerprint_environment_hash_length
    CHECK (LENGTH(environment_hash) = 64);

-- CPU count must be positive when present
ALTER TABLE reproducibility_service.environment_fingerprints
    ADD CONSTRAINT chk_fingerprint_cpu_count_positive
    CHECK (cpu_count IS NULL OR cpu_count > 0);

-- Total memory must be positive when present
ALTER TABLE reproducibility_service.environment_fingerprints
    ADD CONSTRAINT chk_fingerprint_memory_positive
    CHECK (total_memory_mb IS NULL OR total_memory_mb > 0);

-- =============================================================================
-- Table: reproducibility_service.seed_configurations
-- =============================================================================
-- Seed configuration records for deterministic execution. Each record captures
-- the global seed, NumPy seed, PyTorch seed, custom seeds (JSONB for
-- framework-specific seeds), and a computed hash of the seed configuration.
-- Used to ensure identical random number generator state across executions.

CREATE TABLE reproducibility_service.seed_configurations (
    seed_config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id VARCHAR(255),
    global_seed BIGINT NOT NULL DEFAULT 42,
    numpy_seed BIGINT DEFAULT 42,
    torch_seed BIGINT DEFAULT 42,
    python_hash_seed BIGINT DEFAULT 0,
    custom_seeds JSONB NOT NULL DEFAULT '{}'::jsonb,
    seed_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default'
);

-- Global seed must be non-negative
ALTER TABLE reproducibility_service.seed_configurations
    ADD CONSTRAINT chk_seed_global_non_negative
    CHECK (global_seed >= 0);

-- NumPy seed must be non-negative when present
ALTER TABLE reproducibility_service.seed_configurations
    ADD CONSTRAINT chk_seed_numpy_non_negative
    CHECK (numpy_seed IS NULL OR numpy_seed >= 0);

-- Torch seed must be non-negative when present
ALTER TABLE reproducibility_service.seed_configurations
    ADD CONSTRAINT chk_seed_torch_non_negative
    CHECK (torch_seed IS NULL OR torch_seed >= 0);

-- Python hash seed must be non-negative when present
ALTER TABLE reproducibility_service.seed_configurations
    ADD CONSTRAINT chk_seed_python_hash_non_negative
    CHECK (python_hash_seed IS NULL OR python_hash_seed >= 0);

-- Seed hash must be 64-character hex
ALTER TABLE reproducibility_service.seed_configurations
    ADD CONSTRAINT chk_seed_hash_length
    CHECK (LENGTH(seed_hash) = 64);

-- =============================================================================
-- Table: reproducibility_service.version_manifests
-- =============================================================================
-- Version manifest records. Each manifest captures a snapshot of all agent
-- versions, model versions, emission factor versions, and data source versions
-- active during an execution. The manifest hash allows quick comparison of
-- version state across executions for reproducibility verification.

CREATE TABLE reproducibility_service.version_manifests (
    manifest_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_versions JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_versions JSONB NOT NULL DEFAULT '{}'::jsonb,
    factor_versions JSONB NOT NULL DEFAULT '{}'::jsonb,
    data_versions JSONB NOT NULL DEFAULT '{}'::jsonb,
    config_versions JSONB NOT NULL DEFAULT '{}'::jsonb,
    schema_versions JSONB NOT NULL DEFAULT '{}'::jsonb,
    manifest_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_by VARCHAR(255) DEFAULT 'system'
);

-- Manifest hash must be 64-character hex
ALTER TABLE reproducibility_service.version_manifests
    ADD CONSTRAINT chk_manifest_hash_length
    CHECK (LENGTH(manifest_hash) = 64);

-- =============================================================================
-- Table: reproducibility_service.drift_baselines
-- =============================================================================
-- Drift baseline records. A baseline is a reference snapshot of expected
-- output for a given execution, against which subsequent runs are compared
-- for drift detection. Each baseline has a name, description, the baseline
-- data (JSONB), a hash for integrity, and an active flag. Only one baseline
-- per name per tenant can be active at a time.

CREATE TABLE reproducibility_service.drift_baselines (
    baseline_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT DEFAULT '',
    baseline_data JSONB NOT NULL,
    baseline_hash VARCHAR(64) NOT NULL,
    drift_threshold DOUBLE PRECISION NOT NULL DEFAULT 0.01,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_by VARCHAR(255) DEFAULT 'system'
);

-- Unique constraint on name + tenant_id when active
-- (only one active baseline per name per tenant)
CREATE UNIQUE INDEX uq_drift_baseline_active_name
    ON reproducibility_service.drift_baselines (name, tenant_id)
    WHERE is_active = true;

-- Baseline hash must be 64-character hex
ALTER TABLE reproducibility_service.drift_baselines
    ADD CONSTRAINT chk_baseline_hash_length
    CHECK (LENGTH(baseline_hash) = 64);

-- Drift threshold must be positive
ALTER TABLE reproducibility_service.drift_baselines
    ADD CONSTRAINT chk_baseline_drift_threshold_positive
    CHECK (drift_threshold > 0);

-- =============================================================================
-- Table: reproducibility_service.drift_detections
-- =============================================================================
-- Drift detection records. Each record captures a comparison of an execution
-- output against a drift baseline, the severity of detected drift, the drift
-- percentage, details of the drift (JSONB with field-level breakdown), and
-- the provenance hash for integrity verification.

CREATE TABLE reproducibility_service.drift_detections (
    detection_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    baseline_id UUID NOT NULL REFERENCES reproducibility_service.drift_baselines(baseline_id) ON DELETE CASCADE,
    verification_id UUID REFERENCES reproducibility_service.verification_runs(verification_id) ON DELETE SET NULL,
    execution_id VARCHAR(255) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    drift_percentage DOUBLE PRECISION NOT NULL DEFAULT 0,
    drift_details JSONB NOT NULL DEFAULT '{}'::jsonb,
    fields_drifted TEXT[] DEFAULT '{}',
    provenance_hash VARCHAR(64) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_by VARCHAR(255) DEFAULT 'system'
);

-- Severity constraint
ALTER TABLE reproducibility_service.drift_detections
    ADD CONSTRAINT chk_drift_severity
    CHECK (severity IN ('none', 'minor', 'moderate', 'critical'));

-- Drift percentage must be non-negative
ALTER TABLE reproducibility_service.drift_detections
    ADD CONSTRAINT chk_drift_percentage_non_negative
    CHECK (drift_percentage >= 0);

-- Provenance hash must be 64-character hex
ALTER TABLE reproducibility_service.drift_detections
    ADD CONSTRAINT chk_drift_provenance_hash_length
    CHECK (LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: reproducibility_service.replay_sessions (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording replay execution sessions. Each replay
-- session attempts to re-execute a prior execution with the same inputs,
-- environment, seeds, and versions, then compares outputs for exact or
-- approximate match. Partitioned by started_at for time-series queries.
-- Retained for 30 days with compression after 3 days.

CREATE TABLE reproducibility_service.replay_sessions (
    replay_id UUID NOT NULL DEFAULT gen_random_uuid(),
    original_execution_id VARCHAR(255) NOT NULL,
    replay_execution_id VARCHAR(255),
    environment_match BOOLEAN,
    seed_match BOOLEAN,
    version_match BOOLEAN,
    output_match BOOLEAN,
    replay_status VARCHAR(20) NOT NULL,
    replay_details JSONB DEFAULT '{}'::jsonb,
    original_output_hash VARCHAR(64),
    replay_output_hash VARCHAR(64),
    discrepancies JSONB DEFAULT '[]'::jsonb,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    processing_time_ms DOUBLE PRECISION,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_by VARCHAR(255) DEFAULT 'system',
    PRIMARY KEY (replay_id, started_at)
);

-- Create hypertable partitioned by started_at
SELECT create_hypertable('reproducibility_service.replay_sessions', 'started_at', if_not_exists => TRUE);

-- Replay status constraint
ALTER TABLE reproducibility_service.replay_sessions
    ADD CONSTRAINT chk_replay_status
    CHECK (replay_status IN ('pending', 'running', 'completed', 'failed', 'cancelled'));

-- Processing time must be non-negative when present
ALTER TABLE reproducibility_service.replay_sessions
    ADD CONSTRAINT chk_replay_processing_time_positive
    CHECK (processing_time_ms IS NULL OR processing_time_ms >= 0);

-- Output hashes must be 64-character hex when present
ALTER TABLE reproducibility_service.replay_sessions
    ADD CONSTRAINT chk_replay_original_hash_length
    CHECK (original_output_hash IS NULL OR LENGTH(original_output_hash) = 64);

ALTER TABLE reproducibility_service.replay_sessions
    ADD CONSTRAINT chk_replay_replay_hash_length
    CHECK (replay_output_hash IS NULL OR LENGTH(replay_output_hash) = 64);

-- =============================================================================
-- Table: reproducibility_service.reproducibility_audit_log (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording comprehensive audit events for all
-- reproducibility operations. Each event captures the entity being operated
-- on, event type, action, data hashes (current, previous, chain), details
-- (JSONB), user, source IP, tenant, and timestamp. Partitioned by timestamp
-- for time-series queries. Retained for 365 days with compression after 30
-- days.

CREATE TABLE reproducibility_service.reproducibility_audit_log (
    audit_id UUID NOT NULL DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    action VARCHAR(50) NOT NULL,
    data_hash VARCHAR(64),
    previous_hash VARCHAR(64),
    chain_hash VARCHAR(64),
    details JSONB DEFAULT '{}'::jsonb,
    user_id VARCHAR(255) DEFAULT 'system',
    source_ip VARCHAR(45),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    PRIMARY KEY (audit_id, timestamp)
);

-- Create hypertable partitioned by timestamp
SELECT create_hypertable('reproducibility_service.reproducibility_audit_log', 'timestamp', if_not_exists => TRUE);

-- Event type constraint
ALTER TABLE reproducibility_service.reproducibility_audit_log
    ADD CONSTRAINT chk_audit_event_type
    CHECK (event_type IN (
        'verification_created', 'verification_completed', 'verification_failed',
        'artifact_hashed', 'artifact_compared', 'artifact_mismatch',
        'environment_captured', 'environment_compared', 'environment_mismatch',
        'seed_configured', 'seed_compared', 'seed_mismatch',
        'manifest_created', 'manifest_compared', 'manifest_mismatch',
        'baseline_created', 'baseline_updated', 'baseline_deactivated',
        'drift_detected', 'drift_resolved', 'drift_acknowledged',
        'replay_started', 'replay_completed', 'replay_failed',
        'non_determinism_detected', 'non_determinism_resolved',
        'admin_action', 'cache_invalidated', 'config_changed'
    ));

-- Entity type constraint
ALTER TABLE reproducibility_service.reproducibility_audit_log
    ADD CONSTRAINT chk_audit_entity_type
    CHECK (entity_type IN (
        'verification_run', 'artifact_hash', 'environment_fingerprint',
        'seed_configuration', 'version_manifest', 'drift_baseline',
        'drift_detection', 'replay_session', 'system'
    ));

-- Action constraint
ALTER TABLE reproducibility_service.reproducibility_audit_log
    ADD CONSTRAINT chk_audit_action
    CHECK (action IN (
        'create', 'update', 'delete', 'compare', 'verify',
        'replay', 'detect', 'resolve', 'acknowledge',
        'activate', 'deactivate', 'system', 'admin'
    ));

-- Hash fields must be 64-character hex when present
ALTER TABLE reproducibility_service.reproducibility_audit_log
    ADD CONSTRAINT chk_audit_data_hash_length
    CHECK (data_hash IS NULL OR LENGTH(data_hash) = 64);

ALTER TABLE reproducibility_service.reproducibility_audit_log
    ADD CONSTRAINT chk_audit_previous_hash_length
    CHECK (previous_hash IS NULL OR LENGTH(previous_hash) = 64);

ALTER TABLE reproducibility_service.reproducibility_audit_log
    ADD CONSTRAINT chk_audit_chain_hash_length
    CHECK (chain_hash IS NULL OR LENGTH(chain_hash) = 64);

-- =============================================================================
-- Continuous Aggregate: reproducibility_service.hourly_verification_stats
-- =============================================================================
-- Precomputed hourly verification run statistics by status for dashboard
-- queries, trend analysis, and SLI tracking. Shows the number of
-- verifications per status, average processing time, reproducible count,
-- and non-reproducible count per hour.

CREATE MATERIALIZED VIEW reproducibility_service.hourly_verification_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at) AS bucket,
    status,
    tenant_id,
    COUNT(*) AS verification_count,
    AVG(processing_time_ms) AS avg_processing_time_ms,
    MAX(processing_time_ms) AS max_processing_time_ms,
    SUM(CASE WHEN is_reproducible THEN 1 ELSE 0 END) AS reproducible_count,
    SUM(CASE WHEN NOT is_reproducible THEN 1 ELSE 0 END) AS non_reproducible_count
FROM reproducibility_service.verification_runs
WHERE created_at IS NOT NULL
GROUP BY bucket, status, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('reproducibility_service.hourly_verification_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Continuous Aggregate: reproducibility_service.hourly_audit_stats
-- =============================================================================
-- Precomputed hourly counts of audit events by event type and entity type
-- for compliance reporting, dashboard queries, and long-term trend analysis.

CREATE MATERIALIZED VIEW reproducibility_service.hourly_audit_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS bucket,
    event_type,
    entity_type,
    tenant_id,
    COUNT(*) AS event_count,
    COUNT(DISTINCT entity_id) AS unique_entities,
    COUNT(DISTINCT user_id) AS unique_users
FROM reproducibility_service.reproducibility_audit_log
WHERE timestamp IS NOT NULL
GROUP BY bucket, event_type, entity_type, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 30 minutes, covering the last 2 days
SELECT add_continuous_aggregate_policy('reproducibility_service.hourly_audit_stats',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- verification_runs indexes
CREATE INDEX idx_vr_execution_id ON reproducibility_service.verification_runs(execution_id);
CREATE INDEX idx_vr_status ON reproducibility_service.verification_runs(status);
CREATE INDEX idx_vr_is_reproducible ON reproducibility_service.verification_runs(is_reproducible);
CREATE INDEX idx_vr_environment_fingerprint ON reproducibility_service.verification_runs(environment_fingerprint_id);
CREATE INDEX idx_vr_seed_config ON reproducibility_service.verification_runs(seed_config_id);
CREATE INDEX idx_vr_version_manifest ON reproducibility_service.verification_runs(version_manifest_id);
CREATE INDEX idx_vr_drift_baseline ON reproducibility_service.verification_runs(drift_baseline_id);
CREATE INDEX idx_vr_input_hash ON reproducibility_service.verification_runs(input_hash);
CREATE INDEX idx_vr_output_hash ON reproducibility_service.verification_runs(output_hash);
CREATE INDEX idx_vr_provenance_hash ON reproducibility_service.verification_runs(provenance_hash);
CREATE INDEX idx_vr_created_at ON reproducibility_service.verification_runs(created_at DESC);
CREATE INDEX idx_vr_tenant ON reproducibility_service.verification_runs(tenant_id);
CREATE INDEX idx_vr_tenant_status ON reproducibility_service.verification_runs(tenant_id, status);
CREATE INDEX idx_vr_tenant_created ON reproducibility_service.verification_runs(tenant_id, created_at DESC);
CREATE INDEX idx_vr_checks ON reproducibility_service.verification_runs USING GIN (checks);
CREATE INDEX idx_vr_non_determinism_sources ON reproducibility_service.verification_runs USING GIN (non_determinism_sources);
CREATE INDEX idx_vr_non_determinism_details ON reproducibility_service.verification_runs USING GIN (non_determinism_details);

-- artifact_hashes indexes
CREATE INDEX idx_ah_artifact_id ON reproducibility_service.artifact_hashes(artifact_id);
CREATE INDEX idx_ah_artifact_type ON reproducibility_service.artifact_hashes(artifact_type);
CREATE INDEX idx_ah_data_hash ON reproducibility_service.artifact_hashes(data_hash);
CREATE INDEX idx_ah_algorithm ON reproducibility_service.artifact_hashes(algorithm);
CREATE INDEX idx_ah_provenance_hash ON reproducibility_service.artifact_hashes(provenance_hash);
CREATE INDEX idx_ah_computed_at ON reproducibility_service.artifact_hashes(computed_at DESC);
CREATE INDEX idx_ah_tenant ON reproducibility_service.artifact_hashes(tenant_id);
CREATE INDEX idx_ah_artifact_tenant ON reproducibility_service.artifact_hashes(artifact_id, tenant_id);
CREATE INDEX idx_ah_metadata ON reproducibility_service.artifact_hashes USING GIN (metadata);

-- environment_fingerprints indexes
CREATE INDEX idx_ef_python_version ON reproducibility_service.environment_fingerprints(python_version);
CREATE INDEX idx_ef_platform_system ON reproducibility_service.environment_fingerprints(platform_system);
CREATE INDEX idx_ef_platform_machine ON reproducibility_service.environment_fingerprints(platform_machine);
CREATE INDEX idx_ef_greenlang_version ON reproducibility_service.environment_fingerprints(greenlang_version);
CREATE INDEX idx_ef_environment_hash ON reproducibility_service.environment_fingerprints(environment_hash);
CREATE INDEX idx_ef_captured_at ON reproducibility_service.environment_fingerprints(captured_at DESC);
CREATE INDEX idx_ef_tenant ON reproducibility_service.environment_fingerprints(tenant_id);
CREATE INDEX idx_ef_dependency_versions ON reproducibility_service.environment_fingerprints USING GIN (dependency_versions);
CREATE INDEX idx_ef_gpu_info ON reproducibility_service.environment_fingerprints USING GIN (gpu_info);

-- seed_configurations indexes
CREATE INDEX idx_sc_execution_id ON reproducibility_service.seed_configurations(execution_id);
CREATE INDEX idx_sc_global_seed ON reproducibility_service.seed_configurations(global_seed);
CREATE INDEX idx_sc_seed_hash ON reproducibility_service.seed_configurations(seed_hash);
CREATE INDEX idx_sc_created_at ON reproducibility_service.seed_configurations(created_at DESC);
CREATE INDEX idx_sc_tenant ON reproducibility_service.seed_configurations(tenant_id);
CREATE INDEX idx_sc_custom_seeds ON reproducibility_service.seed_configurations USING GIN (custom_seeds);

-- version_manifests indexes
CREATE INDEX idx_vm_manifest_hash ON reproducibility_service.version_manifests(manifest_hash);
CREATE INDEX idx_vm_created_at ON reproducibility_service.version_manifests(created_at DESC);
CREATE INDEX idx_vm_tenant ON reproducibility_service.version_manifests(tenant_id);
CREATE INDEX idx_vm_agent_versions ON reproducibility_service.version_manifests USING GIN (agent_versions);
CREATE INDEX idx_vm_model_versions ON reproducibility_service.version_manifests USING GIN (model_versions);
CREATE INDEX idx_vm_factor_versions ON reproducibility_service.version_manifests USING GIN (factor_versions);
CREATE INDEX idx_vm_data_versions ON reproducibility_service.version_manifests USING GIN (data_versions);
CREATE INDEX idx_vm_config_versions ON reproducibility_service.version_manifests USING GIN (config_versions);
CREATE INDEX idx_vm_schema_versions ON reproducibility_service.version_manifests USING GIN (schema_versions);

-- drift_baselines indexes
CREATE INDEX idx_db_name ON reproducibility_service.drift_baselines(name);
CREATE INDEX idx_db_is_active ON reproducibility_service.drift_baselines(is_active);
CREATE INDEX idx_db_baseline_hash ON reproducibility_service.drift_baselines(baseline_hash);
CREATE INDEX idx_db_created_at ON reproducibility_service.drift_baselines(created_at DESC);
CREATE INDEX idx_db_updated_at ON reproducibility_service.drift_baselines(updated_at DESC);
CREATE INDEX idx_db_tenant ON reproducibility_service.drift_baselines(tenant_id);
CREATE INDEX idx_db_active_tenant ON reproducibility_service.drift_baselines(tenant_id, is_active);
CREATE INDEX idx_db_baseline_data ON reproducibility_service.drift_baselines USING GIN (baseline_data);

-- drift_detections indexes
CREATE INDEX idx_dd_baseline ON reproducibility_service.drift_detections(baseline_id);
CREATE INDEX idx_dd_verification ON reproducibility_service.drift_detections(verification_id);
CREATE INDEX idx_dd_execution_id ON reproducibility_service.drift_detections(execution_id);
CREATE INDEX idx_dd_severity ON reproducibility_service.drift_detections(severity);
CREATE INDEX idx_dd_detected_at ON reproducibility_service.drift_detections(detected_at DESC);
CREATE INDEX idx_dd_tenant ON reproducibility_service.drift_detections(tenant_id);
CREATE INDEX idx_dd_tenant_severity ON reproducibility_service.drift_detections(tenant_id, severity);
CREATE INDEX idx_dd_drift_details ON reproducibility_service.drift_detections USING GIN (drift_details);
CREATE INDEX idx_dd_fields_drifted ON reproducibility_service.drift_detections USING GIN (fields_drifted);

-- replay_sessions indexes (hypertable-aware)
CREATE INDEX idx_rs_original_execution ON reproducibility_service.replay_sessions(original_execution_id, started_at DESC);
CREATE INDEX idx_rs_replay_execution ON reproducibility_service.replay_sessions(replay_execution_id, started_at DESC);
CREATE INDEX idx_rs_replay_status ON reproducibility_service.replay_sessions(replay_status, started_at DESC);
CREATE INDEX idx_rs_output_match ON reproducibility_service.replay_sessions(output_match, started_at DESC);
CREATE INDEX idx_rs_tenant ON reproducibility_service.replay_sessions(tenant_id, started_at DESC);
CREATE INDEX idx_rs_replay_details ON reproducibility_service.replay_sessions USING GIN (replay_details);
CREATE INDEX idx_rs_discrepancies ON reproducibility_service.replay_sessions USING GIN (discrepancies);

-- reproducibility_audit_log indexes (hypertable-aware)
CREATE INDEX idx_ral_event_type ON reproducibility_service.reproducibility_audit_log(event_type, timestamp DESC);
CREATE INDEX idx_ral_entity_type ON reproducibility_service.reproducibility_audit_log(entity_type, timestamp DESC);
CREATE INDEX idx_ral_entity_id ON reproducibility_service.reproducibility_audit_log(entity_id, timestamp DESC);
CREATE INDEX idx_ral_action ON reproducibility_service.reproducibility_audit_log(action, timestamp DESC);
CREATE INDEX idx_ral_user ON reproducibility_service.reproducibility_audit_log(user_id, timestamp DESC);
CREATE INDEX idx_ral_tenant ON reproducibility_service.reproducibility_audit_log(tenant_id, timestamp DESC);
CREATE INDEX idx_ral_data_hash ON reproducibility_service.reproducibility_audit_log(data_hash);
CREATE INDEX idx_ral_chain_hash ON reproducibility_service.reproducibility_audit_log(chain_hash);
CREATE INDEX idx_ral_details ON reproducibility_service.reproducibility_audit_log USING GIN (details);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE reproducibility_service.verification_runs ENABLE ROW LEVEL SECURITY;
CREATE POLICY vr_tenant_read ON reproducibility_service.verification_runs
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY vr_tenant_write ON reproducibility_service.verification_runs
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE reproducibility_service.artifact_hashes ENABLE ROW LEVEL SECURITY;
CREATE POLICY ah_tenant_read ON reproducibility_service.artifact_hashes
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ah_tenant_write ON reproducibility_service.artifact_hashes
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE reproducibility_service.environment_fingerprints ENABLE ROW LEVEL SECURITY;
CREATE POLICY ef_tenant_read ON reproducibility_service.environment_fingerprints
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ef_tenant_write ON reproducibility_service.environment_fingerprints
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE reproducibility_service.seed_configurations ENABLE ROW LEVEL SECURITY;
CREATE POLICY sc_tenant_read ON reproducibility_service.seed_configurations
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY sc_tenant_write ON reproducibility_service.seed_configurations
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE reproducibility_service.version_manifests ENABLE ROW LEVEL SECURITY;
CREATE POLICY vm_tenant_read ON reproducibility_service.version_manifests
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY vm_tenant_write ON reproducibility_service.version_manifests
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE reproducibility_service.drift_baselines ENABLE ROW LEVEL SECURITY;
CREATE POLICY dbl_tenant_read ON reproducibility_service.drift_baselines
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY dbl_tenant_write ON reproducibility_service.drift_baselines
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE reproducibility_service.drift_detections ENABLE ROW LEVEL SECURITY;
CREATE POLICY dd_tenant_read ON reproducibility_service.drift_detections
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY dd_tenant_write ON reproducibility_service.drift_detections
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE reproducibility_service.replay_sessions ENABLE ROW LEVEL SECURITY;
CREATE POLICY rs_tenant_read ON reproducibility_service.replay_sessions
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY rs_tenant_write ON reproducibility_service.replay_sessions
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE reproducibility_service.reproducibility_audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY ral_tenant_read ON reproducibility_service.reproducibility_audit_log
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ral_tenant_write ON reproducibility_service.reproducibility_audit_log
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA reproducibility_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA reproducibility_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA reproducibility_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON reproducibility_service.hourly_verification_stats TO greenlang_app;
GRANT SELECT ON reproducibility_service.hourly_audit_stats TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA reproducibility_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA reproducibility_service TO greenlang_readonly;
GRANT SELECT ON reproducibility_service.hourly_verification_stats TO greenlang_readonly;
GRANT SELECT ON reproducibility_service.hourly_audit_stats TO greenlang_readonly;

-- Add reproducibility service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'reproducibility:verifications:read', 'reproducibility', 'verifications_read', 'View verification run records and results'),
    (gen_random_uuid(), 'reproducibility:verifications:write', 'reproducibility', 'verifications_write', 'Create and update verification runs'),
    (gen_random_uuid(), 'reproducibility:artifacts:read', 'reproducibility', 'artifacts_read', 'View artifact hash records'),
    (gen_random_uuid(), 'reproducibility:artifacts:write', 'reproducibility', 'artifacts_write', 'Compute and store artifact hashes'),
    (gen_random_uuid(), 'reproducibility:environments:read', 'reproducibility', 'environments_read', 'View environment fingerprint records'),
    (gen_random_uuid(), 'reproducibility:environments:write', 'reproducibility', 'environments_write', 'Capture environment fingerprints'),
    (gen_random_uuid(), 'reproducibility:seeds:read', 'reproducibility', 'seeds_read', 'View seed configuration records'),
    (gen_random_uuid(), 'reproducibility:seeds:write', 'reproducibility', 'seeds_write', 'Create and manage seed configurations'),
    (gen_random_uuid(), 'reproducibility:manifests:read', 'reproducibility', 'manifests_read', 'View version manifest records'),
    (gen_random_uuid(), 'reproducibility:manifests:write', 'reproducibility', 'manifests_write', 'Create version manifests'),
    (gen_random_uuid(), 'reproducibility:drift:read', 'reproducibility', 'drift_read', 'View drift baselines and detections'),
    (gen_random_uuid(), 'reproducibility:drift:write', 'reproducibility', 'drift_write', 'Create and manage drift baselines and detections'),
    (gen_random_uuid(), 'reproducibility:audit:read', 'reproducibility', 'audit_read', 'View reproducibility audit event log'),
    (gen_random_uuid(), 'reproducibility:admin', 'reproducibility', 'admin', 'Reproducibility service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep replay sessions for 30 days
SELECT add_retention_policy('reproducibility_service.replay_sessions', INTERVAL '30 days');

-- Keep audit events for 365 days
SELECT add_retention_policy('reproducibility_service.reproducibility_audit_log', INTERVAL '365 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on replay_sessions after 3 days
ALTER TABLE reproducibility_service.replay_sessions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'started_at DESC'
);

SELECT add_compression_policy('reproducibility_service.replay_sessions', INTERVAL '3 days');

-- Enable compression on reproducibility_audit_log after 30 days
ALTER TABLE reproducibility_service.reproducibility_audit_log SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('reproducibility_service.reproducibility_audit_log', INTERVAL '30 days');

-- =============================================================================
-- Seed: Register the Reproducibility Agent (GL-FOUND-X-008) in Agent Registry
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES
('GL-FOUND-X-008', 'Reproducibility Agent',
 'Ensures deterministic, reproducible execution across GreenLang Climate OS. Provides artifact hashing (SHA-256/SHA-512/BLAKE2b), determinism verification with configurable tolerances, drift detection against baselines, replay mode for exact re-execution, environment fingerprinting, seed management for RNG state, and version pinning for all agents, models, factors, and data sources.',
 1, 'sync', true, true, 20, '1.0.0', false,
 'GreenLang Platform Team', 'https://docs.greenlang.ai/agents/reproducibility', true, 'default')
ON CONFLICT (agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Agent Version for Reproducibility Agent
-- =============================================================================

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES
('GL-FOUND-X-008', '1.0.0',
 '{"cpu_request": "100m", "cpu_limit": "500m", "memory_request": "256Mi", "memory_limit": "512Mi", "gpu": false}'::jsonb,
 '{"image": "greenlang/reproducibility-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"foundation", "reproducibility", "determinism", "verification", "drift"}',
 '{"cross-sector"}',
 'b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9')
ON CONFLICT (agent_id, version) DO NOTHING;

-- =============================================================================
-- Seed: Agent Capabilities for Reproducibility Agent
-- =============================================================================

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES

('GL-FOUND-X-008', '1.0.0', 'artifact_hashing', 'computation',
 'Compute SHA-256/SHA-512/BLAKE2b hashes of input/output artifacts with optional normalization for floating-point data',
 '{"artifact_data", "artifact_type"}', '{"artifact_hash", "provenance_hash"}',
 '{"algorithms": ["sha256", "sha384", "sha512", "blake2b", "blake3"], "normalization": true, "floating_point_precision": 15}'::jsonb),

('GL-FOUND-X-008', '1.0.0', 'determinism_verification', 'validation',
 'Verify execution determinism by comparing input/output hashes, environment fingerprints, seed configurations, and version manifests',
 '{"execution_id", "verification_context"}', '{"verification_result", "non_determinism_sources"}',
 '{"absolute_tolerance": 1e-9, "relative_tolerance": 1e-6, "check_environment": true, "check_seeds": true, "check_versions": true}'::jsonb),

('GL-FOUND-X-008', '1.0.0', 'drift_detection', 'analysis',
 'Detect output drift against established baselines with severity classification (none, minor, moderate, critical)',
 '{"execution_output", "baseline_id"}', '{"drift_result", "drift_details", "severity"}',
 '{"severity_thresholds": {"minor": 0.001, "moderate": 0.01, "critical": 0.05}, "field_level_comparison": true}'::jsonb),

('GL-FOUND-X-008', '1.0.0', 'replay_execution', 'orchestration',
 'Replay a previous execution with identical inputs, environment, seeds, and versions to verify reproducibility',
 '{"original_execution_id"}', '{"replay_result", "output_match", "discrepancies"}',
 '{"match_environment": true, "match_seeds": true, "match_versions": true, "timeout_seconds": 600}'::jsonb),

('GL-FOUND-X-008', '1.0.0', 'environment_fingerprinting', 'computation',
 'Capture and compare execution environment fingerprints including Python version, platform, dependencies, and hardware',
 '{"environment_context"}', '{"fingerprint", "environment_hash"}',
 '{"capture_dependencies": true, "capture_hardware": true, "redact_secrets": true}'::jsonb),

('GL-FOUND-X-008', '1.0.0', 'seed_management', 'computation',
 'Configure and verify seed state for deterministic random number generation across Python, NumPy, PyTorch, and custom frameworks',
 '{"seed_config"}', '{"seed_state", "seed_hash"}',
 '{"default_global_seed": 42, "frameworks": ["python", "numpy", "torch", "tensorflow"]}'::jsonb)

ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: Agent Dependencies for Reproducibility Agent
-- =============================================================================

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES

-- Reproducibility depends on Schema Compiler for input validation
('GL-FOUND-X-008', 'GL-FOUND-X-002', '>=1.0.0', false,
 'Verification requests and artifact data are validated against JSON Schema definitions'),

-- Reproducibility depends on Registry for agent version tracking
('GL-FOUND-X-008', 'GL-FOUND-X-007', '>=1.0.0', false,
 'Version manifests are resolved against the agent registry for agent version verification'),

-- Reproducibility optionally uses Citations for provenance
('GL-FOUND-X-008', 'GL-FOUND-X-005', '>=1.0.0', true,
 'Provenance chain hashes are tracked via the citation service for audit trail'),

-- Reproducibility optionally uses Orchestrator for replay
('GL-FOUND-X-008', 'GL-FOUND-X-001', '>=1.0.0', true,
 'Replay mode uses the orchestrator to re-execute DAGs with identical configuration')

ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Service Catalog Entry for Reproducibility Agent
-- =============================================================================

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES
('GL-FOUND-X-008', 'Reproducibility Agent',
 'Determinism verification, artifact hashing, drift detection, replay mode, environment fingerprinting, and seed management for reproducible climate calculations.',
 'foundation', 'active', 'default')
ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA reproducibility_service IS 'Reproducibility Agent for GreenLang Climate OS (AGENT-FOUND-008) - determinism verification, artifact hashing, drift detection, replay mode, environment fingerprinting, seed management, and version pinning for reproducible climate calculations';
COMMENT ON TABLE reproducibility_service.verification_runs IS 'Core verification run records with execution hashes, environment/seed/version references, reproducibility result, non-determinism sources, and provenance hash';
COMMENT ON TABLE reproducibility_service.artifact_hashes IS 'Computed hash records for input/output artifacts with algorithm, normalization flag, metadata, and provenance hash';
COMMENT ON TABLE reproducibility_service.environment_fingerprints IS 'Environment fingerprint snapshots capturing Python version, platform, dependencies, hardware, and computed environment hash';
COMMENT ON TABLE reproducibility_service.seed_configurations IS 'Seed configuration records for deterministic RNG state across Python, NumPy, PyTorch, and custom frameworks';
COMMENT ON TABLE reproducibility_service.version_manifests IS 'Version manifest snapshots of all agent, model, factor, data, config, and schema versions active during an execution';
COMMENT ON TABLE reproducibility_service.drift_baselines IS 'Drift detection baselines with reference data, threshold, and active flag for output drift comparison';
COMMENT ON TABLE reproducibility_service.drift_detections IS 'Drift detection records with severity, percentage, field-level details, and provenance hash';
COMMENT ON TABLE reproducibility_service.replay_sessions IS 'TimescaleDB hypertable: replay execution sessions comparing original and replay outputs for exact or approximate match';
COMMENT ON TABLE reproducibility_service.reproducibility_audit_log IS 'TimescaleDB hypertable: comprehensive audit events for all reproducibility operations with hash chain integrity';
COMMENT ON MATERIALIZED VIEW reproducibility_service.hourly_verification_stats IS 'Continuous aggregate: hourly verification run statistics by status for dashboard queries, trend analysis, and SLI tracking';
COMMENT ON MATERIALIZED VIEW reproducibility_service.hourly_audit_stats IS 'Continuous aggregate: hourly audit event counts by event type and entity type for compliance reporting and trend analysis';

COMMENT ON COLUMN reproducibility_service.verification_runs.status IS 'Verification status: pass, fail, warning, skipped';
COMMENT ON COLUMN reproducibility_service.verification_runs.is_reproducible IS 'Whether the execution is confirmed reproducible (all checks passed within tolerance)';
COMMENT ON COLUMN reproducibility_service.verification_runs.non_determinism_sources IS 'Array of detected non-determinism sources: floating_point, random_state, parallel_execution, system_time, external_api, file_system, network, memory_allocation, gpu_computation, hash_ordering';
COMMENT ON COLUMN reproducibility_service.verification_runs.absolute_tolerance IS 'Absolute tolerance for floating-point comparison (default 1e-9)';
COMMENT ON COLUMN reproducibility_service.verification_runs.relative_tolerance IS 'Relative tolerance for floating-point comparison (default 1e-6)';
COMMENT ON COLUMN reproducibility_service.verification_runs.provenance_hash IS 'SHA-256 hash of the verification content for integrity verification';

COMMENT ON COLUMN reproducibility_service.artifact_hashes.artifact_type IS 'Artifact type: input_data, output_data, model_weights, configuration, emission_factor, calculation_result, report, intermediate, checkpoint, schema, reference_data, seed_state';
COMMENT ON COLUMN reproducibility_service.artifact_hashes.algorithm IS 'Hash algorithm: sha256, sha384, sha512, blake2b, blake3';
COMMENT ON COLUMN reproducibility_service.artifact_hashes.normalization_applied IS 'Whether floating-point normalization was applied before hashing';

COMMENT ON COLUMN reproducibility_service.environment_fingerprints.environment_hash IS 'SHA-256 hash of the entire environment fingerprint for quick comparison';
COMMENT ON COLUMN reproducibility_service.environment_fingerprints.dependency_versions IS 'JSONB map of package name to version for all Python dependencies';

COMMENT ON COLUMN reproducibility_service.seed_configurations.global_seed IS 'Global seed for all random number generators (default 42)';
COMMENT ON COLUMN reproducibility_service.seed_configurations.seed_hash IS 'SHA-256 hash of the seed configuration for quick comparison';

COMMENT ON COLUMN reproducibility_service.version_manifests.manifest_hash IS 'SHA-256 hash of the entire version manifest for quick comparison';

COMMENT ON COLUMN reproducibility_service.drift_baselines.drift_threshold IS 'Maximum allowed drift percentage before triggering an alert (default 0.01 = 1%)';

COMMENT ON COLUMN reproducibility_service.drift_detections.severity IS 'Drift severity: none, minor, moderate, critical';
COMMENT ON COLUMN reproducibility_service.drift_detections.drift_percentage IS 'Overall drift percentage as a decimal (0.01 = 1%)';

COMMENT ON COLUMN reproducibility_service.replay_sessions.replay_status IS 'Replay status: pending, running, completed, failed, cancelled';
COMMENT ON COLUMN reproducibility_service.replay_sessions.output_match IS 'Whether the replay output matches the original output within tolerance';

COMMENT ON COLUMN reproducibility_service.reproducibility_audit_log.event_type IS 'Audit event type: verification_created, artifact_hashed, drift_detected, replay_started, etc.';
COMMENT ON COLUMN reproducibility_service.reproducibility_audit_log.entity_type IS 'Entity type: verification_run, artifact_hash, environment_fingerprint, seed_configuration, version_manifest, drift_baseline, drift_detection, replay_session, system';
COMMENT ON COLUMN reproducibility_service.reproducibility_audit_log.chain_hash IS 'SHA-256 hash chain linking this event to the previous event for tamper detection';
