-- =============================================================================
-- GreenLang Climate OS - DAG Orchestrator Service Schema
-- =============================================================================
-- Migration: V021
-- Component: AGENT-FOUND-001 GreenLang Orchestrator (DAG Execution Engine)
-- Description: Creates orchestrator schema with DAG workflow definitions,
--              execution records, node traces (hypertable), checkpoints,
--              execution provenance chain, hourly node trace continuous
--              aggregate, and sample DAG workflow seed data.
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS orchestrator;

-- =============================================================================
-- Table: orchestrator.dag_workflows
-- =============================================================================
-- Core DAG workflow definitions. Each row represents a reusable DAG workflow
-- template with its full graph definition stored as JSONB, a SHA-256 hash
-- for integrity, and versioning for change tracking. Multi-tenant via
-- tenant_id with Row-Level Security.

CREATE TABLE orchestrator.dag_workflows (
    dag_id VARCHAR(128) PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    description TEXT,
    version VARCHAR(32) NOT NULL DEFAULT '1.0.0',
    definition JSONB NOT NULL,
    hash VARCHAR(64) NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    max_parallel_nodes INTEGER NOT NULL DEFAULT 10,
    default_retry_policy JSONB NOT NULL DEFAULT '{"max_retries": 2, "strategy": "exponential", "base_delay": 1.0, "max_delay": 60.0, "jitter": true}',
    default_timeout_policy JSONB NOT NULL DEFAULT '{"timeout_seconds": 60, "on_timeout": "fail"}',
    on_failure VARCHAR(32) NOT NULL DEFAULT 'fail_fast',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(128),
    tenant_id VARCHAR(128)
);

-- On-failure strategy constraint
ALTER TABLE orchestrator.dag_workflows
    ADD CONSTRAINT chk_dag_on_failure
    CHECK (on_failure IN ('fail_fast', 'continue', 'compensate'));

-- Max parallel nodes must be positive
ALTER TABLE orchestrator.dag_workflows
    ADD CONSTRAINT chk_dag_max_parallel_positive
    CHECK (max_parallel_nodes > 0 AND max_parallel_nodes <= 100);

-- =============================================================================
-- Table: orchestrator.dag_executions
-- =============================================================================
-- Tracks each invocation of a DAG workflow. Stores input data, topology
-- snapshot, execution status, timing, error details, and the final
-- provenance chain hash for deterministic audit. Multi-tenant.

CREATE TABLE orchestrator.dag_executions (
    execution_id VARCHAR(128) PRIMARY KEY,
    dag_id VARCHAR(128) NOT NULL REFERENCES orchestrator.dag_workflows(dag_id) ON DELETE CASCADE,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    input_data JSONB,
    output_data JSONB,
    topology_levels JSONB,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error TEXT,
    provenance_chain_hash VARCHAR(64),
    deterministic_mode BOOLEAN NOT NULL DEFAULT TRUE,
    checkpoint_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    total_nodes INTEGER NOT NULL DEFAULT 0,
    completed_nodes INTEGER NOT NULL DEFAULT 0,
    failed_nodes INTEGER NOT NULL DEFAULT 0,
    skipped_nodes INTEGER NOT NULL DEFAULT 0,
    tenant_id VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Execution status constraint
ALTER TABLE orchestrator.dag_executions
    ADD CONSTRAINT chk_execution_status
    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'paused'));

-- Completed nodes must not exceed total
ALTER TABLE orchestrator.dag_executions
    ADD CONSTRAINT chk_execution_node_counts
    CHECK (completed_nodes + failed_nodes + skipped_nodes <= total_nodes);

-- =============================================================================
-- Table: orchestrator.node_traces
-- =============================================================================
-- TimescaleDB hypertable recording per-node execution traces. Each row
-- captures a single node execution attempt with status, input/output hashes,
-- duration, retry count, and provenance hash. Partitioned by started_at
-- for efficient time-series queries.

CREATE TABLE orchestrator.node_traces (
    trace_id VARCHAR(128) NOT NULL,
    execution_id VARCHAR(128) NOT NULL REFERENCES orchestrator.dag_executions(execution_id) ON DELETE CASCADE,
    node_id VARCHAR(128) NOT NULL,
    status VARCHAR(32) NOT NULL,
    input_hash VARCHAR(64),
    output_hash VARCHAR(64),
    duration_ms DOUBLE PRECISION,
    attempt_count INTEGER NOT NULL DEFAULT 1,
    level_index INTEGER,
    error TEXT,
    provenance_hash VARCHAR(64),
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    tenant_id VARCHAR(128)
);

-- Create hypertable partitioned by started_at
SELECT create_hypertable('orchestrator.node_traces', 'started_at', if_not_exists => TRUE);

-- Node trace status constraint
ALTER TABLE orchestrator.node_traces
    ADD CONSTRAINT chk_node_trace_status
    CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped', 'timed_out', 'retrying'));

-- Attempt count must be positive
ALTER TABLE orchestrator.node_traces
    ADD CONSTRAINT chk_node_trace_attempt_positive
    CHECK (attempt_count > 0);

-- =============================================================================
-- Table: orchestrator.dag_checkpoints
-- =============================================================================
-- DAG-aware checkpoint storage. Each row captures the state of a single
-- node after execution, including its outputs and output hash for
-- integrity verification on resume. Used to skip completed nodes when
-- resuming from a failure point.

CREATE TABLE orchestrator.dag_checkpoints (
    checkpoint_id VARCHAR(128) PRIMARY KEY,
    execution_id VARCHAR(128) NOT NULL REFERENCES orchestrator.dag_executions(execution_id) ON DELETE CASCADE,
    node_id VARCHAR(128) NOT NULL,
    status VARCHAR(32) NOT NULL,
    outputs JSONB,
    output_hash VARCHAR(64),
    attempt_count INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(128),
    UNIQUE (execution_id, node_id)
);

-- Checkpoint status constraint
ALTER TABLE orchestrator.dag_checkpoints
    ADD CONSTRAINT chk_checkpoint_status
    CHECK (status IN ('completed', 'failed', 'skipped', 'timed_out'));

-- =============================================================================
-- Table: orchestrator.execution_provenance
-- =============================================================================
-- Cryptographic provenance chain for regulatory audit. Each row records
-- a single node's provenance: input/output hashes, duration, attempt count,
-- parent provenance hashes (predecessor nodes), and a chain hash linking
-- this record to its predecessors via SHA-256.

CREATE TABLE orchestrator.execution_provenance (
    provenance_id VARCHAR(128) PRIMARY KEY,
    execution_id VARCHAR(128) NOT NULL REFERENCES orchestrator.dag_executions(execution_id) ON DELETE CASCADE,
    node_id VARCHAR(128) NOT NULL,
    input_hash VARCHAR(64) NOT NULL,
    output_hash VARCHAR(64) NOT NULL,
    duration_ms DOUBLE PRECISION NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 1,
    parent_hashes JSONB NOT NULL DEFAULT '[]',
    chain_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(128)
);

-- =============================================================================
-- Continuous Aggregate: orchestrator.hourly_node_summaries
-- =============================================================================
-- Precomputed hourly node execution summaries for efficient dashboard
-- queries. Aggregates node trace data into per-execution hourly statistics
-- including node count, average/min/max duration, success rate, and
-- retry count.

CREATE MATERIALIZED VIEW orchestrator.hourly_node_summaries
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', started_at) AS bucket,
    execution_id,
    COUNT(*) AS node_count,
    AVG(duration_ms) AS avg_duration_ms,
    MIN(duration_ms) AS min_duration_ms,
    MAX(duration_ms) AS max_duration_ms,
    AVG(CASE WHEN status = 'completed' THEN 1.0 ELSE 0.0 END) * 100 AS success_rate_percent,
    SUM(attempt_count - 1) AS total_retries,
    COUNT(CASE WHEN status = 'timed_out' THEN 1 END) AS timeout_count
FROM orchestrator.node_traces
WHERE started_at IS NOT NULL
GROUP BY bucket, execution_id
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('orchestrator.hourly_node_summaries',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- dag_workflows indexes
CREATE INDEX idx_dag_workflows_name ON orchestrator.dag_workflows(name);
CREATE INDEX idx_dag_workflows_tenant ON orchestrator.dag_workflows(tenant_id);
CREATE INDEX idx_dag_workflows_enabled ON orchestrator.dag_workflows(enabled) WHERE enabled = true;
CREATE INDEX idx_dag_workflows_created_at ON orchestrator.dag_workflows(created_at DESC);
CREATE INDEX idx_dag_workflows_definition ON orchestrator.dag_workflows USING GIN (definition);

-- dag_executions indexes
CREATE INDEX idx_dag_executions_dag_id ON orchestrator.dag_executions(dag_id);
CREATE INDEX idx_dag_executions_status ON orchestrator.dag_executions(status);
CREATE INDEX idx_dag_executions_tenant ON orchestrator.dag_executions(tenant_id);
CREATE INDEX idx_dag_executions_started_at ON orchestrator.dag_executions(started_at DESC);
CREATE INDEX idx_dag_executions_dag_status ON orchestrator.dag_executions(dag_id, status);

-- node_traces indexes (hypertable-aware)
CREATE INDEX idx_node_traces_execution ON orchestrator.node_traces(execution_id, started_at DESC);
CREATE INDEX idx_node_traces_node_id ON orchestrator.node_traces(node_id, started_at DESC);
CREATE INDEX idx_node_traces_status ON orchestrator.node_traces(status);
CREATE INDEX idx_node_traces_tenant ON orchestrator.node_traces(tenant_id);
CREATE INDEX idx_node_traces_execution_node ON orchestrator.node_traces(execution_id, node_id);

-- dag_checkpoints indexes
CREATE INDEX idx_dag_checkpoints_execution ON orchestrator.dag_checkpoints(execution_id);
CREATE INDEX idx_dag_checkpoints_tenant ON orchestrator.dag_checkpoints(tenant_id);

-- execution_provenance indexes
CREATE INDEX idx_execution_provenance_execution ON orchestrator.execution_provenance(execution_id);
CREATE INDEX idx_execution_provenance_node ON orchestrator.execution_provenance(execution_id, node_id);
CREATE INDEX idx_execution_provenance_chain_hash ON orchestrator.execution_provenance(chain_hash);
CREATE INDEX idx_execution_provenance_tenant ON orchestrator.execution_provenance(tenant_id);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE orchestrator.dag_workflows ENABLE ROW LEVEL SECURITY;
CREATE POLICY orchestrator_dag_workflows_tenant_read ON orchestrator.dag_workflows
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY orchestrator_dag_workflows_tenant_write ON orchestrator.dag_workflows
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE orchestrator.dag_executions ENABLE ROW LEVEL SECURITY;
CREATE POLICY orchestrator_dag_executions_tenant_read ON orchestrator.dag_executions
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY orchestrator_dag_executions_tenant_write ON orchestrator.dag_executions
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE orchestrator.node_traces ENABLE ROW LEVEL SECURITY;
CREATE POLICY orchestrator_node_traces_tenant_read ON orchestrator.node_traces
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY orchestrator_node_traces_tenant_write ON orchestrator.node_traces
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE orchestrator.dag_checkpoints ENABLE ROW LEVEL SECURITY;
CREATE POLICY orchestrator_dag_checkpoints_tenant_read ON orchestrator.dag_checkpoints
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY orchestrator_dag_checkpoints_tenant_write ON orchestrator.dag_checkpoints
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE orchestrator.execution_provenance ENABLE ROW LEVEL SECURITY;
CREATE POLICY orchestrator_execution_provenance_tenant_read ON orchestrator.execution_provenance
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY orchestrator_execution_provenance_tenant_write ON orchestrator.execution_provenance
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA orchestrator TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA orchestrator TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA orchestrator TO greenlang_app;

-- Grant SELECT on the continuous aggregate
GRANT SELECT ON orchestrator.hourly_node_summaries TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA orchestrator TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA orchestrator TO greenlang_readonly;
GRANT SELECT ON orchestrator.hourly_node_summaries TO greenlang_readonly;

-- Add orchestrator permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'orchestrator:dags:read', 'orchestrator', 'read', 'View DAG workflow definitions'),
    (gen_random_uuid(), 'orchestrator:dags:write', 'orchestrator', 'write', 'Create/update DAG workflow definitions'),
    (gen_random_uuid(), 'orchestrator:dags:delete', 'orchestrator', 'delete', 'Delete DAG workflow definitions'),
    (gen_random_uuid(), 'orchestrator:executions:read', 'orchestrator', 'execution_read', 'View DAG execution records and traces'),
    (gen_random_uuid(), 'orchestrator:executions:write', 'orchestrator', 'execution_write', 'Execute DAGs and manage executions'),
    (gen_random_uuid(), 'orchestrator:executions:cancel', 'orchestrator', 'execution_cancel', 'Cancel running DAG executions'),
    (gen_random_uuid(), 'orchestrator:checkpoints:read', 'orchestrator', 'checkpoint_read', 'View DAG checkpoints'),
    (gen_random_uuid(), 'orchestrator:checkpoints:delete', 'orchestrator', 'checkpoint_delete', 'Delete DAG checkpoints'),
    (gen_random_uuid(), 'orchestrator:provenance:read', 'orchestrator', 'provenance_read', 'View execution provenance chains'),
    (gen_random_uuid(), 'orchestrator:admin', 'orchestrator', 'admin', 'Orchestrator administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Trigger: auto-update updated_at on dag_workflows
-- =============================================================================

CREATE OR REPLACE FUNCTION orchestrator.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_dag_workflows_updated_at
    BEFORE UPDATE ON orchestrator.dag_workflows
    FOR EACH ROW
    EXECUTE FUNCTION orchestrator.update_updated_at();

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep node traces for 180 days
SELECT add_retention_policy('orchestrator.node_traces', INTERVAL '180 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on node traces after 7 days
ALTER TABLE orchestrator.node_traces SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'execution_id',
    timescaledb.compress_orderby = 'started_at DESC'
);

SELECT add_compression_policy('orchestrator.node_traces', INTERVAL '7 days');

-- =============================================================================
-- Seed: Sample DAG Workflows
-- =============================================================================

-- Seed 1: Linear 3-node workflow (intake -> validate -> report)
INSERT INTO orchestrator.dag_workflows (dag_id, name, description, version, definition, hash, enabled, max_parallel_nodes, on_failure, created_by, tenant_id) VALUES
('dag-linear-3-node', 'Linear 3-Node Pipeline', 'Simple linear pipeline: intake, validate, report. No parallelism.', '1.0.0',
'{
  "nodes": {
    "intake": {
      "node_id": "intake",
      "agent_id": "intake_agent",
      "depends_on": [],
      "output_key": "raw_data",
      "priority": 1,
      "on_failure": "stop"
    },
    "validate": {
      "node_id": "validate",
      "agent_id": "validation_agent",
      "depends_on": ["intake"],
      "input_mapping": {"data": "results.intake.raw_data"},
      "output_key": "validated_data",
      "priority": 1,
      "on_failure": "stop"
    },
    "report": {
      "node_id": "report",
      "agent_id": "reporting_agent",
      "depends_on": ["validate"],
      "input_mapping": {"data": "results.validate.validated_data"},
      "output_key": "report_url",
      "priority": 1,
      "on_failure": "stop"
    }
  },
  "topology_levels": [["intake"], ["validate"], ["report"]]
}'::jsonb,
'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6abcd', TRUE, 10, 'fail_fast', 'system', NULL),

-- Seed 2: Diamond 4-node workflow (intake -> [scope1, scope2] -> aggregate)
('dag-diamond-4-node', 'Diamond 4-Node Emissions', 'Diamond pattern: intake feeds parallel scope1 and scope2 calculations, results aggregated.', '1.0.0',
'{
  "nodes": {
    "intake": {
      "node_id": "intake",
      "agent_id": "intake_agent",
      "depends_on": [],
      "output_key": "raw_data",
      "priority": 1,
      "on_failure": "stop"
    },
    "scope1": {
      "node_id": "scope1",
      "agent_id": "scope1_calc_agent",
      "depends_on": ["intake"],
      "input_mapping": {"data": "results.intake.raw_data"},
      "output_key": "scope1_emissions",
      "priority": 1,
      "retry_policy": {"max_retries": 3, "strategy": "exponential", "base_delay": 1.0},
      "timeout_policy": {"timeout_seconds": 120},
      "on_failure": "stop"
    },
    "scope2": {
      "node_id": "scope2",
      "agent_id": "scope2_calc_agent",
      "depends_on": ["intake"],
      "input_mapping": {"data": "results.intake.raw_data"},
      "output_key": "scope2_emissions",
      "priority": 2,
      "retry_policy": {"max_retries": 3, "strategy": "exponential", "base_delay": 1.0},
      "timeout_policy": {"timeout_seconds": 120},
      "on_failure": "stop"
    },
    "aggregate": {
      "node_id": "aggregate",
      "agent_id": "aggregation_agent",
      "depends_on": ["scope1", "scope2"],
      "input_mapping": {
        "scope1": "results.scope1.scope1_emissions",
        "scope2": "results.scope2.scope2_emissions"
      },
      "output_key": "total_emissions",
      "priority": 1,
      "on_failure": "stop"
    }
  },
  "topology_levels": [["intake"], ["scope1", "scope2"], ["aggregate"]]
}'::jsonb,
'b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6abcdef', TRUE, 10, 'fail_fast', 'system', NULL),

-- Seed 3: Complex 6-node workflow (intake -> validate -> [scope1, scope2, scope3] -> aggregate)
('dag-complex-6-node', 'Complex 6-Node Full Emissions', 'Full emissions pipeline: intake, validate, parallel scope 1/2/3, aggregate. Demonstrates retry, timeout, and compensation.', '1.0.0',
'{
  "nodes": {
    "intake": {
      "node_id": "intake",
      "agent_id": "intake_agent",
      "depends_on": [],
      "output_key": "raw_data",
      "priority": 1,
      "on_failure": "stop"
    },
    "validate": {
      "node_id": "validate",
      "agent_id": "validation_agent",
      "depends_on": ["intake"],
      "input_mapping": {"data": "results.intake.raw_data"},
      "output_key": "validated_data",
      "priority": 1,
      "on_failure": "stop"
    },
    "scope1": {
      "node_id": "scope1",
      "agent_id": "scope1_calc_agent",
      "depends_on": ["validate"],
      "input_mapping": {"data": "results.validate.validated_data"},
      "output_key": "scope1_emissions",
      "priority": 1,
      "retry_policy": {"max_retries": 3, "strategy": "exponential", "base_delay": 1.0, "max_delay": 30.0, "jitter": true},
      "timeout_policy": {"timeout_seconds": 120},
      "on_failure": "skip"
    },
    "scope2": {
      "node_id": "scope2",
      "agent_id": "scope2_calc_agent",
      "depends_on": ["validate"],
      "input_mapping": {"data": "results.validate.validated_data"},
      "output_key": "scope2_emissions",
      "priority": 2,
      "retry_policy": {"max_retries": 3, "strategy": "exponential", "base_delay": 1.0, "max_delay": 30.0, "jitter": true},
      "timeout_policy": {"timeout_seconds": 120},
      "on_failure": "skip"
    },
    "scope3": {
      "node_id": "scope3",
      "agent_id": "scope3_calc_agent",
      "depends_on": ["validate"],
      "input_mapping": {"data": "results.validate.validated_data"},
      "output_key": "scope3_emissions",
      "priority": 3,
      "retry_policy": {"max_retries": 5, "strategy": "exponential", "base_delay": 2.0, "max_delay": 60.0, "jitter": true},
      "timeout_policy": {"timeout_seconds": 300, "on_timeout": "skip"},
      "on_failure": "skip"
    },
    "aggregate": {
      "node_id": "aggregate",
      "agent_id": "aggregation_agent",
      "depends_on": ["scope1", "scope2", "scope3"],
      "input_mapping": {
        "scope1": "results.scope1.scope1_emissions",
        "scope2": "results.scope2.scope2_emissions",
        "scope3": "results.scope3.scope3_emissions"
      },
      "output_key": "total_emissions",
      "priority": 1,
      "on_failure": "stop"
    }
  },
  "topology_levels": [["intake"], ["validate"], ["scope1", "scope2", "scope3"], ["aggregate"]]
}'::jsonb,
'c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6abcdefab', TRUE, 10, 'continue', 'system', NULL)
ON CONFLICT (dag_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA orchestrator IS 'DAG Orchestrator execution engine for GreenLang Climate OS (AGENT-FOUND-001)';
COMMENT ON TABLE orchestrator.dag_workflows IS 'DAG workflow definitions with full graph structure, retry/timeout defaults, and versioning';
COMMENT ON TABLE orchestrator.dag_executions IS 'DAG execution records tracking status, timing, node counts, and provenance chain hash';
COMMENT ON TABLE orchestrator.node_traces IS 'TimescaleDB hypertable: per-node execution traces with timing, hashes, and retry counts';
COMMENT ON TABLE orchestrator.dag_checkpoints IS 'DAG-aware checkpoints enabling resume from failure with output hash integrity verification';
COMMENT ON TABLE orchestrator.execution_provenance IS 'Cryptographic provenance chain linking node executions via SHA-256 hash chain for regulatory audit';
COMMENT ON MATERIALIZED VIEW orchestrator.hourly_node_summaries IS 'Continuous aggregate: hourly node execution summaries for dashboard and trend analysis';

COMMENT ON COLUMN orchestrator.dag_workflows.dag_id IS 'Unique identifier for the DAG workflow definition';
COMMENT ON COLUMN orchestrator.dag_workflows.definition IS 'Full DAG definition as JSONB (nodes, edges, policies, mappings)';
COMMENT ON COLUMN orchestrator.dag_workflows.hash IS 'SHA-256 hash of the definition for integrity and versioning';
COMMENT ON COLUMN orchestrator.dag_workflows.max_parallel_nodes IS 'Maximum number of nodes that can execute concurrently within a level';
COMMENT ON COLUMN orchestrator.dag_workflows.on_failure IS 'DAG-level failure strategy: fail_fast, continue, or compensate';

COMMENT ON COLUMN orchestrator.dag_executions.execution_id IS 'Unique execution identifier (deterministic UUID when deterministic_mode is true)';
COMMENT ON COLUMN orchestrator.dag_executions.topology_levels IS 'Snapshot of topological sort level grouping at execution start';
COMMENT ON COLUMN orchestrator.dag_executions.provenance_chain_hash IS 'Final SHA-256 hash of the complete provenance chain for audit verification';
COMMENT ON COLUMN orchestrator.dag_executions.deterministic_mode IS 'Whether this execution used deterministic scheduling and timestamping';

COMMENT ON COLUMN orchestrator.node_traces.input_hash IS 'SHA-256 hash of node input data for deterministic replay verification';
COMMENT ON COLUMN orchestrator.node_traces.output_hash IS 'SHA-256 hash of node output data for integrity verification';
COMMENT ON COLUMN orchestrator.node_traces.duration_ms IS 'Wall-clock execution duration in milliseconds';
COMMENT ON COLUMN orchestrator.node_traces.level_index IS 'Topological level index (0-based) within the DAG execution';

COMMENT ON COLUMN orchestrator.dag_checkpoints.output_hash IS 'SHA-256 hash of checkpoint outputs for integrity verification on resume';

COMMENT ON COLUMN orchestrator.execution_provenance.parent_hashes IS 'JSON array of chain_hash values from predecessor node provenances';
COMMENT ON COLUMN orchestrator.execution_provenance.chain_hash IS 'SHA-256(node_id + input_hash + output_hash + sorted(parent_hashes)) for chain integrity';
