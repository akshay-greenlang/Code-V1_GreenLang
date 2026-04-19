-- V438__comply_hub_jobs.sql
-- GL-Comply-APP: unified compliance jobs + per-framework results + applicability cache
-- Depends on: V437 (scope_computations)

CREATE TABLE IF NOT EXISTS comply_jobs (
    job_id UUID PRIMARY KEY,
    entity_id VARCHAR(128) NOT NULL,
    legal_name VARCHAR(256) NOT NULL,
    jurisdiction VARCHAR(8) NOT NULL,
    reporting_period_start TIMESTAMPTZ NOT NULL,
    reporting_period_end TIMESTAMPTZ NOT NULL,
    frameworks_requested TEXT[] NOT NULL,
    overall_status VARCHAR(32) NOT NULL,
    aggregate_provenance_hash CHAR(64) NOT NULL,
    gap_analysis JSONB NOT NULL DEFAULT '[]'::jsonb,
    unified_report_uri VARCHAR(512),
    report_format VARCHAR(16) NOT NULL DEFAULT 'pdf',
    tenant_id VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_comply_jobs_entity
    ON comply_jobs (entity_id, reporting_period_start DESC);

CREATE INDEX IF NOT EXISTS idx_comply_jobs_tenant
    ON comply_jobs (tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_comply_jobs_status
    ON comply_jobs (overall_status, created_at DESC);

CREATE TABLE IF NOT EXISTS comply_framework_results (
    result_id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES comply_jobs(job_id) ON DELETE CASCADE,
    framework VARCHAR(32) NOT NULL,
    compliance_status VARCHAR(32) NOT NULL,
    report_uri VARCHAR(512),
    findings_summary TEXT,
    provenance_hash CHAR(64),
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    duration_ms INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (job_id, framework)
);

CREATE INDEX IF NOT EXISTS idx_comply_framework_results_framework
    ON comply_framework_results (framework, created_at DESC);

CREATE TABLE IF NOT EXISTS comply_unified_reports (
    report_id UUID PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES comply_jobs(job_id) ON DELETE CASCADE,
    format VARCHAR(16) NOT NULL,
    content_uri VARCHAR(512) NOT NULL,
    content_hash CHAR(64) NOT NULL,
    size_bytes BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_comply_unified_reports_job
    ON comply_unified_reports (job_id);

CREATE TABLE IF NOT EXISTS comply_applicability_cache (
    cache_key CHAR(64) PRIMARY KEY,
    entity_id VARCHAR(128) NOT NULL,
    reporting_year INT NOT NULL,
    applicable_frameworks TEXT[] NOT NULL,
    rationale JSONB NOT NULL DEFAULT '{}'::jsonb,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_comply_applicability_entity
    ON comply_applicability_cache (entity_id, reporting_year);

COMMENT ON TABLE comply_jobs IS
  'GL-Comply-APP: one row per compliance orchestration run';
COMMENT ON TABLE comply_framework_results IS
  'GL-Comply-APP: per-framework result for each job (1:N on comply_jobs)';
COMMENT ON TABLE comply_unified_reports IS
  'GL-Comply-APP: generated report artifacts (JSON/PDF/XML) with content hashes';
COMMENT ON TABLE comply_applicability_cache IS
  'GL-Comply-APP: cached applicability evaluations (key = sha256 of entity + year + rule version)';
