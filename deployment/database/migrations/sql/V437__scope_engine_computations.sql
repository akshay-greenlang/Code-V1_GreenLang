-- V437__scope_engine_computations.sql
-- Scope Engine: computation + per-result persistence
-- Depends on: V001 (extensions), V426 (factors catalog)

CREATE TABLE IF NOT EXISTS scope_computations (
    computation_id UUID PRIMARY KEY,
    entity_id VARCHAR(128),
    reporting_period_start TIMESTAMPTZ NOT NULL,
    reporting_period_end TIMESTAMPTZ NOT NULL,
    gwp_basis VARCHAR(32) NOT NULL,
    consolidation VARCHAR(32) NOT NULL,
    total_co2e_kg NUMERIC(24, 6) NOT NULL,
    scope_1_co2e_kg NUMERIC(24, 6) NOT NULL DEFAULT 0,
    scope_2_location_co2e_kg NUMERIC(24, 6) NOT NULL DEFAULT 0,
    scope_2_market_co2e_kg NUMERIC(24, 6) NOT NULL DEFAULT 0,
    scope_3_co2e_kg NUMERIC(24, 6) NOT NULL DEFAULT 0,
    scope_3_by_category JSONB NOT NULL DEFAULT '{}'::jsonb,
    computation_hash CHAR(64) NOT NULL,
    tenant_id VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_scope_computations_entity
    ON scope_computations (entity_id, reporting_period_start DESC);

CREATE INDEX IF NOT EXISTS idx_scope_computations_tenant
    ON scope_computations (tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_scope_computations_hash
    ON scope_computations (computation_hash);

CREATE TABLE IF NOT EXISTS scope_activity_results (
    result_id BIGSERIAL PRIMARY KEY,
    computation_id UUID NOT NULL REFERENCES scope_computations(computation_id) ON DELETE CASCADE,
    activity_id VARCHAR(128) NOT NULL,
    scope VARCHAR(2) NOT NULL,
    gas VARCHAR(8) NOT NULL,
    gas_amount_kg NUMERIC(24, 6) NOT NULL,
    gwp_basis VARCHAR(32) NOT NULL,
    co2e_kg NUMERIC(24, 6) NOT NULL,
    factor_id VARCHAR(256) NOT NULL,
    factor_source VARCHAR(128),
    factor_vintage INT NOT NULL,
    formula_hash CHAR(64) NOT NULL,
    cached BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_scope_activity_results_computation
    ON scope_activity_results (computation_id);

CREATE INDEX IF NOT EXISTS idx_scope_activity_results_scope_gas
    ON scope_activity_results (scope, gas);

CREATE INDEX IF NOT EXISTS idx_scope_activity_results_factor
    ON scope_activity_results (factor_id, factor_vintage);

COMMENT ON TABLE scope_computations IS
  'Scope Engine: one row per compute() call, aggregated by entity + period';
COMMENT ON TABLE scope_activity_results IS
  'Scope Engine: one row per (activity, gas) with provenance hash';
COMMENT ON COLUMN scope_computations.computation_hash IS
  'SHA-256 over (request, results) — deterministic content hash for dedup/audit';
COMMENT ON COLUMN scope_activity_results.formula_hash IS
  'SHA-256 over (activity, factor, gas, gwp_basis) — per-result provenance';
