-- =============================================================================
-- PACK-048 GHG Assurance Prep Pack
-- Migration: V399 - Provenance Chains
-- =============================================================================
-- Pack:         PACK-048 (GHG Assurance Prep Pack)
-- Migration:    004 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates provenance chain tables for demonstrating the complete audit trail
-- from source data through calculation to final emissions values. Each chain
-- covers a specific scope and source category with step-by-step traceability.
-- Steps track input/output values, formulas, emission factors, and SHA-256
-- hashes forming a Merkle-like chain for tamper detection.
--
-- Tables (2):
--   1. ghg_assurance.gl_ap_provenance_chains
--   2. ghg_assurance.gl_ap_provenance_steps
--
-- Also includes: indexes, RLS, comments.
-- Previous: V398__pack048_readiness.sql
-- =============================================================================

SET search_path TO ghg_assurance, public;

-- =============================================================================
-- Table 1: ghg_assurance.gl_ap_provenance_chains
-- =============================================================================
-- Top-level provenance chain per scope and source category. Tracks the total
-- number of steps, completeness percentage, root and final hashes forming
-- a Merkle chain, and the total tCO2e produced by the chain. Each chain
-- represents a complete audit trail from raw data to final emission value.

CREATE TABLE ghg_assurance.gl_ap_provenance_chains (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_configurations(id) ON DELETE CASCADE,
    scope                       VARCHAR(20)     NOT NULL,
    source_category             VARCHAR(100)    NOT NULL,
    total_steps                 INTEGER         NOT NULL DEFAULT 0,
    completeness_pct            NUMERIC(5,2)    NOT NULL DEFAULT 0,
    root_hash                   TEXT,
    final_hash                  TEXT,
    total_tco2e                 NUMERIC(20,6),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    generated_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p048_pc_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3')
    ),
    CONSTRAINT chk_p048_pc_steps CHECK (
        total_steps >= 0
    ),
    CONSTRAINT chk_p048_pc_completeness CHECK (
        completeness_pct >= 0 AND completeness_pct <= 100
    ),
    CONSTRAINT chk_p048_pc_tco2e CHECK (
        total_tco2e IS NULL OR total_tco2e >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_pc_tenant            ON ghg_assurance.gl_ap_provenance_chains(tenant_id);
CREATE INDEX idx_p048_pc_config            ON ghg_assurance.gl_ap_provenance_chains(config_id);
CREATE INDEX idx_p048_pc_scope             ON ghg_assurance.gl_ap_provenance_chains(scope);
CREATE INDEX idx_p048_pc_category          ON ghg_assurance.gl_ap_provenance_chains(source_category);
CREATE INDEX idx_p048_pc_completeness      ON ghg_assurance.gl_ap_provenance_chains(completeness_pct);
CREATE INDEX idx_p048_pc_root_hash         ON ghg_assurance.gl_ap_provenance_chains(root_hash);
CREATE INDEX idx_p048_pc_final_hash        ON ghg_assurance.gl_ap_provenance_chains(final_hash);
CREATE INDEX idx_p048_pc_generated         ON ghg_assurance.gl_ap_provenance_chains(generated_at DESC);
CREATE INDEX idx_p048_pc_created           ON ghg_assurance.gl_ap_provenance_chains(created_at DESC);
CREATE INDEX idx_p048_pc_provenance        ON ghg_assurance.gl_ap_provenance_chains(provenance_hash);
CREATE INDEX idx_p048_pc_metadata          ON ghg_assurance.gl_ap_provenance_chains USING GIN(metadata);

-- Composite: config + scope for scope-filtered retrieval
CREATE INDEX idx_p048_pc_config_scope      ON ghg_assurance.gl_ap_provenance_chains(config_id, scope);

-- Composite: tenant + config for scoped listing
CREATE INDEX idx_p048_pc_tenant_config     ON ghg_assurance.gl_ap_provenance_chains(tenant_id, config_id);

-- Composite: scope + category for cross-config analysis
CREATE INDEX idx_p048_pc_scope_cat         ON ghg_assurance.gl_ap_provenance_chains(scope, source_category);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_pc_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_provenance_chains
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_assurance.gl_ap_provenance_steps
-- =============================================================================
-- Individual steps within a provenance chain. Each step records the
-- transformation from input to output with formula references, emission
-- factor sources, data quality grading, and SHA-256 hashing. Steps form
-- a linked chain via parent_hash -> step_hash references, enabling
-- tamper detection and complete audit trail reconstruction.

CREATE TABLE ghg_assurance.gl_ap_provenance_steps (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    chain_id                    UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_provenance_chains(id) ON DELETE CASCADE,
    step_order                  INTEGER         NOT NULL,
    step_type                   VARCHAR(30)     NOT NULL,
    description                 TEXT,
    input_value                 NUMERIC(20,6),
    input_unit                  VARCHAR(50),
    output_value                NUMERIC(20,6),
    output_unit                 VARCHAR(50),
    formula_reference           TEXT,
    ef_source                   VARCHAR(255),
    ef_version                  VARCHAR(50),
    data_quality_grade          VARCHAR(20),
    parent_hash                 TEXT,
    step_hash                   TEXT            NOT NULL,
    step_timestamp              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p048_ps_order CHECK (
        step_order >= 1
    ),
    CONSTRAINT chk_p048_ps_type CHECK (
        step_type IN (
            'SOURCE_DATA', 'EMISSION_FACTOR', 'FORMULA',
            'INTERMEDIATE', 'FINAL'
        )
    ),
    CONSTRAINT chk_p048_ps_input CHECK (
        input_value IS NULL OR input_value >= 0
    ),
    CONSTRAINT chk_p048_ps_output CHECK (
        output_value IS NULL OR output_value >= 0
    ),
    CONSTRAINT chk_p048_ps_dq_grade CHECK (
        data_quality_grade IS NULL OR data_quality_grade IN (
            'EXCELLENT', 'GOOD', 'ADEQUATE', 'MARGINAL', 'INSUFFICIENT'
        )
    ),
    CONSTRAINT uq_p048_ps_chain_order UNIQUE (chain_id, step_order)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_ps_tenant            ON ghg_assurance.gl_ap_provenance_steps(tenant_id);
CREATE INDEX idx_p048_ps_chain             ON ghg_assurance.gl_ap_provenance_steps(chain_id);
CREATE INDEX idx_p048_ps_order             ON ghg_assurance.gl_ap_provenance_steps(step_order);
CREATE INDEX idx_p048_ps_type              ON ghg_assurance.gl_ap_provenance_steps(step_type);
CREATE INDEX idx_p048_ps_ef_source         ON ghg_assurance.gl_ap_provenance_steps(ef_source);
CREATE INDEX idx_p048_ps_dq_grade          ON ghg_assurance.gl_ap_provenance_steps(data_quality_grade);
CREATE INDEX idx_p048_ps_step_hash         ON ghg_assurance.gl_ap_provenance_steps(step_hash);
CREATE INDEX idx_p048_ps_parent_hash       ON ghg_assurance.gl_ap_provenance_steps(parent_hash);
CREATE INDEX idx_p048_ps_timestamp         ON ghg_assurance.gl_ap_provenance_steps(step_timestamp DESC);
CREATE INDEX idx_p048_ps_created           ON ghg_assurance.gl_ap_provenance_steps(created_at DESC);
CREATE INDEX idx_p048_ps_metadata          ON ghg_assurance.gl_ap_provenance_steps USING GIN(metadata);

-- Composite: chain + order for ordered step retrieval
CREATE INDEX idx_p048_ps_chain_order       ON ghg_assurance.gl_ap_provenance_steps(chain_id, step_order ASC);

-- Composite: chain + type for type-filtered queries
CREATE INDEX idx_p048_ps_chain_type        ON ghg_assurance.gl_ap_provenance_steps(chain_id, step_type);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_assurance.gl_ap_provenance_chains ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_assurance.gl_ap_provenance_steps ENABLE ROW LEVEL SECURITY;

CREATE POLICY p048_pc_tenant_isolation
    ON ghg_assurance.gl_ap_provenance_chains
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_pc_service_bypass
    ON ghg_assurance.gl_ap_provenance_chains
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p048_ps_tenant_isolation
    ON ghg_assurance.gl_ap_provenance_steps
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_ps_service_bypass
    ON ghg_assurance.gl_ap_provenance_steps
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_provenance_chains TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_provenance_steps TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_assurance.gl_ap_provenance_chains IS
    'Top-level provenance chains per scope and source category with Merkle-like hash chain for tamper detection.';
COMMENT ON TABLE ghg_assurance.gl_ap_provenance_steps IS
    'Individual provenance steps with input/output values, formula references, emission factor sources, and hash linking.';

COMMENT ON COLUMN ghg_assurance.gl_ap_provenance_chains.scope IS 'Emission scope: SCOPE_1, SCOPE_2, or SCOPE_3.';
COMMENT ON COLUMN ghg_assurance.gl_ap_provenance_chains.source_category IS 'Source category within scope, e.g. stationary_combustion, purchased_electricity, business_travel.';
COMMENT ON COLUMN ghg_assurance.gl_ap_provenance_chains.root_hash IS 'SHA-256 hash of the first step (source data), forming the chain root.';
COMMENT ON COLUMN ghg_assurance.gl_ap_provenance_chains.final_hash IS 'SHA-256 hash of the final step, incorporating all prior hashes for tamper detection.';
COMMENT ON COLUMN ghg_assurance.gl_ap_provenance_chains.total_tco2e IS 'Total emissions (tCO2e) produced by this chain.';
COMMENT ON COLUMN ghg_assurance.gl_ap_provenance_steps.step_type IS 'Step type: SOURCE_DATA (raw input), EMISSION_FACTOR (EF lookup), FORMULA (calculation), INTERMEDIATE (sub-total), FINAL (result).';
COMMENT ON COLUMN ghg_assurance.gl_ap_provenance_steps.formula_reference IS 'Reference to the formula or calculation method used, e.g. GHG_PROTOCOL_CH4_COMBUSTION.';
COMMENT ON COLUMN ghg_assurance.gl_ap_provenance_steps.ef_source IS 'Emission factor source, e.g. DEFRA_2025, EPA_eGRID_2024, IEA_2023.';
COMMENT ON COLUMN ghg_assurance.gl_ap_provenance_steps.ef_version IS 'Emission factor version or publication year for change tracking.';
COMMENT ON COLUMN ghg_assurance.gl_ap_provenance_steps.parent_hash IS 'SHA-256 hash of the preceding step, forming the chain link. NULL for first step.';
COMMENT ON COLUMN ghg_assurance.gl_ap_provenance_steps.step_hash IS 'SHA-256 hash of this step (inputs + outputs + parent_hash) for integrity verification.';
