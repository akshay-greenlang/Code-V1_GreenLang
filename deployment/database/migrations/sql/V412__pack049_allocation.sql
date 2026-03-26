-- =============================================================================
-- PACK-049 GHG Multi-Site Management Pack
-- Migration: V412 - Allocation
-- =============================================================================
-- Pack:         PACK-049 (GHG Multi-Site Management Pack)
-- Migration:    007 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates allocation tables for distributing shared emissions across sites.
-- Allocation is necessary when emissions cannot be directly metered at the
-- site level (e.g., shared utilities, cogeneration, landlord-tenant splits,
-- shared fleet). Supports 7 allocation methods: floor area, headcount,
-- revenue, production, energy use, operating hours, and custom.
--
-- Tables (5):
--   1. ghg_multisite.gl_ms_allocation_configs
--   2. ghg_multisite.gl_ms_allocation_runs
--   3. ghg_multisite.gl_ms_allocation_results
--   4. ghg_multisite.gl_ms_landlord_tenant_splits
--   5. ghg_multisite.gl_ms_cogeneration_allocations
--
-- Also includes: indexes, RLS, comments.
-- Previous: V411__pack049_consolidation.sql
-- =============================================================================

SET search_path TO ghg_multisite, public;

-- =============================================================================
-- Table 1: ghg_multisite.gl_ms_allocation_configs
-- =============================================================================
-- Configuration for allocation rules. Defines how shared emissions are
-- distributed across sites when direct measurement is not available.

CREATE TABLE ghg_multisite.gl_ms_allocation_configs (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_configurations(id) ON DELETE CASCADE,
    allocation_name             VARCHAR(255)    NOT NULL,
    allocation_method           VARCHAR(30)     NOT NULL DEFAULT 'FLOOR_AREA',
    allocation_scope            VARCHAR(30)     NOT NULL DEFAULT 'ALL_SCOPES',
    description                 TEXT,
    applies_to_facility_types   JSONB           DEFAULT '[]',
    applies_to_site_groups      JSONB           DEFAULT '[]',
    priority                    INTEGER         NOT NULL DEFAULT 100,
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    use_weighted_average        BOOLEAN         NOT NULL DEFAULT false,
    weight_factor               NUMERIC(10,4)   NOT NULL DEFAULT 1.0000,
    rounding_precision          INTEGER         NOT NULL DEFAULT 6,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p049_ac_method CHECK (
        allocation_method IN (
            'FLOOR_AREA', 'HEADCOUNT', 'REVENUE', 'PRODUCTION',
            'ENERGY_USE', 'OPERATING_HOURS', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p049_ac_scope CHECK (
        allocation_scope IN (
            'SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'ALL_SCOPES',
            'SCOPE_1_2', 'SCOPE_2_3'
        )
    ),
    CONSTRAINT chk_p049_ac_priority CHECK (priority >= 1 AND priority <= 999),
    CONSTRAINT chk_p049_ac_weight CHECK (weight_factor > 0 AND weight_factor <= 100),
    CONSTRAINT chk_p049_ac_rounding CHECK (rounding_precision >= 0 AND rounding_precision <= 10),
    CONSTRAINT uq_p049_ac_cfg_name UNIQUE (config_id, allocation_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_ac_tenant          ON ghg_multisite.gl_ms_allocation_configs(tenant_id);
CREATE INDEX idx_p049_ac_config          ON ghg_multisite.gl_ms_allocation_configs(config_id);
CREATE INDEX idx_p049_ac_method          ON ghg_multisite.gl_ms_allocation_configs(allocation_method);
CREATE INDEX idx_p049_ac_scope           ON ghg_multisite.gl_ms_allocation_configs(allocation_scope);
CREATE INDEX idx_p049_ac_active          ON ghg_multisite.gl_ms_allocation_configs(config_id, is_active)
    WHERE is_active = true;
CREATE INDEX idx_p049_ac_priority        ON ghg_multisite.gl_ms_allocation_configs(config_id, priority);
CREATE INDEX idx_p049_ac_facility        ON ghg_multisite.gl_ms_allocation_configs USING gin(applies_to_facility_types);
CREATE INDEX idx_p049_ac_groups          ON ghg_multisite.gl_ms_allocation_configs USING gin(applies_to_site_groups);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_allocation_configs ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_ac_tenant_isolation ON ghg_multisite.gl_ms_allocation_configs
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 2: ghg_multisite.gl_ms_allocation_runs
-- =============================================================================
-- Execution record of an allocation process. Tracks which config was used,
-- how many sites received allocations, and overall results.

CREATE TABLE ghg_multisite.gl_ms_allocation_runs (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    allocation_config_id        UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_allocation_configs(id) ON DELETE CASCADE,
    consolidation_run_id        UUID            REFERENCES ghg_multisite.gl_ms_consolidation_runs(id) ON DELETE SET NULL,
    period_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_reporting_periods(id) ON DELETE CASCADE,
    status                      VARCHAR(20)     NOT NULL DEFAULT 'IN_PROGRESS',
    sites_allocated             INTEGER         NOT NULL DEFAULT 0,
    total_input_tco2e           NUMERIC(20,6)   NOT NULL DEFAULT 0,
    total_allocated_tco2e       NUMERIC(20,6)   NOT NULL DEFAULT 0,
    unallocated_tco2e           NUMERIC(20,6)   NOT NULL DEFAULT 0,
    allocation_residual_tco2e   NUMERIC(20,6)   NOT NULL DEFAULT 0,
    variance_pct                NUMERIC(10,4)   NOT NULL DEFAULT 0.0000,
    started_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    completed_at                TIMESTAMPTZ,
    error_message               TEXT,
    provenance_hash             VARCHAR(64),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p049_ar_status CHECK (
        status IN ('IN_PROGRESS', 'COMPLETED', 'FAILED', 'REVIEWED', 'APPROVED')
    ),
    CONSTRAINT chk_p049_ar_sites CHECK (sites_allocated >= 0),
    CONSTRAINT chk_p049_ar_input CHECK (total_input_tco2e >= 0),
    CONSTRAINT chk_p049_ar_allocated CHECK (total_allocated_tco2e >= 0),
    CONSTRAINT chk_p049_ar_unalloc CHECK (unallocated_tco2e >= 0),
    CONSTRAINT chk_p049_ar_residual CHECK (allocation_residual_tco2e >= 0)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_ar_tenant          ON ghg_multisite.gl_ms_allocation_runs(tenant_id);
CREATE INDEX idx_p049_ar_config          ON ghg_multisite.gl_ms_allocation_runs(allocation_config_id);
CREATE INDEX idx_p049_ar_consol          ON ghg_multisite.gl_ms_allocation_runs(consolidation_run_id)
    WHERE consolidation_run_id IS NOT NULL;
CREATE INDEX idx_p049_ar_period          ON ghg_multisite.gl_ms_allocation_runs(period_id);
CREATE INDEX idx_p049_ar_status          ON ghg_multisite.gl_ms_allocation_runs(status);
CREATE INDEX idx_p049_ar_completed       ON ghg_multisite.gl_ms_allocation_runs(status)
    WHERE status IN ('COMPLETED', 'APPROVED');

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_allocation_runs ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_ar_tenant_isolation ON ghg_multisite.gl_ms_allocation_runs
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 3: ghg_multisite.gl_ms_allocation_results
-- =============================================================================
-- Per-site allocation results. Each row shows the denominator value (e.g.,
-- floor area), the site's share percentage, and the allocated emissions.

CREATE TABLE ghg_multisite.gl_ms_allocation_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    run_id                      UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_allocation_runs(id) ON DELETE CASCADE,
    site_id                     UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    denominator_value           NUMERIC(20,6)   NOT NULL,
    denominator_unit            VARCHAR(50)     NOT NULL,
    total_denominator           NUMERIC(20,6)   NOT NULL,
    share_pct                   NUMERIC(10,6)   NOT NULL,
    input_tco2e                 NUMERIC(20,6)   NOT NULL DEFAULT 0,
    allocated_scope1_tco2e      NUMERIC(20,6)   NOT NULL DEFAULT 0,
    allocated_scope2_tco2e      NUMERIC(20,6)   NOT NULL DEFAULT 0,
    allocated_scope3_tco2e      NUMERIC(20,6)   NOT NULL DEFAULT 0,
    allocated_total_tco2e       NUMERIC(20,6)   NOT NULL DEFAULT 0,
    adjustment_factor           NUMERIC(10,4)   NOT NULL DEFAULT 1.0000,
    is_manual_override          BOOLEAN         NOT NULL DEFAULT false,
    override_justification      TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_ares_denom CHECK (denominator_value >= 0),
    CONSTRAINT chk_p049_ares_total_denom CHECK (total_denominator > 0),
    CONSTRAINT chk_p049_ares_share CHECK (share_pct >= 0 AND share_pct <= 100),
    CONSTRAINT chk_p049_ares_input CHECK (input_tco2e >= 0),
    CONSTRAINT chk_p049_ares_s1 CHECK (allocated_scope1_tco2e >= 0),
    CONSTRAINT chk_p049_ares_s2 CHECK (allocated_scope2_tco2e >= 0),
    CONSTRAINT chk_p049_ares_s3 CHECK (allocated_scope3_tco2e >= 0),
    CONSTRAINT chk_p049_ares_total CHECK (allocated_total_tco2e >= 0),
    CONSTRAINT chk_p049_ares_adj CHECK (adjustment_factor > 0 AND adjustment_factor <= 100),
    CONSTRAINT uq_p049_ares_run_site UNIQUE (run_id, site_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_ares_tenant        ON ghg_multisite.gl_ms_allocation_results(tenant_id);
CREATE INDEX idx_p049_ares_run           ON ghg_multisite.gl_ms_allocation_results(run_id);
CREATE INDEX idx_p049_ares_site          ON ghg_multisite.gl_ms_allocation_results(site_id);
CREATE INDEX idx_p049_ares_override      ON ghg_multisite.gl_ms_allocation_results(run_id)
    WHERE is_manual_override = true;
CREATE INDEX idx_p049_ares_share         ON ghg_multisite.gl_ms_allocation_results(share_pct DESC);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_allocation_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_ares_tenant_isolation ON ghg_multisite.gl_ms_allocation_results
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 4: ghg_multisite.gl_ms_landlord_tenant_splits
-- =============================================================================
-- Landlord-tenant emission splits for leased sites. Determines how
-- building-level emissions are divided between landlord (building owner)
-- and tenant (occupier) based on lease terms and metering arrangements.

CREATE TABLE ghg_multisite.gl_ms_landlord_tenant_splits (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_configurations(id) ON DELETE CASCADE,
    site_id                     UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    split_type                  VARCHAR(30)     NOT NULL DEFAULT 'AREA_BASED',
    reporting_party             VARCHAR(20)     NOT NULL DEFAULT 'TENANT',
    tenant_share_pct            NUMERIC(10,4)   NOT NULL,
    landlord_share_pct          NUMERIC(10,4)   NOT NULL,
    common_area_pct             NUMERIC(10,4)   NOT NULL DEFAULT 0.0000,
    lease_start_date            DATE,
    lease_end_date              DATE,
    is_sub_metered              BOOLEAN         NOT NULL DEFAULT false,
    sub_meter_coverage          VARCHAR(20),
    lease_type                  VARCHAR(30)     NOT NULL DEFAULT 'STANDARD',
    applies_to_scope1           BOOLEAN         NOT NULL DEFAULT true,
    applies_to_scope2           BOOLEAN         NOT NULL DEFAULT true,
    applies_to_scope3           BOOLEAN         NOT NULL DEFAULT false,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_lts_split_type CHECK (
        split_type IN (
            'AREA_BASED', 'METERED', 'FIXED_PERCENTAGE',
            'HEADCOUNT_BASED', 'NEGOTIATED', 'REGULATORY'
        )
    ),
    CONSTRAINT chk_p049_lts_party CHECK (
        reporting_party IN ('TENANT', 'LANDLORD', 'BOTH')
    ),
    CONSTRAINT chk_p049_lts_tenant_share CHECK (
        tenant_share_pct >= 0 AND tenant_share_pct <= 100
    ),
    CONSTRAINT chk_p049_lts_landlord_share CHECK (
        landlord_share_pct >= 0 AND landlord_share_pct <= 100
    ),
    CONSTRAINT chk_p049_lts_common CHECK (
        common_area_pct >= 0 AND common_area_pct <= 100
    ),
    CONSTRAINT chk_p049_lts_sum CHECK (
        tenant_share_pct + landlord_share_pct + common_area_pct <= 100.0001
    ),
    CONSTRAINT chk_p049_lts_sub_meter CHECK (
        sub_meter_coverage IS NULL OR sub_meter_coverage IN (
            'FULL', 'PARTIAL', 'ELECTRICITY_ONLY', 'GAS_ONLY', 'NONE'
        )
    ),
    CONSTRAINT chk_p049_lts_lease CHECK (
        lease_type IN (
            'STANDARD', 'GREEN_LEASE', 'NET_LEASE', 'GROSS_LEASE',
            'TRIPLE_NET', 'MODIFIED_GROSS', 'OTHER'
        )
    ),
    CONSTRAINT chk_p049_lts_dates CHECK (
        lease_end_date IS NULL OR lease_start_date IS NULL OR lease_end_date > lease_start_date
    ),
    CONSTRAINT uq_p049_lts_cfg_site UNIQUE (config_id, site_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_lts_tenant_col     ON ghg_multisite.gl_ms_landlord_tenant_splits(tenant_id);
CREATE INDEX idx_p049_lts_config         ON ghg_multisite.gl_ms_landlord_tenant_splits(config_id);
CREATE INDEX idx_p049_lts_site           ON ghg_multisite.gl_ms_landlord_tenant_splits(site_id);
CREATE INDEX idx_p049_lts_type           ON ghg_multisite.gl_ms_landlord_tenant_splits(split_type);
CREATE INDEX idx_p049_lts_party          ON ghg_multisite.gl_ms_landlord_tenant_splits(reporting_party);
CREATE INDEX idx_p049_lts_lease          ON ghg_multisite.gl_ms_landlord_tenant_splits(lease_type);
CREATE INDEX idx_p049_lts_metered        ON ghg_multisite.gl_ms_landlord_tenant_splits(config_id)
    WHERE is_sub_metered = true;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_landlord_tenant_splits ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_lts_tenant_isolation ON ghg_multisite.gl_ms_landlord_tenant_splits
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 5: ghg_multisite.gl_ms_cogeneration_allocations
-- =============================================================================
-- Allocation of cogeneration (CHP) plant emissions across sites that
-- consume the generated electricity, steam, or heat. Follows GHG Protocol
-- guidance on allocating CHP emissions.

CREATE TABLE ghg_multisite.gl_ms_cogeneration_allocations (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_configurations(id) ON DELETE CASCADE,
    period_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_reporting_periods(id) ON DELETE CASCADE,
    source_site_id              UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    consuming_site_id           UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    allocation_method           VARCHAR(30)     NOT NULL DEFAULT 'EFFICIENCY',
    total_generation_mwh        NUMERIC(20,6)   NOT NULL,
    electricity_mwh             NUMERIC(20,6)   NOT NULL DEFAULT 0,
    heat_mwh                    NUMERIC(20,6)   NOT NULL DEFAULT 0,
    steam_mwh                   NUMERIC(20,6)   NOT NULL DEFAULT 0,
    consumed_mwh                NUMERIC(20,6)   NOT NULL,
    consumption_share_pct       NUMERIC(10,4)   NOT NULL,
    total_plant_emissions_tco2e NUMERIC(20,6)   NOT NULL,
    allocated_tco2e             NUMERIC(20,6)   NOT NULL,
    efficiency_pct              NUMERIC(10,4),
    provenance_hash             VARCHAR(64),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_cog_method CHECK (
        allocation_method IN (
            'EFFICIENCY', 'ENERGY_CONTENT', 'WORK_POTENTIAL',
            'DIRECT_METERING', 'CAPACITY_SHARE'
        )
    ),
    CONSTRAINT chk_p049_cog_generation CHECK (total_generation_mwh > 0),
    CONSTRAINT chk_p049_cog_elec CHECK (electricity_mwh >= 0),
    CONSTRAINT chk_p049_cog_heat CHECK (heat_mwh >= 0),
    CONSTRAINT chk_p049_cog_steam CHECK (steam_mwh >= 0),
    CONSTRAINT chk_p049_cog_consumed CHECK (consumed_mwh >= 0),
    CONSTRAINT chk_p049_cog_share CHECK (
        consumption_share_pct >= 0 AND consumption_share_pct <= 100
    ),
    CONSTRAINT chk_p049_cog_plant CHECK (total_plant_emissions_tco2e >= 0),
    CONSTRAINT chk_p049_cog_allocated CHECK (allocated_tco2e >= 0),
    CONSTRAINT chk_p049_cog_eff CHECK (
        efficiency_pct IS NULL OR (efficiency_pct >= 0 AND efficiency_pct <= 100)
    ),
    CONSTRAINT chk_p049_cog_no_self CHECK (source_site_id != consuming_site_id),
    CONSTRAINT uq_p049_cog_period_src_cons UNIQUE (
        period_id, source_site_id, consuming_site_id, allocation_method
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_cog_tenant         ON ghg_multisite.gl_ms_cogeneration_allocations(tenant_id);
CREATE INDEX idx_p049_cog_config         ON ghg_multisite.gl_ms_cogeneration_allocations(config_id);
CREATE INDEX idx_p049_cog_period         ON ghg_multisite.gl_ms_cogeneration_allocations(period_id);
CREATE INDEX idx_p049_cog_source         ON ghg_multisite.gl_ms_cogeneration_allocations(source_site_id);
CREATE INDEX idx_p049_cog_consumer       ON ghg_multisite.gl_ms_cogeneration_allocations(consuming_site_id);
CREATE INDEX idx_p049_cog_method         ON ghg_multisite.gl_ms_cogeneration_allocations(allocation_method);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_cogeneration_allocations ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_cog_tenant_isolation ON ghg_multisite.gl_ms_cogeneration_allocations
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_multisite.gl_ms_allocation_configs IS
    'PACK-049: Allocation configuration (7 methods, 6 scopes) with priority, weighting, and facility type filters.';
COMMENT ON TABLE ghg_multisite.gl_ms_allocation_runs IS
    'PACK-049: Allocation execution records with input/output totals, residuals, and variance tracking.';
COMMENT ON TABLE ghg_multisite.gl_ms_allocation_results IS
    'PACK-049: Per-site allocation results with denominator values, share percentages, and scope-level totals.';
COMMENT ON TABLE ghg_multisite.gl_ms_landlord_tenant_splits IS
    'PACK-049: Landlord-tenant emission splits (6 types, 7 lease types) with sub-metering and scope applicability.';
COMMENT ON TABLE ghg_multisite.gl_ms_cogeneration_allocations IS
    'PACK-049: CHP/cogeneration emission allocation (5 methods) across consuming sites with efficiency tracking.';
