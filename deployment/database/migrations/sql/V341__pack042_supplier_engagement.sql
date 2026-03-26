-- =============================================================================
-- V341: PACK-042 Scope 3 Starter Pack - Supplier Engagement & Data Collection
-- =============================================================================
-- Pack:         PACK-042 (Scope 3 Starter Pack)
-- Migration:    006 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates supplier engagement and data collection tables. Manages the
-- supplier master list, engagement plans with data quality improvement
-- targets, outbound data requests, inbound supplier responses, and
-- aggregate engagement metrics. Supports the transition from spend-based
-- (Tier 1) to supplier-specific (Tier 3) calculation methods.
--
-- Tables (5):
--   1. ghg_accounting_scope3.suppliers
--   2. ghg_accounting_scope3.engagement_plans
--   3. ghg_accounting_scope3.data_requests
--   4. ghg_accounting_scope3.supplier_responses
--   5. ghg_accounting_scope3.engagement_metrics
--
-- Also includes: indexes, RLS, comments.
-- Previous: V340__pack042_hotspot_analysis.sql
-- =============================================================================

SET search_path TO ghg_accounting_scope3, public;

-- =============================================================================
-- Table 1: ghg_accounting_scope3.suppliers
-- =============================================================================
-- Supplier master list for Scope 3 engagement. Each supplier record tracks
-- industry classification, geographic location, procurement spend, emission
-- contribution estimate, and engagement tier (strategic, important, standard).
-- Engagement tier determines the level of data quality effort applied.

CREATE TABLE ghg_accounting_scope3.suppliers (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    -- Supplier identification
    name                        VARCHAR(500)    NOT NULL,
    supplier_code               VARCHAR(100),
    legal_name                  VARCHAR(500),
    duns_number                 VARCHAR(20),
    -- Industry
    industry_naics              VARCHAR(10),
    industry_description        VARCHAR(500),
    sector                      VARCHAR(100),
    -- Location
    country                     VARCHAR(3)      NOT NULL DEFAULT 'US',
    city                        VARCHAR(200),
    region                      VARCHAR(200),
    -- Financials
    procurement_spend           NUMERIC(18,2)   DEFAULT 0,
    spend_currency              VARCHAR(3)      DEFAULT 'USD',
    spend_period_start          DATE,
    spend_period_end            DATE,
    spend_pct_of_total          DECIMAL(5,2),
    -- Emission contribution
    emission_contribution_tco2e DECIMAL(12,3),
    emission_contribution_pct   DECIMAL(5,2),
    primary_scope3_categories   ghg_accounting_scope3.scope3_category_type[],
    -- Engagement
    engagement_tier             VARCHAR(20)     NOT NULL DEFAULT 'STANDARD',
    has_sustainability_report   BOOLEAN         DEFAULT false,
    has_cdp_score               BOOLEAN         DEFAULT false,
    cdp_score                   VARCHAR(5),
    has_sbti_target             BOOLEAN         DEFAULT false,
    -- Contact
    primary_contact_name        VARCHAR(255),
    primary_contact_email       VARCHAR(255),
    primary_contact_phone       VARCHAR(50),
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'ACTIVE',
    onboarded_at                TIMESTAMPTZ,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p042_sup_country_len CHECK (
        LENGTH(country) BETWEEN 2 AND 3
    ),
    CONSTRAINT chk_p042_sup_spend CHECK (
        procurement_spend IS NULL OR procurement_spend >= 0
    ),
    CONSTRAINT chk_p042_sup_spend_pct CHECK (
        spend_pct_of_total IS NULL OR (spend_pct_of_total >= 0 AND spend_pct_of_total <= 100)
    ),
    CONSTRAINT chk_p042_sup_emission_tco2e CHECK (
        emission_contribution_tco2e IS NULL OR emission_contribution_tco2e >= 0
    ),
    CONSTRAINT chk_p042_sup_emission_pct CHECK (
        emission_contribution_pct IS NULL OR (emission_contribution_pct >= 0 AND emission_contribution_pct <= 100)
    ),
    CONSTRAINT chk_p042_sup_tier CHECK (
        engagement_tier IN ('STRATEGIC', 'IMPORTANT', 'STANDARD', 'MINIMAL', 'INACTIVE')
    ),
    CONSTRAINT chk_p042_sup_status CHECK (
        status IN ('ACTIVE', 'INACTIVE', 'ONBOARDING', 'SUSPENDED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p042_sup_spend_dates CHECK (
        spend_period_start IS NULL OR spend_period_end IS NULL OR
        spend_period_start <= spend_period_end
    ),
    CONSTRAINT uq_p042_sup_tenant_code UNIQUE (tenant_id, supplier_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_sup_tenant            ON ghg_accounting_scope3.suppliers(tenant_id);
CREATE INDEX idx_p042_sup_name              ON ghg_accounting_scope3.suppliers(name);
CREATE INDEX idx_p042_sup_code              ON ghg_accounting_scope3.suppliers(supplier_code);
CREATE INDEX idx_p042_sup_naics             ON ghg_accounting_scope3.suppliers(industry_naics);
CREATE INDEX idx_p042_sup_country           ON ghg_accounting_scope3.suppliers(country);
CREATE INDEX idx_p042_sup_spend             ON ghg_accounting_scope3.suppliers(procurement_spend DESC);
CREATE INDEX idx_p042_sup_emission          ON ghg_accounting_scope3.suppliers(emission_contribution_tco2e DESC);
CREATE INDEX idx_p042_sup_tier              ON ghg_accounting_scope3.suppliers(engagement_tier);
CREATE INDEX idx_p042_sup_status            ON ghg_accounting_scope3.suppliers(status);
CREATE INDEX idx_p042_sup_cdp               ON ghg_accounting_scope3.suppliers(has_cdp_score) WHERE has_cdp_score = true;
CREATE INDEX idx_p042_sup_sbti              ON ghg_accounting_scope3.suppliers(has_sbti_target) WHERE has_sbti_target = true;
CREATE INDEX idx_p042_sup_created           ON ghg_accounting_scope3.suppliers(created_at DESC);
CREATE INDEX idx_p042_sup_metadata          ON ghg_accounting_scope3.suppliers USING GIN(metadata);
CREATE INDEX idx_p042_sup_categories        ON ghg_accounting_scope3.suppliers USING GIN(primary_scope3_categories);

-- Composite: tenant + active + tier for engagement priority
CREATE INDEX idx_p042_sup_tenant_active     ON ghg_accounting_scope3.suppliers(tenant_id, engagement_tier, procurement_spend DESC)
    WHERE status = 'ACTIVE';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_sup_updated
    BEFORE UPDATE ON ghg_accounting_scope3.suppliers
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_accounting_scope3.engagement_plans
-- =============================================================================
-- Per-supplier engagement plan defining the target data quality level,
-- timeline, and activities required to move from current to target state.
-- Plans are linked to supplier records and track progress over time.

CREATE TABLE ghg_accounting_scope3.engagement_plans (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    supplier_id                 UUID            NOT NULL REFERENCES ghg_accounting_scope3.suppliers(id) ON DELETE CASCADE,
    -- Data quality targets
    current_data_quality_level  VARCHAR(20)     NOT NULL DEFAULT 'SPEND_BASED',
    target_data_quality_level   VARCHAR(20)     NOT NULL DEFAULT 'AVERAGE_DATA',
    -- Timeline
    plan_start_date             DATE            NOT NULL,
    plan_end_date               DATE            NOT NULL,
    milestones                  JSONB           DEFAULT '[]',
    -- Activities
    engagement_activities       JSONB           DEFAULT '[]',
    training_provided           BOOLEAN         DEFAULT false,
    template_shared             BOOLEAN         DEFAULT false,
    -- Progress
    status                      VARCHAR(30)     NOT NULL DEFAULT 'PLANNED',
    progress_pct                DECIMAL(5,2)    DEFAULT 0,
    last_activity_date          TIMESTAMPTZ,
    next_activity_date          TIMESTAMPTZ,
    -- Resources
    internal_owner              VARCHAR(255),
    supplier_contact            VARCHAR(255),
    estimated_effort_hours      INTEGER,
    -- Notes
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_ep_current_level CHECK (
        current_data_quality_level IN (
            'SPEND_BASED', 'AVERAGE_DATA', 'SUPPLIER_SPECIFIC', 'VERIFIED'
        )
    ),
    CONSTRAINT chk_p042_ep_target_level CHECK (
        target_data_quality_level IN (
            'SPEND_BASED', 'AVERAGE_DATA', 'SUPPLIER_SPECIFIC', 'VERIFIED'
        )
    ),
    CONSTRAINT chk_p042_ep_dates CHECK (
        plan_start_date <= plan_end_date
    ),
    CONSTRAINT chk_p042_ep_status CHECK (
        status IN (
            'PLANNED', 'IN_PROGRESS', 'ON_HOLD', 'COMPLETED',
            'CANCELLED', 'DEFERRED'
        )
    ),
    CONSTRAINT chk_p042_ep_progress CHECK (
        progress_pct >= 0 AND progress_pct <= 100
    ),
    CONSTRAINT chk_p042_ep_effort CHECK (
        estimated_effort_hours IS NULL OR estimated_effort_hours >= 0
    ),
    CONSTRAINT uq_p042_ep_supplier UNIQUE (supplier_id, plan_start_date)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_ep_tenant             ON ghg_accounting_scope3.engagement_plans(tenant_id);
CREATE INDEX idx_p042_ep_supplier           ON ghg_accounting_scope3.engagement_plans(supplier_id);
CREATE INDEX idx_p042_ep_status             ON ghg_accounting_scope3.engagement_plans(status);
CREATE INDEX idx_p042_ep_target             ON ghg_accounting_scope3.engagement_plans(target_data_quality_level);
CREATE INDEX idx_p042_ep_start              ON ghg_accounting_scope3.engagement_plans(plan_start_date);
CREATE INDEX idx_p042_ep_end                ON ghg_accounting_scope3.engagement_plans(plan_end_date);
CREATE INDEX idx_p042_ep_progress           ON ghg_accounting_scope3.engagement_plans(progress_pct);
CREATE INDEX idx_p042_ep_next_activity      ON ghg_accounting_scope3.engagement_plans(next_activity_date);
CREATE INDEX idx_p042_ep_created            ON ghg_accounting_scope3.engagement_plans(created_at DESC);

-- Composite: active plans by next activity date
CREATE INDEX idx_p042_ep_active_next        ON ghg_accounting_scope3.engagement_plans(next_activity_date)
    WHERE status IN ('PLANNED', 'IN_PROGRESS');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_ep_updated
    BEFORE UPDATE ON ghg_accounting_scope3.engagement_plans
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_accounting_scope3.data_requests
-- =============================================================================
-- Outbound data collection requests sent to suppliers. Tracks request date,
-- due date, template type, status, and reminder count. Supports multiple
-- template types (CDP-style, custom questionnaire, direct data upload).

CREATE TABLE ghg_accounting_scope3.data_requests (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    supplier_id                 UUID            NOT NULL REFERENCES ghg_accounting_scope3.suppliers(id) ON DELETE CASCADE,
    -- Request details
    request_date                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    due_date                    DATE            NOT NULL,
    reporting_period_year       INTEGER         NOT NULL,
    -- Template
    template_type               VARCHAR(30)     NOT NULL DEFAULT 'STANDARD',
    template_version            VARCHAR(20)     DEFAULT '1.0',
    template_url                TEXT,
    -- Data scope
    requested_categories        ghg_accounting_scope3.scope3_category_type[],
    requested_data_points       JSONB           DEFAULT '[]',
    data_granularity            VARCHAR(20)     DEFAULT 'ANNUAL',
    -- Communication
    sent_to_email               VARCHAR(255),
    sent_by                     VARCHAR(255),
    reminder_count              INTEGER         NOT NULL DEFAULT 0,
    last_reminder_date          TIMESTAMPTZ,
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'SENT',
    response_received           BOOLEAN         NOT NULL DEFAULT false,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_dr_year CHECK (
        reporting_period_year >= 1990 AND reporting_period_year <= 2100
    ),
    CONSTRAINT chk_p042_dr_template CHECK (
        template_type IN (
            'STANDARD', 'CDP_SUPPLY_CHAIN', 'CUSTOM_QUESTIONNAIRE',
            'DIRECT_DATA_UPLOAD', 'API_INTEGRATION', 'SIMPLIFIED'
        )
    ),
    CONSTRAINT chk_p042_dr_granularity CHECK (
        data_granularity IS NULL OR data_granularity IN (
            'ANNUAL', 'QUARTERLY', 'MONTHLY', 'PRODUCT_LEVEL'
        )
    ),
    CONSTRAINT chk_p042_dr_reminder CHECK (
        reminder_count >= 0
    ),
    CONSTRAINT chk_p042_dr_status CHECK (
        status IN (
            'DRAFT', 'SENT', 'ACKNOWLEDGED', 'IN_PROGRESS',
            'RECEIVED', 'OVERDUE', 'CANCELLED', 'EXPIRED'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_dr_tenant             ON ghg_accounting_scope3.data_requests(tenant_id);
CREATE INDEX idx_p042_dr_supplier           ON ghg_accounting_scope3.data_requests(supplier_id);
CREATE INDEX idx_p042_dr_request_date       ON ghg_accounting_scope3.data_requests(request_date DESC);
CREATE INDEX idx_p042_dr_due_date           ON ghg_accounting_scope3.data_requests(due_date);
CREATE INDEX idx_p042_dr_year               ON ghg_accounting_scope3.data_requests(reporting_period_year);
CREATE INDEX idx_p042_dr_template           ON ghg_accounting_scope3.data_requests(template_type);
CREATE INDEX idx_p042_dr_status             ON ghg_accounting_scope3.data_requests(status);
CREATE INDEX idx_p042_dr_received           ON ghg_accounting_scope3.data_requests(response_received);
CREATE INDEX idx_p042_dr_created            ON ghg_accounting_scope3.data_requests(created_at DESC);
CREATE INDEX idx_p042_dr_categories         ON ghg_accounting_scope3.data_requests USING GIN(requested_categories);

-- Composite: overdue requests for follow-up
CREATE INDEX idx_p042_dr_overdue            ON ghg_accounting_scope3.data_requests(due_date, reminder_count)
    WHERE status IN ('SENT', 'ACKNOWLEDGED', 'IN_PROGRESS', 'OVERDUE') AND response_received = false;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_dr_updated
    BEFORE UPDATE ON ghg_accounting_scope3.data_requests
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_accounting_scope3.supplier_responses
-- =============================================================================
-- Inbound supplier responses to data requests. Stores the response date,
-- data quality level of the response, reported emissions, methodology used,
-- and validation status. Responses are validated and then used to upgrade
-- category calculations from spend-based to supplier-specific.

CREATE TABLE ghg_accounting_scope3.supplier_responses (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    request_id                  UUID            NOT NULL REFERENCES ghg_accounting_scope3.data_requests(id) ON DELETE CASCADE,
    supplier_id                 UUID            NOT NULL REFERENCES ghg_accounting_scope3.suppliers(id) ON DELETE CASCADE,
    -- Response details
    response_date               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    response_format             VARCHAR(30)     DEFAULT 'QUESTIONNAIRE',
    -- Data quality
    data_quality_level          VARCHAR(20)     NOT NULL DEFAULT 'AVERAGE_DATA',
    completeness_pct            DECIMAL(5,2)    DEFAULT 100,
    -- Reported emissions
    emissions_reported_tco2e    DECIMAL(12,3),
    emissions_scope1            DECIMAL(12,3),
    emissions_scope2            DECIMAL(12,3),
    emissions_scope3_upstream   DECIMAL(12,3),
    emission_intensity          DECIMAL(12,6),
    emission_intensity_unit     VARCHAR(50),
    -- Methodology
    methodology                 VARCHAR(100),
    reporting_boundary           VARCHAR(50),
    verification_status         VARCHAR(30),
    verification_body           VARCHAR(255),
    -- Allocation
    allocated_emissions_tco2e   DECIMAL(12,3),
    allocation_method           VARCHAR(50),
    allocation_factor           DECIMAL(6,4),
    -- Validation
    validated                   BOOLEAN         NOT NULL DEFAULT false,
    validated_by                VARCHAR(255),
    validated_at                TIMESTAMPTZ,
    validation_notes            TEXT,
    -- Attachments
    attachment_paths            TEXT[],
    raw_data                    JSONB           DEFAULT '{}',
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_sres_format CHECK (
        response_format IS NULL OR response_format IN (
            'QUESTIONNAIRE', 'SPREADSHEET', 'API_DATA',
            'CDP_RESPONSE', 'SUSTAINABILITY_REPORT', 'DIRECT_ENTRY', 'OTHER'
        )
    ),
    CONSTRAINT chk_p042_sres_quality CHECK (
        data_quality_level IN (
            'SPEND_BASED', 'AVERAGE_DATA', 'SUPPLIER_SPECIFIC', 'VERIFIED'
        )
    ),
    CONSTRAINT chk_p042_sres_completeness CHECK (
        completeness_pct IS NULL OR (completeness_pct >= 0 AND completeness_pct <= 100)
    ),
    CONSTRAINT chk_p042_sres_emissions CHECK (
        emissions_reported_tco2e IS NULL OR emissions_reported_tco2e >= 0
    ),
    CONSTRAINT chk_p042_sres_scope1 CHECK (
        emissions_scope1 IS NULL OR emissions_scope1 >= 0
    ),
    CONSTRAINT chk_p042_sres_scope2 CHECK (
        emissions_scope2 IS NULL OR emissions_scope2 >= 0
    ),
    CONSTRAINT chk_p042_sres_scope3 CHECK (
        emissions_scope3_upstream IS NULL OR emissions_scope3_upstream >= 0
    ),
    CONSTRAINT chk_p042_sres_allocated CHECK (
        allocated_emissions_tco2e IS NULL OR allocated_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p042_sres_alloc_factor CHECK (
        allocation_factor IS NULL OR (allocation_factor >= 0 AND allocation_factor <= 1)
    ),
    CONSTRAINT chk_p042_sres_alloc_method CHECK (
        allocation_method IS NULL OR allocation_method IN (
            'ECONOMIC', 'PHYSICAL', 'SPEND_PROPORTIONAL',
            'PRODUCT_SPECIFIC', 'MASS_BASED', 'OTHER'
        )
    ),
    CONSTRAINT chk_p042_sres_verification CHECK (
        verification_status IS NULL OR verification_status IN (
            'NOT_VERIFIED', 'LIMITED_ASSURANCE', 'REASONABLE_ASSURANCE',
            'THIRD_PARTY_VERIFIED', 'SELF_DECLARED'
        )
    ),
    CONSTRAINT uq_p042_sres_request UNIQUE (request_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_sres_tenant           ON ghg_accounting_scope3.supplier_responses(tenant_id);
CREATE INDEX idx_p042_sres_request          ON ghg_accounting_scope3.supplier_responses(request_id);
CREATE INDEX idx_p042_sres_supplier         ON ghg_accounting_scope3.supplier_responses(supplier_id);
CREATE INDEX idx_p042_sres_date             ON ghg_accounting_scope3.supplier_responses(response_date DESC);
CREATE INDEX idx_p042_sres_quality          ON ghg_accounting_scope3.supplier_responses(data_quality_level);
CREATE INDEX idx_p042_sres_validated        ON ghg_accounting_scope3.supplier_responses(validated);
CREATE INDEX idx_p042_sres_emissions        ON ghg_accounting_scope3.supplier_responses(emissions_reported_tco2e DESC);
CREATE INDEX idx_p042_sres_verification     ON ghg_accounting_scope3.supplier_responses(verification_status);
CREATE INDEX idx_p042_sres_created          ON ghg_accounting_scope3.supplier_responses(created_at DESC);
CREATE INDEX idx_p042_sres_raw_data         ON ghg_accounting_scope3.supplier_responses USING GIN(raw_data);

-- Composite: supplier + validated for usable responses
CREATE INDEX idx_p042_sres_sup_validated    ON ghg_accounting_scope3.supplier_responses(supplier_id, response_date DESC)
    WHERE validated = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_sres_updated
    BEFORE UPDATE ON ghg_accounting_scope3.supplier_responses
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 5: ghg_accounting_scope3.engagement_metrics
-- =============================================================================
-- Aggregate engagement metrics by reporting period. Tracks the overall
-- supplier engagement program performance including response rates,
-- average data quality, and emissions coverage.

CREATE TABLE ghg_accounting_scope3.engagement_metrics (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    -- Period
    period_start                DATE            NOT NULL,
    period_end                  DATE            NOT NULL,
    reporting_year              INTEGER         NOT NULL,
    -- Supplier counts
    total_suppliers             INTEGER         NOT NULL DEFAULT 0,
    engaged_suppliers           INTEGER         NOT NULL DEFAULT 0,
    responded_suppliers         INTEGER         NOT NULL DEFAULT 0,
    response_rate_pct           DECIMAL(5,2)    GENERATED ALWAYS AS (
        CASE WHEN engaged_suppliers > 0
            THEN ROUND(((responded_suppliers::DECIMAL / engaged_suppliers) * 100)::NUMERIC, 2)
            ELSE 0
        END
    ) STORED,
    -- Data quality
    avg_data_quality_level      VARCHAR(20),
    suppliers_with_primary_data INTEGER         DEFAULT 0,
    primary_data_coverage_pct   DECIMAL(5,2),
    -- Emissions coverage
    total_scope3_tco2e          DECIMAL(15,3),
    covered_by_responses_tco2e  DECIMAL(15,3),
    coverage_pct                DECIMAL(5,2),
    -- Tier distribution
    tier_strategic_count        INTEGER         DEFAULT 0,
    tier_important_count        INTEGER         DEFAULT 0,
    tier_standard_count         INTEGER         DEFAULT 0,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_em_period CHECK (
        period_start < period_end
    ),
    CONSTRAINT chk_p042_em_year CHECK (
        reporting_year >= 1990 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p042_em_total_sup CHECK (
        total_suppliers >= 0
    ),
    CONSTRAINT chk_p042_em_engaged CHECK (
        engaged_suppliers >= 0 AND engaged_suppliers <= total_suppliers
    ),
    CONSTRAINT chk_p042_em_responded CHECK (
        responded_suppliers >= 0 AND responded_suppliers <= engaged_suppliers
    ),
    CONSTRAINT chk_p042_em_quality CHECK (
        avg_data_quality_level IS NULL OR avg_data_quality_level IN (
            'SPEND_BASED', 'AVERAGE_DATA', 'SUPPLIER_SPECIFIC', 'VERIFIED'
        )
    ),
    CONSTRAINT chk_p042_em_primary_data CHECK (
        suppliers_with_primary_data IS NULL OR suppliers_with_primary_data >= 0
    ),
    CONSTRAINT chk_p042_em_coverage_pct CHECK (
        coverage_pct IS NULL OR (coverage_pct >= 0 AND coverage_pct <= 100)
    ),
    CONSTRAINT chk_p042_em_primary_pct CHECK (
        primary_data_coverage_pct IS NULL OR (primary_data_coverage_pct >= 0 AND primary_data_coverage_pct <= 100)
    ),
    CONSTRAINT uq_p042_em_tenant_year UNIQUE (tenant_id, reporting_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_em_tenant             ON ghg_accounting_scope3.engagement_metrics(tenant_id);
CREATE INDEX idx_p042_em_year               ON ghg_accounting_scope3.engagement_metrics(reporting_year);
CREATE INDEX idx_p042_em_period             ON ghg_accounting_scope3.engagement_metrics(period_start, period_end);
CREATE INDEX idx_p042_em_created            ON ghg_accounting_scope3.engagement_metrics(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_em_updated
    BEFORE UPDATE ON ghg_accounting_scope3.engagement_metrics
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_accounting_scope3.suppliers ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.engagement_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.data_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.supplier_responses ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.engagement_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY p042_sup_tenant_isolation ON ghg_accounting_scope3.suppliers
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_sup_service_bypass ON ghg_accounting_scope3.suppliers
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_ep_tenant_isolation ON ghg_accounting_scope3.engagement_plans
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_ep_service_bypass ON ghg_accounting_scope3.engagement_plans
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_dr_tenant_isolation ON ghg_accounting_scope3.data_requests
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_dr_service_bypass ON ghg_accounting_scope3.data_requests
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_sres_tenant_isolation ON ghg_accounting_scope3.supplier_responses
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_sres_service_bypass ON ghg_accounting_scope3.supplier_responses
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_em_tenant_isolation ON ghg_accounting_scope3.engagement_metrics
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_em_service_bypass ON ghg_accounting_scope3.engagement_metrics
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.suppliers TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.engagement_plans TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.data_requests TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.supplier_responses TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.engagement_metrics TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_accounting_scope3.suppliers IS
    'Supplier master list with industry classification, procurement spend, emission contribution estimate, and engagement tier for Scope 3 data collection.';
COMMENT ON TABLE ghg_accounting_scope3.engagement_plans IS
    'Per-supplier engagement plan with data quality improvement targets, timeline, milestones, and progress tracking.';
COMMENT ON TABLE ghg_accounting_scope3.data_requests IS
    'Outbound data collection requests sent to suppliers with template type, due date, and reminder tracking.';
COMMENT ON TABLE ghg_accounting_scope3.supplier_responses IS
    'Inbound supplier responses with reported emissions, data quality level, allocation, and validation status.';
COMMENT ON TABLE ghg_accounting_scope3.engagement_metrics IS
    'Aggregate supplier engagement program metrics by reporting period including response rates and data quality coverage.';

COMMENT ON COLUMN ghg_accounting_scope3.suppliers.engagement_tier IS 'Supplier engagement priority: STRATEGIC (top 20 by spend), IMPORTANT, STANDARD, MINIMAL, INACTIVE.';
COMMENT ON COLUMN ghg_accounting_scope3.suppliers.emission_contribution_pct IS 'Estimated percentage of total Scope 3 emissions attributable to this supplier.';
COMMENT ON COLUMN ghg_accounting_scope3.suppliers.primary_scope3_categories IS 'Array of Scope 3 categories this supplier primarily contributes to.';

COMMENT ON COLUMN ghg_accounting_scope3.supplier_responses.allocated_emissions_tco2e IS 'Supplier emissions allocated to the reporting company based on allocation_method and allocation_factor.';
COMMENT ON COLUMN ghg_accounting_scope3.supplier_responses.allocation_method IS 'Method for allocating supplier emissions: ECONOMIC, PHYSICAL, SPEND_PROPORTIONAL, PRODUCT_SPECIFIC, MASS_BASED.';

COMMENT ON COLUMN ghg_accounting_scope3.engagement_metrics.response_rate_pct IS 'Generated column: (responded / engaged) * 100.';
COMMENT ON COLUMN ghg_accounting_scope3.engagement_metrics.coverage_pct IS 'Percentage of total Scope 3 emissions covered by supplier-specific responses.';
