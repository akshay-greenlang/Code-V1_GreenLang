-- =============================================================================
-- V160: PACK-026 SME Net Zero - Quick Wins Library & Selected Actions
-- =============================================================================
-- Pack:         PACK-026 (SME Net Zero Pack)
-- Migration:    003 of 008
-- Date:         March 2026
--
-- Pre-populated library of 50+ quick-win decarbonization actions with cost,
-- savings, payback, and sector applicability. Selected actions table tracks
-- SME implementation progress with actual vs estimated outcomes.
--
-- Tables (2):
--   1. pack026_sme_net_zero.quick_wins_library
--   2. pack026_sme_net_zero.selected_actions
--
-- Reference Data: 55 pre-populated quick wins across energy, transport,
--   procurement, waste, digital, and water categories.
--
-- Previous: V159__PACK026_baselines_and_targets.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack026_sme_net_zero.quick_wins_library
-- =============================================================================
-- Pre-populated library of quick-win decarbonization actions with estimated
-- cost, annual savings, payback period, difficulty rating, and sector
-- applicability. Reference data for all tenants.

CREATE TABLE pack026_sme_net_zero.quick_wins_library (
    win_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    name                    VARCHAR(255)    NOT NULL,
    description             TEXT,
    category                VARCHAR(50)     NOT NULL,
    scope                   VARCHAR(10)     NOT NULL,
    -- Emission reduction
    reduction_tco2e         DECIMAL(12,4),
    reduction_pct           DECIMAL(6,2),
    -- Cost and savings
    cost_gbp                DECIMAL(12,2),
    cost_eur                DECIMAL(12,2),
    annual_savings_gbp      DECIMAL(12,2),
    annual_savings_eur      DECIMAL(12,2),
    payback_years           DECIMAL(6,2),
    -- Implementation
    difficulty              VARCHAR(10)     NOT NULL DEFAULT 'EASY',
    implementation_weeks    INTEGER,
    -- Applicability
    sector_applicability    TEXT[]          NOT NULL DEFAULT '{}',
    size_tier_applicability TEXT[]          DEFAULT ARRAY['MICRO', 'SMALL', 'MEDIUM'],
    prerequisites           TEXT,
    -- Funding
    grant_eligible          BOOLEAN         DEFAULT FALSE,
    grant_programs          TEXT[],
    -- Quality
    evidence_strength       VARCHAR(20)     DEFAULT 'MODERATE',
    source_reference        TEXT,
    -- Status
    active                  BOOLEAN         DEFAULT TRUE,
    sort_order              INTEGER         DEFAULT 0,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p026_qw_category CHECK (
        category IN ('ENERGY', 'TRANSPORT', 'PROCUREMENT', 'WASTE', 'DIGITAL',
                     'WATER', 'BUILDING', 'HEATING', 'LIGHTING', 'REFRIGERATION',
                     'BEHAVIOUR', 'SUPPLY_CHAIN', 'OFFSET')
    ),
    CONSTRAINT chk_p026_qw_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'MULTI')
    ),
    CONSTRAINT chk_p026_qw_difficulty CHECK (
        difficulty IN ('EASY', 'MODERATE', 'HARD')
    ),
    CONSTRAINT chk_p026_qw_evidence CHECK (
        evidence_strength IS NULL OR evidence_strength IN ('STRONG', 'MODERATE', 'INDICATIVE')
    ),
    CONSTRAINT chk_p026_qw_cost_non_neg CHECK (
        cost_gbp IS NULL OR cost_gbp >= 0
    ),
    CONSTRAINT chk_p026_qw_savings_non_neg CHECK (
        annual_savings_gbp IS NULL OR annual_savings_gbp >= 0
    ),
    CONSTRAINT chk_p026_qw_payback_non_neg CHECK (
        payback_years IS NULL OR payback_years >= 0
    ),
    CONSTRAINT chk_p026_qw_reduction_pct CHECK (
        reduction_pct IS NULL OR (reduction_pct >= 0 AND reduction_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for quick_wins_library
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p026_qw_category         ON pack026_sme_net_zero.quick_wins_library(category);
CREATE INDEX idx_p026_qw_scope            ON pack026_sme_net_zero.quick_wins_library(scope);
CREATE INDEX idx_p026_qw_difficulty       ON pack026_sme_net_zero.quick_wins_library(difficulty);
CREATE INDEX idx_p026_qw_payback          ON pack026_sme_net_zero.quick_wins_library(payback_years);
CREATE INDEX idx_p026_qw_grant_eligible   ON pack026_sme_net_zero.quick_wins_library(grant_eligible);
CREATE INDEX idx_p026_qw_active           ON pack026_sme_net_zero.quick_wins_library(active);
CREATE INDEX idx_p026_qw_sort             ON pack026_sme_net_zero.quick_wins_library(sort_order);
CREATE INDEX idx_p026_qw_created          ON pack026_sme_net_zero.quick_wins_library(created_at DESC);
CREATE INDEX idx_p026_qw_sectors          ON pack026_sme_net_zero.quick_wins_library USING GIN(sector_applicability);
CREATE INDEX idx_p026_qw_size_tiers       ON pack026_sme_net_zero.quick_wins_library USING GIN(size_tier_applicability);
CREATE INDEX idx_p026_qw_grants           ON pack026_sme_net_zero.quick_wins_library USING GIN(grant_programs);
CREATE INDEX idx_p026_qw_metadata         ON pack026_sme_net_zero.quick_wins_library USING GIN(metadata);

-- =============================================================================
-- Table 2: pack026_sme_net_zero.selected_actions
-- =============================================================================
-- SME-selected decarbonization actions from the quick wins library with
-- implementation status tracking and actual vs estimated outcome comparison.

CREATE TABLE pack026_sme_net_zero.selected_actions (
    action_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sme_id                  UUID            NOT NULL REFERENCES pack026_sme_net_zero.sme_profiles(sme_id) ON DELETE CASCADE,
    quick_win_id            UUID            NOT NULL REFERENCES pack026_sme_net_zero.quick_wins_library(win_id) ON DELETE RESTRICT,
    tenant_id               UUID            NOT NULL,
    -- Implementation
    implementation_status   VARCHAR(20)     NOT NULL DEFAULT 'PLANNED',
    priority                VARCHAR(10)     DEFAULT 'MEDIUM',
    start_date              DATE,
    target_completion_date  DATE,
    completion_date         DATE,
    -- Estimated outcomes
    estimated_cost_eur      DECIMAL(12,2),
    estimated_savings_eur   DECIMAL(12,2),
    estimated_reduction_tco2e DECIMAL(12,4),
    -- Actual outcomes
    actual_cost             DECIMAL(12,2),
    actual_savings          DECIMAL(12,2),
    actual_reduction_tco2e  DECIMAL(12,4),
    -- Notes
    implementation_notes    TEXT,
    barriers_encountered    TEXT,
    lessons_learned         TEXT,
    -- Supplier / contractor
    supplier_name           VARCHAR(255),
    quote_reference         VARCHAR(100),
    -- Metadata
    warnings                TEXT[]          DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p026_sa_status CHECK (
        implementation_status IN ('PLANNED', 'IN_PROGRESS', 'COMPLETED', 'CANCELLED', 'ON_HOLD')
    ),
    CONSTRAINT chk_p026_sa_priority CHECK (
        priority IS NULL OR priority IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p026_sa_cost_non_neg CHECK (
        actual_cost IS NULL OR actual_cost >= 0
    ),
    CONSTRAINT chk_p026_sa_savings_non_neg CHECK (
        actual_savings IS NULL OR actual_savings >= 0
    ),
    CONSTRAINT chk_p026_sa_reduction_non_neg CHECK (
        actual_reduction_tco2e IS NULL OR actual_reduction_tco2e >= 0
    ),
    CONSTRAINT chk_p026_sa_dates CHECK (
        start_date IS NULL OR completion_date IS NULL OR completion_date >= start_date
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for selected_actions
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p026_sa_sme              ON pack026_sme_net_zero.selected_actions(sme_id);
CREATE INDEX idx_p026_sa_quick_win        ON pack026_sme_net_zero.selected_actions(quick_win_id);
CREATE INDEX idx_p026_sa_tenant           ON pack026_sme_net_zero.selected_actions(tenant_id);
CREATE INDEX idx_p026_sa_status           ON pack026_sme_net_zero.selected_actions(implementation_status);
CREATE INDEX idx_p026_sa_priority         ON pack026_sme_net_zero.selected_actions(priority);
CREATE INDEX idx_p026_sa_sme_status       ON pack026_sme_net_zero.selected_actions(sme_id, implementation_status);
CREATE INDEX idx_p026_sa_start_date       ON pack026_sme_net_zero.selected_actions(start_date);
CREATE INDEX idx_p026_sa_completion       ON pack026_sme_net_zero.selected_actions(completion_date);
CREATE INDEX idx_p026_sa_created          ON pack026_sme_net_zero.selected_actions(created_at DESC);
CREATE INDEX idx_p026_sa_metadata         ON pack026_sme_net_zero.selected_actions USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p026_quick_wins_updated
    BEFORE UPDATE ON pack026_sme_net_zero.quick_wins_library
    FOR EACH ROW EXECUTE FUNCTION pack026_sme_net_zero.fn_set_updated_at();

CREATE TRIGGER trg_p026_selected_actions_updated
    BEFORE UPDATE ON pack026_sme_net_zero.selected_actions
    FOR EACH ROW EXECUTE FUNCTION pack026_sme_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
-- quick_wins_library is shared reference data (no tenant_id), so no RLS
-- selected_actions is tenant-scoped
ALTER TABLE pack026_sme_net_zero.selected_actions ENABLE ROW LEVEL SECURITY;

CREATE POLICY p026_sa_tenant_isolation
    ON pack026_sme_net_zero.selected_actions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p026_sa_service_bypass
    ON pack026_sme_net_zero.selected_actions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT ON pack026_sme_net_zero.quick_wins_library TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack026_sme_net_zero.selected_actions TO PUBLIC;

-- =============================================================================
-- Pre-populated Quick Wins Reference Data (55 entries)
-- =============================================================================

-- ----------------------------- ENERGY (1-10) ---------------------------------
INSERT INTO pack026_sme_net_zero.quick_wins_library
    (name, description, category, scope, reduction_tco2e, reduction_pct, cost_gbp, cost_eur, annual_savings_gbp, annual_savings_eur, payback_years, difficulty, implementation_weeks, sector_applicability, grant_eligible, evidence_strength, sort_order)
VALUES
    ('LED lighting retrofit', 'Replace all conventional lighting with LED equivalents across office and warehouse spaces.', 'LIGHTING', 'SCOPE_2', 2.50, 5.00, 2000.00, 2300.00, 800.00, 920.00, 2.50, 'EASY', 4, ARRAY['ALL'], TRUE, 'STRONG', 1),
    ('Smart thermostat installation', 'Install programmable smart thermostats with occupancy sensing and scheduling.', 'HEATING', 'SCOPE_1', 1.80, 4.00, 500.00, 575.00, 400.00, 460.00, 1.25, 'EASY', 2, ARRAY['ALL'], TRUE, 'STRONG', 2),
    ('Renewable electricity PPA', 'Switch to 100% renewable electricity via a Power Purchase Agreement or green tariff.', 'ENERGY', 'SCOPE_2', 15.00, 30.00, 0.00, 0.00, 200.00, 230.00, 0.00, 'EASY', 2, ARRAY['ALL'], FALSE, 'STRONG', 3),
    ('HVAC optimization and maintenance', 'Professional HVAC audit, filter replacement, duct sealing, and system optimization.', 'HEATING', 'SCOPE_1', 3.20, 6.00, 1500.00, 1725.00, 600.00, 690.00, 2.50, 'MODERATE', 6, ARRAY['ALL'], TRUE, 'STRONG', 4),
    ('Building insulation upgrade', 'Install or upgrade wall, roof, and floor insulation to reduce heating demand.', 'BUILDING', 'SCOPE_1', 4.50, 8.00, 5000.00, 5750.00, 1200.00, 1380.00, 4.17, 'MODERATE', 8, ARRAY['ALL'], TRUE, 'STRONG', 5),
    ('Solar PV installation (rooftop)', 'Install rooftop solar photovoltaic panels for on-site electricity generation.', 'ENERGY', 'SCOPE_2', 8.00, 15.00, 12000.00, 13800.00, 2000.00, 2300.00, 6.00, 'HARD', 12, ARRAY['ALL'], TRUE, 'STRONG', 6),
    ('Motion sensor lighting controls', 'Install PIR motion sensors in low-occupancy areas (toilets, storerooms, corridors).', 'LIGHTING', 'SCOPE_2', 1.00, 2.00, 300.00, 345.00, 200.00, 230.00, 1.50, 'EASY', 2, ARRAY['ALL'], FALSE, 'STRONG', 7),
    ('Double/triple glazing windows', 'Replace single-glazed windows with double or triple glazing to reduce heat loss.', 'BUILDING', 'SCOPE_1', 2.80, 5.00, 8000.00, 9200.00, 700.00, 805.00, 11.43, 'HARD', 8, ARRAY['ALL'], TRUE, 'MODERATE', 8),
    ('Heat pump installation (air source)', 'Replace gas boiler with air source heat pump for space heating and hot water.', 'HEATING', 'SCOPE_1', 6.00, 12.00, 10000.00, 11500.00, 1500.00, 1725.00, 6.67, 'HARD', 10, ARRAY['ALL'], TRUE, 'STRONG', 9),
    ('Energy monitoring system', 'Install sub-metering and energy monitoring dashboard for real-time consumption tracking.', 'ENERGY', 'MULTI', 2.00, 4.00, 1000.00, 1150.00, 500.00, 575.00, 2.00, 'EASY', 3, ARRAY['ALL'], TRUE, 'STRONG', 10);

-- ----------------------------- TRANSPORT (11-20) -----------------------------
INSERT INTO pack026_sme_net_zero.quick_wins_library
    (name, description, category, scope, reduction_tco2e, reduction_pct, cost_gbp, cost_eur, annual_savings_gbp, annual_savings_eur, payback_years, difficulty, implementation_weeks, sector_applicability, grant_eligible, evidence_strength, sort_order)
VALUES
    ('EV company car transition', 'Replace petrol/diesel company cars with battery electric vehicles at lease renewal.', 'TRANSPORT', 'SCOPE_1', 4.00, 8.00, 0.00, 0.00, 1200.00, 1380.00, 0.00, 'MODERATE', 12, ARRAY['ALL'], TRUE, 'STRONG', 11),
    ('EV charging points on-site', 'Install workplace EV charging infrastructure for staff and fleet vehicles.', 'TRANSPORT', 'SCOPE_1', 1.50, 3.00, 3000.00, 3450.00, 200.00, 230.00, 15.00, 'MODERATE', 8, ARRAY['ALL'], TRUE, 'MODERATE', 12),
    ('Cycle-to-work scheme', 'Implement salary sacrifice cycle-to-work scheme with secure bike storage and showers.', 'TRANSPORT', 'SCOPE_3', 0.80, 1.50, 500.00, 575.00, 100.00, 115.00, 5.00, 'EASY', 4, ARRAY['ALL'], FALSE, 'MODERATE', 13),
    ('Remote/hybrid work policy', 'Formalize hybrid working policy to reduce commuting emissions by 2-3 days per week.', 'TRANSPORT', 'SCOPE_3', 2.50, 5.00, 0.00, 0.00, 300.00, 345.00, 0.00, 'EASY', 2, ARRAY['OFFICE', 'TECH', 'SERVICES', 'PROFESSIONAL'], FALSE, 'MODERATE', 14),
    ('Fleet telematics and eco-driving', 'Install GPS telematics and eco-driving training to optimize fleet fuel consumption.', 'TRANSPORT', 'SCOPE_1', 3.00, 6.00, 2000.00, 2300.00, 1500.00, 1725.00, 1.33, 'MODERATE', 6, ARRAY['LOGISTICS', 'DELIVERY', 'CONSTRUCTION', 'MANUFACTURING'], TRUE, 'STRONG', 15),
    ('Consolidate deliveries', 'Optimize delivery routes and consolidate shipments to reduce van/truck miles.', 'TRANSPORT', 'SCOPE_1', 2.00, 4.00, 500.00, 575.00, 800.00, 920.00, 0.63, 'EASY', 4, ARRAY['LOGISTICS', 'DELIVERY', 'RETAIL', 'WHOLESALE'], FALSE, 'MODERATE', 16),
    ('Electric van/light commercial', 'Replace diesel vans with electric equivalents for urban delivery routes.', 'TRANSPORT', 'SCOPE_1', 5.00, 10.00, 5000.00, 5750.00, 2500.00, 2875.00, 2.00, 'HARD', 12, ARRAY['LOGISTICS', 'DELIVERY', 'CONSTRUCTION'], TRUE, 'STRONG', 17),
    ('Video conferencing over travel', 'Replace non-essential business travel with video conferencing tools.', 'TRANSPORT', 'SCOPE_3', 1.50, 3.00, 200.00, 230.00, 2000.00, 2300.00, 0.10, 'EASY', 1, ARRAY['ALL'], FALSE, 'STRONG', 18),
    ('Public transport subsidies', 'Provide subsidized public transport passes for employee commuting.', 'TRANSPORT', 'SCOPE_3', 1.00, 2.00, 3000.00, 3450.00, 0.00, 0.00, NULL, 'EASY', 4, ARRAY['ALL'], FALSE, 'MODERATE', 19),
    ('Car-sharing program', 'Implement car-sharing and ride-pooling for employees with similar commute routes.', 'TRANSPORT', 'SCOPE_3', 0.60, 1.20, 100.00, 115.00, 150.00, 172.50, 0.67, 'EASY', 4, ARRAY['ALL'], FALSE, 'INDICATIVE', 20);

-- ----------------------------- PROCUREMENT (21-30) ---------------------------
INSERT INTO pack026_sme_net_zero.quick_wins_library
    (name, description, category, scope, reduction_tco2e, reduction_pct, cost_gbp, cost_eur, annual_savings_gbp, annual_savings_eur, payback_years, difficulty, implementation_weeks, sector_applicability, grant_eligible, evidence_strength, sort_order)
VALUES
    ('Supplier sustainability criteria', 'Add emissions and sustainability scoring to procurement evaluation criteria.', 'SUPPLY_CHAIN', 'SCOPE_3', 3.00, 6.00, 0.00, 0.00, 0.00, 0.00, 0.00, 'MODERATE', 8, ARRAY['ALL'], FALSE, 'MODERATE', 21),
    ('Local sourcing policy', 'Prioritize local suppliers within 50km radius to reduce transportation emissions.', 'SUPPLY_CHAIN', 'SCOPE_3', 1.50, 3.00, 0.00, 0.00, 200.00, 230.00, 0.00, 'EASY', 6, ARRAY['ALL'], FALSE, 'MODERATE', 22),
    ('Recycled materials procurement', 'Switch to recycled content materials where quality and cost are comparable.', 'PROCUREMENT', 'SCOPE_3', 2.00, 4.00, 500.00, 575.00, 100.00, 115.00, 5.00, 'MODERATE', 8, ARRAY['MANUFACTURING', 'CONSTRUCTION', 'RETAIL'], FALSE, 'MODERATE', 23),
    ('Paper-free office', 'Eliminate unnecessary printing with digital-first document management.', 'PROCUREMENT', 'SCOPE_3', 0.30, 0.60, 0.00, 0.00, 500.00, 575.00, 0.00, 'EASY', 4, ARRAY['ALL'], FALSE, 'MODERATE', 24),
    ('Green cleaning products', 'Switch to eco-certified, low-carbon cleaning products and services.', 'PROCUREMENT', 'SCOPE_3', 0.20, 0.40, 100.00, 115.00, 0.00, 0.00, NULL, 'EASY', 2, ARRAY['ALL'], FALSE, 'INDICATIVE', 25),
    ('Refurbished IT equipment', 'Purchase refurbished laptops, monitors, and peripherals instead of new.', 'PROCUREMENT', 'SCOPE_3', 1.00, 2.00, 0.00, 0.00, 2000.00, 2300.00, 0.00, 'EASY', 4, ARRAY['ALL'], FALSE, 'MODERATE', 26),
    ('Sustainable packaging switch', 'Replace plastic packaging with recyclable/compostable alternatives.', 'PROCUREMENT', 'SCOPE_3', 0.80, 1.50, 1000.00, 1150.00, 0.00, 0.00, NULL, 'MODERATE', 8, ARRAY['RETAIL', 'WHOLESALE', 'MANUFACTURING', 'FOOD'], FALSE, 'MODERATE', 27),
    ('Bulk purchasing optimization', 'Consolidate orders to reduce delivery frequency and packaging waste.', 'PROCUREMENT', 'SCOPE_3', 0.50, 1.00, 0.00, 0.00, 300.00, 345.00, 0.00, 'EASY', 4, ARRAY['ALL'], FALSE, 'INDICATIVE', 28),
    ('Low-carbon catering policy', 'Reduce meat-heavy options, increase plant-based catering for meetings and canteen.', 'PROCUREMENT', 'SCOPE_3', 0.40, 0.80, 0.00, 0.00, 100.00, 115.00, 0.00, 'EASY', 2, ARRAY['ALL'], FALSE, 'MODERATE', 29),
    ('Supplier engagement program', 'Run annual supplier engagement program to collect Scope 3 emissions data.', 'SUPPLY_CHAIN', 'SCOPE_3', 5.00, 10.00, 2000.00, 2300.00, 0.00, 0.00, NULL, 'HARD', 16, ARRAY['ALL'], FALSE, 'MODERATE', 30);

-- ----------------------------- WASTE (31-37) ---------------------------------
INSERT INTO pack026_sme_net_zero.quick_wins_library
    (name, description, category, scope, reduction_tco2e, reduction_pct, cost_gbp, cost_eur, annual_savings_gbp, annual_savings_eur, payback_years, difficulty, implementation_weeks, sector_applicability, grant_eligible, evidence_strength, sort_order)
VALUES
    ('Waste segregation and recycling', 'Implement comprehensive waste segregation bins with clear signage and training.', 'WASTE', 'SCOPE_3', 0.50, 1.00, 200.00, 230.00, 300.00, 345.00, 0.67, 'EASY', 4, ARRAY['ALL'], FALSE, 'STRONG', 31),
    ('Composting organic waste', 'Divert food and organic waste to on-site or local composting facilities.', 'WASTE', 'SCOPE_3', 0.30, 0.60, 300.00, 345.00, 100.00, 115.00, 3.00, 'EASY', 4, ARRAY['FOOD', 'HOSPITALITY', 'RETAIL'], FALSE, 'MODERATE', 32),
    ('Zero waste-to-landfill target', 'Set and work towards zero waste-to-landfill with waste audit and diversion plan.', 'WASTE', 'SCOPE_3', 1.00, 2.00, 500.00, 575.00, 400.00, 460.00, 1.25, 'MODERATE', 12, ARRAY['ALL'], FALSE, 'MODERATE', 33),
    ('Reusable packaging return scheme', 'Implement returnable packaging for B2B deliveries and customer orders.', 'WASTE', 'SCOPE_3', 0.60, 1.20, 2000.00, 2300.00, 500.00, 575.00, 4.00, 'MODERATE', 12, ARRAY['RETAIL', 'WHOLESALE', 'FOOD', 'MANUFACTURING'], FALSE, 'MODERATE', 34),
    ('Water-efficient fixtures', 'Install low-flow taps, dual-flush toilets, and water-efficient appliances.', 'WATER', 'SCOPE_3', 0.10, 0.20, 500.00, 575.00, 200.00, 230.00, 2.50, 'EASY', 4, ARRAY['ALL'], FALSE, 'MODERATE', 35),
    ('Reduce food waste (if applicable)', 'Implement food waste tracking, portion control, and surplus redistribution.', 'WASTE', 'SCOPE_3', 0.80, 1.50, 200.00, 230.00, 1000.00, 1150.00, 0.20, 'EASY', 6, ARRAY['FOOD', 'HOSPITALITY'], FALSE, 'STRONG', 36),
    ('E-waste recycling program', 'Establish certified e-waste recycling for electronics, batteries, and toner.', 'WASTE', 'SCOPE_3', 0.20, 0.40, 100.00, 115.00, 0.00, 0.00, NULL, 'EASY', 2, ARRAY['ALL'], FALSE, 'MODERATE', 37);

-- ----------------------------- DIGITAL (38-42) -------------------------------
INSERT INTO pack026_sme_net_zero.quick_wins_library
    (name, description, category, scope, reduction_tco2e, reduction_pct, cost_gbp, cost_eur, annual_savings_gbp, annual_savings_eur, payback_years, difficulty, implementation_weeks, sector_applicability, grant_eligible, evidence_strength, sort_order)
VALUES
    ('Green web hosting', 'Migrate website to a green-certified hosting provider powered by renewables.', 'DIGITAL', 'SCOPE_3', 0.30, 0.60, 100.00, 115.00, 0.00, 0.00, NULL, 'EASY', 2, ARRAY['ALL'], FALSE, 'MODERATE', 38),
    ('Cloud optimization', 'Right-size cloud instances and enable auto-scaling to reduce idle compute.', 'DIGITAL', 'SCOPE_3', 0.50, 1.00, 0.00, 0.00, 500.00, 575.00, 0.00, 'MODERATE', 4, ARRAY['TECH', 'SERVICES', 'PROFESSIONAL'], FALSE, 'MODERATE', 39),
    ('Email cleanup and data hygiene', 'Reduce stored emails, delete unused data, and optimize cloud storage footprint.', 'DIGITAL', 'SCOPE_3', 0.10, 0.20, 0.00, 0.00, 50.00, 57.50, 0.00, 'EASY', 2, ARRAY['ALL'], FALSE, 'INDICATIVE', 40),
    ('Energy-efficient IT equipment', 'Replace old desktops with energy-efficient laptops and enable power management.', 'DIGITAL', 'SCOPE_2', 0.80, 1.50, 3000.00, 3450.00, 400.00, 460.00, 7.50, 'MODERATE', 8, ARRAY['ALL'], FALSE, 'MODERATE', 41),
    ('Server room cooling optimization', 'Optimize server room airflow, set temperature to 27C, use blanking panels.', 'DIGITAL', 'SCOPE_2', 1.00, 2.00, 500.00, 575.00, 600.00, 690.00, 0.83, 'MODERATE', 4, ARRAY['TECH', 'SERVICES', 'PROFESSIONAL'], FALSE, 'MODERATE', 42);

-- ----------------------------- REFRIGERATION (43-45) -------------------------
INSERT INTO pack026_sme_net_zero.quick_wins_library
    (name, description, category, scope, reduction_tco2e, reduction_pct, cost_gbp, cost_eur, annual_savings_gbp, annual_savings_eur, payback_years, difficulty, implementation_weeks, sector_applicability, grant_eligible, evidence_strength, sort_order)
VALUES
    ('Low-GWP refrigerant transition', 'Replace high-GWP F-gas refrigerants (R-404A) with low-GWP alternatives (R-290, CO2).', 'REFRIGERATION', 'SCOPE_1', 5.00, 10.00, 4000.00, 4600.00, 500.00, 575.00, 8.00, 'HARD', 12, ARRAY['FOOD', 'HOSPITALITY', 'RETAIL'], TRUE, 'STRONG', 43),
    ('Refrigeration door/lid installation', 'Add doors or lids to open display refrigerators and freezers.', 'REFRIGERATION', 'SCOPE_2', 2.00, 4.00, 3000.00, 3450.00, 800.00, 920.00, 3.75, 'MODERATE', 6, ARRAY['FOOD', 'RETAIL'], FALSE, 'STRONG', 44),
    ('Refrigerant leak detection system', 'Install automatic leak detection to reduce F-gas losses and maintenance costs.', 'REFRIGERATION', 'SCOPE_1', 3.00, 6.00, 2000.00, 2300.00, 600.00, 690.00, 3.33, 'MODERATE', 4, ARRAY['FOOD', 'HOSPITALITY', 'RETAIL', 'PHARMACEUTICAL'], TRUE, 'STRONG', 45);

-- ----------------------------- BEHAVIOUR (46-50) -----------------------------
INSERT INTO pack026_sme_net_zero.quick_wins_library
    (name, description, category, scope, reduction_tco2e, reduction_pct, cost_gbp, cost_eur, annual_savings_gbp, annual_savings_eur, payback_years, difficulty, implementation_weeks, sector_applicability, grant_eligible, evidence_strength, sort_order)
VALUES
    ('Staff sustainability training', 'Deliver carbon literacy training to all staff with practical action plans.', 'BEHAVIOUR', 'MULTI', 1.50, 3.00, 500.00, 575.00, 300.00, 345.00, 1.67, 'EASY', 4, ARRAY['ALL'], TRUE, 'MODERATE', 46),
    ('Green champion network', 'Appoint departmental green champions to drive behaviour change and ideas.', 'BEHAVIOUR', 'MULTI', 0.50, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 'EASY', 2, ARRAY['ALL'], FALSE, 'INDICATIVE', 47),
    ('Switch-off campaign', 'Run awareness campaign for switching off lights, screens, and equipment when not in use.', 'BEHAVIOUR', 'SCOPE_2', 0.80, 1.50, 0.00, 0.00, 200.00, 230.00, 0.00, 'EASY', 2, ARRAY['ALL'], FALSE, 'MODERATE', 48),
    ('Temperature setpoint policy', 'Set heating to max 19C and cooling to min 24C with seasonal adjustment policy.', 'BEHAVIOUR', 'SCOPE_1', 1.20, 2.50, 0.00, 0.00, 400.00, 460.00, 0.00, 'EASY', 1, ARRAY['ALL'], FALSE, 'STRONG', 49),
    ('Sustainability in onboarding', 'Include net-zero goals and individual actions in employee onboarding process.', 'BEHAVIOUR', 'MULTI', 0.30, 0.60, 0.00, 0.00, 0.00, 0.00, 0.00, 'EASY', 2, ARRAY['ALL'], FALSE, 'INDICATIVE', 50);

-- ----------------------------- OFFSET (51-55) --------------------------------
INSERT INTO pack026_sme_net_zero.quick_wins_library
    (name, description, category, scope, reduction_tco2e, reduction_pct, cost_gbp, cost_eur, annual_savings_gbp, annual_savings_eur, payback_years, difficulty, implementation_weeks, sector_applicability, grant_eligible, evidence_strength, sort_order)
VALUES
    ('Tree planting / rewilding', 'Sponsor local tree planting or rewilding projects for residual carbon sequestration.', 'OFFSET', 'MULTI', 2.00, 4.00, 1000.00, 1150.00, 0.00, 0.00, NULL, 'EASY', 4, ARRAY['ALL'], TRUE, 'MODERATE', 51),
    ('Verified carbon credit purchase', 'Purchase Gold Standard or Verra-verified carbon credits for residual emissions.', 'OFFSET', 'MULTI', 5.00, 10.00, 2500.00, 2875.00, 0.00, 0.00, NULL, 'EASY', 2, ARRAY['ALL'], FALSE, 'STRONG', 52),
    ('Biodiversity net gain project', 'Invest in local biodiversity net gain projects with carbon co-benefits.', 'OFFSET', 'MULTI', 1.00, 2.00, 2000.00, 2300.00, 0.00, 0.00, NULL, 'MODERATE', 8, ARRAY['ALL'], TRUE, 'MODERATE', 53),
    ('Peatland restoration', 'Support peatland restoration projects for high-impact carbon sequestration.', 'OFFSET', 'MULTI', 3.00, 6.00, 1500.00, 1725.00, 0.00, 0.00, NULL, 'MODERATE', 8, ARRAY['ALL'], TRUE, 'STRONG', 54),
    ('Insetting (supply chain)', 'Fund emission reduction projects within own supply chain as insetting initiatives.', 'OFFSET', 'SCOPE_3', 4.00, 8.00, 5000.00, 5750.00, 0.00, 0.00, NULL, 'HARD', 16, ARRAY['ALL'], FALSE, 'MODERATE', 55);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack026_sme_net_zero.quick_wins_library IS
    'Pre-populated library of 55 quick-win decarbonization actions with cost, savings, payback, difficulty, and sector applicability for SMEs.';
COMMENT ON TABLE pack026_sme_net_zero.selected_actions IS
    'SME-selected decarbonization actions from the quick wins library with implementation status tracking and actual vs estimated outcome comparison.';

COMMENT ON COLUMN pack026_sme_net_zero.quick_wins_library.win_id IS 'Unique quick win identifier.';
COMMENT ON COLUMN pack026_sme_net_zero.quick_wins_library.category IS 'Action category: ENERGY, TRANSPORT, PROCUREMENT, WASTE, DIGITAL, WATER, BUILDING, HEATING, LIGHTING, REFRIGERATION, BEHAVIOUR, SUPPLY_CHAIN, OFFSET.';
COMMENT ON COLUMN pack026_sme_net_zero.quick_wins_library.scope IS 'GHG scope affected: SCOPE_1, SCOPE_2, SCOPE_3, MULTI.';
COMMENT ON COLUMN pack026_sme_net_zero.quick_wins_library.reduction_tco2e IS 'Estimated annual emission reduction in tCO2e (typical SME).';
COMMENT ON COLUMN pack026_sme_net_zero.quick_wins_library.payback_years IS 'Simple payback period in years (NULL = no financial payback).';
COMMENT ON COLUMN pack026_sme_net_zero.quick_wins_library.difficulty IS 'Implementation difficulty: EASY, MODERATE, HARD.';
COMMENT ON COLUMN pack026_sme_net_zero.quick_wins_library.sector_applicability IS 'Array of applicable sectors (ALL = universal).';
COMMENT ON COLUMN pack026_sme_net_zero.quick_wins_library.grant_eligible IS 'Whether this action is typically eligible for government grants or subsidies.';
COMMENT ON COLUMN pack026_sme_net_zero.selected_actions.action_id IS 'Unique selected action identifier.';
COMMENT ON COLUMN pack026_sme_net_zero.selected_actions.implementation_status IS 'Action status: PLANNED, IN_PROGRESS, COMPLETED, CANCELLED, ON_HOLD.';
COMMENT ON COLUMN pack026_sme_net_zero.selected_actions.actual_reduction_tco2e IS 'Actual measured emission reduction after completion (tCO2e).';
