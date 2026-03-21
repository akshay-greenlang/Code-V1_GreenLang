-- =============================================================================
-- V161: PACK-026 SME Net Zero - Grants & Certifications
-- =============================================================================
-- Pack:         PACK-026 (SME Net Zero Pack)
-- Migration:    004 of 008
-- Date:         March 2026
--
-- Grant programs directory with eligibility criteria, funding ranges, and
-- deadlines. Grant applications tracking for SMEs. Certification status
-- tracking across multiple sustainability certification pathways.
--
-- Tables (3):
--   1. pack026_sme_net_zero.grant_programs
--   2. pack026_sme_net_zero.grant_applications
--   3. pack026_sme_net_zero.certifications
--
-- Reference Data: 50+ pre-populated grant programs across EU/UK/International.
--
-- Previous: V160__PACK026_quick_wins_and_actions.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack026_sme_net_zero.grant_programs
-- =============================================================================
-- Directory of available grant programs for SME decarbonization with funding
-- ranges, eligibility criteria, deadlines, and sector applicability.

CREATE TABLE pack026_sme_net_zero.grant_programs (
    grant_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    name                    VARCHAR(500)    NOT NULL,
    description             TEXT,
    funding_body            VARCHAR(255)    NOT NULL,
    country                 VARCHAR(3)      NOT NULL,
    region                  VARCHAR(100),
    program_type            VARCHAR(50)     NOT NULL,
    -- Funding
    min_funding_eur         DECIMAL(12,2),
    max_funding_eur         DECIMAL(12,2),
    co_funding_pct          DECIMAL(6,2),
    -- Timeline
    deadline                DATE,
    opening_date            DATE,
    recurring               BOOLEAN         DEFAULT FALSE,
    recurrence_note         VARCHAR(255),
    -- Eligibility
    eligibility_criteria    JSONB           NOT NULL DEFAULT '{}',
    sector_codes            TEXT[]          DEFAULT '{}',
    size_tiers              TEXT[]          DEFAULT ARRAY['MICRO', 'SMALL', 'MEDIUM'],
    -- Categories covered
    categories_funded       TEXT[]          DEFAULT '{}',
    -- Contact
    contact_url             TEXT,
    application_url         TEXT,
    contact_email           VARCHAR(255),
    -- Status
    active                  BOOLEAN         DEFAULT TRUE,
    verified                BOOLEAN         DEFAULT FALSE,
    last_verified_date      DATE,
    sort_order              INTEGER         DEFAULT 0,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p026_gp_program_type CHECK (
        program_type IN ('GRANT', 'LOAN', 'TAX_RELIEF', 'SUBSIDY', 'VOUCHER',
                         'MATCH_FUNDING', 'COMPETITION', 'GUARANTEE', 'EQUITY', 'OTHER')
    ),
    CONSTRAINT chk_p026_gp_country_len CHECK (
        LENGTH(country) BETWEEN 2 AND 3
    ),
    CONSTRAINT chk_p026_gp_min_funding CHECK (
        min_funding_eur IS NULL OR min_funding_eur >= 0
    ),
    CONSTRAINT chk_p026_gp_max_funding CHECK (
        max_funding_eur IS NULL OR max_funding_eur >= 0
    ),
    CONSTRAINT chk_p026_gp_funding_order CHECK (
        min_funding_eur IS NULL OR max_funding_eur IS NULL OR max_funding_eur >= min_funding_eur
    ),
    CONSTRAINT chk_p026_gp_co_funding CHECK (
        co_funding_pct IS NULL OR (co_funding_pct >= 0 AND co_funding_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for grant_programs
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p026_gp_country          ON pack026_sme_net_zero.grant_programs(country);
CREATE INDEX idx_p026_gp_region           ON pack026_sme_net_zero.grant_programs(country, region);
CREATE INDEX idx_p026_gp_program_type     ON pack026_sme_net_zero.grant_programs(program_type);
CREATE INDEX idx_p026_gp_deadline         ON pack026_sme_net_zero.grant_programs(deadline);
CREATE INDEX idx_p026_gp_active           ON pack026_sme_net_zero.grant_programs(active);
CREATE INDEX idx_p026_gp_funding_body     ON pack026_sme_net_zero.grant_programs(funding_body);
CREATE INDEX idx_p026_gp_max_funding      ON pack026_sme_net_zero.grant_programs(max_funding_eur);
CREATE INDEX idx_p026_gp_created          ON pack026_sme_net_zero.grant_programs(created_at DESC);
CREATE INDEX idx_p026_gp_eligibility      ON pack026_sme_net_zero.grant_programs USING GIN(eligibility_criteria);
CREATE INDEX idx_p026_gp_sectors          ON pack026_sme_net_zero.grant_programs USING GIN(sector_codes);
CREATE INDEX idx_p026_gp_size_tiers       ON pack026_sme_net_zero.grant_programs USING GIN(size_tiers);
CREATE INDEX idx_p026_gp_categories       ON pack026_sme_net_zero.grant_programs USING GIN(categories_funded);
CREATE INDEX idx_p026_gp_metadata         ON pack026_sme_net_zero.grant_programs USING GIN(metadata);

-- =============================================================================
-- Table 2: pack026_sme_net_zero.grant_applications
-- =============================================================================
-- SME grant application tracking with status workflow from draft to approval
-- and funding disbursement.

CREATE TABLE pack026_sme_net_zero.grant_applications (
    application_id          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sme_id                  UUID            NOT NULL REFERENCES pack026_sme_net_zero.sme_profiles(sme_id) ON DELETE CASCADE,
    grant_id                UUID            NOT NULL REFERENCES pack026_sme_net_zero.grant_programs(grant_id) ON DELETE RESTRICT,
    tenant_id               UUID            NOT NULL,
    -- Application
    application_date        DATE,
    submission_date         DATE,
    reference_number        VARCHAR(100),
    status                  VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    -- Funding
    funding_requested_eur   DECIMAL(12,2),
    funding_approved_eur    DECIMAL(12,2),
    funding_disbursed_eur   DECIMAL(12,2),
    -- Actions funded
    action_ids              UUID[]          DEFAULT '{}',
    project_description     TEXT,
    -- Timeline
    decision_date           DATE,
    project_start_date      DATE,
    project_end_date        DATE,
    reporting_deadline      DATE,
    -- Notes
    assessor_feedback       TEXT,
    rejection_reason        TEXT,
    notes                   TEXT,
    -- Metadata
    warnings                TEXT[]          DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p026_ga_status CHECK (
        status IN ('DRAFT', 'SUBMITTED', 'UNDER_REVIEW', 'APPROVED', 'REJECTED',
                   'WITHDRAWN', 'DISBURSED', 'COMPLETED', 'EXPIRED')
    ),
    CONSTRAINT chk_p026_ga_requested CHECK (
        funding_requested_eur IS NULL OR funding_requested_eur >= 0
    ),
    CONSTRAINT chk_p026_ga_approved CHECK (
        funding_approved_eur IS NULL OR funding_approved_eur >= 0
    ),
    CONSTRAINT chk_p026_ga_disbursed CHECK (
        funding_disbursed_eur IS NULL OR funding_disbursed_eur >= 0
    ),
    CONSTRAINT chk_p026_ga_approved_le_requested CHECK (
        funding_approved_eur IS NULL OR funding_requested_eur IS NULL
        OR funding_approved_eur <= funding_requested_eur
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for grant_applications
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p026_ga_sme              ON pack026_sme_net_zero.grant_applications(sme_id);
CREATE INDEX idx_p026_ga_grant            ON pack026_sme_net_zero.grant_applications(grant_id);
CREATE INDEX idx_p026_ga_tenant           ON pack026_sme_net_zero.grant_applications(tenant_id);
CREATE INDEX idx_p026_ga_status           ON pack026_sme_net_zero.grant_applications(status);
CREATE INDEX idx_p026_ga_app_date         ON pack026_sme_net_zero.grant_applications(application_date);
CREATE INDEX idx_p026_ga_sme_status       ON pack026_sme_net_zero.grant_applications(sme_id, status);
CREATE INDEX idx_p026_ga_reference        ON pack026_sme_net_zero.grant_applications(reference_number);
CREATE INDEX idx_p026_ga_created          ON pack026_sme_net_zero.grant_applications(created_at DESC);
CREATE INDEX idx_p026_ga_action_ids       ON pack026_sme_net_zero.grant_applications USING GIN(action_ids);
CREATE INDEX idx_p026_ga_metadata         ON pack026_sme_net_zero.grant_applications USING GIN(metadata);

-- =============================================================================
-- Table 3: pack026_sme_net_zero.certifications
-- =============================================================================
-- SME sustainability certification tracking across multiple pathways
-- including SME Climate Hub, B Corp, ISO 14001, Carbon Trust, and Climate Active.

CREATE TABLE pack026_sme_net_zero.certifications (
    cert_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sme_id                  UUID            NOT NULL REFERENCES pack026_sme_net_zero.sme_profiles(sme_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    -- Certification details
    certification_type      VARCHAR(30)     NOT NULL,
    certification_body      VARCHAR(255),
    level                   VARCHAR(50),
    certificate_number      VARCHAR(100),
    -- Status
    status                  VARCHAR(20)     NOT NULL DEFAULT 'NOT_STARTED',
    -- Dates
    application_date        DATE,
    submission_date         DATE,
    verification_date       DATE,
    issue_date              DATE,
    expiry_date             DATE,
    renewal_date            DATE,
    -- Deliverables
    badge_url               TEXT,
    certificate_url         TEXT,
    public_listing_url      TEXT,
    -- Assessment
    readiness_score         DECIMAL(6,2),
    checklist_completion    DECIMAL(6,2),
    requirements_met        INTEGER         DEFAULT 0,
    requirements_total      INTEGER,
    -- Cost
    certification_cost_eur  DECIMAL(12,2),
    annual_fee_eur          DECIMAL(12,2),
    -- Metadata
    notes                   TEXT,
    warnings                TEXT[]          DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p026_cert_type CHECK (
        certification_type IN ('SME_CLIMATE_HUB', 'B_CORP', 'ISO14001', 'CARBON_TRUST',
                               'CLIMATE_ACTIVE', 'PLANET_MARK', 'TOITU', 'SBTi_SME',
                               'ECOVADIS', 'CDP', 'OTHER')
    ),
    CONSTRAINT chk_p026_cert_status CHECK (
        status IN ('NOT_STARTED', 'IN_PROGRESS', 'SUBMITTED', 'UNDER_REVIEW',
                   'VERIFIED', 'CERTIFIED', 'EXPIRED', 'REVOKED', 'RENEWAL_DUE')
    ),
    CONSTRAINT chk_p026_cert_readiness CHECK (
        readiness_score IS NULL OR (readiness_score >= 0 AND readiness_score <= 100)
    ),
    CONSTRAINT chk_p026_cert_checklist CHECK (
        checklist_completion IS NULL OR (checklist_completion >= 0 AND checklist_completion <= 100)
    ),
    CONSTRAINT chk_p026_cert_cost CHECK (
        certification_cost_eur IS NULL OR certification_cost_eur >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for certifications
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p026_cert_sme            ON pack026_sme_net_zero.certifications(sme_id);
CREATE INDEX idx_p026_cert_tenant         ON pack026_sme_net_zero.certifications(tenant_id);
CREATE INDEX idx_p026_cert_type           ON pack026_sme_net_zero.certifications(certification_type);
CREATE INDEX idx_p026_cert_status         ON pack026_sme_net_zero.certifications(status);
CREATE INDEX idx_p026_cert_sme_type       ON pack026_sme_net_zero.certifications(sme_id, certification_type);
CREATE INDEX idx_p026_cert_expiry         ON pack026_sme_net_zero.certifications(expiry_date);
CREATE INDEX idx_p026_cert_issue          ON pack026_sme_net_zero.certifications(issue_date);
CREATE INDEX idx_p026_cert_created        ON pack026_sme_net_zero.certifications(created_at DESC);
CREATE INDEX idx_p026_cert_metadata       ON pack026_sme_net_zero.certifications USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p026_grant_programs_updated
    BEFORE UPDATE ON pack026_sme_net_zero.grant_programs
    FOR EACH ROW EXECUTE FUNCTION pack026_sme_net_zero.fn_set_updated_at();

CREATE TRIGGER trg_p026_grant_applications_updated
    BEFORE UPDATE ON pack026_sme_net_zero.grant_applications
    FOR EACH ROW EXECUTE FUNCTION pack026_sme_net_zero.fn_set_updated_at();

CREATE TRIGGER trg_p026_certifications_updated
    BEFORE UPDATE ON pack026_sme_net_zero.certifications
    FOR EACH ROW EXECUTE FUNCTION pack026_sme_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
-- grant_programs is shared reference data (no tenant_id), so no RLS
ALTER TABLE pack026_sme_net_zero.grant_applications ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack026_sme_net_zero.certifications ENABLE ROW LEVEL SECURITY;

CREATE POLICY p026_ga_tenant_isolation
    ON pack026_sme_net_zero.grant_applications
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p026_ga_service_bypass
    ON pack026_sme_net_zero.grant_applications
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p026_cert_tenant_isolation
    ON pack026_sme_net_zero.certifications
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p026_cert_service_bypass
    ON pack026_sme_net_zero.certifications
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT ON pack026_sme_net_zero.grant_programs TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack026_sme_net_zero.grant_applications TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack026_sme_net_zero.certifications TO PUBLIC;

-- =============================================================================
-- Pre-populated Grant Programs Reference Data (52 entries)
-- =============================================================================

-- ----------------------------- UK GRANTS (1-15) ------------------------------
INSERT INTO pack026_sme_net_zero.grant_programs
    (name, description, funding_body, country, region, program_type, min_funding_eur, max_funding_eur, co_funding_pct, deadline, eligibility_criteria, sector_codes, size_tiers, categories_funded, active, sort_order)
VALUES
    ('UK Industrial Energy Transformation Fund (IETF)', 'Support for industrial decarbonization through energy efficiency and deep decarbonization projects.', 'UK BEIS / DESNZ', 'GB', NULL, 'GRANT', 115000.00, 17250000.00, 50.00, '2026-09-30', '{"max_employees": 250, "sectors": ["manufacturing", "industrial"], "min_energy_spend": 50000}', ARRAY['C10-C33'], ARRAY['SMALL', 'MEDIUM'], ARRAY['ENERGY', 'HEATING', 'BUILDING'], TRUE, 1),
    ('UK SME Climate Hub Commitment', 'Free commitment platform for SMEs to pledge halving emissions by 2030 and reaching net zero by 2050.', 'SME Climate Hub', 'GB', NULL, 'OTHER', 0.00, 0.00, 0.00, NULL, '{"max_employees": 250, "all_sectors": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['BEHAVIOUR'], TRUE, 2),
    ('UK Green Business Fund', 'Free energy audits and up to GBP 10,000 in match funding for energy efficiency improvements.', 'Carbon Trust', 'GB', NULL, 'MATCH_FUNDING', 0.00, 11500.00, 50.00, '2026-12-31', '{"max_employees": 250, "all_sectors": true, "energy_spend_range": [5000, 500000]}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'LIGHTING', 'HEATING'], TRUE, 3),
    ('UK Workplace Charging Scheme', 'Voucher for up to 75% of EV charge point installation costs (capped at GBP 350/socket).', 'OZEV', 'GB', NULL, 'VOUCHER', 0.00, 4025.00, 75.00, NULL, '{"all_sectors": true, "min_sockets": 1, "max_sockets": 40}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['TRANSPORT'], TRUE, 4),
    ('UK Low Carbon Heating Grant', 'Boiler Upgrade Scheme providing GBP 5,000-6,000 towards heat pump installation.', 'Ofgem', 'GB', NULL, 'GRANT', 5750.00, 6900.00, 0.00, '2028-03-31', '{"property_type": "non_domestic", "existing_fossil_fuel_heating": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['HEATING'], TRUE, 5),
    ('Scotland SME Loan Fund', 'Interest-free loans up to GBP 100,000 for energy efficiency and renewable energy.', 'Scottish Government / EST', 'GB', 'Scotland', 'LOAN', 1150.00, 115000.00, 0.00, NULL, '{"country": "Scotland", "all_sectors": true, "energy_efficiency_focus": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'HEATING', 'BUILDING', 'LIGHTING'], TRUE, 6),
    ('Wales SME Green Fund', 'Support for Welsh SMEs to reduce carbon footprint with expert advice and funding.', 'Welsh Government', 'GB', 'Wales', 'GRANT', 1150.00, 57500.00, 50.00, NULL, '{"country": "Wales", "all_sectors": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'TRANSPORT', 'BUILDING'], TRUE, 7),
    ('UK Enhanced Capital Allowances', '100% first-year tax relief on qualifying energy-efficient equipment.', 'HMRC', 'GB', NULL, 'TAX_RELIEF', 0.00, NULL, 100.00, NULL, '{"all_sectors": true, "equipment_on_etl": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'HEATING', 'LIGHTING', 'REFRIGERATION'], TRUE, 8),
    ('UK Innovate UK Smart Grants', 'R&D funding for innovative clean technology and net-zero solutions.', 'Innovate UK', 'GB', NULL, 'GRANT', 28750.00, 575000.00, 70.00, '2026-06-30', '{"innovation_focus": true, "technology_readiness": "TRL4+"}', ARRAY['ALL'], ARRAY['SMALL', 'MEDIUM'], ARRAY['ENERGY', 'DIGITAL', 'TRANSPORT'], TRUE, 9),
    ('UK Community Energy Fund', 'Support for community-owned renewable energy projects with SME participation.', 'UK Community Renewal Fund', 'GB', NULL, 'GRANT', 5750.00, 115000.00, 50.00, NULL, '{"community_benefit": true, "renewable_energy": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY'], TRUE, 10),
    ('UK EV Fleet Transition Grant', 'Support for SMEs transitioning commercial fleets to electric vehicles.', 'DfT / OZEV', 'GB', NULL, 'GRANT', 2875.00, 57500.00, 35.00, '2026-12-31', '{"fleet_size_min": 2, "commercial_vehicles": true}', ARRAY['ALL'], ARRAY['SMALL', 'MEDIUM'], ARRAY['TRANSPORT'], TRUE, 11),
    ('UK Net Zero Innovation Portfolio', 'DESNZ innovation funding for novel decarbonization technologies.', 'DESNZ', 'GB', NULL, 'GRANT', 57500.00, 1150000.00, 50.00, '2026-09-30', '{"innovation_focus": true, "net_zero_contribution": true}', ARRAY['ALL'], ARRAY['SMALL', 'MEDIUM'], ARRAY['ENERGY', 'HEATING', 'BUILDING'], TRUE, 12),
    ('Northern Ireland Energy Efficiency Loan', 'Interest-free loans for NI businesses investing in energy efficiency.', 'Invest NI', 'GB', 'Northern Ireland', 'LOAN', 1150.00, 460000.00, 0.00, NULL, '{"country": "Northern Ireland", "energy_saving_min_pct": 10}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'HEATING', 'LIGHTING'], TRUE, 13),
    ('UK Circular Economy Fund', 'Funding for waste reduction, reuse, and circular business model innovation.', 'WRAP / Defra', 'GB', NULL, 'GRANT', 5750.00, 230000.00, 50.00, '2026-06-30', '{"circular_economy_focus": true}', ARRAY['ALL'], ARRAY['SMALL', 'MEDIUM'], ARRAY['WASTE', 'PROCUREMENT'], TRUE, 14),
    ('UK Carbon Literacy Training Subsidy', 'Subsidized carbon literacy certification for SME staff teams.', 'Carbon Literacy Project', 'GB', NULL, 'SUBSIDY', 0.00, 2300.00, 75.00, NULL, '{"all_sectors": true, "min_participants": 5}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['BEHAVIOUR'], TRUE, 15);

-- ----------------------------- EU GRANTS (16-30) -----------------------------
INSERT INTO pack026_sme_net_zero.grant_programs
    (name, description, funding_body, country, region, program_type, min_funding_eur, max_funding_eur, co_funding_pct, deadline, eligibility_criteria, sector_codes, size_tiers, categories_funded, active, sort_order)
VALUES
    ('EU LIFE Clean Energy Transition', 'EU LIFE programme supporting clean energy transition projects for SMEs.', 'European Commission', 'EU', NULL, 'GRANT', 500000.00, 3000000.00, 40.00, '2026-09-15', '{"eu_member_state": true, "clean_energy_focus": true}', ARRAY['ALL'], ARRAY['SMALL', 'MEDIUM'], ARRAY['ENERGY', 'HEATING', 'BUILDING'], TRUE, 16),
    ('EU Horizon Europe EIC Accelerator', 'Grants and equity for SMEs with breakthrough clean technologies.', 'European Innovation Council', 'EU', NULL, 'GRANT', 500000.00, 2500000.00, 70.00, '2026-10-01', '{"eu_member_state": true, "innovation_focus": true, "technology_readiness": "TRL6+"}', ARRAY['ALL'], ARRAY['SMALL', 'MEDIUM'], ARRAY['ENERGY', 'DIGITAL', 'TRANSPORT'], TRUE, 17),
    ('EU Innovation Fund Small Scale', 'EU ETS Innovation Fund for small-scale clean technology projects.', 'European Commission', 'EU', NULL, 'GRANT', 2500000.00, 7500000.00, 60.00, '2026-11-15', '{"eu_member_state": true, "ghg_reduction_potential": true}', ARRAY['ALL'], ARRAY['MEDIUM'], ARRAY['ENERGY', 'HEATING', 'BUILDING'], TRUE, 18),
    ('EU Cohesion Fund Green SME', 'Regional development funding for SME green transition in less developed regions.', 'European Regional Development Fund', 'EU', NULL, 'GRANT', 10000.00, 500000.00, 50.00, NULL, '{"eu_cohesion_region": true, "sme_status": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'TRANSPORT', 'BUILDING'], TRUE, 19),
    ('Germany KfW Climate Action SME', 'Low-interest loans for SME climate action including energy efficiency and renewables.', 'KfW', 'DE', NULL, 'LOAN', 25000.00, 25000000.00, 0.00, NULL, '{"country": "Germany", "climate_action_plan": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'HEATING', 'BUILDING', 'TRANSPORT'], TRUE, 20),
    ('France Plan de Relance Eco-Energie', 'French recovery fund supporting SME energy audits and efficiency investments.', 'ADEME / Bpifrance', 'FR', NULL, 'GRANT', 5000.00, 200000.00, 50.00, '2026-12-31', '{"country": "France", "energy_audit_completed": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'HEATING', 'BUILDING'], TRUE, 21),
    ('Netherlands SDE++ Renewable Energy', 'Operating subsidy for renewable energy and CO2 reduction projects.', 'RVO Netherlands', 'NL', NULL, 'SUBSIDY', 25000.00, 10000000.00, 0.00, '2026-10-01', '{"country": "Netherlands", "renewable_energy_or_co2_reduction": true}', ARRAY['ALL'], ARRAY['SMALL', 'MEDIUM'], ARRAY['ENERGY', 'HEATING'], TRUE, 22),
    ('Spain PERTE Decarbonization', 'Spanish recovery funding for industrial decarbonization projects.', 'IDAE Spain', 'ES', NULL, 'GRANT', 50000.00, 5000000.00, 40.00, '2026-09-30', '{"country": "Spain", "industrial_decarbonization": true}', ARRAY['C10-C33'], ARRAY['SMALL', 'MEDIUM'], ARRAY['ENERGY', 'HEATING', 'BUILDING'], TRUE, 23),
    ('Italy Transizione 5.0 Tax Credit', 'Tax credits for Italian SMEs investing in energy-efficient technologies.', 'Italian Government', 'IT', NULL, 'TAX_RELIEF', 0.00, 5000000.00, 45.00, '2026-12-31', '{"country": "Italy", "energy_saving_min_pct": 3}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'DIGITAL', 'HEATING'], TRUE, 24),
    ('Belgium Flanders Ecology Premium', 'Flemish subsidy for SME investments in environmental technologies.', 'VLAIO', 'BE', 'Flanders', 'SUBSIDY', 3000.00, 1000000.00, 30.00, NULL, '{"country": "Belgium", "region": "Flanders", "ecology_investment": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'WASTE', 'WATER'], TRUE, 25),
    ('Austria Environmental Support Scheme', 'Austrian federal support for SME environmental and climate protection investments.', 'Austrian Federal Ministry', 'AT', NULL, 'GRANT', 5000.00, 500000.00, 30.00, NULL, '{"country": "Austria", "environmental_benefit": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'HEATING', 'TRANSPORT', 'BUILDING'], TRUE, 26),
    ('Sweden Climate Leap (Klimatklivet)', 'Local climate investment support for Swedish SMEs.', 'Swedish EPA (Naturvardsverket)', 'SE', NULL, 'GRANT', 10000.00, 5000000.00, 50.00, '2026-08-31', '{"country": "Sweden", "ghg_reduction_measurable": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'TRANSPORT', 'HEATING'], TRUE, 27),
    ('Denmark Green Business Fund', 'Danish business fund for green transition and climate technology adoption.', 'Danish Business Authority', 'DK', NULL, 'GRANT', 10000.00, 1000000.00, 50.00, '2026-10-31', '{"country": "Denmark", "green_transition_focus": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'DIGITAL', 'TRANSPORT'], TRUE, 28),
    ('Finland Energy Aid (Energiatuki)', 'Finnish energy aid for renewable energy and energy efficiency investments.', 'Business Finland / Motiva', 'FI', NULL, 'GRANT', 5000.00, 2000000.00, 40.00, NULL, '{"country": "Finland", "energy_focus": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'HEATING'], TRUE, 29),
    ('Ireland SEAI SME Energy Program', 'Energy audits, training, and capital grants for Irish SMEs.', 'SEAI Ireland', 'IE', NULL, 'GRANT', 2000.00, 500000.00, 30.00, NULL, '{"country": "Ireland", "energy_audit_first": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'HEATING', 'LIGHTING', 'BUILDING'], TRUE, 30);

-- ----------------------------- INTERNATIONAL GRANTS (31-40) ------------------
INSERT INTO pack026_sme_net_zero.grant_programs
    (name, description, funding_body, country, region, program_type, min_funding_eur, max_funding_eur, co_funding_pct, deadline, eligibility_criteria, sector_codes, size_tiers, categories_funded, active, sort_order)
VALUES
    ('EBRD Green Economy Transition', 'European Bank for Reconstruction and Development green financing for SMEs in target countries.', 'EBRD', 'EU', NULL, 'LOAN', 50000.00, 5000000.00, 0.00, NULL, '{"ebrd_country": true, "green_investment": true}', ARRAY['ALL'], ARRAY['SMALL', 'MEDIUM'], ARRAY['ENERGY', 'TRANSPORT', 'BUILDING'], TRUE, 31),
    ('IFC Green SME Finance Facility', 'International Finance Corporation green credit lines through partner banks for SME sustainability.', 'IFC / World Bank', 'INT', NULL, 'LOAN', 10000.00, 1000000.00, 0.00, NULL, '{"developing_country": true, "sme_status": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'TRANSPORT', 'BUILDING'], TRUE, 32),
    ('GEF Small Grants Programme', 'UNDP/GEF small grants for community-based environmental projects including SME participation.', 'UNDP / GEF', 'INT', NULL, 'GRANT', 5000.00, 50000.00, 50.00, NULL, '{"community_benefit": true, "environmental_focus": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL'], ARRAY['ENERGY', 'WASTE', 'WATER'], TRUE, 33),
    ('GGGI Green Growth SME Fund', 'Global Green Growth Institute support for SMEs in green growth pathways.', 'GGGI', 'INT', NULL, 'GRANT', 10000.00, 250000.00, 40.00, NULL, '{"gggi_partner_country": true, "green_growth_potential": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'SUPPLY_CHAIN'], TRUE, 34),
    ('Canada CEBA Green Retrofit', 'Canadian SME green retrofit loans with partial forgiveness.', 'Export Development Canada', 'CA', NULL, 'LOAN', 7000.00, 75000.00, 0.00, NULL, '{"country": "Canada", "green_retrofit": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'HEATING', 'BUILDING'], TRUE, 35),
    ('Australia Clean Energy Finance SME', 'CEFC low-interest loans for Australian SME clean energy investments.', 'CEFC Australia', 'AU', NULL, 'LOAN', 15000.00, 1500000.00, 0.00, NULL, '{"country": "Australia", "clean_energy_investment": true}', ARRAY['ALL'], ARRAY['SMALL', 'MEDIUM'], ARRAY['ENERGY', 'TRANSPORT'], TRUE, 36),
    ('New Zealand GIDI Fund', 'Government Investment in Decarbonising Industry fund for process heat conversion.', 'EECA New Zealand', 'NZ', NULL, 'GRANT', 25000.00, 5000000.00, 40.00, '2026-08-31', '{"country": "New Zealand", "process_heat_conversion": true}', ARRAY['C10-C33'], ARRAY['SMALL', 'MEDIUM'], ARRAY['HEATING', 'ENERGY'], TRUE, 37),
    ('Japan GX Green Transformation SME', 'Japanese support for SME green transformation investments.', 'METI Japan', 'JP', NULL, 'SUBSIDY', 10000.00, 500000.00, 50.00, NULL, '{"country": "Japan", "gx_transformation": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'DIGITAL'], TRUE, 38),
    ('South Korea Green New Deal SME', 'Korean government support for SME participation in the Green New Deal.', 'Korean Government', 'KR', NULL, 'GRANT', 10000.00, 300000.00, 50.00, NULL, '{"country": "South Korea", "green_new_deal_aligned": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'DIGITAL', 'TRANSPORT'], TRUE, 39),
    ('Singapore Enterprise Sustainability Programme', 'Singapore government support for SME sustainability capabilities.', 'Enterprise Singapore', 'SG', NULL, 'GRANT', 5000.00, 200000.00, 70.00, NULL, '{"country": "Singapore", "sustainability_capability": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'DIGITAL', 'SUPPLY_CHAIN'], TRUE, 40);

-- ----------------------------- SECTOR-SPECIFIC GRANTS (41-52) ----------------
INSERT INTO pack026_sme_net_zero.grant_programs
    (name, description, funding_body, country, region, program_type, min_funding_eur, max_funding_eur, co_funding_pct, deadline, eligibility_criteria, sector_codes, size_tiers, categories_funded, active, sort_order)
VALUES
    ('EU Farm to Fork SME Fund', 'Support for food and agriculture SMEs reducing emissions in the supply chain.', 'European Commission', 'EU', NULL, 'GRANT', 50000.00, 2000000.00, 50.00, '2026-11-30', '{"eu_member_state": true, "food_agriculture_sector": true}', ARRAY['A01', 'A02', 'C10', 'C11'], ARRAY['SMALL', 'MEDIUM'], ARRAY['SUPPLY_CHAIN', 'WASTE', 'ENERGY'], TRUE, 41),
    ('UK Construction Net Zero Fund', 'Support for construction SMEs transitioning to low-carbon materials and methods.', 'CITB / UK Government', 'GB', NULL, 'GRANT', 5750.00, 115000.00, 50.00, '2026-12-31', '{"construction_sector": true}', ARRAY['F41', 'F42', 'F43'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['PROCUREMENT', 'TRANSPORT', 'BUILDING'], TRUE, 42),
    ('EU Textile Circular Economy Fund', 'Funding for textile and fashion SMEs adopting circular economy practices.', 'European Commission', 'EU', NULL, 'GRANT', 25000.00, 500000.00, 60.00, '2026-10-15', '{"textile_fashion_sector": true, "circular_economy_focus": true}', ARRAY['C13', 'C14', 'C15'], ARRAY['SMALL', 'MEDIUM'], ARRAY['PROCUREMENT', 'WASTE', 'SUPPLY_CHAIN'], TRUE, 43),
    ('UK Hospitality Green Recovery', 'Support for hospitality SMEs investing in energy efficiency and waste reduction.', 'UK Hospitality / Carbon Trust', 'GB', NULL, 'GRANT', 1150.00, 23000.00, 50.00, NULL, '{"hospitality_sector": true}', ARRAY['I55', 'I56'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'WASTE', 'WATER'], TRUE, 44),
    ('EU Digital Green SME Programme', 'Support for SMEs using digital technologies for sustainability improvements.', 'European Commission', 'EU', NULL, 'GRANT', 10000.00, 200000.00, 75.00, '2026-09-30', '{"eu_member_state": true, "digital_sustainability": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['DIGITAL', 'ENERGY'], TRUE, 45),
    ('UK Retail Decarbonization Fund', 'Grants for retail SMEs investing in energy-efficient refrigeration and lighting.', 'BRC / Carbon Trust', 'GB', NULL, 'GRANT', 2300.00, 57500.00, 40.00, NULL, '{"retail_sector": true}', ARRAY['G47'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['REFRIGERATION', 'LIGHTING', 'ENERGY'], TRUE, 46),
    ('EU Automotive Supply Chain Green Fund', 'Support for SME suppliers in the automotive chain to decarbonize operations.', 'European Commission', 'EU', NULL, 'GRANT', 100000.00, 2000000.00, 50.00, '2026-11-30', '{"automotive_supply_chain": true}', ARRAY['C29', 'C30', 'C22', 'C24', 'C25'], ARRAY['SMALL', 'MEDIUM'], ARRAY['ENERGY', 'TRANSPORT', 'SUPPLY_CHAIN'], TRUE, 47),
    ('UK Health Sector Greener NHS Fund', 'Support for healthcare SME suppliers aligned with NHS net-zero commitments.', 'NHS England / Carbon Trust', 'GB', NULL, 'GRANT', 5750.00, 115000.00, 50.00, NULL, '{"nhs_supplier": true, "health_sector": true}', ARRAY['Q86', 'Q87'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'TRANSPORT', 'PROCUREMENT'], TRUE, 48),
    ('EU Chemicals Green Transition Fund', 'Support for chemicals SMEs transitioning to greener processes and products.', 'European Commission', 'EU', NULL, 'GRANT', 50000.00, 1000000.00, 50.00, '2026-10-31', '{"chemicals_sector": true, "green_chemistry_focus": true}', ARRAY['C20', 'C21'], ARRAY['SMALL', 'MEDIUM'], ARRAY['ENERGY', 'WASTE', 'SUPPLY_CHAIN'], TRUE, 49),
    ('UK Professional Services Net Zero', 'Support for professional services firms to measure and reduce emissions.', 'Various UK Bodies', 'GB', NULL, 'VOUCHER', 575.00, 5750.00, 75.00, NULL, '{"professional_services": true}', ARRAY['M69', 'M70', 'M71', 'M72', 'M73'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['DIGITAL', 'TRANSPORT', 'BEHAVIOUR'], TRUE, 50),
    ('EU Horizon Europe Clean Steel Partnership', 'Support for steel and metals SMEs in clean technology adoption.', 'European Commission', 'EU', NULL, 'GRANT', 200000.00, 5000000.00, 60.00, '2026-09-15', '{"steel_metals_sector": true}', ARRAY['C24', 'C25'], ARRAY['MEDIUM'], ARRAY['ENERGY', 'HEATING'], TRUE, 51),
    ('Global SME Climate Alliance Fund', 'International coalition funding for SME climate action across developing economies.', 'UN Global Compact / UNFCCC', 'INT', NULL, 'GRANT', 5000.00, 100000.00, 50.00, NULL, '{"un_global_compact_member": true, "developing_country": true}', ARRAY['ALL'], ARRAY['MICRO', 'SMALL', 'MEDIUM'], ARRAY['ENERGY', 'TRANSPORT', 'SUPPLY_CHAIN'], TRUE, 52);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack026_sme_net_zero.grant_programs IS
    'Directory of 52 grant and funding programs for SME decarbonization across UK, EU, and international sources.';
COMMENT ON TABLE pack026_sme_net_zero.grant_applications IS
    'SME grant application tracking with status workflow from draft to approval and funding disbursement.';
COMMENT ON TABLE pack026_sme_net_zero.certifications IS
    'SME sustainability certification tracking across multiple pathways (SME Climate Hub, B Corp, ISO 14001, Carbon Trust, Climate Active).';

COMMENT ON COLUMN pack026_sme_net_zero.grant_programs.grant_id IS 'Unique grant program identifier.';
COMMENT ON COLUMN pack026_sme_net_zero.grant_programs.program_type IS 'Funding type: GRANT, LOAN, TAX_RELIEF, SUBSIDY, VOUCHER, MATCH_FUNDING, COMPETITION, GUARANTEE, EQUITY, OTHER.';
COMMENT ON COLUMN pack026_sme_net_zero.grant_programs.eligibility_criteria IS 'JSONB eligibility requirements for grant applications.';
COMMENT ON COLUMN pack026_sme_net_zero.grant_programs.sector_codes IS 'Array of applicable NACE sector codes (ALL = universal).';
COMMENT ON COLUMN pack026_sme_net_zero.grant_programs.size_tiers IS 'Array of eligible SME size tiers (MICRO, SMALL, MEDIUM).';
COMMENT ON COLUMN pack026_sme_net_zero.grant_applications.application_id IS 'Unique grant application identifier.';
COMMENT ON COLUMN pack026_sme_net_zero.grant_applications.status IS 'Application status: DRAFT, SUBMITTED, UNDER_REVIEW, APPROVED, REJECTED, WITHDRAWN, DISBURSED, COMPLETED, EXPIRED.';
COMMENT ON COLUMN pack026_sme_net_zero.certifications.cert_id IS 'Unique certification identifier.';
COMMENT ON COLUMN pack026_sme_net_zero.certifications.certification_type IS 'Certification type: SME_CLIMATE_HUB, B_CORP, ISO14001, CARBON_TRUST, CLIMATE_ACTIVE, PLANET_MARK, TOITU, SBTi_SME, ECOVADIS, CDP.';
COMMENT ON COLUMN pack026_sme_net_zero.certifications.status IS 'Certification status: NOT_STARTED, IN_PROGRESS, SUBMITTED, UNDER_REVIEW, VERIFIED, CERTIFIED, EXPIRED, REVOKED, RENEWAL_DUE.';
COMMENT ON COLUMN pack026_sme_net_zero.certifications.badge_url IS 'URL to downloadable certification badge/logo.';
COMMENT ON COLUMN pack026_sme_net_zero.certifications.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
