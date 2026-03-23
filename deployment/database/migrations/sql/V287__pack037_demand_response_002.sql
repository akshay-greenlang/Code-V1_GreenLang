-- =============================================================================
-- V287: PACK-037 Demand Response Pack - DR Program Database
-- =============================================================================
-- Pack:         PACK-037 (Demand Response Pack)
-- Migration:    002 of 010
-- Date:         March 2026
--
-- DR program definitions, requirements, compensation structures, penalties,
-- enrollment tracking, and eligibility rules. Covers ISO/RTO programs across
-- PJM, ERCOT, CAISO, ISO-NE, NYISO, MISO and European markets (UK, DE,
-- FR, NL). Seed data provides 200+ real-world program definitions.
--
-- Tables (6):
--   1. pack037_demand_response.dr_programs
--   2. pack037_demand_response.dr_program_requirements
--   3. pack037_demand_response.dr_program_compensation
--   4. pack037_demand_response.dr_program_penalties
--   5. pack037_demand_response.dr_program_enrollment
--   6. pack037_demand_response.dr_program_eligibility
--
-- Previous: V286__pack037_demand_response_001.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack037_demand_response.dr_programs
-- =============================================================================
-- Master list of demand response programs offered by ISOs, RTOs, utilities,
-- aggregators, and European TSOs. Each program defines the market type,
-- dispatch method, season, and product characteristics.

CREATE TABLE pack037_demand_response.dr_programs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    program_code            VARCHAR(50)     NOT NULL UNIQUE,
    program_name            VARCHAR(255)    NOT NULL,
    iso_rto_region          VARCHAR(30)     NOT NULL,
    program_sponsor         VARCHAR(255)    NOT NULL,
    program_type            VARCHAR(50)     NOT NULL,
    market_type             VARCHAR(30)     NOT NULL,
    dispatch_method         VARCHAR(30)     NOT NULL,
    season                  VARCHAR(20)     NOT NULL DEFAULT 'ALL_YEAR',
    notification_lead_time_min INTEGER      NOT NULL DEFAULT 60,
    min_curtailment_kw      NUMERIC(12,4)   NOT NULL DEFAULT 100,
    max_event_duration_hours NUMERIC(6,2)   NOT NULL DEFAULT 4,
    max_events_per_year     INTEGER,
    max_hours_per_year      INTEGER,
    measurement_verification VARCHAR(50)    NOT NULL DEFAULT 'BASELINE_CBL',
    baseline_methodology    VARCHAR(50),
    telemetry_required      BOOLEAN         DEFAULT false,
    aggregation_allowed     BOOLEAN         DEFAULT true,
    min_aggregation_kw      NUMERIC(12,4),
    effective_date          DATE            NOT NULL,
    expiration_date         DATE,
    program_url             TEXT,
    program_status          VARCHAR(20)     NOT NULL DEFAULT 'ACTIVE',
    country_code            CHAR(2)         NOT NULL DEFAULT 'US',
    currency_code           CHAR(3)         NOT NULL DEFAULT 'USD',
    description             TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_prg_region CHECK (
        iso_rto_region IN (
            'PJM', 'ERCOT', 'CAISO', 'ISO_NE', 'NYISO', 'MISO', 'SPP',
            'UK_NGESO', 'DE_TENNET', 'DE_AMPRION', 'DE_50HZ', 'DE_TRANSNET',
            'FR_RTE', 'NL_TENNET', 'ES_REE', 'IT_TERNA', 'AU_AEMO',
            'JP_TEPCO', 'JP_KEPCO', 'OTHER'
        )
    ),
    CONSTRAINT chk_p037_prg_type CHECK (
        program_type IN (
            'CAPACITY', 'ENERGY', 'ANCILLARY', 'EMERGENCY', 'ECONOMIC',
            'RELIABILITY', 'FREQUENCY_REGULATION', 'RESERVES',
            'PEAK_SHAVING', 'LOAD_SHIFTING', 'INTERRUPTIBLE',
            'CRITICAL_PEAK_PRICING', 'REAL_TIME_PRICING', 'TIME_OF_USE',
            'AGGREGATED_FLEXIBILITY', 'VOLTAGE_SUPPORT'
        )
    ),
    CONSTRAINT chk_p037_prg_market CHECK (
        market_type IN (
            'WHOLESALE', 'RETAIL', 'BILATERAL', 'CAPACITY_MARKET',
            'ENERGY_MARKET', 'ANCILLARY_SERVICES', 'BALANCING_MARKET'
        )
    ),
    CONSTRAINT chk_p037_prg_dispatch CHECK (
        dispatch_method IN (
            'MANUAL_CALL', 'AUTO_DISPATCH', 'PRICE_SIGNAL', 'OPENADR',
            'DIRECT_LOAD_CONTROL', 'AGGREGATOR_MANAGED'
        )
    ),
    CONSTRAINT chk_p037_prg_season CHECK (
        season IN ('SUMMER', 'WINTER', 'SHOULDER', 'ALL_YEAR')
    ),
    CONSTRAINT chk_p037_prg_mv CHECK (
        measurement_verification IN (
            'BASELINE_CBL', 'BASELINE_ADJUSTED', 'METERING_GENERATOR',
            'WHOLE_FACILITY', 'SUB_METERING', 'DEEMED_SAVINGS',
            'STATISTICAL_SAMPLING', 'REGRESSION'
        )
    ),
    CONSTRAINT chk_p037_prg_status CHECK (
        program_status IN ('ACTIVE', 'UPCOMING', 'SUSPENDED', 'CLOSED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p037_prg_min_curt CHECK (
        min_curtailment_kw >= 0
    ),
    CONSTRAINT chk_p037_prg_max_dur CHECK (
        max_event_duration_hours > 0
    ),
    CONSTRAINT chk_p037_prg_dates CHECK (
        expiration_date IS NULL OR expiration_date >= effective_date
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_prg_region         ON pack037_demand_response.dr_programs(iso_rto_region);
CREATE INDEX idx_p037_prg_type           ON pack037_demand_response.dr_programs(program_type);
CREATE INDEX idx_p037_prg_market         ON pack037_demand_response.dr_programs(market_type);
CREATE INDEX idx_p037_prg_status         ON pack037_demand_response.dr_programs(program_status);
CREATE INDEX idx_p037_prg_country        ON pack037_demand_response.dr_programs(country_code);
CREATE INDEX idx_p037_prg_sponsor        ON pack037_demand_response.dr_programs(program_sponsor);
CREATE INDEX idx_p037_prg_effective      ON pack037_demand_response.dr_programs(effective_date);
CREATE INDEX idx_p037_prg_created        ON pack037_demand_response.dr_programs(created_at DESC);
CREATE INDEX idx_p037_prg_metadata       ON pack037_demand_response.dr_programs USING GIN(metadata);

-- Composite: region + status + type for program discovery
CREATE INDEX idx_p037_prg_region_status  ON pack037_demand_response.dr_programs(iso_rto_region, program_status, program_type);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_prg_updated
    BEFORE UPDATE ON pack037_demand_response.dr_programs
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack037_demand_response.dr_program_requirements
-- =============================================================================
-- Technical and operational requirements for each DR program including
-- metering, communication, testing, and performance standards.

CREATE TABLE pack037_demand_response.dr_program_requirements (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    program_id              UUID            NOT NULL REFERENCES pack037_demand_response.dr_programs(id) ON DELETE CASCADE,
    requirement_category    VARCHAR(50)     NOT NULL,
    requirement_name        VARCHAR(255)    NOT NULL,
    requirement_description TEXT            NOT NULL,
    is_mandatory            BOOLEAN         NOT NULL DEFAULT true,
    threshold_value         NUMERIC(14,4),
    threshold_unit          VARCHAR(30),
    compliance_method       TEXT,
    verification_frequency  VARCHAR(30),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_req_category CHECK (
        requirement_category IN (
            'METERING', 'COMMUNICATION', 'TELEMETRY', 'TESTING',
            'PERFORMANCE', 'NOTIFICATION', 'REPORTING', 'ENROLLMENT',
            'OPERATIONAL', 'FINANCIAL', 'TECHNICAL'
        )
    ),
    CONSTRAINT chk_p037_req_freq CHECK (
        verification_frequency IS NULL OR verification_frequency IN (
            'ONCE', 'MONTHLY', 'QUARTERLY', 'SEMI_ANNUAL', 'ANNUAL', 'PER_EVENT'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_req_program        ON pack037_demand_response.dr_program_requirements(program_id);
CREATE INDEX idx_p037_req_category       ON pack037_demand_response.dr_program_requirements(requirement_category);
CREATE INDEX idx_p037_req_mandatory      ON pack037_demand_response.dr_program_requirements(is_mandatory);

-- =============================================================================
-- Table 3: pack037_demand_response.dr_program_compensation
-- =============================================================================
-- Compensation rates and structures for DR programs including capacity
-- payments, energy payments, performance incentives, and availability bonuses.

CREATE TABLE pack037_demand_response.dr_program_compensation (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    program_id              UUID            NOT NULL REFERENCES pack037_demand_response.dr_programs(id) ON DELETE CASCADE,
    compensation_type       VARCHAR(50)     NOT NULL,
    rate_structure          VARCHAR(30)     NOT NULL,
    base_rate               NUMERIC(12,4)   NOT NULL,
    rate_unit               VARCHAR(30)     NOT NULL,
    peak_multiplier         NUMERIC(6,3)    DEFAULT 1.0,
    seasonal_multiplier     NUMERIC(6,3)    DEFAULT 1.0,
    performance_bonus_pct   NUMERIC(6,2),
    availability_bonus_rate NUMERIC(12,4),
    min_payment             NUMERIC(14,2),
    max_payment             NUMERIC(14,2),
    payment_frequency       VARCHAR(20)     NOT NULL DEFAULT 'MONTHLY',
    settlement_delay_days   INTEGER         DEFAULT 30,
    effective_date          DATE            NOT NULL,
    expiration_date         DATE,
    currency_code           CHAR(3)         NOT NULL DEFAULT 'USD',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_comp_type CHECK (
        compensation_type IN (
            'CAPACITY_PAYMENT', 'ENERGY_PAYMENT', 'AVAILABILITY_PAYMENT',
            'PERFORMANCE_INCENTIVE', 'RESERVATION_FEE', 'EVENT_PAYMENT',
            'FREQUENCY_REGULATION', 'SPINNING_RESERVE', 'NON_SPINNING_RESERVE'
        )
    ),
    CONSTRAINT chk_p037_comp_structure CHECK (
        rate_structure IN (
            'FIXED', 'TIERED', 'MARKET_INDEXED', 'AUCTION_BASED',
            'PERFORMANCE_SCALED', 'REAL_TIME'
        )
    ),
    CONSTRAINT chk_p037_comp_rate CHECK (
        base_rate >= 0
    ),
    CONSTRAINT chk_p037_comp_unit CHECK (
        rate_unit IN (
            'USD_PER_KW_YEAR', 'USD_PER_KW_MONTH', 'USD_PER_KW_DAY',
            'USD_PER_KWH', 'USD_PER_MWH', 'USD_PER_MW_HOUR',
            'EUR_PER_KW_YEAR', 'EUR_PER_KW_MONTH', 'EUR_PER_KWH',
            'EUR_PER_MWH', 'GBP_PER_KW_YEAR', 'GBP_PER_MWH'
        )
    ),
    CONSTRAINT chk_p037_comp_frequency CHECK (
        payment_frequency IN (
            'PER_EVENT', 'WEEKLY', 'MONTHLY', 'QUARTERLY', 'SEMI_ANNUAL', 'ANNUAL'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_comp_program       ON pack037_demand_response.dr_program_compensation(program_id);
CREATE INDEX idx_p037_comp_type          ON pack037_demand_response.dr_program_compensation(compensation_type);
CREATE INDEX idx_p037_comp_effective     ON pack037_demand_response.dr_program_compensation(effective_date);
CREATE INDEX idx_p037_comp_rate          ON pack037_demand_response.dr_program_compensation(base_rate DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_comp_updated
    BEFORE UPDATE ON pack037_demand_response.dr_program_compensation
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack037_demand_response.dr_program_penalties
-- =============================================================================
-- Penalty structures for non-performance, under-delivery, and non-compliance
-- within DR programs.

CREATE TABLE pack037_demand_response.dr_program_penalties (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    program_id              UUID            NOT NULL REFERENCES pack037_demand_response.dr_programs(id) ON DELETE CASCADE,
    penalty_type            VARCHAR(50)     NOT NULL,
    trigger_condition       TEXT            NOT NULL,
    penalty_rate            NUMERIC(12,4)   NOT NULL,
    penalty_unit            VARCHAR(30)     NOT NULL,
    max_penalty_per_event   NUMERIC(14,2),
    max_penalty_per_year    NUMERIC(14,2),
    grace_period_events     INTEGER         DEFAULT 0,
    escalation_multiplier   NUMERIC(6,3)    DEFAULT 1.0,
    cure_period_days        INTEGER,
    effective_date          DATE            NOT NULL,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_pen_type CHECK (
        penalty_type IN (
            'UNDER_DELIVERY', 'NON_RESPONSE', 'NON_COMPLIANCE',
            'TELEMETRY_FAILURE', 'LATE_NOTIFICATION', 'TEST_FAILURE',
            'CAPACITY_DEFICIENCY', 'AVAILABILITY_SHORTFALL'
        )
    ),
    CONSTRAINT chk_p037_pen_rate CHECK (
        penalty_rate >= 0
    ),
    CONSTRAINT chk_p037_pen_unit CHECK (
        penalty_unit IN (
            'USD_PER_KW', 'USD_PER_MWH', 'USD_FLAT', 'PCT_OF_PAYMENT',
            'EUR_PER_KW', 'EUR_PER_MWH', 'EUR_FLAT', 'GBP_PER_KW',
            'GBP_PER_MWH', 'GBP_FLAT'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_pen_program        ON pack037_demand_response.dr_program_penalties(program_id);
CREATE INDEX idx_p037_pen_type           ON pack037_demand_response.dr_program_penalties(penalty_type);
CREATE INDEX idx_p037_pen_effective      ON pack037_demand_response.dr_program_penalties(effective_date);

-- =============================================================================
-- Table 5: pack037_demand_response.dr_program_enrollment
-- =============================================================================
-- Tracks facility enrollment in DR programs including enrollment status,
-- committed capacity, contract terms, and program-specific identifiers.

CREATE TABLE pack037_demand_response.dr_program_enrollment (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    program_id              UUID            NOT NULL REFERENCES pack037_demand_response.dr_programs(id) ON DELETE CASCADE,
    facility_profile_id     UUID            NOT NULL REFERENCES pack037_demand_response.dr_facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    enrollment_status       VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    committed_capacity_kw   NUMERIC(12,4)   NOT NULL,
    qualified_capacity_kw   NUMERIC(12,4),
    enrollment_date         DATE            NOT NULL,
    qualification_date      DATE,
    contract_start          DATE,
    contract_end            DATE,
    auto_renew              BOOLEAN         DEFAULT false,
    aggregator_name         VARCHAR(255),
    aggregator_id           VARCHAR(100),
    program_account_id      VARCHAR(100),
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              UUID,
    -- Constraints
    CONSTRAINT chk_p037_enr_status CHECK (
        enrollment_status IN (
            'PENDING', 'APPLIED', 'QUALIFIED', 'ENROLLED', 'ACTIVE',
            'SUSPENDED', 'WITHDRAWN', 'DISQUALIFIED', 'EXPIRED'
        )
    ),
    CONSTRAINT chk_p037_enr_committed CHECK (
        committed_capacity_kw > 0
    ),
    CONSTRAINT chk_p037_enr_qualified CHECK (
        qualified_capacity_kw IS NULL OR qualified_capacity_kw >= 0
    ),
    CONSTRAINT chk_p037_enr_contract CHECK (
        contract_end IS NULL OR contract_end >= contract_start
    ),
    CONSTRAINT uq_p037_enr_program_facility UNIQUE (program_id, facility_profile_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_enr_program        ON pack037_demand_response.dr_program_enrollment(program_id);
CREATE INDEX idx_p037_enr_facility       ON pack037_demand_response.dr_program_enrollment(facility_profile_id);
CREATE INDEX idx_p037_enr_tenant         ON pack037_demand_response.dr_program_enrollment(tenant_id);
CREATE INDEX idx_p037_enr_status         ON pack037_demand_response.dr_program_enrollment(enrollment_status);
CREATE INDEX idx_p037_enr_committed      ON pack037_demand_response.dr_program_enrollment(committed_capacity_kw DESC);
CREATE INDEX idx_p037_enr_enrollment     ON pack037_demand_response.dr_program_enrollment(enrollment_date DESC);
CREATE INDEX idx_p037_enr_contract_end   ON pack037_demand_response.dr_program_enrollment(contract_end);
CREATE INDEX idx_p037_enr_created        ON pack037_demand_response.dr_program_enrollment(created_at DESC);
CREATE INDEX idx_p037_enr_metadata       ON pack037_demand_response.dr_program_enrollment USING GIN(metadata);

-- Composite: active enrollments by facility for event dispatch
CREATE INDEX idx_p037_enr_fac_active     ON pack037_demand_response.dr_program_enrollment(facility_profile_id, program_id)
    WHERE enrollment_status IN ('ENROLLED', 'ACTIVE');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p037_enr_updated
    BEFORE UPDATE ON pack037_demand_response.dr_program_enrollment
    FOR EACH ROW EXECUTE FUNCTION pack037_demand_response.fn_set_updated_at();

-- =============================================================================
-- Table 6: pack037_demand_response.dr_program_eligibility
-- =============================================================================
-- Eligibility criteria per program defining which facility types, sizes,
-- regions, and load categories qualify for enrollment.

CREATE TABLE pack037_demand_response.dr_program_eligibility (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    program_id              UUID            NOT NULL REFERENCES pack037_demand_response.dr_programs(id) ON DELETE CASCADE,
    criterion_type          VARCHAR(50)     NOT NULL,
    criterion_field         VARCHAR(100)    NOT NULL,
    operator                VARCHAR(20)     NOT NULL,
    criterion_value         TEXT            NOT NULL,
    is_mandatory            BOOLEAN         NOT NULL DEFAULT true,
    description             TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p037_elig_type CHECK (
        criterion_type IN (
            'FACILITY_TYPE', 'LOAD_SIZE', 'REGION', 'METERING',
            'AUTOMATION', 'LOAD_CATEGORY', 'TELEMETRY', 'AGGREGATION',
            'UTILITY', 'CUSTOMER_CLASS', 'VOLTAGE_LEVEL'
        )
    ),
    CONSTRAINT chk_p037_elig_operator CHECK (
        operator IN (
            'EQ', 'NEQ', 'GT', 'GTE', 'LT', 'LTE', 'IN', 'NOT_IN',
            'BETWEEN', 'LIKE', 'EXISTS'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p037_elig_program       ON pack037_demand_response.dr_program_eligibility(program_id);
CREATE INDEX idx_p037_elig_type          ON pack037_demand_response.dr_program_eligibility(criterion_type);
CREATE INDEX idx_p037_elig_mandatory     ON pack037_demand_response.dr_program_eligibility(is_mandatory);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack037_demand_response.dr_programs ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_program_requirements ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_program_compensation ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_program_penalties ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_program_enrollment ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack037_demand_response.dr_program_eligibility ENABLE ROW LEVEL SECURITY;

-- Programs are shared reference data - all tenants can read
CREATE POLICY p037_prg_read_all ON pack037_demand_response.dr_programs
    FOR SELECT USING (TRUE);
CREATE POLICY p037_prg_service_bypass ON pack037_demand_response.dr_programs
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_req_read_all ON pack037_demand_response.dr_program_requirements
    FOR SELECT USING (TRUE);
CREATE POLICY p037_req_service_bypass ON pack037_demand_response.dr_program_requirements
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_comp_read_all ON pack037_demand_response.dr_program_compensation
    FOR SELECT USING (TRUE);
CREATE POLICY p037_comp_service_bypass ON pack037_demand_response.dr_program_compensation
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_pen_read_all ON pack037_demand_response.dr_program_penalties
    FOR SELECT USING (TRUE);
CREATE POLICY p037_pen_service_bypass ON pack037_demand_response.dr_program_penalties
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_enr_tenant_isolation ON pack037_demand_response.dr_program_enrollment
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p037_enr_service_bypass ON pack037_demand_response.dr_program_enrollment
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p037_elig_read_all ON pack037_demand_response.dr_program_eligibility
    FOR SELECT USING (TRUE);
CREATE POLICY p037_elig_service_bypass ON pack037_demand_response.dr_program_eligibility
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_programs TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_program_requirements TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_program_compensation TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_program_penalties TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_program_enrollment TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack037_demand_response.dr_program_eligibility TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack037_demand_response.dr_programs IS
    'Master list of demand response programs offered by ISOs, RTOs, utilities, and aggregators across US and European markets.';
COMMENT ON TABLE pack037_demand_response.dr_program_requirements IS
    'Technical and operational requirements for each DR program including metering, communication, and testing standards.';
COMMENT ON TABLE pack037_demand_response.dr_program_compensation IS
    'Compensation rates and structures for DR programs including capacity, energy, and performance payments.';
COMMENT ON TABLE pack037_demand_response.dr_program_penalties IS
    'Penalty structures for non-performance, under-delivery, and non-compliance within DR programs.';
COMMENT ON TABLE pack037_demand_response.dr_program_enrollment IS
    'Facility enrollment records in DR programs with committed capacity, contract terms, and aggregator details.';
COMMENT ON TABLE pack037_demand_response.dr_program_eligibility IS
    'Eligibility criteria per program defining qualifying facility types, sizes, regions, and load categories.';

COMMENT ON COLUMN pack037_demand_response.dr_programs.program_code IS 'Unique program code (e.g., PJM_ELR, CAISO_PDR, ERCOT_ERS_30).';
COMMENT ON COLUMN pack037_demand_response.dr_programs.program_type IS 'Program classification: CAPACITY, ENERGY, ANCILLARY, EMERGENCY, ECONOMIC, RELIABILITY, etc.';
COMMENT ON COLUMN pack037_demand_response.dr_programs.market_type IS 'Market type: WHOLESALE, RETAIL, BILATERAL, CAPACITY_MARKET, ENERGY_MARKET, etc.';
COMMENT ON COLUMN pack037_demand_response.dr_programs.dispatch_method IS 'How events are dispatched: MANUAL_CALL, AUTO_DISPATCH, PRICE_SIGNAL, OPENADR, DIRECT_LOAD_CONTROL.';
COMMENT ON COLUMN pack037_demand_response.dr_programs.measurement_verification IS 'M&V methodology: BASELINE_CBL, BASELINE_ADJUSTED, WHOLE_FACILITY, SUB_METERING, etc.';
COMMENT ON COLUMN pack037_demand_response.dr_programs.aggregation_allowed IS 'Whether load aggregation across sites is permitted for this program.';

COMMENT ON COLUMN pack037_demand_response.dr_program_enrollment.enrollment_status IS 'Enrollment lifecycle: PENDING, APPLIED, QUALIFIED, ENROLLED, ACTIVE, SUSPENDED, WITHDRAWN, DISQUALIFIED, EXPIRED.';
COMMENT ON COLUMN pack037_demand_response.dr_program_enrollment.committed_capacity_kw IS 'Capacity in kW committed to this program for the contract period.';
COMMENT ON COLUMN pack037_demand_response.dr_program_enrollment.aggregator_name IS 'Demand response aggregator managing this enrollment, if applicable.';
COMMENT ON COLUMN pack037_demand_response.dr_program_enrollment.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
