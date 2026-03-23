-- =============================================================================
-- V303: PACK-038 Peak Shaving Pack - Power Factor Analysis
-- =============================================================================
-- Pack:         PACK-038 (Peak Shaving Pack)
-- Migration:    008 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Power factor analysis and correction tables. Poor power factor
-- increases apparent power (kVA) demand, leading to higher billing
-- demand on kVA-billed tariffs and power factor penalty charges.
-- Covers PF measurement, reactive demand tracking, correction equipment
-- sizing, harmonic analysis, and penalty cost calculations.
--
-- Tables (5):
--   1. pack038_peak_shaving.ps_power_factor_data
--   2. pack038_peak_shaving.ps_reactive_demand
--   3. pack038_peak_shaving.ps_correction_sizing
--   4. pack038_peak_shaving.ps_harmonic_profiles
--   5. pack038_peak_shaving.ps_pf_penalties
--
-- Previous: V302__pack038_peak_shaving_007.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack038_peak_shaving.ps_power_factor_data
-- =============================================================================
-- Time-series power factor measurements capturing real power (kW),
-- reactive power (kVAR), apparent power (kVA), and computed PF values
-- for trending, correction sizing, and penalty analysis.

CREATE TABLE pack038_peak_shaving.ps_power_factor_data (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    measurement_timestamp   TIMESTAMPTZ     NOT NULL,
    measurement_point       VARCHAR(100)    NOT NULL DEFAULT 'MAIN_METER',
    kw_demand               NUMERIC(12,3)   NOT NULL,
    kvar_demand             NUMERIC(12,3)   NOT NULL,
    kva_demand              NUMERIC(12,3)   NOT NULL,
    power_factor            NUMERIC(5,4)    NOT NULL,
    pf_type                 VARCHAR(20)     NOT NULL DEFAULT 'LAGGING',
    displacement_pf         NUMERIC(5,4),
    true_pf                 NUMERIC(5,4),
    voltage_v               NUMERIC(10,3),
    current_a               NUMERIC(10,3),
    thd_v_pct               NUMERIC(5,2),
    thd_i_pct               NUMERIC(5,2),
    interval_length_minutes INTEGER         NOT NULL DEFAULT 15,
    is_peak_period          BOOLEAN         DEFAULT false,
    temperature_f           NUMERIC(6,2),
    data_quality            VARCHAR(20)     NOT NULL DEFAULT 'MEASURED',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_pfd_kw CHECK (
        kw_demand >= 0
    ),
    CONSTRAINT chk_p038_pfd_kva CHECK (
        kva_demand >= 0
    ),
    CONSTRAINT chk_p038_pfd_pf CHECK (
        power_factor >= 0 AND power_factor <= 1.0
    ),
    CONSTRAINT chk_p038_pfd_pf_type CHECK (
        pf_type IN ('LAGGING', 'LEADING', 'UNITY')
    ),
    CONSTRAINT chk_p038_pfd_displacement CHECK (
        displacement_pf IS NULL OR (displacement_pf >= 0 AND displacement_pf <= 1.0)
    ),
    CONSTRAINT chk_p038_pfd_true_pf CHECK (
        true_pf IS NULL OR (true_pf >= 0 AND true_pf <= 1.0)
    ),
    CONSTRAINT chk_p038_pfd_thd_v CHECK (
        thd_v_pct IS NULL OR (thd_v_pct >= 0 AND thd_v_pct <= 100)
    ),
    CONSTRAINT chk_p038_pfd_thd_i CHECK (
        thd_i_pct IS NULL OR (thd_i_pct >= 0 AND thd_i_pct <= 100)
    ),
    CONSTRAINT chk_p038_pfd_interval CHECK (
        interval_length_minutes IN (1, 5, 15, 30, 60)
    ),
    CONSTRAINT chk_p038_pfd_quality CHECK (
        data_quality IN ('MEASURED', 'ESTIMATED', 'CALCULATED', 'VALIDATED')
    ),
    CONSTRAINT uq_p038_pfd_profile_ts UNIQUE (profile_id, measurement_point, measurement_timestamp)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_pfd_profile        ON pack038_peak_shaving.ps_power_factor_data(profile_id);
CREATE INDEX idx_p038_pfd_tenant         ON pack038_peak_shaving.ps_power_factor_data(tenant_id);
CREATE INDEX idx_p038_pfd_timestamp      ON pack038_peak_shaving.ps_power_factor_data(measurement_timestamp DESC);
CREATE INDEX idx_p038_pfd_pf             ON pack038_peak_shaving.ps_power_factor_data(power_factor);
CREATE INDEX idx_p038_pfd_point          ON pack038_peak_shaving.ps_power_factor_data(measurement_point);
CREATE INDEX idx_p038_pfd_kva            ON pack038_peak_shaving.ps_power_factor_data(kva_demand DESC);
CREATE INDEX idx_p038_pfd_created        ON pack038_peak_shaving.ps_power_factor_data(created_at DESC);

-- Composite: profile + time for trending
CREATE INDEX idx_p038_pfd_profile_ts     ON pack038_peak_shaving.ps_power_factor_data(profile_id, measurement_timestamp DESC, power_factor);

-- Low PF intervals for penalty analysis
CREATE INDEX idx_p038_pfd_low_pf         ON pack038_peak_shaving.ps_power_factor_data(profile_id, power_factor)
    WHERE power_factor < 0.90;

-- =============================================================================
-- Table 2: pack038_peak_shaving.ps_reactive_demand
-- =============================================================================
-- Aggregated reactive power demand statistics per billing period.
-- Summarises kVAR demand patterns for correction equipment sizing
-- and penalty cost forecasting.

CREATE TABLE pack038_peak_shaving.ps_reactive_demand (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    period_type             VARCHAR(20)     NOT NULL DEFAULT 'MONTHLY',
    avg_power_factor        NUMERIC(5,4)    NOT NULL,
    min_power_factor        NUMERIC(5,4),
    weighted_avg_pf         NUMERIC(5,4),
    peak_kvar               NUMERIC(12,3)   NOT NULL,
    avg_kvar                NUMERIC(12,3),
    peak_kva                NUMERIC(12,3),
    avg_kva                 NUMERIC(12,3),
    peak_kw                 NUMERIC(12,3),
    total_kvarh             NUMERIC(15,3),
    total_kwh               NUMERIC(15,3),
    pf_below_threshold_pct  NUMERIC(5,2),
    pf_threshold            NUMERIC(5,4)    NOT NULL DEFAULT 0.9000,
    excess_kvar             NUMERIC(12,3),
    kvar_correction_needed  NUMERIC(12,3),
    billing_pf              NUMERIC(5,4),
    pf_adjustment_factor    NUMERIC(5,4),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_rd_avg_pf CHECK (
        avg_power_factor >= 0 AND avg_power_factor <= 1.0
    ),
    CONSTRAINT chk_p038_rd_min_pf CHECK (
        min_power_factor IS NULL OR (min_power_factor >= 0 AND min_power_factor <= 1.0)
    ),
    CONSTRAINT chk_p038_rd_weighted CHECK (
        weighted_avg_pf IS NULL OR (weighted_avg_pf >= 0 AND weighted_avg_pf <= 1.0)
    ),
    CONSTRAINT chk_p038_rd_peak_kvar CHECK (
        peak_kvar >= 0
    ),
    CONSTRAINT chk_p038_rd_period CHECK (
        period_type IN ('DAILY', 'WEEKLY', 'MONTHLY', 'BILLING_PERIOD', 'ANNUAL')
    ),
    CONSTRAINT chk_p038_rd_dates CHECK (
        period_start <= period_end
    ),
    CONSTRAINT chk_p038_rd_threshold CHECK (
        pf_threshold >= 0.5 AND pf_threshold <= 1.0
    ),
    CONSTRAINT chk_p038_rd_below_pct CHECK (
        pf_below_threshold_pct IS NULL OR (pf_below_threshold_pct >= 0 AND pf_below_threshold_pct <= 100)
    ),
    CONSTRAINT uq_p038_rd_profile_period UNIQUE (profile_id, period_type, period_start)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_rd_profile         ON pack038_peak_shaving.ps_reactive_demand(profile_id);
CREATE INDEX idx_p038_rd_tenant          ON pack038_peak_shaving.ps_reactive_demand(tenant_id);
CREATE INDEX idx_p038_rd_period_start    ON pack038_peak_shaving.ps_reactive_demand(period_start DESC);
CREATE INDEX idx_p038_rd_avg_pf          ON pack038_peak_shaving.ps_reactive_demand(avg_power_factor);
CREATE INDEX idx_p038_rd_peak_kvar       ON pack038_peak_shaving.ps_reactive_demand(peak_kvar DESC);
CREATE INDEX idx_p038_rd_correction      ON pack038_peak_shaving.ps_reactive_demand(kvar_correction_needed DESC);
CREATE INDEX idx_p038_rd_created         ON pack038_peak_shaving.ps_reactive_demand(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_rd_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_reactive_demand
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack038_peak_shaving.ps_correction_sizing
-- =============================================================================
-- Power factor correction equipment sizing calculations. Determines
-- the required capacitor bank or active filter capacity to achieve
-- the target power factor, considering load variability and harmonics.

CREATE TABLE pack038_peak_shaving.ps_correction_sizing (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    sizing_name             VARCHAR(255)    NOT NULL,
    current_pf              NUMERIC(5,4)    NOT NULL,
    target_pf               NUMERIC(5,4)    NOT NULL,
    correction_type         VARCHAR(30)     NOT NULL,
    installation_point      VARCHAR(30)     NOT NULL DEFAULT 'MAIN',
    peak_kw_at_correction   NUMERIC(12,3)   NOT NULL,
    required_kvar           NUMERIC(12,3)   NOT NULL,
    recommended_kvar        NUMERIC(12,3)   NOT NULL,
    number_of_steps         INTEGER,
    step_size_kvar          NUMERIC(10,3),
    detuning_reactor_pct    NUMERIC(5,2),
    resonance_frequency_hz  NUMERIC(8,2),
    switching_type          VARCHAR(30),
    controller_type         VARCHAR(30),
    expected_pf_after       NUMERIC(5,4),
    expected_kva_reduction  NUMERIC(12,3),
    expected_kvar_reduction NUMERIC(12,3),
    equipment_cost          NUMERIC(12,2),
    installation_cost       NUMERIC(12,2),
    annual_maintenance_cost NUMERIC(10,2),
    total_cost              NUMERIC(12,2),
    annual_savings          NUMERIC(12,2),
    simple_payback_months   NUMERIC(6,1),
    manufacturer            VARCHAR(100),
    model_recommendation    VARCHAR(100),
    sizing_status           VARCHAR(20)     NOT NULL DEFAULT 'CALCULATED',
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_cs_current_pf CHECK (
        current_pf >= 0 AND current_pf <= 1.0
    ),
    CONSTRAINT chk_p038_cs_target_pf CHECK (
        target_pf >= 0 AND target_pf <= 1.0 AND target_pf > current_pf
    ),
    CONSTRAINT chk_p038_cs_correction_type CHECK (
        correction_type IN (
            'FIXED_CAPACITOR', 'SWITCHED_CAPACITOR', 'AUTOMATIC_CAPACITOR',
            'ACTIVE_FILTER', 'HYBRID_FILTER', 'SYNCHRONOUS_CONDENSER',
            'STATIC_VAR_COMPENSATOR', 'STATCOM'
        )
    ),
    CONSTRAINT chk_p038_cs_installation CHECK (
        installation_point IN (
            'MAIN', 'FEEDER', 'MCC', 'INDIVIDUAL_MOTOR', 'TRANSFORMER',
            'DISTRIBUTION_PANEL', 'OTHER'
        )
    ),
    CONSTRAINT chk_p038_cs_required CHECK (
        required_kvar > 0
    ),
    CONSTRAINT chk_p038_cs_recommended CHECK (
        recommended_kvar > 0
    ),
    CONSTRAINT chk_p038_cs_switching CHECK (
        switching_type IS NULL OR switching_type IN (
            'CONTACTOR', 'THYRISTOR', 'IGBT', 'VACUUM', 'STATIC'
        )
    ),
    CONSTRAINT chk_p038_cs_controller CHECK (
        controller_type IS NULL OR controller_type IN (
            'MANUAL', 'AUTOMATIC_PF', 'AUTOMATIC_KVAR', 'PLC', 'ADAPTIVE'
        )
    ),
    CONSTRAINT chk_p038_cs_status CHECK (
        sizing_status IN ('CALCULATED', 'REVIEWED', 'APPROVED', 'ORDERED', 'INSTALLED')
    ),
    CONSTRAINT chk_p038_cs_detuning CHECK (
        detuning_reactor_pct IS NULL OR (detuning_reactor_pct >= 5 AND detuning_reactor_pct <= 14)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_cs_profile         ON pack038_peak_shaving.ps_correction_sizing(profile_id);
CREATE INDEX idx_p038_cs_tenant          ON pack038_peak_shaving.ps_correction_sizing(tenant_id);
CREATE INDEX idx_p038_cs_type            ON pack038_peak_shaving.ps_correction_sizing(correction_type);
CREATE INDEX idx_p038_cs_point           ON pack038_peak_shaving.ps_correction_sizing(installation_point);
CREATE INDEX idx_p038_cs_kvar            ON pack038_peak_shaving.ps_correction_sizing(recommended_kvar DESC);
CREATE INDEX idx_p038_cs_savings         ON pack038_peak_shaving.ps_correction_sizing(annual_savings DESC);
CREATE INDEX idx_p038_cs_payback         ON pack038_peak_shaving.ps_correction_sizing(simple_payback_months);
CREATE INDEX idx_p038_cs_status          ON pack038_peak_shaving.ps_correction_sizing(sizing_status);
CREATE INDEX idx_p038_cs_created         ON pack038_peak_shaving.ps_correction_sizing(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_cs_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_correction_sizing
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack038_peak_shaving.ps_harmonic_profiles
-- =============================================================================
-- Harmonic distortion profiles at the facility level. Harmonics reduce
-- true power factor even with displacement PF correction and can cause
-- resonance with capacitor banks, requiring detuned or active solutions.

CREATE TABLE pack038_peak_shaving.ps_harmonic_profiles (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    measurement_date        DATE            NOT NULL,
    measurement_point       VARCHAR(100)    NOT NULL DEFAULT 'MAIN_METER',
    thd_voltage_pct         NUMERIC(5,2)    NOT NULL,
    thd_current_pct         NUMERIC(5,2)    NOT NULL,
    harmonic_3rd_pct        NUMERIC(5,2),
    harmonic_5th_pct        NUMERIC(5,2),
    harmonic_7th_pct        NUMERIC(5,2),
    harmonic_9th_pct        NUMERIC(5,2),
    harmonic_11th_pct       NUMERIC(5,2),
    harmonic_13th_pct       NUMERIC(5,2),
    harmonic_spectrum       JSONB           DEFAULT '{}',
    displacement_pf         NUMERIC(5,4),
    true_pf                 NUMERIC(5,4),
    distortion_pf           NUMERIC(5,4),
    k_factor                NUMERIC(8,4),
    crest_factor            NUMERIC(5,3),
    ieee_519_compliant      BOOLEAN,
    iec_61000_compliant     BOOLEAN,
    primary_harmonic_sources JSONB          DEFAULT '[]',
    resonance_risk          VARCHAR(20),
    recommended_action      TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_hp_thd_v CHECK (
        thd_voltage_pct >= 0 AND thd_voltage_pct <= 100
    ),
    CONSTRAINT chk_p038_hp_thd_i CHECK (
        thd_current_pct >= 0 AND thd_current_pct <= 100
    ),
    CONSTRAINT chk_p038_hp_displacement CHECK (
        displacement_pf IS NULL OR (displacement_pf >= 0 AND displacement_pf <= 1.0)
    ),
    CONSTRAINT chk_p038_hp_true CHECK (
        true_pf IS NULL OR (true_pf >= 0 AND true_pf <= 1.0)
    ),
    CONSTRAINT chk_p038_hp_distortion CHECK (
        distortion_pf IS NULL OR (distortion_pf >= 0 AND distortion_pf <= 1.0)
    ),
    CONSTRAINT chk_p038_hp_resonance CHECK (
        resonance_risk IS NULL OR resonance_risk IN ('NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT uq_p038_hp_profile_date UNIQUE (profile_id, measurement_point, measurement_date)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_hp_profile         ON pack038_peak_shaving.ps_harmonic_profiles(profile_id);
CREATE INDEX idx_p038_hp_tenant          ON pack038_peak_shaving.ps_harmonic_profiles(tenant_id);
CREATE INDEX idx_p038_hp_date            ON pack038_peak_shaving.ps_harmonic_profiles(measurement_date DESC);
CREATE INDEX idx_p038_hp_thd_v           ON pack038_peak_shaving.ps_harmonic_profiles(thd_voltage_pct DESC);
CREATE INDEX idx_p038_hp_thd_i           ON pack038_peak_shaving.ps_harmonic_profiles(thd_current_pct DESC);
CREATE INDEX idx_p038_hp_resonance       ON pack038_peak_shaving.ps_harmonic_profiles(resonance_risk);
CREATE INDEX idx_p038_hp_compliant       ON pack038_peak_shaving.ps_harmonic_profiles(ieee_519_compliant);
CREATE INDEX idx_p038_hp_created         ON pack038_peak_shaving.ps_harmonic_profiles(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_hp_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_harmonic_profiles
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack038_peak_shaving.ps_pf_penalties
-- =============================================================================
-- Power factor penalty and adjustment charges per billing period.
-- Captures the financial impact of poor power factor including
-- kVA-based billing adjustments, reactive demand charges, and
-- explicit PF penalty surcharges.

CREATE TABLE pack038_peak_shaving.ps_pf_penalties (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id              UUID            NOT NULL REFERENCES pack038_peak_shaving.ps_load_profiles(id) ON DELETE CASCADE,
    tariff_id               UUID            REFERENCES pack038_peak_shaving.ps_tariff_structures(id),
    tenant_id               UUID            NOT NULL,
    billing_period_start    DATE            NOT NULL,
    billing_period_end      DATE            NOT NULL,
    penalty_type            VARCHAR(30)     NOT NULL,
    measured_pf             NUMERIC(5,4)    NOT NULL,
    minimum_pf_required     NUMERIC(5,4)    NOT NULL DEFAULT 0.9000,
    actual_kw_demand        NUMERIC(12,3)   NOT NULL,
    actual_kva_demand       NUMERIC(12,3),
    actual_kvar_demand      NUMERIC(12,3),
    billed_demand_kw        NUMERIC(12,3),
    billed_demand_kva       NUMERIC(12,3),
    pf_adjustment_factor    NUMERIC(5,4),
    penalty_charge          NUMERIC(12,2)   NOT NULL,
    reactive_demand_charge  NUMERIC(12,2),
    kva_excess_charge       NUMERIC(12,2),
    total_pf_cost           NUMERIC(12,2)   NOT NULL,
    cost_with_correction    NUMERIC(12,2),
    potential_savings       NUMERIC(12,2),
    correction_applied      BOOLEAN         DEFAULT false,
    correction_kvar         NUMERIC(12,3),
    corrected_pf            NUMERIC(5,4),
    currency_code           CHAR(3)         NOT NULL DEFAULT 'USD',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p038_pfp_type CHECK (
        penalty_type IN (
            'KVA_BILLING', 'REACTIVE_DEMAND', 'PERCENTAGE_SURCHARGE',
            'FLAT_PENALTY', 'TIERED_PENALTY', 'KW_ADJUSTMENT',
            'EXCESS_KVAR', 'COMBINED'
        )
    ),
    CONSTRAINT chk_p038_pfp_measured CHECK (
        measured_pf >= 0 AND measured_pf <= 1.0
    ),
    CONSTRAINT chk_p038_pfp_minimum CHECK (
        minimum_pf_required >= 0.5 AND minimum_pf_required <= 1.0
    ),
    CONSTRAINT chk_p038_pfp_kw CHECK (
        actual_kw_demand >= 0
    ),
    CONSTRAINT chk_p038_pfp_penalty CHECK (
        penalty_charge >= 0
    ),
    CONSTRAINT chk_p038_pfp_total CHECK (
        total_pf_cost >= 0
    ),
    CONSTRAINT chk_p038_pfp_corrected CHECK (
        corrected_pf IS NULL OR (corrected_pf >= 0 AND corrected_pf <= 1.0)
    ),
    CONSTRAINT chk_p038_pfp_dates CHECK (
        billing_period_start <= billing_period_end
    ),
    CONSTRAINT uq_p038_pfp_profile_period UNIQUE (profile_id, tariff_id, billing_period_start)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p038_pfp_profile        ON pack038_peak_shaving.ps_pf_penalties(profile_id);
CREATE INDEX idx_p038_pfp_tariff         ON pack038_peak_shaving.ps_pf_penalties(tariff_id);
CREATE INDEX idx_p038_pfp_tenant         ON pack038_peak_shaving.ps_pf_penalties(tenant_id);
CREATE INDEX idx_p038_pfp_period         ON pack038_peak_shaving.ps_pf_penalties(billing_period_start DESC);
CREATE INDEX idx_p038_pfp_type           ON pack038_peak_shaving.ps_pf_penalties(penalty_type);
CREATE INDEX idx_p038_pfp_pf             ON pack038_peak_shaving.ps_pf_penalties(measured_pf);
CREATE INDEX idx_p038_pfp_total_cost     ON pack038_peak_shaving.ps_pf_penalties(total_pf_cost DESC);
CREATE INDEX idx_p038_pfp_savings        ON pack038_peak_shaving.ps_pf_penalties(potential_savings DESC);
CREATE INDEX idx_p038_pfp_created        ON pack038_peak_shaving.ps_pf_penalties(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p038_pfp_updated
    BEFORE UPDATE ON pack038_peak_shaving.ps_pf_penalties
    FOR EACH ROW EXECUTE FUNCTION pack038_peak_shaving.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack038_peak_shaving.ps_power_factor_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_reactive_demand ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_correction_sizing ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_harmonic_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack038_peak_shaving.ps_pf_penalties ENABLE ROW LEVEL SECURITY;

CREATE POLICY p038_pfd_tenant_isolation
    ON pack038_peak_shaving.ps_power_factor_data
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_pfd_service_bypass
    ON pack038_peak_shaving.ps_power_factor_data
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_rd_tenant_isolation
    ON pack038_peak_shaving.ps_reactive_demand
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_rd_service_bypass
    ON pack038_peak_shaving.ps_reactive_demand
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_cs_tenant_isolation
    ON pack038_peak_shaving.ps_correction_sizing
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_cs_service_bypass
    ON pack038_peak_shaving.ps_correction_sizing
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_hp_tenant_isolation
    ON pack038_peak_shaving.ps_harmonic_profiles
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_hp_service_bypass
    ON pack038_peak_shaving.ps_harmonic_profiles
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p038_pfp_tenant_isolation
    ON pack038_peak_shaving.ps_pf_penalties
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p038_pfp_service_bypass
    ON pack038_peak_shaving.ps_pf_penalties
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_power_factor_data TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_reactive_demand TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_correction_sizing TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_harmonic_profiles TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack038_peak_shaving.ps_pf_penalties TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack038_peak_shaving.ps_power_factor_data IS
    'Time-series power factor measurements with kW, kVAR, kVA, displacement PF, true PF, and THD values.';
COMMENT ON TABLE pack038_peak_shaving.ps_reactive_demand IS
    'Aggregated reactive power demand statistics per billing period for correction sizing and penalty forecasting.';
COMMENT ON TABLE pack038_peak_shaving.ps_correction_sizing IS
    'Power factor correction equipment sizing with capacitor bank requirements, active filter options, and ROI calculations.';
COMMENT ON TABLE pack038_peak_shaving.ps_harmonic_profiles IS
    'Harmonic distortion profiles with individual harmonic percentages, K-factor, crest factor, and IEEE 519 compliance.';
COMMENT ON TABLE pack038_peak_shaving.ps_pf_penalties IS
    'Power factor penalty and adjustment charges per billing period with savings potential from correction.';

COMMENT ON COLUMN pack038_peak_shaving.ps_power_factor_data.displacement_pf IS 'Displacement power factor from fundamental frequency only (ignores harmonics).';
COMMENT ON COLUMN pack038_peak_shaving.ps_power_factor_data.true_pf IS 'True power factor including harmonic distortion effects (always <= displacement PF).';
COMMENT ON COLUMN pack038_peak_shaving.ps_harmonic_profiles.k_factor IS 'K-factor rating for transformer derating due to harmonic loading. Higher values require higher-rated transformers.';
COMMENT ON COLUMN pack038_peak_shaving.ps_harmonic_profiles.resonance_risk IS 'Risk of parallel resonance between capacitor banks and system inductance: NONE, LOW, MEDIUM, HIGH, CRITICAL.';
COMMENT ON COLUMN pack038_peak_shaving.ps_correction_sizing.detuning_reactor_pct IS 'Detuning reactor impedance as percentage (typically 7% or 14%) to prevent harmonic resonance with capacitor banks.';
COMMENT ON COLUMN pack038_peak_shaving.ps_pf_penalties.pf_adjustment_factor IS 'Multiplier applied to kW demand to calculate kVA billing demand (e.g., kW / PF = adjusted demand).';
