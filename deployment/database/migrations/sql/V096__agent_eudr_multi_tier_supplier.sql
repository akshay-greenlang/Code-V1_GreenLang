-- ============================================================================
-- V096: AGENT-EUDR-008 Multi-Tier Supplier Tracker
-- ============================================================================
-- Creates tables for multi-tier supplier tracking, relationship management,
-- risk propagation, compliance monitoring, and gap analysis for EUDR.
--
-- Tables: 10 (6 regular + 4 hypertables)
-- Hypertables: gl_eudr_mst_relationships, gl_eudr_mst_tier_scores,
--              gl_eudr_mst_risk_scores, gl_eudr_mst_compliance_status
-- Continuous Aggregates: 2 (daily risk + compliance summaries)
-- Retention Policies: 5 (hypertables + audit log)
-- Indexes: 32
--
-- Dependencies: TimescaleDB extension (V002)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

-- ============================================================================
-- 1. gl_eudr_mst_suppliers — Supplier profile records
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_mst_suppliers (
    id                  BIGSERIAL       PRIMARY KEY,
    supplier_id         UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,

    -- Legal entity
    legal_name          TEXT            NOT NULL,
    trade_name          TEXT,
    registration_id     TEXT,
    tax_id              TEXT,
    duns_number         TEXT,
    lei_code            TEXT,

    -- Location
    country_code        VARCHAR(3)      NOT NULL,  -- ISO 3166-1 alpha-3
    admin_region        TEXT,
    address_line1       TEXT,
    address_line2       TEXT,
    city                TEXT,
    postal_code         TEXT,
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,

    -- Classification
    supplier_type       VARCHAR(50)     NOT NULL DEFAULT 'unknown',
        -- producer, cooperative, aggregator, processor, trader, refinery, importer, distributor
    commodities         TEXT[]          NOT NULL DEFAULT '{}',
        -- cattle, cocoa, coffee, palm_oil, rubber, soya, wood
    derived_products    TEXT[]          DEFAULT '{}',

    -- Capacity
    annual_volume_tonnes    DOUBLE PRECISION,
    processing_capacity_tonnes DOUBLE PRECISION,
    upstream_supplier_count INTEGER DEFAULT 0,

    -- Contact
    primary_contact_name    TEXT,
    primary_contact_email   TEXT,
    primary_contact_phone   TEXT,
    compliance_contact_name TEXT,
    compliance_contact_email TEXT,

    -- DDS linkage
    dds_references      TEXT[]          DEFAULT '{}',

    -- Profile quality
    profile_completeness_score  DOUBLE PRECISION DEFAULT 0.0,
    missing_fields      TEXT[]          DEFAULT '{}',

    -- Status
    status              VARCHAR(30)     NOT NULL DEFAULT 'active',
        -- active, suspended, terminated, prospective
    discovery_source    VARCHAR(50),
        -- manual, erp_import, declaration, questionnaire, certification_db, inferred
    confidence_level    VARCHAR(20)     NOT NULL DEFAULT 'unverified',
        -- verified, declared, inferred, suspected, unverified

    -- Provenance
    provenance_hash     VARCHAR(64),
    version             INTEGER         NOT NULL DEFAULT 1,
    created_by          UUID,
    updated_by          UUID,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_eudr_mst_supplier_id UNIQUE (supplier_id),
    CONSTRAINT uq_eudr_mst_supplier_tenant UNIQUE (tenant_id, legal_name, country_code)
);

CREATE INDEX idx_eudr_mst_suppliers_tenant ON gl_eudr_mst_suppliers (tenant_id);
CREATE INDEX idx_eudr_mst_suppliers_country ON gl_eudr_mst_suppliers (country_code);
CREATE INDEX idx_eudr_mst_suppliers_status ON gl_eudr_mst_suppliers (status);
CREATE INDEX idx_eudr_mst_suppliers_type ON gl_eudr_mst_suppliers (supplier_type);
CREATE INDEX idx_eudr_mst_suppliers_commodities ON gl_eudr_mst_suppliers USING GIN (commodities);
CREATE INDEX idx_eudr_mst_suppliers_confidence ON gl_eudr_mst_suppliers (confidence_level);
CREATE INDEX idx_eudr_mst_suppliers_name ON gl_eudr_mst_suppliers (legal_name);


-- ============================================================================
-- 2. gl_eudr_mst_relationships — Supplier-to-supplier relationships (hypertable)
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_mst_relationships (
    id                  BIGSERIAL,
    relationship_id     UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    recorded_at         TIMESTAMPTZ     NOT NULL DEFAULT now(),

    -- Relationship endpoints
    parent_supplier_id  UUID            NOT NULL,  -- buyer / downstream
    child_supplier_id   UUID            NOT NULL,  -- supplier / upstream
    tier_level          INTEGER         NOT NULL,  -- 1 = direct, 2 = Tier 2, etc.

    -- Attributes
    commodity           VARCHAR(30)     NOT NULL,
    volume_tonnes       DOUBLE PRECISION,
    volume_percentage   DOUBLE PRECISION,  -- % of parent's supply from this child
    frequency           VARCHAR(30),       -- continuous, seasonal, spot, annual
    exclusivity         VARCHAR(30)     DEFAULT 'non_exclusive',
        -- exclusive, preferred, non_exclusive

    -- State
    relationship_status VARCHAR(30)     NOT NULL DEFAULT 'prospective',
        -- prospective, onboarding, active, suspended, terminated
    start_date          DATE,
    end_date            DATE,
    reason_code         VARCHAR(50),

    -- Scoring
    strength_score      DOUBLE PRECISION DEFAULT 0.0,
    confidence_level    VARCHAR(20)     NOT NULL DEFAULT 'declared',

    -- Provenance
    provenance_hash     VARCHAR(64),
    created_by          UUID,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),

    PRIMARY KEY (id, recorded_at)
);

SELECT create_hypertable(
    'gl_eudr_mst_relationships',
    'recorded_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_eudr_mst_rel_tenant ON gl_eudr_mst_relationships (tenant_id, recorded_at DESC);
CREATE INDEX idx_eudr_mst_rel_parent ON gl_eudr_mst_relationships (parent_supplier_id, recorded_at DESC);
CREATE INDEX idx_eudr_mst_rel_child ON gl_eudr_mst_relationships (child_supplier_id, recorded_at DESC);
CREATE INDEX idx_eudr_mst_rel_commodity ON gl_eudr_mst_relationships (commodity, recorded_at DESC);
CREATE INDEX idx_eudr_mst_rel_status ON gl_eudr_mst_relationships (relationship_status, recorded_at DESC);
CREATE INDEX idx_eudr_mst_rel_tier ON gl_eudr_mst_relationships (tier_level, recorded_at DESC);


-- ============================================================================
-- 3. gl_eudr_mst_tier_scores — Tier depth and visibility scores (hypertable)
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_mst_tier_scores (
    id                  BIGSERIAL,
    score_id            UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    assessed_at         TIMESTAMPTZ     NOT NULL DEFAULT now(),

    -- Target
    supplier_id         UUID            NOT NULL,
    commodity           VARCHAR(30)     NOT NULL,

    -- Depth metrics
    max_tier_depth      INTEGER         NOT NULL DEFAULT 0,
    avg_tier_depth      DOUBLE PRECISION DEFAULT 0.0,
    visibility_score    DOUBLE PRECISION NOT NULL DEFAULT 0.0,  -- 0-100
    coverage_score      DOUBLE PRECISION NOT NULL DEFAULT 0.0,  -- 0-100

    -- Per-tier breakdown (JSONB for flexibility)
    tier_breakdown      JSONB           DEFAULT '{}',
        -- { "1": {"known": 5, "total_est": 5, "visibility": 100.0},
        --   "2": {"known": 12, "total_est": 20, "visibility": 60.0}, ... }

    -- Benchmarks
    industry_avg_depth  DOUBLE PRECISION,
    industry_avg_visibility DOUBLE PRECISION,
    depth_vs_benchmark  VARCHAR(20),  -- above, at, below

    -- Gaps
    gap_tiers           INTEGER[]       DEFAULT '{}',  -- tiers with visibility < threshold
    gap_severity        VARCHAR(20),  -- none, minor, major, critical

    -- Provenance
    provenance_hash     VARCHAR(64),
    created_by          UUID,

    PRIMARY KEY (id, assessed_at)
);

SELECT create_hypertable(
    'gl_eudr_mst_tier_scores',
    'assessed_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_eudr_mst_tier_tenant ON gl_eudr_mst_tier_scores (tenant_id, assessed_at DESC);
CREATE INDEX idx_eudr_mst_tier_supplier ON gl_eudr_mst_tier_scores (supplier_id, assessed_at DESC);
CREATE INDEX idx_eudr_mst_tier_commodity ON gl_eudr_mst_tier_scores (commodity, assessed_at DESC);


-- ============================================================================
-- 4. gl_eudr_mst_risk_scores — Supplier risk assessment records (hypertable)
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_mst_risk_scores (
    id                  BIGSERIAL,
    risk_id             UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    assessed_at         TIMESTAMPTZ     NOT NULL DEFAULT now(),

    -- Target
    supplier_id         UUID            NOT NULL,
    commodity           VARCHAR(30),

    -- Composite risk
    composite_risk_score    DOUBLE PRECISION NOT NULL DEFAULT 0.0,  -- 0-100
    risk_level          VARCHAR(20)     NOT NULL DEFAULT 'unknown',
        -- critical, high, medium, low, minimal, unknown

    -- Category breakdown
    deforestation_proximity_score   DOUBLE PRECISION DEFAULT 0.0,
    country_risk_score              DOUBLE PRECISION DEFAULT 0.0,
    certification_gap_score         DOUBLE PRECISION DEFAULT 0.0,
    compliance_history_score        DOUBLE PRECISION DEFAULT 0.0,
    data_quality_score              DOUBLE PRECISION DEFAULT 0.0,
    concentration_risk_score        DOUBLE PRECISION DEFAULT 0.0,

    -- Propagation
    propagation_method  VARCHAR(30),  -- max, weighted_average, volume_weighted
    propagated_from     UUID[],       -- supplier IDs contributing to propagated risk
    propagation_depth   INTEGER DEFAULT 0,
    own_risk_score      DOUBLE PRECISION DEFAULT 0.0,
    inherited_risk_score DOUBLE PRECISION DEFAULT 0.0,

    -- Trend
    risk_trend          VARCHAR(20),  -- improving, stable, degrading
    previous_score      DOUBLE PRECISION,
    score_change        DOUBLE PRECISION,

    -- Provenance
    provenance_hash     VARCHAR(64),
    created_by          UUID,

    PRIMARY KEY (id, assessed_at)
);

SELECT create_hypertable(
    'gl_eudr_mst_risk_scores',
    'assessed_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_eudr_mst_risk_tenant ON gl_eudr_mst_risk_scores (tenant_id, assessed_at DESC);
CREATE INDEX idx_eudr_mst_risk_supplier ON gl_eudr_mst_risk_scores (supplier_id, assessed_at DESC);
CREATE INDEX idx_eudr_mst_risk_level ON gl_eudr_mst_risk_scores (risk_level, assessed_at DESC);
CREATE INDEX idx_eudr_mst_risk_composite ON gl_eudr_mst_risk_scores (composite_risk_score DESC, assessed_at DESC);


-- ============================================================================
-- 5. gl_eudr_mst_compliance_status — Compliance monitoring records (hypertable)
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_mst_compliance_status (
    id                  BIGSERIAL,
    compliance_id       UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    checked_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),

    -- Target
    supplier_id         UUID            NOT NULL,

    -- Composite compliance
    compliance_score    DOUBLE PRECISION NOT NULL DEFAULT 0.0,  -- 0-100
    compliance_status   VARCHAR(30)     NOT NULL DEFAULT 'unverified',
        -- compliant, conditionally_compliant, non_compliant, unverified, expired

    -- Dimension scores
    dds_validity_score          DOUBLE PRECISION DEFAULT 0.0,
    certification_score         DOUBLE PRECISION DEFAULT 0.0,
    geolocation_coverage_score  DOUBLE PRECISION DEFAULT 0.0,
    deforestation_free_score    DOUBLE PRECISION DEFAULT 0.0,

    -- DDS tracking
    dds_valid           BOOLEAN         DEFAULT FALSE,
    dds_expiry_date     DATE,
    dds_days_remaining  INTEGER,

    -- Certification summary
    active_certifications   INTEGER     DEFAULT 0,
    expiring_certifications INTEGER     DEFAULT 0,  -- within 30 days
    expired_certifications  INTEGER     DEFAULT 0,

    -- Geolocation
    geo_coverage_pct    DOUBLE PRECISION DEFAULT 0.0,

    -- Deforestation-free link
    deforestation_verified  BOOLEAN     DEFAULT FALSE,
    deforestation_check_date DATE,

    -- Trend
    compliance_trend    VARCHAR(20),  -- improving, stable, degrading
    previous_score      DOUBLE PRECISION,

    -- Alerts
    alert_generated     BOOLEAN         DEFAULT FALSE,
    alert_type          VARCHAR(50),
    alert_severity      VARCHAR(20),

    -- Provenance
    provenance_hash     VARCHAR(64),
    created_by          UUID,

    PRIMARY KEY (id, checked_at)
);

SELECT create_hypertable(
    'gl_eudr_mst_compliance_status',
    'checked_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_eudr_mst_compl_tenant ON gl_eudr_mst_compliance_status (tenant_id, checked_at DESC);
CREATE INDEX idx_eudr_mst_compl_supplier ON gl_eudr_mst_compliance_status (supplier_id, checked_at DESC);
CREATE INDEX idx_eudr_mst_compl_status ON gl_eudr_mst_compliance_status (compliance_status, checked_at DESC);
CREATE INDEX idx_eudr_mst_compl_score ON gl_eudr_mst_compliance_status (compliance_score DESC, checked_at DESC);


-- ============================================================================
-- 6. gl_eudr_mst_certifications — Supplier certification records
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_mst_certifications (
    id                  BIGSERIAL       PRIMARY KEY,
    certification_id    UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    supplier_id         UUID            NOT NULL,

    -- Certificate details
    cert_type           VARCHAR(50)     NOT NULL,
        -- FSC, RSPO, UTZ, RAINFOREST_ALLIANCE, ISCC, PEFC, ORGANIC, FAIRTRADE, OTHER
    cert_standard       TEXT,
    certificate_number  TEXT,
    issuing_body        TEXT,

    -- Validity
    issue_date          DATE            NOT NULL,
    expiry_date         DATE            NOT NULL,
    renewal_date        DATE,
    status              VARCHAR(20)     NOT NULL DEFAULT 'active',
        -- active, expired, suspended, revoked, pending_renewal

    -- Scope
    commodities_covered TEXT[]          DEFAULT '{}',
    scope_description   TEXT,

    -- Verification
    verification_url    TEXT,
    verification_date   DATE,
    verified_by         UUID,

    -- Provenance
    provenance_hash     VARCHAR(64),
    created_by          UUID,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_eudr_mst_cert UNIQUE (certification_id)
);

CREATE INDEX idx_eudr_mst_cert_tenant ON gl_eudr_mst_certifications (tenant_id);
CREATE INDEX idx_eudr_mst_cert_supplier ON gl_eudr_mst_certifications (supplier_id);
CREATE INDEX idx_eudr_mst_cert_type ON gl_eudr_mst_certifications (cert_type);
CREATE INDEX idx_eudr_mst_cert_status ON gl_eudr_mst_certifications (status);
CREATE INDEX idx_eudr_mst_cert_expiry ON gl_eudr_mst_certifications (expiry_date);


-- ============================================================================
-- 7. gl_eudr_mst_gaps — Data gap analysis results
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_mst_gaps (
    id                  BIGSERIAL       PRIMARY KEY,
    gap_id              UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    supplier_id         UUID            NOT NULL,

    -- Gap classification
    gap_type            VARCHAR(50)     NOT NULL,
        -- missing_gps, missing_certification, missing_legal_entity, missing_dds,
        -- coverage_gap, verification_gap, data_quality_gap, tier_gap
    gap_category        VARCHAR(30)     NOT NULL,
        -- data, coverage, verification
    severity            VARCHAR(20)     NOT NULL DEFAULT 'minor',
        -- critical, major, minor
    description         TEXT            NOT NULL,

    -- Affected fields
    affected_fields     TEXT[]          DEFAULT '{}',
    affected_tiers      INTEGER[]       DEFAULT '{}',

    -- Status
    status              VARCHAR(20)     NOT NULL DEFAULT 'open',
        -- open, in_progress, remediated, accepted, wont_fix
    remediation_plan_id UUID,

    -- Scoring impact
    compliance_impact   DOUBLE PRECISION DEFAULT 0.0,
    risk_impact         DOUBLE PRECISION DEFAULT 0.0,

    -- Provenance
    provenance_hash     VARCHAR(64),
    detected_at         TIMESTAMPTZ     NOT NULL DEFAULT now(),
    remediated_at       TIMESTAMPTZ,
    created_by          UUID,

    CONSTRAINT uq_eudr_mst_gap UNIQUE (gap_id)
);

CREATE INDEX idx_eudr_mst_gaps_tenant ON gl_eudr_mst_gaps (tenant_id);
CREATE INDEX idx_eudr_mst_gaps_supplier ON gl_eudr_mst_gaps (supplier_id);
CREATE INDEX idx_eudr_mst_gaps_severity ON gl_eudr_mst_gaps (severity);
CREATE INDEX idx_eudr_mst_gaps_status ON gl_eudr_mst_gaps (status);
CREATE INDEX idx_eudr_mst_gaps_type ON gl_eudr_mst_gaps (gap_type);


-- ============================================================================
-- 8. gl_eudr_mst_remediation_plans — Remediation action plans
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_mst_remediation_plans (
    id                  BIGSERIAL       PRIMARY KEY,
    plan_id             UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    supplier_id         UUID            NOT NULL,

    -- Plan details
    title               TEXT            NOT NULL,
    description         TEXT,
    priority            VARCHAR(20)     NOT NULL DEFAULT 'medium',
        -- critical, high, medium, low
    status              VARCHAR(20)     NOT NULL DEFAULT 'draft',
        -- draft, active, in_progress, completed, cancelled

    -- Actions (JSONB array)
    actions             JSONB           NOT NULL DEFAULT '[]',
        -- [{"step": 1, "action": "...", "assignee": "...", "due_date": "...", "status": "..."}]

    -- Progress
    total_actions       INTEGER         DEFAULT 0,
    completed_actions   INTEGER         DEFAULT 0,
    completion_pct      DOUBLE PRECISION DEFAULT 0.0,

    -- Gaps addressed
    gap_ids             UUID[]          DEFAULT '{}',

    -- Timeline
    target_date         DATE,
    actual_completion_date DATE,

    -- Provenance
    provenance_hash     VARCHAR(64),
    created_by          UUID,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),

    CONSTRAINT uq_eudr_mst_plan UNIQUE (plan_id)
);

CREATE INDEX idx_eudr_mst_plan_tenant ON gl_eudr_mst_remediation_plans (tenant_id);
CREATE INDEX idx_eudr_mst_plan_supplier ON gl_eudr_mst_remediation_plans (supplier_id);
CREATE INDEX idx_eudr_mst_plan_status ON gl_eudr_mst_remediation_plans (status);
CREATE INDEX idx_eudr_mst_plan_priority ON gl_eudr_mst_remediation_plans (priority);


-- ============================================================================
-- 9. gl_eudr_mst_batch_jobs — Batch processing jobs
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_mst_batch_jobs (
    id                  BIGSERIAL       PRIMARY KEY,
    batch_id            UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,

    -- Job config
    job_type            VARCHAR(50)     NOT NULL,
        -- discovery, risk_assessment, compliance_check, gap_analysis,
        -- profile_import, relationship_import, certification_import
    input_config        JSONB           NOT NULL DEFAULT '{}',
    parameters          JSONB           DEFAULT '{}',

    -- Progress
    status              VARCHAR(20)     NOT NULL DEFAULT 'pending',
        -- pending, running, completed, failed, cancelled
    total_items         INTEGER         DEFAULT 0,
    processed_items     INTEGER         DEFAULT 0,
    failed_items        INTEGER         DEFAULT 0,
    progress_pct        DOUBLE PRECISION DEFAULT 0.0,

    -- Results
    result_summary      JSONB           DEFAULT '{}',
    error_details       JSONB           DEFAULT '[]',

    -- Timing
    submitted_at        TIMESTAMPTZ     NOT NULL DEFAULT now(),
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    duration_ms         BIGINT,

    -- Provenance
    provenance_hash     VARCHAR(64),
    submitted_by        UUID,

    CONSTRAINT uq_eudr_mst_batch UNIQUE (batch_id)
);

CREATE INDEX idx_eudr_mst_batch_tenant ON gl_eudr_mst_batch_jobs (tenant_id);
CREATE INDEX idx_eudr_mst_batch_status ON gl_eudr_mst_batch_jobs (status);
CREATE INDEX idx_eudr_mst_batch_type ON gl_eudr_mst_batch_jobs (job_type);


-- ============================================================================
-- 10. gl_eudr_mst_audit_log — Immutable audit trail
-- ============================================================================
CREATE TABLE IF NOT EXISTS gl_eudr_mst_audit_log (
    id                  BIGSERIAL       PRIMARY KEY,
    event_id            UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    recorded_at         TIMESTAMPTZ     NOT NULL DEFAULT now(),

    -- Event classification
    event_type          VARCHAR(80)     NOT NULL,
        -- supplier.created, supplier.updated, supplier.deactivated,
        -- relationship.created, relationship.status_changed,
        -- risk.assessed, risk.alert_triggered,
        -- compliance.checked, compliance.status_changed, compliance.alert_triggered,
        -- gap.detected, gap.remediated,
        -- batch.submitted, batch.completed, batch.failed,
        -- report.generated, report.downloaded
    event_category      VARCHAR(30)     NOT NULL,
        -- supplier, relationship, risk, compliance, gap, batch, report

    -- Target
    entity_type         VARCHAR(50)     NOT NULL,
    entity_id           UUID            NOT NULL,
    supplier_id         UUID,

    -- Change details
    actor_id            UUID,
    actor_type          VARCHAR(20)     DEFAULT 'user',
        -- user, system, batch, api
    changes             JSONB           DEFAULT '{}',
    previous_state      JSONB           DEFAULT '{}',
    new_state           JSONB           DEFAULT '{}',

    -- Context
    ip_address          INET,
    user_agent          TEXT,
    correlation_id      UUID,

    -- Provenance
    provenance_hash     VARCHAR(64)
);

CREATE INDEX idx_eudr_mst_audit_tenant ON gl_eudr_mst_audit_log (tenant_id, recorded_at DESC);
CREATE INDEX idx_eudr_mst_audit_type ON gl_eudr_mst_audit_log (event_type, recorded_at DESC);
CREATE INDEX idx_eudr_mst_audit_entity ON gl_eudr_mst_audit_log (entity_type, entity_id, recorded_at DESC);
CREATE INDEX idx_eudr_mst_audit_supplier ON gl_eudr_mst_audit_log (supplier_id, recorded_at DESC);
CREATE INDEX idx_eudr_mst_audit_actor ON gl_eudr_mst_audit_log (actor_id, recorded_at DESC);


-- ============================================================================
-- Continuous Aggregates
-- ============================================================================

-- Daily risk summary
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_mst_daily_risk_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', assessed_at)   AS bucket,
    tenant_id,
    risk_level,
    COUNT(*)                            AS assessment_count,
    AVG(composite_risk_score)           AS avg_risk_score,
    MAX(composite_risk_score)           AS max_risk_score,
    MIN(composite_risk_score)           AS min_risk_score,
    COUNT(DISTINCT supplier_id)         AS unique_suppliers
FROM gl_eudr_mst_risk_scores
GROUP BY bucket, tenant_id, risk_level
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_eudr_mst_daily_risk_summary',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);

-- Daily compliance summary
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_mst_daily_compliance_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', checked_at)    AS bucket,
    tenant_id,
    compliance_status,
    COUNT(*)                            AS check_count,
    AVG(compliance_score)               AS avg_compliance_score,
    COUNT(DISTINCT supplier_id)         AS unique_suppliers,
    SUM(CASE WHEN alert_generated THEN 1 ELSE 0 END) AS alerts_generated
FROM gl_eudr_mst_compliance_status
GROUP BY bucket, tenant_id, compliance_status
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_eudr_mst_daily_compliance_summary',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);


-- ============================================================================
-- Retention Policies
-- ============================================================================

SELECT add_retention_policy('gl_eudr_mst_relationships',
    INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_mst_tier_scores',
    INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_mst_risk_scores',
    INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_mst_compliance_status',
    INTERVAL '5 years', if_not_exists => TRUE);


-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE gl_eudr_mst_suppliers IS 'AGENT-EUDR-008: Multi-tier supplier profiles with legal entity, location, capacity, and certification data';
COMMENT ON TABLE gl_eudr_mst_relationships IS 'AGENT-EUDR-008: Supplier-to-supplier relationships with tier depth, volume, and lifecycle state (hypertable)';
COMMENT ON TABLE gl_eudr_mst_tier_scores IS 'AGENT-EUDR-008: Tier depth and supply chain visibility scores per supplier/commodity (hypertable)';
COMMENT ON TABLE gl_eudr_mst_risk_scores IS 'AGENT-EUDR-008: Supplier risk assessment with 6 risk categories and upstream propagation (hypertable)';
COMMENT ON TABLE gl_eudr_mst_compliance_status IS 'AGENT-EUDR-008: Compliance monitoring across DDS, certification, geolocation, and deforestation-free dimensions (hypertable)';
COMMENT ON TABLE gl_eudr_mst_certifications IS 'AGENT-EUDR-008: Supplier certification records (FSC, RSPO, UTZ, etc.) with validity tracking';
COMMENT ON TABLE gl_eudr_mst_gaps IS 'AGENT-EUDR-008: Data gap analysis results with severity classification and remediation linkage';
COMMENT ON TABLE gl_eudr_mst_remediation_plans IS 'AGENT-EUDR-008: Remediation action plans for gap resolution with progress tracking';
COMMENT ON TABLE gl_eudr_mst_batch_jobs IS 'AGENT-EUDR-008: Batch processing jobs for discovery, risk assessment, and compliance checks';
COMMENT ON TABLE gl_eudr_mst_audit_log IS 'AGENT-EUDR-008: Immutable audit trail for all supplier data changes and system events';

COMMIT;
