-- =============================================================================
-- V246: PACK-033 Quick Wins Identifier - Core Schema & Scan Management
-- =============================================================================
-- Pack:         PACK-033 (Quick Wins Identifier Pack)
-- Migration:    001 of 010
-- Date:         March 2026
--
-- Creates the pack033_quick_wins schema and foundational tables for quick-win
-- energy efficiency action identification. Tracks facility scans, individual
-- scan results, and the master action library.
--
-- Tables (3):
--   1. pack033_quick_wins.quick_wins_scans
--   2. pack033_quick_wins.scan_results
--   3. pack033_quick_wins.action_library
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V245__PACK030_permissions.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS pack033_quick_wins;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION pack033_quick_wins.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: pack033_quick_wins.quick_wins_scans
-- =============================================================================
-- Facility-level scans that identify quick-win energy efficiency actions.
-- Each scan captures aggregate savings potential and scan metadata.

CREATE TABLE pack033_quick_wins.quick_wins_scans (
    scan_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    scan_type               VARCHAR(50)     NOT NULL,
    building_type           VARCHAR(100),
    scan_date               DATE            NOT NULL DEFAULT CURRENT_DATE,
    status                  VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    total_actions_found     INTEGER         DEFAULT 0,
    total_savings_kwh       NUMERIC(16,2)   DEFAULT 0,
    total_savings_cost      NUMERIC(16,2)   DEFAULT 0,
    total_co2e_reduction    NUMERIC(16,4)   DEFAULT 0,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p033_qs_scan_type CHECK (
        scan_type IN ('WALKTHROUGH', 'DETAILED', 'INVESTMENT_GRADE', 'BEHAVIORAL', 'REMOTE', 'HYBRID')
    ),
    CONSTRAINT chk_p033_qs_status CHECK (
        status IN ('DRAFT', 'IN_PROGRESS', 'COMPLETED', 'APPROVED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p033_qs_building_type CHECK (
        building_type IS NULL OR building_type IN (
            'OFFICE', 'RETAIL', 'WAREHOUSE', 'INDUSTRIAL', 'RESIDENTIAL',
            'HOTEL', 'HOSPITAL', 'SCHOOL', 'UNIVERSITY', 'DATA_CENTER',
            'MIXED_USE', 'LABORATORY', 'LEISURE', 'PUBLIC_BUILDING', 'OTHER'
        )
    ),
    CONSTRAINT chk_p033_qs_actions_found CHECK (
        total_actions_found >= 0
    ),
    CONSTRAINT chk_p033_qs_savings_kwh CHECK (
        total_savings_kwh >= 0
    ),
    CONSTRAINT chk_p033_qs_savings_cost CHECK (
        total_savings_cost >= 0
    ),
    CONSTRAINT chk_p033_qs_co2e CHECK (
        total_co2e_reduction >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_qs_tenant        ON pack033_quick_wins.quick_wins_scans(tenant_id);
CREATE INDEX idx_p033_qs_facility      ON pack033_quick_wins.quick_wins_scans(facility_id);
CREATE INDEX idx_p033_qs_scan_type     ON pack033_quick_wins.quick_wins_scans(scan_type);
CREATE INDEX idx_p033_qs_status        ON pack033_quick_wins.quick_wins_scans(status);
CREATE INDEX idx_p033_qs_scan_date     ON pack033_quick_wins.quick_wins_scans(scan_date DESC);
CREATE INDEX idx_p033_qs_created       ON pack033_quick_wins.quick_wins_scans(created_at DESC);
CREATE INDEX idx_p033_qs_metadata      ON pack033_quick_wins.quick_wins_scans USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p033_qs_updated
    BEFORE UPDATE ON pack033_quick_wins.quick_wins_scans
    FOR EACH ROW EXECUTE FUNCTION pack033_quick_wins.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack033_quick_wins.scan_results
-- =============================================================================
-- Individual action results from a scan, each linked to an action in the
-- library. Contains estimated savings, payback, and confidence levels.

CREATE TABLE pack033_quick_wins.scan_results (
    result_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id                 UUID            NOT NULL REFERENCES pack033_quick_wins.quick_wins_scans(scan_id) ON DELETE CASCADE,
    action_id               UUID,
    category                VARCHAR(100)    NOT NULL,
    subcategory             VARCHAR(100),
    description             TEXT            NOT NULL,
    priority_score          NUMERIC(6,2),
    estimated_savings_kwh   NUMERIC(14,2),
    estimated_savings_cost  NUMERIC(14,2),
    estimated_co2e          NUMERIC(14,4),
    payback_months          NUMERIC(8,2),
    implementation_cost     NUMERIC(14,2),
    confidence_level        VARCHAR(20)     DEFAULT 'MEDIUM',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p033_sr_priority CHECK (
        priority_score IS NULL OR (priority_score >= 0 AND priority_score <= 100)
    ),
    CONSTRAINT chk_p033_sr_savings_kwh CHECK (
        estimated_savings_kwh IS NULL OR estimated_savings_kwh >= 0
    ),
    CONSTRAINT chk_p033_sr_savings_cost CHECK (
        estimated_savings_cost IS NULL OR estimated_savings_cost >= 0
    ),
    CONSTRAINT chk_p033_sr_co2e CHECK (
        estimated_co2e IS NULL OR estimated_co2e >= 0
    ),
    CONSTRAINT chk_p033_sr_payback CHECK (
        payback_months IS NULL OR payback_months >= 0
    ),
    CONSTRAINT chk_p033_sr_impl_cost CHECK (
        implementation_cost IS NULL OR implementation_cost >= 0
    ),
    CONSTRAINT chk_p033_sr_confidence CHECK (
        confidence_level IN ('LOW', 'MEDIUM', 'HIGH', 'VERIFIED')
    ),
    CONSTRAINT chk_p033_sr_category CHECK (
        category IN ('LIGHTING', 'HVAC', 'CONTROLS', 'ENVELOPE', 'PLUG_LOADS',
                      'MOTORS', 'COMPRESSED_AIR', 'STEAM', 'WATER_HEATING',
                      'BEHAVIORAL', 'PROCESS', 'RENEWABLE', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_sr_scan          ON pack033_quick_wins.scan_results(scan_id);
CREATE INDEX idx_p033_sr_action        ON pack033_quick_wins.scan_results(action_id);
CREATE INDEX idx_p033_sr_category      ON pack033_quick_wins.scan_results(category);
CREATE INDEX idx_p033_sr_priority      ON pack033_quick_wins.scan_results(priority_score DESC);
CREATE INDEX idx_p033_sr_payback       ON pack033_quick_wins.scan_results(payback_months);
CREATE INDEX idx_p033_sr_confidence    ON pack033_quick_wins.scan_results(confidence_level);
CREATE INDEX idx_p033_sr_created       ON pack033_quick_wins.scan_results(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p033_sr_updated
    BEFORE UPDATE ON pack033_quick_wins.scan_results
    FOR EACH ROW EXECUTE FUNCTION pack033_quick_wins.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack033_quick_wins.action_library
-- =============================================================================
-- Master library of quick-win energy efficiency actions with typical savings,
-- payback, complexity, and applicability metadata.

CREATE TABLE pack033_quick_wins.action_library (
    action_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    action_code             VARCHAR(50)     NOT NULL,
    category                VARCHAR(100)    NOT NULL,
    subcategory             VARCHAR(100),
    title                   VARCHAR(500)    NOT NULL,
    description             TEXT,
    typical_savings_pct     NUMERIC(6,2),
    typical_payback_months  NUMERIC(8,2),
    complexity              VARCHAR(20)     NOT NULL DEFAULT 'LOW',
    disruption_level        VARCHAR(20)     NOT NULL DEFAULT 'LOW',
    applicable_building_types TEXT[],
    applicable_sectors      TEXT[],
    is_behavioral           BOOLEAN         DEFAULT FALSE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT uq_p033_al_action_code UNIQUE (tenant_id, action_code),
    CONSTRAINT chk_p033_al_savings_pct CHECK (
        typical_savings_pct IS NULL OR (typical_savings_pct >= 0 AND typical_savings_pct <= 100)
    ),
    CONSTRAINT chk_p033_al_payback CHECK (
        typical_payback_months IS NULL OR typical_payback_months >= 0
    ),
    CONSTRAINT chk_p033_al_complexity CHECK (
        complexity IN ('LOW', 'MEDIUM', 'HIGH')
    ),
    CONSTRAINT chk_p033_al_disruption CHECK (
        disruption_level IN ('NONE', 'LOW', 'MEDIUM', 'HIGH')
    ),
    CONSTRAINT chk_p033_al_category CHECK (
        category IN ('LIGHTING', 'HVAC', 'CONTROLS', 'ENVELOPE', 'PLUG_LOADS',
                      'MOTORS', 'COMPRESSED_AIR', 'STEAM', 'WATER_HEATING',
                      'BEHAVIORAL', 'PROCESS', 'RENEWABLE', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_al_tenant        ON pack033_quick_wins.action_library(tenant_id);
CREATE INDEX idx_p033_al_code          ON pack033_quick_wins.action_library(action_code);
CREATE INDEX idx_p033_al_category      ON pack033_quick_wins.action_library(category);
CREATE INDEX idx_p033_al_complexity    ON pack033_quick_wins.action_library(complexity);
CREATE INDEX idx_p033_al_behavioral    ON pack033_quick_wins.action_library(is_behavioral);
CREATE INDEX idx_p033_al_building_types ON pack033_quick_wins.action_library USING GIN(applicable_building_types);
CREATE INDEX idx_p033_al_sectors       ON pack033_quick_wins.action_library USING GIN(applicable_sectors);
CREATE INDEX idx_p033_al_metadata      ON pack033_quick_wins.action_library USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p033_al_updated
    BEFORE UPDATE ON pack033_quick_wins.action_library
    FOR EACH ROW EXECUTE FUNCTION pack033_quick_wins.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack033_quick_wins.quick_wins_scans ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack033_quick_wins.scan_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack033_quick_wins.action_library ENABLE ROW LEVEL SECURITY;

CREATE POLICY p033_qs_tenant_isolation
    ON pack033_quick_wins.quick_wins_scans
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p033_qs_service_bypass
    ON pack033_quick_wins.quick_wins_scans
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p033_sr_tenant_isolation
    ON pack033_quick_wins.scan_results
    USING (scan_id IN (
        SELECT scan_id FROM pack033_quick_wins.quick_wins_scans
        WHERE tenant_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p033_sr_service_bypass
    ON pack033_quick_wins.scan_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p033_al_tenant_isolation
    ON pack033_quick_wins.action_library
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p033_al_service_bypass
    ON pack033_quick_wins.action_library
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA pack033_quick_wins TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.quick_wins_scans TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.scan_results TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.action_library TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack033_quick_wins.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA pack033_quick_wins IS
    'PACK-033 Quick Wins Identifier Pack - identifies, prioritizes, and tracks low-cost, high-impact energy efficiency actions with rapid payback.';

COMMENT ON TABLE pack033_quick_wins.quick_wins_scans IS
    'Facility-level scans that identify quick-win energy efficiency actions with aggregate savings potential.';

COMMENT ON TABLE pack033_quick_wins.scan_results IS
    'Individual action results from a scan with estimated savings, payback period, and confidence levels.';

COMMENT ON TABLE pack033_quick_wins.action_library IS
    'Master library of quick-win energy efficiency actions with typical savings, payback, complexity, and applicability.';

COMMENT ON COLUMN pack033_quick_wins.quick_wins_scans.scan_id IS
    'Unique identifier for the facility scan.';
COMMENT ON COLUMN pack033_quick_wins.quick_wins_scans.tenant_id IS
    'Multi-tenant isolation key.';
COMMENT ON COLUMN pack033_quick_wins.quick_wins_scans.facility_id IS
    'Reference to the facility being scanned.';
COMMENT ON COLUMN pack033_quick_wins.quick_wins_scans.scan_type IS
    'Type of scan performed (WALKTHROUGH, DETAILED, INVESTMENT_GRADE, BEHAVIORAL, REMOTE, HYBRID).';
COMMENT ON COLUMN pack033_quick_wins.quick_wins_scans.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack033_quick_wins.scan_results.confidence_level IS
    'Confidence in savings estimate (LOW, MEDIUM, HIGH, VERIFIED).';
COMMENT ON COLUMN pack033_quick_wins.action_library.action_code IS
    'Unique code per tenant for referencing actions (e.g., QW-LIGHT-001).';
COMMENT ON COLUMN pack033_quick_wins.action_library.is_behavioral IS
    'Whether this action is behavioral (no capital cost) versus equipment-based.';
COMMENT ON COLUMN pack033_quick_wins.action_library.applicable_building_types IS
    'Array of building types this action applies to.';
