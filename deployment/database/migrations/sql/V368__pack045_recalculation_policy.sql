-- =============================================================================
-- V368: PACK-045 Base Year Management Pack - Recalculation Policy
-- =============================================================================
-- Pack:         PACK-045 (Base Year Management Pack)
-- Migration:    003 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates recalculation policy tables that define when and how base year
-- emissions must be recalculated. Policies specify significance thresholds,
-- trigger rules, approval workflows, and effective dates. Policy versioning
-- tracks the evolution of recalculation rules over time per GHG Protocol
-- Chapter 5 requirements.
--
-- Tables (2):
--   1. ghg_base_year.gl_by_policies
--   2. ghg_base_year.gl_by_policy_versions
--
-- Also includes: indexes, RLS, comments.
-- Previous: V367__pack045_inventory.sql
-- =============================================================================

SET search_path TO ghg_base_year, public;

-- =============================================================================
-- Table 1: ghg_base_year.gl_by_policies
-- =============================================================================
-- Recalculation policies defining the rules for when base year recalculation
-- is triggered. Each policy specifies threshold percentages, trigger types,
-- approval requirements, and effective dates. An organisation may have
-- separate policies for different policy types (e.g., structural changes
-- vs. methodology changes).

CREATE TABLE ghg_base_year.gl_by_policies (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    policy_name                 VARCHAR(255)    NOT NULL,
    policy_type                 VARCHAR(30)     NOT NULL DEFAULT 'GENERAL',
    significance_threshold_pct  NUMERIC(5,2)    NOT NULL DEFAULT 5.00,
    cumulative_threshold_pct    NUMERIC(5,2)    DEFAULT 10.00,
    de_minimis_threshold_tco2e  NUMERIC(12,3)   DEFAULT 100.000,
    trigger_rules_json          JSONB           NOT NULL DEFAULT '{}',
    approval_workflow_json      JSONB           DEFAULT '{}',
    escalation_rules_json       JSONB           DEFAULT '{}',
    effective_date              DATE            NOT NULL DEFAULT CURRENT_DATE,
    expiry_date                 DATE,
    version                     INTEGER         NOT NULL DEFAULT 1,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    approved_by                 VARCHAR(255),
    approved_date               DATE,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p045_pol_type CHECK (
        policy_type IN (
            'GENERAL', 'STRUCTURAL', 'METHODOLOGICAL', 'ERROR_CORRECTION',
            'OUTSOURCING', 'REGULATORY', 'SECTOR_SPECIFIC'
        )
    ),
    CONSTRAINT chk_p045_pol_threshold CHECK (
        significance_threshold_pct > 0 AND significance_threshold_pct <= 100
    ),
    CONSTRAINT chk_p045_pol_cumulative CHECK (
        cumulative_threshold_pct IS NULL OR (cumulative_threshold_pct > 0 AND cumulative_threshold_pct <= 100)
    ),
    CONSTRAINT chk_p045_pol_deminimis CHECK (
        de_minimis_threshold_tco2e IS NULL OR de_minimis_threshold_tco2e >= 0
    ),
    CONSTRAINT chk_p045_pol_version CHECK (
        version >= 1
    ),
    CONSTRAINT chk_p045_pol_status CHECK (
        status IN ('DRAFT', 'UNDER_REVIEW', 'APPROVED', 'SUPERSEDED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p045_pol_dates CHECK (
        expiry_date IS NULL OR expiry_date > effective_date
    ),
    CONSTRAINT uq_p045_pol_org_type_version UNIQUE (org_id, policy_type, version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p045_pol_tenant         ON ghg_base_year.gl_by_policies(tenant_id);
CREATE INDEX idx_p045_pol_org            ON ghg_base_year.gl_by_policies(org_id);
CREATE INDEX idx_p045_pol_type           ON ghg_base_year.gl_by_policies(policy_type);
CREATE INDEX idx_p045_pol_status         ON ghg_base_year.gl_by_policies(status);
CREATE INDEX idx_p045_pol_active         ON ghg_base_year.gl_by_policies(is_active) WHERE is_active = true;
CREATE INDEX idx_p045_pol_effective      ON ghg_base_year.gl_by_policies(effective_date);
CREATE INDEX idx_p045_pol_created        ON ghg_base_year.gl_by_policies(created_at DESC);
CREATE INDEX idx_p045_pol_trigger_rules  ON ghg_base_year.gl_by_policies USING GIN(trigger_rules_json);
CREATE INDEX idx_p045_pol_approval       ON ghg_base_year.gl_by_policies USING GIN(approval_workflow_json);

-- Composite: org + active policies
CREATE INDEX idx_p045_pol_org_active     ON ghg_base_year.gl_by_policies(org_id, policy_type)
    WHERE is_active = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p045_pol_updated
    BEFORE UPDATE ON ghg_base_year.gl_by_policies
    FOR EACH ROW EXECUTE FUNCTION ghg_base_year.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_base_year.gl_by_policy_versions
-- =============================================================================
-- Version history for recalculation policies. Each time a policy is modified,
-- a snapshot of the previous version is stored here for audit purposes. This
-- enables reconstruction of what policy was in effect at any point in time.

CREATE TABLE ghg_base_year.gl_by_policy_versions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    policy_id                   UUID            NOT NULL REFERENCES ghg_base_year.gl_by_policies(id) ON DELETE CASCADE,
    version_number              INTEGER         NOT NULL,
    policy_snapshot_json        JSONB           NOT NULL,
    change_reason               TEXT            NOT NULL,
    change_type                 VARCHAR(30)     NOT NULL DEFAULT 'UPDATE',
    changed_by                  VARCHAR(255),
    effective_from              DATE            NOT NULL,
    effective_to                DATE,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p045_pv_version CHECK (
        version_number >= 1
    ),
    CONSTRAINT chk_p045_pv_change_type CHECK (
        change_type IN ('CREATION', 'UPDATE', 'THRESHOLD_CHANGE', 'TRIGGER_CHANGE', 'WORKFLOW_CHANGE', 'SUPERSEDED')
    ),
    CONSTRAINT chk_p045_pv_dates CHECK (
        effective_to IS NULL OR effective_to >= effective_from
    ),
    CONSTRAINT uq_p045_pv_policy_version UNIQUE (policy_id, version_number)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p045_pv_tenant          ON ghg_base_year.gl_by_policy_versions(tenant_id);
CREATE INDEX idx_p045_pv_policy          ON ghg_base_year.gl_by_policy_versions(policy_id);
CREATE INDEX idx_p045_pv_version         ON ghg_base_year.gl_by_policy_versions(version_number);
CREATE INDEX idx_p045_pv_change_type     ON ghg_base_year.gl_by_policy_versions(change_type);
CREATE INDEX idx_p045_pv_effective_from  ON ghg_base_year.gl_by_policy_versions(effective_from);
CREATE INDEX idx_p045_pv_created         ON ghg_base_year.gl_by_policy_versions(created_at DESC);

-- Composite: policy + date range for point-in-time lookups
CREATE INDEX idx_p045_pv_policy_dates    ON ghg_base_year.gl_by_policy_versions(policy_id, effective_from, effective_to);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_base_year.gl_by_policies ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_base_year.gl_by_policy_versions ENABLE ROW LEVEL SECURITY;

CREATE POLICY p045_pol_tenant_isolation
    ON ghg_base_year.gl_by_policies
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p045_pol_service_bypass
    ON ghg_base_year.gl_by_policies
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p045_pv_tenant_isolation
    ON ghg_base_year.gl_by_policy_versions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p045_pv_service_bypass
    ON ghg_base_year.gl_by_policy_versions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_base_year.gl_by_policies TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_base_year.gl_by_policy_versions TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_base_year.gl_by_policies IS
    'Recalculation policies defining significance thresholds, trigger rules, and approval workflows per GHG Protocol Chapter 5.';
COMMENT ON TABLE ghg_base_year.gl_by_policy_versions IS
    'Version history snapshots of recalculation policies for point-in-time audit reconstruction.';

COMMENT ON COLUMN ghg_base_year.gl_by_policies.significance_threshold_pct IS 'Percentage threshold above which a change is deemed significant and triggers recalculation (e.g., 5%).';
COMMENT ON COLUMN ghg_base_year.gl_by_policies.cumulative_threshold_pct IS 'Cumulative percentage threshold for multiple small changes that individually fall below the significance threshold.';
COMMENT ON COLUMN ghg_base_year.gl_by_policies.de_minimis_threshold_tco2e IS 'Absolute emission threshold below which changes are considered immaterial regardless of percentage impact.';
COMMENT ON COLUMN ghg_base_year.gl_by_policies.trigger_rules_json IS 'JSON defining which change types (structural, methodology, error, etc.) can trigger recalculation.';
COMMENT ON COLUMN ghg_base_year.gl_by_policies.approval_workflow_json IS 'JSON defining approval requirements: minimum approvers, escalation rules, notification recipients.';
COMMENT ON COLUMN ghg_base_year.gl_by_policy_versions.policy_snapshot_json IS 'Complete JSON snapshot of the policy at this version for audit trail.';
