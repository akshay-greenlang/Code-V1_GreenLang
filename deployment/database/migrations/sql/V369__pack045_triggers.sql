-- =============================================================================
-- V369: PACK-045 Base Year Management Pack - Recalculation Triggers
-- =============================================================================
-- Pack:         PACK-045 (Base Year Management Pack)
-- Migration:    004 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates the trigger detection table that records events which may require
-- base year recalculation. Triggers are detected automatically (e.g., M&A
-- events, methodology changes) or raised manually by users. Each trigger
-- tracks its type, detection method, estimated emission impact, significance
-- percentage, and current workflow status. Triggers flow into significance
-- assessments (V370) and then into adjustment packages (V371).
--
-- Tables (1):
--   1. ghg_base_year.gl_by_triggers
--
-- Also includes: indexes, RLS, DB trigger for status audit, comments.
-- Previous: V368__pack045_recalculation_policy.sql
-- =============================================================================

SET search_path TO ghg_base_year, public;

-- =============================================================================
-- Table 1: ghg_base_year.gl_by_triggers
-- =============================================================================
-- Records events that may trigger base year recalculation. Each trigger is
-- evaluated against the recalculation policy thresholds to determine whether
-- recalculation is required. Triggers may be auto-detected by the platform
-- (e.g., boundary change, emission factor update) or manually raised.

CREATE TABLE ghg_base_year.gl_by_triggers (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    base_year_id                UUID            NOT NULL REFERENCES ghg_base_year.gl_by_base_years(id) ON DELETE CASCADE,
    trigger_type                VARCHAR(40)     NOT NULL,
    trigger_subtype             VARCHAR(60),
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DETECTED',
    detection_method            VARCHAR(30)     NOT NULL DEFAULT 'AUTOMATIC',
    detected_date               DATE            NOT NULL DEFAULT CURRENT_DATE,
    detected_by                 VARCHAR(255),
    event_date                  DATE,
    event_description           TEXT            NOT NULL,
    affected_scope              VARCHAR(10),
    affected_category           VARCHAR(60),
    affected_facility_ids       UUID[],
    emission_impact_tco2e       NUMERIC(14,3),
    significance_pct            NUMERIC(6,3),
    resolution                  VARCHAR(30),
    resolution_date             DATE,
    resolution_notes            TEXT,
    related_trigger_ids         UUID[],
    details_json                JSONB           DEFAULT '{}',
    evidence_refs               TEXT[],
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p045_trg_type CHECK (
        trigger_type IN (
            'ACQUISITION', 'DIVESTITURE', 'MERGER', 'OUTSOURCING', 'INSOURCING',
            'METHODOLOGY_CHANGE', 'EMISSION_FACTOR_UPDATE', 'GWP_VERSION_CHANGE',
            'ERROR_CORRECTION', 'DATA_IMPROVEMENT', 'BOUNDARY_CHANGE',
            'STRUCTURAL_CHANGE', 'PRODUCTION_CHANGE', 'FUEL_MIX_CHANGE',
            'REGULATORY_MANDATE', 'CAPACITY_CHANGE', 'SECTOR_RECLASSIFICATION',
            'ORGANIC_GROWTH_DECLINE', 'OTHER'
        )
    ),
    CONSTRAINT chk_p045_trg_status CHECK (
        status IN (
            'DETECTED', 'UNDER_ASSESSMENT', 'SIGNIFICANT', 'NOT_SIGNIFICANT',
            'RECALCULATION_REQUIRED', 'RECALCULATION_COMPLETE', 'DEFERRED',
            'DISMISSED', 'ARCHIVED'
        )
    ),
    CONSTRAINT chk_p045_trg_detection CHECK (
        detection_method IN ('AUTOMATIC', 'MANUAL', 'SCHEDULED_REVIEW', 'EXTERNAL_AUDIT')
    ),
    CONSTRAINT chk_p045_trg_scope CHECK (
        affected_scope IS NULL OR affected_scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'ALL')
    ),
    CONSTRAINT chk_p045_trg_resolution CHECK (
        resolution IS NULL OR resolution IN (
            'RECALCULATED', 'BELOW_THRESHOLD', 'DE_MINIMIS', 'DEFERRED_TO_NEXT_CYCLE',
            'NOT_APPLICABLE', 'DISMISSED_WITH_JUSTIFICATION'
        )
    ),
    CONSTRAINT chk_p045_trg_impact CHECK (
        emission_impact_tco2e IS NULL OR emission_impact_tco2e >= 0
    ),
    CONSTRAINT chk_p045_trg_significance CHECK (
        significance_pct IS NULL OR (significance_pct >= 0 AND significance_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p045_trg_tenant         ON ghg_base_year.gl_by_triggers(tenant_id);
CREATE INDEX idx_p045_trg_org            ON ghg_base_year.gl_by_triggers(org_id);
CREATE INDEX idx_p045_trg_base_year      ON ghg_base_year.gl_by_triggers(base_year_id);
CREATE INDEX idx_p045_trg_type           ON ghg_base_year.gl_by_triggers(trigger_type);
CREATE INDEX idx_p045_trg_status         ON ghg_base_year.gl_by_triggers(status);
CREATE INDEX idx_p045_trg_detection      ON ghg_base_year.gl_by_triggers(detection_method);
CREATE INDEX idx_p045_trg_detected_date  ON ghg_base_year.gl_by_triggers(detected_date);
CREATE INDEX idx_p045_trg_resolution     ON ghg_base_year.gl_by_triggers(resolution);
CREATE INDEX idx_p045_trg_scope          ON ghg_base_year.gl_by_triggers(affected_scope);
CREATE INDEX idx_p045_trg_provenance     ON ghg_base_year.gl_by_triggers(provenance_hash);
CREATE INDEX idx_p045_trg_created        ON ghg_base_year.gl_by_triggers(created_at DESC);
CREATE INDEX idx_p045_trg_details        ON ghg_base_year.gl_by_triggers USING GIN(details_json);
CREATE INDEX idx_p045_trg_metadata       ON ghg_base_year.gl_by_triggers USING GIN(metadata);

-- Composite: org + open triggers for dashboard
CREATE INDEX idx_p045_trg_org_open       ON ghg_base_year.gl_by_triggers(org_id, trigger_type)
    WHERE status IN ('DETECTED', 'UNDER_ASSESSMENT', 'SIGNIFICANT', 'RECALCULATION_REQUIRED');

-- Composite: base_year + status for recalculation queue
CREATE INDEX idx_p045_trg_by_status      ON ghg_base_year.gl_by_triggers(base_year_id, status);

-- ---------------------------------------------------------------------------
-- Trigger: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p045_trg_updated
    BEFORE UPDATE ON ghg_base_year.gl_by_triggers
    FOR EACH ROW EXECUTE FUNCTION ghg_base_year.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- DB Trigger: log status transitions to audit trail
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION ghg_base_year.fn_trigger_status_audit()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status IS DISTINCT FROM NEW.status THEN
        INSERT INTO ghg_base_year.gl_by_audit_trail (
            tenant_id, org_id, base_year_id, event_type, actor,
            description, before_value, after_value, provenance_hash
        ) VALUES (
            NEW.tenant_id, NEW.org_id, NEW.base_year_id,
            'TRIGGER_STATUS_CHANGE', COALESCE(current_setting('app.current_user', true), 'system'),
            format('Trigger %s status changed from %s to %s', NEW.id, OLD.status, NEW.status),
            OLD.status, NEW.status, NEW.provenance_hash
        );
    END IF;
    RETURN NEW;
END;
$$;

-- Note: The audit trail trigger is created after gl_by_audit_trail exists (V374).
-- A deferred trigger creation statement is placed in V374.

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_base_year.gl_by_triggers ENABLE ROW LEVEL SECURITY;

CREATE POLICY p045_trg_tenant_isolation
    ON ghg_base_year.gl_by_triggers
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p045_trg_service_bypass
    ON ghg_base_year.gl_by_triggers
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_base_year.gl_by_triggers TO PUBLIC;
GRANT EXECUTE ON FUNCTION ghg_base_year.fn_trigger_status_audit() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_base_year.gl_by_triggers IS
    'Recalculation trigger events detected automatically or raised manually, with emission impact estimates and workflow status per GHG Protocol Chapter 5.';

COMMENT ON COLUMN ghg_base_year.gl_by_triggers.trigger_type IS 'Category of change event: ACQUISITION, DIVESTITURE, METHODOLOGY_CHANGE, ERROR_CORRECTION, etc.';
COMMENT ON COLUMN ghg_base_year.gl_by_triggers.detection_method IS 'How the trigger was identified: AUTOMATIC (platform), MANUAL (user), SCHEDULED_REVIEW, EXTERNAL_AUDIT.';
COMMENT ON COLUMN ghg_base_year.gl_by_triggers.emission_impact_tco2e IS 'Estimated absolute emission impact of the trigger event in tCO2e.';
COMMENT ON COLUMN ghg_base_year.gl_by_triggers.significance_pct IS 'Impact as percentage of base year total: (impact / base_year_total) * 100.';
COMMENT ON COLUMN ghg_base_year.gl_by_triggers.resolution IS 'Final disposition: RECALCULATED, BELOW_THRESHOLD, DE_MINIMIS, DEFERRED, NOT_APPLICABLE, DISMISSED.';
COMMENT ON COLUMN ghg_base_year.gl_by_triggers.provenance_hash IS 'SHA-256 hash for audit provenance of this trigger record.';
