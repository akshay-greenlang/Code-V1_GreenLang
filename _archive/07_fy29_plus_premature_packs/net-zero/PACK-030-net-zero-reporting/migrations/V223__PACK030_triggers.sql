-- =============================================================================
-- V223: PACK-030 Net Zero Reporting Pack - Audit Triggers
-- =============================================================================
-- Pack:         PACK-030 (Net Zero Reporting Pack)
-- Migration:    013 of 015
-- Date:         March 2026
--
-- Audit triggers that automatically log lifecycle events to the audit trail
-- table when reports, sections, metrics, and other entities are created,
-- updated, or deleted.
--
-- Trigger Functions (6):
--   1. fn_audit_report_changes
--   2. fn_audit_section_changes
--   3. fn_audit_metric_changes
--   4. fn_audit_narrative_changes
--   5. fn_audit_evidence_changes
--   6. fn_audit_validation_changes
--
-- Triggers (12):
--   2 per function (INSERT/UPDATE, DELETE)
--
-- Previous: V222__PACK030_functions.sql
-- =============================================================================

-- =============================================================================
-- Trigger Function 1: Audit report lifecycle changes
-- =============================================================================
CREATE OR REPLACE FUNCTION pack030_nz_reporting.fn_audit_report_changes()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    v_event_type VARCHAR(100);
    v_details JSONB;
BEGIN
    IF TG_OP = 'INSERT' THEN
        v_event_type := 'REPORT_CREATED';
        v_details := jsonb_build_object(
            'framework', NEW.framework,
            'reporting_year', NEW.reporting_year,
            'status', NEW.status,
            'report_type', NEW.report_type
        );
        INSERT INTO pack030_nz_reporting.gl_nz_audit_trail (
            tenant_id, organization_id, report_id, event_type, event_category,
            actor_id, actor_type, details, created_at
        ) VALUES (
            NEW.tenant_id, NEW.organization_id, NEW.report_id, v_event_type, 'REPORT',
            NEW.created_by, 'USER', v_details, NOW()
        );
        RETURN NEW;

    ELSIF TG_OP = 'UPDATE' THEN
        -- Determine event type based on status change
        IF OLD.status != NEW.status THEN
            v_event_type := CASE NEW.status
                WHEN 'REVIEW' THEN 'REPORT_SUBMITTED_FOR_REVIEW'
                WHEN 'APPROVED' THEN 'REPORT_APPROVED'
                WHEN 'REJECTED' THEN 'REPORT_REJECTED'
                WHEN 'PUBLISHED' THEN 'REPORT_PUBLISHED'
                WHEN 'ARCHIVED' THEN 'REPORT_ARCHIVED'
                ELSE 'REPORT_UPDATED'
            END;
        ELSE
            v_event_type := 'REPORT_UPDATED';
        END IF;

        v_details := jsonb_build_object(
            'previous_status', OLD.status,
            'new_status', NEW.status,
            'framework', NEW.framework,
            'reporting_year', NEW.reporting_year,
            'version_number', NEW.version_number
        );

        INSERT INTO pack030_nz_reporting.gl_nz_audit_trail (
            tenant_id, organization_id, report_id, event_type, event_category,
            actor_id, actor_type, previous_state, new_state, details, created_at
        ) VALUES (
            NEW.tenant_id, NEW.organization_id, NEW.report_id, v_event_type, 'REPORT',
            COALESCE(NEW.status_changed_by, NEW.created_by), 'USER',
            jsonb_build_object('status', OLD.status, 'completeness', OLD.data_completeness_pct),
            jsonb_build_object('status', NEW.status, 'completeness', NEW.data_completeness_pct),
            v_details, NOW()
        );
        RETURN NEW;

    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO pack030_nz_reporting.gl_nz_audit_trail (
            tenant_id, organization_id, report_id, event_type, event_category,
            actor_id, actor_type, details, created_at
        ) VALUES (
            OLD.tenant_id, OLD.organization_id, OLD.report_id, 'REPORT_DELETED', 'REPORT',
            OLD.created_by, 'SYSTEM',
            jsonb_build_object('framework', OLD.framework, 'status', OLD.status),
            NOW()
        );
        RETURN OLD;
    END IF;

    RETURN NULL;
END;
$$;

-- =============================================================================
-- Trigger Function 2: Audit section changes
-- =============================================================================
CREATE OR REPLACE FUNCTION pack030_nz_reporting.fn_audit_section_changes()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO pack030_nz_reporting.gl_nz_audit_trail (
            tenant_id, organization_id, report_id, event_type, event_category,
            resource_type, resource_id, actor_id, actor_type, details, created_at
        ) VALUES (
            NEW.tenant_id,
            (SELECT organization_id FROM pack030_nz_reporting.gl_nz_reports WHERE report_id = NEW.report_id),
            NEW.report_id, 'SECTION_CREATED', 'SECTION',
            'section', NEW.section_id,
            COALESCE(NEW.reviewed_by, '00000000-0000-0000-0000-000000000000'::UUID), 'SYSTEM',
            jsonb_build_object('section_type', NEW.section_type, 'language', NEW.language, 'word_count', NEW.word_count),
            NOW()
        );
        RETURN NEW;

    ELSIF TG_OP = 'UPDATE' THEN
        IF OLD.review_status != NEW.review_status THEN
            INSERT INTO pack030_nz_reporting.gl_nz_audit_trail (
                tenant_id, organization_id, report_id, event_type, event_category,
                resource_type, resource_id, actor_id, actor_type, details, created_at
            ) VALUES (
                NEW.tenant_id,
                (SELECT organization_id FROM pack030_nz_reporting.gl_nz_reports WHERE report_id = NEW.report_id),
                NEW.report_id, 'SECTION_UPDATED', 'SECTION',
                'section', NEW.section_id,
                COALESCE(NEW.reviewed_by, '00000000-0000-0000-0000-000000000000'::UUID), 'USER',
                jsonb_build_object('previous_review_status', OLD.review_status, 'new_review_status', NEW.review_status, 'section_type', NEW.section_type),
                NOW()
            );
        END IF;
        RETURN NEW;

    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO pack030_nz_reporting.gl_nz_audit_trail (
            tenant_id, organization_id, report_id, event_type, event_category,
            resource_type, resource_id, actor_id, actor_type, details, created_at
        ) VALUES (
            OLD.tenant_id,
            (SELECT organization_id FROM pack030_nz_reporting.gl_nz_reports WHERE report_id = OLD.report_id),
            OLD.report_id, 'SECTION_DELETED', 'SECTION',
            'section', OLD.section_id,
            '00000000-0000-0000-0000-000000000000'::UUID, 'SYSTEM',
            jsonb_build_object('section_type', OLD.section_type),
            NOW()
        );
        RETURN OLD;
    END IF;

    RETURN NULL;
END;
$$;

-- =============================================================================
-- Trigger Function 3: Audit metric changes
-- =============================================================================
CREATE OR REPLACE FUNCTION pack030_nz_reporting.fn_audit_metric_changes()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO pack030_nz_reporting.gl_nz_audit_trail (
            tenant_id, organization_id, report_id, event_type, event_category,
            resource_type, resource_id, actor_id, actor_type, details, created_at
        ) VALUES (
            NEW.tenant_id,
            (SELECT organization_id FROM pack030_nz_reporting.gl_nz_reports WHERE report_id = NEW.report_id),
            NEW.report_id, 'METRIC_ADDED', 'METRIC',
            'metric', NEW.metric_id,
            '00000000-0000-0000-0000-000000000000'::UUID, 'SYSTEM',
            jsonb_build_object('metric_name', NEW.metric_name, 'metric_value', NEW.metric_value, 'unit', NEW.unit, 'scope', NEW.scope, 'source_system', NEW.source_system),
            NOW()
        );
        RETURN NEW;

    ELSIF TG_OP = 'UPDATE' THEN
        IF OLD.metric_value != NEW.metric_value OR OLD.verification_status != NEW.verification_status THEN
            INSERT INTO pack030_nz_reporting.gl_nz_audit_trail (
                tenant_id, organization_id, report_id, event_type, event_category,
                resource_type, resource_id, actor_id, actor_type,
                previous_state, new_state, details, created_at
            ) VALUES (
                NEW.tenant_id,
                (SELECT organization_id FROM pack030_nz_reporting.gl_nz_reports WHERE report_id = NEW.report_id),
                NEW.report_id,
                CASE WHEN OLD.verification_status != NEW.verification_status THEN 'METRIC_VERIFIED' ELSE 'METRIC_UPDATED' END,
                'METRIC',
                'metric', NEW.metric_id,
                '00000000-0000-0000-0000-000000000000'::UUID, 'SYSTEM',
                jsonb_build_object('metric_value', OLD.metric_value, 'verification_status', OLD.verification_status),
                jsonb_build_object('metric_value', NEW.metric_value, 'verification_status', NEW.verification_status),
                jsonb_build_object('metric_name', NEW.metric_name),
                NOW()
            );
        END IF;
        RETURN NEW;

    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO pack030_nz_reporting.gl_nz_audit_trail (
            tenant_id, organization_id, report_id, event_type, event_category,
            resource_type, resource_id, actor_id, actor_type, details, created_at
        ) VALUES (
            OLD.tenant_id,
            (SELECT organization_id FROM pack030_nz_reporting.gl_nz_reports WHERE report_id = OLD.report_id),
            OLD.report_id, 'METRIC_DELETED', 'METRIC',
            'metric', OLD.metric_id,
            '00000000-0000-0000-0000-000000000000'::UUID, 'SYSTEM',
            jsonb_build_object('metric_name', OLD.metric_name, 'metric_value', OLD.metric_value),
            NOW()
        );
        RETURN OLD;
    END IF;

    RETURN NULL;
END;
$$;

-- =============================================================================
-- Trigger Function 4: Audit narrative changes
-- =============================================================================
CREATE OR REPLACE FUNCTION pack030_nz_reporting.fn_audit_narrative_changes()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO pack030_nz_reporting.gl_nz_audit_trail (
            tenant_id, organization_id, event_type, event_category,
            resource_type, resource_id, actor_id, actor_type, details, created_at
        ) VALUES (
            NEW.tenant_id, NEW.organization_id, 'NARRATIVE_CREATED', 'NARRATIVE',
            'narrative', NEW.narrative_id,
            COALESCE(NEW.reviewed_by, '00000000-0000-0000-0000-000000000000'::UUID), 'SYSTEM',
            jsonb_build_object('framework', NEW.framework, 'section_type', NEW.section_type, 'language', NEW.language, 'generation_method', NEW.generation_method),
            NOW()
        );
        RETURN NEW;

    ELSIF TG_OP = 'UPDATE' THEN
        IF OLD.human_reviewed = FALSE AND NEW.human_reviewed = TRUE THEN
            INSERT INTO pack030_nz_reporting.gl_nz_audit_trail (
                tenant_id, organization_id, event_type, event_category,
                resource_type, resource_id, actor_id, actor_type, details, created_at
            ) VALUES (
                NEW.tenant_id, NEW.organization_id, 'NARRATIVE_REVIEWED', 'NARRATIVE',
                'narrative', NEW.narrative_id,
                COALESCE(NEW.reviewed_by, '00000000-0000-0000-0000-000000000000'::UUID), 'USER',
                jsonb_build_object('framework', NEW.framework, 'section_type', NEW.section_type, 'consistency_score', NEW.consistency_score),
                NOW()
            );
        END IF;
        RETURN NEW;
    END IF;

    RETURN COALESCE(NEW, OLD);
END;
$$;

-- =============================================================================
-- Trigger Function 5: Audit evidence changes
-- =============================================================================
CREATE OR REPLACE FUNCTION pack030_nz_reporting.fn_audit_evidence_changes()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO pack030_nz_reporting.gl_nz_audit_trail (
            tenant_id, organization_id, report_id, event_type, event_category,
            resource_type, resource_id, actor_id, actor_type, details, created_at
        ) VALUES (
            NEW.tenant_id, NEW.organization_id, NEW.report_id, 'EVIDENCE_UPLOADED', 'EVIDENCE',
            'evidence', NEW.evidence_id,
            '00000000-0000-0000-0000-000000000000'::UUID, 'SYSTEM',
            jsonb_build_object('evidence_type', NEW.evidence_type, 'evidence_tier', NEW.evidence_tier, 'checksum', NEW.checksum),
            NOW()
        );
        RETURN NEW;

    ELSIF TG_OP = 'UPDATE' THEN
        IF OLD.auditor_reviewed = FALSE AND NEW.auditor_reviewed = TRUE THEN
            INSERT INTO pack030_nz_reporting.gl_nz_audit_trail (
                tenant_id, organization_id, report_id, event_type, event_category,
                resource_type, resource_id, actor_id, actor_type, details, created_at
            ) VALUES (
                NEW.tenant_id, NEW.organization_id, NEW.report_id,
                CASE NEW.review_outcome WHEN 'REJECTED' THEN 'EVIDENCE_REJECTED' ELSE 'EVIDENCE_REVIEWED' END,
                'EVIDENCE',
                'evidence', NEW.evidence_id,
                '00000000-0000-0000-0000-000000000000'::UUID, 'USER',
                jsonb_build_object('evidence_type', NEW.evidence_type, 'review_outcome', NEW.review_outcome, 'auditor_name', NEW.auditor_name),
                NOW()
            );
        END IF;
        RETURN NEW;
    END IF;

    RETURN COALESCE(NEW, OLD);
END;
$$;

-- =============================================================================
-- Trigger Function 6: Audit validation result changes
-- =============================================================================
CREATE OR REPLACE FUNCTION pack030_nz_reporting.fn_audit_validation_changes()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'UPDATE' AND OLD.resolved = FALSE AND NEW.resolved = TRUE THEN
        INSERT INTO pack030_nz_reporting.gl_nz_audit_trail (
            tenant_id, organization_id, report_id, event_type, event_category,
            resource_type, resource_id, actor_id, actor_type, details, created_at
        ) VALUES (
            NEW.tenant_id, NEW.organization_id, NEW.report_id, 'VALIDATION_RESOLVED', 'VALIDATION',
            'validation', NEW.validation_id,
            COALESCE(NEW.resolved_by, '00000000-0000-0000-0000-000000000000'::UUID), 'USER',
            jsonb_build_object('severity', NEW.severity, 'validation_category', NEW.validation_category, 'resolution_method', NEW.resolution_method),
            NOW()
        );
    END IF;
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Create Triggers
-- =============================================================================

-- Report audit triggers
CREATE TRIGGER trg_p030_audit_report_insert_update
    AFTER INSERT OR UPDATE ON pack030_nz_reporting.gl_nz_reports
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_audit_report_changes();

CREATE TRIGGER trg_p030_audit_report_delete
    BEFORE DELETE ON pack030_nz_reporting.gl_nz_reports
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_audit_report_changes();

-- Section audit triggers
CREATE TRIGGER trg_p030_audit_section_insert_update
    AFTER INSERT OR UPDATE ON pack030_nz_reporting.gl_nz_report_sections
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_audit_section_changes();

CREATE TRIGGER trg_p030_audit_section_delete
    BEFORE DELETE ON pack030_nz_reporting.gl_nz_report_sections
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_audit_section_changes();

-- Metric audit triggers
CREATE TRIGGER trg_p030_audit_metric_insert_update
    AFTER INSERT OR UPDATE ON pack030_nz_reporting.gl_nz_report_metrics
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_audit_metric_changes();

CREATE TRIGGER trg_p030_audit_metric_delete
    BEFORE DELETE ON pack030_nz_reporting.gl_nz_report_metrics
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_audit_metric_changes();

-- Narrative audit triggers
CREATE TRIGGER trg_p030_audit_narrative_insert_update
    AFTER INSERT OR UPDATE ON pack030_nz_reporting.gl_nz_narratives
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_audit_narrative_changes();

-- Evidence audit triggers
CREATE TRIGGER trg_p030_audit_evidence_insert_update
    AFTER INSERT OR UPDATE ON pack030_nz_reporting.gl_nz_assurance_evidence
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_audit_evidence_changes();

-- Validation result audit triggers
CREATE TRIGGER trg_p030_audit_validation_update
    AFTER UPDATE ON pack030_nz_reporting.gl_nz_validation_results
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_audit_validation_changes();

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_audit_report_changes() TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_audit_section_changes() TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_audit_metric_changes() TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_audit_narrative_changes() TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_audit_evidence_changes() TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_audit_validation_changes() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON FUNCTION pack030_nz_reporting.fn_audit_report_changes IS
    'Trigger function that logs report lifecycle events (create, status change, approve, publish, delete) to the audit trail.';
COMMENT ON FUNCTION pack030_nz_reporting.fn_audit_section_changes IS
    'Trigger function that logs section changes (create, review status change, delete) to the audit trail.';
COMMENT ON FUNCTION pack030_nz_reporting.fn_audit_metric_changes IS
    'Trigger function that logs metric changes (add, value update, verification status change, delete) to the audit trail.';
COMMENT ON FUNCTION pack030_nz_reporting.fn_audit_narrative_changes IS
    'Trigger function that logs narrative events (create, human review) to the audit trail.';
COMMENT ON FUNCTION pack030_nz_reporting.fn_audit_evidence_changes IS
    'Trigger function that logs evidence events (upload, auditor review/rejection) to the audit trail.';
COMMENT ON FUNCTION pack030_nz_reporting.fn_audit_validation_changes IS
    'Trigger function that logs validation resolution events to the audit trail.';
