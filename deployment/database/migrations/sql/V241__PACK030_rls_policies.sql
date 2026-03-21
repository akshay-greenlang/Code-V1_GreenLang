-- =============================================================================
-- V221: PACK-030 Net Zero Reporting Pack - Row-Level Security Policies
-- =============================================================================
-- Pack:         PACK-030 (Net Zero Reporting Pack)
-- Migration:    011 of 015
-- Date:         March 2026
--
-- Comprehensive row-level security policies for multi-tenant data isolation
-- across all PACK-030 tables. Supplements the basic tenant isolation and
-- service bypass policies created in V211-V218 with granular role-based
-- access policies.
--
-- Policies (30):
--   - 15 tables x 2 base policies (tenant_isolation + service_bypass) = 30
--     (already created in V211-V218)
--   - This migration adds role-based access policies for fine-grained control
--
-- Previous: V220__PACK030_views.sql
-- =============================================================================

-- =============================================================================
-- Role-Based Access Policies for Reports
-- =============================================================================

-- Readers can see published reports for their organization
CREATE POLICY p030_rpt_reader_access
    ON pack030_nz_reporting.gl_nz_reports
    FOR SELECT
    TO greenlang_reader
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
        AND status = 'PUBLISHED'
    );

-- Editors can view and modify draft/in-progress reports
CREATE POLICY p030_rpt_editor_access
    ON pack030_nz_reporting.gl_nz_reports
    TO greenlang_editor
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
        AND status IN ('DRAFT', 'IN_PROGRESS', 'REVIEW', 'REJECTED')
    )
    WITH CHECK (
        tenant_id = current_setting('app.current_tenant')::UUID
        AND status IN ('DRAFT', 'IN_PROGRESS')
    );

-- Approvers can view all and approve/reject
CREATE POLICY p030_rpt_approver_access
    ON pack030_nz_reporting.gl_nz_reports
    TO greenlang_approver
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
    )
    WITH CHECK (
        tenant_id = current_setting('app.current_tenant')::UUID
    );

-- Auditors have read-only access to all reports including archived
CREATE POLICY p030_rpt_auditor_access
    ON pack030_nz_reporting.gl_nz_reports
    FOR SELECT
    TO greenlang_auditor
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
    );

-- =============================================================================
-- Role-Based Access Policies for Report Sections
-- =============================================================================

CREATE POLICY p030_sec_reader_access
    ON pack030_nz_reporting.gl_nz_report_sections
    FOR SELECT
    TO greenlang_reader
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
        AND report_id IN (
            SELECT report_id FROM pack030_nz_reporting.gl_nz_reports
            WHERE status = 'PUBLISHED'
              AND tenant_id = current_setting('app.current_tenant')::UUID
        )
    );

CREATE POLICY p030_sec_editor_access
    ON pack030_nz_reporting.gl_nz_report_sections
    TO greenlang_editor
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
    )
    WITH CHECK (
        tenant_id = current_setting('app.current_tenant')::UUID
    );

CREATE POLICY p030_sec_auditor_access
    ON pack030_nz_reporting.gl_nz_report_sections
    FOR SELECT
    TO greenlang_auditor
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
    );

-- =============================================================================
-- Role-Based Access Policies for Metrics
-- =============================================================================

CREATE POLICY p030_met_reader_access
    ON pack030_nz_reporting.gl_nz_report_metrics
    FOR SELECT
    TO greenlang_reader
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
        AND report_id IN (
            SELECT report_id FROM pack030_nz_reporting.gl_nz_reports
            WHERE status = 'PUBLISHED'
              AND tenant_id = current_setting('app.current_tenant')::UUID
        )
    );

CREATE POLICY p030_met_editor_access
    ON pack030_nz_reporting.gl_nz_report_metrics
    TO greenlang_editor
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
    )
    WITH CHECK (
        tenant_id = current_setting('app.current_tenant')::UUID
    );

CREATE POLICY p030_met_auditor_access
    ON pack030_nz_reporting.gl_nz_report_metrics
    FOR SELECT
    TO greenlang_auditor
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
    );

-- =============================================================================
-- Role-Based Access Policies for Assurance Evidence
-- =============================================================================

CREATE POLICY p030_ae_auditor_full_access
    ON pack030_nz_reporting.gl_nz_assurance_evidence
    TO greenlang_auditor
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
    )
    WITH CHECK (
        tenant_id = current_setting('app.current_tenant')::UUID
    );

CREATE POLICY p030_ae_editor_access
    ON pack030_nz_reporting.gl_nz_assurance_evidence
    TO greenlang_editor
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
    )
    WITH CHECK (
        tenant_id = current_setting('app.current_tenant')::UUID
    );

-- =============================================================================
-- Role-Based Access Policies for Audit Trail (Read-Only)
-- =============================================================================

CREATE POLICY p030_at_auditor_access
    ON pack030_nz_reporting.gl_nz_audit_trail
    FOR SELECT
    TO greenlang_auditor
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
    );

CREATE POLICY p030_at_approver_access
    ON pack030_nz_reporting.gl_nz_audit_trail
    FOR SELECT
    TO greenlang_approver
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
    );

-- =============================================================================
-- Role-Based Access Policies for XBRL Tags
-- =============================================================================

CREATE POLICY p030_xb_editor_access
    ON pack030_nz_reporting.gl_nz_xbrl_tags
    TO greenlang_editor
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
    )
    WITH CHECK (
        tenant_id = current_setting('app.current_tenant')::UUID
    );

CREATE POLICY p030_xb_auditor_access
    ON pack030_nz_reporting.gl_nz_xbrl_tags
    FOR SELECT
    TO greenlang_auditor
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
    );

-- =============================================================================
-- Role-Based Access Policies for Validation Results
-- =============================================================================

CREATE POLICY p030_vr_editor_access
    ON pack030_nz_reporting.gl_nz_validation_results
    TO greenlang_editor
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
    )
    WITH CHECK (
        tenant_id = current_setting('app.current_tenant')::UUID
    );

CREATE POLICY p030_vr_auditor_access
    ON pack030_nz_reporting.gl_nz_validation_results
    FOR SELECT
    TO greenlang_auditor
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
    );

-- =============================================================================
-- Role-Based Access Policies for Dashboard Views
-- =============================================================================

CREATE POLICY p030_dv_reader_access
    ON pack030_nz_reporting.gl_nz_dashboard_views
    FOR SELECT
    TO greenlang_reader
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
        AND (is_public = TRUE OR created_by = current_setting('app.current_user_id')::UUID)
    );

CREATE POLICY p030_dv_editor_access
    ON pack030_nz_reporting.gl_nz_dashboard_views
    TO greenlang_editor
    USING (
        tenant_id = current_setting('app.current_tenant')::UUID
    )
    WITH CHECK (
        tenant_id = current_setting('app.current_tenant')::UUID
    );

-- =============================================================================
-- Comments
-- =============================================================================
COMMENT ON POLICY p030_rpt_reader_access ON pack030_nz_reporting.gl_nz_reports IS
    'Readers can only view published reports within their tenant.';
COMMENT ON POLICY p030_rpt_editor_access ON pack030_nz_reporting.gl_nz_reports IS
    'Editors can view draft/review/rejected reports and modify draft/in-progress reports within their tenant.';
COMMENT ON POLICY p030_rpt_approver_access ON pack030_nz_reporting.gl_nz_reports IS
    'Approvers have full access to all reports within their tenant for approval workflow.';
COMMENT ON POLICY p030_rpt_auditor_access ON pack030_nz_reporting.gl_nz_reports IS
    'Auditors have read-only access to all reports including archived for assurance purposes.';
COMMENT ON POLICY p030_at_auditor_access ON pack030_nz_reporting.gl_nz_audit_trail IS
    'Auditors have read-only access to the complete audit trail within their tenant.';
COMMENT ON POLICY p030_dv_reader_access ON pack030_nz_reporting.gl_nz_dashboard_views IS
    'Readers can view public dashboards and their own private dashboards.';
