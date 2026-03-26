-- =============================================================================
-- V357: PACK-044 GHG Inventory Management - Data Collection Tables
-- =============================================================================
-- Pack:         PACK-044 (GHG Inventory Management)
-- Migration:    002 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Data collection campaign management tables. Campaigns define a time-bound
-- collection exercise for an inventory period. Data requests are sent to
-- individual data owners (facility managers, site contacts). Submissions
-- track the data returned with evidence attachments. Reminders automate
-- follow-up for overdue submissions.
--
-- Tables (4):
--   1. ghg_inventory.gl_inv_collection_campaigns
--   2. ghg_inventory.gl_inv_data_requests
--   3. ghg_inventory.gl_inv_data_submissions
--   4. ghg_inventory.gl_inv_collection_reminders
--
-- Previous: V356__pack044_core_schema.sql
-- =============================================================================

SET search_path TO ghg_inventory, public;

-- =============================================================================
-- Table 1: ghg_inventory.gl_inv_collection_campaigns
-- =============================================================================
-- A data collection campaign for a given inventory period and time window.
-- Campaigns may be QUARTERLY, MONTHLY, SEASONAL, or ANNUAL. Each campaign
-- generates data requests to responsible parties and tracks overall progress.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_collection_campaigns (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    period_id                   UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_inventory_periods(id) ON DELETE CASCADE,
    campaign_name               VARCHAR(300)    NOT NULL,
    campaign_type               VARCHAR(30)     NOT NULL DEFAULT 'QUARTERLY',
    collection_start_date       DATE            NOT NULL,
    collection_end_date         DATE            NOT NULL,
    data_window_start           DATE            NOT NULL,
    data_window_end             DATE            NOT NULL,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    total_requests              INTEGER         NOT NULL DEFAULT 0,
    completed_requests          INTEGER         NOT NULL DEFAULT 0,
    completion_pct              NUMERIC(5,2)    DEFAULT 0.00,
    auto_reminders_enabled      BOOLEAN         NOT NULL DEFAULT true,
    reminder_before_days        INTEGER         DEFAULT 7,
    escalation_after_days       INTEGER         DEFAULT 14,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p044_cc_type CHECK (
        campaign_type IN ('MONTHLY', 'QUARTERLY', 'SEASONAL', 'SEMI_ANNUAL', 'ANNUAL', 'AD_HOC')
    ),
    CONSTRAINT chk_p044_cc_collection_dates CHECK (
        collection_start_date <= collection_end_date
    ),
    CONSTRAINT chk_p044_cc_data_window CHECK (
        data_window_start <= data_window_end
    ),
    CONSTRAINT chk_p044_cc_status CHECK (
        status IN ('DRAFT', 'OPEN', 'IN_PROGRESS', 'CLOSED', 'CANCELLED')
    ),
    CONSTRAINT chk_p044_cc_requests CHECK (
        total_requests >= 0 AND completed_requests >= 0 AND completed_requests <= total_requests
    ),
    CONSTRAINT chk_p044_cc_completion CHECK (
        completion_pct IS NULL OR (completion_pct >= 0 AND completion_pct <= 100)
    ),
    CONSTRAINT chk_p044_cc_reminder_days CHECK (
        reminder_before_days IS NULL OR reminder_before_days >= 0
    ),
    CONSTRAINT chk_p044_cc_escalation_days CHECK (
        escalation_after_days IS NULL OR escalation_after_days >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_cc_tenant          ON ghg_inventory.gl_inv_collection_campaigns(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_cc_period          ON ghg_inventory.gl_inv_collection_campaigns(period_id);
CREATE INDEX IF NOT EXISTS idx_p044_cc_type            ON ghg_inventory.gl_inv_collection_campaigns(campaign_type);
CREATE INDEX IF NOT EXISTS idx_p044_cc_status          ON ghg_inventory.gl_inv_collection_campaigns(status);
CREATE INDEX IF NOT EXISTS idx_p044_cc_dates           ON ghg_inventory.gl_inv_collection_campaigns(collection_start_date, collection_end_date);
CREATE INDEX IF NOT EXISTS idx_p044_cc_created         ON ghg_inventory.gl_inv_collection_campaigns(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_p044_cc_metadata        ON ghg_inventory.gl_inv_collection_campaigns USING GIN(metadata);

-- Composite: period + open campaigns
CREATE INDEX IF NOT EXISTS idx_p044_cc_period_open     ON ghg_inventory.gl_inv_collection_campaigns(period_id, collection_end_date)
    WHERE status IN ('OPEN', 'IN_PROGRESS');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_cc_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_collection_campaigns
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_inventory.gl_inv_data_requests
-- =============================================================================
-- Individual data requests sent to data owners as part of a collection
-- campaign. Each request targets a specific facility and source category
-- (or set of categories) and tracks the submission status.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_data_requests (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    campaign_id                 UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_collection_campaigns(id) ON DELETE CASCADE,
    facility_id                 UUID            NOT NULL,
    source_category             VARCHAR(60),
    assigned_to_user_id         UUID,
    assigned_to_email           VARCHAR(320),
    assigned_to_name            VARCHAR(255),
    status                      VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    due_date                    DATE            NOT NULL,
    submitted_at                TIMESTAMPTZ,
    reminder_count              INTEGER         NOT NULL DEFAULT 0,
    last_reminder_at            TIMESTAMPTZ,
    escalated                   BOOLEAN         NOT NULL DEFAULT false,
    escalated_at                TIMESTAMPTZ,
    escalated_to                VARCHAR(255),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_dr_status CHECK (
        status IN (
            'PENDING', 'SENT', 'VIEWED', 'IN_PROGRESS', 'SUBMITTED',
            'RETURNED', 'ACCEPTED', 'OVERDUE', 'CANCELLED'
        )
    ),
    CONSTRAINT chk_p044_dr_reminder_count CHECK (
        reminder_count >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_dr_tenant          ON ghg_inventory.gl_inv_data_requests(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_dr_campaign        ON ghg_inventory.gl_inv_data_requests(campaign_id);
CREATE INDEX IF NOT EXISTS idx_p044_dr_facility        ON ghg_inventory.gl_inv_data_requests(facility_id);
CREATE INDEX IF NOT EXISTS idx_p044_dr_assigned        ON ghg_inventory.gl_inv_data_requests(assigned_to_user_id);
CREATE INDEX IF NOT EXISTS idx_p044_dr_status          ON ghg_inventory.gl_inv_data_requests(status);
CREATE INDEX IF NOT EXISTS idx_p044_dr_due             ON ghg_inventory.gl_inv_data_requests(due_date);
CREATE INDEX IF NOT EXISTS idx_p044_dr_created         ON ghg_inventory.gl_inv_data_requests(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_p044_dr_metadata        ON ghg_inventory.gl_inv_data_requests USING GIN(metadata);

-- Composite: campaign + pending/overdue requests
CREATE INDEX IF NOT EXISTS idx_p044_dr_campaign_open   ON ghg_inventory.gl_inv_data_requests(campaign_id, due_date)
    WHERE status IN ('PENDING', 'SENT', 'VIEWED', 'IN_PROGRESS', 'OVERDUE');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_dr_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_data_requests
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_inventory.gl_inv_data_submissions
-- =============================================================================
-- Data submissions from data owners in response to data requests. Contains
-- the actual activity data values, units, evidence references, and data
-- quality indicators. A single request may have multiple submissions if
-- data is returned and resubmitted.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_data_submissions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    request_id                  UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_data_requests(id) ON DELETE CASCADE,
    submission_version          INTEGER         NOT NULL DEFAULT 1,
    submitted_by_user_id        UUID,
    submitted_by_name           VARCHAR(255),
    submitted_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    activity_data               JSONB           NOT NULL DEFAULT '{}',
    activity_value              NUMERIC(18,6),
    activity_unit               VARCHAR(50),
    source_description          TEXT,
    data_source_type            VARCHAR(30)     DEFAULT 'METERED',
    evidence_file_ids           UUID[],
    evidence_urls               TEXT[],
    estimation_method           VARCHAR(100),
    estimation_justification    TEXT,
    data_quality_indicator      VARCHAR(30),
    is_estimated                BOOLEAN         NOT NULL DEFAULT false,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'SUBMITTED',
    reviewer_notes              TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_ds_version CHECK (
        submission_version >= 1
    ),
    CONSTRAINT chk_p044_ds_source_type CHECK (
        data_source_type IS NULL OR data_source_type IN (
            'METERED', 'INVOICED', 'ESTIMATED', 'CALCULATED', 'SUPPLIER_PROVIDED',
            'ERP_IMPORT', 'TELEMATICS', 'BMS', 'MANUAL_ENTRY'
        )
    ),
    CONSTRAINT chk_p044_ds_quality CHECK (
        data_quality_indicator IS NULL OR data_quality_indicator IN (
            'HIGH', 'MEDIUM', 'LOW', 'VERY_LOW', 'NOT_ASSESSED'
        )
    ),
    CONSTRAINT chk_p044_ds_status CHECK (
        status IN ('SUBMITTED', 'UNDER_REVIEW', 'ACCEPTED', 'RETURNED', 'REJECTED', 'SUPERSEDED')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_ds_tenant          ON ghg_inventory.gl_inv_data_submissions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_ds_request         ON ghg_inventory.gl_inv_data_submissions(request_id);
CREATE INDEX IF NOT EXISTS idx_p044_ds_submitted_by    ON ghg_inventory.gl_inv_data_submissions(submitted_by_user_id);
CREATE INDEX IF NOT EXISTS idx_p044_ds_status          ON ghg_inventory.gl_inv_data_submissions(status);
CREATE INDEX IF NOT EXISTS idx_p044_ds_source_type     ON ghg_inventory.gl_inv_data_submissions(data_source_type);
CREATE INDEX IF NOT EXISTS idx_p044_ds_quality         ON ghg_inventory.gl_inv_data_submissions(data_quality_indicator);
CREATE INDEX IF NOT EXISTS idx_p044_ds_created         ON ghg_inventory.gl_inv_data_submissions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_p044_ds_activity        ON ghg_inventory.gl_inv_data_submissions USING GIN(activity_data);
CREATE INDEX IF NOT EXISTS idx_p044_ds_metadata        ON ghg_inventory.gl_inv_data_submissions USING GIN(metadata);

-- Composite: request + latest accepted submission
CREATE INDEX IF NOT EXISTS idx_p044_ds_request_accept  ON ghg_inventory.gl_inv_data_submissions(request_id, submission_version DESC)
    WHERE status = 'ACCEPTED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_ds_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_data_submissions
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_inventory.gl_inv_collection_reminders
-- =============================================================================
-- Automated and manual reminders sent for overdue or upcoming data requests.
-- Tracks reminder type (email, in-app, escalation), delivery status, and
-- response within the collection workflow.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_collection_reminders (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    request_id                  UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_data_requests(id) ON DELETE CASCADE,
    reminder_type               VARCHAR(30)     NOT NULL DEFAULT 'EMAIL',
    reminder_reason             VARCHAR(30)     NOT NULL DEFAULT 'UPCOMING',
    sent_at                     TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    sent_to_email               VARCHAR(320),
    sent_to_name                VARCHAR(255),
    delivery_status             VARCHAR(30)     NOT NULL DEFAULT 'SENT',
    is_escalation               BOOLEAN         NOT NULL DEFAULT false,
    escalated_to_email          VARCHAR(320),
    escalated_to_name           VARCHAR(255),
    message_template            VARCHAR(100),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_cr_type CHECK (
        reminder_type IN ('EMAIL', 'IN_APP', 'SMS', 'WEBHOOK', 'MANUAL')
    ),
    CONSTRAINT chk_p044_cr_reason CHECK (
        reminder_reason IN ('UPCOMING', 'DUE_TODAY', 'OVERDUE', 'ESCALATION', 'FOLLOW_UP')
    ),
    CONSTRAINT chk_p044_cr_delivery CHECK (
        delivery_status IN ('SENT', 'DELIVERED', 'BOUNCED', 'FAILED', 'PENDING')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_cr_tenant          ON ghg_inventory.gl_inv_collection_reminders(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_cr_request         ON ghg_inventory.gl_inv_collection_reminders(request_id);
CREATE INDEX IF NOT EXISTS idx_p044_cr_type            ON ghg_inventory.gl_inv_collection_reminders(reminder_type);
CREATE INDEX IF NOT EXISTS idx_p044_cr_reason          ON ghg_inventory.gl_inv_collection_reminders(reminder_reason);
CREATE INDEX IF NOT EXISTS idx_p044_cr_sent            ON ghg_inventory.gl_inv_collection_reminders(sent_at DESC);
CREATE INDEX IF NOT EXISTS idx_p044_cr_delivery        ON ghg_inventory.gl_inv_collection_reminders(delivery_status);
CREATE INDEX IF NOT EXISTS idx_p044_cr_escalation      ON ghg_inventory.gl_inv_collection_reminders(is_escalation) WHERE is_escalation = true;

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_inventory.gl_inv_collection_campaigns ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_data_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_data_submissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_collection_reminders ENABLE ROW LEVEL SECURITY;

CREATE POLICY p044_cc_tenant_isolation
    ON ghg_inventory.gl_inv_collection_campaigns
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_cc_service_bypass
    ON ghg_inventory.gl_inv_collection_campaigns
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_dr_tenant_isolation
    ON ghg_inventory.gl_inv_data_requests
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_dr_service_bypass
    ON ghg_inventory.gl_inv_data_requests
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_ds_tenant_isolation
    ON ghg_inventory.gl_inv_data_submissions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_ds_service_bypass
    ON ghg_inventory.gl_inv_data_submissions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_cr_tenant_isolation
    ON ghg_inventory.gl_inv_collection_reminders
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_cr_service_bypass
    ON ghg_inventory.gl_inv_collection_reminders
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_collection_campaigns TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_data_requests TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_data_submissions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_collection_reminders TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_inventory.gl_inv_collection_campaigns IS
    'Data collection campaigns that define time-bound data gathering exercises for inventory periods.';
COMMENT ON TABLE ghg_inventory.gl_inv_data_requests IS
    'Individual data requests sent to facility data owners within a collection campaign.';
COMMENT ON TABLE ghg_inventory.gl_inv_data_submissions IS
    'Data submissions from data owners with activity data, evidence, and quality indicators.';
COMMENT ON TABLE ghg_inventory.gl_inv_collection_reminders IS
    'Automated and manual reminders for overdue or upcoming data collection requests.';

COMMENT ON COLUMN ghg_inventory.gl_inv_collection_campaigns.campaign_type IS 'Collection frequency: MONTHLY, QUARTERLY, SEASONAL, SEMI_ANNUAL, ANNUAL, AD_HOC.';
COMMENT ON COLUMN ghg_inventory.gl_inv_collection_campaigns.completion_pct IS 'Percentage of data requests completed (0-100).';
COMMENT ON COLUMN ghg_inventory.gl_inv_data_requests.escalated IS 'Whether the request has been escalated due to non-response past the deadline.';
COMMENT ON COLUMN ghg_inventory.gl_inv_data_submissions.data_source_type IS 'How the data was obtained: METERED, INVOICED, ESTIMATED, ERP_IMPORT, etc.';
COMMENT ON COLUMN ghg_inventory.gl_inv_data_submissions.is_estimated IS 'Whether this submission contains estimated (rather than measured) data.';
COMMENT ON COLUMN ghg_inventory.gl_inv_collection_reminders.is_escalation IS 'Whether this reminder is an escalation to a superior rather than a standard reminder.';
