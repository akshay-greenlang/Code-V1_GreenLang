-- =============================================================================
-- V354: PACK-043 Scope 3 Complete Pack - Assurance & Verification
-- =============================================================================
-- Pack:         PACK-043 (Scope 3 Complete Pack)
-- Migration:    009 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates assurance, verification, and audit trail tables. Supports evidence
-- package assembly for third-party verification, calculation provenance with
-- SHA-256 hash chains, methodology decision documentation, verifier query
-- management, audit finding tracking, and a TimescaleDB-backed append-only
-- audit trail for complete traceability of all PACK-043 data changes.
--
-- Tables (6):
--   1. ghg_accounting_scope3_complete.evidence_packages
--   2. ghg_accounting_scope3_complete.calculation_provenance
--   3. ghg_accounting_scope3_complete.methodology_decisions
--   4. ghg_accounting_scope3_complete.verifier_queries
--   5. ghg_accounting_scope3_complete.audit_findings
--   6. ghg_accounting_scope3_complete.scope3_complete_audit_trail (hypertable)
--
-- Also includes: TimescaleDB hypertable, indexes, RLS, comments.
-- Previous: V353__pack043_sector_specific.sql
-- =============================================================================

SET search_path TO ghg_accounting_scope3_complete, public;

-- =============================================================================
-- Table 1: ghg_accounting_scope3_complete.evidence_packages
-- =============================================================================
-- Evidence package assembled for third-party verification engagement. Each
-- package contains all supporting documentation for an inventory's Scope 3
-- disclosure, with readiness scoring and verifier assignment.

CREATE TABLE ghg_accounting_scope3_complete.evidence_packages (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL,
    -- Package details
    package_name                VARCHAR(500)    NOT NULL,
    package_date                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    package_version             INTEGER         NOT NULL DEFAULT 1,
    -- Assurance
    assurance_level             ghg_accounting_scope3_complete.assurance_level NOT NULL DEFAULT 'LIMITED',
    assurance_standard          VARCHAR(100)    DEFAULT 'ISAE_3410',
    -- Readiness
    readiness_score             DECIMAL(5,2)    NOT NULL DEFAULT 0,
    completeness_pct            DECIMAL(5,2)    DEFAULT 0,
    documentation_items         INTEGER         DEFAULT 0,
    documentation_complete      INTEGER         DEFAULT 0,
    -- Verifier
    verifier_name               VARCHAR(500),
    verifier_contact            VARCHAR(255),
    verifier_accreditation      VARCHAR(200),
    engagement_start            DATE,
    engagement_end              DATE,
    engagement_fee_usd          NUMERIC(14,2),
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    submitted_at                TIMESTAMPTZ,
    -- Documents
    documents                   JSONB           DEFAULT '[]',
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_ep_readiness CHECK (readiness_score >= 0 AND readiness_score <= 100),
    CONSTRAINT chk_p043_ep_completeness CHECK (
        completeness_pct IS NULL OR (completeness_pct >= 0 AND completeness_pct <= 100)
    ),
    CONSTRAINT chk_p043_ep_docs CHECK (documentation_items IS NULL OR documentation_items >= 0),
    CONSTRAINT chk_p043_ep_docs_complete CHECK (documentation_complete IS NULL OR documentation_complete >= 0),
    CONSTRAINT chk_p043_ep_standard CHECK (
        assurance_standard IS NULL OR assurance_standard IN (
            'ISAE_3000', 'ISAE_3410', 'AA1000AS', 'ISO_14064_3', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p043_ep_dates CHECK (
        engagement_start IS NULL OR engagement_end IS NULL OR engagement_start <= engagement_end
    ),
    CONSTRAINT chk_p043_ep_fee CHECK (engagement_fee_usd IS NULL OR engagement_fee_usd >= 0),
    CONSTRAINT chk_p043_ep_version CHECK (package_version >= 1),
    CONSTRAINT chk_p043_ep_status CHECK (
        status IN ('DRAFT', 'IN_PREPARATION', 'READY', 'SUBMITTED', 'IN_REVIEW', 'COMPLETED', 'ARCHIVED')
    ),
    CONSTRAINT uq_p043_ep_inventory_version UNIQUE (inventory_id, package_version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_ep_tenant             ON ghg_accounting_scope3_complete.evidence_packages(tenant_id);
CREATE INDEX idx_p043_ep_inventory          ON ghg_accounting_scope3_complete.evidence_packages(inventory_id);
CREATE INDEX idx_p043_ep_date               ON ghg_accounting_scope3_complete.evidence_packages(package_date DESC);
CREATE INDEX idx_p043_ep_level              ON ghg_accounting_scope3_complete.evidence_packages(assurance_level);
CREATE INDEX idx_p043_ep_readiness          ON ghg_accounting_scope3_complete.evidence_packages(readiness_score DESC);
CREATE INDEX idx_p043_ep_status             ON ghg_accounting_scope3_complete.evidence_packages(status);
CREATE INDEX idx_p043_ep_verifier           ON ghg_accounting_scope3_complete.evidence_packages(verifier_name);
CREATE INDEX idx_p043_ep_created            ON ghg_accounting_scope3_complete.evidence_packages(created_at DESC);
CREATE INDEX idx_p043_ep_documents          ON ghg_accounting_scope3_complete.evidence_packages USING GIN(documents);

-- Composite: inventory + latest package
CREATE INDEX idx_p043_ep_inv_latest         ON ghg_accounting_scope3_complete.evidence_packages(inventory_id, package_date DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_ep_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.evidence_packages
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_accounting_scope3_complete.calculation_provenance
-- =============================================================================
-- Calculation provenance chain for audit trail. Each row documents a
-- calculation step with input data, output value, and a SHA-256 hash.
-- Hash chains (parent_hash) enable verification of data integrity from
-- raw inputs through to final reported values.

CREATE TABLE ghg_accounting_scope3_complete.calculation_provenance (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    evidence_id                 UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.evidence_packages(id) ON DELETE CASCADE,
    -- Calculation step
    step_number                 INTEGER         NOT NULL,
    step_name                   VARCHAR(200)    NOT NULL,
    calculation_type            VARCHAR(50)     NOT NULL,
    -- Category context
    category                    ghg_accounting_scope3_complete.scope3_category_type,
    -- Input/output
    input_data                  JSONB           NOT NULL DEFAULT '{}',
    input_description           TEXT,
    formula                     TEXT,
    output_value                DECIMAL(15,6)   NOT NULL,
    output_unit                 VARCHAR(50)     DEFAULT 'tCO2e',
    -- Emission factor
    ef_used                     DECIMAL(12,6),
    ef_source                   VARCHAR(200),
    ef_year                     INTEGER,
    -- Provenance hash chain
    provenance_hash             VARCHAR(64)     NOT NULL,
    parent_hash                 VARCHAR(64),
    -- Data quality
    data_quality_rating         DECIMAL(3,1),
    primary_data_flag           BOOLEAN         DEFAULT false,
    -- Metadata
    calculation_timestamp       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    agent_name                  VARCHAR(100),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_cp_step CHECK (step_number >= 1),
    CONSTRAINT chk_p043_cp_type CHECK (
        calculation_type IN (
            'SPEND_BASED', 'ACTIVITY_BASED', 'SUPPLIER_SPECIFIC',
            'HYBRID', 'LCA', 'ALLOCATION', 'AGGREGATION',
            'CONVERSION', 'NORMALIZATION', 'ADJUSTMENT'
        )
    ),
    CONSTRAINT chk_p043_cp_ef CHECK (ef_used IS NULL OR ef_used >= 0),
    CONSTRAINT chk_p043_cp_ef_year CHECK (ef_year IS NULL OR (ef_year >= 1990 AND ef_year <= 2100)),
    CONSTRAINT chk_p043_cp_dqr CHECK (
        data_quality_rating IS NULL OR (data_quality_rating >= 1.0 AND data_quality_rating <= 5.0)
    ),
    CONSTRAINT uq_p043_cp_evidence_step UNIQUE (evidence_id, step_number)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_cp_tenant             ON ghg_accounting_scope3_complete.calculation_provenance(tenant_id);
CREATE INDEX idx_p043_cp_evidence           ON ghg_accounting_scope3_complete.calculation_provenance(evidence_id);
CREATE INDEX idx_p043_cp_step               ON ghg_accounting_scope3_complete.calculation_provenance(step_number);
CREATE INDEX idx_p043_cp_type               ON ghg_accounting_scope3_complete.calculation_provenance(calculation_type);
CREATE INDEX idx_p043_cp_category           ON ghg_accounting_scope3_complete.calculation_provenance(category);
CREATE INDEX idx_p043_cp_provenance         ON ghg_accounting_scope3_complete.calculation_provenance(provenance_hash);
CREATE INDEX idx_p043_cp_parent             ON ghg_accounting_scope3_complete.calculation_provenance(parent_hash);
CREATE INDEX idx_p043_cp_agent              ON ghg_accounting_scope3_complete.calculation_provenance(agent_name);
CREATE INDEX idx_p043_cp_timestamp          ON ghg_accounting_scope3_complete.calculation_provenance(calculation_timestamp DESC);
CREATE INDEX idx_p043_cp_created            ON ghg_accounting_scope3_complete.calculation_provenance(created_at DESC);
CREATE INDEX idx_p043_cp_input              ON ghg_accounting_scope3_complete.calculation_provenance USING GIN(input_data);

-- Composite: evidence + ordered steps
CREATE INDEX idx_p043_cp_ev_steps           ON ghg_accounting_scope3_complete.calculation_provenance(evidence_id, step_number);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_cp_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.calculation_provenance
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_accounting_scope3_complete.methodology_decisions
-- =============================================================================
-- Documents methodology decisions made during Scope 3 calculations. Each
-- record captures a decision point, the chosen approach, rationale,
-- alternatives considered, and the decision-maker. Required for assurance
-- readiness and regulatory defensibility.

CREATE TABLE ghg_accounting_scope3_complete.methodology_decisions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    evidence_id                 UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.evidence_packages(id) ON DELETE CASCADE,
    -- Decision context
    category                    ghg_accounting_scope3_complete.scope3_category_type,
    decision_area               VARCHAR(100)    NOT NULL,
    -- Decision
    decision                    TEXT            NOT NULL,
    rationale                   TEXT            NOT NULL,
    alternatives_considered     JSONB           DEFAULT '[]',
    -- Impact
    impact_on_total_tco2e       DECIMAL(15,3),
    impact_pct                  DECIMAL(8,2),
    sensitivity_flag            BOOLEAN         DEFAULT false,
    -- Decision-maker
    decision_date               DATE            NOT NULL DEFAULT CURRENT_DATE,
    decision_by                 VARCHAR(255)    NOT NULL,
    decision_role               VARCHAR(100),
    -- Approval
    approved                    BOOLEAN         NOT NULL DEFAULT false,
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    -- Reference
    reference_standard          VARCHAR(200),
    reference_section           VARCHAR(100),
    supporting_evidence         JSONB           DEFAULT '[]',
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_md_area CHECK (
        decision_area IN (
            'METHODOLOGY_TIER', 'EMISSION_FACTOR', 'ALLOCATION_METHOD',
            'BOUNDARY_SETTING', 'DATA_SOURCE', 'EXCLUSION', 'ESTIMATION',
            'RECALCULATION', 'AGGREGATION', 'REPORTING', 'OTHER'
        )
    ),
    CONSTRAINT chk_p043_md_impact CHECK (impact_on_total_tco2e IS NULL OR impact_on_total_tco2e >= 0)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_md_tenant             ON ghg_accounting_scope3_complete.methodology_decisions(tenant_id);
CREATE INDEX idx_p043_md_evidence           ON ghg_accounting_scope3_complete.methodology_decisions(evidence_id);
CREATE INDEX idx_p043_md_category           ON ghg_accounting_scope3_complete.methodology_decisions(category);
CREATE INDEX idx_p043_md_area               ON ghg_accounting_scope3_complete.methodology_decisions(decision_area);
CREATE INDEX idx_p043_md_date               ON ghg_accounting_scope3_complete.methodology_decisions(decision_date DESC);
CREATE INDEX idx_p043_md_by                ON ghg_accounting_scope3_complete.methodology_decisions(decision_by);
CREATE INDEX idx_p043_md_approved           ON ghg_accounting_scope3_complete.methodology_decisions(approved);
CREATE INDEX idx_p043_md_sensitivity        ON ghg_accounting_scope3_complete.methodology_decisions(sensitivity_flag) WHERE sensitivity_flag = true;
CREATE INDEX idx_p043_md_created            ON ghg_accounting_scope3_complete.methodology_decisions(created_at DESC);
CREATE INDEX idx_p043_md_alternatives       ON ghg_accounting_scope3_complete.methodology_decisions USING GIN(alternatives_considered);

-- Composite: evidence + category for per-category decisions
CREATE INDEX idx_p043_md_ev_category        ON ghg_accounting_scope3_complete.methodology_decisions(evidence_id, category, decision_date DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_md_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.methodology_decisions
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_accounting_scope3_complete.verifier_queries
-- =============================================================================
-- Query and response tracking during verification engagements. Each record
-- is a question raised by the verifier, the response provided, evidence
-- references, and resolution status. Supports iterative Q&A workflow.

CREATE TABLE ghg_accounting_scope3_complete.verifier_queries (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    evidence_id                 UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.evidence_packages(id) ON DELETE CASCADE,
    -- Query
    query_number                INTEGER         NOT NULL,
    query_date                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    query_category              VARCHAR(100),
    query_text                  TEXT            NOT NULL,
    priority                    VARCHAR(20)     DEFAULT 'MEDIUM',
    -- Response
    response_text               TEXT,
    response_date               TIMESTAMPTZ,
    responded_by                VARCHAR(255),
    -- Evidence
    evidence_ref                TEXT,
    evidence_documents          JSONB           DEFAULT '[]',
    -- Resolution
    status                      VARCHAR(30)     NOT NULL DEFAULT 'OPEN',
    resolved_date               TIMESTAMPTZ,
    resolution_notes            TEXT,
    -- Escalation
    escalated                   BOOLEAN         NOT NULL DEFAULT false,
    escalated_to                VARCHAR(255),
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_vq_number CHECK (query_number >= 1),
    CONSTRAINT chk_p043_vq_priority CHECK (
        priority IS NULL OR priority IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p043_vq_status CHECK (
        status IN ('OPEN', 'RESPONDED', 'FOLLOW_UP', 'RESOLVED', 'CLOSED', 'WITHDRAWN')
    ),
    CONSTRAINT uq_p043_vq_evidence_number UNIQUE (evidence_id, query_number)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_vq_tenant             ON ghg_accounting_scope3_complete.verifier_queries(tenant_id);
CREATE INDEX idx_p043_vq_evidence           ON ghg_accounting_scope3_complete.verifier_queries(evidence_id);
CREATE INDEX idx_p043_vq_date               ON ghg_accounting_scope3_complete.verifier_queries(query_date DESC);
CREATE INDEX idx_p043_vq_category           ON ghg_accounting_scope3_complete.verifier_queries(query_category);
CREATE INDEX idx_p043_vq_priority           ON ghg_accounting_scope3_complete.verifier_queries(priority);
CREATE INDEX idx_p043_vq_status             ON ghg_accounting_scope3_complete.verifier_queries(status);
CREATE INDEX idx_p043_vq_resolved           ON ghg_accounting_scope3_complete.verifier_queries(resolved_date);
CREATE INDEX idx_p043_vq_created            ON ghg_accounting_scope3_complete.verifier_queries(created_at DESC);

-- Composite: evidence + open queries for resolution queue
CREATE INDEX idx_p043_vq_ev_open            ON ghg_accounting_scope3_complete.verifier_queries(evidence_id, priority, query_date)
    WHERE status IN ('OPEN', 'FOLLOW_UP');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_vq_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.verifier_queries
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 5: ghg_accounting_scope3_complete.audit_findings
-- =============================================================================
-- Findings from internal or external audits/verifications. Each finding is
-- classified by type (non-conformity, observation, opportunity), severity,
-- and tracked through remediation to closure.

CREATE TABLE ghg_accounting_scope3_complete.audit_findings (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    evidence_id                 UUID            NOT NULL REFERENCES ghg_accounting_scope3_complete.evidence_packages(id) ON DELETE CASCADE,
    -- Finding
    finding_number              INTEGER         NOT NULL,
    finding_type                VARCHAR(30)     NOT NULL,
    severity                    VARCHAR(20)     NOT NULL DEFAULT 'MINOR',
    title                       VARCHAR(500)    NOT NULL,
    description                 TEXT            NOT NULL,
    -- Root cause
    root_cause                  TEXT,
    category                    ghg_accounting_scope3_complete.scope3_category_type,
    affected_data_points        INTEGER,
    -- Impact
    impact_on_tco2e             DECIMAL(15,3),
    materiality_flag            BOOLEAN         DEFAULT false,
    -- Remediation
    remediation                 TEXT,
    remediation_owner           VARCHAR(255),
    remediation_plan            JSONB           DEFAULT '[]',
    -- Timeline
    identified_date             DATE            NOT NULL DEFAULT CURRENT_DATE,
    due_date                    DATE,
    closed_date                 DATE,
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'OPEN',
    -- Verification
    verified_closed             BOOLEAN         NOT NULL DEFAULT false,
    verified_by                 VARCHAR(255),
    verified_at                 TIMESTAMPTZ,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p043_af_number CHECK (finding_number >= 1),
    CONSTRAINT chk_p043_af_type CHECK (
        finding_type IN ('NON_CONFORMITY', 'OBSERVATION', 'OPPORTUNITY', 'RECOMMENDATION', 'GOOD_PRACTICE')
    ),
    CONSTRAINT chk_p043_af_severity CHECK (
        severity IN ('CRITICAL', 'MAJOR', 'MINOR', 'OBSERVATION')
    ),
    CONSTRAINT chk_p043_af_affected CHECK (affected_data_points IS NULL OR affected_data_points >= 0),
    CONSTRAINT chk_p043_af_due CHECK (
        due_date IS NULL OR due_date >= identified_date
    ),
    CONSTRAINT chk_p043_af_closed CHECK (
        closed_date IS NULL OR closed_date >= identified_date
    ),
    CONSTRAINT chk_p043_af_status CHECK (
        status IN ('OPEN', 'IN_PROGRESS', 'REMEDIATED', 'VERIFIED_CLOSED', 'ACCEPTED', 'DEFERRED')
    ),
    CONSTRAINT uq_p043_af_evidence_number UNIQUE (evidence_id, finding_number)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_af_tenant             ON ghg_accounting_scope3_complete.audit_findings(tenant_id);
CREATE INDEX idx_p043_af_evidence           ON ghg_accounting_scope3_complete.audit_findings(evidence_id);
CREATE INDEX idx_p043_af_type               ON ghg_accounting_scope3_complete.audit_findings(finding_type);
CREATE INDEX idx_p043_af_severity           ON ghg_accounting_scope3_complete.audit_findings(severity);
CREATE INDEX idx_p043_af_category           ON ghg_accounting_scope3_complete.audit_findings(category);
CREATE INDEX idx_p043_af_status             ON ghg_accounting_scope3_complete.audit_findings(status);
CREATE INDEX idx_p043_af_due                ON ghg_accounting_scope3_complete.audit_findings(due_date);
CREATE INDEX idx_p043_af_materiality        ON ghg_accounting_scope3_complete.audit_findings(materiality_flag) WHERE materiality_flag = true;
CREATE INDEX idx_p043_af_created            ON ghg_accounting_scope3_complete.audit_findings(created_at DESC);

-- Composite: evidence + open findings by severity
CREATE INDEX idx_p043_af_ev_open            ON ghg_accounting_scope3_complete.audit_findings(evidence_id, severity, due_date)
    WHERE status IN ('OPEN', 'IN_PROGRESS');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p043_af_updated
    BEFORE UPDATE ON ghg_accounting_scope3_complete.audit_findings
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3_complete.fn_set_updated_at();

-- =============================================================================
-- Table 6: ghg_accounting_scope3_complete.scope3_complete_audit_trail
-- =============================================================================
-- Append-only audit trail for all ghg_accounting_scope3_complete data changes.
-- Implemented as a TimescaleDB hypertable partitioned by timestamp for
-- efficient time-range queries and automatic chunk management.

CREATE TABLE ghg_accounting_scope3_complete.scope3_complete_audit_trail (
    id                          UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    -- Event
    event_type                  VARCHAR(30)     NOT NULL,
    entity_type                 VARCHAR(100)    NOT NULL,
    entity_id                   UUID            NOT NULL,
    -- Context
    inventory_id                UUID,
    category                    ghg_accounting_scope3_complete.scope3_category_type,
    -- Data
    event_data                  JSONB           NOT NULL DEFAULT '{}',
    old_values                  JSONB,
    new_values                  JSONB,
    changed_fields              TEXT[],
    change_summary              TEXT,
    -- User
    user_id                     UUID,
    user_name                   VARCHAR(255),
    user_role                   VARCHAR(50),
    -- Timestamp
    timestamp                   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- System
    source_agent                VARCHAR(100),
    source_system               VARCHAR(100)    DEFAULT 'GREENLANG',
    session_id                  VARCHAR(100),
    ip_address                  VARCHAR(45),
    -- Provenance
    provenance_hash             VARCHAR(64)     NOT NULL,
    parent_hash                 VARCHAR(64),
    -- Constraints
    CONSTRAINT chk_p043_at_event_type CHECK (
        event_type IN (
            'CREATE', 'UPDATE', 'DELETE', 'CALCULATE', 'RECALCULATE',
            'CLASSIFY', 'SCREEN', 'ASSESS', 'VERIFY', 'APPROVE',
            'REJECT', 'SUBMIT', 'PUBLISH', 'ARCHIVE', 'IMPORT',
            'EXPORT', 'RECONCILE', 'ENGAGE', 'RESPOND', 'SYSTEM',
            'SCENARIO_CREATE', 'TARGET_SET', 'RISK_ASSESS', 'LCA_COMPUTE'
        )
    )
);

-- Primary key includes timestamp for hypertable partitioning
ALTER TABLE ghg_accounting_scope3_complete.scope3_complete_audit_trail
    ADD PRIMARY KEY (id, timestamp);

-- ---------------------------------------------------------------------------
-- TimescaleDB Hypertable
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable(
            'ghg_accounting_scope3_complete.scope3_complete_audit_trail',
            'timestamp',
            chunk_time_interval => INTERVAL '1 month',
            if_not_exists => TRUE
        );
        RAISE NOTICE 'TimescaleDB hypertable created for scope3_complete_audit_trail';
    ELSE
        RAISE NOTICE 'TimescaleDB not available - scope3_complete_audit_trail created as regular table';
    END IF;
END;
$$;

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p043_at_tenant             ON ghg_accounting_scope3_complete.scope3_complete_audit_trail(tenant_id, timestamp DESC);
CREATE INDEX idx_p043_at_event_type         ON ghg_accounting_scope3_complete.scope3_complete_audit_trail(event_type, timestamp DESC);
CREATE INDEX idx_p043_at_entity_type        ON ghg_accounting_scope3_complete.scope3_complete_audit_trail(entity_type, timestamp DESC);
CREATE INDEX idx_p043_at_entity_id          ON ghg_accounting_scope3_complete.scope3_complete_audit_trail(entity_id, timestamp DESC);
CREATE INDEX idx_p043_at_inventory          ON ghg_accounting_scope3_complete.scope3_complete_audit_trail(inventory_id, timestamp DESC);
CREATE INDEX idx_p043_at_category           ON ghg_accounting_scope3_complete.scope3_complete_audit_trail(category, timestamp DESC);
CREATE INDEX idx_p043_at_user_id            ON ghg_accounting_scope3_complete.scope3_complete_audit_trail(user_id, timestamp DESC);
CREATE INDEX idx_p043_at_source_agent       ON ghg_accounting_scope3_complete.scope3_complete_audit_trail(source_agent, timestamp DESC);
CREATE INDEX idx_p043_at_provenance         ON ghg_accounting_scope3_complete.scope3_complete_audit_trail(provenance_hash);
CREATE INDEX idx_p043_at_parent             ON ghg_accounting_scope3_complete.scope3_complete_audit_trail(parent_hash);
CREATE INDEX idx_p043_at_event_data         ON ghg_accounting_scope3_complete.scope3_complete_audit_trail USING GIN(event_data);
CREATE INDEX idx_p043_at_changed_fields     ON ghg_accounting_scope3_complete.scope3_complete_audit_trail USING GIN(changed_fields);

-- Composite: tenant + entity + time for entity history
CREATE INDEX idx_p043_at_tenant_entity      ON ghg_accounting_scope3_complete.scope3_complete_audit_trail(tenant_id, entity_type, entity_id, timestamp DESC);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_accounting_scope3_complete.evidence_packages ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.calculation_provenance ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.methodology_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.verifier_queries ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.audit_findings ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3_complete.scope3_complete_audit_trail ENABLE ROW LEVEL SECURITY;

CREATE POLICY p043_ep_tenant_isolation ON ghg_accounting_scope3_complete.evidence_packages
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_ep_service_bypass ON ghg_accounting_scope3_complete.evidence_packages
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_cp_tenant_isolation ON ghg_accounting_scope3_complete.calculation_provenance
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_cp_service_bypass ON ghg_accounting_scope3_complete.calculation_provenance
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_md_tenant_isolation ON ghg_accounting_scope3_complete.methodology_decisions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_md_service_bypass ON ghg_accounting_scope3_complete.methodology_decisions
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_vq_tenant_isolation ON ghg_accounting_scope3_complete.verifier_queries
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_vq_service_bypass ON ghg_accounting_scope3_complete.verifier_queries
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_af_tenant_isolation ON ghg_accounting_scope3_complete.audit_findings
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_af_service_bypass ON ghg_accounting_scope3_complete.audit_findings
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p043_at_tenant_isolation ON ghg_accounting_scope3_complete.scope3_complete_audit_trail
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p043_at_service_bypass ON ghg_accounting_scope3_complete.scope3_complete_audit_trail
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.evidence_packages TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.calculation_provenance TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.methodology_decisions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.verifier_queries TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.audit_findings TO PUBLIC;
GRANT SELECT, INSERT ON ghg_accounting_scope3_complete.scope3_complete_audit_trail TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3_complete.scope3_complete_audit_trail TO greenlang_service;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_accounting_scope3_complete.evidence_packages IS
    'Evidence package for third-party verification with assurance level, readiness scoring, verifier assignment, and document inventory.';
COMMENT ON TABLE ghg_accounting_scope3_complete.calculation_provenance IS
    'Calculation provenance chain with SHA-256 hash linking each step from raw inputs to final values for audit trail integrity.';
COMMENT ON TABLE ghg_accounting_scope3_complete.methodology_decisions IS
    'Documented methodology decisions with rationale, alternatives considered, and impact assessment for assurance defensibility.';
COMMENT ON TABLE ghg_accounting_scope3_complete.verifier_queries IS
    'Verifier query/response workflow tracking during assurance engagements with priority, evidence references, and resolution status.';
COMMENT ON TABLE ghg_accounting_scope3_complete.audit_findings IS
    'Audit findings (non-conformity, observation, opportunity) with severity, root cause, remediation plan, and closure tracking.';
COMMENT ON TABLE ghg_accounting_scope3_complete.scope3_complete_audit_trail IS
    'Append-only TimescaleDB audit trail for all ghg_accounting_scope3_complete data changes with SHA-256 hash chain provenance.';

COMMENT ON COLUMN ghg_accounting_scope3_complete.evidence_packages.assurance_level IS 'ISAE 3000/3410 assurance level: LIMITED or REASONABLE.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.evidence_packages.readiness_score IS 'Percentage readiness for verification (0-100).';

COMMENT ON COLUMN ghg_accounting_scope3_complete.calculation_provenance.provenance_hash IS 'SHA-256 hash of step inputs, formula, and output for chain-of-custody verification.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.calculation_provenance.parent_hash IS 'Hash of the preceding calculation step (forms a hash chain from raw data to final value).';

COMMENT ON COLUMN ghg_accounting_scope3_complete.scope3_complete_audit_trail.event_type IS 'Action type including standard CRUD plus domain-specific: SCENARIO_CREATE, TARGET_SET, RISK_ASSESS, LCA_COMPUTE.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.scope3_complete_audit_trail.provenance_hash IS 'SHA-256 hash of this audit entry for chain-of-custody integrity.';
COMMENT ON COLUMN ghg_accounting_scope3_complete.scope3_complete_audit_trail.parent_hash IS 'Hash of the previous audit entry for the same entity (forms a hash chain).';
