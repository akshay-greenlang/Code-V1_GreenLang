-- =============================================================================
-- V016: SOC 2 Type II Preparation Infrastructure
-- =============================================================================
-- Description: Creates SOC 2 audit preparation tables including assessments,
--              evidence, control tests, auditor requests, findings, remediations,
--              attestations, and audit project management with full audit trails.
-- Author: GreenLang Security Team
-- PRD: SEC-009 SOC 2 Type II Preparation
-- Requires: TimescaleDB (V002), uuid-ossp (V001)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Schema Setup
-- -----------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS soc2;

SET search_path TO soc2, public;

-- -----------------------------------------------------------------------------
-- 1. Assessments Table
-- -----------------------------------------------------------------------------
-- Self-assessment runs for SOC 2 readiness evaluation.
-- Tracks overall maturity scores and assessment lifecycle.

CREATE TABLE soc2.assessments (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Identification
    name VARCHAR(256) NOT NULL,
    description TEXT,

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'draft',

    -- Trust Service Categories (stored as JSONB array)
    tsc_categories JSONB NOT NULL DEFAULT '["security"]'::jsonb,

    -- Scores
    overall_score DECIMAL(5,2) NOT NULL DEFAULT 0,
    criteria_assessed INTEGER NOT NULL DEFAULT 0,
    criteria_compliant INTEGER NOT NULL DEFAULT 0,
    gaps_count INTEGER NOT NULL DEFAULT 0,
    evidence_count INTEGER NOT NULL DEFAULT 0,

    -- Ownership
    assessed_by UUID,
    reviewed_by UUID,
    approved_by UUID,

    -- Tenant Isolation
    tenant_id UUID,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT chk_assessment_status CHECK (
        status IN ('draft', 'in_progress', 'completed', 'reviewed', 'approved', 'archived')
    ),
    CONSTRAINT chk_assessment_score CHECK (
        overall_score >= 0 AND overall_score <= 100
    ),
    CONSTRAINT chk_assessment_counts CHECK (
        criteria_compliant <= criteria_assessed
    )
);

COMMENT ON TABLE soc2.assessments IS
    'Self-assessment runs for SOC 2 readiness. Each assessment evaluates '
    'the organization against applicable Trust Service Criteria and produces '
    'a maturity score with gap analysis.';

-- -----------------------------------------------------------------------------
-- 2. Assessment Criteria Table
-- -----------------------------------------------------------------------------
-- Individual criterion evaluations within an assessment.

CREATE TABLE soc2.assessment_criteria (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Foreign Keys
    assessment_id UUID NOT NULL
        REFERENCES soc2.assessments(id) ON DELETE CASCADE,

    -- Criterion Identification
    criterion_id VARCHAR(20) NOT NULL,

    -- Scores
    score INTEGER NOT NULL DEFAULT 0,
    control_status VARCHAR(30) NOT NULL DEFAULT 'not_started',

    -- Evidence
    evidence_count INTEGER NOT NULL DEFAULT 0,
    evidence_ids UUID[] DEFAULT '{}',

    -- Analysis
    gaps_identified TEXT,
    recommendations TEXT,
    notes TEXT,

    -- Ownership
    assessed_by UUID,
    assessed_at TIMESTAMPTZ,

    -- Tenant Isolation
    tenant_id UUID,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_criterion_id CHECK (
        criterion_id ~ '^(CC[1-9](\.[1-9][0-9]?)?|A1\.[1-3]|C1\.[1-2]|PI1\.[1-2]|P[1-8]\.[0-9])$'
    ),
    CONSTRAINT chk_criterion_score CHECK (
        score >= 0 AND score <= 4
    ),
    CONSTRAINT chk_control_status CHECK (
        control_status IN (
            'not_started', 'in_design', 'in_implementation',
            'implemented', 'in_testing', 'operating', 'not_applicable'
        )
    ),

    -- Unique constraint on assessment + criterion
    CONSTRAINT uq_assessment_criterion UNIQUE (assessment_id, criterion_id)
);

COMMENT ON TABLE soc2.assessment_criteria IS
    'Individual criterion evaluations within a self-assessment. Each record '
    'represents the assessment of a single SOC 2 criterion (e.g., CC6.1) '
    'with maturity score, evidence links, and gap analysis.';

-- -----------------------------------------------------------------------------
-- 3. Evidence Table
-- -----------------------------------------------------------------------------
-- Evidence artifacts supporting SOC 2 controls.

CREATE TABLE soc2.evidence (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Identification
    title VARCHAR(256) NOT NULL,
    description TEXT,
    evidence_type VARCHAR(30) NOT NULL,

    -- SOC 2 Mapping
    criterion_ids TEXT[] NOT NULL DEFAULT '{}',

    -- File Information
    file_path VARCHAR(1024) NOT NULL,
    file_name VARCHAR(256) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    mime_type VARCHAR(128) DEFAULT 'application/octet-stream',

    -- Collection
    collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    collection_method VARCHAR(30) DEFAULT 'manual',
    is_automated BOOLEAN NOT NULL DEFAULT FALSE,

    -- Retention
    retention_until DATE,

    -- Ownership
    uploaded_by UUID,
    verified_by UUID,
    verified_at TIMESTAMPTZ,

    -- Metadata
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Tenant Isolation
    tenant_id UUID,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_evidence_type CHECK (
        evidence_type IN (
            'document', 'screenshot', 'log', 'report', 'configuration',
            'inquiry', 'observation', 'reperformance', 'sample', 'ticket', 'metric'
        )
    ),
    CONSTRAINT chk_evidence_hash CHECK (
        file_hash ~ '^[a-f0-9]{64}$'
    ),
    CONSTRAINT chk_collection_method CHECK (
        collection_method IN ('manual', 'automated', 'api', 'import')
    )
);

COMMENT ON TABLE soc2.evidence IS
    'Evidence artifacts supporting SOC 2 control effectiveness. Stores '
    'metadata for evidence files (actual files in S3). Includes SHA-256 '
    'hash for integrity verification and audit trail.';

-- -----------------------------------------------------------------------------
-- 4. Evidence Packages Table
-- -----------------------------------------------------------------------------
-- Collections of evidence for auditor delivery.

CREATE TABLE soc2.evidence_packages (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Identification
    name VARCHAR(256) NOT NULL,
    description TEXT,

    -- Links
    auditor_request_id UUID,
    evidence_ids UUID[] NOT NULL DEFAULT '{}',

    -- Package Integrity
    package_hash VARCHAR(64),

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'draft',

    -- Ownership
    prepared_by UUID,
    delivered_at TIMESTAMPTZ,

    -- Tenant Isolation
    tenant_id UUID,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_package_status CHECK (
        status IN ('draft', 'ready', 'delivered', 'archived')
    )
);

COMMENT ON TABLE soc2.evidence_packages IS
    'Collections of evidence artifacts bundled for delivery to auditors. '
    'Links to auditor requests (PBC items) and tracks delivery status.';

-- -----------------------------------------------------------------------------
-- 5. Control Tests Table
-- -----------------------------------------------------------------------------
-- Tests of control design and operating effectiveness.

CREATE TABLE soc2.control_tests (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Identification
    criterion_id VARCHAR(20) NOT NULL,
    control_id VARCHAR(50) NOT NULL,
    test_type VARCHAR(20) NOT NULL,
    test_name VARCHAR(256) NOT NULL,

    -- Test Definition
    test_procedure TEXT NOT NULL,
    expected_result TEXT,

    -- Results
    actual_result TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    is_effective BOOLEAN,

    -- Period
    period_start DATE,
    period_end DATE,

    -- Tenant Isolation
    tenant_id UUID,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_test_type CHECK (
        test_type IN ('design', 'operating', 'both')
    ),
    CONSTRAINT chk_test_status CHECK (
        status IN ('pending', 'in_progress', 'completed', 'cancelled')
    )
);

COMMENT ON TABLE soc2.control_tests IS
    'Tests of control design and operating effectiveness. Each test validates '
    'that a control addresses its risk (design) and works as intended '
    '(operating effectiveness) throughout the audit period.';

-- -----------------------------------------------------------------------------
-- 6. Test Results Table
-- -----------------------------------------------------------------------------
-- Detailed results of control tests.

CREATE TABLE soc2.test_results (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Foreign Key
    control_test_id UUID NOT NULL
        REFERENCES soc2.control_tests(id) ON DELETE CASCADE,

    -- Sampling
    sample_size INTEGER NOT NULL DEFAULT 0,
    samples_tested INTEGER NOT NULL DEFAULT 0,

    -- Results
    exceptions_found INTEGER NOT NULL DEFAULT 0,
    exception_rate DECIMAL(5,2) NOT NULL DEFAULT 0,
    exception_details TEXT,
    conclusion TEXT,

    -- Ownership
    tested_by UUID,
    tested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Evidence
    evidence_ids UUID[] DEFAULT '{}',

    -- Tenant Isolation
    tenant_id UUID,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_result_counts CHECK (
        samples_tested <= sample_size
    ),
    CONSTRAINT chk_exception_rate CHECK (
        exception_rate >= 0 AND exception_rate <= 100
    )
);

COMMENT ON TABLE soc2.test_results IS
    'Detailed test results including sample sizes, exception rates, and '
    'conclusions. Links to evidence supporting the test results.';

-- -----------------------------------------------------------------------------
-- 7. Auditor Requests Table (PBC List)
-- -----------------------------------------------------------------------------
-- Requests from external auditors (Prepared By Client items).

CREATE TABLE soc2.auditor_requests (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Identification
    request_number VARCHAR(50) NOT NULL,
    title VARCHAR(256) NOT NULL,
    description TEXT,

    -- Priority and SLA
    priority VARCHAR(20) NOT NULL DEFAULT 'normal',
    requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    due_at TIMESTAMPTZ NOT NULL,

    -- SOC 2 Mapping
    criterion_ids TEXT[] DEFAULT '{}',

    -- Requestor
    requested_by VARCHAR(256),

    -- Assignment
    assigned_to UUID,

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    response TEXT,
    responded_at TIMESTAMPTZ,

    -- Links
    evidence_package_id UUID
        REFERENCES soc2.evidence_packages(id) ON DELETE SET NULL,
    audit_project_id UUID,

    -- Notes
    notes TEXT,

    -- Tenant Isolation
    tenant_id UUID,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_request_priority CHECK (
        priority IN ('critical', 'high', 'normal', 'low')
    ),
    CONSTRAINT chk_request_status CHECK (
        status IN ('open', 'in_progress', 'fulfilled', 'on_hold', 'cancelled')
    )
);

COMMENT ON TABLE soc2.auditor_requests IS
    'Auditor requests (PBC list items). Tracks external auditor information '
    'requests with SLA tracking, assignment, and evidence package delivery.';

-- -----------------------------------------------------------------------------
-- 8. Findings Table
-- -----------------------------------------------------------------------------
-- Audit findings and observations.

CREATE TABLE soc2.findings (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Identification
    finding_number VARCHAR(50) NOT NULL UNIQUE,
    title VARCHAR(256) NOT NULL,
    description TEXT NOT NULL,

    -- Classification
    classification VARCHAR(30) NOT NULL,

    -- SOC 2 Mapping
    criterion_ids TEXT[] DEFAULT '{}',
    control_id VARCHAR(50),

    -- Analysis
    root_cause TEXT,
    management_response TEXT,

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'open',

    -- Ownership
    identified_by UUID,
    identified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Links
    audit_project_id UUID,

    -- Repeat Finding
    is_repeat BOOLEAN NOT NULL DEFAULT FALSE,
    prior_finding_id UUID
        REFERENCES soc2.findings(id) ON DELETE SET NULL,

    -- Evidence
    evidence_ids UUID[] DEFAULT '{}',

    -- Tenant Isolation
    tenant_id UUID,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_finding_classification CHECK (
        classification IN (
            'observation', 'exception', 'deficiency',
            'significant_deficiency', 'material_weakness'
        )
    ),
    CONSTRAINT chk_finding_status CHECK (
        status IN ('open', 'remediated', 'closed', 'risk_accepted', 'deferred')
    )
);

COMMENT ON TABLE soc2.findings IS
    'Audit findings and observations identified during SOC 2 audits. '
    'Classifies findings by severity (observation through material weakness) '
    'and tracks remediation progress.';

-- -----------------------------------------------------------------------------
-- 9. Remediations Table
-- -----------------------------------------------------------------------------
-- Remediation plans for findings.

CREATE TABLE soc2.remediations (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Foreign Key
    finding_id UUID NOT NULL
        REFERENCES soc2.findings(id) ON DELETE CASCADE,

    -- Plan
    plan TEXT NOT NULL,

    -- Ownership
    owner UUID,

    -- Status
    status VARCHAR(30) NOT NULL DEFAULT 'open',

    -- Timeline
    target_date DATE,
    actual_completion_date DATE,

    -- Effort
    effort_hours INTEGER NOT NULL DEFAULT 0,
    actual_hours INTEGER NOT NULL DEFAULT 0,

    -- Validation
    validation_method TEXT,
    validated_by UUID,
    validated_at TIMESTAMPTZ,

    -- Notes
    notes TEXT,

    -- Tenant Isolation
    tenant_id UUID,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_remediation_status CHECK (
        status IN (
            'open', 'in_progress', 'pending_validation',
            'validated', 'closed', 'deferred', 'risk_accepted'
        )
    )
);

COMMENT ON TABLE soc2.remediations IS
    'Remediation plans for audit findings. Tracks remediation owner, timeline, '
    'effort, and validation status.';

-- -----------------------------------------------------------------------------
-- 10. Attestations Table
-- -----------------------------------------------------------------------------
-- Management attestation documents.

CREATE TABLE soc2.attestations (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Identification
    title VARCHAR(256) NOT NULL,
    content TEXT NOT NULL,
    attestation_type VARCHAR(50) NOT NULL,

    -- Links
    audit_project_id UUID,

    -- Signers
    required_signers UUID[] NOT NULL DEFAULT '{}',

    -- Status
    status VARCHAR(30) NOT NULL DEFAULT 'draft',

    -- Dates
    effective_date DATE,
    expiration_date DATE,

    -- Supersession
    supersedes_id UUID
        REFERENCES soc2.attestations(id) ON DELETE SET NULL,

    -- Tenant Isolation
    tenant_id UUID,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_attestation_type CHECK (
        attestation_type IN (
            'management_rep', 'sod', 'subservice', 'completeness',
            'assertion', 'carve_out', 'inclusive', 'other'
        )
    ),
    CONSTRAINT chk_attestation_status CHECK (
        status IN ('draft', 'pending_signatures', 'signed', 'expired', 'superseded')
    )
);

COMMENT ON TABLE soc2.attestations IS
    'Management attestation documents for SOC 2 audits. Includes management '
    'representations, statements on design, and other formal attestations.';

-- -----------------------------------------------------------------------------
-- 11. Attestation Signatures Table
-- -----------------------------------------------------------------------------
-- Electronic signatures on attestations.

CREATE TABLE soc2.attestation_signatures (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Foreign Key
    attestation_id UUID NOT NULL
        REFERENCES soc2.attestations(id) ON DELETE CASCADE,

    -- Signer
    signer_id UUID NOT NULL,
    signer_name VARCHAR(256) NOT NULL,
    signer_title VARCHAR(256) NOT NULL,

    -- Signature
    signed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    signature_hash VARCHAR(64) NOT NULL,

    -- Audit Trail
    ip_address VARCHAR(45),
    user_agent VARCHAR(512),

    -- Tenant Isolation
    tenant_id UUID,

    -- Constraints
    CONSTRAINT chk_signature_hash CHECK (
        signature_hash ~ '^[a-f0-9]{64}$'
    ),

    -- Unique signer per attestation
    CONSTRAINT uq_attestation_signer UNIQUE (attestation_id, signer_id)
);

COMMENT ON TABLE soc2.attestation_signatures IS
    'Electronic signatures on management attestations. Provides non-repudiation '
    'with signature hash, timestamp, and client information.';

-- -----------------------------------------------------------------------------
-- 12. Audit Projects Table
-- -----------------------------------------------------------------------------
-- SOC 2 audit engagements.

CREATE TABLE soc2.audit_projects (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Identification
    name VARCHAR(256) NOT NULL,
    description TEXT,

    -- Audit Firm
    audit_firm VARCHAR(256),
    lead_auditor VARCHAR(256),

    -- Scope
    tsc_categories JSONB NOT NULL DEFAULT '["security"]'::jsonb,
    report_type VARCHAR(20) NOT NULL DEFAULT 'type_ii',

    -- Period
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,

    -- Status
    status VARCHAR(30) NOT NULL DEFAULT 'planning',

    -- Ownership
    compliance_officer_id UUID,

    -- Tenant Isolation
    tenant_id UUID,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_project_report_type CHECK (
        report_type IN ('type_i', 'type_ii')
    ),
    CONSTRAINT chk_project_status CHECK (
        status IN ('planning', 'readiness', 'fieldwork', 'reporting', 'complete', 'cancelled')
    ),
    CONSTRAINT chk_project_period CHECK (
        period_start < period_end
    )
);

COMMENT ON TABLE soc2.audit_projects IS
    'SOC 2 audit engagements. Tracks the entire audit lifecycle from planning '
    'through report issuance, including audit firm, scope, and timeline.';

-- -----------------------------------------------------------------------------
-- 13. Audit Milestones Table
-- -----------------------------------------------------------------------------
-- Key audit milestones.

CREATE TABLE soc2.audit_milestones (
    id UUID NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Foreign Key
    audit_project_id UUID NOT NULL
        REFERENCES soc2.audit_projects(id) ON DELETE CASCADE,

    -- Identification
    name VARCHAR(256) NOT NULL,
    description TEXT,

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'pending',

    -- Timeline
    target_date DATE NOT NULL,
    actual_date DATE,

    -- Ownership
    owner UUID,

    -- Dependencies
    dependencies UUID[] DEFAULT '{}',

    -- Notes
    notes TEXT,

    -- Tenant Isolation
    tenant_id UUID,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_milestone_status CHECK (
        status IN ('pending', 'in_progress', 'completed', 'delayed', 'at_risk')
    )
);

COMMENT ON TABLE soc2.audit_milestones IS
    'Key audit milestones within a SOC 2 engagement. Tracks target and actual '
    'dates, dependencies, and status for project management.';

-- -----------------------------------------------------------------------------
-- 14. Auditor Access Log Table (TimescaleDB Hypertable)
-- -----------------------------------------------------------------------------
-- Immutable audit trail of all auditor portal access.

CREATE TABLE soc2.auditor_access_log (
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    id UUID NOT NULL DEFAULT uuid_generate_v4(),

    -- Session
    session_id UUID NOT NULL,
    auditor_email VARCHAR(256) NOT NULL,
    auditor_firm VARCHAR(256),

    -- Action
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    resource_name VARCHAR(256),

    -- Request Details
    ip_address VARCHAR(45),
    user_agent VARCHAR(512),
    request_path VARCHAR(1024),
    request_method VARCHAR(10),

    -- Response
    response_code INTEGER,
    duration_ms INTEGER,

    -- Context
    audit_project_id UUID,

    -- Tenant Isolation
    tenant_id UUID,

    -- Primary key on time + id for hypertable
    PRIMARY KEY (time, id),

    -- Constraints
    CONSTRAINT chk_access_action CHECK (
        action IN (
            'login', 'logout', 'view', 'download', 'search',
            'request_create', 'request_update', 'comment', 'export'
        )
    ),
    CONSTRAINT chk_access_resource_type CHECK (
        resource_type IS NULL OR resource_type IN (
            'assessment', 'evidence', 'evidence_package', 'finding',
            'attestation', 'control_test', 'request', 'report'
        )
    )
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable(
    'soc2.auditor_access_log',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Set retention policy (7 years for SOC 2 compliance)
SELECT add_retention_policy(
    'soc2.auditor_access_log',
    INTERVAL '2555 days',
    if_not_exists => TRUE
);

COMMENT ON TABLE soc2.auditor_access_log IS
    'Immutable audit trail of all auditor portal access. TimescaleDB hypertable '
    'with 7-day chunks and 7-year retention for SOC 2 compliance requirements.';

-- -----------------------------------------------------------------------------
-- 15. Indexes - Assessments
-- -----------------------------------------------------------------------------

CREATE INDEX idx_assessment_status ON soc2.assessments (status, created_at DESC);
CREATE INDEX idx_assessment_tenant ON soc2.assessments (tenant_id, status)
    WHERE tenant_id IS NOT NULL;
CREATE INDEX idx_assessment_score ON soc2.assessments (overall_score DESC)
    WHERE status = 'approved';

-- -----------------------------------------------------------------------------
-- 16. Indexes - Assessment Criteria
-- -----------------------------------------------------------------------------

CREATE INDEX idx_criteria_assessment ON soc2.assessment_criteria (assessment_id);
CREATE INDEX idx_criteria_criterion ON soc2.assessment_criteria (criterion_id, score);
CREATE INDEX idx_criteria_tenant ON soc2.assessment_criteria (tenant_id)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 17. Indexes - Evidence
-- -----------------------------------------------------------------------------

CREATE INDEX idx_evidence_type ON soc2.evidence (evidence_type, created_at DESC);
CREATE INDEX idx_evidence_hash ON soc2.evidence (file_hash);
CREATE INDEX idx_evidence_tenant ON soc2.evidence (tenant_id, evidence_type)
    WHERE tenant_id IS NOT NULL;
CREATE INDEX idx_evidence_criteria ON soc2.evidence USING GIN (criterion_ids);
CREATE INDEX idx_evidence_retention ON soc2.evidence (retention_until)
    WHERE retention_until IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 18. Indexes - Evidence Packages
-- -----------------------------------------------------------------------------

CREATE INDEX idx_package_status ON soc2.evidence_packages (status, created_at DESC);
CREATE INDEX idx_package_request ON soc2.evidence_packages (auditor_request_id)
    WHERE auditor_request_id IS NOT NULL;
CREATE INDEX idx_package_tenant ON soc2.evidence_packages (tenant_id)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 19. Indexes - Control Tests
-- -----------------------------------------------------------------------------

CREATE INDEX idx_test_criterion ON soc2.control_tests (criterion_id, test_type);
CREATE INDEX idx_test_status ON soc2.control_tests (status, is_effective);
CREATE INDEX idx_test_tenant ON soc2.control_tests (tenant_id)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 20. Indexes - Auditor Requests
-- -----------------------------------------------------------------------------

CREATE INDEX idx_request_status ON soc2.auditor_requests (status, due_at);
CREATE INDEX idx_request_priority ON soc2.auditor_requests (priority, status)
    WHERE status NOT IN ('fulfilled', 'cancelled');
CREATE INDEX idx_request_overdue ON soc2.auditor_requests (due_at)
    WHERE status NOT IN ('fulfilled', 'cancelled');
CREATE INDEX idx_request_assigned ON soc2.auditor_requests (assigned_to, status)
    WHERE assigned_to IS NOT NULL;
CREATE INDEX idx_request_project ON soc2.auditor_requests (audit_project_id)
    WHERE audit_project_id IS NOT NULL;
CREATE INDEX idx_request_tenant ON soc2.auditor_requests (tenant_id)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 21. Indexes - Findings
-- -----------------------------------------------------------------------------

CREATE INDEX idx_finding_classification ON soc2.findings (classification, status);
CREATE INDEX idx_finding_status ON soc2.findings (status, identified_at DESC);
CREATE INDEX idx_finding_project ON soc2.findings (audit_project_id)
    WHERE audit_project_id IS NOT NULL;
CREATE INDEX idx_finding_repeat ON soc2.findings (prior_finding_id)
    WHERE is_repeat = TRUE;
CREATE INDEX idx_finding_tenant ON soc2.findings (tenant_id)
    WHERE tenant_id IS NOT NULL;
CREATE INDEX idx_finding_criteria ON soc2.findings USING GIN (criterion_ids);

-- -----------------------------------------------------------------------------
-- 22. Indexes - Remediations
-- -----------------------------------------------------------------------------

CREATE INDEX idx_remediation_finding ON soc2.remediations (finding_id);
CREATE INDEX idx_remediation_status ON soc2.remediations (status, target_date);
CREATE INDEX idx_remediation_owner ON soc2.remediations (owner, status)
    WHERE owner IS NOT NULL;
CREATE INDEX idx_remediation_overdue ON soc2.remediations (target_date)
    WHERE status IN ('open', 'in_progress') AND target_date IS NOT NULL;
CREATE INDEX idx_remediation_tenant ON soc2.remediations (tenant_id)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 23. Indexes - Attestations
-- -----------------------------------------------------------------------------

CREATE INDEX idx_attestation_type ON soc2.attestations (attestation_type, status);
CREATE INDEX idx_attestation_project ON soc2.attestations (audit_project_id)
    WHERE audit_project_id IS NOT NULL;
CREATE INDEX idx_attestation_expiry ON soc2.attestations (expiration_date)
    WHERE status = 'signed' AND expiration_date IS NOT NULL;
CREATE INDEX idx_attestation_tenant ON soc2.attestations (tenant_id)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 24. Indexes - Audit Projects
-- -----------------------------------------------------------------------------

CREATE INDEX idx_project_status ON soc2.audit_projects (status, period_end);
CREATE INDEX idx_project_period ON soc2.audit_projects (period_start, period_end);
CREATE INDEX idx_project_tenant ON soc2.audit_projects (tenant_id)
    WHERE tenant_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 25. Indexes - Audit Milestones
-- -----------------------------------------------------------------------------

CREATE INDEX idx_milestone_project ON soc2.audit_milestones (audit_project_id, target_date);
CREATE INDEX idx_milestone_status ON soc2.audit_milestones (status, target_date);
CREATE INDEX idx_milestone_owner ON soc2.audit_milestones (owner, status)
    WHERE owner IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 26. Indexes - Auditor Access Log
-- -----------------------------------------------------------------------------

CREATE INDEX idx_access_session ON soc2.auditor_access_log (session_id, time DESC);
CREATE INDEX idx_access_auditor ON soc2.auditor_access_log (auditor_email, time DESC);
CREATE INDEX idx_access_action ON soc2.auditor_access_log (action, time DESC);
CREATE INDEX idx_access_resource ON soc2.auditor_access_log (resource_type, resource_id, time DESC)
    WHERE resource_type IS NOT NULL;
CREATE INDEX idx_access_project ON soc2.auditor_access_log (audit_project_id, time DESC)
    WHERE audit_project_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 27. Row-Level Security
-- -----------------------------------------------------------------------------

-- Enable RLS on all tables
ALTER TABLE soc2.assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE soc2.assessment_criteria ENABLE ROW LEVEL SECURITY;
ALTER TABLE soc2.evidence ENABLE ROW LEVEL SECURITY;
ALTER TABLE soc2.evidence_packages ENABLE ROW LEVEL SECURITY;
ALTER TABLE soc2.control_tests ENABLE ROW LEVEL SECURITY;
ALTER TABLE soc2.test_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE soc2.auditor_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE soc2.findings ENABLE ROW LEVEL SECURITY;
ALTER TABLE soc2.remediations ENABLE ROW LEVEL SECURITY;
ALTER TABLE soc2.attestations ENABLE ROW LEVEL SECURITY;
ALTER TABLE soc2.attestation_signatures ENABLE ROW LEVEL SECURITY;
ALTER TABLE soc2.audit_projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE soc2.audit_milestones ENABLE ROW LEVEL SECURITY;
ALTER TABLE soc2.auditor_access_log ENABLE ROW LEVEL SECURITY;

-- Generic tenant isolation policy (applied to all tables)
CREATE POLICY tenant_isolation ON soc2.assessments
    FOR ALL
    USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'compliance_officer', 'auditor')
    )
    WITH CHECK (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'compliance_officer')
    );

-- Apply similar policies to other tables
CREATE POLICY tenant_isolation ON soc2.assessment_criteria
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'compliance_officer', 'auditor')
    );

CREATE POLICY tenant_isolation ON soc2.evidence
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'compliance_officer', 'auditor')
    );

CREATE POLICY tenant_isolation ON soc2.evidence_packages
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'compliance_officer', 'auditor')
    );

CREATE POLICY tenant_isolation ON soc2.control_tests
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'compliance_officer', 'auditor')
    );

CREATE POLICY tenant_isolation ON soc2.test_results
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'compliance_officer', 'auditor')
    );

CREATE POLICY tenant_isolation ON soc2.auditor_requests
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'compliance_officer', 'auditor')
    );

CREATE POLICY tenant_isolation ON soc2.findings
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'compliance_officer', 'auditor')
    );

CREATE POLICY tenant_isolation ON soc2.remediations
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'compliance_officer', 'auditor')
    );

CREATE POLICY tenant_isolation ON soc2.attestations
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'compliance_officer', 'auditor')
    );

CREATE POLICY tenant_isolation ON soc2.attestation_signatures
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'compliance_officer', 'auditor')
    );

CREATE POLICY tenant_isolation ON soc2.audit_projects
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'compliance_officer', 'auditor')
    );

CREATE POLICY tenant_isolation ON soc2.audit_milestones
    FOR ALL USING (
        tenant_id IS NULL
        OR tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid
        OR NULLIF(current_setting('app.user_role', true), '') IN ('admin', 'compliance_officer', 'auditor')
    );

-- Auditor access log - read only for most roles, insert for system
CREATE POLICY access_log_read ON soc2.auditor_access_log
    FOR SELECT USING (
        NULLIF(current_setting('app.user_role', true), '') IN (
            'admin', 'compliance_officer', 'auditor', 'security_admin'
        )
    );

CREATE POLICY access_log_insert ON soc2.auditor_access_log
    FOR INSERT WITH CHECK (
        NULLIF(current_setting('app.user_role', true), '') IN (
            'admin', 'system', 'compliance_officer'
        )
    );

-- -----------------------------------------------------------------------------
-- 28. Triggers - Auto-update timestamps
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION soc2.update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_assessments_update
    BEFORE UPDATE ON soc2.assessments
    FOR EACH ROW EXECUTE FUNCTION soc2.update_timestamp();

CREATE TRIGGER trg_assessment_criteria_update
    BEFORE UPDATE ON soc2.assessment_criteria
    FOR EACH ROW EXECUTE FUNCTION soc2.update_timestamp();

CREATE TRIGGER trg_evidence_update
    BEFORE UPDATE ON soc2.evidence
    FOR EACH ROW EXECUTE FUNCTION soc2.update_timestamp();

CREATE TRIGGER trg_evidence_packages_update
    BEFORE UPDATE ON soc2.evidence_packages
    FOR EACH ROW EXECUTE FUNCTION soc2.update_timestamp();

CREATE TRIGGER trg_control_tests_update
    BEFORE UPDATE ON soc2.control_tests
    FOR EACH ROW EXECUTE FUNCTION soc2.update_timestamp();

CREATE TRIGGER trg_auditor_requests_update
    BEFORE UPDATE ON soc2.auditor_requests
    FOR EACH ROW EXECUTE FUNCTION soc2.update_timestamp();

CREATE TRIGGER trg_findings_update
    BEFORE UPDATE ON soc2.findings
    FOR EACH ROW EXECUTE FUNCTION soc2.update_timestamp();

CREATE TRIGGER trg_remediations_update
    BEFORE UPDATE ON soc2.remediations
    FOR EACH ROW EXECUTE FUNCTION soc2.update_timestamp();

CREATE TRIGGER trg_attestations_update
    BEFORE UPDATE ON soc2.attestations
    FOR EACH ROW EXECUTE FUNCTION soc2.update_timestamp();

CREATE TRIGGER trg_audit_projects_update
    BEFORE UPDATE ON soc2.audit_projects
    FOR EACH ROW EXECUTE FUNCTION soc2.update_timestamp();

CREATE TRIGGER trg_audit_milestones_update
    BEFORE UPDATE ON soc2.audit_milestones
    FOR EACH ROW EXECUTE FUNCTION soc2.update_timestamp();

-- -----------------------------------------------------------------------------
-- 29. Permissions Table
-- -----------------------------------------------------------------------------
-- SOC 2 specific permissions for RBAC integration.

CREATE TABLE IF NOT EXISTS public.permissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    resource VARCHAR(100),
    action VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert SOC 2 permissions (15 permissions)
INSERT INTO public.permissions (name, description, resource, action) VALUES
    -- Assessment permissions
    ('soc2:assessment:create', 'Create SOC 2 self-assessments', 'soc2.assessment', 'create'),
    ('soc2:assessment:read', 'View SOC 2 self-assessments', 'soc2.assessment', 'read'),
    ('soc2:assessment:update', 'Update SOC 2 self-assessments', 'soc2.assessment', 'update'),
    ('soc2:assessment:approve', 'Approve SOC 2 self-assessments', 'soc2.assessment', 'approve'),
    -- Evidence permissions
    ('soc2:evidence:create', 'Upload SOC 2 evidence', 'soc2.evidence', 'create'),
    ('soc2:evidence:read', 'View SOC 2 evidence', 'soc2.evidence', 'read'),
    ('soc2:evidence:delete', 'Delete SOC 2 evidence', 'soc2.evidence', 'delete'),
    -- Request permissions
    ('soc2:request:manage', 'Manage auditor requests', 'soc2.request', 'manage'),
    ('soc2:request:respond', 'Respond to auditor requests', 'soc2.request', 'respond'),
    -- Finding permissions
    ('soc2:finding:manage', 'Manage audit findings', 'soc2.finding', 'manage'),
    ('soc2:remediation:manage', 'Manage finding remediations', 'soc2.remediation', 'manage'),
    -- Attestation permissions
    ('soc2:attestation:sign', 'Sign management attestations', 'soc2.attestation', 'sign'),
    ('soc2:attestation:manage', 'Manage attestation documents', 'soc2.attestation', 'manage'),
    -- Project permissions
    ('soc2:project:manage', 'Manage audit projects', 'soc2.project', 'manage'),
    -- Auditor portal permissions
    ('soc2:auditor_portal:access', 'Access auditor portal', 'soc2.auditor_portal', 'access')
ON CONFLICT (name) DO NOTHING;

-- -----------------------------------------------------------------------------
-- 30. Role Mappings
-- -----------------------------------------------------------------------------
-- Map permissions to roles.

-- Create role_permissions table if not exists
CREATE TABLE IF NOT EXISTS public.role_permissions (
    role_name VARCHAR(50) NOT NULL,
    permission_id UUID NOT NULL REFERENCES public.permissions(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (role_name, permission_id)
);

-- Compliance Officer role - full SOC 2 access
INSERT INTO public.role_permissions (role_name, permission_id)
SELECT 'compliance_officer', id FROM public.permissions
WHERE name LIKE 'soc2:%'
ON CONFLICT DO NOTHING;

-- Auditor role - read-only access to assessments, evidence, findings
INSERT INTO public.role_permissions (role_name, permission_id)
SELECT 'auditor', id FROM public.permissions
WHERE name IN (
    'soc2:assessment:read',
    'soc2:evidence:read',
    'soc2:request:manage',
    'soc2:auditor_portal:access'
)
ON CONFLICT DO NOTHING;

-- Internal Audit role - read and manage findings
INSERT INTO public.role_permissions (role_name, permission_id)
SELECT 'internal_audit', id FROM public.permissions
WHERE name IN (
    'soc2:assessment:read',
    'soc2:assessment:create',
    'soc2:evidence:read',
    'soc2:evidence:create',
    'soc2:finding:manage',
    'soc2:remediation:manage'
)
ON CONFLICT DO NOTHING;

-- Control Owner role - update assessments and evidence
INSERT INTO public.role_permissions (role_name, permission_id)
SELECT 'control_owner', id FROM public.permissions
WHERE name IN (
    'soc2:assessment:read',
    'soc2:assessment:update',
    'soc2:evidence:read',
    'soc2:evidence:create',
    'soc2:request:respond',
    'soc2:remediation:manage'
)
ON CONFLICT DO NOTHING;

-- Executive role - sign attestations
INSERT INTO public.role_permissions (role_name, permission_id)
SELECT 'executive', id FROM public.permissions
WHERE name IN (
    'soc2:assessment:read',
    'soc2:assessment:approve',
    'soc2:attestation:sign'
)
ON CONFLICT DO NOTHING;

-- -----------------------------------------------------------------------------
-- 31. Verification
-- -----------------------------------------------------------------------------

DO $$
DECLARE
    tbl_name TEXT;
    required_tables TEXT[] := ARRAY[
        'assessments',
        'assessment_criteria',
        'evidence',
        'evidence_packages',
        'control_tests',
        'test_results',
        'auditor_requests',
        'findings',
        'remediations',
        'attestations',
        'attestation_signatures',
        'audit_projects',
        'audit_milestones',
        'auditor_access_log'
    ];
    perm_count INTEGER;
BEGIN
    -- Verify all tables exist
    FOREACH tbl_name IN ARRAY required_tables
    LOOP
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'soc2' AND table_name = tbl_name
        ) THEN
            RAISE EXCEPTION 'Required table soc2.% was not created', tbl_name;
        ELSE
            RAISE NOTICE 'Table soc2.% created successfully', tbl_name;
        END IF;
    END LOOP;

    -- Verify RLS is enabled on assessments
    IF NOT EXISTS (
        SELECT 1 FROM pg_tables
        WHERE schemaname = 'soc2'
          AND tablename = 'assessments'
          AND rowsecurity = TRUE
    ) THEN
        RAISE EXCEPTION 'RLS not enabled on soc2.assessments';
    END IF;

    -- Verify permissions were inserted
    SELECT COUNT(*) INTO perm_count
    FROM public.permissions
    WHERE name LIKE 'soc2:%';

    IF perm_count < 15 THEN
        RAISE EXCEPTION 'Expected 15 SOC 2 permissions, found %', perm_count;
    END IF;

    -- Verify hypertable was created
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables
        WHERE hypertable_schema = 'soc2'
          AND hypertable_name = 'auditor_access_log'
    ) THEN
        RAISE EXCEPTION 'TimescaleDB hypertable not created for auditor_access_log';
    END IF;

    RAISE NOTICE 'V016 SOC 2 Preparation migration completed successfully';
    RAISE NOTICE 'Created 14 tables, 15 permissions, hypertable with 7-year retention';
END $$;
