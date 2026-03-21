-- =============================================================================
-- V183: PACK-031 Industrial Energy Audit - Energy Audits
-- =============================================================================
-- Pack:         PACK-031 (Industrial Energy Audit Pack)
-- Migration:    003 of 010
-- Date:         March 2026
--
-- Core energy audit records with EN 16247 compliance checklists, audit
-- findings with savings estimates, and energy end-use breakdown analysis.
--
-- Tables (4):
--   1. pack031_energy_audit.energy_audits
--   2. pack031_energy_audit.audit_findings
--   3. pack031_energy_audit.energy_end_uses
--   4. pack031_energy_audit.en16247_checklists
--
-- Previous: V182__pack031_energy_audit_002.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack031_energy_audit.energy_audits
-- =============================================================================
-- Energy audit records tracking audit type, scope, total consumption,
-- EN 16247 compliance, and quality scoring.

CREATE TABLE pack031_energy_audit.energy_audits (
    audit_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    audit_type              VARCHAR(50)     NOT NULL,
    audit_date              DATE            NOT NULL,
    auditor_name            VARCHAR(255),
    auditor_organization    VARCHAR(255),
    auditor_certification   VARCHAR(100),
    audit_scope             JSONB           DEFAULT '{}',
    total_consumption_kwh   NUMERIC(18,6),
    total_cost_eur          NUMERIC(14,4),
    en16247_compliant       BOOLEAN         DEFAULT FALSE,
    quality_score           NUMERIC(5,2),
    status                  VARCHAR(30)     DEFAULT 'planned',
    report_url              TEXT,
    executive_summary       TEXT,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    completed_at            TIMESTAMPTZ,
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_audit_type CHECK (
        audit_type IN ('WALKTHROUGH', 'STANDARD', 'DETAILED', 'INVESTMENT_GRADE',
                       'PRELIMINARY', 'EN16247_1', 'EN16247_2', 'EN16247_3',
                       'EN16247_4', 'EN16247_5', 'ISO_50002')
    ),
    CONSTRAINT chk_p031_audit_consumption CHECK (
        total_consumption_kwh IS NULL OR total_consumption_kwh >= 0
    ),
    CONSTRAINT chk_p031_audit_cost CHECK (
        total_cost_eur IS NULL OR total_cost_eur >= 0
    ),
    CONSTRAINT chk_p031_audit_quality CHECK (
        quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 100)
    ),
    CONSTRAINT chk_p031_audit_status CHECK (
        status IN ('planned', 'in_progress', 'review', 'completed', 'archived')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p031_audit_facility   ON pack031_energy_audit.energy_audits(facility_id);
CREATE INDEX idx_p031_audit_tenant     ON pack031_energy_audit.energy_audits(tenant_id);
CREATE INDEX idx_p031_audit_type       ON pack031_energy_audit.energy_audits(audit_type);
CREATE INDEX idx_p031_audit_date       ON pack031_energy_audit.energy_audits(audit_date);
CREATE INDEX idx_p031_audit_status     ON pack031_energy_audit.energy_audits(status);
CREATE INDEX idx_p031_audit_en16247    ON pack031_energy_audit.energy_audits(en16247_compliant);
CREATE INDEX idx_p031_audit_quality    ON pack031_energy_audit.energy_audits(quality_score);
CREATE INDEX idx_p031_audit_created    ON pack031_energy_audit.energy_audits(created_at DESC);
CREATE INDEX idx_p031_audit_scope      ON pack031_energy_audit.energy_audits USING GIN(audit_scope);

-- Trigger
CREATE TRIGGER trg_p031_audit_updated
    BEFORE UPDATE ON pack031_energy_audit.energy_audits
    FOR EACH ROW EXECUTE FUNCTION pack031_energy_audit.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack031_energy_audit.audit_findings
-- =============================================================================
-- Audit findings with recommended actions, estimated savings, implementation
-- cost, payback period, priority, and complexity assessment.

CREATE TABLE pack031_energy_audit.audit_findings (
    finding_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id                UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audits(audit_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    system                  VARCHAR(100)    NOT NULL,
    description             TEXT            NOT NULL,
    current_state           TEXT,
    recommended_action      TEXT            NOT NULL,
    estimated_savings_kwh   NUMERIC(14,4),
    estimated_savings_eur   NUMERIC(14,4),
    implementation_cost_eur NUMERIC(14,4),
    payback_years           NUMERIC(8,2),
    priority                VARCHAR(20)     NOT NULL DEFAULT 'medium',
    complexity              VARCHAR(20)     DEFAULT 'medium',
    status                  VARCHAR(30)     DEFAULT 'identified',
    assigned_to             VARCHAR(255),
    target_date             DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_finding_savings_kwh CHECK (
        estimated_savings_kwh IS NULL OR estimated_savings_kwh >= 0
    ),
    CONSTRAINT chk_p031_finding_savings_eur CHECK (
        estimated_savings_eur IS NULL OR estimated_savings_eur >= 0
    ),
    CONSTRAINT chk_p031_finding_cost CHECK (
        implementation_cost_eur IS NULL OR implementation_cost_eur >= 0
    ),
    CONSTRAINT chk_p031_finding_payback CHECK (
        payback_years IS NULL OR payback_years >= 0
    ),
    CONSTRAINT chk_p031_finding_priority CHECK (
        priority IN ('critical', 'high', 'medium', 'low', 'informational')
    ),
    CONSTRAINT chk_p031_finding_complexity CHECK (
        complexity IS NULL OR complexity IN ('simple', 'medium', 'complex', 'major_project')
    ),
    CONSTRAINT chk_p031_finding_status CHECK (
        status IN ('identified', 'evaluated', 'approved', 'in_progress', 'implemented', 'verified', 'rejected')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p031_finding_audit    ON pack031_energy_audit.audit_findings(audit_id);
CREATE INDEX idx_p031_finding_tenant   ON pack031_energy_audit.audit_findings(tenant_id);
CREATE INDEX idx_p031_finding_system   ON pack031_energy_audit.audit_findings(system);
CREATE INDEX idx_p031_finding_priority ON pack031_energy_audit.audit_findings(priority);
CREATE INDEX idx_p031_finding_complex  ON pack031_energy_audit.audit_findings(complexity);
CREATE INDEX idx_p031_finding_status   ON pack031_energy_audit.audit_findings(status);
CREATE INDEX idx_p031_finding_payback  ON pack031_energy_audit.audit_findings(payback_years);

-- =============================================================================
-- Table 3: pack031_energy_audit.energy_end_uses
-- =============================================================================
-- Energy end-use breakdown by category showing annual consumption,
-- cost, and percentage contribution to total facility energy.

CREATE TABLE pack031_energy_audit.energy_end_uses (
    end_use_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id                UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audits(audit_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    category                VARCHAR(100)    NOT NULL,
    sub_category            VARCHAR(100),
    carrier_type            VARCHAR(100),
    annual_kwh              NUMERIC(18,6)   NOT NULL,
    annual_cost_eur         NUMERIC(14,4),
    percentage_total        NUMERIC(5,2),
    data_source             VARCHAR(100),
    confidence_level        VARCHAR(20),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_enduse_kwh CHECK (annual_kwh >= 0),
    CONSTRAINT chk_p031_enduse_cost CHECK (
        annual_cost_eur IS NULL OR annual_cost_eur >= 0
    ),
    CONSTRAINT chk_p031_enduse_pct CHECK (
        percentage_total IS NULL OR (percentage_total >= 0 AND percentage_total <= 100)
    ),
    CONSTRAINT chk_p031_enduse_category CHECK (
        category IN ('HVAC', 'LIGHTING', 'COMPRESSED_AIR', 'STEAM', 'MOTORS',
                     'PUMPS', 'FANS', 'PROCESS_HEAT', 'PROCESS_COOLING',
                     'REFRIGERATION', 'MATERIAL_HANDLING', 'OFFICE_EQUIPMENT',
                     'DATA_CENTER', 'WATER_TREATMENT', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p031_enduse_audit     ON pack031_energy_audit.energy_end_uses(audit_id);
CREATE INDEX idx_p031_enduse_tenant    ON pack031_energy_audit.energy_end_uses(tenant_id);
CREATE INDEX idx_p031_enduse_category  ON pack031_energy_audit.energy_end_uses(category);
CREATE INDEX idx_p031_enduse_kwh       ON pack031_energy_audit.energy_end_uses(annual_kwh DESC);

-- =============================================================================
-- Table 4: pack031_energy_audit.en16247_checklists
-- =============================================================================
-- EN 16247 compliance checklists tracking clause-by-clause compliance
-- with evidence documentation and auditor notes.

CREATE TABLE pack031_energy_audit.en16247_checklists (
    checklist_id            UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id                UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audits(audit_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    clause                  VARCHAR(30)     NOT NULL,
    requirement             TEXT            NOT NULL,
    status                  VARCHAR(30)     DEFAULT 'not_assessed',
    evidence                TEXT,
    notes                   TEXT,
    assessed_by             VARCHAR(255),
    assessed_date           DATE,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_checklist_status CHECK (
        status IN ('compliant', 'non_compliant', 'partial', 'not_applicable', 'not_assessed')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p031_checklist_audit  ON pack031_energy_audit.en16247_checklists(audit_id);
CREATE INDEX idx_p031_checklist_tenant ON pack031_energy_audit.en16247_checklists(tenant_id);
CREATE INDEX idx_p031_checklist_clause ON pack031_energy_audit.en16247_checklists(clause);
CREATE INDEX idx_p031_checklist_status ON pack031_energy_audit.en16247_checklists(status);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack031_energy_audit.energy_audits ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.audit_findings ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.energy_end_uses ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.en16247_checklists ENABLE ROW LEVEL SECURITY;

CREATE POLICY p031_audit_tenant_isolation
    ON pack031_energy_audit.energy_audits
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_audit_service_bypass
    ON pack031_energy_audit.energy_audits
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_finding_tenant_isolation
    ON pack031_energy_audit.audit_findings
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_finding_service_bypass
    ON pack031_energy_audit.audit_findings
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_enduse_tenant_isolation
    ON pack031_energy_audit.energy_end_uses
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_enduse_service_bypass
    ON pack031_energy_audit.energy_end_uses
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_checklist_tenant_isolation
    ON pack031_energy_audit.en16247_checklists
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_checklist_service_bypass
    ON pack031_energy_audit.en16247_checklists
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.energy_audits TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.audit_findings TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.energy_end_uses TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.en16247_checklists TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack031_energy_audit.energy_audits IS
    'Energy audit records tracking audit type, scope, total consumption, EN 16247 compliance, and quality scoring.';
COMMENT ON TABLE pack031_energy_audit.audit_findings IS
    'Audit findings with recommended actions, estimated savings, implementation cost, payback, priority, and complexity.';
COMMENT ON TABLE pack031_energy_audit.energy_end_uses IS
    'Energy end-use breakdown by category showing annual consumption, cost, and percentage contribution to facility total.';
COMMENT ON TABLE pack031_energy_audit.en16247_checklists IS
    'EN 16247 compliance checklists tracking clause-by-clause compliance with evidence documentation.';

COMMENT ON COLUMN pack031_energy_audit.energy_audits.audit_type IS
    'Type of audit: WALKTHROUGH, STANDARD, DETAILED, INVESTMENT_GRADE, EN16247 parts 1-5, ISO 50002.';
COMMENT ON COLUMN pack031_energy_audit.energy_audits.en16247_compliant IS
    'Whether the audit meets EN 16247 standard requirements.';
COMMENT ON COLUMN pack031_energy_audit.audit_findings.payback_years IS
    'Simple payback period in years for the recommended action.';
COMMENT ON COLUMN pack031_energy_audit.en16247_checklists.clause IS
    'EN 16247 clause reference (e.g., 5.1, 5.2, 6.1).';
