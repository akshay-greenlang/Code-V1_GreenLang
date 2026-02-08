-- =============================================================================
-- GreenLang Climate OS - Citations & Evidence Agent Service Schema
-- =============================================================================
-- Migration: V025
-- Component: AGENT-FOUND-005 Citations & Evidence Agent
-- Description: Creates citations_service schema with citation registry,
--              evidence packages, verification tracking, methodology
--              references, regulatory mappings, data source attributions,
--              SHA-256 audit trail, hypertables for citation_versions and
--              citation_verifications, continuous aggregates, RLS policies,
--              and seed data for common emission factor citations (DEFRA,
--              EPA, IPCC, GHG Protocol, ISO, Ecoinvent).
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS citations_service;

-- =============================================================================
-- Table: citations_service.citations
-- =============================================================================
-- Main registry of all citations used in zero-hallucination compliance
-- calculations. Each citation has a unique ID, type, source authority,
-- publication metadata, verification status, regulatory framework tags,
-- provenance hash, and multi-tenant isolation via tenant_id.

CREATE TABLE citations_service.citations (
    citation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    citation_type VARCHAR(50) NOT NULL,
    source_authority VARCHAR(50) NOT NULL,
    -- Metadata
    title TEXT NOT NULL,
    authors TEXT[] DEFAULT '{}',
    publication_date DATE,
    version VARCHAR(50),
    edition VARCHAR(50),
    publisher VARCHAR(255),
    url TEXT,
    doi VARCHAR(255),
    isbn VARCHAR(20),
    issn VARCHAR(20),
    page_numbers VARCHAR(50),
    chapter VARCHAR(100),
    section VARCHAR(100),
    table_reference VARCHAR(100),
    -- Versioning
    effective_date DATE NOT NULL,
    expiration_date DATE,
    superseded_by UUID REFERENCES citations_service.citations(citation_id),
    supersedes UUID REFERENCES citations_service.citations(citation_id),
    -- Verification
    verification_status VARCHAR(20) NOT NULL DEFAULT 'unverified',
    verified_at TIMESTAMPTZ,
    verified_by VARCHAR(255),
    -- Regulatory
    regulatory_frameworks TEXT[] DEFAULT '{}',
    -- Content
    abstract TEXT,
    key_values JSONB DEFAULT '{}',
    notes TEXT,
    -- Provenance
    content_hash VARCHAR(64),
    -- Multi-tenancy
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255),
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE
);

-- Citation type constraint
ALTER TABLE citations_service.citations
    ADD CONSTRAINT chk_citation_type
    CHECK (citation_type IN (
        'emission_factor', 'regulatory', 'methodology', 'scientific',
        'company_data', 'guidance', 'database'
    ));

-- Source authority constraint
ALTER TABLE citations_service.citations
    ADD CONSTRAINT chk_source_authority
    CHECK (source_authority IN (
        'defra', 'epa', 'ecoinvent', 'ipcc', 'ghg_protocol',
        'iso', 'iea', 'eea', 'unfccc', 'cdp', 'sbti',
        'gri', 'sasb', 'tcfd', 'tnfd', 'eu_commission',
        'uk_gov', 'us_gov', 'custom'
    ));

-- Verification status constraint
ALTER TABLE citations_service.citations
    ADD CONSTRAINT chk_verification_status
    CHECK (verification_status IN (
        'unverified', 'pending', 'verified', 'rejected', 'expired', 'superseded'
    ));

-- Content hash must be 64-character hex when present
ALTER TABLE citations_service.citations
    ADD CONSTRAINT chk_citation_content_hash_length
    CHECK (content_hash IS NULL OR LENGTH(content_hash) = 64);

-- Expiration date must be after effective date when set
ALTER TABLE citations_service.citations
    ADD CONSTRAINT chk_citation_effective_range
    CHECK (expiration_date IS NULL OR expiration_date >= effective_date);

-- =============================================================================
-- Table: citations_service.citation_versions
-- =============================================================================
-- TimescaleDB hypertable recording every version of every citation for
-- full audit trail and historical analysis. Each row is an immutable snapshot
-- of a citation at a point in time, with change details, provenance hashes.
-- Partitioned by timestamp for efficient time-series queries.

CREATE TABLE citations_service.citation_versions (
    version_id UUID DEFAULT gen_random_uuid(),
    citation_id UUID NOT NULL REFERENCES citations_service.citations(citation_id),
    version_number INTEGER NOT NULL,
    change_type VARCHAR(20) NOT NULL,
    changed_fields JSONB DEFAULT '{}',
    previous_hash VARCHAR(64),
    current_hash VARCHAR(64),
    changed_by VARCHAR(255),
    change_reason TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    PRIMARY KEY (version_id, timestamp)
);

-- Create hypertable partitioned by timestamp
SELECT create_hypertable('citations_service.citation_versions', 'timestamp', if_not_exists => TRUE);

-- Change type constraint
ALTER TABLE citations_service.citation_versions
    ADD CONSTRAINT chk_version_change_type
    CHECK (change_type IN (
        'create', 'update', 'delete', 'verify', 'supersede',
        'expire', 'restore', 'merge'
    ));

-- Version number must be positive
ALTER TABLE citations_service.citation_versions
    ADD CONSTRAINT chk_citation_version_number_positive
    CHECK (version_number > 0);

-- Hash must be 64-character hex when present
ALTER TABLE citations_service.citation_versions
    ADD CONSTRAINT chk_version_current_hash_length
    CHECK (current_hash IS NULL OR LENGTH(current_hash) = 64);

ALTER TABLE citations_service.citation_versions
    ADD CONSTRAINT chk_version_previous_hash_length
    CHECK (previous_hash IS NULL OR LENGTH(previous_hash) = 64);

-- =============================================================================
-- Table: citations_service.evidence_packages
-- =============================================================================
-- Evidence packages bundle citations and supporting data for a specific
-- calculation or compliance context. Each package can be finalized to
-- create an immutable, hash-verified evidence record for audit submission.

CREATE TABLE citations_service.evidence_packages (
    package_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT DEFAULT '',
    calculation_context JSONB DEFAULT '{}',
    calculation_result JSONB DEFAULT '{}',
    regulatory_frameworks TEXT[] DEFAULT '{}',
    compliance_notes TEXT,
    package_hash VARCHAR(64),
    is_finalized BOOLEAN NOT NULL DEFAULT FALSE,
    finalized_at TIMESTAMPTZ,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255),
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE
);

-- Package hash must be 64-character hex when present
ALTER TABLE citations_service.evidence_packages
    ADD CONSTRAINT chk_package_hash_length
    CHECK (package_hash IS NULL OR LENGTH(package_hash) = 64);

-- =============================================================================
-- Table: citations_service.evidence_items
-- =============================================================================
-- Individual evidence items within an evidence package. Each item links to
-- one or more citations, has a content hash, and records the source system
-- and agent that produced it.

CREATE TABLE citations_service.evidence_items (
    item_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    package_id UUID NOT NULL REFERENCES citations_service.evidence_packages(package_id),
    evidence_type VARCHAR(30) NOT NULL,
    description TEXT NOT NULL,
    data JSONB DEFAULT '{}',
    citation_ids UUID[] DEFAULT '{}',
    source_system VARCHAR(100),
    source_agent VARCHAR(100),
    content_hash VARCHAR(64),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default'
);

-- Evidence type constraint
ALTER TABLE citations_service.evidence_items
    ADD CONSTRAINT chk_evidence_type
    CHECK (evidence_type IN (
        'emission_factor', 'calculation', 'assumption', 'methodology',
        'data_source', 'conversion', 'regulatory', 'attestation',
        'measurement', 'estimate', 'proxy'
    ));

-- Content hash must be 64-character hex when present
ALTER TABLE citations_service.evidence_items
    ADD CONSTRAINT chk_evidence_item_hash_length
    CHECK (content_hash IS NULL OR LENGTH(content_hash) = 64);

-- =============================================================================
-- Table: citations_service.citation_verifications
-- =============================================================================
-- TimescaleDB hypertable recording every verification event for citations.
-- Provides a full audit trail of when citations were verified, by whom,
-- using which method, and the outcome. Retained for 90 days with
-- compression after 7 days.

CREATE TABLE citations_service.citation_verifications (
    verification_id UUID DEFAULT gen_random_uuid(),
    citation_id UUID NOT NULL REFERENCES citations_service.citations(citation_id),
    status VARCHAR(20) NOT NULL,
    previous_status VARCHAR(20),
    verification_method VARCHAR(50),
    details JSONB DEFAULT '{}',
    verified_by VARCHAR(255),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    PRIMARY KEY (verification_id, timestamp)
);

-- Create hypertable partitioned by timestamp
SELECT create_hypertable('citations_service.citation_verifications', 'timestamp', if_not_exists => TRUE);

-- Verification status constraint
ALTER TABLE citations_service.citation_verifications
    ADD CONSTRAINT chk_verification_status_val
    CHECK (status IN (
        'unverified', 'pending', 'verified', 'rejected', 'expired', 'superseded'
    ));

-- Verification method constraint
ALTER TABLE citations_service.citation_verifications
    ADD CONSTRAINT chk_verification_method
    CHECK (verification_method IS NULL OR verification_method IN (
        'manual_review', 'automated_check', 'cross_reference',
        'url_validation', 'doi_lookup', 'registry_check',
        'expert_review', 'peer_review', 'periodic_recheck'
    ));

-- =============================================================================
-- Table: citations_service.methodology_references
-- =============================================================================
-- Methodology references link compliance standards and calculation methods
-- to their authoritative citations. Each reference records which scopes
-- and categories the methodology applies to.

CREATE TABLE citations_service.methodology_references (
    reference_id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    standard VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    section VARCHAR(100),
    description TEXT DEFAULT '',
    citation_id UUID REFERENCES citations_service.citations(citation_id),
    scope_1_applicable BOOLEAN DEFAULT FALSE,
    scope_2_applicable BOOLEAN DEFAULT FALSE,
    scope_3_applicable BOOLEAN DEFAULT FALSE,
    applicable_categories TEXT[] DEFAULT '{}',
    formula_id VARCHAR(100),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- Table: citations_service.regulatory_mappings
-- =============================================================================
-- Maps regulatory framework requirements to their authoritative citations.
-- Each mapping records the specific article, requirement text, compliance
-- deadline, and applicable scopes.

CREATE TABLE citations_service.regulatory_mappings (
    mapping_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    framework VARCHAR(30) NOT NULL,
    article VARCHAR(50),
    requirement_text TEXT NOT NULL,
    citation_id UUID REFERENCES citations_service.citations(citation_id),
    effective_date DATE NOT NULL,
    compliance_deadline DATE,
    applies_to_scope_1 BOOLEAN DEFAULT FALSE,
    applies_to_scope_2 BOOLEAN DEFAULT FALSE,
    applies_to_scope_3 BOOLEAN DEFAULT FALSE,
    compliance_status VARCHAR(30),
    compliance_evidence UUID[] DEFAULT '{}',
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Framework constraint
ALTER TABLE citations_service.regulatory_mappings
    ADD CONSTRAINT chk_regulatory_framework
    CHECK (framework IN (
        'csrd', 'cbam', 'eu_taxonomy', 'sfdr', 'sec_climate',
        'sb253', 'sb261', 'ghg_protocol', 'iso_14064',
        'tcfd', 'tnfd', 'cdp', 'gri', 'custom'
    ));

-- Compliance status constraint
ALTER TABLE citations_service.regulatory_mappings
    ADD CONSTRAINT chk_compliance_status
    CHECK (compliance_status IS NULL OR compliance_status IN (
        'not_started', 'in_progress', 'compliant', 'non_compliant',
        'partially_compliant', 'exempted', 'pending_review'
    ));

-- =============================================================================
-- Table: citations_service.data_source_attributions
-- =============================================================================
-- Tracks the provenance of data values extracted from authoritative sources.
-- Each attribution links a source dataset/version to a citation and records
-- the extracted values with validity dates.

CREATE TABLE citations_service.data_source_attributions (
    attribution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_authority VARCHAR(50) NOT NULL,
    dataset_name VARCHAR(255) NOT NULL,
    dataset_version VARCHAR(50) NOT NULL,
    citation_id UUID REFERENCES citations_service.citations(citation_id),
    extracted_values JSONB DEFAULT '{}',
    extraction_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    extracted_by VARCHAR(255),
    valid_from DATE NOT NULL,
    valid_until DATE,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Source authority constraint
ALTER TABLE citations_service.data_source_attributions
    ADD CONSTRAINT chk_attribution_source_authority
    CHECK (source_authority IN (
        'defra', 'epa', 'ecoinvent', 'ipcc', 'ghg_protocol',
        'iso', 'iea', 'eea', 'unfccc', 'cdp', 'sbti',
        'gri', 'sasb', 'tcfd', 'tnfd', 'eu_commission',
        'uk_gov', 'us_gov', 'custom'
    ));

-- Valid_until must be after valid_from when set
ALTER TABLE citations_service.data_source_attributions
    ADD CONSTRAINT chk_attribution_validity_range
    CHECK (valid_until IS NULL OR valid_until >= valid_from);

-- =============================================================================
-- Continuous Aggregate: citations_service.daily_verification_counts
-- =============================================================================
-- Precomputed daily counts of citation verification events by status for
-- dashboard queries and trend analysis.

CREATE MATERIALIZED VIEW citations_service.daily_verification_counts
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS bucket,
    status,
    COUNT(*) AS total_verifications,
    COUNT(DISTINCT citation_id) AS citations_verified,
    COUNT(DISTINCT verified_by) AS unique_verifiers
FROM citations_service.citation_verifications
WHERE timestamp IS NOT NULL
GROUP BY bucket, status
WITH NO DATA;

-- Refresh policy: refresh every 30 minutes, covering the last 2 days
SELECT add_continuous_aggregate_policy('citations_service.daily_verification_counts',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Continuous Aggregate: citations_service.daily_operation_counts
-- =============================================================================
-- Precomputed daily citation version operation summaries for monitoring
-- change velocity and operation type distribution.

CREATE MATERIALIZED VIEW citations_service.daily_operation_counts
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS bucket,
    change_type,
    COUNT(*) AS total_operations,
    COUNT(DISTINCT citation_id) AS affected_citations,
    COUNT(DISTINCT changed_by) AS unique_users
FROM citations_service.citation_versions
WHERE timestamp IS NOT NULL
GROUP BY bucket, change_type
WITH NO DATA;

-- Refresh policy: refresh every 30 minutes, covering the last 2 days
SELECT add_continuous_aggregate_policy('citations_service.daily_operation_counts',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- citations indexes
CREATE INDEX idx_citations_type ON citations_service.citations(citation_type);
CREATE INDEX idx_citations_source_authority ON citations_service.citations(source_authority);
CREATE INDEX idx_citations_verification_status ON citations_service.citations(verification_status);
CREATE INDEX idx_citations_tenant ON citations_service.citations(tenant_id);
CREATE INDEX idx_citations_effective_date ON citations_service.citations(effective_date);
CREATE INDEX idx_citations_expiration_date ON citations_service.citations(expiration_date);
CREATE INDEX idx_citations_doi ON citations_service.citations(doi);
CREATE INDEX idx_citations_isbn ON citations_service.citations(isbn);
CREATE INDEX idx_citations_issn ON citations_service.citations(issn);
CREATE INDEX idx_citations_content_hash ON citations_service.citations(content_hash);
CREATE INDEX idx_citations_superseded_by ON citations_service.citations(superseded_by);
CREATE INDEX idx_citations_supersedes ON citations_service.citations(supersedes);
CREATE INDEX idx_citations_created_at ON citations_service.citations(created_at DESC);
CREATE INDEX idx_citations_updated_at ON citations_service.citations(updated_at DESC);
CREATE INDEX idx_citations_publisher ON citations_service.citations(publisher);
CREATE INDEX idx_citations_is_deleted ON citations_service.citations(is_deleted);
CREATE INDEX idx_citations_title_text ON citations_service.citations USING GIN (to_tsvector('english', title));
CREATE INDEX idx_citations_regulatory_frameworks ON citations_service.citations USING GIN (regulatory_frameworks);
CREATE INDEX idx_citations_authors ON citations_service.citations USING GIN (authors);
CREATE INDEX idx_citations_key_values ON citations_service.citations USING GIN (key_values);

-- citation_versions indexes (hypertable-aware)
CREATE INDEX idx_versions_citation ON citations_service.citation_versions(citation_id, timestamp DESC);
CREATE INDEX idx_versions_tenant ON citations_service.citation_versions(tenant_id, timestamp DESC);
CREATE INDEX idx_versions_changed_by ON citations_service.citation_versions(changed_by, timestamp DESC);
CREATE INDEX idx_versions_change_type ON citations_service.citation_versions(change_type, timestamp DESC);
CREATE INDEX idx_versions_current_hash ON citations_service.citation_versions(current_hash);
CREATE INDEX idx_versions_number ON citations_service.citation_versions(citation_id, version_number DESC, timestamp DESC);

-- evidence_packages indexes
CREATE INDEX idx_packages_tenant ON citations_service.evidence_packages(tenant_id);
CREATE INDEX idx_packages_finalized ON citations_service.evidence_packages(is_finalized, tenant_id);
CREATE INDEX idx_packages_created_at ON citations_service.evidence_packages(created_at DESC);
CREATE INDEX idx_packages_created_by ON citations_service.evidence_packages(created_by);
CREATE INDEX idx_packages_hash ON citations_service.evidence_packages(package_hash);
CREATE INDEX idx_packages_is_deleted ON citations_service.evidence_packages(is_deleted);
CREATE INDEX idx_packages_regulatory ON citations_service.evidence_packages USING GIN (regulatory_frameworks);
CREATE INDEX idx_packages_calc_context ON citations_service.evidence_packages USING GIN (calculation_context);

-- evidence_items indexes
CREATE INDEX idx_items_package ON citations_service.evidence_items(package_id);
CREATE INDEX idx_items_type ON citations_service.evidence_items(evidence_type);
CREATE INDEX idx_items_tenant ON citations_service.evidence_items(tenant_id);
CREATE INDEX idx_items_source_system ON citations_service.evidence_items(source_system);
CREATE INDEX idx_items_source_agent ON citations_service.evidence_items(source_agent);
CREATE INDEX idx_items_content_hash ON citations_service.evidence_items(content_hash);
CREATE INDEX idx_items_timestamp ON citations_service.evidence_items(timestamp DESC);
CREATE INDEX idx_items_citation_ids ON citations_service.evidence_items USING GIN (citation_ids);

-- citation_verifications indexes (hypertable-aware)
CREATE INDEX idx_verifications_citation ON citations_service.citation_verifications(citation_id, timestamp DESC);
CREATE INDEX idx_verifications_status ON citations_service.citation_verifications(status, timestamp DESC);
CREATE INDEX idx_verifications_tenant ON citations_service.citation_verifications(tenant_id, timestamp DESC);
CREATE INDEX idx_verifications_verified_by ON citations_service.citation_verifications(verified_by, timestamp DESC);
CREATE INDEX idx_verifications_method ON citations_service.citation_verifications(verification_method, timestamp DESC);

-- methodology_references indexes
CREATE INDEX idx_methodology_standard ON citations_service.methodology_references(standard);
CREATE INDEX idx_methodology_citation ON citations_service.methodology_references(citation_id);
CREATE INDEX idx_methodology_tenant ON citations_service.methodology_references(tenant_id);
CREATE INDEX idx_methodology_scope1 ON citations_service.methodology_references(scope_1_applicable);
CREATE INDEX idx_methodology_scope2 ON citations_service.methodology_references(scope_2_applicable);
CREATE INDEX idx_methodology_scope3 ON citations_service.methodology_references(scope_3_applicable);
CREATE INDEX idx_methodology_categories ON citations_service.methodology_references USING GIN (applicable_categories);

-- regulatory_mappings indexes
CREATE INDEX idx_regulatory_framework ON citations_service.regulatory_mappings(framework);
CREATE INDEX idx_regulatory_citation ON citations_service.regulatory_mappings(citation_id);
CREATE INDEX idx_regulatory_tenant ON citations_service.regulatory_mappings(tenant_id);
CREATE INDEX idx_regulatory_effective ON citations_service.regulatory_mappings(effective_date);
CREATE INDEX idx_regulatory_deadline ON citations_service.regulatory_mappings(compliance_deadline);
CREATE INDEX idx_regulatory_status ON citations_service.regulatory_mappings(compliance_status);
CREATE INDEX idx_regulatory_evidence ON citations_service.regulatory_mappings USING GIN (compliance_evidence);

-- data_source_attributions indexes
CREATE INDEX idx_attributions_source ON citations_service.data_source_attributions(source_authority);
CREATE INDEX idx_attributions_dataset ON citations_service.data_source_attributions(dataset_name);
CREATE INDEX idx_attributions_citation ON citations_service.data_source_attributions(citation_id);
CREATE INDEX idx_attributions_tenant ON citations_service.data_source_attributions(tenant_id);
CREATE INDEX idx_attributions_valid_from ON citations_service.data_source_attributions(valid_from);
CREATE INDEX idx_attributions_valid_until ON citations_service.data_source_attributions(valid_until);
CREATE INDEX idx_attributions_extracted_values ON citations_service.data_source_attributions USING GIN (extracted_values);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE citations_service.citations ENABLE ROW LEVEL SECURITY;
CREATE POLICY citations_tenant_read ON citations_service.citations
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY citations_tenant_write ON citations_service.citations
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE citations_service.citation_versions ENABLE ROW LEVEL SECURITY;
CREATE POLICY versions_tenant_read ON citations_service.citation_versions
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY versions_tenant_write ON citations_service.citation_versions
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE citations_service.evidence_packages ENABLE ROW LEVEL SECURITY;
CREATE POLICY packages_tenant_read ON citations_service.evidence_packages
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY packages_tenant_write ON citations_service.evidence_packages
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE citations_service.evidence_items ENABLE ROW LEVEL SECURITY;
CREATE POLICY items_tenant_read ON citations_service.evidence_items
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY items_tenant_write ON citations_service.evidence_items
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE citations_service.citation_verifications ENABLE ROW LEVEL SECURITY;
CREATE POLICY verifications_tenant_read ON citations_service.citation_verifications
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY verifications_tenant_write ON citations_service.citation_verifications
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE citations_service.methodology_references ENABLE ROW LEVEL SECURITY;
CREATE POLICY methodology_tenant_read ON citations_service.methodology_references
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY methodology_tenant_write ON citations_service.methodology_references
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE citations_service.regulatory_mappings ENABLE ROW LEVEL SECURITY;
CREATE POLICY regulatory_tenant_read ON citations_service.regulatory_mappings
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY regulatory_tenant_write ON citations_service.regulatory_mappings
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE citations_service.data_source_attributions ENABLE ROW LEVEL SECURITY;
CREATE POLICY attributions_tenant_read ON citations_service.data_source_attributions
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY attributions_tenant_write ON citations_service.data_source_attributions
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA citations_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA citations_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA citations_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON citations_service.daily_verification_counts TO greenlang_app;
GRANT SELECT ON citations_service.daily_operation_counts TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA citations_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA citations_service TO greenlang_readonly;
GRANT SELECT ON citations_service.daily_verification_counts TO greenlang_readonly;
GRANT SELECT ON citations_service.daily_operation_counts TO greenlang_readonly;

-- Add citations service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'citations:read', 'citations', 'read', 'View citations and their metadata'),
    (gen_random_uuid(), 'citations:write', 'citations', 'write', 'Create and update citations'),
    (gen_random_uuid(), 'citations:delete', 'citations', 'delete', 'Delete citations from the registry'),
    (gen_random_uuid(), 'citations:versions:read', 'citations', 'versions_read', 'View citation version history'),
    (gen_random_uuid(), 'citations:versions:write', 'citations', 'versions_write', 'Create new citation versions'),
    (gen_random_uuid(), 'citations:verify', 'citations', 'verify', 'Verify and validate citations'),
    (gen_random_uuid(), 'citations:evidence:read', 'citations', 'evidence_read', 'View evidence packages and items'),
    (gen_random_uuid(), 'citations:evidence:write', 'citations', 'evidence_write', 'Create and manage evidence packages'),
    (gen_random_uuid(), 'citations:evidence:finalize', 'citations', 'evidence_finalize', 'Finalize evidence packages for audit'),
    (gen_random_uuid(), 'citations:methodology:read', 'citations', 'methodology_read', 'View methodology references'),
    (gen_random_uuid(), 'citations:methodology:write', 'citations', 'methodology_write', 'Manage methodology references'),
    (gen_random_uuid(), 'citations:regulatory:read', 'citations', 'regulatory_read', 'View regulatory mappings'),
    (gen_random_uuid(), 'citations:export', 'citations', 'export', 'Export citations and evidence packages'),
    (gen_random_uuid(), 'citations:admin', 'citations', 'admin', 'Citations service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep citation verifications for 90 days
SELECT add_retention_policy('citations_service.citation_verifications', INTERVAL '90 days');

-- citation_versions: no retention (unlimited history for compliance)

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on citation_versions after 30 days
ALTER TABLE citations_service.citation_versions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'citation_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('citations_service.citation_versions', INTERVAL '30 days');

-- Enable compression on citation_verifications after 7 days
ALTER TABLE citations_service.citation_verifications SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'citation_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('citations_service.citation_verifications', INTERVAL '7 days');

-- =============================================================================
-- Seed: Citation Registry - DEFRA 2024/2025 Publications
-- =============================================================================

INSERT INTO citations_service.citations (citation_id, citation_type, source_authority, title, authors, publication_date, version, publisher, url, effective_date, expiration_date, verification_status, verified_at, verified_by, regulatory_frameworks, key_values, content_hash, tenant_id, created_by) VALUES

-- DEFRA 2024
('a0000001-0001-4000-8000-000000000001', 'emission_factor', 'defra', 'UK Government GHG Conversion Factors for Company Reporting 2024', '{"Department for Energy Security and Net Zero", "DEFRA"}', '2024-06-01', '2024', 'Crown Copyright', 'https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024', '2024-06-01', '2025-05-31', 'verified', '2026-01-15 10:00:00+00', 'system', '{"ghg_protocol", "iso_14064", "csrd"}', '{"scope1_fuels": true, "scope2_electricity": true, "scope3_transport": true, "units": "kgCO2e"}', 'a1a1a1a1b2b2b2b2c3c3c3c3d4d4d4d4e5e5e5e5f6f6f6f6a7a7a7a7b8b8b8b8', 'default', 'system'),

-- DEFRA 2025
('a0000001-0001-4000-8000-000000000002', 'emission_factor', 'defra', 'UK Government GHG Conversion Factors for Company Reporting 2025', '{"Department for Energy Security and Net Zero", "DEFRA"}', '2025-06-01', '2025', 'Crown Copyright', 'https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2025', '2025-06-01', NULL, 'verified', '2026-01-20 10:00:00+00', 'system', '{"ghg_protocol", "iso_14064", "csrd"}', '{"scope1_fuels": true, "scope2_electricity": true, "scope3_transport": true, "units": "kgCO2e"}', 'b2b2b2b2c3c3c3c3d4d4d4d4e5e5e5e5f6f6f6f6a7a7a7a7b8b8b8b8c9c9c9c9', 'default', 'system'),

-- EPA eGRID 2024
('a0000001-0001-4000-8000-000000000003', 'emission_factor', 'epa', 'Emissions & Generation Resource Integrated Database (eGRID) 2024', '{"US Environmental Protection Agency"}', '2024-01-31', '2024', 'US EPA', 'https://www.epa.gov/egrid', '2024-01-31', NULL, 'verified', '2026-01-15 11:00:00+00', 'system', '{"ghg_protocol", "sec_climate"}', '{"grid_factors": true, "subregions": true, "units": "lbCO2e/MWh", "year": 2022}', 'c3c3c3c3d4d4d4d4e5e5e5e5f6f6f6f6a7a7a7a7b8b8b8b8c9c9c9c9d0d0d0d0', 'default', 'system'),

-- EPA AP-42
('a0000001-0001-4000-8000-000000000004', 'emission_factor', 'epa', 'Compilation of Air Pollutant Emission Factors (AP-42), Fifth Edition', '{"US Environmental Protection Agency"}', '2023-09-01', '5th Edition', 'US EPA', 'https://www.epa.gov/air-emissions-factors-and-quantification/ap-42-compilation-air-emission-factors', '2023-09-01', NULL, 'verified', '2026-01-15 12:00:00+00', 'system', '{"ghg_protocol"}', '{"stationary_combustion": true, "mobile_sources": true, "units": "various"}', 'd4d4d4d4e5e5e5e5f6f6f6f6a7a7a7a7b8b8b8b8c9c9c9c9d0d0d0d0e1e1e1e1', 'default', 'system'),

-- EPA 40 CFR Part 98
('a0000001-0001-4000-8000-000000000005', 'regulatory', 'epa', 'Mandatory Greenhouse Gas Reporting Rule (40 CFR Part 98)', '{"US Environmental Protection Agency"}', '2024-01-01', '2024', 'US Federal Register', 'https://www.epa.gov/ghgreporting', '2024-01-01', NULL, 'verified', '2026-01-15 13:00:00+00', 'system', '{"ghg_protocol", "sec_climate"}', '{"subparts": 41, "reporting_threshold_mtco2e": 25000}', 'e5e5e5e5f6f6f6f6a7a7a7a7b8b8b8b8c9c9c9c9d0d0d0d0e1e1e1e1f2f2f2f2', 'default', 'system'),

-- IPCC AR6 WG1
('a0000001-0001-4000-8000-000000000006', 'scientific', 'ipcc', 'Climate Change 2021: The Physical Science Basis (AR6 WG1)', '{"IPCC Working Group I"}', '2021-08-09', 'AR6', 'Cambridge University Press', 'https://www.ipcc.ch/report/ar6/wg1/', '2021-08-09', NULL, 'verified', '2026-01-10 10:00:00+00', 'system', '{"ghg_protocol", "csrd", "cbam"}', '{"gwp_ch4_100yr": 27.9, "gwp_n2o_100yr": 273, "gwp_sf6_100yr": 25200}', 'f6f6f6f6a7a7a7a7b8b8b8b8c9c9c9c9d0d0d0d0e1e1e1e1f2f2f2f2a3a3a3a3', 'default', 'system'),

-- IPCC 2006 Guidelines (2019 Refinement)
('a0000001-0001-4000-8000-000000000007', 'methodology', 'ipcc', '2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories', '{"IPCC Task Force on National Greenhouse Gas Inventories"}', '2019-05-01', '2019 Refinement', 'IPCC', 'https://www.ipcc-nggip.iges.or.jp/public/2019rf/index.html', '2019-05-01', NULL, 'verified', '2026-01-10 11:00:00+00', 'system', '{"ghg_protocol", "unfccc"}', '{"volumes": 5, "sectors": ["energy", "ippu", "agriculture", "lulucf", "waste"]}', 'a7a7a7a7b8b8b8b8c9c9c9c9d0d0d0d0e1e1e1e1f2f2f2f2a3a3a3a3b4b4b4b4', 'default', 'system'),

-- GHG Protocol Corporate Standard
('a0000001-0001-4000-8000-000000000008', 'methodology', 'ghg_protocol', 'A Corporate Accounting and Reporting Standard (Revised Edition)', '{"World Resources Institute", "WBCSD"}', '2015-01-01', 'Revised', 'WRI/WBCSD', 'https://ghgprotocol.org/corporate-standard', '2015-01-01', NULL, 'verified', '2026-01-10 12:00:00+00', 'system', '{"ghg_protocol", "csrd", "cdp", "sbti"}', '{"scopes": [1, 2, 3], "principles": ["relevance", "completeness", "consistency", "transparency", "accuracy"]}', 'b8b8b8b8c9c9c9c9d0d0d0d0e1e1e1e1f2f2f2f2a3a3a3a3b4b4b4b4c5c5c5c5', 'default', 'system'),

-- GHG Protocol Scope 3
('a0000001-0001-4000-8000-000000000009', 'methodology', 'ghg_protocol', 'Corporate Value Chain (Scope 3) Accounting and Reporting Standard', '{"World Resources Institute", "WBCSD"}', '2011-10-01', '1.0', 'WRI/WBCSD', 'https://ghgprotocol.org/standards/scope-3-standard', '2011-10-01', NULL, 'verified', '2026-01-10 13:00:00+00', 'system', '{"ghg_protocol", "csrd", "cdp", "sbti"}', '{"categories": 15, "upstream": 8, "downstream": 7}', 'c9c9c9c9d0d0d0d0e1e1e1e1f2f2f2f2a3a3a3a3b4b4b4b4c5c5c5c5d6d6d6d6', 'default', 'system'),

-- GHG Protocol Scope 2
('a0000001-0001-4000-8000-000000000010', 'methodology', 'ghg_protocol', 'Scope 2 Guidance: An amendment to the GHG Protocol Corporate Standard', '{"World Resources Institute", "WBCSD"}', '2015-01-01', '1.0', 'WRI/WBCSD', 'https://ghgprotocol.org/scope_2_guidance', '2015-01-01', NULL, 'verified', '2026-01-10 14:00:00+00', 'system', '{"ghg_protocol", "csrd", "cdp"}', '{"methods": ["location-based", "market-based"]}', 'd0d0d0d0e1e1e1e1f2f2f2f2a3a3a3a3b4b4b4b4c5c5c5c5d6d6d6d6e7e7e7e7', 'default', 'system'),

-- ISO 14064-1:2018
('a0000001-0001-4000-8000-000000000011', 'methodology', 'iso', 'ISO 14064-1:2018 Greenhouse gases -- Part 1: Specification with guidance for quantification and reporting', '{"International Organization for Standardization"}', '2018-12-01', '2018', 'ISO', 'https://www.iso.org/standard/66453.html', '2018-12-01', NULL, 'verified', '2026-01-12 10:00:00+00', 'system', '{"iso_14064", "csrd"}', '{"parts": 3, "scopes": [1, 2, 3], "categories": 6}', 'e1e1e1e1f2f2f2f2a3a3a3a3b4b4b4b4c5c5c5c5d6d6d6d6e7e7e7e7f8f8f8f8', 'default', 'system'),

-- ISO 14064-3:2019
('a0000001-0001-4000-8000-000000000012', 'methodology', 'iso', 'ISO 14064-3:2019 Greenhouse gases -- Part 3: Specification with guidance for validation and verification', '{"International Organization for Standardization"}', '2019-04-01', '2019', 'ISO', 'https://www.iso.org/standard/66455.html', '2019-04-01', NULL, 'verified', '2026-01-12 11:00:00+00', 'system', '{"iso_14064", "csrd"}', '{"verification_levels": ["limited", "reasonable"]}', 'f2f2f2f2a3a3a3a3b4b4b4b4c5c5c5c5d6d6d6d6e7e7e7e7f8f8f8f8a9a9a9a9', 'default', 'system'),

-- Ecoinvent 3.10
('a0000001-0001-4000-8000-000000000013', 'database', 'ecoinvent', 'ecoinvent Database v3.10', '{"ecoinvent Association"}', '2023-12-01', '3.10', 'ecoinvent Association', 'https://ecoinvent.org/the-ecoinvent-database/', '2023-12-01', NULL, 'verified', '2026-01-15 15:00:00+00', 'system', '{"ghg_protocol", "iso_14064"}', '{"datasets": 21238, "geographies": 144, "system_models": ["cutoff", "apos", "consequential"]}', 'a3a3a3a3b4b4b4b4c5c5c5c5d6d6d6d6e7e7e7e7f8f8f8f8a9a9a9a9b0b0b0b0', 'default', 'system'),

-- IEA Emission Factors 2024
('a0000001-0001-4000-8000-000000000014', 'emission_factor', 'iea', 'IEA CO2 Emissions from Fuel Combustion 2024', '{"International Energy Agency"}', '2024-10-01', '2024', 'OECD/IEA', 'https://www.iea.org/data-and-statistics/data-product/co2-emissions-from-fuel-combustion', '2024-10-01', NULL, 'verified', '2026-01-15 16:00:00+00', 'system', '{"ghg_protocol", "cdp"}', '{"countries": 150, "fuels": true, "electricity": true}', 'b4b4b4b4c5c5c5c5d6d6d6d6e7e7e7e7f8f8f8f8a9a9a9a9b0b0b0b0c1c1c1c1', 'default', 'system'),

-- EEA Emission Factors
('a0000001-0001-4000-8000-000000000015', 'emission_factor', 'eea', 'EEA Greenhouse Gas Emission Intensity of Electricity Generation in Europe', '{"European Environment Agency"}', '2024-11-01', '2024', 'EEA', 'https://www.eea.europa.eu/data-and-maps/indicators/overview-of-the-electricity-production-3', '2024-11-01', NULL, 'verified', '2026-01-15 17:00:00+00', 'system', '{"csrd", "eu_taxonomy"}', '{"eu_member_states": 27, "units": "gCO2/kWh"}', 'c5c5c5c5d6d6d6d6e7e7e7e7f8f8f8f8a9a9a9a9b0b0b0b0c1c1c1c1d2d2d2d2', 'default', 'system'),

-- CSRD (EU)
('a0000001-0001-4000-8000-000000000016', 'regulatory', 'eu_commission', 'Corporate Sustainability Reporting Directive (CSRD) - Directive (EU) 2022/2464', '{"European Parliament", "Council of the European Union"}', '2022-12-14', '2022/2464', 'Official Journal of the European Union', 'https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022L2464', '2024-01-01', NULL, 'verified', '2026-01-10 15:00:00+00', 'system', '{"csrd"}', '{"applies_to": "large_undertakings", "reporting_standards": "ESRS"}', 'd6d6d6d6e7e7e7e7f8f8f8f8a9a9a9a9b0b0b0b0c1c1c1c1d2d2d2d2e3e3e3e3', 'default', 'system'),

-- CBAM (EU)
('a0000001-0001-4000-8000-000000000017', 'regulatory', 'eu_commission', 'Carbon Border Adjustment Mechanism (CBAM) - Regulation (EU) 2023/956', '{"European Parliament", "Council of the European Union"}', '2023-05-17', '2023/956', 'Official Journal of the European Union', 'https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32023R0956', '2023-10-01', NULL, 'verified', '2026-01-10 16:00:00+00', 'system', '{"cbam"}', '{"sectors": ["iron_steel", "cement", "aluminium", "fertilizers", "electricity", "hydrogen"], "transition_period": "2023-2025", "definitive_period": "2026+"}', 'e7e7e7e7f8f8f8f8a9a9a9a9b0b0b0b0c1c1c1c1d2d2d2d2e3e3e3e3f4f4f4f4', 'default', 'system'),

-- SBTi Net-Zero Standard
('a0000001-0001-4000-8000-000000000018', 'guidance', 'sbti', 'SBTi Corporate Net-Zero Standard v1.1', '{"Science Based Targets initiative"}', '2023-11-01', '1.1', 'SBTi', 'https://sciencebasedtargets.org/net-zero', '2023-11-01', NULL, 'verified', '2026-01-12 12:00:00+00', 'system', '{"sbti", "ghg_protocol"}', '{"pathways": ["1.5C", "well_below_2C"], "reduction_rate_annual_pct": 4.2}', 'f8f8f8f8a9a9a9a9b0b0b0b0c1c1c1c1d2d2d2d2e3e3e3e3f4f4f4f4a5a5a5a5', 'default', 'system'),

-- CDP Climate Questionnaire
('a0000001-0001-4000-8000-000000000019', 'guidance', 'cdp', 'CDP Climate Change 2024 Questionnaire', '{"CDP"}', '2024-02-01', '2024', 'CDP', 'https://www.cdp.net/en/guidance/guidance-for-companies', '2024-02-01', NULL, 'verified', '2026-01-12 13:00:00+00', 'system', '{"cdp", "ghg_protocol"}', '{"modules": ["governance", "risks_opportunities", "strategy", "targets_performance", "emissions_methodology", "verification", "carbon_pricing", "engagement"]}', 'a9a9a9a9b0b0b0b0c1c1c1c1d2d2d2d2e3e3e3e3f4f4f4f4a5a5a5a5b6b6b6b6', 'default', 'system'),

-- UNFCCC National Inventory Submissions
('a0000001-0001-4000-8000-000000000020', 'database', 'unfccc', 'UNFCCC National Inventory Submissions 2024', '{"UNFCCC Secretariat"}', '2024-04-15', '2024', 'UNFCCC', 'https://unfccc.int/ghg-inventories-annex-i-parties/2024', '2024-04-15', NULL, 'verified', '2026-01-15 18:00:00+00', 'system', '{"unfccc", "ghg_protocol"}', '{"annex_i_parties": 43, "reporting_year": 2022}', 'b0b0b0b0c1c1c1c1d2d2d2d2e3e3e3e3f4f4f4f4a5a5a5a5b6b6b6b6c7c7c7c7', 'default', 'system'),

-- DEFRA 2023 (superseded by 2024)
('a0000001-0001-4000-8000-000000000021', 'emission_factor', 'defra', 'UK Government GHG Conversion Factors for Company Reporting 2023', '{"Department for Energy Security and Net Zero", "DEFRA"}', '2023-06-01', '2023', 'Crown Copyright', 'https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2023', '2023-06-01', '2024-05-31', 'superseded', '2024-06-01 10:00:00+00', 'system', '{"ghg_protocol", "iso_14064"}', '{"scope1_fuels": true, "scope2_electricity": true, "scope3_transport": true, "units": "kgCO2e"}', 'c1c1c1c1d2d2d2d2e3e3e3e3f4f4f4f4a5a5a5a5b6b6b6b6c7c7c7c7d8d8d8d8', 'default', 'system'),

-- GRI Standards 2021
('a0000001-0001-4000-8000-000000000022', 'guidance', 'gri', 'GRI Universal Standards 2021', '{"Global Reporting Initiative"}', '2021-10-05', '2021', 'GRI', 'https://www.globalreporting.org/standards/', '2023-01-01', NULL, 'verified', '2026-01-12 14:00:00+00', 'system', '{"gri", "csrd"}', '{"standards": ["GRI 1", "GRI 2", "GRI 3"], "topic_standards": 31}', 'd2d2d2d2e3e3e3e3f4f4f4f4a5a5a5a5b6b6b6b6c7c7c7c7d8d8d8d8e9e9e9e9', 'default', 'system')

ON CONFLICT (citation_id) DO NOTHING;

-- Mark superseded citations
UPDATE citations_service.citations
SET superseded_by = 'a0000001-0001-4000-8000-000000000001'
WHERE citation_id = 'a0000001-0001-4000-8000-000000000021';

UPDATE citations_service.citations
SET supersedes = 'a0000001-0001-4000-8000-000000000021'
WHERE citation_id = 'a0000001-0001-4000-8000-000000000001';

-- =============================================================================
-- Seed: Initial Versions for Seeded Citations
-- =============================================================================

INSERT INTO citations_service.citation_versions (citation_id, version_number, change_type, changed_fields, current_hash, changed_by, change_reason, tenant_id) VALUES
('a0000001-0001-4000-8000-000000000001', 1, 'create', '{}', 'a1a1a1a1b2b2b2b2c3c3c3c3d4d4d4d4e5e5e5e5f6f6f6f6a7a7a7a7b8b8b8b8', 'system', 'Initial seed: DEFRA 2024 conversion factors', 'default'),
('a0000001-0001-4000-8000-000000000002', 1, 'create', '{}', 'b2b2b2b2c3c3c3c3d4d4d4d4e5e5e5e5f6f6f6f6a7a7a7a7b8b8b8b8c9c9c9c9', 'system', 'Initial seed: DEFRA 2025 conversion factors', 'default'),
('a0000001-0001-4000-8000-000000000003', 1, 'create', '{}', 'c3c3c3c3d4d4d4d4e5e5e5e5f6f6f6f6a7a7a7a7b8b8b8b8c9c9c9c9d0d0d0d0', 'system', 'Initial seed: EPA eGRID 2024', 'default'),
('a0000001-0001-4000-8000-000000000004', 1, 'create', '{}', 'd4d4d4d4e5e5e5e5f6f6f6f6a7a7a7a7b8b8b8b8c9c9c9c9d0d0d0d0e1e1e1e1', 'system', 'Initial seed: EPA AP-42 5th Edition', 'default'),
('a0000001-0001-4000-8000-000000000005', 1, 'create', '{}', 'e5e5e5e5f6f6f6f6a7a7a7a7b8b8b8b8c9c9c9c9d0d0d0d0e1e1e1e1f2f2f2f2', 'system', 'Initial seed: EPA 40 CFR Part 98', 'default'),
('a0000001-0001-4000-8000-000000000006', 1, 'create', '{}', 'f6f6f6f6a7a7a7a7b8b8b8b8c9c9c9c9d0d0d0d0e1e1e1e1f2f2f2f2a3a3a3a3', 'system', 'Initial seed: IPCC AR6 WG1', 'default'),
('a0000001-0001-4000-8000-000000000007', 1, 'create', '{}', 'a7a7a7a7b8b8b8b8c9c9c9c9d0d0d0d0e1e1e1e1f2f2f2f2a3a3a3a3b4b4b4b4', 'system', 'Initial seed: IPCC 2019 Refinement', 'default'),
('a0000001-0001-4000-8000-000000000008', 1, 'create', '{}', 'b8b8b8b8c9c9c9c9d0d0d0d0e1e1e1e1f2f2f2f2a3a3a3a3b4b4b4b4c5c5c5c5', 'system', 'Initial seed: GHG Protocol Corporate Standard', 'default'),
('a0000001-0001-4000-8000-000000000009', 1, 'create', '{}', 'c9c9c9c9d0d0d0d0e1e1e1e1f2f2f2f2a3a3a3a3b4b4b4b4c5c5c5c5d6d6d6d6', 'system', 'Initial seed: GHG Protocol Scope 3 Standard', 'default'),
('a0000001-0001-4000-8000-000000000010', 1, 'create', '{}', 'd0d0d0d0e1e1e1e1f2f2f2f2a3a3a3a3b4b4b4b4c5c5c5c5d6d6d6d6e7e7e7e7', 'system', 'Initial seed: GHG Protocol Scope 2 Guidance', 'default'),
('a0000001-0001-4000-8000-000000000011', 1, 'create', '{}', 'e1e1e1e1f2f2f2f2a3a3a3a3b4b4b4b4c5c5c5c5d6d6d6d6e7e7e7e7f8f8f8f8', 'system', 'Initial seed: ISO 14064-1:2018', 'default'),
('a0000001-0001-4000-8000-000000000012', 1, 'create', '{}', 'f2f2f2f2a3a3a3a3b4b4b4b4c5c5c5c5d6d6d6d6e7e7e7e7f8f8f8f8a9a9a9a9', 'system', 'Initial seed: ISO 14064-3:2019', 'default'),
('a0000001-0001-4000-8000-000000000013', 1, 'create', '{}', 'a3a3a3a3b4b4b4b4c5c5c5c5d6d6d6d6e7e7e7e7f8f8f8f8a9a9a9a9b0b0b0b0', 'system', 'Initial seed: ecoinvent 3.10', 'default'),
('a0000001-0001-4000-8000-000000000014', 1, 'create', '{}', 'b4b4b4b4c5c5c5c5d6d6d6d6e7e7e7e7f8f8f8f8a9a9a9a9b0b0b0b0c1c1c1c1', 'system', 'Initial seed: IEA Emissions 2024', 'default'),
('a0000001-0001-4000-8000-000000000015', 1, 'create', '{}', 'c5c5c5c5d6d6d6d6e7e7e7e7f8f8f8f8a9a9a9a9b0b0b0b0c1c1c1c1d2d2d2d2', 'system', 'Initial seed: EEA Emission Factors', 'default'),
('a0000001-0001-4000-8000-000000000016', 1, 'create', '{}', 'd6d6d6d6e7e7e7e7f8f8f8f8a9a9a9a9b0b0b0b0c1c1c1c1d2d2d2d2e3e3e3e3', 'system', 'Initial seed: CSRD Directive', 'default'),
('a0000001-0001-4000-8000-000000000017', 1, 'create', '{}', 'e7e7e7e7f8f8f8f8a9a9a9a9b0b0b0b0c1c1c1c1d2d2d2d2e3e3e3e3f4f4f4f4', 'system', 'Initial seed: CBAM Regulation', 'default'),
('a0000001-0001-4000-8000-000000000018', 1, 'create', '{}', 'f8f8f8f8a9a9a9a9b0b0b0b0c1c1c1c1d2d2d2d2e3e3e3e3f4f4f4f4a5a5a5a5', 'system', 'Initial seed: SBTi Net-Zero Standard', 'default'),
('a0000001-0001-4000-8000-000000000019', 1, 'create', '{}', 'a9a9a9a9b0b0b0b0c1c1c1c1d2d2d2d2e3e3e3e3f4f4f4f4a5a5a5a5b6b6b6b6', 'system', 'Initial seed: CDP Climate 2024', 'default'),
('a0000001-0001-4000-8000-000000000020', 1, 'create', '{}', 'b0b0b0b0c1c1c1c1d2d2d2d2e3e3e3e3f4f4f4f4a5a5a5a5b6b6b6b6c7c7c7c7', 'system', 'Initial seed: UNFCCC Inventories 2024', 'default'),
('a0000001-0001-4000-8000-000000000021', 1, 'create', '{}', 'c1c1c1c1d2d2d2d2e3e3e3e3f4f4f4f4a5a5a5a5b6b6b6b6c7c7c7c7d8d8d8d8', 'system', 'Initial seed: DEFRA 2023 (superseded)', 'default'),
('a0000001-0001-4000-8000-000000000022', 1, 'create', '{}', 'd2d2d2d2e3e3e3e3f4f4f4f4a5a5a5a5b6b6b6b6c7c7c7c7d8d8d8d8e9e9e9e9', 'system', 'Initial seed: GRI Standards 2021', 'default');

-- =============================================================================
-- Seed: Methodology References (3 defaults)
-- =============================================================================

INSERT INTO citations_service.methodology_references (reference_id, name, standard, version, section, description, citation_id, scope_1_applicable, scope_2_applicable, scope_3_applicable, applicable_categories, formula_id, tenant_id) VALUES

('meth-ghg-corporate', 'GHG Protocol Corporate Accounting Standard', 'GHG Protocol', 'Revised Edition', 'Chapters 4-8', 'The GHG Protocol Corporate Accounting and Reporting Standard provides requirements and guidance for companies preparing a corporate-level GHG emissions inventory covering Scope 1, 2, and 3 emissions.', 'a0000001-0001-4000-8000-000000000008', true, true, true, '{"stationary_combustion", "mobile_combustion", "process_emissions", "fugitive_emissions", "purchased_electricity", "purchased_heat"}', NULL, 'default'),

('meth-ghg-scope3', 'GHG Protocol Scope 3 Standard', 'GHG Protocol', '1.0', 'Chapters 5-7', 'The Corporate Value Chain (Scope 3) Standard provides requirements and guidance for companies to assess their entire value chain emissions. Covers 15 categories of upstream and downstream activities.', 'a0000001-0001-4000-8000-000000000009', false, false, true, '{"cat1_purchased_goods", "cat2_capital_goods", "cat3_fuel_energy", "cat4_upstream_transport", "cat5_waste", "cat6_business_travel", "cat7_employee_commuting", "cat8_upstream_leased", "cat9_downstream_transport", "cat10_processing", "cat11_use_of_sold", "cat12_eol_treatment", "cat13_downstream_leased", "cat14_franchises", "cat15_investments"}', NULL, 'default'),

('meth-iso-14064-1', 'ISO 14064-1 Quantification and Reporting', 'ISO 14064', '2018', 'Clauses 5-9', 'ISO 14064-1:2018 specifies principles and requirements for designing, developing, managing, reporting, and verifying an organization-level GHG inventory. Includes direct and indirect emission categories.', 'a0000001-0001-4000-8000-000000000011', true, true, true, '{"direct_emissions", "indirect_energy", "indirect_transport", "indirect_products", "indirect_other"}', NULL, 'default')

ON CONFLICT (reference_id) DO NOTHING;

-- =============================================================================
-- Seed: Regulatory Mappings (CSRD, CBAM)
-- =============================================================================

INSERT INTO citations_service.regulatory_mappings (framework, article, requirement_text, citation_id, effective_date, compliance_deadline, applies_to_scope_1, applies_to_scope_2, applies_to_scope_3, compliance_status, tenant_id) VALUES

-- CSRD Requirements
('csrd', 'ESRS E1-6', 'Disclose gross Scope 1, 2, and 3 GHG emissions in metric tonnes of CO2 equivalent, using GHG Protocol methodologies and IPCC AR5/AR6 GWP values.', 'a0000001-0001-4000-8000-000000000016', '2024-01-01', '2025-12-31', true, true, true, 'in_progress', 'default'),
('csrd', 'ESRS E1-4', 'Disclose climate change transition plan including targets, actions, and resources for GHG emission reduction aligned with limiting global warming to 1.5C.', 'a0000001-0001-4000-8000-000000000016', '2024-01-01', '2025-12-31', true, true, true, 'in_progress', 'default'),
('csrd', 'ESRS E1-5', 'Disclose energy consumption from non-renewable and renewable sources, energy intensity, and energy efficiency measures.', 'a0000001-0001-4000-8000-000000000016', '2024-01-01', '2025-12-31', true, true, false, 'in_progress', 'default'),
('csrd', 'ESRS E1-9', 'Disclose potential financial effects from material physical and transition risks related to climate change.', 'a0000001-0001-4000-8000-000000000016', '2024-01-01', '2025-12-31', true, true, true, 'not_started', 'default'),

-- CBAM Requirements
('cbam', 'Article 35', 'CBAM certificates shall be purchased based on the embedded emissions of imported goods, calculated using actual or default emission factors for iron, steel, cement, aluminium, fertilizers, electricity, and hydrogen.', 'a0000001-0001-4000-8000-000000000017', '2023-10-01', '2026-01-01', true, true, false, 'in_progress', 'default'),
('cbam', 'Article 7', 'During the transitional period (Oct 2023 - Dec 2025), importers must report the embedded emissions of imported goods on a quarterly basis using actual emission data or default values.', 'a0000001-0001-4000-8000-000000000017', '2023-10-01', '2025-12-31', true, true, false, 'in_progress', 'default'),
('cbam', 'Article 10', 'Embedded emissions shall include direct emissions from the production process and indirect emissions from electricity consumed during production, using verified emission factors.', 'a0000001-0001-4000-8000-000000000017', '2023-10-01', '2026-01-01', true, true, false, 'in_progress', 'default')

ON CONFLICT (mapping_id) DO NOTHING;

-- =============================================================================
-- Seed: Data Source Attributions
-- =============================================================================

INSERT INTO citations_service.data_source_attributions (source_authority, dataset_name, dataset_version, citation_id, extracted_values, extracted_by, valid_from, valid_until, tenant_id) VALUES
('defra', 'DEFRA GHG Conversion Factors', '2024', 'a0000001-0001-4000-8000-000000000001', '{"diesel_co2_kgL": 2.68, "gasoline_co2_kgL": 2.31, "natural_gas_co2_m3": 1.89, "grid_uk_kgCO2e_kWh": 0.207}', 'system', '2024-06-01', '2025-05-31', 'default'),
('epa', 'EPA eGRID', '2024', 'a0000001-0001-4000-8000-000000000003', '{"grid_us_avg_lbCO2e_MWh": 852.3, "grid_us_avg_kgCO2e_kWh": 0.386}', 'system', '2024-01-31', NULL, 'default'),
('ipcc', 'IPCC AR6 GWP Values', 'AR6 2021', 'a0000001-0001-4000-8000-000000000006', '{"gwp_ch4_100yr": 27.9, "gwp_n2o_100yr": 273, "gwp_sf6_100yr": 25200, "gwp_hfc134a_100yr": 1530}', 'system', '2021-08-09', NULL, 'default'),
('ecoinvent', 'ecoinvent LCI Database', '3.10', 'a0000001-0001-4000-8000-000000000013', '{"datasets": 21238, "geographies": 144, "flows": 4700}', 'system', '2023-12-01', NULL, 'default'),
('defra', 'DEFRA GHG Conversion Factors', '2025', 'a0000001-0001-4000-8000-000000000002', '{"diesel_co2_kgL": 2.69, "gasoline_co2_kgL": 2.32, "natural_gas_co2_m3": 1.89, "grid_uk_kgCO2e_kWh": 0.195}', 'system', '2025-06-01', NULL, 'default')
ON CONFLICT (attribution_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA citations_service IS 'Citations & Evidence Agent service for GreenLang Climate OS (AGENT-FOUND-005) - citation registry, evidence packages, verification tracking, methodology references, regulatory mappings, data source attributions';
COMMENT ON TABLE citations_service.citations IS 'Main registry of all citations used in zero-hallucination compliance calculations with verification status, regulatory framework tags, and provenance hashes';
COMMENT ON TABLE citations_service.citation_versions IS 'TimescaleDB hypertable: immutable version history of all citation changes with provenance hashes';
COMMENT ON TABLE citations_service.evidence_packages IS 'Evidence packages bundling citations and supporting data for calculation contexts with hash-verified finalization for audit submission';
COMMENT ON TABLE citations_service.evidence_items IS 'Individual evidence items within evidence packages linking to citations with content hashes and source tracking';
COMMENT ON TABLE citations_service.citation_verifications IS 'TimescaleDB hypertable: full audit trail of citation verification events with method, status, and verifier tracking';
COMMENT ON TABLE citations_service.methodology_references IS 'Methodology references linking compliance standards and calculation methods to their authoritative citations with scope applicability';
COMMENT ON TABLE citations_service.regulatory_mappings IS 'Regulatory framework requirement mappings to authoritative citations with compliance status and deadline tracking';
COMMENT ON TABLE citations_service.data_source_attributions IS 'Data source attribution tracking linking extracted values to their authoritative source datasets with validity dates';
COMMENT ON MATERIALIZED VIEW citations_service.daily_verification_counts IS 'Continuous aggregate: daily verification event counts by status for dashboard and trend analysis';
COMMENT ON MATERIALIZED VIEW citations_service.daily_operation_counts IS 'Continuous aggregate: daily citation version operation summaries for monitoring change velocity';

COMMENT ON COLUMN citations_service.citations.citation_id IS 'Unique identifier for the citation (UUID)';
COMMENT ON COLUMN citations_service.citations.citation_type IS 'Citation type: emission_factor, regulatory, methodology, scientific, company_data, guidance, database';
COMMENT ON COLUMN citations_service.citations.source_authority IS 'Source authority: defra, epa, ecoinvent, ipcc, ghg_protocol, iso, iea, eea, etc.';
COMMENT ON COLUMN citations_service.citations.verification_status IS 'Verification status: unverified, pending, verified, rejected, expired, superseded';
COMMENT ON COLUMN citations_service.citations.content_hash IS 'SHA-256 hash of citation content for integrity verification and deduplication';
COMMENT ON COLUMN citations_service.citations.regulatory_frameworks IS 'Array of regulatory frameworks this citation supports (ghg_protocol, csrd, cbam, etc.)';
COMMENT ON COLUMN citations_service.citations.key_values IS 'JSONB of key data values extracted from this citation for quick reference';

COMMENT ON COLUMN citations_service.citation_versions.change_type IS 'Type of change: create, update, delete, verify, supersede, expire, restore, merge';
COMMENT ON COLUMN citations_service.citation_versions.current_hash IS 'SHA-256 hash of the citation content after this change';
COMMENT ON COLUMN citations_service.citation_versions.previous_hash IS 'SHA-256 hash of the citation content before this change (provenance chain)';

COMMENT ON COLUMN citations_service.evidence_packages.package_hash IS 'SHA-256 hash of the finalized evidence package for integrity verification';
COMMENT ON COLUMN citations_service.evidence_packages.is_finalized IS 'Whether the package has been finalized (immutable after finalization)';

COMMENT ON COLUMN citations_service.evidence_items.evidence_type IS 'Evidence type: emission_factor, calculation, assumption, methodology, data_source, conversion, regulatory, attestation, measurement, estimate, proxy';
COMMENT ON COLUMN citations_service.evidence_items.citation_ids IS 'Array of citation UUIDs supporting this evidence item';

COMMENT ON COLUMN citations_service.citation_verifications.verification_method IS 'Method used: manual_review, automated_check, cross_reference, url_validation, doi_lookup, registry_check, expert_review, peer_review, periodic_recheck';
