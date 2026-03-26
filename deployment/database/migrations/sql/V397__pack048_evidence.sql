-- =============================================================================
-- PACK-048 GHG Assurance Prep Pack
-- Migration: V397 - Evidence Packaging
-- =============================================================================
-- Pack:         PACK-048 (GHG Assurance Prep Pack)
-- Migration:    002 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates evidence packaging tables for organising, versioning, and quality-
-- scoring audit evidence. Evidence packages group related items by scope and
-- category. Individual evidence items are quality-graded (EXCELLENT through
-- INSUFFICIENT) with file hashing for integrity. Evidence links create
-- cross-references between evidence items and any external entities (e.g.,
-- emission records, calculation runs, data sources).
--
-- Tables (3):
--   1. ghg_assurance.gl_ap_evidence_packages
--   2. ghg_assurance.gl_ap_evidence_items
--   3. ghg_assurance.gl_ap_evidence_links
--
-- Also includes: indexes, RLS, comments.
-- Previous: V396__pack048_core_schema.sql
-- =============================================================================

SET search_path TO ghg_assurance, public;

-- =============================================================================
-- Table 1: ghg_assurance.gl_ap_evidence_packages
-- =============================================================================
-- Versioned evidence packages that aggregate evidence items for a specific
-- assurance engagement. Each package tracks scope coverage, completeness
-- scoring, total item count, and quality distribution across grades.
-- Packages progress through DRAFT, REVIEW, and FINAL stages.

CREATE TABLE ghg_assurance.gl_ap_evidence_packages (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_configurations(id) ON DELETE CASCADE,
    engagement_id               UUID            REFERENCES ghg_assurance.gl_ap_engagements(id) ON DELETE SET NULL,
    package_name                VARCHAR(255)    NOT NULL,
    package_version             VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    scope_coverage              JSONB           NOT NULL DEFAULT '[]',
    completeness_score          NUMERIC(5,2),
    total_items                 INTEGER         NOT NULL DEFAULT 0,
    quality_distribution        JSONB           DEFAULT '{}',
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    generated_at                TIMESTAMPTZ,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p048_ep_version CHECK (
        package_version IN ('DRAFT', 'REVIEW', 'FINAL')
    ),
    CONSTRAINT chk_p048_ep_completeness CHECK (
        completeness_score IS NULL OR (completeness_score >= 0 AND completeness_score <= 100)
    ),
    CONSTRAINT chk_p048_ep_total_items CHECK (
        total_items >= 0
    ),
    CONSTRAINT uq_p048_ep_config_name_ver UNIQUE (config_id, package_name, package_version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_ep_tenant            ON ghg_assurance.gl_ap_evidence_packages(tenant_id);
CREATE INDEX idx_p048_ep_config            ON ghg_assurance.gl_ap_evidence_packages(config_id);
CREATE INDEX idx_p048_ep_engagement        ON ghg_assurance.gl_ap_evidence_packages(engagement_id);
CREATE INDEX idx_p048_ep_version           ON ghg_assurance.gl_ap_evidence_packages(package_version);
CREATE INDEX idx_p048_ep_completeness      ON ghg_assurance.gl_ap_evidence_packages(completeness_score);
CREATE INDEX idx_p048_ep_generated         ON ghg_assurance.gl_ap_evidence_packages(generated_at DESC);
CREATE INDEX idx_p048_ep_created           ON ghg_assurance.gl_ap_evidence_packages(created_at DESC);
CREATE INDEX idx_p048_ep_scope             ON ghg_assurance.gl_ap_evidence_packages USING GIN(scope_coverage);
CREATE INDEX idx_p048_ep_quality           ON ghg_assurance.gl_ap_evidence_packages USING GIN(quality_distribution);
CREATE INDEX idx_p048_ep_metadata          ON ghg_assurance.gl_ap_evidence_packages USING GIN(metadata);

-- Composite: tenant + config for listing
CREATE INDEX idx_p048_ep_tenant_config     ON ghg_assurance.gl_ap_evidence_packages(tenant_id, config_id);

-- Composite: config + version for lifecycle queries
CREATE INDEX idx_p048_ep_config_version    ON ghg_assurance.gl_ap_evidence_packages(config_id, package_version);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_ep_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_evidence_packages
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_assurance.gl_ap_evidence_items
-- =============================================================================
-- Individual evidence items within an evidence package. Each item is
-- classified by emission scope and evidence category, quality-graded,
-- and integrity-hashed. Source references link back to original data
-- sources, calculations, or external documents. Data quality scoring
-- provides a numeric assessment (0-5) for aggregation.

CREATE TABLE ghg_assurance.gl_ap_evidence_items (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    package_id                  UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_evidence_packages(id) ON DELETE CASCADE,
    scope                       VARCHAR(20)     NOT NULL,
    category                    VARCHAR(30)     NOT NULL,
    item_name                   VARCHAR(255)    NOT NULL,
    item_description            TEXT,
    source_reference            TEXT,
    quality_grade               VARCHAR(20)     NOT NULL DEFAULT 'ADEQUATE',
    file_hash                   TEXT,
    file_path                   TEXT,
    data_quality_score          NUMERIC(3,1),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p048_ei_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'CROSS_SCOPE')
    ),
    CONSTRAINT chk_p048_ei_category CHECK (
        category IN (
            'SOURCE_DATA', 'EMISSION_FACTOR', 'CALCULATION',
            'ASSUMPTION', 'METHODOLOGY', 'BOUNDARY',
            'COMPLETENESS', 'CONTROL', 'APPROVAL', 'EXTERNAL'
        )
    ),
    CONSTRAINT chk_p048_ei_grade CHECK (
        quality_grade IN (
            'EXCELLENT', 'GOOD', 'ADEQUATE',
            'MARGINAL', 'INSUFFICIENT'
        )
    ),
    CONSTRAINT chk_p048_ei_dq CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 5)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_ei_tenant            ON ghg_assurance.gl_ap_evidence_items(tenant_id);
CREATE INDEX idx_p048_ei_package           ON ghg_assurance.gl_ap_evidence_items(package_id);
CREATE INDEX idx_p048_ei_scope             ON ghg_assurance.gl_ap_evidence_items(scope);
CREATE INDEX idx_p048_ei_category          ON ghg_assurance.gl_ap_evidence_items(category);
CREATE INDEX idx_p048_ei_grade             ON ghg_assurance.gl_ap_evidence_items(quality_grade);
CREATE INDEX idx_p048_ei_dq               ON ghg_assurance.gl_ap_evidence_items(data_quality_score);
CREATE INDEX idx_p048_ei_file_hash         ON ghg_assurance.gl_ap_evidence_items(file_hash);
CREATE INDEX idx_p048_ei_created           ON ghg_assurance.gl_ap_evidence_items(created_at DESC);
CREATE INDEX idx_p048_ei_metadata          ON ghg_assurance.gl_ap_evidence_items USING GIN(metadata);

-- Composite: package + scope for scope-filtered retrieval
CREATE INDEX idx_p048_ei_pkg_scope         ON ghg_assurance.gl_ap_evidence_items(package_id, scope);

-- Composite: package + category for category-filtered retrieval
CREATE INDEX idx_p048_ei_pkg_category      ON ghg_assurance.gl_ap_evidence_items(package_id, category);

-- Composite: scope + category for cross-package analysis
CREATE INDEX idx_p048_ei_scope_category    ON ghg_assurance.gl_ap_evidence_items(scope, category);

-- =============================================================================
-- Table 3: ghg_assurance.gl_ap_evidence_links
-- =============================================================================
-- Cross-references between evidence items and external entities. Enables
-- traceability from evidence back to source emission records, calculation
-- runs, data quality assessments, or any other entity in the GreenLang
-- platform. The linked_entity_type + linked_entity_id pattern allows
-- polymorphic referencing without rigid FK constraints.

CREATE TABLE ghg_assurance.gl_ap_evidence_links (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    evidence_item_id            UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_evidence_items(id) ON DELETE CASCADE,
    linked_entity_type          VARCHAR(100)    NOT NULL,
    linked_entity_id            UUID            NOT NULL,
    link_description            TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_el_tenant            ON ghg_assurance.gl_ap_evidence_links(tenant_id);
CREATE INDEX idx_p048_el_evidence          ON ghg_assurance.gl_ap_evidence_links(evidence_item_id);
CREATE INDEX idx_p048_el_entity_type       ON ghg_assurance.gl_ap_evidence_links(linked_entity_type);
CREATE INDEX idx_p048_el_entity_id         ON ghg_assurance.gl_ap_evidence_links(linked_entity_id);
CREATE INDEX idx_p048_el_created           ON ghg_assurance.gl_ap_evidence_links(created_at DESC);
CREATE INDEX idx_p048_el_metadata          ON ghg_assurance.gl_ap_evidence_links USING GIN(metadata);

-- Composite: entity type + entity id for reverse lookups
CREATE INDEX idx_p048_el_entity            ON ghg_assurance.gl_ap_evidence_links(linked_entity_type, linked_entity_id);

-- Composite: evidence item + entity type for typed link queries
CREATE INDEX idx_p048_el_item_type         ON ghg_assurance.gl_ap_evidence_links(evidence_item_id, linked_entity_type);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_assurance.gl_ap_evidence_packages ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_assurance.gl_ap_evidence_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_assurance.gl_ap_evidence_links ENABLE ROW LEVEL SECURITY;

CREATE POLICY p048_ep_tenant_isolation
    ON ghg_assurance.gl_ap_evidence_packages
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_ep_service_bypass
    ON ghg_assurance.gl_ap_evidence_packages
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p048_ei_tenant_isolation
    ON ghg_assurance.gl_ap_evidence_items
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_ei_service_bypass
    ON ghg_assurance.gl_ap_evidence_items
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p048_el_tenant_isolation
    ON ghg_assurance.gl_ap_evidence_links
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_el_service_bypass
    ON ghg_assurance.gl_ap_evidence_links
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_evidence_packages TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_evidence_items TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_evidence_links TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_assurance.gl_ap_evidence_packages IS
    'Versioned evidence packages aggregating evidence items for assurance engagements with completeness scoring and quality distribution.';
COMMENT ON TABLE ghg_assurance.gl_ap_evidence_items IS
    'Individual evidence items classified by scope and category with quality grading, file hashing, and data quality scoring.';
COMMENT ON TABLE ghg_assurance.gl_ap_evidence_links IS
    'Cross-references between evidence items and external platform entities for full traceability.';

COMMENT ON COLUMN ghg_assurance.gl_ap_evidence_packages.package_version IS 'Evidence package lifecycle stage: DRAFT (collection), REVIEW (QA), FINAL (locked for verifier).';
COMMENT ON COLUMN ghg_assurance.gl_ap_evidence_packages.scope_coverage IS 'JSON array of scopes covered by this package, e.g. ["SCOPE_1","SCOPE_2"].';
COMMENT ON COLUMN ghg_assurance.gl_ap_evidence_packages.completeness_score IS 'Percentage completeness of evidence (0-100). Calculated from item coverage vs requirements.';
COMMENT ON COLUMN ghg_assurance.gl_ap_evidence_packages.quality_distribution IS 'JSON object mapping quality grades to counts, e.g. {"EXCELLENT":5,"GOOD":12,"ADEQUATE":8}.';
COMMENT ON COLUMN ghg_assurance.gl_ap_evidence_items.scope IS 'Emission scope: SCOPE_1, SCOPE_2, SCOPE_3, or CROSS_SCOPE for items spanning multiple scopes.';
COMMENT ON COLUMN ghg_assurance.gl_ap_evidence_items.category IS 'Evidence category: SOURCE_DATA, EMISSION_FACTOR, CALCULATION, ASSUMPTION, METHODOLOGY, BOUNDARY, COMPLETENESS, CONTROL, APPROVAL, EXTERNAL.';
COMMENT ON COLUMN ghg_assurance.gl_ap_evidence_items.quality_grade IS 'Quality assessment: EXCELLENT (verified, complete), GOOD (reliable), ADEQUATE (acceptable), MARGINAL (improvement needed), INSUFFICIENT (major gaps).';
COMMENT ON COLUMN ghg_assurance.gl_ap_evidence_items.file_hash IS 'SHA-256 hash of the evidence file for integrity verification.';
COMMENT ON COLUMN ghg_assurance.gl_ap_evidence_items.data_quality_score IS 'Numeric quality score (0-5) for aggregation: 5=verified, 4=reliable, 3=adequate, 2=marginal, 1=insufficient, 0=missing.';
COMMENT ON COLUMN ghg_assurance.gl_ap_evidence_links.linked_entity_type IS 'Type of linked entity, e.g. emission_record, calculation_run, data_source, control_test.';
COMMENT ON COLUMN ghg_assurance.gl_ap_evidence_links.linked_entity_id IS 'UUID of the linked entity in the referenced table.';
