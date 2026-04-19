-- =============================================================================
-- V213: PACK-030 Net Zero Reporting Pack - Narrative & Translation Tables
-- =============================================================================
-- Pack:         PACK-030 (Net Zero Reporting Pack)
-- Migration:    003 of 015
-- Date:         March 2026
--
-- Narrative library for reusable framework-specific text blocks with citation
-- management, consistency scoring, and multi-language translation support.
--
-- Tables (2):
--   1. pack030_nz_reporting.gl_nz_narratives
--   2. pack030_nz_reporting.gl_nz_translations
--
-- Previous: V212__PACK030_framework_tables.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack030_nz_reporting.gl_nz_narratives
-- =============================================================================
-- Reusable narrative library with framework-specific text blocks, citation
-- management, consistency scoring, multi-language support, version history,
-- and usage tracking for AI-assisted narrative generation.

CREATE TABLE pack030_nz_reporting.gl_nz_narratives (
    narrative_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    -- Framework context
    framework                   VARCHAR(50)     NOT NULL,
    section_type                VARCHAR(100)    NOT NULL,
    -- Content
    title                       VARCHAR(500)    NOT NULL,
    content                     TEXT            NOT NULL,
    content_format              VARCHAR(20)     NOT NULL DEFAULT 'MARKDOWN',
    word_count                  INTEGER,
    -- Language
    language                    VARCHAR(5)      NOT NULL DEFAULT 'en',
    available_languages         JSONB           NOT NULL DEFAULT '["en"]',
    -- Citations
    citations                   JSONB           NOT NULL DEFAULT '[]',
    citation_count              INTEGER         NOT NULL DEFAULT 0,
    -- Consistency
    consistency_score           DECIMAL(5,2),
    consistency_checked_at      TIMESTAMPTZ,
    consistency_issues          JSONB           NOT NULL DEFAULT '[]',
    cross_framework_refs        JSONB           NOT NULL DEFAULT '[]',
    -- Generation
    generation_method           VARCHAR(30)     NOT NULL DEFAULT 'MANUAL',
    ai_model_used               VARCHAR(100),
    ai_prompt_template          TEXT,
    human_reviewed              BOOLEAN         NOT NULL DEFAULT FALSE,
    reviewed_by                 UUID,
    reviewed_at                 TIMESTAMPTZ,
    -- Usage
    usage_count                 INTEGER         NOT NULL DEFAULT 0,
    last_used_at                TIMESTAMPTZ,
    last_used_in_report_id      UUID,
    -- Versioning
    version_number              INTEGER         NOT NULL DEFAULT 1,
    previous_version_id         UUID,
    is_latest                   BOOLEAN         NOT NULL DEFAULT TRUE,
    -- Categorization
    narrative_category          VARCHAR(50),
    tone                        VARCHAR(30)     DEFAULT 'PROFESSIONAL',
    audience                    VARCHAR(50)     DEFAULT 'GENERAL',
    -- Status
    is_active                   BOOLEAN         NOT NULL DEFAULT TRUE,
    is_template                 BOOLEAN         NOT NULL DEFAULT FALSE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           NOT NULL DEFAULT '{}',
    tags                        JSONB           NOT NULL DEFAULT '[]',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p030_nar_framework CHECK (
        framework IN ('SBTi', 'CDP', 'TCFD', 'GRI', 'ISSB', 'SEC', 'CSRD', 'MULTI_FRAMEWORK', 'CUSTOM')
    ),
    CONSTRAINT chk_p030_nar_content_format CHECK (
        content_format IN ('MARKDOWN', 'HTML', 'PLAIN_TEXT', 'RICH_TEXT')
    ),
    CONSTRAINT chk_p030_nar_language CHECK (
        language IN ('en', 'de', 'fr', 'es', 'pt', 'it', 'nl', 'ja', 'zh')
    ),
    CONSTRAINT chk_p030_nar_generation_method CHECK (
        generation_method IN ('MANUAL', 'AI_GENERATED', 'AI_ASSISTED', 'TEMPLATE_BASED', 'TRANSLATED')
    ),
    CONSTRAINT chk_p030_nar_consistency CHECK (
        consistency_score IS NULL OR (consistency_score >= 0 AND consistency_score <= 100)
    ),
    CONSTRAINT chk_p030_nar_tone CHECK (
        tone IN ('PROFESSIONAL', 'TECHNICAL', 'EXECUTIVE', 'MARKETING', 'REGULATORY')
    ),
    CONSTRAINT chk_p030_nar_audience CHECK (
        audience IN ('GENERAL', 'INVESTOR', 'REGULATOR', 'CUSTOMER', 'EMPLOYEE', 'AUDITOR', 'BOARD')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes: gl_nz_narratives
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p030_nar_tenant              ON pack030_nz_reporting.gl_nz_narratives(tenant_id);
CREATE INDEX idx_p030_nar_org                 ON pack030_nz_reporting.gl_nz_narratives(organization_id);
CREATE INDEX idx_p030_nar_org_fw              ON pack030_nz_reporting.gl_nz_narratives(organization_id, framework);
CREATE INDEX idx_p030_nar_org_fw_section      ON pack030_nz_reporting.gl_nz_narratives(organization_id, framework, section_type);
CREATE INDEX idx_p030_nar_framework           ON pack030_nz_reporting.gl_nz_narratives(framework);
CREATE INDEX idx_p030_nar_section_type        ON pack030_nz_reporting.gl_nz_narratives(section_type);
CREATE INDEX idx_p030_nar_language            ON pack030_nz_reporting.gl_nz_narratives(language);
CREATE INDEX idx_p030_nar_consistency         ON pack030_nz_reporting.gl_nz_narratives(consistency_score);
CREATE INDEX idx_p030_nar_generation          ON pack030_nz_reporting.gl_nz_narratives(generation_method);
CREATE INDEX idx_p030_nar_usage_count         ON pack030_nz_reporting.gl_nz_narratives(usage_count DESC);
CREATE INDEX idx_p030_nar_active              ON pack030_nz_reporting.gl_nz_narratives(organization_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p030_nar_latest              ON pack030_nz_reporting.gl_nz_narratives(organization_id, framework, section_type) WHERE is_latest = TRUE;
CREATE INDEX idx_p030_nar_templates           ON pack030_nz_reporting.gl_nz_narratives(framework, section_type) WHERE is_template = TRUE;
CREATE INDEX idx_p030_nar_unreviewed          ON pack030_nz_reporting.gl_nz_narratives(organization_id) WHERE human_reviewed = FALSE AND generation_method IN ('AI_GENERATED', 'AI_ASSISTED');
CREATE INDEX idx_p030_nar_created             ON pack030_nz_reporting.gl_nz_narratives(created_at DESC);
CREATE INDEX idx_p030_nar_content_fts         ON pack030_nz_reporting.gl_nz_narratives USING GIN(to_tsvector('english', content));
CREATE INDEX idx_p030_nar_citations           ON pack030_nz_reporting.gl_nz_narratives USING GIN(citations);
CREATE INDEX idx_p030_nar_cross_fw_refs       ON pack030_nz_reporting.gl_nz_narratives USING GIN(cross_framework_refs);
CREATE INDEX idx_p030_nar_metadata            ON pack030_nz_reporting.gl_nz_narratives USING GIN(metadata);
CREATE INDEX idx_p030_nar_tags                ON pack030_nz_reporting.gl_nz_narratives USING GIN(tags);

-- ---------------------------------------------------------------------------
-- Trigger: gl_nz_narratives
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p030_narratives_updated
    BEFORE UPDATE ON pack030_nz_reporting.gl_nz_narratives
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security: gl_nz_narratives
-- ---------------------------------------------------------------------------
ALTER TABLE pack030_nz_reporting.gl_nz_narratives ENABLE ROW LEVEL SECURITY;

CREATE POLICY p030_nar_tenant_isolation
    ON pack030_nz_reporting.gl_nz_narratives
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p030_nar_service_bypass
    ON pack030_nz_reporting.gl_nz_narratives
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- =============================================================================
-- Table 2: pack030_nz_reporting.gl_nz_translations
-- =============================================================================
-- Multi-language translation records with source/target text, quality scoring,
-- translator identification, terminology consistency tracking, and citation
-- preservation verification.

CREATE TABLE pack030_nz_reporting.gl_nz_translations (
    translation_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    -- Source reference
    source_type                 VARCHAR(30)     NOT NULL,
    source_id                   UUID            NOT NULL,
    -- Source content
    source_text                 TEXT            NOT NULL,
    source_language             VARCHAR(5)      NOT NULL,
    source_word_count           INTEGER,
    -- Target content
    target_language             VARCHAR(5)      NOT NULL,
    translated_text             TEXT            NOT NULL,
    target_word_count           INTEGER,
    -- Quality
    quality_score               DECIMAL(5,2),
    fluency_score               DECIMAL(5,2),
    accuracy_score              DECIMAL(5,2),
    terminology_score           DECIMAL(5,2),
    -- Translator
    translator_type             VARCHAR(30)     NOT NULL,
    translator_service          VARCHAR(100),
    translator_model            VARCHAR(100),
    translator_name             VARCHAR(255),
    -- Climate terminology
    glossary_terms_used         JSONB           NOT NULL DEFAULT '[]',
    terminology_consistent      BOOLEAN,
    -- Citation preservation
    citations_preserved         BOOLEAN         NOT NULL DEFAULT TRUE,
    citation_verification       JSONB           NOT NULL DEFAULT '{}',
    -- Review
    human_reviewed              BOOLEAN         NOT NULL DEFAULT FALSE,
    reviewed_by                 UUID,
    reviewed_at                 TIMESTAMPTZ,
    review_edits_count          INTEGER         DEFAULT 0,
    -- Status
    is_active                   BOOLEAN         NOT NULL DEFAULT TRUE,
    is_approved                 BOOLEAN         NOT NULL DEFAULT FALSE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           NOT NULL DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p030_tr_source_type CHECK (
        source_type IN ('NARRATIVE', 'SECTION', 'REPORT', 'TEMPLATE', 'CUSTOM')
    ),
    CONSTRAINT chk_p030_tr_source_lang CHECK (
        source_language IN ('en', 'de', 'fr', 'es', 'pt', 'it', 'nl', 'ja', 'zh')
    ),
    CONSTRAINT chk_p030_tr_target_lang CHECK (
        target_language IN ('en', 'de', 'fr', 'es', 'pt', 'it', 'nl', 'ja', 'zh')
    ),
    CONSTRAINT chk_p030_tr_different_langs CHECK (
        source_language != target_language
    ),
    CONSTRAINT chk_p030_tr_translator_type CHECK (
        translator_type IN ('MACHINE', 'HUMAN', 'HYBRID', 'AI_ASSISTED')
    ),
    CONSTRAINT chk_p030_tr_quality_scores CHECK (
        (quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 100))
        AND (fluency_score IS NULL OR (fluency_score >= 0 AND fluency_score <= 100))
        AND (accuracy_score IS NULL OR (accuracy_score >= 0 AND accuracy_score <= 100))
        AND (terminology_score IS NULL OR (terminology_score >= 0 AND terminology_score <= 100))
    )
);

-- ---------------------------------------------------------------------------
-- Indexes: gl_nz_translations
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p030_tr_tenant               ON pack030_nz_reporting.gl_nz_translations(tenant_id);
CREATE INDEX idx_p030_tr_org                  ON pack030_nz_reporting.gl_nz_translations(organization_id);
CREATE INDEX idx_p030_tr_source               ON pack030_nz_reporting.gl_nz_translations(source_type, source_id);
CREATE INDEX idx_p030_tr_source_lang          ON pack030_nz_reporting.gl_nz_translations(source_language);
CREATE INDEX idx_p030_tr_target_lang          ON pack030_nz_reporting.gl_nz_translations(target_language);
CREATE INDEX idx_p030_tr_lang_pair            ON pack030_nz_reporting.gl_nz_translations(source_language, target_language);
CREATE INDEX idx_p030_tr_quality              ON pack030_nz_reporting.gl_nz_translations(quality_score);
CREATE INDEX idx_p030_tr_translator_type      ON pack030_nz_reporting.gl_nz_translations(translator_type);
CREATE INDEX idx_p030_tr_unreviewed           ON pack030_nz_reporting.gl_nz_translations(organization_id) WHERE human_reviewed = FALSE;
CREATE INDEX idx_p030_tr_unapproved           ON pack030_nz_reporting.gl_nz_translations(organization_id) WHERE is_approved = FALSE AND is_active = TRUE;
CREATE INDEX idx_p030_tr_active               ON pack030_nz_reporting.gl_nz_translations(organization_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p030_tr_created              ON pack030_nz_reporting.gl_nz_translations(created_at DESC);
CREATE INDEX idx_p030_tr_source_fts           ON pack030_nz_reporting.gl_nz_translations USING GIN(to_tsvector('english', source_text));
CREATE INDEX idx_p030_tr_glossary             ON pack030_nz_reporting.gl_nz_translations USING GIN(glossary_terms_used);
CREATE INDEX idx_p030_tr_metadata             ON pack030_nz_reporting.gl_nz_translations USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger: gl_nz_translations
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p030_translations_updated
    BEFORE UPDATE ON pack030_nz_reporting.gl_nz_translations
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security: gl_nz_translations
-- ---------------------------------------------------------------------------
ALTER TABLE pack030_nz_reporting.gl_nz_translations ENABLE ROW LEVEL SECURITY;

CREATE POLICY p030_tr_tenant_isolation
    ON pack030_nz_reporting.gl_nz_translations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p030_tr_service_bypass
    ON pack030_nz_reporting.gl_nz_translations
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_narratives TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_translations TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack030_nz_reporting.gl_nz_narratives IS
    'Reusable narrative library with framework-specific text blocks, AI-assisted generation, citation management, cross-framework consistency scoring, multi-language support, version history, and usage tracking.';

COMMENT ON COLUMN pack030_nz_reporting.gl_nz_narratives.narrative_id IS 'Unique narrative identifier.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_narratives.framework IS 'Target framework: SBTi, CDP, TCFD, GRI, ISSB, SEC, CSRD, MULTI_FRAMEWORK, CUSTOM.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_narratives.consistency_score IS 'Cross-framework narrative consistency score (0-100).';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_narratives.generation_method IS 'How the narrative was created: MANUAL, AI_GENERATED, AI_ASSISTED, TEMPLATE_BASED, TRANSLATED.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_narratives.provenance_hash IS 'SHA-256 hash for content integrity and audit provenance.';

COMMENT ON TABLE pack030_nz_reporting.gl_nz_translations IS
    'Multi-language translation records with source/target content, quality scoring (fluency, accuracy, terminology), translator identification, climate glossary term tracking, and citation preservation verification.';

COMMENT ON COLUMN pack030_nz_reporting.gl_nz_translations.translation_id IS 'Unique translation identifier.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_translations.quality_score IS 'Overall translation quality score (0-100).';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_translations.glossary_terms_used IS 'JSONB array of climate-specific glossary terms used in translation.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_translations.citations_preserved IS 'Whether all citation references were preserved during translation.';
