-- ============================================================================
-- V110: AGENT-EUDR-022 Protected Area Validator Agent
-- ============================================================================
-- Creates tables for protected area registry (WDPA), plot-protected area
-- overlap analysis, buffer zone violation tracking, designation validation,
-- risk assessments, violation alerts, compliance reports, PADDD event
-- tracking, IUCN category mappings, country buffer regulations, plot
-- associations, and time-series hypertables for overlap analysis logs,
-- buffer zone monitoring logs, violation events, and audit trails.
--
-- Schema: eudr_protected_area_validator (15 tables)
-- Tables: 15 (11 regular + 4 hypertables)
-- Hypertables: gl_eudr_pav_overlap_analysis_log (30d chunks),
--              gl_eudr_pav_buffer_zone_monitoring_log (30d chunks),
--              gl_eudr_pav_violation_events (30d chunks),
--              gl_eudr_pav_audit_log (30d chunks)
-- Continuous Aggregates: 3 (daily_overlap_by_iucn, monthly_violations_by_severity,
--                           weekly_paddd_by_type)
-- Retention Policies: 4 (5 years per EUDR Article 31)
-- Indexes: ~160 (B-tree, GIST, GIN, partial)
-- GIST spatial indexes: 3 (on geometry columns)
-- GIN indexes: 18 (on JSONB columns)
-- Partial indexes: 8 (for active/filtered records)
--
-- Dependencies: TimescaleDB extension (V002), PostGIS extension
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V110: Creating AGENT-EUDR-022 Protected Area Validator tables...';


-- ============================================================================
-- 1. gl_eudr_pav_protected_areas — Master protected area registry
-- ============================================================================
RAISE NOTICE 'V110 [1/15]: Creating gl_eudr_pav_protected_areas...';

CREATE TABLE IF NOT EXISTS gl_eudr_pav_protected_areas (
    area_id                     UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    wdpa_id                     INTEGER,
        -- World Database on Protected Areas unique identifier
    area_name                   VARCHAR(500)    NOT NULL,
        -- Official protected area name
    iucn_category               VARCHAR(20)     NOT NULL DEFAULT 'Not Reported',
        -- IUCN management category classification
    designation_type            VARCHAR(100)    NOT NULL,
        -- Legal designation type (National Park, Nature Reserve, etc.)
    country_code                CHAR(2)         NOT NULL,
        -- ISO 3166-1 alpha-2 country code
    geometry                    GEOMETRY(MULTIPOLYGON, 4326),
        -- PostGIS multipolygon in WGS-84 (SRID 4326)
    area_hectares               NUMERIC(12,2)   CHECK (area_hectares >= 0),
        -- Total protected area in hectares
    legal_status                VARCHAR(50)     NOT NULL DEFAULT 'Designated',
        -- Current legal status of the protected area
    designation_date            DATE,
        -- Date when the area was officially designated
    management_authority        VARCHAR(500),
        -- Organization or body responsible for management
    wdpa_status                 VARCHAR(50)     NOT NULL DEFAULT 'Designated',
        -- WDPA status classification
    data_source                 VARCHAR(100),
        -- Origin dataset identifier (e.g., 'WDPA', 'UNEP-WCMC', 'national_registry')
    last_updated                DATE,
        -- Date the record was last updated from the source
    metadata                    JSONB           DEFAULT '{}',
        -- Additional protected area attributes (marine, governance, etc.)
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_pav_iucn_category CHECK (iucn_category IN (
        'Ia', 'Ib', 'II', 'III', 'IV', 'V', 'VI',
        'Not Applicable', 'Not Reported', 'Not Assigned'
    )),
    CONSTRAINT chk_pav_legal_status CHECK (legal_status IN (
        'Designated', 'Inscribed', 'Adopted', 'Established',
        'Proposed', 'Not Reported'
    )),
    CONSTRAINT chk_pav_wdpa_status CHECK (wdpa_status IN (
        'Designated', 'Inscribed', 'Adopted', 'Established',
        'Proposed', 'Not Reported'
    ))
);

-- GIST spatial index on geometry
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_pa_geometry ON gl_eudr_pav_protected_areas USING GIST (geometry);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_pa_wdpa_id ON gl_eudr_pav_protected_areas (wdpa_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_pa_name ON gl_eudr_pav_protected_areas (area_name);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_pa_iucn ON gl_eudr_pav_protected_areas (iucn_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_pa_designation ON gl_eudr_pav_protected_areas (designation_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_pa_country ON gl_eudr_pav_protected_areas (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_pa_area ON gl_eudr_pav_protected_areas (area_hectares DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_pa_legal ON gl_eudr_pav_protected_areas (legal_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_pa_wdpa_status ON gl_eudr_pav_protected_areas (wdpa_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_pa_source ON gl_eudr_pav_protected_areas (data_source);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_pa_tenant ON gl_eudr_pav_protected_areas (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_pa_created ON gl_eudr_pav_protected_areas (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_pa_country_iucn ON gl_eudr_pav_protected_areas (country_code, iucn_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for strictly protected areas (IUCN Ia, Ib, II)
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_pa_strict ON gl_eudr_pav_protected_areas (country_code, area_hectares DESC)
        WHERE iucn_category IN ('Ia', 'Ib', 'II');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_pa_metadata ON gl_eudr_pav_protected_areas USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_pav_protected_areas IS 'Master registry of protected areas from WDPA and national databases with PostGIS geometry, IUCN category, designation type, and management metadata';
COMMENT ON COLUMN gl_eudr_pav_protected_areas.geometry IS 'PostGIS MULTIPOLYGON in WGS-84 (SRID 4326) representing protected area boundaries';
COMMENT ON COLUMN gl_eudr_pav_protected_areas.iucn_category IS 'IUCN management category: Ia (Strict Nature Reserve), Ib (Wilderness Area), II (National Park), III (Natural Monument), IV (Habitat/Species Management), V (Protected Landscape), VI (Sustainable Use), Not Applicable, Not Reported, Not Assigned';
COMMENT ON COLUMN gl_eudr_pav_protected_areas.wdpa_status IS 'WDPA status: Designated (legally enacted), Inscribed (World Heritage), Adopted (Ramsar), Established, Proposed, Not Reported';


-- ============================================================================
-- 2. gl_eudr_pav_protected_area_overlaps — Plot-protected area overlap results
-- ============================================================================
RAISE NOTICE 'V110 [2/15]: Creating gl_eudr_pav_protected_area_overlaps...';

CREATE TABLE IF NOT EXISTS gl_eudr_pav_protected_area_overlaps (
    overlap_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id                     UUID            NOT NULL,
        -- Production plot being analyzed
    area_id                     UUID            NOT NULL REFERENCES gl_eudr_pav_protected_areas(area_id),
        -- Protected area that overlaps with the plot
    overlap_type                VARCHAR(50)     NOT NULL,
        -- Classification of the overlap relationship
    overlap_geometry            GEOMETRY(POLYGON, 4326),
        -- PostGIS polygon representing the actual intersection area (WGS-84)
    overlap_area_hectares       NUMERIC(12,4)   CHECK (overlap_area_hectares >= 0),
        -- Area of spatial intersection in hectares
    overlap_percentage          NUMERIC(5,2)    CHECK (overlap_percentage >= 0 AND overlap_percentage <= 100),
        -- Percentage of the plot area that overlaps with the protected area
    iucn_category               VARCHAR(20),
        -- IUCN category of the overlapping protected area (denormalized for queries)
    risk_level                  VARCHAR(20)     NOT NULL DEFAULT 'medium',
        -- Risk classification based on overlap characteristics
    risk_score                  NUMERIC(5,2)    CHECK (risk_score >= 0 AND risk_score <= 100),
        -- Numeric risk score (0-100) for ranking and prioritization
    detected_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the overlap was first detected
    resolved                    BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether the overlap has been resolved
    resolved_at                 TIMESTAMPTZ,
        -- Timestamp when the overlap was resolved
    resolution_notes            TEXT,
        -- Notes documenting how the overlap was resolved
    tenant_id                   UUID            NOT NULL,

    CONSTRAINT chk_pav_ovlp_type CHECK (overlap_type IN (
        'full_containment', 'partial_overlap', 'boundary_adjacent',
        'buffer_zone', 'corridor', 'enclave'
    )),
    CONSTRAINT chk_pav_ovlp_risk CHECK (risk_level IN (
        'critical', 'high', 'medium', 'low', 'informational'
    ))
);

-- GIST spatial index on overlap geometry
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ovlp_geometry ON gl_eudr_pav_protected_area_overlaps USING GIST (overlap_geometry);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ovlp_plot ON gl_eudr_pav_protected_area_overlaps (plot_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ovlp_area ON gl_eudr_pav_protected_area_overlaps (area_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ovlp_type ON gl_eudr_pav_protected_area_overlaps (overlap_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ovlp_iucn ON gl_eudr_pav_protected_area_overlaps (iucn_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ovlp_risk ON gl_eudr_pav_protected_area_overlaps (risk_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ovlp_score ON gl_eudr_pav_protected_area_overlaps (risk_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ovlp_detected ON gl_eudr_pav_protected_area_overlaps (detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ovlp_resolved ON gl_eudr_pav_protected_area_overlaps (resolved);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ovlp_tenant ON gl_eudr_pav_protected_area_overlaps (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ovlp_plot_area ON gl_eudr_pav_protected_area_overlaps (plot_id, area_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for unresolved overlaps requiring action
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ovlp_unresolved ON gl_eudr_pav_protected_area_overlaps (risk_level, risk_score DESC)
        WHERE resolved = FALSE;
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_pav_protected_area_overlaps IS 'Spatial overlap analysis results between production plots and protected areas with IUCN category, risk scoring, and resolution tracking';
COMMENT ON COLUMN gl_eudr_pav_protected_area_overlaps.overlap_type IS 'Overlap classification: full_containment (plot entirely within PA), partial_overlap, boundary_adjacent (within buffer), buffer_zone, corridor (transit route), enclave (PA within plot)';
COMMENT ON COLUMN gl_eudr_pav_protected_area_overlaps.risk_score IS 'Numeric risk score (0-100) derived from overlap percentage, IUCN category strictness, and designation type';


-- ============================================================================
-- 3. gl_eudr_pav_buffer_zone_violations — Buffer zone compliance violations
-- ============================================================================
RAISE NOTICE 'V110 [3/15]: Creating gl_eudr_pav_buffer_zone_violations...';

CREATE TABLE IF NOT EXISTS gl_eudr_pav_buffer_zone_violations (
    violation_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id                     UUID            NOT NULL,
        -- Production plot that violates the buffer zone
    area_id                     UUID            NOT NULL REFERENCES gl_eudr_pav_protected_areas(area_id),
        -- Protected area whose buffer zone is violated
    buffer_distance_m           NUMERIC(10,2)   NOT NULL CHECK (buffer_distance_m >= 0),
        -- Required buffer zone distance in meters
    actual_distance_m           NUMERIC(10,2)   NOT NULL CHECK (actual_distance_m >= 0),
        -- Actual measured distance from plot to protected area boundary in meters
    encroachment_area_ha        NUMERIC(12,4)   CHECK (encroachment_area_ha >= 0),
        -- Area of encroachment into buffer zone in hectares
    severity                    VARCHAR(20)     NOT NULL DEFAULT 'medium',
        -- Violation severity classification
    severity_score              NUMERIC(5,2)    CHECK (severity_score >= 0 AND severity_score <= 100),
        -- Numeric severity score for ranking (0-100)
    violation_date              DATE            NOT NULL,
        -- Date the violation was detected
    status                      VARCHAR(50)     NOT NULL DEFAULT 'open',
        -- Current violation handling status
    resolution                  TEXT,
        -- Description of the resolution or remediation applied
    resolved_at                 TIMESTAMPTZ,
        -- Timestamp when the violation was resolved
    evidence                    JSONB           DEFAULT '[]',
        -- [{ "type": "satellite_imagery", "url": "...", "date": "..." }, ...]
    metadata                    JSONB           DEFAULT '{}',
        -- Additional violation context
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_pav_bv_severity CHECK (severity IN (
        'critical', 'high', 'medium', 'low', 'informational'
    )),
    CONSTRAINT chk_pav_bv_status CHECK (status IN (
        'open', 'investigating', 'confirmed', 'remediation_in_progress',
        'resolved', 'dismissed', 'escalated'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bv_plot ON gl_eudr_pav_buffer_zone_violations (plot_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bv_area ON gl_eudr_pav_buffer_zone_violations (area_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bv_severity ON gl_eudr_pav_buffer_zone_violations (severity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bv_score ON gl_eudr_pav_buffer_zone_violations (severity_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bv_status ON gl_eudr_pav_buffer_zone_violations (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bv_date ON gl_eudr_pav_buffer_zone_violations (violation_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bv_tenant ON gl_eudr_pav_buffer_zone_violations (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bv_created ON gl_eudr_pav_buffer_zone_violations (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bv_sev_status ON gl_eudr_pav_buffer_zone_violations (severity, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bv_plot_area ON gl_eudr_pav_buffer_zone_violations (plot_id, area_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for open/active violations
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bv_active ON gl_eudr_pav_buffer_zone_violations (severity, severity_score DESC)
        WHERE status IN ('open', 'investigating', 'confirmed', 'remediation_in_progress', 'escalated');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bv_evidence ON gl_eudr_pav_buffer_zone_violations USING GIN (evidence);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bv_metadata ON gl_eudr_pav_buffer_zone_violations USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_pav_buffer_zone_violations IS 'Buffer zone compliance violations where production plots encroach on legally required buffer distances around protected areas';
COMMENT ON COLUMN gl_eudr_pav_buffer_zone_violations.buffer_distance_m IS 'Required buffer zone distance in meters as defined by national or international regulation';
COMMENT ON COLUMN gl_eudr_pav_buffer_zone_violations.actual_distance_m IS 'Actual measured distance from plot boundary to nearest protected area boundary in meters';


-- ============================================================================
-- 4. gl_eudr_pav_designation_validations — Designation status validation records
-- ============================================================================
RAISE NOTICE 'V110 [4/15]: Creating gl_eudr_pav_designation_validations...';

CREATE TABLE IF NOT EXISTS gl_eudr_pav_designation_validations (
    validation_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    area_id                     UUID            NOT NULL REFERENCES gl_eudr_pav_protected_areas(area_id),
        -- Protected area being validated
    validation_type             VARCHAR(50)     NOT NULL,
        -- Type of validation performed
    validation_date             DATE            NOT NULL,
        -- Date of the validation
    previous_status             VARCHAR(50),
        -- Status before this validation
    validated_status            VARCHAR(50)     NOT NULL,
        -- Status confirmed or updated by this validation
    validation_source           VARCHAR(100)    NOT NULL,
        -- Source used for the validation (e.g., 'WDPA_update', 'government_gazette')
    confidence_score            NUMERIC(3,2)    CHECK (confidence_score >= 0 AND confidence_score <= 1),
        -- Validation confidence (0.0 = low, 1.0 = authoritative)
    findings                    JSONB           DEFAULT '{}',
        -- { "designation_confirmed": true, "boundary_changes": false, "category_change": null }
    validator                   VARCHAR(200),
        -- User or system agent performing the validation
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for validation integrity
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_pav_dv_type CHECK (validation_type IN (
        'initial_registration', 'periodic_review', 'status_change',
        'boundary_update', 'category_reclassification', 'degazettement_check',
        'wdpa_sync', 'manual_verification'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_dv_area ON gl_eudr_pav_designation_validations (area_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_dv_type ON gl_eudr_pav_designation_validations (validation_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_dv_date ON gl_eudr_pav_designation_validations (validation_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_dv_validated ON gl_eudr_pav_designation_validations (validated_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_dv_source ON gl_eudr_pav_designation_validations (validation_source);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_dv_confidence ON gl_eudr_pav_designation_validations (confidence_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_dv_provenance ON gl_eudr_pav_designation_validations (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_dv_tenant ON gl_eudr_pav_designation_validations (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_dv_created ON gl_eudr_pav_designation_validations (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_dv_area_date ON gl_eudr_pav_designation_validations (area_id, validation_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_dv_findings ON gl_eudr_pav_designation_validations USING GIN (findings);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_pav_designation_validations IS 'Validation records for protected area designation status including periodic reviews, boundary updates, and WDPA synchronization events';
COMMENT ON COLUMN gl_eudr_pav_designation_validations.confidence_score IS 'Validation confidence: 0.0 = unverified, 1.0 = authoritative government source confirmation';


-- ============================================================================
-- 5. gl_eudr_pav_risk_assessments — Risk scoring results
-- ============================================================================
RAISE NOTICE 'V110 [5/15]: Creating gl_eudr_pav_risk_assessments...';

CREATE TABLE IF NOT EXISTS gl_eudr_pav_risk_assessments (
    assessment_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id                     UUID            NOT NULL,
        -- Production plot being assessed
    area_id                     UUID            REFERENCES gl_eudr_pav_protected_areas(area_id),
        -- Related protected area (nullable for aggregate assessments)
    assessment_type             VARCHAR(50)     NOT NULL,
        -- Type of risk assessment
    overall_risk_score          NUMERIC(5,2)    NOT NULL CHECK (overall_risk_score >= 0 AND overall_risk_score <= 100),
        -- Composite risk score (0-100)
    risk_level                  VARCHAR(20)     NOT NULL,
        -- Derived risk level
    score_components            JSONB           NOT NULL DEFAULT '{}',
        -- { "proximity_score": 85, "iucn_severity": 90, "overlap_pct": 45,
        --   "buffer_compliance": 70, "paddd_risk": 20, "designation_stability": 95 }
    contributing_factors        JSONB           DEFAULT '[]',
        -- [{ "factor": "strict_iucn_category", "weight": 0.3, "impact": "high" }, ...]
    recommendations             JSONB           DEFAULT '[]',
        -- [{ "action": "cease_operations", "priority": "urgent", "deadline": "2026-06-01" }]
    assessed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Assessment timestamp
    assessor                    VARCHAR(200),
        -- User or system agent performing the assessment
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for assessment integrity
    tenant_id                   UUID            NOT NULL,

    CONSTRAINT chk_pav_ra_type CHECK (assessment_type IN (
        'initial_screening', 'detailed_analysis', 'periodic_review',
        'incident_triggered', 'regulatory_update', 'supplier_onboarding'
    )),
    CONSTRAINT chk_pav_ra_risk CHECK (risk_level IN (
        'critical', 'high', 'medium', 'low', 'negligible'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ra_plot ON gl_eudr_pav_risk_assessments (plot_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ra_area ON gl_eudr_pav_risk_assessments (area_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ra_type ON gl_eudr_pav_risk_assessments (assessment_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ra_score ON gl_eudr_pav_risk_assessments (overall_risk_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ra_risk ON gl_eudr_pav_risk_assessments (risk_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ra_assessed ON gl_eudr_pav_risk_assessments (assessed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ra_provenance ON gl_eudr_pav_risk_assessments (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ra_tenant ON gl_eudr_pav_risk_assessments (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ra_risk_score ON gl_eudr_pav_risk_assessments (risk_level, overall_risk_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ra_plot_area ON gl_eudr_pav_risk_assessments (plot_id, area_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for high/critical risk assessments requiring action
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ra_high_risk ON gl_eudr_pav_risk_assessments (assessed_at DESC, overall_risk_score DESC)
        WHERE risk_level IN ('critical', 'high');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ra_components ON gl_eudr_pav_risk_assessments USING GIN (score_components);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ra_factors ON gl_eudr_pav_risk_assessments USING GIN (contributing_factors);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ra_recommendations ON gl_eudr_pav_risk_assessments USING GIN (recommendations);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_pav_risk_assessments IS 'Risk scoring results for plot-protected area interactions with multi-factor composite scores, contributing factors, and remediation recommendations';
COMMENT ON COLUMN gl_eudr_pav_risk_assessments.overall_risk_score IS 'Composite score (0-100) from proximity, IUCN category severity, overlap percentage, buffer compliance, PADDD risk, and designation stability';


-- ============================================================================
-- 6. gl_eudr_pav_violation_alerts — Detected violations and alerts
-- ============================================================================
RAISE NOTICE 'V110 [6/15]: Creating gl_eudr_pav_violation_alerts...';

CREATE TABLE IF NOT EXISTS gl_eudr_pav_violation_alerts (
    alert_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id                     UUID            NOT NULL,
        -- Production plot where the violation occurred
    area_id                     UUID            NOT NULL REFERENCES gl_eudr_pav_protected_areas(area_id),
        -- Affected protected area
    alert_type                  VARCHAR(50)     NOT NULL,
        -- Type of violation alert
    severity                    VARCHAR(20)     NOT NULL DEFAULT 'medium',
        -- Alert severity classification
    severity_score              NUMERIC(5,2)    CHECK (severity_score >= 0 AND severity_score <= 100),
        -- Numeric severity score for ranking
    title                       VARCHAR(500)    NOT NULL,
        -- Human-readable alert title
    description                 TEXT,
        -- Detailed alert description
    status                      VARCHAR(50)     NOT NULL DEFAULT 'open',
        -- Current alert lifecycle status
    assigned_to                 VARCHAR(200),
        -- User or team assigned to investigate
    detected_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp when the violation was detected
    resolved_at                 TIMESTAMPTZ,
        -- Timestamp when the alert was resolved
    resolution_notes            TEXT,
        -- Notes documenting the resolution
    evidence                    JSONB           DEFAULT '[]',
        -- Supporting evidence for the alert
    metadata                    JSONB           DEFAULT '{}',
        -- Additional alert context
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_pav_va_type CHECK (alert_type IN (
        'direct_overlap', 'buffer_encroachment', 'boundary_proximity',
        'designation_change', 'paddd_event', 'illegal_activity',
        'land_use_change', 'degazettement_risk'
    )),
    CONSTRAINT chk_pav_va_severity CHECK (severity IN (
        'critical', 'high', 'medium', 'low', 'informational'
    )),
    CONSTRAINT chk_pav_va_status CHECK (status IN (
        'open', 'investigating', 'confirmed', 'remediation_in_progress',
        'resolved', 'dismissed', 'escalated', 'false_positive'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_va_plot ON gl_eudr_pav_violation_alerts (plot_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_va_area ON gl_eudr_pav_violation_alerts (area_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_va_type ON gl_eudr_pav_violation_alerts (alert_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_va_severity ON gl_eudr_pav_violation_alerts (severity);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_va_score ON gl_eudr_pav_violation_alerts (severity_score DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_va_status ON gl_eudr_pav_violation_alerts (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_va_assigned ON gl_eudr_pav_violation_alerts (assigned_to);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_va_detected ON gl_eudr_pav_violation_alerts (detected_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_va_tenant ON gl_eudr_pav_violation_alerts (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_va_created ON gl_eudr_pav_violation_alerts (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_va_sev_status ON gl_eudr_pav_violation_alerts (severity, status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for unresolved alerts
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_va_unresolved ON gl_eudr_pav_violation_alerts (severity, severity_score DESC)
        WHERE status NOT IN ('resolved', 'dismissed', 'false_positive');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_va_evidence ON gl_eudr_pav_violation_alerts USING GIN (evidence);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_va_metadata ON gl_eudr_pav_violation_alerts USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_pav_violation_alerts IS 'Detected violations and alerts for protected area encroachment, buffer zone breaches, designation changes, and PADDD events';
COMMENT ON COLUMN gl_eudr_pav_violation_alerts.alert_type IS 'Alert classification: direct_overlap, buffer_encroachment, boundary_proximity, designation_change, paddd_event, illegal_activity, land_use_change, degazettement_risk';


-- ============================================================================
-- 7. gl_eudr_pav_compliance_reports — Generated compliance reports
-- ============================================================================
RAISE NOTICE 'V110 [7/15]: Creating gl_eudr_pav_compliance_reports...';

CREATE TABLE IF NOT EXISTS gl_eudr_pav_compliance_reports (
    report_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id                     UUID            NOT NULL,
        -- Production plot this report covers
    report_type                 VARCHAR(50)     NOT NULL,
        -- Type of compliance report
    compliance_status           VARCHAR(50)     NOT NULL,
        -- Overall compliance determination
    areas_checked               INTEGER         NOT NULL DEFAULT 0,
        -- Number of protected areas checked in this assessment
    overlaps_found              INTEGER         NOT NULL DEFAULT 0,
        -- Number of overlaps detected
    buffer_violations           INTEGER         NOT NULL DEFAULT 0,
        -- Number of buffer zone violations found
    findings                    JSONB           DEFAULT '[]',
        -- [{ "finding": "...", "severity": "high", "reference": "EUDR Art.3(a)" }, ...]
    recommendations             JSONB           DEFAULT '[]',
        -- [{ "action": "...", "priority": "urgent", "deadline": "2026-06-01" }, ...]
    generated_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Report generation timestamp
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for report integrity
    tenant_id                   UUID            NOT NULL,

    CONSTRAINT chk_pav_rpt_type CHECK (report_type IN (
        'pre_sourcing_assessment', 'due_diligence_report', 'annual_review',
        'incident_report', 'remediation_plan', 'audit_response',
        'regulatory_submission'
    )),
    CONSTRAINT chk_pav_rpt_compliance CHECK (compliance_status IN (
        'compliant', 'non_compliant', 'partially_compliant',
        'under_review', 'remediation_required'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_rpt_plot ON gl_eudr_pav_compliance_reports (plot_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_rpt_type ON gl_eudr_pav_compliance_reports (report_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_rpt_compliance ON gl_eudr_pav_compliance_reports (compliance_status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_rpt_generated ON gl_eudr_pav_compliance_reports (generated_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_rpt_provenance ON gl_eudr_pav_compliance_reports (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_rpt_tenant ON gl_eudr_pav_compliance_reports (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for non-compliant reports
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_rpt_non_compliant ON gl_eudr_pav_compliance_reports (generated_at DESC)
        WHERE compliance_status IN ('non_compliant', 'remediation_required');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_rpt_findings ON gl_eudr_pav_compliance_reports USING GIN (findings);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_rpt_recommendations ON gl_eudr_pav_compliance_reports USING GIN (recommendations);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_pav_compliance_reports IS 'Generated protected area compliance reports with findings, recommendations, and compliance determination for EUDR due diligence';
COMMENT ON COLUMN gl_eudr_pav_compliance_reports.compliance_status IS 'compliant (no overlaps or violations), non_compliant (overlaps with strict PAs), partially_compliant (buffer zone only), under_review, remediation_required';


-- ============================================================================
-- 8. gl_eudr_pav_paddd_events — PADDD event tracking
-- ============================================================================
RAISE NOTICE 'V110 [8/15]: Creating gl_eudr_pav_paddd_events...';

CREATE TABLE IF NOT EXISTS gl_eudr_pav_paddd_events (
    event_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    area_id                     UUID            NOT NULL REFERENCES gl_eudr_pav_protected_areas(area_id),
        -- Protected area affected by the PADDD event
    event_type                  VARCHAR(50)     NOT NULL,
        -- PADDD event classification
    event_date                  DATE            NOT NULL,
        -- Date of the PADDD event
    description                 TEXT,
        -- Detailed description of the event
    area_affected_ha            NUMERIC(12,2)   CHECK (area_affected_ha >= 0),
        -- Area affected by the PADDD event in hectares
    previous_iucn_category      VARCHAR(20),
        -- IUCN category before the event
    new_iucn_category           VARCHAR(20),
        -- IUCN category after the event (null for degazettement)
    legal_authority             VARCHAR(500),
        -- Legal authority or document authorizing the change
    rationale                   TEXT,
        -- Stated rationale for the PADDD event
    data_source                 VARCHAR(100),
        -- Source of PADDD information (e.g., 'PADDDtracker', 'government_gazette')
    risk_impact                 VARCHAR(20)     NOT NULL DEFAULT 'medium',
        -- Impact on risk assessment for affected supply chains
    metadata                    JSONB           DEFAULT '{}',
        -- Additional PADDD context
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_pav_paddd_type CHECK (event_type IN (
        'downgrading', 'downsizing', 'degazettement',
        'proposed_downgrading', 'proposed_downsizing', 'proposed_degazettement',
        'reversed', 'enacted'
    )),
    CONSTRAINT chk_pav_paddd_impact CHECK (risk_impact IN (
        'critical', 'high', 'medium', 'low', 'informational'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_paddd_area ON gl_eudr_pav_paddd_events (area_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_paddd_type ON gl_eudr_pav_paddd_events (event_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_paddd_date ON gl_eudr_pav_paddd_events (event_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_paddd_source ON gl_eudr_pav_paddd_events (data_source);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_paddd_impact ON gl_eudr_pav_paddd_events (risk_impact);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_paddd_tenant ON gl_eudr_pav_paddd_events (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_paddd_created ON gl_eudr_pav_paddd_events (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_paddd_area_date ON gl_eudr_pav_paddd_events (area_id, event_date DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_paddd_type_impact ON gl_eudr_pav_paddd_events (event_type, risk_impact);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for enacted PADDD events (active threats)
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_paddd_enacted ON gl_eudr_pav_paddd_events (event_date DESC, area_affected_ha DESC)
        WHERE event_type IN ('downgrading', 'downsizing', 'degazettement', 'enacted');
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_paddd_metadata ON gl_eudr_pav_paddd_events USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_pav_paddd_events IS 'Protected Area Downgrading, Downsizing, and Degazettement (PADDD) event tracking with impact on risk assessments and supply chain compliance';
COMMENT ON COLUMN gl_eudr_pav_paddd_events.event_type IS 'PADDD classification: downgrading (legal protection weakened), downsizing (area reduced), degazettement (legal protection fully removed), proposed_* (pending), reversed (rolled back), enacted (finalized)';


-- ============================================================================
-- 9. gl_eudr_pav_iucn_category_mappings — IUCN category metadata
-- ============================================================================
RAISE NOTICE 'V110 [9/15]: Creating gl_eudr_pav_iucn_category_mappings...';

CREATE TABLE IF NOT EXISTS gl_eudr_pav_iucn_category_mappings (
    category_code               VARCHAR(20)     PRIMARY KEY,
        -- IUCN category code (Ia, Ib, II, III, IV, V, VI, Not Applicable, Not Reported)
    category_name               VARCHAR(200)    NOT NULL,
        -- Full IUCN category name
    protection_level            VARCHAR(20)     NOT NULL,
        -- Aggregated protection strictness level
    risk_weight                 NUMERIC(3,2)    NOT NULL CHECK (risk_weight >= 0 AND risk_weight <= 1),
        -- Weight applied in risk calculations (1.0 = strictest)
    description                 TEXT,
        -- Detailed description of the category
    management_objectives       JSONB           DEFAULT '[]',
        -- [{ "objective": "Scientific research", "primary": true }, ...]
    eudr_relevance              VARCHAR(50)     NOT NULL DEFAULT 'high',
        -- Relevance of this category for EUDR compliance
    metadata                    JSONB           DEFAULT '{}',
        -- Additional category metadata
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_pav_icm_protection CHECK (protection_level IN (
        'strict', 'high', 'moderate', 'sustainable_use', 'unclassified'
    )),
    CONSTRAINT chk_pav_icm_relevance CHECK (eudr_relevance IN (
        'critical', 'high', 'medium', 'low', 'informational'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_icm_protection ON gl_eudr_pav_iucn_category_mappings (protection_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_icm_weight ON gl_eudr_pav_iucn_category_mappings (risk_weight DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_icm_relevance ON gl_eudr_pav_iucn_category_mappings (eudr_relevance);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_icm_tenant ON gl_eudr_pav_iucn_category_mappings (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_icm_objectives ON gl_eudr_pav_iucn_category_mappings USING GIN (management_objectives);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_icm_metadata ON gl_eudr_pav_iucn_category_mappings USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_pav_iucn_category_mappings IS 'IUCN management category metadata including protection levels, risk weights, management objectives, and EUDR compliance relevance';
COMMENT ON COLUMN gl_eudr_pav_iucn_category_mappings.risk_weight IS 'Risk weight for calculations: Ia=1.0, Ib=0.95, II=0.90, III=0.80, IV=0.70, V=0.50, VI=0.40, Not Applicable=0.20';


-- ============================================================================
-- 10. gl_eudr_pav_country_buffer_regulations — National buffer zone requirements
-- ============================================================================
RAISE NOTICE 'V110 [10/15]: Creating gl_eudr_pav_country_buffer_regulations...';

CREATE TABLE IF NOT EXISTS gl_eudr_pav_country_buffer_regulations (
    regulation_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code                CHAR(2)         NOT NULL,
        -- ISO 3166-1 alpha-2 country code
    iucn_category               VARCHAR(20),
        -- IUCN category this regulation applies to (null for all categories)
    designation_type            VARCHAR(100),
        -- Designation type this applies to (null for all types)
    buffer_distance_m           NUMERIC(10,2)   NOT NULL CHECK (buffer_distance_m >= 0),
        -- Required buffer zone distance in meters
    legal_reference             TEXT,
        -- Legal document or statute reference
    enforcement_level           VARCHAR(50)     NOT NULL DEFAULT 'moderate',
        -- Assessment of enforcement effectiveness
    effective_date              DATE,
        -- Date the regulation became effective
    expiry_date                 DATE,
        -- Date the regulation expires (null for indefinite)
    metadata                    JSONB           DEFAULT '{}',
        -- Additional regulatory context
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_pav_cbr_enforcement CHECK (enforcement_level IN (
        'strong', 'moderate', 'weak', 'none', 'unknown'
    ))
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_cbr_country ON gl_eudr_pav_country_buffer_regulations (country_code);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_cbr_iucn ON gl_eudr_pav_country_buffer_regulations (iucn_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_cbr_designation ON gl_eudr_pav_country_buffer_regulations (designation_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_cbr_buffer ON gl_eudr_pav_country_buffer_regulations (buffer_distance_m DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_cbr_enforcement ON gl_eudr_pav_country_buffer_regulations (enforcement_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_cbr_effective ON gl_eudr_pav_country_buffer_regulations (effective_date);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_cbr_tenant ON gl_eudr_pav_country_buffer_regulations (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_cbr_country_iucn ON gl_eudr_pav_country_buffer_regulations (country_code, iucn_category);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
-- Partial index for currently active regulations
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_cbr_active ON gl_eudr_pav_country_buffer_regulations (country_code, buffer_distance_m DESC)
        WHERE expiry_date IS NULL OR expiry_date > NOW();
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_cbr_metadata ON gl_eudr_pav_country_buffer_regulations USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_pav_country_buffer_regulations IS 'Per-country buffer zone distance requirements around protected areas by IUCN category and designation type with enforcement assessment';
COMMENT ON COLUMN gl_eudr_pav_country_buffer_regulations.buffer_distance_m IS 'Required buffer zone distance in meters; varies by country, IUCN category, and designation type';


-- ============================================================================
-- 11. gl_eudr_pav_protected_area_plot_associations — Many-to-many associations
-- ============================================================================
RAISE NOTICE 'V110 [11/15]: Creating gl_eudr_pav_protected_area_plot_associations...';

CREATE TABLE IF NOT EXISTS gl_eudr_pav_protected_area_plot_associations (
    association_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id                     UUID            NOT NULL,
        -- Production plot
    area_id                     UUID            NOT NULL REFERENCES gl_eudr_pav_protected_areas(area_id),
        -- Protected area
    association_type            VARCHAR(50)     NOT NULL,
        -- Nature of the association
    proximity_m                 NUMERIC(10,2)   CHECK (proximity_m >= 0),
        -- Distance in meters between plot boundary and protected area boundary
    intersection_geometry       GEOMETRY(POLYGON, 4326),
        -- PostGIS polygon of the spatial intersection (WGS-84), null if no overlap
    tenant_id                   UUID            NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_pav_assoc_type CHECK (association_type IN (
        'spatial_overlap', 'buffer_zone', 'adjacent', 'proximate',
        'corridor', 'watershed', 'ecological_connectivity'
    )),
    CONSTRAINT uq_pav_plot_area_type UNIQUE (plot_id, area_id, association_type)
);

-- GIST spatial index on intersection geometry
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_assoc_geometry ON gl_eudr_pav_protected_area_plot_associations USING GIST (intersection_geometry);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_assoc_plot ON gl_eudr_pav_protected_area_plot_associations (plot_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_assoc_area ON gl_eudr_pav_protected_area_plot_associations (area_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_assoc_type ON gl_eudr_pav_protected_area_plot_associations (association_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_assoc_proximity ON gl_eudr_pav_protected_area_plot_associations (proximity_m);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_assoc_tenant ON gl_eudr_pav_protected_area_plot_associations (tenant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_assoc_created ON gl_eudr_pav_protected_area_plot_associations (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_assoc_plot_area ON gl_eudr_pav_protected_area_plot_associations (plot_id, area_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_pav_protected_area_plot_associations IS 'Many-to-many linkage between production plots and protected areas with association type and proximity classification';
COMMENT ON COLUMN gl_eudr_pav_protected_area_plot_associations.association_type IS 'Association nature: spatial_overlap (geometric intersection), buffer_zone (within buffer), adjacent (boundary contact), proximate (within monitoring distance), corridor, watershed, ecological_connectivity';


-- ============================================================================
-- 12. gl_eudr_pav_overlap_analysis_log — Overlap analysis audit (hypertable)
-- ============================================================================
RAISE NOTICE 'V110 [12/15]: Creating gl_eudr_pav_overlap_analysis_log (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_pav_overlap_analysis_log (
    event_id                    UUID            DEFAULT gen_random_uuid(),
    analyzed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp of the analysis event (partition key)
    plot_id                     UUID            NOT NULL,
        -- Production plot analyzed
    areas_checked               INTEGER         NOT NULL DEFAULT 0,
        -- Number of protected areas checked in this analysis
    overlaps_found              INTEGER         NOT NULL DEFAULT 0,
        -- Number of overlaps detected
    iucn_categories_found       JSONB           DEFAULT '[]',
        -- IUCN categories of overlapping areas (e.g., ["Ia", "II", "IV"])
    analysis_method             VARCHAR(50)     NOT NULL,
        -- Spatial analysis method used
    analysis_parameters         JSONB           DEFAULT '{}',
        -- { "buffer_m": 500, "min_overlap_pct": 0.01, "crs": "EPSG:4326" }
    execution_time_ms           INTEGER,
        -- Analysis execution time in milliseconds
    result_summary              JSONB           DEFAULT '{}',
        -- { "total_overlap_ha": 45.2, "max_overlap_pct": 23.5, "risk_levels": {...} }
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for analysis integrity
    tenant_id                   UUID            NOT NULL,

    PRIMARY KEY (event_id, analyzed_at),

    CONSTRAINT chk_pav_oal_method CHECK (analysis_method IN (
        'st_intersects', 'st_overlaps', 'st_contains', 'st_within',
        'buffer_analysis', 'proximity_analysis', 'composite'
    ))
);

SELECT create_hypertable(
    'gl_eudr_pav_overlap_analysis_log',
    'analyzed_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_oal_plot ON gl_eudr_pav_overlap_analysis_log (plot_id, analyzed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_oal_method ON gl_eudr_pav_overlap_analysis_log (analysis_method, analyzed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_oal_overlaps ON gl_eudr_pav_overlap_analysis_log (overlaps_found, analyzed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_oal_exec_time ON gl_eudr_pav_overlap_analysis_log (execution_time_ms DESC, analyzed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_oal_provenance ON gl_eudr_pav_overlap_analysis_log (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_oal_tenant ON gl_eudr_pav_overlap_analysis_log (tenant_id, analyzed_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_oal_iucn ON gl_eudr_pav_overlap_analysis_log USING GIN (iucn_categories_found);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_oal_params ON gl_eudr_pav_overlap_analysis_log USING GIN (analysis_parameters);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_oal_summary ON gl_eudr_pav_overlap_analysis_log USING GIN (result_summary);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_pav_overlap_analysis_log IS 'Immutable audit log for spatial overlap analysis runs with method, parameters, execution time, IUCN categories, and result summaries';
COMMENT ON COLUMN gl_eudr_pav_overlap_analysis_log.analysis_method IS 'PostGIS spatial function used: st_intersects, st_overlaps, st_contains, st_within, buffer_analysis, proximity_analysis, composite';


-- ============================================================================
-- 13. gl_eudr_pav_buffer_zone_monitoring_log — Buffer monitoring audit (hypertable)
-- ============================================================================
RAISE NOTICE 'V110 [13/15]: Creating gl_eudr_pav_buffer_zone_monitoring_log (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_pav_buffer_zone_monitoring_log (
    event_id                    UUID            DEFAULT gen_random_uuid(),
    monitored_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Timestamp of the monitoring event (partition key)
    plot_id                     UUID            NOT NULL,
        -- Production plot being monitored
    area_id                     UUID            NOT NULL,
        -- Protected area whose buffer zone is being monitored
    buffer_distance_m           NUMERIC(10,2)   NOT NULL,
        -- Required buffer zone distance in meters
    measured_distance_m         NUMERIC(10,2)   NOT NULL,
        -- Measured distance from plot to PA boundary in meters
    is_compliant                BOOLEAN         NOT NULL,
        -- Whether the plot is within the required buffer distance
    monitoring_method           VARCHAR(50)     NOT NULL,
        -- Method used for buffer monitoring
    change_detected             BOOLEAN         NOT NULL DEFAULT FALSE,
        -- Whether a change was detected since last monitoring
    details                     JSONB           DEFAULT '{}',
        -- { "satellite_source": "sentinel2", "ndvi_change": -0.15, ... }
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for monitoring integrity
    tenant_id                   UUID            NOT NULL,

    PRIMARY KEY (event_id, monitored_at),

    CONSTRAINT chk_pav_bzml_method CHECK (monitoring_method IN (
        'satellite_analysis', 'field_survey', 'gps_measurement',
        'lidar_scan', 'drone_survey', 'automated_check', 'manual_review'
    ))
);

SELECT create_hypertable(
    'gl_eudr_pav_buffer_zone_monitoring_log',
    'monitored_at',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bzml_plot ON gl_eudr_pav_buffer_zone_monitoring_log (plot_id, monitored_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bzml_area ON gl_eudr_pav_buffer_zone_monitoring_log (area_id, monitored_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bzml_compliant ON gl_eudr_pav_buffer_zone_monitoring_log (is_compliant, monitored_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bzml_method ON gl_eudr_pav_buffer_zone_monitoring_log (monitoring_method, monitored_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bzml_change ON gl_eudr_pav_buffer_zone_monitoring_log (change_detected, monitored_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bzml_provenance ON gl_eudr_pav_buffer_zone_monitoring_log (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bzml_tenant ON gl_eudr_pav_buffer_zone_monitoring_log (tenant_id, monitored_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bzml_plot_area ON gl_eudr_pav_buffer_zone_monitoring_log (plot_id, area_id, monitored_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_bzml_details ON gl_eudr_pav_buffer_zone_monitoring_log USING GIN (details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_pav_buffer_zone_monitoring_log IS 'Immutable audit log for buffer zone monitoring events with compliance status, measured distances, and change detection';
COMMENT ON COLUMN gl_eudr_pav_buffer_zone_monitoring_log.is_compliant IS 'TRUE if measured_distance_m >= buffer_distance_m (plot respects required buffer), FALSE if encroaching';


-- ============================================================================
-- 14. gl_eudr_pav_violation_events — Violation event stream (hypertable)
-- ============================================================================
RAISE NOTICE 'V110 [14/15]: Creating gl_eudr_pav_violation_events (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_pav_violation_events (
    event_id                    UUID            DEFAULT gen_random_uuid(),
    event_timestamp             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Event timestamp (partition key)
    alert_id                    UUID            NOT NULL,
        -- Reference to the parent violation alert
    event_type                  VARCHAR(50)     NOT NULL,
        -- Type of violation lifecycle event
    actor                       VARCHAR(200)    NOT NULL,
        -- User or system agent performing the action
    previous_status             VARCHAR(50),
        -- Status before this event
    new_status                  VARCHAR(50)     NOT NULL,
        -- Status after this event
    severity                    VARCHAR(20),
        -- Severity level at the time of this event
    details                     JSONB           DEFAULT '{}',
        -- { "changed_fields": [...], "reason": "...", "attachments": [...] }
    ip_address                  INET,
        -- Source IP address of the actor
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for event integrity
    tenant_id                   UUID            NOT NULL,

    PRIMARY KEY (event_id, event_timestamp),

    CONSTRAINT chk_pav_ve_event_type CHECK (event_type IN (
        'detected', 'acknowledged', 'investigation_started',
        'evidence_added', 'severity_updated', 'escalated',
        'remediation_started', 'remediation_completed', 'resolved',
        'dismissed', 'reopened', 'comment_added'
    ))
);

SELECT create_hypertable(
    'gl_eudr_pav_violation_events',
    'event_timestamp',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ve_alert ON gl_eudr_pav_violation_events (alert_id, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ve_type ON gl_eudr_pav_violation_events (event_type, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ve_actor ON gl_eudr_pav_violation_events (actor, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ve_new_status ON gl_eudr_pav_violation_events (new_status, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ve_severity ON gl_eudr_pav_violation_events (severity, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ve_provenance ON gl_eudr_pav_violation_events (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ve_tenant ON gl_eudr_pav_violation_events (tenant_id, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_ve_details ON gl_eudr_pav_violation_events USING GIN (details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_pav_violation_events IS 'Immutable event stream for violation alert lifecycle tracking with actor, status transitions, severity, and provenance hashing';
COMMENT ON COLUMN gl_eudr_pav_violation_events.provenance_hash IS 'SHA-256 hash for immutability verification and audit integrity';


-- ============================================================================
-- 15. gl_eudr_pav_audit_log — Comprehensive audit trail (hypertable)
-- ============================================================================
RAISE NOTICE 'V110 [15/15]: Creating gl_eudr_pav_audit_log (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_pav_audit_log (
    event_id                    UUID            DEFAULT gen_random_uuid(),
    timestamp                   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
        -- Event timestamp (partition key)
    event_type                  VARCHAR(50)     NOT NULL,
        -- 'area_registered', 'area_updated', 'overlap_detected', 'overlap_resolved',
        -- 'buffer_violation_detected', 'buffer_violation_resolved',
        -- 'designation_validated', 'risk_assessed', 'alert_created',
        -- 'alert_resolved', 'paddd_event_recorded', 'compliance_report_generated',
        -- 'association_created', 'regulation_updated', 'data_refreshed'
    entity_type                 VARCHAR(50)     NOT NULL,
        -- 'protected_area', 'overlap', 'buffer_violation', 'designation_validation',
        -- 'risk_assessment', 'violation_alert', 'compliance_report',
        -- 'paddd_event', 'iucn_mapping', 'buffer_regulation', 'plot_association'
    entity_id                   VARCHAR(100)    NOT NULL,
        -- UUID of the entity being audited
    actor                       VARCHAR(100)    NOT NULL,
        -- User ID or system agent identifier
    details                     JSONB,
        -- { "changed_fields": [...], "old_values": {...}, "new_values": {...} }
    ip_address                  INET,
        -- Source IP address of the actor
    user_agent                  TEXT,
        -- HTTP user agent or system agent name
    provenance_hash             VARCHAR(64),
        -- SHA-256 hash for immutability verification
    metadata                    JSONB           DEFAULT '{}',
    tenant_id                   UUID            NOT NULL,

    PRIMARY KEY (event_id, timestamp)
);

SELECT create_hypertable(
    'gl_eudr_pav_audit_log',
    'timestamp',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_audit_event_type ON gl_eudr_pav_audit_log (event_type, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_audit_entity_type ON gl_eudr_pav_audit_log (entity_type, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_audit_entity_id ON gl_eudr_pav_audit_log (entity_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_audit_actor ON gl_eudr_pav_audit_log (actor, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_audit_provenance ON gl_eudr_pav_audit_log (provenance_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_audit_tenant ON gl_eudr_pav_audit_log (tenant_id, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_audit_entity_action ON gl_eudr_pav_audit_log (entity_type, event_type, timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_audit_details ON gl_eudr_pav_audit_log USING GIN (details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_pav_audit_metadata ON gl_eudr_pav_audit_log USING GIN (metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;

COMMENT ON TABLE gl_eudr_pav_audit_log IS 'Comprehensive audit trail for all protected area validator operations and state changes';
COMMENT ON COLUMN gl_eudr_pav_audit_log.provenance_hash IS 'SHA-256 hash for immutability verification and audit integrity';


-- ============================================================================
-- CONTINUOUS AGGREGATES
-- ============================================================================

-- Daily overlap counts by IUCN category
RAISE NOTICE 'V110: Creating continuous aggregate: daily_overlap_by_iucn...';

DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_pav_daily_overlap_by_iucn
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 day', analyzed_at)       AS day,
        tenant_id,
        COUNT(*)                                AS analysis_count,
        SUM(overlaps_found)                     AS total_overlaps_found,
        AVG(overlaps_found)                     AS avg_overlaps_per_analysis,
        AVG(execution_time_ms)                  AS avg_execution_time_ms,
        MAX(overlaps_found)                     AS max_overlaps_found
    FROM gl_eudr_pav_overlap_analysis_log
    GROUP BY day, tenant_id;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_pav_daily_overlap_by_iucn',
        start_offset => INTERVAL '3 days',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_pav_daily_overlap_by_iucn IS 'Daily rollup of overlap analysis runs with total overlaps found, average per analysis, and execution time statistics';


-- Monthly violation counts by severity
RAISE NOTICE 'V110: Creating continuous aggregate: monthly_violations_by_severity...';

DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_pav_monthly_violations_by_severity
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('30 days', event_timestamp) AS month,
        tenant_id,
        new_status,
        severity,
        COUNT(*)                                AS event_count,
        COUNT(DISTINCT alert_id)                AS unique_alerts
    FROM gl_eudr_pav_violation_events
    GROUP BY month, tenant_id, new_status, severity;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_pav_monthly_violations_by_severity',
        start_offset => INTERVAL '60 days',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_pav_monthly_violations_by_severity IS 'Monthly rollup of violation events by status and severity with event counts and unique alert counts';


-- Weekly PADDD events by event type
RAISE NOTICE 'V110: Creating continuous aggregate: weekly_paddd_by_type...';

DO $$ BEGIN
    CREATE MATERIALIZED VIEW gl_eudr_pav_weekly_paddd_by_type
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('7 days', timestamp)        AS week,
        tenant_id,
        event_type                              AS audit_event_type,
        entity_type,
        COUNT(*)                                AS event_count
    FROM gl_eudr_pav_audit_log
    WHERE entity_type = 'paddd_event'
    GROUP BY week, tenant_id, event_type, entity_type;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    SELECT add_continuous_aggregate_policy('gl_eudr_pav_weekly_paddd_by_type',
        start_offset => INTERVAL '14 days',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

COMMENT ON MATERIALIZED VIEW gl_eudr_pav_weekly_paddd_by_type IS 'Weekly rollup of PADDD-related audit events by type for trend monitoring of protected area downgrading, downsizing, and degazettement';


-- ============================================================================
-- RETENTION POLICIES (5 years per EUDR Article 31)
-- ============================================================================

RAISE NOTICE 'V110: Creating retention policies (5 years per EUDR Article 31)...';

-- 5 years for overlap analysis logs
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_pav_overlap_analysis_log', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for buffer zone monitoring logs
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_pav_buffer_zone_monitoring_log', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for violation events
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_pav_violation_events', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 5 years for audit logs
DO $$ BEGIN
    SELECT add_retention_policy('gl_eudr_pav_audit_log', INTERVAL '5 years');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;


-- ============================================================================
-- GRANTS -- greenlang_app role
-- ============================================================================

RAISE NOTICE 'V110: Granting permissions to greenlang_app...';

-- Regular tables
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_pav_protected_areas TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_pav_protected_area_overlaps TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_pav_buffer_zone_violations TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_pav_designation_validations TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_pav_risk_assessments TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_pav_violation_alerts TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_pav_compliance_reports TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_pav_paddd_events TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_pav_iucn_category_mappings TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_pav_country_buffer_regulations TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON gl_eudr_pav_protected_area_plot_associations TO greenlang_app;

-- Hypertables
GRANT SELECT, INSERT ON gl_eudr_pav_overlap_analysis_log TO greenlang_app;
GRANT SELECT, INSERT ON gl_eudr_pav_buffer_zone_monitoring_log TO greenlang_app;
GRANT SELECT, INSERT ON gl_eudr_pav_violation_events TO greenlang_app;
GRANT SELECT, INSERT ON gl_eudr_pav_audit_log TO greenlang_app;

-- Continuous aggregates
GRANT SELECT ON gl_eudr_pav_daily_overlap_by_iucn TO greenlang_app;
GRANT SELECT ON gl_eudr_pav_monthly_violations_by_severity TO greenlang_app;
GRANT SELECT ON gl_eudr_pav_weekly_paddd_by_type TO greenlang_app;

-- Read-only role (conditional)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_readonly') THEN
        GRANT SELECT ON gl_eudr_pav_protected_areas TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_pav_protected_area_overlaps TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_pav_buffer_zone_violations TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_pav_designation_validations TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_pav_risk_assessments TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_pav_violation_alerts TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_pav_compliance_reports TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_pav_paddd_events TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_pav_iucn_category_mappings TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_pav_country_buffer_regulations TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_pav_protected_area_plot_associations TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_pav_overlap_analysis_log TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_pav_buffer_zone_monitoring_log TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_pav_violation_events TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_pav_audit_log TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_pav_daily_overlap_by_iucn TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_pav_monthly_violations_by_severity TO greenlang_readonly;
        GRANT SELECT ON gl_eudr_pav_weekly_paddd_by_type TO greenlang_readonly;
    END IF;
END
$$;


-- ============================================================================
-- FINALIZE
-- ============================================================================

RAISE NOTICE 'V110: AGENT-EUDR-022 Protected Area Validator tables created successfully!';
RAISE NOTICE 'V110: Created 15 tables (11 regular + 4 hypertables), 3 continuous aggregates, ~160 indexes';
RAISE NOTICE 'V110: 3 GIST spatial indexes, 18 GIN indexes on JSONB, 8 partial indexes for active records';
RAISE NOTICE 'V110: Retention policies: 5y on all hypertables per EUDR Article 31';
RAISE NOTICE 'V110: Grants applied for greenlang_app and greenlang_readonly roles';

COMMIT;
