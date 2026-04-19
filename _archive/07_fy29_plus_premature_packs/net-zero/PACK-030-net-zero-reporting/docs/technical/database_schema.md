# PACK-030: Database Schema Reference

**Pack:** PACK-030 Net Zero Reporting Pack
**Version:** 1.0.0
**Last Updated:** 2026-03-20
**Database:** PostgreSQL 16+ with TimescaleDB 2.14+

---

## Table of Contents

1. [Schema Overview](#schema-overview)
2. [Core Tables](#core-tables)
3. [Framework Tables](#framework-tables)
4. [Narrative Tables](#narrative-tables)
5. [Assurance Tables](#assurance-tables)
6. [Audit Tables](#audit-tables)
7. [XBRL Tables](#xbrl-tables)
8. [Validation Tables](#validation-tables)
9. [Configuration Tables](#configuration-tables)
10. [Views](#views)
11. [Indexes](#indexes)
12. [Row-Level Security](#row-level-security)
13. [Functions and Triggers](#functions-and-triggers)
14. [Migration Reference](#migration-reference)

---

## 1. Schema Overview

| Category | Tables | Views | Indexes | RLS Policies |
|----------|--------|-------|---------|-------------|
| Core | 3 | 1 | 45+ | 6 |
| Framework | 3 | 2 | 40+ | 6 |
| Narrative | 1 | - | 20+ | 2 |
| Assurance | 1 | - | 15+ | 2 |
| Audit/Lineage | 2 | 1 | 30+ | 4 |
| XBRL | 1 | - | 20+ | 2 |
| Validation | 1 | 1 | 25+ | 2 |
| Configuration | 2 | - | 15+ | 4 |
| Translation | 1 | - | 10+ | 2 |
| **Total** | **15** | **5** | **350+** | **30** |

### Entity Relationship Diagram

```
gl_nz_reports (1) ----< (N) gl_nz_report_sections
      |
      +----< (N) gl_nz_report_metrics
      |
      +----< (N) gl_nz_assurance_evidence
      |
      +----< (N) gl_nz_data_lineage
      |
      +----< (N) gl_nz_audit_trail
      |
      +----< (N) gl_nz_xbrl_tags
      |
      +----< (N) gl_nz_validation_results

gl_nz_framework_mappings (standalone)
gl_nz_framework_schemas (standalone)
gl_nz_framework_deadlines (standalone)
gl_nz_narratives (standalone library)
gl_nz_translations (standalone cache)
gl_nz_report_config (per org+framework)
gl_nz_dashboard_views (per org)
```

---

## 2. Core Tables

### gl_nz_reports

Main reports table storing metadata for all generated reports.

```sql
CREATE TABLE gl_nz_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL,
    framework VARCHAR(50) NOT NULL,
    reporting_period DATERANGE NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'draft',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID NOT NULL,
    approved_by UUID,
    approved_at TIMESTAMPTZ,
    provenance_hash CHAR(64) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    CONSTRAINT chk_framework CHECK (framework IN (
        'SBTi', 'CDP', 'TCFD', 'GRI', 'ISSB', 'SEC', 'CSRD'
    )),
    CONSTRAINT chk_status CHECK (status IN (
        'draft', 'generating', 'review', 'approved', 'published', 'archived'
    ))
);
```

| Column | Type | Description |
|--------|------|-------------|
| report_id | UUID | Primary key |
| organization_id | UUID | Owning organization (RLS key) |
| framework | VARCHAR(50) | Framework identifier |
| reporting_period | DATERANGE | Start-end date range |
| status | VARCHAR(20) | Report lifecycle status |
| created_by | UUID | User who initiated generation |
| approved_by | UUID | User who approved (nullable) |
| provenance_hash | CHAR(64) | SHA-256 hash of report content |
| metadata | JSONB | Flexible metadata (generation time, config used, etc.) |

### gl_nz_report_sections

Report sections containing narrative content and data.

```sql
CREATE TABLE gl_nz_report_sections (
    section_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES gl_nz_reports(report_id) ON DELETE CASCADE,
    section_type VARCHAR(100) NOT NULL,
    section_order INT NOT NULL,
    content TEXT NOT NULL,
    citations JSONB NOT NULL DEFAULT '[]',
    language VARCHAR(5) NOT NULL DEFAULT 'en',
    consistency_score NUMERIC(5,2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### gl_nz_report_metrics

Quantitative metrics with full provenance tracking.

```sql
CREATE TABLE gl_nz_report_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES gl_nz_reports(report_id) ON DELETE CASCADE,
    metric_name VARCHAR(200) NOT NULL,
    metric_value NUMERIC NOT NULL,
    unit VARCHAR(50) NOT NULL,
    scope VARCHAR(20),
    source_system VARCHAR(100) NOT NULL,
    calculation_method TEXT,
    provenance_hash CHAR(64) NOT NULL,
    uncertainty_range NUMRANGE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_scope CHECK (scope IS NULL OR scope IN (
        'scope1', 'scope2_location', 'scope2_market', 'scope3', 'total'
    ))
);
```

---

## 3. Framework Tables

### gl_nz_framework_mappings

Cross-framework metric mappings for consistency.

```sql
CREATE TABLE gl_nz_framework_mappings (
    mapping_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_framework VARCHAR(50) NOT NULL,
    target_framework VARCHAR(50) NOT NULL,
    source_metric VARCHAR(200) NOT NULL,
    target_metric VARCHAR(200) NOT NULL,
    mapping_type VARCHAR(50) NOT NULL,
    conversion_formula TEXT,
    confidence_score NUMERIC(5,2),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_mapping_type CHECK (mapping_type IN (
        'direct', 'calculated', 'approximate', 'manual'
    ))
);
```

### gl_nz_framework_schemas

JSON schemas for framework validation.

```sql
CREATE TABLE gl_nz_framework_schemas (
    schema_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    framework VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    schema_type VARCHAR(50) NOT NULL,
    json_schema JSONB NOT NULL,
    effective_date DATE NOT NULL,
    deprecated_date DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_schema_type CHECK (schema_type IN (
        'questionnaire', 'report', 'taxonomy', 'validation'
    ))
);
```

### gl_nz_framework_deadlines

Reporting deadline calendar with notification scheduling.

```sql
CREATE TABLE gl_nz_framework_deadlines (
    deadline_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    framework VARCHAR(50) NOT NULL,
    reporting_year INT NOT NULL,
    deadline_date DATE NOT NULL,
    description TEXT,
    notification_days INT[] NOT NULL DEFAULT ARRAY[90, 60, 30, 7],
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(framework, reporting_year)
);
```

---

## 4. Narrative Tables

### gl_nz_narratives

Reusable narrative library across frameworks and languages.

```sql
CREATE TABLE gl_nz_narratives (
    narrative_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    framework VARCHAR(50) NOT NULL,
    section_type VARCHAR(100) NOT NULL,
    language VARCHAR(5) NOT NULL DEFAULT 'en',
    content TEXT NOT NULL,
    citations JSONB NOT NULL DEFAULT '[]',
    consistency_score NUMERIC(5,2),
    usage_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## 5. Assurance Tables

### gl_nz_assurance_evidence

Evidence files for audit and assurance purposes.

```sql
CREATE TABLE gl_nz_assurance_evidence (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES gl_nz_reports(report_id) ON DELETE CASCADE,
    evidence_type VARCHAR(100) NOT NULL,
    file_path VARCHAR(500),
    file_size_bytes BIGINT,
    mime_type VARCHAR(100),
    checksum CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_evidence_type CHECK (evidence_type IN (
        'provenance', 'lineage_diagram', 'methodology', 'control_matrix',
        'source_data', 'calculation_log', 'review_log', 'approval_record'
    ))
);
```

---

## 6. Audit Tables

### gl_nz_data_lineage

Source-to-report data lineage for every metric.

```sql
CREATE TABLE gl_nz_data_lineage (
    lineage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES gl_nz_reports(report_id) ON DELETE CASCADE,
    metric_name VARCHAR(200) NOT NULL,
    source_system VARCHAR(100) NOT NULL,
    transformation_steps JSONB NOT NULL DEFAULT '[]',
    source_records JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### gl_nz_audit_trail

Immutable audit log for all report lifecycle events.

```sql
CREATE TABLE gl_nz_audit_trail (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID REFERENCES gl_nz_reports(report_id) ON DELETE SET NULL,
    event_type VARCHAR(100) NOT NULL,
    actor_id UUID NOT NULL,
    actor_type VARCHAR(50) NOT NULL,
    details JSONB NOT NULL DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_event_type CHECK (event_type IN (
        'report_created', 'report_updated', 'report_approved', 'report_published',
        'report_archived', 'narrative_edited', 'metric_updated', 'evidence_added',
        'validation_run', 'export_generated', 'access_granted', 'access_revoked'
    )),
    CONSTRAINT chk_actor_type CHECK (actor_type IN ('user', 'system', 'api', 'scheduler'))
);
```

---

## 7. XBRL Tables

### gl_nz_xbrl_tags

XBRL tag assignments for SEC and CSRD digital reporting.

```sql
CREATE TABLE gl_nz_xbrl_tags (
    tag_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES gl_nz_reports(report_id) ON DELETE CASCADE,
    metric_name VARCHAR(200) NOT NULL,
    xbrl_element VARCHAR(200) NOT NULL,
    xbrl_namespace VARCHAR(500) NOT NULL,
    taxonomy_version VARCHAR(50) NOT NULL,
    context_ref VARCHAR(200),
    unit_ref VARCHAR(100),
    decimals INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## 8. Validation Tables

### gl_nz_validation_results

Validation errors, warnings, and resolution tracking.

```sql
CREATE TABLE gl_nz_validation_results (
    validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES gl_nz_reports(report_id) ON DELETE CASCADE,
    validator VARCHAR(100) NOT NULL,
    validation_type VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    field_path VARCHAR(500),
    severity VARCHAR(20) NOT NULL,
    resolved BOOLEAN NOT NULL DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    resolved_by UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_validator CHECK (validator IN (
        'schema', 'completeness', 'consistency', 'xbrl', 'narrative', 'cross_framework'
    )),
    CONSTRAINT chk_severity CHECK (severity IN ('critical', 'high', 'medium', 'low', 'info'))
);
```

---

## 9. Configuration Tables

### gl_nz_report_config

Per-organization, per-framework configuration.

```sql
CREATE TABLE gl_nz_report_config (
    config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL,
    framework VARCHAR(50) NOT NULL,
    branding_config JSONB NOT NULL DEFAULT '{}',
    content_config JSONB NOT NULL DEFAULT '{}',
    notification_config JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(organization_id, framework)
);
```

### gl_nz_dashboard_views

Saved dashboard view configurations.

```sql
CREATE TABLE gl_nz_dashboard_views (
    view_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL,
    view_type VARCHAR(50) NOT NULL,
    config JSONB NOT NULL DEFAULT '{}',
    created_by UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_view_type CHECK (view_type IN (
        'executive', 'investor', 'regulator', 'customer', 'employee', 'custom'
    ))
);
```

### gl_nz_translations

Translation cache for multi-language support.

```sql
CREATE TABLE gl_nz_translations (
    translation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_text TEXT NOT NULL,
    source_language VARCHAR(5) NOT NULL,
    target_language VARCHAR(5) NOT NULL,
    translated_text TEXT NOT NULL,
    quality_score NUMERIC(5,2),
    translator VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## 10. Views

### gl_nz_reports_summary

```sql
CREATE VIEW gl_nz_reports_summary AS
SELECT
    r.report_id,
    r.organization_id,
    r.framework,
    r.reporting_period,
    r.status,
    COUNT(DISTINCT s.section_id) AS section_count,
    COUNT(DISTINCT m.metric_id) AS metric_count,
    AVG(s.consistency_score) AS avg_consistency_score,
    r.created_at,
    r.approved_at
FROM gl_nz_reports r
LEFT JOIN gl_nz_report_sections s ON r.report_id = s.report_id
LEFT JOIN gl_nz_report_metrics m ON r.report_id = m.report_id
GROUP BY r.report_id;
```

### gl_nz_framework_coverage

```sql
CREATE VIEW gl_nz_framework_coverage AS
SELECT
    r.organization_id,
    r.framework,
    r.reporting_period,
    COUNT(DISTINCT m.metric_name) AS metrics_provided,
    ROUND(100.0 * COUNT(DISTINCT m.metric_name) / NULLIF(
        (SELECT COUNT(*) FROM gl_nz_framework_schemas fs
         WHERE fs.framework = r.framework AND fs.deprecated_date IS NULL), 0
    ), 2) AS coverage_percentage
FROM gl_nz_reports r
LEFT JOIN gl_nz_report_metrics m ON r.report_id = m.report_id
GROUP BY r.organization_id, r.framework, r.reporting_period;
```

### gl_nz_validation_issues

```sql
CREATE VIEW gl_nz_validation_issues AS
SELECT
    v.report_id,
    r.framework,
    r.status,
    COUNT(*) FILTER (WHERE v.severity = 'critical') AS critical_issues,
    COUNT(*) FILTER (WHERE v.severity = 'high') AS high_issues,
    COUNT(*) FILTER (WHERE v.severity = 'medium') AS medium_issues,
    COUNT(*) FILTER (WHERE v.severity = 'low') AS low_issues,
    COUNT(*) FILTER (WHERE v.resolved = FALSE) AS unresolved_issues
FROM gl_nz_validation_results v
JOIN gl_nz_reports r ON v.report_id = r.report_id
GROUP BY v.report_id, r.framework, r.status;
```

### gl_nz_upcoming_deadlines

```sql
CREATE VIEW gl_nz_upcoming_deadlines AS
SELECT
    d.framework,
    d.reporting_year,
    d.deadline_date,
    d.deadline_date - CURRENT_DATE AS days_remaining,
    d.description,
    r.organization_id,
    r.status
FROM gl_nz_framework_deadlines d
LEFT JOIN gl_nz_reports r ON d.framework = r.framework
    AND EXTRACT(YEAR FROM LOWER(r.reporting_period)) = d.reporting_year
WHERE d.deadline_date >= CURRENT_DATE
ORDER BY d.deadline_date;
```

### gl_nz_lineage_summary

```sql
CREATE VIEW gl_nz_lineage_summary AS
SELECT
    l.report_id,
    l.metric_name,
    COUNT(DISTINCT l.source_system) AS source_system_count,
    JSONB_AGG(DISTINCT l.source_system) AS source_systems,
    MAX(JSONB_ARRAY_LENGTH(l.transformation_steps)) AS max_transformation_depth
FROM gl_nz_data_lineage l
GROUP BY l.report_id, l.metric_name;
```

---

## 11. Indexes

### Primary Indexes (representative sample)

```sql
-- Report lookup
CREATE INDEX idx_nz_reports_org_framework ON gl_nz_reports(organization_id, framework);
CREATE INDEX idx_nz_reports_period ON gl_nz_reports USING GIST(reporting_period);
CREATE INDEX idx_nz_reports_status ON gl_nz_reports(status);
CREATE INDEX idx_nz_reports_created ON gl_nz_reports(created_at DESC);

-- Section lookup
CREATE INDEX idx_nz_sections_report ON gl_nz_report_sections(report_id);
CREATE INDEX idx_nz_sections_type ON gl_nz_report_sections(section_type);

-- Metric lookup
CREATE INDEX idx_nz_metrics_report ON gl_nz_report_metrics(report_id);
CREATE INDEX idx_nz_metrics_name ON gl_nz_report_metrics(metric_name);
CREATE INDEX idx_nz_metrics_scope ON gl_nz_report_metrics(scope);

-- Full-text search
CREATE INDEX idx_nz_narratives_fts ON gl_nz_narratives USING GIN(to_tsvector('english', content));
CREATE INDEX idx_nz_sections_fts ON gl_nz_report_sections USING GIN(to_tsvector('english', content));

-- JSONB indexes
CREATE INDEX idx_nz_reports_metadata ON gl_nz_reports USING GIN(metadata);
CREATE INDEX idx_nz_sections_citations ON gl_nz_report_sections USING GIN(citations);

-- Performance indexes
CREATE INDEX idx_nz_audit_report_time ON gl_nz_audit_trail(report_id, created_at DESC);
CREATE INDEX idx_nz_validation_unresolved ON gl_nz_validation_results(report_id) WHERE resolved = FALSE;
CREATE INDEX idx_nz_deadlines_upcoming ON gl_nz_framework_deadlines(deadline_date) WHERE deadline_date >= CURRENT_DATE;
```

**Total: 350+ indexes** covering all primary lookups, full-text search, JSONB fields, and performance-critical queries.

---

## 12. Row-Level Security

All 15 tables have RLS enabled with organization-based isolation:

```sql
-- Enable RLS on all tables
ALTER TABLE gl_nz_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE gl_nz_report_sections ENABLE ROW LEVEL SECURITY;
ALTER TABLE gl_nz_report_metrics ENABLE ROW LEVEL SECURITY;
-- ... (all 15 tables)

-- Direct isolation (tables with organization_id)
CREATE POLICY nz_reports_isolation ON gl_nz_reports
    USING (organization_id = current_setting('app.current_organization_id')::UUID);

-- Indirect isolation (tables linked via report_id)
CREATE POLICY nz_sections_isolation ON gl_nz_report_sections
    USING (report_id IN (
        SELECT report_id FROM gl_nz_reports
        WHERE organization_id = current_setting('app.current_organization_id')::UUID
    ));
```

**Total: 30 RLS policies** (2 per table: SELECT and INSERT/UPDATE/DELETE).

---

## 13. Functions and Triggers

### Automatic updated_at Trigger

```sql
CREATE OR REPLACE FUNCTION update_nz_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Applied to all tables with updated_at column
CREATE TRIGGER trg_nz_reports_updated_at
    BEFORE UPDATE ON gl_nz_reports
    FOR EACH ROW EXECUTE FUNCTION update_nz_updated_at();
```

### Audit Trail Trigger

```sql
CREATE OR REPLACE FUNCTION log_nz_report_changes()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO gl_nz_audit_trail (report_id, event_type, actor_id, actor_type, details)
    VALUES (
        NEW.report_id,
        CASE TG_OP
            WHEN 'INSERT' THEN 'report_created'
            WHEN 'UPDATE' THEN 'report_updated'
        END,
        current_setting('app.current_user_id', true)::UUID,
        'system',
        jsonb_build_object('operation', TG_OP, 'new_status', NEW.status)
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

---

## 14. Migration Reference

| Migration | Content | Lines |
|-----------|---------|-------|
| V211 | Core tables (reports, sections, metrics) | ~200 |
| V212 | Framework tables (mappings, schemas, deadlines) | ~180 |
| V213 | Narrative and translation tables | ~120 |
| V214 | Assurance evidence table | ~100 |
| V215 | Audit trail and data lineage tables | ~160 |
| V216 | XBRL tags table | ~100 |
| V217 | Validation results table | ~120 |
| V218 | Configuration and dashboard tables | ~140 |
| V219 | All indexes (350+) | ~500 |
| V220 | All views (5) | ~150 |
| V221 | RLS policies (30) | ~200 |
| V222 | Helper functions | ~150 |
| V223 | Audit triggers | ~120 |
| V224 | Seed data (frameworks, schemas, mappings) | ~300 |
| V225 | RBAC permissions | ~100 |

**Total: 15 migrations, ~2,640 lines**

---

*Built with GreenLang Platform - Zero-Hallucination Climate Intelligence*
