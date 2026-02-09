-- =============================================================================
-- V038: Supplier Questionnaire Service Schema
-- =============================================================================
-- Component: AGENT-DATA-008 (Supplier Questionnaire Processor)
-- Agent ID:  GL-DATA-SUP-001
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Supplier Questionnaire Processor Agent (GL-DATA-SUP-001) with capabilities
-- for questionnaire template management, section and question definitions,
-- supplier distribution and campaign tracking, response collection and
-- answer storage, validation and data quality scoring, supplier performance
-- scoring with framework benchmarking, and follow-up action orchestration.
-- =============================================================================
-- Tables (10):
--   1. questionnaire_templates  - Questionnaire template definitions with framework and versioning
--   2. template_sections        - Ordered sections within a questionnaire template
--   3. template_questions       - Individual questions within template sections
--   4. distributions            - Questionnaire distribution to suppliers with lifecycle tracking
--   5. distribution_events      - Distribution lifecycle event log (hypertable)
--   6. responses                - Supplier response records with completion tracking
--   7. response_answers         - Individual answers to questionnaire questions
--   8. validation_results       - Response validation check results (hypertable)
--   9. scores                   - Supplier performance scores with framework benchmarking
--  10. follow_up_actions        - Reminder and escalation action tracking (hypertable)
--
-- Continuous Aggregates (2):
--   1. supplier_quest_distribution_hourly - Hourly distribution event aggregates
--   2. supplier_quest_validation_hourly   - Hourly validation result aggregates
--
-- Also includes: 50+ indexes (B-tree, GIN), RLS policies per tenant,
-- retention policies (90 days on hypertables), compression policies,
-- updated_at trigger, security permissions, and seed data registering
-- GL-DATA-SUP-001 in the agent registry.
-- Previous: V037__deforestation_satellite_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS supplier_questionnaire_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================
-- Reusable trigger function for tables with updated_at columns.

CREATE OR REPLACE FUNCTION supplier_questionnaire_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: supplier_questionnaire_service.questionnaire_templates
-- =============================================================================
-- Questionnaire template definitions. Each template captures the name,
-- description, sustainability framework (CDP, GRI, SASB, EcoVadis, custom),
-- version, language, publication status, tags for discovery, total question
-- count, creator, tenant scope, and provenance hash. Tenant-scoped.

CREATE TABLE supplier_questionnaire_service.questionnaire_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_id VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(256) NOT NULL,
    description TEXT,
    framework VARCHAR(32) NOT NULL DEFAULT 'custom',
    version VARCHAR(16) NOT NULL DEFAULT '1.0.0',
    language VARCHAR(8) NOT NULL DEFAULT 'en',
    status VARCHAR(32) NOT NULL DEFAULT 'draft',
    tags TEXT[] DEFAULT '{}',
    total_questions INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(128),
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(128)
);

-- Template ID must not be empty
ALTER TABLE supplier_questionnaire_service.questionnaire_templates
    ADD CONSTRAINT chk_qt_template_id_not_empty
    CHECK (LENGTH(TRIM(template_id)) > 0);

-- Name must not be empty
ALTER TABLE supplier_questionnaire_service.questionnaire_templates
    ADD CONSTRAINT chk_qt_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- Framework constraint
ALTER TABLE supplier_questionnaire_service.questionnaire_templates
    ADD CONSTRAINT chk_qt_framework
    CHECK (framework IN (
        'cdp', 'gri', 'sasb', 'ecovadis', 'tcfd', 'sbti',
        'iso14001', 'ghg_protocol', 'csrd', 'custom'
    ));

-- Status constraint
ALTER TABLE supplier_questionnaire_service.questionnaire_templates
    ADD CONSTRAINT chk_qt_status
    CHECK (status IN ('draft', 'active', 'archived', 'deprecated', 'review'));

-- Total questions must be non-negative
ALTER TABLE supplier_questionnaire_service.questionnaire_templates
    ADD CONSTRAINT chk_qt_total_questions_non_negative
    CHECK (total_questions >= 0);

-- Version must not be empty
ALTER TABLE supplier_questionnaire_service.questionnaire_templates
    ADD CONSTRAINT chk_qt_version_not_empty
    CHECK (LENGTH(TRIM(version)) > 0);

-- Language must not be empty
ALTER TABLE supplier_questionnaire_service.questionnaire_templates
    ADD CONSTRAINT chk_qt_language_not_empty
    CHECK (LENGTH(TRIM(language)) > 0);

-- updated_at trigger
CREATE TRIGGER trg_qt_updated_at
    BEFORE UPDATE ON supplier_questionnaire_service.questionnaire_templates
    FOR EACH ROW
    EXECUTE FUNCTION supplier_questionnaire_service.set_updated_at();

-- =============================================================================
-- Table 2: supplier_questionnaire_service.template_sections
-- =============================================================================
-- Ordered sections within a questionnaire template. Each section captures
-- the section name, description, display order, minimum required answers
-- for section completion, and translations for multi-language support.
-- Linked to questionnaire_templates via template_id.

CREATE TABLE supplier_questionnaire_service.template_sections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    section_id VARCHAR(64) UNIQUE NOT NULL,
    template_id VARCHAR(64) NOT NULL,
    section_name VARCHAR(256) NOT NULL,
    description TEXT,
    section_order INTEGER NOT NULL DEFAULT 0,
    min_required_answers INTEGER DEFAULT 0,
    translations JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Foreign key to questionnaire_templates
ALTER TABLE supplier_questionnaire_service.template_sections
    ADD CONSTRAINT fk_ts_template_id
    FOREIGN KEY (template_id) REFERENCES supplier_questionnaire_service.questionnaire_templates(template_id)
    ON DELETE CASCADE;

-- Section ID must not be empty
ALTER TABLE supplier_questionnaire_service.template_sections
    ADD CONSTRAINT chk_ts_section_id_not_empty
    CHECK (LENGTH(TRIM(section_id)) > 0);

-- Section name must not be empty
ALTER TABLE supplier_questionnaire_service.template_sections
    ADD CONSTRAINT chk_ts_section_name_not_empty
    CHECK (LENGTH(TRIM(section_name)) > 0);

-- Section order must be non-negative
ALTER TABLE supplier_questionnaire_service.template_sections
    ADD CONSTRAINT chk_ts_section_order_non_negative
    CHECK (section_order >= 0);

-- Min required answers must be non-negative if specified
ALTER TABLE supplier_questionnaire_service.template_sections
    ADD CONSTRAINT chk_ts_min_required_non_negative
    CHECK (min_required_answers IS NULL OR min_required_answers >= 0);

-- =============================================================================
-- Table 3: supplier_questionnaire_service.template_questions
-- =============================================================================
-- Individual questions within template sections. Each question captures
-- the question text, type (text, numeric, single_choice, multi_choice,
-- date, file_upload, boolean, scale), required flag, answer options
-- (JSONB for choice questions), conditional display rules, scoring weight,
-- help text, translations, and display order. Linked to template_sections
-- via section_id.

CREATE TABLE supplier_questionnaire_service.template_questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question_id VARCHAR(64) UNIQUE NOT NULL,
    section_id VARCHAR(64) NOT NULL,
    template_id VARCHAR(64) NOT NULL,
    question_text TEXT NOT NULL,
    question_type VARCHAR(32) NOT NULL DEFAULT 'text',
    required BOOLEAN NOT NULL DEFAULT false,
    options JSONB DEFAULT '[]'::jsonb,
    conditional_rules JSONB DEFAULT '[]'::jsonb,
    score_weight NUMERIC(5,3) NOT NULL DEFAULT 1.0,
    help_text TEXT,
    translations JSONB DEFAULT '{}'::jsonb,
    question_order INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Foreign key to template_sections
ALTER TABLE supplier_questionnaire_service.template_questions
    ADD CONSTRAINT fk_tq_section_id
    FOREIGN KEY (section_id) REFERENCES supplier_questionnaire_service.template_sections(section_id)
    ON DELETE CASCADE;

-- Question ID must not be empty
ALTER TABLE supplier_questionnaire_service.template_questions
    ADD CONSTRAINT chk_tq_question_id_not_empty
    CHECK (LENGTH(TRIM(question_id)) > 0);

-- Question text must not be empty
ALTER TABLE supplier_questionnaire_service.template_questions
    ADD CONSTRAINT chk_tq_question_text_not_empty
    CHECK (LENGTH(TRIM(question_text)) > 0);

-- Question type constraint
ALTER TABLE supplier_questionnaire_service.template_questions
    ADD CONSTRAINT chk_tq_question_type
    CHECK (question_type IN (
        'text', 'numeric', 'single_choice', 'multi_choice',
        'date', 'file_upload', 'boolean', 'scale', 'textarea'
    ));

-- Score weight must be non-negative
ALTER TABLE supplier_questionnaire_service.template_questions
    ADD CONSTRAINT chk_tq_score_weight_non_negative
    CHECK (score_weight >= 0);

-- Question order must be non-negative
ALTER TABLE supplier_questionnaire_service.template_questions
    ADD CONSTRAINT chk_tq_question_order_non_negative
    CHECK (question_order >= 0);

-- Template ID must not be empty
ALTER TABLE supplier_questionnaire_service.template_questions
    ADD CONSTRAINT chk_tq_template_id_not_empty
    CHECK (LENGTH(TRIM(template_id)) > 0);

-- =============================================================================
-- Table 4: supplier_questionnaire_service.distributions
-- =============================================================================
-- Questionnaire distribution records tracking the lifecycle of sending
-- questionnaires to suppliers. Each distribution captures the template,
-- supplier details, delivery channel (email, portal, api), campaign
-- grouping, access token for secure supplier access, lifecycle timestamps
-- (sent, delivered, opened, started, completed), reminder count,
-- deadline, tenant scope, and provenance hash. Tenant-scoped.

CREATE TABLE supplier_questionnaire_service.distributions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    distribution_id VARCHAR(64) UNIQUE NOT NULL,
    template_id VARCHAR(64) NOT NULL,
    supplier_id VARCHAR(128) NOT NULL,
    supplier_name VARCHAR(256) NOT NULL,
    supplier_email VARCHAR(256),
    channel VARCHAR(32) NOT NULL DEFAULT 'email',
    campaign_id VARCHAR(64),
    status VARCHAR(32) NOT NULL DEFAULT 'queued',
    deadline TIMESTAMPTZ,
    access_token VARCHAR(256),
    sent_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    opened_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    reminder_count INTEGER NOT NULL DEFAULT 0,
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Distribution ID must not be empty
ALTER TABLE supplier_questionnaire_service.distributions
    ADD CONSTRAINT chk_dist_distribution_id_not_empty
    CHECK (LENGTH(TRIM(distribution_id)) > 0);

-- Template ID must not be empty
ALTER TABLE supplier_questionnaire_service.distributions
    ADD CONSTRAINT chk_dist_template_id_not_empty
    CHECK (LENGTH(TRIM(template_id)) > 0);

-- Supplier ID must not be empty
ALTER TABLE supplier_questionnaire_service.distributions
    ADD CONSTRAINT chk_dist_supplier_id_not_empty
    CHECK (LENGTH(TRIM(supplier_id)) > 0);

-- Supplier name must not be empty
ALTER TABLE supplier_questionnaire_service.distributions
    ADD CONSTRAINT chk_dist_supplier_name_not_empty
    CHECK (LENGTH(TRIM(supplier_name)) > 0);

-- Channel constraint
ALTER TABLE supplier_questionnaire_service.distributions
    ADD CONSTRAINT chk_dist_channel
    CHECK (channel IN ('email', 'portal', 'api', 'bulk_email', 'manual'));

-- Status constraint
ALTER TABLE supplier_questionnaire_service.distributions
    ADD CONSTRAINT chk_dist_status
    CHECK (status IN (
        'queued', 'sent', 'delivered', 'opened', 'started',
        'completed', 'expired', 'bounced', 'failed', 'cancelled'
    ));

-- Reminder count must be non-negative
ALTER TABLE supplier_questionnaire_service.distributions
    ADD CONSTRAINT chk_dist_reminder_count_non_negative
    CHECK (reminder_count >= 0);

-- =============================================================================
-- Table 5: supplier_questionnaire_service.distribution_events (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording distribution lifecycle events.
-- Each event captures the distribution reference, event type (sent,
-- delivered, opened, started, submitted, reminder_sent, expired, bounced),
-- status transitions, and metadata. Partitioned by created_at for
-- time-series queries. Retained for 90 days with compression after 7 days.

CREATE TABLE supplier_questionnaire_service.distribution_events (
    id UUID DEFAULT gen_random_uuid(),
    distribution_id VARCHAR(64) NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    old_status VARCHAR(32),
    new_status VARCHAR(32),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
);

-- Create hypertable partitioned by created_at with 7-day chunks
SELECT create_hypertable('supplier_questionnaire_service.distribution_events', 'created_at',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Distribution ID must not be empty
ALTER TABLE supplier_questionnaire_service.distribution_events
    ADD CONSTRAINT chk_de_distribution_id_not_empty
    CHECK (LENGTH(TRIM(distribution_id)) > 0);

-- Event type constraint
ALTER TABLE supplier_questionnaire_service.distribution_events
    ADD CONSTRAINT chk_de_event_type
    CHECK (event_type IN (
        'created', 'queued', 'sent', 'delivered', 'opened',
        'started', 'submitted', 'reminder_sent', 'expired',
        'bounced', 'failed', 'cancelled', 'reopened',
        'escalated', 'status_change'
    ));

-- =============================================================================
-- Table 6: supplier_questionnaire_service.responses
-- =============================================================================
-- Supplier response records tracking the overall response lifecycle.
-- Each response links a distribution to a supplier's answers with
-- completion percentage, version tracking for re-submissions, revision
-- notes, submission and validation timestamps, language, tenant scope,
-- and provenance hash. Tenant-scoped.

CREATE TABLE supplier_questionnaire_service.responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id VARCHAR(64) UNIQUE NOT NULL,
    distribution_id VARCHAR(64) NOT NULL,
    template_id VARCHAR(64) NOT NULL,
    supplier_id VARCHAR(128) NOT NULL,
    supplier_name VARCHAR(256) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'draft',
    completion_pct NUMERIC(5,2) NOT NULL DEFAULT 0,
    language VARCHAR(8) NOT NULL DEFAULT 'en',
    version INTEGER NOT NULL DEFAULT 1,
    revision_notes TEXT,
    submitted_at TIMESTAMPTZ,
    validated_at TIMESTAMPTZ,
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Response ID must not be empty
ALTER TABLE supplier_questionnaire_service.responses
    ADD CONSTRAINT chk_resp_response_id_not_empty
    CHECK (LENGTH(TRIM(response_id)) > 0);

-- Distribution ID must not be empty
ALTER TABLE supplier_questionnaire_service.responses
    ADD CONSTRAINT chk_resp_distribution_id_not_empty
    CHECK (LENGTH(TRIM(distribution_id)) > 0);

-- Template ID must not be empty
ALTER TABLE supplier_questionnaire_service.responses
    ADD CONSTRAINT chk_resp_template_id_not_empty
    CHECK (LENGTH(TRIM(template_id)) > 0);

-- Supplier ID must not be empty
ALTER TABLE supplier_questionnaire_service.responses
    ADD CONSTRAINT chk_resp_supplier_id_not_empty
    CHECK (LENGTH(TRIM(supplier_id)) > 0);

-- Supplier name must not be empty
ALTER TABLE supplier_questionnaire_service.responses
    ADD CONSTRAINT chk_resp_supplier_name_not_empty
    CHECK (LENGTH(TRIM(supplier_name)) > 0);

-- Status constraint
ALTER TABLE supplier_questionnaire_service.responses
    ADD CONSTRAINT chk_resp_status
    CHECK (status IN (
        'draft', 'in_progress', 'submitted', 'validated',
        'rejected', 'revision_requested', 'approved', 'expired'
    ));

-- Completion percentage must be between 0 and 100
ALTER TABLE supplier_questionnaire_service.responses
    ADD CONSTRAINT chk_resp_completion_pct_range
    CHECK (completion_pct >= 0 AND completion_pct <= 100);

-- Version must be positive
ALTER TABLE supplier_questionnaire_service.responses
    ADD CONSTRAINT chk_resp_version_positive
    CHECK (version > 0);

-- Language must not be empty
ALTER TABLE supplier_questionnaire_service.responses
    ADD CONSTRAINT chk_resp_language_not_empty
    CHECK (LENGTH(TRIM(language)) > 0);

-- updated_at trigger
CREATE TRIGGER trg_resp_updated_at
    BEFORE UPDATE ON supplier_questionnaire_service.responses
    FOR EACH ROW
    EXECUTE FUNCTION supplier_questionnaire_service.set_updated_at();

-- =============================================================================
-- Table 7: supplier_questionnaire_service.response_answers
-- =============================================================================
-- Individual answers to questionnaire questions within a supplier response.
-- Each answer captures the text value, numeric value (for quantitative
-- questions), selected choices (for multi-choice), file attachment IDs,
-- confidence score, and notes. Linked to responses via response_id.

CREATE TABLE supplier_questionnaire_service.response_answers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    answer_id VARCHAR(64) UNIQUE NOT NULL,
    response_id VARCHAR(64) NOT NULL,
    question_id VARCHAR(64) NOT NULL,
    section_id VARCHAR(64) NOT NULL,
    answer_value TEXT,
    answer_numeric NUMERIC(20,6),
    answer_choices TEXT[] DEFAULT '{}',
    file_attachment_ids TEXT[] DEFAULT '{}',
    confidence_score NUMERIC(5,3),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Foreign key to responses
ALTER TABLE supplier_questionnaire_service.response_answers
    ADD CONSTRAINT fk_ra_response_id
    FOREIGN KEY (response_id) REFERENCES supplier_questionnaire_service.responses(response_id)
    ON DELETE CASCADE;

-- Answer ID must not be empty
ALTER TABLE supplier_questionnaire_service.response_answers
    ADD CONSTRAINT chk_ra_answer_id_not_empty
    CHECK (LENGTH(TRIM(answer_id)) > 0);

-- Question ID must not be empty
ALTER TABLE supplier_questionnaire_service.response_answers
    ADD CONSTRAINT chk_ra_question_id_not_empty
    CHECK (LENGTH(TRIM(question_id)) > 0);

-- Section ID must not be empty
ALTER TABLE supplier_questionnaire_service.response_answers
    ADD CONSTRAINT chk_ra_section_id_not_empty
    CHECK (LENGTH(TRIM(section_id)) > 0);

-- Confidence score must be between 0 and 1 if specified
ALTER TABLE supplier_questionnaire_service.response_answers
    ADD CONSTRAINT chk_ra_confidence_score_range
    CHECK (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1));

-- =============================================================================
-- Table 8: supplier_questionnaire_service.validation_results (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording response validation check results.
-- Each validation check captures the check identifier, response reference,
-- validation level (field, section, response, cross_field), check name,
-- result (pass, fail, warning, skip), message, severity, field path,
-- suggested correction, and data quality score. Partitioned by created_at
-- for time-series queries. Retained for 90 days with compression after
-- 7 days.

CREATE TABLE supplier_questionnaire_service.validation_results (
    id UUID DEFAULT gen_random_uuid(),
    check_id VARCHAR(64) NOT NULL,
    response_id VARCHAR(64) NOT NULL,
    level VARCHAR(32) NOT NULL,
    check_name VARCHAR(128) NOT NULL,
    result VARCHAR(16) NOT NULL,
    message TEXT NOT NULL,
    severity VARCHAR(16) NOT NULL DEFAULT 'error',
    field_path VARCHAR(256),
    suggestion TEXT,
    data_quality_score NUMERIC(5,2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
);

-- Create hypertable partitioned by created_at with 7-day chunks
SELECT create_hypertable('supplier_questionnaire_service.validation_results', 'created_at',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Check ID must not be empty
ALTER TABLE supplier_questionnaire_service.validation_results
    ADD CONSTRAINT chk_vr_check_id_not_empty
    CHECK (LENGTH(TRIM(check_id)) > 0);

-- Response ID must not be empty
ALTER TABLE supplier_questionnaire_service.validation_results
    ADD CONSTRAINT chk_vr_response_id_not_empty
    CHECK (LENGTH(TRIM(response_id)) > 0);

-- Level constraint
ALTER TABLE supplier_questionnaire_service.validation_results
    ADD CONSTRAINT chk_vr_level
    CHECK (level IN ('field', 'section', 'response', 'cross_field', 'completeness', 'consistency'));

-- Check name must not be empty
ALTER TABLE supplier_questionnaire_service.validation_results
    ADD CONSTRAINT chk_vr_check_name_not_empty
    CHECK (LENGTH(TRIM(check_name)) > 0);

-- Result constraint
ALTER TABLE supplier_questionnaire_service.validation_results
    ADD CONSTRAINT chk_vr_result
    CHECK (result IN ('pass', 'fail', 'warning', 'skip', 'error', 'info'));

-- Severity constraint
ALTER TABLE supplier_questionnaire_service.validation_results
    ADD CONSTRAINT chk_vr_severity
    CHECK (severity IN ('critical', 'error', 'warning', 'info'));

-- Data quality score must be between 0 and 100 if specified
ALTER TABLE supplier_questionnaire_service.validation_results
    ADD CONSTRAINT chk_vr_data_quality_score_range
    CHECK (data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100));

-- =============================================================================
-- Table 9: supplier_questionnaire_service.scores
-- =============================================================================
-- Supplier performance scores derived from questionnaire responses.
-- Each score captures the response and template references, supplier,
-- sustainability framework, overall score, performance tier (leader,
-- advanced, intermediate, beginner, non_responsive), CDP letter grade,
-- per-section score breakdown (JSONB), confidence score, benchmark
-- percentile, tenant scope, and provenance hash. Tenant-scoped.

CREATE TABLE supplier_questionnaire_service.scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    score_id VARCHAR(64) UNIQUE NOT NULL,
    response_id VARCHAR(64) NOT NULL,
    template_id VARCHAR(64) NOT NULL,
    supplier_id VARCHAR(128) NOT NULL,
    framework VARCHAR(32) NOT NULL,
    overall_score NUMERIC(6,2) NOT NULL,
    performance_tier VARCHAR(32) NOT NULL,
    cdp_grade VARCHAR(8),
    section_scores JSONB DEFAULT '{}'::jsonb,
    confidence_score NUMERIC(5,3),
    benchmark_percentile NUMERIC(5,2),
    tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(128),
    scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Score ID must not be empty
ALTER TABLE supplier_questionnaire_service.scores
    ADD CONSTRAINT chk_sc_score_id_not_empty
    CHECK (LENGTH(TRIM(score_id)) > 0);

-- Response ID must not be empty
ALTER TABLE supplier_questionnaire_service.scores
    ADD CONSTRAINT chk_sc_response_id_not_empty
    CHECK (LENGTH(TRIM(response_id)) > 0);

-- Template ID must not be empty
ALTER TABLE supplier_questionnaire_service.scores
    ADD CONSTRAINT chk_sc_template_id_not_empty
    CHECK (LENGTH(TRIM(template_id)) > 0);

-- Supplier ID must not be empty
ALTER TABLE supplier_questionnaire_service.scores
    ADD CONSTRAINT chk_sc_supplier_id_not_empty
    CHECK (LENGTH(TRIM(supplier_id)) > 0);

-- Framework constraint
ALTER TABLE supplier_questionnaire_service.scores
    ADD CONSTRAINT chk_sc_framework
    CHECK (framework IN (
        'cdp', 'gri', 'sasb', 'ecovadis', 'tcfd', 'sbti',
        'iso14001', 'ghg_protocol', 'csrd', 'custom'
    ));

-- Overall score must be non-negative
ALTER TABLE supplier_questionnaire_service.scores
    ADD CONSTRAINT chk_sc_overall_score_non_negative
    CHECK (overall_score >= 0);

-- Performance tier constraint
ALTER TABLE supplier_questionnaire_service.scores
    ADD CONSTRAINT chk_sc_performance_tier
    CHECK (performance_tier IN (
        'leader', 'advanced', 'intermediate', 'beginner',
        'non_responsive', 'insufficient_data'
    ));

-- Confidence score must be between 0 and 1 if specified
ALTER TABLE supplier_questionnaire_service.scores
    ADD CONSTRAINT chk_sc_confidence_score_range
    CHECK (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1));

-- Benchmark percentile must be between 0 and 100 if specified
ALTER TABLE supplier_questionnaire_service.scores
    ADD CONSTRAINT chk_sc_benchmark_percentile_range
    CHECK (benchmark_percentile IS NULL OR (benchmark_percentile >= 0 AND benchmark_percentile <= 100));

-- =============================================================================
-- Table 10: supplier_questionnaire_service.follow_up_actions (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording follow-up reminder and escalation
-- actions for supplier questionnaire distributions. Each action captures
-- the distribution and campaign references, supplier, reminder type
-- (reminder, escalation, final_notice, overdue), escalation level,
-- scheduling and execution timestamps, delivery status, recipient
-- details, and message template. Partitioned by created_at for
-- time-series queries. Retained for 90 days with compression after
-- 7 days.

CREATE TABLE supplier_questionnaire_service.follow_up_actions (
    id UUID DEFAULT gen_random_uuid(),
    action_id VARCHAR(64) NOT NULL,
    distribution_id VARCHAR(64) NOT NULL,
    campaign_id VARCHAR(64),
    supplier_id VARCHAR(128) NOT NULL,
    reminder_type VARCHAR(16) NOT NULL,
    escalation_level VARCHAR(16) NOT NULL DEFAULT 'level_1',
    scheduled_at TIMESTAMPTZ NOT NULL,
    sent_at TIMESTAMPTZ,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    recipient_email VARCHAR(256),
    recipient_name VARCHAR(256),
    message_template VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
);

-- Create hypertable partitioned by created_at with 7-day chunks
SELECT create_hypertable('supplier_questionnaire_service.follow_up_actions', 'created_at',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Action ID must not be empty
ALTER TABLE supplier_questionnaire_service.follow_up_actions
    ADD CONSTRAINT chk_fua_action_id_not_empty
    CHECK (LENGTH(TRIM(action_id)) > 0);

-- Distribution ID must not be empty
ALTER TABLE supplier_questionnaire_service.follow_up_actions
    ADD CONSTRAINT chk_fua_distribution_id_not_empty
    CHECK (LENGTH(TRIM(distribution_id)) > 0);

-- Supplier ID must not be empty
ALTER TABLE supplier_questionnaire_service.follow_up_actions
    ADD CONSTRAINT chk_fua_supplier_id_not_empty
    CHECK (LENGTH(TRIM(supplier_id)) > 0);

-- Reminder type constraint
ALTER TABLE supplier_questionnaire_service.follow_up_actions
    ADD CONSTRAINT chk_fua_reminder_type
    CHECK (reminder_type IN ('reminder', 'escalation', 'final_notice', 'overdue'));

-- Escalation level constraint
ALTER TABLE supplier_questionnaire_service.follow_up_actions
    ADD CONSTRAINT chk_fua_escalation_level
    CHECK (escalation_level IN ('level_1', 'level_2', 'level_3', 'executive'));

-- Status constraint
ALTER TABLE supplier_questionnaire_service.follow_up_actions
    ADD CONSTRAINT chk_fua_status
    CHECK (status IN ('pending', 'sent', 'delivered', 'failed', 'cancelled', 'skipped'));

-- =============================================================================
-- Continuous Aggregate: supplier_questionnaire_service.supplier_quest_distribution_hourly
-- =============================================================================
-- Precomputed hourly distribution event statistics by event type for
-- dashboard queries, campaign monitoring, and supplier engagement tracking.

CREATE MATERIALIZED VIEW supplier_questionnaire_service.supplier_quest_distribution_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at) AS bucket,
    event_type,
    COUNT(*) AS total_events,
    COUNT(DISTINCT distribution_id) AS unique_distributions,
    COUNT(*) FILTER (WHERE new_status = 'completed') AS completed_count,
    COUNT(*) FILTER (WHERE new_status = 'expired') AS expired_count,
    COUNT(*) FILTER (WHERE new_status = 'bounced') AS bounced_count,
    COUNT(*) FILTER (WHERE new_status = 'failed') AS failed_count
FROM supplier_questionnaire_service.distribution_events
WHERE created_at IS NOT NULL
GROUP BY bucket, event_type
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('supplier_questionnaire_service.supplier_quest_distribution_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Continuous Aggregate: supplier_questionnaire_service.supplier_quest_validation_hourly
-- =============================================================================
-- Precomputed hourly validation result statistics by level and result for
-- data quality monitoring, validation trend analysis, and SLI tracking.

CREATE MATERIALIZED VIEW supplier_questionnaire_service.supplier_quest_validation_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at) AS bucket,
    level,
    result,
    COUNT(*) AS total_checks,
    COUNT(DISTINCT response_id) AS unique_responses,
    COUNT(*) FILTER (WHERE severity = 'critical') AS critical_count,
    COUNT(*) FILTER (WHERE severity = 'error') AS error_count,
    COUNT(*) FILTER (WHERE severity = 'warning') AS warning_count,
    COUNT(*) FILTER (WHERE severity = 'info') AS info_count,
    AVG(data_quality_score) AS avg_data_quality_score
FROM supplier_questionnaire_service.validation_results
WHERE created_at IS NOT NULL
GROUP BY bucket, level, result
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('supplier_questionnaire_service.supplier_quest_validation_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- questionnaire_templates indexes
CREATE INDEX idx_qt_template_id ON supplier_questionnaire_service.questionnaire_templates(template_id);
CREATE INDEX idx_qt_name ON supplier_questionnaire_service.questionnaire_templates(name);
CREATE INDEX idx_qt_framework ON supplier_questionnaire_service.questionnaire_templates(framework);
CREATE INDEX idx_qt_status ON supplier_questionnaire_service.questionnaire_templates(status);
CREATE INDEX idx_qt_framework_status ON supplier_questionnaire_service.questionnaire_templates(framework, status);
CREATE INDEX idx_qt_tenant ON supplier_questionnaire_service.questionnaire_templates(tenant_id);
CREATE INDEX idx_qt_created_at ON supplier_questionnaire_service.questionnaire_templates(created_at DESC);
CREATE INDEX idx_qt_updated_at ON supplier_questionnaire_service.questionnaire_templates(updated_at DESC);
CREATE INDEX idx_qt_provenance ON supplier_questionnaire_service.questionnaire_templates(provenance_hash);
CREATE INDEX idx_qt_tenant_framework ON supplier_questionnaire_service.questionnaire_templates(tenant_id, framework);
CREATE INDEX idx_qt_tenant_status ON supplier_questionnaire_service.questionnaire_templates(tenant_id, status);
CREATE INDEX idx_qt_tags ON supplier_questionnaire_service.questionnaire_templates USING GIN (tags);

-- template_sections indexes
CREATE INDEX idx_ts_section_id ON supplier_questionnaire_service.template_sections(section_id);
CREATE INDEX idx_ts_template_id ON supplier_questionnaire_service.template_sections(template_id);
CREATE INDEX idx_ts_section_order ON supplier_questionnaire_service.template_sections(template_id, section_order);
CREATE INDEX idx_ts_created_at ON supplier_questionnaire_service.template_sections(created_at DESC);
CREATE INDEX idx_ts_translations ON supplier_questionnaire_service.template_sections USING GIN (translations);

-- template_questions indexes
CREATE INDEX idx_tq_question_id ON supplier_questionnaire_service.template_questions(question_id);
CREATE INDEX idx_tq_section_id ON supplier_questionnaire_service.template_questions(section_id);
CREATE INDEX idx_tq_template_id ON supplier_questionnaire_service.template_questions(template_id);
CREATE INDEX idx_tq_question_type ON supplier_questionnaire_service.template_questions(question_type);
CREATE INDEX idx_tq_required ON supplier_questionnaire_service.template_questions(required);
CREATE INDEX idx_tq_section_order ON supplier_questionnaire_service.template_questions(section_id, question_order);
CREATE INDEX idx_tq_template_section ON supplier_questionnaire_service.template_questions(template_id, section_id);
CREATE INDEX idx_tq_created_at ON supplier_questionnaire_service.template_questions(created_at DESC);
CREATE INDEX idx_tq_options ON supplier_questionnaire_service.template_questions USING GIN (options);
CREATE INDEX idx_tq_conditional_rules ON supplier_questionnaire_service.template_questions USING GIN (conditional_rules);
CREATE INDEX idx_tq_translations ON supplier_questionnaire_service.template_questions USING GIN (translations);

-- distributions indexes
CREATE INDEX idx_dist_distribution_id ON supplier_questionnaire_service.distributions(distribution_id);
CREATE INDEX idx_dist_template_id ON supplier_questionnaire_service.distributions(template_id);
CREATE INDEX idx_dist_supplier_id ON supplier_questionnaire_service.distributions(supplier_id);
CREATE INDEX idx_dist_supplier_name ON supplier_questionnaire_service.distributions(supplier_name);
CREATE INDEX idx_dist_campaign_id ON supplier_questionnaire_service.distributions(campaign_id);
CREATE INDEX idx_dist_status ON supplier_questionnaire_service.distributions(status);
CREATE INDEX idx_dist_channel ON supplier_questionnaire_service.distributions(channel);
CREATE INDEX idx_dist_deadline ON supplier_questionnaire_service.distributions(deadline);
CREATE INDEX idx_dist_template_status ON supplier_questionnaire_service.distributions(template_id, status);
CREATE INDEX idx_dist_tenant ON supplier_questionnaire_service.distributions(tenant_id);
CREATE INDEX idx_dist_created_at ON supplier_questionnaire_service.distributions(created_at DESC);
CREATE INDEX idx_dist_provenance ON supplier_questionnaire_service.distributions(provenance_hash);
CREATE INDEX idx_dist_tenant_template ON supplier_questionnaire_service.distributions(tenant_id, template_id);
CREATE INDEX idx_dist_tenant_supplier ON supplier_questionnaire_service.distributions(tenant_id, supplier_id);
CREATE INDEX idx_dist_tenant_campaign ON supplier_questionnaire_service.distributions(tenant_id, campaign_id);
CREATE INDEX idx_dist_tenant_status ON supplier_questionnaire_service.distributions(tenant_id, status);
CREATE INDEX idx_dist_tenant_deadline ON supplier_questionnaire_service.distributions(tenant_id, deadline);

-- distribution_events indexes (hypertable-aware)
CREATE INDEX idx_de_distribution_id ON supplier_questionnaire_service.distribution_events(distribution_id, created_at DESC);
CREATE INDEX idx_de_event_type ON supplier_questionnaire_service.distribution_events(event_type, created_at DESC);
CREATE INDEX idx_de_new_status ON supplier_questionnaire_service.distribution_events(new_status, created_at DESC);
CREATE INDEX idx_de_metadata ON supplier_questionnaire_service.distribution_events USING GIN (metadata);

-- responses indexes
CREATE INDEX idx_resp_response_id ON supplier_questionnaire_service.responses(response_id);
CREATE INDEX idx_resp_distribution_id ON supplier_questionnaire_service.responses(distribution_id);
CREATE INDEX idx_resp_template_id ON supplier_questionnaire_service.responses(template_id);
CREATE INDEX idx_resp_supplier_id ON supplier_questionnaire_service.responses(supplier_id);
CREATE INDEX idx_resp_supplier_name ON supplier_questionnaire_service.responses(supplier_name);
CREATE INDEX idx_resp_status ON supplier_questionnaire_service.responses(status);
CREATE INDEX idx_resp_template_status ON supplier_questionnaire_service.responses(template_id, status);
CREATE INDEX idx_resp_tenant ON supplier_questionnaire_service.responses(tenant_id);
CREATE INDEX idx_resp_created_at ON supplier_questionnaire_service.responses(created_at DESC);
CREATE INDEX idx_resp_updated_at ON supplier_questionnaire_service.responses(updated_at DESC);
CREATE INDEX idx_resp_provenance ON supplier_questionnaire_service.responses(provenance_hash);
CREATE INDEX idx_resp_tenant_template ON supplier_questionnaire_service.responses(tenant_id, template_id);
CREATE INDEX idx_resp_tenant_supplier ON supplier_questionnaire_service.responses(tenant_id, supplier_id);
CREATE INDEX idx_resp_tenant_status ON supplier_questionnaire_service.responses(tenant_id, status);
CREATE INDEX idx_resp_completion_pct ON supplier_questionnaire_service.responses(completion_pct);

-- response_answers indexes
CREATE INDEX idx_ra_answer_id ON supplier_questionnaire_service.response_answers(answer_id);
CREATE INDEX idx_ra_response_id ON supplier_questionnaire_service.response_answers(response_id);
CREATE INDEX idx_ra_question_id ON supplier_questionnaire_service.response_answers(question_id);
CREATE INDEX idx_ra_section_id ON supplier_questionnaire_service.response_answers(section_id);
CREATE INDEX idx_ra_response_question ON supplier_questionnaire_service.response_answers(response_id, question_id);
CREATE INDEX idx_ra_response_section ON supplier_questionnaire_service.response_answers(response_id, section_id);
CREATE INDEX idx_ra_created_at ON supplier_questionnaire_service.response_answers(created_at DESC);
CREATE INDEX idx_ra_answer_choices ON supplier_questionnaire_service.response_answers USING GIN (answer_choices);
CREATE INDEX idx_ra_file_attachments ON supplier_questionnaire_service.response_answers USING GIN (file_attachment_ids);

-- validation_results indexes (hypertable-aware)
CREATE INDEX idx_vr_check_id ON supplier_questionnaire_service.validation_results(check_id, created_at DESC);
CREATE INDEX idx_vr_response_id ON supplier_questionnaire_service.validation_results(response_id, created_at DESC);
CREATE INDEX idx_vr_level ON supplier_questionnaire_service.validation_results(level, created_at DESC);
CREATE INDEX idx_vr_result ON supplier_questionnaire_service.validation_results(result, created_at DESC);
CREATE INDEX idx_vr_severity ON supplier_questionnaire_service.validation_results(severity, created_at DESC);
CREATE INDEX idx_vr_check_name ON supplier_questionnaire_service.validation_results(check_name, created_at DESC);
CREATE INDEX idx_vr_response_level ON supplier_questionnaire_service.validation_results(response_id, level, created_at DESC);
CREATE INDEX idx_vr_response_result ON supplier_questionnaire_service.validation_results(response_id, result, created_at DESC);

-- scores indexes
CREATE INDEX idx_sc_score_id ON supplier_questionnaire_service.scores(score_id);
CREATE INDEX idx_sc_response_id ON supplier_questionnaire_service.scores(response_id);
CREATE INDEX idx_sc_template_id ON supplier_questionnaire_service.scores(template_id);
CREATE INDEX idx_sc_supplier_id ON supplier_questionnaire_service.scores(supplier_id);
CREATE INDEX idx_sc_framework ON supplier_questionnaire_service.scores(framework);
CREATE INDEX idx_sc_performance_tier ON supplier_questionnaire_service.scores(performance_tier);
CREATE INDEX idx_sc_overall_score ON supplier_questionnaire_service.scores(overall_score DESC);
CREATE INDEX idx_sc_supplier_framework ON supplier_questionnaire_service.scores(supplier_id, framework);
CREATE INDEX idx_sc_tenant ON supplier_questionnaire_service.scores(tenant_id);
CREATE INDEX idx_sc_scored_at ON supplier_questionnaire_service.scores(scored_at DESC);
CREATE INDEX idx_sc_provenance ON supplier_questionnaire_service.scores(provenance_hash);
CREATE INDEX idx_sc_tenant_supplier ON supplier_questionnaire_service.scores(tenant_id, supplier_id);
CREATE INDEX idx_sc_tenant_framework ON supplier_questionnaire_service.scores(tenant_id, framework);
CREATE INDEX idx_sc_tenant_tier ON supplier_questionnaire_service.scores(tenant_id, performance_tier);
CREATE INDEX idx_sc_tenant_template ON supplier_questionnaire_service.scores(tenant_id, template_id);
CREATE INDEX idx_sc_section_scores ON supplier_questionnaire_service.scores USING GIN (section_scores);

-- follow_up_actions indexes (hypertable-aware)
CREATE INDEX idx_fua_action_id ON supplier_questionnaire_service.follow_up_actions(action_id, created_at DESC);
CREATE INDEX idx_fua_distribution_id ON supplier_questionnaire_service.follow_up_actions(distribution_id, created_at DESC);
CREATE INDEX idx_fua_campaign_id ON supplier_questionnaire_service.follow_up_actions(campaign_id, created_at DESC);
CREATE INDEX idx_fua_supplier_id ON supplier_questionnaire_service.follow_up_actions(supplier_id, created_at DESC);
CREATE INDEX idx_fua_reminder_type ON supplier_questionnaire_service.follow_up_actions(reminder_type, created_at DESC);
CREATE INDEX idx_fua_escalation_level ON supplier_questionnaire_service.follow_up_actions(escalation_level, created_at DESC);
CREATE INDEX idx_fua_scheduled_at ON supplier_questionnaire_service.follow_up_actions(scheduled_at, created_at DESC);
CREATE INDEX idx_fua_status ON supplier_questionnaire_service.follow_up_actions(status, created_at DESC);
CREATE INDEX idx_fua_status_scheduled ON supplier_questionnaire_service.follow_up_actions(status, scheduled_at);
CREATE INDEX idx_fua_supplier_status ON supplier_questionnaire_service.follow_up_actions(supplier_id, status, created_at DESC);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE supplier_questionnaire_service.questionnaire_templates ENABLE ROW LEVEL SECURITY;
CREATE POLICY qt_tenant_read ON supplier_questionnaire_service.questionnaire_templates
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY qt_tenant_write ON supplier_questionnaire_service.questionnaire_templates
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE supplier_questionnaire_service.template_sections ENABLE ROW LEVEL SECURITY;
CREATE POLICY ts_tenant_read ON supplier_questionnaire_service.template_sections
    FOR SELECT USING (TRUE);
CREATE POLICY ts_tenant_write ON supplier_questionnaire_service.template_sections
    FOR ALL USING (TRUE);

ALTER TABLE supplier_questionnaire_service.template_questions ENABLE ROW LEVEL SECURITY;
CREATE POLICY tq_tenant_read ON supplier_questionnaire_service.template_questions
    FOR SELECT USING (TRUE);
CREATE POLICY tq_tenant_write ON supplier_questionnaire_service.template_questions
    FOR ALL USING (TRUE);

ALTER TABLE supplier_questionnaire_service.distributions ENABLE ROW LEVEL SECURITY;
CREATE POLICY dist_tenant_read ON supplier_questionnaire_service.distributions
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY dist_tenant_write ON supplier_questionnaire_service.distributions
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE supplier_questionnaire_service.distribution_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY de_tenant_read ON supplier_questionnaire_service.distribution_events
    FOR SELECT USING (TRUE);
CREATE POLICY de_tenant_write ON supplier_questionnaire_service.distribution_events
    FOR ALL USING (TRUE);

ALTER TABLE supplier_questionnaire_service.responses ENABLE ROW LEVEL SECURITY;
CREATE POLICY resp_tenant_read ON supplier_questionnaire_service.responses
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY resp_tenant_write ON supplier_questionnaire_service.responses
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE supplier_questionnaire_service.response_answers ENABLE ROW LEVEL SECURITY;
CREATE POLICY ra_tenant_read ON supplier_questionnaire_service.response_answers
    FOR SELECT USING (TRUE);
CREATE POLICY ra_tenant_write ON supplier_questionnaire_service.response_answers
    FOR ALL USING (TRUE);

ALTER TABLE supplier_questionnaire_service.validation_results ENABLE ROW LEVEL SECURITY;
CREATE POLICY vr_tenant_read ON supplier_questionnaire_service.validation_results
    FOR SELECT USING (TRUE);
CREATE POLICY vr_tenant_write ON supplier_questionnaire_service.validation_results
    FOR ALL USING (TRUE);

ALTER TABLE supplier_questionnaire_service.scores ENABLE ROW LEVEL SECURITY;
CREATE POLICY sc_tenant_read ON supplier_questionnaire_service.scores
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY sc_tenant_write ON supplier_questionnaire_service.scores
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE supplier_questionnaire_service.follow_up_actions ENABLE ROW LEVEL SECURITY;
CREATE POLICY fua_tenant_read ON supplier_questionnaire_service.follow_up_actions
    FOR SELECT USING (TRUE);
CREATE POLICY fua_tenant_write ON supplier_questionnaire_service.follow_up_actions
    FOR ALL USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA supplier_questionnaire_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA supplier_questionnaire_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA supplier_questionnaire_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON supplier_questionnaire_service.supplier_quest_distribution_hourly TO greenlang_app;
GRANT SELECT ON supplier_questionnaire_service.supplier_quest_validation_hourly TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA supplier_questionnaire_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA supplier_questionnaire_service TO greenlang_readonly;
GRANT SELECT ON supplier_questionnaire_service.supplier_quest_distribution_hourly TO greenlang_readonly;
GRANT SELECT ON supplier_questionnaire_service.supplier_quest_validation_hourly TO greenlang_readonly;

-- Add supplier questionnaire service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'supplier_questionnaire:templates:read', 'supplier_questionnaire', 'templates_read', 'View questionnaire templates, sections, and questions'),
    (gen_random_uuid(), 'supplier_questionnaire:templates:write', 'supplier_questionnaire', 'templates_write', 'Create and manage questionnaire templates'),
    (gen_random_uuid(), 'supplier_questionnaire:distributions:read', 'supplier_questionnaire', 'distributions_read', 'View questionnaire distributions and delivery status'),
    (gen_random_uuid(), 'supplier_questionnaire:distributions:write', 'supplier_questionnaire', 'distributions_write', 'Create and manage questionnaire distributions'),
    (gen_random_uuid(), 'supplier_questionnaire:responses:read', 'supplier_questionnaire', 'responses_read', 'View supplier responses and answer data'),
    (gen_random_uuid(), 'supplier_questionnaire:responses:write', 'supplier_questionnaire', 'responses_write', 'Create and manage supplier responses'),
    (gen_random_uuid(), 'supplier_questionnaire:validation:read', 'supplier_questionnaire', 'validation_read', 'View response validation results and data quality scores'),
    (gen_random_uuid(), 'supplier_questionnaire:validation:write', 'supplier_questionnaire', 'validation_write', 'Execute and manage response validation checks'),
    (gen_random_uuid(), 'supplier_questionnaire:scores:read', 'supplier_questionnaire', 'scores_read', 'View supplier performance scores and benchmarks'),
    (gen_random_uuid(), 'supplier_questionnaire:scores:write', 'supplier_questionnaire', 'scores_write', 'Calculate and manage supplier performance scores'),
    (gen_random_uuid(), 'supplier_questionnaire:followup:read', 'supplier_questionnaire', 'followup_read', 'View follow-up actions and reminder schedules'),
    (gen_random_uuid(), 'supplier_questionnaire:followup:write', 'supplier_questionnaire', 'followup_write', 'Create and manage follow-up actions and escalations'),
    (gen_random_uuid(), 'supplier_questionnaire:campaigns:read', 'supplier_questionnaire', 'campaigns_read', 'View campaign statistics and supplier engagement metrics'),
    (gen_random_uuid(), 'supplier_questionnaire:admin', 'supplier_questionnaire', 'admin', 'Supplier questionnaire service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep distribution event records for 90 days
SELECT add_retention_policy('supplier_questionnaire_service.distribution_events', INTERVAL '90 days');

-- Keep validation result records for 90 days
SELECT add_retention_policy('supplier_questionnaire_service.validation_results', INTERVAL '90 days');

-- Keep follow-up action records for 90 days
SELECT add_retention_policy('supplier_questionnaire_service.follow_up_actions', INTERVAL '90 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on distribution_events after 7 days
ALTER TABLE supplier_questionnaire_service.distribution_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'distribution_id',
    timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_compression_policy('supplier_questionnaire_service.distribution_events', INTERVAL '7 days');

-- Enable compression on validation_results after 7 days
ALTER TABLE supplier_questionnaire_service.validation_results SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'response_id',
    timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_compression_policy('supplier_questionnaire_service.validation_results', INTERVAL '7 days');

-- Enable compression on follow_up_actions after 7 days
ALTER TABLE supplier_questionnaire_service.follow_up_actions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'distribution_id',
    timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_compression_policy('supplier_questionnaire_service.follow_up_actions', INTERVAL '7 days');

-- =============================================================================
-- Seed: Register the Supplier Questionnaire Processor Agent (GL-DATA-SUP-001)
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES
('GL-DATA-SUP-001', 'Supplier Questionnaire Processor',
 'Processes supplier sustainability questionnaires for GreenLang Climate OS. Manages questionnaire template creation with multi-framework support (CDP, GRI, SASB, EcoVadis, TCFD, SBTi, ISO 14001, GHG Protocol, CSRD), distributes questionnaires to suppliers via email/portal/API channels, tracks distribution lifecycle events, collects and validates supplier responses with field/section/cross-field validation checks, scores supplier performance with framework-specific benchmarking and tier classification (leader/advanced/intermediate/beginner), and orchestrates follow-up reminders with escalation levels.',
 2, 'async', true, true, 5, '1.0.0', true,
 'GreenLang Data Team', 'https://docs.greenlang.ai/agents/supplier-questionnaire', true, 'default')
ON CONFLICT (agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Agent Version for Supplier Questionnaire Processor
-- =============================================================================

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES
('GL-DATA-SUP-001', '1.0.0',
 '{"cpu_request": "250m", "cpu_limit": "1000m", "memory_request": "512Mi", "memory_limit": "2Gi", "gpu": false}'::jsonb,
 '{"image": "greenlang/supplier-questionnaire-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"supplier", "questionnaire", "survey", "cdp", "gri", "sasb", "ecovadis", "scoring", "engagement"}',
 '{"cross-sector", "manufacturing", "retail", "energy"}',
 'd4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5')
ON CONFLICT (agent_id, version) DO NOTHING;

-- =============================================================================
-- Seed: Agent Capabilities for Supplier Questionnaire Processor
-- =============================================================================

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES

('GL-DATA-SUP-001', '1.0.0', 'template_management', 'data_management',
 'Create and manage questionnaire templates with multi-framework support, versioned sections and questions, conditional logic, scoring weights, and multi-language translations',
 '{"framework", "name", "sections", "questions"}', '{"template_id", "version", "total_questions"}',
 '{"supported_frameworks": ["cdp", "gri", "sasb", "ecovadis", "tcfd", "sbti", "iso14001", "ghg_protocol", "csrd", "custom"], "question_types": ["text", "numeric", "single_choice", "multi_choice", "date", "file_upload", "boolean", "scale"], "max_sections": 50, "max_questions_per_section": 100}'::jsonb),

('GL-DATA-SUP-001', '1.0.0', 'questionnaire_distribution', 'distribution',
 'Distribute questionnaires to suppliers via email, portal, or API with campaign grouping, deadline management, access token generation, and delivery lifecycle tracking',
 '{"template_id", "suppliers", "channel", "deadline"}', '{"distribution_ids", "sent_count", "failed_count"}',
 '{"channels": ["email", "portal", "api", "bulk_email", "manual"], "batch_size": 500, "retry_on_bounce": true, "access_token_expiry_days": 90}'::jsonb),

('GL-DATA-SUP-001', '1.0.0', 'response_collection', 'ingestion',
 'Collect and store supplier responses with answer validation, completion tracking, version management for re-submissions, and file attachment support',
 '{"distribution_id", "answers"}', '{"response_id", "completion_pct", "validation_status"}',
 '{"auto_save": true, "max_file_size_mb": 50, "supported_file_types": ["pdf", "xlsx", "csv", "docx", "png", "jpg"]}'::jsonb),

('GL-DATA-SUP-001', '1.0.0', 'response_validation', 'validation',
 'Validate supplier responses at field, section, response, and cross-field levels with data quality scoring, severity classification, and suggested corrections',
 '{"response_id"}', '{"check_count", "pass_count", "fail_count", "data_quality_score"}',
 '{"validation_levels": ["field", "section", "response", "cross_field", "completeness", "consistency"], "severities": ["critical", "error", "warning", "info"], "auto_validate_on_submit": true}'::jsonb),

('GL-DATA-SUP-001', '1.0.0', 'supplier_scoring', 'computation',
 'Score supplier sustainability performance using framework-specific algorithms with tier classification, CDP letter grading, section-level scoring, confidence assessment, and benchmark percentile ranking',
 '{"response_id", "framework"}', '{"score_id", "overall_score", "performance_tier", "cdp_grade"}',
 '{"performance_tiers": ["leader", "advanced", "intermediate", "beginner", "non_responsive"], "cdp_grades": ["A", "A-", "B", "B-", "C", "C-", "D", "D-", "F"], "benchmark_enabled": true}'::jsonb),

('GL-DATA-SUP-001', '1.0.0', 'follow_up_orchestration', 'orchestration',
 'Orchestrate follow-up reminders and escalations for non-responsive suppliers with configurable reminder types, escalation levels, scheduling, and message templates',
 '{"distribution_id", "reminder_type", "escalation_level"}', '{"action_id", "scheduled_at", "status"}',
 '{"reminder_types": ["reminder", "escalation", "final_notice", "overdue"], "escalation_levels": ["level_1", "level_2", "level_3", "executive"], "default_intervals_days": [7, 14, 21, 30]}'::jsonb),

('GL-DATA-SUP-001', '1.0.0', 'campaign_analytics', 'reporting',
 'Analyze campaign performance with response rates, completion metrics, average scores, supplier engagement trends, and benchmark comparisons across distributions',
 '{"campaign_id", "date_range"}', '{"response_rate", "avg_completion_pct", "avg_score", "tier_distribution"}',
 '{"metrics": ["response_rate", "completion_rate", "avg_score", "tier_distribution", "time_to_complete", "bounce_rate"], "export_formats": ["json", "csv", "xlsx"]}'::jsonb)

ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: Agent Dependencies for Supplier Questionnaire Processor
-- =============================================================================

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES

-- Supplier Questionnaire depends on Schema Compiler for input/output validation
('GL-DATA-SUP-001', 'GL-FOUND-X-002', '>=1.0.0', false,
 'Questionnaire templates, responses, and scores are validated against JSON Schema definitions'),

-- Supplier Questionnaire depends on Registry for agent discovery
('GL-DATA-SUP-001', 'GL-FOUND-X-007', '>=1.0.0', false,
 'Agent version and capability lookup for questionnaire processing pipeline orchestration'),

-- Supplier Questionnaire depends on Access Guard for policy enforcement
('GL-DATA-SUP-001', 'GL-FOUND-X-006', '>=1.0.0', false,
 'Data classification and access control enforcement for supplier response data and scores'),

-- Supplier Questionnaire depends on Observability Agent for metrics
('GL-DATA-SUP-001', 'GL-FOUND-X-010', '>=1.0.0', false,
 'Distribution metrics, response collection statistics, and scoring telemetry are reported to observability'),

-- Supplier Questionnaire optionally uses Citations for provenance tracking
('GL-DATA-SUP-001', 'GL-FOUND-X-005', '>=1.0.0', true,
 'Response data and scoring provenance chains are registered with the citation service for audit trail'),

-- Supplier Questionnaire optionally uses Reproducibility for determinism
('GL-DATA-SUP-001', 'GL-FOUND-X-008', '>=1.0.0', true,
 'Supplier scoring calculations are verified for reproducibility across re-execution'),

-- Supplier Questionnaire optionally integrates with Excel Normalizer
('GL-DATA-SUP-001', 'GL-DATA-X-002', '>=1.0.0', true,
 'Supplier-uploaded Excel/CSV attachments are normalized through the Excel Normalizer agent'),

-- Supplier Questionnaire optionally integrates with PDF Extractor
('GL-DATA-SUP-001', 'GL-DATA-X-001', '>=1.0.0', true,
 'Supplier-uploaded PDF evidence documents are processed through the PDF Extractor agent')

ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Service Catalog Entry for Supplier Questionnaire Processor
-- =============================================================================

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES
('GL-DATA-SUP-001', 'Supplier Questionnaire Processor',
 'End-to-end supplier sustainability questionnaire management. Creates multi-framework templates (CDP, GRI, SASB, EcoVadis, TCFD, SBTi), distributes to suppliers via email/portal/API, collects and validates responses with data quality scoring, calculates supplier performance scores with tier classification and benchmarking, and orchestrates follow-up reminders with escalation. SHA-256 provenance chains for audit trail.',
 'data', 'active', 'default')
ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA supplier_questionnaire_service IS 'Supplier Questionnaire Processor for GreenLang Climate OS (AGENT-DATA-008) - questionnaire template management, supplier distribution, response collection, validation, performance scoring, and follow-up orchestration with multi-framework support';
COMMENT ON TABLE supplier_questionnaire_service.questionnaire_templates IS 'Questionnaire template definitions with framework (CDP/GRI/SASB/EcoVadis/TCFD/SBTi/ISO14001/GHG Protocol/CSRD/custom), versioning, language, status, tags, and provenance hash';
COMMENT ON TABLE supplier_questionnaire_service.template_sections IS 'Ordered sections within questionnaire templates with section name, description, display order, minimum required answers, and multi-language translations';
COMMENT ON TABLE supplier_questionnaire_service.template_questions IS 'Individual questions within template sections with question type (text/numeric/single_choice/multi_choice/date/file_upload/boolean/scale), conditional rules, scoring weight, and translations';
COMMENT ON TABLE supplier_questionnaire_service.distributions IS 'Questionnaire distribution records tracking supplier delivery lifecycle with channel (email/portal/api), campaign grouping, access token, deadline, and event timestamps (sent/delivered/opened/started/completed)';
COMMENT ON TABLE supplier_questionnaire_service.distribution_events IS 'TimescaleDB hypertable: distribution lifecycle event log with event type, status transitions, and metadata for engagement tracking';
COMMENT ON TABLE supplier_questionnaire_service.responses IS 'Supplier response records with completion percentage, version tracking for re-submissions, submission and validation timestamps, and provenance hash';
COMMENT ON TABLE supplier_questionnaire_service.response_answers IS 'Individual answers to questionnaire questions with text value, numeric value, selected choices, file attachments, confidence score, and notes';
COMMENT ON TABLE supplier_questionnaire_service.validation_results IS 'TimescaleDB hypertable: response validation check results with level (field/section/response/cross_field), result (pass/fail/warning), severity, field path, suggestion, and data quality score';
COMMENT ON TABLE supplier_questionnaire_service.scores IS 'Supplier performance scores with framework-specific overall score, performance tier (leader/advanced/intermediate/beginner), CDP grade, section scores, confidence, benchmark percentile, and provenance hash';
COMMENT ON TABLE supplier_questionnaire_service.follow_up_actions IS 'TimescaleDB hypertable: follow-up reminder and escalation actions with reminder type, escalation level (level_1/level_2/level_3/executive), scheduling, delivery status, and message template';
COMMENT ON MATERIALIZED VIEW supplier_questionnaire_service.supplier_quest_distribution_hourly IS 'Continuous aggregate: hourly distribution event statistics by event type with counts, unique distributions, and status breakdown for campaign monitoring';
COMMENT ON MATERIALIZED VIEW supplier_questionnaire_service.supplier_quest_validation_hourly IS 'Continuous aggregate: hourly validation result statistics by level and result with severity breakdown and average data quality score for quality monitoring';

COMMENT ON COLUMN supplier_questionnaire_service.questionnaire_templates.framework IS 'Sustainability framework: cdp, gri, sasb, ecovadis, tcfd, sbti, iso14001, ghg_protocol, csrd, custom';
COMMENT ON COLUMN supplier_questionnaire_service.questionnaire_templates.status IS 'Template publication status: draft, active, archived, deprecated, review';
COMMENT ON COLUMN supplier_questionnaire_service.questionnaire_templates.provenance_hash IS 'SHA-256 provenance hash of template content for integrity verification';
COMMENT ON COLUMN supplier_questionnaire_service.template_questions.question_type IS 'Question type: text, numeric, single_choice, multi_choice, date, file_upload, boolean, scale, textarea';
COMMENT ON COLUMN supplier_questionnaire_service.template_questions.score_weight IS 'Relative scoring weight for this question (default 1.0) used in performance score calculation';
COMMENT ON COLUMN supplier_questionnaire_service.template_questions.conditional_rules IS 'JSONB conditional display rules determining when this question is shown based on previous answers';
COMMENT ON COLUMN supplier_questionnaire_service.distributions.channel IS 'Distribution channel: email, portal, api, bulk_email, manual';
COMMENT ON COLUMN supplier_questionnaire_service.distributions.status IS 'Distribution lifecycle status: queued, sent, delivered, opened, started, completed, expired, bounced, failed, cancelled';
COMMENT ON COLUMN supplier_questionnaire_service.distributions.access_token IS 'Secure access token for supplier portal authentication (generated per distribution)';
COMMENT ON COLUMN supplier_questionnaire_service.distributions.provenance_hash IS 'SHA-256 provenance hash of distribution parameters for audit trail';
COMMENT ON COLUMN supplier_questionnaire_service.distribution_events.event_type IS 'Distribution event type: created, queued, sent, delivered, opened, started, submitted, reminder_sent, expired, bounced, failed, cancelled, reopened, escalated, status_change';
COMMENT ON COLUMN supplier_questionnaire_service.responses.completion_pct IS 'Response completion percentage (0-100) based on answered required questions';
COMMENT ON COLUMN supplier_questionnaire_service.responses.version IS 'Response version number for tracking re-submissions and revisions';
COMMENT ON COLUMN supplier_questionnaire_service.responses.provenance_hash IS 'SHA-256 provenance hash of response content for integrity verification';
COMMENT ON COLUMN supplier_questionnaire_service.response_answers.confidence_score IS 'Answer confidence score (0-1) indicating reliability of the provided answer';
COMMENT ON COLUMN supplier_questionnaire_service.validation_results.level IS 'Validation level: field, section, response, cross_field, completeness, consistency';
COMMENT ON COLUMN supplier_questionnaire_service.validation_results.result IS 'Validation result: pass, fail, warning, skip, error, info';
COMMENT ON COLUMN supplier_questionnaire_service.validation_results.severity IS 'Validation severity: critical, error, warning, info';
COMMENT ON COLUMN supplier_questionnaire_service.validation_results.data_quality_score IS 'Data quality score (0-100) for the validated field or section';
COMMENT ON COLUMN supplier_questionnaire_service.scores.performance_tier IS 'Supplier performance tier: leader, advanced, intermediate, beginner, non_responsive, insufficient_data';
COMMENT ON COLUMN supplier_questionnaire_service.scores.cdp_grade IS 'CDP letter grade (A, A-, B, B-, C, C-, D, D-, F) for CDP framework scoring';
COMMENT ON COLUMN supplier_questionnaire_service.scores.benchmark_percentile IS 'Benchmark percentile ranking (0-100) compared to peer suppliers in the same framework';
COMMENT ON COLUMN supplier_questionnaire_service.scores.provenance_hash IS 'SHA-256 provenance hash of scoring inputs and results for audit trail';
COMMENT ON COLUMN supplier_questionnaire_service.follow_up_actions.reminder_type IS 'Reminder type: reminder, escalation, final_notice, overdue';
COMMENT ON COLUMN supplier_questionnaire_service.follow_up_actions.escalation_level IS 'Escalation level: level_1 (procurement), level_2 (sustainability team), level_3 (management), executive';
COMMENT ON COLUMN supplier_questionnaire_service.follow_up_actions.status IS 'Follow-up action status: pending, sent, delivered, failed, cancelled, skipped';
