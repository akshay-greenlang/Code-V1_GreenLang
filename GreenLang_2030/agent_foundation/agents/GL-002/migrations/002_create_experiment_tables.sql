-- Migration: Create A/B Testing Experiment Tables
-- Version: 002
-- Description: Creates tables for A/B testing experiments and analytics
-- Author: GL-BackendDeveloper
-- Date: 2025-11-17

-- ============================================================================
-- EXPERIMENT TABLES
-- ============================================================================

-- Main experiments table
CREATE TABLE IF NOT EXISTS experiments (
    id BIGSERIAL PRIMARY KEY,
    experiment_id VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    hypothesis TEXT NOT NULL,

    -- Variant configuration (stored as JSONB)
    variants JSONB NOT NULL,

    -- Metrics configuration
    primary_metric VARCHAR(100) NOT NULL,
    primary_metric_type VARCHAR(50) NOT NULL,
    secondary_metrics TEXT[] DEFAULT '{}',

    -- Statistical configuration
    min_sample_size INTEGER NOT NULL DEFAULT 100,
    significance_level DECIMAL(4, 3) NOT NULL DEFAULT 0.05,
    power DECIMAL(4, 3) NOT NULL DEFAULT 0.8,

    -- Lifecycle
    status VARCHAR(20) NOT NULL DEFAULT 'draft',
    start_date TIMESTAMP WITH TIME ZONE,
    end_date TIMESTAMP WITH TIME ZONE,
    duration_days INTEGER,

    -- Metadata
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    tags TEXT[] DEFAULT '{}'
);

-- Indexes for experiments
CREATE INDEX IF NOT EXISTS idx_experiments_status
    ON experiments(status);

CREATE INDEX IF NOT EXISTS idx_experiments_created
    ON experiments(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_experiments_name
    ON experiments(name);

CREATE INDEX IF NOT EXISTS idx_experiments_tags
    ON experiments USING GIN (tags);

-- Check constraint for valid status
ALTER TABLE experiments
    ADD CONSTRAINT check_experiment_status
    CHECK (status IN ('draft', 'running', 'paused', 'completed', 'archived'));


-- Experiment metrics table (raw data)
CREATE TABLE IF NOT EXISTS experiment_metrics (
    id BIGSERIAL PRIMARY KEY,
    experiment_id VARCHAR(255) NOT NULL,
    variant_name VARCHAR(100) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15, 4) NOT NULL,
    metadata JSONB DEFAULT '{}',
    recorded_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Foreign key to experiments
    CONSTRAINT fk_experiment
        FOREIGN KEY (experiment_id)
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE
);

-- Indexes for metrics
CREATE INDEX IF NOT EXISTS idx_metrics_experiment
    ON experiment_metrics(experiment_id);

CREATE INDEX IF NOT EXISTS idx_metrics_variant
    ON experiment_metrics(experiment_id, variant_name);

CREATE INDEX IF NOT EXISTS idx_metrics_user
    ON experiment_metrics(user_id);

CREATE INDEX IF NOT EXISTS idx_metrics_recorded
    ON experiment_metrics(recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_metrics_name
    ON experiment_metrics(metric_name);

-- Composite index for aggregations
CREATE INDEX IF NOT EXISTS idx_metrics_aggregation
    ON experiment_metrics(experiment_id, variant_name, metric_name);


-- User variant assignments table
CREATE TABLE IF NOT EXISTS experiment_assignments (
    id BIGSERIAL PRIMARY KEY,
    experiment_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    variant_name VARCHAR(100) NOT NULL,
    assigned_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    session_id VARCHAR(255),
    assignment_hash VARCHAR(64) NOT NULL,

    -- Ensure consistent assignment
    CONSTRAINT unique_assignment UNIQUE (experiment_id, user_id),

    -- Foreign key
    CONSTRAINT fk_assignment_experiment
        FOREIGN KEY (experiment_id)
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE
);

-- Indexes for assignments
CREATE INDEX IF NOT EXISTS idx_assignments_experiment
    ON experiment_assignments(experiment_id);

CREATE INDEX IF NOT EXISTS idx_assignments_user
    ON experiment_assignments(user_id);

CREATE INDEX IF NOT EXISTS idx_assignments_variant
    ON experiment_assignments(experiment_id, variant_name);


-- Experiment results summary table
CREATE TABLE IF NOT EXISTS experiment_results (
    id BIGSERIAL PRIMARY KEY,
    experiment_id VARCHAR(255) NOT NULL UNIQUE,

    -- Winner information
    winner_variant VARCHAR(100),
    winner_confidence VARCHAR(20),
    winner_improvement DECIMAL(8, 2),

    -- Overall statistics
    total_samples INTEGER NOT NULL DEFAULT 0,
    is_conclusive BOOLEAN DEFAULT FALSE,
    days_running INTEGER,

    -- Analysis results
    statistical_tests JSONB DEFAULT '[]',
    variant_metrics JSONB DEFAULT '[]',

    -- Recommendation
    final_recommendation TEXT NOT NULL,
    key_insights TEXT[] DEFAULT '{}',

    -- Metadata
    analyzed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    provenance_hash VARCHAR(64) NOT NULL,

    -- Foreign key
    CONSTRAINT fk_result_experiment
        FOREIGN KEY (experiment_id)
        REFERENCES experiments(experiment_id)
        ON DELETE CASCADE
);

-- Index for results
CREATE INDEX IF NOT EXISTS idx_results_experiment
    ON experiment_results(experiment_id);

CREATE INDEX IF NOT EXISTS idx_results_conclusive
    ON experiment_results(is_conclusive);

CREATE INDEX IF NOT EXISTS idx_results_analyzed
    ON experiment_results(analyzed_at DESC);


-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Trigger for updated_at
CREATE TRIGGER update_experiments_updated_at
    BEFORE UPDATE ON experiments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


-- Function to validate experiment variants
CREATE OR REPLACE FUNCTION validate_experiment_variants()
RETURNS TRIGGER AS $$
DECLARE
    variants_array JSONB;
    total_traffic DECIMAL;
    variant JSONB;
    control_count INTEGER := 0;
BEGIN
    variants_array := NEW.variants;

    -- Check minimum 2 variants
    IF jsonb_array_length(variants_array) < 2 THEN
        RAISE EXCEPTION 'Experiment must have at least 2 variants';
    END IF;

    -- Check traffic splits sum to 1.0
    total_traffic := 0;
    FOR variant IN SELECT * FROM jsonb_array_elements(variants_array) LOOP
        total_traffic := total_traffic + (variant->>'traffic_split')::DECIMAL;

        -- Count control variants
        IF (variant->>'is_control')::BOOLEAN THEN
            control_count := control_count + 1;
        END IF;
    END LOOP;

    IF total_traffic < 0.99 OR total_traffic > 1.01 THEN
        RAISE EXCEPTION 'Variant traffic splits must sum to 1.0, got %', total_traffic;
    END IF;

    -- Ensure exactly one control
    IF control_count != 1 THEN
        RAISE EXCEPTION 'Experiment must have exactly one control variant, got %', control_count;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to validate variants
CREATE TRIGGER validate_variants
    BEFORE INSERT OR UPDATE ON experiments
    FOR EACH ROW EXECUTE FUNCTION validate_experiment_variants();


-- Function to auto-update experiment status based on dates
CREATE OR REPLACE FUNCTION auto_update_experiment_status()
RETURNS TRIGGER AS $$
BEGIN
    -- If start_date is set and in the past, ensure status is running
    IF NEW.start_date IS NOT NULL AND NEW.start_date <= NOW() AND NEW.status = 'draft' THEN
        NEW.status := 'running';
    END IF;

    -- If end_date is set and in the past, mark as completed
    IF NEW.end_date IS NOT NULL AND NEW.end_date <= NOW() AND NEW.status = 'running' THEN
        NEW.status := 'completed';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for auto-status update
CREATE TRIGGER auto_status_update
    BEFORE INSERT OR UPDATE ON experiments
    FOR EACH ROW EXECUTE FUNCTION auto_update_experiment_status();


-- ============================================================================
-- MATERIALIZED VIEWS FOR ANALYTICS
-- ============================================================================

-- Experiment performance summary
CREATE MATERIALIZED VIEW IF NOT EXISTS experiment_performance AS
SELECT
    e.experiment_id,
    e.name,
    e.status,
    e.start_date,
    e.end_date,
    COUNT(DISTINCT a.user_id) AS total_users,
    COUNT(DISTINCT m.user_id) AS users_with_metrics,
    COALESCE(r.is_conclusive, FALSE) AS is_conclusive,
    r.winner_variant,
    r.winner_improvement,
    e.created_at
FROM experiments e
LEFT JOIN experiment_assignments a ON e.experiment_id = a.experiment_id
LEFT JOIN experiment_metrics m ON e.experiment_id = m.experiment_id
LEFT JOIN experiment_results r ON e.experiment_id = r.experiment_id
GROUP BY
    e.experiment_id, e.name, e.status, e.start_date, e.end_date,
    e.created_at, r.is_conclusive, r.winner_variant, r.winner_improvement
ORDER BY e.created_at DESC;


-- Variant performance comparison
CREATE MATERIALIZED VIEW IF NOT EXISTS variant_performance AS
SELECT
    em.experiment_id,
    em.variant_name,
    COUNT(DISTINCT em.user_id) AS sample_size,
    AVG(em.metric_value) AS mean_value,
    STDDEV(em.metric_value) AS std_value,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY em.metric_value) AS median_value,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY em.metric_value) AS q1_value,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY em.metric_value) AS q3_value,
    MIN(em.metric_value) AS min_value,
    MAX(em.metric_value) AS max_value
FROM experiment_metrics em
GROUP BY em.experiment_id, em.variant_name
ORDER BY em.experiment_id, em.variant_name;


-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to get variant statistics
CREATE OR REPLACE FUNCTION get_variant_stats(
    p_experiment_id VARCHAR,
    p_variant_name VARCHAR
)
RETURNS TABLE (
    sample_size BIGINT,
    mean_value DECIMAL,
    std_value DECIMAL,
    median_value DECIMAL,
    ci_lower DECIMAL,
    ci_upper DECIMAL
) AS $$
DECLARE
    t_critical DECIMAL := 1.96; -- 95% CI for large samples
    std_error DECIMAL;
BEGIN
    RETURN QUERY
    SELECT
        COUNT(DISTINCT user_id),
        AVG(metric_value),
        STDDEV(metric_value),
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY metric_value),
        AVG(metric_value) - (t_critical * STDDEV(metric_value) / SQRT(COUNT(*))),
        AVG(metric_value) + (t_critical * STDDEV(metric_value) / SQRT(COUNT(*)))
    FROM experiment_metrics
    WHERE experiment_id = p_experiment_id
      AND variant_name = p_variant_name;
END;
$$ LANGUAGE plpgsql;


-- Function to get assignment distribution
CREATE OR REPLACE FUNCTION get_assignment_distribution(
    p_experiment_id VARCHAR
)
RETURNS TABLE (
    variant_name VARCHAR,
    assignment_count BIGINT,
    percentage DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ea.variant_name,
        COUNT(*),
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER ()
    FROM experiment_assignments ea
    WHERE ea.experiment_id = p_experiment_id
    GROUP BY ea.variant_name
    ORDER BY COUNT(*) DESC;
END;
$$ LANGUAGE plpgsql;


-- Function to refresh all experiment materialized views
CREATE OR REPLACE FUNCTION refresh_experiment_views()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY experiment_performance;
    REFRESH MATERIALIZED VIEW CONCURRENTLY variant_performance;
    RAISE NOTICE 'Experiment views refreshed successfully';
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- PARTITIONING FOR LARGE-SCALE EXPERIMENTS
-- ============================================================================

-- Create partitioned table for metrics (if expecting high volume)
-- Partition by experiment_id range or by date

-- CREATE TABLE experiment_metrics_partitioned (
--     LIKE experiment_metrics INCLUDING ALL
-- ) PARTITION BY RANGE (recorded_at);

-- CREATE TABLE experiment_metrics_2025_q4 PARTITION OF experiment_metrics_partitioned
--     FOR VALUES FROM ('2025-10-01') TO ('2026-01-01');


-- ============================================================================
-- GRANTS AND PERMISSIONS
-- ============================================================================

-- Grant permissions (adjust user as needed)
-- GRANT SELECT, INSERT, UPDATE ON experiments TO gl002_app_user;
-- GRANT SELECT, INSERT ON experiment_metrics TO gl002_app_user;
-- GRANT SELECT, INSERT ON experiment_assignments TO gl002_app_user;
-- GRANT SELECT, INSERT, UPDATE ON experiment_results TO gl002_app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO gl002_app_user;


-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE experiments IS
    'A/B testing experiments configuration and metadata';

COMMENT ON COLUMN experiments.variants IS
    'JSONB array of variant configurations with traffic splits and settings';

COMMENT ON TABLE experiment_metrics IS
    'Raw metric observations for each user in each variant';

COMMENT ON TABLE experiment_assignments IS
    'User assignments to variants - ensures consistency via assignment_hash';

COMMENT ON TABLE experiment_results IS
    'Analysis results and conclusions for completed experiments';

COMMENT ON MATERIALIZED VIEW experiment_performance IS
    'High-level experiment performance summary - refresh periodically';

COMMENT ON FUNCTION get_variant_stats IS
    'Calculate statistical summary for a specific variant';

COMMENT ON FUNCTION refresh_experiment_views IS
    'Refresh all experiment materialized views - run daily via cron';
