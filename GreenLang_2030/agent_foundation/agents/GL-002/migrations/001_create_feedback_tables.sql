-- Migration: Create Feedback System Tables
-- Version: 001
-- Description: Creates tables for user feedback collection and analysis
-- Author: GL-BackendDeveloper
-- Date: 2025-11-17

-- ============================================================================
-- FEEDBACK TABLES
-- ============================================================================

-- Main feedback table
CREATE TABLE IF NOT EXISTS optimization_feedback (
    id BIGSERIAL PRIMARY KEY,
    optimization_id VARCHAR(255) NOT NULL,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    actual_savings DECIMAL(12, 2),
    predicted_savings DECIMAL(12, 2),
    category VARCHAR(50) NOT NULL DEFAULT 'other',
    user_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    savings_accuracy DECIMAL(5, 2),
    provenance_hash VARCHAR(64) NOT NULL,

    -- Indexes for performance
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_feedback_optimization_id
    ON optimization_feedback(optimization_id);

CREATE INDEX IF NOT EXISTS idx_feedback_user_id
    ON optimization_feedback(user_id);

CREATE INDEX IF NOT EXISTS idx_feedback_timestamp
    ON optimization_feedback(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_feedback_rating
    ON optimization_feedback(rating);

CREATE INDEX IF NOT EXISTS idx_feedback_category
    ON optimization_feedback(category);

-- Unique constraint to prevent duplicate feedback
CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_unique
    ON optimization_feedback(optimization_id, user_id);

-- GIN index for JSONB metadata queries
CREATE INDEX IF NOT EXISTS idx_feedback_metadata
    ON optimization_feedback USING GIN (metadata);


-- Satisfaction trends table (aggregated daily)
CREATE TABLE IF NOT EXISTS satisfaction_trends (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    average_rating DECIMAL(3, 2) NOT NULL,
    feedback_count INTEGER NOT NULL DEFAULT 0,
    nps_score DECIMAL(5, 2),

    -- Moving averages
    ma_7day DECIMAL(3, 2),
    ma_30day DECIMAL(3, 2),

    -- Anomaly detection
    is_anomaly BOOLEAN DEFAULT FALSE,
    anomaly_score DECIMAL(5, 2),

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Index for date range queries
CREATE INDEX IF NOT EXISTS idx_satisfaction_date
    ON satisfaction_trends(date DESC);


-- Feedback alerts table
CREATE TABLE IF NOT EXISTS feedback_alerts (
    id BIGSERIAL PRIMARY KEY,
    alert_id VARCHAR(255) NOT NULL UNIQUE,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('critical', 'warning', 'info')),
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,

    -- Trigger information
    triggered_by VARCHAR(100) NOT NULL,
    threshold_value DECIMAL(12, 2) NOT NULL,
    actual_value DECIMAL(12, 2) NOT NULL,

    -- Related data
    affected_optimizations TEXT[] DEFAULT '{}',
    related_feedback_ids BIGINT[] DEFAULT '{}',

    -- Alert lifecycle
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by VARCHAR(255),
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for alerts
CREATE INDEX IF NOT EXISTS idx_alerts_severity
    ON feedback_alerts(severity);

CREATE INDEX IF NOT EXISTS idx_alerts_created
    ON feedback_alerts(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged
    ON feedback_alerts(acknowledged) WHERE NOT acknowledged;


-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_feedback_updated_at
    BEFORE UPDATE ON optimization_feedback
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_satisfaction_updated_at
    BEFORE UPDATE ON satisfaction_trends
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


-- Function to validate rating
CREATE OR REPLACE FUNCTION validate_rating()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.rating < 1 OR NEW.rating > 5 THEN
        RAISE EXCEPTION 'Rating must be between 1 and 5, got %', NEW.rating;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to validate rating
CREATE TRIGGER validate_feedback_rating
    BEFORE INSERT OR UPDATE ON optimization_feedback
    FOR EACH ROW EXECUTE FUNCTION validate_rating();


-- Function to calculate savings accuracy
CREATE OR REPLACE FUNCTION calculate_savings_accuracy()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.actual_savings IS NOT NULL AND NEW.predicted_savings IS NOT NULL AND NEW.predicted_savings > 0 THEN
        NEW.savings_accuracy := 100 * (1 - ABS(NEW.actual_savings - NEW.predicted_savings) / NEW.predicted_savings);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-calculate savings accuracy
CREATE TRIGGER auto_calculate_accuracy
    BEFORE INSERT OR UPDATE ON optimization_feedback
    FOR EACH ROW EXECUTE FUNCTION calculate_savings_accuracy();


-- ============================================================================
-- MATERIALIZED VIEWS FOR ANALYTICS
-- ============================================================================

-- Weekly feedback summary
CREATE MATERIALIZED VIEW IF NOT EXISTS weekly_feedback_summary AS
SELECT
    DATE_TRUNC('week', timestamp) AS week_start,
    COUNT(*) AS total_feedback,
    AVG(rating) AS avg_rating,
    STDDEV(rating) AS stddev_rating,
    AVG(savings_accuracy) AS avg_accuracy,
    COUNT(CASE WHEN rating >= 4 THEN 1 END) AS promoters,
    COUNT(CASE WHEN rating <= 3 THEN 1 END) AS detractors,
    100.0 * (COUNT(CASE WHEN rating >= 4 THEN 1 END) - COUNT(CASE WHEN rating <= 3 THEN 1 END)) / COUNT(*) AS nps_score
FROM optimization_feedback
GROUP BY DATE_TRUNC('week', timestamp)
ORDER BY week_start DESC;

-- Create index on materialized view
CREATE INDEX IF NOT EXISTS idx_weekly_summary_week
    ON weekly_feedback_summary(week_start DESC);


-- Category performance summary
CREATE MATERIALIZED VIEW IF NOT EXISTS category_performance AS
SELECT
    category,
    COUNT(*) AS feedback_count,
    AVG(rating) AS avg_rating,
    AVG(savings_accuracy) AS avg_accuracy,
    COUNT(CASE WHEN rating <= 2 THEN 1 END) AS low_rating_count,
    MAX(timestamp) AS last_feedback_at
FROM optimization_feedback
GROUP BY category
ORDER BY feedback_count DESC;


-- ============================================================================
-- HELPER FUNCTIONS FOR COMMON QUERIES
-- ============================================================================

-- Function to get feedback stats for a date range
CREATE OR REPLACE FUNCTION get_feedback_stats(
    start_date TIMESTAMP WITH TIME ZONE,
    end_date TIMESTAMP WITH TIME ZONE
)
RETURNS TABLE (
    total_count BIGINT,
    avg_rating DECIMAL,
    rating_1 BIGINT,
    rating_2 BIGINT,
    rating_3 BIGINT,
    rating_4 BIGINT,
    rating_5 BIGINT,
    avg_accuracy DECIMAL,
    nps DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*),
        AVG(rating),
        COUNT(CASE WHEN rating = 1 THEN 1 END),
        COUNT(CASE WHEN rating = 2 THEN 1 END),
        COUNT(CASE WHEN rating = 3 THEN 1 END),
        COUNT(CASE WHEN rating = 4 THEN 1 END),
        COUNT(CASE WHEN rating = 5 THEN 1 END),
        AVG(savings_accuracy),
        100.0 * (COUNT(CASE WHEN rating >= 4 THEN 1 END) - COUNT(CASE WHEN rating <= 3 THEN 1 END)) / NULLIF(COUNT(*), 0)
    FROM optimization_feedback
    WHERE timestamp >= start_date AND timestamp <= end_date;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- GRANTS AND PERMISSIONS
-- ============================================================================

-- Grant permissions (adjust user as needed)
-- GRANT SELECT, INSERT, UPDATE ON optimization_feedback TO gl002_app_user;
-- GRANT SELECT, INSERT, UPDATE ON satisfaction_trends TO gl002_app_user;
-- GRANT SELECT, INSERT, UPDATE ON feedback_alerts TO gl002_app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO gl002_app_user;


-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE optimization_feedback IS
    'User feedback for optimization recommendations - tracks satisfaction and actual results';

COMMENT ON COLUMN optimization_feedback.savings_accuracy IS
    'Auto-calculated percentage accuracy of predicted vs actual savings';

COMMENT ON COLUMN optimization_feedback.provenance_hash IS
    'SHA-256 hash for audit trail and data integrity verification';

COMMENT ON TABLE satisfaction_trends IS
    'Daily aggregated satisfaction metrics with moving averages for trend analysis';

COMMENT ON TABLE feedback_alerts IS
    'Alerts triggered by low ratings or accuracy issues requiring attention';

COMMENT ON MATERIALIZED VIEW weekly_feedback_summary IS
    'Weekly aggregated feedback metrics including NPS - refresh daily';

COMMENT ON FUNCTION get_feedback_stats IS
    'Helper function to retrieve feedback statistics for any date range';
