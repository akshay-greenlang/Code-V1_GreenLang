-- =============================================================================
-- V430: Factors Watch Results Table
-- Stores results from automated source monitoring (F050)
-- =============================================================================

CREATE TABLE IF NOT EXISTS factors_catalog.watch_results (
    id              BIGSERIAL PRIMARY KEY,
    source_id       TEXT        NOT NULL,
    check_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    watch_mechanism TEXT        NOT NULL DEFAULT 'http_head',
    url             TEXT,
    http_status     INTEGER,
    file_hash       TEXT,
    previous_hash   TEXT,
    change_detected BOOLEAN     NOT NULL DEFAULT FALSE,
    change_type     TEXT,
    response_ms     INTEGER,
    error_message   TEXT,
    metadata_json   JSONB       NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_watch_results_source
    ON factors_catalog.watch_results (source_id, check_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_watch_results_changes
    ON factors_catalog.watch_results (change_detected, check_timestamp DESC)
    WHERE change_detected = TRUE;

CREATE INDEX IF NOT EXISTS idx_watch_results_timestamp
    ON factors_catalog.watch_results (check_timestamp DESC);

COMMENT ON TABLE factors_catalog.watch_results IS
    'Automated source monitoring results from daily watch runs';
