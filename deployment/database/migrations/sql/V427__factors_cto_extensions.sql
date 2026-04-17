-- CTO backlog extensions: governance columns + ingestion / QA / policy applicability (Postgres).

ALTER TABLE factors_catalog.catalog_factors
    ADD COLUMN IF NOT EXISTS factor_status TEXT DEFAULT 'certified',
    ADD COLUMN IF NOT EXISTS source_id TEXT,
    ADD COLUMN IF NOT EXISTS source_release TEXT,
    ADD COLUMN IF NOT EXISTS source_record_id TEXT,
    ADD COLUMN IF NOT EXISTS release_version TEXT,
    ADD COLUMN IF NOT EXISTS validation_flags JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS replacement_factor_id TEXT;

CREATE INDEX IF NOT EXISTS idx_factors_catalog_status
    ON factors_catalog.catalog_factors (edition_id, factor_status);

CREATE TABLE IF NOT EXISTS factors_catalog.raw_artifacts (
    artifact_id     TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL,
    retrieved_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    url             TEXT,
    content_type    TEXT,
    sha256          TEXT NOT NULL,
    bytes_size      BIGINT,
    storage_uri     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS factors_catalog.ingest_runs (
    run_id          TEXT PRIMARY KEY,
    artifact_id     TEXT,
    edition_id      TEXT,
    parser_id       TEXT NOT NULL,
    status          TEXT NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    row_counts_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    owner           TEXT,
    error           TEXT
);

CREATE TABLE IF NOT EXISTS factors_catalog.factor_lineage (
    edition_id      TEXT NOT NULL,
    factor_id       TEXT NOT NULL,
    artifact_id     TEXT,
    ingest_run_id   TEXT,
    lineage_json    JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (edition_id, factor_id)
);

CREATE TABLE IF NOT EXISTS factors_catalog.qa_reviews (
    review_id       TEXT PRIMARY KEY,
    edition_id      TEXT NOT NULL,
    factor_id       TEXT NOT NULL,
    status          TEXT NOT NULL,
    payload_json    JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS factors_catalog.api_usage_events (
    id              BIGSERIAL PRIMARY KEY,
    path            TEXT NOT NULL,
    api_key_hash    TEXT,
    tier            TEXT,
    hit_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS factors_catalog.policy_applicability (
    rule_id         TEXT NOT NULL,
    version         TEXT NOT NULL,
    regulation_tag  TEXT NOT NULL,
    payload_json    JSONB NOT NULL,
    PRIMARY KEY (rule_id, version)
);
