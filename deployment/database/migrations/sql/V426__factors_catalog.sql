-- GreenLang Factors FY27: optional Postgres catalog (mirrors SQLite schema used by API ingest).
-- Operators may sync from SQLite bundles or load directly via application ETL.

CREATE SCHEMA IF NOT EXISTS factors_catalog;

CREATE TABLE IF NOT EXISTS factors_catalog.editions (
    edition_id      TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    label           TEXT NOT NULL DEFAULT '',
    manifest_hash   TEXT NOT NULL DEFAULT '',
    manifest_json   JSONB NOT NULL DEFAULT '{}'::jsonb,
    changelog_json  JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS factors_catalog.catalog_factors (
    edition_id     TEXT NOT NULL REFERENCES factors_catalog.editions (edition_id) ON DELETE CASCADE,
    factor_id      TEXT NOT NULL,
    fuel_type      TEXT NOT NULL DEFAULT '',
    geography      TEXT NOT NULL DEFAULT '',
    scope          TEXT NOT NULL DEFAULT '',
    boundary       TEXT NOT NULL DEFAULT '',
    search_blob    TEXT NOT NULL DEFAULT '',
    payload_json   JSONB NOT NULL,
    content_hash   TEXT NOT NULL,
    PRIMARY KEY (edition_id, factor_id)
);

CREATE INDEX IF NOT EXISTS idx_factors_catalog_geo
    ON factors_catalog.catalog_factors (edition_id, geography);

CREATE INDEX IF NOT EXISTS idx_factors_catalog_fuel
    ON factors_catalog.catalog_factors (edition_id, fuel_type);

CREATE INDEX IF NOT EXISTS idx_factors_catalog_scope
    ON factors_catalog.catalog_factors (edition_id, scope);

CREATE TABLE IF NOT EXISTS factors_catalog.policy_factor_map (
    policy_rule_id TEXT NOT NULL,
    factor_id      TEXT NOT NULL,
    notes          TEXT,
    PRIMARY KEY (policy_rule_id, factor_id)
);

CREATE INDEX IF NOT EXISTS idx_factors_catalog_policy_rule
    ON factors_catalog.policy_factor_map (policy_rule_id);
