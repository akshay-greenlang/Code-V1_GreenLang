-- Canonical v0.1 optional facets (denormalized for SQL filters; full record remains in payload_json).

ALTER TABLE factors_catalog.catalog_factors
    ADD COLUMN IF NOT EXISTS license_class TEXT,
    ADD COLUMN IF NOT EXISTS activity_tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS sector_tags JSONB NOT NULL DEFAULT '[]'::jsonb;

CREATE INDEX IF NOT EXISTS idx_factors_catalog_license
    ON factors_catalog.catalog_factors (edition_id, license_class);
