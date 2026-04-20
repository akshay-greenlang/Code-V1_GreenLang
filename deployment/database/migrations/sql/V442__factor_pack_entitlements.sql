-- V442__factor_pack_entitlements.sql
-- Phase F8: Per-tenant premium-pack entitlements.
-- Companion to V439 (Climate Ledger) + V440 (Evidence Vault) + V441 (Entity Graph).

CREATE TABLE IF NOT EXISTS factor_pack_entitlements (
    id                    BIGSERIAL     PRIMARY KEY,
    tenant_id             TEXT          NOT NULL,
    pack_sku              TEXT          NOT NULL,
    granted_at            TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    expires_at            TIMESTAMPTZ,
    oem_rights            TEXT          NOT NULL DEFAULT 'forbidden',
    seat_cap              INTEGER,
    volume_cap_per_month  INTEGER,
    active                BOOLEAN       NOT NULL DEFAULT TRUE,
    notes                 TEXT,
    CONSTRAINT chk_oem_rights CHECK (oem_rights IN (
        'forbidden', 'internal_only', 'redistributable'
    )),
    CONSTRAINT chk_pack_sku CHECK (pack_sku IN (
        'electricity_premium', 'freight_premium', 'product_carbon_premium',
        'epd_premium', 'agrifood_premium', 'finance_premium',
        'cbam_premium', 'land_premium'
    )),
    UNIQUE (tenant_id, pack_sku)
);

CREATE INDEX IF NOT EXISTS idx_fpe_tenant
    ON factor_pack_entitlements (tenant_id, active);
CREATE INDEX IF NOT EXISTS idx_fpe_sku_active
    ON factor_pack_entitlements (pack_sku, active);
CREATE INDEX IF NOT EXISTS idx_fpe_expires
    ON factor_pack_entitlements (expires_at)
    WHERE active = TRUE AND expires_at IS NOT NULL;

COMMENT ON TABLE factor_pack_entitlements IS
    'Per-tenant premium pack SKUs (Phase F8). See greenlang.factors.entitlements '
    'for the Python API and docs/factors/FACTORS_EXECUTION_PLAN.md §F8.';
