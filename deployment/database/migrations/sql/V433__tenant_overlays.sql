-- V433: Tenant overlay tables (F064)
-- Enterprise tenant-scoped factor overlays with RLS.

CREATE TABLE IF NOT EXISTS factors_catalog.tenant_overlays (
    overlay_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       VARCHAR(255) NOT NULL,
    factor_id       VARCHAR(255) NOT NULL,
    override_value  DECIMAL(16,8) NOT NULL,
    override_unit   VARCHAR(50) NOT NULL DEFAULT 'kg_co2e',
    valid_from      DATE NOT NULL,
    valid_to        DATE,
    source          VARCHAR(255),
    notes           TEXT,
    created_by      VARCHAR(255),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    active          BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_to_tenant_factor
    ON factors_catalog.tenant_overlays (tenant_id, factor_id, valid_from);

CREATE INDEX IF NOT EXISTS idx_to_tenant_active
    ON factors_catalog.tenant_overlays (tenant_id, active)
    WHERE active = TRUE;

CREATE INDEX IF NOT EXISTS idx_to_tenant_created
    ON factors_catalog.tenant_overlays (tenant_id, created_at DESC);

-- Tenant overlay audit trail (immutable)
CREATE TABLE IF NOT EXISTS factors_catalog.tenant_overlay_audit (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id       VARCHAR(255) NOT NULL,
    overlay_id      UUID NOT NULL,
    action          VARCHAR(20) NOT NULL,
    actor           VARCHAR(255) NOT NULL,
    old_value       DECIMAL(16,8),
    new_value       DECIMAL(16,8),
    details_json    JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_toa_tenant_ts
    ON factors_catalog.tenant_overlay_audit (tenant_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_toa_overlay
    ON factors_catalog.tenant_overlay_audit (overlay_id, timestamp DESC);

-- Row-Level Security for tenant isolation
ALTER TABLE factors_catalog.tenant_overlays ENABLE ROW LEVEL SECURITY;

-- Policy: tenants can only see their own overlays
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'tenant_overlays'
        AND policyname = 'tenant_isolation_overlays'
    ) THEN
        CREATE POLICY tenant_isolation_overlays ON factors_catalog.tenant_overlays
            USING (tenant_id = current_setting('app.current_tenant_id', TRUE));
    END IF;
END
$$;

COMMENT ON TABLE factors_catalog.tenant_overlays IS
    'Enterprise tenant-scoped factor overlays (F064). RLS enforced.';
COMMENT ON TABLE factors_catalog.tenant_overlay_audit IS
    'Immutable audit trail for tenant overlay changes (F064).';
