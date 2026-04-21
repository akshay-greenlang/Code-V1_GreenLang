/**
 * 04 — Tenant-specific factor override (Consulting/Platform tier).
 *
 * When a supplier has audited data that differs from the default
 * emission factor, platform-tier tenants can write an override that
 * applies only to their tenant. Subsequent resolve calls for that
 * tenant automatically pick up the override.
 */
import { FactorsClient, Override } from '@greenlang/factors';

async function main(): Promise<void> {
  const client = new FactorsClient({
    baseUrl: process.env.GL_FACTORS_API_URL ?? 'https://api.greenlang.io',
    apiKey: process.env.GL_FACTORS_API_KEY,
    edition: process.env.GL_FACTORS_EDITION,
  });

  // Write an override — tenant_id is inferred from auth context server-side.
  const override: Override = {
    factor_id: 'ef_us_diesel_scope1_v2',
    co2e_per_unit: 2.58, // supplier-audited value
    justification: 'Supplier X audited data - 2026-Q1 report',
    effective_from: '2026-01-01',
    effective_to: '2026-12-31',
    metadata: {
      reviewer: 'jane.doe@example.com',
      audit_id: 'AUD-2026-042',
    },
  };

  const saved = await client.setOverride(override);
  console.log('Saved override:', saved.factor_id, 'tenant:', saved.tenant_id);

  // List all overrides for this tenant.
  const overrides = await client.listOverrides();
  console.log(`\n${overrides.length} active overrides:`);
  for (const o of overrides) {
    console.log(`  ${o.factor_id}  co2e=${o.co2e_per_unit}  valid=${o.effective_from}..${o.effective_to}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
