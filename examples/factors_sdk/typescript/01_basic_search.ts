/**
 * 01 — Basic search
 *
 * Run:
 *   npx ts-node examples/factors_sdk/typescript/01_basic_search.ts
 *
 * Requires GL_FACTORS_API_KEY to be set; uses the public API by default.
 */
import { FactorsClient } from '@greenlang/factors';

async function main(): Promise<void> {
  const apiKey = process.env.GL_FACTORS_API_KEY;
  if (!apiKey) {
    throw new Error('Please set GL_FACTORS_API_KEY');
  }

  const client = new FactorsClient({
    baseUrl: process.env.GL_FACTORS_API_URL ?? 'https://api.greenlang.io',
    apiKey,
    edition: process.env.GL_FACTORS_EDITION,
    timeoutMs: 30_000,
  });

  // 1. A simple full-text search.
  const simple = await client.search('natural gas US', { limit: 5 });
  console.log(`Got ${simple.factors.length} simple matches`);
  for (const f of simple.factors) {
    console.log(`  ${f.factor_id}  co2e=${f.co2e_per_unit ?? 'n/a'} /${f.unit}`);
  }

  // 2. Advanced filters via /search/v2.
  const advanced = await client.searchV2('diesel', {
    geography: 'US',
    scope: '1',
    dqsMin: 3.5,
    sortBy: 'dqs_score',
    sortOrder: 'desc',
    limit: 10,
  });
  console.log(`\nGot ${advanced.factors.length} advanced matches (total=${advanced.total_count})`);
  for (const f of advanced.factors) {
    console.log(
      `  ${f.factor_id}  DQS=${f.data_quality?.overall_score ?? 'n/a'}  ${f.geography}/${f.scope}`,
    );
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
