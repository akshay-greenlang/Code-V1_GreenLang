/**
 * 03 — Batch resolution with polling.
 *
 * Submit N ResolutionRequests, then `waitForBatch` polls the job
 * endpoint until the status reaches a terminal state.
 */
import { FactorsClient, ResolutionRequest } from '@greenlang/factors';

async function main(): Promise<void> {
  const client = new FactorsClient({
    baseUrl: process.env.GL_FACTORS_API_URL ?? 'https://api.greenlang.io',
    apiKey: process.env.GL_FACTORS_API_KEY,
    edition: process.env.GL_FACTORS_EDITION,
  });

  const requests: ResolutionRequest[] = [
    {
      activity: 'diesel combustion',
      method_profile: 'corporate_scope1',
      jurisdiction: 'US',
    },
    {
      activity: 'natural gas combustion',
      method_profile: 'corporate_scope1',
      jurisdiction: 'US',
    },
    {
      activity: 'grid electricity',
      method_profile: 'corporate_scope2_location_based',
      jurisdiction: 'US-CA',
    },
  ];

  const handle = await client.resolveBatch(requests);
  console.log('Submitted batch job:', handle.job_id, 'status:', handle.status);

  const final = await client.waitForBatch(handle, {
    pollIntervalMs: 2_000,
    timeoutMs: 300_000,
  });

  console.log('\nFinal status :', final.status);
  console.log('Items done   :', final.processed_items, '/', final.total_items);
  console.log('Completed at :', final.completed_at);

  if (final.results) {
    for (const r of final.results) {
      console.log('  ', r);
    }
  } else if (final.results_url) {
    console.log('Results downloadable at:', final.results_url);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
