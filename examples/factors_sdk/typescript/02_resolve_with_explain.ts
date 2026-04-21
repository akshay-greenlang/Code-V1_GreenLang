/**
 * 02 — Resolve an activity with full explain payload (Pro+ tier).
 *
 * Shows the 7-step cascade result, assumptions, gas breakdown, and
 * quality score for a concrete activity.
 */
import { FactorsClient, FactorsAPIError, TierError } from '@greenlang/factors';

async function main(): Promise<void> {
  const client = new FactorsClient({
    baseUrl: process.env.GL_FACTORS_API_URL ?? 'https://api.greenlang.io',
    apiKey: process.env.GL_FACTORS_API_KEY,
    edition: process.env.GL_FACTORS_EDITION ?? 'ef_2026_q1',
    methodProfile: 'corporate_scope1',
  });

  try {
    const resolved = await client.resolve(
      {
        activity: 'diesel combustion',
        method_profile: 'corporate_scope1',
        jurisdiction: 'US-CA',
        reporting_date: '2026-01-15',
        facility_id: 'plant-42',
      },
      { alternates: 3 },
    );

    console.log('Chosen factor :', resolved.chosen_factor_id ?? resolved.factor_id);
    console.log('Cascade step  :', resolved.fallback_rank, resolved.step_label);
    console.log('Method profile:', resolved.method_profile);
    console.log('Why           :', resolved.why_chosen);
    console.log('DQS           :', resolved.quality_score?.overall_score);
    console.log('CO2e basis    :', resolved.co2e_basis);
    console.log('Gas breakdown :', resolved.gas_breakdown);
    console.log('Assumptions   :');
    for (const a of resolved.assumptions ?? []) console.log('  -', a);
  } catch (err) {
    if (err instanceof TierError) {
      console.error('This endpoint requires Pro+ tier:', err.message);
      console.error('Remediation:', err.remediation);
    } else if (err instanceof FactorsAPIError) {
      console.error(`API error (HTTP ${err.statusCode}):`, err.message);
    } else {
      throw err;
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
