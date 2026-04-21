/**
 * 05 — Audit bundle export (Enterprise tier).
 *
 * Pulls the full verification-chain artefact for a factor — this is
 * what assurance engagements present to their auditor. Include
 * content_hash + payload_sha256 for later integrity checks.
 */
import { writeFileSync } from 'node:fs';

import { FactorsClient, LicenseError, TierError } from '@greenlang/factors';

async function main(): Promise<void> {
  const factorId = process.argv[2] ?? 'ef_us_diesel_scope1_v2';
  const edition = process.argv[3] ?? process.env.GL_FACTORS_EDITION;

  const client = new FactorsClient({
    baseUrl: process.env.GL_FACTORS_API_URL ?? 'https://api.greenlang.io',
    apiKey: process.env.GL_FACTORS_API_KEY,
    edition,
  });

  try {
    const bundle = await client.auditBundle(factorId, { edition });
    const outPath = `./audit-${factorId}.json`;
    writeFileSync(outPath, JSON.stringify(bundle, null, 2));
    console.log('Audit bundle saved:', outPath);
    console.log('  factor_id       :', bundle.factor_id);
    console.log('  edition_id      :', bundle.edition_id);
    console.log('  content_hash    :', bundle.content_hash);
    console.log('  payload_sha256  :', bundle.payload_sha256);
    console.log('  qa_errors       :', bundle.qa_errors?.length ?? 0);
    console.log('  reviewer        :', bundle.reviewer_decision);
  } catch (err) {
    if (err instanceof TierError) {
      console.error('Audit bundles require Enterprise tier.');
    } else if (err instanceof LicenseError) {
      console.error('Factor license does not allow audit export.');
    } else {
      throw err;
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
