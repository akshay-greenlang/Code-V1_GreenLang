# Edition pinning, diff, and audit bundles

GreenLang publishes catalog editions on a quarterly cadence
(`ef_2025_q4`, `ef_2026_q1`, …). For reproducibility in compliance
workflows you should **pin** to a specific edition and only move
forward after validating the diff.

## Pin at client level

```ts
const client = new FactorsClient({
  baseUrl: 'https://api.greenlang.io',
  apiKey: process.env.GL_FACTORS_API_KEY,
  edition: 'ef_2026_q1',   // sent as X-Factors-Edition on every request
});
```

Every response echoes `X-Factors-Edition` (accessible via
`response.edition` in transport-level calls) so downstream consumers
can verify they received data from the expected catalog.

## Pin per call

```ts
await client.resolve(request, { edition: 'ef_2026_q1' });
await client.getFactor('ef_us_diesel_scope1_v2', { edition: 'ef_2026_q1' });
await client.search('diesel', { edition: 'ef_2026_q1' });
```

Per-call pinning overrides the client default.

## List & inspect editions

```ts
const editions = await client.listEditions({ includePending: true });
for (const e of editions) {
  console.log(`${e.edition_id}  ${e.status}  ${e.manifest_hash}`);
}

const changelog = await client.getEdition('ef_2026_q1');
```

## Diff two editions for a factor

```ts
const d = await client.diff('ef_us_diesel_scope1_v2', 'ef_2025_q4', 'ef_2026_q1');

console.log(d.status);              // 'changed' | 'unchanged' | 'added' | 'removed' | 'not_found'
console.log(d.left_content_hash);
console.log(d.right_content_hash);
for (const c of d.changes ?? []) {
  console.log(c.field, c.old_value, '→', c.new_value);
}
```

## Audit bundle (Enterprise)

The audit bundle is the full verification-chain artefact — everything
an assurance engagement needs to reproduce a number:

```ts
const bundle = await client.auditBundle('ef_us_diesel_scope1_v2', {
  edition: 'ef_2026_q1',
});

console.log(bundle.content_hash);
console.log(bundle.payload_sha256);
console.log(bundle.provenance);
console.log(bundle.verification_chain);
```

Bundles should be stored immutably (S3 + object lock, or your
evidence vault of choice). Their `content_hash` and `payload_sha256`
let you detect tampering.

## ETag caching

The transport layer transparently caches successful GET responses by
`ETag`. Subsequent calls with the same URL + params send
`If-None-Match`; on a 304 Not Modified the cached body is returned
without a full round-trip. Plug your own cache:

```ts
import { ETagCache, FactorsClient } from '@greenlang/factors';

const shared = new ETagCache(2048);
const c1 = new FactorsClient({ baseUrl: '...', apiKey: '...', cache: shared });
const c2 = new FactorsClient({ baseUrl: '...', apiKey: '...', cache: shared });
// c1 and c2 share one ETag cache.
```
