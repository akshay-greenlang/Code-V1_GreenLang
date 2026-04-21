# Factor resolution (the 7-step cascade)

Resolution turns an *activity description* into a concrete emission
factor plus a full provenance trail. The server runs a 7-step cascade:

1. Tenant override
2. Supplier/facility-specific factor
3. Utility/grid region factor
4. Method-profile preferred sources
5. Jurisdiction-specific default
6. Country default
7. Global fallback

Every response includes `fallback_rank` (1 = best, 7 = last resort),
`why_chosen`, `quality_score`, and an `alternates` array with the
runner-up candidates.

## Quick resolve (POST /factors/resolve-explain)

```ts
const resolved = await client.resolve({
  activity: 'diesel combustion',
  method_profile: 'corporate_scope1',
  jurisdiction: 'US-CA',
  reporting_date: '2026-01-15',
  facility_id: 'plant-42',
}, { alternates: 3, edition: 'ef_2026_q1' });

console.log(resolved.chosen_factor_id, resolved.co2e_basis);
```

## Explain an existing factor (GET /factors/{id}/explain)

If you already know a factor_id and want the cascade reasoning:

```ts
const r = await client.resolveExplain('ef_us_diesel_scope1_v2', {
  methodProfile: 'corporate_scope1',
  alternates: 5,
});
```

## Batch resolution

For bulk workloads, submit many requests as a single asynchronous job:

```ts
const handle = await client.resolveBatch([
  { activity: 'diesel', method_profile: 'corporate_scope1', jurisdiction: 'US' },
  { activity: 'natural gas', method_profile: 'corporate_scope1', jurisdiction: 'US' },
  { activity: 'grid electricity', method_profile: 'corporate_scope2_location_based', jurisdiction: 'US-CA' },
]);

const final = await client.waitForBatch(handle, {
  pollIntervalMs: 2_000,
  timeoutMs: 300_000,
});

for (const r of final.results ?? []) console.log(r);
```

`waitForBatch` throws `FactorsAPIError` on failed status and on timeout.

## Default method profile

You can set a default once:

```ts
const client = new FactorsClient({
  baseUrl: '...',
  apiKey: '...',
  methodProfile: 'corporate_scope1',   // used when a call omits it
});
```

The default is consulted by `resolve` (if the payload does not supply
`method_profile`), `resolveExplain`, and `alternates`.

## Gas breakdown

For Scope 1/2 factors the server keeps CO₂, CH₄, N₂O, HFCs, PFCs, SF₆,
NF₃, and biogenic CO₂ as *separate* components. Never roll them up
client-side — always use the aggregate `co2e_per_unit`:

```ts
const r = await client.resolve(request);
console.log(r.gas_breakdown?.CO2, r.gas_breakdown?.CH4, r.gas_breakdown?.biogenic_CO2);
console.log(r.co2e_basis);  // e.g. 'AR6-GWP100'
```
