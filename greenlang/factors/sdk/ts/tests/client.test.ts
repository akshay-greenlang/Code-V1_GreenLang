import {
  APIKeyAuth,
  AuthError,
  FactorNotFoundError,
  FactorsClient,
  HMACAuth,
  JWTAuth,
  LicenseError,
  RateLimitError,
  TierError,
  ValidationError,
} from '../src';
import { makeMockFetch, noSleep } from './mocks';

function client(fetchImpl: ReturnType<typeof makeMockFetch>['fetchImpl'], extra: Record<string, unknown> = {}) {
  return new FactorsClient({
    baseUrl: 'https://api.test',
    apiKey: 'gl_test',
    fetchImpl,
    sleep: noSleep,
    maxRetries: 3,
    ...extra,
  });
}

describe('FactorsClient — instantiation', () => {
  it('accepts an apiKey and sets X-API-Key', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      { body: { factors: [], count: 0 } },
    ]);
    const c = client(fetchImpl);
    await c.search('x');
    expect(invocations[0].headers['X-API-Key']).toBe('gl_test');
  });

  it('accepts a jwtToken and sets Authorization', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      { body: { factors: [], count: 0 } },
    ]);
    const c = new FactorsClient({
      baseUrl: 'https://api.test',
      jwtToken: 'eyJ.abc.def',
      fetchImpl,
      sleep: noSleep,
    });
    await c.search('x');
    expect(invocations[0].headers['Authorization']).toBe('Bearer eyJ.abc.def');
  });

  it('accepts an explicit AuthProvider that takes precedence', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      { body: { factors: [] } },
    ]);
    const c = new FactorsClient({
      baseUrl: 'https://api.test',
      apiKey: 'should-not-win',
      auth: new JWTAuth('winner'),
      fetchImpl,
      sleep: noSleep,
    });
    await c.search('x');
    expect(invocations[0].headers['Authorization']).toBe('Bearer winner');
    expect(invocations[0].headers['X-API-Key']).toBeUndefined();
  });

  it('sets X-Factors-Edition when `edition` is provided', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      { body: { factors: [] } },
    ]);
    const c = client(fetchImpl, { edition: 'ef_2026_q1' });
    await c.search('x');
    expect(invocations[0].headers['X-Factors-Edition']).toBe('ef_2026_q1');
  });
});

describe('FactorsClient — search endpoints', () => {
  it('search() hits GET /api/v1/factors/search', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      { body: { factors: [{ factor_id: 'f1' }], count: 1 } },
    ]);
    const c = client(fetchImpl);
    const out = await c.search('diesel', { geography: 'US', limit: 5 });
    expect(invocations[0].method).toBe('GET');
    expect(invocations[0].url).toContain('/api/v1/factors/search?');
    expect(invocations[0].url).toContain('q=diesel');
    expect(invocations[0].url).toContain('geography=US');
    expect(invocations[0].url).toContain('limit=5');
    expect(out.factors).toHaveLength(1);
    expect(out.factors[0].factor_id).toBe('f1');
  });

  it('searchV2() POSTs to /api/v1/factors/search/v2 with body', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      { body: { factors: [], total_count: 0 } },
    ]);
    const c = client(fetchImpl);
    await c.searchV2('steel', { geography: 'EU', dqsMin: 3.5, sectorTags: ['mfg'] });
    expect(invocations[0].method).toBe('POST');
    expect(invocations[0].url).toContain('/api/v1/factors/search/v2');
    const body = JSON.parse(invocations[0].body!);
    expect(body.query).toBe('steel');
    expect(body.geography).toBe('EU');
    expect(body.dqs_min).toBe(3.5);
    expect(body.sector_tags).toEqual(['mfg']);
  });

  it('listFactors() uses GET /api/v1/factors with query params', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      { body: { factors: [], page: 1 } },
    ]);
    const c = client(fetchImpl);
    await c.listFactors({ fuelType: 'diesel', page: 2, limit: 50 });
    expect(invocations[0].url).toContain('fuel_type=diesel');
    expect(invocations[0].url).toContain('page=2');
    expect(invocations[0].url).toContain('limit=50');
  });

  it('paginateSearch() async-iterates across pages', async () => {
    const { fetchImpl } = makeMockFetch([
      {
        body: {
          factors: [{ factor_id: 'a' }, { factor_id: 'b' }],
          total_count: 3,
        },
      },
      {
        body: { factors: [{ factor_id: 'c' }], total_count: 3 },
      },
    ]);
    const c = client(fetchImpl);
    const pager = c.paginateSearch('x', { pageSize: 2 });
    const out: string[] = [];
    for await (const f of pager) {
      out.push(f.factor_id);
    }
    expect(out).toEqual(['a', 'b', 'c']);
  });
});

describe('FactorsClient — factors / match / coverage', () => {
  it('getFactor() returns parsed Factor', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      { body: { factor_id: 'ef1', co2e_per_unit: 2.5 } },
    ]);
    const c = client(fetchImpl);
    const f = await c.getFactor('ef1');
    expect(invocations[0].url).toContain('/api/v1/factors/ef1');
    expect(f.factor_id).toBe('ef1');
    expect(f.co2e_per_unit).toBe(2.5);
  });

  it('match() returns candidates array', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      { body: { candidates: [{ factor_id: 'a', score: 0.9 }] } },
    ]);
    const c = client(fetchImpl);
    const out = await c.match('burn diesel', { limit: 3 });
    expect(invocations[0].method).toBe('POST');
    expect(out).toHaveLength(1);
    expect(out[0].factor_id).toBe('a');
    expect(out[0].score).toBe(0.9);
  });

  it('coverage() parses CoverageReport', async () => {
    const { fetchImpl } = makeMockFetch([
      {
        body: {
          total_factors: 327,
          by_geography: { US: 100, EU: 50 },
          edition_id: 'ef_2026_q1',
        },
      },
    ]);
    const c = client(fetchImpl);
    const cov = await c.coverage();
    expect(cov.total_factors).toBe(327);
    expect(cov.by_geography?.US).toBe(100);
  });
});

describe('FactorsClient — resolution', () => {
  it('resolveExplain() → GET /factors/{id}/explain', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      {
        body: { chosen_factor_id: 'ef1', fallback_rank: 2 },
      },
    ]);
    const c = client(fetchImpl);
    const out = await c.resolveExplain('ef1', { methodProfile: 'corporate_scope1' });
    expect(invocations[0].url).toContain('/api/v1/factors/ef1/explain');
    expect(invocations[0].url).toContain('method_profile=corporate_scope1');
    expect(out.chosen_factor_id).toBe('ef1');
  });

  it('resolve() → POST /factors/resolve-explain', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      { body: { chosen_factor_id: 'ef_resolved' } },
    ]);
    const c = client(fetchImpl, { methodProfile: 'corporate_scope1' });
    const out = await c.resolve({
      activity: 'diesel',
      method_profile: 'corporate_scope1',
      jurisdiction: 'US',
    });
    expect(invocations[0].method).toBe('POST');
    expect(invocations[0].url).toContain('/api/v1/factors/resolve-explain');
    const body = JSON.parse(invocations[0].body!);
    expect(body.activity).toBe('diesel');
    expect(body.jurisdiction).toBe('US');
    expect(out.chosen_factor_id).toBe('ef_resolved');
  });

  it('alternates() → GET /factors/{id}/alternates', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      { body: { alternates: [{ factor_id: 'alt1' }] } },
    ]);
    const c = client(fetchImpl);
    const out = await c.alternates('ef1', { limit: 5 });
    expect(invocations[0].url).toContain('/api/v1/factors/ef1/alternates');
    expect(invocations[0].url).toContain('limit=5');
    expect((out.alternates as Array<{ factor_id: string }>)[0].factor_id).toBe('alt1');
  });
});

describe('FactorsClient — batch resolution', () => {
  it('resolveBatch() + getBatchJob() + waitForBatch()', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      // 1. submit batch
      { body: { job_id: 'j1', status: 'queued' } },
      // 2. first poll: running
      { body: { job_id: 'j1', status: 'running' } },
      // 3. second poll: completed
      { body: { job_id: 'j1', status: 'completed', processed_items: 3 } },
    ]);
    const c = client(fetchImpl);
    const handle = await c.resolveBatch([
      { activity: 'diesel', method_profile: 'corporate_scope1' },
    ]);
    expect(handle.job_id).toBe('j1');
    expect(invocations[0].method).toBe('POST');
    expect(invocations[0].url).toContain('/api/v1/factors/resolve/batch');

    const final = await c.waitForBatch(handle, { pollIntervalMs: 0, timeoutMs: 0 });
    expect(final.status).toBe('completed');
    expect(final.processed_items).toBe(3);
  });

  it('waitForBatch() raises on failed job', async () => {
    const { fetchImpl } = makeMockFetch([
      { body: { job_id: 'j2', status: 'failed', error_message: 'boom' } },
    ]);
    const c = client(fetchImpl);
    await expect(
      c.waitForBatch('j2', { pollIntervalMs: 0, timeoutMs: 0 }),
    ).rejects.toThrow(/Batch job j2 failed: boom/);
  });
});

describe('FactorsClient — editions / sources / method packs / overrides', () => {
  it('listEditions() unwraps {editions:[]}', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      { body: { editions: [{ edition_id: 'ef_2026_q1' }] } },
    ]);
    const c = client(fetchImpl);
    const eds = await c.listEditions();
    expect(invocations[0].url).toContain('/api/v1/editions');
    expect(eds).toHaveLength(1);
    expect(eds[0].edition_id).toBe('ef_2026_q1');
  });

  it('getEdition() fetches /editions/{id}/changelog', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      { body: { edition_id: 'ef_2026_q1', changes: [] } },
    ]);
    const c = client(fetchImpl);
    const out = await c.getEdition('ef_2026_q1');
    expect(invocations[0].url).toContain('/api/v1/editions/ef_2026_q1/changelog');
    expect(out.edition_id).toBe('ef_2026_q1');
  });

  it('diff() fetches /factors/{id}/diff with both editions', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      {
        body: {
          factor_id: 'f1',
          left_edition: 'a',
          right_edition: 'b',
          status: 'changed',
          changes: [{ field: 'co2e_per_unit' }],
        },
      },
    ]);
    const c = client(fetchImpl);
    const d = await c.diff('f1', 'a', 'b');
    expect(invocations[0].url).toContain('/api/v1/factors/f1/diff');
    expect(invocations[0].url).toContain('left_edition=a');
    expect(invocations[0].url).toContain('right_edition=b');
    expect(d.status).toBe('changed');
  });

  it('auditBundle() fetches /factors/{id}/audit-bundle', async () => {
    const { fetchImpl } = makeMockFetch([
      { body: { factor_id: 'f1', edition_id: 'ef_2026_q1' } },
    ]);
    const c = client(fetchImpl);
    const b = await c.auditBundle('f1');
    expect(b.factor_id).toBe('f1');
  });

  it('listSources() unwraps {sources:[]}', async () => {
    const { fetchImpl } = makeMockFetch([
      { body: { sources: [{ source_id: 'EPA' }] } },
    ]);
    const c = client(fetchImpl);
    const out = await c.listSources();
    expect(out).toHaveLength(1);
    expect(out[0].source_id).toBe('EPA');
  });

  it('listMethodPacks() unwraps {method_packs:[]}', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      {
        body: { method_packs: [{ method_pack_id: 'corporate_scope1' }] },
      },
    ]);
    const c = client(fetchImpl);
    const out = await c.listMethodPacks();
    expect(invocations[0].url).toContain('/api/v1/method-packs');
    expect(out[0].method_pack_id).toBe('corporate_scope1');
  });

  it('setOverride() posts body', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      { body: { factor_id: 'f1', tenant_id: 't1', co2e_per_unit: 1.5 } },
    ]);
    const c = client(fetchImpl);
    const out = await c.setOverride({
      factor_id: 'f1',
      co2e_per_unit: 1.5,
      justification: 'supplier-specific',
    });
    expect(invocations[0].method).toBe('POST');
    const body = JSON.parse(invocations[0].body!);
    expect(body.factor_id).toBe('f1');
    expect(out.tenant_id).toBe('t1');
  });

  it('listOverrides() accepts tenantId', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      { body: { overrides: [{ factor_id: 'f1' }] } },
    ]);
    const c = client(fetchImpl);
    const out = await c.listOverrides({ tenantId: 't1' });
    expect(invocations[0].url).toContain('tenant_id=t1');
    expect(out).toHaveLength(1);
  });
});

describe('FactorsClient — retry + caching + errors', () => {
  it('retries on 503 then succeeds', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      { status: 503, text: 'upstream' },
      { body: { factors: [] } },
    ]);
    const c = client(fetchImpl);
    await c.search('x');
    expect(invocations).toHaveLength(2);
  });

  it('retries on 429 then throws RateLimitError if exhausted', async () => {
    const { fetchImpl } = makeMockFetch([
      { status: 429, body: { detail: 'slow down' } },
      { status: 429, body: { detail: 'slow down' } },
      { status: 429, body: { detail: 'slow down' } },
    ]);
    const c = client(fetchImpl);
    await expect(c.search('x')).rejects.toBeInstanceOf(RateLimitError);
  });

  it('maps 401 → AuthError', async () => {
    const { fetchImpl } = makeMockFetch([
      { status: 401, body: { detail: 'bad token' } },
    ]);
    const c = client(fetchImpl);
    await expect(c.getFactor('x')).rejects.toBeInstanceOf(AuthError);
  });

  it('maps 403 → TierError by default', async () => {
    const { fetchImpl } = makeMockFetch([
      { status: 403, body: { detail: 'insufficient tier' } },
    ]);
    const c = client(fetchImpl);
    await expect(c.getFactor('x')).rejects.toBeInstanceOf(TierError);
  });

  it('maps 403 with license wording → LicenseError', async () => {
    const { fetchImpl } = makeMockFetch([
      { status: 403, body: { detail: 'connector_only license restriction' } },
    ]);
    const c = client(fetchImpl);
    await expect(c.getFactor('x')).rejects.toBeInstanceOf(LicenseError);
  });

  it('maps 404 on /factors/ → FactorNotFoundError', async () => {
    const { fetchImpl } = makeMockFetch([
      { status: 404, body: { detail: 'not found' } },
    ]);
    const c = client(fetchImpl);
    await expect(c.getFactor('x')).rejects.toBeInstanceOf(FactorNotFoundError);
  });

  it('maps 422 → ValidationError', async () => {
    const { fetchImpl } = makeMockFetch([
      { status: 422, body: { detail: 'bad shape' } },
    ]);
    const c = client(fetchImpl);
    await expect(c.getFactor('x')).rejects.toBeInstanceOf(ValidationError);
  });

  it('honours ETag cache: 200 → ETag stored, 304 → cached body returned', async () => {
    const etag = 'W/"abc123"';
    const { fetchImpl, invocations } = makeMockFetch([
      {
        status: 200,
        headers: { ETag: etag, 'Content-Type': 'application/json' },
        body: { factor_id: 'f1', cached: true },
      },
      {
        status: 304,
        headers: { ETag: etag },
        text: '',
      },
    ]);
    const c = client(fetchImpl);
    const a = await c.getFactor('f1');
    const b = await c.getFactor('f1');
    expect(a.factor_id).toBe('f1');
    expect((b as unknown as Record<string, unknown>).cached).toBe(true);
    expect(invocations[1].headers['If-None-Match']).toBe(etag);
  });
});

describe('HMACAuth', () => {
  it('sets all four GL headers with a deterministic clock', async () => {
    const { fetchImpl, invocations } = makeMockFetch([
      { body: { factors: [] } },
    ]);
    const c = new FactorsClient({
      baseUrl: 'https://api.test',
      auth: new HMACAuth({
        apiKeyId: 'key-1',
        secret: 'supersecret',
        clock: () => 1_700_000_000,
        primary: new APIKeyAuth({ apiKey: 'gl_test' }),
      }),
      fetchImpl,
      sleep: noSleep,
    });
    await c.search('x');
    const h = invocations[0].headers;
    expect(h['X-GL-Key-Id']).toBe('key-1');
    expect(h['X-GL-Timestamp']).toBe('1700000000');
    expect(h['X-GL-Signature']).toMatch(/^sha256=[0-9a-f]{64}$/);
    expect(h['X-API-Key']).toBe('gl_test');
    expect(h['X-GL-Nonce']).toBeTruthy();
  });
});
