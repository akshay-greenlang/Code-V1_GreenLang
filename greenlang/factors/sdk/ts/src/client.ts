/**
 * High-level FactorsClient — TypeScript parity of the Python
 * `FactorsClient` in `greenlang/factors/sdk/python/client.py`.
 *
 * Every method wraps a single REST endpoint under `/api/v1/factors`
 * or `/api/v1/editions` and returns a typed model (see `models.ts`).
 */

import {
  APIKeyAuth,
  AuthProvider,
  JWTAuth,
} from './auth';
import { FactorsAPIError } from './errors';
import {
  AuditBundle,
  BatchJobHandle,
  CoverageReport,
  Edition,
  Factor,
  FactorDiff,
  FactorMatch,
  MethodPack,
  Override,
  ResolutionRequest,
  ResolvedFactor,
  SearchResponse,
  Source,
  isTerminalBatchStatus,
} from './models';
import { OffsetPaginator } from './pagination';
import {
  DEFAULT_MAX_RETRIES,
  DEFAULT_TIMEOUT_MS,
  DEFAULT_USER_AGENT,
  ETagCache,
  FetchLike,
  Transport,
  TransportResponse,
} from './transport';

// ---------------------------------------------------------------------------
// Client options
// ---------------------------------------------------------------------------

export interface FactorsClientOptions {
  /** API host, e.g. `https://api.greenlang.io`. The SDK prepends `/api/v1`. */
  baseUrl: string;
  /** Explicit auth provider (takes precedence over apiKey/jwtToken). */
  auth?: AuthProvider;
  /** Shortcut: create an `APIKeyAuth` internally. */
  apiKey?: string;
  /** Shortcut: create a `JWTAuth` internally. */
  jwtToken?: string;
  /** Pinned edition, sent on every request as `X-Factors-Edition`. */
  defaultEdition?: string;
  /** Alias for `defaultEdition`. */
  edition?: string;
  /** Default method profile forwarded to resolution endpoints. */
  methodProfile?: string;
  /** Per-request timeout (ms). */
  timeoutMs?: number;
  /** Alias for `timeoutMs` (seconds). */
  timeout?: number;
  /** Retry budget on 429/5xx/network errors. */
  maxRetries?: number;
  /** Custom User-Agent. */
  userAgent?: string;
  /** Shared ETag cache. */
  cache?: ETagCache;
  /** Override `/api/v1` prefix. */
  apiPrefix?: string;
  /** Extra headers applied to every request. */
  extraHeaders?: Record<string, string>;
  /** Custom fetch impl (for tests or older Node). */
  fetchImpl?: FetchLike;
  /** Deterministic sleep for tests. */
  sleep?: (ms: number) => Promise<void>;
}

function resolveAuth(opts: FactorsClientOptions): AuthProvider | undefined {
  if (opts.auth) return opts.auth;
  if (opts.apiKey) return new APIKeyAuth({ apiKey: opts.apiKey });
  if (opts.jwtToken) return new JWTAuth(opts.jwtToken);
  return undefined;
}

function boolParam(v: boolean | undefined): string | undefined {
  if (v === undefined) return undefined;
  return v ? 'true' : 'false';
}

function buildSearchResponse(payload: unknown): SearchResponse {
  if (payload && typeof payload === 'object' && !Array.isArray(payload)) {
    return payload as SearchResponse;
  }
  if (Array.isArray(payload)) {
    const factors = payload as Factor[];
    return { factors, count: factors.length };
  }
  return { factors: [] };
}

const TERMINAL_STATES = new Set(['completed', 'failed', 'cancelled']);

export interface SearchV2Options {
  geography?: string;
  fuelType?: string;
  scope?: string;
  sourceId?: string;
  factorStatus?: string;
  licenseClass?: string;
  dqsMin?: number;
  validOnDate?: string;
  sectorTags?: string[];
  activityTags?: string[];
  sortBy?: string;
  sortOrder?: string;
  offset?: number;
  limit?: number;
  edition?: string;
  includePreview?: boolean;
  includeConnector?: boolean;
}

// ---------------------------------------------------------------------------
// FactorsClient
// ---------------------------------------------------------------------------

export class FactorsClient {
  public static readonly DEFAULT_API_PREFIX = '/api/v1';

  public readonly defaultMethodProfile?: string;

  private readonly apiPrefix: string;
  private readonly transport: Transport;

  constructor(options: FactorsClientOptions) {
    this.apiPrefix = (options.apiPrefix ?? FactorsClient.DEFAULT_API_PREFIX)
      .replace(/\/+$/, '');
    this.defaultMethodProfile = options.methodProfile;

    const timeoutMs =
      options.timeoutMs ??
      (options.timeout !== undefined ? options.timeout * 1000 : DEFAULT_TIMEOUT_MS);

    this.transport = new Transport({
      baseUrl: options.baseUrl,
      auth: resolveAuth(options),
      timeoutMs,
      maxRetries: options.maxRetries ?? DEFAULT_MAX_RETRIES,
      userAgent: options.userAgent ?? DEFAULT_USER_AGENT,
      defaultEdition: options.defaultEdition ?? options.edition,
      cache: options.cache,
      extraHeaders: options.extraHeaders,
      fetchImpl: options.fetchImpl,
      sleep: options.sleep,
    });
  }

  get cache(): ETagCache {
    return this.transport.cache;
  }

  close(): void {
    this.transport.close();
  }

  // ---- Path helpers ----------------------------------------------------

  private path(suffix: string): string {
    const s = suffix.startsWith('/') ? suffix : '/' + suffix;
    return this.apiPrefix + s;
  }

  private get<T = unknown>(
    suffix: string,
    params?: Record<string, unknown> | null,
    useCache = true,
  ): Promise<TransportResponse<T>> {
    return this.transport.request<T>('GET', this.path(suffix), {
      params: params ?? undefined,
      useCache,
    });
  }

  private post<T = unknown>(
    suffix: string,
    jsonBody?: unknown,
    params?: Record<string, unknown> | null,
  ): Promise<TransportResponse<T>> {
    return this.transport.request<T>('POST', this.path(suffix), {
      jsonBody,
      params: params ?? undefined,
      useCache: false,
    });
  }

  // =====================================================================
  // Search / listing
  // =====================================================================

  async search(
    query: string,
    opts: {
      geography?: string;
      limit?: number;
      edition?: string;
      includePreview?: boolean;
      includeConnector?: boolean;
    } = {},
  ): Promise<SearchResponse> {
    const params: Record<string, unknown> = {
      q: query,
      limit: opts.limit ?? 20,
    };
    if (opts.geography) params.geography = opts.geography;
    if (opts.edition) params.edition = opts.edition;
    const ip = boolParam(opts.includePreview);
    const ic = boolParam(opts.includeConnector);
    if (ip !== undefined) params.include_preview = ip;
    if (ic !== undefined) params.include_connector = ic;

    const resp = await this.get('/factors/search', params);
    return buildSearchResponse(resp.data);
  }

  async searchV2(
    query: string,
    opts: SearchV2Options = {},
  ): Promise<SearchResponse> {
    const body: Record<string, unknown> = {
      query,
      sort_by: opts.sortBy ?? 'relevance',
      sort_order: opts.sortOrder ?? 'desc',
      offset: opts.offset ?? 0,
      limit: opts.limit ?? 20,
    };
    const maps: Array<[string, unknown]> = [
      ['geography', opts.geography],
      ['fuel_type', opts.fuelType],
      ['scope', opts.scope],
      ['source_id', opts.sourceId],
      ['factor_status', opts.factorStatus],
      ['license_class', opts.licenseClass],
      ['dqs_min', opts.dqsMin],
      ['valid_on_date', opts.validOnDate],
      ['sector_tags', opts.sectorTags],
      ['activity_tags', opts.activityTags],
    ];
    for (const [k, v] of maps) {
      if (v !== undefined && v !== null) body[k] = v;
    }
    if (opts.includePreview !== undefined) body.include_preview = opts.includePreview;
    if (opts.includeConnector !== undefined) body.include_connector = opts.includeConnector;

    const params: Record<string, unknown> = {};
    if (opts.edition) params.edition = opts.edition;

    const resp = await this.post('/factors/search/v2', body, params);
    return buildSearchResponse(resp.data);
  }

  async listFactors(
    opts: {
      fuelType?: string;
      geography?: string;
      scope?: string;
      boundary?: string;
      edition?: string;
      page?: number;
      limit?: number;
      includePreview?: boolean;
      includeConnector?: boolean;
    } = {},
  ): Promise<SearchResponse> {
    const params: Record<string, unknown> = {
      page: opts.page ?? 1,
      limit: opts.limit ?? 100,
    };
    const maps: Array<[string, unknown]> = [
      ['fuel_type', opts.fuelType],
      ['geography', opts.geography],
      ['scope', opts.scope],
      ['boundary', opts.boundary],
      ['edition', opts.edition],
    ];
    for (const [k, v] of maps) {
      if (v !== undefined && v !== null) params[k] = v;
    }
    const ip = boolParam(opts.includePreview);
    const ic = boolParam(opts.includeConnector);
    if (ip !== undefined) params.include_preview = ip;
    if (ic !== undefined) params.include_connector = ic;

    const resp = await this.get('/factors', params);
    return buildSearchResponse(resp.data);
  }

  /** Iterate over all matches of `/search/v2`. */
  paginateSearch(
    query: string,
    opts: SearchV2Options & {
      pageSize?: number;
      maxItems?: number;
    } = {},
  ): OffsetPaginator<Factor> {
    const { pageSize, maxItems, ...rest } = opts;
    const fetcher = async (offset: number, limit: number) => {
      const resp = await this.searchV2(query, {
        ...rest,
        offset,
        limit,
      });
      return {
        items: resp.factors ?? [],
        totalCount: resp.total_count ?? null,
      };
    };
    return new OffsetPaginator(fetcher, {
      pageSize: pageSize ?? 100,
      maxItems,
    });
  }

  // =====================================================================
  // Factors
  // =====================================================================

  async getFactor(
    factorId: string,
    opts: { edition?: string } = {},
  ): Promise<Factor> {
    const params: Record<string, unknown> = {};
    if (opts.edition) params.edition = opts.edition;
    const resp = await this.get(`/factors/${encodeURIComponent(factorId)}`, params);
    return resp.data as Factor;
  }

  async match(
    activityDescription: string,
    opts: {
      geography?: string;
      fuelType?: string;
      scope?: string;
      limit?: number;
      edition?: string;
    } = {},
  ): Promise<FactorMatch[]> {
    const body: Record<string, unknown> = {
      activity_description: activityDescription,
      limit: opts.limit ?? 10,
    };
    if (opts.geography) body.geography = opts.geography;
    if (opts.fuelType) body.fuel_type = opts.fuelType;
    if (opts.scope) body.scope = opts.scope;
    const params: Record<string, unknown> = {};
    if (opts.edition) params.edition = opts.edition;

    const resp = await this.post('/factors/match', body, params);
    const data = resp.data as { candidates?: FactorMatch[] } | unknown;
    if (data && typeof data === 'object' && 'candidates' in data) {
      return (data as { candidates?: FactorMatch[] }).candidates ?? [];
    }
    return [];
  }

  async coverage(opts: { edition?: string } = {}): Promise<CoverageReport> {
    const params: Record<string, unknown> = {};
    if (opts.edition) params.edition = opts.edition;
    const resp = await this.get('/factors/coverage', params);
    return resp.data as CoverageReport;
  }

  // =====================================================================
  // Resolution (Pro+ tier)
  // =====================================================================

  async resolveExplain(
    factorId: string,
    opts: {
      methodProfile?: string;
      alternates?: number;
      edition?: string;
    } = {},
  ): Promise<ResolvedFactor> {
    const params: Record<string, unknown> = {};
    const mp = opts.methodProfile ?? this.defaultMethodProfile;
    if (mp) params.method_profile = mp;
    if (opts.alternates !== undefined) params.limit = opts.alternates;
    if (opts.edition) params.edition = opts.edition;

    const resp = await this.get(
      `/factors/${encodeURIComponent(factorId)}/explain`,
      params,
    );
    return resp.data as ResolvedFactor;
  }

  async resolve(
    request: ResolutionRequest | Record<string, unknown>,
    opts: {
      alternates?: number;
      edition?: string;
      includePreview?: boolean;
      includeConnector?: boolean;
    } = {},
  ): Promise<ResolvedFactor> {
    const body: Record<string, unknown> = { ...(request as Record<string, unknown>) };
    if (!body.method_profile && this.defaultMethodProfile) {
      body.method_profile = this.defaultMethodProfile;
    }
    const params: Record<string, unknown> = {};
    if (opts.alternates !== undefined) params.limit = opts.alternates;
    if (opts.edition) params.edition = opts.edition;
    if (opts.includePreview !== undefined)
      params.include_preview = boolParam(opts.includePreview);
    if (opts.includeConnector !== undefined)
      params.include_connector = boolParam(opts.includeConnector);

    const resp = await this.post('/factors/resolve-explain', body, params);
    return resp.data as ResolvedFactor;
  }

  async alternates(
    factorId: string,
    opts: {
      methodProfile?: string;
      limit?: number;
      edition?: string;
      includePreview?: boolean;
      includeConnector?: boolean;
    } = {},
  ): Promise<Record<string, unknown>> {
    const params: Record<string, unknown> = {};
    const mp = opts.methodProfile ?? this.defaultMethodProfile;
    if (mp) params.method_profile = mp;
    if (opts.limit !== undefined) params.limit = opts.limit;
    if (opts.edition) params.edition = opts.edition;
    if (opts.includePreview !== undefined)
      params.include_preview = boolParam(opts.includePreview);
    if (opts.includeConnector !== undefined)
      params.include_connector = boolParam(opts.includeConnector);

    const resp = await this.get(
      `/factors/${encodeURIComponent(factorId)}/alternates`,
      params,
    );
    if (resp.data && typeof resp.data === 'object' && !Array.isArray(resp.data)) {
      return resp.data as Record<string, unknown>;
    }
    return { raw: resp.data };
  }

  // =====================================================================
  // Batch resolution
  // =====================================================================

  async resolveBatch(
    requests: Array<ResolutionRequest | Record<string, unknown>>,
    opts: { edition?: string } = {},
  ): Promise<BatchJobHandle> {
    const body = {
      requests: requests.map((r) => ({ ...(r as Record<string, unknown>) })),
    };
    const params: Record<string, unknown> = {};
    if (opts.edition) params.edition = opts.edition;
    const resp = await this.post('/factors/resolve/batch', body, params);
    return resp.data as BatchJobHandle;
  }

  async getBatchJob(jobId: string): Promise<BatchJobHandle> {
    const resp = await this.get(
      `/factors/jobs/${encodeURIComponent(jobId)}`,
      null,
      false,
    );
    return resp.data as BatchJobHandle;
  }

  async waitForBatch(
    job: BatchJobHandle | string,
    opts: {
      pollIntervalMs?: number;
      timeoutMs?: number;
    } = {},
  ): Promise<BatchJobHandle> {
    const pollIntervalMs = opts.pollIntervalMs ?? 2000;
    const timeoutMs = opts.timeoutMs ?? 600_000;
    const jobId = typeof job === 'string' ? job : job.job_id;
    const deadline = timeoutMs > 0 ? Date.now() + timeoutMs : null;

    // eslint-disable-next-line no-constant-condition
    while (true) {
      const current = await this.getBatchJob(jobId);
      if (isTerminalBatchStatus(current.status) || TERMINAL_STATES.has(current.status)) {
        if (current.status === 'failed') {
          throw new FactorsAPIError(
            `Batch job ${jobId} failed: ${current.error_message ?? 'unknown error'}`,
            { context: { job_id: jobId, status: current.status } },
          );
        }
        return current;
      }
      if (deadline !== null && Date.now() > deadline) {
        throw new FactorsAPIError(
          `Timeout waiting for batch job ${jobId} (status=${current.status})`,
          { context: { job_id: jobId, timeout_ms: timeoutMs } },
        );
      }
      await new Promise((r) => setTimeout(r, pollIntervalMs));
    }
  }

  // =====================================================================
  // Editions
  // =====================================================================

  async listEditions(
    opts: { includePending?: boolean } = {},
  ): Promise<Edition[]> {
    const params: Record<string, unknown> = {
      include_pending: boolParam(opts.includePending ?? true),
    };
    const resp = await this.get('/editions', params);
    const data = resp.data as { editions?: Edition[] } | unknown;
    if (data && typeof data === 'object' && 'editions' in data) {
      return (data as { editions?: Edition[] }).editions ?? [];
    }
    return [];
  }

  async getEdition(editionId: string): Promise<Record<string, unknown>> {
    const resp = await this.get(
      `/editions/${encodeURIComponent(editionId)}/changelog`,
    );
    if (resp.data && typeof resp.data === 'object' && !Array.isArray(resp.data)) {
      return resp.data as Record<string, unknown>;
    }
    return { raw: resp.data };
  }

  async diff(
    factorId: string,
    leftEdition: string,
    rightEdition: string,
  ): Promise<FactorDiff> {
    const params = {
      left_edition: leftEdition,
      right_edition: rightEdition,
    };
    const resp = await this.get(
      `/factors/${encodeURIComponent(factorId)}/diff`,
      params,
    );
    return resp.data as FactorDiff;
  }

  async auditBundle(
    factorId: string,
    opts: { edition?: string } = {},
  ): Promise<AuditBundle> {
    const params: Record<string, unknown> = {};
    if (opts.edition) params.edition = opts.edition;
    const resp = await this.get(
      `/factors/${encodeURIComponent(factorId)}/audit-bundle`,
      params,
    );
    return resp.data as AuditBundle;
  }

  // =====================================================================
  // Sources / method packs
  // =====================================================================

  async listSources(opts: { edition?: string } = {}): Promise<Source[]> {
    const params: Record<string, unknown> = {};
    if (opts.edition) params.edition = opts.edition;
    const resp = await this.get('/factors/source-registry', params);
    return pickListField<Source>(resp.data, ['sources', 'items']);
  }

  async getSource(sourceId: string): Promise<Source> {
    const resp = await this.get(`/factors/sources/${encodeURIComponent(sourceId)}`);
    return resp.data as Source;
  }

  async listMethodPacks(): Promise<MethodPack[]> {
    const resp = await this.get('/method-packs');
    return pickListField<MethodPack>(resp.data, ['method_packs', 'items']);
  }

  async getMethodPack(methodPackId: string): Promise<MethodPack> {
    const resp = await this.get(`/method-packs/${encodeURIComponent(methodPackId)}`);
    return resp.data as MethodPack;
  }

  // =====================================================================
  // Tenant overrides
  // =====================================================================

  async setOverride(
    override: Override | Record<string, unknown>,
  ): Promise<Override> {
    const body: Record<string, unknown> = { ...(override as Record<string, unknown>) };
    const resp = await this.post('/factors/overrides', body);
    return resp.data as Override;
  }

  async listOverrides(
    opts: { tenantId?: string } = {},
  ): Promise<Override[]> {
    const params: Record<string, unknown> = {};
    if (opts.tenantId) params.tenant_id = opts.tenantId;
    const resp = await this.get('/factors/overrides', params);
    return pickListField<Override>(resp.data, ['overrides', 'items']);
  }
}

function pickListField<T>(data: unknown, keys: string[]): T[] {
  if (data && typeof data === 'object' && !Array.isArray(data)) {
    const obj = data as Record<string, unknown>;
    for (const k of keys) {
      if (Array.isArray(obj[k])) return obj[k] as T[];
    }
    return [];
  }
  if (Array.isArray(data)) return data as T[];
  return [];
}
