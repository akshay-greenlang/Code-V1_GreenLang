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
import {
  CertificatePinError,
  EditionMismatchError,
  EditionPinError,
  EntitlementError,
  FactorsAPIError,
  LicensingGapError,
  RateLimitError,
} from './errors';
import {
  ReceiptVerificationError,
  VerifiedReceipt,
  VerifyReceiptOptions,
  verifyReceipt as verifyReceiptStandalone,
} from './verify';
import {
  AuditBundle,
  BatchJobHandle,
  CoverageReport,
  Edition,
  Factor,
  FactorDiff,
  FactorMatch,
  MethodPack,
  MethodPackCoverageReport,
  Override,
  ResolutionRequest,
  ResolvedFactor,
  SearchResponse,
  Source,
  inflateMethodPackCoverage,
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
// Certificate pinning (Node) — fingerprint of the bundled GreenLang CA.
// ---------------------------------------------------------------------------

/**
 * SHA-256 fingerprint of the bundled GreenLang CA, in colon-separated
 * uppercase hex (the format Node's `tls.checkServerIdentity` emits).
 *
 * The real fingerprint is substituted at package build time. The value
 * below is a fixture so the SDK can be imported in dev / test without
 * the build pipeline having run. Customers auditing the pin should read
 * it from `client.getPinFingerprint()`.
 */
export const GREENLANG_CA_FINGERPRINT_SHA256 =
  'PLACEHOLDER_SHA256_FINGERPRINT_INJECTED_AT_BUILD_TIME_DO_NOT_TRUST_IN_PRODUCTION';

/** Hostnames matched for automatic pin enforcement (suffix match). */
export const PINNED_HOST_SUFFIXES: readonly string[] = ['greenlang.io'];

function normalizeFingerprint(fp: string): string {
  return (fp || '').replace(/:/g, '').toLowerCase().trim();
}

function hostMatchesPin(host: string): boolean {
  const h = (host || '').toLowerCase().replace(/^\.+|\.+$/g, '');
  return PINNED_HOST_SUFFIXES.some((suffix) => {
    const s = suffix.toLowerCase().replace(/^\.+|\.+$/g, '');
    return h === s || h.endsWith('.' + s);
  });
}

/**
 * Subresource Integrity (SRI) helper for docs / static asset hosting.
 *
 * Browser TLS validation (backed by the OS trust store) already defends
 * fetches to `*.greenlang.io` from MITM — you do NOT need an app-layer
 * pin inside a browser. For static assets pulled from a CDN, however,
 * customers often want a byte-for-byte integrity check; this helper
 * produces the `<script integrity="...">` attribute value for a given
 * payload.
 *
 * Node-only (uses the Web Crypto `subtle` API which is also available
 * in modern browsers).
 */
export async function computeSubresourceIntegrity(
  payload: string | Uint8Array,
  algorithm: 'sha256' | 'sha384' | 'sha512' = 'sha384',
): Promise<string> {
  const subtle = (globalThis as unknown as { crypto?: { subtle?: SubtleCrypto } })
    .crypto?.subtle;
  if (!subtle) {
    throw new Error(
      'Web Crypto subtle API unavailable — computeSubresourceIntegrity ' +
        'requires Node 18+ or a browser.',
    );
  }
  const bytes =
    typeof payload === 'string' ? new TextEncoder().encode(payload) : payload;
  const algoMap: Record<string, string> = {
    sha256: 'SHA-256',
    sha384: 'SHA-384',
    sha512: 'SHA-512',
  };
  const digest = await subtle.digest(algoMap[algorithm], bytes);
  // btoa is available in both modern Node and browsers.
  const b64 =
    typeof btoa === 'function'
      ? btoa(String.fromCharCode(...new Uint8Array(digest)))
      : Buffer.from(new Uint8Array(digest)).toString('base64');
  return `${algorithm}-${b64}`;
}

/** Lazy build of a Node https.Agent that enforces the pin. */
function buildPinnedHttpsAgent(
  expectedFingerprint: string,
): unknown | undefined {
  // The SDK does not want a hard dependency on Node internals — we only
  // try to wire a pinned agent when running under Node with `https`.
  let https: unknown;
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires, global-require
    https = require('https');
  } catch {
    return undefined;
  }
  const Agent = (https as { Agent?: new (opts: unknown) => unknown }).Agent;
  if (!Agent) return undefined;

  const expected = normalizeFingerprint(expectedFingerprint);

  // `checkServerIdentity` receives the leaf cert's fingerprint256 and
  // the full chain via `issuerCertificate`. We walk the chain and
  // accept if ANY presented cert matches the pin.
  const checkServerIdentity = (
    _host: string,
    cert: { fingerprint256?: string; issuerCertificate?: unknown },
  ): Error | undefined => {
    // eslint-disable-next-line @typescript-eslint/no-var-requires, global-require
    const tls = require('tls') as {
      checkServerIdentity: (
        h: string,
        c: { fingerprint256?: string; issuerCertificate?: unknown },
      ) => Error | undefined;
    };
    // Delegate hostname check to Node's default implementation.
    const hostnameErr = tls.checkServerIdentity(_host, cert);
    if (hostnameErr) return hostnameErr;
    let walker: { fingerprint256?: string; issuerCertificate?: unknown } | undefined =
      cert;
    const seen = new Set<unknown>();
    while (walker && !seen.has(walker)) {
      seen.add(walker);
      const fp = normalizeFingerprint(walker.fingerprint256 ?? '');
      if (fp && fp === expected) return undefined;
      walker = walker.issuerCertificate as
        | { fingerprint256?: string; issuerCertificate?: unknown }
        | undefined;
    }
    return new CertificatePinError(
      'GreenLang certificate pin mismatch: no cert in the presented chain matched the bundled SHA-256 pin.',
      {
        expectedFingerprint: expected,
        presentedFingerprint: normalizeFingerprint(cert.fingerprint256 ?? ''),
      },
    );
  };

  return new (Agent as new (opts: unknown) => unknown)({
    keepAlive: true,
    checkServerIdentity,
  });
}

/**
 * Build a FetchLike that injects a Node https.Agent. Falls back to the
 * global fetch unmodified on browsers / platforms where `require` is
 * unavailable.
 */
function wrapFetchWithAgent(agent: unknown): FetchLike | undefined {
  const g = globalThis as unknown as { fetch?: unknown };
  if (typeof g.fetch !== 'function') return undefined;
  const baseFetch = g.fetch.bind(globalThis) as (
    input: string,
    init?: Record<string, unknown>,
  ) => Promise<unknown>;
  return async (input, init) => {
    // Node's fetch respects `dispatcher` (undici) or `agent` (legacy
    // adapters). We pass both so downstream adapters pick whichever
    // they support. Neither causes errors on browser fetch.
    const mergedInit: Record<string, unknown> = {
      ...(init ?? {}),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      agent: (agent as any),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      dispatcher: (agent as any),
    };
    const r = await baseFetch(input as string, mergedInit);
    // The returned object is the native fetch Response; cast through
    // `unknown` to satisfy our FetchResponseLike contract (the
    // transport uses only `status`, `statusText`, `ok`, `headers`,
    // `text`, `url`, all of which are native).
    return r as unknown as import('./transport').FetchResponseLike;
  };
}

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
  /**
   * Enable TLS certificate pinning for `*.greenlang.io` hosts (default: true).
   *
   * In Node, a pinned `https.Agent` is attached that validates any cert
   * in the presented chain matches the bundled GreenLang CA SHA-256
   * fingerprint. In browsers, TLS validation is handled by the user
   * agent — this flag has no effect there (use
   * `computeSubresourceIntegrity()` for static-asset integrity).
   *
   * Disable for air-gapped / corporate-proxy environments that terminate
   * TLS at an intermediate device.
   */
  verifyGreenlangCert?: boolean;
  /**
   * Pinned edition sent as `X-GreenLang-Edition` on every request AND
   * validated against the response header. Prefer `client.pinEdition()`
   * or `client.edition()` for scoped use.
   */
  pinnedEdition?: string;
  /**
   * Override the pin fingerprint. Exposed for tests; in production the
   * bundled `GREENLANG_CA_FINGERPRINT_SHA256` is used.
   */
  expectedCertFingerprint?: string;
}

/** Accepted edition-id formats: `v1.0.0`, `2027.Q1`, `2027.Q1-electricity`, `2027-04-01-freight`. */
const EDITION_ID_REGEX =
  /^(v\d+(?:\.\d+){0,2}(?:-[A-Za-z0-9_]+)?|\d{4}\.Q[1-4](?:-[A-Za-z0-9_]+)?|\d{4}-\d{2}-\d{2}(?:-[A-Za-z0-9_]+)?)$/;

/**
 * Validate an edition id before sending it as a pin header.
 *
 * Throws an `EditionPinError` (loaded lazily so tests don't accidentally
 * import the full client to check format-only behaviour) when the input
 * is empty, the wrong type, or does not match a known edition format.
 */
function validateEditionId(editionId: string): void {
  if (!editionId || typeof editionId !== 'string') {
    throw new EditionPinError(
      'pinEdition() / withEdition() require a non-empty string editionId.',
      { editionId },
    );
  }
  if (!EDITION_ID_REGEX.test(editionId)) {
    throw new EditionPinError(
      `Edition id ${JSON.stringify(editionId)} is not in a recognised format. Use one of: v1.0.0, 2027.Q1, 2027.Q1-electricity, 2027-04-01-freight.`,
      { editionId },
    );
  }
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
  private readonly pinnedEdition_?: string;
  private readonly verifyGreenlangCert_: boolean;
  private readonly expectedCertFingerprint_: string;
  private readonly cloneOpts: FactorsClientOptions;

  constructor(options: FactorsClientOptions) {
    this.apiPrefix = (options.apiPrefix ?? FactorsClient.DEFAULT_API_PREFIX)
      .replace(/\/+$/, '');
    this.defaultMethodProfile = options.methodProfile;
    this.pinnedEdition_ = options.pinnedEdition;
    this.verifyGreenlangCert_ = options.verifyGreenlangCert ?? true;
    this.expectedCertFingerprint_ =
      options.expectedCertFingerprint ?? GREENLANG_CA_FINGERPRINT_SHA256;
    this.cloneOpts = { ...options };

    const timeoutMs =
      options.timeoutMs ??
      (options.timeout !== undefined ? options.timeout * 1000 : DEFAULT_TIMEOUT_MS);

    // Merge the edition pin into the per-request header bag so every
    // route picks it up via the transport's `extraHeaders`.
    const extraHeaders: Record<string, string> = {
      ...(options.extraHeaders ?? {}),
    };
    if (this.pinnedEdition_ && !('X-GreenLang-Edition' in extraHeaders)) {
      extraHeaders['X-GreenLang-Edition'] = this.pinnedEdition_;
    }

    // In Node, attach a pinned https.Agent via a wrapped fetch if the
    // caller enabled pinning and targeted a known GreenLang host AND
    // did not supply their own fetchImpl.
    let fetchImpl: FetchLike | undefined = options.fetchImpl;
    if (fetchImpl === undefined && this.verifyGreenlangCert_) {
      try {
        const host = new URL(options.baseUrl).hostname;
        if (hostMatchesPin(host)) {
          const agent = buildPinnedHttpsAgent(this.expectedCertFingerprint_);
          if (agent !== undefined) {
            fetchImpl = wrapFetchWithAgent(agent);
          }
        }
      } catch {
        // URL parse failure or require('https') failure: fall through
        // without pinning. Clients running in the browser land here
        // because `require` is unavailable, which is exactly what we
        // want: browser TLS is handled by the user agent.
      }
    }

    this.transport = new Transport({
      baseUrl: options.baseUrl,
      auth: resolveAuth(options),
      timeoutMs,
      maxRetries: options.maxRetries ?? DEFAULT_MAX_RETRIES,
      userAgent: options.userAgent ?? DEFAULT_USER_AGENT,
      defaultEdition: options.defaultEdition ?? options.edition,
      cache: options.cache,
      extraHeaders,
      fetchImpl,
      sleep: options.sleep,
    });
  }

  /**
   * SHA-256 fingerprint of the bundled GreenLang CA pin. Exposed for
   * customer audits — record this in your onboarding checklist to
   * detect unexpected SDK rotations.
   */
  getPinFingerprint(): string {
    return normalizeFingerprint(this.expectedCertFingerprint_);
  }

  /** Currently pinned edition, if any. */
  get pinnedEdition(): string | undefined {
    return this.pinnedEdition_;
  }

  /**
   * Return a NEW client with the edition pin set.
   *
   * The new client shares auth + cache state but sends
   * `X-GreenLang-Edition: {editionId}` on every request and validates
   * the response header on the way back.
   */
  pinEdition(editionId: string): FactorsClient {
    validateEditionId(editionId);
    return new FactorsClient({
      ...this.cloneOpts,
      pinnedEdition: editionId,
    });
  }

  /**
   * Run `fn` with a freshly-pinned client. The parent client's pin is
   * unaffected. Mirrors Python's `with client.edition(...)` context manager.
   */
  async edition<T>(
    editionId: string,
    fn: (scoped: FactorsClient) => Promise<T>,
  ): Promise<T> {
    const scoped = this.pinEdition(editionId);
    try {
      return await fn(scoped);
    } finally {
      scoped.close();
    }
  }

  /** Alias for `edition()`. */
  async withEdition<T>(
    editionId: string,
    fn: (scoped: FactorsClient) => Promise<T>,
  ): Promise<T> {
    return this.edition(editionId, fn);
  }

  /**
   * Verify a signed-receipt-bearing response **offline**.
   *
   * Convenience wrapper around the standalone {@link verifyReceiptStandalone}
   * function so the SDK exposes a single import surface.
   *
   * @param response Parsed response object, JSON string, or raw bytes.
   * @param options Verification options (secret, jwksUrl, algorithm).
   * @returns Verified-receipt summary.
   * @throws ReceiptVerificationError when verification fails.
   */
  async verifyReceipt(
    response: unknown,
    options: VerifyReceiptOptions = {},
  ): Promise<VerifiedReceipt> {
    return verifyReceiptStandalone(response, options);
  }

  /** Enforce the edition pin against a response. No-op if no pin. */
  private assertEditionPin(
    resp: TransportResponse<unknown>,
    path: string,
  ): void {
    if (!this.pinnedEdition_) return;
    const returned = resp.edition;
    if (!returned) return;
    if (returned !== this.pinnedEdition_) {
      throw new EditionMismatchError(
        `Server returned edition ${JSON.stringify(returned)} but client is pinned to ${JSON.stringify(
          this.pinnedEdition_,
        )} (path=${path}).`,
        {
          pinnedEdition: this.pinnedEdition_,
          returnedEdition: returned,
          context: { path, request_id: resp.requestId },
        },
      );
    }
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

  private async get<T = unknown>(
    suffix: string,
    params?: Record<string, unknown> | null,
    useCache = true,
  ): Promise<TransportResponse<T>> {
    const fullPath = this.path(suffix);
    const resp = await this.transport.request<T>('GET', fullPath, {
      params: params ?? undefined,
      useCache,
    });
    this.assertEditionPin(resp, fullPath);
    return resp;
  }

  private async post<T = unknown>(
    suffix: string,
    jsonBody?: unknown,
    params?: Record<string, unknown> | null,
  ): Promise<TransportResponse<T>> {
    const fullPath = this.path(suffix);
    const resp = await this.transport.request<T>('POST', fullPath, {
      jsonBody,
      params: params ?? undefined,
      useCache: false,
    });
    this.assertEditionPin(resp, fullPath);
    return resp;
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

  /**
   * GET /method-packs/coverage — canonical v1.3 coverage report.
   *
   * Returns a single canonical {@link MethodPackCoverageReport} shape
   * regardless of whether a specific pack was requested via `pack`.
   * When `pack` is supplied the response still includes
   * `packs: [one entry]` plus an `overall` roll-up mirroring the single
   * entry, so downstream code never has to branch on the call mode.
   *
   * The Wave 4-G legacy payload shape is inflated transparently.
   */
  async methodPackCoverage(
    opts: { pack?: string } = {},
  ): Promise<MethodPackCoverageReport> {
    const params: Record<string, unknown> = {};
    if (opts.pack) params.pack = opts.pack;
    const resp = await this.get('/method-packs/coverage', params);
    return inflateMethodPackCoverage(resp.data);
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
