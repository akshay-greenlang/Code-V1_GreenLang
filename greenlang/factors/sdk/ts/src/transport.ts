/**
 * HTTP transport layer for the Factors SDK.
 *
 * Parity with `greenlang/factors/sdk/python/transport.py`:
 *
 *   - Exponential backoff on 429 / 5xx / network errors
 *   - Retry-After header honoring
 *   - Transparent ETag response cache (If-None-Match)
 *   - X-Request-ID / X-RateLimit-* / X-Factors-Edition parsing
 *   - Error mapping via `errorFromResponse`
 *
 * The transport is built on the native `fetch` API which is available
 * in modern browsers and Node.js 18+. For older Node releases, users
 * can install `undici` and pass a polyfilled `fetch` via the
 * `fetchImpl` option.
 */

import {
  AuthContext,
  AuthProvider,
  composeAuthHeaders,
} from './auth';
import { canonicalJsonCompactBytes } from './canonical';
import { FactorsAPIError, errorFromResponse } from './errors';

export const DEFAULT_TIMEOUT_MS = 30_000;
export const DEFAULT_MAX_RETRIES = 3;
export const DEFAULT_USER_AGENT = 'greenlang-factors-sdk-ts/1.0.0';

// -----------------------------------------------------------------------------
// Fetch typing (lets us accept custom fetch impls like undici's without a
// hard dependency on DOM lib types).
// -----------------------------------------------------------------------------

export type FetchLike = (
  input: string,
  init?: {
    method?: string;
    headers?: Record<string, string>;
    body?: Uint8Array | string;
    signal?: AbortSignal;
  },
) => Promise<FetchResponseLike>;

export interface FetchResponseLike {
  status: number;
  statusText: string;
  ok: boolean;
  headers: { get(name: string): string | null };
  text(): Promise<string>;
  url?: string;
}

function getGlobalFetch(): FetchLike {
  const g = globalThis as unknown as { fetch?: FetchLike };
  if (typeof g.fetch !== 'function') {
    throw new Error(
      'No global fetch() available. Pass a FetchLike via `fetchImpl` ' +
        '(e.g. `(await import("undici")).fetch`) or run on Node 18+ / a browser.',
    );
  }
  return g.fetch.bind(globalThis) as FetchLike;
}

// -----------------------------------------------------------------------------
// Rate-limit info
// -----------------------------------------------------------------------------

export interface RateLimitInfo {
  limit?: number;
  remaining?: number;
  reset?: number;
  retryAfter?: number;
}

function parseRateLimit(headers: { get(k: string): string | null }): RateLimitInfo {
  const toInt = (v: string | null): number | undefined => {
    if (v === null) return undefined;
    const n = parseInt(v, 10);
    return Number.isFinite(n) ? n : undefined;
  };
  const toFloat = (v: string | null): number | undefined => {
    if (v === null) return undefined;
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : undefined;
  };
  return {
    limit: toInt(headers.get('X-RateLimit-Limit')),
    remaining: toInt(headers.get('X-RateLimit-Remaining')),
    reset: toInt(headers.get('X-RateLimit-Reset')),
    retryAfter: toFloat(headers.get('Retry-After')),
  };
}

// -----------------------------------------------------------------------------
// ETag cache
// -----------------------------------------------------------------------------

interface ETagEntry {
  etag: string;
  data: unknown;
  headers: Record<string, string>;
}

export class ETagCache {
  private readonly entries = new Map<string, ETagEntry>();
  private readonly maxEntries: number;

  constructor(maxEntries: number = 512) {
    this.maxEntries = Math.max(1, maxEntries);
  }

  static key(
    method: string,
    url: string,
    params?: Record<string, unknown> | null,
  ): string {
    let qs = '';
    if (params) {
      const pairs: Array<[string, string]> = [];
      for (const [k, v] of Object.entries(params)) {
        if (v === undefined || v === null) continue;
        pairs.push([k, String(v)]);
      }
      pairs.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
      qs = pairs.map(([k, v]) => `${k}=${v}`).join('&');
    }
    return `${method.toUpperCase()} ${url}?${qs}`;
  }

  get(key: string): ETagEntry | undefined {
    return this.entries.get(key);
  }

  set(
    key: string,
    etag: string,
    data: unknown,
    headers: Record<string, string> = {},
  ): void {
    if (this.entries.size >= this.maxEntries) {
      const firstKey = this.entries.keys().next().value;
      if (firstKey !== undefined) this.entries.delete(firstKey);
    }
    this.entries.set(key, { etag, data, headers: { ...headers } });
  }

  clear(): void {
    this.entries.clear();
  }

  size(): number {
    return this.entries.size;
  }
}

// -----------------------------------------------------------------------------
// Transport response
// -----------------------------------------------------------------------------

export interface TransportResponse<T = unknown> {
  statusCode: number;
  data: T;
  headers: Record<string, string>;
  etag?: string;
  fromCache: boolean;
  requestId?: string;
  rateLimit: RateLimitInfo;
  edition?: string;
}

// -----------------------------------------------------------------------------
// Transport options & class
// -----------------------------------------------------------------------------

export interface TransportOptions {
  baseUrl: string;
  auth?: AuthProvider;
  timeoutMs?: number;
  maxRetries?: number;
  userAgent?: string;
  defaultEdition?: string;
  cache?: ETagCache;
  extraHeaders?: Record<string, string>;
  fetchImpl?: FetchLike;
  /** Override backoff (ms) for tests. */
  sleep?: (ms: number) => Promise<void>;
}

function defaultSleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function shouldRetry(status: number): boolean {
  return status === 429 || (status >= 500 && status < 600);
}

function computeWait(attempt: number, retryAfterSec?: number): number {
  if (retryAfterSec !== undefined && retryAfterSec >= 0) {
    return Math.min(retryAfterSec, 60) * 1000;
  }
  return Math.min(2 ** (attempt - 1), 30) * 1000;
}

function decodeBody(text: string, contentType: string): unknown {
  const ct = (contentType || '').toLowerCase();
  const trimmed = text.trimStart();
  if (
    ct.includes('application/json') ||
    trimmed.startsWith('{') ||
    trimmed.startsWith('[')
  ) {
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  }
  return text;
}

function headersToRecord(
  headers: { get(name: string): string | null },
  knownNames: readonly string[],
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const n of knownNames) {
    const v = headers.get(n);
    if (v !== null) out[n] = v;
  }
  return out;
}

const KNOWN_HEADER_NAMES = [
  'ETag',
  'Content-Type',
  'Content-Length',
  'X-Request-ID',
  'X-RateLimit-Limit',
  'X-RateLimit-Remaining',
  'X-RateLimit-Reset',
  'Retry-After',
  'X-Factors-Edition',
  'X-GreenLang-Edition',
  'Cache-Control',
] as const;

export class Transport {
  public readonly baseUrl: string;
  public readonly cache: ETagCache;

  private readonly auth?: AuthProvider;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;
  private readonly userAgent: string;
  private readonly defaultEdition?: string;
  private readonly extraHeaders: Record<string, string>;
  private readonly fetchImpl: FetchLike;
  private readonly sleep: (ms: number) => Promise<void>;

  constructor(opts: TransportOptions) {
    if (!opts.baseUrl) throw new Error('Transport requires a baseUrl');
    this.baseUrl = opts.baseUrl.replace(/\/+$/, '');
    this.auth = opts.auth;
    this.timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.maxRetries = Math.max(1, opts.maxRetries ?? DEFAULT_MAX_RETRIES);
    this.userAgent = opts.userAgent ?? DEFAULT_USER_AGENT;
    this.defaultEdition = opts.defaultEdition;
    this.cache = opts.cache ?? new ETagCache();
    this.extraHeaders = { ...(opts.extraHeaders ?? {}) };
    this.fetchImpl = opts.fetchImpl ?? getGlobalFetch();
    this.sleep = opts.sleep ?? defaultSleep;
  }

  private buildUrl(path: string, params?: Record<string, unknown> | null): string {
    const norm = path.startsWith('/') ? path : '/' + path;
    let url = this.baseUrl + norm;
    if (params) {
      const entries = Object.entries(params).filter(
        ([, v]) => v !== undefined && v !== null,
      );
      if (entries.length > 0) {
        const qs = entries
          .flatMap(([k, v]) => {
            if (Array.isArray(v)) {
              return v.map(
                (vv) =>
                  encodeURIComponent(k) + '=' + encodeURIComponent(String(vv)),
              );
            }
            return [
              encodeURIComponent(k) + '=' + encodeURIComponent(String(v)),
            ];
          })
          .join('&');
        url += '?' + qs;
      }
    }
    return url;
  }

  private async headersFor(
    method: string,
    path: string,
    body: Uint8Array | undefined,
    extra?: Record<string, string>,
  ): Promise<Record<string, string>> {
    const base: Record<string, string> = {
      Accept: 'application/json',
      'User-Agent': this.userAgent,
      ...this.extraHeaders,
    };
    if (this.defaultEdition) base['X-Factors-Edition'] = this.defaultEdition;
    if (extra) Object.assign(base, extra);
    const ctx: AuthContext = {
      method: method.toUpperCase(),
      path: path.startsWith('/') ? path : '/' + path,
      body,
    };
    return composeAuthHeaders(this.auth, base, ctx);
  }

  async request<T = unknown>(
    method: string,
    path: string,
    opts: {
      params?: Record<string, unknown> | null;
      jsonBody?: unknown;
      extraHeaders?: Record<string, string>;
      useCache?: boolean;
    } = {},
  ): Promise<TransportResponse<T>> {
    const useCache = opts.useCache ?? true;
    const normPath = path.startsWith('/') ? path : '/' + path;
    const url = this.buildUrl(normPath, opts.params ?? null);
    const cacheKey = ETagCache.key(method, normPath, opts.params ?? null);

    let bodyBytes: Uint8Array | undefined;
    if (opts.jsonBody !== undefined) {
      // Compact canonical JSON matches Python transport byte-for-byte so
      // HMAC body digests are identical across clients.
      bodyBytes = canonicalJsonCompactBytes(opts.jsonBody);
    }

    const cached =
      useCache && method.toUpperCase() === 'GET'
        ? this.cache.get(cacheKey)
        : undefined;

    const condExtra: Record<string, string> = { ...(opts.extraHeaders ?? {}) };
    if (cached && !('If-None-Match' in condExtra)) {
      condExtra['If-None-Match'] = cached.etag;
    }

    let headers = await this.headersFor(method, normPath, bodyBytes, condExtra);
    if (bodyBytes !== undefined) {
      headers = { ...headers, 'Content-Type': 'application/json' };
    }

    let lastErr: Error | undefined;
    for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);

      let response: FetchResponseLike;
      try {
        response = await this.fetchImpl(url, {
          method: method.toUpperCase(),
          headers,
          body: bodyBytes,
          signal: controller.signal,
        });
      } catch (exc) {
        clearTimeout(timer);
        lastErr = exc instanceof Error ? exc : new Error(String(exc));
        if (attempt >= this.maxRetries) {
          throw new FactorsAPIError(`Network error: ${lastErr.message}`, {
            context: { attempts: attempt, url: normPath },
          });
        }
        await this.sleep(computeWait(attempt, undefined));
        continue;
      }
      clearTimeout(timer);

      const rateLimit = parseRateLimit(response.headers);
      const requestId = response.headers.get('X-Request-ID') ?? undefined;
      const editionHeader =
        response.headers.get('X-Factors-Edition') ??
        response.headers.get('X-GreenLang-Edition') ??
        undefined;
      const etag = response.headers.get('ETag') ?? undefined;

      if (response.status === 304 && cached) {
        const hdrs = headersToRecord(response.headers, KNOWN_HEADER_NAMES);
        return {
          statusCode: 200,
          data: cached.data as T,
          headers: hdrs,
          etag: cached.etag,
          fromCache: true,
          requestId,
          rateLimit,
          edition: editionHeader,
        };
      }

      if (shouldRetry(response.status) && attempt < this.maxRetries) {
        const wait = computeWait(attempt, rateLimit.retryAfter);
        await this.sleep(wait);
        continue;
      }

      const text = await response.text();
      const body = decodeBody(text, response.headers.get('Content-Type') ?? '');

      if (response.ok) {
        const hdrs = headersToRecord(response.headers, KNOWN_HEADER_NAMES);
        if (etag && method.toUpperCase() === 'GET' && useCache) {
          this.cache.set(cacheKey, etag, body, hdrs);
        }
        return {
          statusCode: response.status,
          data: body as T,
          headers: hdrs,
          etag,
          fromCache: false,
          requestId,
          rateLimit,
          edition: editionHeader,
        };
      }

      throw errorFromResponse({
        statusCode: response.status,
        url: response.url ?? url,
        body,
        requestId,
        retryAfter: rateLimit.retryAfter,
      });
    }

    if (lastErr) {
      throw new FactorsAPIError(
        `Request failed after ${this.maxRetries} attempts: ${lastErr.message}`,
      );
    }
    throw new FactorsAPIError(
      `Request failed after ${this.maxRetries} attempts`,
    );
  }

  close(): void {
    this.cache.clear();
  }
}
