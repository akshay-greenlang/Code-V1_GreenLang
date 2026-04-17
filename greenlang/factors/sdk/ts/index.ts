/**
 * GreenLang Factors TypeScript SDK
 *
 * Full typed HTTP client for the GreenLang Factors API.
 * Supports: editions, search, match, calculate, export, audit-bundle, diff.
 *
 * Usage:
 *   import { FactorsClient } from '@greenlang/factors-sdk';
 *
 *   const client = new FactorsClient({
 *     baseUrl: 'https://api.greenlang.io/api/v1',
 *     apiKey: 'gl_...',
 *   });
 *
 *   const results = await client.search('diesel US Scope 1');
 *   for (const f of results.factors) {
 *     console.log(f.factor_id, f.co2e_per_unit);
 *   }
 */

export const SDK_VERSION = '1.0.0';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface FactorsConfig {
  baseUrl: string;
  apiKey?: string;
  edition?: string;
  timeout?: number;
  maxRetries?: number;
  retryBackoff?: number;
  userAgent?: string;
}

export interface EditionRow {
  edition_id: string;
  status: string;
  label: string;
  manifest_hash: string;
}

export interface FactorSummary {
  factor_id: string;
  fuel_type: string;
  geography: string;
  scope: string;
  boundary: string;
  co2e_per_unit: number;
  unit: string;
  source: string;
  source_year: number;
  dqs_score: number;
  factor_status: string;
  source_id: string | null;
}

export interface SearchV2Request {
  query: string;
  geography?: string;
  fuel_type?: string;
  scope?: string;
  source_id?: string;
  factor_status?: string;
  license_class?: string;
  dqs_min?: number;
  valid_on_date?: string;
  sector_tags?: string[];
  activity_tags?: string[];
  sort_by?: 'relevance' | 'dqs_score' | 'co2e_total' | 'source_year' | 'factor_id';
  sort_order?: 'asc' | 'desc';
  offset?: number;
  limit?: number;
}

export interface SearchV2Response {
  factors: FactorSummary[];
  total_count: number;
  offset: number;
  limit: number;
  query: string;
  sort_by: string;
  sort_order: string;
}

export interface MatchRequest {
  activity_description: string;
  geography?: string;
  fuel_type?: string;
  scope?: string;
  limit?: number;
}

export interface CalculateRequest {
  fuel_type: string;
  activity_amount: number;
  activity_unit: string;
  geography?: string;
  scope?: string;
  boundary?: string;
}

export interface FactorDiff {
  factor_id: string;
  left_edition: string;
  right_edition: string;
  left_exists: boolean;
  right_exists: boolean;
  status: 'added' | 'removed' | 'changed' | 'unchanged' | 'not_found';
  changes: Array<{
    field: string;
    type: 'added' | 'removed' | 'changed';
    old_value?: unknown;
    new_value?: unknown;
  }>;
}

export interface HealthResponse {
  status: string;
  edition: string;
  factor_count: number;
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

export class FactorsApiError extends Error {
  public readonly statusCode: number;
  public readonly body: string | null;

  constructor(statusCode: number, message: string, body?: string | null) {
    super(`HTTP ${statusCode}: ${message}`);
    this.name = 'FactorsApiError';
    this.statusCode = statusCode;
    this.body = body ?? null;
  }
}

export class FactorsConnectionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'FactorsConnectionError';
  }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

export class FactorsClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly edition?: string;
  private readonly timeout: number;
  private readonly maxRetries: number;
  private readonly retryBackoff: number;
  private readonly userAgent: string;

  constructor(config: FactorsConfig) {
    this.baseUrl = config.baseUrl.replace(/\/+$/, '');
    this.apiKey = config.apiKey;
    this.edition = config.edition;
    this.timeout = config.timeout ?? 60_000;
    this.maxRetries = config.maxRetries ?? 3;
    this.retryBackoff = config.retryBackoff ?? 1_000;
    this.userAgent = config.userAgent ?? `greenlang-factors-sdk-ts/${SDK_VERSION}`;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = {
      Accept: 'application/json',
      'User-Agent': this.userAgent,
    };
    if (this.apiKey) h['Authorization'] = `Bearer ${this.apiKey}`;
    if (this.edition) h['X-Factors-Edition'] = this.edition;
    return h;
  }

  private async request<T = Record<string, unknown>>(
    method: string,
    path: string,
    params?: Record<string, string | number | boolean | undefined>,
    body?: Record<string, unknown>,
  ): Promise<T> {
    let url = `${this.baseUrl}${path}`;
    if (params) {
      const entries = Object.entries(params).filter(
        ([, v]) => v !== undefined && v !== null,
      );
      if (entries.length > 0) {
        const qs = entries.map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`).join('&');
        url = `${url}?${qs}`;
      }
    }

    const headers = this.headers();
    const fetchInit: RequestInit = { method, headers };
    if (body) {
      fetchInit.body = JSON.stringify(body);
      headers['Content-Type'] = 'application/json';
    }

    let lastErr: Error | null = null;
    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), this.timeout);
        fetchInit.signal = controller.signal;

        const resp = await fetch(url, fetchInit);
        clearTimeout(timer);

        if (!resp.ok) {
          const respBody = await resp.text().catch(() => '');
          if ([429, 500, 502, 503, 504].includes(resp.status) && attempt < this.maxRetries - 1) {
            const wait = this.retryBackoff * 2 ** attempt;
            await new Promise((r) => setTimeout(r, wait));
            lastErr = new FactorsApiError(resp.status, resp.statusText, respBody);
            continue;
          }
          throw new FactorsApiError(resp.status, resp.statusText, respBody);
        }

        const text = await resp.text();
        return text ? (JSON.parse(text) as T) : ({} as T);
      } catch (err) {
        if (err instanceof FactorsApiError) throw err;
        if (attempt < this.maxRetries - 1) {
          const wait = this.retryBackoff * 2 ** attempt;
          await new Promise((r) => setTimeout(r, wait));
          lastErr = err instanceof Error ? new FactorsConnectionError(err.message) : new FactorsConnectionError(String(err));
          continue;
        }
        throw err instanceof Error ? new FactorsConnectionError(err.message) : new FactorsConnectionError(String(err));
      }
    }
    throw lastErr ?? new FactorsConnectionError('Max retries exceeded');
  }

  // ---- Edition endpoints ----

  async listEditions(includePending = true): Promise<Record<string, unknown>> {
    return this.request('GET', '/editions', { include_pending: includePending });
  }

  async getChangelog(editionId: string): Promise<Record<string, unknown>> {
    return this.request('GET', `/editions/${editionId}/changelog`);
  }

  async compareEditions(left: string, right: string): Promise<Record<string, unknown>> {
    return this.request('GET', '/editions/compare', { left, right });
  }

  // ---- Factor endpoints ----

  async listFactors(opts?: {
    fuelType?: string;
    geography?: string;
    scope?: string;
    page?: number;
    limit?: number;
    includePreview?: boolean;
    includeConnector?: boolean;
  }): Promise<Record<string, unknown>> {
    return this.request('GET', '/factors', {
      fuel_type: opts?.fuelType,
      geography: opts?.geography,
      scope: opts?.scope,
      page: opts?.page ?? 1,
      limit: opts?.limit ?? 100,
      include_preview: opts?.includePreview ?? false,
      include_connector: opts?.includeConnector ?? false,
    });
  }

  async getFactor(factorId: string): Promise<Record<string, unknown>> {
    return this.request('GET', `/factors/${factorId}`);
  }

  async getProvenance(factorId: string): Promise<Record<string, unknown>> {
    return this.request('GET', `/factors/${factorId}/provenance`);
  }

  async getReplacements(factorId: string): Promise<Record<string, unknown>> {
    return this.request('GET', `/factors/${factorId}/replacements`);
  }

  async getAuditBundle(factorId: string): Promise<Record<string, unknown>> {
    return this.request('GET', `/factors/${factorId}/audit-bundle`);
  }

  async diffFactor(factorId: string, leftEdition: string, rightEdition: string): Promise<FactorDiff> {
    return this.request<FactorDiff>('GET', `/factors/${factorId}/diff`, {
      left_edition: leftEdition,
      right_edition: rightEdition,
    });
  }

  // ---- Search endpoints ----

  async search(
    query: string,
    opts?: { geography?: string; limit?: number; includePreview?: boolean },
  ): Promise<Record<string, unknown>> {
    return this.request('GET', '/factors/search', {
      q: query,
      geography: opts?.geography,
      limit: opts?.limit ?? 20,
      include_preview: opts?.includePreview ?? false,
    });
  }

  async searchV2(req: SearchV2Request): Promise<SearchV2Response> {
    const body: Record<string, unknown> = {
      query: req.query,
      sort_by: req.sort_by ?? 'relevance',
      sort_order: req.sort_order ?? 'desc',
      offset: req.offset ?? 0,
      limit: req.limit ?? 20,
    };
    if (req.geography) body.geography = req.geography;
    if (req.fuel_type) body.fuel_type = req.fuel_type;
    if (req.scope) body.scope = req.scope;
    if (req.source_id) body.source_id = req.source_id;
    if (req.factor_status) body.factor_status = req.factor_status;
    if (req.license_class) body.license_class = req.license_class;
    if (req.dqs_min !== undefined) body.dqs_min = req.dqs_min;
    if (req.valid_on_date) body.valid_on_date = req.valid_on_date;
    if (req.sector_tags) body.sector_tags = req.sector_tags;
    if (req.activity_tags) body.activity_tags = req.activity_tags;
    return this.request<SearchV2Response>('POST', '/factors/search/v2', undefined, body);
  }

  async getFacets(includePreview = false): Promise<Record<string, unknown>> {
    return this.request('GET', '/factors/search/facets', { include_preview: includePreview });
  }

  // ---- Match endpoint ----

  async match(req: MatchRequest): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = {
      activity_description: req.activity_description,
      limit: req.limit ?? 10,
    };
    if (req.geography) body.geography = req.geography;
    if (req.fuel_type) body.fuel_type = req.fuel_type;
    if (req.scope) body.scope = req.scope;
    return this.request('POST', '/factors/match', undefined, body);
  }

  // ---- Calculation endpoints ----

  async calculate(req: CalculateRequest): Promise<Record<string, unknown>> {
    return this.request('POST', '/calculate', undefined, {
      fuel_type: req.fuel_type,
      activity_amount: req.activity_amount,
      activity_unit: req.activity_unit,
      geography: req.geography ?? 'US',
      scope: req.scope ?? '1',
      boundary: req.boundary ?? 'combustion',
    });
  }

  async calculateBatch(calculations: CalculateRequest[]): Promise<Record<string, unknown>> {
    return this.request('POST', '/calculate/batch', undefined, { calculations });
  }

  // ---- Export endpoint ----

  async exportFactors(opts?: {
    status?: string;
    geography?: string;
    fuelType?: string;
    scope?: string;
    sourceId?: string;
    format?: string;
  }): Promise<Record<string, unknown>> {
    return this.request('GET', '/factors/export', {
      status: opts?.status,
      geography: opts?.geography,
      fuel_type: opts?.fuelType,
      scope: opts?.scope,
      source_id: opts?.sourceId,
      format: opts?.format ?? 'json',
    });
  }

  // ---- System endpoints ----

  async health(): Promise<HealthResponse> {
    return this.request<HealthResponse>('GET', '/health');
  }

  async stats(): Promise<Record<string, unknown>> {
    return this.request('GET', '/stats');
  }

  async coverage(): Promise<Record<string, unknown>> {
    return this.request('GET', '/stats/coverage');
  }

  async sourceRegistry(): Promise<Record<string, unknown>> {
    return this.request('GET', '/factors/source-registry');
  }
}

export default FactorsClient;
