/**
 * factorsClient — typed fetch wrapper for the GreenLang Factors API.
 *
 * Owns:
 *   - base URL resolution (VITE_FACTORS_API_BASE, falling back to relative
 *     `/api` so the existing dev proxy + production reverse proxy keep
 *     working unchanged)
 *   - bearer-token attachment from localStorage (`gl.auth.token`)
 *   - X-GL-Edition pin from `editionStore`
 *   - typed errors for 401 / 402 / 403 / 429 with friendly messages —
 *     in particular the `licensing_gap` upgrade prompt that drives the
 *     pricing CTA on the Connector-only badge.
 *
 * NOTE: Endpoint paths follow the backend spec (`/v1/...`). When
 * VITE_FACTORS_API_BASE is unset we prefix `/api` so the deployed shell's
 * existing `/api/v1/...` reverse-proxy contract still resolves; the
 * standalone hosted Factors API can override with its own base URL.
 */
import { getPinnedEdition } from "./editionStore";

// ---------------------------------------------------------------- types

export type FactorTier = "certified" | "preview" | "connector_only" | "deprecated";

export interface CoverageTotals {
  certified: number;
  preview: number;
  connector_only: number;
  deprecated?: number;
  all: number;
}

export interface CoverageByFamily extends CoverageTotals {
  family: string;
}

export interface CoverageResponse {
  edition_id: string;
  totals: CoverageTotals;
  by_family: CoverageByFamily[];
  /** legacy: by_source from older /status/summary callers */
  by_source?: Array<CoverageTotals & { source_id: string }>;
  generated_at: string;
}

export interface FqsComponentScore {
  temporal: number;
  geographic: number;
  technology: number;
  verification: number;
  completeness: number;
}

export interface FqsByFamily {
  family: string;
  count: number;
  mean_fqs: number;
  median_fqs: number;
  /** distribution buckets: {"0-20": n, "20-40": n, ...} */
  distribution: Record<string, number>;
  components_mean: FqsComponentScore;
}

export interface FqsResponse {
  edition_id: string;
  by_family: FqsByFamily[];
  generated_at: string;
}

export interface FactorSummary {
  factor_id: string;
  family?: string;
  fuel_type?: string;
  unit?: string;
  geography?: string;
  scope?: string;
  co2e_per_unit?: number;
  source?: string;
  source_year?: number;
  data_quality_score?: number;
  fqs?: number;
  factor_status?: FactorTier;
  source_id?: string | null;
  license_class?: string | null;
}

export interface FactorSearchResponse {
  factors: FactorSummary[];
  total_count?: number;
  edition_id?: string;
}

export interface FactorExplain {
  factor_id: string;
  resolution_path: Array<{
    step: string;
    source?: string;
    rationale?: string;
    score?: number;
  }>;
  fqs?: number;
  components?: FqsComponentScore;
  citations?: Array<{ id: string; title: string; url?: string }>;
  notes?: string;
  /** raw dump of the explain payload — UIs may render arbitrary fields */
  [extra: string]: unknown;
}

export interface SourceRecord {
  source_id: string;
  display_name?: string;
  cadence?: string;
  health?: "healthy" | "stale" | "error" | "unknown";
  latest_timestamp?: string | null;
  enabled?: boolean;
  factor_count?: number;
}

export interface IngestionRun {
  run_id: string;
  source_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  started_at: string;
  finished_at?: string | null;
  factor_count?: number;
  error?: string | null;
}

export interface MappingSuggestion {
  family: string;
  canonical_key: unknown;
  confidence: number;
  band: "high" | "medium" | "low" | string;
  rationale?: string;
  matched_pattern?: string;
  alternates?: Array<{ family: string; canonical_key: unknown; confidence: number }>;
}

export interface QueueItem {
  review_id: string;
  factor_id?: string;
  family?: string;
  current_status: string;
  proposed_status: string;
  submitted_by: string;
  submitted_at: string;
  rationale: string;
  reviewer: string | null;
  due_date: string | null;
  evidence?: Record<string, unknown> | null;
}

export interface OverrideRecord {
  id: string;
  tenant_id: string;
  factor_id: string;
  override_value: number;
  unit: string;
  valid_from: string;
  valid_to: string | null;
  active: boolean;
  rationale?: string;
}

export interface DiffFieldChange {
  field: string;
  type: "added" | "removed" | "changed";
  old_value?: unknown;
  new_value?: unknown;
}

export interface EditionDiffResponse {
  from_edition: string;
  to_edition: string;
  added_factors: string[];
  removed_factors: string[];
  changed_factors: Array<{ factor_id: string; changes: DiffFieldChange[] }>;
  generated_at: string;
}

export interface ImpactRequest {
  /** Replace by ID. */
  from_factor_id?: string;
  to_factor_id?: string;
  /** ...or a proposed payload (synthetic factor) */
  to_factor_payload?: Record<string, unknown>;
  /** Multi-replace mode (back-compat with the legacy simulator). */
  replaced_factor_ids?: string[];
}

export interface ImpactedComputation {
  computation_id: string;
  tenant_id: string | null;
  factor_id: string;
  old_value: number | null;
  new_value: number | null;
  delta_abs: number | null;
  delta_pct: number | null;
  evidence_bundle: string | null;
}

export interface ImpactReport {
  simulated_at: string;
  affected_factor_ids: string[];
  tenants: string[];
  computation_count: number;
  inventory_count?: number;
  customer_count?: number;
  summary: {
    mean_pct_delta?: number;
    max_pct_delta?: number;
    min_pct_delta?: number;
    [k: string]: number | undefined;
  };
  /** delta_pct -> count, for the histogram */
  distribution?: Record<string, number>;
  computations: ImpactedComputation[];
}

// ---------------------------------------------------------------- errors

export type FactorsErrorCode =
  | "unauthorized"
  | "payment_required"
  | "licensing_gap"
  | "forbidden"
  | "rate_limited"
  | "not_found"
  | "bad_request"
  | "server_error"
  | "network_error";

export class FactorsApiError extends Error {
  readonly status: number;
  readonly code: FactorsErrorCode;
  readonly userMessage: string;
  readonly upgradeUrl?: string;
  readonly raw?: unknown;

  constructor(opts: {
    status: number;
    code: FactorsErrorCode;
    userMessage: string;
    upgradeUrl?: string;
    message?: string;
    raw?: unknown;
  }) {
    super(opts.message ?? opts.userMessage);
    this.name = "FactorsApiError";
    this.status = opts.status;
    this.code = opts.code;
    this.userMessage = opts.userMessage;
    this.upgradeUrl = opts.upgradeUrl;
    this.raw = opts.raw;
  }
}

// ---------------------------------------------------------------- internals

const TOKEN_STORAGE_KEY = "gl.auth.token";

function getBaseUrl(): string {
  // import.meta.env is provided by Vite. Empty string -> relative URLs
  // (works with the dev proxy + the existing `/api/v1/...` reverse-proxy
  // contract used by the AdminPage / RunsPage).
  const fromEnv =
    typeof import.meta !== "undefined" &&
    (import.meta as { env?: Record<string, string | undefined> }).env?.VITE_FACTORS_API_BASE;
  if (fromEnv && fromEnv.trim().length > 0) {
    return fromEnv.replace(/\/+$/, "");
  }
  // Default: relative `/api` prefix so existing reverse-proxy mappings
  // (`/api/v1/...`) keep resolving. The hosted Factors API can override
  // by setting VITE_FACTORS_API_BASE=https://api.greenlang.ai
  return "/api";
}

function getAuthToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
}

function buildHeaders(extra?: HeadersInit): Headers {
  const headers = new Headers(extra ?? {});
  if (!headers.has("Accept")) headers.set("Accept", "application/json");
  const tok = getAuthToken();
  if (tok) headers.set("Authorization", `Bearer ${tok}`);
  const edition = getPinnedEdition();
  if (edition) headers.set("X-GL-Edition", edition);
  return headers;
}

async function readErrorBody(res: Response): Promise<{ message?: string; code?: string; upgrade_url?: string; raw?: unknown }> {
  // Robust to non-JSON responses (HTML 502 pages, etc).
  const ct = res.headers.get("content-type") ?? "";
  if (ct.includes("application/json")) {
    try {
      const body = (await res.json()) as Record<string, unknown>;
      return {
        message:
          typeof body.message === "string"
            ? body.message
            : typeof body.detail === "string"
              ? body.detail
              : undefined,
        code: typeof body.code === "string" ? body.code : undefined,
        upgrade_url: typeof body.upgrade_url === "string" ? body.upgrade_url : undefined,
        raw: body,
      };
    } catch {
      return {};
    }
  }
  try {
    const text = await res.text();
    return { message: text.slice(0, 500) };
  } catch {
    return {};
  }
}

function classifyError(
  status: number,
  body: { message?: string; code?: string; upgrade_url?: string; raw?: unknown },
): FactorsApiError {
  if (status === 401) {
    return new FactorsApiError({
      status,
      code: "unauthorized",
      userMessage: body.message ?? "You're signed out. Please log in again to continue.",
      raw: body.raw,
    });
  }
  if (status === 402 || (status === 403 && body.code === "licensing_gap")) {
    return new FactorsApiError({
      status,
      code: "licensing_gap",
      userMessage:
        body.message ??
        "This factor is in the Connector-only tier and requires an upgraded plan or a customer-supplied connector. Open the Pricing page to unlock.",
      upgradeUrl: body.upgrade_url ?? "/pricing",
      raw: body.raw,
    });
  }
  if (status === 403) {
    return new FactorsApiError({
      status,
      code: "forbidden",
      userMessage:
        body.message ??
        "Your role doesn't grant access here. Contact a workspace admin if you need the `factors:admin` scope.",
      raw: body.raw,
    });
  }
  if (status === 429) {
    return new FactorsApiError({
      status,
      code: "rate_limited",
      userMessage:
        body.message ??
        "Too many requests. The Factors API is throttling — please retry in a few seconds.",
      raw: body.raw,
    });
  }
  if (status === 404) {
    return new FactorsApiError({
      status,
      code: "not_found",
      userMessage: body.message ?? "Resource not found.",
      raw: body.raw,
    });
  }
  if (status >= 400 && status < 500) {
    return new FactorsApiError({
      status,
      code: "bad_request",
      userMessage: body.message ?? `Request failed (${status}).`,
      raw: body.raw,
    });
  }
  return new FactorsApiError({
    status,
    code: "server_error",
    userMessage: body.message ?? "The Factors API is temporarily unavailable. Please retry shortly.",
    raw: body.raw,
  });
}

async function request<T>(
  path: string,
  init?: RequestInit & { query?: Record<string, string | number | undefined | null> },
): Promise<T> {
  const base = getBaseUrl();
  let url = `${base}${path.startsWith("/") ? "" : "/"}${path}`;
  if (init?.query) {
    const sp = new URLSearchParams();
    Object.entries(init.query).forEach(([k, v]) => {
      if (v !== undefined && v !== null && String(v).length > 0) sp.set(k, String(v));
    });
    const qs = sp.toString();
    if (qs) url += (url.includes("?") ? "&" : "?") + qs;
  }
  let res: Response;
  try {
    res = await fetch(url, {
      ...init,
      headers: buildHeaders(init?.headers),
    });
  } catch (e) {
    throw new FactorsApiError({
      status: 0,
      code: "network_error",
      userMessage:
        "Couldn't reach the Factors API. Check your network connection and the API base URL.",
      message: (e as Error).message,
    });
  }
  if (!res.ok) {
    const body = await readErrorBody(res);
    throw classifyError(res.status, body);
  }
  if (res.status === 204) return undefined as unknown as T;
  const ct = res.headers.get("content-type") ?? "";
  if (!ct.includes("application/json")) return undefined as unknown as T;
  return (await res.json()) as T;
}

// ---------------------------------------------------------------- public

export async function getCoverage(): Promise<CoverageResponse> {
  return request<CoverageResponse>("/v1/coverage");
}

export async function getFqs(): Promise<FqsResponse> {
  return request<FqsResponse>("/v1/quality/fqs");
}

export async function searchFactors(
  q: string,
  opts?: { geography?: string; scope?: string; family?: string; limit?: number },
): Promise<FactorSearchResponse> {
  return request<FactorSearchResponse>("/v1/factors", {
    query: {
      q: q || "*",
      geography: opts?.geography,
      scope: opts?.scope,
      family: opts?.family,
      limit: opts?.limit ?? 50,
    },
  });
}

export async function getFactor(id: string): Promise<FactorSummary & Record<string, unknown>> {
  return request<FactorSummary & Record<string, unknown>>(`/v1/factors/${encodeURIComponent(id)}`);
}

export async function explainFactor(id: string): Promise<FactorExplain> {
  return request<FactorExplain>(`/v1/factors/${encodeURIComponent(id)}/explain`);
}

export async function listSources(): Promise<{ sources: SourceRecord[] }> {
  return request<{ sources: SourceRecord[] }>("/v1/admin/sources");
}

export async function ingestSource(sourceId: string): Promise<{ run_id: string; status: string }> {
  return request<{ run_id: string; status: string }>(
    `/v1/admin/sources/ingest`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_id: sourceId }),
    },
  );
}

export async function getSourceRuns(sourceId: string): Promise<{ runs: IngestionRun[] }> {
  return request<{ runs: IngestionRun[] }>(
    `/v1/admin/sources/${encodeURIComponent(sourceId)}/runs`,
  );
}

export async function suggestMapping(taxonomy: string, description: string): Promise<MappingSuggestion> {
  return request<MappingSuggestion>("/v1/admin/mapping/suggest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ taxonomy, description }),
  });
}

export async function confirmMapping(payload: {
  taxonomy: string;
  description: string;
  family: string;
  canonical_key: unknown;
  approved_by?: string;
}): Promise<{ rule_id: string }> {
  return request<{ rule_id: string }>("/v1/admin/mapping/confirm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function getQueue(): Promise<{ items: QueueItem[] }> {
  return request<{ items: QueueItem[] }>("/v1/admin/queue");
}

export async function approve(id: string, opts?: { note?: string }): Promise<{ ok: true; new_status: string }> {
  return request<{ ok: true; new_status: string }>(
    `/v1/admin/queue/${encodeURIComponent(id)}/approve`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ note: opts?.note ?? null }),
    },
  );
}

export async function reject(id: string, reason: string): Promise<{ ok: true; new_status: string }> {
  return request<{ ok: true; new_status: string }>(
    `/v1/admin/queue/${encodeURIComponent(id)}/reject`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reason }),
    },
  );
}

export async function proposeQueueItem(payload: {
  factor_id?: string;
  family?: string;
  proposed_status?: string;
  rationale: string;
  evidence?: Record<string, unknown>;
}): Promise<{ review_id: string }> {
  return request<{ review_id: string }>("/v1/admin/queue", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      proposed_status: payload.proposed_status ?? "preview",
      ...payload,
    }),
  });
}

export async function listOverrides(opts?: { tenant_id?: string }): Promise<{ overrides: OverrideRecord[] }> {
  return request<{ overrides: OverrideRecord[] }>("/v1/admin/overrides", {
    query: { tenant_id: opts?.tenant_id ?? undefined },
  });
}

export async function createOverride(payload: {
  tenant_id: string;
  factor_id: string;
  override_value: number;
  unit: string;
  valid_from: string;
  valid_to?: string | null;
  rationale?: string;
}): Promise<OverrideRecord> {
  return request<OverrideRecord>("/v1/admin/overrides", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function deleteOverride(id: string): Promise<void> {
  return request<void>(`/v1/admin/overrides/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export async function diffEditions(fromEdition: string, toEdition: string): Promise<EditionDiffResponse> {
  return request<EditionDiffResponse>(
    `/v1/admin/diff/${encodeURIComponent(fromEdition)}/${encodeURIComponent(toEdition)}`,
  );
}

export async function simulateImpact(payload: ImpactRequest): Promise<ImpactReport> {
  return request<ImpactReport>("/v1/admin/impact-simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

// ---------------------------------------------------------------- helpers

/**
 * Build a queue-proposal payload from a simulation result so the
 * "Promote to Preview" button on the impact simulator can attach the
 * full simulation as evidence.
 */
export function proposeQueueItemFromSimulation(args: {
  fromFactorId?: string;
  toFactorId?: string;
  rationale: string;
  report: ImpactReport;
}): {
  factor_id?: string;
  rationale: string;
  proposed_status: string;
  evidence: Record<string, unknown>;
} {
  return {
    factor_id: args.toFactorId ?? args.fromFactorId,
    proposed_status: "preview",
    rationale: args.rationale,
    evidence: {
      simulation: {
        affected_factor_ids: args.report.affected_factor_ids,
        computation_count: args.report.computation_count,
        tenant_count: args.report.tenants.length,
        inventory_count: args.report.inventory_count,
        customer_count: args.report.customer_count,
        mean_pct_delta: args.report.summary.mean_pct_delta,
        max_pct_delta: args.report.summary.max_pct_delta,
        min_pct_delta: args.report.summary.min_pct_delta,
        simulated_at: args.report.simulated_at,
      },
      from_factor_id: args.fromFactorId,
      to_factor_id: args.toFactorId,
    },
  };
}

/**
 * Validate / normalize an impact-simulation request from the form fields.
 * Used by the simulator UI to keep validation logic testable.
 */
export function composeImpactRequest(input: {
  fromFactorId?: string;
  toFactorId?: string;
  toFactorPayloadJson?: string;
  bulkIds?: string;
}): ImpactRequest {
  const trimmed = (s?: string) => (s ?? "").trim();
  const from = trimmed(input.fromFactorId);
  const to = trimmed(input.toFactorId);
  const payloadStr = trimmed(input.toFactorPayloadJson);
  const bulk = trimmed(input.bulkIds);

  if (bulk.length > 0) {
    const ids = bulk
      .split(/[\n,\s]+/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (ids.length === 0) {
      throw new Error("Enter at least one factor_id to replace.");
    }
    return { replaced_factor_ids: ids };
  }

  if (!from) {
    throw new Error("Specify a `from_factor_id` (the factor being replaced).");
  }
  if (!to && !payloadStr) {
    throw new Error("Specify either a `to_factor_id` or a proposed factor payload (JSON).");
  }
  if (to && payloadStr) {
    throw new Error("Choose one of `to_factor_id` or proposed payload — not both.");
  }
  if (payloadStr) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(payloadStr);
    } catch (e) {
      throw new Error(`Proposed factor payload is not valid JSON: ${(e as Error).message}`);
    }
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("Proposed factor payload must be a JSON object.");
    }
    return { from_factor_id: from, to_factor_payload: parsed as Record<string, unknown> };
  }
  return { from_factor_id: from, to_factor_id: to };
}
