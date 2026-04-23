/**
 * factorsClient — v1.2.0 typed wrapper around the GreenLang Factors SDK.
 *
 * Owned by Agent W4-D. The existing `frontend/src/lib/api/factorsClient.ts`
 * wrapper is kept for admin console endpoints (`/v1/admin/...`) that are
 * not part of the public SDK. This file is the authoritative bridge
 * between the v1.2.0 Python / TypeScript SDK shape and the operator
 * console pages.
 *
 * Responsibilities
 *   - Singleton `FactorsClient` configured from `VITE_FACTORS_API_BASE` /
 *     `REACT_APP_FACTORS_API_URL`, API key from localStorage, and the
 *     `editionStore` pin.
 *   - Typed helpers (`safeResolve`, `formatApiError`) that translate the
 *     v1.2.0 error hierarchy into friendly `{ code, message, modal }`
 *     tuples the operator surfaces can render directly.
 *   - Request-id propagation via `FactorsAPIError.requestId`.
 *   - Light retry budget on `RateLimitError`/5xx through the SDK
 *     transport's `maxRetries` (default 3).
 *
 * The SDK is imported from a local path alias (workspace) so the types
 * are v1.2.0 even before the package is published to the npm registry.
 */
import {
  FactorsAPIError,
  FactorCannotResolveSafelyError,
  FactorsClient,
  LicensingGapError,
  RateLimitError,
  type ChosenFactor,
  type DeprecationStatus,
  type LicensingEnvelope,
  type QualityEnvelope,
  type ResolutionRequest,
  type ResolvedFactor,
  type SignedReceiptEnvelope,
  type SourceDescriptor,
  type UncertaintyEnvelope,
} from "@greenlang/factors-sdk";
import { getPinnedEdition } from "./api/editionStore";

// ---------------------------------------------------------------- env

/**
 * Resolve the Factors API base URL from env. Accepts both the Vite-style
 * `VITE_FACTORS_API_BASE` and the legacy CRA `REACT_APP_FACTORS_API_URL`.
 * Falls back to the relative `/api` prefix so the existing reverse-proxy
 * contract in dev + prod keeps resolving.
 */
function resolveBaseUrl(): string {
  const env =
    typeof import.meta !== "undefined"
      ? (import.meta as { env?: Record<string, string | undefined> }).env
      : undefined;
  const fromVite = env?.VITE_FACTORS_API_BASE;
  const fromCra = env?.REACT_APP_FACTORS_API_URL;
  const raw = (fromVite || fromCra || "").trim();
  if (raw.length > 0) return raw.replace(/\/+$/, "");
  return "/api";
}

function resolveApiKey(): string | undefined {
  try {
    const t = window.localStorage.getItem("gl.auth.token");
    if (t && t.trim().length > 0) return t;
  } catch {
    /* ignore */
  }
  return undefined;
}

// ---------------------------------------------------------------- singleton

let _client: FactorsClient | null = null;

/**
 * Get the singleton `FactorsClient`. Re-builds if the edition pin or
 * auth token changed in localStorage since the last call (common in dev
 * when the operator flips roles).
 */
export function getFactorsClient(): FactorsClient {
  const edition = getPinnedEdition() ?? undefined;
  const apiKey = resolveApiKey();
  if (
    _client &&
    _client.pinnedEdition === edition &&
    // We can't read the apiKey back out of the SDK — rebuild when either
    // changes so operator flows pick up new bearer tokens immediately.
    (_client as FactorsClient & { _lastApiKey?: string })._lastApiKey === apiKey
  ) {
    return _client;
  }
  _client = new FactorsClient({
    baseUrl: resolveBaseUrl(),
    apiKey,
    pinnedEdition: edition,
    // Off-site pin enforcement would fail against the dev reverse-proxy.
    verifyGreenlangCert: false,
    maxRetries: 3,
  });
  (_client as FactorsClient & { _lastApiKey?: string })._lastApiKey = apiKey;
  return _client;
}

/** Clear the cached client — tests + auth-state flips use this. */
export function resetFactorsClient(): void {
  if (_client) {
    try {
      _client.close();
    } catch {
      /* ignore */
    }
  }
  _client = null;
}

// ---------------------------------------------------------------- error surface

export interface FriendlyApiError {
  code: string;
  title: string;
  message: string;
  /** When true the caller should render a modal, not just a toast. */
  modal: boolean;
  requestId?: string;
  /** For `FactorCannotResolveSafelyError` — drives the helpful resolver hint. */
  evaluatedCandidates?: number;
  packId?: string;
  methodProfile?: string;
  upgradeUrl?: string;
}

/**
 * Translate a v1.2.0 SDK error (or any thrown value) into a friendly
 * shape we can render uniformly across every page. All operator pages
 * funnel errors through this so the copy is consistent.
 */
export function formatApiError(err: unknown): FriendlyApiError {
  if (err instanceof FactorCannotResolveSafelyError) {
    return {
      code: "factor_cannot_resolve_safely",
      title: "No safe factor could be resolved",
      message:
        `No candidate met the method pack's safety floor` +
        (err.evaluatedCandidatesCount !== undefined
          ? ` (evaluated ${err.evaluatedCandidatesCount} candidates).`
          : `.`) +
        ` Options: try a less strict method pack, upload a customer-specific factor, or contact methodology.`,
      modal: true,
      requestId: err.requestId,
      evaluatedCandidates: err.evaluatedCandidatesCount,
      packId: err.packId,
      methodProfile: err.methodProfile,
    };
  }
  if (err instanceof LicensingGapError) {
    return {
      code: "licensing_gap",
      title: "Upgrade required",
      message: err.message,
      modal: true,
      requestId: err.requestId,
      upgradeUrl: "/pricing",
    };
  }
  if (err instanceof RateLimitError) {
    return {
      code: "rate_limited",
      title: "Rate limited",
      message:
        err.retryAfter !== undefined
          ? `Too many requests — retry after ${err.retryAfter}s.`
          : `Too many requests — please retry shortly.`,
      modal: false,
      requestId: err.requestId,
    };
  }
  if (err instanceof FactorsAPIError) {
    return {
      code: err.errorCode ?? `http_${err.statusCode ?? "unknown"}`,
      title: err.name,
      message: err.message,
      modal: (err.statusCode ?? 0) >= 500,
      requestId: err.requestId,
    };
  }
  if (err instanceof Error) {
    return {
      code: "unknown",
      title: "Unexpected error",
      message: err.message,
      modal: false,
    };
  }
  return {
    code: "unknown",
    title: "Unexpected error",
    message: String(err),
    modal: false,
  };
}

/**
 * Call `client.resolve(...)` and always return a discriminated union so
 * pages can render the safety-floor modal without a try/catch boilerplate.
 */
export async function safeResolve(
  request: ResolutionRequest | Record<string, unknown>,
): Promise<
  | { ok: true; resolved: ResolvedFactor }
  | { ok: false; error: FriendlyApiError }
> {
  try {
    const resolved = await getFactorsClient().resolve(request);
    return { ok: true, resolved };
  } catch (err) {
    return { ok: false, error: formatApiError(err) };
  }
}

// ---------------------------------------------------------------- field helpers

/**
 * Narrow `deprecation_status` to the structured `DeprecationStatus` shape
 * regardless of whether the server returned a string (pre-Wave-2) or an
 * object (Wave 2+). Returns `null` when the factor is active.
 */
export function normalizeDeprecation(
  value: string | DeprecationStatus | null | undefined,
): DeprecationStatus | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "string") {
    if (value === "" || value.toLowerCase() === "active") return null;
    return { status: value };
  }
  const status = (value.status ?? "").toString().toLowerCase();
  if (status === "" || status === "active") return null;
  return value;
}

/** Human label for a license_class value. */
export function licenseClassLabel(cls?: string | null): string {
  switch ((cls ?? "").toLowerCase()) {
    case "certified":
      return "Certified";
    case "preview":
      return "Preview";
    case "connector_only":
      return "Connector-only";
    case "redistributable":
      return "Redistributable";
    default:
      return cls ?? "unknown";
  }
}

export type LicenseBadgeColor = "success" | "warning" | "default" | "info" | "error";

/** Semantic color bucket for a license_class badge. */
export function licenseClassColor(cls?: string | null): LicenseBadgeColor {
  switch ((cls ?? "").toLowerCase()) {
    case "certified":
      return "success";
    case "preview":
      return "warning";
    case "connector_only":
      return "default";
    case "redistributable":
      return "info";
    default:
      return "default";
  }
}

// ---------------------------------------------------------------- re-exports

export type {
  ChosenFactor,
  DeprecationStatus,
  LicensingEnvelope,
  QualityEnvelope,
  ResolutionRequest,
  ResolvedFactor,
  SignedReceiptEnvelope,
  SourceDescriptor,
  UncertaintyEnvelope,
};

export {
  FactorsAPIError,
  FactorCannotResolveSafelyError,
  LicensingGapError,
  RateLimitError,
};
