/**
 * Error hierarchy for the GreenLang Factors TypeScript SDK.
 *
 * Parity with `greenlang/factors/sdk/python/errors.py`:
 *
 *   400 -> ValidationError
 *   401 -> AuthError
 *   403 -> TierError (or LicenseError for factor-license messages)
 *   404 -> FactorNotFoundError (on /factors/ paths; else FactorsAPIError)
 *   422 -> ValidationError
 *   429 -> RateLimitError
 *   5xx -> FactorsAPIError
 */

export interface FactorsAPIErrorOptions {
  statusCode?: number;
  responseBody?: unknown;
  requestId?: string;
  errorCode?: string;
  remediation?: string;
  context?: Record<string, unknown>;
}

/** Base class for all Factors SDK errors. */
export class FactorsAPIError extends Error {
  public readonly statusCode?: number;
  public readonly responseBody?: unknown;
  public readonly requestId?: string;
  public readonly errorCode?: string;
  public readonly remediation?: string;
  public readonly context: Record<string, unknown>;

  constructor(message: string, opts: FactorsAPIErrorOptions = {}) {
    super(message);
    this.name = 'FactorsAPIError';
    this.statusCode = opts.statusCode;
    this.responseBody = opts.responseBody;
    this.requestId = opts.requestId;
    this.errorCode = opts.errorCode;
    this.remediation = opts.remediation;
    this.context = { ...(opts.context ?? {}) };
    if (this.statusCode !== undefined) this.context.status_code = this.statusCode;
    if (this.responseBody !== undefined) this.context.response_body = this.responseBody;
    if (this.requestId !== undefined) this.context.request_id = this.requestId;
    // Restore prototype chain for instanceof checks across compiled targets.
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** 401 — authentication failed (missing/invalid JWT or API key). */
export class AuthError extends FactorsAPIError {
  constructor(message: string, opts: FactorsAPIErrorOptions = {}) {
    super(message, {
      ...opts,
      remediation:
        opts.remediation ??
        'Check that your API key or JWT token is valid and has not expired.',
    });
    this.name = 'AuthError';
    Object.setPrototypeOf(this, AuthError.prototype);
  }
}

/** 403 — tier insufficient for requested endpoint. */
export class TierError extends FactorsAPIError {
  constructor(message: string, opts: FactorsAPIErrorOptions = {}) {
    super(message, {
      ...opts,
      remediation:
        opts.remediation ??
        'Upgrade your GreenLang plan (Pro+ / Enterprise) to access this endpoint.',
    });
    this.name = 'TierError';
    Object.setPrototypeOf(this, TierError.prototype);
  }
}

/** 403 — factor is connector_only / redistribution not allowed. */
export class LicenseError extends FactorsAPIError {
  constructor(message: string, opts: FactorsAPIErrorOptions = {}) {
    super(message, {
      ...opts,
      remediation:
        opts.remediation ??
        'This factor is connector_only under its upstream license; contact your account manager for licensed access.',
    });
    this.name = 'LicenseError';
    Object.setPrototypeOf(this, LicenseError.prototype);
  }
}

/** 404 — factor id does not exist in the edition. */
export class FactorNotFoundError extends FactorsAPIError {
  constructor(message: string, opts: FactorsAPIErrorOptions = {}) {
    super(message, {
      ...opts,
      remediation:
        opts.remediation ??
        'Verify the factor_id and edition are correct; use client.search() to discover valid ids.',
    });
    this.name = 'FactorNotFoundError';
    Object.setPrototypeOf(this, FactorNotFoundError.prototype);
  }
}

/** 400/422 — request payload or query parameters invalid. */
export class ValidationError extends FactorsAPIError {
  constructor(message: string, opts: FactorsAPIErrorOptions = {}) {
    super(message, {
      ...opts,
      remediation:
        opts.remediation ??
        'Check required fields and value formats; see the API docs for the endpoint schema.',
    });
    this.name = 'ValidationError';
    Object.setPrototypeOf(this, ValidationError.prototype);
  }
}

/** 429 — caller exceeded the tier rate limit. */
export class RateLimitError extends FactorsAPIError {
  public readonly retryAfter?: number;

  constructor(
    message: string,
    opts: FactorsAPIErrorOptions & { retryAfter?: number } = {},
  ) {
    super(message, {
      ...opts,
      statusCode: opts.statusCode ?? 429,
      remediation:
        opts.remediation ??
        'Back off and retry after the Retry-After interval, or upgrade your plan for higher limits.',
    });
    this.name = 'RateLimitError';
    this.retryAfter = opts.retryAfter;
    if (opts.retryAfter !== undefined) {
      this.context.retry_after = opts.retryAfter;
    }
    Object.setPrototypeOf(this, RateLimitError.prototype);
  }
}

// ---------------------------------------------------------------------------
// Status-code mapper (parity with Python error_from_response)
// ---------------------------------------------------------------------------

function extractDetail(body: unknown): string | null {
  if (body && typeof body === 'object' && !Array.isArray(body)) {
    const obj = body as Record<string, unknown>;
    const d = obj.detail ?? obj.message;
    if (typeof d === 'string') return d;
    if (Array.isArray(d) && d.length > 0) {
      const first = d[0] as unknown;
      if (first && typeof first === 'object') {
        const fo = first as Record<string, unknown>;
        const m = fo.msg ?? fo.message;
        if (typeof m === 'string') return m;
      }
      return String(first);
    }
    if (d && typeof d === 'object') {
      const fo = d as Record<string, unknown>;
      const m = fo.msg ?? fo.message;
      if (typeof m === 'string') return m;
    }
  }
  if (typeof body === 'string' && body.trim()) return body;
  return null;
}

/** Map an HTTP response onto the SDK exception hierarchy. */
export function errorFromResponse(args: {
  statusCode: number;
  url: string;
  body: unknown;
  requestId?: string;
  retryAfter?: number;
}): FactorsAPIError {
  const { statusCode, url, body, requestId, retryAfter } = args;
  const msg = extractDetail(body) ?? `HTTP ${statusCode} from ${url}`;

  if (statusCode === 401) {
    return new AuthError(msg, { statusCode, responseBody: body, requestId });
  }
  if (statusCode === 403) {
    const lower = msg.toLowerCase();
    if (
      lower.includes('license') ||
      lower.includes('connector_only') ||
      lower.includes('redistribution')
    ) {
      return new LicenseError(msg, { statusCode, responseBody: body, requestId });
    }
    return new TierError(msg, { statusCode, responseBody: body, requestId });
  }
  if (statusCode === 404 && url.includes('/factors/')) {
    return new FactorNotFoundError(msg, {
      statusCode,
      responseBody: body,
      requestId,
    });
  }
  if (statusCode === 400 || statusCode === 422) {
    return new ValidationError(msg, { statusCode, responseBody: body, requestId });
  }
  if (statusCode === 429) {
    return new RateLimitError(msg, {
      statusCode,
      responseBody: body,
      requestId,
      retryAfter,
    });
  }
  return new FactorsAPIError(msg, {
    statusCode,
    responseBody: body,
    requestId,
  });
}
