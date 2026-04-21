/**
 * Authentication providers for the Factors TypeScript SDK.
 *
 * Parity with `greenlang/factors/sdk/python/auth.py`:
 *
 *   1. JWTAuth     -> Authorization: Bearer <jwt>
 *   2. APIKeyAuth  -> X-API-Key: <key>
 *   3. HMACAuth    -> request-signing wrapper for Pro+ tiers
 *
 * All implementations are async-aware because HMAC signing requires
 * WebCrypto/Node crypto (both async in the browser).
 */

import { hmacSha256Hex, sha256Hex } from './hash';

export interface AuthContext {
  method: string;
  /** Absolute or relative path (path only — no host/query). */
  path: string;
  body?: Uint8Array;
}

/** Base auth interface — mutates/returns the headers dict. */
export interface AuthProvider {
  applyAuth(
    headers: Record<string, string>,
    ctx: AuthContext,
  ): Promise<Record<string, string>> | Record<string, string>;
}

// ---------------------------------------------------------------------------
// JWT Bearer
// ---------------------------------------------------------------------------

export class JWTAuth implements AuthProvider {
  public readonly token: string;

  constructor(token: string) {
    if (!token) throw new Error('JWTAuth requires a non-empty token');
    this.token = token;
  }

  applyAuth(headers: Record<string, string>): Record<string, string> {
    headers['Authorization'] = 'Bearer ' + this.token;
    return headers;
  }
}

// ---------------------------------------------------------------------------
// API Key
// ---------------------------------------------------------------------------

export interface APIKeyAuthOptions {
  apiKey: string;
  /** Header name (defaults to `X-API-Key` to match the server). */
  headerName?: string;
}

export class APIKeyAuth implements AuthProvider {
  public readonly apiKey: string;
  public readonly headerName: string;

  constructor(opts: APIKeyAuthOptions) {
    if (!opts.apiKey) throw new Error('APIKeyAuth requires a non-empty apiKey');
    this.apiKey = opts.apiKey;
    this.headerName = opts.headerName ?? 'X-API-Key';
  }

  applyAuth(headers: Record<string, string>): Record<string, string> {
    headers[this.headerName] = this.apiKey;
    return headers;
  }
}

// ---------------------------------------------------------------------------
// HMAC request signing (Pro+ tiers)
// ---------------------------------------------------------------------------

export interface HMACAuthOptions {
  apiKeyId: string;
  secret: string;
  primary?: AuthProvider;
  signatureHeader?: string;
  timestampHeader?: string;
  nonceHeader?: string;
  keyIdHeader?: string;
  /** Override clock for deterministic tests (returns seconds since epoch). */
  clock?: () => number;
}

const BASE64URL_ALPHABET =
  'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';

function base64UrlEncode(bytes: Uint8Array): string {
  // Minimal base64url encoder (no '=' padding, '-' '_' alphabet).
  let out = '';
  let i = 0;
  for (; i + 2 < bytes.length; i += 3) {
    const n = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
    out +=
      BASE64URL_ALPHABET[(n >> 18) & 0x3f] +
      BASE64URL_ALPHABET[(n >> 12) & 0x3f] +
      BASE64URL_ALPHABET[(n >> 6) & 0x3f] +
      BASE64URL_ALPHABET[n & 0x3f];
  }
  if (i < bytes.length) {
    const b0 = bytes[i];
    const b1 = i + 1 < bytes.length ? bytes[i + 1] : 0;
    const n = (b0 << 16) | (b1 << 8);
    out += BASE64URL_ALPHABET[(n >> 18) & 0x3f];
    out += BASE64URL_ALPHABET[(n >> 12) & 0x3f];
    if (i + 1 < bytes.length) out += BASE64URL_ALPHABET[(n >> 6) & 0x3f];
  }
  return out;
}

function hexToBytes(hex: string): Uint8Array {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = parseInt(hex.substring(i * 2, i * 2 + 2), 16);
  }
  return out;
}

export class HMACAuth implements AuthProvider {
  public readonly apiKeyId: string;
  public readonly secret: string;
  public readonly primary?: AuthProvider;
  public readonly signatureHeader: string;
  public readonly timestampHeader: string;
  public readonly nonceHeader: string;
  public readonly keyIdHeader: string;
  private readonly clock: () => number;

  constructor(opts: HMACAuthOptions) {
    if (!opts.apiKeyId || !opts.secret) {
      throw new Error('HMACAuth requires apiKeyId and secret');
    }
    this.apiKeyId = opts.apiKeyId;
    this.secret = opts.secret;
    this.primary = opts.primary;
    this.signatureHeader = opts.signatureHeader ?? 'X-GL-Signature';
    this.timestampHeader = opts.timestampHeader ?? 'X-GL-Timestamp';
    this.nonceHeader = opts.nonceHeader ?? 'X-GL-Nonce';
    this.keyIdHeader = opts.keyIdHeader ?? 'X-GL-Key-Id';
    this.clock = opts.clock ?? (() => Math.floor(Date.now() / 1000));
  }

  async applyAuth(
    headers: Record<string, string>,
    ctx: AuthContext,
  ): Promise<Record<string, string>> {
    if (this.primary) {
      const res = await this.primary.applyAuth(headers, ctx);
      // In case the primary returned a new object.
      for (const [k, v] of Object.entries(res)) headers[k] = v;
    }

    const timestamp = String(this.clock());

    const nonceSourceHex = await sha256Hex(
      `${timestamp}:${this.apiKeyId}:${ctx.path}`,
    );
    const nonceBytes = hexToBytes(nonceSourceHex);
    const nonce = base64UrlEncode(nonceBytes).slice(0, 22);

    const bodyDigest = await sha256Hex(ctx.body ?? new Uint8Array(0));
    const canonical = [
      ctx.method.toUpperCase(),
      ctx.path,
      timestamp,
      nonce,
      bodyDigest,
    ].join('\n');

    const signature = await hmacSha256Hex(this.secret, canonical);

    headers[this.keyIdHeader] = this.apiKeyId;
    headers[this.timestampHeader] = timestamp;
    headers[this.nonceHeader] = nonce;
    headers[this.signatureHeader] = 'sha256=' + signature;
    return headers;
  }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

export async function composeAuthHeaders(
  auth: AuthProvider | undefined,
  baseHeaders: Record<string, string>,
  ctx: AuthContext,
): Promise<Record<string, string>> {
  const out: Record<string, string> = { ...baseHeaders };
  if (auth) {
    const res = await auth.applyAuth(out, ctx);
    for (const [k, v] of Object.entries(res)) out[k] = v;
  }
  return out;
}
