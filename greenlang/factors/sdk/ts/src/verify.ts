/**
 * Offline signed-receipt verifier for the GreenLang Factors TypeScript SDK.
 *
 * Mirrors `greenlang/factors/sdk/python/verify.py` so a customer can pick
 * either SDK and verify the same receipts with bit-identical results.
 *
 * Two algorithms are supported:
 *
 *  - `sha256-hmac` (Community / Developer Pro): symmetric secret shared
 *    via the `GL_FACTORS_SIGNING_SECRET` env var. Verified using Node's
 *    built-in `crypto.createHmac`.
 *  - `ed25519` (Consulting / Platform / Enterprise): asymmetric, verified
 *    against a JWKS document fetched from
 *    `https://api.greenlang.io/.well-known/jwks.json` (or a customer-side
 *    copy passed via `jwksUrl`). Verified using the optional `jose`
 *    package -- install with `npm install jose` if you need Ed25519
 *    receipt verification, otherwise the HMAC path works out of the box.
 */

import { canonicalJsonStringify } from './webhooks';
import { sha256Hex, hmacSha256Hex, constantTimeEqual } from './hash';

/** Maximum drift (seconds) allowed between `signed_at` and the local clock. */
export const DEFAULT_FUTURE_TOLERANCE_SEC = 600;

const DEFAULT_JWKS_URL =
  'https://api.greenlang.io/.well-known/jwks.json';

/**
 * Parsed receipt block.
 *
 * The server may also include `edition_id`, `factor_ids`, and
 * `caller_id` for additional scope assertions; those fields are
 * surfaced verbatim in {@link VerifiedReceipt}.
 */
export interface SignedReceipt {
  signature: string;
  algorithm: 'sha256-hmac' | 'ed25519' | string;
  signed_at?: string;
  key_id?: string;
  payload_hash?: string;
  edition_id?: string;
  factor_ids?: readonly string[];
  caller_id?: string;
}

export interface VerifiedReceipt {
  verified: true;
  algorithm: string;
  key_id: string;
  signed_at: string;
  payload_hash: string;
  edition_id?: string;
  factor_ids?: readonly string[];
  caller_id?: string;
}

export interface VerifyReceiptOptions {
  /** HMAC secret (defaults to `process.env.GL_FACTORS_SIGNING_SECRET`). */
  secret?: string | Uint8Array;
  /** JWKS URL for Ed25519 receipts (defaults to GreenLang's public JWKS). */
  jwksUrl?: string;
  /** Force a specific algorithm; otherwise the receipt's `algorithm` field is trusted. */
  algorithm?: 'sha256-hmac' | 'ed25519';
  /** Maximum forward drift (seconds) on `signed_at`. */
  futureToleranceSec?: number;
}

export class ReceiptVerificationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ReceiptVerificationError';
  }
}

/**
 * Strip the receipt block from a response so the remaining body can be
 * canonically rehashed. Mirrors the server-side `_extract_payload`.
 */
function extractPayload(response: unknown): unknown {
  if (response && typeof response === 'object' && !Array.isArray(response)) {
    const obj = response as Record<string, unknown>;
    if ('receipt' in obj) {
      const cleaned: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(obj)) {
        if (k !== 'receipt') cleaned[k] = v;
      }
      return cleaned;
    }
    if (
      obj.meta &&
      typeof obj.meta === 'object' &&
      'receipt' in (obj.meta as Record<string, unknown>)
    ) {
      const meta = obj.meta as Record<string, unknown>;
      const cleanedMeta: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(meta)) {
        if (k !== 'receipt') cleanedMeta[k] = v;
      }
      const cleaned: Record<string, unknown> = { ...obj, meta: cleanedMeta };
      return cleaned;
    }
  }
  return response;
}

/** Locate the receipt block inside `response`, regardless of its envelope shape. */
function extractReceipt(response: unknown): SignedReceipt | undefined {
  if (!response || typeof response !== 'object') return undefined;
  const obj = response as Record<string, unknown>;
  if (obj.receipt && typeof obj.receipt === 'object') {
    return obj.receipt as SignedReceipt;
  }
  if (obj.meta && typeof obj.meta === 'object') {
    const meta = obj.meta as Record<string, unknown>;
    if (meta.receipt && typeof meta.receipt === 'object') {
      return meta.receipt as SignedReceipt;
    }
  }
  if ('signature' in obj && 'algorithm' in obj) {
    return obj as unknown as SignedReceipt;
  }
  return undefined;
}

/** SHA-256 hex of the canonical JSON of `payload`. */
async function canonicalHashHex(payload: unknown): Promise<string> {
  const text = canonicalJsonStringify(payload);
  return sha256Hex(text);
}

function parseIsoUtc(value: string): Date {
  let s = value.trim();
  if (s.endsWith('Z')) {
    // Date.parse handles "Z" natively; nothing to do.
  }
  const ms = Date.parse(s);
  if (Number.isNaN(ms)) {
    throw new ReceiptVerificationError(
      `Receipt signed_at is not a valid ISO-8601 timestamp: ${value}`,
    );
  }
  return new Date(ms);
}

interface JWKSEntry {
  kid: string;
  kty: string;
  crv?: string;
  x?: string;
  alg?: string;
}

const _JWKS_CACHE = new Map<string, JWKSEntry[]>();

async function fetchJwks(url: string): Promise<JWKSEntry[]> {
  const cached = _JWKS_CACHE.get(url);
  if (cached) return cached;
  let resp: Response;
  try {
    resp = await fetch(url, {
      headers: { Accept: 'application/json' },
    });
  } catch (err) {
    throw new ReceiptVerificationError(
      `Failed to fetch JWKS from ${url}: ${(err as Error).message}`,
    );
  }
  if (!resp.ok) {
    throw new ReceiptVerificationError(
      `JWKS fetch returned HTTP ${resp.status} from ${url}`,
    );
  }
  const doc = (await resp.json()) as { keys?: JWKSEntry[] };
  const keys = doc.keys ?? [];
  _JWKS_CACHE.set(url, keys);
  return keys;
}

function selectJwk(keys: JWKSEntry[], kid: string): JWKSEntry {
  for (const k of keys) {
    if (k.kid === kid) return k;
  }
  throw new ReceiptVerificationError(
    `No JWK with kid=${JSON.stringify(kid)} (available: ${JSON.stringify(
      keys.map((k) => k.kid),
    )})`,
  );
}

function resolveSecretBytes(
  secret: string | Uint8Array | undefined,
): Uint8Array {
  if (secret == null) {
    const env =
      typeof process !== 'undefined' && process.env
        ? process.env.GL_FACTORS_SIGNING_SECRET
        : undefined;
    if (!env) {
      throw new ReceiptVerificationError(
        'HMAC receipt verification requires a secret. Pass `secret` or set GL_FACTORS_SIGNING_SECRET.',
      );
    }
    return new TextEncoder().encode(env);
  }
  if (typeof secret === 'string') return new TextEncoder().encode(secret);
  return secret;
}

async function verifyHmac(
  payloadHashHex: string,
  signatureB64: string,
  secret: Uint8Array,
): Promise<boolean> {
  // Re-HMAC the payload hash and compare base64s in constant time.
  const sigHex = await hmacSha256Hex(secret, payloadHashHex);
  // Convert the freshly computed HMAC hex to base64 so we compare like
  // for like with the receipt.signature value.
  const expectedB64 = Buffer.from(sigHex, 'hex').toString('base64');
  return constantTimeEqual(expectedB64, signatureB64.trim());
}

async function verifyEd25519(
  payloadHashHex: string,
  signatureB64: string,
  publicKeyXB64Url: string,
): Promise<boolean> {
  // The `jose` package is the canonical Node JWS verifier. We import it
  // dynamically so the SDK does not require it for the HMAC path.
  let jose: typeof import('jose');
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    jose = (await import('jose')) as typeof import('jose');
  } catch (err) {
    throw new ReceiptVerificationError(
      'Ed25519 receipt verification requires the optional `jose` package. Install with: npm install jose',
    );
  }
  try {
    const publicKey = await jose.importJWK(
      { kty: 'OKP', crv: 'Ed25519', x: publicKeyXB64Url },
      'EdDSA',
    );
    const signature = b64ToBytes(signatureB64);
    const data = new TextEncoder().encode(payloadHashHex);
    // jose has no standalone "raw signature verify" helper; fall back to
    // the Web Crypto API for Ed25519 raw verification when available.
    const subtle = globalThis.crypto?.subtle;
    if (subtle && typeof subtle.verify === 'function') {
      const cryptoKey = await subtle.importKey(
        'raw',
        b64UrlToBytes(publicKeyXB64Url),
        { name: 'Ed25519' } as unknown as AlgorithmIdentifier,
        false,
        ['verify'],
      );
      return await subtle.verify('Ed25519', cryptoKey, signature, data);
    }
    // jose-based fallback: wrap into a JWS compact form and verify.
    const header = bytesToB64Url(
      new TextEncoder().encode(JSON.stringify({ alg: 'EdDSA' })),
    );
    const payload = bytesToB64Url(data);
    const sig = bytesToB64Url(signature);
    const jws = `${header}.${payload}.${sig}`;
    await jose.compactVerify(jws, publicKey);
    return true;
  } catch (err) {
    if ((err as { name?: string }).name === 'JWSSignatureVerificationFailed') {
      return false;
    }
    throw new ReceiptVerificationError(
      `Ed25519 verification error: ${(err as Error).message}`,
    );
  }
}

function b64ToBytes(s: string): Uint8Array {
  const t = s.trim().replace(/-/g, '+').replace(/_/g, '/');
  const padding = (4 - (t.length % 4)) % 4;
  return new Uint8Array(Buffer.from(t + '='.repeat(padding), 'base64'));
}

function b64UrlToBytes(s: string): Uint8Array {
  return b64ToBytes(s);
}

function bytesToB64Url(b: Uint8Array): string {
  return Buffer.from(b)
    .toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/g, '');
}

/**
 * Verify a signed-receipt-bearing response **offline**.
 *
 * @param response Parsed response body (object), JSON string, or raw bytes.
 * @param options Verification options.
 * @returns Verified-receipt summary.
 * @throws ReceiptVerificationError when verification fails.
 */
export async function verifyReceipt(
  response: unknown,
  options: VerifyReceiptOptions = {},
): Promise<VerifiedReceipt> {
  let parsed: unknown = response;
  if (typeof parsed === 'string') {
    try {
      parsed = JSON.parse(parsed);
    } catch (err) {
      throw new ReceiptVerificationError(
        `Could not parse response string as JSON: ${(err as Error).message}`,
      );
    }
  } else if (parsed instanceof Uint8Array) {
    try {
      parsed = JSON.parse(new TextDecoder('utf-8').decode(parsed));
    } catch (err) {
      throw new ReceiptVerificationError(
        `Could not parse response bytes as JSON: ${(err as Error).message}`,
      );
    }
  }

  const receipt = extractReceipt(parsed);
  if (!receipt) {
    throw new ReceiptVerificationError(
      'Response does not contain a receipt block. Either the server did not sign it (Community tier) or the receipt was stripped in transit.',
    );
  }

  const payload = extractPayload(parsed);
  const expectedHash = await canonicalHashHex(payload);

  if (!receipt.payload_hash || !constantTimeEqual(expectedHash, receipt.payload_hash)) {
    throw new ReceiptVerificationError(
      'Payload hash mismatch: response body has been modified since the receipt was issued.',
    );
  }

  const algorithm = (options.algorithm ?? receipt.algorithm ?? '')
    .toString()
    .toLowerCase()
    .trim();
  if (!algorithm) {
    throw new ReceiptVerificationError('Receipt is missing the algorithm field.');
  }

  if (!receipt.signature) {
    throw new ReceiptVerificationError('Receipt is missing the signature field.');
  }

  const futureTolerance =
    options.futureToleranceSec ?? DEFAULT_FUTURE_TOLERANCE_SEC;
  const signedAtStr = receipt.signed_at ?? '';
  if (signedAtStr) {
    const signedAt = parseIsoUtc(signedAtStr);
    const drift = (signedAt.getTime() - Date.now()) / 1000;
    if (drift > futureTolerance) {
      throw new ReceiptVerificationError(
        `Receipt signed_at is in the future by ${Math.round(drift)}s (tolerance=${futureTolerance}s).`,
      );
    }
  }

  if (algorithm === 'sha256-hmac') {
    const secretBytes = resolveSecretBytes(options.secret);
    const ok = await verifyHmac(expectedHash, receipt.signature, secretBytes);
    if (!ok) {
      throw new ReceiptVerificationError('HMAC-SHA256 receipt signature does not match.');
    }
  } else if (algorithm === 'ed25519') {
    const url = options.jwksUrl ?? DEFAULT_JWKS_URL;
    const keys = await fetchJwks(url);
    const kid = receipt.key_id ?? '';
    if (!kid) {
      throw new ReceiptVerificationError(
        'Ed25519 receipt missing key_id; cannot select JWK.',
      );
    }
    const jwk = selectJwk(keys, kid);
    if ((jwk.kty || '').toUpperCase() !== 'OKP' || (jwk.crv || '').toLowerCase() !== 'ed25519') {
      throw new ReceiptVerificationError(
        `JWK kid=${JSON.stringify(kid)} is not an Ed25519 key.`,
      );
    }
    if (!jwk.x) {
      throw new ReceiptVerificationError(
        `JWK kid=${JSON.stringify(kid)} is missing the public-key 'x' parameter.`,
      );
    }
    const ok = await verifyEd25519(expectedHash, receipt.signature, jwk.x);
    if (!ok) {
      throw new ReceiptVerificationError('Ed25519 receipt signature does not match.');
    }
  } else {
    throw new ReceiptVerificationError(`Unknown receipt algorithm: ${algorithm}`);
  }

  return {
    verified: true,
    algorithm,
    key_id: receipt.key_id ?? '',
    signed_at: signedAtStr,
    payload_hash: expectedHash,
    edition_id: receipt.edition_id,
    factor_ids: receipt.factor_ids,
    caller_id: receipt.caller_id,
  };
}
