/**
 * Webhook signature verifier for GreenLang Factors webhooks.
 *
 * Parity with `greenlang/factors/sdk/python/webhooks.py`:
 *
 *   HMAC-SHA256 over canonical JSON(body with keys sorted alphabetically).
 *
 * Serialisation matches Python `json.dumps(payload, sort_keys=True,
 * default=str)` — see `canonical.ts` for the implementation.
 */

import { canonicalJsonBytes, canonicalJsonStringify } from './canonical';
import { constantTimeEqual, hmacSha256Hex } from './hash';

export class WebhookVerificationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'WebhookVerificationError';
    Object.setPrototypeOf(this, WebhookVerificationError.prototype);
  }
}

/** Compute the canonical HMAC-SHA256 hex signature for an object payload. */
export async function computeSignature(
  payload: unknown,
  secret: string,
): Promise<string> {
  const body = canonicalJsonBytes(payload);
  return hmacSha256Hex(secret, body);
}

/** Compute HMAC-SHA256 over pre-serialised bytes. */
export async function computeSignatureBytes(
  body: Uint8Array | string,
  secret: string,
): Promise<string> {
  return hmacSha256Hex(secret, body);
}

export type WebhookPayload =
  | Record<string, unknown>
  | Array<unknown>
  | Uint8Array
  | string;

/**
 * Constant-time webhook signature verification.
 *
 * Accepts either a parsed payload (dict/array — canonicalised before
 * hashing) or raw bytes/string (hashed as-is).  The `signature` may
 * optionally carry a scheme prefix (e.g. `"sha256="`) which is stripped.
 *
 * Returns `true` on match, `false` otherwise — never throws.
 */
export async function verifyWebhook(
  payload: WebhookPayload,
  signature: string,
  secret: string,
  opts: { scheme?: string } = {},
): Promise<boolean> {
  if (!signature || !secret) return false;

  const sig = signature.trim();
  let sigValue: string;
  if (sig.includes('=') && !/^[0-9a-fA-F=]+$/.test(sig)) {
    const idx = sig.indexOf('=');
    const prefix = sig.slice(0, idx);
    sigValue = sig.slice(idx + 1);
    if (opts.scheme && prefix.toLowerCase() !== opts.scheme.toLowerCase()) {
      return false;
    }
  } else if (sig.startsWith('sha256=')) {
    sigValue = sig.slice('sha256='.length);
  } else {
    sigValue = sig;
  }

  let expected: string;
  if (payload instanceof Uint8Array) {
    expected = await computeSignatureBytes(payload, secret);
  } else if (typeof payload === 'string') {
    expected = await computeSignatureBytes(payload, secret);
  } else if (payload && typeof payload === 'object') {
    expected = await computeSignature(payload, secret);
  } else {
    return false;
  }

  return constantTimeEqual(expected, sigValue.toLowerCase());
}

/** Verify signature over raw request body bytes. */
export async function verifyWebhookBytes(
  body: Uint8Array | string,
  signature: string,
  secret: string,
): Promise<boolean> {
  return verifyWebhook(body, signature, secret);
}

/** Same as verifyWebhook but throws on mismatch. */
export async function verifyWebhookStrict(
  payload: WebhookPayload,
  signature: string,
  secret: string,
): Promise<void> {
  if (!(await verifyWebhook(payload, signature, secret))) {
    throw new WebhookVerificationError(
      'Webhook signature verification failed',
    );
  }
}

/**
 * Produce a signature for `payload` using `secret`. Intended for tests
 * and local development only — never sign webhooks on a client in
 * production; signing is the server's responsibility.
 */
export async function signWebhook(
  payload: WebhookPayload,
  secret: string,
  opts: { scheme?: string } = {},
): Promise<string> {
  let sig: string;
  if (payload instanceof Uint8Array) {
    sig = await computeSignatureBytes(payload, secret);
  } else if (typeof payload === 'string') {
    sig = await computeSignatureBytes(payload, secret);
  } else {
    sig = await computeSignature(payload, secret);
  }
  return opts.scheme ? `${opts.scheme}=${sig}` : `sha256=${sig}`;
}

/** Parse comma-separated key=value signature headers (Stripe-style). */
export function parseSignatureHeader(headerValue: string): Record<string, string> {
  const out: Record<string, string> = {};
  if (!headerValue) return out;
  for (const chunk of headerValue.split(',')) {
    const idx = chunk.indexOf('=');
    if (idx > -1) {
      out[chunk.slice(0, idx).trim()] = chunk.slice(idx + 1).trim();
    }
  }
  return out;
}

// Re-export canonicaliser for consumers who want to pre-hash themselves.
export { canonicalJsonStringify };
