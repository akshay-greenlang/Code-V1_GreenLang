/**
 * Cross-environment SHA-256 + HMAC-SHA256 helpers.
 *
 * Uses WebCrypto (`crypto.subtle`) in browsers/modern Node, with a
 * graceful fallback to the Node `crypto` module (via dynamic import
 * in an async wrapper) on environments where WebCrypto is unavailable.
 */

export type BytesLike = Uint8Array | ArrayBuffer;

/** True when we're running in a Node.js process. */
export function isNode(): boolean {
  return (
    typeof process !== 'undefined' &&
    typeof (process as { versions?: { node?: string } }).versions?.node ===
      'string'
  );
}

function toUint8(x: BytesLike | string): Uint8Array {
  if (typeof x === 'string') return new TextEncoder().encode(x);
  if (x instanceof Uint8Array) return x;
  return new Uint8Array(x);
}

function bytesToHex(bytes: Uint8Array): string {
  let hex = '';
  for (let i = 0; i < bytes.length; i++) {
    hex += bytes[i].toString(16).padStart(2, '0');
  }
  return hex;
}

interface SubtleHost {
  subtle: SubtleCrypto;
}

function getSubtle(): SubtleHost | null {
  const g = globalThis as unknown as {
    crypto?: { subtle?: SubtleCrypto };
  };
  if (g.crypto && typeof g.crypto.subtle?.digest === 'function') {
    return { subtle: g.crypto.subtle };
  }
  return null;
}

/** Compute SHA-256(message) and return the lowercase hex digest. */
export async function sha256Hex(message: BytesLike | string): Promise<string> {
  const bytes = toUint8(message);
  const subtle = getSubtle();
  if (subtle) {
    const digest = await subtle.subtle.digest('SHA-256', bytes);
    return bytesToHex(new Uint8Array(digest));
  }
  // Node fallback
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const nodeCrypto = await import('node:crypto');
  const hash = nodeCrypto.createHash('sha256');
  hash.update(Buffer.from(bytes));
  return hash.digest('hex');
}

/** Compute HMAC-SHA256(secret, message) and return the lowercase hex digest. */
export async function hmacSha256Hex(
  secret: string | BytesLike,
  message: BytesLike | string,
): Promise<string> {
  const secretBytes = toUint8(secret);
  const msgBytes = toUint8(message);

  const subtle = getSubtle();
  if (subtle) {
    const key = await subtle.subtle.importKey(
      'raw',
      secretBytes,
      { name: 'HMAC', hash: 'SHA-256' },
      false,
      ['sign'],
    );
    const sig = await subtle.subtle.sign('HMAC', key, msgBytes);
    return bytesToHex(new Uint8Array(sig));
  }
  // Node fallback
  const nodeCrypto = await import('node:crypto');
  const hmac = nodeCrypto.createHmac('sha256', Buffer.from(secretBytes));
  hmac.update(Buffer.from(msgBytes));
  return hmac.digest('hex');
}

/**
 * Constant-time string compare. Length-mismatched inputs are rejected
 * after a dummy iteration so timing leaks nothing about the length.
 */
export function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) {
    // Still walk to prevent early-return timing leak on length.
    let sink = 0;
    for (let i = 0; i < a.length; i++) sink |= a.charCodeAt(i);
    return false;
  }
  let diff = 0;
  for (let i = 0; i < a.length; i++) {
    diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return diff === 0;
}
