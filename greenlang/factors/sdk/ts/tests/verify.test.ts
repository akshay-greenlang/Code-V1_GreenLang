/**
 * Tests for the offline signed-receipt verifier.
 *
 * These exercise the HMAC-SHA256 happy path, the error paths
 * (missing receipt, payload tampering, future timestamp, wrong
 * algorithm), and confirm the verifier strips the receipt block
 * before re-hashing.
 */
import * as crypto from 'crypto';
import {
  ReceiptVerificationError,
  VerifyReceiptOptions,
  canonicalJsonStringify,
  verifyReceipt,
} from '../src';

const SECRET = 'unit-test-secret-do-not-use-in-prod';

/**
 * Compute the receipt the server would attach for `payload` under
 * HMAC-SHA256. Mirrors `greenlang.factors.signing.sign_sha256_hmac`.
 */
function signReceipt(
  payload: unknown,
  opts: { signed_at?: string; key_id?: string; secret?: string } = {},
): {
  signature: string;
  algorithm: 'sha256-hmac';
  signed_at: string;
  key_id: string;
  payload_hash: string;
} {
  const text = canonicalJsonStringify(payload);
  const payloadHash = crypto.createHash('sha256').update(text, 'utf-8').digest('hex');
  const sigBuf = crypto
    .createHmac('sha256', opts.secret ?? SECRET)
    .update(payloadHash, 'utf-8')
    .digest();
  return {
    signature: sigBuf.toString('base64'),
    algorithm: 'sha256-hmac',
    signed_at: opts.signed_at ?? new Date().toISOString(),
    key_id: opts.key_id ?? 'gl-factors-v1',
    payload_hash: payloadHash,
  };
}

function makeResponse(payload: Record<string, unknown>): Record<string, unknown> {
  const receipt = signReceipt(payload);
  return { ...payload, receipt };
}

describe('verifyReceipt -- HMAC-SHA256 happy path', () => {
  it('verifies a fresh receipt and returns the summary', async () => {
    const payload = { factor_id: 'ef:co2:diesel:us:2026', co2e_per_unit: 10.21, unit: 'gal' };
    const response = makeResponse(payload);
    const summary = await verifyReceipt(response, { secret: SECRET });
    expect(summary.verified).toBe(true);
    expect(summary.algorithm).toBe('sha256-hmac');
    expect(summary.key_id).toBe('gl-factors-v1');
    expect(summary.payload_hash).toBeDefined();
  });

  it('verifies a JSON string just like a parsed object', async () => {
    const payload = { x: 1, y: [2, 3] };
    const response = JSON.stringify(makeResponse(payload));
    const summary = await verifyReceipt(response, { secret: SECRET });
    expect(summary.verified).toBe(true);
  });

  it('verifies raw bytes', async () => {
    const payload = { a: 'b' };
    const response = new TextEncoder().encode(JSON.stringify(makeResponse(payload)));
    const summary = await verifyReceipt(response, { secret: SECRET });
    expect(summary.verified).toBe(true);
  });
});

describe('verifyReceipt -- error paths', () => {
  it('throws when the receipt is missing entirely', async () => {
    await expect(verifyReceipt({ factor_id: 'x' }, { secret: SECRET })).rejects.toBeInstanceOf(
      ReceiptVerificationError,
    );
  });

  it('throws when the payload has been tampered with', async () => {
    const response = makeResponse({ a: 1 });
    (response as { a: number }).a = 999; // mutate after signing
    await expect(verifyReceipt(response, { secret: SECRET })).rejects.toThrow(
      /Payload hash mismatch/i,
    );
  });

  it('throws when the secret is wrong', async () => {
    const response = makeResponse({ a: 1 });
    await expect(verifyReceipt(response, { secret: 'WRONG' })).rejects.toThrow(
      /signature does not match/i,
    );
  });

  it('throws when signed_at is in the future beyond tolerance', async () => {
    const future = new Date(Date.now() + 60 * 60 * 1000).toISOString();
    const payload = { x: 1 };
    const receipt = signReceipt(payload, { signed_at: future });
    const response = { ...payload, receipt };
    await expect(
      verifyReceipt(response, { secret: SECRET, futureToleranceSec: 10 }),
    ).rejects.toThrow(/in the future/i);
  });

  it('throws on unknown algorithm', async () => {
    const payload = { x: 1 };
    const receipt = signReceipt(payload);
    const tampered = { ...payload, receipt: { ...receipt, algorithm: 'gibberish' } };
    await expect(verifyReceipt(tampered, { secret: SECRET })).rejects.toThrow(
      /Unknown receipt algorithm/i,
    );
  });

  it('throws when HMAC verification has no secret available', async () => {
    const original = process.env.GL_FACTORS_SIGNING_SECRET;
    delete process.env.GL_FACTORS_SIGNING_SECRET;
    try {
      const response = makeResponse({ a: 1 });
      await expect(verifyReceipt(response)).rejects.toThrow(/requires a secret/i);
    } finally {
      if (original !== undefined) process.env.GL_FACTORS_SIGNING_SECRET = original;
    }
  });
});

describe('verifyReceipt -- meta envelope shape', () => {
  it('finds the receipt under response.meta.receipt and strips it before rehashing', async () => {
    const payload = { meta: { run_id: 'r-42' }, factor_id: 'x' };
    // Build a response where the receipt is nested inside meta:
    const text = canonicalJsonStringify(payload);
    const payloadHash = crypto.createHash('sha256').update(text, 'utf-8').digest('hex');
    const sigBuf = crypto.createHmac('sha256', SECRET).update(payloadHash, 'utf-8').digest();
    const receipt = {
      signature: sigBuf.toString('base64'),
      algorithm: 'sha256-hmac' as const,
      signed_at: new Date().toISOString(),
      key_id: 'gl-factors-v1',
      payload_hash: payloadHash,
    };
    const response = {
      meta: { ...payload.meta, receipt },
      factor_id: payload.factor_id,
    };
    const summary = await verifyReceipt(response, { secret: SECRET });
    expect(summary.verified).toBe(true);
  });
});
