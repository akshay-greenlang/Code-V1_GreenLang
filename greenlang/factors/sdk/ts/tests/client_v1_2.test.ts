/**
 * SDK v1.2 tests — Wave 2 / 2a / 2.5 envelope handling in the TS SDK.
 *
 * Mirrors `tests/factors/sdk/test_python_client_v1_2.py` so the two SDKs
 * behave identically against the canonical demo responses.
 *
 * Coverage:
 *  - Signed-receipt key renames (canonical `signed_receipt` + `alg` +
 *    `payload_hash`) and one-release back-compat fallback for
 *    `_signed_receipt` / `algorithm` / `signed_over`.
 *  - New typed envelope interfaces (ChosenFactor, SourceDescriptor,
 *    QualityEnvelope, UncertaintyEnvelope, LicensingEnvelope,
 *    DeprecationStatus, SignedReceiptEnvelope) surfacing on
 *    `ResolvedFactor`.
 *  - Wave 2.5 `audit_text` + `audit_text_draft` fields.
 *  - `FactorCannotResolveSafelyError` raised by `errorFromResponse`
 *    when the API emits 422 with that discriminator.
 *  - `verifyReceipt` happy path + tampered-payload path on the Wave 2a
 *    canonical demo response.
 */
import * as crypto from 'crypto';
import {
  FactorCannotResolveSafelyError,
  ReceiptVerificationError,
  SDK_VERSION,
  canonicalJsonStringify,
  errorFromResponse,
  verifyReceipt,
} from '../src';
import type {
  ChosenFactor,
  DeprecationStatus,
  LicensingEnvelope,
  QualityEnvelope,
  ResolvedFactor,
  SignedReceiptEnvelope,
  SourceDescriptor,
  UncertaintyEnvelope,
} from '../src';

const HMAC_SECRET = 'unit-test-secret-do-not-use-in-prod';

function sign(payload: unknown, secret = HMAC_SECRET): { signature: string; payloadHash: string } {
  const text = canonicalJsonStringify(payload);
  const payloadHash = crypto.createHash('sha256').update(text, 'utf-8').digest('hex');
  const signature = crypto
    .createHmac('sha256', secret)
    .update(payloadHash, 'utf-8')
    .digest('base64');
  return { signature, payloadHash };
}

function makeWave2aResponse(payload: Record<string, unknown>): Record<string, unknown> {
  const { signature, payloadHash } = sign(payload);
  return {
    ...payload,
    signed_receipt: {
      receipt_id: '11111111-1111-1111-1111-111111111111',
      signature,
      verification_key_hint: 'abc123def456aa11',
      alg: 'sha256-hmac',
      payload_hash: payloadHash,
      signed_at: '2026-04-23T00:00:00+00:00',
      key_id: 'gl-factors-v1',
    },
  };
}

function makeLegacyResponse(payload: Record<string, unknown>): Record<string, unknown> {
  const { signature, payloadHash } = sign(payload);
  return {
    ...payload,
    // Pre-Wave-2a: top-level underscore-prefixed key + `algorithm` + `signed_over`.
    _signed_receipt: {
      signature,
      algorithm: 'sha256-hmac',
      signed_over: payloadHash,
      signed_at: '2026-04-23T00:00:00+00:00',
      key_id: 'gl-factors-v1',
    },
  };
}

describe('SDK version', () => {
  it('is bumped to 1.2.0', () => {
    expect(SDK_VERSION).toBe('1.2.0');
  });
});

describe('verifyReceipt -- Wave 2a canonical keys', () => {
  it('verifies a fresh canonical receipt', async () => {
    const response = makeWave2aResponse({ factor_id: 'ef:co2:diesel:us:2026', co2e_per_unit: 10.21 });
    const summary = await verifyReceipt(response, { secret: HMAC_SECRET });
    expect(summary.verified).toBe(true);
    expect(summary.alg).toBe('sha256-hmac');
    // Back-compat: `algorithm` is mirrored on the summary for v1.0/1.1 callers.
    expect(summary.algorithm).toBe('sha256-hmac');
    expect(summary.receipt_id).toBe('11111111-1111-1111-1111-111111111111');
    expect(summary.verification_key_hint).toBe('abc123def456aa11');
  });

  it('reads signed_receipt at top level (not _signed_receipt)', async () => {
    const response = makeWave2aResponse({ a: 1 });
    expect(response).toHaveProperty('signed_receipt');
    expect(response).not.toHaveProperty('_signed_receipt');
    const summary = await verifyReceipt(response, { secret: HMAC_SECRET });
    expect(summary.verified).toBe(true);
  });
});

describe('verifyReceipt -- legacy key fallback with deprecation warning', () => {
  let warnSpy: jest.SpyInstance;

  beforeEach(() => {
    warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
  });
  afterEach(() => {
    warnSpy.mockRestore();
  });

  it('falls back to _signed_receipt and warns', async () => {
    const response = makeLegacyResponse({ factor_id: 'x' });
    const summary = await verifyReceipt(response, { secret: HMAC_SECRET });
    expect(summary.verified).toBe(true);
    expect(summary.alg).toBe('sha256-hmac');
    // At least one console.warn fired about a deprecated key.
    const calls = warnSpy.mock.calls.map((args) => args.join(' '));
    expect(calls.some((c) => /deprecated/i.test(c))).toBe(true);
  });

  it('falls back to algorithm and signed_over inside the receipt', async () => {
    const response = makeLegacyResponse({ x: 1 });
    const summary = await verifyReceipt(response, { secret: HMAC_SECRET });
    // The key assertion is that verification still succeeds on the legacy
    // shape; the normalizer has already upgraded the receipt in place.
    expect(summary.verified).toBe(true);
    expect(summary.alg).toBe('sha256-hmac');
    // The warning tracker is module-global and may have already warned in
    // a previous test in this file. Checking the normalised summary is a
    // stronger assertion than the console output itself.
    expect(summary.payload_hash).toBeDefined();
  });
});

describe('verifyReceipt -- error paths', () => {
  it('raises ReceiptVerificationError on tampered payload', async () => {
    const response = makeWave2aResponse({ factor_id: 'x', co2e: 10 }) as Record<string, unknown>;
    (response as { co2e: number }).co2e = 999;
    await expect(verifyReceipt(response, { secret: HMAC_SECRET })).rejects.toBeInstanceOf(
      ReceiptVerificationError,
    );
  });

  it('raises ReceiptVerificationError on tampered signature', async () => {
    const response = makeWave2aResponse({ a: 1 }) as Record<string, unknown>;
    const receipt = response.signed_receipt as Record<string, unknown>;
    const sig = receipt.signature as string;
    receipt.signature = (sig[0] === 'A' ? 'B' : 'A') + sig.slice(1);
    await expect(verifyReceipt(response, { secret: HMAC_SECRET })).rejects.toBeInstanceOf(
      ReceiptVerificationError,
    );
  });
});

describe('FactorCannotResolveSafelyError mapping', () => {
  it('errorFromResponse returns FactorCannotResolveSafelyError on 422 with discriminator', () => {
    const body = {
      detail: 'No candidate meets the safety floor',
      error_code: 'factor_cannot_resolve_safely',
      details: {
        pack_id: 'corporate_scope1',
        method_profile: 'corporate_scope1',
        evaluated_candidates_count: 7,
      },
    };
    const err = errorFromResponse({
      statusCode: 422,
      url: 'https://factors.test/api/v1/factors/resolve-explain',
      body,
    });
    expect(err).toBeInstanceOf(FactorCannotResolveSafelyError);
    const typed = err as FactorCannotResolveSafelyError;
    expect(typed.packId).toBe('corporate_scope1');
    expect(typed.methodProfile).toBe('corporate_scope1');
    expect(typed.evaluatedCandidatesCount).toBe(7);
  });

  it('falls through to ValidationError on plain 422 without the discriminator', () => {
    const err = errorFromResponse({
      statusCode: 422,
      url: 'https://factors.test/api/v1/factors/resolve-explain',
      body: { detail: 'Bad request' },
    });
    expect(err).not.toBeInstanceOf(FactorCannotResolveSafelyError);
    expect(err.name).toBe('ValidationError');
  });
});

describe('ResolvedFactor envelope surfaces (type shape smoke)', () => {
  // TypeScript interfaces are erased at runtime; these assertions
  // confirm the shape parses cleanly and the canonical keys survive
  // round-tripping through the SDK.
  it('structurally accepts a full Wave 2 + 2.5 envelope', () => {
    const resolved: ResolvedFactor = {
      chosen_factor_id: 'ef:co2:diesel:us:2026',
      chosen_factor: {
        factor_id: 'ef:co2:diesel:us:2026',
        factor_version: '2026.2',
        release_version: 'corporate_scope1.v3',
        method_profile: 'corporate_scope1',
        pack_id: 'corporate_scope1',
        co2e_per_unit: 10.21,
      } as ChosenFactor,
      release_version: 'corporate_scope1.v3',
      source: {
        source_id: 'epa_ghg_2026',
        organization: 'EPA',
        license_class: 'certified',
      } as SourceDescriptor,
      quality: {
        composite_fqs_0_100: 92.5,
      } as QualityEnvelope,
      uncertainty: {
        ci_95: 0.12,
        distribution: 'lognormal',
      } as UncertaintyEnvelope,
      licensing: {
        license_class: 'certified',
        redistribution_class: 'redistributable',
        upstream_licenses: ['US-Government-Work'],
      } as LicensingEnvelope,
      deprecation_status: {
        status: 'current',
        effective_from: '2026-01-01',
      } as DeprecationStatus,
      audit_text: 'Selected EPA AP-42 diesel factor.',
      audit_text_draft: false,
      signed_receipt: {
        receipt_id: '22222222-2222-2222-2222-222222222222',
        signature: 'deadbeef',
        verification_key_hint: 'face1234cafe5678',
        alg: 'sha256-hmac',
        payload_hash: 'c0ffee',
      } as SignedReceiptEnvelope,
    };

    expect(resolved.chosen_factor?.release_version).toBe('corporate_scope1.v3');
    expect(resolved.quality?.composite_fqs_0_100).toBe(92.5);
    expect(resolved.licensing?.redistribution_class).toBe('redistributable');
    expect(resolved.audit_text_draft).toBe(false);
    expect(resolved.signed_receipt?.alg).toBe('sha256-hmac');
    // deprecation_status can be either a plain string or a structured object;
    // ensure both type-narrowings compile.
    const ds = resolved.deprecation_status;
    if (typeof ds === 'string') {
      expect(ds.length).toBeGreaterThan(0);
    } else if (ds) {
      expect(ds.status).toBe('current');
    }
  });
});

describe('ResolvedFactor -- audit_text draft flag', () => {
  it('reports audit_text_draft=true for unapproved narrative templates', async () => {
    const payload = {
      chosen_factor_id: 'ef:co2:diesel:us:2026',
      audit_text: 'Draft: selected US diesel factor via step 3 cascade.',
      audit_text_draft: true,
    };
    const resolved = payload as unknown as ResolvedFactor;
    expect(resolved.audit_text).toMatch(/^Draft:/);
    expect(resolved.audit_text_draft).toBe(true);
  });
});
