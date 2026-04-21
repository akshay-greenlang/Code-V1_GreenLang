import {
  AuthError,
  FactorNotFoundError,
  FactorsAPIError,
  LicenseError,
  RateLimitError,
  TierError,
  ValidationError,
  errorFromResponse,
} from '../src';

describe('errorFromResponse', () => {
  const url = 'https://api.test/api/v1/factors/ef1';

  it('401 → AuthError', () => {
    const e = errorFromResponse({
      statusCode: 401,
      url,
      body: { detail: 'bad token' },
    });
    expect(e).toBeInstanceOf(AuthError);
    expect(e.statusCode).toBe(401);
    expect(e.remediation).toBeTruthy();
  });

  it('403 plain → TierError', () => {
    const e = errorFromResponse({ statusCode: 403, url, body: { detail: 'upgrade' } });
    expect(e).toBeInstanceOf(TierError);
  });

  it('403 with connector_only → LicenseError', () => {
    const e = errorFromResponse({
      statusCode: 403,
      url,
      body: { detail: 'connector_only license' },
    });
    expect(e).toBeInstanceOf(LicenseError);
  });

  it('404 on /factors/ → FactorNotFoundError', () => {
    const e = errorFromResponse({ statusCode: 404, url, body: { detail: 'gone' } });
    expect(e).toBeInstanceOf(FactorNotFoundError);
  });

  it('400 → ValidationError', () => {
    const e = errorFromResponse({ statusCode: 400, url, body: { detail: 'x' } });
    expect(e).toBeInstanceOf(ValidationError);
  });

  it('422 → ValidationError', () => {
    const e = errorFromResponse({ statusCode: 422, url, body: { detail: 'x' } });
    expect(e).toBeInstanceOf(ValidationError);
  });

  it('429 → RateLimitError, retains retryAfter', () => {
    const e = errorFromResponse({
      statusCode: 429,
      url,
      body: { detail: 'slow' },
      retryAfter: 3.5,
    }) as RateLimitError;
    expect(e).toBeInstanceOf(RateLimitError);
    expect(e.retryAfter).toBe(3.5);
  });

  it('500 → FactorsAPIError (base)', () => {
    const e = errorFromResponse({ statusCode: 500, url, body: 'boom' });
    expect(e).toBeInstanceOf(FactorsAPIError);
    expect(e.message).toBe('boom');
  });

  it('extracts nested FastAPI-style detail[].msg', () => {
    const e = errorFromResponse({
      statusCode: 422,
      url,
      body: { detail: [{ msg: 'field required', loc: ['body', 'x'] }] },
    });
    expect(e.message).toBe('field required');
  });
});
