# Error handling

The SDK maps every HTTP error to a typed exception class so you can
`instanceof` them rather than sniffing status codes.

## Hierarchy

```
FactorsAPIError           (base; any HTTP error or network failure)
├── AuthError             (401 — missing/invalid auth)
├── TierError             (403 — tier insufficient)
├── LicenseError          (403 — factor is connector_only / not licensed)
├── FactorNotFoundError   (404 on /factors/ routes)
├── ValidationError       (400 / 422)
└── RateLimitError        (429; carries `retryAfter`)
```

Every instance exposes:

| Field           | Description                                         |
|-----------------|-----------------------------------------------------|
| `message`       | Human-readable detail (from `detail` / `message`)   |
| `statusCode`    | HTTP status code                                    |
| `responseBody`  | Raw parsed body for post-mortem                     |
| `requestId`     | Value of `X-Request-ID` (if the server sent one)    |
| `remediation`   | Built-in hint — what to do next                     |
| `context`       | Free-form dict (includes job_id, url, etc.)         |

## Typical pattern

```ts
import {
  AuthError,
  FactorNotFoundError,
  FactorsAPIError,
  RateLimitError,
  TierError,
  ValidationError,
} from '@greenlang/factors';

try {
  const factor = await client.getFactor('does-not-exist');
} catch (err) {
  if (err instanceof FactorNotFoundError) {
    console.warn('Factor not found — is your edition pinned correctly?');
  } else if (err instanceof AuthError) {
    await refreshToken();
  } else if (err instanceof RateLimitError) {
    await sleep((err.retryAfter ?? 1) * 1000);
    return retry();
  } else if (err instanceof TierError) {
    console.error('Needs a Pro+ plan:', err.remediation);
  } else if (err instanceof ValidationError) {
    console.error('Request invalid:', err.message, err.responseBody);
  } else if (err instanceof FactorsAPIError) {
    console.error('API error', err.statusCode, err.message);
  } else {
    throw err; // unknown — bubble up
  }
}
```

## Automatic retry

Transport retries with exponential backoff on:

- 429 (honours `Retry-After`)
- 5xx
- Network errors (socket hangup, DNS, etc.)

By default: 3 attempts, up to 30 seconds per sleep. Configure via
`maxRetries` on the client constructor.

## Custom mapping

`errorFromResponse(...)` is exported so you can hand-build the same
error objects if you integrate the SDK's types into your own HTTP
layer:

```ts
import { errorFromResponse } from '@greenlang/factors';

const err = errorFromResponse({
  statusCode: 404,
  url: '/api/v1/factors/xyz',
  body: { detail: 'not found' },
  requestId: 'req_abc',
});
```
