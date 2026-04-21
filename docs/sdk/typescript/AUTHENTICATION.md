# Authentication

The SDK supports three authentication flows, mirroring the Python SDK:

1. **API key** — `X-API-Key: gl_fac_<token>`
2. **JWT bearer** — `Authorization: Bearer <jwt>`
3. **HMAC request signing** — Pro+ tier, layered on top of #1 or #2

Secrets should **never** be hard-coded. Use environment variables,
platform secret managers (GreenLang Vault, AWS SSM, GCP Secret Manager),
or your CI secrets store.

## API key

```ts
import { FactorsClient } from '@greenlang/factors';

const client = new FactorsClient({
  baseUrl: 'https://api.greenlang.io',
  apiKey: process.env.GL_FACTORS_API_KEY!,
});
```

Internally this creates an `APIKeyAuth` that attaches the key to the
`X-API-Key` header on every request.

## JWT

```ts
const client = new FactorsClient({
  baseUrl: 'https://api.greenlang.io',
  jwtToken: process.env.GL_FACTORS_JWT!,
});
```

Attaches `Authorization: Bearer <token>`. When the JWT expires, catch
`AuthError` and refresh from your identity provider — the SDK does not
perform token refresh automatically so you stay in control of the
refresh policy.

## HMAC (Pro+ tiers)

HMAC is composed *around* a primary auth provider. The signing string
is identical to the Python SDK:

```
METHOD\n
PATH\n
X-GL-Timestamp\n
X-GL-Nonce\n
sha256_hex(body)
```

```ts
import { APIKeyAuth, FactorsClient, HMACAuth } from '@greenlang/factors';

const client = new FactorsClient({
  baseUrl: 'https://api.greenlang.io',
  auth: new HMACAuth({
    apiKeyId: process.env.GL_FACTORS_KEY_ID!,
    secret: process.env.GL_FACTORS_HMAC_SECRET!,
    primary: new APIKeyAuth({ apiKey: process.env.GL_FACTORS_API_KEY! }),
  }),
});
```

Four headers are added on every request:

| Header             | Value                                    |
|--------------------|------------------------------------------|
| `X-GL-Key-Id`      | Your API key identifier                  |
| `X-GL-Timestamp`   | Unix seconds (integer)                   |
| `X-GL-Nonce`       | 22-char base64url of SHA-256 seed        |
| `X-GL-Signature`   | `sha256=<hex HMAC>`                      |

The server regenerates the signature using your shared secret and
compares in constant time. Clock skew tolerance is ±5 minutes.

## Custom auth provider

Implement `AuthProvider` to integrate any auth scheme (mTLS token
exchange, Kerberos, bespoke broker):

```ts
import { AuthContext, AuthProvider } from '@greenlang/factors';

class MyAuth implements AuthProvider {
  async applyAuth(headers: Record<string, string>, ctx: AuthContext) {
    const token = await fetchToken();
    headers['Authorization'] = 'Bearer ' + token;
    return headers;
  }
}
```

Pass it via `new FactorsClient({ auth: new MyAuth(), ... })`.
