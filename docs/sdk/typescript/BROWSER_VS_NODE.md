# Browser vs Node

The SDK is designed to run unchanged in:

- **Node.js 18+** (native `fetch` + WebCrypto available as `crypto.webcrypto`)
- **Modern browsers** (Chrome ≥ 63, Firefox ≥ 79, Safari ≥ 11.1)
- **Node.js 16** via the [`undici`](https://www.npmjs.com/package/undici) polyfill
- **Cloudflare Workers / Deno / Bun** — tested on each; all provide global `fetch`

## What needs Node?

| Capability               | Node | Browser | Notes                                      |
|--------------------------|------|---------|--------------------------------------------|
| `FactorsClient`          | yes  | yes     | Uses global `fetch`                        |
| `search` / `resolve` etc | yes  | yes     |                                            |
| `verifyWebhook`          | yes  | yes     | Uses WebCrypto in browser, falls back to `node:crypto` |
| `HMACAuth`               | yes  | yes     | Same as above                              |
| CLI (`glfactors`)        | yes  | —       | Requires `node:fs` for `resolve <file>`    |

## Node 16 / older runtimes

Install `undici` and pass its `fetch`:

```ts
import { fetch } from 'undici';
import { FactorsClient } from '@greenlang/factors';

const client = new FactorsClient({
  baseUrl: 'https://api.greenlang.io',
  apiKey: process.env.GL_FACTORS_API_KEY,
  fetchImpl: fetch as unknown as typeof globalThis.fetch,
});
```

## Browsers

Use the ESM entrypoint. Never hard-code secrets in browser code —
use a short-lived JWT issued by your backend:

```ts
import { FactorsClient, JWTAuth } from '@greenlang/factors';

const tokenResp = await fetch('/api/gl-jwt');
const { jwt } = await tokenResp.json();

const client = new FactorsClient({
  baseUrl: 'https://api.greenlang.io',
  auth: new JWTAuth(jwt),
});
```

CORS: the GreenLang API allows all `*.greenlang.io` origins by default.
If you host your SPA elsewhere, contact your account manager to
whitelist the origin.

## Cloudflare Workers

Works out of the box — Workers provide global `fetch` and WebCrypto:

```ts
import { FactorsClient } from '@greenlang/factors';

export default {
  async fetch(request: Request, env: { GL_API_KEY: string }) {
    const client = new FactorsClient({
      baseUrl: 'https://api.greenlang.io',
      apiKey: env.GL_API_KEY,
    });
    const hits = await client.search(new URL(request.url).searchParams.get('q')!);
    return Response.json(hits);
  },
};
```

## Deno

Use the npm specifier:

```ts
import { FactorsClient } from 'npm:@greenlang/factors';
```

## Typed fetch

If your host environment types `fetch` differently (e.g. custom lib),
the SDK accepts any `FetchLike`:

```ts
type FetchLike = (input: string, init?: {
  method?: string;
  headers?: Record<string, string>;
  body?: Uint8Array | string;
  signal?: AbortSignal;
}) => Promise<FetchResponseLike>;
```

Pass via `fetchImpl`. Only `status`, `statusText`, `ok`, `headers.get`,
and `text()` are exercised — no streaming body APIs required.
