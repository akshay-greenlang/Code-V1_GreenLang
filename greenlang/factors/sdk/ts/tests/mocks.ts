/**
 * Test helpers: a programmable FetchLike mock.
 */
import type { FetchLike, FetchResponseLike } from '../src/transport';

export interface MockResponseSpec {
  status?: number;
  statusText?: string;
  headers?: Record<string, string>;
  body?: unknown;
  /** If set, the body will be returned as a string as-is. */
  text?: string;
  /** When provided, a network error is raised. */
  error?: Error;
  /** Inspect the outgoing request before returning. */
  inspect?: (req: {
    url: string;
    method: string;
    headers: Record<string, string>;
    body?: string;
  }) => void;
}

export interface MockInvocation {
  url: string;
  method: string;
  headers: Record<string, string>;
  body?: string;
}

export function makeMockFetch(responses: MockResponseSpec[]): {
  fetchImpl: FetchLike;
  invocations: MockInvocation[];
} {
  const invocations: MockInvocation[] = [];
  let idx = 0;

  const fetchImpl: FetchLike = async (url, init) => {
    const method = (init?.method ?? 'GET').toUpperCase();
    const headers = { ...(init?.headers ?? {}) };
    const rawBody = init?.body;
    let bodyStr: string | undefined;
    if (rawBody instanceof Uint8Array) {
      bodyStr = new TextDecoder().decode(rawBody);
    } else if (typeof rawBody === 'string') {
      bodyStr = rawBody;
    }
    invocations.push({ url, method, headers, body: bodyStr });

    const spec = responses[Math.min(idx, responses.length - 1)];
    idx += 1;
    if (spec.inspect) {
      spec.inspect({ url, method, headers, body: bodyStr });
    }
    if (spec.error) {
      throw spec.error;
    }

    const status = spec.status ?? 200;
    const statusText = spec.statusText ?? 'OK';
    const hdrs: Record<string, string> = { 'Content-Type': 'application/json', ...(spec.headers ?? {}) };
    const bodyText =
      spec.text !== undefined
        ? spec.text
        : spec.body !== undefined
        ? JSON.stringify(spec.body)
        : '';

    const response: FetchResponseLike = {
      status,
      statusText,
      ok: status >= 200 && status < 300,
      url,
      headers: {
        get(name: string): string | null {
          for (const key of Object.keys(hdrs)) {
            if (key.toLowerCase() === name.toLowerCase()) return hdrs[key];
          }
          return null;
        },
      },
      async text() {
        return bodyText;
      },
    };
    return response;
  };

  return { fetchImpl, invocations };
}

/** Zero sleep for deterministic retry tests. */
export const noSleep = () => Promise.resolve();
