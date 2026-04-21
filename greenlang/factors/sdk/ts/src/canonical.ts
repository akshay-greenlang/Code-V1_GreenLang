/**
 * Canonical JSON serialisation, matching Python
 * `json.dumps(payload, sort_keys=True, default=str)`.
 *
 * Cross-language parity is load-bearing for webhook HMAC verification.
 * JavaScript's `JSON.stringify` does not sort keys, so we walk the
 * value tree recursively and emit keys in alphabetical order.
 *
 * Rules applied (mirroring Python's ``json.dumps`` defaults):
 *   - Object keys emitted in code-point ascending order (sort_keys=True).
 *   - Non-finite numbers (NaN, Infinity) serialised as "NaN" / "Infinity"
 *     (Python default; JSON spec deviation).
 *   - undefined and functions are dropped inside objects (like Python
 *     doesn't encounter them); inside arrays they become `null` to
 *     match JSON.stringify-on-array behaviour.
 *   - Dates serialised via `.toISOString()` (Python default=str would
 *     call `str(dt)`; ISO-8601 is the only stable cross-lang choice).
 *   - bigint serialised as plain integer literal (no quotes), matching
 *     Python int.
 *
 * The tradeoff: if you mix in non-JSON types (Map/Set/...) you'll get
 * their `String(...)` representation, consistent with Python's
 * `default=str` escape hatch.
 */

function encodeString(s: string): string {
  return JSON.stringify(s);
}

function encodeNumber(n: number): string {
  if (!Number.isFinite(n)) {
    // Python default: NaN -> "NaN"; +/-Infinity -> "Infinity" / "-Infinity"
    if (Number.isNaN(n)) return 'NaN';
    return n > 0 ? 'Infinity' : '-Infinity';
  }
  // Integers: render as plain integers. JSON.stringify already normalises.
  return JSON.stringify(n);
}

export interface CanonicalOptions {
  /** Separator after `,` (default: `" "` to match `json.dumps` defaults). */
  itemSep?: string;
  /** Separator after `:` (default: `" "` to match `json.dumps` defaults). */
  kvSep?: string;
}

const DEFAULT_SEPS: Required<CanonicalOptions> = { itemSep: ' ', kvSep: ' ' };
const COMPACT_SEPS: Required<CanonicalOptions> = { itemSep: '', kvSep: '' };

function canonicaliseWith(value: unknown, sep: Required<CanonicalOptions>): string {
  if (value === null) return 'null';

  if (typeof value === 'string') return encodeString(value);
  if (typeof value === 'number') return encodeNumber(value);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'bigint') return value.toString(10);

  if (value instanceof Date) return encodeString(value.toISOString());

  if (Array.isArray(value)) {
    const parts = value.map((v) => {
      if (v === undefined || typeof v === 'function') return 'null';
      return canonicaliseWith(v, sep);
    });
    return '[' + parts.join(',' + sep.itemSep) + ']';
  }

  if (typeof value === 'object') {
    const obj = value as Record<string, unknown>;
    const keys = Object.keys(obj).sort();
    const parts: string[] = [];
    for (const k of keys) {
      const v = obj[k];
      if (v === undefined || typeof v === 'function') continue;
      parts.push(encodeString(k) + ':' + sep.kvSep + canonicaliseWith(v, sep));
    }
    return '{' + parts.join(',' + sep.itemSep) + '}';
  }

  // undefined, function, symbol, etc — fall back to String() ("default=str").
  return encodeString(String(value));
}

/**
 * Serialise `value` into a canonical JSON string.
 *
 * Defaults match Python `json.dumps(value, sort_keys=True, default=str)`
 * with its default separators — that is, a space after `,` and after `:`.
 * Set `options.itemSep = options.kvSep = ""` for the compact form Python
 * uses with `separators=(",", ":")`.
 */
export function canonicalJsonStringify(
  value: unknown,
  options: CanonicalOptions = {},
): string {
  return canonicaliseWith(value, {
    itemSep: options.itemSep ?? DEFAULT_SEPS.itemSep,
    kvSep: options.kvSep ?? DEFAULT_SEPS.kvSep,
  });
}

/** Encode canonical JSON (default separators) as UTF-8 bytes. */
export function canonicalJsonBytes(value: unknown): Uint8Array {
  return new TextEncoder().encode(canonicalJsonStringify(value));
}

/**
 * Compact canonical JSON (`separators=(",", ":")`) — matches
 * `greenlang/factors/sdk/python/transport.py`'s request body encoding.
 * Used by the HTTP transport so HMAC signatures stay byte-stable.
 */
export function canonicalJsonCompactBytes(value: unknown): Uint8Array {
  return new TextEncoder().encode(canonicaliseWith(value, COMPACT_SEPS));
}
