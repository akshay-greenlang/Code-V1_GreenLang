# CLI — `gl-factors`

The `gl-factors` CLI (also available as `greenlang-factors` and `factors` via shell aliases) ships with the Python SDK (`pip install greenlang-factors`) and exposes every core workflow from the shell.

---

## Install

```bash
pip install greenlang-factors==1.2.0

gl-factors --version    # 1.2.0
```

Authenticate via `GL_API_KEY` env var or `--api-key` flag.

---

## `factors resolve`

Resolve a single activity.

```bash
gl-factors resolve \
  --factor-family electricity \
  --quantity 12500 \
  --unit kWh \
  --method-profile corporate_scope2_location_based \
  --jurisdiction IN \
  --valid-at 2026-12-31
```

The command prints a 200-char preview of `audit_text` followed by the full response as JSON.

Options:

| Flag | Notes |
|---|---|
| `--show-full-audit` | Print the full `audit_text` narrative (not just the preview). |
| `--pretty` | Grouped output — chosen factor, method, source+licensing, quality+uncertainty, status, audit. |
| `--edition <id>` | Pin a specific edition. |
| `--output <file>` | Write the raw JSON to a file (so you can pass it to `verify-receipt`). |

See [`api-reference/resolve.md`](../api-reference/resolve.md).

---

## `factors explain`

Run `/explain` on a specific factor with the same inputs as a prior resolve.

```bash
gl-factors explain \
  --factor-id "EF:IN:grid:CEA:FY2024-25:v1" \
  --method-profile corporate_scope2_location_based \
  --quantity 12500 --unit kWh --jurisdiction IN \
  --pretty
```

With `--pretty`, the CLI groups the 16 envelope fields (chosen factor, method, source+licensing, quality+uncertainty, status, audit) above the raw JSON dump.

See [`api-reference/explain.md`](../api-reference/explain.md).

---

## `factors verify-receipt`

Verify a signed receipt offline. The secret is read from a file so it never sits in shell history.

```bash
# Ed25519 (default) — fetches JWKS automatically
gl-factors verify-receipt ./response.json

# HS256 with a shared secret
gl-factors verify-receipt ./response.json --key /secrets/hmac.key
```

Prints `valid=true|false alg=... verification_key_hint=...`. Exit code 0 on valid, 1 on invalid.

See [`concepts/signed_receipt.md`](../concepts/signed_receipt.md).

---

## `factors connector add`

Register a BYO-credentials connector for a commercial source that does not permit redistribution (ecoinvent, IEA, Electricity Maps, EC3, pre-contract Green-e / GLEC / TCR).

```bash
gl-factors connector add \
  --source ecoinvent \
  --credential-id "$ECOINVENT_LICENSE_ID" \
  --license-terms-url https://ecoinvent.org/the-ecoinvent-database/access-the-database/licenses/
```

The command:

1. Displays the publisher's license terms in-line for the user to acknowledge.
2. Stores the credential ID (not the secret) in the tenant connector registry.
3. Verifies the credential with a health-check call against the publisher.
4. Prints a confirmation that factors from this source are now resolvable via the connector at query time (values NOT persisted in the GreenLang catalog).

See [`licensing.md`](../licensing.md) for the carve-out posture.

---

## `factors pin`

Pin a client configuration file to a specific edition so every subsequent CLI command runs against it:

```bash
gl-factors pin --edition builtin-v1.0.0
gl-factors pin --show       # prints current pin
gl-factors pin --clear      # removes the pin
```

---

## `factors search`

```bash
gl-factors search \
  --factor-family electricity \
  --country IN \
  --limit 20 \
  --sort "quality.composite_fqs:desc"
```

---

## `factors sources list`

```bash
gl-factors sources list
gl-factors sources list --redistribution-class open
gl-factors sources get india_cea_co2_baseline
```

---

## `factors method-packs list`

```bash
gl-factors method-packs list
gl-factors method-packs get corporate_scope2_location_based
```

---

## Output formats

| Flag | Output |
|---|---|
| (default) | JSON-pretty |
| `--output-format json` | single-line JSON (for piping) |
| `--output-format yaml` | YAML |
| `--output-format table` | columnar table (list commands only) |

---

## Environment variables

| Var | Purpose |
|---|---|
| `GL_API_KEY` | API key. |
| `GL_BASE_URL` | Override base URL (default `https://api.greenlang.io`). |
| `GL_FACTORS_EDITION` | Default edition pin. |
| `GL_FACTORS_JWKS_URL` | Override JWKS URL for receipt verification. |
| `GL_FACTORS_HMAC_KEY` | HMAC secret file path for HS256 receipts. |

---

## Related

- [Python SDK](python.md), [TypeScript SDK](typescript.md).
- [`quickstart.md`](../quickstart.md).
