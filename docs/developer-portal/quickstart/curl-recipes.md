# cURL Recipes for CI/CD

Shell-only snippets for automating GreenLang Factors inside CI pipelines, cron jobs, make targets, and shell scripts. No SDK, no language runtime — just `curl`, `jq`, and `openssl`.

All examples assume:

```bash
export GL_API_BASE="https://api.greenlang.io"
export GL_API_KEY="gl_pk_your_key_here"
export GL_FACTORS_SIGNING_SECRET="shh_hmac_key"
export GL_EDITION="2027.Q1-electricity"
```

---

## 1. Smoke test a key

```bash
curl -sS -o /dev/null -w "%{http_code}\n" \
  -H "Authorization: Bearer $GL_API_KEY" \
  "$GL_API_BASE/api/v1/factors/coverage"
```

Expect `200`. `401` means the key is invalid, `403` means tier too low, `429` means rate-limited.

---

## 2. Resolve a factor (Scope 1 diesel, US)

```bash
curl -sS -X POST "$GL_API_BASE/api/v1/factors/resolve-explain" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Factors-Edition: $GL_EDITION" \
  -d '{
    "activity": "diesel combustion stationary",
    "method_profile": "corporate_scope1",
    "jurisdiction": "US",
    "reporting_date": "2026-06-01"
  }' | jq '{
    factor_id, unit, co2e_per_unit, edition_id,
    step_won: .fallback_rank,
    source: .source.organization
  }'
```

---

## 3. Resolve Scope 2 market-based for a Swedish facility

```bash
curl -sS -X POST "$GL_API_BASE/api/v1/factors/resolve-explain" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Factors-Edition: $GL_EDITION" \
  -d '{
    "activity": "purchased_electricity",
    "method_profile": "corporate_scope2_market_based",
    "jurisdiction": "SE",
    "reporting_date": "2026-06-01",
    "extras": {"certificate_type": "GO"}
  }' | jq '.co2e_per_unit, .gas_breakdown'
```

---

## 4. Extract and verify the HMAC receipt

```bash
curl -sS -X POST "$GL_API_BASE/api/v1/factors/resolve-explain" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Factors-Edition: $GL_EDITION" \
  -d '{"activity":"diesel combustion stationary","method_profile":"corporate_scope1","jurisdiction":"US","reporting_date":"2026-06-01"}' \
  > resolved.json

# Strip receipt, canonicalise, hash.
jq 'del(._signed_receipt)' resolved.json | jq -S -c '.' > payload.json

EXPECTED_HASH=$(openssl dgst -sha256 -hex payload.json | awk '{print $2}')
ACTUAL_HASH=$(jq -r '._signed_receipt.payload_hash' resolved.json)
[ "$EXPECTED_HASH" = "$ACTUAL_HASH" ] || { echo "payload tampered"; exit 1; }

# Re-derive HMAC.
SIG=$(jq -r '._signed_receipt.signature' resolved.json)
EXPECTED_SIG=$(openssl dgst -sha256 -hmac "$GL_FACTORS_SIGNING_SECRET" -binary payload.json | base64)
[ "$SIG" = "$EXPECTED_SIG" ] && echo "OK" || { echo "signature mismatch"; exit 1; }
```

Wrap this in a shell function and call it from any CI step.

---

## 5. Search with facets (for filter UIs)

```bash
curl -sS -H "Authorization: Bearer $GL_API_KEY" \
  "$GL_API_BASE/api/v1/factors/search?q=natural+gas&geography=US&scope=1&limit=25" \
  | jq '.factors[] | {factor_id, co2e_per_unit, data_quality_score}'
```

Advanced POST-body search with sort:

```bash
curl -sS -X POST "$GL_API_BASE/api/v1/factors/search/v2" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "natural gas combustion",
    "filters": {"geography": "US", "scope": "1"},
    "sort": [{"field": "data_quality_score", "order": "desc"}],
    "page": 1,
    "per_page": 50
  }' | jq '.pagination, (.items | length)'
```

---

## 6. List editions (for CI pipelines that need the latest stable)

```bash
curl -sS -H "Authorization: Bearer $GL_API_KEY" \
  "$GL_API_BASE/api/v1/editions?status=stable" \
  | jq -r '.editions[0].edition_id'
```

Save into an env var for subsequent calls:

```bash
export GL_EDITION=$(curl -sS -H "Authorization: Bearer $GL_API_KEY" \
  "$GL_API_BASE/api/v1/editions?status=stable" \
  | jq -r '.editions[0].edition_id')
```

---

## 7. Compare a factor across two editions (diff)

```bash
curl -sS -H "Authorization: Bearer $GL_API_KEY" \
  "$GL_API_BASE/api/v1/factors/EF:US:diesel:2024:v1/diff?from=2026.Q4&to=2027.Q1-electricity" \
  | jq '.changed_fields'
```

Routes to `/api/v1/factors/{factor_id}/diff` — defined in `greenlang/integration/api/routes/factors.py`.

---

## 8. Submit a batch resolution job

For large-volume resolution (> 1k rows) use the batch endpoint instead of the sync `/resolve-explain`:

```bash
# Submit.
JOB_ID=$(curl -sS -X POST "$GL_API_BASE/api/v1/factors/batch/submit" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Factors-Edition: $GL_EDITION" \
  -d @requests.json \
  | jq -r '.job_id')

# Poll until terminal.
while true; do
  STATUS=$(curl -sS -H "Authorization: Bearer $GL_API_KEY" \
    "$GL_API_BASE/api/v1/factors/batch/$JOB_ID" | jq -r '.status')
  echo "$STATUS"
  case "$STATUS" in
    completed|failed|cancelled) break ;;
  esac
  sleep 10
done

# Download results (paginated).
curl -sS -H "Authorization: Bearer $GL_API_KEY" \
  "$GL_API_BASE/api/v1/factors/batch/$JOB_ID/results?page=1&per_page=1000"
```

---

## 9. Retry with backoff on 429

```bash
resolve_with_backoff() {
  local attempt=0
  local max=5
  while [ "$attempt" -lt "$max" ]; do
    HTTP=$(curl -sS -o resolved.json -w "%{http_code}" \
      -X POST "$GL_API_BASE/api/v1/factors/resolve-explain" \
      -H "Authorization: Bearer $GL_API_KEY" \
      -H "Content-Type: application/json" \
      -H "X-Factors-Edition: $GL_EDITION" \
      -D headers.txt \
      -d "$1")
    case "$HTTP" in
      200|201|202) return 0 ;;
      429)
        SLEEP=$(grep -i '^Retry-After:' headers.txt | awk '{print $2}' | tr -d '\r')
        SLEEP=${SLEEP:-5}
        echo "rate limited, sleeping ${SLEEP}s"
        sleep "$SLEEP"
        ;;
      5*) sleep $((2 ** attempt)) ;;
      *)  echo "hard error $HTTP"; return 1 ;;
    esac
    attempt=$((attempt + 1))
  done
  return 1
}
```

---

## 10. GitHub Actions snippet

```yaml
- name: Compute factor for GHG inventory
  env:
    GL_API_BASE: https://api.greenlang.io
    GL_API_KEY: ${{ secrets.GL_API_KEY }}
    GL_FACTORS_SIGNING_SECRET: ${{ secrets.GL_FACTORS_SIGNING_SECRET }}
    GL_EDITION: "2027.Q1-electricity"
  run: |
    curl -sS -fail -X POST "$GL_API_BASE/api/v1/factors/resolve-explain" \
      -H "Authorization: Bearer $GL_API_KEY" \
      -H "X-Factors-Edition: $GL_EDITION" \
      -H "Content-Type: application/json" \
      -d @payload.json \
      > resolved.json
    # Verify receipt (see recipe 4) - fail the build if it fails.
    ./scripts/verify_receipt.sh resolved.json
```

---

## See also

- [Authentication](../api-reference/authentication.md)
- [Rate limits](../api-reference/rate-limits.md)
- [Errors](../api-reference/errors.md)
- [Signed receipts](../concepts/signed-receipts.md)
