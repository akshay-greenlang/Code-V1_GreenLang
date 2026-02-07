# Entity Resolution Low Confidence

## Alert

**Alert Names:**

| Alert | Severity | Condition |
|-------|----------|-----------|
| `NormalizerEntityResolutionLowConfidence` | Warning | Average entity resolution confidence < 0.7 for 10 minutes |
| `NormalizerHighUnresolvedRate` | Warning | >20% of entity resolution requests return unresolved for 10 minutes |
| `NormalizerReviewQueueBacklog` | Warning | Unresolved entity review queue > 500 items for 30 minutes |

**Thresholds:**

```promql
# NormalizerEntityResolutionLowConfidence
avg(glnorm_entity_resolution_confidence) < 0.7
# sustained for 10 minutes

# NormalizerHighUnresolvedRate
rate(glnorm_entity_resolution_unresolved_total[5m]) /
  rate(glnorm_entity_resolution_total[5m]) > 0.2
# sustained for 10 minutes

# NormalizerReviewQueueBacklog
glnorm_entity_review_queue_size > 500
# sustained for 30 minutes
```

---

## Description

These alerts fire when the entity resolution subsystem of the Unit & Reference Normalizer service (AGENT-FOUND-003) experiences quality degradation. Entity resolution is the process of matching free-text names for fuels, materials, and processes to their canonical GreenLang vocabulary entries.

### How Entity Resolution Works

The normalizer resolves entities through a multi-stage matching pipeline:

1. **Input Normalization** -- The input name is lowercased, stripped of whitespace, and common separators (hyphens, underscores) are standardized
2. **Exact Match** -- Direct lookup in the vocabulary table for an exact match (confidence: 1.0)
3. **Tenant Override Check** -- If a `tenant_id` is provided, check tenant-specific vocabulary mappings first
4. **Fuzzy Match (Substring)** -- Check if the input is a substring of a vocabulary entry or vice versa. Score is calculated as `min(len(input), len(key)) / max(len(input), len(key))` (confidence: 0.5-0.99)
5. **Fuzzy Match (Word Overlap)** -- Split both input and vocabulary key into words, calculate word overlap ratio (confidence: 0.45-0.89, with 0.9x multiplier)
6. **Threshold Check** -- If the best fuzzy match score >= 0.5, return it. Otherwise, return "unresolved" with confidence 0.0
7. **Provenance Recording** -- Record the match result, confidence, source, and SHA-256 provenance hash

### Vocabulary Coverage

The normalizer ships with built-in vocabularies:

| Vocabulary | Entries | Categories | Source |
|-----------|---------|------------|--------|
| **Fuels** | ~70 entries | gaseous, liquid, solid, biofuel, electricity | EPA, IEA, IPCC |
| **Materials** | ~60 entries | metals, plastics, construction, paper, chemicals | GreenLang, Industry Standard |
| **Processes** | Extensible | manufacturing, transport, energy generation | Tenant-configurable |

Tenant-specific vocabularies extend the base vocabulary with custom entries, aliases, and overrides.

### What Low Confidence Means

1. **New Data Source with Unfamiliar Naming**: A new ERP connector or data feed uses fuel/material names that do not match the existing vocabulary. For example, a German ERP might send "Erdgas" instead of "Natural Gas".

2. **Vocabulary Gaps**: The built-in vocabulary does not cover a specific fuel, material, or process that a customer uses. For example, specialized industrial chemicals or regional fuel blends.

3. **Ambiguous Names**: Input names that could match multiple vocabulary entries with similar scores. For example, "gas" could match "Natural Gas", "Gasoline", or "LPG".

4. **Threshold Too Strict**: The fuzzy match threshold (default: 0.5) may be rejecting valid matches. This is more likely when input data uses abbreviations or regional terms.

5. **Threshold Too Lenient**: The fuzzy match threshold may be accepting incorrect matches. For example, "coal" matching "coastal" due to substring overlap.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium | Unresolved entities require manual review; users see "unresolved" status for some data |
| **Data Impact** | Medium | Unresolved entities block downstream calculations; emissions factors cannot be applied without entity identification |
| **SLA Impact** | Low-Medium | Processing time increases due to manual review queue; SLA may be violated for tenants with high unresolved rates |
| **Revenue Impact** | Low | Degraded automation; manual review increases operational cost |
| **Compliance Impact** | Medium | Unresolved entities may delay regulatory submissions if they block report generation |
| **Downstream Impact** | Medium | Calculation agents cannot apply correct emission factors without resolved entity identities |

---

## Symptoms

### Low Average Confidence

- `NormalizerEntityResolutionLowConfidence` alert firing
- `glnorm_entity_resolution_confidence` histogram shifting toward lower values
- High proportion of resolution results with confidence between 0.5-0.7 (borderline matches)
- Users reporting "incorrect fuel type" or "wrong material match" in resolved entities

### High Unresolved Rate

- `NormalizerHighUnresolvedRate` alert firing
- `glnorm_entity_resolution_unresolved_total` counter incrementing rapidly
- Downstream calculation agents logging "entity not resolved" for specific data records
- Data quality reports showing increased "entity_unknown" flags

### Review Queue Backlog

- `NormalizerReviewQueueBacklog` alert firing
- `glnorm_entity_review_queue_size` gauge growing steadily
- Manual review team overwhelmed
- Old unresolved items aging out without resolution

---

## Diagnostic Steps

### Step 1: Check Entity Resolution Metrics

```bash
# Port-forward to the normalizer service
kubectl port-forward -n greenlang svc/normalizer-service 8080:8080

# Get current entity resolution metrics
curl -s http://localhost:8080/metrics | python3 -c "
import sys, json
data = json.load(sys.stdin)
total = data.get('entity_resolution_total', 0)
resolved = data.get('entity_resolution_resolved', 0)
unresolved = data.get('entity_resolution_unresolved', 0)
avg_conf = data.get('entity_resolution_avg_confidence', 0)
queue_size = data.get('review_queue_size', 0)
rate = (resolved / total * 100) if total > 0 else 0
print(f'Total resolutions: {total}')
print(f'Resolved: {resolved} ({rate:.1f}%)')
print(f'Unresolved: {unresolved} ({100-rate:.1f}%)')
print(f'Average confidence: {avg_conf:.3f}')
print(f'Review queue size: {queue_size}')
"
```

```promql
# Resolution success rate over time
rate(glnorm_entity_resolution_resolved_total[5m]) /
  rate(glnorm_entity_resolution_total[5m])

# Average confidence trend
avg(glnorm_entity_resolution_confidence)

# Unresolved rate by entity type (fuel, material, process)
sum by (entity_type) (rate(glnorm_entity_resolution_unresolved_total[5m]))

# Confidence distribution
histogram_quantile(0.50, rate(glnorm_entity_resolution_confidence_bucket[5m]))
histogram_quantile(0.25, rate(glnorm_entity_resolution_confidence_bucket[5m]))
histogram_quantile(0.10, rate(glnorm_entity_resolution_confidence_bucket[5m]))

# Review queue growth rate
deriv(glnorm_entity_review_queue_size[30m])
```

### Step 2: Identify the Most Common Unresolved Entities

```bash
# Check logs for unresolved entity names
kubectl logs -n greenlang -l app=normalizer-service --tail=2000 \
  | grep -i "unresolved\|confidence=0\|unknown.*entity\|no.*match" \
  | head -50

# Get the top unresolved entities from the review queue
kubectl run pg-unresolved --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT original_name, entity_type, count(*) as occurrences,
          min(created_at) as first_seen, max(created_at) as last_seen
   FROM normalizer_review_queue
   WHERE status = 'pending'
   GROUP BY original_name, entity_type
   ORDER BY occurrences DESC
   LIMIT 30;"

# Check which tenants or data sources are generating the most unresolved items
kubectl run pg-sources --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT tenant_id, data_source, count(*) as unresolved_count
   FROM normalizer_review_queue
   WHERE status = 'pending'
   GROUP BY tenant_id, data_source
   ORDER BY unresolved_count DESC
   LIMIT 20;"
```

### Step 3: Check Vocabulary Completeness

```bash
# Count vocabulary entries by type
kubectl run pg-vocab-count --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT 'fuel_vocabulary' as vocab, count(*), count(DISTINCT category) as categories
   FROM normalizer_fuel_vocabulary WHERE active = true
   UNION ALL
   SELECT 'material_vocabulary', count(*), count(DISTINCT category)
   FROM normalizer_material_vocabulary WHERE active = true
   UNION ALL
   SELECT 'process_vocabulary', count(*), count(DISTINCT category)
   FROM normalizer_process_vocabulary WHERE active = true
   UNION ALL
   SELECT 'tenant_overrides', count(*), count(DISTINCT tenant_id)
   FROM normalizer_tenant_vocab_overrides WHERE active = true;"

# Check for vocabulary entries without aliases
kubectl run pg-no-alias --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT fv.canonical_name, fv.category, fv.code
   FROM normalizer_fuel_vocabulary fv
   LEFT JOIN normalizer_vocab_aliases va ON fv.canonical_name = va.canonical_name AND va.vocab_type = 'fuel'
   WHERE fv.active = true AND va.id IS NULL
   LIMIT 20;"
```

### Step 4: Check Fuzzy Matching Quality

```bash
# Test specific unresolved names against the vocabulary
# Replace <name> with actual unresolved entity names from Step 2
curl -s -X POST http://localhost:8080/v1/normalizer/resolve \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{
    "entity_type": "fuel",
    "name": "<unresolved_name>",
    "return_candidates": true,
    "max_candidates": 5
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Input: {data.get(\"original_name\")}')
print(f'Match: {data.get(\"standardized_name\", \"NONE\")}')
print(f'Confidence: {data.get(\"confidence\", 0):.3f}')
print(f'Code: {data.get(\"code\", \"N/A\")}')
if data.get('candidates'):
    print(f'Top candidates:')
    for c in data['candidates']:
        print(f'  {c[\"name\"]} (score={c[\"score\"]:.3f}, code={c[\"code\"]})')
"
```

### Step 5: Check Confidence Threshold Configuration

```bash
# Check current threshold configuration
kubectl get configmap normalizer-service-config -n greenlang -o yaml \
  | grep -i "confidence\|threshold\|fuzzy\|match"

# Check environment variables
kubectl get deployment normalizer-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].env}' | python3 -c "
import sys, json
envs = json.loads(sys.stdin.read())
for env in envs:
    name = env.get('name', '')
    if any(k in name.upper() for k in ['CONFIDENCE', 'THRESHOLD', 'FUZZY', 'MATCH', 'RESOLUTION']):
        print(f'{name}={env.get(\"value\", \"<from-secret>\")}')"
```

### Step 6: Check for Recent Data Source Changes

```bash
# Check if new data sources or tenants were recently onboarded
kubectl run pg-new-sources --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT DISTINCT data_source, tenant_id, min(created_at) as first_seen, count(*) as total_items
   FROM normalizer_review_queue
   WHERE created_at > NOW() - INTERVAL '7 days'
   GROUP BY data_source, tenant_id
   ORDER BY first_seen DESC;"

# Check if the unresolved spike correlates with a specific data import
kubectl logs -n greenlang -l app.kubernetes.io/component=intake-agent --tail=500 \
  | grep -i "import\|upload\|ingest\|data source\|erp"
```

---

## Resolution Steps

### Option 1: Add Vocabulary Entries for Common Unresolved Names

The fastest resolution for known vocabulary gaps is to add the missing entries directly.

```bash
# Step 1: Identify the top unresolved names (from Step 2 above)
# Step 2: Add them to the vocabulary via API

# Add a fuel vocabulary entry
curl -X POST http://localhost:8080/v1/normalizer/admin/vocabulary/fuel \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "canonical_name": "Natural Gas",
    "code": "NG",
    "category": "gaseous",
    "aliases": ["Erdgas", "gaz naturel", "gas natural", "natuerliches gas"],
    "source": "IEA Energy Statistics Manual",
    "effective_date": "2026-01-01"
  }' | python3 -m json.tool

# Add a material vocabulary entry
curl -X POST http://localhost:8080/v1/normalizer/admin/vocabulary/material \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "canonical_name": "Aluminum",
    "code": "ALU",
    "category": "metals",
    "aliases": ["aluminium", "alu", "aluminum alloy", "al"],
    "source": "GreenLang Standard",
    "effective_date": "2026-01-01"
  }' | python3 -m json.tool

# Clear the vocabulary cache to pick up new entries
curl -X POST http://localhost:8080/v1/normalizer/cache/clear \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### Option 2: Add Tenant-Specific Vocabulary Overrides

If unresolved names are specific to a single tenant's data format:

```bash
# Add tenant-specific aliases
curl -X POST http://localhost:8080/v1/normalizer/admin/vocabulary/tenant-override \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "tenant_id": "tenant_123",
    "entity_type": "fuel",
    "alias": "NG-Pipeline",
    "canonical_name": "Natural Gas",
    "code": "NG",
    "category": "gaseous",
    "source": "Tenant ERP mapping"
  }' | python3 -m json.tool
```

### Option 3: Adjust Confidence Thresholds

If the thresholds are too strict and rejecting valid matches:

```bash
# Lower the minimum confidence threshold (from 0.5 to 0.4)
kubectl set env deployment/normalizer-service -n greenlang \
  GL_NORM_ENTITY_MIN_CONFIDENCE=0.4

# Optionally lower the auto-accept threshold (matches above this are auto-accepted)
kubectl set env deployment/normalizer-service -n greenlang \
  GL_NORM_ENTITY_AUTO_ACCEPT_CONFIDENCE=0.85

# Restart to apply
kubectl rollout restart deployment/normalizer-service -n greenlang
```

**Caution:** Lowering the threshold increases the risk of incorrect matches. Monitor the precision/recall balance after the change. Consider lowering only for specific entity types or tenants.

If the thresholds are too lenient and accepting incorrect matches:

```bash
# Raise the minimum confidence threshold (from 0.5 to 0.6)
kubectl set env deployment/normalizer-service -n greenlang \
  GL_NORM_ENTITY_MIN_CONFIDENCE=0.6

# Raise the auto-accept threshold
kubectl set env deployment/normalizer-service -n greenlang \
  GL_NORM_ENTITY_AUTO_ACCEPT_CONFIDENCE=0.95

kubectl rollout restart deployment/normalizer-service -n greenlang
```

### Option 4: Batch Review Unresolved Items

Process the review queue to clear the backlog:

```bash
# Export the review queue for manual review
kubectl run pg-export --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "COPY (
    SELECT original_name, entity_type, tenant_id, data_source,
           count(*) as occurrences, min(created_at) as first_seen
    FROM normalizer_review_queue
    WHERE status = 'pending'
    GROUP BY original_name, entity_type, tenant_id, data_source
    ORDER BY occurrences DESC
  ) TO STDOUT WITH CSV HEADER" > /tmp/review_queue_export.csv

# After manual review, batch-resolve items via API
curl -X POST http://localhost:8080/v1/normalizer/admin/review-queue/batch-resolve \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "resolutions": [
      {
        "original_name": "Erdgas",
        "entity_type": "fuel",
        "resolved_canonical": "Natural Gas",
        "resolved_code": "NG",
        "add_as_alias": true
      },
      {
        "original_name": "Stahl",
        "entity_type": "material",
        "resolved_canonical": "Steel",
        "resolved_code": "STL",
        "add_as_alias": true
      }
    ]
  }' | python3 -m json.tool
```

### Option 5: Improve Fuzzy Matching (Code Change)

If the built-in substring/word-overlap matching is insufficient, enable enhanced matching features:

```bash
# Enable Levenshtein distance matching (edit distance)
kubectl set env deployment/normalizer-service -n greenlang \
  GL_NORM_ENTITY_ENABLE_LEVENSHTEIN=true \
  GL_NORM_ENTITY_LEVENSHTEIN_MAX_DISTANCE=3

# Enable phonetic matching (Soundex/Metaphone) for names with spelling variations
kubectl set env deployment/normalizer-service -n greenlang \
  GL_NORM_ENTITY_ENABLE_PHONETIC=true

kubectl rollout restart deployment/normalizer-service -n greenlang
```

---

## Vocabulary Curation Process

The standard process for adding or updating vocabulary entries follows a propose-review-approve-deploy workflow:

### Step 1: Propose

- Data team or tenant identifies missing or incorrect vocabulary entries
- Submit a vocabulary change request via the admin API or pull request to the vocabulary repository
- Include: canonical name, code, category, aliases, authoritative source citation

### Step 2: Review

- Domain expert reviews the proposed entry for accuracy
- Verify the canonical name matches the authoritative source
- Verify aliases are unambiguous (no conflicts with existing entries)
- Verify the category assignment is correct
- Check that the source citation is valid

### Step 3: Approve

- L2 approver (data quality lead or domain expert) approves the change
- Approval is recorded in the vocabulary audit log
- Entry is marked as "approved" in the staging environment

### Step 4: Deploy

- Approved vocabulary changes are deployed to production via the CI/CD pipeline
- The deployment includes a database migration that inserts the new vocabulary entries
- The normalizer service vocabulary cache is automatically refreshed
- Previously unresolved items matching the new entries are re-resolved

### Step 5: Verify

- Monitor entity resolution metrics after deployment
- Confirm that previously unresolved items are now resolving correctly
- Verify no regressions (existing resolutions still correct)

---

## Post-Resolution Verification

```promql
# 1. Average confidence should be above threshold
avg(glnorm_entity_resolution_confidence) > 0.7

# 2. Unresolved rate should be below 20%
rate(glnorm_entity_resolution_unresolved_total[5m]) /
  rate(glnorm_entity_resolution_total[5m]) < 0.2

# 3. Review queue should be shrinking
deriv(glnorm_entity_review_queue_size[30m]) <= 0

# 4. Resolution rate by entity type should be improving
sum by (entity_type) (rate(glnorm_entity_resolution_resolved_total[5m])) /
  sum by (entity_type) (rate(glnorm_entity_resolution_total[5m]))
```

```bash
# 5. Test previously unresolved names
for name in "Erdgas" "Stahl" "gaz naturel"; do
  result=$(curl -s -X POST http://localhost:8080/v1/normalizer/resolve \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -d "{\"entity_type\": \"fuel\", \"name\": \"$name\"}")
  resolved=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d.get(\"standardized_name\",\"UNRESOLVED\")} (conf={d.get(\"confidence\",0):.2f})')")
  echo "$name -> $resolved"
done
```

---

## Prevention

### Vocabulary Monitoring

- **Dashboard:** Normalizer Service Health (`/d/normalizer-service-health`) -- entity resolution panels
- **Dashboard:** Unit Conversion Overview (`/d/normalizer-conversion-overview`)
- **Alert:** `NormalizerEntityResolutionLowConfidence` (this alert)
- **Alert:** `NormalizerHighUnresolvedRate` (this alert)
- **Alert:** `NormalizerReviewQueueBacklog` (this alert)
- **Key metrics to watch:**
  - `glnorm_entity_resolution_confidence` histogram (median should be >0.8)
  - `glnorm_entity_resolution_unresolved_total` rate (should be <20% of total)
  - `glnorm_entity_review_queue_size` gauge (should be <100 in steady state)
  - `glnorm_vocabulary_size` by entity type (should be stable or growing)

### Vocabulary Completeness Best Practices

1. **Proactively add aliases** for international variations (German, French, Spanish, Chinese names for common fuels and materials)
2. **Add aliases for ERP-specific naming** when onboarding new tenants (SAP material IDs, Oracle product codes)
3. **Review the unresolved queue weekly** to identify systematic gaps
4. **Automate alias suggestions** from the review queue -- items that appear more than 5 times likely represent a vocabulary gap
5. **Maintain vocabulary test fixtures** -- a set of known input-output pairs that are validated on every deployment

### Onboarding Process for New Data Sources

1. Before connecting a new data source, obtain a sample data file
2. Run the sample through the entity resolver in dry-run mode
3. Identify all unresolved entities and add vocabulary entries before going live
4. Set up tenant-specific overrides for non-standard naming conventions
5. Monitor the resolution rate for the new data source for the first week

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Team + Data Quality Team
- **Review cadence:** Quarterly or after any entity resolution incident
- **Related alerts:** `NormalizerServiceDown`, `NormalizerConversionAccuracyDrift`
- **Related dashboards:** Normalizer Service Health, Unit Conversion Overview
- **Related runbooks:** [Normalizer Service Down](./normalizer-service-down.md), [Conversion Accuracy Drift](./conversion-accuracy-drift.md), [Normalizer High Latency](./normalizer-high-latency.md)
