# Schema IR Cache Corruption

## Alert

**Alert Names:**

| Alert | Severity | Condition |
|-------|----------|-----------|
| `SchemaCacheHitRateLow` | Warning | Cache hit rate <50% for 10 minutes |
| `SchemaCacheCompilationAfterHit` | Critical | Compilation errors after cache hit (hash mismatch) |
| `SchemaIRHashMismatch` | Critical | Cached IR hash does not match schema source hash |

**Thresholds:**

```promql
# SchemaCacheHitRateLow
glschema_cache_hits / (glschema_cache_hits + glschema_cache_misses) < 0.5
# sustained for 10 minutes

# SchemaCacheCompilationAfterHit (proxy: cache hits increasing but validation errors also increasing)
increase(glschema_cache_hits[5m]) > 0 AND increase(glschema_validations_failed[5m]) > increase(glschema_validations_failed[5m] offset 10m) * 2

# SchemaIRHashMismatch (detected in application logs)
# Triggered by log pattern: "IR hash mismatch" or "cache integrity error"
```

---

## Description

These alerts fire when the schema IR cache exhibits integrity issues. The IR cache is the performance-critical layer that stores compiled schema Intermediate Representations to avoid recompiling schemas on every validation request. Cache corruption or degradation impacts validation correctness and performance.

### Cache Architecture Overview

```
+------------------+       +------------------+       +------------------+
|  Git Schema      | ----> |  Schema Compiler | ----> |  SchemaIR        |
|  Registry        |       |  (10-step        |       |  (Compiled IR)   |
|  (source of      |       |   pipeline)      |       |                  |
|   truth)         |       |                  |       |  - Properties    |
|                  |       |  Parse           |       |  - Constraints   |
|  schemas/        |       |  Hash            |       |  - Patterns      |
|    emissions/    |       |  Flatten         |       |  - Unit specs    |
|    energy/       |       |  Constraints     |       |  - Rule bindings |
|    transport/    |       |  Patterns        |       |  - Deprecations  |
+------------------+       |  Units           |       |  - Enums         |
                           |  Rules           |       |  - Schema hash   |
                           |  Deprecations    |       +--------+---------+
                           |  Enums           |                |
                           +--------+---------+                v
                                    |                +------------------+
                                    |                | IRCacheService   |
                                    +--------------->| (LRU + TTL)      |
                                                     |                  |
                                                     | Key: schema_id + |
                                                     |   version +      |
                                                     |   compiler_ver   |
                                                     |                  |
                                                     | Eviction: LRU    |
                                                     | TTL: 1 hour      |
                                                     | Max: 1000 entries|
                                                     | Max mem: 256 MB  |
                                                     +--------+---------+
                                                              |
                                                              v
                                                     +------------------+
                                                     | Validator        |
                                                     | (uses cached IR) |
                                                     +------------------+
```

The cache key is composed of `(schema_id, version, compiler_version)`. This ensures that when the compiler is upgraded, cached IRs compiled by the old compiler are automatically invalidated.

### What Cache Corruption Means

1. **Stale IR**: The cached IR was compiled from an older version of the schema source. The schema was updated in the registry, but the cache was not invalidated. Validation runs against outdated rules.

2. **Hash mismatch**: The schema hash stored in the cached IR does not match the hash of the current schema source. This indicates the schema content changed without cache invalidation.

3. **Corrupted IR structure**: The cached IR has invalid or missing fields due to a compiler bug, memory corruption, or serialization error. Validation may crash or produce incorrect results.

4. **Eviction-induced cold cache**: Redis eviction or memory pressure has cleared most cache entries, causing all requests to trigger full compilation. This is a performance issue, not a correctness issue.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium-High | Stale IRs may cause false positives (rejecting valid data) or false negatives (accepting invalid data) |
| **Data Impact** | High | If corrupted IRs allow invalid data through, downstream calculations may produce incorrect results |
| **SLA Impact** | Medium | Cache misses cause higher latency (full compilation instead of cached lookup); P95 latency SLA may be violated |
| **Revenue Impact** | Low-Medium | Degraded validation performance; potential data quality issues |
| **Compliance Impact** | High | False negatives (accepting invalid data) create compliance risk; provenance chain may reference incorrect schema hashes |
| **Performance Impact** | High | Cold cache: every validation request triggers full schema compilation (~10-100x slower than cached) |

---

## Symptoms

### Cache Hit Rate Drop

- `SchemaCacheHitRateLow` alert firing
- `glschema_cache_hits / (glschema_cache_hits + glschema_cache_misses)` ratio dropping significantly
- `glschema_validation_time_avg_ms` increasing (compilation on every request)
- `glschema_validation_time_p95_ms` exceeding SLA targets

### Compilation Errors After Cache Hit

- Validation returning unexpected errors for previously valid payloads
- Logs showing "IR hash mismatch" or "schema hash verification failed"
- Validation results inconsistent between requests (cache hit vs. miss produce different results)

### Warm-up Scheduler Issues

- Cache size remains 0 or very low after service start
- Logs showing "warm-up failed" or "compile function not configured"
- `CacheWarmupScheduler` thread not running (check logs for "Warm-up scheduler started")

---

## Diagnostic Steps

### Step 1: Check Cache Metrics

```bash
# Port-forward to the schema service
kubectl port-forward -n greenlang svc/schema-service 8080:8080

# Get current cache metrics
curl -s http://localhost:8080/metrics | python3 -c "
import sys, json
data = json.load(sys.stdin)
hits = data.get('cache_hits', 0)
misses = data.get('cache_misses', 0)
total = hits + misses
hit_rate = (hits / total * 100) if total > 0 else 0
print(f'Cache size: {data.get(\"cache_size\", 0)} entries')
print(f'Cache hits: {hits}')
print(f'Cache misses: {misses}')
print(f'Hit rate: {hit_rate:.1f}%')
print(f'Avg validation time: {data.get(\"avg_validation_time_ms\", 0):.1f}ms')
print(f'P95 validation time: {data.get(\"p95_validation_time_ms\", 0):.1f}ms')
"
```

```promql
# Cache hit rate trend
glschema_cache_hits / (glschema_cache_hits + glschema_cache_misses)

# Cache size trend (drops indicate restarts or evictions)
glschema_cache_size

# Validation latency trend (spikes indicate cache misses)
glschema_validation_time_p95_ms

# Correlation: cache size vs. validation latency
glschema_cache_size
glschema_validation_time_avg_ms
```

### Step 2: Verify IR Hash Integrity

```bash
# Check schema service logs for hash mismatch errors
kubectl logs -n greenlang -l app=schema-service --tail=500 \
  | grep -i "hash mismatch\|integrity\|corrupted\|cache.*error\|stale"

# Check for schema compilation being triggered unexpectedly
kubectl logs -n greenlang -l app=schema-service --tail=500 \
  | grep -i "compil\|cache miss\|cache expired\|cache evict"
```

### Step 3: Check Redis Connectivity and State (if applicable)

```bash
# Verify Redis connectivity
kubectl run redis-test --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 PING

# Check Redis memory usage and eviction stats
kubectl run redis-info --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 INFO memory

# Check Redis eviction statistics
kubectl run redis-stats --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 INFO stats \
  | grep -i "evicted\|expired\|keyspace"

# Check schema IR keys in Redis
kubectl run redis-keys --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 KEYS 'gl:schema:ir:*' | wc -l
```

### Step 4: Check Cache Configuration

```bash
# Verify cache configuration
kubectl get configmap schema-service-config -n greenlang -o yaml \
  | grep -i "cache\|ttl\|max_size\|memory\|evict"

# Check environment variables for cache settings
kubectl get deployment schema-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].env[*]}' | python3 -c "
import sys, json
envs = json.loads(f'[{sys.stdin.read()}]') if '{' in sys.stdin.read() else []
for env in envs:
    name = env.get('name', '')
    if 'CACHE' in name or 'TTL' in name or 'MEMORY' in name:
        print(f'{name}={env.get(\"value\", \"<from-secret>\")}')"
```

### Step 5: Check Warm-up Scheduler Status

```bash
# Check if the warm-up scheduler is running
kubectl logs -n greenlang -l app=schema-service --tail=200 \
  | grep -i "warm-up\|warmup\|scheduler"

# Look for warm-up cycle completion messages
kubectl logs -n greenlang -l app=schema-service --tail=500 \
  | grep -i "warm-up cycle\|schemas cached"
```

### Step 6: Check for Recent Schema Updates Without Cache Invalidation

```bash
# Check recent schema registry changes
kubectl exec -n greenlang <schema-service-pod> -- \
  git -C /data/schema-registry log --oneline -20 --since="2 hours ago"

# Compare schema file modification times with cache entry ages
# If schemas were updated more recently than the oldest cache entry,
# the cache may contain stale IRs
```

---

## Resolution Steps

### Step 1: Flush the Entire IR Cache

The safest first step is to clear the entire cache. This forces all schemas to be recompiled from the current registry source.

```bash
# Option A: Clear via API (if endpoint exists)
curl -X POST http://localhost:8080/v1/schema/cache/clear \
  -H "X-API-Key: $SCHEMA_API_KEY" \
  | python3 -m json.tool

# Option B: Clear via restart (cache is in-memory)
kubectl rollout restart deployment/schema-service -n greenlang
kubectl rollout status deployment/schema-service -n greenlang
```

**Expected behavior after flush:**
- Cache size drops to 0
- Cache miss rate temporarily reaches 100%
- Validation latency temporarily increases (all requests require compilation)
- Cache gradually repopulates as validation requests arrive
- Warm-up scheduler repopulates popular schemas within one interval (default: 5 minutes)

### Step 2: Trigger Manual Cache Warm-up

After flushing the cache, trigger a manual warm-up to repopulate popular schemas immediately.

```bash
# Option A: Trigger warm-up via API (if endpoint exists)
curl -X POST http://localhost:8080/v1/schema/cache/warmup \
  -H "X-API-Key: $SCHEMA_API_KEY" \
  | python3 -m json.tool

# Option B: Pre-warm by validating a test payload for each popular schema
for schema in emissions/activity energy/consumption transport/shipment; do
  curl -X POST http://localhost:8080/v1/schema/validate \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $SCHEMA_API_KEY" \
    -d "{
      \"schema_ref\": {\"schema_id\": \"$schema\", \"version\": \"latest\"},
      \"payload\": {}
    }" > /dev/null 2>&1
  echo "Warmed up: $schema"
done
```

### Step 3: Verify Schema Registry Integrity

If cache corruption was caused by a corrupted schema in the registry, verify the registry.

```bash
# Check Git repository integrity
kubectl exec -n greenlang <schema-service-pod> -- \
  git -C /data/schema-registry fsck

# Verify schema files are valid YAML/JSON
kubectl exec -n greenlang <schema-service-pod> -- \
  find /data/schema-registry/schemas -name "*.yaml" -exec python3 -c "
import yaml, sys
try:
    yaml.safe_load(open(sys.argv[1]))
    print(f'OK: {sys.argv[1]}')
except Exception as e:
    print(f'INVALID: {sys.argv[1]}: {e}')
" {} \;
```

### Step 4: Fix Redis Eviction Issues (if applicable)

If the cache hit rate drop was caused by Redis memory pressure and eviction:

```bash
# Check Redis max memory configuration
kubectl run redis-config --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 CONFIG GET maxmemory

# Check eviction policy
kubectl run redis-policy --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 CONFIG GET maxmemory-policy

# If eviction is too aggressive, increase Redis memory or change policy
# Note: Schema IR cache should use volatile-lru policy to prefer evicting
# keys with TTL set
```

### Step 5: Adjust Cache Configuration

If the cache is too small for the number of active schemas, increase the cache size.

```bash
# Increase cache size and memory limit
kubectl set env deployment/schema-service -n greenlang \
  GL_SCHEMA_CACHE_MAX_SIZE=2000 \
  GL_SCHEMA_IR_CACHE_MAX_MEMORY_MB=512 \
  GL_SCHEMA_CACHE_TTL_SECONDS=7200

# Restart to apply
kubectl rollout restart deployment/schema-service -n greenlang
```

### Step 6: Fix Version Mismatch (Compiler Upgrade)

If the cache corruption was caused by a compiler version change without cache invalidation:

```bash
# The cache key includes compiler_version, so a compiler upgrade should
# automatically invalidate old entries. If it does not, force a cache clear:
kubectl rollout restart deployment/schema-service -n greenlang

# Verify the new compiler version is in use
kubectl logs -n greenlang -l app=schema-service --tail=50 \
  | grep -i "compiler.*version\|SchemaCompiler initialized"
```

---

## Post-Resolution Verification

```bash
# 1. Verify cache is being populated
curl -s http://localhost:8080/metrics | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Cache size: {data.get(\"cache_size\", 0)}')
print(f'Cache hits: {data.get(\"cache_hits\", 0)}')
print(f'Cache misses: {data.get(\"cache_misses\", 0)}')
"
```

```promql
# 2. Verify cache hit rate is recovering
glschema_cache_hits / (glschema_cache_hits + glschema_cache_misses) > 0.5

# 3. Verify validation latency is returning to normal
glschema_validation_time_p95_ms < 500

# 4. Verify cache size is stabilizing
glschema_cache_size
```

```bash
# 5. Test a validation request and verify it uses the cache on second call
# First call (cache miss - compiles schema)
time curl -s -X POST http://localhost:8080/v1/schema/validate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SCHEMA_API_KEY" \
  -d '{"schema_ref": {"schema_id": "emissions/activity", "version": "1.3.0"}, "payload": {}}' \
  > /dev/null

# Second call (cache hit - should be faster)
time curl -s -X POST http://localhost:8080/v1/schema/validate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SCHEMA_API_KEY" \
  -d '{"schema_ref": {"schema_id": "emissions/activity", "version": "1.3.0"}, "payload": {}}' \
  > /dev/null
```

---

## Prevention

### Cache Monitoring

- **Dashboard:** Schema Service Health (`/d/schema-service-health`) -- cache panels
- **Alert:** `SchemaCacheHitRateLow` (this alert)
- **Alert:** `SchemaCacheCompilationAfterHit` (this alert)
- **Key metrics to watch:**
  - `glschema_cache_size` (should match number of active schemas)
  - `glschema_cache_hits / (glschema_cache_hits + glschema_cache_misses)` (target >80% in steady state)
  - `glschema_validation_time_p95_ms` (spikes indicate cache misses)
  - Redis memory usage and eviction count (if using Redis backend)

### Cache Sizing Guidelines

| Environment | Max Size | TTL | Max Memory | Rationale |
|-------------|----------|-----|------------|-----------|
| Development | 100 | 300s (5m) | 64 MB | Fast iteration, frequent schema changes |
| Staging | 500 | 1800s (30m) | 128 MB | Mirror production proportionally |
| Production | 1000 | 3600s (1h) | 256 MB | All active schemas cached, good hit rate |
| Production (large) | 5000 | 7200s (2h) | 512 MB | Large schema registry with many versions |

### Cache Invalidation Best Practices

1. **Always invalidate cache after schema registry updates.** If the schema registry is updated (Git push), invalidate cached IRs for the affected schemas.
2. **Include compiler version in cache key.** The IRCacheService already does this by default. Do not override this behavior.
3. **Set appropriate TTL.** The TTL provides a safety net: even if explicit invalidation is missed, stale entries expire naturally. Default 1 hour is recommended for production.
4. **Monitor warm-up scheduler.** Ensure the `CacheWarmupScheduler` is running and successfully pre-compiling popular schemas.

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Team
- **Review cadence:** Quarterly or after any cache-related incident
- **Related alerts:** `SchemaServiceDown`, `SchemaHighLatency`
- **Related dashboards:** Schema Service Health
- **Related runbooks:** [Schema Service Down](./schema-service-down.md), [High Validation Errors](./high-validation-errors.md), [Compilation Timeout](./compilation-timeout.md)
