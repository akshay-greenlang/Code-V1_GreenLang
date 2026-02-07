# Schema Compilation Timeout / Failure

## Alert

**Alert Names:**

| Alert | Severity | Condition |
|-------|----------|-----------|
| `SchemaCompilationFailure` | Warning | Compilation errors > 0 for 5 minutes |
| `SchemaCompilationTimeout` | Warning | Compilation time exceeds 5 seconds for any schema |
| `SchemaReDoSDetected` | Critical | ReDoS-vulnerable regex pattern detected in schema |

**Thresholds:**

```promql
# SchemaCompilationFailure
increase(glschema_compilation_errors_total[5m]) > 0

# SchemaCompilationTimeout (proxy: high compilation latency)
histogram_quantile(0.99, rate(glschema_compilation_duration_seconds_bucket[5m])) > 5

# SchemaReDoSDetected (detected in application logs)
# Triggered by log pattern: "unsafe pattern" or "ReDoS" or "complexity score exceeds"
```

---

## Description

These alerts fire when the GreenLang Schema Compiler (AGENT-FOUND-002) fails to compile a schema to Intermediate Representation (IR) or takes an abnormally long time to do so. Schema compilation is the process of transforming a JSON Schema (Draft 2020-12) with GreenLang extensions into an optimized IR that the validator uses for fast payload validation.

### The 10-Step Compilation Pipeline

The `SchemaCompiler` transforms schema sources through a deterministic pipeline:

```
Schema Source (YAML/JSON)
        |
        v
  +-- Step 1: Parse ----------------------- Parse YAML/JSON into dict
  |   Step 2: Hash  ----------------------- Compute SHA-256 of canonical JSON
  |   Step 3: Flatten Properties ---------- Recursive traversal; build O(1) path-indexed map
  |   Step 4: Collect Required Paths ------ Recursive traversal; collect required field paths
  |   Step 5: Compile Constraints --------- Extract numeric/string/array constraints
  |   Step 6: Compile Patterns ------------ Regex compilation + ReDoS safety analysis  <-- BOTTLENECK
  |   Step 7: Extract Unit Specs ---------- Process $unit/$dimension extensions
  |   Step 8: Extract Rule Bindings ------- Process $rules extensions
  |   Step 9: Extract Deprecations -------- Process $deprecated/$renamed_from extensions
  |   Step 10: Extract Enums -------------- Collect enum constraints
  |
  v
SchemaIR (Intermediate Representation)
  - properties: Dict[path, PropertyIR]
  - required_paths: Set[str]
  - constraints: numeric + string + array
  - patterns: Dict[path, CompiledPattern]
  - unit_specs, rule_bindings, deprecations, enums
  - schema_hash (SHA-256 provenance)
```

### Why Compilations Fail or Time Out

1. **Pathological regex patterns (ReDoS)**: Schema `pattern` or `patternProperties` fields contain regex patterns with nested quantifiers, overlapping alternations, or exponential backtracking risk. The `RegexAnalyzer` detects these and rejects the schema, but the analysis itself takes time for complex patterns.

2. **Deeply nested `$ref` chains**: Schemas with excessive `$ref` references that expand recursively. The compiler enforces a limit of 10,000 `$ref` expansions (`MAX_REF_EXPANSIONS`), but approaching this limit causes slow compilation.

3. **Circular `$ref` references**: A schema that references itself directly or indirectly (A -> B -> C -> A). The `RefResolver` should detect cycles, but malformed schemas may cause infinite recursion before detection.

4. **Extremely large schemas**: Schemas with thousands of properties, deeply nested objects (limit: 50 levels), or massive enum sets. The property flattening step (Step 3) and constraint compilation step (Step 5) scale linearly with schema size.

5. **Invalid schema syntax**: Parse errors in YAML/JSON, invalid JSON Schema keywords, or malformed GreenLang extensions ($unit, $rules, etc.).

6. **`$ref` resolution failures**: External schema references that cannot be resolved because the referenced schema does not exist in the registry or the version is not found.

### ReDoS (Regular Expression Denial of Service)

The schema compiler includes a dedicated `RegexAnalyzer` that analyzes every regex pattern in a schema before compilation. The analyzer uses:

- **Known dangerous pattern matching**: Checks against a set of known ReDoS patterns (e.g., `(a+)+`, `(.*)*`, `(.+)+`)
- **AST-based analysis**: Parses regex into AST using Python's `sre_parse` module to detect nested quantifiers at any depth
- **Overlapping alternation detection**: Identifies alternation branches that can match the same input prefix
- **Complexity scoring**: Computes a score from 0.0 (safe) to 1.0 (dangerous) based on pattern features (quantifier count, nesting depth, alternation count, backreferences, lookarounds)
- **RE2 compatibility checking**: Flags patterns that use features incompatible with the RE2 linear-time engine (backreferences, lookahead/lookbehind, atomic groups, possessive quantifiers)

A pattern is rejected if:
- It matches a known dangerous pattern
- It contains nested quantifiers detected via AST traversal
- It contains overlapping alternations inside a quantifier
- Its complexity score exceeds `MAX_REGEX_COMPLEXITY_SCORE` (default: 0.8)
- Its length exceeds `MAX_REGEX_LENGTH` (default: 1000 characters)

### Compilation Limits

| Limit | Default | Environment Variable | Purpose |
|-------|---------|---------------------|---------|
| Max payload size | 1 MB | `GL_SCHEMA_MAX_PAYLOAD_BYTES` | Prevent memory exhaustion |
| Max schema size | 2 MB | `GL_SCHEMA_MAX_SCHEMA_BYTES` | Prevent large schema compilation |
| Max object depth | 50 levels | `GL_SCHEMA_MAX_OBJECT_DEPTH` | Prevent deep recursion |
| Max array items | 10,000 | `GL_SCHEMA_MAX_ARRAY_ITEMS` | Prevent large array schemas |
| Max total nodes | 200,000 | `GL_SCHEMA_MAX_TOTAL_NODES` | Prevent combinatorial explosion |
| Max `$ref` expansions | 10,000 | `GL_SCHEMA_MAX_REF_EXPANSIONS` | Prevent circular ref explosion |
| Max regex length | 1,000 chars | `GL_SCHEMA_MAX_REGEX_LENGTH` | Prevent complex patterns |
| Regex timeout | 100 ms | `GL_SCHEMA_REGEX_TIMEOUT_MS` | Limit regex analysis time |
| Max regex complexity | 0.8 | `GL_SCHEMA_MAX_REGEX_COMPLEXITY_SCORE` | Reject dangerous patterns |
| Max object properties | 10,000 | `GL_SCHEMA_MAX_OBJECT_PROPERTIES` | Prevent wide schemas |

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium | Schema cannot be used for validation; submissions referencing the schema will fail |
| **Data Impact** | Low | No data is corrupted; validation simply cannot proceed for the affected schema |
| **SLA Impact** | Medium | If compilation happens at request time (cache miss), the validation request will timeout or fail |
| **Revenue Impact** | Low-Medium | Degraded experience for users whose data references the failing schema |
| **Compliance Impact** | Medium | If the failing schema is required for a regulatory pipeline, submissions are blocked |
| **Performance Impact** | High | A single pathological schema can consume CPU and block the async event loop if compilation is not properly isolated |

---

## Symptoms

- `SchemaCompilationFailure` alert firing
- Logs showing "Schema compilation failed", "Parse error", "Unexpected compilation error"
- Logs showing "unsafe pattern", "Nested quantifier detected", "Pattern complexity score exceeds"
- Logs showing "Pattern at <path> has high complexity score"
- Logs showing "Pattern at <path> exceeds maximum length"
- `glschema_validations_failed` increasing for a specific schema
- POST `/v1/schema/compile` returning error responses
- POST `/v1/schema/validate` returning 500 errors when the referenced schema has not been compiled and cached
- CPU usage spike on schema service pods during compilation attempts
- Cache hit rate remaining at 0% for a specific schema (compilation keeps failing, so IR is never cached)
- Orchestrator DAG nodes that depend on schema validation are stuck or failing

---

## Diagnostic Steps

### Step 1: Identify the Failing Schema

```bash
# Port-forward to the schema service
kubectl port-forward -n greenlang svc/schema-service 8080:8080

# Check logs for compilation errors
kubectl logs -n greenlang -l app=schema-service --tail=500 \
  | grep -i "compilation.*failed\|compile.*error\|parse error\|unexpected.*error"

# Identify which schema is failing
kubectl logs -n greenlang -l app=schema-service --tail=500 \
  | grep -i "Compiling schema\|compilation" \
  | grep -i "fail\|error"

# Check for specific GLSCHEMA error codes
kubectl logs -n greenlang -l app=schema-service --tail=500 \
  | grep -i "GLSCHEMA-E5"
```

```promql
# Compilation error rate by schema
# (if error metrics are labeled by schema_id)
topk(10, sum(rate(glschema_compilation_errors_total[5m])) by (schema_id))

# Overall compilation duration trend
histogram_quantile(0.99, rate(glschema_compilation_duration_seconds_bucket[5m]))
```

### Step 2: Check Schema Complexity

```bash
# Get the failing schema content
curl -s http://localhost:8080/v1/schema/<schema_id>/<version> \
  -H "X-API-Key: $SCHEMA_API_KEY" | python3 -c "
import sys, json

schema = json.load(sys.stdin)

# Count properties recursively
def count_properties(s, depth=0):
    count = 0
    max_depth = depth
    for key in s.get('properties', {}):
        count += 1
        sub = s['properties'][key]
        if isinstance(sub, dict):
            sub_count, sub_depth = count_properties(sub, depth + 1)
            count += sub_count
            max_depth = max(max_depth, sub_depth)
    return count, max_depth

prop_count, max_depth = count_properties(schema)

# Count patterns
patterns = []
def find_patterns(s, path=''):
    if 'pattern' in s:
        patterns.append((path, s['pattern']))
    if 'patternProperties' in s:
        for p in s['patternProperties']:
            patterns.append((f'{path}/patternProperties', p))
    for key in s.get('properties', {}):
        if isinstance(s['properties'][key], dict):
            find_patterns(s['properties'][key], f'{path}/{key}')

find_patterns(schema)

# Count refs
ref_count = str(schema).count('\$ref')

print(f'Total properties: {prop_count}')
print(f'Max nesting depth: {max_depth}')
print(f'Pattern count: {len(patterns)}')
print(f'\$ref count: {ref_count}')
print(f'Schema size: {len(json.dumps(schema))} bytes')
print()
print('Patterns found:')
for path, pat in patterns:
    print(f'  {path}: {pat[:80]}...' if len(pat) > 80 else f'  {path}: {pat}')
"
```

### Step 3: Check Regex Patterns for ReDoS Vulnerability

```bash
# Check logs for ReDoS-related warnings and errors
kubectl logs -n greenlang -l app=schema-service --tail=500 \
  | grep -i "unsafe\|redos\|nested quantifier\|overlapping\|complexity score\|pattern.*length"

# Extract regex patterns from the schema and analyze them
curl -s http://localhost:8080/v1/schema/<schema_id>/<version> \
  -H "X-API-Key: $SCHEMA_API_KEY" | python3 -c "
import sys, json, re

schema = json.load(sys.stdin)

def find_patterns(s, path=''):
    results = []
    if 'pattern' in s:
        results.append((path, s['pattern']))
    if 'patternProperties' in s:
        for p in s['patternProperties']:
            results.append((f'{path}/patternProperties', p))
    for key in s.get('properties', {}):
        if isinstance(s['properties'][key], dict):
            results.extend(find_patterns(s['properties'][key], f'{path}/{key}'))
    if 'items' in s and isinstance(s.get('items'), dict):
        results.extend(find_patterns(s['items'], f'{path}/items'))
    return results

patterns = find_patterns(schema)

# Quick ReDoS indicators
DANGEROUS = [
    r'\([^)]*[+*][^)]*\)[+*]',  # Nested quantifiers
    r'\.\*\.\*\.\*',             # Multiple wildcards
]

print(f'Found {len(patterns)} regex patterns:')
for path, pat in patterns:
    flags = []
    if len(pat) > 1000:
        flags.append('TOO_LONG')
    for d in DANGEROUS:
        if re.search(d, pat):
            flags.append('NESTED_QUANTIFIER')
            break

    # Count quantifiers
    q_count = pat.count('+') + pat.count('*') + len(re.findall(r'\{[\d,]+\}', pat))
    if q_count > 5:
        flags.append('MANY_QUANTIFIERS')

    # Check nesting depth
    max_depth = 0
    depth = 0
    for c in pat:
        if c == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif c == ')':
            depth = max(0, depth - 1)
    if max_depth > 3:
        flags.append(f'DEEP_NESTING({max_depth})')

    status = ' [' + ', '.join(flags) + ']' if flags else ' [OK]'
    print(f'  {path}: {pat[:60]}...{status}' if len(pat) > 60 else f'  {path}: {pat}{status}')
"
```

### Step 4: Check `$ref` Depth and Circular References

```bash
# Check for deep or circular $ref chains
curl -s http://localhost:8080/v1/schema/<schema_id>/<version> \
  -H "X-API-Key: $SCHEMA_API_KEY" | python3 -c "
import sys, json

schema = json.load(sys.stdin)

# Find all \$ref values
refs = []
def find_refs(s, path='', depth=0):
    if isinstance(s, dict):
        if '\$ref' in s:
            refs.append((path, s['\$ref'], depth))
        for key, val in s.items():
            find_refs(val, f'{path}/{key}', depth + 1)
    elif isinstance(s, list):
        for i, val in enumerate(s):
            find_refs(val, f'{path}[{i}]', depth + 1)

find_refs(schema)

print(f'Total \$ref count: {len(refs)}')
print(f'Max \$ref depth: {max(d for _, _, d in refs) if refs else 0}')
print()
print('All \$refs:')
for path, target, depth in sorted(refs, key=lambda x: -x[2]):
    print(f'  depth={depth} {path} -> {target}')

# Check for potential cycles
targets = set(t for _, t, _ in refs)
sources = set()
for path, target, _ in refs:
    # Simple heuristic: if a definition references another definition that also has refs
    parts = target.split('/')
    if len(parts) > 1:
        sources.add(parts[-1])

# Check if any target name appears in source paths
for path, target, _ in refs:
    for src in sources:
        if src in path:
            print(f'  WARNING: Possible circular ref: {path} -> {target} (involves {src})')
"
```

### Step 5: Try Manual Compilation

```bash
# Attempt to compile the schema directly and capture the full error
curl -X POST http://localhost:8080/v1/schema/compile \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SCHEMA_API_KEY" \
  -d '{
    "schema_id": "<schema_id>",
    "version": "<version>"
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
success = data.get('success', False)
print(f'Compilation success: {success}')
print(f'Compile time: {data.get(\"compile_time_ms\", \"N/A\")} ms')
if data.get('warnings'):
    print(f'Warnings ({len(data[\"warnings\"])}):')
    for w in data['warnings']:
        print(f'  - {w}')
if data.get('errors'):
    print(f'Errors ({len(data[\"errors\"])}):')
    for e in data['errors']:
        print(f'  - {e}')
if data.get('ir'):
    ir = data['ir']
    print(f'IR schema_hash: {ir.get(\"schema_hash\", \"N/A\")[:16]}...')
    print(f'IR properties: {len(ir.get(\"properties\", {}))}')
    print(f'IR patterns: {len(ir.get(\"patterns\", {}))}')
    print(f'IR constraints: numeric={len(ir.get(\"numeric_constraints\", {}))}, '
          f'string={len(ir.get(\"string_constraints\", {}))}, '
          f'array={len(ir.get(\"array_constraints\", {}))}')
"
```

### Step 6: Check Recent Schema Registry Changes

```bash
# Check recent commits to the schema registry
kubectl exec -n greenlang <schema-service-pod> -- \
  git -C /data/schema-registry log --oneline -20 --since="24 hours ago"

# Check if the failing schema was recently modified
kubectl exec -n greenlang <schema-service-pod> -- \
  git -C /data/schema-registry log --oneline -10 -- schemas/<schema_id>/

# Diff the latest change to the failing schema
kubectl exec -n greenlang <schema-service-pod> -- \
  git -C /data/schema-registry diff HEAD~1 HEAD -- schemas/<schema_id>/

# Check if any new patterns were added
kubectl exec -n greenlang <schema-service-pod> -- \
  git -C /data/schema-registry diff HEAD~1 HEAD -- schemas/<schema_id>/ \
  | grep -i "pattern\|regex"
```

### Step 7: Check Resource Consumption During Compilation

```bash
# Check CPU usage during compilation attempts
kubectl top pods -n greenlang -l app=schema-service

# Check for CPU throttling
# PromQL: sum(rate(container_cpu_cfs_throttled_seconds_total{namespace="greenlang", pod=~"schema-service.*"}[5m]))
```

```promql
# CPU usage during compilation
rate(container_cpu_usage_seconds_total{namespace="greenlang", pod=~"schema-service.*"}[5m])

# Memory usage during compilation
container_memory_working_set_bytes{namespace="greenlang", pod=~"schema-service.*"}

# Correlation: compilation errors with CPU spikes
rate(glschema_compilation_errors_total[5m])
rate(container_cpu_usage_seconds_total{namespace="greenlang", pod=~"schema-service.*"}[5m])
```

---

## Resolution Steps

### Option 1: Fix Pathological Regex Patterns

If the compilation failure is caused by a ReDoS-vulnerable or overly complex regex pattern.

**1. Identify the problematic pattern(s) from logs:**

```bash
kubectl logs -n greenlang -l app=schema-service --tail=500 \
  | grep -i "pattern.*unsafe\|complexity score\|nested quantifier\|overlapping" \
  | head -20
```

**2. Common dangerous patterns and their safe replacements:**

| Dangerous Pattern | Why Dangerous | Safe Replacement |
|-------------------|---------------|------------------|
| `(a+)+` | Nested quantifier -- exponential backtracking | `a+` |
| `(a*)*` | Double Kleene star -- redundant and dangerous | `a*` |
| `(.*)+` | Nested wildcard quantifier | `.*` |
| `(.+)+` | Nested quantifier with dot | `.+` |
| `(a\|a)+` | Overlapping alternation -- duplicate branches | `a+` |
| `(a\|ab)+` | Overlapping alternation -- prefix overlap | `a(b)?+` or reorder as `(ab\|a)+` |
| `([a-zA-Z]+)*` | Character class with nested quantifier | `[a-zA-Z]*` |
| `(\d+)+` | Nested quantifier on digit class | `\d+` |
| `(\w+)+` | Nested quantifier on word class | `\w+` |
| `.*.*.*x` | Multiple adjacent wildcards | `.*x` |

**3. Update the schema in the registry with safe patterns:**

```bash
# Edit the schema file (substitute actual path)
kubectl exec -n greenlang <schema-service-pod> -- \
  vi /data/schema-registry/schemas/<schema_id>/v<version>.yaml

# Or apply a patch via Git
kubectl exec -n greenlang <schema-service-pod> -- \
  git -C /data/schema-registry add schemas/<schema_id>/
kubectl exec -n greenlang <schema-service-pod> -- \
  git -C /data/schema-registry commit -m "Fix ReDoS-vulnerable regex patterns in <schema_id>"

# Invalidate the cached IR for this schema
curl -X POST http://localhost:8080/v1/schema/cache/invalidate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SCHEMA_API_KEY" \
  -d '{"schema_id": "<schema_id>"}'
```

**4. Verify the fix:**

```bash
# Re-attempt compilation
curl -X POST http://localhost:8080/v1/schema/compile \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SCHEMA_API_KEY" \
  -d '{"schema_id": "<schema_id>", "version": "<version>"}' \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Success: {data.get(\"success\", False)}')
print(f'Compile time: {data.get(\"compile_time_ms\", \"N/A\")} ms')
print(f'Warnings: {len(data.get(\"warnings\", []))}')
print(f'Errors: {len(data.get(\"errors\", []))}')
"
```

### Option 2: Simplify the Schema Structure

If the compilation failure is caused by excessive schema complexity (too many properties, deep nesting, or many `$ref` expansions).

**1. Check which limit is being exceeded:**

```bash
kubectl logs -n greenlang -l app=schema-service --tail=500 \
  | grep -i "exceeds\|limit\|max_ref\|max_depth\|max_nodes\|too large"
```

**2. Refactor the schema to reduce complexity:**

Common refactoring strategies:
- **Split large schemas** into smaller, focused schemas with `$ref` composition
- **Remove deeply nested objects** -- flatten where possible
- **Reduce enum values** -- use external reference tables instead of inline enums
- **Limit `$ref` chain depth** -- avoid A -> B -> C -> D -> E chains; prefer flat composition
- **Use `$defs` instead of external `$ref`** for frequently referenced sub-schemas to avoid resolution overhead

**3. Temporarily increase compilation limits (use with caution):**

```bash
# Increase $ref expansion limit
kubectl set env deployment/schema-service -n greenlang \
  GL_SCHEMA_MAX_REF_EXPANSIONS=20000

# Increase object depth limit
kubectl set env deployment/schema-service -n greenlang \
  GL_SCHEMA_MAX_OBJECT_DEPTH=100

# Increase max total nodes
kubectl set env deployment/schema-service -n greenlang \
  GL_SCHEMA_MAX_TOTAL_NODES=500000

# Restart to apply
kubectl rollout restart deployment/schema-service -n greenlang
```

**Caution:** Increasing limits may allow schemas that cause slow compilation or high memory usage. Only increase limits as a temporary measure while the schema is being simplified.

### Option 3: Fix Circular `$ref` References

If the compilation failure is caused by circular references.

**1. Identify the cycle:**

```bash
kubectl logs -n greenlang -l app=schema-service --tail=500 \
  | grep -i "circular\|cycle\|recursion\|max.*ref.*expansion"
```

**2. Break the cycle by refactoring the schema:**

- Replace the recursive `$ref` with an inline definition at the point of recursion
- Add a `maxDepth` annotation to limit recursion depth
- Use `$defs` to create a non-recursive version of the referenced schema

**3. Verify the fix by re-compiling the schema (see Step 5 in diagnostics).**

### Option 4: Adjust Regex Safety Thresholds

If the compilation failure is caused by regex patterns that are flagged as dangerous but are actually safe for your use case.

**Caution:** Lowering regex safety thresholds increases the risk of ReDoS attacks. Only do this if you have verified the patterns are safe.

```bash
# Increase the maximum regex complexity score (from 0.8 to 0.9)
kubectl set env deployment/schema-service -n greenlang \
  GL_SCHEMA_MAX_REGEX_COMPLEXITY_SCORE=0.9

# Increase the maximum regex pattern length
kubectl set env deployment/schema-service -n greenlang \
  GL_SCHEMA_MAX_REGEX_LENGTH=2000

# Restart to apply
kubectl rollout restart deployment/schema-service -n greenlang
```

**Important:** After adjusting thresholds, manually review all patterns that were previously rejected to ensure they do not cause catastrophic backtracking. Test each pattern with adversarial inputs.

### Option 5: Rollback Schema Version

If a recent schema change introduced the compilation failure, rollback to the previous version.

```bash
# Identify the breaking commit
kubectl exec -n greenlang <schema-service-pod> -- \
  git -C /data/schema-registry log --oneline -10 -- schemas/<schema_id>/

# Revert the commit
kubectl exec -n greenlang <schema-service-pod> -- \
  git -C /data/schema-registry revert <commit-hash> --no-edit

# Invalidate cache for the affected schema
curl -X POST http://localhost:8080/v1/schema/cache/invalidate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SCHEMA_API_KEY" \
  -d '{"schema_id": "<schema_id>"}'

# Verify compilation succeeds with the rolled-back version
curl -X POST http://localhost:8080/v1/schema/compile \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SCHEMA_API_KEY" \
  -d '{"schema_id": "<schema_id>", "version": "<version>"}'
```

### Option 6: Fix Parse Errors

If the compilation failure is caused by invalid YAML/JSON syntax.

```bash
# Check for parse errors in logs
kubectl logs -n greenlang -l app=schema-service --tail=500 \
  | grep -i "parse error\|yaml\|json.*error\|syntax"

# Validate the schema file syntax
kubectl exec -n greenlang <schema-service-pod> -- \
  python3 -c "
import yaml, json, sys
path = '/data/schema-registry/schemas/<schema_id>/v<version>.yaml'
try:
    with open(path) as f:
        data = yaml.safe_load(f)
    print(f'YAML valid: {path}')
    print(f'Top-level keys: {list(data.keys())[:20]}')
except Exception as e:
    print(f'YAML parse error: {e}')
"

# Fix the syntax error and commit
kubectl exec -n greenlang <schema-service-pod> -- \
  vi /data/schema-registry/schemas/<schema_id>/v<version>.yaml
```

---

## Post-Resolution Verification

```bash
# 1. Verify the compilation succeeds
curl -X POST http://localhost:8080/v1/schema/compile \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SCHEMA_API_KEY" \
  -d '{"schema_id": "<schema_id>", "version": "<version>"}' \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
assert data.get('success', False), f'Compilation still failing: {data.get(\"errors\")}'
print(f'Compilation SUCCESS in {data.get(\"compile_time_ms\", \"?\")} ms')
print(f'Schema hash: {data.get(\"ir\", {}).get(\"schema_hash\", \"?\")[:16]}...')
print(f'Warnings: {len(data.get(\"warnings\", []))}')
"
```

```promql
# 2. Verify compilation error rate has dropped to 0
rate(glschema_compilation_errors_total[5m]) == 0

# 3. Verify validation is working for the previously failing schema
increase(glschema_validations_total{schema_id="<schema_id>"}[5m]) > 0

# 4. Verify cache is being populated for this schema
glschema_cache_size > 0
```

```bash
# 5. Test a validation against the fixed schema
curl -X POST http://localhost:8080/v1/schema/validate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $SCHEMA_API_KEY" \
  -d '{
    "schema_ref": {
      "schema_id": "<schema_id>",
      "version": "<version>"
    },
    "payload": <sample_valid_payload>
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Valid: {data.get(\"valid\")}')
print(f'Errors: {data.get(\"summary\", {}).get(\"error_count\", 0)}')
print(f'Warnings: {data.get(\"summary\", {}).get(\"warning_count\", 0)}')
"
```

---

## Schema Complexity Guidelines

To prevent compilation timeouts, follow these guidelines when authoring schemas.

### Property Limits

| Metric | Recommended Maximum | Hard Limit | Notes |
|--------|---------------------|------------|-------|
| Total properties | 500 | 10,000 | More properties = slower flattening |
| Nesting depth | 10 levels | 50 levels | Each level doubles traversal paths |
| Array items schema | Simple types preferred | 10,000 items | Complex item schemas multiply cost |
| Enum values | 100 | No hard limit | Large enums increase memory but not time |

### `$ref` Best Practices

| Practice | Recommendation |
|----------|---------------|
| Max `$ref` chain depth | Keep under 5 levels (A -> B -> C max) |
| Circular references | Never use; always break cycles with inline definitions |
| External references | Minimize; use `$defs` for frequently referenced sub-schemas |
| `$ref` count per schema | Keep under 50; more refs = more resolution overhead |
| Version pinning | Always pin `$ref` to specific versions (not "latest") |

### Regex Best Practices

| Practice | Recommendation |
|----------|---------------|
| Pattern length | Keep under 200 characters (limit: 1000) |
| Quantifier nesting | Never nest quantifiers: avoid `(a+)+`, `(a*)*` |
| Alternation | Ensure branches are mutually exclusive |
| Anchors | Always use `^` and `$` anchors for full-string patterns |
| Character classes | Prefer `[a-z]` over `.` (dot matches too broadly) |
| Backreferences | Avoid (not RE2-compatible, high complexity) |
| Lookaround | Avoid (not RE2-compatible, high complexity) |
| Possessive quantifiers | Use when available to prevent backtracking |
| Testing | Test all patterns with adversarial inputs before deploying |

### VulnerabilityType Reference

The `RegexAnalyzer` classifies vulnerabilities into the following types:

| Type | Description | Example | Estimated Worst-Case |
|------|-------------|---------|---------------------|
| `NESTED_QUANTIFIER` | Quantifier inside a group that also has a quantifier | `(a+)+` | O(2^n) |
| `OVERLAPPING_ALTERNATION` | Alternation branches that match the same prefix | `(a\|ab)+` | O(2^n) |
| `EXPONENTIAL_BACKTRACK` | Pattern with multiple adjacent wildcards | `.*.*.*x` | O(n^k) |
| `CATASTROPHIC_BACKTRACK` | Pattern exceeding complexity score or length limit | Complex patterns | Variable |
| `UNBOUNDED_REPETITION` | Repetition without upper bound on complex group | `([a-z]+)*` | O(n^2) |

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Single schema compilation failing, no pipeline blocked | On-call engineer | 30 minutes |
| L2 | Multiple schemas failing or critical pipeline blocked | Platform team lead | 15 minutes |
| L3 | ReDoS pattern detected in production, potential security incident | Platform team + Security team | Immediate |
| L4 | Compilation failures causing cascading downstream outages | All-hands engineering + incident commander | Immediate |

---

## Prevention

### Schema Review Process

1. **All schema changes must pass CI/CD validation** before merging to the registry
2. **Run `glschema validate` in CI** against test fixtures to catch breaking changes early
3. **Run regex safety analysis** as a pre-commit hook:
   ```bash
   glschema validate --schema-ref <schema_id>:<version> \
     --test-data tests/fixtures/<schema_id>/valid_payloads/*.json
   ```
4. **Review all regex patterns** in schema PRs for ReDoS risk
5. **Keep schemas under complexity limits** (see guidelines above)

### Automated Safety Checks

- CI pipeline should run the `RegexAnalyzer` on all `pattern` and `patternProperties` fields
- Block deployment if any pattern has a complexity score > 0.8
- Block deployment if any known dangerous pattern is detected
- Enforce `MAX_REGEX_LENGTH` in CI (patterns > 1000 characters are rejected)

### Monitoring

- **Dashboard:** Schema Service Health (`/d/schema-service-health`) -- compilation panels
- **Dashboard:** Schema Validation Overview (`/d/schema-validation-overview`)
- **Alert:** `SchemaCompilationFailure` (this alert)
- **Alert:** `SchemaCompilationTimeout` (this alert)
- **Alert:** `SchemaReDoSDetected` (this alert)
- **Key metrics to watch:**
  - `glschema_compilation_errors_total` (should be 0 in steady state)
  - `glschema_compilation_duration_seconds` (P99 should be < 1s)
  - `glschema_validations_failed` rate correlated with compilation errors
  - Cache hit rate (compilation failures mean no IR is cached)

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Team
- **Review cadence:** Quarterly or after any compilation failure incident
- **Related alerts:** `SchemaServiceDown`, `SchemaHighValidationErrorRate`, `SchemaCacheHitRateLow`
- **Related dashboards:** Schema Service Health, Schema Validation Overview
- **Related runbooks:** [Schema Service Down](./schema-service-down.md), [High Validation Errors](./high-validation-errors.md), [Schema Cache Corruption](./schema-cache-corruption.md)
