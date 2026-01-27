# ADR-006: Deterministic Plan Generation

**Date:** 2026-01-27
**Status:** Accepted
**Deciders:** Architecture Team, Platform Engineering, Governance/Compliance
**Consulted:** Agent Development Teams, QA/Testing, External Auditors

---

## Context

### Problem Statement
GL-FOUND-X-001 (GreenLang Orchestrator) must guarantee that identical inputs produce identical execution plans. This determinism is critical for:
- Audit reproducibility (same plan hash proves same execution intent)
- Governance verification (policy decisions based on stable plan)
- Debugging (reproduce issues with same plan)
- Zero-hallucination (no non-deterministic AI behavior in planning)

### Current Situation
- **Regulatory Requirement:** Auditors must verify that emission calculations are reproducible
- **AI Agent Concern:** LLM-based agents can introduce non-determinism
- **Pipeline Complexity:** 402 agents, complex DAGs, variable configurations
- **Compliance Standard:** ISO 14064 requires documented, verifiable methodologies

### Business Impact
- **Audit Efficiency:** Same inputs = same plan = quick verification
- **Trust:** Customers need confidence in calculation reproducibility
- **Debugging:** Non-determinism makes issues impossible to reproduce
- **Certification:** Third-party assurance requires deterministic behavior

---

## Decision

### What We're Implementing
**Content-addressable plan_id** derived from normalized inputs, with **normalized YAML** processing to ensure deterministic compilation.

### Core Principles

1. **Content-Addressable Planning**
   - `plan_id` is hash of normalized inputs
   - Same inputs always produce same `plan_id`
   - Plan hash proves planning intent

2. **Normalized Input Processing**
   ```
   Raw Pipeline YAML
         |
         v
   +------------------+
   | YAML Parser      |  <-- Parse with strict mode
   +------------------+
         |
         v
   +------------------+
   | Key Ordering     |  <-- Sort all keys alphabetically
   +------------------+
         |
         v
   +------------------+
   | Whitespace Norm  |  <-- Remove insignificant whitespace
   +------------------+
         |
         v
   +------------------+
   | Value Canonical  |  <-- Normalize booleans, numbers, nulls
   +------------------+
         |
         v
   Normalized YAML (deterministic string)
   ```

3. **Plan Hash Computation**
   ```python
   def compute_plan_hash(
       pipeline_yaml: str,
       run_config: dict,
       registry_snapshot: dict,
       policy_bundle_hash: str
   ) -> str:
       """Compute deterministic plan hash from inputs."""

       # Normalize pipeline YAML
       normalized_pipeline = normalize_yaml(pipeline_yaml)

       # Create canonical input structure
       plan_input = {
           "pipeline_yaml_hash": sha256(normalized_pipeline),
           "run_config": canonicalize(run_config),
           "registry_snapshot_hash": registry_snapshot["hash"],
           "policy_bundle_hash": policy_bundle_hash
       }

       # Compute hash
       canonical_json = json.dumps(plan_input, sort_keys=True)
       return "sha256:" + hashlib.sha256(canonical_json.encode()).hexdigest()
   ```

4. **Plan ID Structure**
   ```
   plan_id = "pln_" + base62(plan_hash[:16])

   Example: pln_7Ks9xYz2mNpQrT4w
   ```

### Normalization Rules

| Element | Normalization Rule |
|---------|-------------------|
| Key ordering | Alphabetical at all levels |
| Whitespace | Single space, no trailing |
| Booleans | `true`/`false` (lowercase) |
| Numbers | No leading zeros, no trailing decimals |
| Nulls | `null` (lowercase) |
| Strings | UTF-8, escape special chars |
| Lists | Order preserved (significant) |
| Comments | Stripped |
| Anchors/Aliases | Expanded inline |

### Technology Stack
- **YAML Parser:** PyYAML with safe loader
- **Normalization:** Custom normalizer with RFC 8785 (Canonical JSON)
- **Hashing:** SHA-256
- **Encoding:** Base62 for human-readable IDs

### Code Location
- `greenlang/orchestrator/planning/`
  - `normalizer.py` - YAML normalization
  - `hasher.py` - Plan hash computation
  - `compiler.py` - Pipeline compilation
  - `plan.py` - Plan data structures
  - `registry.py` - Agent registry snapshotting

---

## Rationale

### Why Content-Addressable Plan ID

**1. Reproducibility Guarantee**
- Identical inputs = identical plan_id
- Auditors can verify by recomputing hash
- No hidden state affects planning

**2. Audit-Friendly**
- Plan hash is proof of planning intent
- Can verify plan without re-executing
- Clear link between inputs and plan

**3. Zero-Hallucination**
- No AI/LLM in planning path
- Pure deterministic computation
- No temperature/sampling variations

**4. Debugging**
- Reproduce exact plan from inputs
- Compare plan hashes to detect differences
- Track plan evolution over time

**5. Governance**
- Policy decisions based on stable plan
- Same plan = same policy evaluation
- Approved plans can be allowlisted

### Why Normalized YAML

**1. Whitespace Independence**
- Formatting doesn't affect hash
- Teams can use different editors
- No accidental plan changes from reformatting

**2. Key Order Independence**
- YAML allows arbitrary key order
- Normalization removes ordering ambiguity
- Consistent regardless of authoring tool

**3. Comment Stripping**
- Comments are documentation, not semantics
- Removing comments ensures semantic hashing
- Authors can update comments without changing plan

---

## Alternatives Considered

### Alternative 1: Random UUID for plan_id
**Pros:**
- Simple to generate
- Guaranteed unique
- No normalization needed

**Cons:**
- Same inputs produce different plan_ids
- Cannot prove reproducibility
- Auditors cannot verify planning
- Debugging difficult

**Why Rejected:** Does not meet audit reproducibility requirements. Plan ID must prove planning intent.

### Alternative 2: Timestamp-Based IDs
**Pros:**
- Natural ordering
- Easy to generate
- Human-readable

**Cons:**
- Same inputs at different times = different IDs
- Cannot prove reproducibility
- Time dependency adds non-determinism

**Why Rejected:** Temporal dependency violates determinism requirement.

### Alternative 3: Hash Only Pipeline YAML
**Pros:**
- Simpler computation
- Less input dependency

**Cons:**
- Ignores run config variations
- Ignores agent version changes
- Ignores policy bundle changes
- Same YAML with different configs = same hash (wrong)

**Why Rejected:** Incomplete input coverage. Plan must capture all factors affecting execution.

### Alternative 4: AI-Assisted Planning
**Pros:**
- Could optimize execution
- Could suggest improvements
- More intelligent scheduling

**Cons:**
- Non-deterministic outputs
- Temperature/sampling variance
- Cannot guarantee reproducibility
- Audit risk

**Why Rejected:** Conflicts with zero-hallucination requirement. AI assistance can be post-planning recommendation, not in planning path.

---

## Consequences

### Positive
- **Reproducibility:** Identical inputs = identical plan, guaranteed
- **Audit Proof:** Plan hash proves planning intent
- **Debugging:** Reproduce any plan from recorded inputs
- **Governance:** Stable basis for policy decisions
- **Zero-Hallucination:** No AI variance in planning

### Negative
- **Normalization Complexity:** Must handle all YAML edge cases
- **Input Capture:** Must snapshot all inputs at planning time
- **Registry Dependency:** Registry changes affect plan hash
- **Storage Overhead:** Must store registry snapshots
- **Performance:** Normalization adds computation (minimal)

### Neutral
- **Learning Curve:** Developers must understand determinism requirements
- **Tooling:** Need tools to compare and debug plans

---

## Implementation Plan

### Phase 1: Normalization (Week 1-2)
1. Implement YAML normalizer with all rules
2. Build comprehensive test suite for edge cases
3. Create normalization CLI tool for debugging
4. Document normalization rules

### Phase 2: Hashing (Week 3)
1. Implement plan hash computation
2. Build registry snapshot capture
3. Create plan_id generation
4. Add hash verification utilities

### Phase 3: Compiler (Week 4-5)
1. Integrate normalization into pipeline compiler
2. Implement DAG generation with stable ordering
3. Add step_id generation from plan_hash
4. Create plan serialization/storage

### Phase 4: Verification (Week 6)
1. Build determinism test framework
2. Create regression tests for plan stability
3. Add CI/CD checks for determinism
4. Document verification process for auditors

---

## Determinism Guarantees

### What IS Deterministic
- Pipeline YAML parsing and normalization
- Agent version resolution from registry snapshot
- DAG topology computation
- Step ordering within topological constraints
- Plan hash computation
- Step ID generation

### What is NOT Deterministic (by design)
- `run_id` (includes timestamp or random nonce)
- Execution wall-clock times
- Step completion order (parallel execution)
- Log timestamps
- External system responses

### Verification Test
```python
def test_determinism():
    """Same inputs must produce same plan hash."""
    pipeline_yaml = load_test_pipeline()
    run_config = {"params": {"input": "s3://test"}}
    registry = snapshot_registry()
    policy_hash = "sha256:abc..."

    # Compile 100 times
    hashes = set()
    for _ in range(100):
        plan = compile_pipeline(
            pipeline_yaml,
            run_config,
            registry,
            policy_hash
        )
        hashes.add(plan.plan_hash)

    # Must produce exactly one unique hash
    assert len(hashes) == 1, "Non-deterministic planning detected!"
```

---

## Step ID Generation

### Stable Step IDs
```python
def generate_step_id(plan_hash: str, pipeline_step_id: str) -> str:
    """Generate stable step ID from plan and step definition."""
    input_string = f"{plan_hash}:{pipeline_step_id}"
    step_hash = hashlib.sha256(input_string.encode()).hexdigest()[:12]
    return f"stp_{pipeline_step_id}_{step_hash}"

# Example:
# plan_hash: "sha256:abc123..."
# pipeline_step_id: "ingest"
# step_id: "stp_ingest_7ks9xyz2mnpq"
```

### Benefits
- Same plan = same step IDs
- Step IDs are human-readable (include pipeline_step_id)
- Unique across plans (plan_hash included)
- Auditable (can verify derivation)

---

## Compliance & Security

### Security Considerations
- **Hash Algorithm:** SHA-256 (collision resistant)
- **Canonical Format:** RFC 8785 prevents manipulation
- **Input Integrity:** All inputs captured and hashed
- **No Secrets in Hash:** Secrets resolved at runtime, not planning

### Audit Considerations
- **Reproducibility Test:** Auditors can recompute plan hash
- **Input Archive:** Store all inputs for audit replay
- **Plan Comparison:** Diff tool for comparing plans
- **Certification:** Determinism can be certified

---

## Migration Plan

### Short-term (0-6 months)
- Deploy deterministic planner
- Store plan hashes with all runs
- Build verification tools

### Medium-term (6-12 months)
- Add plan comparison dashboards
- Implement plan approval workflows
- Create audit reports with plan verification

### Long-term (12+ months)
- Third-party certification of determinism
- Plan optimization recommendations (post-planning)
- Plan version control and history

---

## Links & References

- **PRD:** GL-FOUND-X-001 GreenLang Orchestrator
- **Related ADRs:** ADR-001 (GLIP v1), ADR-005 (Hash-Chained Audit)
- **Canonical JSON:** [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785)
- **Content-Addressable Storage:** [Wikipedia](https://en.wikipedia.org/wiki/Content-addressable_storage)

---

## Updates

### 2026-01-27 - Status: Accepted
ADR approved by Architecture and Governance teams. Normalizer implementation starting Q1 2026.

---

**Template Version:** 1.0
**Last Updated:** 2026-01-27
**ADR Author:** Platform Architecture Team
**Reviewers:** Governance/Compliance, QA/Testing, External Auditors
