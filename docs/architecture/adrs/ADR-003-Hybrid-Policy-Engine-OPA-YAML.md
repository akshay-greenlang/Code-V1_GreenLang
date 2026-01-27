# ADR-003: Hybrid Policy Engine (OPA + YAML)

**Date:** 2026-01-27
**Status:** Accepted
**Deciders:** Architecture Team, Governance/Compliance, Security Team
**Consulted:** Platform Engineering, Agent Development Teams, Enterprise Customers

---

## Context

### Problem Statement
GL-FOUND-X-001 (GreenLang Orchestrator) requires a governance layer to enforce policies at multiple points:
- Pre-run validation (pipeline approval)
- Pre-step execution (permissions, budgets)
- Post-step validation (artifact classification, export controls)

The policy engine must balance **power** (complex rule expression) with **usability** (accessible to non-developers).

### Current Situation
- **Regulatory Requirements:** CBAM, CSRD, GHG Protocol require auditable policy enforcement
- **User Personas:** Policy authors range from developers (Rego proficient) to compliance officers (prefer YAML)
- **Policy Complexity:** Simple rules (cost limits) to complex rules (data residency, PII handling)
- **Audit Needs:** Every policy decision must be recorded with reason codes

### Business Impact
- **Compliance Risk:** Missing policy enforcement = regulatory violations
- **User Adoption:** Steep learning curve = policy gaps
- **Audit Efficiency:** Clear reason codes reduce audit time
- **Enterprise Sales:** Governance capabilities are key differentiators

---

## Decision

### What We're Implementing
**Hybrid Policy Engine**: OPA/Rego for complex rules, YAML for simple rules, with unified evaluation and audit trail.

### Core Architecture

1. **Two Policy Languages**
   - **YAML Policies:** Simple, declarative rules for common cases
   - **Rego Policies:** Full OPA/Rego for complex logic

2. **YAML Policy Format (Simple Rules)**
   ```yaml
   apiVersion: policy.greenlang.io/v1
   kind: SimplePolicy
   metadata:
     name: cost-budget-limit
     namespace: production
   spec:
     type: pre-run
     rules:
       - name: max-cost-per-run
         condition:
           field: run_config.budgets.max_cost_usd
           operator: lessThanOrEqual
           value: 100
         action: deny
         message: "Run cost budget exceeds $100 limit"
         severity: error

       - name: require-approval-for-production
         condition:
           field: pipeline.metadata.namespace
           operator: equals
           value: production
         requires:
           approval:
             roles: [PipelineApprover]
             min_approvers: 1
         action: require-approval
         message: "Production runs require approval"
         severity: warning
   ```

3. **Rego Policy Format (Complex Rules)**
   ```rego
   # policy/data_residency.rego
   package greenlang.policy.data_residency

   import future.keywords.in
   import future.keywords.if

   default allow := false

   allow if {
       not violates_residency
   }

   violates_residency if {
       some step in input.plan.steps
       step.agent.data_regions[_] != input.run_config.required_region
   }

   deny[msg] if {
       violates_residency
       msg := sprintf(
           "Step '%s' agent processes data in region '%s', but run requires '%s'",
           [input.plan.steps[_].pipeline_step_id,
            input.plan.steps[_].agent.data_regions[_],
            input.run_config.required_region]
       )
   }
   ```

4. **Unified Evaluation Pipeline**
   ```
   Policy Input (run/step context)
          |
          v
   +------------------+
   | Policy Compiler  |  <-- Converts YAML to internal representation
   +------------------+
          |
          v
   +------------------+
   | YAML Evaluator   |  <-- Fast path for simple rules
   +------------------+
          |
          v
   +------------------+
   | OPA Evaluator    |  <-- Complex rules in Rego
   +------------------+
          |
          v
   +------------------+
   | Decision Merger  |  <-- Combine results, apply precedence
   +------------------+
          |
          v
   Policy Decision (allow/deny + reason codes)
   ```

### Technology Stack
- **OPA:** Open Policy Agent v0.60+ (embedded or sidecar)
- **YAML Parser:** Pydantic models for schema validation
- **Evaluation Mode:** Hybrid (YAML first, then OPA)
- **Caching:** Policy bundle caching with version hashing

### Code Location
- `greenlang/orchestrator/policy/`
  - `engine.py` - Unified policy evaluation engine
  - `yaml_evaluator.py` - YAML policy evaluator
  - `opa_client.py` - OPA integration client
  - `compiler.py` - YAML to internal representation
  - `models.py` - Policy input/output models
  - `audit.py` - Policy decision auditing

---

## Rationale

### Why Hybrid (OPA + YAML)

**1. Balance of Power and Usability**
- YAML: Accessible to compliance officers and business analysts
- Rego: Full expressiveness for security engineers and developers
- Users choose based on their skill level and rule complexity

**2. Progressive Complexity**
- Start with YAML for simple rules
- Graduate to Rego when YAML becomes limiting
- No forced learning curve for basic governance

**3. Audit-Friendly**
- Both languages produce structured reason codes
- Unified audit trail regardless of policy language
- Clear mapping from policy to decision

**4. Industry Standard**
- OPA is the de facto standard for cloud-native policy
- YAML is universally understood
- Skills are transferable and hirable

**5. Performance**
- YAML evaluation is fast (simple comparisons)
- OPA evaluates in parallel
- Caching reduces repeated evaluations

---

## Alternatives Considered

### Alternative 1: Pure OPA/Rego
**Pros:**
- Single policy language
- Maximum expressiveness
- Strong ecosystem and tooling

**Cons:**
- Steep learning curve for non-developers
- Rego syntax unfamiliar to compliance teams
- Overkill for simple rules (cost < 100)
- Slower adoption in enterprise settings

**Why Rejected:** Accessibility concerns. Compliance officers and business analysts struggle with Rego syntax, leading to policy gaps or shadow policies.

### Alternative 2: Custom DSL
**Pros:**
- Tailored to GreenLang domain
- Can be optimized for common cases
- Full control over syntax and semantics

**Cons:**
- Must build and maintain parser, evaluator, tooling
- No existing ecosystem or community
- Learning curve for new hires
- Long development time

**Why Rejected:** Building a custom DSL is significant engineering effort with limited benefit over existing solutions. OPA provides mature tooling we would need to replicate.

### Alternative 3: Python Functions
**Pros:**
- Full programming language expressiveness
- Familiar to engineering teams
- Easy testing and debugging

**Cons:**
- Security risk (arbitrary code execution)
- Sandboxing is complex
- Not accessible to non-developers
- Harder to audit (code vs. declarations)

**Why Rejected:** Security concerns with arbitrary code execution. Policy-as-code should be declarative, not imperative, for audit clarity.

### Alternative 4: Pure YAML
**Pros:**
- Maximum accessibility
- Simple to understand and audit
- No learning curve

**Cons:**
- Limited expressiveness
- Cannot handle complex logic (data residency, PII detection)
- Would need many escape hatches
- Eventually becomes Turing-incomplete mess

**Why Rejected:** Insufficient expressiveness for complex regulatory requirements. Would hit limitations quickly and require workarounds.

---

## Consequences

### Positive
- **Accessibility:** YAML policies accessible to non-developers
- **Power:** Rego available for complex requirements
- **Adoption:** Lower barrier to governance adoption
- **Audit:** Unified audit trail with clear reason codes
- **Ecosystem:** Leverage OPA tooling, testing, and community
- **Flexibility:** Users choose appropriate tool for the job

### Negative
- **Two Languages:** Must maintain and document two policy languages
- **Training:** Teams need training on both YAML and Rego
- **Complexity:** Hybrid evaluation adds implementation complexity
- **Precedence:** Must define clear rules when YAML and Rego conflict
- **Testing:** Need test frameworks for both languages

### Neutral
- **OPA Dependency:** Adds OPA as infrastructure dependency
- **Performance Tuning:** May need optimization for high-volume scenarios

---

## Implementation Plan

### Phase 1: YAML Evaluator (Week 1-2)
1. Define YAML policy schema (OpenAPI/JSON Schema)
2. Implement YAML policy parser and validator
3. Build simple rule evaluator (field comparisons)
4. Add condition operators (equals, lessThan, contains, regex)

### Phase 2: OPA Integration (Week 3-4)
1. Deploy OPA as embedded library or sidecar
2. Implement OPA client with bundle loading
3. Build policy input schema for Rego
4. Add decision parsing and reason extraction

### Phase 3: Unified Engine (Week 5-6)
1. Implement policy compiler (YAML to internal)
2. Build decision merger with precedence rules
3. Add policy caching with version invalidation
4. Implement audit logging for all decisions

### Phase 4: Tooling (Week 7-8)
1. Create policy testing framework
2. Build policy linting CLI
3. Add VS Code extension for policy authoring
4. Write comprehensive documentation

---

## Policy Evaluation Points

### Pre-Run Policies
- Pipeline submission authorization
- Budget validation
- Namespace restrictions
- Schedule constraints

### Pre-Step Policies
- Agent permissions
- Data access authorization
- Resource quota checks
- Approval gate verification

### Post-Step Policies
- Artifact classification validation
- Export control enforcement
- PII detection results
- Lineage completeness

---

## Compliance & Security

### Security Considerations
- **OPA Isolation:** Run OPA with minimal privileges
- **Policy Integrity:** Sign policy bundles
- **Input Sanitization:** Validate all policy inputs
- **No Code Injection:** YAML/Rego only, no arbitrary code

### Audit Considerations
- **Decision Records:** Store all policy decisions
- **Reason Codes:** Structured, machine-readable reasons
- **Policy Versions:** Link decisions to policy bundle version
- **Immutability:** Append-only decision log

### Example Audit Record
```json
{
  "decision_id": "dec_abc123",
  "timestamp": "2026-01-27T10:30:00Z",
  "evaluation_point": "pre-run",
  "run_id": "run_xyz789",
  "policy_bundle_hash": "sha256:abc...",
  "policies_evaluated": [
    {
      "policy": "cost-budget-limit",
      "type": "yaml",
      "result": "allow"
    },
    {
      "policy": "data_residency",
      "type": "rego",
      "result": "deny",
      "reasons": [
        "Step 'ingest' agent processes data in region 'us-east-1', but run requires 'eu-west-1'"
      ]
    }
  ],
  "final_decision": "deny",
  "reason_codes": ["DATA_RESIDENCY_VIOLATION"]
}
```

---

## Migration Plan

### Short-term (0-6 months)
- Deploy hybrid policy engine
- Migrate existing permission checks to YAML policies
- Train compliance team on YAML authoring

### Medium-term (6-12 months)
- Develop Rego policies for complex regulations (CBAM, CSRD)
- Build policy testing and CI/CD integration
- Create policy library for common patterns

### Long-term (12+ months)
- Policy marketplace for reusable policies
- AI-assisted policy generation from requirements
- Continuous compliance monitoring

---

## Links & References

- **PRD:** GL-FOUND-X-001 GreenLang Orchestrator
- **Related ADRs:** ADR-005 (Hash-Chained Audit)
- **OPA Documentation:** [openpolicyagent.org](https://www.openpolicyagent.org/)
- **Rego Playground:** [play.openpolicyagent.org](https://play.openpolicyagent.org/)

---

## Updates

### 2026-01-27 - Status: Accepted
ADR approved by Architecture and Governance teams. YAML schema design starting Q1 2026.

---

**Template Version:** 1.0
**Last Updated:** 2026-01-27
**ADR Author:** Platform Architecture Team
**Reviewers:** Governance/Compliance, Security Team, Enterprise Customers
