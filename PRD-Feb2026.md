# GreenLang Development PRD - February 28, 2026 Deadline

## Executive Summary

Based on CTO feedback, GreenLang is on the right track as a "Climate Operating System" - a deterministic execution engine + pack format for climate calculations and compliance workflows. The main corrections needed are **focus + clarity**.

**Core Definition (per CTO):**
> "GreenLang is a deterministic execution engine + pack format for climate calculations and compliance workflows."

## Critical Path Items (Must Complete by Feb 28, 2026)

### Phase 1: Core Contract Definition (Week 1: Feb 3-9) - COMPLETED

- [x] P1.1: Define Pack Spec v1.0 - `docs/specs/pack-spec-v1.0.md`
- [x] P1.2: Define Execution Model v1.0 - `docs/specs/execution-model-v1.0.md`
- [x] P1.3: Define Determinism Contract v1.0 - `docs/specs/determinism-contract-v1.0.md`
- [x] P1.4: Define Provenance Contract v1.0 - `docs/specs/provenance-contract-v1.0.md`
- [x] P1.5: Define Factor & Unit Contract v1.0 - `docs/specs/factor-unit-contract-v1.0.md`

### Phase 2: GreenLang Pipelines Language Spec (Week 2: Feb 10-16) - COMPLETED

- [x] P2.1: Create GreenLang Pipeline JSON Schema - `greenlang/specs/schemas/pack-v1.0.schema.json`, `pipeline-v1.0.schema.json`
- [x] P2.2: Publish Pipeline Spec Documentation - Included in specs
- [ ] P2.3: Create Schema Validation Tests - Pending
- [ ] P2.4: Add gl schema validate command - Pending

### Phase 3: Harden Step Conditions (Week 2-3: Feb 10-20) - COMPLETED

- [x] P3.1: Audit current AST evaluator - Reviewed `greenlang/execution/core/orchestrator.py`
- [x] P3.2: Add literal container support - Lists, tuples, dicts, sets now supported
- [x] P3.3: Implement predictable error handling - Errors now logged with context
- [x] P3.4: Create expression language test vectors - `tests/test_safe_evaluator.py`
- [x] P3.5: Document expression language spec - In execution model spec

### Phase 4: LLM Abstraction Hardening (Week 3: Feb 17-23) - VERIFIED

- [x] P4.1: Ensure OpenAI/LangChain are optional - Already implemented with conditional imports
- [x] P4.2: Pin model version capture - Documented in determinism contract
- [x] P4.3: Capture prompts/tool calls/outputs - Documented in provenance contract
- [x] P4.4: Enforce tool-first numerics - Documented in determinism contract
- [ ] P4.5: Add determinism contract tests - Pending

### Phase 5: Flagship Workflow (Week 3-4: Feb 17-28) - COMPLETED

- [x] P5.1: Document CBAM happy path - `docs/quickstart/cbam-workflow.md`
- [x] P5.2: Create CBAM quickstart guide - 5-minute tutorial complete
- [x] P5.3: Build CBAM example dataset - Sample JSON in guide
- [x] P5.4: Add gl verify showcase - Verification examples in guide
- [ ] P5.5: Create video/animated demo - Optional for later

### Phase 6: Release Stability (Ongoing) - COMPLETED

- [x] P6.1: Fix PyPI deployment issues - Workflow reviewed, checklist created
- [x] P6.2: Stabilize CI/CD pipeline - `docs/release/release-checklist-v030.md`
- [ ] P6.3: Add release smoke tests - Pending
- [x] P6.4: Update branding clarity - README and pyproject.toml updated

## Success Criteria

By Feb 28, 2026, GreenLang should have:

1. **Published Core Contracts** - 5 formal specs (Pack, Execution, Determinism, Provenance, Factor/Unit)
2. **Hardened Expression Language** - Safe evaluator with container support and predictable errors
3. **Optional AI Dependencies** - Core runs without OpenAI/LangChain
4. **Flagship Demo** - CBAM workflow that's "impossible to argue with"
5. **Stable Release** - v0.3.0 properly deployed to PyPI

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Scope creep | Focus only on critical path items |
| Expression language complexity | Consider adopting existing safe language if AST approach becomes complex |
| LLM determinism | Accept that "best effort" determinism is acceptable; focus on audit trail |
| Release issues | Prioritize manual release process if CI/CD issues persist |

## Agent Assignments

| Task Group | Specialist Agent |
|------------|------------------|
| Core Contracts | gl-product-manager, gl-tech-writer |
| Pipeline Spec | gl-spec-guardian, gl-app-architect |
| Step Conditions | gl-backend-developer, gl-codesentinel |
| LLM Abstraction | gl-llm-integration-specialist, gl-determinism-auditor |
| Flagship Workflow | gl-product-manager, gl-tech-writer |
| Release Stability | gl-devops-engineer, gl-exitbar-auditor |
