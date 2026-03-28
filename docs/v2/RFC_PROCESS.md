# GreenLang V2 RFC Process

## Purpose

Prevent app-specific drift by requiring versioned decisions for runtime, contract, and policy changes.

## RFC Required For

- runtime command grammar/exit-code changes
- `pack.yaml` / `gl.yaml` breaking changes
- pack tier policy changes
- determinism contract changes
- release train promotion criteria changes

## RFC Lifecycle

1. **Draft**: author, context, alternatives, migration impact.
2. **Review**: architecture + security + SRE signoff.
3. **Accepted**: assigned implementation owner and target train.
4. **Implemented**: code, tests, docs, migration playbook complete.
5. **Verified**: release-gate evidence attached.

## RFC SLAs

- draft-to-review start: <= 3 business days from submission.
- review-to-accept/reject decision: <= 7 business days after review starts.
- accepted-to-implementation start: <= 10 business days.
- exception approvals: max 30-day validity, then mandatory renewal or closure.

## No-Merge Governance Rule

- breaking runtime or contract changes must not merge without an accepted RFC.
- pull requests with `breaking-change` label require RFC ID and acceptance link.
- emergency exceptions require Security Council + Architecture Board co-approval and 30-day expiry.

## Required Sections

- problem statement
- proposed change
- compatibility impact
- rollout plan
- fallback/rollback plan
- test plan
- migration notes
- ownership and on-call impact
