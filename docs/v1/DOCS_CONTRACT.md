# GreenLang v1 Documentation Contract

Each v1 app profile must provide:

1. Quickstart (how to run and validate)
2. Runbook (ops and incident handling)
3. Artifact glossary
4. Policy expectations
5. Determinism verification steps

## Required Document Set

- `docs/v1/QUICKSTART.md`
- `docs/v1/RUNBOOK_TEMPLATE.md` (instantiated per app)
- `docs/v1/SECURITY_POLICY_BASELINE.md`
- `docs/v1/STANDARDS.md`
- `docs/v1/RELEASE_CHECKLIST.md`

## Acceptance

A non-core team should be able to run:

```bash
gl v1 status
gl v1 validate-contracts
gl v1 check-policy
gl v1 gate
```

without additional tribal knowledge.

