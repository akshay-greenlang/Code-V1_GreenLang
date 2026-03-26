# GreenLang v1 Pack Lifecycle Playbook

## Lifecycle States

- `draft`: early development, no support commitment
- `candidate`: conformance-ready, awaiting promotion evidence
- `supported`: v1 approved, signed, and covered by gate checks
- `deprecated`: migration path required, support sunset defined

## Promotion Rules

1. Contract validation passes (`gl v1 validate-contracts`).
2. Signed-pack baseline passes (`gl v1 check-policy`).
3. Release gate passes (`gl v1 gate`).
4. App owner approves promotion in release review.

## Demotion Rules

- Repeated gate failures
- Security or policy violations
- Unsupported runtime conventions

## Deprecation Process

1. Mark lifecycle as `deprecated`.
2. Publish migration target and timeline.
3. Keep security fixes during deprecation window.

