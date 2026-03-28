# GreenLang V2 Platform Handbook

## Purpose

Single operational and engineering handbook for V2 scale-and-productization delivery.

## Core Sections

1. Runtime and CLI contract
2. App profile onboarding
3. Pack tier lifecycle and signing
4. Agent lifecycle governance
5. Connector reliability operations
6. Security and policy governance
7. Determinism and audit contract
8. Release train and promotion policy
9. Enterprise UX governance
10. Go/no-go evidence model

## Golden Paths

### New App Onboarding

1. add `applications/<app>/v2/pack.yaml` and `gl.yaml`
2. register app profile in `greenlang/v2/profiles.py`
3. validate with `gl v2 validate-contracts`
4. pass `gl v2 gate` and V2 CI blockers

### Pack Promotion

1. pass contract tests
2. pass security scans
3. pass determinism checks
4. complete docs/runbook
5. apply tier promotion in governance review
