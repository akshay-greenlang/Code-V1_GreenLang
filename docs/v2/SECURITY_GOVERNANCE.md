# GreenLang V2 Security Governance

## Objective

Enforce org-level policy-as-code and security scans as mandatory release blockers.

## Mandatory Security Gates

1. secrets scan gate
2. static application security testing gate
3. dependency vulnerability gate
4. policy bundle validation gate
5. signed-pack policy gate for protected tiers

## Branch Protection Policy

- `main`, `master`, and `release/**` require all V2 security jobs green.
- no waiver may bypass critical findings.
- waivers for medium findings require expiration date and approver.

## Policy-as-Code Control Set

- pack tier/signature enforcement
- execution policy baseline and deny-by-default posture
- restricted egress for regulated-critical workflows
- governance contract checks for release artifacts
