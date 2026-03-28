# GreenLang V2 RC Soak Log

## Candidate

- Candidate version: v2.0-rc1
- Start: 2026-03-28T01:30:00Z
- End: 2026-03-28T02:00:00Z
- Evidence scope: V2 contracts, policy, determinism, release-train, UX blockers

## Soak Checks

| Check | Result | Notes |
| --- | --- | --- |
| `gl v2 validate-contracts` | Pass | V2 profile contracts valid. |
| `gl v2 runtime-checks` | Pass | Command grammar and runtime conventions aligned. |
| `gl v2 docs-check` | Pass | Required V2 docs present. |
| `gl v2 gate` | Pass | Combined release gate checks passed. |
| `pytest -q tests/v2` | Pass | V2 policy/reliability/determinism tests passed. |
| UX visual baseline tests | Pass | Shell visual baseline test suite green. |

## Conclusion

V2 RC soak acceptance met for go/no-go review.
