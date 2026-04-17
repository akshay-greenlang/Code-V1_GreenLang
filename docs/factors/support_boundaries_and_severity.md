# Support boundaries (P6) and issue severity (C6)

## Public (community / dev tier)

- Best-effort documentation and community channels.
- No contractual SLA; no guaranteed response time.
- Preview and connector-only factors may be hidden or rate-limited.

## Enterprise

- Contractual SLA (see commercial appendix in runbook).
- Severity matrix drives response and escalation.

## Severity matrix

| Level | Definition | Initial response | Escalation |
|-------|------------|------------------|------------|
| SEV1 | Wrong default certified factor affecting many customers / compliance | 1h | CTO + methodology |
| SEV2 | API outage or data corruption in certified slice | 4h | On-call engineer |
| SEV3 | Single-tenant connector or preview issue | 1 business day | Support |
| SEV4 | Documentation / non-blocking bug | Best effort | Backlog |

## Public vs enterprise data

- **Never** promise open redistribution for connector-only sources.
- Enterprise may receive **audit exports** and **source diff** views under contract.
