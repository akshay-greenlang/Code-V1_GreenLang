# GreenLang V2 Prioritized Connectors

This cohort is the Phase 3 reliability acceptance target. All connectors below must satisfy runtime profile validation, acceptance tests, and alert mapping.

| Connector ID | App Scope | Owner Team | On-call Channel | SLO Targets | Escalation |
| --- | --- | --- | --- | --- | --- |
| `sap-erp` | GL-VCCI-Carbon-APP | data-integration | #gl-vcci-oncall | availability >= 99.9%, p95 <= 3000ms, success >= 99% | SRE on-call board |
| `oracle-erp` | GL-VCCI-Carbon-APP | data-integration | #gl-vcci-oncall | availability >= 99.9%, p95 <= 3000ms, success >= 99% | SRE on-call board |
| `workday-erp` | GL-CSRD-APP | data-integration | #gl-csrd-oncall | availability >= 99.9%, p95 <= 3000ms, success >= 99% | SRE on-call board |
| `azure-iot` | GL-EUDR-APP | app-reliability | #gl-eudr-oncall | availability >= 99.9%, p95 <= 2000ms, success >= 99% | SRE on-call board |

## Required Reliability Evidence

- idempotency checks pass for write-like operations.
- retry and timeout budgets validated with transient failures.
- circuit-breaker open/half-open behavior validated.
- error classification metrics emitted for transient/auth/schema/permanent classes.
- dashboards and alert routes mapped in `docs/runbooks/V2_CONNECTOR_ALERT_MATRIX.md`.
