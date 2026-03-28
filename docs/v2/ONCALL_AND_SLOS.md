# GreenLang V2 On-Call and SLO Standards

## On-Call Coverage

- Runtime/CLI primary + secondary
- App reliability primary + secondary
- Security policy on-call
- Frontend shell on-call

## Prioritized Connector Ownership

| Connector ID | Owner Team | On-call Channel | Escalation Board |
| --- | --- | --- | --- |
| `sap-erp` | data-integration | #gl-vcci-oncall | SRE On-call Board |
| `oracle-erp` | data-integration | #gl-vcci-oncall | SRE On-call Board |
| `workday-erp` | data-integration | #gl-csrd-oncall | SRE On-call Board |
| `azure-iot` | app-reliability | #gl-eudr-oncall | SRE On-call Board |

## SLO Baselines

- API availability: >= 99.9%
- release-gate success rate: >= 99%
- p95 workflow latency by app profile (defined in app runbooks)
- incident MTTR target: <= 60 minutes for Severity 1

## Alert Policy

- Severity 1: page immediately.
- Severity 2: page business-hours on-call.
- Severity 3: ticket and triage in next workday.

## Escalation References

- connector alert matrix: `docs/runbooks/V2_CONNECTOR_ALERT_MATRIX.md`
- connector registry: `applications/connectors/v2_connector_registry.yaml`

## Error Budget Use

- if error budget burn exceeds threshold, feature work pauses for reliability hardening.
- release train promotion is blocked on unresolved Severity 1/2 incidents.
