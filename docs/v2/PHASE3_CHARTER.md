# GreenLang V2 Phase 3 Charter

## Scope

Phase 3 formalizes and enforces:

- agent lifecycle states with ownership, deprecation, and retirement controls.
- connector reliability baseline for prioritized connectors (idempotency, retry, timeout budgets, circuit-breakers, error classification).
- SLO dashboard and alert mapping with named on-call ownership.

## Prioritized Connector Cohort

The connector cohort and reliability profiles are defined in:

- `applications/connectors/v2_connector_registry.yaml`

## Gate Owners

- Agent Lifecycle Gate Owner: Platform Runtime Team
- Connector Reliability Gate Owner: App Reliability Team
- SLO/Alerting Gate Owner: SRE On-call Board

## Exit Gate

Phase 3 is complete when all are true:

1. Agent lifecycle registry is validated in CLI and CI.
2. Runtime blocks retired agents and deprecated agents without replacement.
3. Prioritized connectors have reliability profiles validated in runtime and CI.
4. Behavioral reliability acceptance tests pass.
5. Connector-to-dashboard/alert/escalation mapping is published.
6. `docs/v2/PHASE3_GATE_STATUS.json` marks `phase3_complete: true` with evidence links.
