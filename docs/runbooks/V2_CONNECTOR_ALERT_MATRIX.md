# V2 Connector Alert Matrix

## Purpose

Map each prioritized connector to its dashboard, alert rules, and escalation targets used for Phase 3 reliability governance.

## Matrix

| Connector ID | Dashboard | Alert Rules | Severity Routing | Escalation |
| --- | --- | --- | --- | --- |
| `sap-erp` | `deployment/monitoring/dashboards/erp-connector-service.json` | `deployment/monitoring/alerts/erp-connector-service-alerts.yaml` | Sev1 pager + Sev2 business-hours + Sev3 ticket | `#gl-vcci-oncall` -> SRE on-call board |
| `oracle-erp` | `deployment/monitoring/dashboards/erp-connector-service.json` | `deployment/monitoring/alerts/erp-connector-service-alerts.yaml` | Sev1 pager + Sev2 business-hours + Sev3 ticket | `#gl-vcci-oncall` -> SRE on-call board |
| `workday-erp` | `deployment/monitoring/dashboards/erp-connector-service.json` | `deployment/monitoring/alerts/erp-connector-service-alerts.yaml` | Sev1 pager + Sev2 business-hours + Sev3 ticket | `#gl-csrd-oncall` -> SRE on-call board |
| `azure-iot` | `deployment/monitoring/dashboards/observability-agent-service.json` | `deployment/monitoring/alerts/observability-agent-service-alerts.yaml` | Sev1 pager + Sev2 business-hours + Sev3 ticket | `#gl-eudr-oncall` -> SRE on-call board |

## Validation Checklist

- each connector has a named owner and on-call channel in `applications/connectors/v2_connector_registry.yaml`.
- each connector has a reliability profile validated via `gl v2 connector-checks`.
- all mapped alert rules are included in CI via `.github/workflows/greenlang-v2-platform-ci.yml` gate lanes.
