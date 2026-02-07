# GreenLang Operational Runbooks

This directory contains operational runbooks for the GreenLang platform. Each runbook provides step-by-step guidance for diagnosing and resolving alerts related to metrics collection, storage, alerting, tracing, SLO management, and agent orchestration.

---

## Quick Navigation

### OBS-001: Prometheus Metrics Collection

| Category | Runbook | Alerts Covered |
|----------|---------|----------------|
| [Prometheus Memory](./prometheus-high-memory.md) | Memory management | PrometheusHighMemoryUsage, OOMKill |
| [Prometheus Targets](./prometheus-target-down.md) | Target health | PrometheusTargetMissing, ServiceDown |
| [Prometheus Queries](./prometheus-slow-queries.md) | Query performance | PrometheusSlowQueries, QueryTimeout |
| [Thanos Compactor](./thanos-compactor-halted.md) | Long-term storage | ThanosCompactorHalted, BlockOverlap |
| [Thanos Store Gateway](./thanos-store-gateway-issues.md) | Historical queries | ThanosStoreGatewayBucketOperationsFailed |
| [Alertmanager](./alertmanager-notifications-failing.md) | Notifications | AlertmanagerNotificationsFailing |
| [Batch Jobs](./batch-job-metrics-stale.md) | PushGateway jobs | BatchJobStale, PushgatewayDown |

### OBS-002: Grafana Dashboards

| Category | Runbook | Alerts Covered |
|----------|---------|----------------|
| [Grafana Down](./grafana-down.md) | Service availability | GrafanaDown |
| [Grafana High Memory](./grafana-high-memory.md) | Memory management | GrafanaHighMemory |
| [Grafana Dashboard Slow](./grafana-dashboard-slow.md) | Query performance | GrafanaDashboardSlow |

### OBS-003: OpenTelemetry Tracing

| Category | Runbook | Alerts Covered |
|----------|---------|----------------|
| [Tempo Ingester Failures](./tempo-ingester-failures.md) | Trace ingestion | TempoIngesterFailures |
| [OTel Collector Dropped Spans](./otel-collector-dropped-spans.md) | Span collection | OTelCollectorDroppedSpans |
| [Trace Search Slow](./trace-search-slow.md) | Query performance | TraceSearchSlow |

### OBS-004: Alerting & Notification Platform

| Category | Runbook | Alerts Covered |
|----------|---------|----------------|
| [Alerting Service Down](./alerting-service-down.md) | Service availability | AlertingServiceDown |
| [Notification Delivery Failing](./notification-delivery-failing.md) | Delivery health | NotificationDeliveryFailing |
| [PagerDuty Integration Down](./pagerduty-integration-down.md) | PagerDuty channel | PagerDutyIntegrationDown |

### OBS-005: SLO/SLI Definitions & Error Budget Management

| Category | Runbook | Alerts Covered |
|----------|---------|----------------|
| [SLO Service Down](./slo-service-down.md) | Service availability | SLOServiceDown, SLOEvaluationFailing |
| [Error Budget Exhausted](./error-budget-exhausted.md) | Budget management | ErrorBudgetExhausted, ErrorBudgetCritical |
| [High Burn Rate](./high-burn-rate.md) | Burn rate alerting | HighBurnRateFast, HighBurnRateMedium, HighBurnRateSlow |
| [SLO Compliance Degraded](./slo-compliance-degraded.md) | Compliance reporting | SLOComplianceBelow95 |

### AGENT-FOUND-001: GreenLang Orchestrator (DAG Execution Engine)

| Category | Runbook | Alerts Covered |
|----------|---------|----------------|
| [Orchestrator Service Down](./orchestrator-service-down.md) | Service availability | OrchestratorServiceDown |
| [DAG Execution Stuck](./dag-execution-stuck.md) | Execution timeout | OrchestratorExecutionTimeout |
| [Checkpoint Corruption](./checkpoint-corruption.md) | Data integrity | OrchestratorCheckpointFailure, OrchestratorProvenanceChainBroken |
| [High Execution Latency](./high-execution-latency.md) | Performance | OrchestratorHighLatency |

### AGENT-FOUND-002: Schema Compiler & Validator

| Category | Runbook | Alerts Covered |
|----------|---------|----------------|
| [Schema Service Down](./schema-service-down.md) | Service availability | SchemaServiceDown |
| [High Validation Errors](./high-validation-errors.md) | Validation quality | SchemaHighValidationErrorRate |
| [Schema Cache Corruption](./schema-cache-corruption.md) | Cache integrity | SchemaCacheHitRateLow, SchemaCacheCompilationAfterHit, SchemaIRHashMismatch |
| [Compilation Timeout](./compilation-timeout.md) | Compilation failures | SchemaCompilationFailure, SchemaCompilationTimeout, SchemaReDoSDetected |

### AGENT-FOUND-003: Unit & Reference Normalizer

| Category | Runbook | Alerts Covered |
|----------|---------|----------------|
| [Normalizer Service Down](./normalizer-service-down.md) | Service availability | NormalizerServiceDown |
| [Conversion Accuracy Drift](./conversion-accuracy-drift.md) | Conversion correctness | NormalizerConversionAccuracyDrift, NormalizerGWPVersionMismatch, NormalizerProvenanceChainBroken |
| [Entity Resolution Low Confidence](./entity-resolution-low-confidence.md) | Entity resolution quality | NormalizerEntityResolutionLowConfidence, NormalizerHighUnresolvedRate, NormalizerReviewQueueBacklog |
| [Normalizer High Latency](./normalizer-high-latency.md) | Performance | NormalizerHighLatency, NormalizerBatchTimeout, NormalizerQueueBacklog |

### AGENT-FOUND-004: Assumptions Registry Service

| Category | Runbook | Alerts Covered |
|----------|---------|----------------|
| [Assumptions Service Down](./assumptions-service-down.md) | Service availability | AssumptionsServiceDown |
| [Assumption Validation Failures](./assumption-validation-failures.md) | Validation quality | AssumptionsValidationFailureSpike, AssumptionsValidationRuleError, AssumptionsValidationBypassed |
| [Scenario Drift Detection](./scenario-drift-detection.md) | Scenario consistency | AssumptionsScenarioDrift, AssumptionsStaleScenario, AssumptionsSensitivityAnomaly |
| [Assumptions Audit Compliance](./assumptions-audit-compliance.md) | Audit integrity | AssumptionsAuditGap, AssumptionsProvenanceChainBroken, AssumptionsExportIntegrityFailure, AssumptionsAuditRetentionRisk |

---

## Alert Severity Reference

| Severity | Response Time | Notification | Example Alerts |
|----------|---------------|--------------|----------------|
| **Critical** | Immediate (<5 min) | PagerDuty page + Slack | PrometheusTargetMissing, ThanosCompactorHalted, HighBurnRateFast, ErrorBudgetExhausted, OrchestratorServiceDown, OrchestratorCheckpointFailure, SchemaServiceDown, SchemaCacheCompilationAfterHit, SchemaIRHashMismatch, SchemaReDoSDetected, NormalizerServiceDown, NormalizerGWPVersionMismatch, NormalizerProvenanceChainBroken, AssumptionsServiceDown, AssumptionsValidationBypassed, AssumptionsAuditGap, AssumptionsProvenanceChainBroken, AssumptionsExportIntegrityFailure |
| **Warning** | Within 30 min | Slack notification | PrometheusHighMemoryUsage, BatchJobStale, HighBurnRateMedium, SLOComplianceBelow95, OrchestratorExecutionTimeout, OrchestratorHighLatency, SchemaHighValidationErrorRate, SchemaCompilationFailure, SchemaCompilationTimeout, SchemaCacheHitRateLow, SchemaHighLatency, NormalizerConversionAccuracyDrift, NormalizerEntityResolutionLowConfidence, NormalizerHighUnresolvedRate, NormalizerHighLatency, NormalizerBatchTimeout, NormalizerQueueBacklog, NormalizerReviewQueueBacklog, NormalizerHighErrorRate, AssumptionsValidationFailureSpike, AssumptionsValidationRuleError, AssumptionsScenarioDrift, AssumptionsStaleScenario, AssumptionsSensitivityAnomaly, AssumptionsHighLatency, AssumptionsHighErrorRate, AssumptionsAuditRetentionRisk |
| **Info** | Next business day | Email digest | PrometheusConfigReloaded, HighBurnRateSlow |

---

## Alert Summary by Component

### Prometheus Server Alerts

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [PrometheusHighMemoryUsage](./prometheus-high-memory.md) | Warning | >80% memory | Check cardinality, add recording rules |
| [PrometheusTargetMissing](./prometheus-target-down.md) | Critical | up == 0 for 5m | Check pod, network policy, ServiceMonitor |
| [PrometheusSlowQueries](./prometheus-slow-queries.md) | Warning | P99 >30s | Add recording rules, increase resources |
| PrometheusConfigReloadFailed | Critical | reload != 1 | Check config syntax, restart |
| PrometheusStorageAlmostFull | Warning | >80% storage | Reduce retention, expand PVC |
| PrometheusTSDBCompactionsFailed | Warning | failures > 0 | Check disk I/O, restart |

### Thanos Alerts

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [ThanosCompactorHalted](./thanos-compactor-halted.md) | Critical | halted == 1 | Check overlaps, S3 access, disk |
| [ThanosStoreGatewayBucketOperationsFailed](./thanos-store-gateway-issues.md) | Warning | failures > 0 | Check IRSA, S3 permissions |
| ThanosQueryHighDNSFailures | Warning | failures > 0.5/s | Check service discovery |
| ThanosSidecarPrometheusDown | Critical | prometheus_up != 1 | Check Prometheus, sidecar |
| ThanosCompactorMultipleRunning | Critical | compactors > 1 | Scale down to 1 |

### Alertmanager Alerts

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [AlertmanagerNotificationsFailing](./alertmanager-notifications-failing.md) | Critical | failures > 0 | Check webhooks, credentials |
| AlertmanagerConfigInconsistent | Warning | config mismatch | Sync config, restart |
| AlertmanagerClusterDown | Critical | members < expected | Check mesh, network |
| AlertmanagerMembersInconsistent | Warning | peer count varies | Check gossip ports |

### PushGateway Alerts

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [BatchJobStale](./batch-job-metrics-stale.md) | Warning | >1h since success | Check CronJob, push failures |
| PushGatewayDown | Critical | up == 0 | Restart deployment |
| BatchJobDurationHigh | Warning | duration > threshold | Investigate job performance |
| BatchJobErrorsHigh | Warning | errors increasing | Check job logs |

### SLO Service Alerts (OBS-005)

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [SLOServiceDown](./slo-service-down.md) | Critical | up == 0 for 5m | Check pods, Prometheus/DB/Redis connectivity |
| SLOEvaluationFailing | Critical | errors > 10% for 10m | Check Prometheus queries, SLO definitions |
| [ErrorBudgetExhausted](./error-budget-exhausted.md) | Critical | budget <= 0% | Deployment freeze, root cause analysis |
| ErrorBudgetCritical | Warning | budget < 50% for 1h | Investigate SLI degradation, plan remediation |
| ErrorBudgetWarning | Info | budget < 80% for 6h | Review burn rate, plan reliability work |
| [HighBurnRateFast](./high-burn-rate.md) | Critical | 14.4x for 5m (1h/5m windows) | Immediate: rollback, scale, fix root cause |
| [HighBurnRateMedium](./high-burn-rate.md) | Warning | 6.0x for 30m (6h/30m windows) | Investigate sustained degradation |
| HighBurnRateSlow | Info | 1.0x for 6h (3d/6h windows) | Plan reliability improvements |
| [SLOComplianceBelow95](./slo-compliance-degraded.md) | Warning | compliance < 95% for 1h | Review failing SLOs, prioritize fixes |
| RecordingRuleGenerationFailed | Warning | generation error | Check SLO definitions, Prometheus config |
| BudgetSnapshotStale | Warning | no snapshots for 15m | Check SLO service, TimescaleDB connectivity |

### Orchestrator Alerts (AGENT-FOUND-001)

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [OrchestratorServiceDown](./orchestrator-service-down.md) | Critical | up == 0 for 5m | Check pods, DB/Redis connectivity, init containers |
| OrchestratorHighFailureRate | Warning | >10% failures for 15m | Check agent health, review execution errors |
| [OrchestratorHighLatency](./high-execution-latency.md) | Warning | p99 >30s for 10m | Identify bottleneck nodes, reduce concurrency, scale agents |
| [OrchestratorExecutionTimeout](./dag-execution-stuck.md) | Warning | execution > timeout | Cancel/resume execution, check stuck agent, release locks |
| [OrchestratorCheckpointFailure](./checkpoint-corruption.md) | Critical | checkpoint errors > 0 for 5m | Check storage health, verify DB connectivity, fix corruption |
| [OrchestratorProvenanceChainBroken](./checkpoint-corruption.md) | Critical | chain_valid == 0 | Stop new executions, identify affected runs, re-execute |

### Schema Service Alerts (AGENT-FOUND-002)

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [SchemaServiceDown](./schema-service-down.md) | Critical | up == 0 for 5m | Check pods, Redis/Git connectivity, container errors |
| [SchemaHighValidationErrorRate](./high-validation-errors.md) | Warning | >25% errors for 5m | Check error distribution by code, recent schema changes |
| SchemaHighLatency | Warning | P95 > SLA for 10m | Check cache hit rate, compilation load, resource usage |
| [SchemaCompilationFailure](./compilation-timeout.md) | Warning | errors > 0 for 5m | Identify failing schema, check regex patterns, check $ref depth |
| [SchemaCompilationTimeout](./compilation-timeout.md) | Warning | P99 > 5s | Simplify schema, fix regex patterns, increase limits |
| [SchemaReDoSDetected](./compilation-timeout.md) | Critical | ReDoS pattern found | Fix regex pattern, review schema, block deployment |
| [SchemaCacheHitRateLow](./schema-cache-corruption.md) | Warning | hit rate < 50% for 10m | Check cache config, Redis connectivity, warm-up scheduler |
| [SchemaCacheCompilationAfterHit](./schema-cache-corruption.md) | Critical | hash mismatch after hit | Flush cache, verify registry integrity, restart |
| [SchemaIRHashMismatch](./schema-cache-corruption.md) | Critical | IR hash != source hash | Flush cache, invalidate affected schemas, restart |

### Normalizer Service Alerts (AGENT-FOUND-003)

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [NormalizerServiceDown](./normalizer-service-down.md) | Critical | up == 0 for 5m | Check pods, DB/Redis connectivity, vocabulary loading |
| NormalizerHighErrorRate | Warning | >10% errors for 5m | Check conversion logs, unit alias conflicts, DB connectivity |
| [NormalizerHighLatency](./normalizer-high-latency.md) | Warning | P95 > 100ms for 10m | Check cache hit rate, batch sizes, DB pool, scale pods |
| [NormalizerBatchTimeout](./normalizer-high-latency.md) | Warning | batch timeout > 0 | Reduce batch size, increase timeout, scale pods |
| [NormalizerQueueBacklog](./normalizer-high-latency.md) | Warning | queue > 1000 for 5m | Scale pods, reduce batch size, check downstream |
| [NormalizerConversionAccuracyDrift](./conversion-accuracy-drift.md) | Warning | factor validation failures > 0 | Check GWP version, audit custom factors, verify reference sources |
| [NormalizerGWPVersionMismatch](./conversion-accuracy-drift.md) | Critical | multiple GWP versions active | Pin GWP version, restart services, reconvert affected data |
| [NormalizerProvenanceChainBroken](./conversion-accuracy-drift.md) | Critical | provenance hash failures > 0 | Check serialization, verify data integrity, recompute hashes |
| [NormalizerEntityResolutionLowConfidence](./entity-resolution-low-confidence.md) | Warning | avg confidence < 0.7 for 10m | Check vocabulary completeness, add aliases, adjust thresholds |
| [NormalizerHighUnresolvedRate](./entity-resolution-low-confidence.md) | Warning | >20% unresolved for 10m | Add vocabulary entries, onboard new data source aliases |
| [NormalizerReviewQueueBacklog](./entity-resolution-low-confidence.md) | Warning | queue > 500 for 30m | Batch review unresolved items, add vocabulary entries |

### Assumptions Service Alerts (AGENT-FOUND-004)

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [AssumptionsServiceDown](./assumptions-service-down.md) | Critical | up == 0 for 5m | Check pods, DB connectivity, init containers, ConfigMap |
| AssumptionsHighErrorRate | Warning | >10% errors for 5m | Check assumption operation logs, DB connectivity |
| AssumptionsHighLatency | Warning | P95 > 15ms for 10m | Check DB pool, dependency graph depth, scale pods |
| [AssumptionsValidationFailureSpike](./assumption-validation-failures.md) | Warning | >10% validation failures for 10m | Check validation rules, data quality, custom validators |
| [AssumptionsValidationRuleError](./assumption-validation-failures.md) | Warning | validator execution errors > 0 | Fix or disable failing custom validator |
| [AssumptionsValidationBypassed](./assumption-validation-failures.md) | Critical | validation bypass in strict mode | Check config, investigate potential security issue |
| [AssumptionsScenarioDrift](./scenario-drift-detection.md) | Warning | >30% deviation from baseline for 15m | Review overrides, update stale values, check baseline changes |
| [AssumptionsStaleScenario](./scenario-drift-detection.md) | Warning | active scenario not updated >90 days | Review or deactivate stale scenario |
| [AssumptionsSensitivityAnomaly](./scenario-drift-detection.md) | Warning | sensitivity variance >50% | Run sensitivity analysis, review high-impact assumptions |
| [AssumptionsAuditGap](./assumptions-audit-compliance.md) | Critical | missing audit entries with active writes | Repair audit gaps from version history, check async pipeline |
| [AssumptionsProvenanceChainBroken](./assumptions-audit-compliance.md) | Critical | provenance hash verification failure | Check for direct DB modifications, rebuild provenance chain |
| [AssumptionsExportIntegrityFailure](./assumptions-audit-compliance.md) | Critical | export hash verification failure | Regenerate export, check signing key, verify DB integrity |
| [AssumptionsAuditRetentionRisk](./assumptions-audit-compliance.md) | Warning | records approaching retention limit | Archive old records to S3, verify archival job |

---

## On-Call Quick Reference

### First Response Checklist

1. **Acknowledge** the alert in PagerDuty
2. **Identify** the affected component and runbook
3. **Diagnose** using the runbook diagnostic steps
4. **Mitigate** with quick actions
5. **Communicate** status in #greenlang-incidents
6. **Resolve** or escalate per runbook
7. **Document** actions in incident timeline

### Common Commands

```bash
# Check Prometheus stack health
kubectl get pods -n monitoring

# Check Prometheus targets
kubectl port-forward -n monitoring svc/prometheus-server 9090:9090
# Open http://localhost:9090/targets

# Check Alertmanager
kubectl port-forward -n monitoring svc/alertmanager 9093:9093
# Open http://localhost:9093

# Check Thanos Query
kubectl port-forward -n monitoring svc/thanos-query 9091:9090
# Open http://localhost:9091

# Restart components
kubectl rollout restart statefulset -n monitoring prometheus-server
kubectl rollout restart statefulset -n monitoring alertmanager
kubectl rollout restart deployment -n monitoring thanos-query

# Check logs
kubectl logs -n monitoring -l app.kubernetes.io/name=prometheus --tail=100
kubectl logs -n monitoring -l app.kubernetes.io/name=alertmanager --tail=100
kubectl logs -n monitoring -l app.kubernetes.io/name=thanos-compactor --tail=100

# Check SLO service health
kubectl get pods -n greenlang-slo -l app=slo-service
kubectl logs -n greenlang-slo -l app=slo-service --tail=100

# Check SLO service API
kubectl port-forward -n greenlang-slo svc/slo-service 8080:8080
# curl http://localhost:8080/api/v1/slos/health
# curl http://localhost:8080/api/v1/slos/overview

# Check orchestrator service health
kubectl get pods -n greenlang -l app=orchestrator-service
kubectl logs -n greenlang -l app=orchestrator-service --tail=100

# Check orchestrator API
kubectl port-forward -n greenlang svc/orchestrator-service 8080:8080
# curl http://localhost:8080/api/v1/orchestrator/health
# curl http://localhost:8080/api/v1/orchestrator/executions?status=running

# Check active DAG executions
# curl http://localhost:8080/api/v1/orchestrator/metrics

# Restart orchestrator
kubectl rollout restart deployment/orchestrator-service -n greenlang

# Check schema service health
kubectl get pods -n greenlang -l app=schema-service
kubectl logs -n greenlang -l app=schema-service --tail=100

# Check schema service API
kubectl port-forward -n greenlang svc/schema-service 8080:8080
# curl http://localhost:8080/health
# curl http://localhost:8080/metrics

# Check schema validation metrics
# curl http://localhost:8080/metrics/prometheus

# Restart schema service
kubectl rollout restart deployment/schema-service -n greenlang

# Check normalizer service health
kubectl get pods -n greenlang -l app=normalizer-service
kubectl logs -n greenlang -l app=normalizer-service --tail=100

# Check normalizer service API
kubectl port-forward -n greenlang svc/normalizer-service 8080:8080
# curl http://localhost:8080/health
# curl http://localhost:8080/metrics

# Test a normalizer conversion
# curl -X POST http://localhost:8080/v1/normalizer/convert \
#   -H "Content-Type: application/json" \
#   -d '{"value": 1000, "from_unit": "kg", "to_unit": "tonnes"}'

# Check normalizer vocabulary cache status
# curl http://localhost:8080/v1/normalizer/admin/cache/status

# Restart normalizer service
kubectl rollout restart deployment/normalizer-service -n greenlang

# Check assumptions service health
kubectl get pods -n greenlang -l app=assumptions-service
kubectl logs -n greenlang -l app=assumptions-service --tail=100

# Check assumptions service API
kubectl port-forward -n greenlang svc/assumptions-service 8080:8080
# curl http://localhost:8080/health
# curl http://localhost:8080/metrics

# Test assumption query
# curl http://localhost:8080/v1/assumptions?limit=5 \
#   -H "Authorization: Bearer $ACCESS_TOKEN"

# Check scenarios
# curl http://localhost:8080/v1/assumptions/scenarios \
#   -H "Authorization: Bearer $ACCESS_TOKEN"

# Run provenance verification
# curl -X POST http://localhost:8080/v1/assumptions/admin/verify-provenance \
#   -H "Authorization: Bearer $ADMIN_TOKEN"

# Restart assumptions service
kubectl rollout restart deployment/assumptions-service -n greenlang
```

### Key Dashboards

| Dashboard | URL | Purpose |
|-----------|-----|---------|
| Prometheus Health | https://grafana.greenlang.io/d/prometheus-health | Prometheus server metrics |
| Thanos Overview | https://grafana.greenlang.io/d/thanos-overview | Thanos component health |
| Alertmanager | https://grafana.greenlang.io/d/alertmanager | Alert delivery status |
| Targets | https://grafana.greenlang.io/d/prometheus-targets | Scrape target status |
| Cardinality | https://grafana.greenlang.io/d/cardinality-explorer | Metric cardinality analysis |
| SLO Overview | https://grafana.greenlang.io/d/slo-overview | SLO compliance and error budgets |
| SLO Error Budget | https://grafana.greenlang.io/d/slo-error-budget | Error budget deep dive |
| SLO Burn Rate | https://grafana.greenlang.io/d/slo-burn-rate | Multi-window burn rate visualization |
| SLO Service Health | https://grafana.greenlang.io/d/slo-service-health | SLO service self-monitoring |
| Orchestrator Health | https://grafana.greenlang.io/d/orchestrator-health | Orchestrator service metrics |
| DAG Execution Overview | https://grafana.greenlang.io/d/dag-execution-overview | DAG execution status and latency |
| Node Performance | https://grafana.greenlang.io/d/node-performance | Per-node execution latency breakdown |
| Schema Service Health | https://grafana.greenlang.io/d/schema-service-health | Schema service self-monitoring |
| Schema Validation Overview | https://grafana.greenlang.io/d/schema-validation-overview | Validation throughput, error rates, latency |
| Normalizer Service Health | https://grafana.greenlang.io/d/normalizer-service-health | Normalizer service self-monitoring |
| Unit Conversion Overview | https://grafana.greenlang.io/d/normalizer-conversion-overview | Conversion throughput, accuracy, entity resolution |
| Assumptions Registry Health | https://grafana.greenlang.io/d/assumptions-service-health | Assumptions service self-monitoring |
| Assumptions Operations Overview | https://grafana.greenlang.io/d/assumptions-operations-overview | Assumption operations, validation, scenarios, audit |

### Escalation Contacts

| Team | Slack Channel | PagerDuty Schedule |
|------|---------------|-------------------|
| Platform | #platform-oncall | platform-oncall |
| Observability | #observability | observability-oncall |
| Infrastructure | #infrastructure-oncall | infrastructure-oncall |

---

## Component Architecture

```
                                    +------------------+
                                    |    Grafana       |
                                    +--------+---------+
                                             |
                            +----------------+----------------+
                            |                                 |
                   +--------v--------+              +---------v---------+
                   |  Thanos Query   |              |   Alertmanager    |
                   +--------+--------+              +---------+---------+
                            |                                 |
        +-------------------+-------------------+             |
        |                   |                   |             |
+-------v-------+  +--------v--------+  +-------v-------+    |
| Thanos Sidecar|  | Thanos Store GW |  | Thanos Ruler  +----+
+-------+-------+  +--------+--------+  +---------------+
        |                   |
+-------v-------+           |
|  Prometheus   |           |
+-------+-------+  +--------v--------+
        |          |   S3 Bucket     |
        |          +--------+--------+
        |                   |
+-------v-------+  +--------v--------+
|  PushGateway  |  | Thanos Compactor|
+---------------+  +-----------------+
```

---

## Incident Severity Levels

| Level | Description | Response | Example |
|-------|-------------|----------|---------|
| P1 | Complete monitoring outage | Immediate, all hands | Prometheus down, no alerts |
| P2 | Major degradation | Within 15 minutes | All notifications failing |
| P3 | Minor issue | Within 1 hour | Single target down |
| P4 | Low priority | Next business day | Slow historical queries |

---

## Maintenance Windows

Regular maintenance windows for the Prometheus stack:

| Day | Time (UTC) | Duration | Purpose |
|-----|------------|----------|---------|
| Sunday | 02:00-04:00 | 2 hours | Helm upgrades, PVC expansion |
| Daily | 03:00-03:30 | 30 min | Thanos compaction (automated) |

During maintenance windows:
- Silence non-critical alerts
- Notify #greenlang-ops before starting
- Have rollback plan ready

---

## Contributing to Runbooks

### When to Update

- After resolving an incident not covered by existing runbooks
- When discovering new diagnostic techniques
- When resolution steps change due to architecture changes
- After post-incident reviews with new learnings

### Runbook Structure

Each runbook section should include:

1. **Alert Details** - Name, severity, threshold, PromQL
2. **Description** - What the alert means
3. **Impact Assessment** - User, data, SLA, revenue impact
4. **Diagnostic Steps** - How to identify the problem
5. **Resolution Steps** - Multiple scenarios with commands
6. **Emergency Actions** - Immediate mitigation
7. **Escalation Path** - When and who to escalate to
8. **Prevention** - How to avoid in the future
9. **Related Dashboards** - Links to monitoring
10. **Related Alerts** - Other relevant alerts

### Style Guidelines

- Use clear, actionable language
- Include actual commands that can be copy-pasted
- Explain what each command does
- Provide multiple resolution scenarios
- Keep commands idempotent where possible
- Test all commands before documenting

---

## Alert Configuration

Alert rules are defined in:

```
deployment/kubernetes/monitoring/prometheus-rules/
  - prometheus-health-alerts.yaml
  - thanos-health-alerts.yaml
  - alertmanager-health-alerts.yaml
  - pushgateway-alerts.yaml

deployment/monitoring/alerts/
  - slo-service-alerts.yaml              # SLO service self-monitoring alerts
  - slo-burn-rate-alerts.yaml            # Auto-generated multi-window burn rate alerts

deployment/kubernetes/orchestrator/
  - orchestrator-alerts.yaml             # Orchestrator service alerts (AGENT-FOUND-001)

deployment/kubernetes/schema-service/
  - schema-service-alerts.yaml           # Schema service alerts (AGENT-FOUND-002)

deployment/kubernetes/normalizer-service/
  - normalizer-service-alerts.yaml       # Normalizer service alerts (AGENT-FOUND-003)

deployment/kubernetes/assumptions-service/
  - assumptions-service-alerts.yaml      # Assumptions service alerts (AGENT-FOUND-004)
```

To modify alert thresholds or add new alerts:

```bash
# Edit alert rules
kubectl edit prometheusrule -n monitoring prometheus-health-alerts

# Or apply from file
kubectl apply -f deployment/kubernetes/monitoring/prometheus-rules/
```

---

## Related Documentation

- [Prometheus Stack Architecture](../architecture/prometheus-stack.md)
- [Prometheus Operations Guide](../operations/prometheus-operations.md)
- [Metrics Developer Guide](../development/metrics-guide.md)
- [GreenLang Monitoring Setup](../monitoring/README.md)
- [Incident Response Procedure](../operations/incident-response.md)
- [AGENT-FOUND-001 PRD: GreenLang Orchestrator](../../GreenLang%20Development/05-Documentation/PRD-AGENT-FOUND-001-GreenLang-Orchestrator.md)
- [AGENT-FOUND-002 PRD: Schema Compiler & Validator](../../GreenLang%20Development/05-Documentation/PRD-AGENT-FOUND-002-Schema-Compiler-Validator.md)
- [AGENT-FOUND-003 PRD: Unit & Reference Normalizer](../../GreenLang%20Development/05-Documentation/PRD-AGENT-FOUND-003-Unit-Reference-Normalizer.md)
- [AGENT-FOUND-004 PRD: Assumptions Registry Service](../../GreenLang%20Development/05-Documentation/PRD-AGENT-FOUND-004-Assumptions-Registry.md)

---

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Thanos Documentation](https://thanos.io/tip/thanos/getting-started.md/)
- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)
