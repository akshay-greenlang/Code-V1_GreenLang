# ERP Connector Service Down

## Alert

**Alert Name:** `ERPConnectorServiceDown`

**Severity:** Critical

**Threshold:** `up{job="erp-connector-service"} == 0` for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when no instances of the GreenLang ERP/Finance Connector (AGENT-DATA-003) are running. The ERP Connector is the enterprise data integration service for all GreenLang Climate OS procurement-based data sources. It is responsible for:

1. **ERP system connectivity** -- Managing connections to enterprise ERP systems (SAP S/4HANA, SAP ECC, Oracle ERP Cloud, Oracle EBS, Microsoft Dynamics 365, NetSuite, Workday, Sage Intacct, Xero, QuickBooks) with health monitoring and automatic reconnection
2. **Spend data synchronization** -- Extracting spend/procurement transaction records with full/incremental/delta sync modes, currency conversion to USD, and Scope 3 category classification
3. **Purchase order synchronization** -- Extracting PO headers and line items with status tracking, vendor mapping, and Scope 3 classification
4. **Vendor mapping** -- Mapping ERP vendor master data to GreenLang spend categories and assigning EEIO emission factors (kgCO2e/USD) for spend-based Scope 3 calculations
5. **Material mapping** -- Mapping ERP material master data to GreenLang material groups and assigning activity-based emission factors (kgCO2e/unit) for procurement emissions
6. **Scope 3 emission calculation** -- Calculating emissions from spend records using vendor or material emission factors with spend-based, activity-based, or hybrid methodology
7. **Inventory synchronization** -- Extracting inventory position snapshots with material quantities, costs, and warehouse locations
8. **Currency conversion** -- Converting spend amounts from source currencies to USD using configurable exchange rate providers
9. **Provenance hash chains** -- Maintaining SHA-256 hash chains across all synced data and audit events for tamper detection and compliance
10. **Emitting Prometheus metrics** (12+ metrics under the `gl_erp_connector_*` prefix) for monitoring sync rates, connection health, emission calculations, and service health

When the ERP Connector is down:
- **Spend data ingestion stops** and no new procurement transactions can be synced from ERP systems
- **Scope 3 emission calculations are interrupted** from procurement data sources (Categories 1-15)
- **Sync job queue will grow** and scheduled syncs will accumulate without processing
- **Vendor/material mapping is unavailable** and new vendors cannot be classified
- **Audit trail has a gap** and compliance requirements for traceable data ingestion are violated

**Note:** All connections, vendor/material mappings, spend records, purchase orders, inventory snapshots, sync jobs, emission calculations, and audit events are stored in PostgreSQL with TimescaleDB and are not affected by a service outage. Once the service recovers, the full state will be immediately available. Pending sync jobs will need to be reprocessed.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | No new spend data can be synced or emission calculations updated |
| **Data Impact** | High | Procurement-based Scope 3 emission data feed interrupted; sync queue growing |
| **SLA Impact** | High | ERP sync processing SLA violated (all sync jobs fail) |
| **Revenue Impact** | Medium | Enterprise customers require timely procurement data sync for reporting |
| **Compliance Impact** | High | CSRD, GHG Protocol require traceable, verifiable Scope 3 procurement data |
| **Downstream Impact** | High | Emission calculation agents waiting for synced spend/PO data |

---

## Symptoms

- `up{job="erp-connector-service"}` metric returns 0 or is absent
- No pods running in the `greenlang` namespace with label `app=erp-connector-service`
- `gl_erp_connector_sync_jobs_total` counter stops incrementing
- `gl_erp_connector_spend_records_total` counter stops incrementing
- `gl_erp_connector_active_connections` drops to 0
- REST API returns 503 Service Unavailable or connection refused
- Health endpoint `GET /health` is unreachable
- Sync job endpoints return errors
- Grafana ERP Connector dashboard shows "No Data" or stale timestamps
- Sync job queue backlog grows without being processed

---

## Diagnostic Steps

### Step 1: Check Pod Status

```bash
# List ERP connector service pods
kubectl get pods -n greenlang -l app=erp-connector-service

# Check for failed pods, CrashLoopBackOff, or ImagePullBackOff
kubectl get pods -n greenlang -l app=erp-connector-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].state}{"\n"}{end}'

# Check events for the namespace related to ERP connector service
kubectl get events -n greenlang --sort-by='.lastTimestamp' \
  --field-selector involvedObject.name=erp-connector-service | tail -30

# Check deployment status
kubectl describe deployment erp-connector-service -n greenlang
```

### Step 2: Check Deployment and ReplicaSet

```bash
# Verify deployment desired vs available replicas (expect 2 minimum)
kubectl get deployment erp-connector-service -n greenlang

# Check ReplicaSet status
kubectl get replicaset -n greenlang -l app=erp-connector-service

# Check for rollout issues
kubectl rollout status deployment/erp-connector-service -n greenlang

# Check HPA status (scales 2-8 replicas)
kubectl get hpa -n greenlang -l app=erp-connector-service
```

### Step 3: Check Container Logs

```bash
# Get logs from the most recent pod (even if crashed)
kubectl logs -n greenlang -l app=erp-connector-service --tail=200

# Get logs from a specific crashed pod (previous container)
kubectl logs -n greenlang <pod-name> --previous --tail=200

# Look for specific error patterns
kubectl logs -n greenlang -l app=erp-connector-service --tail=500 \
  | grep -i "error\|fatal\|panic\|fail\|connection refused"

# Look for ERP-specific errors
kubectl logs -n greenlang -l app=erp-connector-service --tail=500 \
  | grep -i "erp\|sap\|oracle\|dynamics\|netsuite\|sync\|spend\|emission"

# Look for database connection errors
kubectl logs -n greenlang -l app=erp-connector-service --tail=500 \
  | grep -i "database\|postgres\|timescale\|connection\|pool\|migration"
```

### Step 4: Check Resource Usage

```bash
# Check current CPU and memory usage of ERP connector service pods
kubectl top pods -n greenlang -l app=erp-connector-service

# Check if pods were OOMKilled (batch sync processing is memory-intensive)
kubectl get pods -n greenlang -l app=erp-connector-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\t"}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}'

# Check node resource availability
kubectl top nodes
```

### Step 5: Check Database Connectivity

```bash
# Verify PostgreSQL connectivity
kubectl run pg-test --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  pg_isready -h greenlang-db.postgres.svc.cluster.local -p 5432

# Check if the erp_connector_service schema exists
kubectl run pg-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT table_name FROM information_schema.tables
   WHERE table_schema='erp_connector_service'
   ORDER BY table_name;"
```

### Step 6: Check ConfigMap and Secrets

```bash
# Verify the ERP connector service ConfigMap exists and is valid
kubectl get configmap erp-connector-service-config -n greenlang
kubectl get configmap erp-connector-service-config -n greenlang -o yaml | head -50

# Verify secrets exist (including ERP system credentials)
kubectl get secret erp-connector-service-secrets -n greenlang

# Check environment variables are set correctly
kubectl get deployment erp-connector-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].env[*].name}' | tr ' ' '\n' | sort
```

### Step 7: Check Network Policies

```bash
# Check network policies affecting the ERP connector service
kubectl get networkpolicy -n greenlang | grep erp-connector

# Verify the ERP connector service can reach PostgreSQL
kubectl run net-test --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv greenlang-db.postgres.svc.cluster.local 5432'

# Verify egress to ERP systems (HTTPS)
kubectl run net-test-erp --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv 0.0.0.0 443'
```

---

## Resolution Steps

### Scenario 1: OOMKilled (Out of Memory)

**Symptoms:** Pod status shows OOMKilled, container exits with code 137. Common with large batch sync operations or high concurrency.

**Resolution:**

1. Confirm the OOM cause:
```bash
kubectl describe pod -n greenlang <pod-name> | grep -A10 "Last State"
kubectl get events -n greenlang --field-selector reason=OOMKilling --sort-by='.lastTimestamp'
```

2. Increase memory limits (batch sync processing requires more memory for large datasets):
```bash
kubectl patch deployment erp-connector-service -n greenlang -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "erp-connector-service",
            "resources": {
              "limits": {
                "cpu": "2",
                "memory": "2Gi"
              },
              "requests": {
                "cpu": "500m",
                "memory": "1Gi"
              }
            }
          }
        ]
      }
    }
  }
}'
```

3. Verify pods restart successfully:
```bash
kubectl rollout status deployment/erp-connector-service -n greenlang
kubectl get pods -n greenlang -l app=erp-connector-service
```

### Scenario 2: CrashLoopBackOff -- Database Migration Failure

**Symptoms:** Pod status shows CrashLoopBackOff, init container logs show migration errors.

**Resolution:**

1. Check init container logs:
```bash
kubectl logs -n greenlang <pod-name> -c check-db-migration --tail=100
```

2. Verify database schema:
```bash
kubectl run pg-migration --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT version, description, success FROM flyway_schema_history
   ORDER BY installed_rank DESC LIMIT 5;"
```

3. Restart the deployment after fixing:
```bash
kubectl rollout restart deployment/erp-connector-service -n greenlang
kubectl rollout status deployment/erp-connector-service -n greenlang
```

### Scenario 3: Code Bug -- Rollback Deployment

**Symptoms:** Service was working before a recent deployment, logs show new error patterns.

**Resolution:**

1. Check recent deployment history:
```bash
kubectl rollout history deployment/erp-connector-service -n greenlang
```

2. Rollback to the previous version:
```bash
kubectl rollout undo deployment/erp-connector-service -n greenlang
kubectl rollout status deployment/erp-connector-service -n greenlang
```

3. Verify the rollback resolved the issue:
```bash
kubectl get pods -n greenlang -l app=erp-connector-service
kubectl port-forward -n greenlang svc/erp-connector-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

---

## Post-Incident Steps

### Step 1: Verify Service Health

```bash
# Check all pods are running and ready
kubectl get pods -n greenlang -l app=erp-connector-service

# Check the health endpoint
kubectl port-forward -n greenlang svc/erp-connector-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

### Step 2: Verify Prometheus Metrics Are Flowing

```promql
# Verify the ERP connector service is being scraped
up{job="erp-connector-service"} == 1

# Verify sync job count metric is populated
gl_erp_connector_sync_jobs_total > 0

# Verify spend record metrics are incrementing
increase(gl_erp_connector_spend_records_total[5m])
```

### Step 3: Reprocess Pending Sync Jobs

```bash
# Check for stuck or failed sync jobs during the outage
curl -s "http://localhost:8080/v1/sync-jobs?status=pending&limit=20" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Reprocess failed sync jobs
curl -s -X POST "http://localhost:8080/v1/sync-jobs/reprocess" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "failed", "since": "2024-01-01T00:00:00Z"}' \
  | python3 -m json.tool
```

---

## Interim Mitigation

While the ERP Connector is being restored:

1. **Synced data is safe.** All connections, vendor/material mappings, spend records, purchase orders, inventory snapshots, sync jobs, emission calculations, and audit events are stored in PostgreSQL with TimescaleDB. The database persists independently.

2. **Sync jobs will queue up.** Scheduled syncs will not execute. Once the service recovers, the sync queue will need to be drained.

3. **Scope 3 emission calculations are delayed.** Procurement-based emission data will not be updated until spend records are synced and calculations run.

4. **Manual data entry may be needed.** For time-critical emission reports, manual entry of procurement spend data can bypass the ERP sync pipeline temporarily.

5. **Communicate to affected teams.** Notify the following channels:
   - `#greenlang-incidents` -- incident status
   - `#platform-data` -- engineering response
   - `#platform-oncall` -- on-call engineer
   - `#data-pipeline-ops` -- data pipeline impact notification

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | ERP connector down, sync queue growing | On-call engineer | Immediate (<5 min) |
| L2 | ERP connector down > 15 minutes, Scope 3 data feed interrupted | Platform team lead + #platform-data | 15 minutes |
| L3 | ERP connector down > 30 minutes, compliance reporting blocked | Platform team + compliance team + CTO notification | Immediate |
| L4 | ERP connector down due to infrastructure failure affecting multiple services | All-hands engineering + incident commander | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** ERP Connector Health (`/d/erp-connector-service`)
- **Alert:** `ERPConnectorServiceDown` (this alert)
- **Key metrics to watch:**
  - `up{job="erp-connector-service"}` (should always be >= 2)
  - `gl_erp_connector_sync_jobs_total` rate (should be non-zero during business hours)
  - `gl_erp_connector_spend_records_total` rate (should be non-zero)
  - `gl_erp_connector_active_connections` (should match expected connections)
  - `gl_erp_connector_connection_errors_total` (should be 0 or near-zero)
  - `gl_erp_connector_sync_duration_seconds` p99 (should stay below 300s)
  - Pod restart count (should be 0)
  - Container memory usage vs limit (should stay below 80%)

### Capacity Planning

1. **Maintain minimum 2 replicas** across different availability zones
2. **PDB ensures at least 1 pod** is available during disruptions
3. **HPA scales from 2 to 8 replicas** based on CPU and memory utilization
4. **Database connection pool** is sized for expected concurrency (default: min 2, max 10)
5. **Sync batch size** is configurable (default: 1000 records per batch)

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `ERPConnectorServiceDown` | Critical | This alert -- no ERP connector pods running |
| `ERPHighSyncFailureRateCritical` | Critical | >25% of sync jobs are failing |
| `ERPConnectionFailure` | Critical | No active ERP connections |
| `ERPAuditGap` | Critical | Sync operations without audit entries |
| `ERPDatabaseConnectionFailure` | Critical | Database connection errors |

### Runbook Maintenance

- **Last reviewed:** 2026-02-08
- **Owner:** Platform Data Team
- **Review cadence:** Quarterly or after any P1 ERP connector incident
- **Related runbooks:** [ERP Connector Sync Failures](./erp-connector-sync-failures.md)
