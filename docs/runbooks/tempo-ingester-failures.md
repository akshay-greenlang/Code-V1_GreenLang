# Tempo Ingester Failures

## Alert

**Alert Name:** `TempoIngesterFlushFailing`

**Severity:** Critical

**Threshold:** `sum(rate(tempo_ingester_failed_flushes_total[5m])) > 0` for 10 minutes

**Duration:** 10 minutes

---

## Description

This alert fires when Grafana Tempo ingesters are failing to flush completed blocks to S3 object storage. The ingester is responsible for:

1. **Receiving spans** from the distributor and assembling them into traces
2. **Holding traces in memory** until the trace idle timeout or max block lifetime
3. **Writing completed blocks** to the Write-Ahead Log (WAL) on local disk
4. **Flushing WAL blocks** to S3 for durable long-term storage

When flushes fail, data accumulates in the WAL. If the WAL fills up or the ingester is restarted before a successful flush, trace data is permanently lost.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | Recent traces may be lost; trace search returns incomplete results |
| **Data Impact** | Critical | Unflushed trace data exists only in-memory and WAL; loss risk is high |
| **SLA Impact** | High | Trace data completeness SLA (99.9%) is at risk |
| **Revenue Impact** | Medium | Compliance agents rely on traces for audit evidence |

---

## Symptoms

- `tempo_ingester_failed_flushes_total` counter is incrementing
- WAL size (`tempo_ingester_wal_bytes`) is growing without corresponding flushes
- Live traces in memory (`tempo_ingester_live_traces`) are accumulating above normal
- Ingester pod memory usage is climbing toward limits
- S3 backend request errors are visible in Tempo Operations dashboard
- Recent traces are missing from Grafana Explore search results

---

## Diagnostic Steps

### Step 1: Check Ingester Flush Metrics

```promql
# Failed flushes per ingester pod
sum(rate(tempo_ingester_failed_flushes_total[5m])) by (pod)

# WAL size per ingester (should not grow continuously)
sum(tempo_ingester_wal_bytes) by (pod)

# Live traces in memory (elevated count indicates backlog)
sum(tempo_ingester_live_traces) by (pod)

# Flush duration (high values indicate slow S3 writes)
histogram_quantile(0.99, sum(rate(tempo_ingester_flush_duration_seconds_bucket[5m])) by (le, pod))

# Successful flushes (should be non-zero)
sum(rate(tempo_ingester_blocks_flushed_total[5m])) by (pod)
```

### Step 2: Check Ingester Pod Health

```bash
# List ingester pods and their status
kubectl get pods -n tracing -l app.kubernetes.io/component=ingester

# Check for OOMKilled restarts
kubectl get pods -n tracing -l app.kubernetes.io/component=ingester \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\t"}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}'

# Check resource usage
kubectl top pods -n tracing -l app.kubernetes.io/component=ingester

# Check events for scheduling or resource issues
kubectl get events -n tracing --field-selector involvedObject.kind=Pod --sort-by='.lastTimestamp' | grep ingester
```

### Step 3: Check Ingester Logs

```bash
# Get recent error logs from all ingesters
kubectl logs -n tracing -l app.kubernetes.io/component=ingester --tail=200 | grep -i "error\|fail\|flush"

# Check for S3-specific errors
kubectl logs -n tracing -l app.kubernetes.io/component=ingester --tail=500 | grep -i "s3\|bucket\|access denied\|timeout\|connection refused"

# Check for WAL corruption errors
kubectl logs -n tracing -l app.kubernetes.io/component=ingester --tail=500 | grep -i "wal\|corrupt\|invalid"

# Check for memory pressure
kubectl logs -n tracing -l app.kubernetes.io/component=ingester --tail=200 | grep -i "memory\|oom\|limit\|pressure"
```

### Step 4: Check S3 Backend Health

```promql
# S3 request errors
sum(rate(tempo_tempodb_backend_request_duration_seconds_count{status_code=~"5.."}[5m])) by (operation)

# S3 latency by operation (PUT latency directly affects flush)
histogram_quantile(0.99, sum(rate(tempo_tempodb_backend_request_duration_seconds_bucket[5m])) by (le, operation))

# S3 request rate
sum(rate(tempo_tempodb_backend_request_duration_seconds_count[5m])) by (operation)
```

### Step 5: Check WAL Disk Usage

```bash
# Check PVC usage on each ingester
for pod in $(kubectl get pods -n tracing -l app.kubernetes.io/component=ingester -o jsonpath='{.items[*].metadata.name}'); do
  echo "=== $pod ==="
  kubectl exec -n tracing $pod -- df -h /var/tempo/wal
  kubectl exec -n tracing $pod -- du -sh /var/tempo/wal/*
done

# Check PVC status
kubectl get pvc -n tracing | grep ingester
```

### Step 6: Check Network Connectivity to S3

```bash
# Exec into an ingester pod and test S3 connectivity
kubectl exec -n tracing -it <ingester-pod> -- /bin/sh

# Test DNS resolution for S3
nslookup s3.us-east-1.amazonaws.com

# Test connectivity to S3 VPC endpoint (if using)
wget -q --spider https://s3.us-east-1.amazonaws.com && echo "S3 reachable" || echo "S3 unreachable"
```

---

## Resolution Steps

### Scenario 1: S3 Permission Errors (Access Denied)

**Symptoms:** Logs show "access denied", "forbidden", or "authorization" errors

**Cause:** IRSA role/policy was modified, or STS token expired.

**Resolution:**

1. **Verify IRSA service account annotation:**

```bash
kubectl get sa -n tracing tempo-ingester -o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}'
```

2. **Verify IAM policy allows required S3 operations:**

```bash
ROLE_ARN=$(kubectl get sa -n tracing tempo-ingester -o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}')
ROLE_NAME=$(echo $ROLE_ARN | cut -d'/' -f2)

# Check attached policies
aws iam list-attached-role-policies --role-name $ROLE_NAME

# Verify policy document includes s3:PutObject, s3:GetObject, s3:DeleteObject, s3:ListBucket
aws iam get-role-policy --role-name $ROLE_NAME --policy-name tempo-s3-access
```

3. **Required IAM permissions for Tempo ingester:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": [
        "arn:aws:s3:::gl-tempo-traces-*",
        "arn:aws:s3:::gl-tempo-traces-*/*"
      ]
    }
  ]
}
```

4. **Restart ingesters to pick up new STS tokens:**

```bash
kubectl rollout restart statefulset -n tracing tempo-ingester
```

### Scenario 2: S3 Connectivity Issues

**Symptoms:** Logs show "connection refused", "timeout", "no such host"

**Cause:** VPC endpoint misconfiguration, security group changes, or DNS resolution failure.

**Resolution:**

1. **Check VPC endpoint status:**

```bash
aws ec2 describe-vpc-endpoints --filters "Name=service-name,Values=com.amazonaws.us-east-1.s3" \
  --query "VpcEndpoints[*].{Id:VpcEndpointId,State:State,RouteTableIds:RouteTableIds}"
```

2. **Verify security groups allow S3 traffic:**

```bash
# Get the security group for the EKS node group
NODE_SG=$(aws eks describe-cluster --name gl-eks-prod --query "cluster.resourcesVpcConfig.clusterSecurityGroupId" --output text)

# Check egress rules for HTTPS (443) to S3 prefix list
aws ec2 describe-security-groups --group-ids $NODE_SG --query "SecurityGroups[0].IpPermissionsEgress"
```

3. **Test from within the cluster:**

```bash
kubectl run s3-test --rm -it --image=amazonlinux:2 --restart=Never -- \
  bash -c "yum install -y aws-cli && aws s3 ls s3://gl-tempo-traces-prod/ --max-items 1"
```

### Scenario 3: WAL Disk Full

**Symptoms:** Logs show "no space left on device", PVC at 100%

**Cause:** Prolonged flush failures caused WAL to grow beyond PVC capacity.

**Resolution:**

1. **Check current WAL size:**

```bash
kubectl exec -n tracing <ingester-pod> -- df -h /var/tempo/wal
```

2. **If root cause (S3) is fixed, wait for natural flush:**

The ingester will automatically retry flushing. Monitor:

```promql
# Watch for successful flushes resuming
sum(rate(tempo_ingester_blocks_flushed_total[5m])) by (pod)

# Watch WAL size decreasing
sum(tempo_ingester_wal_bytes) by (pod)
```

3. **If WAL is corrupt or disk is completely full:**

```bash
# CAUTION: This will lose unflushed trace data
# Only do this if flush cannot recover

# Scale down the specific ingester
kubectl scale statefulset tempo-ingester -n tracing --replicas=<N-1>

# Delete the PVC (data loss for unflushed traces)
kubectl delete pvc tempo-ingester-wal-tempo-ingester-<N> -n tracing

# Scale back up (new PVC will be provisioned)
kubectl scale statefulset tempo-ingester -n tracing --replicas=<N>
```

4. **Increase PVC size for future prevention:**

```yaml
# In Helm values
ingester:
  persistence:
    size: 30Gi  # Increase from 10Gi
```

```bash
# If StorageClass supports expansion
kubectl patch pvc tempo-ingester-wal-tempo-ingester-0 -n tracing \
  -p '{"spec":{"resources":{"requests":{"storage":"30Gi"}}}}'
```

### Scenario 4: Ingester Memory Pressure

**Symptoms:** OOMKilled restarts, memory usage near limits, slow GC in logs

**Cause:** Too many live traces in memory, high ingestion rate without corresponding flush rate.

**Resolution:**

1. **Check memory usage vs limits:**

```bash
kubectl top pods -n tracing -l app.kubernetes.io/component=ingester
kubectl get pods -n tracing -l app.kubernetes.io/component=ingester -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[0].resources.limits.memory}{"\n"}{end}'
```

2. **Increase memory limits:**

```yaml
# In Helm values
ingester:
  resources:
    requests:
      memory: 4Gi
      cpu: "2"
    limits:
      memory: 8Gi
      cpu: "4"
```

3. **Tune trace limits to reduce memory pressure:**

```yaml
# In Tempo config (overrides section)
overrides:
  defaults:
    ingestion:
      max_traces_per_user: 50000      # Reduce from 100000
      max_bytes_per_trace: 5000000    # 5MB per trace max
    global:
      max_bytes_per_trace: 5000000
```

4. **Reduce flush interval to free memory faster:**

```yaml
# In Tempo config
ingester:
  max_block_duration: 5m           # Flush blocks every 5 minutes (default 30m)
  max_block_bytes: 524288000       # 500MB max block size before flush
  trace_idle_period: 10s           # Consider trace complete after 10s idle
```

### Scenario 5: High Ingestion Rate Overload

**Symptoms:** All ingesters at high memory/CPU, flush duration increasing, distributor showing backpressure

**Resolution:**

1. **Scale ingesters horizontally:**

```bash
# Increase replica count
kubectl scale statefulset tempo-ingester -n tracing --replicas=6  # from 3

# Or use HPA if configured
kubectl get hpa -n tracing tempo-ingester
```

2. **Add head sampling in the OTel Collector to reduce volume:**

```yaml
# In OTel Collector config
processors:
  probabilistic_sampler:
    sampling_percentage: 50  # Sample 50% of traces
```

3. **Apply per-tenant rate limits:**

```yaml
# In Tempo config
overrides:
  defaults:
    ingestion:
      rate_limit_bytes: 15000000     # 15MB/sec per tenant
      burst_size_bytes: 20000000     # 20MB burst
```

---

## WAL Recovery Procedure

If an ingester crashed with WAL data that was not flushed, the WAL replay process will attempt recovery on restart.

### Automatic WAL Recovery

When the ingester restarts, it automatically:

1. Scans the WAL directory for unflushed blocks
2. Replays WAL entries to reconstruct in-memory state
3. Attempts to flush recovered blocks to S3
4. Logs recovery progress and any irrecoverable entries

Monitor recovery:

```bash
# Watch ingester startup logs
kubectl logs -n tracing <ingester-pod> -f | grep -i "wal\|replay\|recover"
```

```promql
# Track WAL replay progress
tempo_ingester_wal_replay_duration_seconds
```

### Manual WAL Recovery

If automatic recovery fails:

1. **Stop the ingester:**

```bash
kubectl scale statefulset tempo-ingester -n tracing --replicas=0
```

2. **Create a recovery pod with access to the PVC:**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: wal-recovery
  namespace: tracing
spec:
  containers:
  - name: recovery
    image: grafana/tempo:2.6.1
    command: ["/bin/sh", "-c", "sleep 3600"]
    volumeMounts:
    - name: wal
      mountPath: /var/tempo/wal
  volumes:
  - name: wal
    persistentVolumeClaim:
      claimName: tempo-ingester-wal-tempo-ingester-0
```

3. **Inspect WAL contents:**

```bash
kubectl exec -n tracing wal-recovery -- ls -la /var/tempo/wal/
kubectl exec -n tracing wal-recovery -- du -sh /var/tempo/wal/*
```

4. **Clean corrupt WAL segments if identified:**

```bash
# Remove corrupt segments (data loss for those segments)
kubectl exec -n tracing wal-recovery -- rm /var/tempo/wal/<corrupt-segment>
```

5. **Restart the ingester:**

```bash
kubectl delete pod wal-recovery -n tracing
kubectl scale statefulset tempo-ingester -n tracing --replicas=3
```

---

## Prevention

### Monitoring

- **Dashboard:** Tempo Operations (`/d/tempo-operations`)
- **Alert:** `TempoIngesterFlushFailing` (this alert)
- **Key metrics to watch:**
  - `tempo_ingester_failed_flushes_total` (should be 0)
  - `tempo_ingester_wal_bytes` (should not grow continuously)
  - `tempo_ingester_flush_duration_seconds` P99 (should stay under 5s)
  - S3 backend error rate (should be 0)

### Capacity Planning

1. **Size ingester memory** at 2x the expected WAL size to allow for burst
2. **Size WAL PVC** at 3x the expected max WAL size for recovery headroom
3. **Plan for 3+ ingesters** for high availability with replication factor 2
4. **Monitor S3 costs** -- rising PUT costs indicate more frequent flushes

### Configuration Best Practices

```yaml
# Recommended production ingester settings
ingester:
  max_block_duration: 10m
  max_block_bytes: 1073741824    # 1GB
  trace_idle_period: 30s
  flush_check_period: 10s
  persistence:
    size: 30Gi                    # 3x expected WAL size
  resources:
    requests:
      memory: 4Gi
      cpu: "2"
    limits:
      memory: 8Gi
      cpu: "4"
  replicas: 3
```

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Observability Team
- **Review cadence:** Quarterly or after any P1 tracing incident
- **Related alerts:** `TempoStorageErrors`, `TempoCompactorHalted`, `NoTracesReceived`
- **Related dashboards:** Tempo Operations, Tracing Overview
