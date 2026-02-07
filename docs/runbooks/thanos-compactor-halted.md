# Thanos Compactor Halted

## Alert

**Alert Name:** `ThanosCompactorHalted`

**Severity:** Critical

**Threshold:** `thanos_compact_halted == 1`

**Duration:** 5 minutes

---

## Description

This alert fires when the Thanos Compactor has halted operations. The Compactor is responsible for:

1. **Compacting blocks** - Merging small blocks into larger ones
2. **Downsampling** - Creating 5m and 1h resolution data
3. **Retention enforcement** - Deleting blocks older than retention period
4. **Deduplication** - Removing duplicate data from HA Prometheus pairs

When halted, storage costs increase, queries slow down, and retention is not enforced.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium | Slower historical queries |
| **Data Impact** | High | Storage bloat, retention not enforced |
| **SLA Impact** | Low | Current data unaffected |
| **Revenue Impact** | Medium | S3 costs increase over time |

---

## Symptoms

- `thanos_compact_halted` metric is 1
- S3 bucket size growing rapidly
- Historical queries slower than normal
- Many small blocks in object storage
- Compactor pod logs show errors

---

## Diagnostic Steps

### Step 1: Check Compactor Status

```promql
# Compactor halted status
thanos_compact_halted

# Compactor runs
thanos_compact_group_compactions_total

# Compactor failures
thanos_compact_group_compactions_failures_total
```

### Step 2: Check Compactor Logs

```bash
# Get compactor pod
kubectl get pods -n monitoring -l app.kubernetes.io/name=thanos-compactor

# Check logs
kubectl logs -n monitoring -l app.kubernetes.io/name=thanos-compactor --tail=200

# Look for specific errors
kubectl logs -n monitoring -l app.kubernetes.io/name=thanos-compactor | grep -i "error\|halt\|fatal"
```

### Step 3: Common Error Patterns

```bash
# Overlapping blocks error
kubectl logs -n monitoring -l app.kubernetes.io/name=thanos-compactor | grep "overlapping"

# S3 access errors
kubectl logs -n monitoring -l app.kubernetes.io/name=thanos-compactor | grep "bucket\|s3\|access denied"

# Out of disk space
kubectl logs -n monitoring -l app.kubernetes.io/name=thanos-compactor | grep "disk\|space\|no space"
```

### Step 4: Check S3 Bucket Access

```bash
# Test S3 access from compactor pod
kubectl exec -n monitoring -it <compactor-pod> -- /bin/sh

# List bucket contents
/bin/thanos tools bucket ls --objstore.config-file=/etc/thanos/objstore.yaml

# Verify bucket access
/bin/thanos tools bucket verify --objstore.config-file=/etc/thanos/objstore.yaml
```

### Step 5: Check for Overlapping Blocks

```bash
# Exec into compactor pod
kubectl exec -n monitoring -it <compactor-pod> -- /bin/sh

# List blocks and check for overlaps
/bin/thanos tools bucket ls \
  --objstore.config-file=/etc/thanos/objstore.yaml \
  -o json | jq '.[] | {ulid, minTime, maxTime}'
```

### Step 6: Check Compactor Disk Space

```bash
# Check PVC usage
kubectl exec -n monitoring <compactor-pod> -- df -h /data

# Check for large temporary files
kubectl exec -n monitoring <compactor-pod> -- du -sh /data/*
```

---

## Resolution Steps

### Scenario 1: Overlapping Blocks

**Symptoms:** Logs show "overlapping blocks" error

**Cause:** Multiple Prometheus instances uploaded blocks for the same time range, or split-brain during HA.

**Resolution:**

1. **Identify overlapping blocks:**

```bash
kubectl exec -n monitoring -it <compactor-pod> -- /bin/thanos tools bucket verify \
  --objstore.config-file=/etc/thanos/objstore.yaml \
  --issues=overlapping
```

2. **Mark or delete overlapping blocks:**

```bash
# Option 1: Use Thanos bucket tools to analyze
kubectl exec -n monitoring -it <compactor-pod> -- /bin/thanos tools bucket ls \
  --objstore.config-file=/etc/thanos/objstore.yaml

# Option 2: Delete overlapping blocks manually from S3
# First, identify the ULID of overlapping blocks from logs
aws s3 rm s3://gl-thanos-metrics-prod/<ULID>/ --recursive
```

3. **Restart compactor with --wait flag:**

```yaml
# Update compactor args in Helm values
compactor:
  extraArgs:
    - --wait  # Wait for initial sync before compacting
```

```bash
kubectl rollout restart deployment -n monitoring thanos-compactor
```

### Scenario 2: S3 Permission Errors

**Symptoms:** Logs show "access denied" or "bucket operation failed"

**Resolution:**

1. **Verify IRSA configuration:**

```bash
# Check service account
kubectl get sa -n monitoring thanos-compactor -o yaml

# Verify IAM role annotation
kubectl get sa -n monitoring thanos-compactor -o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}'
```

2. **Verify IAM policy:**

```bash
# Get the role name
ROLE_ARN=$(kubectl get sa -n monitoring thanos-compactor -o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}')

# Check attached policies
aws iam list-attached-role-policies --role-name $(echo $ROLE_ARN | cut -d'/' -f2)

# Check policy permissions
aws iam get-role-policy --role-name $(echo $ROLE_ARN | cut -d'/' -f2) --policy-name thanos-s3
```

3. **Update IAM policy if needed:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": [
        "arn:aws:s3:::gl-thanos-metrics-*",
        "arn:aws:s3:::gl-thanos-metrics-*/*"
      ]
    }
  ]
}
```

4. **Restart compactor:**

```bash
kubectl rollout restart deployment -n monitoring thanos-compactor
```

### Scenario 3: Out of Disk Space

**Symptoms:** Logs show "no space left on device"

**Resolution:**

1. **Check PVC size:**

```bash
kubectl get pvc -n monitoring | grep compactor
kubectl exec -n monitoring <compactor-pod> -- df -h /data
```

2. **Clean up temporary files:**

```bash
kubectl exec -n monitoring <compactor-pod> -- rm -rf /data/compact-tmp/*
```

3. **Increase PVC size:**

```yaml
# Helm values
compactor:
  persistence:
    size: 200Gi  # Increase from 100Gi
```

4. **Apply changes (requires PVC resize support):**

```bash
kubectl patch pvc thanos-compactor-data -n monitoring -p '{"spec":{"resources":{"requests":{"storage":"200Gi"}}}}'
```

5. **Or recreate with larger PVC:**

```bash
# Delete compactor (data is in S3, safe to delete)
kubectl delete deployment -n monitoring thanos-compactor
helm upgrade prometheus prometheus-community/kube-prometheus-stack -n monitoring -f values.yaml
```

### Scenario 4: Corrupt Blocks

**Symptoms:** Logs show "block verification failed" or "corrupt"

**Resolution:**

1. **Identify corrupt blocks:**

```bash
kubectl exec -n monitoring -it <compactor-pod> -- /bin/thanos tools bucket verify \
  --objstore.config-file=/etc/thanos/objstore.yaml \
  --issues=corrupted
```

2. **Delete corrupt blocks:**

```bash
# Get ULID of corrupt block from logs
CORRUPT_ULID="01EXAMPLE123"

# Delete from S3
aws s3 rm s3://gl-thanos-metrics-prod/$CORRUPT_ULID/ --recursive
```

3. **Restart compactor:**

```bash
kubectl rollout restart deployment -n monitoring thanos-compactor
```

### Scenario 5: Compactor Taking Too Long

**Symptoms:** Compactor halts due to timeout or too many blocks

**Resolution:**

1. **Increase compactor resources:**

```yaml
compactor:
  resources:
    requests:
      cpu: 500m
      memory: 2Gi
    limits:
      cpu: 2000m
      memory: 8Gi
```

2. **Adjust compaction settings:**

```yaml
compactor:
  extraArgs:
    - --compact.concurrency=4  # Increase parallelism
    - --delete-delay=48h       # Delay deletion
```

3. **Apply and restart:**

```bash
helm upgrade prometheus prometheus-community/kube-prometheus-stack -n monitoring -f values.yaml
```

---

## Emergency Actions

### If Compactor Has Been Halted for Days

1. **First, clean up temporary data:**

```bash
kubectl exec -n monitoring <compactor-pod> -- rm -rf /data/compact-tmp/*
```

2. **Run compactor with --wait to ensure clean start:**

```bash
# Add to compactor args
- --wait
- --wait-interval=5m
```

3. **Monitor compactor progress:**

```bash
kubectl logs -n monitoring -l app.kubernetes.io/name=thanos-compactor -f
```

4. **After successful compaction, remove --wait flag:**

```bash
kubectl rollout restart deployment -n monitoring thanos-compactor
```

---

## Monitoring Compaction Progress

```promql
# Blocks in bucket
thanos_blocks_meta_synced{component="compactor"}

# Compaction runs
rate(thanos_compact_group_compactions_total[1h])

# Compaction duration
thanos_compact_group_compaction_runs_duration_seconds

# Retention deletions
thanos_compact_blocks_cleaned_total
```

---

## Escalation Path

| Level | Condition | Contact |
|-------|-----------|---------|
| L1 | Compactor halted <24h | On-call engineer |
| L2 | Compactor halted >24h, storage growing | Platform team lead |
| L3 | Data corruption, need S3 manual intervention | Platform team + SRE |

---

## Prevention

1. **Monitor compactor health:**

```promql
# Alert before halt
increase(thanos_compact_group_compactions_failures_total[1h]) > 5
```

2. **Regular disk space monitoring:**

```promql
# PVC usage
kubelet_volume_stats_used_bytes{persistentvolumeclaim=~".*compactor.*"} /
kubelet_volume_stats_capacity_bytes{persistentvolumeclaim=~".*compactor.*"} > 0.8
```

3. **Ensure proper HA labeling:**
   - All Prometheus instances should have unique `replica` labels
   - Prevents overlapping blocks

4. **Regular S3 bucket analysis:**

```bash
# Monthly bucket analysis
thanos tools bucket analyze --objstore.config-file=/etc/thanos/objstore.yaml
```

---

## Related Dashboards

| Dashboard | URL |
|-----------|-----|
| Thanos Overview | https://grafana.greenlang.io/d/thanos-overview |
| Compactor Status | https://grafana.greenlang.io/d/thanos-compactor |
| S3 Storage | https://grafana.greenlang.io/d/s3-storage |

---

## Related Alerts

- `ThanosCompactorHighCompactionFailures`
- `ThanosCompactorIsNotRunning`
- `ThanosStoreGatewayBucketOperationsFailed`
- `S3BucketSizeGrowingFast`

---

## References

- [Thanos Compactor Documentation](https://thanos.io/tip/components/compact.md/)
- [Thanos Bucket Tools](https://thanos.io/tip/components/tools.md/)
- [Handling Overlapping Blocks](https://thanos.io/tip/operating/troubleshooting.md/#overlapping-blocks)
