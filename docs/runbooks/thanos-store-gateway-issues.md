# Thanos Store Gateway Issues

## Alert

**Alert Name:** `ThanosStoreGatewayBucketOperationsFailed`

**Severity:** Warning / Critical

**Threshold:** `rate(thanos_objstore_bucket_operation_failures_total{component="store"}[5m]) > 0`

**Duration:** 5 minutes

---

## Description

This alert fires when the Thanos Store Gateway is experiencing failures accessing the S3 bucket. The Store Gateway is responsible for:

1. **Serving historical data** - Queries beyond Prometheus local retention
2. **Block indexing** - Maintaining an index of all blocks in S3
3. **Caching** - Caching frequently accessed chunks
4. **Query optimization** - Filtering blocks based on time range

Failures impact the ability to query historical data and may cause dashboard timeouts.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | Historical queries fail |
| **Data Impact** | Low | Data still exists in S3 |
| **SLA Impact** | Medium | Long-term dashboards affected |
| **Revenue Impact** | Low | Compliance queries may fail |

---

## Symptoms

- Historical queries (>7 days) failing or timing out
- Grafana dashboards showing gaps for older data
- Store Gateway logs showing bucket operation errors
- `thanos_objstore_bucket_operation_failures_total` increasing
- `thanos_store_bucket_cache_getrange_fetched_bytes_total` not increasing

---

## Diagnostic Steps

### Step 1: Check Store Gateway Status

```promql
# Store Gateway up
up{job="thanos-storegateway"}

# Bucket operation failures
rate(thanos_objstore_bucket_operation_failures_total{component="store"}[5m])

# Successful operations
rate(thanos_objstore_bucket_operations_total{component="store"}[5m])

# Blocks loaded
thanos_bucket_store_blocks_loaded
```

### Step 2: Check Store Gateway Logs

```bash
# Get store gateway pods
kubectl get pods -n monitoring -l app.kubernetes.io/name=thanos-storegateway

# Check logs
kubectl logs -n monitoring -l app.kubernetes.io/name=thanos-storegateway --tail=200

# Look for specific errors
kubectl logs -n monitoring -l app.kubernetes.io/name=thanos-storegateway | grep -i "error\|fail\|denied"
```

### Step 3: Check S3 Connectivity

```bash
# Exec into store gateway pod
kubectl exec -n monitoring -it <storegateway-pod> -- /bin/sh

# Test S3 access
/bin/thanos tools bucket ls --objstore.config-file=/etc/thanos/objstore.yaml

# Check bucket stats
/bin/thanos tools bucket verify --objstore.config-file=/etc/thanos/objstore.yaml --check-index
```

### Step 4: Check IRSA Configuration

```bash
# Verify service account annotation
kubectl get sa -n monitoring thanos-storegateway -o yaml

# Check for IAM role
kubectl get sa -n monitoring thanos-storegateway -o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}'
```

### Step 5: Check Cache Status

```promql
# Cache hit rate
thanos_store_bucket_cache_getrange_hits_total /
(thanos_store_bucket_cache_getrange_hits_total + thanos_store_bucket_cache_getrange_misses_total)

# Cache size
thanos_store_bucket_cache_size_bytes

# Cache evictions
rate(thanos_store_bucket_cache_evictions_total[5m])
```

### Step 6: Check Resource Usage

```bash
# Check memory usage (store gateway is memory-intensive)
kubectl top pods -n monitoring -l app.kubernetes.io/name=thanos-storegateway

# Check for OOMKill events
kubectl describe pods -n monitoring -l app.kubernetes.io/name=thanos-storegateway | grep -A 5 "State:"
```

---

## Resolution Steps

### Scenario 1: S3 Access Denied

**Symptoms:** Logs show "AccessDenied" or "access denied"

**Resolution:**

1. **Verify IRSA configuration:**

```bash
# Check service account has role annotation
kubectl get sa -n monitoring thanos-storegateway -o jsonpath='{.metadata.annotations}'

# Should see eks.amazonaws.com/role-arn
```

2. **Verify IAM role trust policy:**

```bash
ROLE_ARN=$(kubectl get sa -n monitoring thanos-storegateway -o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}')
ROLE_NAME=$(echo $ROLE_ARN | cut -d'/' -f2)

# Check trust policy
aws iam get-role --role-name $ROLE_NAME --query 'Role.AssumeRolePolicyDocument'
```

3. **Ensure trust policy allows the service account:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::ACCOUNT_ID:oidc-provider/oidc.eks.REGION.amazonaws.com/id/OIDC_ID"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "oidc.eks.REGION.amazonaws.com/id/OIDC_ID:sub": "system:serviceaccount:monitoring:thanos-storegateway"
        }
      }
    }
  ]
}
```

4. **Restart store gateway to pick up new credentials:**

```bash
kubectl rollout restart statefulset -n monitoring thanos-storegateway
```

### Scenario 2: S3 Rate Limiting / Throttling

**Symptoms:** Logs show "SlowDown" or "503" errors

**Resolution:**

1. **Check S3 request rate:**

```bash
# In CloudWatch, check S3 bucket metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/S3 \
  --metric-name AllRequests \
  --dimensions Name=BucketName,Value=gl-thanos-metrics-prod \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --period 300 \
  --statistics Sum
```

2. **Add request prefixes to distribute load:**

```yaml
# In objstore.yaml configuration
type: S3
config:
  bucket: gl-thanos-metrics-prod
  endpoint: s3.us-east-1.amazonaws.com
  # Add prefix to distribute across partitions
  prefix: thanos/
```

3. **Increase store gateway replicas for parallel reads:**

```yaml
storegateway:
  replicaCount: 4  # Increase from 2
```

### Scenario 3: Memory Pressure / OOMKill

**Symptoms:** Pods restarting, OOMKilled status

**Resolution:**

1. **Check current memory usage:**

```bash
kubectl top pods -n monitoring -l app.kubernetes.io/name=thanos-storegateway
```

2. **Increase memory limits:**

```yaml
storegateway:
  resources:
    requests:
      cpu: 500m
      memory: 2Gi
    limits:
      cpu: 2000m
      memory: 8Gi  # Increase from 4Gi
```

3. **Tune index cache size:**

```yaml
storegateway:
  extraArgs:
    - --index-cache-size=1GB  # Adjust based on available memory
    - --chunk-pool-size=1GB
```

4. **Apply changes:**

```bash
helm upgrade thanos bitnami/thanos -n monitoring -f thanos-values.yaml
```

### Scenario 4: Block Sync Issues

**Symptoms:** New blocks not appearing, stale data

**Resolution:**

1. **Check sync status:**

```promql
# Blocks synced
thanos_blocks_meta_synced{component="store"}

# Sync failures
thanos_blocks_meta_sync_failures_total
```

2. **Force a resync:**

```bash
# Delete the local block cache to force resync
kubectl exec -n monitoring thanos-storegateway-0 -- rm -rf /data/cache/*

# Restart store gateway
kubectl rollout restart statefulset -n monitoring thanos-storegateway
```

3. **Check for corrupt index files in S3:**

```bash
kubectl exec -n monitoring -it <storegateway-pod> -- /bin/thanos tools bucket verify \
  --objstore.config-file=/etc/thanos/objstore.yaml \
  --check-index
```

### Scenario 5: Slow Query Performance

**Symptoms:** Historical queries timing out but bucket operations succeed

**Resolution:**

1. **Enable index header lazy loading:**

```yaml
storegateway:
  extraArgs:
    - --store.index-header-lazy-reader=true
```

2. **Increase replica count for query parallelism:**

```yaml
storegateway:
  replicaCount: 4
  shardingStrategy: "time"  # Shard by time for better distribution
```

3. **Add chunk cache:**

```yaml
storegateway:
  extraArgs:
    - --store.caching-bucket.config=
        type: MEMCACHED
        config:
          addresses: ["memcached.monitoring.svc:11211"]
          max_get_multi_concurrency: 100
          max_item_size: 1MiB
```

### Scenario 6: Network Issues to S3

**Symptoms:** Intermittent failures, timeouts

**Resolution:**

1. **Check VPC endpoint configuration:**

```bash
# Verify S3 VPC endpoint exists
aws ec2 describe-vpc-endpoints --filters Name=service-name,Values=com.amazonaws.*.s3
```

2. **Create S3 VPC endpoint if missing:**

```hcl
resource "aws_vpc_endpoint" "s3" {
  vpc_id       = var.vpc_id
  service_name = "com.amazonaws.${var.region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = var.private_route_table_ids
}
```

3. **Increase HTTP timeout:**

```yaml
storegateway:
  extraArgs:
    - --objstore.config=
        type: S3
        config:
          http_config:
            idle_conn_timeout: 2m
            response_header_timeout: 2m
```

---

## Emergency Actions

### If All Historical Queries Are Failing

1. **Check if Thanos Query is using fallback:**

```promql
# Verify Query is trying to reach Store Gateway
thanos_query_store_apis_dns_lookups_total
thanos_query_store_apis_dns_failures_total
```

2. **Temporarily increase Prometheus retention:**

```yaml
# In prometheus values
prometheus:
  prometheusSpec:
    retention: 14d  # Increase from 7d temporarily
```

3. **Bypass Store Gateway if data is recent:**
   - Query Prometheus directly for recent data
   - Wait for Store Gateway recovery

---

## Scaling Store Gateway

### Horizontal Scaling

```yaml
storegateway:
  replicaCount: 4  # Add more replicas

  # Enable sharding for better distribution
  sharding:
    enabled: true
    hashringConfigMap: thanos-storegateway-hashring
```

### Vertical Scaling

```yaml
storegateway:
  resources:
    requests:
      cpu: 1000m
      memory: 4Gi
    limits:
      cpu: 4000m
      memory: 16Gi
```

---

## Escalation Path

| Level | Condition | Contact |
|-------|-----------|---------|
| L1 | Intermittent failures | On-call engineer |
| L2 | All historical queries failing | Platform team lead |
| L3 | S3 access issues affecting multiple services | Platform team + AWS Support |

---

## Prevention

1. **Monitor bucket operations:**

```promql
# Alert on increasing failure rate
rate(thanos_objstore_bucket_operation_failures_total{component="store"}[5m]) > 0.1
```

2. **Set up S3 CloudWatch alarms:**
   - Request rate alarms
   - 4xx/5xx error rate alarms

3. **Regular capacity planning:**
   - Monitor `thanos_bucket_store_blocks_loaded`
   - Scale before hitting limits

4. **Cache warm-up after restarts:**
   - Consider pre-warming cache for frequently queried time ranges

---

## Related Dashboards

| Dashboard | URL |
|-----------|-----|
| Thanos Store Gateway | https://grafana.greenlang.io/d/thanos-store |
| S3 Bucket Metrics | https://grafana.greenlang.io/d/s3-thanos |
| Query Performance | https://grafana.greenlang.io/d/thanos-query |

---

## Related Alerts

- `ThanosStoreGatewayIsDown`
- `ThanosStoreGatewayReplicasMismatch`
- `ThanosQueryHighDNSFailures`
- `ThanosSidecarBucketOperationsFailed`

---

## References

- [Thanos Store Gateway Documentation](https://thanos.io/tip/components/store.md/)
- [Thanos Caching](https://thanos.io/tip/components/store.md/#caching)
- [AWS S3 Performance Guidelines](https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html)
