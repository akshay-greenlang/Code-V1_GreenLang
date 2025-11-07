# GreenLang Troubleshooting Guide

**Version:** 1.0
**Last Updated:** 2025-11-07
**Document Classification:** Operations Reference
**Review Cycle:** Quarterly

---

## Executive Summary

This guide provides systematic troubleshooting procedures for common issues in the GreenLang production environment. It includes symptom identification, diagnostic procedures, and resolution steps for performance, application, and infrastructure problems.

**Troubleshooting Philosophy:**
1. **Observe:** Gather facts before making changes
2. **Hypothesize:** Form theories based on evidence
3. **Test:** Validate hypotheses systematically
4. **Document:** Record findings and solutions
5. **Prevent:** Implement fixes to avoid recurrence

---

## Table of Contents

1. [Performance Issues](#performance-issues)
2. [Application Issues](#application-issues)
3. [Infrastructure Issues](#infrastructure-issues)
4. [Database Issues](#database-issues)
5. [Network Issues](#network-issues)
6. [Debugging Tools](#debugging-tools)
7. [Log Analysis](#log-analysis)

---

## Performance Issues

### Issue 1: Slow Response Times (p95 >1s)

**Symptoms:**
- API requests taking longer than usual
- Timeout errors
- Customer complaints about slowness

**Diagnostic Steps:**

1. **Check Current Metrics**
```bash
# Check p95 latency
curl -G "http://prometheus.greenlang.io/api/v1/query" \
  --data-urlencode 'query=histogram_quantile(0.95, rate(gl_api_latency_seconds_bucket[5m]))'

# Check which endpoints are slow
kubectl logs -l app=greenlang-api --tail=1000 | \
  grep "duration_ms" | awk '{print $5, $7}' | sort -k2 -rn | head -20
```

2. **Identify Bottleneck**
```bash
# Check CPU usage
kubectl top pods -l app=greenlang-api

# Check memory usage
kubectl top pods -l app=greenlang-api --sort-by=memory

# Check database connections
psql -h db.greenlang.io -c "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"
```

**Common Causes & Solutions:**

**Cause 1: Database Slow Queries**
```bash
# Identify slow queries
psql -h db.greenlang.io -c "
  SELECT pid, now() - query_start AS duration, query
  FROM pg_stat_activity
  WHERE state = 'active' AND now() - query_start > interval '1 second'
  ORDER BY duration DESC LIMIT 10;
"

# Solution: Add missing index
psql -h db.greenlang.io -c "CREATE INDEX CONCURRENTLY idx_executions_created_at ON executions(created_at);"

# Or: Kill long-running query (if safe)
psql -h db.greenlang.io -c "SELECT pg_terminate_backend(PID);"
```

**Cause 2: High CPU Usage**
```bash
# Check CPU usage
kubectl top nodes

# Solution: Scale horizontally
kubectl scale deployment greenlang-api --replicas=10

# Or: Scale vertically (increase CPU limits)
kubectl set resources deployment greenlang-api --limits=cpu=2000m
```

**Cause 3: LLM Provider Latency**
```bash
# Check provider latency in Grafana
# Dashboard: https://grafana.greenlang.io/d/llm-providers

# Solution: Switch to faster provider
kubectl set env deployment/greenlang-api PRIMARY_LLM_PROVIDER=anthropic

# Or: Reduce timeout
kubectl set env deployment/greenlang-api LLM_TIMEOUT_MS=10000
```

**Cause 4: Cache Miss Rate High**
```bash
# Check cache hit rate
curl -G "http://prometheus.greenlang.io/api/v1/query" \
  --data-urlencode 'query=rate(gl_cache_hits_total[5m]) / (rate(gl_cache_hits_total[5m]) + rate(gl_cache_misses_total[5m]))'

# Solution: Warm up cache
for i in {1..100}; do curl -s https://api.greenlang.io/v1/agents > /dev/null; done

# Or: Increase cache size
kubectl set env deployment/greenlang-api CACHE_SIZE_MB=512
```

**Validation:**
```bash
# Verify improvement
# p95 latency should be < 500ms
# Monitor for 15 minutes to confirm stability
```

---

### Issue 2: High CPU Usage (>80%)

**Symptoms:**
- CPU usage consistently above 80%
- Slow request processing
- Increased latency

**Diagnostic Steps:**

1. **Identify CPU-Heavy Processes**
```bash
# Check pod CPU usage
kubectl top pods --sort-by=cpu

# Check which containers using CPU
kubectl exec -it deploy/greenlang-api -- top

# Check CPU by process
kubectl exec -it deploy/greenlang-api -- ps aux --sort=-%cpu | head -20
```

2. **Check for CPU-Intensive Operations**
```bash
# Check logs for long-running operations
kubectl logs -l app=greenlang-api --tail=500 | grep "duration_ms" | \
  awk '{print $7}' | sort -rn | head -20

# Check for infinite loops or stuck threads
kubectl exec -it deploy/greenlang-api -- pstack PID
```

**Common Causes & Solutions:**

**Cause 1: Traffic Spike**
```bash
# Check request rate
curl -G "http://prometheus.greenlang.io/api/v1/query" \
  --data-urlencode 'query=rate(gl_requests_total[5m])'

# Solution: Scale horizontally
kubectl scale deployment greenlang-api --replicas=12
```

**Cause 2: Inefficient Code Path**
```bash
# Identify hot code paths using profiling
kubectl exec -it deploy/greenlang-api -- py-spy top --pid 1

# Solution: Optimize code or add caching
# Document finding and create improvement ticket
```

**Cause 3: External API Delays**
```bash
# Check if waiting on external APIs
kubectl logs -l app=greenlang-api | grep "external_api" | grep "duration_ms"

# Solution: Increase timeout or use async processing
kubectl set env deployment/greenlang-api EXTERNAL_API_TIMEOUT=5000
```

**Validation:**
- CPU usage should drop below 70%
- Request processing time should improve

---

### Issue 3: High Memory Usage (>85%)

**Symptoms:**
- Memory usage growing over time
- OOMKilled pods
- Pods restarting frequently

**Diagnostic Steps:**

1. **Check Memory Usage**
```bash
# Check pod memory
kubectl top pods --sort-by=memory

# Check memory by process
kubectl exec -it deploy/greenlang-api -- ps aux --sort=-%mem | head -20

# Check for memory leaks
kubectl logs deploy/greenlang-api | grep -i "memory\|oom"
```

2. **Analyze Memory Growth**
```bash
# Check memory over time in Grafana
# Dashboard: https://grafana.greenlang.io/d/memory-usage

# Generate memory dump (for analysis)
kubectl exec -it deploy/greenlang-api -- python -c "
import gc
import sys
for obj in gc.get_objects():
    print(sys.getsizeof(obj), type(obj))
" | sort -rn | head -20
```

**Common Causes & Solutions:**

**Cause 1: Memory Leak**
```bash
# Restart pods as immediate mitigation
kubectl rollout restart deployment/greenlang-api

# Solution: Fix memory leak in code
# Document leak pattern and create fix ticket
```

**Cause 2: Large Objects in Memory**
```bash
# Check cache sizes
kubectl exec -it deploy/greenlang-api -- python -c "
from greenlang.cache import get_cache
cache = get_cache()
print(f'Cache size: {cache.size()} items')
print(f'Cache memory: {cache.memory_usage()} bytes')
"

# Solution: Reduce cache size
kubectl set env deployment/greenlang-api CACHE_MAX_ITEMS=1000
```

**Cause 3: Too Many Concurrent Requests**
```bash
# Check concurrent request count
kubectl exec -it deploy/greenlang-api -- netstat -an | grep ESTABLISHED | wc -l

# Solution: Increase memory limits (temporary)
kubectl set resources deployment greenlang-api --limits=memory=4Gi

# Long-term: Optimize memory usage or scale horizontally
```

**Validation:**
- Memory usage should stabilize below 75%
- No OOMKilled events
- Pods not restarting

---

### Issue 4: Database Connection Pool Exhausted

**Symptoms:**
- "Too many connections" errors
- Timeouts on database operations
- Connection refused errors

**Diagnostic Steps:**

1. **Check Connection Pool Status**
```bash
# Check active connections
psql -h db.greenlang.io -c "
  SELECT count(*), state, application_name
  FROM pg_stat_activity
  WHERE datname = 'greenlang'
  GROUP BY state, application_name
  ORDER BY count(*) DESC;
"

# Check connection pool in application
kubectl logs -l app=greenlang-api | grep "connection_pool"
```

2. **Identify Connection Leaks**
```bash
# Check for idle connections
psql -h db.greenlang.io -c "
  SELECT count(*), state
  FROM pg_stat_activity
  WHERE datname = 'greenlang' AND state = 'idle'
  GROUP BY state;
"

# Check connection age
psql -h db.greenlang.io -c "
  SELECT pid, state, now() - state_change AS idle_time
  FROM pg_stat_activity
  WHERE datname = 'greenlang' AND state = 'idle'
  ORDER BY idle_time DESC LIMIT 20;
"
```

**Solutions:**

**Solution 1: Increase Pool Size**
```bash
# Increase connection pool in application
kubectl set env deployment/greenlang-api DB_POOL_SIZE=50

# Increase max connections in database (requires restart)
psql -h db.greenlang.io -c "ALTER SYSTEM SET max_connections = 500;"
# Then restart database
```

**Solution 2: Kill Idle Connections**
```bash
# Kill connections idle for >5 minutes
psql -h db.greenlang.io -c "
  SELECT pg_terminate_backend(pid)
  FROM pg_stat_activity
  WHERE datname = 'greenlang'
    AND state = 'idle'
    AND now() - state_change > interval '5 minutes';
"
```

**Solution 3: Enable Connection Pooling (PgBouncer)**
```bash
# Deploy PgBouncer
kubectl apply -f k8s/pgbouncer.yaml

# Update application to use PgBouncer
kubectl set env deployment/greenlang-api DB_HOST=pgbouncer.greenlang.svc.cluster.local
```

**Validation:**
- No connection errors
- Connection pool usage < 80%
- Database responsive

---

## Application Issues

### Issue 5: Agent Execution Failures

**Symptoms:**
- Agent executions failing with errors
- Timeout errors
- "Agent not found" errors

**Diagnostic Steps:**

1. **Check Agent Execution Logs**
```bash
# Check recent execution errors
kubectl logs -l app=greenlang-agent --tail=500 | grep ERROR

# Check specific agent
kubectl logs -l app=greenlang-agent | grep "agent_name=calculator"

# Check execution metrics
curl -G "http://prometheus.greenlang.io/api/v1/query" \
  --data-urlencode 'query=rate(gl_agent_executions_failed_total[5m])'
```

2. **Validate Agent Configuration**
```bash
# List available agents
curl https://api.greenlang.io/v1/agents

# Get specific agent details
curl https://api.greenlang.io/v1/agents/calculator

# Validate agent pack
gl agent validate packs/calculator.yml
```

**Common Causes & Solutions:**

**Cause 1: Invalid Agent Configuration**
```bash
# Check agent pack syntax
gl agent validate packs/problematic-agent.yml

# Solution: Fix configuration error
vim packs/problematic-agent.yml
gl agent reload
```

**Cause 2: LLM API Errors**
```bash
# Check LLM provider status
curl https://status.openai.com/api/v2/status.json

# Check API key validity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# Solution: Rotate API key or switch provider
kubectl create secret generic llm-credentials \
  --from-literal=openai-key=$NEW_KEY --dry-run=client -o yaml | kubectl apply -f -
kubectl rollout restart deployment/greenlang-agent
```

**Cause 3: Agent Timeout**
```bash
# Check execution duration
kubectl logs -l app=greenlang-agent | grep "duration_ms" | \
  awk '{print $7}' | sort -rn | head -10

# Solution: Increase timeout
kubectl set env deployment/greenlang-agent AGENT_TIMEOUT_MS=60000
```

**Validation:**
- Agent executions succeeding
- No timeout errors
- Success rate > 99%

---

### Issue 6: Configuration Errors

**Symptoms:**
- Application failing to start
- "Configuration not found" errors
- "Invalid configuration" errors

**Diagnostic Steps:**

1. **Check Configuration Loading**
```bash
# Check pod logs for config errors
kubectl logs deploy/greenlang-api | grep -i "config\|configuration"

# Check mounted config maps
kubectl describe configmap greenlang-config

# Verify environment variables
kubectl exec -it deploy/greenlang-api -- env | grep GREENLANG
```

2. **Validate Configuration Files**
```bash
# Check configuration file syntax
python -c "import yaml; yaml.safe_load(open('config/production.yaml'))"

# Verify required fields
python scripts/validate-config.py config/production.yaml
```

**Solutions:**

**Solution 1: Fix Configuration Syntax**
```bash
# Edit configuration
kubectl edit configmap greenlang-config

# Restart pods to pick up new config
kubectl rollout restart deployment/greenlang-api
```

**Solution 2: Add Missing Environment Variables**
```bash
# Add missing variable
kubectl set env deployment/greenlang-api NEW_VAR=value

# Or update from secret
kubectl create secret generic greenlang-secrets \
  --from-literal=api-key=$NEW_KEY --dry-run=client -o yaml | kubectl apply -f -
```

**Validation:**
- Application starts successfully
- No configuration errors in logs
- All features working

---

### Issue 7: Cache Issues

**Symptoms:**
- Stale data being returned
- Cache miss rate high
- Cache-related errors

**Diagnostic Steps:**

1. **Check Cache Status**
```bash
# Check cache hit rate
curl -G "http://prometheus.greenlang.io/api/v1/query" \
  --data-urlencode 'query=rate(gl_cache_hits_total[5m]) / (rate(gl_cache_hits_total[5m]) + rate(gl_cache_misses_total[5m]))'

# Check cache size
kubectl exec -it deploy/greenlang-api -- redis-cli INFO stats
```

2. **Check for Cache Corruption**
```bash
# Check cache errors in logs
kubectl logs -l app=greenlang-api | grep -i "cache.*error"

# Test cache connectivity
kubectl exec -it deploy/greenlang-api -- redis-cli PING
```

**Solutions:**

**Solution 1: Clear Cache**
```bash
# Clear specific keys
kubectl exec -it deploy/greenlang-api -- redis-cli KEYS "agents:*" | \
  xargs kubectl exec -it deploy/greenlang-api -- redis-cli DEL

# Or: Clear all cache (use with caution)
kubectl exec -it deploy/greenlang-api -- redis-cli FLUSHALL
```

**Solution 2: Restart Cache Server**
```bash
# Restart Redis
kubectl rollout restart deployment/redis

# Wait for ready
kubectl wait --for=condition=ready pod -l app=redis --timeout=300s
```

**Solution 3: Adjust Cache TTL**
```bash
# Reduce TTL to refresh cache more often
kubectl set env deployment/greenlang-api CACHE_TTL_SECONDS=300
```

**Validation:**
- Cache hit rate > 80%
- No stale data
- No cache errors

---

## Infrastructure Issues

### Issue 8: Network Connectivity Issues

**Symptoms:**
- Services unable to reach each other
- Timeouts on service-to-service calls
- DNS resolution failures

**Diagnostic Steps:**

1. **Test Network Connectivity**
```bash
# Test pod-to-pod connectivity
kubectl exec -it deploy/greenlang-api -- ping greenlang-worker.greenlang.svc.cluster.local

# Test external connectivity
kubectl exec -it deploy/greenlang-api -- curl -I https://api.openai.com

# Check DNS resolution
kubectl exec -it deploy/greenlang-api -- nslookup db.greenlang.io
```

2. **Check Network Policies**
```bash
# List network policies
kubectl get networkpolicies -A

# Describe specific policy
kubectl describe networkpolicy greenlang-api-policy
```

3. **Check Service Configuration**
```bash
# List services
kubectl get svc

# Check service endpoints
kubectl get endpoints greenlang-api

# Describe service
kubectl describe svc greenlang-api
```

**Solutions:**

**Solution 1: Fix Network Policy**
```bash
# Temporarily disable network policy for testing
kubectl delete networkpolicy greenlang-api-policy

# If that fixes it, update policy
kubectl apply -f k8s/network-policies/greenlang-api-fixed.yaml
```

**Solution 2: Fix DNS**
```bash
# Restart CoreDNS
kubectl rollout restart deployment coredns -n kube-system

# Test DNS after restart
kubectl exec -it deploy/greenlang-api -- nslookup kubernetes.default
```

**Solution 3: Check Firewall Rules**
```bash
# Check if firewall blocking
# AWS: Check security groups
aws ec2 describe-security-groups --group-ids sg-xxxxx

# Update security group rules if needed
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxx --protocol tcp --port 8000 --cidr 10.0.0.0/16
```

**Validation:**
- Services can reach each other
- External connectivity working
- DNS resolving correctly

---

### Issue 9: Disk Space Issues

**Symptoms:**
- "No space left on device" errors
- Services failing to write logs
- Database unable to write

**Diagnostic Steps:**

1. **Check Disk Usage**
```bash
# Check node disk usage
kubectl get nodes -o wide
ssh node1 "df -h"

# Check pod volume usage
kubectl exec -it deploy/greenlang-api -- df -h

# Find large files
ssh node1 "du -sh /var/* | sort -rh | head -20"
```

2. **Identify What's Using Space**
```bash
# Check log files
ssh node1 "du -sh /var/log/* | sort -rh | head -20"

# Check Docker images
ssh node1 "docker system df"

# Check persistent volumes
kubectl get pv
```

**Solutions:**

**Solution 1: Clean Up Logs**
```bash
# Delete old logs
ssh node1 "find /var/log -name '*.log' -mtime +7 -delete"

# Rotate logs
ssh node1 "logrotate -f /etc/logrotate.conf"
```

**Solution 2: Clean Up Docker**
```bash
# Remove unused images
ssh node1 "docker image prune -af"

# Remove unused volumes
ssh node1 "docker volume prune -f"

# Full cleanup
ssh node1 "docker system prune -af --volumes"
```

**Solution 3: Expand Disk**
```bash
# AWS: Expand EBS volume
aws ec2 modify-volume --volume-id vol-xxxxx --size 500

# Resize filesystem
ssh node1 "sudo resize2fs /dev/xvda1"

# Verify
ssh node1 "df -h"
```

**Validation:**
- Disk usage < 80%
- No space errors
- Services writing successfully

---

### Issue 10: File Permission Issues

**Symptoms:**
- "Permission denied" errors
- Services unable to write files
- Configuration files not readable

**Diagnostic Steps:**

1. **Check File Permissions**
```bash
# Check file permissions
kubectl exec -it deploy/greenlang-api -- ls -la /app

# Check process user
kubectl exec -it deploy/greenlang-api -- whoami

# Check file ownership
kubectl exec -it deploy/greenlang-api -- stat /app/config.yaml
```

2. **Check Security Context**
```bash
# Check pod security context
kubectl get pod greenlang-api-xxxxx -o yaml | grep -A 10 securityContext
```

**Solutions:**

**Solution 1: Fix File Permissions**
```bash
# Fix permissions in container
kubectl exec -it deploy/greenlang-api -- chmod 644 /app/config.yaml

# Fix ownership
kubectl exec -it deploy/greenlang-api -- chown app:app /app/config.yaml
```

**Solution 2: Update Security Context**
```yaml
# Update deployment with correct security context
apiVersion: apps/v1
kind: Deployment
metadata:
  name: greenlang-api
spec:
  template:
    spec:
      securityContext:
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
```

```bash
# Apply update
kubectl apply -f k8s/deployments/greenlang-api.yaml
```

**Validation:**
- No permission errors
- Services can read/write files
- Logs being written successfully

---

## Database Issues

### Issue 11: Replication Lag

**Symptoms:**
- Stale data being read from replicas
- Replication lag alert firing
- "Replication slot overflow" warnings

**Diagnostic Steps:**

1. **Check Replication Status**
```bash
# Check replication lag
psql -h db-replica.greenlang.io -c "
  SELECT now() - pg_last_xact_replay_timestamp() AS replication_lag;
"

# Check replication slots
psql -h db-primary.greenlang.io -c "
  SELECT slot_name, active, restart_lsn, confirmed_flush_lsn
  FROM pg_replication_slots;
"
```

2. **Check Network Between Primary and Replica**
```bash
# Test connectivity
ping db-replica.greenlang.io

# Check bandwidth
iperf3 -c db-replica.greenlang.io
```

**Solutions:**

**Solution 1: Temporary - Route Reads to Primary**
```bash
# Route all reads to primary temporarily
kubectl set env deployment/greenlang-api DB_READ_HOST=db-primary.greenlang.io
```

**Solution 2: Increase Replica Resources**
```bash
# AWS RDS: Scale up replica
aws rds modify-db-instance \
  --db-instance-identifier greenlang-replica \
  --db-instance-class db.r5.2xlarge \
  --apply-immediately
```

**Solution 3: Reduce Write Load**
```bash
# Enable write batching
kubectl set env deployment/greenlang-api DB_BATCH_WRITES=true

# Reduce unnecessary writes
kubectl set env deployment/greenlang-api LOG_LEVEL=WARNING
```

**Validation:**
- Replication lag < 10 seconds
- Replica keeping up with primary
- No replication errors

---

## Debugging Tools

### Essential Commands

**Pod Information:**
```bash
# Get pod details
kubectl describe pod greenlang-api-xxxxx

# Get pod logs
kubectl logs greenlang-api-xxxxx
kubectl logs greenlang-api-xxxxx --previous  # Previous container

# Get pod events
kubectl get events --field-selector involvedObject.name=greenlang-api-xxxxx

# Execute command in pod
kubectl exec -it greenlang-api-xxxxx -- bash
```

**Network Debugging:**
```bash
# Check connectivity
kubectl exec -it deploy/greenlang-api -- curl -v https://api.openai.com

# DNS lookup
kubectl exec -it deploy/greenlang-api -- nslookup db.greenlang.io

# Trace route
kubectl exec -it deploy/greenlang-api -- traceroute db.greenlang.io

# Check open connections
kubectl exec -it deploy/greenlang-api -- netstat -an | grep ESTABLISHED
```

**Database Debugging:**
```bash
# Check database connections
psql -h db.greenlang.io -c "SELECT * FROM pg_stat_activity;"

# Check slow queries
psql -h db.greenlang.io -c "
  SELECT pid, now() - query_start AS duration, query
  FROM pg_stat_activity
  WHERE state = 'active' AND now() - query_start > interval '5 seconds'
  ORDER BY duration DESC;
"

# Check table sizes
psql -h db.greenlang.io -c "
  SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
  FROM pg_tables
  ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
  LIMIT 20;
"
```

**Performance Profiling:**
```bash
# Python profiling
kubectl exec -it deploy/greenlang-api -- py-spy top --pid 1

# Memory profiling
kubectl exec -it deploy/greenlang-api -- python -m memory_profiler script.py

# CPU profiling
kubectl exec -it deploy/greenlang-api -- python -m cProfile -s cumtime script.py
```

---

## Log Analysis

### Finding Patterns in Logs

**Search for Errors:**
```bash
# All errors in last hour
kubectl logs -l app=greenlang-api --since=1h | grep ERROR

# Specific error type
kubectl logs -l app=greenlang-api --since=1h | grep "DatabaseError"

# Count errors by type
kubectl logs -l app=greenlang-api --since=1h | grep ERROR | \
  awk '{print $5}' | sort | uniq -c | sort -rn
```

**Analyze Request Patterns:**
```bash
# Slow requests (>1s)
kubectl logs -l app=greenlang-api --since=1h | grep "duration_ms" | \
  awk '$7 > 1000 {print $0}' | sort -k7 -rn

# Most requested endpoints
kubectl logs -l app=greenlang-api --since=1h | grep "path=" | \
  awk -F'path=' '{print $2}' | awk '{print $1}' | sort | uniq -c | sort -rn

# Requests by status code
kubectl logs -l app=greenlang-api --since=1h | grep "status=" | \
  awk -F'status=' '{print $2}' | awk '{print $1}' | sort | uniq -c | sort -rn
```

**Correlation Analysis:**
```bash
# Follow a specific request through logs
REQUEST_ID="abc123"
kubectl logs -l app=greenlang-api --since=1h | grep $REQUEST_ID

# Find all errors around a specific time
TIME="2025-11-07T14:30"
kubectl logs -l app=greenlang-api --since-time=$TIME --until-time=2025-11-07T14:35 | grep ERROR
```

---

## Metric Interpretation

### Key Metrics Dashboard

**Error Rate:**
```promql
# Current error rate
rate(gl_errors_total[5m]) / rate(gl_requests_total[5m])

# Healthy: < 0.01 (1%)
# Warning: 0.01 - 0.05 (1-5%)
# Critical: > 0.05 (>5%)
```

**Latency:**
```promql
# P95 latency
histogram_quantile(0.95, rate(gl_api_latency_seconds_bucket[5m]))

# Healthy: < 0.5s
# Warning: 0.5s - 1s
# Critical: > 1s
```

**Throughput:**
```promql
# Requests per second
rate(gl_requests_total[5m])

# Monitor for: Unexpected spikes or drops
```

**Resource Usage:**
```promql
# CPU usage
avg(gl_cpu_usage_percent)

# Memory usage
avg(gl_memory_usage_percent)

# Healthy: < 70%
# Warning: 70-85%
# Critical: > 85%
```

---

## How to Collect Diagnostics

### For Support Escalation

```bash
# Run diagnostic collection script
./scripts/collect-diagnostics.sh

# This collects:
# - Pod logs (last 1 hour)
# - Pod descriptions
# - Events
# - Metrics snapshot
# - Configuration
# - Database status

# Output: diagnostics-2025-11-07-14-30.tar.gz
```

### Manual Collection

```bash
# Create diagnostics directory
mkdir diagnostics-$(date +%Y%m%d-%H%M%S)
cd diagnostics-*

# Collect pod information
kubectl get pods -A > pods.txt
kubectl describe pods -l app=greenlang-api > pod-descriptions.txt

# Collect logs
kubectl logs -l app=greenlang-api --tail=5000 > api-logs.txt
kubectl logs -l app=greenlang-agent --tail=5000 > agent-logs.txt

# Collect metrics
curl http://prometheus.greenlang.io/api/v1/query?query=gl_error_rate > metrics.json

# Collect configuration
kubectl get configmap greenlang-config -o yaml > config.yaml

# Collect events
kubectl get events --sort-by='.lastTimestamp' > events.txt

# Create archive
cd ..
tar -czf diagnostics-$(date +%Y%m%d-%H%M%S).tar.gz diagnostics-*/
```

---

## Appendix: Quick Reference

### Troubleshooting Checklist

When troubleshooting any issue:
- [ ] Check monitoring dashboards first
- [ ] Review recent changes (deployments, config)
- [ ] Check logs for errors
- [ ] Verify external dependencies
- [ ] Test hypotheses systematically
- [ ] Document findings
- [ ] Create ticket for permanent fix

### Essential URLs

- Grafana: https://grafana.greenlang.io
- Prometheus: https://prometheus.greenlang.io
- Jaeger: https://jaeger.greenlang.io
- Kibana: https://kibana.greenlang.io
- Status Page: https://status.greenlang.io

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-07 | Operations Team | Initial comprehensive guide |

**Next Review Date:** 2026-02-07
