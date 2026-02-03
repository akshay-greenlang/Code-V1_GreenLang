# Pod Crash Loop Runbook

## Alert Names
- `PodCrashLoopBackOff`
- `HighPodRestartRate`
- `PodOOMKilled`
- `ContainerWaiting`

## Overview

This runbook provides guidance for diagnosing and resolving pods that are stuck in CrashLoopBackOff or experiencing frequent restarts. CrashLoopBackOff occurs when a container repeatedly fails to start, with Kubernetes backing off exponentially before retrying.

---

## Quick Reference

| Severity | Condition | Response Time | Escalation |
|----------|-----------|---------------|------------|
| Critical | CrashLoopBackOff >5 min | 5 minutes | On-call engineer |
| Warning  | >5 restarts/hour | 15 minutes | On-call engineer |
| Critical | OOMKilled | 5 minutes | On-call engineer |

---

## Common Causes

1. **Application Errors**
   - Unhandled exceptions during startup
   - Missing dependencies or configuration
   - Failed health checks

2. **Resource Issues**
   - Out of Memory (OOM) kills
   - CPU throttling causing startup timeouts

3. **Configuration Problems**
   - Invalid environment variables
   - Missing secrets or ConfigMaps
   - Incorrect image tags

4. **Infrastructure Issues**
   - Unable to pull container image
   - PVC not bound
   - Network connectivity issues

---

## Diagnosis Steps

### 1. Get Pod Status and Events

```bash
# Get pod status
kubectl get pod <pod-name> -n <namespace> -o wide

# Get detailed pod description (CRITICAL - shows events and reasons)
kubectl describe pod <pod-name> -n <namespace>

# Get recent events for the namespace
kubectl get events -n <namespace> --sort-by='.lastTimestamp' | tail -20
```

### 2. Check Container Logs

```bash
# Get current container logs
kubectl logs <pod-name> -n <namespace>

# Get logs from previous crashed container (IMPORTANT for crash loops)
kubectl logs <pod-name> -n <namespace> --previous

# Get logs for a specific container in multi-container pod
kubectl logs <pod-name> -n <namespace> -c <container-name> --previous

# Follow logs in real-time
kubectl logs <pod-name> -n <namespace> -f

# Get last 100 lines with timestamps
kubectl logs <pod-name> -n <namespace> --tail=100 --timestamps
```

### 3. Check for OOM Kills

```bash
# Check if container was OOMKilled
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.status.containerStatuses[*].lastState.terminated.reason}'

# Check container exit code (137 = OOMKilled, 1 = application error)
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.status.containerStatuses[*].lastState.terminated.exitCode}'

# Check resource limits
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[*].resources}'
```

### 4. Check Image and Pull Status

```bash
# Verify image exists and is accessible
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[*].image}'

# Check image pull secrets
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.imagePullSecrets}'

# Verify secret exists
kubectl get secret <secret-name> -n <namespace>
```

### 5. Check Configuration

```bash
# Check environment variables
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[*].env}'

# Check ConfigMap references
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[*].envFrom}'

# Verify ConfigMaps exist
kubectl get configmap -n <namespace>

# Verify Secrets exist
kubectl get secrets -n <namespace>
```

### 6. Check Liveness/Readiness Probes

```bash
# Get probe configuration
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[*].livenessProbe}'
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[*].readinessProbe}'

# Check if probes are failing (look in describe output)
kubectl describe pod <pod-name> -n <namespace> | grep -A 10 "Liveness\|Readiness"
```

---

## Remediation Steps

### Based on Exit Code

#### Exit Code 1 (Application Error)

```bash
# 1. Check application logs for the error
kubectl logs <pod-name> -n <namespace> --previous

# 2. If configuration issue, fix the ConfigMap/Secret
kubectl edit configmap <configmap-name> -n <namespace>

# 3. Restart the deployment
kubectl rollout restart deployment <deployment-name> -n <namespace>
```

#### Exit Code 137 (OOMKilled)

```bash
# 1. Increase memory limits
kubectl patch deployment <deployment-name> -n <namespace> --type='json' \
  -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/memory", "value": "1Gi"}]'

# 2. Also increase requests proportionally
kubectl patch deployment <deployment-name> -n <namespace> --type='json' \
  -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/resources/requests/memory", "value": "512Mi"}]'
```

#### Exit Code 143 (SIGTERM - Graceful Shutdown Timeout)

```bash
# 1. Increase termination grace period
kubectl patch deployment <deployment-name> -n <namespace> --type='json' \
  -p='[{"op": "replace", "path": "/spec/template/spec/terminationGracePeriodSeconds", "value": 60}]'
```

### Based on Pod State

#### ImagePullBackOff

```bash
# 1. Verify image exists
docker pull <image-name>

# 2. Check image pull secret
kubectl get secret <pull-secret> -n <namespace> -o yaml

# 3. Update image pull secret if needed
kubectl create secret docker-registry <secret-name> \
  --docker-server=<registry> \
  --docker-username=<user> \
  --docker-password=<password> \
  -n <namespace>
```

#### CreateContainerConfigError

```bash
# 1. Check for missing ConfigMaps
kubectl get configmaps -n <namespace>

# 2. Check for missing Secrets
kubectl get secrets -n <namespace>

# 3. Create missing resources
kubectl create configmap <name> --from-file=<file> -n <namespace>
```

### Probe Failures

#### Liveness Probe Failure

```bash
# 1. Check if the health endpoint is correct
kubectl exec -it <pod-name> -n <namespace> -- curl -v http://localhost:8000/health

# 2. Increase initial delay if app takes time to start
kubectl patch deployment <deployment-name> -n <namespace> --type='json' \
  -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/livenessProbe/initialDelaySeconds", "value": 60}]'

# 3. Increase timeout if health check is slow
kubectl patch deployment <deployment-name> -n <namespace> --type='json' \
  -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/livenessProbe/timeoutSeconds", "value": 10}]'
```

### Emergency Actions

#### Force Delete Stuck Pod

```bash
# Force delete (use with caution)
kubectl delete pod <pod-name> -n <namespace> --grace-period=0 --force
```

#### Rollback to Previous Version

```bash
# Check rollout history
kubectl rollout history deployment <deployment-name> -n <namespace>

# Rollback to previous revision
kubectl rollout undo deployment <deployment-name> -n <namespace>

# Rollback to specific revision
kubectl rollout undo deployment <deployment-name> -n <namespace> --to-revision=2
```

#### Scale Down Temporarily

```bash
# Scale to zero
kubectl scale deployment <deployment-name> -n <namespace> --replicas=0

# Scale back up after fix
kubectl scale deployment <deployment-name> -n <namespace> --replicas=3
```

---

## Debugging Inside Container

If the container runs long enough:

```bash
# Get a shell into the container
kubectl exec -it <pod-name> -n <namespace> -- /bin/sh

# Check processes
ps aux

# Check disk space
df -h

# Check network connectivity
curl -v http://database-service:5432
nslookup database-service

# Check environment variables
env | sort

# Check file permissions
ls -la /app
```

---

## Prevention Checklist

1. **Resource Limits**
   - Set appropriate memory limits (2x typical usage)
   - Set CPU requests (not limits unless necessary)

2. **Health Checks**
   - Configure appropriate liveness probes
   - Use readiness probes for traffic control
   - Set adequate `initialDelaySeconds`

3. **Configuration**
   - Use ConfigMaps and Secrets properly
   - Validate configurations in CI/CD
   - Use init containers for dependencies

4. **Graceful Shutdown**
   - Handle SIGTERM in application
   - Set appropriate `terminationGracePeriodSeconds`

---

## Verification

After remediation, verify the pod is stable:

```bash
# Watch pod status
kubectl get pods -n <namespace> -w

# Check no restarts after 10 minutes
kubectl get pods -n <namespace> -o jsonpath='{.items[*].status.containerStatuses[*].restartCount}'

# Verify application is healthy
kubectl exec -it <pod-name> -n <namespace> -- curl http://localhost:8000/health
```

---

## Escalation

If the issue persists:

1. **After 15 minutes of troubleshooting**
   - Page senior engineer
   - Consider rollback

2. **If multiple pods affected**
   - Declare incident
   - Page incident commander
   - Slack: #greenlang-incidents

3. **If rollback doesn't help**
   - Engage application development team
   - Check for recent infrastructure changes

---

## Related Runbooks

- [High CPU Usage Runbook](./high-cpu-runbook.md)
- [Database Connection Issues Runbook](./database-connection-runbook.md)
- [High Memory Usage Runbook](./high-memory-runbook.md)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-01-15 | Platform Team | Initial version |
| 1.1 | 2024-01-25 | SRE Team | Added OOM section |
| 1.2 | 2024-02-10 | Platform Team | Added probe troubleshooting |
