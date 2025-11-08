# Kubernetes Deployment - Apply Order

This document specifies the correct order to apply Kubernetes manifests for the CSRD/ESRS platform.

## Quick Start (All-in-One)

```bash
# Create namespace and apply all manifests
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
kubectl apply -f statefulset.yaml
kubectl apply -f service.yaml
kubectl apply -f deployment.yaml
kubectl apply -f hpa.yaml
kubectl apply -f ingress.yaml
```

## Detailed Step-by-Step

### 1. Namespace (Foundation)

```bash
kubectl apply -f namespace.yaml
kubectl config set-context --current --namespace=production
```

**Why first?** All other resources need a namespace to exist in.

### 2. ConfigMap (Configuration)

```bash
kubectl apply -f configmap.yaml
kubectl get configmap -n production
```

**Why second?** Application pods need configuration to start.

### 3. Secrets (Credentials)

```bash
# IMPORTANT: Edit secrets.yaml with actual values first!
cp secrets.yaml.example secrets.yaml
# Edit secrets.yaml with real credentials
kubectl apply -f secrets.yaml
kubectl get secrets -n production
```

**Why third?** Application pods need credentials to connect to services.

### 4. StatefulSets (Databases)

```bash
kubectl apply -f statefulset.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l component=database --timeout=300s -n production
kubectl wait --for=condition=ready pod -l component=cache --timeout=300s -n production
kubectl wait --for=condition=ready pod -l component=vector-db --timeout=300s -n production
```

**Why fourth?** Application needs databases running before it can start.

**Verify:**
```bash
kubectl get statefulsets -n production
kubectl get pods -n production
kubectl get pvc -n production
```

### 5. Services (Networking)

```bash
kubectl apply -f service.yaml
kubectl get services -n production
```

**Why fifth?** Services should exist before pods try to connect to them.

### 6. Deployment (Application)

```bash
kubectl apply -f deployment.yaml

# Wait for rollout
kubectl rollout status deployment/csrd-app -n production
```

**Why sixth?** Application is the main workload that depends on everything else.

**Verify:**
```bash
kubectl get deployments -n production
kubectl get pods -n production
kubectl logs -f deployment/csrd-app -n production
```

### 7. HPA (Auto-scaling)

```bash
kubectl apply -f hpa.yaml
kubectl get hpa -n production
```

**Why seventh?** HPA needs the deployment to exist before it can scale it.

**Verify:**
```bash
kubectl describe hpa csrd-hpa -n production
```

### 8. Ingress (External Access)

```bash
# Prerequisites:
# - Ingress controller installed (nginx, traefik)
# - TLS certificate created (cert-manager or manual)

kubectl apply -f ingress.yaml
kubectl get ingress -n production
```

**Why last?** Ingress should only expose the service after everything is healthy.

**Verify:**
```bash
kubectl describe ingress csrd-ingress -n production
kubectl get certificate -n production  # If using cert-manager
```

## Verification Checklist

After applying all manifests, verify:

```bash
# 1. All pods are running
kubectl get pods -n production
# Expected: All pods in Running state

# 2. All services have endpoints
kubectl get endpoints -n production
# Expected: Each service has at least one endpoint

# 3. Deployments are at desired replica count
kubectl get deployments -n production
# Expected: READY shows 3/3 or similar

# 4. HPA is monitoring
kubectl get hpa -n production
# Expected: Shows current metrics

# 5. Ingress has an address
kubectl get ingress -n production
# Expected: ADDRESS field is populated

# 6. Health checks pass
kubectl run test-health --rm -i --restart=Never \
  --image=curlimages/curl:latest \
  --namespace=production \
  -- curl -f http://csrd-service/health
# Expected: {"status":"healthy",...}
```

## Troubleshooting

### Pods stuck in Pending

```bash
kubectl describe pod <pod-name> -n production
# Look for events like "Insufficient CPU" or "PersistentVolumeClaim not found"

# Solutions:
# - Check if PVCs are bound: kubectl get pvc -n production
# - Check node resources: kubectl describe nodes
# - Check for scheduling constraints (taints, node selectors)
```

### Pods in CrashLoopBackOff

```bash
kubectl logs <pod-name> -n production
kubectl logs <pod-name> -n production --previous

# Common causes:
# - Missing secrets or configmaps
# - Database connection failed
# - Wrong environment variables

# Solutions:
# - Verify secrets exist: kubectl get secrets -n production
# - Check database is ready: kubectl get pods -l component=database
# - Verify environment variables in deployment.yaml
```

### Services have no endpoints

```bash
kubectl get endpoints -n production
kubectl describe service <service-name> -n production

# Common causes:
# - Selector doesn't match pod labels
# - Pods are not ready (failing health checks)

# Solutions:
# - Check pod labels: kubectl get pods --show-labels -n production
# - Check readiness probes: kubectl describe pod <pod-name> -n production
```

### Ingress not working

```bash
kubectl describe ingress csrd-ingress -n production

# Common causes:
# - Ingress controller not installed
# - DNS not pointing to ingress IP
# - TLS secret missing or invalid

# Solutions:
# - Install ingress controller (nginx, traefik)
# - Update DNS records to point to ingress IP
# - Verify TLS secret: kubectl get secret csrd-tls -n production
```

## Update Procedure

To update an existing deployment:

```bash
# 1. Update ConfigMap (if changed)
kubectl apply -f configmap.yaml
kubectl rollout restart deployment/csrd-app -n production

# 2. Update Secrets (if changed)
kubectl apply -f secrets.yaml
kubectl rollout restart deployment/csrd-app -n production

# 3. Update Deployment (new image version)
kubectl set image deployment/csrd-app \
  csrd-app=greenlang/csrd-app:v1.1.0 \
  -n production
kubectl rollout status deployment/csrd-app -n production

# 4. Update HPA (if scaling parameters changed)
kubectl apply -f hpa.yaml

# 5. Update Ingress (if routes changed)
kubectl apply -f ingress.yaml
```

## Rollback Procedure

If something goes wrong:

```bash
# Rollback deployment
kubectl rollout undo deployment/csrd-app -n production

# Or rollback to specific revision
kubectl rollout history deployment/csrd-app -n production
kubectl rollout undo deployment/csrd-app --to-revision=2 -n production

# Verify rollback
kubectl rollout status deployment/csrd-app -n production
```

## Complete Teardown

To remove everything:

```bash
# Delete in reverse order
kubectl delete -f ingress.yaml
kubectl delete -f hpa.yaml
kubectl delete -f deployment.yaml
kubectl delete -f service.yaml
kubectl delete -f statefulset.yaml
kubectl delete -f secrets.yaml
kubectl delete -f configmap.yaml

# Delete PVCs (WARNING: Data loss!)
kubectl delete pvc --all -n production

# Delete namespace (WARNING: Deletes everything!)
kubectl delete namespace production
```

## Production Best Practices

1. **Always apply to staging first** before production
2. **Use GitOps** (ArgoCD, Flux) for declarative deployments
3. **Version control** all manifests
4. **Test rollback** procedures regularly
5. **Monitor** deployments with alerts
6. **Document** any manual steps
7. **Backup** data before major changes
8. **Use secrets management** (Vault, Sealed Secrets)
9. **Implement** admission controllers (OPA, Kyverno)
10. **Regular security scans** (Trivy, Snyk)
