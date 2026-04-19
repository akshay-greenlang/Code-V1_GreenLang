# GL-016 WATERGUARD - Deployment Guide

## Overview

This directory contains Kubernetes deployment manifests for GL-016 WATERGUARD, the intelligent water treatment quality control agent in the GreenLang Industrial Sustainability Framework.

## Files

### Core Deployment Files

1. **deployment.yaml** - Main Kubernetes Deployment
   - 3 replicas with rolling update strategy
   - Resource requests/limits
   - Liveness, readiness, and startup probes
   - Security context (non-root user)
   - Volume mounts for config, certs, models, and data

2. **service.yaml** - Kubernetes Services
   - ClusterIP service for internal communication
   - LoadBalancer service for external access
   - Metrics service for monitoring

3. **configmap.yaml** - Configuration Data
   - Agent parameters
   - Water quality limits
   - Chemical dosing settings
   - Integration endpoints

4. **secret.yaml** - Sensitive Data (Template)
   - SCADA credentials
   - ERP API keys
   - Analyzer passwords
   - TLS certificates

### Scaling & Availability

5. **hpa.yaml** - Horizontal Pod Autoscaler
   - Min: 3 replicas, Max: 10 replicas
   - CPU target: 70%
   - Memory target: 80%
   - Custom metrics support
   - Vertical Pod Autoscaler (VPA)
   - Pod Disruption Budget (PDB)

### Networking

6. **networkpolicy.yaml** - Network Policies
   - Default deny all ingress
   - Allow from monitoring (Prometheus, Grafana)
   - Allow from ingress controller
   - Allow inter-agent communication
   - Egress to SCADA, ERP, analyzers

7. **ingress.yaml** - Ingress Configuration
   - TLS termination
   - Path-based routing
   - Rate limiting
   - CORS configuration
   - Security headers

### Access Control

8. **serviceaccount.yaml** - RBAC Configuration
   - Service account
   - Role and RoleBinding
   - ClusterRole and ClusterRoleBinding

### Storage

9. **pvc.yaml** - Persistent Volume Claims
   - Models storage (5Gi, ReadOnlyMany)
   - Data storage (50Gi, ReadWriteMany)
   - Backups storage (100Gi, ReadWriteMany)

### GitOps

10. **kustomization.yaml** - Kustomize Configuration
    - Common labels and annotations
    - Resource management
    - Image configuration
    - Patches for customization

## Deployment Instructions

### Prerequisites

- Kubernetes cluster (1.25+)
- kubectl configured
- Namespace created: `kubectl create namespace greenlang`
- Container registry access
- Required secrets configured

### Quick Start

```bash
# 1. Create namespace
kubectl create namespace greenlang

# 2. Create secrets (replace with actual values)
kubectl create secret generic gl-016-waterguard-secrets \
  --from-env-file=../.env \
  --namespace=greenlang

# 3. Create TLS certificates
kubectl create secret tls gl-016-waterguard-certs \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  --namespace=greenlang

# 4. Deploy using kubectl
kubectl apply -f . --namespace=greenlang

# 5. Verify deployment
kubectl get pods -n greenlang -l app=waterguard
kubectl get svc -n greenlang -l app=waterguard
```

### Using Kustomize

```bash
# Development
kubectl apply -k overlays/dev

# Staging
kubectl apply -k overlays/staging

# Production
kubectl apply -k overlays/production
```

### Using Helm (if Helm chart is available)

```bash
helm install gl-016-waterguard ./helm-chart \
  --namespace greenlang \
  --values values-production.yaml
```

## Configuration

### Environment Variables

All environment variables are defined in:
- `configmap.yaml` - Non-sensitive configuration
- `secret.yaml` - Sensitive credentials

Copy `.env.template` to `.env` and customize values.

### Water Quality Parameters

Configure water quality limits in `configmap.yaml`:

```yaml
PH_MIN: "6.5"
PH_MAX: "8.5"
PH_TARGET: "7.2"
TURBIDITY_MAX: "1.0"
CHLORINE_MIN: "0.2"
CHLORINE_MAX: "2.0"
```

### Chemical Dosing

Configure dosing parameters:

```yaml
DOSING_ENABLED: "true"
DOSING_MODE: "automatic"
PH_UP_MAX_RATE: "10.0"
CHLORINE_MAX_RATE: "5.0"
```

## Monitoring

### Health Checks

- Health endpoint: `http://waterguard.greenlang.io/health`
- Ready endpoint: `http://waterguard.greenlang.io/ready`
- Metrics endpoint: `http://waterguard.greenlang.io:9090/metrics`

### Prometheus Integration

The agent exposes Prometheus metrics on port 9090. Metrics include:
- Water quality measurements
- Chemical dosing rates
- SCADA connection status
- API request metrics
- System performance

### Logs

View logs:
```bash
# All pods
kubectl logs -n greenlang -l app=waterguard --tail=100

# Specific pod
kubectl logs -n greenlang gl-016-waterguard-xxx-xxx

# Follow logs
kubectl logs -n greenlang -l app=waterguard -f
```

## Scaling

### Manual Scaling

```bash
kubectl scale deployment gl-016-waterguard --replicas=5 -n greenlang
```

### Auto Scaling

HPA automatically scales between 3-10 replicas based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)
- Custom metrics (requests/sec, measurements/sec)

## Security

### Best Practices

1. **Secrets Management**
   - Use external secret managers (Vault, AWS Secrets Manager)
   - Never commit secrets to Git
   - Use Sealed Secrets or External Secrets Operator

2. **Network Policies**
   - Default deny all ingress
   - Explicit allow rules for required traffic
   - Separate internal and external access

3. **RBAC**
   - Principle of least privilege
   - Service account per deployment
   - Audit access regularly

4. **Container Security**
   - Run as non-root user (UID 1000)
   - Read-only root filesystem
   - Drop all capabilities
   - Security context enforced

## Troubleshooting

### Pod not starting

```bash
# Check pod status
kubectl describe pod gl-016-waterguard-xxx -n greenlang

# Check events
kubectl get events -n greenlang --sort-by='.lastTimestamp'

# Check logs
kubectl logs gl-016-waterguard-xxx -n greenlang
```

### Network issues

```bash
# Test service connectivity
kubectl run test --rm -it --image=busybox -n greenlang -- sh
wget -O- http://gl-016-waterguard:8000/health

# Check network policies
kubectl get networkpolicies -n greenlang
```

### Configuration issues

```bash
# Validate configuration
python ../validate_config.py

# Check ConfigMap
kubectl get configmap gl-016-waterguard-config -n greenlang -o yaml

# Check Secrets
kubectl get secret gl-016-waterguard-secrets -n greenlang
```

## Maintenance

### Update deployment

```bash
# Update image
kubectl set image deployment/gl-016-waterguard \
  waterguard=registry.greenlang.io/agents/waterguard:1.1.0 \
  -n greenlang

# Update ConfigMap
kubectl apply -f configmap.yaml

# Restart deployment
kubectl rollout restart deployment/gl-016-waterguard -n greenlang
```

### Rollback

```bash
# View rollout history
kubectl rollout history deployment/gl-016-waterguard -n greenlang

# Rollback to previous version
kubectl rollout undo deployment/gl-016-waterguard -n greenlang

# Rollback to specific revision
kubectl rollout undo deployment/gl-016-waterguard --to-revision=2 -n greenlang
```

## Support

For issues or questions:
- Email: devops@greenlang.io
- Documentation: https://docs.greenlang.io/agents/gl-016
- Repository: https://github.com/greenlang/agents

## License

Copyright (c) 2024 GreenLang. All rights reserved.
