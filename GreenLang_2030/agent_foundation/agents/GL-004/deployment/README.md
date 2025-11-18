# GL-004 Deployment Guide

## Prerequisites
- Kubernetes cluster v1.24+
- kubectl configured
- Docker registry access

## Deployment Steps

1. Create namespace:
   ```bash
   kubectl create namespace greenlang
   ```

2. Apply manifests:
   ```bash
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   kubectl apply -f configmap.yaml
   ```

3. Verify deployment:
   ```bash
   kubectl get pods -n greenlang -l app=gl-004
   ```

## Monitoring
- Metrics: http://gl-004.greenlang.svc.cluster.local:8001/metrics
- Health: http://gl-004.greenlang.svc.cluster.local:8000/health
