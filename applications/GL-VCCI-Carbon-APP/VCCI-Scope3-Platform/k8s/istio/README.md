# Istio Service Mesh Configuration for GL-VCCI

## Overview

Istio provides advanced traffic management, security, and observability features for the GL-VCCI platform. This is an optional but recommended enhancement for production deployments.

## Benefits

- **Traffic Management**: Advanced routing, load balancing, and traffic splitting
- **Security**: mTLS encryption between services, authorization policies
- **Observability**: Distributed tracing, metrics, and service graphs
- **Resilience**: Circuit breaking, retries, timeouts, fault injection
- **Canary Deployments**: Fine-grained traffic control for progressive rollouts

## Installation

### Prerequisites

```bash
# Ensure Kubernetes cluster is running
kubectl cluster-info

# Install Istio CLI
curl -L https://istio.io/downloadIstio | sh -
cd istio-*
export PATH=$PWD/bin:$PATH
```

### Install Istio

```bash
# Install Istio with default profile
istioctl install --set profile=production -y

# Enable automatic sidecar injection for vcci namespaces
kubectl label namespace vcci-production istio-injection=enabled
kubectl label namespace vcci-staging istio-injection=enabled
kubectl label namespace vcci-dev istio-injection=enabled

# Verify installation
kubectl get pods -n istio-system
```

## Configuration Files

All Istio configuration files are located in `k8s/istio/`:

- `gateway.yaml` - Ingress gateway configuration
- `virtual-service.yaml` - Traffic routing rules
- `destination-rule.yaml` - Load balancing and connection pool settings
- `authorization-policy.yaml` - Service-to-service authorization
- `peer-authentication.yaml` - mTLS settings
- `service-entry.yaml` - External service access
- `telemetry.yaml` - Observability configuration

## Usage

```bash
# Apply all Istio configurations
kubectl apply -f k8s/istio/

# Verify configurations
istioctl analyze -n vcci-production
```

## Monitoring

Access Istio observability tools:

```bash
# Kiali (Service Mesh Dashboard)
istioctl dashboard kiali

# Grafana (Metrics)
istioctl dashboard grafana

# Jaeger (Distributed Tracing)
istioctl dashboard jaeger

# Prometheus (Metrics Backend)
istioctl dashboard prometheus
```

## See Individual Configuration Files

Refer to the following files for detailed configurations:
- `gateway.yaml` - Ingress configuration
- `virtual-service.yaml` - Routing rules
- `destination-rule.yaml` - Load balancing
- `authorization-policy.yaml` - Security policies
- `peer-authentication.yaml` - mTLS configuration
