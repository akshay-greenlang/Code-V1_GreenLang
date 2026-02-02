# GreenLang Agent Helm Chart

A Helm chart for deploying GreenLang Industrial AI Agents to Kubernetes.

## Prerequisites

- Kubernetes 1.25+
- Helm 3.0+
- PV provisioner support (if persistence is enabled)
- Ingress controller (if ingress is enabled)
- cert-manager (if TLS is enabled)

## Installation

### Add the Helm Repository

```bash
helm repo add greenlang https://charts.greenlang.io
helm repo update
```

### Install the Chart

```bash
# Install with default values
helm install my-agent greenlang/greenlang-agent

# Install with custom values
helm install my-agent greenlang/greenlang-agent -f values.yaml

# Install for a specific agent
helm install thermalcommand greenlang/greenlang-agent \
  --set agent.id=gl-001 \
  --set agent.name=Thermalcommand \
  --set agent.group=thermal
```

### Install from Local Directory

```bash
# Navigate to the helm chart directory
cd deployment/helm

# Install
helm install my-agent .

# Install with custom values
helm install my-agent . -f custom-values.yaml
```

## Configuration

The following table lists the configurable parameters and their default values.

### Global Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global.imageRegistry` | Global Docker image registry | `ghcr.io` |
| `global.imagePullSecrets` | Global Docker registry secrets | `[]` |
| `global.storageClass` | Global storage class | `""` |

### Agent Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `agent.id` | Agent identifier (e.g., gl-001) | `gl-xxx` |
| `agent.name` | Agent name | `AgentName` |
| `agent.group` | Agent group for network policies | `thermal` |

### Image Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `image.repository` | Image repository | `greenlang/agent` |
| `image.tag` | Image tag | `Chart.appVersion` |
| `image.pullPolicy` | Image pull policy | `Always` |

### Deployment Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `3` |
| `strategy.type` | Deployment strategy | `RollingUpdate` |
| `resources.requests.cpu` | CPU request | `250m` |
| `resources.requests.memory` | Memory request | `256Mi` |
| `resources.limits.cpu` | CPU limit | `1000m` |
| `resources.limits.memory` | Memory limit | `1Gi` |

### Service Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `service.type` | Service type | `ClusterIP` |
| `service.port` | Service port | `80` |
| `service.targetPort` | Target port | `8000` |

### Ingress Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ingress.enabled` | Enable ingress | `true` |
| `ingress.className` | Ingress class name | `nginx` |
| `ingress.hosts[0].host` | Ingress hostname | `agent.greenlang.io` |
| `ingress.tls` | TLS configuration | `[]` |

### Autoscaling Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `autoscaling.enabled` | Enable HPA | `true` |
| `autoscaling.minReplicas` | Minimum replicas | `3` |
| `autoscaling.maxReplicas` | Maximum replicas | `20` |

### Pod Disruption Budget

| Parameter | Description | Default |
|-----------|-------------|---------|
| `pdb.enabled` | Enable PDB | `true` |
| `pdb.minAvailable` | Minimum available pods | `2` |

### Network Policy

| Parameter | Description | Default |
|-----------|-------------|---------|
| `networkPolicy.enabled` | Enable network policy | `true` |
| `networkPolicy.ingressNamespaces` | Allowed ingress namespaces | `[ingress-nginx, monitoring]` |

### Monitoring

| Parameter | Description | Default |
|-----------|-------------|---------|
| `monitoring.serviceMonitor.enabled` | Enable ServiceMonitor | `true` |
| `monitoring.prometheusRule.enabled` | Enable PrometheusRule | `true` |

## Examples

### Deploy GL-001 Thermalcommand

```yaml
# values-gl-001.yaml
agent:
  id: "gl-001"
  name: "Thermalcommand"
  group: "thermal"

image:
  repository: greenlang/gl-001-thermalcommand
  tag: "1.0.0"

ingress:
  hosts:
    - host: thermalcommand.greenlang.io
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: thermalcommand-tls
      hosts:
        - thermalcommand.greenlang.io

resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 2000m
    memory: 2Gi

autoscaling:
  minReplicas: 3
  maxReplicas: 10
```

Install:

```bash
helm install thermalcommand greenlang/greenlang-agent -f values-gl-001.yaml
```

### Production Deployment with External Secrets

```yaml
# values-production.yaml
secrets:
  enabled: false
  externalSecrets:
    enabled: true
    secretStoreRef:
      name: aws-secrets-manager
      kind: ClusterSecretStore
    data:
      - secretKey: database-url
        remoteRef:
          key: greenlang/production/gl-001
          property: database_url
      - secretKey: redis-url
        remoteRef:
          key: greenlang/production/gl-001
          property: redis_url

postgresql:
  enabled: false  # Use external RDS

redis:
  enabled: false  # Use external ElastiCache
```

### Development Deployment

```yaml
# values-development.yaml
replicaCount: 1

autoscaling:
  enabled: false

pdb:
  enabled: false

resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 500m
    memory: 512Mi

postgresql:
  enabled: true

redis:
  enabled: true

ingress:
  enabled: false

service:
  type: NodePort
```

## Upgrading

```bash
# Upgrade to a new version
helm upgrade my-agent greenlang/greenlang-agent --version 1.1.0

# Upgrade with new values
helm upgrade my-agent greenlang/greenlang-agent -f new-values.yaml

# Dry run to see changes
helm upgrade my-agent greenlang/greenlang-agent --dry-run --debug
```

## Uninstallation

```bash
helm uninstall my-agent
```

## Troubleshooting

### Check deployment status

```bash
kubectl get pods -l app.kubernetes.io/name=greenlang-agent
kubectl describe deployment my-agent
```

### Check logs

```bash
kubectl logs -l app.kubernetes.io/name=greenlang-agent -f
```

### Run Helm tests

```bash
helm test my-agent
```

### Debug template rendering

```bash
helm template my-agent . --debug
```

## Contributing

Please read our contributing guidelines before submitting pull requests.

## License

Apache License 2.0
