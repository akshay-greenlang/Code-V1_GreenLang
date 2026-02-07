# GreenLang Prometheus Stack Module

**GreenLang Climate OS | OBS-001**

This Terraform module deploys a production-grade Prometheus monitoring stack for GreenLang Climate OS, including:

- **Prometheus** (HA) - Metrics collection and short-term storage
- **Thanos** - Long-term storage (S3) and HA query layer
- **Alertmanager** - Alert routing and notifications
- **PushGateway** - Metrics from batch jobs
- **Grafana** - Visualization (via kube-prometheus-stack)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GreenLang Prometheus Stack                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Prometheus Layer                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ Prometheus  │  │ Prometheus  │  │ Thanos      │  │ Thanos      │  │   │
│  │  │ Server 0    │  │ Server 1    │  │ Sidecar 0   │  │ Sidecar 1   │  │   │
│  │  │ (HA Pair)   │  │ (HA Pair)   │  │             │  │             │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Thanos Layer                                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ Thanos      │  │ Thanos      │  │ Thanos      │  │ Thanos      │  │   │
│  │  │ Query       │  │ Store GW    │  │ Compactor   │  │ Ruler       │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       Storage Layer (S3)                              │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐│   │
│  │  │  gl-thanos-metrics-{env}  (Intelligent Tiering, 2-year retention)││   │
│  │  └──────────────────────────────────────────────────────────────────┘│   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Requirements

| Name | Version |
|------|---------|
| terraform | >= 1.5.0 |
| aws | >= 5.0 |
| helm | >= 2.12 |
| kubernetes | >= 2.25 |

## Providers

| Name | Version |
|------|---------|
| aws | >= 5.0 |
| helm | >= 2.12 |
| kubernetes | >= 2.25 |

## Usage

### Basic Usage (Development)

```hcl
module "prometheus_stack" {
  source = "../../modules/prometheus-stack"

  environment  = "dev"
  cluster_name = "greenlang-dev"

  # Disable Thanos for dev (saves cost)
  enable_thanos = false

  # Single replica for dev
  prometheus_replica_count = 1
  prometheus_retention_days = 2
  prometheus_storage_size   = "10Gi"

  # Alertmanager (Slack only)
  alertmanager_slack_webhook = var.slack_webhook_url

  tags = {
    Environment = "dev"
    Project     = "GreenLang"
  }
}
```

### Production Usage

```hcl
module "prometheus_stack" {
  source = "../../modules/prometheus-stack"

  environment  = "prod"
  cluster_name = "greenlang-prod"
  aws_region   = "eu-west-1"

  # Prometheus HA
  prometheus_replica_count  = 2
  prometheus_retention_days = 7
  prometheus_storage_size   = "50Gi"

  # Thanos for long-term storage
  enable_thanos         = true
  thanos_retention_raw  = "30d"
  thanos_retention_5m   = "120d"
  thanos_retention_1h   = "730d"  # 2 years

  # Alertmanager HA with PagerDuty
  alertmanager_replica_count = 2
  alertmanager_slack_webhook = var.slack_webhook_url
  alertmanager_pagerduty_key = var.pagerduty_key

  # PushGateway for batch jobs
  enable_pushgateway = true

  # IRSA configuration
  oidc_provider     = module.eks.oidc_provider
  oidc_provider_arn = module.eks.oidc_provider_arn

  tags = {
    Environment = "prod"
    Project     = "GreenLang"
  }
}
```

## Input Variables

### Required Variables

| Name | Description | Type |
|------|-------------|------|
| `environment` | Deployment environment (dev, staging, prod) | `string` |
| `cluster_name` | EKS cluster name for IRSA configuration | `string` |

### Prometheus Configuration

| Name | Description | Type | Default |
|------|-------------|------|---------|
| `prometheus_replica_count` | Number of Prometheus replicas | `number` | `2` |
| `prometheus_retention_days` | Local data retention in days | `number` | `7` |
| `prometheus_storage_size` | PVC size for Prometheus | `string` | `"50Gi"` |
| `prometheus_storage_class` | Kubernetes StorageClass | `string` | `"gp3"` |

### Thanos Configuration

| Name | Description | Type | Default |
|------|-------------|------|---------|
| `enable_thanos` | Enable Thanos for long-term storage | `bool` | `true` |
| `thanos_retention_raw` | Raw metrics retention | `string` | `"30d"` |
| `thanos_retention_5m` | 5-minute downsampled retention | `string` | `"120d"` |
| `thanos_retention_1h` | 1-hour downsampled retention | `string` | `"730d"` |
| `thanos_query_replicas` | Thanos Query replicas | `number` | `2` |
| `thanos_store_gateway_replicas` | Store Gateway replicas | `number` | `2` |

### Alertmanager Configuration

| Name | Description | Type | Default |
|------|-------------|------|---------|
| `alertmanager_replica_count` | Alertmanager replicas | `number` | `2` |
| `alertmanager_slack_webhook` | Slack webhook URL (sensitive) | `string` | `""` |
| `alertmanager_pagerduty_key` | PagerDuty key (sensitive) | `string` | `""` |
| `alertmanager_slack_channel` | Default Slack channel | `string` | `"#greenlang-alerts"` |

### PushGateway Configuration

| Name | Description | Type | Default |
|------|-------------|------|---------|
| `enable_pushgateway` | Enable PushGateway | `bool` | `true` |
| `pushgateway_replica_count` | PushGateway replicas | `number` | `2` |

### Helm Chart Versions

| Name | Description | Type | Default |
|------|-------------|------|---------|
| `kube_prometheus_stack_chart_version` | kube-prometheus-stack version | `string` | `"56.21.4"` |
| `thanos_chart_version` | Bitnami Thanos version | `string` | `"15.7.10"` |
| `pushgateway_chart_version` | PushGateway version | `string` | `"2.8.0"` |

## Outputs

| Name | Description |
|------|-------------|
| `prometheus_endpoint` | Prometheus server endpoint |
| `thanos_query_endpoint` | Thanos Query endpoint |
| `alertmanager_endpoint` | Alertmanager endpoint |
| `pushgateway_endpoint` | PushGateway endpoint |
| `thanos_bucket_name` | S3 bucket for Thanos |
| `thanos_bucket_arn` | S3 bucket ARN |
| `prometheus_iam_role_arn` | IAM role for IRSA |
| `grafana_endpoint` | Grafana endpoint |

## Data Retention

| Resolution | Retention | Storage |
|------------|-----------|---------|
| Raw (15s) | 30 days | S3 |
| 5-minute | 120 days | S3 |
| 1-hour | 2 years | S3 |
| Local | 7 days | PVC |

## S3 Lifecycle

- **0-30 days**: STANDARD storage
- **30+ days**: INTELLIGENT_TIERING
- **730 days**: Objects deleted

## Alert Routing

| Severity | Destination |
|----------|-------------|
| `critical` | PagerDuty + Slack |
| `warning` | Slack (warning channel) |
| `info` | Slack (default channel) |

## ServiceMonitors Created

- `greenlang-api` - GreenLang API metrics
- `agent-factory` - Agent Factory metrics
- `postgresql` - PostgreSQL exporter
- `redis` - Redis exporter
- `kong-gateway` - Kong API Gateway

## PodMonitors Created

- `greenlang-pods` - All pods with `prometheus.io/scrape=true`
- `greenlang-agents` - Agent pods in gl-* namespaces

## Security

- **IRSA**: No static AWS credentials
- **Network Policies**: Restricted ingress/egress
- **TLS**: S3 bucket enforces HTTPS
- **Encryption**: S3 bucket uses AES256/KMS

## Troubleshooting

### Prometheus not scraping targets

1. Check ServiceMonitor labels match Prometheus selector
2. Verify network policies allow scraping
3. Check target pods are running and healthy

### Thanos Sidecar not uploading

1. Check IRSA role is correctly configured
2. Verify S3 bucket policy allows access
3. Check Thanos sidecar logs: `kubectl logs -n monitoring prometheus-0 -c thanos-sidecar`

### High memory usage

1. Check cardinality: `prometheus_tsdb_head_series`
2. Add recording rules for high-cardinality queries
3. Consider increasing memory limits or reducing retention

## Related Documentation

- [PRD-OBS-001](../../../GreenLang%20Development/05-Documentation/PRD-OBS-001-Prometheus-Metrics-Collection.md)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Thanos Documentation](https://thanos.io/tip/thanos/getting-started.md/)
- [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)

## License

Copyright 2026 GreenLang. All rights reserved.
