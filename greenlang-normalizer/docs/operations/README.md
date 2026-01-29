# Operations Documentation

This directory contains operations documentation for deployment, monitoring, and incident response.

## Contents

- `deployment.md` - Deployment guide
- `monitoring.md` - Monitoring and alerting
- `scaling.md` - Scaling guidelines
- `incident-response.md` - Incident response procedures
- `runbooks/` - Operational runbooks

## Deployment

### Prerequisites

- Kubernetes 1.27+
- Helm 3.x
- PostgreSQL 15+
- Redis 7+
- Kafka 3.x (optional)

### Quick Deploy

```bash
# Add Helm repository
helm repo add greenlang https://charts.greenlang.io

# Install normalizer
helm install normalizer greenlang/gl-normalizer \
  --set redis.enabled=true \
  --set monitoring.enabled=true
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection | Yes |
| `REDIS_URL` | Redis connection | Yes |
| `KAFKA_BROKERS` | Kafka broker list | No |
| `VOCAB_PATH` | Vocabulary directory | Yes |
| `LOG_LEVEL` | Logging level | No |

## Monitoring

### Metrics

Prometheus metrics exposed at `/metrics`:

- `glnorm_requests_total` - Total requests
- `glnorm_request_duration_seconds` - Request duration
- `glnorm_conversions_total` - Conversions performed
- `glnorm_resolutions_total` - Resolutions performed
- `glnorm_errors_total` - Total errors
- `glnorm_cache_hit_ratio` - Cache effectiveness

### Alerts

Default alert rules:

- High error rate (>1% for 5m)
- High latency (p99 > 500ms)
- Low cache hit rate (<80%)
- Vocabulary sync failures

### Dashboards

Grafana dashboards available:

- Service Overview
- Performance Metrics
- Error Analysis
- Vocabulary Statistics

## Scaling

### Horizontal Scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: normalizer
spec:
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Performance Targets

- p50 latency: <50ms
- p99 latency: <200ms
- Throughput: 10,000 req/s per pod
- Availability: 99.9%

## Incident Response

### Severity Levels

- **P1**: Service down, no workaround
- **P2**: Significant degradation
- **P3**: Minor impact
- **P4**: Cosmetic issue

### Escalation

1. On-call engineer (15 min response)
2. Team lead (30 min)
3. Engineering director (1 hour)
4. CTO (2 hours)
