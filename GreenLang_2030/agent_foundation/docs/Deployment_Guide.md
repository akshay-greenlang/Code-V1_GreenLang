# Deployment Guide

## Production Deployment of GreenLang Agent Foundation

This guide covers deployment strategies, configurations, and best practices for running GreenLang agents in production environments.

---

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Infrastructure Requirements](#infrastructure-requirements)
6. [Configuration Management](#configuration-management)
7. [Security Hardening](#security-hardening)
8. [Monitoring Setup](#monitoring-setup)
9. [Scaling Strategies](#scaling-strategies)
10. [Disaster Recovery](#disaster-recovery)

---

## Deployment Overview

### Deployment Architecture

```mermaid
graph TB
    subgraph "Production Environment"
        LB[Load Balancer]
        subgraph "Agent Cluster"
            A1[Agent Pod 1]
            A2[Agent Pod 2]
            A3[Agent Pod N]
        end
        subgraph "Data Layer"
            PG[PostgreSQL]
            RD[Redis]
            KF[Kafka]
        end
        subgraph "Observability"
            PR[Prometheus]
            GR[Grafana]
            JG[Jaeger]
        end
    end

    LB --> A1
    LB --> A2
    LB --> A3
    A1 --> PG
    A2 --> RD
    A3 --> KF
    A1 --> PR
```

---

## Docker Deployment

### Dockerfile

```dockerfile
# Production Dockerfile for GreenLang Agent
FROM python:3.11-slim as builder

# Build stage
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim

# Security: Run as non-root user
RUN useradd -m -u 1000 greenlang && \
    mkdir -p /app/logs /app/data && \
    chown -R greenlang:greenlang /app

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /home/greenlang/.local
COPY --chown=greenlang:greenlang . .

USER greenlang

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  agent:
    build: .
    image: greenlang/agent:latest
    container_name: greenlang-agent
    environment:
      - ENV=production
      - LOG_LEVEL=INFO
      - DATABASE_URL=postgresql://user:pass@postgres:5432/greenlang
      - REDIS_URL=redis://redis:6379
      - KAFKA_BROKER=kafka:9092
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
      - kafka
    networks:
      - greenlang-network
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  postgres:
    image: postgres:14-alpine
    environment:
      - POSTGRES_DB=greenlang
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - greenlang-network

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    networks:
      - greenlang-network

  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      - KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka:9092
    depends_on:
      - zookeeper
    networks:
      - greenlang-network

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      - ZOOKEEPER_CLIENT_PORT=2181
    networks:
      - greenlang-network

volumes:
  postgres-data:
  redis-data:

networks:
  greenlang-network:
    driver: bridge
```

---

## Kubernetes Deployment

### Namespace and ConfigMap

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: greenlang

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
  namespace: greenlang
data:
  config.yaml: |
    agent:
      name: production-agent
      version: 1.0.0
      capabilities:
        - reasoning
        - memory
        - orchestration
    llm:
      provider: openai
      model: gpt-4
    memory:
      short_term:
        backend: redis
        capacity: 10GB
      long_term:
        backend: postgresql
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: greenlang-agent
  namespace: greenlang
spec:
  replicas: 10
  selector:
    matchLabels:
      app: greenlang-agent
  template:
    metadata:
      labels:
        app: greenlang-agent
    spec:
      serviceAccountName: greenlang-sa
      containers:
      - name: agent
        image: greenlang/agent:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        env:
        - name: ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /app/config
      volumes:
      - name: config
        configMap:
          name: agent-config
```

### Service and Ingress

```yaml
apiVersion: v1
kind: Service
metadata:
  name: greenlang-agent-service
  namespace: greenlang
spec:
  selector:
    app: greenlang-agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: greenlang-ingress
  namespace: greenlang
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.greenlang.ai
    secretName: greenlang-tls
  rules:
  - host: api.greenlang.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: greenlang-agent-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: greenlang-agent-hpa
  namespace: greenlang
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: greenlang-agent
  minReplicas: 10
  maxReplicas: 1000
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## Cloud Deployment

### AWS Deployment

```terraform
# Terraform configuration for AWS

provider "aws" {
  region = "us-west-2"
}

# ECS Cluster
resource "aws_ecs_cluster" "greenlang" {
  name = "greenlang-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Task Definition
resource "aws_ecs_task_definition" "agent" {
  family                   = "greenlang-agent"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "2048"
  memory                   = "4096"

  container_definitions = jsonencode([
    {
      name  = "agent"
      image = "greenlang/agent:latest"
      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]
      environment = [
        {
          name  = "ENV"
          value = "production"
        }
      ]
      secrets = [
        {
          name      = "DATABASE_URL"
          valueFrom = aws_secretsmanager_secret.database_url.arn
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = "/ecs/greenlang-agent"
          awslogs-region        = "us-west-2"
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

# ECS Service
resource "aws_ecs_service" "agent" {
  name            = "greenlang-agent-service"
  cluster         = aws_ecs_cluster.greenlang.id
  task_definition = aws_ecs_task_definition.agent.arn
  desired_count   = 10
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.agent.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.agent.arn
    container_name   = "agent"
    container_port   = 8000
  }

  auto_scaling_target {
    min_capacity = 10
    max_capacity = 100
  }
}
```

### Azure Deployment

```yaml
# Azure Container Instances
apiVersion: 2019-12-01
location: westus2
name: greenlang-agent-group
properties:
  containers:
  - name: greenlang-agent
    properties:
      image: greenlang/agent:latest
      resources:
        requests:
          cpu: 2
          memoryInGb: 4
      ports:
      - port: 8000
      environmentVariables:
      - name: ENV
        value: production
      - name: DATABASE_URL
        secureValue: ${DATABASE_URL}
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8000
```

### GCP Deployment

```yaml
# Google Cloud Run
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: greenlang-agent
  namespace: default
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "10"
        autoscaling.knative.dev/maxScale: "1000"
    spec:
      containers:
      - image: gcr.io/project/greenlang-agent:latest
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
        env:
        - name: ENV
          value: production
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
```

---

## Infrastructure Requirements

### Hardware Requirements

| Component | Development | Staging | Production |
|-----------|-------------|---------|------------|
| CPU Cores | 4 | 16 | 64+ |
| Memory | 16GB | 64GB | 256GB+ |
| Storage | 100GB SSD | 500GB SSD | 2TB+ NVMe |
| Network | 100Mbps | 1Gbps | 10Gbps |
| Nodes | 1 | 3 | 10+ |

### Software Requirements

```yaml
dependencies:
  runtime:
    python: ">=3.11"
    docker: ">=24.0"
    kubernetes: ">=1.28"

  databases:
    postgresql: ">=14"
    redis: ">=7.0"
    neo4j: ">=5.0"

  message_queue:
    kafka: ">=3.6"
    redis_streams: "optional"

  observability:
    prometheus: ">=2.47"
    grafana: ">=10.0"
    jaeger: ">=1.50"
```

---

## Configuration Management

### Environment Variables

```bash
# .env.production
ENV=production
LOG_LEVEL=INFO
DEBUG=false

# Database
DATABASE_URL=postgresql://user:pass@db.greenlang.internal:5432/production
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# Redis
REDIS_URL=redis://redis.greenlang.internal:6379
REDIS_MAX_CONNECTIONS=50

# Kafka
KAFKA_BROKERS=kafka1:9092,kafka2:9092,kafka3:9092
KAFKA_CONSUMER_GROUP=greenlang-agents

# LLM Providers
OPENAI_API_KEY=${OPENAI_API_KEY}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}

# Security
JWT_SECRET=${JWT_SECRET}
ENCRYPTION_KEY=${ENCRYPTION_KEY}

# Monitoring
PROMETHEUS_PORT=9090
JAEGER_ENDPOINT=http://jaeger:14268/api/traces
```

### Secrets Management

```yaml
# Using Kubernetes Secrets
apiVersion: v1
kind: Secret
metadata:
  name: greenlang-secrets
  namespace: greenlang
type: Opaque
data:
  database-url: <base64-encoded>
  api-keys: <base64-encoded>
  jwt-secret: <base64-encoded>
```

---

## Security Hardening

### Security Checklist

- ✅ Run containers as non-root user
- ✅ Use read-only root filesystem
- ✅ Enable network policies
- ✅ Implement RBAC
- ✅ Use secrets management
- ✅ Enable TLS everywhere
- ✅ Regular security scanning
- ✅ Implement rate limiting
- ✅ Enable audit logging
- ✅ Use security contexts

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: greenlang-network-policy
  namespace: greenlang
spec:
  podSelector:
    matchLabels:
      app: greenlang-agent
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: greenlang-gateway
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
```

---

## Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'greenlang-agents'
    kubernetes_sd_configs:
    - role: pod
      namespaces:
        names:
        - greenlang
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      action: keep
      regex: greenlang-agent
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "GreenLang Agent Monitoring",
    "panels": [
      {
        "title": "Agent Status",
        "targets": [
          {
            "expr": "up{job='greenlang-agents'}"
          }
        ]
      },
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "targets": [
          {
            "expr": "process_resident_memory_bytes"
          }
        ]
      }
    ]
  }
}
```

---

## Scaling Strategies

### Horizontal Scaling

```python
# Auto-scaling configuration
scaling_config = {
    'min_instances': 10,
    'max_instances': 1000,
    'target_cpu': 70,
    'target_memory': 80,
    'scale_up_rate': '10 instances/minute',
    'scale_down_rate': '5 instances/minute',
    'cooldown_period': 300  # seconds
}
```

### Vertical Scaling

```yaml
resources:
  small:
    cpu: 1
    memory: 2Gi
  medium:
    cpu: 2
    memory: 4Gi
  large:
    cpu: 4
    memory: 8Gi
  xlarge:
    cpu: 8
    memory: 16Gi
```

---

## Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# Automated backup script

# Database backup
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql

# Redis backup
redis-cli --rdb backup_$(date +%Y%m%d_%H%M%S).rdb

# Upload to S3
aws s3 sync ./backups s3://greenlang-backups/$(date +%Y%m%d)/

# Cleanup old backups
find ./backups -mtime +7 -delete
```

### Recovery Procedures

1. **Database Recovery**
   ```bash
   psql $DATABASE_URL < backup.sql
   ```

2. **Redis Recovery**
   ```bash
   redis-cli --rdb restore.rdb
   ```

3. **Application Recovery**
   ```bash
   kubectl rollout restart deployment/greenlang-agent -n greenlang
   ```

---

## Production Checklist

### Pre-Deployment

- [ ] Security scan completed
- [ ] Load testing completed
- [ ] Backup strategy tested
- [ ] Monitoring configured
- [ ] Alerts configured
- [ ] Documentation updated
- [ ] Runbooks created
- [ ] Team trained

### Post-Deployment

- [ ] Health checks passing
- [ ] Metrics visible
- [ ] Logs aggregating
- [ ] Alerts working
- [ ] Performance acceptable
- [ ] Security verified
- [ ] Backup running
- [ ] Documentation published

---

**Last Updated**: November 2024
**Version**: 1.0.0
**Maintainer**: GreenLang DevOps Team