# GL-017 CONDENSYNC Deployment Guide

## Overview

This guide covers deployment of GL-017 CONDENSYNC in various environments: local development, Docker, Kubernetes, and cloud platforms. CONDENSYNC supports three deployment modes: Edge, Edge+Central, and Offline.

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4 cores |
| Memory | 512 MB | 2 GB |
| Storage | 1 GB | 10 GB |
| Python | 3.10+ | 3.11+ |

### Software Dependencies

- Docker 20.10+ (for containerized deployment)
- Kubernetes 1.25+ (for orchestrated deployment)
- Helm 3.10+ (for Kubernetes deployment)
- PostgreSQL 14+ or TimescaleDB 2.x (optional, for persistence)

### Network Requirements

| Port | Protocol | Purpose |
|------|----------|---------|
| 8017 | HTTP | REST API |
| 9017 | HTTP | Prometheus metrics |
| 4840 | OPC-UA | DCS integration (optional) |
| 1883 | MQTT | IoT sensors (optional) |

---

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/greenlang/gl-017-condensync.git
cd gl-017-condensync

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set environment variables
export CONDENSYNC_MODE=development
export CONDENSYNC_LOG_LEVEL=DEBUG

# Run the application
python -m condensync.main

# Verify health
curl http://localhost:8017/api/v1/health
```

### Docker Quick Start

```bash
# Pull the image
docker pull greenlang/condensync:latest

# Run with default settings
docker run -d \
  --name condensync \
  -p 8017:8017 \
  -p 9017:9017 \
  greenlang/condensync:latest

# Verify health
curl http://localhost:8017/api/v1/health
```

---

## Docker Deployment

### Building the Image

```dockerfile
# Dockerfile
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN groupadd -r condensync && useradd -r -g condensync condensync

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy application
COPY condensync/ ./condensync/
COPY config/ ./config/

# Set ownership
RUN chown -R condensync:condensync /app

# Switch to non-root user
USER condensync

# Environment
ENV PYTHONUNBUFFERED=1
ENV CONDENSYNC_MODE=production

# Expose ports
EXPOSE 8017 9017

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8017/api/v1/health')"

# Entrypoint
ENTRYPOINT ["python", "-m", "condensync.main"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  condensync:
    image: greenlang/condensync:latest
    container_name: condensync
    restart: unless-stopped
    ports:
      - "8017:8017"
      - "9017:9017"
    environment:
      - CONDENSYNC_MODE=production
      - CONDENSYNC_LOG_LEVEL=INFO
      - CONDENSYNC_DB_URL=postgresql://condensync:password@postgres:5432/condensync
      - CONDENSYNC_REDIS_URL=redis://redis:6379/0
    volumes:
      - ./config:/app/config:ro
      - condensync-data:/app/data
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8017/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - condensync-net

  postgres:
    image: timescale/timescaledb:latest-pg14
    container_name: condensync-db
    restart: unless-stopped
    environment:
      - POSTGRES_USER=condensync
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=condensync
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - condensync-net

  redis:
    image: redis:7-alpine
    container_name: condensync-redis
    restart: unless-stopped
    volumes:
      - redis-data:/data
    networks:
      - condensync-net

  prometheus:
    image: prom/prometheus:latest
    container_name: condensync-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    networks:
      - condensync-net

  grafana:
    image: grafana/grafana:latest
    container_name: condensync-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
      - grafana-data:/var/lib/grafana
    depends_on:
      - prometheus
    networks:
      - condensync-net

volumes:
  condensync-data:
  postgres-data:
  redis-data:
  prometheus-data:
  grafana-data:

networks:
  condensync-net:
    driver: bridge
```

### Running with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f condensync

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

---

## Kubernetes Deployment

### Namespace and ConfigMap

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: condensync
  labels:
    app: condensync
    environment: production
```

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: condensync-config
  namespace: condensync
data:
  config.yaml: |
    agent:
      id: GL-017
      name: CONDENSYNC
      mode: monitoring
      log_level: INFO

    api:
      host: 0.0.0.0
      port: 8017
      cors_origins:
        - "*"

    metrics:
      enabled: true
      port: 9017

    hei:
      version: "11th_Edition"
      correction_factors: true

    thresholds:
      cf_clean_pct: 85
      cf_light_fouling_pct: 75
      cf_moderate_fouling_pct: 60
      ttd_warning_k: 5
      ttd_critical_k: 10
```

### Secrets

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: condensync-secrets
  namespace: condensync
type: Opaque
stringData:
  db-url: "postgresql://condensync:password@postgres:5432/condensync"
  api-key: "your-api-key-here"
  opc-ua-credentials: |
    {
      "username": "opc_user",
      "password": "opc_password"
    }
```

### Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: condensync
  namespace: condensync
  labels:
    app: condensync
    version: v1.0.0
spec:
  replicas: 2
  selector:
    matchLabels:
      app: condensync
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: condensync
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9017"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: condensync
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
        - name: condensync
          image: greenlang/condensync:1.0.0
          imagePullPolicy: IfNotPresent
          ports:
            - name: http
              containerPort: 8017
              protocol: TCP
            - name: metrics
              containerPort: 9017
              protocol: TCP
          env:
            - name: CONDENSYNC_MODE
              value: "production"
            - name: CONDENSYNC_DB_URL
              valueFrom:
                secretKeyRef:
                  name: condensync-secrets
                  key: db-url
          envFrom:
            - configMapRef:
                name: condensync-env
          volumeMounts:
            - name: config
              mountPath: /app/config
              readOnly: true
            - name: data
              mountPath: /app/data
          resources:
            requests:
              cpu: 250m
              memory: 512Mi
            limits:
              cpu: 1000m
              memory: 2Gi
          livenessProbe:
            httpGet:
              path: /api/v1/health
              port: http
            initialDelaySeconds: 10
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /api/v1/health/ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop:
                - ALL
      volumes:
        - name: config
          configMap:
            name: condensync-config
        - name: data
          persistentVolumeClaim:
            claimName: condensync-data
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app: condensync
                topologyKey: kubernetes.io/hostname
```

### Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: condensync
  namespace: condensync
  labels:
    app: condensync
spec:
  type: ClusterIP
  ports:
    - name: http
      port: 8017
      targetPort: http
      protocol: TCP
    - name: metrics
      port: 9017
      targetPort: metrics
      protocol: TCP
  selector:
    app: condensync
```

### Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: condensync
  namespace: condensync
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
    - hosts:
        - condensync.greenlang.io
      secretName: condensync-tls
  rules:
    - host: condensync.greenlang.io
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: condensync
                port:
                  number: 8017
```

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: condensync
  namespace: condensync
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: condensync
  minReplicas: 2
  maxReplicas: 10
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
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
        - type: Pods
          value: 4
          periodSeconds: 15
      selectPolicy: Max
```

### Pod Disruption Budget

```yaml
# pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: condensync
  namespace: condensync
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: condensync
```

### ServiceMonitor (for Prometheus Operator)

```yaml
# servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: condensync
  namespace: condensync
  labels:
    app: condensync
spec:
  selector:
    matchLabels:
      app: condensync
  endpoints:
    - port: metrics
      interval: 30s
      path: /metrics
  namespaceSelector:
    matchNames:
      - condensync
```

### Deploying to Kubernetes

```bash
# Create namespace
kubectl apply -f namespace.yaml

# Apply configurations
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml

# Deploy application
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml
kubectl apply -f pdb.yaml

# Verify deployment
kubectl -n condensync get pods
kubectl -n condensync get svc
kubectl -n condensync get ingress

# Check logs
kubectl -n condensync logs -f deployment/condensync

# Port forward for local testing
kubectl -n condensync port-forward svc/condensync 8017:8017
```

---

## Helm Chart Deployment

### Using the Helm Chart

```bash
# Add GreenLang Helm repository
helm repo add greenlang https://charts.greenlang.io
helm repo update

# Install with default values
helm install condensync greenlang/condensync \
  --namespace condensync \
  --create-namespace

# Install with custom values
helm install condensync greenlang/condensync \
  --namespace condensync \
  --create-namespace \
  -f values-production.yaml

# Upgrade
helm upgrade condensync greenlang/condensync \
  --namespace condensync \
  -f values-production.yaml

# Uninstall
helm uninstall condensync --namespace condensync
```

### Sample values.yaml

```yaml
# values.yaml
replicaCount: 2

image:
  repository: greenlang/condensync
  tag: "1.0.0"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8017
  metricsPort: 9017

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: condensync.greenlang.io
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: condensync-tls
      hosts:
        - condensync.greenlang.io

resources:
  requests:
    cpu: 250m
    memory: 512Mi
  limits:
    cpu: 1000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

config:
  mode: production
  logLevel: INFO
  heiVersion: "11th_Edition"
  thresholds:
    cfCleanPct: 85
    cfLightFoulingPct: 75
    cfModerateFoulingPct: 60

postgresql:
  enabled: true
  auth:
    database: condensync
    username: condensync

redis:
  enabled: true
  architecture: standalone

prometheus:
  enabled: true

grafana:
  enabled: true
  adminPassword: admin
```

---

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CONDENSYNC_MODE` | Operation mode (development, production) | production |
| `CONDENSYNC_LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `CONDENSYNC_API_HOST` | API bind address | 0.0.0.0 |
| `CONDENSYNC_API_PORT` | API port | 8017 |
| `CONDENSYNC_METRICS_PORT` | Prometheus metrics port | 9017 |
| `CONDENSYNC_DB_URL` | Database connection URL | - |
| `CONDENSYNC_REDIS_URL` | Redis connection URL | - |
| `CONDENSYNC_OPC_UA_ENDPOINT` | OPC-UA server endpoint | - |
| `CONDENSYNC_HISTORIAN_URL` | Historian API URL | - |

### Configuration File

```yaml
# config.yaml
agent:
  id: GL-017
  name: CONDENSYNC
  version: "1.0.0"
  mode: monitoring  # monitoring, survey, batch

api:
  host: 0.0.0.0
  port: 8017
  workers: 4
  timeout: 30
  cors_origins:
    - "https://app.greenlang.io"
  rate_limit:
    requests_per_minute: 100
    burst: 20

metrics:
  enabled: true
  port: 9017
  prefix: condensync

database:
  url: postgresql://localhost:5432/condensync
  pool_size: 10
  max_overflow: 20

cache:
  url: redis://localhost:6379/0
  ttl: 3600

hei:
  version: "11th_Edition"
  correction_factors:
    material: true
    velocity: true
    temperature: true

thresholds:
  cleanliness_factor:
    clean: 85
    light_fouling: 75
    moderate_fouling: 60
  ttd:
    warning: 5
    critical: 10

integrations:
  opc_ua:
    enabled: false
    endpoint: opc.tcp://localhost:4840
    security_mode: SignAndEncrypt
    security_policy: Basic256Sha256
  historian:
    enabled: false
    type: osisoft_pi
    url: https://pi-server/piwebapi
  cmms:
    enabled: false
    type: maximo
    url: https://maximo-server/api

logging:
  level: INFO
  format: json
  output: stdout
```

---

## Troubleshooting

### Common Issues

#### 1. Container fails to start

**Symptom:** Pod in CrashLoopBackOff

**Check logs:**
```bash
kubectl -n condensync logs -f deployment/condensync --previous
```

**Common causes:**
- Missing configuration file
- Invalid database connection string
- Port already in use

**Solution:**
```bash
# Verify config exists
kubectl -n condensync get configmap condensync-config -o yaml

# Check secrets
kubectl -n condensync get secrets condensync-secrets -o yaml
```

#### 2. Health check failing

**Symptom:** Pod not Ready

**Check:**
```bash
# Test health endpoint
kubectl -n condensync exec -it deployment/condensync -- curl localhost:8017/api/v1/health

# Check readiness probe
kubectl -n condensync describe pod -l app=condensync
```

#### 3. OPC-UA connection issues

**Symptom:** Unable to connect to DCS

**Check:**
```bash
# Verify network connectivity
kubectl -n condensync exec -it deployment/condensync -- nc -zv opc-server 4840

# Check OPC-UA configuration
kubectl -n condensync exec -it deployment/condensync -- cat /app/config/config.yaml | grep -A5 opc_ua
```

#### 4. High memory usage

**Symptom:** OOMKilled

**Solution:**
```yaml
# Increase memory limits
resources:
  limits:
    memory: 4Gi
```

#### 5. Slow API responses

**Symptom:** Timeouts or high latency

**Check:**
```bash
# Check database connection
kubectl -n condensync exec -it deployment/condensync -- python -c "from condensync.db import engine; print(engine.execute('SELECT 1').fetchone())"

# Check Redis connection
kubectl -n condensync exec -it deployment/condensync -- redis-cli -h redis ping
```

### Diagnostic Commands

```bash
# Get pod status
kubectl -n condensync get pods -o wide

# Describe pod
kubectl -n condensync describe pod -l app=condensync

# Get logs
kubectl -n condensync logs -f deployment/condensync

# Get events
kubectl -n condensync get events --sort-by=.metadata.creationTimestamp

# Check resource usage
kubectl -n condensync top pods

# Port forward for debugging
kubectl -n condensync port-forward svc/condensync 8017:8017

# Execute shell in container
kubectl -n condensync exec -it deployment/condensync -- /bin/bash
```

---

## Monitoring and Alerting

### Prometheus Alerts

```yaml
# alerts.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: condensync-alerts
  namespace: condensync
spec:
  groups:
    - name: condensync
      rules:
        - alert: CondensyncDown
          expr: up{job="condensync"} == 0
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "CONDENSYNC is down"
            description: "CONDENSYNC has been down for more than 5 minutes"

        - alert: CondensyncHighErrorRate
          expr: rate(condensync_errors_total[5m]) > 0.1
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "High error rate in CONDENSYNC"
            description: "Error rate is {{ $value }} errors/sec"

        - alert: CondensyncLowCleanlinessFactory
          expr: condensync_cleanliness_factor < 60
          for: 1h
          labels:
            severity: warning
          annotations:
            summary: "Low cleanliness factor detected"
            description: "Condenser {{ $labels.condenser }} CF at {{ $value }}%"
```

### Grafana Dashboard

Import the provided dashboard from `monitoring/grafana/dashboards/condensync.json` or use dashboard ID from Grafana.com.

---

## Security Considerations

### Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: condensync
  namespace: condensync
spec:
  podSelector:
    matchLabels:
      app: condensync
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8017
    - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
      ports:
        - protocol: TCP
          port: 9017
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: condensync
      ports:
        - protocol: TCP
          port: 5432  # PostgreSQL
        - protocol: TCP
          port: 6379  # Redis
    - to:
        - ipBlock:
            cidr: 10.0.0.0/8  # OPC-UA servers
      ports:
        - protocol: TCP
          port: 4840
```

### Pod Security

```yaml
# pod-security.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: condensync
spec:
  privileged: false
  runAsUser:
    rule: MustRunAsNonRoot
  seLinux:
    rule: RunAsAny
  fsGroup:
    rule: RunAsAny
  volumes:
    - configMap
    - secret
    - persistentVolumeClaim
    - emptyDir
```

---

## Backup and Recovery

### Database Backup

```bash
# Backup PostgreSQL
kubectl -n condensync exec -it postgresql-0 -- pg_dump -U condensync condensync > backup.sql

# Restore
kubectl -n condensync exec -i postgresql-0 -- psql -U condensync condensync < backup.sql
```

### Configuration Backup

```bash
# Export all resources
kubectl -n condensync get all,configmap,secret,pvc -o yaml > condensync-backup.yaml
```

---

## Support

- **Documentation:** https://docs.greenlang.io/condensync
- **Issues:** https://github.com/greenlang/gl-017-condensync/issues
- **Support:** support@greenlang.io
