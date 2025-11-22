# GL-008 TRAPCATCHER - Deployment Guide

**Complete Production Deployment Guide for SteamTrapInspector**

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Platform Deployment](#cloud-platform-deployment)
6. [Configuration](#configuration)
7. [Monitoring & Operations](#monitoring--operations)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

**Minimum:**
- CPU: 1 core
- RAM: 512 MB
- Storage: 2 GB
- Python: 3.10+

**Recommended (Production):**
- CPU: 2-4 cores
- RAM: 2 GB
- Storage: 10 GB
- Python: 3.10+

### Software Dependencies

```bash
- Python 3.10 or higher
- Docker 20.10+ (for containerization)
- Kubernetes 1.24+ (for orchestration)
- kubectl (for K8s management)
- Git (for source control)
```

---

## Local Development

### Step 1: Clone Repository

```bash
git clone https://github.com/greenlang/agents/gl-008.git
cd gl-008
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
# Create .env file
cat > .env <<EOF
AGENT_ID=GL-008-LOCAL
LOG_LEVEL=INFO
ENABLE_METRICS=true
ENABLE_CACHING=true
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-haiku
LLM_TEMPERATURE=0.0
LLM_SEED=42
EOF
```

### Step 5: Run Tests

```bash
# Run all tests
pytest tests/ -v --cov=. --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Step 6: Run Agent

```bash
# Run single inspection
python steam_trap_inspector.py --mode monitor --config run.json

# Run examples
python examples/basic_usage.py
```

---

## Docker Deployment

### Step 1: Build Image

```bash
# Build image
docker build -t greenlang/gl-008-trapcatcher:1.0.0 .

# Verify build
docker images | grep gl-008
```

### Step 2: Run Container

```bash
# Run with default configuration
docker run -d \
  --name steam-trap-inspector \
  -p 9090:9090 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  greenlang/gl-008-trapcatcher:1.0.0

# View logs
docker logs -f steam-trap-inspector

# Check health
docker exec steam-trap-inspector python -c "import steam_trap_inspector; print('OK')"
```

### Step 3: Custom Configuration

```bash
# Run with custom config
docker run -d \
  --name steam-trap-inspector \
  -p 9090:9090 \
  -e LOG_LEVEL=DEBUG \
  -e ENABLE_CACHING=true \
  -e LLM_TEMPERATURE=0.0 \
  -v $(pwd)/custom-config.json:/app/run.json \
  -v $(pwd)/logs:/app/logs \
  greenlang/gl-008-trapcatcher:1.0.0
```

### Step 4: Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  steam-trap-inspector:
    image: greenlang/gl-008-trapcatcher:1.0.0
    container_name: gl-008
    ports:
      - "9090:9090"
    environment:
      - AGENT_ID=GL-008-PROD
      - LOG_LEVEL=INFO
      - ENABLE_METRICS=true
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Kubernetes Deployment

### Step 1: Create Namespace

```bash
kubectl create namespace greenlang-agents
```

### Step 2: Create ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gl-008-config
  namespace: greenlang-agents
data:
  run.json: |
    {
      "agent_id": "GL-008-K8S",
      "log_level": "INFO",
      "enable_metrics": true,
      "enable_caching": true
    }
```

```bash
kubectl apply -f k8s/configmap.yaml
```

### Step 3: Create Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: steam-trap-inspector
  namespace: greenlang-agents
spec:
  replicas: 3
  selector:
    matchLabels:
      app: steam-trap-inspector
  template:
    metadata:
      labels:
        app: steam-trap-inspector
        version: v1.0.0
    spec:
      containers:
      - name: inspector
        image: greenlang/gl-008-trapcatcher:1.0.0
        ports:
        - containerPort: 9090
          name: metrics
        env:
        - name: AGENT_ID
          value: "GL-008-K8S"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        volumeMounts:
        - name: config
          mountPath: /app/run.json
          subPath: run.json
        - name: logs
          mountPath: /app/logs
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "import sys; sys.exit(0)"
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "import sys; sys.exit(0)"
          initialDelaySeconds: 10
          periodSeconds: 10
      volumes:
      - name: config
        configMap:
          name: gl-008-config
      - name: logs
        emptyDir: {}
```

```bash
kubectl apply -f k8s/deployment.yaml
```

### Step 4: Create Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: steam-trap-inspector
  namespace: greenlang-agents
spec:
  selector:
    app: steam-trap-inspector
  ports:
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP
```

```bash
kubectl apply -f k8s/service.yaml
```

### Step 5: Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: steam-trap-inspector-hpa
  namespace: greenlang-agents
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: steam-trap-inspector
  minReplicas: 1
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
```

```bash
kubectl apply -f k8s/hpa.yaml
```

### Step 6: Verify Deployment

```bash
# Check pods
kubectl get pods -n greenlang-agents

# Check service
kubectl get svc -n greenlang-agents

# View logs
kubectl logs -f deployment/steam-trap-inspector -n greenlang-agents

# Port forward for testing
kubectl port-forward -n greenlang-agents svc/steam-trap-inspector 9090:9090
```

---

## Cloud Platform Deployment

### AWS ECS

```bash
# Build and push image to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

docker tag greenlang/gl-008-trapcatcher:1.0.0 <account>.dkr.ecr.us-east-1.amazonaws.com/gl-008:1.0.0

docker push <account>.dkr.ecr.us-east-1.amazonaws.com/gl-008:1.0.0

# Create ECS task definition and service
aws ecs create-service \
  --cluster greenlang-cluster \
  --service-name steam-trap-inspector \
  --task-definition gl-008:1 \
  --desired-count 3
```

### Google Cloud Run

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/<project>/gl-008:1.0.0

# Deploy to Cloud Run
gcloud run deploy steam-trap-inspector \
  --image gcr.io/<project>/gl-008:1.0.0 \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2
```

### Azure Container Instances

```bash
# Push to ACR
az acr build --registry <registry-name> --image gl-008:1.0.0 .

# Deploy to ACI
az container create \
  --resource-group greenlang-rg \
  --name steam-trap-inspector \
  --image <registry>.azurecr.io/gl-008:1.0.0 \
  --cpu 2 \
  --memory 2
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_ID` | GL-008-TRAPCATCHER | Agent identifier |
| `LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `ENABLE_METRICS` | true | Enable Prometheus metrics |
| `ENABLE_CACHING` | true | Enable result caching |
| `CACHE_TTL_SECONDS` | 300 | Cache time-to-live |
| `LLM_PROVIDER` | anthropic | LLM provider (anthropic, openai) |
| `LLM_MODEL` | claude-3-haiku | LLM model name |
| `LLM_TEMPERATURE` | 0.0 | LLM temperature (must be 0.0) |
| `LLM_SEED` | 42 | LLM seed (must be 42) |

---

## Monitoring & Operations

### Prometheus Metrics

```yaml
# prometheus-config.yaml
scrape_configs:
  - job_name: 'steam-trap-inspector'
    static_configs:
      - targets: ['steam-trap-inspector:9090']
```

### Grafana Dashboard

Import dashboard ID: `gl-008-trapcatcher`

**Key Metrics:**
- Inspections per second
- Average execution time
- Cache hit rate
- Error rate
- Anomalies detected
- Energy loss identified

### Logging

```bash
# View logs
kubectl logs -f deployment/steam-trap-inspector -n greenlang-agents

# Filter by level
kubectl logs deployment/steam-trap-inspector -n greenlang-agents | grep ERROR

# Export logs
kubectl logs deployment/steam-trap-inspector -n greenlang-agents > inspector.log
```

---

## Troubleshooting

### Common Issues

**Issue: Container fails to start**
```bash
# Check logs
docker logs steam-trap-inspector

# Check file permissions
ls -la /app

# Rebuild image
docker build --no-cache -t greenlang/gl-008-trapcatcher:1.0.0 .
```

**Issue: High memory usage**
```bash
# Check memory
docker stats steam-trap-inspector

# Reduce cache size
# Set CACHE_MAX_SIZE=500 in environment
```

**Issue: Low cache hit rate**
```bash
# Increase cache TTL
# Set CACHE_TTL_SECONDS=600 in environment
```

---

## Security Checklist

- [ ] Secrets stored in Kubernetes Secrets (not ConfigMaps)
- [ ] Container runs as non-root user
- [ ] Network policies configured
- [ ] TLS enabled for all external connections
- [ ] Image scanned for vulnerabilities
- [ ] RBAC permissions minimal
- [ ] Audit logging enabled

---

## Production Readiness Checklist

- [ ] All tests passing (85%+ coverage)
- [ ] Docker image built and scanned
- [ ] Kubernetes manifests validated
- [ ] Monitoring configured (Prometheus + Grafana)
- [ ] Logging configured (centralized)
- [ ] Alerting configured (PagerDuty/Slack)
- [ ] Documentation complete
- [ ] Runbooks created
- [ ] Disaster recovery plan documented
- [ ] Performance benchmarks validated
- [ ] Security audit completed

---

**Version:** 1.0.0
**Last Updated:** 2025-01-22
**Maintained By:** GreenLang Engineering Team
