# Monitoring & Observability - Setup Guide

**Quick Start Guide for Team B3 Deliverables**

---

## Prerequisites

- Python 3.10+
- Docker (for Prometheus, Grafana)
- Kubernetes cluster (for production)
- Sentry account (optional, for error tracking)

---

## Step 1: Install Dependencies

```bash
# Navigate to project directory
cd C:\Users\aksha\Code-V1_GreenLang\GL-CSRD-APP\CSRD-Reporting-Platform

# Install/upgrade dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify monitoring dependencies
python -c "import prometheus_client, sentry_sdk, structlog, psutil; print('✓ All monitoring dependencies installed')"
```

**New Dependencies Added:**
- `prometheus-client>=0.19.0` - Prometheus metrics
- `psutil>=5.9.0` - System monitoring
- `structlog>=24.1.0` - Already present
- `python-json-logger>=2.0.0` - Already present
- `sentry-sdk>=1.40.0` - Already present

---

## Step 2: Configure Environment Variables

Create a `.env` file or set environment variables:

```bash
# Sentry Configuration (optional)
SENTRY_DSN=https://your-key@sentry.io/project-id
RELEASE_VERSION=csrd-platform@1.0.0

# Environment
ENVIRONMENT=production  # or staging, development

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/csrd-app.log
ENABLE_JSON_LOGGING=true

# Database (for health checks)
DATABASE_URL=postgresql://user:pass@localhost:5432/csrd

# Redis (for health checks)
REDIS_URL=redis://localhost:6379/0
```

---

## Step 3: Integrate with FastAPI Application

### Create/Update `main.py` or `app.py`

```python
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import monitoring components
from backend.health import health_router
from backend.metrics import metrics_router
from backend.logging_config import (
    setup_structured_logging,
    logging_middleware,
    get_logger
)
from backend.error_tracking import init_sentry

# Initialize FastAPI app
app = FastAPI(
    title="CSRD Reporting Platform",
    version="1.0.0",
    description="ESRS/CSRD Digital Reporting Platform"
)

# Setup structured logging
setup_structured_logging(
    log_level=os.getenv('LOG_LEVEL', 'INFO'),
    log_file=os.getenv('LOG_FILE', 'logs/csrd-app.log'),
    enable_json=os.getenv('ENABLE_JSON_LOGGING', 'true').lower() == 'true'
)

# Get logger
logger = get_logger(__name__)

# Initialize Sentry (if configured)
if sentry_dsn := os.getenv('SENTRY_DSN'):
    init_sentry(
        dsn=sentry_dsn,
        environment=os.getenv('ENVIRONMENT', 'production'),
        release=os.getenv('RELEASE_VERSION', 'csrd-platform@1.0.0'),
        traces_sample_rate=0.1,
        enable_tracing=True
    )
    logger.info("Sentry error tracking initialized")
else:
    logger.warning("Sentry DSN not configured - error tracking disabled")

# Add logging middleware
app.middleware("http")(logging_middleware)

# Add CORS middleware (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include monitoring routers
app.include_router(health_router)
app.include_router(metrics_router)

# Include your other routers
# app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info("CSRD Reporting Platform starting up")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("CSRD Reporting Platform shutting down")

# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "CSRD Reporting Platform",
        "version": "1.0.0",
        "status": "operational",
        "health": "/health",
        "metrics": "/metrics"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_config=None  # Use our custom logging
    )
```

---

## Step 4: Test Health Checks

```bash
# Start the application
python main.py

# Test health endpoints (in another terminal)
# Basic health
curl http://localhost:8000/health

# Liveness probe
curl http://localhost:8000/health/live

# Readiness probe
curl http://localhost:8000/health/ready | jq

# ESRS health
curl http://localhost:8000/health/esrs | jq

# Specific ESRS standard
curl http://localhost:8000/health/esrs/E1 | jq

# Prometheus metrics
curl http://localhost:8000/metrics
```

**Expected Responses:**

```json
// /health
{
  "status": "healthy",
  "timestamp": "2025-11-08T10:30:00Z",
  "version": "1.0.0"
}

// /health/ready
{
  "status": "ready",
  "timestamp": "2025-11-08T10:30:00Z",
  "checks": {
    "database": {"healthy": true, "message": "Database connection OK"},
    "cache": {"healthy": true, "message": "Cache connection OK"},
    "disk_space": {"healthy": true, "percent_used": 65.0},
    "memory": {"healthy": true, "percent_used": 75.0},
    "esrs_data": {"healthy": true, "data_points_available": 5432}
  }
}
```

---

## Step 5: Setup Prometheus

### Option A: Docker Compose (Recommended for development)

Create `docker-compose.monitoring.yml`:

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: csrd-prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/alerts:/etc/prometheus/alerts
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    ports:
      - "9090:9090"
    networks:
      - monitoring

  alertmanager:
    image: prom/alertmanager:latest
    container_name: csrd-alertmanager
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
    ports:
      - "9093:9093"
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: csrd-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    ports:
      - "3000:3000"
    networks:
      - monitoring

volumes:
  prometheus-data:
  grafana-data:

networks:
  monitoring:
    driver: bridge
```

Start monitoring stack:

```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

### Option B: Kubernetes (Production)

Apply monitoring manifests:

```bash
kubectl apply -f deployment/kubernetes/monitoring/
```

---

## Step 6: Configure Prometheus

The Prometheus configuration is already created at:
`monitoring/prometheus.yml`

**Update targets** to point to your application:

```yaml
scrape_configs:
  - job_name: 'csrd-api'
    metrics_path: '/metrics'
    scrape_interval: 10s
    static_configs:
      - targets:
          - 'host.docker.internal:8000'  # For Docker on Windows/Mac
          # - 'localhost:8000'             # For Linux
          # - 'csrd-api:8000'              # For Kubernetes
```

Verify Prometheus is scraping:

```bash
# Open Prometheus UI
http://localhost:9090

# Check targets
http://localhost:9090/targets

# Query a metric
http://localhost:9090/graph?g0.expr=csrd_esrs_data_point_coverage_ratio
```

---

## Step 7: Import Grafana Dashboard

### Method 1: UI Import

1. Open Grafana: http://localhost:3000
2. Login (admin/admin)
3. Go to Dashboards → Import
4. Upload `monitoring/grafana-csrd-dashboard.json`
5. Select Prometheus data source
6. Import

### Method 2: API Import

```bash
# Get API key from Grafana UI (Configuration → API Keys)
GRAFANA_API_KEY="your-api-key"

# Import dashboard
curl -X POST \
  http://localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -d @monitoring/grafana-csrd-dashboard.json
```

### Method 3: Provisioning (Automated)

Place dashboard in:
```
monitoring/grafana/provisioning/dashboards/csrd-dashboard.json
```

---

## Step 8: Configure Alertmanager

Create `monitoring/alertmanager.yml`:

```yaml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'esrs_standard']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
  - match:
      severity: warning
    receiver: 'warning-alerts'

receivers:
- name: 'default'
  webhook_configs:
  - url: 'http://localhost:5001/webhook'

- name: 'critical-alerts'
  # Email configuration
  email_configs:
  - to: 'team@example.com'
    from: 'alertmanager@example.com'
    smarthost: 'smtp.gmail.com:587'
    auth_username: 'alerts@example.com'
    auth_password: 'password'

  # Slack configuration
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
    channel: '#csrd-critical'
    title: 'CRITICAL: {{ .GroupLabels.alertname }}'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  # PagerDuty configuration
  pagerduty_configs:
  - service_key: 'your-pagerduty-key'

- name: 'warning-alerts'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
    channel: '#csrd-alerts'
```

Restart Alertmanager:

```bash
docker-compose -f docker-compose.monitoring.yml restart alertmanager
```

---

## Step 9: Test Monitoring End-to-End

### Generate Test Metrics

```python
from backend.metrics import (
    record_esrs_coverage,
    record_http_request,
    esrs_data_point_coverage
)

# Record ESRS coverage
record_esrs_coverage(
    esrs_standard="E1",
    company_id="test-company",
    required_points=150,
    available_points=100  # 66.7% coverage - should trigger alert
)

# Manually set coverage below threshold
esrs_data_point_coverage.labels(
    esrs_standard="E1",
    company_id="test-company"
).set(0.65)  # Should trigger ESRSDataCoverageCritical alert
```

### Verify Metrics in Prometheus

```bash
# Open Prometheus
http://localhost:9090

# Query
csrd_esrs_data_point_coverage_ratio{esrs_standard="E1"}

# Check alerts
http://localhost:9090/alerts
```

### Verify Dashboard in Grafana

```bash
# Open Grafana
http://localhost:3000

# Navigate to CSRD Compliance Dashboard
# Verify panels are populating with data
```

---

## Step 10: Kubernetes Deployment (Production)

### Deploy Application with Health Checks

Create `deployment/kubernetes/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: csrd-platform
  labels:
    app: csrd-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: csrd-platform
  template:
    metadata:
      labels:
        app: csrd-platform
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: csrd-app
        image: csrd-platform:1.0.0
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        - name: SENTRY_DSN
          valueFrom:
            secretKeyRef:
              name: csrd-secrets
              key: sentry-dsn

        # Liveness Probe
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        # Readiness Probe
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3

        # Startup Probe
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8000
          initialDelaySeconds: 0
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 30

        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: csrd-platform
  labels:
    app: csrd-platform
spec:
  type: ClusterIP
  ports:
  - port: 8000
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: csrd-platform
```

Deploy:

```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
```

Verify:

```bash
# Check pods
kubectl get pods -l app=csrd-platform

# Check health
kubectl port-forward svc/csrd-platform 8000:8000
curl http://localhost:8000/health/ready
```

---

## Step 11: Verify Logging

### Check Structured Logs

```bash
# View application logs
tail -f logs/csrd-app.log | jq

# Example log entry
{
  "timestamp": "2025-11-08T10:30:00.123Z",
  "level": "INFO",
  "logger": "csrd.health",
  "message": "Health check performed",
  "request_id": "req-abc123",
  "esrs_standard": "E1",
  "company_id": "comp-123",
  "service": "csrd-reporting-platform",
  "environment": "production"
}
```

### Configure Log Aggregation (Optional)

**ELK Stack:**

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/csrd/*.log
  json.keys_under_root: true
  json.add_error_key: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
```

**CloudWatch (AWS):**

```python
import watchtower
import logging

logger = logging.getLogger()
logger.addHandler(watchtower.CloudWatchLogHandler(
    log_group='/csrd/production',
    stream_name='csrd-app'
))
```

---

## Step 12: Verify Sentry Integration

1. Trigger a test error:

```python
from backend.error_tracking import capture_exception, set_esrs_context

try:
    set_esrs_context(esrs_standard="E1", company_id="test-123")
    raise ValueError("Test error for Sentry")
except Exception as e:
    capture_exception(e, tags={"test": "true"})
```

2. Check Sentry dashboard:
   - Go to your Sentry project
   - Verify error appears with ESRS context
   - Check breadcrumbs and stack trace

---

## Troubleshooting

### Health Checks Return 503

**Problem:** `/health/ready` returns 503

**Solution:**
1. Check database connection
2. Verify Redis is running
3. Check disk space
4. Review logs for specific failure

### Metrics Not Appearing

**Problem:** Metrics not showing in Prometheus

**Solution:**
1. Verify `/metrics` endpoint is accessible
2. Check Prometheus targets: http://localhost:9090/targets
3. Verify scrape configuration in prometheus.yml
4. Check network connectivity

### Alerts Not Firing

**Problem:** Expected alerts not received

**Solution:**
1. Check alert rules are loaded: http://localhost:9090/rules
2. Verify metric values exceed thresholds
3. Check Alertmanager configuration
4. Verify notification channels (Slack, email, etc.)

### Dashboard Shows No Data

**Problem:** Grafana dashboard panels are empty

**Solution:**
1. Verify Prometheus data source is configured
2. Check time range (last 6 hours by default)
3. Verify metrics are being collected
4. Check dashboard variable filters

---

## Next Steps

1. **Customize Thresholds**
   - Review alert thresholds in `monitoring/alerts/alerts-csrd.yml`
   - Adjust based on your specific requirements

2. **Configure Notifications**
   - Set up Slack webhook
   - Configure PagerDuty integration
   - Set up email alerts

3. **Add Custom Dashboards**
   - Create team-specific dashboards
   - Add business-specific panels

4. **Implement Runbooks**
   - Create runbooks for common alerts
   - Document remediation procedures

5. **Train Team**
   - Dashboard walkthrough
   - Alert response procedures
   - Incident management process

---

## Resources

- **Documentation:** `MONITORING.md`
- **Implementation Summary:** `MONITORING_IMPLEMENTATION_SUMMARY.md`
- **Health Checks:** `backend/health.py`
- **Metrics:** `backend/metrics.py`
- **Logging:** `backend/logging_config.py`
- **Error Tracking:** `backend/error_tracking.py`
- **Dashboard:** `monitoring/grafana-csrd-dashboard.json`
- **Alerts:** `monitoring/alerts/alerts-csrd.yml`

---

## Support

For questions or issues:
- **Team:** Team B3 - GL-CSRD Monitoring & Observability
- **Documentation:** See `MONITORING.md`
- **Slack:** #csrd-monitoring

---

**Setup Complete!** 🎉

Your CSRD Reporting Platform now has production-grade monitoring and observability.
