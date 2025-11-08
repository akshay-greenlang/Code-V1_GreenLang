# CBAM Monitoring - Quick Start Guide

**Get production-grade monitoring running in 5 minutes!**

---

## Prerequisites

- Docker and Docker Compose installed
- Python 3.9+ (for the CBAM application)

---

## Option 1: CLI-Only Mode (Current Setup)

The CBAM application currently runs as a CLI tool. You can still collect metrics by pushing them to Prometheus Pushgateway.

### Step 1: Install Dependencies

```bash
pip install prometheus-client psutil
```

### Step 2: Test Health Checks

```bash
cd C:\Users\aksha\Code-V1_GreenLang\GL-CBAM-APP\CBAM-Importer-Copilot
python backend/health.py --check all --json
```

### Step 3: Test Metrics Collection

```bash
python backend/metrics.py
```

### Step 4: Start Monitoring Stack

```bash
cd monitoring
docker-compose up -d
```

### Step 5: Push Metrics from CLI

```python
from backend.metrics import CBAMMetrics, MetricsExporter

# Create metrics
metrics = CBAMMetrics()

# Record some metrics
metrics.record_pipeline_execution("success", 45.5, "total")

# Push to Pushgateway
exporter = MetricsExporter(metrics)
exporter.push_to_gateway("http://localhost:9091", job="cbam-cli")
```

### Step 6: View Metrics

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

---

## Option 2: Web API Mode (Recommended for Production)

Deploy CBAM as a web service with full monitoring integration.

### Step 1: Install Web Dependencies

```bash
pip install fastapi uvicorn[standard] prometheus-client psutil
```

### Step 2: Start the CBAM API

```bash
cd C:\Users\aksha\Code-V1_GreenLang\GL-CBAM-APP\CBAM-Importer-Copilot
python backend/app.py
```

Or with uvicorn:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### Step 3: Verify Health Checks

```bash
# Basic health
curl http://localhost:8000/health

# Readiness
curl http://localhost:8000/health/ready

# Liveness
curl http://localhost:8000/health/live
```

### Step 4: View Metrics

```bash
curl http://localhost:8000/metrics
```

### Step 5: Start Monitoring Stack

```bash
cd monitoring
docker-compose up -d
```

### Step 6: Access Dashboards

- **CBAM API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Alertmanager**: http://localhost:9093

### Step 7: Import Grafana Dashboard

1. Open Grafana: http://localhost:3000
2. Login: admin/admin
3. Go to Dashboards → Import
4. Upload `monitoring/grafana-dashboard.json`
5. Select Prometheus datasource
6. Click Import

---

## Quick Verification

### Check Services

```bash
# Check Docker containers
docker-compose ps

# Should show:
# - cbam-prometheus (9090)
# - cbam-grafana (3000)
# - cbam-alertmanager (9093)
# - cbam-node-exporter (9100)
# - cbam-pushgateway (9091)
# - cbam-blackbox-exporter (9115)
```

### Check Health Endpoints

```bash
# Test all health checks
curl http://localhost:8000/health | jq
curl http://localhost:8000/health/ready | jq
curl http://localhost:8000/health/live | jq
```

### Check Metrics

```bash
# View Prometheus metrics
curl http://localhost:8000/metrics | grep cbam_

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq
```

---

## Configure Logging

### Development

```python
from backend.logging_config import configure_development_logging

# Human-readable logs
logger = configure_development_logging()
```

### Production

```python
from backend.logging_config import configure_production_logging

# JSON logs with rotation
logger = configure_production_logging(log_dir="/var/log/cbam")
```

### Using Structured Logging

```python
from backend.logging_config import StructuredLogger

logger = StructuredLogger("cbam.pipeline")

# Log with context
logger.info(
    "Processing shipment",
    shipment_id="S-12345",
    quantity=100,
    supplier="ACME Corp"
)

# Track operations with timing
with logger.operation("calculate_emissions", shipment_count=1000):
    # ... do work ...
    pass
```

---

## Integration with Existing Pipeline

### Update cbam_pipeline.py

```python
from backend.metrics import CBAMMetrics
from backend.logging_config import StructuredLogger

# Initialize at top of file
metrics = CBAMMetrics()
logger = StructuredLogger("cbam.pipeline")

# In CBAMPipeline class
def run(self, ...):
    start_time = time.time()

    try:
        # ... existing pipeline code ...

        # Record success metrics
        duration = time.time() - start_time
        metrics.record_pipeline_execution("success", duration, "total")

        logger.info("Pipeline completed successfully", duration=duration)

        return final_report

    except Exception as e:
        duration = time.time() - start_time
        metrics.record_pipeline_execution("failed", duration, "total")
        metrics.record_exception(e)

        logger.error("Pipeline failed", exception=e)

        raise
```

---

## Troubleshooting

### Metrics Not Showing in Prometheus

1. Check if CBAM app is exposing metrics:
   ```bash
   curl http://localhost:8000/metrics
   ```

2. Check Prometheus targets:
   ```bash
   curl http://localhost:9090/api/v1/targets | jq
   ```

3. Verify Prometheus can reach CBAM:
   ```bash
   docker exec cbam-prometheus wget -O- http://host.docker.internal:8000/metrics
   ```

### Grafana Dashboard Not Loading

1. Verify Prometheus datasource:
   - Configuration → Data Sources → Prometheus
   - Test connection

2. Check dashboard JSON format:
   - Ensure valid JSON
   - Check datasource UID matches

### Docker Containers Not Starting

```bash
# View logs
docker-compose logs -f

# Restart specific service
docker-compose restart prometheus

# Rebuild and restart
docker-compose down
docker-compose up -d --build
```

---

## Next Steps

1. **Configure Alertmanager**
   - Edit `monitoring/alertmanager.yml`
   - Set up email/Slack/PagerDuty
   - Test alerts: `curl -X POST http://localhost:9093/-/reload`

2. **Customize Dashboard**
   - Modify `monitoring/grafana-dashboard.json`
   - Add business-specific metrics
   - Create custom alerts

3. **Production Deployment**
   - Review `MONITORING.md` for production setup
   - Configure Kubernetes health probes
   - Set up log aggregation (ELK/CloudWatch)
   - Enable TLS/SSL
   - Configure authentication

4. **Error Tracking (Optional)**
   - Uncomment `sentry-sdk` in requirements.txt
   - Add Sentry integration
   - Configure DSN

---

## Resources

- **Full Documentation**: `MONITORING.md`
- **Alert Rules**: `monitoring/alerts.yml`
- **Prometheus Config**: `monitoring/prometheus.yml`
- **Health Checks**: `backend/health.py`
- **Metrics**: `backend/metrics.py`
- **Logging**: `backend/logging_config.py`

---

## Support

If you encounter issues:

1. Check logs: `docker-compose logs -f`
2. Verify network connectivity
3. Review `MONITORING.md` troubleshooting section
4. Test health endpoints individually
5. Check Prometheus targets status

---

**Monitoring ready! Start monitoring your CBAM pipelines in production!**
