# GreenLang Monitoring - Quick Start Guide

Get the monitoring system up and running in 15 minutes.

## Prerequisites

- Python 3.10+
- Prometheus (with Pushgateway)
- Grafana
- Redis
- PostgreSQL

## Step 1: Install Dependencies (2 minutes)

```bash
cd C:\Users\aksha\Code-V1_GreenLang\greenlang\monitoring
pip install -r requirements.txt
```

## Step 2: Configure Environment (3 minutes)

Create `.env` file:

```bash
# Grafana
GRAFANA_API_KEY=your-grafana-api-key-here

# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# PagerDuty
PAGERDUTY_INTEGRATION_KEY=your-pagerduty-key

# Email
EMAIL_USERNAME=your-email@gmail.com
EMAIL_PASSWORD=your-app-password

# API
API_KEY_PROD=greenlang-prod-key-change-me
API_KEY_DEV=greenlang-dev-key-change-me
```

Update `config.yaml` with your service URLs if needed.

## Step 3: Generate Dashboards (2 minutes)

```bash
# Generate all dashboard JSON files
python dashboards/infrastructure_usage.py
python dashboards/cost_savings.py
python dashboards/performance.py
python dashboards/compliance.py
python dashboards/productivity.py
python dashboards/health.py
```

Dashboards will be saved as `.json` files in the dashboards directory.

## Step 4: Import to Grafana (3 minutes)

1. Open Grafana: `http://localhost:3000`
2. Login (default: admin/admin)
3. Go to Dashboards → Import
4. Upload each `.json` file from `dashboards/` directory
5. Select "Prometheus" as datasource
6. Click Import

You should now have 6 dashboards:
- Infrastructure Usage Metrics
- Cost Savings & ROI
- Performance Monitoring
- Compliance & Quality
- Developer Productivity
- Infrastructure Health

## Step 5: Deploy Alert Rules (2 minutes)

```bash
# Generate alert rules
python alerts/alert_rules.py
```

This creates:
- `alerts/prometheus_rules.json` - Copy to Prometheus config
- `alerts/grafana_alerts.json` - Import to Grafana alerts

## Step 6: Start Analytics API (1 minute)

```bash
# Start the API server
python api/analytics_api.py
```

API will be available at:
- Base URL: `http://localhost:8080`
- API Docs: `http://localhost:8080/docs`
- Metrics: `http://localhost:8080/metrics`

Test with:
```bash
curl -H "X-API-Key: greenlang-prod-key-change-me" http://localhost:8080/api/health
```

## Step 7: Test Data Collection (2 minutes)

Run collectors once to verify they work:

```bash
# Test metrics collector
python collectors/metrics_collector.py

# Test health checker
python collectors/health_checker.py

# Test log aggregator
python collectors/log_aggregator.py

# Test violation scanner
python collectors/violation_scanner.py
```

Check Prometheus for metrics: `http://localhost:9090/targets`

## Step 8: Set Up Cron Jobs (Optional)

For Linux/Mac, add to crontab:

```bash
crontab -e
```

Add these lines:

```cron
*/5 * * * * cd /path/to/greenlang && python monitoring/collectors/metrics_collector.py
*/5 * * * * cd /path/to/greenlang && python monitoring/collectors/log_aggregator.py
* * * * * cd /path/to/greenlang && python monitoring/collectors/health_checker.py
0 * * * * cd /path/to/greenlang && python monitoring/collectors/violation_scanner.py
```

For Windows, use Task Scheduler instead.

## Quick Test

Verify everything works:

### 1. Check Dashboards
Visit each dashboard URL:
- `http://localhost:3000/d/greenlang-ium`
- `http://localhost:3000/d/greenlang-cost-savings`
- `http://localhost:3000/d/greenlang-performance`
- `http://localhost:3000/d/greenlang-compliance`
- `http://localhost:3000/d/greenlang-productivity`
- `http://localhost:3000/d/greenlang-health`

### 2. Check API
```bash
# Test API endpoint
curl -H "X-API-Key: greenlang-prod-key-change-me" \
  http://localhost:8080/api/ium/overall

# Should return JSON with IUM data
```

### 3. Check Metrics in Prometheus
Visit `http://localhost:9090/graph` and query:
```
greenlang_ium_score
greenlang_cost_savings_usd
greenlang_service_up
```

## Common Issues

### Dashboards not showing data
- **Check**: Prometheus datasource configured in Grafana
- **Check**: Collectors have run at least once
- **Fix**: Run `python collectors/metrics_collector.py` manually

### API returns 403
- **Check**: API key in request header
- **Fix**: Use `-H "X-API-Key: your-key"` in curl

### Alerts not firing
- **Check**: Alert rules imported to Prometheus
- **Check**: Alertmanager configured
- **Fix**: Copy `prometheus_rules.json` to Prometheus config dir

### Services showing as DOWN
- **Check**: Services actually running
- **Check**: Service URLs correct in `config.yaml`
- **Fix**: Update URLs and restart health checker

## Next Steps

1. **Customize Thresholds**: Edit `config.yaml` to adjust IUM targets, SLA thresholds
2. **Configure Alerts**: Set up Slack/Email/PagerDuty webhooks
3. **Schedule Reports**: Configure weekly/monthly report generation
4. **Train Team**: Share dashboard URLs and API documentation

## Getting Help

- **Documentation**: See `README.md` for detailed docs
- **Deployment Report**: See `DEPLOYMENT_REPORT.md` for complete specs
- **Issues**: Create issue in GitHub repository
- **Email**: platform-team@greenlang.io

## Success Checklist

- [ ] All dependencies installed
- [ ] Environment variables configured
- [ ] 6 dashboards imported to Grafana
- [ ] Alert rules deployed
- [ ] Analytics API running
- [ ] Data collectors tested
- [ ] Metrics visible in Prometheus
- [ ] Dashboards showing data
- [ ] API endpoints responding
- [ ] Alerts configured

If all checkboxes are checked, you're ready for production!

---

**Time to Complete**: ~15 minutes
**Difficulty**: Easy
**Support**: platform-team@greenlang.io
