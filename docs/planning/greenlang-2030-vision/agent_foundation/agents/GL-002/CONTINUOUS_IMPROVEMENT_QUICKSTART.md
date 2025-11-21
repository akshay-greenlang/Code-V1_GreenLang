# GL-002 Continuous Improvement - Quick Start Guide

Get the feedback and experimentation system running in 15 minutes.

## Prerequisites

- Python 3.9+
- PostgreSQL 13+
- Redis 6+
- Docker (optional)

## Step 1: Install Dependencies

```bash
cd GreenLang_2030/agent_foundation/agents/GL-002

# Install Python packages
pip install -r requirements.txt

# Additional dependencies for continuous improvement
pip install scipy numpy prometheus-client redis asyncpg
```

## Step 2: Setup Database

```bash
# Create database
createdb gl002_continuous_improvement

# Run migrations
psql -d gl002_continuous_improvement -f migrations/001_create_feedback_tables.sql
psql -d gl002_continuous_improvement -f migrations/002_create_experiment_tables.sql

# Verify tables created
psql -d gl002_continuous_improvement -c "\dt"
```

## Step 3: Start Redis

```bash
# Using Docker
docker run -d -p 6379:6379 redis:7-alpine

# Or install locally
redis-server
```

## Step 4: Test Feedback System

```python
# test_feedback.py
import asyncio
from feedback import FeedbackCollector, OptimizationFeedback, FeedbackCategory

async def test_feedback():
    # Initialize
    collector = FeedbackCollector(
        db_url="postgresql://localhost/gl002_continuous_improvement"
    )
    await collector.initialize()

    # Submit feedback
    feedback = OptimizationFeedback(
        optimization_id="test_opt_001",
        rating=5,
        comment="Great results! Efficiency improved by 8%",
        actual_savings=2500.0,
        predicted_savings=2400.0,
        category=FeedbackCategory.SAVINGS,
        user_id="test_user"
    )

    result = await collector.collect_feedback(feedback)
    print(f"✅ Feedback submitted: {result}")

    # Get stats
    stats = await collector.get_stats(days=30)
    print(f"📊 Stats: Avg Rating={stats.average_rating}, NPS={stats.nps_score}")

    await collector.close()

asyncio.run(test_feedback())
```

Run test:
```bash
python test_feedback.py
```

## Step 5: Create Your First Experiment

```python
# test_experiment.py
import asyncio
from experiments import ExperimentManager
from experiments.example_experiments import COMBUSTION_ALGORITHM_TEST

async def test_experiment():
    # Initialize
    manager = ExperimentManager(
        db_url="postgresql://localhost/gl002_continuous_improvement",
        redis_url="redis://localhost:6379/0"
    )
    await manager.initialize()

    # Create experiment
    experiment = await manager.create_experiment(
        **COMBUSTION_ALGORITHM_TEST
    )
    print(f"✅ Experiment created: {experiment.experiment_id}")

    # Start experiment
    await manager.start_experiment(experiment.experiment_id)
    print(f"🚀 Experiment started!")

    # Simulate user assignment
    variant = await manager.traffic_router.get_variant(
        experiment.experiment_id,
        "user_123"
    )
    print(f"👤 User assigned to variant: {variant}")

    # Record metric
    await manager.record_metric(
        experiment_id=experiment.experiment_id,
        user_id="user_123",
        metric_name="energy_savings_kwh",
        metric_value=2500.0
    )
    print(f"📈 Metric recorded")

    await manager.close()

asyncio.run(test_experiment())
```

Run test:
```bash
python test_experiment.py
```

## Step 6: Start API Server

```python
# app.py
from fastapi import FastAPI
from feedback.feedback_api import create_feedback_router

app = FastAPI(title="GL-002 Continuous Improvement API")

# Add feedback router
feedback_router = create_feedback_router(
    db_url="postgresql://localhost/gl002_continuous_improvement"
)
app.include_router(feedback_router, prefix="/api/v1/feedback", tags=["feedback"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Start server:
```bash
python app.py
```

Test API:
```bash
# Submit feedback via API
curl -X POST "http://localhost:8000/api/v1/feedback/optimization/test_opt_001" \
  -H "Content-Type: application/json" \
  -d '{
    "rating": 5,
    "comment": "Excellent!",
    "actual_savings": 2500.0,
    "predicted_savings": 2400.0,
    "category": "savings",
    "user_id": "user_123",
    "metadata": {}
  }'

# Get stats
curl "http://localhost:8000/api/v1/feedback/stats?days=30"
```

## Step 7: Start Metrics Collection

```python
# metrics_server.py
import asyncio
from prometheus_client import start_http_server
from monitoring.feedback_metrics import start_metrics_collector

async def main():
    # Start Prometheus metrics endpoint
    start_http_server(9090)
    print("📊 Metrics server started on :9090/metrics")

    # Start metrics collection
    await start_metrics_collector(
        db_url="postgresql://localhost/gl002_continuous_improvement",
        collection_interval=60
    )

asyncio.run(main())
```

Start metrics:
```bash
python metrics_server.py
```

Check metrics:
```bash
curl http://localhost:9090/metrics | grep gl002
```

## Step 8: Setup Grafana Dashboard

```bash
# Import dashboard
curl -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -u admin:admin \
  --data @monitoring/grafana/feedback_dashboard.json
```

Access dashboard: http://localhost:3000

## Step 9: Run Automated Analysis

```python
# weekly_insights.py
import asyncio
from analysis import FeedbackAnalyzer

async def generate_insights():
    analyzer = FeedbackAnalyzer(
        db_url="postgresql://localhost/gl002_continuous_improvement"
    )
    await analyzer.initialize()

    # Generate weekly insights
    insights = await analyzer.generate_weekly_insights()

    print(f"\n📊 Week {insights['week_number']} Insights:")
    print(f"Average Rating: {insights['summary']['avg_rating']}")
    print(f"Trend: {insights['summary']['trend']}")
    print(f"\n🎯 Recommendations:")
    for rec in insights['recommendations']:
        print(f"  - {rec}")

    await analyzer.close()

asyncio.run(generate_insights())
```

Run analysis:
```bash
python weekly_insights.py
```

## Step 10: Verify Everything Works

```bash
# Check database
psql -d gl002_continuous_improvement -c "SELECT COUNT(*) FROM optimization_feedback;"
psql -d gl002_continuous_improvement -c "SELECT COUNT(*) FROM experiments;"

# Check Redis
redis-cli KEYS "experiment:*"

# Check API
curl http://localhost:8000/api/v1/feedback/health

# Check metrics
curl http://localhost:9090/metrics | grep gl002_feedback_total
```

## Environment Configuration

Create `.env` file:

```bash
# Database
DB_URL=postgresql://localhost/gl002_continuous_improvement
REDIS_URL=redis://localhost:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8000

# Metrics
PROMETHEUS_PORT=9090
METRICS_INTERVAL=60

# Alerts
ALERT_THRESHOLD_RATING=2.5
ALERT_THRESHOLD_ACCURACY=75.0
ALERT_EMAIL=team@greenlang.com

# Experiments
MIN_SAMPLE_SIZE=100
SIGNIFICANCE_LEVEL=0.05
```

Load environment:
```bash
export $(cat .env | xargs)
```

## Docker Quick Start

```bash
# Build image
docker-compose build

# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Next Steps

1. **Read full documentation:** `CONTINUOUS_IMPROVEMENT.md`
2. **Review example experiments:** `experiments/example_experiments.py`
3. **Customize Grafana dashboard:** `monitoring/grafana/feedback_dashboard.json`
4. **Set up scheduled analysis:** Add cron job for weekly reports
5. **Configure alerts:** Set up email/Slack notifications
6. **Deploy to production:** Follow deployment guide in main docs

## Common Issues

### Issue: Database connection failed
**Solution:** Check PostgreSQL is running and credentials are correct
```bash
psql -d gl002_continuous_improvement -c "SELECT 1;"
```

### Issue: Redis connection refused
**Solution:** Ensure Redis is running
```bash
redis-cli ping  # Should return "PONG"
```

### Issue: Metrics not updating
**Solution:** Check metrics collector is running
```bash
ps aux | grep metrics_server
```

### Issue: Experiment variant not assigned
**Solution:** Verify experiment is in RUNNING status
```sql
SELECT experiment_id, status FROM experiments;
```

## Support

- **Documentation:** `CONTINUOUS_IMPROVEMENT.md`
- **API Docs:** http://localhost:8000/docs
- **Example Code:** `experiments/example_experiments.py`
- **Issue Tracker:** GitHub issues

---

**You're all set!** 🎉

The continuous improvement system is now running. Start collecting feedback and running experiments to optimize GL-002 performance.
