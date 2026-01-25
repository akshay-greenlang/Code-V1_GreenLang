# GL-VCCI Load Testing Suite - Quick Start Guide

## 1. Install Dependencies (5 minutes)

```bash
cd tests/load
pip install -r requirements.txt
```

## 2. Verify Setup (1 minute)

```bash
# Check Locust version
locust --version

# Check application health
curl http://localhost:8000/health
```

## 3. Run Your First Test (10 minutes)

### Option A: Web UI (Recommended)
```bash
locust -f locustfile_rampup.py --host=http://localhost:8000
# Open browser: http://localhost:8089
# Set users=1000, spawn-rate=1.67
# Click "Start swarming"
```

### Option B: Headless (Automated)
```bash
locust -f locustfile_rampup.py --host=http://localhost:8000 \
       --users=1000 --spawn-rate=1.67 --run-time=10m \
       --headless --csv=rampup_results
```

## 4. Generate Report (2 minutes)

```bash
python generate_performance_report.py \
       --results rampup_results_stats.csv \
       --output rampup_report.html \
       --test-name "Ramp-Up Test"

# Open report in browser
open rampup_report.html
```

## Test Scenarios Available

| Scenario | File | Duration | Users | Command |
|----------|------|----------|-------|---------|
| **Ramp-Up** | `locustfile_rampup.py` | 10 min | 0→1K | See above |
| **Sustained** | `locustfile_sustained.py` | 1 hour | 1K | `--users=1000 --run-time=1h` |
| **Spike** | `locustfile_spike.py` | 25 min | 1K→5K→1K | `--run-time=25m` |
| **Endurance** | `locustfile_endurance.py` | 24 hours | 500 | `--users=500 --run-time=24h` |

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| p95 Latency | < 200ms | ✅ |
| Error Rate | < 0.1% | ✅ |
| Throughput | ≥ 1000 RPS | ✅ |
| CPU | < 70% | ✅ |
| Memory | Stable | ✅ |

## Common Commands

```bash
# Ramp-up test (10 min)
locust -f locustfile_rampup.py --host=http://localhost:8000 \
       --users=1000 --spawn-rate=1.67 --run-time=10m \
       --headless --csv=rampup

# Sustained load (1 hour)
locust -f locustfile_sustained.py --host=http://localhost:8000 \
       --users=1000 --spawn-rate=50 --run-time=1h \
       --headless --csv=sustained

# Spike test (25 min)
locust -f locustfile_spike.py --host=http://localhost:8000 \
       --headless --run-time=25m --csv=spike

# Endurance test (4 hours)
locust -f locustfile_endurance.py --host=http://localhost:8000 \
       --users=500 --spawn-rate=10 --run-time=4h \
       --headless --csv=endurance
```

## Troubleshooting

**Problem**: Connection refused
```bash
# Solution: Start the application first
curl http://localhost:8000/health
```

**Problem**: Authentication fails
```bash
# Solution: Create test users (loadtest_1@example.com, etc.)
# Check application user creation process
```

**Problem**: High error rates
```bash
# Solution: Reduce load
locust -f locustfile_rampup.py --host=http://localhost:8000 \
       --users=100 --spawn-rate=10 --run-time=5m
```

## Documentation

- **Full Guide**: See `README.md`
- **Delivery Report**: See `LOAD_TEST_SUITE_DELIVERY.md`
- **Locust Docs**: https://docs.locust.io/

## Support

Questions? Check the README.md or contact GL-VCCI team.
