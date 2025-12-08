# GreenLang 5-Minute Quick Start

## Overview

Get GreenLang running in under 5 minutes with this minimal viable deployment guide. This quick start is ideal for:

- Evaluating GreenLang capabilities
- Development and testing environments
- Proof-of-concept demonstrations
- Learning the platform basics

**Prerequisites:**
- Docker and Docker Compose installed
- 4 GB RAM available
- 10 GB disk space
- Internet connection (for pulling images)

---

## Step 1: Download and Configure (1 minute)

### Clone the Repository

```bash
# Clone GreenLang
git clone https://github.com/greenlang/greenlang-quickstart.git
cd greenlang-quickstart

# Or download directly
curl -L https://github.com/greenlang/greenlang-quickstart/archive/main.tar.gz | tar xz
cd greenlang-quickstart-main
```

### Create Environment File

```bash
# Copy example environment
cp .env.example .env

# The defaults work for quick start - no edits needed
# Optional: View/edit configuration
# nano .env
```

**Default Configuration:**

```env
# .env defaults (no changes required for quick start)
GREENLANG_PORT=8000
GREENLANG_DASHBOARD_PORT=8080
POSTGRES_PASSWORD=quickstart123
REDIS_PASSWORD=quickstart123
JWT_SECRET=quickstart-jwt-secret
```

---

## Step 2: Start GreenLang (2 minutes)

### Launch Services

```bash
# Start all services
docker-compose up -d

# Watch startup progress
docker-compose logs -f
# Press Ctrl+C when you see "GreenLang API ready"
```

### Verify Services

```bash
# Check all services are running
docker-compose ps

# Expected output:
# NAME                    STATUS    PORTS
# greenlang-api           running   0.0.0.0:8000->8000/tcp
# greenlang-dashboard     running   0.0.0.0:8080->8080/tcp
# greenlang-agent         running
# greenlang-ml            running
# greenlang-postgres      running   5432/tcp
# greenlang-redis         running   6379/tcp
```

### Quick Health Check

```bash
# API health check
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","version":"1.0.0","timestamp":"2025-12-07T10:00:00Z"}
```

---

## Step 3: Configure Your First Agent (1 minute)

### Access the Dashboard

1. Open your browser
2. Navigate to: **http://localhost:8080**
3. Login with default credentials:
   - **Username:** admin
   - **Password:** admin123

### Create a Demo Agent

**Option A: Using the UI**

1. Click **"Agents"** in the sidebar
2. Click **"+ New Agent"**
3. Select **"Process Heat Demo"** template
4. Click **"Create"**

**Option B: Using the CLI**

```bash
# Create demo agent via API
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(curl -s -X POST http://localhost:8000/api/v1/auth/token \
      -d 'username=admin&password=admin123' | jq -r '.access_token')" \
  -d '{
    "name": "demo_furnace",
    "type": "process_heat_agent",
    "template": "demo",
    "config": {
      "simulation_mode": true,
      "update_interval": 1000
    }
  }'
```

**Option C: Using the GreenLang CLI**

```bash
# Install CLI (if not included in Docker)
pip install greenlang-cli

# Login
greenlang-cli login --url http://localhost:8000 --username admin --password admin123

# Create agent from template
greenlang-cli agent create --template demo --name demo_furnace
```

---

## Step 4: Basic Monitoring Setup (1 minute)

### View Dashboard

1. Return to the dashboard: **http://localhost:8080**
2. Click **"Overview"** to see the demo agent
3. You should see:
   - Simulated temperature readings
   - Basic trends chart
   - Agent status indicators

### Monitor in Real-Time

```
+--------------------------------------------------+
| Demo Furnace Dashboard                            |
+--------------------------------------------------+
|                                                  |
|   Temperature    Fuel Flow      Status          |
|   +--------+    +--------+    +--------+        |
|   |  847C  |    | 95.2%  |    |   OK   |        |
|   +--------+    +--------+    +--------+        |
|                                                  |
|   Temperature Trend (Last Hour)                 |
|   860 |          *                              |
|   850 |----*--*-----*--*--*----------------------|
|   840 |  *              *                        |
|       +----------------------------------------- |
|                                                  |
|   [Alarms: 0 Active]  [ML: Running]             |
+--------------------------------------------------+
```

### View Logs

```bash
# View agent logs
docker-compose logs -f greenlang-agent

# View ML engine logs
docker-compose logs -f greenlang-ml
```

---

## Quick Verification Steps

### Test 1: API Response

```bash
# Get agent status
curl http://localhost:8000/api/v1/agents/demo_furnace/status

# Expected: JSON with current readings and status
```

### Test 2: Dashboard Access

1. Navigate to http://localhost:8080
2. Verify login works
3. Confirm data is updating in real-time

### Test 3: ML Predictions

```bash
# Get temperature prediction
curl http://localhost:8000/api/v1/agents/demo_furnace/predict?horizon=3600

# Expected: Prediction with confidence interval
```

### Test 4: Alarm Simulation

```bash
# Trigger a test alarm
curl -X POST http://localhost:8000/api/v1/test/alarm \
  -H "Content-Type: application/json" \
  -d '{"agent": "demo_furnace", "type": "high_temperature"}'

# Check dashboard for alarm notification
```

---

## What You Now Have

Your quick start deployment includes:

| Component | Description | Access |
|-----------|-------------|--------|
| API Server | REST API for all operations | http://localhost:8000 |
| Dashboard | Web-based operator interface | http://localhost:8080 |
| Demo Agent | Simulated process heat agent | Via dashboard |
| ML Engine | Prediction and anomaly detection | Automatic |
| Database | PostgreSQL with TimescaleDB | Internal |
| Cache | Redis for performance | Internal |

---

## Next Steps

### Explore the Platform

- **Dashboard Tour:** Click through all menu items
- **Agent Config:** Modify agent parameters and see effects
- **Alarms:** Create and test alarm configurations
- **Reports:** Generate a sample report

### Connect Real Data

Ready to connect to real equipment? See:
- [OPC-UA Integration Guide](../integrations/opc_ua.md)
- [Modbus Integration Guide](../integrations/modbus.md)
- [MQTT Integration Guide](../integrations/mqtt.md)

### Production Deployment

For production environments, see:
- [Production Deployment Guide](./production_deployment.md)
- [Security Hardening](../security/hardening.md)
- [High Availability Setup](../deployment/high_availability.md)

---

## Common Issues

### Issue: Services Won't Start

```bash
# Check Docker resources
docker system info

# Free up resources if needed
docker system prune -a

# Restart
docker-compose down
docker-compose up -d
```

### Issue: Can't Access Dashboard

```bash
# Check port availability
netstat -an | grep 8080

# If port in use, edit .env
# GREENLANG_DASHBOARD_PORT=8081
# Then restart
docker-compose down
docker-compose up -d
```

### Issue: Login Failed

```bash
# Reset admin password
docker-compose exec greenlang-api greenlang-cli user reset-password admin --password newpassword
```

### Issue: Agent Not Showing Data

```bash
# Check agent logs
docker-compose logs greenlang-agent

# Restart agent
docker-compose restart greenlang-agent
```

---

## Cleanup

When you're done:

```bash
# Stop all services
docker-compose down

# Remove all data (full cleanup)
docker-compose down -v

# Remove images (complete removal)
docker-compose down -v --rmi all
```

---

## Getting Help

- **Documentation:** https://docs.greenlang.io
- **Community Forum:** https://community.greenlang.io
- **GitHub Issues:** https://github.com/greenlang/greenlang/issues
- **Email Support:** support@greenlang.io

---

**Congratulations!** You have GreenLang running. Explore the platform and proceed to the [Production Deployment Guide](./production_deployment.md) when you're ready for a real deployment.

---

*Quick Start Version: 1.0.0*
*Last Updated: December 2025*
