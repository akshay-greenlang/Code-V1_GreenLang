# GL-001 ProcessHeatOrchestrator - Quick Start Guide

**5-Minute Setup**: Get GL-001 running in development mode

---

## Prerequisites

- Python 3.11+
- PostgreSQL 14+ (with TimescaleDB extension)
- Redis 7+
- Git

---

## 1. Install Dependencies (2 minutes)

```bash
cd GreenLang_2030/agent_foundation/agents/GL-001

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest pytest-asyncio pytest-cov black isort mypy bandit
```

---

## 2. Configure Environment (2 minutes)

```bash
# Copy template
cp .env.template .env

# Edit critical variables (MINIMUM for development)
nano .env
```

**Minimum Required Variables**:

```bash
# Database (local PostgreSQL)
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=greenlang_dev
DATABASE_USER=postgres
DATABASE_PASSWORD=your_password_here

# Redis (local)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# AI/LLM (for classification only)
ANTHROPIC_API_KEY=sk-ant-api03-...your-key-here...

# Development mode
GL_001_ENVIRONMENT=development
DEBUG_MODE=true
```

**Optional** - Skip SCADA/ERP for local testing:
```bash
SCADA_PLANT1_ENABLED=false
ERP_SAP_ENABLED=false
MOCK_SCADA_ENABLED=true
MOCK_ERP_ENABLED=true
```

---

## 3. Initialize Database (1 minute)

```bash
# Start PostgreSQL (if not running)
# macOS: brew services start postgresql
# Linux: sudo systemctl start postgresql
# Windows: net start postgresql-x64-14

# Create database
psql -U postgres -c "CREATE DATABASE greenlang_dev;"

# Enable TimescaleDB (optional)
psql -U postgres -d greenlang_dev -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# Run migrations (if available)
# alembic upgrade head
```

---

## 4. Run Tests (Optional - 1 minute)

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=term

# Run specific test
pytest tests/test_process_heat_orchestrator.py -v
```

---

## 5. Start Application (30 seconds)

```bash
# Development mode
python -m uvicorn process_heat_orchestrator:app --reload --host 0.0.0.0 --port 8000

# Or run directly
python example_usage.py
```

**API Available at**: `http://localhost:8000`

**Metrics**: `http://localhost:9090/metrics`

**Health Check**: `http://localhost:8000/api/v1/health`

---

## 6. Install Pre-commit Hooks (Optional - 30 seconds)

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually (first time will download dependencies)
pre-commit run --all-files
```

---

## Production Deployment

See `DEPLOYMENT_GUIDE.md` for:
- Kubernetes deployment
- Docker containerization
- Multi-plant SCADA integration
- ERP system integration
- High availability setup

---

## Common Issues

### Issue: Database connection failed
**Solution**: Check PostgreSQL is running and credentials in `.env` are correct
```bash
psql -U postgres -c "SELECT version();"
```

### Issue: Redis connection timeout
**Solution**: Start Redis server
```bash
# macOS: brew services start redis
# Linux: sudo systemctl start redis
# Windows: redis-server.exe
```

### Issue: Import errors
**Solution**: Ensure virtual environment is activated and dependencies installed
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Pre-commit hooks failing
**Solution**: Run auto-fix
```bash
pre-commit run --all-files
```

---

## Development Workflow

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**
   - Edit code
   - Pre-commit hooks run automatically on `git commit`

3. **Run tests**
   ```bash
   pytest tests/ -v --cov=.
   ```

4. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: your feature description"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

---

## Configuration Reference

### Full Configuration
See `.env.template` for all 80+ environment variables

### Key Sections
- **Database**: PostgreSQL/TimescaleDB connection
- **Redis**: Caching and coordination
- **SCADA**: Multi-plant OPC UA/Modbus integration
- **ERP**: SAP/Oracle/Dynamics integration
- **Sub-Agents**: GL-002 through GL-005 coordination
- **Monitoring**: Prometheus, OpenTelemetry, logging
- **Security**: TLS, API keys, JWT tokens

---

## Next Steps

- Read `CONFIGURATION_FILES_MANIFEST.md` for detailed documentation
- Review `DEPLOYMENT_GUIDE.md` for production deployment
- Check `SECURITY_AUDIT_REPORT.md` for security best practices
- See `MONITORING.md` for observability setup

---

**Questions?** Contact engineering@greenlang.io
