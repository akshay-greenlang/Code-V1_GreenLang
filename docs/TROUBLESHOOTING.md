# GreenLang Troubleshooting Guide

**Version:** 1.0
**Last Updated:** November 2025
**Audience:** Developers, DevOps Engineers, System Administrators

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Database Connection Errors](#database-connection-errors)
3. [Agent Failures](#agent-failures)
4. [Calculation Errors](#calculation-errors)
5. [Performance Problems](#performance-problems)
6. [API Errors](#api-errors)
7. [Authentication Issues](#authentication-issues)
8. [Docker Issues](#docker-issues)
9. [Kubernetes Issues](#kubernetes-issues)
10. [Common Error Messages](#common-error-messages)
11. [Logging and Debugging](#logging-and-debugging)
12. [Getting Help](#getting-help)

---

## Installation Issues

### Issue 1: `pip install greenlang` Fails

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement greenlang
ERROR: No matching distribution found for greenlang
```

**Possible Causes:**
1. Python version too old
2. pip not up to date
3. Network issues
4. Typo in package name

**Solutions:**

**Solution 1: Check Python Version**
```bash
python --version  # Must be 3.9 or higher

# If too old, install Python 3.11
# Ubuntu/Debian
sudo apt install python3.11 python3.11-venv

# macOS
brew install python@3.11

# Windows
# Download from https://www.python.org/downloads/
```

**Solution 2: Update pip**
```bash
python -m pip install --upgrade pip
```

**Solution 3: Check Internet Connection**
```bash
# Test PyPI connectivity
curl -I https://pypi.org

# If behind proxy, configure pip
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
pip install greenlang
```

**Solution 4: Use Specific Index**
```bash
# Use PyPI mirror
pip install greenlang --index-url https://pypi.org/simple

# Or install from GitHub (development version)
pip install git+https://github.com/greenlang/greenlang.git
```

---

### Issue 2: ImportError: No module named 'greenlang'

**Symptoms:**
```python
>>> import greenlang
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'greenlang'
```

**Possible Causes:**
1. GreenLang not installed
2. Wrong Python environment
3. Virtual environment not activated

**Solutions:**

**Solution 1: Verify Installation**
```bash
pip list | grep greenlang

# If not found, install
pip install greenlang
```

**Solution 2: Check Python Path**
```bash
which python
# Should match where you installed GreenLang

# Check sys.path
python -c "import sys; print('\n'.join(sys.path))"
```

**Solution 3: Activate Virtual Environment**
```bash
# If using venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# If using conda
conda activate greenlang-env
```

**Solution 4: Reinstall in Current Environment**
```bash
pip uninstall greenlang
pip install greenlang
```

---

### Issue 3: Dependency Conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
package-x 1.0.0 requires pandas<2.0, but you have pandas 2.1.0 which is incompatible.
```

**Solutions:**

**Solution 1: Use Fresh Virtual Environment**
```bash
# Create new environment
python3.11 -m venv greenlang-env
source greenlang-env/bin/activate

# Install GreenLang
pip install greenlang
```

**Solution 2: Install with pip-tools**
```bash
pip install pip-tools
pip-compile requirements.txt
pip-sync requirements.txt
```

**Solution 3: Use Docker**
```bash
# Use official Docker image (no dependency issues)
docker run -it greenlang/greenlang:latest bash
```

---

### Issue 4: Permission Denied During Installation

**Symptoms:**
```
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied: '/usr/local/lib/python3.11/site-packages/greenlang'
```

**Solutions:**

**Solution 1: Use Virtual Environment (Recommended)**
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install greenlang
```

**Solution 2: User Installation**
```bash
pip install --user greenlang
```

**Solution 3: Use sudo (Not Recommended)**
```bash
sudo pip install greenlang
# Warning: Can break system Python packages
```

---

### Issue 5: SSL Certificate Verification Failed

**Symptoms:**
```
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solutions:**

**Solution 1: Update Certificates**
```bash
# macOS
/Applications/Python\ 3.11/Install\ Certificates.command

# Ubuntu/Debian
sudo apt install ca-certificates
sudo update-ca-certificates

# Windows
# Reinstall Python from python.org
```

**Solution 2: Temporary Workaround (Not Recommended)**
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org greenlang
```

---

## Database Connection Errors

### Issue 6: Cannot Connect to PostgreSQL

**Symptoms:**
```
sqlalchemy.exc.OperationalError: could not connect to server: Connection refused
    Is the server running on host "localhost" (127.0.0.1) and accepting
    TCP/IP connections on port 5432?
```

**Diagnosis:**

```bash
# 1. Check if PostgreSQL is running
# Linux
sudo systemctl status postgresql

# macOS
brew services list | grep postgresql

# Windows
# Check Services app for "postgresql" service

# 2. Test connection
psql -h localhost -U greenlang -d greenlang

# 3. Check port
sudo netstat -tlnp | grep 5432  # Linux
lsof -i :5432                   # macOS
netstat -an | findstr 5432      # Windows
```

**Solutions:**

**Solution 1: Start PostgreSQL**
```bash
# Linux
sudo systemctl start postgresql
sudo systemctl enable postgresql  # Start on boot

# macOS
brew services start postgresql@14

# Windows
# Start via Services app or:
pg_ctl -D "C:\Program Files\PostgreSQL\14\data" start
```

**Solution 2: Check Connection Settings**
```python
# config.py or .env
DATABASE_URL = "postgresql://user:password@host:port/database"

# Example:
DATABASE_URL = "postgresql://greenlang:mypassword@localhost:5432/greenlang"
```

**Solution 3: Check pg_hba.conf**
```bash
# Location: /etc/postgresql/14/main/pg_hba.conf (Linux)
# Or: /usr/local/var/postgres/pg_hba.conf (macOS)

# Add line:
host    all             all             0.0.0.0/0               md5

# Reload PostgreSQL
sudo systemctl reload postgresql
```

**Solution 4: Check Firewall**
```bash
# Allow PostgreSQL port
sudo ufw allow 5432/tcp          # Linux (UFW)
sudo firewall-cmd --add-port=5432/tcp --permanent  # Linux (firewalld)

# macOS
# System Preferences > Security & Privacy > Firewall > Firewall Options
```

**Solution 5: Use Docker PostgreSQL**
```bash
docker run -d \
  --name greenlang-postgres \
  -e POSTGRES_DB=greenlang \
  -e POSTGRES_USER=greenlang \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  postgres:14

# Connection string:
DATABASE_URL="postgresql://greenlang:password@localhost:5432/greenlang"
```

---

### Issue 7: Authentication Failed for User

**Symptoms:**
```
psycopg2.OperationalError: FATAL:  password authentication failed for user "greenlang"
FATAL:  no pg_hba.conf entry for host "192.168.1.100", user "greenlang", database "greenlang", SSL off
```

**Solutions:**

**Solution 1: Verify Credentials**
```bash
# Test manually
psql -h localhost -U greenlang -d greenlang

# If password is correct, update config
# .env
DATABASE_URL="postgresql://greenlang:correct_password@localhost:5432/greenlang"
```

**Solution 2: Reset PostgreSQL Password**
```sql
-- Connect as postgres superuser
sudo -u postgres psql

-- Reset password
ALTER USER greenlang WITH PASSWORD 'new_password';

-- Verify
\du
```

**Solution 3: Create User if Missing**
```sql
-- Connect as postgres
sudo -u postgres psql

-- Create user
CREATE USER greenlang WITH PASSWORD 'password';
CREATE DATABASE greenlang OWNER greenlang;
GRANT ALL PRIVILEGES ON DATABASE greenlang TO greenlang;
```

**Solution 4: Fix pg_hba.conf**
```bash
# Edit pg_hba.conf
sudo nano /etc/postgresql/14/main/pg_hba.conf

# Add lines (order matters):
# TYPE  DATABASE        USER            ADDRESS                 METHOD
host    greenlang       greenlang       127.0.0.1/32           md5
host    greenlang       greenlang       ::1/128                md5
host    all             all             0.0.0.0/0              md5

# Reload
sudo systemctl reload postgresql
```

---

### Issue 8: Database Does Not Exist

**Symptoms:**
```
psycopg2.OperationalError: FATAL:  database "greenlang" does not exist
```

**Solutions:**

**Solution 1: Create Database**
```bash
# Method 1: Using createdb
createdb -h localhost -U postgres greenlang

# Method 2: Using psql
psql -h localhost -U postgres -c "CREATE DATABASE greenlang;"

# Method 3: Using GreenLang CLI
greenlang db init
```

**Solution 2: Run Migrations**
```bash
# Initialize database schema
greenlang db upgrade

# Or with Alembic directly
alembic upgrade head
```

---

### Issue 9: Too Many Database Connections

**Symptoms:**
```
sqlalchemy.exc.OperationalError: FATAL:  sorry, too many clients already
FATAL:  remaining connection slots are reserved for non-replication superuser connections
```

**Diagnosis:**

```sql
-- Check current connections
SELECT count(*) FROM pg_stat_activity;

-- Check max connections
SHOW max_connections;

-- See active connections
SELECT pid, usename, application_name, client_addr, state
FROM pg_stat_activity
WHERE datname = 'greenlang';
```

**Solutions:**

**Solution 1: Increase max_connections**
```bash
# Edit postgresql.conf
sudo nano /etc/postgresql/14/main/postgresql.conf

# Change:
max_connections = 100  # Default
# To:
max_connections = 200

# Restart PostgreSQL
sudo systemctl restart postgresql
```

**Solution 2: Use Connection Pooling**
```python
# config.py
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,          # Number of persistent connections
    max_overflow=20,       # Max temporary connections
    pool_timeout=30,       # Connection timeout
    pool_recycle=3600      # Recycle connections after 1 hour
)
```

**Solution 3: Use PgBouncer**
```bash
# Install PgBouncer
sudo apt install pgbouncer

# Configure /etc/pgbouncer/pgbouncer.ini
[databases]
greenlang = host=localhost port=5432 dbname=greenlang

[pgbouncer]
listen_addr = 127.0.0.1
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25

# Update connection string
DATABASE_URL="postgresql://greenlang:password@localhost:6432/greenlang"
```

**Solution 4: Kill Idle Connections**
```sql
-- Kill idle connections older than 5 minutes
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'greenlang'
  AND state = 'idle'
  AND state_change < NOW() - INTERVAL '5 minutes';
```

---

### Issue 10: SQLite Database Locked

**Symptoms:**
```
sqlite3.OperationalError: database is locked
```

**Solutions:**

**Solution 1: Increase Timeout**
```python
# config.py
import sqlite3

DATABASE_URL = "sqlite:///greenlang.db?timeout=30"
```

**Solution 2: Use WAL Mode**
```python
from sqlalchemy import event
from sqlalchemy.engine import Engine

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()
```

**Solution 3: Switch to PostgreSQL (Recommended)**
```bash
# SQLite is not suitable for production or concurrent access
# Migrate to PostgreSQL:

# 1. Export SQLite data
sqlite3 greenlang.db .dump > dump.sql

# 2. Create PostgreSQL database
createdb greenlang

# 3. Import (may need manual fixes)
psql greenlang < dump.sql

# 4. Update config
DATABASE_URL="postgresql://greenlang:password@localhost:5432/greenlang"
```

---

## Agent Failures

### Issue 11: Agent Job Stuck in "Processing" State

**Symptoms:**
- Job status remains "processing" indefinitely
- No errors in logs
- Agent appears to be hanging

**Diagnosis:**

```bash
# Check job status
greenlang jobs status job_abc123

# Check agent logs
greenlang logs --agent calculator --job job_abc123

# Check Celery workers
celery -A greenlang.worker inspect active

# Check system resources
top
df -h
```

**Solutions:**

**Solution 1: Set Job Timeout**
```python
# config.py
AGENT_JOB_TIMEOUT = 3600  # 1 hour timeout

# Or per-agent
AGENT_TIMEOUTS = {
    'calculator': 600,      # 10 minutes
    'report': 300,          # 5 minutes
    'intake': 1800          # 30 minutes
}
```

**Solution 2: Restart Celery Workers**
```bash
# Find worker processes
ps aux | grep celery

# Graceful restart
celery -A greenlang.worker control shutdown

# Start workers
celery -A greenlang.worker worker --loglevel=info

# Or with systemd
sudo systemctl restart greenlang-worker
```

**Solution 3: Kill Stuck Job**
```bash
# Cancel job
greenlang jobs cancel job_abc123

# Or manually in database
psql greenlang -c "UPDATE jobs SET status='failed', error='Job timeout' WHERE job_id='job_abc123';"
```

**Solution 4: Check for Deadlocks**
```sql
-- Check for PostgreSQL locks
SELECT
    pid,
    usename,
    pg_blocking_pids(pid) as blocked_by,
    query
FROM pg_stat_activity
WHERE cardinality(pg_blocking_pids(pid)) > 0;

-- Kill blocking query
SELECT pg_terminate_backend(pid);
```

---

### Issue 12: Agent Fails with "Out of Memory"

**Symptoms:**
```
MemoryError: Unable to allocate array
Killed (signal 9)
```

**Diagnosis:**

```bash
# Check memory usage
free -h
top -o %MEM

# Check swap
swapon --show

# Check logs for OOM killer
dmesg | grep -i "out of memory"
sudo journalctl -xe | grep -i oom
```

**Solutions:**

**Solution 1: Increase Memory**
```bash
# For Docker
docker run -m 8g greenlang/greenlang:latest

# For Kubernetes
# In deployment.yaml:
resources:
  limits:
    memory: "8Gi"
  requests:
    memory: "4Gi"
```

**Solution 2: Process Data in Chunks**
```python
# Instead of loading entire file
data = pd.read_csv('large_file.csv')  # May OOM

# Use chunking
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process_chunk(chunk)
```

**Solution 3: Optimize Agent Code**
```python
# Use generators instead of lists
def process_records(records):
    for record in records:  # Generator - memory efficient
        yield process_record(record)

# Instead of:
def process_records(records):
    return [process_record(r) for r in records]  # List - loads all in memory
```

**Solution 4: Use Dask for Large Datasets**
```python
import dask.dataframe as dd

# Use Dask instead of pandas for large files
df = dd.read_csv('huge_file.csv')
result = df.groupby('category').emissions.sum().compute()
```

**Solution 5: Add Swap Space**
```bash
# Create 4GB swap file
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

### Issue 13: Agent Fails with "Permission Denied"

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: '/var/lib/greenlang/data/output.csv'
```

**Solutions:**

**Solution 1: Fix File Permissions**
```bash
# Check current permissions
ls -la /var/lib/greenlang/data/

# Fix ownership
sudo chown -R greenlang:greenlang /var/lib/greenlang/

# Fix permissions
sudo chmod -R 755 /var/lib/greenlang/
```

**Solution 2: Run as Correct User**
```bash
# Check current user
whoami

# Run as greenlang user
sudo -u greenlang greenlang run cbam --input data.csv

# Or in systemd service
# /etc/systemd/system/greenlang-worker.service
[Service]
User=greenlang
Group=greenlang
```

**Solution 3: Use Docker Volumes with Correct UID**
```bash
# Check UID of greenlang user in container
docker run greenlang/greenlang:latest id -u greenlang
# Returns: 1000

# Run with matching UID
docker run --user 1000:1000 -v ./data:/data greenlang/greenlang:latest
```

**Solution 4: Configure Writable Directories**
```python
# config.py
import os

DATA_DIR = os.getenv('GREENLANG_DATA_DIR', '/tmp/greenlang/data')
UPLOAD_DIR = os.getenv('GREENLANG_UPLOAD_DIR', '/tmp/greenlang/uploads')
REPORT_DIR = os.getenv('GREENLANG_REPORT_DIR', '/tmp/greenlang/reports')

# Ensure directories exist and are writable
for directory in [DATA_DIR, UPLOAD_DIR, REPORT_DIR]:
    os.makedirs(directory, exist_ok=True)
```

---

### Issue 14: Celery Worker Won't Start

**Symptoms:**
```
ERROR: Cannot connect to amqp://guest:**@localhost:5672//
```

**Solutions:**

**Solution 1: Start Redis (Broker)**
```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG

# If not running:
# Linux
sudo systemctl start redis
sudo systemctl enable redis

# macOS
brew services start redis

# Docker
docker run -d -p 6379:6379 redis:7
```

**Solution 2: Check Celery Configuration**
```python
# config.py
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

# Or with RabbitMQ
CELERY_BROKER_URL = 'amqp://guest:guest@localhost:5672//'
```

**Solution 3: Test Broker Connection**
```bash
# Test Redis
redis-cli -h localhost -p 6379 ping

# Test RabbitMQ
telnet localhost 5672
```

**Solution 4: Start Worker with Verbose Logging**
```bash
celery -A greenlang.worker worker --loglevel=debug
```

---

## Calculation Errors

### Issue 15: "Emission Factor Not Found"

**Symptoms:**
```
EmissionFactorNotFoundError: No emission factor found for activity='electricity_consumption' region='US-XX'
```

**Diagnosis:**

```python
# Check available emission factors
from greenlang.emission_factors import EmissionFactorRegistry

registry = EmissionFactorRegistry()

# List all factors for activity
factors = registry.get_factors(activity='electricity_consumption')
for factor in factors:
    print(f"{factor.region}: {factor.value} {factor.unit}")

# Check specific region
factor = registry.get_factor(activity='electricity_consumption', region='US-CA')
print(factor)
```

**Solutions:**

**Solution 1: Use Correct Region Code**
```python
# Wrong: 'US-XX' (invalid region)
result = gl.calculate_emissions(
    activity='electricity_consumption',
    amount=1000,
    region='US-CA'  # Correct: Use valid region code
)

# Check valid regions
regions = registry.get_available_regions()
print(regions)  # ['US-CA', 'US-TX', 'US-NY', 'EU-WEST', ...]
```

**Solution 2: Fall Back to Country-Level Factor**
```python
# If region-specific factor not available, use country-level
result = gl.calculate_emissions(
    activity='electricity_consumption',
    amount=1000,
    region='US',  # Country-level instead of state
    fallback=True
)
```

**Solution 3: Update Emission Factors**
```bash
# Download latest emission factors
greenlang update-emission-factors

# Or manually
greenlang emission-factors sync --source IEA --source EPA
```

**Solution 4: Add Custom Emission Factor**
```python
from greenlang.emission_factors import EmissionFactorRegistry

registry = EmissionFactorRegistry()

# Add custom factor
registry.add_factor(
    name='electricity_consumption',
    value=0.45,
    unit='kg CO2e / kWh',
    region='US-XX',
    source='Custom measurement 2024',
    year=2024
)

# Now calculation will work
result = gl.calculate_emissions(
    activity='electricity_consumption',
    amount=1000,
    region='US-XX'
)
```

---

### Issue 16: Calculation Returns Zero or Negative Emissions

**Symptoms:**
```json
{
  "emissions_kg_co2e": 0.0,
  "activity": "natural_gas_combustion",
  "amount": 1000,
  "unit": "m3"
}
```

**Diagnosis:**

```bash
# Enable debug logging
export GREENLANG_LOG_LEVEL=DEBUG
greenlang calculate --input data.csv --output report.pdf

# Check calculation log
greenlang logs --job job_abc123 --level DEBUG
```

**Possible Causes:**
1. Amount is zero or null
2. Wrong unit conversion
3. Emission factor is zero
4. Bug in calculation logic

**Solutions:**

**Solution 1: Verify Input Data**
```python
# Check input data
print(f"Amount: {amount}")
print(f"Unit: {unit}")
print(f"Activity: {activity}")

# Ensure amount is positive
assert amount > 0, f"Amount must be positive, got {amount}"
```

**Solution 2: Check Unit Conversion**
```python
# Ensure units match
# If emission factor is in "kg CO2e / kWh" but you provide "MWh", convert:

from greenlang.units import convert_unit

amount_kwh = convert_unit(amount, from_unit='MWh', to_unit='kWh')
# 1000 MWh = 1,000,000 kWh

result = gl.calculate_emissions(
    activity='electricity_consumption',
    amount=amount_kwh,
    unit='kWh'
)
```

**Solution 3: Verify Emission Factor**
```python
# Get emission factor used
factor = registry.get_factor(activity='natural_gas_combustion', region='US')
print(f"Emission factor: {factor.value} {factor.unit}")

# If factor is 0, it's likely an error
if factor.value == 0:
    print("WARNING: Emission factor is zero!")
```

**Solution 4: Check for Calculation Bug**
```python
# Run with detailed logging
result = gl.calculate_emissions(
    activity='natural_gas_combustion',
    amount=1000,
    unit='m3',
    region='US',
    debug=True
)

# Inspect calculation steps
print(result.calculation_log)
```

---

### Issue 17: "Unit Mismatch" Error

**Symptoms:**
```
UnitMismatchError: Cannot convert from 'kg' to 'kWh' (incompatible dimensions)
```

**Solutions:**

**Solution 1: Use Correct Units**
```python
# Wrong: Mixing mass and energy units
result = gl.calculate_emissions(
    activity='electricity_consumption',  # Requires energy unit
    amount=1000,
    unit='kg'  # Wrong: This is a mass unit
)

# Correct: Use energy unit
result = gl.calculate_emissions(
    activity='electricity_consumption',
    amount=1000,
    unit='kWh'  # Correct: Energy unit
)
```

**Solution 2: Check Emission Factor Units**
```python
# Get emission factor to see expected unit
factor = registry.get_factor(activity='electricity_consumption', region='US')
print(f"Expected unit: {factor.unit}")
# Output: "kg CO2e / kWh"

# Match your input unit to the denominator
result = gl.calculate_emissions(
    activity='electricity_consumption',
    amount=1000,
    unit='kWh'  # Matches denominator of emission factor
)
```

**Solution 3: Convert Units**
```python
from greenlang.units import convert_unit

# Convert units before calculation
amount_kwh = convert_unit(1000, from_unit='MWh', to_unit='kWh')
# 1000 MWh → 1,000,000 kWh

result = gl.calculate_emissions(
    activity='electricity_consumption',
    amount=amount_kwh,
    unit='kWh'
)
```

---

### Issue 18: Data Quality Score Too Low

**Symptoms:**
```json
{
  "data_quality_score": 35.5,
  "status": "warning",
  "message": "Data quality below acceptable threshold (50)"
}
```

**Diagnosis:**

```python
# Get detailed quality report
report = gl.validate_data(data)

print(f"Overall score: {report.quality_score}")
print("\nIssues by severity:")
print(f"Critical: {len(report.critical_issues)}")
print(f"High: {len(report.high_issues)}")
print(f"Medium: {len(report.medium_issues)}")
print(f"Low: {len(report.low_issues)}")

# View specific issues
for issue in report.critical_issues:
    print(f"- {issue.field}: {issue.message}")
```

**Common Data Quality Issues:**

| Issue | Impact | Fix |
|-------|--------|-----|
| Missing values | -10 points per field | Fill in missing data |
| Invalid format | -5 points | Correct data format |
| Out of range | -5 points | Fix outliers |
| Inconsistent units | -3 points | Standardize units |
| Future dates | -2 points | Correct dates |

**Solutions:**

**Solution 1: Fix Missing Data**
```python
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Check missing values
print(df.isnull().sum())

# Fill missing values
df['country'].fillna('Unknown', inplace=True)
df['quantity'].fillna(0, inplace=True)

# Save cleaned data
df.to_csv('data_cleaned.csv', index=False)
```

**Solution 2: Validate Before Processing**
```python
# Validate data
validation_result = gl.validate_data(data)

if validation_result.quality_score < 50:
    print("Data quality too low. Please fix the following issues:")
    for issue in validation_result.issues:
        print(f"- Row {issue.row}: {issue.field} - {issue.message}")
    exit(1)

# Only process if quality is acceptable
result = gl.calculate_emissions(data)
```

**Solution 3: Use Data Cleaning Agent**
```python
from greenlang.agents import DataCleanerAgent

cleaner = DataCleanerAgent()

# Automatically clean common issues
cleaned_data = cleaner.clean(data, auto_fix=True)

print(f"Original quality: {gl.validate_data(data).quality_score}")
print(f"Cleaned quality: {gl.validate_data(cleaned_data).quality_score}")
```

**Solution 4: Lower Quality Threshold (Not Recommended)**
```python
# config.py
DATA_QUALITY_THRESHOLD = 35  # Lower from default 50

# Warning: This may reduce accuracy of calculations
```

---

## Performance Problems

### Issue 19: Slow Calculations

**Symptoms:**
- Calculations take minutes instead of seconds
- High CPU usage
- Long processing times

**Diagnosis:**

```bash
# Time a calculation
time greenlang calculate --input data.csv

# Profile Python code
python -m cProfile -o profile.stats run_calculation.py

# Analyze profile
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

# Check database query performance
# In PostgreSQL:
EXPLAIN ANALYZE SELECT * FROM calculations WHERE job_id = 'job_abc123';
```

**Solutions:**

**Solution 1: Enable Caching**
```python
# config.py
CACHE_ENABLED = True
CACHE_BACKEND = 'redis'
CACHE_URL = 'redis://localhost:6379/1'

# Emission factors will be cached
# Repeated calculations will be faster
```

**Solution 2: Use Batch Processing**
```python
# Instead of processing one at a time
for record in records:
    result = gl.calculate_emissions(record)

# Use batch processing
results = gl.calculate_emissions_batch(records)
# 10-100x faster for large datasets
```

**Solution 3: Optimize Database Queries**
```sql
-- Add indexes
CREATE INDEX idx_calculations_job_id ON calculations(job_id);
CREATE INDEX idx_calculations_created_at ON calculations(created_at);
CREATE INDEX idx_emission_factors_activity_region ON emission_factors(activity, region);

-- Analyze tables
ANALYZE calculations;
ANALYZE emission_factors;
```

**Solution 4: Use Parallel Processing**
```python
# config.py
PARALLEL_PROCESSING = True
MAX_WORKERS = 4  # Number of CPU cores

# Or use multiprocessing
from multiprocessing import Pool

with Pool(processes=4) as pool:
    results = pool.map(calculate_emissions, records)
```

**Solution 5: Upgrade Hardware**
```bash
# Add more CPU cores
# Increase RAM
# Use SSD instead of HDD
# Use faster network connection
```

---

### Issue 20: High Memory Usage

**Symptoms:**
```
Process using 8+ GB RAM
System becomes unresponsive
Swap usage at 100%
```

**Diagnosis:**

```bash
# Monitor memory usage
top -o %MEM
htop

# Python memory profiler
pip install memory_profiler
python -m memory_profiler calculation.py

# Check specific process
ps aux | grep greenlang
```

**Solutions:**

**Solution 1: Process Data in Chunks**
```python
# Instead of loading entire file
df = pd.read_csv('huge_file.csv')  # Loads entire file into RAM

# Use chunking
chunk_size = 10000
for chunk in pd.read_csv('huge_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
    # Each chunk is processed and released from memory
```

**Solution 2: Use Generators**
```python
# Instead of storing all results in memory
results = [calculate_emissions(r) for r in records]  # List in memory

# Use generator
results = (calculate_emissions(r) for r in records)  # Lazy evaluation
for result in results:
    save_result(result)
    # Result is processed and released immediately
```

**Solution 3: Limit Concurrent Workers**
```python
# config.py
MAX_WORKERS = 2  # Reduce from 4 or 8

# Each worker uses memory, fewer workers = less total memory
```

**Solution 4: Clear Cache Regularly**
```python
import gc

# Force garbage collection
gc.collect()

# Clear emission factor cache
from greenlang.emission_factors import EmissionFactorRegistry
registry = EmissionFactorRegistry()
registry.clear_cache()
```

**Solution 5: Increase System Memory**
```bash
# Add more RAM to server
# Add swap space (temporary solution)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

### Issue 21: Database Queries Taking Too Long

**Symptoms:**
```
Query execution time: 15+ seconds
Database CPU at 100%
Long query logs
```

**Diagnosis:**

```sql
-- Enable query logging
-- postgresql.conf
log_min_duration_statement = 1000  # Log queries >1 second

-- Find slow queries
SELECT
    pid,
    now() - pg_stat_activity.query_start AS duration,
    query,
    state
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY duration DESC;

-- Analyze specific query
EXPLAIN ANALYZE
SELECT * FROM calculations WHERE activity = 'electricity_consumption';
```

**Solutions:**

**Solution 1: Add Database Indexes**
```sql
-- Identify missing indexes
-- Check for sequential scans in EXPLAIN output

-- Add indexes for common queries
CREATE INDEX idx_calculations_activity ON calculations(activity);
CREATE INDEX idx_calculations_job_id ON calculations(job_id);
CREATE INDEX idx_calculations_created_at ON calculations(created_at DESC);
CREATE INDEX idx_emission_factors_activity_region ON emission_factors(activity, region);

-- Composite index for common queries
CREATE INDEX idx_calculations_job_activity ON calculations(job_id, activity);

-- Index on JSONB columns
CREATE INDEX idx_calculations_metadata ON calculations USING GIN(metadata);
```

**Solution 2: Optimize Queries**
```python
# Bad: Loads all columns
results = db.query(Calculation).filter_by(job_id=job_id).all()

# Good: Load only needed columns
results = db.query(
    Calculation.id,
    Calculation.activity,
    Calculation.emissions_kg_co2e
).filter_by(job_id=job_id).all()

# Bad: N+1 query problem
for calc in calculations:
    factor = db.query(EmissionFactor).get(calc.factor_id)

# Good: Use join
calculations = db.query(Calculation).join(EmissionFactor).filter_by(job_id=job_id).all()
```

**Solution 3: Use Connection Pooling**
```python
# config.py
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True  # Verify connections
)
```

**Solution 4: Vacuum and Analyze**
```sql
-- Regular maintenance
VACUUM ANALYZE calculations;
VACUUM ANALYZE emission_factors;

-- Or enable autovacuum
-- postgresql.conf
autovacuum = on
```

**Solution 5: Use Materialized Views**
```sql
-- Create materialized view for common aggregations
CREATE MATERIALIZED VIEW calculation_summary AS
SELECT
    job_id,
    activity,
    COUNT(*) as count,
    SUM(emissions_kg_co2e) as total_emissions
FROM calculations
GROUP BY job_id, activity;

-- Refresh periodically
REFRESH MATERIALIZED VIEW calculation_summary;

-- Query the view (much faster)
SELECT * FROM calculation_summary WHERE job_id = 'job_abc123';
```

---

## API Errors

### Issue 22: 401 Unauthorized

**Symptoms:**
```http
HTTP/1.1 401 Unauthorized
{
  "error": "unauthorized",
  "message": "Invalid or missing authentication token"
}
```

**Solutions:**

**Solution 1: Verify API Key**
```bash
# Check if API key is set
echo $GREENLANG_API_KEY

# Test with curl
curl -H "Authorization: Bearer $GREENLANG_API_KEY" \
  https://api.greenlang.io/v1/status

# If 401, regenerate API key
greenlang auth generate-key
```

**Solution 2: Check Token Expiration**
```python
import jwt

# Decode token (without verification)
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
decoded = jwt.decode(token, options={"verify_signature": False})

print(f"Token expires at: {decoded['exp']}")

# If expired, get new token
from greenlang.client import Client
client = Client(api_key="your_api_key")
new_token = client.auth.refresh_token()
```

**Solution 3: Verify Authorization Header**
```python
import requests

# Correct format
headers = {
    "Authorization": f"Bearer {api_key}",  # Note: "Bearer " prefix
    "Content-Type": "application/json"
}

response = requests.get("https://api.greenlang.io/v1/status", headers=headers)
```

---

### Issue 23: 429 Rate Limit Exceeded

**Symptoms:**
```http
HTTP/1.1 429 Too Many Requests
Retry-After: 42

{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded. Try again in 42 seconds."
}
```

**Solutions:**

**Solution 1: Implement Exponential Backoff**
```python
import time
import requests

def call_api_with_retry(url, max_retries=5):
    for attempt in range(max_retries):
        response = requests.get(url)

        if response.status_code != 429:
            return response

        # Get retry-after from header
        retry_after = int(response.headers.get('Retry-After', 2 ** attempt))
        print(f"Rate limited. Waiting {retry_after} seconds...")
        time.sleep(retry_after)

    raise Exception("Max retries exceeded")
```

**Solution 2: Use Batch Endpoints**
```python
# Instead of 100 individual requests
for item in items:
    response = requests.post('/api/v1/calculate', json=item)

# Use batch endpoint (1 request)
response = requests.post('/api/v1/calculate/batch', json={'items': items})
```

**Solution 3: Upgrade Plan**
```bash
# Free tier: 100 requests/minute
# Pro tier: 500 requests/minute
# Enterprise: Unlimited (fair use)

# Contact sales@greenlang.io to upgrade
```

**Solution 4: Self-Host (No Rate Limits)**
```bash
# Deploy your own instance
docker-compose up -d

# No rate limits on self-hosted instances
```

---

### Issue 24: 500 Internal Server Error

**Symptoms:**
```http
HTTP/1.1 500 Internal Server Error
{
  "error": "internal_error",
  "message": "An unexpected error occurred",
  "request_id": "req_abc123"
}
```

**Solutions:**

**Solution 1: Retry the Request**
```python
# 500 errors may be transient, retry after a delay
import time

response = requests.post(url, json=data)
if response.status_code == 500:
    time.sleep(5)
    response = requests.post(url, json=data)  # Retry once
```

**Solution 2: Check Server Logs**
```bash
# For self-hosted instances
greenlang logs --level ERROR --request-id req_abc123

# Or check log files
tail -n 100 /var/log/greenlang/error.log | grep req_abc123
```

**Solution 3: Report to Support**
```bash
# If using GreenLang Cloud, report the error
# Include the request_id from the error response

# Email: support@greenlang.io
# Subject: 500 Error - Request ID: req_abc123
```

**Solution 4: Check for Known Issues**
```bash
# Check status page
curl https://status.greenlang.io

# Check GitHub issues
# https://github.com/greenlang/greenlang/issues
```

---

## Authentication Issues

### Issue 25: Unable to Generate API Key

**Symptoms:**
```
ERROR: Failed to generate API key
```

**Solutions:**

**Solution 1: Verify User Account**
```bash
# Check if user exists
greenlang users list

# Create user if missing
greenlang users create --username admin --email admin@example.com

# Set password
greenlang users set-password admin
```

**Solution 2: Check Database Connection**
```bash
# Ensure database is accessible
psql -h localhost -U greenlang -d greenlang -c "SELECT 1;"

# Run migrations if needed
greenlang db upgrade
```

**Solution 3: Generate Key Manually**
```python
from greenlang.auth import generate_api_key
from greenlang.models import User
from greenlang.database import db_session

# Get user
user = db_session.query(User).filter_by(username='admin').first()

# Generate key
api_key = generate_api_key(user.id)

print(f"API Key: {api_key}")
```

---

### Issue 26: Token Expired

**Symptoms:**
```json
{
  "error": "token_expired",
  "message": "JWT token has expired"
}
```

**Solutions:**

**Solution 1: Refresh Token**
```python
from greenlang.client import Client

client = Client(api_key="your_api_key")

# Refresh access token
new_token = client.auth.refresh_token()
print(f"New token: {new_token}")
```

**Solution 2: Increase Token Expiration**
```python
# config.py
JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour (default)
JWT_REFRESH_TOKEN_EXPIRES = 86400 * 30  # 30 days
```

**Solution 3: Use API Key Instead of JWT**
```python
# API keys don't expire
import requests

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get("https://api.greenlang.io/v1/status", headers=headers)
```

---

## Docker Issues

### Issue 27: Docker Container Won't Start

**Symptoms:**
```
Error response from daemon: OCI runtime create failed
```

**Diagnosis:**

```bash
# Check container logs
docker logs greenlang

# Check container status
docker ps -a | grep greenlang

# Inspect container
docker inspect greenlang
```

**Solutions:**

**Solution 1: Check Port Conflicts**
```bash
# Check if port is already in use
sudo netstat -tlnp | grep 8000
# Or
lsof -i :8000

# Use different port
docker run -p 8001:8000 greenlang/greenlang:latest

# Or kill process using port
sudo kill -9 <PID>
```

**Solution 2: Check Environment Variables**
```bash
# Verify required environment variables
docker run -e DATABASE_URL=postgresql://... greenlang/greenlang:latest

# Or use .env file
docker run --env-file .env greenlang/greenlang:latest
```

**Solution 3: Check Volume Mounts**
```bash
# Ensure volumes exist and have correct permissions
ls -la /var/lib/greenlang

# Create if missing
sudo mkdir -p /var/lib/greenlang/data
sudo chown -R 1000:1000 /var/lib/greenlang
```

**Solution 4: Pull Latest Image**
```bash
# Pull latest image
docker pull greenlang/greenlang:latest

# Remove old containers
docker rm greenlang

# Start new container
docker run -d --name greenlang greenlang/greenlang:latest
```

---

### Issue 28: Cannot Connect to Container Database

**Symptoms:**
```
could not connect to server: Connection refused
```

**Solutions:**

**Solution 1: Use Docker Network**
```bash
# Create network
docker network create greenlang-net

# Start PostgreSQL
docker run -d \
  --name greenlang-postgres \
  --network greenlang-net \
  -e POSTGRES_DB=greenlang \
  -e POSTGRES_USER=greenlang \
  -e POSTGRES_PASSWORD=password \
  postgres:14

# Start GreenLang (use container name as host)
docker run -d \
  --name greenlang \
  --network greenlang-net \
  -e DATABASE_URL=postgresql://greenlang:password@greenlang-postgres:5432/greenlang \
  greenlang/greenlang:latest
```

**Solution 2: Use Docker Compose**
```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: greenlang
      POSTGRES_USER: greenlang
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  greenlang:
    image: greenlang/greenlang:latest
    depends_on:
      - postgres
    environment:
      DATABASE_URL: postgresql://greenlang:password@postgres:5432/greenlang
    ports:
      - "8000:8000"

volumes:
  postgres_data:
```

```bash
docker-compose up -d
```

---

### Issue 29: Docker Build Fails

**Symptoms:**
```
ERROR [build 5/8] RUN pip install -r requirements.txt
```

**Solutions:**

**Solution 1: Clear Build Cache**
```bash
# Build without cache
docker build --no-cache -t greenlang:latest .
```

**Solution 2: Check Dockerfile**
```dockerfile
# Ensure base image is correct
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app
```

**Solution 3: Check requirements.txt**
```bash
# Test locally
pip install -r requirements.txt

# If fails, fix dependencies
```

---

## Kubernetes Issues

### Issue 30: Pod Stuck in Pending State

**Symptoms:**
```
NAME                         READY   STATUS    RESTARTS   AGE
greenlang-api-6d5f4b7c8-abc  0/1     Pending   0          5m
```

**Diagnosis:**

```bash
# Describe pod
kubectl describe pod greenlang-api-6d5f4b7c8-abc -n greenlang

# Check events
kubectl get events -n greenlang --sort-by='.lastTimestamp'

# Check node resources
kubectl top nodes
```

**Possible Causes:**
1. Insufficient resources (CPU/memory)
2. Volume mount issues
3. Image pull errors
4. Node selector mismatch

**Solutions:**

**Solution 1: Check Resource Requests**
```yaml
# deployment.yaml
resources:
  requests:
    memory: "2Gi"  # Reduce if nodes don't have enough memory
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

**Solution 2: Check Node Resources**
```bash
# Check available resources
kubectl describe nodes

# If nodes are full, scale cluster
# Or reduce resource requests
```

**Solution 3: Fix Image Pull Errors**
```bash
# Check image pull secret
kubectl get secrets -n greenlang

# Create image pull secret if needed
kubectl create secret docker-registry regcred \
  --docker-server=registry.example.com \
  --docker-username=user \
  --docker-password=password \
  -n greenlang

# Reference in deployment
# deployment.yaml
imagePullSecrets:
  - name: regcred
```

---

### Issue 31: Service Not Accessible

**Symptoms:**
```bash
curl: (7) Failed to connect to greenlang.example.com port 443: Connection refused
```

**Diagnosis:**

```bash
# Check service
kubectl get svc -n greenlang

# Check ingress
kubectl get ingress -n greenlang

# Describe ingress
kubectl describe ingress greenlang -n greenlang

# Check ingress controller logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx
```

**Solutions:**

**Solution 1: Verify Ingress Configuration**
```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: greenlang
  namespace: greenlang
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - greenlang.example.com
      secretName: greenlang-tls
  rules:
    - host: greenlang.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: greenlang-api
                port:
                  number: 8000
```

**Solution 2: Check DNS**
```bash
# Verify DNS resolves to correct IP
nslookup greenlang.example.com

# Should point to ingress load balancer IP
kubectl get ingress greenlang -n greenlang -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

**Solution 3: Port Forward for Testing**
```bash
# Bypass ingress and test service directly
kubectl port-forward -n greenlang svc/greenlang-api 8000:8000

# Test
curl http://localhost:8000/health
```

---

### Issue 32: Pod Crashes with OOMKilled

**Symptoms:**
```
NAME                         READY   STATUS      RESTARTS   AGE
greenlang-api-6d5f4b7c8-abc  0/1     OOMKilled   3          10m
```

**Solutions:**

**Solution 1: Increase Memory Limits**
```yaml
# deployment.yaml
resources:
  requests:
    memory: "4Gi"  # Increased from 2Gi
  limits:
    memory: "8Gi"  # Increased from 4Gi
```

**Solution 2: Add Swap (Not Recommended for K8s)**
```yaml
# Better solution: Fix memory leak or optimize code
# See Issue 12 for memory optimization strategies
```

**Solution 3: Use Horizontal Pod Autoscaling**
```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: greenlang-api
  namespace: greenlang
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: greenlang-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 70
```

---

## Common Error Messages

### Error: `ModuleNotFoundError: No module named 'greenlang'`

**See:** [Issue 2: ImportError](#issue-2-importerror-no-module-named-greenlang)

---

### Error: `psycopg2.OperationalError: could not connect to server`

**See:** [Issue 6: Cannot Connect to PostgreSQL](#issue-6-cannot-connect-to-postgresql)

---

### Error: `EmissionFactorNotFoundError`

**See:** [Issue 15: Emission Factor Not Found](#issue-15-emission-factor-not-found)

---

### Error: `UnitMismatchError`

**See:** [Issue 17: Unit Mismatch Error](#issue-17-unit-mismatch-error)

---

### Error: `MemoryError: Unable to allocate array`

**See:** [Issue 12: Agent Fails with Out of Memory](#issue-12-agent-fails-with-out-of-memory)

---

### Error: `PermissionError: [Errno 13] Permission denied`

**See:** [Issue 13: Agent Fails with Permission Denied](#issue-13-agent-fails-with-permission-denied)

---

### Error: `ValidationError: Invalid input data`

**Symptoms:**
```json
{
  "error": "validation_error",
  "message": "Invalid input data",
  "details": {
    "field": "quantity",
    "issue": "must be a positive number",
    "value": -10
  }
}
```

**Solutions:**

**Solution 1: Fix Input Data**
```python
# Ensure data meets validation requirements
data = {
    "activity": "electricity_consumption",
    "amount": 1000,  # Must be positive
    "unit": "kWh",   # Must be valid unit
    "region": "US-CA"  # Must be valid region
}
```

**Solution 2: Check Validation Rules**
```python
from greenlang.validation import validate_input

# Get validation schema
schema = validate_input.get_schema('electricity_consumption')
print(schema)

# Validate before submitting
is_valid, errors = validate_input(data)
if not is_valid:
    print(f"Validation errors: {errors}")
```

---

### Error: `DatabaseError: deadlock detected`

**Symptoms:**
```
sqlalchemy.exc.OperationalError: deadlock detected
DETAIL:  Process 1234 waits for ShareLock on transaction 5678; blocked by process 9012.
```

**Solutions:**

**Solution 1: Retry Transaction**
```python
from sqlalchemy.exc import OperationalError
import time

max_retries = 3
for attempt in range(max_retries):
    try:
        # Your transaction
        db.session.commit()
        break
    except OperationalError as e:
        if "deadlock" in str(e):
            db.session.rollback()
            time.sleep(0.5 * (2 ** attempt))  # Exponential backoff
        else:
            raise
```

**Solution 2: Reduce Lock Contention**
```python
# Use SELECT FOR UPDATE with NOWAIT or SKIP LOCKED
record = db.session.query(Record).filter_by(id=123).with_for_update(nowait=True).first()

# Or skip locked rows
records = db.session.query(Record).with_for_update(skip_locked=True).all()
```

**Solution 3: Use Optimistic Locking**
```python
from sqlalchemy import Column, Integer

class Record(Base):
    __tablename__ = 'records'
    id = Column(Integer, primary_key=True)
    version = Column(Integer, default=0)  # Version column

    # Update with version check
    db.session.query(Record).filter_by(id=123, version=current_version).update({
        'data': new_data,
        'version': current_version + 1
    })
```

---

## Logging and Debugging

### Enabling Debug Logging

**Environment Variable:**
```bash
export GREENLANG_LOG_LEVEL=DEBUG
greenlang calculate --input data.csv
```

**Python Code:**
```python
import logging

# Set logging level
logging.basicConfig(level=logging.DEBUG)

# Or for GreenLang logger specifically
greenlang_logger = logging.getLogger('greenlang')
greenlang_logger.setLevel(logging.DEBUG)
```

**Configuration File:**
```python
# config.py
LOG_LEVEL = "DEBUG"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "/var/log/greenlang/debug.log"
```

---

### Viewing Logs

**CLI:**
```bash
# View all logs
greenlang logs

# Filter by level
greenlang logs --level ERROR

# Filter by agent
greenlang logs --agent calculator

# Filter by job
greenlang logs --job job_abc123

# Follow logs (tail -f)
greenlang logs --follow

# Export logs
greenlang logs --export logs.json
```

**Log Files:**
```bash
# Default log locations
/var/log/greenlang/greenlang.log        # Main log
/var/log/greenlang/error.log            # Errors only
/var/log/greenlang/agent_calculator.log # Agent-specific

# View logs
tail -f /var/log/greenlang/greenlang.log

# Search logs
grep "ERROR" /var/log/greenlang/greenlang.log
```

**Docker Logs:**
```bash
# View container logs
docker logs greenlang

# Follow logs
docker logs -f greenlang

# View last N lines
docker logs --tail 100 greenlang
```

**Kubernetes Logs:**
```bash
# View pod logs
kubectl logs -n greenlang greenlang-api-6d5f4b7c8-abc

# Follow logs
kubectl logs -n greenlang -f greenlang-api-6d5f4b7c8-abc

# View previous container logs (after crash)
kubectl logs -n greenlang greenlang-api-6d5f4b7c8-abc --previous
```

---

### Debugging Tips

**1. Enable SQL Query Logging:**
```python
# config.py
SQLALCHEMY_ECHO = True  # Log all SQL queries
```

**2. Use Python Debugger:**
```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use ipdb for better experience
import ipdb; ipdb.set_trace()
```

**3. Profile Performance:**
```bash
# Profile Python code
python -m cProfile -o profile.stats run_calculation.py

# Analyze with snakeviz
pip install snakeviz
snakeviz profile.stats
```

**4. Memory Profiling:**
```bash
# Install memory_profiler
pip install memory_profiler

# Profile function
@profile
def calculate_emissions(data):
    # Your code
    pass

# Run with profiler
python -m memory_profiler calculation.py
```

**5. Network Debugging:**
```bash
# Check API requests
export GREENLANG_DEBUG_HTTP=1
greenlang calculate --input data.csv

# Or use mitmproxy
mitmproxy -p 8080
export HTTP_PROXY=http://localhost:8080
```

---

## Getting Help

If you cannot resolve your issue:

**1. Check Documentation:**
- Main docs: `docs/`
- FAQ: `docs/FAQ.md`
- API reference: `docs/API_REFERENCE_COMPLETE.md`

**2. Search Issues:**
- GitHub issues: https://github.com/greenlang/greenlang/issues
- Stack Overflow: [greenlang] tag

**3. Community Support:**
- GitHub Discussions: https://github.com/greenlang/greenlang/discussions
- Discord: https://discord.gg/greenlang

**4. Professional Support:**
- Email: support@greenlang.io
- Include:
  - GreenLang version: `greenlang --version`
  - Python version: `python --version`
  - Operating system
  - Error messages and logs
  - Steps to reproduce
  - Minimal code example

**5. Report Bugs:**
- Security issues: security@greenlang.io (private)
- Bug reports: https://github.com/greenlang/greenlang/issues/new

---

**Document Information:**
- **Version:** 1.0
- **Last Updated:** November 2025
- **Maintained By:** GreenLang Documentation Team
- **License:** MIT License
- **Contribute:** https://github.com/greenlang/greenlang/blob/main/docs/TROUBLESHOOTING.md

**Still having issues?**
Join our community: https://discord.gg/greenlang
