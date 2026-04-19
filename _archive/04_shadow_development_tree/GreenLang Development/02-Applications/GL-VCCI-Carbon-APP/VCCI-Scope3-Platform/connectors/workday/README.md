# Workday RaaS Connector

**Version:** 1.0.0
**Phase:** 4 (Weeks 24-26)
**Status:** Production-Ready
**Total Lines:** 3,189 lines of production Python code

## Overview

Complete Workday RaaS (Report as a Service) connector for the GL-VCCI Scope 3 Carbon Platform. Extracts HCM data (expense reports and commute surveys) for Scope 3 Categories 6 (Business Travel) and 7 (Employee Commuting).

## Architecture

```
workday/
├── __init__.py (146 lines)           # Main exports and factory
├── config.py (407 lines)             # Configuration management with Pydantic
├── auth.py (350 lines)               # OAuth 2.0 authentication
├── client.py (538 lines)             # RaaS API client
├── exceptions.py (363 lines)         # Custom exceptions
├── extractors/
│   ├── __init__.py (17 lines)
│   ├── base.py (182 lines)           # Base extractor with delta sync
│   └── hcm_extractor.py (331 lines)  # HCM data extractor
├── mappers/
│   ├── __init__.py (17 lines)
│   ├── expense_mapper.py (277 lines) # Expense → logistics_v1.0.json
│   └── commute_mapper.py (246 lines) # Commute → Category 7 format
└── jobs/
    ├── __init__.py (15 lines)
    └── delta_sync.py (349 lines)     # Celery jobs
```

## Key Features

### 1. OAuth 2.0 Authentication
- Client credentials flow
- Token caching with Redis
- Automatic token refresh
- Multi-environment support (sandbox, implementation, production)

### 2. RaaS API Client
- Pagination support (offset/limit)
- Date range filtering
- JSON and XML response parsing
- Exponential backoff retry (1s, 2s, 4s, 8s)
- Rate limiting (configurable, default 10 req/min)

### 3. Data Extraction
- **Expense Reports**: Category 6 (Business Travel)
  - Flights, car rentals, ground transportation, trains
  - Origin/destination cities
  - Distance and spend data
- **Commute Surveys**: Category 7 (Employee Commuting)
  - Home/office locations
  - Commute modes (car, bus, train, bike, walk)
  - Frequency and distance

### 4. Schema Mapping
- **Expense → logistics_v1.0.json**: Maps travel expenses to VCCI logistics schema
- **Commute → Category 7 format**: Custom schema for employee commuting

### 5. Delta Synchronization
- Incremental extraction since last sync
- Redis-based deduplication (90-day TTL)
- Celery Beat integration for scheduling
- Daily sync for expenses, weekly for commutes

## Installation

```bash
# Install dependencies
pip install pydantic requests redis celery

# Set environment variables
export WORKDAY_TENANT_URL="https://impl.workday.com/tenant_name"
export WORKDAY_TENANT_NAME="tenant_name"
export WORKDAY_CLIENT_ID="ISU_client_123"
export WORKDAY_CLIENT_SECRET="your_secret"
export WORKDAY_TOKEN_URL="https://impl.workday.com/tenant_name/oauth2/token"
export WORKDAY_ENVIRONMENT="sandbox"
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
```

## Usage

### Quick Start

```python
from connectors.workday import create_workday_connector

# Create connector
client = create_workday_connector(
    tenant_url="https://impl.workday.com/acme",
    tenant_name="acme",
    client_id="ISU_client_123",
    client_secret="your_secret",
    token_url="https://impl.workday.com/acme/oauth2/token",
    environment="sandbox"
)

# Get expense reports
from datetime import date, timedelta
expenses = client.get_report(
    report_name="expense_reports",
    from_date=date.today() - timedelta(days=30),
    to_date=date.today()
)

print(f"Extracted {len(expenses)} expense reports")
```

### Extract and Map Data

```python
from connectors.workday import (
    WorkdayConnectorConfig,
    WorkdayRaaSClient,
    HCMExtractor,
    ExpenseMapper,
    CommuteMapper
)
from datetime import date, timedelta

# Load configuration from environment
config = WorkdayConnectorConfig.from_env()

# Create client and extractors
client = WorkdayRaaSClient(config)
extractor = HCMExtractor(client, config)

# Extract expense reports
expenses = extractor.extract_expense_reports(
    from_date=date.today() - timedelta(days=30),
    to_date=date.today()
)

# Map to logistics schema
mapper = ExpenseMapper(tenant_id="tenant-acme")
logistics_records = mapper.map_expenses_batch(expenses)

print(f"Mapped {len(logistics_records)} logistics records")

# Extract commute surveys
commutes = extractor.extract_commute_surveys(
    from_date=date.today() - timedelta(days=90),
    to_date=date.today()
)

# Map to Category 7 format
commute_mapper = CommuteMapper(tenant_id="tenant-acme")
commute_records = commute_mapper.map_commutes_batch(commutes)

print(f"Mapped {len(commute_records)} commute records")
```

### Scheduled Sync with Celery

```python
from celery import Celery
from connectors.workday.jobs import sync_expense_reports, sync_commute_surveys

app = Celery('vcci_tasks')

# Configure Celery Beat schedule
app.conf.beat_schedule = {
    'sync-expenses-daily': {
        'task': 'connectors.workday.jobs.sync_expense_reports',
        'schedule': 86400.0,  # Daily
        'args': ('tenant-acme',)
    },
    'sync-commutes-weekly': {
        'task': 'connectors.workday.jobs.sync_commute_surveys',
        'schedule': 604800.0,  # Weekly
        'args': ('tenant-acme',)
    },
}

# Run manually
result = sync_expense_reports.delay('tenant-acme')
print(result.get())
```

## RaaS Report Configuration

The connector expects two custom RaaS reports in Workday:

### 1. Expense_Report_for_Carbon

**Fields Required:**
- Employee_ID
- Employee (name)
- Expense_Date
- Expense_Category (Flight, Hotel, Car Rental, etc.)
- Amount
- Currency
- Origin_City
- Destination_City
- Distance_KM (optional)
- Expense_ID
- Description

### 2. Commute_Survey_Results

**Fields Required:**
- Employee_ID
- Employee (name)
- Survey_Date
- Home_Location
- Office_Location
- Commute_Mode (Car, Bus, Train, Bike, Walk, etc.)
- Days_Per_Week
- Distance_KM (one-way)
- Vehicle_Type (optional)
- Carpool_Size (optional)

## API Patterns

### RaaS URL Format
```
GET https://{tenant}.workday.com/ccx/service/{tenant}/RaaS/{owner}/{report_name}
    ?format=json
    &From_Date=2024-01-01
    &To_Date=2024-01-31
    &offset=0
    &limit=1000
```

### Response Format
```json
{
  "Report_Entry": [
    {
      "Employee": "John Doe",
      "Expense_Date": "2024-01-15",
      "Expense_Category": "Flight",
      "Amount": "450.00",
      "Currency": "USD",
      "Origin_City": "San Francisco",
      "Destination_City": "New York"
    }
  ]
}
```

## Mapping Logic

### Expense → Logistics

**Expense Categories → Transport Modes:**
- Flight → Air_Freight_ShortHaul / Air_Freight_LongHaul (based on distance)
- Car Rental / Taxi / Rideshare → Road_Truck_LessThan7.5t
- Train → Rail_Freight
- Bus → Road_Truck_LessThan7.5t

**Calculation Methods:**
- Distance-based: If distance_km available
- Spend-based: If only spend amount available

### Commute → Category 7

**Emission Factors (kg CO2e/km):**
- Car: 0.192
- Bus: 0.089
- Train/Metro: 0.041
- Motorcycle: 0.113
- Bike/Walk: 0.0

**Annual Distance Calculation:**
```
annual_distance = distance_km_one_way × 2 × days_per_week × 52
```

## Error Handling

```python
from connectors.workday.exceptions import (
    WorkdayConnectionError,
    WorkdayAuthenticationError,
    WorkdayRateLimitError,
    WorkdayDataError,
    WorkdayTimeoutError
)

try:
    expenses = client.get_report("expense_reports")
except WorkdayAuthenticationError as e:
    print(f"Auth failed: {e.message}")
    print(f"Resolution: {e.details['resolution']}")
except WorkdayRateLimitError as e:
    retry_after = e.details['retry_after_seconds']
    print(f"Rate limited, retry after {retry_after}s")
except WorkdayConnectionError as e:
    print(f"Connection failed: {e.message}")
except WorkdayDataError as e:
    print(f"Data error: {e.message}")
```

## Integration Points

### SAP Utilities Reused
- RateLimiter (token bucket algorithm)
- Retry logic with exponential backoff
- OAuth 2.0 authentication pattern
- Exception hierarchy

### Future Integrations
- Database storage for logistics records
- PCF Exchange for supplier data
- Factor Broker for emission factors
- Policy Engine for validation

## Performance

- **Pagination**: 1,000 records per request (configurable)
- **Rate Limiting**: 10 requests/minute (configurable)
- **Timeout**: 60s read, 10s connect (configurable)
- **Retry**: Max 3 attempts with exponential backoff
- **Deduplication**: Redis-based, 90-day TTL

## Data Quality

The connector calculates ILCD-based Data Quality Indicators (DQI):
- Reliability (1-5)
- Completeness (1-5)
- Temporal Correlation (1-5)
- Geographical Correlation (1-5)
- Technological Correlation (1-5)

**DQI Ratings:**
- Excellent: 4.5-5.0
- Good: 3.5-4.4
- Fair: 2.5-3.4
- Poor: < 2.5

## Testing

```python
# Unit tests
pytest tests/connectors/workday/test_config.py
pytest tests/connectors/workday/test_auth.py
pytest tests/connectors/workday/test_client.py
pytest tests/connectors/workday/test_extractors.py
pytest tests/connectors/workday/test_mappers.py

# Integration tests
pytest tests/connectors/workday/test_integration.py
```

## Environment Variables

```bash
# Required
WORKDAY_TENANT_URL=https://impl.workday.com/tenant_name
WORKDAY_TENANT_NAME=tenant_name
WORKDAY_CLIENT_ID=ISU_client_123
WORKDAY_CLIENT_SECRET=your_secret
WORKDAY_TOKEN_URL=https://impl.workday.com/tenant_name/oauth2/token

# Optional
WORKDAY_ENVIRONMENT=sandbox              # sandbox, implementation, production
WORKDAY_REFRESH_TOKEN=refresh_token      # Optional refresh token
WORKDAY_RATE_LIMIT_RPM=10               # Requests per minute
WORKDAY_BATCH_SIZE=1000                 # Records per page
WORKDAY_MAX_RETRIES=3                   # Max retry attempts
WORKDAY_RETRY_BASE_DELAY=1.0            # Base delay (seconds)
WORKDAY_RETRY_MAX_DELAY=8.0             # Max delay (seconds)
WORKDAY_CONNECT_TIMEOUT=10.0            # Connection timeout (seconds)
WORKDAY_READ_TIMEOUT=60.0               # Read timeout (seconds)
WORKDAY_DEBUG_MODE=false                # Enable debug logging

# Redis (for caching and deduplication)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

## Completion Status

- [x] config.py - Configuration management (407 lines)
- [x] auth.py - OAuth 2.0 authentication (350 lines)
- [x] exceptions.py - Custom exceptions (363 lines)
- [x] client.py - RaaS API client (538 lines)
- [x] extractors/base.py - Base extractor (182 lines)
- [x] extractors/hcm_extractor.py - HCM extractor (331 lines)
- [x] mappers/expense_mapper.py - Expense mapper (277 lines)
- [x] mappers/commute_mapper.py - Commute mapper (246 lines)
- [x] jobs/delta_sync.py - Celery jobs (349 lines)
- [x] __init__.py - Main exports (146 lines)

**Total: 3,189 lines of production code**

## License

Copyright 2025 GreenLang / GL-VCCI Team. All rights reserved.
