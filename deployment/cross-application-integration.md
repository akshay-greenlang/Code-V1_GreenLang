# Cross-Application Integration Guide

**Version:** 1.0.0
**Last Updated:** 2025-11-08

---

## Overview

This guide provides detailed instructions for integrating the three GreenLang applications (CBAM, CSRD, VCCI) to enable seamless data flow and unified reporting.

---

## Integration Architecture

```
┌────────────────────────────────────────────────────────────┐
│                  Integration Patterns                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. Shared Authentication (JWT)                            │
│  2. Cross-App API Calls (REST)                             │
│  3. Event-Driven Sync (Message Queue)                      │
│  4. Shared Database (users, orgs)                          │
│  5. Unified Dashboard (Data Aggregation)                   │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 1. Shared Authentication System

### JWT Token Structure

All three applications use the same JWT structure for seamless authentication:

```json
{
  "sub": "user_id_uuid",
  "email": "user@example.com",
  "org_id": "org_uuid",
  "name": "John Doe",
  "roles": ["admin", "analyst"],
  "apps": {
    "cbam": ["admin"],
    "csrd": ["analyst"],
    "vcci": ["viewer"]
  },
  "exp": 1699999999,
  "iat": 1699913599,
  "iss": "greenlang-platform"
}
```

### Implementation

#### Centralized Auth Service

```python
# auth_service.py
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext

JWT_SECRET = os.getenv("JWT_SECRET")  # Min 32 characters
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(user_data: dict) -> str:
    """Create JWT access token"""
    to_encode = user_data.copy()
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS)
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> dict:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)
```

#### Middleware for Each App

```python
# middleware/auth.py (in each app)
from fastapi import Header, HTTPException
from auth_service import verify_token

async def get_current_user(authorization: str = Header(None)):
    """Extract and verify user from JWT token"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")

    token = authorization.replace("Bearer ", "")
    user_data = verify_token(token)

    # Check if user has access to this app
    app_name = "cbam"  # or "csrd", "vcci" depending on the app
    if app_name not in user_data.get("apps", {}):
        raise HTTPException(status_code=403, detail="Access denied to this application")

    return user_data

# Usage in endpoints
@app.get("/api/protected-resource")
async def protected_route(current_user: dict = Depends(get_current_user)):
    return {"message": f"Hello {current_user['email']}"}
```

### Single Sign-On (SSO) Flow

```
┌────────────┐
│   User     │
└─────┬──────┘
      │
      ▼
┌────────────────────┐
│ Login (Auth Service)│
│  - Verify credentials│
│  - Generate JWT     │
└────────┬───────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  JWT Token (valid for all 3 apps)   │
└──────────────────────────────────────┘
         │
    ┌────┼────┐
    ▼    ▼    ▼
 ┌────┐ ┌────┐ ┌────┐
 │CBAM│ │CSRD│ │VCCI│
 └────┘ └────┘ └────┘
```

---

## 2. Cross-Application API Calls

### Use Case: CSRD Importing VCCI Emissions Data

```python
# In CSRD app: services/vcci_integration.py
import httpx
from typing import Dict, List

class VCCIIntegration:
    def __init__(self, vcci_base_url: str, jwt_token: str):
        self.base_url = vcci_base_url
        self.headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json"
        }

    async def get_scope3_emissions(self, org_id: str, period: str) -> Dict:
        """Fetch Scope 3 emissions from VCCI app"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/emissions/scope3",
                headers=self.headers,
                params={"org_id": org_id, "period": period},
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()

    async def get_ghg_inventory(self, org_id: str, year: int) -> Dict:
        """Fetch complete GHG inventory from VCCI app"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/reports/ghg-inventory/{year}",
                headers=self.headers,
                params={"org_id": org_id},
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()

# Usage in CSRD report generation
@app.post("/api/v1/reports/csrd")
async def generate_csrd_report(
    org_id: str,
    period: str,
    current_user: dict = Depends(get_current_user)
):
    # Initialize VCCI integration
    vcci = VCCIIntegration(
        vcci_base_url=os.getenv("VCCI_BASE_URL", "http://vcci-backend-api:8000"),
        jwt_token=current_user["token"]  # Pass through user's token
    )

    # Fetch Scope 3 emissions from VCCI
    try:
        scope3_data = await vcci.get_scope3_emissions(org_id, period)
    except httpx.HTTPError as e:
        # Handle error (use defaults or fail gracefully)
        scope3_data = None

    # Generate CSRD report with VCCI data
    report = await generate_report(org_id, period, scope3_data)
    return report
```

### API Endpoints for Cross-App Integration

#### VCCI Exposes

```python
# VCCI: api/integration.py
@app.get("/api/integration/scope3-summary")
async def get_scope3_summary(
    org_id: str,
    period: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get Scope 3 emissions summary for CSRD E1 reporting

    Returns:
    {
        "org_id": "uuid",
        "period": "2025-Q1",
        "total_emissions_tco2": 12345.67,
        "by_category": {
            "cat1_purchased_goods": 5000,
            "cat2_capital_goods": 1000,
            ...
        },
        "data_quality": "high",
        "calculation_date": "2025-11-08T10:00:00Z"
    }
    """
    # Implementation
    pass

@app.get("/api/integration/suppliers/{supplier_id}")
async def get_supplier_details(
    supplier_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get supplier details (for CBAM or CSRD supply chain disclosures)"""
    pass
```

#### CBAM Exposes

```python
# CBAM: api/integration.py
@app.get("/api/integration/cbam-imports-summary")
async def get_cbam_imports_summary(
    org_id: str,
    quarter: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get CBAM imports summary for CSRD E1 climate disclosures

    Returns:
    {
        "org_id": "uuid",
        "quarter": "2025-Q4",
        "total_shipments": 150,
        "total_embedded_emissions_tco2": 456.78,
        "by_product_group": {
            "steel": 300.5,
            "cement": 100.2,
            "aluminum": 56.08
        },
        "top_origin_countries": ["CN", "TR", "IN"]
    }
    """
    pass
```

---

## 3. Event-Driven Integration (Message Queue)

### Message Queue Architecture

```
Publisher                   Exchange                 Queue                Consumer
(VCCI)                     (RabbitMQ)               (RabbitMQ)           (CSRD)

Calculate ──────────────> greenlang.emissions ──> csrd.emissions.sync ──> Process
Scope 3                    .calculated                                      & Store
Emissions
```

### Implementation

#### VCCI Publishing Events

```python
# VCCI: services/event_publisher.py
import aio_pika
import json

class EventPublisher:
    def __init__(self, rabbitmq_url: str):
        self.rabbitmq_url = rabbitmq_url
        self.connection = None
        self.channel = None

    async def connect(self):
        """Connect to RabbitMQ"""
        self.connection = await aio_pika.connect_robust(self.rabbitmq_url)
        self.channel = await self.connection.channel()

    async def publish_emissions_calculated(
        self,
        org_id: str,
        period: str,
        emissions_data: dict
    ):
        """Publish emissions calculated event"""
        exchange = await self.channel.declare_exchange(
            "greenlang.emissions.calculated",
            aio_pika.ExchangeType.TOPIC,
            durable=True
        )

        message_body = {
            "event": "emissions_calculated",
            "org_id": org_id,
            "period": period,
            "data": emissions_data,
            "timestamp": datetime.utcnow().isoformat()
        }

        message = aio_pika.Message(
            body=json.dumps(message_body).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            content_type="application/json"
        )

        routing_key = f"vcci.scope3.{org_id}"
        await exchange.publish(message, routing_key=routing_key)

# Usage in VCCI calculator
async def calculate_emissions(org_id: str, period: str):
    # ... perform calculation ...
    emissions_data = {...}

    # Publish event
    publisher = EventPublisher(os.getenv("RABBITMQ_URL"))
    await publisher.connect()
    await publisher.publish_emissions_calculated(org_id, period, emissions_data)
```

#### CSRD Consuming Events

```python
# CSRD: services/event_consumer.py
import aio_pika
import json

class EventConsumer:
    def __init__(self, rabbitmq_url: str):
        self.rabbitmq_url = rabbitmq_url
        self.connection = None
        self.channel = None

    async def connect(self):
        """Connect to RabbitMQ"""
        self.connection = await aio_pika.connect_robust(self.rabbitmq_url)
        self.channel = await self.connection.channel()

    async def consume_emissions_events(self):
        """Consume emissions calculated events"""
        # Declare queue
        queue = await self.channel.declare_queue(
            "csrd.emissions.sync",
            durable=True
        )

        # Bind to exchange
        exchange = await self.channel.declare_exchange(
            "greenlang.emissions.calculated",
            aio_pika.ExchangeType.TOPIC,
            durable=True
        )

        await queue.bind(exchange, routing_key="vcci.scope3.#")

        # Start consuming
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    body = json.loads(message.body.decode())
                    await self.handle_emissions_event(body)

    async def handle_emissions_event(self, event_data: dict):
        """Handle emissions calculated event"""
        org_id = event_data["org_id"]
        period = event_data["period"]
        emissions = event_data["data"]

        # Store in CSRD database for E1 reporting
        await store_scope3_emissions(org_id, period, emissions)

# Run consumer as background task
@app.on_event("startup")
async def startup_event():
    consumer = EventConsumer(os.getenv("RABBITMQ_URL"))
    await consumer.connect()
    asyncio.create_task(consumer.consume_emissions_events())
```

---

## 4. Shared Database for Users & Organizations

### Database Schema

```sql
-- shared_db: users and organizations
CREATE DATABASE shared_db;

\c shared_db

-- Organizations table (shared by all apps)
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    legal_name VARCHAR(255),
    country VARCHAR(2),
    industry VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Users table (shared by all apps)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    org_id UUID REFERENCES organizations(id),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- User roles per app
CREATE TABLE user_app_roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    app_name VARCHAR(50) NOT NULL,  -- 'cbam', 'csrd', 'vcci'
    roles JSONB NOT NULL,  -- ["admin", "analyst", "viewer"]
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, app_name)
);

-- Cross-app data sync table
CREATE TABLE cross_app_sync (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_app VARCHAR(50) NOT NULL,
    target_app VARCHAR(50) NOT NULL,
    org_id UUID REFERENCES organizations(id),
    data_type VARCHAR(100) NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    synced_at TIMESTAMP,
    INDEX idx_target_org (target_app, org_id, synced_at)
);
```

### Usage in Applications

```python
# Each app connects to shared_db for user/org data
SHARED_DATABASE_URL = os.getenv("SHARED_DATABASE_URL")

async def get_organization(org_id: str):
    """Fetch organization (shared across all apps)"""
    async with get_db_connection(SHARED_DATABASE_URL) as conn:
        result = await conn.fetchrow(
            "SELECT * FROM organizations WHERE id = $1",
            org_id
        )
        return result

async def get_user_by_email(email: str):
    """Fetch user (shared across all apps)"""
    async with get_db_connection(SHARED_DATABASE_URL) as conn:
        result = await conn.fetchrow(
            "SELECT * FROM users WHERE email = $1",
            email
        )
        return result

async def get_user_roles(user_id: str, app_name: str):
    """Get user roles for specific app"""
    async with get_db_connection(SHARED_DATABASE_URL) as conn:
        result = await conn.fetchrow(
            "SELECT roles FROM user_app_roles WHERE user_id = $1 AND app_name = $2",
            user_id, app_name
        )
        return result["roles"] if result else []
```

---

## 5. Unified Dashboard

### Data Aggregation Service

```python
# unified_dashboard_service.py
class UnifiedDashboardService:
    def __init__(self, jwt_token: str):
        self.cbam_client = CBAMClient(jwt_token)
        self.csrd_client = CSRDClient(jwt_token)
        self.vcci_client = VCCIClient(jwt_token)

    async def get_unified_dashboard_data(self, org_id: str) -> dict:
        """Fetch data from all 3 apps and combine"""

        # Fetch data from all apps in parallel
        cbam_data, csrd_data, vcci_data = await asyncio.gather(
            self.cbam_client.get_summary(org_id),
            self.csrd_client.get_summary(org_id),
            self.vcci_client.get_summary(org_id),
            return_exceptions=True
        )

        # Handle errors gracefully
        if isinstance(cbam_data, Exception):
            cbam_data = {"error": str(cbam_data)}
        if isinstance(csrd_data, Exception):
            csrd_data = {"error": str(csrd_data)}
        if isinstance(vcci_data, Exception):
            vcci_data = {"error": str(vcci_data)}

        # Combine data
        return {
            "org_id": org_id,
            "cbam": {
                "total_shipments_ytd": cbam_data.get("total_shipments", 0),
                "total_emissions_tco2": cbam_data.get("total_emissions", 0),
                "last_report_date": cbam_data.get("last_report_date")
            },
            "csrd": {
                "reporting_year": csrd_data.get("year"),
                "esrs_coverage_pct": csrd_data.get("coverage_pct", 0),
                "last_submission_date": csrd_data.get("last_submission")
            },
            "vcci": {
                "scope3_emissions_ytd": vcci_data.get("scope3_total", 0),
                "suppliers_engaged": vcci_data.get("suppliers_engaged", 0),
                "pcf_coverage_pct": vcci_data.get("pcf_coverage", 0)
            },
            "combined_metrics": {
                "total_carbon_footprint": (
                    cbam_data.get("total_emissions", 0) +
                    vcci_data.get("scope3_total", 0)
                )
            }
        }
```

---

## Integration Testing

### End-to-End Integration Test

```python
# tests/integration/test_cross_app_integration.py
import pytest
import httpx

@pytest.mark.asyncio
async def test_vcci_to_csrd_emissions_sync():
    """Test VCCI emissions sync to CSRD"""

    # 1. Calculate emissions in VCCI
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://vcci-backend:8000/api/emissions/calculate",
            json={
                "org_id": "test-org",
                "period": "2025-Q1",
                "transactions": [...]
            },
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        vcci_job_id = response.json()["job_id"]

    # 2. Wait for calculation to complete
    await asyncio.sleep(10)

    # 3. Verify event was published
    # (Check RabbitMQ or use test consumer)

    # 4. Check CSRD received the data
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://csrd-web:8002/api/v1/scope3-data",
            params={"org_id": "test-org", "period": "2025-Q1"},
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        csrd_data = response.json()

        # Verify data matches
        assert csrd_data["total_emissions"] > 0

@pytest.mark.asyncio
async def test_unified_dashboard():
    """Test unified dashboard aggregates data from all apps"""

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://api-gateway/api/dashboard/unified",
            params={"org_id": "test-org"},
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        data = response.json()

        # Verify all app data present
        assert "cbam" in data
        assert "csrd" in data
        assert "vcci" in data
        assert "combined_metrics" in data
```

---

## Monitoring Integration Health

### Health Check Endpoint

```python
@app.get("/api/health/integrations")
async def check_integrations_health():
    """Check health of all cross-app integrations"""

    results = {}

    # Check VCCI connection
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{VCCI_BASE_URL}/health/live",
                timeout=5.0
            )
            results["vcci"] = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception as e:
        results["vcci"] = f"error: {str(e)}"

    # Check CBAM connection
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{CBAM_BASE_URL}/health",
                timeout=5.0
            )
            results["cbam"] = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception as e:
        results["cbam"] = f"error: {str(e)}"

    # Check message queue
    try:
        connection = await aio_pika.connect_robust(RABBITMQ_URL)
        await connection.close()
        results["message_queue"] = "healthy"
    except Exception as e:
        results["message_queue"] = f"error: {str(e)}"

    # Check shared database
    try:
        async with get_db_connection(SHARED_DATABASE_URL) as conn:
            await conn.fetchval("SELECT 1")
        results["shared_database"] = "healthy"
    except Exception as e:
        results["shared_database"] = f"error: {str(e)}"

    overall_health = all(v == "healthy" for v in results.values())

    return {
        "status": "healthy" if overall_health else "degraded",
        "components": results
    }
```

---

## Troubleshooting

### Common Issues

**1. JWT Token Not Recognized**
- Ensure JWT_SECRET is the same across all apps
- Check token expiry
- Verify token format (Bearer <token>)

**2. Message Queue Events Not Received**
- Check RabbitMQ connection
- Verify exchange and queue bindings
- Check routing keys match

**3. Cross-App API Call Timeout**
- Increase timeout duration
- Check network connectivity
- Verify target service is healthy

**4. Data Sync Delays**
- Check message queue lag
- Verify consumer is running
- Check for errors in consumer logs

---

**Document Owner:** Integration Team
**Last Updated:** 2025-11-08
