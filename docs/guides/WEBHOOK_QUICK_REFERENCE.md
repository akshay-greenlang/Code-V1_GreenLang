# Webhook System - Quick Reference

## Core Implementation Files

| File | Lines | Purpose |
|------|-------|---------|
| `greenlang/infrastructure/api/webhooks.py` | 399 | WebhookSubscriberManager implementation |
| `tests/unit/test_webhook_manager.py` | 489 | Comprehensive unit tests (35+ test methods) |
| `docs/webhooks_usage.md` | 357 | Complete user documentation |

## Quick Start

### 1. Initialize
```python
from greenlang.infrastructure.api.webhooks import WebhookSubscriberManager

manager = WebhookSubscriberManager()
```

### 2. Register Webhook
```python
webhook_id = manager.register_webhook(
    url="https://api.example.com/webhook",
    events=["calculation.completed"],
    secret="shared-secret-key"
)
```

### 3. Send Event
```python
import asyncio

async def send():
    await manager.trigger_webhook(
        "calculation.completed",
        {"result": 123.45}
    )

asyncio.run(send())
```

### 4. Verify Signature (in your webhook handler)
```python
if manager.verify_signature(payload, signature, secret):
    process(payload)
else:
    reject()
```

## Class Methods

### WebhookSubscriberManager

| Method | Signature | Returns | Notes |
|--------|-----------|---------|-------|
| `register_webhook` | `(url, events, secret, metadata=None)` | `str` (webhook_id) | Creates new subscriber |
| `unregister_webhook` | `(webhook_id)` | `bool` | Returns True if found |
| `list_webhooks` | `()` | `List[WebhookModel]` | All subscribers |
| `trigger_webhook` | `(event_type, payload, webhook_ids=None)` | `str` (delivery_id) | Async event dispatch |
| `verify_signature` | `(payload, signature, secret)` | `bool` | HMAC verification |
| `cleanup` | `()` | `None` | Async cleanup (close HTTP client) |

## Data Models

### WebhookModel
```python
webhook_id: str                          # UUID
url: str                                 # Callback URL
events: List[str]                        # ["calculation.completed"]
secret: str                              # HMAC signing secret
is_active: bool                          # Active/inactive toggle
health_status: str                       # "healthy" or "unhealthy"
consecutive_failures: int                # Failure counter
last_triggered_at: Optional[datetime]    # Last successful delivery
created_at: datetime                     # Registration timestamp
metadata: Dict[str, Any]                 # Custom metadata
```

### WebhookDelivery
```python
delivery_id: str                         # UUID
webhook_id: str                          # Target webhook
event_type: str                          # "calculation.completed"
payload: Dict[str, Any]                  # Event data
signature: str                           # HMAC-SHA256 hex
status: WebhookStatus                    # PENDING|SENT|FAILED|RETRYING
attempt: int                             # Current attempt (1-3)
http_status: Optional[int]               # Response status
error_message: Optional[str]             # Error details
created_at: datetime                     # Delivery created
sent_at: Optional[datetime]              # Delivery sent
provenance_hash: str                     # SHA-256 audit trail
```

## Event Types

```python
PROCESS_HEAT_EVENTS = {
    "calculation.completed",      # Calculation finished
    "calculation.failed",         # Calculation error
    "alarm.triggered",            # Alert condition
    "alarm.cleared",              # Alert resolved
    "model.deployed",             # ML model deployed
    "model.degraded",             # ML model performance issue
    "compliance.violation",       # Rule violation
    "compliance.report_ready",    # Report generated
}
```

## HTTP Headers

Webhook deliveries include:

```
X-Webhook-Signature: sha256=<64-char-hex>
X-Webhook-ID: <delivery_uuid>
X-Event-Type: <event_type>
Content-Type: application/json
```

## Retry Strategy

| Attempt | Status | Wait Before Next |
|---------|--------|------------------|
| 1 | PENDING | - |
| 2 | PENDING → RETRYING | 2 seconds |
| 3 | RETRYING | 4 seconds |
| 4 | RETRYING | 8 seconds |
| Final | SENT or FAILED | - |

## Health Monitoring

```python
webhook = manager.list_webhooks()[0]

print(f"Health: {webhook.health_status}")           # "healthy" or "unhealthy"
print(f"Failures: {webhook.consecutive_failures}")  # 0-5+
print(f"Last triggered: {webhook.last_triggered_at}")

# Auto-deactivation after 5+ consecutive failures
if webhook.consecutive_failures >= 5:
    webhook.is_active = False  # Webhooks marked unhealthy automatically
```

## FastAPI Routes

### Register
```http
POST /api/v1/webhooks
Content-Type: application/json

{
  "url": "https://api.example.com/webhook",
  "events": ["calculation.completed"],
  "secret": "my-secret",
  "metadata": {"env": "prod"}
}

HTTP/1.1 201 Created
{
  "webhook_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Webhook registered successfully"
}
```

### List
```http
GET /api/v1/webhooks

HTTP/1.1 200 OK
[
  {
    "webhook_id": "550e8400-e29b-41d4-a716-446655440000",
    "url": "https://api.example.com/webhook",
    "events": ["calculation.completed"],
    "is_active": true,
    "health_status": "healthy",
    ...
  }
]
```

### Unregister
```http
DELETE /api/v1/webhooks/550e8400-e29b-41d4-a716-446655440000

HTTP/1.1 200 OK
{
  "webhook_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Webhook unregistered successfully"
}
```

### Test
```http
POST /api/v1/webhooks/550e8400-e29b-41d4-a716-446655440000/test

HTTP/1.1 200 OK
{
  "delivery_id": "...",
  "webhook_id": "...",
  "status": "sent",
  "message": "Test webhook sent"
}
```

## Signature Verification (Python)

```python
import hmac
import hashlib
import json

# In your webhook receiver:
def verify_webhook(body: bytes, signature: str, secret: str) -> bool:
    payload = json.loads(body)
    payload_str = json.dumps(payload, sort_keys=True, default=str)
    expected = hmac.new(
        secret.encode(),
        payload_str.encode(),
        hashlib.sha256
    ).hexdigest()

    provided = signature.replace("sha256=", "")
    return hmac.compare_digest(expected, provided)
```

## Integration Example

```python
from fastapi import FastAPI, Request
from greenlang.infrastructure.api.webhooks import WebhookSubscriberManager

app = FastAPI()
manager = WebhookSubscriberManager()
app.include_router(manager.router)

@app.post("/calculate")
async def calculate(data: dict):
    result = 45.67

    # Send event to subscribers
    await manager.trigger_webhook(
        "calculation.completed",
        {
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

    return {"result": result}

@app.shutdown_event
async def shutdown():
    await manager.cleanup()
```

## Testing

```python
# Test registration
manager.register_webhook(
    url="https://test.com/webhook",
    events=["calculation.completed"],
    secret="test-secret"
)

# Test signature
sig = manager._create_signature({"data": "test"}, "secret")
assert manager.verify_signature({"data": "test"}, sig, "secret")

# Test invalid event
with pytest.raises(ValueError):
    manager.register_webhook(..., events=["invalid"], ...)
```

## Performance

- Registration: O(1)
- Listing: O(n) where n = webhooks
- Delivery: Async, non-blocking
- Signature: O(payload_size)

## Configuration

```python
manager = WebhookSubscriberManager(
    base_url="/api/v1/webhooks"  # FastAPI route prefix
)

# Optional: Change HTTP timeout
import httpx
manager._http_client = httpx.AsyncClient(timeout=30)
```

## Error Handling

```python
try:
    webhook_id = manager.register_webhook(
        url="...",
        events=["invalid.event"],
        secret="..."
    )
except ValueError as e:
    print(f"Invalid event: {e}")
```

## Common Patterns

### Send Event to Specific Webhooks
```python
await manager.trigger_webhook(
    "calculation.completed",
    payload,
    webhook_ids=["id1", "id2"]  # Only these webhooks
)
```

### Deactivate Unhealthy Webhook
```python
for webhook in manager.list_webhooks():
    if webhook.health_status == "unhealthy":
        webhook.is_active = False
```

### Check Delivery Status
```python
delivery = manager.deliveries[delivery_id]
print(f"Status: {delivery.status}")
print(f"HTTP: {delivery.http_status}")
print(f"Error: {delivery.error_message}")
```

## Dependencies

```
httpx>=0.23.0
pydantic>=1.10.0
fastapi>=0.90.0  # Optional: for REST endpoints
```

## Files Reference

- Implementation: `/c/Users/aksha/Code-V1_GreenLang/greenlang/infrastructure/api/webhooks.py`
- Tests: `/c/Users/aksha/Code-V1_GreenLang/tests/unit/test_webhook_manager.py`
- Docs: `/c/Users/aksha/Code-V1_GreenLang/docs/webhooks_usage.md`
- Summary: `/c/Users/aksha/Code-V1_GreenLang/TASK-135-IMPLEMENTATION.md`
