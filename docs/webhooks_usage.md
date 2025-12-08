# Webhook Manager Usage Guide

## Overview

The WebhookSubscriberManager provides event-driven webhook delivery for Process Heat agents. It enables external systems to subscribe to agent events and receive callbacks with HMAC-SHA256 signatures for secure verification.

## Supported Events

Process Heat agents emit the following webhook events:

- `calculation.completed` - Calculation finished successfully
- `calculation.failed` - Calculation encountered an error
- `alarm.triggered` - Alarm condition detected
- `alarm.cleared` - Alarm condition resolved
- `model.deployed` - ML model deployed to production
- `model.degraded` - ML model performance degraded
- `compliance.violation` - Compliance rule violation detected
- `compliance.report_ready` - Compliance report generated

## Basic Usage

### Initialize the Manager

```python
from greenlang.infrastructure.api.webhooks import WebhookSubscriberManager

manager = WebhookSubscriberManager(base_url="/api/v1/webhooks")
```

### Register a Webhook

```python
webhook_id = manager.register_webhook(
    url="https://api.example.com/events",
    events=["calculation.completed", "alarm.triggered"],
    secret="my-shared-secret-key"
)
```

### List Registered Webhooks

```python
webhooks = manager.list_webhooks()
for webhook in webhooks:
    print(f"ID: {webhook.webhook_id}")
    print(f"URL: {webhook.url}")
    print(f"Events: {webhook.events}")
    print(f"Health: {webhook.health_status}")
```

### Trigger an Event

```python
import asyncio

async def send_event():
    payload = {
        "calculation_id": "calc_abc123",
        "result": 45.67,
        "timestamp": "2025-12-07T10:30:00Z"
    }

    delivery_id = await manager.trigger_webhook(
        "calculation.completed",
        payload
    )

    print(f"Delivery ID: {delivery_id}")

    # Check delivery status
    delivery = manager.deliveries[delivery_id]
    print(f"Status: {delivery.status}")
    print(f"HTTP Status: {delivery.http_status}")

asyncio.run(send_event())
```

### Unregister a Webhook

```python
if manager.unregister_webhook(webhook_id):
    print("Webhook unregistered successfully")
```

## FastAPI Integration

### Setup Routes

```python
from fastapi import FastAPI
from greenlang.infrastructure.api.webhooks import WebhookSubscriberManager

app = FastAPI()

manager = WebhookSubscriberManager(base_url="/api/v1/webhooks")
app.include_router(manager.router)
```

### Available Endpoints

```
POST /api/v1/webhooks
  Register a new webhook subscriber

  Request body:
  {
    "url": "https://api.example.com/webhook",
    "events": ["calculation.completed"],
    "secret": "my-secret",
    "metadata": {"env": "production"}
  }

  Response:
  {
    "webhook_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "Webhook registered successfully"
  }

DELETE /api/v1/webhooks/{webhook_id}
  Unregister a webhook

  Response:
  {
    "webhook_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "Webhook unregistered successfully"
  }

GET /api/v1/webhooks
  List all registered webhooks

  Response: Array of webhook definitions

POST /api/v1/webhooks/{webhook_id}/test
  Test webhook delivery

  Response:
  {
    "delivery_id": "...",
    "webhook_id": "...",
    "status": "sent",
    "message": "Test webhook sent"
  }
```

## Signature Verification

Webhooks are signed using HMAC-SHA256. The signature is sent in the `X-Webhook-Signature` header.

### Verifying Signatures (Python)

```python
import hmac
import hashlib
import json

def verify_webhook(request_body, signature_header, secret):
    payload_str = json.dumps(json.loads(request_body), sort_keys=True)
    expected = hmac.new(
        secret.encode(),
        payload_str.encode(),
        hashlib.sha256
    ).hexdigest()

    provided_sig = signature_header.replace("sha256=", "")
    return hmac.compare_digest(expected, provided_sig)
```

### Webhook Headers

Every webhook delivery includes security headers:

```
X-Webhook-Signature: sha256=<HMAC-SHA256 signature>
X-Webhook-ID: <delivery_id>
X-Event-Type: calculation.completed
Content-Type: application/json
```

## Delivery Guarantees

### Retry Strategy

Failed deliveries are retried with exponential backoff:
- Attempt 1: Immediate
- Attempt 2: Wait 2 seconds (2^1)
- Attempt 3: Wait 4 seconds (2^2)
- Attempt 4: Wait 8 seconds (2^3)

Max 3 attempts total.

### Health Monitoring

Webhooks are monitored for health:

- Track consecutive failures per webhook
- After 5+ consecutive failures, webhook is marked "unhealthy" and deactivated
- Deactivated webhooks no longer receive events
- Manually re-activate by calling `webhook.is_active = True`

### Delivery Status

```python
from greenlang.infrastructure.api.webhooks import WebhookStatus

# Status values:
# - PENDING: Awaiting delivery
# - SENT: Successfully delivered (2xx HTTP status)
# - FAILED: All retry attempts exhausted
# - RETRYING: Waiting to retry
```

## Event Payload Structure

All webhook payloads follow a consistent structure:

```json
{
  "event_type": "calculation.completed",
  "timestamp": "2025-12-07T10:30:00Z",
  "data": {
    "calculation_id": "calc_abc123",
    "result": 45.67,
    "framework": "eudr"
  }
}
```

## Best Practices

1. **Secret Management**
   - Use strong secrets (32+ characters)
   - Rotate secrets periodically
   - Store secrets in secure vaults

2. **URL Validation**
   - Use HTTPS endpoints in production
   - Validate SSL certificates
   - Implement timeout handling

3. **Signature Verification**
   - Always verify signatures in your webhook handler
   - Use timing-safe comparison (like `hmac.compare_digest`)
   - Replay attack prevention with timestamp header

4. **Event Processing**
   - Return 2xx HTTP status within 30 seconds
   - Process events asynchronously if needed
   - Log all webhook events for debugging

5. **Monitoring**
   - Track webhook health scores
   - Alert on repeated failures
   - Monitor event latency and throughput

## Example: Complete Integration

```python
from fastapi import FastAPI, Request
from greenlang.infrastructure.api.webhooks import WebhookSubscriberManager
import hmac
import hashlib
import json

app = FastAPI()
manager = WebhookSubscriberManager()
app.include_router(manager.router)

# Register a webhook for your service
@app.on_event("startup")
async def register_webhook():
    await manager.trigger_webhook(
        "startup",
        {"timestamp": datetime.utcnow().isoformat()}
    )

# Receive webhook from another service
@app.post("/my-webhook-receiver")
async def receive_webhook(request: Request):
    body = await request.body()
    payload = json.loads(body)
    signature = request.headers.get("X-Webhook-Signature", "")

    # Verify signature
    if not manager.verify_signature(
        payload,
        signature.replace("sha256=", ""),
        "my-shared-secret"
    ):
        return {"error": "Invalid signature"}, 401

    # Process event
    event_type = payload.get("event_type")
    print(f"Received event: {event_type}")

    return {"status": "received"}

# Trigger event to subscribers
@app.post("/calculate")
async def calculate(data: dict):
    result = perform_calculation(data)

    await manager.trigger_webhook(
        "calculation.completed",
        {
            "calculation_id": str(uuid4()),
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

    return {"result": result}
```

## Troubleshooting

### Webhook Not Delivering

1. Check if webhook is active: `webhook.is_active == True`
2. Check health status: `webhook.health_status == "healthy"`
3. Verify event type is subscribed
4. Check logs for delivery errors

### Signature Verification Failing

1. Ensure secret matches exactly
2. Verify JSON payload is sorted by keys
3. Check signature format (should include "sha256=" prefix)
4. Use timing-safe comparison

### High Failure Rate

1. Check webhook URL is accessible
2. Verify endpoint returns 2xx status
3. Monitor response times (max 30s timeout)
4. Enable webhook test endpoint to diagnose

## Performance Characteristics

- Registration: O(1) constant time
- Delivery: Non-blocking async with configurable timeout
- Retry queue: In-memory (survives process restart with persistent backend)
- Signature verification: O(n) where n is payload size

## Configuration

```python
manager = WebhookSubscriberManager(
    base_url="/api/v1/webhooks"  # FastAPI route prefix
)

# Adjust timeout (default 10s)
manager._http_client = httpx.AsyncClient(timeout=30)

# Manual retry configuration
max_attempts = 3
retry_delays = [2, 4, 8]  # exponential backoff
```
