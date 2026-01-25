# Webhooks Guide

## GL-VCCI Platform Webhooks

Webhooks allow the GL-VCCI platform to notify your application in real-time when events occur, eliminating the need for polling.

---

## Overview

Instead of polling the API for job status or data changes, configure webhooks to receive push notifications when events occur.

**Benefits:**
- Real-time notifications
- Reduced API calls
- Lower latency
- Simpler application logic
- No rate limit impact

---

## Supported Events

### Data Processing Events

| Event Type | Description | Frequency |
|-----------|-------------|-----------|
| `job.completed` | Batch processing job finished | Per job |
| `job.failed` | Batch processing job failed | Per job |
| `transaction.processed` | Transaction ingested and validated | Per transaction |
| `transaction.failed` | Transaction processing failed | Per transaction |

### Calculation Events

| Event Type | Description | Frequency |
|-----------|-------------|-----------|
| `calculation.completed` | Emissions calculation finished | Per calculation |
| `calculation.failed` | Emissions calculation failed | Per calculation |
| `report.generated` | Report generation completed | Per report |
| `report.failed` | Report generation failed | Per report |

### Supplier Events

| Event Type | Description | Frequency |
|-----------|-------------|-----------|
| `supplier.resolved` | Supplier entity resolved | Per resolution |
| `supplier.enriched` | Supplier data enriched | Per enrichment |
| `pcf.received` | Supplier PCF received | Per PCF |
| `campaign.response` | Supplier responded to campaign | Per response |

### Review Queue Events

| Event Type | Description | Frequency |
|-----------|-------------|-----------|
| `review.created` | Item queued for review | Per item |
| `review.resolved` | Review item resolved | Per item |

---

## Webhook Configuration

### Create a Webhook Endpoint

```http
POST /v2/webhooks
Authorization: Bearer YOUR_ACCESS_TOKEN
Content-Type: application/json

{
  "url": "https://yourapp.com/webhooks/vcci",
  "events": [
    "job.completed",
    "job.failed",
    "calculation.completed"
  ],
  "description": "Production webhook endpoint",
  "secret": "whsec_abc123xyz789"
}
```

**Response:**
```json
{
  "id": "webhook_abc123",
  "url": "https://yourapp.com/webhooks/vcci",
  "events": [
    "job.completed",
    "job.failed",
    "calculation.completed"
  ],
  "status": "active",
  "created_at": "2024-11-06T10:00:00Z"
}
```

### List Webhooks

```http
GET /v2/webhooks
Authorization: Bearer YOUR_ACCESS_TOKEN
```

### Update Webhook

```http
PATCH /v2/webhooks/webhook_abc123
Authorization: Bearer YOUR_ACCESS_TOKEN
Content-Type: application/json

{
  "events": [
    "job.completed",
    "calculation.completed",
    "report.generated"
  ]
}
```

### Delete Webhook

```http
DELETE /v2/webhooks/webhook_abc123
Authorization: Bearer YOUR_ACCESS_TOKEN
```

---

## Webhook Payload Structure

All webhooks follow a consistent structure:

```json
{
  "id": "evt_abc123xyz789",
  "type": "job.completed",
  "created_at": "2024-11-06T10:30:00Z",
  "data": {
    "object": "job",
    "id": "job_abc123",
    "status": "completed",
    "results": {
      "success": 4850,
      "errors": 150
    }
  }
}
```

### Payload Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique event ID |
| `type` | string | Event type (e.g., `job.completed`) |
| `created_at` | string | ISO 8601 timestamp |
| `data.object` | string | Object type (job, calculation, etc.) |
| `data.id` | string | Object ID |
| `data.*` | varies | Event-specific data |

---

## Event Payloads

### job.completed

```json
{
  "id": "evt_123",
  "type": "job.completed",
  "created_at": "2024-11-06T10:30:00Z",
  "data": {
    "object": "job",
    "id": "job_abc123",
    "status": "completed",
    "job_type": "batch_transaction_upload",
    "results": {
      "processed": 5000,
      "success": 4850,
      "errors": 150,
      "warnings": 200
    },
    "started_at": "2024-11-06T10:00:00Z",
    "completed_at": "2024-11-06T10:30:00Z"
  }
}
```

### calculation.completed

```json
{
  "id": "evt_456",
  "type": "calculation.completed",
  "created_at": "2024-11-06T10:35:00Z",
  "data": {
    "object": "calculation",
    "id": "calc_abc123",
    "transaction_id": "txn_xyz789",
    "category": "1",
    "tier": "2",
    "emissions_kg_co2e": 1250.5,
    "uncertainty": {
      "range": {
        "lower_bound": 1000.4,
        "upper_bound": 1500.6
      }
    },
    "calculated_at": "2024-11-06T10:35:00Z"
  }
}
```

### pcf.received

```json
{
  "id": "evt_789",
  "type": "pcf.received",
  "created_at": "2024-11-06T11:00:00Z",
  "data": {
    "object": "pcf",
    "id": "pcf_abc123",
    "supplier_id": "sup_xyz789",
    "product_id": "prod_123",
    "product_name": "Steel beams",
    "pcf_value_kg_co2e": 850.2,
    "format": "pact",
    "verification": {
      "verified": true,
      "verifier": "TÜV SÜD"
    }
  }
}
```

---

## Signature Verification

All webhook requests include a signature for authentication and integrity verification.

### Signature Header

```http
POST /webhooks/vcci
X-VCCI-Signature: t=1699275600,v1=abc123def456...
Content-Type: application/json

{...webhook payload...}
```

### Verify Signature (Python)

```python
import hmac
import hashlib
import time

def verify_webhook_signature(payload, signature_header, secret):
    """
    Verify webhook signature to ensure request authenticity.

    Args:
        payload: Raw request body (bytes)
        signature_header: X-VCCI-Signature header value
        secret: Webhook secret from configuration

    Returns:
        bool: True if signature is valid
    """
    # Parse signature header
    parts = dict(item.split('=') for item in signature_header.split(','))
    timestamp = parts.get('t')
    signature = parts.get('v1')

    if not timestamp or not signature:
        return False

    # Check timestamp (reject if older than 5 minutes)
    current_time = int(time.time())
    if abs(current_time - int(timestamp)) > 300:
        return False

    # Compute expected signature
    signed_payload = f"{timestamp}.{payload.decode('utf-8')}"
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        signed_payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    # Compare signatures (constant-time comparison)
    return hmac.compare_digest(signature, expected_signature)

# Flask example
from flask import Flask, request, jsonify

app = Flask(__name__)
WEBHOOK_SECRET = "whsec_abc123xyz789"

@app.route('/webhooks/vcci', methods=['POST'])
def handle_webhook():
    # Get signature header
    signature = request.headers.get('X-VCCI-Signature')

    # Verify signature
    if not verify_webhook_signature(request.data, signature, WEBHOOK_SECRET):
        return jsonify({'error': 'Invalid signature'}), 401

    # Process webhook
    event = request.json
    process_webhook_event(event)

    return jsonify({'status': 'success'}), 200
```

### Verify Signature (Node.js)

```javascript
const crypto = require('crypto');

function verifyWebhookSignature(payload, signatureHeader, secret) {
  // Parse signature header
  const parts = {};
  signatureHeader.split(',').forEach(part => {
    const [key, value] = part.split('=');
    parts[key] = value;
  });

  const timestamp = parts.t;
  const signature = parts.v1;

  if (!timestamp || !signature) {
    return false;
  }

  // Check timestamp (reject if older than 5 minutes)
  const currentTime = Math.floor(Date.now() / 1000);
  if (Math.abs(currentTime - parseInt(timestamp)) > 300) {
    return false;
  }

  // Compute expected signature
  const signedPayload = `${timestamp}.${payload}`;
  const expectedSignature = crypto
    .createHmac('sha256', secret)
    .update(signedPayload)
    .digest('hex');

  // Compare signatures (constant-time comparison)
  return crypto.timingSafeEqual(
    Buffer.from(signature),
    Buffer.from(expectedSignature)
  );
}

// Express example
const express = require('express');
const app = express();

const WEBHOOK_SECRET = 'whsec_abc123xyz789';

app.post('/webhooks/vcci', express.raw({type: 'application/json'}), (req, res) => {
  const signature = req.headers['x-vcci-signature'];

  // Verify signature
  if (!verifyWebhookSignature(req.body.toString(), signature, WEBHOOK_SECRET)) {
    return res.status(401).json({error: 'Invalid signature'});
  }

  // Process webhook
  const event = JSON.parse(req.body);
  processWebhookEvent(event);

  res.json({status: 'success'});
});
```

---

## Retry Logic

If your webhook endpoint fails, the GL-VCCI platform will automatically retry delivery.

### Retry Schedule

| Attempt | Delay | Total Time |
|---------|-------|------------|
| 1 | Immediate | 0s |
| 2 | 5 seconds | 5s |
| 3 | 30 seconds | 35s |
| 4 | 5 minutes | 5m 35s |
| 5 | 30 minutes | 35m 35s |
| 6 | 2 hours | 2h 35m 35s |
| 7 | 6 hours | 8h 35m 35s |

After 7 failed attempts, the webhook is disabled and you'll receive an email notification.

### Success Criteria

Your endpoint must:
- Return HTTP 2xx status code
- Respond within 30 seconds
- Return valid JSON (optional)

```python
@app.route('/webhooks/vcci', methods=['POST'])
def handle_webhook():
    try:
        event = request.json

        # Process webhook (should be fast)
        process_webhook_event(event)

        # Return success immediately
        return jsonify({'status': 'success'}), 200

    except Exception as e:
        # Log error but still return 200 if event was received
        logger.error(f"Webhook processing error: {e}")

        # Only return 5xx if you want GL-VCCI to retry
        # return jsonify({'error': str(e)}), 500

        # Return 200 to acknowledge receipt
        return jsonify({'status': 'received'}), 200
```

---

## Idempotency

Webhooks may be delivered more than once. Make your webhook handler idempotent.

### Idempotent Processing

```python
import redis

cache = redis.Redis(host='localhost', port=6379)

def process_webhook_event(event):
    event_id = event['id']

    # Check if already processed
    if cache.get(f"webhook:{event_id}"):
        print(f"Event {event_id} already processed. Skipping.")
        return

    # Process event
    if event['type'] == 'job.completed':
        handle_job_completed(event['data'])
    elif event['type'] == 'calculation.completed':
        handle_calculation_completed(event['data'])

    # Mark as processed (TTL: 7 days)
    cache.setex(f"webhook:{event_id}", 604800, "1")
```

---

## Testing Webhooks

### Manual Test Event

Send a test webhook to verify your endpoint:

```http
POST /v2/webhooks/webhook_abc123/test
Authorization: Bearer YOUR_ACCESS_TOKEN
Content-Type: application/json

{
  "event_type": "job.completed"
}
```

This sends a sample `job.completed` event to your webhook URL.

### Local Development with ngrok

```bash
# Start ngrok tunnel
ngrok http 5000

# Use ngrok URL for webhook
# https://abc123.ngrok.io/webhooks/vcci
```

### Webhook Testing Service

Use https://webhook.site to inspect webhook payloads during development.

---

## Best Practices

### DO ✅

- Verify signatures on all webhook requests
- Respond quickly (< 30 seconds)
- Process webhooks asynchronously (queue for background processing)
- Make handlers idempotent
- Log all webhook events
- Return 2xx for successfully received webhooks
- Use HTTPS endpoints only
- Monitor webhook health

### DON'T ❌

- Skip signature verification
- Perform long-running operations in webhook handler
- Return errors for successfully received events
- Expose webhook URLs publicly without authentication
- Use HTTP (non-encrypted) endpoints
- Process the same event multiple times

---

## Async Processing Pattern

```python
from flask import Flask, request, jsonify
from celery import Celery

app = Flask(__name__)
celery = Celery('tasks', broker='redis://localhost:6379')

@celery.task
def process_webhook_async(event):
    """Process webhook in background"""
    if event['type'] == 'job.completed':
        job_data = event['data']
        # Perform long-running operations
        send_email_notification(job_data)
        update_dashboard(job_data)
        sync_to_data_warehouse(job_data)

@app.route('/webhooks/vcci', methods=['POST'])
def handle_webhook():
    # Verify signature
    signature = request.headers.get('X-VCCI-Signature')
    if not verify_webhook_signature(request.data, signature, WEBHOOK_SECRET):
        return jsonify({'error': 'Invalid signature'}), 401

    # Queue for async processing
    event = request.json
    process_webhook_async.delay(event)

    # Return success immediately
    return jsonify({'status': 'queued'}), 200
```

---

## Monitoring Webhooks

### Webhook Logs

View webhook delivery logs:

```http
GET /v2/webhooks/webhook_abc123/logs?limit=100
Authorization: Bearer YOUR_ACCESS_TOKEN
```

**Response:**
```json
{
  "logs": [
    {
      "id": "log_123",
      "event_id": "evt_456",
      "event_type": "job.completed",
      "delivered_at": "2024-11-06T10:30:00Z",
      "status_code": 200,
      "response_time_ms": 150,
      "success": true
    },
    {
      "id": "log_124",
      "event_id": "evt_457",
      "event_type": "job.failed",
      "attempted_at": "2024-11-06T10:31:00Z",
      "status_code": 500,
      "response_time_ms": 2000,
      "success": false,
      "retry_count": 1,
      "next_retry_at": "2024-11-06T10:31:05Z"
    }
  ]
}
```

### Webhook Health Metrics

```http
GET /v2/webhooks/webhook_abc123/metrics
Authorization: Bearer YOUR_ACCESS_TOKEN
```

**Response:**
```json
{
  "webhook_id": "webhook_abc123",
  "metrics": {
    "total_deliveries": 10000,
    "successful_deliveries": 9950,
    "failed_deliveries": 50,
    "success_rate": 99.5,
    "avg_response_time_ms": 180,
    "p95_response_time_ms": 350,
    "p99_response_time_ms": 500
  },
  "period": "last_30_days"
}
```

---

## Troubleshooting

### Webhook Not Receiving Events

1. **Check webhook status:** Ensure webhook is `active`
2. **Verify URL:** Confirm URL is publicly accessible
3. **Check firewall:** Allow traffic from GL-VCCI IP ranges
4. **Review logs:** Check webhook delivery logs for errors
5. **Test endpoint:** Use test event to verify connectivity

### Signature Verification Failing

1. **Check secret:** Ensure using correct webhook secret
2. **Verify timestamp:** Check clock synchronization
3. **Raw body:** Use raw request body, not parsed JSON
4. **Character encoding:** Ensure UTF-8 encoding

### High Failure Rate

1. **Response time:** Optimize endpoint to respond < 30s
2. **Error handling:** Return 200 even if processing fails
3. **Async processing:** Move long operations to background queue
4. **Scaling:** Ensure endpoint can handle webhook volume

---

## IP Allowlist

For added security, allowlist these GL-VCCI IP ranges:

```
52.89.214.238/32
34.212.75.30/32
54.218.53.128/32
```

Update regularly as IP ranges may change. Subscribe to updates: https://status.vcci.greenlang.io

---

**Last Updated:** November 6, 2025
**Version:** 2.0.0
