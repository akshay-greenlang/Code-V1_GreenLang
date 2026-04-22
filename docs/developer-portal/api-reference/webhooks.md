# Webhooks

Register an HTTPS endpoint and GreenLang will push events to it when things change that your downstream systems care about — a factor you depend on getting deprecated, a new edition being published, a license class changing, an impact simulation completing.

**Implementation:** `greenlang/factors/webhooks.py`
**Event type constants:** `greenlang/factors/webhooks.py::WebhookEventType`

---

## The 11 event types

| Event | Fires when | Typical action |
|---|---|---|
| `factor.added` | A new factor is introduced in a newly-promoted edition. | Refresh your catalog cache. |
| `factor.updated` | A factor's numeric value, gas breakdown, uncertainty, or metadata changed across editions. | Recompute downstream emissions under the new edition. |
| `factor.deprecated` | A factor is marked superseded; `replacement_factor_id` is published. | Migrate references before the replacement cutover. |
| `factor.removed` | A factor physically purged (rare; 2+ years after deprecation). | Final cleanup. |
| `license.changed` | A factor's `license_class` changed (e.g. `open` → `restricted`). | Re-check redistribution eligibility. |
| `methodology.changed` | A method pack's selection rule, GWP-set set, or `allowed_statuses` changed. | Review whether your resolutions should be re-run. |
| `source.artifact_changed` | The upstream source file changed (eGRID publishes new annual PDF). | Usually informational; an edition promotion follows. |
| `source.unavailable` | An upstream source is temporarily unavailable (ecoinvent outage, IEA rate-limit). | Consider pausing batch jobs that depend on that source. |
| `source.breaking_change` | An upstream source published an incompatible schema change. | Legal / methodology review before the next edition. |
| `edition.published` | A new edition moved to `stable`. | Re-pin if you run on "latest stable" policy; ignore if pinned. |
| `impact_simulation.complete` | An async impact simulation job for your tenant finished. | Fetch the result body. |

Constants live on `WebhookEventType` (`greenlang/factors/webhooks.py`, line 33). `WebhookEventType.ALL` enumerates all 11.

---

## Registering a subscription

```bash
curl -sS -X POST "$GL_API_BASE/api/v1/webhooks/subscriptions" \
  -H "Authorization: Bearer $GL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "target_url": "https://hooks.acme-climate.com/greenlang",
    "event_types": [
      "factor.deprecated",
      "factor.updated",
      "edition.published",
      "license.changed"
    ],
    "secret": "<32-byte hex secret you generate>"
  }'
```

Response:

```json
{
  "subscription_id": "wh_sub_01HYP...",
  "tenant_id": "acme-climate",
  "target_url": "https://hooks.acme-climate.com/greenlang",
  "event_types": ["factor.deprecated","factor.updated","edition.published","license.changed"],
  "active": true,
  "created_at": "2026-04-22T14:33:02Z"
}
```

The `secret` is what the server uses for HMAC signing of deliveries. Store it — GreenLang will not show it again.

---

## Delivery shape

Each POST to your endpoint looks like:

```http
POST /greenlang HTTP/1.1
Host: hooks.acme-climate.com
Content-Type: application/json
User-Agent: GreenLang-Webhook/1.0
X-GreenLang-Event: factor.deprecated
X-GreenLang-Delivery-Id: dlv_01HYP...
X-GreenLang-Subscription-Id: wh_sub_01HYP...
X-GreenLang-Signature: sha256=<hex>
X-GreenLang-Timestamp: 2026-04-22T14:33:02Z

{
  "event_type": "factor.deprecated",
  "triggered_at": "2026-04-22T14:33:02Z",
  "delivery_id": "dlv_01HYP...",
  "payload": {
    "factor_id": "EF:US:electricity:2020:egrid",
    "deprecated_in_edition": "2027.Q1-electricity",
    "replacement_factor_id": "EF:US:electricity:2024:egrid",
    "reason": "eGRID 2024 release supersedes 2020 subregion SERC"
  }
}
```

---

## Verifying the signature

The signature is `hex(HMAC_SHA256(secret, timestamp + "." + raw_body))`. This binds the body to a timestamp so an attacker cannot replay an old delivery with a fresh timestamp.

### Node.js (Express)

```ts
import { createHmac, timingSafeEqual } from "node:crypto";
import express from "express";

const app = express();
app.use(express.raw({ type: "application/json" }));

app.post("/greenlang", (req, res) => {
  const ts = req.header("X-GreenLang-Timestamp")!;
  const sig = req.header("X-GreenLang-Signature")!.replace(/^sha256=/, "");
  const body = req.body as Buffer;

  const mac = createHmac("sha256", process.env.GL_WEBHOOK_SECRET!)
    .update(ts + "." + body.toString("utf-8"))
    .digest("hex");

  const ok = sig.length === mac.length
    && timingSafeEqual(Buffer.from(sig, "hex"), Buffer.from(mac, "hex"));
  if (!ok) return res.status(401).end();

  // Reject deliveries older than 5 minutes to defeat replay.
  if (Date.now() - Date.parse(ts) > 5 * 60 * 1000) return res.status(401).end();

  const event = JSON.parse(body.toString("utf-8"));
  // ... handle event.
  res.status(200).end();
});
```

### Python (FastAPI)

```python
import hmac, hashlib, os, time
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()
SECRET = os.environ["GL_WEBHOOK_SECRET"].encode("utf-8")

@app.post("/greenlang")
async def hook(req: Request):
    ts = req.headers.get("X-GreenLang-Timestamp", "")
    sig = req.headers.get("X-GreenLang-Signature", "").removeprefix("sha256=")
    body = await req.body()

    expected = hmac.new(SECRET, (ts + "." + body.decode()).encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        raise HTTPException(401, "bad signature")

    # Reject deliveries > 5 minutes old.
    try:
        from datetime import datetime, timezone
        age = time.time() - datetime.fromisoformat(ts.replace("Z","+00:00")).timestamp()
        if age > 300: raise HTTPException(401, "stale")
    except ValueError:
        raise HTTPException(401, "bad timestamp")

    # ... process body as JSON.
    return {"ok": True}
```

---

## Delivery retries

When your endpoint returns a non-2xx response or times out, GreenLang retries with exponential backoff:

| Attempt | Delay from the previous attempt |
|---|---|
| 1 | immediate (first try) |
| 2 | 30s |
| 3 | 2m |
| 4 | 10m |
| 5 | 1h |
| 6 | 6h |
| 7 | 24h |

Total retry window: 24 hours from the first attempt. After attempt 7 fails, the delivery is marked `failed` and the event is dropped (you can always re-derive state from `GET /api/v1/editions` + `GET /api/v1/factors`).

Timeout budget per attempt: **10 seconds**. If your handler cannot respond that fast, queue the work internally and return 200 immediately.

### Delivery inspection

```bash
# List deliveries for a subscription.
curl -sS -H "Authorization: Bearer $GL_API_KEY" \
  "$GL_API_BASE/api/v1/webhooks/subscriptions/wh_sub_01HYP.../deliveries"

# Inspect a single delivery.
curl -sS -H "Authorization: Bearer $GL_API_KEY" \
  "$GL_API_BASE/api/v1/webhooks/deliveries/dlv_01HYP..."

# Replay a specific delivery.
curl -sS -X POST -H "Authorization: Bearer $GL_API_KEY" \
  "$GL_API_BASE/api/v1/webhooks/deliveries/dlv_01HYP.../replay"
```

---

## Idempotency

Every delivery carries a unique `delivery_id` (`X-GreenLang-Delivery-Id` header and `payload.delivery_id`). Because deliveries can be retried, your handler **must** be idempotent.

Recommended pattern:

1. On receipt, record `delivery_id` in a dedupe table with TTL 48 hours.
2. If already present, return 200 without re-processing.
3. Otherwise, process, then record the delivery_id after success.

```sql
CREATE TABLE webhook_deliveries (
  delivery_id TEXT PRIMARY KEY,
  received_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- INSERT ... ON CONFLICT (delivery_id) DO NOTHING;
-- If rowcount == 0, skip processing.
```

---

## Test deliveries

```bash
curl -sS -X POST -H "Authorization: Bearer $GL_API_KEY" \
  "$GL_API_BASE/api/v1/webhooks/subscriptions/wh_sub_01HYP.../test" \
  -d '{"event_type":"edition.published"}'
```

The server will send a synthetic event with `payload.test = true`. Useful for smoke-testing your handler after deploy.

---

## Unsubscribing

```bash
curl -sS -X DELETE -H "Authorization: Bearer $GL_API_KEY" \
  "$GL_API_BASE/api/v1/webhooks/subscriptions/wh_sub_01HYP..."
```

204 No Content on success. In-flight retries for that subscription stop on the next retry tick.

---

## See also

- [Signed receipts](../concepts/signed-receipts.md) — the same HMAC primitives power webhook signatures.
- [Version pinning](../concepts/version-pinning.md) — `edition.published` and `factor.updated` events.
- [Licensing classes](../concepts/licensing-classes.md) — `license.changed` semantics.

---

## File citations

| Piece | File |
|---|---|
| Event types enumeration | `greenlang/factors/webhooks.py::WebhookEventType` (line 33) |
| Subscription model + HMAC delivery | `greenlang/factors/webhooks.py` |
| Webhook routes | `greenlang/integration/api/routes/` (see `webhooks.py` or marketplace.py) |
| Delivery retry schedule | `greenlang/factors/webhooks.py` (delivery queue + scheduler) |
