# PagerDuty Integration Down

## Alert

**Alert Name:** `PagerDutyIntegrationDown`

**Severity:** Critical

**Threshold:** PagerDuty notification failure rate > 50% for 10 minutes, or all PagerDuty notifications failing

**Duration:** 10 minutes

---

## Description

This alert fires when the GreenLang Alerting Service cannot successfully deliver notifications to PagerDuty via the Events API v2. PagerDuty is the primary paging channel for critical alerts and is responsible for:

1. **Creating incidents** when critical alerts fire
2. **Paging on-call responders** via phone, SMS, push notification, and email
3. **Tracking incident acknowledgment** and escalation
4. **Auto-resolving incidents** when alerts clear
5. **Providing deduplication** via dedup_key to prevent duplicate pages

When the PagerDuty integration is down, critical alerts will NOT trigger pages to on-call responders. The alerting service will attempt to route through secondary channels (Opsgenie, Slack), but the primary paging path is broken.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Critical | On-call responders do not receive phone/SMS pages for critical alerts |
| **Data Impact** | Low | Alert events are queued; PagerDuty incidents will be created on recovery |
| **SLA Impact** | Critical | MTTA for critical alerts will spike; SLA breach risk is high |
| **Revenue Impact** | High | Extended MTTR if responders are unaware of critical incidents |

---

## Symptoms

- `PagerDutyIntegrationDown` alert is firing
- PagerDuty channel in the Alerting Service dashboard shows 0% success rate
- `gl_alert_notifications_total{channel="pagerduty", status="failed"}` incrementing
- `gl_alert_channel_health{channel="pagerduty"}` equals 0
- No new PagerDuty incidents being created despite active critical alerts
- Escalation events increasing (alerts escalating through non-PD channels)
- Error logs showing HTTP 4xx/5xx responses from events.pagerduty.com

---

## Diagnostic Steps

### Step 1: Identify the Error Type

```bash
# Get PagerDuty-specific error logs
kubectl logs -n greenlang-alerting -l app=alerting-service --tail=300 | grep -i "pagerduty"

# Look for HTTP status codes
kubectl logs -n greenlang-alerting -l app=alerting-service --tail=300 | grep -i "pagerduty" | grep -oP "status[_\s]?code[:\s]*\d+"

# Check notification latency (high latency may indicate timeouts)
```

```promql
# PagerDuty notification latency P99
histogram_quantile(0.99, sum(rate(gl_alert_notification_duration_seconds_bucket{channel="pagerduty"}[5m])) by (le))

# PagerDuty failure rate
sum(rate(gl_alert_notifications_total{channel="pagerduty", status="failed"}[5m])) / sum(rate(gl_alert_notifications_total{channel="pagerduty"}[5m]))
```

### Step 2: Check PagerDuty Service Status

Visit https://status.pagerduty.com to check for:
- Events API v2 degradation or outage
- Elevated error rates on the PagerDuty platform
- Scheduled maintenance windows

```bash
# Programmatic check (if available)
curl -s https://status.pagerduty.com/api/v2/status.json | python3 -c "import sys,json; d=json.load(sys.stdin); print('Status:', d.get('status', {}).get('description', 'unknown'))"
```

### Step 3: Validate the Routing Key

```bash
# Get the current routing key from secrets
ROUTING_KEY=$(kubectl get secret alerting-service-secrets -n greenlang-alerting -o jsonpath='{.data.PAGERDUTY_ROUTING_KEY}' | base64 -d)

# Test the routing key with a diagnostic event
curl -s -o /dev/null -w "%{http_code}" \
  -X POST https://events.pagerduty.com/v2/enqueue \
  -H "Content-Type: application/json" \
  -d "{
    \"routing_key\": \"$ROUTING_KEY\",
    \"event_action\": \"trigger\",
    \"dedup_key\": \"runbook-test-$(date +%s)\",
    \"payload\": {
      \"summary\": \"[TEST] Runbook connectivity test - please resolve\",
      \"source\": \"runbook-diagnostic\",
      \"severity\": \"info\",
      \"component\": \"alerting-service\",
      \"group\": \"diagnostic\"
    }
  }"
# Expected: 202 (Accepted)
# 400 = Invalid payload
# 401/403 = Invalid routing key
# 429 = Rate limited
# 5xx = PagerDuty server error
```

### Step 4: Check Network Connectivity

```bash
# Test HTTPS connectivity to PagerDuty from alerting pod
kubectl exec -n greenlang-alerting <pod-name> -- \
  curl -v --connect-timeout 5 --max-time 10 \
  https://events.pagerduty.com/v2/enqueue

# Test DNS resolution
kubectl exec -n greenlang-alerting <pod-name> -- nslookup events.pagerduty.com

# Check TLS certificate validity
kubectl exec -n greenlang-alerting <pod-name> -- \
  openssl s_client -connect events.pagerduty.com:443 -servername events.pagerduty.com </dev/null 2>/dev/null \
  | openssl x509 -noout -dates
```

### Step 5: Check Rate Limiting

PagerDuty Events API v2 has rate limits. Check if the alerting service is being throttled:

```bash
# Look for HTTP 429 responses
kubectl logs -n greenlang-alerting -l app=alerting-service --tail=500 | grep -i "pagerduty" | grep "429"

# Check notification rate
```

```promql
# PagerDuty notification rate (should be well under limits)
sum(rate(gl_alert_notifications_total{channel="pagerduty"}[5m])) * 60
```

---

## Resolution Steps

### Scenario 1: Invalid or Expired Routing Key (HTTP 401/403)

**Symptoms:** Logs show "Invalid Routing Key" or HTTP 401/403 responses

**Cause:** The routing key (integration key) has been rotated in PagerDuty but not updated in the alerting service, or the PagerDuty service/integration was deleted.

**Resolution:**

1. **Verify the PagerDuty service exists:**

Log into PagerDuty UI > Services > Search for "GreenLang Platform"

2. **Get the current integration key:**

PagerDuty UI > Services > GreenLang Platform > Integrations > Events API v2 > Integration Key

3. **Update the SSM parameter:**

```bash
aws ssm put-parameter \
  --name "/gl/prod/alerting/pagerduty-routing-key" \
  --value "<new-routing-key>" \
  --type SecureString \
  --overwrite
```

4. **Force ESO to resync the secret:**

```bash
kubectl annotate externalsecret alerting-service-secrets -n greenlang-alerting \
  force-sync=$(date +%s) --overwrite

# Verify the secret was updated
kubectl get externalsecret alerting-service-secrets -n greenlang-alerting -o jsonpath='{.status.conditions[0]}'
```

5. **Restart the alerting service:**

```bash
kubectl rollout restart deployment/alerting-service -n greenlang-alerting
kubectl rollout status deployment/alerting-service -n greenlang-alerting
```

6. **Verify recovery:**

```bash
# Send a test event
curl -X POST https://events.pagerduty.com/v2/enqueue \
  -H "Content-Type: application/json" \
  -d "{
    \"routing_key\": \"<new-routing-key>\",
    \"event_action\": \"trigger\",
    \"dedup_key\": \"recovery-test-$(date +%s)\",
    \"payload\": {
      \"summary\": \"[TEST] PagerDuty integration recovered\",
      \"source\": \"runbook-recovery\",
      \"severity\": \"info\"
    }
  }"
```

### Scenario 2: PagerDuty Service Outage

**Symptoms:** PagerDuty status page shows degradation, HTTP 5xx responses

**Cause:** PagerDuty platform is experiencing an outage.

**Resolution:**

1. **Check PagerDuty status page** for incident updates and estimated resolution time.

2. **Enable fallback channels:**

The alerting service should automatically fall back to Opsgenie and Slack for critical alerts. Verify fallback is working:

```promql
# Check if Opsgenie is receiving escalated PD failures
sum(rate(gl_alert_notifications_total{channel="opsgenie"}[5m]))

# Check if Slack is receiving critical alerts
sum(rate(gl_alert_notifications_total{channel="slack", severity="critical"}[5m]))
```

3. **If manual intervention needed, use Opsgenie directly:**

```bash
# Create an Opsgenie alert directly
curl -X POST https://api.opsgenie.com/v2/alerts \
  -H "Content-Type: application/json" \
  -H "Authorization: GenieKey <opsgenie-api-key>" \
  -d '{
    "message": "PagerDuty outage - manual alert routing active",
    "priority": "P1",
    "tags": ["pagerduty-outage", "manual-routing"]
  }'
```

4. **Monitor PagerDuty recovery and verify events are processed:**

After PagerDuty recovers, events queued during the outage should be processed automatically. The alerting service uses dedup_key, so duplicate events will be deduplicated by PagerDuty.

### Scenario 3: Rate Limiting (HTTP 429)

**Symptoms:** HTTP 429 responses, high notification volume

**Cause:** Alert storm generating excessive PagerDuty events.

**Resolution:**

1. **Identify the alert storm source:**

```promql
# Top alert names by fire count
topk(10, sum by (alertname) (rate(gl_alert_fired_total[15m])))
```

2. **Silence noisy alerts in Alertmanager:**

```bash
# Create a silence in Alertmanager for the noisy alert
amtool silence add alertname="<noisy-alert>" --duration="2h" \
  --author="runbook" --comment="Silencing during alert storm investigation" \
  --alertmanager.url="http://gl-prometheus-alertmanager.monitoring.svc.cluster.local:9093"
```

3. **The alerting service should automatically backoff and retry.** Monitor:

```promql
sum(rate(gl_alert_rate_limited_total{channel="pagerduty"}[5m]))
```

### Scenario 4: Routing Key Misconfiguration (HTTP 400)

**Symptoms:** HTTP 400 Bad Request responses

**Cause:** Malformed event payload or invalid routing key format.

**Resolution:**

1. **Check the routing key format:**

The routing key should be a 32-character hexadecimal string. Verify there are no whitespace or encoding issues:

```bash
kubectl get secret alerting-service-secrets -n greenlang-alerting \
  -o jsonpath='{.data.PAGERDUTY_ROUTING_KEY}' | base64 -d | xxd | head -5
```

2. **If the key contains unexpected characters (newlines, spaces), re-set it:**

```bash
aws ssm put-parameter \
  --name "/gl/prod/alerting/pagerduty-routing-key" \
  --value "<clean-routing-key>" \
  --type SecureString \
  --overwrite
```

---

## Key Rotation Procedure

PagerDuty routing keys should be rotated every 90 days:

1. **Create a new integration in PagerDuty:**

PagerDuty UI > Services > GreenLang Platform > Integrations > Add Integration > Events API v2

2. **Update SSM with the new key:**

```bash
aws ssm put-parameter \
  --name "/gl/prod/alerting/pagerduty-routing-key" \
  --value "<new-routing-key>" \
  --type SecureString \
  --overwrite
```

3. **Force ESO resync and restart:**

```bash
kubectl annotate externalsecret alerting-service-secrets -n greenlang-alerting \
  force-sync=$(date +%s) --overwrite
kubectl rollout restart deployment/alerting-service -n greenlang-alerting
```

4. **Verify the new key works with a test event.**

5. **Delete the old integration in PagerDuty** after confirming the new one works.

---

## Prevention

### Monitoring

- **Dashboard:** Alerting Service (`/d/alerting-service`)
- **Key panels:** Channel Health Status, Notification Delivery Failures
- **Alert:** `PagerDutyIntegrationDown` (this alert)
- **Key metrics:**
  - `gl_alert_channel_health{channel="pagerduty"}` (should be 1)
  - `gl_alert_notifications_total{channel="pagerduty", status="failed"}` (should be 0)
  - `gl_alert_notification_duration_seconds{channel="pagerduty"}` P99 (should be < 3s)

### Best Practices

1. **Test PagerDuty integration** after every key rotation or Terraform apply
2. **Subscribe to PagerDuty status updates** at https://status.pagerduty.com
3. **Maintain Opsgenie as a secondary** paging channel for PagerDuty failover
4. **Set up PagerDuty maintenance windows** to align with planned change windows
5. **Review PagerDuty event rules** quarterly to ensure routing is correct

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Observability Team
- **Review cadence:** Quarterly or after any PagerDuty integration incident
- **Related alerts:** `AlertingServiceDown`, `NotificationDeliveryFailing`, `OpsgenieIntegrationDown`
- **Related dashboards:** Alerting Service
