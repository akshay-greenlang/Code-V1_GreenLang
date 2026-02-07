# Notification Delivery Failing

## Alert

**Alert Name:** `NotificationDeliveryFailing`

**Severity:** Critical

**Threshold:** Notification failure rate > 10% for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when the overall notification delivery failure rate exceeds 10% across all channels. The Alerting Service delivers notifications through multiple channels:

1. **PagerDuty** - Events API v2 for critical incident paging
2. **Opsgenie** - Alert API v2 for on-call routing and escalation
3. **Slack** - Incoming webhooks for team-channel notifications
4. **Email (SES)** - AWS SES for email notifications
5. **Webhook** - Custom webhook integrations

When notification delivery fails, on-call responders may not receive critical alerts, leading to delayed incident response and extended outages.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Critical | On-call responders may not receive pages for active incidents |
| **Data Impact** | Medium | Failed notifications are logged; alerts are not lost but delivery is delayed |
| **SLA Impact** | Critical | MTTA increases as responders are not notified promptly |
| **Revenue Impact** | High | Extended MTTR due to delayed awareness of production issues |

---

## Symptoms

- `NotificationDeliveryFailing` alert is firing
- Multiple channels showing elevated failure rates on the dashboard
- `gl_alert_notifications_total{status="failed"}` counter incrementing rapidly
- Notification latency P95 increasing (may indicate timeouts)
- Escalation events increasing (alerts being escalated due to non-acknowledgment)
- Slack channels silent despite known active Prometheus alerts

---

## Diagnostic Steps

### Step 1: Identify Failing Channels

```promql
# Failure rate per channel
sum by (channel) (rate(gl_alert_notifications_total{job="alerting-service", status="failed"}[5m]))
/
sum by (channel) (rate(gl_alert_notifications_total{job="alerting-service"}[5m]))

# Absolute failure count per channel
sum by (channel) (increase(gl_alert_notifications_total{job="alerting-service", status="failed"}[15m]))

# Success rate per channel
1 - (
  sum by (channel) (rate(gl_alert_notifications_total{job="alerting-service", status="failed"}[5m]))
  /
  sum by (channel) (rate(gl_alert_notifications_total{job="alerting-service"}[5m]))
)
```

### Step 2: Check Error Messages

```bash
# Get recent error logs from alerting service
kubectl logs -n greenlang-alerting -l app=alerting-service --tail=300 | grep -i "error\|fail\|timeout\|refused"

# Filter for specific channel errors
kubectl logs -n greenlang-alerting -l app=alerting-service --tail=300 | grep -i "pagerduty"
kubectl logs -n greenlang-alerting -l app=alerting-service --tail=300 | grep -i "opsgenie"
kubectl logs -n greenlang-alerting -l app=alerting-service --tail=300 | grep -i "slack"
kubectl logs -n greenlang-alerting -l app=alerting-service --tail=300 | grep -i "ses\|email"
```

### Step 3: Check Network Egress

```bash
# Verify external HTTPS connectivity from alerting pod
kubectl exec -n greenlang-alerting <pod-name> -- curl -s -o /dev/null -w "%{http_code}" https://events.pagerduty.com/v2/enqueue
kubectl exec -n greenlang-alerting <pod-name> -- curl -s -o /dev/null -w "%{http_code}" https://api.opsgenie.com/v2/heartbeats/ping
kubectl exec -n greenlang-alerting <pod-name> -- curl -s -o /dev/null -w "%{http_code}" https://hooks.slack.com

# Check network policies are not blocking egress
kubectl get networkpolicy -n greenlang-alerting -o yaml | grep -A20 "egress"

# Check DNS resolution
kubectl exec -n greenlang-alerting <pod-name> -- nslookup events.pagerduty.com
kubectl exec -n greenlang-alerting <pod-name> -- nslookup api.opsgenie.com
kubectl exec -n greenlang-alerting <pod-name> -- nslookup hooks.slack.com
```

### Step 4: Check API Credentials

```bash
# Verify secrets are present and populated
kubectl get secret alerting-service-secrets -n greenlang-alerting -o jsonpath='{.data}' | python3 -c "import sys,json,base64; d=json.load(sys.stdin); [print(k, ':', 'SET' if base64.b64decode(v).decode() != 'not-configured' else 'NOT SET') for k,v in d.items()]"

# Check ESO sync status for credential freshness
kubectl get externalsecrets -n greenlang-alerting
kubectl describe externalsecret alerting-service-secrets -n greenlang-alerting
```

### Step 5: Check Third-Party Service Status

Verify the external services are operational:

- **PagerDuty:** https://status.pagerduty.com
- **Opsgenie:** https://opsgenie.atlassian.com/service-health
- **Slack:** https://status.slack.com
- **AWS SES:** https://health.aws.amazon.com (check SES in your region)

---

## Resolution Steps

### Scenario 1: PagerDuty API Failures

**Symptoms:** Logs show HTTP 401/403 from events.pagerduty.com, or connection timeouts

**Resolution:**

1. **Test the routing key directly:**

```bash
# Test PagerDuty Events API v2 with a test event
curl -X POST https://events.pagerduty.com/v2/enqueue \
  -H "Content-Type: application/json" \
  -d '{
    "routing_key": "<routing-key>",
    "event_action": "trigger",
    "payload": {
      "summary": "Test notification from runbook",
      "source": "runbook-test",
      "severity": "info"
    }
  }'
```

2. **If 401/403: Rotate the routing key:**

```bash
# Update SSM parameter with new routing key
aws ssm put-parameter \
  --name "/gl/prod/alerting/pagerduty-routing-key" \
  --value "<new-routing-key>" \
  --type SecureString \
  --overwrite

# Force ESO to resync
kubectl annotate externalsecret alerting-service-secrets -n greenlang-alerting \
  force-sync=$(date +%s) --overwrite

# Restart alerting service to pick up new secret
kubectl rollout restart deployment/alerting-service -n greenlang-alerting
```

3. **If timeout: Check PagerDuty status and network:**

```bash
# Test from within the pod
kubectl exec -n greenlang-alerting <pod-name> -- \
  curl -v --connect-timeout 5 https://events.pagerduty.com/v2/enqueue
```

### Scenario 2: Slack Webhook Failures

**Symptoms:** Logs show HTTP 403/404/410 from hooks.slack.com

**Resolution:**

1. **HTTP 403 (Forbidden):** Webhook URL has been revoked or the Slack app was removed. Regenerate the webhook URL in Slack workspace settings.

2. **HTTP 404 (Not Found):** Channel was deleted or webhook URL is malformed.

3. **HTTP 410 (Gone):** Webhook has been permanently deleted.

4. **HTTP 429 (Rate Limited):** Slack limits to 1 message per second per webhook. The alerting service should automatically handle this with backoff.

```bash
# Update the webhook URL in SSM
aws ssm put-parameter \
  --name "/gl/prod/alerting/slack-webhook-critical" \
  --value "<new-webhook-url>" \
  --type SecureString \
  --overwrite

# Repeat for warning and info webhooks if affected
```

### Scenario 3: AWS SES Email Failures

**Symptoms:** Logs show SES errors, bounces, or sending quota exceeded

**Resolution:**

1. **Check SES sending statistics:**

```bash
aws ses get-send-quota --region us-east-1
aws ses get-send-statistics --region us-east-1
```

2. **Check if sender is verified:**

```bash
aws ses get-identity-verification-attributes --identities alerts@greenlang.io --region us-east-1
```

3. **Check bounce/complaint rates (must be < 5% bounce, < 0.1% complaint):**

```bash
aws ses get-account --region us-east-1
```

4. **If in SES sandbox:** SES sandbox mode only allows sending to verified addresses. Request production access if needed.

### Scenario 4: Network Egress Blocked

**Symptoms:** All channels failing simultaneously with connection refused/timeout

**Resolution:**

1. **Check network policies:**

```bash
kubectl get networkpolicy -n greenlang-alerting -o yaml
```

2. **Verify egress to HTTPS (port 443) is allowed for external IPs:**

The `alerting-service-egress` network policy must allow egress to `0.0.0.0/0:443` (excluding private CIDR ranges).

3. **Check security group rules (if using AWS VPC CNI):**

```bash
# Get the node security group
NODE_SG=$(aws eks describe-cluster --name gl-eks-prod --query "cluster.resourcesVpcConfig.clusterSecurityGroupId" --output text)

# Check egress rules
aws ec2 describe-security-groups --group-ids $NODE_SG --query "SecurityGroups[0].IpPermissionsEgress"
```

4. **Check NAT Gateway status (pods need NAT for external access):**

```bash
aws ec2 describe-nat-gateways --filter "Name=state,Values=available"
```

### Scenario 5: Certificate Expiry

**Symptoms:** TLS handshake failures in logs

**Resolution:**

1. **Check system CA bundle is up to date:**

```bash
kubectl exec -n greenlang-alerting <pod-name> -- openssl s_client -connect events.pagerduty.com:443 -servername events.pagerduty.com </dev/null 2>/dev/null | openssl x509 -noout -dates
```

2. **If using custom CA bundle, verify it is mounted correctly:**

```bash
kubectl exec -n greenlang-alerting <pod-name> -- ls -la /etc/ssl/certs/
```

---

## Prevention

### Monitoring

- **Dashboard:** Alerting Service (`/d/alerting-service`)
- **Key panels:** Notification Success Rate by Channel, Notification Delivery Failures
- **Key metrics:**
  - `gl_alert_notifications_total{status="failed"}` (should be near 0)
  - `gl_alert_notification_duration_seconds` P99 (should be < 5s)
  - `gl_alert_channel_health` (should be 1 for all channels)

### API Key Rotation Schedule

| Channel | Rotation Cadence | Method |
|---------|-----------------|--------|
| PagerDuty | 90 days | Terraform apply + ESO sync |
| Opsgenie | 90 days | Terraform apply + ESO sync |
| Slack | On revocation | Manual + SSM update |
| SES | N/A | IAM role-based, no key rotation needed |

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Observability Team
- **Review cadence:** Quarterly or after any notification delivery incident
- **Related alerts:** `PagerDutyIntegrationDown`, `OpsgenieIntegrationDown`, `SlackIntegrationDown`
- **Related dashboards:** Alerting Service, Alertmanager Health
