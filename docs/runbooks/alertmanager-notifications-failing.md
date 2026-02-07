# Alertmanager Notifications Failing

## Alert

**Alert Name:** `AlertmanagerNotificationsFailing`

**Severity:** Critical

**Threshold:** `rate(alertmanager_notifications_failed_total[5m]) > 0`

**Duration:** 5 minutes

---

## Description

This alert fires when Alertmanager is unable to deliver notifications to configured receivers (Slack, PagerDuty, email, webhooks). This is critical because:

1. **Alert blindness** - On-call engineers not notified of incidents
2. **SLA violations** - No awareness of outages
3. **Compliance risk** - Security alerts not delivered
4. **Extended MTTR** - Delayed incident response

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Critical | No visibility into production issues |
| **Data Impact** | Low | Alerts are queued, not lost |
| **SLA Impact** | Critical | May miss SLA violations |
| **Revenue Impact** | High | Undetected outages impact customers |

---

## Symptoms

- `alertmanager_notifications_failed_total` increasing
- No Slack messages in alert channels
- No PagerDuty pages despite active alerts
- Alerts visible in Alertmanager UI but not delivered
- Email bounces or delays

---

## Diagnostic Steps

### Step 1: Check Notification Metrics

```promql
# Failed notifications by receiver
rate(alertmanager_notifications_failed_total[5m]) by (integration)

# Successful notifications
rate(alertmanager_notifications_total[5m]) by (integration)

# Notification latency
alertmanager_notification_latency_seconds

# Pending alerts
alertmanager_alerts{state="pending"}
alertmanager_alerts{state="active"}
```

### Step 2: Check Alertmanager Logs

```bash
# Get Alertmanager pods
kubectl get pods -n monitoring -l app.kubernetes.io/name=alertmanager

# Check logs
kubectl logs -n monitoring -l app.kubernetes.io/name=alertmanager --tail=200

# Look for notification errors
kubectl logs -n monitoring -l app.kubernetes.io/name=alertmanager | grep -i "notify\|error\|fail"
```

### Step 3: Check Alertmanager Configuration

```bash
# Get current config
kubectl exec -n monitoring alertmanager-0 -- cat /etc/alertmanager/config/alertmanager.yml

# Check config secret
kubectl get secret -n monitoring alertmanager-config -o jsonpath='{.data.alertmanager\.yaml}' | base64 -d
```

### Step 4: Test Notification Endpoints

```bash
# Test Slack webhook manually
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"Test alert from Alertmanager"}' \
  https://hooks.slack.com/services/T00/B00/XXX

# Test PagerDuty
curl -X POST https://events.pagerduty.com/v2/enqueue \
  -H 'Content-Type: application/json' \
  -d '{
    "routing_key": "YOUR_ROUTING_KEY",
    "event_action": "trigger",
    "payload": {
      "summary": "Test alert",
      "severity": "info",
      "source": "alertmanager-test"
    }
  }'
```

### Step 5: Check Network Connectivity

```bash
# Exec into Alertmanager pod
kubectl exec -n monitoring -it alertmanager-0 -- /bin/sh

# Test Slack connectivity
wget -O- https://hooks.slack.com 2>&1 | head -5

# Test PagerDuty connectivity
wget -O- https://events.pagerduty.com 2>&1 | head -5
```

### Step 6: Check Secrets Configuration

```bash
# List alertmanager secrets
kubectl get secrets -n monitoring | grep alertmanager

# Check secret contents (base64 encoded)
kubectl get secret -n monitoring alertmanager-secrets -o yaml
```

---

## Resolution Steps

### Scenario 1: Slack Webhook Failures

**Symptoms:** Logs show "webhook returned status 404" or "invalid_payload"

**Resolution:**

1. **Verify Slack webhook URL is valid:**

```bash
# Test webhook
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"Test message"}' \
  'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
```

2. **Regenerate webhook if needed:**
   - Go to Slack App settings
   - Create new Incoming Webhook
   - Update secret in Kubernetes

3. **Update Alertmanager secret:**

```bash
# Create updated secret
kubectl create secret generic alertmanager-slack-webhook \
  -n monitoring \
  --from-literal=webhook-url='https://hooks.slack.com/services/NEW/WEBHOOK/URL' \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart Alertmanager
kubectl rollout restart statefulset -n monitoring alertmanager
```

4. **Check Slack channel still exists:**
   - Archived channels cannot receive webhooks
   - Verify channel name matches configuration

### Scenario 2: PagerDuty Integration Key Invalid

**Symptoms:** Logs show "Invalid Routing Key" or "401 Unauthorized"

**Resolution:**

1. **Verify integration key:**

```bash
# Test integration key
curl -X POST https://events.pagerduty.com/v2/enqueue \
  -H 'Content-Type: application/json' \
  -d '{
    "routing_key": "YOUR_KEY",
    "event_action": "trigger",
    "payload": {
      "summary": "Test",
      "severity": "info",
      "source": "test"
    }
  }'
```

2. **Get new integration key from PagerDuty:**
   - Go to PagerDuty > Services > Your Service > Integrations
   - Create new Events API v2 integration
   - Copy the integration key

3. **Update Kubernetes secret:**

```bash
kubectl create secret generic alertmanager-pagerduty \
  -n monitoring \
  --from-literal=service-key='NEW_INTEGRATION_KEY' \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl rollout restart statefulset -n monitoring alertmanager
```

### Scenario 3: Network/Firewall Blocking

**Symptoms:** Connection timeouts, "no route to host"

**Resolution:**

1. **Check egress network policy:**

```bash
kubectl get networkpolicy -n monitoring -o yaml | grep -A 20 egress
```

2. **Create egress policy for Alertmanager:**

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: alertmanager-egress
  namespace: monitoring
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: alertmanager
  policyTypes:
    - Egress
  egress:
    # Allow DNS
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
    # Allow Slack
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - protocol: TCP
          port: 443
```

3. **Check NAT Gateway for outbound internet:**

```bash
# Verify pods can reach external endpoints
kubectl exec -n monitoring alertmanager-0 -- wget -O- https://slack.com 2>&1 | head -5
```

### Scenario 4: TLS Certificate Issues

**Symptoms:** "certificate verify failed" or "x509" errors

**Resolution:**

1. **Update CA certificates:**

```yaml
# In Alertmanager config
global:
  http_config:
    tls_config:
      insecure_skip_verify: false  # Should be false in production
      ca_file: /etc/alertmanager/certs/ca.crt
```

2. **Mount custom CA certificates:**

```yaml
alertmanager:
  alertmanagerSpec:
    volumes:
      - name: custom-ca
        secret:
          secretName: custom-ca-cert
    volumeMounts:
      - name: custom-ca
        mountPath: /etc/alertmanager/certs
```

### Scenario 5: Rate Limiting

**Symptoms:** "429 Too Many Requests" or "rate limited"

**Resolution:**

1. **Check Alertmanager grouping settings:**

```yaml
# In alertmanager.yml
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 30s       # Wait before sending first notification
  group_interval: 5m    # Wait before sending notification about new alerts in group
  repeat_interval: 4h   # Wait before resending
```

2. **Implement inhibition rules:**

```yaml
inhibit_rules:
  # Don't alert for service issues if the whole cluster is down
  - source_match:
      severity: critical
      alertname: ClusterDown
    target_match:
      severity: warning
    equal: ['cluster']
```

3. **Check Slack rate limits:**
   - Slack allows ~1 message per second per webhook
   - Group alerts more aggressively

### Scenario 6: Alertmanager HA Cluster Issues

**Symptoms:** Some notifications sent, others lost

**Resolution:**

1. **Check Alertmanager mesh status:**

```promql
# Cluster members
alertmanager_cluster_members

# Cluster health
alertmanager_cluster_health_score
```

2. **Verify mesh configuration:**

```bash
kubectl exec -n monitoring alertmanager-0 -- alertmanager --cluster.peer=alertmanager-1:9094
```

3. **Check cluster gossip ports:**

```yaml
alertmanager:
  alertmanagerSpec:
    listenLocal: false
    clusterAdvertiseAddress: ""  # Let it auto-discover
```

---

## Emergency Actions

### If No Notifications Are Being Sent

1. **Set up temporary notification bypass:**

```bash
# Forward Alertmanager UI locally
kubectl port-forward -n monitoring svc/alertmanager 9093:9093

# Manually review alerts at http://localhost:9093
```

2. **Send manual notification for critical alerts:**

```bash
# Manual Slack notification
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"MANUAL ALERT: Alertmanager notifications failing. Active critical alerts: X"}' \
  'https://hooks.slack.com/services/BACKUP/WEBHOOK/URL'
```

3. **Set up CloudWatch alarm as backup:**

```bash
# Create backup alarm in CloudWatch to alert on Alertmanager issues
aws cloudwatch put-metric-alarm \
  --alarm-name "AlertmanagerDown" \
  --metric-name "AlertmanagerNotificationsFailed" \
  --namespace "GreenLang" \
  --threshold 1 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1 \
  --period 300 \
  --alarm-actions "arn:aws:sns:us-east-1:ACCOUNT:backup-alerts"
```

---

## Testing Alertmanager

### Test Alert Generation

```bash
# Send test alert via amtool
kubectl exec -n monitoring alertmanager-0 -- amtool alert add \
  alertname="TestAlert" \
  severity="warning" \
  service="test" \
  --annotation=summary="Test alert from amtool"

# Or via API
curl -X POST http://localhost:9093/api/v1/alerts \
  -H 'Content-Type: application/json' \
  -d '[{
    "labels": {
      "alertname": "TestAlert",
      "severity": "warning"
    },
    "annotations": {
      "summary": "Test alert"
    }
  }]'
```

### Verify Configuration

```bash
# Check config validity
kubectl exec -n monitoring alertmanager-0 -- amtool check-config /etc/alertmanager/config/alertmanager.yml

# Show current routing
kubectl exec -n monitoring alertmanager-0 -- amtool config routes show
```

---

## Escalation Path

| Level | Condition | Contact |
|-------|-----------|---------|
| L1 | Intermittent notification failures | On-call engineer |
| L2 | All notifications failing >15min | Platform team lead + backup notification |
| L3 | Prolonged outage, manual alerting needed | Platform team + all team leads |

---

## Prevention

1. **Monitor notification health:**

```promql
# Alert on notification failures
rate(alertmanager_notifications_failed_total[5m]) > 0
```

2. **Regular webhook validation:**
   - Test webhooks monthly
   - Rotate webhook URLs periodically

3. **Backup notification channels:**
   - Configure multiple receivers for critical alerts
   - Use email as fallback for Slack

4. **Alertmanager HA:**
   - Run 3 replicas for high availability
   - Test failover regularly

---

## Related Dashboards

| Dashboard | URL |
|-----------|-----|
| Alertmanager Overview | https://grafana.greenlang.io/d/alertmanager |
| Notification Health | https://grafana.greenlang.io/d/notification-health |

---

## Related Alerts

- `AlertmanagerConfigInconsistent`
- `AlertmanagerClusterDown`
- `AlertmanagerClusterCrashlooping`
- `AlertmanagerMembersInconsistent`

---

## References

- [Alertmanager Configuration](https://prometheus.io/docs/alerting/latest/configuration/)
- [Slack Incoming Webhooks](https://api.slack.com/messaging/webhooks)
- [PagerDuty Events API v2](https://developer.pagerduty.com/docs/events-api-v2/overview/)
