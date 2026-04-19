# Certificate Renewal Runbook

**Scenario**: Manage TLS/SSL certificate lifecycle including automatic renewal via cert-manager, manual renewal procedures, certificate validation, and troubleshooting certificate-related issues.

**Severity**: P1 (Certificates expiring < 7 days) / P2 (Routine renewal)

**RTO/RPO**: N/A (Preventive maintenance)

**Owner**: Platform Team / Security Team

## Prerequisites

- kubectl access to EKS cluster
- Access to cert-manager namespace
- DNS management access (Route53)
- Understanding of Let's Encrypt rate limits
- Access to certificate monitoring dashboard

## Detection

### Certificate Expiration Monitoring

1. **Prometheus Alerts**:
   - `CertificateExpiresIn7Days` - Warning
   - `CertificateExpiresIn3Days` - Critical
   - `CertificateRenewalFailed` - Critical

2. **Cert-Manager Status**:
   - Certificate in "Issuing" state for > 1 hour
   - Repeated failed renewal attempts
   - ACME challenge failures

3. **Application Symptoms**:
   - Browser certificate warnings
   - API clients rejecting connections
   - SSL handshake errors in logs

### Check Certificate Status

```bash
# List all certificates
kubectl get certificates -A

# Check specific certificate details
kubectl describe certificate api-vcci-scope3-com -n vcci-scope3

# Check certificate expiration dates
kubectl get certificates -A -o custom-columns=\
NAME:.metadata.name,\
NAMESPACE:.metadata.namespace,\
READY:.status.conditions[0].status,\
EXPIRY:.status.notAfter,\
RENEW:.status.renewalTime

# Check cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager --tail=100
```

**Expected Output**:
```
NAME                    NAMESPACE      READY   EXPIRY                RENEW
api-vcci-scope3-com    vcci-scope3    True    2024-04-15T10:30:00Z  2024-03-16T10:30:00Z
```

## Step-by-Step Procedure

### Part 1: Routine Certificate Monitoring

#### Step 1: Verify Cert-Manager Installation

```bash
# Check cert-manager pods are running
kubectl get pods -n cert-manager

# Expected output: All pods in Running state
# NAME                                      READY   STATUS
# cert-manager-7d9f8b6c5d-abc12            1/1     Running
# cert-manager-cainjector-5f6g7h8i9-def   1/1     Running
# cert-manager-webhook-3c4d5e6f7g-ghi     1/1     Running

# Check cert-manager version
kubectl get deployment cert-manager -n cert-manager -o jsonpath='{.spec.template.spec.containers[0].image}'

# Verify webhook is responding
kubectl get validatingwebhookconfigurations cert-manager-webhook
```

#### Step 2: Review ClusterIssuer Configuration

```bash
# List all ClusterIssuers
kubectl get clusterissuer

# Check Let's Encrypt production issuer
kubectl describe clusterissuer letsencrypt-prod

# Verify ACME account registration
kubectl get clusterissuer letsencrypt-prod -o jsonpath='{.status.acme.uri}'
```

**Sample ClusterIssuer**:
```yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: security@company.com
    privateKeySecretRef:
      name: letsencrypt-prod-account-key
    solvers:
    - http01:
        ingress:
          class: nginx
    - dns01:
        route53:
          region: us-west-2
          hostedZoneID: Z1234567890ABC
```

#### Step 3: Check All Certificates Status

```bash
# List certificates with detailed status
kubectl get certificates -A -o wide

# Check for any failed certificates
kubectl get certificates -A -o json | \
  jq -r '.items[] | select(.status.conditions[0].status != "True") |
  "\(.metadata.namespace)/\(.metadata.name): \(.status.conditions[0].message)"'

# Get certificates expiring soon (within 30 days)
kubectl get certificates -A -o json | \
  jq -r --arg date "$(date -d '+30 days' -u +%Y-%m-%dT%H:%M:%SZ)" \
  '.items[] | select(.status.notAfter < $date) |
  "\(.metadata.namespace)/\(.metadata.name): Expires \(.status.notAfter)"'
```

#### Step 4: Monitor Renewal Process

```bash
# Watch certificate renewal in real-time
kubectl get certificates -n vcci-scope3 -w

# Check CertificateRequest objects
kubectl get certificaterequest -n vcci-scope3

# View Order status (ACME challenges)
kubectl get order -n vcci-scope3

# Check Challenge status
kubectl get challenge -n vcci-scope3
```

### Part 2: Automatic Renewal Verification

#### Step 5: Verify Automatic Renewal Configuration

```bash
# Check certificate renewal settings
kubectl get certificate api-vcci-scope3-com -n vcci-scope3 -o yaml | grep -A 5 "renewBefore"

# Typical renewal window: 30 days before expiry
# Certificate lifetime: 90 days
# Renewal trigger: 60 days after issuance (30 days before expiry)

# Check cert-manager controller logs for renewal activity
kubectl logs -n cert-manager deployment/cert-manager --since=24h | grep -i "renew"
```

**Expected Renewal Behavior**:
- Certificates renewed 30 days before expiration
- Automatic retry on failure with exponential backoff
- DNS/HTTP-01 challenges automatically handled

#### Step 6: Test ACME Challenge Resolution

```bash
# For HTTP-01 challenge
# Verify ingress can serve ACME challenge
curl http://api.vcci-scope3.com/.well-known/acme-challenge/test

# For DNS-01 challenge
# Check Route53 hosted zone
aws route53 list-resource-record-sets \
  --hosted-zone-id Z1234567890ABC \
  --query "ResourceRecordSets[?contains(Name, '_acme-challenge')]"

# Test DNS resolution of challenge
dig _acme-challenge.api.vcci-scope3.com TXT +short
```

### Part 3: Manual Certificate Renewal

#### Step 7: Force Certificate Renewal

**When to Use**:
- Testing renewal process
- Certificate compromised
- Need to update certificate parameters

```bash
# Option A: Delete CertificateRequest to trigger new issuance
kubectl delete certificaterequest -n vcci-scope3 api-vcci-scope3-com-xxxxx

# Option B: Annotate certificate to force renewal
kubectl annotate certificate api-vcci-scope3-com -n vcci-scope3 \
  cert-manager.io/issue-temporary-certificate="true" \
  --overwrite

# Option C: Delete and recreate certificate
kubectl delete certificate api-vcci-scope3-com -n vcci-scope3
# Then reapply the Certificate resource

# Monitor renewal progress
watch -n 5 'kubectl get certificate,certificaterequest,order,challenge -n vcci-scope3'
```

#### Step 8: Handle Renewal with DNS-01 Challenge

```bash
# Create Certificate with DNS-01 solver
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: wildcard-vcci-scope3-com
  namespace: vcci-scope3
spec:
  secretName: wildcard-vcci-scope3-com-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  commonName: "*.vcci-scope3.com"
  dnsNames:
  - "*.vcci-scope3.com"
  - vcci-scope3.com
  duration: 2160h  # 90 days
  renewBefore: 720h  # 30 days
  privateKey:
    algorithm: RSA
    size: 2048
  usages:
    - digital signature
    - key encipherment
EOF

# Monitor DNS challenge creation
kubectl get challenge -n vcci-scope3 -w

# Check if DNS TXT record was created
CHALLENGE_DOMAIN=$(kubectl get challenge -n vcci-scope3 -o jsonpath='{.items[0].spec.dnsName}')
dig _acme-challenge.$CHALLENGE_DOMAIN TXT +short

# Verify challenge validation
kubectl describe challenge -n vcci-scope3
```

#### Step 9: Manual Certificate Upload (For External Certificates)

```bash
# If using certificates from external CA (e.g., DigiCert, Sectigo)

# Create TLS secret manually
kubectl create secret tls custom-certificate-tls \
  --cert=/path/to/certificate.crt \
  --key=/path/to/private.key \
  -n vcci-scope3

# Include intermediate certificates
cat certificate.crt intermediate.crt > fullchain.crt
kubectl create secret tls custom-certificate-tls \
  --cert=fullchain.crt \
  --key=private.key \
  -n vcci-scope3 \
  --dry-run=client -o yaml | kubectl apply -f -

# Verify secret created
kubectl describe secret custom-certificate-tls -n vcci-scope3

# Update Ingress to use the secret
kubectl patch ingress api-ingress -n vcci-scope3 --patch '
spec:
  tls:
  - hosts:
    - api.vcci-scope3.com
    secretName: custom-certificate-tls
'
```

### Part 4: Certificate Validation

#### Step 10: Validate Certificate Installation

```bash
# Get the certificate from secret
kubectl get secret api-vcci-scope3-com-tls -n vcci-scope3 -o jsonpath='{.data.tls\.crt}' | base64 -d > /tmp/cert.crt

# Check certificate details
openssl x509 -in /tmp/cert.crt -text -noout

# Verify certificate chain
openssl x509 -in /tmp/cert.crt -text -noout | grep -A 2 "Issuer"

# Check certificate expiration
openssl x509 -in /tmp/cert.crt -noout -enddate

# Verify SAN (Subject Alternative Names)
openssl x509 -in /tmp/cert.crt -noout -ext subjectAltName
```

**Expected Output**:
```
X509v3 Subject Alternative Name:
    DNS:api.vcci-scope3.com, DNS:www.api.vcci-scope3.com
```

#### Step 11: Test Certificate from External Client

```bash
# Test HTTPS endpoint
echo | openssl s_client -connect api.vcci-scope3.com:443 -servername api.vcci-scope3.com 2>/dev/null | openssl x509 -noout -dates

# Check certificate chain validity
echo | openssl s_client -connect api.vcci-scope3.com:443 -servername api.vcci-scope3.com -showcerts 2>/dev/null | grep -E "s:|i:"

# Test with curl
curl -vI https://api.vcci-scope3.com 2>&1 | grep -E "subject:|issuer:|expire"

# Verify SSL Labs grade (external validation)
# Visit: https://www.ssllabs.com/ssltest/analyze.html?d=api.vcci-scope3.com
```

#### Step 12: Validate Ingress Configuration

```bash
# Check Ingress TLS configuration
kubectl get ingress api-ingress -n vcci-scope3 -o yaml | grep -A 10 tls

# Verify NGINX ingress is serving correct certificate
kubectl exec -n ingress-nginx deployment/ingress-nginx-controller -- \
  cat /etc/nginx/nginx.conf | grep ssl_certificate

# Check NGINX logs for SSL errors
kubectl logs -n ingress-nginx -l app.kubernetes.io/component=controller --tail=100 | grep -i ssl
```

### Part 5: Troubleshooting Certificate Issues

#### Step 13: Debug Failed Certificate Issuance

```bash
# Check certificate status and error messages
kubectl describe certificate api-vcci-scope3-com -n vcci-scope3 | tail -20

# Check CertificateRequest for failure reasons
kubectl describe certificaterequest -n vcci-scope3 | grep -A 5 "Status\|Message"

# Check Order status
kubectl describe order -n vcci-scope3

# Check Challenge failures
kubectl describe challenge -n vcci-scope3 | grep -A 10 "Status\|Reason"

# Review cert-manager logs for errors
kubectl logs -n cert-manager deployment/cert-manager --tail=200 | grep -i error
```

**Common Error Messages**:
- "dns: lookup failed" - DNS propagation issue
- "http-01 challenge failed" - Ingress not accessible
- "rate limit exceeded" - Let's Encrypt rate limits hit
- "too many certificates already issued" - Duplicate requests

#### Step 14: Fix DNS-01 Challenge Issues

```bash
# Verify Route53 permissions
aws sts get-caller-identity

# Check IAM role attached to cert-manager
kubectl get serviceaccount cert-manager -n cert-manager -o yaml | grep -A 5 annotations

# Test Route53 access
aws route53 list-hosted-zones

# Check if TXT record was created
aws route53 list-resource-record-sets \
  --hosted-zone-id Z1234567890ABC \
  --query "ResourceRecordSets[?Type=='TXT' && contains(Name, '_acme-challenge')]"

# If missing, check cert-manager has correct IAM permissions
kubectl describe clusterissuer letsencrypt-prod | grep -A 10 "dns01"

# Manually create DNS record for testing
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890ABC \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "_acme-challenge.api.vcci-scope3.com",
        "Type": "TXT",
        "TTL": 300,
        "ResourceRecords": [{"Value": "\"test-challenge-value\""}]
      }
    }]
  }'
```

#### Step 15: Fix HTTP-01 Challenge Issues

```bash
# Verify Ingress is accessible
curl -I http://api.vcci-scope3.com/.well-known/acme-challenge/test

# Check Ingress controller logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/component=controller --tail=50

# Verify cert-manager solver ingress
kubectl get ingress -n vcci-scope3 | grep cm-acme-http-solver

# Check if solver pod is running
kubectl get pods -n vcci-scope3 | grep cm-acme-http-solver

# Test challenge endpoint accessibility
kubectl run -it --rm curl-test \
  --image=curlimages/curl \
  --restart=Never \
  -- curl -v http://api.vcci-scope3.com/.well-known/acme-challenge/test

# Check for firewall/security group blocking
# Ensure port 80 is accessible from Let's Encrypt servers
```

### Part 6: Certificate Rotation Best Practices

#### Step 16: Implement Certificate Rotation Strategy

```bash
# Configure shorter certificate lifetimes for increased rotation
kubectl patch certificate api-vcci-scope3-com -n vcci-scope3 --type=merge --patch '
spec:
  duration: 1440h  # 60 days instead of 90
  renewBefore: 720h  # 30 days before expiry
'

# Enable email notifications for expiration
kubectl patch clusterissuer letsencrypt-prod --type=merge --patch '
spec:
  acme:
    email: security-alerts@company.com
'

# Set up Prometheus alerts
cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: certificate-expiry-alerts
  namespace: monitoring
spec:
  groups:
  - name: certificates
    interval: 1m
    rules:
    - alert: CertificateExpiresIn7Days
      expr: certmanager_certificate_expiration_timestamp_seconds - time() < 7 * 24 * 3600
      labels:
        severity: warning
      annotations:
        summary: "Certificate {{ \$labels.name }} expiring soon"
        description: "Certificate {{ \$labels.name }} in {{ \$labels.namespace }} expires in less than 7 days"
    - alert: CertificateExpiresIn3Days
      expr: certmanager_certificate_expiration_timestamp_seconds - time() < 3 * 24 * 3600
      labels:
        severity: critical
      annotations:
        summary: "Certificate {{ \$labels.name }} expiring very soon"
        description: "Certificate {{ \$labels.name }} in {{ \$labels.namespace }} expires in less than 3 days"
EOF
```

#### Step 17: Backup Certificate Secrets

```bash
# Export all certificate secrets
mkdir -p /backup/certificates/$(date +%Y%m%d)

kubectl get secrets -n vcci-scope3 -l certmanager.k8s.io/certificate-name -o yaml > \
  /backup/certificates/$(date +%Y%m%d)/vcci-scope3-certificates.yaml

# Encrypt backup
gpg --encrypt --recipient security@company.com \
  /backup/certificates/$(date +%Y%m%d)/vcci-scope3-certificates.yaml

# Upload to secure S3 bucket
aws s3 cp /backup/certificates/$(date +%Y%m%d)/vcci-scope3-certificates.yaml.gpg \
  s3://vcci-secure-backups/certificates/$(date +%Y%m%d)/ \
  --server-side-encryption AES256

# Verify backup
aws s3 ls s3://vcci-secure-backups/certificates/$(date +%Y%m%d)/
```

#### Step 18: Update Certificate Documentation

```bash
# Generate certificate inventory
cat > /tmp/certificate_inventory.md << 'EOF'
# VCCI Scope 3 Platform - Certificate Inventory

**Generated**: $(date)

## Production Certificates

| Domain | Namespace | Issuer | Expiry | Auto-Renew | Status |
|--------|-----------|--------|--------|------------|--------|
EOF

kubectl get certificates -A -o json | jq -r '
  .items[] |
  "| \(.spec.dnsNames[0]) | \(.metadata.namespace) | \(.spec.issuerRef.name) | \(.status.notAfter) | Yes | \(.status.conditions[0].status) |"
' >> /tmp/certificate_inventory.md

cat /tmp/certificate_inventory.md
```

## Validation

### Post-Renewal Validation Checklist

```bash
# 1. Certificate is ready
kubectl get certificate api-vcci-scope3-com -n vcci-scope3 -o jsonpath='{.status.conditions[0].status}'
# Expected: "True"

# 2. Certificate not expired
kubectl get certificate api-vcci-scope3-com -n vcci-scope3 -o jsonpath='{.status.notAfter}'
# Expected: Date > 30 days in future

# 3. Secret exists and populated
kubectl get secret api-vcci-scope3-com-tls -n vcci-scope3 -o jsonpath='{.data.tls\.crt}' | base64 -d | openssl x509 -noout -dates

# 4. Ingress using correct secret
kubectl get ingress api-ingress -n vcci-scope3 -o jsonpath='{.spec.tls[0].secretName}'

# 5. HTTPS endpoint accessible
curl -I https://api.vcci-scope3.com

# 6. Certificate chain valid
echo | openssl s_client -connect api.vcci-scope3.com:443 -servername api.vcci-scope3.com 2>/dev/null | openssl x509 -noout -issuer

# 7. No certificate warnings in browser
# Manual test: Visit https://api.vcci-scope3.com in browser

# 8. Prometheus metrics showing valid certificate
curl -s http://prometheus:9090/api/v1/query?query=certmanager_certificate_ready_status | jq
```

## Troubleshooting

### Issue 1: Rate Limit Exceeded

**Symptoms**: "too many certificates already issued for exact set of domains"

**Cause**: Let's Encrypt rate limits
- 50 certificates per registered domain per week
- 5 duplicate certificates per week

**Resolution**:
```bash
# Check rate limit status
# Visit: https://crt.sh/?q=vcci-scope3.com

# Use staging environment for testing
kubectl patch clusterissuer letsencrypt-staging --type=merge --patch '
spec:
  acme:
    server: https://acme-staging-v02.api.letsencrypt.org/directory
'

# Update certificate to use staging
kubectl patch certificate api-vcci-scope3-com -n vcci-scope3 --type=merge --patch '
spec:
  issuerRef:
    name: letsencrypt-staging
'

# Wait 1 week for rate limit reset, then switch back to production
```

### Issue 2: Cert-Manager Webhook Not Responding

**Symptoms**: "Internal error occurred: failed calling webhook"

**Resolution**:
```bash
# Check webhook pod
kubectl get pods -n cert-manager -l app.kubernetes.io/component=webhook

# Check webhook service
kubectl get svc -n cert-manager cert-manager-webhook

# Test webhook endpoint
kubectl run -it --rm curl-test \
  --image=curlimages/curl \
  --restart=Never \
  -- curl -v https://cert-manager-webhook.cert-manager.svc:443/healthz

# Delete webhook to force recreation
kubectl delete pod -n cert-manager -l app.kubernetes.io/component=webhook

# If still failing, reinstall cert-manager
helm upgrade cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --set installCRDs=true
```

### Issue 3: Private Key Mismatch

**Symptoms**: "tls: private key does not match public key"

**Resolution**:
```bash
# Verify private key and certificate match
kubectl get secret api-vcci-scope3-com-tls -n vcci-scope3 -o jsonpath='{.data.tls\.key}' | base64 -d > /tmp/private.key
kubectl get secret api-vcci-scope3-com-tls -n vcci-scope3 -o jsonpath='{.data.tls\.crt}' | base64 -d > /tmp/cert.crt

# Extract public key from private key
openssl rsa -in /tmp/private.key -pubout -out /tmp/public_from_private.key 2>/dev/null

# Extract public key from certificate
openssl x509 -in /tmp/cert.crt -pubkey -noout > /tmp/public_from_cert.key

# Compare
diff /tmp/public_from_private.key /tmp/public_from_cert.key

# If different, delete and recreate certificate
kubectl delete secret api-vcci-scope3-com-tls -n vcci-scope3
kubectl delete certificate api-vcci-scope3-com -n vcci-scope3
kubectl apply -f certificate.yaml
```

### Issue 4: DNS Propagation Delays

**Symptoms**: DNS-01 challenge timing out

**Resolution**:
```bash
# Check DNS propagation globally
dig @8.8.8.8 _acme-challenge.api.vcci-scope3.com TXT +short
dig @1.1.1.1 _acme-challenge.api.vcci-scope3.com TXT +short

# Use online tools: https://dnschecker.org/

# Increase DNS-01 challenge timeout
kubectl patch clusterissuer letsencrypt-prod --type=merge --patch '
spec:
  acme:
    solvers:
    - dns01:
        route53:
          region: us-west-2
        acmeChallengeSolverDNS01:
          timeout: 180s  # Increase from default 60s
'

# Manually wait for propagation before retrying
sleep 300
kubectl delete challenge -n vcci-scope3 --all
```

## Related Documentation

- [Incident Response Runbook](./INCIDENT_RESPONSE.md)
- [Security Incident Runbook](./SECURITY_INCIDENT.md)
- [Deployment Rollback Runbook](./DEPLOYMENT_ROLLBACK.md)
- [cert-manager Documentation](https://cert-manager.io/docs/)
- [Let's Encrypt Rate Limits](https://letsencrypt.org/docs/rate-limits/)
- [AWS Route53 DNS Validation](https://docs.aws.amazon.com/acm/latest/userguide/dns-validation.html)

## Appendix: Certificate Monitoring Dashboard

### Grafana Dashboard Queries

```promql
# Certificate expiration time
certmanager_certificate_expiration_timestamp_seconds

# Time until expiration
certmanager_certificate_expiration_timestamp_seconds - time()

# Renewal success rate
rate(certmanager_certificate_renewal_total{status="success"}[1h])

# Failed renewals
rate(certmanager_certificate_renewal_total{status="failed"}[1h])
```

## Contact Information

- **Security Team**: security@company.com
- **Platform Team**: platform-team@company.com
- **On-Call Engineer**: PagerDuty escalation
- **Let's Encrypt Support**: https://community.letsencrypt.org/
