# GreenLang Kubernetes Manifests - Template Variables

This document lists all template variables in the Kubernetes manifests that must be replaced
before applying to a cluster.

## How to Replace Variables

### Option 1: Using envsubst (recommended)
```bash
export AWS_ACM_CERTIFICATE_ARN="arn:aws:acm:us-east-1:123456789012:certificate/abc12345"
export AWS_KMS_KEY_ARN="arn:aws:kms:us-east-1:123456789012:key/abc12345"
# ... set all variables

for f in deployment/kubernetes/manifests/*.yaml; do
  envsubst < "$f" > "${f}.rendered"
done
```

### Option 2: Using sed
```bash
sed -i 's/${AWS_ACM_CERTIFICATE_ARN}/arn:aws:acm:..../g' deployment/kubernetes/manifests/*.yaml
```

### Option 3: Using Kustomize overlays
Create environment-specific overlays that patch these values.

---

## Required Variables

### AWS Infrastructure

| Variable | File | Description | Example |
|----------|------|-------------|---------|
| `${AWS_ACM_CERTIFICATE_ARN}` | api-service.yaml | AWS ACM certificate ARN for HTTPS | `arn:aws:acm:us-east-1:123456789012:certificate/abc123` |
| `${AWS_KMS_KEY_ARN}` | pvc.yaml | AWS KMS key ARN for EBS encryption | `arn:aws:kms:us-east-1:123456789012:key/abc123` |
| `${AWS_EFS_FILE_SYSTEM_ID}` | pvc.yaml | AWS EFS file system ID | `fs-0123456789abcdef0` |
| `${AWS_REGION}` | ingress.yaml | AWS region for Route53 | `us-east-1` |
| `${AWS_ROUTE53_HOSTED_ZONE_ID}` | ingress.yaml | Route53 hosted zone ID | `Z0123456789ABCDEFGHIJ` |

### Docker/Container Registry

| Variable | File | Description | Example |
|----------|------|-------------|---------|
| `${DOCKER_CONFIG_JSON}` | configmap.yaml, secrets.yaml | Base64 encoded Docker config | See below |

To generate Docker config JSON:
```bash
kubectl create secret docker-registry greenlang-registry \
  --docker-server=ghcr.io \
  --docker-username=USERNAME \
  --docker-password=TOKEN \
  --dry-run=client -o jsonpath='{.data.\.dockerconfigjson}'
```

### Application Configuration

| Variable | File | Description | Example |
|----------|------|-------------|---------|
| `${CONFIG_CHECKSUM}` | worker-deployment.yaml | SHA256 of worker config | `sha256:abc123...` |

To generate config checksum:
```bash
sha256sum deployment/kubernetes/manifests/configmap.yaml | cut -d' ' -f1
```

---

## Secrets (Managed by External Secrets Operator)

The following secrets are automatically synced from AWS Secrets Manager:

| Secret Name | Keys | AWS Secrets Manager Path |
|-------------|------|--------------------------|
| greenlang-secrets | redis-password, database-url, openai-api-key, anthropic-api-key | greenlang-prod/* |
| greenlang-runner-secrets | runner-token, registry-credentials | greenlang-prod/runner |
| greenlang-jwt-secrets | jwt-signing-key, jwt-public-key | greenlang-prod/auth |
| greenlang-observability-secrets | datadog-api-key, sentry-dsn | greenlang-prod/observability |
| greenlang-s3-secrets | aws-access-key-id, aws-secret-access-key, s3-bucket-name | greenlang-prod/storage |

---

## Pre-deployment Checklist

1. [ ] External Secrets Operator is installed
2. [ ] ClusterSecretStore `aws-secrets-manager` is configured
3. [ ] All AWS Secrets Manager secrets are populated
4. [ ] All template variables are replaced
5. [ ] cert-manager is installed (for TLS certificates)
6. [ ] NGINX Ingress Controller is installed
7. [ ] Prometheus Operator is installed (for ServiceMonitors)
8. [ ] Storage classes (fast-ssd, efs-storage) are available

---

## Deployment Order

Apply manifests in this order:
1. namespace.yaml
2. podsecuritypolicy.yaml
3. rbac.yaml
4. pvc.yaml
5. configmap.yaml
6. secrets.yaml
7. networkpolicy.yaml
8. resource-quota.yaml
9. api-service.yaml
10. redis-sentinel-statefulset.yaml
11. executor-deployment.yaml
12. runner-deployment.yaml
13. worker-deployment.yaml
14. worker-hpa.yaml
15. ingress.yaml
16. servicemonitor.yaml

Or use Kustomize:
```bash
kubectl apply -k deployment/kubernetes/manifests/
```
