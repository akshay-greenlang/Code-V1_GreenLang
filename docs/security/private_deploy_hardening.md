# Factors API — Private / VPC-Isolated Deployment Hardening

> Target: Enterprise customers who contract for a dedicated deployment in
> their own AWS account or in a GreenLang-managed but isolated VPC.
> Everything in this document is **stricter** than the default staging /
> production profile; apply it on top of the standard kustomize base.

## 1. Network topology

### 1.1 VPC

- Dedicated VPC, CIDR `10.42.0.0/16` (customer-customisable).
- Three AZs minimum.
- No default route to the internet through an IGW.

### 1.2 Subnets

| Purpose | CIDR | Exposure |
|---------|------|----------|
| Kubernetes nodes | `10.42.0.0/20` | Private |
| RDS (Postgres + pgvector) | `10.42.16.0/24` | Private |
| ElastiCache (Redis) | `10.42.17.0/24` | Private |
| NAT egress (optional) | `10.42.250.0/28` | Public via NAT only |
| Vault | `10.42.18.0/24` | Private |

### 1.3 Egress

- No public ALB / NLB.
- Customer traffic arrives exclusively via **AWS PrivateLink** to the
  cluster's internal NLB (VPC Endpoint Service).
- If outbound internet access is required (e.g., billing webhooks),
  traffic exits through a single NAT Gateway in the egress subnet and
  egress rules whitelist specific destinations.

## 2. Kubernetes policy

### 2.1 NetworkPolicy

- Default-deny ingress in the Factors namespace.
- Allow ingress only from the PrivateLink load balancer SG.
- Allow egress to: Postgres, Redis, Vault, optional egress NAT.
- Block all Pod -> Pod traffic outside the namespace.

### 2.2 PodSecurity

- `PodSecurity` standard set to `restricted` on the namespace.
- No privileged pods; `readOnlyRootFilesystem: true` on the API deployment.
- `runAsNonRoot: true`, UID 10001.

### 2.3 Secrets

- **Customer-managed KMS keys** encrypt Vault storage and RDS.
- Per-tenant KMS key alias: `alias/factors-{tenant}-{env}`.
- Automatic rotation every 90 days.
- External Secrets Operator pinned to the customer's Vault URL.

## 3. Image provenance

- Only images signed with the GreenLang release key are admitted.
- Cosign verification enforced by a Kyverno policy.
- Private registry mirror in the customer VPC (ECR).

## 4. Logging + monitoring

- Loki + Prometheus instances scoped to the private VPC.
- Logs never leave the VPC (no SaaS log sink).
- Optional: forward scrubbed metrics to GreenLang's operational
  dashboard via a customer-approved metric-only relay.

## 5. Operator access

- Break-glass access via SSM Session Manager + MFA.
- All break-glass sessions recorded and forwarded to the customer's SIEM.
- Per-session approval workflow in ChatOps (Slack + PagerDuty).

## 6. Change management

- Customer's named approver must sign off on every production deploy.
- CI pipelines run in the GreenLang build org but push to the
  customer registry via OIDC federation with scoped permissions.

## 7. Compliance overlays

- Applies `FIPS-140-2` crypto profile when required (OpenSSL FIPS
  provider in the runtime image).
- Enables audit-log WORM storage via S3 Object Lock on a customer bucket.
- Retention policy forced to Enterprise tier defaults.

## 8. Validation checklist

Before cutover, verify:

- [ ] `kubectl -n factors-enterprise get netpol` shows deny-all default.
- [ ] `aws ec2 describe-vpc-endpoint-services` shows the PrivateLink service.
- [ ] `vault kv metadata get kv/factors/enterprise/signing` returns key rotated < 90d.
- [ ] No SG rule allows 0.0.0.0/0 ingress.
- [ ] HPA, PrometheusRule, ServiceMonitor all applied.
- [ ] Signed receipts required flag set (`SIGNING_REQUIRED=true`).
- [ ] Edition pin matches the customer's contract (`EDITION=<version>`).
