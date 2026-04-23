# CC6 — Logical and Physical Access Controls

**Owner:** Security. **Review cadence:** continuous (automated) + quarterly review.

## Controls

### CC6.1 — Logical access - authentication

- Evidence: SSO (SAML/OIDC), JWT, API keys with tier gating.
- Collection: automated.
- Artifacts:
  - [ ] `greenlang/factors/middleware/sso_saml.py`.
  - [ ] `greenlang/factors/middleware/sso_oidc.py`.
  - [ ] `greenlang/factors/api_auth.py`.
  - [ ] Quarterly review of MFA enforcement per tenant.

### CC6.2 — User provisioning and deprovisioning

- Evidence: SCIM 2.0 endpoints + retention purge on deprovision.
- Collection: automated.
- Artifacts:
  - [ ] `greenlang/factors/middleware/scim.py`.
  - [ ] `greenlang/factors/middleware/retention.py`.
  - [ ] Monthly deprovisioning test report.

### CC6.3 — Access authorization

- Evidence: RBAC (SEC-002) + per-endpoint tier gate.
- Collection: automated.

### CC6.6 — Boundary protection

- Evidence: Kong ingress, NetworkPolicy, PrivateLink for enterprise deploys.
- Collection: automated.
- Artifacts:
  - [ ] `deployment/k8s/factors/base/networkpolicy.yaml`.
  - [ ] `deployment/k8s/factors/overlays/private_deploy/`.

### CC6.7 — Data in transit encryption

- Evidence: TLS 1.3 enforced via ingress; mTLS between services.
- Collection: automated.

### CC6.8 — Data at rest encryption

- Evidence: AES-256 on RDS + S3 + pgvector; Vault-managed keys.
- Collection: automated.
- Artifacts:
  - [ ] `deployment/terraform/` KMS + RDS encryption flags.
  - [ ] Vault key rotation policy (kv/factors/*/signing).
