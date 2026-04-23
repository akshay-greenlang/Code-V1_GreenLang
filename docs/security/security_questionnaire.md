# GreenLang Factors — Security Questionnaire (Pre-Answered)

> This document consolidates the answers we give for customer RFPs and
> enterprise due-diligence questionnaires. Security owns the file;
> updates require CISO sign-off. Any paragraph marked **`LEGAL REVIEW
> REQUIRED`** must not be shared externally without Legal's approval.

## 1. Company + certifications

| Topic | Answer |
|-------|--------|
| Product covered | GreenLang Factors API (canonical climate reference layer) |
| Hosting | AWS (EKS us-east-1 primary; region-selectable for Enterprise) |
| SOC 2 Type II | Observation window active; report available under NDA on request |
| ISO 27001 | Gap assessment complete; formal certification targeted FY28 Q1 |
| PCI | Out of scope — Stripe handles cardholder data; we receive tokens only |
| GDPR | Applicable — see `docs/security/dpa_template.md` |
| CCPA / CPRA | Applicable — see DPA Schedule A |

## 2. Encryption

- **In transit:** TLS 1.3 minimum at the ingress. HSTS enabled. Weak
  ciphers disabled. Certificate rotation via cert-manager (Let's
  Encrypt prod for production).
- **At rest:** AES-256 across RDS (Postgres + pgvector), S3, Redis
  snapshots, and Velero backup archives. Keys managed by AWS KMS with
  Vault-mediated access; customer-managed keys available on Enterprise
  private-deploy overlays.

## 3. Authentication and authorization

- **SSO:** SAML 2.0 and OpenID Connect (authorization code + PKCE).
  Tested IdPs: Okta, Azure AD, Auth0, OneLogin.
- **MFA:** Enforced at the IdP. Configurable per tenant.
- **SCIM 2.0:** Full user + group lifecycle automation.
- **Internal auth:** Short-lived JWT, per-tenant API keys, tier-based
  endpoint gating, RBAC (SEC-002).
- **Session handling:** HTTP-only secure cookies for browser clients;
  bearer tokens for API clients.

## 4. Audit logging

- Structured JSON logs shipped to Loki (INFRA-009).
- All security-relevant events emit an entry with tenant_id, user_id,
  event type, outcome, and tamper-evident signature.
- Retention per tier: 30d community / 90d Pro / 1y Platform / 7y Enterprise.

## 5. Operational security

- **Vulnerability management:** Trivy + Grype in CI on every image.
  Critical CVEs block the image push. SLAs: Critical 24h, High 7d,
  Medium 30d, Low 90d.
- **Dependency scanning:** pip-audit on every PR.
- **SAST:** CodeQL on Python + TypeScript.
- **DAST:** ZAP baseline scan on staging nightly.
- **Secrets scanning:** gitleaks on every commit.

## 6. Incident response

- On-call rotation via PagerDuty (service `greenlang-factors`).
- Severity definitions: SEV1 (15 min response), SEV2 (30 min), SEV3 (2h).
- Breach notification: **72 hours** per GDPR Art. 33.
- Playbooks: `deployment/runbooks/factors_incidents.md`.
- Postmortem within 5 business days of resolution.

## 7. Availability and DR

- **SLA:** 99.9% monthly for Platform and Enterprise tiers.
- **Error budget:** monitored continuously with burn-rate alerts.
- **Backup:** daily RDS snapshots, hourly WAL archiving, cold-region
  replica. Retention 35 days; 1y for Enterprise private deploys.
- **RTO:** 4 hours. **RPO:** 1 hour.
- **Drills:** quarterly backup restore + annual region failover.

## 8. Data residency and sub-processors

- **Default region:** AWS us-east-1.
- **Alternate regions:** eu-west-1, ap-southeast-2 (Enterprise only).
- **Sub-processors:** see `docs/security/dpa_template.md` Schedule C
  and `https://greenlang.ai/subprocessors`. Notices of change: 30 days.

## 9. Access controls (internal GreenLang)

- Employee access to production gated by SSO + MFA + just-in-time
  elevation via ChatOps approval.
- Quarterly access review (CISO sign-off).
- Termination: access revoked within 4 business hours.

## 10. Privacy

- `LEGAL REVIEW REQUIRED` — the following paragraph is drafted but
  should be confirmed for each jurisdiction before it is shared.

> GreenLang acts as a Processor for Customer Personal Data submitted
> to the Factors API. We do not sell Personal Data. We honor
> Data Subject requests within 30 days via the admin endpoints or a
> direct privacy@greenlang.ai request.

## 11. Source-code protection

- GitHub Enterprise with branch protection on main and production
  branches. CODEOWNERS enforced. Commit signing required.

## 12. Physical security

- N/A beyond AWS standard. AWS SOC 2, ISO 27001, PCI certifications
  inherited for the physical layer.

## 13. Contact

- Security inbox: `security@greenlang.ai`.
- security.txt: `https://factors.greenlang.com/.well-known/security.txt`.
- Responsible disclosure: see `docs/security/SECURITY.md`.

---

_Version: SecQ-2026.04 — CISO owns. Answer updates must be reviewed in
the Security weekly._
