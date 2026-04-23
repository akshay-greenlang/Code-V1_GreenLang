# GreenLang Factors — RFP Response Pack

> **Audience:** Enterprise procurement, security review boards, legal
> assessors. Pre-answered responses to the top 40 RFP questions.
> **Pricing proposal v1 — subject to CTO / Commercial approval.**
> **Version:** 1.0 (2026-04-23)

---

## A. Company + Product Overview

### A.1 Company
GreenLang AI Inc., Delaware C-corp, founded 2024. HQ: San Francisco, CA.
Additional engineering: London, Bangalore.

### A.2 Product
GreenLang Factors is a climate-reference data API providing signed,
provenance-rich emission factors for corporate GHG accounting,
regulatory reporting (CSRD, CBAM, TCFD, ISSB, CDP), and product carbon
footprinting.

### A.3 Funding + runway
Private. Runway > 18 months at current burn. Financial references
available under NDA.

---

## B. Security

### B.1 Compliance certifications
- **SOC 2 Type II** — report available under NDA
- **ISO 27001** — certified
- **GDPR** — DPA available; Standard Contractual Clauses supported
- **CCPA** — compliant; DSR endpoint available at `/v1/privacy/dsr`
- **HIPAA** — N/A (we do not process PHI)

Evidence bundles linkable via the GreenLang Trust Center:
`https://trust.greenlang.io` (under NDA).

### B.2 Encryption
- **At rest**: AES-256-GCM for all customer data; keys managed via AWS
  KMS / Azure Key Vault / GCP Cloud KMS depending on deployment
- **In transit**: TLS 1.3 only; TLS 1.2 for legacy compatibility (disabled
  by default, opt-in)
- **API signing**: Ed25519 on every response; key rotation quarterly

### B.3 Authentication + authorization
- **API keys** — scoped, rotatable, revocable
- **JWT** — for user-facing flows; RS256 with rotating JWKS
- **SSO** — SAML 2.0 + OpenID Connect; tested with Okta, Azure AD,
  Google, Ping, OneLogin
- **SCIM 2.0** — for user provisioning
- **RBAC** — 8 built-in roles + custom-role support

### B.4 Data residency
- **Shared SaaS** — US-East (Virginia) primary, EU-West (Frankfurt)
  secondary
- **VPC deployment** (Enterprise) — customer-selected region
- **Dedicated EU** — opt-in, separate subdomain, no US replication

### B.5 Penetration testing
- Annual third-party pentest (Bishop Fox)
- Continuous vulnerability scanning (Snyk, Trivy, GitHub Advanced Security)
- Last pentest: 2026-01-15; report available under NDA

### B.6 Incident response
- 24/7 on-call rotation
- **RTO**: 4 hours; **RPO**: 15 minutes
- Customer notification: within 72 hours (GDPR Article 33 compliant)
- Post-incident review + public RCA within 14 days

---

## C. SLA + Support

### C.1 Uptime SLA
| Tier | Uptime |
|---|---|
| Community | No SLA |
| Developer Pro | 99.5% |
| Consulting | 99.9% |
| Platform | 99.9% |
| Enterprise | 99.95% (credit schedule in MSA) |

### C.2 Response time SLAs (Enterprise)
| Severity | Response | Resolution target |
|---|---|---|
| P1 (service down) | 15 minutes | 4 hours |
| P2 (degraded) | 1 hour | 8 hours |
| P3 (impaired) | 4 hours | 2 business days |
| P4 (question) | 1 business day | 5 business days |

### C.3 Support channels
- **Community**: Slack (community.greenlang.io)
- **Developer Pro**: Email (support@greenlang.io)
- **Consulting**: Dedicated Slack-Connect channel
- **Platform**: Slack-Connect + named TAM
- **Enterprise**: 24/7 phone + Slack + TAM + backup TAM

---

## D. Data Rights

### D.1 Who owns what?
- **Customer data** (your uploads, overrides, audit bundles): customer-owned
- **GreenLang catalog** (our factors, methodology): GreenLang-owned,
  licensed to customer per MSA
- **Customer-derived factors** (your overrides + your inputs):
  customer-owned, available for export on demand

### D.2 Data retention
- **Active customer**: data retained for lifetime of agreement
- **Terminated customer**: 30-day grace period for export, then deletion
- **Audit bundles**: retained per customer configuration (default 7
  years for Enterprise, 1 year for Pro)

### D.3 Data egress
- Full-fidelity export via `/v1/export` (JSON Lines, CSV, Parquet)
- Audit bundle export via `/v1/audit-bundle` (tar.gz with Ed25519
  signature file)
- No egress fees
- API-only; no "lock-in" SQL dumps

### D.4 Customer audit rights
- **Shared SaaS**: annual written audit of SOC 2 evidence
- **VPC / Enterprise**: on-site audit (1 per year, 3 auditor-days
  included; additional at time + materials)

---

## E. Deployment Options

### E.1 Shared SaaS
- US East / EU West
- Multi-tenant with strict isolation
- Available to all tiers

### E.2 VPC
- Customer's AWS / Azure / GCP account
- Our software, your infrastructure
- Available: Platform + Enterprise
- SLA: 99.9% (we still operate; dependencies within your VPC are
  customer-owned)

### E.3 On-prem / Air-gapped
- Not supported for v1. Roadmap item for 2027.
- Interim: VPC deployment in a regulated-cloud region (AWS GovCloud,
  Azure Gov, GCP Assured Workloads)

---

## F. Sample Flows

### F.1 Redaction flow
1. Customer uploads spend data (contains supplier PII)
2. PII detector flags supplier email + phone columns
3. Customer chooses redact / tokenize / retain
4. Redacted copy used for factor matching; original encrypted + vaulted
5. Audit bundle references redacted copy only

### F.2 Redistribution flow (Platform tier)
1. OEM customer onboards end-customer as sub-tenant
2. End-customer queries `/resolve` via OEM's white-label domain
3. Response signed by OEM key + GreenLang key (chain)
4. Usage metered to OEM's Stripe account; end-customer invisible to GL
5. Per-tenant license enforcement restricts `customer_private` factors
   to the owning sub-tenant only

### F.3 Audit export flow
1. Customer requests audit bundle for period [2025-01-01, 2025-12-31]
2. System assembles every `/resolve` receipt in range
3. Includes: factor record, explain block, source attribution, signed
   receipt, edition manifest
4. Packaged as `tar.gz` with Ed25519 signature + SHA-256 manifest
5. Delivered via signed S3 URL (24-hour expiry) or direct download

---

## G. Pricing + Commercials

### G.1 Pricing model
Per the **pricing proposal v1** (subject to CTO / Commercial approval):

| Tier | Monthly | Annual |
|---|---|---|
| Community | $0 | — |
| Developer Pro | $299 | $2,990 |
| Consulting | $2,499 | $24,990 |
| Platform | $4,999 | $50,000 |
| Enterprise | Custom ($75K-$500K typical) | — |

Premium Data Packs (add-on): $399-$999 / month each.

### G.2 Payment terms
- Annual invoicing (net 30) standard for Consulting / Platform / Enterprise
- Credit card monthly for Developer Pro
- Customer PO + wire supported

### G.3 Price protection
- 12-month price lock on annual contracts
- Multi-year contracts available with step-ups documented upfront

### G.4 Volume discounts
- 10% off at $100K ACV
- 15% off at $250K ACV
- 20% off at $500K ACV
- Custom above $750K ACV

---

## H. Open Questions + Redlines

We understand most procurement teams have specific language
requirements. We redline our MSA / DPA / SLA templates directly.
Typical turnaround: 5 business days for first redline pass.

Common redline topics we have pre-prepared positions on:
- Indemnification caps
- Data processing subprocessor list
- Insurance coverage (we carry $10M cyber + $5M E&O)
- Change-control process for service updates
- Termination for convenience / for cause

---

**Contact:** [rfp@greenlang.io](mailto:rfp@greenlang.io). Reference this
pack version 1.0 in your procurement request to accelerate our response.
