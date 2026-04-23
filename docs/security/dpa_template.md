# Data Processing Agreement (DPA) — Template

> **LEGAL REVIEW REQUIRED before customer sign.** This template is a
> starting point. Legal must review and tailor each engagement. Do not
> paste this into a commercial contract without their approval.

---

**This Data Processing Agreement ("DPA") forms part of the Subscription
Agreement ("Agreement") between `__CUSTOMER_LEGAL_NAME__` ("Customer",
"Controller") and GreenLang, Inc. ("GreenLang", "Processor").**

## 1. Definitions

Capitalised terms follow the meanings assigned under the EU General Data
Protection Regulation (GDPR) and the California Consumer Privacy Act
(CCPA / CPRA), as applicable. "Processing", "Personal Data",
"Controller", "Processor", "Data Subject", and "Sub-processor" carry the
meanings assigned under GDPR Art. 4.

## 2. Scope and Roles

- Customer is the Controller of Personal Data it submits to the
  GreenLang Factors API.
- GreenLang processes such Personal Data solely as a Processor on the
  documented instructions of the Customer.

## 3. Nature and Purpose of Processing

| Field | Value |
|-------|-------|
| Purpose | Provision of the Factors API (emissions factor resolution, explainability, signed receipts) |
| Categories of Data Subjects | Customer's employees, contractors, and API users |
| Categories of Personal Data | Account identifiers (email, name), usage metadata, IP address |
| Special Categories | None processed |
| Retention | Per Schedule A of this DPA and the Factors retention policy |

## 4. Sub-processors

Customer authorizes GreenLang to engage the Sub-processors listed at
`https://greenlang.ai/subprocessors`. GreenLang will provide 30-day
notice of new Sub-processors; Customer may object in writing, in which
case parties will discuss a commercially reasonable alternative.

## 5. Security Measures

GreenLang implements the technical and organizational measures
described in **Schedule B** (Technical and Organizational Measures),
including encryption in transit (TLS 1.3) and at rest (AES-256),
role-based access, SCIM provisioning, audit logging, and SOC 2 Type II
attested controls.

## 6. Data Subject Rights

GreenLang assists Customer in responding to Data Subject requests
(access, rectification, erasure, portability, objection) within the
times required by applicable law. Customer may trigger deletion via
the admin console or `/v1/admin/tenants/{id}/purge` endpoint; requests
are completed within 30 days and acknowledged in writing.

## 7. Breach Notification

GreenLang notifies Customer **within 72 hours** of becoming aware of a
Personal Data Breach affecting Customer's data, per GDPR Art. 33.
Notifications include the nature, categories, approximate number of
Data Subjects, likely consequences, and mitigation measures.

## 8. International Transfers

Where Personal Data is transferred from the EEA / UK / Switzerland to a
country not recognized as adequate, the transfer is governed by the
EU Standard Contractual Clauses ("SCCs") Module 2 (2021/914) and the
UK International Data Transfer Addendum, incorporated by reference.

## 9. Data Return and Deletion

Upon termination of the Agreement, GreenLang deletes or returns
Customer's Personal Data within 90 days, unless retention is required
by applicable law. Certificate of destruction provided on request.

## 10. Audits

Customer (or an independent auditor Customer engages) may audit
GreenLang's compliance with this DPA no more than once per 12-month
period, on 30 days' notice and during business hours. GreenLang may
satisfy this right by providing its current SOC 2 Type II report.

## 11. Changes to Law

The parties agree to negotiate in good faith any amendment required by
a change in applicable law.

---

### Schedule A — Retention Periods

| Resource | Default Retention |
|----------|-------------------|
| Application logs | Per tier (30d community / 90d Pro / 1y Platform / 7y Enterprise) |
| Signed receipts | 90d / 1y / 3y / 7y |
| Explain history | 30d / 90d / 1y / indefinite |
| Customer private factors | Tenant-controlled (default: retained until tenant deletes) |
| Raw source artifacts | 10 years |

### Schedule B — Technical and Organizational Measures

Reference: `docs/security/security_questionnaire.md` §5 (Security
Controls). Includes: TLS 1.3, AES-256 at rest, role-based access,
principle of least privilege, audit logging, separation of duties,
secure SDLC, quarterly backup drills, 99.9% availability SLO.

### Schedule C — Sub-processors (current list)

See `https://greenlang.ai/subprocessors`. Notable entries:

- AWS (EU / US hosting; Customer may choose region).
- Stripe (billing, where paid plan is in effect).
- Sentry (error telemetry, scrubbed of personal data).
- Anthropic / OpenAI (optional, for Enterprise LLM features; disabled by default).

---

**Signatures required from:** Customer authorised signatory + GreenLang authorised signatory.

_Version: DPA-2026.04 — LEGAL REVIEW REQUIRED before any customer sign._
