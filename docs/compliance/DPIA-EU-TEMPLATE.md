# Data Protection Impact Assessment (DPIA) — EU Customer Deployment Template

**Document type**: Template
**Applies to**: GreenLang Factors API + Platform deployments where at
least one data subject is resident in the EEA (including UK under
UK GDPR and Switzerland under the FADP).
**Methodology**: ICO UK "Sample DPIA template" (v0.5) + EDPB
"Guidelines on Data Protection Impact Assessment (wp248rev.01)".
**Owner**: Data Protection Officer (DPO) — see §4
**Cadence**: annual review + mandatory re-assessment on any "major
change" (new sub-processor, new data category, new third-country
transfer, new authentication mechanism).
**Template version**: v1.0 — 2026-04-22

How to use this template
------------------------
1. Copy this file to `docs/compliance/dpia/DPIA-<customer>-<YYYYMMDD>.md`.
2. Fill every section. **Do not** delete unanswered sections — write
   "Not applicable" + justification. Empty sections are an audit finding.
3. DPO signs §4 and §11 after review.
4. File a copy in the customer's data-room + the central DPIA register
   `docs/compliance/dpia/REGISTER.md`.
5. Schedule the review date (§11) into the ops calendar.

---

## 1. Processing activity

### 1.1 Systematic description

| Field | Entry |
|---|---|
| Processing name | GreenLang Factors API — emission-factor resolution, tenant overlays, signed responses |
| Go-live date | YYYY-MM-DD |
| Processing environment | `prod.factors.greenlang.io` |
| Customer | {customer_legal_name} |
| GreenLang contract ref | {contract_id} |
| Factors-API edition pinned | {edition_id} |

### 1.2 Nature, scope, context, and purpose (Art 35(7)(a) GDPR)

**Nature of processing.** The GreenLang Factors API receives
machine-to-machine requests from the customer's sustainability reporting
system, resolves an emission factor using the 7-step cascade
(`greenlang/factors/resolution/engine.py`), and returns a signed
response plus an audit trail. Processing is **automated** and
**non-human-in-the-loop** for the default path; the LLM rerank step
(`greenlang/factors/matching/llm_rerank.py`) is Enterprise-tier opt-in
and is gated by the rate limiter.

**Scope of processing.** Customer supplies (a) an API key issued via
the GreenLang ops console, (b) a JWT carrying `tenant_id`, `user_id`,
and `tier`, and (c) an `activity_text` / factor-id lookup key. The
platform resolves a factor, optionally applies a tenant overlay, and
returns the chosen factor, its alternates, and the audit bundle.

**Context.** The customer has a contractual obligation to report
Scope 1/2/3 emissions under CSRD (EU Directive 2022/2464), UK SECR,
and/or US SB 253. Processing is therefore embedded in a **statutory
compliance workflow** — Recital 47 GDPR "legitimate interest" is
supported by the corresponding statutory obligation on the controller.

**Purpose.**

* Resolve an emission factor for a given activity (primary purpose).
* Maintain an append-only audit log of every resolution for assurance
  purposes (7-year retention, SEC-005).
* Meter API usage for billing (Stripe integration, §2.5 below).
* Detect and prevent abuse (rate-limit audit, license firewall, SEC-010).

### 1.3 Data categories processed

| # | Category | Example | Source | Retention | Legal basis | Notes |
|---|---|---|---|---|---|---|
| D1 | Identifiers | `user_id`, `tenant_id`, API key suffix | Customer JWT | 7 y (SEC-005) | Art 6(1)(f) | `api_key_full` never stored |
| D2 | Technical metadata | Source IP, user-agent, TLS fingerprint | Kong ingress | 90 d (Loki INFRA-009) | Art 6(1)(f) | anonymised after retention |
| D3 | Request context | `activity_text`, `jurisdiction`, `reporting_date` | Customer request body | 2 y | Art 6(1)(b) | Can carry employee names if customer embeds them — see §6.4 |
| D4 | Signed receipts | HMAC/Ed25519 signature + body hash | Server-generated | 7 y | Art 6(1)(f) | Immutable |
| D5 | Overlay metadata | Supplier audit values, actor, notes | Customer tenant admin | 7 y | Art 6(1)(b) | Per-tenant; never cross-tenant |
| D6 | Billing records | Stripe customer id, seat count, usage volume | Stripe webhook | 10 y (tax law) | Art 6(1)(b) + (c) | Forwarded to Stripe (US sub-processor) |
| D7 | Audit events | Who resolved what, when, with which profile | FastAPI audit mw | 7 y | Art 6(1)(c) | Append-only immutable log |

**None of the above are "special category" data under Art 9 GDPR**
unless the customer writes a special-category label into D3 or D5
themselves. The DPO reviews quarterly samples (see §8) to confirm.

### 1.4 Data subjects

| Data subject category | Volume estimate | Notes |
|---|---|---|
| Customer's employees (users of the reporting system) | {e.g. 200} | Primary population |
| Customer's supplier contacts (if supplier audit uploaded) | {e.g. 5000} | Only if supplier module is enabled |
| Customer's end-customers | None | Not processed |

---

## 2. Parties

### 2.1 Controller

| Field | Entry |
|---|---|
| Legal name | {customer_legal_name} |
| Address | {customer_registered_office} |
| Art 27 representative (if outside EEA) | {representative} |
| DPO contact (customer) | {customer_dpo_email} |

### 2.2 Processor

| Field | Entry |
|---|---|
| Legal name | GreenLang Inc. |
| Address | {greenlang_registered_office} |
| DPO contact | dpo@greenlang.io |
| Authorised representative (EU) | {EU_representative} |

### 2.3 Sub-processors (Art 28(2) GDPR)

| # | Name | Role | Location | Safeguard | Data categories exposed |
|---|---|---|---|---|---|
| SP1 | Amazon Web Services (Ireland) | Infrastructure (EKS, RDS, S3) | eu-west-1 / eu-central-1 | SCCs 2021/914 (Module 3); AWS GDPR DPA | D1–D7 (encrypted at rest) |
| SP2 | Datadog EU | Observability / Loki mirror | eu-west-1 | SCCs 2021/914 (Module 2); Datadog DPA | D2, D7 (metrics + logs) |
| SP3 | Stripe Payments Europe Ltd | Billing | Ireland (EU); CDE via US | SCCs + Ireland data-residency option | D6 |
| SP4 | HashiCorp Cloud (Vault) | Secret storage | eu-central-1 | SCCs; HashiCorp DPA | Encryption keys only |
| SP5 | Anthropic PBC (Enterprise LLM rerank only) | Re-ranking | US (Oregon) | SCCs + TIA §7.3; zero-retention contract clause | D3 activity_text only |
| SP6 | OpenAI Ireland Ltd (Enterprise LLM rerank alt.) | Re-ranking | Ireland + US | SCCs + TIA §7.3; zero-retention contract clause | D3 activity_text only |
| SP7 | GitHub (GreenLang Inc.) | Source-of-truth repo, GHA CI | US | SCCs + MSA | Never customer data |

---

## 3. Necessity and proportionality

### 3.1 Legal basis (Art 6 GDPR)

| Activity | Basis | Rationale |
|---|---|---|
| Factor resolution (primary API) | Art 6(1)(f) legitimate interest | Customer requires the output to meet statutory GHG reporting duties; interest balanced against data subject rights in §3.2 |
| Tenant overlay CRUD | Art 6(1)(b) contract performance | Enterprise-tier contract explicitly includes overlay registry |
| Billing + metering | Art 6(1)(b) + (c) | Contract + VAT + revenue-recognition obligations |
| Audit log retention (7 y) | Art 6(1)(c) + (f) | SOC 2 CC7, ISAE 3000, CSRD assurance requirement |
| Abuse detection + security logs | Art 6(1)(f) | Recital 49 — network/information security |

### 3.2 Legitimate-interest balancing test (LIA) for Art 6(1)(f)

| Leg of the LIA | Assessment |
|---|---|
| (i) Purpose | Enable the customer to meet statutory emissions-reporting duties; secure the service against abuse. |
| (ii) Necessity | No less-intrusive alternative: the signed-receipt model requires a server-stored signature; abuse detection requires IP + UA capture. |
| (iii) Balance | Data subjects are employees of the corporate controller using a reporting system they were instructed to use; minimal expectation of privacy for role-based identifiers. No automated decisions with legal effects (Art 22) are taken. |

### 3.3 Data minimisation (Art 5(1)(c))

* API accepts only `user_id`, `tenant_id`, `tier`, `activity_text`, and
  optional `jurisdiction` / `reporting_date`.
* `activity_text` is capped at 8 KB (WAF rule) and the LLM rerank path
  further truncates to `max_candidates=20` before sending to the
  provider.
* Raw IP addresses are retained 90 d; after that they are hashed using
  a per-tenant salt (INFRA-009 sanitiser).

### 3.4 Accuracy (Art 5(1)(d))

* Factor values are append-only versioned (Non-negotiable #2 —
  `edition_manifest.py`).
* Customer can correct tenant overlays via the API; every correction is
  audit-logged.

### 3.5 Storage limitation (Art 5(1)(e))

| Dataset | Retention | Automatic deletion | Reference |
|---|---|---|---|
| Audit log | 7 y | S3 object-lock + lifecycle rule | SEC-005 |
| Loki logs | 90 d | Retention policy | INFRA-009 |
| Request body cache | 24 h | Redis TTL | INFRA-003 |
| Billing events | 10 y | Revenue-recognition policy | Finance |
| Overlay audit | 7 y | Per DB retention job | tenant_overlay_audit |

---

## 4. Technical and organisational measures (TOMs, Art 32 GDPR)

### 4.1 Technical measures

| Measure | Control | Component |
|---|---|---|
| Encryption in transit | TLS 1.3 only; HSTS `max-age=63072000; includeSubDomains` | SEC-004 |
| Encryption at rest | AES-256 SSE-KMS on S3; AES-256 on RDS EBS | SEC-003 |
| Per-tenant encryption (Enterprise) | Vault transit backend keyed by `tenant_id` | SEC-006 + INFRA-008 |
| Key rotation | JWT 30 d; Ed25519 signing 90 d; OEM sub-key 24 h | Vault PKI |
| Signed response receipts | HMAC-SHA256 / Ed25519 with body-hash | `middleware/signed_receipts.py` |
| Pseudonymisation | Audit actor = opaque `user_id` (not email); IP hashed after 90 d | INFRA-009 |
| RBAC | 8 roles; SoD author ≠ approver | SEC-002 + `review_workflow.py` |
| PII scanner | Every parser_log + raw-source write | `security/pii_scanner.py` (SEC-011) |
| Network segmentation | K8s NetworkPolicy + Linkerd mTLS | INFRA-001 |
| WAF | AWS WAFv2 with Managed Rules, rate limit, geo-allow, bot control | `deployment/waf/factors-marketing.yaml` |
| Monitoring | Prometheus (OBS-001), Loki (INFRA-009), Tempo (OBS-003) | — |
| Backup + PITR | RDS 7-day PITR + 30-day daily snapshots | INFRA-002 |
| Vulnerability management | GHA `pip-audit` + `trivy image` on every PR | SEC-007 |

### 4.2 Organisational measures

| Measure | Detail |
|---|---|
| Access management | Quarterly access review; all grants via ops console + Slack audit trail |
| Segregation of duties | Author ≠ approver on every release-signoff (`release_signoff.py`) |
| Rotation cadence | See §4.1 row "Key rotation" |
| Onboarding + offboarding | Git-backed role definitions; automated revoke on HR termination event |
| Staff training | Annual security + GDPR refresher; role-specific training for anyone with production access |
| Incident response | 5-scenario playbook in `docs/security/incident-response.md`; tabletop exercise quarterly |
| Breach notification (Art 33) | DPO notified within 24 h of detection; supervisory authority within 72 h; data-subject notification (Art 34) if high-risk |
| Vendor due diligence | Every sub-processor listed in §2.3 carries a signed DPA + TIA |

---

## 5. Cross-border transfers (Chap V GDPR)

### 5.1 Transfer map

| From | To | Sub-processor | Mechanism |
|---|---|---|---|
| EEA | EEA (Ireland) | AWS Ireland, Stripe Europe | No transfer — processing stays in EEA |
| EEA | US | Stripe (card processing), Anthropic, OpenAI (if enabled), GitHub (source-of-truth) | SCCs 2021/914 + TIA (§5.2) |
| EEA | Oregon, US | Anthropic PBC (Enterprise LLM rerank) | SCCs + TIA + zero-retention clause |

### 5.2 Transfer Impact Assessment (TIA) — US sub-processors

**Step 1 — Mapping.** See §5.1.

**Step 2 — Applicable law in third country (US).**
FISA 702, Executive Order 12333, Cloud Act. Following *Schrems II* the
SCCs alone are insufficient without supplementary measures.

**Step 3 — Effectiveness of SCCs.**
EU-US Data Privacy Framework (DPF) adopted 10 July 2023; Stripe,
Anthropic, and OpenAI are certified under DPF. Certification has been
verified at `https://www.dataprivacyframework.gov/` on the review date
(§11).

**Step 4 — Supplementary technical measures.**
1. End-to-end TLS 1.3 for every sub-processor hop.
2. For Anthropic + OpenAI (LLM rerank): only `activity_text` is sent;
   never a factor value, never a user identifier, never a tenant
   identifier. Zero-retention clauses in both contracts.
3. For Stripe: customer email field is pseudonymised via a customer
   token before it is forwarded (GreenLang billing layer).
4. For GitHub: never customer data — only source code.

**Step 5 — Supplementary organisational + legal measures.**
* Transparency: customer receives a yearly "sub-processor notice" and
  can object within 30 d (Art 28(2)).
* Defensive publication: if any US authority serves a Section 702
  request, GreenLang will (to the extent legally permitted) publish the
  fact in the trust-centre transparency report.

**Step 6 — Ongoing review.**
DPO reviews TIA annually and on any DPF de-certification event.

---

## 6. Risk register

### 6.1 Risk scoring

| Risk ID | Description | Inherent L × I | Mitigation | Residual L × I |
|---|---|---|---|---|
| R1 | Cross-tenant data leak via bad SQL filter | M × H | RLS + regression test `test_cross_tenant_leak.py`; STRIDE T#26 | L × H |
| R2 | LLM rerank leaks `activity_text` to provider | M × M | Zero-retention DPA; SCCs; `max_candidates=20` cap; no factor values in prompt | L × M |
| R3 | Licensed-connector value reaches a non-entitled caller | M × H | 451 firewall (`license_manager.py`); regression test `test_connector_only_451.py` | L × H |
| R4 | Unauthorised overlay mutation | L × H | SoD + RBAC (SEC-002); audit log; overlay-audit regression | L × M |
| R5 | Breach of Stripe webhook forge | L × H | HMAC verification; idempotency; ops alert on unknown event type | L × L |
| R6 | Key exfil from Vault | L × H | AppRole TTL 30 d; audit device; per-tenant transit | L × M |
| R7 | US Section-702 compelled access to sub-processor | L × H | DPF + TIA §5.2 step 4–6; transparency report | L × M |
| R8 | PII accidentally embedded in `activity_text` | M × M | PII scanner (SEC-011); DPO quarterly sample review | L × M |
| R9 | Data-subject rights request unfulfilled in 30 d | L × M | Rights workflow in ops console + 30-day SLA alert | L × L |
| R10 | Sub-processor sub-sub-processor change without notice | L × M | Quarterly vendor-posture audit | L × L |

### 6.2 Residual risk statement

The residual risks above are all L (likelihood) × {L, M, H} (impact).
No residual is H × H after mitigation. The DPO accepts the residual
portfolio as **proportionate** to the statutory compliance purpose
stated in §3.1, subject to the annual review in §11.

---

## 7. Data-subject rights (Arts 15–22 GDPR)

| Right | Article | How GreenLang honours it | SLA |
|---|---|---|---|
| Access | 15 | Controller (customer) queries the audit log via `/api/v1/factors/{id}/audit-bundle`; GreenLang assists within 10 business days | 30 d (Art 12(3)) |
| Rectification | 16 | Controller updates tenant overlay; audit log records the correction | 30 d |
| Erasure ("right to be forgotten") | 17 | Controller-triggered DELETE on tenant cascades to tenant_overlay + entitlements; 7-y audit log retained under Art 17(3)(e) legal-obligation exemption | 30 d |
| Restriction | 18 | Tenant is flagged `restricted`; reads succeed, writes refused | 30 d |
| Portability | 20 | Controller exports via `/api/v1/factors/export` (JSON Lines) + audit bundle | 30 d |
| Objection | 21 | Controller opts out of LLM rerank via `X-GreenLang-Opt-Out: llm_rerank` header | Immediate |
| No automated decision | 22 | Factor resolution has no legal effect on a data subject; customer is informed in the Acceptable Use Policy | — |

**Rights request escalation path.** Customer submits via
`privacy@{customer}` → forwarded to `dpo@greenlang.io` → ops console
creates a ticket → DPO triages within 24 h.

---

## 8. Consultation outcomes

### 8.1 Internal stakeholders consulted

| Role | Name | Date | Outcome |
|---|---|---|---|
| DPO | {name} | {date} | Approved with residuals in §6 |
| Security lead | {name} | {date} | Approved |
| Legal counsel | {name} | {date} | Approved |
| Factors Eng lead | {name} | {date} | Approved |
| Customer DPO | {name} | {date} | Approved |

### 8.2 Art 35(9) consultation with data subjects (if applicable)

If the population in §1.4 exceeds 10k data subjects or includes
special-category contacts, GreenLang publishes a DPIA summary on the
trust centre and invites comments for 14 d before go-live.

### 8.3 Prior consultation with supervisory authority (Art 36)

Required only if the residual risk after mitigation is "high". The
current residual portfolio in §6 does not meet that threshold, so no
Art 36 consultation is initiated. **Document the decision** in this
section with the DPO signature.

---

## 9. Breach notification procedure (Art 33 + 34)

1. **Detection.** Alert fires in PagerDuty (OBS-004) or a security
   review surfaces a finding.
2. **Triage (T+0h).** On-call engineer pages the DPO.
3. **Assessment (T+24h).** DPO evaluates whether the breach is
   "likely to result in a risk to the rights and freedoms" of data
   subjects. If so, Art 33 notification is triggered.
4. **Supervisory authority (T+72h).** DPO files with the lead DPA
   (Ireland DPC for EU customers; ICO UK for UK customers) using the
   standard notification template.
5. **Data-subject notification (T+72h — "without undue delay").** If
   the breach is likely to result in a **high** risk, controller
   notifies affected data subjects in plain language. GreenLang
   supports controller with message copy + per-tenant mailing list.
6. **Post-mortem (T+7d).** Root-cause written up and filed in
   `docs/incident-response/{YYYY-MM-DD}-<name>.md`; STRIDE model
   (`STRIDE-THREAT-MODEL-FACTORS.md`) updated within 14 d.

---

## 10. Records of processing activities (Art 30)

This DPIA is the authoritative record for the processing activity
described in §1. The `docs/compliance/dpia/REGISTER.md` file maintains a
rolling index. Retention of the DPIA itself: 5 years after the
processing ends, per our standard records-management policy.

---

## 11. Review cadence + sign-off

### 11.1 Review triggers

| Trigger | Action |
|---|---|
| Annual | Full re-read + refresh of §5 TIA |
| New sub-processor | Add to §2.3, re-score §6 rows that touch it |
| New data category | Add to §1.3, re-score §6 |
| New third-country transfer | Re-run §5 TIA |
| Incident with GDPR impact | Re-score §6, update §9 numbers |
| Major architectural change | Full re-read |

### 11.2 Sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| Customer DPO | _______________ | _______________ | _______________ |
| GreenLang DPO | _______________ | _______________ | _______________ |
| Customer Legal | _______________ | _______________ | _______________ |
| GreenLang CTO | _______________ | _______________ | _______________ |

Next review date: **YYYY-MM-DD** (set by DPO during sign-off).

---

## Appendix A — Cross-reference to internal controls

| GDPR article | Internal control | Component |
|---|---|---|
| Art 5 | Data minimisation + accuracy + storage limitation | See §3 table |
| Art 6 | Legal basis documented per activity | See §3.1 |
| Art 7 (consent) | Not applicable — Art 6(1)(b)/(c)/(f) instead | — |
| Art 9 (special category) | Not applicable unless controller embeds in `activity_text`; DPO quarterly sample review | §1.3 |
| Art 12–22 | Data-subject rights procedure | §7 |
| Art 25 | Data protection by design + by default | SEC-001..011 suite |
| Art 28 | Processor contract + SCCs | §2.3 |
| Art 30 | Records of processing | §10 |
| Art 32 | Technical + organisational measures | §4 |
| Art 33–34 | Breach notification | §9 |
| Art 35 | DPIA (this document) | — |
| Art 36 | Prior consultation (if needed) | §8.3 |
| Art 37–39 | DPO role | §4.2, §11 |
| Art 44–50 | Cross-border transfers | §5 |

## Appendix B — UK-GDPR adaptations

Where the customer is a UK controller:

* Swap supervisory authority from DPC Ireland to ICO UK.
* Use the UK International Data Transfer Agreement (IDTA) or the UK
  Addendum to the EU SCCs in §5.1.
* Retention of breach records per ICO "Report a breach" guidance.

## Appendix C — Swiss-FADP adaptations

Where the customer is a Swiss controller:

* Add the FDPIC (Federal Data Protection and Information Commissioner)
  to the §9 notification list.
* Replace SCCs with the revised SCCs recognised by the FDPIC.
* Acknowledge no `Chapter V` equivalent — Swiss residents benefit from
  the same protections via the revised FADP.
