# Licensed Data Connectors — Onboarding Playbook (PLATFORM 4, task #28)

Status: **Blocked on business partnerships.** Code frameworks exist; credentials and legal agreements do not.

## Target sources (ranked by factor count)

| Source | Est. factors | License model | Typical lead time |
|--------|--------------|---------------|--------------------|
| ecoinvent | ~20,000 LCA activities | Annual site license, per-user | 60-90 days |
| IEA Emissions Factors | ~5,000 grid + sector | Annual subscription | 30-60 days |
| Electricity Maps | ~2,000 real-time grid intensities | SaaS subscription + API | 2-4 weeks |
| Sphera GaBi | ~15,000 LCA datasets | Per-user seat license | 60-90 days |
| Carbon Disclosure Project (CDP) | ~50,000 company-level | Member access + data license | 90 days |

## Code status

Connector scaffolds already exist at:
- `greenlang/factors/connectors/iea_connector.py`
- `greenlang/factors/connectors/ecoinvent_connector.py`
- `greenlang/factors/connectors/electricity_maps_connector.py`

Each includes:
- Authentication stub (awaits real credentials)
- Response-parser mapping to `EmissionFactorRecord`
- License-tag gating via `factor_record.license_info`
- Enterprise-tier visibility flag

## Contract-to-prod checklist (once credentials secured)

1. **Legal (30 days)**
   - Counsel review of data-license terms
   - Tenant-flow agreement (who can view, redistribute, derive from)
   - DPA/GDPR addenda if EU-resident data

2. **Security (7 days)**
   - Credentials stored in Vault (SEC-006) under `greenlang/factors/licensed/<vendor>/`
   - Network egress allowlist updated (policy linter)
   - Audit-log policy: every licensed-factor read recorded

3. **Engineering (7 days)**
   - Wire credentials via ExternalSecret CRD (already templated)
   - Enable connector in `greenlang/factors/source_registry.yaml`
   - Run phase-source-acquire + phase-parse against licensed endpoint
   - Promote through Q1-Q6 + approval gate (G5-G6)
   - Tag factors with `tier_required=enterprise`, `license=<vendor>-<year>`

4. **Go-live (1 day)**
   - Re-generate embeddings (only for new licensed factors)
   - Announce to enterprise tenants via changelog
   - Update sales collateral with new coverage numbers

## Target: 180K additional factors (cumulative 280K) once all 5 partnerships close.

## Why this is unblocked on engineering

All the plumbing is in the codebase. The 30-day contract-to-prod SLA assumes:
- Vault is operational (done, SEC-006)
- Helm chart supports ExternalSecrets (done, task #11)
- Connector framework is tested (factors test suite)
- Tier enforcement is live (done, task #26)

The ONLY missing pieces are:
- Signed MSAs / data-license agreements with each vendor
- API credentials issued to a GreenLang account
