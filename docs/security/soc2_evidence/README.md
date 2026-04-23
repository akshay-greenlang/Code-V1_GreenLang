# SOC 2 Type II Evidence (Factors API)

This directory holds the per-control evidence checklist consumed by our
SOC 2 auditor. One markdown file per Trust Services Criteria (TSC) group.

| File | Criterion | Coverage |
|------|-----------|----------|
| `CC1_control_environment.md` | CC1 | Code of conduct, workforce competence |
| `CC2_communication.md` | CC2 | Information quality, internal + external comms |
| `CC3_risk_assessment.md` | CC3 | Annual risk assessment, threat model |
| `CC4_monitoring.md` | CC4 | Continuous monitoring, deviations handling |
| `CC5_control_activities.md` | CC5 | Technology general controls, segregation of duties |
| `CC6_logical_access.md` | CC6 | Authentication, authorization, encryption |
| `CC7_system_operations.md` | CC7 | Threat detection, incident response, backup/recovery |
| `CC8_change_management.md` | CC8 | Change approval, review, deployment |
| `A1_availability.md` | A1 | Capacity management, SLO tracking, DR |

## How evidence is collected

- **Automated** controls: `greenlang/factors/security/soc2_controls.py`
  pulls live metrics from Prometheus, audit logs, backup-drill receipts,
  and vulnerability scan outputs. Run via `GET /v1/admin/soc2/evidence`
  (admin tier).
- **Semi-automated** controls: machines collect the facts, a human
  signs off on the interpretation (e.g., incident postmortems).
- **Manual** controls: policy documents and annual attestations.
  Owners listed per file.

## Refresh cadence

Audit window is monthly. Before each audit cycle, the Security lead:

1. Runs `python -m greenlang.factors.security.soc2_controls --write out.json`.
2. Attaches the JSON to the audit evidence folder.
3. Reviews stale controls (`status=stale`) and fills gaps.
4. Signs off on manual controls with date + initials in this folder.

## Legal review

Documents in this folder that describe **company commitments** (not
technical controls) must be reviewed by Legal before publication to
customers. These are marked `LEGAL REVIEW REQUIRED` in their file
headers.
