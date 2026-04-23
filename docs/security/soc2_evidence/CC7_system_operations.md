# CC7 — System Operations

**Owner:** SRE. **Review cadence:** continuous (monitoring) + quarterly (drills).

## Controls

### CC7.1 — Change and threat detection

- Evidence: signed receipts (every response), Vault audit log, Loki anomaly alerts.
- Collection: automated.
- Artifacts:
  - [ ] `greenlang/factors/middleware/signed_receipts.py`.
  - [ ] Loki Grafana dashboard for anomalies.
  - [ ] Vault audit log retention proof.

### CC7.2 — Detection of malicious use

- Evidence: rate limiter, anomaly dashboard, brute-force alerts.
- Collection: automated.

### CC7.3 — Incident response

- Evidence: runbook, PagerDuty tree, postmortem repository.
- Collection: semi-automated (attestation per drill).
- Artifacts:
  - [ ] `deployment/runbooks/factors_incidents.md`.
  - [ ] Quarterly tabletop exercise notes.

### CC7.4 — Security event containment

- Evidence: isolate-tenant CLI, key rotation runbook.
- Collection: manual + automated.

### CC7.5 — Backup and recovery

- Evidence: quarterly backup-restore drill; RTO 4h / RPO 1h.
- Collection: automated.
- Artifacts:
  - [ ] `deployment/backup/factors_backup_drill.sh`.
  - [ ] `docs/runbooks/backup_restore.md`.
  - [ ] Last four quarterly drill reports.
