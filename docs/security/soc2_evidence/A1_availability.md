# A1 — Availability Criterion

**Owner:** SRE. **Review cadence:** monthly (SLO) + quarterly (DR).

## Controls

### A1.1 — Performance monitoring

- Evidence: Prometheus metrics + Grafana dashboards.
- Collection: automated.

### A1.2 — Capacity management

- Evidence: HPA config, monthly load tests, cost + headroom dashboards.
- Collection: automated.
- Artifacts:
  - [ ] `deployment/k8s/factors/base/hpa.yaml`.
  - [ ] `tests/factors/load/factors_load_test.py` run report.
  - [ ] Monthly capacity review notes.

### A1.3 — Availability SLO tracking

- Evidence: 99.9% monthly availability target with error budget burn alerts.
- Collection: automated.
- Artifacts:
  - [ ] `greenlang/factors/sla/sla_tracker.py` report.
  - [ ] Error-budget burn-rate alert rules.

### A1.4 — Recovery

- Evidence: quarterly DR drill; `deployment/backup/factors_backup_drill.sh`.
- Collection: automated.
