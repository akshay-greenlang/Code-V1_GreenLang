# CC2 — Communication and Information

**Owner:** Security + Eng Mgr. **Review cadence:** quarterly.

## Controls

### CC2.1 — Quality information

- Evidence: inventory of logging pipelines (Loki, Prometheus, Grafana).
- Collection: automated via `soc2_controls.py` (reads OBS-001/OBS-002).

### CC2.2 — Internal communication of objectives and responsibilities

- Evidence: runbook index, on-call rotation schedule, incident slack channels list.
- Collection: semi-automated.
- Artifacts:
  - [ ] `deployment/runbooks/` listing.
  - [ ] PagerDuty on-call export.

### CC2.3 — External communication

- Evidence: customer status page incidents, security.txt, breach-notification SOP.
- Collection: semi-automated.
- Artifacts:
  - [ ] `deployment/status_page/` config.
  - [ ] `docs/security/security.txt`.
  - [ ] Breach notification SOP (references DPA template).
