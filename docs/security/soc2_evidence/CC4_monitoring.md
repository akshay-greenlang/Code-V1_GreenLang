# CC4 — Monitoring Activities

**Owner:** SRE + Security. **Review cadence:** continuous.

## Controls

### CC4.1 — Ongoing and separate evaluations

- Evidence: Prometheus + Grafana dashboards; alert rule set.
- Collection: automated.
- Artifacts:
  - [ ] `deployment/observability/grafana/dashboards/factors.json`.
  - [ ] `deployment/k8s/factors/base/prometheusrule.yaml`.
  - [ ] Monthly alert noise review (PagerDuty export).

### CC4.2 — Deviations communicated and corrected

- Evidence: incident postmortem folder + action-item tracker.
- Collection: semi-automated.
- Artifacts:
  - [ ] Postmortem directory listing.
  - [ ] Jira link for open remediation tickets.
