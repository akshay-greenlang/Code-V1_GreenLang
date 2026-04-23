# CC8 — Change Management

**Owner:** Eng Mgr. **Review cadence:** continuous.

## Controls

### CC8.1 — Change approval

- Evidence: pull-request approval rules, CI gates, CODEOWNERS file.
- Collection: automated.
- Artifacts:
  - [ ] `.github/workflows/factors_ci.yml`.
  - [ ] `.github/CODEOWNERS`.
  - [ ] Branch protection rules screenshot (main / production branches).
  - [ ] Monthly sample of 10 PRs with reviewer approvals attached.

### CC8.2 — Change deployment

- Evidence: blue/green rollout configuration, automated rollback.
- Collection: automated.
- Artifacts:
  - [ ] `deployment/k8s/factors/blue_green/`.
  - [ ] Rollout history log (`kubectl argo rollouts history`).
