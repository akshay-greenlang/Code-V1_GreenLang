# Factors API - Blue/Green Rollback Playbook

**Scope:** production cut-over for the Factors API.
**Applies to:** Argo Rollouts manifests (`rollout.yaml`) and Flagger
(`flagger.yaml`). Pick whichever is deployed to the cluster.

## 0. Pre-requisites

- `kubectl` with `factors-production` context.
- Argo Rollouts plugin (`kubectl argo rollouts`) if using Argo.
- PagerDuty access to confirm alerting state before/after.
- Read-only Grafana dashboard open on `Factors API - Production`.

## 1. When to roll back

Automatic rollback fires when:

- p95 latency on canary > 1.5x stable for 5 minutes,
- 5xx rate on canary > 1% for 5 minutes,
- Fast-burn SLO > 6x for 5 minutes.

Manually roll back when:

- Customer-reported wrong factor values (resolver regression).
- Signed-receipt verification failures in the wild.
- Edition manifest mismatch between canary and stable.
- Data-quality dashboards flag a regression not covered by automation.

## 2. Procedure (Argo Rollouts)

```bash
# 1. Freeze new canary steps (automatic on alert; manual otherwise)
kubectl argo rollouts abort factors-api -n factors-production

# 2. Roll back to previous revision (swaps canary weight to 0)
kubectl argo rollouts undo factors-api -n factors-production

# 3. Verify stable pods are healthy
kubectl argo rollouts get rollout factors-api -n factors-production

# 4. Confirm traffic is 100% stable via Kong dashboards
```

## 3. Procedure (Flagger)

```bash
# 1. Trigger manual rollback
kubectl -n factors-production annotate canary factors-api \
  flagger.app/rollback="true" --overwrite

# 2. Watch canary state until primary is restored
kubectl -n factors-production get canary factors-api -w
```

## 4. Post-rollback

- Update status page to "Monitoring" state (see
  `docs/runbooks/status_page_updates.md`).
- Capture evidence: save the Rollout / Canary status YAML, Grafana snapshot,
  last 500 lines of logs for the failing revision.
- Open postmortem ticket; schedule review within 5 business days.
- If the rolled-back image has persisted to a registry tag that can be
  accidentally promoted (e.g., `latest`), retag with a `broken-` prefix.

## 5. Escalation

- SEV1 if customer-visible for > 15 minutes.
- SEV2 if SLO burn sustained but no complete outage.
- SEV3 if latency only regressed.

Full escalation tree: `deployment/runbooks/factors_incidents.md` section
"Severity definitions".
