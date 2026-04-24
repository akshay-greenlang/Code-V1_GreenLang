# GreenLang Factors — Alerts Runbook

Companion to `deployment/k8s/factors/base/prometheusrule.yaml`. One section
per alert name slug, matched by `runbook_url` anchors
(`#<lowercased-alert-name>`). Each section follows:

> **Symptom** — what the alert observed.
> **Likely cause** — what to check first.
> **Remediation** — concrete steps, in order.

On-call owners:

| Severity | Routing                           |
|----------|-----------------------------------|
| warning  | Slack `#factors-alerts`           |
| critical | PagerDuty service `greenlang-factors` + Slack `#factors-incident` |

Keep this file in sync with the PrometheusRule whenever alert names or
thresholds change. Related docs:

- `deployment/runbooks/factors_incidents.md` (incident playbook)
- `docs/runbooks/factors_support.md` (edition pinning, env vars, API key rotation)

---

## FactorsAPIHighLatencyP95

- **Symptom**: p95 of `http_request_duration_seconds` for service=factors-api > 500ms for 5m.
- **Likely cause**: noisy neighbour pod, pgvector HNSW cold-start after deploy, embedding-model provider 429s, DB connection saturation.
- **Remediation**:
  1. `kubectl top pods -n greenlang-factors` — confirm no pod is CPU-capped.
  2. Check the factors-production Grafana (`greenlang-factors-production`) panel 1 for tier-level breakout.
  3. Inspect recent deploys via the dashboard's deploy annotations; consider rollback if correlated.
  4. If embedding provider 429s: temporarily raise `GL_FACTORS_EMBED_RATE_LIMIT` and file a capacity ticket.

## FactorsAPIHighLatencyP95Critical

- **Symptom**: p95 > 2000ms for 5m.
- **Likely cause**: DB outage, embedding provider outage, runaway query, or stuck pgvector index rebuild.
- **Remediation**:
  1. Page DB on-call.
  2. Rollback latest deploy if correlated (`kubectl rollout undo deploy/factors-api -n greenlang-factors`).
  3. Scale replicas up (`kubectl scale deploy/factors-api --replicas=8`).
  4. Open an incident channel and follow `deployment/runbooks/factors_incidents.md#resolve-latency`.

## FactorsAPIHighErrorRate

- **Symptom**: combined (5xx + cannot_resolve_safely) rate > 1% of all requests for 10m.
- **Likely cause**: bad deploy introducing regressions, DB connectivity loss, or a method-pack pushed with missing factors.
- **Remediation**:
  1. Panel 2 on factors-production dashboard — identify which error class is dominant.
  2. If `cannot_resolve_safely` is dominant: confirm which pack/method_profile via `factors_cannot_resolve_safely_total` labels; engage methodology lead.
  3. If `5xx`: check pod logs and DB health.
  4. Rollback if correlated with a deploy.

## FactorsAPIHighErrorRateCritical

- **Symptom**: combined error rate > 5% for 5m.
- **Likely cause**: service outage or catastrophic regression.
- **Remediation**:
  1. Rollback the most recent deploy (`kubectl rollout undo`).
  2. If already rolled back, scale to 0 and page the CTO.
  3. Post status-page update.

## FactorsGoldEvalRegression

- **Symptom**: Nightly gold-eval `factors_gold_eval_precision_at_1` < 0.33.
- **Likely cause**: Method-pack change, embedding-model rev, or resolver tweak regressing match quality. Wave 4 baseline is 37.8%; threshold of 33% gives ~5pp headroom.
- **Remediation**:
  1. Open the most recent nightly CI run (`.github/workflows/factors_gold_eval.yml`).
  2. Compare per-family P@1 against the main-branch baseline artifact.
  3. Bisect merges since the last passing run; the gold-eval gate should have caught this on PR — if it didn't, the gate itself is misconfigured.
  4. Do NOT promote this edition to Certified until P@1 recovers.

## FactorsGoldEvalR3Critical

- **Symptom**: `factors_gold_eval_recall_at_3` < 0.42.
- **Likely cause**: Broken matcher candidate-generation path.
- **Remediation**:
  1. Halt the edition-promotion pipeline (`gl factors freeze-promotion`).
  2. Page methodology on-call.
  3. Restore previous embedding checkpoint or revert the method-pack change.

## FactorsSignedReceiptFailure

- **Symptom**: `factors_signed_receipt_failures_total / factors_signed_receipts_issued_total` > 0.1% over 10m.
- **Likely cause**: Vault key-mount unhealthy, signing-key expired, or clock skew > 5 minutes.
- **Remediation**:
  1. `kubectl exec -n greenlang-factors deploy/factors-api -- curl -sf http://vault-agent:8100/v1/sys/health`.
  2. Check signing-key TTL via Vault UI.
  3. Confirm node clock via `kubectl get --raw /api/v1/nodes | jq '.items[].status.conditions'`.
  4. Do NOT disable signing — without signed receipts the audit trail is unenforceable.

## FactorsSigningFailures

- **Symptom**: `factors_signing_failures_total` rate > 0 over 5m.
- **Remediation**: Same as `FactorsSignedReceiptFailure` above.

## FactorsSourceLicenseViolation

- **Symptom**: A provenance check rejected a redistribution attempt.
- **Likely cause**: A premium-connector factor was included in a Community edition build, or an OEM-restricted source was exposed to a non-OEM customer.
- **Remediation**:
  1. Freeze the active publishing job (`kubectl scale deploy/factors-publisher --replicas=0`).
  2. Page legal on-call.
  3. Do NOT unfreeze until legal signs off.
  4. Log the offending `factor_id` + `license_class` pair for post-mortem.

## FactorsAPIReplicasBelowPDB

- **Symptom**: Fewer than 2 healthy replicas of factors-api.
- **Likely cause**: Crash-looping pods, node drain in progress, or cluster capacity shortage.
- **Remediation**:
  1. `kubectl get pods -n greenlang-factors` — identify failing pods.
  2. Check `kubectl describe pod <name>` for scheduling/image-pull errors.
  3. Verify HPA is not stuck at `minReplicas` incorrectly.
  4. Escalate to platform on-call if cluster capacity.

## FactorsAPICrashLooping

- **Symptom**: > 3 restarts/15m on any factors-api pod.
- **Remediation**: `kubectl logs -n greenlang-factors <pod> --previous`; rollback if the cause is the latest image.

## FactorsTenantLeakSuspected

- **Symptom**: `X-GreenLang-Tier` header disagrees with JWT claim.
- **Likely cause**: Credential replay, mis-configured proxy stripping auth, or middleware bug.
- **Remediation**:
  1. Capture the offending request ID from logs.
  2. Revoke the associated JWT (`gl factors revoke-token <jti>`).
  3. Page security on-call.
  4. Do not close the alert until root cause confirmed (log the incident in `docs/security/incidents/`).

## FactorsResolveLatencyHigh

- Per-family p95 > 500ms. Same playbook as `FactorsAPIHighLatencyP95` but scoped to `/v1/resolve`. Check panel 1 of the factors-production dashboard.

## FactorsResolveLatencyCritical

- Per-family p95 > 2s. Same playbook as `FactorsAPIHighLatencyP95Critical`.

## FactorsResolveErrorRateHigh

- Outcome != success > 1% for 10m. Check panel 2. Usually a bad deploy, occasionally a methodology gap — inspect `outcome` labels.

## FactorsResolveErrorRateCritical

- Outcome != success > 5% for 5m. Rollback and page on-call.

## FactorsCannotResolveSafelySpike

- **Symptom**: `cannot_resolve_safely` events > 5/min for 10m.
- **Likely cause**: Method-profile / pack mismatch, or a customer sending a new activity type we don't yet support.
- **Remediation**:
  1. Look at `pack_id` + `method_profile` labels to identify the gap.
  2. File a methodology-gap ticket.
  3. Offer the customer a looser `method_profile` if contract permits.

## FactorsEntitlementDenialsSpike

- **Symptom**: Denials > 10/min for 10m.
- **Likely cause**: Credential probing / stolen key, or a botched tier migration.
- **Remediation**:
  1. Filter denials by `tier` + `class` to identify the pattern.
  2. If a single tenant: contact them and rotate their API key.
  3. If widespread: check whether the last tier-migration PR silently flipped `entitled_packs`.

## FactorsSourceWatchDown

- **Symptom**: 3+ watch failures for a single source_id in 30m.
- **Remediation**:
  1. Check source URL, auth creds in Vault, network egress.
  2. If upstream is down: silence alert for 4h and track upstream status page.
  3. If auth rotated: update Vault secret and restart watch worker.

## FactorsCatalogStale

- **Symptom**: No successful source-watch run for a source_id in > 36h.
- **Remediation**:
  1. Verify the source-watch CronJob is scheduled and recent runs exist.
  2. Check if this source was intentionally paused (see `config/source_watch_pauses.yaml`).
  3. Re-run the watch manually (`kubectl create job --from=cronjob/factors-source-watch manual-$(date +%s)`).

## FactorsBillingLag

- **Symptom**: Stripe webhook lag > 15m.
- **Likely cause**: Webhook consumer paused, DB write throughput saturated, or Stripe outage.
- **Remediation**:
  1. Check the Stripe status page.
  2. Verify the `factors-stripe-webhook` deployment is healthy.
  3. Inspect DB `factors_billing_events` write latency.
  4. If needed, replay events using `scripts/replay_stripe_webhooks.py` (always `--dry-run` first).

## FactorsBillingLagCritical

- **Symptom**: Stripe webhook lag > 1h.
- **Remediation**: Same as above, plus page Commercial on-call — revenue-recognition events are delayed.

## FactorsBatchBacklog

- **Symptom**: > 100 pending batch jobs for 15m.
- **Remediation**:
  1. Scale factors-worker: `kubectl scale deploy/factors-worker --replicas=N`.
  2. Check for poison-pill jobs that are retry-looping.
  3. If DB throughput is the bottleneck, verify `factors_batch_jobs` index health.

## FactorsWebhookDLQ

- **Symptom**: Webhook DLQ non-empty for 15m.
- **Likely cause**: Customer endpoint returning 4xx/5xx.
- **Remediation**:
  1. Inspect DLQ entries: `gl factors webhook list-dlq`.
  2. Contact the customer if the endpoint is theirs.
  3. Requeue after their fix: `gl factors webhook requeue --job-id=<id>`.
  4. If our config is wrong (e.g., signing secret mis-rotated): fix and requeue.
