# GreenLang Factors API — Incident Runbook

> **Scope:** Factors API (`factors-staging` and `factors-prod` namespaces).
> **Audience:** On-call engineer and SRE. Every procedure assumes you already
> have `kubectl` against the cluster and access to Vault, Grafana, and
> PagerDuty.
> **Related:** `deployment/k8s/factors/base/prometheusrule.yaml` (alert
> sources), `deployment/observability/grafana/dashboards/factors.json`
> (the staging dashboard), `GreenLang_Factors_CTO_Master_ToDo.md` DEP8 row.

---

## Severity definitions

| Severity | Definition | Response time | Who gets paged |
|----------|------------|---------------|----------------|
| SEV1 (critical) | Customer-visible outage, audit-trail breach, or wrong factor returned at scale | 15 min | Oncall → Eng Manager → CTO within 30 min if not mitigated |
| SEV2 (major) | Degraded SLO, single-source data loss, elevated error-rate | 30 min | Oncall → Eng Manager within 1h if not mitigated |
| SEV3 (minor) | Single-tenant impact, slow queries, non-blocking source-watch failure | 2h (next business day after hours) | Oncall handles, no escalation required |

**Paging path:** PagerDuty service `greenlang-factors` → #factors-oncall Slack
→ Eng Manager (`@tara`) → CTO (`@akshay`). Never wait longer than 30 min on
SEV1 without escalation.

---

## Universal evidence-capture checklist

Before touching anything, gather these artifacts. Paste links into the
incident Slack thread.

1. Alert name and trigger time (from PagerDuty).
2. Grafana snapshot: `GreenLang Factors — Staging` (or prod) for the 30
   min window bracketing the alert.
3. `kubectl -n <ns> get pods,svc,ingress,deploy,hpa -o wide`
4. `kubectl -n <ns> logs deploy/factors-api --tail=500 --timestamps`
5. Latest `release_version` and `edition` from
   `GET /v1/health` and `GET /v1/editions/<current>`.
6. User-supplied request ID (`X-GreenLang-Request-Id` response header)
   if the report came from a customer.
7. Vault path audited: `vault kv metadata get kv/factors/<env>/signing`
   (confirms last rotation).

---

## Incident 1 — Wrong factor returned

**Symptoms:** Customer reports the factor value or units do not match the
authoritative source; `/v1/factors/{id}/explain` points at an unexpected
`source_id` or `method_profile`. Could also surface as a flood of
`FactorsCannotResolveSafelySpike` when the guard is working correctly.

**Procedure**

1. Capture the full report: `factor_id`, `release_version`,
   `method_pack_version`, client-side edition pin, jurisdiction.
2. Call `/v1/factors/{id}/explain?method_profile=<profile>` and compare the
   returned `alternates` list — verify the selected factor matches the
   highest-ranked viable candidate.
3. Query the override table:
   `SELECT * FROM factors.overrides WHERE factor_id=$1 ORDER BY created_at DESC;`
   A stale tenant override is the most common cause.
4. Check `deprecation_status` on the factor row — if `deprecated` or
   `retired`, the resolver should have skipped it; if it did not, this is a
   resolver bug and goes to Agent B on-call.
5. If the explain shows a structural methodology issue (wrong fallback
   rank logic, wrong profile applied), engage the methodology lead and
   mark the incident SEV2.
6. Mitigation: pin the affected tenant to the previous `release_version`
   via `kubectl -n <ns> exec deploy/factors-api -- gl-factors override-pin
   --tenant <id> --release <prev>`. Public remediation follows in the
   next release changelog.

**DO NOT** bypass the override table to "fix" a single customer report —
every change must be reproducible from the signed edition manifest.

---

## Incident 2 — Source license issue

**Symptoms:** Legal notification, upstream provider revocation, or the
`FactorsSourceLicenseViolation` / `FactorsSourceWatchDown` alerts fire in
a way correlated with a known source (e.g. DEFRA, IEA, EPA eGRID).

**Procedure**

1. Identify every factor carrying the affected `source_id`:
   `SELECT factor_id FROM factors.factors WHERE source_id=$1;`
2. **Quarantine:** set `status='retired'` and
   `retired_reason='source-license-revoked'` for every affected
   `factor_id` in the current edition draft. Do NOT publish a new edition
   yet — this lands in a hotfix edition after legal sign-off.
3. Notify `legal@greenlang.ai`; open a ticket in the legal tracker and
   attach the upstream correspondence.
4. Post a public changelog entry flagging the retirement window.
5. **Entitlement audit:** `SELECT tenant_id, COUNT(*) FROM factors.usage
   WHERE source_id=$1 AND ts > now() - interval '30 days' GROUP BY 1;` —
   proactively contact customers who consumed the retired factors inside
   the SLA window.
6. Cut a hotfix `release_version`, sign it, and promote through staging.

---

## Incident 3 — API outage

**Symptoms:** `FactorsAPIReplicasBelowPDB`, `FactorsAPIHighErrorRate`,
readiness probe failures, or a blue dashboard with zero traffic but
healthy ingress.

**Procedure**

1. Check IngressRoute and Service:
   `kubectl -n factors-staging get ingressroute,ingress,svc -o wide` and
   `kubectl -n factors-staging describe ingress factors-api`.
2. Check pod health:
   `kubectl -n factors-staging get pods -o wide` and
   `kubectl -n factors-staging describe pod <failing-pod>`. Look at
   `Events:` for OOM, ImagePullBackOff, or probe failures.
3. Check DB pool and Redis:
   `kubectl -n factors-staging exec deploy/factors-api -- python -c
   "from greenlang.factors.service import FactorCatalogService;
   FactorCatalogService.from_environment().repo.health_check()"`.
4. Check signing keys with the new endpoint:
   `curl -sSf -H "X-API-Key: $STAGING_KEY"
   https://factors-staging.greenlang.com/v1/health/signing-status`.
   - If `signing_installed=false`: the ExternalSecret did not sync.
     `kubectl -n factors-staging get externalsecret factors-api-secrets
     -o yaml` — look at `status.conditions`. Fix the Vault mount or the
     `ClusterSecretStore vault-staging` auth binding. **Do NOT disable
     signing** to get traffic back — failover the cluster before you
     disable receipts.
5. Rollback if a recent deploy triggered the outage:
   `kubectl -n factors-staging rollout undo deploy/factors-api`.

---

## Incident 4 — Bad release

**Symptoms:** `FactorsResolveErrorRateHigh`,
`FactorsCannotResolveSafelySpike`, FQS distribution shifts visibly
downward, or customers report systematic numeric drift after a release
promotion.

**Procedure**

1. Identify the offending `release_version` via
   `GET /v1/editions/<current>` (look at `promoted_at`).
2. Roll back the edition pin:
   `kubectl -n factors-staging exec deploy/factors-api -- gl-factors
   edition-pin --set <prev-release-version>`.
3. Confirm the rollback:
   `curl -sSf https://factors-staging.greenlang.com/v1/health | jq .edition`.
4. Freeze all publish jobs: `kubectl -n factors-staging scale
   deploy/factors-worker-ingest --replicas=0`.
5. Run the full regression suite against the pinned-back edition;
   compare gold-eval labels.
6. Root cause → post-mortem in `docs/postmortems/factors/<date>.md`
   within 5 business days.

---

## Incident 5 — Signed-receipt verification failure

**Symptoms:** `FactorsSignedReceiptFailures` /
`FactorsSigningFailures` alerts. Customer reports their verifier rejects
receipts signed since a known timestamp.

**Procedure**

1. Compare the fingerprint the customer was expecting vs the one in
   `/v1/health/signing-status` (`key_fingerprint`). A mismatch means we
   published or rotated without coordinating.
2. Check `rotation_status`:
   - `current` + mismatch → bug in the verifier client, triage via sdk
     channel.
   - `due` → expected soft-warning, schedule rotation this week.
   - `overdue` → immediate rotation via Vault:
     `vault kv put kv/factors/<env>/signing ed25519_priv=@new_priv.pem
     ed25519_pub=@new_pub.pem rotated_at=$(date -u +%FT%TZ)
     next_rotation_due_at=$(date -u -d '+180 days' +%FT%TZ)`.
3. If a stale public key is live in the customer-facing docs, mark
   outstanding receipts (signed with the retired key) as invalid in the
   `receipts.revocations` table; email affected customers.
4. **Never** serve the old key from two pods at once — run a rolling
   restart after `SignedReceiptsMiddleware` picks up the new env:
   `kubectl -n factors-staging rollout restart deploy/factors-api`.

---

## Incident 6 — DB / pgvector failover

**Symptoms:** DB connection errors in logs, `pgvector` query latency
spike, `FactorsResolveP95High` firing for every family simultaneously.

**Procedure**

1. Check the RDS cluster via AWS console or
   `aws rds describe-db-clusters --db-cluster-identifier factors-staging`.
2. If the writer is unhealthy and a read-replica exists, promote it:
   `aws rds failover-db-cluster --db-cluster-identifier factors-staging`.
3. Update the ExternalSecret-backed `DB_URL` in Vault if the endpoint
   DNS changed:
   `vault kv patch kv/factors/staging/db url=postgresql://...`.
   External Secrets Operator refreshes within 5 min; to force:
   `kubectl -n factors-staging annotate externalsecret factors-api-secrets
   force-sync=$(date +%s) --overwrite`.
4. Warm the Redis cache: run the warm-up job
   `kubectl -n factors-staging create job --from=cronjob/factors-cache-warmup
   cache-warmup-$(date +%s)`.
5. Reconcile the pgvector HNSW index if it is out of sync:
   `kubectl -n factors-staging exec deploy/factors-api -- python -m
   greenlang.factors.index_manager --reconcile`.
6. Monitor resolve latency for 30 min post-failover; expect an elevated
   baseline for the first 10 min until the cache rewarms.

---

## Rollback quick reference

```bash
# Kustomize-level rollback to the previously-applied revision
kubectl -n factors-staging rollout history deploy/factors-api
kubectl -n factors-staging rollout undo deploy/factors-api --to-revision=<N>

# Edition pin rollback (no pod restart needed)
kubectl -n factors-staging exec deploy/factors-api -- \
  gl-factors edition-pin --set <prev-release-version>

# Emergency scale-up to absorb an unexpected traffic spike
kubectl -n factors-staging scale deploy/factors-api --replicas=6
```

---

## Vault prerequisites (one-time, SRE-owned)

Before the first staging apply the SRE team must:

1. Create the `ClusterSecretStore vault-staging` pointing at
   `https://vault.security.svc.cluster.local:8200` with Kubernetes
   auth mounted at `kubernetes` and role `factors-staging`.
2. Write the following keys at Vault path `kv/factors/staging/`:
   - `db.url`
   - `redis.url`
   - `jwt.secret`
   - `webhooks.hmac_secret`
   - `signing.ed25519_priv`, `signing.ed25519_pub`,
     `signing.rotated_at`, `signing.next_rotation_due_at`
   - `stripe.api_key`, `stripe.webhook_secret`
3. Allowlist the `factors-staging` namespace ServiceAccount
   `staging-factors-api` in the Vault policy for that path.
4. Provision the DNS CNAME `factors-staging.greenlang.com → <ingress-lb>`
   and the cert-manager `letsencrypt-staging` cluster-issuer.
