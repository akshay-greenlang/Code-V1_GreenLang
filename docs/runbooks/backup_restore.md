# Factors API - Backup / Restore Procedure

**Owner:** SRE on-call. **Drill cadence:** quarterly.
**Related:** `deployment/backup/factors_backup_drill.sh`.

## 1. What gets backed up

| Resource | Method | Frequency | Retention |
|----------|--------|-----------|-----------|
| Postgres (factors.*) | RDS automated snapshots + `pg_dump` weekly | 1/day + 1/week | 35 days (RDS), 1y cold-storage (S3) |
| pgvector tables | Part of Postgres snapshot | See above | See above |
| Redis | RDB snapshot uploaded to S3 hourly | 1/hour | 7 days |
| Signed receipts | Postgres + S3 object copies | Per-write | Per tier (90d-7y) |
| Vault KV | Raft snapshots | 1/hour | 30 days |

## 2. Routine restore

1. Identify the target timestamp (use `factors_incidents.md` for the
   incident that prompted the restore).
2. From the ops laptop, authenticate to the staging cluster:
   `aws eks update-kubeconfig --name greenlang-staging --region us-east-1`.
3. Spin up an ephemeral namespace:
   `kubectl create ns factors-restore-<date>`.
4. Apply the kustomize overlay:
   `kubectl apply -k deployment/k8s/factors/overlays/staging -n factors-restore-<date>`.
5. Populate it from the S3 snapshot:
   `bash deployment/backup/factors_backup_drill.sh --restore-from=<s3-prefix>` (use a recent drill run).
6. Verify `GET /v1/health` shows the expected edition.
7. Redirect a canary 1% of traffic via Kong; observe metrics for 30 min.

## 3. Full DR

If primary region is lost:

1. Assume the `factors-dr` role in the secondary region.
2. Promote the Postgres read replica to primary.
3. Apply the production overlay in the secondary EKS cluster.
4. Update Route53 to point `factors.greenlang.com` at the secondary ingress.
5. Announce the incident on the status page (SEV1).

## 4. Quarterly drill

The scripted drill runs end-to-end:

```bash
bash deployment/backup/factors_backup_drill.sh
```

- Dumps Postgres + Redis
- Uploads encrypted (KMS) to `s3://greenlang-factors-backup-drill`
- Restores into an ephemeral namespace
- Runs `scripts/bootstrap_catalog.py verify` against the restored DB
- Writes `/var/lib/greenlang/backup-drill/last.json` (auto-collected by
  `soc2_controls.py` for SOC 2 evidence)

**Targets:** RTO 4h, RPO 1h. Misses are SEV2 incidents.

## 5. Post-drill

- Open a PR to bump the `last_drill_at` claim in
  `docs/security/soc2_evidence/CC7_system_operations.md` (auto-updated
  when the evidence collector is run).
- Remove the ephemeral namespace: `kubectl delete ns factors-drill-*`.
- Archive the drill report in the Security drive (annual audit folder).
