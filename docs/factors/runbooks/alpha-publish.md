# Operator Runbook — Factors v0.1 Alpha Staging-to-Production Flip

**Audience:** Climate-methodology lead + Release engineer / SRE on-call.
**Wave:** E / TaskCreate #23 / WS9-T1.
**CTO doc reference:** §19.1 — *"Runbook for a manual 'publish to production
namespace' step: climate-methodology lead reviews staging diffs and flips
visibility."*

This runbook walks through the manual gate that promotes v0.1 alpha factor
records from the open `staging` namespace to the closed `production`
namespace. It is the only sanctioned path for surfacing new factors to the
SDK's partner-key callers.

---

## 1. Pre-flight checks

Before any subcommand runs:

1. **Environment variables.** Decide whether you are operating against
   *local SQLite* (interactive review on a laptop) or *staging Postgres*.

   * Local SQLite (default):
     ```bash
     export GL_FACTORS_ALPHA_DSN="sqlite:///./alpha_factors_v0_1.db"
     ```
   * Staging Postgres:
     ```bash
     export GL_FACTORS_ALPHA_DSN="postgresql://factors_alpha:***@stg-pg-1.greenlang.internal:5432/factors"
     ```

2. **Alpha profile active.** The release profile MUST be `alpha`:
   ```bash
   python -c "from greenlang.factors.release_profile import get_active_profile; print(get_active_profile())"
   # expected: alpha
   ```
   If this prints `beta` or `ga`, abort — the alpha publish flow is
   strictly v0.1-shape and the contract differs in beta+.

3. **Repo schema present.** For Postgres, verify Alembic 0001 has been
   applied (`SELECT 1 FROM factors_v0_1.factor LIMIT 0`). The publisher
   adds the `namespace` column idempotently on first call.

4. **Methodology lead role assigned.** The flip MUST be approved by a
   `human:<email>` identifier whose mailbox the lead controls. There is
   NO bot-approved flip path. Check the on-call rota for who is acting
   methodology lead today; their email goes in `--approved-by`.

5. **Slack channel armed.** Post the heads-up in `#factors-launch`
   *before* the flip — this is the audit anchor that pairs with the
   `factor_publish_log` row.

---

## 2. Stage — push seeds into the staging namespace

Run once per source you intend to release. Seeds live under
`greenlang/factors/data/catalog_seed_v0_1/<source_id>/v1.json` (Wave D #6
backfill output).

```bash
python scripts/factors_alpha_publish.py staging --source ipcc_2006_nggi
python scripts/factors_alpha_publish.py staging --source desnz_ghg_conversion
python scripts/factors_alpha_publish.py staging --source egrid
# ...repeat per alpha-flagged source
```

Each invocation prints `published=N already_staged=M failed=K`. The
`AlphaProvenanceGate` runs at this step — any record missing
`extraction.raw_artifact_sha256`, an approver email, or with the wrong
`gwp_basis` is rejected here and never reaches the DB.

If `failed > 0`, FIX THE SEED. Do not proceed to a flip. Failed records
are reported with their URN to stderr.

---

## 3. Diff review — generate the methodology-lead artifact

```bash
python scripts/factors_alpha_publish.py diff --write
# diff report written to out/factors/v0_1_alpha/staging-diff-{ts}.md
```

The Markdown report contains four sections:

* **Additions** — staging URNs without a production counterpart.
* **Removals** — production URNs missing from staging (rare; should be
  zero unless you are deliberately retiring records).
* **Changes** — `(old_urn, new_urn)` pairs where the staging record's
  `supersedes_urn` points at a production record.
* **Unchanged** — sanity count of identical rows in both namespaces.

Open the file. **Do not skip this step** — the next subcommand is
irreversible-without-rollback once it runs.

---

## 4. Methodology-lead sign-off

The lead reviews the diff and signs off in writing. Acceptable sign-off
artifacts:

* Reply in `#factors-launch` Slack with the report file name and
  `:approved:`.
* Email to `factors-release@greenlang.io` quoting the diff summary line.
* PR comment on the audit ticket.

The sign-off MUST quote the `Summary` line from the report
(e.g. `+12 additions, -0 removals, ~3 supersedes, =180 unchanged`) so a
later auditor can pair the approval with the report.

---

## 5. Flip — promote to production

Bulk path (most common — every staged record promoted in one batch):

```bash
python scripts/factors_alpha_publish.py flip \
    --all-staging \
    --approved-by human:methodology-lead@greenlang.io
```

Single-URN path (used when only one record needs to land):

```bash
python scripts/factors_alpha_publish.py flip \
    --urn 'urn:gl:factor:ipcc-2006-nggi:IPCC:en_stat_anthracite_unknown_gj:GLOBAL:2019:v1' \
    --approved-by human:methodology-lead@greenlang.io
```

The CLI prints `flip: promoted=N of M requested approved_by=...`. The
flip is **idempotent**: re-running the same command after a successful
flip is a no-op (already-production URNs are skipped, and no extra
publish-log entries are added). The batch_id surfaces in the log table
for traceability — capture it from the log query in step 7 if you need
to roll back.

---

## 6. Verify — SDK fetch via partner API key

Confirm the records are now visible to the SDK:

```bash
curl -H "Authorization: Bearer $GL_PARTNER_API_KEY" \
     "https://api-staging.greenlang.io/v0.1/factors/<urn>"
```

Or use the SDK directly:

```python
from greenlang.sdk.alpha_v0_1 import AlphaFactorsClient
client = AlphaFactorsClient(api_key=os.environ["GL_PARTNER_API_KEY"])
record = client.get_factor("urn:gl:factor:ipcc-2006-nggi:...:v1")
assert record["review"]["review_status"] == "approved"
```

The fetch should return the freshly promoted record. If it returns 404,
something between the publisher and the API edge is mis-configured —
escalate to SRE before issuing more flips.

---

## 7. Rollback — demote a batch

If the lead retracts approval, demote the batch back to staging.

```bash
# 1. Find the batch_id (the flip CLI prints it; if not captured, query):
python -c "
from greenlang.factors.release import AlphaPublisher
from greenlang.factors.repositories import AlphaFactorRepository
import os
pub = AlphaPublisher(AlphaFactorRepository(os.environ['GL_FACTORS_ALPHA_DSN']))
for r in pub.list_log()[-20:]:
    print(r)
"

# 2. Run rollback:
python scripts/factors_alpha_publish.py rollback \
    --batch-id flip-<id> \
    --approved-by human:methodology-lead@greenlang.io
```

A rollback **demotes** records back to `staging` — it does NOT delete
them. The v0.1 alpha contract is URN-immutable. The rollback also writes
an append-only entry to `factor_publish_log` so the visibility history is
fully recoverable.

---

## 8. Five things to watch for during a flip

1. **Large additions** — if `additions` count exceeds the historical
   median by 10x, pause. Likely cause: a parser regression duplicated
   records under fresh URNs.
2. **Mass removals** — any non-zero `removals` deserves scrutiny.
   Removals only appear when a URN that was in production is now absent
   from staging *and* not superseded. Almost always a seed-file mistake.
3. **Supersede chains crossing source boundaries** — a `changes` entry
   where `old_urn` and `new_urn` belong to different `source_urn` values
   is a methodology error. Reject the flip.
4. **Missing approver email** — the CLI rejects empty
   `--approved-by` and any value not starting with `human:`. If the lead
   is on PTO, do NOT substitute a bot identifier; defer the flip.
5. **Drift in `factor_pack_urn`** — if the staging records carry a
   different `factor_pack_urn` than the production records they are
   replacing, the pack manifest is mis-aligned. Verify pack version
   bumps with the methodology team.

---

## 9. Local SQLite vs Staging Postgres — quick reference

| Step | Local SQLite | Staging Postgres |
|------|--------------|------------------|
| DSN | `sqlite:///./alpha_factors_v0_1.db` | `postgresql://...stg-pg-1.../factors` |
| Schema bootstrap | Auto on first publisher call | Alembic 0001 must be pre-applied |
| Namespace column | Added via `ALTER TABLE alpha_factors_v0_1 ADD COLUMN namespace TEXT NOT NULL DEFAULT 'staging'` (idempotent) | Added via `ALTER TABLE factors_v0_1.factor ADD COLUMN IF NOT EXISTS namespace ...` |
| Publish log table | `factor_publish_log` (root schema) | `factors_v0_1.factor_publish_log` |
| Concurrency | Single-writer; CLI invocations serialise on file lock | Postgres row-level locking |
| Rollback latency | Instant (single UPDATE) | Single UPDATE; replication lag may delay SDK visibility |

---

## 10. Acceptance criterion

This runbook satisfies CTO doc §19.1 verbatim: the methodology lead
reviews the staging diff produced by step 3, signs off via the artifact
in step 4, and flips visibility via the command in step 5. Steps 1, 2,
6, and 7 are operational scaffolding. The flip itself is purely a
visibility change — the gate runs at staging entry, not at flip time.
