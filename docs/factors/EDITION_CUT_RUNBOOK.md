# GreenLang Factors v1.0 Edition Cut — Runbook

**Owner:** Methodology Lead
**Audience:** Methodology Lead + Platform Eng on-call
**Frequency:** Once per Certified release (target: 2026-05-v1.0)
**Time budget:** 90 minutes (incl. signoff meeting), assuming pre-flight is green

---

## 0. Preconditions (do these the day before)

- [ ] `factors-gold-eval` CI job on main is **green** (precision@1 ≥ 0.85 globally).
- [ ] `factors-launch-v1-coverage` matrix on main is **green** for all 6 families.
- [ ] `factors-ci` (full pytest under tests/factors) on main is **green**.
- [ ] No P0/P1 review-queue items unresolved (`gl-factors queue list --status open`).
- [ ] Source-version diff vs the previous edition has been spot-checked:
      `python -m greenlang.factors.cli edition diff <prev> <current>`
- [ ] Methodology lead has read this runbook and the auto-generated draft
      release notes from a `--dry-run`.

If any precondition is red, **stop**. Cutting a non-green edition burns
the "Certified" label.

---

## 1. Pre-flight (dry run, no writes)

```bash
export GL_FACTORS_SQLITE_PATH=/var/data/greenlang/factors.sqlite
python scripts/cut_factors_edition_v1.py --dry-run
```

The script will:

1. Boot `FactorCatalogService.from_environment()`.
2. Walk every factor and run the 9-point release-signoff checklist
   (`greenlang/factors/quality/release_signoff.py`).
3. Print the manifest summary (factor count, family counts, label counts,
   pinned source versions).
4. Exit `0` if all 9 signoff items pass, `1` if any required items fail.

**Action:** read the printed JSON. Pay attention to:

- `S1: QA gates pass for all factors` — must be 100%.
- `S2: No unresolved duplicate pairs` — must be 0.
- `S3: Cross-source consistency reviewed` — must be 0 outstanding.
- `manifest.factor_count` — sanity-check against the previous edition;
  large drops (>5%) indicate an ingestion regression and should block.

---

## 2. Sign-off meeting (synchronous)

Required attendees: Methodology Lead, Platform Eng on-call,
Compliance/Legal observer.

Agenda (30 min):

1. Walk the dry-run output (5 min).
2. Confirm the 4 gates that humans own:
   - **S4** changelog reviewed?  (Methodology lead)
   - **S6** methodology signed off?  (Methodology lead)
   - **S7** legal confirmed licensing classes?  (Legal)
   - **S8** load test passed (p95 < 500ms)?  (Platform Eng)
3. Sign the runbook PR. The PR description records the approver email
   and the timestamp; this is the audit artefact, not Slack.

If any of S4/S6/S7/S8 is "no", **abort**. Re-schedule once the gap closes.

---

## 3. Commit the edition (write)

After all four humans have signed:

```bash
export GL_FACTORS_APPROVER="methodology-lead@greenlang.ai"
export GL_FACTORS_SIGNING_SECRET="$(vault read -field=secret kv/greenlang/factors/signing)"
# Optional Ed25519 (preferred for production):
# export GL_FACTORS_ED25519_PRIVATE_KEY="$(vault read -field=key kv/greenlang/factors/ed25519)"

python scripts/cut_factors_edition_v1.py \
  --commit \
  --approver "$GL_FACTORS_APPROVER" \
  --methodology-signed \
  --legal-confirmed \
  --changelog-reviewed \
  --gold-eval-precision 0.91
```

The script will:

- Sign the canonical-JSON manifest (Ed25519 if configured, else HMAC).
- Write `greenlang/factors/data/editions/2026-05-v1.0/manifest.json` and
  `manifest.sig`.
- Update `greenlang/factors/data/editions/active.txt` to point at the
  new edition.
- Generate `docs/factors/RELEASE_NOTES_v1.0.md` with the manifest hash,
  factor counts per family, source-version table.

---

## 4. Verify

```bash
# Manifest exists and is signed:
test -f greenlang/factors/data/editions/2026-05-v1.0/manifest.json
test -f greenlang/factors/data/editions/2026-05-v1.0/manifest.sig

# Active pointer flipped:
cat greenlang/factors/data/editions/active.txt
# → 2026-05-v1.0

# CLI shows the edition:
python -m greenlang.factors.cli edition show 2026-05-v1.0

# API health reports the new edition (if API is running locally):
curl -s http://localhost:8080/v1/health | jq .edition
# → "2026-05-v1.0"

# Round-trip a known factor and confirm `edition_id` is pinned:
curl -s -H "X-GL-Edition: 2026-05-v1.0" \
  -H "X-API-Key: gl_fac_dev_001" \
  http://localhost:8080/v1/factors/EF:DESNZ:s1_natural_gas_kwh:UK:2026:v1 \
  | jq '{factor_id, edition_id, _signed_receipt}'
```

---

## 5. Tag git, publish hash

```bash
git add greenlang/factors/data/editions/2026-05-v1.0 \
        greenlang/factors/data/editions/active.txt \
        docs/factors/RELEASE_NOTES_v1.0.md
git commit -m "Cut Factors v1.0 Certified edition (2026-05-v1.0)

Approver: $GL_FACTORS_APPROVER
Manifest hash: $(sha256sum greenlang/factors/data/editions/2026-05-v1.0/manifest.json | cut -d' ' -f1)
"

git tag -s factors/v1.0 -m "GreenLang Factors v1.0 Certified edition (2026-05-v1.0)"
git push origin master --tags
```

The release notes get linked from the v1.0 git tag and from the
developer portal at `developers.greenlang.ai/changelog`.

---

## 6. Post-cut

- [ ] Promote the SQLite catalog to staging:
      `aws s3 cp /var/data/greenlang/factors.sqlite s3://gl-factors-prod/editions/2026-05-v1.0/factors.sqlite`
- [ ] Roll API workers to pick up the new active edition (Helm chart
      reads `GL_FACTORS_SQLITE_PATH` at boot; rolling restart is enough).
- [ ] Three-label dashboard (`greenlang.ai/factors/coverage`) updates
      automatically — verify it shows the new counts.
- [ ] Notify design partners via the changelog email.

---

## 7. Rollback

If something is wrong post-cut:

```bash
echo "<previous_edition_id>" > greenlang/factors/data/editions/active.txt
git commit -am "Rollback to <previous_edition_id> (incident #...)"
git push
```

Roll API workers. The factors v1.0 manifest and signature stay on disk
(immutable), so a re-cut is a fresh edition id (e.g. `2026-05-v1.1`),
never an overwrite — this preserves CTO non-negotiable #2.

---

## 8. Break-glass (`--force`)

`--force` exists for genuine emergencies (e.g. critical compliance
factor must ship even though one signoff item is yellow). It writes a
`force_committed: true` annotation into the release notes, which the
auditor will see. **Do not use it to make red signoff items green.**

```bash
python scripts/cut_factors_edition_v1.py --commit --force \
  --approver "$GL_FACTORS_APPROVER" \
  --methodology-signed --legal-confirmed --changelog-reviewed
```

---

## 9. Schedule

Cadence: ad-hoc per Certified release; minor preview cuts can happen
weekly without this runbook (preview editions don't carry the Certified
label).

The next edition (`2026-06-v1.1`) follows the same script with a new
`--target-edition`.
