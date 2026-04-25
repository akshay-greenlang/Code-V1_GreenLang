# GreenLang Factors v0.1 Alpha — Edition Cut Runbook

**Owner:** Climate-Methodology Lead + Backend Lead
**Frequency:** On every signed-off catalog change for the v0.1 Alpha track
**Spec refs:** CTO doc §6.3 (per-pack checksums), §19.1 (URN scheme sign-off)

This runbook is the **canonical procedure** for cutting a v0.1 Alpha edition
of the GreenLang Factors catalog. It produces a deterministic, hash-locked,
optionally Ed25519-signed bundle suitable for SDK pinning.

---

## 1. Pre-flight gate

Before you cut, verify:

- [ ] **Provenance gate passes for ALL records.** The Wave D #6 backfill
      brought every v0.1 record to a passing provenance state. Re-run:

      ```bash
      pytest tests/factors/v0_1_alpha/test_alpha_provenance_gate.py -x
      ```

      and inspect `docs/factors/v0_1_alpha/PROVENANCE-BACKFILL-REPORT.md`
      (Wave D #4) and the per-source vintage audit
      (`docs/factors/v0_1_alpha/SOURCE-VINTAGE-AUDIT.md`, Wave D #6).

- [ ] **Schema is still frozen.** The cut hashes
      `config/schemas/factor_record_v0_1.schema.json` and
      `config/schemas/FACTOR_RECORD_V0_1_FREEZE.md`. If either changed,
      that's a **new** edition (different `manifest_sha256`); confirm the
      change is intentional via the freeze-note process before continuing.

- [ ] **All `tests/factors/v0_1_alpha/` tests are green** on the cut commit.

- [ ] **Per-source counts match expectations.** Quick check:

      ```bash
      python -c "
      import json, pathlib
      root = pathlib.Path('greenlang/factors/data/catalog_seed_v0_1')
      total = 0
      for src in sorted(root.iterdir()):
          v1 = src / 'v1.json'
          if v1.is_file():
              n = len(json.loads(v1.read_text())['records'])
              print(f'{src.name}: {n}')
              total += n
      print('TOTAL:', total)
      "
      ```

      Expected (post-Wave-D #6): `TOTAL: 691`.

---

## 2. Methodology-lead review checklist

Send the climate-methodology lead the following **before** running the cut:

- [ ] Per-source vintage audit (`SOURCE-VINTAGE-AUDIT.md`) — flag any
      sources older than 18 months from publication.
- [ ] Methodology-exception files under
      `docs/factors/v0_1_alpha/methodology-exceptions/` — confirm every
      exception has an explicit accept/reject decision.
- [ ] URN-scheme worked examples per CTO §19.1 — sign-off must be
      recorded in writing (email or PR comment) before cut.
- [ ] Approver string for the manifest. Format:
      `human:<email@domain>` (e.g.
      `human:methodology-lead@greenlang.io`).

When the lead replies "approved", capture their reply timestamp — that
becomes `methodology_lead_approved_at` in the manifest.

---

## 3. Cut command

```bash
python scripts/factors_alpha_cut_edition.py \
    --edition-id factors-v0.1.0-alpha-2026-04-25 \
    --out releases/ \
    --approver human:methodology-lead@greenlang.io
```

Arguments:

- `--edition-id` (optional): if omitted, defaults to today's UTC date in
  the form `factors-v0.1.0-alpha-YYYY-MM-DD`. The format is **strictly
  validated** — a typo in the id raises `ValueError` and aborts the cut.
- `--out`: output directory root. The cut writes into
  `<out>/<edition-id>/`.
- `--approver`: must match the format `(human|bot):<email>`.
- `--no-sign`: skip Ed25519 signing even if the env var is set
  (writes `manifest.json.sig.placeholder` instead).

Outputs (all under `releases/<edition-id>/`):

- `manifest.json` — the canonical-JSON manifest.
- `manifest.json.sig` *or* `manifest.json.sig.placeholder` — Ed25519
  signature, or a placeholder note when no key is available.
- `RELEASE_NOTES.md` — auto-generated from the manifest.
- `MANIFEST_HASH.txt` — single line: `sha256:<hex>`.

---

## 4. Sign with the Ed25519 key

The cut script reads the methodology-lead's private key from the
environment variable `GL_FACTORS_ED25519_PRIVATE_KEY` (PEM-encoded).
The same shape of key powers `greenlang/factors/signing.py`'s receipt
signing, so a single key can serve both flows.

### Generate a fresh keypair (one-time)

```bash
# Private key (PEM, PKCS8, no encryption)
openssl genpkey -algorithm ed25519 -out methodology_lead_ed25519.pem

# Derive the public key
openssl pkey -in methodology_lead_ed25519.pem -pubout \
    -out methodology_lead_ed25519.pub.pem
```

Store the **private** key in Vault under
`secret/factors/release/methodology-lead-ed25519`. The **public** key
ships alongside every signed edition (e.g. published in the SDK and on
the GreenLang docs site under `/factors/keys/`).

### Inject the key for a cut

```bash
export GL_FACTORS_ED25519_PRIVATE_KEY="$(cat methodology_lead_ed25519.pem)"
python scripts/factors_alpha_cut_edition.py \
    --edition-id factors-v0.1.0-alpha-2026-04-25 \
    --out releases/
unset GL_FACTORS_ED25519_PRIVATE_KEY
```

If the env var is unset, `manifest.json.sig.placeholder` is written
instead. Downstream `verify_manifest(...)` returns `False` for any
`*.placeholder` file — i.e. the SDK pin treats the cut as **unattested**
until a signed re-cut lands.

---

## 5. Verify

```bash
python -c "
from pathlib import Path
from greenlang.factors.release.alpha_edition_manifest import verify_manifest
ok = verify_manifest(
    manifest_path=Path('releases/factors-v0.1.0-alpha-2026-04-25/manifest.json'),
    signature_path=Path('releases/factors-v0.1.0-alpha-2026-04-25/manifest.json.sig'),
    public_key=Path('keys/methodology_lead_ed25519.pub.pem').read_bytes(),
)
print('verified:', ok)
"
```

`verify_manifest(...)` returns:

- `True` — signature checks out against the bundled manifest bytes.
- `False` — any of: signature missing, placeholder, tampered manifest,
  wrong public key. The function never raises on bad input; bad input
  always means "unverified".

To independently confirm the rollup hash:

```bash
python -c "
import hashlib, json
from pathlib import Path
data = json.loads(Path('releases/factors-v0.1.0-alpha-2026-04-25/manifest.json').read_bytes())
expected = data['manifest_sha256']
data.pop('manifest_sha256')
canonical = json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
computed = hashlib.sha256(canonical).hexdigest()
print('expected:', expected)
print('computed:', computed)
assert computed == expected
"
```

---

## 6. Publish + tag

1. Commit the entire `releases/factors-v0.1.0-alpha-YYYY-MM-DD/` directory
   on a release branch:

   ```bash
   git add releases/factors-v0.1.0-alpha-2026-04-25/
   git commit -m "release: factors v0.1.0-alpha 2026-04-25"
   ```

2. Tag the commit with the same edition id:

   ```bash
   git tag -a factors-v0.1.0-alpha-2026-04-25 -m "Alpha edition cut"
   git push origin factors-v0.1.0-alpha-2026-04-25
   ```

3. Upload the four artefacts (`manifest.json`, `manifest.json.sig`,
   `RELEASE_NOTES.md`, `MANIFEST_HASH.txt`) to the public release
   storage (S3 bucket `greenlang-factors-editions/`).

4. Open a PR against the SDK pin (next step).

---

## 7. SDK pin update

The SDK reads its current edition from
`greenlang/factors/release_profile.py`. Bump `current_edition` (or the
equivalent constant — the SDK release-profile module owns the source of
truth) to the new edition id, then run:

```bash
pytest tests/factors/v0_1_alpha/test_release_profile.py
pytest tests/factors/v0_1_alpha/test_sdk_alpha_surface.py
pytest tests/factors/v0_1_alpha/test_alpha_edition_manifest.py
```

Roll out the SDK bump with the same release branch as the manifest cut.

---

## 8. If anything goes wrong

- **`ValueError: edition_id ... does not match required format`** —
  The id must be `factors-v0.1.0-alpha-YYYY-MM-DD`. Any deviation aborts
  the cut. Fix the `--edition-id` arg and rerun.
- **`ValueError: methodology_lead_approver must be of the form ...`** —
  Approver must be `human:<email>` or `bot:<email>`. Bare e-mails are
  rejected.
- **`FileNotFoundError: Catalog seed root not found`** — the Wave D #6
  seeds aren't on disk. Re-run the backfill
  (`scripts/factors_alpha_v0_1_backfill.py`) before cutting.
- **Placeholder signature where a real one is expected** — double-check
  `GL_FACTORS_ED25519_PRIVATE_KEY` is exported in the shell that ran the
  cut. Cuts are idempotent: rerun with the env var set.

---

## Appendix — manifest schema (informational)

Top-level fields produced by `build_manifest`:

| Field                             | Description                                                |
|-----------------------------------|------------------------------------------------------------|
| `edition_id`                      | `factors-v0.1.0-alpha-YYYY-MM-DD`                          |
| `schema_id`                       | `https://schemas.greenlang.io/factors/...`                 |
| `schema_sha256`                   | sha256 of the FROZEN v0.1 schema bytes                     |
| `factor_record_v0_1_freeze_sha256`| sha256 of the freeze-note markdown                         |
| `sources[]`                       | per-source metadata + factor counts + parser commits       |
| `factors[]`                       | one entry per record: `urn`, `factor_pack_urn`, sha256     |
| `parser_commits`                  | `{source_id: git-commit-hash}` map                         |
| `methodology_lead_approver`       | actor:email of the lead who signed off                     |
| `methodology_lead_approved_at`    | ISO-8601 UTC timestamp                                     |
| `build_timestamp`                 | ISO-8601 UTC timestamp of the cut                          |
| `builder`                         | always `bot:alpha_edition_manifest`                        |
| `git_commit`                      | repo HEAD at cut time                                      |
| `sdk_version`                     | `0.1.0`                                                    |
| `api_release_profile`             | `alpha-v0.1`                                               |
| `manifest_sha256`                 | sha256 over canonical JSON of every other field            |
