# GreenLang Factors factors-v0.1.0-alpha-2026-04-25

**Build timestamp:** 2026-04-25T07:56:04Z
**SDK version:** 0.1.0
**API release profile:** alpha-v0.1
**Schema:** [https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json](https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json)
**Schema sha256:** `bf48b596fe031b478e82965868edab849d04bd369c56468c5bf59088e19b2fe7`
**Freeze-note sha256:** `a9af9119899caa5987a2f87994d837c1f8a023484fddbe7fdc5b335c5abf70a1`
**Manifest sha256:** `d6694c8e0060cb070427a5a835f6e0cf3bcf64c14b74d2e88ca94fb10e7ff590`
**Git commit:** `72154a93917d98c30f8689fe6ad0814611736426`

This release manifest catalogs **691** v0.1-alpha factor records
across **6** upstream sources. Per CTO doc §6.3 every
factor belongs to exactly one pack and carries a per-record content hash;
per CTO doc §19.1 the climate-methodology lead has signed off that the URN
scheme covers tier-1 sources without compromise.

## Per-source counts

| Source | Version | Records | Licence |
|---|---|---:|---|
| `cbam_default_values` | 2024.1 | 60 | EU-Publication |
| `desnz_ghg_conversion` | 2024.1 | 195 | OGL-UK-v3 |
| `egrid` | 2022.1 | 79 | US-Public-Domain |
| `epa_hub` | 2024.1 | 84 | US-Public-Domain |
| `india_cea_co2_baseline` | 20.0 | 38 | Government of India — public use with attribution |
| `ipcc_2006_nggi` | 2019.1 | 235 | IPCC-Guideline |

**Total factors:** 691

## Top supersede chains

Wave E v0.1: the supersede graph is empty for the alpha cut — every record is
its own root. Subsequent cuts will list the top 5 chains here automatically
(e.g. eGRID 2022 -> 2024 once the 2024 release lands).

## Known caveats

- See `docs/factors/v0_1_alpha/SOURCE-VINTAGE-AUDIT.md` for the per-source
  vintage / publication-window audit (preview entries flag any source whose
  upstream publication date is older than the 18-month freshness floor).
- See `docs/factors/v0_1_alpha/methodology-exceptions/` for any per-source
  exceptions the methodology lead has explicitly accepted.
- Supersede chains are only computed when a higher-vintage source replaces
  an earlier-vintage one; alpha cuts ship without that wiring.

## Methodology-lead approval

Approved by `human:methodology-lead@greenlang.io` at
`2026-04-25T07:56:04Z`.

## Verification

```python
from pathlib import Path
from greenlang.factors.release.alpha_edition_manifest import verify_manifest

# `public_key` is either the Ed25519 raw 32-byte public key OR a PEM-encoded
# Ed25519 public key (bytes). Distribute alongside the release-notes URL.
ok = verify_manifest(
    manifest_path=Path("releases/factors-v0.1.0-alpha-2026-04-25/manifest.json"),
    signature_path=Path("releases/factors-v0.1.0-alpha-2026-04-25/manifest.json.sig"),
    public_key=Path("path/to/methodology_lead_pubkey.pem").read_bytes(),
)
assert ok, "Edition signature did not verify against the published key."
```

If the file ends in `.sig.placeholder` the cut is **unsigned** — the
methodology-lead Ed25519 key was not available at cut time. The cut is still
content-verifiable (every factor's `record_sha256` is reproducible and the
top-level `manifest_sha256` chains them all), but downstream SDK pins should
treat the edition as **unattested** until a signed re-cut lands.
