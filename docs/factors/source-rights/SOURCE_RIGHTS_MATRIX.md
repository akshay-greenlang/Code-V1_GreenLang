# GreenLang Factors — Source Rights Matrix

| Field            | Value                                                                       |
| ---------------- | --------------------------------------------------------------------------- |
| Status           | Phase 1 baseline (56 sources classified)                                    |
| Date             | 2026-04-26                                                                  |
| Owner            | Compliance/Security Lead (R) / CTO (A) / Climate Methodology Lead (C)       |
| Source-of-truth  | `greenlang/factors/data/source_registry.yaml` (this matrix is the human-readable mirror) |
| Schema           | `config/schemas/source_registry_v0_1.schema.json`                           |
| Enforcement      | `greenlang/factors/rights/service.py` (SourceRightsService)                 |
| Tests            | `tests/factors/v0_1_alpha/test_source_rights_*.py`                          |

This matrix mirrors `source_registry.yaml`. **The YAML is the source
of truth** — this document exists for human review (Compliance,
Methodology, partner conversations). Any drift between this file
and the YAML is a bug; CI verifies the count matches.

## Legend

### Licence class (per CTO Phase 1)

| Value                  | Meaning                                                                     |
| ---------------------- | --------------------------------------------------------------------------- |
| `community_open`       | Free / public / open-government source. Redistribute with attribution.      |
| `method_only`          | Method or framework text public; values restricted (PCAF, GHGP, GLEC).      |
| `commercial_licensed`  | Requires commercial licence (Ecoinvent, IEA, CEDA, NIES, EXIOBASE).         |
| `private_tenant_scoped`| Tenant-uploaded or tenant-private factors.                                  |
| `connector_only`       | Served via API/connector only; no bulk redistribution (Electricity Maps).   |
| `blocked`              | Cannot be ingested or served until reclassified.                            |

### Redistribution class

| Value                       | Meaning                                                                     |
| --------------------------- | --------------------------------------------------------------------------- |
| `redistribution_allowed`    | Values + metadata + bulk download allowed.                                  |
| `attribution_required`      | Redistribution allowed; caller MUST surface attribution.                    |
| `metadata_only`             | List + describe but do NOT return values.                                   |
| `derived_values_only`       | Return our normalized derived values, not raw upstream values.              |
| `tenant_entitled_only`      | Return only to tenants with active entitlement record.                      |
| `no_redistribution`         | Never return values; only describe existence.                               |
| `blocked`                   | Never return at all.                                                        |

### Entitlement model

| Value                          | Meaning                                                                     |
| ------------------------------ | --------------------------------------------------------------------------- |
| `public_no_entitlement`        | No check (community_open).                                                  |
| `tenant_entitlement_required`  | Tenant must have active EntitlementRecord.                                  |
| `private_tenant_owner_only`    | Only the owning tenant may read.                                            |
| `connector_only_no_bulk`       | Served via connector path only; bulk download blocked.                      |
| `blocked`                      | Deny everything.                                                            |

### Legal signoff status

| Value                  | Meaning                                                                     |
| ---------------------- | --------------------------------------------------------------------------- |
| `pending_legal_review` | Not yet reviewed by Compliance/Security Lead. Cannot ship to production.    |
| `in_review`            | Counsel has the file; status awaiting countersign.                          |
| `approved`             | Approved; reviewer + timestamp + evidence URI on record.                    |
| `rejected`             | Counsel has declined. Source is moved to `blocked`.                         |
| `expired`              | Signoff expired (annual licence renewal). Treat as `pending_legal_review`.  |

## Matrix

### v0.1 Alpha (FY27 Q1) — 6 sources, ALL approved

| Release | source_id                  | licence_class    | redistribution_class    | entitlement_model              | legal_signoff |
| ------- | -------------------------- | ---------------- | ----------------------- | ------------------------------ | ------------- |
| v0.1    | `cbam_default_values`      | community_open   | redistribution_allowed  | public_no_entitlement          | approved      |
| v0.1    | `desnz_ghg_conversion`     | community_open   | attribution_required    | public_no_entitlement          | approved      |
| v0.1    | `egrid`                    | community_open   | attribution_required    | public_no_entitlement          | approved      |
| v0.1    | `epa_hub`                  | community_open   | attribution_required    | public_no_entitlement          | approved      |
| v0.1    | `india_cea_co2_baseline`   | community_open   | redistribution_allowed  | public_no_entitlement          | approved      |
| v0.1    | `ipcc_2006_nggi`           | community_open   | redistribution_allowed  | public_no_entitlement          | approved      |

Per-source legal note: see `docs/factors/source-rights/legal-notes/<source_id>.md`.

### v0.5 Closed Beta (FY27 Q2) — 4 sources

| Release | source_id                  | licence_class       | redistribution_class    | entitlement_model              | legal_signoff |
| ------- | -------------------------- | ------------------- | ----------------------- | ------------------------------ | ------------- |
| v0.5    | `defra_wtt`                | community_open      | attribution_required    | public_no_entitlement          | pending       |
| v0.5    | `iea`                      | connector_only      | metadata_only           | connector_only_no_bulk         | pending       |
| v0.5    | `iea_emission_factors`     | commercial_licensed | tenant_entitled_only    | tenant_entitlement_required    | pending       |
| v0.5    | `india_bee_pat`            | community_open      | attribution_required    | public_no_entitlement          | pending       |

### v0.9 Public Beta (FY27 Q3) — 2 sources

| Release | source_id                  | licence_class    | redistribution_class    | entitlement_model              | legal_signoff |
| ------- | -------------------------- | ---------------- | ----------------------- | ------------------------------ | ------------- |
| v0.9    | `edgar`                    | community_open   | attribution_required    | public_no_entitlement          | pending       |
| v0.9    | `unfccc_nir_bur_btr`       | community_open   | attribution_required    | public_no_entitlement          | pending       |

### v1.0 GA (FY27 Q4) — 24 sources

(default-bucket for sources not yet promoted to a higher milestone)

| Release | source_id                              | licence_class       | redistribution_class    | entitlement_model              | legal_signoff |
| ------- | -------------------------------------- | ------------------- | ----------------------- | ------------------------------ | ------------- |
| v1.0    | `aib_residual_mix_eu`                  | community_open      | redistribution_allowed  | public_no_entitlement          | pending       |
| v1.0    | `australia_nga_factors`                | community_open      | redistribution_allowed  | public_no_entitlement          | pending       |
| v1.0    | `beis_uk_residual`                     | community_open      | redistribution_allowed  | public_no_entitlement          | pending       |
| v1.0    | `ceda_pbe`                             | connector_only      | tenant_entitled_only    | connector_only_no_bulk         | pending       |
| v1.0    | `cer_canada_residual`                  | community_open      | redistribution_allowed  | public_no_entitlement          | pending       |
| v1.0    | `defra_conversion`                     | community_open      | attribution_required    | public_no_entitlement          | pending       |
| v1.0    | `defra_uk_env_accounts`                | community_open      | redistribution_allowed  | public_no_entitlement          | pending       |
| v1.0    | `ec3_buildings_epd`                    | connector_only      | tenant_entitled_only    | connector_only_no_bulk         | pending       |
| v1.0    | `eea_waste_stats`                      | community_open      | redistribution_allowed  | public_no_entitlement          | pending       |
| v1.0    | `ef_3_1_secondary`                     | community_open      | redistribution_allowed  | public_no_entitlement          | pending       |
| v1.0    | `ema_singapore_residual`               | community_open      | redistribution_allowed  | public_no_entitlement          | pending       |
| v1.0    | `eu_cbam`                              | community_open      | attribution_required    | public_no_entitlement          | pending       |
| v1.0    | `ghgp_method_refs`                     | method_only         | metadata_only           | public_no_entitlement          | approved for v0.1 method-reference metadata |
| v1.0    | `green_e_residual`                     | connector_only      | metadata_only           | connector_only_no_bulk         | pending       |
| v1.0    | `green_e_residual_mix`                 | commercial_licensed | tenant_entitled_only    | tenant_entitlement_required    | pending       |
| v1.0    | `greenlang_builtin`                    | community_open      | attribution_required    | public_no_entitlement          | pending       |
| v1.0    | `india_ccts_baselines`                 | community_open      | redistribution_allowed  | public_no_entitlement          | pending       |
| v1.0    | `ipcc_2006_afolu_v2019`                | community_open      | redistribution_allowed  | public_no_entitlement          | pending       |
| v1.0    | `ipcc_waste_vol5_in`                   | community_open      | redistribution_allowed  | public_no_entitlement          | pending       |
| v1.0    | `japan_meti_electric_emission_factors` | community_open      | redistribution_allowed  | public_no_entitlement          | pending       |
| v1.0    | `kemco_korea_residual`                 | community_open      | redistribution_allowed  | public_no_entitlement          | pending       |
| v1.0    | `nger_au_state_residual`               | community_open      | redistribution_allowed  | public_no_entitlement          | pending       |
| v1.0    | `tcr_grp_defaults`                     | method_only         | metadata_only           | public_no_entitlement          | pending       |
| v1.0    | `us_epa_suseeio`                       | community_open      | redistribution_allowed  | public_no_entitlement          | pending       |

### v1.5 (FY28 Q3) — 7 sources

| Release | source_id                  | licence_class    | redistribution_class    | entitlement_model              | legal_signoff |
| ------- | -------------------------- | ---------------- | ----------------------- | ------------------------------ | ------------- |
| v1.5    | `ademe_base_carbone`       | community_open   | attribution_required    | public_no_entitlement          | pending       |
| v1.5    | `ashrae_ahri`              | method_only      | metadata_only           | public_no_entitlement          | pending       |
| v1.5    | `climate_trace`            | community_open   | attribution_required    | public_no_entitlement          | pending       |
| v1.5    | `community_contributions`  | community_open   | attribution_required    | public_no_entitlement          | pending       |
| v1.5    | `glec_framework`           | method_only      | tenant_entitled_only    | public_no_entitlement          | pending       |
| v1.5    | `iso_14083`                | method_only      | metadata_only           | public_no_entitlement          | pending       |
| v1.5    | `pact_pathfinder`          | method_only      | tenant_entitled_only    | public_no_entitlement          | pending       |

### v2.0 (FY29 Q2) — 5 sources

| Release | source_id              | licence_class       | redistribution_class    | entitlement_model              | legal_signoff |
| ------- | ---------------------- | ------------------- | ----------------------- | ------------------------------ | ------------- |
| v2.0    | `ecoinvent`            | connector_only      | metadata_only           | connector_only_no_bulk         | pending       |
| v2.0    | `exiobase_v3`          | commercial_licensed | tenant_entitled_only    | tenant_entitlement_required    | pending       |
| v2.0    | `nies_japan`           | commercial_licensed | tenant_entitled_only    | tenant_entitlement_required    | pending       |
| v2.0    | `pcaf_global_std_v2`   | method_only         | tenant_entitled_only    | public_no_entitlement          | pending       |
| v2.0    | `wri_aqueduct`         | community_open      | attribution_required    | public_no_entitlement          | pending       |

### v2.5 (FY30 Q2) — 8 sources

| Release | source_id              | licence_class    | redistribution_class    | entitlement_model              | legal_signoff |
| ------- | ---------------------- | ---------------- | ----------------------- | ------------------------------ | ------------- |
| v2.5    | `agribalyse`           | community_open   | attribution_required    | public_no_entitlement          | pending       |
| v2.5    | `electricity_maps`     | connector_only   | metadata_only           | connector_only_no_bulk         | pending       |
| v2.5    | `entsoe_realtime`      | connector_only   | metadata_only           | connector_only_no_bulk         | pending       |
| v2.5    | `faostat`              | community_open   | attribution_required    | public_no_entitlement          | pending       |
| v2.5    | `grid_india_realtime`  | connector_only   | metadata_only           | connector_only_no_bulk         | pending       |
| v2.5    | `nies_idea`            | commercial_licensed | tenant_entitled_only | tenant_entitlement_required    | pending       |
| v2.5    | `us_iso_rto`           | connector_only   | metadata_only           | connector_only_no_bulk         | pending       |
| v2.5    | `watttime`             | connector_only   | metadata_only           | connector_only_no_bulk         | pending       |

## Enforcement Summary

| Layer                  | Gate function                                      | Behavior                                                                     |
| ---------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| Ingestion              | `check_ingestion_allowed(source_urn)`              | Reject `blocked` and `pending_legal_review` sources from production publish. |
| Query (factor read)    | `check_factor_read_allowed(tenant, source_urn)`    | Filter records the tenant cannot see; emit audit event for licensed access.  |
| Pack download          | `check_pack_download_allowed(tenant, pack_urn)`    | Block downloads when the pack's source is `tenant_entitled_only` without entitlement. |
| Audit                  | `audit_licensed_access(tenant, source_urn, ..., decision)` | One audit event per access to a non-`community_open` source.            |

## Open Items (tracked outside this file)

* Compliance/Security Lead countersign for the 6 v0.1 approvals,
  GHGP metadata approval, and methodology-lead countersign for the 3
  accepted alpha source-vintage exceptions (currently CTO-delegated).
* Legal review of all 49 `pending_legal_review` sources as their
  release milestone approaches.
* Commercial licence procurement for `iea_emission_factors`,
  `ecoinvent`, `exiobase_v3`, `nies_japan`, `green_e_residual_mix`,
  `ceda_pbe`, `nies_idea` before their release windows.
* Connector contracts for `electricity_maps`, `entsoe_realtime`,
  `watttime`, `us_iso_rto`, `grid_india_realtime`.
