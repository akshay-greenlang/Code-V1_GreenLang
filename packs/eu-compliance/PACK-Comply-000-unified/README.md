# PACK-Comply-000-unified

The **FY27 Comply meta-pack.**  It does not ship agents of its own;
instead, it composes the CSRD, CBAM, Scope 1/2/3, and supporting packs
under the `greenlang.comply.ComplyOrchestrator`, which chains the v3
substrate (Policy Graph → Scope Engine → Evidence Vault → Climate
Ledger) into a single run.

## Usage

```bash
gl comply run request.json --bundle out/evidence.zip --output out/result.json
```

## What's inside

This pack is a manifest only (`pack.yaml`).  The actual behaviour is in
two places:

- `greenlang/comply/orchestrator.py` — the orchestrator implementation
- `greenlang/cli/cmd_comply.py` — the `gl comply` CLI surface

## Composed packs (FY27-active)

| Pack | Role |
|---|---|
| PACK-001-csrd-starter, PACK-002, PACK-003 | CSRD tier lineup |
| PACK-012, PACK-013, PACK-014 | Sector-specific CSRD |
| PACK-015-double-materiality, PACK-016-esrs-e1-climate, PACK-017-esrs-full-coverage | ESRS coverage |
| PACK-004-cbam-readiness, PACK-005-cbam-complete | CBAM flagship wedge |
| PACK-041-scope-1-2-complete, PACK-042-scope-3-starter, PACK-043-scope-3-complete | Scope Engine source |

## Related

- Battlecard: `docs/sales/COMPLY_BATTLECARD.md` (Phase 3.4)
- Pricing: `docs/sales/COMPLY_PRICING.md` (Phase 3.4)
- Repo tour: `docs/REPO_TOUR.md`
