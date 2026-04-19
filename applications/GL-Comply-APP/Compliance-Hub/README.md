# GL-Comply-APP — Unified Compliance Hub

Single API / SDK / CLI for 10 compliance frameworks: CSRD, CBAM, EUDR,
GHG Protocol, ISO 14064, SB 253, SBTi, EU Taxonomy, TCFD, CDP.

## Run locally

```bash
cd applications/GL-Comply-APP/Compliance-Hub
docker-compose up
# -> http://localhost:8080/health
```

## Python SDK

```python
from sdk.client import ComplyClient
client = ComplyClient(base_url="http://localhost:8080/api/v1")
report = client.intake(request)
```

## CLI

```bash
PYTHONPATH=. python -m cli.commands run request.json
PYTHONPATH=. python -m cli.commands applicability entity.json
PYTHONPATH=. python -m cli.commands frameworks
```

## Tests

```bash
cd applications/GL-Comply-APP/Compliance-Hub
python -m pytest tests/ -q
```

## Architecture

```
api/v1.py           -> POST /compliance/intake, /applicability; GET /jobs/{id}, /frameworks
agents/
  orchestrator_agent.py  -> async parallel dispatch, timeout, gap analysis
  unified_report_agent.py-> JSON/PDF/XBRL-lite
services/
  applicability.py  -> jurisdiction + threshold rules for 10 frameworks
  normalizer.py     -> entity + activity dedup
  registry.py       -> FrameworkAdapter protocol + registry
  adapters/         -> 10 adapters (csrd, cbam, eudr, ghg, iso14064, sb253,
                       sbti, taxonomy, tcfd, cdp); all but EUDR+Taxonomy
                       delegate to greenlang.scope_engine
  store.py          -> in-memory job store (swap for PG in prod)
schemas/models.py   -> ComplianceRequest, FrameworkResult, UnifiedComplianceReport
sdk/client.py       -> ComplyClient (sync), AsyncComplyClient
cli/commands.py     -> comply run/status/applicability/frameworks
```

## Provenance

Every compute run produces:
- Per-framework `provenance_hash` (SHA-256 over inputs + framework-specific outputs)
- `aggregate_provenance_hash` over the full unified report
- Ledger entries via `greenlang.climate_ledger` when enabled

## DB schema

`deployment/database/migrations/sql/V438__comply_hub_jobs.sql`
  - comply_jobs
  - comply_framework_results
  - comply_unified_reports
  - comply_applicability_cache

## Build status

| Task | Status |
|------|--------|
| COMPLY-APP 1 Scaffold | DONE |
| COMPLY-APP 2 Framework adapters | DONE (10/10) |
| COMPLY-APP 3 Orchestrator + applicability + normalizer | DONE |
| COMPLY-APP 4 FastAPI + SDK + CLI | DONE |
| COMPLY-APP 5 Unified reporting | DONE |
| COMPLY-APP 6 Migrations + deployment + tests | DONE |
