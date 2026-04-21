# Resolution (7-Step Cascade)

The `resolve` / `resolve_explain` endpoints run a deterministic 7-step cascade to pick the best emission factor for a given activity. The SDK surfaces both the chosen factor and the list of alternates considered so consultants and auditors can justify the selection.

## The 7 steps

| Rank | Step label | Source priority |
|-----:|----------------------------------|-----------------|
| 1 | `customer_override` | tenant overlay (see `04_tenant_override.py`) |
| 2 | `supplier_specific` | supplier-matched factor |
| 3 | `facility_specific` | facility-matched factor |
| 4 | `utility_or_grid_region` | residual mix for the region |
| 5 | `country_or_sector_average` | national/sector average |
| 6 | `continental_or_proxy` | closest proxy region |
| 7 | `global_default` | global default of last resort |

`ResolvedFactor.fallback_rank` records which step fired.

## Minimal request

```python
from greenlang.factors.sdk.python import FactorsClient
from greenlang.factors.sdk.python.models import ResolutionRequest

req = ResolutionRequest(
    activity="diesel combustion stationary",
    method_profile="corporate_scope1",
    jurisdiction="US",
    reporting_date="2026-06-01",
)

with FactorsClient(base_url="...", api_key="...") as c:
    resolved = c.resolve(req, alternates=5)
```

## Rich request fields

```python
req = ResolutionRequest(
    activity="electricity purchased",
    method_profile="corporate_scope2_location_based",
    jurisdiction="US-CA",
    reporting_date="2026-06-01",
    supplier_id="PG&E",
    facility_id="fac_san_jose_hq",
    utility_or_grid_region="CAISO",
    preferred_sources=["EPA", "EIA"],
    extras={"contract_type": "bundled", "recs_retired": False},
)
```

Unknown fields are rejected (`extra="forbid"`) so typos fail loudly before the round-trip.

## Explain by factor id

```python
payload = c.resolve_explain("EF:US:diesel:2024:v1", alternates=5, method_profile="corporate_scope1")
print(payload.step_label, payload.fallback_rank, payload.why_chosen)
for gas, qty in (payload.gas_breakdown or {}).items():
    print(gas, qty)
```

## Alternates only

```python
alts = c.alternates("EF:US:diesel:2024:v1", limit=10)
for a in alts["alternates"]:
    print(a["factor_id"], a.get("score"))
```

## Batch resolution

```python
batch = [req1, req2, req3]
handle = c.resolve_batch(batch)
final = c.wait_for_batch(handle, poll_interval=2.0, timeout=300.0)

for row in final.results or []:
    print(row["chosen_factor_id"], row["fallback_rank"])
```

Async callers use the same API via `AsyncFactorsClient.resolve_batch` + `await c.wait_for_batch(...)`.

## What is always in the explain payload

The server guarantees (CTO non-negotiables):

- `chosen_factor_id` + `factor_version`
- `fallback_rank` (1..7) + `step_label` + `why_chosen`
- `alternates` (top-N, configurable, max 20)
- `quality_score` + `uncertainty`
- **Separate** gas components: CO2, CH4, N2O, HFCs, PFCs, SF6, NF3, biogenic_CO2 (never rolled into CO2e)
- `co2e_basis` (which GWP set was applied)
- `assumptions` (list of text assumption strings)
- `deprecation_status` + `deprecation_replacement`

Response headers:
- `X-GreenLang-Edition: <edition_id>`
- `X-GreenLang-Method-Profile: <method_profile>`

The SDK surfaces edition + request-id via `TransportResponse` if you need them for audit logs.
