# Cookbook: Financed Emissions (PCAF Scope 3 Category 15)

Banks, asset managers, pension funds, and insurers must report **financed emissions** — the share of their customers' emissions attributable to their financing. PCAF (Partnership for Carbon Accounting Financials) publishes the Global GHG Accounting & Reporting Standard with **six asset-class methodologies** plus sovereign debt.

**Method profile:** `finance_proxy`.
**Asset class routing:** `extras.asset_class`.
**Implementation:** `greenlang/factors/method_packs/finance_proxy.py`.

---

## Asset classes supported

| `asset_class` | PCAF methodology | Typical input | Emissions attribution |
|---|---|---|---|
| `listed_equity` | PCAF 5.1 | Market cap, equity stake | `stake / EVIC * investee_emissions` |
| `corporate_bonds` | PCAF 5.2 | Face value, EVIC | `holding / EVIC * investee_emissions` |
| `corporate_loans` | PCAF 5.3 | Outstanding balance, EVIC | Same ratio |
| `project_finance` | PCAF 5.4 | Loan amount, project total CapEx | `loan / total_project_value * project_emissions` |
| `commercial_real_estate` | PCAF 5.5 | Outstanding balance, property value | Per-sqm emission factor by asset class + region |
| `mortgages` | PCAF 5.6 | Outstanding balance, property value | Per-sqm residential factor |
| `motor_vehicle_loans` | PCAF 5.7 | Outstanding balance, vehicle category | Per-km factor x annual km assumption |
| `sovereign_debt` | PCAF 5.8 | Bond face value | Country-level GHG intensity (production + consumption) |

---

## Scenario

A pension fund with three exposures in its 2026 reporting year:

1. **Listed equity:** EUR 5,000,000 in ABC AG (German industrial, EVIC 8,000 MEUR, reported Scope 1+2 of 320,000 tCO2e).
2. **Corporate loan:** EUR 2,000,000 outstanding to XYZ SpA (Italian mid-cap industrial, EVIC 450 MEUR, estimated emissions).
3. **Commercial real estate:** EUR 12,000,000 loan on a 15,000 sqm Class-A office in Amsterdam.

Reporting date: 2026-12-31.

---

## Step 1: Listed equity (best case — investee discloses)

ABC AG publishes verified Scope 1+2 emissions. PCAF data quality score (DQS) 1 (best). Use primary data:

```python
from greenlang.factors.sdk.python import FactorsClient

client = FactorsClient(
    base_url="https://api.greenlang.io",
    api_key="gl_pk_...",
    default_edition="2027.Q1-finance",
)

# Simple attribution — direct math, no catalog factor needed.
stake_eur = 5_000_000
evic_eur = 8_000_000_000
investee_tco2e = 320_000

attributed = (stake_eur / evic_eur) * investee_tco2e
# = 200 tCO2e
```

But you should still log the attribution via the method pack so the evidence bundle is signed and edition-pinned:

```python
resolved = client.resolve_explain({
    "activity": "financed_emissions",
    "method_profile": "finance_proxy",
    "jurisdiction": "DE",
    "reporting_date": "2026-12-31",
    "extras": {
        "asset_class": "listed_equity",
        "investee": {
            "name": "ABC AG",
            "lei": "529900ABCDEFGHIJ1234",
            "evic_eur": 8_000_000_000,
            "disclosed_scope12_tco2e": 320_000,
            "scope12_data_quality": 1,
            "source": "ABC AG Annual Report 2026, pp.82-85, verified by Ernst & Young"
        },
        "exposure_eur": 5_000_000,
    }
})

# The response returns the attributed emissions AND the PCAF data-quality score.
print(f"Attributed: {resolved.co2e_per_unit:,.1f} tCO2e  (PCAF DQS {resolved.dqs.overall_score})")
```

---

## Step 2: Corporate loan (secondary data — PCAF proxy)

XYZ SpA does not disclose emissions. The pack falls through to a sector-average proxy:

```python
resolved = client.resolve_explain({
    "activity": "financed_emissions",
    "method_profile": "finance_proxy",
    "jurisdiction": "IT",
    "reporting_date": "2026-12-31",
    "extras": {
        "asset_class": "corporate_loans",
        "investee": {
            "name": "XYZ SpA",
            "lei": "529900XYZABCDEFG5678",
            "evic_eur": 450_000_000,
            "sector_nace": "C24.10",       # manufacture of basic iron and steel
            "revenue_eur": 620_000_000
        },
        "exposure_eur": 2_000_000
    }
})

print(f"Attributed: {resolved.co2e_per_unit:,.1f} tCO2e")
print(f"PCAF DQS:   {resolved.dqs.overall_score}")
print(f"Method:     {resolved.source.methodology}")
```

The pack uses a PCAF-aligned **economic activity** proxy: sector-mean tCO2e per MEUR revenue, scaled by the investee's revenue and then attributed by `exposure / EVIC`. PCAF DQS 4 or 5 for this route (lowest quality). Invest in better disclosure to improve.

---

## Step 3: Commercial real estate

Route by asset class + sub-type + jurisdiction:

```python
resolved = client.resolve_explain({
    "activity": "financed_emissions",
    "method_profile": "finance_proxy",
    "jurisdiction": "NL",
    "reporting_date": "2026-12-31",
    "extras": {
        "asset_class": "commercial_real_estate",
        "property": {
            "property_type": "office_class_a",
            "area_sqm": 15_000,
            "property_value_eur": 65_000_000,
            "energy_performance_cert": "A+",
            "actual_energy_kwh_per_sqm_year": 95
        },
        "exposure_eur": 12_000_000
    }
})

# Pack prefers actual metered energy (PCAF DQS 1-2). Falls back to
# EPC-derived intensity (DQS 2-3), then to sector-average by country
# (DQS 4-5) if neither is present.
print(resolved.co2e_per_unit, "tCO2e")
```

---

## Step 4: Aggregate into a portfolio view

```python
exposures = [
    # (asset_class, request, exposure_eur)
    ("listed_equity",          req_equity,   5_000_000),
    ("corporate_loans",        req_loan,     2_000_000),
    ("commercial_real_estate", req_cre,     12_000_000),
]

portfolio_tco2e = 0.0
portfolio_eur = sum(e for (_, _, e) in exposures)

for asset_class, req, eur in exposures:
    r = client.resolve_explain(req)
    portfolio_tco2e += r.co2e_per_unit   # already attributed

print(f"Portfolio financed emissions: {portfolio_tco2e:,.1f} tCO2e")
print(f"Portfolio EUR: {portfolio_eur:,}")
print(f"Intensity: {portfolio_tco2e / (portfolio_eur/1_000_000):,.2f} tCO2e / MEUR")
```

---

## PCAF data quality hierarchy

PCAF scores each attribution on a 1-5 scale (1 = best). The pack surfaces this on `resolved.dqs.overall_score` (mapped to the GreenLang 0-100 FQS) and on `extras.pcaf_dqs` (the raw 1-5 score).

| PCAF score | Data used |
|---|---|
| 1 | Audited / verified disclosed emissions. |
| 2 | Disclosed emissions (not verified). |
| 3 | Physical-activity-based estimate (e.g. kWh x grid factor). |
| 4 | Economic-activity-based estimate (revenue x sector intensity). |
| 5 | Sector-average proxy without revenue input. |

Your regulator wants `>= 60%` of exposures to be PCAF 3 or better by 2027 under the EU CRD VI / CSRD expectations.

---

## Portfolio coverage requirements

PCAF asks you to publish:

- Total portfolio AUM/exposure in scope.
- Percentage covered by financed-emissions data.
- Weighted-average PCAF data quality score.
- Absolute financed emissions (tCO2e).
- Emissions intensity (tCO2e / MEUR).

The impact simulation endpoint (`POST /api/v1/factors/impact-simulation/batch`) is useful for running "what-if we upgraded the PCAF quality of the bottom 10% exposures" analyses against your current edition pin.

---

## Common pitfalls

- **EVIC miscalculation:** EVIC (Enterprise Value Including Cash) must include market cap + total debt + preferred equity + minority interest + cash. Omitting cash is the single most common error. Under PCAF 5.1 you use EVIC, not enterprise value (EV).
- **Over-reporting through double counting:** The same investee covered by both a loan and an equity stake should have each exposure attributed separately; do not sum the investee's emissions and attribute once.
- **Proxy for disclosed emitters:** If the investee discloses primary-data, you must use it. Falling back to a proxy when primary is available is a methodology violation.
- **Sovereign double counting:** Sovereign debt uses consumption-based OR production-based emissions (not both). PCAF 5.8 prefers consumption; the pack defaults to consumption with `extras.sovereign_basis` override available.

---

## See also

- [Method packs — finance_proxy](../concepts/method-packs.md)
- [Resolution cascade](../concepts/resolution-cascade.md)
- [Quality scores](../concepts/quality-scores.md) — PCAF DQS 1-5 to GreenLang 0-100 mapping.
- [Version pinning](../concepts/version-pinning.md) — critical for portfolio restatements.

---

## File citations

| Piece | File |
|---|---|
| PCAF method pack | `greenlang/factors/method_packs/finance_proxy.py` |
| Profile enum | `greenlang/data/canonical_v2.py::MethodProfile.FINANCE_PROXY` |
| Impact simulation endpoints | `greenlang/integration/api/routes/factors.py` (`/impact-simulation`, line 1606) |
| Batch submission | `greenlang/integration/api/routes/factors.py` (`/batch/submit`, line 1791) |
