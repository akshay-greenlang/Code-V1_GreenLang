# -*- coding: utf-8 -*-
"""Expand gold eval to 500+ cases - part 2."""
import json
from pathlib import Path

gold_path = Path("tests/factors/fixtures/gold_eval_full.json")
cases = json.loads(gold_path.read_text(encoding="utf-8"))
next_id = len(cases) + 1


def add(activity, fuel, geo=None, domain=None, diff="normal", scope=None):
    global next_id
    c = {"id": f"g{next_id:03d}", "activity": activity, "expected_fuel_type": fuel}
    if geo:
        c["geography"] = geo
    if domain:
        c["domain"] = domain
    if diff != "normal":
        c["difficulty"] = diff
    if scope:
        c["scope"] = scope
    cases.append(c)
    next_id += 1


# Context-rich queries (20)
context = [
    ("our fleet uses diesel fuel for delivery trucks in the US", "diesel", "US", "transport"),
    ("we buy electricity from the local utility company", "electricity", None, "energy"),
    ("the boiler runs on natural gas for space heating", "natural_gas", None, "buildings"),
    ("we burn coal in our power plant", "coal", None, "energy"),
    ("company vehicles run on gasoline", "gasoline", None, "transport"),
    ("backup diesel generators at our data center", "diesel", "US", "industry"),
    ("European office electricity consumption", "electricity", "EU", "buildings"),
    ("UK warehouse heating with gas", "natural_gas", "UK", "buildings"),
    ("diesel-powered construction equipment on site", "diesel", None, "industry"),
    ("electricity for server rooms", "electricity", None, "industry"),
    ("natural gas turbine for CHP", "natural_gas", "US", "energy"),
    ("coal combustion for district heating", "coal", None, "buildings"),
    ("gasoline-powered lawn care", "gasoline", None, "agriculture"),
    ("diesel fuel for agricultural machinery", "diesel", "US", "agriculture"),
    ("purchased grid electricity for manufacturing", "electricity", None, "industry"),
    ("natural gas for food processing", "natural_gas", None, "industry"),
    ("coal boiler in cement factory", "coal", "US", "industry"),
    ("fleet of gasoline cars for sales", "gasoline", "US", "transport"),
    ("diesel buses for employee shuttle", "diesel", None, "transport"),
    ("electricity consumed in retail stores", "electricity", "US", "buildings"),
]
for activity, fuel, geo, domain in context:
    add(activity, fuel, geo, domain)

# GHG protocol terms (15)
ghg = [
    ("Scope 1 stationary combustion diesel", "diesel", "US", "ghg_protocol"),
    ("Scope 1 mobile combustion gasoline", "gasoline", "US", "ghg_protocol"),
    ("Scope 2 location-based electricity", "electricity", "US", "ghg_protocol"),
    ("Scope 2 market-based electricity", "electricity", "EU", "ghg_protocol"),
    ("Scope 1 coal process emissions", "coal", None, "ghg_protocol"),
    ("direct GHG emissions diesel", "diesel", None, "ghg_protocol"),
    ("indirect GHG emissions electricity", "electricity", None, "ghg_protocol"),
    ("energy indirect GHG scope 2", "electricity", None, "ghg_protocol"),
    ("WRI GHG Protocol scope 1 diesel", "diesel", "US", "ghg_protocol"),
    ("ISO 14064 direct coal emissions", "coal", None, "ghg_protocol"),
    ("GRI 305-1 direct gasoline", "gasoline", None, "ghg_protocol"),
    ("GRI 305-2 electricity indirect", "electricity", None, "ghg_protocol"),
    ("CDP diesel consumption report", "diesel", None, "ghg_protocol"),
    ("TCFD climate diesel reporting", "diesel", None, "ghg_protocol"),
    ("SBTi target natural gas", "natural_gas", None, "ghg_protocol"),
]
for activity, fuel, geo, domain in ghg:
    add(activity, fuel, geo, domain)

# Regulatory (10)
reg = [
    ("EPA emission factor diesel", "diesel", "US", "regulatory"),
    ("DEFRA electricity factor UK", "electricity", "UK", "regulatory"),
    ("IPCC default natural gas", "natural_gas", None, "regulatory"),
    ("EU ETS coal factor", "coal", "EU", "regulatory"),
    ("CARB gasoline factor", "gasoline", "US", "regulatory"),
    ("eGRID electricity factor", "electricity", "US", "regulatory"),
    ("DEFRA diesel conversion", "diesel", "UK", "regulatory"),
    ("BEIS gas conversion factor", "natural_gas", "UK", "regulatory"),
    ("EPA AP-42 diesel", "diesel", "US", "regulatory"),
    ("NAEI UK electricity", "electricity", "UK", "regulatory"),
]
for activity, fuel, geo, domain in reg:
    add(activity, fuel, geo, domain)

# Multilingual hints (10)
multi = [
    ("diesel verbrennung", "diesel", "EU", "multilingual"),
    ("strom verbrauch", "electricity", "EU", "multilingual"),
    ("charbon combustion", "coal", "EU", "multilingual"),
    ("essence carburant", "gasoline", "EU", "multilingual"),
    ("gaz naturel chauffage", "natural_gas", "EU", "multilingual"),
    ("gasolio combustione", "diesel", "EU", "multilingual"),
    ("elettricita consumo", "electricity", "EU", "multilingual"),
    ("carbon combustible", "coal", "EU", "multilingual"),
    ("gasolina vehiculo", "gasoline", "EU", "multilingual"),
    ("gas natural calefaccion", "natural_gas", "EU", "multilingual"),
]
for activity, fuel, geo, domain in multi:
    add(activity, fuel, geo, domain, "hard")

# Temporal (10)
temporal = [
    ("winter heating natural gas", "natural_gas", "US", "temporal"),
    ("summer cooling electricity", "electricity", "US", "temporal"),
    ("annual diesel consumption", "diesel", None, "temporal"),
    ("monthly electricity bill", "electricity", None, "temporal"),
    ("quarterly gasoline purchase", "gasoline", "US", "temporal"),
    ("2024 coal consumption", "coal", "US", "temporal"),
    ("FY2025 electricity usage", "electricity", None, "temporal"),
    ("Q4 natural gas heating", "natural_gas", None, "temporal"),
    ("year-round diesel fleet", "diesel", "US", "temporal"),
    ("peak season electricity", "electricity", None, "temporal"),
]
for activity, fuel, geo, domain in temporal:
    add(activity, fuel, geo, domain)

# More edge cases
edges = [
    ("red diesel", "diesel", "UK", "edge"),
    ("white diesel", "diesel", None, "edge"),
    ("heating oil kerosene", "diesel", None, "edge"),
    ("town gas", "natural_gas", "UK", "edge"),
    ("propane LPG gas", "natural_gas", None, "edge"),
    ("bunker fuel marine", "diesel", None, "edge"),
    ("jet fuel kerosene", "diesel", None, "edge"),
    ("avgas aviation gasoline", "gasoline", None, "edge"),
    ("brown coal lignite", "coal", None, "edge"),
    ("sub-bituminous coal", "coal", None, "edge"),
    ("woodchip biomass", "coal", None, "edge"),
    ("biogas methane", "natural_gas", None, "edge"),
    ("renewable electricity", "electricity", None, "edge"),
    ("green electricity tariff", "electricity", "UK", "edge"),
    ("off-peak electricity", "electricity", None, "edge"),
]
for activity, fuel, geo, domain in edges:
    add(activity, fuel, geo, domain, "hard")

print(f"Total cases: {len(cases)}")
gold_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
print("Written successfully")
