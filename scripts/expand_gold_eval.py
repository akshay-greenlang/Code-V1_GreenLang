# -*- coding: utf-8 -*-
"""Expand gold eval set to 500+ cases for F044."""
import json
from pathlib import Path

gold_path = Path("tests/factors/fixtures/gold_eval_full.json")
cases = json.loads(gold_path.read_text(encoding="utf-8"))
next_id = 106


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


# Domain: energy - diesel variations
for g in ["US", "EU", "UK", None]:
    add("diesel fuel combustion", "diesel", g, "energy")
    add("diesel oil stationary", "diesel", g, "energy")
    add("diesel generator power", "diesel", g, "energy")
    add("#2 fuel oil combustion", "diesel", g, "energy")
    add("DERV diesel road vehicle", "diesel", g, "energy")
    add("gas oil heating", "diesel", g, "energy")
    add("diesel backup generator", "diesel", g, "energy")

# Domain: energy - natural gas variations
for g in ["US", "EU", "UK", None]:
    add("natural gas heating", "natural_gas", g, "energy")
    add("methane combustion boiler", "natural_gas", g, "energy")
    add("pipeline gas furnace", "natural_gas", g, "energy")
    add("natural gas therms", "natural_gas", g, "energy")
    add("gas fired boiler", "natural_gas", g, "energy")
    add("CNG compressed natural gas", "natural_gas", g, "energy")
    add("LNG liquefied natural gas", "natural_gas", g, "energy")

# Domain: energy - electricity
for g in ["US", "EU", "UK", None]:
    add("electricity grid consumption", "electricity", g, "energy", scope="2")
    add("purchased electricity", "electricity", g, "energy", scope="2")
    add("grid power", "electricity", g, "energy", scope="2")
    add("electrical power consumption", "electricity", g, "energy", scope="2")
    add("mains electricity supply", "electricity", g, "energy", scope="2")
    add("electric utility", "electricity", g, "energy", scope="2")
    add("kWh electricity purchased", "electricity", g, "energy", scope="2")

# Domain: energy - coal
for g in ["US", None]:
    add("bituminous coal combustion", "coal", g, "energy")
    add("coal fired boiler", "coal", g, "energy")
    add("anthracite coal burning", "coal", g, "energy")
    add("hard coal combustion", "coal", g, "energy")
    add("thermal coal stationary", "coal", g, "energy")
    add("coal power generation", "coal", g, "energy")
    add("solid fossil fuel coal", "coal", g, "energy")
    add("steam coal boiler", "coal", g, "energy")
    add("coal combustion tons", "coal", g, "energy")
    add("lignite brown coal", "coal", g, "energy")

# Domain: energy - gasoline
for g in ["US", None]:
    add("gasoline vehicle fleet", "gasoline", g, "energy")
    add("petrol car transport", "gasoline", g, "energy")
    add("motor gasoline", "gasoline", g, "energy")
    add("unleaded petrol", "gasoline", g, "energy")
    add("E10 gasoline blend", "gasoline", g, "energy")
    add("premium gasoline fuel", "gasoline", g, "energy")
    add("regular gasoline gallons", "gasoline", g, "energy")
    add("petrol fuel cars", "gasoline", g, "energy")
    add("automobile gasoline", "gasoline", g, "energy")
    add("gasoline spark ignition", "gasoline", g, "energy")

# Domain: transport
for g in ["US", "EU", None]:
    add("diesel truck freight", "diesel", g, "transport")
    add("diesel vehicle road", "diesel", g, "transport")
    add("diesel locomotive rail", "diesel", g, "transport")
    add("marine diesel fuel", "diesel", g, "transport")
    add("diesel bus fleet", "diesel", g, "transport")
    add("gasoline car commuting", "gasoline", g, "transport")
    add("petrol vehicle passenger", "gasoline", g, "transport")
    add("gasoline fleet vehicles", "gasoline", g, "transport")

# Domain: industry
for g in ["US", "EU", None]:
    add("coal furnace industrial", "coal", g, "industry")
    add("natural gas process heat", "natural_gas", g, "industry")
    add("diesel forklift warehouse", "diesel", g, "industry")
    add("industrial electricity consumption", "electricity", g, "industry", scope="2")
    add("factory power usage", "electricity", g, "industry", scope="2")
    add("manufacturing natural gas", "natural_gas", g, "industry")
    add("process steam coal fired", "coal", g, "industry")
    add("industrial diesel generator", "diesel", g, "industry")

# Domain: buildings
for g in ["US", "EU", "UK", None]:
    add("office building electricity", "electricity", g, "buildings", scope="2")
    add("building heating natural gas", "natural_gas", g, "buildings")
    add("commercial HVAC gas", "natural_gas", g, "buildings")
    add("building diesel backup", "diesel", g, "buildings")
    add("office lighting power", "electricity", g, "buildings", scope="2")

# Domain: agriculture
for g in ["US", None]:
    add("farm diesel equipment", "diesel", g, "agriculture")
    add("agricultural diesel tractor", "diesel", g, "agriculture")
    add("farm electricity irrigation", "electricity", g, "agriculture", scope="2")
    add("grain drying natural gas", "natural_gas", g, "agriculture")
    add("livestock heating gas", "natural_gas", g, "agriculture")
    add("farm gasoline ATV", "gasoline", g, "agriculture")
    add("greenhouse heating gas", "natural_gas", g, "agriculture")

# Difficulty: hard - misspellings
hard_misspellings = [
    ("deisel fule", "diesel"), ("disel combust", "diesel"),
    ("electrcity grd", "electricity"), ("elctricity", "electricity"),
    ("natual gass", "natural_gas"), ("natrual gas heatng", "natural_gas"),
    ("gasoline gasolene", "gasoline"), ("petral car", "gasoline"),
    ("cole burning", "coal"), ("coel combustion", "coal"),
    ("diesel fual oil", "diesel"), ("elektricty", "electricity"),
    ("naturel gas", "natural_gas"), ("gasoleen", "gasoline"),
    ("charcoal coal", "coal"), ("diesle generator", "diesel"),
    ("electricty kwh", "electricity"), ("nateral gas therms", "natural_gas"),
    ("gazoline fuel", "gasoline"), ("anthrcite coal", "coal"),
    ("diesel fue combuston", "diesel"), ("elctricity purchsed", "electricity"),
    ("nateral gaz", "natural_gas"), ("gasolin vehicle", "gasoline"),
    ("diesal truck", "diesel"),
]
for activity, fuel in hard_misspellings:
    add(activity, fuel, None, "misspelling", "hard")

# Difficulty: hard - abbreviations
hard_abbrevs = [
    ("NG boiler", "natural_gas"), ("nat gas", "natural_gas"),
    ("elec grid", "electricity"), ("pwr purchased", "electricity"),
    ("dsl generator", "diesel"), ("gas mobile", "gasoline"),
    ("NG therms", "natural_gas"), ("elec kwh", "electricity"),
    ("dsl fuel oil", "diesel"), ("pet gasoline", "gasoline"),
    ("NatGas heating", "natural_gas"), ("NG combustion", "natural_gas"),
    ("dsl combustion stat", "diesel"), ("elec consumption", "electricity"),
    ("HFO heavy fuel oil", "diesel"), ("MGO marine gas oil", "diesel"),
    ("LPG propane", "natural_gas"), ("CNG fleet", "natural_gas"),
    ("E85 fuel blend", "gasoline"), ("bio diesel blend", "diesel"),
]
for activity, fuel in hard_abbrevs:
    add(activity, fuel, None, "abbreviation", "hard")

# Difficulty: hard - ambiguous
hard_ambiguous = [
    ("fuel combustion", "diesel"), ("gas heating", "natural_gas"),
    ("energy consumption", "electricity"), ("fossil fuel burning", "coal"),
    ("liquid fuel", "diesel"), ("solid fuel", "coal"),
    ("heating fuel", "natural_gas"), ("motor fuel", "gasoline"),
    ("power generation", "electricity"), ("fuel oil", "diesel"),
    ("combustion stationary", "natural_gas"), ("fuel transport mobile", "diesel"),
    ("grid energy", "electricity"), ("thermal fuel", "natural_gas"),
    ("liquid petroleum", "diesel"),
]
for activity, fuel in hard_ambiguous:
    add(activity, fuel, None, "ambiguous", "hard")

# Cross-geography
cross_geo = [
    ("American diesel", "diesel", "US"), ("European electricity", "electricity", "EU"),
    ("British gas", "natural_gas", "UK"), ("USA coal", "coal", "US"),
    ("United States gasoline", "gasoline", "US"), ("EU grid electricity", "electricity", "EU"),
    ("UK power grid", "electricity", "UK"), ("US natural gas pipeline", "natural_gas", "US"),
    ("continental Europe electricity", "electricity", "EU"),
    ("England electricity grid", "electricity", "UK"),
    ("North American diesel", "diesel", "US"), ("Western Europe gas", "natural_gas", "EU"),
    ("Britain electricity", "electricity", "UK"), ("CONUS grid power", "electricity", "US"),
    ("Americas diesel fuel", "diesel", "US"), ("Old world electricity", "electricity", "EU"),
    ("Great Britain power", "electricity", "UK"), ("European Union electricity", "electricity", "EU"),
    ("United Kingdom gas", "natural_gas", "UK"), ("US EPA diesel factor", "diesel", "US"),
]
for activity, fuel, geo in cross_geo:
    add(activity, fuel, geo, "cross_geography", "hard")

# Scoped queries
for g in ["US", None]:
    add("scope 1 diesel combustion", "diesel", g, "scoped", scope="1")
    add("scope 2 electricity grid", "electricity", g, "scoped", scope="2")
    add("scope 1 natural gas stationary", "natural_gas", g, "scoped", scope="1")
    add("scope 1 coal boiler", "coal", g, "scoped", scope="1")
    add("scope 1 gasoline mobile", "gasoline", g, "scoped", scope="1")
    add("direct emissions diesel", "diesel", g, "scoped", scope="1")
    add("indirect emissions electricity", "electricity", g, "scoped", scope="2")
    add("Scope 1 stationary combustion gas", "natural_gas", g, "scoped", scope="1")
    add("market-based electricity", "electricity", g, "scoped", scope="2")
    add("location-based grid power", "electricity", g, "scoped", scope="2")

# Unit-specific queries
unit_queries = [
    ("diesel gallons", "diesel", "US"), ("diesel liters", "diesel", "US"),
    ("electricity kwh", "electricity", "US"), ("natural gas therms", "natural_gas", "US"),
    ("coal short tons", "coal", "US"), ("gasoline gallons", "gasoline", "US"),
    ("diesel litres", "diesel", "EU"), ("electricity megawatt hours", "electricity", "EU"),
    ("natural gas cubic meters", "natural_gas", None), ("coal metric tonnes", "coal", None),
    ("diesel barrels", "diesel", None), ("gasoline liters", "gasoline", None),
    ("natural gas mmbtu", "natural_gas", "US"), ("coal btu", "coal", "US"),
    ("electricity mwh", "electricity", None), ("diesel kg", "diesel", None),
    ("natural gas GJ", "natural_gas", None), ("coal GJ energy", "coal", None),
    ("gasoline kg", "gasoline", None), ("electricity GWh", "electricity", None),
]
for activity, fuel, geo in unit_queries:
    add(activity, fuel, geo, "units")

print(f"Total cases: {len(cases)}")
gold_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
print("Written successfully")
