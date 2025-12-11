#!/usr/bin/env python3
"""
GreenLang 10,400 Agent Catalog Generator
Based on CTO's multi-domain portfolio design with context expansion.

Domain Allocations (per CTO design):
- Industrial & Process Heat: ~2,500 agents
- Power, Grids & Storage: ~1,900 agents
- Buildings, HVAC & Cities: ~1,800 agents
- Transport & Logistics: ~1,500 agents
- Agriculture, Land, Nature & Water: ~1,100 agents
- Carbon Capture, Removal & Utilization: ~400 agents
- Supply Chain, Corporate, Finance & Reg: ~800 agents
- Cross-cutting Data, Risk, Governance: ~300 agents
"""

import csv
from typing import Dict, List, Any
from collections import Counter
from itertools import cycle

# =============================================================================
# DOMAIN CONFIGURATIONS - Balanced for diverse category distribution
# =============================================================================

DOMAINS = {
    "Industrial": {
        "target_agents": 2500,
        "subdomains": {
            "Process Heat": {"carbon_impact": 2.5, "market_base": 2.0},
            "Heat Recovery": {"carbon_impact": 1.8, "market_base": 1.5},
            "Heat Pumps": {"carbon_impact": 2.2, "market_base": 1.8},
            "Process Integration": {"carbon_impact": 1.0, "market_base": 0.8},
            "Future Fuels": {"carbon_impact": 1.5, "market_base": 1.2},
            "Steam & Boilers": {"carbon_impact": 1.6, "market_base": 1.0},
            "Combustion": {"carbon_impact": 1.4, "market_base": 0.9}
        },
        "sectors": ["GenericIndustrial", "Steel", "Cement", "Chemicals", "Refining", "Petrochemicals", "PulpPaper", "FoodBeverage", "Glass", "Mining", "Ceramics"],
        "levels": ["Asset", "Site", "Fleet", "Portfolio"]
    },

    "Energy Systems": {
        "target_agents": 1900,
        "subdomains": {
            "Grid Optimization": {"carbon_impact": 1.5, "market_base": 1.0},
            "Renewable Integration": {"carbon_impact": 2.0, "market_base": 1.5},
            "Energy Storage": {"carbon_impact": 1.0, "market_base": 0.8},
            "Market Operations": {"carbon_impact": 0.5, "market_base": 0.6},
            "Demand Response": {"carbon_impact": 0.8, "market_base": 0.7}
        },
        "sectors": ["UtilityScale", "Commercial", "Industrial", "Residential", "Microgrid", "TSO", "DSO", "IPP"],
        "levels": ["Asset", "Plant", "Substation", "Grid", "Region"]
    },

    "Buildings": {
        "target_agents": 1800,
        "subdomains": {
            "HVAC Core": {"carbon_impact": 0.8, "market_base": 0.8},
            "Building Types": {"carbon_impact": 1.0, "market_base": 1.0},
            "Smart Control": {"carbon_impact": 0.6, "market_base": 0.8},
            "District Energy": {"carbon_impact": 0.5, "market_base": 0.5},
            "Building Envelope": {"carbon_impact": 0.4, "market_base": 0.4}
        },
        "sectors": ["CommercialOffice", "Hospital", "DataCenter", "Retail", "Warehouse", "Hotel", "School", "University", "MixedUse", "Laboratory"],
        "levels": ["Asset", "Site", "Campus", "Portfolio"]
    },

    "Transport": {
        "target_agents": 1500,
        "subdomains": {
            "Electric Vehicles": {"carbon_impact": 1.2, "market_base": 1.5},
            "Charging Infrastructure": {"carbon_impact": 0.8, "market_base": 1.0},
            "Logistics": {"carbon_impact": 1.0, "market_base": 1.2},
            "Shipping": {"carbon_impact": 0.6, "market_base": 0.5},
            "Aviation": {"carbon_impact": 0.4, "market_base": 0.4},
            "Rail": {"carbon_impact": 0.3, "market_base": 0.3}
        },
        "sectors": ["PassengerFleet", "FreightFleet", "LastMile", "LongHaul", "PublicTransit", "Shipping", "Aviation", "Rail"],
        "levels": ["Vehicle", "Fleet", "Network", "Region"]
    },

    "Agriculture": {
        "target_agents": 1100,
        "subdomains": {
            "Precision Agriculture": {"carbon_impact": 0.5, "market_base": 0.5},
            "Irrigation": {"carbon_impact": 0.4, "market_base": 0.4},
            "Livestock": {"carbon_impact": 0.7, "market_base": 0.3},
            "Land & Forestry": {"carbon_impact": 0.6, "market_base": 0.3},
            "Soil Carbon": {"carbon_impact": 0.3, "market_base": 0.2}
        },
        "sectors": ["RowCrops", "Orchards", "Dairy", "Beef", "Poultry", "Forestry", "Pasture", "Aquaculture"],
        "levels": ["Field", "Farm", "Cooperative", "Basin", "Region"]
    },

    "CDR": {
        "target_agents": 400,
        "subdomains": {
            "Point-Source CCS": {"carbon_impact": 0.8, "market_base": 0.6},
            "Engineered CDR": {"carbon_impact": 0.5, "market_base": 0.4},
            "Nature-Based CDR": {"carbon_impact": 0.4, "market_base": 0.3},
            "MRV & Credits": {"carbon_impact": 0.2, "market_base": 0.3}
        },
        "sectors": ["PowerCCS", "IndustrialCCS", "DAC", "BECCS", "Biochar", "ForestryCredits"],
        "levels": ["Project", "Site", "Portfolio", "Region"]
    },

    "Supply Chain & Finance": {
        "target_agents": 800,
        "subdomains": {
            "Supplier Emissions": {"carbon_impact": 1.0, "market_base": 0.8},
            "Product Footprint": {"carbon_impact": 0.6, "market_base": 0.5},
            "Climate Risk": {"carbon_impact": 0.0, "market_base": 1.0},
            "Corporate Reporting": {"carbon_impact": 0.0, "market_base": 0.5},
            "Compliance": {"carbon_impact": 0.0, "market_base": 0.4}
        },
        "sectors": ["Manufacturing", "Retail", "Banks", "Insurance", "AssetManagers", "Corporates", "Utilities"],
        "levels": ["Supplier", "Company", "Portfolio", "Sector", "Enterprise"]
    },

    "Data & Governance": {
        "target_agents": 300,
        "subdomains": {
            "Data Management": {"carbon_impact": 0.0, "market_base": 0.2},
            "Risk & Governance": {"carbon_impact": 0.0, "market_base": 0.2},
            "Culture & Capability": {"carbon_impact": 0.0, "market_base": 0.1}
        },
        "sectors": ["Enterprise", "Platform", "Global"],
        "levels": ["System", "Domain", "Enterprise", "Global"]
    }
}

# Full set of categories
ALL_CATEGORIES = ["Coordinator", "Optimizer", "Monitor", "Controller", "Automator",
                  "Predictor", "Analyzer", "Calculator", "Reporter", "Integrator", "Communicator"]

CATEGORIES_CONFIG = {
    "Coordinator": {"complexity": "High", "priority_boost": 0},
    "Optimizer": {"complexity": "High", "priority_boost": 0},
    "Monitor": {"complexity": "Medium", "priority_boost": 1},
    "Controller": {"complexity": "Medium", "priority_boost": 1},
    "Automator": {"complexity": "Medium", "priority_boost": 1},
    "Predictor": {"complexity": "Medium", "priority_boost": 1},
    "Analyzer": {"complexity": "Medium", "priority_boost": 1},
    "Calculator": {"complexity": "Low", "priority_boost": 2},
    "Reporter": {"complexity": "Low", "priority_boost": 2},
    "Integrator": {"complexity": "High", "priority_boost": 0},
    "Communicator": {"complexity": "Low", "priority_boost": 2}
}

REGIONS = ["Global", "US", "EU", "UK", "China", "India", "SEAsia", "MENA", "LatAm", "Africa", "Japan", "Canada", "Nordics", "Australia", "Brazil"]

TIMELINES = {
    "P0": ["Q4 2025", "Q1 2026"],
    "P1": ["Q2 2026", "Q3 2026"],
    "P2": ["Q4 2026", "Q1 2027"]
}

PAYBACK_YEARS = {"High": "2-5", "Medium": "1-3", "Low": "0.5-2"}

def calculate_market_size(base: float, level_idx: int) -> float:
    multipliers = [1.0, 2.0, 3.0, 4.0, 5.0]
    return round(base * multipliers[min(level_idx, len(multipliers)-1)], 1)

def determine_priority(category: str, subdomain: str) -> str:
    high_leverage = ["Process Heat", "Heat Recovery", "Grid Optimization", "Renewable Integration",
                     "Electric Vehicles", "Supplier Emissions", "Corporate Reporting", "Point-Source CCS",
                     "HVAC Core", "Logistics", "Demand Response"]
    base = CATEGORIES_CONFIG.get(category, {}).get("priority_boost", 1)
    if subdomain in high_leverage and category in ["Coordinator", "Optimizer", "Integrator"]:
        return "P0"
    return ["P0", "P1", "P2"][min(base, 2)]

def get_timeline(priority: str, idx: int) -> str:
    return TIMELINES.get(priority, ["Q2 2026"])[idx % len(TIMELINES.get(priority, ["Q2 2026"]))]

def generate_description(domain: str, subdomain: str, category: str, sector: str, region: str, level: str) -> str:
    verbs = {
        "Coordinator": "coordinates and orchestrates", "Optimizer": "optimizes and enhances",
        "Monitor": "monitors and tracks", "Controller": "controls and regulates",
        "Automator": "automates and manages", "Predictor": "predicts and forecasts",
        "Analyzer": "analyzes and evaluates", "Calculator": "calculates and quantifies",
        "Reporter": "reports and documents", "Integrator": "integrates and connects",
        "Communicator": "communicates and engages"
    }
    verb = verbs.get(category, "manages")
    impact = "reduce emissions and operating cost" if domain not in ["Data & Governance", "Supply Chain & Finance"] or subdomain in ["Supplier Emissions", "Product Footprint"] else "enable climate data governance and compliance"
    return f"AI {category.lower()} agent that {verb} {subdomain.lower()} in the {domain.lower()} sector for {sector} in {region} at the {level.lower()} level to {impact}."

def generate_agents() -> List[Dict[str, Any]]:
    """Generate 10,400 agents with balanced distribution across all axes."""
    agents = []
    agent_id = 1
    domain_counts = {}

    for domain_name, domain_config in DOMAINS.items():
        target = domain_config["target_agents"]
        subdomains = domain_config["subdomains"]
        sectors = domain_config["sectors"]
        levels = domain_config["levels"]

        # Create cyclic iterators for balanced distribution
        category_cycle = cycle(ALL_CATEGORIES)
        subdomain_names = list(subdomains.keys())
        subdomain_cycle = cycle(subdomain_names)
        sector_cycle = cycle(sectors)
        region_cycle = cycle(REGIONS)
        level_cycle = cycle(enumerate(levels))

        domain_agents = []

        for _ in range(target):
            category = next(category_cycle)
            subdomain_name = next(subdomain_cycle)
            sector = next(sector_cycle)
            region = next(region_cycle)
            level_idx, level = next(level_cycle)

            subdomain_config = subdomains[subdomain_name]
            carbon_impact = subdomain_config["carbon_impact"]
            market_base = subdomain_config["market_base"]

            priority = determine_priority(category, subdomain_name)
            complexity = CATEGORIES_CONFIG.get(category, {}).get("complexity", "Medium")

            agent = {
                "Agent_ID": f"GLA-{agent_id:05d}",
                "Blueprint_Name": f"{domain_name} {subdomain_name} {category}",
                "Agent_Name": f"{subdomain_name} {category} for {sector} in {region} ({level} level)",
                "Domain": domain_name,
                "Subdomain": subdomain_name,
                "Category": category,
                "Level": level,
                "Sector_Instance": sector,
                "Region": region,
                "Priority": priority,
                "Complexity": complexity,
                "Market_Size_USD_Bn": calculate_market_size(market_base, level_idx),
                "Carbon_Impact_GtCO2_per_year": carbon_impact if carbon_impact > 0 else "N/A",
                "Payback_Years": PAYBACK_YEARS.get(complexity, "2-4") if carbon_impact > 0 else "N/A",
                "Development_Timeline": get_timeline(priority, agent_id % 4),
                "Description": generate_description(domain_name, subdomain_name, category, sector, region, level)
            }

            domain_agents.append(agent)
            agent_id += 1

        agents.extend(domain_agents)
        domain_counts[domain_name] = len(domain_agents)

    print(f"\nDomain agent counts: {domain_counts}")
    print(f"Total generated: {len(agents)}")

    # Ensure exactly 10,400 agents
    while len(agents) < 10400:
        # Add more from high-leverage domains
        for domain_name in ["Industrial", "Energy Systems", "Buildings"]:
            if len(agents) >= 10400:
                break
            domain_config = DOMAINS[domain_name]
            subdomain_name = list(domain_config["subdomains"].keys())[len(agents) % len(domain_config["subdomains"])]
            subdomain_config = domain_config["subdomains"][subdomain_name]
            category = ALL_CATEGORIES[len(agents) % len(ALL_CATEGORIES)]
            sector = domain_config["sectors"][len(agents) % len(domain_config["sectors"])]
            region = REGIONS[len(agents) % len(REGIONS)]
            level_idx = len(agents) % len(domain_config["levels"])
            level = domain_config["levels"][level_idx]

            priority = determine_priority(category, subdomain_name)
            complexity = CATEGORIES_CONFIG.get(category, {}).get("complexity", "Medium")

            agent = {
                "Agent_ID": f"GLA-{len(agents)+1:05d}",
                "Blueprint_Name": f"{domain_name} {subdomain_name} {category}",
                "Agent_Name": f"{subdomain_name} {category} for {sector} in {region} ({level} level)",
                "Domain": domain_name,
                "Subdomain": subdomain_name,
                "Category": category,
                "Level": level,
                "Sector_Instance": sector,
                "Region": region,
                "Priority": priority,
                "Complexity": complexity,
                "Market_Size_USD_Bn": calculate_market_size(subdomain_config["market_base"], level_idx),
                "Carbon_Impact_GtCO2_per_year": subdomain_config["carbon_impact"] if subdomain_config["carbon_impact"] > 0 else "N/A",
                "Payback_Years": PAYBACK_YEARS.get(complexity, "2-4") if subdomain_config["carbon_impact"] > 0 else "N/A",
                "Development_Timeline": get_timeline(priority, len(agents) % 4),
                "Description": generate_description(domain_name, subdomain_name, category, sector, region, level)
            }
            agents.append(agent)

    # Trim if over
    if len(agents) > 10400:
        agents = agents[:10400]
        for i, a in enumerate(agents):
            a["Agent_ID"] = f"GLA-{i+1:05d}"

    return agents

def write_catalog(agents: List[Dict[str, Any]], output_path: str):
    fieldnames = ["Agent_ID", "Blueprint_Name", "Agent_Name", "Domain", "Subdomain",
                  "Category", "Level", "Sector_Instance", "Region", "Priority",
                  "Complexity", "Market_Size_USD_Bn", "Carbon_Impact_GtCO2_per_year",
                  "Payback_Years", "Development_Timeline", "Description"]
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(agents)

def print_stats(agents: List[Dict[str, Any]]):
    print(f"\n{'='*70}")
    print("GL_AGENT_CATALOG_10000.csv - FINAL STATISTICS")
    print(f"{'='*70}\n")
    print(f"Total Agents: {len(agents):,}")

    domains = Counter(a["Domain"] for a in agents)
    print(f"\n=== DOMAIN DISTRIBUTION ===")
    for d, c in domains.most_common():
        print(f"  {d}: {c:,}")

    subdomains = Counter(a["Subdomain"] for a in agents)
    print(f"\n=== SUBDOMAIN DISTRIBUTION (top 15) ===")
    for s, c in subdomains.most_common(15):
        print(f"  {s}: {c:,}")

    priorities = Counter(a["Priority"] for a in agents)
    print(f"\n=== PRIORITY DISTRIBUTION ===")
    for p, c in sorted(priorities.items()):
        print(f"  {p}: {c:,}")

    categories = Counter(a["Category"] for a in agents)
    print(f"\n=== CATEGORY DISTRIBUTION ===")
    for cat, c in categories.most_common():
        print(f"  {cat}: {c:,}")

    levels = Counter(a["Level"] for a in agents)
    print(f"\n=== LEVEL DISTRIBUTION ===")
    for l, c in levels.most_common():
        print(f"  {l}: {c:,}")

    regions = Counter(a["Region"] for a in agents)
    print(f"\n=== REGION DISTRIBUTION ===")
    for r, c in regions.most_common():
        print(f"  {r}: {c:,}")

    timelines = Counter(a["Development_Timeline"] for a in agents)
    print(f"\n=== DEVELOPMENT TIMELINE ===")
    for t, c in sorted(timelines.items()):
        print(f"  {t}: {c:,}")

    blueprints = set(a["Blueprint_Name"] for a in agents)
    print(f"\n=== UNIQUE BLUEPRINTS: {len(blueprints)} ===")

if __name__ == "__main__":
    import sys
    output_path = sys.argv[1] if len(sys.argv) > 1 else "GL_AGENT_CATALOG_10000.csv"

    print("="*70)
    print("GreenLang 10,400 Multi-Domain Agent Catalog Generator")
    print("Based on CTO's portfolio design with context expansion")
    print("="*70)

    agents = generate_agents()
    write_catalog(agents, output_path)
    print_stats(agents)
    print(f"\n{'='*70}")
    print(f"Catalog written to: {output_path}")
    print(f"Total agents: {len(agents):,}")
    print(f"{'='*70}")
