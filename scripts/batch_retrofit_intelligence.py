#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Intelligence Retrofit Script

This script retrofits ALL existing GreenLang agents with LLM intelligence
capabilities, solving the "Intelligence Paradox" at scale.

Usage:
    # Dry run - see what would be retrofitted
    python scripts/batch_retrofit_intelligence.py --dry-run

    # Retrofit all agents
    python scripts/batch_retrofit_intelligence.py

    # Retrofit specific module
    python scripts/batch_retrofit_intelligence.py --module greenlang.agents

    # Generate intelligent versions
    python scripts/batch_retrofit_intelligence.py --generate

Author: GreenLang Intelligence Framework
Date: December 2025
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from greenlang.agents.intelligence_mixin import (
    IntelligenceMixin,
    IntelligenceConfig,
    retrofit_agent_class,
    retrofit_all_agents_in_module,
)
from greenlang.agents.intelligence_interface import (
    AgentIntelligenceValidator,
    ValidationResult,
    IntelligenceLevel,
)
from greenlang.agents.base import BaseAgent

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AgentInfo:
    """Information about an agent to retrofit."""
    name: str
    module_path: str
    file_path: str
    current_function: str
    intelligence_opportunities: List[str] = field(default_factory=list)
    retrofit_status: str = "pending"  # pending, retrofitted, skipped, error
    error_message: Optional[str] = None


# =============================================================================
# AGENT CATALOG (from background agent exploration)
# =============================================================================

AGENT_CATALOG = [
    # Core greenlang agents
    AgentInfo(
        name="CarbonAgent",
        module_path="greenlang.agents.carbon_agent",
        file_path="greenlang/agents/carbon_agent.py",
        current_function="Aggregates emissions and calculates total carbon footprint",
        intelligence_opportunities=[
            "LLM-generated carbon footprint summaries",
            "Natural language emission breakdown explanations",
            "AI-powered reduction recommendations"
        ]
    ),
    AgentInfo(
        name="FuelAgent",
        module_path="greenlang.agents.fuel_agent",
        file_path="greenlang/agents/fuel_agent.py",
        current_function="Calculates emissions based on fuel consumption",
        intelligence_opportunities=[
            "LLM-powered fuel substitution recommendations",
            "Natural language consumption pattern analysis",
            "AI-assisted fuel efficiency explanations"
        ]
    ),
    AgentInfo(
        name="GridFactorAgent",
        module_path="greenlang.agents.grid_factor_agent",
        file_path="greenlang/agents/grid_factor_agent.py",
        current_function="Retrieves regional emission factors",
        intelligence_opportunities=[
            "LLM-powered emission factor source explanations",
            "Natural language regional grid comparison",
            "AI-assisted grid decarbonization projections"
        ]
    ),
    AgentInfo(
        name="RecommendationAgent",
        module_path="greenlang.agents.recommendation_agent",
        file_path="greenlang/agents/recommendation_agent.py",
        current_function="Provides actionable recommendations",
        intelligence_opportunities=[
            "LLM-powered recommendation prioritization",
            "Natural language ROI explanations",
            "AI-assisted implementation planning"
        ]
    ),
    AgentInfo(
        name="BenchmarkAgent",
        module_path="greenlang.agents.benchmark_agent",
        file_path="greenlang/agents/benchmark_agent.py",
        current_function="Compares against industry benchmarks",
        intelligence_opportunities=[
            "LLM-powered peer comparison narratives",
            "Natural language performance gap explanations"
        ]
    ),
    AgentInfo(
        name="IntensityAgent",
        module_path="greenlang.agents.intensity_agent",
        file_path="greenlang/agents/intensity_agent.py",
        current_function="Calculates emission intensity metrics",
        intelligence_opportunities=[
            "LLM-powered intensity trend explanations",
            "Natural language benchmark context"
        ]
    ),
    AgentInfo(
        name="BoilerAgent",
        module_path="greenlang.agents.boiler_agent",
        file_path="greenlang/agents/boiler_agent.py",
        current_function="Calculates boiler emissions",
        intelligence_opportunities=[
            "LLM-powered efficiency optimization recommendations",
            "Natural language boiler performance narratives"
        ]
    ),
    AgentInfo(
        name="BuildingProfileAgent",
        module_path="greenlang.agents.building_profile_agent",
        file_path="greenlang/agents/building_profile_agent.py",
        current_function="Generates building energy profiles",
        intelligence_opportunities=[
            "LLM-generated building performance summaries",
            "AI-powered retrofit recommendations"
        ]
    ),
    AgentInfo(
        name="EnergyBalanceAgent",
        module_path="greenlang.agents.energy_balance_agent",
        file_path="greenlang/agents/energy_balance_agent.py",
        current_function="Calculates energy balance",
        intelligence_opportunities=[
            "LLM-powered energy flow explanations",
            "AI-assisted efficiency improvement suggestions"
        ]
    ),
    AgentInfo(
        name="LoadProfileAgent",
        module_path="greenlang.agents.load_profile_agent",
        file_path="greenlang/agents/load_profile_agent.py",
        current_function="Analyzes load profiles",
        intelligence_opportunities=[
            "LLM-powered load pattern explanations",
            "AI-assisted demand response recommendations"
        ]
    ),
    AgentInfo(
        name="SiteInputAgent",
        module_path="greenlang.agents.site_input_agent",
        file_path="greenlang/agents/site_input_agent.py",
        current_function="Processes site input data",
        intelligence_opportunities=[
            "LLM-powered data validation explanations",
            "AI-assisted data enrichment suggestions"
        ]
    ),
    AgentInfo(
        name="SolarResourceAgent",
        module_path="greenlang.agents.solar_resource_agent",
        file_path="greenlang/agents/solar_resource_agent.py",
        current_function="Evaluates solar resource potential",
        intelligence_opportunities=[
            "LLM-powered solar potential explanations",
            "AI-assisted system sizing recommendations"
        ]
    ),
    AgentInfo(
        name="FieldLayoutAgent",
        module_path="greenlang.agents.field_layout_agent",
        file_path="greenlang/agents/field_layout_agent.py",
        current_function="Generates field layouts",
        intelligence_opportunities=[
            "LLM-powered layout optimization explanations",
            "AI-assisted spacing recommendations"
        ]
    ),
    # GL-Agent-Factory agents (13 more)
    AgentInfo(
        name="GL-001 Carbon Emissions Calculator",
        module_path="GL-Agent-Factory.backend.agents.gl_001_carbon_emissions.agent",
        file_path="GL-Agent-Factory/backend/agents/gl_001_carbon_emissions/agent.py",
        current_function="GHG emissions for Scope 1/2/3",
        intelligence_opportunities=[
            "Natural language explanations of emission factor selection",
            "LLM-powered recommendations for emission reduction"
        ]
    ),
    AgentInfo(
        name="GL-002 CBAM Compliance",
        module_path="GL-Agent-Factory.backend.agents.gl_002_cbam_compliance.agent",
        file_path="GL-Agent-Factory/backend/agents/gl_002_cbam_compliance/agent.py",
        current_function="EU CBAM embedded emissions calculation",
        intelligence_opportunities=[
            "LLM-assisted CN code classification",
            "Natural language CBAM report generation"
        ]
    ),
    AgentInfo(
        name="GL-005 Building Energy",
        module_path="GL-Agent-Factory.backend.agents.gl_005_building_energy.agent",
        file_path="GL-Agent-Factory/backend/agents/gl_005_building_energy/agent.py",
        current_function="Building energy performance calculation",
        intelligence_opportunities=[
            "LLM-generated retrofit recommendations",
            "AI-powered CRREM pathway interpretation"
        ]
    ),
    AgentInfo(
        name="GL-006 Scope 3 Emissions",
        module_path="GL-Agent-Factory.backend.agents.gl_006_scope3_emissions.agent",
        file_path="GL-Agent-Factory/backend/agents/gl_006_scope3_emissions/agent.py",
        current_function="Value chain Scope 3 emissions",
        intelligence_opportunities=[
            "LLM-assisted spend category classification",
            "AI-powered emission hotspot explanation"
        ]
    ),
    AgentInfo(
        name="GL-007 EU Taxonomy",
        module_path="GL-Agent-Factory.backend.agents.gl_007_eu_taxonomy.agent",
        file_path="GL-Agent-Factory/backend/agents/gl_007_eu_taxonomy/agent.py",
        current_function="EU Taxonomy alignment evaluation",
        intelligence_opportunities=[
            "LLM-powered NACE code classification",
            "Natural language TSC compliance explanations"
        ]
    ),
    AgentInfo(
        name="GL-008 Green Claims",
        module_path="GL-Agent-Factory.backend.agents.gl_008_green_claims.agent",
        file_path="GL-Agent-Factory/backend/agents/gl_008_green_claims/agent.py",
        current_function="Green claims verification",
        intelligence_opportunities=[
            "LLM-powered greenwashing detection",
            "AI-generated claim improvement recommendations"
        ]
    ),
    AgentInfo(
        name="GL-009 Product Carbon Footprint",
        module_path="GL-Agent-Factory.backend.agents.gl_009_product_carbon_footprint.agent",
        file_path="GL-Agent-Factory/backend/agents/gl_009_product_carbon_footprint/agent.py",
        current_function="Lifecycle PCF calculation",
        intelligence_opportunities=[
            "LLM-assisted BOM classification",
            "AI-powered PCF report narrative generation"
        ]
    ),
]


def discover_agents(module_paths: List[str] = None) -> List[AgentInfo]:
    """
    Discover all agents in the codebase.

    If module_paths is provided, only discover agents in those modules.
    Otherwise, return the pre-cataloged agents.
    """
    if module_paths:
        # TODO: Implement dynamic discovery
        logger.info(f"Discovering agents in: {module_paths}")

    return AGENT_CATALOG


def validate_agent(agent_class: type) -> ValidationResult:
    """Validate an agent for intelligence compliance."""
    validator = AgentIntelligenceValidator()
    return validator.validate_class(agent_class)


def retrofit_single_agent(
    agent_info: AgentInfo,
    intelligence_config: IntelligenceConfig,
    dry_run: bool = False
) -> AgentInfo:
    """
    Retrofit a single agent with intelligence.

    Args:
        agent_info: Information about the agent
        intelligence_config: Intelligence configuration
        dry_run: If True, don't actually retrofit

    Returns:
        Updated AgentInfo with retrofit status
    """
    try:
        logger.info(f"Processing: {agent_info.name}")

        if dry_run:
            logger.info(f"  [DRY RUN] Would retrofit {agent_info.name}")
            agent_info.retrofit_status = "would_retrofit"
            return agent_info

        # Try to import the agent class
        try:
            module_parts = agent_info.module_path.rsplit(".", 1)
            if len(module_parts) == 2:
                module_name, class_name = module_parts
            else:
                module_name = agent_info.module_path
                class_name = agent_info.name

            module = __import__(module_name, fromlist=[class_name])
            agent_class = getattr(module, class_name.replace(" ", "").replace("-", ""))
        except (ImportError, AttributeError) as e:
            logger.warning(f"  Could not import {agent_info.name}: {e}")
            agent_info.retrofit_status = "skipped"
            agent_info.error_message = str(e)
            return agent_info

        # Check if already intelligent
        if hasattr(agent_class, 'generate_explanation'):
            logger.info(f"  Already intelligent: {agent_info.name}")
            agent_info.retrofit_status = "already_intelligent"
            return agent_info

        # Retrofit the agent
        intelligent_class = retrofit_agent_class(agent_class, intelligence_config)
        logger.info(f"  Retrofitted: {agent_info.name} -> Intelligent{agent_info.name}")

        agent_info.retrofit_status = "retrofitted"
        return agent_info

    except Exception as e:
        logger.error(f"  Error retrofitting {agent_info.name}: {e}")
        agent_info.retrofit_status = "error"
        agent_info.error_message = str(e)
        return agent_info


def batch_retrofit(
    agents: List[AgentInfo],
    intelligence_config: IntelligenceConfig,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Retrofit multiple agents in batch.

    Returns:
        Summary of retrofit operation
    """
    results = {
        "total": len(agents),
        "retrofitted": 0,
        "already_intelligent": 0,
        "skipped": 0,
        "errors": 0,
        "agents": []
    }

    for agent_info in agents:
        updated_info = retrofit_single_agent(agent_info, intelligence_config, dry_run)
        results["agents"].append({
            "name": updated_info.name,
            "status": updated_info.retrofit_status,
            "error": updated_info.error_message
        })

        if updated_info.retrofit_status == "retrofitted":
            results["retrofitted"] += 1
        elif updated_info.retrofit_status == "already_intelligent":
            results["already_intelligent"] += 1
        elif updated_info.retrofit_status == "skipped":
            results["skipped"] += 1
        elif updated_info.retrofit_status == "error":
            results["errors"] += 1

    return results


def generate_intelligent_versions(agents: List[AgentInfo], output_dir: Path):
    """
    Generate intelligent versions of agents as new files.

    This creates new *_intelligent.py files for each agent.
    """
    template = '''# -*- coding: utf-8 -*-
"""
Intelligent {name} - AI-Native Version

AUTO-GENERATED by batch_retrofit_intelligence.py
Original: {file_path}

This is the INTELLIGENT version with LLM capabilities:
- generate_explanation()
- generate_recommendations()
- detect_anomalies()
"""

from greenlang.agents.intelligence_mixin import IntelligenceMixin, IntelligenceConfig
from {module_path} import {class_name}


class Intelligent{class_name}(IntelligenceMixin, {class_name}):
    """
    Intelligent version of {class_name}.

    Intelligence Opportunities:
{opportunities}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_intelligence(IntelligenceConfig(
            enabled=True,
            model="auto",
            max_budget_per_call_usd=0.10,
            enable_explanations=True,
            enable_recommendations=True,
            domain_context="climate and sustainability"
        ))


# Factory function
def create_intelligent_{snake_name}(**kwargs):
    """Create an intelligent {name}."""
    return Intelligent{class_name}(**kwargs)
'''

    output_dir.mkdir(parents=True, exist_ok=True)

    for agent_info in agents:
        try:
            class_name = agent_info.name.replace(" ", "").replace("-", "")
            snake_name = agent_info.name.lower().replace(" ", "_").replace("-", "_")
            opportunities = "\n".join(f"    - {opp}" for opp in agent_info.intelligence_opportunities)

            content = template.format(
                name=agent_info.name,
                file_path=agent_info.file_path,
                module_path=agent_info.module_path.rsplit(".", 1)[0] if "." in agent_info.module_path else agent_info.module_path,
                class_name=class_name,
                snake_name=snake_name,
                opportunities=opportunities
            )

            output_file = output_dir / f"{snake_name}_intelligent.py"
            output_file.write_text(content)
            logger.info(f"Generated: {output_file}")

        except Exception as e:
            logger.error(f"Error generating {agent_info.name}: {e}")


def print_summary(results: Dict[str, Any]):
    """Print a summary of the retrofit operation."""
    print("\n" + "=" * 60)
    print("INTELLIGENCE RETROFIT SUMMARY")
    print("=" * 60)
    print(f"\nTotal agents processed: {results['total']}")
    print(f"  Retrofitted:         {results['retrofitted']}")
    print(f"  Already intelligent: {results['already_intelligent']}")
    print(f"  Skipped:             {results['skipped']}")
    print(f"  Errors:              {results['errors']}")

    if results['errors'] > 0:
        print("\nErrors:")
        for agent in results['agents']:
            if agent['status'] == 'error':
                print(f"  - {agent['name']}: {agent['error']}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Batch retrofit GreenLang agents with LLM intelligence"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--module",
        type=str,
        help="Specific module to retrofit"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate intelligent versions as new files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "greenlang" / "agents" / "intelligent",
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    # Configure intelligence
    intelligence_config = IntelligenceConfig(
        enabled=True,
        model="auto",
        max_budget_per_call_usd=0.10,
        enable_explanations=True,
        enable_recommendations=True,
        enable_anomaly_detection=True,
        domain_context="climate and sustainability"
    )

    # Discover agents
    module_paths = [args.module] if args.module else None
    agents = discover_agents(module_paths)

    logger.info(f"Found {len(agents)} agents to process")

    if args.generate:
        # Generate intelligent versions as new files
        generate_intelligent_versions(agents, args.output_dir)
        print(f"\nGenerated intelligent versions in: {args.output_dir}")
    else:
        # Retrofit in-place
        results = batch_retrofit(agents, intelligence_config, args.dry_run)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print_summary(results)


if __name__ == "__main__":
    main()
