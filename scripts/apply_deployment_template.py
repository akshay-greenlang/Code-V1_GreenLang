#!/usr/bin/env python3
"""Apply Deployment Pack Template for GreenLang AI Agents.

This script takes the universal deployment pack template and generates
agent-specific deployment configurations with appropriate values filled in.

Features:
- Reads universal template from templates/agent_deployment_pack.yaml
- Fills in agent-specific values (name, domain, resources, etc.)
- Validates agent exists in the codebase
- Creates deployment pack in packs/{agent_name}/ directory
- Supports batch generation for all agents
- Provides interactive mode for custom configuration
- Validates generated configuration

Usage:
    # Generate deployment pack for a single agent
    python scripts/apply_deployment_template.py --agent fuel_ai

    # Generate with custom domain
    python scripts/apply_deployment_template.py --agent fuel_ai --domain emissions

    # Generate for all AI agents
    python scripts/apply_deployment_template.py --all-agents

    # Interactive mode (prompts for values)
    python scripts/apply_deployment_template.py --interactive

    # Generate with custom output directory
    python scripts/apply_deployment_template.py --agent fuel_ai --output ./custom_packs

    # Dry run (show what would be generated)
    python scripts/apply_deployment_template.py --agent fuel_ai --dry-run

Author: GreenLang Framework Team
Date: October 2025
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import re
from datetime import datetime


# ============================================================================
# AGENT REGISTRY
# ============================================================================
# All GreenLang AI agents with their configurations
AGENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "fuel_ai": {
        "domain": "emissions",
        "name": "AI-Powered Fuel Emissions Calculator",
        "description": "Calculate CO2e emissions from fuel consumption with AI-enhanced explanations and recommendations",
        "agent_type": "ai_orchestration",
        "memory_mb": 512,
        "cpu_cores": 1.0,
        "ai_budget_usd": 0.50,
        "module_path": "greenlang.agents.fuel_agent_ai",
        "class_name": "FuelAgentAI",
    },
    "carbon_ai": {
        "domain": "emissions",
        "name": "AI-Powered Carbon Footprint Analyzer",
        "description": "Comprehensive carbon footprint analysis with AI-powered insights and reduction strategies",
        "agent_type": "ai_orchestration",
        "memory_mb": 512,
        "cpu_cores": 1.0,
        "ai_budget_usd": 0.50,
        "module_path": "greenlang.agents.carbon_agent_ai",
        "class_name": "CarbonAgentAI",
    },
    "grid_factor_ai": {
        "domain": "emissions",
        "name": "AI-Powered Grid Emission Factor Calculator",
        "description": "Calculate grid emission factors with AI-enhanced temporal and regional analysis",
        "agent_type": "ai_orchestration",
        "memory_mb": 512,
        "cpu_cores": 1.0,
        "ai_budget_usd": 0.50,
        "module_path": "greenlang.agents.grid_factor_agent_ai",
        "class_name": "GridFactorAgentAI",
    },
    "recommendation_ai": {
        "domain": "analytics",
        "name": "AI-Powered Emission Reduction Recommendation Engine",
        "description": "Generate personalized emission reduction recommendations using AI analysis",
        "agent_type": "ai_orchestration",
        "memory_mb": 768,
        "cpu_cores": 1.5,
        "ai_budget_usd": 0.75,
        "module_path": "greenlang.agents.recommendation_agent_ai",
        "class_name": "RecommendationAgentAI",
    },
    "report_ai": {
        "domain": "reporting",
        "name": "AI-Powered Sustainability Report Generator",
        "description": "Generate comprehensive sustainability reports with AI-enhanced narratives and insights",
        "agent_type": "ai_orchestration",
        "memory_mb": 1024,
        "cpu_cores": 1.5,
        "ai_budget_usd": 1.0,
        "module_path": "greenlang.agents.report_agent_ai",
        "class_name": "ReportAgentAI",
    },
    "forecast_sarima": {
        "domain": "analytics",
        "name": "SARIMA Time Series Forecasting Agent",
        "description": "Forecast emissions and energy consumption using SARIMA statistical models",
        "agent_type": "ml_inference",
        "memory_mb": 1024,
        "cpu_cores": 2.0,
        "ai_budget_usd": 0.0,  # No AI, pure ML
        "module_path": "greenlang.agents.forecast_agent_sarima",
        "class_name": "ForecastAgentSARIMA",
    },
    "anomaly_iforest": {
        "domain": "analytics",
        "name": "Isolation Forest Anomaly Detection Agent",
        "description": "Detect anomalies in emissions and energy data using Isolation Forest algorithm",
        "agent_type": "ml_inference",
        "memory_mb": 768,
        "cpu_cores": 1.5,
        "ai_budget_usd": 0.0,  # No AI, pure ML
        "module_path": "greenlang.agents.anomaly_agent_iforest",
        "class_name": "AnomalyAgentIForest",
    },
    "industrial_process_heat_ai": {
        "domain": "industrial",
        "name": "AI-Powered Industrial Process Heat Analyzer",
        "description": "Analyze industrial process heat systems with AI-enhanced optimization recommendations",
        "agent_type": "ai_orchestration",
        "memory_mb": 512,
        "cpu_cores": 1.0,
        "ai_budget_usd": 0.50,
        "module_path": "greenlang.agents.industrial_process_heat_agent_ai",
        "class_name": "IndustrialProcessHeatAgentAI",
    },
    "boiler_replacement_ai": {
        "domain": "industrial",
        "name": "AI-Powered Boiler Replacement Advisor",
        "description": "Analyze boiler systems and recommend replacements with AI-powered feasibility analysis",
        "agent_type": "ai_orchestration",
        "memory_mb": 512,
        "cpu_cores": 1.0,
        "ai_budget_usd": 0.50,
        "module_path": "greenlang.agents.boiler_replacement_agent_ai",
        "class_name": "BoilerReplacementAgentAI",
    },
    "industrial_heat_pump_ai": {
        "domain": "industrial",
        "name": "AI-Powered Industrial Heat Pump Analyzer",
        "description": "Analyze industrial heat pump opportunities with AI-enhanced feasibility and ROI analysis",
        "agent_type": "ai_orchestration",
        "memory_mb": 512,
        "cpu_cores": 1.0,
        "ai_budget_usd": 0.50,
        "module_path": "greenlang.agents.industrial_heat_pump_agent_ai",
        "class_name": "IndustrialHeatPumpAgentAI",
    },
}


# ============================================================================
# TEMPLATE PROCESSING
# ============================================================================
def load_template(template_path: Path) -> str:
    """Load the deployment pack template.

    Args:
        template_path: Path to the template YAML file

    Returns:
        str: Template content as string

    Raises:
        FileNotFoundError: If template file doesn't exist
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def fill_template(
    template_content: str,
    agent_id: str,
    agent_config: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> str:
    """Fill template with agent-specific values.

    Args:
        template_content: Template YAML content as string
        agent_id: Agent identifier (e.g., "fuel_ai")
        agent_config: Agent configuration from AGENT_REGISTRY
        overrides: Optional dictionary of override values

    Returns:
        str: Filled template content
    """
    # Merge overrides into agent_config
    if overrides:
        agent_config = {**agent_config, **overrides}

    # Extract values
    domain = agent_config["domain"]
    name = agent_config["name"]
    description = agent_config["description"]
    agent_type = agent_config["agent_type"]
    memory_mb = agent_config["memory_mb"]
    cpu_cores = agent_config["cpu_cores"]
    ai_budget_usd = agent_config["ai_budget_usd"]

    # Replace placeholders
    filled = template_content

    # Pack metadata
    filled = filled.replace("{domain}/{agent_name}", f"{domain}/{agent_id}")
    filled = filled.replace("{agent_name}", agent_id)
    filled = filled.replace("{domain}", domain)
    filled = filled.replace("{Agent description}", description)
    filled = filled.replace("{Agent description - what this agent does and why it matters}", description)

    # Resource requirements
    filled = re.sub(
        r"memory_mb: 512",
        f"memory_mb: {memory_mb}",
        filled,
        count=1
    )
    filled = re.sub(
        r"cpu_cores: 1\.0",
        f"cpu_cores: {cpu_cores}",
        filled,
        count=1
    )

    # Agent type
    filled = re.sub(
        r'agent_type: "ai_orchestration"',
        f'agent_type: "{agent_type}"',
        filled,
        count=1
    )

    # AI budget
    filled = re.sub(
        r"default_budget_usd: 0\.50",
        f"default_budget_usd: {ai_budget_usd}",
        filled,
        count=1
    )

    # Docker image
    filled = filled.replace(
        'image: "greenlang/agent:{agent_name}:latest"',
        f'image: "greenlang/agent:{agent_id}:latest"'
    )

    return filled


def validate_agent_exists(agent_id: str, root_dir: Path) -> bool:
    """Validate that the agent exists in the codebase.

    Args:
        agent_id: Agent identifier
        root_dir: Project root directory

    Returns:
        bool: True if agent exists, False otherwise
    """
    if agent_id not in AGENT_REGISTRY:
        print(f"[ERROR] Agent '{agent_id}' not found in registry")
        return False

    # Check if agent file exists
    agent_config = AGENT_REGISTRY[agent_id]
    module_path = agent_config["module_path"]

    # Convert module path to file path
    # e.g., greenlang.agents.fuel_agent_ai -> greenlang/agents/fuel_agent_ai.py
    file_path = root_dir / (module_path.replace(".", "/") + ".py")

    if not file_path.exists():
        print(f"[WARNING] Agent file not found: {file_path}")
        print(f"          Continuing anyway (agent may not be implemented yet)")

    return True


def generate_deployment_pack(
    agent_id: str,
    template_path: Path,
    output_dir: Path,
    overrides: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> bool:
    """Generate deployment pack for an agent.

    Args:
        agent_id: Agent identifier
        template_path: Path to template file
        output_dir: Output directory for deployment pack
        overrides: Optional configuration overrides
        dry_run: If True, only show what would be generated

    Returns:
        bool: True if successful, False otherwise
    """
    # Validate agent exists
    root_dir = template_path.parent.parent
    if not validate_agent_exists(agent_id, root_dir):
        return False

    # Get agent configuration
    agent_config = AGENT_REGISTRY[agent_id]

    # Load template
    try:
        template_content = load_template(template_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    # Fill template
    filled_content = fill_template(template_content, agent_id, agent_config, overrides)

    # Create output directory
    agent_output_dir = output_dir / agent_id
    deployment_file = agent_output_dir / "deployment_pack.yaml"

    if dry_run:
        print(f"\n{'='*80}")
        print(f"DRY RUN: Would generate deployment pack for '{agent_id}'")
        print(f"{'='*80}")
        print(f"Output: {deployment_file}")
        print(f"\nAgent Configuration:")
        print(f"  Domain: {agent_config['domain']}")
        print(f"  Name: {agent_config['name']}")
        print(f"  Type: {agent_config['agent_type']}")
        print(f"  Memory: {agent_config['memory_mb']} MB")
        print(f"  CPU: {agent_config['cpu_cores']} cores")
        print(f"  AI Budget: ${agent_config['ai_budget_usd']}")
        print(f"\nPreview (first 50 lines):")
        print("-" * 80)
        lines = filled_content.split("\n")
        for line in lines[:50]:
            print(line)
        print("-" * 80)
        return True

    # Create directories
    agent_output_dir.mkdir(parents=True, exist_ok=True)

    # Write deployment pack
    try:
        with open(deployment_file, "w", encoding="utf-8") as f:
            f.write(filled_content)

        print(f"[SUCCESS] Generated deployment pack: {deployment_file}")

        # Validate YAML syntax
        try:
            yaml.safe_load(filled_content)
            print(f"[SUCCESS] YAML validation passed")
        except yaml.YAMLError as e:
            print(f"[WARNING] YAML validation failed: {e}")
            return False

        return True

    except Exception as e:
        print(f"[ERROR] Error writing deployment pack: {e}")
        return False


def generate_all_agents(
    template_path: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> int:
    """Generate deployment packs for all agents.

    Args:
        template_path: Path to template file
        output_dir: Output directory for deployment packs
        dry_run: If True, only show what would be generated

    Returns:
        int: Number of successfully generated packs
    """
    print(f"\n{'='*80}")
    print(f"Generating deployment packs for all {len(AGENT_REGISTRY)} agents")
    print(f"{'='*80}\n")

    success_count = 0
    failed_agents = []

    for agent_id in sorted(AGENT_REGISTRY.keys()):
        print(f"\n[PACK] Processing agent: {agent_id}")
        print("-" * 80)

        if generate_deployment_pack(agent_id, template_path, output_dir, dry_run=dry_run):
            success_count += 1
        else:
            failed_agents.append(agent_id)

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"[SUCCESS] Successfully generated: {success_count}/{len(AGENT_REGISTRY)}")

    if failed_agents:
        print(f"[ERROR] Failed: {len(failed_agents)}")
        print(f"        Failed agents: {', '.join(failed_agents)}")

    print(f"\nOutput directory: {output_dir}")
    print(f"{'='*80}\n")

    return success_count


def interactive_mode(template_path: Path, output_dir: Path) -> bool:
    """Interactive mode for custom agent configuration.

    Args:
        template_path: Path to template file
        output_dir: Output directory for deployment packs

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*80)
    print("INTERACTIVE DEPLOYMENT PACK GENERATOR")
    print("="*80 + "\n")

    # Show available agents
    print("Available agents:")
    for i, (agent_id, config) in enumerate(sorted(AGENT_REGISTRY.items()), 1):
        print(f"  {i}. {agent_id:30} - {config['name']}")

    # Get agent selection
    while True:
        selection = input("\nSelect agent (number or ID): ").strip()

        # Try as number
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(AGENT_REGISTRY):
                agent_id = sorted(AGENT_REGISTRY.keys())[idx]
                break
        except ValueError:
            pass

        # Try as agent ID
        if selection in AGENT_REGISTRY:
            agent_id = selection
            break

        print("[ERROR] Invalid selection. Please try again.")

    agent_config = AGENT_REGISTRY[agent_id]
    print(f"\n[SELECTED] {agent_id}")
    print(f"           {agent_config['name']}")

    # Get overrides
    print("\n" + "-"*80)
    print("Configuration (press Enter to use defaults)")
    print("-"*80)

    overrides = {}

    # Domain
    domain = input(f"Domain [{agent_config['domain']}]: ").strip()
    if domain:
        overrides["domain"] = domain

    # Memory
    memory = input(f"Memory (MB) [{agent_config['memory_mb']}]: ").strip()
    if memory:
        try:
            overrides["memory_mb"] = int(memory)
        except ValueError:
            print("[WARNING] Invalid memory value, using default")

    # CPU
    cpu = input(f"CPU cores [{agent_config['cpu_cores']}]: ").strip()
    if cpu:
        try:
            overrides["cpu_cores"] = float(cpu)
        except ValueError:
            print("[WARNING] Invalid CPU value, using default")

    # AI Budget
    if agent_config["ai_budget_usd"] > 0:
        budget = input(f"AI Budget (USD) [{agent_config['ai_budget_usd']}]: ").strip()
        if budget:
            try:
                overrides["ai_budget_usd"] = float(budget)
            except ValueError:
                print("[WARNING] Invalid budget value, using default")

    # Generate
    print("\n" + "-"*80)
    confirm = input("Generate deployment pack? [Y/n]: ").strip().lower()

    if confirm in ("", "y", "yes"):
        return generate_deployment_pack(
            agent_id,
            template_path,
            output_dir,
            overrides=overrides if overrides else None,
        )
    else:
        print("[CANCELLED] Deployment pack generation cancelled")
        return False


def list_agents() -> None:
    """List all available agents."""
    print("\n" + "="*80)
    print("GREENLANG AI AGENTS")
    print("="*80 + "\n")

    domains = {}
    for agent_id, config in AGENT_REGISTRY.items():
        domain = config["domain"]
        if domain not in domains:
            domains[domain] = []
        domains[domain].append((agent_id, config))

    for domain in sorted(domains.keys()):
        print(f"\n[{domain.upper()}]")
        print("-" * 80)

        for agent_id, config in sorted(domains[domain]):
            print(f"  * {agent_id:30} - {config['name']}")
            print(f"    Type: {config['agent_type']:20} | "
                  f"Memory: {config['memory_mb']} MB | "
                  f"CPU: {config['cpu_cores']} cores")

    print(f"\n{'='*80}")
    print(f"Total: {len(AGENT_REGISTRY)} agents")
    print(f"{'='*80}\n")


# ============================================================================
# MAIN
# ============================================================================
def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Apply deployment pack template for GreenLang AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate deployment pack for a single agent
  python scripts/apply_deployment_template.py --agent fuel_ai

  # Generate for all agents
  python scripts/apply_deployment_template.py --all-agents

  # Interactive mode
  python scripts/apply_deployment_template.py --interactive

  # Dry run
  python scripts/apply_deployment_template.py --agent fuel_ai --dry-run

  # List all agents
  python scripts/apply_deployment_template.py --list
        """
    )

    parser.add_argument(
        "--agent",
        type=str,
        help="Agent ID to generate deployment pack for"
    )

    parser.add_argument(
        "--all-agents",
        action="store_true",
        help="Generate deployment packs for all agents"
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode with prompts"
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available agents"
    )

    parser.add_argument(
        "--template",
        type=Path,
        help="Path to template file (default: templates/agent_deployment_pack.yaml)"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output directory (default: packs/)"
    )

    parser.add_argument(
        "--domain",
        type=str,
        help="Override agent domain"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing files"
    )

    args = parser.parse_args()

    # Determine project root
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent

    # Set defaults
    if args.template is None:
        args.template = project_root / "templates" / "agent_deployment_pack.yaml"

    if args.output is None:
        args.output = project_root / "packs"

    # List agents
    if args.list:
        list_agents()
        return 0

    # Interactive mode
    if args.interactive:
        return 0 if interactive_mode(args.template, args.output) else 1

    # Generate all agents
    if args.all_agents:
        success_count = generate_all_agents(args.template, args.output, args.dry_run)
        return 0 if success_count > 0 else 1

    # Single agent
    if args.agent:
        # Build overrides
        overrides = {}
        if args.domain:
            overrides["domain"] = args.domain

        success = generate_deployment_pack(
            args.agent,
            args.template,
            args.output,
            overrides=overrides if overrides else None,
            dry_run=args.dry_run,
        )
        return 0 if success else 1

    # No action specified
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
