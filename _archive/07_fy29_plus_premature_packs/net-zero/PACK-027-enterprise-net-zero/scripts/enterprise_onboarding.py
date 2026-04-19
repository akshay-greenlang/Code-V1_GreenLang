#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PACK-027: Enterprise Onboarding Script
=======================================

Interactive wizard for onboarding large enterprises (>250 employees, >$50M revenue)
to the Enterprise Net Zero Pack.

Usage:
    python scripts/enterprise_onboarding.py [--api-url http://localhost:8000]

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-027 Enterprise Net Zero Pack
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add pack to path
PACK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACK_DIR))

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EnterpriseOnboardingWizard:
    """
    8-step enterprise onboarding wizard for PACK-027.

    Steps:
        1. Organization Profile
        2. Multi-Entity Hierarchy
        3. Consolidation Approach
        4. Data Sources & ERP Integration
        5. Baseline Year Selection
        6. Target Setting Strategy
        7. Assurance Level
        8. Configuration Review & Deployment
    """

    def __init__(self, api_url: str):
        """Initialize onboarding wizard."""
        self.api_url = api_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=60.0)
        self.config = {}

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

    def print_header(self, step: int, title: str):
        """Print step header."""
        print("\n" + "=" * 80)
        print(f"STEP {step}/8: {title}")
        print("=" * 80 + "\n")

    def print_section(self, title: str):
        """Print section divider."""
        print(f"\n--- {title} ---\n")

    async def step1_organization_profile(self):
        """Step 1: Collect organization profile information."""
        self.print_header(1, "Organization Profile")

        print("Please provide basic information about your organization:\n")

        self.config["organization"] = {
            "organization_id": str(uuid.uuid4()),
            "organization_name": input("Organization Legal Name: ").strip(),
            "headquarters_country": input("Headquarters Country: ").strip(),
            "sector": input("Primary Sector (e.g., manufacturing, financial_services, technology): ").strip(),
            "employee_count": int(input("Total Employees: ")),
            "revenue_usd": float(input("Annual Revenue (USD): ")),
            "fiscal_year_end": input("Fiscal Year End (MM-DD, e.g., 12-31): ").strip(),
            "created_at": datetime.utcnow().isoformat(),
        }

        print(f"\n✓ Organization profile created: {self.config['organization']['organization_name']}")

    async def step2_entity_hierarchy(self):
        """Step 2: Define multi-entity hierarchy."""
        self.print_header(2, "Multi-Entity Hierarchy")

        print("Define your organizational structure:\n")

        entity_count = int(input("Number of legal entities (including parent): "))

        self.config["entities"] = []

        for i in range(entity_count):
            self.print_section(f"Entity {i+1}/{entity_count}")

            entity = {
                "entity_id": str(uuid.uuid4()),
                "name": input("  Entity Name: ").strip(),
                "country": input("  Country: ").strip(),
                "ownership_pct": float(input("  Ownership % (0-100): ")),
                "is_parent": i == 0,  # First entity is parent
            }

            self.config["entities"].append(entity)

        print(f"\n✓ {len(self.config['entities'])} entities defined")

    async def step3_consolidation_approach(self):
        """Step 3: Select consolidation approach."""
        self.print_header(3, "Consolidation Approach")

        print("Select your GHG Protocol consolidation approach:\n")
        print("  1. Financial Control (recommended for most enterprises)")
        print("  2. Operational Control")
        print("  3. Equity Share\n")

        approaches = {
            "1": "FINANCIAL_CONTROL",
            "2": "OPERATIONAL_CONTROL",
            "3": "EQUITY_SHARE",
        }

        choice = input("Selection (1-3): ").strip()
        self.config["consolidation_approach"] = approaches.get(choice, "FINANCIAL_CONTROL")

        print(f"\n✓ Consolidation approach: {self.config['consolidation_approach']}")

    async def step4_data_sources(self):
        """Step 4: Configure data sources and ERP integration."""
        self.print_header(4, "Data Sources & ERP Integration")

        print("Configure your enterprise data sources:\n")

        erp_systems = {
            "1": "SAP",
            "2": "Oracle",
            "3": "Workday",
            "4": "None",
        }

        print("Primary ERP System:")
        for key, value in erp_systems.items():
            print(f"  {key}. {value}")

        erp_choice = input("\nSelection (1-4): ").strip()
        erp_system = erp_systems.get(erp_choice, "None")

        self.config["data_sources"] = {
            "erp_system": erp_system,
            "erp_enabled": erp_system != "None",
        }

        if erp_system != "None":
            print(f"\n✓ {erp_system} integration will be configured")
            print("  (API credentials will be configured in deployment step)")

    async def step5_baseline_year(self):
        """Step 5: Select baseline year."""
        self.print_header(5, "Baseline Year Selection")

        print("Select your GHG inventory baseline year:\n")

        baseline_year = int(input("Baseline Year (e.g., 2019): "))
        reporting_year = int(input("Current Reporting Year (e.g., 2025): "))

        self.config["baseline"] = {
            "baseline_year": baseline_year,
            "reporting_year": reporting_year,
        }

        print(f"\n✓ Baseline year: {baseline_year}")
        print(f"✓ Reporting year: {reporting_year}")

    async def step6_target_strategy(self):
        """Step 6: Define target setting strategy."""
        self.print_header(6, "Target Setting Strategy")

        print("Define your net zero target strategy:\n")

        sbti_pathways = {
            "1": "ACA_15C (Absolute Contraction Approach - 1.5°C)",
            "2": "ACA_WB2C (Absolute Contraction Approach - Well Below 2°C)",
            "3": "SDA (Sectoral Decarbonization Approach)",
            "4": "MIXED (ACA + SDA for different scopes)",
        }

        print("SBTi Pathway:")
        for key, value in sbti_pathways.items():
            print(f"  {key}. {value}")

        pathway_choice = input("\nSelection (1-4): ").strip()
        pathway_map = {
            "1": "ACA_15C",
            "2": "ACA_WB2C",
            "3": "SDA",
            "4": "MIXED",
        }
        pathway = pathway_map.get(pathway_choice, "ACA_15C")

        target_year = int(input("Near-term target year (e.g., 2030): "))
        net_zero_year = int(input("Net zero target year (e.g., 2050): "))

        self.config["targets"] = {
            "sbti_pathway": pathway,
            "target_year": target_year,
            "net_zero_year": net_zero_year,
        }

        print(f"\n✓ SBTi pathway: {pathway}")
        print(f"✓ Near-term target: {target_year}")
        print(f"✓ Net zero target: {net_zero_year}")

    async def step7_assurance_level(self):
        """Step 7: Select external assurance level."""
        self.print_header(7, "External Assurance")

        print("Configure external assurance requirements:\n")

        assurance_levels = {
            "1": "LIMITED (ISO 14064-3 limited assurance)",
            "2": "REASONABLE (ISO 14064-3 reasonable assurance)",
            "3": "NONE (No external assurance required)",
        }

        print("Assurance Level:")
        for key, value in assurance_levels.items():
            print(f"  {key}. {value}")

        assurance_choice = input("\nSelection (1-3): ").strip()
        assurance_map = {
            "1": "LIMITED",
            "2": "REASONABLE",
            "3": "NONE",
        }
        assurance_level = assurance_map.get(assurance_choice, "NONE")

        self.config["assurance"] = {
            "enabled": assurance_level != "NONE",
            "level": assurance_level,
        }

        if assurance_level != "NONE":
            print(f"\n✓ Assurance level: {assurance_level}")
        else:
            print("\n✓ No external assurance configured")

    async def step8_review_and_deploy(self):
        """Step 8: Review configuration and deploy."""
        self.print_header(8, "Configuration Review & Deployment")

        print("Review your enterprise configuration:\n")

        # Print summary
        print("ORGANIZATION:")
        print(f"  Name:           {self.config['organization']['organization_name']}")
        print(f"  Sector:         {self.config['organization']['sector']}")
        print(f"  Employees:      {self.config['organization']['employee_count']:,}")
        print(f"  Revenue:        ${self.config['organization']['revenue_usd']:,.0f}")

        print("\nENTITIES:")
        print(f"  Total Entities: {len(self.config['entities'])}")
        print(f"  Consolidation:  {self.config['consolidation_approach']}")

        print("\nDATA SOURCES:")
        print(f"  ERP System:     {self.config['data_sources']['erp_system']}")

        print("\nBASELINE:")
        print(f"  Baseline Year:  {self.config['baseline']['baseline_year']}")
        print(f"  Reporting Year: {self.config['baseline']['reporting_year']}")

        print("\nTARGETS:")
        print(f"  SBTi Pathway:   {self.config['targets']['sbti_pathway']}")
        print(f"  Target Year:    {self.config['targets']['target_year']}")
        print(f"  Net Zero Year:  {self.config['targets']['net_zero_year']}")

        print("\nASSURANCE:")
        print(f"  Enabled:        {self.config['assurance']['enabled']}")
        if self.config['assurance']['enabled']:
            print(f"  Level:          {self.config['assurance']['level']}")

        # Save configuration
        config_file = PACK_DIR / "config" / f"enterprise_config_{self.config['organization']['organization_id'][:8]}.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"\n✓ Configuration saved to: {config_file}")

        # Deployment instructions
        print("\n" + "=" * 80)
        print("DEPLOYMENT INSTRUCTIONS")
        print("=" * 80 + "\n")

        print("To deploy this configuration:\n")
        print("1. Start Docker Desktop")
        print("2. Apply database migrations:")
        print(f"   python scripts/apply_migrations.py\n")
        print("3. Verify migrations:")
        print(f"   python scripts/verify_migrations.py\n")
        print("4. Deploy to Kubernetes:")
        print(f"   bash scripts/deploy.sh production\n")
        print("5. Run health checks:")
        print(f"   python scripts/health_check.py\n")
        print("6. Access the pack:")
        print(f"   API: {self.api_url}")
        print(f"   Docs: {self.api_url}/docs\n")

        print("=" * 80)
        print("✅ Enterprise onboarding complete!")
        print("=" * 80 + "\n")

    async def run_wizard(self):
        """Run the complete 8-step onboarding wizard."""
        print("\n")
        print("=" * 80)
        print("PACK-027: Enterprise Net Zero Pack - Onboarding Wizard")
        print("=" * 80)
        print("\nWelcome! This wizard will guide you through onboarding your enterprise")
        print("to the Enterprise Net Zero Pack in 8 steps.\n")

        input("Press Enter to begin...")

        # Run all 8 steps
        await self.step1_organization_profile()
        await self.step2_entity_hierarchy()
        await self.step3_consolidation_approach()
        await self.step4_data_sources()
        await self.step5_baseline_year()
        await self.step6_target_strategy()
        await self.step7_assurance_level()
        await self.step8_review_and_deploy()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PACK-027 Enterprise Onboarding")
    parser.add_argument(
        "--api-url",
        default=os.getenv("API_URL", "http://localhost:8000"),
        help="API base URL",
    )
    args = parser.parse_args()

    wizard = EnterpriseOnboardingWizard(args.api_url)

    try:
        await wizard.run_wizard()
    finally:
        await wizard.close()


if __name__ == "__main__":
    asyncio.run(main())
