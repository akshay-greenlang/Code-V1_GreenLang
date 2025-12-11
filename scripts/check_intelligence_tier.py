#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreenLang Intelligence Tier Checker

Check your current intelligence tier status and test it interactively.

Usage:
    python scripts/check_intelligence_tier.py [--test] [--verbose]

Options:
    --test      Run a test query to verify intelligence is working
    --verbose   Show detailed tier information
    --upgrade   Show instructions to upgrade to higher tiers

This script helps open-source developers understand:
1. Which intelligence tier they have access to
2. How to upgrade to higher tiers
3. Test that intelligence is working correctly

Author: GreenLang Intelligence Framework
Date: December 2025
"""

import sys
import os
import asyncio
import argparse

# Add greenlang to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_banner():
    """Print GreenLang banner."""
    print()
    print("=" * 60)
    print("  GreenLang Multi-Tier Intelligence System")
    print("=" * 60)
    print()


def print_tier_explanation():
    """Print explanation of tier system."""
    print("TIER SYSTEM:")
    print("-" * 60)
    print()
    print("  TIER 2 - Cloud LLM (BYOK - Bring Your Own Key)")
    print("    Providers: OpenAI (GPT-4o), Anthropic (Claude-3)")
    print("    Quality: Highest")
    print("    Cost: Pay-per-use via your API key")
    print("    Setup: Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    print()
    print("  TIER 1 - Local LLM (Free, Private)")
    print("    Providers: Ollama (Llama 3, Mistral, Phi-3)")
    print("    Quality: High (real LLM reasoning)")
    print("    Cost: Free (runs on your hardware)")
    print("    Setup: Install Ollama + pull a model")
    print()
    print("  TIER 0 - Deterministic (Always Available)")
    print("    Provider: DeterministicProvider")
    print("    Quality: Good (templates + rules + statistics)")
    print("    Cost: Free (no external calls)")
    print("    Setup: None! Works immediately after pip install")
    print()


async def check_tiers():
    """Check all tier availability."""
    from greenlang.intelligence.providers.router import SmartProviderRouter

    print("CHECKING TIER AVAILABILITY...")
    print("-" * 60)
    print()

    router = SmartProviderRouter()
    status = await router.detect_available_tiers()

    # Sort by tier value (highest first)
    sorted_tiers = sorted(status.items(), key=lambda x: -x[0].value)

    results = []

    for tier, tier_status in sorted_tiers:
        if tier_status.available:
            icon = "[OK]"
            color_start = "\033[92m"  # Green
        else:
            icon = "[--]"
            color_start = "\033[91m"  # Red

        color_end = "\033[0m"

        # Build status line
        status_line = f"  {color_start}{icon}{color_end} {tier.display_name}"

        if tier_status.available:
            status_line += f" - {tier_status.provider_name}"
            if tier_status.model:
                status_line += f" ({tier_status.model})"
            results.append(("available", tier, tier_status))
        else:
            status_line += f" - {tier_status.reason}"
            results.append(("unavailable", tier, tier_status))

        print(status_line)

    print()

    # Determine active tier
    await router.initialize()
    active_tier = router.current_tier

    if active_tier:
        print(f"ACTIVE TIER: {active_tier.display_name}")
        active_status = status.get(active_tier)
        if active_status:
            print(f"  Provider: {active_status.provider_name}")
            print(f"  Model: {active_status.model}")
    else:
        print("ACTIVE TIER: None (this shouldn't happen!)")

    print()

    return router, status, active_tier


def print_upgrade_instructions(status, active_tier):
    """Print instructions to upgrade tiers."""
    from greenlang.intelligence.providers.router import IntelligenceTier

    print("UPGRADE INSTRUCTIONS:")
    print("-" * 60)
    print()

    tier2_status = status.get(IntelligenceTier.TIER_2_CLOUD)
    tier1_status = status.get(IntelligenceTier.TIER_1_LOCAL)

    if active_tier == IntelligenceTier.TIER_2_CLOUD:
        print("  You're already on the highest tier! No upgrade needed.")
        print()
        return

    # Tier 1 upgrade
    if tier1_status and not tier1_status.available:
        print("  UPGRADE TO TIER 1 (Local LLM - Free):")
        print("  " + "-" * 40)
        print("  1. Download Ollama from https://ollama.ai/download")
        print("  2. Install and start Ollama")
        print("  3. Pull a model:")
        print("     ollama pull llama3.2     # Recommended (8GB RAM)")
        print("     ollama pull phi3         # Faster (4GB RAM)")
        print("     ollama pull mistral      # Good reasoning")
        print("  4. Run this script again to verify")
        print()

    # Tier 2 upgrade
    if tier2_status and not tier2_status.available:
        print("  UPGRADE TO TIER 2 (Cloud LLM - Highest Quality):")
        print("  " + "-" * 40)
        print("  Option A: OpenAI")
        print("    1. Get API key from https://platform.openai.com")
        print("    2. Set environment variable:")
        print("       export OPENAI_API_KEY=sk-...")
        print()
        print("  Option B: Anthropic")
        print("    1. Get API key from https://console.anthropic.com")
        print("    2. Set environment variable:")
        print("       export ANTHROPIC_API_KEY=sk-...")
        print()


async def test_intelligence(router, verbose=False):
    """Test intelligence with a sample query."""
    from greenlang.intelligence.schemas.messages import ChatMessage, Role
    from greenlang.intelligence.runtime.budget import Budget

    print("TESTING INTELLIGENCE...")
    print("-" * 60)
    print()

    test_query = "Explain the carbon footprint of 100 gallons of diesel fuel."
    print(f"Query: {test_query}")
    print()

    messages = [
        ChatMessage(role=Role.user, content=test_query)
    ]

    try:
        response = await router.chat(messages, budget=Budget(max_usd=1.0))

        print("Response:")
        print("-" * 40)

        # Show first 500 chars or full response
        if len(response.text) > 500 and not verbose:
            print(response.text[:500] + "...")
            print()
            print(f"[Response truncated - {len(response.text)} chars total. Use --verbose for full response]")
        else:
            print(response.text)

        print()
        print("Metadata:")
        print(f"  Provider: {response.provider_info.provider}")
        print(f"  Model: {response.provider_info.model}")
        print(f"  Tokens: {response.usage.total_tokens}")
        print(f"  Cost: ${response.usage.cost_usd:.6f}")

        if response.provider_info.extra:
            tier_name = response.provider_info.extra.get('tier_name', 'unknown')
            print(f"  Tier: {tier_name}")

        print()
        print("TEST PASSED - Intelligence is working!")

    except Exception as e:
        print(f"TEST FAILED: {e}")
        print()
        print("Please check your setup and try again.")

    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check and test GreenLang intelligence tiers"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a test query to verify intelligence"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information"
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Show upgrade instructions"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )

    args = parser.parse_args()

    # Disable colors if requested or not a terminal
    if args.no_color or not sys.stdout.isatty():
        # Remove ANSI color codes by redefining
        import builtins
        original_print = builtins.print

        def print_no_color(*args, **kwargs):
            import re
            new_args = []
            for arg in args:
                if isinstance(arg, str):
                    arg = re.sub(r'\033\[\d+m', '', arg)
                new_args.append(arg)
            original_print(*new_args, **kwargs)

        builtins.print = print_no_color

    async def run():
        print_banner()

        if args.verbose:
            print_tier_explanation()

        router, status, active_tier = await check_tiers()

        if args.upgrade:
            print_upgrade_instructions(status, active_tier)

        if args.test:
            await test_intelligence(router, verbose=args.verbose)

        # Summary
        print("=" * 60)
        print()
        print("  GreenLang Intelligence: READY")
        print(f"  Active Tier: {active_tier.display_name if active_tier else 'None'}")
        print()

        if not args.test:
            print("  Run with --test to verify intelligence is working")
        if not args.upgrade:
            print("  Run with --upgrade for tier upgrade instructions")

        print()
        print("=" * 60)

        await router.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
