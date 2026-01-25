"""
GL-009 THERMALIQ - Entry Point

This module provides the entry point for running GL-009_ThermalIQ as a module:
    python -m GL-009_ThermalIQ

Execution modes:
    --mode api       Start REST/GraphQL API server
    --mode demo      Run demonstration with sample data
    --mode health    Perform health check and exit
    --mode version   Print version and exit

Examples:
    python -m GL-009_ThermalIQ --mode api --port 8009
    python -m GL-009_ThermalIQ --mode demo
    python -m GL-009_ThermalIQ --mode health
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from typing import Optional


def get_version() -> str:
    """Return version information."""
    from . import __version__, __agent_id__, __unique_name__, __agent_name__
    return (
        f"GL-009 THERMALIQ v{__version__}\n"
        f"Agent ID: {__agent_id__}\n"
        f"Unique Name: {__unique_name__}\n"
        f"Agent Name: {__agent_name__}"
    )


async def run_demo() -> None:
    """Run demonstration with sample data."""
    from .core import ThermalIQOrchestrator, ThermalIQConfig, ThermalAnalysisInput

    print("=" * 60)
    print("GL-009 THERMALIQ - Demonstration Mode")
    print("=" * 60)
    print()

    # Initialize orchestrator
    config = ThermalIQConfig()
    orchestrator = ThermalIQOrchestrator(config)

    # Demo 1: Thermal Efficiency Calculation
    print("[1] Thermal Efficiency Calculation")
    print("-" * 40)
    efficiency = orchestrator.calculate_efficiency(
        energy_in_kW=1000.0,
        heat_out_kW=850.0
    )
    print(f"    Input Energy: 1000.0 kW")
    print(f"    Output Heat:  850.0 kW")
    print(f"    Efficiency:   {efficiency:.2f}%")
    print()

    # Demo 2: Exergy Analysis
    print("[2] Exergy Analysis (Second-Law)")
    print("-" * 40)
    exergy_result = orchestrator.calculate_exergy(
        energy_in_kW=1000.0,
        heat_out_kW=850.0,
        source_temperature_K=573.15,  # 300C
        sink_temperature_K=323.15,    # 50C
        reference_temperature_K=298.15  # 25C
    )
    print(f"    Source Temperature: 573.15 K (300 C)")
    print(f"    Sink Temperature:   323.15 K (50 C)")
    print(f"    Reference:          298.15 K (25 C)")
    print(f"    Exergy Input:       {exergy_result.exergy_input_kW:.2f} kW")
    print(f"    Exergy Output:      {exergy_result.exergy_output_kW:.2f} kW")
    print(f"    Exergy Destruction: {exergy_result.exergy_destruction_kW:.2f} kW")
    print(f"    Exergy Efficiency:  {exergy_result.exergy_efficiency_pct:.2f}%")
    print()

    # Demo 3: Fluid Properties
    print("[3] Fluid Properties Lookup")
    print("-" * 40)
    fluid_props = orchestrator.get_fluid_properties(
        fluid_name="therminol_66",
        temperature_K=473.15  # 200C
    )
    print(f"    Fluid: Therminol 66")
    print(f"    Temperature: 473.15 K (200 C)")
    print(f"    Density:          {fluid_props.density_kg_m3:.2f} kg/m3")
    print(f"    Specific Heat:    {fluid_props.specific_heat_J_kgK:.2f} J/(kg*K)")
    print(f"    Thermal Cond:     {fluid_props.thermal_conductivity_W_mK:.4f} W/(m*K)")
    print(f"    Viscosity:        {fluid_props.dynamic_viscosity_Pa_s:.6f} Pa*s")
    print()

    # Demo 4: Fluid Recommendation
    print("[4] Fluid Recommendation")
    print("-" * 40)
    recommendation = orchestrator.recommend_fluid(
        min_temperature_K=373.15,  # 100C
        max_temperature_K=573.15,  # 300C
        application="heat_transfer"
    )
    print(f"    Temperature Range: 373.15 K - 573.15 K (100 C - 300 C)")
    print(f"    Application: Heat Transfer")
    print(f"    Recommended Fluid: {recommendation.fluid_name}")
    print(f"    Suitability Score: {recommendation.suitability_score:.2f}")
    print(f"    Rationale: {recommendation.rationale}")
    print()

    # Demo 5: Health Check
    print("[5] Health Check")
    print("-" * 40)
    health = orchestrator.health_check()
    print(f"    Status:        {health.status}")
    print(f"    Agent ID:      {health.agent_id}")
    print(f"    Version:       {health.version}")
    print(f"    Timestamp:     {health.timestamp}")
    print()

    print("=" * 60)
    print("Demonstration Complete")
    print("=" * 60)


async def run_health_check() -> bool:
    """Run health check and return status."""
    from .core import ThermalIQOrchestrator, ThermalIQConfig

    print("GL-009 THERMALIQ - Health Check")
    print("-" * 40)

    try:
        config = ThermalIQConfig()
        orchestrator = ThermalIQOrchestrator(config)
        health = orchestrator.health_check()

        print(f"Status:    {health.status}")
        print(f"Agent ID:  {health.agent_id}")
        print(f"Version:   {health.version}")
        print(f"Timestamp: {health.timestamp}")

        if hasattr(health, 'components'):
            print("\nComponent Status:")
            for component, status in health.components.items():
                print(f"  {component}: {status}")

        return health.status == "healthy"

    except Exception as e:
        print(f"ERROR: Health check failed - {str(e)}")
        return False


def run_api_server(host: str = "0.0.0.0", port: int = 8009) -> None:
    """Start the REST/GraphQL API server."""
    try:
        import uvicorn
        from .api import create_app

        print(f"GL-009 THERMALIQ - Starting API Server")
        print(f"Host: {host}")
        print(f"Port: {port}")
        print("-" * 40)

        app = create_app()
        uvicorn.run(app, host=host, port=port)

    except ImportError as e:
        print(f"ERROR: Missing dependency for API server: {e}")
        print("Install with: pip install uvicorn fastapi")
        sys.exit(1)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GL-009 THERMALIQ - Thermal Fluid Analyzer Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m GL-009_ThermalIQ --mode demo       Run demonstration
  python -m GL-009_ThermalIQ --mode api        Start API server
  python -m GL-009_ThermalIQ --mode health     Health check
  python -m GL-009_ThermalIQ --mode version    Show version
        """
    )

    parser.add_argument(
        "--mode",
        choices=["api", "demo", "health", "version"],
        default="demo",
        help="Execution mode (default: demo)"
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API server host (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8009,
        help="API server port (default: 8009)"
    )

    args = parser.parse_args()

    # Handle version mode
    if args.mode == "version":
        print(get_version())
        return 0

    # Handle health check mode
    if args.mode == "health":
        success = asyncio.run(run_health_check())
        return 0 if success else 1

    # Handle demo mode
    if args.mode == "demo":
        asyncio.run(run_demo())
        return 0

    # Handle API mode
    if args.mode == "api":
        run_api_server(host=args.host, port=args.port)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
