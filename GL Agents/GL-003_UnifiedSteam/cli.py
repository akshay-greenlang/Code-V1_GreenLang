"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - Command Line Interface

Unified steam optimizer with IAPWS-IF97 thermodynamic calculations, SHAP/LIME
explainability, causal inference for root cause analysis, steam quality control,
desuperheater optimization, enthalpy balance, condensate recovery, and real-time
uncertainty quantification.

Absorbed capabilities from GL-008, GL-012, GL-017.

Author: GreenLang AI Agent Workforce
Version: 1.0.0
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GL-003-CLI")


def setup_argparse() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="gl003-unifiedsteam",
        description="GL-003 UNIFIEDSTEAM - Steam System Optimizer",
        epilog="For more information, visit https://greenlang.ai/agents/gl-003"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="GL-003 UNIFIEDSTEAM v1.0.0"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Start server command
    start_parser = subparsers.add_parser("start", help="Start the GL-003 server")
    start_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    start_parser.add_argument(
        "--port",
        type=int,
        default=8003,
        help="Port to listen on"
    )
    start_parser.add_argument(
        "--mode",
        type=str,
        choices=["advisory", "closed-loop"],
        default="advisory",
        help="Operating mode"
    )
    start_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes"
    )

    # Compute properties command
    compute_parser = subparsers.add_parser(
        "compute",
        help="Compute steam properties"
    )
    compute_parser.add_argument(
        "--pressure",
        type=float,
        required=True,
        help="Steam pressure in kPa(g)"
    )
    compute_parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        help="Steam temperature in °C"
    )
    compute_parser.add_argument(
        "--quality",
        type=float,
        default=None,
        help="Steam quality (dryness fraction) if known"
    )
    compute_parser.add_argument(
        "--output",
        type=str,
        choices=["json", "table", "yaml"],
        default="table",
        help="Output format"
    )

    # Optimize command
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Run optimization"
    )
    optimize_parser.add_argument(
        "--type",
        type=str,
        choices=["desuperheater", "condensate", "network", "traps", "combined"],
        required=True,
        help="Optimization type"
    )
    optimize_parser.add_argument(
        "--site",
        type=str,
        required=True,
        help="Site identifier"
    )
    optimize_parser.add_argument(
        "--area",
        type=str,
        default=None,
        help="Area identifier (optional)"
    )
    optimize_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate optimization without applying changes"
    )

    # Diagnose traps command
    traps_parser = subparsers.add_parser(
        "diagnose-traps",
        help="Diagnose steam traps"
    )
    traps_parser.add_argument(
        "--site",
        type=str,
        required=True,
        help="Site identifier"
    )
    traps_parser.add_argument(
        "--trap-ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific trap IDs to diagnose"
    )
    traps_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for report"
    )

    # Root cause analysis command
    rca_parser = subparsers.add_parser(
        "rca",
        help="Root cause analysis"
    )
    rca_parser.add_argument(
        "--deviation",
        type=str,
        required=True,
        help="Description of deviation to analyze"
    )
    rca_parser.add_argument(
        "--site",
        type=str,
        required=True,
        help="Site identifier"
    )
    rca_parser.add_argument(
        "--time-window",
        type=str,
        default="1h",
        help="Time window for analysis (e.g., 1h, 4h, 1d)"
    )

    # Health check command
    health_parser = subparsers.add_parser(
        "health",
        help="Check system health"
    )
    health_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed health information"
    )

    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show system status"
    )
    status_parser.add_argument(
        "--site",
        type=str,
        default=None,
        help="Site identifier (optional)"
    )

    # Export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export data or reports"
    )
    export_parser.add_argument(
        "--type",
        type=str,
        choices=["kpis", "recommendations", "audit", "climate-impact", "m&v"],
        required=True,
        help="Export type"
    )
    export_parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv", "pdf", "excel"],
        default="json",
        help="Export format"
    )
    export_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path"
    )
    export_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (ISO format)"
    )
    export_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (ISO format)"
    )

    return parser


async def cmd_start(args: argparse.Namespace) -> int:
    """Start the GL-003 server."""
    logger.info(f"Starting GL-003 UNIFIEDSTEAM server on {args.host}:{args.port}")
    logger.info(f"Operating mode: {args.mode}")
    logger.info(f"Workers: {args.workers}")

    try:
        # Import here to avoid circular imports
        from api.main import create_app
        import uvicorn

        app = create_app(mode=args.mode)

        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level.lower()
        )
        server = uvicorn.Server(config)
        await server.serve()
        return 0

    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Ensure all dependencies are installed: pip install -r requirements.txt")
        return 1
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1


async def cmd_compute(args: argparse.Namespace) -> int:
    """Compute steam properties."""
    logger.info(f"Computing properties for P={args.pressure} kPa, T={args.temperature} °C")

    try:
        from thermodynamics import compute_properties, detect_steam_state

        # Compute properties
        properties = compute_properties(
            pressure_kpa=args.pressure,
            temperature_c=args.temperature,
            quality=args.quality
        )

        steam_state = detect_steam_state(args.pressure, args.temperature)

        result = {
            "input": {
                "pressure_kpa": args.pressure,
                "temperature_c": args.temperature,
                "quality": args.quality
            },
            "steam_state": steam_state.value,
            "properties": {
                "specific_enthalpy_kj_kg": properties.enthalpy,
                "specific_entropy_kj_kg_k": properties.entropy,
                "specific_volume_m3_kg": properties.specific_volume,
                "density_kg_m3": properties.density,
                "superheat_degree_c": properties.superheat_degree,
                "dryness_fraction": properties.dryness_fraction
            },
            "provenance_hash": properties.provenance_hash,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        if args.output == "json":
            print(json.dumps(result, indent=2))
        elif args.output == "yaml":
            import yaml
            print(yaml.dump(result, default_flow_style=False))
        else:  # table
            print("\n" + "=" * 50)
            print("GL-003 UNIFIEDSTEAM - Steam Properties")
            print("=" * 50)
            print(f"Steam State: {steam_state.value}")
            print("-" * 50)
            print(f"{'Property':<30} {'Value':>18}")
            print("-" * 50)
            for key, value in result["properties"].items():
                if value is not None:
                    print(f"{key:<30} {value:>18.4f}")
            print("-" * 50)
            print(f"Provenance: {properties.provenance_hash[:16]}...")
            print("=" * 50 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Computation error: {e}")
        return 1


async def cmd_optimize(args: argparse.Namespace) -> int:
    """Run optimization."""
    logger.info(f"Running {args.type} optimization for site {args.site}")

    if args.dry_run:
        logger.info("DRY RUN - No changes will be applied")

    try:
        from optimization import (
            DesuperheaterOptimizer,
            CondensateRecoveryOptimizer,
            SteamNetworkOptimizer,
            TrapMaintenanceOptimizer
        )

        # Select optimizer based on type
        if args.type == "desuperheater":
            optimizer = DesuperheaterOptimizer(site_id=args.site)
        elif args.type == "condensate":
            optimizer = CondensateRecoveryOptimizer(site_id=args.site)
        elif args.type == "network":
            optimizer = SteamNetworkOptimizer(site_id=args.site)
        elif args.type == "traps":
            optimizer = TrapMaintenanceOptimizer(site_id=args.site)
        else:  # combined
            logger.info("Running combined optimization...")
            # Run all optimizers in sequence
            results = []
            for opt_class in [DesuperheaterOptimizer, CondensateRecoveryOptimizer,
                            SteamNetworkOptimizer, TrapMaintenanceOptimizer]:
                opt = opt_class(site_id=args.site)
                result = await opt.optimize(dry_run=args.dry_run)
                results.append(result)

            print(json.dumps({"combined_results": results}, indent=2))
            return 0

        result = await optimizer.optimize(dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
        return 0

    except Exception as e:
        logger.error(f"Optimization error: {e}")
        return 1


async def cmd_diagnose_traps(args: argparse.Namespace) -> int:
    """Diagnose steam traps."""
    logger.info(f"Diagnosing traps for site {args.site}")

    try:
        from calculators import TrapDiagnosticsCalculator

        calculator = TrapDiagnosticsCalculator(site_id=args.site)

        if args.trap_ids:
            results = await calculator.diagnose_traps(trap_ids=args.trap_ids)
        else:
            results = await calculator.diagnose_all_traps()

        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        else:
            print(json.dumps(results, indent=2))

        return 0

    except Exception as e:
        logger.error(f"Trap diagnostics error: {e}")
        return 1


async def cmd_rca(args: argparse.Namespace) -> int:
    """Root cause analysis."""
    logger.info(f"Running RCA for deviation: {args.deviation}")

    try:
        from causal import RootCauseAnalyzer

        analyzer = RootCauseAnalyzer(site_id=args.site)
        result = await analyzer.analyze(
            deviation=args.deviation,
            time_window=args.time_window
        )

        print(json.dumps(result, indent=2))
        return 0

    except Exception as e:
        logger.error(f"RCA error: {e}")
        return 1


async def cmd_health(args: argparse.Namespace) -> int:
    """Check system health."""
    try:
        from monitoring import HealthMonitor

        monitor = HealthMonitor()

        if args.detailed:
            health = await monitor.get_detailed_health()
        else:
            health = await monitor.get_overall_health()

        print(json.dumps(health, indent=2))
        return 0 if health.get("status") == "healthy" else 1

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return 1


async def cmd_status(args: argparse.Namespace) -> int:
    """Show system status."""
    try:
        from core import SteamSystemOrchestrator

        orchestrator = SteamSystemOrchestrator()
        status = await orchestrator.get_status(site_id=args.site)

        print(json.dumps(status, indent=2))
        return 0

    except Exception as e:
        logger.error(f"Status error: {e}")
        return 1


async def cmd_export(args: argparse.Namespace) -> int:
    """Export data or reports."""
    logger.info(f"Exporting {args.type} to {args.output} as {args.format}")

    try:
        from audit import ComplianceReporter

        reporter = ComplianceReporter()

        export_result = await reporter.export(
            export_type=args.type,
            format=args.format,
            output_path=args.output,
            start_date=args.start_date,
            end_date=args.end_date
        )

        logger.info(f"Export completed: {export_result}")
        return 0

    except Exception as e:
        logger.error(f"Export error: {e}")
        return 1


async def main() -> int:
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Route to appropriate command handler
    if args.command == "start":
        return await cmd_start(args)
    elif args.command == "compute":
        return await cmd_compute(args)
    elif args.command == "optimize":
        return await cmd_optimize(args)
    elif args.command == "diagnose-traps":
        return await cmd_diagnose_traps(args)
    elif args.command == "rca":
        return await cmd_rca(args)
    elif args.command == "health":
        return await cmd_health(args)
    elif args.command == "status":
        return await cmd_status(args)
    elif args.command == "export":
        return await cmd_export(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
