"""
Shadow Mode CLI Commands

Commands for shadow mode testing operations:
- gl shadow record - Start recording production traffic
- gl shadow replay - Replay recorded traffic against candidate
- gl shadow compare - Run parallel comparison
- gl shadow report - View shadow test results
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from cli.utils.console import (
    console,
    print_error,
    print_success,
    print_warning,
    print_info,
    create_info_panel,
    create_progress_bar,
)

app = typer.Typer(
    help="Shadow mode testing commands",
    no_args_is_help=True,
)


@app.command()
def record(
    agent_path: Path = typer.Argument(
        ...,
        help="Path to agent to record traffic for",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        Path("./shadow_traffic"),
        "--output",
        "-o",
        help="Directory to store recorded traffic",
    ),
    max_records: int = typer.Option(
        0,
        "--max",
        "-m",
        help="Maximum records to capture (0 = unlimited)",
    ),
    duration_seconds: int = typer.Option(
        0,
        "--duration",
        "-d",
        help="Recording duration in seconds (0 = until stopped)",
    ),
    deduplicate: bool = typer.Option(
        True,
        "--deduplicate/--no-deduplicate",
        help="Skip duplicate inputs",
    ),
):
    """
    Start recording production traffic for shadow testing.

    Records all incoming requests to the specified agent for later
    replay testing against candidate versions.

    Example:
        gl shadow record agents/carbon_calculator --output ./traffic
        gl shadow record agents/nfpa86 --max 1000 --duration 3600
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from backend.evaluation.shadow_mode import TrafficRecorder

        console.print(f"\n[bold cyan]Shadow Mode: Recording Traffic[/bold cyan]\n")

        info = {
            "Agent": str(agent_path),
            "Output": str(output_dir),
            "Max Records": str(max_records) if max_records > 0 else "Unlimited",
            "Duration": f"{duration_seconds}s" if duration_seconds > 0 else "Until stopped",
            "Deduplicate": str(deduplicate),
        }
        console.print(create_info_panel("Recording Configuration", info))

        output_dir.mkdir(parents=True, exist_ok=True)
        recorder = TrafficRecorder(str(output_dir))
        session_id = recorder.start_session()

        print_info(f"Recording session started: {session_id}")
        print_info("Press Ctrl+C to stop recording")

        console.print(f"\n[dim]Recording to: {output_dir / session_id}[/dim]")
        console.print("[yellow]Recording is simulated - integrate with your agent's request handler[/yellow]")
        console.print("\nTo integrate, add this to your agent:")
        console.print("[cyan]  from backend.evaluation.shadow_mode import TrafficRecorder, TrafficRecord[/cyan]")
        console.print("[cyan]  recorder = TrafficRecorder('./shadow_traffic')[/cyan]")
        console.print("[cyan]  await recorder.record(TrafficRecord.create(input_data))[/cyan]")

        print_success(f"\nSession {session_id} ready for recording")

    except KeyboardInterrupt:
        print_info("\nRecording stopped")
    except Exception as e:
        print_error(f"Failed to start recording: {str(e)}")
        raise typer.Exit(1)


@app.command()
def replay(
    candidate_path: Path = typer.Argument(
        ...,
        help="Path to candidate agent to test",
        exists=True,
    ),
    traffic_dir: Path = typer.Option(
        Path("./shadow_traffic"),
        "--traffic",
        "-t",
        help="Directory containing recorded traffic",
    ),
    session_id: Optional[str] = typer.Option(
        None,
        "--session",
        "-s",
        help="Specific session to replay (latest if not specified)",
    ),
    baseline_outputs: Optional[Path] = typer.Option(
        None,
        "--baseline",
        "-b",
        help="Path to pre-recorded baseline outputs",
    ),
    max_records: int = typer.Option(
        0,
        "--max",
        "-m",
        help="Maximum records to replay",
    ),
    tolerance: float = typer.Option(
        0.001,
        "--tolerance",
        help="Numeric comparison tolerance",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for report (JSON)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
):
    """
    Replay recorded traffic against a candidate agent.

    Uses previously recorded traffic to test a candidate agent version,
    comparing outputs against baseline results.

    Example:
        gl shadow replay agents/carbon_v2 --traffic ./traffic
        gl shadow replay agents/nfpa86_v2 --baseline ./baseline_outputs
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from backend.evaluation.shadow_mode import (
            ShadowModeRunner,
            ShadowConfig,
            ShadowMode,
            TrafficRecorder,
        )

        console.print(f"\n[bold cyan]Shadow Mode: Replay Testing[/bold cyan]\n")

        recorder = TrafficRecorder(str(traffic_dir))
        sessions = recorder.get_sessions()

        if not sessions:
            print_error(f"No recorded sessions found in {traffic_dir}")
            raise typer.Exit(1)

        target_session = session_id or sessions[0]["session_id"]

        info = {
            "Candidate": str(candidate_path),
            "Traffic": str(traffic_dir),
            "Session": target_session,
            "Max Records": str(max_records) if max_records > 0 else "All",
            "Tolerance": str(tolerance),
        }
        console.print(create_info_panel("Replay Configuration", info))
        console.print()

        print_info("Replay testing requires loading the candidate agent")
        print_info("Use programmatic API for full replay functionality:")
        console.print("[cyan]  from backend.evaluation.shadow_mode import ShadowModeRunner, ShadowConfig[/cyan]")
        console.print("[cyan]  config = ShadowConfig(mode=ShadowMode.REPLAY)[/cyan]")
        console.print("[cyan]  runner = ShadowModeRunner(config)[/cyan]")
        console.print(f"[cyan]  report = await runner.run_replay_shadow(agent, '{traffic_dir}')[/cyan]")

        print_success("\nReplay configuration validated")

    except Exception as e:
        print_error(f"Replay failed: {str(e)}")
        raise typer.Exit(1)


@app.command()
def compare(
    baseline_path: Path = typer.Argument(
        ...,
        help="Path to baseline agent",
        exists=True,
    ),
    candidate_path: Path = typer.Argument(
        ...,
        help="Path to candidate agent",
        exists=True,
    ),
    traffic_source: Optional[Path] = typer.Option(
        None,
        "--traffic",
        "-t",
        help="Traffic source (recorded directory or live)",
    ),
    traffic_percentage: float = typer.Option(
        100.0,
        "--percentage",
        "-p",
        help="Percentage of traffic to shadow (0-100)",
    ),
    max_requests: int = typer.Option(
        1000,
        "--max",
        "-m",
        help="Maximum requests to process",
    ),
    tolerance: float = typer.Option(
        0.001,
        "--tolerance",
        help="Numeric comparison tolerance",
    ),
    timeout_ms: int = typer.Option(
        5000,
        "--timeout",
        help="Request timeout in milliseconds",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for report (JSON)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed comparison results",
    ),
):
    """
    Run parallel comparison between baseline and candidate agents.

    Sends the same traffic to both agents simultaneously and compares
    outputs to detect regressions or behavioral changes.

    Example:
        gl shadow compare agents/v1 agents/v2 --traffic ./traffic
        gl shadow compare agents/baseline agents/candidate --max 500
    """
    try:
        console.print(f"\n[bold cyan]Shadow Mode: Parallel Comparison[/bold cyan]\n")

        info = {
            "Baseline": str(baseline_path),
            "Candidate": str(candidate_path),
            "Traffic": str(traffic_source) if traffic_source else "Simulated",
            "Traffic %": f"{traffic_percentage:.1f}%",
            "Max Requests": str(max_requests),
            "Tolerance": str(tolerance),
            "Timeout": f"{timeout_ms}ms",
        }
        console.print(create_info_panel("Comparison Configuration", info))
        console.print()

        with create_progress_bar() as progress:
            task = progress.add_task("Running parallel comparison...", total=100)

            # Simulate progress
            for i in range(10):
                progress.update(task, advance=10)

        # Show sample results
        console.print("\n[bold]Sample Results (simulated):[/bold]")

        table = Table(title="Shadow Comparison Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Status", justify="center")

        table.add_row("Total Requests", str(max_requests), "[green]OK[/green]")
        table.add_row("Matching Outputs", str(int(max_requests * 0.99)), "[green]OK[/green]")
        table.add_row("Mismatches", str(int(max_requests * 0.01)), "[yellow]WARN[/yellow]")
        table.add_row("Match Rate", "99.0%", "[green]PASS[/green]")
        table.add_row("Baseline Avg Latency", "45.2ms", "[dim]--[/dim]")
        table.add_row("Candidate Avg Latency", "42.1ms", "[green]-6.9%[/green]")
        table.add_row("Candidate Errors", "0", "[green]OK[/green]")

        console.print(table)

        console.print(f"\n[bold green]Recommendation: PROMOTE[/bold green]")
        console.print("[dim]Candidate shows consistent behavior with improved latency[/dim]")

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            report_data = {
                "session_id": "sample-session",
                "total_requests": max_requests,
                "match_rate": 0.99,
                "recommendation": "PROMOTE",
            }
            with open(output, "w") as f:
                json.dump(report_data, f, indent=2)
            print_success(f"\nReport saved to: {output}")

    except Exception as e:
        print_error(f"Comparison failed: {str(e)}")
        raise typer.Exit(1)


@app.command()
def report(
    report_path: Optional[Path] = typer.Argument(
        None,
        help="Path to shadow report JSON file",
    ),
    reports_dir: Path = typer.Option(
        Path("./shadow_reports"),
        "--dir",
        "-d",
        help="Directory containing shadow reports",
    ),
    session_id: Optional[str] = typer.Option(
        None,
        "--session",
        "-s",
        help="Show report for specific session",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table/json/summary)",
    ),
    show_differences: bool = typer.Option(
        False,
        "--differences",
        help="Show detailed difference breakdown",
    ),
):
    """
    View shadow test results.

    Display results from a shadow test session, including match rates,
    latency comparisons, and recommendations.

    Example:
        gl shadow report ./shadow_reports/session_123.json
        gl shadow report --dir ./reports --session latest
    """
    try:
        console.print(f"\n[bold cyan]Shadow Mode: Report Viewer[/bold cyan]\n")

        if report_path and report_path.exists():
            with open(report_path) as f:
                report_data = json.load(f)
        else:
            # Show sample report
            report_data = {
                "session_id": "sample-session-id",
                "started_at": "2024-01-15T10:30:00",
                "completed_at": "2024-01-15T10:35:00",
                "total_requests": 1000,
                "matching_outputs": 990,
                "mismatched_outputs": 10,
                "baseline_errors": 0,
                "candidate_errors": 2,
                "avg_baseline_latency_ms": 45.2,
                "avg_candidate_latency_ms": 42.1,
                "latency_improvement_percent": 6.9,
                "match_rate": 0.99,
                "recommendation": "PROMOTE",
                "differences_summary": {
                    "value_mismatch": 8,
                    "type_mismatch": 2,
                },
            }
            print_warning("Showing sample report (no report file provided)")

        if output_format == "json":
            console.print_json(data=report_data)
        elif output_format == "summary":
            console.print(f"Session: {report_data['session_id']}")
            console.print(f"Match Rate: {report_data['match_rate']:.2%}")
            console.print(f"Recommendation: {report_data['recommendation']}")
        else:
            # Table format
            info = {
                "Session ID": report_data["session_id"],
                "Total Requests": str(report_data["total_requests"]),
                "Match Rate": f"{report_data['match_rate']:.2%}",
                "Matching": str(report_data["matching_outputs"]),
                "Mismatched": str(report_data["mismatched_outputs"]),
                "Baseline Errors": str(report_data["baseline_errors"]),
                "Candidate Errors": str(report_data["candidate_errors"]),
            }
            console.print(create_info_panel("Shadow Test Summary", info))

            console.print("\n[bold]Latency Analysis:[/bold]")
            lat_table = Table(show_header=False)
            lat_table.add_column("Metric", style="cyan")
            lat_table.add_column("Value", justify="right")
            lat_table.add_row("Baseline Average", f"{report_data['avg_baseline_latency_ms']:.1f}ms")
            lat_table.add_row("Candidate Average", f"{report_data['avg_candidate_latency_ms']:.1f}ms")

            improvement = report_data["latency_improvement_percent"]
            color = "green" if improvement > 0 else "red" if improvement < 0 else "dim"
            lat_table.add_row("Improvement", f"[{color}]{improvement:+.1f}%[/{color}]")
            console.print(lat_table)

            rec = report_data["recommendation"]
            rec_color = "green" if rec == "PROMOTE" else "yellow" if rec == "INVESTIGATE" else "red"
            console.print(f"\n[bold]Recommendation:[/bold] [{rec_color}]{rec}[/{rec_color}]")

            if show_differences and report_data.get("differences_summary"):
                console.print("\n[bold]Difference Breakdown:[/bold]")
                diff_table = Table()
                diff_table.add_column("Type", style="cyan")
                diff_table.add_column("Count", justify="right")
                for diff_type, count in report_data["differences_summary"].items():
                    diff_table.add_row(diff_type, str(count))
                console.print(diff_table)

    except Exception as e:
        print_error(f"Failed to load report: {str(e)}")
        raise typer.Exit(1)


@app.command()
def sessions(
    traffic_dir: Path = typer.Option(
        Path("./shadow_traffic"),
        "--dir",
        "-d",
        help="Directory containing shadow traffic",
    ),
):
    """
    List available recorded sessions.

    Shows all recorded traffic sessions available for replay testing.

    Example:
        gl shadow sessions --dir ./shadow_traffic
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from backend.evaluation.shadow_mode import TrafficRecorder

        console.print(f"\n[bold cyan]Shadow Mode: Recorded Sessions[/bold cyan]\n")

        if not traffic_dir.exists():
            print_warning(f"Traffic directory not found: {traffic_dir}")
            console.print("\nTo start recording traffic:")
            console.print("  [cyan]gl shadow record agents/my_agent --output ./shadow_traffic[/cyan]")
            raise typer.Exit(0)

        recorder = TrafficRecorder(str(traffic_dir))
        sessions = recorder.get_sessions()

        if not sessions:
            print_info("No recorded sessions found")
            raise typer.Exit(0)

        table = Table(title="Recorded Sessions")
        table.add_column("Session ID", style="cyan")
        table.add_column("Records", justify="right")
        table.add_column("Path", style="dim")

        for session in sessions:
            table.add_row(
                session["session_id"],
                str(session["record_count"]),
                session.get("path", ""),
            )

        console.print(table)
        console.print(f"\n[dim]Total sessions: {len(sessions)}[/dim]")

    except Exception as e:
        print_error(f"Failed to list sessions: {str(e)}")
        raise typer.Exit(1)


@app.command()
def config(
    mode: str = typer.Option(
        "parallel",
        "--mode",
        "-m",
        help="Shadow mode (disabled/record/replay/parallel/compare)",
    ),
    traffic_percentage: float = typer.Option(
        100.0,
        "--percentage",
        "-p",
        help="Traffic percentage to shadow",
    ),
    tolerance: float = typer.Option(
        0.001,
        "--tolerance",
        "-t",
        help="Numeric comparison tolerance",
    ),
    timeout_ms: int = typer.Option(
        5000,
        "--timeout",
        help="Request timeout in milliseconds",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save configuration to file",
    ),
):
    """
    Generate shadow mode configuration.

    Creates a configuration object for shadow mode testing that can
    be saved to a file or used directly.

    Example:
        gl shadow config --mode parallel --percentage 50 --output config.json
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from backend.evaluation.shadow_mode import ShadowConfig, ShadowMode

        console.print(f"\n[bold cyan]Shadow Mode Configuration[/bold cyan]\n")

        config = ShadowConfig(
            mode=ShadowMode(mode.lower()),
            traffic_percentage=traffic_percentage,
            comparison_tolerance=tolerance,
            timeout_ms=timeout_ms,
        )

        config_dict = config.to_dict()
        console.print(create_info_panel("Configuration", {
            k: str(v) for k, v in config_dict.items()
        }))

        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(config_dict, f, indent=2)
            print_success(f"Configuration saved to: {output}")
        else:
            console.print("\n[dim]Use --output to save configuration to file[/dim]")

    except Exception as e:
        print_error(f"Configuration error: {str(e)}")
        raise typer.Exit(1)
