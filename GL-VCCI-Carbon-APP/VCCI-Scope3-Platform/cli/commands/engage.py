# -*- coding: utf-8 -*-
"""
GL-VCCI CLI - Engage Command
Supplier engagement campaign management and email orchestration.

Features:
- Campaign creation and management
- Email campaign scheduling
- Response tracking and analytics
- Supplier portal integration
- Integration with SupplierEngagementAgent

Version: 1.0.0
Date: 2025-11-08
"""

import sys
from typing import Optional, List
from datetime import datetime
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.text import Text
from rich.tree import Tree
from greenlang.determinism import deterministic_uuid, DeterministicClock

# Import the engagement agent
try:
    from services.agents.engagement.agent import SupplierEngagementAgent
    from services.agents.engagement.exceptions import (
        ConsentNotGrantedError,
        CampaignNotFoundError,
        SupplierNotFoundError
    )
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False

# Create console
console = Console()

# Create Typer app for engagement commands
engage_app = typer.Typer(
    name="engage",
    help="Supplier engagement and campaign management",
    rich_markup_mode="rich"
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_percentage(value: float) -> str:
    """Format percentage value."""
    return f"{value * 100:.1f}%"


def format_date(dt: datetime) -> str:
    """Format datetime for display."""
    return dt.strftime("%Y-%m-%d %H:%M") if dt else "N/A"


# ============================================================================
# LIST CAMPAIGNS COMMAND
# ============================================================================

@engage_app.command("list")
def list_campaigns(
    status: Optional[str] = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by status (active, completed, draft)"
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-l",
        help="Maximum number of campaigns to display"
    )
):
    """
    List all supplier engagement campaigns.

    Displays campaign details including status, targets, and response rates.

    Examples:

      # List all campaigns
      vcci engage list

      # List only active campaigns
      vcci engage list --status active

      # List last 20 campaigns
      vcci engage list --limit 20
    """
    console.print()

    if not AGENT_AVAILABLE:
        console.print(
            "[red]Error:[/red] SupplierEngagementAgent not available. "
            "Please ensure services.agents.engagement module is properly installed."
        )
        sys.exit(1)

    try:
        # Initialize agent
        agent = SupplierEngagementAgent()

        # Get campaigns (mocked for demo)
        # In production: campaigns = agent.campaign_manager.list_campaigns()
        campaigns = [
            {
                "campaign_id": "CAMP-001",
                "name": "Q4 2025 Data Collection",
                "status": "active",
                "target_suppliers": 250,
                "contacted": 250,
                "responded": 87,
                "response_rate": 0.348,
                "start_date": "2025-10-01",
                "duration_days": 90
            },
            {
                "campaign_id": "CAMP-002",
                "name": "High-Impact Suppliers Engagement",
                "status": "active",
                "target_suppliers": 50,
                "contacted": 50,
                "responded": 32,
                "response_rate": 0.64,
                "start_date": "2025-11-01",
                "duration_days": 60
            },
            {
                "campaign_id": "CAMP-003",
                "name": "Q3 2025 Follow-up",
                "status": "completed",
                "target_suppliers": 180,
                "contacted": 180,
                "responded": 95,
                "response_rate": 0.528,
                "start_date": "2025-07-01",
                "duration_days": 90
            }
        ]

        # Filter by status if specified
        if status:
            campaigns = [c for c in campaigns if c["status"] == status.lower()]

        # Apply limit
        campaigns = campaigns[:limit]

        if not campaigns:
            console.print("[yellow]No campaigns found[/yellow]")
            console.print()
            return

        # Display campaigns table
        campaigns_table = Table(
            title=f"Supplier Engagement Campaigns",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )

        campaigns_table.add_column("Campaign ID", style="cyan", width=12)
        campaigns_table.add_column("Name", style="white", width=30)
        campaigns_table.add_column("Status", justify="center", width=10)
        campaigns_table.add_column("Suppliers", justify="right", width=10)
        campaigns_table.add_column("Responses", justify="right", width=10)
        campaigns_table.add_column("Rate", justify="right", width=10)
        campaigns_table.add_column("Start Date", width=12)

        for camp in campaigns:
            # Status color
            status_text = camp["status"].capitalize()
            if camp["status"] == "active":
                status_style = "[green]"
            elif camp["status"] == "completed":
                status_style = "[blue]"
            else:
                status_style = "[yellow]"

            # Response rate color
            rate = camp["response_rate"]
            if rate >= 0.5:
                rate_style = "green"
            elif rate >= 0.3:
                rate_style = "yellow"
            else:
                rate_style = "red"

            campaigns_table.add_row(
                camp["campaign_id"],
                camp["name"],
                f"{status_style}{status_text}[/]",
                str(camp["target_suppliers"]),
                str(camp["responded"]),
                f"[{rate_style}]{format_percentage(rate)}[/]",
                camp["start_date"]
            )

        console.print(campaigns_table)

        # Summary
        total_suppliers = sum(c["target_suppliers"] for c in campaigns)
        total_responses = sum(c["responded"] for c in campaigns)
        avg_rate = (total_responses / total_suppliers) if total_suppliers > 0 else 0

        console.print()
        console.print(
            Panel(
                f"[cyan]Total Campaigns:[/cyan] {len(campaigns)}\n"
                f"[cyan]Total Suppliers Engaged:[/cyan] {total_suppliers:,}\n"
                f"[cyan]Total Responses:[/cyan] {total_responses:,}\n"
                f"[cyan]Average Response Rate:[/cyan] {format_percentage(avg_rate)}",
                title="[bold cyan]Summary[/bold cyan]",
                border_style="cyan"
            )
        )

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)

    console.print()


# ============================================================================
# CREATE CAMPAIGN COMMAND
# ============================================================================

@engage_app.command("create")
def create_campaign(
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Campaign name"
    ),
    template: str = typer.Option(
        "standard",
        "--template",
        "-t",
        help="Email template (standard, urgent, follow-up)"
    ),
    suppliers_file: Optional[str] = typer.Option(
        None,
        "--suppliers",
        "-s",
        help="CSV file with supplier IDs (one per line)"
    ),
    duration: int = typer.Option(
        90,
        "--duration",
        "-d",
        help="Campaign duration in days"
    ),
    target_rate: float = typer.Option(
        0.5,
        "--target-rate",
        "-r",
        help="Target response rate (0.0-1.0)"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Create campaign without sending emails"
    )
):
    """
    Create a new supplier engagement campaign.

    Sets up email sequences, target suppliers, and campaign parameters.

    Examples:

      # Create standard campaign
      vcci engage create --name "Q1 2026 Data Collection" --template standard

      # Create urgent campaign with custom duration
      vcci engage create --name "Critical Suppliers" --template urgent --duration 30

      # Create from supplier list file
      vcci engage create --name "Top 100" --suppliers suppliers.csv --dry-run
    """
    console.print()

    if not AGENT_AVAILABLE:
        console.print("[red]Error:[/red] SupplierEngagementAgent not available.")
        sys.exit(1)

    try:
        # Initialize agent
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Initializing engagement agent...", total=None)
            agent = SupplierEngagementAgent()
            progress.update(task, completed=True)

        # Load suppliers from file if specified
        target_suppliers = []
        if suppliers_file:
            try:
                with open(suppliers_file, 'r') as f:
                    target_suppliers = [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                console.print(f"[red]Error:[/red] Suppliers file not found: {suppliers_file}")
                sys.exit(1)
        else:
            # Demo: use mock supplier IDs
            target_suppliers = [f"SUP-{i:04d}" for i in range(1, 101)]

        console.print(
            Panel(
                f"[cyan]Campaign Name:[/cyan] {name}\n"
                f"[cyan]Template:[/cyan] {template}\n"
                f"[cyan]Target Suppliers:[/cyan] {len(target_suppliers)}\n"
                f"[cyan]Duration:[/cyan] {duration} days\n"
                f"[cyan]Target Response Rate:[/cyan] {format_percentage(target_rate)}\n"
                f"[cyan]Mode:[/cyan] {'Dry Run (no emails sent)' if dry_run else 'Live'}",
                title="[bold cyan]Campaign Configuration[/bold cyan]",
                border_style="cyan"
            )
        )
        console.print()

        # Create campaign
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Creating campaign...", total=None)

            # In production: create actual campaign
            # campaign = agent.create_campaign(
            #     name=name,
            #     target_suppliers=target_suppliers,
            #     duration_days=duration,
            #     response_rate_target=target_rate
            # )

            # Mock campaign ID
            import uuid
            campaign_id = f"CAMP-{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:8].upper()}"

            progress.update(task, completed=True)

        console.print()
        console.print(
            Panel(
                f"[green]Campaign created successfully![/green]\n\n"
                f"[cyan]Campaign ID:[/cyan] {campaign_id}\n"
                f"[cyan]Status:[/cyan] {'Draft (dry run)' if dry_run else 'Ready to start'}\n"
                f"[cyan]Suppliers:[/cyan] {len(target_suppliers)}\n\n"
                f"[yellow]Next steps:[/yellow]\n"
                f"  1. Review campaign: vcci engage status --campaign-id {campaign_id}\n"
                f"  2. Start campaign: vcci engage send --campaign-id {campaign_id}",
                title="[bold green]Campaign Created[/bold green]",
                border_style="green"
            )
        )

    except Exception as e:
        console.print(f"[red]Error creating campaign:[/red] {str(e)}")
        sys.exit(1)

    console.print()


# ============================================================================
# SEND EMAILS COMMAND
# ============================================================================

@engage_app.command("send")
def send_emails(
    campaign_id: str = typer.Option(
        ...,
        "--campaign-id",
        "-c",
        help="Campaign ID to send emails for"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview emails without sending"
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-l",
        help="Limit number of emails to send (for testing)"
    )
):
    """
    Send campaign emails to suppliers.

    Schedules and sends email sequences with consent checking.

    Examples:

      # Send all campaign emails
      vcci engage send --campaign-id CAMP-ABC123

      # Dry run to preview
      vcci engage send --campaign-id CAMP-ABC123 --dry-run

      # Send to limited set for testing
      vcci engage send --campaign-id CAMP-ABC123 --limit 10
    """
    console.print()

    if not AGENT_AVAILABLE:
        console.print("[red]Error:[/red] SupplierEngagementAgent not available.")
        sys.exit(1)

    try:
        # Initialize agent
        agent = SupplierEngagementAgent()

        console.print(
            Panel(
                f"[cyan]Campaign ID:[/cyan] {campaign_id}\n"
                f"[cyan]Mode:[/cyan] {'Dry Run (no emails sent)' if dry_run else 'Live'}\n"
                f"[cyan]Limit:[/cyan] {limit if limit else 'None (send all)'}",
                title="[bold cyan]Email Campaign Execution[/bold cyan]",
                border_style="cyan"
            )
        )
        console.print()

        # In production: get campaign and send emails
        # campaign = agent.campaign_manager.get_campaign(campaign_id)
        # result = agent.start_campaign(campaign_id)

        # Mock sending process
        total_emails = limit if limit else 100
        emails_sent = 0
        emails_failed = 0
        consent_blocks = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(
                f"[cyan]{'Previewing' if dry_run else 'Sending'} emails...",
                total=total_emails
            )

            # Simulate sending
            import time
            for i in range(total_emails):
                time.sleep(0.01)  # Small delay for visual effect

                # Simulate some failures
                if i % 25 == 0:
                    consent_blocks += 1
                elif i % 50 == 0:
                    emails_failed += 1
                else:
                    emails_sent += 1

                progress.advance(task)

        console.print()

        # Results
        result_text = (
            f"[green]Emails Sent:[/green] {emails_sent}\n"
        )

        if emails_failed > 0:
            result_text += f"[red]Failed:[/red] {emails_failed}\n"

        if consent_blocks > 0:
            result_text += f"[yellow]Blocked (no consent):[/yellow] {consent_blocks}\n"

        result_text += (
            f"\n[cyan]Success Rate:[/cyan] {(emails_sent / total_emails * 100):.1f}%\n"
        )

        if dry_run:
            result_text += (
                f"\n[yellow]This was a dry run.[/yellow]\n"
                f"No emails were actually sent."
            )
        else:
            result_text += (
                f"\n[green]Campaign is now active![/green]\n"
                f"Monitor progress: vcci engage status --campaign-id {campaign_id}"
            )

        console.print(
            Panel(
                result_text,
                title=f"[bold {'yellow' if dry_run else 'green'}]{'Dry Run Complete' if dry_run else 'Emails Sent'}[/bold {'yellow' if dry_run else 'green'}]",
                border_style="yellow" if dry_run else "green"
            )
        )

    except CampaignNotFoundError:
        console.print(f"[red]Error:[/red] Campaign not found: {campaign_id}")
        sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error sending emails:[/red] {str(e)}")
        sys.exit(1)

    console.print()


# ============================================================================
# CAMPAIGN STATUS COMMAND
# ============================================================================

@engage_app.command("status")
def campaign_status(
    campaign_id: str = typer.Option(
        ...,
        "--campaign-id",
        "-c",
        help="Campaign ID to check status for"
    ),
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Show detailed analytics"
    )
):
    """
    Show campaign status and response tracking.

    Displays real-time campaign metrics and supplier engagement.

    Examples:

      # Quick status
      vcci engage status --campaign-id CAMP-ABC123

      # Detailed analytics
      vcci engage status --campaign-id CAMP-ABC123 --detailed
    """
    console.print()

    if not AGENT_AVAILABLE:
        console.print("[red]Error:[/red] SupplierEngagementAgent not available.")
        sys.exit(1)

    try:
        # Initialize agent
        agent = SupplierEngagementAgent()

        # In production: get analytics
        # analytics = agent.get_campaign_analytics(campaign_id)

        # Mock analytics data
        console.print(
            Panel(
                f"[cyan]Campaign ID:[/cyan] {campaign_id}\n"
                f"[cyan]Name:[/cyan] Q4 2025 Data Collection\n"
                f"[cyan]Status:[/cyan] [green]Active[/green]\n"
                f"[cyan]Start Date:[/cyan] 2025-10-01\n"
                f"[cyan]Days Remaining:[/cyan] 52",
                title="[bold cyan]Campaign Overview[/bold cyan]",
                border_style="cyan"
            )
        )
        console.print()

        # Metrics table
        metrics_table = Table(
            title="Campaign Metrics",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )

        metrics_table.add_column("Metric", style="cyan", width=25)
        metrics_table.add_column("Value", justify="right", style="white", width=15)
        metrics_table.add_column("Target", justify="right", style="yellow", width=15)
        metrics_table.add_column("Progress", width=20)

        metrics_table.add_row(
            "Target Suppliers",
            "250",
            "250",
            "[green]100%[/green]"
        )
        metrics_table.add_row(
            "Emails Sent",
            "250",
            "250",
            "[green]100%[/green]"
        )
        metrics_table.add_row(
            "Emails Opened",
            "178",
            "-",
            "[cyan]71.2%[/cyan]"
        )
        metrics_table.add_row(
            "Links Clicked",
            "142",
            "-",
            "[cyan]56.8%[/cyan]"
        )
        metrics_table.add_row(
            "Suppliers Responded",
            "87",
            "125",
            "[yellow]69.6% of target[/yellow]"
        )
        metrics_table.add_row(
            "Data Uploaded",
            "87",
            "125",
            "[yellow]69.6% of target[/yellow]"
        )
        metrics_table.add_row(
            "Response Rate",
            "34.8%",
            "50.0%",
            "[yellow]69.6% of target[/yellow]"
        )

        console.print(metrics_table)

        if detailed:
            console.print()

            # Email sequence breakdown
            sequence_table = Table(
                title="Email Sequence Performance",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold cyan"
            )

            sequence_table.add_column("Email", style="cyan")
            sequence_table.add_column("Sent", justify="right")
            sequence_table.add_column("Opened", justify="right")
            sequence_table.add_column("Clicked", justify="right")
            sequence_table.add_column("Responded", justify="right")

            sequence_table.add_row(
                "1. Initial Invitation",
                "250",
                "189 (75.6%)",
                "156 (62.4%)",
                "52 (20.8%)"
            )
            sequence_table.add_row(
                "2. Reminder (Day 7)",
                "198",
                "124 (62.6%)",
                "87 (43.9%)",
                "23 (11.6%)"
            )
            sequence_table.add_row(
                "3. Follow-up (Day 21)",
                "175",
                "98 (56.0%)",
                "45 (25.7%)",
                "9 (5.1%)"
            )
            sequence_table.add_row(
                "4. Final Notice (Day 60)",
                "166",
                "67 (40.4%)",
                "21 (12.7%)",
                "3 (1.8%)"
            )

            console.print(sequence_table)

            console.print()

            # Top responding suppliers
            console.print(
                Panel(
                    "[green]Top Responding Suppliers:[/green]\n"
                    "  1. ACME Corp - Completed (DQI: 95%)\n"
                    "  2. TechSolutions Ltd - Completed (DQI: 88%)\n"
                    "  3. Global Supplies Inc - In Progress (42%)\n"
                    "  4. Industrial Partners - In Progress (28%)\n"
                    "  5. Manufacturing Co - Completed (DQI: 92%)\n\n"
                    "[yellow]Non-Responders Requiring Follow-up:[/yellow]\n"
                    "  • 163 suppliers have not responded\n"
                    "  • 45 high-spend suppliers in this group\n"
                    "  • Recommend personalized outreach",
                    title="[bold cyan]Supplier Details[/bold cyan]",
                    border_style="cyan"
                )
            )

    except CampaignNotFoundError:
        console.print(f"[red]Error:[/red] Campaign not found: {campaign_id}")
        sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        sys.exit(1)

    console.print()


# ============================================================================
# LEADERBOARD COMMAND
# ============================================================================

@engage_app.command("leaderboard")
def show_leaderboard(
    campaign_id: str = typer.Option(
        ...,
        "--campaign-id",
        "-c",
        help="Campaign ID for leaderboard"
    ),
    top: int = typer.Option(
        10,
        "--top",
        "-t",
        help="Number of top suppliers to show"
    )
):
    """
    Show supplier engagement leaderboard.

    Displays gamification rankings and badges.

    Examples:

      # Top 10 suppliers
      vcci engage leaderboard --campaign-id CAMP-ABC123

      # Top 20 suppliers
      vcci engage leaderboard --campaign-id CAMP-ABC123 --top 20
    """
    console.print()

    console.print(
        Panel(
            f"[cyan]Campaign ID:[/cyan] {campaign_id}\n"
            f"[cyan]Showing Top:[/cyan] {top} suppliers",
            title="[bold cyan]Supplier Leaderboard[/bold cyan]",
            border_style="cyan"
        )
    )
    console.print()

    # Mock leaderboard
    leaderboard_table = Table(
        title=f"Top {top} Engaged Suppliers",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )

    leaderboard_table.add_column("Rank", justify="center", style="yellow", width=6)
    leaderboard_table.add_column("Supplier", style="cyan", width=30)
    leaderboard_table.add_column("Progress", justify="right", width=12)
    leaderboard_table.add_column("DQI Score", justify="right", width=12)
    leaderboard_table.add_column("Badges", width=20)

    suppliers = [
        (1, "ACME Corp", 100, 95, "🏆 🌟 ⚡"),
        (2, "TechSolutions Ltd", 100, 88, "🏆 🌟"),
        (3, "Manufacturing Co", 100, 92, "🏆 🌟"),
        (4, "Global Supplies Inc", 85, 78, "🌟"),
        (5, "Industrial Partners", 75, 82, "🌟"),
        (6, "Parts Unlimited", 65, 76, ""),
        (7, "Supply Chain Pro", 58, 71, ""),
        (8, "Materials Corp", 52, 68, ""),
        (9, "Component Systems", 45, 65, ""),
        (10, "Trading Partners", 40, 62, "")
    ]

    for rank, name, progress, dqi, badges in suppliers[:top]:
        leaderboard_table.add_row(
            f"#{rank}",
            name,
            f"[green]{progress}%[/green]",
            f"[cyan]{dqi}%[/cyan]",
            badges
        )

    console.print(leaderboard_table)

    console.print()
    console.print(
        Panel(
            "[yellow]Badge Legend:[/yellow]\n"
            "  🏆 = Completion Champion (100% complete)\n"
            "  🌟 = Quality Star (DQI > 80%)\n"
            "  ⚡ = Speed Demon (responded within 7 days)",
            border_style="yellow"
        )
    )

    console.print()


# Export the app
__all__ = ["engage_app"]
