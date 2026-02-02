"""
Rich Console Utilities

Provides styled console output using Rich library.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
from rich.markdown import Markdown


# Create a global console instance
console = Console()


def print_banner():
    """Print the GreenLang Agent Factory banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     🌿 GreenLang Agent Factory CLI                       ║
    ║     Build Production-Grade AI Agents                     ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold green")


def print_error(message: str, details: Optional[str] = None):
    """Print an error message with optional details."""
    console.print(f"[bold red]✗[/bold red] {message}")
    if details:
        console.print(f"  [dim]{details}[/dim]")


def print_success(message: str, details: Optional[str] = None):
    """Print a success message with optional details."""
    console.print(f"[bold green]✓[/bold green] {message}")
    if details:
        console.print(f"  [dim]{details}[/dim]")


def print_warning(message: str, details: Optional[str] = None):
    """Print a warning message with optional details."""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")
    if details:
        console.print(f"  [dim]{details}[/dim]")


def print_info(message: str, details: Optional[str] = None):
    """Print an info message with optional details."""
    console.print(f"[bold cyan]ℹ[/bold cyan] {message}")
    if details:
        console.print(f"  [dim]{details}[/dim]")


def create_agent_table(agents: List[Dict[str, Any]]) -> Table:
    """
    Create a formatted table for displaying agents.

    Args:
        agents: List of agent dictionaries with id, name, version, etc.

    Returns:
        Rich Table object
    """
    table = Table(
        title="Available Agents",
        show_header=True,
        header_style="bold cyan",
        border_style="blue",
    )

    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Version", style="yellow")
    table.add_column("Type", style="magenta")
    table.add_column("Status", style="white")
    table.add_column("Last Updated", style="dim")

    for agent in agents:
        status_color = "green" if agent.get("status") == "active" else "yellow"
        table.add_row(
            agent.get("id", "N/A"),
            agent.get("name", "N/A"),
            agent.get("version", "N/A"),
            agent.get("type", "N/A"),
            f"[{status_color}]{agent.get('status', 'unknown')}[/{status_color}]",
            agent.get("updated", "N/A"),
        )

    return table


def create_directory_tree(root_path: Path, max_depth: int = 3) -> Tree:
    """
    Create a tree view of a directory structure.

    Args:
        root_path: Root directory to display
        max_depth: Maximum depth to traverse

    Returns:
        Rich Tree object
    """
    def add_directory_contents(tree: Tree, path: Path, depth: int = 0):
        if depth >= max_depth:
            return

        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            for item in items:
                if item.name.startswith("."):
                    continue

                if item.is_dir():
                    branch = tree.add(f"[bold blue]📁 {item.name}[/bold blue]")
                    add_directory_contents(branch, item, depth + 1)
                else:
                    icon = "📄"
                    if item.suffix in [".py", ".yaml", ".yml", ".json"]:
                        icon = "📝"
                    tree.add(f"{icon} {item.name}")
        except PermissionError:
            tree.add("[dim red]Permission denied[/dim red]")

    tree = Tree(f"[bold cyan]📦 {root_path.name}[/bold cyan]")
    add_directory_contents(tree, root_path)
    return tree


def create_progress_bar() -> Progress:
    """
    Create a Rich progress bar for long-running operations.

    Returns:
        Rich Progress object
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )


def display_code(code: str, language: str = "python", theme: str = "monokai"):
    """
    Display syntax-highlighted code.

    Args:
        code: Code string to display
        language: Programming language for syntax highlighting
        theme: Color theme to use
    """
    syntax = Syntax(code, language, theme=theme, line_numbers=True)
    console.print(syntax)


def display_yaml(yaml_content: str):
    """
    Display YAML content with syntax highlighting.

    Args:
        yaml_content: YAML string to display
    """
    display_code(yaml_content, language="yaml")


def display_markdown(markdown_content: str):
    """
    Display formatted markdown content.

    Args:
        markdown_content: Markdown string to display
    """
    md = Markdown(markdown_content)
    console.print(md)


def create_info_panel(
    title: str,
    content: Dict[str, Any],
    style: str = "cyan",
) -> Panel:
    """
    Create an information panel with key-value pairs.

    Args:
        title: Panel title
        content: Dictionary of key-value pairs to display
        style: Border style color

    Returns:
        Rich Panel object
    """
    lines = []
    for key, value in content.items():
        lines.append(f"[bold]{key}:[/bold] {value}")

    return Panel(
        "\n".join(lines),
        title=title,
        border_style=style,
        padding=(1, 2),
    )


def confirm_action(message: str, default: bool = False) -> bool:
    """
    Ask user for confirmation.

    Args:
        message: Confirmation message
        default: Default value if user just presses Enter

    Returns:
        True if confirmed, False otherwise
    """
    default_str = "Y/n" if default else "y/N"
    response = console.input(f"{message} [{default_str}]: ").strip().lower()

    if not response:
        return default

    return response in ["y", "yes"]


def print_validation_results(results: Dict[str, Any]):
    """
    Print validation results in a formatted way.

    Args:
        results: Validation results dictionary
    """
    if results.get("valid", False):
        print_success("Validation passed!")

        if results.get("warnings"):
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in results["warnings"]:
                print_warning(warning)
    else:
        print_error("Validation failed!")

        if results.get("errors"):
            console.print("\n[bold red]Errors:[/bold red]")
            for error in results["errors"]:
                console.print(f"  • {error}")


def print_test_results(results: Dict[str, Any]):
    """
    Print test results in a formatted way.

    Args:
        results: Test results dictionary
    """
    total = results.get("total", 0)
    passed = results.get("passed", 0)
    failed = results.get("failed", 0)
    skipped = results.get("skipped", 0)

    # Create summary table
    table = Table(title="Test Results", show_header=False, border_style="blue")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total Tests", str(total))
    table.add_row("Passed", f"[green]{passed}[/green]")
    table.add_row("Failed", f"[red]{failed}[/red]")
    table.add_row("Skipped", f"[yellow]{skipped}[/yellow]")

    if total > 0:
        success_rate = (passed / total) * 100
        color = "green" if success_rate >= 80 else "yellow" if success_rate >= 60 else "red"
        table.add_row("Success Rate", f"[{color}]{success_rate:.1f}%[/{color}]")

    console.print(table)

    # Print failed test details
    if failed > 0 and results.get("failures"):
        console.print("\n[bold red]Failed Tests:[/bold red]")
        for failure in results["failures"]:
            console.print(f"\n  • [red]{failure['name']}[/red]")
            console.print(f"    {failure.get('message', 'No details available')}")


def print_generation_summary(output_path: Path, files_created: List[Path]):
    """
    Print a summary of generated files.

    Args:
        output_path: Root output directory
        files_created: List of created file paths
    """
    console.print("\n[bold green]Agent generated successfully![/bold green]\n")

    console.print(f"[bold]Output directory:[/bold] {output_path}")
    console.print(f"[bold]Files created:[/bold] {len(files_created)}\n")

    # Group files by directory
    from collections import defaultdict
    files_by_dir = defaultdict(list)

    for file_path in files_created:
        rel_path = file_path.relative_to(output_path)
        dir_name = rel_path.parent if rel_path.parent != Path(".") else "root"
        files_by_dir[str(dir_name)].append(rel_path.name)

    # Display as tree
    tree = Tree(f"[bold cyan]📦 {output_path.name}[/bold cyan]")

    for dir_name, files in sorted(files_by_dir.items()):
        if dir_name == "root":
            for file_name in files:
                tree.add(f"📄 {file_name}")
        else:
            branch = tree.add(f"[bold blue]📁 {dir_name}[/bold blue]")
            for file_name in files:
                branch.add(f"📄 {file_name}")

    console.print(tree)
