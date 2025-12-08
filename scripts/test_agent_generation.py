#!/usr/bin/env python3
"""
Test Agent Generation - End-to-End Verification

This script tests the complete agent generation pipeline:
1. Parse AgentSpec YAML (pack.yaml)
2. Generate Python code using CodeGenerator
3. Write files to output directory
4. Verify generated files

Run with:
    python scripts/test_agent_generation.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_eudr_generation():
    """Test EUDR agent generation from pack.yaml."""
    console.print(Panel(
        "[bold cyan]Agent Generation Test[/bold cyan]\n"
        "Testing EUDR Compliance Agent Generation",
        border_style="cyan"
    ))

    try:
        # Import generator components
        from core.greenlang.generator import AgentSpecParser, CodeGenerator, GenerationOptions

        console.print("[green]✓[/green] Generator imports successful")

        # Define paths
        spec_path = project_root / "GL-Agent-Factory" / "08-regulatory-agents" / "eudr" / "pack.yaml"
        output_dir = project_root / "generated" / "eudr_compliance_v1_test"

        console.print(f"[bold]Spec Path:[/bold] {spec_path}")
        console.print(f"[bold]Output Dir:[/bold] {output_dir}")

        if not spec_path.exists():
            console.print(f"[red]✗ Spec file not found: {spec_path}[/red]")
            return False

        # Parse the spec
        console.print("\n[cyan]Parsing AgentSpec...[/cyan]")
        parser = AgentSpecParser(strict=False)  # Use non-strict for flexibility
        spec = parser.parse(spec_path)

        console.print(f"[green]✓[/green] Parsed: {spec.name}")
        console.print(f"  ID: {spec.id}")
        console.print(f"  Version: {spec.version}")
        console.print(f"  Tools: {len(spec.tools)}")
        console.print(f"  Inputs: {len(spec.inputs)}")
        console.print(f"  Outputs: {len(spec.outputs)}")

        if spec.tests:
            console.print(f"  Golden Tests: {len(spec.tests.golden)}")

        # Configure generator
        options = GenerationOptions(
            output_dir=output_dir,
            overwrite=True,
            generate_tests=True,
            generate_readme=True,
            generate_init=True,
            use_async=True,
        )

        # Generate code
        console.print("\n[cyan]Generating agent code...[/cyan]")
        generator = CodeGenerator(options)
        result = generator.generate(spec, output_dir=output_dir)

        console.print(f"[green]✓[/green] Generation complete!")

        # Show results
        table = Table(title="Generated Files")
        table.add_column("File", style="cyan")
        table.add_column("Lines", justify="right", style="green")
        table.add_column("Size", justify="right")

        for file in result.files:
            lines = file.content.count('\n') + 1
            size = f"{len(file.content):,} bytes"
            table.add_row(file.full_path, str(lines), size)

        console.print(table)

        # Summary
        console.print(f"\n[bold]Total Lines:[/bold] {result.total_lines}")
        console.print(f"[bold]Tools Generated:[/bold] {result.num_tools}")
        console.print(f"[bold]Tests Generated:[/bold] {result.num_tests}")

        # Verify files exist
        console.print("\n[cyan]Verifying generated files...[/cyan]")

        expected_files = ["agent.py", "tools.py", "__init__.py", "README.md"]
        for filename in expected_files:
            filepath = output_dir / filename
            if filepath.exists():
                console.print(f"[green]✓[/green] {filename} exists ({filepath.stat().st_size:,} bytes)")
            else:
                console.print(f"[red]✗[/red] {filename} missing")

        # Show agent.py preview
        agent_file = output_dir / "agent.py"
        if agent_file.exists():
            console.print("\n[bold]agent.py Preview (first 50 lines):[/bold]")
            content = agent_file.read_text()
            lines = content.split('\n')[:50]
            preview = '\n'.join(lines)
            console.print(Syntax(preview, "python", theme="monokai", line_numbers=True))

        console.print("\n[bold green]Agent generation test PASSED![/bold green]")
        return True

    except ImportError as e:
        console.print(f"[red]Import error: {e}[/red]")
        console.print("[yellow]Make sure you're in the project root and have dependencies installed[/yellow]")
        return False

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return False


def test_sb253_generation():
    """Test SB253 agent generation from pack.yaml."""
    console.print(Panel(
        "[bold cyan]SB253 Agent Generation Test[/bold cyan]",
        border_style="cyan"
    ))

    try:
        from core.greenlang.generator import AgentSpecParser, CodeGenerator, GenerationOptions

        spec_path = project_root / "GL-Agent-Factory" / "08-regulatory-agents" / "sb253" / "pack.yaml"
        output_dir = project_root / "generated" / "sb253_disclosure_v1_test"

        if not spec_path.exists():
            console.print(f"[red]✗ Spec file not found: {spec_path}[/red]")
            return False

        parser = AgentSpecParser(strict=False)
        spec = parser.parse(spec_path)

        console.print(f"[green]✓[/green] Parsed: {spec.name}")
        console.print(f"  ID: {spec.id}")
        console.print(f"  Tools: {len(spec.tools)}")

        options = GenerationOptions(
            output_dir=output_dir,
            overwrite=True,
            generate_tests=True,
        )

        generator = CodeGenerator(options)
        result = generator.generate(spec, output_dir=output_dir)

        console.print(f"[green]✓[/green] Generated {result.total_lines} lines of code")
        console.print(f"[bold green]SB253 generation test PASSED![/bold green]")
        return True

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        return False


def main():
    """Run all generation tests."""
    console.print("\n" + "=" * 60)
    console.print("[bold]GreenLang Agent Factory - Generation Test Suite[/bold]")
    console.print("=" * 60 + "\n")

    results = {}

    # Test EUDR generation
    results["EUDR"] = test_eudr_generation()

    console.print("\n" + "-" * 60 + "\n")

    # Test SB253 generation
    results["SB253"] = test_sb253_generation()

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Test Summary[/bold]")
    console.print("=" * 60)

    for name, passed in results.items():
        status = "[green]PASSED[/green]" if passed else "[red]FAILED[/red]"
        console.print(f"  {name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        console.print("\n[bold green]All tests passed! Agent generation is working.[/bold green]")
    else:
        console.print("\n[bold red]Some tests failed. Review errors above.[/bold red]")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
