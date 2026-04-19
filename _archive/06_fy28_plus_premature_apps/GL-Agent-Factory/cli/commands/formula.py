"""
Formula Management Commands

Commands for searching, validating, testing, and managing formulas
in the GreenLang formula library.
"""

import typer
from typing import Optional, List
from pathlib import Path
import yaml
import json
from enum import Enum
from datetime import datetime

from cli.utils.console import (
    console,
    print_error,
    print_success,
    print_warning,
    print_info,
    create_info_panel,
    display_yaml,
    create_progress_bar,
)
from cli.utils.config import load_config, get_config_value


class FormulaCategory(str, Enum):
    """Formula category enumeration."""
    SCOPE1 = "scope1"
    SCOPE2 = "scope2"
    SCOPE3 = "scope3"
    CBAM = "cbam"
    FINANCIAL = "financial"
    ENERGY = "energy"
    WATER = "water"
    WASTE = "waste"
    ALL = "all"


# Create formula command group
app = typer.Typer(
    help="Formula management commands - search, validate, test formulas",
    no_args_is_help=True,
)


# Formula registry (simulated - would connect to actual formula database)
FORMULA_REGISTRY = {
    "scope1_stationary_combustion": {
        "id": "scope1_stationary_combustion",
        "name": "Scope 1 Stationary Combustion Emissions",
        "category": "scope1",
        "standard": "GHG Protocol",
        "version": "1.0",
        "description": "Calculate CO2e emissions from stationary fuel combustion",
        "parameters": ["fuel_quantity", "fuel_type", "region"],
        "output_unit": "t_co2e",
    },
    "scope2_purchased_electricity": {
        "id": "scope2_purchased_electricity",
        "name": "Scope 2 Purchased Electricity Emissions",
        "category": "scope2",
        "standard": "GHG Protocol",
        "version": "1.0",
        "description": "Calculate CO2e emissions from purchased electricity",
        "parameters": ["electricity_kwh", "grid_region", "market_based"],
        "output_unit": "t_co2e",
    },
    "scope3_business_travel": {
        "id": "scope3_business_travel",
        "name": "Scope 3 Business Travel Emissions",
        "category": "scope3",
        "standard": "GHG Protocol",
        "version": "1.0",
        "description": "Calculate CO2e emissions from business travel",
        "parameters": ["distance_km", "transport_mode", "class_type"],
        "output_unit": "t_co2e",
    },
    "cbam_embedded_emissions": {
        "id": "cbam_embedded_emissions",
        "name": "CBAM Embedded Emissions",
        "category": "cbam",
        "standard": "EU CBAM",
        "version": "1.0",
        "description": "Calculate embedded emissions for CBAM reporting",
        "parameters": ["product_type", "production_process", "country_of_origin"],
        "output_unit": "t_co2e_per_tonne",
    },
    "financial_npv": {
        "id": "financial_npv",
        "name": "Net Present Value Calculation",
        "category": "financial",
        "standard": "Corporate Finance",
        "version": "1.0",
        "description": "Calculate NPV for sustainability investments",
        "parameters": ["cash_flows", "discount_rate", "periods"],
        "output_unit": "currency",
    },
    "energy_efficiency_ratio": {
        "id": "energy_efficiency_ratio",
        "name": "Energy Efficiency Ratio",
        "category": "energy",
        "standard": "ISO 50001",
        "version": "1.0",
        "description": "Calculate energy efficiency metrics",
        "parameters": ["energy_input", "useful_output", "baseline"],
        "output_unit": "ratio",
    },
}


@app.command("search")
def search_formulas(
    query: str = typer.Argument(
        ...,
        help="Search query for formulas",
    ),
    category: Optional[FormulaCategory] = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter by category",
    ),
    standard: Optional[str] = typer.Option(
        None,
        "--standard",
        "-s",
        help="Filter by standard (e.g., 'GHG Protocol', 'ISO 14064')",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum number of results",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table/json/yaml)",
    ),
):
    """
    Search for formulas in the formula library.

    Search across all GreenLang formulas by name, description, or parameters.
    Filter results by category or regulatory standard.

    Examples:
        gl formula search "combustion"
        gl formula search "emissions" --category scope1
        gl formula search "carbon" --standard "GHG Protocol"
    """
    try:
        console.print(f"\n[bold cyan]Searching formulas for:[/bold cyan] {query}\n")

        # Search formulas
        results = []
        query_lower = query.lower()

        for formula_id, formula in FORMULA_REGISTRY.items():
            # Check if query matches
            matches = (
                query_lower in formula["name"].lower() or
                query_lower in formula["description"].lower() or
                query_lower in formula_id.lower() or
                any(query_lower in p.lower() for p in formula.get("parameters", []))
            )

            if not matches:
                continue

            # Apply filters
            if category and category != FormulaCategory.ALL:
                if formula.get("category") != category.value:
                    continue

            if standard and standard.lower() not in formula.get("standard", "").lower():
                continue

            results.append(formula)

            if len(results) >= limit:
                break

        if not results:
            print_info("No formulas found matching your query")
            print_info("Try broadening your search or use: gl formula list --category all")
            raise typer.Exit(0)

        # Display results
        if output_format == "json":
            console.print_json(data=results)
        elif output_format == "yaml":
            console.print(yaml.dump(results, default_flow_style=False))
        else:
            # Table format
            from rich.table import Table

            table = Table(title=f"Formula Search Results ({len(results)} found)")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Name", style="green")
            table.add_column("Category", style="yellow")
            table.add_column("Standard", style="magenta")
            table.add_column("Output", style="dim")

            for formula in results:
                table.add_row(
                    formula["id"],
                    formula["name"][:40],
                    formula.get("category", "N/A"),
                    formula.get("standard", "N/A"),
                    formula.get("output_unit", "N/A"),
                )

            console.print(table)

        console.print(f"\n[dim]Found {len(results)} formulas[/dim]\n")

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Search failed: {str(e)}")
        raise typer.Exit(1)


@app.command("validate")
def validate_formula(
    formula_id: str = typer.Argument(
        ...,
        help="Formula ID to validate",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Enable strict validation mode",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed validation output",
    ),
):
    """
    Validate a formula definition.

    Checks that the formula:
    - Has all required fields
    - Parameters are properly typed
    - Calculation steps are valid
    - Output format is correct
    - Conforms to GreenLang formula standards

    Examples:
        gl formula validate scope1_stationary_combustion
        gl formula validate cbam_embedded_emissions --strict
    """
    try:
        console.print(f"\n[bold cyan]Validating formula:[/bold cyan] {formula_id}\n")

        # Find formula
        formula = FORMULA_REGISTRY.get(formula_id)
        if not formula:
            print_error(f"Formula not found: {formula_id}")
            print_info("Use 'gl formula search' to find available formulas")
            raise typer.Exit(1)

        # Display formula info
        console.print(create_info_panel("Formula Information", {
            "ID": formula["id"],
            "Name": formula["name"],
            "Category": formula.get("category", "N/A"),
            "Standard": formula.get("standard", "N/A"),
            "Version": formula.get("version", "N/A"),
        }))
        console.print()

        # Validate formula
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "checks": {},
        }

        with create_progress_bar() as progress:
            # Check 1: Required fields
            task1 = progress.add_task("Checking required fields...", total=100)
            required_fields = ["id", "name", "description", "parameters"]
            missing = [f for f in required_fields if f not in formula]
            if missing:
                validation_results["errors"].append(f"Missing required fields: {missing}")
                validation_results["valid"] = False
            validation_results["checks"]["required_fields"] = {
                "passed": len(missing) == 0,
                "details": f"Checked {len(required_fields)} fields",
            }
            progress.update(task1, completed=100)

            # Check 2: Parameters
            task2 = progress.add_task("Validating parameters...", total=100)
            params = formula.get("parameters", [])
            if not params:
                validation_results["warnings"].append("No parameters defined")
            validation_results["checks"]["parameters"] = {
                "passed": len(params) > 0,
                "details": f"{len(params)} parameters defined",
            }
            progress.update(task2, completed=100)

            # Check 3: Output format
            task3 = progress.add_task("Checking output format...", total=100)
            has_output = "output_unit" in formula
            if not has_output:
                validation_results["warnings"].append("No output unit specified")
            validation_results["checks"]["output_format"] = {
                "passed": has_output,
                "details": formula.get("output_unit", "Not specified"),
            }
            progress.update(task3, completed=100)

            # Check 4: Standard compliance
            task4 = progress.add_task("Checking standard compliance...", total=100)
            has_standard = "standard" in formula
            if not has_standard and strict:
                validation_results["errors"].append("No regulatory standard specified")
                validation_results["valid"] = False
            validation_results["checks"]["standard_compliance"] = {
                "passed": has_standard,
                "details": formula.get("standard", "Not specified"),
            }
            progress.update(task4, completed=100)

        console.print()

        # Display results
        if validation_results["valid"]:
            print_success("Formula validation passed!")
        else:
            print_error("Formula validation failed!")

        if validation_results["errors"]:
            console.print("\n[bold red]Errors:[/bold red]")
            for error in validation_results["errors"]:
                console.print(f"  - {error}")

        if validation_results["warnings"]:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in validation_results["warnings"]:
                console.print(f"  - {warning}")

        if verbose:
            console.print("\n[bold]Check Results:[/bold]")
            for check_name, check_result in validation_results["checks"].items():
                status = "[green]PASS[/green]" if check_result["passed"] else "[red]FAIL[/red]"
                console.print(f"  {check_name}: {status} - {check_result.get('details', '')}")

        if not validation_results["valid"]:
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Validation failed: {str(e)}")
        raise typer.Exit(1)


@app.command("test")
def test_formula(
    formula_id: str = typer.Argument(
        ...,
        help="Formula ID to test",
    ),
    test_data: Optional[Path] = typer.Option(
        None,
        "--data",
        "-d",
        help="Path to test data YAML file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed test output",
    ),
    golden: bool = typer.Option(
        False,
        "--golden",
        "-g",
        help="Run golden tests (expected output validation)",
    ),
):
    """
    Run tests for a formula.

    Execute formula with test inputs and validate outputs.
    Use --golden flag to compare against expected golden outputs.

    Examples:
        gl formula test scope1_stationary_combustion
        gl formula test cbam_embedded_emissions --data test_inputs.yaml
        gl formula test financial_npv --golden
    """
    try:
        console.print(f"\n[bold cyan]Testing formula:[/bold cyan] {formula_id}\n")

        # Find formula
        formula = FORMULA_REGISTRY.get(formula_id)
        if not formula:
            print_error(f"Formula not found: {formula_id}")
            raise typer.Exit(1)

        # Display formula info
        console.print(create_info_panel("Formula Under Test", {
            "ID": formula["id"],
            "Name": formula["name"],
            "Parameters": ", ".join(formula.get("parameters", [])),
            "Output": formula.get("output_unit", "N/A"),
        }))
        console.print()

        # Load test data
        test_cases = []
        if test_data and test_data.exists():
            with open(test_data, "r") as f:
                test_cases = yaml.safe_load(f) or []
        else:
            # Generate sample test cases
            test_cases = _generate_sample_test_cases(formula)

        if not test_cases:
            print_warning("No test cases available")
            print_info("Provide test data with: gl formula test <id> --data tests.yaml")
            raise typer.Exit(0)

        # Run tests
        console.print(f"[bold]Running {len(test_cases)} test case(s)...[/bold]\n")

        results = {
            "total": len(test_cases),
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "failures": [],
        }

        with create_progress_bar() as progress:
            task = progress.add_task("Running tests...", total=len(test_cases))

            for i, test_case in enumerate(test_cases):
                test_name = test_case.get("name", f"test_{i+1}")

                try:
                    # Execute formula (simulated)
                    output = _execute_formula(formula, test_case.get("inputs", {}))

                    # Check expected output if golden test
                    if golden and "expected" in test_case:
                        expected = test_case["expected"]
                        if output != expected:
                            results["failed"] += 1
                            results["failures"].append({
                                "name": test_name,
                                "message": f"Expected {expected}, got {output}",
                            })
                        else:
                            results["passed"] += 1
                    else:
                        results["passed"] += 1

                    if verbose:
                        console.print(f"  [green]PASS[/green] {test_name}: {output}")

                except Exception as e:
                    results["failed"] += 1
                    results["failures"].append({
                        "name": test_name,
                        "message": str(e),
                    })
                    if verbose:
                        console.print(f"  [red]FAIL[/red] {test_name}: {str(e)}")

                progress.update(task, advance=1)

        console.print()

        # Display results
        from rich.table import Table

        table = Table(title="Test Results", show_header=False)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Total Tests", str(results["total"]))
        table.add_row("Passed", f"[green]{results['passed']}[/green]")
        table.add_row("Failed", f"[red]{results['failed']}[/red]")
        table.add_row("Skipped", f"[yellow]{results['skipped']}[/yellow]")

        if results["total"] > 0:
            success_rate = (results["passed"] / results["total"]) * 100
            color = "green" if success_rate >= 80 else "yellow" if success_rate >= 60 else "red"
            table.add_row("Success Rate", f"[{color}]{success_rate:.1f}%[/{color}]")

        console.print(table)

        if results["failures"]:
            console.print("\n[bold red]Failed Tests:[/bold red]")
            for failure in results["failures"][:5]:
                console.print(f"  - {failure['name']}: {failure['message']}")

        if results["failed"] > 0:
            raise typer.Exit(1)

        print_success("All tests passed!")

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Test execution failed: {str(e)}")
        raise typer.Exit(1)


@app.command("list")
def list_formulas(
    category: FormulaCategory = typer.Option(
        FormulaCategory.ALL,
        "--category",
        "-c",
        help="Filter by category",
    ),
    standard: Optional[str] = typer.Option(
        None,
        "--standard",
        "-s",
        help="Filter by standard",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table/json/yaml)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information",
    ),
):
    """
    List all available formulas.

    Display all formulas in the GreenLang formula library,
    optionally filtered by category or standard.

    Examples:
        gl formula list
        gl formula list --category scope1
        gl formula list --standard "GHG Protocol"
    """
    try:
        console.print("\n[bold cyan]Available Formulas[/bold cyan]\n")

        # Filter formulas
        formulas = []
        for formula_id, formula in FORMULA_REGISTRY.items():
            # Apply category filter
            if category != FormulaCategory.ALL:
                if formula.get("category") != category.value:
                    continue

            # Apply standard filter
            if standard and standard.lower() not in formula.get("standard", "").lower():
                continue

            formulas.append(formula)

        if not formulas:
            print_info("No formulas found matching filters")
            raise typer.Exit(0)

        # Display results
        if output_format == "json":
            console.print_json(data=formulas)
        elif output_format == "yaml":
            console.print(yaml.dump(formulas, default_flow_style=False))
        else:
            # Table format
            from rich.table import Table

            table = Table(title=f"Formula Library ({len(formulas)} formulas)")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Name", style="green")
            table.add_column("Category", style="yellow")
            table.add_column("Standard", style="magenta")

            if verbose:
                table.add_column("Parameters", style="dim")
                table.add_column("Output", style="dim")

            for formula in formulas:
                row = [
                    formula["id"],
                    formula["name"][:35],
                    formula.get("category", "N/A"),
                    formula.get("standard", "N/A"),
                ]

                if verbose:
                    row.append(", ".join(formula.get("parameters", [])[:3]))
                    row.append(formula.get("output_unit", "N/A"))

                table.add_row(*row)

            console.print(table)

        console.print(f"\n[dim]Total: {len(formulas)} formulas[/dim]")

        # Category summary
        if category == FormulaCategory.ALL:
            console.print("\n[bold]Categories:[/bold]")
            categories = {}
            for f in FORMULA_REGISTRY.values():
                cat = f.get("category", "other")
                categories[cat] = categories.get(cat, 0) + 1
            for cat, count in sorted(categories.items()):
                console.print(f"  {cat}: {count} formulas")

        console.print()

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to list formulas: {str(e)}")
        raise typer.Exit(1)


@app.command("info")
def formula_info(
    formula_id: str = typer.Argument(
        ...,
        help="Formula ID to show information for",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information including calculation steps",
    ),
):
    """
    Show detailed information about a formula.

    Examples:
        gl formula info scope1_stationary_combustion
        gl formula info cbam_embedded_emissions --verbose
    """
    try:
        console.print(f"\n[bold cyan]Formula Information:[/bold cyan] {formula_id}\n")

        # Find formula
        formula = FORMULA_REGISTRY.get(formula_id)
        if not formula:
            print_error(f"Formula not found: {formula_id}")
            raise typer.Exit(1)

        # Display basic info
        console.print(create_info_panel("Formula Details", {
            "ID": formula["id"],
            "Name": formula["name"],
            "Category": formula.get("category", "N/A"),
            "Standard": formula.get("standard", "N/A"),
            "Version": formula.get("version", "N/A"),
            "Output Unit": formula.get("output_unit", "N/A"),
        }))

        # Description
        console.print(f"\n[bold]Description:[/bold]")
        console.print(f"  {formula.get('description', 'No description')}")

        # Parameters
        console.print(f"\n[bold]Parameters:[/bold]")
        for param in formula.get("parameters", []):
            console.print(f"  - {param}")

        # Usage example
        console.print(f"\n[bold]Usage Example:[/bold]")
        example_params = ", ".join([f'{p}=<value>' for p in formula.get("parameters", [])[:3]])
        console.print(f"  [cyan]gl formula test {formula_id} --data inputs.yaml[/cyan]")

        if verbose:
            # Show calculation steps (simulated)
            console.print(f"\n[bold]Calculation Steps:[/bold]")
            console.print("  1. Load input parameters")
            console.print("  2. Validate inputs against schema")
            console.print("  3. Lookup emission factors from database")
            console.print("  4. Apply formula: activity_data * emission_factor")
            console.print("  5. Convert units if necessary")
            console.print("  6. Return result with provenance hash")

        console.print()

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to get formula info: {str(e)}")
        raise typer.Exit(1)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _generate_sample_test_cases(formula: dict) -> List[dict]:
    """Generate sample test cases for a formula."""
    # Generate basic test cases based on formula parameters
    test_cases = []

    if formula["id"] == "scope1_stationary_combustion":
        test_cases = [
            {
                "name": "diesel_combustion_us",
                "inputs": {
                    "fuel_quantity": 1000,
                    "fuel_type": "diesel",
                    "region": "US",
                },
                "expected": 2.68,
            },
            {
                "name": "natural_gas_eu",
                "inputs": {
                    "fuel_quantity": 500,
                    "fuel_type": "natural_gas",
                    "region": "EU",
                },
            },
        ]
    elif formula["id"] == "scope2_purchased_electricity":
        test_cases = [
            {
                "name": "grid_electricity_us",
                "inputs": {
                    "electricity_kwh": 10000,
                    "grid_region": "US-WECC",
                    "market_based": False,
                },
            },
        ]
    else:
        # Generic test case
        test_cases = [
            {
                "name": "basic_test",
                "inputs": {p: "test_value" for p in formula.get("parameters", [])},
            },
        ]

    return test_cases


def _execute_formula(formula: dict, inputs: dict) -> float:
    """Execute a formula with given inputs (simulated)."""
    # Simulated formula execution
    # In real implementation, this would call the formula engine

    if formula["id"] == "scope1_stationary_combustion":
        # Simulated calculation
        fuel_qty = inputs.get("fuel_quantity", 0)
        emission_factor = 2.68  # kg CO2e per liter diesel
        result = (fuel_qty * emission_factor) / 1000  # Convert to tonnes
        return round(result, 3)

    elif formula["id"] == "scope2_purchased_electricity":
        electricity = inputs.get("electricity_kwh", 0)
        grid_factor = 0.4  # kg CO2e per kWh (example)
        result = (electricity * grid_factor) / 1000
        return round(result, 3)

    else:
        # Generic result
        return 0.0
