"""
Standards Management Commands

Commands for searching, viewing, and managing regulatory standards
in the GreenLang standards library (NFPA, OSHA, ISO, GHG Protocol, etc.).
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


class StandardType(str, Enum):
    """Standard type enumeration."""
    SAFETY = "safety"
    ENVIRONMENTAL = "environmental"
    QUALITY = "quality"
    ENERGY = "energy"
    FINANCIAL = "financial"
    ALL = "all"


# Create standards command group
app = typer.Typer(
    help="Standards management commands - search, view, and manage regulatory standards",
    no_args_is_help=True,
)


# Standards registry (simulated - would connect to actual standards database)
STANDARDS_REGISTRY = {
    "NFPA-86": {
        "code": "NFPA-86",
        "name": "Standard for Ovens and Furnaces",
        "type": "safety",
        "organization": "National Fire Protection Association",
        "version": "2023",
        "description": "Safety requirements for ovens, furnaces, and related equipment",
        "equipment_types": ["furnace", "oven", "dryer", "thermal_oxidizer"],
        "sections": {
            "Chapter 1": "Administration",
            "Chapter 2": "Referenced Publications",
            "Chapter 3": "Definitions",
            "Chapter 4": "General Requirements",
            "Chapter 5": "Location and Construction",
            "Chapter 6": "Safety Equipment and Application",
            "Chapter 7": "Heating Systems",
            "Chapter 8": "Special Atmospheres",
            "Chapter 9": "Fire Protection",
            "Chapter 10": "Operations and Maintenance",
        },
    },
    "OSHA-1910.106": {
        "code": "OSHA-1910.106",
        "name": "Flammable Liquids",
        "type": "safety",
        "organization": "Occupational Safety and Health Administration",
        "version": "current",
        "description": "Storage and handling of flammable liquids",
        "equipment_types": ["storage_tank", "piping", "process_equipment"],
        "sections": {
            "1910.106(a)": "Definitions",
            "1910.106(b)": "Container and Portable Tank Storage",
            "1910.106(c)": "Industrial Plants",
            "1910.106(d)": "Container and Portable Tank Storage",
            "1910.106(e)": "Industrial Plants",
            "1910.106(f)": "Bulk Plants",
            "1910.106(g)": "Service Stations",
            "1910.106(h)": "Processing Plants",
            "1910.106(i)": "Refineries, Chemical Plants",
            "1910.106(j)": "Scope",
        },
    },
    "OSHA-1910.119": {
        "code": "OSHA-1910.119",
        "name": "Process Safety Management",
        "type": "safety",
        "organization": "Occupational Safety and Health Administration",
        "version": "current",
        "description": "Process safety management of highly hazardous chemicals",
        "equipment_types": ["process_equipment", "reactor", "storage", "piping"],
        "sections": {
            "1910.119(c)": "Employee Participation",
            "1910.119(d)": "Process Safety Information",
            "1910.119(e)": "Process Hazard Analysis",
            "1910.119(f)": "Operating Procedures",
            "1910.119(g)": "Training",
            "1910.119(h)": "Contractors",
            "1910.119(i)": "Pre-Startup Safety Review",
            "1910.119(j)": "Mechanical Integrity",
            "1910.119(k)": "Hot Work Permit",
            "1910.119(l)": "Management of Change",
            "1910.119(m)": "Incident Investigation",
            "1910.119(n)": "Emergency Planning",
            "1910.119(o)": "Compliance Audits",
        },
    },
    "ISO-14064": {
        "code": "ISO-14064",
        "name": "Greenhouse Gas Accounting and Verification",
        "type": "environmental",
        "organization": "International Organization for Standardization",
        "version": "2018",
        "description": "Specification for quantification and reporting of GHG emissions",
        "equipment_types": ["all"],
        "sections": {
            "Part 1": "Organization Level - Design and Development",
            "Part 2": "Project Level - Quantification and Reporting",
            "Part 3": "Verification and Validation",
        },
    },
    "ISO-50001": {
        "code": "ISO-50001",
        "name": "Energy Management Systems",
        "type": "energy",
        "organization": "International Organization for Standardization",
        "version": "2018",
        "description": "Requirements for energy management systems",
        "equipment_types": ["all"],
        "sections": {
            "Clause 4": "Context of the Organization",
            "Clause 5": "Leadership",
            "Clause 6": "Planning",
            "Clause 7": "Support",
            "Clause 8": "Operation",
            "Clause 9": "Performance Evaluation",
            "Clause 10": "Improvement",
        },
    },
    "GHG-PROTOCOL": {
        "code": "GHG-PROTOCOL",
        "name": "GHG Protocol Corporate Standard",
        "type": "environmental",
        "organization": "World Resources Institute / WBCSD",
        "version": "Revised 2015",
        "description": "Corporate accounting and reporting standard for GHG emissions",
        "equipment_types": ["all"],
        "sections": {
            "Chapter 1": "GHG Accounting and Reporting Principles",
            "Chapter 2": "Business Goals and Inventory Design",
            "Chapter 3": "Setting Organizational Boundaries",
            "Chapter 4": "Setting Operational Boundaries",
            "Chapter 5": "Tracking Emissions Over Time",
            "Chapter 6": "Identifying and Calculating GHG Emissions",
            "Chapter 7": "Managing Inventory Quality",
            "Chapter 8": "Accounting for GHG Reductions",
            "Chapter 9": "Reporting GHG Emissions",
            "Chapter 10": "Verification of GHG Emissions",
        },
    },
    "EU-CBAM": {
        "code": "EU-CBAM",
        "name": "Carbon Border Adjustment Mechanism",
        "type": "environmental",
        "organization": "European Union",
        "version": "2023",
        "description": "Carbon border adjustment for imports into the EU",
        "equipment_types": ["all"],
        "sections": {
            "Article 1": "Subject Matter",
            "Article 2": "Scope",
            "Article 3": "Definitions",
            "Article 4": "CBAM Certificates",
            "Article 5": "Application for Authorization",
            "Article 6": "CBAM Declaration",
            "Article 7": "Calculation of Embedded Emissions",
            "Article 8": "Default Values",
            "Article 9": "Verification",
        },
    },
}


# Equipment to standards mapping
EQUIPMENT_STANDARDS_MAP = {
    "furnace": ["NFPA-86", "OSHA-1910.106", "ISO-50001"],
    "oven": ["NFPA-86", "ISO-50001"],
    "boiler": ["NFPA-85", "ISO-50001", "EPA-40CFR60"],
    "thermal_oxidizer": ["NFPA-86", "EPA-40CFR60"],
    "storage_tank": ["OSHA-1910.106", "API-650", "EPA-40CFR60"],
    "process_equipment": ["OSHA-1910.119", "API-RP-752"],
    "reactor": ["OSHA-1910.119", "NFPA-652"],
    "dryer": ["NFPA-86", "NFPA-652"],
    "compressor": ["API-617", "ISO-50001"],
}


@app.command("search")
def search_standards(
    query: str = typer.Argument(
        ...,
        help="Search query for standards",
    ),
    standard_type: Optional[StandardType] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by standard type",
    ),
    organization: Optional[str] = typer.Option(
        None,
        "--org",
        "-o",
        help="Filter by organization (e.g., 'NFPA', 'OSHA', 'ISO')",
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
    Search for regulatory standards.

    Search across all standards by name, code, description, or organization.

    Examples:
        gl standards search "furnace"
        gl standards search "safety" --type safety
        gl standards search "emissions" --org ISO
    """
    try:
        console.print(f"\n[bold cyan]Searching standards for:[/bold cyan] {query}\n")

        # Search standards
        results = []
        query_lower = query.lower()

        for code, standard in STANDARDS_REGISTRY.items():
            # Check if query matches
            matches = (
                query_lower in standard["name"].lower() or
                query_lower in standard["description"].lower() or
                query_lower in code.lower() or
                query_lower in standard.get("organization", "").lower() or
                any(query_lower in eq.lower() for eq in standard.get("equipment_types", []))
            )

            if not matches:
                continue

            # Apply filters
            if standard_type and standard_type != StandardType.ALL:
                if standard.get("type") != standard_type.value:
                    continue

            if organization and organization.lower() not in standard.get("organization", "").lower():
                continue

            results.append(standard)

            if len(results) >= limit:
                break

        if not results:
            print_info("No standards found matching your query")
            raise typer.Exit(0)

        # Display results
        if output_format == "json":
            console.print_json(data=results)
        elif output_format == "yaml":
            console.print(yaml.dump(results, default_flow_style=False))
        else:
            # Table format
            from rich.table import Table

            table = Table(title=f"Standards Search Results ({len(results)} found)")
            table.add_column("Code", style="cyan", no_wrap=True)
            table.add_column("Name", style="green")
            table.add_column("Type", style="yellow")
            table.add_column("Organization", style="magenta")
            table.add_column("Version", style="dim")

            for standard in results:
                table.add_row(
                    standard["code"],
                    standard["name"][:40],
                    standard.get("type", "N/A"),
                    standard.get("organization", "N/A")[:25],
                    standard.get("version", "N/A"),
                )

            console.print(table)

        console.print(f"\n[dim]Found {len(results)} standards[/dim]\n")

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Search failed: {str(e)}")
        raise typer.Exit(1)


@app.command("equipment")
def equipment_standards(
    equipment_type: str = typer.Argument(
        ...,
        help="Equipment type to find standards for",
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
        help="Show detailed standard information",
    ),
):
    """
    Get applicable standards for an equipment type.

    Find all regulatory standards that apply to a specific equipment type.

    Examples:
        gl standards equipment furnace
        gl standards equipment boiler --verbose
        gl standards equipment storage_tank --format json
    """
    try:
        equipment_lower = equipment_type.lower().replace(" ", "_").replace("-", "_")

        console.print(f"\n[bold cyan]Standards for equipment:[/bold cyan] {equipment_type}\n")

        # Find applicable standards
        applicable_codes = EQUIPMENT_STANDARDS_MAP.get(equipment_lower, [])

        if not applicable_codes:
            # Try fuzzy match
            for eq_type, codes in EQUIPMENT_STANDARDS_MAP.items():
                if equipment_lower in eq_type or eq_type in equipment_lower:
                    applicable_codes = codes
                    break

        if not applicable_codes:
            print_warning(f"No standards found for equipment type: {equipment_type}")
            print_info("\nAvailable equipment types:")
            for eq in sorted(EQUIPMENT_STANDARDS_MAP.keys()):
                console.print(f"  - {eq}")
            raise typer.Exit(0)

        # Get standard details
        standards = []
        for code in applicable_codes:
            if code in STANDARDS_REGISTRY:
                standards.append(STANDARDS_REGISTRY[code])
            else:
                standards.append({
                    "code": code,
                    "name": code,
                    "type": "unknown",
                    "description": "Standard reference",
                })

        # Display results
        if output_format == "json":
            console.print_json(data=standards)
        elif output_format == "yaml":
            console.print(yaml.dump(standards, default_flow_style=False))
        else:
            console.print(create_info_panel("Equipment Standards", {
                "Equipment Type": equipment_type,
                "Applicable Standards": len(standards),
            }))
            console.print()

            from rich.table import Table

            table = Table(title=f"Applicable Standards for {equipment_type}")
            table.add_column("Code", style="cyan", no_wrap=True)
            table.add_column("Name", style="green")
            table.add_column("Type", style="yellow")
            table.add_column("Organization", style="magenta")

            for standard in standards:
                table.add_row(
                    standard["code"],
                    standard.get("name", "N/A")[:40],
                    standard.get("type", "N/A"),
                    standard.get("organization", "N/A")[:30],
                )

            console.print(table)

            if verbose:
                console.print("\n[bold]Standard Details:[/bold]")
                for standard in standards:
                    console.print(f"\n  [cyan]{standard['code']}[/cyan]")
                    console.print(f"    {standard.get('description', 'No description')[:80]}")

        console.print(f"\n[dim]Total: {len(standards)} applicable standards[/dim]\n")

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to get equipment standards: {str(e)}")
        raise typer.Exit(1)


@app.command("section")
def section_details(
    code: str = typer.Argument(
        ...,
        help="Standard code (e.g., 'NFPA-86', 'ISO-14064')",
    ),
    section: Optional[str] = typer.Argument(
        None,
        help="Section identifier (e.g., 'Chapter 6', '1910.119(e)')",
    ),
    output_format: str = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format (text/json/yaml)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed section content",
    ),
):
    """
    Get section details from a standard.

    View specific sections or all sections of a regulatory standard.

    Examples:
        gl standards section NFPA-86
        gl standards section NFPA-86 "Chapter 6"
        gl standards section ISO-14064 "Part 1" --verbose
    """
    try:
        # Find standard
        standard = STANDARDS_REGISTRY.get(code.upper()) or STANDARDS_REGISTRY.get(code)

        if not standard:
            print_error(f"Standard not found: {code}")
            print_info("Use 'gl standards search' to find available standards")
            raise typer.Exit(1)

        console.print(f"\n[bold cyan]Standard:[/bold cyan] {standard['name']}\n")

        # Display standard info
        console.print(create_info_panel("Standard Information", {
            "Code": standard["code"],
            "Name": standard["name"],
            "Organization": standard.get("organization", "N/A"),
            "Version": standard.get("version", "N/A"),
        }))
        console.print()

        sections = standard.get("sections", {})

        if not sections:
            print_warning("No section information available for this standard")
            raise typer.Exit(0)

        # If specific section requested
        if section:
            section_key = None
            section_title = None

            # Find matching section
            for key, title in sections.items():
                if section.lower() in key.lower() or section.lower() in title.lower():
                    section_key = key
                    section_title = title
                    break

            if not section_key:
                print_error(f"Section not found: {section}")
                print_info("\nAvailable sections:")
                for key, title in sections.items():
                    console.print(f"  {key}: {title}")
                raise typer.Exit(1)

            # Display section
            console.print(f"[bold]Section:[/bold] {section_key}")
            console.print(f"[bold]Title:[/bold] {section_title}")

            if verbose:
                # Show simulated section content
                console.print("\n[bold]Content:[/bold]")
                console.print(f"  This section covers requirements related to {section_title.lower()}.")
                console.print("  [dim](Full content available in standard document)[/dim]")

                console.print("\n[bold]Key Requirements:[/bold]")
                console.print("  - Requirement 1: [dim]Placeholder[/dim]")
                console.print("  - Requirement 2: [dim]Placeholder[/dim]")
                console.print("  - Requirement 3: [dim]Placeholder[/dim]")

        else:
            # Display all sections
            if output_format == "json":
                console.print_json(data=sections)
            elif output_format == "yaml":
                console.print(yaml.dump(sections, default_flow_style=False))
            else:
                console.print("[bold]Sections:[/bold]\n")
                for key, title in sections.items():
                    console.print(f"  [cyan]{key}[/cyan]: {title}")

        console.print()

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to get section details: {str(e)}")
        raise typer.Exit(1)


@app.command("list")
def list_standards(
    standard_type: StandardType = typer.Option(
        StandardType.ALL,
        "--type",
        "-t",
        help="Filter by standard type",
    ),
    organization: Optional[str] = typer.Option(
        None,
        "--org",
        "-o",
        help="Filter by organization",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table/json/yaml)",
    ),
):
    """
    List all available standards.

    Display all standards in the GreenLang standards library.

    Examples:
        gl standards list
        gl standards list --type safety
        gl standards list --org OSHA
    """
    try:
        console.print("\n[bold cyan]Available Standards[/bold cyan]\n")

        # Filter standards
        standards = []
        for code, standard in STANDARDS_REGISTRY.items():
            # Apply type filter
            if standard_type != StandardType.ALL:
                if standard.get("type") != standard_type.value:
                    continue

            # Apply organization filter
            if organization and organization.lower() not in standard.get("organization", "").lower():
                continue

            standards.append(standard)

        if not standards:
            print_info("No standards found matching filters")
            raise typer.Exit(0)

        # Display results
        if output_format == "json":
            console.print_json(data=standards)
        elif output_format == "yaml":
            console.print(yaml.dump(standards, default_flow_style=False))
        else:
            from rich.table import Table

            table = Table(title=f"Standards Library ({len(standards)} standards)")
            table.add_column("Code", style="cyan", no_wrap=True)
            table.add_column("Name", style="green")
            table.add_column("Type", style="yellow")
            table.add_column("Organization", style="magenta")
            table.add_column("Version", style="dim")

            for standard in standards:
                table.add_row(
                    standard["code"],
                    standard["name"][:35],
                    standard.get("type", "N/A"),
                    standard.get("organization", "N/A")[:20],
                    standard.get("version", "N/A"),
                )

            console.print(table)

        console.print(f"\n[dim]Total: {len(standards)} standards[/dim]")

        # Type summary
        if standard_type == StandardType.ALL:
            console.print("\n[bold]Types:[/bold]")
            types = {}
            for s in STANDARDS_REGISTRY.values():
                t = s.get("type", "other")
                types[t] = types.get(t, 0) + 1
            for t, count in sorted(types.items()):
                console.print(f"  {t}: {count} standards")

        console.print()

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to list standards: {str(e)}")
        raise typer.Exit(1)


@app.command("info")
def standard_info(
    code: str = typer.Argument(
        ...,
        help="Standard code to show information for",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information including sections",
    ),
):
    """
    Show detailed information about a standard.

    Examples:
        gl standards info NFPA-86
        gl standards info ISO-14064 --verbose
    """
    try:
        # Find standard
        standard = STANDARDS_REGISTRY.get(code.upper()) or STANDARDS_REGISTRY.get(code)

        if not standard:
            print_error(f"Standard not found: {code}")
            raise typer.Exit(1)

        console.print(f"\n[bold cyan]Standard Information:[/bold cyan] {code}\n")

        # Display info
        console.print(create_info_panel("Standard Details", {
            "Code": standard["code"],
            "Name": standard["name"],
            "Type": standard.get("type", "N/A"),
            "Organization": standard.get("organization", "N/A"),
            "Version": standard.get("version", "N/A"),
        }))

        # Description
        console.print(f"\n[bold]Description:[/bold]")
        console.print(f"  {standard.get('description', 'No description')}")

        # Equipment types
        equipment = standard.get("equipment_types", [])
        if equipment and equipment != ["all"]:
            console.print(f"\n[bold]Applicable Equipment:[/bold]")
            for eq in equipment:
                console.print(f"  - {eq}")

        # Sections
        if verbose:
            sections = standard.get("sections", {})
            if sections:
                console.print(f"\n[bold]Sections ({len(sections)}):[/bold]")
                for key, title in sections.items():
                    console.print(f"  [cyan]{key}[/cyan]: {title}")

        # Usage
        console.print(f"\n[bold]Usage:[/bold]")
        console.print(f"  View sections: [cyan]gl standards section {code}[/cyan]")
        console.print(f"  Find equipment: [cyan]gl standards equipment <type>[/cyan]")

        console.print()

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to get standard info: {str(e)}")
        raise typer.Exit(1)
