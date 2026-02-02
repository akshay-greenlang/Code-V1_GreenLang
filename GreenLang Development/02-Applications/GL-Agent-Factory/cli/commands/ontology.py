"""
Ontology Management Commands

Commands for querying and managing the GreenLang industrial equipment ontology.
Supports SPARQL queries, equipment lookups, and OWL/XML export.
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


class ExportFormat(str, Enum):
    """Export format enumeration."""
    OWL = "owl"
    XML = "xml"
    JSON = "json"
    TURTLE = "turtle"
    NTRIPLES = "ntriples"


# Create ontology command group
app = typer.Typer(
    help="Ontology commands - query and manage the GreenLang equipment ontology",
    no_args_is_help=True,
)


# Simulated ontology data (would connect to actual RDF/OWL store)
ONTOLOGY_CLASSES = {
    "Equipment": {
        "uri": "gl:Equipment",
        "description": "Base class for all industrial equipment",
        "subclasses": ["ThermalEquipment", "ProcessEquipment", "UtilityEquipment"],
        "properties": ["hasManufacturer", "hasModel", "hasCapacity", "hasEmissionFactor"],
    },
    "ThermalEquipment": {
        "uri": "gl:ThermalEquipment",
        "description": "Equipment that generates or uses heat",
        "parent": "Equipment",
        "subclasses": ["Furnace", "Boiler", "Oven", "Dryer", "ThermalOxidizer"],
        "properties": ["hasMaxTemperature", "hasFuelType", "hasThermalEfficiency"],
    },
    "Furnace": {
        "uri": "gl:Furnace",
        "description": "Industrial furnace for heat treatment",
        "parent": "ThermalEquipment",
        "subclasses": ["DirectFiredFurnace", "IndirectFiredFurnace", "ElectricFurnace"],
        "properties": ["hasBurnerType", "hasAtmosphere", "hasControlSystem"],
        "standards": ["NFPA-86", "OSHA-1910.106"],
    },
    "DirectFiredFurnace": {
        "uri": "gl:DirectFiredFurnace",
        "description": "Furnace where combustion products contact the work",
        "parent": "Furnace",
        "properties": ["hasFlamePattern", "hasCombustionEfficiency"],
    },
    "IndirectFiredFurnace": {
        "uri": "gl:IndirectFiredFurnace",
        "description": "Furnace where combustion products do not contact the work",
        "parent": "Furnace",
        "properties": ["hasRadiantTubes", "hasHeatExchanger"],
    },
    "Boiler": {
        "uri": "gl:Boiler",
        "description": "Equipment for generating steam or hot water",
        "parent": "ThermalEquipment",
        "subclasses": ["FiretubeBoiler", "WatertubeBoiler"],
        "properties": ["hasSteamPressure", "hasSteamCapacity"],
        "standards": ["NFPA-85", "ASME-BPVC"],
    },
    "ProcessEquipment": {
        "uri": "gl:ProcessEquipment",
        "description": "Equipment for process operations",
        "parent": "Equipment",
        "subclasses": ["Reactor", "Separator", "Compressor", "Pump"],
        "properties": ["hasProcessType", "hasOperatingPressure"],
    },
    "Reactor": {
        "uri": "gl:Reactor",
        "description": "Chemical reactor equipment",
        "parent": "ProcessEquipment",
        "properties": ["hasReactionType", "hasResidenceTime", "hasCatalyst"],
        "standards": ["OSHA-1910.119"],
    },
    "EmissionSource": {
        "uri": "gl:EmissionSource",
        "description": "Source of greenhouse gas emissions",
        "subclasses": ["CombustionSource", "ProcessSource", "FugitiveSource"],
        "properties": ["hasEmissionFactor", "hasActivityData", "hasScope"],
    },
    "CombustionSource": {
        "uri": "gl:CombustionSource",
        "description": "Emission source from fuel combustion",
        "parent": "EmissionSource",
        "properties": ["hasFuelType", "hasFuelQuantity", "hasCO2Factor", "hasCH4Factor", "hasN2OFactor"],
    },
}

ONTOLOGY_PROPERTIES = {
    "hasManufacturer": {
        "uri": "gl:hasManufacturer",
        "domain": "Equipment",
        "range": "xsd:string",
        "description": "Equipment manufacturer name",
    },
    "hasEmissionFactor": {
        "uri": "gl:hasEmissionFactor",
        "domain": "EmissionSource",
        "range": "xsd:decimal",
        "unit": "kgCO2e/unit",
        "description": "Emission factor for the source",
    },
    "hasFuelType": {
        "uri": "gl:hasFuelType",
        "domain": "CombustionSource",
        "range": "gl:FuelType",
        "description": "Type of fuel used",
    },
    "hasMaxTemperature": {
        "uri": "gl:hasMaxTemperature",
        "domain": "ThermalEquipment",
        "range": "xsd:decimal",
        "unit": "degC",
        "description": "Maximum operating temperature",
    },
    "hasThermalEfficiency": {
        "uri": "gl:hasThermalEfficiency",
        "domain": "ThermalEquipment",
        "range": "xsd:decimal",
        "unit": "percent",
        "description": "Thermal efficiency percentage",
    },
    "hasScope": {
        "uri": "gl:hasScope",
        "domain": "EmissionSource",
        "range": "gl:EmissionScope",
        "description": "GHG Protocol scope (1, 2, or 3)",
    },
}


@app.command("query")
def query_ontology(
    sparql: str = typer.Argument(
        ...,
        help="SPARQL query to execute",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table/json/csv)",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        "-l",
        help="Maximum number of results",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show query execution details",
    ),
):
    """
    Run a SPARQL query against the ontology.

    Execute SPARQL SELECT queries against the GreenLang equipment ontology.

    Examples:
        gl ontology query "SELECT ?class WHERE { ?class a owl:Class }"
        gl ontology query "SELECT ?equip ?type WHERE { ?equip a gl:Furnace }"
        gl ontology query @query.sparql  # Load query from file
    """
    try:
        # Check if query is a file reference
        if sparql.startswith("@"):
            query_file = Path(sparql[1:])
            if not query_file.exists():
                print_error(f"Query file not found: {query_file}")
                raise typer.Exit(1)
            sparql = query_file.read_text()

        console.print(f"\n[bold cyan]Executing SPARQL query[/bold cyan]\n")

        if verbose:
            console.print("[bold]Query:[/bold]")
            console.print(f"  [dim]{sparql[:200]}{'...' if len(sparql) > 200 else ''}[/dim]\n")

        # Parse and execute query (simulated)
        results = _execute_sparql(sparql, limit)

        if not results["bindings"]:
            print_info("No results found")
            raise typer.Exit(0)

        # Display results
        if output_format == "json":
            console.print_json(data=results)
        elif output_format == "csv":
            # CSV format
            if results["variables"]:
                console.print(",".join(results["variables"]))
                for binding in results["bindings"]:
                    row = [str(binding.get(var, "")) for var in results["variables"]]
                    console.print(",".join(row))
        else:
            # Table format
            from rich.table import Table

            table = Table(title=f"Query Results ({len(results['bindings'])} rows)")

            for var in results["variables"]:
                table.add_column(var, style="cyan")

            for binding in results["bindings"][:limit]:
                row = [str(binding.get(var, "N/A")) for var in results["variables"]]
                table.add_row(*row)

            console.print(table)

        console.print(f"\n[dim]Returned {len(results['bindings'])} results[/dim]\n")

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Query failed: {str(e)}")
        raise typer.Exit(1)


@app.command("equipment")
def equipment_info(
    equipment_type: str = typer.Argument(
        ...,
        help="Equipment type to get information for",
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
        help="Show detailed information including parent classes",
    ),
):
    """
    Get ontology information for an equipment type.

    Retrieve class definition, properties, and relationships for equipment.

    Examples:
        gl ontology equipment Furnace
        gl ontology equipment DirectFiredFurnace --verbose
        gl ontology equipment Boiler --format json
    """
    try:
        # Normalize equipment type
        equipment_key = equipment_type.title().replace(" ", "").replace("_", "")

        console.print(f"\n[bold cyan]Equipment Ontology:[/bold cyan] {equipment_type}\n")

        # Find in ontology
        equipment_class = ONTOLOGY_CLASSES.get(equipment_key)

        if not equipment_class:
            # Try fuzzy match
            for key, cls in ONTOLOGY_CLASSES.items():
                if equipment_type.lower() in key.lower():
                    equipment_class = cls
                    equipment_key = key
                    break

        if not equipment_class:
            print_error(f"Equipment type not found: {equipment_type}")
            print_info("\nAvailable equipment types:")
            for key in sorted(ONTOLOGY_CLASSES.keys()):
                console.print(f"  - {key}")
            raise typer.Exit(1)

        # Display results
        if output_format == "json":
            console.print_json(data=equipment_class)
        elif output_format == "yaml":
            console.print(yaml.dump(equipment_class, default_flow_style=False))
        else:
            # Text format
            console.print(create_info_panel("Class Definition", {
                "Class": equipment_key,
                "URI": equipment_class.get("uri", "N/A"),
                "Parent": equipment_class.get("parent", "None"),
            }))

            console.print(f"\n[bold]Description:[/bold]")
            console.print(f"  {equipment_class.get('description', 'No description')}")

            # Subclasses
            subclasses = equipment_class.get("subclasses", [])
            if subclasses:
                console.print(f"\n[bold]Subclasses ({len(subclasses)}):[/bold]")
                for sub in subclasses:
                    console.print(f"  - [cyan]{sub}[/cyan]")

            # Properties
            properties = equipment_class.get("properties", [])
            if properties:
                console.print(f"\n[bold]Properties ({len(properties)}):[/bold]")
                for prop in properties:
                    prop_info = ONTOLOGY_PROPERTIES.get(prop, {})
                    prop_range = prop_info.get("range", "xsd:string")
                    console.print(f"  - [green]{prop}[/green]: {prop_range}")

            # Standards
            standards = equipment_class.get("standards", [])
            if standards:
                console.print(f"\n[bold]Applicable Standards:[/bold]")
                for std in standards:
                    console.print(f"  - [yellow]{std}[/yellow]")

            # Hierarchy (verbose)
            if verbose:
                console.print(f"\n[bold]Class Hierarchy:[/bold]")
                _print_hierarchy(equipment_key, indent=2)

        console.print()

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to get equipment info: {str(e)}")
        raise typer.Exit(1)


@app.command("export")
def export_ontology(
    output: Path = typer.Argument(
        ...,
        help="Output file path",
    ),
    export_format: ExportFormat = typer.Option(
        ExportFormat.OWL,
        "--format",
        "-f",
        help="Export format",
    ),
    include_instances: bool = typer.Option(
        False,
        "--include-instances",
        "-i",
        help="Include instance data",
    ),
    namespace: str = typer.Option(
        "https://greenlang.io/ontology#",
        "--namespace",
        "-n",
        help="Base namespace URI",
    ),
):
    """
    Export the ontology to OWL/XML format.

    Export the GreenLang equipment ontology to various RDF formats.

    Examples:
        gl ontology export equipment.owl
        gl ontology export equipment.xml --format xml
        gl ontology export equipment.ttl --format turtle
    """
    try:
        console.print(f"\n[bold cyan]Exporting ontology[/bold cyan]\n")

        console.print(create_info_panel("Export Configuration", {
            "Output": str(output),
            "Format": export_format.value,
            "Namespace": namespace,
            "Include Instances": "Yes" if include_instances else "No",
        }))
        console.print()

        with create_progress_bar() as progress:
            task = progress.add_task("Generating ontology export...", total=100)

            # Generate export content
            if export_format in [ExportFormat.OWL, ExportFormat.XML]:
                content = _generate_owl_export(namespace, include_instances)
            elif export_format == ExportFormat.TURTLE:
                content = _generate_turtle_export(namespace, include_instances)
            elif export_format == ExportFormat.NTRIPLES:
                content = _generate_ntriples_export(namespace, include_instances)
            else:
                content = _generate_json_export(namespace, include_instances)

            progress.update(task, completed=50)

            # Write to file
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content)

            progress.update(task, completed=100)

        print_success(f"Ontology exported to: {output}")

        # Statistics
        console.print(f"\n[bold]Export Statistics:[/bold]")
        console.print(f"  Classes: {len(ONTOLOGY_CLASSES)}")
        console.print(f"  Properties: {len(ONTOLOGY_PROPERTIES)}")
        console.print(f"  File Size: {output.stat().st_size:,} bytes")

        console.print()

    except Exception as e:
        print_error(f"Export failed: {str(e)}")
        raise typer.Exit(1)


@app.command("list")
def list_classes(
    parent: Optional[str] = typer.Option(
        None,
        "--parent",
        "-p",
        help="Filter by parent class",
    ),
    output_format: str = typer.Option(
        "tree",
        "--format",
        "-f",
        help="Output format (tree/table/json)",
    ),
):
    """
    List all ontology classes.

    Display the class hierarchy of the GreenLang equipment ontology.

    Examples:
        gl ontology list
        gl ontology list --parent Equipment
        gl ontology list --format table
    """
    try:
        console.print("\n[bold cyan]Ontology Classes[/bold cyan]\n")

        if output_format == "json":
            console.print_json(data=ONTOLOGY_CLASSES)
        elif output_format == "table":
            from rich.table import Table

            table = Table(title="Ontology Classes")
            table.add_column("Class", style="cyan")
            table.add_column("Parent", style="green")
            table.add_column("Subclasses", style="yellow")
            table.add_column("Properties", style="dim")

            for name, cls in sorted(ONTOLOGY_CLASSES.items()):
                if parent and cls.get("parent") != parent:
                    continue

                table.add_row(
                    name,
                    cls.get("parent", "-"),
                    str(len(cls.get("subclasses", []))),
                    str(len(cls.get("properties", []))),
                )

            console.print(table)
        else:
            # Tree format
            from rich.tree import Tree

            tree = Tree("[bold]GreenLang Equipment Ontology[/bold]")

            def add_children(parent_tree, parent_name):
                for name, cls in ONTOLOGY_CLASSES.items():
                    if cls.get("parent") == parent_name:
                        subclasses = cls.get("subclasses", [])
                        label = f"[cyan]{name}[/cyan]"
                        if subclasses:
                            label += f" [dim]({len(subclasses)} subclasses)[/dim]"
                        branch = parent_tree.add(label)
                        add_children(branch, name)

            # Add root classes
            for name, cls in ONTOLOGY_CLASSES.items():
                if "parent" not in cls:
                    subclasses = cls.get("subclasses", [])
                    label = f"[cyan]{name}[/cyan]"
                    if subclasses:
                        label += f" [dim]({len(subclasses)} subclasses)[/dim]"
                    branch = tree.add(label)
                    add_children(branch, name)

            console.print(tree)

        console.print(f"\n[dim]Total: {len(ONTOLOGY_CLASSES)} classes[/dim]\n")

    except Exception as e:
        print_error(f"Failed to list classes: {str(e)}")
        raise typer.Exit(1)


@app.command("properties")
def list_properties(
    domain: Optional[str] = typer.Option(
        None,
        "--domain",
        "-d",
        help="Filter by domain class",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table/json)",
    ),
):
    """
    List all ontology properties.

    Display data and object properties in the ontology.

    Examples:
        gl ontology properties
        gl ontology properties --domain Equipment
    """
    try:
        console.print("\n[bold cyan]Ontology Properties[/bold cyan]\n")

        # Filter properties
        properties = {}
        for name, prop in ONTOLOGY_PROPERTIES.items():
            if domain and prop.get("domain") != domain:
                continue
            properties[name] = prop

        if not properties:
            print_info("No properties found matching filter")
            raise typer.Exit(0)

        if output_format == "json":
            console.print_json(data=properties)
        else:
            from rich.table import Table

            table = Table(title="Ontology Properties")
            table.add_column("Property", style="cyan")
            table.add_column("Domain", style="green")
            table.add_column("Range", style="yellow")
            table.add_column("Unit", style="magenta")
            table.add_column("Description", style="dim")

            for name, prop in sorted(properties.items()):
                table.add_row(
                    name,
                    prop.get("domain", "-"),
                    prop.get("range", "-"),
                    prop.get("unit", "-"),
                    prop.get("description", "")[:30],
                )

            console.print(table)

        console.print(f"\n[dim]Total: {len(properties)} properties[/dim]\n")

    except typer.Exit:
        raise
    except Exception as e:
        print_error(f"Failed to list properties: {str(e)}")
        raise typer.Exit(1)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _execute_sparql(query: str, limit: int) -> dict:
    """Execute a SPARQL query (simulated)."""
    # Simulated SPARQL execution
    results = {
        "variables": [],
        "bindings": [],
    }

    query_lower = query.lower()

    # Parse SELECT variables
    if "select" in query_lower:
        # Extract variable names
        import re
        var_match = re.search(r'select\s+(.+?)\s+where', query_lower)
        if var_match:
            vars_str = var_match.group(1)
            results["variables"] = [v.strip().replace("?", "") for v in vars_str.split() if v.startswith("?")]

    # Generate mock results based on query
    if "class" in query_lower or "owl:class" in query_lower:
        results["variables"] = ["class"]
        for cls in list(ONTOLOGY_CLASSES.keys())[:limit]:
            results["bindings"].append({"class": f"gl:{cls}"})

    elif "furnace" in query_lower:
        results["variables"] = ["equipment", "type"]
        results["bindings"] = [
            {"equipment": "gl:Furnace001", "type": "gl:DirectFiredFurnace"},
            {"equipment": "gl:Furnace002", "type": "gl:IndirectFiredFurnace"},
            {"equipment": "gl:Furnace003", "type": "gl:ElectricFurnace"},
        ]

    elif "property" in query_lower:
        results["variables"] = ["property", "domain", "range"]
        for name, prop in list(ONTOLOGY_PROPERTIES.items())[:limit]:
            results["bindings"].append({
                "property": prop["uri"],
                "domain": prop.get("domain", ""),
                "range": prop.get("range", ""),
            })

    else:
        # Generic results
        results["variables"] = ["subject", "predicate", "object"]
        results["bindings"] = [
            {"subject": "gl:Equipment", "predicate": "rdf:type", "object": "owl:Class"},
            {"subject": "gl:Furnace", "predicate": "rdfs:subClassOf", "object": "gl:ThermalEquipment"},
        ]

    return results


def _print_hierarchy(class_name: str, indent: int = 0):
    """Print class hierarchy."""
    cls = ONTOLOGY_CLASSES.get(class_name, {})
    parent = cls.get("parent")

    if parent:
        _print_hierarchy(parent, indent)

    prefix = "  " * indent
    console.print(f"{prefix}[cyan]{class_name}[/cyan]")


def _generate_owl_export(namespace: str, include_instances: bool) -> str:
    """Generate OWL/XML export."""
    lines = [
        '<?xml version="1.0"?>',
        f'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
        '         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"',
        '         xmlns:owl="http://www.w3.org/2002/07/owl#"',
        f'         xmlns:gl="{namespace}">',
        '',
        f'  <owl:Ontology rdf:about="{namespace}">',
        '    <rdfs:label>GreenLang Equipment Ontology</rdfs:label>',
        '    <rdfs:comment>Industrial equipment ontology for sustainability applications</rdfs:comment>',
        '  </owl:Ontology>',
        '',
    ]

    # Add classes
    for name, cls in ONTOLOGY_CLASSES.items():
        lines.append(f'  <owl:Class rdf:about="{namespace}{name}">')
        lines.append(f'    <rdfs:label>{name}</rdfs:label>')
        if "description" in cls:
            lines.append(f'    <rdfs:comment>{cls["description"]}</rdfs:comment>')
        if "parent" in cls:
            lines.append(f'    <rdfs:subClassOf rdf:resource="{namespace}{cls["parent"]}"/>')
        lines.append('  </owl:Class>')
        lines.append('')

    # Add properties
    for name, prop in ONTOLOGY_PROPERTIES.items():
        prop_type = "owl:DatatypeProperty" if prop["range"].startswith("xsd:") else "owl:ObjectProperty"
        lines.append(f'  <{prop_type} rdf:about="{namespace}{name}">')
        lines.append(f'    <rdfs:label>{name}</rdfs:label>')
        if "description" in prop:
            lines.append(f'    <rdfs:comment>{prop["description"]}</rdfs:comment>')
        lines.append(f'    <rdfs:domain rdf:resource="{namespace}{prop["domain"]}"/>')
        if prop["range"].startswith("xsd:"):
            lines.append(f'    <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#{prop["range"][4:]}"/>')
        else:
            lines.append(f'    <rdfs:range rdf:resource="{namespace}{prop["range"].replace("gl:", "")}"/>')
        lines.append(f'  </{prop_type}>')
        lines.append('')

    lines.append('</rdf:RDF>')

    return '\n'.join(lines)


def _generate_turtle_export(namespace: str, include_instances: bool) -> str:
    """Generate Turtle format export."""
    lines = [
        f'@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .',
        f'@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .',
        f'@prefix owl: <http://www.w3.org/2002/07/owl#> .',
        f'@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .',
        f'@prefix gl: <{namespace}> .',
        '',
        f'<{namespace}> a owl:Ontology ;',
        '    rdfs:label "GreenLang Equipment Ontology" .',
        '',
    ]

    for name, cls in ONTOLOGY_CLASSES.items():
        lines.append(f'gl:{name} a owl:Class ;')
        lines.append(f'    rdfs:label "{name}" ;')
        if "description" in cls:
            lines.append(f'    rdfs:comment "{cls["description"]}" ;')
        if "parent" in cls:
            lines.append(f'    rdfs:subClassOf gl:{cls["parent"]} ;')
        lines[-1] = lines[-1].rstrip(' ;') + ' .'
        lines.append('')

    return '\n'.join(lines)


def _generate_ntriples_export(namespace: str, include_instances: bool) -> str:
    """Generate N-Triples format export."""
    lines = []

    for name, cls in ONTOLOGY_CLASSES.items():
        uri = f'<{namespace}{name}>'
        lines.append(f'{uri} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2002/07/owl#Class> .')
        lines.append(f'{uri} <http://www.w3.org/2000/01/rdf-schema#label> "{name}" .')
        if "parent" in cls:
            lines.append(f'{uri} <http://www.w3.org/2000/01/rdf-schema#subClassOf> <{namespace}{cls["parent"]}> .')

    return '\n'.join(lines)


def _generate_json_export(namespace: str, include_instances: bool) -> str:
    """Generate JSON-LD format export."""
    export = {
        "@context": {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "gl": namespace,
        },
        "@graph": [],
    }

    for name, cls in ONTOLOGY_CLASSES.items():
        node = {
            "@id": f"gl:{name}",
            "@type": "owl:Class",
            "rdfs:label": name,
        }
        if "description" in cls:
            node["rdfs:comment"] = cls["description"]
        if "parent" in cls:
            node["rdfs:subClassOf"] = {"@id": f"gl:{cls['parent']}"}

        export["@graph"].append(node)

    return json.dumps(export, indent=2)
