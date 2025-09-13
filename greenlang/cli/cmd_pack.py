"""
Pack management commands for GreenLang CLI
"""

import click
import json
import yaml
import sys
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from rich.syntax import Syntax

from greenlang.packs.manifest import PackManifest, Contents

console = Console()


@click.group()
def pack():
    """Pack management commands"""
    pass


@pack.command()
@click.argument('pack_type', type=click.Choice(['pack-basic', 'dataset', 'connector']), default='pack-basic')
@click.argument('name')
@click.option('--path', '-p', type=click.Path(), help='Directory to create pack in')
@click.option('--author', help='Author name')
@click.option('--description', help='Pack description')
def init(pack_type: str, name: str, path: Optional[str], author: Optional[str], description: Optional[str]):
    """
    Initialize a new pack with v1.0 specification
    
    Examples:
        gl pack init pack-basic my-pack
        gl pack init dataset weather-data --path ./packs
        gl pack init connector api-connector --author "John Doe"
    """
    # Validate pack name
    import re
    if not re.match(r'^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$', name):
        console.print(f"[red]Invalid pack name '{name}'[/red]")
        console.print("Pack name must be DNS-safe: lowercase, alphanumeric, hyphens only")
        sys.exit(1)
    
    # Determine target directory
    target_dir = Path(path) / name if path else Path(name)
    
    if target_dir.exists():
        console.print(f"[red]Directory '{target_dir}' already exists[/red]")
        sys.exit(1)
    
    # Get template directory
    template_dir = Path(__file__).parent / 'templates' / pack_type
    if not template_dir.exists():
        # Fallback to pack-basic if template doesn't exist
        template_dir = Path(__file__).parent / 'templates' / 'pack_basic'
    
    try:
        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy template files
        for template_file in template_dir.glob('**/*'):
            if template_file.is_file():
                rel_path = template_file.relative_to(template_dir)
                target_file = target_dir / rel_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Read template and replace placeholders
                content = template_file.read_text()
                content = content.replace('{{PACK_NAME}}', name)
                content = content.replace('{{PACK_DESCRIPTION}}', description or f'A GreenLang {pack_type}')
                content = content.replace('{{AUTHOR_NAME}}', author or 'Unknown')
                
                # Write to target
                target_file.write_text(content)
        
        # Create additional directories
        (target_dir / 'input').mkdir(exist_ok=True)
        (target_dir / 'output').mkdir(exist_ok=True)
        (target_dir / 'tests').mkdir(exist_ok=True)
        
        # Create .gitignore
        gitignore_content = """# GreenLang Pack
output/
*.pyc
__pycache__/
.env
.venv/
*.log
.DS_Store
"""
        (target_dir / '.gitignore').write_text(gitignore_content)
        
        # Validate the created pack
        manifest_file = target_dir / 'pack.yaml'
        try:
            manifest = PackManifest.from_file(manifest_file)
            console.print(f"[green]SUCCESS[/green] Created pack '{manifest.name}' v{manifest.version}")
            console.print(f"  Location: {target_dir}")
            console.print(f"  Type: {manifest.kind}")
            console.print(f"  License: {manifest.license}")
            console.print("\nNext steps:")
            console.print(f"  1. cd {target_dir}")
            console.print(f"  2. gl pack validate")
            console.print(f"  3. gl run gl.yaml")
        except Exception as e:
            console.print(f"[yellow]Warning: Created pack but validation failed: {e}[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Failed to create pack: {e}[/red]")
        # Clean up on failure
        if target_dir.exists():
            shutil.rmtree(target_dir)
        sys.exit(1)


@pack.command()
@click.argument('pack_path', type=click.Path(exists=True), default='.')
@click.option('--json', 'output_json', is_flag=True, help='Output results as JSON')
@click.option('--strict', is_flag=True, help='Fail on warnings')
@click.option('--schema', type=click.Path(exists=True), help='Path to JSON schema file')
def validate(pack_path: str, output_json: bool, strict: bool, schema: Optional[str]):
    """
    Validate a pack manifest against v1.0 specification
    
    Examples:
        gl pack validate                  # Validate current directory
        gl pack validate ./my-pack        # Validate specific pack
        gl pack validate --json           # JSON output
        gl pack validate --strict         # Fail on warnings
    """
    pack_path = Path(pack_path)
    
    # Find manifest file
    manifest_file = None
    if pack_path.is_file() and pack_path.name in ['pack.yaml', 'pack.yml', 'pack.json']:
        manifest_file = pack_path
        pack_dir = pack_path.parent
    else:
        # Look for manifest in directory
        for name in ['pack.yaml', 'pack.yml', 'pack.json']:
            candidate = pack_path / name
            if candidate.exists():
                manifest_file = candidate
                pack_dir = pack_path
                break
    
    if not manifest_file:
        error_msg = f"No pack manifest found in {pack_path}"
        if output_json:
            result = {
                'status': 'ERROR',
                'error': error_msg,
                'path': str(pack_path)
            }
            console.print(json.dumps(result, indent=2))
        else:
            console.print(f"[red]ERROR[/red] {error_msg}")
        sys.exit(1)
    
    # Validation results
    errors = []
    warnings = []
    
    try:
        # Load and parse manifest
        manifest = PackManifest.from_file(manifest_file)
        
        # Check file existence
        missing_files = manifest.validate_files_exist(pack_dir)
        if missing_files:
            errors.extend(missing_files)
        
        # Get warnings
        warnings = manifest.get_warnings()
        
        # Validate against JSON schema if provided
        if schema:
            schema_errors = validate_against_schema(manifest_file, schema)
            if schema_errors:
                errors.extend(schema_errors)
        
        # Determine overall status
        if errors:
            status = 'FAIL'
        elif warnings and strict:
            status = 'FAIL'
        else:
            status = 'PASS'
        
        # Output results
        if output_json:
            result = {
                'status': status,
                'manifest': {
                    'name': manifest.name,
                    'version': manifest.version,
                    'kind': manifest.kind,
                    'license': manifest.license
                },
                'errors': errors,
                'warnings': warnings,
                'path': str(manifest_file),
                'spec_version': '1.0'
            }
            console.print(json.dumps(result, indent=2))
        else:
            # Pretty output
            if status == 'PASS':
                console.print(f"\n[green]PASS[/green] Pack validation [green]PASSED[/green]")
                console.print(f"  Name: [cyan]{manifest.name}[/cyan]")
                console.print(f"  Version: [cyan]{manifest.version}[/cyan]")
                console.print(f"  Kind: [cyan]{manifest.kind}[/cyan]")
                console.print(f"  License: [cyan]{manifest.license}[/cyan]")
                console.print(f"  Spec: v1.0")
                
                if warnings:
                    console.print(f"\n[yellow]Warnings ({len(warnings)}):[/yellow]")
                    for warning in warnings:
                        console.print(f"  WARNING: {warning}")
            else:
                console.print(f"\n[red]FAIL[/red] Pack validation [red]FAILED[/red]")
                console.print(f"  Path: {manifest_file}")
                
                if errors:
                    console.print(f"\n[red]Errors ({len(errors)}):[/red]")
                    for error in errors:
                        console.print(f"  ERROR: {error}")
                
                if warnings:
                    console.print(f"\n[yellow]Warnings ({len(warnings)}):[/yellow]")
                    for warning in warnings:
                        console.print(f"  WARNING: {warning}")
        
        # Exit code
        sys.exit(0 if status == 'PASS' else 1)
        
    except Exception as e:
        # Handle parsing errors
        error_msg = str(e)
        
        if output_json:
            result = {
                'status': 'ERROR',
                'error': error_msg,
                'path': str(manifest_file)
            }
            console.print(json.dumps(result, indent=2))
        else:
            console.print(f"[red]ERROR[/red] Validation error: {error_msg}")
            
            # Provide helpful hints for common errors
            if "name" in error_msg.lower():
                console.print("\n[dim]Hint: Pack name must be DNS-safe (lowercase, alphanumeric, hyphens)[/dim]")
            elif "version" in error_msg.lower():
                console.print("\n[dim]Hint: Version must be semantic (e.g., 1.0.0)[/dim]")
            elif "contents" in error_msg.lower():
                console.print("\n[dim]Hint: Contents must have at least one pipeline defined[/dim]")
        
        sys.exit(1)


@pack.command()
@click.argument('directory', type=click.Path(), default='.')
def list(directory: str):
    """List all packs in a directory"""
    directory = Path(directory)
    
    # Find all pack.yaml files
    packs = []
    for pack_file in directory.glob('**/pack.yaml'):
        try:
            manifest = PackManifest.from_file(pack_file)
            packs.append({
                'name': manifest.name,
                'version': manifest.version,
                'kind': manifest.kind,
                'path': str(pack_file.parent.relative_to(directory))
            })
        except Exception as e:
            # Skip invalid packs
            continue
    
    if not packs:
        console.print("[yellow]No packs found[/yellow]")
        return
    
    # Display table
    table = Table(title=f"Packs in {directory}")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Kind", style="yellow")
    table.add_column("Path", style="dim")
    
    for pack in sorted(packs, key=lambda p: p['name']):
        table.add_row(
            pack['name'],
            pack['version'],
            pack['kind'],
            pack['path']
        )
    
    console.print(table)


@pack.command()
@click.argument('pack_path', type=click.Path(exists=True), default='.')
def info(pack_path: str):
    """Show detailed information about a pack"""
    pack_path = Path(pack_path)
    
    # Find manifest
    if pack_path.is_file():
        manifest_file = pack_path
    else:
        manifest_file = pack_path / 'pack.yaml'
        if not manifest_file.exists():
            manifest_file = pack_path / 'pack.yml'
    
    if not manifest_file.exists():
        console.print(f"[red]No pack manifest found in {pack_path}[/red]")
        sys.exit(1)
    
    try:
        manifest = PackManifest.from_file(manifest_file)
        
        # Create info panel
        info_text = f"""
[bold]Pack Information[/bold]
  Name: [cyan]{manifest.name}[/cyan]
  Version: [green]{manifest.version}[/green]
  Kind: [yellow]{manifest.kind}[/yellow]
  License: {manifest.license}

[bold]Contents[/bold]
  Pipelines: {len(manifest.contents.pipelines)}
  Agents: {len(manifest.contents.agents)}
  Datasets: {len(manifest.contents.datasets)}
  Reports: {len(manifest.contents.reports)}
"""
        
        if manifest.compat:
            info_text += f"""
[bold]Compatibility[/bold]
  GreenLang: {manifest.compat.greenlang or 'Any'}
  Python: {manifest.compat.python or 'Any'}
"""
        
        if manifest.dependencies:
            info_text += f"""
[bold]Dependencies[/bold]
  Count: {len(manifest.dependencies)}
"""
        
        console.print(Panel(info_text.strip(), title=f"Pack: {manifest.name}", expand=False))
        
        # Show manifest content if verbose
        if click.get_current_context().parent.params.get('verbose'):
            console.print("\n[bold]Manifest Content:[/bold]")
            syntax = Syntax(manifest.to_yaml(), "yaml", theme="monokai", line_numbers=True)
            console.print(syntax)
        
    except Exception as e:
        console.print(f"[red]Error reading pack: {e}[/red]")
        sys.exit(1)


def validate_against_schema(manifest_file: Path, schema_file: str) -> List[str]:
    """Validate manifest against JSON schema"""
    errors = []
    
    try:
        import jsonschema
        
        # Load schema
        with open(schema_file, 'r') as f:
            schema = json.load(f)
        
        # Load manifest as JSON
        with open(manifest_file, 'r') as f:
            if manifest_file.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        # Validate
        validator = jsonschema.Draft7Validator(schema)
        for error in validator.iter_errors(data):
            error_path = '.'.join(str(p) for p in error.path) if error.path else 'root'
            errors.append(f"Schema validation: {error_path}: {error.message}")
            
    except ImportError:
        errors.append("jsonschema library not installed for schema validation")
    except Exception as e:
        errors.append(f"Schema validation error: {e}")
    
    return errors