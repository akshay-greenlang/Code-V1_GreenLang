"""
GL-FOUND-X-003: GreenLang Normalizer CLI - Configuration Commands

This module implements configuration management commands for initializing,
modifying, and displaying CLI configuration settings.

Configuration is stored at ~/.glnorm/config.yaml by default.

Example:
    >>> glnorm config init
    >>> glnorm config set api_key YOUR_API_KEY
    >>> glnorm config show
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import yaml

console = Console()

# Create Typer app for config commands
app = typer.Typer(
    name="config",
    help="Manage CLI configuration settings.",
    no_args_is_help=True,
)

# Default configuration directory and file
CONFIG_DIR_NAME = ".glnorm"
CONFIG_FILE_NAME = "config.yaml"

# Default configuration values
DEFAULT_CONFIG: Dict[str, Any] = {
    "api_url": "https://api.greenlang.io/normalizer/v1",
    "api_key": None,
    "default_policy_mode": "LENIENT",
    "default_output_format": "table",
    "vocabulary_version": "latest",
    "local_mode": False,
    "verbose": False,
    "cache_enabled": True,
    "cache_ttl_seconds": 3600,
    "timeout_seconds": 30,
}

# Configuration schema with descriptions
CONFIG_SCHEMA: Dict[str, Dict[str, Any]] = {
    "api_url": {
        "type": "string",
        "description": "Base URL for the normalization API",
        "example": "https://api.greenlang.io/normalizer/v1",
    },
    "api_key": {
        "type": "string",
        "description": "API key for authentication (can also use GLNORM_API_KEY env var)",
        "secret": True,
    },
    "default_policy_mode": {
        "type": "string",
        "description": "Default policy mode for normalizations",
        "choices": ["STRICT", "LENIENT"],
    },
    "default_output_format": {
        "type": "string",
        "description": "Default output format for results",
        "choices": ["json", "yaml", "table", "csv"],
    },
    "vocabulary_version": {
        "type": "string",
        "description": "Default vocabulary version to use",
        "example": "2026.01.0 or 'latest'",
    },
    "local_mode": {
        "type": "boolean",
        "description": "Use local normalization engine by default",
    },
    "verbose": {
        "type": "boolean",
        "description": "Enable verbose output by default",
    },
    "cache_enabled": {
        "type": "boolean",
        "description": "Enable local caching of vocabulary data",
    },
    "cache_ttl_seconds": {
        "type": "integer",
        "description": "Cache time-to-live in seconds",
    },
    "timeout_seconds": {
        "type": "integer",
        "description": "API request timeout in seconds",
    },
}


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    # Check for custom config dir via environment variable
    custom_dir = os.environ.get("GLNORM_CONFIG_DIR")
    if custom_dir:
        return Path(custom_dir)

    # Use ~/.glnorm by default
    return Path.home() / CONFIG_DIR_NAME


def get_config_path() -> Path:
    """Get the configuration file path."""
    return get_config_dir() / CONFIG_FILE_NAME


def load_config() -> Dict[str, Any]:
    """
    Load configuration from file.

    Returns:
        Configuration dictionary with defaults applied
    """
    config = DEFAULT_CONFIG.copy()
    config_path = get_config_path()

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f) or {}
                config.update(file_config)
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Could not load config: {e}")

    # Override with environment variables
    env_mappings = {
        "GLNORM_API_KEY": "api_key",
        "GLNORM_API_URL": "api_url",
        "GLNORM_POLICY_MODE": "default_policy_mode",
        "GLNORM_VOCAB_VERSION": "vocabulary_version",
        "GLNORM_LOCAL_MODE": "local_mode",
        "GLNORM_VERBOSE": "verbose",
    }

    for env_var, config_key in env_mappings.items():
        env_value = os.environ.get(env_var)
        if env_value is not None:
            # Handle boolean conversion
            if config_key in ("local_mode", "verbose", "cache_enabled"):
                config[config_key] = env_value.lower() in ("true", "1", "yes")
            elif config_key in ("cache_ttl_seconds", "timeout_seconds"):
                config[config_key] = int(env_value)
            else:
                config[config_key] = env_value

    return config


def save_config(config: Dict[str, Any]) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary to save
    """
    config_dir = get_config_dir()
    config_path = get_config_path()

    # Create directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)

    # Write config file
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


@app.command(name="init")
def init_config(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing configuration file.",
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        "-i/-I",
        help="Run interactive setup wizard.",
    ),
) -> None:
    """
    Initialize a new configuration file.

    Creates a configuration file at ~/.glnorm/config.yaml with default
    settings. Use --interactive to set up common options.

    [bold]Examples:[/bold]

        glnorm config init

        glnorm config init --force

        glnorm config init --no-interactive
    """
    config_path = get_config_path()

    if config_path.exists() and not force:
        console.print(
            f"[yellow]Configuration file already exists at:[/yellow] {config_path}"
        )
        console.print("Use --force to overwrite.")
        raise typer.Exit(code=1)

    config = DEFAULT_CONFIG.copy()

    if interactive:
        console.print()
        console.print("[bold cyan]GreenLang Normalizer CLI Configuration Wizard[/bold cyan]")
        console.print()

        # API Key
        api_key = typer.prompt(
            "API Key (press Enter to skip, can be set later)",
            default="",
            show_default=False,
        )
        if api_key:
            config["api_key"] = api_key

        # API URL
        api_url = typer.prompt(
            "API URL",
            default=config["api_url"],
        )
        config["api_url"] = api_url

        # Policy Mode
        policy_mode = typer.prompt(
            "Default policy mode (STRICT/LENIENT)",
            default=config["default_policy_mode"],
        )
        if policy_mode.upper() in ("STRICT", "LENIENT"):
            config["default_policy_mode"] = policy_mode.upper()

        # Output Format
        output_format = typer.prompt(
            "Default output format (json/yaml/table/csv)",
            default=config["default_output_format"],
        )
        if output_format.lower() in ("json", "yaml", "table", "csv"):
            config["default_output_format"] = output_format.lower()

        # Local Mode
        local_mode = typer.confirm(
            "Use local mode by default (no API calls)?",
            default=config["local_mode"],
        )
        config["local_mode"] = local_mode

        console.print()

    # Save configuration
    try:
        save_config(config)
        console.print(f"[green]Configuration saved to:[/green] {config_path}")
        console.print()
        console.print("[dim]You can modify settings with 'glnorm config set <key> <value>'[/dim]")

    except Exception as e:
        console.print(f"[red]Error saving configuration:[/red] {e}")
        raise typer.Exit(code=1)


@app.command(name="set")
def set_config_value(
    key: str = typer.Argument(
        ...,
        help="Configuration key to set.",
    ),
    value: str = typer.Argument(
        ...,
        help="Value to set for the key.",
    ),
) -> None:
    """
    Set a configuration value.

    Updates a single configuration setting in the config file.

    [bold]Available Keys:[/bold]

        api_url             - API endpoint URL
        api_key             - API authentication key
        default_policy_mode - STRICT or LENIENT
        default_output_format - json, yaml, table, or csv
        vocabulary_version  - Version string or 'latest'
        local_mode          - true/false
        verbose             - true/false
        cache_enabled       - true/false
        cache_ttl_seconds   - Cache TTL in seconds
        timeout_seconds     - API timeout in seconds

    [bold]Examples:[/bold]

        glnorm config set api_key YOUR_API_KEY

        glnorm config set default_policy_mode STRICT

        glnorm config set local_mode true

        glnorm config set timeout_seconds 60
    """
    if key not in CONFIG_SCHEMA:
        console.print(f"[red]Error:[/red] Unknown configuration key: {key}")
        console.print("[dim]Available keys:[/dim]")
        for k in CONFIG_SCHEMA:
            console.print(f"  - {k}")
        raise typer.Exit(code=1)

    # Load existing config
    config = load_config()

    # Validate and convert value
    schema = CONFIG_SCHEMA[key]
    try:
        if schema["type"] == "boolean":
            converted_value = value.lower() in ("true", "1", "yes", "on")
        elif schema["type"] == "integer":
            converted_value = int(value)
        else:
            converted_value = value

        # Validate choices if applicable
        if "choices" in schema and converted_value not in schema["choices"]:
            # Try case-insensitive match for strings
            if schema["type"] == "string":
                for choice in schema["choices"]:
                    if value.upper() == choice.upper():
                        converted_value = choice
                        break
                else:
                    console.print(
                        f"[red]Error:[/red] Invalid value for {key}. "
                        f"Must be one of: {', '.join(schema['choices'])}"
                    )
                    raise typer.Exit(code=1)

        config[key] = converted_value
        save_config(config)

        # Show confirmation
        display_value = "***" if schema.get("secret") else str(converted_value)
        console.print(f"[green]Set {key} = {display_value}[/green]")

    except ValueError as e:
        console.print(f"[red]Error:[/red] Invalid value for {key}: {e}")
        raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to save configuration: {e}")
        raise typer.Exit(code=1)


@app.command(name="get")
def get_config_value(
    key: str = typer.Argument(
        ...,
        help="Configuration key to retrieve.",
    ),
) -> None:
    """
    Get a configuration value.

    Retrieves and displays a single configuration setting.

    [bold]Example:[/bold]

        glnorm config get api_url
    """
    if key not in CONFIG_SCHEMA:
        console.print(f"[red]Error:[/red] Unknown configuration key: {key}")
        raise typer.Exit(code=1)

    config = load_config()
    value = config.get(key, DEFAULT_CONFIG.get(key))

    schema = CONFIG_SCHEMA[key]
    if schema.get("secret") and value:
        console.print(f"{key}: ***")
    else:
        console.print(f"{key}: {value}")


@app.command(name="show")
def show_config(
    show_secrets: bool = typer.Option(
        False,
        "--show-secrets",
        help="Show secret values (like API key) in output.",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json, or yaml.",
    ),
) -> None:
    """
    Display current configuration.

    Shows all configuration settings with their current values,
    including whether they come from file, environment, or defaults.

    [bold]Examples:[/bold]

        glnorm config show

        glnorm config show --show-secrets

        glnorm config show --format yaml
    """
    config = load_config()
    config_path = get_config_path()

    if format == "json":
        import json

        output_config = config.copy()
        if not show_secrets:
            for key, schema in CONFIG_SCHEMA.items():
                if schema.get("secret") and output_config.get(key):
                    output_config[key] = "***"
        console.print(json.dumps(output_config, indent=2))
        return

    if format == "yaml":
        output_config = config.copy()
        if not show_secrets:
            for key, schema in CONFIG_SCHEMA.items():
                if schema.get("secret") and output_config.get(key):
                    output_config[key] = "***"
        console.print(yaml.dump(output_config, default_flow_style=False, sort_keys=False))
        return

    # Table format
    console.print()
    console.print(f"[bold cyan]Configuration[/bold cyan]")
    console.print(f"[dim]File: {config_path}[/dim]")
    console.print()

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Description", style="dim")
    table.add_column("Source", style="dim")

    for key, schema in CONFIG_SCHEMA.items():
        value = config.get(key, DEFAULT_CONFIG.get(key))

        # Determine source
        env_var = f"GLNORM_{key.upper()}"
        if os.environ.get(env_var):
            source = "env"
        elif config_path.exists():
            source = "file"
        else:
            source = "default"

        # Mask secrets
        if schema.get("secret") and value and not show_secrets:
            display_value = "***" + str(value)[-4:] if value else "(not set)"
        else:
            display_value = str(value) if value is not None else "(not set)"

        table.add_row(
            key,
            display_value,
            schema.get("description", "")[:40],
            source,
        )

    console.print(table)
    console.print()
    console.print("[dim]Use 'glnorm config set <key> <value>' to modify settings[/dim]")


@app.command(name="unset")
def unset_config_value(
    key: str = typer.Argument(
        ...,
        help="Configuration key to remove.",
    ),
) -> None:
    """
    Remove a configuration value (reset to default).

    Removes the specified key from the configuration file,
    causing it to use the default value.

    [bold]Example:[/bold]

        glnorm config unset api_key
    """
    if key not in CONFIG_SCHEMA:
        console.print(f"[red]Error:[/red] Unknown configuration key: {key}")
        raise typer.Exit(code=1)

    config_path = get_config_path()
    if not config_path.exists():
        console.print("[yellow]No configuration file exists.[/yellow]")
        raise typer.Exit(code=0)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        if key in config:
            del config[key]
            save_config(config)
            console.print(f"[green]Removed {key} from configuration (using default)[/green]")
        else:
            console.print(f"[yellow]{key} was not set in configuration[/yellow]")

    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to update configuration: {e}")
        raise typer.Exit(code=1)


@app.command(name="path")
def show_config_path() -> None:
    """
    Show the configuration file path.

    Displays the full path to the configuration file and directory.

    [bold]Example:[/bold]

        glnorm config path
    """
    config_dir = get_config_dir()
    config_path = get_config_path()

    console.print(f"Config directory: {config_dir}")
    console.print(f"Config file:      {config_path}")
    console.print(f"File exists:      {'yes' if config_path.exists() else 'no'}")

    # Show environment variable if set
    if os.environ.get("GLNORM_CONFIG_DIR"):
        console.print(f"GLNORM_CONFIG_DIR: {os.environ.get('GLNORM_CONFIG_DIR')}")


@app.command(name="reset")
def reset_config(
    confirm: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt.",
    ),
) -> None:
    """
    Reset configuration to defaults.

    Removes the configuration file, causing all settings to use defaults.

    [bold]Example:[/bold]

        glnorm config reset --yes
    """
    config_path = get_config_path()

    if not config_path.exists():
        console.print("[yellow]No configuration file exists.[/yellow]")
        raise typer.Exit(code=0)

    if not confirm:
        confirm = typer.confirm(
            f"This will delete {config_path}. Continue?",
            default=False,
        )

    if confirm:
        try:
            config_path.unlink()
            console.print("[green]Configuration reset to defaults.[/green]")
        except Exception as e:
            console.print(f"[red]Error:[/red] Failed to delete configuration: {e}")
            raise typer.Exit(code=1)
    else:
        console.print("[yellow]Cancelled.[/yellow]")
