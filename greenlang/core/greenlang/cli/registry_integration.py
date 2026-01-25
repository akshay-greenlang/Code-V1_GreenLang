"""
Registry CLI Integration
Add this to main.py after console = Console():

# Import and register registry subcommands
from . import cmd_registry
app.add_typer(cmd_registry.app, name="agent")
"""
