# -*- coding: utf-8 -*-
"""
RBAC Management CLI Commands
=============================

CLI commands for managing agent-level Role-Based Access Control (RBAC).

Commands:
    greenlang rbac grant <agent_id> <user> <role>    - Grant role to user
    greenlang rbac revoke <agent_id> <user> <role>   - Revoke role from user
    greenlang rbac list <agent_id>                   - List all grants for agent
    greenlang rbac audit <user>                      - Audit user permissions
    greenlang rbac roles                             - List available roles
    greenlang rbac check <agent_id> <user> <permission> - Check permission

Example:
    $ greenlang rbac grant GL-001 user@example.com agent_operator
    $ greenlang rbac list GL-001
    $ greenlang rbac audit user@example.com
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
import json

# Import policy enforcer and RBAC components
try:
    from greenlang.policy.enforcer import PolicyEnforcer
    from greenlang.core.greenlang.policy.agent_rbac import (
        AgentPermission,
        PREDEFINED_ROLES
    )
except ImportError as e:
    print(f"Error: Could not import RBAC modules: {e}")
    raise typer.Exit(1)


app = typer.Typer(
    name="rbac",
    help="Manage agent-level Role-Based Access Control",
    no_args_is_help=True,
)

console = Console()


def get_enforcer() -> PolicyEnforcer:
    """Get PolicyEnforcer instance."""
    return PolicyEnforcer()


@app.command()
def grant(
    agent_id: str = typer.Argument(..., help="Agent ID (e.g., GL-001)"),
    user: str = typer.Argument(..., help="User email or identifier"),
    role: str = typer.Argument(..., help="Role to grant"),
):
    """
    Grant role to user for agent.

    Example:
        greenlang rbac grant GL-001 user@example.com agent_operator
    """
    try:
        enforcer = get_enforcer()

        # Validate role exists
        if role not in PREDEFINED_ROLES:
            console.print(f"[red]Error:[/red] Role '{role}' does not exist")
            console.print("\n[yellow]Available roles:[/yellow]")
            for role_name, role_obj in PREDEFINED_ROLES.items():
                console.print(f"  • {role_name}: {role_obj.description}")
            raise typer.Exit(1)

        # Grant role
        enforcer.grant_agent_role(agent_id, user, role)

        console.print(f"[green]✓[/green] Granted role [bold]{role}[/bold] to [bold]{user}[/bold] for agent [bold]{agent_id}[/bold]")

        # Show granted permissions
        role_obj = PREDEFINED_ROLES[role]
        permissions_str = ", ".join([p.value for p in role_obj.permissions])
        console.print(f"\n[dim]Permissions granted:[/dim] {permissions_str}")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def revoke(
    agent_id: str = typer.Argument(..., help="Agent ID (e.g., GL-001)"),
    user: str = typer.Argument(..., help="User email or identifier"),
    role: str = typer.Argument(..., help="Role to revoke"),
):
    """
    Revoke role from user for agent.

    Example:
        greenlang rbac revoke GL-001 user@example.com agent_operator
    """
    try:
        enforcer = get_enforcer()

        # Revoke role
        enforcer.revoke_agent_role(agent_id, user, role)

        console.print(f"[green]✓[/green] Revoked role [bold]{role}[/bold] from [bold]{user}[/bold] for agent [bold]{agent_id}[/bold]")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def list(
    agent_id: str = typer.Argument(..., help="Agent ID (e.g., GL-001)"),
    format: str = typer.Option("table", help="Output format: table, json"),
):
    """
    List all RBAC grants for agent.

    Example:
        greenlang rbac list GL-001
        greenlang rbac list GL-001 --format json
    """
    try:
        enforcer = get_enforcer()

        # Get ACL for agent
        acl = enforcer.rbac_manager.get_acl(agent_id)

        if not acl:
            console.print(f"[yellow]No RBAC grants found for agent {agent_id}[/yellow]")
            console.print("\n[dim]Tip: Use 'greenlang rbac grant' to add grants[/dim]")
            return

        # Output format
        if format == "json":
            output = {
                "agent_id": agent_id,
                "user_roles": acl.user_roles,
                "custom_roles": {name: role.to_dict() for name, role in acl.custom_roles.items()}
            }
            console.print(json.dumps(output, indent=2))
            return

        # Table format
        table = Table(title=f"RBAC Grants for Agent {agent_id}")
        table.add_column("User", style="cyan")
        table.add_column("Roles", style="green")
        table.add_column("Permissions", style="yellow")

        for user, roles in acl.user_roles.items():
            # Collect all permissions from all roles
            all_permissions = set()
            for role_name in roles:
                role = PREDEFINED_ROLES.get(role_name) or acl.custom_roles.get(role_name)
                if role:
                    all_permissions.update(role.permissions)

            roles_str = ", ".join(roles)
            permissions_str = ", ".join([p.value for p in sorted(all_permissions, key=lambda x: x.value)])

            table.add_row(user, roles_str, permissions_str)

        console.print(table)

        # Show custom roles if any
        if acl.custom_roles:
            console.print(f"\n[bold]Custom Roles:[/bold]")
            for role_name, role in acl.custom_roles.items():
                console.print(f"  • {role_name}: {role.description}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def audit(
    user: str = typer.Argument(..., help="User email or identifier"),
    format: str = typer.Option("table", help="Output format: table, json"),
):
    """
    Audit all agent permissions for user.

    Example:
        greenlang rbac audit user@example.com
        greenlang rbac audit user@example.com --format json
    """
    try:
        enforcer = get_enforcer()

        # Get audit data
        audit_data = enforcer.audit_user_agent_access(user)

        if not audit_data:
            console.print(f"[yellow]No agent access found for user {user}[/yellow]")
            return

        # Output format
        if format == "json":
            output = {
                "user": user,
                "agent_access": audit_data
            }
            console.print(json.dumps(output, indent=2))
            return

        # Table format
        table = Table(title=f"Agent Access Audit for {user}")
        table.add_column("Agent ID", style="cyan")
        table.add_column("Roles", style="green")
        table.add_column("Permissions", style="yellow")

        for agent_id, roles in audit_data.items():
            acl = enforcer.rbac_manager.get_acl(agent_id)

            # Collect all permissions
            all_permissions = set()
            for role_name in roles:
                role = PREDEFINED_ROLES.get(role_name)
                if role and acl:
                    role_obj = acl.custom_roles.get(role_name) if role_name not in PREDEFINED_ROLES else role
                    if role_obj:
                        all_permissions.update(role_obj.permissions)

            roles_str = ", ".join(roles)
            permissions_str = ", ".join([p.value for p in sorted(all_permissions, key=lambda x: x.value)])

            table.add_row(agent_id, roles_str, permissions_str)

        console.print(table)

        # Show summary
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Total agents accessible: {len(audit_data)}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def roles(
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed permissions"),
):
    """
    List all available predefined roles.

    Example:
        greenlang rbac roles
        greenlang rbac roles --detailed
    """
    try:
        console.print("[bold]Available Predefined Roles:[/bold]\n")

        for role_name, role in PREDEFINED_ROLES.items():
            console.print(f"[cyan]{role_name}[/cyan]")
            console.print(f"  {role.description}")

            if detailed:
                permissions = ", ".join([p.value for p in role.permissions])
                console.print(f"  [dim]Permissions:[/dim] {permissions}")

            console.print()

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def check(
    agent_id: str = typer.Argument(..., help="Agent ID (e.g., GL-001)"),
    user: str = typer.Argument(..., help="User email or identifier"),
    permission: str = typer.Argument(..., help="Permission to check (e.g., execute, read_data)"),
):
    """
    Check if user has specific permission for agent.

    Example:
        greenlang rbac check GL-001 user@example.com execute
        greenlang rbac check GL-001 user@example.com read_data
    """
    try:
        enforcer = get_enforcer()

        # Parse permission
        try:
            perm = AgentPermission.from_string(permission)
        except ValueError:
            console.print(f"[red]Error:[/red] Invalid permission: {permission}")
            console.print("\n[yellow]Valid permissions:[/yellow]")
            for p in AgentPermission:
                console.print(f"  • {p.value}")
            raise typer.Exit(1)

        # Check permission
        has_permission = enforcer.rbac_manager.check_permission(agent_id, user, perm)

        if has_permission:
            console.print(f"[green]✓ GRANTED[/green]")
            console.print(f"User [bold]{user}[/bold] has permission [bold]{permission}[/bold] for agent [bold]{agent_id}[/bold]")

            # Show which roles grant this permission
            acl = enforcer.rbac_manager.get_acl(agent_id)
            if acl:
                user_roles = acl.list_user_roles(user)
                granting_roles = []
                for role_name in user_roles:
                    role = PREDEFINED_ROLES.get(role_name) or acl.custom_roles.get(role_name)
                    if role and role.has_permission(perm):
                        granting_roles.append(role_name)

                if granting_roles:
                    console.print(f"\n[dim]Granted by roles:[/dim] {', '.join(granting_roles)}")
        else:
            console.print(f"[red]✗ DENIED[/red]")
            console.print(f"User [bold]{user}[/bold] does NOT have permission [bold]{permission}[/bold] for agent [bold]{agent_id}[/bold]")

            # Show what roles the user has (if any)
            acl = enforcer.rbac_manager.get_acl(agent_id)
            if acl:
                user_roles = acl.list_user_roles(user)
                if user_roles:
                    console.print(f"\n[dim]User roles:[/dim] {', '.join(user_roles)}")
                else:
                    console.print(f"\n[dim]User has no roles for this agent[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def export(
    output: Path = typer.Option("rbac_audit.json", help="Output file path"),
):
    """
    Export complete RBAC audit log.

    Example:
        greenlang rbac export
        greenlang rbac export --output my_audit.json
    """
    try:
        enforcer = get_enforcer()

        # Export audit log
        enforcer.rbac_manager.export_audit_log(output)

        console.print(f"[green]✓[/green] Exported RBAC audit log to [bold]{output}[/bold]")

        # Show summary
        acl_count = len(enforcer.rbac_manager.access_controls)
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Total agents with ACLs: {acl_count}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def create_acl(
    agent_id: str = typer.Argument(..., help="Agent ID (e.g., GL-001)"),
):
    """
    Create new ACL for agent.

    Example:
        greenlang rbac create-acl GL-001
    """
    try:
        enforcer = get_enforcer()

        # Create ACL
        acl = enforcer.rbac_manager.create_acl(agent_id)

        console.print(f"[green]✓[/green] Created ACL for agent [bold]{agent_id}[/bold]")
        console.print(f"\n[dim]Tip: Use 'greenlang rbac grant' to add user permissions[/dim]")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def delete_acl(
    agent_id: str = typer.Argument(..., help="Agent ID (e.g., GL-001)"),
    confirm: bool = typer.Option(False, "--confirm", help="Skip confirmation prompt"),
):
    """
    Delete ACL for agent (removes all grants).

    Example:
        greenlang rbac delete-acl GL-001 --confirm
    """
    try:
        enforcer = get_enforcer()

        # Check if ACL exists
        acl = enforcer.rbac_manager.get_acl(agent_id)
        if not acl:
            console.print(f"[yellow]No ACL found for agent {agent_id}[/yellow]")
            return

        # Confirm deletion
        if not confirm:
            user_count = len(acl.user_roles)
            console.print(f"[yellow]Warning:[/yellow] This will delete the ACL for agent {agent_id}")
            console.print(f"  Affected users: {user_count}")
            response = typer.confirm("Are you sure?")
            if not response:
                console.print("[yellow]Cancelled[/yellow]")
                return

        # Delete ACL
        enforcer.rbac_manager.delete_acl(agent_id)

        console.print(f"[green]✓[/green] Deleted ACL for agent [bold]{agent_id}[/bold]")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
