# -*- coding: utf-8 -*-
"""
GreenLang Tenant Management CLI Commands
"""

import click
import json
import yaml
from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from greenlang.auth import (
from greenlang.determinism import DeterministicClock
    TenantManager,
    TenantQuota,
    TenantIsolation,
    RBACManager,
    AuthManager,
)
from greenlang.auth.audit import get_audit_logger, AuditEventType

console = Console()


@click.group()
def tenant():
    """Manage GreenLang tenants"""


@tenant.command()
@click.option("--name", required=True, help="Tenant name")
@click.option("--domain", help="Tenant domain")
@click.option("--admin-email", required=True, help="Admin email")
@click.option("--admin-name", help="Admin name")
@click.option(
    "--isolation",
    type=click.Choice(["shared", "namespace", "cluster", "physical"]),
    default="namespace",
    help="Isolation level",
)
@click.option("--max-users", type=int, default=100, help="Maximum users")
@click.option("--max-pipelines", type=int, default=1000, help="Maximum pipelines")
@click.option("--max-storage-gb", type=int, default=100, help="Maximum storage in GB")
@click.option(
    "--max-compute-hours", type=int, default=1000, help="Maximum compute hours"
)
@click.option(
    "--output", "-o", type=click.Choice(["json", "yaml"]), help="Output format"
)
def create(
    name: str,
    domain: Optional[str],
    admin_email: str,
    admin_name: Optional[str],
    isolation: str,
    max_users: int,
    max_pipelines: int,
    max_storage_gb: int,
    max_compute_hours: int,
    output: Optional[str],
):
    """Create a new tenant"""
    try:
        manager = TenantManager()

        # Create quota
        quota = TenantQuota(
            max_users=max_users,
            max_pipelines=max_pipelines,
            max_storage_gb=max_storage_gb,
            max_compute_hours=max_compute_hours,
        )

        # Create tenant
        tenant = manager.create_tenant(
            name=name,
            domain=domain,
            admin_email=admin_email,
            admin_name=admin_name or admin_email.split("@")[0],
            quota=quota,
            isolation=TenantIsolation[isolation.upper()],
        )

        if output == "json":
            click.echo(json.dumps(tenant.to_dict(), indent=2, default=str))
        elif output == "yaml":
            click.echo(yaml.dump(tenant.to_dict(), default_flow_style=False))
        else:
            console.print(
                Panel(
                    f"[green]Tenant created successfully![/green]\n\n"
                    f"ID: {tenant.tenant_id}\n"
                    f"Name: {tenant.name}\n"
                    f"Domain: {tenant.domain or 'N/A'}\n"
                    f"Admin: {tenant.admin_email}\n"
                    f"Isolation: {tenant.isolation.value}\n"
                    f"Status: {tenant.status}",
                    title="Tenant Created",
                )
            )

        # Log audit event
        get_audit_logger().log_resource_access(
            user_id="cli",
            resource_type="tenant",
            resource_id=tenant.tenant_id,
            action="create",
            tenant_id=tenant.tenant_id,
        )

    except Exception as e:
        console.print(f"[red]Error creating tenant: {e}[/red]")
        raise click.ClickException(str(e))


@tenant.command()
@click.argument("tenant_id")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
    help="Output format",
)
def get(tenant_id: str, output: str):
    """Get tenant details"""
    try:
        manager = TenantManager()
        tenant = manager.get_tenant(tenant_id)

        if not tenant:
            console.print(f"[red]Tenant not found: {tenant_id}[/red]")
            return

        if output == "json":
            click.echo(json.dumps(tenant.to_dict(), indent=2, default=str))
        elif output == "yaml":
            click.echo(yaml.dump(tenant.to_dict(), default_flow_style=False))
        else:
            # Create table
            table = Table(title=f"Tenant: {tenant.name}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("ID", tenant.tenant_id)
            table.add_row("Name", tenant.name)
            table.add_row("Domain", tenant.domain or "N/A")
            table.add_row("Admin Email", tenant.admin_email)
            table.add_row("Admin Name", tenant.admin_name)
            table.add_row("Status", tenant.status)
            table.add_row("Isolation", tenant.isolation.value)
            table.add_row("Created", tenant.created_at.strftime("%Y-%m-%d %H:%M:%S"))
            table.add_row("Updated", tenant.updated_at.strftime("%Y-%m-%d %H:%M:%S"))

            # Quota information
            if tenant.quota:
                table.add_section()
                table.add_row("Max Users", str(tenant.quota.max_users))
                table.add_row("Used Users", str(tenant.quota.used_users))
                table.add_row("Max Pipelines", str(tenant.quota.max_pipelines))
                table.add_row("Used Pipelines", str(tenant.quota.used_pipelines))
                table.add_row("Max Storage (GB)", str(tenant.quota.max_storage_gb))
                table.add_row(
                    "Used Storage (GB)", f"{tenant.quota.used_storage_gb:.2f}"
                )
                table.add_row("Max Compute Hours", str(tenant.quota.max_compute_hours))
                table.add_row(
                    "Used Compute Hours", f"{tenant.quota.used_compute_hours:.2f}"
                )

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error getting tenant: {e}[/red]")
        raise click.ClickException(str(e))


@tenant.command()
@click.option(
    "--status",
    type=click.Choice(["active", "suspended", "all"]),
    default="active",
    help="Filter by status",
)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
    help="Output format",
)
def list(status: str, output: str):
    """List all tenants"""
    try:
        manager = TenantManager()
        tenants = manager.list_tenants()

        # Filter by status
        if status != "all":
            tenants = [t for t in tenants if t.status == status]

        if output == "json":
            data = [t.to_dict() for t in tenants]
            click.echo(json.dumps(data, indent=2, default=str))
        elif output == "yaml":
            data = [t.to_dict() for t in tenants]
            click.echo(yaml.dump(data, default_flow_style=False))
        else:
            # Create table
            table = Table(title="Tenants")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="white")
            table.add_column("Domain", style="white")
            table.add_column("Admin", style="white")
            table.add_column("Status", style="green")
            table.add_column("Isolation", style="yellow")
            table.add_column("Users", style="white")
            table.add_column("Created", style="white")

            for t in tenants:
                table.add_row(
                    t.tenant_id[:8] + "...",
                    t.name,
                    t.domain or "N/A",
                    t.admin_email,
                    t.status,
                    t.isolation.value,
                    f"{t.quota.used_users}/{t.quota.max_users}" if t.quota else "N/A",
                    t.created_at.strftime("%Y-%m-%d"),
                )

            console.print(table)
            console.print(f"\nTotal: {len(tenants)} tenants")

    except Exception as e:
        console.print(f"[red]Error listing tenants: {e}[/red]")
        raise click.ClickException(str(e))


@tenant.command()
@click.argument("tenant_id")
@click.option("--name", help="New tenant name")
@click.option("--domain", help="New domain")
@click.option("--admin-email", help="New admin email")
@click.option("--status", type=click.Choice(["active", "suspended"]), help="New status")
@click.option("--max-users", type=int, help="New max users")
@click.option("--max-pipelines", type=int, help="New max pipelines")
@click.option("--max-storage-gb", type=int, help="New max storage")
@click.option("--max-compute-hours", type=int, help="New max compute hours")
def update(tenant_id: str, **kwargs):
    """Update tenant configuration"""
    try:
        manager = TenantManager()

        # Build updates dict
        updates = {}
        quota_updates = {}

        for key, value in kwargs.items():
            if value is not None:
                if key.startswith("max_"):
                    quota_updates[key] = value
                else:
                    updates[key] = value

        if quota_updates:
            updates["quota"] = quota_updates

        if not updates:
            console.print("[yellow]No updates specified[/yellow]")
            return

        # Update tenant
        tenant = manager.update_tenant(tenant_id, updates)

        if tenant:
            console.print(
                Panel(
                    f"[green]Tenant updated successfully![/green]\n\n"
                    f"ID: {tenant.tenant_id}\n"
                    f"Name: {tenant.name}\n"
                    f"Status: {tenant.status}",
                    title="Tenant Updated",
                )
            )
        else:
            console.print("[red]Failed to update tenant[/red]")

    except Exception as e:
        console.print(f"[red]Error updating tenant: {e}[/red]")
        raise click.ClickException(str(e))


@tenant.command()
@click.argument("tenant_id")
@click.option("--force", is_flag=True, help="Force deletion without confirmation")
def delete(tenant_id: str, force: bool):
    """Delete a tenant"""
    try:
        manager = TenantManager()

        # Get tenant first
        tenant = manager.get_tenant(tenant_id)
        if not tenant:
            console.print(f"[red]Tenant not found: {tenant_id}[/red]")
            return

        # Confirm deletion
        if not force:
            if not click.confirm(
                f"Are you sure you want to delete tenant '{tenant.name}'?"
            ):
                console.print("[yellow]Deletion cancelled[/yellow]")
                return

        # Delete tenant
        if manager.delete_tenant(tenant_id):
            console.print(f"[green]Tenant '{tenant.name}' deleted successfully[/green]")

            # Log audit event
            get_audit_logger().log_resource_access(
                user_id="cli",
                resource_type="tenant",
                resource_id=tenant_id,
                action="delete",
            )
        else:
            console.print("[red]Failed to delete tenant[/red]")

    except Exception as e:
        console.print(f"[red]Error deleting tenant: {e}[/red]")
        raise click.ClickException(str(e))


@tenant.command()
@click.argument("tenant_id")
def suspend(tenant_id: str):
    """Suspend a tenant"""
    try:
        manager = TenantManager()

        if manager.suspend_tenant(tenant_id):
            console.print(f"[yellow]Tenant suspended: {tenant_id}[/yellow]")

            # Log audit event
            get_audit_logger().log_resource_access(
                user_id="cli",
                resource_type="tenant",
                resource_id=tenant_id,
                action="suspend",
                tenant_id=tenant_id,
            )
        else:
            console.print("[red]Failed to suspend tenant[/red]")

    except Exception as e:
        console.print(f"[red]Error suspending tenant: {e}[/red]")
        raise click.ClickException(str(e))


@tenant.command()
@click.argument("tenant_id")
def activate(tenant_id: str):
    """Activate a suspended tenant"""
    try:
        manager = TenantManager()

        if manager.activate_tenant(tenant_id):
            console.print(f"[green]Tenant activated: {tenant_id}[/green]")

            # Log audit event
            get_audit_logger().log_resource_access(
                user_id="cli",
                resource_type="tenant",
                resource_id=tenant_id,
                action="activate",
                tenant_id=tenant_id,
            )
        else:
            console.print("[red]Failed to activate tenant[/red]")

    except Exception as e:
        console.print(f"[red]Error activating tenant: {e}[/red]")
        raise click.ClickException(str(e))


@tenant.command()
@click.argument("tenant_id")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
    help="Output format",
)
def quota(tenant_id: str, output: str):
    """Show tenant quota usage"""
    try:
        manager = TenantManager()
        tenant = manager.get_tenant(tenant_id)

        if not tenant or not tenant.quota:
            console.print("[red]Tenant or quota not found[/red]")
            return

        quota_dict = tenant.quota.to_dict()

        if output == "json":
            click.echo(json.dumps(quota_dict, indent=2))
        elif output == "yaml":
            click.echo(yaml.dump(quota_dict, default_flow_style=False))
        else:
            # Create table
            table = Table(title=f"Quota Usage: {tenant.name}")
            table.add_column("Resource", style="cyan")
            table.add_column("Used", style="yellow")
            table.add_column("Limit", style="white")
            table.add_column("Usage %", style="green")

            # Calculate usage percentages
            user_pct = (
                (tenant.quota.used_users / tenant.quota.max_users * 100)
                if tenant.quota.max_users > 0
                else 0
            )
            pipeline_pct = (
                (tenant.quota.used_pipelines / tenant.quota.max_pipelines * 100)
                if tenant.quota.max_pipelines > 0
                else 0
            )
            storage_pct = (
                (tenant.quota.used_storage_gb / tenant.quota.max_storage_gb * 100)
                if tenant.quota.max_storage_gb > 0
                else 0
            )
            compute_pct = (
                (tenant.quota.used_compute_hours / tenant.quota.max_compute_hours * 100)
                if tenant.quota.max_compute_hours > 0
                else 0
            )

            table.add_row(
                "Users",
                str(tenant.quota.used_users),
                str(tenant.quota.max_users),
                f"{user_pct:.1f}%",
            )
            table.add_row(
                "Pipelines",
                str(tenant.quota.used_pipelines),
                str(tenant.quota.max_pipelines),
                f"{pipeline_pct:.1f}%",
            )
            table.add_row(
                "Storage (GB)",
                f"{tenant.quota.used_storage_gb:.2f}",
                str(tenant.quota.max_storage_gb),
                f"{storage_pct:.1f}%",
            )
            table.add_row(
                "Compute Hours",
                f"{tenant.quota.used_compute_hours:.2f}",
                str(tenant.quota.max_compute_hours),
                f"{compute_pct:.1f}%",
            )

            console.print(table)

            # Show warnings if approaching limits
            warnings = []
            if user_pct > 80:
                warnings.append(
                    f"[yellow]Warning: User quota is at {user_pct:.1f}%[/yellow]"
                )
            if storage_pct > 80:
                warnings.append(
                    f"[yellow]Warning: Storage quota is at {storage_pct:.1f}%[/yellow]"
                )
            if compute_pct > 80:
                warnings.append(
                    f"[yellow]Warning: Compute hours quota is at {compute_pct:.1f}%[/yellow]"
                )

            for warning in warnings:
                console.print(warning)

    except Exception as e:
        console.print(f"[red]Error getting quota: {e}[/red]")
        raise click.ClickException(str(e))


# Role management commands
@tenant.group()
def role():
    """Manage tenant roles and permissions"""


@role.command("list")
@click.argument("tenant_id")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
    help="Output format",
)
def list_roles(tenant_id: str, output: str):
    """List available roles"""
    try:
        rbac = RBACManager()
        roles = list(rbac.roles.values())

        if output == "json":
            data = [r.to_dict() for r in roles]
            click.echo(json.dumps(data, indent=2, default=str))
        elif output == "yaml":
            data = [r.to_dict() for r in roles]
            click.echo(yaml.dump(data, default_flow_style=False))
        else:
            table = Table(title="Available Roles")
            table.add_column("Name", style="cyan")
            table.add_column("Description", style="white")
            table.add_column("Permissions", style="yellow")
            table.add_column("Parent Roles", style="green")

            for r in roles:
                table.add_row(
                    r.name,
                    r.description,
                    str(len(r.permissions)),
                    ", ".join(r.parent_roles) if r.parent_roles else "None",
                )

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error listing roles: {e}[/red]")
        raise click.ClickException(str(e))


@role.command("assign")
@click.argument("tenant_id")
@click.argument("user_id")
@click.argument("role_name")
def assign_role(tenant_id: str, user_id: str, role_name: str):
    """Assign role to user"""
    try:
        rbac = RBACManager()

        if rbac.assign_role(user_id, role_name):
            console.print(
                f"[green]Role '{role_name}' assigned to user '{user_id}'[/green]"
            )

            # Log audit event
            get_audit_logger().log(
                AuditEvent(
                    event_type=AuditEventType.ROLE_ASSIGNED,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    action="role_assign",
                    metadata={"role": role_name},
                )
            )
        else:
            console.print("[red]Failed to assign role[/red]")

    except Exception as e:
        console.print(f"[red]Error assigning role: {e}[/red]")
        raise click.ClickException(str(e))


@role.command("revoke")
@click.argument("tenant_id")
@click.argument("user_id")
@click.argument("role_name")
def revoke_role(tenant_id: str, user_id: str, role_name: str):
    """Revoke role from user"""
    try:
        rbac = RBACManager()

        if rbac.revoke_role(user_id, role_name):
            console.print(
                f"[yellow]Role '{role_name}' revoked from user '{user_id}'[/yellow]"
            )

            # Log audit event
            get_audit_logger().log(
                AuditEvent(
                    event_type=AuditEventType.ROLE_REVOKED,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    action="role_revoke",
                    metadata={"role": role_name},
                )
            )
        else:
            console.print("[red]Failed to revoke role[/red]")

    except Exception as e:
        console.print(f"[red]Error revoking role: {e}[/red]")
        raise click.ClickException(str(e))


# API key management commands
@tenant.group()
def apikey():
    """Manage API keys"""


@apikey.command("create")
@click.argument("tenant_id")
@click.argument("name")
@click.option("--user-id", required=True, help="User ID for the API key")
@click.option("--expires-days", type=int, default=365, help="Days until expiration")
@click.option("--scopes", multiple=True, help="API key scopes")
def create_apikey(
    tenant_id: str, name: str, user_id: str, expires_days: int, scopes: tuple
):
    """Create a new API key"""
    try:
        auth = AuthManager()

        # Create API key
        api_key = auth.create_api_key(
            user_id=user_id,
            name=name,
            expires_days=expires_days,
            scopes=list(scopes) if scopes else [],
        )

        if api_key:
            console.print(
                Panel(
                    f"[green]API Key created successfully![/green]\n\n"
                    f"Name: {api_key.name}\n"
                    f"Key: [yellow]{api_key.key}[/yellow]\n"
                    f"User: {api_key.user_id}\n"
                    f"Expires: {api_key.expires_at.strftime('%Y-%m-%d %H:%M:%S') if api_key.expires_at else 'Never'}\n\n"
                    f"[red]Save this key securely! It won't be shown again.[/red]",
                    title="API Key Created",
                )
            )

            # Log audit event
            get_audit_logger().log(
                AuditEvent(
                    event_type=AuditEventType.API_KEY_CREATED,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    action="apikey_create",
                    metadata={"name": name},
                )
            )
        else:
            console.print("[red]Failed to create API key[/red]")

    except Exception as e:
        console.print(f"[red]Error creating API key: {e}[/red]")
        raise click.ClickException(str(e))


@apikey.command("list")
@click.argument("tenant_id")
@click.argument("user_id")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
    help="Output format",
)
def list_apikeys(tenant_id: str, user_id: str, output: str):
    """List user's API keys"""
    try:
        auth = AuthManager()
        keys = auth.list_api_keys(user_id)

        if output == "json":
            data = [k.to_dict() for k in keys]
            click.echo(json.dumps(data, indent=2, default=str))
        elif output == "yaml":
            data = [k.to_dict() for k in keys]
            click.echo(yaml.dump(data, default_flow_style=False))
        else:
            table = Table(title=f"API Keys for User: {user_id}")
            table.add_column("Name", style="cyan")
            table.add_column("Key (prefix)", style="yellow")
            table.add_column("Active", style="green")
            table.add_column("Scopes", style="white")
            table.add_column("Created", style="white")
            table.add_column("Expires", style="white")

            for key in keys:
                table.add_row(
                    key.name,
                    key.key[:12] + "...",
                    "Yes" if key.is_active else "No",
                    ", ".join(key.scopes) if key.scopes else "All",
                    key.created_at.strftime("%Y-%m-%d"),
                    key.expires_at.strftime("%Y-%m-%d") if key.expires_at else "Never",
                )

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error listing API keys: {e}[/red]")
        raise click.ClickException(str(e))


@apikey.command("revoke")
@click.argument("tenant_id")
@click.argument("api_key")
def revoke_apikey(tenant_id: str, api_key: str):
    """Revoke an API key"""
    try:
        auth = AuthManager()

        if auth.revoke_api_key(api_key):
            console.print(f"[yellow]API key revoked: {api_key[:12]}...[/yellow]")

            # Log audit event
            get_audit_logger().log(
                AuditEvent(
                    event_type=AuditEventType.API_KEY_REVOKED,
                    tenant_id=tenant_id,
                    action="apikey_revoke",
                    metadata={"key_prefix": api_key[:12]},
                )
            )
        else:
            console.print("[red]Failed to revoke API key[/red]")

    except Exception as e:
        console.print(f"[red]Error revoking API key: {e}[/red]")
        raise click.ClickException(str(e))


# Audit commands
@tenant.group()
def audit():
    """View audit logs and reports"""


@audit.command("logs")
@click.argument("tenant_id")
@click.option("--user", help="Filter by user ID")
@click.option("--event-type", help="Filter by event type")
@click.option("--days", type=int, default=7, help="Number of days to look back")
@click.option("--limit", type=int, default=100, help="Maximum events to show")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
    help="Output format",
)
def audit_logs(
    tenant_id: str,
    user: Optional[str],
    event_type: Optional[str],
    days: int,
    limit: int,
    output: str,
):
    """View audit logs"""
    try:
        from datetime import timedelta

        audit_logger = get_audit_logger()

        # Query events
        start_time = DeterministicClock.utcnow() - timedelta(days=days)
        events = audit_logger.query(
            tenant_id=tenant_id, start_time=start_time, user_id=user, limit=limit
        )

        # Filter by event type if specified
        if event_type:
            events = [e for e in events if event_type in e.event_type.value]

        if output == "json":
            data = [e.to_dict() for e in events]
            click.echo(json.dumps(data, indent=2, default=str))
        elif output == "yaml":
            data = [e.to_dict() for e in events]
            click.echo(yaml.dump(data, default_flow_style=False))
        else:
            table = Table(title=f"Audit Logs - Last {days} days")
            table.add_column("Time", style="white")
            table.add_column("Event", style="cyan")
            table.add_column("User", style="yellow")
            table.add_column("Resource", style="green")
            table.add_column("Result", style="white")

            for event in events:
                table.add_row(
                    event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    event.event_type.value,
                    event.user_id or "N/A",
                    (
                        f"{event.resource_type}:{event.resource_id}"
                        if event.resource_type
                        else "N/A"
                    ),
                    event.result or "N/A",
                )

            console.print(table)
            console.print(f"\nTotal: {len(events)} events")

    except Exception as e:
        console.print(f"[red]Error getting audit logs: {e}[/red]")
        raise click.ClickException(str(e))


@audit.command("report")
@click.argument("tenant_id")
@click.option("--days", type=int, default=30, help="Report period in days")
@click.option(
    "--output", "-o", type=click.Choice(["json", "yaml"]), help="Output format"
)
def audit_report(tenant_id: str, days: int, output: Optional[str]):
    """Generate audit report"""
    try:
        from datetime import timedelta

        audit_logger = get_audit_logger()

        # Generate report
        start_time = DeterministicClock.utcnow() - timedelta(days=days)
        end_time = DeterministicClock.utcnow()
        report = audit_logger.get_report(tenant_id, start_time, end_time)

        if output == "json":
            click.echo(json.dumps(report, indent=2, default=str))
        elif output == "yaml":
            click.echo(yaml.dump(report, default_flow_style=False))
        else:
            console.print(
                Panel(
                    f"[cyan]Audit Report[/cyan]\n"
                    f"Tenant: {tenant_id}\n"
                    f"Period: {report['period']['start']} to {report['period']['end']}\n\n"
                    f"[yellow]Summary:[/yellow]\n"
                    f"Total Events: {report['total_events']}\n"
                    f"Failed Logins: {report['failed_logins']}\n"
                    f"Permission Denials: {report['permission_denials']}\n"
                    f"Security Alerts: {report['security_alerts']}\n\n"
                    f"[yellow]Top Users:[/yellow]",
                    title="Audit Report",
                )
            )

            # Show top users
            for user, count in report["top_users"][:5]:
                console.print(f"  {user}: {count} events")

    except Exception as e:
        console.print(f"[red]Error generating report: {e}[/red]")
        raise click.ClickException(str(e))
