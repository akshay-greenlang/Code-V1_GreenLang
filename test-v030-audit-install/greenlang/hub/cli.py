"""
CLI Commands for GreenLang Hub Operations
"""

import click
import logging
from pathlib import Path
import json
import sys

from .client import HubClient
from .auth import HubAuth, PackSigner
from .manifest import create_manifest, load_manifest
from .index import PackIndex, SearchFilters, SortOrder, PackCategory

logger = logging.getLogger(__name__)


@click.group(name="hub")
def hub_cli():
    """GreenLang Hub registry commands"""


@hub_cli.command()
@click.option("--username", "-u", prompt=True, help="Hub username")
@click.option("--password", "-p", prompt=True, hide_input=True, help="Password")
@click.option("--registry", default="https://hub.greenlang.io", help="Registry URL")
def login(username: str, password: str, registry: str):
    """Login to GreenLang Hub"""
    try:
        auth = HubAuth()
        if auth.login(username, password, registry):
            click.echo(f"✓ Successfully logged in as {username}")
        else:
            click.echo("✗ Login failed", err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@hub_cli.command()
def logout():
    """Logout from GreenLang Hub"""
    try:
        auth = HubAuth()
        auth.logout()
        click.echo("✓ Successfully logged out")
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@hub_cli.command()
@click.argument("pack_path", type=click.Path(exists=True))
@click.option("--tags", "-t", multiple=True, help="Pack tags")
@click.option("--description", "-d", help="Pack description")
@click.option("--sign", is_flag=True, help="Sign the pack")
@click.option("--key", type=click.Path(exists=True), help="Private key for signing")
@click.option("--registry", default="https://hub.greenlang.io", help="Registry URL")
def push(
    pack_path: str, tags: tuple, description: str, sign: bool, key: str, registry: str
):
    """Push pack to registry"""
    try:
        pack_path = Path(pack_path)

        # Load authentication
        auth = HubAuth()
        if not auth.token and not auth.api_key:
            click.echo("✗ Not authenticated. Please login first.", err=True)
            sys.exit(1)

        # Sign pack if requested
        signature = None
        if sign:
            if not key:
                key = Path.home() / ".greenlang" / "keys" / "private.pem"

            signer = PackSigner(Path(key))
            with open(pack_path, "rb") as f:
                signature = signer.sign_pack(f.read())
            click.echo(f"✓ Pack signed")

        # Push to registry
        with HubClient(registry, auth) as client:
            result = client.push(
                pack_path,
                signature=signature,
                tags=list(tags) if tags else None,
                description=description,
            )

            pack_id = result.get("id", "unknown")
            pack_url = result.get("url", f"{registry}/packs/{pack_id}")

            click.echo(f"✓ Successfully pushed pack")
            click.echo(f"  ID: {pack_id}")
            click.echo(f"  URL: {pack_url}")

    except Exception as e:
        click.echo(f"✗ Push failed: {e}", err=True)
        sys.exit(1)


@hub_cli.command()
@click.argument("pack_ref")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--verify", is_flag=True, default=True, help="Verify signature")
@click.option("--registry", default="https://hub.greenlang.io", help="Registry URL")
def pull(pack_ref: str, output: str, verify: bool, registry: str):
    """Pull pack from registry"""
    try:
        # Load authentication (optional for public packs)
        auth = HubAuth()

        # Pull from registry
        with HubClient(registry, auth) as client:
            output_dir = Path(output) if output else None
            pack_dir = client.pull(pack_ref, output_dir, verify_signature=verify)

            # Load manifest to show info
            manifest = load_manifest(pack_dir)

            click.echo(f"✓ Successfully pulled {manifest.name} v{manifest.version}")
            click.echo(f"  Location: {pack_dir}")

    except Exception as e:
        click.echo(f"✗ Pull failed: {e}", err=True)
        sys.exit(1)


@hub_cli.command()
@click.option("--query", "-q", help="Search query")
@click.option("--tag", "-t", multiple=True, help="Filter by tag")
@click.option("--author", "-a", help="Filter by author")
@click.option("--limit", "-l", default=20, help="Maximum results")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--registry", default="https://hub.greenlang.io", help="Registry URL")
def search(
    query: str, tag: tuple, author: str, limit: int, output_json: bool, registry: str
):
    """Search for packs in registry"""
    try:
        # Search doesn't require authentication
        with HubClient(registry) as client:
            results = client.search(
                query=query, tags=list(tag) if tag else None, author=author, limit=limit
            )

            if output_json:
                click.echo(json.dumps(results, indent=2))
            else:
                if not results:
                    click.echo("No packs found")
                else:
                    click.echo(f"Found {len(results)} pack(s):\n")
                    for pack in results:
                        name = pack.get("name", "unknown")
                        version = pack.get("version", "")
                        desc = pack.get("description", "")
                        author = pack.get("author", {}).get("name", "unknown")

                        click.echo(f"  {name}@{version}")
                        click.echo(f"    {desc}")
                        click.echo(f"    by {author}")
                        click.echo()

    except Exception as e:
        click.echo(f"✗ Search failed: {e}", err=True)
        sys.exit(1)


@hub_cli.command()
@click.option("--user", "-u", help="Filter by user")
@click.option("--limit", "-l", default=50, help="Maximum results")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--registry", default="https://hub.greenlang.io", help="Registry URL")
def list(user: str, limit: int, output_json: bool, registry: str):
    """List packs from registry"""
    try:
        # List doesn't require authentication
        with HubClient(registry) as client:
            packs = client.list_packs(user=user, limit=limit)

            if output_json:
                click.echo(json.dumps(packs, indent=2))
            else:
                if not packs:
                    click.echo("No packs found")
                else:
                    click.echo(f"Listing {len(packs)} pack(s):\n")
                    for pack in packs:
                        name = pack.get("name", "unknown")
                        version = pack.get("version", "")
                        downloads = pack.get("downloads", 0)
                        stars = pack.get("stars", 0)

                        click.echo(f"  {name}@{version}")
                        click.echo(f"    Downloads: {downloads} | Stars: {stars}")

    except Exception as e:
        click.echo(f"✗ List failed: {e}", err=True)
        sys.exit(1)


@hub_cli.command()
@click.argument("pack_ref")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--registry", default="https://hub.greenlang.io", help="Registry URL")
def info(pack_ref: str, output_json: bool, registry: str):
    """Get detailed pack information"""
    try:
        # Info doesn't require authentication
        with HubClient(registry) as client:
            info = client.get_pack_info(pack_ref)

            if output_json:
                click.echo(json.dumps(info, indent=2))
            else:
                manifest = info.get("manifest", {})
                stats = info.get("stats", {})

                click.echo(
                    f"\n{manifest.get('name', 'unknown')} v{manifest.get('version', '')}"
                )
                click.echo("=" * 50)
                click.echo(f"Description: {manifest.get('description', '')}")

                author = manifest.get("author", {})
                if author:
                    click.echo(f"Author: {author.get('name', 'unknown')}")

                click.echo(f"License: {manifest.get('license', 'Not specified')}")
                click.echo(f"Homepage: {manifest.get('homepage', 'Not specified')}")

                click.echo(f"\nStatistics:")
                click.echo(f"  Downloads: {stats.get('downloads', 0)}")
                click.echo(f"  Stars: {stats.get('stars', 0)}")
                click.echo(f"  Published: {info.get('published_at', 'unknown')}")

                deps = manifest.get("dependencies", [])
                if deps:
                    click.echo(f"\nDependencies:")
                    for dep in deps:
                        click.echo(f"  - {dep.get('name')}@{dep.get('version')}")

    except Exception as e:
        click.echo(f"✗ Info failed: {e}", err=True)
        sys.exit(1)


@hub_cli.command()
@click.argument("pack_ref")
@click.option("--confirm", is_flag=True, help="Skip confirmation")
@click.option("--registry", default="https://hub.greenlang.io", help="Registry URL")
def delete(pack_ref: str, confirm: bool, registry: str):
    """Delete pack from registry"""
    try:
        # Load authentication
        auth = HubAuth()
        if not auth.token and not auth.api_key:
            click.echo("✗ Not authenticated. Please login first.", err=True)
            sys.exit(1)

        # Confirm deletion
        if not confirm:
            if not click.confirm(f"Are you sure you want to delete {pack_ref}?"):
                click.echo("Cancelled")
                return

        # Delete from registry
        with HubClient(registry, auth) as client:
            if client.delete_pack(pack_ref):
                click.echo(f"✓ Successfully deleted {pack_ref}")
            else:
                click.echo(f"✗ Failed to delete {pack_ref}", err=True)
                sys.exit(1)

    except Exception as e:
        click.echo(f"✗ Delete failed: {e}", err=True)
        sys.exit(1)


@hub_cli.command()
@click.argument("pack_dir", type=click.Path(exists=True))
@click.option("--name", "-n", help="Pack name")
@click.option("--version", "-v", default="0.1.0", help="Pack version")
@click.option("--description", "-d", help="Pack description")
@click.option("--author", "-a", help="Author name")
@click.option("--license", "-l", help="License")
@click.option("--output", "-o", help="Output format (json/yaml)", default="json")
def init(
    pack_dir: str,
    name: str,
    version: str,
    description: str,
    author: str,
    license: str,
    output: str,
):
    """Initialize pack manifest"""
    try:
        pack_path = Path(pack_dir)

        # Check if manifest already exists
        manifest_files = ["manifest.json", "manifest.yaml", "greenlang.json"]
        for mf in manifest_files:
            if (pack_path / mf).exists():
                if not click.confirm(f"{mf} already exists. Overwrite?"):
                    click.echo("Cancelled")
                    return

        # Create manifest
        manifest = create_manifest(
            pack_path,
            name=name,
            version=version,
            description=description,
            author=author,
            license=license,
        )

        # Save manifest
        from .manifest import save_manifest

        manifest_file = save_manifest(pack_path, manifest, format=output)

        click.echo(f"✓ Created manifest at {manifest_file}")
        click.echo(f"  Name: {manifest.name}")
        click.echo(f"  Version: {manifest.version}")
        click.echo(f"  Modules: {len(manifest.modules)}")
        click.echo(f"  Resources: {len(manifest.resources)}")

    except Exception as e:
        click.echo(f"✗ Init failed: {e}", err=True)
        sys.exit(1)


@hub_cli.group()
def key():
    """Manage signing keys"""


@key.command("generate")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--size", "-s", default=2048, help="Key size in bits")
def generate_key(output: str, size: int):
    """Generate new signing key pair"""
    try:
        if output:
            output_dir = Path(output)
        else:
            output_dir = Path.home() / ".greenlang" / "keys"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate keys
        signer = PackSigner()
        signer.generate_keys(size)

        # Save keys
        private_path = output_dir / "private.pem"
        public_path = output_dir / "public.pem"

        signer.save_keys(private_path, public_path)

        click.echo(f"✓ Generated {size}-bit RSA key pair")
        click.echo(f"  Private key: {private_path}")
        click.echo(f"  Public key: {public_path}")
        click.echo("\n⚠ Keep your private key secure and never share it!")

    except Exception as e:
        click.echo(f"✗ Key generation failed: {e}", err=True)
        sys.exit(1)


@key.command("verify")
@click.argument("pack_path", type=click.Path(exists=True))
@click.argument("signature_file", type=click.Path(exists=True))
@click.option("--key", "-k", type=click.Path(exists=True), help="Public key file")
def verify_signature(pack_path: str, signature_file: str, key: str):
    """Verify pack signature"""
    try:
        # Load signature
        with open(signature_file, "r") as f:
            signature = json.load(f)

        # Load pack data
        with open(pack_path, "rb") as f:
            pack_data = f.read()

        # Load public key if provided
        public_key_pem = None
        if key:
            with open(key, "rb") as f:
                public_key_pem = f.read()

        # Verify
        signer = PackSigner()
        if signer.verify_signature(pack_data, signature, public_key_pem):
            click.echo("✓ Signature verification successful")
            click.echo(f"  Algorithm: {signature.get('algorithm')}")
            click.echo(f"  Timestamp: {signature.get('timestamp')}")
        else:
            click.echo("✗ Signature verification failed", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"✗ Verification failed: {e}", err=True)
        sys.exit(1)


@hub_cli.group(name="discover")
def discover():
    """Pack discovery commands"""


@discover.command("featured")
@click.option("--limit", "-l", default=10, help="Number of packs to show")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def discover_featured(limit: int, output_json: bool):
    """Show featured packs"""
    try:
        index = PackIndex()
        packs = index.get_featured(limit=limit)

        if output_json:
            click.echo(json.dumps([p.to_dict() for p in packs], indent=2))
        else:
            if not packs:
                click.echo("No featured packs available")
            else:
                click.echo("Featured Packs:")
                click.echo("=" * 60)
                for pack in packs:
                    _display_pack_info(pack)

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@discover.command("trending")
@click.option(
    "--period",
    "-p",
    type=click.Choice(["day", "week", "month"]),
    default="week",
    help="Time period",
)
@click.option("--limit", "-l", default=10, help="Number of packs to show")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def discover_trending(period: str, limit: int, output_json: bool):
    """Show trending packs"""
    try:
        index = PackIndex()
        packs = index.get_trending(period=period, limit=limit)

        if output_json:
            click.echo(json.dumps([p.to_dict() for p in packs], indent=2))
        else:
            if not packs:
                click.echo(f"No trending packs for {period}")
            else:
                click.echo(f"Trending Packs ({period}):")
                click.echo("=" * 60)
                for pack in packs:
                    _display_pack_info(pack)

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@discover.command("category")
@click.argument("category", type=click.Choice([c.value for c in PackCategory]))
@click.option("--limit", "-l", default=20, help="Number of packs to show")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def discover_category(category: str, limit: int, output_json: bool):
    """Browse packs by category"""
    try:
        index = PackIndex()
        cat_enum = PackCategory(category)
        packs = index.get_by_category(cat_enum, limit=limit)

        if output_json:
            click.echo(json.dumps([p.to_dict() for p in packs], indent=2))
        else:
            if not packs:
                click.echo(f"No packs in category: {category}")
            else:
                click.echo(f"Category: {category}")
                click.echo("=" * 60)
                for pack in packs:
                    _display_pack_info(pack)

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@discover.command("search")
@click.argument("query", required=False)
@click.option("--category", "-c", multiple=True, help="Filter by category")
@click.option("--tag", "-t", multiple=True, help="Filter by tag")
@click.option("--author", "-a", help="Filter by author")
@click.option("--license", "-l", help="Filter by license")
@click.option("--min-stars", type=int, help="Minimum stars")
@click.option("--min-downloads", type=int, help="Minimum downloads")
@click.option("--verified", is_flag=True, help="Verified packs only")
@click.option("--official", is_flag=True, help="Official packs only")
@click.option(
    "--sort",
    type=click.Choice(
        ["relevance", "downloads", "stars", "updated", "created", "name"]
    ),
    default="relevance",
    help="Sort order",
)
@click.option("--limit", default=20, help="Maximum results")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def discover_search(
    query: str,
    category: tuple,
    tag: tuple,
    author: str,
    license: str,
    min_stars: int,
    min_downloads: int,
    verified: bool,
    official: bool,
    sort: str,
    limit: int,
    output_json: bool,
):
    """Advanced pack search"""
    try:
        # Build filters
        filters = SearchFilters(
            categories=list(category) if category else None,
            tags=list(tag) if tag else None,
            author=author,
            license=license,
            min_stars=min_stars,
            min_downloads=min_downloads,
            verified_only=verified,
            official_only=official,
        )

        # Search
        index = PackIndex()
        sort_order = SortOrder(sort)
        packs = index.search(query=query, filters=filters, sort=sort_order, limit=limit)

        if output_json:
            click.echo(json.dumps([p.to_dict() for p in packs], indent=2))
        else:
            if not packs:
                click.echo("No packs found matching criteria")
            else:
                click.echo(f"Search Results ({len(packs)} packs):")
                click.echo("=" * 60)
                for pack in packs:
                    _display_pack_info(pack)

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@discover.command("similar")
@click.argument("pack_id")
@click.option("--limit", "-l", default=10, help="Number of similar packs")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def discover_similar(pack_id: str, limit: int, output_json: bool):
    """Find similar packs"""
    try:
        index = PackIndex()
        packs = index.get_similar(pack_id, limit=limit)

        if output_json:
            click.echo(json.dumps([p.to_dict() for p in packs], indent=2))
        else:
            if not packs:
                click.echo(f"No similar packs found for {pack_id}")
            else:
                click.echo(f"Packs similar to {pack_id}:")
                click.echo("=" * 60)
                for pack in packs:
                    _display_pack_info(pack)

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@discover.command("recommend")
@click.option("--installed", "-i", multiple=True, help="Installed pack IDs")
@click.option("--limit", "-l", default=10, help="Number of recommendations")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def discover_recommend(installed: tuple, limit: int, output_json: bool):
    """Get personalized pack recommendations"""
    try:
        index = PackIndex()
        packs = index.get_recommendations(
            user_packs=list(installed) if installed else None, limit=limit
        )

        if output_json:
            click.echo(json.dumps([p.to_dict() for p in packs], indent=2))
        else:
            if not packs:
                click.echo("No recommendations available")
            else:
                click.echo("Recommended for you:")
                click.echo("=" * 60)
                for pack in packs:
                    _display_pack_info(pack)

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@discover.command("stats")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def discover_stats(output_json: bool):
    """Show registry statistics"""
    try:
        index = PackIndex()
        stats = index.get_statistics()

        if output_json:
            click.echo(json.dumps(stats, indent=2))
        else:
            click.echo("Registry Statistics:")
            click.echo("=" * 60)
            for key, value in stats.items():
                click.echo(f"  {key.replace('_', ' ').title()}: {value:,}")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@discover.command("categories")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def discover_categories(output_json: bool):
    """List all categories"""
    try:
        index = PackIndex()
        categories = index.get_categories()

        if output_json:
            click.echo(json.dumps(categories, indent=2))
        else:
            click.echo("Available Categories:")
            click.echo("=" * 60)
            for cat in categories:
                name = cat.get("display_name", cat.get("name"))
                count = cat.get("count", 0)
                click.echo(f"  {name}: {count} packs")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@discover.command("tags")
@click.option("--limit", "-l", default=50, help="Number of tags to show")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def discover_tags(limit: int, output_json: bool):
    """Show popular tags"""
    try:
        index = PackIndex()
        tags = index.get_tags(limit=limit)

        if output_json:
            click.echo(json.dumps(tags, indent=2))
        else:
            click.echo(f"Popular Tags (Top {limit}):")
            click.echo("=" * 60)
            for tag in tags:
                name = tag.get("name", "unknown")
                count = tag.get("count", 0)
                click.echo(f"  #{name}: {count} packs")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@discover.command("update-index")
def discover_update_index():
    """Update local pack index"""
    try:
        click.echo("Updating local pack index...")
        index = PackIndex()
        index.update_local_index()
        click.echo("✓ Local index updated successfully")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


def _display_pack_info(pack):
    """Display pack information in a formatted way"""
    click.echo(f"\n  {pack.name}@{pack.version}")

    # Badges
    badges = []
    if pack.verified:
        badges.append("✓ Verified")
    if pack.official:
        badges.append("Official")
    if pack.featured:
        badges.append("Featured")
    if pack.trending:
        badges.append("Trending")
    if pack.deprecated:
        badges.append("⚠ Deprecated")

    if badges:
        click.echo(f"  [{' | '.join(badges)}]")

    click.echo(f"  {pack.description}")
    click.echo(f"  by {pack.author.get('name', 'unknown')}")

    # Stats
    stats = []
    if pack.downloads > 0:
        stats.append(f"↓ {pack.downloads:,}")
    if pack.stars > 0:
        stats.append(f"★ {pack.stars:,}")

    if stats:
        click.echo(f"  {' | '.join(stats)}")

    # Categories and tags
    if pack.categories:
        click.echo(f"  Categories: {', '.join(pack.categories)}")
    if pack.tags:
        click.echo(f"  Tags: {', '.join(['#' + t for t in pack.tags[:5]])}")

    click.echo()


def register_hub_commands(cli):
    """Register hub commands with main CLI"""
    cli.add_command(hub_cli)
