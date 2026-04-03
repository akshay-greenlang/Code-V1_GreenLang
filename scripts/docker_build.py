#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreenLang Docker Build Script
==============================

Centralized build tool for all GreenLang Docker images using
parameterized Dockerfile templates.

Templates live in deployment/docker/templates/:
  - Dockerfile.agent  -- GL Agent services (FastAPI + uvicorn)
  - Dockerfile.cli    -- GreenLang CLI images (core, full, secure)
  - Dockerfile.api    -- Application API services (CSRD, VCCI, CBAM)

Usage:
  python scripts/docker_build.py agent GL-001              # Build single agent
  python scripts/docker_build.py agent --all               # Build all agents
  python scripts/docker_build.py cli core                  # Build CLI core variant
  python scripts/docker_build.py cli full                  # Build CLI full variant
  python scripts/docker_build.py app csrd                  # Build CSRD app
  python scripts/docker_build.py app --all                 # Build all apps
  python scripts/docker_build.py list                      # List all buildable images
  python scripts/docker_build.py audit                     # Audit stale Dockerfiles

Options:
  --dry-run       Print docker commands without executing
  --no-cache      Pass --no-cache to docker build
  --push          Push images after building
  --registry URL  Override container registry (default: ghcr.io/greenlang)
  --version VER   Override version tag (default: read from pyproject.toml)
  --platform P    Target platform (e.g. linux/amd64,linux/arm64)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root -- everything is relative to this
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Template paths
# ---------------------------------------------------------------------------
TEMPLATE_DIR = PROJECT_ROOT / "deployment" / "docker" / "templates"
TEMPLATE_AGENT = TEMPLATE_DIR / "Dockerfile.agent"
TEMPLATE_CLI = TEMPLATE_DIR / "Dockerfile.cli"
TEMPLATE_API = TEMPLATE_DIR / "Dockerfile.api"

# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------
DEFAULT_REGISTRY = "ghcr.io/greenlang"

# ---------------------------------------------------------------------------
# Agent Registry
# ---------------------------------------------------------------------------
# Every GL Agent that can be built as a container image.
# Keys are the canonical GL-NNN identifiers.
AGENTS: dict[str, dict[str, Any]] = {
    "GL-001": {
        "name": "Thermalcommand",
        "port": 8000,
        "path": "applications/GL Agents/GL-001_Thermalcommand",
        "description": "Multi-Equipment Thermal Asset Optimization",
    },
    "GL-002": {
        "name": "Flameguard",
        "port": 8000,
        "path": "applications/GL Agents/GL-002_Flameguard",
        "description": "Industrial Flame Safety and Monitoring",
    },
    "GL-003": {
        "name": "UnifiedSteam",
        "port": 8000,
        "path": "applications/GL Agents/GL-003_UnifiedSteam",
        "description": "Unified Steam System Management",
    },
    "GL-004": {
        "name": "Burnmaster",
        "port": 8000,
        "path": "applications/GL Agents/GL-004_Burnmaster",
        "description": "Combustion Process Optimization",
    },
    "GL-005": {
        "name": "Combusense",
        "port": 8000,
        "path": "applications/GL Agents/GL-005_Combusense",
        "description": "Combustion Sensing and Analytics",
    },
    "GL-006": {
        "name": "HeatReclaim",
        "port": 8000,
        "path": "applications/GL Agents/GL-006_HEATRECLAIM",
        "description": "Waste Heat Recovery Optimization",
    },
    "GL-007": {
        "name": "FurnacePulse",
        "port": 8000,
        "path": "applications/GL Agents/GL-007_FurnacePulse",
        "description": "Industrial Furnace Performance Monitoring",
    },
    "GL-008": {
        "name": "Trapcatcher",
        "port": 8000,
        "path": "applications/GL Agents/GL-008_Trapcatcher",
        "description": "Steam Trap Failure Detection",
    },
    "GL-009": {
        "name": "ThermalIQ",
        "port": 8080,
        "path": "applications/GL Agents/GL-009_ThermalIQ",
        "description": "Thermal Intelligence and Diagnostics",
    },
    "GL-010": {
        "name": "EmissionGuardian",
        "port": 8000,
        "path": "applications/GL Agents/GL-010_EmissionGuardian",
        "description": "Real-time Emission Monitoring and Alerting",
    },
    "GL-011": {
        "name": "FuelCraft",
        "port": 8000,
        "path": "applications/GL Agents/GL-011_FuelCraft",
        "description": "Fuel Mix Optimization and Tracking",
    },
    "GL-012": {
        "name": "SteamQual",
        "port": 8000,
        "path": "applications/GL Agents/GL-012_SteamQual",
        "description": "Steam Quality Assurance Agent",
    },
    "GL-014": {
        "name": "Exchangerpro",
        "port": 8000,
        "path": "applications/GL Agents/GL-014_Exchangerpro",
        "description": "Heat Exchanger Performance Optimization",
    },
    "GL-017": {
        "name": "Condensync",
        "port": 8000,
        "path": "applications/GL Agents/GL-017_Condensync",
        "description": "Condensate Recovery Synchronization",
    },
}

# ---------------------------------------------------------------------------
# CLI Variants
# ---------------------------------------------------------------------------
CLI_VARIANTS: dict[str, dict[str, str]] = {
    "core": {
        "extras": "",
        "description": "Minimal CLI -- greenlang-cli with no optional deps",
    },
    "full": {
        "extras": "full",
        "description": "Full development environment with all optional deps",
    },
    "secure": {
        "extras": "server,security",
        "description": "Server runtime with security hardening packages",
    },
}

# ---------------------------------------------------------------------------
# Application Registry
# ---------------------------------------------------------------------------
APPS: dict[str, dict[str, Any]] = {
    "csrd": {
        "name": "CSRD Platform",
        "module": "main:app",
        "port": 8000,
        "path": "applications/GL-CSRD-APP/CSRD-Reporting-Platform",
        "description": "CSRD/ESRS Digital Reporting Platform",
    },
    "vcci-backend": {
        "name": "VCCI Backend",
        "module": "main:app",
        "port": 8000,
        "path": "applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/backend",
        "description": "VCCI Scope 3 Carbon Platform API",
    },
    "vcci-worker": {
        "name": "VCCI Worker",
        "module": "worker:app",
        "port": 8001,
        "path": "applications/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/worker",
        "description": "VCCI Scope 3 Background Worker",
    },
    "cbam": {
        "name": "CBAM Copilot",
        "module": "main:app",
        "port": 8000,
        "path": "applications/GL-CBAM-APP/CBAM-Importer-Copilot",
        "description": "EU CBAM Importer Compliance Copilot",
    },
}


# ============================================================================
# Helpers
# ============================================================================


def _get_project_version() -> str:
    """Read the canonical version from pyproject.toml."""
    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.exists():
        print("[WARN] pyproject.toml not found, falling back to 0.0.0-dev")
        return "0.0.0-dev"
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match:
        return match.group(1)
    print("[WARN] Could not parse version from pyproject.toml, falling back to 0.0.0-dev")
    return "0.0.0-dev"


def _get_vcs_ref() -> str:
    """Return the current git short SHA, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _get_build_date() -> str:
    """ISO-8601 UTC build timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _image_tag(registry: str, name: str, tag: str) -> str:
    """Build a full image reference: registry/name:tag."""
    return f"{registry}/{name}:{tag}"


def _run_docker(
    cmd: list[str],
    *,
    dry_run: bool = False,
    label: str = "",
) -> int:
    """Execute a docker command (or print it in dry-run mode)."""
    display = " ".join(cmd)
    if dry_run:
        print(f"[DRY-RUN] {display}")
        return 0
    print(f"\n{'=' * 72}")
    if label:
        print(f"  {label}")
        print(f"{'=' * 72}")
    print(f"$ {display}\n")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def _ensure_template(template_path: Path) -> bool:
    """Verify the Dockerfile template exists."""
    if not template_path.exists():
        print(
            f"[ERROR] Template not found: {template_path}\n"
            f"  Expected at: {template_path.relative_to(PROJECT_ROOT)}\n"
            f"  Run the template generation step first."
        )
        return False
    return True


# ============================================================================
# Build Functions
# ============================================================================


def build_agent(
    agent_id: str,
    *,
    registry: str,
    version: str,
    dry_run: bool = False,
    no_cache: bool = False,
    push: bool = False,
    platform: str | None = None,
) -> int:
    """Build a single GL Agent Docker image."""
    agent_id_upper = agent_id.upper()
    if agent_id_upper not in AGENTS:
        print(f"[ERROR] Unknown agent: {agent_id}")
        print(f"  Available: {', '.join(sorted(AGENTS.keys()))}")
        return 1

    if not _ensure_template(TEMPLATE_AGENT):
        return 1

    spec = AGENTS[agent_id_upper]
    agent_name = spec["name"]
    agent_port = spec["port"]
    agent_path = spec["path"]
    image_name = f"gl-agent-{agent_name.lower()}"
    vcs_ref = _get_vcs_ref()
    build_date = _get_build_date()

    # Verify agent source path exists
    agent_src = PROJECT_ROOT / agent_path
    if not agent_src.exists():
        print(f"[WARN] Agent source path does not exist: {agent_path}")
        print(f"  Build will proceed but may fail if the context is wrong.")

    # Construct docker build command
    cmd = ["docker", "build"]

    # Build args for the template
    build_args = {
        "AGENT_ID": agent_id_upper,
        "AGENT_NAME": agent_name,
        "AGENT_PORT": str(agent_port),
        "AGENT_PATH": agent_path,
        "VERSION": version,
        "BUILD_DATE": build_date,
        "VCS_REF": vcs_ref,
        "GL_VERSION": version,
    }

    for key, val in build_args.items():
        cmd.extend(["--build-arg", f"{key}={val}"])

    # Dockerfile template
    cmd.extend(["-f", str(TEMPLATE_AGENT)])

    # Tags: version + latest
    version_tag = _image_tag(registry, image_name, version)
    latest_tag = _image_tag(registry, image_name, "latest")
    cmd.extend(["-t", version_tag, "-t", latest_tag])

    # Optional flags
    if no_cache:
        cmd.append("--no-cache")
    if platform:
        cmd.extend(["--platform", platform])

    # OCI labels
    cmd.extend([
        "--label", f"org.opencontainers.image.title={agent_id_upper} {agent_name}",
        "--label", f"org.opencontainers.image.description={spec.get('description', '')}",
        "--label", f"org.opencontainers.image.version={version}",
        "--label", f"org.opencontainers.image.created={build_date}",
        "--label", f"org.opencontainers.image.revision={vcs_ref}",
        "--label", "org.opencontainers.image.vendor=GreenLang",
    ])

    # Build context is project root so templates can COPY from any path
    cmd.append(".")

    rc = _run_docker(cmd, dry_run=dry_run, label=f"Building agent {agent_id_upper} ({agent_name})")
    if rc != 0:
        return rc

    # Push if requested
    if push:
        for tag in (version_tag, latest_tag):
            rc = _run_docker(["docker", "push", tag], dry_run=dry_run, label=f"Pushing {tag}")
            if rc != 0:
                return rc

    print(f"\n  [OK] {agent_id_upper} ({agent_name})")
    print(f"       {version_tag}")
    print(f"       {latest_tag}")
    return 0


def build_cli(
    variant: str,
    *,
    registry: str,
    version: str,
    dry_run: bool = False,
    no_cache: bool = False,
    push: bool = False,
    platform: str | None = None,
) -> int:
    """Build a GreenLang CLI Docker image variant."""
    variant_lower = variant.lower()
    if variant_lower not in CLI_VARIANTS:
        print(f"[ERROR] Unknown CLI variant: {variant}")
        print(f"  Available: {', '.join(sorted(CLI_VARIANTS.keys()))}")
        return 1

    if not _ensure_template(TEMPLATE_CLI):
        return 1

    spec = CLI_VARIANTS[variant_lower]
    extras = spec["extras"]
    image_name = f"greenlang-cli-{variant_lower}"
    vcs_ref = _get_vcs_ref()
    build_date = _get_build_date()

    cmd = ["docker", "build"]

    build_args = {
        "CLI_VARIANT": variant_lower,
        "CLI_EXTRAS": extras,
        "VERSION": version,
        "GL_VERSION": version,
        "BUILD_DATE": build_date,
        "VCS_REF": vcs_ref,
    }

    for key, val in build_args.items():
        cmd.extend(["--build-arg", f"{key}={val}"])

    cmd.extend(["-f", str(TEMPLATE_CLI)])

    version_tag = _image_tag(registry, image_name, version)
    latest_tag = _image_tag(registry, image_name, "latest")
    cmd.extend(["-t", version_tag, "-t", latest_tag])

    if no_cache:
        cmd.append("--no-cache")
    if platform:
        cmd.extend(["--platform", platform])

    cmd.extend([
        "--label", f"org.opencontainers.image.title=GreenLang CLI ({variant_lower})",
        "--label", f"org.opencontainers.image.description={spec['description']}",
        "--label", f"org.opencontainers.image.version={version}",
        "--label", f"org.opencontainers.image.created={build_date}",
        "--label", f"org.opencontainers.image.revision={vcs_ref}",
        "--label", "org.opencontainers.image.vendor=GreenLang",
    ])

    cmd.append(".")

    rc = _run_docker(cmd, dry_run=dry_run, label=f"Building CLI variant: {variant_lower}")
    if rc != 0:
        return rc

    if push:
        for tag in (version_tag, latest_tag):
            rc = _run_docker(["docker", "push", tag], dry_run=dry_run, label=f"Pushing {tag}")
            if rc != 0:
                return rc

    print(f"\n  [OK] CLI {variant_lower}")
    print(f"       {version_tag}")
    print(f"       {latest_tag}")
    return 0


def build_app(
    app_key: str,
    *,
    registry: str,
    version: str,
    dry_run: bool = False,
    no_cache: bool = False,
    push: bool = False,
    platform: str | None = None,
) -> int:
    """Build an application API Docker image."""
    app_lower = app_key.lower()
    if app_lower not in APPS:
        print(f"[ERROR] Unknown app: {app_key}")
        print(f"  Available: {', '.join(sorted(APPS.keys()))}")
        return 1

    if not _ensure_template(TEMPLATE_API):
        return 1

    spec = APPS[app_lower]
    app_name = spec["name"]
    app_module = spec["module"]
    app_port = spec["port"]
    app_path = spec["path"]
    image_name = f"greenlang-{app_lower}"
    vcs_ref = _get_vcs_ref()
    build_date = _get_build_date()

    # Verify app source path exists
    app_src = PROJECT_ROOT / app_path
    if not app_src.exists():
        print(f"[WARN] App source path does not exist: {app_path}")
        print(f"  Build will proceed but may fail if the context is wrong.")

    cmd = ["docker", "build"]

    build_args = {
        "APP_NAME": app_name,
        "APP_MODULE": app_module,
        "APP_PORT": str(app_port),
        "APP_PATH": app_path,
        "VERSION": version,
        "GL_VERSION": version,
        "BUILD_DATE": build_date,
        "VCS_REF": vcs_ref,
    }

    for key, val in build_args.items():
        cmd.extend(["--build-arg", f"{key}={val}"])

    cmd.extend(["-f", str(TEMPLATE_API)])

    version_tag = _image_tag(registry, image_name, version)
    latest_tag = _image_tag(registry, image_name, "latest")
    cmd.extend(["-t", version_tag, "-t", latest_tag])

    if no_cache:
        cmd.append("--no-cache")
    if platform:
        cmd.extend(["--platform", platform])

    cmd.extend([
        "--label", f"org.opencontainers.image.title={app_name}",
        "--label", f"org.opencontainers.image.description={spec.get('description', '')}",
        "--label", f"org.opencontainers.image.version={version}",
        "--label", f"org.opencontainers.image.created={build_date}",
        "--label", f"org.opencontainers.image.revision={vcs_ref}",
        "--label", "org.opencontainers.image.vendor=GreenLang",
    ])

    cmd.append(".")

    rc = _run_docker(cmd, dry_run=dry_run, label=f"Building app: {app_lower} ({app_name})")
    if rc != 0:
        return rc

    if push:
        for tag in (version_tag, latest_tag):
            rc = _run_docker(["docker", "push", tag], dry_run=dry_run, label=f"Pushing {tag}")
            if rc != 0:
                return rc

    print(f"\n  [OK] App {app_lower} ({app_name})")
    print(f"       {version_tag}")
    print(f"       {latest_tag}")
    return 0


# ============================================================================
# List Command
# ============================================================================


def cmd_list(args: argparse.Namespace) -> int:
    """List all buildable images."""
    version = args.version or _get_project_version()
    registry = args.registry

    print("=" * 72)
    print("  GreenLang Docker Images")
    print(f"  Version: {version}  |  Registry: {registry}")
    print("=" * 72)

    # Agents
    print(f"\n  AGENTS ({len(AGENTS)} images)")
    print(f"  {'ID':<10} {'Name':<20} {'Port':<6} {'Image'}")
    print(f"  {'-'*10} {'-'*20} {'-'*6} {'-'*40}")
    for agent_id in sorted(AGENTS.keys()):
        spec = AGENTS[agent_id]
        img = f"gl-agent-{spec['name'].lower()}"
        print(f"  {agent_id:<10} {spec['name']:<20} {spec['port']:<6} {registry}/{img}:{version}")

    # CLI
    print(f"\n  CLI VARIANTS ({len(CLI_VARIANTS)} images)")
    print(f"  {'Variant':<10} {'Extras':<25} {'Image'}")
    print(f"  {'-'*10} {'-'*25} {'-'*40}")
    for variant, spec in sorted(CLI_VARIANTS.items()):
        extras_display = spec["extras"] if spec["extras"] else "(none)"
        img = f"greenlang-cli-{variant}"
        print(f"  {variant:<10} {extras_display:<25} {registry}/{img}:{version}")

    # Apps
    print(f"\n  APPLICATIONS ({len(APPS)} images)")
    print(f"  {'Key':<16} {'Name':<20} {'Port':<6} {'Image'}")
    print(f"  {'-'*16} {'-'*20} {'-'*6} {'-'*40}")
    for app_key in sorted(APPS.keys()):
        spec = APPS[app_key]
        img = f"greenlang-{app_key}"
        print(f"  {app_key:<16} {spec['name']:<20} {spec['port']:<6} {registry}/{img}:{version}")

    total = len(AGENTS) + len(CLI_VARIANTS) + len(APPS)
    print(f"\n  TOTAL: {total} buildable images")
    print(f"  Templates: {TEMPLATE_DIR.relative_to(PROJECT_ROOT)}/")
    for tmpl in (TEMPLATE_AGENT, TEMPLATE_CLI, TEMPLATE_API):
        exists = "[OK]" if tmpl.exists() else "[MISSING]"
        print(f"    {exists} {tmpl.name}")

    return 0


# ============================================================================
# Audit Command
# ============================================================================


# Known template paths (relative to PROJECT_ROOT, normalized to forward slash)
TEMPLATE_RELPATHS = {
    "deployment/docker/templates/Dockerfile.agent",
    "deployment/docker/templates/Dockerfile.cli",
    "deployment/docker/templates/Dockerfile.api",
    "deployment/docker/base/Dockerfile.base",
}

# Directories whose Dockerfiles are planning documents, not build artifacts
PLANNING_DIRS = {
    "docs/planning",
    "GreenLang Development",
}

# Directories whose Dockerfiles are generated artifacts (reports, test templates)
GENERATED_DIRS = {
    "reports/results/artifacts",
    "greenlang/tests/templates",
}

# Cookiecutter / Jinja templates (not actual Dockerfiles)
COOKIECUTTER_PATTERNS = {
    "{{cookiecutter",
    ".j2",
}


def _classify_dockerfile(rel_path: str) -> str:
    """
    Classify a Dockerfile by its location and purpose.

    Returns one of:
        template           -- One of the 3 canonical templates or the base image
        in-use             -- Currently required (e.g. normalizer infra, frontend)
        replaceable        -- Agent/app Dockerfile that could use a template instead
        generated-artifact -- Auto-generated output in reports or test fixtures
        planning-doc       -- Vision/planning docs, not real build files
        cookiecutter       -- Jinja2 template for code generation
        duplicate-mirror   -- Copy under GreenLang Development/ mirroring applications/
    """
    norm = rel_path.replace("\\", "/")

    # Templates themselves
    if norm in TEMPLATE_RELPATHS:
        return "template"

    # Cookiecutter / Jinja2
    for pat in COOKIECUTTER_PATTERNS:
        if pat in norm:
            return "cookiecutter"

    # Planning docs
    for pd in PLANNING_DIRS:
        if norm.startswith(pd):
            # GreenLang Development mirrors are duplicates
            if norm.startswith("GreenLang Development/"):
                return "duplicate-mirror"
            return "planning-doc"

    # Generated artifacts
    for gd in GENERATED_DIRS:
        if norm.startswith(gd):
            return "generated-artifact"

    # Agent Dockerfiles (could be replaced by Dockerfile.agent template)
    if "GL Agents/" in norm or "GL-Agent" in norm:
        return "replaceable"

    # App Dockerfiles that have a matching APPS entry
    for app_key, spec in APPS.items():
        if norm.startswith(spec["path"].replace("\\", "/")):
            return "replaceable"

    # deployment/docker/ agent-specific Dockerfiles
    if norm.startswith("deployment/docker/Dockerfile."):
        basename = norm.split("/")[-1]
        # The named agent Dockerfiles (duplicate-detector, outlier-detector, etc.)
        if basename not in ("Dockerfile.base",):
            return "replaceable"

    # deployment/ top-level Dockerfiles that map to CLI variants
    if norm.startswith("deployment/Dockerfile."):
        return "replaceable"

    # Root Dockerfile
    if norm == "Dockerfile":
        return "replaceable"

    # Everything else is considered in-use (frontend, normalizer, etc.)
    return "in-use"


def cmd_audit(args: argparse.Namespace) -> int:
    """Find all Dockerfiles and classify them relative to the template strategy."""
    print("=" * 72)
    print("  GreenLang Dockerfile Audit")
    print(f"  Scanning: {PROJECT_ROOT}")
    print("=" * 72)

    # Collect all Dockerfile* files
    dockerfiles: list[Path] = []
    for root, dirs, files in os.walk(str(PROJECT_ROOT)):
        # Skip .git and node_modules
        root_path = Path(root)
        skip = False
        for part in root_path.parts:
            if part in (".git", "node_modules", "__pycache__", ".tox", "venv", ".venv"):
                skip = True
                break
        if skip:
            continue
        for f in files:
            if f.startswith("Dockerfile") or f.endswith(".Dockerfile"):
                dockerfiles.append(Path(root) / f)

    if not dockerfiles:
        print("\n  No Dockerfiles found.")
        return 0

    # Classify
    categories: dict[str, list[str]] = {
        "template": [],
        "in-use": [],
        "replaceable": [],
        "generated-artifact": [],
        "planning-doc": [],
        "cookiecutter": [],
        "duplicate-mirror": [],
    }

    for df in sorted(dockerfiles):
        try:
            rel = str(df.relative_to(PROJECT_ROOT))
        except ValueError:
            rel = str(df)
        classification = _classify_dockerfile(rel)
        categories[classification].append(rel)

    # Report
    print(f"\n  Found {len(dockerfiles)} Dockerfile(s)\n")

    category_labels = {
        "template": "TEMPLATES (canonical, keep)",
        "in-use": "IN-USE (not replaceable by agent/cli/api templates)",
        "replaceable": "REPLACEABLE (can migrate to templates)",
        "generated-artifact": "GENERATED ARTIFACTS (auto-generated, safe to ignore)",
        "planning-doc": "PLANNING DOCS (vision/design docs, not real builds)",
        "cookiecutter": "COOKIECUTTER TEMPLATES (Jinja2 code-gen, keep)",
        "duplicate-mirror": "DUPLICATE MIRRORS (GreenLang Development/ copies)",
    }

    for cat, label in category_labels.items():
        items = categories[cat]
        if not items:
            continue
        print(f"  [{len(items):>3}] {label}")
        for item in items:
            print(f"        {item}")
        print()

    # Summary
    replaceable_count = len(categories["replaceable"])
    mirror_count = len(categories["duplicate-mirror"])
    planning_count = len(categories["planning-doc"])
    generated_count = len(categories["generated-artifact"])
    template_count = len(categories["template"])
    in_use_count = len(categories["in-use"])

    print("  " + "-" * 68)
    print(f"  SUMMARY")
    print(f"    Templates:             {template_count:>3}")
    print(f"    In-use (keep):         {in_use_count:>3}")
    print(f"    Replaceable:           {replaceable_count:>3}  <-- migrate to templates")
    print(f"    Duplicate mirrors:     {mirror_count:>3}  <-- safe to delete")
    print(f"    Planning docs:         {planning_count:>3}  <-- not real builds")
    print(f"    Generated artifacts:   {generated_count:>3}  <-- auto-generated")
    print(f"    Cookiecutter:          {len(categories['cookiecutter']):>3}  <-- code-gen templates")
    print(f"  " + "-" * 68)

    removable = replaceable_count + mirror_count
    print(f"\n  ACTION: {removable} Dockerfiles can be removed/replaced by templates.")
    if removable > 0:
        print(f"  Use 'python scripts/docker_build.py list' to see the template-based builds.")

    # Output machine-readable JSON if requested
    if getattr(args, "json", False):
        output = {
            "total": len(dockerfiles),
            "categories": {k: v for k, v in categories.items()},
            "summary": {
                "templates": template_count,
                "in_use": in_use_count,
                "replaceable": replaceable_count,
                "duplicate_mirrors": mirror_count,
                "planning_docs": planning_count,
                "generated_artifacts": generated_count,
            },
        }
        json_path = PROJECT_ROOT / "reports" / "dockerfile_audit.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\n  JSON report written to: {json_path.relative_to(PROJECT_ROOT)}")

    return 0


# ============================================================================
# CLI Entrypoint
# ============================================================================


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add flags common to all build subcommands."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print docker commands without executing them",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Pass --no-cache to docker build",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        default=False,
        help="Push images to registry after building",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=DEFAULT_REGISTRY,
        help=f"Container registry (default: {DEFAULT_REGISTRY})",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Override version tag (default: read from pyproject.toml)",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Target platform (e.g. linux/amd64,linux/arm64)",
    )


def cmd_agent(args: argparse.Namespace) -> int:
    """Handle the 'agent' subcommand."""
    version = args.version or _get_project_version()
    common = dict(
        registry=args.registry,
        version=version,
        dry_run=args.dry_run,
        no_cache=args.no_cache,
        push=args.push,
        platform=args.platform,
    )

    if args.all:
        print(f"Building ALL {len(AGENTS)} agents (version {version})...\n")
        failed = []
        for agent_id in sorted(AGENTS.keys()):
            rc = build_agent(agent_id, **common)
            if rc != 0:
                failed.append(agent_id)
        if failed:
            print(f"\n[FAIL] {len(failed)} agent(s) failed: {', '.join(failed)}")
            return 1
        print(f"\n[OK] All {len(AGENTS)} agents built successfully.")
        return 0
    else:
        if not args.target:
            print("[ERROR] Specify an agent ID (e.g. GL-001) or use --all")
            return 1
        return build_agent(args.target, **common)


def cmd_cli_build(args: argparse.Namespace) -> int:
    """Handle the 'cli' subcommand."""
    version = args.version or _get_project_version()
    common = dict(
        registry=args.registry,
        version=version,
        dry_run=args.dry_run,
        no_cache=args.no_cache,
        push=args.push,
        platform=args.platform,
    )

    if args.all:
        print(f"Building ALL {len(CLI_VARIANTS)} CLI variants (version {version})...\n")
        failed = []
        for variant in sorted(CLI_VARIANTS.keys()):
            rc = build_cli(variant, **common)
            if rc != 0:
                failed.append(variant)
        if failed:
            print(f"\n[FAIL] {len(failed)} CLI variant(s) failed: {', '.join(failed)}")
            return 1
        print(f"\n[OK] All {len(CLI_VARIANTS)} CLI variants built successfully.")
        return 0
    else:
        if not args.target:
            print("[ERROR] Specify a CLI variant (core, full, secure) or use --all")
            return 1
        return build_cli(args.target, **common)


def cmd_app_build(args: argparse.Namespace) -> int:
    """Handle the 'app' subcommand."""
    version = args.version or _get_project_version()
    common = dict(
        registry=args.registry,
        version=version,
        dry_run=args.dry_run,
        no_cache=args.no_cache,
        push=args.push,
        platform=args.platform,
    )

    if args.all:
        print(f"Building ALL {len(APPS)} apps (version {version})...\n")
        failed = []
        for app_key in sorted(APPS.keys()):
            rc = build_app(app_key, **common)
            if rc != 0:
                failed.append(app_key)
        if failed:
            print(f"\n[FAIL] {len(failed)} app(s) failed: {', '.join(failed)}")
            return 1
        print(f"\n[OK] All {len(APPS)} apps built successfully.")
        return 0
    else:
        if not args.target:
            print("[ERROR] Specify an app key (csrd, vcci-backend, etc.) or use --all")
            return 1
        return build_app(args.target, **common)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="docker_build",
        description="GreenLang Docker Build Tool -- build images from parameterized templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/docker_build.py agent GL-001              Build single agent
  python scripts/docker_build.py agent --all               Build all agents
  python scripts/docker_build.py agent GL-005 --dry-run    Preview build command
  python scripts/docker_build.py cli core                  Build CLI core
  python scripts/docker_build.py cli full --push           Build and push CLI full
  python scripts/docker_build.py app csrd                  Build CSRD app
  python scripts/docker_build.py app --all --no-cache      Rebuild all apps from scratch
  python scripts/docker_build.py list                      List all buildable images
  python scripts/docker_build.py audit                     Audit stale Dockerfiles
  python scripts/docker_build.py audit --json              Audit with JSON report
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Build command")

    # -- agent --
    agent_parser = subparsers.add_parser("agent", help="Build GL Agent image(s)")
    agent_parser.add_argument("target", nargs="?", default=None, help="Agent ID (e.g. GL-001)")
    agent_parser.add_argument("--all", action="store_true", help="Build all agents")
    _add_common_args(agent_parser)

    # -- cli --
    cli_parser = subparsers.add_parser("cli", help="Build GreenLang CLI image")
    cli_parser.add_argument("target", nargs="?", default=None, help="CLI variant (core, full, secure)")
    cli_parser.add_argument("--all", action="store_true", help="Build all CLI variants")
    _add_common_args(cli_parser)

    # -- app --
    app_parser = subparsers.add_parser("app", help="Build application API image")
    app_parser.add_argument("target", nargs="?", default=None, help="App key (csrd, vcci-backend, etc.)")
    app_parser.add_argument("--all", action="store_true", help="Build all apps")
    _add_common_args(app_parser)

    # -- list --
    list_parser = subparsers.add_parser("list", help="List all buildable images")
    list_parser.add_argument(
        "--registry",
        type=str,
        default=DEFAULT_REGISTRY,
        help=f"Container registry (default: {DEFAULT_REGISTRY})",
    )
    list_parser.add_argument("--version", type=str, default=None, help="Override version tag")

    # -- audit --
    audit_parser = subparsers.add_parser("audit", help="Audit Dockerfiles not using templates")
    audit_parser.add_argument("--json", action="store_true", help="Write JSON report to reports/")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    dispatch = {
        "agent": cmd_agent,
        "cli": cmd_cli_build,
        "app": cmd_app_build,
        "list": cmd_list,
        "audit": cmd_audit,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
