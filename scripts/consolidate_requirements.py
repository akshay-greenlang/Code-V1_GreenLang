#!/usr/bin/env python3
"""
Consolidate Requirements Files
==============================
Deletes redundant requirements*.txt files that are now absorbed into the root
pyproject.toml optional-dependency groups.

Usage:
    python scripts/consolidate_requirements.py --scan     # List files to delete
    python scripts/consolidate_requirements.py --apply    # Delete redundant files
    python scripts/consolidate_requirements.py --verify   # Confirm cleanup done
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────────────────────────────────────
# Files to DELETE (consolidated into pyproject.toml)
# ──────────────────────────────────────────────────────────────────────────────
DELETE_FILES = [
    # config/ -- all absorbed into pyproject.toml [project.dependencies] + extras
    "config/requirements.txt",
    "config/requirements_v0.1.txt",
    "config/requirements-test.txt",
    "config/requirements-pinned.txt",
    "config/requirements-freeze-current.txt",
    "config/requirements-lock.txt",
    # tests/ -- absorbed into pyproject.toml [test]
    "tests/requirements_test.txt",
    # greenlang/ internal -- absorbed into pyproject.toml extras
    "greenlang/monitoring/requirements.txt",
    "greenlang/integration/api/requirements.txt",
    "greenlang/integration/api/graphql/requirements.txt",
    "greenlang/config/greenlang_registry/requirements.txt",
    ".greenlang/tools/requirements.txt",
    # greenlang/tests/templates -- example/template files, deps in pyproject.toml
    "greenlang/tests/templates/data-intake-app/requirements.txt",
    "greenlang/tests/templates/calculation-app/requirements.txt",
    "greenlang/tests/templates/llm-analysis-app/requirements.txt",
    # deployment/lambda -- tiny Lambda-specific deps, absorbed into pyproject.toml [server]
    "deployment/lambda/s3-events/artifact-validator/requirements.txt",
    "deployment/lambda/s3-events/audit-logger/requirements.txt",
    "deployment/lambda/s3-events/cost-tracker/requirements.txt",
    "deployment/lambda/s3-events/report-indexer/requirements.txt",
    "deployment/terraform/modules/database-secrets/lambda/requirements.txt",
    # docs/planning -- prototype/vision files, not production
    "docs/planning/greenlang-2030-vision/agent_foundation/requirements.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/requirements-dev.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/requirements-test.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/cache/requirements.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/api/requirements.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/messaging/requirements.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-001/requirements.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-001/tests/integration/requirements-test.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-002/requirements.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-002/tests/integration/requirements-test.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-003/requirements.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-003/tests/integration/requirements-test.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-004/requirements.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-005/requirements.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-005/integrations/requirements.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-006/requirements.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-007/requirements.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-007/requirements-test.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-008/requirements.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-009/requirements.txt",
    "docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-010/requirements.txt",
    # reports/ -- generated artifacts, not dependencies
    "reports/results/artifacts/generated/fuel_analyzer_agent/requirements.txt",
    "reports/results/artifacts/generated/carbon_intensity_v1/requirements.txt",
    "reports/results/artifacts/generated/energy_performance_v1/requirements.txt",
    "reports/results/artifacts/generated/eudr_compliance_v1/requirements.txt",
]

# ──────────────────────────────────────────────────────────────────────────────
# Files to KEEP (separate applications with their own lifecycle)
# ──────────────────────────────────────────────────────────────────────────────
KEEP_NOTES = {
    "deployment/docker/base/requirements-base.txt": "Replaced with thin wrapper -> pip install greenlang-cli[server]",
    "applications/*/requirements*.txt": "Separate apps with independent deployment",
    "cbam-pack-mvp/pyproject.toml": "Separate package",
}


def run_scan():
    """List all files that will be deleted."""
    found = 0
    missing = 0
    for rel in DELETE_FILES:
        fp = ROOT / rel
        if fp.exists():
            found += 1
            print(f"  DELETE: {rel}")
        else:
            missing += 1
            print(f"  SKIP (not found): {rel}")
    print(f"\n{'='*60}")
    print(f"SCAN: {found} files to delete, {missing} already gone")
    return found


def run_apply():
    """Delete redundant requirements files and replace Docker base."""
    deleted = 0
    skipped = 0
    for rel in DELETE_FILES:
        fp = ROOT / rel
        if fp.exists():
            fp.unlink()
            deleted += 1
            print(f"  DELETED: {rel}")
        else:
            skipped += 1

    # Replace Docker base with thin wrapper
    docker_base = ROOT / "deployment" / "docker" / "base" / "requirements-base.txt"
    if docker_base.exists() or True:  # Always create/replace
        docker_base.parent.mkdir(parents=True, exist_ok=True)
        docker_base.write_text(
            "# GreenLang Docker Base Image Dependencies\n"
            "# All deps are now managed in the root pyproject.toml\n"
            "# Install via: pip install greenlang-cli[server,security]\n"
            "#\n"
            "# For local development: pip install -e .[server,security]\n"
            "# For full install:      pip install -e .[all]\n"
            "-e .[server,security]\n",
            encoding="utf-8",
        )
        print(f"  REPLACED: deployment/docker/base/requirements-base.txt (thin wrapper)")

    print(f"\n{'='*60}")
    print(f"APPLIED: {deleted} files deleted, {skipped} already gone")
    return deleted


def run_verify():
    """Confirm no redundant requirements files remain."""
    remaining = 0
    for rel in DELETE_FILES:
        fp = ROOT / rel
        if fp.exists():
            remaining += 1
            print(f"  REMAINING: {rel}")
    if remaining == 0:
        print("VERIFY: PASS - all redundant requirements files removed")
    else:
        print(f"VERIFY: FAIL - {remaining} file(s) still present")
    return remaining


def main():
    parser = argparse.ArgumentParser(description="Consolidate requirements files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scan", action="store_true")
    group.add_argument("--apply", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.scan:
        run_scan()
    elif args.apply:
        run_apply()
    elif args.verify:
        count = run_verify()
        sys.exit(1 if count > 0 else 0)


if __name__ == "__main__":
    main()
