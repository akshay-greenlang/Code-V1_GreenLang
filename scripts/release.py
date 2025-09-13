#!/usr/bin/env python3
"""
GreenLang Release Script

This script handles the manual release process for GreenLang:
1. Bump version in pyproject.toml and greenlang/__init__.py
2. Generate changelog from commits
3. Create git tag
4. Push to origin

Usage:
    python scripts/release.py --version <version> [--dry-run] [--push]

Examples:
    python scripts/release.py --version 0.1.1 --dry-run
    python scripts/release.py --version 0.2.0 --push
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import toml
import git


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def validate_version(version: str) -> bool:
    """Validate that version follows semantic versioning."""
    pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
    return re.match(pattern, version) is not None


def get_current_version() -> str:
    """Get current version from pyproject.toml."""
    project_root = get_project_root()
    pyproject_path = project_root / "pyproject.toml"

    with open(pyproject_path, "r") as f:
        config = toml.load(f)

    return config["project"]["version"]


def bump_version_in_files(version: str, dry_run: bool = False) -> None:
    """Bump version in pyproject.toml and greenlang/__init__.py."""
    project_root = get_project_root()

    # Update pyproject.toml
    pyproject_path = project_root / "pyproject.toml"
    print(f"Updating version in {pyproject_path}")

    if not dry_run:
        with open(pyproject_path, "r") as f:
            config = toml.load(f)

        config["project"]["version"] = version

        with open(pyproject_path, "w") as f:
            toml.dump(config, f)

    # Update greenlang/__init__.py
    init_path = project_root / "greenlang" / "__init__.py"
    print(f"Updating version in {init_path}")

    if not dry_run:
        with open(init_path, "r") as f:
            content = f.read()

        # Replace version line
        content = re.sub(
            r'__version__ = "[^"]+"',
            f'__version__ = "{version}"',
            content
        )

        with open(init_path, "w") as f:
            f.write(content)


def generate_changelog(repo: git.Repo, version: str, from_tag: Optional[str] = None) -> str:
    """Generate changelog from git commits."""
    print(f"Generating changelog for version {version}")

    # Get all tags sorted by date
    tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime, reverse=True)

    if not from_tag and tags:
        from_tag = tags[0].name

    # Generate changelog
    changelog_lines = [
        f"# Changelog",
        f"",
        f"## [{version}] - {datetime.now().strftime('%Y-%m-%d')}",
        f"",
    ]

    if from_tag:
        print(f"Generating changelog from {from_tag} to HEAD")
        commits = list(repo.iter_commits(f"{from_tag}..HEAD"))
    else:
        print("Generating changelog from beginning")
        commits = list(repo.iter_commits())

    # Categorize commits by type
    features = []
    fixes = []
    docs = []
    refactors = []
    tests = []
    chores = []
    others = []

    for commit in reversed(commits):
        # Skip merge commits
        if len(commit.parents) > 1:
            continue

        message = commit.message.strip().split('\n')[0]
        short_sha = str(commit)[:7]

        if message.startswith('feat:') or message.startswith('feature:'):
            features.append(f"- {message[5:].strip()} ({short_sha})")
        elif message.startswith('fix:'):
            fixes.append(f"- {message[4:].strip()} ({short_sha})")
        elif message.startswith('docs:'):
            docs.append(f"- {message[5:].strip()} ({short_sha})")
        elif message.startswith('refactor:'):
            refactors.append(f"- {message[9:].strip()} ({short_sha})")
        elif message.startswith('test:'):
            tests.append(f"- {message[5:].strip()} ({short_sha})")
        elif message.startswith('chore:'):
            chores.append(f"- {message[6:].strip()} ({short_sha})")
        else:
            others.append(f"- {message} ({short_sha})")

    # Add sections to changelog
    if features:
        changelog_lines.extend(["### Features", ""] + features + [""])

    if fixes:
        changelog_lines.extend(["### Bug Fixes", ""] + fixes + [""])

    if refactors:
        changelog_lines.extend(["### Refactoring", ""] + refactors + [""])

    if docs:
        changelog_lines.extend(["### Documentation", ""] + docs + [""])

    if tests:
        changelog_lines.extend(["### Testing", ""] + tests + [""])

    if chores:
        changelog_lines.extend(["### Chores", ""] + chores + [""])

    if others:
        changelog_lines.extend(["### Other Changes", ""] + others + [""])

    if from_tag:
        changelog_lines.extend([
            f"**Full Changelog**: https://github.com/YOUR_ORG/greenlang/compare/{from_tag}...v{version}",
            ""
        ])

    return "\n".join(changelog_lines)


def update_changelog_file(changelog_content: str, dry_run: bool = False) -> None:
    """Update CHANGELOG.md file."""
    project_root = get_project_root()
    changelog_path = project_root / "CHANGELOG.md"

    print(f"Updating {changelog_path}")

    if not dry_run:
        if changelog_path.exists():
            # Read existing changelog and prepend new content
            with open(changelog_path, "r") as f:
                existing_content = f.read()

            # Insert new changelog after the main title
            lines = existing_content.split('\n')
            if lines and lines[0].startswith('# '):
                # Insert after main title
                new_content = lines[0] + '\n\n' + changelog_content + '\n' + '\n'.join(lines[1:])
            else:
                # Prepend to existing content
                new_content = changelog_content + '\n\n' + existing_content
        else:
            new_content = changelog_content

        with open(changelog_path, "w") as f:
            f.write(new_content)


def create_git_tag(repo: git.Repo, version: str, dry_run: bool = False) -> None:
    """Create git tag for the release."""
    tag_name = f"v{version}"
    message = f"Release {version}"

    print(f"Creating git tag {tag_name}")

    if not dry_run:
        # Stage changes
        repo.git.add(".")

        # Commit changes
        repo.git.commit("-m", f"chore: bump version to {version}")

        # Create tag
        repo.create_tag(tag_name, message=message)
        print(f"Created tag {tag_name}")


def push_to_origin(repo: git.Repo, version: str, dry_run: bool = False) -> None:
    """Push commits and tags to origin."""
    print("Pushing to origin...")

    if not dry_run:
        # Push commits
        origin = repo.remote("origin")
        origin.push()

        # Push tags
        origin.push(tags=True)
        print("Pushed to origin successfully")


def main():
    parser = argparse.ArgumentParser(description="GreenLang Release Script")
    parser.add_argument("--version", required=True, help="Version to release (e.g., 0.1.1)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--push", action="store_true", help="Push changes and tags to origin")
    parser.add_argument("--from-tag", help="Generate changelog from specific tag")

    args = parser.parse_args()

    # Validate version
    if not validate_version(args.version):
        print(f"Error: Invalid version format: {args.version}")
        print("Version must follow semantic versioning (e.g., 0.1.1, 1.0.0-alpha.1)")
        sys.exit(1)

    # Get project root and repo
    project_root = get_project_root()

    try:
        repo = git.Repo(project_root)
    except git.InvalidGitRepositoryError:
        print("Error: Not in a git repository")
        sys.exit(1)

    # Check if working directory is clean
    if repo.is_dirty():
        print("Error: Working directory is not clean. Please commit or stash changes.")
        sys.exit(1)

    current_version = get_current_version()
    print(f"Current version: {current_version}")
    print(f"New version: {args.version}")

    if args.dry_run:
        print("\n=== DRY RUN MODE ===")

    try:
        # 1. Bump version in files
        bump_version_in_files(args.version, dry_run=args.dry_run)

        # 2. Generate changelog
        changelog_content = generate_changelog(repo, args.version, args.from_tag)
        print("\nGenerated changelog:")
        print("=" * 50)
        print(changelog_content)
        print("=" * 50)

        # 3. Update changelog file
        update_changelog_file(changelog_content, dry_run=args.dry_run)

        # 4. Create git tag
        create_git_tag(repo, args.version, dry_run=args.dry_run)

        # 5. Push to origin (if requested)
        if args.push:
            push_to_origin(repo, args.version, dry_run=args.dry_run)

        if args.dry_run:
            print("\n=== DRY RUN COMPLETED ===")
            print("Run without --dry-run to actually make changes")
        else:
            print(f"\n✅ Release {args.version} prepared successfully!")
            if not args.push:
                print("Run with --push to push changes to origin")
                print(f"Or manually push: git push origin main && git push origin v{args.version}")

    except Exception as e:
        print(f"Error during release process: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()