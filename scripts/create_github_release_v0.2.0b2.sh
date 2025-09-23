#!/bin/bash

# GitHub Release Script for GreenLang v0.2.0b2
# This script creates a pre-release on GitHub with all artifacts

set -e  # Exit on any error

# Configuration
VERSION="v0.2.0b2"
RELEASE_TITLE="v0.2.0b2 – Infra Seed (Beta 2)"
PRERELEASE=true

# Paths (adjust these based on your actual file locations)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DIST_DIR="$PROJECT_ROOT/dist"
SBOM_DIR="$PROJECT_ROOT/sbom"
RELEASE_NOTES="$PROJECT_ROOT/RELEASE_NOTES_v0.2.0b2.md"

echo "🚀 Creating GitHub pre-release for GreenLang $VERSION"
echo "📁 Project root: $PROJECT_ROOT"

# Verify required files exist
echo "🔍 Verifying artifacts..."

# Check distribution files
WHEEL_FILE="$DIST_DIR/greenlang-0.2.0b2-py3-none-any.whl"
TARBALL_FILE="$DIST_DIR/greenlang-0.2.0b2.tar.gz"

if [ ! -f "$WHEEL_FILE" ]; then
    echo "❌ ERROR: Wheel file not found: $WHEEL_FILE"
    echo "   Run 'python -m build' to generate distribution files"
    exit 1
fi

if [ ! -f "$TARBALL_FILE" ]; then
    echo "❌ ERROR: Tarball not found: $TARBALL_FILE"
    echo "   Run 'python -m build' to generate distribution files"
    exit 1
fi

echo "✅ Distribution files found"

# Check SBOM files
SBOM_FILES=(
    "$SBOM_DIR/greenlang-full-0.2.0.spdx.json"
    "$SBOM_DIR/greenlang-dist-0.2.0.spdx.json"
    "$SBOM_DIR/greenlang-runner-0.2.0.spdx.json"
)

for sbom_file in "${SBOM_FILES[@]}"; do
    if [ ! -f "$sbom_file" ]; then
        echo "⚠️  WARNING: SBOM file not found: $sbom_file"
        echo "   SBOM files will be skipped in release"
    else
        echo "✅ SBOM found: $(basename "$sbom_file")"
    fi
done

# Check release notes
if [ ! -f "$RELEASE_NOTES" ]; then
    echo "❌ ERROR: Release notes not found: $RELEASE_NOTES"
    exit 1
fi

echo "✅ Release notes found"

# Verify gh CLI is installed and authenticated
if ! command -v gh &> /dev/null; then
    echo "❌ ERROR: GitHub CLI (gh) is not installed"
    echo "   Install from: https://cli.github.com/"
    exit 1
fi

if ! gh auth status &> /dev/null; then
    echo "❌ ERROR: GitHub CLI is not authenticated"
    echo "   Run 'gh auth login' to authenticate"
    exit 1
fi

echo "✅ GitHub CLI is ready"

# Check if we're in a git repository
if [ ! -d "$PROJECT_ROOT/.git" ]; then
    echo "❌ ERROR: Not in a git repository"
    exit 1
fi

# Check if tag already exists
if git tag -l | grep -q "^$VERSION$"; then
    echo "⚠️  WARNING: Tag $VERSION already exists"
    echo "   Use 'git tag -d $VERSION' to delete it if needed"
else
    echo "✅ Tag $VERSION is available"
fi

echo ""
echo "📋 Release Summary:"
echo "   Version: $VERSION"
echo "   Title: $RELEASE_TITLE"
echo "   Pre-release: $PRERELEASE"
echo "   Wheel: $(basename "$WHEEL_FILE")"
echo "   Tarball: $(basename "$TARBALL_FILE")"
for sbom_file in "${SBOM_FILES[@]}"; do
    if [ -f "$sbom_file" ]; then
        echo "   SBOM: $(basename "$sbom_file")"
    fi
done
echo ""

# Function to create the release
create_release() {
    echo "🏷️  Creating git tag..."
    git tag "$VERSION" -m "Release $VERSION"

    echo "📤 Pushing tag to origin..."
    git push origin "$VERSION"

    echo "🎉 Creating GitHub release..."

    # Build the gh release create command
    RELEASE_CMD="gh release create $VERSION"

    # Add release notes
    RELEASE_CMD="$RELEASE_CMD --notes-file '$RELEASE_NOTES'"

    # Add title
    RELEASE_CMD="$RELEASE_CMD --title '$RELEASE_TITLE'"

    # Mark as pre-release
    if [ "$PRERELEASE" = true ]; then
        RELEASE_CMD="$RELEASE_CMD --prerelease"
    fi

    # Add distribution files
    RELEASE_CMD="$RELEASE_CMD '$WHEEL_FILE' '$TARBALL_FILE'"

    # Add SBOM files if they exist
    for sbom_file in "${SBOM_FILES[@]}"; do
        if [ -f "$sbom_file" ]; then
            RELEASE_CMD="$RELEASE_CMD '$sbom_file'"
        fi
    done

    echo "📝 Executing: $RELEASE_CMD"
    eval "$RELEASE_CMD"

    echo ""
    echo "🎉 GitHub release created successfully!"
    echo "🔗 View at: https://github.com/$(gh repo view --json owner,name -q '.owner.login + "/" + .name')/releases/tag/$VERSION"
}

# Function to show what would be executed (dry run)
show_dry_run() {
    echo "🧪 DRY RUN - Commands that would be executed:"
    echo ""
    echo "1. Create git tag:"
    echo "   git tag $VERSION -m 'Release $VERSION'"
    echo ""
    echo "2. Push tag:"
    echo "   git push origin $VERSION"
    echo ""
    echo "3. Create GitHub release:"

    RELEASE_CMD="gh release create $VERSION"
    RELEASE_CMD="$RELEASE_CMD --notes-file '$RELEASE_NOTES'"
    RELEASE_CMD="$RELEASE_CMD --title '$RELEASE_TITLE'"

    if [ "$PRERELEASE" = true ]; then
        RELEASE_CMD="$RELEASE_CMD --prerelease"
    fi

    RELEASE_CMD="$RELEASE_CMD '$WHEEL_FILE' '$TARBALL_FILE'"

    for sbom_file in "${SBOM_FILES[@]}"; do
        if [ -f "$sbom_file" ]; then
            RELEASE_CMD="$RELEASE_CMD '$sbom_file'"
        fi
    done

    echo "   $RELEASE_CMD"
    echo ""
    echo "🎯 To execute the release, run:"
    echo "   $0 --execute"
}

# Parse command line arguments
if [ "$1" = "--execute" ]; then
    echo "⚡ EXECUTING RELEASE..."
    create_release
elif [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [--execute|--help]"
    echo ""
    echo "Options:"
    echo "  --execute    Execute the release (default is dry run)"
    echo "  --help       Show this help message"
    echo ""
    echo "By default, this script runs in dry-run mode to show what would be executed."
else
    show_dry_run
fi