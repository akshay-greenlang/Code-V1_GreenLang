#!/bin/bash
# Build all applications in the GreenLang monorepo using Turborepo
# This script provides optimized parallel builds with caching

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  GreenLang Monorepo Build Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if turbo is installed
if ! command -v turbo &> /dev/null; then
    echo -e "${YELLOW}Turborepo not found. Installing...${NC}"
    npm install -g turbo
fi

# Change to project root
cd "${PROJECT_ROOT}"

# Parse command line arguments
FILTER=""
FORCE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --filter)
            FILTER="--filter=$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --filter <package>    Build only specified package"
            echo "  --force               Force rebuild (ignore cache)"
            echo "  --dry-run            Show what would be built"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Build all packages"
            echo "  $0 --filter=@greenlang/frontend      # Build only frontend"
            echo "  $0 --force                            # Force rebuild all"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Build turbo command
TURBO_CMD="turbo run build"

if [ -n "$FILTER" ]; then
    TURBO_CMD="$TURBO_CMD $FILTER"
fi

if [ "$FORCE" = true ]; then
    TURBO_CMD="$TURBO_CMD --force"
fi

if [ "$DRY_RUN" = true ]; then
    TURBO_CMD="$TURBO_CMD --dry-run"
fi

echo -e "${BLUE}Command: ${TURBO_CMD}${NC}"
echo ""

# Run the build
if eval "$TURBO_CMD"; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Build completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"

    # Show cache statistics
    if [ "$DRY_RUN" = false ]; then
        echo ""
        echo -e "${BLUE}Cache statistics:${NC}"
        turbo run build --dry-run=json 2>/dev/null | grep -E "(cache|tasks)" || true
    fi
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Build failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
