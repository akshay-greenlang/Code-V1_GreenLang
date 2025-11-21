#!/bin/bash
# Clean all build artifacts and caches in the GreenLang monorepo
# This script removes node_modules, dist, and .turbo directories

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
echo -e "${BLUE}  GreenLang Monorepo Clean Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Change to project root
cd "${PROJECT_ROOT}"

# Parse command line arguments
CLEAN_CACHE=false
CLEAN_DEPS=false
CLEAN_BUILD=false
CLEAN_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cache)
            CLEAN_CACHE=true
            shift
            ;;
        --deps)
            CLEAN_DEPS=true
            shift
            ;;
        --build)
            CLEAN_BUILD=true
            shift
            ;;
        --all)
            CLEAN_ALL=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cache    Clean Turborepo cache (.turbo)"
            echo "  --deps     Clean dependencies (node_modules)"
            echo "  --build    Clean build artifacts (dist, build)"
            echo "  --all      Clean everything"
            echo "  --help     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --cache                # Clean only cache"
            echo "  $0 --build --deps         # Clean build and deps"
            echo "  $0 --all                  # Clean everything"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# If no options specified, show help
if [ "$CLEAN_CACHE" = false ] && [ "$CLEAN_DEPS" = false ] && [ "$CLEAN_BUILD" = false ] && [ "$CLEAN_ALL" = false ]; then
    echo -e "${YELLOW}No clean option specified. Use --help for usage.${NC}"
    exit 1
fi

# Set all flags if --all is specified
if [ "$CLEAN_ALL" = true ]; then
    CLEAN_CACHE=true
    CLEAN_DEPS=true
    CLEAN_BUILD=true
fi

# Clean Turborepo cache
if [ "$CLEAN_CACHE" = true ]; then
    echo -e "${BLUE}Cleaning Turborepo cache...${NC}"
    if [ -d ".turbo" ]; then
        rm -rf .turbo
        echo -e "${GREEN}✓ Removed .turbo${NC}"
    else
        echo -e "${YELLOW}No .turbo directory found${NC}"
    fi

    # Clean workspace .turbo directories
    find . -type d -name ".turbo" -not -path "*/node_modules/*" -exec rm -rf {} + 2>/dev/null || true
    echo ""
fi

# Clean build artifacts
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${BLUE}Cleaning build artifacts...${NC}"

    # Remove dist directories
    find . -type d -name "dist" -not -path "*/node_modules/*" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ Removed dist directories${NC}"

    # Remove build directories
    find . -type d -name "build" -not -path "*/node_modules/*" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ Removed build directories${NC}"

    # Remove Python build artifacts
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    echo -e "${GREEN}✓ Removed Python build artifacts${NC}"

    # Remove coverage reports
    find . -type d -name "coverage" -not -path "*/node_modules/*" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "htmlcov" -not -path "*/node_modules/*" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name ".coverage" -delete 2>/dev/null || true
    echo -e "${GREEN}✓ Removed coverage reports${NC}"

    echo ""
fi

# Clean dependencies
if [ "$CLEAN_DEPS" = true ]; then
    echo -e "${BLUE}Cleaning dependencies...${NC}"
    echo -e "${YELLOW}This may take a while...${NC}"

    # Remove node_modules
    if [ -d "node_modules" ]; then
        rm -rf node_modules
        echo -e "${GREEN}✓ Removed root node_modules${NC}"
    fi

    # Remove workspace node_modules
    find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ Removed workspace node_modules${NC}"

    # Remove Python virtual environments
    find . -type d -name "venv" -not -path "*/node_modules/*" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".venv" -not -path "*/node_modules/*" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ Removed Python virtual environments${NC}"

    echo ""
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Cleanup completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}To reinstall dependencies, run:${NC}"
echo -e "  npm install"
echo ""
