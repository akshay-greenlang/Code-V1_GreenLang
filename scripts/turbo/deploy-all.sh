#!/bin/bash
# Deploy all applications in the GreenLang monorepo using Turborepo
# This script ensures build, test, and lint pass before deployment

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
echo -e "${BLUE}  GreenLang Monorepo Deploy Script${NC}"
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
ENVIRONMENT="staging"
SKIP_TESTS=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --filter)
            FILTER="--filter=$2"
            shift 2
            ;;
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS=true
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
            echo "  --filter <package>    Deploy only specified package"
            echo "  --env <environment>   Target environment (dev|staging|production)"
            echo "  --skip-tests          Skip test execution (not recommended)"
            echo "  --dry-run             Show what would be deployed"
            echo "  --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --env=production                   # Deploy all to production"
            echo "  $0 --filter=@greenlang/frontend       # Deploy only frontend"
            echo "  $0 --env=staging --skip-tests         # Deploy to staging without tests"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|production)$ ]]; then
    echo -e "${RED}Invalid environment: $ENVIRONMENT${NC}"
    echo -e "${YELLOW}Valid environments: dev, staging, production${NC}"
    exit 1
fi

# Confirmation for production
if [ "$ENVIRONMENT" = "production" ] && [ "$DRY_RUN" = false ]; then
    echo -e "${YELLOW}WARNING: You are about to deploy to PRODUCTION!${NC}"
    read -p "Are you sure? (yes/no): " -r
    echo
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        echo -e "${RED}Deployment cancelled.${NC}"
        exit 1
    fi
fi

echo -e "${BLUE}Target Environment: ${ENVIRONMENT}${NC}"
echo ""

# Step 1: Lint
echo -e "${BLUE}Step 1/4: Running linters...${NC}"
LINT_CMD="turbo run lint"
if [ -n "$FILTER" ]; then
    LINT_CMD="$LINT_CMD $FILTER"
fi

if ! eval "$LINT_CMD"; then
    echo -e "${RED}Linting failed! Fix errors before deploying.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Linting passed${NC}"
echo ""

# Step 2: Build
echo -e "${BLUE}Step 2/4: Building applications...${NC}"
BUILD_CMD="turbo run build"
if [ -n "$FILTER" ]; then
    BUILD_CMD="$BUILD_CMD $FILTER"
fi

if ! eval "$BUILD_CMD"; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Build successful${NC}"
echo ""

# Step 3: Test
if [ "$SKIP_TESTS" = false ]; then
    echo -e "${BLUE}Step 3/4: Running tests...${NC}"
    TEST_CMD="turbo run test"
    if [ -n "$FILTER" ]; then
        TEST_CMD="$TEST_CMD $FILTER"
    fi

    if ! eval "$TEST_CMD"; then
        echo -e "${RED}Tests failed! Fix tests before deploying.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Tests passed${NC}"
    echo ""
else
    echo -e "${YELLOW}Step 3/4: Skipping tests (not recommended)${NC}"
    echo ""
fi

# Step 4: Deploy
echo -e "${BLUE}Step 4/4: Deploying to ${ENVIRONMENT}...${NC}"

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN: Would deploy the following:${NC}"
    DEPLOY_CMD="turbo run deploy --dry-run"
else
    DEPLOY_CMD="turbo run deploy"
fi

if [ -n "$FILTER" ]; then
    DEPLOY_CMD="$DEPLOY_CMD $FILTER"
fi

export NODE_ENV="$ENVIRONMENT"

if eval "$DEPLOY_CMD"; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Deployment completed successfully!${NC}"
    echo -e "${GREEN}  Environment: ${ENVIRONMENT}${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Deployment failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
