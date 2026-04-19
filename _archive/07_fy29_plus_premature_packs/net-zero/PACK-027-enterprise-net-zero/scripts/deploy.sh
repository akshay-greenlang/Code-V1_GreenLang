#!/bin/bash
# PACK-027: Enterprise Net Zero Pack - Deployment Script
# ======================================================
#
# Automated deployment script for production environments.
#
# Usage:
#   ./scripts/deploy.sh [environment]
#
# Environments: dev, staging, production
#
# Author: GreenLang Platform Team
# Date: March 2026

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PACK_ID="PACK-027"
PACK_NAME="Enterprise Net Zero Pack"
PACK_VERSION="1.0.0"
ENVIRONMENT="${1:-dev}"

# Kubernetes namespace based on environment
case $ENVIRONMENT in
    dev)
        NAMESPACE="greenlang-enterprise-dev"
        REPLICAS=1
        ;;
    staging)
        NAMESPACE="greenlang-enterprise-staging"
        REPLICAS=2
        ;;
    production)
        NAMESPACE="greenlang-enterprise"
        REPLICAS=3
        ;;
    *)
        echo -e "${RED}✗ Invalid environment: ${ENVIRONMENT}${NC}"
        echo "Valid environments: dev, staging, production"
        exit 1
        ;;
esac

echo "======================================================================"
echo "PACK-027: Enterprise Net Zero Pack - Deployment"
echo "======================================================================"
echo "Environment: ${ENVIRONMENT}"
echo "Namespace: ${NAMESPACE}"
echo "Version: ${PACK_VERSION}"
echo "======================================================================"

# Pre-deployment checks
echo -e "\n${YELLOW}1. Running pre-deployment checks...${NC}"

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}✗ kubectl not found. Please install kubectl.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ kubectl installed${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found. Please install Docker.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker installed${NC}"

# Check if connected to Kubernetes cluster
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}✗ Not connected to Kubernetes cluster${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Connected to Kubernetes cluster${NC}"

# Build Docker image
echo -e "\n${YELLOW}2. Building Docker image...${NC}"
cd "$(dirname "$0")/.."
docker build -t greenlang/${PACK_ID}:${PACK_VERSION} -f deployment/Dockerfile .
echo -e "${GREEN}✓ Docker image built${NC}"

# Tag for registry (if using private registry)
if [ ! -z "${DOCKER_REGISTRY:-}" ]; then
    echo -e "\n${YELLOW}3. Tagging for registry...${NC}"
    docker tag greenlang/${PACK_ID}:${PACK_VERSION} ${DOCKER_REGISTRY}/greenlang/${PACK_ID}:${PACK_VERSION}

    echo -e "\n${YELLOW}4. Pushing to registry...${NC}"
    docker push ${DOCKER_REGISTRY}/greenlang/${PACK_ID}:${PACK_VERSION}
    echo -e "${GREEN}✓ Image pushed to registry${NC}"
fi

# Create namespace if it doesn't exist
echo -e "\n${YELLOW}5. Creating namespace...${NC}"
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
echo -e "${GREEN}✓ Namespace ready${NC}"

# Apply database migrations
echo -e "\n${YELLOW}6. Applying database migrations...${NC}"
python scripts/apply_migrations.py
echo -e "${GREEN}✓ Migrations applied${NC}"

# Verify migrations
echo -e "\n${YELLOW}7. Verifying migrations...${NC}"
python scripts/verify_migrations.py
echo -e "${GREEN}✓ Migrations verified${NC}"

# Deploy to Kubernetes
echo -e "\n${YELLOW}8. Deploying to Kubernetes...${NC}"
kubectl apply -f deployment/kubernetes.yaml -n ${NAMESPACE}
echo -e "${GREEN}✓ Kubernetes resources created${NC}"

# Wait for deployment to be ready
echo -e "\n${YELLOW}9. Waiting for deployment to be ready...${NC}"
kubectl rollout status deployment/pack027-enterprise -n ${NAMESPACE} --timeout=5m
echo -e "${GREEN}✓ Deployment ready${NC}"

# Run health checks
echo -e "\n${YELLOW}10. Running health checks...${NC}"
PACK_URL="http://localhost:8000"
if [ "${ENVIRONMENT}" != "dev" ]; then
    PACK_URL="https://pack027.greenlang.io"
fi
python scripts/health_check.py --api-url ${PACK_URL}
echo -e "${GREEN}✓ Health checks passed${NC}"

# Print deployment summary
echo -e "\n======================================================================"
echo -e "${GREEN}✅ Deployment complete!${NC}"
echo "======================================================================"
echo "Pack: ${PACK_NAME}"
echo "Version: ${PACK_VERSION}"
echo "Environment: ${ENVIRONMENT}"
echo "Namespace: ${NAMESPACE}"
echo "Replicas: ${REPLICAS}"
echo "======================================================================"

# Print access information
if [ "${ENVIRONMENT}" == "production" ]; then
    echo -e "\n${GREEN}Production Deployment:${NC}"
    echo "  API URL: https://pack027.greenlang.io"
    echo "  Docs: https://pack027.greenlang.io/docs"
    echo "  Metrics: https://pack027.greenlang.io/metrics"
else
    echo -e "\n${YELLOW}Development/Staging:${NC}"
    echo "  Forward port to access locally:"
    echo "  kubectl port-forward -n ${NAMESPACE} svc/pack027-service 8000:80"
    echo "  Then access at: http://localhost:8000"
fi

echo -e "\n${GREEN}Deployment successful!${NC}"
