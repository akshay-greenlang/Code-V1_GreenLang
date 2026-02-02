#!/bin/bash
# ==============================================================================
# GL-VCCI Docker Image Build Script
# ==============================================================================

set -e

VERSION="$1"
REGISTRY="${DOCKER_REGISTRY:-YOUR_REGISTRY}"

if [ -z "$VERSION" ]; then
    echo "[ERROR] Version is required"
    echo "Usage: $0 <version>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "[INFO] Building Docker images for version: $VERSION"
echo "[INFO] Registry: $REGISTRY"

# Build backend image
echo "[INFO] Building backend image..."
docker build \
    -t "${REGISTRY}/vcci-backend:${VERSION}" \
    -t "${REGISTRY}/vcci-backend:latest" \
    -f "$PROJECT_ROOT/backend/Dockerfile" \
    "$PROJECT_ROOT"

# Build worker image
echo "[INFO] Building worker image..."
docker build \
    -t "${REGISTRY}/vcci-worker:${VERSION}" \
    -t "${REGISTRY}/vcci-worker:latest" \
    -f "$PROJECT_ROOT/worker/Dockerfile" \
    "$PROJECT_ROOT"

# Build frontend image
echo "[INFO] Building frontend image..."
docker build \
    -t "${REGISTRY}/vcci-frontend:${VERSION}" \
    -t "${REGISTRY}/vcci-frontend:latest" \
    -f "$PROJECT_ROOT/frontend/Dockerfile" \
    "$PROJECT_ROOT/frontend"

# Push images
echo "[INFO] Pushing images to registry..."
docker push "${REGISTRY}/vcci-backend:${VERSION}"
docker push "${REGISTRY}/vcci-backend:latest"
docker push "${REGISTRY}/vcci-worker:${VERSION}"
docker push "${REGISTRY}/vcci-worker:latest"
docker push "${REGISTRY}/vcci-frontend:${VERSION}"
docker push "${REGISTRY}/vcci-frontend:latest"

echo "[INFO] Image build and push completed successfully!"
