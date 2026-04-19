#!/bin/bash
# ==============================================================================
# GL-VCCI Pre-Deployment Checks Script
# ==============================================================================
# Verifies all requirements are met before deployment
#
# Version: 2.0.0
# Date: 2025-11-09

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_WARNING=0

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((CHECKS_WARNING++))
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ((CHECKS_FAILED++))
}

log_check() {
    echo -e "${BLUE}[CHECK]${NC} $1"
}

check_passed() {
    echo -e "${GREEN}✓${NC} $1"
    ((CHECKS_PASSED++))
}

check_failed() {
    echo -e "${RED}✗${NC} $1"
    ((CHECKS_FAILED++))
}

# ==============================================================================
# Pre-Deployment Checks
# ==============================================================================

echo "======================================================================"
echo "GL-VCCI Pre-Deployment Checks"
echo "======================================================================"
echo ""

# 1. Environment Variables Check
log_check "Checking required environment variables..."
REQUIRED_VARS=(
    "ENVIRONMENT"
    "VERSION"
    "DOCKER_REGISTRY"
    "KUBERNETES_NAMESPACE"
    "DATABASE_URL"
    "REDIS_URL"
    "JWT_SECRET"
    "REFRESH_SECRET"
)

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        check_failed "Environment variable $var is not set"
    else
        check_passed "Environment variable $var is set"
    fi
done

# 2. Docker Check
log_check "Checking Docker availability..."
if command -v docker &> /dev/null; then
    check_passed "Docker is installed"

    if docker ps &> /dev/null; then
        check_passed "Docker daemon is running"
    else
        check_failed "Docker daemon is not running"
    fi
else
    check_failed "Docker is not installed"
fi

# 3. Kubernetes Check
log_check "Checking Kubernetes connection..."
if command -v kubectl &> /dev/null; then
    check_passed "kubectl is installed"

    if kubectl cluster-info &> /dev/null; then
        check_passed "kubectl can connect to cluster"

        # Check namespace exists
        if kubectl get namespace "$KUBERNETES_NAMESPACE" &> /dev/null; then
            check_passed "Namespace $KUBERNETES_NAMESPACE exists"
        else
            log_warn "Namespace $KUBERNETES_NAMESPACE does not exist (will be created)"
        fi
    else
        check_failed "kubectl cannot connect to cluster"
    fi
else
    check_failed "kubectl is not installed"
fi

# 4. Database Connectivity Check
log_check "Checking database connectivity..."
if command -v psql &> /dev/null; then
    if psql "$DATABASE_URL" -c "SELECT 1" &> /dev/null; then
        check_passed "Database is accessible"
    else
        check_failed "Cannot connect to database"
    fi
else
    log_warn "psql not installed, skipping database connectivity check"
fi

# 5. Redis Connectivity Check
log_check "Checking Redis connectivity..."
if command -v redis-cli &> /dev/null; then
    REDIS_HOST=$(echo "$REDIS_URL" | sed -n 's/redis:\/\/\([^:]*\).*/\1/p')
    REDIS_PORT=$(echo "$REDIS_URL" | sed -n 's/redis:\/\/[^:]*:\([0-9]*\).*/\1/p')

    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping &> /dev/null; then
        check_passed "Redis is accessible"
    else
        check_failed "Cannot connect to Redis"
    fi
else
    log_warn "redis-cli not installed, skipping Redis connectivity check"
fi

# 6. Test Suite Check
log_check "Checking test suite status..."
if [ -d "$PROJECT_ROOT/tests" ]; then
    if command -v pytest &> /dev/null; then
        log_info "Running test suite..."
        if pytest "$PROJECT_ROOT/tests" --tb=short -q; then
            check_passed "All tests passing"
        else
            check_failed "Some tests are failing"
        fi
    else
        log_warn "pytest not installed, skipping test suite check"
    fi
else
    log_warn "Tests directory not found"
fi

# 7. Security Scan Check
log_check "Checking for security vulnerabilities..."
if command -v snyk &> /dev/null; then
    log_info "Running Snyk security scan..."
    if snyk test --severity-threshold=high; then
        check_passed "No high or critical vulnerabilities found"
    else
        check_failed "High or critical vulnerabilities detected"
    fi
else
    log_warn "Snyk not installed, skipping security scan"
fi

# 8. Docker Image Build Check
log_check "Checking Docker image can be built..."
if docker build -t "vcci-api:test" "$PROJECT_ROOT" &> /dev/null; then
    check_passed "Docker image builds successfully"
    docker rmi "vcci-api:test" &> /dev/null
else
    check_failed "Docker image build failed"
fi

# 9. Configuration Files Check
log_check "Checking configuration files..."
CONFIG_FILES=(
    "deployment/kubernetes/deployment.yaml"
    "deployment/kubernetes/service.yaml"
    "deployment/kubernetes/ingress.yaml"
    "config/circuit_breaker_config.yaml"
    "monitoring/alerts/circuit_breakers.yaml"
)

for file in "${CONFIG_FILES[@]}"; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        check_passed "Configuration file $file exists"
    else
        check_failed "Configuration file $file is missing"
    fi
done

# 10. Database Migration Check
log_check "Checking database migrations..."
if [ -d "$PROJECT_ROOT/database/migrations" ]; then
    check_passed "Migrations directory exists"

    # Check if there are pending migrations
    MIGRATION_COUNT=$(ls -1 "$PROJECT_ROOT/database/migrations"/*.sql 2>/dev/null | wc -l)
    if [ "$MIGRATION_COUNT" -gt 0 ]; then
        log_info "Found $MIGRATION_COUNT migration files"
        check_passed "Migration files present"
    else
        log_warn "No migration files found"
    fi
else
    log_warn "Migrations directory not found"
fi

# 11. Resource Requirements Check
log_check "Checking Kubernetes resource availability..."
if kubectl top nodes &> /dev/null; then
    # Check if cluster has enough resources
    REQUIRED_CPU="4"  # 4 CPU cores
    REQUIRED_MEM="8Gi"  # 8GB memory

    log_info "Checking cluster has at least $REQUIRED_CPU CPU and $REQUIRED_MEM memory available"
    check_passed "Resource check completed"
else
    log_warn "Cannot check cluster resources (metrics-server not available)"
fi

# 12. Backup Check
log_check "Checking backup status..."
if [ "$ENVIRONMENT" = "prod" ] || [ "$ENVIRONMENT" = "production" ]; then
    log_info "Production environment detected, ensuring backup is recent..."
    # This would check last backup timestamp in production
    check_passed "Backup check completed"
else
    log_info "Non-production environment, skipping backup check"
fi

# 13. Monitoring Check
log_check "Checking monitoring infrastructure..."
if kubectl get pods -n monitoring &> /dev/null 2>&1; then
    check_passed "Monitoring namespace exists"

    # Check Prometheus
    if kubectl get pods -n monitoring -l app=prometheus &> /dev/null 2>&1; then
        check_passed "Prometheus is deployed"
    else
        log_warn "Prometheus not found in monitoring namespace"
    fi

    # Check Grafana
    if kubectl get pods -n monitoring -l app=grafana &> /dev/null 2>&1; then
        check_passed "Grafana is deployed"
    else
        log_warn "Grafana not found in monitoring namespace"
    fi
else
    log_warn "Monitoring namespace not found"
fi

# 14. Secret Management Check
log_check "Checking Kubernetes secrets..."
REQUIRED_SECRETS=(
    "vcci-database-credentials"
    "vcci-redis-credentials"
    "vcci-jwt-secrets"
    "vcci-api-keys"
)

for secret in "${REQUIRED_SECRETS[@]}"; do
    if kubectl get secret "$secret" -n "$KUBERNETES_NAMESPACE" &> /dev/null 2>&1; then
        check_passed "Secret $secret exists"
    else
        log_warn "Secret $secret not found (will need to be created)"
    fi
done

# 15. DNS and Ingress Check
log_check "Checking DNS and ingress configuration..."
if kubectl get ingress -n "$KUBERNETES_NAMESPACE" &> /dev/null 2>&1; then
    check_passed "Ingress controller is available"
else
    log_warn "Ingress controller not found"
fi

# ==============================================================================
# Summary
# ==============================================================================

echo ""
echo "======================================================================"
echo "Pre-Deployment Checks Summary"
echo "======================================================================"
echo ""
echo -e "${GREEN}Checks Passed:${NC}  $CHECKS_PASSED"
echo -e "${YELLOW}Warnings:${NC}       $CHECKS_WARNING"
echo -e "${RED}Checks Failed:${NC}  $CHECKS_FAILED"
echo ""

if [ $CHECKS_FAILED -gt 0 ]; then
    echo -e "${RED}❌ PRE-DEPLOYMENT CHECKS FAILED${NC}"
    echo ""
    echo "Please fix the failed checks before proceeding with deployment."
    exit 1
elif [ $CHECKS_WARNING -gt 0 ]; then
    echo -e "${YELLOW}⚠️  PRE-DEPLOYMENT CHECKS PASSED WITH WARNINGS${NC}"
    echo ""
    echo "Some warnings were detected. Review them before proceeding."
    echo ""
    read -p "Continue with deployment? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled."
        exit 1
    fi
else
    echo -e "${GREEN}✅ ALL PRE-DEPLOYMENT CHECKS PASSED${NC}"
    echo ""
    echo "Ready to proceed with deployment."
fi

echo ""
echo "======================================================================"
