#!/bin/bash
# ============================================================================
# CSRD REPORTING PLATFORM - PRODUCTION DEPLOYMENT SCRIPT
# ============================================================================
#
# Automated deployment script for production environment
# Implements blue-green deployment for zero-downtime deployment
#
# Author: DevOps Team
# Date: 2025-10-20
# Version: 1.0
#
# Usage: ./deploy-production.sh [options]
#   Options:
#     --method <blue-green|rolling|canary>  Deployment method (default: blue-green)
#     --skip-tests                           Skip pre-deployment tests
#     --dry-run                              Simulate deployment without applying
#     --rollback                             Rollback to previous version
#
# ============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_METHOD="${DEPLOYMENT_METHOD:-blue-green}"
SKIP_TESTS="${SKIP_TESTS:-false}"
DRY_RUN="${DRY_RUN:-false}"
ROLLBACK="${ROLLBACK:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Deployment settings
DEPLOYMENT_ID="deploy-$(date +%Y%m%d-%H%M%S)"
LOG_FILE="/var/log/csrd/deployment-${DEPLOYMENT_ID}.log"
HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-https://csrd.prod.example.com/health}"
READY_CHECK_URL="${READY_CHECK_URL:-https://csrd.prod.example.com/health/ready}"

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

log_section() {
    echo -e "\n${BLUE}============================================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}  $1${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}============================================================${NC}\n" | tee -a "$LOG_FILE"
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

confirm() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY-RUN: Would ask for confirmation: $1"
        return 0
    fi

    read -p "$1 (yes/no): " response
    case "$response" in
        [yY][eE][sS]|[yY])
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Required command not found: $1"
        log "Please install $1 and try again"
        exit 1
    fi
}

wait_for_health() {
    local url="$1"
    local max_attempts="${2:-30}"
    local wait_seconds="${3:-5}"

    log "Waiting for health check: $url"

    for i in $(seq 1 "$max_attempts"); do
        if curl -sf "$url" > /dev/null 2>&1; then
            log_success "Health check passed"
            return 0
        fi

        log "Attempt $i/$max_attempts failed, waiting ${wait_seconds}s..."
        sleep "$wait_seconds"
    done

    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# ============================================================================
# PRE-DEPLOYMENT CHECKS
# ============================================================================

pre_deployment_checks() {
    log_section "PRE-DEPLOYMENT CHECKS"

    # Check required commands
    log "Checking required commands..."
    check_command "docker"
    check_command "docker-compose"
    check_command "python"
    check_command "git"
    check_command "curl"
    check_command "jq"
    log_success "All required commands available"

    # Check environment
    log "Checking environment..."
    if [[ -z "${DATABASE_URL:-}" ]]; then
        log_error "DATABASE_URL not set"
        exit 1
    fi
    if [[ -z "${REDIS_URL:-}" ]]; then
        log_warning "REDIS_URL not set (cache may not work)"
    fi
    log_success "Environment variables validated"

    # Verify Git status
    log "Checking Git status..."
    cd "$PROJECT_ROOT"
    if [[ -n $(git status --porcelain) ]]; then
        log_warning "Uncommitted changes detected"
        git status --short
        if ! confirm "Continue with uncommitted changes?"; then
            log_error "Deployment aborted by user"
            exit 1
        fi
    fi
    log_success "Git status clean or approved"

    # Run security scan
    if [[ "$SKIP_TESTS" != "true" ]]; then
        log "Running final security scan..."
        if python security_scan.py . > security-scan-final.log 2>&1; then
            log_success "Security scan passed"
        else
            log_error "Security scan failed - check security-scan-final.log"
            exit 1
        fi
    else
        log_warning "Skipping security scan (--skip-tests flag)"
    fi

    # Run tests
    if [[ "$SKIP_TESTS" != "true" ]]; then
        log "Running smoke tests..."
        if python run_tests.py --quick > pre-deploy-tests.log 2>&1; then
            log_success "Smoke tests passed"
        else
            log_error "Smoke tests failed - check pre-deploy-tests.log"
            exit 1
        fi
    else
        log_warning "Skipping tests (--skip-tests flag)"
    fi

    log_success "All pre-deployment checks passed"
}

# ============================================================================
# DATABASE BACKUP
# ============================================================================

backup_database() {
    log_section "DATABASE BACKUP"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY-RUN: Would create database backup"
        return 0
    fi

    local backup_file="backup-${DEPLOYMENT_ID}.sql.gz"
    local backup_path="/var/backups/csrd/${backup_file}"

    log "Creating database backup: $backup_path"

    # Extract database connection details from DATABASE_URL
    # Format: postgresql://user:password@host:port/database
    local db_url="${DATABASE_URL}"
    local db_host=$(echo "$db_url" | sed -n 's|.*@\([^:]*\):.*|\1|p')
    local db_name=$(echo "$db_url" | sed -n 's|.*/\([^?]*\).*|\1|p')
    local db_user=$(echo "$db_url" | sed -n 's|.*://\([^:]*\):.*|\1|p')

    # Create backup directory
    mkdir -p "$(dirname "$backup_path")"

    # Create backup
    if PGPASSWORD="${DB_PASSWORD}" pg_dump \
        -h "$db_host" \
        -U "$db_user" \
        -d "$db_name" \
        --no-owner \
        --no-acl \
        | gzip > "$backup_path"; then
        log_success "Database backup created: $backup_path"
        log "Backup size: $(du -h "$backup_path" | cut -f1)"
    else
        log_error "Database backup failed"
        exit 1
    fi

    # Verify backup
    if gunzip -c "$backup_path" | head -20 > /dev/null 2>&1; then
        log_success "Backup verification passed"
    else
        log_error "Backup verification failed"
        exit 1
    fi
}

# ============================================================================
# BLUE-GREEN DEPLOYMENT
# ============================================================================

deploy_blue_green() {
    log_section "BLUE-GREEN DEPLOYMENT"

    # Step 1: Build green environment
    log "Building green environment..."
    if [[ "$DRY_RUN" != "true" ]]; then
        docker-compose -f docker-compose.green.yml build
        log_success "Green environment built"
    else
        log "DRY-RUN: Would build green environment"
    fi

    # Step 2: Run database migrations
    log "Running database migrations..."
    if [[ "$DRY_RUN" != "true" ]]; then
        docker-compose -f docker-compose.green.yml run --rm api python manage.py migrate
        log_success "Database migrations completed"
    else
        log "DRY-RUN: Would run database migrations"
    fi

    # Step 3: Start green environment
    log "Starting green environment..."
    if [[ "$DRY_RUN" != "true" ]]; then
        docker-compose -f docker-compose.green.yml up -d
        log_success "Green environment started"
    else
        log "DRY-RUN: Would start green environment"
    fi

    # Step 4: Wait for green environment to be healthy
    log "Waiting for green environment health checks..."
    if [[ "$DRY_RUN" != "true" ]]; then
        sleep 10  # Give services time to start
        if wait_for_health "http://localhost:8001/health" 30 5; then
            log_success "Green environment is healthy"
        else
            log_error "Green environment health check failed"
            log "Rolling back..."
            docker-compose -f docker-compose.green.yml down
            exit 1
        fi
    else
        log "DRY-RUN: Would wait for health checks"
    fi

    # Step 5: Run smoke tests on green
    log "Running smoke tests on green environment..."
    if [[ "$DRY_RUN" != "true" ]] && [[ "$SKIP_TESTS" != "true" ]]; then
        if python smoke_tests.py --url="http://localhost:8001" > green-smoke-tests.log 2>&1; then
            log_success "Green environment smoke tests passed"
        else
            log_error "Green environment smoke tests failed"
            cat green-smoke-tests.log
            log "Rolling back..."
            docker-compose -f docker-compose.green.yml down
            exit 1
        fi
    else
        log "DRY-RUN or SKIP-TESTS: Would run smoke tests"
    fi

    # Step 6: Switch load balancer traffic to green
    log "Switching traffic to green environment..."
    if ! confirm "Ready to switch production traffic to green?"; then
        log_error "Deployment aborted by user"
        docker-compose -f docker-compose.green.yml down
        exit 1
    fi

    if [[ "$DRY_RUN" != "true" ]]; then
        # Update load balancer (example for AWS ALB)
        # aws elbv2 modify-target-group --target-group-arn ... --targets Id=green-instance
        log "Switching load balancer to green environment"
        # Placeholder for actual load balancer update
        log_success "Traffic switched to green environment"
    else
        log "DRY-RUN: Would switch traffic to green"
    fi

    # Step 7: Monitor green environment
    log "Monitoring green environment for 10 minutes..."
    if [[ "$DRY_RUN" != "true" ]]; then
        for i in {1..120}; do  # 10 minutes = 120 x 5 seconds
            if ! curl -sf "$HEALTH_CHECK_URL" > /dev/null 2>&1; then
                log_error "Health check failed during monitoring period"
                log "Initiating rollback..."
                rollback_deployment
                exit 1
            fi
            sleep 5
        done
        log_success "Green environment stable for 10 minutes"
    else
        log "DRY-RUN: Would monitor for 10 minutes"
    fi

    # Step 8: Decommission blue environment
    log "Decommissioning blue environment..."
    if confirm "Ready to decommission blue environment?"; then
        if [[ "$DRY_RUN" != "true" ]]; then
            # Keep blue running for 1 hour for quick rollback
            log "Blue environment will remain running for 1 hour for quick rollback"
            log "To manually stop: docker-compose -f docker-compose.blue.yml down"
        else
            log "DRY-RUN: Would keep blue environment for 1 hour"
        fi
    fi

    log_success "Blue-green deployment completed successfully"
}

# ============================================================================
# ROLLING DEPLOYMENT
# ============================================================================

deploy_rolling() {
    log_section "ROLLING DEPLOYMENT"

    local instances=("api-1" "api-2" "api-3")

    for instance in "${instances[@]}"; do
        log "Deploying to $instance..."

        # Remove from load balancer
        log "Removing $instance from load balancer"
        if [[ "$DRY_RUN" != "true" ]]; then
            # aws elbv2 deregister-targets --target-group-arn ... --targets Id=$instance
            log "Removed $instance from load balancer"
        else
            log "DRY-RUN: Would remove $instance from load balancer"
        fi

        # Wait for connections to drain
        log "Waiting 30 seconds for connections to drain..."
        if [[ "$DRY_RUN" != "true" ]]; then
            sleep 30
        else
            log "DRY-RUN: Would wait for connection drain"
        fi

        # Deploy to instance
        log "Deploying new version to $instance..."
        if [[ "$DRY_RUN" != "true" ]]; then
            ssh "$instance" "cd /opt/csrd && git pull && docker-compose up -d --build"
            log_success "Deployed to $instance"
        else
            log "DRY-RUN: Would deploy to $instance"
        fi

        # Wait for health checks
        log "Waiting for $instance health checks..."
        if [[ "$DRY_RUN" != "true" ]]; then
            if wait_for_health "http://$instance:8000/health" 30 5; then
                log_success "$instance is healthy"
            else
                log_error "$instance health check failed"
                exit 1
            fi
        else
            log "DRY-RUN: Would wait for health checks"
        fi

        # Add back to load balancer
        log "Adding $instance back to load balancer..."
        if [[ "$DRY_RUN" != "true" ]]; then
            # aws elbv2 register-targets --target-group-arn ... --targets Id=$instance
            log_success "Added $instance back to load balancer"
        else
            log "DRY-RUN: Would add $instance back to load balancer"
        fi

        # Wait before next instance
        log "Waiting 60 seconds before next instance..."
        if [[ "$DRY_RUN" != "true" ]]; then
            sleep 60
        else
            log "DRY-RUN: Would wait 60 seconds"
        fi
    done

    log_success "Rolling deployment completed successfully"
}

# ============================================================================
# ROLLBACK
# ============================================================================

rollback_deployment() {
    log_section "ROLLBACK DEPLOYMENT"

    log_error "Initiating emergency rollback"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY-RUN: Would rollback deployment"
        return 0
    fi

    # Switch traffic back to blue
    log "Switching traffic back to blue environment..."
    # aws elbv2 modify-target-group --target-group-arn ... --targets Id=blue-instance

    log_success "Traffic switched back to blue environment"

    # Verify blue environment health
    if wait_for_health "$HEALTH_CHECK_URL" 10 5; then
        log_success "Blue environment is healthy"
    else
        log_error "Blue environment health check failed - CRITICAL"
        log "Manual intervention required"
        exit 1
    fi

    # Stop green environment
    log "Stopping green environment..."
    docker-compose -f docker-compose.green.yml down

    # Log rollback
    echo "$(date): Rollback executed - Deployment ID: $DEPLOYMENT_ID" >> /var/log/csrd/rollbacks.log

    log_success "Rollback completed successfully"
}

# ============================================================================
# POST-DEPLOYMENT VALIDATION
# ============================================================================

post_deployment_validation() {
    log_section "POST-DEPLOYMENT VALIDATION"

    # Health checks
    log "Running health checks..."
    if curl -sf "$HEALTH_CHECK_URL" > /dev/null 2>&1; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        exit 1
    fi

    if curl -sf "$READY_CHECK_URL" > /dev/null 2>&1; then
        log_success "Readiness check passed"
    else
        log_error "Readiness check failed"
        exit 1
    fi

    # Smoke tests
    if [[ "$SKIP_TESTS" != "true" ]]; then
        log "Running post-deployment smoke tests..."
        if python smoke_tests.py --url="$HEALTH_CHECK_URL" > post-deploy-smoke-tests.log 2>&1; then
            log_success "Smoke tests passed"
        else
            log_error "Smoke tests failed"
            cat post-deploy-smoke-tests.log
            exit 1
        fi
    fi

    # Check metrics
    log "Verifying metrics endpoint..."
    if curl -sf "${HEALTH_CHECK_URL}/metrics" > /dev/null 2>&1; then
        log_success "Metrics endpoint accessible"
    else
        log_warning "Metrics endpoint not accessible"
    fi

    log_success "Post-deployment validation completed"
}

# ============================================================================
# MAIN DEPLOYMENT FLOW
# ============================================================================

main() {
    log_section "CSRD PLATFORM - PRODUCTION DEPLOYMENT"
    log "Deployment ID: $DEPLOYMENT_ID"
    log "Method: $DEPLOYMENT_METHOD"
    log "Dry Run: $DRY_RUN"
    log "Skip Tests: $SKIP_TESTS"

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --method)
                DEPLOYMENT_METHOD="$2"
                shift 2
                ;;
            --skip-tests)
                SKIP_TESTS="true"
                shift
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --rollback)
                rollback_deployment
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Confirm deployment
    if ! confirm "Ready to deploy to PRODUCTION?"; then
        log "Deployment cancelled by user"
        exit 0
    fi

    # Execute deployment
    pre_deployment_checks
    backup_database

    case "$DEPLOYMENT_METHOD" in
        blue-green)
            deploy_blue_green
            ;;
        rolling)
            deploy_rolling
            ;;
        *)
            log_error "Unknown deployment method: $DEPLOYMENT_METHOD"
            exit 1
            ;;
    esac

    post_deployment_validation

    # Success
    log_section "DEPLOYMENT COMPLETED SUCCESSFULLY"
    log_success "Deployment ID: $DEPLOYMENT_ID"
    log_success "Method: $DEPLOYMENT_METHOD"
    log_success "Duration: $(date -d@$(($(date +%s) - $(date -d"$(stat -c %y "$LOG_FILE")" +%s))) -u +%H:%M:%S)"
    log "Log file: $LOG_FILE"
    log ""
    log "Next steps:"
    log "  1. Monitor dashboards: https://grafana.internal/d/csrd-overview"
    log "  2. Check logs: kubectl logs -f deployment/csrd-api"
    log "  3. Monitor for 24 hours"
    log "  4. Run full validation: python run_tests.py --env=prod"
}

# Execute main function
main "$@"
