#!/bin/bash
# ============================================================================
# GreenLang Platform - Quick Deployment Script
# ============================================================================
# Usage: ./deploy.sh [start|stop|restart|status|logs]
# ============================================================================

set -e

COMPOSE_FILE="docker-compose-unified.yml"
PROJECT_NAME="greenlang-platform"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo ""
    echo "============================================================================"
    echo "$1"
    echo "============================================================================"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}!${NC} $1"
}

check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_success "Docker installed: $(docker --version)"

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    print_success "Docker Compose installed: $(docker-compose --version)"

    # Check .env file
    if [ ! -f .env ]; then
        print_warning ".env file not found"
        echo "Creating .env from .env.example..."
        cp .env.example .env
        print_warning "Please edit .env with your configuration before deploying"
        exit 1
    fi
    print_success ".env file found"

    # Check available ports
    for port in 8000 8001 8002 3000 5432 6379 5672 8080 9090 15672; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            print_warning "Port $port is already in use"
        fi
    done
}

start_infrastructure() {
    print_header "Starting Shared Infrastructure"

    docker-compose -f $COMPOSE_FILE up -d postgres redis rabbitmq weaviate

    echo "Waiting for infrastructure to be healthy (30 seconds)..."
    sleep 30

    # Check health
    docker-compose -f $COMPOSE_FILE ps postgres redis rabbitmq weaviate

    print_success "Infrastructure started"
}

start_applications() {
    print_header "Starting Applications"

    docker-compose -f $COMPOSE_FILE up -d cbam-api csrd-web vcci-backend vcci-worker

    echo "Waiting for applications to start (30 seconds)..."
    sleep 30

    print_success "Applications started"
}

start_monitoring() {
    print_header "Starting Monitoring Stack"

    docker-compose -f $COMPOSE_FILE up -d prometheus grafana

    print_success "Monitoring stack started"
}

show_status() {
    print_header "Platform Status"

    docker-compose -f $COMPOSE_FILE ps

    echo ""
    echo "Access Points:"
    echo "  CBAM API:       http://localhost:8001/health"
    echo "  CSRD Web:       http://localhost:8002/health"
    echo "  VCCI Backend:   http://localhost:8000/health/live"
    echo "  Grafana:        http://localhost:3000 (admin/greenlang2024)"
    echo "  Prometheus:     http://localhost:9090"
    echo "  RabbitMQ UI:    http://localhost:15672 (greenlang/greenlang_rabbit_2024)"
    echo ""
}

health_check() {
    print_header "Health Checks"

    # CBAM
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        print_success "CBAM API is healthy"
    else
        print_error "CBAM API is not responding"
    fi

    # CSRD
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        print_success "CSRD Web is healthy"
    else
        print_error "CSRD Web is not responding"
    fi

    # VCCI
    if curl -s http://localhost:8000/health/live > /dev/null 2>&1; then
        print_success "VCCI Backend is healthy"
    else
        print_error "VCCI Backend is not responding"
    fi

    # Prometheus
    if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
        print_success "Prometheus is healthy"
    else
        print_error "Prometheus is not responding"
    fi

    # Grafana
    if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
        print_success "Grafana is healthy"
    else
        print_error "Grafana is not responding"
    fi
}

# Main commands
case "${1:-start}" in
    start)
        print_header "GreenLang Platform Deployment"
        check_prerequisites
        start_infrastructure
        start_applications
        start_monitoring
        show_status
        echo ""
        print_success "Platform deployment complete!"
        echo ""
        echo "Run './deploy.sh health' to check service health"
        echo "Run './deploy.sh logs' to view logs"
        ;;

    stop)
        print_header "Stopping GreenLang Platform"
        docker-compose -f $COMPOSE_FILE down
        print_success "Platform stopped"
        ;;

    restart)
        print_header "Restarting GreenLang Platform"
        docker-compose -f $COMPOSE_FILE restart
        print_success "Platform restarted"
        ;;

    status)
        show_status
        ;;

    health)
        health_check
        ;;

    logs)
        docker-compose -f $COMPOSE_FILE logs -f
        ;;

    build)
        print_header "Building Application Images"

        # CBAM
        echo "Building CBAM..."
        docker build -t greenlang/cbam-app:latest ../GL-CBAM-APP/CBAM-Importer-Copilot/

        # CSRD
        echo "Building CSRD..."
        docker build -t greenlang/csrd-app:latest ../GL-CSRD-APP/CSRD-Reporting-Platform/

        # VCCI
        echo "Building VCCI Backend..."
        docker build -t greenlang/vcci-backend:latest -f ../GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/backend/Dockerfile ../GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/

        echo "Building VCCI Worker..."
        docker build -t greenlang/vcci-worker:latest -f ../GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/worker/Dockerfile ../GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/

        print_success "All images built"
        ;;

    clean)
        print_header "Cleaning Up"
        print_warning "This will remove all containers and volumes (DATA WILL BE LOST)"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            docker-compose -f $COMPOSE_FILE down -v
            print_success "Cleanup complete"
        else
            echo "Cancelled"
        fi
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|status|health|logs|build|clean}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the entire platform (default)"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  status  - Show service status"
        echo "  health  - Run health checks"
        echo "  logs    - Show and follow logs"
        echo "  build   - Build application images"
        echo "  clean   - Remove all containers and volumes"
        exit 1
        ;;
esac
