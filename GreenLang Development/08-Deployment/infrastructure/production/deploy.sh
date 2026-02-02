#!/bin/bash
# GreenLang Climate OS - Production Deployment Script
# This script deploys the complete GreenLang infrastructure

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="greenlang-production"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GreenLang Climate OS - Production Deploy${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"

    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}kubectl not found. Please install kubectl.${NC}"
        exit 1
    fi

    if ! command -v helm &> /dev/null; then
        echo -e "${RED}helm not found. Please install helm.${NC}"
        exit 1
    fi

    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}Cannot connect to Kubernetes cluster.${NC}"
        exit 1
    fi

    echo -e "${GREEN}Prerequisites check passed!${NC}"
}

# Create namespace
create_namespace() {
    echo -e "${YELLOW}Creating namespace...${NC}"
    kubectl apply -f "${SCRIPT_DIR}/namespace.yaml"
    echo -e "${GREEN}Namespace created!${NC}"
}

# Apply RBAC
apply_rbac() {
    echo -e "${YELLOW}Applying RBAC configuration...${NC}"
    kubectl apply -f "${SCRIPT_DIR}/rbac.yaml"
    echo -e "${GREEN}RBAC applied!${NC}"
}

# Apply network policies
apply_network_policies() {
    echo -e "${YELLOW}Applying network policies...${NC}"
    kubectl apply -f "${SCRIPT_DIR}/network-policies.yaml"
    echo -e "${GREEN}Network policies applied!${NC}"
}

# Deploy PostgreSQL with TimescaleDB
deploy_postgresql() {
    echo -e "${YELLOW}Deploying PostgreSQL + TimescaleDB...${NC}"

    # Create secrets first (should be replaced with external-secrets in production)
    kubectl apply -f "${SCRIPT_DIR}/postgresql/secrets.yaml" 2>/dev/null || true

    # Apply configmaps
    kubectl apply -f "${SCRIPT_DIR}/postgresql/configmap.yaml"

    # Deploy statefulset
    kubectl apply -f "${SCRIPT_DIR}/postgresql/statefulset.yaml"

    # Wait for PostgreSQL to be ready
    echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"
    kubectl rollout status statefulset/postgresql-timescaledb -n ${NAMESPACE} --timeout=300s

    echo -e "${GREEN}PostgreSQL deployed!${NC}"
}

# Deploy Redis Sentinel
deploy_redis() {
    echo -e "${YELLOW}Deploying Redis Sentinel cluster...${NC}"

    # Apply configmaps
    kubectl apply -f "${SCRIPT_DIR}/redis/configmap.yaml"

    # Deploy statefulset
    kubectl apply -f "${SCRIPT_DIR}/redis/statefulset.yaml"

    # Wait for Redis to be ready
    echo -e "${YELLOW}Waiting for Redis to be ready...${NC}"
    kubectl rollout status statefulset/redis-sentinel -n ${NAMESPACE} --timeout=300s

    echo -e "${GREEN}Redis deployed!${NC}"
}

# Deploy Agent Factory
deploy_agent_factory() {
    echo -e "${YELLOW}Deploying Agent Factory v1.0...${NC}"

    # Apply configmaps
    kubectl apply -f "${SCRIPT_DIR}/agent-factory/configmap.yaml"

    # Deploy
    kubectl apply -f "${SCRIPT_DIR}/agent-factory/deployment.yaml"

    # Wait for deployment
    echo -e "${YELLOW}Waiting for Agent Factory to be ready...${NC}"
    kubectl rollout status deployment/agent-factory -n ${NAMESPACE} --timeout=300s

    echo -e "${GREEN}Agent Factory deployed!${NC}"
}

# Deploy monitoring stack
deploy_monitoring() {
    echo -e "${YELLOW}Deploying monitoring stack...${NC}"

    # Apply Prometheus config
    kubectl apply -f "${SCRIPT_DIR}/monitoring/prometheus.yaml"

    # Deploy Prometheus using helm (if not already deployed)
    if ! helm status prometheus -n ${NAMESPACE} &> /dev/null; then
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
        helm repo update
        helm install prometheus prometheus-community/kube-prometheus-stack \
            --namespace ${NAMESPACE} \
            --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
            --set grafana.enabled=true \
            --set alertmanager.enabled=true
    fi

    echo -e "${GREEN}Monitoring deployed!${NC}"
}

# Apply ingress
apply_ingress() {
    echo -e "${YELLOW}Applying ingress configuration...${NC}"
    kubectl apply -f "${SCRIPT_DIR}/ingress.yaml"
    echo -e "${GREEN}Ingress applied!${NC}"
}

# Verify deployment
verify_deployment() {
    echo -e "${YELLOW}Verifying deployment...${NC}"

    echo ""
    echo "Pods:"
    kubectl get pods -n ${NAMESPACE}

    echo ""
    echo "Services:"
    kubectl get services -n ${NAMESPACE}

    echo ""
    echo "StatefulSets:"
    kubectl get statefulsets -n ${NAMESPACE}

    echo ""
    echo "Deployments:"
    kubectl get deployments -n ${NAMESPACE}
}

# Main deployment sequence
main() {
    echo -e "${YELLOW}Starting GreenLang production deployment...${NC}"
    echo ""

    check_prerequisites
    echo ""

    # INFRA-001: Kubernetes Production Cluster
    create_namespace
    apply_rbac
    apply_network_policies
    echo ""

    # INFRA-002: PostgreSQL + TimescaleDB
    deploy_postgresql
    echo ""

    # INFRA-003: Redis Sentinel Cluster
    deploy_redis
    echo ""

    # INFRA-004: Agent Factory v1.0
    deploy_agent_factory
    echo ""

    # INFRA-008: Monitoring (Prometheus + Grafana)
    deploy_monitoring
    echo ""

    # Ingress
    apply_ingress
    echo ""

    # Verification
    verify_deployment
    echo ""

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Configure external secrets (replace placeholder values)"
    echo "2. Configure DNS for ingress hostnames"
    echo "3. Verify TLS certificates"
    echo "4. Run integration tests"
    echo ""
    echo "Access points:"
    echo "- API: https://api.greenlang.io"
    echo "- Agents: https://agents.greenlang.io"
    echo "- App: https://app.greenlang.io"
    echo "- Monitoring: https://monitoring.greenlang.io"
}

# Run main function
main "$@"
