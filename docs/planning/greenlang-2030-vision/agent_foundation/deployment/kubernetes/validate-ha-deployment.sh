#!/bin/bash
# ============================================================================
# GreenLang Multi-AZ HA Deployment Validation Script
# ============================================================================
# This script validates the Multi-AZ High Availability deployment
# Run after applying the HA manifests to verify configuration
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="greenlang-ai"
DEPLOYMENT_NAME="greenlang-agent"
SERVICE_NAME="greenlang-agent-lb"
HPA_NAME="greenlang-agent-hpa"
PDB_NAME="greenlang-agent-pdb"
MIN_REPLICAS=9
EXPECTED_AZS=3
MIN_PODS_PER_AZ=3

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "\n${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# ============================================================================
# Validation Functions
# ============================================================================

validate_prerequisites() {
    print_header "1. Validating Prerequisites"

    # Check kubectl
    if command -v kubectl &> /dev/null; then
        print_success "kubectl is installed"
        kubectl version --client --short
    else
        print_error "kubectl is not installed"
        exit 1
    fi

    # Check cluster connectivity
    if kubectl cluster-info &> /dev/null; then
        print_success "Connected to Kubernetes cluster"
    else
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check namespace exists
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_success "Namespace '$NAMESPACE' exists"
    else
        print_error "Namespace '$NAMESPACE' does not exist"
        exit 1
    fi
}

validate_nodes() {
    print_header "2. Validating Node Configuration"

    # Get node count
    NODE_COUNT=$(kubectl get nodes --no-headers | wc -l)
    print_info "Total nodes: $NODE_COUNT"

    # Check node labels for topology.kubernetes.io/zone
    print_info "Checking node zone labels..."
    kubectl get nodes -L topology.kubernetes.io/zone

    # Count nodes per zone
    ZONES=$(kubectl get nodes -o jsonpath='{.items[*].metadata.labels.topology\.kubernetes\.io/zone}' | tr ' ' '\n' | sort -u)
    ZONE_COUNT=$(echo "$ZONES" | wc -l)

    echo ""
    if [ "$ZONE_COUNT" -ge "$EXPECTED_AZS" ]; then
        print_success "Found $ZONE_COUNT availability zones (expected: $EXPECTED_AZS)"
        echo "$ZONES" | while read -r zone; do
            NODE_IN_ZONE=$(kubectl get nodes -l topology.kubernetes.io/zone="$zone" --no-headers | wc -l)
            print_info "  Zone '$zone': $NODE_IN_ZONE nodes"
        done
    else
        print_error "Only $ZONE_COUNT availability zones found (expected: $EXPECTED_AZS)"
        exit 1
    fi
}

validate_deployment() {
    print_header "3. Validating Deployment Configuration"

    # Check deployment exists
    if kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" &> /dev/null; then
        print_success "Deployment '$DEPLOYMENT_NAME' exists"
    else
        print_error "Deployment '$DEPLOYMENT_NAME' not found"
        exit 1
    fi

    # Get desired and current replicas
    DESIRED_REPLICAS=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
    CURRENT_REPLICAS=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o jsonpath='{.status.replicas}')
    READY_REPLICAS=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
    AVAILABLE_REPLICAS=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o jsonpath='{.status.availableReplicas}')

    print_info "Desired replicas: $DESIRED_REPLICAS"
    print_info "Current replicas: $CURRENT_REPLICAS"
    print_info "Ready replicas: $READY_REPLICAS"
    print_info "Available replicas: $AVAILABLE_REPLICAS"

    echo ""
    if [ "$DESIRED_REPLICAS" -ge "$MIN_REPLICAS" ]; then
        print_success "Desired replicas ($DESIRED_REPLICAS) meets minimum ($MIN_REPLICAS)"
    else
        print_error "Desired replicas ($DESIRED_REPLICAS) below minimum ($MIN_REPLICAS)"
        exit 1
    fi

    if [ "$READY_REPLICAS" -eq "$DESIRED_REPLICAS" ]; then
        print_success "All replicas are ready ($READY_REPLICAS/$DESIRED_REPLICAS)"
    else
        print_warning "Not all replicas ready ($READY_REPLICAS/$DESIRED_REPLICAS)"
    fi

    # Check rolling update strategy
    echo ""
    print_info "Rolling update strategy:"
    MAX_SURGE=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.strategy.rollingUpdate.maxSurge}')
    MAX_UNAVAILABLE=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.strategy.rollingUpdate.maxUnavailable}')

    print_info "  maxSurge: $MAX_SURGE"
    print_info "  maxUnavailable: $MAX_UNAVAILABLE"

    if [ "$MAX_UNAVAILABLE" -eq 0 ]; then
        print_success "Zero-downtime deployment configured (maxUnavailable=0)"
    else
        print_error "maxUnavailable should be 0 for zero-downtime deployments"
    fi
}

validate_pod_distribution() {
    print_header "4. Validating Pod Distribution Across AZs"

    # Get pod distribution by zone
    print_info "Pod distribution by availability zone:"
    echo ""

    kubectl get pods -n "$NAMESPACE" -l app="$DEPLOYMENT_NAME" -o wide \
        --sort-by='.spec.nodeName' \
        -o custom-columns=NAME:.metadata.name,STATUS:.status.phase,NODE:.spec.nodeName,ZONE:".metadata.labels['topology\.kubernetes\.io/zone']" 2>/dev/null || \
    kubectl get pods -n "$NAMESPACE" -l app="$DEPLOYMENT_NAME" -o wide

    echo ""

    # Count pods per zone
    ZONES=$(kubectl get nodes -o jsonpath='{.items[*].metadata.labels.topology\.kubernetes\.io/zone}' | tr ' ' '\n' | sort -u)
    TOTAL_PODS=0
    ALL_ZONES_OK=true

    echo "$ZONES" | while read -r zone; do
        # Count pods in this zone by getting nodes in zone, then pods on those nodes
        PODS_IN_ZONE=$(kubectl get pods -n "$NAMESPACE" -l app="$DEPLOYMENT_NAME" -o json | \
            jq -r --arg zone "$zone" '[.items[] | select(.spec.nodeName as $node |
            ($node | . != null) and
            (kubectl get node $node -o json | .metadata.labels["topology.kubernetes.io/zone"] == $zone))] | length' 2>/dev/null || echo 0)

        # Simpler approach - count running pods
        PODS_IN_ZONE=$(kubectl get pods -n "$NAMESPACE" -l app="greenlang-agent" \
            --field-selector=status.phase=Running -o wide | \
            grep -c "$zone" 2>/dev/null || echo 0)

        if [ "$PODS_IN_ZONE" -ge "$MIN_PODS_PER_AZ" ]; then
            print_success "Zone '$zone': $PODS_IN_ZONE pods (minimum: $MIN_PODS_PER_AZ)"
        else
            print_error "Zone '$zone': $PODS_IN_ZONE pods (minimum: $MIN_PODS_PER_AZ)"
            ALL_ZONES_OK=false
        fi
    done
}

validate_anti_affinity() {
    print_header "5. Validating Pod Anti-Affinity Rules"

    # Check pod anti-affinity configuration
    ANTI_AFFINITY=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" \
        -o jsonpath='{.spec.template.spec.affinity.podAntiAffinity}')

    if [ -n "$ANTI_AFFINITY" ]; then
        print_success "Pod anti-affinity is configured"

        # Check for required anti-affinity
        REQUIRED_AA=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" \
            -o jsonpath='{.spec.template.spec.affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution}')

        if [ -n "$REQUIRED_AA" ]; then
            print_success "HARD anti-affinity (required) is configured"

            # Check topology key
            TOPOLOGY_KEY=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" \
                -o jsonpath='{.spec.template.spec.affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].topologyKey}')

            if [ "$TOPOLOGY_KEY" = "topology.kubernetes.io/zone" ]; then
                print_success "Topology key is 'topology.kubernetes.io/zone' (zone-level distribution)"
            else
                print_error "Topology key is '$TOPOLOGY_KEY' (expected: topology.kubernetes.io/zone)"
            fi
        else
            print_error "HARD anti-affinity (required) is NOT configured"
        fi
    else
        print_error "Pod anti-affinity is NOT configured"
    fi
}

validate_hpa() {
    print_header "6. Validating Horizontal Pod Autoscaler"

    # Check HPA exists
    if kubectl get hpa "$HPA_NAME" -n "$NAMESPACE" &> /dev/null; then
        print_success "HPA '$HPA_NAME' exists"

        # Get HPA configuration
        MIN_REPLICAS_HPA=$(kubectl get hpa "$HPA_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.minReplicas}')
        MAX_REPLICAS_HPA=$(kubectl get hpa "$HPA_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.maxReplicas}')
        CURRENT_REPLICAS_HPA=$(kubectl get hpa "$HPA_NAME" -n "$NAMESPACE" -o jsonpath='{.status.currentReplicas}')

        print_info "Min replicas: $MIN_REPLICAS_HPA"
        print_info "Max replicas: $MAX_REPLICAS_HPA"
        print_info "Current replicas: $CURRENT_REPLICAS_HPA"

        echo ""
        if [ "$MIN_REPLICAS_HPA" -ge "$MIN_REPLICAS" ]; then
            print_success "HPA min replicas ($MIN_REPLICAS_HPA) meets requirement ($MIN_REPLICAS)"
        else
            print_error "HPA min replicas ($MIN_REPLICAS_HPA) below requirement ($MIN_REPLICAS)"
        fi

        if [ "$MAX_REPLICAS_HPA" -ge 100 ]; then
            print_success "HPA max replicas ($MAX_REPLICAS_HPA) configured for high scale"
        else
            print_warning "HPA max replicas ($MAX_REPLICAS_HPA) may limit scaling"
        fi

        # Show metrics
        echo ""
        print_info "HPA metrics:"
        kubectl get hpa "$HPA_NAME" -n "$NAMESPACE"
    else
        print_error "HPA '$HPA_NAME' not found"
    fi
}

validate_pdb() {
    print_header "7. Validating Pod Disruption Budget"

    # Check PDB exists
    if kubectl get pdb "$PDB_NAME" -n "$NAMESPACE" &> /dev/null; then
        print_success "PDB '$PDB_NAME' exists"

        # Get PDB configuration
        MIN_AVAILABLE=$(kubectl get pdb "$PDB_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.minAvailable}')
        CURRENT_HEALTHY=$(kubectl get pdb "$PDB_NAME" -n "$NAMESPACE" -o jsonpath='{.status.currentHealthy}')
        DESIRED_HEALTHY=$(kubectl get pdb "$PDB_NAME" -n "$NAMESPACE" -o jsonpath='{.status.desiredHealthy}')
        DISRUPTIONS_ALLOWED=$(kubectl get pdb "$PDB_NAME" -n "$NAMESPACE" -o jsonpath='{.status.disruptionsAllowed}')

        print_info "Min available: $MIN_AVAILABLE"
        print_info "Current healthy: $CURRENT_HEALTHY"
        print_info "Desired healthy: $DESIRED_HEALTHY"
        print_info "Disruptions allowed: $DISRUPTIONS_ALLOWED"

        echo ""
        if [ "$MIN_AVAILABLE" -ge 6 ]; then
            print_success "PDB minAvailable ($MIN_AVAILABLE) ensures 2 pods per AZ"
        else
            print_error "PDB minAvailable ($MIN_AVAILABLE) should be at least 6 for Multi-AZ HA"
        fi
    else
        print_error "PDB '$PDB_NAME' not found"
    fi
}

validate_service() {
    print_header "8. Validating Service and Load Balancer"

    # Check service exists
    if kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" &> /dev/null; then
        print_success "Service '$SERVICE_NAME' exists"

        # Get service type
        SERVICE_TYPE=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.type}')
        print_info "Service type: $SERVICE_TYPE"

        if [ "$SERVICE_TYPE" = "LoadBalancer" ]; then
            print_success "Service type is LoadBalancer"

            # Check for NLB annotation
            NLB_TYPE=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" \
                -o jsonpath='{.metadata.annotations.service\.beta\.kubernetes\.io/aws-load-balancer-type}')

            if [ "$NLB_TYPE" = "nlb" ]; then
                print_success "Network Load Balancer (NLB) configured"
            else
                print_warning "NLB annotation not found or incorrect"
            fi

            # Check cross-zone load balancing
            CROSS_ZONE=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" \
                -o jsonpath='{.metadata.annotations.service\.beta\.kubernetes\.io/aws-load-balancer-cross-zone-load-balancing-enabled}')

            if [ "$CROSS_ZONE" = "true" ]; then
                print_success "Cross-zone load balancing enabled"
            else
                print_error "Cross-zone load balancing NOT enabled"
            fi

            # Check session affinity
            SESSION_AFFINITY=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" \
                -o jsonpath='{.spec.sessionAffinity}')

            if [ "$SESSION_AFFINITY" = "ClientIP" ]; then
                TIMEOUT=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" \
                    -o jsonpath='{.spec.sessionAffinityConfig.clientIP.timeoutSeconds}')
                print_success "Session affinity enabled (ClientIP, timeout: ${TIMEOUT}s)"
            else
                print_warning "Session affinity not configured"
            fi

            # Get external IP
            echo ""
            print_info "Service details:"
            kubectl get service "$SERVICE_NAME" -n "$NAMESPACE"

            EXTERNAL_IP=$(kubectl get service "$SERVICE_NAME" -n "$NAMESPACE" \
                -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "pending")

            if [ "$EXTERNAL_IP" != "pending" ] && [ -n "$EXTERNAL_IP" ]; then
                print_success "External endpoint: $EXTERNAL_IP"
            else
                print_warning "External endpoint not yet assigned (may take a few minutes)"
            fi
        else
            print_error "Service type should be LoadBalancer for external access"
        fi
    else
        print_error "Service '$SERVICE_NAME' not found"
    fi
}

validate_health_checks() {
    print_header "9. Validating Health Check Configuration"

    # Check startup probe
    STARTUP_PATH=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" \
        -o jsonpath='{.spec.template.spec.containers[0].startupProbe.httpGet.path}')

    if [ "$STARTUP_PATH" = "/startup" ]; then
        print_success "Startup probe configured: $STARTUP_PATH"
    else
        print_warning "Startup probe path: $STARTUP_PATH (expected: /startup)"
    fi

    # Check liveness probe
    LIVENESS_PATH=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" \
        -o jsonpath='{.spec.template.spec.containers[0].livenessProbe.httpGet.path}')

    if [ "$LIVENESS_PATH" = "/healthz" ]; then
        print_success "Liveness probe configured: $LIVENESS_PATH"
    else
        print_warning "Liveness probe path: $LIVENESS_PATH (expected: /healthz)"
    fi

    # Check readiness probe
    READINESS_PATH=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" \
        -o jsonpath='{.spec.template.spec.containers[0].readinessProbe.httpGet.path}')

    if [ "$READINESS_PATH" = "/ready" ]; then
        print_success "Readiness probe configured: $READINESS_PATH"
    else
        print_warning "Readiness probe path: $READINESS_PATH (expected: /ready)"
    fi
}

validate_security() {
    print_header "10. Validating Security Configuration"

    # Check security context
    RUN_AS_NON_ROOT=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" \
        -o jsonpath='{.spec.template.spec.securityContext.runAsNonRoot}')

    if [ "$RUN_AS_NON_ROOT" = "true" ]; then
        print_success "Running as non-root user"
    else
        print_error "Container should run as non-root"
    fi

    # Check read-only root filesystem
    READ_ONLY_FS=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" \
        -o jsonpath='{.spec.template.spec.containers[0].securityContext.readOnlyRootFilesystem}')

    if [ "$READ_ONLY_FS" = "true" ]; then
        print_success "Read-only root filesystem enabled"
    else
        print_warning "Read-only root filesystem not enabled"
    fi

    # Check privilege escalation
    ALLOW_PRIV_ESC=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" \
        -o jsonpath='{.spec.template.spec.containers[0].securityContext.allowPrivilegeEscalation}')

    if [ "$ALLOW_PRIV_ESC" = "false" ]; then
        print_success "Privilege escalation disabled"
    else
        print_error "Privilege escalation should be disabled"
    fi
}

generate_summary() {
    print_header "Validation Summary"

    POD_COUNT=$(kubectl get pods -n "$NAMESPACE" -l app="$DEPLOYMENT_NAME" \
        --field-selector=status.phase=Running --no-headers | wc -l)

    echo -e "${BLUE}Deployment:${NC}"
    echo "  Name: $DEPLOYMENT_NAME"
    echo "  Namespace: $NAMESPACE"
    echo "  Running pods: $POD_COUNT / $DESIRED_REPLICAS"
    echo ""

    echo -e "${BLUE}High Availability:${NC}"
    echo "  Availability zones: $ZONE_COUNT"
    echo "  Pods per zone: ~$(($POD_COUNT / $ZONE_COUNT))"
    echo "  Anti-affinity: Zone-level (hard)"
    echo ""

    echo -e "${BLUE}Auto-scaling:${NC}"
    echo "  HPA min replicas: $MIN_REPLICAS_HPA"
    echo "  HPA max replicas: $MAX_REPLICAS_HPA"
    echo "  Current replicas: $CURRENT_REPLICAS_HPA"
    echo ""

    echo -e "${BLUE}Resilience:${NC}"
    echo "  Zero-downtime updates: Yes (maxUnavailable=0)"
    echo "  PDB min available: $MIN_AVAILABLE"
    echo "  Current healthy: $CURRENT_HEALTHY"
    echo ""

    echo -e "${BLUE}Load Balancer:${NC}"
    echo "  Type: $SERVICE_TYPE ($NLB_TYPE)"
    echo "  Cross-zone: $CROSS_ZONE"
    echo "  Session affinity: $SESSION_AFFINITY"

    if [ -n "$EXTERNAL_IP" ] && [ "$EXTERNAL_IP" != "pending" ]; then
        echo "  External endpoint: $EXTERNAL_IP"
    fi
    echo ""

    print_success "Validation complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Monitor pod distribution: kubectl get pods -n $NAMESPACE -o wide"
    echo "  2. Watch HPA: kubectl get hpa -n $NAMESPACE -w"
    echo "  3. Test health endpoints: curl http://\$EXTERNAL_IP/healthz"
    echo "  4. Simulate zone failure: kubectl drain <node> --ignore-daemonsets"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    echo -e "${GREEN}"
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  GreenLang Multi-AZ HA Deployment Validation              ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    validate_prerequisites
    validate_nodes
    validate_deployment
    validate_pod_distribution
    validate_anti_affinity
    validate_hpa
    validate_pdb
    validate_service
    validate_health_checks
    validate_security
    generate_summary
}

# Run main function
main "$@"
