# DevOps Deployment - Detailed To-Do List for 3 Agents

**Version:** 1.0
**Date:** December 3, 2025
**Status:** READY FOR EXECUTION
**Scope:** Deploy 3 Production-Ready Agents to Kubernetes
**Agents:** Fuel Emissions Analyzer, CBAM Carbon Intensity Calculator, Building Energy Performance Calculator

---

## Executive Summary

This document provides a **granular, actionable to-do list** for deploying the 3 locally-tested agents to Kubernetes. Each item includes the component/tool, configuration details, required files, integration points, and monitoring requirements.

### Agents to Deploy

| Agent | Module Path | Lines of Code | Tools |
|-------|-------------|---------------|-------|
| **Fuel Emissions Analyzer** | `generated/fuel_analyzer_agent/` | 797 | LookupEmissionFactor, CalculateEmissions, ValidateFuelInput |
| **CBAM Carbon Intensity** | `generated/carbon_intensity_v1/` | 988 | LookupCbamBenchmark, CalculateCarbonIntensity |
| **Building Energy Performance** | `generated/energy_performance_v1/` | 1,212 | CalculateEui, LookupBpsThreshold, CheckBpsCompliance |

---

## Phase 1: Docker Containerization

### 1.1 Base Docker Image

**Component:** Base Python Docker Image
**Tool:** Docker, BuildKit
**Priority:** P0 - Critical Path

#### Tasks:

- [ ] **1.1.1** Create base image Dockerfile at `docker/base/Dockerfile.base`
  - **Configuration:**
    - Base: `python:3.11-slim`
    - Install system dependencies: `libpq-dev`, `curl`
    - Set non-root user (UID 1000)
    - Configure health check endpoint support
  - **Files to Create:**
    - `docker/base/Dockerfile.base`
    - `docker/base/requirements-base.txt`
  - **Integration:** ECR/GHCR container registry
  - **Monitoring:** Image size target < 250MB

- [ ] **1.1.2** Create multi-stage build template
  - **Configuration:**
    - Stage 1: Build (install dependencies)
    - Stage 2: Runtime (copy venv, app code)
    - Layer optimization (requirements before code)
  - **Files to Create:**
    - `docker/templates/Dockerfile.agent.template`
  - **Acceptance Criteria:** Build time < 3 minutes with cache

- [ ] **1.1.3** Configure BuildKit caching
  - **Configuration:**
    - Enable `DOCKER_BUILDKIT=1`
    - Configure cache mount for pip
    - Set up GitHub Actions cache-from/cache-to
  - **Files to Create:**
    - `.github/workflows/docker-build.yml` (partial)
  - **Monitoring:** Cache hit ratio > 80%

### 1.2 Agent-Specific Dockerfiles

**Component:** Per-Agent Docker Images
**Tool:** Docker
**Priority:** P0 - Critical Path

#### Tasks:

- [ ] **1.2.1** Create Dockerfile for Fuel Emissions Analyzer
  - **Configuration:**
    ```dockerfile
    # Path: generated/fuel_analyzer_agent/Dockerfile
    FROM greenlang/base:python3.11 as builder
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    FROM greenlang/base:python3.11
    COPY --from=builder /opt/venv /opt/venv
    COPY . /app
    USER greenlang
    EXPOSE 8000
    HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health || exit 1
    CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]
    ```
  - **Files to Create:**
    - `generated/fuel_analyzer_agent/Dockerfile`
    - `generated/fuel_analyzer_agent/requirements.txt`
    - `generated/fuel_analyzer_agent/.dockerignore`
  - **Integration:** CI/CD pipeline triggers on code changes
  - **Monitoring:** Image size target < 350MB

- [ ] **1.2.2** Create Dockerfile for CBAM Carbon Intensity
  - **Configuration:** Same pattern as 1.2.1
  - **Files to Create:**
    - `generated/carbon_intensity_v1/Dockerfile`
    - `generated/carbon_intensity_v1/requirements.txt`
    - `generated/carbon_intensity_v1/.dockerignore`
  - **Integration:** Shared base image layer
  - **Monitoring:** Image size target < 350MB

- [ ] **1.2.3** Create Dockerfile for Building Energy Performance
  - **Configuration:** Same pattern as 1.2.1
  - **Files to Create:**
    - `generated/energy_performance_v1/Dockerfile`
    - `generated/energy_performance_v1/requirements.txt`
    - `generated/energy_performance_v1/.dockerignore`
  - **Integration:** Shared base image layer
  - **Monitoring:** Image size target < 350MB

- [ ] **1.2.4** Create Docker Compose for local development
  - **Configuration:**
    ```yaml
    # Path: docker-compose.yml
    version: '3.9'
    services:
      fuel-analyzer:
        build: ./generated/fuel_analyzer_agent
        ports: ["8001:8000"]
        environment:
          - ENVIRONMENT=development
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      carbon-intensity:
        build: ./generated/carbon_intensity_v1
        ports: ["8002:8000"]
      energy-performance:
        build: ./generated/energy_performance_v1
        ports: ["8003:8000"]
    ```
  - **Files to Create:**
    - `docker-compose.yml`
    - `docker-compose.override.yml` (local secrets)
  - **Acceptance Criteria:** `docker compose up` runs all 3 agents

### 1.3 Container Security Hardening

**Component:** Security Configuration
**Tool:** Trivy, Docker Scout
**Priority:** P1 - High

#### Tasks:

- [ ] **1.3.1** Configure non-root user in all Dockerfiles
  - **Configuration:**
    ```dockerfile
    RUN useradd -m -u 1000 -s /bin/bash greenlang
    USER greenlang
    ```
  - **Files to Modify:** All agent Dockerfiles
  - **Monitoring:** User ID 1000 verified in running containers

- [ ] **1.3.2** Implement read-only root filesystem
  - **Configuration:**
    ```yaml
    securityContext:
      readOnlyRootFilesystem: true
    volumes:
      - name: tmp
        emptyDir: {}
    volumeMounts:
      - name: tmp
        mountPath: /tmp
    ```
  - **Files to Modify:** Kubernetes deployments
  - **Integration:** Pod Security Standards enforcement

- [ ] **1.3.3** Add Trivy scanning to CI/CD
  - **Configuration:**
    ```yaml
    - name: Trivy vulnerability scan
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ env.IMAGE_TAG }}
        severity: 'CRITICAL,HIGH'
        exit-code: '1'
    ```
  - **Files to Create:**
    - `.github/workflows/security-scan.yml`
  - **Monitoring:** Zero CRITICAL/HIGH vulnerabilities
  - **Alert:** Slack notification on scan failure

- [ ] **1.3.4** Configure image signing (Cosign)
  - **Configuration:**
    - Generate signing key pair
    - Sign images in CI/CD pipeline
    - Verify signatures in Kubernetes admission
  - **Files to Create:**
    - `scripts/sign-image.sh`
    - `kubernetes/policies/image-signature-policy.yaml`
  - **Integration:** Kubernetes admission controller

---

## Phase 2: Kubernetes Manifests

### 2.1 Namespace and RBAC

**Component:** Kubernetes Namespace Configuration
**Tool:** kubectl, Kustomize
**Priority:** P0 - Critical Path

#### Tasks:

- [ ] **2.1.1** Create namespace manifest
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/namespace.yaml
    apiVersion: v1
    kind: Namespace
    metadata:
      name: greenlang-agents
      labels:
        app.kubernetes.io/part-of: greenlang
        environment: production
        pod-security.kubernetes.io/enforce: restricted
    ```
  - **Files to Create:**
    - `kubernetes/manifests/namespace.yaml`
  - **Integration:** Existing greenlang namespace structure
  - **Monitoring:** Namespace resource quotas

- [ ] **2.1.2** Create ServiceAccount for agents
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/serviceaccount.yaml
    apiVersion: v1
    kind: ServiceAccount
    metadata:
      name: agent-service-account
      namespace: greenlang-agents
      annotations:
        eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT:role/AgentRole
    ```
  - **Files to Create:**
    - `kubernetes/manifests/serviceaccount.yaml`
  - **Integration:** AWS IAM Roles for Service Accounts (IRSA)
  - **Monitoring:** IAM role assumption logs

- [ ] **2.1.3** Create RBAC Role and RoleBinding
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/rbac.yaml
    apiVersion: rbac.authorization.k8s.io/v1
    kind: Role
    metadata:
      name: agent-role
      namespace: greenlang-agents
    rules:
      - apiGroups: [""]
        resources: ["configmaps", "secrets"]
        verbs: ["get", "list", "watch"]
      - apiGroups: [""]
        resources: ["pods"]
        verbs: ["get", "list"]
    ---
    apiVersion: rbac.authorization.k8s.io/v1
    kind: RoleBinding
    metadata:
      name: agent-role-binding
      namespace: greenlang-agents
    roleRef:
      apiGroup: rbac.authorization.k8s.io
      kind: Role
      name: agent-role
    subjects:
      - kind: ServiceAccount
        name: agent-service-account
        namespace: greenlang-agents
    ```
  - **Files to Create:**
    - `kubernetes/manifests/rbac.yaml`
  - **Integration:** Least privilege principle
  - **Monitoring:** RBAC audit logs

- [ ] **2.1.4** Create ResourceQuota
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/resource-quota.yaml
    apiVersion: v1
    kind: ResourceQuota
    metadata:
      name: agent-quota
      namespace: greenlang-agents
    spec:
      hard:
        requests.cpu: "50"
        requests.memory: "100Gi"
        limits.cpu: "100"
        limits.memory: "200Gi"
        pods: "100"
    ```
  - **Files to Create:**
    - `kubernetes/manifests/resource-quota.yaml`
  - **Monitoring:** Quota utilization dashboard
  - **Alert:** 80% quota threshold

- [ ] **2.1.5** Create LimitRange
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/limit-range.yaml
    apiVersion: v1
    kind: LimitRange
    metadata:
      name: agent-limit-range
      namespace: greenlang-agents
    spec:
      limits:
        - type: Container
          default:
            cpu: "500m"
            memory: "512Mi"
          defaultRequest:
            cpu: "100m"
            memory: "128Mi"
          max:
            cpu: "2"
            memory: "4Gi"
          min:
            cpu: "50m"
            memory: "64Mi"
    ```
  - **Files to Create:**
    - `kubernetes/manifests/limit-range.yaml`
  - **Integration:** Prevent resource starvation

### 2.2 ConfigMaps

**Component:** Configuration Management
**Tool:** kubectl
**Priority:** P0 - Critical Path

#### Tasks:

- [ ] **2.2.1** Create shared agent ConfigMap
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/configmap-shared.yaml
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: agent-config-shared
      namespace: greenlang-agents
    data:
      LOG_LEVEL: "INFO"
      LOG_FORMAT: "json"
      METRICS_PORT: "9090"
      HEALTH_PORT: "8000"
      TRACING_ENABLED: "true"
      TRACING_SAMPLE_RATE: "0.1"
      CACHE_TTL_SECONDS: "3600"
    ```
  - **Files to Create:**
    - `kubernetes/manifests/configmap-shared.yaml`
  - **Integration:** All agents mount this ConfigMap
  - **Monitoring:** ConfigMap change audit

- [ ] **2.2.2** Create Fuel Analyzer ConfigMap
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/configmap-fuel-analyzer.yaml
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: fuel-analyzer-config
      namespace: greenlang-agents
    data:
      AGENT_NAME: "fuel-emissions-analyzer"
      AGENT_VERSION: "1.0.0"
      EMISSION_FACTOR_SOURCE: "DEFRA_2023"
      SUPPORTED_FUELS: "natural_gas,diesel,gasoline,lpg,fuel_oil,coal,electricity_grid,propane,kerosene,biomass"
      SUPPORTED_REGIONS: "US,GB,EU"
      DEFAULT_GWP_SET: "AR5"
    ```
  - **Files to Create:**
    - `kubernetes/manifests/configmap-fuel-analyzer.yaml`
  - **Integration:** DEFRA 2023 emission factor database
  - **Monitoring:** Configuration drift detection

- [ ] **2.2.3** Create CBAM Carbon Intensity ConfigMap
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/configmap-carbon-intensity.yaml
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: carbon-intensity-config
      namespace: greenlang-agents
    data:
      AGENT_NAME: "cbam-carbon-intensity-calculator"
      AGENT_VERSION: "1.0.0"
      REGULATION_SOURCE: "EU_2023_1773"
      EFFECTIVE_DATE: "2026-01-01"
      SUPPORTED_PRODUCTS: "steel,cement,aluminum,fertilizers,electricity,hydrogen"
    ```
  - **Files to Create:**
    - `kubernetes/manifests/configmap-carbon-intensity.yaml`
  - **Integration:** EU CBAM regulation database

- [ ] **2.2.4** Create Building Energy Performance ConfigMap
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/configmap-energy-performance.yaml
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: energy-performance-config
      namespace: greenlang-agents
    data:
      AGENT_NAME: "building-energy-performance-calculator"
      AGENT_VERSION: "1.0.0"
      THRESHOLD_SOURCES: "NYC_LL97,ENERGY_STAR,ASHRAE_90.1"
      SUPPORTED_BUILDING_TYPES: "office,residential,retail,industrial,warehouse,hotel,hospital,school,restaurant"
      CLIMATE_ZONES: "1A,2A,3A,4A,5A,6A,7"
    ```
  - **Files to Create:**
    - `kubernetes/manifests/configmap-energy-performance.yaml`
  - **Integration:** BPS threshold database

### 2.3 Secrets Management

**Component:** Secrets Configuration
**Tool:** AWS Secrets Manager, External Secrets Operator
**Priority:** P0 - Critical Path

#### Tasks:

- [ ] **2.3.1** Deploy External Secrets Operator
  - **Configuration:**
    ```bash
    helm repo add external-secrets https://charts.external-secrets.io
    helm install external-secrets external-secrets/external-secrets \
      --namespace external-secrets \
      --create-namespace
    ```
  - **Files to Create:**
    - `kubernetes/helm/external-secrets-values.yaml`
  - **Integration:** AWS Secrets Manager
  - **Monitoring:** ESO controller health

- [ ] **2.3.2** Create SecretStore for AWS Secrets Manager
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/secret-store.yaml
    apiVersion: external-secrets.io/v1beta1
    kind: SecretStore
    metadata:
      name: aws-secrets-manager
      namespace: greenlang-agents
    spec:
      provider:
        aws:
          service: SecretsManager
          region: us-east-1
          auth:
            jwt:
              serviceAccountRef:
                name: agent-service-account
    ```
  - **Files to Create:**
    - `kubernetes/manifests/secret-store.yaml`
  - **Integration:** AWS IAM IRSA role

- [ ] **2.3.3** Create ExternalSecret for database credentials
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/external-secret-db.yaml
    apiVersion: external-secrets.io/v1beta1
    kind: ExternalSecret
    metadata:
      name: agent-db-credentials
      namespace: greenlang-agents
    spec:
      refreshInterval: 1h
      secretStoreRef:
        name: aws-secrets-manager
        kind: SecretStore
      target:
        name: agent-db-secret
        creationPolicy: Owner
      data:
        - secretKey: DATABASE_URL
          remoteRef:
            key: greenlang/agents/database
            property: url
        - secretKey: DATABASE_PASSWORD
          remoteRef:
            key: greenlang/agents/database
            property: password
    ```
  - **Files to Create:**
    - `kubernetes/manifests/external-secret-db.yaml`
  - **Monitoring:** Secret sync status
  - **Alert:** Secret sync failure

- [ ] **2.3.4** Create ExternalSecret for API keys
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/external-secret-api.yaml
    apiVersion: external-secrets.io/v1beta1
    kind: ExternalSecret
    metadata:
      name: agent-api-keys
      namespace: greenlang-agents
    spec:
      refreshInterval: 1h
      secretStoreRef:
        name: aws-secrets-manager
        kind: SecretStore
      target:
        name: agent-api-secret
      data:
        - secretKey: ANTHROPIC_API_KEY
          remoteRef:
            key: greenlang/agents/api-keys
            property: anthropic
    ```
  - **Files to Create:**
    - `kubernetes/manifests/external-secret-api.yaml`
  - **Integration:** LLM provider authentication
  - **Monitoring:** API key rotation tracking

### 2.4 Deployments

**Component:** Kubernetes Deployments
**Tool:** kubectl, Kustomize
**Priority:** P0 - Critical Path

#### Tasks:

- [ ] **2.4.1** Create Fuel Analyzer Deployment
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/deployment-fuel-analyzer.yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: fuel-analyzer
      namespace: greenlang-agents
      labels:
        app: fuel-analyzer
        version: v1.0.0
    spec:
      replicas: 3
      strategy:
        type: RollingUpdate
        rollingUpdate:
          maxSurge: 1
          maxUnavailable: 0
      selector:
        matchLabels:
          app: fuel-analyzer
      template:
        metadata:
          labels:
            app: fuel-analyzer
            version: v1.0.0
          annotations:
            prometheus.io/scrape: "true"
            prometheus.io/port: "9090"
            prometheus.io/path: "/metrics"
        spec:
          serviceAccountName: agent-service-account
          securityContext:
            runAsNonRoot: true
            runAsUser: 1000
            fsGroup: 1000
          containers:
            - name: fuel-analyzer
              image: ghcr.io/greenlang/fuel-analyzer:v1.0.0
              imagePullPolicy: Always
              ports:
                - name: http
                  containerPort: 8000
                - name: metrics
                  containerPort: 9090
              envFrom:
                - configMapRef:
                    name: agent-config-shared
                - configMapRef:
                    name: fuel-analyzer-config
              env:
                - name: DATABASE_URL
                  valueFrom:
                    secretKeyRef:
                      name: agent-db-secret
                      key: DATABASE_URL
              resources:
                requests:
                  cpu: "250m"
                  memory: "256Mi"
                limits:
                  cpu: "1000m"
                  memory: "1Gi"
              livenessProbe:
                httpGet:
                  path: /health/live
                  port: 8000
                initialDelaySeconds: 30
                periodSeconds: 10
                timeoutSeconds: 5
                failureThreshold: 3
              readinessProbe:
                httpGet:
                  path: /health/ready
                  port: 8000
                initialDelaySeconds: 10
                periodSeconds: 5
                timeoutSeconds: 3
                successThreshold: 1
              securityContext:
                allowPrivilegeEscalation: false
                readOnlyRootFilesystem: true
                capabilities:
                  drop:
                    - ALL
              volumeMounts:
                - name: tmp
                  mountPath: /tmp
          volumes:
            - name: tmp
              emptyDir: {}
          topologySpreadConstraints:
            - maxSkew: 1
              topologyKey: topology.kubernetes.io/zone
              whenUnsatisfiable: DoNotSchedule
              labelSelector:
                matchLabels:
                  app: fuel-analyzer
    ```
  - **Files to Create:**
    - `kubernetes/manifests/deployment-fuel-analyzer.yaml`
  - **Integration:** Service mesh (Istio/Linkerd) sidecar
  - **Monitoring:** Pod ready status, restart count
  - **Alert:** Pod crash loop, high restart count

- [ ] **2.4.2** Create CBAM Carbon Intensity Deployment
  - **Configuration:** Same pattern as 2.4.1
  - **Files to Create:**
    - `kubernetes/manifests/deployment-carbon-intensity.yaml`
  - **Specific Config:**
    - Image: `ghcr.io/greenlang/carbon-intensity:v1.0.0`
    - Replicas: 3
    - Resources: cpu 250m-1000m, memory 256Mi-1Gi

- [ ] **2.4.3** Create Building Energy Performance Deployment
  - **Configuration:** Same pattern as 2.4.1
  - **Files to Create:**
    - `kubernetes/manifests/deployment-energy-performance.yaml`
  - **Specific Config:**
    - Image: `ghcr.io/greenlang/energy-performance:v1.0.0`
    - Replicas: 3
    - Resources: cpu 250m-1000m, memory 256Mi-1Gi

- [ ] **2.4.4** Create PodDisruptionBudget for each agent
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/pdb.yaml
    apiVersion: policy/v1
    kind: PodDisruptionBudget
    metadata:
      name: fuel-analyzer-pdb
      namespace: greenlang-agents
    spec:
      minAvailable: 2
      selector:
        matchLabels:
          app: fuel-analyzer
    ---
    apiVersion: policy/v1
    kind: PodDisruptionBudget
    metadata:
      name: carbon-intensity-pdb
      namespace: greenlang-agents
    spec:
      minAvailable: 2
      selector:
        matchLabels:
          app: carbon-intensity
    ---
    apiVersion: policy/v1
    kind: PodDisruptionBudget
    metadata:
      name: energy-performance-pdb
      namespace: greenlang-agents
    spec:
      minAvailable: 2
      selector:
        matchLabels:
          app: energy-performance
    ```
  - **Files to Create:**
    - `kubernetes/manifests/pdb.yaml`
  - **Integration:** Cluster upgrade protection
  - **Monitoring:** PDB violation events

### 2.5 Services

**Component:** Kubernetes Services
**Tool:** kubectl
**Priority:** P0 - Critical Path

#### Tasks:

- [ ] **2.5.1** Create ClusterIP Services for each agent
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/services.yaml
    apiVersion: v1
    kind: Service
    metadata:
      name: fuel-analyzer
      namespace: greenlang-agents
      labels:
        app: fuel-analyzer
    spec:
      type: ClusterIP
      selector:
        app: fuel-analyzer
      ports:
        - name: http
          port: 80
          targetPort: 8000
        - name: metrics
          port: 9090
          targetPort: 9090
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: carbon-intensity
      namespace: greenlang-agents
    spec:
      type: ClusterIP
      selector:
        app: carbon-intensity
      ports:
        - name: http
          port: 80
          targetPort: 8000
        - name: metrics
          port: 9090
          targetPort: 9090
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: energy-performance
      namespace: greenlang-agents
    spec:
      type: ClusterIP
      selector:
        app: energy-performance
      ports:
        - name: http
          port: 80
          targetPort: 8000
        - name: metrics
          port: 9090
          targetPort: 9090
    ```
  - **Files to Create:**
    - `kubernetes/manifests/services.yaml`
  - **Integration:** Internal service discovery
  - **Monitoring:** Service endpoint health

### 2.6 Ingress Configuration

**Component:** Ingress and TLS
**Tool:** NGINX Ingress Controller, cert-manager
**Priority:** P0 - Critical Path

#### Tasks:

- [ ] **2.6.1** Create Ingress resource
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/ingress.yaml
    apiVersion: networking.k8s.io/v1
    kind: Ingress
    metadata:
      name: agents-ingress
      namespace: greenlang-agents
      annotations:
        kubernetes.io/ingress.class: "nginx"
        nginx.ingress.kubernetes.io/ssl-redirect: "true"
        nginx.ingress.kubernetes.io/proxy-body-size: "10m"
        nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
        nginx.ingress.kubernetes.io/rate-limit: "100"
        nginx.ingress.kubernetes.io/rate-limit-burst-multiplier: "5"
        cert-manager.io/cluster-issuer: "letsencrypt-prod"
    spec:
      tls:
        - hosts:
            - agents.greenlang.ai
          secretName: agents-tls
      rules:
        - host: agents.greenlang.ai
          http:
            paths:
              - path: /api/v1/fuel-analyzer
                pathType: Prefix
                backend:
                  service:
                    name: fuel-analyzer
                    port:
                      number: 80
              - path: /api/v1/carbon-intensity
                pathType: Prefix
                backend:
                  service:
                    name: carbon-intensity
                    port:
                      number: 80
              - path: /api/v1/energy-performance
                pathType: Prefix
                backend:
                  service:
                    name: energy-performance
                    port:
                      number: 80
    ```
  - **Files to Create:**
    - `kubernetes/manifests/ingress.yaml`
  - **Integration:** DNS (Route 53), TLS (cert-manager)
  - **Monitoring:** Ingress controller logs, 5xx rate
  - **Alert:** High 5xx error rate

- [ ] **2.6.2** Create cert-manager Certificate
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/certificate.yaml
    apiVersion: cert-manager.io/v1
    kind: Certificate
    metadata:
      name: agents-tls
      namespace: greenlang-agents
    spec:
      secretName: agents-tls
      issuerRef:
        name: letsencrypt-prod
        kind: ClusterIssuer
      dnsNames:
        - agents.greenlang.ai
        - "*.agents.greenlang.ai"
    ```
  - **Files to Create:**
    - `kubernetes/manifests/certificate.yaml`
  - **Monitoring:** Certificate expiry (30 days warning)
  - **Alert:** Certificate expiring soon

### 2.7 Network Policies

**Component:** Network Security
**Tool:** Calico/Cilium
**Priority:** P1 - High

#### Tasks:

- [ ] **2.7.1** Create default deny NetworkPolicy
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/network-policy-default-deny.yaml
    apiVersion: networking.k8s.io/v1
    kind: NetworkPolicy
    metadata:
      name: default-deny-all
      namespace: greenlang-agents
    spec:
      podSelector: {}
      policyTypes:
        - Ingress
        - Egress
    ```
  - **Files to Create:**
    - `kubernetes/manifests/network-policy-default-deny.yaml`
  - **Integration:** Zero-trust networking

- [ ] **2.7.2** Create allow-list NetworkPolicy for agents
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/network-policy-agents.yaml
    apiVersion: networking.k8s.io/v1
    kind: NetworkPolicy
    metadata:
      name: allow-agent-traffic
      namespace: greenlang-agents
    spec:
      podSelector:
        matchLabels:
          app.kubernetes.io/part-of: greenlang-agents
      policyTypes:
        - Ingress
        - Egress
      ingress:
        # Allow from ingress controller
        - from:
            - namespaceSelector:
                matchLabels:
                  name: ingress-nginx
          ports:
            - protocol: TCP
              port: 8000
        # Allow from monitoring
        - from:
            - namespaceSelector:
                matchLabels:
                  name: monitoring
          ports:
            - protocol: TCP
              port: 9090
      egress:
        # Allow DNS
        - to:
            - namespaceSelector:
                matchLabels:
                  name: kube-system
          ports:
            - protocol: UDP
              port: 53
        # Allow HTTPS (external APIs)
        - to:
            - ipBlock:
                cidr: 0.0.0.0/0
          ports:
            - protocol: TCP
              port: 443
    ```
  - **Files to Create:**
    - `kubernetes/manifests/network-policy-agents.yaml`
  - **Monitoring:** Network policy audit logs

### 2.8 HorizontalPodAutoscaler

**Component:** Auto-scaling
**Tool:** HPA v2
**Priority:** P1 - High

#### Tasks:

- [ ] **2.8.1** Create HPA for Fuel Analyzer
  - **Configuration:**
    ```yaml
    # Path: kubernetes/manifests/hpa.yaml
    apiVersion: autoscaling/v2
    kind: HorizontalPodAutoscaler
    metadata:
      name: fuel-analyzer-hpa
      namespace: greenlang-agents
    spec:
      scaleTargetRef:
        apiVersion: apps/v1
        kind: Deployment
        name: fuel-analyzer
      minReplicas: 3
      maxReplicas: 20
      metrics:
        - type: Resource
          resource:
            name: cpu
            target:
              type: Utilization
              averageUtilization: 70
        - type: Resource
          resource:
            name: memory
            target:
              type: Utilization
              averageUtilization: 80
      behavior:
        scaleUp:
          stabilizationWindowSeconds: 60
          policies:
            - type: Pods
              value: 4
              periodSeconds: 60
            - type: Percent
              value: 100
              periodSeconds: 60
          selectPolicy: Max
        scaleDown:
          stabilizationWindowSeconds: 300
          policies:
            - type: Pods
              value: 2
              periodSeconds: 60
          selectPolicy: Min
    ```
  - **Files to Create:**
    - `kubernetes/manifests/hpa.yaml` (includes all 3 agents)
  - **Monitoring:** HPA current/desired replicas
  - **Alert:** HPA at max replicas

- [ ] **2.8.2** Create HPA for CBAM Carbon Intensity
  - **Configuration:** Same pattern as 2.8.1
  - **Specific:** minReplicas: 3, maxReplicas: 15

- [ ] **2.8.3** Create HPA for Building Energy Performance
  - **Configuration:** Same pattern as 2.8.1
  - **Specific:** minReplicas: 3, maxReplicas: 15

---

## Phase 3: Helm Charts

### 3.1 Helm Chart Structure

**Component:** Helm Chart Packaging
**Tool:** Helm 3
**Priority:** P1 - High

#### Tasks:

- [ ] **3.1.1** Create umbrella chart structure
  - **Configuration:**
    ```
    helm/greenlang-agents/
    ├── Chart.yaml
    ├── values.yaml
    ├── values-dev.yaml
    ├── values-staging.yaml
    ├── values-prod.yaml
    ├── templates/
    │   ├── _helpers.tpl
    │   ├── namespace.yaml
    │   ├── serviceaccount.yaml
    │   ├── rbac.yaml
    │   ├── configmap-shared.yaml
    │   ├── secret-store.yaml
    │   └── NOTES.txt
    └── charts/
        ├── fuel-analyzer/
        ├── carbon-intensity/
        └── energy-performance/
    ```
  - **Files to Create:**
    - `helm/greenlang-agents/Chart.yaml`
    - `helm/greenlang-agents/values.yaml`
  - **Integration:** Helm repository (GitHub Pages or Chartmuseum)

- [ ] **3.1.2** Create Chart.yaml
  - **Configuration:**
    ```yaml
    # Path: helm/greenlang-agents/Chart.yaml
    apiVersion: v2
    name: greenlang-agents
    description: GreenLang Agent Factory - Production Agents
    type: application
    version: 1.0.0
    appVersion: "1.0.0"
    dependencies:
      - name: fuel-analyzer
        version: "1.0.0"
        repository: "file://charts/fuel-analyzer"
      - name: carbon-intensity
        version: "1.0.0"
        repository: "file://charts/carbon-intensity"
      - name: energy-performance
        version: "1.0.0"
        repository: "file://charts/energy-performance"
    ```
  - **Files to Create:**
    - `helm/greenlang-agents/Chart.yaml`

- [ ] **3.1.3** Create values.yaml with defaults
  - **Configuration:**
    ```yaml
    # Path: helm/greenlang-agents/values.yaml
    global:
      namespace: greenlang-agents
      image:
        registry: ghcr.io/greenlang
        pullPolicy: Always
      environment: production
      logging:
        level: INFO
        format: json
      tracing:
        enabled: true
        sampleRate: 0.1
      monitoring:
        prometheus:
          enabled: true
          port: 9090

    fuel-analyzer:
      enabled: true
      replicaCount: 3
      image:
        repository: fuel-analyzer
        tag: "v1.0.0"
      resources:
        requests:
          cpu: 250m
          memory: 256Mi
        limits:
          cpu: 1000m
          memory: 1Gi
      hpa:
        enabled: true
        minReplicas: 3
        maxReplicas: 20
        targetCPUUtilization: 70

    carbon-intensity:
      enabled: true
      replicaCount: 3
      # ... similar structure

    energy-performance:
      enabled: true
      replicaCount: 3
      # ... similar structure
    ```
  - **Files to Create:**
    - `helm/greenlang-agents/values.yaml`
    - `helm/greenlang-agents/values-dev.yaml`
    - `helm/greenlang-agents/values-staging.yaml`
    - `helm/greenlang-agents/values-prod.yaml`

- [ ] **3.1.4** Create sub-chart for Fuel Analyzer
  - **Configuration:** Standard Helm chart structure
  - **Files to Create:**
    - `helm/greenlang-agents/charts/fuel-analyzer/Chart.yaml`
    - `helm/greenlang-agents/charts/fuel-analyzer/values.yaml`
    - `helm/greenlang-agents/charts/fuel-analyzer/templates/deployment.yaml`
    - `helm/greenlang-agents/charts/fuel-analyzer/templates/service.yaml`
    - `helm/greenlang-agents/charts/fuel-analyzer/templates/configmap.yaml`
    - `helm/greenlang-agents/charts/fuel-analyzer/templates/hpa.yaml`
    - `helm/greenlang-agents/charts/fuel-analyzer/templates/servicemonitor.yaml`

- [ ] **3.1.5** Create sub-charts for remaining agents
  - **Files to Create:**
    - `helm/greenlang-agents/charts/carbon-intensity/` (full chart)
    - `helm/greenlang-agents/charts/energy-performance/` (full chart)

- [ ] **3.1.6** Create Helmfile for multi-environment deployment
  - **Configuration:**
    ```yaml
    # Path: helmfile.yaml
    repositories:
      - name: greenlang
        url: https://charts.greenlang.ai

    environments:
      dev:
        values:
          - environments/dev.yaml
      staging:
        values:
          - environments/staging.yaml
      prod:
        values:
          - environments/prod.yaml

    releases:
      - name: greenlang-agents
        namespace: greenlang-agents
        chart: ./helm/greenlang-agents
        values:
          - helm/greenlang-agents/values.yaml
          - helm/greenlang-agents/values-{{ .Environment.Name }}.yaml
    ```
  - **Files to Create:**
    - `helmfile.yaml`
    - `environments/dev.yaml`
    - `environments/staging.yaml`
    - `environments/prod.yaml`

---

## Phase 4: CI/CD Pipeline

### 4.1 GitHub Actions Workflows

**Component:** CI/CD Automation
**Tool:** GitHub Actions
**Priority:** P0 - Critical Path

#### Tasks:

- [ ] **4.1.1** Create PR validation workflow
  - **Configuration:**
    ```yaml
    # Path: .github/workflows/pr-validation.yml
    name: PR Validation
    on:
      pull_request:
        branches: [main, develop]
        paths:
          - 'generated/**'
          - 'kubernetes/**'
          - 'helm/**'

    jobs:
      lint:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - name: Python lint
            uses: chartboost/ruff-action@v1
          - name: YAML lint
            uses: ibiqlik/action-yamllint@v3
          - name: Helm lint
            run: helm lint helm/greenlang-agents

      test:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - uses: actions/setup-python@v5
            with:
              python-version: '3.11'
          - name: Install dependencies
            run: pip install -r requirements-test.txt
          - name: Run tests
            run: pytest --cov=generated --cov-report=xml
          - name: Upload coverage
            uses: codecov/codecov-action@v3

      security:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - name: Run Bandit
            uses: jpetrucciani/bandit-check@main
          - name: Run Safety
            run: pip install safety && safety check
          - name: Run Trivy FS scan
            uses: aquasecurity/trivy-action@master
            with:
              scan-type: 'fs'
              severity: 'CRITICAL,HIGH'
    ```
  - **Files to Create:**
    - `.github/workflows/pr-validation.yml`
  - **Integration:** GitHub branch protection rules
  - **Monitoring:** PR check status

- [ ] **4.1.2** Create Docker build and push workflow
  - **Configuration:**
    ```yaml
    # Path: .github/workflows/docker-build.yml
    name: Build and Push Docker Images
    on:
      push:
        branches: [main]
        paths:
          - 'generated/**'
      workflow_dispatch:

    env:
      REGISTRY: ghcr.io
      IMAGE_PREFIX: greenlang

    jobs:
      build-matrix:
        strategy:
          matrix:
            agent:
              - name: fuel-analyzer
                path: generated/fuel_analyzer_agent
              - name: carbon-intensity
                path: generated/carbon_intensity_v1
              - name: energy-performance
                path: generated/energy_performance_v1

        runs-on: ubuntu-latest
        permissions:
          contents: read
          packages: write

        steps:
          - uses: actions/checkout@v4

          - name: Set up Docker Buildx
            uses: docker/setup-buildx-action@v3

          - name: Login to GHCR
            uses: docker/login-action@v3
            with:
              registry: ${{ env.REGISTRY }}
              username: ${{ github.actor }}
              password: ${{ secrets.GITHUB_TOKEN }}

          - name: Extract metadata
            id: meta
            uses: docker/metadata-action@v5
            with:
              images: ${{ env.REGISTRY }}/${{ env.IMAGE_PREFIX }}/${{ matrix.agent.name }}
              tags: |
                type=sha,prefix=
                type=ref,event=branch
                type=semver,pattern={{version}}

          - name: Build and push
            uses: docker/build-push-action@v5
            with:
              context: ${{ matrix.agent.path }}
              push: true
              tags: ${{ steps.meta.outputs.tags }}
              labels: ${{ steps.meta.outputs.labels }}
              cache-from: type=gha
              cache-to: type=gha,mode=max

          - name: Trivy scan
            uses: aquasecurity/trivy-action@master
            with:
              image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_PREFIX }}/${{ matrix.agent.name }}:${{ github.sha }}
              severity: 'CRITICAL,HIGH'
              exit-code: '1'
    ```
  - **Files to Create:**
    - `.github/workflows/docker-build.yml`
  - **Integration:** Container registry (GHCR)
  - **Monitoring:** Build duration, success rate
  - **Alert:** Build failure notification

- [ ] **4.1.3** Create deployment workflow
  - **Configuration:**
    ```yaml
    # Path: .github/workflows/deploy.yml
    name: Deploy to Kubernetes
    on:
      workflow_run:
        workflows: ["Build and Push Docker Images"]
        types: [completed]
        branches: [main]
      workflow_dispatch:
        inputs:
          environment:
            description: 'Deployment environment'
            required: true
            default: 'staging'
            type: choice
            options:
              - staging
              - production

    jobs:
      deploy-staging:
        if: ${{ github.event.workflow_run.conclusion == 'success' || github.event.inputs.environment == 'staging' }}
        runs-on: ubuntu-latest
        environment: staging
        steps:
          - uses: actions/checkout@v4

          - name: Configure kubectl
            uses: azure/k8s-set-context@v3
            with:
              kubeconfig: ${{ secrets.KUBECONFIG_STAGING }}

          - name: Deploy with Helm
            run: |
              helm upgrade --install greenlang-agents ./helm/greenlang-agents \
                --namespace greenlang-agents \
                --create-namespace \
                -f helm/greenlang-agents/values-staging.yaml \
                --set global.image.tag=${{ github.sha }} \
                --wait --timeout 10m

          - name: Verify deployment
            run: |
              kubectl rollout status deployment/fuel-analyzer -n greenlang-agents
              kubectl rollout status deployment/carbon-intensity -n greenlang-agents
              kubectl rollout status deployment/energy-performance -n greenlang-agents

          - name: Run smoke tests
            run: |
              ./scripts/smoke-test.sh staging

      deploy-production:
        needs: deploy-staging
        if: ${{ github.event.inputs.environment == 'production' }}
        runs-on: ubuntu-latest
        environment: production
        steps:
          - uses: actions/checkout@v4

          - name: Configure kubectl
            uses: azure/k8s-set-context@v3
            with:
              kubeconfig: ${{ secrets.KUBECONFIG_PROD }}

          - name: Deploy with Helm (Blue-Green)
            run: |
              # Deploy to green environment
              helm upgrade --install greenlang-agents-green ./helm/greenlang-agents \
                --namespace greenlang-agents \
                -f helm/greenlang-agents/values-prod.yaml \
                --set global.image.tag=${{ github.sha }} \
                --set global.deployment.color=green \
                --wait --timeout 15m

          - name: Run production smoke tests
            run: |
              ./scripts/smoke-test.sh production-green

          - name: Switch traffic to green
            run: |
              kubectl patch service fuel-analyzer -n greenlang-agents \
                -p '{"spec":{"selector":{"deployment-color":"green"}}}'
              # Repeat for other services

          - name: Cleanup old blue deployment
            run: |
              helm uninstall greenlang-agents-blue -n greenlang-agents || true
    ```
  - **Files to Create:**
    - `.github/workflows/deploy.yml`
  - **Integration:** GitHub Environments (staging, production)
  - **Monitoring:** Deployment duration, rollback count
  - **Alert:** Deployment failure, rollback triggered

- [ ] **4.1.4** Create rollback workflow
  - **Configuration:**
    ```yaml
    # Path: .github/workflows/rollback.yml
    name: Rollback Deployment
    on:
      workflow_dispatch:
        inputs:
          environment:
            description: 'Environment to rollback'
            required: true
            type: choice
            options:
              - staging
              - production
          revision:
            description: 'Helm revision to rollback to (leave empty for previous)'
            required: false

    jobs:
      rollback:
        runs-on: ubuntu-latest
        environment: ${{ github.event.inputs.environment }}
        steps:
          - name: Configure kubectl
            uses: azure/k8s-set-context@v3
            with:
              kubeconfig: ${{ secrets[format('KUBECONFIG_{0}', github.event.inputs.environment)] }}

          - name: Rollback Helm release
            run: |
              if [ -n "${{ github.event.inputs.revision }}" ]; then
                helm rollback greenlang-agents ${{ github.event.inputs.revision }} -n greenlang-agents
              else
                helm rollback greenlang-agents -n greenlang-agents
              fi

          - name: Verify rollback
            run: |
              kubectl rollout status deployment/fuel-analyzer -n greenlang-agents
              kubectl rollout status deployment/carbon-intensity -n greenlang-agents
              kubectl rollout status deployment/energy-performance -n greenlang-agents

          - name: Notify Slack
            uses: slackapi/slack-github-action@v1
            with:
              payload: |
                {
                  "text": "Rollback completed for ${{ github.event.inputs.environment }}"
                }
            env:
              SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
    ```
  - **Files to Create:**
    - `.github/workflows/rollback.yml`
  - **Monitoring:** Rollback frequency
  - **Alert:** Post-rollback notification

### 4.2 ArgoCD GitOps (Optional Alternative)

**Component:** GitOps Deployment
**Tool:** ArgoCD
**Priority:** P2 - Medium

#### Tasks:

- [ ] **4.2.1** Create ArgoCD Application manifest
  - **Configuration:**
    ```yaml
    # Path: argocd/applications/greenlang-agents.yaml
    apiVersion: argoproj.io/v1alpha1
    kind: Application
    metadata:
      name: greenlang-agents
      namespace: argocd
    spec:
      project: default
      source:
        repoURL: https://github.com/greenlang/gl-agent-factory
        targetRevision: HEAD
        path: helm/greenlang-agents
        helm:
          valueFiles:
            - values-prod.yaml
      destination:
        server: https://kubernetes.default.svc
        namespace: greenlang-agents
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
        syncOptions:
          - CreateNamespace=true
    ```
  - **Files to Create:**
    - `argocd/applications/greenlang-agents.yaml`
  - **Integration:** ArgoCD server
  - **Monitoring:** ArgoCD sync status

---

## Phase 5: Infrastructure as Code (Terraform)

### 5.1 Terraform Module Structure

**Component:** Cloud Infrastructure
**Tool:** Terraform
**Priority:** P1 - High

#### Tasks:

- [ ] **5.1.1** Create Terraform project structure
  - **Configuration:**
    ```
    terraform/
    ├── modules/
    │   ├── vpc/
    │   ├── eks/
    │   ├── rds/
    │   ├── elasticache/
    │   ├── s3/
    │   └── secrets-manager/
    ├── environments/
    │   ├── dev/
    │   │   ├── main.tf
    │   │   ├── variables.tf
    │   │   └── terraform.tfvars
    │   ├── staging/
    │   └── prod/
    ├── backend.tf
    └── versions.tf
    ```
  - **Files to Create:**
    - `terraform/versions.tf`
    - `terraform/backend.tf`
  - **Integration:** Terraform Cloud or S3 backend

- [ ] **5.1.2** Create VPC module
  - **Configuration:**
    ```hcl
    # Path: terraform/modules/vpc/main.tf
    module "vpc" {
      source  = "terraform-aws-modules/vpc/aws"
      version = "~> 5.0"

      name = "greenlang-${var.environment}"
      cidr = var.vpc_cidr

      azs             = var.availability_zones
      private_subnets = var.private_subnet_cidrs
      public_subnets  = var.public_subnet_cidrs

      enable_nat_gateway     = true
      single_nat_gateway     = var.environment != "prod"
      enable_dns_hostnames   = true
      enable_dns_support     = true

      tags = {
        Environment = var.environment
        Project     = "greenlang-agents"
        ManagedBy   = "terraform"
      }
    }
    ```
  - **Files to Create:**
    - `terraform/modules/vpc/main.tf`
    - `terraform/modules/vpc/variables.tf`
    - `terraform/modules/vpc/outputs.tf`

- [ ] **5.1.3** Create EKS module
  - **Configuration:**
    ```hcl
    # Path: terraform/modules/eks/main.tf
    module "eks" {
      source  = "terraform-aws-modules/eks/aws"
      version = "~> 19.0"

      cluster_name    = "greenlang-${var.environment}"
      cluster_version = "1.28"

      vpc_id     = var.vpc_id
      subnet_ids = var.private_subnet_ids

      cluster_endpoint_public_access = true

      eks_managed_node_groups = {
        system = {
          instance_types = ["m6i.xlarge"]
          min_size       = 3
          max_size       = 5
          desired_size   = 3
          labels = {
            workload-type = "system"
          }
        }
        agents = {
          instance_types = ["c6i.xlarge"]
          min_size       = 3
          max_size       = 20
          desired_size   = 3
          labels = {
            workload-type = "agents"
          }
        }
      }

      tags = {
        Environment = var.environment
      }
    }
    ```
  - **Files to Create:**
    - `terraform/modules/eks/main.tf`
    - `terraform/modules/eks/variables.tf`
    - `terraform/modules/eks/outputs.tf`

- [ ] **5.1.4** Create RDS module for PostgreSQL
  - **Configuration:**
    ```hcl
    # Path: terraform/modules/rds/main.tf
    resource "aws_db_instance" "postgresql" {
      identifier = "greenlang-${var.environment}"

      engine         = "postgres"
      engine_version = "15.4"
      instance_class = var.instance_class

      allocated_storage     = var.allocated_storage
      max_allocated_storage = var.max_allocated_storage
      storage_encrypted     = true

      db_name  = "greenlang"
      username = "greenlang_admin"
      password = random_password.db_password.result

      vpc_security_group_ids = [aws_security_group.rds.id]
      db_subnet_group_name   = aws_db_subnet_group.main.name

      multi_az               = var.environment == "prod"
      backup_retention_period = 30
      backup_window          = "03:00-04:00"

      skip_final_snapshot = var.environment != "prod"

      tags = {
        Environment = var.environment
      }
    }
    ```
  - **Files to Create:**
    - `terraform/modules/rds/main.tf`
    - `terraform/modules/rds/variables.tf`
    - `terraform/modules/rds/outputs.tf`

- [ ] **5.1.5** Create ElastiCache module for Redis
  - **Configuration:**
    ```hcl
    # Path: terraform/modules/elasticache/main.tf
    resource "aws_elasticache_replication_group" "redis" {
      replication_group_id       = "greenlang-${var.environment}"
      description                = "GreenLang Redis cluster"

      engine               = "redis"
      engine_version       = "7.0"
      node_type            = var.node_type
      num_cache_clusters   = var.num_cache_clusters

      automatic_failover_enabled = var.environment == "prod"
      multi_az_enabled          = var.environment == "prod"

      subnet_group_name  = aws_elasticache_subnet_group.main.name
      security_group_ids = [aws_security_group.redis.id]

      at_rest_encryption_enabled = true
      transit_encryption_enabled = true

      snapshot_retention_limit = 7
      snapshot_window         = "03:00-05:00"

      tags = {
        Environment = var.environment
      }
    }
    ```
  - **Files to Create:**
    - `terraform/modules/elasticache/main.tf`
    - `terraform/modules/elasticache/variables.tf`
    - `terraform/modules/elasticache/outputs.tf`

- [ ] **5.1.6** Create Secrets Manager module
  - **Configuration:**
    ```hcl
    # Path: terraform/modules/secrets-manager/main.tf
    resource "aws_secretsmanager_secret" "agent_secrets" {
      name = "greenlang/agents/${var.environment}"

      tags = {
        Environment = var.environment
      }
    }

    resource "aws_secretsmanager_secret_version" "agent_secrets" {
      secret_id = aws_secretsmanager_secret.agent_secrets.id
      secret_string = jsonencode({
        database_url     = var.database_url
        redis_url        = var.redis_url
        anthropic_api_key = var.anthropic_api_key
      })
    }
    ```
  - **Files to Create:**
    - `terraform/modules/secrets-manager/main.tf`
    - `terraform/modules/secrets-manager/variables.tf`

- [ ] **5.1.7** Create production environment configuration
  - **Configuration:**
    ```hcl
    # Path: terraform/environments/prod/main.tf
    module "vpc" {
      source = "../../modules/vpc"

      environment         = "prod"
      vpc_cidr           = "10.0.0.0/16"
      availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
    }

    module "eks" {
      source = "../../modules/eks"

      environment        = "prod"
      vpc_id            = module.vpc.vpc_id
      private_subnet_ids = module.vpc.private_subnet_ids
    }

    module "rds" {
      source = "../../modules/rds"

      environment       = "prod"
      instance_class    = "db.r6g.xlarge"
      allocated_storage = 100
    }

    module "redis" {
      source = "../../modules/elasticache"

      environment        = "prod"
      node_type         = "cache.r6g.large"
      num_cache_clusters = 3
    }
    ```
  - **Files to Create:**
    - `terraform/environments/prod/main.tf`
    - `terraform/environments/prod/variables.tf`
    - `terraform/environments/prod/terraform.tfvars`

---

## Phase 6: Monitoring and Observability

### 6.1 Prometheus Configuration

**Component:** Metrics Collection
**Tool:** Prometheus, Prometheus Operator
**Priority:** P1 - High

#### Tasks:

- [ ] **6.1.1** Create ServiceMonitor for agents
  - **Configuration:**
    ```yaml
    # Path: kubernetes/monitoring/servicemonitor-agents.yaml
    apiVersion: monitoring.coreos.com/v1
    kind: ServiceMonitor
    metadata:
      name: greenlang-agents
      namespace: monitoring
      labels:
        release: prometheus
    spec:
      selector:
        matchLabels:
          app.kubernetes.io/part-of: greenlang-agents
      namespaceSelector:
        matchNames:
          - greenlang-agents
      endpoints:
        - port: metrics
          interval: 15s
          path: /metrics
          scrapeTimeout: 10s
    ```
  - **Files to Create:**
    - `kubernetes/monitoring/servicemonitor-agents.yaml`
  - **Integration:** Prometheus Operator
  - **Monitoring:** Scrape success rate

- [ ] **6.1.2** Create PrometheusRule for agent alerts
  - **Configuration:**
    ```yaml
    # Path: kubernetes/monitoring/prometheus-rules-agents.yaml
    apiVersion: monitoring.coreos.com/v1
    kind: PrometheusRule
    metadata:
      name: greenlang-agents-alerts
      namespace: monitoring
    spec:
      groups:
        - name: greenlang-agents.rules
          rules:
            - alert: AgentHighErrorRate
              expr: |
                sum(rate(agent_request_errors_total[5m])) by (agent)
                / sum(rate(agent_requests_total[5m])) by (agent)
                > 0.01
              for: 5m
              labels:
                severity: critical
                team: platform
              annotations:
                summary: "High error rate on {{ $labels.agent }}"
                description: "Agent {{ $labels.agent }} error rate is {{ $value | humanizePercentage }}"

            - alert: AgentHighLatency
              expr: |
                histogram_quantile(0.95,
                  sum(rate(agent_request_duration_seconds_bucket[5m])) by (le, agent)
                ) > 0.5
              for: 5m
              labels:
                severity: warning
              annotations:
                summary: "High latency on {{ $labels.agent }}"
                description: "Agent {{ $labels.agent }} P95 latency is {{ $value }}s"

            - alert: AgentPodNotReady
              expr: |
                kube_deployment_status_replicas_ready{deployment=~"fuel-analyzer|carbon-intensity|energy-performance"}
                < kube_deployment_spec_replicas{deployment=~"fuel-analyzer|carbon-intensity|energy-performance"}
              for: 5m
              labels:
                severity: warning
              annotations:
                summary: "Agent pods not ready"

            - alert: AgentHPAMaxReplicas
              expr: |
                kube_horizontalpodautoscaler_status_current_replicas
                == kube_horizontalpodautoscaler_spec_max_replicas
              for: 15m
              labels:
                severity: warning
              annotations:
                summary: "HPA at maximum replicas"

            - alert: AgentCPUHigh
              expr: |
                sum(rate(container_cpu_usage_seconds_total{namespace="greenlang-agents"}[5m])) by (pod)
                / sum(container_spec_cpu_quota{namespace="greenlang-agents"}/container_spec_cpu_period{namespace="greenlang-agents"}) by (pod)
                > 0.8
              for: 10m
              labels:
                severity: warning
              annotations:
                summary: "High CPU usage on {{ $labels.pod }}"

            - alert: AgentMemoryHigh
              expr: |
                sum(container_memory_working_set_bytes{namespace="greenlang-agents"}) by (pod)
                / sum(container_spec_memory_limit_bytes{namespace="greenlang-agents"}) by (pod)
                > 0.85
              for: 10m
              labels:
                severity: warning
              annotations:
                summary: "High memory usage on {{ $labels.pod }}"
    ```
  - **Files to Create:**
    - `kubernetes/monitoring/prometheus-rules-agents.yaml`
  - **Integration:** Alertmanager
  - **Monitoring:** Alert firing rate

### 6.2 Grafana Dashboards

**Component:** Visualization
**Tool:** Grafana
**Priority:** P1 - High

#### Tasks:

- [ ] **6.2.1** Create Agent Overview Dashboard
  - **Configuration:**
    ```json
    // Dashboard: GreenLang Agents Overview
    // Panels:
    // - Total Requests (counter)
    // - Request Rate (rate over time)
    // - Error Rate (percentage)
    // - P50/P95/P99 Latency (histogram)
    // - Active Pods (gauge)
    // - CPU/Memory Usage (time series)
    ```
  - **Files to Create:**
    - `monitoring/grafana/dashboards/agents-overview.json`
  - **Key Metrics:**
    - `agent_requests_total`
    - `agent_request_duration_seconds`
    - `agent_request_errors_total`
    - `agent_active_connections`

- [ ] **6.2.2** Create Per-Agent Detail Dashboard
  - **Configuration:**
    - Variable: `$agent` (dropdown)
    - Panels: Request breakdown by tool, error details, latency histogram
  - **Files to Create:**
    - `monitoring/grafana/dashboards/agent-detail.json`
  - **Key Metrics:**
    - `agent_tool_invocations_total{agent="$agent"}`
    - `agent_tool_duration_seconds{agent="$agent"}`
    - `agent_provenance_hashes_total{agent="$agent"}`

- [ ] **6.2.3** Create SLO Dashboard
  - **Configuration:**
    - Availability SLO: 99.9%
    - Latency SLO: P95 < 500ms
    - Error Rate SLO: < 0.5%
    - Error Budget burn rate
  - **Files to Create:**
    - `monitoring/grafana/dashboards/slo-dashboard.json`
  - **Integration:** SLO tracking and alerting

- [ ] **6.2.4** Create Cost Tracking Dashboard
  - **Configuration:**
    - Compute cost per agent
    - LLM API cost tracking
    - Storage cost breakdown
    - Cost trend over time
  - **Files to Create:**
    - `monitoring/grafana/dashboards/cost-tracking.json`
  - **Integration:** AWS Cost Explorer API

### 6.3 Logging Configuration

**Component:** Log Aggregation
**Tool:** ELK Stack (Elasticsearch, Logstash, Kibana) or Loki
**Priority:** P1 - High

#### Tasks:

- [ ] **6.3.1** Configure structured logging in agents
  - **Configuration:**
    ```python
    # Logging configuration for agents
    import structlog

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    ```
  - **Files to Modify:**
    - `generated/fuel_analyzer_agent/logging_config.py`
    - `generated/carbon_intensity_v1/logging_config.py`
    - `generated/energy_performance_v1/logging_config.py`
  - **Log Format:** JSON with trace_id, agent_name, tool_name

- [ ] **6.3.2** Deploy Fluent Bit for log shipping
  - **Configuration:**
    ```yaml
    # Path: kubernetes/logging/fluent-bit-config.yaml
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: fluent-bit-config
      namespace: logging
    data:
      fluent-bit.conf: |
        [SERVICE]
            Flush         5
            Log_Level     info
            Parsers_File  parsers.conf

        [INPUT]
            Name              tail
            Path              /var/log/containers/*greenlang-agents*.log
            Parser            docker
            Tag               kube.*
            Refresh_Interval  5

        [FILTER]
            Name   kubernetes
            Match  kube.*
            Merge_Log On
            Keep_Log Off
            K8S-Logging.Parser On

        [OUTPUT]
            Name            es
            Match           *
            Host            elasticsearch.logging.svc
            Port            9200
            Index           greenlang-agents
            Type            _doc
    ```
  - **Files to Create:**
    - `kubernetes/logging/fluent-bit-config.yaml`
    - `kubernetes/logging/fluent-bit-daemonset.yaml`
  - **Integration:** Elasticsearch cluster
  - **Monitoring:** Log ingestion rate

- [ ] **6.3.3** Create Kibana index patterns and dashboards
  - **Configuration:**
    - Index pattern: `greenlang-agents-*`
    - Dashboards: Error logs, Request traces, Agent activity
  - **Files to Create:**
    - `monitoring/kibana/index-patterns.json`
    - `monitoring/kibana/dashboards/agent-logs.json`

### 6.4 Distributed Tracing

**Component:** Request Tracing
**Tool:** OpenTelemetry, Jaeger/Tempo
**Priority:** P2 - Medium

#### Tasks:

- [ ] **6.4.1** Integrate OpenTelemetry SDK in agents
  - **Configuration:**
    ```python
    # OpenTelemetry configuration
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)

    otlp_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317")
    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))
    ```
  - **Files to Modify:**
    - `generated/*/tracing.py`
  - **Integration:** OTLP collector

- [ ] **6.4.2** Deploy OpenTelemetry Collector
  - **Configuration:**
    ```yaml
    # Path: kubernetes/tracing/otel-collector.yaml
    apiVersion: opentelemetry.io/v1alpha1
    kind: OpenTelemetryCollector
    metadata:
      name: otel-collector
      namespace: monitoring
    spec:
      config: |
        receivers:
          otlp:
            protocols:
              grpc:
                endpoint: 0.0.0.0:4317
        processors:
          batch:
            timeout: 1s
        exporters:
          jaeger:
            endpoint: jaeger-collector:14250
        service:
          pipelines:
            traces:
              receivers: [otlp]
              processors: [batch]
              exporters: [jaeger]
    ```
  - **Files to Create:**
    - `kubernetes/tracing/otel-collector.yaml`

### 6.5 Alerting Integration

**Component:** Alert Routing
**Tool:** Alertmanager, PagerDuty, Slack
**Priority:** P1 - High

#### Tasks:

- [ ] **6.5.1** Configure Alertmanager routing
  - **Configuration:**
    ```yaml
    # Path: kubernetes/monitoring/alertmanager-config.yaml
    apiVersion: v1
    kind: Secret
    metadata:
      name: alertmanager-config
      namespace: monitoring
    stringData:
      alertmanager.yaml: |
        global:
          resolve_timeout: 5m
          slack_api_url: '<slack-webhook-url>'
          pagerduty_url: 'https://events.pagerduty.com/v2/enqueue'

        route:
          group_by: ['alertname', 'severity']
          group_wait: 30s
          group_interval: 5m
          repeat_interval: 4h
          receiver: 'slack-notifications'
          routes:
            - match:
                severity: critical
              receiver: 'pagerduty-critical'
            - match:
                severity: warning
              receiver: 'slack-notifications'

        receivers:
          - name: 'pagerduty-critical'
            pagerduty_configs:
              - service_key: '<pagerduty-service-key>'
                severity: critical

          - name: 'slack-notifications'
            slack_configs:
              - channel: '#greenlang-alerts'
                send_resolved: true
                title: '{{ .Status | toUpper }}: {{ .CommonAnnotations.summary }}'
                text: '{{ .CommonAnnotations.description }}'
    ```
  - **Files to Create:**
    - `kubernetes/monitoring/alertmanager-config.yaml`
  - **Integration:** PagerDuty, Slack
  - **Monitoring:** Alert delivery rate

---

## Phase 7: Security Scanning

### 7.1 Container Security

**Component:** Image Vulnerability Scanning
**Tool:** Trivy, Snyk
**Priority:** P0 - Critical Path

#### Tasks:

- [ ] **7.1.1** Integrate Trivy in CI pipeline
  - **Configuration:** (Already covered in 4.1.2)
  - **Acceptance Criteria:** Zero CRITICAL/HIGH CVEs in production images

- [ ] **7.1.2** Configure Trivy Operator for runtime scanning
  - **Configuration:**
    ```yaml
    # Path: kubernetes/security/trivy-operator.yaml
    apiVersion: aquasecurity.github.io/v1alpha1
    kind: ClusterComplianceReport
    metadata:
      name: cis-benchmark
    spec:
      cron: "0 0 * * *"
      compliance:
        name: cis
        version: "1.6.0"
    ```
  - **Files to Create:**
    - `kubernetes/security/trivy-operator.yaml`
  - **Monitoring:** Vulnerability count over time
  - **Alert:** New CRITICAL vulnerability detected

### 7.2 Dependency Scanning

**Component:** Supply Chain Security
**Tool:** Snyk, Dependabot
**Priority:** P1 - High

#### Tasks:

- [ ] **7.2.1** Configure Snyk integration
  - **Configuration:**
    ```yaml
    # Path: .github/workflows/snyk.yml
    name: Snyk Security Scan
    on:
      push:
        branches: [main]
      schedule:
        - cron: '0 6 * * *'

    jobs:
      snyk:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - name: Run Snyk
            uses: snyk/actions/python@master
            env:
              SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
            with:
              args: --severity-threshold=high
    ```
  - **Files to Create:**
    - `.github/workflows/snyk.yml`
  - **Monitoring:** Dependency vulnerability count

- [ ] **7.2.2** Configure Dependabot
  - **Configuration:**
    ```yaml
    # Path: .github/dependabot.yml
    version: 2
    updates:
      - package-ecosystem: "pip"
        directory: "/"
        schedule:
          interval: "weekly"
        open-pull-requests-limit: 10
        groups:
          python-minor:
            patterns:
              - "*"
            update-types:
              - "minor"
              - "patch"
    ```
  - **Files to Create:**
    - `.github/dependabot.yml`

### 7.3 Static Code Analysis

**Component:** SAST
**Tool:** Bandit, Semgrep
**Priority:** P1 - High

#### Tasks:

- [ ] **7.3.1** Configure Bandit for Python security
  - **Configuration:**
    ```yaml
    # Path: .bandit
    [bandit]
    exclude: tests,venv
    skips: B101,B601
    ```
  - **Files to Create:**
    - `.bandit`
  - **Integration:** CI pipeline

- [ ] **7.3.2** Configure Semgrep rules
  - **Configuration:**
    ```yaml
    # Path: .semgrep.yml
    rules:
      - id: hardcoded-secret
        patterns:
          - pattern: $X = "..."
          - metavariable-regex:
              metavariable: $X
              regex: (api_key|secret|password|token)
        message: Potential hardcoded secret
        severity: ERROR
    ```
  - **Files to Create:**
    - `.semgrep.yml`

---

## Phase 8: Cost Optimization

### 8.1 Resource Right-Sizing

**Component:** Resource Optimization
**Tool:** VPA, Goldilocks
**Priority:** P2 - Medium

#### Tasks:

- [ ] **8.1.1** Deploy Vertical Pod Autoscaler (VPA)
  - **Configuration:**
    ```yaml
    # Path: kubernetes/autoscaling/vpa.yaml
    apiVersion: autoscaling.k8s.io/v1
    kind: VerticalPodAutoscaler
    metadata:
      name: fuel-analyzer-vpa
      namespace: greenlang-agents
    spec:
      targetRef:
        apiVersion: apps/v1
        kind: Deployment
        name: fuel-analyzer
      updatePolicy:
        updateMode: "Off"  # Recommendation only
      resourcePolicy:
        containerPolicies:
          - containerName: fuel-analyzer
            minAllowed:
              cpu: 100m
              memory: 128Mi
            maxAllowed:
              cpu: 2
              memory: 4Gi
    ```
  - **Files to Create:**
    - `kubernetes/autoscaling/vpa.yaml`
  - **Monitoring:** VPA recommendations vs actual usage
  - **Review:** Monthly resource right-sizing review

- [ ] **8.1.2** Deploy Goldilocks for recommendations
  - **Configuration:**
    ```bash
    helm install goldilocks fairwinds-stable/goldilocks \
      --namespace goldilocks \
      --set vpa.enabled=true
    ```
  - **Integration:** Goldilocks dashboard
  - **Monitoring:** Cost savings recommendations

### 8.2 Spot Instance Usage

**Component:** Cost Reduction
**Tool:** Karpenter, Spot Instance Advisor
**Priority:** P2 - Medium

#### Tasks:

- [ ] **8.2.1** Configure Karpenter for spot instances
  - **Configuration:**
    ```yaml
    # Path: kubernetes/autoscaling/karpenter-provisioner.yaml
    apiVersion: karpenter.sh/v1alpha5
    kind: Provisioner
    metadata:
      name: agent-spot
    spec:
      requirements:
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["spot", "on-demand"]
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
        - key: node.kubernetes.io/instance-type
          operator: In
          values: ["c6i.xlarge", "c6i.2xlarge", "m6i.xlarge"]
      limits:
        resources:
          cpu: 1000
      providerRef:
        name: default
      ttlSecondsAfterEmpty: 300
    ```
  - **Files to Create:**
    - `kubernetes/autoscaling/karpenter-provisioner.yaml`
  - **Monitoring:** Spot instance percentage, interruption rate
  - **Target:** 60% spot instances for agent workloads

---

## Phase 9: Disaster Recovery

### 9.1 Backup Procedures

**Component:** Data Protection
**Tool:** Velero, AWS Backup
**Priority:** P1 - High

#### Tasks:

- [ ] **9.1.1** Deploy Velero for Kubernetes backups
  - **Configuration:**
    ```bash
    velero install \
      --provider aws \
      --plugins velero/velero-plugin-for-aws:v1.8.0 \
      --bucket greenlang-backups \
      --secret-file ./credentials-velero \
      --backup-location-config region=us-east-1 \
      --snapshot-location-config region=us-east-1
    ```
  - **Files to Create:**
    - `kubernetes/backup/velero-schedule.yaml`
  - **Schedule:** Daily at 2 AM UTC
  - **Retention:** 30 days

- [ ] **9.1.2** Create Velero backup schedule
  - **Configuration:**
    ```yaml
    # Path: kubernetes/backup/velero-schedule.yaml
    apiVersion: velero.io/v1
    kind: Schedule
    metadata:
      name: greenlang-agents-daily
      namespace: velero
    spec:
      schedule: "0 2 * * *"
      template:
        includedNamespaces:
          - greenlang-agents
        ttl: 720h
        storageLocation: default
        volumeSnapshotLocations:
          - default
    ```
  - **Files to Create:**
    - `kubernetes/backup/velero-schedule.yaml`
  - **Monitoring:** Backup success rate
  - **Alert:** Backup failure

### 9.2 Restore Procedures

**Component:** Recovery
**Tool:** Velero, RDS snapshot restore
**Priority:** P1 - High

#### Tasks:

- [ ] **9.2.1** Create restore runbook
  - **Documentation:**
    ```markdown
    # Restore Runbook: GreenLang Agents

    ## Prerequisites
    - kubectl access to target cluster
    - Velero CLI installed
    - AWS credentials with backup access

    ## Steps

    1. List available backups:
       ```bash
       velero backup get
       ```

    2. Describe backup to verify contents:
       ```bash
       velero backup describe <backup-name> --details
       ```

    3. Restore namespace:
       ```bash
       velero restore create --from-backup <backup-name>
       ```

    4. Monitor restore progress:
       ```bash
       velero restore describe <restore-name>
       ```

    5. Verify pods are running:
       ```bash
       kubectl get pods -n greenlang-agents
       ```

    6. Run smoke tests:
       ```bash
       ./scripts/smoke-test.sh
       ```

    ## RTO: 1 hour
    ## RPO: 24 hours (daily backup)
    ```
  - **Files to Create:**
    - `docs/runbooks/restore-procedure.md`

- [ ] **9.2.2** Test restore procedure
  - **Test:** Monthly DR drill
  - **Acceptance Criteria:** RTO < 1 hour, all services healthy post-restore
  - **Documentation:** DR drill report template

---

## Summary Checklist

### Phase 1: Docker Containerization
- [ ] 1.1.1 Create base Docker image
- [ ] 1.1.2 Create multi-stage build template
- [ ] 1.1.3 Configure BuildKit caching
- [ ] 1.2.1 Dockerfile for Fuel Analyzer
- [ ] 1.2.2 Dockerfile for CBAM Carbon Intensity
- [ ] 1.2.3 Dockerfile for Building Energy Performance
- [ ] 1.2.4 Docker Compose for local development
- [ ] 1.3.1 Non-root user configuration
- [ ] 1.3.2 Read-only root filesystem
- [ ] 1.3.3 Trivy scanning in CI/CD
- [ ] 1.3.4 Image signing (Cosign)

### Phase 2: Kubernetes Manifests
- [ ] 2.1.1-2.1.5 Namespace and RBAC
- [ ] 2.2.1-2.2.4 ConfigMaps
- [ ] 2.3.1-2.3.4 Secrets Management
- [ ] 2.4.1-2.4.4 Deployments and PDBs
- [ ] 2.5.1 Services
- [ ] 2.6.1-2.6.2 Ingress and TLS
- [ ] 2.7.1-2.7.2 Network Policies
- [ ] 2.8.1-2.8.3 HPA Configuration

### Phase 3: Helm Charts
- [ ] 3.1.1-3.1.6 Helm chart structure and Helmfile

### Phase 4: CI/CD Pipeline
- [ ] 4.1.1 PR validation workflow
- [ ] 4.1.2 Docker build and push workflow
- [ ] 4.1.3 Deployment workflow
- [ ] 4.1.4 Rollback workflow
- [ ] 4.2.1 ArgoCD Application (optional)

### Phase 5: Terraform
- [ ] 5.1.1-5.1.7 Terraform modules and environments

### Phase 6: Monitoring
- [ ] 6.1.1-6.1.2 Prometheus configuration
- [ ] 6.2.1-6.2.4 Grafana dashboards
- [ ] 6.3.1-6.3.3 Logging configuration
- [ ] 6.4.1-6.4.2 Distributed tracing
- [ ] 6.5.1 Alerting integration

### Phase 7: Security Scanning
- [ ] 7.1.1-7.1.2 Container security
- [ ] 7.2.1-7.2.2 Dependency scanning
- [ ] 7.3.1-7.3.2 Static code analysis

### Phase 8: Cost Optimization
- [ ] 8.1.1-8.1.2 Resource right-sizing
- [ ] 8.2.1 Spot instance configuration

### Phase 9: Disaster Recovery
- [ ] 9.1.1-9.1.2 Backup procedures
- [ ] 9.2.1-9.2.2 Restore procedures

---

## Files to Create Summary

### Docker Files (5 files)
- `docker/base/Dockerfile.base`
- `generated/fuel_analyzer_agent/Dockerfile`
- `generated/carbon_intensity_v1/Dockerfile`
- `generated/energy_performance_v1/Dockerfile`
- `docker-compose.yml`

### Kubernetes Manifests (15 files)
- `kubernetes/manifests/namespace.yaml`
- `kubernetes/manifests/serviceaccount.yaml`
- `kubernetes/manifests/rbac.yaml`
- `kubernetes/manifests/resource-quota.yaml`
- `kubernetes/manifests/limit-range.yaml`
- `kubernetes/manifests/configmap-*.yaml` (4 files)
- `kubernetes/manifests/secret-store.yaml`
- `kubernetes/manifests/external-secret-*.yaml` (2 files)
- `kubernetes/manifests/deployment-*.yaml` (3 files)
- `kubernetes/manifests/pdb.yaml`
- `kubernetes/manifests/services.yaml`
- `kubernetes/manifests/ingress.yaml`
- `kubernetes/manifests/certificate.yaml`
- `kubernetes/manifests/network-policy-*.yaml` (2 files)
- `kubernetes/manifests/hpa.yaml`

### Helm Charts (20+ files)
- `helm/greenlang-agents/Chart.yaml`
- `helm/greenlang-agents/values*.yaml` (4 files)
- `helm/greenlang-agents/templates/*.yaml` (7 files)
- `helm/greenlang-agents/charts/*/` (3 sub-charts, 5 files each)
- `helmfile.yaml`

### CI/CD Workflows (5 files)
- `.github/workflows/pr-validation.yml`
- `.github/workflows/docker-build.yml`
- `.github/workflows/deploy.yml`
- `.github/workflows/rollback.yml`
- `.github/workflows/snyk.yml`

### Terraform (15 files)
- `terraform/modules/*/main.tf` (6 modules)
- `terraform/modules/*/variables.tf` (6 modules)
- `terraform/modules/*/outputs.tf` (6 modules)
- `terraform/environments/prod/*.tf` (3 files)

### Monitoring (10 files)
- `kubernetes/monitoring/servicemonitor-agents.yaml`
- `kubernetes/monitoring/prometheus-rules-agents.yaml`
- `kubernetes/monitoring/alertmanager-config.yaml`
- `monitoring/grafana/dashboards/*.json` (4 dashboards)
- `kubernetes/logging/fluent-bit-*.yaml` (2 files)
- `kubernetes/tracing/otel-collector.yaml`

### Security (5 files)
- `.bandit`
- `.semgrep.yml`
- `.github/dependabot.yml`
- `kubernetes/security/trivy-operator.yaml`

### Backup/DR (3 files)
- `kubernetes/backup/velero-schedule.yaml`
- `docs/runbooks/restore-procedure.md`
- `kubernetes/autoscaling/vpa.yaml`

---

**Total Configuration Files:** ~75 files
**Estimated Implementation Time:** 2-3 weeks
**Team Required:** 2-3 DevOps Engineers

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-03 | GL-DevOpsEngineer | Initial detailed deployment to-do list |

---

**End of Document**
