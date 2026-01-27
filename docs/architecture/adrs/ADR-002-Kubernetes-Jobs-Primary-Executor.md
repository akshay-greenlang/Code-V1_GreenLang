# ADR-002: Kubernetes Jobs as Primary Execution Backend

**Date:** 2026-01-27
**Status:** Accepted
**Deciders:** Platform Engineering, SRE/Ops, Architecture Team
**Consulted:** Security Team, Cloud Infrastructure, Agent Development Teams

---

## Context

### Problem Statement
GL-FOUND-X-001 (GreenLang Orchestrator) requires a reliable, isolated execution backend for running agent tasks. The execution backend must support:
- Reliable task execution with automatic retries
- Resource isolation between tenants and agents
- Horizontal scaling for variable workloads
- Integration with cloud-native observability

### Current Situation
- **Agent Count:** 402 agents requiring execution infrastructure
- **Concurrency Target:** 200+ concurrent step executions
- **Multi-tenant:** Enterprise customers require strict isolation
- **Scale Variability:** Workloads vary from 10 to 10,000 daily runs

### Business Impact
- **Reliability:** Production SLA of 99.9% availability for orchestration
- **Cost Efficiency:** Pay-per-use scaling reduces infrastructure costs
- **Compliance:** Resource isolation required for data residency
- **Operational Excellence:** Unified platform reduces operational burden

---

## Decision

### What We're Implementing
**Kubernetes Jobs** as the primary execution backend for all agent invocations in GL-FOUND-X-001.

### Core Implementation

1. **Job-per-Step Model**
   - Each DAG step executes as a single Kubernetes Job
   - Job spec generated from GLIP v1 invocation manifest
   - Automatic cleanup via TTL after completion

2. **Job Specification Template**
   ```yaml
   apiVersion: batch/v1
   kind: Job
   metadata:
     name: "glip-${run_id}-${step_id}"
     namespace: "${tenant_namespace}"
     labels:
       app: greenlang-agent
       run_id: "${run_id}"
       step_id: "${step_id}"
       agent: "${agent_name}"
   spec:
     ttlSecondsAfterFinished: 3600
     backoffLimit: 0  # Orchestrator manages retries
     activeDeadlineSeconds: ${timeout_seconds}
     template:
       spec:
         restartPolicy: Never
         serviceAccountName: "glip-agent-${tenant}"
         securityContext:
           runAsNonRoot: true
           runAsUser: 1000
           fsGroup: 1000
         containers:
           - name: agent
             image: "${agent_image}"
             resources:
               requests:
                 cpu: "${cpu_request}"
                 memory: "${memory_request}"
               limits:
                 cpu: "${cpu_limit}"
                 memory: "${memory_limit}"
             env:
               - name: GLIP_INVOCATION_ID
                 value: "${invocation_id}"
               - name: GLIP_INPUT_MANIFEST
                 value: "/glip/inputs/manifest.json"
               - name: GLIP_OUTPUT_DIR
                 value: "/glip/outputs"
             volumeMounts:
               - name: glip-inputs
                 mountPath: /glip/inputs
                 readOnly: true
               - name: glip-outputs
                 mountPath: /glip/outputs
         volumes:
           - name: glip-inputs
             configMap:
               name: "glip-inputs-${invocation_id}"
           - name: glip-outputs
             emptyDir: {}
   ```

3. **Executor Architecture**
   ```
   Orchestrator Scheduler
          |
          v
   +------------------+
   | K8s Job Executor |
   +------------------+
          |
          +---> K8s API Server ---> Job Controller
          |                              |
          +---> Job Watcher <------------+
          |
          v
   Status Updates to State Store
   ```

### Technology Stack
- **Kubernetes:** 1.28+ (Job TTL, Pod Security Standards)
- **Container Runtime:** containerd
- **Networking:** Cilium/Calico with NetworkPolicies
- **Storage:** CSI drivers for artifact volumes

### Code Location
- `greenlang/orchestrator/executors/kubernetes/`
  - `job_executor.py` - Job creation and management
  - `job_watcher.py` - Status monitoring via watch API
  - `resource_calculator.py` - Resource hint to K8s resource mapping
  - `namespace_manager.py` - Tenant namespace provisioning

---

## Rationale

### Why Kubernetes Jobs

**1. Cloud-Native and Portable**
- Runs on any Kubernetes cluster (EKS, GKE, AKS, on-prem)
- No vendor lock-in; portable across cloud providers
- Aligns with GreenLang's cloud-native architecture

**2. Built-in Reliability**
- Job controller ensures task completion
- activeDeadlineSeconds provides hard timeouts
- Pod eviction handling with graceful termination

**3. Resource Isolation**
- Namespace-based tenant isolation
- Resource quotas prevent noisy neighbors
- Pod security contexts enforce least privilege

**4. Auto-Scaling**
- Cluster autoscaler adds nodes on demand
- Scale to zero when idle (cost savings)
- Handles burst workloads automatically

**5. Observability Integration**
- Native integration with Prometheus metrics
- Pod logs available via Kubernetes logging
- Distributed tracing with OpenTelemetry sidecar

**6. Ecosystem Maturity**
- Battle-tested at scale (used by Airflow, Argo, Tekton)
- Rich tooling for debugging and operations
- Large community and support resources

---

## Alternatives Considered

### Alternative 1: Worker Pool (Celery/RQ)
**Pros:**
- Lower latency (no container startup)
- Simpler for small deployments
- Familiar to Python developers

**Cons:**
- Workers must pre-install all agent dependencies
- Harder to isolate resources between tasks
- Manual scaling configuration required
- Single language (Python) bias

**Why Rejected:** Does not provide sufficient isolation for multi-tenant workloads. Dependency management becomes complex with 402 agents.

### Alternative 2: Serverless Functions (AWS Lambda, Cloud Functions)
**Pros:**
- True scale-to-zero
- Sub-second cold starts for some runtimes
- Managed infrastructure

**Cons:**
- Execution time limits (15 min Lambda, 9 min Cloud Functions)
- Memory limits (10GB Lambda)
- Vendor lock-in
- Custom container support limited
- Cold start latency for large containers

**Why Rejected:** Execution time limits incompatible with long-running agent tasks. Vendor lock-in conflicts with portability requirements.

### Alternative 3: Hybrid (K8s Jobs + Worker Pool)
**Pros:**
- Low latency for simple tasks (worker pool)
- Isolation for complex tasks (K8s Jobs)

**Cons:**
- Two execution paths to maintain
- Routing logic adds complexity
- Inconsistent operational model
- Harder to reason about capacity

**Why Rejected:** Operational complexity outweighs latency benefits. Consistency in execution model simplifies operations and debugging.

### Alternative 4: Argo Workflows
**Pros:**
- Purpose-built for DAG execution on K8s
- Rich workflow primitives
- Active community

**Cons:**
- Another dependency to manage
- Learning curve for operators
- May conflict with orchestrator's own DAG engine
- Less control over execution semantics

**Why Rejected:** Overlaps with GL-FOUND-X-001's own DAG orchestration. Direct K8s Jobs provide more control and fewer dependencies.

---

## Consequences

### Positive
- **Isolation:** Strong resource isolation via namespaces and pods
- **Portability:** Runs on any Kubernetes cluster
- **Scalability:** Cluster autoscaler handles variable load
- **Reliability:** Job controller ensures task completion
- **Observability:** Native integration with K8s monitoring stack
- **Ecosystem:** Leverage K8s tooling (kubectl, k9s, Lens)

### Negative
- **K8s Expertise Required:** Operations team must be K8s proficient
- **Cold Start Latency:** Job creation adds 5-15 seconds overhead
- **Cluster Management:** Requires cluster administration (or managed K8s)
- **Complexity:** K8s adds operational complexity vs. simple VM deployments
- **Cost:** Control plane costs (managed K8s) and node overhead

### Neutral
- **Container Requirement:** All agents must be containerized (also a GLIP v1 requirement)
- **Infrastructure Team:** May require dedicated platform team for cluster operations

---

## Implementation Plan

### Phase 1: Cluster Setup (Week 1-2)
1. Provision Kubernetes clusters (dev, staging, prod)
2. Configure cluster autoscaler
3. Set up namespace-per-tenant model
4. Implement network policies for isolation

### Phase 2: Executor Development (Week 3-4)
1. Implement Job executor with GLIP v1 integration
2. Build Job watcher for status updates
3. Add resource hint to K8s resource mapping
4. Implement graceful termination handling

### Phase 3: Observability (Week 5-6)
1. Deploy Prometheus for metrics collection
2. Configure log aggregation (Loki/ELK)
3. Set up distributed tracing (Jaeger/Tempo)
4. Build Grafana dashboards for Job execution

### Phase 4: Production Hardening (Week 7-8)
1. Implement circuit breakers for API failures
2. Add Pod Disruption Budgets
3. Configure resource quotas per namespace
4. Chaos testing (pod failures, node eviction)

---

## Compliance & Security

### Security Considerations
- **Pod Security Standards:** Enforce "restricted" profile
- **RBAC:** Minimal permissions for Job creation
- **Network Policies:** Deny egress by default, allow specific endpoints
- **Secrets:** Kubernetes Secrets with encryption at rest
- **Image Security:** Only allow signed images from trusted registries

### Operational Considerations
- **Monitoring:** Prometheus alerts for Job failures, queue depth
- **Logging:** Centralized log collection with run_id correlation
- **Capacity:** Monitor node utilization, autoscaler effectiveness
- **Cost:** Track per-tenant resource consumption

---

## Capacity Planning

### Resource Estimates

| Metric | MVP Target | GA Target |
|--------|------------|-----------|
| Concurrent Jobs | 50 | 200+ |
| Daily Job Executions | 1,000 | 10,000+ |
| Avg Job Duration | 5 min | 5 min |
| Node Pool Size | 5-20 nodes | 20-100 nodes |
| Cluster Autoscaler Range | 3-30 nodes | 10-200 nodes |

### Node Sizing
- **Agent Nodes:** 4 vCPU, 16GB RAM (general purpose)
- **Large Agent Nodes:** 16 vCPU, 64GB RAM (data processing)
- **GPU Nodes:** As needed for ML agents (spot instances)

---

## Migration Plan

### Short-term (0-6 months)
- Deploy K8s Job executor for new GLIP v1 agents
- Legacy HTTP agents use adapter (ADR-007)
- Validate isolation and scaling

### Medium-term (6-12 months)
- Migrate all agents to K8s Job execution
- Optimize autoscaler configuration
- Implement advanced scheduling (priority, preemption)

### Long-term (12+ months)
- Evaluate Kubernetes improvements (Job indexing, etc.)
- Consider multi-cluster for disaster recovery
- Explore GPU/TPU scheduling for ML workloads

---

## Links & References

- **PRD:** GL-FOUND-X-001 GreenLang Orchestrator
- **Related ADRs:** ADR-001 (GLIP v1), ADR-004 (S3 Artifacts)
- **K8s Documentation:** [Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- **Cluster Autoscaler:** [Docs](https://github.com/kubernetes/autoscaler)

---

## Updates

### 2026-01-27 - Status: Accepted
ADR approved by Platform Engineering and SRE. Cluster provisioning scheduled for Q1 2026.

---

**Template Version:** 1.0
**Last Updated:** 2026-01-27
**ADR Author:** Platform Architecture Team
**Reviewers:** SRE Team, Security Team, Cloud Infrastructure
