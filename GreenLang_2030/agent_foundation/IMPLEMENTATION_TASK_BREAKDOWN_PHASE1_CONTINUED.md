# GreenLang Agent Factory - Phase 1 Continued (Epic 1.3-1.10)

**Continuation of IMPLEMENTATION_TASK_BREAKDOWN.md**

---

## EPIC 1.3: HIGH AVAILABILITY (60 person-weeks)

### Task 1.3.1: Multi-AZ Kubernetes Deployment
**Priority:** P0 (BLOCKING)
**Effort:** 40 hours
**Dependencies:** Task 1.2.1 (Database), Task 1.2.2 (Redis)
**Assignee:** DevOps Engineer (Senior) + Cloud Architect

#### Subtasks:
- [ ] 1. Create EKS cluster configuration with 3 availability zones - 4 hours
- [ ] 2. Set up pod anti-affinity rules to spread across AZs - 3 hours
- [ ] 3. Configure 9 pods (3 per AZ) with resource limits - 4 hours
- [ ] 4. Implement rolling update strategy (maxUnavailable=0) - 3 hours
- [ ] 5. Set up Network Load Balancer (Layer 4) - 4 hours
- [ ] 6. Configure cross-zone load balancing - 3 hours
- [ ] 7. Implement session affinity (ClientIP) - 3 hours
- [ ] 8. Set up autoscaling policies (HPA, VPA, Cluster Autoscaler) - 6 hours
- [ ] 9. Create monitoring and alerting for cluster health - 5 hours
- [ ] 10. Document deployment procedures and runbooks - 5 hours

#### Files to Create:
- `C:\Users\aksha\Code-V1_GreenLang\infrastructure\kubernetes\cluster\eks-cluster.yaml` - EKS cluster definition
- `C:\Users\aksha\Code-V1_GreenLang\infrastructure\kubernetes\deployments\agent-foundation-deployment.yaml` - Deployment manifest
- `C:\Users\aksha\Code-V1_GreenLang\infrastructure\kubernetes\services\agent-foundation-service.yaml` - Service definition
- `C:\Users\aksha\Code-V1_GreenLang\infrastructure\kubernetes\autoscaling\hpa.yaml` - Horizontal Pod Autoscaler
- `C:\Users\aksha\Code-V1_GreenLang\infrastructure\kubernetes\autoscaling\vpa.yaml` - Vertical Pod Autoscaler
- `C:\Users\aksha\Code-V1_GreenLang\infrastructure\kubernetes\autoscaling\cluster-autoscaler.yaml` - Cluster Autoscaler
- `C:\Users\aksha\Code-V1_GreenLang\infrastructure\kubernetes\network\network-policy.yaml` - Network policies
- `C:\Users\aksha\Code-V1_GreenLang\infrastructure\terraform\eks.tf` - Terraform EKS configuration
- `C:\Users\aksha\Code-V1_GreenLang\infrastructure\terraform\nlb.tf` - Network Load Balancer config

#### Kubernetes Manifests:
```yaml
# infrastructure/kubernetes/deployments/agent-foundation-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-foundation
  namespace: greenlang
  labels:
    app: agent-foundation
    version: v1.0.0
spec:
  replicas: 9
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0
      maxSurge: 3
  selector:
    matchLabels:
      app: agent-foundation
  template:
    metadata:
      labels:
        app: agent-foundation
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchExpressions:
                  - key: app
                    operator: In
                    values:
                      - agent-foundation
              topologyKey: topology.kubernetes.io/zone
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                    - key: app
                      operator: In
                      values:
                        - agent-foundation
                topologyKey: kubernetes.io/hostname

      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: agent-foundation

      containers:
        - name: agent-foundation
          image: greenlang/agent-foundation:latest
          imagePullPolicy: Always
          ports:
            - name: http
              containerPort: 8000
              protocol: TCP
            - name: metrics
              containerPort: 9090
              protocol: TCP

          env:
            - name: ENVIRONMENT
              value: "production"
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: POD_NAMESPACE
              valueFrom:
                fieldRef:
                  fieldPath: metadata.namespace
            - name: POD_IP
              valueFrom:
                fieldRef:
                  fieldPath: status.podIP

          envFrom:
            - configMapRef:
                name: agent-foundation-config
            - secretRef:
                name: agent-foundation-secrets

          resources:
            requests:
              cpu: "2000m"
              memory: "4Gi"
            limits:
              cpu: "4000m"
              memory: "8Gi"

          livenessProbe:
            httpGet:
              path: /health/live
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            successThreshold: 1
            failureThreshold: 3

          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
            timeoutSeconds: 3
            successThreshold: 1
            failureThreshold: 3

          startupProbe:
            httpGet:
              path: /health/startup
              port: 8000
            initialDelaySeconds: 0
            periodSeconds: 5
            timeoutSeconds: 3
            successThreshold: 1
            failureThreshold: 30

          volumeMounts:
            - name: config
              mountPath: /app/config
              readOnly: true
            - name: logs
              mountPath: /app/logs

      volumes:
        - name: config
          configMap:
            name: agent-foundation-config
        - name: logs
          emptyDir: {}

      serviceAccountName: agent-foundation
      terminationGracePeriodSeconds: 60

---
# infrastructure/kubernetes/autoscaling/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-foundation-hpa
  namespace: greenlang
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-foundation
  minReplicas: 9
  maxReplicas: 100
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
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "1000"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 50
          periodSeconds: 30
        - type: Pods
          value: 5
          periodSeconds: 30
      selectPolicy: Max
```

#### Infrastructure Changes (Terraform):
```hcl
# infrastructure/terraform/eks.tf
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "19.16.0"

  cluster_name    = "greenlang-production"
  cluster_version = "1.28"

  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  enable_irsa = true

  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  eks_managed_node_groups = {
    agent_foundation = {
      name = "agent-foundation"

      instance_types = ["c6i.2xlarge"] # 8 vCPU, 16GB RAM
      capacity_type  = "ON_DEMAND"

      min_size     = 9
      max_size     = 30
      desired_size = 9

      disk_size = 100
      disk_type = "gp3"

      labels = {
        workload = "agent-foundation"
      }

      taints = []

      update_config = {
        max_unavailable_percentage = 33
      }

      iam_role_additional_policies = {
        AmazonEBSCSIDriverPolicy = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
      }
    }
  }

  node_security_group_additional_rules = {
    ingress_self_all = {
      description = "Node to node all ports/protocols"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      self        = true
    }
    egress_all = {
      description      = "Node all egress"
      protocol         = "-1"
      from_port        = 0
      to_port          = 0
      type             = "egress"
      cidr_blocks      = ["0.0.0.0/0"]
      ipv6_cidr_blocks = ["::/0"]
    }
  }

  tags = {
    Environment = "production"
    Terraform   = "true"
  }
}

# infrastructure/terraform/nlb.tf
resource "aws_lb" "agent_foundation" {
  name               = "greenlang-agent-foundation"
  internal           = false
  load_balancer_type = "network"
  subnets            = module.vpc.public_subnets

  enable_deletion_protection       = true
  enable_cross_zone_load_balancing = true

  tags = {
    Name        = "GreenLang Agent Foundation NLB"
    Environment = "production"
  }
}

resource "aws_lb_target_group" "agent_foundation" {
  name        = "greenlang-agent-foundation"
  port        = 80
  protocol    = "TCP"
  target_type = "ip"
  vpc_id      = module.vpc.vpc_id

  health_check {
    enabled             = true
    interval            = 10
    port                = "traffic-port"
    protocol            = "HTTP"
    path                = "/health/ready"
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 2
  }

  deregistration_delay = 30

  stickiness {
    type    = "source_ip"
    enabled = true
  }

  tags = {
    Name = "GreenLang Agent Foundation Target Group"
  }
}

resource "aws_lb_listener" "agent_foundation_http" {
  load_balancer_arn = aws_lb.agent_foundation.arn
  port              = "80"
  protocol          = "TCP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.agent_foundation.arn
  }
}

resource "aws_lb_listener" "agent_foundation_https" {
  load_balancer_arn = aws_lb.agent_foundation.arn
  port              = "443"
  protocol          = "TLS"
  certificate_arn   = aws_acm_certificate.greenlang.arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.agent_foundation.arn
  }
}
```

#### Configuration Changes:
```yaml
# config/kubernetes_config.yaml
kubernetes:
  cluster:
    name: greenlang-production
    region: us-east-1
    availability_zones:
      - us-east-1a
      - us-east-1b
      - us-east-1c

  deployment:
    replicas: 9
    replica_distribution:
      us-east-1a: 3
      us-east-1b: 3
      us-east-1c: 3

  resources:
    requests:
      cpu: 2000m
      memory: 4Gi
    limits:
      cpu: 4000m
      memory: 8Gi

  autoscaling:
    horizontal:
      enabled: true
      min_replicas: 9
      max_replicas: 100
      target_cpu_percentage: 70
      target_memory_percentage: 80
      scale_up_stabilization: 0
      scale_down_stabilization: 300

    vertical:
      enabled: true
      update_mode: Auto
      min_allowed_cpu: 1000m
      max_allowed_cpu: 8000m
      min_allowed_memory: 2Gi
      max_allowed_memory: 16Gi

    cluster:
      enabled: true
      min_nodes: 9
      max_nodes: 30
      scale_down_delay_after_add: 10m
      scale_down_unneeded_time: 10m

  health_checks:
    liveness:
      path: /health/live
      port: 8000
      initial_delay: 30
      period: 10
      timeout: 5
      failure_threshold: 3

    readiness:
      path: /health/ready
      port: 8000
      initial_delay: 10
      period: 5
      timeout: 3
      failure_threshold: 3

    startup:
      path: /health/startup
      port: 8000
      initial_delay: 0
      period: 5
      timeout: 3
      failure_threshold: 30

  load_balancer:
    type: network
    scheme: internet-facing
    cross_zone: true
    session_affinity: source_ip
    deregistration_delay: 30
    health_check_interval: 10
```

#### Files to Modify:
- `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\base_agent.py` - Add health check endpoints
- Create: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\health\health_checks.py` - Health check implementation

#### API Changes:
```python
# New health check endpoints
# GET /health/live - Liveness probe (basic alive check)
# GET /health/ready - Readiness probe (database, Redis, Kafka checks)
# GET /health/startup - Startup probe (for slow-starting agents)

# C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\health\health_checks.py
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from typing import Dict
import asyncio

router = APIRouter()

@router.get("/health/live")
async def liveness_check() -> JSONResponse:
    """
    Liveness probe - Returns 200 if the application is alive.
    This is a basic check that doesn't verify dependencies.
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "alive", "timestamp": datetime.now().isoformat()}
    )

@router.get("/health/ready")
async def readiness_check() -> JSONResponse:
    """
    Readiness probe - Returns 200 if the application is ready to serve traffic.
    Checks all critical dependencies: database, Redis, Kafka.
    """
    health_status = {
        "status": "ready",
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }

    all_healthy = True

    # Check database
    try:
        async with database.engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        all_healthy = False

    # Check Redis
    try:
        await redis_client.ping()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
        all_healthy = False

    # Check Kafka (if enabled)
    if config.kafka_enabled:
        try:
            await kafka_producer.send_and_wait("health-check", b"ping")
            health_status["checks"]["kafka"] = "healthy"
        except Exception as e:
            health_status["checks"]["kafka"] = f"unhealthy: {str(e)}"
            all_healthy = False

    status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    health_status["status"] = "ready" if all_healthy else "not_ready"

    return JSONResponse(status_code=status_code, content=health_status)

@router.get("/health/startup")
async def startup_check() -> JSONResponse:
    """
    Startup probe - Returns 200 once the application has finished starting up.
    Kubernetes will not send traffic until this returns 200.
    """
    # Check if initialization is complete
    if not app_state.initialized:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "starting", "message": "Application still initializing"}
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "started", "timestamp": app_state.startup_time.isoformat()}
    )
```

#### Tests Required:
- [ ] Unit tests: `test_kubernetes_deployment.py` (10 test cases)
  - Test deployment YAML validation
  - Test pod anti-affinity rules
  - Test resource limits enforcement
  - Test HPA configuration
  - Test VPA configuration
  - Test cluster autoscaler configuration
  - Test health check endpoints
  - Test rolling update strategy
  - Test session affinity
  - Test network policies
- [ ] Integration tests: `test_ha_integration.py` (8 test cases)
  - Test pod distribution across AZs
  - Test rolling update without downtime
  - Test autoscaling under load
  - Test load balancer health checks
  - Test session affinity behavior
  - Test pod failure recovery
  - Test AZ failure simulation
  - Test cluster autoscaling

#### Acceptance Criteria:
- [ ] EKS cluster deployed with 3 AZs
- [ ] 9 pods distributed evenly (3 per AZ)
- [ ] Rolling updates complete with zero downtime
- [ ] HPA scales from 9 to 100 pods based on load
- [ ] VPA provides resource recommendations
- [ ] Cluster autoscaler adds/removes nodes automatically
- [ ] Health checks pass with <100ms latency
- [ ] Load balancer distributes traffic evenly
- [ ] Session affinity works for stateful connections
- [ ] Pod recovery completes in <2 minutes after failure

---

### Task 1.3.2: Circuit Breaker Pattern Implementation
**Priority:** P0 (BLOCKING)
**Effort:** 24 hours
**Dependencies:** Task 1.3.1
**Assignee:** Backend Engineer (Senior)

#### Subtasks:
- [ ] 1. Create circuit breaker library - 6 hours
- [ ] 2. Implement 3 states (Closed, Open, Half-Open) - 4 hours
- [ ] 3. Add failure threshold configuration (5 consecutive failures) - 2 hours
- [ ] 4. Implement recovery timeout (60 seconds) - 2 hours
- [ ] 5. Add circuit breaker decorator for all external calls - 4 hours
- [ ] 6. Create monitoring and metrics - 3 hours
- [ ] 7. Add alerting for circuit breaker state changes - 3 hours

#### Files to Create:
- `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\resilience\circuit_breaker.py` - Circuit breaker implementation
- `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\resilience\decorators.py` - Circuit breaker decorators
- `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\resilience\metrics.py` - Circuit breaker metrics

#### Files to Modify:
- `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\llm\providers\anthropic_provider.py` - Add circuit breaker
- `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\llm\providers\openai_provider.py` - Add circuit breaker
- `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\database\engine.py` - Add circuit breaker for DB calls
- `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\cache\redis_client.py` - Add circuit breaker for Redis

#### Database Changes:
```sql
-- Migration: 013_add_circuit_breaker_state.sql
CREATE TABLE circuit_breaker_state (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    service_name VARCHAR(100) NOT NULL UNIQUE,
    state VARCHAR(20) NOT NULL DEFAULT 'closed', -- closed, open, half_open
    failure_count INTEGER DEFAULT 0,
    last_failure_time TIMESTAMP,
    last_success_time TIMESTAMP,
    next_attempt_time TIMESTAMP,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_circuit_breaker_service ON circuit_breaker_state(service_name);

-- Migration: 014_add_circuit_breaker_events.sql
CREATE TABLE circuit_breaker_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    service_name VARCHAR(100) NOT NULL,
    event_type VARCHAR(50) NOT NULL, -- state_change, failure, success, half_open_attempt
    from_state VARCHAR(20),
    to_state VARCHAR(20),
    failure_count INTEGER,
    message TEXT,
    occurred_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_cb_events_service ON circuit_breaker_events(service_name, occurred_at);
CREATE INDEX idx_cb_events_type ON circuit_breaker_events(event_type);
```

#### Circuit Breaker Implementation:
```python
# C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\resilience\circuit_breaker.py
import asyncio
import time
from enum import Enum
from typing import Callable, Optional, Any
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Circuit is open, all calls fail fast
    HALF_OPEN = "half_open" # Testing if service recovered

class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    States:
    - CLOSED: Normal operation, failures are counted
    - OPEN: All calls fail fast without attempting the operation
    - HALF_OPEN: A few test calls are allowed to check if service recovered
    """

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
        success_threshold: int = 2
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_state_change_time = time.time()
        self.half_open_calls = 0

        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            # Check if circuit should transition to half-open
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_state_change_time >= self.recovery_timeout:
                    logger.info(f"Circuit breaker {self.service_name}: OPEN -> HALF_OPEN")
                    self._change_state(CircuitState.HALF_OPEN)
                else:
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker for {self.service_name} is OPEN"
                    )

            # Limit calls in half-open state
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker for {self.service_name} is HALF_OPEN with max calls reached"
                    )
                self.half_open_calls += 1

        # Execute the function
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise

    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                logger.info(
                    f"Circuit breaker {self.service_name}: Success in HALF_OPEN "
                    f"({self.success_count}/{self.success_threshold})"
                )

                if self.success_count >= self.success_threshold:
                    logger.info(f"Circuit breaker {self.service_name}: HALF_OPEN -> CLOSED")
                    self._change_state(CircuitState.CLOSED)
                    self.failure_count = 0
                    self.success_count = 0
                    self.half_open_calls = 0

            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    async def _on_failure(self, exception: Exception):
        """Handle failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            logger.warning(
                f"Circuit breaker {self.service_name}: Failure "
                f"({self.failure_count}/{self.failure_threshold}): {str(exception)}"
            )

            if self.state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit breaker {self.service_name}: HALF_OPEN -> OPEN")
                self._change_state(CircuitState.OPEN)
                self.success_count = 0
                self.half_open_calls = 0

            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    logger.error(f"Circuit breaker {self.service_name}: CLOSED -> OPEN")
                    self._change_state(CircuitState.OPEN)

    def _change_state(self, new_state: CircuitState):
        """Change circuit breaker state."""
        old_state = self.state
        self.state = new_state
        self.last_state_change_time = time.time()

        # Log state change to database
        asyncio.create_task(self._log_state_change(old_state, new_state))

    async def _log_state_change(self, from_state: CircuitState, to_state: CircuitState):
        """Log state change to database."""
        try:
            async with database.engine.begin() as conn:
                await conn.execute(
                    text("""
                        INSERT INTO circuit_breaker_events
                        (service_name, event_type, from_state, to_state, failure_count)
                        VALUES (:service, :event, :from, :to, :failures)
                    """),
                    {
                        "service": self.service_name,
                        "event": "state_change",
                        "from": from_state.value,
                        "to": to_state.value,
                        "failures": self.failure_count
                    }
                )
        except Exception as e:
            logger.error(f"Failed to log circuit breaker state change: {e}")

class CircuitBreakerOpenException(Exception):
    """Raised when circuit breaker is open."""
    pass

# Decorator for easy circuit breaker application
def circuit_breaker(
    service_name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60
):
    """Decorator to apply circuit breaker to async functions."""
    cb = CircuitBreaker(
        service_name=service_name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout
    )

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await cb.call(func, *args, **kwargs)
        return wrapper
    return decorator
```

#### Configuration Changes:
```yaml
# config/resilience_config.yaml
circuit_breaker:
  enabled: true

  services:
    anthropic_api:
      failure_threshold: 5
      recovery_timeout: 60
      half_open_max_calls: 3
      success_threshold: 2

    openai_api:
      failure_threshold: 5
      recovery_timeout: 60
      half_open_max_calls: 3
      success_threshold: 2

    database:
      failure_threshold: 3
      recovery_timeout: 30
      half_open_max_calls: 2
      success_threshold: 2

    redis:
      failure_threshold: 3
      recovery_timeout: 30
      half_open_max_calls: 2
      success_threshold: 2

    kafka:
      failure_threshold: 5
      recovery_timeout: 60
      half_open_max_calls: 3
      success_threshold: 2

  monitoring:
    track_state_changes: true
    alert_on_open: true
    alert_channels:
      - slack
      - pagerduty
```

#### Tests Required:
- [ ] Unit tests: `test_circuit_breaker.py` (12 test cases)
  - Test closed state normal operation
  - Test transition to open after threshold failures
  - Test fast-fail in open state
  - Test transition to half-open after timeout
  - Test transition back to closed after successes
  - Test transition back to open on half-open failure
  - Test concurrent calls handling
  - Test state persistence
  - Test metrics tracking
  - Test half-open max calls limit
  - Test success threshold in half-open
  - Test recovery timeout adjustment
- [ ] Integration tests: `test_circuit_breaker_integration.py` (5 test cases)
  - Test circuit breaker with real API calls
  - Test circuit breaker with database failures
  - Test circuit breaker with Redis failures
  - Test circuit breaker recovery behavior
  - Test circuit breaker under load

#### Acceptance Criteria:
- [ ] Circuit breaker activates after 5 consecutive failures
- [ ] Open circuit fails fast without calling service
- [ ] Circuit transitions to half-open after 60 seconds
- [ ] Circuit closes after 2 successful half-open calls
- [ ] Circuit opens immediately on half-open failure
- [ ] State changes logged to database
- [ ] Alerts sent when circuit opens
- [ ] Metrics tracked for all state changes
- [ ] Zero false positives (premature opening)
- [ ] Recovery detection works reliably

---

## SUMMARY: PHASE 1 TASKS COMPLETED SO FAR

I've now documented:

**Epic 1.1: Real LLM Integration (120 person-weeks)**
- ✅ Task 1.1.1: Anthropic API (32 hours)
- ✅ Task 1.1.2: OpenAI API (32 hours)
- ✅ Task 1.1.3: Multi-Provider Failover (24 hours)
- ✅ Task 1.1.4: Integration Testing (32 hours)

**Epic 1.2: Database & Caching (80 person-weeks)**
- ✅ Task 1.2.1: PostgreSQL Production Setup (40 hours)
- ✅ Task 1.2.2: Redis Cluster Setup (32 hours)
- ✅ Task 1.2.3: 4-Tier Caching (40 hours)

**Epic 1.3: High Availability (60 person-weeks)**
- ✅ Task 1.3.1: Multi-AZ Kubernetes (40 hours)
- ✅ Task 1.3.2: Circuit Breaker Pattern (24 hours)

**Progress:** 9 tasks documented out of ~200+ total tasks
**Total Hours Documented:** 296 hours (7.4 person-weeks) out of 3,616 person-weeks

**Remaining for Phase 1:**
- Epic 1.3: 3 more tasks (Failover Testing)
- Epic 1.4: Security Hardening (100 person-weeks) - 15+ tasks
- Epic 1.5: Compliance (200 person-weeks) - 20+ tasks
- Epic 1.6: Cost Optimization (40 person-weeks) - 8+ tasks
- Epic 1.7: Multi-Tenancy (40 person-weeks) - 10+ tasks
- Epic 1.8: Advanced RBAC (20 person-weeks) - 6+ tasks
- Epic 1.9: Data Residency (15 person-weeks) - 5+ tasks
- Epic 1.10: SLA Management (20 person-weeks) - 6+ tasks

**Remaining Phases:**
- Phase 2: Intelligence (Q2-Q3 2026) - 1,162 person-weeks
- Phase 3: Excellence (Q4 2026-Q1 2027) - 730 person-weeks
- Phase 4: Operations (Q2-Q3 2027) - 980 person-weeks

---

**Document Status:** IN PROGRESS
**Next Section:** Complete Epic 1.3, then Epic 1.4 (Security Hardening)
**Format:** Each task <40 hours with complete implementation details
