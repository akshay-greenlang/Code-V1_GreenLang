# PRD: INFRA-001 - Deploy Kubernetes Production Cluster

**Document Version:** 1.0
**Date:** February 3, 2026
**Status:** READY FOR EXECUTION
**Priority:** P0 - CRITICAL
**Owner:** Infrastructure Team
**Ralphy Task ID:** INFRA-001

---

## Executive Summary

Deploy a production-ready AWS EKS (Elastic Kubernetes Service) cluster to host the GreenLang Climate OS platform. The Infrastructure-as-Code (Terraform) is complete but has not been applied to AWS.

### Current State
- ✅ Terraform modules complete (26+ .tf files)
- ✅ Kubernetes manifests ready (250+ YAML files)
- ✅ Helm charts configured (8+ charts)
- ✅ Kustomize overlays defined (dev/staging/prod)
- ❌ AWS resources NOT provisioned
- ❌ EKS cluster NOT running
- ❌ Production workloads on Docker Compose only

### Target State
- AWS EKS cluster running in production
- Multi-AZ high availability
- Auto-scaling node groups
- Monitoring and observability stack deployed
- Applications migrated from Docker Compose to Kubernetes

---

## Scope

### In Scope
1. Apply Terraform to provision AWS infrastructure
2. Deploy EKS cluster with node groups
3. Configure networking (VPC, subnets, security groups)
4. Set up supporting services (RDS, ElastiCache, S3)
5. Deploy monitoring stack (Prometheus, Grafana)
6. Configure IAM roles and security
7. Deploy GreenLang applications to EKS
8. Validate production readiness

### Out of Scope
- Application code changes
- New feature development
- Disaster recovery setup (separate task)
- Multi-region deployment (future phase)

---

## Technical Specifications

### AWS Infrastructure (via Terraform)

#### 1. VPC Configuration
```
Location: deployment/terraform/modules/vpc/
```
| Resource | Specification |
|----------|---------------|
| CIDR | 10.0.0.0/16 |
| Availability Zones | 3 (us-east-1a, 1b, 1c) |
| Public Subnets | 3 (10.0.1.0/24, 10.0.2.0/24, 10.0.3.0/24) |
| Private Subnets | 3 (10.0.11.0/24, 10.0.12.0/24, 10.0.13.0/24) |
| Database Subnets | 3 (10.0.21.0/24, 10.0.22.0/24, 10.0.23.0/24) |
| NAT Gateways | 3 (one per AZ) |
| VPC Endpoints | S3, ECR, CloudWatch, Secrets Manager |
| Flow Logs | Enabled (90-day retention) |

#### 2. EKS Cluster
```
Location: deployment/terraform/modules/eks/
```
| Resource | Specification |
|----------|---------------|
| Cluster Name | greenlang-prod-eks |
| Kubernetes Version | 1.28+ |
| Endpoint Access | Private + Public (restricted) |
| Encryption | KMS for secrets |
| Logging | API, audit, authenticator, controllerManager, scheduler |
| OIDC Provider | Enabled (for IRSA) |

#### 3. Node Groups
| Node Group | Instance Type | Desired | Min | Max | Purpose |
|------------|---------------|---------|-----|-----|---------|
| System | m6i.xlarge | 3 | 2 | 5 | Core K8s workloads |
| API | c6i.2xlarge | 3 | 3 | 10 | API Gateway, high-traffic |
| Agent Runtime | c6i.xlarge | 5 | 3 | 25 | GreenLang agents |

#### 4. RDS PostgreSQL
```
Location: deployment/terraform/modules/rds/
```
| Resource | Specification |
|----------|---------------|
| Engine | PostgreSQL 15.4 |
| Instance Class | db.r6g.xlarge |
| Storage | 200GB gp3 (auto-scaling to 2TB) |
| Multi-AZ | Enabled |
| Read Replicas | 2 |
| Backup Retention | 30 days |
| Encryption | KMS at-rest |
| Performance Insights | Enabled |

#### 5. ElastiCache Redis
```
Location: deployment/terraform/modules/elasticache/
```
| Resource | Specification |
|----------|---------------|
| Engine | Redis 7.x |
| Node Type | cache.r6g.xlarge |
| Cluster Mode | Enabled |
| Nodes | 3 (1 primary + 2 replicas) |
| Multi-AZ | Enabled |
| Encryption | In-transit + at-rest |
| Snapshot Retention | 14 days |

#### 6. S3 Buckets
```
Location: deployment/terraform/modules/s3/
```
| Bucket | Purpose | Retention |
|--------|---------|-----------|
| greenlang-prod-artifacts | Build artifacts | 90 days |
| greenlang-prod-logs | Application logs | 365 days |
| greenlang-prod-backups | Database backups | 7 years (Object Lock) |
| greenlang-prod-data | Application data | Versioned |
| greenlang-prod-static | Static assets | CDN-enabled |

---

## Ralphy Task Configuration

```yaml
# RALPHY-INFRA-001.yaml
task:
  id: "INFRA-001"
  name: "Deploy Kubernetes Production Cluster"
  priority: "P0"
  effort: "1 week"

config:
  terraform_version: "1.6+"
  aws_region: "us-east-1"
  environment: "prod"

steps:
  - id: "INFRA-001-01"
    name: "Initialize Terraform Backend"
    command: |
      cd deployment/terraform
      terraform init -backend-config=environments/prod/backend.hcl
    validation: "terraform validate"

  - id: "INFRA-001-02"
    name: "Plan Infrastructure"
    command: |
      cd deployment/terraform/environments/prod
      terraform plan -out=tfplan
    validation: "terraform show tfplan"
    approval_required: true

  - id: "INFRA-001-03"
    name: "Apply VPC Module"
    command: |
      terraform apply -target=module.vpc -auto-approve
    validation: |
      aws ec2 describe-vpcs --filters "Name=tag:Name,Values=greenlang-prod-vpc"

  - id: "INFRA-001-04"
    name: "Apply EKS Module"
    command: |
      terraform apply -target=module.eks -auto-approve
    validation: |
      aws eks describe-cluster --name greenlang-prod-eks
    depends_on: ["INFRA-001-03"]

  - id: "INFRA-001-05"
    name: "Apply RDS Module"
    command: |
      terraform apply -target=module.rds -auto-approve
    validation: |
      aws rds describe-db-instances --db-instance-identifier greenlang-prod-db
    depends_on: ["INFRA-001-03"]

  - id: "INFRA-001-06"
    name: "Apply ElastiCache Module"
    command: |
      terraform apply -target=module.elasticache -auto-approve
    validation: |
      aws elasticache describe-cache-clusters --cache-cluster-id greenlang-prod-redis
    depends_on: ["INFRA-001-03"]

  - id: "INFRA-001-07"
    name: "Apply S3 Module"
    command: |
      terraform apply -target=module.s3 -auto-approve
    validation: |
      aws s3 ls | grep greenlang-prod

  - id: "INFRA-001-08"
    name: "Apply IAM Module"
    command: |
      terraform apply -target=module.iam -auto-approve
    validation: |
      aws iam list-roles | grep greenlang
    depends_on: ["INFRA-001-04"]

  - id: "INFRA-001-09"
    name: "Configure kubectl"
    command: |
      aws eks update-kubeconfig --name greenlang-prod-eks --region us-east-1
    validation: |
      kubectl cluster-info
    depends_on: ["INFRA-001-04"]

  - id: "INFRA-001-10"
    name: "Deploy Kubernetes Add-ons"
    command: |
      # AWS Load Balancer Controller
      helm repo add eks https://aws.github.io/eks-charts
      helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
        -n kube-system \
        --set clusterName=greenlang-prod-eks

      # Cluster Autoscaler
      helm install cluster-autoscaler autoscaler/cluster-autoscaler \
        -n kube-system \
        --set autoDiscovery.clusterName=greenlang-prod-eks

      # Metrics Server
      kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
    validation: |
      kubectl get pods -n kube-system | grep -E "(aws-load-balancer|cluster-autoscaler|metrics-server)"
    depends_on: ["INFRA-001-09"]

  - id: "INFRA-001-11"
    name: "Deploy Monitoring Stack"
    command: |
      cd deployment/infrastructure/helm/greenlang
      helm dependency update
      helm install monitoring . -f values-prod.yaml -n monitoring --create-namespace
    validation: |
      kubectl get pods -n monitoring
    depends_on: ["INFRA-001-10"]

  - id: "INFRA-001-12"
    name: "Deploy GreenLang Applications"
    command: |
      cd deployment/kustomize/overlays/prod
      kubectl apply -k .
    validation: |
      kubectl get pods -n greenlang
    depends_on: ["INFRA-001-11"]

  - id: "INFRA-001-13"
    name: "Validate Production Readiness"
    command: |
      # Check all pods running
      kubectl get pods -A | grep -v Running | grep -v Completed

      # Check services
      kubectl get svc -n greenlang

      # Check ingress
      kubectl get ingress -n greenlang

      # Health check
      curl -s https://api.greenlang.io/health
    validation: |
      All pods in Running state
      Services have endpoints
      Ingress has external IP
      Health check returns 200
    depends_on: ["INFRA-001-12"]

quality_gates:
  - "All Terraform resources created successfully"
  - "EKS cluster accessible via kubectl"
  - "All node groups healthy and scaling"
  - "RDS accessible from EKS pods"
  - "Redis accessible from EKS pods"
  - "Monitoring stack operational"
  - "Applications responding to health checks"
  - "Auto-scaling triggers working"

rollback:
  enabled: true
  command: |
    terraform destroy -auto-approve
  approval_required: true

notifications:
  slack_channel: "#infrastructure"
  on_success: true
  on_failure: true
  on_approval_needed: true
```

---

## Execution Plan

### Phase 1: Foundation (Day 1-2)
| Step | Task | Owner | Duration |
|------|------|-------|----------|
| 1.1 | Review and validate Terraform code | DevOps | 2 hours |
| 1.2 | Create Terraform backend (S3 + DynamoDB) | DevOps | 1 hour |
| 1.3 | Initialize Terraform | DevOps | 30 min |
| 1.4 | Plan and review infrastructure | DevOps + Arch | 2 hours |
| 1.5 | Apply VPC module | DevOps | 30 min |
| 1.6 | Validate networking | DevOps | 1 hour |

### Phase 2: Core Infrastructure (Day 2-3)
| Step | Task | Owner | Duration |
|------|------|-------|----------|
| 2.1 | Apply EKS module | DevOps | 20 min |
| 2.2 | Wait for EKS provisioning | - | 15 min |
| 2.3 | Configure kubectl access | DevOps | 15 min |
| 2.4 | Apply RDS module | DevOps | 15 min |
| 2.5 | Wait for RDS provisioning | - | 20 min |
| 2.6 | Apply ElastiCache module | DevOps | 10 min |
| 2.7 | Apply S3 and IAM modules | DevOps | 10 min |

### Phase 3: Kubernetes Setup (Day 3-4)
| Step | Task | Owner | Duration |
|------|------|-------|----------|
| 3.1 | Deploy AWS Load Balancer Controller | DevOps | 30 min |
| 3.2 | Deploy Cluster Autoscaler | DevOps | 30 min |
| 3.3 | Deploy External Secrets Operator | DevOps | 30 min |
| 3.4 | Deploy cert-manager | DevOps | 30 min |
| 3.5 | Configure Ingress | DevOps | 1 hour |
| 3.6 | Deploy monitoring stack | DevOps | 1 hour |

### Phase 4: Application Deployment (Day 4-5)
| Step | Task | Owner | Duration |
|------|------|-------|----------|
| 4.1 | Configure secrets in AWS Secrets Manager | DevOps | 1 hour |
| 4.2 | Deploy GreenLang namespace and RBAC | DevOps | 30 min |
| 4.3 | Deploy GL-CSRD-APP | DevOps | 30 min |
| 4.4 | Deploy GL-CBAM-APP | DevOps | 30 min |
| 4.5 | Deploy GL-VCCI-APP | DevOps | 30 min |
| 4.6 | Deploy Agent Factory | DevOps | 30 min |

### Phase 5: Validation (Day 5)
| Step | Task | Owner | Duration |
|------|------|-------|----------|
| 5.1 | Run integration tests | QA | 2 hours |
| 5.2 | Performance baseline | QA | 2 hours |
| 5.3 | Security scan | Security | 2 hours |
| 5.4 | Documentation update | DevOps | 2 hours |
| 5.5 | Handover to operations | DevOps | 1 hour |

---

## Cost Estimation

### Monthly AWS Costs (Production)

| Service | Configuration | Monthly Cost |
|---------|---------------|--------------|
| EKS Control Plane | 1 cluster | $73 |
| EC2 (System nodes) | 3x m6i.xlarge | $432 |
| EC2 (API nodes) | 3x c6i.2xlarge | $612 |
| EC2 (Agent nodes) | 5x c6i.xlarge | $680 |
| RDS PostgreSQL | db.r6g.xlarge + 2 replicas | $1,200 |
| ElastiCache Redis | 3x cache.r6g.xlarge | $900 |
| NAT Gateway | 3 gateways | $135 |
| S3 Storage | ~500GB | $12 |
| Data Transfer | ~1TB | $90 |
| CloudWatch | Logs + Metrics | $100 |
| **Total** | | **~$4,234/month** |

### Cost Optimization Opportunities
- Reserved Instances: 30-40% savings
- Spot Instances for agent nodes: 60-70% savings
- S3 Intelligent Tiering: 20% savings

---

## Success Criteria

### Must Have (P0)
- [ ] EKS cluster running with 3 node groups
- [ ] All applications accessible via ingress
- [ ] Database connections working
- [ ] Redis caching functional
- [ ] Health checks passing
- [ ] Monitoring dashboards accessible

### Should Have (P1)
- [ ] Auto-scaling validated under load
- [ ] Alerts configured and tested
- [ ] Backup and restore tested
- [ ] Security scan passed

### Nice to Have (P2)
- [ ] Performance benchmarks documented
- [ ] Runbook created
- [ ] Cost optimization implemented

---

## Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Terraform state corruption | High | Low | Use S3 backend with versioning + DynamoDB locking |
| EKS provisioning failure | High | Low | Use official AWS modules, test in staging first |
| Network connectivity issues | Medium | Medium | Validate VPC endpoints, security groups |
| Cost overrun | Medium | Medium | Set up billing alerts, use Reserved Instances |
| Application migration issues | Medium | Medium | Gradual migration, maintain Docker Compose fallback |

---

## Approvals Required

| Approver | Role | Approval For |
|----------|------|--------------|
| CTO | Technical Lead | Architecture approval |
| DevOps Lead | Infrastructure | Terraform plan review |
| Security Lead | Security | IAM and network policies |
| Finance | Budget | Cost approval |

---

## References

### Existing IaC Files
- **Terraform Root**: `deployment/terraform/`
- **EKS Module**: `deployment/terraform/modules/eks/`
- **VPC Module**: `deployment/terraform/modules/vpc/`
- **Production Config**: `deployment/terraform/environments/prod/`

### Kubernetes Manifests
- **Helm Charts**: `deployment/helm/greenlang-agents/`
- **Kustomize**: `deployment/kustomize/overlays/prod/`
- **K8s Manifests**: `deployment/kubernetes/manifests/`

### Documentation
- **Deployment Guide**: `deployment/DEPLOYMENT_GUIDE.md`
- **Environment Sizing**: `deployment/environment-sizing-guide.md`
- **Cost Estimation**: `deployment/cost-estimation.md`

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-03 | AI Agent Team | Initial PRD |

---

**END OF PRD**
