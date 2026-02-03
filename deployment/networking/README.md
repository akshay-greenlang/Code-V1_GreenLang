# GreenLang Networking Infrastructure

**INFRA-001: Service Mesh and Networking Configuration**

This directory contains all networking configurations for the GreenLang platform, including Istio service mesh, DNS management, and TLS certificate automation.

## Architecture Overview

```
                                   Internet
                                      |
                              [AWS Route53 DNS]
                                      |
                              [AWS NLB / ALB]
                                      |
                         +------------+------------+
                         |                         |
                    [Istio Ingress]          [Istio Egress]
                    Gateway (443)            Gateway (443)
                         |                         |
                         v                         v
              +--------------------+      [External APIs]
              |   Virtual Services |      - OpenAI
              |   (Traffic Routing)|      - Anthropic
              +--------------------+      - Stripe
                         |                - SendGrid
              +--------------------+      - AWS Services
              |  Destination Rules |
              |  (Load Balancing)  |
              +--------------------+
                         |
              +--------------------+
              | Authorization      |
              | Policies (mTLS)    |
              +--------------------+
                         |
         +-------+-------+-------+-------+
         |       |       |       |       |
      [API]   [Web]  [Admin] [Worker] [gRPC]
```

## Directory Structure

```
deployment/networking/
├── README.md                           # This file
├── istio/
│   ├── istio-values.yaml              # Istio Helm values
│   ├── gateway.yaml                   # Ingress/Egress gateways
│   ├── virtual-services.yaml          # Traffic routing rules
│   ├── destination-rules.yaml         # Load balancing policies
│   └── authorization-policies.yaml    # mTLS and access control
├── dns/
│   ├── external-dns-values.yaml       # ExternalDNS for Route53
│   └── coredns-config.yaml            # Custom CoreDNS config
└── certificates/
    ├── cert-manager-values.yaml       # cert-manager Helm values
    ├── cluster-issuer.yaml            # Let's Encrypt issuers
    └── certificates.yaml              # TLS certificate definitions
```

## Prerequisites

Before deploying the networking infrastructure, ensure:

1. **Kubernetes Cluster**: EKS cluster is running (v1.27+)
2. **Helm**: Helm v3.x installed
3. **IAM Roles**: IRSA roles created for cert-manager and external-dns
4. **Route53**: Hosted zone created for greenlang.io

## Installation Order

The networking components must be installed in the following order:

### 1. Install cert-manager

```bash
# Add Jetstack Helm repository
helm repo add jetstack https://charts.jetstack.io
helm repo update

# Install cert-manager with CRDs
helm install cert-manager jetstack/cert-manager \
  -n cert-manager --create-namespace \
  -f certificates/cert-manager-values.yaml
```

### 2. Create Cluster Issuers

```bash
# Wait for cert-manager to be ready
kubectl wait --for=condition=Available deployment/cert-manager -n cert-manager --timeout=300s

# Apply cluster issuers
kubectl apply -f certificates/cluster-issuer.yaml
```

### 3. Install Istio

```bash
# Add Istio Helm repository
helm repo add istio https://istio-release.storage.googleapis.com/charts
helm repo update

# Install Istio base (CRDs)
helm install istio-base istio/base \
  -n istio-system --create-namespace

# Install Istiod (control plane)
helm install istiod istio/istiod \
  -n istio-system \
  -f istio/istio-values.yaml

# Install Istio Ingress Gateway
helm install istio-ingress istio/gateway \
  -n istio-ingress --create-namespace \
  -f istio/istio-values.yaml
```

### 4. Configure Istio Resources

```bash
# Apply gateway configurations
kubectl apply -f istio/gateway.yaml

# Apply virtual services
kubectl apply -f istio/virtual-services.yaml

# Apply destination rules
kubectl apply -f istio/destination-rules.yaml

# Apply authorization policies
kubectl apply -f istio/authorization-policies.yaml
```

### 5. Create TLS Certificates

```bash
# Apply certificate resources
kubectl apply -f certificates/certificates.yaml

# Verify certificates are issued
kubectl get certificates -n istio-system
```

### 6. Install ExternalDNS

```bash
# Add ExternalDNS Helm repository
helm repo add external-dns https://kubernetes-sigs.github.io/external-dns/
helm repo update

# Install ExternalDNS
helm install external-dns external-dns/external-dns \
  -n external-dns --create-namespace \
  -f dns/external-dns-values.yaml
```

### 7. Apply CoreDNS Configuration

```bash
# Apply custom CoreDNS configuration
kubectl apply -f dns/coredns-config.yaml

# Restart CoreDNS to pick up changes
kubectl rollout restart deployment/coredns -n kube-system
```

## Configuration Details

### Istio Service Mesh

#### Gateway Configuration
- **HTTPS on port 443**: TLS termination at the gateway
- **HTTP to HTTPS redirect**: All HTTP traffic redirected to HTTPS
- **gRPC support**: Dedicated gRPC gateway on port 15443
- **Egress control**: Managed egress for external API calls

#### Traffic Management
- **Canary deployments**: Weighted traffic routing (90/10 stable/canary)
- **Circuit breaking**: Outlier detection with automatic pod ejection
- **Retry policies**: Automatic retries with exponential backoff
- **Timeout configuration**: Per-service timeout settings

#### Security
- **mTLS**: STRICT mode enforced cluster-wide
- **Zero-trust**: Default deny with explicit allow rules
- **JWT validation**: Request authentication for API endpoints
- **Rate limiting**: EnvoyFilter-based rate limiting

### DNS Management

#### ExternalDNS
- **Automatic record creation**: Creates Route53 records from Kubernetes resources
- **TXT record ownership**: Prevents conflicts with manual records
- **Sync policy**: Full sync (create, update, delete)
- **Sources**: Services, Ingress, Istio Gateway, VirtualService

#### CoreDNS
- **Custom zones**: Internal zone for greenlang.local
- **AWS integration**: VPC DNS resolver for AWS services
- **Caching**: 30-second TTL for internal queries
- **HA**: 2 replicas with HPA scaling

### Certificate Management

#### Let's Encrypt
- **Production issuer**: For production certificates
- **Staging issuer**: For testing (avoids rate limits)
- **DNS-01 challenge**: For wildcard certificates
- **HTTP-01 challenge**: For standard certificates

#### Internal CA
- **Root CA**: Self-signed root for internal use
- **mTLS certificates**: For service-to-service communication
- **1-year validity**: With 30-day renewal window

## Traffic Flow

### External Request Flow

```
1. Client Request
   └── DNS Resolution (Route53)
       └── greenlang.io → NLB IP

2. Load Balancer
   └── AWS NLB
       └── Forwards to Istio Ingress Gateway

3. Istio Ingress Gateway
   └── TLS Termination
       └── Certificate from cert-manager

4. Virtual Service
   └── Route matching
       └── Header-based routing
       └── URI-based routing

5. Destination Rule
   └── Load balancing (LEAST_REQUEST)
       └── Circuit breaker check
       └── Subset selection (stable/canary)

6. Authorization Policy
   └── JWT validation
       └── mTLS verification
       └── Permission check

7. Service Pod
   └── Application processing
       └── Response returned
```

### Service-to-Service Flow

```
1. Service A Request
   └── Envoy sidecar intercepts

2. mTLS Establishment
   └── ISTIO_MUTUAL mode
       └── Certificate from Istio CA

3. Destination Rule
   └── Connection pooling
       └── Load balancing

4. Authorization Policy
   └── Service account validation
       └── Namespace check

5. Service B Pod
   └── Request processed
```

## Monitoring and Observability

### Istio Metrics

```bash
# View Istio metrics
kubectl exec -n istio-system deploy/istiod -- curl localhost:15014/metrics

# View gateway metrics
kubectl exec -n istio-ingress deploy/istio-ingressgateway -- curl localhost:15020/stats/prometheus
```

### Certificate Status

```bash
# Check certificate status
kubectl get certificates -A
kubectl describe certificate greenlang-api-tls -n istio-system

# Check certificate expiry
kubectl get secret greenlang-api-tls -n istio-system -o jsonpath='{.data.tls\.crt}' | \
  base64 -d | openssl x509 -noout -dates
```

### DNS Status

```bash
# Check ExternalDNS logs
kubectl logs -n external-dns deploy/external-dns

# Verify DNS records
aws route53 list-resource-record-sets --hosted-zone-id <ZONE_ID>
```

## Troubleshooting

### Common Issues

#### 1. Certificate Not Issued

```bash
# Check certificate status
kubectl describe certificate <name> -n <namespace>

# Check certificate request
kubectl get certificaterequest -n <namespace>

# Check ACME challenges
kubectl get challenges -n <namespace>

# Check cert-manager logs
kubectl logs -n cert-manager deploy/cert-manager
```

#### 2. Service Not Reachable

```bash
# Check Istio configuration
istioctl analyze -n greenlang

# Check virtual service
kubectl describe virtualservice <name> -n greenlang

# Check destination rule
kubectl describe destinationrule <name> -n greenlang

# Check envoy proxy
istioctl proxy-status
istioctl proxy-config cluster <pod-name> -n greenlang
```

#### 3. mTLS Errors

```bash
# Check peer authentication
kubectl get peerauthentication -A

# Check TLS mode
istioctl authn tls-check <pod-name> -n greenlang

# Check authorization policy
kubectl describe authorizationpolicy <name> -n greenlang
```

#### 4. DNS Not Resolving

```bash
# Check ExternalDNS logs
kubectl logs -n external-dns deploy/external-dns -f

# Test DNS resolution from pod
kubectl run -it --rm debug --image=busybox --restart=Never -- nslookup api.greenlang.io

# Check CoreDNS logs
kubectl logs -n kube-system -l k8s-app=kube-dns
```

## Security Considerations

### Network Policies

In addition to Istio authorization policies, consider implementing Kubernetes NetworkPolicies for defense in depth:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-ingress
  namespace: greenlang
spec:
  podSelector: {}
  policyTypes:
    - Ingress
```

### Secret Management

- All TLS certificates are stored as Kubernetes secrets
- Use external secret management (AWS Secrets Manager) for sensitive data
- Enable secret encryption at rest in etcd

### Access Control

- All services require mTLS for communication
- JWT tokens validated at the gateway level
- Service accounts used for authorization

## Maintenance

### Certificate Renewal

cert-manager automatically renews certificates 15 days before expiry. Monitor:

```bash
# Set up alerting for expiring certificates
kubectl get certificates -A -o jsonpath='{range .items[*]}{.metadata.name}: {.status.notAfter}{"\n"}{end}'
```

### Istio Upgrades

```bash
# Check current version
istioctl version

# Perform canary upgrade
helm upgrade istiod istio/istiod -n istio-system -f istio/istio-values.yaml

# Restart workloads to pick up new proxy
kubectl rollout restart deployment -n greenlang
```

### Configuration Changes

```bash
# Validate changes before applying
istioctl analyze -f istio/virtual-services.yaml

# Apply changes
kubectl apply -f istio/virtual-services.yaml

# Verify
istioctl proxy-status
```

## References

- [Istio Documentation](https://istio.io/latest/docs/)
- [cert-manager Documentation](https://cert-manager.io/docs/)
- [ExternalDNS Documentation](https://kubernetes-sigs.github.io/external-dns/)
- [CoreDNS Documentation](https://coredns.io/manual/toc/)
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)
