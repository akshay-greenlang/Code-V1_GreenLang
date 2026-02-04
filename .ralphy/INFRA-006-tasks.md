# INFRA-006: Deploy API Gateway (Kong) - Task List

## Terraform Module
- [x] Create deployment/terraform/modules/kong-gateway/main.tf
- [x] Create deployment/terraform/modules/kong-gateway/variables.tf
- [x] Create deployment/terraform/modules/kong-gateway/outputs.tf

## Helm Chart
- [x] Create deployment/helm/kong-gateway/Chart.yaml
- [x] Create deployment/helm/kong-gateway/values.yaml (production)
- [x] Create deployment/helm/kong-gateway/values-dev.yaml
- [x] Create deployment/helm/kong-gateway/values-staging.yaml
- [x] Create deployment/helm/kong-gateway/templates/_helpers.tpl
- [x] Create deployment/helm/kong-gateway/templates/deployment.yaml
- [x] Create deployment/helm/kong-gateway/templates/service.yaml
- [x] Create deployment/helm/kong-gateway/templates/hpa.yaml
- [x] Create deployment/helm/kong-gateway/templates/pdb.yaml
- [x] Create deployment/helm/kong-gateway/templates/configmap.yaml
- [x] Create deployment/helm/kong-gateway/templates/secret.yaml
- [x] Create deployment/helm/kong-gateway/templates/servicemonitor.yaml

## Kong Declarative Configuration
- [x] Create deployment/config/kong/kong.yaml (routes, services, plugins, consumers, upstreams)

## Kong Plugins
- [x] Configure rate-limiting plugin (Redis-backed)
- [x] Configure jwt plugin
- [x] Configure cors plugin
- [x] Configure request-transformer plugin
- [x] Configure response-transformer plugin
- [x] Configure ip-restriction plugin
- [x] Configure bot-detection plugin
- [x] Configure request-size-limiting plugin
- [x] Configure prometheus plugin
- [x] Configure http-log plugin
- [x] Configure acl plugin

## Custom Plugins
- [x] Create deployment/config/kong/custom-plugins/gl-tenant-isolation/handler.lua
- [x] Create deployment/config/kong/custom-plugins/gl-tenant-isolation/schema.lua

## Kubernetes CRDs
- [x] Create deployment/kubernetes/kong-gateway/kong-plugins.yaml (KongPlugin + KongClusterPlugin)
- [x] Create deployment/kubernetes/kong-gateway/kong-consumers.yaml (KongConsumer)
- [x] Create deployment/kubernetes/kong-gateway/kong-routes.yaml (Ingress + IngressClass)
- [x] Create deployment/kubernetes/kong-gateway/networkpolicy.yaml

## Monitoring
- [x] Create deployment/monitoring/dashboards/kong-gateway.json (Grafana dashboard)
- [x] Create deployment/monitoring/alerts/kong-alerts.yaml (Prometheus alerts)

## Documentation
- [x] Create PRD-INFRA-006-API-Gateway.md
- [x] Create .ralphy/INFRA-006-tasks.md (this file)

## Validation (Post-Deployment)
- [ ] Verify Terraform plan applies cleanly
- [ ] Verify Helm chart lints successfully
- [ ] Verify Kong declarative config validates (deck validate)
- [ ] Verify health endpoint accessible through Kong
- [ ] Verify rate limiting enforced per route
- [ ] Verify JWT authentication working
- [ ] Verify Grafana dashboard loads
- [ ] Verify Prometheus alerts firing correctly
- [ ] Verify custom plugin tenant isolation
- [ ] Load test: verify P99 < 10ms gateway overhead
