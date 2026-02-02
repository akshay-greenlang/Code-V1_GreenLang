# GreenLang Agent Deployment Pack Template

**Version:** 1.0.0
**Author:** GreenLang Framework Team
**Date:** October 2025

## Overview

This directory contains a universal deployment pack template and automation script for deploying all GreenLang AI agents in production environments. The template provides comprehensive configuration for resources, dependencies, APIs, monitoring, security, and scaling.

## Contents

### Files

1. **`agent_deployment_pack.yaml`** - Universal deployment pack template
   - Complete YAML configuration covering all deployment aspects
   - 870+ lines of production-ready settings
   - Extensively commented with inline documentation
   - Supports all agent types: AI orchestration, deterministic, hybrid, ML inference

2. **`../scripts/apply_deployment_template.py`** - Template application script
   - Automated deployment pack generation for all 10 AI agents
   - Interactive and batch modes
   - Validation and dry-run capabilities
   - Python 3.10+ compatible

3. **`README_DEPLOYMENT_TEMPLATE.md`** - This documentation

## Quick Start

### List All Available Agents

```bash
python scripts/apply_deployment_template.py --list
```

Output shows all 10 agents organized by domain:
- **Analytics**: anomaly_iforest, forecast_sarima, recommendation_ai
- **Emissions**: carbon_ai, fuel_ai, grid_factor_ai
- **Industrial**: boiler_replacement_ai, industrial_heat_pump_ai, industrial_process_heat_ai
- **Reporting**: report_ai

### Generate Deployment Pack for a Single Agent

```bash
# Generate for fuel_ai agent
python scripts/apply_deployment_template.py --agent fuel_ai

# Output: packs/fuel_ai/deployment_pack.yaml
```

### Generate Deployment Packs for All Agents

```bash
# Generate all 10 deployment packs
python scripts/apply_deployment_template.py --all-agents

# Output: packs/*/deployment_pack.yaml (10 files)
```

### Dry Run (Preview Without Writing)

```bash
# Preview what would be generated
python scripts/apply_deployment_template.py --agent fuel_ai --dry-run
```

### Interactive Mode

```bash
# Interactive mode with prompts for custom configuration
python scripts/apply_deployment_template.py --interactive
```

## Template Structure

The `agent_deployment_pack.yaml` template includes:

### 1. Pack Metadata
- Pack ID (domain/agent_name format)
- Version (semantic versioning)
- Agent type classification
- Description and tags
- Author and license
- Maintainer contact info

### 2. Resource Requirements
- **Memory**: 512-1024 MB (configurable)
- **CPU**: 1.0-2.0 cores (configurable)
- **GPU**: Optional for ML inference agents
- **Disk**: 100+ MB
- **Scaling**: Auto-scaling configuration (1-10 instances)
- **Custom metrics**: Queue depth, response time

### 3. Dependencies
- **Python**: >=3.9,<4.0
- **Core packages**: pydantic, typing-extensions
- **AI/ML packages**: anthropic, openai, numpy, pandas, scikit-learn, statsmodels
- **GreenLang modules**: agents.base, intelligence, core, types
- **System packages**: Optional system-level dependencies
- **External services**: Database, API connections

### 4. API Endpoints

#### Primary Endpoints
- `/api/v1/agents/{agent_id}/execute` - Execute agent analysis
  - Method: POST
  - Authentication: Required
  - Rate limit: 100 req/min
  - Timeout: 30s

- `/api/v1/agents/{agent_id}/health` - Health check
  - Method: GET
  - Authentication: Optional
  - Rate limit: 1000 req/min
  - Timeout: 5s

- `/api/v1/agents/{agent_id}/performance` - Performance metrics
  - Method: GET
  - Authentication: Required
  - Rate limit: 100 req/min
  - Timeout: 5s

- `/api/v1/agents/{agent_id}/batch` - Batch execution
  - Method: POST
  - Authentication: Required
  - Rate limit: 10 req/min
  - Timeout: 5 min

### 5. Configuration Parameters

#### AI Settings (for AI agents)
- **Budget**: $0.50-$1.00 per execution (configurable)
- **Temperature**: 0.0 (deterministic)
- **Seed**: 42 (reproducible)
- **Max iterations**: 5
- **Explanations**: Enabled
- **Recommendations**: Enabled
- **Provider**: Auto-detect (Anthropic, OpenAI, Azure)

#### Operational Settings
- **Logging**: INFO level, JSON format, stdout
- **Tracing**: Jaeger distributed tracing
- **Metrics**: Prometheus metrics enabled
- **Caching**: 1 hour TTL
- **Timeout**: 30 seconds
- **Retry**: 3 attempts with exponential backoff

### 6. Monitoring & Observability

#### Prometheus Metrics
- `agent_execution_count` - Total executions (counter)
- `agent_execution_duration_ms` - Execution time (histogram)
- `agent_cost_usd` - AI cost per execution (gauge)
- `agent_error_rate` - Error percentage (gauge)
- `agent_token_usage` - AI tokens consumed (counter)
- `agent_tool_calls` - Tool call count (counter)

#### Health Checks
- **Liveness probe**: 30s interval, 5s timeout
- **Readiness probe**: 10s interval, 5s timeout

#### Logging
- Format: JSON structured logging
- Output: stdout (container-friendly)
- Fields: agent_id, request_id, user_id, execution_time_ms, status, error_type
- Rotation: 100 MB max size, 7 days retention

#### Distributed Tracing
- Backend: Jaeger
- Sampling rate: 10%
- Full request/response tracing

#### Alerting Rules
- High error rate (>5%) - Warning
- Critical error rate (>20%) - Critical
- High latency (>10s p95) - Warning
- Budget exceeded (>$1.00) - Warning

### 7. Security

#### Authentication
- **API Key**: Header-based (X-API-Key)
- **JWT**: Optional token authentication
- **RBAC**: Role-based access control

#### Authorization
- Required roles: agent_user, emissions_analyst

#### TLS/HTTPS
- Enabled by default
- Minimum TLS 1.2
- Certificate from cert-manager

#### Secrets Management
- Environment variable injection
- Secrets: ANTHROPIC_API_KEY, GREENLANG_API_KEY, DATABASE_PASSWORD

#### CORS
- Enabled for trusted origins
- Methods: GET, POST, PUT, DELETE
- Headers: Content-Type, Authorization, X-API-Key

#### Network Policies
- Ingress: From greenlang-api and greenlang-web namespaces
- Egress: DNS resolution, AI provider APIs

#### Security Scanning
- Scanner: Trivy
- Schedule: Daily at 2 AM
- Block on: HIGH severity vulnerabilities

#### Pod Security
- Run as non-root (UID 1000)
- Read-only root filesystem
- No privilege escalation
- Drop all capabilities
- SELinux context

### 8. Deployment

#### Environment
- Production-ready defaults
- Region: us-east-1
- Availability zones: us-east-1a, us-east-1b

#### Docker
- Image: `greenlang/agent:{agent_name}:latest`
- Pull policy: Always
- Private registry support

#### Kubernetes
- Namespace: greenlang-agents
- Service account: agent-runner
- Rolling update strategy (zero downtime)
- Pod anti-affinity for HA
- Node selector for AI workload nodes

#### Service
- Type: ClusterIP
- Port: 80 (external) → 8080 (container)
- Session affinity: ClientIP

#### Ingress
- Enabled by default
- Class: nginx
- Host: agents.greenlang.com
- Path: /api/v1/agents/{agent_name}
- TLS: Enabled with cert

### 9. Testing & Validation
- Unit tests required
- Integration tests required
- Smoke tests required
- Performance benchmarks (p95 < 5s, throughput > 10 rps)
- Schema validation
- Security scanning
- License compliance checks
- Dependency vulnerability scanning

### 10. Maintenance
- Backup: Optional daily backups
- Update policy: Patch-only auto-updates
- Cleanup: 7 days logs, 30 days metrics, 7 days traces

## Agent Registry

The script includes a built-in registry of all 10 GreenLang AI agents:

| Agent ID | Domain | Type | Memory | CPU | AI Budget | Description |
|----------|--------|------|--------|-----|-----------|-------------|
| fuel_ai | emissions | ai_orchestration | 512 MB | 1.0 | $0.50 | Fuel emissions calculator |
| carbon_ai | emissions | ai_orchestration | 512 MB | 1.0 | $0.50 | Carbon footprint analyzer |
| grid_factor_ai | emissions | ai_orchestration | 512 MB | 1.0 | $0.50 | Grid emission factor calculator |
| recommendation_ai | analytics | ai_orchestration | 768 MB | 1.5 | $0.75 | Emission reduction recommendations |
| report_ai | reporting | ai_orchestration | 1024 MB | 1.5 | $1.00 | Sustainability report generator |
| forecast_sarima | analytics | ml_inference | 1024 MB | 2.0 | $0.00 | SARIMA time series forecasting |
| anomaly_iforest | analytics | ml_inference | 768 MB | 1.5 | $0.00 | Isolation Forest anomaly detection |
| industrial_process_heat_ai | industrial | ai_orchestration | 512 MB | 1.0 | $0.50 | Industrial process heat analyzer |
| boiler_replacement_ai | industrial | ai_orchestration | 512 MB | 1.0 | $0.50 | Boiler replacement advisor |
| industrial_heat_pump_ai | industrial | ai_orchestration | 512 MB | 1.0 | $0.50 | Industrial heat pump analyzer |

## Advanced Usage

### Custom Output Directory

```bash
python scripts/apply_deployment_template.py \
  --agent fuel_ai \
  --output ./custom_packs
```

### Override Agent Domain

```bash
python scripts/apply_deployment_template.py \
  --agent fuel_ai \
  --domain custom_domain
```

### Custom Template Path

```bash
python scripts/apply_deployment_template.py \
  --agent fuel_ai \
  --template ./my_custom_template.yaml
```

## Customization

### Modifying the Template

1. Edit `templates/agent_deployment_pack.yaml`
2. Modify any section (metadata, resources, APIs, security, etc.)
3. Save the file
4. Re-run the script to generate updated deployment packs

### Agent-Specific Overrides

Add overrides in the `agent_specific` section at the bottom of the template:

```yaml
agent_specific:
  # For high-memory agents
  resource_requirements:
    memory_mb: 2048
    cpu_cores: 4.0

  # For GPU-accelerated agents
  resource_requirements:
    gpu_required: true
    gpu_specs:
      type: "nvidia-tesla-t4"
      memory_gb: 16

  # For agents with higher AI budgets
  configuration:
    ai_settings:
      default_budget_usd: 2.0
```

### Extending the Agent Registry

Add new agents to `scripts/apply_deployment_template.py`:

```python
AGENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ... existing agents ...

    "my_new_agent": {
        "domain": "custom",
        "name": "My New Agent",
        "description": "Description of what this agent does",
        "agent_type": "ai_orchestration",
        "memory_mb": 512,
        "cpu_cores": 1.0,
        "ai_budget_usd": 0.50,
        "module_path": "greenlang.agents.my_new_agent",
        "class_name": "MyNewAgent",
    },
}
```

## Validation

### YAML Validation

The script automatically validates generated YAML:

```bash
python scripts/apply_deployment_template.py --agent fuel_ai
# [SUCCESS] YAML validation passed
```

### Manual Validation

```bash
# Using Python YAML parser
python -c "import yaml; yaml.safe_load(open('packs/fuel_ai/deployment_pack.yaml'))"

# Using yamllint (if installed)
yamllint packs/fuel_ai/deployment_pack.yaml
```

## Deployment Workflow

### Step 1: Generate Deployment Pack

```bash
python scripts/apply_deployment_template.py --agent fuel_ai
```

### Step 2: Review Generated Configuration

```bash
cat packs/fuel_ai/deployment_pack.yaml
```

### Step 3: Customize (Optional)

Edit `packs/fuel_ai/deployment_pack.yaml` to add agent-specific settings.

### Step 4: Deploy to Kubernetes

```bash
# Convert to Kubernetes manifests (requires custom tooling)
gl deploy pack packs/fuel_ai/deployment_pack.yaml

# Or manually create Kubernetes resources
kubectl apply -f packs/fuel_ai/k8s/
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy Agent

on:
  push:
    branches: [main]
    paths:
      - 'greenlang/agents/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Generate Deployment Pack
        run: |
          python scripts/apply_deployment_template.py --agent fuel_ai

      - name: Validate Configuration
        run: |
          yamllint packs/fuel_ai/deployment_pack.yaml

      - name: Deploy to Kubernetes
        run: |
          kubectl apply -f packs/fuel_ai/k8s/
```

## Troubleshooting

### Issue: Agent not found in registry

```
[ERROR] Agent 'unknown_agent' not found in registry
```

**Solution**: Use `--list` to see available agents, or add the agent to the registry.

### Issue: YAML validation failed

```
[WARNING] YAML validation failed: ...
```

**Solution**: Check the generated file for syntax errors. Use a YAML validator or linter.

### Issue: File permission denied

```
[ERROR] Error writing deployment pack: Permission denied
```

**Solution**: Ensure you have write permissions to the output directory.

## Best Practices

1. **Version Control**: Commit generated deployment packs to version control
2. **Review Changes**: Always review generated configurations before deploying
3. **Environment-Specific**: Maintain separate configs for dev/staging/production
4. **Security**: Never commit secrets; use environment variables or secret managers
5. **Testing**: Test deployment packs in staging before production
6. **Documentation**: Update documentation when modifying the template
7. **Validation**: Always run dry-run mode first to preview changes

## Support

For questions or issues:

- **Email**: support@greenlang.com
- **Slack**: #greenlang-agents
- **GitHub**: https://github.com/greenlang/agents/issues

## Changelog

### Version 1.0.0 (October 2025)
- Initial release
- Support for all 10 AI agents
- Comprehensive deployment configuration
- Automated generation script
- Interactive and batch modes
- YAML validation
- Dry-run capability

## License

Proprietary - GreenLang Framework Team

---

**Generated by GreenLang Framework Team | October 2025**
