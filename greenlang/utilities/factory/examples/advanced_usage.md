# GreenLang Agent Factory - Advanced Usage Examples

## CLI Usage Examples

### 1. Creating an Agent with Full Options

```bash
# Create a Scope 3 emissions calculator with all bells and whistles
greenlang agent create scope3_emissions.yaml \
  --output-dir=./production-agents \
  --template=industrial \
  --framework=langchain \
  --validate \
  --verbose \
  --config=./production.yaml

# Output:
Creating agent: scope3-emissions-calculator
[████████████████████████████] 100% | Step 6/6: Finalizing agent package
├─ Validating specification... ✓ (12/12 dimensions passed)
├─ Generating implementation... ✓ (8 files created)
├─ Creating test suite... ✓ (145 tests generated)
├─ Generating documentation... ✓ (6 documents created)
├─ Running validation checks... ✓ (All quality gates passed)
└─ Finalizing agent package... ✓

Agent created successfully: scope3-emissions-calculator
Location: ./production-agents/scope3-emissions-calculator
Quality Score: 98.5/100
Test Coverage: 92%
Determinism: 100% (verified over 1000 iterations)

Time elapsed: 00:03:42
```

### 2. Interactive Agent Scaffolding

```bash
# Start interactive scaffolding
greenlang agent scaffold my-regulatory-agent --interactive

# Interactive prompts:
? Select agent category: regulatory
? Choose template: regulatory
? Select framework: langgraph
? Target language: python
? Add capabilities:
  ✓ Data validation
  ✓ Compliance checking
  ✓ Report generation
  ✓ Audit logging
? Configure quality targets:
  Determinism: 100%
  Accuracy: 99.9%
  Coverage: 90%
? Include components:
  ✓ Unit tests
  ✓ Integration tests
  ✓ Documentation
  ✓ CI/CD pipeline

Generating agent scaffold...
✓ Created specification: my-regulatory-agent_spec.yaml
✓ Created implementation skeleton
✓ Created test templates
✓ Created documentation structure
✓ Created CI/CD configuration

Agent scaffolded successfully!
Next steps:
1. Review and update my-regulatory-agent_spec.yaml
2. Run: greenlang agent create my-regulatory-agent_spec.yaml
```

### 3. Comprehensive Validation with Custom Rules

```bash
# Validate with custom rules and strict mode
greenlang agent validate emissions-agent \
  --dimension=determinism,security,compliance \
  --strict \
  --rules=./custom-validation-rules.yaml \
  --report=validation-report.html

# Output:
Validating agent: emissions-agent
Running 3 dimension checks with custom rules...

✓ Determinism (Score: 100/100)
  • No random operations without seeds
  • All calculations are reproducible
  • Hash verification passed

⚠ Security (Score: 85/100)
  Warning: Potential SQL injection risk in query builder
  File: agents/emissions-agent/database.py:45
  Suggestion: Use parameterized queries

✗ Compliance (Score: 65/100)
  Error: Missing required audit fields for CSRD
  File: agents/emissions-agent/models.py:120
  Required fields: materiality_assessment, double_materiality_score

Overall Score: 83.3/100
Status: FAILED (strict mode - warnings count as failures)

Full report saved to: validation-report.html
```

### 4. Advanced Testing with Performance Benchmarks

```bash
# Run comprehensive tests with benchmarking
greenlang agent test scope3-agent \
  --type=all \
  --coverage-min=90 \
  --parallel=8 \
  --determinism \
  --compliance \
  --performance \
  --report=junit

# Output:
Running test suite for: scope3-agent
Test framework: pytest
Parallel workers: 8

Unit Tests:
  ✓ 125/125 passed (100%)
  Coverage: 94.2%
  Duration: 12.3s

Integration Tests:
  ✓ 48/48 passed (100%)
  Duration: 34.5s

E2E Tests:
  ✓ 15/15 passed (100%)
  Duration: 89.2s

Performance Tests:
  ✓ Throughput: 1,250 ops/sec (target: 1,000)
  ✓ Latency p50: 45ms (target: <100ms)
  ✓ Latency p99: 125ms (target: <500ms)
  ✓ Memory usage: 125MB (target: <500MB)

Determinism Tests:
  ✓ 1000/1000 iterations produced identical results
  ✓ SHA-256 hash consistency verified

Compliance Tests:
  ✓ CSRD requirements: PASS
  ✓ EUDR requirements: PASS
  ✓ SEC requirements: PASS

Test Summary:
  Total: 188 tests
  Passed: 188 (100%)
  Failed: 0
  Coverage: 94.2%
  Duration: 136.0s

Report saved to: test-results.xml
```

### 5. Multi-Platform Build and Deployment

```bash
# Build for multiple platforms
greenlang agent build emission-calculator \
  --platform=docker \
  --optimize \
  --tag=v2.1.0 \
  --registry=registry.greenlang.io \
  --multi-arch

# Deploy with canary strategy
greenlang agent deploy emission-calculator \
  --env=production \
  --strategy=canary \
  --replicas=5 \
  --auto-scale \
  --health-check \
  --rollback-on-failure

# Output:
Building agent: emission-calculator
Platform: docker (multi-arch: amd64, arm64)
Optimizations: enabled

Build steps:
✓ Analyzing dependencies... (238 packages)
✓ Optimizing code... (size reduced by 34%)
✓ Building container...
  • Base image: python:3.11-slim
  • Final size: 145MB
✓ Running security scan... (0 vulnerabilities)
✓ Pushing to registry... registry.greenlang.io/emission-calculator:v2.1.0

Deploying agent: emission-calculator
Environment: production
Strategy: canary (10% -> 50% -> 100%)

Deployment progress:
✓ Phase 1: Deploying to 10% of traffic...
  • Health checks: PASS
  • Error rate: 0.01%
✓ Phase 2: Expanding to 50% of traffic...
  • Health checks: PASS
  • Error rate: 0.02%
✓ Phase 3: Full deployment...
  • Health checks: PASS
  • Error rate: 0.01%

Deployment successful!
Endpoints:
  - https://api.greenlang.io/v2/emission-calculator
  - https://api-eu.greenlang.io/v2/emission-calculator
  - https://api-asia.greenlang.io/v2/emission-calculator

Metrics dashboard: https://metrics.greenlang.io/emission-calculator
```

### 6. Batch Operations for Scale

```bash
# Batch create 50 agents from specs
greenlang agent batch-create ./agent-specs \
  --pattern="scope3/*.yaml" \
  --parallel=10 \
  --validate-first \
  --continue-on-error \
  --report=batch-creation-report.json

# Output:
Batch agent creation started
Found 50 specification files matching pattern: scope3/*.yaml

Phase 1: Validation
[████████████████████████████] 100% | 50/50 specs validated
✓ All specifications valid

Phase 2: Creation (10 parallel workers)
[███████████████████░░░░░░░░░] 72% | 36/50 agents created

Progress:
• Completed: 36
• In Progress: 10
• Remaining: 4
• Failed: 0

[████████████████████████████] 100% | 50/50 agents created

Batch creation complete:
• Successfully created: 50 agents
• Failed: 0
• Total time: 00:18:34
• Average per agent: 22.3s

Report saved to: batch-creation-report.json
```

## Python SDK Usage Examples

### 1. Complete Agent Creation Pipeline

```python
from greenlang.factory import (
    AgentFactory,
    AgentTemplate,
    Framework,
    Language,
    Environment,
    QualityDimension
)
import asyncio

async def create_production_agent():
    """Create a production-ready agent with full validation."""

    # Initialize factory with custom config
    factory = AgentFactory({
        "output_dir": "./production-agents",
        "coverage_min": 90,
        "complexity_max": 8,
        "registry": {
            "url": "https://registry.greenlang.io",
            "auth": {
                "type": "token",
                "credentials": os.environ["GREENLANG_TOKEN"]
            }
        }
    })

    # Create agent from specification
    print("Creating agent from specification...")
    agent = await factory.create_agent(
        "specs/scope3_transport.yaml",
        validate=True,
        force=False,
        with_tests=True,
        with_docs=True
    )

    # Validate all quality dimensions
    print("Validating quality dimensions...")
    validation = await factory.validate_agent(
        agent,
        dimensions=[
            QualityDimension.DETERMINISM,
            QualityDimension.ACCURACY,
            QualityDimension.SECURITY,
            QualityDimension.COMPLIANCE
        ],
        strict=True
    )

    if not validation.passes_all_dimensions():
        print(f"Validation failed: {validation.errors}")
        return

    print(f"Quality score: {validation.score}/100")

    # Generate comprehensive tests
    print("Generating test suite...")
    test_result = await factory.generate_tests(
        agent,
        test_type="all",
        coverage_target=90.0,
        fixtures=True,
        mocks=True
    )

    # Run tests
    print("Running tests...")
    test_execution = await factory.test_agent(
        agent,
        test_type="all",
        coverage_min=90.0,
        parallel=8
    )

    if not test_execution.meets_coverage_target(90.0):
        print(f"Coverage too low: {test_execution.coverage_percent}%")
        return

    print(f"Tests passed: {test_execution.tests_passed}/{test_execution.tests_run}")
    print(f"Coverage: {test_execution.coverage_percent}%")

    # Build for deployment
    print("Building agent...")
    build_result = await factory.build_agent(
        agent,
        platform=Platform.DOCKER,
        optimize=True,
        tag="v1.0.0"
    )

    print(f"Build complete: {build_result.tag}")
    print(f"Size: {build_result.size / 1024 / 1024:.2f}MB")

    # Deploy to staging
    print("Deploying to staging...")
    deployment = await factory.deploy_agent(
        agent,
        env=Environment.STAGING,
        strategy=DeploymentStrategy.BLUE_GREEN,
        replicas=3,
        auto_scale=True,
        health_check=True
    )

    if deployment.success:
        print(f"Deployed successfully!")
        print(f"Endpoints: {deployment.endpoints}")
    else:
        print(f"Deployment failed: {deployment.errors}")

# Run the pipeline
asyncio.run(create_production_agent())
```

### 2. Custom Plugin Development

```python
from greenlang.factory import AgentFactory, Plugin
from typing import Dict, Any

class CustomCompliancePlugin(Plugin):
    """Custom plugin for CSRD compliance validation."""

    def __init__(self):
        self.name = "csrd-compliance"
        self.version = "1.0.0"
        self.enabled = True
        self.description = "CSRD compliance validation plugin"

    def init(self, factory: AgentFactory) -> None:
        """Initialize plugin with factory."""
        # Register custom validators
        factory.register_validator("csrd", self.validate_csrd_compliance)

        # Register custom generators
        factory.register_generator("csrd-report", self.generate_csrd_report)

    def validate_csrd_compliance(
        self,
        agent: Any,
        options: Dict[str, Any]
    ) -> ValidationResult:
        """Validate agent for CSRD compliance."""
        result = ValidationResult(
            agent_name=agent.name,
            timestamp=datetime.now(),
            passed=True,
            score=100.0
        )

        # Check for required CSRD fields
        required_fields = [
            "materiality_assessment",
            "double_materiality_score",
            "eu_taxonomy_alignment",
            "sustainability_metrics"
        ]

        for field in required_fields:
            if field not in agent.specification.outputs:
                result.passed = False
                result.errors.append({
                    "type": "missing_field",
                    "message": f"Required CSRD field missing: {field}",
                    "severity": "error"
                })

        # Check for audit trail
        if not agent.specification.config.get("audit_enabled", False):
            result.warnings.append({
                "type": "audit",
                "message": "Audit trail not enabled for CSRD compliance",
                "severity": "warning"
            })

        return result

    def generate_csrd_report(
        self,
        agent: Any,
        data: Dict[str, Any]
    ) -> str:
        """Generate CSRD-compliant report."""
        # Implementation of CSRD report generation
        pass

# Using the custom plugin
factory = AgentFactory()
factory.install_plugin(CustomCompliancePlugin())

# Now the CSRD validation is available
agent = factory.create_agent("csrd_agent.yaml")
csrd_validation = factory.validate_agent(agent, dimensions=["csrd"])
```

### 3. Parallel Batch Processing

```python
from greenlang.factory import AgentFactory
import asyncio
from concurrent.futures import ProcessPoolExecutor

async def process_agent_batch():
    """Process large batch of agents in parallel."""

    factory = AgentFactory({
        "parallel": 16  # Use 16 workers
    })

    # Find all specification files
    spec_files = [
        "specs/scope1_stationary.yaml",
        "specs/scope1_mobile.yaml",
        "specs/scope2_purchased.yaml",
        "specs/scope3_upstream.yaml",
        "specs/scope3_downstream.yaml",
        # ... 100+ more specs
    ]

    # Progress tracking
    completed = 0
    total = len(spec_files)

    async def create_with_progress(spec_file):
        nonlocal completed
        agent = await factory.create_agent(spec_file)
        completed += 1
        print(f"Progress: {completed}/{total} ({completed*100/total:.1f}%)")
        return agent

    # Create all agents in parallel
    tasks = [create_with_progress(spec) for spec in spec_files]
    agents = await asyncio.gather(*tasks)

    # Test all agents in parallel
    test_tasks = [
        factory.test_agent(agent, coverage_min=85)
        for agent in agents
    ]
    test_results = await asyncio.gather(*test_tasks)

    # Generate summary report
    passed = sum(1 for r in test_results if r.passed)
    avg_coverage = sum(r.coverage_percent for r in test_results) / len(test_results)

    print(f"\nBatch Processing Complete:")
    print(f"• Agents created: {len(agents)}")
    print(f"• Tests passed: {passed}/{len(test_results)}")
    print(f"• Average coverage: {avg_coverage:.1f}%")

    # Deploy successful agents
    successful_agents = [
        agent for agent, result in zip(agents, test_results)
        if result.passed
    ]

    deployment_tasks = [
        factory.deploy_agent(agent, env="staging")
        for agent in successful_agents
    ]
    deployments = await asyncio.gather(*deployment_tasks)

    print(f"• Deployed: {sum(1 for d in deployments if d.success)}/{len(deployments)}")

# Run batch processing
asyncio.run(process_agent_batch())
```

## TypeScript SDK Usage Examples

### 1. Complete TypeScript Pipeline

```typescript
import {
  AgentFactory,
  AgentTemplate,
  Framework,
  Environment,
  QualityDimension
} from '@greenlang/factory';

async function createProductionAgent() {
  // Initialize factory
  const factory = new AgentFactory({
    outputDir: './production-agents',
    coverageMin: 90,
    registry: {
      url: 'https://registry.greenlang.io',
      auth: {
        type: 'token',
        credentials: process.env.GREENLANG_TOKEN
      }
    }
  });

  // Listen to events
  factory.on('create:progress', (info) => {
    console.log(`Progress: ${info.percentage}% - ${info.step}`);
  });

  factory.on('validate:complete', (result) => {
    console.log(`Validation score: ${result.score}/100`);
  });

  try {
    // Create agent
    const agent = await factory.createAgent('specs/emissions.yaml', {
      validate: true,
      withTests: true,
      withDocs: true,
      onProgress: (info) => {
        console.log(`${info.current}/${info.total}: ${info.message}`);
      }
    });

    // Validate quality
    const validation = await factory.validateAgent(agent, {
      dimensions: [
        QualityDimension.DETERMINISM,
        QualityDimension.SECURITY,
        QualityDimension.COMPLIANCE
      ],
      strict: true
    });

    if (!validation.passesAllDimensions()) {
      throw new Error(`Validation failed: ${validation.errors}`);
    }

    // Run tests
    const testResult = await factory.testAgent(agent, {
      testType: 'all',
      coverageMin: 90,
      parallel: 8,
      report: 'junit'
    });

    console.log(`Tests: ${testResult.testsPassed}/${testResult.testsRun}`);
    console.log(`Coverage: ${testResult.coveragePercent}%`);

    // Build and deploy
    const buildResult = await factory.buildAgent(agent, {
      platform: 'docker',
      optimize: true,
      tag: 'v1.0.0'
    });

    const deployment = await factory.deployAgent(agent, {
      env: Environment.PRODUCTION,
      strategy: 'canary',
      replicas: 5,
      autoScale: true
    });

    console.log('Deployment successful!');
    console.log('Endpoints:', deployment.endpoints);

  } catch (error) {
    console.error('Pipeline failed:', error);
    process.exit(1);
  }
}

// Run the pipeline
createProductionAgent();
```

### 2. React Integration Example

```typescript
// AgentDashboard.tsx
import React, { useState, useEffect } from 'react';
import { AgentFactory, Agent, ValidationResult } from '@greenlang/factory';

const AgentDashboard: React.FC = () => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [validations, setValidations] = useState<Map<string, ValidationResult>>(new Map());
  const [loading, setLoading] = useState(false);

  const factory = new AgentFactory();

  const createAgent = async (specFile: File) => {
    setLoading(true);
    try {
      // Upload and create agent
      const agent = await factory.createAgent(specFile, {
        validate: true,
        withTests: true
      });

      // Validate agent
      const validation = await factory.validateAgent(agent);

      setAgents([...agents, agent]);
      setValidations(new Map(validations).set(agent.name, validation));
    } catch (error) {
      console.error('Failed to create agent:', error);
    } finally {
      setLoading(false);
    }
  };

  const deployAgent = async (agent: Agent) => {
    try {
      const result = await factory.deployAgent(agent, {
        env: 'staging',
        strategy: 'rolling',
        replicas: 3
      });

      if (result.success) {
        alert(`Agent deployed: ${result.endpoints.join(', ')}`);
      }
    } catch (error) {
      console.error('Deployment failed:', error);
    }
  };

  return (
    <div className="agent-dashboard">
      <h1>GreenLang Agent Factory</h1>

      <div className="upload-section">
        <input
          type="file"
          accept=".yaml,.yml"
          onChange={(e) => e.target.files && createAgent(e.target.files[0])}
          disabled={loading}
        />
      </div>

      <div className="agents-grid">
        {agents.map(agent => {
          const validation = validations.get(agent.name);
          return (
            <div key={agent.name} className="agent-card">
              <h3>{agent.name}</h3>
              <p>Version: {agent.version}</p>
              <p>Framework: {agent.specification.framework}</p>

              {validation && (
                <div className="validation-status">
                  <p>Quality Score: {validation.score}/100</p>
                  <p>Status: {validation.passed ? '✅ Passed' : '❌ Failed'}</p>
                </div>
              )}

              <button onClick={() => deployAgent(agent)}>
                Deploy to Staging
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default AgentDashboard;
```

## Advanced Configuration Examples

### 1. Multi-Environment Configuration

`.greenlang.yaml`:
```yaml
version: 1.0
project:
  name: greenlang-emissions-platform
  description: Enterprise emissions calculation platform

environments:
  development:
    output_dir: ./agents-dev
    coverage_min: 70
    complexity_max: 15
    deployment:
      replicas: 1
      auto_scale: false
      platform: docker

  staging:
    output_dir: ./agents-staging
    coverage_min: 85
    complexity_max: 10
    deployment:
      replicas: 2
      auto_scale: true
      platform: kubernetes
      strategy: rolling

  production:
    output_dir: ./agents-prod
    coverage_min: 95
    complexity_max: 8
    deployment:
      replicas: 5
      auto_scale: true
      platform: kubernetes
      strategy: blue-green
      health_check:
        enabled: true
        interval: 30s
        timeout: 10s
        retries: 3

quality:
  dimensions:
    determinism:
      enabled: true
      iterations: 1000
      tolerance: 0.0

    security:
      level: strict
      scan_dependencies: true
      check_secrets: true
      fail_on: medium

    compliance:
      frameworks:
        - CSRD
        - EUDR
        - SEC
      validate_outputs: true
      audit_trail: required

plugins:
  - name: "@greenlang/csrd-validator"
    version: "2.0.0"
    enabled: true
    config:
      strict_mode: true

  - name: "@greenlang/performance-profiler"
    version: "1.5.0"
    enabled: true
    config:
      benchmark_iterations: 100

  - name: "@greenlang/deployment-verifier"
    version: "1.2.0"
    enabled: true
```

### 2. CI/CD Pipeline Integration

`.github/workflows/agent-pipeline.yml`:
```yaml
name: Agent Factory Pipeline

on:
  push:
    paths:
      - 'agent-specs/**/*.yaml'

jobs:
  create-and-test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Install GreenLang CLI
        run: npm install -g @greenlang/cli

      - name: Validate Specifications
        run: |
          for spec in agent-specs/**/*.yaml; do
            greenlang agent validate "$spec" --strict
          done

      - name: Create Agents
        run: |
          greenlang agent batch-create ./agent-specs \
            --parallel=4 \
            --validate-first

      - name: Run Tests
        run: |
          greenlang agent batch-test ./agents \
            --coverage-min=90 \
            --report=junit \
            --parallel=8

      - name: Security Scan
        run: |
          for agent in agents/*; do
            greenlang agent security-scan "$(basename $agent)" \
              --level=strict \
              --fail-on=high
          done

      - name: Build Containers
        if: github.ref == 'refs/heads/main'
        run: |
          for agent in agents/*; do
            greenlang agent build "$(basename $agent)" \
              --platform=docker \
              --optimize \
              --tag=${{ github.sha }}
          done

      - name: Deploy to Staging
        if: github.ref == 'refs/heads/main'
        run: |
          greenlang agent batch-deploy deployment-list.txt \
            --env=staging \
            --strategy=canary \
            --stagger=30
```

## Performance Optimization Tips

1. **Parallel Processing**: Use `--parallel` flag for batch operations
2. **Caching**: Enable build cache with `--cache` flag
3. **Incremental Builds**: Use `--incremental` for faster rebuilds
4. **Resource Limits**: Configure memory/CPU limits in deployment config
5. **Lazy Loading**: Use async imports in TypeScript SDK
6. **Connection Pooling**: Reuse factory instances in long-running applications

## Troubleshooting Common Issues

### Issue: Validation failures
```bash
# Get detailed validation report
greenlang agent validate my-agent --verbose --report=detailed.html
```

### Issue: Test coverage too low
```bash
# Generate additional test cases
greenlang agent generate-tests my-agent --type=unit --golden --fixtures
```

### Issue: Deployment failures
```bash
# Check deployment status and logs
greenlang agent status my-agent --env=prod --detailed --logs
```

### Issue: Performance problems
```bash
# Run performance profiling
greenlang agent test my-agent --type=performance --benchmark --report=perf.html
```

## Best Practices

1. **Always validate before deployment**
2. **Use strict mode in production**
3. **Maintain 90%+ test coverage**
4. **Enable all security scans**
5. **Use canary deployments for critical agents**
6. **Monitor agent metrics post-deployment**
7. **Keep audit logs for compliance**
8. **Version all agent specifications**
9. **Use semantic versioning for agents**
10. **Implement proper error handling in custom plugins**