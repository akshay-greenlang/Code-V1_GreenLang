# GreenLang Agent Factory CLI & SDK Specification

## Overview

The GreenLang Agent Factory provides a developer-friendly CLI and SDK for generating, testing, and deploying production-ready agents that maintain 12-dimension quality standards across 10,000+ agent types.

## CLI Command Reference

### Core Generation Commands

#### `greenlang agent create`
Creates a new agent from specification.

```bash
greenlang agent create <spec.yaml> [options]

Options:
  --output-dir=<path>        Output directory (default: ./agents)
  --name=<name>              Override agent name from spec
  --validate                 Validate spec before generation (default: true)
  --dry-run                  Preview without generating files
  --force                    Overwrite existing files
  --template=<type>          Base template (industrial|hvac|crosscutting|regulatory)
  --framework=<fw>           Target framework (langchain|langgraph|crewai|autogen)
  --language=<lang>          Implementation language (python|typescript)
  --verbose                  Show detailed progress
  --parallel=<n>             Parallel workers for batch operations
  --config=<file>            Custom configuration file

Examples:
  greenlang agent create emissions_calc.yaml
  greenlang agent create spec.yaml --output-dir=./my-agents --template=industrial
  greenlang agent create batch/*.yaml --parallel=4
```

#### `greenlang agent scaffold`
Generates agent boilerplate and specification template.

```bash
greenlang agent scaffold <agent-name> [options]

Options:
  --template=<type>          Base template (default: base)
  --category=<cat>           Agent category (scope1|scope2|scope3|regulatory)
  --framework=<fw>           Target framework
  --interactive              Interactive mode with prompts
  --spec-only                Generate only specification file
  --impl-only                Generate only implementation files
  --with-tests               Include test scaffolding
  --with-docs                Include documentation templates

Examples:
  greenlang agent scaffold emission-calculator --template=industrial
  greenlang agent scaffold my-agent --interactive
```

#### `greenlang agent validate`
Validates agent specification against standards.

```bash
greenlang agent validate <spec.yaml> [options]

Options:
  --dimension=<dim>          Validate specific dimension(s)
  --strict                   Fail on warnings
  --json                     Output as JSON
  --report=<file>            Save validation report
  --rules=<file>             Custom validation rules
  --schema=<version>         Schema version (default: latest)

Examples:
  greenlang agent validate spec.yaml
  greenlang agent validate spec.yaml --dimension=determinism,security
  greenlang agent validate batch/*.yaml --json > validation.json
```

#### `greenlang agent generate-tests`
Generates comprehensive test suite.

```bash
greenlang agent generate-tests <agent-name> [options]

Options:
  --type=<type>              Test types (unit|integration|e2e|performance|all)
  --coverage-target=<n>      Target coverage percentage (default: 85)
  --framework=<fw>           Test framework (pytest|jest|mocha)
  --fixtures                 Generate test fixtures
  --mocks                    Generate mock data
  --golden                   Generate golden test files
  --compliance               Include regulatory compliance tests
  --benchmark                Include performance benchmarks

Examples:
  greenlang agent generate-tests emission-agent --type=all
  greenlang agent generate-tests my-agent --coverage-target=95 --fixtures
```

#### `greenlang agent generate-docs`
Generates comprehensive documentation.

```bash
greenlang agent generate-docs <agent-name> [options]

Options:
  --format=<fmt>             Output format (markdown|html|pdf|openapi)
  --include=<sections>       Sections to include (api|usage|architecture|all)
  --language=<lang>          Documentation language (en|es|fr|de)
  --examples                 Include usage examples
  --diagrams                 Generate architecture diagrams
  --api-spec                 Generate OpenAPI specification
  --changelog                Generate changelog template

Examples:
  greenlang agent generate-docs emission-agent --format=markdown
  greenlang agent generate-docs my-agent --format=html --diagrams
```

### Quality & Testing Commands

#### `greenlang agent test`
Runs agent test suite.

```bash
greenlang agent test <agent-name> [options]

Options:
  --type=<type>              Test types to run
  --coverage-min=<n>         Minimum coverage required (default: 85)
  --parallel=<n>             Parallel test workers
  --bail                     Stop on first failure
  --watch                    Watch mode for development
  --report=<format>          Test report format (junit|html|json)
  --determinism              Run determinism tests
  --compliance               Run compliance tests
  --performance              Run performance tests
  --integration              Run integration tests

Examples:
  greenlang agent test emission-agent --coverage-min=90
  greenlang agent test my-agent --type=unit,integration --parallel=4
```

#### `greenlang agent lint`
Runs code quality checks.

```bash
greenlang agent lint <agent-name> [options]

Options:
  --fix                      Auto-fix issues where possible
  --format=<fmt>             Output format (stylish|json|junit)
  --rules=<file>             Custom lint rules
  --ignore=<patterns>        Ignore patterns
  --max-warnings=<n>         Maximum warnings allowed
  --complexity-max=<n>       Maximum cyclomatic complexity

Examples:
  greenlang agent lint emission-agent --fix
  greenlang agent lint my-agent --format=json --max-warnings=0
```

#### `greenlang agent security-scan`
Performs security analysis.

```bash
greenlang agent security-scan <agent-name> [options]

Options:
  --level=<level>            Security level (basic|standard|strict)
  --cve-check                Check for known CVEs
  --dependency-check         Scan dependencies
  --secret-scan              Scan for secrets/credentials
  --sast                     Static application security testing
  --report=<file>            Security report output
  --fail-on=<severity>       Fail on severity (low|medium|high|critical)

Examples:
  greenlang agent security-scan emission-agent --level=strict
  greenlang agent security-scan my-agent --cve-check --dependency-check
```

#### `greenlang agent determinism-check`
Verifies deterministic behavior.

```bash
greenlang agent determinism-check <agent-name> [options]

Options:
  --iterations=<n>           Number of test iterations (default: 100)
  --input-file=<file>        Test input data file
  --tolerance=<n>            Numerical tolerance for calculations
  --seed=<n>                 Random seed for reproducibility
  --report=<file>            Determinism report output
  --fail-fast                Stop on first non-deterministic behavior

Examples:
  greenlang agent determinism-check emission-agent --iterations=1000
  greenlang agent determinism-check my-agent --input-file=test-data.json
```

### Deployment Commands

#### `greenlang agent build`
Builds agent for deployment.

```bash
greenlang agent build <agent-name> [options]

Options:
  --platform=<platform>      Target platform (docker|k8s|lambda|azure|gcp)
  --optimize                 Apply optimizations
  --minify                   Minify code
  --bundle                   Bundle dependencies
  --registry=<url>           Container registry URL
  --tag=<tag>                Version tag
  --multi-arch               Build for multiple architectures
  --cache                    Use build cache

Examples:
  greenlang agent build emission-agent --platform=docker
  greenlang agent build my-agent --platform=k8s --optimize --tag=v1.0.0
```

#### `greenlang agent deploy`
Deploys agent to environment.

```bash
greenlang agent deploy <agent-name> [options]

Options:
  --env=<env>                Target environment (dev|staging|prod)
  --strategy=<strategy>      Deployment strategy (rolling|blue-green|canary)
  --replicas=<n>             Number of replicas
  --auto-scale               Enable auto-scaling
  --health-check             Enable health checks
  --rollback-on-failure      Auto rollback on failure
  --config=<file>            Deployment configuration
  --dry-run                  Preview deployment

Examples:
  greenlang agent deploy emission-agent --env=staging
  greenlang agent deploy my-agent --env=prod --strategy=canary --replicas=3
```

#### `greenlang agent status`
Shows agent deployment status.

```bash
greenlang agent status <agent-name> [options]

Options:
  --env=<env>                Environment to check
  --detailed                 Show detailed status
  --metrics                  Include metrics
  --logs                     Show recent logs
  --health                   Include health status
  --format=<fmt>             Output format (table|json|yaml)

Examples:
  greenlang agent status emission-agent --env=prod
  greenlang agent status my-agent --detailed --metrics
```

### Batch Operations Commands

#### `greenlang agent batch-create`
Creates multiple agents from specifications.

```bash
greenlang agent batch-create <specs-dir> [options]

Options:
  --pattern=<glob>           File pattern (default: *.yaml)
  --parallel=<n>             Parallel workers (default: 4)
  --continue-on-error        Continue on failures
  --report=<file>            Batch operation report
  --validate-first           Validate all before creating
  --progress                 Show progress bar

Examples:
  greenlang agent batch-create ./specs
  greenlang agent batch-create ./specs --pattern="scope3/*.yaml" --parallel=8
```

#### `greenlang agent batch-test`
Tests multiple agents.

```bash
greenlang agent batch-test <agents-dir> [options]

Options:
  --pattern=<glob>           Agent pattern
  --parallel=<n>             Parallel workers
  --fail-fast                Stop on first failure
  --report=<file>            Combined test report
  --coverage-report=<file>   Combined coverage report

Examples:
  greenlang agent batch-test ./agents
  greenlang agent batch-test ./agents --pattern="scope3-*" --parallel=4
```

#### `greenlang agent batch-deploy`
Deploys multiple agents.

```bash
greenlang agent batch-deploy <agents-list.txt> [options]

Options:
  --env=<env>                Target environment
  --strategy=<strategy>      Deployment strategy
  --stagger=<seconds>        Stagger deployments
  --rollback-all-on-failure  Rollback all on any failure
  --verify                   Verify each deployment

Examples:
  greenlang agent batch-deploy agents.txt --env=prod
  greenlang agent batch-deploy list.txt --env=staging --stagger=30
```

### Management Commands

#### `greenlang agent list`
Lists available agents.

```bash
greenlang agent list [options]

Options:
  --env=<env>                Environment filter
  --status=<status>          Status filter (deployed|pending|failed)
  --category=<cat>           Category filter
  --format=<fmt>             Output format
  --sort=<field>             Sort by field

Examples:
  greenlang agent list --env=prod
  greenlang agent list --category=scope3 --format=json
```

#### `greenlang agent info`
Shows detailed agent information.

```bash
greenlang agent info <agent-name> [options]

Options:
  --env=<env>                Environment
  --version=<ver>            Specific version
  --show-spec                Include specification
  --show-config              Include configuration
  --show-metrics             Include metrics

Examples:
  greenlang agent info emission-agent
  greenlang agent info my-agent --env=prod --show-metrics
```

#### `greenlang agent logs`
Shows agent logs.

```bash
greenlang agent logs <agent-name> [options]

Options:
  --env=<env>                Environment
  --tail=<n>                 Last N lines
  --follow                   Follow log output
  --since=<time>             Since timestamp
  --filter=<pattern>         Filter pattern
  --level=<level>            Log level filter

Examples:
  greenlang agent logs emission-agent --env=prod --tail=100
  greenlang agent logs my-agent --follow --level=error
```

#### `greenlang agent rollback`
Rolls back agent deployment.

```bash
greenlang agent rollback <agent-name> [options]

Options:
  --env=<env>                Environment
  --version=<ver>            Target version (default: previous)
  --verify                   Verify after rollback
  --force                    Force rollback

Examples:
  greenlang agent rollback emission-agent --env=prod
  greenlang agent rollback my-agent --env=staging --version=v1.2.3
```

### Configuration Management

#### `greenlang config init`
Initializes configuration.

```bash
greenlang config init [options]

Options:
  --global                   Initialize global config
  --project                  Initialize project config
  --interactive              Interactive setup
  --template=<name>          Config template

Examples:
  greenlang config init --project
  greenlang config init --global --interactive
```

#### `greenlang config set`
Sets configuration value.

```bash
greenlang config set <key> <value> [options]

Options:
  --global                   Set in global config
  --project                  Set in project config
  --env=<env>                Environment-specific

Examples:
  greenlang config set registry.url "https://registry.greenlang.io"
  greenlang config set deploy.strategy "blue-green" --env=prod
```

#### `greenlang config get`
Gets configuration value.

```bash
greenlang config get <key> [options]

Options:
  --global                   Get from global config
  --project                  Get from project config
  --env=<env>                Environment-specific

Examples:
  greenlang config get registry.url
  greenlang config get deploy.strategy --env=prod
```

### Plugin Management

#### `greenlang plugin install`
Installs a plugin.

```bash
greenlang plugin install <plugin-name> [options]

Options:
  --version=<ver>            Plugin version
  --global                   Install globally
  --force                    Force install

Examples:
  greenlang plugin install @greenlang/custom-generator
  greenlang plugin install my-plugin --version=1.2.3
```

#### `greenlang plugin list`
Lists installed plugins.

```bash
greenlang plugin list [options]

Options:
  --global                   List global plugins
  --enabled                  Show only enabled
  --format=<fmt>             Output format

Examples:
  greenlang plugin list
  greenlang plugin list --global --enabled
```

## Global Options

All commands support these global options:

```bash
Global Options:
  --help, -h                 Show help
  --version, -v              Show version
  --quiet, -q                Suppress output
  --verbose                  Verbose output
  --debug                    Debug mode
  --no-color                 Disable colored output
  --config=<file>            Configuration file
  --profile=<name>           Configuration profile
  --output=<format>          Output format (text|json|yaml)
  --log-file=<file>          Log to file
  --log-level=<level>        Log level (debug|info|warn|error)
```

## Environment Variables

```bash
GREENLANG_HOME              GreenLang installation directory
GREENLANG_CONFIG            Configuration file path
GREENLANG_REGISTRY          Default registry URL
GREENLANG_ENV               Default environment
GREENLANG_LOG_LEVEL         Default log level
GREENLANG_PARALLEL          Default parallelism
GREENLANG_TIMEOUT           Command timeout
GREENLANG_CACHE_DIR         Cache directory
GREENLANG_PLUGIN_PATH       Plugin search path
GREENLANG_API_KEY           API key for cloud services
```

## Configuration File

`.greenlang.yaml`:
```yaml
version: 1.0
project:
  name: my-project
  description: My GreenLang project

defaults:
  output_dir: ./agents
  template: base
  framework: langchain
  language: python

testing:
  coverage_min: 85
  parallel: 4
  frameworks:
    python: pytest
    typescript: jest

deployment:
  environments:
    dev:
      replicas: 1
      auto_scale: false
    staging:
      replicas: 2
      auto_scale: true
      strategy: rolling
    prod:
      replicas: 3
      auto_scale: true
      strategy: blue-green

registry:
  url: https://registry.greenlang.io
  auth:
    type: token

quality:
  lint:
    max_warnings: 10
    complexity_max: 10
  security:
    level: standard
    fail_on: high

plugins:
  - name: "@greenlang/custom-generator"
    version: "1.0.0"
    enabled: true
```

## Exit Codes

- `0`: Success
- `1`: General error
- `2`: Validation error
- `3`: Build error
- `4`: Test failure
- `5`: Deployment error
- `6`: Security violation
- `7`: Determinism failure
- `8`: Coverage below threshold
- `9`: Configuration error
- `10`: Plugin error
- `127`: Command not found

## Progress Reporting

Long-running operations show progress:

```
Creating agent: emission-calculator
[████████████████████░░░░░░░░] 67% | Step 4/6: Generating tests
├─ Validating specification... ✓
├─ Generating implementation... ✓
├─ Creating test suite... ✓
├─ Generating documentation... ⚡
├─ Running validation checks...
└─ Finalizing agent package...

Time elapsed: 00:02:34
Estimated remaining: 00:01:12
```

## Error Messages

Structured error reporting:

```
ERROR: Agent validation failed

  ✗ Dimension: Determinism
    Issue: Non-deterministic calculation detected
    File: agents/emission-agent/calculator.py:45
    Details: Random sampling without seed

  ✗ Dimension: Security
    Issue: Hardcoded credentials found
    File: agents/emission-agent/config.py:12
    Details: API key exposed in source

  Suggested fixes:
    - Use seeded random for reproducibility
    - Move credentials to environment variables

  Run 'greenlang agent validate --help' for more information
```