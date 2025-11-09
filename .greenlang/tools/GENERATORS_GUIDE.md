# GreenLang Generators Guide

Complete guide to GreenLang's powerful boilerplate generation system.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Application Generator](#application-generator)
4. [Component Generator](#component-generator)
5. [Configuration Generator](#configuration-generator)
6. [Test Generator](#test-generator)
7. [CI/CD Generator](#cicd-generator)
8. [Deployment Generator](#deployment-generator)
9. [Best Practices](#best-practices)
10. [Examples](#examples)

---

## Overview

GreenLang's generator system allows you to create production-ready applications with a single command. All generated code uses ONLY GreenLang infrastructure - no custom implementations.

### What Gets Generated

- **Complete applications** with agents, tests, and CI/CD
- **Infrastructure configuration** for all environments
- **Deployment manifests** for Kubernetes, Docker, Terraform
- **Comprehensive tests** with fixtures and mocking
- **CI/CD pipelines** for GitHub, GitLab, Jenkins

### Philosophy

- **Infrastructure-First**: All code uses GreenLang infrastructure
- **Best Practices**: Follows industry standards and GreenLang patterns
- **Production-Ready**: Includes monitoring, security, and scaling
- **Customizable**: Easy to extend and modify generated code

---

## Quick Start

### Create Your First Application

```bash
# Interactive mode (recommended for beginners)
greenlang create-app my-sustainability-app

# With options (for experienced users)
greenlang create-app emissions-tracker \
  --template calculation \
  --llm openai \
  --cache redis \
  --database postgresql
```

### Add Components to Existing Apps

```bash
# Add an agent
greenlang add agent EmissionsCalculator --template calculator

# Add LLM integration
greenlang add llm --provider openai --caching

# Add caching
greenlang add cache --type redis

# Add database
greenlang add database --type postgresql
```

### Generate Supporting Files

```bash
# Generate tests
greenlang generate-tests app/agents/my_agent.py

# Generate CI/CD
greenlang generate-cicd --platform github

# Generate deployment configs
greenlang generate-deployment --platform kubernetes
```

---

## Application Generator

Create complete GreenLang applications from templates.

### Command

```bash
greenlang create-app <name> [options]
```

### Templates

#### 1. Data Intake (`data-intake`)

For validating and processing incoming data files.

**Features:**
- CSV/Excel/JSON validation
- Data transformation
- Database loading
- Error reporting

**Agents:**
- `DataValidatorAgent`: Validates input data
- `DataTransformerAgent`: Transforms data formats
- `DataLoaderAgent`: Loads data to database

**Use Cases:**
- ESG data collection
- Carbon footprint data intake
- Supplier data processing

**Example:**
```bash
greenlang create-app esg-data-intake \
  --template data-intake \
  --database postgresql
```

#### 2. Calculation Engine (`calculation`)

For complex sustainability calculations.

**Features:**
- Formula-based calculations
- Batch processing
- Result caching
- Aggregation

**Agents:**
- `CalculatorAgent`: Performs calculations
- `ValidatorAgent`: Validates inputs/outputs
- `AggregatorAgent`: Aggregates results

**Use Cases:**
- Emissions calculations
- Carbon credit calculations
- Sustainability metrics

**Example:**
```bash
greenlang create-app emissions-calculator \
  --template calculation \
  --cache redis \
  --database postgresql
```

#### 3. LLM Analysis (`llm-analysis`)

For AI-powered analysis and insights.

**Features:**
- LLM integration (OpenAI/Anthropic)
- Intelligent caching
- Report generation
- Classification

**Agents:**
- `LLMAnalyzerAgent`: Analyzes data with AI
- `SummarizerAgent`: Generates summaries
- `ClassifierAgent`: Classifies content

**Use Cases:**
- Sustainability report analysis
- Policy compliance checking
- Risk assessment

**Example:**
```bash
greenlang create-app report-analyzer \
  --template llm-analysis \
  --llm openai \
  --cache redis
```

#### 4. Data Pipeline (`pipeline`)

For multi-stage data processing workflows.

**Features:**
- Sequential processing
- Parallel execution support
- Error recovery
- Progress tracking

**Agents:**
- `IngestAgent`: Data ingestion
- `ProcessAgent`: Data processing
- `ValidateAgent`: Validation
- `ExportAgent`: Export results

**Use Cases:**
- ETL pipelines
- Data transformation workflows
- Multi-stage validation

**Example:**
```bash
greenlang create-app data-pipeline \
  --template pipeline \
  --database postgresql \
  --cache redis
```

#### 5. Reporting Application (`reporting`)

For generating sustainability reports.

**Features:**
- Data aggregation
- Report generation
- Multiple export formats
- LLM-powered insights

**Agents:**
- `DataCollectorAgent`: Collects data
- `ReportGeneratorAgent`: Generates reports
- `ExportAgent`: Exports to various formats

**Use Cases:**
- CSRD reporting
- Carbon disclosure
- ESG reporting

**Example:**
```bash
greenlang create-app csrd-reporter \
  --template reporting \
  --llm anthropic \
  --database postgresql
```

#### 6. API Service (`api-service`)

For REST API endpoints.

**Features:**
- Request validation
- Response formatting
- Rate limiting
- Caching

**Agents:**
- `RequestValidatorAgent`: Validates requests
- `ProcessorAgent`: Processes requests
- `ResponseFormatterAgent`: Formats responses

**Use Cases:**
- Data APIs
- Calculation services
- Integration endpoints

**Example:**
```bash
greenlang create-app emissions-api \
  --template api-service \
  --cache redis \
  --database postgresql
```

### Options

| Option | Values | Description |
|--------|--------|-------------|
| `--template` | See templates above | Application template |
| `--llm` | `openai`, `anthropic`, `all` | LLM provider |
| `--cache` | `memory`, `redis`, `both` | Cache type |
| `--database` | `postgresql`, `mongodb`, `both` | Database type |
| `--no-tests` | flag | Skip test generation |
| `--no-cicd` | flag | Skip CI/CD generation |
| `--no-monitoring` | flag | Skip monitoring setup |
| `--output` | path | Output directory |

### Interactive Mode

For beginners or when you want guided setup:

```bash
greenlang create-app
```

You'll be prompted for:
1. Application name
2. Template selection
3. LLM integration (yes/no + provider)
4. Caching (yes/no + type)
5. Database (yes/no + type)
6. Test generation (yes/no)
7. CI/CD generation (yes/no)
8. Monitoring (yes/no)

### Generated Structure

```
my-app/
├── README.md                 # Complete documentation
├── requirements.txt          # Python dependencies
├── config.yaml              # Application configuration
├── .env.example             # Environment template
├── .gitignore               # Git ignore rules
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Multi-container setup
├── app/
│   ├── __init__.py
│   ├── main.py              # Application entry point
│   ├── config.py            # Configuration management
│   ├── agents/              # Agent implementations
│   │   ├── __init__.py
│   │   ├── agent1.py
│   │   ├── agent2.py
│   │   └── agent3.py
│   └── utils/               # Utility functions
│       └── __init__.py
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── conftest.py          # Pytest fixtures
│   ├── test_agent1.py
│   ├── test_agent2.py
│   └── test_agent3.py
├── .github/                 # CI/CD (if enabled)
│   └── workflows/
│       └── ci.yml
└── docs/                    # Documentation
    ├── architecture.md
    └── api.md
```

---

## Component Generator

Add components to existing GreenLang applications.

### Add Agent

```bash
greenlang add agent <name> [options]
```

**Templates:**

1. **Basic** (`basic`)
   - Standard agent with validation
   - Input/output validation
   - Error handling

2. **Calculator** (`calculator`)
   - Calculation-focused agent
   - Caching support
   - Batch processing

3. **LLM Analyzer** (`llm-analyzer`)
   - LLM integration
   - Caching for API calls
   - Prompt management

4. **Validator** (`validator`)
   - Data validation focus
   - Schema validation
   - Error reporting

**Example:**
```bash
greenlang add agent EmissionsCalculator --template calculator
```

**Generated Files:**
- `app/agents/emissionscalculator.py`: Agent implementation
- `tests/test_emissionscalculator.py`: Unit tests

### Add LLM Integration

```bash
greenlang add llm --provider <provider> [--caching]
```

**Providers:**
- `openai`: OpenAI (GPT models)
- `anthropic`: Anthropic (Claude models)

**What Gets Added:**
- LLM dependencies in `requirements.txt`
- Environment variables in `.env.example`
- Configuration in `config.py`

**Example:**
```bash
greenlang add llm --provider openai --caching
```

### Add Caching

```bash
greenlang add cache --type <type>
```

**Cache Types:**
- `memory`: In-memory caching (development)
- `redis`: Redis caching (production)

**What Gets Added:**
- Cache dependencies
- Redis service in `docker-compose.yml`
- Configuration variables
- Cache utilities

**Example:**
```bash
greenlang add cache --type redis
```

### Add Database

```bash
greenlang add database --type <type>
```

**Database Types:**
- `postgresql`: PostgreSQL relational database
- `mongodb`: MongoDB document database

**What Gets Added:**
- Database drivers in `requirements.txt`
- Database service in `docker-compose.yml`
- Connection configuration
- Migration support

**Example:**
```bash
greenlang add database --type postgresql
```

### Add Monitoring

```bash
greenlang add monitoring --dashboard <type>
```

**Dashboard Types:**
- `grafana`: Grafana + Prometheus
- `prometheus`: Prometheus only

**What Gets Added:**
- Prometheus client library
- Prometheus configuration
- Grafana setup (if selected)
- Metrics endpoints
- Dashboard definitions

**Example:**
```bash
greenlang add monitoring --dashboard grafana
```

---

## Configuration Generator

Generate configuration files for different environments.

### Command

```bash
greenlang generate-config [options]
```

### Modes

#### Interactive Mode

```bash
greenlang generate-config --interactive
```

Prompts for:
- Application name
- Features (LLM, cache, database, monitoring)
- Environments (dev, staging, prod)

#### Command-Line Mode

```bash
# Generate specific environment
greenlang generate-config \
  --app-name my-app \
  --environment production \
  --output .env.production

# Generate all environments
greenlang generate-config \
  --app-name my-app \
  --all-environments
```

### Generated Files

1. **config.yaml**: Main configuration with all environments
2. **.env.development**: Development environment
3. **.env.staging**: Staging environment
4. **.env.production**: Production environment
5. **.env.example**: Template for new developers
6. **secrets.yaml.template**: Kubernetes secrets template

### Configuration Structure

```yaml
app:
  name: my-app
  version: 1.0.0
  description: Application description

environments:
  development:
    debug: true
    log_level: DEBUG
    llm:
      provider: openai
      model: gpt-4
      temperature: 0.7
    cache:
      type: memory
      ttl: 3600
    database:
      pool_size: 5
      echo: true

  production:
    debug: false
    log_level: WARNING
    llm:
      provider: openai
      model: gpt-4
      temperature: 0.5
      retry_attempts: 3
    cache:
      type: redis
      ttl: 7200
    database:
      pool_size: 20
      echo: false
    security:
      enable_cors: true
      enable_csrf: true
    rate_limiting:
      enabled: true
      requests_per_minute: 1000
```

### Validation

```bash
greenlang generate-config --validate config.yaml
```

Checks for:
- Required fields
- Security settings in production
- Valid value ranges
- Placeholder values

---

## Test Generator

Auto-generate comprehensive tests for agents.

### Command

```bash
greenlang generate-tests <agent_file> [options]
```

### Features

- **Code Analysis**: Analyzes agent to extract methods and features
- **Unit Tests**: Generates comprehensive unit tests
- **Integration Tests**: Generates end-to-end tests
- **Fixtures**: Creates reusable test fixtures
- **Mocking**: Sets up mocks for dependencies

### Generated Tests

1. **Initialization Test**: Verifies agent creates correctly
2. **Execute Success**: Tests successful execution
3. **Invalid Input**: Tests error handling
4. **Validation Tests**: Tests input/output validation
5. **Batch Processing**: Tests batch execution
6. **Error Handling**: Tests exception handling
7. **Integration Tests**: Tests real-world scenarios
8. **Performance Tests**: Tests with large datasets
9. **Concurrent Tests**: Tests thread safety

### Example

```bash
# Generate tests for an agent
greenlang generate-tests app/agents/calculator.py

# With custom output
greenlang generate-tests app/agents/calculator.py \
  --output tests/test_calculator.py

# Also generate conftest.py
greenlang generate-tests app/agents/calculator.py --conftest
```

### Generated Test Structure

```python
import pytest
from app.agents.calculator import Calculator

class TestCalculator:
    """Unit tests."""

    @pytest.fixture
    def agent(self):
        return Calculator()

    def test_execute_success(self, agent):
        """Test successful execution."""
        # Test implementation

    def test_invalid_input(self, agent):
        """Test error handling."""
        # Test implementation

    # More tests...

class TestCalculatorIntegration:
    """Integration tests."""

    def test_end_to_end(self):
        """Test complete workflow."""
        # Test implementation
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_calculator.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run only unit tests
pytest tests/ -v -m "not integration"

# Run only integration tests
pytest tests/ -v -m integration
```

---

## CI/CD Generator

Generate CI/CD pipelines for multiple platforms.

### Command

```bash
greenlang generate-cicd --platform <platform> [options]
```

### Platforms

#### GitHub Actions

```bash
greenlang generate-cicd --platform github --with-deploy
```

**Generated:** `.github/workflows/ci.yml`

**Includes:**
- Code quality checks (Black, isort, mypy, pylint)
- Tests on multiple Python versions (3.10, 3.11, 3.12)
- Security scanning (Bandit, Safety)
- Docker image building
- Code coverage reporting
- Deployment (optional)

**Example Workflow:**
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Black
        run: black --check app/ tests/
      # More steps...

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11, 3.12]
    steps:
      - name: Run tests
        run: pytest tests/ -v --cov=app
      # More steps...

  security:
    runs-on: ubuntu-latest
    steps:
      - name: Run Bandit
        run: bandit -r app/
      # More steps...

  build:
    needs: [lint, test, security]
    steps:
      - name: Build Docker image
        run: docker build -t myapp:latest .
      # More steps...
```

#### GitLab CI

```bash
greenlang generate-cicd --platform gitlab
```

**Generated:** `.gitlab-ci.yml`

**Stages:**
1. Lint: Code quality checks
2. Test: Unit and integration tests
3. Security: Security scanning
4. Build: Docker image build
5. Deploy: Deployment (manual trigger)

#### Jenkins

```bash
greenlang generate-cicd --platform jenkins
```

**Generated:** `Jenkinsfile`

**Features:**
- Pipeline as code
- Parallel execution
- Email notifications
- Artifact archiving

### Generate All Platforms

```bash
greenlang generate-cicd --all --app-name my-app
```

Generates configurations for all supported platforms.

---

## Deployment Generator

Generate deployment configurations for cloud platforms.

### Command

```bash
greenlang generate-deployment --platform <platform> [options]
```

### Platforms

#### Kubernetes

```bash
greenlang generate-deployment --platform kubernetes
```

**Generated:** `k8s/` directory with:

1. **deployment.yaml**: Application deployment
   - 3 replicas (default)
   - Resource limits
   - Health checks
   - Rolling updates

2. **service.yaml**: LoadBalancer service
   - External access
   - Port configuration

3. **configmap.yaml**: Configuration management
   - Environment-specific settings
   - Feature flags

4. **secret.yaml.template**: Secrets template
   - API keys
   - Database credentials
   - Encryption keys

5. **hpa.yaml**: Horizontal Pod Autoscaler
   - CPU-based scaling (70%)
   - Memory-based scaling (80%)
   - Min: 2, Max: 10 replicas

**Deployment:**
```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment
kubectl get deployments
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/my-app

# Scale manually
kubectl scale deployment my-app --replicas=5
```

#### Terraform

```bash
greenlang generate-deployment --platform terraform
```

**Generated:** `terraform/` directory with:

1. **main.tf**: Main Terraform configuration
   - Provider setup
   - Backend configuration

2. **variables.tf**: Input variables
   - AWS region
   - Instance types
   - Environment settings

3. **resources.tf**: AWS resources
   - ECS cluster
   - ECR repository
   - VPC and networking
   - Security groups

4. **outputs.tf**: Output values
   - Cluster name
   - Repository URL
   - VPC ID

**Usage:**
```bash
cd terraform/

# Initialize
terraform init

# Plan
terraform plan

# Apply
terraform apply

# Destroy
terraform destroy
```

#### Docker Compose (Production)

```bash
greenlang generate-deployment --platform docker
```

**Generated:** `docker-compose.prod.yml`

**Services:**
- Application (3 replicas)
- Redis (caching)
- PostgreSQL (database)
- Prometheus (metrics)
- Grafana (dashboards)

**Features:**
- Health checks
- Resource limits
- Restart policies
- Volume persistence
- Network isolation

**Usage:**
```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Scale application
docker-compose -f docker-compose.prod.yml up -d --scale app=5

# Stop all services
docker-compose -f docker-compose.prod.yml down
```

#### Helm Chart

```bash
greenlang generate-deployment --platform helm
```

**Generated:** `helm/<app-name>/` directory with:

1. **Chart.yaml**: Chart metadata
2. **values.yaml**: Default values
3. **templates/**: Kubernetes templates

**Usage:**
```bash
# Install chart
helm install my-app ./helm/my-app

# Upgrade
helm upgrade my-app ./helm/my-app

# Uninstall
helm uninstall my-app
```

### Generate All Platforms

```bash
greenlang generate-deployment --all --app-name my-app
```

Generates deployment configs for all platforms.

---

## Best Practices

### 1. Application Development

**Start with Templates:**
- Use templates that match your use case
- Customize generated code as needed
- Follow the established patterns

**Infrastructure-First:**
- Always use GreenLang infrastructure
- No custom implementations
- Leverage built-in features

**Testing:**
- Generate tests early
- Add custom test cases
- Maintain high coverage (>80%)

**Configuration:**
- Use environment-specific configs
- Never commit secrets
- Use secret management tools

### 2. Component Addition

**Incremental Development:**
- Start simple, add features incrementally
- Test after each addition
- Update documentation

**Dependency Management:**
- Keep requirements.txt updated
- Pin major versions
- Regular security updates

### 3. Deployment

**Environment Progression:**
1. Development
2. Staging
3. Production

**Pre-Production Checklist:**
- All tests passing
- Security scan clean
- Performance tested
- Monitoring configured
- Secrets configured
- Backup strategy in place

**Production Configuration:**
- Enable rate limiting
- Configure CORS properly
- Set appropriate timeouts
- Enable metrics collection
- Configure alerting

### 4. Security

**Secrets Management:**
- Use environment variables
- Never commit .env files
- Use secret management (AWS Secrets Manager, etc.)
- Rotate secrets regularly

**API Security:**
- Enable authentication
- Use HTTPS only
- Implement rate limiting
- Validate all inputs
- Sanitize outputs

**Dependencies:**
- Regular security scans
- Keep dependencies updated
- Monitor for vulnerabilities
- Use dependabot

### 5. Performance

**Caching Strategy:**
- Use Redis for production
- Cache expensive operations
- Set appropriate TTLs
- Monitor cache hit rates

**Database Optimization:**
- Use connection pooling
- Optimize queries
- Add indexes
- Monitor slow queries

**Scaling:**
- Use horizontal pod autoscaling
- Set appropriate resource limits
- Monitor resource usage
- Load test regularly

---

## Examples

### Example 1: ESG Data Collection Platform

**Requirements:**
- Collect ESG data from suppliers
- Validate data quality
- Calculate sustainability metrics
- Generate compliance reports

**Implementation:**

```bash
# 1. Create application
greenlang create-app esg-platform \
  --template data-intake \
  --database postgresql \
  --cache redis

# 2. Add LLM for report analysis
cd esg-platform
greenlang add llm --provider openai --caching

# 3. Add custom calculator agent
greenlang add agent ESGScoreCalculator --template calculator

# 4. Generate tests
greenlang generate-tests app/agents/esgscorec calculator.py

# 5. Setup CI/CD
greenlang generate-cicd --platform github --with-deploy

# 6. Setup deployment
greenlang generate-deployment --platform kubernetes

# 7. Generate production configs
greenlang generate-config --environment production
```

### Example 2: Carbon Emissions API

**Requirements:**
- REST API for emissions calculations
- High performance with caching
- Multi-region deployment
- Comprehensive monitoring

**Implementation:**

```bash
# 1. Create API service
greenlang create-app emissions-api \
  --template api-service \
  --cache redis \
  --database postgresql

# 2. Add monitoring
cd emissions-api
greenlang add monitoring --dashboard grafana

# 3. Generate deployment configs
greenlang generate-deployment --all

# 4. Setup CI/CD for all platforms
greenlang generate-cicd --all

# 5. Generate environment configs
greenlang generate-config --all-environments
```

### Example 3: Sustainability Report Analyzer

**Requirements:**
- Analyze PDF sustainability reports
- Extract key metrics using AI
- Generate insights and summaries
- Store results in database

**Implementation:**

```bash
# 1. Create LLM analysis app
greenlang create-app report-analyzer \
  --template llm-analysis \
  --llm anthropic \
  --database mongodb \
  --cache redis

# 2. Add PDF processing agent
cd report-analyzer
greenlang add agent PDFExtractor --template basic

# 3. Generate comprehensive tests
greenlang generate-tests app/agents/pdfextractor.py --conftest
greenlang generate-tests app/agents/llmanalyzeragent.py

# 4. Setup deployment
greenlang generate-deployment --platform kubernetes
greenlang generate-cicd --platform github

# 5. Configure for production
greenlang generate-config --environment production
```

### Example 4: Real-time Emissions Dashboard

**Requirements:**
- Real-time data processing pipeline
- Live dashboard
- Historical data storage
- Alerting for anomalies

**Implementation:**

```bash
# 1. Create pipeline application
greenlang create-app emissions-dashboard \
  --template pipeline \
  --database postgresql \
  --cache redis

# 2. Add monitoring with Grafana
cd emissions-dashboard
greenlang add monitoring --dashboard grafana

# 3. Add alert processor
greenlang add agent AlertProcessor --template validator

# 4. Setup infrastructure
greenlang generate-deployment --platform terraform
greenlang generate-cicd --platform gitlab

# 5. Configure environments
greenlang generate-config --all-environments
```

---

## Advanced Usage

### Custom Templates

You can customize generated code:

1. Generate base application
2. Modify generated files
3. Keep infrastructure usage
4. Add business logic

### Extending Agents

```python
# Generated agent
class MyAgent(BaseAgent):
    def execute(self, input_data):
        # Add your custom logic here
        result = self._custom_processing(input_data)
        return {"status": "success", "result": result}

    def _custom_processing(self, data):
        # Your business logic
        pass
```

### Multi-Environment Deployment

```bash
# Development
greenlang generate-config --environment development
docker-compose up

# Staging
greenlang generate-config --environment staging
kubectl apply -f k8s/ --namespace staging

# Production
greenlang generate-config --environment production
kubectl apply -f k8s/ --namespace production
```

---

## Troubleshooting

### Common Issues

**Issue: Import errors after generation**
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Issue: Tests failing**
```bash
# Solution: Check test data and assertions
pytest tests/ -v --tb=long
```

**Issue: Docker build fails**
```bash
# Solution: Check Dockerfile and dependencies
docker build -t myapp:test .
```

**Issue: Kubernetes deployment fails**
```bash
# Solution: Check pod logs
kubectl get pods
kubectl logs <pod-name>
kubectl describe pod <pod-name>
```

### Getting Help

1. Check this guide
2. Review generated README.md
3. Check GreenLang documentation
4. Use `greenlang chat` for AI assistance
5. Open GitHub issue

---

## Summary

GreenLang generators provide:

- **Fast Development**: Create apps in minutes
- **Best Practices**: Production-ready code
- **Infrastructure-First**: Only use GreenLang infrastructure
- **Complete Stack**: From code to deployment
- **Flexibility**: Easy to customize
- **Quality**: Comprehensive tests and monitoring

### Next Steps

1. Try the Quick Start examples
2. Create your first application
3. Explore different templates
4. Add components incrementally
5. Deploy to your platform

Happy generating!
