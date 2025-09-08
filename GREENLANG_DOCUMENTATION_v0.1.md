# GreenLang v0.1 Platform Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Pack System](#pack-system)
3. [Core SDK](#core-sdk)
4. [Unified CLI](#unified-cli)
5. [Policy Enforcement](#policy-enforcement)
6. [Provenance & Security](#provenance--security)
7. [Runtime Profiles](#runtime-profiles)
8. [Migration Guide](#migration-guide)
9. [Pack Development](#pack-development)
10. [API Reference](#api-reference)

---

## Architecture Overview

### v0.1 Philosophy
```
GreenLang = Infrastructure (not domain logic)
Domain Logic = Lives in Packs
Platform = SDK + CLI + Runtime + Hub + Policy + Provenance
Success = Developer Love + Trust + Distribution
```

### Component Architecture
```
┌──────────────────────────────────────────────────────┐
│                    User Interface                     │
│                  (CLI: gl command)                    │
├──────────────────────────────────────────────────────┤
│                    Pack Registry                      │
│            (Discovery & Distribution)                 │
├──────────────────────────────────────────────────────┤
│     SDK Core          │        Runtime Engine        │
│  (Base Abstractions)  │    (Executor + Profiles)     │
├──────────────────────────────────────────────────────┤
│   Policy Engine       │      Provenance System       │
│   (OPA Policies)      │    (SBOM + Signatures)       │
├──────────────────────────────────────────────────────┤
│                       Packs                          │
│   ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│   │emissions-   │  │building-    │  │hvac-       │ │
│   │core         │  │analysis     │  │measures    │ │
│   └─────────────┘  └─────────────┘  └────────────┘ │
└──────────────────────────────────────────────────────┘
```

---

## Pack System

### What is a Pack?
A pack is a self-contained module that provides domain-specific functionality. Packs contain:
- **Agents**: Domain-specific processing units
- **Pipelines**: Workflow definitions
- **Datasets**: Data with documentation cards
- **Policies**: Security and compliance rules
- **Tests**: Quality assurance

### Pack Manifest (pack.yaml)
```yaml
name: emissions-core
version: 1.0.0
type: domain  # domain|connector|report|policy|dataset
description: Core emissions calculation functionality

authors:
  - name: GreenLang Team
    email: team@greenlang.io

dependencies:
  - name: greenlang-sdk
    version: ">=0.1.0"
  - name: data-connectors
    version: ">=0.5.0"

requirements:  # Python packages
  - pandas>=1.3.0
  - numpy>=1.20.0

exports:
  agents:
    - name: FuelEmissions
      class_path: agents.fuel:FuelAgent
      description: Calculate fuel-based emissions
      inputs:
        fuel_type: string
        amount: number
        unit: string
      outputs:
        co2e_kg: number
        methodology: string
  
  pipelines:
    - name: building-analysis
      file: pipelines/building.yaml
      description: Complete building emissions analysis
  
  datasets:
    - name: emission-factors
      path: data/emission_factors.json
      format: json
      card: cards/emission_factors.md
      size: 45KB

policy:
  install: policies/install.rego
  runtime: policies/runtime.rego

provenance:
  sbom: true
  signing: true
  
test_command: pytest tests/
min_greenlang_version: 0.1.0
```

### Pack Registry
```python
from greenlang import PackRegistry

registry = PackRegistry()

# List installed packs
packs = registry.list()

# Get specific pack
pack = registry.get("emissions-core")

# Register new pack
registry.register(Path("./my-pack"))

# Verify pack integrity
registry.verify("emissions-core")
```

### Pack Loader
```python
from greenlang import PackLoader

loader = PackLoader()

# Load pack and its components
pack = loader.load("emissions-core")

# Access agents
FuelAgent = loader.get_agent("emissions-core.FuelEmissions")

# Access pipelines
pipeline = loader.get_pipeline("emissions-core.building-analysis")

# Access datasets
dataset = loader.get_dataset("emissions-core.emission-factors")
```

---

## Core SDK

### Base Abstractions

#### Agent
```python
from greenlang import Agent, Result

class CustomAgent(Agent):
    """Domain-agnostic agent base class"""
    
    def validate(self, input_data):
        """Validate input data"""
        # Validation logic
        return True
    
    def process(self, input_data):
        """Process input and produce output"""
        # Processing logic
        return output_data
    
    def get_input_schema(self):
        """JSON schema for input validation"""
        return {
            "type": "object",
            "properties": {
                "value": {"type": "number"}
            },
            "required": ["value"]
        }

# Usage
agent = CustomAgent()
result = agent.run({"value": 42})
if result.success:
    print(result.data)
```

#### Pipeline
```python
from greenlang import Pipeline

class CustomPipeline(Pipeline):
    """Orchestrate multiple agents"""
    
    def execute(self, input_data):
        """Execute pipeline steps"""
        for agent in self.agents:
            result = agent.run(input_data)
            if not result.success:
                return result
            input_data = result.data
        return Result(success=True, data=input_data)

# Usage
pipeline = CustomPipeline()
pipeline.add_agent(Agent1())
pipeline.add_agent(Agent2())
result = pipeline.execute(input_data)
```

#### Connector
```python
from greenlang import Connector

class APIConnector(Connector):
    """Connect to external systems"""
    
    def connect(self):
        """Establish connection"""
        # Connection logic
        self.connected = True
        return True
    
    def disconnect(self):
        """Close connection"""
        self.connected = False
        return True
    
    def read(self, query):
        """Read data from source"""
        # Read logic
        return Result(success=True, data=data)
    
    def write(self, data):
        """Write data to destination"""
        # Write logic
        return Result(success=True)

# Usage with context manager
with APIConnector(config) as conn:
    data = conn.read({"endpoint": "/data"})
```

#### Dataset
```python
from greenlang import Dataset

class EmissionsDataset(Dataset):
    """Dataset with metadata and provenance"""
    
    def load(self):
        """Load dataset"""
        return pd.read_csv(self.path)
    
    def save(self, data):
        """Save dataset"""
        data.to_csv(self.path)
        return True
    
    def describe(self):
        """Dataset description"""
        return {
            "rows": 1000,
            "columns": 10,
            "size": "1.2MB"
        }
    
    def get_card(self):
        """Dataset documentation card"""
        return """
        # Emissions Dataset
        
        Global emission factors for 50+ countries.
        
        ## Schema
        - country: ISO code
        - factor: kg CO2e/kWh
        - year: Reference year
        """
```

#### Report
```python
from greenlang import Report

class EmissionsReport(Report):
    """Generate formatted reports"""
    
    def generate(self, data, format="markdown"):
        """Generate report content"""
        if format == "markdown":
            return self._generate_markdown(data)
        elif format == "html":
            return self._generate_html(data)
        elif format == "json":
            return json.dumps(data, indent=2)
    
    def save(self, content, path):
        """Save report to file"""
        with open(path, 'w') as f:
            f.write(content)
        return True
```

---

## Unified CLI

### Core Commands

```bash
# Version and help
gl --version
gl --help

# Pack management
gl init --name my-pack --type domain
gl pack list
gl pack info emissions-core
gl pack add emissions-core
gl pack remove emissions-core
gl pack verify emissions-core
gl pack publish ./my-pack

# Running pipelines
gl run emissions-core.calculate --input data.json
gl run pipeline.yaml --output results.json --profile k8s

# Policy management
gl policy check runtime --file policy.rego
gl policy list
gl policy add --file custom-policy.rego

# Provenance
gl verify artifact.json --sig artifact.json.sig

# System health
gl doctor
```

### CLI Configuration
```bash
# Set default registry
export GREENLANG_REGISTRY=hub.greenlang.io

# Set runtime profile
export GREENLANG_PROFILE=k8s

# Set policy directory
export GREENLANG_POLICIES=~/.greenlang/policies
```

---

## Policy Enforcement

### Policy Types

#### Install Policy
```rego
package greenlang.install

# Deny unsigned packs
deny[msg] {
    input.pack.provenance.signing == false
    msg := "Pack must be signed"
}

# Deny large packs
deny[msg] {
    input.pack.size > 100000000  # 100MB
    msg := "Pack exceeds size limit"
}

# Allow verified publishers
allow {
    input.pack.publisher in ["greenlang", "verified"]
}
```

#### Runtime Policy
```rego
package greenlang.runtime

# Resource limits
deny[msg] {
    input.resources.memory > 4096  # 4GB
    msg := "Excessive memory requested"
}

# Rate limiting
deny[msg] {
    input.user.requests_per_minute > 100
    msg := "Rate limit exceeded"
}

# Require authentication
deny[msg] {
    input.user.authenticated == false
    msg := "Authentication required"
}
```

#### Data Policy
```rego
package greenlang.data

# Data residency
deny[msg] {
    input.data.region != input.user.region
    input.data.residency_required == true
    msg := "Data residency violation"
}

# Encryption requirement
deny[msg] {
    input.data.sensitive == true
    input.connection.encrypted == false
    msg := "Encryption required for sensitive data"
}
```

### Policy Integration
```python
from greenlang import PolicyEnforcer

enforcer = PolicyEnforcer()

# Check install policy
if enforcer.check_install(pack_manifest):
    install_pack()

# Check runtime policy
if enforcer.check_runtime(pipeline, input_data):
    execute_pipeline()

# Custom policy check
result = enforcer.check(
    policy_file=Path("custom.rego"),
    input_data={"user": user, "action": action}
)
```

---

## Provenance & Security

### SBOM Generation
```python
from greenlang.provenance import generate_sbom

# Generate SBOM for a pack
sbom = generate_sbom(
    pack_path=Path("./my-pack"),
    output_path=Path("./my-pack/sbom.json")
)

# SBOM contains:
# - All dependencies with versions
# - File hashes
# - Component inventory
# - Vulnerability data (if available)
```

### Artifact Signing
```python
from greenlang.provenance import sign_artifact, verify_artifact

# Sign an artifact
signature = sign_artifact(
    artifact_path=Path("results.json"),
    key_path=Path("~/.greenlang/keys/private.key")
)

# Verify signature
is_valid = verify_artifact(
    artifact_path=Path("results.json"),
    signature_path=Path("results.json.sig")
)
```

### Pack Signing
```python
from greenlang.provenance import sign_pack, verify_pack

# Sign entire pack
signature = sign_pack(
    pack_path=Path("./my-pack"),
    key_path=Path("~/.greenlang/keys/private.key")
)

# Verify pack integrity
is_valid = verify_pack(Path("./my-pack"))
```

---

## Runtime Profiles

### Local Profile
```python
from greenlang import Executor

executor = Executor(profile="local")
result = executor.run("pipeline", input_data)

# Characteristics:
# - Runs on local machine
# - Direct Python execution
# - No resource isolation
# - Fast iteration
```

### Kubernetes Profile
```python
executor = Executor(profile="k8s")
result = executor.run(
    "pipeline",
    input_data,
    resources={
        "memory": "2Gi",
        "cpu": "1000m"
    }
)

# Characteristics:
# - Runs as K8s Jobs/Pods
# - Resource isolation
# - Scalable execution
# - Network policies
```

### Cloud Profile
```python
executor = Executor(profile="cloud")
result = executor.run(
    "pipeline",
    input_data,
    config={
        "provider": "aws",
        "region": "us-west-2",
        "function": "lambda"
    }
)

# Characteristics:
# - Serverless execution
# - Auto-scaling
# - Pay-per-use
# - Geographic distribution
```

### Execution Artifacts
```python
# All executions generate:
# 1. run.json - Reproducible execution record
# 2. Run ledger entry - Immutable history
# 3. Artifacts - Output files

runs = executor.list_runs()
run_details = executor.get_run(run_id)
```

---

## Migration Guide

### From v0.0.1 to v0.1.0

#### Step 1: Update Installation
```bash
# Upgrade GreenLang
pip install greenlang==0.1.0

# Install migration tool
pip install greenlang-migrate
```

#### Step 2: Install Domain Packs
```bash
# Install packs that replace built-in agents
gl pack add emissions-core
gl pack add building-analysis
gl pack add boiler-solar
```

#### Step 3: Update Code

**Old Code (v0.0.1)**
```python
from greenlang import FuelAgent, Orchestrator

agent = FuelAgent()
result = agent.run({
    "fuel_type": "natural_gas",
    "amount": 100,
    "unit": "therms"
})

orchestrator = Orchestrator()
orchestrator.register_agent("fuel", agent)
```

**New Code (v0.1.0)**
```python
from greenlang import PackLoader, Executor

# Load pack
loader = PackLoader()
pack = loader.load("emissions-core")
FuelAgent = pack.agents["FuelEmissions"]

# Or use executor for pipelines
executor = Executor()
result = executor.run(
    "emissions-core.calculate",
    {"fuel_type": "natural_gas", "amount": 100, "unit": "therms"}
)
```

#### Step 4: Migrate Custom Agents

1. Create pack structure:
```bash
gl init --name my-agents --type domain
```

2. Move agents to pack:
```bash
mv my_agent.py my-agents/agents/
```

3. Create pack.yaml:
```yaml
name: my-agents
version: 1.0.0
exports:
  agents:
    - name: MyAgent
      class_path: agents.my_agent:MyAgent
```

4. Install locally:
```bash
gl pack add ./my-agents
```

---

## Pack Development

### Creating a New Pack

#### 1. Initialize Structure
```bash
gl init --name awesome-emissions --type domain
cd awesome-emissions
```

#### 2. Implement Agents
```python
# agents/awesome.py
from greenlang import Agent

class AwesomeAgent(Agent):
    def validate(self, input_data):
        return "value" in input_data
    
    def process(self, input_data):
        result = input_data["value"] * 2
        return {"result": result}
```

#### 3. Define Pipelines
```yaml
# pipelines/process.yaml
name: awesome-process
description: Process data awesomely
steps:
  - agent: AwesomeAgent
    name: double
  - agent: AwesomeAgent
    name: double_again
output:
  final_result: $results.double_again.result
```

#### 4. Add Datasets
```json
// data/factors.json
{
  "factors": {
    "awesome": 42
  }
}
```

#### 5. Write Tests
```python
# tests/test_awesome.py
def test_awesome_agent():
    from agents.awesome import AwesomeAgent
    
    agent = AwesomeAgent()
    result = agent.run({"value": 21})
    assert result.success
    assert result.data["result"] == 42
```

#### 6. Create Documentation
```markdown
# cards/awesome.md

## Awesome Agent

Doubles any input value for maximum awesomeness.

### Inputs
- value: number to double

### Outputs  
- result: doubled value
```

#### 7. Publish Pack
```bash
# Test locally
pytest tests/

# Build and sign
gl pack publish . --sign

# Upload to registry
gl pack upload awesome-emissions --registry hub.greenlang.io
```

---

## API Reference

### Core Classes

#### greenlang.Agent
```python
class Agent(ABC, Generic[TInput, TOutput]):
    metadata: Metadata
    
    @abstractmethod
    def validate(self, input_data: TInput) -> bool
    @abstractmethod
    def process(self, input_data: TInput) -> TOutput
    def run(self, input_data: TInput) -> Result
    def describe(self) -> Dict[str, Any]
```

#### greenlang.Pipeline
```python
class Pipeline(ABC):
    metadata: Metadata
    agents: List[Agent]
    
    def add_agent(self, agent: Agent) -> Pipeline
    @abstractmethod
    def execute(self, input_data: Any) -> Result
    def describe(self) -> Dict[str, Any]
```

#### greenlang.PackRegistry
```python
class PackRegistry:
    def register(self, pack_path: Path) -> InstalledPack
    def unregister(self, pack_name: str) -> None
    def get(self, pack_name: str) -> Optional[InstalledPack]
    def list(self, pack_type: Optional[str] = None) -> List[InstalledPack]
    def verify(self, pack_name: str) -> bool
```

#### greenlang.PackLoader
```python
class PackLoader:
    def load(self, pack_name: str) -> Dict[str, Any]
    def get_agent(self, agent_ref: str) -> Optional[Any]
    def get_pipeline(self, pipeline_ref: str) -> Optional[Any]
    def get_dataset(self, dataset_ref: str) -> Optional[Any]
```

#### greenlang.Executor
```python
class Executor:
    def __init__(self, profile: str = "local")
    def run(self, pipeline_ref: str, input_data: Dict[str, Any]) -> Result
    def list_runs(self) -> List[Dict[str, Any]]
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]
```

#### greenlang.PolicyEnforcer
```python
class PolicyEnforcer:
    def check(self, policy_file: Path, input_data: Dict[str, Any]) -> bool
    def check_install(self, pack_manifest: Dict[str, Any]) -> bool
    def check_runtime(self, pipeline: str, input_data: Dict[str, Any]) -> bool
    def add_policy(self, policy_file: Path) -> None
    def list_policies(self) -> List[str]
```

---

## Summary

GreenLang v0.1 represents a fundamental shift from a monolithic framework to a modular infrastructure platform. The key changes:

1. **Separation of Concerns**: Infrastructure vs Domain Logic
2. **Pack System**: Modular, distributable components
3. **Policy & Provenance**: Trust and security by default
4. **Runtime Flexibility**: Local, K8s, and cloud execution
5. **Developer Experience**: Unified CLI and clear abstractions

The platform now focuses on enabling developers to build, share, and trust climate intelligence applications through a robust infrastructure layer, while all domain-specific knowledge lives in packs.