# GreenLang Execution Model v1.0

**Version:** 1.0.0
**Status:** Draft
**Last Updated:** 2026-02-03

## Overview

The GreenLang Execution Model defines how pipelines execute: from input data through agent steps to output artifacts. This specification ensures predictable, auditable execution for climate compliance workflows.

**Core Contract:**
> inputs -> steps/tools -> artifacts (with full provenance)

## Execution Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Input     │───>│ Validation   │───>│   Steps     │───>│   Artifacts  │
│   Data      │    │ & Binding    │    │  Execution  │    │   Output     │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                          │                   │                   │
                          v                   v                   v
                   ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
                   │   Policy     │    │ Checkpoint  │    │  Provenance  │
                   │   Check      │    │  Manager    │    │   Record     │
                   └──────────────┘    └─────────────┘    └──────────────┘
```

## Pipeline Definition (gl.yaml)

### Schema

```yaml
# gl.yaml - Pipeline Definition

# === METADATA ===
api_version: glip/v1                  # API version
kind: Pipeline                        # resource kind
metadata:
  name: pipeline-name                 # identifier
  description: Pipeline description
  version: 1.0.0                      # pipeline version
  labels:                             # arbitrary labels
    team: carbon-team
    category: emissions

# === PARAMETERS ===
parameters:
  - name: param_name
    type: string|number|boolean|array|object
    required: true|false
    default: default_value
    description: Parameter description
    validation:                       # optional validation
      min: 0
      max: 100
      pattern: "^[a-z]+$"

# === STEPS ===
steps:
  - name: step-name                   # unique within pipeline
    agent: agent-id                   # agent to invoke
    description: What this step does

    # Input mapping
    inputs:
      field: ${input.field}           # from pipeline input
      other: ${steps.prev.output}     # from previous step

    # Output declaration
    outputs:
      - result_field
      - another_field

    # Conditional execution
    condition: results.prev.success == true

    # Error handling
    retry:
      attempts: 3
      backoff: exponential
      delay_seconds: 5

    on_failure: stop|skip|continue

    # Timeout
    timeout_seconds: 300

    # Resource requirements
    resources:
      cpu: 1
      memory: 1Gi

# === OUTPUT MAPPING ===
outputs:
  final_result: ${steps.last.result}
  summary: ${steps.aggregator.summary}

# === POLICY ATTACHMENTS ===
policy:
  - policy-bundle-name
  - inline:
      name: custom-policy
      rules:
        - rule: max_cost
          limit: 100
```

## Execution Phases

### Phase 1: Initialization

1. **Load Pipeline Definition**
   - Parse gl.yaml
   - Validate against schema
   - Resolve template imports

2. **Parameter Binding**
   - Bind input parameters to declared types
   - Apply default values
   - Validate constraints

3. **Policy Pre-Check**
   - Evaluate policy bundles
   - Check cost budgets
   - Verify data residency compliance

4. **Create Run Context**
   ```python
   context = {
       "run_id": "pipeline_abc_123",
       "input": validated_input,
       "results": {},
       "errors": [],
       "metadata": {
           "started_at": "2026-02-03T12:00:00Z",
           "executor": "local"
       }
   }
   ```

### Phase 2: Step Execution

For each step in topological order:

1. **Condition Evaluation**
   ```python
   if step.condition:
       should_execute = evaluate_condition(step.condition, context)
       if not should_execute:
           log("Skipping step: condition not met")
           continue
   ```

2. **Input Preparation**
   ```python
   step_input = resolve_input_mapping(step.inputs, context)
   step_input["_checkpoint_contract"] = {
       "run_id": context["run_id"],
       "step_id": step.name,
       "idempotency_key": compute_idempotency_key(),
       "attempt": 1
   }
   ```

3. **Agent Invocation**
   ```python
   agent = registry.get(step.agent)
   result = agent.run(step_input)
   ```

4. **Result Handling**
   ```python
   if result.success:
       context["results"][step.name] = result.data
       save_checkpoint(step.name, "completed", result.data)
   else:
       handle_failure(step, result, context)
   ```

5. **Retry Logic**
   ```python
   for attempt in range(1, step.retry.attempts + 1):
       try:
           result = agent.run(step_input)
           if result.success:
               break
       except Exception as e:
           if attempt < step.retry.attempts:
               delay = compute_backoff(step.retry, attempt)
               sleep(delay)
           else:
               raise
   ```

### Phase 3: Artifact Generation

1. **Output Mapping**
   ```python
   outputs = {}
   for key, path in pipeline.outputs.items():
       outputs[key] = resolve_path(path, context)
   ```

2. **run.json Generation**
   ```python
   run_artifact = {
       "run_id": context["run_id"],
       "pipeline": pipeline.metadata.name,
       "version": pipeline.metadata.version,
       "success": len(context["errors"]) == 0,
       "started_at": context["metadata"]["started_at"],
       "completed_at": now(),
       "inputs": sanitized_inputs,
       "outputs": outputs,
       "steps": step_records,
       "provenance": provenance_record
   }
   ```

3. **Artifact Storage**
   - Local: Write to output directory
   - S3: Upload with metadata

### Phase 4: Completion

1. **Policy Post-Check**
   - Verify output compliance
   - Check data egress rules

2. **Cleanup**
   - Release resources
   - Archive temporary files

3. **Emit Completion Event**
   - Send webhook notifications
   - Update run status

## Step Execution Contract

### Agent Input Contract

Agents receive a dictionary with:

```python
{
    # User-provided inputs (from input mapping)
    "field1": value1,
    "field2": value2,

    # System-provided context
    "_checkpoint_contract": {
        "run_id": "run_abc_123",
        "step_id": "step_name",
        "idempotency_key": "hash_xyz",
        "attempt": 1,
        "is_retry": False,
        "checkpoint_enabled": True
    }
}
```

### Agent Output Contract

Agents MUST return either:

**Option 1: Dictionary**
```python
{
    "success": True,
    "data": {
        "result_field": value,
        "another_field": value
    },
    "error": None,
    "metadata": {
        "execution_time_ms": 150,
        "records_processed": 1000
    }
}
```

**Option 2: AgentResult Object**
```python
AgentResult(
    success=True,
    data={"result_field": value},
    error=None,
    metadata={"execution_time_ms": 150},
    provenance=ProvenanceRecord(...)
)
```

### Context Propagation

Context flows through the pipeline:

```yaml
steps:
  - name: step1
    outputs: [result_a]

  - name: step2
    inputs:
      from_step1: ${steps.step1.result_a}  # Reference previous
      from_input: ${input.original}         # Reference input
```

## Error Handling

### Failure Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| `stop` | Halt pipeline immediately | Critical steps |
| `skip` | Log error, continue to next step | Optional steps |
| `continue` | Log error, try dependent steps | Partial failures OK |

### Error Structure

```python
{
    "step": "step_name",
    "error": "Error message",
    "error_code": "AGENT_TIMEOUT",
    "attempts": 3,
    "timestamp": "2026-02-03T12:05:00Z",
    "context": {
        "input_summary": "...",
        "stack_trace": "..."
    }
}
```

### Retry Configuration

```yaml
retry:
  attempts: 3              # Total attempts (1 = no retry)
  backoff: exponential     # none|linear|exponential
  delay_seconds: 5         # Initial delay
  max_delay_seconds: 60    # Maximum delay
  retry_on:                # Retry only on these errors
    - TIMEOUT
    - RATE_LIMIT
    - TRANSIENT
```

### Backoff Calculation

| Strategy | Delay Formula |
|----------|--------------|
| none | delay_seconds |
| linear | delay_seconds * attempt |
| exponential | delay_seconds * (2 ^ (attempt - 1)) |

## Execution Backends

### Local Executor

```python
class LocalExecutor:
    """Execute steps in the current process."""

    def execute(self, step, input_data):
        agent = self.load_agent(step.agent)
        return agent.run(input_data)
```

### Kubernetes Executor

```python
class K8sExecutor:
    """Execute steps as Kubernetes Jobs."""

    def execute(self, step, input_data):
        job = self.create_job(
            name=f"gl-{step.name}-{run_id}",
            image=self.get_image(step),
            resources=step.resources,
            env=self.prepare_env(input_data)
        )
        return self.wait_for_completion(job)
```

### Resource Requirements

```yaml
resources:
  cpu: 1                   # CPU cores
  memory: 1Gi              # Memory limit
  gpu: 0                   # GPU count
  ephemeral_storage: 10Gi  # Temp storage
```

## Artifacts

### run.json Structure

```json
{
  "schema_version": "1.0",
  "run_id": "pipeline_abc_123",
  "pipeline": {
    "name": "emissions-calculation",
    "version": "1.0.0",
    "hash": "sha256:abc123..."
  },
  "status": "completed",
  "success": true,
  "started_at": "2026-02-03T12:00:00Z",
  "completed_at": "2026-02-03T12:05:00Z",
  "duration_ms": 300000,
  "executor": "local",
  "inputs": {
    "param1": "value1"
  },
  "outputs": {
    "total_emissions": 1234.56,
    "report_path": "outputs/report.pdf"
  },
  "steps": [
    {
      "name": "step1",
      "agent": "agent-id",
      "status": "completed",
      "started_at": "2026-02-03T12:00:00Z",
      "completed_at": "2026-02-03T12:01:00Z",
      "duration_ms": 60000,
      "attempts": 1,
      "outputs": {}
    }
  ],
  "errors": [],
  "provenance": {
    "hash": "sha256:def456...",
    "factors_used": ["DEFRA-2024", "EPA-2024"],
    "model_versions": []
  },
  "checkpoints": {
    "enabled": true,
    "store": "local",
    "path": ".checkpoints/run_abc_123"
  }
}
```

### Artifact Storage

**Local Storage:**
```
outputs/
├── run.json              # Run metadata
├── report.pdf            # Generated report
├── data.json             # Output data
└── .provenance/          # Provenance records
    └── run_abc_123.json
```

**S3 Storage:**
```
s3://bucket/runs/
├── run_abc_123/
│   ├── run.json
│   ├── artifacts/
│   └── provenance/
```

### Retention Policy

| Artifact Type | Default Retention | Configurable |
|---------------|-------------------|--------------|
| run.json | 1 year | Yes |
| Output data | 90 days | Yes |
| Checkpoints | 7 days | Yes |
| Provenance | 7 years | No (audit requirement) |

## Checkpoint System

### Checkpoint State

```python
@dataclass
class CheckpointState:
    run_id: str
    step_id: str
    status: CheckpointStatus  # PENDING|RUNNING|COMPLETED|FAILED
    outputs: Dict[str, Any]
    idempotency_key: str
    attempt: int
    error_message: Optional[str]
    timestamp: datetime
```

### Resume from Checkpoint

```python
# Resume a failed run
checkpoint = orchestrator.get_run_checkpoint("run_abc_123")
result = orchestrator.execute_workflow(
    "pipeline-name",
    input_data,
    resume_from_checkpoint=True,
    run_checkpoint=checkpoint
)
```

### Idempotency Key

```python
def generate_idempotency_key(plan_hash, step_id, attempt):
    """Generate unique key for step execution."""
    data = f"{plan_hash}:{step_id}:{attempt}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]
```

## Expression Language

### Supported in Conditions

| Expression | Example | Description |
|------------|---------|-------------|
| Comparison | `a == b` | Equality, inequality |
| Boolean | `a and b` | and, or, not |
| Membership | `x in [1,2,3]` | List/dict membership |
| Attribute | `obj.field` | Object attribute access |
| Subscript | `data["key"]` | Dictionary/list access |
| Literals | `[1,2,3]`, `{"a":1}` | List, dict, tuple literals |

### NOT Supported (Security)

- Function calls (except allowlisted)
- Lambda expressions
- Import statements
- eval/exec

### Examples

```yaml
# Simple comparison
condition: input.amount > 0

# Boolean logic
condition: results.step1.success and input.validate == true

# Membership check
condition: input.region in ["US", "EU", "APAC"]

# Nested access
condition: results.validation.data.is_valid == true
```

## Policy Enforcement

### Evaluation Points

1. **Pre-execution**: Before pipeline starts
2. **Per-step**: Before each step executes
3. **Post-execution**: After pipeline completes
4. **On-error**: When errors occur

### Policy Decision

```python
class PolicyDecision(Enum):
    APPROVE = "approve"       # Allow execution
    REJECT = "reject"         # Block execution
    REQUIRE_APPROVAL = "require_approval"  # Need human approval
```

### Example Policy

```rego
# policies/cost_budget.rego
package greenlang.policy

default allow = false

allow {
    input.estimated_cost <= data.budget.max_cost
}

deny[msg] {
    input.estimated_cost > data.budget.max_cost
    msg := sprintf("Estimated cost %v exceeds budget %v",
                   [input.estimated_cost, data.budget.max_cost])
}
```

## Appendix A: GLIP Protocol

GLIP (GreenLang Interoperability Protocol) defines the wire format for orchestrator communication.

### Version

Current: GLIP/v1

### Message Format

```json
{
  "glip_version": "v1",
  "message_type": "step_request|step_response|event",
  "correlation_id": "uuid",
  "timestamp": "ISO8601",
  "payload": {}
}
```

## Appendix B: Executor Interface

```python
class ExecutorBackend(ABC):
    @abstractmethod
    def execute(
        self,
        step: StepDefinition,
        input_data: Dict[str, Any],
        context: ExecutionContext
    ) -> StepResult:
        """Execute a pipeline step."""
        pass

    @abstractmethod
    def cancel(self, execution_id: str) -> bool:
        """Cancel a running execution."""
        pass

    @abstractmethod
    def get_status(self, execution_id: str) -> ExecutionStatus:
        """Get execution status."""
        pass
```
