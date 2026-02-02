# GL-CBAM-APP Migration Guide: v1 to v2 (GreenLang SDK Integration)

**Version:** 1.0
**Date:** 2025-11-09
**Audience:** Developers migrating CBAM agents to GreenLang SDK

---

## Quick Start

### Install GreenLang SDK

```bash
# From project root
cd C:/Users/aksha/Code-V1_GreenLang

# Install GreenLang SDK (local development)
pip install -e core/

# Verify installation
python -c "from greenlang.sdk.base import Agent; print('✓ SDK ready')"
```

### Run v2 Agents

```bash
cd GL-CBAM-APP/CBAM-Importer-Copilot

# Agent 1: Shipment Intake v2
python agents/shipment_intake_agent_v2.py \
    --input examples/demo_shipments.csv \
    --output output/validated_v2.json \
    --cn-codes data/cn_codes.json \
    --rules rules/cbam_rules.yaml \
    --suppliers examples/demo_suppliers.yaml

# Agent 2: Emissions Calculator v2
python agents/emissions_calculator_agent_v2.py \
    --input output/validated_v2.json \
    --output output/emissions_v2.json \
    --suppliers examples/demo_suppliers.yaml \
    --rules rules/cbam_rules.yaml
```

---

## Migration Pattern

### Step 1: Add Framework Imports

```python
# Add to imports
from greenlang.sdk.base import Agent, Metadata, Result
from pydantic import BaseModel
from typing import Any, Dict, List
```

### Step 2: Define Input/Output Types

```python
class MyAgentInput(BaseModel):
    """Typed input for agent"""
    data_path: str
    config: Dict[str, Any]

class MyAgentOutput(BaseModel):
    """Typed output from agent"""
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
```

### Step 3: Inherit from Agent Base Class

```python
# Before (v1)
class MyAgent:
    def __init__(self, config):
        self.config = config

# After (v2)
class MyAgent_v2(Agent[MyAgentInput, MyAgentOutput]):
    def __init__(self, config):
        metadata = Metadata(
            id="my-agent-v2",
            name="My Agent v2",
            version="2.0.0",
            description="My agent using GreenLang SDK"
        )
        super().__init__(metadata)
        self.config = config
```

### Step 4: Implement Framework Interface

```python
def validate(self, input_data: MyAgentInput) -> bool:
    """
    Validate INPUT structure (not business data).

    This checks that the agent received valid input parameters,
    not that the business data itself is valid.
    """
    try:
        if not input_data.data_path:
            logger.error("data_path is required")
            return False
        # Add input structure checks
        return True
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        return False

def process(self, input_data: MyAgentInput) -> MyAgentOutput:
    """
    Process data (business logic only).

    Framework handles:
    - Error wrapping (try/catch)
    - Result container creation
    - Metadata attachment

    You handle:
    - Core business logic
    - Data transformations
    - Business validations
    """
    # Your business logic here
    results = self._do_work(input_data)

    metadata = {
        "processed_at": datetime.now().isoformat(),
        "record_count": len(results)
    }

    return MyAgentOutput(results=results, metadata=metadata)
```

### Step 5: Add Backward-Compatible Wrapper (Optional)

```python
def legacy_method(self, old_param_1, old_param_2):
    """
    v1-compatible API wrapper.

    Allows existing code to use v2 agent without changes.
    """
    # Convert v1 params to v2 input
    input_data = MyAgentInput(
        data_path=old_param_1,
        config=old_param_2
    )

    # Execute using framework
    result = self.run(input_data)

    # Handle errors
    if not result.success:
        raise RuntimeError(f"Processing failed: {result.error}")

    # Convert v2 output to v1 format
    return result.data.results  # Return what v1 expected
```

---

## Complete Example: Shipment Intake Agent

### Before (v1 - 680 lines)

```python
class ShipmentIntakeAgent:
    def __init__(self, cn_codes_path, cbam_rules_path, suppliers_path=None):
        self.cn_codes_path = Path(cn_codes_path)
        # ... load reference data ...
        self.stats = {
            "total_records": 0,
            "valid_records": 0,
            # ... many stat fields ...
        }

    def process(self, input_file, output_file=None):
        # Custom batch processing
        df = self.read_shipments(input_file)
        validated_shipments = []
        all_errors = []

        for idx, row in df.iterrows():
            shipment = row.to_dict()
            is_valid, issues = self.validate_shipment(shipment)
            if is_valid:
                shipment, warnings = self.enrich_shipment(shipment)
            # ... custom error collection ...
            # ... custom statistics ...

        # ... build result dict manually ...
        return result
```

### After (v2 - 531 lines)

```python
from greenlang.sdk.base import Agent, Metadata, Result

class IntakeInput(BaseModel):
    file_path: str

class IntakeOutput(BaseModel):
    shipments: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    validation_errors: List[Dict[str, Any]]

class ShipmentIntakeAgent_v2(Agent[IntakeInput, IntakeOutput]):
    def __init__(self, cn_codes_path, cbam_rules_path, suppliers_path=None):
        # Framework-managed metadata
        metadata = Metadata(
            id="cbam-intake-v2",
            name="CBAM Shipment Intake Agent v2",
            version="2.0.0"
        )
        super().__init__(metadata)

        # Load reference data (same as v1)
        self.cn_codes = self._load_cn_codes()
        # ... rest of setup ...

    def validate(self, input_data: IntakeInput) -> bool:
        """Framework interface: Validate INPUT structure"""
        return Path(input_data.file_path).exists()

    def process(self, input_data: IntakeInput) -> IntakeOutput:
        """Framework interface: Process data"""
        # Same business logic as v1
        df = self.read_shipments(input_data.file_path)
        validated_shipments = []
        all_errors = []

        for idx, row in df.iterrows():
            shipment = row.to_dict()
            is_valid, issues = self.validate_shipment(shipment)
            if is_valid:
                shipment, warnings = self.enrich_shipment(shipment)
            validated_shipments.append(shipment)
            all_errors.extend([i.dict() for i in issues])

        # Framework handles Result wrapping
        return IntakeOutput(
            shipments=validated_shipments,
            metadata={...},
            validation_errors=all_errors
        )

    # v1-compatible wrapper
    def process_file(self, input_file):
        """Backward-compatible with v1 API"""
        input_data = IntakeInput(file_path=str(input_file))
        result = self.run(input_data)  # Framework execution
        if not result.success:
            raise RuntimeError(result.error)
        return {
            "shipments": result.data.shipments,
            "metadata": result.data.metadata,
            "validation_errors": result.data.validation_errors
        }
```

**Key Differences:**
- ✅ Type safety with Pydantic models
- ✅ Framework handles error wrapping
- ✅ Standardized metadata via Metadata class
- ✅ Backward-compatible wrapper for v1 API
- ✅ Same business logic preserved
- ✅ 21.9% LOC reduction (149 lines removed)

---

## Testing Your Migration

### 1. Unit Testing

```python
# test_shipment_intake_agent_v2.py

def test_validate_input():
    """Test framework validate() method"""
    agent = ShipmentIntakeAgent_v2(
        cn_codes_path="data/cn_codes.json",
        cbam_rules_path="rules/cbam_rules.yaml"
    )

    # Valid input
    valid_input = IntakeInput(file_path="examples/demo_shipments.csv")
    assert agent.validate(valid_input) == True

    # Invalid input (missing file)
    invalid_input = IntakeInput(file_path="nonexistent.csv")
    assert agent.validate(invalid_input) == False

def test_process_shipments():
    """Test framework process() method"""
    agent = ShipmentIntakeAgent_v2(...)

    input_data = IntakeInput(file_path="examples/demo_shipments.csv")
    output = agent.process(input_data)

    assert isinstance(output, IntakeOutput)
    assert len(output.shipments) > 0
    assert "total_records" in output.metadata

def test_run_framework_execution():
    """Test framework run() method (full execution)"""
    agent = ShipmentIntakeAgent_v2(...)

    input_data = IntakeInput(file_path="examples/demo_shipments.csv")
    result = agent.run(input_data)

    assert result.success == True
    assert result.data is not None
    assert result.error is None
    assert "agent" in result.metadata
```

### 2. Output Equivalence Testing

```bash
# Compare v1 and v2 outputs
python agents/shipment_intake_agent.py --input test.csv --output v1_output.json
python agents/shipment_intake_agent_v2.py --input test.csv --output v2_output.json

# Should produce equivalent results
diff <(jq -S '.shipments' v1_output.json) <(jq -S '.shipments' v2_output.json)
```

### 3. Performance Benchmarking

```python
# benchmark.py
import time
from agents.shipment_intake_agent import ShipmentIntakeAgent
from agents.shipment_intake_agent_v2 import ShipmentIntakeAgent_v2, IntakeInput

# Setup
agent_v1 = ShipmentIntakeAgent(...)
agent_v2 = ShipmentIntakeAgent_v2(...)

# Benchmark v1
start = time.time()
result_v1 = agent_v1.process("large_dataset.csv")
time_v1 = time.time() - start

# Benchmark v2
start = time.time()
result_v2 = agent_v2.process_file("large_dataset.csv")
time_v2 = time.time() - start

print(f"v1: {time_v1:.2f}s")
print(f"v2: {time_v2:.2f}s")
print(f"Overhead: {((time_v2/time_v1 - 1) * 100):.1f}%")

# Expected: <5% overhead from framework
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Mixing Input Validation with Business Validation

**Wrong:**
```python
def validate(self, input_data: IntakeInput) -> bool:
    # DON'T validate business data here
    df = pd.read_csv(input_data.file_path)
    for row in df.iterrows():
        if not self._is_valid_shipment(row):  # Business validation
            return False
    return True
```

**Correct:**
```python
def validate(self, input_data: IntakeInput) -> bool:
    """Validate INPUT STRUCTURE only"""
    # Check that file exists and is readable
    if not Path(input_data.file_path).exists():
        return False
    return True

def process(self, input_data: IntakeInput) -> IntakeOutput:
    """Business validation happens here"""
    df = pd.read_csv(input_data.file_path)
    for row in df.iterrows():
        is_valid = self._is_valid_shipment(row)  # ✓ Business validation
        # ...
```

### Pitfall 2: Forgetting Backward Compatibility

If existing code calls:
```python
agent = ShipmentIntakeAgent(...)
result = agent.process("input.csv")  # v1 API
```

Your v2 agent MUST provide:
```python
def process_file(self, input_file):
    """v1-compatible wrapper"""
    input_data = IntakeInput(file_path=str(input_file))
    result = self.run(input_data)
    if not result.success:
        raise RuntimeError(result.error)
    return {
        "shipments": result.data.shipments,
        "metadata": result.data.metadata,
        "validation_errors": result.data.validation_errors
    }
```

### Pitfall 3: Not Handling Framework Errors

**Wrong:**
```python
result = agent.run(input_data)
data = result.data  # Crashes if result.success = False!
```

**Correct:**
```python
result = agent.run(input_data)
if not result.success:
    logger.error(f"Agent failed: {result.error}")
    raise RuntimeError(result.error)

data = result.data  # Safe: only if success
```

---

## Deployment Checklist

### Pre-Deployment

- [ ] All v2 agents pass unit tests
- [ ] Output equivalence verified (v1 vs v2)
- [ ] Performance benchmarks show <5% overhead
- [ ] Backward compatibility tested with existing code
- [ ] Zero Hallucination guarantee verified (for calculator agent)

### Deployment

- [ ] Deploy v2 agents alongside v1 (parallel deployment)
- [ ] Add feature flag: `USE_V2_AGENTS=false` (default off)
- [ ] Update documentation with v2 examples
- [ ] Monitor error rates and performance

### Post-Deployment

- [ ] Run v2 in shadow mode (compare with v1, don't use results)
- [ ] Gradually increase v2 traffic (10% → 50% → 100%)
- [ ] Monitor for 1 week at each stage
- [ ] Mark v1 as deprecated once v2 is stable
- [ ] Schedule v1 removal (e.g., +6 months)

---

## FAQ

**Q: Do I need to refactor all agents at once?**
A: No. Refactor incrementally: Agent 1 → test → Agent 2 → test → etc.

**Q: Will v2 agents work with v1 pipeline?**
A: Yes, if you provide v1-compatible wrapper methods.

**Q: What if I need features not in the framework?**
A: Implement custom logic in your agent. Framework is infrastructure, not business logic.

**Q: How do I test Zero Hallucination guarantee?**
A: Run calculations 3+ times with same input. Outputs must be bit-identical.

**Q: Can I mix v1 and v2 agents in the same pipeline?**
A: Yes, temporarily. Long-term, migrate all agents to v2 for consistency.

---

## Support & Resources

### Documentation
- GreenLang SDK: `C:\Users\aksha\Code-V1_GreenLang\core\greenlang\sdk\`
- Refactoring Report: `REFACTORING_REPORT.md`
- ADRs: See REFACTORING_REPORT.md → "Architecture Decision Records"

### Examples
- Agent 1 v2: `agents/shipment_intake_agent_v2.py`
- Agent 2 v2: `agents/emissions_calculator_agent_v2.py`
- CBAM-Refactored (reference): `../CBAM-Refactored/` (uses hypothetical framework)

### Get Help
- Internal: GL-CBAM-APP Refactoring Team
- Framework issues: GreenLang SDK team

---

**Guide Version:** 1.0
**Last Updated:** 2025-11-09
**Next Update:** Upon completion of Pipeline refactoring

---

END OF MIGRATION GUIDE
