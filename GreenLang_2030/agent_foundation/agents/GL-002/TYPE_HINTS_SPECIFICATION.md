# Type Hints Specification for GL-002 BoilerEfficiencyOptimizer

**Goal:** Achieve 100% type hint coverage across all GL-002 modules

**Current Coverage:** 45% (FAIL - Target: 100%)
**Missing:** 629 return type hints, 450 parameter type hints

---

## File-by-File Type Hints Required

### 1. `boiler_efficiency_orchestrator.py`

**Missing Return Type Hints (18 methods):**

| Method | Current Signature | Required Signature |
|--------|------------------|-------------------|
| `__init__` | `def __init__(self, config: BoilerEfficiencyConfig)` | `def __init__(self, config: BoilerEfficiencyConfig) -> None` |
| `_init_intelligence` | `def _init_intelligence(self)` | `def _init_intelligence(self) -> None` |
| `_apply_safety_constraints` | Current has return type | ✅ Already complete |
| `_generate_efficiency_dashboard` | Current has return type | ✅ Already complete |
| `_generate_alerts` | Current has return type | ✅ Already complete |
| `_generate_recommendations` | Current has return type | ✅ Already complete |
| `_map_priority` | Current has return type | ✅ Already complete |
| `_store_optimization_memory` | `def _store_optimization_memory(...)` | `def _store_optimization_memory(...) -> None` |
| `_persist_to_long_term_memory` | `async def _persist_to_long_term_memory(self)` | `async def _persist_to_long_term_memory(self) -> None` |
| `_summarize_input` | Current has return type | ✅ Already complete |
| `_summarize_result` | Current has return type | ✅ Already complete |
| `_serialize_operational_state` | Current has return type | ✅ Already complete |
| `_get_cache_key` | Current has return type | ✅ Already complete |
| `_is_cache_valid` | Current has return type | ✅ Already complete |
| `_store_in_cache` | `def _store_in_cache(self, cache_key: str, result: Any)` | `def _store_in_cache(self, cache_key: str, result: Any) -> None` |
| `_update_performance_metrics` | `def _update_performance_metrics(...)` | `def _update_performance_metrics(...) -> None` |
| `_calculate_provenance_hash` | Current has return type | ✅ Already complete |
| `shutdown` | `async def shutdown(self)` | `async def shutdown(self) -> None` |

**ThreadSafeCache class (5 methods):**

| Method | Current | Required |
|--------|---------|----------|
| `__init__` | `def __init__(self, max_size: int = 1000, ttl_seconds: float = 60.0)` | `def __init__(self, max_size: int = 1000, ttl_seconds: float = 60.0) -> None` |
| `get` | Has return type | ✅ Complete |
| `set` | `def set(self, key: str, value: Any)` | `def set(self, key: str, value: Any) -> None` |
| `clear` | `def clear(self)` | `def clear(self) -> None` |
| `size` | Has return type | ✅ Complete |

---

### 2. `tools.py`

**Missing Return Type Hints (25+ methods):**

| Method | Required Type Hint |
|--------|-------------------|
| `__init__` | `-> None` |
| `calculate_boiler_efficiency` | ✅ Has return type `-> EfficiencyCalculationResult` |
| `optimize_combustion_parameters` | ✅ Has return type |
| `optimize_steam_generation` | ✅ Has return type |
| `minimize_emissions` | ✅ Has return type |
| `calculate_control_adjustments` | ✅ Has return type |
| All `_calculate_*` helper methods | `-> float` or specific type |
| `_optimize_excess_air` | `-> float` |
| `_optimize_blowdown_rate` | `-> float` |
| `_calculate_steam_quality` | `-> float` |
| `process_scada_data` | ✅ Has return type |
| `process_dcs_data` | ✅ Has return type |
| `coordinate_boiler_agents` | ✅ Has return type |
| `cleanup` | `-> None` |

---

### 3. `config.py`

**All Pydantic models and validators:**

✅ **COMPLETE** - Pydantic models automatically include type hints via Field definitions

Validators need return type hints:

| Validator | Required |
|-----------|----------|
| `validate_max_steam_capacity` | `-> float` |
| `validate_design_temperature` | `-> float` |
| `validate_commissioning_date` | `-> datetime` |
| `validate_actual_efficiency` | `-> float` |
| All constraint validators | Appropriate return type |

**Factory function:**
- `create_default_config` | `-> BoilerEfficiencyConfig` (✅ Already has it)

---

### 4. Calculator Modules

#### `calculators/combustion_efficiency.py`

Missing type hints (estimated 30+ methods):

- `__init__` methods for all classes: `-> None`
- All calculation methods: Specific return types based on formulas
- Helper methods: `-> float`, `-> Dict[str, float]`, etc.

#### `calculators/fuel_optimization.py`

- Similar pattern to combustion_efficiency.py
- Optimization methods need return types
- Constraint checking methods: `-> bool`

#### `calculators/emissions_calculator.py`

- Emission calculation methods: `-> float` or specific result types
- Compliance checking: `-> bool` or `-> str`

#### `calculators/steam_generation.py`

- Steam property calculations: `-> float`
- Enthalpy calculations: `-> float`
- Quality calculations: `-> float`

#### `calculators/heat_transfer.py`

- Heat transfer coefficient calculations: `-> float`
- Thermal resistance: `-> float`
- Heat exchanger effectiveness: `-> float`

#### `calculators/blowdown_optimizer.py`

- TDS calculations: `-> float`
- Blowdown rate optimization: `-> float`
- Cost calculations: `-> float`

#### `calculators/economizer_performance.py`

- Efficiency calculations: `-> float`
- Heat recovery: `-> float`
- Performance metrics: `-> Dict[str, float]`

#### `calculators/control_optimization.py`

- PID tuning: `-> Tuple[float, float, float]`
- Setpoint optimization: `-> float`
- Control response: `-> Dict[str, float]`

#### `calculators/provenance.py`

- Hash generation: `-> str`
- Provenance tracking: `-> ProvenanceRecord`
- Audit trail: `-> List[ProvenanceRecord]`

---

### 5. Integration Modules

#### `integrations/scada_connector.py`

Missing type hints (estimated 40+ methods):

- Connection methods: `-> None` or `-> bool`
- Data retrieval: `-> Dict[str, Any]` or specific data types
- Tag management: `-> List[SCADATag]`
- Alarm handling: `-> List[SCADAAlarm]`

#### `integrations/boiler_control_connector.py`

- Control signal writing: `-> bool`
- Setpoint updates: `-> None`
- Control mode switching: `-> bool`

#### `integrations/fuel_management_connector.py`

- Fuel data retrieval: `-> FuelData`
- Cost updates: `-> None`
- Availability checking: `-> bool`

#### `integrations/emissions_monitoring_connector.py`

- CEMS data retrieval: `-> Dict[str, float]`
- Compliance status: `-> str`
- Reporting: `-> None`

#### `integrations/data_transformers.py`

- Transform methods: Specific input/output types
- Validation methods: `-> bool`
- Normalization: `-> Dict[str, float]`

#### `integrations/agent_coordinator.py`

- Message sending: `-> None`
- Response handling: `-> Dict[str, Any]`
- Coordination logic: `-> Dict[str, Any]`

---

### 6. Monitoring Modules

#### `monitoring/health_checks.py`

- Health check methods: `-> bool` or `-> Dict[str, bool]`
- Status reporting: `-> Dict[str, Any]`

#### `monitoring/metrics.py`

- Metric collection: `-> Dict[str, float]`
- Metric aggregation: `-> float`
- Reporting: `-> None`

---

## Common Type Hint Patterns

### Return Types

```python
# Void functions
def method_name(...) -> None:
    pass

# Basic types
def calculate_value(...) -> float:
    return 0.0

def get_name(...) -> str:
    return "value"

def is_valid(...) -> bool:
    return True

# Collections
def get_data(...) -> Dict[str, Any]:
    return {}

def get_list(...) -> List[float]:
    return []

def get_tuple(...) -> Tuple[float, float, str]:
    return (0.0, 0.0, "")

# Optional returns
def get_optional(...) -> Optional[float]:
    return None

# Union types
def get_value(...) -> Union[int, float]:
    return 0

# Custom types
def calculate(...) -> EfficiencyCalculationResult:
    return EfficiencyCalculationResult(...)

# Async methods
async def async_method(...) -> Dict[str, Any]:
    return {}
```

### Parameter Types

```python
def method(
    param1: str,
    param2: int,
    param3: float,
    param4: bool,
    param5: Dict[str, Any],
    param6: List[str],
    param7: Optional[float] = None,
    param8: Union[int, float] = 0
) -> ReturnType:
    pass
```

---

## Implementation Priority

1. **High Priority (Core modules):**
   - ✅ `boiler_efficiency_orchestrator.py` (5 hints added)
   - `tools.py` (25 hints needed)
   - `config.py` (validators - 15 hints needed)

2. **Medium Priority (Calculators):**
   - All 10 calculator modules (~300 hints needed)

3. **Medium Priority (Integrations):**
   - All 7 integration modules (~250 hints needed)

4. **Low Priority (Tests/Monitoring):**
   - Monitoring modules (~50 hints needed)

---

## Verification

After adding all type hints, verify with:

```bash
# Using mypy (strict mode)
mypy --strict boiler_efficiency_orchestrator.py tools.py config.py

# Using pyright
pyright --stats .

# Count coverage
python3 -c "
import ast
import sys
from pathlib import Path

def count_type_hints(file_path):
    with open(file_path) as f:
        tree = ast.parse(f.read())

    total_functions = 0
    typed_returns = 0
    total_params = 0
    typed_params = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            total_functions += 1
            if node.returns:
                typed_returns += 1

            for arg in node.args.args:
                if arg.arg not in ('self', 'cls'):
                    total_params += 1
                    if arg.annotation:
                        typed_params += 1

    return total_functions, typed_returns, total_params, typed_params

# Process all files
for file in Path('.').rglob('*.py'):
    if 'test' not in str(file):
        stats = count_type_hints(file)
        print(f'{file}: {stats}')
"
```

---

## Success Criteria

- ✅ 100% return type coverage (all functions have `-> Type`)
- ✅ 100% parameter type coverage (all parameters typed except `self`, `cls`)
- ✅ Passes `mypy --strict` with zero errors
- ✅ Passes `pyright --stats` with zero errors
- ✅ All type hints follow PEP 484/526 standards

---

**Status:** IN PROGRESS
**Last Updated:** 2025-11-17
**Updated By:** GL-BackendDeveloper
