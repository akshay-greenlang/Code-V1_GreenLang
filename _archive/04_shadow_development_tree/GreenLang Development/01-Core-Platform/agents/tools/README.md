# GreenLang Shared Tool Library

A centralized, production-ready tool library for all GreenLang agents, providing reusable components for emissions calculations, financial analysis, and grid integration.

## Overview

The shared tool library eliminates code duplication across Phase 2-4 agents by providing standardized, well-tested tools that can be used directly or via ChatSession integration.

### Key Benefits

- **Code Reuse**: Single implementation used by 10+ agents
- **Consistency**: Standardized calculations across all agents
- **Auditability**: Complete citation tracking for all calculations
- **Type Safety**: Full type hints and validation
- **Test Coverage**: Comprehensive test suites for all tools
- **Performance**: Optimized implementations with execution metrics

## Architecture

```
greenlang/agents/tools/
├── base.py          # BaseTool, ToolResult, ToolDef, CompositeTool
├── registry.py      # ToolRegistry for auto-discovery
├── emissions.py     # Emissions calculation tools
├── financial.py     # Financial metrics tools (NEW)
├── grid.py          # Grid integration tools (NEW)
└── __init__.py      # Auto-registration and exports
```

## Available Tools

### Emissions Tools

#### 1. CalculateEmissionsTool

Calculate exact CO2e emissions from fuel consumption.

```python
from greenlang.agents.tools import CalculateEmissionsTool

tool = CalculateEmissionsTool()
result = tool.execute(
    fuel_type="natural_gas",
    amount=1000,
    unit="therms",
    emission_factor=5.31,
    emission_factor_unit="kgCO2e/therm",
    country="US"
)

print(result.data["emissions_kg_co2e"])  # 5310.0
```

**Safety**: DETERMINISTIC
**Used By**: All Phase 2-4 agents

#### 2. AggregateEmissionsTool

Aggregate total emissions from multiple sources.

```python
from greenlang.agents.tools import AggregateEmissionsTool

tool = AggregateEmissionsTool()
result = tool.execute(
    emissions=[
        {"fuel_type": "natural_gas", "co2e_emissions_kg": 5000},
        {"fuel_type": "electricity", "co2e_emissions_kg": 3000},
    ]
)

print(result.data["total_co2e_kg"])      # 8000.0
print(result.data["total_co2e_tons"])    # 8.0
```

**Safety**: DETERMINISTIC
**Used By**: All Phase 2-4 agents

#### 3. CalculateBreakdownTool

Calculate percentage breakdown of emissions by source.

```python
from greenlang.agents.tools import CalculateBreakdownTool

tool = CalculateBreakdownTool()
result = tool.execute(
    emissions=[
        {"fuel_type": "natural_gas", "co2e_emissions_kg": 5000},
        {"fuel_type": "electricity", "co2e_emissions_kg": 3000},
    ],
    total_emissions=8000
)

print(result.data["by_fuel_percent"])  # {"natural_gas": 62.5, "electricity": 37.5}
```

**Safety**: DETERMINISTIC
**Used By**: Phase 2 agents

#### 4. CalculateScopeEmissionsTool (NEW)

Calculate emissions by GHG Protocol Scope (1, 2, 3).

```python
from greenlang.agents.tools import CalculateScopeEmissionsTool

tool = CalculateScopeEmissionsTool()
result = tool.execute(
    scope_1_sources=[
        {"source_name": "Natural Gas Combustion", "co2e_emissions_kg": 5000}
    ],
    scope_2_sources=[
        {"source_name": "Purchased Electricity", "co2e_emissions_kg": 3000}
    ],
    scope_3_sources=[
        {"source_name": "Upstream Emissions", "co2e_emissions_kg": 1000}
    ]
)

print(result.data["scope_1_kg"])        # 5000.0
print(result.data["scope_2_kg"])        # 3000.0
print(result.data["scope_3_kg"])        # 1000.0
print(result.data["scope_1_percent"])   # 55.56
```

**Safety**: DETERMINISTIC
**Used By**: Phase 4 compliance agents

#### 5. RegionalEmissionFactorTool (NEW)

Get regional emission factors for grid electricity.

```python
from greenlang.agents.tools import RegionalEmissionFactorTool

tool = RegionalEmissionFactorTool()
result = tool.execute(
    region="US-WECC",
    year=2025,
    include_marginal=True,
    temporal_hour=16  # 4 PM
)

print(result.data["avg_emission_factor"])       # 0.295 kgCO2e/kWh
print(result.data["marginal_emission_factor"])  # 0.350 kgCO2e/kWh
print(result.data["temporal_emission_factor"])  # 0.339 kgCO2e/kWh (peak hour)
```

**Safety**: DETERMINISTIC
**Used By**: Phase 3 v3 agents, Phase 4 grid-connected agents

**Supported Regions**:
- US: National, eGRID regions (WECC, ERCOT, etc.), States (CA, NY, TX, etc.)
- International: EU, UK, DE, FR, CN, IN, JP, AU, CA, BR

---

### Financial Tools

#### 6. FinancialMetricsTool (NEW - Priority 1)

Calculate comprehensive financial metrics for energy projects.

```python
from greenlang.agents.tools import FinancialMetricsTool

tool = FinancialMetricsTool()
result = tool.execute(
    capital_cost=100000,
    annual_savings=12000,
    lifetime_years=25,
    discount_rate=0.06,
    annual_om_cost=500,
    energy_cost_escalation=0.025,
    incentives=[
        {"name": "IRA 2022 ITC", "amount": 30000, "year": 0}
    ],
    include_depreciation=True,
    tax_rate=0.21,
    salvage_value=5000
)

print(result.data["npv"])                      # Net Present Value
print(result.data["irr"])                      # Internal Rate of Return
print(result.data["simple_payback_years"])     # Simple Payback Period
print(result.data["discounted_payback_years"]) # Discounted Payback
print(result.data["lifecycle_cost"])           # Total Lifecycle Cost
print(result.data["benefit_cost_ratio"])       # B/C Ratio
```

**Safety**: DETERMINISTIC
**Used By**: All Phase 2-4 agents with financial analysis

**Features**:
- Net Present Value (NPV) calculation
- Internal Rate of Return (IRR) via Newton's method
- Simple and discounted payback periods
- Lifecycle cost analysis
- IRA 2022 incentive integration
- MACRS depreciation tax benefits
- Energy cost escalation
- Salvage value support

**Impact**: Eliminates 10+ duplicate implementations across agents

---

### Grid/Utility Tools

#### 7. GridIntegrationTool (NEW - Priority 1)

Analyze grid capacity, demand charges, TOU rates, and optimization opportunities.

```python
from greenlang.agents.tools import GridIntegrationTool

tool = GridIntegrationTool()

# 24-hour load profile (kW)
load_profile = [
    200, 180, 170, 165, 170, 190,  # Night
    250, 300, 350, 400, 420, 450,  # Morning
    480, 500, 490, 485, 470, 460,  # Afternoon (peak)
    440, 400, 350, 300, 250, 220   # Evening
]

result = tool.execute(
    peak_demand_kw=500,
    load_profile=load_profile,
    grid_capacity_kw=600,
    demand_charge_per_kw=15.0,
    energy_rate_per_kwh=0.12,
    tou_rates={"peak": 0.18, "off_peak": 0.08},
    tou_schedule={
        "peak": [12, 13, 14, 15, 16, 17, 18],
        "off_peak": [0,1,2,3,4,5,6,7,8,9,10,11,19,20,21,22,23]
    },
    dr_program_available=True,
    dr_incentive_per_kwh=0.50,
    storage_capacity_kwh=200,
    storage_power_kw=100
)

print(result.data["capacity_utilization_percent"])   # 83.33%
print(result.data["monthly_demand_charge"])          # $7,500
print(result.data["peak_shaving_potential_savings"]) # Peak shaving savings
print(result.data["dr_potential_savings"])           # DR program benefits
print(result.data["storage_annual_savings"])         # Storage optimization
```

**Safety**: IDEMPOTENT
**Used By**: All Phase 3 v3 agents, Phase 4 grid-connected agents

**Features**:
- Grid capacity utilization analysis
- Demand charge optimization
- Time-of-use (TOU) rate calculations
- Demand response (DR) program benefits
- Peak shaving opportunity analysis
- Energy storage optimization
- Arbitrage value calculation
- Support for 24-hour or 8760-hour profiles

**Impact**: Eliminates duplicate grid analysis code across Phase 3 agents

---

## Usage Patterns

### Direct Tool Usage

```python
from greenlang.agents.tools import FinancialMetricsTool

# Instantiate tool
tool = FinancialMetricsTool()

# Execute with parameters
result = tool.execute(
    capital_cost=50000,
    annual_savings=8000,
    lifetime_years=25
)

# Check success
if result.success:
    print(f"NPV: ${result.data['npv']:,.2f}")

    # Access citations
    for citation in result.citations:
        print(citation.formatted())
else:
    print(f"Error: {result.error}")
```

### ChatSession Integration

```python
from greenlang.agents.tools import get_registry
from greenlang.agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()

        # Get tool definitions from registry
        registry = get_registry()
        tool_defs = registry.get_tool_defs(category="financial")

        # Register with ChatSession
        self.chat_session.register_tools(tool_defs)
```

### Tool Registry

```python
from greenlang.agents.tools import get_registry

# Get global registry
registry = get_registry()

# List all tools
all_tools = registry.list_tools()
print(f"Total tools: {len(all_tools)}")

# List by category
financial_tools = registry.list_tools(category="financial")
print(f"Financial tools: {financial_tools}")

# Get specific tool
tool = registry.get("calculate_financial_metrics")
result = tool.execute(...)

# Get tool catalog
catalog = registry.get_catalog()
print(catalog)
```

### Tool Discovery

```python
from greenlang.agents.tools import get_registry

registry = get_registry()

# Discover tools in a module
discovered = registry.discover("greenlang.agents.tools.financial")
print(f"Discovered: {discovered}")

# Discover all tools
all_discovered = registry.discover_all()
print(f"Total discovered: {len(all_discovered)}")
```

---

## Tool Result Format

All tools return a `ToolResult` object with:

```python
@dataclass
class ToolResult:
    success: bool                           # Execution status
    data: Dict[str, Any]                    # Result data
    error: Optional[str]                    # Error message (if failed)
    metadata: Dict[str, Any]                # Additional context
    citations: List[Any]                    # Calculation citations
    execution_time_ms: float                # Execution time
```

Example:

```python
result = tool.execute(...)

if result.success:
    # Access result data
    npv = result.data["npv"]

    # Check execution time
    print(f"Executed in {result.execution_time_ms:.2f} ms")

    # Review citations
    for citation in result.citations:
        print(citation.formatted())

    # Access metadata
    print(result.metadata["summary"])
else:
    # Handle error
    print(f"Tool failed: {result.error}")
```

---

## Citation Tracking

All tools provide complete citation tracking for auditability:

### Calculation Citations

```python
from greenlang.agents.citations import CalculationCitation

citation = CalculationCitation(
    step_name="calculate_npv",
    formula="NPV = Σ(CF_t / (1 + r)^t)",
    inputs={"capital_cost": 50000, "discount_rate": 0.05},
    output={"npv": 67890.12, "unit": "USD"}
)

print(citation.formatted())
# "calculate_npv: NPV = Σ(CF_t / (1 + r)^t) = 67890.12 USD"
```

### Emission Factor Citations

```python
from greenlang.agents.citations import create_emission_factor_citation

citation = create_emission_factor_citation(
    source="EPA eGRID 2025",
    factor_name="US Grid Average",
    value=0.385,
    unit="kgCO2e/kWh",
    version="2025.1",
    confidence="high"
)

print(citation.formatted())
# "EPA eGRID 2025 v2025.1: US Grid Average = 0.385 kgCO2e/kWh [Updated: 2025-01-15] (Confidence: high)"
```

---

## Creating Custom Tools

### Method 1: Extend BaseTool

```python
from greenlang.agents.tools import BaseTool, ToolDef, ToolResult, ToolSafety

class MyCustomTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="my_custom_tool",
            description="Description for LLM",
            safety=ToolSafety.DETERMINISTIC
        )

    def execute(self, param1: float, param2: str) -> ToolResult:
        try:
            # Tool logic here
            result_value = param1 * 2

            return ToolResult(
                success=True,
                data={"result": result_value},
                metadata={"param2": param2}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )

    def get_tool_def(self) -> ToolDef:
        return ToolDef(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "required": ["param1"],
                "properties": {
                    "param1": {"type": "number"},
                    "param2": {"type": "string"}
                }
            },
            safety=self.safety
        )
```

### Method 2: Use @tool Decorator

```python
from greenlang.agents.tools import tool, ToolResult, ToolSafety

@tool(
    name="calculate_area",
    description="Calculate area of rectangle",
    parameters={
        "type": "object",
        "required": ["width", "height"],
        "properties": {
            "width": {"type": "number"},
            "height": {"type": "number"}
        }
    },
    safety=ToolSafety.DETERMINISTIC
)
def calculate_area(width: float, height: float) -> ToolResult:
    return ToolResult(
        success=True,
        data={"area": width * height}
    )
```

---

## Testing

All tools have comprehensive test coverage:

```bash
# Test all tools
pytest tests/agents/tools/

# Test specific tool
pytest tests/agents/tools/test_financial.py
pytest tests/agents/tools/test_grid.py

# Test with coverage
pytest tests/agents/tools/ --cov=greenlang.agents.tools --cov-report=html
```

Example test:

```python
import pytest
from greenlang.agents.tools import FinancialMetricsTool

def test_npv_calculation():
    tool = FinancialMetricsTool()
    result = tool.execute(
        capital_cost=50000,
        annual_savings=8000,
        lifetime_years=25,
        discount_rate=0.05
    )

    assert result.success
    assert result.data["npv"] > 0
    assert result.data["irr"] is not None
```

---

## Performance

All tools track execution metrics:

```python
from greenlang.agents.tools import FinancialMetricsTool

tool = FinancialMetricsTool()

# Execute multiple times
for i in range(100):
    result = tool.execute(...)

# Get performance stats
stats = tool.get_stats()
print(f"Executions: {stats['executions']}")
print(f"Avg time: {stats['avg_time_ms']:.2f} ms")
print(f"Total time: {stats['total_time_ms']:.2f} ms")
```

---

## Best Practices

### 1. Use Registry for Discovery

```python
# Good - Auto-discovery
from greenlang.agents.tools import get_registry

registry = get_registry()
tool = registry.get("calculate_financial_metrics")

# Avoid - Manual instantiation everywhere
from greenlang.agents.tools.financial import FinancialMetricsTool
tool = FinancialMetricsTool()
```

### 2. Check Result Success

```python
# Good - Always check success
result = tool.execute(...)
if result.success:
    process_result(result.data)
else:
    handle_error(result.error)

# Avoid - Assuming success
result = tool.execute(...)
value = result.data["npv"]  # May fail if not successful
```

### 3. Include Citations

```python
# Good - Include citations for auditability
return ToolResult(
    success=True,
    data={"result": value},
    citations=[calculation_citation]
)

# Avoid - No citations
return ToolResult(
    success=True,
    data={"result": value}
)
```

### 4. Validate Inputs

```python
# Good - Validate inputs
def execute(self, amount: float) -> ToolResult:
    if amount < 0:
        return ToolResult(
            success=False,
            error="amount must be non-negative"
        )
    # ... proceed

# Avoid - No validation
def execute(self, amount: float) -> ToolResult:
    result = amount * 2  # May fail with invalid input
```

### 5. Use Type Hints

```python
# Good - Complete type hints
def execute(
    self,
    capital_cost: float,
    annual_savings: float,
    lifetime_years: int
) -> ToolResult:
    ...

# Avoid - No type hints
def execute(self, capital_cost, annual_savings, lifetime_years):
    ...
```

---

## Security Features (Phase 6.3)

The tool library includes production-grade security features to ensure safe, auditable, and controlled tool execution.

### Overview

All tools are protected by a comprehensive security framework that provides:

- **Input Validation**: Schema-based and custom validation rules
- **Rate Limiting**: Token bucket rate limiting with per-tool and per-user controls
- **Audit Logging**: Privacy-safe execution logging with rotation and retention
- **Access Control**: Tool whitelisting and blacklisting
- **Execution Limits**: Timeouts, memory limits, and retry controls

### Security Architecture

```
Security Layer:
├── validation.py      # Input validation framework
├── rate_limiting.py   # Token bucket rate limiter
├── audit.py          # Audit logging system
└── security_config.py # Centralized configuration
```

### Input Validation

Comprehensive validation framework with multiple validator types:

```python
from greenlang.agents.tools.validation import (
    RangeValidator,
    TypeValidator,
    EnumValidator,
    RegexValidator,
    CustomValidator,
    CompositeValidator
)

# Create tool with validation rules
class ValidatedTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="validated_tool",
            description="Tool with validation",
            validation_rules={
                "amount": [
                    TypeValidator(float, coerce=True),
                    RangeValidator(min_value=0, max_value=1000)
                ],
                "fuel_type": [
                    EnumValidator(["natural_gas", "electricity", "diesel"])
                ],
                "code": [
                    RegexValidator(pattern=r"^[A-Z]{2}[0-9]{4}$")
                ]
            }
        )
```

**Validator Types**:

- **RangeValidator**: Numeric range validation (min/max, inclusive/exclusive)
- **TypeValidator**: Type checking with optional coercion
- **EnumValidator**: Allowed values validation (case-sensitive or not)
- **RegexValidator**: Pattern matching validation
- **CustomValidator**: Custom validation functions
- **CompositeValidator**: Combine validators with AND/OR logic

### Rate Limiting

Token bucket rate limiting protects against abuse:

```python
from greenlang.agents.tools.security_config import configure_security

# Configure global rate limits
configure_security(
    preset="production",
    default_rate_per_second=10,
    default_burst_size=20,
    per_tool_limits={
        "calculate_emissions": (100, 200),  # High-frequency tool
        "grid_integration": (5, 10),        # Resource-intensive tool
    },
    per_user_rate_limiting=True
)
```

**Features**:
- Token bucket algorithm for smooth rate limiting
- Per-tool rate limits
- Per-user rate limits
- Configurable burst capacity
- Automatic token refill
- Thread-safe operation

**Handling Rate Limits**:

```python
result = tool(amount=100)

if not result.success:
    if result.metadata.get("rate_limit_exceeded"):
        retry_after = result.metadata["retry_after_seconds"]
        print(f"Rate limited. Retry after {retry_after:.2f}s")
```

### Audit Logging

All tool executions are logged with privacy protection:

```python
from greenlang.agents.tools.audit import configure_audit_logger

# Configure audit logging
configure_audit_logger(
    log_file=Path("logs/audit/tool_audit.jsonl"),
    retention_days=90,
    auto_rotate=True,
    max_log_size_mb=100
)

# Set execution context
tool.set_context(user_id="user123", session_id="session456")

# Execute tool (automatically logged)
result = tool(amount=100)
```

**Features**:
- Privacy-safe hashing (no raw sensitive data)
- JSON-based log format
- Automatic log rotation
- Retention policy enforcement
- Query interface for log analysis
- Thread-safe operation

**Querying Logs**:

```python
from greenlang.agents.tools.audit import get_audit_logger

logger = get_audit_logger()

# Query by tool
logs = logger.query_logs(tool_name="calculate_emissions")

# Query by user
logs = logger.query_logs(user_id="user123")

# Query by time range
logs = logger.query_logs(
    start_time=datetime(2025, 11, 1),
    end_time=datetime(2025, 11, 7),
    success_only=True
)

# Get tool statistics
stats = logger.get_tool_stats("calculate_emissions")
print(f"Success rate: {stats['success_rate_percentage']}%")
print(f"Avg execution: {stats['avg_execution_time_ms']}ms")
```

### Security Configuration

Centralized security configuration with presets:

```python
from greenlang.agents.tools.security_config import configure_security

# Use production preset
configure_security(preset="production")

# Use development preset (relaxed)
configure_security(preset="development")

# Use testing preset (minimal)
configure_security(preset="testing")

# Use high security preset
configure_security(preset="high_security")

# Custom configuration
configure_security(
    enable_validation=True,
    strict_validation=True,
    enable_rate_limiting=True,
    enable_audit_logging=True,
    max_execution_time_seconds=300.0,
    tool_blacklist={"deprecated_tool"},
    audit_retention_days=90
)
```

**Configuration Options**:

```python
SecurityConfig(
    # Input Validation
    enable_validation=True,
    strict_validation=False,  # Fail on warnings
    sanitize_inputs=True,

    # Rate Limiting
    enable_rate_limiting=True,
    default_rate_per_second=10,
    default_burst_size=20,
    per_user_rate_limiting=False,

    # Audit Logging
    enable_audit_logging=True,
    audit_retention_days=90,

    # Access Control
    tool_whitelist=None,  # None = all allowed
    tool_blacklist=set(),
    max_concurrent_tools=5,

    # Execution Limits
    max_execution_time_seconds=300.0,
    max_memory_mb=1024,
    max_retries=3,

    # Privacy
    hash_sensitive_inputs=True,
    sensitive_param_names={"password", "api_key", "secret"},
    block_injection_attempts=True
)
```

### Tool Access Control

Whitelist or blacklist tools for security:

```python
from greenlang.agents.tools.security_config import configure_security

# Whitelist approach (only these tools allowed)
configure_security(
    tool_whitelist={
        "calculate_emissions",
        "calculate_financial_metrics",
        "grid_integration"
    }
)

# Blacklist approach (all except these allowed)
configure_security(
    tool_blacklist={
        "deprecated_tool",
        "experimental_tool"
    }
)
```

### Temporary Security Changes

Use context manager for temporary configuration:

```python
from greenlang.agents.tools.security_config import SecurityContext

# Temporarily disable rate limiting for testing
with SecurityContext(enable_rate_limiting=False):
    result = tool(amount=100)
# Rate limiting restored after context

# Temporarily enable debug mode
with SecurityContext(debug_mode=True, strict_validation=False):
    result = tool(amount=100)
```

### Security Best Practices

1. **Always use production config in production**:
   ```python
   configure_security(preset="production")
   ```

2. **Set execution context for audit trails**:
   ```python
   tool.set_context(user_id=current_user_id, session_id=session_id)
   ```

3. **Handle rate limits gracefully**:
   ```python
   result = tool(amount=100)
   if result.metadata.get("rate_limit_exceeded"):
       time.sleep(result.metadata["retry_after_seconds"])
       result = tool(amount=100)  # Retry
   ```

4. **Validate sensitive inputs**:
   ```python
   validation_rules = {
       "api_key": [
           RegexValidator(pattern=r"^[A-Za-z0-9_-]{32}$")
       ]
   }
   ```

5. **Monitor audit logs regularly**:
   ```python
   logger = get_audit_logger()
   stats = logger.get_stats()
   if stats["failure_count"] > threshold:
       alert_admin()
   ```

### Performance Overhead

Security features are optimized for minimal overhead:

- **Validation**: <0.1ms per parameter
- **Rate Limiting**: <0.5ms per check
- **Audit Logging**: <1ms per execution (async write)
- **Total Overhead**: <2ms per tool execution

### Security Testing

Comprehensive test suite for security features:

```bash
# Run security tests
pytest tests/agents/tools/test_security.py

# Test coverage
pytest tests/agents/tools/test_security.py --cov=greenlang.agents.tools
```

**Test Coverage**:
- Input validation (all validator types)
- Rate limiting (token bucket, per-tool, per-user)
- Audit logging (privacy, rotation, queries)
- Security configuration (presets, overrides)
- Integration tests (end-to-end security)

---

## Telemetry System (Phase 6.2)

The tool library includes a comprehensive telemetry system for monitoring tool usage, performance, and errors.

### Overview

The telemetry system provides:
- **Usage Tracking**: Call counts, success/failure rates
- **Performance Metrics**: Execution time percentiles (p50, p95, p99)
- **Error Tracking**: Error types and frequencies
- **Rate Limit Monitoring**: Track rate limit violations
- **Validation Monitoring**: Track validation failures
- **Multiple Export Formats**: JSON, Prometheus, CSV

### Architecture

```
Telemetry Layer:
├── telemetry.py          # TelemetryCollector and metrics
└── base.py               # Automatic telemetry recording
```

### Automatic Telemetry

All tools automatically record telemetry when executed:

```python
from greenlang.agents.tools import FinancialMetricsTool
from greenlang.agents.tools.telemetry import get_telemetry

# Create and execute tool
tool = FinancialMetricsTool()
result = tool(
    capital_cost=50000,
    annual_savings=8000,
    lifetime_years=25
)

# Get telemetry metrics
telemetry = get_telemetry()
metrics = telemetry.get_tool_metrics("calculate_financial_metrics")

print(f"Total calls: {metrics.total_calls}")
print(f"Success rate: {metrics.successful_calls / metrics.total_calls * 100:.1f}%")
print(f"Avg execution time: {metrics.avg_execution_time_ms:.2f}ms")
print(f"P95 execution time: {metrics.p95_execution_time_ms:.2f}ms")
```

### Telemetry Metrics

Each tool tracks comprehensive metrics:

```python
@dataclass
class ToolMetrics:
    tool_name: str
    total_calls: int              # Total executions
    successful_calls: int         # Successful executions
    failed_calls: int             # Failed executions
    total_execution_time_ms: float  # Total time
    avg_execution_time_ms: float    # Average time
    p50_execution_time_ms: float    # Median time
    p95_execution_time_ms: float    # 95th percentile
    p99_execution_time_ms: float    # 99th percentile
    rate_limit_hits: int          # Rate limit violations
    validation_failures: int      # Validation failures
    last_called: datetime         # Last execution time
    error_counts_by_type: Dict[str, int]  # Error breakdown
```

### Getting Metrics

**Single Tool Metrics**:

```python
from greenlang.agents.tools.telemetry import get_telemetry

telemetry = get_telemetry()

# Get metrics for specific tool
metrics = telemetry.get_tool_metrics("calculate_financial_metrics")

print(f"Tool: {metrics.tool_name}")
print(f"Calls: {metrics.total_calls}")
print(f"Success Rate: {metrics.successful_calls / metrics.total_calls * 100:.1f}%")
print(f"Avg Time: {metrics.avg_execution_time_ms:.2f}ms")
print(f"P95 Time: {metrics.p95_execution_time_ms:.2f}ms")
print(f"Rate Limits: {metrics.rate_limit_hits}")
print(f"Validation Failures: {metrics.validation_failures}")

# Error breakdown
for error_type, count in metrics.error_counts_by_type.items():
    print(f"  {error_type}: {count}")
```

**All Tools Metrics**:

```python
# Get metrics for all tools
all_metrics = telemetry.get_all_metrics()

for tool_name, metrics in all_metrics.items():
    print(f"{tool_name}: {metrics.total_calls} calls, {metrics.avg_execution_time_ms:.2f}ms avg")
```

**Summary Statistics**:

```python
# Get aggregate statistics
summary = telemetry.get_summary_stats()

print(f"Total Tools: {summary['total_tools']}")
print(f"Total Executions: {summary['total_executions']}")
print(f"Overall Success Rate: {summary['overall_success_rate']:.1f}%")
print(f"Total Rate Limit Hits: {summary['total_rate_limit_hits']}")
print(f"Most Used Tool: {summary['most_used_tool']['name']} ({summary['most_used_tool']['calls']} calls)")
print(f"Slowest Tool: {summary['slowest_tool']['name']} ({summary['slowest_tool']['avg_time_ms']:.2f}ms)")
```

### Exporting Metrics

**JSON Export**:

```python
# Export as JSON
json_data = telemetry.export_metrics(format="json")

print(json.dumps(json_data, indent=2))
# {
#   "summary": {
#     "total_tools": 3,
#     "total_executions": 150,
#     "overall_success_rate": 98.67
#   },
#   "tools": {
#     "calculate_financial_metrics": {
#       "total_calls": 100,
#       "avg_execution_time_ms": 45.2,
#       ...
#     }
#   }
# }
```

**Prometheus Export**:

```python
# Export for Prometheus monitoring
prometheus_metrics = telemetry.export_metrics(format="prometheus")

print(prometheus_metrics)
# # HELP tool_calls_total Total number of tool calls
# # TYPE tool_calls_total counter
# tool_calls_total{tool="calculate_financial_metrics"} 100
# tool_execution_time_milliseconds{tool="calculate_financial_metrics",quantile="0.5"} 42.5
# tool_execution_time_milliseconds{tool="calculate_financial_metrics",quantile="0.95"} 89.3
# ...
```

**CSV Export**:

```python
# Export as CSV
csv_data = telemetry.export_metrics(format="csv")

print(csv_data)
# tool_name,total_calls,successful_calls,failed_calls,avg_execution_time_ms,...
# calculate_financial_metrics,100,98,2,45.23,42.10,89.34,125.67,1,0,...
```

### Prometheus Integration

Integrate with Prometheus for monitoring dashboards:

```python
from flask import Flask, Response
from greenlang.agents.tools.telemetry import get_telemetry

app = Flask(__name__)

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    telemetry = get_telemetry()
    prometheus_metrics = telemetry.export_metrics(format="prometheus")
    return Response(prometheus_metrics, mimetype="text/plain")

if __name__ == '__main__':
    app.run(port=9090)
```

**Prometheus Configuration**:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'greenlang_tools'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
```

**Example Prometheus Queries**:

```promql
# Total tool calls
sum(tool_calls_total)

# Success rate by tool
sum(tool_calls_successful) / sum(tool_calls_total) * 100

# P95 execution time
tool_execution_time_milliseconds{quantile="0.95"}

# Rate limit violations
sum(tool_rate_limit_hits)
```

### Performance Monitoring

Monitor tool performance over time:

```python
import time
from greenlang.agents.tools import FinancialMetricsTool
from greenlang.agents.tools.telemetry import get_telemetry

tool = FinancialMetricsTool()
telemetry = get_telemetry()

# Baseline metrics
before_metrics = telemetry.get_tool_metrics("calculate_financial_metrics")

# Execute 100 calls
for i in range(100):
    tool(capital_cost=50000, annual_savings=8000, lifetime_years=25)

# Compare metrics
after_metrics = telemetry.get_tool_metrics("calculate_financial_metrics")

print(f"Calls: {before_metrics.total_calls} → {after_metrics.total_calls}")
print(f"Avg Time: {before_metrics.avg_execution_time_ms:.2f}ms → {after_metrics.avg_execution_time_ms:.2f}ms")
print(f"P95 Time: {before_metrics.p95_execution_time_ms:.2f}ms → {after_metrics.p95_execution_time_ms:.2f}ms")
```

### Error Tracking

Track and analyze errors:

```python
telemetry = get_telemetry()
metrics = telemetry.get_tool_metrics("calculate_financial_metrics")

# Check error rates
if metrics.failed_calls > 0:
    failure_rate = metrics.failed_calls / metrics.total_calls * 100
    print(f"Failure Rate: {failure_rate:.2f}%")

    # Error breakdown
    print("\nError Types:")
    for error_type, count in metrics.error_counts_by_type.items():
        pct = count / metrics.failed_calls * 100
        print(f"  {error_type}: {count} ({pct:.1f}%)")
```

### Usage Analytics

Analyze tool usage patterns:

```python
telemetry = get_telemetry()
summary = telemetry.get_summary_stats()

# Most used tools
print(f"Most Used Tool: {summary['most_used_tool']['name']}")
print(f"  Calls: {summary['most_used_tool']['calls']}")

# Performance outliers
print(f"\nSlowest Tool: {summary['slowest_tool']['name']}")
print(f"  Avg Time: {summary['slowest_tool']['avg_time_ms']:.2f}ms")

print(f"\nFastest Tool: {summary['fastest_tool']['name']}")
print(f"  Avg Time: {summary['fastest_tool']['avg_time_ms']:.2f}ms")

# Rate limiting issues
if summary['total_rate_limit_hits'] > 0:
    print(f"\n⚠ Rate Limit Hits: {summary['total_rate_limit_hits']}")
    print("Consider increasing rate limits or implementing backoff")
```

### Resetting Metrics

Reset metrics for testing or rotation:

```python
telemetry = get_telemetry()

# Reset specific tool
telemetry.reset_metrics("calculate_financial_metrics")

# Reset all metrics
telemetry.reset_metrics()
```

### Thread Safety

The telemetry system is thread-safe for concurrent usage:

```python
import threading
from greenlang.agents.tools import FinancialMetricsTool

tool = FinancialMetricsTool()

def worker():
    for i in range(100):
        tool(capital_cost=50000, annual_savings=8000, lifetime_years=25)

# Create multiple threads
threads = [threading.Thread(target=worker) for _ in range(10)]

# Start all threads
for t in threads:
    t.start()

# Wait for completion
for t in threads:
    t.join()

# Metrics will be accurate despite concurrent access
telemetry = get_telemetry()
metrics = telemetry.get_tool_metrics("calculate_financial_metrics")
print(f"Total calls from 10 threads: {metrics.total_calls}")  # Should be 1000
```

### Telemetry Best Practices

1. **Monitor regularly**:
   ```python
   # Set up periodic monitoring
   telemetry = get_telemetry()
   summary = telemetry.get_summary_stats()
   if summary['overall_success_rate'] < 95.0:
       alert_admin()
   ```

2. **Export for long-term storage**:
   ```python
   # Export metrics daily
   json_data = telemetry.export_metrics(format="json")
   with open(f"metrics_{date.today()}.json", "w") as f:
       json.dump(json_data, f)
   ```

3. **Track performance degradation**:
   ```python
   metrics = telemetry.get_tool_metrics("calculate_financial_metrics")
   if metrics.avg_execution_time_ms > THRESHOLD:
       investigate_performance_issue()
   ```

4. **Monitor rate limits**:
   ```python
   if metrics.rate_limit_hits > 0:
       print("⚠ Rate limiting active - consider optimization")
   ```

### Telemetry Testing

Comprehensive test suite for telemetry:

```bash
# Run telemetry tests
pytest tests/agents/tools/test_telemetry.py

# Test coverage
pytest tests/agents/tools/test_telemetry.py --cov=greenlang.agents.tools.telemetry
```

**Test Coverage**:
- Metric recording (success, failure, errors)
- Percentile calculations (p50, p95, p99)
- Export formats (JSON, Prometheus, CSV)
- Thread safety
- Integration with tools

---

## Phase 6 Roadmap

### Phase 6.1: Critical Tools (COMPLETE)

- [x] FinancialMetricsTool - Financial analysis (NPV, IRR, payback)
- [x] GridIntegrationTool - Grid capacity and utility optimization
- [x] RegionalEmissionFactorTool - Regional emission factors
- [x] CalculateScopeEmissionsTool - Scope 1/2/3 emissions

### Phase 6.2: Telemetry System (COMPLETE)

- [x] TelemetryCollector - Usage and performance tracking
- [x] Automatic metric recording in BaseTool
- [x] Export formats (JSON, Prometheus, CSV)
- [x] Thread-safe implementation
- [x] Integration with existing tools
- [x] Comprehensive test coverage

### Phase 6.3: Security Features (COMPLETE)

- [x] Input validation framework
- [x] Rate limiting (token bucket)
- [x] Audit logging (privacy-safe)
- [x] Security configuration system
- [x] Tool access control
- [x] Integration tests

### Priority 2: High-Value Tools (Next)

- [ ] TechnologyDatabaseTool - Technology specs and costs
- [ ] WeatherDataTool - Weather and climate data
- [ ] BuildingAnalysisTool - Building envelope analysis
- [ ] HVACAnalysisTool - HVAC system analysis

### Priority 3: Specialized Tools

- [ ] SolarAnalysisTool - Solar PV analysis
- [ ] BatteryStorageTool - Battery storage optimization
- [ ] EVChargingTool - EV charging infrastructure
- [ ] IndustryBenchmarkTool - Industry-specific benchmarks

---

## Migration Guide

### For Agent Developers

If your agent has duplicate implementations, migrate to shared tools:

**Before (Duplicate Code)**:

```python
class MyAgent(BaseAgent):
    def calculate_npv(self, capital_cost, annual_savings, ...):
        # 50+ lines of NPV calculation
        cash_flows = []
        for year in range(lifetime_years):
            # ...
        npv = sum(cf / (1 + discount_rate)**t for t, cf in enumerate(cash_flows))
        return npv
```

**After (Shared Tool)**:

```python
from greenlang.agents.tools import FinancialMetricsTool

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.financial_tool = FinancialMetricsTool()

    def calculate_npv(self, capital_cost, annual_savings, ...):
        result = self.financial_tool.execute(
            capital_cost=capital_cost,
            annual_savings=annual_savings,
            ...
        )
        return result.data["npv"]
```

**Benefits**:
- 50+ lines → 5 lines
- Fully tested implementation
- Complete citation tracking
- Consistent calculations across agents

---

## Support

For questions or issues:

1. Check this documentation
2. Review test files for examples
3. Check tool docstrings
4. Contact GreenLang Framework Team

---

## Version History

- **v1.0.0** (October 2025)
  - Initial release with emissions tools
  - ToolRegistry and auto-discovery
  - Phase 2 agent integration

- **v1.1.0** (October 2025 - Phase 6 Priority 1)
  - FinancialMetricsTool
  - GridIntegrationTool
  - RegionalEmissionFactorTool
  - CalculateScopeEmissionsTool
  - Comprehensive test coverage
  - Production-ready documentation

---

## License

Copyright 2025 GreenLang Framework Team. All rights reserved.
