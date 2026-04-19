# GL-017 CONDENSYNC Core Files Reference

## Quick Reference Guide

**Version:** 1.0.0
**Last Updated:** December 2025

---

## File Overview

| File | Lines | Purpose |
|------|-------|---------|
| `condenser_optimization_agent.py` | 1,858 | Main orchestrator and data models |
| `tools.py` | 2,768 | Deterministic calculation tools |
| `config.py` | 1,266 | Configuration models (Pydantic) |
| `gl.yaml` | 1,895 | GreenLang agent specification |
| `pack.yaml` | 1,580 | Package configuration |
| `example_usage.py` | 1,137 | Usage examples and demos |
| `run.json` | 570 | Runtime configuration |
| `.gitlab-ci.yml` | 349 | CI/CD pipeline |
| `docker-compose.yml` | 208 | Local development stack |
| `__init__.py` | 96 | Package initialization |

**Total:** ~11,727 lines of code

---

## Core Source Files

### condenser_optimization_agent.py (1,858 lines)

**Purpose:** Main agent orchestrator implementing the `CondenserOptimizationAgent` class.

**Key Components:**

| Component | Lines | Description |
|-----------|-------|-------------|
| Data Classes | 1-520 | `CoolingWaterData`, `VacuumData`, `CondensateData`, `HeatTransferData`, `FoulingAssessment`, `OptimizationResult`, `CondenserPerformanceResult` |
| Orchestrator | 521-1000 | `CondenserOptimizationAgent` class with `orchestrate()` and `execute()` methods |
| Analysis Methods | 1001-1450 | `analyze_condenser_performance()`, `optimize_vacuum_pressure()`, `optimize_cooling_water_flow()` |
| Detection Methods | 1451-1650 | `detect_air_inleakage()`, `predict_fouling()`, `recommend_tube_cleaning()` |
| Integration Methods | 1651-1858 | `integrate_cooling_tower()`, `coordinate_with_turbine()`, data gathering methods |

**Key Classes:**

```python
class CondenserOptimizationAgent(BaseOrchestrator):
    """Main orchestrator for condenser optimization."""

    async def execute(self) -> CondenserPerformanceResult:
        """Execute main optimization workflow."""

@dataclass
class CondenserPerformanceResult:
    """Complete analysis result with provenance."""
    performance_score: float
    provenance_hash: str
```

---

### tools.py (2,768 lines)

**Purpose:** Deterministic calculation tools with zero-hallucination guarantees.

**Key Components:**

| Component | Lines | Description |
|-----------|-------|-------------|
| Enumerations | 1-90 | `TubeMaterial`, `CondenserType`, `LeakageSeverity`, `CleaningMethod`, `AirEjectorType` |
| Data Classes | 91-230 | Result structures for each tool |
| Constants | 231-285 | Reference data (thermal conductivity, fouling factors, saturation temps) |
| Tool Definitions | 286-1094 | JSON schemas for 10 tools |
| Tool Executor | 1095-2768 | `CondenserToolExecutor` class with calculation implementations |

**10 Tools Implemented:**

1. `analyze_condenser_performance` - Comprehensive performance analysis
2. `calculate_heat_transfer_coefficient` - U-value, LMTD, TTD
3. `optimize_vacuum_pressure` - Vacuum setpoint optimization
4. `detect_air_inleakage` - Leak detection and severity assessment
5. `calculate_fouling_factor` - HEI fouling calculations
6. `predict_tube_cleaning_schedule` - Predictive maintenance
7. `optimize_cooling_water_flow` - CW pump optimization
8. `generate_performance_report` - Report generation
9. `calculate_condenser_duty` - Heat duty calculation
10. `assess_condenser_health` - Health scoring algorithm

**Key Class:**

```python
class CondenserToolExecutor:
    """Execute deterministic calculation tools."""

    async def execute_tool(self, tool_name: str, parameters: Dict) -> Dict:
        """Execute a tool by name with provenance tracking."""
```

---

### config.py (1,266 lines)

**Purpose:** Pydantic configuration models with validation.

**Key Components:**

| Component | Lines | Description |
|-----------|-------|-------------|
| Enumerations | 1-115 | `CondenserType`, `CoolingSystemType`, `TubePattern`, `CleaningMethod`, `FoulingType`, `TubeMaterial`, `VacuumPumpType`, `AnalyzerType` |
| CondenserConfiguration | 116-280 | Condenser design parameters |
| CoolingWaterConfig | 281-380 | CW system configuration |
| VacuumSystemConfig | 381-470 | Vacuum equipment configuration |
| TubeConfiguration | 471-580 | Tube geometry and materials |
| WaterQualityLimits | 581-680 | Water chemistry limits |
| PerformanceTargets | 681-775 | KPI target values |
| AlertThresholds | 776-870 | Warning/critical thresholds |
| SCADAIntegration | 871-970 | SCADA connection settings |
| CoolingTowerIntegration | 971-1030 | CT coordination |
| TurbineCoordination | 1031-1105 | Turbine backpressure |
| AgentConfiguration | 1106-1266 | Master configuration |

**Key Classes:**

```python
class AgentConfiguration(BaseModel):
    """Complete agent configuration."""
    condensers: List[CondenserConfiguration]
    cooling_water_config: CoolingWaterConfig
    vacuum_system_config: VacuumSystemConfig
    scada_integration: SCADAIntegration
```

---

### gl.yaml (1,895 lines)

**Purpose:** GreenLang agent specification defining runtime, data sources, monitoring, alerts, security, and deployment.

**Key Sections:**

| Section | Lines | Description |
|---------|-------|-------------|
| Metadata | 1-35 | Agent ID, codename, labels |
| Runtime | 36-95 | Python runtime, deterministic config |
| AI Config | 96-135 | Claude model settings, prohibited operations |
| Data Sources | 136-400 | OPC-UA, Modbus, REST API connections |
| Data Sinks | 401-550 | CMMS, databases, event bus |
| Monitoring | 551-790 | Prometheus metrics, Grafana dashboards |
| Alerts | 791-1010 | Alert rules and channels |
| Security | 1011-1200 | Auth, RBAC, encryption |
| Deployment | 1201-1360 | Kubernetes resources |
| Compliance | 1361-1500 | HEI, ASME standards |
| Calculations | 1501-1765 | Formula definitions |
| Data Flows | 1766-1895 | Processing pipelines |

---

### example_usage.py (1,137 lines)

**Purpose:** Demonstration code and usage examples.

**Key Sections:**

| Section | Description |
|---------|-------------|
| Basic Usage | Initialize agent with default config |
| Tool Execution | Call individual tools |
| Full Workflow | Complete optimization cycle |
| SCADA Integration | Connect to OPC-UA server |
| Report Generation | Generate PDF/Excel reports |
| Alert Configuration | Set up notifications |
| Testing | Unit test examples |

---

## Configuration Files

### pack.yaml (1,580 lines)

**Purpose:** Package metadata, dependencies, and build configuration.

**Key Sections:**
- Package identification
- Python dependencies
- Build targets
- Test configuration
- Documentation settings

### run.json (570 lines)

**Purpose:** Runtime configuration for deployment.

**Key Fields:**
- Environment variables
- Service endpoints
- Resource limits
- Feature flags

### docker-compose.yml (208 lines)

**Purpose:** Local development stack.

**Services:**
- `condensync` - Main agent
- `postgres` - Database
- `redis` - Cache
- `influxdb` - Time series
- `kafka` - Message bus

### .gitlab-ci.yml (349 lines)

**Purpose:** CI/CD pipeline stages.

**Stages:**
- `lint` - Code quality
- `test` - Unit/integration tests
- `build` - Docker image
- `deploy` - Kubernetes deployment

---

## Directory Structure

```
GL-017/
|-- condenser_optimization_agent.py  # Main orchestrator (1,858 lines)
|-- tools.py                         # Calculation tools (2,768 lines)
|-- config.py                        # Configuration models (1,266 lines)
|-- gl.yaml                          # Agent specification (1,895 lines)
|-- pack.yaml                        # Package config (1,580 lines)
|-- example_usage.py                 # Usage examples (1,137 lines)
|-- run.json                         # Runtime config (570 lines)
|-- __init__.py                      # Package init (96 lines)
|-- Dockerfile                       # Container build
|-- docker-compose.yml               # Dev stack (208 lines)
|-- Makefile                         # Build commands
|-- .gitlab-ci.yml                   # CI/CD (349 lines)
|
|-- calculators/                     # Domain calculators
|   |-- __init__.py
|   |-- heat_transfer_calculator.py
|   |-- vacuum_calculator.py
|   |-- fouling_calculator.py
|   |-- efficiency_calculator.py
|   |-- provenance.py
|
|-- integrations/                    # External connectors
|   |-- __init__.py
|   |-- scada_integration.py
|   |-- cooling_tower_integration.py
|   |-- dcs_integration.py
|   |-- historian_integration.py
|   |-- cmms_integration.py
|   |-- message_bus_integration.py
|
|-- deployment/                      # Kubernetes manifests
|   |-- deployment.yaml
|   |-- service.yaml
|   |-- configmap.yaml
|   |-- secret.yaml
|   |-- hpa.yaml
|   |-- networkpolicy.yaml
|   |-- ingress.yaml
|   |-- pvc.yaml
|   |-- serviceaccount.yaml
|   |-- kustomization.yaml
|
|-- tests/                           # Test suites
|   |-- __init__.py
|   |-- conftest.py
|   |-- pytest.ini
|   |-- test_condenser_optimization_agent.py
|   |-- test_tools.py
|   |-- test_calculators.py
|   |-- test_integrations.py
|   |-- determinism/
|   |   |-- test_determinism.py
|   |-- performance/
|       |-- test_performance.py
|
|-- monitoring/                      # Observability configs
|-- runbooks/                        # Operational runbooks
|-- security/                        # Security policies
|-- sbom/                            # Software bill of materials
```

---

## Import Quick Reference

```python
# Main agent
from greenlang.GL_017 import CondenserOptimizationAgent
from greenlang.GL_017.condenser_optimization_agent import (
    CondenserPerformanceResult,
    CoolingWaterData,
    VacuumData,
    HeatTransferData,
    FoulingAssessment,
    OptimizationResult
)

# Tools
from greenlang.GL_017.tools import (
    CondenserToolExecutor,
    CONDENSER_TOOLS,
    CondenserPerformanceMetrics,
    HeatTransferResult,
    VacuumOptimizationResult,
    AirInleakageAssessment,
    FoulingAnalysisResult,
    CleaningScheduleResult,
    CoolingWaterOptimizationResult
)

# Configuration
from greenlang.GL_017.config import (
    AgentConfiguration,
    CondenserConfiguration,
    CoolingWaterConfig,
    VacuumSystemConfig,
    TubeConfiguration,
    WaterQualityLimits,
    PerformanceTargets,
    AlertThresholds,
    SCADAIntegration,
    CoolingTowerIntegration,
    TurbineCoordination
)

# Enums
from greenlang.GL_017.config import (
    CondenserType,
    CoolingSystemType,
    TubePattern,
    CleaningMethod,
    FoulingType,
    TubeMaterial,
    VacuumPumpType
)
```

---

## Related Documentation

- [README.md](README.md) - Comprehensive documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [TOOLS_README.md](TOOLS_README.md) - Tool documentation
- [deployment/README.md](deployment/README.md) - Deployment guide

---

*GL-017 CONDENSYNC Core Files Reference - Version 1.0.0*
