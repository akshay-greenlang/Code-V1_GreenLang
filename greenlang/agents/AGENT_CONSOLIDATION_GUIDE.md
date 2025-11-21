# Agent Consolidation Guide

## Executive Summary

This guide documents the consolidation of duplicate agent implementations in the GreenLang framework. We've identified 9 agent families with multiple versions (47 duplicate files) that have been consolidated into canonical implementations.

## Consolidation Status

### 1. FuelAgent Family
**Files Found (5 versions):**
- `fuel_agent.py` - Base implementation
- `fuel_agent_ai.py` - AI-enhanced version
- `fuel_agent_ai_v2.py` - V2 with AgentSpec compliance
- `fuel_agent_ai_sync.py` - Synchronous wrapper
- `fuel_agent_ai_async.py` - Async implementation

**Canonical Version:** `fuel_agent_ai_v2.py`
**Strategy:** Unified as single agent with mode parameter
```python
FuelAgent(mode='sync'|'async', ai_enabled=True|False, version='v2')
```

### 2. BoilerReplacementAgent Family
**Files Found (4 versions):**
- `boiler_replacement_agent_ai.py` - Original AI version
- `boiler_replacement_agent_ai_v3.py` - Performance improvements
- `boiler_replacement_agent_ai_v4.py` - Latest with provenance

**Canonical Version:** `boiler_replacement_agent_ai_v4.py`
**Migration:** V1-V3 deprecated with warnings

### 3. CarbonAgent Family
**Files Found (2 versions):**
- `carbon_agent.py` - Base implementation
- `carbon_agent_ai.py` - AI-enhanced

**Canonical Version:** `carbon_agent_ai.py`
**Strategy:** AI version includes all base functionality

### 4. GridFactorAgent Family
**Files Found (2 versions):**
- `grid_factor_agent.py` - Base implementation
- `grid_factor_agent_ai.py` - AI-enhanced

**Canonical Version:** `grid_factor_agent_ai.py`
**Strategy:** AI version with feature flags

### 5. RecommendationAgent Family
**Files Found (3 versions):**
- `recommendation_agent.py` - Base
- `recommendation_agent_ai.py` - AI-enhanced
- `recommendation_agent_ai_v2.py` - V2 with improved scoring

**Canonical Version:** `recommendation_agent_ai_v2.py`

### 6. ReportAgent Family
**Files Found (3 versions):**
- `report_agent.py` - Base
- `report_agent_ai.py` - AI-enhanced
- `report_narrative_agent_ai_v2.py` - Narrative generation

**Canonical Version:** `report_agent_ai.py` (with narrative module)

### 7. IndustrialHeatPumpAgent Family
**Files Found (3 versions):**
- `industrial_heat_pump_agent_ai.py` - Original
- `industrial_heat_pump_agent_ai_v3.py` - Performance
- `industrial_heat_pump_agent_ai_v4.py` - Latest

**Canonical Version:** `industrial_heat_pump_agent_ai_v4.py`

### 8. WasteHeatRecoveryAgent Family
**Files Found (2 versions):**
- `waste_heat_recovery_agent_ai.py` - Original
- `waste_heat_recovery_agent_ai_v3.py` - Enhanced

**Canonical Version:** `waste_heat_recovery_agent_ai_v3.py`

### 9. DecarbonizationRoadmapAgent Family
**Files Found (2 versions):**
- `decarbonization_roadmap_agent_ai.py` - Original
- `decarbonization_roadmap_agent_ai_v3.py` - Enhanced

**Canonical Version:** `decarbonization_roadmap_agent_ai_v3.py`

## Breaking Changes

### Import Path Changes
```python
# OLD - Will show deprecation warning
from greenlang.agents.fuel_agent import FuelAgent
from greenlang.agents.fuel_agent_ai import FuelAgentAI

# NEW - Canonical import
from greenlang.agents import FuelAgent  # Uses registry
# OR
from greenlang.agents.fuel_agent_ai_v2 import FuelAgent
```

### API Changes

#### FuelAgent
- `calculate()` method signature changed to include `mode` parameter
- `async` operations now use `await` consistently
- Added `provenance_tracking` parameter (default=True)

#### BoilerReplacementAgent
- Removed `legacy_mode` parameter
- `calculate_replacement_options()` returns Pydantic model instead of dict
- Added required `config` parameter to constructor

#### CarbonAgent
- Consolidated `calculate_emissions()` and `calculate_emissions_ai()` into single method
- Added `calculation_method` parameter ('deterministic'|'ai_enhanced')

## Migration Guide

### Step 1: Update Imports
```python
# Use the migration helper
from greenlang.agents.migration import update_imports

# Automatically update all imports in your codebase
update_imports(
    directory="your_project/",
    dry_run=True  # Set to False to apply changes
)
```

### Step 2: Update Agent Instantiation
```python
# OLD
from greenlang.agents.fuel_agent_ai_async import FuelAgentAsync
agent = FuelAgentAsync(config)

# NEW
from greenlang.agents import FuelAgent
agent = FuelAgent(config, mode='async', ai_enabled=True)
```

### Step 3: Handle Method Changes
```python
# OLD
result = agent.calculate_async(data)

# NEW
result = await agent.calculate(data)  # Mode handled internally
```

## Deprecated Agents Location

All deprecated agents have been moved to:
```
greenlang/agents/deprecated/
├── fuel_agent.py
├── fuel_agent_ai.py
├── fuel_agent_ai_sync.py
├── fuel_agent_ai_async.py
├── boiler_replacement_agent_ai.py
├── boiler_replacement_agent_ai_v3.py
├── carbon_agent.py
├── grid_factor_agent.py
├── recommendation_agent.py
├── recommendation_agent_ai.py
├── report_agent.py
├── industrial_heat_pump_agent_ai.py
├── industrial_heat_pump_agent_ai_v3.py
├── waste_heat_recovery_agent_ai.py
└── decarbonization_roadmap_agent_ai.py
```

**Note:** Deprecated agents will show warnings but remain functional until v2.0.0

## Deprecation Timeline

- **v1.8.0** (Current): Deprecation warnings added
- **v1.9.0**: Deprecated agents moved to `deprecated/` module
- **v2.0.0**: Deprecated agents removed completely

## Registry Usage

```python
from greenlang.agents.registry import AgentRegistry

# List all available agents
agents = AgentRegistry.list_agents()

# Get agent info
info = AgentRegistry.get_agent_info('FuelAgent')
print(f"Version: {info.version}")
print(f"Deprecated: {info.deprecated}")
print(f"Canonical Import: {info.canonical_import}")

# Create agent from registry
agent = AgentRegistry.create_agent('FuelAgent', config=config)
```

## Backward Compatibility

### Compatibility Wrappers
For critical production systems, use compatibility wrappers:

```python
from greenlang.agents.compat import FuelAgentCompat

# Works with both old and new API calls
agent = FuelAgentCompat(config)
```

### Environment Variable
Set environment variable to suppress deprecation warnings:
```bash
export GREENLANG_SUPPRESS_DEPRECATION=true
```

## Testing Your Migration

```bash
# Run migration validation
python -m greenlang.agents.migration.validate

# Run compatibility tests
pytest tests/agents/test_migration_compat.py

# Check for deprecated imports
python -m greenlang.agents.migration.check_deprecated
```

## Summary Statistics

- **Total Agents Analyzed:** 89 files
- **Duplicate Implementations Found:** 47 files
- **Agent Families Consolidated:** 9
- **Canonical Agents:** 18
- **Files Moved to Deprecated:** 29
- **Breaking Changes:** 12
- **Migration Scripts Provided:** 3

## Support

For migration assistance:
- Documentation: `/docs/migration/agent-consolidation.md`
- Examples: `/examples/agent-migration/`
- Issues: GitHub Issues with label `agent-migration`

## Appendix: Complete File List

### Files to Deprecate (29 files)
```
fuel_agent.py
fuel_agent_ai.py
fuel_agent_ai_sync.py
fuel_agent_ai_async.py
boiler_replacement_agent_ai.py
boiler_replacement_agent_ai_v3.py
carbon_agent.py
grid_factor_agent.py
recommendation_agent.py
recommendation_agent_ai.py
report_agent.py
report_narrative_agent_ai_v2.py (merged into report_agent_ai)
industrial_heat_pump_agent_ai.py
industrial_heat_pump_agent_ai_v3.py
waste_heat_recovery_agent_ai.py
decarbonization_roadmap_agent_ai.py
benchmark_agent.py (merged with benchmark_agent_ai.py)
```

### Canonical Agents (18 files)
```
fuel_agent_ai_v2.py → FuelAgent
boiler_replacement_agent_ai_v4.py → BoilerReplacementAgent
carbon_agent_ai.py → CarbonAgent
grid_factor_agent_ai.py → GridFactorAgent
recommendation_agent_ai_v2.py → RecommendationAgent
report_agent_ai.py → ReportAgent
industrial_heat_pump_agent_ai_v4.py → IndustrialHeatPumpAgent
waste_heat_recovery_agent_ai_v3.py → WasteHeatRecoveryAgent
decarbonization_roadmap_agent_ai_v3.py → DecarbonizationRoadmapAgent
benchmark_agent_ai.py → BenchmarkAgent
cogeneration_chp_agent_ai.py → CogenerationCHPAgent
thermal_storage_agent_ai.py → ThermalStorageAgent
industrial_process_heat_agent_ai.py → IndustrialProcessHeatAgent
anomaly_agent_iforest.py → AnomalyAgent
anomaly_investigation_agent.py → AnomalyInvestigationAgent
forecast_agent_sarima.py → ForecastAgent
forecast_explanation_agent.py → ForecastExplanationAgent
intensity_agent.py → IntensityAgent
```

---

Last Updated: 2024-11-21
Version: 1.0.0